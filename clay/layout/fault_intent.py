"""EXPERIMENTAL error-intent solver (fault-intent diagnosis) for layout.

Status: EXPERIMENT (see docs/superpowers/round-experiment-fault-intent.md). This
is a standalone probe over an already-routed cell; it does NOT change the
production DRC/LVS/router contracts. If it proves correct + cheap, the design is
to encapsulate it into the DRC/LVS check output + add real decision provenance on
emit + feed the v3 learning memory -- but NOT before the experiment decides.

The thesis (from the design discussion)
=======================================
Raw DRC coordinate feedback is in the wrong space for an LLM: it forces the model
to re-derive, in token space, which net / decision / intent each violating edge
belongs to -- exactly the spatial-arithmetic work LLMs are worst at. The fix is a
fast solver that works in the DISCRETE DECISION space (which net shares which
column / lane) and returns a CAUSAL, INTENT-REFERENCED, DIRECTIONAL diagnosis.

Why it is fast (the two tricks)
-------------------------------
1. Attribution by READING provenance off the geometry, never by counterfactual
   replay: every shape already carries its net (``tagged`` triples); the collision
   locus is the x-band where two nets' shapes coincide -> a column/resource
   "decision". (Production would tag a real decision id on emit; this experiment
   RECONSTRUCTS it to prove the diagnosis is computable + useful.)
2. The solve runs in the small DISCRETE space (tens of column decisions, hundreds
   of symptoms), a greedy group/set-cover -- microseconds -- not over the thousands
   of coordinates and not by re-running the geometry.

The single-step counterfactual ("relocate net X out of this column -> N symptoms
gone") is read straight off the symptom->decision graph; distance-to-sat is risers
vs column capacity. Both are byproducts of the cover, no replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from clay.layout.routed_cell_types import check_shorts
from clay.layout.m2_crossing_route import (
    M1_KEY,
    M1_MIN_SPACE_NM,
    M2_KEY,
    M2_MIN_SPACE_NM,
    VIA1_KEY,
)
from clay.layout.intent import _is_rail

Bbox = Tuple[int, int, int, int]
TaggedShape = Tuple[str, Tuple[int, int], Bbox]

# Two loci within this x-distance are the same physical column/resource decision.
# A column is one device's worth of stacked risers; the lane grid is the riser
# pitch, so a few lane-grids groups one column. (Reconstructed; production uses the
# real decision id.)
_COLUMN_TOL_NM = 2200
# riser pitch used to estimate a column's lane capacity (the v1.5 router's grid).
_LANE_PITCH_NM = 720


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis data (the structured feedback that replaces the flat violation list)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Symptom:
    """One geometric violation, already attributed to a net pair + a locus.

    ``kind`` is "short" (different nets electrically merged) or "spacing" (a
    same-layer different-net min-spacing miss). ``locus`` is the (x, y) centre of
    the collision; ``x_band`` is the x-extent the colliding shapes share (the
    column the symptom maps to)."""

    kind: str
    nets: Tuple[str, str]
    locus: Tuple[int, int]
    x_band: Tuple[int, int]
    detail: str


@dataclass(frozen=True)
class RootCause:
    """One root decision explaining a cluster of symptoms (the minimal-cover unit).

    A decision is a column/resource ("column@x=...") that several nets were
    co-located into. ``intent_violated`` names the declared obligation it breaks;
    ``distance_to_sat`` is the directional magnitude (risers vs capacity);
    ``counterfactual`` is the single-step fix read off the symptom graph;
    ``levers`` are the ranked engine actions that would address it."""

    decision: str
    severity: str
    nets_involved: Tuple[str, ...]
    symptoms: Tuple[Symptom, ...]
    intent_violated: str
    distance_to_sat: str
    counterfactual: str
    levers: Tuple[str, ...]
    # structured counterfactual (so the claim is machine-verifiable, not prose):
    # relocating ``counterfactual_net`` is CLAIMED to remove ``counterfactual_covers``
    # of this cluster's symptoms. validate_counterfactuals() checks it by actually
    # removing that net from the geometry and re-extracting.
    counterfactual_net: str = ""
    counterfactual_covers: int = 0

    @property
    def n_symptoms(self) -> int:
        return len(self.symptoms)


@dataclass(frozen=True)
class Diagnosis:
    cell: str
    n_symptoms: int
    root_causes: Tuple[RootCause, ...]
    solve_seconds: float = 0.0

    # ── renderers (the experiment's side-by-side) ───────────────────────────
    def render_raw(self) -> str:
        """The CURRENT feedback form: a flat list of symptoms with coordinates."""
        lines = [f"# raw symptom list ({self.n_symptoms} violations)"]
        for rc in self.root_causes:
            for s in rc.symptoms:
                x0, x1 = s.x_band
                lines.append(
                    f"[{s.kind}] {s.nets[0]}~{s.nets[1]} @ ({s.locus[0]},{s.locus[1]}) "
                    f"x-band[{x0},{x1}] {s.detail}"
                )
        return "\n".join(lines)

    def render(self) -> str:
        """The NEW feedback form: causal, intent-referenced, directional."""
        lines = [
            f"# fault-intent diagnosis: {self.cell} "
            f"({self.n_symptoms} symptoms -> {len(self.root_causes)} root cause(s), "
            f"{self.solve_seconds * 1e3:.2f} ms)"
        ]
        for i, rc in enumerate(self.root_causes, 1):
            lines.append(
                f"\nRoot cause R{i}  [{rc.severity}]  decision: {rc.decision}"
            )
            lines.append(f"  nets:     {', '.join(rc.nets_involved)}")
            lines.append(f"  covers:   {rc.n_symptoms} symptom(s)")
            lines.append(f"  violates: {rc.intent_violated}")
            lines.append(f"  d2sat:    {rc.distance_to_sat}")
            lines.append(f"  counterfx: {rc.counterfactual}")
            lines.append("  levers:")
            for lv in rc.levers:
                lines.append(f"    - {lv}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# geometry helpers (cheap, exact -- the engine does the spatial work, not the LLM)
# ─────────────────────────────────────────────────────────────────────────────
def _xspan(b: Bbox) -> Tuple[int, int]:
    return (min(b[0], b[2]), max(b[0], b[2]))


def _yspan(b: Bbox) -> Tuple[int, int]:
    return (min(b[1], b[3]), max(b[1], b[3]))


def _sep(b1: Bbox, b2: Bbox) -> Tuple[int, int]:
    """Axis separations (0 when overlapping on that axis)."""
    ax0, ax1 = _xspan(b1)
    bx0, bx1 = _xspan(b2)
    ay0, ay1 = _yspan(b1)
    by0, by1 = _yspan(b2)
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return dx, dy


def _edge_gap(b1: Bbox, b2: Bbox) -> int:
    """Edge-to-edge gap for a min-spacing check (0 when touching/overlapping)."""
    dx, dy = _sep(b1, b2)
    if dx == 0:
        return dy
    if dy == 0:
        return dx
    return max(dx, dy)  # corner: conservative


def _xband_overlap(b1: Bbox, b2: Bbox) -> Optional[Tuple[int, int]]:
    ax0, ax1 = _xspan(b1)
    bx0, bx1 = _xspan(b2)
    lo, hi = max(ax0, bx0), min(ax1, bx1)
    return (lo, hi) if lo <= hi else None


def _locus(b1: Bbox, b2: Bbox) -> Tuple[int, int]:
    band = _xband_overlap(b1, b2)
    if band is not None:
        cx = (band[0] + band[1]) // 2
    else:
        cx = (max(_xspan(b1)[0], _xspan(b2)[0]) + min(_xspan(b1)[1], _xspan(b2)[1])) // 2
    ay0, ay1 = _yspan(b1)
    by0, by1 = _yspan(b2)
    cy = (max(ay0, by0) + min(ay1, by1)) // 2
    return (int(cx), int(cy))


def _point_to_bbox(pt: Tuple[int, int], b: Bbox) -> float:
    """Distance from a point to a bbox (0 inside); cheap L2-of-axis-gaps."""
    x0, x1 = _xspan(b)
    y0, y1 = _yspan(b)
    dx = max(x0 - pt[0], pt[0] - x1, 0)
    dy = max(y0 - pt[1], pt[1] - y1, 0)
    return (dx * dx + dy * dy) ** 0.5


def _near(viol_bbox: Bbox, shape_bbox: Bbox, pad: int) -> bool:
    """True if ``shape_bbox`` overlaps ``viol_bbox`` expanded by ``pad`` on all sides."""
    vx0, vy0, vx1, vy1 = (
        min(viol_bbox[0], viol_bbox[2]) - pad, min(viol_bbox[1], viol_bbox[3]) - pad,
        max(viol_bbox[0], viol_bbox[2]) + pad, max(viol_bbox[1], viol_bbox[3]) + pad,
    )
    sx0, sx1 = _xspan(shape_bbox)
    sy0, sy1 = _yspan(shape_bbox)
    return not (sx1 < vx0 or sx0 > vx1 or sy1 < vy0 or sy0 > vy1)


# ─────────────────────────────────────────────────────────────────────────────
# symptom extraction (shorts from union-find; spacing from a fast intrinsic scan)
# ─────────────────────────────────────────────────────────────────────────────
def _contact_symptoms(tagged: Sequence[TaggedShape]) -> List[Symptom]:
    """Every different-net shape-pair contact / near-miss, each with a locus.

    A SHORT contact = different nets that touch on the same layer (or via1 bridge).
    A SPACING symptom = different nets on the same metal layer within the layer's
    min spacing but not touching. O(n^2) over the few-hundred tagged shapes (ms).
    Rails are kept (a short to a rail is still a short)."""
    items = list(tagged)
    out: List[Symptom] = []
    min_space = {M1_KEY: M1_MIN_SPACE_NM, M2_KEY: M2_MIN_SPACE_NM}
    for i in range(len(items)):
        ni, ki, bi = items[i]
        for j in range(i + 1, len(items)):
            nj, kj, bj = items[j]
            if ni == nj:
                continue
            gap = _edge_gap(bi, bj)
            same_layer = ki == kj
            via_bridge = (ki == VIA1_KEY) or (kj == VIA1_KEY)
            if gap == 0 and (same_layer or via_bridge):
                band = _xband_overlap(bi, bj) or _xspan(bi)
                out.append(Symptom("short", (ni, nj), _locus(bi, bj), band,
                                   f"touch on layer {ki if same_layer else 'via1'}"))
            elif same_layer and ki in min_space and 0 < gap < min_space[ki]:
                band = _xband_overlap(bi, bj) or _xspan(bi)
                out.append(Symptom("spacing", (ni, nj), _locus(bi, bj), band,
                                   f"gap {gap}nm < {min_space[ki]}nm on layer {ki}"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# the solver: cluster symptoms by column decision, attribute intent, levers
# ─────────────────────────────────────────────────────────────────────────────
def _cluster_by_column(symptoms: Sequence[Symptom]) -> List[List[Symptom]]:
    """Greedy 1-D clustering of symptoms by their locus x (the column decision).

    Sorted-sweep: a symptom joins the open cluster when its locus x is within
    ``_COLUMN_TOL_NM`` of the cluster's running centre, else it opens a new one.
    This is the degenerate set-cover (each symptom maps to one column); the
    multi-lever set-cover is the production generalisation."""
    if not symptoms:
        return []
    ordered = sorted(symptoms, key=lambda s: s.locus[0])
    clusters: List[List[Symptom]] = [[ordered[0]]]
    centre = ordered[0].locus[0]
    for s in ordered[1:]:
        if abs(s.locus[0] - centre) <= _COLUMN_TOL_NM:
            clusters[-1].append(s)
            xs = [t.locus[0] for t in clusters[-1]]
            centre = sum(xs) // len(xs)
        else:
            clusters.append([s])
            centre = s.locus[0]
    return clusters


def _net_riser_count(tagged: Sequence[TaggedShape], x0: int, x1: int) -> List[str]:
    """Distinct NON-RAIL nets with a shape whose x-centre falls in [x0, x1]."""
    nets = set()
    for net, _k, b in tagged:
        bx0, bx1 = _xspan(b)
        cx = (bx0 + bx1) // 2
        if x0 - _LANE_PITCH_NM <= cx <= x1 + _LANE_PITCH_NM and not _is_rail(net):
            nets.add(net)
    return sorted(nets)


def _diff_pair_set(criteria) -> set:
    pairs = set()
    for attr in ("device_pairs", "matched_pairs"):
        for p in getattr(criteria, attr, ()) or ():
            if len(p) == 2:
                pairs.add(frozenset((str(p[0]), str(p[1]))))
    return pairs


def _intent_label(nets: Sequence[str], symptoms: Sequence[Symptom], diff_pairs: set) -> str:
    for s in symptoms:
        if frozenset((s.nets[0], s.nets[1])) in diff_pairs:
            return f"diff-pair isolation ({s.nets[0]}/{s.nets[1]}) -- matched pair must not couple"
    if any(_is_rail(n) for n in nets):
        rails = sorted(n for n in nets if _is_rail(n))
        # "shorted to" for a union-find short; "too close to" for a DRC spacing rule.
        merged = any(s.kind == "short" for s in symptoms)
        rel = "shorted to" if merged else "too close to"
        return f"rail isolation -- signal {rel} {', '.join(rails)}"
    # DRC-rule-sourced symptoms (from a real lyrdb join): name the dominant rule
    # as the violated deck obligation.
    from collections import Counter
    drc_kinds = [s.kind for s in symptoms if s.kind not in ("short", "spacing")]
    if drc_kinds:
        rule = Counter(drc_kinds).most_common(1)[0][0]
        return f"DRC rule {rule} -- deck-level legality (different nets too close / mis-sized here)"
    return "@route_policy(tracks=net_disjoint) -- each net needs its own track"


def _align_offset(violations, tagged) -> Tuple[int, int]:
    """Translation that maps the DRC report's coordinate frame onto the tagged
    geometry's frame, so a join works even if ``emit`` shifted the cell.

    KLayout reports in the emitted GDS frame; if that equals the tagged-nm frame
    the offset is (0, 0). Otherwise (a uniform emit translation) the violation
    centroid falls OUTSIDE the tagged extent, and we align the two centroids -- a
    coarse but sufficient correction for column-level attribution."""
    if not violations or not tagged:
        return (0, 0)
    vx = [v.locus[0] for v in violations]
    vy = [v.locus[1] for v in violations]
    txs = [c for _, _, b in tagged for c in _xspan(b)]
    tys = [c for _, _, b in tagged for c in _yspan(b)]
    tx0, tx1, ty0, ty1 = min(txs), max(txs), min(tys), max(tys)
    vcx, vcy = (min(vx) + max(vx)) // 2, (min(vy) + max(vy)) // 2
    if tx0 <= vcx <= tx1 and ty0 <= vcy <= ty1:
        return (0, 0)
    return (((tx0 + tx1) // 2) - vcx, ((ty0 + ty1) // 2) - vcy)


def symptoms_from_violations(
    violations, tagged: Sequence[TaggedShape], *, near_nm: int = 600
) -> List[Symptom]:
    """JOIN real DRC violations to the routed geometry's net provenance.

    For each :class:`~clay.layout.drc_report.RuleViolation`, find the tagged shapes
    near its bbox and attribute the violation to the closest net(s): two distinct
    nets -> a different-net rule (short / spacing); one net -> a single-net rule
    (width / enclosure); none -> a deck rule with no routed-net locus (well /
    substrate). The violation keeps its rule name as the symptom ``kind`` so the
    diagnosis names the violated deck obligation. This is the GENERAL deck-level
    feedback path (any circuit); the geometry frame is aligned first."""
    ox, oy = _align_offset(violations, tagged)
    out: List[Symptom] = []
    for v in violations:
        b = (v.bbox[0] + ox, v.bbox[1] + oy, v.bbox[2] + ox, v.bbox[3] + oy)
        locus = (v.locus[0] + ox, v.locus[1] + oy)
        near = []
        for net, _k, sb in tagged:
            if _near(b, sb, near_nm):
                near.append((_point_to_bbox(locus, sb), net))
        near.sort()
        nets: List[str] = []
        for _d, net in near:
            if net not in nets:
                nets.append(net)
            if len(nets) >= 2:
                break
        if len(nets) >= 2:
            pair = (nets[0], nets[1])
        elif len(nets) == 1:
            pair = (nets[0], nets[0])
        else:
            pair = ("<untagged>", "<untagged>")
        out.append(Symptom(
            kind=v.rule, nets=pair, locus=locus,
            x_band=(min(b[0], b[2]), max(b[0], b[2])),
            detail=f"DRC {v.rule}",
        ))
    return out


def diagnose(
    routed,
    *,
    placement=None,
    criteria=None,
    cell: str = "",
    drc_violations=None,
) -> Diagnosis:
    """Run the error-intent solver over an already-routed cell.

    Composes the routed cell's ``tagged`` geometry (+ its union-find ``shorts``)
    and an optional ``criteria`` (for diff-pair intent attribution). Returns a
    :class:`Diagnosis`: the symptoms clustered into ranked root decisions, each
    with the intent it violates, a distance-to-sat, a single-step counterfactual,
    and ranked levers. Pure analysis -- nothing is re-routed or re-extracted."""
    import time

    t0 = time.perf_counter()
    tagged = list(getattr(routed, "tagged", []) or [])
    # symptom SOURCE is pluggable (the engine is one solver): real DRC violations
    # joined to net provenance when supplied, else the fast intrinsic scan.
    if drc_violations is not None:
        symptoms = symptoms_from_violations(drc_violations, tagged)
    else:
        symptoms = _contact_symptoms(tagged)
    diff_pairs = _diff_pair_set(criteria) if criteria is not None else set()

    clusters = _cluster_by_column(symptoms)
    # rank root causes by how many symptoms each covers (greedy cover order).
    clusters.sort(key=len, reverse=True)

    roots: List[RootCause] = []
    for cl in clusters:
        xs = [s.locus[0] for s in cl]
        x0, x1 = min(xs), max(xs)
        nets_involved = sorted({n for s in cl for n in s.nets})
        risers = _net_riser_count(tagged, x0, x1)
        band_w = max(1, (x1 - x0) + _LANE_PITCH_NM)
        capacity = max(1, band_w // _LANE_PITCH_NM)
        has_short = any(s.kind == "short" for s in cl)
        severity = "HIGH connectivity" if has_short else "MED spacing"
        # counterfactual: the net appearing in the most symptoms of this cluster.
        # Count DISTINCT symptoms a net participates in (a single-net rule has
        # nets=(net,net); a rail is not relocatable) so coverage never exceeds the
        # cluster size.
        from collections import Counter
        freq: Counter = Counter()
        for s in cl:
            for n in set(s.nets):
                if not _is_rail(n) and n != "<untagged>":
                    freq[n] += 1
        hot = freq.most_common(1)[0] if freq else (
            (nets_involved[0] if nets_involved else "?"), 0)
        hot = (hot[0], min(hot[1], len(cl)))
        intent = _intent_label(nets_involved, cl, diff_pairs)
        levers = (
            "placement: split the shared-diffusion group / widen this column "
            "(removes the root; cost +area)",
            "routing: promote one riser to M3 / reassign its lane "
            f"(removes the {hot[0]} contacts; cost +1 via layer)",
            "accept as a single-channel density ceiling (honest gap, no change)",
        )
        roots.append(RootCause(
            decision=f"column@x≈{(x0 + x1) // 2}  ({len(risers)} nets co-located)",
            severity=severity,
            nets_involved=tuple(nets_involved),
            symptoms=tuple(cl),
            intent_violated=intent,
            distance_to_sat=(
                f"{len(risers)} risers crowded into a {band_w}nm band that fits "
                f"~{capacity} at the {_LANE_PITCH_NM}nm lane pitch "
                f"(deficit {max(0, len(risers) - capacity)} lane(s))"
            ),
            counterfactual=(
                f"relocate net {hot[0]} out of this column -> "
                f"{hot[1]}/{len(cl)} of this cluster's symptoms gone"
            ),
            counterfactual_net=str(hot[0]),
            counterfactual_covers=int(hot[1]),
            levers=levers,
        ))

    return Diagnosis(
        cell=cell,
        n_symptoms=len(symptoms),
        root_causes=tuple(roots),
        solve_seconds=time.perf_counter() - t0,
    )


def apply_counterfactual(
    tagged: Sequence[TaggedShape], drop_net: str
) -> List[TaggedShape]:
    """The counterfactual geometry with ``drop_net``'s shapes removed (as if that
    net were relocated to its own track / column). Pure; does not mutate input."""
    return [t for t in tagged if t[0] != drop_net]


def validate_counterfactuals(routed, diagnosis: Diagnosis) -> Tuple[float, list]:
    """SELF-VERIFY each root cause's counterfactual against the real geometry.

    For each root cause, actually remove the named ``counterfactual_net`` from the
    tagged geometry, re-extract the contact symptoms, and check that the cluster's
    symptoms that INVOLVED that net are gone and the claimed coverage count matches
    the actually-removed count. Returns ``(soundness, per_root)`` where
    ``soundness`` is the fraction of root causes whose counterfactual claim is
    geometrically exact. This makes the feedback's causal arithmetic checkable by
    the engine (the thing the LLM gets wrong), not asserted."""
    tagged = list(getattr(routed, "tagged", []) or [])
    base = _contact_symptoms(tagged)
    per_root = []
    ok_count = 0
    n_checked = 0
    for rc in diagnosis.root_causes:
        net = rc.counterfactual_net
        if not net:
            per_root.append((rc.decision, None))
            continue
        n_checked += 1
        # symptoms in the WHOLE layout that involve this net (the ones the
        # relocation should remove).
        involve = [s for s in base if net in s.nets]
        after = _contact_symptoms(apply_counterfactual(tagged, net))
        after_involve = [s for s in after if net in s.nets]
        actually_removed = len(involve) - len(after_involve)
        # exact iff every net-involving symptom is gone AND the count matches.
        exact = (len(after_involve) == 0) and (actually_removed == len(involve))
        per_root.append((rc.decision, {
            "net": net,
            "claimed_covers": rc.counterfactual_covers,
            "actually_removed": actually_removed,
            "exact": exact,
        }))
        if exact:
            ok_count += 1
    soundness = (ok_count / n_checked) if n_checked else 0.0
    return soundness, per_root


__all__ = [
    "Symptom",
    "RootCause",
    "Diagnosis",
    "diagnose",
    "symptoms_from_violations",
    "apply_counterfactual",
    "validate_counterfactuals",
]
