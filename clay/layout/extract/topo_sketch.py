"""Design-only placement STRUCTURE + geometric SKETCH (leakage-remediation Phase 1).

THE CONTRACT THIS MODULE EXISTS FOR: placement structure (grouping, row
partition, within-row order, symmetry) must be DESIGN-sourced -- derived from
netlist topology (kind / W / L / M / nets) and declared intent -- and the
geometry solved at the target PDK's device sizes. Reference-layout coordinates
(``intent.device_bboxes`` as filled by ``correspond_to_gds``) feed VERIFICATION
only; nothing in this module may read them. The SEV-1 incident this remedies:
``cluster_bands`` / ``_band_orders`` / ``_stack_bands`` derived all of the above
from reference centroids (docs/audits/INCIDENT-placement-coordinate-leakage-
2026-06-10.html).

What the netlist DOES determine (derived here, mechanically):
  * mirror groups -- connected components of ``intent.mirror_pairs`` (one shared
    gate net per component by construction), with the diode-connected member
    (gate == drain) as the reference leg;
  * diff pairs -- ``intent.diff_pairs`` (shared source, distinct gates);
  * connectivity -- which units share nets (hypergraph clique weighting, so a
    many-device rail downweights itself: no name-based power detection).

What the netlist does NOT determine (GEOMETRY -- solved, with every
underdetermined STRUCTURAL choice made deterministically and recorded in
``DesignSketch.gaps``, never silently and never from the reference):
  * absolute positions, row Y, the left/right side of a symmetric twin, the
    member order inside a matched group.

The sketch is a ROUGH legal placement whose coordinates CARRY the derived
structure into the existing solve path (``relgraph.build_hierarchical_graph``
-> ``solve_hierarchical_nonoverlap``): rows become bands, within-row x-order
becomes the band order, all design-sourced. No second solver path is built.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_XY = Tuple[float, float]
_Bbox = Tuple[float, float, float, float]

# Sketch spacing knobs. The sketch is structural, not signoff geometry: the
# within-row gap merely keeps sketch bboxes disjoint; the row pitch must exceed
# relgraph's band-split threshold (max(3 * median-gap, 8 um) -- within-row
# centroid-y gaps are 0 here, so the threshold degenerates to the 8 um floor)
# so ``cluster_bands`` recovers exactly the sketch rows.
_SKETCH_DEV_GAP_UM: float = 1.0
_SKETCH_ROW_CHANNEL_UM: float = 2.0
_SKETCH_MIN_ROW_PITCH_UM: float = 9.0

# R8 in-row 2D packing: the sketch encodes a stacked (sub-row) member as a
# SMALL +y centroid offset above its host at the SAME centroid x. The offset
# is a STRUCTURAL MARKER, not geometry: the band constraint graph reads the
# shared centroid x as an ``align:x`` column (so it must exceed relgraph's
# structural eps, 0.05 um) and the member must NEVER leave its row's band
# (so it stays far below the 8 um band-split floor). The within-band solve
# then materializes the REAL stack gap from the target sizes + min_gap --
# the sketch bboxes of a stacked pair overlap by construction (structure
# carrier; final non-overlap is the solver's by-construction guarantee,
# pinned by the R8 tests in tests/test_placement_quality.py).
_STACK_SUBLEVEL_DY_UM: float = 0.5

# Name prefix of a RAIL-ROW pseudo entry in the sketch bboxes: a reserved,
# device-free row that keeps a through-going corridor open for a power-rail
# trunk (plan S4). The pseudo entry exists ONLY in ``DesignSketch.bboxes`` (so
# the band machinery and the non-overlap solve preserve the corridor); it never
# appears in ``DesignSketch.rows`` and consumers must strip it from the final
# placement. Single-sourced from relgraph (the stacker recognises the prefix
# to stack reservation seams FLUSH -- the P2 corridor-accounting contract).
from clay.layout.extract.relgraph import RESERVATION_ROW_PREFIX as RAIL_ROW_PREFIX


@dataclass(frozen=True)
class MirrorGroup:
    """A current-mirror group: same-kind, same-L devices sharing one gate net.

    ``members`` is name-sorted. ``diode_ref`` is the diode-connected member
    (gate == drain) anchoring the mirror, or None when no member is
    diode-connected (an externally biased matched group -- honest, not faked).
    """

    members: Tuple[str, ...]
    kind: str
    gate_net: str
    diode_ref: Optional[str]


@dataclass(frozen=True)
class DesignSketch:
    """A design-only rough placement: structure carried as geometry.

    ``bboxes``: name -> (x0, y0, x1, y1) um, sized exactly to the given
    target-PDK sizes; disjoint by construction EXCEPT on the opt-in
    ``pack_rows_2d`` path, where a stacked pair overlaps on purpose (the
    ``_STACK_SUBLEVEL_DY_UM`` structural marker -- the solve, not the
    sketch, owns final non-overlap). ``rows``: bottom-up rows, each a
    left-to-right name tuple. ``gaps``: every structural choice the topology
    could not decide, made deterministically and recorded here (the
    language-intent gap account a designer can override in Clay source).
    ``provenance``: ``'design-only'`` (the default, leakage-free path) or
    ``'style-transfer'`` (row partition / order consumed a reference-derived
    :class:`~clay.layout.seqpair_model.RelativeOrder` -- self-declared, never
    impersonating the design-only path).
    """

    bboxes: Dict[str, _Bbox] = field(default_factory=dict)
    rows: Tuple[Tuple[str, ...], ...] = ()
    gaps: Tuple[str, ...] = ()
    provenance: str = "design-only"
    # Rail-row reservations (plan S4): ``(pseudo_name, net, position)`` per
    # reserved power-rail band, position in {'top','bottom','between_rows:K'}.
    # The pseudo bbox lives in ``bboxes`` (so the corridor survives the band
    # solve) but never in ``rows``. Empty when no rails were requested.
    rail_reservations: Tuple[Tuple[str, str, str], ...] = ()
    # Channel demands (plan S6): DESIGN-sourced crossing-net count per row
    # boundary -- ``channel_demands[k]`` is the number of distinct nets whose
    # devices sit both at-or-below emitted row k and at-or-above emitted row
    # k+1. Indexed over the FULL emitted bottom-up row stack (device rows PLUS
    # any rail pseudo rows -- the same stack ``cluster_bands`` recovers as
    # bands, so index k is band gap k for the solve face). Derived from the
    # intent's device/passive nets only; never from reference geometry.
    channel_demands: Tuple[int, ...] = ()
    # P0 compaction (opt-in via ``derive_design_sketch(compact=True)``): the
    # design-derived :class:`~clay.layout.extract.compaction.CompactionSpec`
    # the solve face applies POST-stack (outline press, intent edges in the
    # graph). ``None`` (the default) = no compaction, byte-identical solve.
    compaction_spec: Optional[Any] = None
    # P1 placement engine (opt-in via ``derive_design_sketch(engine=...)``):
    # 'bands' (the default, byte-identical historical chain) or 'relax' (the
    # continuous-relaxation refinement stage --
    # :mod:`clay.layout.extract.place_relax` -- run by
    # ``solve_hierarchical_nonoverlap`` over its own band+press output).
    engine: str = "bands"
    # P1b mirror-rigid candidacy (opt-in via ``derive_design_sketch(
    # mirror_pairs_rigid=True)``, engine='relax' only): the relax stage
    # rigidifies declared diff/mirror pairs as mirror bodies about a common
    # vertical axis -- (axis_x, d, y) -- and reports each solved axis on
    # ``RelaxReport.mirror_pairs`` (the routing Mirror constraint's axis
    # supply). False (the default) is byte-identical.
    mirror_rigid: bool = False


class _SketchIntentView:
    """An intent view whose GEOMETRY is the design sketch (never the reference).

    Carries the topology fields the downstream constraint-graph build consumes
    (matched / mirror / diff pairs) and the sketch bboxes; ``symmetry_axis`` is
    None (the reference axis coordinate is exactly the kind of value that must
    not leak)."""

    def __init__(self, intent: Any, sketch: DesignSketch) -> None:
        self.cell = getattr(intent, "cell", "")
        self.devices = list(getattr(intent, "devices", []) or [])
        self.passives = list(getattr(intent, "passives", []) or [])
        self.matched_pairs = list(getattr(intent, "matched_pairs", []) or [])
        self.mirror_pairs = list(getattr(intent, "mirror_pairs", []) or [])
        self.diff_pairs = list(getattr(intent, "diff_pairs", []) or [])
        self.device_bboxes = dict(sketch.bboxes)
        self.symmetry_axis = None
        # P0 compaction: ride the sketch's spec through to the band solve
        # (relgraph reads it off the intent view -- the anchor wiring stays
        # one parameter on the sketch call).
        self.compaction_spec = getattr(sketch, "compaction_spec", None)
        # P1 placement engine: ride the sketch's engine request the same way
        # (relgraph stashes the view on the hier when it says 'relax').
        self.placement_engine = getattr(sketch, "engine", "bands")
        # P1b mirror-rigid request: same ride-along channel; consumed by
        # place_relax.refine_banded_placement (folded into the knobs face).
        self.relax_mirror_rigid = bool(getattr(sketch, "mirror_rigid", False))
        # @edge_anchor intent: a DECLARED design prior carried on the intent
        # itself (not a sketch param) -- the edge-anchor potential in
        # place_relax pulls every matching device flush to one die edge without
        # growing the die. Default empty -> the relax energy term is inert
        # (byte-identical).
        self.edge_anchor_devices = list(
            getattr(intent, "edge_anchor_devices", []) or [])
        # @edge_anchor IO-edge exclusion: the categorical set of die edges an
        # IO-edge intent already claims (VIP/VIN left, VOUT right). The relax
        # anchor reads it so the cap's SOLVED clear side avoids an IO-occupied
        # wall. Default empty -> no exclusion (byte-identical).
        self.edge_anchor_io_edges = list(
            getattr(intent, "edge_anchor_io_edges", []) or [])
        # P4 closure: the EXTRACTED QSR relation network (style-transfer) is the
        # OTHER source of edge-anchor members (edge_anchor_members derives a
        # device from an anchored device<->frame TPP relation). The view must
        # carry it like edge_anchor_devices, else place_relax's
        # edge_anchor_members(view) sees only the DECLARED prior and the
        # extracted intent never reaches the lowering -- the cap stays un-hugged
        # on the extraction-driven path (measured 2026-06-14: extracted-only
        # path left the cap 11.2um off the wall vs the declared path's flush).
        # Zero-coordinate container (P1), no leakage. Default None -> inert.
        self.placement_relations = getattr(intent, "placement_relations", None)


def sketch_intent_view(intent: Any, sketch: DesignSketch) -> _SketchIntentView:
    """The design-sourced stand-in for the reference-bboxed intent: feed THIS to
    ``relgraph.build_hierarchical_graph`` so band membership / order / stacking
    derive from the sketch (design) instead of the reference layout."""
    return _SketchIntentView(intent, sketch)


# --------------------------------------------------------------------------- #
# structure: mirror groups
# --------------------------------------------------------------------------- #
def mirror_groups(intent: Any) -> List[MirrorGroup]:
    """Connected components of ``intent.mirror_pairs`` -> :class:`MirrorGroup`s.

    A mirror pair shares the gate net and a device has exactly one gate, so each
    component shares ONE gate net by construction. Pure topology: only
    ``mirror_pairs`` + the device records (kind / nets) are read."""
    devs = {str(d.name): d for d in getattr(intent, "devices", []) or []}
    parent: Dict[str, str] = {}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for a, b in getattr(intent, "mirror_pairs", []) or []:
        a, b = str(a), str(b)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    comps: Dict[str, List[str]] = defaultdict(list)
    for n in parent:
        comps[find(n)].append(n)

    out: List[MirrorGroup] = []
    for members in comps.values():
        members_t = tuple(sorted(members))
        in_devs = [devs[m] for m in members_t if m in devs]
        if not in_devs:
            continue
        gate = str(in_devs[0].nets.get("gate", ""))
        kind = str(in_devs[0].kind)
        diode = next(
            (d.name for d in in_devs
             if d.nets.get("gate") == d.nets.get("drain")),
            None,
        )
        out.append(MirrorGroup(members=members_t, kind=kind, gate_net=gate,
                               diode_ref=diode))
    return sorted(out, key=lambda g: g.members)


# --------------------------------------------------------------------------- #
# units: the indivisible placement groups
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Unit:
    key: str                      # deterministic id = first member name
    members: Tuple[str, ...]      # in within-row order
    why: str                      # 'mirror' | 'diff' | 'mos' | 'passive'


def _derive_units(intent: Any) -> Tuple[List[_Unit], List[str]]:
    """Partition all devices+passives into indivisible placement units.

    Priority: mirror groups (shared-gate bus) > diff pairs > singletons. Every
    underdetermined ORDER choice inside a unit is recorded as a gap."""
    gaps: List[str] = []
    claimed: set = set()
    units: List[_Unit] = []

    devs = {str(d.name): d for d in getattr(intent, "devices", []) or []}

    for g in mirror_groups(intent):
        members = [m for m in g.members if m not in claimed]
        if len(members) < 2:
            continue
        # within-group order: descending M then name -- deterministic, recorded.
        ordered = tuple(sorted(
            members, key=lambda n: (-int(getattr(devs.get(n), "M", 1) or 1), n)
        ))
        if len(ordered) > 1:
            gaps.append(
                f"mirror group {'/'.join(g.members)}: within-row member order "
                f"underdetermined by topology; chose descending-M-then-name "
                f"{'->'.join(ordered)} (declare @order in Clay to override)"
            )
        claimed.update(ordered)
        units.append(_Unit(key=ordered[0], members=ordered, why="mirror"))

    for a, b in getattr(intent, "diff_pairs", []) or []:
        a, b = str(a), str(b)
        if a in claimed or b in claimed:
            continue
        left, right = sorted((a, b))
        gaps.append(
            f"diff pair {a}/{b}: left/right twin side underdetermined by "
            f"topology (renamed-isomorph ambiguity); chose {left} left of "
            f"{right} by name (declare @symmetry order in Clay to override)"
        )
        claimed.update((a, b))
        units.append(_Unit(key=left, members=(left, right), why="diff"))

    for d in getattr(intent, "devices", []) or []:
        n = str(d.name)
        if n not in claimed:
            claimed.add(n)
            units.append(_Unit(key=n, members=(n,), why="mos"))
    for p in getattr(intent, "passives", []) or []:
        n = str(p.name)
        if n not in claimed:
            claimed.add(n)
            units.append(_Unit(key=n, members=(n,), why="passive"))

    return sorted(units, key=lambda u: u.key), gaps


def _unit_connectivity(intent: Any, units: Sequence[_Unit]) -> Dict[frozenset, float]:
    """Net-sharing weight between unit pairs, hypergraph clique-weighted.

    Each net contributes 1/(k-1) to every pair of the k units it touches, so a
    rail touching most units self-downweights -- connectivity is design-derived
    with NO name-based power-net special-casing."""
    unit_of: Dict[str, str] = {}
    for u in units:
        for m in u.members:
            unit_of[m] = u.key

    nets_units: Dict[str, set] = defaultdict(set)
    for d in getattr(intent, "devices", []) or []:
        uk = unit_of.get(str(d.name))
        if uk is None:
            continue
        for net in (d.nets or {}).values():
            nets_units[str(net)].add(uk)
    for p in getattr(intent, "passives", []) or []:
        uk = unit_of.get(str(p.name))
        if uk is None:
            continue
        for net in (p.nets or {}).values():
            nets_units[str(net)].add(uk)

    w: Dict[frozenset, float] = defaultdict(float)
    for uks in nets_units.values():
        k = len(uks)
        if k < 2:
            continue
        share = 1.0 / (k - 1)
        ulist = sorted(uks)
        for i in range(len(ulist)):
            for j in range(i + 1, len(ulist)):
                w[frozenset((ulist[i], ulist[j]))] += share
    return dict(w)


def _spec_keys(intent: Any) -> Dict[str, Tuple]:
    """Design-sourced device SPEC key per name: ``(kind, W, L)`` for a MOS,
    ``(kind, value)`` for a passive -- pure netlist data (the P4a clustering
    vocabulary; never a geometric or reference value)."""
    out: Dict[str, Tuple] = {}
    for d in getattr(intent, "devices", []) or []:
        out[str(d.name)] = (str(getattr(d, "kind", "dev")),
                            float(getattr(d, "W", 0.0) or 0.0),
                            float(getattr(d, "L", 0.0) or 0.0))
    for p in getattr(intent, "passives", []) or []:
        out[str(p.name)] = (str(getattr(p, "kind", "passive")),
                            float(getattr(p, "value", 0.0) or 0.0))
    return out


def _seriate_units(
    units: Sequence[_Unit],
    w: Dict[frozenset, float],
    specs: Optional[Dict[str, Tuple]] = None,
) -> List[_Unit]:
    """Greedy 1D chain seriation: keep strongly-connected units adjacent.

    Start at the unit with the highest total connectivity (ties by key); extend
    the chain at the tail with the unplaced unit most connected to the tail
    (ties: same (kind, W, L) spec as the tail unit -- the P4a clustering
    preference, acting ONLY where connectivity ties; then total connectivity,
    then key). Deterministic; wirelength-driven, spec-clustered on ties."""
    if not units:
        return []
    by_key = {u.key: u for u in units}
    total: Dict[str, float] = {u.key: 0.0 for u in units}
    for pair, wt in w.items():
        for k in pair:
            if k in total:
                total[k] += wt

    def spec_of(unit_key: str) -> Optional[Tuple]:
        if specs is None:
            return None
        return specs.get(by_key[unit_key].members[0])

    start = max(sorted(total), key=lambda k: total[k])
    chain = [start]
    left = set(total) - {start}
    while left:
        tail = chain[-1]
        tail_spec = spec_of(tail)
        nxt = max(
            sorted(left),
            key=lambda k: (
                w.get(frozenset((tail, k)), 0.0),
                1 if (tail_spec is not None and spec_of(k) == tail_spec) else 0,
                total[k],
            ),
        )
        chain.append(nxt)
        left.remove(nxt)
    return [by_key[k] for k in chain]


# --------------------------------------------------------------------------- #
# the sketch
# --------------------------------------------------------------------------- #
def _pack_chain_into_rows(
    chain: Sequence[_Unit],
    unit_width,
    row_w_target: float,
    dev_gap_um: float,
) -> List[List[_Unit]]:
    """Greedy left-to-right packing of the unit chain into rows of width
    ``row_w_target`` (the original derive_design_sketch packing, factored out
    unchanged so the default path stays byte-identical)."""
    rows: List[List[_Unit]] = [[]]
    cur_w = 0.0
    for u in chain:
        uw = unit_width(u)
        add = uw if not rows[-1] else uw + dev_gap_um
        if rows[-1] and cur_w + add > row_w_target:
            rows.append([u])
            cur_w = uw
        else:
            rows[-1].append(u)
            cur_w += add
    return rows


def _packed_outline_estimate(
    rows: Sequence[Sequence[_Unit]],
    unit_width,
    size_of,
    dev_gap_um: float,
    row_channel_um: float,
) -> Tuple[float, float]:
    """Outline (w, h) a unit-row packing would emit, mirroring the
    :func:`_emit_name_rows` pitch math (uniform channel, min row pitch) --
    the ESTIMATE the aspect-target width search scores; never emitted."""
    live = [r for r in rows if r]
    if not live:
        return 0.0, 0.0
    w = max(sum(unit_width(u) for u in r) + dev_gap_um * (len(r) - 1)
            for r in live)
    heights = [max(size_of(m)[1] for u in r for m in u.members) for r in live]
    cy = 0.0
    lo = -heights[0] / 2.0
    hi = heights[0] / 2.0
    for k in range(1, len(heights)):
        pitch = max(
            heights[k - 1] / 2.0 + row_channel_um + heights[k] / 2.0,
            _SKETCH_MIN_ROW_PITCH_UM,
        )
        cy += pitch
        hi = cy + heights[k] / 2.0
    return w, hi - lo


def _aspect_row_w_target(
    chain: Sequence[_Unit],
    unit_width,
    size_of,
    total_area: float,
    widest: float,
    aspect_target: float,
    dev_gap_um: float,
    row_channel_um: float,
) -> float:
    """Solve the row-packing target width for a DECLARED outline aspect
    (width/height ratio -- a proportion, never a coordinate).

    Seed = ``sqrt(total_area * aspect)`` (area * ratio = width^2); because
    packed rows under- or over-fill any single seed (channel overhead,
    row-height quantization, indivisible wide units make the achievable
    ratios DISCRETE), a fixed deterministic width ladder around the seed is
    packed and the width whose ACHIEVED packed outline ratio is log-closest
    to the declared aspect wins (ties keep the narrower width). Pure design
    objective: only the unit chain + target sizes are read; bounded
    candidates, no randomness."""
    import math

    seed = max((total_area * float(aspect_target)) ** 0.5, widest)
    best_score: Optional[float] = None
    best_w = seed
    for f in (0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75, 2.1, 2.5, 3.0):
        w_t = max(seed * f, widest)
        rows = _pack_chain_into_rows(chain, unit_width, w_t, dev_gap_um)
        w_a, h_a = _packed_outline_estimate(
            rows, unit_width, size_of, dev_gap_um, row_channel_um
        )
        if w_a <= 0.0 or h_a <= 0.0:
            continue
        score = abs(math.log(w_a / h_a) - math.log(float(aspect_target)))
        if best_score is None or score < best_score - 1e-12:
            best_score, best_w = score, w_t
    return best_w


def _emit_name_rows(
    names_rows: Sequence[Sequence[str]],
    size_of,
    dev_gap_um: float,
    row_channel_um: float,
    channel_per_gap: Optional[Sequence[float]] = None,
    stacks_per_row: Optional[Sequence[Dict[str, str]]] = None,
) -> Tuple[Dict[str, _Bbox], Tuple[Tuple[str, ...], ...]]:
    """Emit sketch geometry for bottom-up rows of device NAMES (the original
    derive_design_sketch emission loop, factored out unchanged: identical float
    operations in identical order -> the default path stays byte-identical).

    ``channel_per_gap`` (plan S6, optional): per-row-boundary channel width
    (um) -- entry k is the channel between rows k and k+1, replacing the
    uniform ``row_channel_um`` there. ``None`` keeps the historical uniform
    channel byte-for-byte.

    ``stacks_per_row`` (R8, optional): per-row ``{rider -> host}`` stacks from
    :func:`_pack_name_rows_2d`. A row with stacks emits its lane (riders
    removed) left-to-right, each stacked column at the column's shared center
    x (column width = max of host/rider widths) with the rider raised by the
    ``_STACK_SUBLEVEL_DY_UM`` structural marker. Row pitch math is identical
    (the row height is still the tallest MEMBER -- packing never grows it).
    ``None`` (the default) keeps the flat emission byte-for-byte."""
    bboxes: Dict[str, _Bbox] = {}
    out_rows: List[Tuple[str, ...]] = []
    prev_cy: Optional[float] = None
    prev_h = 0.0
    for gi, names in enumerate(names_rows):
        names = list(names)
        row_h = max(size_of(n)[1] for n in names)
        if prev_cy is None:
            cy = 0.0
        else:
            chan = (row_channel_um if channel_per_gap is None
                    else float(channel_per_gap[gi - 1]))
            pitch = max(
                prev_h / 2.0 + chan + row_h / 2.0,
                _SKETCH_MIN_ROW_PITCH_UM,
            )
            cy = prev_cy + pitch
        st = (stacks_per_row[gi] if stacks_per_row is not None else None) or {}
        if not st:
            row_w = (sum(size_of(n)[0] for n in names)
                     + dev_gap_um * (len(names) - 1))
            x = -row_w / 2.0
            for n in names:
                nw, nh = size_of(n)
                bboxes[n] = (x, cy - nh / 2.0, x + nw, cy + nh / 2.0)
                x += nw + dev_gap_um
        else:
            riders_of: Dict[str, str] = {}
            for rider, host in st.items():
                riders_of[host] = rider
            lane = [n for n in names if n not in st]

            def item_w(n: str) -> float:
                w = size_of(n)[0]
                r = riders_of.get(n)
                if r is not None:
                    w = max(w, size_of(r)[0])
                return w

            row_w = (sum(item_w(n) for n in lane)
                     + dev_gap_um * (len(lane) - 1))
            x = -row_w / 2.0
            for n in lane:
                iw = item_w(n)
                xc = x + iw / 2.0
                nw, nh = size_of(n)
                bboxes[n] = (xc - nw / 2.0, cy - nh / 2.0,
                             xc + nw / 2.0, cy + nh / 2.0)
                r = riders_of.get(n)
                if r is not None:
                    rw, rh = size_of(r)
                    rcy = cy + _STACK_SUBLEVEL_DY_UM
                    bboxes[r] = (xc - rw / 2.0, rcy - rh / 2.0,
                                 xc + rw / 2.0, rcy + rh / 2.0)
                x += iw + dev_gap_um
        out_rows.append(tuple(names))
        prev_cy, prev_h = cy, row_h
    return bboxes, tuple(out_rows)


def _pair_partners(intent: Any) -> Dict[str, set]:
    """name -> declared partner names (matched / mirror / diff pairs) -- the
    design-sourced protection set for row balancing: a member whose partner
    shares its row never moves alone (colocated matching stays colocated)."""
    out: Dict[str, set] = defaultdict(set)
    for attr in ("matched_pairs", "mirror_pairs", "diff_pairs"):
        for ab in getattr(intent, attr, ()) or ():
            if len(ab) == 2:
                a, b = str(ab[0]), str(ab[1])
                out[a].add(b)
                out[b].add(a)
    return dict(out)


def _rail_overhead_um(rails: Optional[Sequence[Dict[str, Any]]]) -> float:
    """Total corridor height (um) the positioned rail entries will reserve --
    the constant vertical overhead the balance objective accounts for."""
    tot = 0.0
    for e in rails or []:
        if e.get("width_um") is None:
            continue
        if e.get("side") in ("top", "bottom") or e.get("between_rows") is not None:
            tot += float(e["width_um"])
    return tot


def _balance_name_rows(
    intent: Any,
    names_rows: Sequence[Sequence[str]],
    size_of,
    dev_gap_um: float,
    row_channel_um: float,
    rail_overhead_um: float,
) -> Tuple[List[List[str]], List[str]]:
    """P3 row balance: move weakly-bound members off over-wide rows when the
    ESTIMATED outline area strictly decreases (a pure design objective).

    Estimate: ``max-row-width x (sum of row heights + inter-row channels +
    rail corridor overhead)`` -- widths/heights from the caller-given target
    sizes, never a reference value. Deterministic greedy: per iteration, every
    (movable member, strictly NARROWER row) candidate is scored and the
    strictly-best move applies (ties by name, then target row). Moves go from
    wider to narrower rows only -- balancing never widens the widest row and
    never degenerates the row structure into one row. PROTECTION (design-sourced):
    a member with a declared matched/mirror/diff partner in its OWN row never
    moves alone -- colocated matching intent outranks area. Insertion position
    follows net connectivity (next to the strongest-connected member of the
    target row; same-spec neighbour breaks ties). Every move is recorded as a
    note for ``DesignSketch.gaps`` -- never silent.

    R3 CALLER CONTRACT: pass ONLY rows the design owns. Style rows (the
    ``style_order`` partition) are a hard membership constraint and must
    never reach this function -- the style path freezes them and balances
    only its design-only fallback rows.
    """
    rows: List[List[str]] = [list(r) for r in names_rows if r]
    if len(rows) < 2:
        return rows, []

    names = [n for r in rows for n in r]
    singles = [_Unit(key=n, members=(n,), why="balance") for n in sorted(names)]
    conn = _unit_connectivity(intent, singles)
    partners = _pair_partners(intent)
    specs = _spec_keys(intent)

    def row_w(r: Sequence[str]) -> float:
        return (sum(size_of(n)[0] for n in r)
                + dev_gap_um * (len(r) - 1)) if r else 0.0

    def row_h(r: Sequence[str]) -> float:
        return max((size_of(n)[1] for n in r), default=0.0)

    def est_area(rs: Sequence[Sequence[str]]) -> float:
        live = [r for r in rs if r]
        if not live:
            return 0.0
        w = max(row_w(r) for r in live)
        h = (sum(row_h(r) for r in live)
             + row_channel_um * (len(live) - 1) + rail_overhead_um)
        return w * h

    notes: List[str] = []
    for _round in range(3 * len(names)):
        cur = est_area(rows)
        best = None  # (new_area, member, src_idx, dst_idx)
        for si, src in enumerate(rows):
            for m in src:
                if any(p in src for p in partners.get(m, ())):
                    continue  # colocated matching protection
                for di, dst in enumerate(rows):
                    if di == si or row_w(dst) >= row_w(src):
                        continue  # only wider -> strictly narrower moves
                    trial = [list(r) for r in rows]
                    trial[si].remove(m)
                    trial[di].append(m)
                    a = est_area(trial)
                    if a < cur - 1e-9:
                        cand = (a, m, si, di)
                        if best is None or cand < best:
                            best = cand
        if best is None:
            break
        new_area, m, si, di = best
        rows[si].remove(m)
        dst = rows[di]
        if dst:
            # insertion anchor: strongest net bond, then same-spec neighbour,
            # then the RIGHTMOST position (an unbonded member appends at the
            # row end instead of splitting an existing neighbour pair).
            anchor = max(
                range(len(dst)),
                key=lambda i: (
                    conn.get(frozenset((m, dst[i])), 0.0),
                    1 if specs.get(m) == specs.get(dst[i]) else 0,
                    i,
                ),
            )

            def _splits_pair(slot: int) -> bool:
                # inserting at ``slot`` lands between dst[slot-1] and dst[slot]
                # -- splitting them is only penalised when they are a same-spec
                # neighbour pair (the P4a clustering the insertion must respect).
                if slot <= 0 or slot >= len(dst):
                    return False
                return specs.get(dst[slot - 1]) is not None and \
                    specs.get(dst[slot - 1]) == specs.get(dst[slot])

            slot = anchor + 1
            if _splits_pair(slot) and not _splits_pair(anchor):
                slot = anchor  # insert on the anchor's other side instead
            dst.insert(slot, m)
        else:
            dst.append(m)
        notes.append(
            f"row-balance: moved {m} from row {si} to row {di} "
            f"(estimated outline {cur:.1f} -> {new_area:.1f} um^2)"
        )
        rows = [r for r in rows if r]
    return rows, notes


def _pack_name_rows_2d(
    intent: Any,
    names_rows: Sequence[Sequence[str]],
    size_of,
    dev_gap_um: float,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """R8 IN-ROW 2D PACKING: stack short members as two-level columns INSIDE
    their own row -> ``(stacks_per_row, notes)``, ``stacks_per_row[k]`` =
    ``{rider -> host}`` for row k.

    Pure design objective (outline area): the row WIDTH shrinks by
    ``min(w_host, w_rider) + dev_gap`` per stack while the row HEIGHT is a
    HARD bound (a stack must fit under the row's tallest member). Membership
    is untouched by construction -- packing rearranges geometry inside a row,
    never across rows (R3 style fidelity outranks packing).

    Constraints, all design-sourced:
      * PROTECTION: a member with a declared matched/mirror/diff partner in
        its OWN row never stacks (as host or rider) -- colocated matching
        intent outranks area, same rule as row balance.
      * COUPLED HEIGHT BOUND: all riders of a row share ONE sub-level bottom
        line in the solve (their equal sketch sub-level y makes them an
        align:y group), at tallest-host top + gap; so feasibility is
        ``max(host heights) + dev_gap + max(rider heights) <= row height``
        over the WHOLE selected set, not per pair.
      * Deterministic greedy: per step the feasible pair with the largest
        width saving wins (ties by host then rider name); host = the taller
        (then wider, then name-smaller) member. Every stack is recorded.
    """
    partners = _pair_partners(intent)
    stacks: List[Dict[str, str]] = []
    notes: List[str] = []
    for ri, row in enumerate(names_rows):
        row = list(row)
        st: Dict[str, str] = {}
        stacks.append(st)
        if len(row) < 2:
            continue
        row_h = max(size_of(n)[1] for n in row)
        rowset = set(row)
        movable = [n for n in row
                   if not (partners.get(n, set()) & (rowset - {n}))]
        used: set = set()
        max_host_h = 0.0
        max_rider_h = 0.0
        while True:
            best: Optional[Tuple[float, str, str]] = None
            for i in range(len(movable)):
                for j in range(i + 1, len(movable)):
                    a, b = movable[i], movable[j]
                    if a in used or b in used:
                        continue
                    wa, ha = size_of(a)
                    wb, hb = size_of(b)
                    if (ha, wa) > (hb, wb) or ((ha, wa) == (hb, wb) and a < b):
                        host, rider = a, b
                    else:
                        host, rider = b, a
                    hh = size_of(host)[1]
                    rh = size_of(rider)[1]
                    if max(max_host_h, hh) + dev_gap_um + \
                            max(max_rider_h, rh) > row_h + 1e-9:
                        continue
                    saving = dev_gap_um + min(wa, wb)
                    cand = (-saving, host, rider)
                    if best is None or cand < best:
                        best = cand
            if best is None:
                break
            neg_saving, host, rider = best
            st[rider] = host
            used.update((host, rider))
            max_host_h = max(max_host_h, size_of(host)[1])
            max_rider_h = max(max_rider_h, size_of(rider)[1])
            notes.append(
                f"row-pack: stacked {rider} above {host} in row {ri} "
                f"(saves {-neg_saving:.2f} um row width; row height "
                f"{row_h:.2f} um kept)"
            )
    return stacks, notes


def _align_stacks_to_rows(
    names_rows: Sequence[Sequence[str]],
    stacks: Optional[List[Dict[str, str]]],
) -> Optional[List[Dict[str, str]]]:
    """Map pre-rail-injection per-device-row stacks onto the post-injection
    row list (rail pseudo rows carry no stacks; device-row order is preserved
    by :func:`_inject_rail_rows`)."""
    if stacks is None:
        return None
    it = iter(stacks)
    out: List[Dict[str, str]] = []
    for row in names_rows:
        if any(str(n).startswith(RAIL_ROW_PREFIX) for n in row):
            out.append({})
        else:
            out.append(next(it, {}))
    return out


def _channel_crossing_nets(
    intent: Any, names_rows: Sequence[Sequence[str]]
) -> Tuple[Tuple[Tuple[str, ...], ...], ...]:
    """DESIGN-sourced crossing-net topology per row boundary (plan S6 single
    source -- :func:`_channel_demands` is exactly the per-boundary count of
    this; one derivation, no second bookkeeping).

    For each net, take the set of bottom-up row indices its devices/passives
    occupy; the net CROSSES boundary k iff that set spans the boundary
    (min(rows) <= k < max(rows)). Element k is the tuple of nets crossing
    boundary k (sorted by net name, deterministic), each net given as the
    sorted tuple of its member device/passive NAMES in the row stack --
    names only, never a coordinate: the geometric pin x interval is resolved
    by the consumer from ITS OWN placement (compaction's x-resolved channel
    capacity check), so this stays pure topology with no reference read.
    Rail pseudo rows carry no devices, so they contribute nothing (but keep
    their index, so boundaries align with the band stack)."""
    row_of: Dict[str, int] = {}
    for k, row in enumerate(names_rows):
        for n in row:
            row_of[str(n)] = k

    nets_rows: Dict[str, set] = defaultdict(set)
    nets_members: Dict[str, set] = defaultdict(set)
    for rec in list(getattr(intent, "devices", []) or []) + list(
            getattr(intent, "passives", []) or []):
        k = row_of.get(str(rec.name))
        if k is None:
            continue
        for net in (rec.nets or {}).values():
            nets_rows[str(net)].add(k)
            nets_members[str(net)].add(str(rec.name))

    n_gaps = max(len(names_rows) - 1, 0)
    out: List[List[Tuple[str, ...]]] = [[] for _ in range(n_gaps)]
    for net in sorted(nets_rows):
        lo, hi = min(nets_rows[net]), max(nets_rows[net])
        members = tuple(sorted(nets_members[net]))
        for g in range(lo, hi):
            out[g].append(members)
    return tuple(tuple(g) for g in out)


def _channel_demands(
    intent: Any, names_rows: Sequence[Sequence[str]]
) -> Tuple[int, ...]:
    """DESIGN-sourced channel demand per row boundary (plan S6).

    ``demands[k]`` is the count of distinct nets crossing boundary k --
    by definition the per-boundary length of
    :func:`_channel_crossing_nets` (the single derivation both faces
    consume; see there for the crossing rule and the leakage face)."""
    return tuple(
        len(g) for g in _channel_crossing_nets(intent, names_rows)
    )


def _channel_per_gap(
    demands: Sequence[int],
    row_channel_um: float,
    channel_track_um: Optional[float],
) -> Optional[List[float]]:
    """Lower demands -> per-gap sketch channel widths (um): each gap is
    ``max(row_channel_um, demand * channel_track_um)``. ``None`` track keeps
    the historical uniform channel (returns None -> byte-identical path)."""
    if channel_track_um is None:
        return None
    return [max(float(row_channel_um), d * float(channel_track_um))
            for d in demands]


def sketch_channel_um_per_gap(
    sketch: DesignSketch, track_um: float
) -> Dict[int, float]:
    """Lower the sketch's ``channel_demands`` into the per-band-gap channel
    dict ``relgraph.solve_hierarchical_nonoverlap(channel_um_per_gap=)``
    consumes: ``{gap_index: demand * track_um}``. The track width is the
    caller's PDK pitch-class datum (parameter face -- relgraph itself stays
    pure geometry and looks up no PDK); the historical ``band_channel_um``
    floor is applied at the consumption side, never weakened here."""
    return {g: float(d) * float(track_um)
            for g, d in enumerate(sketch.channel_demands)}


def _compact_mode(compact: Any) -> Optional[str]:
    """Normalize the ``compact`` parameter face: ``False``/``None`` -> None
    (no spec, byte-identical), ``True`` -> ``'press'`` (the historical
    press-only semantics), ``'2d'`` -> the full-2D face (band-order-directed
    press + the solve-side embed search). Anything else raises."""
    if compact is None or compact is False:
        return None
    if compact is True or compact == "press":
        return "press"
    if compact == "2d":
        return "2d"
    raise ValueError(
        f"derive_design_sketch: unknown compact face {compact!r} "
        "(expected False, True, or '2d')"
    )


def _compaction_gap_note(mode: str) -> str:
    """The honest self-declaration recorded in the sketch gaps per compact
    face (the 'press' text is byte-stable for the historical compact=True)."""
    if mode == "2d":
        return (
            "compaction: full-2D face enabled (press directs cross-row "
            "y edges by ROW ORDER; embed search may re-slot a partner-rigid "
            "group across row x-packs -- membership/order/channels kept, "
            "within-row x order of a repacked row given up)"
        )
    return (
        "compaction: solve-side outline press enabled (intent edges in "
        "the graph; row members may leave the shared bottom line)"
    )


def _build_compaction_spec(intent: Any, out_rows: Sequence[Sequence[str]],
                           mode: str = "press"):
    """Lower the emitted row stack + declared pair topology into the
    :class:`~clay.layout.extract.compaction.CompactionSpec` the solve-side
    press consumes (P0). DESIGN-sourced only: the FULL emitted bottom-up row
    partition (device rows + rail pseudo rows -- row gap k is band gap k for
    the solve's ``channel_um_per_gap``) and the intent's declared
    matched/mirror/diff pairs restricted to SAME-ROW pairs (cross-row
    partners keep order via the press edges; pinning their offset would
    chain whole rows rigid). ``mode`` is the validated compact face
    (``'press'`` or ``'2d'`` -- see :func:`_compact_mode`).

    ``channel_xnets`` rides along for the mode='2d' embed stage: the
    per-boundary crossing-net topology from the SAME derivation as the
    sketch's ``channel_demands`` (:func:`_channel_crossing_nets` -- single
    source), names only, so the solve-side x-resolved channel capacity
    check can hold each crossing net's lane segment against an embed
    candidate without a second accounting.

    ``keepout_names`` likewise: the cap passives matched by the intent's
    declared/extracted ``no_route_under`` entries (the same '*'-or-exact
    rule the router's keep-out lowering uses), so the embed stage can hold
    each hard keep-out footprint's flank corridors. No entries -> empty
    (vacuous guard, byte-identical path)."""
    from clay.layout.extract.compaction import CompactionSpec

    row_of = {str(n): k for k, row in enumerate(out_rows) for n in row}
    rail_rows = frozenset(
        k for k, row in enumerate(out_rows)
        if any(str(n).startswith(RAIL_ROW_PREFIX) for n in row)
    )
    partners = set()
    for a, partners_of in _pair_partners(intent).items():
        for b in partners_of:
            if row_of.get(a) is not None and row_of.get(a) == row_of.get(b):
                partners.add((a, b) if a < b else (b, a))
    keepouts: List[str] = []
    if getattr(intent, "no_route_under", None):
        from clay.layout.extract.under_cap_intent import matching_entries

        for p in getattr(intent, "passives", []) or []:
            kind = str(getattr(p, "kind", "")).lower()
            if not (kind.startswith("cap") or kind == "c"):
                continue
            name = str(getattr(p, "name", ""))
            if name in row_of and matching_entries(intent, name):
                keepouts.append(name)
    return CompactionSpec(row_of=row_of, rail_rows=rail_rows,
                          partners=tuple(sorted(partners)), mode=str(mode),
                          channel_xnets=_channel_crossing_nets(
                              intent, out_rows),
                          keepout_names=tuple(sorted(keepouts)))


def _inject_rail_rows(
    names_rows: Sequence[Sequence[str]],
    rails: Optional[Sequence[Dict[str, Any]]],
    size_of,
    dev_gap_um: float,
) -> Tuple[List[List[str]], Dict[str, _XY], List[Tuple[str, str, str]], List[str]]:
    """Insert one device-free RAIL pseudo row per positioned rail entry.

    Each rail entry (see ``style_order.extract_rail_intent`` for the intent
    vocabulary) needs a relative position (``side`` in {'top','bottom'} or
    ``between_rows=k`` = the gap between bottom-up device rows k and k+1) AND a
    ``width_um`` (the DESIGN-derived rail width class the consumer attaches --
    this module never invents one). An entry missing either degrades honestly:
    recorded as a gap note, no reservation, never a silent default position.

    Returns ``(names_rows', rail_sizes, reservations, notes)`` where the pseudo
    row spans the widest device row (a through-going band) at ``width_um``
    height. Empty/None ``rails`` returns the input unchanged.
    """
    rows: List[List[str]] = [list(r) for r in names_rows]
    rail_sizes: Dict[str, _XY] = {}
    reservations: List[Tuple[str, str, str]] = []
    notes: List[str] = []
    if not rails:
        return rows, rail_sizes, reservations, notes

    n_dev_rows = len(rows)
    row_w = max(
        (sum(size_of(n)[0] for n in r) + dev_gap_um * (len(r) - 1)
         for r in rows if r),
        default=0.0,
    )

    inserts: List[Tuple[int, int, str, str, str, float]] = []
    for order, entry in enumerate(rails):
        net = str(entry.get("net", "") or "")
        side = entry.get("side")
        between = entry.get("between_rows")
        width = entry.get("width_um")
        if not net:
            continue
        pseudo = f"{RAIL_ROW_PREFIX}{net}"
        if any(p == pseudo for _a, _o, p, _n, _l, _w in inserts):
            notes.append(f"rail {net}: duplicate rail entry skipped")
            continue
        if width is None:
            notes.append(
                f"rail {net}: no width_um attached; reservation skipped "
                "(honest gap -- this module invents no rail width)"
            )
            continue
        if side == "bottom":
            at, label = 0, "bottom"
        elif side == "top":
            at, label = n_dev_rows, "top"
        elif between is not None and 0 <= int(between) < n_dev_rows - 1:
            at, label = int(between) + 1, f"between_rows:{int(between)}"
        elif between is not None:
            notes.append(
                f"rail {net}: between_rows={between} out of range for "
                f"{n_dev_rows} device rows; reservation skipped"
            )
            continue
        else:
            notes.append(
                f"rail {net}: no relative position (side/between_rows); "
                "reservation skipped (rail position is intent, never a "
                "built-in default)"
            )
            continue
        inserts.append((at, order, pseudo, net, label, float(width)))

    # apply DESCENDING insertion index so earlier inserts do not shift later
    # ones; within the same gap, later-listed entries insert first so the
    # listed order reads bottom-up at that gap. Deterministic.
    for at, _order, pseudo, net, label, width in sorted(
        inserts, key=lambda t: (-t[0], -t[1])
    ):
        rows.insert(at, [pseudo])
        rail_sizes[pseudo] = (row_w, width)
    for _at, _order, pseudo, net, label, _width in sorted(
        inserts, key=lambda t: t[1]
    ):
        reservations.append((pseudo, net, label))
    return rows, rail_sizes, reservations, notes


def derive_design_sketch(
    intent: Any,
    sizes: Dict[str, _XY],
    *,
    dev_gap_um: float = _SKETCH_DEV_GAP_UM,
    row_channel_um: float = _SKETCH_ROW_CHANNEL_UM,
    style_order: Optional[Any] = None,
    style_objective: bool = False,
    rails: Optional[List[Dict[str, Any]]] = None,
    channel_track_um: Optional[float] = None,
    balance_rows: bool = False,
    aspect_target: Optional[float] = None,
    pack_rows_2d: bool = False,
    compact: Any = False,
    engine: str = "bands",
    mirror_pairs_rigid: bool = False,
) -> DesignSketch:
    """Derive the design-only placement sketch: structure from topology, geometry
    from the given target-PDK ``sizes`` -- ``intent.device_bboxes`` is NEVER read.

    PLACEMENT ENGINE (P1 -- opt-in, parameter face): ``engine='relax'``
    requests the continuous-relaxation refinement stage
    (:mod:`clay.layout.extract.place_relax`) over the band solve's own
    output -- soft style bands, rigid symmetry bodies, compaction-press
    projection; the relax report (including the honest off-band deviation
    notes) rides back on ``HierGraph.relax_report``. The default
    ``'bands'`` is BYTE-IDENTICAL to the historical chain (pinned by
    tests/test_place_relax.py); any other value raises ValueError.

    MIRROR-RIGID CANDIDACY (P1b -- opt-in, requires ``engine='relax'``):
    ``mirror_pairs_rigid=True`` asks the relax stage to rigidify the
    intent's declared symmetry pairs as mirror bodies about a common
    vertical axis (the (axis_x, d, y) change of variables -- diff pairs
    always, cross-row converging under the soft band face with a reported
    deviation; 2-member mirror groups same-row only, cross-row honestly
    noted). Each realized pair's solved axis is reported on
    ``RelaxReport.mirror_pairs`` -- the placement-side axis supply for the
    routing Mirror constraint. The default (False) is BYTE-IDENTICAL;
    requesting it with ``engine='bands'`` raises ValueError (the band solve
    has no softened-row stage to converge a pair in).

    Rows: the seriated unit chain is packed into rows against a near-square
    width target (sqrt of total footprint area; objective-driven, no
    kind-banding). Units are indivisible (a mirror group / diff pair never
    splits across rows). Rows are emitted bottom-up, each centered on x=0, with
    a row pitch that the band machinery's split threshold is guaranteed to cut.

    ASPECT TARGET (R7 -- opt-in, declared style face): ``aspect_target`` is a
    width/height OUTLINE RATIO (a declared proportion -- a ratio, never a
    coordinate; e.g. a designer declaration or a style-extracted aspect
    class). When given, the row packing target width is solved for the
    declared ratio: seed ``sqrt(total_area * aspect_target)`` (area * ratio
    = width^2) plus a deterministic fixed-point refinement against the
    ACHIEVED packed outline (see :func:`_aspect_row_w_target` -- packed rows
    under-fill the seed), so e.g. 1.67 packs a wide, flat outline. ``None``
    (the default) keeps the historical near-square ``sqrt(total_area)``
    target BYTE-IDENTICAL. ``aspect_target <= 0`` raises ValueError.

    STYLE CHANNEL (plan S1/S2 -- opt-in, self-declaring):

    * ``style_order`` (a :class:`~clay.layout.seqpair_model.RelativeOrder`
      carrying ONLY name-level relative relations, e.g. from
      ``style_order.extract_reference_order`` + ``map_order_to_design``): the
      row partition and within-row order follow the style rows; devices the
      alignment did not cover fall back to the design-only seriation (recorded
      in ``gaps``). The result self-declares ``provenance='style-transfer'``.
    * ``style_objective`` (only with ``style_order``): a DETERMINISTIC
      candidate enumeration (style-following + design-packing variants, <= 24,
      no randomness) scored by ``constraint_graph_distance`` against the style;
      the minimum-distance candidate wins (ties -> first candidate, which is
      the style-following sketch, so the result is never worse than plain
      style following).
    * ``style_order=None`` (the default): BYTE-IDENTICAL to the historical
      design-only sketch (``provenance='design-only'``); ``style_objective``
      is ignored. Pinned by tests/test_style_order.py byte goldens.

    RAIL ROWS (plan S4 -- opt-in, intent-driven): ``rails`` is a list of
    rail-band intents (``style_order.extract_rail_intent`` output, or a
    designer declaration, each with a consumer-attached ``width_um``); every
    positioned entry reserves a device-free pseudo row (see
    :func:`_inject_rail_rows`) so a power trunk has a through-going corridor.
    Rail position comes ONLY from the entries (declared or style-extracted) --
    this module adds no built-in rail-side rule. ``rails=None``/``[]`` is
    byte-identical to no rails (pinned by tests/test_rail_intent.py).

    CHANNEL-AWARE ROW SPACING (plan S6 -- opt-in, design-sourced): the sketch
    always annotates ``channel_demands`` (crossing-net count per row boundary,
    derived from the design's device nets -- see :func:`_channel_demands`).
    With ``channel_track_um`` (um per crossing net: the consumer's PDK
    metal-pitch-class datum, passed as a PARAMETER -- this module looks up no
    PDK) each row gap widens to ``max(row_channel_um, demand * track)``.
    ``channel_track_um=None`` (the default) keeps the historical uniform
    channel BYTE-IDENTICAL (pinned by tests/test_channel_aware.py).

    ROW BALANCE (P3 -- opt-in, objective-driven) x STYLE FIDELITY (R3 -- the
    hard constraint): ``balance_rows=True`` runs :func:`_balance_name_rows`
    on the rows the design OWNS. On the design-only path that is the whole
    partition: weakly-bound members move off over-wide rows while the
    ESTIMATED outline area strictly decreases; colocated matched/mirror/diff
    pairs are protected; every move is recorded in ``gaps``. On the STYLE
    path (``style_order`` given) the style row MEMBERSHIP is a HARD
    constraint: no optimization (balance, packing, splitting) may move a
    member across a style row or split one -- style fidelity outranks fill
    by contract, so balance degrades to the design-only FALLBACK rows
    (devices the alignment did not cover) and the freeze is recorded in
    ``gaps``. Within-row x order/spacing remain free either way. The default
    (False) is BYTE-IDENTICAL to no balancing (pinned by
    tests/test_placement_quality.py).

    SAME-SPEC CLUSTERING (P4a -- always on, tie-scoped): the design-only
    seriation prefers a same-``(kind, W, L)`` unit when extending the chain,
    ON CONNECTIVITY TIES ONLY (connectivity still dominates; a given
    ``style_order`` is never reordered -- style intent outranks clustering).

    IN-ROW 2D PACKING (R8 -- opt-in, objective-driven): ``pack_rows_2d=True``
    runs :func:`_pack_name_rows_2d` on the FINAL row partition (after style
    freeze / balance): short, unpartnered members stack as two-level columns
    inside their own row, narrowing the row while the row HEIGHT bound (the
    tallest member) is hard and the MEMBERSHIP is untouched by construction
    (R3 style fidelity outranks packing). A stacked pair's sketch bboxes
    overlap on purpose (the ``_STACK_SUBLEVEL_DY_UM`` structural marker --
    an align:x column for the band solve, which materializes the real gap
    from sizes + min_gap). Every stack is recorded in ``gaps``. The default
    (False) is BYTE-IDENTICAL to no packing (pinned by
    tests/test_placement_quality.py).

    OUTLINE COMPACTION (P0 -- opt-in, objective-driven): ``compact=True``
    attaches the design-derived
    :class:`~clay.layout.extract.compaction.CompactionSpec` (the emitted row
    partition + same-row partner pairs) to the sketch; the band solve then
    runs the post-stack outline press
    (:func:`clay.layout.extract.compaction.compact_placement` -- x/y
    alternating constraint-graph compaction, intent edges in the graph:
    order preserved, channels floored, rail seams flush, partner offsets
    pinned; minimal-perturbation clamp, only binding chains move). The
    sketch GEOMETRY is untouched (the spec is solve-side); the request is
    recorded in ``gaps``. The default (False) attaches no spec and is
    BYTE-IDENTICAL end to end (pinned by tests/test_compaction.py).

    FULL-2D FACE (2026-06-12 -- the band "row owns the full width"
    retirement): ``compact='2d'`` attaches the spec with ``mode='2d'``: the
    press directs every x-overlapping CROSS-row y edge by the ROW ORDER
    (hard relative-order intent; x-disjoint cross-row pairs carry no y edge
    -- the embedding freedom), and the solve adds the deterministic
    :func:`clay.layout.extract.compaction.embed_search` stage (cross-row
    x-repack moves of partner-rigid groups, every trial legalized by the
    SAME press, accepted only on strict outline-area gain; accepted moves
    are reported on ``HierGraph.compaction_notes``). Row MEMBERSHIP, partner
    offsets, channels and rail corridors are preserved by construction;
    the within-row x ORDER of a repacked row is honestly given up (the row
    weakens to a relative-order constraint -- the '2d' contract, pinned by
    tests/test_compact2d.py). ``compact=True`` keeps the historical
    press-only semantics; any other value raises ValueError.
    """
    if str(engine) not in ("bands", "relax"):
        raise ValueError(
            f"derive_design_sketch: unknown placement engine {engine!r} "
            "(expected 'bands' or 'relax')"
        )
    if mirror_pairs_rigid and str(engine) != "relax":
        raise ValueError(
            "derive_design_sketch: mirror_pairs_rigid re-poses declared "
            "diff/mirror pairs inside the relax refinement stage; it "
            "requires engine='relax'"
        )
    compact_mode = _compact_mode(compact)
    units, gaps = _derive_units(intent)
    w = _unit_connectivity(intent, units)
    chain = _seriate_units(units, w, _spec_keys(intent))

    def size_of(name: str) -> _XY:
        s = sizes.get(name)
        if s is None:
            raise KeyError(
                f"derive_design_sketch: no target size for {name!r} -- sizes must "
                "cover every device/passive (build them from the PDK cells)"
            )
        return (float(s[0]), float(s[1]))

    def unit_width(u: _Unit) -> float:
        ws = [size_of(m)[0] for m in u.members]
        return sum(ws) + dev_gap_um * (len(ws) - 1)

    total_area = 0.0
    for u in units:
        for m in u.members:
            mw, mh = size_of(m)
            total_area += mw * mh
    widest = max((unit_width(u) for u in units), default=0.0)
    if aspect_target is not None:
        if not float(aspect_target) > 0.0:
            raise ValueError(
                f"derive_design_sketch: aspect_target must be a positive "
                f"width/height ratio, got {aspect_target!r}"
            )
        # seed W = sqrt(area * aspect), then the deterministic fixed-point
        # refinement against the ACHIEVED packed outline (rows under-fill).
        row_w_target = _aspect_row_w_target(
            chain, unit_width, size_of, total_area, widest,
            float(aspect_target), dev_gap_um, row_channel_um,
        )
    else:
        row_w_target = max(total_area ** 0.5, widest)

    if style_order is not None:
        return _derive_style_sketch(
            intent, chain, unit_width, row_w_target, size_of,
            dev_gap_um, row_channel_um, style_order, style_objective,
            rails=rails, channel_track_um=channel_track_um,
            balance_rows=balance_rows, pack_rows_2d=pack_rows_2d,
            compact_mode=compact_mode, engine=str(engine),
            mirror_rigid=bool(mirror_pairs_rigid),
        )

    rows = _pack_chain_into_rows(chain, unit_width, row_w_target, dev_gap_um)
    names_rows = [[m for u in row_units for m in u.members] for row_units in rows]
    if balance_rows:
        names_rows, bal_notes = _balance_name_rows(
            intent, names_rows, size_of, dev_gap_um, row_channel_um,
            _rail_overhead_um(rails),
        )
        gaps.extend(bal_notes)
    pk_stacks: Optional[List[Dict[str, str]]] = None
    if pack_rows_2d:
        pk_stacks, pk_notes = _pack_name_rows_2d(
            intent, names_rows, size_of, dev_gap_um
        )
        gaps.extend(pk_notes)
        if not any(pk_stacks):
            pk_stacks = None
    names_rows, rail_sizes, rail_resv, rail_notes = _inject_rail_rows(
        names_rows, rails, size_of, dev_gap_um
    )
    size_of_r = (lambda n: rail_sizes[n] if n in rail_sizes else size_of(n)) \
        if rail_sizes else size_of
    demands = _channel_demands(intent, names_rows)
    bboxes, out_rows = _emit_name_rows(
        names_rows, size_of_r, dev_gap_um, row_channel_um,
        channel_per_gap=_channel_per_gap(demands, row_channel_um,
                                         channel_track_um),
        stacks_per_row=_align_stacks_to_rows(names_rows, pk_stacks),
    )
    dev_rows = tuple(r for r in out_rows
                     if not any(n.startswith(RAIL_ROW_PREFIX) for n in r))
    comp_spec = None
    if compact_mode is not None:
        comp_spec = _build_compaction_spec(intent, out_rows, compact_mode)
        gaps.append(_compaction_gap_note(compact_mode))
    return DesignSketch(bboxes=bboxes, rows=dev_rows,
                        gaps=tuple(gaps + rail_notes),
                        rail_reservations=tuple(rail_resv),
                        channel_demands=demands,
                        compaction_spec=comp_spec,
                        engine=str(engine),
                        mirror_rigid=bool(mirror_pairs_rigid))


def _derive_style_sketch(
    intent: Any,
    chain: Sequence[_Unit],
    unit_width,
    row_w_target: float,
    size_of,
    dev_gap_um: float,
    row_channel_um: float,
    style_order: Any,
    style_objective: bool,
    rails: Optional[List[Dict[str, Any]]] = None,
    channel_track_um: Optional[float] = None,
    balance_rows: bool = False,
    pack_rows_2d: bool = False,
    compact_mode: Optional[str] = None,
    engine: str = "bands",
    mirror_rigid: bool = False,
) -> DesignSketch:
    """The style-conditioned sketch: SAME machinery, style parameter face.

    Row partition / within-row order come from ``style_order.rows`` (names
    only); unaligned devices fall back to the design-only seriated chain and
    are recorded in ``gaps``. With ``style_objective`` the deterministic
    candidate set below is scored by ``constraint_graph_distance`` vs the
    style and the minimum wins. Always ``provenance='style-transfer'``.
    """
    from clay.layout.extract.style_order import rows_to_relative_order
    from clay.layout.seqpair_model import constraint_graph_distance

    known = set()
    for d in getattr(intent, "devices", []) or []:
        known.add(str(d.name))
    for p in getattr(intent, "passives", []) or []:
        known.add(str(p.name))

    style_rows: List[List[str]] = []
    covered: set = set()
    for row in getattr(style_order, "rows", ()) or ():
        names = [str(n) for n in row if str(n) in known and str(n) not in covered]
        if names:
            style_rows.append(names)
            covered.update(names)

    gaps: List[str] = []
    fallback_rows: List[List[str]] = []
    unaligned = sorted(known - covered)
    if unaligned:
        un_set = set(unaligned)
        # design-only seriation restricted to the unaligned devices: keep whole
        # units when fully unaligned, demote partially-aligned units' leftovers
        # to singletons (deterministic, recorded).
        sub_chain: List[_Unit] = []
        for u in chain:
            left = [m for m in u.members if m in un_set]
            if not left:
                continue
            if len(left) == len(u.members):
                sub_chain.append(u)
            else:
                for m in left:
                    sub_chain.append(_Unit(key=m, members=(m,), why=u.why))
        packed = _pack_chain_into_rows(
            sub_chain, unit_width, row_w_target, dev_gap_um
        )
        fallback_rows = [
            [m for u in row_units for m in u.members]
            for row_units in packed if row_units
        ]
        for n in unaligned:
            gaps.append(
                f"style-transfer fallback: {n} not aligned to the reference "
                "order; placed by design-only seriation above the style rows"
            )

    candidates: List[Tuple[str, List[List[str]]]] = [
        ("style-following", style_rows + fallback_rows)
    ]
    if style_objective:
        # deterministic design-packing variants: width tier x chain start x row
        # polarity (3 x 2 x 2 = 12) -- plus the style-following candidate above.
        # No randomness, no clock; ties resolve to the FIRST minimal candidate.
        for wf in (0.85, 1.0, 1.15):
            for start in ("head", "tail"):
                for polarity in ("up", "down"):
                    ch = list(chain) if start == "head" else list(reversed(chain))
                    packed = _pack_chain_into_rows(
                        ch, unit_width, row_w_target * wf, dev_gap_um
                    )
                    nr = [
                        [m for u in row_units for m in u.members]
                        for row_units in packed if row_units
                    ]
                    if polarity == "down":
                        nr = list(reversed(nr))
                    candidates.append((f"design wf={wf} {start} {polarity}", nr))

        cell = str(getattr(style_order, "cell", "")) or "style"
        best_idx = 0
        best_d: Optional[float] = None
        for idx, (_tag, nr) in enumerate(candidates):
            cand_order = rows_to_relative_order(cell, nr)
            d = constraint_graph_distance(cand_order, style_order)
            if best_d is None or d < best_d:
                best_idx, best_d = idx, d
        tag, names_rows = candidates[best_idx]
        gaps.append(
            f"style-objective: candidate '{tag}' selected, constraint-graph "
            f"distance {best_d:.4f} over {len(candidates)} deterministic "
            "candidates"
        )
    else:
        names_rows = candidates[0][1]

    if balance_rows:
        # R3 STYLE-ROW FIDELITY (hard constraint): with a style_order present
        # the row membership IS the style partition -- balance (an area
        # objective) may never move a member across a style row or split one;
        # style fidelity outranks fill by contract. Only the design-only
        # FALLBACK rows (stacked above the style rows) stay balanceable
        # among themselves. The freeze is recorded -- never silent.
        if names_rows is candidates[0][1]:
            n_style = len(style_rows)
            frozen = [list(r) for r in names_rows[:n_style]]
            movable = [list(r) for r in names_rows[n_style:]]
            if len(movable) >= 2:
                movable, bal_notes = _balance_name_rows(
                    intent, movable, size_of, dev_gap_um, row_channel_um,
                    _rail_overhead_um(rails),
                )
                gaps.extend(bal_notes)
            names_rows = frozen + [list(r) for r in movable if r]
            gaps.append(
                f"row-balance: {n_style} style row(s) frozen (style row "
                "membership is a hard constraint; balance restricted to "
                "the design-only fallback rows)"
            )
        else:
            # a style-objective design-packing candidate won: its rows carry
            # the style-distance objective, not a balanceable design claim.
            gaps.append(
                "row-balance: skipped (style-objective selected a "
                "design-packing candidate; its rows are scored against the "
                "style order and are not rebalanced)"
            )
    pk_stacks: Optional[List[Dict[str, str]]] = None
    if pack_rows_2d:
        # R8: packing acts on the FINAL membership partition (style rows
        # frozen by R3 + any fallback rows) -- geometry inside a row only,
        # membership untouched by construction.
        pk_stacks, pk_notes = _pack_name_rows_2d(
            intent, names_rows, size_of, dev_gap_um
        )
        gaps.extend(pk_notes)
        if not any(pk_stacks):
            pk_stacks = None
    names_rows, rail_sizes, rail_resv, rail_notes = _inject_rail_rows(
        names_rows, rails, size_of, dev_gap_um
    )
    gaps.extend(rail_notes)
    size_of_r = (lambda n: rail_sizes[n] if n in rail_sizes else size_of(n)) \
        if rail_sizes else size_of
    demands = _channel_demands(intent, names_rows)
    bboxes, out_rows = _emit_name_rows(
        names_rows, size_of_r, dev_gap_um, row_channel_um,
        channel_per_gap=_channel_per_gap(demands, row_channel_um,
                                         channel_track_um),
        stacks_per_row=_align_stacks_to_rows(names_rows, pk_stacks),
    )
    dev_rows = tuple(r for r in out_rows
                     if not any(n.startswith(RAIL_ROW_PREFIX) for n in r))
    comp_spec = None
    if compact_mode is not None:
        comp_spec = _build_compaction_spec(intent, out_rows, compact_mode)
        gaps.append(_compaction_gap_note(compact_mode))
    return DesignSketch(
        bboxes=bboxes, rows=dev_rows, gaps=tuple(gaps),
        provenance="style-transfer",
        rail_reservations=tuple(rail_resv),
        channel_demands=demands,
        compaction_spec=comp_spec,
        engine=str(engine),
        mirror_rigid=bool(mirror_rigid),
    )


def sketch_relative_order(sketch: DesignSketch) -> "Any":
    """The sketch's row structure as a :class:`RelativeOrder` (names only) --
    the comparable form for ``constraint_graph_distance`` scoring against a
    reference style order (same row->relation convention on both sides)."""
    from clay.layout.extract.style_order import rows_to_relative_order

    return rows_to_relative_order("sketch", sketch.rows)
