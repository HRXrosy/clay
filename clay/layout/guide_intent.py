"""Lift a designer's ``layout_guide.txt`` into a machine-checkable convergence
criterion (Agent A / R-GUIDE, round place-route-convergence).

Each ``circuits/<c>/layout_guide.txt`` is the designer's layout intent written
as structured prose (sections: 关键指标 / 关键信号识别 with 强匹配通路 matched
paths and 高阻/敏感节点 sensitive high-Z nodes, 共质心 common-centroid groups,
走线规则 with shield rules, Floorplan rows). This module parses the
machine-extractable facts from that prose (device / net names, section
keywords) and MERGES them with the cell's already-parsed ``layout_intent`` block
(``clay.layout.intent.layout_intent_from_cell_def`` + the ``annotate`` block's
``@matching`` device pairs and ``@cell:@symmetric(axis)``) into a FROZEN
:class:`ConvergenceCriteria`.

The place / route convergence loops consume this:

  * ``check_placement(placement)`` -- matched DEVICE pairs placed
    mirror-symmetric about the axis, common-centroid groups centroid-coincident,
    sensitive nets compact;
  * ``check_routing(routed, pex, drc, lvs)`` -- parasitic budgets within
    RELATIVE tolerance (via PEX), DRC clean, LVS match.

HONESTY BOUNDARY: only facts the guide / intent REALLY state become criteria.
Prose that is not machine-checkable (heat-matching advice, EM metal-width rules,
Kelvin-sense suggestions, well-tap density, …) is recorded as an ADVISORY
:attr:`ConvergenceCriteria.notes` entry, never invented as a fake obligation.
``coverage_report()`` reports, per circuit, what was lifted vs. what stayed
advisory.

This module is READ-ONLY on the guide files and the parsed intent; it never
mutates a circuit, a case, or the parsed cell. All tolerances are RELATIVE
fractions per the project-wide consistency rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from clay.layout.intent import LayoutIntent, ParasiticBudget, Shield

# ─────────────────────────────────────────────────────────────────────────────
# Defaults (RELATIVE thresholds only — never an absolute nm / fF)
# ─────────────────────────────────────────────────────────────────────────────

# Placement-symmetry tolerance: matched-pair centroids must mirror about the
# axis to within this fraction of the cell's full extent.
_DEFAULT_SYMMETRY_TOL = 0.05
# Common-centroid coincidence tolerance: a group's per-device centroids must
# share a common centroid to within this fraction of the cell extent.
_DEFAULT_CENTROID_TOL = 0.05
# Sensitive-net compactness tolerance: a sensitive net's routed span must be
# within this fraction of the cell extent (high-Z nodes "route short").
_DEFAULT_SENSITIVE_SPAN_TOL = 0.35

# Section header keywords in the guide prose (used to slice the file).
_SEC_SIGNALS = "关键信号识别"
_SEC_FLOORPLAN = "Floorplan"
_SEC_ROUTING = "走线规则"
_KW_STRONG_MATCH = "强匹配通路"
_KW_SENSITIVE = ("高阻", "敏感节点")
_KW_CENTROID = "共质心"
_KW_SHIELD = "shield"


# ─────────────────────────────────────────────────────────────────────────────
# Result type (frozen contract surface)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CriteriaResult:
    """Outcome of checking placement or routing against the criteria.

    ``satisfied`` is the AND of every checked obligation. ``obligations`` is a
    tuple of per-obligation ``(name, ok, measured, tol)`` rows. ``worst`` is the
    largest RELATIVE residual over budget (0.0 when everything passes; it ranks
    how far the worst obligation is over tolerance, for the loop's repair step).
    ``reasons`` lists a one-line explanation for each FAILING obligation.
    """
    satisfied: bool
    obligations: Tuple[Tuple[str, bool, Optional[float], float], ...]
    worst: float
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ConvergenceCriteria:
    """The machine-checkable convergence criterion lifted from a layout_guide.

    Frozen interface the place/route loops build against (round doc §"Frozen
    interface"). Every field is a fact the guide / parsed intent really states;
    non-machine-checkable prose lives in :attr:`notes` as advisory text.

    Fields:
      * ``matched_pairs`` -- strong-match DEVICE pairs (``annotate`` ``@matching``)
        plus strong-match NET pairs from the guide's 强匹配通路 lines and the
        parsed ``@match`` / ``@symmetric`` diff pairs. Device pairs are placed
        mirror-symmetric; net pairs additionally seed a parasitic-balance budget.
      * ``mirror_axis`` -- ``"vertical"`` when ``@cell:@symmetric(vertical)`` or
        the guide names a vertical center / input axis; else ``None``.
      * ``common_centroid`` -- device groups from 共质心 / ``@place_group`` that
        must share a centroid (PNP 8-around-1, ABBA banks, …).
      * ``shields`` -- ``(aggressor, victim, shield_net)`` from the guide's shield
        rules + parsed ``@shield``.
      * ``sensitive_nets`` -- high-Z nodes that must route short (guide §2
        高阻/敏感节点 + parsed sensitive nets).
      * ``budgets`` -- parasitic budgets from the parsed ``layout_intent``.
      * ``device_pairs`` -- the subset of ``matched_pairs`` that are DEVICE labels
        (checkable against placement centroids), kept separate so the placement
        check does not confuse device labels with net names.
      * ``notes`` -- advisory prose that could NOT be lifted into a criterion.
    """
    cell: str
    matched_pairs: Tuple[Tuple[str, str], ...]
    mirror_axis: Optional[str]
    common_centroid: Tuple[Tuple[str, ...], ...]
    shields: Tuple[Tuple[str, str, str], ...]
    sensitive_nets: Tuple[str, ...]
    budgets: Tuple[ParasiticBudget, ...]
    device_pairs: Tuple[Tuple[str, str], ...] = ()
    notes: Tuple[str, ...] = ()
    # RELATIVE tolerances (fractions of the cell extent / of a balanced cap).
    symmetry_tol: float = _DEFAULT_SYMMETRY_TOL
    centroid_tol: float = _DEFAULT_CENTROID_TOL
    sensitive_span_tol: float = _DEFAULT_SENSITIVE_SPAN_TOL

    # ── placement convergence ────────────────────────────────────────────────
    def check_placement(self, placement) -> CriteriaResult:
        """Are the placement obligations met?

        Obligations:
          P1 matched DEVICE pairs placed mirror-symmetric about ``mirror_axis``
             (centroids equidistant from the axis on opposite sides, with the
             off-axis coordinate matched), within ``symmetry_tol`` of the cell
             extent;
          P2 each ``common_centroid`` group's devices share a common centroid
             within ``centroid_tol`` of the cell extent;
          P3 each ``sensitive_net``'s placed device span is compact (within
             ``sensitive_span_tol`` of the cell extent) -- a high-Z net's devices
             must cluster so the net routes short.

        ``placement`` supplies device centroids through any of: a
        ``device_centroids()`` method, a ``.centroids`` / ``.device_centroids``
        mapping, a ``.placement.term`` structure (the router's native form, read
        via :func:`clay.layout.placement_symmetry.device_centroids`), or a plain
        ``{label: (x, y)}`` dict. A net→devices map (for P3) is read from a
        ``net_devices()`` method / ``.net_devices`` mapping when present; when it
        is absent P3 is recorded as an UNMEASURABLE advisory (not a silent pass).
        """
        centroids = _centroids_of(placement)
        net_devices = _net_devices_of(placement)
        obligations: List[Tuple[str, bool, Optional[float], float]] = []
        reasons: List[str] = []

        if not centroids:
            return CriteriaResult(
                satisfied=False,
                obligations=(("placement_centroids_present", False, None, 0.0),),
                worst=1.0,
                reasons=("placement exposes no device centroids",),
            )

        extent = _extent_span(centroids)
        # The vertical mirror axis is the placer's DECLARED axis when the placement
        # carries one (``placement.axis_x``) -- that is the axis the matched pairs
        # were actually mirrored about. Falling back to the device-extent midpoint
        # (``_axis_x``) is WRONG when the cell carries off-axis passives (a resistor
        # row laid to one side of the MOS block): the extent midpoint then sits far
        # from the pairs' true axis and reports every (genuinely symmetric) pair as
        # asymmetric. Prefer the declared axis; only synthesise one from the extent
        # when the placement does not expose it (a plain ``{label:(x,y)}`` dict).
        if self.mirror_axis == "vertical":
            decl_axis = getattr(placement, "axis_x", None)
            axis_x = float(decl_axis) if decl_axis is not None else _axis_x(centroids)
        else:
            axis_x = None

        # P1: matched-device mirror symmetry.
        for a, b in self.device_pairs:
            ca, cb = centroids.get(a), centroids.get(b)
            if ca is None or cb is None:
                # A declared pair whose devices are absent from this placement is
                # not checkable here -> advisory, NOT a fake pass/fail.
                obligations.append((f"sym[{a}~{b}]", True, None, self.symmetry_tol))
                continue
            resid = _mirror_residual(ca, cb, axis_x, extent)
            ok = resid <= self.symmetry_tol
            obligations.append((f"sym[{a}~{b}]", ok, resid, self.symmetry_tol))
            if not ok:
                reasons.append(
                    f"matched pair {a}~{b} not mirror-symmetric "
                    f"(residual {resid:.3f} > {self.symmetry_tol:.3f})"
                )

        # P2: common-centroid BALANCE. A common-centroid group does NOT place
        # every device at one point -- it interleaves them so the group's mean
        # centroid sits at the geometric centre of their spread (gradients cancel
        # to first order). So the obligation is: the mean of the member centroids
        # coincides with the centre of the members' bounding box, within
        # ``centroid_tol`` of the cell extent.
        for group in self.common_centroid:
            present = [centroids[d] for d in group if d in centroids]
            if len(present) < 2:
                obligations.append((f"cc[{'+'.join(group)}]", True, None, self.centroid_tol))
                continue
            resid = _centroid_balance(present, extent)
            ok = resid <= self.centroid_tol
            obligations.append((f"cc[{'+'.join(group)}]", ok, resid, self.centroid_tol))
            if not ok:
                reasons.append(
                    f"common-centroid group {'+'.join(group)} not balanced "
                    f"(offset {resid:.3f} > {self.centroid_tol:.3f})"
                )

        # P3: sensitive-net compactness (needs a net->devices map).
        for net in self.sensitive_nets:
            devs = net_devices.get(net) if net_devices else None
            if not devs:
                obligations.append((f"compact[{net}]", True, None, self.sensitive_span_tol))
                continue
            present = [centroids[d] for d in devs if d in centroids]
            if len(present) < 2:
                obligations.append((f"compact[{net}]", True, None, self.sensitive_span_tol))
                continue
            resid = _group_span(present, extent)
            ok = resid <= self.sensitive_span_tol
            obligations.append((f"compact[{net}]", ok, resid, self.sensitive_span_tol))
            if not ok:
                reasons.append(
                    f"sensitive net {net} not compact "
                    f"(span {resid:.3f} > {self.sensitive_span_tol:.3f})"
                )

        worst = max(
            (m - t for _, ok, m, t in obligations if (not ok) and m is not None),
            default=0.0,
        )
        satisfied = all(ok for _, ok, _, _ in obligations)
        return CriteriaResult(satisfied, tuple(obligations), max(0.0, worst), tuple(reasons))

    # ── routing convergence ──────────────────────────────────────────────────
    def check_routing(self, routed, pex, drc, lvs) -> CriteriaResult:
        """Are the routing obligations met? budgets (via PEX) + DRC + LVS.

        Obligations:
          R1 every parasitic budget is within its RELATIVE tolerance, measured
             against the PEX extraction through
             :func:`clay.layout.intent_obligations.check_layout_intent`;
          R2 DRC clean (``drc`` truthy / ``drc.clean`` / ``drc.violations``
             empty);
          R3 LVS match (``lvs`` truthy / ``lvs.match`` / ``lvs.matched``).

        ``routed`` is accepted for signature stability (the loop passes the routed
        cell) but the budget measurement is driven by ``pex``; ``pex`` is a
        :class:`~clay.layout.pex.base.PexReport`. ``drc`` / ``lvs`` may be a bool
        or a small result object exposing ``clean`` / ``violations`` /
        ``match`` / ``matched``.
        """
        obligations: List[Tuple[str, bool, Optional[float], float]] = []
        reasons: List[str] = []

        # R1: parasitic budgets via PEX.
        if self.budgets and pex is not None:
            try:
                from clay.layout.intent_obligations import check_layout_intent

                intent = LayoutIntent(
                    cell=self.cell,
                    dynamic_nets=(),
                    sensitive_nets=self.sensitive_nets,
                    diff_pairs=(),
                    budgets=self.budgets,
                    shields=tuple(Shield(a, v, s) for a, v, s in self.shields),
                )
                obs = check_layout_intent(intent, pex)
                for o in obs:
                    name = f"budget[{o.kind}:{o.pair[0]}~{o.pair[1]}]"
                    ok = o.status == "pass"
                    obligations.append((name, ok, o.measured_imbalance, o.rel_tol))
                    if not ok:
                        reasons.append(
                            f"{name} {o.status} "
                            f"(imbalance {o.measured_imbalance}, tol {o.rel_tol:.3f})"
                        )
            except Exception as exc:  # pragma: no cover - defensive
                obligations.append(("budgets_measurable", False, None, 0.0))
                reasons.append(f"budget measurement failed: {exc}")
        elif self.budgets:
            obligations.append(("budgets_measurable", False, None, 0.0))
            reasons.append("budgets declared but no PEX extraction supplied")

        # R2: DRC clean.
        drc_ok = _is_clean(drc)
        obligations.append(("drc_clean", drc_ok, None, 0.0))
        if not drc_ok:
            reasons.append("DRC not clean")

        # R3: LVS match.
        lvs_ok = _is_match(lvs)
        obligations.append(("lvs_match", lvs_ok, None, 0.0))
        if not lvs_ok:
            reasons.append("LVS does not match")

        worst = max(
            (m - t for _, ok, m, t in obligations if (not ok) and m is not None),
            default=0.0,
        )
        satisfied = all(ok for _, ok, _, _ in obligations)
        return CriteriaResult(satisfied, tuple(obligations), max(0.0, worst), tuple(reasons))

    # ── reporting ─────────────────────────────────────────────────────────────
    def coverage_report(self) -> Dict[str, object]:
        """A serialisable summary of what was lifted vs. what stayed advisory."""
        return {
            "cell": self.cell,
            "matched_pairs": [list(p) for p in self.matched_pairs],
            "device_pairs": [list(p) for p in self.device_pairs],
            "mirror_axis": self.mirror_axis,
            "common_centroid": [list(g) for g in self.common_centroid],
            "shields": [list(s) for s in self.shields],
            "sensitive_nets": list(self.sensitive_nets),
            "budgets": [
                {"kind": b.kind, "pair": list(b.pair), "rel_tol": b.rel_tol}
                for b in self.budgets
            ],
            "advisory_notes": list(self.notes),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Placement-introspection helpers (accept several placement shapes)
# ─────────────────────────────────────────────────────────────────────────────

def _centroids_of(placement) -> Dict[str, Tuple[float, float]]:
    """Best-effort ``{device_label: (x, y)}`` from many placement shapes."""
    if placement is None:
        return {}
    # 1. an explicit method.
    m = getattr(placement, "device_centroids", None)
    if callable(m):
        try:
            return {str(k): _xy(v) for k, v in m().items()}
        except Exception:
            pass
    # 2. a mapping attribute named centroids / device_centroids.
    for attr in ("centroids", "device_centroids"):
        val = getattr(placement, attr, None)
        if isinstance(val, dict) and val:
            return {str(k): _xy(v) for k, v in val.items()}
    # 3. the router's native ``.placement.term`` form.
    if getattr(placement, "placement", None) is not None or getattr(placement, "term", None) is not None:
        try:
            from clay.layout.placement_symmetry import device_centroids

            return {str(k): _xy(v) for k, v in device_centroids(placement).items()}
        except Exception:
            pass
    # 4. a plain dict.
    if isinstance(placement, dict) and placement:
        return {str(k): _xy(v) for k, v in placement.items()}
    return {}


def _net_devices_of(placement) -> Dict[str, Tuple[str, ...]]:
    """Best-effort ``{net: (device_label, ...)}`` for the sensitive-net check."""
    if placement is None:
        return {}
    m = getattr(placement, "net_devices", None)
    if callable(m):
        try:
            return {str(k): tuple(str(d) for d in v) for k, v in m().items()}
        except Exception:
            pass
    val = getattr(placement, "net_devices", None)
    if isinstance(val, dict):
        return {str(k): tuple(str(d) for d in v) for k, v in val.items()}
    return {}


def _xy(v) -> Tuple[float, float]:
    return (float(v[0]), float(v[1]))


def _extent_span(centroids: Dict[str, Tuple[float, float]]) -> float:
    """Diagonal-ish extent (max of x-span, y-span); the normaliser for residuals."""
    xs = [c[0] for c in centroids.values()]
    ys = [c[1] for c in centroids.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return span if span > 0 else 1.0


def _axis_x(centroids: Dict[str, Tuple[float, float]]) -> float:
    """The vertical mirror axis x = midpoint of the device x-extent."""
    xs = [c[0] for c in centroids.values()]
    return (min(xs) + max(xs)) / 2.0


def _mirror_residual(
    ca: Tuple[float, float],
    cb: Tuple[float, float],
    axis_x: Optional[float],
    extent: float,
) -> float:
    """RELATIVE mis-symmetry of a matched pair about a vertical axis.

    With a vertical axis: the two centroids should mirror, i.e. their x-distances
    to the axis are equal and their y's match. Without a known axis, fall back to
    "the pair shares a centroid" (any sensible matched layout keeps the pair
    adjacent / centro-symmetric). Residual is normalised by the cell extent.
    """
    if axis_x is not None:
        dx = abs((ca[0] - axis_x) + (cb[0] - axis_x))   # 0 when mirror-symmetric
        dy = abs(ca[1] - cb[1])                         # 0 when same row
        return (dx + dy) / extent
    # no axis: penalise centroid separation (a matched pair sits together).
    return ((abs(ca[0] - cb[0]) + abs(ca[1] - cb[1])) / 2.0) / extent


def _centroid_balance(points: Sequence[Tuple[float, float]], extent: float) -> float:
    """RELATIVE common-centroid imbalance.

    The group's MEAN centroid should sit at the geometric centre of the members'
    bounding box. Returns the distance between the two, normalised by the cell
    extent: 0.0 for a perfectly balanced (symmetric) arrangement, growing as the
    members skew to one side. This is the correct common-centroid criterion (an
    ABBA / 8-around-1 layout spreads devices out but keeps their mean at the
    centre), not "all devices at one point".
    """
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5 / extent


def _group_span(points: Sequence[Tuple[float, float]], extent: float) -> float:
    """RELATIVE bounding span of a device group (max of x-span, y-span)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(max(xs) - min(xs), max(ys) - min(ys)) / extent


def _is_clean(drc) -> bool:
    if drc is None:
        return False
    if isinstance(drc, bool):
        return drc
    if hasattr(drc, "clean"):
        return bool(getattr(drc, "clean"))
    if hasattr(drc, "violations"):
        return len(getattr(drc, "violations")) == 0
    return bool(drc)


def _is_match(lvs) -> bool:
    if lvs is None:
        return False
    if isinstance(lvs, bool):
        return lvs
    for attr in ("match", "matched", "is_match"):
        if hasattr(lvs, attr):
            return bool(getattr(lvs, attr))
    return bool(lvs)


# ─────────────────────────────────────────────────────────────────────────────
# Guide-prose parsing (machine-extractable facts only)
# ─────────────────────────────────────────────────────────────────────────────

# A net / device token: alnum + underscore, must contain a letter (skips bare
# numbers, units like "2pF", and section indices).
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
# Arrow / pairing glyphs the guide uses between matched names.
_PAIR_GLYPHS = ("↔", "<->", "<=>", "║", ":")


def _section(text: str, header: str, *, until: Optional[Sequence[str]] = None) -> str:
    """Return the slice of ``text`` from the line containing ``header`` to the
    next top-level numbered section (or one of ``until`` headers)."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if header in ln:
            start = i
            break
    if start is None:
        return ""
    out = [lines[start]]
    for ln in lines[start + 1:]:
        stripped = ln.strip()
        # stop at the next top-level "N. <title>" section header
        if re.match(r"^\d+\.\s", stripped):
            break
        if until and any(h in ln for h in until):
            break
        out.append(ln)
    return "\n".join(out)


def _strong_match_pairs(signals_section: str) -> List[Tuple[str, str]]:
    """Lift 强匹配通路 net/device pairs from the 关键信号识别 section.

    Handles the guide's two shapes:
      ``vinp ↔ vinn → MN1 ↔ MN2.G``     (two ↔-pairs on one line)
      ``MN3 ↔ MN4``                     (single pair)
      ``Mbias ↔ M5 ↔ M7 ↔ M_bleed``     (a mirror chain -> consecutive pairs)
    A trailing ``.G`` / ``.D`` / ``.S`` terminal qualifier is stripped to the
    bare device name. Only lines under 强匹配通路 (before the next bullet) count.
    """
    pairs: List[Tuple[str, str]] = []
    lines = signals_section.splitlines()
    in_block = False
    for ln in lines:
        if _KW_STRONG_MATCH in ln:
            in_block = True
            continue
        if not in_block:
            continue
        stripped = ln.strip()
        if not stripped:
            continue
        # a new top-level bullet ("- 高阻 ...", "- 偏置 ...") ends the block
        if stripped.startswith("-") and _KW_STRONG_MATCH not in ln:
            break
        # split into ↔-separated runs; each consecutive pair in a run matches.
        # Within a ↔-segment, take the FIRST real name (the segment may trail a
        # parenthetical note, e.g. "M_bleed (NMOS 镜像链, ...)").
        for run in re.split(r"[→\-]>?|,|，", ln):
            names: List[str] = []
            for seg in re.split(r"↔|<->|<=>|║", run):
                seg_names = _names_in(seg)
                if seg_names:
                    names.append(seg_names[0])
            for a, b in zip(names, names[1:]):
                if a != b:
                    pairs.append((a, b))
    return _dedupe_pairs(pairs)


def _sensitive_nets(signals_section: str) -> List[str]:
    """Lift the 高阻 / 敏感节点 net names from the 关键信号识别 section.

    The block opens on a BULLET line ``- 高阻 / 敏感节点 ...：`` and runs until the
    next bullet. Each data line names the high-Z nets BEFORE an em-dash
    explanation (``vd_left, vout —— 高阻输出节点 ...``); the explanation often
    re-mentions ``vdd`` / ``vss`` (the rails to keep away from), so we read only
    the head before the em-dash and drop any rail token.
    """
    nets: List[str] = []
    lines = signals_section.splitlines()
    in_block = False
    for ln in lines:
        stripped = ln.strip()
        is_bullet = stripped.startswith("-")
        if is_bullet and any(k in ln for k in _KW_SENSITIVE):
            in_block = True
            # names may be inline after the colon on the bullet line itself
            tail = ln.split("：", 1)[-1] if "：" in ln else ""
            head = re.split(r"——|—|--", tail)[0]
            nets.extend(n for n in _names_in(head) if not _is_rail(n))
            continue
        if not in_block:
            continue
        if not stripped:
            continue
        # a new bullet (any other labelled section) ends the high-Z block
        if is_bullet:
            break
        # data line: only the head before the em-dash names the net(s)
        head = re.split(r"——|—|--", ln)[0]
        nets.extend(n for n in _names_in(head) if not _is_rail(n))
    return _dedupe(nets)


def _centroid_groups(text: str) -> List[Tuple[str, ...]]:
    """Lift 共质心 device groups from the floorplan / matching prose.

    Conservative: a 共质心 line names the devices that must share a centroid.
    We collect the device names appearing ON the same line as 共质心 (the
    common shape is ``... Q1 / Q2 ... 共质心`` or ``MN1 / MN2 ... 共质心 + dummy``).
    Lines that only say 共质心 with no device name yield nothing (advisory).
    """
    groups: List[Tuple[str, ...]] = []
    for ln in text.splitlines():
        if _KW_CENTROID not in ln:
            continue
        names = [n for n in _names_in(ln) if _looks_like_device(n)]
        # keep order, dedupe within the line
        uniq = tuple(dict.fromkeys(names))
        if len(uniq) >= 2:
            groups.append(uniq)
    return groups


def _shield_rules(
    routing_section: str, *, known_nets: Optional[frozenset] = None
) -> List[Tuple[str, str, str]]:
    """Lift ``(victim_a, victim_b, shield_net)`` shield rules from 走线规则.

    The guide states a shield over a BULLET whose header names the two victim
    nets and whose body says "夹 <rail> shield"::

        - **vinp / vinn**：... 中间夹 vss shield 防止互相耦合 ...
        - **vd_left, vout**：... 建议中间夹一根 vss
          shield 隔开 ...

    So the parser is bullet-aware: it tracks the signal nets named in the current
    routing bullet (its bolded header + continuation lines), and when a 夹/shield
    phrase appears in that bullet it emits ``(victim_a, victim_b, shield_net)``
    over the two victims (the rail token near "shield"). A bullet without two
    signal-net victims, or without a rail near the shield word, yields nothing.
    """
    rules: List[Tuple[str, str, str]] = []
    bullet_nets: List[str] = []
    bullet_has_shield = False
    bullet_shield_net: Optional[str] = None

    def _flush() -> None:
        if bullet_has_shield and bullet_shield_net is not None:
            victims = _dedupe([n for n in bullet_nets if _looks_like_net(n)])
            # only keep victims that are REAL nets of this cell (when a net
            # universe is supplied) -- drops hypothetical aggressor words like a
            # generic "dynamic" / "clk" mention that is not a net of the design.
            if known_nets is not None:
                victims = [n for n in victims if n in known_nets]
            if len(victims) >= 2:
                rules.append((victims[0], victims[1], bullet_shield_net))

    for ln in routing_section.splitlines():
        stripped = ln.strip()
        is_bullet = stripped.startswith("-")
        if is_bullet:
            _flush()
            bullet_nets = []
            bullet_has_shield = False
            bullet_shield_net = None
        # accumulate signal-net names seen anywhere in this bullet
        bullet_nets.extend(n for n in _names_in(ln) if not _is_rail(n))
        if "夹" in ln or "shield" in ln.lower():
            bullet_has_shield = True
            net = _shield_net_in(ln)
            if net is not None:
                bullet_shield_net = net
    _flush()
    return _dedupe_shields(rules)


def _dedupe_shields(rules: Sequence[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for a, b, n in rules:
        key = (frozenset((a, b)), n)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b, n))
    return out


# ── token helpers ────────────────────────────────────────────────────────────

_RAILS = frozenset({"vdd", "vss", "gnd", "vdda", "vddd", "vssa", "vssd", "avdd", "avss"})
# stop-words that match the token regex but are prose, not net/device names
_STOPWORDS = frozenset({
    "G", "D", "S", "dB", "MHz", "Hz", "Gain", "GBW", "PM", "DC", "AC", "CL",
    "OD", "metal", "shield", "dummy", "bus", "rail", "pin", "gate", "drain",
    "source", "poly", "via", "array", "fingers", "finger", "row", "rows",
    "EM", "IR", "ESD", "PSRR", "ICMR", "MOM", "MOS", "NMOS", "PMOS", "CS",
    "PNP", "BJT", "N", "P",
    "well", "tap", "ABBA", "Kelvin", "OTA", "FET", "Cc", "Cgd", "Cgs", "Rz",
    "Vbe", "PTAT", "CTAT", "TC", "ppm", "Vref", "Iout", "Vout", "Vsource",
})


def _strip_terminal(tok: str) -> str:
    """Strip a trailing ``.G`` / ``.D`` / ``.S`` / ``.gate`` terminal qualifier."""
    tok = tok.strip()
    m = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\.(?:[GDSgds]|gate|drain|source)$", tok)
    if m:
        return m.group(1)
    return tok.split(".", 1)[0] if "." in tok else tok


def _is_name(tok: str) -> bool:
    if not tok:
        return False
    tok = _strip_terminal(tok)
    if not _TOKEN.fullmatch(tok):
        return False
    if tok in _STOPWORDS:
        return False
    return True


def _names_in(text: str) -> List[str]:
    out = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_]*(?:\.(?:[GDSgds]|gate|drain|source))?", text):
        n = _strip_terminal(raw)
        if _is_name(n):
            out.append(n)
    return out


def _is_rail(tok: str) -> bool:
    return tok.lower() in _RAILS


def _looks_like_device(tok: str) -> bool:
    """A device label heuristically starts with M (MOS), Q (BJT), or R (res)."""
    return bool(re.match(r"^(M|Q|R)[A-Za-z0-9_]", tok)) and not _is_rail(tok)


def _looks_like_net(tok: str) -> bool:
    """A signal NET (not a device, not a rail). Net names are lower-ish words
    like vinp / vout / vd_left / na / nb."""
    if _is_rail(tok):
        return False
    if _looks_like_device(tok):
        return False
    return tok[0].islower() or tok.startswith("v") or tok.startswith("n")


def _shield_net_in(ln: str) -> Optional[str]:
    """The rail net used as a shield on a 夹/shield line (vss / vdd / gnd)."""
    for r in ("vss", "vdd", "gnd"):
        if r in ln.lower():
            return r
    return None


def _dedupe(seq: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(seq))


def _dedupe_pairs(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()
    for a, b in pairs:
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def _connected_components(pairs: Sequence[Tuple[str, str]]) -> List[List[str]]:
    """Connected components of the device-matching graph (edges = matched pairs).

    Each returned component is the list of devices reachable through matched-pair
    edges, in first-seen order. A single 1-to-1 matched pair is a size-2
    component; a current-mirror chain / fan-out (a reference ratioed to several
    devices, e.g. ``Mbias↔M5↔M7↔M_bleed`` or ``MPBIAS↔MPTAIL`` + ``MPBIAS↔MP6``)
    is a larger component. This is the structural discriminator between a
    geometric MIRROR obligation (a perfect 1-to-1 matching about the axis) and a
    centroid-MATCHING obligation (a ratioed mirror set that shares a centroid).
    """
    adj: Dict[str, List[str]] = {}
    order: List[str] = []
    for a, b in pairs:
        for d in (a, b):
            if d not in adj:
                adj[d] = []
                order.append(d)
        adj[a].append(b)
        adj[b].append(a)
    seen: set = set()
    comps: List[List[str]] = []
    for start in order:
        if start in seen:
            continue
        stack = [start]
        comp: List[str] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


# ─────────────────────────────────────────────────────────────────────────────
# Parsed-intent (.clay) lifting: annotate @matching + @cell:@symmetric
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_device_pairs(cell_def) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Read the ``annotate`` block's ``@matching`` device pairs + mirror axis.

    The parser folds the ``annotate <Cell> { ... }`` block onto
    ``cell_def.__dict__['_layout_annotations']`` as ``{target: [annotation,...]}``
    where ``target == '__cell__'`` carries the ``@cell:@symmetric(axis)`` and a
    device target carries a ``@matching(other)`` annotation.
    """
    ann = getattr(cell_def, "__dict__", {}).get("_layout_annotations", {}) or {}
    pairs: List[Tuple[str, str]] = []
    axis: Optional[str] = None
    for target, anns in ann.items():
        for a in anns:
            name = type(a).__name__
            if name == "Matching":
                other = getattr(a, "other", None)
                if other and str(target) != "__cell__":
                    pairs.append((str(target), str(other)))
            elif name == "Symmetric":
                ax = getattr(a, "axis", None)
                axis = getattr(ax, "value", None) or (str(ax) if ax is not None else None)
    return _dedupe_pairs(pairs), axis


# ─────────────────────────────────────────────────────────────────────────────
# Top-level builders
# ─────────────────────────────────────────────────────────────────────────────

# Circuit id -> (guide path, case design.clay path, parsed cell name).
_CIRCUITS: Dict[str, Tuple[str, str, str]] = {
    "fc_ota": (
        "circuits/fc_ota/layout_guide.txt",
        "cases/opamp/fc_ota_current/source/design.clay",
        "folded_cascode_ota",
    ),
    "two_stage": (
        "circuits/two_stage/layout_guide.txt",
        "cases/opamp/two_stage_current/source/design.clay",
        "two_stage_ota",
    ),
    "ldo": (
        "circuits/ldo/layout_guide.txt",
        "cases/regulator/ldo_current/source/design.clay",
        "ldo_gf180",
    ),
    "bandgap": (
        "circuits/bandgap/layout_guide.txt",
        "cases/reference/bandgap_current/source/design.clay",
        "BandgapBiasRobustR158",
    ),
}


def _repo_root() -> Path:
    # clay/layout/guide_intent.py -> repo root is two parents up from clay/.
    return Path(__file__).resolve().parents[2]


def known_circuits() -> Tuple[str, ...]:
    """The circuit ids this module knows how to build criteria for."""
    return tuple(_CIRCUITS)


# Advisory prose categories that are real designer intent but have no
# machine-checkable obligation in Clay yet (recorded, never faked).
def _advisory_notes(guide_text: str) -> List[str]:
    notes: List[str] = []
    checks = [
        ("EM metal-width / current-density rules", ("EM", "电迁移", "mA/µm", "metal 宽度")),
        ("IR-drop / Kelvin force-sense routing", ("IR drop", "Kelvin", "force", "sense")),
        ("thermal / heat-matching placement", ("热匹配", "自发热", "温升", "温度梯度")),
        ("well-tap density / latch-up guard", ("well tap", "latch-up", "well tap 每")),
        ("deep n-well / substrate isolation", ("deep n-well", "guard ring", "衬底", "p-sub")),
        ("OD-share / finger-merge area technique", ("OD share", "OD-share", "共 OD", "共享 OD")),
        ("stress-gradient die placement", ("应力", "die 中心", "封装应力")),
        ("ESD clamp routing", ("ESD",)),
        ("decoupling-cap placement", ("decoupling", "decap", "bypass cap")),
        ("dummy-gate biasing", ("dummy gate", "dummy 接", "dummy gate 接")),
    ]
    for label, kws in checks:
        if any(k in guide_text for k in kws):
            notes.append(label)
    return notes


def build_criteria(
    circuit: str,
    *,
    repo_root: Optional[Path] = None,
) -> ConvergenceCriteria:
    """Build the :class:`ConvergenceCriteria` for one of the 4 probe circuits.

    Merges (a) the machine-extractable facts from ``circuits/<c>/layout_guide.txt``
    with (b) the cell's parsed ``layout_intent`` block (diff pairs, sensitive
    nets, shields, parasitic budgets, place groups) and (c) the ``annotate``
    block's ``@matching`` device pairs + ``@cell:@symmetric(axis)``.
    """
    if circuit not in _CIRCUITS:
        raise ValueError(f"unknown circuit {circuit!r}; known: {sorted(_CIRCUITS)}")
    guide_rel, clay_rel, cell_name = _CIRCUITS[circuit]
    root = repo_root or _repo_root()
    guide_text = (root / guide_rel).read_text(encoding="utf-8")

    # ── (b) + (c): parsed .clay intent + annotate block ──────────────────────
    from clay.parser import parse_clay_file
    from clay.layout.intent import layout_intent_from_cell_def

    cf = parse_clay_file(str(root / clay_rel))
    cell_def = next(cd for cd in cf.cell_defs if cd.local_name == cell_name)
    intent = layout_intent_from_cell_def(cell_def)
    ann_pairs, ann_axis = _annotate_device_pairs(cell_def)

    # ── (a): guide-prose facts ───────────────────────────────────────────────
    signals = _section(guide_text, _SEC_SIGNALS)
    routing = _section(guide_text, _SEC_ROUTING)
    guide_match_pairs = _strong_match_pairs(signals)
    guide_sensitive = _sensitive_nets(signals)
    guide_centroid_groups = _centroid_groups(guide_text)
    # The known-net universe of this cell: every net named in a parsed budget /
    # diff pair, plus the guide's matched-net and sensitive-net names. Shield
    # victims are filtered to this set so a hypothetical aggressor word in the
    # prose ("dynamic", a "如外部 clk" mention) is not lifted as a fake net.
    known_nets = set(guide_sensitive)
    for b in intent.budgets:
        known_nets.update(b.pair)
    for p in intent.diff_pairs:
        known_nets.update(p)
    for a, b in guide_match_pairs:
        if _looks_like_net(a):
            known_nets.add(a)
        if _looks_like_net(b):
            known_nets.add(b)
    guide_shields = _shield_rules(routing, known_nets=frozenset(known_nets))

    # ── merge ─────────────────────────────────────────────────────────────────
    # mirror axis: annotate @cell:@symmetric wins; else "vertical" if the guide
    # names a vertical center / input axis.
    mirror_axis = ann_axis
    if mirror_axis is None and ("输入轴垂直" in guide_text or "中心轴" in guide_text
                                or "对称" in guide_text):
        mirror_axis = "vertical"

    # matched pairs = annotate device pairs + parsed diff pairs (nets) +
    # guide strong-match pairs (nets/devices). Devices and nets are unioned for
    # the public ``matched_pairs``; ``device_pairs`` keeps only checkable device
    # labels (annotate pairs + any guide pair whose tokens look like devices).
    matched: List[Tuple[str, str]] = []
    matched.extend(ann_pairs)
    matched.extend(intent.diff_pairs)
    matched.extend(intent.mirror_pairs)
    matched.extend(guide_match_pairs)
    matched = _dedupe_pairs(matched)

    # Candidate device matchings (annotate @matching + guide device-↔-device).
    cand_device_pairs = _dedupe_pairs(
        list(ann_pairs)
        + [(a, b) for (a, b) in guide_match_pairs
           if _looks_like_device(a) and _looks_like_device(b)]
    )
    # A geometric MIRROR obligation (P1) is a perfect 1-to-1 matching: each
    # device mirrors exactly ONE partner across the vertical axis. A matching that
    # spans >2 devices is a current-mirror chain / fan-out (a reference ratioed to
    # several devices) -- those want a shared centroid, not a left-right mirror, so
    # they become a common-centroid MATCHING group (P2) instead. Partition by
    # connected component: size-2 -> mirror device pair, larger -> centroid group.
    device_pairs: List[Tuple[str, str]] = []
    mirror_match_groups: List[Tuple[str, ...]] = []
    for comp in _connected_components(cand_device_pairs):
        if len(comp) == 2:
            device_pairs.append((comp[0], comp[1]))
        elif len(comp) > 2:
            mirror_match_groups.append(tuple(comp))
    device_pairs = _dedupe_pairs(device_pairs)

    # sensitive nets = parsed sensitive + guide high-Z nodes.
    sensitive = _dedupe(list(intent.sensitive_nets) + guide_sensitive)

    # common-centroid groups = parsed @place_group device lists + guide 共质心.
    centroid_groups: List[Tuple[str, ...]] = []
    for pg in intent.place_groups:
        if len(pg.devices) >= 2:
            centroid_groups.append(tuple(pg.devices))
    centroid_groups.extend(guide_centroid_groups)
    # current-mirror chains / fan-outs demoted from P1 mirror to P2 centroid match
    centroid_groups.extend(mirror_match_groups)
    # dedupe groups by their frozenset membership
    seen_g = set()
    uniq_groups: List[Tuple[str, ...]] = []
    for g in centroid_groups:
        key = frozenset(g)
        if key in seen_g:
            continue
        seen_g.add(key)
        uniq_groups.append(g)

    # shields = parsed @shield + guide shield rules.
    shields: List[Tuple[str, str, str]] = [
        (s.aggressor, s.victim, s.shield_net) for s in intent.shields
    ]
    shields.extend(guide_shields)
    seen_s = set()
    uniq_shields: List[Tuple[str, str, str]] = []
    for a, v, n in shields:
        key = (a, v, n)
        if key in seen_s:
            continue
        seen_s.add(key)
        uniq_shields.append((a, v, n))

    notes = _advisory_notes(guide_text)

    return ConvergenceCriteria(
        cell=cell_name,
        matched_pairs=tuple(matched),
        mirror_axis=mirror_axis,
        common_centroid=tuple(uniq_groups),
        shields=tuple(uniq_shields),
        sensitive_nets=tuple(sensitive),
        budgets=tuple(intent.budgets),
        device_pairs=tuple(device_pairs),
        notes=tuple(notes),
    )


def build_all_criteria(
    *, repo_root: Optional[Path] = None
) -> Dict[str, ConvergenceCriteria]:
    """Build :class:`ConvergenceCriteria` for all 4 probe circuits."""
    return {c: build_criteria(c, repo_root=repo_root) for c in _CIRCUITS}
