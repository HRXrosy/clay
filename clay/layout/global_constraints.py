"""Global layout/routing constraints every Clay-built cell must satisfy.

Loaded from clay/layout/global_layout_constraints.yaml (data; elastically
extensible). Each rule id dispatches to a checker function over a routed cell's
tagged geometry. Returns a list of :class:`Violation` records; an empty list
means clean.

Per-cell semantic intent (matched_cc_balance, shield, mirror_pairs, ...) lives
in the LayoutIntent contract (clay/layout/intent.py). This module is the
STRUCTURAL third axis: rules every cell must satisfy regardless of circuit.

Intended use:

    from clay.layout.global_constraints import check_global_constraints
    violations = check_global_constraints(routed_cell)
    if violations:
        raise LayoutConstraintError(violations)

Also designed to plug into ``full_signoff`` as a third proof-token axis
(parasitic + placement + structural).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

_DEFAULT_YAML = Path(__file__).parent / "global_layout_constraints.yaml"

BBox = Tuple[float, float, float, float]   # (x_min, y_min, x_max, y_max)


@dataclass(frozen=True)
class Violation:
    rule_id: str
    severity: str            # "error" | "warning"
    message: str
    net: Optional[str] = None
    layer: Optional[str] = None
    bbox: Optional[BBox] = None
    device: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.upper()}] {self.rule_id}: {self.message}"]
        meta = []
        if self.net:
            meta.append(f"net={self.net}")
        if self.layer:
            meta.append(f"layer={self.layer}")
        if self.device:
            meta.append(f"dev={self.device}")
        if meta:
            parts.append(" (" + ", ".join(meta) + ")")
        return "".join(parts)


def load_constraints(path: Path = _DEFAULT_YAML) -> List[dict]:
    """Read the yaml ruleset; returns the list of rule dicts."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return list(doc.get("rules") or [])


# ─────────────────────────────────────────────────────────────────────────────
# Geometry adapter: pull (net, layer, bbox) tuples from a routed cell.
# Different cells store .tagged in slightly different shapes; this is the
# single seam every checker uses.
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_of(geom: Any) -> Optional[BBox]:
    """Best-effort bbox extraction from a polygon/layer-polygon/raw-tuple."""
    if geom is None:
        return None
    # Already a (x_min,y_min,x_max,y_max) tuple/list.
    if isinstance(geom, (tuple, list)) and len(geom) == 4 and all(
        isinstance(v, (int, float)) for v in geom
    ):
        return tuple(float(v) for v in geom)  # type: ignore[return-value]
    # A LayerPolygon / Polygon-like with .bbox.
    bb = getattr(geom, "bbox", None)
    if bb is not None:
        return _bbox_of(bb)
    # A Polygon with .points: compute bbox.
    pts = getattr(geom, "points", None)
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    # A Bbox-like with x_min/y_min/x_max/y_max.
    if hasattr(geom, "x_min"):
        return (float(geom.x_min), float(geom.y_min),
                float(geom.x_max), float(geom.y_max))
    return None


def _iter_tagged(routed_cell: Any) -> Iterable[Tuple[str, str, BBox]]:
    """Yield (net, layer, bbox) for every tagged polygon on a routed cell."""
    tagged = getattr(routed_cell, "tagged", None)
    if tagged is None:
        return
    for entry in tagged:
        # tagged entry tends to be (net, layer_key, polygon-or-bbox), or a
        # LayerPolygon carrying (net, layer); be liberal.
        if isinstance(entry, (tuple, list)) and len(entry) >= 3:
            net = str(entry[0])
            layer = str(entry[1])
            bb = _bbox_of(entry[2])
        else:
            net = str(getattr(entry, "net", "") or "")
            layer = str(getattr(entry, "layer", "") or "")
            bb = _bbox_of(entry)
        if bb is not None and net:
            yield net, layer, bb


def _bbox_overlap(a: BBox, b: BBox) -> bool:
    """Strict (open-interval) bbox overlap test in 2D."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


# ─────────────────────────────────────────────────────────────────────────────
# Rule implementations. New rules: add a function + register in _DISPATCH.
# ─────────────────────────────────────────────────────────────────────────────

def _rail_net_aliases(params: dict) -> Tuple[set, set]:
    """Accept either ``vdd_net`` (legacy single) or ``vdd_nets`` (preferred list)."""
    vdd_nets = set(params.get("vdd_nets") or [])
    if not vdd_nets:
        vdd_nets = {params.get("vdd_net", "vdd")}
    gnd_nets = set(params.get("gnd_nets") or ["gnd", "vss"])
    return vdd_nets, gnd_nets


def _identify_rail(polys: List[BBox], at_top: bool) -> Optional[BBox]:
    """Pick the actual rail polygon out of all polygons of a power net.

    The rail is a wide horizontal strip; vdd/gnd taps are short vertical
    fingers. Among the polygons reaching the cell's extremum y (top for vdd,
    bottom for gnd), choose the widest one.
    """
    if not polys:
        return None
    if at_top:
        extremum = max(b[3] for b in polys)
        candidates = [b for b in polys if b[3] >= extremum - 1e-6]
    else:
        extremum = min(b[1] for b in polys)
        candidates = [b for b in polys if b[1] <= extremum + 1e-6]
    return max(candidates, key=lambda b: b[2] - b[0])  # widest


def check_containment_within_rails(routed_cell: Any, params: dict) -> List[Violation]:
    """Rule: every non-rail polygon's y-range must lie in [floor, ceiling]
    where ceiling = y_min of the vdd rail polygon, floor = y_max of the gnd
    rail polygon. Rails identified geometrically (widest polygon at the
    y-extremum of each power net), so vdd/gnd taps descending into the cell
    don't fool the check.
    """
    vdd_nets, gnd_nets = _rail_net_aliases(params)
    rail_nets = vdd_nets | gnd_nets
    tol = float(params.get("tolerance_nm", 0))

    vdd_polys = [b for net, _, b in _iter_tagged(routed_cell) if net in vdd_nets]
    gnd_polys = [b for net, _, b in _iter_tagged(routed_cell) if net in gnd_nets]
    if not vdd_polys or not gnd_polys:
        return [Violation(
            "containment_within_rails", "warning",
            f"power rails not tagged with one of vdd={sorted(vdd_nets)} / "
            f"gnd={sorted(gnd_nets)}; this cell uses an unrecognized convention "
            f"and the rule cannot apply -- consider tagging the rail polygons "
            f"with a standard name",
        )]

    vdd_rail = _identify_rail(vdd_polys, at_top=True)
    gnd_rail = _identify_rail(gnd_polys, at_top=False)
    assert vdd_rail is not None and gnd_rail is not None
    ceiling = vdd_rail[1]       # bottom edge of the top vdd rail
    floor = gnd_rail[3]         # top edge of the bottom gnd rail

    violations: List[Violation] = []
    for net, layer, b in _iter_tagged(routed_cell):
        if net in rail_nets:
            continue
        if b[1] < floor - tol or b[3] > ceiling + tol:
            violations.append(Violation(
                "containment_within_rails", "error",
                f"non-rail polygon (net={net}, layer={layer}) y=[{b[1]:.0f},{b[3]:.0f}] "
                f"outside rail band [floor={floor:.0f}, ceiling={ceiling:.0f}]",
                net=net, layer=layer, bbox=b,
            ))
    return violations


def check_no_signal_crosses_rails(routed_cell: Any, params: dict) -> List[Violation]:
    """Rule: no signal polygon overlaps a rail polygon on the SAME layer."""
    vdd_nets, gnd_nets = _rail_net_aliases(params)
    rail_nets = vdd_nets | gnd_nets

    by_layer: Dict[str, List[Tuple[str, BBox]]] = {}
    for net, layer, b in _iter_tagged(routed_cell):
        by_layer.setdefault(layer, []).append((net, b))

    violations: List[Violation] = []
    for layer, items in by_layer.items():
        rails = [(n, b) for n, b in items if n in rail_nets]
        if not rails:
            continue
        signals = [(n, b) for n, b in items if n not in rail_nets]
        for snet, sb in signals:
            for rnet, rb in rails:
                if _bbox_overlap(sb, rb):
                    violations.append(Violation(
                        "no_signal_crosses_rails", "error",
                        f"signal polygon (net={snet}) overlaps rail (net={rnet}) "
                        f"on layer {layer}: signal bbox=({sb[0]:.0f},{sb[1]:.0f},"
                        f"{sb[2]:.0f},{sb[3]:.0f})",
                        net=snet, layer=layer, bbox=sb,
                    ))
    return violations


def _terminal_centroid(terminals: Dict[str, BBox]) -> Tuple[float, float]:
    cs = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in terminals.values()]
    return (sum(c[0] for c in cs) / len(cs), sum(c[1] for c in cs) / len(cs))


def _pair_by_suffix(labels: List[str], pair_suffixes: List[List[str]]) -> List[Tuple[str, str]]:
    """Heuristically pair device labels by name-suffix convention."""
    label_set = set(labels)
    used: set = set()
    pairs: List[Tuple[str, str]] = []
    for label in labels:
        if label in used:
            continue
        ll = label.lower()
        for sa, sb in pair_suffixes:
            if ll.endswith(sa.lower()):
                stem = label[: -len(sa)]
                # try same case + canonical case for the partner suffix
                for partner in (stem + sb, stem + sb.upper(), stem + sb.lower()):
                    if partner in label_set and partner != label and partner not in used:
                        pairs.append((label, partner))
                        used.add(label)
                        used.add(partner)
                        break
                if label in used:
                    break
    return pairs


def check_prefer_symmetry_and_minimal(routed_cell: Any, params: dict) -> List[Violation]:
    """Soft preference: mirror-symmetric placement for matched pairs +
    cell area + wire length reported as info metrics. Warning-severity, so
    violations don't fail a cell -- they flag asymmetric pairs for repair.
    """
    placement = getattr(routed_cell, "placement", None)
    if placement is None or not hasattr(placement, "term"):
        return [Violation(
            "prefer_symmetry_and_minimal", "info",
            "no placement.term -- placement symmetry sub-check skipped",
        )]
    pair_suffixes = params.get("pair_suffixes") or [["_p", "_n"]]
    threshold = float(params.get("placement_symmetry_max_imbalance", 0.02))
    vdd_nets, gnd_nets = _rail_net_aliases(params)
    rail_nets = vdd_nets | gnd_nets

    # --- placement symmetry sub-check (warning) -----------------------------
    labels = list(placement.term.keys())
    pairs = _pair_by_suffix(labels, pair_suffixes)

    violations: List[Violation] = []
    if pairs:
        centroids = {l: _terminal_centroid(placement.term[l]) for l in labels}
        cell_xs = [c[0] for c in centroids.values()]
        cell_ys = [c[1] for c in centroids.values()]
        cell_w = max(cell_xs) - min(cell_xs) or 1.0
        cell_h = max(cell_ys) - min(cell_ys) or 1.0
        # axis_x = mean of pair MIDPOINTS, not mean of all devices: unpaired
        # devices (tail, taps, biases) must not shift the mirror axis.
        pair_mids_x = [(centroids[a][0] + centroids[b][0]) / 2.0 for a, b in pairs]
        axis_x = sum(pair_mids_x) / len(pair_mids_x)

        asymm = 0
        for a, b in pairs:
            cxa, cya = centroids[a]
            cxb, cyb = centroids[b]
            # symmetry quality of THIS pair against the agreed axis:
            #   axis_dev: how far the pair's own midpoint sits off the axis,
            #   dy_offset: how far the two devices disagree in y.
            mid_x = (cxa + cxb) / 2.0
            axis_dev = abs(mid_x - axis_x) / cell_w
            dy = abs(cya - cyb) / cell_h
            if axis_dev > threshold or dy > threshold:
                asymm += 1
                violations.append(Violation(
                    "prefer_symmetry_and_minimal", "warning",
                    f"matched pair ({a},{b}) not mirror-symmetric: "
                    f"axis_dev={axis_dev:.3f} dy_offset={dy:.3f} (tol={threshold:.3f})",
                    device=f"{a}+{b}",
                ))
        violations.append(Violation(
            "prefer_symmetry_and_minimal", "info",
            f"placement symmetry: {len(pairs)-asymm}/{len(pairs)} matched pairs symmetric "
            f"(axis_x={axis_x:.0f})",
        ))

    # --- cell area + total signal trace length (info metrics) ----------------
    bbs = [b for _, _, b in _iter_tagged(routed_cell)]
    if bbs:
        x_min = min(b[0] for b in bbs)
        x_max = max(b[2] for b in bbs)
        y_min = min(b[1] for b in bbs)
        y_max = max(b[3] for b in bbs)
        area_um2 = ((x_max - x_min) * (y_max - y_min)) / 1.0e6  # nm^2 -> um^2
        signal_perim_nm = sum(
            (b[2] - b[0]) + (b[3] - b[1])
            for net, _, b in _iter_tagged(routed_cell)
            if net not in rail_nets
        )
        violations.append(Violation(
            "prefer_symmetry_and_minimal", "info",
            f"cell area = {area_um2:.1f} um^2 "
            f"({x_max - x_min:.0f} x {y_max - y_min:.0f} nm); "
            f"signal trace length proxy = {signal_perim_nm/1000:.0f} um",
        ))
    return violations


_DISPATCH: Dict[str, Any] = {
    "containment_within_rails": check_containment_within_rails,
    "no_signal_crosses_rails": check_no_signal_crosses_rails,
    "prefer_symmetry_and_minimal": check_prefer_symmetry_and_minimal,
}


def check_global_constraints(
    routed_cell: Any,
    *,
    yaml_path: Optional[Path] = None,
    rule_ids: Optional[Iterable[str]] = None,
) -> List[Violation]:
    """Run every rule in the yaml ruleset against ``routed_cell``.

    Optional ``rule_ids`` restricts to a subset (useful for fast tests).
    Returns an aggregated list of :class:`Violation`s (empty = clean).
    """
    rules = load_constraints(yaml_path or _DEFAULT_YAML)
    if rule_ids is not None:
        keep = set(rule_ids)
        rules = [r for r in rules if r["id"] in keep]
    out: List[Violation] = []
    for rule in rules:
        rid = rule["id"]
        fn = _DISPATCH.get(rid)
        if fn is None:
            out.append(Violation(
                rid, "warning",
                f"no checker registered for rule id {rid!r}",
            ))
            continue
        out.extend(fn(routed_cell, rule.get("parameters") or {}))
    return out


def constraint_summary(violations: List[Violation]) -> Dict[str, Any]:
    """Aggregate counts per rule_id + severity. Useful for reporting."""
    per_rule: Dict[str, Dict[str, int]] = {}
    for v in violations:
        slot = per_rule.setdefault(v.rule_id, {"error": 0, "warning": 0, "info": 0})
        slot[v.severity] = slot.get(v.severity, 0) + 1
    return {
        "n_total": len(violations),
        "n_error": sum(1 for v in violations if v.severity == "error"),
        "n_warning": sum(1 for v in violations if v.severity == "warning"),
        "n_info": sum(1 for v in violations if v.severity == "info"),
        "per_rule": per_rule,
        "passed": not any(v.severity == "error" for v in violations),
    }


__all__ = [
    "Violation",
    "load_constraints",
    "check_global_constraints",
    "constraint_summary",
    "check_containment_within_rails",
    "check_no_signal_crosses_rails",
]
