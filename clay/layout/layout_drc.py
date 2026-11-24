"""R136 (BL-136A) — basic geometric DRC verification for routed layouts.

Given a routed ``ParameterizedLibrary`` (or one of its ``NamedCell`` cells) plus
a PDK, run three lightweight design-rule checks against the routed wire metal
and report violations as structured data:

1. ``check_min_spacing`` — same-layer rectangle polygons must be at least the
   PDK minimum spacing apart (edge-to-edge). Polygons that touch or overlap
   are NOT flagged here (they are presumed to be intentionally connected wire
   segments of the same net); only a *positive* gap smaller than the rule is a
   spacing violation. This mirrors how a sign-off DRC space-check treats
   distinct shapes that come too close without merging.

2. ``check_min_width`` — each routed wire's narrow dimension (the smaller of
   its width / height) must be at least the PDK minimum width for its layer.

3. ``check_antenna_ratio`` — a simple antenna check: for each net, the total
   single-layer routed metal area divided by the connected gate (poly) area
   must not exceed a configurable ratio. The default ratio is a clearly-named
   conservative placeholder (NOT a fab-extracted sky130 antenna limit) and is
   flagged with a TODO; pass ``antenna_ratio_limit`` to override with a real
   PDK value once it is sourced.

PDK rule values are read from the repo's real per-layer rule tables
(``clay/layout/layer_pitch.py`` for M1/M2/M3 min-width + min-spacing, and
``clay/layout/drc.py`` SKY130_MIN_DRC_RULES / GF180MCU_MIN_DRC_RULES for the
poly / diff / nwell minima). No PDK-specific number is invented here.

This module is pure data inspection: it imports committed modules read-only and
emits NO geometry. All coordinates are in GDS database units (nm for sky130 /
gf180mcu, where dbu = 1 nm).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from .backend import (
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    ParameterizedLibrary,
)
from .drc import GF180MCU_MIN_DRC_RULES, SKY130_MIN_DRC_RULES, DrcRuleKind
from .layer_pitch import GF180MCU_LAYER_PITCH, SKY130_LAYER_PITCH, LayerPitchRule

# --- antenna ratio default --------------------------------------------------
#
# TODO(BL-136A): this is NOT a fab-extracted sky130 / gf180mcu antenna ratio.
# The repo does not currently store a per-PDK metal-area-to-gate-area antenna
# rule, so we use a single clearly-named conservative placeholder. Real PDK
# antenna ratios are typically in the 200-400 range (cumulative metal area /
# gate area) and differ per metal level; source them from the PDK's antenna
# rule deck and pass ``antenna_ratio_limit`` explicitly when available.
DEFAULT_ANTENNA_RATIO_LIMIT_TODO = 400.0


class DrcCheckKind(Enum):
    MIN_SPACING = "min_spacing"
    MIN_WIDTH = "min_width"
    ANTENNA_RATIO = "antenna_ratio"


@dataclass(frozen=True, eq=True)
class Rect:
    """Axis-aligned bounding box of a polygon in database units."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @staticmethod
    def of(lp: LayerPolygon) -> "Rect":
        xs = [v.x for v in lp.polygon.vertices]
        ys = [v.y for v in lp.polygon.vertices]
        return Rect(min(xs), min(ys), max(xs), max(ys))

    def width(self) -> int:
        return self.x_max - self.x_min

    def height(self) -> int:
        return self.y_max - self.y_min

    def narrow_dim(self) -> int:
        return min(self.width(), self.height())

    def area(self) -> int:
        return self.width() * self.height()


@dataclass(eq=True)
class LayoutDrcViolation:
    """One geometric DRC violation.

    Attributes:
        kind: which check produced this violation.
        rule_id: the originating rule identifier (real rule id, or a
            ``*_TODO_default`` suffix when a conservative placeholder was used).
        layer: GDS layer of the offending shape(s).
        datatype: GDS datatype of the offending shape(s).
        measured: the measured quantity (spacing nm / width nm / area ratio).
        limit: the rule limit the measured value violated.
        rects: the polygon(s) involved (1 for width, 2 for spacing).
        net: the net name when known (antenna; else None).
        message: human-readable summary.
    """

    kind: DrcCheckKind
    rule_id: str
    layer: int
    datatype: int
    measured: float
    limit: float
    rects: List[Rect] = field(default_factory=list)
    net: Optional[str] = None
    message: str = ""


@dataclass
class LayoutDrcReport:
    """Structured result of running layout DRC on a cell."""

    cell_name: str
    pdk_name: str
    rules_checked: List[str] = field(default_factory=list)
    violations: List[LayoutDrcViolation] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.violations

    def add(self, v: LayoutDrcViolation) -> None:
        self.violations.append(v)

    def by_kind(self, kind: DrcCheckKind) -> List[LayoutDrcViolation]:
        return [v for v in self.violations if v.kind == kind]


# --- PDK rule resolution (real values only) ---------------------------------


def _pdk_name(pdk_layers: LayerMap) -> str:
    name = type(pdk_layers).__name__.lower()
    if "sky130" in name:
        return "sky130"
    if "gf180" in name:
        return "gf180mcu"
    raise ValueError(
        f"unknown PDK LayerMap type {type(pdk_layers).__name__!r}; "
        f"layout_drc supports sky130 / gf180mcu"
    )


def _layer_pitch_table(pdk_name: str) -> Dict[str, LayerPitchRule]:
    if pdk_name == "sky130":
        return SKY130_LAYER_PITCH
    if pdk_name == "gf180mcu":
        return GF180MCU_LAYER_PITCH
    raise ValueError(f"no layer-pitch table for PDK {pdk_name!r}")


# Map a GDS (layer, datatype) key -> the LayerPitchRule layer-name string, per
# PDK, using the committed Sky130LayerMap / Gf180mcuLayerMap. POLY/DIFF/NWELL
# share their min-width/spacing with the drc.py rule deck; the routing layers
# (M1/M2/M3) come only from layer_pitch.py.
def _gds_key_to_pitch_layer(
    pdk_layers: LayerMap, pdk_name: str
) -> Dict[Tuple[int, int], str]:
    out: Dict[Tuple[int, int], str] = {}
    table = _layer_pitch_table(pdk_name)
    # logical -> pitch-layer-name
    logical_to_pitch = {
        LogicalLayer.POLY: "POLY",
        LogicalLayer.NDIFF: "DIFF",
        LogicalLayer.PDIFF: "DIFF",
        LogicalLayer.NWELL: "NWELL",
        LogicalLayer.METAL1: "M1",
    }
    for logical, pitch_name in logical_to_pitch.items():
        g = pdk_layers.gds_layer(logical)
        if g is None:
            continue
        if pitch_name not in table:
            continue
        out[(g.layer, g.datatype)] = pitch_name
    return out


def min_width_nm_for_layer(pdk_layers: LayerMap, layer: int, datatype: int) -> Optional[int]:
    """Return the real PDK min-width (nm) for a GDS layer, or None if unknown."""
    pdk_name = _pdk_name(pdk_layers)
    key_map = _gds_key_to_pitch_layer(pdk_layers, pdk_name)
    pitch_name = key_map.get((layer, datatype))
    if pitch_name is None:
        return None
    return _layer_pitch_table(pdk_name)[pitch_name].min_width_nm


def min_spacing_nm_for_layer(pdk_layers: LayerMap, layer: int, datatype: int) -> Optional[int]:
    """Return the real PDK min-spacing (nm) for a GDS layer, or None if unknown."""
    pdk_name = _pdk_name(pdk_layers)
    key_map = _gds_key_to_pitch_layer(pdk_layers, pdk_name)
    pitch_name = key_map.get((layer, datatype))
    if pitch_name is None:
        return None
    return _layer_pitch_table(pdk_name)[pitch_name].min_spacing_nm


# --- geometry helpers -------------------------------------------------------


def _edge_to_edge_gap(a: Rect, b: Rect) -> Optional[int]:
    """Edge-to-edge separation between two axis-aligned rects.

    Returns:
        None if the rects overlap or touch (gap <= 0 in both axes — treated as
            the same connected shape, not a spacing pair).
        Otherwise the positive minimum separation (nm). When the rects'
        projections overlap on one axis, the gap is the clearance on the other
        axis; when they are diagonal, it is the Euclidean corner distance
        rounded to the nearest nm.
    """
    # Gap along x: positive if separated, <=0 if overlapping/touching.
    dx = max(b.x_min - a.x_max, a.x_min - b.x_max)
    dy = max(b.y_min - a.y_max, a.y_min - b.y_max)

    if dx <= 0 and dy <= 0:
        # Overlap / touch on both axes -> same connected metal, not a pair.
        return None
    if dx > 0 and dy > 0:
        # Diagonal separation: corner-to-corner Euclidean distance.
        return int(round((dx * dx + dy * dy) ** 0.5))
    # Separated on exactly one axis: the positive one is the clearance.
    return dx if dx > 0 else dy


def _same_layer_groups(polys: List[LayerPolygon]) -> Dict[Tuple[int, int], List[LayerPolygon]]:
    groups: Dict[Tuple[int, int], List[LayerPolygon]] = {}
    for lp in polys:
        groups.setdefault((lp.layer, lp.datatype), []).append(lp)
    return groups


def _resolve_cell(target: Union[ParameterizedLibrary, NamedCell]) -> NamedCell:
    if isinstance(target, ParameterizedLibrary):
        if not target.cells:
            raise ValueError("library has no cells to check")
        # The top cell (the assembled routed cell) is the last appended.
        return target.cells[-1]
    return target


# --- individual checks ------------------------------------------------------


def check_min_width(
    target: Union[ParameterizedLibrary, NamedCell],
    pdk_layers: LayerMap,
    *,
    layers: Optional[List[Tuple[int, int]]] = None,
) -> List[LayoutDrcViolation]:
    """Flag any polygon whose narrow dimension < the PDK min width for its layer.

    By default checks every layer for which a real min-width rule exists; pass
    ``layers`` to restrict to specific GDS (layer, datatype) keys.
    """
    cell = _resolve_cell(target)
    violations: List[LayoutDrcViolation] = []
    for lp in cell.polygons:
        key = (lp.layer, lp.datatype)
        if layers is not None and key not in layers:
            continue
        limit = min_width_nm_for_layer(pdk_layers, lp.layer, lp.datatype)
        if limit is None:
            continue
        r = Rect.of(lp)
        nd = r.narrow_dim()
        if nd < limit:
            pdk_name = _pdk_name(pdk_layers)
            violations.append(
                LayoutDrcViolation(
                    kind=DrcCheckKind.MIN_WIDTH,
                    rule_id=f"{pdk_name}_min_width_L{lp.layer}D{lp.datatype}",
                    layer=lp.layer,
                    datatype=lp.datatype,
                    measured=float(nd),
                    limit=float(limit),
                    rects=[r],
                    message=(
                        f"narrow dim {nd} nm < min width {limit} nm on "
                        f"layer ({lp.layer},{lp.datatype})"
                    ),
                )
            )
    return violations


def check_min_spacing(
    target: Union[ParameterizedLibrary, NamedCell],
    pdk_layers: LayerMap,
    *,
    layers: Optional[List[Tuple[int, int]]] = None,
) -> List[LayoutDrcViolation]:
    """Flag same-layer rectangle pairs whose positive edge gap < PDK min spacing.

    Touching / overlapping shapes are skipped (treated as one connected net).
    By default checks every layer that has a real min-spacing rule.
    """
    cell = _resolve_cell(target)
    violations: List[LayoutDrcViolation] = []
    groups = _same_layer_groups(cell.polygons)
    for key, group in groups.items():
        if layers is not None and key not in layers:
            continue
        limit = min_spacing_nm_for_layer(pdk_layers, key[0], key[1])
        if limit is None:
            continue
        rects = [Rect.of(lp) for lp in group]
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                gap = _edge_to_edge_gap(rects[i], rects[j])
                if gap is None:
                    continue  # connected / overlapping
                if gap < limit:
                    pdk_name = _pdk_name(pdk_layers)
                    violations.append(
                        LayoutDrcViolation(
                            kind=DrcCheckKind.MIN_SPACING,
                            rule_id=f"{pdk_name}_min_space_L{key[0]}D{key[1]}",
                            layer=key[0],
                            datatype=key[1],
                            measured=float(gap),
                            limit=float(limit),
                            rects=[rects[i], rects[j]],
                            message=(
                                f"spacing {gap} nm < min spacing {limit} nm on "
                                f"layer ({key[0]},{key[1]})"
                            ),
                        )
                    )
    return violations


def check_antenna_ratio(
    net_wires: Dict[str, List[LayerPolygon]],
    gate_area_per_net_nm2: Dict[str, float],
    pdk_layers: LayerMap,
    *,
    antenna_ratio_limit: Optional[float] = None,
) -> List[LayoutDrcViolation]:
    """Simple antenna check: per-net metal area / connected gate area ratio.

    Args:
        net_wires: net name -> routed wire polygons on that net.
        gate_area_per_net_nm2: net name -> total connected gate (poly) area in
            nm^2. A net with zero (or missing) connected gate area is skipped
            (no antenna risk if it touches no gate, e.g. a power rail).
        pdk_layers: the PDK (used only for the rule-id label).
        antenna_ratio_limit: max allowed metal-area / gate-area ratio. If None,
            the conservative TODO placeholder constant is used and the rule id
            is suffixed ``_TODO_default`` so the report is honest about it.

    For each net, the metal area summed across the net's wires (per GDS layer)
    is divided by the net's connected gate area; ratios above the limit are
    flagged. This is a deliberately simple single-net check, not a full
    cumulative-antenna sign-off.
    """
    use_default = antenna_ratio_limit is None
    limit = (
        DEFAULT_ANTENNA_RATIO_LIMIT_TODO if use_default else float(antenna_ratio_limit)
    )
    pdk_name = _pdk_name(pdk_layers)
    rule_id = f"{pdk_name}_antenna_ratio" + ("_TODO_default" if use_default else "")

    violations: List[LayoutDrcViolation] = []
    for net, wires in net_wires.items():
        gate_area = gate_area_per_net_nm2.get(net, 0.0)
        if gate_area <= 0.0:
            continue  # no connected gate -> no antenna risk for this net
        # Sum metal area per layer (single-layer antenna: worst layer wins).
        per_layer_area: Dict[Tuple[int, int], int] = {}
        per_layer_rects: Dict[Tuple[int, int], List[Rect]] = {}
        for lp in wires:
            key = (lp.layer, lp.datatype)
            r = Rect.of(lp)
            per_layer_area[key] = per_layer_area.get(key, 0) + r.area()
            per_layer_rects.setdefault(key, []).append(r)
        for key, metal_area in per_layer_area.items():
            ratio = metal_area / gate_area
            if ratio > limit:
                violations.append(
                    LayoutDrcViolation(
                        kind=DrcCheckKind.ANTENNA_RATIO,
                        rule_id=rule_id,
                        layer=key[0],
                        datatype=key[1],
                        measured=ratio,
                        limit=limit,
                        rects=per_layer_rects[key],
                        net=net,
                        message=(
                            f"net {net!r} antenna ratio {ratio:.1f} > limit "
                            f"{limit:.1f} on layer ({key[0]},{key[1]}) "
                            f"(metal {metal_area} nm^2 / gate {gate_area:.0f} nm^2)"
                        ),
                    )
                )
    return violations


# --- gate-area helper for the antenna check ---------------------------------


def gate_area_per_net_from_pins(
    net_to_devices: Dict[str, List[str]],
    device_pins,
) -> Dict[str, float]:
    """Compute connected gate (poly) area per net from placed DevicePins.

    Uses the device's ``poly_bbox`` (gate poly footprint) as the gate area.
    ``net_to_devices`` maps a net name to the devices whose GATE pin is on that
    net. Devices appearing on a net only via source/drain do NOT contribute
    gate area (the antenna rule cares about gate-connected metal only).
    """
    out: Dict[str, float] = {}
    for net, devs in net_to_devices.items():
        total = 0.0
        for d in devs:
            dp = device_pins.get(d)
            if dp is None:
                continue
            x0, y0, x1, y1 = dp.poly_bbox
            total += float((x1 - x0) * (y1 - y0))
        out[net] = total
    return out


# --- top-level driver -------------------------------------------------------


def run_layout_drc(
    target: Union[ParameterizedLibrary, NamedCell],
    pdk_layers: LayerMap,
    *,
    net_wires: Optional[Dict[str, List[LayerPolygon]]] = None,
    gate_area_per_net_nm2: Optional[Dict[str, float]] = None,
    antenna_ratio_limit: Optional[float] = None,
    layers: Optional[List[Tuple[int, int]]] = None,
) -> LayoutDrcReport:
    """Run min-width + min-spacing on the cell, and (if net data given) antenna.

    Args:
        target: a ParameterizedLibrary (its top cell is checked) or a NamedCell.
        pdk_layers: the PDK LayerMap.
        net_wires / gate_area_per_net_nm2: optional per-net data; if both given,
            the antenna check is also run.
        antenna_ratio_limit: optional override for the antenna ratio limit.
        layers: optional restriction of width/spacing checks to specific layers.

    Returns:
        LayoutDrcReport with every violation found (honest: nothing suppressed).
    """
    cell = _resolve_cell(target)
    pdk_name = _pdk_name(pdk_layers)
    report = LayoutDrcReport(cell_name=cell.name, pdk_name=pdk_name)
    report.rules_checked.append(DrcCheckKind.MIN_WIDTH.value)
    report.rules_checked.append(DrcCheckKind.MIN_SPACING.value)

    for v in check_min_width(cell, pdk_layers, layers=layers):
        report.add(v)
    for v in check_min_spacing(cell, pdk_layers, layers=layers):
        report.add(v)

    if net_wires is not None and gate_area_per_net_nm2 is not None:
        report.rules_checked.append(DrcCheckKind.ANTENNA_RATIO.value)
        for v in check_antenna_ratio(
            net_wires,
            gate_area_per_net_nm2,
            pdk_layers,
            antenna_ratio_limit=antenna_ratio_limit,
        ):
            report.add(v)

    return report


__all__ = [
    "DEFAULT_ANTENNA_RATIO_LIMIT_TODO",
    "DrcCheckKind",
    "Rect",
    "LayoutDrcViolation",
    "LayoutDrcReport",
    "min_width_nm_for_layer",
    "min_spacing_nm_for_layer",
    "check_min_width",
    "check_min_spacing",
    "check_antenna_ratio",
    "gate_area_per_net_from_pins",
    "run_layout_drc",
]
