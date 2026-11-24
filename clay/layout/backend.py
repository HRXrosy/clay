"""Phase 3a backend — pure-Python mirror.

Mirrors ``crates/clay_layout/src/backend/*`` and ``pdk_layers.rs``.

Drift risk vs Rust source is mitigated by the end-to-end test
``tests/test_layout_end_to_end_diffpair.py`` which builds the same
library in Python and emits matching ASCII GDS structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

from .annotations import (
    AnnotatedCell,
    GuardRing,
    Symmetric,
    WellType,
)
from .symbolic_layout import SymbolicLayout


# --- PDK layer mapping ---


@dataclass(frozen=True, eq=True)
class GdsLayer:
    """A GDSII layer + datatype pair (the standard 2-tuple GDS layer key)."""

    layer: int
    datatype: int


class LogicalLayer(Enum):
    """Abstract logical layer name (PDK-independent).

    The base set (NDIFF/PDIFF/POLY/CONTACT/METAL1/NWELL/PWELL_MARKER) is the
    Phase-3a geometry surface. P0 adds the implant + well-isolation members so an
    implant-distinguished process (gf180mcu has ONE diffusion COMP layer, N/P
    devices are told apart by the Nplus/Pplus implant drawn over COMP) can be
    expressed PDK-agnostically:

      * NPLUS / PPLUS -- the n+/p+ source-drain implant markers (gf180 Nplus 32/0,
        Pplus 31/0). Needed to draw implant-correct devices + for implant-based
        LVS bulk recognition.
      * PWELL -- a DRAWN p-well marker (distinct from PWELL_MARKER which is the
        guard-ring annotation target). gf180 LVPWELL 204/0 exists only inside a
        deep-nwell region; on the baseline flow it resolves to None.
      * DNWELL -- the deep n-well isolation layer (gf180 DNWELL 12/0).
    """

    NDIFF = "ndiff"
    PDIFF = "pdiff"
    POLY = "poly"
    CONTACT = "contact"
    METAL1 = "metal1"
    NWELL = "nwell"
    PWELL_MARKER = "pwell_marker"
    NPLUS = "nplus"
    PPLUS = "pplus"
    PWELL = "pwell"
    DNWELL = "dnwell"


class LayerMap(Protocol):
    """Protocol providing layer mapping (mirror of Rust ``trait LayerMap``)."""

    def gds_layer(self, logical: LogicalLayer) -> Optional[GdsLayer]:
        ...


class Sky130LayerMap:
    """Sky130 LayerMap (mirror of ``impl LayerMap for Sky130`` in Rust)."""

    _TABLE = {
        LogicalLayer.NDIFF: GdsLayer(65, 20),
        LogicalLayer.PDIFF: GdsLayer(65, 20),
        LogicalLayer.POLY: GdsLayer(66, 20),
        LogicalLayer.CONTACT: GdsLayer(66, 44),
        LogicalLayer.METAL1: GdsLayer(67, 20),
        LogicalLayer.NWELL: GdsLayer(64, 20),
        LogicalLayer.PWELL_MARKER: None,
        # sky130 implants: nsdm 93/44 (n+ s/d), psdm 94/20 (p+ s/d). pwde/dnwell
        # are not part of the baseline flow here -> None (skywater public PDK).
        LogicalLayer.NPLUS: GdsLayer(93, 44),
        LogicalLayer.PPLUS: GdsLayer(94, 20),
        LogicalLayer.PWELL: GdsLayer(64, 44),   # pwell 64/44
        LogicalLayer.DNWELL: GdsLayer(64, 18),  # dnwell 64/18
    }

    def gds_layer(self, logical: LogicalLayer) -> Optional[GdsLayer]:
        return self._TABLE.get(logical)


class Gf180mcuLayerMap:
    """Gf180mcu LayerMap (mirror of ``impl LayerMap for Gf180mcu`` in Rust)."""

    _TABLE = {
        LogicalLayer.NDIFF: GdsLayer(22, 0),
        LogicalLayer.PDIFF: GdsLayer(22, 0),
        LogicalLayer.POLY: GdsLayer(30, 0),
        LogicalLayer.CONTACT: GdsLayer(33, 0),
        LogicalLayer.METAL1: GdsLayer(34, 0),
        LogicalLayer.NWELL: GdsLayer(21, 0),
        LogicalLayer.PWELL_MARKER: None,
        # gf180mcu implants (gf180mcu.lyp ground truth, GF180_VERIFIED_SOURCES):
        # Nplus 32/0, Pplus 31/0. PWELL (drawn p-well) = LVPWELL 204/0; on the
        # baseline non-deep-nwell flow the substrate IS the p-well so a drawn
        # marker is not required -> resolve to LVPWELL only when explicitly used.
        # DNWELL 12/0 (deep n-well).
        LogicalLayer.NPLUS: GdsLayer(32, 0),
        LogicalLayer.PPLUS: GdsLayer(31, 0),
        LogicalLayer.PWELL: GdsLayer(204, 0),   # LVPWELL
        LogicalLayer.DNWELL: GdsLayer(12, 0),
    }

    def gds_layer(self, logical: LogicalLayer) -> Optional[GdsLayer]:
        return self._TABLE.get(logical)


# PDK fin-pitch constants (mirror of Rust pdk_dims constants).
SKY130_FIN_PITCH_NM = 48.0
GF180_FIN_PITCH_NM = 280.0

# Phase 5.1: PDK min poly-poly spacing for DRC-aware lowering.
# Source: SKY130_MIN_DRC_RULES / GF180MCU_MIN_DRC_RULES in clay/layout/drc.py.
# These constants are exposed at module level so callers don't need to know
# the DrcRule data structure layout. Use with lower_annotated_cell_drc_aware
# or via the auto-lookup helper min_poly_space_nm_for_pdk_layers.
SKY130_MIN_POLY_SPACE_NM = 210.0
GF180_MIN_POLY_SPACE_NM = 280.0

# Phase 5.2: nwell-to-diff enclosure (when GuardRing annotation produces an nwell
# polygon, the nwell must extend at least this many nm beyond the enclosed diff).
# Source: SKY130_MIN_DRC_RULES sky130_nwell_diff_enc rule (180 nm).
# Gf180mcu nwell-diff enclosure is not in our minimum DRC subset; leave as 0 nm
# (no enforcement) — the GF180MCU_MIN_DRC_RULES set doesn't include this rule,
# so Phase 5.2 lowering will fall back to Phase 5 behavior (2 * fin_pitch padding).
SKY130_NWELL_DIFF_ENCLOSURE_NM = 180.0
GF180_NWELL_DIFF_ENCLOSURE_NM = 0.0


def min_poly_space_nm_for_pdk_layers(pdk_layers) -> float:
    """Phase 5.1 auto-lookup: return min poly-poly DRC spacing for a PDK LayerMap.

    Recognizes Sky130LayerMap / Gf180mcuLayerMap by class name; for other
    PDK LayerMap implementations, raise ValueError so the caller falls back
    to passing min_poly_space_nm explicitly.
    """
    name = type(pdk_layers).__name__.lower()
    if "sky130" in name:
        return SKY130_MIN_POLY_SPACE_NM
    if "gf180" in name:
        return GF180_MIN_POLY_SPACE_NM
    raise ValueError(
        f"unknown PDK LayerMap type: {type(pdk_layers).__name__}; "
        f"pass min_poly_space_nm explicitly to lower_annotated_cell_drc_aware"
    )


def nwell_diff_enclosure_nm_for_pdk_layers(pdk_layers) -> float:
    """Phase 5.2 auto-lookup: return nwell-to-diff enclosure DRC rule for a PDK LayerMap.

    Returns 0.0 for PDKs without an enclosure rule in our minimum DRC subset
    (currently gf180mcu) so callers can pass it directly without conditional
    logic. Returns 180.0 for sky130. Raises ValueError on unknown PDK type.
    """
    name = type(pdk_layers).__name__.lower()
    if "sky130" in name:
        return SKY130_NWELL_DIFF_ENCLOSURE_NM
    if "gf180" in name:
        return GF180_NWELL_DIFF_ENCLOSURE_NM
    raise ValueError(
        f"unknown PDK LayerMap type: {type(pdk_layers).__name__}; "
        f"pass min_well_diff_enclosure_nm explicitly"
    )


# --- ParameterizedCell IR ---


@dataclass(frozen=True, eq=True)
class Point:
    x: int
    y: int


@dataclass(eq=True)
class Polygon:
    vertices: List[Point] = field(default_factory=list)

    @staticmethod
    def rectangle(x_min: int, y_min: int, x_max: int, y_max: int) -> "Polygon":
        return Polygon(
            vertices=[
                Point(x_min, y_min),
                Point(x_max, y_min),
                Point(x_max, y_max),
                Point(x_min, y_max),
            ]
        )


@dataclass(eq=True)
class LayerPolygon:
    layer: int
    datatype: int
    polygon: Polygon


@dataclass(eq=True)
class NamedCell:
    name: str
    polygons: List[LayerPolygon] = field(default_factory=list)


@dataclass(eq=True)
class ParameterizedLibrary:
    name: str
    unit_user: float = 0.001
    unit_db_meters: float = 1e-9
    cells: List[NamedCell] = field(default_factory=list)

    def add_cell(self, cell: NamedCell) -> None:
        self.cells.append(cell)


# --- ASCII GDS emit / parse ---


def emit_ascii_gds(lib: ParameterizedLibrary) -> str:
    """Pure-Python mirror of Rust ``emit_ascii_gds``."""
    lines: List[str] = []
    lines.append("HEADER 600")
    lines.append("BGNLIB")
    lines.append(f"LIBNAME {lib.name}")
    lines.append(f"UNITS {lib.unit_user} {lib.unit_db_meters}")
    for cell in lib.cells:
        lines.append("BGNSTR")
        lines.append(f"STRNAME {cell.name}")
        for lp in cell.polygons:
            lines.append("BOUNDARY")
            lines.append(f"LAYER {lp.layer}")
            lines.append(f"DATATYPE {lp.datatype}")
            xy_parts = ["XY"]
            for v in lp.polygon.vertices:
                xy_parts.append(f"{v.x} {v.y}")
            lines.append(" ".join(xy_parts))
            lines.append("ENDEL")
        lines.append("ENDSTR")
    lines.append("ENDLIB")
    return "\n".join(lines) + "\n"


class AsciiGdsParseError(Exception):
    """Raised when ASCII GDS parsing encounters malformed input."""


def parse_ascii_gds(s: str) -> ParameterizedLibrary:
    """Pure-Python mirror of Rust ``parse_ascii_gds``."""
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    idx = 0

    def peek() -> str:
        return lines[idx] if idx < len(lines) else ""

    def consume(expected_prefix: str) -> str:
        nonlocal idx
        if idx >= len(lines):
            raise AsciiGdsParseError(f"truncated, expected `{expected_prefix}`")
        line = lines[idx]
        # exact-match tokens (no trailing space) are bare keywords; otherwise prefix
        if expected_prefix.endswith(" "):
            if not line.startswith(expected_prefix):
                raise AsciiGdsParseError(
                    f"line {idx + 1}: unexpected `{line}`, expected `{expected_prefix.strip()}`"
                )
        else:
            if not (line == expected_prefix or line.startswith(expected_prefix + " ")):
                raise AsciiGdsParseError(
                    f"line {idx + 1}: unexpected `{line}`, expected `{expected_prefix}`"
                )
        idx += 1
        return line

    consume("HEADER")
    consume("BGNLIB")

    libname_line = consume("LIBNAME")
    libname = libname_line[len("LIBNAME "):].strip() if len(libname_line) > len("LIBNAME") else ""

    units_line = consume("UNITS")
    units_rest = units_line[len("UNITS "):] if len(units_line) > len("UNITS") else ""
    units_parts = units_rest.split()
    if len(units_parts) != 2:
        raise AsciiGdsParseError(f"UNITS expects 2 values, got {len(units_parts)}")
    try:
        unit_user = float(units_parts[0])
        unit_db_meters = float(units_parts[1])
    except ValueError as e:
        raise AsciiGdsParseError(f"UNITS number format: {e}")

    lib = ParameterizedLibrary(
        name=libname, unit_user=unit_user, unit_db_meters=unit_db_meters
    )

    while idx < len(lines):
        head = peek()
        if head == "ENDLIB":
            consume("ENDLIB")
            return lib
        consume("BGNSTR")
        strname_line = consume("STRNAME")
        cell_name = (
            strname_line[len("STRNAME "):].strip()
            if len(strname_line) > len("STRNAME")
            else ""
        )
        cell = NamedCell(name=cell_name)
        while idx < len(lines) and peek() != "ENDSTR":
            consume("BOUNDARY")
            layer_line = consume("LAYER")
            datatype_line = consume("DATATYPE")
            xy_line = consume("XY")
            try:
                layer = int(layer_line[len("LAYER "):].strip())
                datatype = int(datatype_line[len("DATATYPE "):].strip())
            except ValueError as e:
                raise AsciiGdsParseError(f"number parse error: {e}")
            xy_rest = xy_line[len("XY"):].strip()
            try:
                nums = [int(t) for t in xy_rest.split()]
            except ValueError as e:
                raise AsciiGdsParseError(f"XY number parse error: {e}")
            if len(nums) % 2 != 0:
                raise AsciiGdsParseError(f"odd XY count: {len(nums)}")
            vertices = [Point(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
            consume("ENDEL")
            cell.polygons.append(
                LayerPolygon(
                    layer=layer, datatype=datatype, polygon=Polygon(vertices=vertices)
                )
            )
        consume("ENDSTR")
        lib.cells.append(cell)

    raise AsciiGdsParseError("truncated, expected ENDLIB")


# --- Lowering (mirror Rust) ---


def _nm_to_dbu(value_nm: float) -> int:
    return int(round(value_nm))


def lower_mos_finger_geometry(
    layout: SymbolicLayout,
    w_um: float,
    l_um: float,
    pdk_layers: LayerMap,
    fin_pitch_nm: float,
) -> List[LayerPolygon]:
    """Pure-Python mirror of Rust ``lower_mos_finger_geometry``.

    Note: the Rust signature takes a single ``pdk: &(impl PdkDims + LayerMap)``
    and accesses both ``fin_pitch_nm()`` and ``gds_layer(...)``. In Python the
    Phase 1 ``PdkDims`` and the Phase 3a ``LayerMap`` are different classes, so
    the caller passes ``fin_pitch_nm`` explicitly. Use the ``SKY130_FIN_PITCH_NM``
    / ``GF180_FIN_PITCH_NM`` module constants to pair with the matching LayerMap.
    """
    polys: List[LayerPolygon] = []
    l_nm = _nm_to_dbu(l_um * 1000.0)
    w_nm = _nm_to_dbu(w_um * 1000.0)
    fin_pitch = _nm_to_dbu(fin_pitch_nm)
    extension = 2 * fin_pitch
    n = layout.finger_count

    poly = pdk_layers.gds_layer(LogicalLayer.POLY)
    ndiff = pdk_layers.gds_layer(LogicalLayer.NDIFF)
    if poly is None or ndiff is None:
        raise ValueError("PDK must define Poly and NDiff layers")

    for i in range(n):
        x_min = i * (l_nm + fin_pitch)
        x_max = x_min + l_nm
        polys.append(
            LayerPolygon(
                layer=poly.layer,
                datatype=poly.datatype,
                polygon=Polygon.rectangle(x_min, 0, x_max, w_nm),
            )
        )

    ndiff_x_min = -extension
    ndiff_x_max = max(0, n - 1) * (l_nm + fin_pitch) + l_nm + extension
    polys.append(
        LayerPolygon(
            layer=ndiff.layer,
            datatype=ndiff.datatype,
            polygon=Polygon.rectangle(
                ndiff_x_min, -fin_pitch, ndiff_x_max, w_nm + fin_pitch
            ),
        )
    )

    return polys


def _bbox(polys: List[LayerPolygon]) -> Optional[Tuple[int, int, int, int]]:
    xs = [v.x for lp in polys for v in lp.polygon.vertices]
    ys = [v.y for lp in polys for v in lp.polygon.vertices]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _translate(polys: List[LayerPolygon], dx: int, dy: int) -> List[LayerPolygon]:
    return [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(
                vertices=[Point(v.x + dx, v.y + dy) for v in lp.polygon.vertices]
            ),
        )
        for lp in polys
    ]


def _mirror_x(polys: List[LayerPolygon], axis_x: int) -> List[LayerPolygon]:
    return [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(
                vertices=list(
                    reversed([Point(2 * axis_x - v.x, v.y) for v in lp.polygon.vertices])
                )
            ),
        )
        for lp in polys
    ]


# Per-device sizing (W and L in um). Mirror of Rust ``DeviceSizing`` type alias.
DeviceSizing = Dict[str, Tuple[float, float]]


def lower_annotated_cell_to_parameterized(
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    fin_pitch_nm: float,
    sizing: DeviceSizing,
) -> ParameterizedLibrary:
    """Pure-Python mirror of Rust ``lower_annotated_cell_to_parameterized``.

    Each device in ``cell.layouts`` becomes a NamedCell in the library, plus a
    top ``cell.cell_name`` NamedCell that contains all device polygons placed
    side-by-side (with Symmetric-driven mirroring of the second half).
    """
    lib = ParameterizedLibrary(name=cell.cell_name)
    device_keys = sorted(cell.layouts.keys())
    device_gap = 4 * _nm_to_dbu(fin_pitch_nm)

    # Detect Symmetric annotation at the cell level.
    symmetric_devices: Optional[List[str]] = None
    for anns in cell.annotations.values():
        for a in anns:
            if isinstance(a, Symmetric):
                symmetric_devices = list(a.devices)
                break
        if symmetric_devices is not None:
            break

    placement_x = 0
    top_polygons: List[LayerPolygon] = []

    for dev in device_keys:
        layout = cell.layouts[dev]
        w_um, l_um = sizing.get(dev, (1.0, 0.5))
        polys = lower_mos_finger_geometry(
            layout, w_um, l_um, pdk_layers, fin_pitch_nm
        )

        # Emit per-device cell at local origin (no placement / no mirror).
        lib.add_cell(NamedCell(name=dev, polygons=[
            LayerPolygon(
                layer=lp.layer,
                datatype=lp.datatype,
                polygon=Polygon(vertices=list(lp.polygon.vertices)),
            )
            for lp in polys
        ]))

        # Symmetric mirroring: if dev is in second half, mirror around local bbox center.
        if symmetric_devices is not None:
            half = len(symmetric_devices) // 2
            if dev in symmetric_devices:
                idx = symmetric_devices.index(dev)
                if idx >= half:
                    bb = _bbox(polys)
                    if bb is not None:
                        polys = _mirror_x(polys, (bb[0] + bb[2]) // 2)

        polys = _translate(polys, placement_x, 0)
        bb_after = _bbox(polys)
        if bb_after is not None:
            placement_x = bb_after[2] + device_gap
        top_polygons.extend(polys)

    # Detect GuardRing annotation; pick first well_type encountered.
    guard_well: Optional[WellType] = None
    for anns in cell.annotations.values():
        for a in anns:
            if isinstance(a, GuardRing):
                guard_well = a.well_type
                break
        if guard_well is not None:
            break

    if guard_well is not None:
        pad = 2 * _nm_to_dbu(fin_pitch_nm)
        bb = _bbox(top_polygons)
        if bb is not None:
            x_min, y_min, x_max, y_max = bb
            if guard_well == WellType.NWELL:
                nwell = pdk_layers.gds_layer(LogicalLayer.NWELL)
                if nwell is not None:
                    top_polygons.append(
                        LayerPolygon(
                            layer=nwell.layer,
                            datatype=nwell.datatype,
                            polygon=Polygon.rectangle(
                                x_min - pad, y_min - pad, x_max + pad, y_max + pad
                            ),
                        )
                    )
            elif guard_well == WellType.PWELL:
                pmarker = pdk_layers.gds_layer(LogicalLayer.PWELL_MARKER)
                if pmarker is not None:
                    top_polygons.append(
                        LayerPolygon(
                            layer=pmarker.layer,
                            datatype=pmarker.datatype,
                            polygon=Polygon.rectangle(
                                x_min - pad, y_min - pad, x_max + pad, y_max + pad
                            ),
                        )
                    )

    lib.add_cell(NamedCell(name=cell.cell_name, polygons=top_polygons))
    return lib


# --- Phase 5: DRC-aware lowering (sky130-DRC-clean inter-finger spacing) ---


def lower_mos_finger_geometry_drc_aware(
    layout: SymbolicLayout,
    w_um: float,
    l_um: float,
    pdk_layers: LayerMap,
    fin_pitch_nm: float,
    min_poly_space_nm: float,
) -> List[LayerPolygon]:
    """Phase 5 DRC-aware mirror of Rust ``lower_mos_finger_geometry_drc_aware``.

    Uses ``max(fin_pitch_nm, min_poly_space_nm)`` for inter-finger spacing so the
    produced geometry can pass sky130/gf180mcu poly-poly minimum spacing DRC.

    ``min_poly_space_nm`` is the DRC value the caller looked up from the Phase 3c
    rule sets (e.g. 210 for sky130_poly_poly_space, 280 for gf180_poly_poly_space).
    Passing ``min_poly_space_nm=0`` is equivalent to ``lower_mos_finger_geometry``.
    """
    polys: List[LayerPolygon] = []
    l_nm = _nm_to_dbu(l_um * 1000.0)
    w_nm = _nm_to_dbu(w_um * 1000.0)
    fin_pitch = _nm_to_dbu(fin_pitch_nm)
    min_poly_space = _nm_to_dbu(min_poly_space_nm)
    inter_finger = max(fin_pitch, min_poly_space)
    extension = 2 * fin_pitch
    n = layout.finger_count

    poly = pdk_layers.gds_layer(LogicalLayer.POLY)
    ndiff = pdk_layers.gds_layer(LogicalLayer.NDIFF)
    if poly is None or ndiff is None:
        raise ValueError("PDK must define Poly and NDiff layers")

    for i in range(n):
        x_min = i * (l_nm + inter_finger)
        x_max = x_min + l_nm
        polys.append(
            LayerPolygon(
                layer=poly.layer,
                datatype=poly.datatype,
                polygon=Polygon.rectangle(x_min, 0, x_max, w_nm),
            )
        )

    ndiff_x_min = -extension
    ndiff_x_max = max(0, n - 1) * (l_nm + inter_finger) + l_nm + extension
    polys.append(
        LayerPolygon(
            layer=ndiff.layer,
            datatype=ndiff.datatype,
            polygon=Polygon.rectangle(
                ndiff_x_min, -fin_pitch, ndiff_x_max, w_nm + fin_pitch
            ),
        )
    )

    return polys


def lower_annotated_cell_drc_aware(
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    fin_pitch_nm: float,
    sizing: DeviceSizing,
    min_poly_space_nm: float,
) -> ParameterizedLibrary:
    """Phase 5 DRC-aware mirror of Rust ``lower_annotated_cell_drc_aware``.

    Threads ``min_poly_space_nm`` through to per-device lowering, and uses
    ``max(4 * fin_pitch, 2 * min_poly_space)`` for device-to-device gap so
    neighboring devices' outermost poly fingers also clear poly-poly DRC.
    """
    lib = ParameterizedLibrary(name=cell.cell_name)
    device_keys = sorted(cell.layouts.keys())
    fp = _nm_to_dbu(fin_pitch_nm)
    mps = _nm_to_dbu(min_poly_space_nm)
    device_gap = max(4 * fp, 2 * mps)

    symmetric_devices: Optional[List[str]] = None
    for anns in cell.annotations.values():
        for a in anns:
            if isinstance(a, Symmetric):
                symmetric_devices = list(a.devices)
                break
        if symmetric_devices is not None:
            break

    placement_x = 0
    top_polygons: List[LayerPolygon] = []

    for dev in device_keys:
        layout = cell.layouts[dev]
        w_um, l_um = sizing.get(dev, (1.0, 0.5))
        polys = lower_mos_finger_geometry_drc_aware(
            layout, w_um, l_um, pdk_layers, fin_pitch_nm, min_poly_space_nm
        )

        # Emit per-device cell at local origin (no placement / no mirror).
        lib.add_cell(NamedCell(name=dev, polygons=[
            LayerPolygon(
                layer=lp.layer,
                datatype=lp.datatype,
                polygon=Polygon(vertices=list(lp.polygon.vertices)),
            )
            for lp in polys
        ]))

        if symmetric_devices is not None and dev in symmetric_devices:
            half = len(symmetric_devices) // 2
            idx = symmetric_devices.index(dev)
            if idx >= half:
                bb = _bbox(polys)
                if bb is not None:
                    polys = _mirror_x(polys, (bb[0] + bb[2]) // 2)

        polys = _translate(polys, placement_x, 0)
        bb_after = _bbox(polys)
        if bb_after is not None:
            placement_x = bb_after[2] + device_gap
        top_polygons.extend(polys)

    guard_well: Optional[WellType] = None
    for anns in cell.annotations.values():
        for a in anns:
            if isinstance(a, GuardRing):
                guard_well = a.well_type
                break
        if guard_well is not None:
            break

    if guard_well is not None:
        pad = 2 * fp
        bb = _bbox(top_polygons)
        if bb is not None:
            x_min, y_min, x_max, y_max = bb
            if guard_well == WellType.NWELL:
                nwell = pdk_layers.gds_layer(LogicalLayer.NWELL)
                if nwell is not None:
                    top_polygons.append(
                        LayerPolygon(
                            layer=nwell.layer,
                            datatype=nwell.datatype,
                            polygon=Polygon.rectangle(
                                x_min - pad, y_min - pad, x_max + pad, y_max + pad
                            ),
                        )
                    )
            elif guard_well == WellType.PWELL:
                pmarker = pdk_layers.gds_layer(LogicalLayer.PWELL_MARKER)
                if pmarker is not None:
                    top_polygons.append(
                        LayerPolygon(
                            layer=pmarker.layer,
                            datatype=pmarker.datatype,
                            polygon=Polygon.rectangle(
                                x_min - pad, y_min - pad, x_max + pad, y_max + pad
                            ),
                        )
                    )

    lib.add_cell(NamedCell(name=cell.cell_name, polygons=top_polygons))
    return lib


def lower_annotated_cell_drc_complete(
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    fin_pitch_nm: float,
    sizing: DeviceSizing,
    *,
    min_poly_space_nm: Optional[float] = None,
    min_well_diff_enclosure_nm: Optional[float] = None,
) -> ParameterizedLibrary:
    """Phase 5.3 DRC-complete wrapper: auto-look up + apply all available
    DRC rule values for sky130 / gf180mcu.

    Combines Phase 5.1 (poly-space lookup) + Phase 5.2 (enclosure lookup) +
    Phase 5 (drc-aware lowering) + nwell expansion post-processing.
    """
    if min_poly_space_nm is None:
        min_poly_space_nm = min_poly_space_nm_for_pdk_layers(pdk_layers)
    if min_well_diff_enclosure_nm is None:
        min_well_diff_enclosure_nm = nwell_diff_enclosure_nm_for_pdk_layers(pdk_layers)

    lib = lower_annotated_cell_drc_aware(
        cell, pdk_layers, fin_pitch_nm, sizing, min_poly_space_nm
    )

    phase5_pad_nm = 2.0 * fin_pitch_nm
    extra_pad_nm = min_well_diff_enclosure_nm - phase5_pad_nm
    if extra_pad_nm <= 0.0:
        return lib

    extra_pad = _nm_to_dbu(extra_pad_nm)
    nwell_gds = pdk_layers.gds_layer(LogicalLayer.NWELL)
    pwell_gds = pdk_layers.gds_layer(LogicalLayer.PWELL_MARKER)
    well_layer_keys = set()
    if nwell_gds is not None:
        well_layer_keys.add((nwell_gds.layer, nwell_gds.datatype))
    if pwell_gds is not None:
        well_layer_keys.add((pwell_gds.layer, pwell_gds.datatype))

    if not well_layer_keys:
        return lib

    for named_cell in lib.cells:
        for lp in named_cell.polygons:
            if (lp.layer, lp.datatype) not in well_layer_keys:
                continue
            verts = lp.polygon.vertices
            if not verts:
                continue
            xs = [v.x for v in verts]
            ys = [v.y for v in verts]
            x_min = min(xs) - extra_pad
            y_min = min(ys) - extra_pad
            x_max = max(xs) + extra_pad
            y_max = max(ys) + extra_pad
            lp.polygon = Polygon.rectangle(x_min, y_min, x_max, y_max)

    return lib
