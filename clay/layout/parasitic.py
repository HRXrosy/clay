"""Phase 7 — Parasitic-aware extraction stub.

Given a ParameterizedLibrary output from Phase 3a/5 lowering, estimate
per-layer parasitic capacitance using area + fringe models. Per-layer
totals (not per-net) since current Phase 3a lowering doesn't have net
information; per-net extraction is Phase 8+ when routing layer lands.

Constants are TENTATIVE; sourced from approximate PDK process averages.
Phase 7 acceptance is structural — that the estimator wires PDK
constants to polygon geometry correctly. Calibration against silicon
data is Phase 7.x follow-up.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backend import (
    ParameterizedLibrary,
    NamedCell,
    LayerPolygon,
    Polygon,
    Point,
    LayerMap,
    LogicalLayer,
    Sky130LayerMap,
    Gf180mcuLayerMap,
)


# === PDK parasitic constants (aF/um^2 for area, aF/um for fringe) ===
# Values approximate; calibrate against silicon data in Phase 7.x.

@dataclass(frozen=True)
class LayerParasiticParams:
    """Per-layer parasitic constants. area_cap = aF/um^2, fringe_cap = aF/um."""
    area_aF_per_um2: float
    fringe_aF_per_um: float


# sky130 per-layer parasitic (logical layer keyed)
SKY130_PARASITIC: Dict[LogicalLayer, LayerParasiticParams] = {
    LogicalLayer.NDIFF: LayerParasiticParams(area_aF_per_um2=100.0, fringe_aF_per_um=50.0),
    LogicalLayer.PDIFF: LayerParasiticParams(area_aF_per_um2=100.0, fringe_aF_per_um=50.0),
    LogicalLayer.POLY:  LayerParasiticParams(area_aF_per_um2=80.0,  fringe_aF_per_um=40.0),
    LogicalLayer.METAL1: LayerParasiticParams(area_aF_per_um2=30.0, fringe_aF_per_um=25.0),
    LogicalLayer.NWELL: LayerParasiticParams(area_aF_per_um2=5.0,   fringe_aF_per_um=2.0),
}

# gf180mcu per-layer parasitic
GF180MCU_PARASITIC: Dict[LogicalLayer, LayerParasiticParams] = {
    LogicalLayer.NDIFF: LayerParasiticParams(area_aF_per_um2=110.0, fringe_aF_per_um=55.0),
    LogicalLayer.PDIFF: LayerParasiticParams(area_aF_per_um2=110.0, fringe_aF_per_um=55.0),
    LogicalLayer.POLY:  LayerParasiticParams(area_aF_per_um2=90.0,  fringe_aF_per_um=45.0),
    LogicalLayer.METAL1: LayerParasiticParams(area_aF_per_um2=35.0, fringe_aF_per_um=30.0),
    LogicalLayer.NWELL: LayerParasiticParams(area_aF_per_um2=6.0,   fringe_aF_per_um=2.5),
}


def parasitic_params_for_pdk_layers(pdk_layers: LayerMap) -> Dict[LogicalLayer, LayerParasiticParams]:
    """Auto-look up parasitic params dict by PDK LayerMap type."""
    name = type(pdk_layers).__name__.lower()
    if "sky130" in name:
        return SKY130_PARASITIC
    if "gf180" in name:
        return GF180MCU_PARASITIC
    raise ValueError(
        f"unknown PDK LayerMap type: {type(pdk_layers).__name__}; "
        f"parasitic params not available"
    )


# === Geometry helpers ===

def polygon_area_um2(polygon: Polygon, dbu_to_um: float = 0.001) -> float:
    """Compute polygon area in um^2 using shoelace formula.

    dbu_to_um: database unit conversion. ParameterizedLibrary uses i32 DBU;
    sky130/gf180mcu default 1 nm DBU = 0.001 um per DBU.
    """
    vertices = polygon.vertices
    n = len(vertices)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += vertices[i].x * vertices[j].y
        s -= vertices[j].x * vertices[i].y
    area_dbu2 = abs(s) / 2.0
    return area_dbu2 * (dbu_to_um ** 2)


def polygon_perimeter_um(polygon: Polygon, dbu_to_um: float = 0.001) -> float:
    """Compute polygon perimeter in um. Closes polygon (last edge from last vertex
    to first)."""
    import math
    vertices = polygon.vertices
    n = len(vertices)
    if n < 2:
        return 0.0
    perim_dbu = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = vertices[j].x - vertices[i].x
        dy = vertices[j].y - vertices[i].y
        perim_dbu += math.hypot(dx, dy)
    return perim_dbu * dbu_to_um


# === Parasitic report ===

@dataclass
class LayerParasiticEntry:
    """Per-layer parasitic estimate for a NamedCell."""
    layer: int
    datatype: int
    logical_layer: Optional[LogicalLayer]
    total_area_um2: float
    total_perimeter_um: float
    area_cap_aF: float    # area_um2 * area_cap_per_um2
    fringe_cap_aF: float  # perimeter_um * fringe_cap_per_um
    total_cap_aF: float   # area + fringe


@dataclass
class CellParasiticReport:
    """Parasitic report for one NamedCell."""
    cell_name: str
    per_layer: List[LayerParasiticEntry] = field(default_factory=list)

    @property
    def total_cap_aF(self) -> float:
        return sum(e.total_cap_aF for e in self.per_layer)


@dataclass
class LibraryParasiticReport:
    """Parasitic report for an entire ParameterizedLibrary."""
    library_name: str
    cells: List[CellParasiticReport] = field(default_factory=list)

    @property
    def total_cap_aF(self) -> float:
        return sum(c.total_cap_aF for c in self.cells)


# === Top-level extraction ===

def _build_logical_layer_lookup(pdk_layers: LayerMap) -> Dict:
    """Build a reverse lookup: (layer, datatype) -> LogicalLayer."""
    lookup = {}
    for ll in LogicalLayer:
        gds = pdk_layers.gds_layer(ll)
        if gds is not None:
            lookup[(gds.layer, gds.datatype)] = ll
    return lookup


def extract_parasitic(
    lib: ParameterizedLibrary,
    pdk_layers: LayerMap,
    parasitic_params: Optional[Dict[LogicalLayer, LayerParasiticParams]] = None,
) -> LibraryParasiticReport:
    """Estimate per-layer parasitic capacitance for every cell in the library.

    Args:
        lib: ParameterizedLibrary from Phase 3a/5 lowering
        pdk_layers: PDK LayerMap (Sky130LayerMap / Gf180mcuLayerMap)
        parasitic_params: optional override of PDK-specific constants;
            None auto-looks up via parasitic_params_for_pdk_layers

    Returns:
        LibraryParasiticReport with per-cell + per-layer breakdown.
    """
    if parasitic_params is None:
        parasitic_params = parasitic_params_for_pdk_layers(pdk_layers)

    dbu_to_um = lib.unit_user  # typical 0.001

    layer_lookup = _build_logical_layer_lookup(pdk_layers)

    lib_report = LibraryParasiticReport(library_name=lib.name)

    for cell in lib.cells:
        cell_report = CellParasiticReport(cell_name=cell.name)
        # Aggregate by (layer, datatype) key
        by_layer_key: Dict = {}
        for lp in cell.polygons:
            key = (lp.layer, lp.datatype)
            if key not in by_layer_key:
                by_layer_key[key] = {"area": 0.0, "perim": 0.0}
            by_layer_key[key]["area"] += polygon_area_um2(lp.polygon, dbu_to_um)
            by_layer_key[key]["perim"] += polygon_perimeter_um(lp.polygon, dbu_to_um)

        for (layer, datatype), stats in by_layer_key.items():
            logical = layer_lookup.get((layer, datatype))
            if logical is None or logical not in parasitic_params:
                # Unknown layer or no params; estimate as 0 (don't crash)
                params = LayerParasiticParams(area_aF_per_um2=0.0, fringe_aF_per_um=0.0)
            else:
                params = parasitic_params[logical]
            area_cap = stats["area"] * params.area_aF_per_um2
            fringe_cap = stats["perim"] * params.fringe_aF_per_um
            cell_report.per_layer.append(LayerParasiticEntry(
                layer=layer,
                datatype=datatype,
                logical_layer=logical,
                total_area_um2=stats["area"],
                total_perimeter_um=stats["perim"],
                area_cap_aF=area_cap,
                fringe_cap_aF=fringe_cap,
                total_cap_aF=area_cap + fringe_cap,
            ))
        # Sort per_layer by total_cap descending for human readability
        cell_report.per_layer.sort(key=lambda e: -e.total_cap_aF)
        lib_report.cells.append(cell_report)

    return lib_report
