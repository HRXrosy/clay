"""Phase 9 - net-aware parasitic extraction (baseline).

Given a ParameterizedLibrary + per-polygon net assignment, partition
Phase 7 parasitic estimates per net. Net assignment is caller-provided
(future Phase 9.x will derive nets from routing geometry overlap).

Pure Python; reuses Phase 7 layer-level parasitic constants and area/
perimeter helpers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .backend import (
    ParameterizedLibrary, NamedCell, LayerPolygon, LayerMap, LogicalLayer,
)
from .parasitic import (
    LayerParasiticParams, parasitic_params_for_pdk_layers,
    polygon_area_um2, polygon_perimeter_um,
)


# Net assignment key: (cell_name, polygon_index_within_cell) -> net_name
NetAssignment = Dict[Tuple[str, int], str]


@dataclass
class NetParasiticEntry:
    """Per-net parasitic estimate."""
    net_name: str
    total_area_um2: float
    total_perimeter_um: float
    total_cap_aF: float


@dataclass
class NetParasiticReport:
    """Net-level parasitic report for a ParameterizedLibrary."""
    library_name: str
    per_net: List[NetParasiticEntry] = field(default_factory=list)
    unassigned_cap_aF: float = 0.0   # Polygons not in net_map: aggregated separately

    @property
    def total_assigned_cap_aF(self) -> float:
        return sum(e.total_cap_aF for e in self.per_net)


def extract_parasitic_net_aware(
    lib: ParameterizedLibrary,
    pdk_layers: LayerMap,
    net_map: NetAssignment,
    parasitic_params: Optional[Dict[LogicalLayer, LayerParasiticParams]] = None,
) -> NetParasiticReport:
    """Net-aware parasitic extraction.

    Walks every polygon in the library, looks up its net via net_map
    using key (cell.name, polygon_index). Aggregates area + perimeter
    per net, multiplies by per-layer parasitic constants for cap totals.

    Polygons without a net assignment go to ``unassigned_cap_aF``.

    Args:
        lib: ParameterizedLibrary from Phase 3a/5 lowering
        pdk_layers: PDK LayerMap (Sky130LayerMap / Gf180mcuLayerMap)
        net_map: per-polygon net assignment (cell_name, polygon_index) -> net_name
        parasitic_params: optional override of per-layer constants

    Returns:
        NetParasiticReport with per_net entries + unassigned_cap_aF.
    """
    if parasitic_params is None:
        parasitic_params = parasitic_params_for_pdk_layers(pdk_layers)

    dbu_to_um = lib.unit_user

    # Reverse lookup: (layer, datatype) -> LogicalLayer
    layer_lookup: Dict[Tuple[int, int], LogicalLayer] = {}
    for ll in LogicalLayer:
        g = pdk_layers.gds_layer(ll)
        if g is not None:
            layer_lookup[(g.layer, g.datatype)] = ll

    # Aggregate per (net_name, logical_layer) -> area + perimeter
    net_layer_stats: Dict[Tuple[str, LogicalLayer], Dict[str, float]] = {}
    unassigned_area_aF = 0.0

    for cell in lib.cells:
        for poly_idx, lp in enumerate(cell.polygons):
            area = polygon_area_um2(lp.polygon, dbu_to_um)
            perim = polygon_perimeter_um(lp.polygon, dbu_to_um)
            ll = layer_lookup.get((lp.layer, lp.datatype))
            params = parasitic_params.get(ll) if ll is not None else None
            if params is None:
                continue   # Unknown layer - skip parasitic accounting
            area_cap = area * params.area_aF_per_um2
            fringe_cap = perim * params.fringe_aF_per_um
            cap = area_cap + fringe_cap

            net = net_map.get((cell.name, poly_idx))
            if net is None:
                unassigned_area_aF += cap
                continue
            key = (net, ll)
            if key not in net_layer_stats:
                net_layer_stats[key] = {"area": 0.0, "perim": 0.0, "cap": 0.0}
            net_layer_stats[key]["area"] += area
            net_layer_stats[key]["perim"] += perim
            net_layer_stats[key]["cap"] += cap

    # Aggregate per net (sum across logical layers)
    per_net_aggregate: Dict[str, Dict[str, float]] = {}
    for (net, _), stats in net_layer_stats.items():
        if net not in per_net_aggregate:
            per_net_aggregate[net] = {"area": 0.0, "perim": 0.0, "cap": 0.0}
        per_net_aggregate[net]["area"] += stats["area"]
        per_net_aggregate[net]["perim"] += stats["perim"]
        per_net_aggregate[net]["cap"] += stats["cap"]

    per_net_entries: List[NetParasiticEntry] = []
    for net_name, stats in per_net_aggregate.items():
        per_net_entries.append(NetParasiticEntry(
            net_name=net_name,
            total_area_um2=stats["area"],
            total_perimeter_um=stats["perim"],
            total_cap_aF=stats["cap"],
        ))
    # Sort by cap descending (most dominant net first)
    per_net_entries.sort(key=lambda e: -e.total_cap_aF)

    return NetParasiticReport(
        library_name=lib.name,
        per_net=per_net_entries,
        unassigned_cap_aF=unassigned_area_aF,
    )
