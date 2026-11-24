"""Per-cell layout statistics CSV exporter.

Walks a ParameterizedLibrary and emits per-NamedCell metrics suitable for
quick triage, CI dashboards, and regression tracking. Point coordinates in
the underlying backend are integer nanometres; this module converts to
micrometres (divide by 1000.0) for the user-facing bbox / area columns.

Columns emitted:
    cell_name, polygon_count, total_vertex_count,
    bbox_w_um, bbox_h_um, area_um2, unique_layer_count

``area_um2`` is the **bounding-box area** (width * height), not the sum of
polygon areas. ``unique_layer_count`` counts distinct ``(layer, datatype)``
GDS layer pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .backend import NamedCell, ParameterizedLibrary


_NM_PER_UM = 1000.0


@dataclass(frozen=True)
class CellStats:
    cell_name: str
    polygon_count: int
    total_vertex_count: int
    bbox_w_um: float
    bbox_h_um: float
    area_um2: float          # bbox area (not polygon area)
    unique_layer_count: int  # distinct (layer, datatype) pairs


def compute_cell_stats(cell: NamedCell) -> CellStats:
    """Walk LayerPolygons, sum vertex counts, find bbox, count distinct layers.

    Empty cell (no polygons, or polygons whose vertex lists are all empty)
    returns all zeros for the numeric fields. Point coordinates are integer
    nanometres in the backend; bbox dimensions are emitted in micrometres.
    """
    polygons = cell.polygons
    polygon_count = len(polygons)

    total_vertex_count = 0
    unique_layers = set()
    xs: List[int] = []
    ys: List[int] = []

    for lp in polygons:
        unique_layers.add((lp.layer, lp.datatype))
        verts = lp.polygon.vertices
        total_vertex_count += len(verts)
        for v in verts:
            xs.append(v.x)
            ys.append(v.y)

    if xs and ys:
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        bbox_w_um = (x_max - x_min) / _NM_PER_UM
        bbox_h_um = (y_max - y_min) / _NM_PER_UM
        area_um2 = bbox_w_um * bbox_h_um
    else:
        bbox_w_um = 0.0
        bbox_h_um = 0.0
        area_um2 = 0.0

    return CellStats(
        cell_name=cell.name,
        polygon_count=polygon_count,
        total_vertex_count=total_vertex_count,
        bbox_w_um=bbox_w_um,
        bbox_h_um=bbox_h_um,
        area_um2=area_um2,
        unique_layer_count=len(unique_layers),
    )


def compute_library_stats(lib: ParameterizedLibrary) -> List[CellStats]:
    """Compute per-cell stats for every cell in ``lib``, sorted by cell_name."""
    stats = [compute_cell_stats(c) for c in lib.cells]
    stats.sort(key=lambda s: s.cell_name)
    return stats


def render_stats_csv(stats: List[CellStats], include_header: bool = True) -> str:
    """Render a list of CellStats as a CSV string.

    Columns:
        cell_name, polygon_count, total_vertex_count,
        bbox_w_um, bbox_h_um, area_um2, unique_layer_count

    Float columns (bbox_w_um, bbox_h_um, area_um2) are formatted with 4
    decimal places. Integer columns are emitted without padding.
    """
    lines: List[str] = []
    if include_header:
        lines.append(
            "cell_name,polygon_count,total_vertex_count,"
            "bbox_w_um,bbox_h_um,area_um2,unique_layer_count"
        )
    for s in stats:
        lines.append(
            f"{s.cell_name},{s.polygon_count},{s.total_vertex_count},"
            f"{s.bbox_w_um:.4f},{s.bbox_h_um:.4f},{s.area_um2:.4f},"
            f"{s.unique_layer_count}"
        )
    return "\n".join(lines) + "\n"


def write_stats_csv(stats: List[CellStats], output_path: str) -> None:
    """Write the rendered CSV (with header) to ``output_path`` as UTF-8 text."""
    text = render_stats_csv(stats, include_header=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
