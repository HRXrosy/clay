"""DRC margin computation - measure how much spacing slack a laid-out cell
has above the PDK minimum spacing rules.

For a given AnnotatedCell + sizing, lower the cell to a ParameterizedLibrary
via the existing DRC-complete API and, per routing layer (POLY/M1/M2/M3),
scan all polygon X-coordinates to find the minimum gap between adjacent
polygons on that layer. Report the gap vs the PDK DRC minimum.

This is intentionally a heuristic. It uses X-projection only (no Y-overlap
check) so a polygon stack with vertically disjoint shapes can still report
a smaller-than-real margin - this matches the brief and keeps the routine
cheap to call from sizing loops that want a "headroom for shrinking" signal.

DIFF / NWELL layers are deliberately not measured here: their DRC rules are
enclosure-based, not spacing-based, and a separate enclosure-margin routine
would be the right home for those.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .annotations import AnnotatedCell
from .backend import (
    DeviceSizing,
    LayerMap,
    LayerPolygon,
    ParameterizedLibrary,
    lower_annotated_cell_drc_complete,
)


# Per-PDK routing-layer table: layer_name -> ((gds_layer, gds_datatype), drc_min_nm)
#
# Only spacing-rule layers are listed (POLY / M1 / M2 / M3). DIFF and NWELL
# are excluded by design - their DRC rules are enclosure-based.
_LAYER_TABLES: Dict[str, Dict[str, Tuple[Tuple[int, int], int]]] = {
    "sky130": {
        "POLY": ((66, 20), 210),
        "M1":   ((67, 20), 140),
        "M2":   ((68, 20), 140),
        "M3":   ((70, 20), 300),
    },
}


@dataclass(frozen=True)
class LayerMargin:
    """Per-layer DRC spacing margin record."""

    layer_name: str
    min_observed_nm: int   # smallest x-gap between adjacent polygons on this layer
    drc_minimum_nm: int    # PDK DRC minimum spacing rule for this layer
    margin_nm: int         # min_observed - drc_minimum  (positive = passes DRC)
    margin_pct: float      # 100 * margin / drc_minimum


@dataclass
class DrcMarginReport:
    """Aggregate DRC margin report for a laid-out cell."""

    cell_name: str
    layer_margins: List[LayerMargin] = field(default_factory=list)

    @property
    def has_negative_margin(self) -> bool:
        """True if any reported layer has a negative margin (would fail DRC)."""
        return any(m.margin_nm < 0 for m in self.layer_margins)

    @property
    def min_margin_layer(self) -> Optional[str]:
        """Name of the layer with the smallest margin (tightest constraint).

        Returns None when the report has no layer margins (empty cell or only
        single-polygon layers).
        """
        if not self.layer_margins:
            return None
        tightest = min(self.layer_margins, key=lambda m: m.margin_nm)
        return tightest.layer_name


def _min_x_gap_between_polygons(polys: List[LayerPolygon]) -> Optional[int]:
    """Return the smallest x-axis gap between adjacent polygons on a layer.

    Sort polygons by their left edge (min x), then walk the list comparing
    each polygon's right edge against the next polygon's left edge. A negative
    result means the two polygons overlap in X (no gap).

    Returns None when fewer than 2 polygons are present - there is no spacing
    to measure with a single polygon.
    """
    if len(polys) < 2:
        return None

    bboxes: List[Tuple[int, int]] = []
    for lp in polys:
        if not lp.polygon.vertices:
            continue
        xs = [v.x for v in lp.polygon.vertices]
        bboxes.append((min(xs), max(xs)))

    if len(bboxes) < 2:
        return None

    # Sort by left edge so adjacency is well-defined.
    bboxes.sort(key=lambda b: b[0])

    min_gap: Optional[int] = None
    # Track the running max of right edges so we always compare the next
    # polygon's left edge against the rightmost polygon seen so far - this
    # handles the case where a later polygon is fully contained within an
    # earlier one (gap would be negative / 0).
    max_right_so_far = bboxes[0][1]
    for left, right in bboxes[1:]:
        gap = left - max_right_so_far
        if min_gap is None or gap < min_gap:
            min_gap = gap
        if right > max_right_so_far:
            max_right_so_far = right

    return min_gap


def _collect_top_cell_polygons(lib: ParameterizedLibrary, cell_name: str) -> List[LayerPolygon]:
    """Pull the polygons of the top cell from the lowered library.

    The lower_annotated_cell_drc_complete API emits both per-device cells and
    a final top cell named after cell.cell_name; we measure margin on the top
    cell because that's where placement-induced spacing appears.
    """
    for named_cell in lib.cells:
        if named_cell.name == cell_name:
            return list(named_cell.polygons)
    # Fall back to the last cell (which is always the top in the existing API).
    if lib.cells:
        return list(lib.cells[-1].polygons)
    return []


def compute_drc_margin(
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    fin_pitch_nm: int,
    sizing: DeviceSizing,
    pdk_name: str = "sky130",
) -> DrcMarginReport:
    """Compute per-layer DRC spacing margin for a laid-out AnnotatedCell.

    Lowers the cell via lower_annotated_cell_drc_complete (read-only call -
    does not mutate the input cell or sizing) and, for each routing layer
    in the cell, computes the smallest X-axis gap between adjacent polygons
    on that layer. Compares the gap to the PDK DRC minimum spacing rule and
    emits a LayerMargin entry.

    Layers with zero or one polygon are skipped - there is no spacing to
    measure. DIFF and NWELL layers are skipped by design (enclosure rules,
    not spacing rules).

    Parameters
    ----------
    cell : AnnotatedCell
        Spatially-annotated cell to lay out and measure.
    pdk_layers : LayerMap
        PDK layer map (Sky130LayerMap or Gf180mcuLayerMap). Passed through
        to the lowering API.
    fin_pitch_nm : int
        PDK fin pitch in nm. Passed through to the lowering API.
    sizing : DeviceSizing
        Per-device (w_um, l_um) sizing dictionary.
    pdk_name : str, default "sky130"
        Which PDK rule table to use. Only "sky130" is supported at present.

    Returns
    -------
    DrcMarginReport
        Per-layer margin records plus convenience accessors.
    """
    if pdk_name not in _LAYER_TABLES:
        raise ValueError(
            f"unsupported pdk_name={pdk_name!r}; supported: {sorted(_LAYER_TABLES)}"
        )
    layer_table = _LAYER_TABLES[pdk_name]

    lib = lower_annotated_cell_drc_complete(
        cell, pdk_layers, float(fin_pitch_nm), sizing
    )

    top_polys = _collect_top_cell_polygons(lib, cell.cell_name)

    # Bucket polygons by (gds_layer, gds_datatype).
    by_gds: Dict[Tuple[int, int], List[LayerPolygon]] = {}
    for lp in top_polys:
        by_gds.setdefault((lp.layer, lp.datatype), []).append(lp)

    layer_margins: List[LayerMargin] = []
    # Iterate in a stable order (POLY -> M1 -> M2 -> M3) so the report is
    # deterministic regardless of polygon emit order.
    for layer_name, (gds_key, drc_min_nm) in layer_table.items():
        polys = by_gds.get(gds_key, [])
        if not polys:
            continue
        min_gap = _min_x_gap_between_polygons(polys)
        if min_gap is None:
            # Only one polygon on this layer; no spacing to measure.
            continue
        margin_nm = min_gap - drc_min_nm
        margin_pct = 100.0 * margin_nm / drc_min_nm if drc_min_nm > 0 else 0.0
        layer_margins.append(
            LayerMargin(
                layer_name=layer_name,
                min_observed_nm=int(min_gap),
                drc_minimum_nm=int(drc_min_nm),
                margin_nm=int(margin_nm),
                margin_pct=margin_pct,
            )
        )

    return DrcMarginReport(cell_name=cell.cell_name, layer_margins=layer_margins)
