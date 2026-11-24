"""Phase 7.5 - Edge-coupling capacitance extension to Phase 7 parasitic.

The Phase 7 parasitic estimator covers area capacitance (aF/um^2 over polygon
area) and fringe capacitance (aF/um around polygon perimeter), both of which
treat a polygon in isolation. Phase 7.5 adds two coupling effects that
require considering polygon pairs:

  1. Same-layer lateral coupling. Adjacent polygons on the same (layer,
     datatype) whose axis-aligned edges face each other within a threshold
     distance contribute lateral coupling proportional to facing edge length.

  2. Cross-layer overlap coupling. Polygons on different layers whose
     bounding boxes overlap contribute parallel-plate-like coupling
     proportional to overlap area.

Phase 7.5 is intentionally simple: bounding-box geometry only, axis-aligned
detection, no proper polygon-polygon intersection routines. The lowering
pipeline (Phase 3a) emits rectangles, so bounding-box analysis covers the
current cell library accurately. Curved or non-Manhattan polygons would
need a Phase 7.6 follow-up with full polygon clipping.

This module deliberately does NOT modify ``clay.layout.parasitic``; users
combine reports manually if they want a total (area + fringe + edge-coupling).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .backend import (
    LayerPolygon,
    NamedCell,
    ParameterizedLibrary,
    Point,
    Polygon,
)


# === Parameters & report ===


@dataclass(frozen=True)
class EdgeCouplingParams:
    """Per-process edge-coupling constants.

    Attributes:
        same_layer_lateral_aF_per_um: lateral coupling cap per um of facing
            edge length between two polygons on the same layer.
        cross_layer_overlap_aF_per_um2: cap per um^2 of overlap area between
            polygons on different layers.
    """

    same_layer_lateral_aF_per_um: float
    cross_layer_overlap_aF_per_um2: float


DEFAULT_EDGE_PARAMS = EdgeCouplingParams(
    same_layer_lateral_aF_per_um=10.0,
    cross_layer_overlap_aF_per_um2=50.0,
)


@dataclass
class EdgeCouplingReport:
    """Aggregate edge-coupling report across an entire ParameterizedLibrary."""

    same_layer_edge_couplings_aF: float
    cross_layer_overlap_couplings_aF: float

    @property
    def total_aF(self) -> float:
        return self.same_layer_edge_couplings_aF + self.cross_layer_overlap_couplings_aF


# === Geometry helpers ===


def _bbox_dbu(polygon: Polygon) -> Optional[Tuple[int, int, int, int]]:
    """Return (x_min, y_min, x_max, y_max) in DBU, or None for empty/degenerate."""
    if not polygon.vertices:
        return None
    xs = [v.x for v in polygon.vertices]
    ys = [v.y for v in polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_overlap_area_dbu2(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> int:
    """Overlap area in DBU^2 between two axis-aligned bounding boxes.

    Returns 0 when boxes only touch on an edge or are disjoint."""
    ax_min, ay_min, ax_max, ay_max = a
    bx_min, by_min, bx_max, by_max = b
    dx = min(ax_max, bx_max) - max(ax_min, bx_min)
    dy = min(ay_max, by_max) - max(ay_min, by_min)
    if dx <= 0 or dy <= 0:
        return 0
    return dx * dy


def _facing_edge_length_dbu(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    distance_threshold_dbu: int,
) -> int:
    """Length in DBU of axis-aligned edges that face each other within the
    distance threshold, between two bounding boxes.

    Considers four facing-edge configurations: A-right / B-left, A-left /
    B-right, A-top / B-bottom, A-bottom / B-top. Returns 0 if no pair faces
    within the threshold or if the boxes overlap (overlap is handled by the
    cross-layer pass, not lateral coupling).

    For each configuration, the facing length is the overlap of the
    perpendicular projection of A and B onto the shared axis.
    """
    ax_min, ay_min, ax_max, ay_max = a
    bx_min, by_min, bx_max, by_max = b

    # If bboxes overlap in 2D, edges aren't "facing" laterally.
    if _bbox_overlap_area_dbu2(a, b) > 0:
        return 0

    total = 0

    # Horizontal facing: A on left of B (A.right faces B.left) or vice versa.
    # The perpendicular-projection overlap is on the y-axis.
    gap_x_ab = bx_min - ax_max  # A.right -> B.left
    gap_x_ba = ax_min - bx_max  # B.right -> A.left
    proj_y_overlap = min(ay_max, by_max) - max(ay_min, by_min)
    if proj_y_overlap > 0:
        if 0 <= gap_x_ab <= distance_threshold_dbu:
            total += proj_y_overlap
        elif 0 <= gap_x_ba <= distance_threshold_dbu:
            total += proj_y_overlap

    # Vertical facing: A below B (A.top faces B.bottom) or vice versa.
    # The perpendicular-projection overlap is on the x-axis.
    gap_y_ab = by_min - ay_max  # A.top -> B.bottom
    gap_y_ba = ay_min - by_max  # B.top -> A.bottom
    proj_x_overlap = min(ax_max, bx_max) - max(ax_min, bx_min)
    if proj_x_overlap > 0:
        if 0 <= gap_y_ab <= distance_threshold_dbu:
            total += proj_x_overlap
        elif 0 <= gap_y_ba <= distance_threshold_dbu:
            total += proj_x_overlap

    return total


# === Per-cell pass ===


def _cell_same_layer_facing_um(
    polygons: List[LayerPolygon],
    distance_threshold_dbu: int,
    dbu_to_um: float,
) -> float:
    """Total facing-edge length (um) summed over all same-layer polygon pairs
    in a cell within the threshold."""
    bboxes_by_key: dict = {}
    for lp in polygons:
        bb = _bbox_dbu(lp.polygon)
        if bb is None:
            continue
        key = (lp.layer, lp.datatype)
        bboxes_by_key.setdefault(key, []).append(bb)

    total_dbu = 0
    for key, bboxes in bboxes_by_key.items():
        n = len(bboxes)
        for i in range(n):
            for j in range(i + 1, n):
                total_dbu += _facing_edge_length_dbu(
                    bboxes[i], bboxes[j], distance_threshold_dbu
                )
    return total_dbu * dbu_to_um


def _cell_cross_layer_overlap_um2(
    polygons: List[LayerPolygon],
    dbu_to_um: float,
) -> float:
    """Total cross-layer bbox overlap area (um^2) summed over all polygon
    pairs from different (layer, datatype) keys."""
    entries: List[Tuple[Tuple[int, int], Tuple[int, int, int, int]]] = []
    for lp in polygons:
        bb = _bbox_dbu(lp.polygon)
        if bb is None:
            continue
        entries.append(((lp.layer, lp.datatype), bb))

    total_dbu2 = 0
    n = len(entries)
    for i in range(n):
        key_i, bb_i = entries[i]
        for j in range(i + 1, n):
            key_j, bb_j = entries[j]
            if key_i == key_j:
                continue  # same-layer handled by the lateral pass
            total_dbu2 += _bbox_overlap_area_dbu2(bb_i, bb_j)
    return total_dbu2 * (dbu_to_um ** 2)


# === Top-level entry point ===


def estimate_edge_coupling(
    lib: ParameterizedLibrary,
    same_layer_distance_threshold_nm: int = 200,
    edge_params: Optional[EdgeCouplingParams] = None,
) -> EdgeCouplingReport:
    """Estimate aggregate edge-coupling capacitance for a ParameterizedLibrary.

    Args:
        lib: ParameterizedLibrary from Phase 3a/5 lowering.
        same_layer_distance_threshold_nm: maximum gap (in nm) between two
            same-layer polygon edges for them to count as coupled. Defaults
            to 200 nm, an aggressive minimum-spacing-class threshold.
        edge_params: optional override of default per-process constants.

    Returns:
        EdgeCouplingReport with same-layer + cross-layer totals.

    Phase 7.5 caveat: bounding-box geometry only. Sufficient for axis-aligned
    rectangles emitted by the current Phase 3a lowering. Curved/Manhattan-
    polygon support is a Phase 7.6 follow-up.
    """
    if edge_params is None:
        edge_params = DEFAULT_EDGE_PARAMS

    dbu_to_um = lib.unit_user  # typically 0.001 (1 nm per DBU)
    # Threshold expressed in DBU. lib.unit_user converts DBU -> um, so:
    # threshold_um = threshold_nm * 1e-3; threshold_dbu = threshold_um / dbu_to_um
    threshold_um = same_layer_distance_threshold_nm * 1e-3
    if dbu_to_um <= 0:
        raise ValueError(
            f"library unit_user must be > 0, got {dbu_to_um}"
        )
    distance_threshold_dbu = int(round(threshold_um / dbu_to_um))

    same_layer_um = 0.0
    cross_layer_um2 = 0.0
    for cell in lib.cells:
        same_layer_um += _cell_same_layer_facing_um(
            cell.polygons, distance_threshold_dbu, dbu_to_um
        )
        cross_layer_um2 += _cell_cross_layer_overlap_um2(cell.polygons, dbu_to_um)

    same_cap = same_layer_um * edge_params.same_layer_lateral_aF_per_um
    cross_cap = cross_layer_um2 * edge_params.cross_layer_overlap_aF_per_um2

    return EdgeCouplingReport(
        same_layer_edge_couplings_aF=same_cap,
        cross_layer_overlap_couplings_aF=cross_cap,
    )
