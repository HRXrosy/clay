"""Differential pair cell template factory."""
from __future__ import annotations
from typing import Tuple

from ..annotations import (
    AnnotatedCell, Matching, MatchingKind, GuardRing,
    Symmetric, SymmetryAxis, WellType,
)
from ..symbolic_layout import SymbolicLayout
from ..backend import DeviceSizing


def build_diff_pair(
    name: str,
    w_um: float,
    l_um: float,
    finger_count: int = 4,
    *,
    with_guard_ring: bool = False,
    guard_ring_well: WellType = WellType.NWELL,
    with_symmetric: bool = True,
    matching_kind: MatchingKind = MatchingKind.RANDOM_VT,
) -> Tuple[AnnotatedCell, DeviceSizing]:
    """Build a 2-MOS differential pair (M1, M2) with matching annotations.

    Returns (AnnotatedCell, DeviceSizing) — pass both into
    lower_annotated_cell_drc_aware.

    Annotations added:
    - M1 and M2 each get a Matching(other=the other one, kind=matching_kind)
      (bidirectional, so the spatial validator sees both)
    - If with_symmetric: M1 also gets a Symmetric([M1, M2], axis=Vertical)
    - If with_guard_ring: M1 gets a GuardRing(well_type=guard_ring_well)

    Args:
        name: cell name (also used as AnnotatedCell.cell_name)
        w_um: device width in um
        l_um: device length in um
        finger_count: per-device finger count (must be >= 1)
        with_guard_ring: if True, add GuardRing annotation
        guard_ring_well: well type if with_guard_ring (default NWELL)
        with_symmetric: if True, add Symmetric annotation
        matching_kind: kind of matching (RANDOM_VT / SYSTEMATIC / BOTH)

    Returns:
        (AnnotatedCell, DeviceSizing) — DeviceSizing is dict mapping device id to (w_um, l_um)
    """
    if finger_count < 1:
        raise ValueError(f"finger_count must be >= 1, got {finger_count}")
    if w_um <= 0 or l_um <= 0:
        raise ValueError(f"w_um and l_um must be > 0, got w={w_um}, l={l_um}")

    cell = AnnotatedCell(name)
    cell.set_layout("M1", SymbolicLayout(finger_count=finger_count))
    cell.set_layout("M2", SymbolicLayout(finger_count=finger_count))

    cell.add_annotation("M1", Matching(other="M2", kind=matching_kind))
    cell.add_annotation("M2", Matching(other="M1", kind=matching_kind))

    if with_symmetric:
        cell.add_annotation("M1", Symmetric(devices=["M1", "M2"], axis=SymmetryAxis.VERTICAL))

    if with_guard_ring:
        cell.add_annotation("M1", GuardRing(well_type=guard_ring_well))

    sizing: DeviceSizing = {"M1": (w_um, l_um), "M2": (w_um, l_um)}
    return cell, sizing
