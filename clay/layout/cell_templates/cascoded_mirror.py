"""Cascoded current mirror cell template factory (Phase 6.1).

A cascoded current mirror is the standard analog load for OTAs / amplifiers.
4 devices: M_ref_low + M_ref_cascode + M_out_low + M_out_cascode.

Topology (schematic, not layout):

    vdd
     |
     +---+---+
     |   |   |
    M_ref_cas  M_out_cas
     |   |
    M_ref_low  M_out_low
     |   |   |
     +---+---+
     |
    gnd / source-coupled-pair tail

Annotations baked in:
- Matching(M_ref_low, M_out_low): bottom mirror pair matched
- Matching(M_ref_cascode, M_out_cascode): cascode pair matched
- Adjacent(M_ref_low, M_out_low): bottom pair adjacent (Fraction(2, 1))
- Adjacent(M_ref_cascode, M_out_cascode): cascode pair adjacent
- Optional Symmetric across the mirror axis (default on)
"""
from __future__ import annotations
from fractions import Fraction
from typing import Tuple

from ..annotations import (
    AnnotatedCell,
    Matching,
    MatchingKind,
    Adjacent,
    Symmetric,
    SymmetryAxis,
)
from ..symbolic_layout import SymbolicLayout
from ..backend import DeviceSizing


def build_cascoded_current_mirror(
    name: str,
    w_um_low: float,
    l_um_low: float,
    w_um_cas: float,
    l_um_cas: float,
    finger_count_low: int = 4,
    finger_count_cas: int = 4,
    *,
    matching_kind: MatchingKind = MatchingKind.RANDOM_VT,
    with_symmetric: bool = True,
    with_adjacent: bool = True,
) -> Tuple[AnnotatedCell, DeviceSizing]:
    """Build a 4-MOS cascoded current mirror.

    Args:
        name: cell name
        w_um_low / l_um_low: bottom mirror device sizing (M_ref_low / M_out_low)
        w_um_cas / l_um_cas: cascode device sizing (M_ref_cascode / M_out_cascode)
        finger_count_low: M_ref_low / M_out_low finger count (must match for spatial validation)
        finger_count_cas: cascode finger count (must match for spatial validation)
        matching_kind: matching annotation kind for both pairs
        with_symmetric: add a top-level Symmetric annotation across the mirror axis
        with_adjacent: add Adjacent annotations for both matched pairs

    Returns:
        (AnnotatedCell, DeviceSizing) — DeviceSizing has 4 entries keyed
        by device name (M_ref_low, M_ref_cascode, M_out_low, M_out_cascode).

    Raises:
        ValueError on non-positive widths/lengths or zero finger counts.
    """
    if finger_count_low < 1 or finger_count_cas < 1:
        raise ValueError("finger counts must be >= 1")
    for v, name_v in [(w_um_low, "w_um_low"), (l_um_low, "l_um_low"),
                       (w_um_cas, "w_um_cas"), (l_um_cas, "l_um_cas")]:
        if v <= 0:
            raise ValueError(f"{name_v} must be > 0, got {v}")

    cell = AnnotatedCell(name)
    cell.set_layout("M_ref_low", SymbolicLayout(finger_count=finger_count_low))
    cell.set_layout("M_out_low", SymbolicLayout(finger_count=finger_count_low))
    cell.set_layout("M_ref_cascode", SymbolicLayout(finger_count=finger_count_cas))
    cell.set_layout("M_out_cascode", SymbolicLayout(finger_count=finger_count_cas))

    # Bidirectional Matching for both pairs
    cell.add_annotation("M_ref_low", Matching(other="M_out_low", kind=matching_kind))
    cell.add_annotation("M_out_low", Matching(other="M_ref_low", kind=matching_kind))
    cell.add_annotation("M_ref_cascode", Matching(other="M_out_cascode", kind=matching_kind))
    cell.add_annotation("M_out_cascode", Matching(other="M_ref_cascode", kind=matching_kind))

    if with_adjacent:
        cell.add_annotation("M_ref_low",
                            Adjacent(to="M_out_low",
                                     max_separation_multiplier=Fraction(2, 1)))
        cell.add_annotation("M_out_low",
                            Adjacent(to="M_ref_low",
                                     max_separation_multiplier=Fraction(2, 1)))
        cell.add_annotation("M_ref_cascode",
                            Adjacent(to="M_out_cascode",
                                     max_separation_multiplier=Fraction(2, 1)))
        cell.add_annotation("M_out_cascode",
                            Adjacent(to="M_ref_cascode",
                                     max_separation_multiplier=Fraction(2, 1)))

    if with_symmetric:
        # Top-level cell symmetric across mirror axis: ref side mirrors to out side
        cell.add_annotation("M_ref_low",
                            Symmetric(devices=["M_ref_low", "M_out_low",
                                               "M_ref_cascode", "M_out_cascode"],
                                      axis=SymmetryAxis.VERTICAL))

    sizing: DeviceSizing = {
        "M_ref_low": (w_um_low, l_um_low),
        "M_out_low": (w_um_low, l_um_low),
        "M_ref_cascode": (w_um_cas, l_um_cas),
        "M_out_cascode": (w_um_cas, l_um_cas),
    }
    return cell, sizing
