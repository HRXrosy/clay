"""Telescopic cascode amplifier cell template (Phase 6.4).

Compact 7-MOS telescopic stack: NMOS input pair + NMOS cascode pair on
top + PMOS load pair as current source + single NMOS tail. Lower power
than folded cascode (Phase 6.3) but limited output swing.

Topology:
    vdd
     +
    M_p_l, M_p_r (PMOS load mirror)
     +
    M_n_cas_l, M_n_cas_r (NMOS cascode)
     +
    M_in_l, M_in_r (NMOS input pair)
     +
    M_tail (NMOS tail)
     +
    gnd
"""
from __future__ import annotations
from fractions import Fraction
from typing import Tuple

from ..annotations import (
    AnnotatedCell, Matching, MatchingKind, Symmetric, SymmetryAxis,
    Adjacent, GuardRing, WellType,
)
from ..symbolic_layout import SymbolicLayout
from ..backend import DeviceSizing


def build_telescopic_cascode(
    name: str,
    input_w_um: float,
    input_l_um: float,
    input_finger_count: int = 4,
    cascode_w_um: float = 4.0,
    cascode_l_um: float = 0.5,
    cascode_finger_count: int = 4,
    pmos_load_w_um: float = 8.0,
    pmos_load_l_um: float = 1.0,
    pmos_load_finger_count: int = 4,
    tail_w_um: float = 8.0,
    tail_l_um: float = 1.0,
    tail_finger_count: int = 4,
    *,
    matching_kind: MatchingKind = MatchingKind.RANDOM_VT,
    with_input_pair_symmetric: bool = True,
    with_cascode_symmetric: bool = True,
    with_pmos_load_symmetric: bool = True,
    with_pair_adjacent: bool = True,
    with_guard_ring: bool = False,
    guard_ring_well: WellType = WellType.NWELL,
) -> Tuple[AnnotatedCell, DeviceSizing]:
    """Build a 7-MOS telescopic cascode amplifier."""
    finger_args = [
        (input_finger_count, "input_finger_count"),
        (cascode_finger_count, "cascode_finger_count"),
        (pmos_load_finger_count, "pmos_load_finger_count"),
        (tail_finger_count, "tail_finger_count"),
    ]
    for v, vn in finger_args:
        if v < 1:
            raise ValueError(f"{vn} must be >= 1, got {v}")
    size_args = [
        (input_w_um, "input_w_um"), (input_l_um, "input_l_um"),
        (cascode_w_um, "cascode_w_um"), (cascode_l_um, "cascode_l_um"),
        (pmos_load_w_um, "pmos_load_w_um"), (pmos_load_l_um, "pmos_load_l_um"),
        (tail_w_um, "tail_w_um"), (tail_l_um, "tail_l_um"),
    ]
    for v, vn in size_args:
        if v <= 0:
            raise ValueError(f"{vn} must be > 0, got {v}")

    cell = AnnotatedCell(name)
    cell.set_layout("M_in_l", SymbolicLayout(finger_count=input_finger_count))
    cell.set_layout("M_in_r", SymbolicLayout(finger_count=input_finger_count))
    cell.set_layout("M_n_cas_l", SymbolicLayout(finger_count=cascode_finger_count))
    cell.set_layout("M_n_cas_r", SymbolicLayout(finger_count=cascode_finger_count))
    cell.set_layout("M_p_l", SymbolicLayout(finger_count=pmos_load_finger_count))
    cell.set_layout("M_p_r", SymbolicLayout(finger_count=pmos_load_finger_count))
    cell.set_layout("M_tail", SymbolicLayout(finger_count=tail_finger_count))

    # 3 pairs of bidirectional Matching
    cell.add_annotation("M_in_l", Matching(other="M_in_r", kind=matching_kind))
    cell.add_annotation("M_in_r", Matching(other="M_in_l", kind=matching_kind))
    cell.add_annotation("M_n_cas_l", Matching(other="M_n_cas_r", kind=matching_kind))
    cell.add_annotation("M_n_cas_r", Matching(other="M_n_cas_l", kind=matching_kind))
    cell.add_annotation("M_p_l", Matching(other="M_p_r", kind=matching_kind))
    cell.add_annotation("M_p_r", Matching(other="M_p_l", kind=matching_kind))

    if with_input_pair_symmetric:
        cell.add_annotation("M_in_l",
            Symmetric(devices=["M_in_l", "M_in_r"], axis=SymmetryAxis.VERTICAL))
    if with_cascode_symmetric:
        cell.add_annotation("M_n_cas_l",
            Symmetric(devices=["M_n_cas_l", "M_n_cas_r"], axis=SymmetryAxis.VERTICAL))
    if with_pmos_load_symmetric:
        cell.add_annotation("M_p_l",
            Symmetric(devices=["M_p_l", "M_p_r"], axis=SymmetryAxis.VERTICAL))

    if with_pair_adjacent:
        cell.add_annotation("M_in_l",
            Adjacent(to="M_in_r", max_separation_multiplier=Fraction(2, 1)))
        cell.add_annotation("M_n_cas_l",
            Adjacent(to="M_n_cas_r", max_separation_multiplier=Fraction(2, 1)))
        cell.add_annotation("M_p_l",
            Adjacent(to="M_p_r", max_separation_multiplier=Fraction(2, 1)))

    if with_guard_ring:
        cell.add_annotation("M_in_l", GuardRing(well_type=guard_ring_well))

    sizing: DeviceSizing = {
        "M_in_l": (input_w_um, input_l_um),
        "M_in_r": (input_w_um, input_l_um),
        "M_n_cas_l": (cascode_w_um, cascode_l_um),
        "M_n_cas_r": (cascode_w_um, cascode_l_um),
        "M_p_l": (pmos_load_w_um, pmos_load_l_um),
        "M_p_r": (pmos_load_w_um, pmos_load_l_um),
        "M_tail": (tail_w_um, tail_l_um),
    }
    return cell, sizing
