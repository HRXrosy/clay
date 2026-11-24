"""Two-stage Miller amplifier cell template (Phase 6.5).

7-transistor + 1-capacitor topology: 5T-OTA front-end (M_in pair + M_load
pair + M_tail) feeds Stage-2 common-source amplifier (M_cs + M_cs_load),
with Miller compensation capacitor C_miller between Stage-2 output and
Stage-1 output node for pole splitting.

The capacitor is modeled as a SymbolicLayout placeholder; future Phase
8+ may add a proper MIM/MoM cap primitive to the layout substrate.
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


def build_two_stage_miller(
    name: str,
    input_w_um: float,
    input_l_um: float,
    input_finger_count: int = 4,
    load_w_um: float = 4.0,
    load_l_um: float = 1.0,
    load_finger_count: int = 4,
    tail_w_um: float = 8.0,
    tail_l_um: float = 1.0,
    tail_finger_count: int = 4,
    cs_w_um: float = 16.0,
    cs_l_um: float = 0.5,
    cs_finger_count: int = 8,
    cs_load_w_um: float = 32.0,
    cs_load_l_um: float = 1.0,
    cs_load_finger_count: int = 8,
    miller_cap_finger_count: int = 4,
    *,
    matching_kind: MatchingKind = MatchingKind.RANDOM_VT,
    with_input_pair_symmetric: bool = True,
    with_load_pair_symmetric: bool = True,
    with_input_to_tail_adjacent: bool = True,
    with_cs_to_cs_load_adjacent: bool = True,
    with_guard_ring: bool = False,
    guard_ring_well: WellType = WellType.NWELL,
) -> Tuple[AnnotatedCell, DeviceSizing]:
    """Build a 7-MOS + 1-cap two-stage Miller amplifier."""
    finger_args = [
        (input_finger_count, "input_finger_count"),
        (load_finger_count, "load_finger_count"),
        (tail_finger_count, "tail_finger_count"),
        (cs_finger_count, "cs_finger_count"),
        (cs_load_finger_count, "cs_load_finger_count"),
        (miller_cap_finger_count, "miller_cap_finger_count"),
    ]
    for v, vn in finger_args:
        if v < 1:
            raise ValueError(f"{vn} must be >= 1, got {v}")
    size_args = [
        (input_w_um, "input_w_um"), (input_l_um, "input_l_um"),
        (load_w_um, "load_w_um"), (load_l_um, "load_l_um"),
        (tail_w_um, "tail_w_um"), (tail_l_um, "tail_l_um"),
        (cs_w_um, "cs_w_um"), (cs_l_um, "cs_l_um"),
        (cs_load_w_um, "cs_load_w_um"), (cs_load_l_um, "cs_load_l_um"),
    ]
    for v, vn in size_args:
        if v <= 0:
            raise ValueError(f"{vn} must be > 0, got {v}")

    cell = AnnotatedCell(name)

    # Stage 1: 5T-OTA
    cell.set_layout("M_in_p", SymbolicLayout(finger_count=input_finger_count))
    cell.set_layout("M_in_n", SymbolicLayout(finger_count=input_finger_count))
    cell.set_layout("M_load_p", SymbolicLayout(finger_count=load_finger_count))
    cell.set_layout("M_load_n", SymbolicLayout(finger_count=load_finger_count))
    cell.set_layout("M_tail", SymbolicLayout(finger_count=tail_finger_count))

    # Stage 2: common-source amp + load
    cell.set_layout("M_cs", SymbolicLayout(finger_count=cs_finger_count))
    cell.set_layout("M_cs_load", SymbolicLayout(finger_count=cs_load_finger_count))

    # Miller compensation capacitor (placeholder; future Phase 8+ adds MIM cap)
    cell.set_layout("C_miller", SymbolicLayout(finger_count=miller_cap_finger_count))

    # Stage 1 matching: input pair + load pair (bidirectional)
    cell.add_annotation("M_in_p", Matching(other="M_in_n", kind=matching_kind))
    cell.add_annotation("M_in_n", Matching(other="M_in_p", kind=matching_kind))
    cell.add_annotation("M_load_p", Matching(other="M_load_n", kind=matching_kind))
    cell.add_annotation("M_load_n", Matching(other="M_load_p", kind=matching_kind))

    if with_input_pair_symmetric:
        cell.add_annotation("M_in_p",
            Symmetric(devices=["M_in_p", "M_in_n"], axis=SymmetryAxis.VERTICAL))
    if with_load_pair_symmetric:
        cell.add_annotation("M_load_p",
            Symmetric(devices=["M_load_p", "M_load_n"], axis=SymmetryAxis.VERTICAL))

    if with_input_to_tail_adjacent:
        cell.add_annotation("M_in_p",
            Adjacent(to="M_tail", max_separation_multiplier=Fraction(3, 1)))
        cell.add_annotation("M_in_n",
            Adjacent(to="M_tail", max_separation_multiplier=Fraction(3, 1)))

    if with_cs_to_cs_load_adjacent:
        cell.add_annotation("M_cs",
            Adjacent(to="M_cs_load", max_separation_multiplier=Fraction(2, 1)))
        cell.add_annotation("M_cs_load",
            Adjacent(to="M_cs", max_separation_multiplier=Fraction(2, 1)))

    if with_guard_ring:
        cell.add_annotation("M_in_p", GuardRing(well_type=guard_ring_well))

    sizing: DeviceSizing = {
        "M_in_p": (input_w_um, input_l_um),
        "M_in_n": (input_w_um, input_l_um),
        "M_load_p": (load_w_um, load_l_um),
        "M_load_n": (load_w_um, load_l_um),
        "M_tail": (tail_w_um, tail_l_um),
        "M_cs": (cs_w_um, cs_l_um),
        "M_cs_load": (cs_load_w_um, cs_load_l_um),
        # Capacitor placeholder: stub (1, 1) sizing; actual MIM cap dimensions
        # are TBD in Phase 8+ when capacitor primitive arrives.
        "C_miller": (1.0, 1.0),
    }
    return cell, sizing
