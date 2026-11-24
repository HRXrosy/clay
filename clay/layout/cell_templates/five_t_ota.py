"""5T-OTA cell template factory (Phase 6.2).

The 5-transistor OTA is the foundational analog amplifier topology:

       vdd
        |
    +---+---+
    |       |
   M_load_p M_load_n  (PMOS current-mirror load; diode-connected on plus side)
    |       |
    +--+----+    (out node = drain of M_load_n / M_in_n)
       |
       +--- v_out
       |
    +--+----+
    |       |
   M_in_p   M_in_n   (NMOS differential input pair)
    |       |
    +---+---+
        |
       M_tail        (NMOS tail current source)
        |
       gnd

5 transistors: input differential pair (matched), load pair (matched current
mirror), tail current source. The cell template bakes in the appropriate
spatial annotations:
- Matching for input pair (random VT)
- Matching for load pair
- Symmetric for both pairs across the OTA's vertical axis
- Adjacent: input pair → tail (close-coupled current source)
"""
from __future__ import annotations
from fractions import Fraction
from typing import Tuple

from ..annotations import (
    AnnotatedCell,
    Matching,
    MatchingKind,
    Symmetric,
    SymmetryAxis,
    Adjacent,
    GuardRing,
    WellType,
)
from ..symbolic_layout import SymbolicLayout
from ..backend import DeviceSizing


def build_5t_ota(
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
    *,
    matching_kind: MatchingKind = MatchingKind.RANDOM_VT,
    with_input_pair_symmetric: bool = True,
    with_load_pair_symmetric: bool = True,
    with_input_to_tail_adjacent: bool = True,
    with_guard_ring: bool = False,
    guard_ring_well: WellType = WellType.NWELL,
) -> Tuple[AnnotatedCell, DeviceSizing]:
    """Build a 5-transistor OTA (M_in_p / M_in_n / M_load_p / M_load_n / M_tail).

    Args:
        name: cell name
        input_w_um / input_l_um / input_finger_count: input pair sizing
            (M_in_p and M_in_n share the same per-device sizing)
        load_w_um / load_l_um / load_finger_count: PMOS load pair sizing
            (M_load_p and M_load_n share sizing)
        tail_w_um / tail_l_um / tail_finger_count: tail current source sizing
        matching_kind: kind for both pairs' Matching annotations
        with_input_pair_symmetric: add Symmetric(input pair) annotation
        with_load_pair_symmetric: add Symmetric(load pair) annotation
        with_input_to_tail_adjacent: add Adjacent annotation from each input
            device to M_tail (close-coupled current source layout)
        with_guard_ring: add GuardRing annotation (well type configurable)

    Returns:
        (AnnotatedCell, DeviceSizing) — 5 devices keyed M_in_p / M_in_n /
        M_load_p / M_load_n / M_tail.

    Raises:
        ValueError on non-positive sizing or zero finger counts.
    """
    for fc, fc_name in [
        (input_finger_count, "input_finger_count"),
        (load_finger_count, "load_finger_count"),
        (tail_finger_count, "tail_finger_count"),
    ]:
        if fc < 1:
            raise ValueError(f"{fc_name} must be >= 1, got {fc}")
    for v, vname in [
        (input_w_um, "input_w_um"), (input_l_um, "input_l_um"),
        (load_w_um, "load_w_um"), (load_l_um, "load_l_um"),
        (tail_w_um, "tail_w_um"), (tail_l_um, "tail_l_um"),
    ]:
        if v <= 0:
            raise ValueError(f"{vname} must be > 0, got {v}")

    cell = AnnotatedCell(name)
    cell.set_layout("M_in_p", SymbolicLayout(finger_count=input_finger_count))
    cell.set_layout("M_in_n", SymbolicLayout(finger_count=input_finger_count))
    cell.set_layout("M_load_p", SymbolicLayout(finger_count=load_finger_count))
    cell.set_layout("M_load_n", SymbolicLayout(finger_count=load_finger_count))
    cell.set_layout("M_tail", SymbolicLayout(finger_count=tail_finger_count))

    # Input pair bidirectional Matching
    cell.add_annotation("M_in_p", Matching(other="M_in_n", kind=matching_kind))
    cell.add_annotation("M_in_n", Matching(other="M_in_p", kind=matching_kind))

    # Load pair bidirectional Matching
    cell.add_annotation("M_load_p", Matching(other="M_load_n", kind=matching_kind))
    cell.add_annotation("M_load_n", Matching(other="M_load_p", kind=matching_kind))

    if with_input_pair_symmetric:
        cell.add_annotation(
            "M_in_p",
            Symmetric(devices=["M_in_p", "M_in_n"], axis=SymmetryAxis.VERTICAL),
        )

    if with_load_pair_symmetric:
        cell.add_annotation(
            "M_load_p",
            Symmetric(devices=["M_load_p", "M_load_n"], axis=SymmetryAxis.VERTICAL),
        )

    if with_input_to_tail_adjacent:
        # Both input devices close to tail (the source of the differential pair)
        cell.add_annotation(
            "M_in_p",
            Adjacent(to="M_tail", max_separation_multiplier=Fraction(3, 1)),
        )
        cell.add_annotation(
            "M_in_n",
            Adjacent(to="M_tail", max_separation_multiplier=Fraction(3, 1)),
        )

    if with_guard_ring:
        # Cell-wide guard ring (well type configurable; typically nwell for
        # PMOS-substrate-rich amplifiers)
        cell.add_annotation("M_in_p", GuardRing(well_type=guard_ring_well))

    sizing: DeviceSizing = {
        "M_in_p": (input_w_um, input_l_um),
        "M_in_n": (input_w_um, input_l_um),
        "M_load_p": (load_w_um, load_l_um),
        "M_load_n": (load_w_um, load_l_um),
        "M_tail": (tail_w_um, tail_l_um),
    }
    return cell, sizing
