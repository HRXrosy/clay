"""Folded cascode amplifier cell template (Phase 6.3).

9-transistor topology: NMOS input pair + NMOS cascode pair + NMOS bias +
PMOS cascode pair + PMOS bias + NMOS tail. Matched pairs across the
amplifier's vertical mirror axis.
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


def build_folded_cascode(
    name: str,
    input_w_um: float,
    input_l_um: float,
    input_finger_count: int = 4,
    n_cascode_w_um: float = 4.0,
    n_cascode_l_um: float = 0.5,
    n_cascode_finger_count: int = 4,
    n_bias_w_um: float = 4.0,
    n_bias_l_um: float = 1.0,
    n_bias_finger_count: int = 4,
    p_cascode_w_um: float = 8.0,
    p_cascode_l_um: float = 0.5,
    p_cascode_finger_count: int = 4,
    p_bias_w_um: float = 8.0,
    p_bias_l_um: float = 1.0,
    p_bias_finger_count: int = 4,
    tail_w_um: float = 8.0,
    tail_l_um: float = 1.0,
    tail_finger_count: int = 4,
    *,
    matching_kind: MatchingKind = MatchingKind.RANDOM_VT,
    with_input_pair_symmetric: bool = True,
    with_n_cascode_symmetric: bool = True,
    with_p_cascode_symmetric: bool = True,
    with_pair_adjacent: bool = True,
    with_guard_ring: bool = False,
    guard_ring_well: WellType = WellType.NWELL,
) -> Tuple[AnnotatedCell, DeviceSizing]:
    """Build a 9-MOS folded cascode amplifier.

    Args:
        name: cell name
        input_w_um / input_l_um / input_finger_count: NMOS input pair sizing
            (M_in_p / M_in_n)
        n_cascode_w_um / n_cascode_l_um / n_cascode_finger_count: NMOS cascode
            pair sizing (M_n_cas_l / M_n_cas_r)
        n_bias_w_um / n_bias_l_um / n_bias_finger_count: NMOS bias mirror
            sizing (M_n_bias)
        p_cascode_w_um / p_cascode_l_um / p_cascode_finger_count: PMOS cascode
            pair sizing (M_p_cas_l / M_p_cas_r)
        p_bias_w_um / p_bias_l_um / p_bias_finger_count: PMOS bias mirror
            sizing (M_p_bias)
        tail_w_um / tail_l_um / tail_finger_count: NMOS tail current source
            sizing (M_tail)
        matching_kind: kind for all three pairs' Matching annotations
        with_input_pair_symmetric: add Symmetric on input pair
        with_n_cascode_symmetric: add Symmetric on n-cascode pair
        with_p_cascode_symmetric: add Symmetric on p-cascode pair
        with_pair_adjacent: add Adjacent within each of the three pairs
        with_guard_ring: add GuardRing annotation on cell

    Returns:
        (AnnotatedCell, DeviceSizing) — 9 devices keyed M_in_p / M_in_n /
        M_n_cas_l / M_n_cas_r / M_n_bias / M_p_cas_l / M_p_cas_r / M_p_bias /
        M_tail.

    Raises:
        ValueError on non-positive sizing or zero finger counts.
    """
    finger_args = [
        (input_finger_count, "input_finger_count"),
        (n_cascode_finger_count, "n_cascode_finger_count"),
        (n_bias_finger_count, "n_bias_finger_count"),
        (p_cascode_finger_count, "p_cascode_finger_count"),
        (p_bias_finger_count, "p_bias_finger_count"),
        (tail_finger_count, "tail_finger_count"),
    ]
    for v, vn in finger_args:
        if v < 1:
            raise ValueError(f"{vn} must be >= 1, got {v}")
    size_args = [
        (input_w_um, "input_w_um"), (input_l_um, "input_l_um"),
        (n_cascode_w_um, "n_cascode_w_um"), (n_cascode_l_um, "n_cascode_l_um"),
        (n_bias_w_um, "n_bias_w_um"), (n_bias_l_um, "n_bias_l_um"),
        (p_cascode_w_um, "p_cascode_w_um"), (p_cascode_l_um, "p_cascode_l_um"),
        (p_bias_w_um, "p_bias_w_um"), (p_bias_l_um, "p_bias_l_um"),
        (tail_w_um, "tail_w_um"), (tail_l_um, "tail_l_um"),
    ]
    for v, vn in size_args:
        if v <= 0:
            raise ValueError(f"{vn} must be > 0, got {v}")

    cell = AnnotatedCell(name)
    cell.set_layout("M_in_p", SymbolicLayout(finger_count=input_finger_count))
    cell.set_layout("M_in_n", SymbolicLayout(finger_count=input_finger_count))
    cell.set_layout("M_n_cas_l", SymbolicLayout(finger_count=n_cascode_finger_count))
    cell.set_layout("M_n_cas_r", SymbolicLayout(finger_count=n_cascode_finger_count))
    cell.set_layout("M_n_bias", SymbolicLayout(finger_count=n_bias_finger_count))
    cell.set_layout("M_p_cas_l", SymbolicLayout(finger_count=p_cascode_finger_count))
    cell.set_layout("M_p_cas_r", SymbolicLayout(finger_count=p_cascode_finger_count))
    cell.set_layout("M_p_bias", SymbolicLayout(finger_count=p_bias_finger_count))
    cell.set_layout("M_tail", SymbolicLayout(finger_count=tail_finger_count))

    # Matching annotations: 3 pairs (input, n_cascode, p_cascode)
    cell.add_annotation("M_in_p", Matching(other="M_in_n", kind=matching_kind))
    cell.add_annotation("M_in_n", Matching(other="M_in_p", kind=matching_kind))
    cell.add_annotation("M_n_cas_l", Matching(other="M_n_cas_r", kind=matching_kind))
    cell.add_annotation("M_n_cas_r", Matching(other="M_n_cas_l", kind=matching_kind))
    cell.add_annotation("M_p_cas_l", Matching(other="M_p_cas_r", kind=matching_kind))
    cell.add_annotation("M_p_cas_r", Matching(other="M_p_cas_l", kind=matching_kind))

    if with_input_pair_symmetric:
        cell.add_annotation(
            "M_in_p",
            Symmetric(devices=["M_in_p", "M_in_n"], axis=SymmetryAxis.VERTICAL),
        )
    if with_n_cascode_symmetric:
        cell.add_annotation(
            "M_n_cas_l",
            Symmetric(devices=["M_n_cas_l", "M_n_cas_r"], axis=SymmetryAxis.VERTICAL),
        )
    if with_p_cascode_symmetric:
        cell.add_annotation(
            "M_p_cas_l",
            Symmetric(devices=["M_p_cas_l", "M_p_cas_r"], axis=SymmetryAxis.VERTICAL),
        )

    if with_pair_adjacent:
        cell.add_annotation(
            "M_in_p",
            Adjacent(to="M_in_n", max_separation_multiplier=Fraction(2, 1)),
        )
        cell.add_annotation(
            "M_n_cas_l",
            Adjacent(to="M_n_cas_r", max_separation_multiplier=Fraction(2, 1)),
        )
        cell.add_annotation(
            "M_p_cas_l",
            Adjacent(to="M_p_cas_r", max_separation_multiplier=Fraction(2, 1)),
        )

    if with_guard_ring:
        cell.add_annotation("M_in_p", GuardRing(well_type=guard_ring_well))

    sizing: DeviceSizing = {
        "M_in_p": (input_w_um, input_l_um),
        "M_in_n": (input_w_um, input_l_um),
        "M_n_cas_l": (n_cascode_w_um, n_cascode_l_um),
        "M_n_cas_r": (n_cascode_w_um, n_cascode_l_um),
        "M_n_bias": (n_bias_w_um, n_bias_l_um),
        "M_p_cas_l": (p_cascode_w_um, p_cascode_l_um),
        "M_p_cas_r": (p_cascode_w_um, p_cascode_l_um),
        "M_p_bias": (p_bias_w_um, p_bias_l_um),
        "M_tail": (tail_w_um, tail_l_um),
    }
    return cell, sizing
