"""Bandgap reference cell template (Phase 6.6).

Simplified Brokaw-cell bandgap voltage reference:
- Q1, Q2: BJT pair, area ratio (Q2 has N_emitter x Q1 area for PTAT term)
- R1, R2: resistors generating PTAT current + CTAT-trimming voltage
- OTA: 5T amplifier summing PTAT (V_be diff) + CTAT (V_be) -> V_ref ~ 1.25 V

Devices modeled as SymbolicLayout placeholders (Clay layout substrate is
MOS-finger-based; BJT/resistor primitives are Phase 12+). finger_count
proxies emitter ratio (BJT) and segment count (resistor).
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


def build_bandgap_reference(
    name: str,
    bjt_unit_emitter_um2: float = 4.0,
    bjt_unit_l_um: float = 1.0,
    q1_emitter_units: int = 1,
    q2_emitter_units: int = 8,
    resistor_r1_units: int = 4,
    resistor_r2_units: int = 16,
    resistor_unit_w_um: float = 1.0,
    resistor_unit_l_um: float = 5.0,
    ota_input_w_um: float = 4.0,
    ota_input_l_um: float = 1.0,
    ota_input_finger_count: int = 4,
    ota_load_w_um: float = 4.0,
    ota_load_l_um: float = 1.0,
    ota_load_finger_count: int = 4,
    ota_tail_w_um: float = 8.0,
    ota_tail_l_um: float = 1.0,
    ota_tail_finger_count: int = 4,
    *,
    matching_kind: MatchingKind = MatchingKind.RANDOM_VT,
    with_input_pair_symmetric: bool = True,
    with_load_pair_symmetric: bool = True,
    with_bjt_q1_q2_adjacent: bool = True,
    with_resistor_pair_matched: bool = True,
    with_guard_ring: bool = False,
    guard_ring_well: WellType = WellType.NWELL,
) -> Tuple[AnnotatedCell, DeviceSizing]:
    """Build a 9-slot bandgap reference (2 BJT placeholders + 2 R placeholders +
    5 OTA MOS).

    Args:
        bjt_unit_emitter_um2: area of a single BJT emitter unit
        bjt_unit_l_um: BJT placeholder L proxy
        q1_emitter_units / q2_emitter_units: emitter area ratio (Q2/Q1 typically 8:1)
        resistor_r1_units / resistor_r2_units: poly-resistor segment count
        resistor_unit_w_um / resistor_unit_l_um: per-segment resistor dimensions
        ota_*: 5T-OTA opamp sizing (input pair + load pair + tail)
        matching_kind: matching annotation kind
        with_input_pair_symmetric / with_load_pair_symmetric: OTA-side symmetric
        with_bjt_q1_q2_adjacent: Q1 and Q2 placed adjacent (minimize gradient mismatch)
        with_resistor_pair_matched: R1 and R2 share Matching annotation
        with_guard_ring: cell-perimeter guard ring
    """
    if q1_emitter_units < 1 or q2_emitter_units < 1:
        raise ValueError("emitter_units must be >= 1")
    if resistor_r1_units < 1 or resistor_r2_units < 1:
        raise ValueError("resistor_units must be >= 1")
    for fc, fn in [
        (ota_input_finger_count, "ota_input_finger_count"),
        (ota_load_finger_count, "ota_load_finger_count"),
        (ota_tail_finger_count, "ota_tail_finger_count"),
    ]:
        if fc < 1:
            raise ValueError(f"{fn} must be >= 1")
    for v, vn in [
        (bjt_unit_emitter_um2, "bjt_unit_emitter_um2"),
        (bjt_unit_l_um, "bjt_unit_l_um"),
        (resistor_unit_w_um, "resistor_unit_w_um"),
        (resistor_unit_l_um, "resistor_unit_l_um"),
        (ota_input_w_um, "ota_input_w_um"), (ota_input_l_um, "ota_input_l_um"),
        (ota_load_w_um, "ota_load_w_um"), (ota_load_l_um, "ota_load_l_um"),
        (ota_tail_w_um, "ota_tail_w_um"), (ota_tail_l_um, "ota_tail_l_um"),
    ]:
        if v <= 0:
            raise ValueError(f"{vn} must be > 0")

    cell = AnnotatedCell(name)

    # BJT placeholders (finger_count = emitter unit count for PTAT generation)
    cell.set_layout("Q1", SymbolicLayout(finger_count=q1_emitter_units))
    cell.set_layout("Q2", SymbolicLayout(finger_count=q2_emitter_units))

    # Resistor placeholders (finger_count = segment count)
    cell.set_layout("R1", SymbolicLayout(finger_count=resistor_r1_units))
    cell.set_layout("R2", SymbolicLayout(finger_count=resistor_r2_units))

    # 5T-OTA opamp
    cell.set_layout("M_in_p", SymbolicLayout(finger_count=ota_input_finger_count))
    cell.set_layout("M_in_n", SymbolicLayout(finger_count=ota_input_finger_count))
    cell.set_layout("M_load_p", SymbolicLayout(finger_count=ota_load_finger_count))
    cell.set_layout("M_load_n", SymbolicLayout(finger_count=ota_load_finger_count))
    cell.set_layout("M_tail", SymbolicLayout(finger_count=ota_tail_finger_count))

    # OTA Matching (input pair + load pair, bidirectional)
    cell.add_annotation("M_in_p", Matching(other="M_in_n", kind=matching_kind))
    cell.add_annotation("M_in_n", Matching(other="M_in_p", kind=matching_kind))
    cell.add_annotation("M_load_p", Matching(other="M_load_n", kind=matching_kind))
    cell.add_annotation("M_load_n", Matching(other="M_load_p", kind=matching_kind))

    # OTA Symmetric (optional)
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

    # BJT Q1/Q2 adjacency (minimize gradient mismatch in V_be PTAT generation)
    if with_bjt_q1_q2_adjacent:
        cell.add_annotation(
            "Q1", Adjacent(to="Q2", max_separation_multiplier=Fraction(2, 1)),
        )
        cell.add_annotation(
            "Q2", Adjacent(to="Q1", max_separation_multiplier=Fraction(2, 1)),
        )

    # Resistor pair matched (R1, R2 share temperature coefficient via matching)
    if with_resistor_pair_matched:
        cell.add_annotation("R1", Matching(other="R2", kind=MatchingKind.BOTH))
        cell.add_annotation("R2", Matching(other="R1", kind=MatchingKind.BOTH))

    if with_guard_ring:
        cell.add_annotation("M_in_p", GuardRing(well_type=guard_ring_well))

    sizing: DeviceSizing = {
        # BJT: report (emitter unit area side, L) via stub
        "Q1": (bjt_unit_emitter_um2, bjt_unit_l_um),
        "Q2": (bjt_unit_emitter_um2, bjt_unit_l_um),
        # Resistor: (per-segment W, per-segment L)
        "R1": (resistor_unit_w_um, resistor_unit_l_um),
        "R2": (resistor_unit_w_um, resistor_unit_l_um),
        # OTA MOS
        "M_in_p": (ota_input_w_um, ota_input_l_um),
        "M_in_n": (ota_input_w_um, ota_input_l_um),
        "M_load_p": (ota_load_w_um, ota_load_l_um),
        "M_load_n": (ota_load_w_um, ota_load_l_um),
        "M_tail": (ota_tail_w_um, ota_tail_l_um),
    }
    return cell, sizing
