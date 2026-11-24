"""Current mirror cell template factory."""
from __future__ import annotations
from typing import Tuple

from ..annotations import (
    AnnotatedCell, Matching, MatchingKind, Adjacent,
)
from ..symbolic_layout import SymbolicLayout
from ..backend import DeviceSizing


def build_current_mirror(
    name: str,
    w_um: float,
    l_um: float,
    finger_count_ref: int = 4,
    finger_count_out: int = 4,
    ratio: int = 1,
    *,
    matching_kind: MatchingKind = MatchingKind.RANDOM_VT,
    with_adjacent: bool = True,
) -> Tuple[AnnotatedCell, DeviceSizing]:
    """Build a 2-MOS current mirror (M_ref, M_out) with matching annotations.

    The mirror has reference transistor M_ref and output transistor M_out.
    Ratio > 1 implies output has more fingers than reference to scale current.

    Args:
        name: cell name
        w_um: per-device width in um
        l_um: per-device length in um (same for both)
        finger_count_ref: M_ref finger count (>= 1)
        finger_count_out: M_out finger count (>= 1). For a 1:N mirror with
            same per-finger sizing, this typically equals ratio * finger_count_ref.
        ratio: nominal mirror ratio (only stored as a hint; actual ratio comes
            from finger_count_out / finger_count_ref relationship; default 1)
        matching_kind: kind of matching annotation between M_ref and M_out.
            Note: when ratios differ, matching finger_count mismatch will be
            flagged by spatial validator — this is intentional reminder to
            either match finger counts or interdigitate manually.
        with_adjacent: if True, add Adjacent(to=other, max_sep_mult=Fraction(2,1))
            annotation requiring M_ref and M_out be placed close together

    Returns:
        (AnnotatedCell, DeviceSizing)
    """
    if finger_count_ref < 1 or finger_count_out < 1:
        raise ValueError("finger counts must be >= 1")
    if w_um <= 0 or l_um <= 0:
        raise ValueError("w_um and l_um must be > 0")
    if ratio < 1:
        raise ValueError("ratio must be >= 1")

    from fractions import Fraction

    cell = AnnotatedCell(name)
    cell.set_layout("M_ref", SymbolicLayout(finger_count=finger_count_ref))
    cell.set_layout("M_out", SymbolicLayout(finger_count=finger_count_out))

    cell.add_annotation("M_ref", Matching(other="M_out", kind=matching_kind))
    cell.add_annotation("M_out", Matching(other="M_ref", kind=matching_kind))

    if with_adjacent:
        cell.add_annotation("M_ref", Adjacent(to="M_out", max_separation_multiplier=Fraction(2, 1)))
        cell.add_annotation("M_out", Adjacent(to="M_ref", max_separation_multiplier=Fraction(2, 1)))

    sizing: DeviceSizing = {"M_ref": (w_um, l_um), "M_out": (w_um, l_um)}
    return cell, sizing
