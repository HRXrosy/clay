"""Frame-specific placement bboxes that cannot be serialized as intent."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_bbox(value: object) -> None:
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(coordinate) is not int for coordinate in value)
        or value[2] <= value[0]
        or value[3] <= value[1]
    ):
        raise TypeError("bbox_nm must be an exact ordered integer bbox")


def _validate_pair(value: object, *, field: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise TypeError(f"{field} must be an exact integer pair")
    return value


@dataclass(frozen=True)
class SourceAbsoluteBBox:
    bbox_nm: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox_nm)


@dataclass(frozen=True)
class SolverMasterLocalBBox:
    bbox_nm: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox_nm)


@dataclass(frozen=True)
class FinalAbsoluteBBox:
    bbox_nm: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox_nm)


def source_absolute_to_solver_local(
    bbox: SourceAbsoluteBBox,
    *,
    source_master_origin_nm: tuple[int, int],
    member_offset_nm: tuple[int, int],
) -> SolverMasterLocalBBox:
    """Project one source-absolute bbox into a composite master local frame."""

    if type(bbox) is not SourceAbsoluteBBox:
        raise TypeError("bbox must be an exact SourceAbsoluteBBox")
    sx, sy = _validate_pair(
        source_master_origin_nm,
        field="source_master_origin_nm",
    )
    dx, dy = _validate_pair(member_offset_nm, field="member_offset_nm")
    x0, y0, x1, y1 = bbox.bbox_nm
    return SolverMasterLocalBBox(
        (
            dx + x0 - sx,
            dy + y0 - sy,
            dx + x1 - sx,
            dy + y1 - sy,
        )
    )


def solver_local_to_final_absolute(
    bbox: SolverMasterLocalBBox,
    *,
    final_master_origin_nm: tuple[int, int],
) -> FinalAbsoluteBBox:
    """Translate one solver-master-local bbox into final absolute placement."""

    if type(bbox) is not SolverMasterLocalBBox:
        raise TypeError("bbox must be an exact SolverMasterLocalBBox")
    mx, my = _validate_pair(
        final_master_origin_nm,
        field="final_master_origin_nm",
    )
    x0, y0, x1, y1 = bbox.bbox_nm
    return FinalAbsoluteBBox((mx + x0, my + y0, mx + x1, my + y1))


__all__ = [
    "FinalAbsoluteBBox",
    "SolverMasterLocalBBox",
    "SourceAbsoluteBBox",
    "solver_local_to_final_absolute",
    "source_absolute_to_solver_local",
]
