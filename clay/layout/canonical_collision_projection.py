"""Canonical refinement scope and geometry replay for collision consumers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence


_Bbox = tuple[int, int, int, int]
_Layer = tuple[int, int] | None
_Atom = tuple[_Layer, _Bbox]


def _bbox(value: object, *, field: str) -> _Bbox:
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(item) is not int for item in value)
        or value[2] <= value[0]
        or value[3] <= value[1]
    ):
        raise ValueError(f"{field} must be one positive exact integer bbox")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class CanonicalCollisionProjectionScope:
    """One refinement decision projected to master and source-member grains."""

    master_owners: tuple[str, ...]
    source_members: tuple[str, ...]
    master_members: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if (
            type(self.master_owners) is not tuple
            or self.master_owners != tuple(sorted(set(self.master_owners)))
            or any(type(owner) is not str or not owner for owner in self.master_owners)
        ):
            raise ValueError("canonical collision master owners are malformed")
        if (
            type(self.master_members) is not tuple
            or self.master_members != tuple(sorted(self.master_members))
            or len({owner for owner, _members in self.master_members})
            != len(self.master_members)
        ):
            raise ValueError("canonical collision ownership is malformed")
        ownership = dict(self.master_members)
        if any(
            type(owner) is not str
            or not owner
            or type(members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
            or any(type(member) is not str or not member for member in members)
            for owner, members in self.master_members
        ):
            raise ValueError("canonical collision member inventory is malformed")
        if not set(self.master_owners) <= set(ownership):
            raise ValueError("canonical collision refinement names an unknown master")
        expected_source = tuple(
            sorted(
                {
                    member
                    for owner in self.master_owners
                    for member in ownership[owner]
                }
            )
        )
        if self.source_members != expected_source:
            raise ValueError("canonical collision source projection differs")

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "master_owners": list(self.master_owners),
            "source_members": list(self.source_members),
            "master_members": [
                [owner, list(members)] for owner, members in self.master_members
            ],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.canonical_data())).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalRowBoundedCavityProjection:
    """One generated-geometry cavity consumed identically by placement stages."""

    obligation_id: str
    source_obligation_sha256: str
    host_master: str
    guest_masters: tuple[str, ...]
    lower_boundary_offset_nm: int
    upper_boundary_offset_nm: int
    lower_boundary: str = "maximal_span_host_atom_row_top_envelope"
    upper_boundary: str = "host_envelope_top"

    def __post_init__(self) -> None:
        for value, label in (
            (self.obligation_id, "cavity obligation id"),
            (self.host_master, "cavity host master"),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{label} is malformed")
        if (
            type(self.source_obligation_sha256) is not str
            or len(self.source_obligation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_obligation_sha256)
        ):
            raise ValueError("cavity source obligation SHA-256 is malformed")
        if (
            type(self.guest_masters) is not tuple
            or not self.guest_masters
            or self.guest_masters != tuple(sorted(set(self.guest_masters)))
            or any(
                type(guest) is not str or not guest or guest == self.host_master
                for guest in self.guest_masters
            )
        ):
            raise ValueError("cavity guest masters are malformed")
        if (
            type(self.lower_boundary_offset_nm) is not int
            or type(self.upper_boundary_offset_nm) is not int
            or not 0 < self.lower_boundary_offset_nm < self.upper_boundary_offset_nm
        ):
            raise ValueError("cavity boundary offsets are malformed")
        if (
            self.lower_boundary != "maximal_span_host_atom_row_top_envelope"
            or self.upper_boundary != "host_envelope_top"
        ):
            raise ValueError("cavity intrinsic boundary selectors differ")

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": "clay.canonical-row-bounded-cavity-projection/v1",
            "obligation_id": self.obligation_id,
            "source_obligation_sha256": self.source_obligation_sha256,
            "host_master": self.host_master,
            "guest_masters": list(self.guest_masters),
            "lower_boundary": self.lower_boundary,
            "upper_boundary": self.upper_boundary,
            "lower_boundary_offset_nm": self.lower_boundary_offset_nm,
            "upper_boundary_offset_nm": self.upper_boundary_offset_nm,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.canonical_data())).hexdigest()


@dataclass(frozen=True, slots=True)
class ComponentCover:
    rectangle_count: int
    envelope_nm: _Bbox

    def __post_init__(self) -> None:
        if type(self.rectangle_count) is not int or self.rectangle_count <= 0:
            raise ValueError("component cover count must be positive")
        _bbox(self.envelope_nm, field="component cover envelope")


@dataclass(frozen=True, slots=True)
class PositiveBboxOverlap:
    intersection_nm: _Bbox
    area_nm2: int

    def __post_init__(self) -> None:
        intersection = _bbox(self.intersection_nm, field="positive bbox intersection")
        expected = (intersection[2] - intersection[0]) * (
            intersection[3] - intersection[1]
        )
        if type(self.area_nm2) is not int or self.area_nm2 != expected:
            raise ValueError("positive bbox overlap area does not replay")


def build_canonical_collision_projection_scope(
    *,
    master_members: Mapping[str, Sequence[str]],
    component_refinement_owners: tuple[str, ...],
) -> CanonicalCollisionProjectionScope:
    """Freeze one master refinement inventory and its exact member projection."""

    if not isinstance(master_members, Mapping):
        raise TypeError("canonical collision ownership must be a mapping")
    rows = []
    for owner, raw_members in master_members.items():
        if type(owner) is not str or not owner or isinstance(raw_members, str):
            raise TypeError("canonical collision ownership row is malformed")
        try:
            members = tuple(raw_members)
        except TypeError as exc:
            raise TypeError("canonical collision members must be finite") from exc
        canonical_members = tuple(sorted(set(members)))
        if (
            not members
            or len(members) != len(canonical_members)
            or any(type(member) is not str or not member for member in members)
        ):
            raise ValueError("canonical collision members are not canonical")
        rows.append((owner, canonical_members))
    ownership = tuple(sorted(rows))
    if (
        type(component_refinement_owners) is not tuple
        or component_refinement_owners
        != tuple(sorted(set(component_refinement_owners)))
    ):
        raise ValueError("component refinement owners are not canonical")
    by_owner = dict(ownership)
    if not set(component_refinement_owners) <= set(by_owner):
        raise ValueError("component refinement names an unknown master")
    return CanonicalCollisionProjectionScope(
        master_owners=component_refinement_owners,
        source_members=tuple(
            sorted(
                {
                    member
                    for owner in component_refinement_owners
                    for member in by_owner[owner]
                }
            )
        ),
        master_members=ownership,
    )


def build_row_bounded_cavity_projection(
    *,
    obligation_id: str,
    source_obligation_sha256: str,
    host_master: str,
    guest_masters: tuple[str, ...],
    host_size_nm: tuple[int, int],
    host_row_footprints_nm: tuple[_Bbox, ...],
) -> CanonicalRowBoundedCavityProjection:
    """Derive intrinsic cavity bounds from one host's local row footprints."""

    if (
        type(host_size_nm) is not tuple
        or len(host_size_nm) != 2
        or any(type(value) is not int or value <= 0 for value in host_size_nm)
    ):
        raise ValueError("cavity host size must be two positive integer nm values")
    if type(host_row_footprints_nm) is not tuple or not host_row_footprints_nm:
        raise TypeError("cavity host row footprints must be one non-empty tuple")
    width, height = host_size_nm
    footprints = tuple(
        _bbox(value, field="cavity host row footprint")
        for value in host_row_footprints_nm
    )
    if footprints != tuple(sorted(set(footprints))):
        raise ValueError("cavity host row footprints must be canonical and unique")
    if any(
        not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height)
        for x0, y0, x1, y1 in footprints
    ):
        raise ValueError("cavity host row footprint escapes the host envelope")

    parent = list(range(len(footprints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for first, left in enumerate(footprints):
        for second in range(first + 1, len(footprints)):
            right = footprints[second]
            if max(left[1], right[1]) < min(left[3], right[3]):
                union(first, second)
    row_groups: dict[int, list[_Bbox]] = {}
    for index, footprint in enumerate(footprints):
        row_groups.setdefault(find(index), []).append(footprint)
    row_envelopes = tuple(
        (
            min(row[0] for row in rows),
            min(row[1] for row in rows),
            max(row[2] for row in rows),
            max(row[3] for row in rows),
        )
        for _root, rows in sorted(row_groups.items())
    )
    maximum_horizontal_span = max(
        row[2] - row[0] for row in row_envelopes
    )
    maximal_span_tops = tuple(
        row[3]
        for row in row_envelopes
        if row[2] - row[0] == maximum_horizontal_span
    )
    if not maximal_span_tops:
        raise ValueError("cavity host exposes no measurable atom-row envelope")
    lower = max(maximal_span_tops)
    if lower >= height:
        raise ValueError("cavity host has no positive row-bounded free height")
    return CanonicalRowBoundedCavityProjection(
        obligation_id=obligation_id,
        source_obligation_sha256=source_obligation_sha256,
        host_master=host_master,
        guest_masters=guest_masters,
        lower_boundary_offset_nm=lower,
        upper_boundary_offset_nm=height,
    )


def component_cover_envelope(
    atoms: Sequence[_Atom],
    *,
    layer: _Layer,
    containing_bbox_nm: _Bbox,
) -> ComponentCover:
    """Replay the envelope of same-layer components inside one source atom."""

    containing = _bbox(containing_bbox_nm, field="component containing bbox")
    rows = []
    for raw_layer, raw_bbox in atoms:
        if raw_layer is not None and (
            type(raw_layer) is not tuple
            or len(raw_layer) != 2
            or any(type(item) is not int or item < 0 for item in raw_layer)
        ):
            raise ValueError("component atom layer is malformed")
        bbox = _bbox(raw_bbox, field="component atom bbox")
        if raw_layer == layer and (
            containing[0] <= bbox[0] < bbox[2] <= containing[2]
            and containing[1] <= bbox[1] < bbox[3] <= containing[3]
        ):
            rows.append(bbox)
    if not rows:
        raise ValueError("component cover has no same-layer rectangles")
    return ComponentCover(
        rectangle_count=len(rows),
        envelope_nm=(
            min(bbox[0] for bbox in rows),
            min(bbox[1] for bbox in rows),
            max(bbox[2] for bbox in rows),
            max(bbox[3] for bbox in rows),
        ),
    )


def positive_bbox_overlap(left: _Bbox, right: _Bbox) -> PositiveBboxOverlap:
    """Return one strict positive-area Manhattan bbox intersection."""

    left_bbox = _bbox(left, field="left bbox")
    right_bbox = _bbox(right, field="right bbox")
    intersection = (
        max(left_bbox[0], right_bbox[0]),
        max(left_bbox[1], right_bbox[1]),
        min(left_bbox[2], right_bbox[2]),
        min(left_bbox[3], right_bbox[3]),
    )
    if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
        raise ValueError("bboxes do not have one positive-area overlap")
    return PositiveBboxOverlap(
        intersection_nm=intersection,
        area_nm2=(intersection[2] - intersection[0])
        * (intersection[3] - intersection[1]),
    )


__all__ = [
    "CanonicalCollisionProjectionScope",
    "CanonicalRowBoundedCavityProjection",
    "ComponentCover",
    "PositiveBboxOverlap",
    "build_canonical_collision_projection_scope",
    "build_row_bounded_cavity_projection",
    "component_cover_envelope",
    "positive_bbox_overlap",
]
