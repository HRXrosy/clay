"""Compile shared-well local owners into replayable hierarchical child plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from clay.layout.intent import GuardRingIntent
from clay.layout.slicing_floorplan import (
    OptimalSlicingFloorplan,
    solve_optimal_slicing_floorplan,
)
from clay.layout.well_island_sharing import (
    SharedWellGuardEnvelopePlan,
    WellIslandSharingError,
    replay_shared_well_guard_envelopes,
)


_HEX = frozenset("0123456789abcdef")
_MAX_COORDINATE_NM = 10**12


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in _HEX for char in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _coordinate(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or abs(value) > _MAX_COORDINATE_NM:
        raise TypeError(f"{field} must be one bounded exact integer")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _bbox(value: object, field: str) -> tuple[int, int, int, int]:
    if type(value) is not tuple or len(value) != 4:
        raise TypeError(f"{field} must be one exact bbox tuple")
    result = tuple(_coordinate(item, field) for item in value)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{field} must have positive area")
    return result  # type: ignore[return-value]


def _canonical_member_rows(
    value: Any,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if type(value) is not tuple:
        raise TypeError("shared guard child ownership must be one exact tuple")
    rebuilt: list[tuple[str, tuple[str, ...]]] = []
    for row in value:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("shared guard child ownership rows must be exact tuples")
        block_id, members = row
        if type(block_id) is not str or not block_id:
            raise TypeError("shared guard child block id must be one string")
        if (
            type(members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
            or any(type(member) is not str or not member for member in members)
        ):
            raise ValueError("shared guard child members must be exact and canonical")
        rebuilt.append((block_id, members))
    canonical = tuple(sorted(rebuilt))
    if tuple(rebuilt) != canonical or len({row[0] for row in rebuilt}) != len(rebuilt):
        raise ValueError("shared guard child ownership must be unique and canonical")
    return canonical


def _canonical_bbox_rows(
    value: Any,
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    if type(value) is not tuple:
        raise TypeError("shared guard child bbox inventory must be one exact tuple")
    rebuilt: list[tuple[str, tuple[int, int, int, int]]] = []
    for row in value:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("shared guard child bbox rows must be exact tuples")
        block_id, raw_bbox = row
        if type(block_id) is not str or not block_id:
            raise TypeError("shared guard child bbox id must be one string")
        rebuilt.append((block_id, _bbox(raw_bbox, "shared guard child source bbox")))
    canonical = tuple(sorted(rebuilt))
    if tuple(rebuilt) != canonical or len({row[0] for row in rebuilt}) != len(rebuilt):
        raise ValueError("shared guard child bboxes must be unique and canonical")
    return canonical


def _canonical_extensions(value: Any) -> tuple[tuple[str, int], ...]:
    if type(value) is not tuple:
        raise TypeError("shared guard child ring extensions must be one exact tuple")
    rebuilt: list[tuple[str, int]] = []
    for row in value:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("shared guard child extension rows must be exact tuples")
        well, extension_nm = row
        if well not in ("nwell", "pwell"):
            raise ValueError("shared guard child extension has an invalid well")
        rebuilt.append(
            (
                well,
                _coordinate(
                    extension_nm,
                    "shared guard child ring extension",
                    positive=True,
                ),
            )
        )
    canonical = tuple(sorted(rebuilt))
    if tuple(rebuilt) != canonical or len({row[0] for row in rebuilt}) != len(rebuilt):
        raise ValueError("shared guard child extensions must be unique and canonical")
    return canonical


@dataclass(frozen=True, slots=True, order=True)
class SharedWellGuardChildMaster:
    block_id: str
    members: tuple[str, ...]
    source_ring_sha256: str
    source_bbox_nm: tuple[int, int, int, int]
    local_bbox_nm: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if type(self.block_id) is not str or not self.block_id:
            raise TypeError("shared guard child master id must be one string")
        if (
            type(self.members) is not tuple
            or not self.members
            or self.members != tuple(sorted(set(self.members)))
        ):
            raise ValueError("shared guard child master members must be canonical")
        _digest(self.source_ring_sha256, "shared guard child source ring SHA")
        source = _bbox(self.source_bbox_nm, "shared guard child source bbox")
        local = _bbox(self.local_bbox_nm, "shared guard child local bbox")
        if (source[2] - source[0], source[3] - source[1]) != (
            local[2] - local[0],
            local[3] - local[1],
        ):
            raise ValueError("shared guard child master changed its rigid dimensions")

    def canonical_data(self) -> tuple[object, ...]:
        return (
            self.block_id,
            self.members,
            self.source_ring_sha256,
            self.source_bbox_nm,
            self.local_bbox_nm,
        )


@dataclass(frozen=True, slots=True)
class SharedWellGuardChildPlan:
    group: str
    envelope_sha256: str
    well: str
    body: str
    ring_extension_nm: int
    masters: tuple[SharedWellGuardChildMaster, ...]
    floorplan: OptimalSlicingFloorplan
    outer_bbox_nm: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if type(self.group) is not str or not self.group:
            raise TypeError("shared guard child group must be one string")
        _digest(self.envelope_sha256, "shared guard child envelope SHA")
        if self.well not in ("nwell", "pwell"):
            raise ValueError("shared guard child has an invalid well")
        if type(self.body) is not str or not self.body:
            raise TypeError("shared guard child body must be one string")
        extension = _coordinate(
            self.ring_extension_nm,
            "shared guard child ring extension",
            positive=True,
        )
        if type(self.masters) is not tuple or any(
            type(master) is not SharedWellGuardChildMaster for master in self.masters
        ):
            raise TypeError("shared guard child masters must be one exact typed tuple")
        if self.masters != tuple(sorted(self.masters)) or len(self.masters) < 2:
            raise ValueError("shared guard child masters must be canonical and plural")
        if len({master.block_id for master in self.masters}) != len(self.masters):
            raise ValueError("shared guard child master ids must be unique")
        if type(self.floorplan) is not OptimalSlicingFloorplan:
            raise TypeError("shared guard child requires one exact slicing floorplan")
        floorplan_dims = dict(self.floorplan.blocks)
        floorplan_positions = dict(self.floorplan.positions)
        for master in self.masters:
            source = master.source_bbox_nm
            if floorplan_dims.get(master.block_id) != (
                source[2] - source[0],
                source[3] - source[1],
            ):
                raise ValueError("shared guard child floorplan differs from its masters")
            position = floorplan_positions.get(master.block_id)
            expected_local = (
                position[0] + extension,
                position[1] + extension,
                position[0] + extension + floorplan_dims[master.block_id][0],
                position[1] + extension + floorplan_dims[master.block_id][1],
            )
            if master.local_bbox_nm != expected_local:
                raise ValueError("shared guard child local bbox differs from floorplan")
        outer = _bbox(self.outer_bbox_nm, "shared guard child outer bbox")
        expected_outer = (
            0,
            0,
            self.floorplan.bbox_nm[2] + 2 * extension,
            self.floorplan.bbox_nm[3] + 2 * extension,
        )
        if outer != expected_outer:
            raise ValueError("shared guard child outer bbox differs from PDK extension")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.shared-well-guard-child-plan/v1",
            "group": self.group,
            "envelope_sha256": self.envelope_sha256,
            "well": self.well,
            "body": self.body,
            "ring_extension_nm": self.ring_extension_nm,
            "masters": tuple(master.canonical_data() for master in self.masters),
            "floorplan_sha256": self.floorplan.sha256,
            "outer_bbox_nm": self.outer_bbox_nm,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())


def compile_shared_well_guard_child_plans(
    envelope_plans: Any,
    guard_ring_intents: Any,
    block_member_devices: Any,
    content_bboxes_nm: Any,
    *,
    gap_nm: Any,
    ring_extension_nm_by_well: Any,
) -> tuple[SharedWellGuardChildPlan, ...]:
    """Compile exact local owners into optimal child-local slicing floorplans."""

    envelopes = replay_shared_well_guard_envelopes(
        envelope_plans,
        guard_ring_intents,
    )
    ownership = dict(_canonical_member_rows(block_member_devices))
    content_bboxes = dict(_canonical_bbox_rows(content_bboxes_nm))
    extensions = dict(_canonical_extensions(ring_extension_nm_by_well))
    gap = _coordinate(gap_nm, "shared guard child gap")
    if gap < 0:
        raise ValueError("shared guard child gap must be non-negative")
    declarations = tuple(guard_ring_intents)
    plans: list[SharedWellGuardChildPlan] = []
    consumed_blocks: set[str] = set()
    for envelope in envelopes:
        if envelope.well not in extensions:
            raise WellIslandSharingError(
                f"shared well {envelope.group!r} lacks installed ring extension"
            )
        rows: list[tuple[str, tuple[str, ...], str, tuple[int, int, int, int]]] = []
        for members, declaration_sha256 in zip(
            envelope.local_ring_members,
            envelope.local_ring_sha256s,
        ):
            indices = tuple(
                index
                for index, declaration in enumerate(declarations)
                if declaration.sha256 == declaration_sha256
            )
            if len(indices) != 1:
                raise WellIslandSharingError(
                    f"shared well {envelope.group!r} lacks an exact local owner partition"
                )
            block_id = f"ISL{indices[0]}"
            if (
                block_id in consumed_blocks
                or ownership.get(block_id) != members
                or block_id not in content_bboxes
            ):
                raise WellIslandSharingError(
                    f"shared well {envelope.group!r} lacks an exact local owner partition"
                )
            rows.append(
                (block_id, members, declaration_sha256, content_bboxes[block_id])
            )
            consumed_blocks.add(block_id)
        rows.sort(key=lambda row: row[0])
        slicing_blocks = tuple(
            (
                block_id,
                (bbox[2] - bbox[0], bbox[3] - bbox[1]),
            )
            for block_id, _members, _sha, bbox in rows
        )
        floorplan = solve_optimal_slicing_floorplan(slicing_blocks, gap_nm=gap)
        positions = dict(floorplan.positions)
        extension = extensions[envelope.well]
        masters = tuple(
            SharedWellGuardChildMaster(
                block_id=block_id,
                members=members,
                source_ring_sha256=declaration_sha256,
                source_bbox_nm=bbox,
                local_bbox_nm=(
                    positions[block_id][0] + extension,
                    positions[block_id][1] + extension,
                    positions[block_id][0] + extension + bbox[2] - bbox[0],
                    positions[block_id][1] + extension + bbox[3] - bbox[1],
                ),
            )
            for block_id, members, declaration_sha256, bbox in rows
        )
        plans.append(
            SharedWellGuardChildPlan(
                group=envelope.group,
                envelope_sha256=envelope.sha256,
                well=envelope.well,
                body=envelope.body,
                ring_extension_nm=extension,
                masters=masters,
                floorplan=floorplan,
                outer_bbox_nm=(
                    0,
                    0,
                    floorplan.bbox_nm[2] + 2 * extension,
                    floorplan.bbox_nm[3] + 2 * extension,
                ),
            )
        )
    return tuple(sorted(plans, key=lambda plan: plan.group))


def replay_shared_well_guard_child_plans(
    value: Any,
    envelope_plans: Any,
    guard_ring_intents: Any,
    block_member_devices: Any,
    content_bboxes_nm: Any,
    *,
    gap_nm: Any,
    ring_extension_nm_by_well: Any,
) -> tuple[SharedWellGuardChildPlan, ...]:
    """Recompile every child plan and require byte-identical source ownership."""

    if type(value) is not tuple or any(
        type(plan) is not SharedWellGuardChildPlan for plan in value
    ):
        raise TypeError("shared guard child replay requires one exact plan tuple")
    replayed = compile_shared_well_guard_child_plans(
        envelope_plans,
        guard_ring_intents,
        block_member_devices,
        content_bboxes_nm,
        gap_nm=gap_nm,
        ring_extension_nm_by_well=ring_extension_nm_by_well,
    )
    if replayed != value:
        raise ValueError("shared guard child plan does not replay")
    return replayed


def apply_shared_well_guard_child_plans(
    plans: Any,
    block_stamps: Any,
    current_content_bboxes_nm: Any,
    placement_um: Any,
) -> dict[str, tuple[float, float]]:
    """Rigidly project each proven local owner into its child-local floorplan."""

    if type(plans) is not tuple or any(
        type(plan) is not SharedWellGuardChildPlan for plan in plans
    ):
        raise TypeError("shared guard child projection requires one exact plan tuple")
    if plans != tuple(sorted(plans, key=lambda plan: plan.group)):
        raise ValueError("shared guard child plans must use canonical group order")
    if type(block_stamps) is not tuple:
        raise TypeError("shared guard child stamp ownership must be one exact tuple")
    stamp_rows: list[tuple[str, tuple[str, ...]]] = []
    all_stamps: list[str] = []
    for row in block_stamps:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("shared guard child stamp rows must be exact tuples")
        block_id, stamps = row
        if type(block_id) is not str or not block_id:
            raise TypeError("shared guard child stamp owner must be one string")
        if (
            type(stamps) is not tuple
            or not stamps
            or stamps != tuple(sorted(set(stamps)))
            or any(type(stamp) is not str or not stamp for stamp in stamps)
        ):
            raise ValueError("shared guard child stamps must be exact and canonical")
        stamp_rows.append((block_id, stamps))
        all_stamps.extend(stamps)
    if tuple(stamp_rows) != tuple(sorted(stamp_rows)) or len(
        {row[0] for row in stamp_rows}
    ) != len(stamp_rows):
        raise ValueError("shared guard child stamp ownership must be canonical")
    if len(all_stamps) != len(set(all_stamps)):
        raise ValueError("shared guard child stamps may have only one owner")
    stamps_by_block = dict(stamp_rows)
    bboxes = dict(_canonical_bbox_rows(current_content_bboxes_nm))
    if not isinstance(placement_um, Mapping):
        raise TypeError("shared guard child placement must be one mapping")
    projected: dict[str, tuple[float, float]] = {}
    for raw_name, raw_point in placement_um.items():
        if type(raw_name) is not str or not raw_name:
            raise TypeError("shared guard child placement name must be one string")
        if type(raw_point) is not tuple or len(raw_point) != 2:
            raise TypeError("shared guard child placement point must be one exact tuple")
        if any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in raw_point
        ):
            raise ValueError("shared guard child placement point must be finite")
        projected[raw_name] = (float(raw_point[0]), float(raw_point[1]))

    consumed_blocks: set[str] = set()
    consumed_stamps: set[str] = set()
    for plan in plans:
        anchor_x = (
            min(master.source_bbox_nm[0] for master in plan.masters)
            - plan.ring_extension_nm
        )
        anchor_y = (
            min(master.source_bbox_nm[1] for master in plan.masters)
            - plan.ring_extension_nm
        )
        for master in plan.masters:
            if (
                master.block_id in consumed_blocks
                or bboxes.get(master.block_id) != master.source_bbox_nm
            ):
                raise ValueError(
                    f"shared guard child source bbox changed for {master.block_id!r}"
                )
            stamps = stamps_by_block.get(master.block_id)
            if not stamps or consumed_stamps.intersection(stamps):
                raise ValueError(
                    f"shared guard child stamp ownership changed for {master.block_id!r}"
                )
            if any(stamp not in projected for stamp in stamps):
                raise ValueError(
                    f"shared guard child placement lacks {master.block_id!r} stamps"
                )
            dx_nm = (
                anchor_x + master.local_bbox_nm[0] - master.source_bbox_nm[0]
            )
            dy_nm = (
                anchor_y + master.local_bbox_nm[1] - master.source_bbox_nm[1]
            )
            for stamp in stamps:
                x_um, y_um = projected[stamp]
                projected[stamp] = (
                    x_um + dx_nm / 1000.0,
                    y_um + dy_nm / 1000.0,
                )
            consumed_blocks.add(master.block_id)
            consumed_stamps.update(stamps)
    return projected


def _shared_well_guard_member_deltas(
    plans: Any,
) -> dict[str, tuple[int, int]]:
    if type(plans) is not tuple or not plans or any(
        type(plan) is not SharedWellGuardChildPlan for plan in plans
    ):
        raise TypeError(
            "shared guard logical projection requires one non-empty exact plan tuple"
        )
    if plans != tuple(sorted(plans, key=lambda plan: plan.group)) or len(
        {plan.group for plan in plans}
    ) != len(plans):
        raise ValueError("shared guard child plans must be canonical and unique")
    deltas: dict[str, tuple[int, int]] = {}
    consumed_blocks: set[str] = set()
    for plan in plans:
        anchor_x = (
            min(master.source_bbox_nm[0] for master in plan.masters)
            - plan.ring_extension_nm
        )
        anchor_y = (
            min(master.source_bbox_nm[1] for master in plan.masters)
            - plan.ring_extension_nm
        )
        for master in plan.masters:
            if master.block_id in consumed_blocks:
                raise ValueError("shared guard child block has multiple owners")
            delta = (
                _coordinate(
                    anchor_x
                    + master.local_bbox_nm[0]
                    - master.source_bbox_nm[0],
                    "shared guard child logical x delta",
                ),
                _coordinate(
                    anchor_y
                    + master.local_bbox_nm[1]
                    - master.source_bbox_nm[1],
                    "shared guard child logical y delta",
                ),
            )
            for member in master.members:
                if member in deltas:
                    raise ValueError(
                        "shared guard child logical member has multiple owners"
                    )
                deltas[member] = delta
            consumed_blocks.add(master.block_id)
    if not deltas:
        raise ValueError("shared guard child logical projection has no members")
    return deltas


def _translated_bbox(
    value: Any,
    delta: tuple[int, int],
    field: str,
) -> tuple[int, int, int, int]:
    source = _bbox(value, field)
    dx_nm, dy_nm = delta
    return (
        source[0] + dx_nm,
        source[1] + dy_nm,
        source[2] + dx_nm,
        source[3] + dy_nm,
    )


def project_shared_well_guard_logical_placement(
    placement: Any,
    plans: Any,
) -> Any:
    """Project the logical device view through the exact child-owner deltas.

    The physical hierarchy moves already-materialized stamps before the global
    block solve.  Final matching-axis replay consumes the generic ``Placement``
    view independently, so this seam transports device bboxes, terminals, row
    physical bboxes, and matching axes by those same rigid deltas.  A matching
    pair split across child owners is unsupported until the child solver owns an
    explicit cross-owner axis constraint; it must not be silently re-signed.
    """

    from clay.layout.envelope_place import _reposition_device
    from clay.layout.generic_placement import (
        Placement,
        _cell_geom,
        attach_physical_macro_projection,
        placement_auxiliary_polys,
        project_stored_row_master,
    )

    if type(placement) is not Placement:
        raise TypeError("shared guard logical projection requires one exact Placement")
    if type(placement.devices) is not dict or not placement.devices:
        raise TypeError("shared guard logical projection requires exact devices")
    deltas = _shared_well_guard_member_deltas(plans)
    missing = tuple(sorted(set(deltas) - set(placement.devices)))
    if missing:
        raise ValueError(
            f"shared guard logical projection lacks owned members: {missing}"
        )

    projected_devices = dict(placement.devices)
    for member, (dx_nm, dy_nm) in sorted(deltas.items()):
        device = placement.devices[member]
        bbox = _bbox(
            getattr(device, "bbox", None),
            f"shared guard logical device {member!r} bbox",
        )
        projected_devices[member] = _reposition_device(
            device,
            (bbox[0] + bbox[2]) // 2 + dx_nm,
            (bbox[1] + bbox[3]) // 2 + dy_nm,
        )

    raw_axes = placement.matching_pair_axes
    if type(raw_axes) is not dict:
        raise TypeError("shared guard logical matching axes must be one exact dict")
    projected_axes: dict[tuple[str, str], tuple[str, float]] = {}
    for raw_pair, raw_axis in raw_axes.items():
        if (
            type(raw_pair) is not tuple
            or len(raw_pair) != 2
            or raw_pair != tuple(sorted(raw_pair))
            or raw_pair[0] == raw_pair[1]
            or any(type(member) is not str or not member for member in raw_pair)
        ):
            raise TypeError("shared guard logical matching pair is not canonical")
        if (
            type(raw_axis) is not tuple
            or len(raw_axis) != 2
            or raw_axis[0] not in {"x", "y"}
            or type(raw_axis[1]) not in {int, float}
            or not math.isfinite(float(raw_axis[1]))
            or not (4.0 * float(raw_axis[1])).is_integer()
        ):
            raise TypeError(
                "shared guard logical matching axis must be a finite quarter-nm tuple"
            )
        if any(member not in placement.devices for member in raw_pair):
            raise ValueError("shared guard logical matching pair lacks a device")
        axis_index = 0 if raw_axis[0] == "x" else 1
        perpendicular_index = 1 - axis_index
        boxes = tuple(
            _bbox(
                placement.devices[member].bbox,
                f"shared guard logical matching member {member!r} bbox",
            )
            for member in raw_pair
        )
        source_coordinate4 = sum(
            bbox[index]
            for bbox in boxes
            for index in (axis_index, axis_index + 2)
        )
        if source_coordinate4 != int(4.0 * float(raw_axis[1])):
            raise ValueError(
                "shared guard logical matching axis does not replay source bboxes"
            )
        if sum(
            boxes[0][index]
            for index in (perpendicular_index, perpendicular_index + 2)
        ) != sum(
            boxes[1][index]
            for index in (perpendicular_index, perpendicular_index + 2)
        ):
            raise ValueError("shared guard logical matching pair is not aligned")
        pair_deltas = tuple(deltas.get(member) for member in raw_pair)
        if (pair_deltas[0] is None) != (pair_deltas[1] is None) or (
            pair_deltas[0] is not None and pair_deltas[0] != pair_deltas[1]
        ):
            raise ValueError("matching pair crosses child rigid owner")
        coordinate = float(raw_axis[1])
        if pair_deltas[0] is not None:
            coordinate += pair_deltas[0][axis_index]
        projected_axes[raw_pair] = (raw_axis[0], coordinate)

    raw_row_free = placement.row_free_physical_bboxes
    if type(raw_row_free) is not dict:
        raise TypeError(
            "shared guard logical row-free physical bboxes must be one exact dict"
        )
    projected_row_free: dict[str, tuple[int, int, int, int]] = {}
    for raw_member, raw_bbox in raw_row_free.items():
        if type(raw_member) is not str or raw_member not in placement.devices:
            raise TypeError(
                "shared guard logical row-free physical member is invalid"
            )
        projected_row_free[raw_member] = _translated_bbox(
            raw_bbox,
            deltas.get(raw_member, (0, 0)),
            f"shared guard logical row-free bbox {raw_member!r}",
        )

    x0 = min(device.bbox[0] for device in projected_devices.values())
    y0 = min(device.bbox[1] for device in projected_devices.values())
    x1 = max(device.bbox[2] for device in projected_devices.values())
    y1 = max(device.bbox[3] for device in projected_devices.values())
    auxiliary_polys = placement_auxiliary_polys(placement)
    projected = replace(
        placement,
        devices=projected_devices,
        axis_x=(x0 + x1) // 2,
        geom=_cell_geom(projected_devices, (x0, y0, x1, y1)),
        polys=[
            *(poly for device in projected_devices.values() for poly in device.polys),
            *auxiliary_polys,
        ],
        row_final_certificate=None,
        row_free_physical_bboxes=projected_row_free,
        row_residuals=(),
        physical_macro_placement_certificate=None,
        rigid_nonmos_placement_certificate=None,
        matching_pair_axes=projected_axes,
        matching_pair_physical_bboxes={},
        note=(
            str(placement.note)
            + " | logical placement projected through shared-well child owners"
        ),
    )
    if placement.row_program is not None or placement.row_materialization is not None:
        projected = project_stored_row_master(projected)
    # This is a child-local preprojection.  Other global masters have not moved
    # yet, so a foreign-intrusion check here would judge a transient overlap as
    # final geometry.  The final route projection replays the same non-MOS macro
    # certificate and performs the global check after the block solve.
    return attach_physical_macro_projection(projected)


__all__ = [
    "SharedWellGuardChildMaster",
    "SharedWellGuardChildPlan",
    "apply_shared_well_guard_child_plans",
    "compile_shared_well_guard_child_plans",
    "project_shared_well_guard_logical_placement",
    "replay_shared_well_guard_child_plans",
]
