"""Source occurrence, materialized member, and rigid solver-master projections."""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from clay.layout.compact_blocks import _NM_PER_UM
from clay.layout.intent_facts import IntentFactSet
from clay.layout.place.final_placement_state import MatchingPairAxisProjection
from clay.layout.place.constraints_spatial import (
    _bbox_clearance_nm,
    _canonical_adjacency_scope_offsets_nm,
    _canonical_layered_collision_footprints,
    _canonical_rigid_compound_footprints,
    _collision_atoms_by_owner,
    _exact_matching_bbox,
    _seed_geometry_legal,
)
from clay.layout.place.placement_constants import (
    _Bbox,
    _GLOBAL_FLOORPLAN_COLUMN_SCOPE,
    _LayeredCollisionFootprint,
    _RigidCompoundFootprint,
)
from clay.layout.place.placement_cuts import (
    _exact_cut_input_blocks,
)
from clay.layout.place.placement_diagnostics import (
    _diagnostic_canonical_value,
)


def _canonical_materialized_block_source_members(
    materialized_blocks: Sequence[Tuple[str, _Bbox]],
    block_member_devices: Mapping[str, set],
    canonical_source_names: Tuple[str, ...],
) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Freeze each emitted physical block's canonical source-device ownership."""

    block_names = tuple(name for name, _bbox in _exact_cut_input_blocks(
        materialized_blocks
    ))
    if (
        type(canonical_source_names) is not tuple
        or not canonical_source_names
        or canonical_source_names != tuple(sorted(set(canonical_source_names)))
        or any(type(name) is not str or not name for name in canonical_source_names)
    ):
        raise TypeError("canonical source member inventory must be one exact tuple")
    if not isinstance(block_member_devices, Mapping):
        raise TypeError("materialized source ownership must be a mapping")
    if set(block_member_devices) != set(block_names):
        raise ValueError(
            "materialized source ownership does not cover exact physical blocks"
        )
    source_names = set(canonical_source_names)
    rows = []
    all_members = []
    for block_name in block_names:
        raw_members = block_member_devices[block_name]
        if type(raw_members) is not set or any(
            type(member) is not str or not member for member in raw_members
        ):
            raise TypeError("materialized source ownership member set is malformed")
        members = tuple(sorted(set(raw_members) & source_names))
        if not members:
            raise ValueError(
                f"materialized physical block {block_name!r} has no source owner"
            )
        rows.append((block_name, members))
        all_members.extend(members)
    if len(all_members) != len(set(all_members)):
        raise ValueError("materialized source ownership repeats a source member")
    return tuple(sorted(rows))

def _materialized_adjacency_scope_bboxes_nm(
    *,
    master_blocks: Sequence[Tuple[str, _Bbox]],
    master_members: Mapping[str, Sequence[str]],
    materialized_stamps: Mapping[str, Sequence[str]],
    materialized_rings: Mapping[str, str],
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Tuple[float, float]],
    tech: Any,
    source_members: Tuple[str, ...] = (),
    rigid_nonmos_macro_plans: Any = None,
    projection_evidence_sink: Optional[Callable[[Any], None]] = None,
    projection_collision_inputs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Tuple[str, _Bbox], ...]:
    """Project independently claimable structure inside each rigid master.

    Exact adjacency claims intentionally omit a canonical well layer after the
    emitter merges that shareable layer.  Replaying the same PDK layer identity
    here prevents the collision bbox (which must retain the well) from posing as
    a non-shareable adjacency edge.  Excluding an unmerged well is conservative:
    the exact claimed envelope can only be larger and therefore closer.
    """

    from clay.layout.placed_geometry import placed_point_nm

    if projection_evidence_sink is not None:
        if not callable(projection_evidence_sink):
            raise TypeError("adjacency projection evidence sink must be callable")
        if (not isinstance(projection_collision_inputs, Mapping)
                or set(projection_collision_inputs) != {
                    "materialized_blocks", "component_refinement_owners"}):
            raise ValueError("adjacency projection lacks complete collision inputs")
        consumed_geometry: Optional[Dict[str, Any]] = {}
    else:
        consumed_geometry = None

    masters = dict(master_blocks)
    if set(master_members) != set(masters):
        raise ValueError("adjacency scope master partition is incomplete")
    if (
        type(source_members) is not tuple
        or tuple(sorted(source_members)) != source_members
        or len(source_members) != len(set(source_members))
        or any(type(name) is not str or not name for name in source_members)
    ):
        raise TypeError("adjacency scope source members must be exact and canonical")
    multi_cluster_masters = {
        master
        for master, members in master_members.items()
        if master.startswith("A_nonrow:") and len(tuple(members)) > 1
    }
    if multi_cluster_masters:
        from clay.layout.nonrow_macro_plan import RigidNonmosMacroPlanSet

        if type(rigid_nonmos_macro_plans) is not RigidNonmosMacroPlanSet:
            raise ValueError(
                "multi-cluster adjacency scope lacks canonical rigid macro authority"
            )
        plans = {
            f"A_nonrow:{plan.macro_id}": plan
            for plan in rigid_nonmos_macro_plans.plans
        }
        source_set = set(source_members)
        for master in sorted(multi_cluster_masters):
            plan = plans.get(master)
            owned = {
                member
                for block_id in master_members[master]
                for stamp_name in materialized_stamps.get(block_id, ())
                for member in {
                    str(stamp_name),
                    *tuple(
                        str(name)
                        for name in (
                            getattr(geometry.get(stamp_name), "composite_members", ())
                            or ()
                        )
                    ),
                }
                if member in source_set
            }
            if plan is None or owned != set(plan.members):
                raise ValueError(
                    "multi-cluster adjacency scope differs from canonical rigid macro"
                )
    try:
        raw_well = tech.build_layer("nwell")
    except Exception as exc:  # noqa: BLE001 - optional canonical PDK layer
        if projection_evidence_sink is not None:
            raise ValueError("adjacency projection requires a resolved nwell layer lookup") from exc
        excluded_layers: set[Tuple[int, int]] = set()
    else:
        if projection_evidence_sink is not None and (
            type(raw_well) is not tuple or len(raw_well) != 2
            or any(type(value) is not int or value < 0 for value in raw_well)
        ):
            raise ValueError("adjacency projection nwell layer key is not exact")
        excluded_layers = {(int(raw_well[0]), int(raw_well[1]))}

    block_scopes: Dict[str, _Bbox] = {}
    materialized_ids = {
        member for members in master_members.values() for member in members
    }
    unknown = materialized_ids - set(materialized_stamps)
    if unknown:
        raise ValueError(
            "adjacency scope lacks materialized stamp ownership: "
            f"{tuple(sorted(unknown))}"
        )
    for block_id in sorted(materialized_ids):
        stamp_names = set(str(name) for name in materialized_stamps[block_id])
        ring_name = materialized_rings.get(block_id)
        if ring_name is not None:
            stamp_names.add(str(ring_name))
        points: list[Tuple[int, int]] = []
        for stamp_name in sorted(stamp_names):
            stamp = geometry.get(stamp_name)
            position = placement_um.get(stamp_name)
            if stamp is None or position is None:
                raise ValueError("adjacency scope stamp lacks placed geometry")
            if consumed_geometry is not None:
                composite_members = getattr(stamp, "composite_members", ()) or ()
                if isinstance(composite_members, str):
                    raise TypeError("adjacency projection composite members must be a sequence")
                consumed_geometry[stamp_name] = {
                    "polys_by_layer": getattr(stamp, "polys_by_layer", {}) or {},
                    "placement_um": position,
                    "composite_members": tuple(composite_members),
                }
            for raw_layer, rings in (
                getattr(stamp, "polys_by_layer", {}) or {}
            ).items():
                try:
                    layer = (int(raw_layer[0]), int(raw_layer[1]))
                except (IndexError, TypeError, ValueError) as exc:
                    raise ValueError("adjacency scope has an invalid layer key") from exc
                if layer in excluded_layers:
                    continue
                for ring in tuple(rings or ()):
                    points.extend(
                        placed_point_nm(float(x), float(y), position)
                        for x, y in tuple(ring)
                    )
        if not points:
            raise ValueError("adjacency structural scope has no observable artwork")
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        block_scopes[block_id] = (min(xs), min(ys), max(xs), max(ys))

    scopes: list[Tuple[str, _Bbox]] = []
    for master, members in sorted(master_members.items()):
        if not members:
            raise ValueError("adjacency scope master has no materialized members")
        boxes = [block_scopes[member] for member in members]
        scope = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        bbox = masters[master]
        if not (
            bbox[0] <= scope[0] < scope[2] <= bbox[2]
            and bbox[1] <= scope[1] < scope[3] <= bbox[3]
        ):
            raise ValueError("adjacency structural scope escapes its rigid master")
        scopes.append((master, scope))
    result = tuple(scopes)
    if projection_evidence_sink is not None:
        from clay.layout.place.generated_incumbent import MaterializedAdjacencyScopeProjection

        canonical_geometry = _diagnostic_canonical_value(consumed_geometry)
        provenance = {
            "schema": "clay.materialized-adjacency-scope-provenance/v1",
            "producer": "clay.layout.place_blocks_2d._materialized_adjacency_scope_bboxes_nm",
            "tech_layer_name": "nwell", "lookup_status": "RESOLVED",
            "excluded_layers": _diagnostic_canonical_value(tuple(sorted(excluded_layers))),
            "master_blocks": _diagnostic_canonical_value(tuple(master_blocks)),
            "master_members": _diagnostic_canonical_value(master_members),
            "materialized_stamps": _diagnostic_canonical_value({
                member: tuple(sorted(materialized_stamps[member]))
                for member in sorted(materialized_ids)
            }),
            "materialized_rings": _diagnostic_canonical_value({
                member: materialized_rings[member] for member in sorted(materialized_ids)
                if member in materialized_rings
            }),
            "source_members": _diagnostic_canonical_value(source_members),
            "rigid_nonmos_macro_plans": (
                _diagnostic_canonical_value(rigid_nonmos_macro_plans.payload())
                if rigid_nonmos_macro_plans is not None else None
            ),
            "geometry_and_placement": canonical_geometry,
            "placed_geometry_sha256": hashlib.sha256(json.dumps(
                canonical_geometry, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ).encode("ascii")).hexdigest(),
            "absolute_scope_bboxes_nm": _diagnostic_canonical_value(result),
            "materialized_blocks": _diagnostic_canonical_value(
                projection_collision_inputs["materialized_blocks"]),
            "component_refinement_owners": _diagnostic_canonical_value(
                projection_collision_inputs["component_refinement_owners"]),
        }
        scope_offsets = _canonical_adjacency_scope_offsets_nm(result, blocks=master_blocks)
        try:
            projection = MaterializedAdjacencyScopeProjection(
                scope_offsets_nm=scope_offsets, provenance=provenance,
            )
        except Exception as exc:
            # Exact-frame construction has not been reached. Retain the actual
            # rejected input without presenting it as a frame or certificate.
            try:
                from clay.layout.place.solver_execution_policy import record_incumbent_frame_observation

                record_incumbent_frame_observation({
                    "schema": "clay.materialized-adjacency-projection-failure/v1",
                    "status": "REJECTED_MATERIALIZED_PROJECTION",
                    "diagnostic_only": True, "promotion": "REJECT",
                    "scope_offsets_nm": _diagnostic_canonical_value(scope_offsets),
                    "provenance": provenance,
                    "provenance_sha256": hashlib.sha256(json.dumps(
                        provenance, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False,
                    ).encode("ascii")).hexdigest(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                })
            except Exception:
                # Diagnostic failure must not replace the original rejection.
                pass
            raise
        projection_evidence_sink(projection)
    return result

def _source_device_bboxes_nm(
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Tuple[float, float]],
    source_devices: Sequence[str],
) -> Dict[str, _Bbox]:
    """Resolve each declared source member to one exact placed stamp bbox."""

    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm

    if (
        type(source_devices) is not tuple
        or any(type(member) is not str or not member for member in source_devices)
        or source_devices != tuple(sorted(set(source_devices)))
    ):
        raise TypeError("source device bbox inventory must be a canonical exact tuple")
    requested = set(source_devices)
    placed = _placed_stamp_bboxes_nm(geometry, placement_um)
    out: Dict[str, _Bbox] = {}
    for stamp_name, stamp in geometry.items():
        raw_members = getattr(stamp, "composite_members", ())
        if (
            type(stamp_name) is not str
            or not stamp_name
            or type(raw_members) is not tuple
            or any(type(member) is not str or not member for member in raw_members)
            or len(set(raw_members)) != len(raw_members)
        ):
            raise TypeError(
                "placed stamp composite member inventory must be an exact tuple"
            )
        owners = raw_members if raw_members else (stamp_name,)
        for owner in owners:
            if owner not in requested:
                continue
            if owner in out or stamp_name not in placed:
                raise ValueError(
                    f"floorplan column device {owner!r} lacks unique stamp bbox"
                )
            out[owner] = placed[stamp_name]
    missing = tuple(sorted(requested - set(out)))
    if missing:
        raise ValueError(
            f"floorplan column devices lack placed bbox authority: {missing!r}"
        )
    return out

def _source_device_occurrence_envelopes_nm(
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Tuple[float, float]],
    source_devices: Sequence[str],
) -> Dict[str, _Bbox]:
    """Return half-open envelopes of materialized unit-occurrence centers.

    MOS arrays expose their evaluated unit centers on the generated stamp.  These
    centers are the coordinate-free pattern frame used by strict structural
    comparison; pin/well artwork can extend asymmetrically beyond that frame.  A
    one-nanometre half-open extent keeps singleton patterns representable by the
    existing bbox frame helpers without changing the measured center coordinate.
    """

    if (
        type(source_devices) is not tuple
        or any(type(member) is not str or not member for member in source_devices)
        or source_devices != tuple(sorted(set(source_devices)))
    ):
        raise TypeError(
            "source occurrence envelope inventory must be a canonical exact tuple"
        )
    requested = set(source_devices)
    centers: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    allowed_orientations = {
        "R0", "R90", "R180", "R270",
        "MR0", "MR90", "MR180", "MR270",
    }

    def place(logical: str, stamp_name: str, raw_rows: object) -> None:
        if logical not in requested:
            return
        base = placement_um.get(stamp_name)
        if (
            logical in centers
            or type(base) is not tuple
            or len(base) != 2
            or any(type(value) not in {int, float} for value in base)
            or type(raw_rows) not in {tuple, list}
            or not raw_rows
        ):
            raise ValueError(
                f"source occurrence envelope for {logical!r} is ambiguous or absent"
            )
        placed: List[Tuple[int, int]] = []
        for row in raw_rows:
            if (
                type(row) not in {tuple, list}
                or len(row) != 3
                or type(row[0]) not in {int, float}
                or type(row[1]) not in {int, float}
                or not math.isfinite(float(row[0]))
                or not math.isfinite(float(row[1]))
                or row[2] not in allowed_orientations
            ):
                raise TypeError("source occurrence envelope row is malformed")
            placed.append(
                (
                    int(round((float(base[0]) + float(row[0])) * _NM_PER_UM)),
                    int(round((float(base[1]) + float(row[1])) * _NM_PER_UM)),
                )
            )
        centers[logical] = tuple(placed)

    for stamp_name, stamp in geometry.items():
        if type(stamp_name) is not str or not stamp_name:
            raise TypeError("source occurrence stamp name must be canonical")
        by_member = getattr(stamp, "physical_occurrences_by_member", None)
        if by_member is not None:
            if type(by_member) is not tuple or not by_member:
                raise TypeError("source member occurrence inventory is malformed")
            for logical, raw_rows in by_member:
                place(str(logical), stamp_name, raw_rows)
            continue
        raw_rows = getattr(stamp, "resolved_master_occurrences", None)
        if not raw_rows:
            raw_rows = getattr(stamp, "physical_occurrences", None)
        if not raw_rows:
            continue
        raw_members = getattr(stamp, "composite_members", ())
        if type(raw_members) is not tuple or any(
            type(member) is not str or not member for member in raw_members
        ):
            raise TypeError("source occurrence composite members are malformed")
        owners = raw_members if raw_members else (stamp_name,)
        targets = tuple(sorted(requested.intersection(owners)))
        if len(targets) > 1:
            raise ValueError("one occurrence stamp ambiguously owns multiple endpoints")
        if targets:
            place(targets[0], stamp_name, raw_rows)

    return {
        logical: (
            min(point[0] for point in rows),
            min(point[1] for point in rows),
            max(point[0] for point in rows) + 1,
            max(point[1] for point in rows) + 1,
        )
        for logical, rows in sorted(centers.items())
    }


def _source_channel_alignment_groups(blocks, block_member_devices, geometry, placement_um, intent):
    """Project unique native source-channel centers onto existing solver axes."""
    declarations = tuple(getattr(intent, 'channel_alignments', ()) or ())
    if not declarations:
        return ()
    from clay.layout.joint_occurrence_center_basis import require_joint_stamp_center_basis
    from clay.layout.place.constraints_spatial import _canonical_axis_alignment_groups
    names = tuple(sorted({name for row in declarations for name in (row.first, row.second)}))
    for name in names:
        stamps = [stamp for stamp in geometry.values()
                  if name in (getattr(stamp, 'composite_members', ()) or ())]
        if len(stamps) != 1:
            raise ValueError('channel alignment source lacks one native row stamp')
        require_joint_stamp_center_basis(stamps[0])
        entries = dict(getattr(stamps[0], 'physical_occurrences_by_member', ()) or ())
        if len(entries.get(name, ())) != 1:
            raise ValueError('channel alignment requires one actual channel per source owner')
    observed = _source_device_occurrence_envelopes_nm(geometry, placement_um, names)
    if set(observed) != set(names):
        raise ValueError('channel alignment lacks complete native channel centers')
    boxes = dict(blocks)
    owners = {}
    # The inventory retains consumed pre-bundle blocks alongside the current
    # composite. Only current solver blocks are owners of this projection.
    # Still reject a source member owned by two *live* masters.
    for owner in boxes:
        if owner not in block_member_devices:
            raise ValueError('channel alignment live solver owner has no member inventory')
        members=block_member_devices[owner]
        for name in set(members) & set(names):
            if name in owners:
                raise ValueError('channel alignment source has duplicate solver owners')
            owners[name] = owner
    if set(owners) != set(names):
        raise ValueError('channel alignment source has no solver owner')
    groups = []
    for row in declarations:
        entries = []
        for name in (row.first, row.second):
            owner = owners[name]
            if owner not in boxes:
                raise ValueError('channel alignment solver owner is absent')
            x0, y0, x1, y1 = observed[name]
            if (x1 - x0, y1 - y0) != (1, 1):
                raise ValueError('channel alignment source is not one exact channel center')
            entries.append((owner, 2 * (x0 - boxes[owner][0])))
        if entries[0][0] == entries[1][0]:
            if entries[0][1] != entries[1][1]:
                raise ValueError('one rigid owner violates source channel alignment')
            continue
        groups.append(tuple(entries))
    dims = {name: (box[2]-box[0], box[3]-box[1]) for name, box in blocks}
    return _canonical_axis_alignment_groups(tuple(groups), block_ids=tuple(boxes), dims=dims)

def _source_bbox_refinement_rows(
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Tuple[float, float]],
    materialized_source_ownership: object,
) -> Tuple[Tuple[str, _Bbox], ...]:
    """Return only physical owners with a complete unique source-bbox split.

    A composite stamp can legitimately make several source devices physically
    inseparable.  Such an owner remains rigid; it must not suppress an
    independent exact refinement proved for another owner.
    """

    if type(materialized_source_ownership) is not tuple:
        raise TypeError("materialized source ownership must be an exact tuple")
    rows: List[Tuple[str, _Bbox]] = []
    for row in materialized_source_ownership:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("materialized source ownership row must be an exact pair")
        _owner, sources = row
        if (
            type(sources) is not tuple
            or not sources
            or sources != tuple(sorted(set(sources)))
        ):
            raise TypeError(
                "materialized source ownership members must be canonical"
            )
        try:
            exact = _source_device_bboxes_nm(geometry, placement_um, sources)
        except ValueError:
            continue
        rows.extend(sorted(exact.items()))
    canonical = tuple(sorted(rows))
    if len({source for source, _bbox in canonical}) != len(canonical):
        raise ValueError("source bbox refinements repeat a source member")
    return canonical

def _project_source_device_bboxes_to_solver_masters(
    source_device_bboxes_nm: object,
    *,
    source_blocks: object,
    source_block_member_devices: object,
    solver_blocks: object,
    solver_block_member_devices: object,
    member_offsets: object,
) -> Dict[str, _Bbox]:
    """Replay source device bboxes in the current solver-master coordinate frame."""

    from clay.layout.place.coordinate_frames import (
        SourceAbsoluteBBox,
        solver_local_to_final_absolute,
        source_absolute_to_solver_local,
    )

    def exact_blocks(value: object, field: str) -> Dict[str, _Bbox]:
        if type(value) is not tuple:
            raise TypeError(f"{field} must be one exact tuple")
        rows: Dict[str, _Bbox] = {}
        for row in value:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError(f"{field} rows must be exact pairs")
            name, bbox = row
            if type(name) is not str or not name or name != name.strip():
                raise TypeError(f"{field} names must be canonical strings")
            if (
                type(bbox) is not tuple
                or len(bbox) != 4
                or any(type(coordinate) is not int for coordinate in bbox)
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise TypeError(f"{field} bboxes must be exact positive tuples")
            if name in rows:
                raise ValueError(f"{field} contains duplicate block {name!r}")
            rows[name] = bbox
        if not rows:
            raise ValueError(f"{field} must be non-empty")
        return rows

    def exact_members(
        value: object,
        *,
        field: str,
        block_names: set[str],
    ) -> Dict[str, set[str]]:
        if type(value) is not dict:
            raise TypeError(f"{field} must be one exact dict")
        if set(value) != block_names:
            raise ValueError(f"{field} must completely cover its blocks")
        rows: Dict[str, set[str]] = {}
        for block in sorted(block_names):
            raw_members = value[block]
            if type(raw_members) is not set or any(
                type(member) is not str or not member or member != member.strip()
                for member in raw_members
            ):
                raise TypeError(f"{field} members must be exact string sets")
            rows[block] = set(raw_members)
        return rows

    if type(source_device_bboxes_nm) is not dict:
        raise TypeError("source floorplan device bboxes must be one exact dict")
    device_bboxes: Dict[str, _Bbox] = {}
    for device, bbox in source_device_bboxes_nm.items():
        if type(device) is not str or not device or device != device.strip():
            raise TypeError("source floorplan device names must be canonical strings")
        if (
            type(bbox) is not tuple
            or len(bbox) != 4
            or any(type(coordinate) is not int for coordinate in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise TypeError("source floorplan device bboxes must be exact tuples")
        device_bboxes[device] = bbox
    if not device_bboxes:
        return {}

    source_by_name = exact_blocks(source_blocks, "source floorplan blocks")
    solver_by_name = exact_blocks(solver_blocks, "solver floorplan blocks")
    source_members = exact_members(
        source_block_member_devices,
        field="source floorplan block ownership",
        block_names=set(source_by_name),
    )
    solver_members = exact_members(
        solver_block_member_devices,
        field="solver floorplan block ownership",
        block_names=set(solver_by_name),
    )
    if type(member_offsets) is not dict:
        raise TypeError("source-to-solver member offsets must be one exact dict")
    offsets: Dict[str, Tuple[str, Tuple[int, int]]] = {}
    for source_block, raw in member_offsets.items():
        if (
            type(source_block) is not str
            or source_block not in source_by_name
            or type(raw) is not tuple
            or len(raw) != 2
            or type(raw[0]) is not str
            or raw[0] not in solver_by_name
            or type(raw[1]) is not tuple
            or len(raw[1]) != 2
            or any(type(coordinate) is not int for coordinate in raw[1])
        ):
            raise TypeError("source-to-solver member offsets contain a malformed row")
        offsets[source_block] = raw

    projected: Dict[str, _Bbox] = {}
    for device, bbox in device_bboxes.items():
        source_owners = tuple(
            block for block, members in source_members.items() if device in members
        )
        solver_owners = tuple(
            block for block, members in solver_members.items() if device in members
        )
        if len(source_owners) != 1 or len(solver_owners) != 1:
            raise ValueError(
                f"source floorplan device {device!r} lacks unique block ownership"
            )
        source_owner = source_owners[0]
        solver_owner = solver_owners[0]
        source_bbox = source_by_name[source_owner]
        solver_bbox = solver_by_name[solver_owner]
        if not (
            source_bbox[0] <= bbox[0]
            and source_bbox[1] <= bbox[1]
            and bbox[2] <= source_bbox[2]
            and bbox[3] <= source_bbox[3]
        ):
            raise ValueError(
                f"source floorplan device {device!r} escapes its source block"
            )
        if source_owner == solver_owner:
            if source_bbox != solver_bbox:
                raise ValueError(
                    "retained source floorplan block changed before solver replay"
                )
            member_offset = (0, 0)
        else:
            offset = offsets.get(source_owner)
            if offset is None or offset[0] != solver_owner:
                raise ValueError(
                    f"source floorplan block {source_owner!r} lacks exact solver offset"
                )
            member_offset = offset[1]
        local_bbox = source_absolute_to_solver_local(
            SourceAbsoluteBBox(bbox),
            source_master_origin_nm=(source_bbox[0], source_bbox[1]),
            member_offset_nm=member_offset,
        )
        replayed = solver_local_to_final_absolute(
            local_bbox,
            final_master_origin_nm=(solver_bbox[0], solver_bbox[1]),
        ).bbox_nm
        if not (
            solver_bbox[0] <= replayed[0]
            and solver_bbox[1] <= replayed[1]
            and replayed[2] <= solver_bbox[2]
            and replayed[3] <= solver_bbox[3]
        ):
            raise ValueError(
                f"source floorplan device {device!r} escapes its solver master"
            )
        projected[device] = replayed
    return projected

def _canonical_mos_matching_axis_rows(
    matching_pair_axes: Any,
    device_kind: Mapping[str, str],
) -> Tuple[Tuple[Tuple[str, str], str, int], ...]:
    """Replay canonical MOS matching declarations without normalizing bad input."""

    if matching_pair_axes is None:
        return ()
    if type(matching_pair_axes) is not dict:
        raise TypeError("matching-axis inventory must be an exact dict")
    rows: List[Tuple[Tuple[str, str], str, int]] = []
    for raw_pair, raw_axis in matching_pair_axes.items():
        if (
            type(raw_pair) is not tuple
            or len(raw_pair) != 2
            or any(type(member) is not str or not member for member in raw_pair)
            or raw_pair[0] >= raw_pair[1]
        ):
            raise TypeError("matching-axis pair must be canonical strings")
        if type(raw_axis) is not tuple or len(raw_axis) != 2:
            raise TypeError("matching-axis declaration must use an exact axis tuple")
        axis, coordinate = raw_axis
        if type(axis) is not str or axis not in {"x", "y"}:
            raise TypeError("matching-axis declaration must name exact x or y")
        if type(coordinate) not in {int, float} or not math.isfinite(float(coordinate)):
            raise TypeError("matching-axis coordinate must be a finite exact number")
        coordinate4 = Decimal(str(coordinate)) * Decimal(4)
        if coordinate4 != coordinate4.to_integral_value():
            raise ValueError("matching-axis coordinate must resolve to quarter-nm grid")
        kinds = tuple(device_kind.get(member) for member in raw_pair)
        mos_members = tuple(kind in {"nmos", "pmos"} for kind in kinds)
        if any(mos_members) and not all(mos_members):
            raise ValueError("matching-axis pair cannot mix MOS and non-MOS members")
        if all(mos_members):
            if kinds[0] != kinds[1]:
                raise ValueError("matching-axis MOS pair must have one polarity")
            rows.append((raw_pair, axis, int(coordinate4)))
    canonical = tuple(sorted(rows))
    if len({pair for pair, _axis, _coordinate4 in canonical}) != len(canonical):
        raise ValueError("matching-axis MOS rows must be unique")
    return canonical

def _bundle_mos_matching_composites(
    blocks: Sequence[Tuple[str, _Bbox]],
    block_member_devices: Mapping[str, set],
    *,
    matching_pair_axes: Any,
    device_kind: Mapping[str, str],
    guard_ring_owner_ids: Any = (),
    flexible_owner_ids: Any = (),
    place_groups: Any = (),
    order_constraints: Any = (),
    gap_nm: int = 0,
) -> Tuple[
    List[Tuple[str, _Bbox]],
    Dict[str, List[str]],
    Dict[str, Tuple[str, Tuple[int, int]]],
    Tuple[Tuple[str, str, str], ...],
]:
    """Lower every split canonical MOS matching pair to one rigid solver master."""

    rows = _canonical_mos_matching_axis_rows(matching_pair_axes, device_kind)
    frozen_blocks = list(blocks)
    if not rows:
        return frozen_blocks, {}, {}, ()
    block_bboxes = dict(frozen_blocks)
    if len(block_bboxes) != len(frozen_blocks):
        raise ValueError("MOS matching source blocks must be unique")
    if not isinstance(block_member_devices, Mapping):
        raise TypeError("MOS matching source ownership must be a mapping")
    forbidden_owner_sets: Dict[str, set[str]] = {}
    for label, raw_ids in (
        ("guard-ring", guard_ring_owner_ids),
        ("flexible envelope", flexible_owner_ids),
    ):
        if (
            type(raw_ids) is not tuple
            or raw_ids != tuple(sorted(set(raw_ids)))
            or any(type(owner) is not str or not owner for owner in raw_ids)
        ):
            raise TypeError(f"MOS matching {label} owners must be canonical")
        forbidden_owner_sets[label] = set(raw_ids)
    if type(place_groups) is not tuple:
        raise TypeError("MOS matching place groups must be an exact tuple")
    if type(gap_nm) is not int or gap_nm < 0:
        raise TypeError("MOS matching order gap must be one non-negative exact integer")
    try:
        raw_orders = tuple(order_constraints or ())
    except TypeError as exc:
        raise TypeError("MOS matching order constraints must be finite") from exc
    canonical_orders = []
    for raw in raw_orders:
        if type(raw) is not tuple or len(raw) != 3:
            raise TypeError("MOS matching order row must be one exact triple")
        first, second, relation = (str(value) for value in raw)
        source_order = (first, second, relation)
        if relation == "above":
            first, second, relation = second, first, "below"
        elif relation not in {"left_of", "below"}:
            continue
        canonical_orders.append((first, second, relation, source_order))
    for block_id in block_bboxes:
        members = block_member_devices.get(block_id)
        if (
            type(members) is not set
            or not members
            or any(type(member) is not str or not member for member in members)
        ):
            raise TypeError("MOS matching source ownership is malformed")

    owners_by_device: Dict[str, List[str]] = {}
    for block_id in block_bboxes:
        for member in block_member_devices[block_id]:
            owners_by_device.setdefault(member, []).append(block_id)

    from clay.layout.intent import place_group_floorplan_row

    matching_column_groups = tuple(
        group for group in place_groups if getattr(group, "col", None) is not None
    )
    matching_column_rows = tuple(
        place_group_floorplan_row(group) for group in matching_column_groups
    )
    if any(row is None for row in matching_column_rows) and any(
        row is not None for row in matching_column_rows
    ):
        raise ValueError(
            "MOS matching place-group columns mix global and row scopes"
        )
    global_matching_column_scope = bool(matching_column_rows) and all(
        row is None for row in matching_column_rows
    )

    floorplan_cells_by_device: Dict[str, set[Tuple[int, int]]] = {}
    for group in place_groups:
        column = getattr(group, "col", None)
        if column is None:
            continue
        if type(column) is not int or column < 0:
            raise TypeError("MOS matching place-group column must be exact")
        row = place_group_floorplan_row(group)
        if row is None:
            if not global_matching_column_scope:
                raise ValueError(
                    "MOS matching place-group column lacks one coherent scope"
                )
            row = _GLOBAL_FLOORPLAN_COLUMN_SCOPE
        members = getattr(group, "devices", None)
        if type(members) is not tuple or not members or any(
            type(member) is not str or not member for member in members
        ):
            raise TypeError("MOS matching place-group members must be exact")
        for member in members:
            floorplan_cells_by_device.setdefault(member, set()).add((row, column))

    consumed: set[str] = set()
    consumed_orders: set[Tuple[str, str, str]] = set()
    composites: Dict[str, List[str]] = {}
    offsets: Dict[str, Tuple[str, Tuple[int, int]]] = {}
    masters: List[Tuple[str, _Bbox]] = []
    for pair, _axis, _coordinate4 in rows:
        pair_cells = tuple(
            tuple(sorted(floorplan_cells_by_device.get(member, ())))
            for member in pair
        )
        if any(len(cells) > 1 for cells in pair_cells) or (
            all(pair_cells) and pair_cells[0] != pair_cells[1]
        ):
            raise ValueError(
                f"MOS matching pair {pair!r} crosses declared floorplan cells"
            )
        member_owners = tuple(
            tuple(sorted(owners_by_device.get(member, ()))) for member in pair
        )
        if any(len(owners) != 1 for owners in member_owners):
            raise ValueError("MOS matching member must have one source block owner")
        owners = tuple(owner_rows[0] for owner_rows in member_owners)
        if owners[0] == owners[1]:
            continue
        pair_orders = tuple(
            row
            for row in canonical_orders
            if {row[0], row[1]} == set(pair)
        )
        for first, second, relation, source_order in pair_orders:
            first_owner = owners_by_device[first][0]
            second_owner = owners_by_device[second][0]
            left = block_bboxes[first_owner]
            right = block_bboxes[second_owner]
            satisfied = (
                left[2] + gap_nm <= right[0]
                if relation == "left_of"
                else left[3] + gap_nm <= right[1]
            )
            if not satisfied:
                raise ValueError(
                    "MOS matching source geometry violates its local order: "
                    f"@order({source_order[0]},{source_order[1]},{source_order[2]})"
                )
            consumed_orders.add(source_order)
        for label, forbidden in forbidden_owner_sets.items():
            overlap = set(owners).intersection(forbidden)
            if overlap:
                raise ValueError(
                    f"MOS matching split pair conflicts with {label} owner "
                    f"{min(overlap)!r}"
                )
        overlap = consumed.intersection(owners)
        if overlap:
            raise ValueError(
                f"MOS matching pairs reuse source block {min(overlap)!r}"
            )
        master = f"MATCHED_MOS:{pair[0]}:{pair[1]}"
        if master in block_bboxes or master in composites:
            raise ValueError("MOS matching rigid master identity is not unique")
        x0 = min(block_bboxes[owner][0] for owner in owners)
        y0 = min(block_bboxes[owner][1] for owner in owners)
        x1 = max(block_bboxes[owner][2] for owner in owners)
        y1 = max(block_bboxes[owner][3] for owner in owners)
        masters.append((master, (0, 0, x1 - x0, y1 - y0)))
        composites[master] = list(owners)
        for owner in owners:
            bbox = block_bboxes[owner]
            offsets[owner] = (master, (bbox[0] - x0, bbox[1] - y0))
        consumed.update(owners)

    retained = [row for row in frozen_blocks if row[0] not in consumed]
    return retained + masters, composites, offsets, tuple(sorted(consumed_orders))

def _edge_alignment_bbox_device_inventory(
    intent: object,
    *,
    known_device_names: Sequence[str],
) -> Tuple[str, ...]:
    """Return device bboxes needed to materialize aligned-composite orders.

    An ``@order`` endpoint can belong to an edge-aligned component without being
    named by an ``@align`` declaration itself (for example, through an already
    materialized matching owner).  Preserve those device bboxes so the local
    order can be proved after the owners are flattened.  Group endpoints are not
    physical devices and therefore remain outside this inventory.
    """

    known = frozenset(str(name) for name in known_device_names)
    devices = {
        str(member)
        for declaration in tuple(getattr(intent, "edge_alignments", ()) or ())
        for member in (declaration.first, declaration.second)
    }
    if not devices:
        return ()
    for raw in tuple(getattr(intent, "order_constraints", ()) or ()):
        if type(raw) is not tuple or len(raw) != 3:
            raise TypeError("edge alignment order row must be one exact triple")
        first, second, _relation = (str(value) for value in raw)
        if first in known:
            devices.add(first)
        if second in known:
            devices.add(second)
    return tuple(sorted(devices))

def _discrete_mosaic_local_order_proofs(
    mosaics: object,
    order_constraints: object,
) -> Tuple[Tuple[str, str, str], ...]:
    """Prove device orders already frozen inside rigid mosaic topology.

    Mosaic rows are bottom-to-top and rank intervals are the coordinate-free
    horizontal frame consumed by the row materializer.  Owner centroids are
    compared in that discrete frame, so no reference coordinate or pitch enters
    this local-materialization proof.
    """

    from clay.layout.spatial.intent_projection import (
        replay_discrete_mosaic_inventory,
    )

    if type(order_constraints) is not tuple or any(
        type(row) is not tuple
        or len(row) != 3
        or any(type(value) is not str or not value for value in row)
        for row in order_constraints
    ):
        raise TypeError("mosaic local orders must be exact canonical triples")
    declarations = replay_discrete_mosaic_inventory(mosaics)
    positions_by_owner: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    mosaic_by_owner: Dict[str, str] = {}
    for declaration in declarations:
        interval_by_node = {
            (owner, ordinal): (left_rank, right_rank)
            for owner, ordinal, left_rank, right_rank in declaration.rank_intervals
        }
        local: Dict[str, List[Tuple[int, int]]] = {}
        for row_index, row in enumerate(declaration.rows):
            for column_index, occurrence in enumerate(row):
                interval = interval_by_node.get(
                    (occurrence.logical_owner, occurrence.occurrence_ordinal)
                )
                horizontal_center2 = (
                    2 * column_index
                    if interval is None
                    else interval[0] + interval[1]
                )
                local.setdefault(occurrence.logical_owner, []).append(
                    (horizontal_center2, row_index)
                )
        for owner, positions in local.items():
            if owner in positions_by_owner:
                raise ValueError(
                    f"mosaic local-order owner {owner!r} appears in two mosaics"
                )
            positions_by_owner[owner] = tuple(positions)
            mosaic_by_owner[owner] = declaration.mosaic_id

    def compare_centroids(
        first: Tuple[Tuple[int, int], ...],
        second: Tuple[Tuple[int, int], ...],
        *,
        coordinate: int,
    ) -> int:
        left = sum(row[coordinate] for row in first) * len(second)
        right = sum(row[coordinate] for row in second) * len(first)
        return (left > right) - (left < right)

    proofs = []
    for first, second, relation in order_constraints:
        if (
            first not in positions_by_owner
            or second not in positions_by_owner
            or mosaic_by_owner[first] != mosaic_by_owner[second]
        ):
            continue
        first_positions = positions_by_owner[first]
        second_positions = positions_by_owner[second]
        if relation == "left_of":
            proven = compare_centroids(
                first_positions, second_positions, coordinate=0
            ) < 0
        elif relation == "above":
            proven = compare_centroids(
                first_positions, second_positions, coordinate=1
            ) > 0
        elif relation == "below":
            proven = compare_centroids(
                first_positions, second_positions, coordinate=1
            ) < 0
        else:
            proven = False
        if proven:
            proofs.append((first, second, relation))
    return tuple(sorted(set(proofs)))

def _expand_edge_alignment_local_domains(
    model, *, local_x, local_y, grid_x, grid_y, grid, widths, heights,
    x_min, y_min, x_max, y_max, extrema_constraints, initial_x_bound, initial_y_bound,
):
    """Derive a sufficient finite domain from the actual local CP constraints.

    After choosing the existing Boolean branches, each supported coordinate
    constraint is a translation-invariant difference constraint. On the grid,
    a feasible difference graph has a normalized shortest-path solution whose
    span is at most (number of owners - 1) times its largest absolute edge
    weight. Adding the largest owner extent bounds that branch's minimum outer
    span; every actual minimum fits the bound because coordinates are
    nonnegative and each owner is inside the max extent. Since the axes are
    independent for a fixed Boolean assignment, their actual minima combine
    without changing the minimum sum of outer spans. Audit the actual model
    rather than maintaining a second copy of
    its PDK, FEOL, order, and abut formulas. Unknown constraint forms fail
    closed before the solver can report an incorrectly bounded INFEASIBLE.
    """
    from ortools.sat.python import cp_model

    if type(grid) is not int or grid <= 0 or not local_x or set(local_x) != set(local_y):
        raise ValueError('edge alignment domain requires complete owners and a positive grid')
    owners = tuple(sorted(local_x))
    if set(widths) != set(owners) or set(heights) != set(owners):
        raise ValueError('edge alignment domain dimensions lack an owner')
    coords = {var.Index(): 'x' for var in local_x.values()}
    coords.update({var.Index(): 'y' for var in local_y.values()})
    if len(coords) != 2 * len(owners):
        raise ValueError('edge alignment coordinate variables are not distinct')
    aliases = {}
    if grid > 1:
        if set(grid_x) != set(owners) or set(grid_y) != set(owners):
            raise ValueError('edge alignment grid aliases lack an owner')
        for positions, grids in ((local_x, grid_x), (local_y, grid_y)):
            for owner in owners:
                aliases[frozenset((positions[owner].Index(), grids[owner].Index()))] = (
                    positions[owner].Index(), grids[owner].Index())
    elif grid_x or grid_y:
        raise ValueError('unit-grid edge alignment has unexpected aliases')
    anchors = {x_min.Index(), y_min.Index()}
    extrema_vars = {x_min.Index(), y_min.Index(), x_max.Index(), y_max.Index()}
    grid_vars = {v.Index() for v in (*grid_x.values(), *grid_y.values())}
    if (len(anchors) != 2 or len(extrema_vars) != 4 or coords.keys() & extrema_vars
            or grid_vars & (set(coords) | extrema_vars) or len(grid_vars) != len(grid_x) + len(grid_y)):
        raise ValueError('edge alignment model variables are not distinct')
    protected = set(coords) | grid_vars | extrema_vars
    extrema = set(extrema_constraints)
    if len(extrema) != 4:
        raise ValueError('edge alignment extrema inventory differs')
    max_weights = {'x': 0, 'y': 0}
    seen_aliases, seen_anchors, seen_extrema = set(), set(), set()
    proto = model.Proto()
    for positions, grids, dimensions, bound in (
        (local_x, grid_x, widths, initial_x_bound),
        (local_y, grid_y, heights, initial_y_bound),
    ):
        if type(bound) is not int or any(type(dimensions[o]) is not int or dimensions[o] <= 0 for o in owners):
            raise ValueError('edge alignment initial domain dimensions are invalid')
        for owner in owners:
            available = bound - dimensions[owner]
            if available < 0 or list(proto.variables[positions[owner].Index()].domain) != [0, available]:
                raise ValueError('edge alignment coordinate domain differs from normalized builder')
            if grid > 1 and list(proto.variables[grids[owner].Index()].domain) != [0, available // grid]:
                raise ValueError('edge alignment grid variable domain differs from normalized builder')
    if (list(proto.variables[x_min.Index()].domain) != [0, initial_x_bound]
            or list(proto.variables[y_min.Index()].domain) != [0, initial_y_bound]
            or list(proto.variables[x_max.Index()].domain) != [1, initial_x_bound]
            or list(proto.variables[y_max.Index()].domain) != [1, initial_y_bound]
            or proto.assumptions):
        raise ValueError('edge alignment normalization domains or assumptions differ')
    expected_extrema = {
        extrema_constraints[0]: (x_min.Index(), -1, tuple((local_x[o].Index(), -1, 0) for o in owners)),
        extrema_constraints[1]: (y_min.Index(), -1, tuple((local_y[o].Index(), -1, 0) for o in owners)),
        extrema_constraints[2]: (x_max.Index(), 1, tuple((local_x[o].Index(), 1, widths[o]) for o in owners)),
        extrema_constraints[3]: (y_max.Index(), 1, tuple((local_y[o].Index(), 1, heights[o]) for o in owners)),
    }
    objective = proto.objective
    if (proto.HasField('floating_point_objective') or objective.domain
            or objective.scaling_was_exact or objective.integer_before_offset
            or objective.integer_after_offset or objective.integer_scaling_factor
            or dict(zip(objective.vars, objective.coeffs)) != {x_max.Index(): 1, y_max.Index(): 1}
            or len(objective.vars) != 2 or objective.offset != 0 or objective.scaling_factor != 1):
        raise ValueError('edge alignment objective differs from normalized span')
    for index, constraint in enumerate(proto.constraints):
        kind = constraint.WhichOneof('constraint')
        for literal in constraint.enforcement_literal:
            var_index = literal if literal >= 0 else -literal - 1
            if var_index in protected or list(proto.variables[var_index].domain) != [0, 1]:
                raise ValueError('edge alignment guard is not a pure Boolean')
        if kind == 'lin_max':
            if index not in extrema or constraint.enforcement_literal:
                raise ValueError('unsupported edge alignment extrema constraint')
            data = constraint.lin_max
            target, coefficient, expressions = expected_extrema[index]
            actual = tuple(sorted((expr.vars[0], expr.coeffs[0], expr.offset)
                                  for expr in data.exprs if len(expr.vars) == len(expr.coeffs) == 1))
            if (list(data.target.vars) != [target] or list(data.target.coeffs) != [coefficient]
                    or data.target.offset != 0 or len(actual) != len(data.exprs)
                    or actual != tuple(sorted(expressions))):
                raise ValueError('edge alignment extrema expression differs')
            seen_extrema.add(index)
            continue
        if kind == 'bool_or':
            for literal in constraint.bool_or.literals:
                var_index = literal if literal >= 0 else -literal - 1
                if var_index in protected or list(proto.variables[var_index].domain) != [0, 1]:
                    raise ValueError('edge alignment branch is not a pure Boolean')
            continue
        if kind != 'linear':
            raise ValueError('unsupported edge alignment constraint: ' + str(kind))
        linear = constraint.linear
        if len(linear.domain) != 2:
            raise ValueError('unsupported edge alignment linear domain')
        terms = dict(zip(linear.vars, linear.coeffs))
        if len(terms) != len(linear.vars):
            raise ValueError('edge alignment linear variables repeat')
        key = frozenset(terms)
        if key in aliases:
            coord, alias = aliases[key]
            if (terms != {coord: 1, alias: -grid} or list(linear.domain) != [0, 0]
                    or constraint.enforcement_literal or key in seen_aliases):
                raise ValueError('edge alignment grid alias differs')
            seen_aliases.add(key)
            continue
        if len(terms) == 1 and next(iter(terms)) in anchors:
            anchor = next(iter(terms))
            if (terms[anchor] != 1 or list(linear.domain) != [0, 0]
                    or constraint.enforcement_literal or anchor in seen_anchors):
                raise ValueError('edge alignment normalization differs')
            seen_anchors.add(anchor)
            continue
        if (len(terms) != 2 or set(terms.values()) != {-1, 1}
                or any(var not in coords for var in terms)
                or len({coords[var] for var in terms}) != 1):
            raise ValueError('unsupported non-difference edge alignment constraint')
        axis = coords[next(iter(terms))]
        lower, upper = linear.domain
        if lower != cp_model.INT_MIN:
            max_weights[axis] = max(max_weights[axis], abs((-int(lower)) // grid))
        if upper != cp_model.INT_MAX:
            max_weights[axis] = max(max_weights[axis], abs(int(upper) // grid))
    if seen_aliases != set(aliases) or seen_anchors != anchors or seen_extrema != extrema:
        raise ValueError('edge alignment model audit is incomplete')
    x_bound = max(initial_x_bound, (len(owners) - 1) * max_weights['x'] * grid + max(widths.values()))
    y_bound = max(initial_y_bound, (len(owners) - 1) * max_weights['y'] * grid + max(heights.values()))
    if x_bound > cp_model.INT_MAX or y_bound > cp_model.INT_MAX or x_bound + y_bound > cp_model.INT_MAX:
        raise ValueError('edge alignment derived domain exceeds integer limits')

    def upper(var, value):
        domain = proto.variables[var.Index()].domain
        if len(domain) != 2 or value < domain[1] or value > cp_model.INT_MAX:
            raise ValueError('edge alignment variable has an invalid derived domain')
        domain[1] = int(value)

    for owner in owners:
        upper(local_x[owner], x_bound - widths[owner])
        upper(local_y[owner], y_bound - heights[owner])
        if grid > 1:
            upper(grid_x[owner], (x_bound - widths[owner]) // grid)
            upper(grid_y[owner], (y_bound - heights[owner]) // grid)
    upper(x_max, x_bound)
    upper(y_max, y_bound)
    invalid = model.Validate()
    if invalid:
        raise ValueError('edge alignment derived-domain model is invalid: ' + invalid)
    return x_bound, y_bound


def _bundle_declared_edge_alignment_composites(
    blocks: Sequence[Tuple[str, _Bbox]],
    block_member_devices: Mapping[str, set],
    *,
    alignments: object,
    order_constraints: object = (),
    abut_pairs: object = (),
    abut_device_bboxes_nm: object = None,
    device_bboxes_nm: object,
    composite_members: object,
    member_offsets: object,
    gap_nm: int,
    manufacturing_grid_nm: int = 1,
    forbidden_owner_ids: Sequence[str] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    layer_clearances_nm: Mapping[Tuple[int, int], int] | None = None,
    pair_layer_clearances_nm: Mapping[
        Tuple[str, str, Tuple[int, int]], int
    ] | None = None,
    overlap_compatible_pair_layers: Tuple[
        Tuple[str, str, Tuple[int, int]], ...
    ] = (),
    native_feol_pairs: tuple = (),
) -> Tuple[
    List[Tuple[str, _Bbox]],
    Dict[str, List[str]],
    Dict[str, Tuple[str, Tuple[int, int]]],
    Dict[str, set],
    Tuple[Tuple[str, str, str], ...],
]:
    """Materialize declared shared-edge components as flat rigid solver masters.

    Edge equalities are solved with the existing leakage-free weighted
    difference-constraint core.  Only materialized bbox edges enter the offset;
    neither reference coordinates nor a circuit-specific placement table is read.
    An axis left free by the declarations is compact-packed deterministically.
    When exact layer atoms are supplied, the local compactor may overlap owner
    envelopes only where their actual layer rectangles remain separated.
    """

    from clay.layout.extract.dcsolve import _solve_axis
    from clay.layout.intent import replay_edge_alignment_inventory

    if type(manufacturing_grid_nm) is not int or manufacturing_grid_nm <= 0:
        raise ValueError("edge alignment manufacturing grid must be a positive integer")
    grid = manufacturing_grid_nm
    declarations = replay_edge_alignment_inventory(alignments)
    frozen_blocks = list(blocks)
    if not declarations:
        return (
            frozen_blocks,
            {str(key): list(value) for key, value in dict(composite_members).items()},
            dict(member_offsets),
            {str(key): set(value) for key, value in block_member_devices.items()},
            (),
        )
    block_bboxes = dict(frozen_blocks)
    if len(block_bboxes) != len(frozen_blocks) or not block_bboxes:
        raise ValueError("edge alignment requires unique materialized blocks")
    if type(device_bboxes_nm) is not dict:
        raise TypeError("edge alignment device bboxes must be one exact dict")
    device_bboxes: Dict[str, _Bbox] = {}
    for name, bbox in device_bboxes_nm.items():
        if (
            type(name) is not str
            or not name
            or type(bbox) is not tuple
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise TypeError("edge alignment device bbox row is malformed")
        device_bboxes[name] = bbox
    if type(composite_members) is not dict or type(member_offsets) is not dict:
        raise TypeError("edge alignment composite authority must use exact dicts")
    composites = {str(key): list(value) for key, value in composite_members.items()}
    offsets = dict(member_offsets)
    members_by_block: Dict[str, set] = {
        str(key): set(value) for key, value in block_member_devices.items()
    }
    if any(name not in members_by_block for name in block_bboxes):
        raise ValueError("edge alignment lacks block member ownership")
    forbidden = set(str(value) for value in forbidden_owner_ids)
    layered_by_owner = dict(layered_collision_footprints)
    per_layer_clearance = dict(layer_clearances_nm or {})
    per_pair_layer_clearance = dict(pair_layer_clearances_nm or {})
    compatible_pair_layers = set(overlap_compatible_pair_layers)
    from clay.layout.place.native_feol_reservation import NativeFeolPairIR
    if type(native_feol_pairs) is not tuple or any(
        type(row) is not NativeFeolPairIR
        or row.left_master not in block_bboxes or row.right_master not in block_bboxes
        for row in native_feol_pairs
    ):
        raise ValueError('edge alignment native FEOL authority is malformed')
    if (
        len(compatible_pair_layers) != len(overlap_compatible_pair_layers)
        or any(
            type(row) is not tuple
            or len(row) != 3
            or type(row[0]) is not str
            or type(row[1]) is not str
            or row[0] >= row[1]
            or row[0] not in block_bboxes
            or row[1] not in block_bboxes
            or type(row[2]) is not tuple
            or len(row[2]) != 2
            or any(type(value) is not int or value < 0 for value in row[2])
            for row in overlap_compatible_pair_layers
        )
    ):
        raise ValueError("edge alignment mergeable layer authority is malformed")
    if len(layered_by_owner) != len(layered_collision_footprints) or any(
        owner not in block_bboxes for owner in layered_by_owner
    ):
        raise ValueError("edge alignment layered collision authority is malformed")

    owner_by_device: Dict[str, str] = {}
    for owner in sorted(block_bboxes):
        for device in sorted(members_by_block[owner]):
            prior = owner_by_device.setdefault(str(device), owner)
            if prior != owner:
                raise ValueError(
                    f"edge alignment device {device!r} has multiple solver owners"
                )

    directional_rows: List[
        Tuple[str, str, str, str, Tuple[str, str, str]]
    ] = []
    try:
        raw_orders = tuple(order_constraints or ())
    except TypeError as exc:
        raise TypeError("edge alignment order constraints must be finite") from exc
    for raw in raw_orders:
        if type(raw) is not tuple or len(raw) != 3:
            raise TypeError("edge alignment order row must be one exact triple")
        first, second, relation = (str(value) for value in raw)
        source_order = (first, second, relation)
        declaration_ref = f"@order({first},{second},{relation})"
        if relation == "above":
            first, second, relation = second, first, "below"
        elif relation not in {"left_of", "below"}:
            continue
        if first not in owner_by_device or second not in owner_by_device:
            continue
        directional_rows.append(
            (first, second, relation, declaration_ref, source_order)
        )

    relation_rows: List[Tuple[str, str, str, int]] = []
    connected: Dict[str, set[str]] = {}
    edge_index = {"left": 0, "bottom": 1, "right": 2, "top": 3}
    for declaration in declarations:
        first = declaration.first
        second = declaration.second
        if first not in owner_by_device or second not in owner_by_device:
            missing = first if first not in owner_by_device else second
            raise ValueError(f"edge alignment endpoint {missing!r} lacks one owner")
        if first not in device_bboxes or second not in device_bboxes:
            missing = first if first not in device_bboxes else second
            raise ValueError(f"edge alignment endpoint {missing!r} lacks exact bbox")
        first_owner = owner_by_device[first]
        second_owner = owner_by_device[second]
        if first_owner in forbidden or second_owner in forbidden:
            raise ValueError("edge alignment cannot consume a guarded/flexible owner")
        first_box = device_bboxes[first]
        second_box = device_bboxes[second]
        if declaration.edge in {"center_x", "center_y"}:
            coord = 0 if declaration.edge == "center_x" else 1
            axis = "x" if coord == 0 else "y"
            twice_first_local = (
                first_box[coord] + first_box[coord + 2]
                - 2 * block_bboxes[first_owner][coord]
            )
            twice_second_local = (
                second_box[coord] + second_box[coord + 2]
                - 2 * block_bboxes[second_owner][coord]
            )
            twice_delta = twice_second_local - twice_first_local - 2 * declaration.offset_nm
            if first_owner == second_owner:
                if abs(twice_delta) > grid:
                    raise ValueError("one rigid owner violates its declared edge alignment")
                continue
            delta_nm = ((twice_delta + grid) // (2 * grid)) * grid
            relation_rows.append((axis, first_owner, second_owner, delta_nm))
        else:
            coordinate = edge_index[declaration.edge]
            first_local = first_box[coordinate] - block_bboxes[first_owner][coordinate % 2]
            second_local = (
                second_box[coordinate] - block_bboxes[second_owner][coordinate % 2]
            )
            if first_owner == second_owner:
                if second_local - first_local != declaration.offset_nm:
                    raise ValueError("one rigid owner violates its declared edge alignment")
                continue
            axis = "x" if declaration.edge in {"left", "right"} else "y"
            relation_rows.append(
                (axis, first_owner, second_owner,
                 second_local - first_local - declaration.offset_nm)
            )
        connected.setdefault(first_owner, set()).add(second_owner)
        connected.setdefault(second_owner, set()).add(first_owner)

    pending = set(connected)
    components: List[Tuple[str, ...]] = []
    while pending:
        seed = min(pending)
        stack = [seed]
        component: set[str] = set()
        while stack:
            owner = stack.pop()
            if owner in component:
                continue
            component.add(owner)
            stack.extend(sorted(connected.get(owner, ()), reverse=True))
        pending -= component
        components.append(tuple(sorted(component)))

    consumed: set[str] = set()
    consumed_orders: set[Tuple[str, str, str]] = set()
    masters: List[Tuple[str, _Bbox]] = []
    for component in sorted(components):
        triples = {"x": [], "y": []}
        for axis, first_owner, second_owner, offset in relation_rows:
            if first_owner in component and second_owner in component:
                triples[axis].append((first_owner, second_owner, float(offset)))
        solutions = {
            axis: _solve_axis(list(component), triples[axis], component[0])
            for axis in ("x", "y")
        }
        if any(solution.conflicts for solution in solutions.values()):
            raise ValueError("declared edge alignments form an inconsistent cycle")

        component_orders = tuple(
            row
            for row in directional_rows
            if owner_by_device[row[0]] in component
            and owner_by_device[row[1]] in component
        )
        component_abuts = tuple((str(a), str(b)) for a, b in tuple(abut_pairs or ())
            if str(a) in owner_by_device and str(b) in owner_by_device
            and owner_by_device[str(a)] in component and owner_by_device[str(b)] in component)
        abut_set = {frozenset(pair) for pair in component_abuts}
        abut_boxes = {} if abut_device_bboxes_nm is None else dict(abut_device_bboxes_nm)
        for first, second in component_abuts:
            directions = [row for row in component_orders if {row[0], row[1]} == {first, second}]
            if (len(directions) != 1 or first not in abut_boxes or second not in abut_boxes
                    or owner_by_device[first] == owner_by_device[second]):
                raise ValueError('declared edge-alignment abut requires distinct owners, exact artwork and one order')

        def parked(axis: str) -> Dict[str, int]:
            index = 0 if axis == "x" else 1
            solution = solutions[axis]
            cursor = 0
            placed: Dict[str, int] = {}
            equality_components = tuple(
                tuple(sorted(names)) for names in solution.components
            )
            equality_owner = {
                owner: names
                for names in equality_components
                for owner in names
            }
            ordered_edges = set()
            for first, second, relation, declaration_ref, _source_order in component_orders:
                if (relation == "left_of") != (axis == "x"):
                    continue
                first_component = equality_owner[owner_by_device[first]]
                second_component = equality_owner[owner_by_device[second]]
                if first_component == second_component:
                    continue
                ordered_edges.add((first_component, second_component))
            if ordered_edges:
                outgoing = {names: set() for names in equality_components}
                incoming = {names: 0 for names in equality_components}
                for first_component, second_component in sorted(ordered_edges):
                    if second_component not in outgoing[first_component]:
                        outgoing[first_component].add(second_component)
                        incoming[second_component] += 1
                ready = sorted(
                    names for names in equality_components if incoming[names] == 0
                )
                ordered_components = []
                while ready:
                    names = ready.pop(0)
                    ordered_components.append(names)
                    for follower in sorted(outgoing[names]):
                        incoming[follower] -= 1
                        if incoming[follower] == 0:
                            ready.append(follower)
                            ready.sort()
                if len(ordered_components) != len(equality_components):
                    raise ValueError(
                        f"declared edge-alignment composite has a cyclic {axis} order"
                    )
            else:
                ordered_components = list(equality_components)
            for names in ordered_components:
                lo = min(solution.pot[name] for name in names)
                hi = max(
                    solution.pot[name]
                    + block_bboxes[name][index + 2]
                    - block_bboxes[name][index]
                    for name in names
                )
                shift = cursor - lo
                for name in names:
                    value = solution.pot[name] + shift
                    rounded = int(round(value))
                    if abs(value - rounded) > 1e-9:
                        raise ValueError("edge alignment escapes the integer-nm grid")
                    if rounded % grid:
                        raise ValueError(
                            "exact edge alignment is incompatible with the manufacturing grid"
                        )
                    placed[name] = rounded
                cursor = int(round(hi + shift)) + gap_nm
                cursor = ((cursor + grid - 1) // grid) * grid
            return placed

        xs = parked("x")
        ys = parked("y")
        component_layered = tuple(
            (owner, layered_by_owner[owner])
            for owner in component
            if owner in layered_by_owner
        )
        if not component_layered and any(
            row.left_master in component and row.right_master in component
            for row in native_feol_pairs
        ):
            raise ValueError('native FEOL edge alignment requires layered collision geometry')
        if component_layered:
            from ortools.sat.python import cp_model

            widths = {
                name: block_bboxes[name][2] - block_bboxes[name][0]
                for name in component
            }
            heights = {
                name: block_bboxes[name][3] - block_bboxes[name][1]
                for name in component
            }
            x_bound = sum(widths.values()) + gap_nm * max(0, len(component) - 1)
            y_bound = sum(heights.values()) + gap_nm * max(0, len(component) - 1)
            # Keep the legal rigid-grid parking candidate inside the local
            # search domain.  Shapes and declared edge offsets are unchanged.
            x_bound = max(x_bound, max(xs[name] + widths[name] for name in component))
            y_bound = max(y_bound, max(ys[name] + heights[name] for name in component))
            model = cp_model.CpModel()
            local_x = {
                name: model.NewIntVar(0, x_bound - widths[name], f"edge_x_{name}")
                for name in component
            }
            local_y = {
                name: model.NewIntVar(0, y_bound - heights[name], f"edge_y_{name}")
                for name in component
            }
            grid_x, grid_y = {}, {}
            if grid > 1:
                for name in component:
                    grid_x[name] = model.NewIntVar(
                        0, (x_bound - widths[name]) // grid, f"edge_x_grid_{name}"
                    )
                    grid_y[name] = model.NewIntVar(
                        0, (y_bound - heights[name]) // grid, f"edge_y_grid_{name}"
                    )
                    model.Add(local_x[name] == grid * grid_x[name])
                    model.Add(local_y[name] == grid * grid_y[name])
            for axis, first_owner, second_owner, offset in relation_rows:
                if first_owner not in component or second_owner not in component:
                    continue
                coordinates = local_x if axis == "x" else local_y
                model.Add(
                    coordinates[first_owner] - coordinates[second_owner]
                    == int(offset)
                )
            for first, second, relation, _declaration_ref, _source_order in component_orders:
                first_owner = owner_by_device[first]
                second_owner = owner_by_device[second]
                first_source = device_bboxes[first]
                second_source = device_bboxes[second]
                first_owner_box = block_bboxes[first_owner]
                second_owner_box = block_bboxes[second_owner]
                order_gap = 0 if frozenset((first, second)) in abut_set else gap_nm
                if relation == "left_of":
                    model.Add(
                        local_x[first_owner]
                        + first_source[2]
                        - first_owner_box[0]
                        + order_gap
                        <= local_x[second_owner]
                        + second_source[0]
                        - second_owner_box[0]
                    )
                else:
                    model.Add(
                        local_y[first_owner]
                        + first_source[3]
                        - first_owner_box[1]
                        + order_gap
                        <= local_y[second_owner]
                        + second_source[1]
                        - second_owner_box[1]
                    )
            from clay.layout.place.native_feol_reservation import master_displacement_covers
            component_feol = tuple(row for row in native_feol_pairs
                                   if row.left_master in component and row.right_master in component)
            for left, right, cover in master_displacement_covers(component_feol):
                dx = local_x[right] - local_x[left]
                dy = local_y[right] - local_y[left]
                for index, (x0, y0, x1, y1) in enumerate(cover):
                    separated = tuple(model.NewBoolVar(f'edge_feol_{left}_{right}_{index}_{axis}')
                                      for axis in ('left', 'right', 'below', 'above'))
                    model.Add(dx <= x0 - 1).OnlyEnforceIf(separated[0])
                    model.Add(dx >= x1).OnlyEnforceIf(separated[1])
                    model.Add(dy <= y0 - 1).OnlyEnforceIf(separated[2])
                    model.Add(dy >= y1).OnlyEnforceIf(separated[3])
                    model.AddBoolOr(separated)
            if component_abuts:
                from clay.layout.geometric_intent_checks import _DEFAULT_ABUT_TOL
                for first, second in component_abuts:
                    ordered, = [row for row in component_orders if {row[0], row[1]} == {first, second}]
                    before, after, relation = ordered[:3]
                    before_owner, after_owner = owner_by_device[before], owner_by_device[after]
                    before_box, after_box = abut_boxes[before], abut_boxes[after]
                    before_master, after_master = block_bboxes[before_owner], block_bboxes[after_owner]
                    axis = 0 if relation == 'left_of' else 1
                    across = 1 - axis
                    coords = local_x if axis == 0 else local_y
                    cross_coords = local_y if axis == 0 else local_x
                    trailing = coords[before_owner] + before_box[axis+2] - before_master[axis]
                    leading = coords[after_owner] + after_box[axis] - after_master[axis]
                    maximum_gap = int(_DEFAULT_ABUT_TOL * max(before_box[axis+2]-before_box[axis],
                                                              after_box[axis+2]-after_box[axis]))
                    model.Add(leading >= trailing)
                    model.Add(leading <= trailing + maximum_gap)
                    # Ensure the declared facing axis is the one measured by
                    # the existing abut observer, not a diagonal separation.
                    model.Add(cross_coords[before_owner] + before_box[across+2] - before_master[across]
                              >= cross_coords[after_owner] + after_box[across] - after_master[across] + 1)
                    model.Add(cross_coords[after_owner] + after_box[across+2] - after_master[across]
                              >= cross_coords[before_owner] + before_box[across] - before_master[across] + 1)
            component_dims = {
                name: (widths[name], heights[name]) for name in component
            }
            collision_atoms = _collision_atoms_by_owner(
                dims=component_dims,
                layered_collision_footprints=component_layered,
            )
            collision_index = 0
            for left_index, left in enumerate(component):
                for right in component[left_index + 1 :]:
                    for left_layer, left_box in collision_atoms[left]:
                        for right_layer, right_box in collision_atoms[right]:
                            if (
                                left_layer is not None
                                and right_layer is not None
                                and left_layer != right_layer
                            ):
                                continue
                            if (
                                left_layer is not None
                                and left_layer == right_layer
                                and (left, right, left_layer)
                                in compatible_pair_layers
                            ):
                                continue
                            atom_clearance_nm = (
                                per_pair_layer_clearance.get(
                                    (left, right, left_layer),
                                    per_layer_clearance.get(left_layer, gap_nm),
                                )
                                if left_layer is not None
                                and left_layer == right_layer
                                else gap_nm
                            )
                            collision_index += 1
                            disjunction = tuple(
                                model.NewBoolVar(
                                    f"edge_atom_{direction}_{collision_index}"
                                )
                                for direction in ("left", "right", "below", "above")
                            )
                            left_of, right_of, below, above = disjunction
                            model.Add(
                                local_x[left] + left_box[2] + atom_clearance_nm
                                <= local_x[right] + right_box[0]
                            ).OnlyEnforceIf(left_of)
                            model.Add(
                                local_x[right] + right_box[2] + atom_clearance_nm
                                <= local_x[left] + left_box[0]
                            ).OnlyEnforceIf(right_of)
                            model.Add(
                                local_y[left] + left_box[3] + atom_clearance_nm
                                <= local_y[right] + right_box[1]
                            ).OnlyEnforceIf(below)
                            model.Add(
                                local_y[right] + right_box[3] + atom_clearance_nm
                                <= local_y[left] + left_box[1]
                            ).OnlyEnforceIf(above)
                            model.AddBoolOr(disjunction)
            x_min = model.NewIntVar(0, x_bound, "edge_x_min")
            y_min = model.NewIntVar(0, y_bound, "edge_y_min")
            x_max = model.NewIntVar(1, x_bound, "edge_x_max")
            y_max = model.NewIntVar(1, y_bound, "edge_y_max")
            extrema_constraints = [
                model.AddMinEquality(x_min, [local_x[name] for name in component]).Index(),
                model.AddMinEquality(y_min, [local_y[name] for name in component]).Index(),
            ]
            extrema_constraints.append(model.AddMaxEquality(
                x_max, [local_x[name] + widths[name] for name in component]
            ).Index())
            extrema_constraints.append(model.AddMaxEquality(
                y_max, [local_y[name] + heights[name] for name in component]
            ).Index())
            model.Add(x_min == 0)
            model.Add(y_min == 0)
            for name in component:
                model.AddHint(local_x[name], xs[name])
                model.AddHint(local_y[name], ys[name])
            model.Minimize(x_max + y_max)
            x_bound, y_bound = _expand_edge_alignment_local_domains(
                model, local_x=local_x, local_y=local_y, grid_x=grid_x, grid_y=grid_y,
                grid=grid, widths=widths, heights=heights, x_min=x_min, y_min=y_min,
                x_max=x_max, y_max=y_max, extrema_constraints=tuple(extrema_constraints),
                initial_x_bound=x_bound, initial_y_bound=y_bound,
            )
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            solver.parameters.max_deterministic_time = 1.0
            status = solver.Solve(model)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                atom_counts = tuple(
                    (name, len(collision_atoms[name])) for name in component
                )
                equality_sets = {
                    axis: tuple(frozenset(names) for names in solutions[axis].components)
                    for axis in ("x", "y")
                }
                fixed_pair_conflicts = []
                for left_index, left in enumerate(component):
                    for right in component[left_index + 1 :]:
                        if not all(
                            any({left, right} <= names for names in equality_sets[axis])
                            for axis in ("x", "y")
                        ):
                            continue
                        for left_layer, left_box in collision_atoms[left]:
                            for right_layer, right_box in collision_atoms[right]:
                                if left_layer is None or left_layer != right_layer:
                                    continue
                                if (
                                    (left, right, left_layer)
                                    in compatible_pair_layers
                                ):
                                    continue
                                shifted_left = (
                                    left_box[0] + xs[left],
                                    left_box[1] + ys[left],
                                    left_box[2] + xs[left],
                                    left_box[3] + ys[left],
                                )
                                shifted_right = (
                                    right_box[0] + xs[right],
                                    right_box[1] + ys[right],
                                    right_box[2] + xs[right],
                                    right_box[3] + ys[right],
                                )
                                clearance_nm = _bbox_clearance_nm(
                                    shifted_left, shifted_right
                                )
                                required_nm = per_pair_layer_clearance.get(
                                    (left, right, left_layer),
                                    per_layer_clearance.get(left_layer, gap_nm),
                                )
                                if clearance_nm < required_nm:
                                    fixed_pair_conflicts.append(
                                        (
                                            left,
                                            right,
                                            left_layer,
                                            clearance_nm,
                                            required_nm,
                                            shifted_left,
                                            shifted_right,
                                        )
                                    )
                raise ValueError(
                    "declared edge-alignment layered compaction did not produce "
                    f"a candidate: status={solver.StatusName(status)}; "
                    f"component={component}; dims={tuple(sorted(component_dims.items()))}; "
                    f"relations={tuple(sorted(row for row in relation_rows if row[1] in component and row[2] in component))}; "
                    f"orders={tuple(row[:3] for row in component_orders)}; "
                    f"atom_counts={atom_counts}; collision_pairs={collision_index}; "
                    f"fixed_pair_conflicts={tuple(fixed_pair_conflicts)}"
                )
            xs = {name: int(solver.Value(local_x[name])) for name in component}
            ys = {name: int(solver.Value(local_y[name])) for name in component}
        elif component_abuts:
            raise ValueError('declared edge-alignment abut requires PDK layered collision geometry')
        placed_boxes = {
            name: (
                xs[name],
                ys[name],
                xs[name] + block_bboxes[name][2] - block_bboxes[name][0],
                ys[name] + block_bboxes[name][3] - block_bboxes[name][1],
            )
            for name in component
        }
        placed_device_boxes = {}
        for device, owner in owner_by_device.items():
            if owner not in component or device not in device_bboxes:
                continue
            source = device_bboxes[device]
            owner_source = block_bboxes[owner]
            dx = xs[owner] - owner_source[0]
            dy = ys[owner] - owner_source[1]
            placed_device_boxes[device] = (
                source[0] + dx,
                source[1] + dy,
                source[2] + dx,
                source[3] + dy,
            )
        for first, second, relation, declaration_ref, source_order in component_orders:
            left = placed_device_boxes[first]
            right = placed_device_boxes[second]
            order_gap = 0 if frozenset((first, second)) in abut_set else gap_nm
            satisfied = (
                left[2] + order_gap <= right[0]
                if relation == "left_of"
                else left[3] + order_gap <= right[1]
            )
            if not satisfied:
                raise ValueError(
                    "declared edge-alignment composite failed to materialize "
                    f"{declaration_ref}"
                )
            consumed_orders.add(source_order)
        if component_abuts:
            from types import SimpleNamespace
            from clay.layout.geometric_intent_checks import _DEFAULT_ABUT_TOL, check_abut
            artwork = {}
            for device, source in abut_boxes.items():
                owner = owner_by_device.get(device)
                if owner in component:
                    dx, dy = xs[owner] - block_bboxes[owner][0], ys[owner] - block_bboxes[owner][1]
                    artwork[device] = (source[0]+dx, source[1]+dy, source[2]+dx, source[3]+dy)
            for first, second in component_abuts:
                measured, missing = check_abut(SimpleNamespace(device_bboxes=artwork,
                    cluster_member_to_a={}), first, second)
                if missing or measured is None or measured > _DEFAULT_ABUT_TOL:
                    raise ValueError('declared edge-alignment abut failed actual artwork replay')
        if component_layered:
            from clay.layout.place.native_feol_reservation import native_feol_rows_satisfied
            component_feol = tuple(row for row in native_feol_pairs
                                   if row.left_master in component and row.right_master in component)
            if not native_feol_rows_satisfied(component_feol,
                    {name: (xs[name], ys[name]) for name in component}):
                raise ValueError('declared edge-alignment native FEOL failed replay')
            if not _seed_geometry_legal(
                {name: (xs[name], ys[name]) for name in component},
                dims={
                    name: (
                        block_bboxes[name][2] - block_bboxes[name][0],
                        block_bboxes[name][3] - block_bboxes[name][1],
                    )
                    for name in component
                },
                gap_nm=gap_nm,
                pair_clearances_nm={},
                layered_collision_footprints=component_layered,
                layer_clearances_nm=per_layer_clearance,
                pair_layer_clearances_nm=per_pair_layer_clearance,
                overlap_compatible_pair_layers=tuple(
                    sorted(compatible_pair_layers)
                ),
            ):
                raise ValueError(
                    "declared edge-alignment layered compaction failed replay"
                )
        else:
            for index, first in enumerate(component):
                for second in component[index + 1 :]:
                    clearance_nm = _bbox_clearance_nm(
                        placed_boxes[first], placed_boxes[second]
                    )
                    if clearance_nm < 0:
                        raise ValueError(
                            "declared edge alignment creates an overlapping rigid "
                            f"component: {first} {placed_boxes[first]} overlaps "
                            f"{second} {placed_boxes[second]} "
                            f"(clearance_nm={clearance_nm})"
                        )
        width = max(box[2] for box in placed_boxes.values())
        height = max(box[3] for box in placed_boxes.values())
        endpoint_names = tuple(
            sorted(
                {
                    name
                    for declaration in declarations
                    for name in (declaration.first, declaration.second)
                    if owner_by_device[name] in component
                }
            )
        )
        master = f"EDGE_ALIGN:{'+'.join(endpoint_names)}"
        if master in block_bboxes or master in composites:
            raise ValueError("edge alignment rigid master identity is not unique")
        flattened: List[str] = []
        final_members: set = set()
        for owner in component:
            owner_offset = (xs[owner], ys[owner])
            final_members.update(members_by_block[owner])
            if owner in composites:
                parts = composites.pop(owner)
                for part in parts:
                    prior_owner, prior_offset = offsets[part]
                    if prior_owner != owner:
                        raise ValueError("nested composite offset authority is stale")
                    offsets[part] = (
                        master,
                        (
                            owner_offset[0] + prior_offset[0],
                            owner_offset[1] + prior_offset[1],
                        ),
                    )
                    flattened.append(part)
            else:
                offsets[owner] = (master, owner_offset)
                flattened.append(owner)
        composites[master] = sorted(flattened)
        members_by_block[master] = final_members
        masters.append((master, (0, 0, width, height)))
        consumed.update(component)

    retained = [row for row in frozen_blocks if row[0] not in consumed]
    return (
        retained + masters,
        composites,
        offsets,
        members_by_block,
        tuple(sorted(consumed_orders)),
    )

def _matching_axis_master_authority(
    *,
    matching_pair_axes: Any,
    device_kind: Mapping[str, str],
    live_master_ids: Tuple[str, ...],
    block_member_devices: Mapping[str, set],
    master_members: Mapping[str, Tuple[str, ...]],
    member_deltas_nm: Mapping[str, Tuple[int, int]],
) -> Tuple[Dict[str, set[str]], Dict[str, Tuple[int, int]]]:
    """Bind each MOS matching pair to one live master with one pair-rigid delta.

    A later coordinate-free alignment can merge the matching owner with unrelated
    source blocks whose pre-solve deltas differ.  Only the source blocks that own
    the two matching endpoints prove preservation of the matching axis; requiring
    every unrelated member of the larger live master to share their delta would
    reject a valid flattened rigid component.
    """

    rows = _canonical_mos_matching_axis_rows(matching_pair_axes, device_kind)
    if not rows:
        return {}, {}
    if (
        type(live_master_ids) is not tuple
        or live_master_ids != tuple(sorted(set(live_master_ids)))
        or any(type(master) is not str or not master for master in live_master_ids)
    ):
        raise TypeError("matching-axis live masters must be exact and canonical")
    for label, value in (
        ("live ownership", block_member_devices),
        ("master members", master_members),
        ("member deltas", member_deltas_nm),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"matching-axis {label} must be a mapping")

    live_ownership: Dict[str, set[str]] = {}
    for master in live_master_ids:
        members = block_member_devices.get(master)
        if (
            type(members) is not set
            or not members
            or any(type(member) is not str or not member for member in members)
        ):
            raise TypeError("matching-axis live master ownership is malformed")
        live_ownership[master] = set(members)

    selected_ownership: Dict[str, set[str]] = {}
    selected_deltas: Dict[str, Tuple[int, int]] = {}
    for pair, _axis, _coordinate4 in rows:
        owners = tuple(
            tuple(
                master
                for master in live_master_ids
                if member in live_ownership[master]
            )
            for member in pair
        )
        if (
            any(len(member_owners) != 1 for member_owners in owners)
            or owners[0][0] != owners[1][0]
        ):
            raise ValueError("matching-axis pair must have one rigid live master")
        owner = owners[0][0]
        source_members = master_members.get(owner)
        if (
            type(source_members) is not tuple
            or not source_members
            or any(type(member) is not str or not member for member in source_members)
        ):
            raise TypeError("matching-axis live master members are malformed")
        endpoint_source_owners = tuple(
            tuple(
                member
                for member in source_members
                if endpoint in set(block_member_devices.get(member, ()))
            )
            for endpoint in pair
        )
        if any(len(owners) != 1 for owners in endpoint_source_owners):
            raise ValueError(
                "matching-axis endpoint lacks one source member in its live master"
            )
        pair_source_members = tuple(sorted({
            owner for owners in endpoint_source_owners for owner in owners
        }))
        deltas = tuple(
            member_deltas_nm.get(member) for member in pair_source_members
        )
        if any(
            type(delta) is not tuple
            or len(delta) != 2
            or any(type(value) is not int for value in delta)
            for delta in deltas
        ):
            raise TypeError("matching-axis source member lacks an exact block delta")
        if len(set(deltas)) != 1:
            raise ValueError("matching-axis live master lacks one rigid solver delta")
        delta = deltas[0]
        prior = selected_deltas.get(owner)
        if prior is not None and prior != delta:
            raise ValueError("matching-axis live master delta changed between pairs")
        selected_ownership[owner] = set(live_ownership[owner])
        selected_deltas[owner] = delta
    return selected_ownership, selected_deltas

def _matching_axis_cumulative_authority(
    *,
    matching_pair_axes: Any,
    device_kind: Mapping[str, str],
    live_master_ids: Tuple[str, ...],
    block_member_devices: Mapping[str, set],
    master_members: Mapping[str, Tuple[str, ...]],
    first_member_deltas_nm: Mapping[str, Tuple[int, int]],
    replay_member_deltas_nm: Mapping[str, Tuple[int, int]],
) -> Tuple[Dict[str, set[str]], Dict[str, Tuple[int, int]]]:
    """Prove each solve-stage rigid delta before composing its final transport."""

    common = dict(
        matching_pair_axes=matching_pair_axes,
        device_kind=device_kind,
        live_master_ids=live_master_ids,
        block_member_devices=block_member_devices,
        master_members=master_members,
    )
    first_ownership, first_deltas = _matching_axis_master_authority(
        **common,
        member_deltas_nm=first_member_deltas_nm,
    )
    replay_ownership, replay_deltas = _matching_axis_master_authority(
        **common,
        member_deltas_nm=replay_member_deltas_nm,
    )
    if first_ownership != replay_ownership or set(first_deltas) != set(replay_deltas):
        raise ValueError("matching-axis live ownership changed across replay")
    return first_ownership, {
        master: (
            first_deltas[master][0] + replay_deltas[master][0],
            first_deltas[master][1] + replay_deltas[master][1],
        )
        for master in sorted(first_deltas)
    }

def _materialized_source_ownership_requested(
    group_route_requested: bool,
    intent_fact_set: IntentFactSet | None,
) -> bool:
    """Keep physical-to-source ownership whenever a downstream fact consumes it."""

    if type(group_route_requested) is not bool:
        raise TypeError("group-route ownership request must be one exact bool")
    if intent_fact_set is not None and type(intent_fact_set) is not IntentFactSet:
        raise TypeError("fact-governed ownership requires an exact IntentFactSet")
    return group_route_requested or intent_fact_set is not None

def _project_mos_matching_axes_through_block_deltas(
    *,
    matching_pair_axes: Any,
    source_device_bboxes: Mapping[str, _Bbox],
    final_device_bboxes: Mapping[str, _Bbox],
    source_block_member_devices: Mapping[str, set[str]],
    device_kind: Mapping[str, str],
    block_deltas_nm: Mapping[str, Tuple[int, int]],
) -> Tuple[MatchingPairAxisProjection, ...]:
    """Certify MOS axes transported by one exact rigid source-block delta.

    The source materializer must already place both equal physical footprints
    around one axis with the same direction as the canonical logical axis.  The
    two coordinate values may differ because an interdigitated PCell and its
    solver envelope are distinct views.  Both views are bound and transported by
    one original block owner's recorded delta.  Final geometry is therefore a
    replay check, never the source of either projected coordinate.
    """

    rows = _canonical_mos_matching_axis_rows(matching_pair_axes, device_kind)
    if not rows:
        return ()
    for label, mapping in (
        ("source device bbox", source_device_bboxes),
        ("final device bbox", final_device_bboxes),
        ("source block ownership", source_block_member_devices),
        ("block delta", block_deltas_nm),
    ):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"matching-axis {label} authority must be a mapping")
    owners_by_device: Dict[str, List[str]] = {}
    for owner, members in source_block_member_devices.items():
        if type(owner) is not str or not owner or type(members) is not set or any(
            type(member) is not str or not member for member in members
        ):
            raise TypeError("matching-axis source block ownership is malformed")
        for member in members:
            owners_by_device.setdefault(member, []).append(owner)
    projections: List[MatchingPairAxisProjection] = []
    for pair, axis, source_logical_coordinate4 in rows:
        source_boxes = tuple(
            _exact_matching_bbox(member, source_device_bboxes.get(member))
            for member in pair
        )
        final_boxes = tuple(
            _exact_matching_bbox(member, final_device_bboxes.get(member))
            for member in pair
        )
        source_sizes = tuple(
            (bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in source_boxes
        )
        final_sizes = tuple(
            (bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in final_boxes
        )
        if source_sizes[0] != source_sizes[1] or final_sizes != source_sizes:
            raise ValueError("matching-axis pair does not preserve equal stamp sizes")
        axis_indices = (0, 2) if axis == "x" else (1, 3)
        perpendicular_indices = (1, 3) if axis == "x" else (0, 2)
        replayed_source_coordinate4 = sum(
            bbox[index] for bbox in source_boxes for index in axis_indices
        )
        if sum(source_boxes[0][index] for index in perpendicular_indices) != sum(
            source_boxes[1][index] for index in perpendicular_indices
        ):
            raise ValueError("matching-axis source pair lacks perpendicular alignment")
        owners = tuple(owners_by_device.get(member, ()) for member in pair)
        if (
            any(len(member_owners) != 1 for member_owners in owners)
            or owners[0][0] != owners[1][0]
        ):
            raise ValueError("matching-axis pair must have one rigid source owner")
        owner = owners[0][0]
        delta = block_deltas_nm.get(owner)
        if (
            type(delta) is not tuple
            or len(delta) != 2
            or any(type(value) is not int for value in delta)
        ):
            raise TypeError("matching-axis source owner lacks an exact block delta")
        for source_bbox, final_bbox in zip(source_boxes, final_boxes):
            expected = (
                source_bbox[0] + delta[0],
                source_bbox[1] + delta[1],
                source_bbox[2] + delta[0],
                source_bbox[3] + delta[1],
            )
            if final_bbox != expected:
                raise ValueError("matching-axis final stamp does not replay rigid delta")
        axis_delta4 = 4 * delta[0 if axis == "x" else 1]
        final_logical_coordinate4 = source_logical_coordinate4 + axis_delta4
        final_coordinate4 = replayed_source_coordinate4 + axis_delta4
        replayed_final_coordinate4 = sum(
            bbox[index] for bbox in final_boxes for index in axis_indices
        )
        if replayed_final_coordinate4 != final_coordinate4:
            raise ValueError("matching-axis final coordinate does not replay rigid delta")
        if sum(final_boxes[0][index] for index in perpendicular_indices) != sum(
            final_boxes[1][index] for index in perpendicular_indices
        ):
            raise ValueError("matching-axis final pair lacks perpendicular alignment")
        projections.append(
            MatchingPairAxisProjection(
                pair=pair,
                axis=axis,
                source_logical_coordinate4_nm=source_logical_coordinate4,
                source_coordinate4_nm=replayed_source_coordinate4,
                source_master=owner,
                delta_nm=delta,
                final_logical_coordinate4_nm=final_logical_coordinate4,
                final_coordinate4_nm=final_coordinate4,
                source_member_bboxes_nm=source_boxes,
                final_member_bboxes_nm=final_boxes,
            )
        )
    return tuple(sorted(projections))

def _bundle_rigid_nonmos_composites(
    blocks: Sequence[Tuple[str, _Bbox]],
    block_member_devices: Mapping[str, set],
    rigid_macro_plans: object,
    route_member_bboxes_nm: Mapping[str, _Bbox],
) -> Tuple[
    List[Tuple[str, _Bbox]],
    Dict[str, List[str]],
    Dict[str, Tuple[str, Tuple[int, int]]],
]:
    """Expose each multi-member precompiled non-MOS plan as one solver atom.

    The generic placer has already materialized the exact plan offsets.  This
    adapter therefore preserves the current block union and relative origins;
    it never repacks members with the compact solver's unrelated global gap.
    """

    from clay.layout.nonrow_macro_plan import RigidNonmosMacroPlanSet

    if type(rigid_macro_plans) is not RigidNonmosMacroPlanSet:
        raise TypeError("rigid non-MOS composite bundling requires an exact plan set")
    bb_by_id = dict(blocks)
    consumed: set[str] = set()
    composites: Dict[str, List[str]] = {}
    offsets: Dict[str, Tuple[str, Tuple[int, int]]] = {}
    masters: List[Tuple[str, _Bbox]] = []
    for plan in rigid_macro_plans.plans:
        if len(plan.members) < 2:
            continue
        route_translations: set[Tuple[int, int]] = set()
        for member_plan in plan.member_plans:
            route_bbox = route_member_bboxes_nm.get(member_plan.member)
            if route_bbox is None:
                raise ValueError(
                    f"rigid macro {plan.macro_id!r} lacks route bbox for "
                    f"{member_plan.member!r}"
                )
            local_bbox = member_plan.local_bbox_nm
            if (
                route_bbox[2] - route_bbox[0],
                route_bbox[3] - route_bbox[1],
            ) != (
                local_bbox[2] - local_bbox[0],
                local_bbox[3] - local_bbox[1],
            ):
                raise ValueError(
                    f"rigid macro {plan.macro_id!r} route footprint differs from plan"
                )
            route_translations.add(
                (
                    route_bbox[0] - local_bbox[0],
                    route_bbox[1] - local_bbox[1],
                )
            )
        if len(route_translations) != 1:
            raise ValueError(
                f"rigid macro {plan.macro_id!r} route offsets differ from plan"
            )
        owned_blocks = tuple(
            sorted(
                {
                    block_id
                    for member in plan.members
                    for block_id, devices in block_member_devices.items()
                    if member in devices
                }
            )
        )
        if not owned_blocks:
            raise ValueError(f"rigid macro {plan.macro_id!r} has no materialized block")
        for member in plan.members:
            hits = tuple(
                block_id
                for block_id in owned_blocks
                if member in block_member_devices.get(block_id, set())
            )
            if len(hits) != 1:
                raise ValueError(
                    f"rigid macro {plan.macro_id!r} member {member!r} has "
                    f"{len(hits)} materialized block owners"
                )
            route_bbox = route_member_bboxes_nm[member]
            owner_bbox = bb_by_id[hits[0]]
            if not (
                owner_bbox[0] <= route_bbox[0]
                and owner_bbox[1] <= route_bbox[1]
                and route_bbox[2] <= owner_bbox[2]
                and route_bbox[3] <= owner_bbox[3]
            ):
                raise ValueError(
                    f"rigid macro {plan.macro_id!r} route member bbox escapes "
                    f"owner block {hits[0]!r}"
                )
        if len(owned_blocks) == 1:
            continue
        overlap = consumed.intersection(owned_blocks)
        if overlap:
            raise ValueError(
                f"rigid macro {plan.macro_id!r} reuses block {min(overlap)!r}"
            )
        consumed.update(owned_blocks)
        x0 = min(bb_by_id[block_id][0] for block_id in owned_blocks)
        y0 = min(bb_by_id[block_id][1] for block_id in owned_blocks)
        x1 = max(bb_by_id[block_id][2] for block_id in owned_blocks)
        y1 = max(bb_by_id[block_id][3] for block_id in owned_blocks)
        master = f"A_nonrow:{plan.macro_id}"
        masters.append((master, (0, 0, x1 - x0, y1 - y0)))
        composites[master] = list(owned_blocks)
        for block_id in owned_blocks:
            bbox = bb_by_id[block_id]
            offsets[block_id] = (master, (bbox[0] - x0, bbox[1] - y0))
    retained = [(block_id, bbox) for block_id, bbox in blocks if block_id not in consumed]
    return retained + masters, composites, offsets

def _rigid_compound_footprints_from_members(
    *,
    solver_blocks: Sequence[Tuple[str, _Bbox]],
    materialized_blocks: Sequence[Tuple[str, _Bbox]],
    master_members: Mapping[str, Sequence[str]],
    compound_owner_ids: Tuple[str, ...],
) -> Tuple[_RigidCompoundFootprint, ...]:
    """Project exact materialized member boxes into rigid-master local space."""

    if type(compound_owner_ids) is not tuple:
        raise TypeError("compound owner ids must be one exact tuple")
    canonical_owners = tuple(sorted(compound_owner_ids))
    if (
        canonical_owners != compound_owner_ids
        or len(set(canonical_owners)) != len(canonical_owners)
        or any(type(owner) is not str or not owner for owner in canonical_owners)
    ):
        raise ValueError("compound owner ids must be canonical and unique")
    if not isinstance(master_members, Mapping):
        raise TypeError("compound master members must be a mapping")
    solver_by_name = dict(_exact_cut_input_blocks(tuple(solver_blocks)))
    materialized_by_name = dict(
        _exact_cut_input_blocks(tuple(materialized_blocks))
    )
    dimensions = {
        owner: (bbox[2] - bbox[0], bbox[3] - bbox[1])
        for owner, bbox in solver_by_name.items()
    }
    footprints: list[_RigidCompoundFootprint] = []
    for owner in canonical_owners:
        if owner not in solver_by_name or owner not in master_members:
            raise ValueError("compound owner is absent from final-master authority")
        members = tuple(master_members[owner])
        if (
            len(members) < 2
            or tuple(sorted(members)) != members
            or len(set(members)) != len(members)
            or any(type(member) is not str or not member for member in members)
        ):
            raise ValueError(
                "compound owner requires canonical unique materialized members"
            )
        missing = tuple(member for member in members if member not in materialized_by_name)
        if missing:
            raise ValueError(
                f"compound owner {owner!r} lacks materialized members {missing!r}"
            )
        boxes = tuple(materialized_by_name[member] for member in members)
        union = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        if (
            union[2] - union[0],
            union[3] - union[1],
        ) != dimensions[owner]:
            raise ValueError(
                f"compound owner {owner!r} dimensions differ from member union"
            )
        footprints.append(
            (
                owner,
                tuple(sorted(
                    (
                        box[0] - union[0],
                        box[1] - union[1],
                        box[2] - union[0],
                        box[3] - union[1],
                    )
                    for box in boxes
                )),
            )
        )
    return _canonical_rigid_compound_footprints(
        tuple(footprints),
        dims=dimensions,
    )

def _fixed_pcell_compound_footprints_from_geometry(
    *,
    solver_blocks: Sequence[Tuple[str, _Bbox]],
    block_stamps: Mapping[str, Sequence[str]],
    geometry: Mapping[str, Any],
    master_members: Mapping[str, Sequence[str]],
) -> Tuple[_RigidCompoundFootprint, ...]:
    """Expose exact cell-local fixed-PCell occurrence envelopes to collision logic.

    This is deliberately generic: a solver master qualifies only when its complete
    materialized ownership resolves to one geometry stamp carrying the optional
    ``collision_rectangles_nm`` bridge.  Absolute placement is never consumed.
    """

    if not isinstance(block_stamps, Mapping):
        raise TypeError("fixed PCell block stamps must be a mapping")
    if not isinstance(geometry, Mapping):
        raise TypeError("fixed PCell geometry must be a mapping")
    if not isinstance(master_members, Mapping):
        raise TypeError("fixed PCell master members must be a mapping")
    solver_by_name = dict(_exact_cut_input_blocks(tuple(solver_blocks)))
    dimensions = {
        owner: (bbox[2] - bbox[0], bbox[3] - bbox[1])
        for owner, bbox in solver_by_name.items()
    }
    footprints: list[_RigidCompoundFootprint] = []
    for owner in sorted(solver_by_name):
        raw_members = master_members.get(owner, (owner,))
        if not isinstance(raw_members, Sequence) or isinstance(raw_members, str):
            raise TypeError("fixed PCell master members must be sequences")
        stamp_names = tuple(sorted({
            str(stamp)
            for member in raw_members
            for stamp in tuple(block_stamps.get(str(member), ()) or ())
        }))
        bridged = []
        for stamp in stamp_names:
            fixed_pcell = getattr(geometry.get(stamp), "fixed_pcell", {}) or {}
            if type(fixed_pcell) is not dict:
                raise TypeError("fixed PCell collision authority must be an exact dict")
            if "collision_rectangles_nm" in fixed_pcell:
                bridged.append((stamp, fixed_pcell["collision_rectangles_nm"]))
        if not bridged:
            continue
        if len(stamp_names) != 1 or len(bridged) != 1:
            raise ValueError(
                "fixed PCell collision authority requires one complete geometry stamp"
            )
        footprints.append((owner, bridged[0][1]))
    return _canonical_rigid_compound_footprints(
        tuple(footprints),
        dims=dimensions,
    )

def _layered_collision_footprints_from_geometry(
    *,
    solver_blocks: Sequence[Tuple[str, _Bbox]],
    materialized_blocks: Sequence[Tuple[str, _Bbox]],
    block_stamps: Mapping[str, Sequence[str]],
    block_gring: Mapping[str, str],
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Tuple[float, float]],
    master_members: Mapping[str, Sequence[str]],
    member_offsets_nm: Mapping[
        str, Tuple[str, Tuple[int, int]]
    ] = MappingProxyType({}),
    component_refinement_owners: Tuple[str, ...] = (),
    flexible_envelope_constraints: tuple = (),
    excluded_layers: Tuple[Tuple[int, int], ...] = (),
) -> Tuple[_LayeredCollisionFootprint, ...]:
    """Project placed artwork and owned ring domains into local collision atoms.

    The polygon inventory remains owned by the existing geometry stamps.  This
    adapter consumes the shared exact-nm placed-layer extractor and removes only
    the rigid master's common translation.  A master with any missing stamp,
    placement, ring, or layer evidence is omitted so downstream collision logic
    falls back to its ordinary solid owner/compound footprint.  A declared guard
    ring owns its interior, not just the drawn frame: its per-layer envelope is
    retained as an exclusion domain for foreign stamps.  In particular, an
    unoccupied active-layer hole inside a solid PMOS body well is not permission
    to place an unrelated substrate NMOS there.  Native component cavities that
    are not declared guard-ring domains retain their exact polygon refinement.
    """

    for label, value in (
        ("block stamps", block_stamps),
        ("block rings", block_gring),
        ("geometry", geometry),
        ("placement", placement_um),
        ("master members", master_members),
        ("member offsets", member_offsets_nm),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"layered collision {label} must be a mapping")
    solver = dict(_exact_cut_input_blocks(tuple(solver_blocks)))
    materialized = dict(_exact_cut_input_blocks(tuple(materialized_blocks)))
    dimensions = {
        owner: (bbox[2] - bbox[0], bbox[3] - bbox[1])
        for owner, bbox in solver.items()
    }
    from .constraints_spatial import _canonical_flexible_envelope_constraints
    flexible = _canonical_flexible_envelope_constraints(
        flexible_envelope_constraints, block_ids=tuple(solver))
    physical_offsets = {member: offset for row in flexible
                        for member, offset in row.member_physical_offsets_nm}
    if set(master_members) != set(solver):
        raise ValueError("layered collision ownership must cover exact solver masters")
    if (
        type(component_refinement_owners) is not tuple
        or component_refinement_owners
        != tuple(sorted(set(component_refinement_owners)))
        or not set(component_refinement_owners) <= set(solver)
    ):
        raise ValueError("layered collision refinement owners are malformed")
    refined_component_owners = set(component_refinement_owners)

    from clay.layout.extract.realize_render import (
        _placed_stamp_layer_bboxes_nm,
        _placed_stamp_layer_rectangles_nm,
    )

    footprints: list[_LayeredCollisionFootprint] = []
    for owner in sorted(solver):
        if owner in physical_offsets:
            # These solver coordinates describe content, whereas the enclosing
            # ring has its own typed physical offsets and foreign exclusion.
            # Keep the conservative physical-offset collision proof; a fixed
            # layered footprint cannot be certified in this smaller frame.
            continue
        raw_members = master_members[owner]
        if isinstance(raw_members, str):
            raise TypeError("layered collision master members must be sequences")
        try:
            members = tuple(raw_members)
        except TypeError as exc:
            raise TypeError("layered collision master members must be finite") from exc
        if (
            not members
            or any(type(member) is not str or not member for member in members)
            or len(set(members)) != len(members)
        ):
            raise ValueError("layered collision master members are malformed")
        if any(member not in materialized for member in members):
            raise ValueError("layered collision master lacks a materialized member")
        member_shifts_nm: Dict[str, Tuple[int, int]] = {}
        member_boxes = []
        for member in members:
            source_box = materialized[member]
            target_box = source_box
            if member in member_offsets_nm:
                raw_binding = member_offsets_nm[member]
                if type(raw_binding) is not tuple or len(raw_binding) != 2:
                    raise TypeError(
                        "layered collision member offsets must be exact pairs"
                    )
                target_owner, raw_offset = raw_binding
                if (
                    target_owner != owner
                    or type(raw_offset) is not tuple
                    or len(raw_offset) != 2
                    or any(type(value) is not int for value in raw_offset)
                ):
                    raise ValueError(
                        "layered collision member offset names a different master"
                    )
                ox, oy = raw_offset
                target_box = (
                    ox,
                    oy,
                    ox + source_box[2] - source_box[0],
                    oy + source_box[3] - source_box[1],
                )
            member_shifts_nm[member] = (
                target_box[0] - source_box[0],
                target_box[1] - source_box[1],
            )
            member_boxes.append(target_box)
        member_boxes = tuple(member_boxes)
        union = (
            min(bbox[0] for bbox in member_boxes),
            min(bbox[1] for bbox in member_boxes),
            max(bbox[2] for bbox in member_boxes),
            max(bbox[3] for bbox in member_boxes),
        )
        if (union[2] - union[0], union[3] - union[1]) != dimensions[owner]:
            # Virtual/flexible solver envelopes do not necessarily equal their
            # current materialized member union.  They remain safe solid owners.
            continue

        complete = True
        stamp_names: list[str] = []
        stamp_shifts_nm: Dict[str, Tuple[int, int]] = {}
        for member in members:
            try:
                stamps = tuple(block_stamps.get(member, ()))
            except TypeError:
                complete = False
                break
            if (
                not stamps
                or any(type(stamp) is not str or not stamp for stamp in stamps)
            ):
                complete = False
                break
            stamp_names.extend(stamps)
            for stamp in stamps:
                shift = member_shifts_nm[member]
                if stamp in stamp_shifts_nm and stamp_shifts_nm[stamp] != shift:
                    raise ValueError(
                        "layered collision stamp has ambiguous composite offsets"
                    )
                stamp_shifts_nm[stamp] = shift
            ring = block_gring.get(member)
            if ring is not None:
                if type(ring) is not str or not ring:
                    complete = False
                    break
                stamp_names.append(ring)
                shift = member_shifts_nm[member]
                if ring in stamp_shifts_nm and stamp_shifts_nm[ring] != shift:
                    raise ValueError(
                        "layered collision ring has ambiguous composite offsets"
                    )
                stamp_shifts_nm[ring] = shift
        canonical_stamps = tuple(sorted(set(stamp_names)))
        if not complete or not canonical_stamps:
            continue

        # This is an optional refinement of the conservative solid-owner
        # collision proof.  Only promote a master when every one of its
        # device/ring stamps exposes the exact polygon inventory consumed by
        # the shared placed-layer extractor.  Simplified or legacy geometry
        # must fall back to the solid owner, not make the entire placement
        # pipeline fail while constructing a finer proof.
        exact_geometry: Dict[str, Any] = {}
        exact_placement: Dict[str, Tuple[float, float]] = {}
        for stamp in canonical_stamps:
            cg = geometry.get(stamp)
            pos = placement_um.get(stamp)
            raw_layers = getattr(cg, "polys_by_layer", None)
            if (
                cg is None
                or pos is None
                or type(raw_layers) is not dict
                or any(
                    type(layer) is not tuple
                    or len(layer) != 2
                    or any(type(value) is not int for value in layer)
                    or type(rings) is not list
                    for layer, rings in raw_layers.items()
                )
            ):
                complete = False
                break
            exact_geometry[stamp] = cg
            shift_x_nm, shift_y_nm = stamp_shifts_nm[stamp]
            exact_placement[stamp] = (
                float(pos[0]) + shift_x_nm / _NM_PER_UM,
                float(pos[1]) + shift_y_nm / _NM_PER_UM,
            )
        if not complete:
            continue
        try:
            placed_layers = _placed_stamp_layer_bboxes_nm(
                exact_geometry,
                exact_placement,
            )
            ring_stamps = tuple(sorted({
                block_gring[member]
                for member in members
                if block_gring.get(member) is not None
            }))
            # Component cavities are real placement opportunities, but a guard
            # ring's empty drawn interior still belongs to its declared members.
            # Refining the ring's active/implant/M1 frame into hollow bars would
            # admit a foreign NMOS under its solid N-well. Same-layer collision
            # checks alone cannot detect that change of semiconductor body.
            # Keep each ring layer's full owned envelope; do not alter any GDS
            # polygon or prevent independent upper-metal placement over FEOL.
            refined_stamps = (
                tuple(stamp for stamp in canonical_stamps if stamp not in ring_stamps)
                if owner in refined_component_owners else ()
            )
            placed_rectangles = _placed_stamp_layer_rectangles_nm(
                {stamp: exact_geometry[stamp] for stamp in refined_stamps},
                {stamp: exact_placement[stamp] for stamp in refined_stamps},
            ) if refined_stamps else {}
        except (TypeError, ValueError):
            # Exact atoms are an optional refinement.  Unsupported polygon
            # geometry retains the conservative solid-owner proof.
            continue
        if (
            any(stamp not in placed_layers for stamp in canonical_stamps)
            or any(stamp not in placed_rectangles for stamp in refined_stamps)
        ):
            continue

        atoms = []
        for stamp in canonical_stamps:
            layer_rectangles = (
                placed_rectangles[stamp]
                if stamp in placed_rectangles
                else {
                    layer: (bbox,)
                    for layer, bbox in placed_layers[stamp].items()
                }
            )
            for layer, rectangles in layer_rectangles.items():
                if layer in excluded_layers:
                    continue
                for bbox in rectangles:
                    if not (
                        union[0] <= bbox[0] < bbox[2] <= union[2]
                        and union[1] <= bbox[1] < bbox[3] <= union[3]
                    ):
                        raise ValueError("layered collision artwork escapes its master: "
                            f"owner={owner!r}, stamp={stamp!r}, layer={layer!r}, "
                            f"artwork={bbox!r}, master={union!r}")
                    atoms.append((
                        layer[0],
                        layer[1],
                        bbox[0] - union[0],
                        bbox[1] - union[1],
                        bbox[2] - union[0],
                        bbox[3] - union[1],
                    ))
        atoms = tuple(sorted(set(atoms)))
        if not atoms:
            continue
        footprints.append((owner, atoms))
    return _canonical_layered_collision_footprints(
        tuple(footprints),
        dims=dimensions,
    )

def _feedback_final_master_inventory(
    blocks: Sequence[Tuple[str, _Bbox]],
    block_member_devices: Mapping[str, set],
    source_certificate: object,
) -> Tuple[
    List[Tuple[str, _Bbox]],
    Dict[str, List[str]],
    Dict[str, Tuple[str, Tuple[int, int]]],
    Dict[str, set],
    Dict[str, Tuple[str, ...]],
]:
    """Recover the sole signed final-master inventory for a feedback re-entry.

    The first placement pass may have bundled several materialized passive blocks
    into one master and then translated that master away from the origin.  A
    repair must solve those exact signed masters.  Re-running the initial
    bundler would create a fresh ``(0, 0, width, height)`` surrogate and sever the
    solve-cut placement hash from the geometry being solved.
    """

    from clay.layout.place.resource_feedback import (
        FinalPlacementCertificateView,
        final_placement_state_digest,
    )

    if not isinstance(source_certificate, FinalPlacementCertificateView):
        raise ValueError("feedback repair requires a typed source certificate")
    # Strictly replay the signed state, including any materialized member map.
    final_placement_state_digest(source_certificate)
    extracted = _exact_cut_input_blocks(blocks)
    source_masters = _exact_cut_input_blocks(source_certificate.blocks)
    raw_member_rows = tuple(source_certificate.materialized_block_members)
    if raw_member_rows:
        materialized = _exact_cut_input_blocks(
            source_certificate.materialized_block_bboxes
        )
        if materialized != extracted:
            raise ValueError(
                "feedback geometry differs from signed materialized blocks"
            )
        master_members = {
            str(master): tuple(str(member) for member in members)
            for master, members in raw_member_rows
        }
    else:
        if source_masters != extracted:
            raise ValueError("feedback geometry differs from signed source blocks")
        master_members = {name: (name,) for name, _bbox in source_masters}

    extracted_by_name = dict(extracted)
    masters_by_name = dict(source_masters)
    if set(master_members) != set(masters_by_name):
        raise ValueError("feedback source master partition is incomplete")
    flattened = tuple(
        member for master in sorted(master_members) for member in master_members[master]
    )
    if len(flattened) != len(set(flattened)) or set(flattened) != set(
        extracted_by_name
    ):
        raise ValueError("feedback source master partition is not exact")

    composite_members: Dict[str, List[str]] = {}
    member_offsets: Dict[str, Tuple[str, Tuple[int, int]]] = {}
    master_device_members: Dict[str, set] = {}
    for master, members in sorted(master_members.items()):
        master_box = masters_by_name[master]
        member_boxes = [extracted_by_name[member] for member in members]
        projected = (
            min(box[0] for box in member_boxes),
            min(box[1] for box in member_boxes),
            max(box[2] for box in member_boxes),
            max(box[3] for box in member_boxes),
        )
        if projected != master_box:
            raise ValueError(
                "feedback source master bbox differs from its materialized members"
            )
        devices: set = set()
        for member in members:
            devices |= set(block_member_devices.get(member, ()))
        master_device_members[master] = devices
        if len(members) > 1:
            composite_members[master] = list(members)
            for member in members:
                member_box = extracted_by_name[member]
                member_offsets[member] = (
                    master,
                    (
                        member_box[0] - master_box[0],
                        member_box[1] - master_box[1],
                    ),
                )

    return (
        list(source_masters),
        composite_members,
        member_offsets,
        master_device_members,
        master_members,
    )

def _diff_axis_positions_in_solver_frame(*, source_blocks, source_members,
        solver_blocks, solver_members, member_offsets, placement_um):
    """Move source access positions with the exact source-to-master transforms.

    Bundled solver masters are often normalized to zero while their source
    geometry remains in an arbitrary absolute frame. Axis offsets must use the
    solver frame. This is the same rigid transform as bbox projection; the
    final materialized replay continues to read actual final coordinates.
    """
    source_bboxes = dict(source_blocks)
    solver_bboxes = dict(solver_blocks)
    result = dict(placement_um)
    for master in solver_bboxes:
        devices = solver_members[master]
        for device in devices:
            if device not in placement_um:
                continue
            owners = [name for name,members in source_members.items() if device in members]
            if len(owners) != 1:
                raise ValueError('diff-axis device lacks unique source block ownership')
            owner = owners[0]
            if owner == master:
                if source_bboxes[owner] != solver_bboxes[master]:
                    raise ValueError('retained diff-axis source block changed frame')
                continue
            offset = member_offsets.get(owner)
            if offset is None or offset[0] != master:
                raise ValueError('diff-axis source block lacks exact solver offset')
            dx = solver_bboxes[master][0] + offset[1][0] - source_bboxes[owner][0]
            dy = solver_bboxes[master][1] + offset[1][1] - source_bboxes[owner][1]
            px,py = placement_um[device]
            result[device] = (px+dx/_NM_PER_UM,py+dy/_NM_PER_UM)
    return result


def _diff_pair_axis_alignment_groups(
    blocks: Sequence[Tuple[str, _Bbox]],
    block_member_devices: Mapping[str, Sequence[str]],
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Tuple[float, float]],
    intent: Any,
    binding: Any,
) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    """Derive hard block-axis groups from exact materialized diff accesses.

    The compact master moves guard-ring/device clusters as rigid blocks.  When
    one declared differential pair occurs in several such blocks within one
    floorplan column, preserving each block internally is insufficient: their
    local pair axes must also land on one column-local x axis.  A complete exact
    floorplan-column inventory is stronger authority than inferred global-axis
    alignment, because balanced inter-column routing does not require unrelated
    columns to share one x coordinate.  Without complete column authority the
    historical global-axis rule remains unchanged.  Within each block, every
    access row must remain strictly off-axis and pure-side, while a symmetric
    cross-coupled block whose row polarity flips cleanly across distinct Y rows
    receives neutral polarity ``0`` so its centerline aligns without
    conflicting with uniform ``±1`` blocks.  This lowerer reads only declared
    ``diff_pairs``, bound device nets, declared place groups, and the
    stamp-owned access inventory; it never uses a circuit or coordinate table.
    """

    role_to_terminal = {
        "source": "s",
        "drain": "d",
        "gate": "g",
        "body": "b",
        "s": "s",
        "d": "d",
        "g": "g",
        "b": "b",
    }
    block_ids = {str(name) for name, _bbox in blocks}
    device_block: Dict[str, str] = {}
    for block_id in sorted(block_ids):
        for device in sorted(
            str(name) for name in block_member_devices.get(block_id, ())
        ):
            prior = device_block.setdefault(device, block_id)
            if prior != block_id:
                raise ValueError(
                    f"device {device!r} belongs to multiple compact master blocks"
                )

    points_by_block_net: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for device in tuple(getattr(binding, "devices", ()) or ()):
        name = str(getattr(device, "name", "") or "")
        block_id = device_block.get(name)
        stamp = geometry.get(name)
        position = placement_um.get(name)
        if block_id is None or stamp is None or position is None:
            continue
        exact = getattr(stamp, "terminal_accesses_um", {}) or {}
        nets = getattr(device, "nets", {}) or {}
        for role, net in nets.items():
            terminal = role_to_terminal.get(str(role).lower())
            accesses = tuple(exact.get(terminal, ()) or ()) if terminal else ()
            if not accesses:
                continue
            for access in accesses:
                try:
                    x_nm = int(
                        round((float(position[0]) + float(access[0])) * _NM_PER_UM)
                    )
                    y_nm = int(
                        round((float(position[1]) + float(access[1])) * _NM_PER_UM)
                    )
                except (IndexError, TypeError, ValueError):
                    raise ValueError(
                        f"malformed exact terminal access on {name!r}:{terminal}"
                    ) from None
                points_by_block_net.setdefault((block_id, str(net)), []).append(
                    (x_nm, y_nm)
                )

    bbox_by_block = {str(name): tuple(int(v) for v in bbox) for name, bbox in blocks}
    raw_place_groups = getattr(intent, "place_groups", ())
    if type(raw_place_groups) is not tuple:
        raise TypeError("diff-axis floorplan authority must be one exact tuple")
    declared_devices: set[str] = set()
    column_by_device: Dict[str, int] = {}
    for group in raw_place_groups:
        members = getattr(group, "devices", ())
        if type(members) is not tuple or any(
            type(member) is not str or not member for member in members
        ):
            raise TypeError(
                "diff-axis floorplan group members must be exact strings in a tuple"
            )
        declared_devices.update(members)
        column = getattr(group, "col", None)
        if column is None:
            continue
        if type(column) is not int or column < 0:
            raise TypeError(
                "diff-axis floorplan column must be one non-negative exact int"
            )
        for member in members:
            prior = column_by_device.get(member)
            if prior is not None and prior != column:
                raise ValueError(
                    f"device {member!r} belongs to conflicting floorplan columns"
                )
            column_by_device[member] = column

    groups: set[Tuple[Tuple[str, int], ...]] = set()
    for net_a, net_b in tuple(getattr(intent, "diff_pairs", ()) or ()):
        evidence: List[Tuple[str, int, int, int, int, int]] = []
        for block_id in sorted(block_ids):
            left = points_by_block_net.get((block_id, str(net_a)), ())
            right = points_by_block_net.get((block_id, str(net_b)), ())
            if not left or not right:
                continue
            by_y_left: Dict[int, List[int]] = {}
            by_y_right: Dict[int, List[int]] = {}
            for x_nm, y_nm in left:
                by_y_left.setdefault(y_nm, []).append(x_nm)
            for x_nm, y_nm in right:
                by_y_right.setdefault(y_nm, []).append(x_nm)
            if set(by_y_left) != set(by_y_right) or any(
                len(by_y_left[y_nm]) != len(by_y_right[y_nm]) for y_nm in by_y_left
            ):
                raise ValueError(
                    f"diff pair {net_a!r}/{net_b!r} has unmatched exact "
                    f"access rows inside block {block_id!r}"
                )
            axis2_values = {
                x_left + x_right
                for y_nm in sorted(by_y_left)
                for x_left, x_right in zip(
                    sorted(by_y_left[y_nm]),
                    reversed(sorted(by_y_right[y_nm])),
                )
            }
            if len(axis2_values) != 1:
                raise ValueError(
                    f"diff pair {net_a!r}/{net_b!r} has no single exact "
                    f"axis inside block {block_id!r}"
                )
            axis2 = next(iter(axis2_values))
            row_polarities: set[int] = set()
            for y_nm in sorted(by_y_left):
                row_signs = {
                    -1 if 2 * x_nm < axis2 else 1 if 2 * x_nm > axis2 else 0
                    for x_nm in by_y_left[y_nm]
                }
                if 0 in row_signs or len(row_signs) != 1:
                    raise ValueError(
                        f"diff pair {net_a!r}/{net_b!r} straddles its own exact "
                        f"axis inside block {block_id!r}"
                    )
                row_polarities.add(next(iter(row_signs)))
            block_polarity = (
                next(iter(row_polarities)) if len(row_polarities) == 1 else 0
            )
            offset2 = axis2 - 2 * bbox_by_block[block_id][0]
            evidence.append(
                (
                    block_id,
                    int(block_polarity),
                    min(x_nm for x_nm, _y_nm in left),
                    max(x_nm for x_nm, _y_nm in left),
                    int(axis2),
                    int(offset2),
                )
            )

        column_by_block: Optional[Dict[str, int]] = None
        if evidence and column_by_device:
            resolved: Dict[str, int] = {}
            complete = True
            for block_id, *_rest in evidence:
                source_members = (
                    set(block_member_devices.get(block_id, ())) & declared_devices
                )
                if not source_members:
                    complete = False
                    break
                columns = {
                    column_by_device[member]
                    for member in source_members
                    if member in column_by_device
                }
                if len(columns) != 1 or any(
                    member not in column_by_device for member in source_members
                ):
                    raise ValueError(
                        f"diff-axis master {block_id!r} crosses or partially "
                        "consumes declared floorplan columns"
                    )
                resolved[block_id] = next(iter(columns))
            if complete:
                column_by_block = resolved

        partitions: Dict[Optional[int], List[Tuple[str, int, int, int, int, int]]] = {}
        for item in evidence:
            column = None if column_by_block is None else column_by_block[item[0]]
            partitions.setdefault(column, []).append(item)
        for column, items in sorted(
            partitions.items(), key=lambda item: (-1 if item[0] is None else item[0])
        ):
            declared_polarity: Optional[int] = None
            polarity_evidence: List[Tuple[str, int, int, int, int]] = []
            entries: List[Tuple[str, int]] = []
            for block_id, block_polarity, xmin, xmax, axis2, offset2 in items:
                polarity_evidence.append(
                    (block_id, block_polarity, xmin, xmax, axis2)
                )
                if block_polarity != 0:
                    if declared_polarity is None:
                        declared_polarity = block_polarity
                    elif block_polarity != declared_polarity:
                        scope = "globally" if column is None else f"in column {column}"
                        raise ValueError(
                            f"diff pair {net_a!r}/{net_b!r} reverses side polarity "
                            f"{scope} at {block_id!r}: expected {declared_polarity}, "
                            f"got {block_polarity}; evidence={tuple(polarity_evidence)!r}"
                        )
                entries.append((block_id, offset2))
            if len(entries) >= 2:
                groups.add(tuple(entries))
    merged: List[Dict[str, int]] = []
    for group in sorted(groups):
        mapping = dict(group)
        while True:
            target_index: Optional[int] = None
            for index, existing in enumerate(merged):
                shared_keys = set(existing) & set(mapping)
                if not shared_keys:
                    continue
                if any(existing[key] != mapping[key] for key in shared_keys):
                    continue
                target_index = index
                break
            if target_index is None:
                break
            existing = merged.pop(target_index)
            mapping = {**existing, **mapping}
        merged.append(mapping)
    return tuple(sorted(tuple(sorted(mapping.items())) for mapping in merged))

def _unbound_adjacency_band_overrides(
    adjacency_pairs: Sequence[Tuple[str, str]],
    *,
    bands: Mapping[str, int],
    declared_band_blocks: set[str],
    movable_blocks: set[str],
) -> Dict[str, int]:
    """Let hard adjacency place electrically unbound MOS masters near anchors.

    Polarity is a fallback for masters outside the declared rail/center sets.
    A connected movable component inherits a unique neighbouring band. Explicit
    owners are never moved; conflicting anchors remain for the original solver
    to diagnose. This does not change isolation, adjacency limits, or source
    rail-affinity validation of the final geometry.
    """
    movable = (set(movable_blocks) & set(bands)) - declared_band_blocks
    neighbours: Dict[str, set[str]] = {block: set() for block in movable}
    for left, right in adjacency_pairs:
        if left == right or left not in bands or right not in bands:
            continue
        if left in movable:
            neighbours[left].add(right)
        if right in movable:
            neighbours[right].add(left)
    result: Dict[str, int] = {}
    unseen = set(movable)
    while unseen:
        component = set()
        pending = [min(unseen)]
        while pending:
            block = pending.pop()
            if block in component:
                continue
            component.add(block)
            pending.extend(neighbours[block] & movable - component)
        unseen -= component
        anchors = {
            bands[other]
            for block in component
            for other in neighbours[block] - component
        }
        if len(anchors) == 1:
            band = next(iter(anchors))
            result.update((block, band) for block in component)
    return result


def _resistor_adjacency_terms(
    res_targets: Mapping[str, set[str]],
    *,
    cap_blocks: set[str],
    bands: Mapping[str, int],
    declared_band_blocks: set[str],
) -> Tuple[Tuple[Tuple[str, str], ...], Dict[str, int]]:
    """Compile resistor proximity without overriding explicitly owned members."""

    pulls: set[Tuple[str, str]] = set()
    overrides: Dict[str, int] = {}
    for resistor in sorted(res_targets):
        targets = set(res_targets[resistor])
        cap_neighbours = sorted(targets & cap_blocks)
        selected = cap_neighbours or sorted(targets)
        pulls.update((resistor, target) for target in selected)
        if resistor not in declared_band_blocks and not cap_neighbours:
            for target in selected:
                if target in bands:
                    overrides[resistor] = bands[target]
    return tuple(sorted(pulls)), overrides
