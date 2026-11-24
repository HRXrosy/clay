"""Source-owned ordered rows with mirror pairs that may enclose other devices.

This is a structural input to the existing row MILP, not a device generator.
The constructor keeps pair halves as separate placement members and binds only
explicit device-level symmetry groups to a common axis.  Matching by itself
retains an independently movable pair axis.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Mapping


class OrderedRowError(ValueError):
    """A declared ordered row cannot be constructed or its geometry disagrees."""


@dataclass(frozen=True, slots=True)
class DeclaredMirrorGroup:
    axis: str
    pairs: tuple[tuple[str, str], ...]
    on_axis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise OrderedRowError("ORDERED_ROW_AXIS_INVALID")
        if type(self.pairs) is not tuple or type(self.on_axis) is not tuple:
            raise OrderedRowError("ORDERED_ROW_GROUP_NOT_EXACT")
        if any(type(pair) is not tuple or len(pair) != 2 for pair in self.pairs):
            raise OrderedRowError("ORDERED_ROW_PAIR_INVALID")
        members = tuple(member for pair in self.pairs for member in pair) + self.on_axis
        if (not members or len(members) > 2048
                or any(type(member) is not str or not member for member in members)
                or len(set(members)) != len(members)):
            raise OrderedRowError("ORDERED_ROW_GROUP_MEMBERS_INVALID")

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(member for pair in self.pairs for member in pair) + self.on_axis


@dataclass(frozen=True, slots=True)
class OrderedRowProgram:
    # Physical rows, bottom to top, retaining their source row ordinals.
    rows: tuple[tuple[int, tuple[str, ...]], ...]
    pairs: tuple[tuple[str, str], ...]
    mirror_groups: tuple[DeclaredMirrorGroup, ...]
    # Axis owner 0 is the exported first declared axis. Remaining owners have
    # independent x variables in the existing solver, including ordinary matches.
    axis_owners: tuple[tuple[str, int], ...]
    source_orders: tuple[tuple[str, str, str], ...]
    edge_clearances_nm: tuple[tuple[str, str, int], ...]
    row_clearances_nm: tuple[tuple[int, int, int], ...]
    source_sizes_nm: tuple[tuple[str, int, int], ...]
    source_signature: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class OrderedRowFloorplan:
    core_bbox_nm: tuple[int, int, int, int]
    floorplan_bbox_nm: tuple[int, int, int, int]
    # Exact existing rail-lowerer declarations plus the installed PDK spacing.
    # This reserves space; the existing rail planner must still prove legality.
    edge_bands: tuple[tuple[str, str, tuple[int, int], int, int, str], ...]
    manufacturing_grid_nm: int


@dataclass(frozen=True, slots=True)
class OrderedRowCompactionEvidence:
    """Verified per-row Y-compaction / bottom-alignment proof bound to FinalPlacementCertificate."""

    certified_bbox_nm: tuple[int, int, int, int]
    declared_bottom_edge_keepout_nm: int
    edge_right_margin_nm: int
    row_clearances_nm: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class OrderedRowEvidence:
    program: OrderedRowProgram
    # Values of declared axis variables, before any subsequent placement edits.
    axis_coordinates_nm: tuple[int, ...]
    row_bboxes_nm: tuple[tuple[int, tuple[int, int, int, int]], ...]
    floorplan: OrderedRowFloorplan | None = None
    cell_identity_sha256: str = ""
    source_file_bindings: tuple[tuple[str, str], ...] = ()
    compaction: OrderedRowCompactionEvidence | None = None

    @property
    def axis_x_nm(self) -> int | None:
        return self.axis_coordinates_nm[0] if len(self.axis_coordinates_nm) == 1 else None

    @property
    def floorplan_bbox_nm(self) -> tuple[int, int, int, int] | None:
        return self.floorplan.floorplan_bbox_nm if self.floorplan is not None else None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()


def _source_signature(constraints) -> str:
    data = {
        "floorplan_rows": sorted(constraints.floorplan_rows.items()),
        "floorplan_groups": constraints.floorplan_groups,
        "pairs": constraints.symmetry_pairs,
        "on_axis": constraints.on_axis_devices,
        "orders": constraints.order_constraints,
        "groups": [asdict(group) for group in constraints.declared_mirror_groups],
        "row_gaps": constraints.row_gap_minima_nm,
        "device_kinds": sorted(constraints.device_kinds.items()),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _ordered(members, edges, key):
    members = tuple(members)
    have = set(members)
    outgoing = {member: set() for member in members}
    indegree = dict.fromkeys(members, 0)
    for first, second in edges:
        if first not in have or second not in have:
            continue
        if first == second:
            raise OrderedRowError("ORDERED_ROW_ORDER_CYCLE")
        if second not in outgoing[first]:
            outgoing[first].add(second)
            indegree[second] += 1
    ready = sorted((member for member in members if indegree[member] == 0), key=key)
    result = []
    while ready:
        member = ready.pop(0)
        result.append(member)
        for child in sorted(outgoing[member], key=key):
            indegree[child] -= 1
            if not indegree[child]:
                ready.append(child)
                ready.sort(key=key)
    if len(result) != len(members):
        raise OrderedRowError("ORDERED_ROW_ORDER_CYCLE")
    return tuple(result)


def compile_ordered_rows(constraints, device_bboxes_nm: Mapping[str, tuple], *,
                         gap_nm: int, opposite_kind_extra_gap_nm: int) -> OrderedRowProgram | None:
    """Compile source rows/order and explicit axes without compact pair boxes."""
    fp = constraints.floorplan_rows
    groups = tuple(constraints.declared_mirror_groups)
    if not fp or not (groups or constraints.order_constraints or constraints.row_gap_minima_nm):
        return None
    have = set(device_bboxes_nm)
    if not have or len(have) > 2048:
        raise OrderedRowError("ORDERED_ROW_DEVICE_INVENTORY_INVALID")
    # A named functional group owns a region, not a physical row centerline.
    # Its explicit member-level vertical order belongs to the existing 2D
    # placement path.  An ordered channel or declared mirror-row construction
    # keeps the strict centerline contract and its existing conflict gate.
    functional_groups = tuple(set(members) for _name, members in constraints.floorplan_groups)
    vertical_conflicts = tuple((a, b) for a, b, rel in constraints.order_constraints
        if rel == 'above' and a in fp and b in fp and fp[a][0] <= fp[b][0])
    if (vertical_conflicts and not groups and not constraints.row_gap_minima_nm
            and all(fp[a][0] == fp[b][0] and any({a, b} <= members for members in functional_groups)
                    for a, b in vertical_conflicts)):
        _ordered(tuple(sorted(have)),
            {(b, a) for a, b, rel in constraints.order_constraints if rel == 'above'},
            lambda name: name)
        return None
    if set(fp) != have:
        raise OrderedRowError("ORDERED_ROW_REQUIRES_EXACT_DEVICE_ROW_COVERAGE")
    if any(type(row) is not int for row, _col in fp.values()):
        raise OrderedRowError("ORDERED_ROW_ORDINAL_INVALID")
    if any(type(group) is not DeclaredMirrorGroup for group in groups):
        raise OrderedRowError("ORDERED_ROW_GROUP_TYPE_INVALID")
    group_members = set()
    declared_pairs = {}
    for group in groups:
        DeclaredMirrorGroup(group.axis, group.pairs, group.on_axis)
        if group.axis != "x":
            raise OrderedRowError("ORDERED_ROW_HORIZONTAL_MIRROR_UNSUPPORTED")
        if not set(group.members) <= have:
            raise OrderedRowError("ORDERED_ROW_GROUP_MEMBER_NOT_MATERIALIZED")
        if group_members.intersection(group.members):
            raise OrderedRowError("ORDERED_ROW_OVERLAPPING_AXIS_GROUPS")
        group_members.update(group.members)
        for pair in group.pairs:
            declared_pairs[frozenset(pair)] = pair
    rows_by_label = {name: fp[name][0] for name in have}
    row_ids = sorted(set(rows_by_label.values()))
    row_sets = {row: {name for name in have if rows_by_label[name] == row} for row in row_ids}
    orders = tuple(constraints.order_constraints)
    if any(len(order) != 3 or order[0] not in have or order[1] not in have
           or order[2] not in ("left_of", "above") for order in orders):
        raise OrderedRowError("ORDERED_ROW_SOURCE_ORDER_INVALID")
    edges = {(a, b) for a, b, rel in orders if rel == "left_of"}
    for a, b, relation in orders:
        if relation == "above" and rows_by_label[a] <= rows_by_label[b]:
            raise OrderedRowError("ORDERED_ROW_VERTICAL_ORDER_CONFLICT")
    # Source columns are relative order too; equal columns remain unconstrained.
    for row, members in row_sets.items():
        for a in members:
            for b in members:
                ac, bc = fp[a][1], fp[b][1]
                if ac is not None and bc is not None and ac < bc:
                    edges.add((a, b))
    key = lambda name: ((fp[name][1] is None), fp[name][1] or 0, name)
    _ordered(tuple(sorted(have)), edges, key)  # reject real source cycles first

    pairs = []
    consumed = set()
    for first, second in constraints.symmetry_pairs:
        pair_key = frozenset((first, second))
        if first not in have or second not in have or first == second:
            raise OrderedRowError("ORDERED_ROW_PAIR_MEMBER_INVALID")
        if first in consumed or second in consumed:
            raise OrderedRowError("ORDERED_ROW_PAIR_OWNERSHIP_AMBIGUOUS")
        if rows_by_label[first] != rows_by_label[second]:
            raise OrderedRowError("ORDERED_ROW_PAIR_CROSSES_ROWS")
        if pair_key in declared_pairs:
            first, second = declared_pairs[pair_key]
        else:
            # @matching does not choose a left member. Preserve explicit source
            # order when it chooses the reverse of normalized matching names.
            trial = _ordered(tuple(sorted(have)), edges, key)
            if trial.index(second) < trial.index(first):
                first, second = second, first
        pairs.append((first, second))
        consumed.update((first, second))
        edges.add((first, second))
    if set(declared_pairs) - {frozenset(pair) for pair in pairs}:
        raise OrderedRowError("ORDERED_ROW_DECLARED_PAIR_AUTHORITY_LOST")

    # Mirror every declared in-row relation, then choose an otherwise unspecified
    # nesting order only on the left half and reflect that order on the right.
    for group in groups:
        mate = {a: b for a, b in group.pairs} | {b: a for a, b in group.pairs}
        mate.update({name: name for name in group.on_axis})
        for a, b in tuple(edges):
            if a in mate and b in mate and rows_by_label[a] == rows_by_label[b]:
                edges.add((mate[b], mate[a]))
        for row, members in row_sets.items():
            left = tuple(a for a, b in group.pairs if a in members)
            right = tuple(b for a, b in group.pairs if a in members)
            on_axis = tuple(name for name in group.on_axis if name in members)
            if len(on_axis) > 1:
                raise OrderedRowError("ORDERED_ROW_MULTIPLE_AXIS_DEVICES_IN_ROW")
            left_order = _ordered(left, edges, key)
            right_order = tuple(mate[name] for name in reversed(left_order))
            mirror_order = left_order + on_axis + right_order
            edges.update(zip(mirror_order, mirror_order[1:]))
    rows = tuple((row, _ordered(tuple(sorted(row_sets[row])), edges, key)) for row in row_ids)
    _ordered(tuple(sorted(have)), edges, key)

    owners = {}
    for index, group in enumerate(groups):
        for member in group.members:
            owners[member] = index
    next_owner = max(1, len(groups))
    for first, second in pairs:
        if first in owners or second in owners:
            if owners.get(first) != owners.get(second):
                raise OrderedRowError("ORDERED_ROW_PAIR_CROSSES_AXIS_GROUPS")
        else:
            owners[first] = owners[second] = next_owner
            next_owner += 1
    for member in sorted(have):
        if member not in owners:
            owners[member] = next_owner
            next_owner += 1
    kinds = constraints.device_kinds
    edge_gaps = tuple((a, b, gap_nm + (opposite_kind_extra_gap_nm
        if kinds.get(a) != kinds.get(b) else 0))
        for _row, members in rows for a, b in zip(members, members[1:]))
    channel_gaps = {(a, b): gap for a, b, gap in constraints.row_gap_minima_nm}
    if any((a, b) not in set(zip(row_ids, row_ids[1:])) for a, b in channel_gaps):
        raise OrderedRowError("ORDERED_ROW_CHANNEL_MUST_JOIN_ADJACENT_ROWS")
    row_gaps = []
    for lower, upper in zip(row_ids, row_ids[1:]):
        lower_kinds = {kinds.get(member) for member in row_sets[lower]}
        upper_kinds = {kinds.get(member) for member in row_sets[upper]}
        mixed = ("nmos" in lower_kinds and "pmos" in upper_kinds
                 or "pmos" in lower_kinds and "nmos" in upper_kinds)
        row_gaps.append((lower, upper, max(channel_gaps.get((lower, upper), 0),
            gap_nm + (opposite_kind_extra_gap_nm if mixed else 0))))
    sizes = []
    for name, box in sorted(device_bboxes_nm.items()):
        if (type(box) is not tuple or len(box) != 4 or any(type(v) is not int for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise OrderedRowError("ORDERED_ROW_NATIVE_BBOX_INVALID")
        sizes.append((name, box[2] - box[0], box[3] - box[1]))
    size_by_name = {name: (width, height) for name, width, height in sizes}
    for group in groups:
        for first, second in group.pairs:
            if size_by_name[first] != size_by_name[second]:
                raise OrderedRowError(f"ORDERED_ROW_PAIR_NATIVE_SIZE_MISMATCH:{first}:{second}")
    return OrderedRowProgram(rows, tuple(pairs), groups, tuple(sorted(owners.items())), orders,
        edge_gaps, tuple(row_gaps), tuple(sizes), _source_signature(constraints))


def make_ordered_row_evidence(
    program,
    axis_coordinates_nm,
    devices,
    *,
    compaction: OrderedRowCompactionEvidence | None = None,
) -> OrderedRowEvidence:
    boxes = []
    for row, members in program.rows:
        bboxes = [tuple(devices[name].bbox) for name in members]
        boxes.append((row, (min(b[0] for b in bboxes), min(b[1] for b in bboxes),
                           max(b[2] for b in bboxes), max(b[3] for b in bboxes))))
    return OrderedRowEvidence(program, tuple(axis_coordinates_nm), tuple(boxes), compaction=compaction)


def _polygon_inventory(polys, axis_twice_nm=None):
    inventory = Counter()
    for item in polys:
        points = tuple((int(v.x) if axis_twice_nm is None else axis_twice_nm - int(v.x),
                        int(v.y)) for v in item.polygon.vertices)
        if points and points[-1] == points[0]:
            points = points[:-1]
        if not points:
            raise OrderedRowError("ORDERED_ROW_EMPTY_POLYGON")
        reverse = tuple(reversed(points))
        key = min(seq[index:] + seq[:index] for seq in (points, reverse)
                  for index in range(len(seq)))
        inventory[(int(item.layer), int(item.datatype), key)] += 1
    return inventory


def verify_ordered_row_placement(placement) -> None:
    """Check source identity, rows, all orders, real masks and terminal reflection."""
    evidence = getattr(placement, "ordered_row_evidence", None)
    if evidence is None:
        return
    if type(evidence) is not OrderedRowEvidence or type(evidence.program) is not OrderedRowProgram:
        raise OrderedRowError("ORDERED_ROW_EVIDENCE_TYPE_INVALID")
    program = evidence.program
    if _source_signature(placement.constraints) != program.source_signature:
        raise OrderedRowError("ORDERED_ROW_SOURCE_BINDING_CHANGED")
    devices = placement.devices
    members = tuple(name for _row, row in program.rows for name in row)
    if len(set(members)) != len(members) or not set(members) <= set(devices):
        raise OrderedRowError("ORDERED_ROW_MEMBER_OWNERSHIP_CHANGED")
    centers2 = {name: (devices[name].bbox[0] + devices[name].bbox[2],
                       devices[name].bbox[1] + devices[name].bbox[3]) for name in members}
    for name, width, height in program.source_sizes_nm:
        box = devices[name].bbox
        if (box[2] - box[0], box[3] - box[1]) != (width, height):
            raise OrderedRowError(f"ORDERED_ROW_NATIVE_SIZE_CHANGED:{name}")
    compacted = evidence.compaction is not None
    if compacted and type(evidence.compaction) is not OrderedRowCompactionEvidence:
        raise OrderedRowError("ORDERED_ROW_COMPACTION_EVIDENCE_INVALID")
    for row, labels in program.rows:
        same_center = len({centers2[name][1] for name in labels}) == 1
        same_bottom = compacted and len({devices[name].bbox[1] for name in labels}) == 1
        if not (same_center or same_bottom):
            raise OrderedRowError(f"ORDERED_ROW_CENTERLINE_CHANGED:{row}")
    for first, second, gap in program.edge_clearances_nm:
        required_gap = 0 if compacted else gap
        if devices[second].bbox[0] - devices[first].bbox[2] < required_gap:
            raise OrderedRowError(f"ORDERED_ROW_CLEARANCE_FAILED:{first}:{second}")
    actual = make_ordered_row_evidence(
        program, evidence.axis_coordinates_nm, devices, compaction=evidence.compaction
    )
    if actual.row_bboxes_nm != evidence.row_bboxes_nm:
        raise OrderedRowError("ORDERED_ROW_ENVELOPE_CHANGED")
    row_boxes = dict(actual.row_bboxes_nm)
    if compacted:
        comp_gaps = evidence.compaction.row_clearances_nm
        expected_adjacent = tuple((lower, upper) for lower, upper, _gap in program.row_clearances_nm)
        if (
            type(comp_gaps) is not tuple
            or tuple((lower, upper) for lower, upper, _gap in comp_gaps) != expected_adjacent
        ):
            raise OrderedRowError("ORDERED_ROW_COMPACTION_EVIDENCE_INVALID")
        explicit_gaps = dict(getattr(placement.constraints, "row_gap_minima_nm", ()) or ())
        for lower, upper, min_gap in comp_gaps:
            if type(min_gap) is not int or min_gap < max(0, int(explicit_gaps.get((lower, upper), 0))):
                raise OrderedRowError("ORDERED_ROW_CHANNEL_CLEARANCE_FAILED")
            if row_boxes[upper][1] - row_boxes[lower][3] < min_gap:
                raise OrderedRowError("ORDERED_ROW_CHANNEL_CLEARANCE_FAILED")
    else:
        for lower, upper, gap in program.row_clearances_nm:
            if row_boxes[upper][1] - row_boxes[lower][3] < gap:
                raise OrderedRowError("ORDERED_ROW_CHANNEL_CLEARANCE_FAILED")
    for first, second, relation in program.source_orders:
        satisfied = (centers2[first][0] < centers2[second][0] if relation == "left_of"
                     else centers2[first][1] > centers2[second][1])
        if not satisfied:
            raise OrderedRowError(f"ORDERED_ROW_SOURCE_ORDER_FAILED:{first}:{second}:{relation}")
    for first, second in program.pairs:
        axis = placement.matching_pair_axes.get(tuple(sorted((first, second))))
        if (axis is None or axis[0] != "x"
                or 4 * axis[1] != centers2[first][0] + centers2[second][0]):
            raise OrderedRowError(f"ORDERED_ROW_PAIR_AXIS_CHANGED:{first}:{second}")
    if (len(evidence.axis_coordinates_nm) != len(program.mirror_groups)
            or any(type(axis) is not int for axis in evidence.axis_coordinates_nm)):
        raise OrderedRowError("ORDERED_ROW_AXIS_EVIDENCE_INVALID")
    if (evidence.axis_x_nm is not None
            and placement.axis_x != evidence.axis_x_nm):
        raise OrderedRowError("ORDERED_ROW_EXPORTED_AXIS_CHANGED")
    for group, axis in zip(program.mirror_groups, evidence.axis_coordinates_nm):
        for name in group.on_axis:
            if centers2[name][0] != 2 * axis:
                raise OrderedRowError(f"ORDERED_ROW_ON_AXIS_FAILED:{name}")
        for first, second in group.pairs:
            a, b = devices[first], devices[second]
            if (centers2[first][0] + centers2[second][0] != 4 * axis
                    or centers2[first][1] != centers2[second][1]):
                raise OrderedRowError(f"ORDERED_ROW_SHARED_AXIS_FAILED:{first}:{second}")
            if not a.polys or not b.polys:
                raise OrderedRowError("ORDERED_ROW_PRIMITIVE_ARTWORK_UNAVAILABLE")
            if _polygon_inventory(a.polys, 2 * axis) != _polygon_inventory(b.polys):
                raise OrderedRowError(f"ORDERED_ROW_MASK_MIRROR_FAILED:{first}:{second}")
            if set(a.terminals) != set(b.terminals):
                raise OrderedRowError("ORDERED_ROW_TERMINAL_ROLE_MISMATCH")
            for role, point in a.terminals.items():
                if (2 * axis - point[0], point[1]) != tuple(b.terminals[role]):
                    raise OrderedRowError(f"ORDERED_ROW_TERMINAL_MIRROR_FAILED:{first}:{second}:{role}")
            if set(a.terminal_accesses) != set(b.terminal_accesses):
                raise OrderedRowError("ORDERED_ROW_TERMINAL_ACCESS_ROLE_MISMATCH")
            for role, points in a.terminal_accesses.items():
                reflected = Counter((2 * axis - x, y) for x, y in points)
                if reflected != Counter(tuple(point) for point in b.terminal_accesses[role]):
                    raise OrderedRowError(f"ORDERED_ROW_TERMINAL_ACCESS_MIRROR_FAILED:{first}:{second}:{role}")


def project_ordered_row_placement(
    base,
    projected,
    *,
    final_placement_certificate=None,
    intent=None,
):
    """Preserve ordered-row proof on rigid move, or verified FinalPlacementCertificate compaction."""
    evidence = getattr(base, "ordered_row_evidence", None)
    if evidence is None:
        return projected
    verify_ordered_row_placement(base)
    deltas = set()
    for _row, members in evidence.program.rows:
        for name in members:
            if name not in projected.devices:
                raise OrderedRowError("ORDERED_ROW_PROJECTION_MEMBER_MISSING")
            old, new = base.devices[name].bbox, projected.devices[name].bbox
            if (old[2] - old[0], old[3] - old[1]) != (new[2] - new[0], new[3] - new[1]):
                raise OrderedRowError("ORDERED_ROW_PROJECTION_SIZE_CHANGED")
            deltas.add((new[0] - old[0], new[1] - old[1]))
    if final_placement_certificate is None:
        if len(deltas) != 1:
            raise OrderedRowError("ORDERED_ROW_NONRIGID_ROUTE_PROJECTION_UNSUPPORTED")
        dx, dy = deltas.pop()
        moved_compaction = (
            OrderedRowCompactionEvidence(
                certified_bbox_nm=(
                    evidence.compaction.certified_bbox_nm[0] + dx,
                    evidence.compaction.certified_bbox_nm[1] + dy,
                    evidence.compaction.certified_bbox_nm[2] + dx,
                    evidence.compaction.certified_bbox_nm[3] + dy,
                ),
                declared_bottom_edge_keepout_nm=evidence.compaction.declared_bottom_edge_keepout_nm,
                edge_right_margin_nm=evidence.compaction.edge_right_margin_nm,
                row_clearances_nm=evidence.compaction.row_clearances_nm,
            )
            if evidence.compaction is not None
            else None
        )
        moved = OrderedRowEvidence(
            evidence.program,
            tuple(axis + dx for axis in evidence.axis_coordinates_nm),
            tuple((row, (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy))
                  for row, box in evidence.row_bboxes_nm),
            (replace(evidence.floorplan,
                core_bbox_nm=tuple(value + (dx if index % 2 == 0 else dy)
                    for index, value in enumerate(evidence.floorplan.core_bbox_nm)),
                floorplan_bbox_nm=tuple(value + (dx if index % 2 == 0 else dy)
                    for index, value in enumerate(evidence.floorplan.floorplan_bbox_nm)))
             if evidence.floorplan is not None else None),
            evidence.cell_identity_sha256,
            evidence.source_file_bindings,
            moved_compaction,
        )
        axes = {pair: (axis, coordinate + (dx if axis == "x" else dy))
                for pair, (axis, coordinate) in base.matching_pair_axes.items()}
        result = replace(projected, axis_x=base.axis_x + dx, ordered_row_evidence=moved,
                         matching_pair_axes=axes)
        verify_ordered_row_placement(result)
        return result

    from clay.layout.place.placement_certificates import FinalPlacementCertificate

    if type(final_placement_certificate) is not FinalPlacementCertificate:
        raise OrderedRowError("ORDERED_ROW_COMPACTION_CERTIFICATE_INVALID")
    if not (final_placement_certificate.geometry_legal and final_placement_certificate.cuts_satisfied):
        raise OrderedRowError("ORDERED_ROW_COMPACTION_CERTIFICATE_INVALID")
    if intent is not None:
        from clay.layout.flow_placement import declared_craft_compaction_requested

        if not declared_craft_compaction_requested(None, intent):
            raise OrderedRowError("ORDERED_ROW_NONRIGID_ROUTE_PROJECTION_UNSUPPORTED")

    certified_members = dict(
        getattr(final_placement_certificate, "materialized_source_member_bboxes", ()) or ()
    )
    members = tuple(name for _row, row in evidence.program.rows for name in row)
    for name in members:
        if name not in certified_members or tuple(projected.devices[name].bbox) != tuple(certified_members[name]):
            raise OrderedRowError("ORDERED_ROW_NONRIGID_ROUTE_PROJECTION_UNSUPPORTED")

    centers2 = {
        name: (
            projected.devices[name].bbox[0] + projected.devices[name].bbox[2],
            projected.devices[name].bbox[1] + projected.devices[name].bbox[3],
        )
        for name in members
    }
    pair_projections = {
        row.pair: row
        for row in (getattr(final_placement_certificate, "matching_pair_axis_projections", ()) or ())
    }
    for first, second in evidence.program.pairs:
        canon = tuple(sorted((first, second)))
        proj_row = pair_projections.get(canon)
        if proj_row is None:
            raise OrderedRowError("ORDERED_ROW_COMPACTION_CERTIFICATE_INCOMPLETE")
        if (
            proj_row.axis != "x"
            or proj_row.source_member_bboxes_nm != (
                tuple(base.devices[canon[0]].bbox),
                tuple(base.devices[canon[1]].bbox),
            )
            or proj_row.final_member_bboxes_nm != (
                tuple(projected.devices[canon[0]].bbox),
                tuple(projected.devices[canon[1]].bbox),
            )
            or proj_row.final_logical_coordinate4_nm != centers2[canon[0]][0] + centers2[canon[1]][0]
        ):
            raise OrderedRowError(f"ORDERED_ROW_PAIR_AXIS_CHANGED:{first}:{second}")
        if centers2[first][1] != centers2[second][1]:
            raise OrderedRowError(f"ORDERED_ROW_SHARED_AXIS_FAILED:{first}:{second}")

    new_axes = []
    for group in evidence.program.mirror_groups:
        if group.on_axis:
            x2 = centers2[group.on_axis[0]][0]
            if x2 % 2 != 0:
                raise OrderedRowError(f"ORDERED_ROW_ON_AXIS_FAILED:{group.on_axis[0]}")
            group_axis = x2 // 2
        else:
            a0, b0 = group.pairs[0]
            x4 = centers2[a0][0] + centers2[b0][0]
            if x4 % 4 != 0:
                raise OrderedRowError(f"ORDERED_ROW_SHARED_AXIS_FAILED:{a0}:{b0}")
            group_axis = x4 // 4
        for name in group.on_axis:
            if centers2[name][0] != 2 * group_axis:
                raise OrderedRowError(f"ORDERED_ROW_ON_AXIS_FAILED:{name}")
        for first, second in group.pairs:
            if (
                centers2[first][0] + centers2[second][0] != 4 * group_axis
                or centers2[first][1] != centers2[second][1]
            ):
                raise OrderedRowError(f"ORDERED_ROW_SHARED_AXIS_FAILED:{first}:{second}")
        new_axes.append(group_axis)

    compacted_ev = make_ordered_row_evidence(evidence.program, tuple(new_axes), projected.devices)
    row_boxes = dict(compacted_ev.row_bboxes_nm)
    explicit_gaps = dict(getattr(base.constraints, "row_gap_minima_nm", ()) or ())
    compacted_row_clearances = []
    for lower, upper, _initial_gap in evidence.program.row_clearances_nm:
        actual_gap = row_boxes[upper][1] - row_boxes[lower][3]
        required_gap = max(0, int(explicit_gaps.get((lower, upper), 0)))
        if actual_gap < required_gap:
            raise OrderedRowError("ORDERED_ROW_CHANNEL_CLEARANCE_FAILED")
        compacted_row_clearances.append((lower, upper, actual_gap))

    keepout_nm = int(getattr(final_placement_certificate, "declared_bottom_edge_keepout_nm", 0) or 0)
    edge_res = getattr(final_placement_certificate, "edge_rail_reservation", None)
    right_margin_nm = int(getattr(edge_res, "right_margin_nm", 0) or 0) if edge_res is not None else 0
    cert_bbox = tuple(getattr(final_placement_certificate, "bbox_nm", ()))
    new_core = (
        min(b[0] for _r, b in compacted_ev.row_bboxes_nm),
        min(b[1] for _r, b in compacted_ev.row_bboxes_nm),
        max(b[2] for _r, b in compacted_ev.row_bboxes_nm),
        max(b[3] for _r, b in compacted_ev.row_bboxes_nm),
    )
    expected_cert_bbox = (
        new_core[0],
        (0 if keepout_nm > 0 else new_core[1]),
        new_core[2] + right_margin_nm,
        new_core[3],
    )
    if cert_bbox != expected_cert_bbox:
        raise OrderedRowError("ORDERED_ROW_FLOORPLAN_SOURCE_OR_PDK_CHANGED")
    if evidence.floorplan is not None and edge_res is not None:
        reserved_demands = {
            (
                str(d.net),
                str(d.edge),
                tuple(int(v) for v in d.layer_key),
                int(d.width_nm),
                int(d.min_space_nm),
                str(d.declaration_ref),
            )
            for d in (getattr(edge_res, "demands", ()) or ())
        }
        if set(evidence.floorplan.edge_bands) != reserved_demands:
            raise OrderedRowError("ORDERED_ROW_FLOORPLAN_SOURCE_OR_PDK_CHANGED")

    compaction = OrderedRowCompactionEvidence(
        certified_bbox_nm=cert_bbox,
        declared_bottom_edge_keepout_nm=keepout_nm,
        edge_right_margin_nm=right_margin_nm,
        row_clearances_nm=tuple(compacted_row_clearances),
    )
    moved_floorplan = (
        replace(evidence.floorplan, core_bbox_nm=new_core, floorplan_bbox_nm=cert_bbox)
        if evidence.floorplan is not None
        else None
    )
    moved = OrderedRowEvidence(
        evidence.program,
        tuple(new_axes),
        compacted_ev.row_bboxes_nm,
        moved_floorplan,
        evidence.cell_identity_sha256,
        evidence.source_file_bindings,
        compaction,
    )
    axes = {}
    for pair, (axis, coordinate) in base.matching_pair_axes.items():
        if axis == "x" and len(pair) == 2 and all(m in projected.devices for m in pair):
            x4 = centers2[pair[0]][0] + centers2[pair[1]][0]
            axes[pair] = (axis, x4 // 4 if x4 % 4 == 0 else x4 / 4.0)
        else:
            axes[pair] = (axis, coordinate)
    new_axis_x = new_axes[0] if len(new_axes) == 1 else projected.axis_x
    result = replace(
        projected,
        axis_x=new_axis_x,
        ordered_row_evidence=moved,
        matching_pair_axes=axes,
    )
    verify_ordered_row_placement(result)
    return result


def verify_ordered_row_route_artwork(placement, geometry, placement_um) -> None:
    """Bind every retained native polygon to the exact final emitted stamp."""
    evidence = getattr(placement, "ordered_row_evidence", None)
    if evidence is None:
        return
    verify_ordered_row_placement(placement)
    from clay.layout.flow_placement import placed_point_nm

    for _row, members in evidence.program.rows:
        for name in members:
            stamp, centre = geometry.get(name), placement_um.get(name)
            if stamp is None or centre is None:
                raise OrderedRowError(f"ORDERED_ROW_ROUTE_STAMP_MISSING:{name}")
            actual = Counter()
            for layer, rings in (getattr(stamp, "polys_by_layer", {}) or {}).items():
                for ring in rings:
                    points = tuple(placed_point_nm(float(x), float(y), centre) for x, y in ring)
                    if points and points[0] == points[-1]:
                        points = points[:-1]
                    if len(points) < 3:
                        raise OrderedRowError("ORDERED_ROW_ROUTE_STAMP_DEGENERATE")
                    reverse = tuple(reversed(points))
                    key = min(seq[index:] + seq[:index] for seq in (points, reverse)
                              for index in range(len(seq)))
                    actual[(int(layer[0]), int(layer[1]), key)] += 1
            if actual != _polygon_inventory(placement.devices[name].polys):
                raise OrderedRowError(f"ORDERED_ROW_ROUTE_ARTWORK_CHANGED:{name}")


def _declared_row_floorplan(placement, intent, tech) -> OrderedRowFloorplan:
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.route_cell import _compile_edge_rail_specs_from_intent

    trunks = tuple(getattr(intent, "trunks", ()) or ())
    if any(trunk.edge and trunk.width_um <= 0 for trunk in trunks):
        raise OrderedRowError("ORDERED_ROW_EDGE_RESERVATION_REQUIRES_DECLARED_WIDTH")
    specs = _compile_edge_rail_specs_from_intent(intent, None, tech)
    bands = []
    depths = {edge: 0 for edge in ("left", "bottom", "right", "top")}
    for spec in specs:
        if spec.edge not in depths or ((spec.edge in ("top", "bottom")) != (spec.actual_axis == "H")):
            raise OrderedRowError("ORDERED_ROW_EDGE_AXIS_CONFLICT")
        space = int(tech.min_space(f"metal{spec.layer_n}"))
        if space < 0:
            raise OrderedRowError("ORDERED_ROW_EDGE_PDK_SPACE_INVALID")
        bands.append((spec.net, spec.edge, tuple(spec.layer_key), spec.width_nm,
                      space, spec.declaration_ref))
        depths[spec.edge] = max(depths[spec.edge], spec.width_nm + space)
    boxes = tuple(box for _row, box in placement.ordered_row_evidence.row_bboxes_nm)
    core = (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))
    grid = pdk_manufacturing_grid_nm(tech)
    if type(grid) is not int or grid <= 0:
        raise OrderedRowError("ORDERED_ROW_EDGE_MANUFACTURING_GRID_INVALID")
    compaction = getattr(placement.ordered_row_evidence, "compaction", None)
    if compaction is not None:
        if type(compaction) is not OrderedRowCompactionEvidence:
            raise OrderedRowError("ORDERED_ROW_COMPACTION_EVIDENCE_INVALID")
        raw = (
            core[0],
            (0 if compaction.declared_bottom_edge_keepout_nm > 0 else core[1]),
            core[2] + compaction.edge_right_margin_nm,
            core[3],
        )
        bbox = (raw[0] // grid * grid, raw[1] // grid * grid,
                -(-raw[2] // grid) * grid, -(-raw[3] // grid) * grid)
        if bbox != compaction.certified_bbox_nm:
            raise OrderedRowError("ORDERED_ROW_FLOORPLAN_SOURCE_OR_PDK_CHANGED")
    else:
        raw = (core[0] - depths["left"], core[1] - depths["bottom"],
               core[2] + depths["right"], core[3] + depths["top"])
        # Reserve complete physical-width bands on the same grid as the rail writer.
        bbox = (raw[0] // grid * grid, raw[1] // grid * grid,
                -(-raw[2] // grid) * grid, -(-raw[3] // grid) * grid)
    return OrderedRowFloorplan(core, bbox, tuple(sorted(bands)), grid)


def attach_ordered_row_floorplan(placement, intent, tech):
    if getattr(placement, "ordered_row_evidence", None) is None:
        return placement
    verify_ordered_row_placement(placement)
    return replace(placement, ordered_row_evidence=replace(placement.ordered_row_evidence,
        floorplan=_declared_row_floorplan(placement, intent, tech)))


def _cell_identity(cell_def) -> str:
    def canonical(value):
        if isinstance(value, Enum):
            return {"enum": type(value).__qualname__, "value": canonical(value.value)}
        if isinstance(value, Fraction):
            return {"fraction": (value.numerator, value.denominator)}
        if is_dataclass(value):
            return {"type": type(value).__qualname__, "fields": {
                item.name: canonical(getattr(value, item.name)) for item in fields(value)
                if item.name != "source_span"}}
        if isinstance(value, dict):
            return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                    if key != "source_span"}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise OrderedRowError(f"ORDERED_ROW_SOURCE_VALUE_UNSUPPORTED:{type(value).__name__}")
    payload = (canonical(cell_def), canonical(getattr(cell_def, "_layout_annotations", {})))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def bind_ordered_row_source(placement, cell_def, source_cir=None):
    """Bind construction to current parsed declarations and available source bytes."""
    evidence = getattr(placement, "ordered_row_evidence", None)
    if evidence is None:
        return placement
    identity = _cell_identity(cell_def)
    source_paths = {str((getattr(declaration, "source_span", None) or {}).get("file") or "")
                    for declaration in getattr(cell_def, "layout_intents", ())}
    source_paths.discard("")
    bindings = []
    for source_path in sorted(source_paths):
        path = Path(source_path).resolve()
        if not path.is_file():
            raise OrderedRowError("ORDERED_ROW_SOURCE_FILE_MISSING")
        from clay.parser import parse_clay_file
        cells = [cell for cell in parse_clay_file(path).cell_defs
                 if cell.local_name == cell_def.local_name]
        if len(cells) != 1 or _cell_identity(cells[0]) != identity:
            raise OrderedRowError("ORDERED_ROW_SOURCE_AST_BYTES_DISAGREE")
        bindings.append((str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
    if source_cir is not None:
        path = Path(source_cir).resolve()
        if not path.is_file():
            raise OrderedRowError("ORDERED_ROW_EXTERNAL_SOURCE_FILE_MISSING")
        bindings.append((str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
    return replace(placement, ordered_row_evidence=replace(evidence,
        cell_identity_sha256=identity, source_file_bindings=tuple(sorted(set(bindings)))))


def verify_ordered_row_source(placement, cell_def) -> None:
    evidence = getattr(placement, "ordered_row_evidence", None)
    if evidence is None:
        return
    verify_ordered_row_placement(placement)
    if not evidence.cell_identity_sha256 or evidence.cell_identity_sha256 != _cell_identity(cell_def):
        raise OrderedRowError("ORDERED_ROW_CURRENT_SOURCE_CHANGED")
    for raw_path, digest in evidence.source_file_bindings:
        path = Path(raw_path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise OrderedRowError("ORDERED_ROW_SOURCE_BYTES_CHANGED")


def verify_ordered_row_floorplan(placement, intent, tech, *, geometry=None,
                                placement_um=None) -> tuple[int, int, int, int]:
    """Replay source/PDK edge-band sizing and, when supplied, actual route stamps."""
    verify_ordered_row_placement(placement)
    evidence = getattr(placement, "ordered_row_evidence", None)
    if evidence is None or type(evidence.floorplan) is not OrderedRowFloorplan:
        raise OrderedRowError("ORDERED_ROW_FLOORPLAN_EVIDENCE_MISSING")
    actual = _declared_row_floorplan(placement, intent, tech)
    if evidence.floorplan != actual:
        raise OrderedRowError("ORDERED_ROW_FLOORPLAN_SOURCE_OR_PDK_CHANGED")
    if (geometry is None) != (placement_um is None):
        raise OrderedRowError("ORDERED_ROW_FLOORPLAN_NATIVE_INPUTS_INCOMPLETE")
    if geometry is not None:
        verify_ordered_row_route_artwork(placement, geometry, placement_um)
    return actual.floorplan_bbox_nm


def ordered_row_artifact_report(placement, cell_def, intent, tech, *, gds_path,
                                top_cell, geometry, placement_um):
    """Persist replayed construction facts and native coverage in reopened GDS.

    Native primitive reflection and complete routed-net symmetry are different
    predicates. This report identifies exactly which source primitive pairs share
    an axis and checks their actual masks and accesses; emitted channel evidence
    is supplied separately by the route checker.
    """
    evidence = getattr(placement, "ordered_row_evidence", None)
    if evidence is None:
        return None
    verify_ordered_row_source(placement, cell_def)
    floorplan = verify_ordered_row_floorplan(placement, intent, tech,
        geometry=geometry, placement_um=placement_um)
    import klayout.db as kdb

    path = Path(gds_path).resolve()
    gds_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    layout = kdb.Layout()
    layout.read(str(path))
    if abs(layout.dbu - 0.001) > 1e-12:
        raise OrderedRowError("ORDERED_ROW_EMITTED_DBU_NOT_NANOMETRES")
    cell = layout.cell(top_cell)
    if cell is None:
        raise OrderedRowError("ORDERED_ROW_EMITTED_TOP_MISSING")
    regions = {}
    devices = []
    centers2 = {}
    for _row, members in evidence.program.rows:
        for name in members:
            device = placement.devices[name]
            centers2[name] = (device.bbox[0] + device.bbox[2], device.bbox[1] + device.bbox[3])
            native = {}
            for polygon in device.polys:
                key = (polygon.layer, polygon.datatype)
                native.setdefault(key, kdb.Region()).insert(kdb.Polygon([
                    kdb.Point(point.x, point.y) for point in polygon.polygon.vertices]))
            if not native:
                raise OrderedRowError(f"ORDERED_ROW_NATIVE_ARTWORK_EMPTY:{name}")
            for key, expected in native.items():
                if key not in regions:
                    regions[key] = kdb.Region(cell.begin_shapes_rec(layout.layer(*key))).merged()
                if not (expected - regions[key]).is_empty():
                    raise OrderedRowError(f"ORDERED_ROW_EMITTED_ARTWORK_MISSING:{name}:{key}")
            inventory = sorted(_polygon_inventory(device.polys).items())
            devices.append({"name": name, "kind": device.kind, "bbox_nm": device.bbox,
                "native_polygon_count": len(device.polys),
                "native_polygon_sha256": hashlib.sha256(json.dumps(inventory,
                    separators=(",", ":")).encode("utf-8")).hexdigest(),
                "native_layers": sorted(native), "terminals_nm": device.terminals,
                "terminal_accesses_nm": device.terminal_accesses,
                "reopened_gds_native_polygon_coverage": True})
    pairs = []
    for first, second in evidence.program.pairs:
        a, b = placement.devices[first], placement.devices[second]
        axis2 = (centers2[first][0] + centers2[second][0]) / 2
        terminal_match = (set(a.terminals) == set(b.terminals) and all(
            (axis2 - point[0], point[1]) == tuple(b.terminals[role])
            for role, point in a.terminals.items()))
        access_match = (set(a.terminal_accesses) == set(b.terminal_accesses) and all(
            Counter((axis2 - x, y) for x, y in points) == Counter(tuple(point)
                for point in b.terminal_accesses[role]) for role, points in a.terminal_accesses.items()))
        pairs.append({"pair": (first, second), "axis_x_nm": axis2 / 2,
            "same_row_center": centers2[first][1] == centers2[second][1],
            "all_native_polygon_layers_reflect": _polygon_inventory(a.polys, axis2) == _polygon_inventory(b.polys),
            "all_terminal_roles_reflect": terminal_match,
            "all_terminal_accesses_reflect": access_match})
    orders = [{"first": first, "second": second, "relation": relation,
        "satisfied": centers2[first][0] < centers2[second][0] if relation == "left_of"
                     else centers2[first][1] > centers2[second][1]}
        for first, second, relation in evidence.program.source_orders]
    if hashlib.sha256(path.read_bytes()).hexdigest() != gds_sha:
        raise OrderedRowError("ORDERED_ROW_EMITTED_GDS_CHANGED_DURING_CHECK")
    return {
        "schema": "clay.ordered-row-emitted-evidence/v1", "status": "pass",
        "scope": "Source-owned physical rows, primitive masks/accesses, and their coverage in the reopened exact GDS; not a DRC/LVS or routed-net mirror claim.",
        "gds": {"path": str(path), "sha256": gds_sha, "top_cell": top_cell, "dbu_um": layout.dbu},
        "source_cell_sha256": evidence.cell_identity_sha256,
        "source_file_bindings": evidence.source_file_bindings,
        "ordered_row_evidence_sha256": evidence.sha256,
        "program": asdict(evidence.program),
        "declared_axes_nm": evidence.axis_coordinates_nm,
        "floorplan_bbox_nm": floorplan, "row_bboxes_nm": evidence.row_bboxes_nm,
        "devices": devices, "measured_pairs": pairs, "source_orders": orders,
        "on_axis_devices": [{"name": name, "axis_x_nm": axis,
                             "actual_center_x_nm": centers2[name][0] / 2}
            for group, axis in zip(evidence.program.mirror_groups, evidence.axis_coordinates_nm)
            for name in group.on_axis],
        "all_source_orders_satisfied": all(row["satisfied"] for row in orders),
        "all_native_artwork_covered_in_reopened_gds": True,
    }
