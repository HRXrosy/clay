"""Source-bound geometry invariants for matching across distinct native rows.

This module does not by itself certify a placement.  A caller must bind its
solver-produced axes and both solver owner poses to the actual final row stamps;
only then can it use these independently replayed source/geometry invariants.
In particular, none of the helpers derives authority from final device bboxes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("cross-row matching " + message)


def _sha(value: str, label: str) -> None:
    _require(type(value) is str and len(value) == 64
             and all(char in "0123456789abcdef" for char in value),
             label + " must be a canonical sha256")


def _channel_offset2_nm(
    channel_bbox_nm: tuple[int, int, int, int], row_bbox_nm: tuple[int, int, int, int],
) -> tuple[int, int]:
    """Offset of channel centre from the exact, possibly half-nm row centre."""
    for label, box in (("channel", channel_bbox_nm), ("row", row_bbox_nm)):
        _require(type(box) is tuple and len(box) == 4 and all(type(v) is int for v in box)
                 and box[2] > box[0] and box[3] > box[1], label + " bbox is malformed")
    _require(row_bbox_nm[0] <= channel_bbox_nm[0] and row_bbox_nm[1] <= channel_bbox_nm[1]
             and channel_bbox_nm[2] <= row_bbox_nm[2] and channel_bbox_nm[3] <= row_bbox_nm[3],
             "channel bbox lies outside its source row")
    return (channel_bbox_nm[0] + channel_bbox_nm[2] - row_bbox_nm[0] - row_bbox_nm[2],
            channel_bbox_nm[1] + channel_bbox_nm[3] - row_bbox_nm[1] - row_bbox_nm[3])


@dataclass(frozen=True, order=True)
class NativeRowChannel:
    occurrence_id: str
    center2_nm: tuple[int, int]
    size_nm: tuple[int, int]
    orientation: str

    def __post_init__(self) -> None:
        _require(type(self.occurrence_id) is str and bool(self.occurrence_id),
                 "occurrence identity is missing")
        _require(type(self.center2_nm) is tuple and len(self.center2_nm) == 2
                 and all(type(x) is int for x in self.center2_nm),
                 "occurrence center must be exact twice-nanometres")
        _require(type(self.size_nm) is tuple and len(self.size_nm) == 2
                 and all(type(x) is int and x > 0 for x in self.size_nm),
                 "occurrence channel size is invalid")
        _require(self.orientation in ("R0", "R90", "R180", "R270", "MR0", "MR90", "MR180", "MR270"),
                 "occurrence orientation is invalid")


@dataclass(frozen=True, order=True)
class NativeRowMatchingEndpoint:
    member: str
    row_owner: str
    model: str
    # RowMemberSpec supplies per-channel W/L and the full occurrence count.
    source_geometry_nm: tuple[int, int, int]
    role_nets: tuple[str, str, str, str]
    channels: tuple[NativeRowChannel, ...]

    def __post_init__(self) -> None:
        _require(all(type(x) is str and x for x in (self.member, self.row_owner, self.model)),
                 "source member, row owner and model must be named")
        _require(type(self.source_geometry_nm) is tuple and len(self.source_geometry_nm) == 3
                 and all(type(x) is int and x > 0 for x in self.source_geometry_nm),
                 "source member dimensions are invalid")
        _require(type(self.role_nets) is tuple and len(self.role_nets) == 4
                 and all(type(x) is str and x for x in self.role_nets),
                 "source d/g/s/b identities are incomplete")
        _require(type(self.channels) is tuple and self.channels
                 and all(type(x) is NativeRowChannel for x in self.channels),
                 "native channel inventory is malformed")
        for channel in self.channels:
            channel.__post_init__()
        _require(self.channels == tuple(sorted(self.channels))
                 and len({x.occurrence_id for x in self.channels}) == len(self.channels)
                 and len(self.channels) == self.source_geometry_nm[2],
                 "native occurrence inventory differs from source")


@dataclass(frozen=True, order=True)
class NativeCrossRowPair:
    endpoints: tuple[NativeRowMatchingEndpoint, NativeRowMatchingEndpoint]
    declaration_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(type(self.endpoints) is tuple and len(self.endpoints) == 2
                 and all(type(x) is NativeRowMatchingEndpoint for x in self.endpoints),
                 "matching endpoints are malformed")
        left, right = self.endpoints
        left.__post_init__(); right.__post_init__()
        _require(left.member < right.member and left.row_owner != right.row_owner,
                 "matching pair must name two canonical, distinct row owners")
        _require(left.model == right.model and left.source_geometry_nm == right.source_geometry_nm,
                 "matching source model or dimensions differ")
        _require(sorted(x.size_nm for x in left.channels) == sorted(x.size_nm for x in right.channels),
                 "matching native channel dimensions differ")
        _require(type(self.declaration_refs) is tuple and self.declaration_refs
                 and self.declaration_refs == tuple(sorted(set(self.declaration_refs)))
                 and all(type(x) is str and x for x in self.declaration_refs),
                 "matching source declarations are incomplete")

    @property
    def members(self) -> tuple[str, str]:
        return tuple(x.member for x in self.endpoints)


@dataclass(frozen=True)
class NativeCrossRowInventory:
    source_sha256: str
    source_projection_sha256: str
    program_sha256: str
    materialization_sha256: str
    graph_sha256: str
    pdk_sha256: str
    row_geometry_sha256: tuple[tuple[str, str], ...]
    pairs: tuple[NativeCrossRowPair, ...]
    row_source_members: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        for label in ("source_sha256", "source_projection_sha256", "program_sha256",
                      "materialization_sha256", "graph_sha256", "pdk_sha256"):
            _sha(getattr(self, label), label)
        _require(type(self.row_geometry_sha256) is tuple, "row geometry inventory must be an exact tuple")
        for row in self.row_geometry_sha256:
            _require(type(row) is tuple and len(row) == 2 and type(row[0]) is str and row[0],
                     "row geometry identity is malformed")
            _sha(row[1], "row geometry")
        _require(self.row_geometry_sha256 == tuple(sorted(self.row_geometry_sha256))
                 and len({row[0] for row in self.row_geometry_sha256}) == len(self.row_geometry_sha256),
                 "row geometry identities must be canonical")
        _require(type(self.pairs) is tuple and all(type(p) is NativeCrossRowPair for p in self.pairs),
                 "source pair inventory is malformed")
        for pair in self.pairs:
            pair.__post_init__()
        keys = tuple(pair.members for pair in self.pairs)
        _require(keys == tuple(sorted(set(keys))), "source pairs must be canonical and unique")
        endpoints = [name for pair in self.pairs for name in pair.members]
        _require(len(set(endpoints)) == len(endpoints),
                 "one source member cannot own multiple matching relations")
        row_names = {row[0] for row in self.row_geometry_sha256}
        _require(all(endpoint.row_owner in row_names for pair in self.pairs for endpoint in pair.endpoints),
                 "matching endpoint names a foreign row owner")
        _require(type(self.row_source_members) is tuple
                 and all(type(row) is tuple and len(row) == 2 and type(row[0]) is str and row[0]
                         and type(row[1]) is tuple and row[1] and row[1] == tuple(sorted(set(row[1])))
                         and all(type(name) is str and name for name in row[1])
                         for row in self.row_source_members)
                 and self.row_source_members == tuple(sorted(self.row_source_members)),
                 "source row member partition is malformed")
        rows = dict(self.row_source_members)
        _require(len(rows) == len(self.row_source_members) and set(rows) == row_names,
                 "source row member partition differs from geometry")
        all_members = [name for names in rows.values() for name in names]
        _require(len(set(all_members)) == len(all_members), "source member has multiple row owners")
        _require(all(endpoint.member in rows[endpoint.row_owner]
                     for pair in self.pairs for endpoint in pair.endpoints),
                 "matching endpoint differs from complete source row partition")

    @property
    def sha256(self) -> str:
        self.__post_init__()
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def native_cross_row_inventory(cell, program, materialization, tech, *, source_cir):
    """Independently rebuild native endpoints from current source and PDK rows.

    This source boundary requires the existing captured row projection, the
    original source-circuit identity and the real terminal/geometry replay. It
    does not accept an externally reconstructed source or occurrence inventory.
    """
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.joint_occurrence_center_basis import replay_joint_channel_centers
    from clay.layout.joint_row_geometry import _region, joint_d4_orientation
    from clay.layout.joint_terminal_inventory import _native_frame
    from clay.layout.physical_macro_graph import physical_macro_graph_from_intent
    from clay.layout.place.row_replay_session import (
        require_current_row_materialization, current_row_source_program,
    )
    from clay.layout.place.row_source_projection import (
        require_actual_row_source_topology,
        require_row_source_projection,
    )
    from clay.pdk.sky130_native_geometry import native_geometry_spec

    _require(getattr(tech, "pdk_id", None) == "sky130", "only the bound native SKY130 profile is supported")
    require_current_row_materialization(materialization, program=program, tech_id=tech.pdk_id)
    projection = getattr(materialization, "source_projection", None)
    require_row_source_projection(projection, program=program)
    current_row_source_program(projection, cell, tech, program=program, materialization=materialization)
    require_actual_row_source_topology(projection, source_cir)
    source_intent = layout_intent_from_cell_def(cell)
    graph = physical_macro_graph_from_intent(cell, source_intent, device_records={})
    entries = {entry.entry_id: entry for entry in program.entries}
    realizations = {row.entry_id: row for row in materialization.realizations}
    _require(set(entries) == set(realizations), "row source/materialization inventory differs")
    owner_of = {}
    for entry in entries.values():
        for member in entry.consumed_devices:
            _require(member not in owner_of, "source member belongs to multiple row owners")
            owner_of[member] = entry.entry_id
    members = {item.device: item for item in program.members}
    devices = {item.instance_name: item for item in cell.devices}
    _require(len(devices) == len(cell.devices), "source device identity is duplicated")
    relations = tuple(m for m in graph.macros if m.kind == "matching_relation")
    requested = []
    for relation in relations:
        pair = tuple(sorted(relation.members))
        _require(len(pair) == 2 and all(m in owner_of for m in pair),
                 "matching relation lacks complete row ownership")
        if owner_of[pair[0]] == owner_of[pair[1]]:
            continue
        row_owners = json.loads(dict(relation.attributes)["row_owners"])
        expected = tuple("matched:" + entries[owner_of[m]].group for m in relation.members)
        _require(tuple(row_owners) == expected, "matching graph source owner projection differs")
        requested.append((pair, tuple(sorted(relation.declaration_refs))))
    needed = {owner_of[name] for pair, _ in requested for name in pair}
    endpoints = {}
    geometry_hashes = []
    pdk_hashes = set()
    for owner in sorted(needed):
        entry, realization = entries[owner], realizations[owner]
        binding = getattr(realization.geometry, "joint_row_binding", None)
        _require(binding is not None, "source row lacks native joint geometry")
        centers = replay_joint_channel_centers(binding, tech=tech, members=members)
        records = {row.occurrence_id: row for row in centers.records}
        row_bbox = realization.local_bbox_nm
        by_member = {}
        for occurrence in binding.occurrences:
            _require(occurrence.owner in entry.consumed_devices,
                     "native occurrence names a foreign source owner")
            raw = records[occurrence.occurrence_id]
            region = _region(raw.channel_geometry_nm).transformed(
                _native_frame(occurrence.master_to_local)).merged()
            box = region.bbox()
            _require(region.size() == 1 and box.width() > 0 and box.height() > 0
                     and region.area() == box.width() * box.height(),
                     "source channel must be an exact native rectangle")
            by_member.setdefault(occurrence.owner, []).append(NativeRowChannel(
                occurrence.occurrence_id,
                _channel_offset2_nm((box.left, box.bottom, box.right, box.top), row_bbox),
                (box.width(), box.height()), joint_d4_orientation(occurrence.master_to_local)))
        _require(set(by_member) == set(entry.consumed_devices),
                 "native channel source owner inventory is incomplete")
        for member, channels in by_member.items():
            spec, row_member = native_geometry_spec(devices[member]), members[member]
            _require(spec["family"] == "mos" and spec["occurrences"] == row_member.fingers,
                     "native source occurrence/model identity differs")
            pins = devices[member].pins
            nets = (row_member.drain_net, row_member.gate_net, row_member.source_net, row_member.body_net)
            _require(nets == tuple(pins.get(k, "") for k in ("d", "g", "s", "b")),
                     "row terminal roles differ from native source")
            endpoints[member] = NativeRowMatchingEndpoint(
                member, owner, spec["model"],
                (row_member.width_nm, row_member.length_nm, row_member.fingers),
                nets, tuple(sorted(channels)))
        geometry_hashes.append((owner, binding.sha256))
        pdk_hashes.add(binding.pdk_provenance_sha256)
    if not requested:
        return None
    _require(len(pdk_hashes) == 1, "native row PDK identities differ")
    pairs = tuple(sorted(NativeCrossRowPair(
        tuple(endpoints[name] for name in pair), refs) for pair, refs in requested))
    return NativeCrossRowInventory(
        projection.source_sha256, projection.sha256, program.sha256,
        materialization.certificate.sha256, graph.sha256, next(iter(pdk_hashes)),
        tuple(sorted(geometry_hashes)), pairs,
        tuple((owner, entries[owner].consumed_devices) for owner in sorted(needed)))


def native_cross_row_inventory_from_projection(program, materialization, tech):
    """Replay from the already-captured *canonical* native source authority.

    The captured native witness retains the original canonical cell and circuit
    operand; the authoring cell is not a substitute, since their source bytes
    can differ. No new cache or deserialized source authority is introduced.
    """
    from clay.layout.native_diffusion_source import NativePCellWitness
    from clay.layout.place.row_source_projection import require_row_source_projection

    projection = getattr(materialization, "source_projection", None)
    require_row_source_projection(projection, program=program)
    witness = projection.native_witness
    _require(type(witness) is NativePCellWitness, "captured native PCell witness is missing")
    return native_cross_row_inventory(witness.cell, program, materialization, tech,
                                      source_cir=projection.source_cir)


def verify_native_cross_row_geometry(
    inventory: NativeCrossRowInventory,
    row_centres2_nm: Mapping[str, tuple[int, int]],
    pair_axes4_nm: Mapping[tuple[str, str], tuple[str, int]],
) -> bool:
    """Check source-bound actual channel reflection about supplied solver axes.

    Row centres and axes must be obtained and bound by a caller's independently
    replayed solver/row certificate. This function never creates an axis from
    final geometry, and it does not replace whole-mask or terminal replay.
    """
    _require(type(inventory) is NativeCrossRowInventory, "inventory type is foreign")
    inventory.__post_init__()
    _require(isinstance(row_centres2_nm, Mapping) and isinstance(pair_axes4_nm, Mapping),
             "row centres and solver axes must be mappings")
    required_rows = {row[0] for row in inventory.row_geometry_sha256}
    _require(set(row_centres2_nm) == required_rows, "solver row owner inventory differs")
    _require(set(pair_axes4_nm) == {pair.members for pair in inventory.pairs},
             "solver pair-axis inventory differs")
    for row, center in row_centres2_nm.items():
        _require(type(center) is tuple and len(center) == 2 and all(type(v) is int for v in center),
                 "solver row center must be exact twice-nanometres")
    for pair in inventory.pairs:
        raw_axis = pair_axes4_nm[pair.members]
        _require(type(raw_axis) is tuple and len(raw_axis) == 2
                 and raw_axis[0] in ("x", "y") and type(raw_axis[1]) is int,
                 "solver matching axis is malformed")
        axis, coordinate4 = raw_axis
        ai, pi = (0, 1) if axis == "x" else (1, 0)
        left, right = pair.endpoints
        reflected = Counter()
        actual = Counter()
        for channel in left.channels:
            p = tuple(row_centres2_nm[left.row_owner][i] + channel.center2_nm[i] for i in (0, 1))
            reflected[(coordinate4 - p[ai], p[pi], channel.size_nm, channel.orientation)] += 1
        for channel in right.channels:
            q = tuple(row_centres2_nm[right.row_owner][i] + channel.center2_nm[i] for i in (0, 1))
            actual[(q[ai], q[pi], channel.size_nm, channel.orientation)] += 1
        _require(reflected == actual,
                 "actual source occurrence does not replay supplied solver axis")
    return True


def transport_native_cross_row_axes(inventory, source_centres2_nm, source_axes4_nm,
                                    final_centres2_nm):
    """Transport proven axes through independently verified rigid row motion.

    Both source and final centres must be supplied from replayed owner
    certificates; this routine never derives either pose from a member bbox.
    Each pair retains its own axis.  A second geometry replay rejects any
    perpendicular motion that breaks any source occurrence reflection.
    """
    verify_native_cross_row_geometry(inventory, source_centres2_nm, source_axes4_nm)
    _require(isinstance(final_centres2_nm, Mapping)
             and set(final_centres2_nm) == set(source_centres2_nm),
             "final row owner inventory differs")
    for center in final_centres2_nm.values():
        _require(type(center) is tuple and len(center) == 2
                 and all(type(v) is int for v in center),
                 "final row center must be exact twice-nanometres")
    result = {}
    for pair in inventory.pairs:
        axis, source_coordinate4 = source_axes4_nm[pair.members]
        index = 0 if axis == "x" else 1
        deltas2 = tuple(final_centres2_nm[e.row_owner][index]
                        - source_centres2_nm[e.row_owner][index]
                        for e in pair.endpoints)
        result[pair.members] = (axis, source_coordinate4 + sum(deltas2))
    verify_native_cross_row_geometry(inventory, final_centres2_nm, result)
    return result


def native_cross_row_center_alignments(inventory, source_centres2_nm, source_axes4_nm,
                                       row_solver_owners):
    """Lower the proven equal-centre profile to existing centre constraints.

    This deliberately supports only x-reflected source rows with identical
    perpendicular centre and local occurrence phase.  Other profiles require
    an explicit offset-aware compaction producer and are rejected here.
    Solver-owner mapping must be independently replayed by the caller.
    """
    verify_native_cross_row_geometry(inventory, source_centres2_nm, source_axes4_nm)
    _require(isinstance(row_solver_owners, Mapping)
             and set(row_solver_owners) == set(source_centres2_nm)
             and all(type(v) is str and v for v in row_solver_owners.values())
             and len(set(row_solver_owners.values())) == len(row_solver_owners),
             "compaction row solver ownership differs")
    groups = set()
    for pair in inventory.pairs:
        _require(source_axes4_nm[pair.members][0] == "x",
                 "compaction only supports x-reflected native rows")
        left, right = pair.endpoints
        _require(source_centres2_nm[left.row_owner][1] == source_centres2_nm[right.row_owner][1],
                 "compaction row perpendicular centers differ")
        y_left = {c.center2_nm[1] for c in left.channels}
        y_right = {c.center2_nm[1] for c in right.channels}
        _require(len(y_left) == 1 and y_left == y_right,
                 "compaction row perpendicular occurrence phase differs")
        first, second = sorted((row_solver_owners[left.row_owner], row_solver_owners[right.row_owner]))
        groups.add((first, second, "vertical"))
    return tuple(sorted(groups))


def native_cross_row_solver_owners(inventory, program, live_master_devices,
                                   master_members, block_stamps, source_device_names):
    """Bind each row to one solver master through canonical source ownership.

    Composite stamp aliases are not source devices.  They are checked
    separately against the exact native row stamp and cannot hide extra source
    members or a second physical stamp.
    """
    _require(type(inventory) is NativeCrossRowInventory, "solver source inventory is foreign")
    inventory.__post_init__()
    _require(all(isinstance(item, Mapping) for item in
                 (live_master_devices, master_members, block_stamps)),
             "solver ownership inputs are malformed")
    _require(type(source_device_names) is tuple and source_device_names == tuple(sorted(set(source_device_names)))
             and all(type(name) is str and name for name in source_device_names),
             "solver source device inventory is malformed")
    source_names = set(source_device_names)
    entries = {entry.entry_id: entry for entry in program.entries}
    result = {}
    for row, members in inventory.row_source_members:
        entry = entries.get(row)
        _require(entry is not None and entry.consumed_devices == members and set(members) <= source_names,
                 "solver row source ownership differs")
        candidates = [master for master, owned in live_master_devices.items()
                      if set(owned) & source_names == set(members)]
        _require(len(candidates) == 1, "solver master source ownership is ambiguous")
        master = candidates[0]
        physical = master_members.get(master)
        _require(type(physical) is tuple and physical
                 and all(name in block_stamps for name in physical),
                 "solver master physical ownership is incomplete")
        stamps = [stamp for block in physical for stamp in block_stamps[block]]
        _require(stamps == [entry.group], "solver row does not have one exact native stamp")
        result[row] = master
    _require(len(set(result.values())) == len(result), "solver row masters are not distinct")
    return result


def verify_native_cross_row_grid_translation(source_centres2_nm, final_centres2_nm,
                                            manufacturing_grid_nm):
    """Keep each already verified row's manufacturing-grid phase unchanged.

    The caller must separately prove source artwork and grid validity; this
    tests the exact signed translation, not a rounded centroid or an inferred
    final origin.
    """
    _require(type(manufacturing_grid_nm) is int and manufacturing_grid_nm > 0,
             "manufacturing grid must be an exact positive integer")
    _require(isinstance(source_centres2_nm, Mapping) and isinstance(final_centres2_nm, Mapping)
             and bool(source_centres2_nm) and set(source_centres2_nm) == set(final_centres2_nm),
             "grid row owner inventory differs")
    for row, initial in source_centres2_nm.items():
        final = final_centres2_nm[row]
        _require(type(initial) is tuple and type(final) is tuple
                 and len(initial) == len(final) == 2
                 and all(type(v) is int for v in initial + final),
                 "grid row centers must be exact twice-nanometres")
        _require(all((final[i] - initial[i]) % (2 * manufacturing_grid_nm) == 0
                     for i in (0, 1)), "row translation changes manufacturing-grid phase")
    return True


@dataclass(frozen=True)
class NativeCrossRowFinalProof:
    """Signed two-owner poses carried to final placement, not bbox substitutes."""
    cp_proof: object
    final_centres2_nm: tuple[tuple[str, tuple[int, int]], ...]
    final_axes4_nm: tuple[tuple[tuple[str, str], tuple[str, int]], ...]
    manufacturing_grid_nm: int

    def __post_init__(self):
        _require(type(self.cp_proof) is NativeCrossRowCpProof, "final CP proof is foreign")
        self.cp_proof.__post_init__()
        _require(type(self.final_centres2_nm) is tuple
                 and self.final_centres2_nm == tuple(sorted(self.final_centres2_nm))
                 and len(dict(self.final_centres2_nm)) == len(self.final_centres2_nm),
                 "final owner poses are not canonical")
        _require(type(self.final_axes4_nm) is tuple
                 and self.final_axes4_nm == tuple(sorted(self.final_axes4_nm))
                 and len(dict(self.final_axes4_nm)) == len(self.final_axes4_nm),
                 "final pair axes are not canonical")
        _require(type(self.manufacturing_grid_nm) is int and self.manufacturing_grid_nm > 0,
                 "final manufacturing grid is invalid")

    @property
    def sha256(self):
        self.__post_init__()
        data = {"cp_proof_sha256": self.cp_proof.sha256,
                "final_centres2_nm": self.final_centres2_nm,
                "final_axes4_nm": self.final_axes4_nm,
                "manufacturing_grid_nm": self.manufacturing_grid_nm}
        return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def native_cross_row_final_proof(inventory, cp_proof, axis_directions, final_centres2_nm,
                                 manufacturing_grid_nm, *, row_program=None, row_materialization=None, tech=None):
    """Build only after the caller replays final owner poses and artwork."""
    initial, axes = replay_native_cross_row_cp_proof(inventory, cp_proof, axis_directions,
        row_program=row_program, row_materialization=row_materialization, tech=tech)
    verify_native_cross_row_grid_translation(initial, final_centres2_nm, manufacturing_grid_nm)
    final_axes = transport_native_cross_row_axes(inventory, initial, axes, final_centres2_nm)
    return NativeCrossRowFinalProof(cp_proof, tuple(sorted(final_centres2_nm.items())),
                                    tuple(sorted(final_axes.items())), manufacturing_grid_nm)


def replay_native_cross_row_final_proof(inventory, proof, axis_directions,
                                        actual_final_centres2_nm, manufacturing_grid_nm, *,
                                        row_program=None, row_materialization=None, tech=None):
    """Replay against caller-owned final row/artwork and current PDK grid."""
    _require(type(proof) is NativeCrossRowFinalProof, "final row proof is foreign")
    proof.__post_init__()
    _require(proof.manufacturing_grid_nm == manufacturing_grid_nm,
             "final manufacturing grid changed")
    expected = native_cross_row_final_proof(inventory, proof.cp_proof, axis_directions,
        actual_final_centres2_nm, manufacturing_grid_nm, row_program=row_program,
        row_materialization=row_materialization, tech=tech)
    _require(proof == expected, "final owner poses or axes differ from replay")
    return dict(proof.final_axes4_nm)


def _exact_stamp_position_nm(position):
    """Use the existing placed-coordinate dust policy without rounding real offsets."""
    import math
    from clay.layout.placed_geometry import _BINARY_NOISE_DECIMALS_NM, placed_coord_nm
    _require(type(position) is tuple and len(position) == 2
             and all(type(v) in (int, float) and math.isfinite(v) for v in position),
             "stamp position is malformed")
    stable = tuple(round(float(value) * 1000.0, _BINARY_NOISE_DECIMALS_NM) for value in position)
    _require(all(value.is_integer() for value in stable),
             "stamp position is not exact integer nanometres")
    return tuple(placed_coord_nm(0.0, value) for value in position)


def _require_unique_native_row_stamp(geometry, group, members):
    """Use the shared artwork owner convention, including standalone names."""
    from clay.layout.owned_artwork import _members
    _require(isinstance(geometry, Mapping), "stamp geometry inventory is malformed")
    occurrences = []
    for name, stamp in geometry.items():
        raw = getattr(stamp, "composite_members", ())
        _require(type(raw) is tuple, "stamp member inventory is malformed")
        owners = _members(raw if raw else (name,))
        if set(members).intersection(owners):
            occurrences.append(name)
    _require(occurrences == [group], "stamp source member is multiply owned")


def native_cross_row_stamp_centres(inventory, program, materialization, geometry, placement_um):
    """Replay current complete native stamps and recover their exact row poses.

    The source inventory must already have been reconstructed from current
    source/PDK operands.  The complete joint artwork is replayed before a
    native-origin translation is applied to the source row's exact extent.
    """
    from clay.layout.joint_row_stamp import JointRowStampBinding, joint_stamp_owner_records

    _require(type(inventory) is NativeCrossRowInventory, "stamp source inventory is foreign")
    inventory.__post_init__()
    _require(isinstance(geometry, Mapping) and isinstance(placement_um, Mapping),
             "stamp geometry and placement are required")
    entries = {entry.entry_id: entry for entry in program.entries}
    rows = {row.entry_id: row for row in materialization.realizations}
    geometry_hashes = dict(inventory.row_geometry_sha256)
    centres = {}
    for owner, members in inventory.row_source_members:
        entry, row = entries.get(owner), rows.get(owner)
        _require(entry is not None and row is not None
                 and entry.consumed_devices == members and row.consumed_devices == members,
                 "stamp row source ownership differs")
        stamp = geometry.get(entry.group)
        binding = getattr(stamp, "joint_row_stamp_binding", None)
        _require(type(binding) is JointRowStampBinding, "complete native row stamp is missing")
        binding.__post_init__()
        _require(binding.source.sha256 == geometry_hashes[owner]
                 and binding.consumed_devices == members
                 and getattr(stamp, "composite_members", None) == members,
                 "stamp source or member ownership differs")
        _require_unique_native_row_stamp(geometry, entry.group, members)
        position = placement_um.get(entry.group)
        exact = _exact_stamp_position_nm(position)
        joint_stamp_owner_records(stamp, position, members)
        bbox = row.local_bbox_nm
        _require(type(bbox) is tuple and len(bbox) == 4 and all(type(v) is int for v in bbox),
                 "stamp source row bbox is malformed")
        dx, dy = (int(exact[i]) - binding.native_origin_nm[i] for i in (0, 1))
        centres[owner] = (bbox[0] + bbox[2] + 2 * dx, bbox[1] + bbox[3] + 2 * dy)
    return centres


@dataclass(frozen=True, order=True)
class NativeCrossRowCpOwner:
    row_owner: str
    solver_owner: str
    x_index: int
    y_index: int
    padded_width_um: int
    padded_height_um: int

    def __post_init__(self) -> None:
        _require(all(type(v) is str and v for v in (self.row_owner, self.solver_owner)),
                 "CP row owner identity is malformed")
        _require(all(type(v) is int and v >= 0 for v in (self.x_index, self.y_index))
                 and self.x_index != self.y_index, "CP owner variables are malformed")
        _require(all(type(v) is int and v > 0 for v in (self.padded_width_um, self.padded_height_um)),
                 "CP padded group dimensions are malformed")


@dataclass(frozen=True, order=True)
class NativeCrossRowCpPair:
    members: tuple[str, str]
    axis: str
    axis_index: int

    def __post_init__(self) -> None:
        _require(type(self.members) is tuple and len(self.members) == 2
                 and all(type(v) is str and v for v in self.members)
                 and self.members[0] < self.members[1], "CP matching pair is malformed")
        _require(self.axis in ("x", "y") and type(self.axis_index) is int and self.axis_index >= 0,
                 "CP matching axis is malformed")


@dataclass(frozen=True)
class NativeCrossRowCpBinding:
    """Exact source projection and model locations; not a placement certificate."""
    inventory_sha256: str
    owners: tuple[NativeCrossRowCpOwner, ...]
    pairs: tuple[NativeCrossRowCpPair, ...]
    constraint_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _sha(self.inventory_sha256, "CP source inventory")
        _require(type(self.owners) is tuple and self.owners == tuple(sorted(self.owners))
                 and all(type(row) is NativeCrossRowCpOwner for row in self.owners),
                 "CP owners are not canonical")
        for row in self.owners: row.__post_init__()
        _require(len({r.row_owner for r in self.owners}) == len(self.owners)
                 and len({r.solver_owner for r in self.owners}) == len(self.owners),
                 "CP row/source owner partition is not unique")
        owner_indices = [index for row in self.owners for index in (row.x_index, row.y_index)]
        _require(len(set(owner_indices)) == len(owner_indices), "CP row owner variables are aliased")
        _require(type(self.pairs) is tuple and self.pairs == tuple(sorted(self.pairs))
                 and all(type(row) is NativeCrossRowCpPair for row in self.pairs),
                 "CP source pairs are not canonical")
        for row in self.pairs: row.__post_init__()
        _require(len({p.members for p in self.pairs}) == len(self.pairs)
                 and len({p.axis_index for p in self.pairs}) == len(self.pairs),
                 "CP pair/axis identities are duplicated")
        _require(not set(owner_indices) & {p.axis_index for p in self.pairs},
                 "CP owner variable aliases a matching axis")
        _require(type(self.constraint_indices) is tuple
                 and all(type(i) is int and i >= 0 for i in self.constraint_indices)
                 and self.constraint_indices == tuple(sorted(set(self.constraint_indices))),
                 "CP source constraints are not canonical")


def _emit_native_cross_row_equalities(model, inventory, owners, pairs):
    """Emit exact per-occurrence CP constraints using existing solver variables."""
    by_row = {row.row_owner: row for row in owners}
    by_pair = {row.members: row for row in pairs}
    indices = []
    for pair in inventory.pairs:
        projection = by_pair[pair.members]
        axis_index = 0 if projection.axis == "x" else 1
        perpendicular_index = 1 - axis_index
        first, second = pair.endpoints
        left = sorted(first.channels, key=lambda c: (c.center2_nm[axis_index], c.occurrence_id))
        right = sorted(second.channels, key=lambda c: (c.center2_nm[axis_index], c.occurrence_id), reverse=True)
        # One-dimensional channel order is required; 2-D occurrence matching
        # needs an explicit producer and must not be guessed by this compiler.
        for endpoint in (first, second):
            _require(len({c.center2_nm[axis_index] for c in endpoint.channels}) == len(endpoint.channels)
                     and len({c.center2_nm[perpendicular_index] for c in endpoint.channels}) == 1,
                     "CP source occurrence order is not one-dimensional")
        axis = model.GetIntVarFromProtoIndex(projection.axis_index)
        for a, b in zip(left, right):
            _require(a.size_nm == b.size_nm and a.orientation == b.orientation,
                     "CP matched native channel shape/orientation differs")
            world = []
            for endpoint, channel in ((first, a), (second, b)):
                owner = by_row[endpoint.row_owner]
                x = model.GetIntVarFromProtoIndex(owner.x_index)
                y = model.GetIntVarFromProtoIndex(owner.y_index)
                world.append((2000 * x + 1000 * owner.padded_width_um + channel.center2_nm[0],
                              2000 * y + 1000 * owner.padded_height_um + channel.center2_nm[1]))
            indices.append(model.Add(axis == world[0][axis_index] + world[1][axis_index]).Index())
            indices.append(model.Add(world[0][perpendicular_index] == world[1][perpendicular_index]).Index())
    return tuple(indices)


def add_native_cross_row_cp_constraints(model, inventory, owner_variables, axis_directions):
    """Add source-bound exact CP constraints, without inferring a shared axis.

    ``owner_variables`` maps every source row owner to the actual unique CP
    owner key, its x/y IntVars and exact padded width/height in solver um units.
    ``axis_directions`` must come from the existing materializer/source matching
    axis directions; the old axis coordinates are deliberately not reused.
    The returned binding is incomplete until a real solver model and solution
    are captured and independently replayed by the caller.
    """
    _require(type(inventory) is NativeCrossRowInventory and inventory.pairs,
             "CP native source inventory is absent")
    inventory.__post_init__()
    _require(isinstance(owner_variables, Mapping) and isinstance(axis_directions, Mapping),
             "CP owner/direction inventories must be mappings")
    _require(set(owner_variables) == {row[0] for row in inventory.row_geometry_sha256}
             and set(axis_directions) == {pair.members for pair in inventory.pairs},
             "CP source row or matching denominator differs")
    owners = []
    for row in sorted(owner_variables):
        raw = owner_variables[row]
        _require(type(raw) is tuple and len(raw) == 5, "CP owner variables are malformed")
        owner, x, y, width, height = raw
        _require(hasattr(x, "Index") and hasattr(y, "Index"), "CP owner variable is foreign")
        x_index, y_index = x.Index(), y.Index()
        # Check that the supplied variables belong to this very model.
        _require(model.GetIntVarFromProtoIndex(x_index).Name() == x.Name()
                 and model.GetIntVarFromProtoIndex(y_index).Name() == y.Name(),
                 "CP owner variable belongs to a foreign model")
        _require(x.Name() == "x_" + owner and y.Name() == "y_" + owner,
                 "CP variables do not identify their solver owner")
        owners.append(NativeCrossRowCpOwner(row, owner, x_index, y_index, width, height))
    _require(len({x.solver_owner for x in owners}) == len(owners), "CP row owners are not distinct")
    pairs = []
    for pair in inventory.pairs:
        axis = axis_directions[pair.members]
        _require(axis in ("x", "y"), "CP source axis direction is missing")
        # Bounds cover a 250 m cell in 4x-nm units; CP-SAT checks actual
        # domains and the caller retains its existing floorplan/budget limits.
        var = model.NewIntVar(-10**12, 10**12, "native_cross_row_axis4_" + "__".join(pair.members))
        pairs.append(NativeCrossRowCpPair(pair.members, axis, var.Index()))
    indices = _emit_native_cross_row_equalities(model, inventory, tuple(owners), tuple(pairs))
    return NativeCrossRowCpBinding(inventory.sha256, tuple(owners), tuple(pairs), indices)


@dataclass(frozen=True, order=True)
class NativeCrossRowCpOwnerV2:
    row_owner: str
    solver_owner: str
    center2_x_index: int
    center2_y_index: int

    def __post_init__(self):
        _require(type(self.row_owner) is str and self.row_owner
                 and type(self.solver_owner) is str and self.solver_owner,
                 'v2 CP owner identity is malformed')
        _require(type(self.center2_x_index) is int and self.center2_x_index>=0
                 and type(self.center2_y_index) is int and self.center2_y_index>=0
                 and self.center2_x_index!=self.center2_y_index,
                 'v2 CP owner variables are malformed')


@dataclass(frozen=True)
class NativeCrossRowCpBindingV2:
    schema: str
    inventory_sha256: str
    frame_binding: object
    owners: tuple[NativeCrossRowCpOwnerV2,...]
    pairs: tuple[NativeCrossRowCpPair,...]
    constraint_indices: tuple[int,...]

    def __post_init__(self):
        from clay.layout.place.native_row_cp_frames import NativeRowCpFrameBinding
        _require(self.schema=='clay.native-cross-row-cp-binding/v2','v2 CP schema differs')
        _sha(self.inventory_sha256,'v2 CP source inventory')
        _require(type(self.frame_binding) is NativeRowCpFrameBinding,'v2 CP source frames are foreign')
        self.frame_binding.__post_init__()
        _require(type(self.owners) is tuple and self.owners==tuple(sorted(self.owners))
                 and all(type(row) is NativeCrossRowCpOwnerV2 for row in self.owners),
                 'v2 CP owners malformed')
        for row in self.owners: row.__post_init__()
        _require(len({row.row_owner for row in self.owners})==len(self.owners)
                 and len({row.solver_owner for row in self.owners})==len(self.owners),
                 'v2 CP owners are not unique')
        frame_by_owner={row.solver_owner:row for row in self.frame_binding.frames}
        for row in self.owners:
            frame=frame_by_owner.get(row.solver_owner)
            _require(frame is not None and (row.center2_x_index,row.center2_y_index)==
                     (frame.center2_x_index,frame.center2_y_index),
                     'v2 CP owner differs from source-frame binding')
        _require(type(self.pairs) is tuple and self.pairs==tuple(sorted(self.pairs))
                 and all(type(row) is NativeCrossRowCpPair for row in self.pairs),
                 'v2 CP pair inventory malformed')
        for row in self.pairs: row.__post_init__()
        _require(len({row.members for row in self.pairs})==len(self.pairs)
                 and len({row.axis_index for row in self.pairs})==len(self.pairs),
                 'v2 CP pair or axis is duplicated')
        owner_indices={v for row in self.owners for v in (row.center2_x_index,row.center2_y_index)}
        _require(len(owner_indices)==2*len(self.owners)
                 and not owner_indices & {row.axis_index for row in self.pairs},
                 'v2 CP owner or axis variables alias')
        _require(type(self.constraint_indices) is tuple
                 and all(type(i) is int and i>=0 for i in self.constraint_indices)
                 and self.constraint_indices==tuple(sorted(set(self.constraint_indices))),
                 'v2 CP constraint inventory malformed')


def _emit_native_cross_row_equalities_v2(model, inventory, owners, pairs):
    by_row={row.row_owner:row for row in owners}
    by_pair={row.members:row for row in pairs}
    indices=[]
    for pair in inventory.pairs:
        projection=by_pair[pair.members]
        ai=0 if projection.axis=='x' else 1; pi=1-ai
        first,second=pair.endpoints
        left=sorted(first.channels,key=lambda c:(c.center2_nm[ai],c.occurrence_id))
        right=sorted(second.channels,key=lambda c:(c.center2_nm[ai],c.occurrence_id),reverse=True)
        for endpoint in (first,second):
            _require(len({c.center2_nm[ai] for c in endpoint.channels})==len(endpoint.channels)
                     and len({c.center2_nm[pi] for c in endpoint.channels})==1,
                     'v2 CP source occurrence order is not one-dimensional')
        axis=model.GetIntVarFromProtoIndex(projection.axis_index)
        for a,b in zip(left,right):
            _require(a.size_nm==b.size_nm and a.orientation==b.orientation,
                     'v2 CP matched source channel shape differs')
            world=[]
            for endpoint,channel in ((first,a),(second,b)):
                owner=by_row[endpoint.row_owner]
                cx=model.GetIntVarFromProtoIndex(owner.center2_x_index)
                cy=model.GetIntVarFromProtoIndex(owner.center2_y_index)
                world.append((cx+channel.center2_nm[0],cy+channel.center2_nm[1]))
            indices.append(model.Add(axis==world[0][ai]+world[1][ai]).Index())
            indices.append(model.Add(world[0][pi]==world[1][pi]).Index())
    return tuple(indices)


def add_native_cross_row_cp_constraints_v2(model, inventory, owner_variables, axis_directions, *, frame_binding):
    """Use exact 2*nm row centres already constrained by the source-frame model.

    owner_variables[row]=(solver_owner, center2_x, center2_y).  The shared
    frame binding is subsequently closed by a current-source frame proof.
    """
    _require(type(inventory) is NativeCrossRowInventory and inventory.pairs,
             'v2 CP native source inventory is absent')
    inventory.__post_init__()
    _require(isinstance(owner_variables,Mapping) and isinstance(axis_directions,Mapping)
             and set(owner_variables)=={r[0] for r in inventory.row_geometry_sha256}
             and set(axis_directions)=={p.members for p in inventory.pairs},
             'v2 CP source row or pair inventory differs')
    owners=[]
    for row in sorted(owner_variables):
        raw=owner_variables[row]
        _require(type(raw) is tuple and len(raw)==3,'v2 CP owner variables malformed')
        owner,cx,cy=raw
        _require(hasattr(cx,'Index') and hasattr(cy,'Index')
                 and model.GetIntVarFromProtoIndex(cx.Index()).Name()==cx.Name()
                 and model.GetIntVarFromProtoIndex(cy.Index()).Name()==cy.Name(),
                 'v2 CP owner variables are foreign')
        owners.append(NativeCrossRowCpOwnerV2(row,owner,cx.Index(),cy.Index()))
    by_row={row.row_owner:row for row in owners}
    pairs=[]
    for pair in inventory.pairs:
        direction=axis_directions[pair.members]
        _require(direction in ('x','y'),'v2 CP axis direction is missing')
        ai=0 if direction=='x' else 1
        left,right=pair.endpoints
        left_idx=(by_row[left.row_owner].center2_x_index,by_row[left.row_owner].center2_y_index)[ai]
        right_idx=(by_row[right.row_owner].center2_x_index,by_row[right.row_owner].center2_y_index)[ai]
        left_domain=model.Proto().variables[left_idx].domain
        right_domain=model.Proto().variables[right_idx].domain
        offsets=[a.center2_nm[ai]+b.center2_nm[ai] for a in left.channels for b in right.channels]
        lower=left_domain[0]+right_domain[0]+min(offsets)
        upper=left_domain[-1]+right_domain[-1]+max(offsets)
        axis=model.NewIntVar(lower,upper,'native_cross_row_axis4_'+'__'.join(pair.members))
        pairs.append(NativeCrossRowCpPair(pair.members,direction,axis.Index()))
    indices=_emit_native_cross_row_equalities_v2(model,inventory,tuple(owners),tuple(pairs))
    return NativeCrossRowCpBindingV2('clay.native-cross-row-cp-binding/v2',inventory.sha256,
                                     frame_binding,tuple(owners),tuple(pairs),indices)


@dataclass(frozen=True)
class NativeCrossRowCpProof:
    """Captured CP model/solution plus its source-derived matching constraints."""
    binding: object
    model_bytes: bytes
    solution: tuple[int, ...]
    native_row_frame_proof: object = None

    def __post_init__(self) -> None:
        _require(type(self.binding) in (NativeCrossRowCpBinding,NativeCrossRowCpBindingV2), "CP binding is foreign")
        self.binding.__post_init__()
        _require(type(self.model_bytes) is bytes and 0 < len(self.model_bytes) <= 16 * 1024 * 1024,
                 "CP model bytes are absent or exceed the bounded capture")
        _require(type(self.solution) is tuple and all(type(v) is int for v in self.solution),
                 "CP solution must be exact integers")
        if type(self.binding) is NativeCrossRowCpBindingV2:
            from clay.layout.place.native_row_cp_frames import NativeRowCpFrameProof
            _require(type(self.native_row_frame_proof) is NativeRowCpFrameProof
                     and self.native_row_frame_proof.binding==self.binding.frame_binding
                     and self.native_row_frame_proof.model_bytes==self.model_bytes
                     and self.native_row_frame_proof.solution==self.solution,
                     'v2 CP proof lacks the exact source-bound frame proof')
        else:
            _require(self.native_row_frame_proof is None,'legacy CP proof cannot acquire v2 frame authority')

    @property
    def sha256(self) -> str:
        self.__post_init__()
        data = {"binding": asdict(self.binding), "model_sha256": sha256(self.model_bytes).hexdigest(),
                "solution": self.solution}
        if self.native_row_frame_proof is not None:
            data['native_row_frame_proof_sha256'] = self.native_row_frame_proof.sha256
        return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def replay_native_cross_row_cp_proof(inventory, proof, axis_directions, *,
                                     row_program=None, row_materialization=None, tech=None):
    """Replay source equalities and the complete captured CP solution.

    The caller must supply an inventory freshly reconstructed from its original
    captured source/PDK/row objects, and must separately replay real row seating.
    Return the signed row centres and axes; never infer an axis from a final bbox.
    """
    from ortools.sat import cp_model_pb2
    from ortools.sat.python import cp_model
    from clay.layout.place.production_model_evidence import _check_proto_solution

    _require(type(inventory) is NativeCrossRowInventory and type(proof) is NativeCrossRowCpProof,
             "CP proof or source inventory is foreign")
    inventory.__post_init__(); proof.__post_init__()
    binding = proof.binding
    _require(binding.inventory_sha256 == inventory.sha256, "CP source inventory changed")
    _require(isinstance(axis_directions, Mapping)
             and dict(axis_directions) == {p.members: p.axis for p in binding.pairs},
             "CP source axis directions changed")
    _require({r.row_owner for r in binding.owners} == {r[0] for r in inventory.row_geometry_sha256}
             and {p.members for p in binding.pairs} == {p.members for p in inventory.pairs},
             "CP source owner or pair denominator changed")
    proto = cp_model_pb2.CpModelProto(); proto.ParseFromString(proof.model_bytes)
    if type(binding) is NativeCrossRowCpBindingV2:
        from clay.layout.place.native_row_cp_frames import replay_native_row_cp_frame_proof
        _require(row_program is not None and row_materialization is not None and tech is not None,
                 'v2 CP replay requires the current source row and PDK')
        frames = replay_native_row_cp_frame_proof(proof.native_row_frame_proof,
                                                  row_program,row_materialization,tech)
        entries={row.entry_id:row for row in row_program.entries}
        for owner in binding.owners:
            entry=entries.get(owner.row_owner)
            _require(entry is not None and owner.solver_owner=='A_'+entry.group
                     and owner.solver_owner in frames,
                     'v2 CP row owner differs from current source projection')
        _require(all(i < len(proto.constraints) for i in binding.constraint_indices),
                 'v2 CP source constraint index is absent')
        regenerated=cp_model.CpModel(); regenerated.Proto().CopyFrom(proto)
        regenerated.Proto().ClearField('constraints')
        expected_indices=_emit_native_cross_row_equalities_v2(regenerated,inventory,binding.owners,binding.pairs)
        expected=tuple(regenerated.Proto().constraints[i].SerializeToString(deterministic=True)
                       for i in expected_indices)
        actual=tuple(proto.constraints[i].SerializeToString(deterministic=True)
                     for i in binding.constraint_indices)
        _require(expected==actual,'v2 CP source-derived equalities differ from captured model')
        centres={row.row_owner:frames[row.solver_owner] for row in binding.owners}
        axes={row.members:(row.axis,proof.solution[row.axis_index]) for row in binding.pairs}
        verify_native_cross_row_geometry(inventory,centres,axes)
        return centres, axes
    checks = _check_proto_solution(proto, proof.solution)
    _require(all(row["status"] == "PASS" for row in checks), "complete CP solution does not replay")
    intervals = {}
    for owner in binding.owners:
        _require(owner.x_index < len(proto.variables) and owner.y_index < len(proto.variables)
                 and proto.variables[owner.x_index].name == "x_" + owner.solver_owner
                 and proto.variables[owner.y_index].name == "y_" + owner.solver_owner,
                 "CP solver owner variable identity differs")
        pair = []
        for prefix, variable, size in (("ix_", owner.x_index, owner.padded_width_um),
                                       ("iy_", owner.y_index, owner.padded_height_um)):
            matches = [(i, row) for i, row in enumerate(proto.constraints)
                       if row.name == prefix + owner.solver_owner]
            _require(len(matches) == 1 and matches[0][1].WhichOneof("constraint") == "interval",
                     "CP solver owner interval identity differs")
            index, row = matches[0]
            start, dimension = row.interval.start, row.interval.size
            _require(tuple(start.vars) == (variable,) and tuple(start.coeffs) == (1,) and start.offset == 0
                     and not dimension.vars and not dimension.coeffs and dimension.offset == size,
                     "CP solver owner interval geometry differs")
            pair.append(index)
        intervals[owner.row_owner] = tuple(pair)
    expected_intervals = set(intervals.values())
    _require(any(row.WhichOneof("constraint") == "no_overlap_2d"
                 and expected_intervals <= set(zip(row.no_overlap_2d.x_intervals, row.no_overlap_2d.y_intervals))
                 for row in proto.constraints), "CP row owners lack common no-overlap")
    _require(all(i < len(proto.constraints) for i in binding.constraint_indices),
             "CP source constraint index is absent")
    regenerated = cp_model.CpModel(); regenerated.Proto().CopyFrom(proto)
    regenerated.Proto().ClearField("constraints")
    expected_indices = _emit_native_cross_row_equalities(regenerated, inventory, binding.owners, binding.pairs)
    expected = tuple(regenerated.Proto().constraints[i].SerializeToString(deterministic=True)
                     for i in expected_indices)
    actual = tuple(proto.constraints[i].SerializeToString(deterministic=True)
                   for i in binding.constraint_indices)
    _require(expected == actual, "CP source-derived equalities differ from captured model")
    centres = {row.row_owner: (
        2000 * proof.solution[row.x_index] + 1000 * row.padded_width_um,
        2000 * proof.solution[row.y_index] + 1000 * row.padded_height_um,
    ) for row in binding.owners}
    axes = {row.members: (row.axis, proof.solution[row.axis_index]) for row in binding.pairs}
    verify_native_cross_row_geometry(inventory, centres, axes)
    return centres, axes
