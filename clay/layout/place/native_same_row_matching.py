"""Source-bound same-row matching occurrence and final-pose certificates."""
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Mapping

from clay.layout.place.cross_row_matching import (
    NativeRowChannel, NativeRowMatchingEndpoint, _channel_offset2_nm,
    _require, _sha, _exact_stamp_position_nm, _require_unique_native_row_stamp,
)

@dataclass(frozen=True, order=True)
class NativeSameRowPair:
    endpoints: tuple[NativeRowMatchingEndpoint, NativeRowMatchingEndpoint]
    declaration_refs: tuple[str, ...]

    def __post_init__(self):
        _require(type(self.endpoints) is tuple and len(self.endpoints) == 2
                 and all(type(x) is NativeRowMatchingEndpoint for x in self.endpoints), 'same-row endpoints malformed')
        left, right = self.endpoints
        left.__post_init__(); right.__post_init__()
        _require(left.member < right.member and left.row_owner == right.row_owner, 'same-row owner differs')
        _require(type(self.declaration_refs) is tuple and self.declaration_refs
                 and self.declaration_refs == tuple(sorted(set(self.declaration_refs))), 'same-row declarations malformed')

    @property
    def members(self):
        return tuple(x.member for x in self.endpoints)

@dataclass(frozen=True)
class NativeSameRowInventory:
    source_sha256: str
    source_projection_sha256: str
    program_sha256: str
    materialization_sha256: str
    graph_sha256: str
    pdk_sha256: str
    row_geometry_sha256: tuple
    pairs: tuple
    row_source_members: tuple

    def __post_init__(self):
        for name in ('source_sha256','source_projection_sha256','program_sha256','materialization_sha256','graph_sha256','pdk_sha256'):
            _sha(getattr(self,name), name)
        _require(type(self.pairs) is tuple and self.pairs and all(type(p) is NativeSameRowPair for p in self.pairs), 'same-row pairs malformed')
        for pair in self.pairs: pair.__post_init__()
        keys=tuple(p.members for p in self.pairs)
        _require(keys==tuple(sorted(set(keys))), 'same-row pair inventory malformed')
        _require(type(self.row_source_members) is tuple and self.row_source_members==tuple(sorted(self.row_source_members)), 'same-row owner inventory malformed')
        rows=dict(self.row_source_members)
        _require(len(rows)==len(self.row_source_members) and set(rows)=={x[0] for x in self.row_geometry_sha256}, 'same-row owner inventory differs')
        for owner, digest in self.row_geometry_sha256: _sha(digest, 'same-row geometry')
        _require(all(p.endpoints[0].row_owner in rows and set(p.members)<=set(rows[p.endpoints[0].row_owner]) for p in self.pairs), 'same-row member ownership differs')

    @property
    def sha256(self):
        self.__post_init__()
        return sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def native_same_row_inventory(cell, program, materialization, tech, *, source_cir):
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
    native_groups = {row.group for row in source_intent.arrays if row.style == 'native_diffusion'}
    requested = []
    for match in source_intent.matching_intents:
        pair = tuple(sorted(match.members))
        if match.requires_interdigitation or not all(m in owner_of for m in pair):
            continue
        if owner_of[pair[0]] != owner_of[pair[1]]:
            continue
        if entries[owner_of[pair[0]]].group not in native_groups:
            continue
        requested.append((pair, tuple(sorted(match.declaration_ids))))
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
    pairs = tuple(sorted(NativeSameRowPair(
        tuple(endpoints[name] for name in pair), refs) for pair, refs in requested))
    return NativeSameRowInventory(
        projection.source_sha256, projection.sha256, program.sha256,
        materialization.certificate.sha256, graph.sha256, next(iter(pdk_hashes)),
        tuple(sorted(geometry_hashes)), pairs,
        tuple((owner, entries[owner].consumed_devices) for owner in sorted(needed)))


def native_same_row_inventory_from_projection(program, materialization, tech):
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
    return native_same_row_inventory(witness.cell, program, materialization, tech,
                                      source_cir=projection.source_cir)


def native_same_row_stamp_centres(inventory, program, materialization, geometry, placement_um):
    """Replay current complete native stamps and recover their exact row poses.

    The source inventory must already have been reconstructed from current
    source/PDK operands.  The complete joint artwork is replayed before a
    native-origin translation is applied to the source row's exact extent.
    """
    from clay.layout.joint_row_stamp import JointRowStampBinding, joint_stamp_owner_records

    _require(type(inventory) is NativeSameRowInventory, "stamp source inventory is foreign")
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


def native_same_row_occurrence_axes(inventory, centres, logical_axes):
    """Measure each occurrence centroid axis, preserving the source direction.

    Ordinary matching is tolerance based, so this does not impose pointwise
    reflection or equality with the abstract solver envelope axis.
    """
    inventory.__post_init__()
    _require(set(centres) == {r[0] for r in inventory.row_source_members}, 'same-row owner inventory differs')
    _require({p.members for p in inventory.pairs} <= set(logical_axes), 'same-row solver axis inventory differs')
    result = {}
    for pair in inventory.pairs:
        direction = logical_axes[pair.members][0]
        _require(direction in ('x','y'), 'same-row solver axis malformed')
        i = 0 if direction == 'x' else 1
        means = []
        for endpoint in pair.endpoints:
            center = centres[endpoint.row_owner]
            _require(type(center) is tuple and len(center)==2 and all(type(v) is int for v in center), 'same-row owner pose malformed')
            means.append(Fraction(sum(center[i]+c.center2_nm[i] for c in endpoint.channels),len(endpoint.channels)))
        # Each mean is in 2*nm; their sum is 4*nm of the midpoint.
        axis4 = sum(means)
        result[pair.members] = (direction, int(axis4) if axis4.denominator == 1 else axis4)
    return result


def verify_native_same_row_geometry(inventory, centres, axes):
    _require(type(inventory) is NativeSameRowInventory, 'same-row inventory type differs')
    _require(set(axes)=={p.members for p in inventory.pairs}, 'same-row solver axis inventory differs')
    _require(native_same_row_occurrence_axes(inventory,centres,axes)==dict(axes), 'same-row source occurrence differs from solver axis')
    return True


@dataclass(frozen=True)
class NativeSameRowFinalProof:
    inventory_sha256: str
    source_centres2_nm: tuple
    source_axes4_nm: tuple
    source_logical_axes4_nm: tuple
    final_centres2_nm: tuple
    final_axes4_nm: tuple
    final_logical_axes4_nm: tuple
    manufacturing_grid_nm: int

    def __post_init__(self):
        _sha(self.inventory_sha256, 'same-row inventory')
        for value in (self.source_centres2_nm,self.source_axes4_nm,self.source_logical_axes4_nm,self.final_centres2_nm,self.final_axes4_nm,self.final_logical_axes4_nm):
            _require(type(value) is tuple and value==tuple(sorted(value)) and len(dict(value))==len(value), 'same-row proof inventory malformed')
        _require(type(self.manufacturing_grid_nm) is int and self.manufacturing_grid_nm>0, 'same-row grid invalid')

    @property
    def sha256(self):
        self.__post_init__()
        def rational(value):
            if type(value) is Fraction:
                return {'numerator':value.numerator,'denominator':value.denominator}
            raise TypeError('same-row proof contains a noncanonical value')
        return sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':'),default=rational).encode()).hexdigest()


def native_same_row_final_proof(inventory, source_centres, axes, final_centres, manufacturing_grid_nm, *, logical_axes=None):
    verify_native_same_row_geometry(inventory,source_centres,axes)
    _require(type(manufacturing_grid_nm) is int and manufacturing_grid_nm>0, 'same-row grid invalid')
    _require(set(final_centres)==set(source_centres), 'same-row final owner inventory differs')
    deltas = {}
    for owner, source in source_centres.items():
        final=final_centres[owner]
        _require(type(final) is tuple and len(final)==2 and all(type(v) is int for v in final), 'same-row final pose malformed')
        delta=tuple(final[i]-source[i] for i in (0,1))
        _require(all(v%(2*manufacturing_grid_nm)==0 for v in delta), 'same-row final pose is off grid')
        deltas[owner]=delta
    if logical_axes is None: logical_axes=axes
    _require(set(logical_axes)==set(axes), 'same-row logical axis inventory differs')
    final_axes={}; final_logical={}
    for pair in inventory.pairs:
        direction, coordinate=axes[pair.members]
        logical=logical_axes[pair.members]
        _require(type(logical) is tuple and len(logical)==2 and logical[0]==direction and type(logical[1]) is int, 'same-row logical axis malformed')
        delta=deltas[pair.endpoints[0].row_owner][0 if direction=='x' else 1]
        final_axes[pair.members]=(direction,coordinate+2*delta)
        final_logical[pair.members]=(direction,logical[1]+2*delta)
    verify_native_same_row_geometry(inventory,final_centres,final_axes)
    return NativeSameRowFinalProof(inventory.sha256,tuple(sorted(source_centres.items())),tuple(sorted(axes.items())),tuple(sorted(logical_axes.items())),tuple(sorted(final_centres.items())),tuple(sorted(final_axes.items())),tuple(sorted(final_logical.items())),manufacturing_grid_nm)


def replay_native_same_row_final_proof(inventory, proof, source_centres, axes, final_centres, manufacturing_grid_nm, *, logical_axes=None):
    _require(type(proof) is NativeSameRowFinalProof, 'same-row final proof is foreign')
    proof.__post_init__()
    expected=native_same_row_final_proof(inventory,source_centres,axes,final_centres,manufacturing_grid_nm,logical_axes=logical_axes)
    _require(proof==expected, 'same-row final pose or axes differ from replay')
    return dict(proof.final_axes4_nm)


