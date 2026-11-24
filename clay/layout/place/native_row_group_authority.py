"""Live source and execution authority for atomic native row footprints."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from hashlib import sha256
import inspect
import json
from weakref import WeakKeyDictionary


@dataclass(frozen=True, eq=False, init=False)
class NativeRowGroupAuthority:
    source_placement: object
    source_cell: object
    source_intent: object
    source_identity: tuple
    gap_um: str
    channel_relax_nets: tuple
    channel_pad_um: str
    channel_relax_passive_only: bool

    def __init__(self,*args,**kwargs):
        raise TypeError('native row group authority requires the actual placement caller')

    @property
    def sha256(self):
        if _CAPTURED.get(self) != (id(self.source_placement),id(self.source_cell),id(self.source_intent),
                self.source_identity,self.gap_um,self.channel_relax_nets,
                self.channel_pad_um,self.channel_relax_passive_only):
            raise ValueError('native row group execution authority changed or is foreign')
        data={'schema':'clay.native-row-group-authority/v1','source_identity':self.source_identity,
              'gap_um':self.gap_um,'channel_relax_nets':self.channel_relax_nets,
              'channel_pad_um':self.channel_pad_um,'channel_relax_passive_only':self.channel_relax_passive_only}
        return sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


_CAPTURED=WeakKeyDictionary()


def _number(value,label):
    number=Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError('native row group '+label+' is invalid')
    return str(number)


def capture_native_row_group_authority(program, materialization, tech, *, gap_um,
        channel_relax_nets, channel_pad_um, channel_relax_passive_only):
    from clay.layout.place.native_row_cp_frames import native_row_frames_from_projection
    from clay.layout.envelope_place import cp_compact_replace_placement
    caller=inspect.currentframe().f_back
    if caller is None or caller.f_code is not cp_compact_replace_placement.__code__:
        raise ValueError('native row group authority requires the actual placement caller')
    local=caller.f_locals
    base=local.get('base'); cell=local.get('cell_def'); intent=local.get('intent')
    if (getattr(base,'row_program',None) is not program
            or getattr(base,'row_materialization',None) is not materialization
            or local.get('tech_id') != tech.pdk_id
            or local.get('gap_um') != gap_um
            or local.get('channel_relax_nets') != channel_relax_nets
            or local.get('channel_pad_um') != channel_pad_um
            or local.get('channel_relax_passive_only') != channel_relax_passive_only):
        raise ValueError('native row group caller configuration differs')
    _,identity=native_row_frames_from_projection(program,materialization,tech)
    if type(channel_relax_passive_only) is not bool:
        raise TypeError('native row channel policy must be boolean')
    nets=tuple(sorted(set(str(value) for value in (channel_relax_nets or ()))))
    values=(base,cell,intent,identity,_number(gap_um,'gap'),nets,_number(channel_pad_um,'channel pad'),channel_relax_passive_only)
    authority=object.__new__(NativeRowGroupAuthority)
    for name,value in zip(('source_placement','source_cell','source_intent','source_identity','gap_um','channel_relax_nets','channel_pad_um',
                           'channel_relax_passive_only'),values):
        object.__setattr__(authority,name,value)
    _CAPTURED[authority]=(id(base),id(cell),id(intent),*values[3:])
    return authority


def replay_native_row_group_authority(authority, program, materialization, tech, *, _precomputed_frames_identity=None):
    """Rebuild actual native tile groups from the live source and caller config."""
    from clay.layout.envelope_place import _extracted_intent_from_cell_def
    from clay.layout.extract.well_islands import compact_matched_groups
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place.native_row_cp_frames import native_row_frames_from_projection
    from clay.layout.synth.placement import (ArrayTile, _array_tile_group,
        _channel_incident_groups, _group_ring_pad_um_by_kind, via_landing_budget_um)
    if type(authority) is not NativeRowGroupAuthority:
        raise ValueError('native row group authority is missing or foreign')
    authority.sha256
    frames, identity = (
        _precomputed_frames_identity
        if _precomputed_frames_identity is not None
        else native_row_frames_from_projection(program, materialization, tech)
    )
    if identity != authority.source_identity:
        raise ValueError('native row group source or PDK changed')
    if (getattr(authority.source_placement,'row_program',None) is not program
            or getattr(authority.source_placement,'row_materialization',None) is not materialization):
        raise ValueError('native row group placement caller changed')
    witness=materialization.source_projection.native_witness
    cell=witness.cell
    source_intent=layout_intent_from_cell_def(cell)
    if tuple(getattr(authority.source_intent,'guard_rings',()) or ()) != tuple(source_intent.guard_rings):
        raise ValueError('native row group caller guard declarations differ from source')
    ext=_extracted_intent_from_cell_def(cell,native_geometry=witness.bundle)
    ext.guard_islands=[{'members':tuple(str(m) for m in row.members),
        'well':str(row.well or ''),'tie':str(row.tie_net or '')} for row in source_intent.guard_rings]
    entries={entry.entry_id:entry for entry in program.entries}
    members={member.device:member for member in program.members}
    array_owner={name:'A_'+entry.group for entry in entries.values() for name in entry.consumed_devices}
    if len(array_owner)!=sum(len(entry.consumed_devices) for entry in entries.values()):
        raise ValueError('native row group owner partition differs')
    groups=compact_matched_groups(ext)
    groups=[(kind,body,tuple(name for name in names if str(name) not in array_owner))
            for kind,body,names in groups]
    groups=[row for row in groups if row[2]]
    channel_groups=_channel_incident_groups(ext,groups,array_owner,authority.channel_relax_nets,
        float(authority.channel_pad_um),authority.channel_relax_passive_only)
    islands={str(name) for row in source_intent.guard_rings for name in row.members}
    pad_by_kind=_group_ring_pad_um_by_kind(ext,tech.pdk_id)
    via_pad=via_landing_budget_um(tech.pdk_id)
    result={}
    for entry in entries.values():
        owner='A_'+entry.group
        if owner not in frames:continue
        box,_grid=frames[owner]
        tile=ArrayTile(entry.group,members[entry.active_devices[0]].kind,entry.consumed_devices,
            (box[2]-box[0])/1000.0,(box[3]-box[1])/1000.0,{})
        group,_island_pad=_array_tile_group(tile,pad_by_kind,via_pad,channel_groups,
                                            float(authority.channel_pad_um),islands)
        def ceil_nm(value):
            return int((Decimal(str(value))*1000).to_integral_value(rounding=ROUND_CEILING))
        result[owner]=(ceil_nm(group.w),ceil_nm(group.h))
    if set(result)!=set(frames):
        raise ValueError('native row group source inventory differs')
    gap_nm=int((Decimal(authority.gap_um)*1000).to_integral_value(rounding=ROUND_CEILING))
    return result,gap_nm


def authenticate_serialized_native_row_group_authority(
    authority,
    *,
    source_placement,
    source_cell,
    source_intent,
    program,
    materialization,
    tech,
    expected_authority_sha256: str,
    _return_replay_tuple: bool = False,
):
    """Re-verify a serialized NativeRowGroupAuthority against live operands."""
    from clay.layout.place.native_row_cp_frames import native_row_frames_from_projection

    if type(authority) is not NativeRowGroupAuthority:
        raise ValueError('native row group authority is missing or foreign')
    if (
        getattr(source_placement, 'row_program', None) is not program
        or getattr(source_placement, 'row_materialization', None) is not materialization
    ):
        raise ValueError('native row group placement caller changed')
    _frames, identity = native_row_frames_from_projection(program, materialization, tech)
    if identity != authority.source_identity:
        raise ValueError('native row group source or PDK changed')
    object.__setattr__(authority, 'source_placement', source_placement)
    object.__setattr__(authority, 'source_cell', source_cell)
    object.__setattr__(authority, 'source_intent', source_intent)
    data = {
        'schema': 'clay.native-row-group-authority/v1',
        'source_identity': authority.source_identity,
        'gap_um': authority.gap_um,
        'channel_relax_nets': authority.channel_relax_nets,
        'channel_pad_um': authority.channel_pad_um,
        'channel_relax_passive_only': authority.channel_relax_passive_only,
    }
    actual_sha = sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual_sha != expected_authority_sha256:
        raise ValueError('native row group authority digest differs from manifest')
    _CAPTURED[authority] = (
        id(source_placement),
        id(source_cell),
        id(source_intent),
        authority.source_identity,
        authority.gap_um,
        authority.channel_relax_nets,
        authority.channel_pad_um,
        authority.channel_relax_passive_only,
    )
    group_map, gap_nm = replay_native_row_group_authority(
        authority,
        program,
        materialization,
        tech,
        _precomputed_frames_identity=(_frames, identity),
    )
    if _return_replay_tuple:
        return authority, (_frames, identity, group_map, gap_nm)
    return authority
