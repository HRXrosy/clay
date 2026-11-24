"""Exact source-row phases inside the atomic solver's conservative boxes."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping


def _require(value, message):
    if not value:
        raise ValueError('native row CP frame: '+message)


@dataclass(frozen=True, order=True)
class NativeRowCpFrame:
    row_owner: str
    solver_owner: str
    local_bbox_nm: tuple[int,int,int,int]
    grid_nm: int
    x_index: int
    y_index: int
    center2_x_index: int
    center2_y_index: int
    padded_width_um: int
    padded_height_um: int
    group_width_nm: int
    group_height_nm: int

    def __post_init__(self):
        _require(type(self.row_owner) is str and self.row_owner and type(self.solver_owner) is str
                 and self.solver_owner, 'owner identity malformed')
        box = self.local_bbox_nm
        _require(type(box) is tuple and len(box)==4 and all(type(v) is int for v in box)
                 and box[2]>box[0] and box[3]>box[1], 'source bbox malformed')
        _require(type(self.grid_nm) is int and self.grid_nm>0, 'PDK grid malformed')
        indices=(self.x_index,self.y_index,self.center2_x_index,self.center2_y_index)
        _require(all(type(v) is int and v>=0 for v in indices) and len(set(indices))==4,
                 'variable identities malformed')
        _require(all(type(v) is int and v>0 for v in (self.padded_width_um,self.padded_height_um,
                                                       self.group_width_nm,self.group_height_nm)),
                 'footprint malformed')
        _require(box[2]-box[0]<=self.group_width_nm and box[3]-box[1]<=self.group_height_nm,
                 'source row escapes group footprint')


@dataclass(frozen=True)
class NativeRowCpFrameBinding:
    schema: str
    frames: tuple[NativeRowCpFrame,...]
    gap_nm: int
    constraint_indices: tuple[int,...]

    def __post_init__(self):
        _require(self.schema=='clay.native-row-cp-frame/v1','schema differs')
        _require(type(self.frames) is tuple and self.frames and self.frames==tuple(sorted(self.frames))
                 and all(type(row) is NativeRowCpFrame for row in self.frames), 'frame inventory malformed')
        for row in self.frames: row.__post_init__()
        _require(len({r.row_owner for r in self.frames})==len(self.frames)
                 and len({r.solver_owner for r in self.frames})==len(self.frames),
                 'row or solver owner is not unique')
        all_indices=[i for r in self.frames for i in (r.x_index,r.y_index,r.center2_x_index,r.center2_y_index)]
        _require(len(set(all_indices))==len(all_indices),'owner variables are aliased')
        _require(type(self.gap_nm) is int and self.gap_nm>=0,'gap malformed')
        _require(type(self.constraint_indices) is tuple
                 and all(type(i) is int and i>=0 for i in self.constraint_indices)
                 and self.constraint_indices==tuple(sorted(set(self.constraint_indices))),
                 'constraint identities malformed')


def _emit_frame_constraints(model, frames, gap_nm):
    indices=[]
    for frame in frames:
        box=frame.local_bbox_nm
        for coarse_index, center_index, padded, width, low, high in (
            (frame.x_index,frame.center2_x_index,frame.padded_width_um,frame.group_width_nm,box[0],box[2]),
            (frame.y_index,frame.center2_y_index,frame.padded_height_um,frame.group_height_nm,box[1],box[3]),
        ):
            coarse=model.GetIntVarFromProtoIndex(coarse_index)
            center=model.GetIntVarFromProtoIndex(center_index)
            indices.append(model.AddModuloEquality((low+high)%(2*frame.grid_nm),
                                                   center,2*frame.grid_nm).Index())
            indices.append(model.Add(center >= 2000*coarse + width + gap_nm).Index())
            indices.append(model.Add(center <= 2000*(coarse+padded)-width-gap_nm).Index())
    return tuple(indices)


def add_native_row_frames_cp(model, frames, coarse, *, gap_nm):
    """Bind source row centres to the real coarse interval and PDK lattice.

    frames[owner]=(local_bbox_nm, grid_nm); coarse[owner]=(solver_owner,x,y,
    padded_w_um,padded_h_um,group_w_nm,group_h_nm).
    """
    _require(isinstance(frames,Mapping) and isinstance(coarse,Mapping)
             and frames and set(frames)==set(coarse),'source row inventory differs')
    _require(type(gap_nm) is int and gap_nm>=0,'gap malformed')
    result={}; rows=[]
    for name in sorted(frames):
        source=frames[name]; raw=coarse[name]
        _require(type(source) is tuple and len(source)==2 and type(raw) is tuple and len(raw)==7,
                 'caller row frame malformed')
        box,grid=source
        owner,x,y,fw,fh,width,height=raw
        _require(hasattr(x,'Index') and hasattr(y,'Index'),'coarse variables are foreign')
        _require(model.GetIntVarFromProtoIndex(x.Index()).Name()==x.Name()
                 and model.GetIntVarFromProtoIndex(y.Index()).Name()==y.Name()
                 and x.Name()=='x_'+owner and y.Name()=='y_'+owner,
                 'coarse variable ownership differs')
        proto=model.Proto()
        _require(proto.variables[x.Index()].domain[0]>=0 and proto.variables[y.Index()].domain[0]>=0,
                 'negative coarse origin is unsupported')
        max_x=2000*(proto.variables[x.Index()].domain[-1]+fw)
        max_y=2000*(proto.variables[y.Index()].domain[-1]+fh)
        center_x=model.NewIntVar(0,max_x,'native_row_center2_x_'+owner)
        center_y=model.NewIntVar(0,max_y,'native_row_center2_y_'+owner)
        row=NativeRowCpFrame(name,owner,box,grid,x.Index(),y.Index(),center_x.Index(),center_y.Index(),
                             fw,fh,width,height)
        rows.append(row);result[name]=(center_x,center_y)
    indices=_emit_frame_constraints(model,tuple(rows),gap_nm)
    binding=NativeRowCpFrameBinding('clay.native-row-cp-frame/v1',tuple(rows),gap_nm,indices)
    return result,binding


def replay_native_row_frames_cp(binding, proto, solution, source_frames, *, gap_nm, group_authority):
    """Replay fresh source extents, exact model constraints and solved owners."""
    from ortools.sat.python import cp_model
    from clay.layout.place.production_model_evidence import _check_proto_solution
    _require(type(binding) is NativeRowCpFrameBinding,'binding type differs')
    binding.__post_init__()
    _require(isinstance(source_frames,Mapping) and type(gap_nm) is int and gap_nm==binding.gap_nm,
             'source frames or gap authority differs')
    expected={r.row_owner:(r.local_bbox_nm,r.grid_nm) for r in binding.frames}
    _require(dict(source_frames)==expected,'current source row or PDK grid changed')
    expected_groups={r.row_owner:(r.group_width_nm,r.group_height_nm) for r in binding.frames}
    _require(isinstance(group_authority,Mapping) and dict(group_authority)==expected_groups,
             'current source-derived group footprint authority differs')
    _require(all(row['status']=='PASS' for row in _check_proto_solution(proto,solution)),
             'complete solver solution does not replay')
    intervals=set()
    for row in binding.frames:
        variables=((row.x_index,'x_'+row.solver_owner),(row.y_index,'y_'+row.solver_owner),
                   (row.center2_x_index,'native_row_center2_x_'+row.solver_owner),
                   (row.center2_y_index,'native_row_center2_y_'+row.solver_owner))
        _require(all(i<len(proto.variables) and proto.variables[i].name==name for i,name in variables),
                 'solver variable identity differs')
        pair=[]
        for prefix,index,size in (('ix_',row.x_index,row.padded_width_um),('iy_',row.y_index,row.padded_height_um)):
            found=[(i,c) for i,c in enumerate(proto.constraints) if c.name==prefix+row.solver_owner]
            _require(len(found)==1 and found[0][1].WhichOneof('constraint')=='interval',
                     'coarse owner interval missing')
            i,interval=found[0]; start=interval.interval.start; length=interval.interval.size
            _require(tuple(start.vars)==(index,) and tuple(start.coeffs)==(1,) and start.offset==0
                     and not length.vars and not length.coeffs and length.offset==size,
                     'coarse interval geometry differs')
            pair.append(i)
        intervals.add(tuple(pair))
    _require(any(row.WhichOneof('constraint')=='no_overlap_2d'
        and intervals <= set(zip(row.no_overlap_2d.x_intervals,row.no_overlap_2d.y_intervals))
        for row in proto.constraints),'coarse row owners lack common no-overlap')
    _require(all(i<len(proto.constraints) for i in binding.constraint_indices),'frame constraint missing')
    regenerated=cp_model.CpModel(); regenerated.Proto().CopyFrom(proto)
    regenerated.Proto().ClearField('constraints')
    expected_indices=_emit_frame_constraints(regenerated,binding.frames,binding.gap_nm)
    expected=tuple(regenerated.Proto().constraints[i].SerializeToString(deterministic=True) for i in expected_indices)
    actual=tuple(proto.constraints[i].SerializeToString(deterministic=True) for i in binding.constraint_indices)
    _require(expected==actual,'source-derived phase or containment constraint differs')
    return {row.row_owner:(solution[row.center2_x_index],solution[row.center2_y_index]) for row in binding.frames}


def native_row_frames_from_projection(program, materialization, tech):
    """Reconstruct every native joint row frame from the current source witness."""
    from clay.layout.native_diffusion_source import NativePCellWitness
    from clay.layout.place.row_replay_session import (
        _CURRENT as _REPLAY_CURRENT,
        require_current_row_materialization,
        current_row_source_program,
        current_row_terminal_inventory,
    )
    from clay.layout.place.row_source_projection import (
        require_actual_row_source_topology,
        require_row_source_projection,
    )
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.resource_world import pdk_provenance_digest
    live_proof = _REPLAY_CURRENT.get()
    has_scoped_source_replay = (
        live_proof is not None
        and live_proof.materialization is materialization
        and live_proof.program is program
        and live_proof.replayed_source_program is not None
        and live_proof.tech.pdk_id == tech.pdk_id
    )
    if not has_scoped_source_replay:
        require_current_row_materialization(materialization,program=program,tech_id=tech.pdk_id)
    projection=getattr(materialization,'source_projection',None)
    require_row_source_projection(projection,program=program)
    witness=getattr(projection,'native_witness',None)
    _require(type(witness) is NativePCellWitness,'current native source witness is missing')
    current_row_source_program(projection,witness.cell,tech,program=program,materialization=materialization)
    require_actual_row_source_topology(projection,projection.source_cir)
    members={row.device:row for row in program.members}
    entries={row.entry_id:row for row in program.entries}
    realizations={row.entry_id:row for row in materialization.realizations}
    _require(set(entries)==set(realizations),'native row realization inventory differs')
    witness_rows=tuple(name for name,_geometry,_folder in witness.rows)
    _require(witness_rows and len(set(witness_rows))==len(witness_rows),
             'native source witness rows are missing or duplicated')
    native_entries=[entry for entry in entries.values() if entry.group in set(witness_rows)]
    _require(len(native_entries)==len(witness_rows)
             and {entry.group for entry in native_entries}==set(witness_rows),
             'native witness does not identify exactly one source row per group')
    frames={}
    grid=pdk_manufacturing_grid_nm(tech)
    for entry in sorted(native_entries,key=lambda entry:entry.entry_id):
        row=realizations[entry.entry_id]
        _require(getattr(row.geometry,'joint_row_binding',None) is not None,
                 'native source row has no joint artwork binding')
        consumed_sorted = tuple(sorted(entry.consumed_devices))
        if has_scoped_source_replay:
            scoped_entry = live_proof.entries.get(id(row.geometry))
            if (
                scoped_entry is None
                or scoped_entry.geometry is not row.geometry
                or scoped_entry.members != consumed_sorted
                or getattr(row.geometry.joint_row_binding, 'joint_inventory', None) is not scoped_entry.inventory
            ):
                current_row_terminal_inventory(row.geometry, consumed_sorted, members, tech)
        else:
            current_row_terminal_inventory(row.geometry, consumed_sorted, members, tech)
        _require(tuple(row.geometry.extent())==row.local_bbox_nm,'native source row bbox changed')
        points=[point for poly in row.geometry.polygons for point in poly.polygon.vertices]
        _require(points and all(type(point.x) is int and type(point.y) is int
                     and point.x%grid==0 and point.y%grid==0 for point in points),
                 'native source row artwork is off the PDK manufacturing grid')
        owner='A_'+entry.group
        _require(owner not in frames,'native source row solver owner is duplicated')
        frames[owner]=(row.local_bbox_nm,grid)
    _require(frames,'no source-bound native row frames')
    return frames, (projection.source_sha256,projection.sha256,program.sha256,
                    materialization.certificate.sha256,pdk_provenance_digest(tech))


@dataclass(frozen=True)
class NativeRowCpFrameProof:
    binding: NativeRowCpFrameBinding
    model_bytes: bytes
    solution: tuple[int,...]
    source_identity: tuple[str,str,str,str,str]
    group_authority: object

    def __post_init__(self):
        _require(type(self.binding) is NativeRowCpFrameBinding,'proof binding differs')
        self.binding.__post_init__()
        _require(type(self.model_bytes) is bytes and 0<len(self.model_bytes)<=16*1024*1024,
                 'proof model bytes missing or too large')
        _require(type(self.solution) is tuple and all(type(value) is int for value in self.solution),
                 'proof solution must be exact integers')
        _require(type(self.source_identity) is tuple and len(self.source_identity)==5
                 and all(type(value) is str and len(value)==64
                         and all(ch in '0123456789abcdef' for ch in value)
                         for value in self.source_identity),'proof source identity malformed')
        from clay.layout.place.native_row_group_authority import NativeRowGroupAuthority
        _require(type(self.group_authority) is NativeRowGroupAuthority,
                 'proof lacks a live source-bound group authority')

    @property
    def sha256(self):
        self.__post_init__()
        payload={'schema':'clay.native-row-cp-frame-proof/v1','binding':asdict(self.binding),
                 'model_sha256':sha256(self.model_bytes).hexdigest(),'solution':self.solution,
                 'source_identity':self.source_identity,
                 'group_authority_sha256':self.group_authority.sha256}
        return sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def capture_native_row_cp_frame_proof(binding, model_bytes, solution, program, materialization, tech, *,
                                      group_authority):
    frames,identity=native_row_frames_from_projection(program,materialization,tech)
    proof=NativeRowCpFrameProof(binding,model_bytes,solution,identity,group_authority)
    replay_native_row_cp_frame_proof(proof,program,materialization,tech)
    return proof


def replay_native_row_cp_frame_proof(proof, program, materialization, tech, *, _precomputed_frames_and_groups=None):
    """Replay source/PDK, all model constraints and exact solved row centres."""
    from ortools.sat import cp_model_pb2
    _require(type(proof) is NativeRowCpFrameProof,'proof type differs')
    proof.__post_init__()
    if _precomputed_frames_and_groups is not None:
        frames, identity, group_authority, gap_nm = _precomputed_frames_and_groups
        _require(identity==proof.source_identity,'current source/PDK row identity changed')
    else:
        frames,identity=native_row_frames_from_projection(program,materialization,tech)
        _require(identity==proof.source_identity,'current source/PDK row identity changed')
        from clay.layout.place.native_row_group_authority import replay_native_row_group_authority
        group_authority,gap_nm=replay_native_row_group_authority(
            proof.group_authority,program,materialization,tech,
            _precomputed_frames_identity=(frames,identity))
    proto=cp_model_pb2.CpModelProto();proto.ParseFromString(proof.model_bytes)
    return replay_native_row_frames_cp(proof.binding,proto,proof.solution,frames,
                                       gap_nm=gap_nm,group_authority=group_authority)
