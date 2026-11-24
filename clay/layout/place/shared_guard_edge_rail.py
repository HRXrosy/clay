"""Replay actual joint guard conductors through the original rail predicate."""
from dataclasses import replace
from collections import Counter,defaultdict


def replay_actual_shared_guard_rails(*, certificate, ir, geometry, placement_um, intent, layout_model):
    from .row_native_geometry import RowNativeGeometryBinding
    from .edge_rail_geometry import _require, _stamp_geometry, _shift, _conductor_box, _sha
    from .edge_rail_reservation import EdgeRailStampIR, EdgeRailConductorAtom
    binding=RowNativeGeometryBinding.from_route_geometry(certificate=certificate,
        geometry=geometry,placement_um=placement_um,intent=intent)
    inventory=binding._replay()
    _require(bool(inventory.shared_guards), 'changed stamp inventory has no typed joint guard replacement')
    native_sources=dict(certificate.materialized_block_source_members)
    masters={native:master for master,natives in certificate.materialized_block_members for native in natives}
    positions={name:box[:2] for name,box in certificate.blocks}
    dims={name:(box[2]-box[0],box[3]-box[1]) for name,box in certificate.blocks}
    replaced=set()
    added=set()
    anchors={}
    signatures={}
    for group in inventory.shared_guards:
        old=[]
        for native in group.native_ids:
            matches=[stamp for stamp in ir.stamps if stamp.master_id==masters[native]
                     and stamp.guard_source_members==native_sources[native]]
            _require(len(matches)==1, 'joint guard replacement lacks exact original guard provenance')
            old.append(matches[0])
        ids={stamp.stamp_id for stamp in old}
        _require(not ids & replaced and group.guard_stamp not in added,
                 'joint guard replacement repeats physical artwork')
        replaced.update(ids)
        added.add(group.guard_stamp)
        anchors[group.guard_stamp]=group.masters[0]
        signatures[group.guard_stamp]={(atom.layer_key,atom.net,atom.kind) for atom in ir.atoms
                                      if atom.stamp_id in ids}
    aliases={}
    for stamp in ir.stamps:
        if stamp.stamp_id in replaced:
            continue
        if stamp.guard_source_members:
            matches=[name for name,artwork in geometry.items()
                if getattr(artwork,'guard_ring_declared_members',())==stamp.guard_source_members
                and getattr(artwork,'guard_ring_intent_sha256','')==stamp.guard_declaration_sha256]
            _require(len(matches)==1, 'unshared guard rename lacks exact declaration provenance')
            aliases[stamp.stamp_id]=matches[0]
        else:
            aliases[stamp.stamp_id]=stamp.stamp_id
    _require(len(set(aliases.values()))==len(aliases)
             and set(geometry)==set(aliases.values())|added,
             'joint guard replacement has extra or missing physical stamps')
    layers={key for demand in ir.demands for key in (demand.layer_key,*demand.access_layer_keys)}
    atom_authority=defaultdict(Counter)
    for atom in ir.atoms:
        if atom.stamp_id in replaced:continue
        name=aliases[atom.stamp_id]
        box=_shift(atom.bbox_nm,positions[atom.master_id])
        atom_authority[(name,atom.layer_key,atom.kind,box)][atom.net]+=1
    current=defaultdict(Counter)
    actual_rows=[]
    for conductor in layout_model.conductors:
        if (conductor.source_stamp in added or tuple(conductor.layer) not in layers
                or conductor.kind=='device_projection'):continue
        key=(conductor.source_stamp,tuple(conductor.layer),conductor.kind,_conductor_box(conductor))
        current[key][conductor.net]+=1
        actual_rows.append(conductor)
    _require(set(current)==set(atom_authority), 'actual unchanged conductor geometry denominator differs')
    refinements=[]
    for key,old in atom_authority.items():
        new=current[key]
        _require(sum(old.values())==sum(new.values()), 'actual unchanged conductor geometry multiplicity differs')
        for net,count in old.items():
            if net is not None:
                _require(new[net]>=count, 'actual known source conductor ownership differs')
        remaining=new-old
        _require(sum(remaining.values())<=old[None], 'actual source conductor ownership exceeds original unknown atoms')
        refinements.extend(dict(stamp=key[0],layer_key=key[1],kind=key[2],bbox_nm=key[3],
            original_net=None,actual_net=net,count=count) for net,count in remaining.items())
    inverse={name:old for old,name in aliases.items()}
    original_stamps={stamp.stamp_id:stamp for stamp in ir.stamps}
    atoms=[]
    for conductor in actual_rows:
        name=conductor.source_stamp
        master=original_stamps[inverse[name]].master_id
        origin=positions[master]
        atoms.append(EdgeRailConductorAtom(name,master,tuple(conductor.layer),conductor.net,
            _shift(_conductor_box(conductor),(-origin[0],-origin[1])),conductor.kind))
    for conductor in layout_model.conductors:
        if (conductor.source_stamp not in added or tuple(conductor.layer) not in layers
                or conductor.kind=='device_projection'):
            continue
        name=conductor.source_stamp
        signature=(tuple(conductor.layer),conductor.net,conductor.kind)
        _require(signature in signatures[name], 'actual joint guard conductor net/layer ownership differs')
        master=anchors[name]
        origin=positions[master]
        atoms.append(EdgeRailConductorAtom(name,master,signature[0],signature[1],
            _shift(_conductor_box(conductor),(-origin[0],-origin[1])),signature[2]))
    stamps=[replace(stamp,stamp_id=aliases[stamp.stamp_id]) for stamp in ir.stamps if stamp.stamp_id not in replaced]
    for group in inventory.shared_guards:
        name=group.guard_stamp
        master=anchors[name]
        origin=positions[master]
        box,shape_sha=_stamp_geometry(geometry[name],placement_um[name])
        local=_shift(box,(-origin[0],-origin[1]))
        owned=tuple(atom for atom in atoms if atom.stamp_id==name)
        access=[(atom.layer_key,atom.net,atom.kind,_shift(atom.bbox_nm,(-local[0],-local[1])))
                for atom in owned]
        stamp=geometry[name]
        stamps.append(EdgeRailStampIR(name,master,local[:2],local,shape_sha,_sha(access),len(owned),
            tuple(stamp.guard_ring_declared_members),stamp.guard_ring_intent_sha256))
    # Original FEOL and native access conditions remain in the retained IR and
    # original collision proof. The new entire guard has independent complete
    # foreign-envelope separation; its actual conductors must now satisfy the
    # very same stripe and closed-ring access predicate.
    actual_ir=replace(ir,stamps=tuple(stamps),atoms=tuple(atoms),native_feol_pairs=(),
                      native_access_pair_clearances_nm=())
    _require(actual_ir.replay(dims=dims,positions=positions,physical_bbox_nm=certificate.bbox_nm),
             'actual joint guard violates the original physical rail predicate')
    return dict(schema='clay.actual-shared-guard-edge-rail/v1',original_reservation_sha256=ir.sha256,
        actual_reservation=actual_ir.payload(),actual_reservation_sha256=actual_ir.sha256,
        native_geometry_binding_sha256=binding.sha256,
        unchanged_stamp_aliases=aliases,
        ownership_refinements=refinements,
        replaced_private_guard_stamps=sorted(replaced),actual_shared_guard_stamps=sorted(added))
