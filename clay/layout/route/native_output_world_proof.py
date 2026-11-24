"""Scoped source-profile check of actual fixed-stack facts at route-cell entry.

Draft only.  Call after ``build_layout_model`` and ``router_obstacles`` and
before constructing/passing a routing world to the engine.  The source Pin
admission remains a separate caller obligation; this helper never makes Pins.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import klayout.db as kdb

PROFILE='sky130_lvt_single_440_output_m12_v1'
INPUT_OUTPUT_PROFILE='sky130_lvt_single_440_output_input_m12_v1'
SCHEMA='clay.source-fixed-stack-world-proof.v1'

class SourceFixedStackWorldError(ValueError):
    pass

def _need(condition,message):
    if not condition:raise SourceFixedStackWorldError(message)

def _box(row):
    return tuple(int(round(float(getattr(row,k))*1000)) for k in ('x0','y0','x1','y1'))

def _rect(box):
    return kdb.Region(kdb.Box(*box))

def _union(rows):
    region=kdb.Region()
    for row in rows:region.insert(kdb.Box(*_box(row)))
    return region.merged()

def _translate(box,delta):
    return (box[0]+delta[0],box[1]+delta[1],box[2]+delta[0],box[3]+delta[1])

def _fact(row):
    return (tuple(row.layer),_box(row),getattr(row,'owner_net',None),
            getattr(row,'kind',None),getattr(row,'terminal_conductor_ref',None))

@dataclass(frozen=True,slots=True)
class FixedStackWorldProof:
    schema: str
    stamp_id: str
    component_id: str
    source_net: str
    source_owner: str
    column: int
    source_identity: str
    receipt_sha256: str
    terminal_ref: str
    pin_xy_nm: tuple[int,int]
    pad_xy_nm: tuple[int,int,int,int]
    cut_boxes_nm: tuple[tuple[int,int,int,int],...]

def verify_source_fixed_stack_world(*,cell,tech,engine,geometry,placement_um,
        row_program,row_materialization,row_final_certificate,intent,
        layout_model,obstacles,pins_by_net):
    """Return one proof per multi-layer AND component, or fail before routing.

    The source face, real receipt and source policy are re-read.  World shapes
    come only from the actual model and obstacles, not from captured coordinates.
    """
    from clay.layout.joint_placement_terminals import build_placement_terminal_face
    from clay.layout.joint_row_geometry import _region
    from clay.layout.native_diffusion_source import declared_native_chains,_require_native_pcell_witness_operands,_require_canonical_native_arrays
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.native_sky130 import native_source_identity
    from clay.layout import native_gate_440
    from clay.layout.route.terminal_identity import PhysicalTerminalPinBinding
    from clay.layout.route.canonical_input_capture import route_engine_configuration
    from clay.layout.route.engine_registry import route_engine_from_policy
    from clay.layout.route.via_model import via_cut_size_nm,via_enclosure_nm
    from clay.layout.route.resource_world import _via_spacing_nm
    from clay.layout.policy import policy_for_cell
    from clay.parser import parse_clay_file

    _need(bool(getattr(engine,'consumes_fixed_multilayer_joint_terminals',False)) and
          bool(getattr(engine,'consumes_artwork_conductors',False)) and
          bool(getattr(engine,'strict_device_m1_provenance',False)),
          'engine lacks the complete fixed-stack world contract')
    doc=parse_clay_file(cell.source_span['file'])
    source_cells=[d for d in doc.cell_defs if d.local_name==cell.local_name]
    _need(len(source_cells)==1 and native_source_identity(source_cells[0])==native_source_identity(cell),
          'route policy source differs')
    canonical_engine=route_engine_from_policy(policy_for_cell(doc,source_cells[0]))
    _need(route_engine_configuration(engine)==route_engine_configuration(canonical_engine),
          'selected routing engine differs from source policy')
    _need(type(engine.via_redundancy) is int and engine.via_redundancy==2,
          'source policy does not select N=2')
    _need((tuple(tech.metal_key(1)),tuple(tech.metal_key(2)),tuple(tech.via_key(1)),
           via_cut_size_nm(tech,'via1'),_via_spacing_nm(tech,'via1'),tuple(via_enclosure_nm(tech,'via1')))
          ==((68,20),(69,20),(68,44),150,170,(85,85)), 'selected PDK stack differs')
    source_sha,_=_require_canonical_native_arrays(cell)
    projection=row_materialization.source_projection;witness=getattr(projection,'native_witness',None)
    _need(witness is not None,'native source witness missing')
    _require_native_pcell_witness_operands(witness,cell=cell,tech=tech,
        overlay=witness.overlay,source_cir=projection.source_cir)
    chains=dict(declared_native_chains(cell,layout_intent_from_cell_def(cell)))
    placement_face=build_placement_terminal_face(tech=tech,geometry=geometry,placement_um=placement_um,
        row_program=row_program,row_materialization=row_materialization,row_final_certificate=row_final_certificate,
        intent=intent,source_cell=cell)
    all_pins=[pin for pins in pins_by_net.values() for pin in pins]
    obstacle_rows=tuple(obstacles);model_rows=tuple(layout_model.conductors)
    # Ensure the supplied engine world is precisely the just-built model.
    _need(sorted(map(repr,layout_model.router_obstacles()))==sorted(map(repr,obstacle_rows)),
          'engine obstacles differ from actual layout model')
    proofs=[]
    for face in placement_face.joint_faces:
        inventory=face.joint_inventory
        routing={row.component_id:row for row in inventory.routing_components}
        sites={site.site_id:site for site in inventory.expected_sites}
        for component in inventory.joint_components:
            _need(len(component.routing_component_ids)==1,'independent OR access is unsupported')
            access=routing[component.routing_component_ids[0]]
            layers=dict(access.geometry_by_layer_nm)
            if len(layers)==1:continue
            _need(set(layers)=={(68,20),(69,20)},'multi-layer component outside scoped profile')
            chain=chains.get(face.stamp_id)
            _need(chain is not None,'multi-layer component has no source row')
            from clay.layout.native_row_candidate import witness_implementation
            profile,candidate=witness_implementation(cell,chain,witness,face.stamp_id)
            _need(profile in (PROFILE,INPUT_OUTPUT_PROFILE),'multi-layer component is not source profile')
            rows=[row for row in witness.rows if row[0]==face.stamp_id]
            _need(len(rows)==1,'source witness row is missing or ambiguous')
            folder=Path(rows[0][2]);receipt_path=folder/'template-receipt.json'
            receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
            _need(receipt.get('source_sha256')==source_sha and receipt.get('source_identity')==chain.source_identity
                  and (candidate is not None or receipt.get('generation_profile')==chain.generation_profile),'profile receipt source differs')
            if candidate is None:
                native_gate_440.verify_receipt(cell,chain,folder,receipt,tech)
            else:
                native_gate_440.verify_receipt(cell,chain,folder,receipt,tech,candidate=candidate)
            evidence=json.loads((folder/'output-access.json').read_text(encoding='utf-8'))
            _need(evidence.get('profile')==profile and evidence.get('source_identity')==chain.source_identity,
                  'output-access source differs')
            candidates=[]
            for row in evidence.get('outputs',()):
                col=row.get('column'); owner=row.get('source_owner')
                if col not in (2,4,6,8) or owner not in {f.owner for f in chain.fingers}:continue
                expected={f'{chain.fingers[i].owner}[{chain.fingers[i].occurrence:03d}]/sd{col}'
                          for i in (col-1,col)}
                if expected==set(component.site_ids) and row.get('net')==component.source_net:
                    candidates.append((row,'d'))
            if profile==INPUT_OUTPUT_PROFILE:
                inputs=json.loads((folder/'input-access.json').read_text(encoding='utf-8'))
                _need(inputs.get('profile')==INPUT_OUTPUT_PROFILE and inputs.get('source_identity')==chain.source_identity,
                      'input-access source differs')
                for row in inputs.get('gate_landings',()):
                    if {row['gate_site']}==set(component.site_ids) and row['net']==component.source_net:
                        candidates.append((row,'g'))
            _need(len(candidates)==1,'component lacks one exact source output record')
            record,role=candidates[0]; owner=record['source_owner'] if role=='d' else record['owner'];net=record['net']
            col=record['column'] if role=='d' else int(record['gate_site'].rsplit('/gate',1)[1])
            _need(all(sites[s].owner==owner and sites[s].source_net==net and role in sites[s].allowed_roles
                      for s in component.site_ids),'component source drain roles differ')
            matching=[p for p in all_pins if type(getattr(p,'terminal_binding',None)) is PhysicalTerminalPinBinding
                and p.terminal_binding.stamp_id==face.stamp_id and p.terminal_binding.component_id==component.component_id]
            _need(len(matching)==1,'component needs one original Pin')
            pin=matching[0];binding=pin.terminal_binding;binding.__post_init__()
            expected_refs=tuple(sorted((site_id+'/'+role,sites[site_id].owner,role)
                for site_id in component.site_ids for role in sites[site_id].allowed_roles))
            _need(binding.source_refs==expected_refs and pin.net==net and tuple(pin.layer)==(68,20),
                  'Pin source binding or layer differs')
            xy=tuple(pin.xy_nm);_need(len(xy)==2 and all(type(v) is int for v in xy),'Pin coordinate differs')
            delta=face.delta_nm;transform=kdb.Trans(*delta)
            expected_m1=_region(layers[(68,20)]).transformed(transform).merged()
            expected_m2=_region(layers[(69,20)]).transformed(transform).merged()
            seed=_rect((xy[0]-1,xy[1]-1,xy[0]+1,xy[1]+1))
            _need((seed-expected_m1).is_empty(),'Pin is outside exact source M1 component')
            m1_candidates=[c for c in model_rows if c.source_stamp==face.stamp_id and c.net==net and
                c.kind=='device' and tuple(c.layer)==(68,20) and not (_rect(_box(c))&seed).is_empty()]
            refs={c.terminal_conductor_ref for c in m1_candidates if c.terminal_conductor_ref}
            _need(len(refs)==1,'Pin M1 terminal seam missing or ambiguous');ref=next(iter(refs))
            m1=[c for c in model_rows if c.source_stamp==face.stamp_id and c.net==net and c.kind=='device'
                and tuple(c.layer)==(68,20) and c.terminal_conductor_ref==ref]
            m2=[c for c in model_rows if c.source_stamp==face.stamp_id and c.net==net and c.kind=='device_upper'
                and tuple(c.layer)==(69,20) and c.terminal_conductor_ref==ref]
            _need(bool(m1) and bool(m2) and (_union(m1)^expected_m1).is_empty()
                  and (_union(m2)^expected_m2).is_empty(),'exact source M1/M2 world coverage differs')
            pad=_translate(tuple(record['pad_nm']),delta)
            cuts=tuple(sorted(_translate(tuple(box),delta) for box in record['cuts_nm']))
            _need(len(cuts)==4 and len(set(cuts))==4 and pad[2]-pad[0]==pad[3]-pad[1]==640,
                  'output pad/four-cut source geometry differs')
            _need((_rect(pad)^expected_m2).is_empty(),'M2 landing differs from source pad')
            xs=sorted({(box[0]+box[2])//2 for box in cuts});ys=sorted({(box[1]+box[3])//2 for box in cuts})
            _need(len(xs)==len(ys)==2 and xs[1]-xs[0]==ys[1]-ys[0]==320 and
                  all(box[2]-box[0]==box[3]-box[1]==150 for box in cuts),
                  'four cuts do not implement the source N=2 array')
            for box in cuts:
                cut_region=_rect(box)
                _need((cut_region-expected_m1).is_empty() and (cut_region-expected_m2).is_empty(),
                      'source cut does not connect exact M1 and M2')
                same=[c for c in model_rows if tuple(c.layer)==(68,44) and _box(c)==box]
                _need(bool(same) and all(c.source_stamp==face.stamp_id and c.net==net and c.kind=='device_cut'
                      for c in same),'source cut missing or ambiguously attributed in layout model')
                projected=[o for o in obstacle_rows if tuple(o.layer)==(68,44) and _box(o)==box]
                _need(bool(projected) and all(o.owner_net==net and o.kind=='device_cut' and
                      o.terminal_conductor_ref is None for o in projected),'cut missing or ambiguous in engine world')
            # Engine admission coalesces exact rectangle claims; an ambiguous
            # owner/ref or abstract duplicate is not an owned literal atom.
            for conductor in m1+m2:
                same=[o for o in obstacle_rows if tuple(o.layer)==tuple(conductor.layer) and _box(o)==_box(conductor)]
                _need(bool(same) and all(o.owner_net==net and o.kind==conductor.kind and
                      o.terminal_conductor_ref==ref for o in same),'planar world claim is missing or ambiguous')
            proofs.append(FixedStackWorldProof(SCHEMA,face.stamp_id,component.component_id,net,owner,col,
                chain.source_identity,sha256(receipt_path.read_bytes()).hexdigest(),ref,xy,pad,cuts))
    return tuple(proofs)
