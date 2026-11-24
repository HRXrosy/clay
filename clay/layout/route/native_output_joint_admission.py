"""Scoped source-owned four-cut output face admission for the 440 nm row."""
from pathlib import Path
import json
import klayout.db as kdb

PROFILE = 'sky130_lvt_single_440_output_m12_v1'
INPUT_OUTPUT_PROFILE = 'sky130_lvt_single_440_output_input_m12_v1'

def require_output_profile_face(face, operands, engine):
    """Replay the declared row, its real PDK output receipt and all four sites.

    This admits a representative pin only for the exact four-output source
    profile. The actual placed world remains a separate caller obligation.
    """
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.native_diffusion_source import declared_native_chains, _require_native_pcell_witness_operands
    from clay.layout import native_gate_440
    from clay.layout.joint_row_geometry import _region
    from clay.layout.policy import policy_for_cell
    from clay.layout.route.engine_registry import route_engine_from_policy
    from clay.layout.route.canonical_input_capture import route_engine_configuration
    from clay.parser import parse_clay_file
    cell=operands.get('source_cell');tech=operands['tech'];materialization=operands.get('row_materialization')
    if cell is None or materialization is None or engine is None:
        raise ValueError('source output multilayer face requires source, placement and engine')
    chains=dict(declared_native_chains(cell,layout_intent_from_cell_def(cell)))
    chain=chains.get(face.stamp_id)
    if chain is None:
        raise ValueError('multilayer face is not the declared source output profile')
    doc=parse_clay_file(cell.source_span['file'])
    canonical=[c for c in doc.cell_defs if c.local_name==cell.local_name]
    if len(canonical)!=1:raise ValueError('source output policy cell is ambiguous')
    selected=route_engine_from_policy(policy_for_cell(doc,canonical[0]))
    from clay.layout.route.source_engine_record import (SourceEngineRecord, freeze_source_engine,
        require_source_engine_with_recorded_experiments)
    record = engine if type(engine) is SourceEngineRecord else freeze_source_engine(engine)
    require_source_engine_with_recorded_experiments(record, selected)
    if type(selected.via_redundancy) is not int or selected.via_redundancy != 2:
        raise ValueError('source output requires the selected N=2 physical-artwork engine')
    projection=materialization.source_projection;witness=getattr(projection,'native_witness',None)
    if witness is None:raise ValueError('source output face lacks native witness')
    _require_native_pcell_witness_operands(witness,cell=cell,tech=tech,overlay=witness.overlay,source_cir=projection.source_cir)
    from clay.layout.native_row_candidate import witness_implementation
    profile,candidate=witness_implementation(cell,chain,witness,face.stamp_id)
    if profile not in (PROFILE,INPUT_OUTPUT_PROFILE):
        raise ValueError('multilayer face is not the declared source output profile')
    rows=[r for r in witness.rows if r[0]==face.stamp_id]
    if len(rows)!=1:raise ValueError('source output witness row is ambiguous')
    folder=Path(rows[0][2]);receipt=json.loads((folder/'template-receipt.json').read_text())
    if candidate is None:
        native_gate_440.verify_receipt(cell,chain,folder,receipt,tech)
    else:
        native_gate_440.verify_receipt(cell,chain,folder,receipt,tech,candidate=candidate)
    evidence=json.loads((folder/'output-access.json').read_text());outputs=evidence['outputs']
    if len(outputs)!=4 or [x['column'] for x in outputs]!=[2,4,6,8]:
        raise ValueError('source output column inventory differs')
    layout=kdb.Layout();layout.read(str(folder/'contacted.gds'));top=layout.top_cell()
    if layout.dbu!=.001 or top is None:raise ValueError('source output template differs')
    def region(layer):
        index=layout.find_layer(*layer)
        return kdb.Region() if index is None else kdb.Region(top.begin_shapes_rec(index)).merged()
    m1,m2,cuts=(region(layer) for layer in ((68,20),(69,20),(68,44)))
    inv=face.joint_inventory;sites={x.site_id:x for x in inv.expected_sites};result=[]
    for row in outputs:
        owner,net,col=row['source_owner'],row['net'],row['column']
        component,=tuple(c for c in inv.joint_components if c.source_net==net and any(s.startswith(owner+'[') and s.endswith('/sd'+str(col)) for s in c.site_ids))
        matching=tuple(sorted(s.site_id for s in inv.expected_sites if s.owner==owner and s.source_net==net and 'd' in s.allowed_roles and s.site_id.endswith('/sd'+str(col))))
        if len(matching)!=2 or tuple(component.site_ids)!=matching or len(component.routing_component_ids)!=1:
            raise ValueError('source output component/occurrence ownership differs')
        access,=tuple(a for a in inv.routing_components if a.component_id in component.routing_component_ids)
        layers=dict(access.geometry_by_layer_nm)
        if set(layers)!={(68,20),(69,20)}:raise ValueError('source output physical routing layers differ')
        pad=kdb.Region(kdb.Box(*row['pad_nm']))
        if any(not (pad-region).is_empty() for region in (m1,m2,_region(layers[(68,20)]),_region(layers[(69,20)]))):
            raise ValueError('source output landing is not on both actual owned metals')
        if len(row['cuts_nm'])!=4 or any(not (kdb.Region(kdb.Box(*cut))-cuts).is_empty() for cut in row['cuts_nm']):
            raise ValueError('source output fixed 2x2 via cuts differ')
        result.append(component.component_id)
    if profile==INPUT_OUTPUT_PROFILE:
        input_evidence=json.loads((folder/'input-access.json').read_text())
        if input_evidence.get('profile')!=INPUT_OUTPUT_PROFILE or input_evidence.get('source_identity')!=chain.source_identity:
            raise ValueError('source input stack evidence differs')
        gate_rows=input_evidence.get('gate_landings',())
        if len(gate_rows)!=5:raise ValueError('source input gate inventory differs')
        for row in gate_rows:
            owner,net,site=row['owner'],row['net'],row['gate_site']
            component,=tuple(c for c in inv.joint_components if c.source_net==net and site in c.site_ids)
            if tuple(component.site_ids)!=(site,) or len(component.routing_component_ids)!=1:
                raise ValueError('source gate component ownership differs')
            source_site=sites[site]
            if source_site.owner!=owner or source_site.source_net!=net or 'g' not in source_site.allowed_roles:
                raise ValueError('source gate role differs')
            access,=tuple(a for a in inv.routing_components if a.component_id in component.routing_component_ids)
            layers=dict(access.geometry_by_layer_nm)
            if set(layers)!={(68,20),(69,20)}:raise ValueError('source gate physical layers differ')
            pad=kdb.Region(kdb.Box(*row['pad_nm']))
            if any(not (pad-region).is_empty() for region in (m1,m2,_region(layers[(68,20)]),_region(layers[(69,20)]))):
                raise ValueError('source gate pad is not on both owned metals')
            if len(row['cuts_nm'])!=4 or any(not (kdb.Region(kdb.Box(*cut))-cuts).is_empty() for cut in row['cuts_nm']):
                raise ValueError('source gate four cuts differ')
            result.append(component.component_id)
    actual={c.component_id for c in inv.joint_components for a in inv.routing_components
            if a.component_id in c.routing_component_ids and len(a.geometry_by_layer_nm)>1}
    if set(result)!=actual:raise ValueError('source output multilayer face includes unproved components')
    return tuple(result)
