"""Consume the production segmented-resistor reference path in artifact checks.

The reference comes entirely from canonical source device/segment declarations
and active PDK recognition data. No extracted geometry or fitted value enters
its generation. The general route-cell reference supplies unchanged MOS, cap,
bulk and BJT semantics; typed resistor topology uses the established flow seam.
"""
from types import SimpleNamespace


def reference_text(cell, *, pdk, top):
    from clay.layout.route.lvs_gate import (
        route_cell_reference_text,_resolved_lvs_config,_cell_def_reference_intent,_fold_passive_banks)
    from clay.layout.generic_placement import _collect_passives,_resistor_segment_intent_map
    from clay.layout.lvs_reference_emit import build_reference_netlist,RefNetlistSpec
    from clay.flow.step import (
        _spice_instance_name,_lvs_ref_profile,_lower_device_card,_expand_segmented_reference_resistors)
    from clay.well_isolation import deep_well_lvs_model_overrides
    declarations=_resistor_segment_intent_map(cell)
    model_overrides=deep_well_lvs_model_overrides(cell,pdk)
    source_devices={_spice_instance_name(d.instance_name,'M'):d for d in cell.devices
                    if d.instance_name in model_overrides}
    if not declarations and not model_overrides:
        return route_cell_reference_text(cell,tech_id=pdk,top_cell=top,full_extraction=True),dict(
            reference_consumer='clay.layout.route.lvs_gate.route_cell_reference_text',resistor_segment_expansions=[])
    original=route_cell_reference_text(cell,tech_id=pdk,top_cell=top,full_extraction=True)
    intent=_cell_def_reference_intent(cell)
    records=_collect_passives(cell,None)
    by_card={_spice_instance_name(name,'R'):(name,record) for name,record in records.items()
             if record['kind']=='res'}
    passives=[]
    for passive in intent.passives:
        fields=dict(vars(passive))
        if passive.kind=='res':
            key='R'+passive.name
            if key not in by_card:raise ValueError('LVS passive identity lost canonical source binding')
            _,record=by_card[key];fields['model']=record.get('model','')
        passives.append(SimpleNamespace(**fields))
    view=SimpleNamespace(devices=intent.devices,passives=passives,bjts=intent.bjts)
    cfg=_resolved_lvs_config(pdk,full_extraction=True)
    text=build_reference_netlist(_fold_passive_banks(view),RefNetlistSpec(
        convention=cfg['convention'],subckt_name=top,mos_models=cfg['devices'],
        fold_parallel=True,pdk_id=pdk,ports=tuple(cell.ports),emit_resistors=cfg.get('res') is not None))
    def cards(value):
        return [line for line in value.splitlines() if line and not line.startswith(('*','.'))]
    if [card for card in cards(text) if not card.startswith('R')]!=[
            card for card in cards(original) if not card.startswith('R')]:
        raise ValueError('segmented reference changed an unrelated source device')
    profile=_lvs_ref_profile(pdk)
    body=[]
    for card in cards(text):
        card_name=card.split()[0]
        if card_name in source_devices:
            # The general bulk-substrate reference normalizes private bodies.
            # Restore the declared terminal before consuming the production
            # deep-well model lowerer. Neither term comes from extracted GDS.
            fields=card.split()
            fields[4]=source_devices[card_name].pins['b']
            lowered,kind=_lower_device_card(' '.join(fields),{},profile,None,model_overrides)
            if kind!='mos' or lowered is None:raise ValueError('private-well reference lacks canonical model lowerer')
            body.append(lowered)
        elif card.startswith('R') and by_card[card_name][0] in declarations:
            lowered,kind=_lower_device_card(card,{},profile,None)
            if kind!='res' or lowered is None:raise ValueError('declared resistor lacks canonical reference lowerer')
            body.append(lowered)
        else:body.append(card)
    body,expansions=_expand_segmented_reference_resistors(body,list(cell.ports),
        layout_cell_def=cell,pdk=pdk,profile=profile)
    return '\n'.join(['.subckt '+top+' '+' '.join(cell.ports),*body,'.ends '+top,'']),dict(
        reference_consumer='clay.flow.step._expand_segmented_reference_resistors',
        original_device_reference_consumer='clay.layout.route.lvs_gate.route_cell_reference_text',
        resistor_segment_expansions=expansions,source_nominal_not_used_as_extracted_measurement=True,
        private_well_model_overrides=model_overrides,
        private_well_body_bindings={name:device.pins['b'] for name,device in source_devices.items()},
        fitted_to_extracted_gds=False)
