"""Source-owned per-gate access prototype, using existing SKY130 helpers.

This isolated module emits a contact Tcl and evidence, never a live patch.
"""
from math import ceil
import klayout.db as kdb

from clay.layout.native_buffer_access import (shared_source_rule_state,
    _native_gate_contact_templates, _draw_contact, _nm_box)

from clay.layout import native_gate_440_planner as P


def build(raw_path,chain,pins,tech,unit_path, *, implementation=None):
    unit=P.inspect_unit(unit_path); plan=P.plan(chain,pins,unit,implementation=implementation);P.check_seams(plan,unit)
    grid,rules,templates=shared_source_rule_state(tech)
    r={rule_id:value for value,rule_id,_provenance in rules}
    if grid!=5:raise P.UnsupportedComposition('unexpected PDK grid')
    layout=kdb.Layout();layout.read(str(raw_path));top=layout.top_cell()
    if top is None or layout.dbu!=.001:raise P.UnsupportedComposition('row GDS top or DBU differs')
    def region(layer):
        idx=layout.find_layer(*layer)
        return kdb.Region() if idx is None else kdb.Region(top.begin_shapes_rec(idx)).merged()
    active,poly=region((65,20)),region((66,20));channels=sorted((active&poly).merged().each(),key=lambda p:p.bbox().left)
    gates=sorted(poly.each(),key=lambda p:p.bbox().left)
    if len(channels)!=10 or len(gates)!=10:raise P.UnsupportedComposition('raw gate/channel inventory differs')
    boxes=[p.bbox() for p in channels]
    if any((b.width(),b.height())!=(150,2000) for b in boxes) or [boxes[i+1].left-boxes[i].left for i in range(9)]!=[440]*9:
        raise P.UnsupportedComposition('raw source W/L or gate pitch differs')
    for gate,ch in zip(gates,channels):
        if ((kdb.Region(gate)&active)^kdb.Region(ch)).area()!=0:
            raise P.UnsupportedComposition('raw gate/source channel binding differs')
    poly_contact,metal_contact=_native_gate_contact_templates(r['mcon.1'],grid)
    contact_width=poly_contact[0]+2*poly_contact[2]
    contact_height=poly_contact[1]+2*poly_contact[2]
    if contact_width%2 or contact_height%2:raise P.UnsupportedComposition('contact grid phase differs')
    poly_width=r['poly.1a'];poly_gap=r['poly.2']
    if poly_width%2 or poly_gap%grid:raise P.UnsupportedComposition('poly rule phase differs')
    high=max(g.bbox().top for g in gates);low=min(g.bbox().bottom for g in gates)
    if any(v%grid for v in (high,low,poly_width,contact_width,contact_height)):
        raise P.UnsupportedComposition('actual gate boundary off grid')
    pad_h=max(r['met1.1'],r['mcon.1']+2*r['met1.4'])
    if pad_h%2:raise P.UnsupportedComposition('metal pad phase differs')
    ceil_grid=lambda n:((n+grid-1)//grid)*grid
    pad_half=max(ceil_grid(ceil(r['met1.6']/(2*pad_h))),ceil_grid((r['mcon.1']+2*r['met1.4']+1)//2))
    gate_access=[];points=[];lines=[];predicted=poly.dup()
    for i,(gate,ch,item) in enumerate(zip(gates,channels,plan)):
        box=ch.bbox();gbox=gate.bbox();x=(box.left+box.right)//2
        if x%grid:raise P.UnsupportedComposition('source gate center off grid')
        if i%2:
            inner=high+poly_gap; y=inner+contact_height//2
            extension=(x-poly_width//2,gbox.top-grid,x+poly_width//2,inner+grid)
        else:
            inner=low-poly_gap; y=inner-contact_height//2
            extension=(x-poly_width//2,inner-grid,x+poly_width//2,gbox.bottom+grid)
        if any(v%grid for v in (*extension,x,y)):
            raise P.UnsupportedComposition('derived gate contact off grid')
        pad=(x-contact_width//2,y-contact_height//2,x+contact_width//2,y+contact_height//2)
        metal=(x-pad_half,y-pad_h//2,x+pad_half,y+pad_h//2)
        predicted.insert(kdb.Box(*extension));predicted.insert(kdb.Box(*pad))
        lines += [_nm_box(extension,'poly'),
                  f'box {x/1000:.3f}um {y/1000:.3f}um {x/1000:.3f}um {y/1000:.3f}um',
                  _draw_contact(poly_contact,'poly pc li horz'),
                  _draw_contact(metal_contact,'li mcon metal1 horz'),_nm_box(metal,'metal1')]
        points.append((x,y));gate_access.append({'owner':item['owner'],'occurrence':item['occurrence'],
            'gate':item['gate'],'poly_extension_nm':extension,'poly_pad_nm':pad,'metal_pad_nm':metal,
            'contact_center_nm':(x,y)})
    components=list(predicted.merged().each())
    assignments=[]
    for channel in channels:
        hits=[idx for idx,p in enumerate(components) if not (kdb.Region(p)&kdb.Region(channel)).is_empty()]
        if len(hits)!=1:raise P.UnsupportedComposition('gate contact disconnected or ambiguous')
        assignments.append(hits[0])
    if len(set(assignments))!=10:raise P.UnsupportedComposition('independent gate polygons merged')
    if not predicted.merged().space_check(poly_gap).is_empty():
        raise P.UnsupportedComposition('predicted poly clearance violates PDK rule')
    labels=P.extraction_labels(plan,unit,tuple(points))
    for name,x,y,_net in labels:
        lines += [f'box {x/1000:.3f}um {y/1000:.3f}um {x/1000:.3f}um {y/1000:.3f}um',f'label {name} c li']
    return '\n'.join(lines),{'gate_access':gate_access,'physical_labels':labels,
        'rule_state':tuple((value,rule_id,str(provenance)) for value,rule_id,provenance in rules),
        'poly_contact_nm':templates[0],'metal_contact_nm':templates[1],
        'manufacturing_grid_nm':grid,'predicted_poly_clearance_nm':poly_gap,
        'scope':'source-owned access candidate; actual GDS and physical signoff required'}
