"""Draft source-owned M1/M2 four-output access for the 440 nm LVT row.

This file is isolated from live Clay.  It derives columns from the validated
source chain and actual contacted PDK row, not from placed world coordinates.
Placement and multi-layer joint pin consumption remain separate obligations.
"""
from hashlib import sha256
from pathlib import Path
import json
import klayout.db as kdb

from clay.layout.route.via_model import via_cut_size_nm, via_enclosure_nm
from clay.layout.route.resource_world import _via_spacing_nm, pdk_provenance_digest

PROFILE = 'sky130_lvt_single_440_output_m12_v1'
INPUT_PROFILE = 'sky130_lvt_single_440_output_input_m12_v1'

def _region(layout, top, key):
    index = layout.find_layer(*key)
    return kdb.Region() if index is None else kdb.Region(top.begin_shapes_rec(index)).merged()

def plan(cell, chain, contacted_path, tech, *, candidate=None):
    if candidate is None:
        profile=chain.generation_profile
    else:
        from clay.layout.native_row_candidate import require_candidate
        profile=require_candidate(cell,chain,candidate)
    if profile not in (PROFILE,INPUT_PROFILE) or tech.pdk_id != 'sky130':
        raise ValueError('source generation profile or PDK differs')
    # This pattern is deliberately source-specific: four distinct physical
    # drain components, two nets, interspersed with the common input source.
    cols=chain.columns
    if (len(cols)!=11 or cols[2]!=cols[4] or cols[6]!=cols[8] or
            len({cols[0],cols[2],cols[6]})!=3 or
            any(cols[i]!=cols[0] for i in (0,1,3,5,7,9,10))):
        raise ValueError('source output/tail column topology differs')
    pins={d.instance_name:dict(d.pins) for d in cell.devices}
    for i in (2,4,6,8):
        for f in (chain.fingers[i-1],chain.fingers[i]):
            if (f.owner != chain.fingers[1 if i<5 else 5].owner or
                    pins[f.owner].get('d')!=cols[i] or pins[f.owner].get('s')!=cols[0]):
                raise ValueError('output column is not a shared source drain')
    cut=via_cut_size_nm(tech,'via1'); space=_via_spacing_nm(tech,'via1')
    encl=tuple(via_enclosure_nm(tech,'via1'))
    if (cut,space,encl,tuple(tech.via_key(1))) != (150,170,(85,85),(68,44)):
        raise ValueError('draft access is limited to the observed SKY130 N2 via')
    layout=kdb.Layout();layout.read(str(contacted_path));top=layout.top_cell()
    if layout.dbu!=.001 or top is None: raise ValueError('row GDS DBU/top differs')
    active=_region(layout,top,(65,20));poly=_region(layout,top,(66,20))
    diff=sorted((active-poly).merged().each(),key=lambda p:p.bbox().left)
    if len(diff)!=11: raise ValueError('source diffusion denominator differs')
    m1=_region(layout,top,(68,20)); out=[]; b=active.bbox()
    for i in (2,4,6,8):
        d=diff[i].bbox()
        hits=[p.bbox() for p in (m1 & kdb.Region(d)).merged().each()]
        if len(hits)!=1 or (hits[0].width(),hits[0].height())!=(230,2000):
            raise ValueError('actual source output M1 differs')
        t=hits[0].left; u=b.top
        m1_boxes=[(t,u-140,t+270,u),(t+130,u-70,t+270,u+255),
                  (t+130,u+115,t+360,u+255),(t+220,u+115,t+360,u+900)]
        pad=(t-30,u+760,t+610,u+1400)
        if pad[2]-pad[0] != cut+space+cut+2*max(encl):
            raise ValueError('N2 pad is inconsistent with PDK via rules')
        cx,cy=t+290,u+1080; offset=(cut+space)//2; half=cut//2
        cuts=[(cx+dx-half,cy+dy-half,cx+dx+half,cy+dy+half)
              for dx in (-offset,offset) for dy in (-offset,offset)]
        out.append({'column':i,'net':cols[i],'source_owner':chain.fingers[i-1].owner,
                    'original_m1_bbox_nm':(hits[0].left,hits[0].bottom,hits[0].right,hits[0].top),
                    'm1_boxes_nm':m1_boxes,'pad_nm':pad,'cuts_nm':cuts,
                    'landing_layer':(69,20),'landing_center_nm':(cx,cy)})
    return {'profile':profile,'source_identity':chain.source_identity,'outputs':out,
            'pdk_provenance_sha256':pdk_provenance_digest(tech),'cut_size_nm':cut,
            'cut_spacing_nm':space,'enclosure_nm':encl,
            'scope':'source-derived row access; fresh placement/route verification required'}

def emit(cell, chain, original_path, destination, tech, *, candidate=None):
    evidence=plan(cell,chain,original_path,tech,candidate=candidate)
    layout=kdb.Layout();layout.read(str(original_path));top=layout.top_cell()
    for output in evidence['outputs']:
        for box in output['m1_boxes_nm']+[output['pad_nm']]:
            top.shapes(layout.layer(68,20)).insert(kdb.Box(*box))
        top.shapes(layout.layer(69,20)).insert(kdb.Box(*output['pad_nm']))
        for box in output['cuts_nm']:
            top.shapes(layout.layer(68,44)).insert(kdb.Box(*box))
    layout.write(str(destination))
    evidence['original_sha256']=sha256(Path(original_path).read_bytes()).hexdigest()
    evidence['output_sha256']=sha256(Path(destination).read_bytes()).hexdigest()
    return evidence

def verify(cell, chain, original_path, candidate_path, evidence, tech, *, candidate=None):
    expected=plan(cell,chain,original_path,tech,candidate=candidate)
    for k,v in expected.items():
        if json.loads(json.dumps(v)) != evidence.get(k): raise ValueError('source access evidence differs: '+k)
    if evidence.get('original_sha256')!=sha256(Path(original_path).read_bytes()).hexdigest():
        raise ValueError('base contacted template changed')
    if evidence.get('output_sha256')!=sha256(Path(candidate_path).read_bytes()).hexdigest():
        raise ValueError('access template changed')
    a=kdb.Layout();a.read(str(original_path));at=a.top_cell()
    b=kdb.Layout();b.read(str(candidate_path));bt=b.top_cell()
    expected_add={}
    for item in expected['outputs']:
        for key,boxes in (((68,20),item['m1_boxes_nm']+[item['pad_nm']]),
                          ((69,20),[item['pad_nm']]),((68,44),item['cuts_nm'])):
            reg=expected_add.setdefault(key,kdb.Region())
            for box in boxes:reg.insert(kdb.Box(*box))
    keys={(a.get_info(i).layer,a.get_info(i).datatype) for i in a.layer_indices()}
    keys|={(b.get_info(i).layer,b.get_info(i).datatype) for i in b.layer_indices()}
    for key in keys:
        orig=_region(a,at,key);actual=_region(b,bt,key)
        planned=(orig+expected_add.get(key,kdb.Region())).merged()
        if not (planned^actual).is_empty(): raise ValueError('access layer differs: '+repr(key))
    return expected
