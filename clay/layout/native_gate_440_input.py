"""Source-declared fixed N=2 gate landing on the lower 440-nm LVT contacts.

The parent native gate producer supplies the current real output-profile GDS
and its gate-access recipe. No placed coordinates or route book are used.
"""
from hashlib import sha256
from pathlib import Path
import json
import klayout.db as kdb

PROFILE='sky130_lvt_single_440_output_input_m12_v1'

def _sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def _region(layout,top,key):
 i=layout.find_layer(*key)
 return kdb.Region() if i is None else kdb.Region(top.begin_shapes_rec(i)).merged()

def plan(cell,chain,contacted_path,gate_access_path,tech, *, candidate=None):
    from clay.parser import parse_clay_file
    from clay.layout.policy import policy_for_cell
    from clay.layout.route.engine_registry import route_engine_from_policy
    from clay.layout.native_sky130 import native_source_identity
    from clay.layout.route.via_model import via_cut_size_nm,via_enclosure_nm
    from clay.layout.route.resource_world import _via_spacing_nm,pdk_provenance_digest
    if candidate is None:
        profile=chain.generation_profile
    else:
        from clay.layout.native_row_candidate import require_candidate
        profile=require_candidate(cell,chain,candidate)
    if profile!=PROFILE or tech.pdk_id!='sky130':raise ValueError('source input profile or PDK differs')
    doc=parse_clay_file(cell.source_span['file']);source,=tuple(c for c in doc.cell_defs if c.local_name==cell.local_name)
    if native_source_identity(source)!=chain.source_identity:raise ValueError('source identity differs')
    engine=route_engine_from_policy(policy_for_cell(doc,source))
    if type(engine.via_redundancy) is not int or engine.via_redundancy!=2:raise ValueError('source policy requires N2')
    cut=via_cut_size_nm(tech,'via1');spacing=_via_spacing_nm(tech,'via1');enclosure=tuple(via_enclosure_nm(tech,'via1'))
    grid=tech.manufacturing_grid_nm
    if (cut,spacing,enclosure,tuple(tech.via_key(1)),grid)!=(150,170,(85,85),(68,44),5):raise ValueError('source PDK stack unsupported')
    gates=json.loads(Path(gate_access_path).read_text())['gate_access']
    if len(gates)!=len(chain.fingers):raise ValueError('source gate inventory differs')
    layout=kdb.Layout();layout.read(str(contacted_path));top=layout.top_cell()
    if layout.dbu!=.001 or top is None:raise ValueError('gate base GDS differs')
    metal=_region(layout,top,(68,20));active=_region(layout,top,(65,20));bbox=active.bbox()
    pad_side=2*cut+spacing+2*max(enclosure);half=pad_side//2;pitch=(cut+spacing)//2;cut_half=cut//2
    if any(v%grid for v in (pad_side,half,pitch,cut_half)):raise ValueError('PDK grid phase differs')
    rows=[]
    for index,(finger,gate) in enumerate(zip(chain.fingers,gates)):
        if (gate['owner'],gate['occurrence'],gate['gate'])!=(finger.owner,finger.occurrence,finger.gate):raise ValueError('gate source role differs')
        g=tuple(gate['metal_pad_nm']);x,y=gate['contact_center_nm']
        if not (kdb.Region(kdb.Box(*g))-metal).is_empty():raise ValueError('source gate contact differs')
        if g[3]>=bbox.bottom:continue
        # Align the square via pad's upper edge to the source metal pad top.
        # Its center and dimensions are derived entirely from source/PDK.
        cy=g[3]-half;pad=(x-half,cy-half,x+half,cy+half)
        if any(v%grid for v in (*pad,x,cy)):raise ValueError('gate stack is off grid')
        cuts=[(x+dx-cut_half,cy+dy-cut_half,x+dx+cut_half,cy+dy+cut_half) for dx in (-pitch,pitch) for dy in (-pitch,pitch)]
        if (kdb.Region(kdb.Box(*g))&kdb.Region(kdb.Box(*pad))).is_empty():raise ValueError('gate stack lacks source contact')
        rows.append({'owner':finger.owner,'occurrence':finger.occurrence,'net':finger.gate,
            'gate_site':f'{finger.owner}[{finger.occurrence:03d}]/gate{index}',
            'source_gate_m1_nm':g,'pad_nm':pad,'cuts_nm':cuts,'landing_center_nm':(x,cy)})
    if len(rows)!=5 or len({r['net'] for r in rows})!=2:raise ValueError('lower source gate inventory differs')
    return {'profile':PROFILE,'source_identity':chain.source_identity,'source_sha256':_sha(cell.source_span['file']),'base_sha256':_sha(contacted_path),
        'gate_access_sha256':_sha(gate_access_path),'pdk_sha256':pdk_provenance_digest(tech),
        'via_redundancy':2,'gate_landings':rows,'scope':'source row candidate; placement and route remain required'}

def emit(cell,chain,base,gate_access,destination,tech, *, candidate=None):
    evidence=plan(cell,chain,base,gate_access,tech,candidate=candidate);ly=kdb.Layout();ly.read(str(base));top=ly.top_cell()
    for row in evidence['gate_landings']:
        for key in ((68,20),(69,20)):top.shapes(ly.layer(*key)).insert(kdb.Box(*row['pad_nm']))
        for cut in row['cuts_nm']:top.shapes(ly.layer(68,44)).insert(kdb.Box(*cut))
    ly.write(str(destination));evidence['output_sha256']=_sha(destination);return evidence

def verify(cell,chain,base,gate_access,candidate_path,evidence,tech, *, candidate=None):
    expected=plan(cell,chain,base,gate_access,tech,candidate=candidate)
    for key,value in expected.items():
        if json.loads(json.dumps(value))!=evidence.get(key):raise ValueError('source input evidence differs '+key)
    if evidence.get('output_sha256')!=_sha(candidate_path):raise ValueError('gate output hash differs')
    a=kdb.Layout();a.read(str(base));b=kdb.Layout();b.read(str(candidate_path));at=a.top_cell();bt=b.top_cell()
    additions={}
    for row in expected['gate_landings']:
        for key in ((68,20),(69,20)):additions.setdefault(key,kdb.Region()).insert(kdb.Box(*row['pad_nm']))
        for cut in row['cuts_nm']:additions.setdefault((68,44),kdb.Region()).insert(kdb.Box(*cut))
    keys={(a.get_info(i).layer,a.get_info(i).datatype) for i in a.layer_indices()}|{(b.get_info(i).layer,b.get_info(i).datatype) for i in b.layer_indices()}
    for key in keys:
        if not ((_region(a,at,key)+additions.get(key,kdb.Region())).merged()^_region(b,bt,key)).is_empty():raise ValueError('gate stack actual masks differ '+repr(key))
    return expected
