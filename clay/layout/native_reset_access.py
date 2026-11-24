"""Source-owned access for the supported native reset row.

The caller must provide the source-declared chain and the already-validated
joint row binding. Candidate metal is generated in the native PCell frame;
the planner preserves the source PCell and uses the selected route policy.
"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import klayout.db as kdb

from clay.layout.joint_row_geometry import joint_row_artwork, joint_row_polygons, replay_joint_row_binding, _region
from clay.layout.route.engine import Obstacle
from clay.layout.route.engine_gridless import _ViaSite, _WireSeg, _RoutedNet
from clay.layout.route.resource_world import pdk_provenance_digest
from clay.layout.native_sky130 import native_source_identity

PROFILE='shared_cross_coupled_m1_v1'

def _bbox(region):
    polys=list(region.merged().each())
    if len(polys)!=1 or not polys[0].is_box():raise ValueError('source M1 component must be one rectangle')
    return polys[0].bbox()

def plan(cell, chain, binding, contacted_path, tech, engine):
    if (tech.pdk_id!='sky130' or chain.source_identity!=native_source_identity(cell)
            or len(chain.fingers)!=6 or len({f.owner for f in chain.fingers})!=6):
        raise ValueError('source chain or PDK is outside supported reset scope')
    if engine.via_redundancy!=2:raise ValueError('source policy must provide the selected N2 route')
    owners=tuple(sorted(f.owner for f in chain.fingers))
    inventory=replay_joint_row_binding(binding,polygons=joint_row_polygons(binding.parts),
                                       consumed_devices=owners,tech=tech)
    if inventory.pdk_provenance_sha256!=pdk_provenance_digest(tech):raise ValueError('PDK provenance differs')
    expected={key:_region(shapes).merged() for key,shapes in joint_row_artwork(binding.parts)}
    layout=kdb.Layout();layout.read(str(contacted_path));top=layout.top_cell()
    if layout.dbu!=0.001 or top is None:raise ValueError('source PCell GDS differs')
    keys=set(expected)|{(layout.get_info(i).layer,layout.get_info(i).datatype) for i in layout.layer_indices()}
    for key in keys:
        actual=kdb.Region(top.begin_shapes_rec(layout.layer(*key))).merged()
        if not (actual^expected.get(key,kdb.Region())).is_empty():raise ValueError('source binding and original PCell masks differ: '+repr(key))
    device={v.instance_name:v for v in cell.devices}
    expected_roles={(s.owner,role,s.source_net) for s in binding.expected_sites for role in s.allowed_roles}
    source_roles={(owner,role,device[owner].pins[role]) for owner in owners for role in ('d','g','s','b')}
    if expected_roles!=source_roles:raise ValueError('source occurrence terminal roles differ')
    sites={s.site_id:s for s in binding.expected_sites}
    components=[]
    for comp in inventory.joint_components:
        layers=dict(comp.geometry_by_layer_nm)
        if (68,20) not in layers:continue
        components.append((comp.source_net, _region(layers[(68,20)]).merged(),
                           tuple(sites[x] for x in comp.site_ids)))
    metal=expected[(68,20)].merged();obstacles=[];rows=[]
    for p in metal.each():
        r=kdb.Region(p);hits=[(n,ss) for n,geo,ss in components if not (geo&r).is_empty()]
        ns={n for n,_ in hits}
        if len(ns)!=1:raise ValueError('M1 source ownership not unique or incomplete')
        net=next(iter(ns));roles=tuple(s for n,ss in hits for s in ss if n==net)
        b=_bbox(r);rows.append((net,b,roles))
        obstacles.append(Obstacle(b.left/1000,b.bottom/1000,b.right/1000,b.top/1000,(68,20),owner_net=net,kind='device'))
    if {(a,b) for a,b in expected if a in (68,69)}!={(68,20)}:
        raise ValueError('multi-layer source requires different terminal authority')
    # Empty-net initialization obtains actual source policy and PDK, not a
    # reimplementation of via dimensions or metal legality.
    box=kdb.Region(top.begin_shapes_rec(layout.layer(65,20))).bbox();margin=max(box.width(),box.height())
    engine.route({},obstacles,[],tech,routing_envelope_nm=(box.left-margin,box.bottom-margin,box.right+margin,box.top+margin))
    halfw,spacing=engine._half_w[0],engine._space[0]; pad_half=max(engine._via_pad_half[0]);grid=tech.manufacturing_grid_nm
    if any(type(v)!=int or v<=0 or v%grid for v in (halfw,spacing,pad_half)):raise ValueError('PDK grid phase unsupported')
    pairs=defaultdict(list)
    for net,box,refs in rows:
        roles={r for s in refs for r in s.allowed_roles}
        if 'd' in roles:pairs[net].append(('d',box,refs))
        if 'g' in roles:pairs[net].append(('g',box,refs))
    target=[]
    for net,vals in sorted(pairs.items()):
        ds=[v for v in vals if v[0]=='d'];gs=[v for v in vals if v[0]=='g']
        if net==chain.body or len(ds)!=1 or len(gs)!=1:continue
        d,g=ds[0][1],gs[0][1]
        def center(a,b):
            if (a+b)%2 or ((a+b)//2)%grid:raise ValueError('contact center off grid')
            return (a+b)//2
        dx=center(d.left,d.right);gx=center(g.left,g.right);gy=center(g.bottom,g.top)
        sy=d.top-halfw if gy>d.top else d.bottom+halfw
        bends={dx,gx};vx={dx,gx};vy={sy,gy}
        for _,b,_ in rows:
            bends.update((b.left-spacing-halfw,b.right+spacing+halfw))
            vx.update((b.left-spacing-pad_half,b.right+spacing+pad_half))
            vy.update((b.bottom-spacing-pad_half,b.top+spacing+pad_half))
        bends=sorted((x for x in bends if x%grid==0),key=lambda x:(abs(x-dx)+abs(x-gx),x))
        vias=sorted(((x,y) for x in vx for y in vy if x%grid==0 and y%grid==0),key=lambda p:(abs(p[0]-gx)+abs(p[1]-gy),p))
        selected=None
        for bend in bends:
            legs=(((dx,sy),(bend,sy)),((bend,sy),(bend,gy)),((bend,gy),(gx,gy)))
            if not all(engine._wire_legal(0,*a,*b,net) for a,b in legs):continue
            segs=[_WireSeg(0,*a,*b) for a,b in legs]; local,_=engine._net_metal(_RoutedNet(net,segs=segs,connected=True))
            land=kdb.Region(g)
            for b in local.get(0,()):land.insert(kdb.Box(*b))
            for x,y in vias:
                if (land&kdb.Region(kdb.Box(x-pad_half,y-pad_half,x+pad_half,y+pad_half))).is_empty():continue
                if not engine._via_legal(x,y,0,net):continue
                route=_RoutedNet(net,segs=segs,vias=[_ViaSite(x,y,0)],connected=True)
                occupied,cuts=engine._net_metal(route);engine._book.add(net,occupied,cuts)
                # Via belongs to the router. The source profile would emit only
                # the source-owned M1 legs, leaving the stack as route option.
                wire_only=engine._emit_net(_RoutedNet(net,segs=segs,connected=True)).wires
                selected={'net':net,'source_drain_sites':[s.site_id for s in ds[0][2] if 'd' in s.allowed_roles],
                          'source_gate_sites':[s.site_id for s in gs[0][2] if 'g' in s.allowed_roles],
                          'm1_legs_nm':legs,'route_owned_candidate_via_nm':(x,y),
                          'm1_polygons_nm':tuple(tuple((q.x,q.y) for q in p.polygon.vertices) for p in wire_only)}
                break
            if selected:break
        if selected is None:raise ValueError('source row has no physical candidate for net '+net)
        target.append(selected)
    if len(target)!=2:raise ValueError('source does not contain two cross-coupled same-net pairs')
    return {'profile':PROFILE,'source_identity':chain.source_identity,'contacted_sha256':sha256(Path(contacted_path).read_bytes()).hexdigest(),
            'pdk_sha256':pdk_provenance_digest(tech),'source_joint_inventory_sha256':inventory.input_sha256,
            'm1_width_nm':2*halfw,'m1_spacing_nm':spacing,'grid_nm':grid,'via_redundancy':engine.via_redundancy,
            'targets':tuple(target),'scope':'isolated local candidate; source materializer and fresh final proof required'}

def _hash(path):return sha256(Path(path).read_bytes()).hexdigest()

def _source_engine(cell):
    from clay.parser import parse_clay_file
    from clay.layout.policy import policy_for_cell
    from clay.layout.route.engine_registry import route_engine_from_policy
    parsed=parse_clay_file(cell.source_span['file']);canonical,=parsed.cell_defs
    if native_source_identity(canonical)!=native_source_identity(cell):raise ValueError('access source changed')
    return route_engine_from_policy(policy_for_cell(parsed,canonical))

def _base_plan(cell,chain,folder,tech):
    from clay.layout.native_diffusion_mosaic import build_native_diffusion_mosaic
    base=Path(folder)/'baseline'
    geometry=build_native_diffusion_mosaic(cell,chain,base,tech)
    return plan(cell,chain,geometry.joint_row_binding,base/'contacted.gds',tech,_source_engine(cell))

def _check_final_masks(baseline,candidate,evidence):
    a=kdb.Layout();a.read(str(baseline));b=kdb.Layout();b.read(str(candidate));at=a.top_cell();bt=b.top_cell()
    if a.dbu!=.001 or b.dbu!=.001 or at is None or bt is None:raise ValueError('source access GDS differs')
    add=kdb.Region()
    for row in evidence['targets']:
        for poly in row['m1_polygons_nm']:add.insert(kdb.Polygon([kdb.Point(*p) for p in poly]))
    keys={(a.get_info(i).layer,a.get_info(i).datatype) for i in a.layer_indices()}
    keys|={(b.get_info(i).layer,b.get_info(i).datatype) for i in b.layer_indices()}
    for key in keys:
        ar=kdb.Region(at.begin_shapes_rec(a.layer(*key))).merged();br=kdb.Region(bt.begin_shapes_rec(b.layer(*key))).merged()
        target=(ar+add).merged() if key==(68,20) else ar
        if not (target^br).is_empty():raise ValueError('source access output masks differ: '+repr(key))

def materialize(cell,bundle,chain,*,tech,output_dir,timeout_seconds=60):
    """Materialize source-bound reset access and fresh PDK evidence."""
    import json,shutil
    from clay.layout.native_diffusion_pcell import materialize_native_diffusion_template,_magic
    from clay.verification.native_sky130 import _input_scripts
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=False)
    base=folder/'baseline'
    materialize_native_diffusion_template(cell,bundle,chain,output_dir=base,timeout_seconds=timeout_seconds)
    evidence=_base_plan(cell,chain,folder,tech)
    layout=kdb.Layout();layout.read(str(base/'contacted.gds'));top=layout.top_cell()
    for row in evidence['targets']:
        for shape in row['m1_polygons_nm']:
            top.shapes(layout.layer(68,20)).insert(kdb.Polygon([kdb.Point(*p) for p in shape]))
    layout.write(str(folder/'contacted.gds'))
    _check_final_masks(base/'contacted.gds',folder/'contacted.gds',evidence)
    for name in ('primitive.gds','raw.gds','generate.tcl','contacts.tcl','scale_um.txt'):
        shutil.copyfile(base/name,folder/name)
    image=next(x for x in json.loads((bundle.output_dir/'command.json').read_text()) if isinstance(x,str) and x.startswith('sha256:'))
    pdk_root=json.loads((bundle.output_dir/'toolchain.json').read_text())['pdk_root'][:-len('/sky130A')]
    (folder/'lvs_extract').mkdir()
    script=_input_scripts('primitive',pdk_root,{'models':{}},{})['lvs.tcl'].replace('/inputs/design.gds','/work/contacted.gds')
    (folder/'extract.tcl').write_text(script,encoding='utf8')
    _magic(folder,image,pdk_root,'extract.tcl','access-extract',timeout_seconds)
    shutil.copyfile(folder/'lvs_extract/primitive.ext',folder/'primitive_flat.ext')
    shutil.copyfile(folder/'lvs_extract/primitive.ext.spc',folder/'native.spice')
    drc='''crashbackups stop\ndrc off\ngds read /work/contacted.gds\nload primitive\nselect top cell\nflatten primitive_flat\nload primitive_flat\nselect top cell\ndrc euclidean on\ndrc style drc(full)\ndrc check\ndrc catchup\nset h [open /work/drc.txt w]\nputs $h [drc listall why]\nclose $h\nputs ACCESS_DRC_COMPLETE\nquit -noprompt\n'''
    (folder/'access_drc.tcl').write_text(drc,encoding='utf8')
    _magic(folder,image,pdk_root,'access_drc.tcl','access-drc',timeout_seconds)
    if (folder/'drc.txt').read_text().strip():raise ValueError('native cross-coupled access failed Magic full DRC')
    receipt=json.loads((base/'template-receipt.json').read_text())
    receipt['access_rule_evidence']=json.loads(json.dumps(evidence))
    receipt['outputs']={p.name:_hash(p) for p in sorted(folder.iterdir()) if p.is_file()}
    receipt['baseline_outputs']={p.name:_hash(p) for p in sorted(base.iterdir()) if p.is_file()}
    (folder/'template-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    return evidence

def validate_receipt(cell,chain,folder,tech,receipt):
    import json
    folder=Path(folder);base=folder/'baseline'
    if receipt.get('access_profile')!=PROFILE:raise ValueError('reset access profile receipt differs')
    if any(_hash(base/name)!=digest for name,digest in receipt.get('baseline_outputs',{}).items()):
        raise ValueError('reset access baseline changed')
    actual=_base_plan(cell,chain,folder,tech)
    if json.loads(json.dumps(actual))!=receipt.get('access_rule_evidence'):raise ValueError('reset access source rule evidence differs')
    _check_final_masks(base/'contacted.gds',folder/'contacted.gds',actual)
    if (folder/'drc.txt').read_text().strip():raise ValueError('reset source access DRC differs')
