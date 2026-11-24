"""Source-bound split-gate implementation of the native precharge row."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

import klayout.db as kdb

from clay.layout.native_sky130 import NativeGeometryBundle, native_source_identity, observe_native_primitive
from clay.pdk.sky130_native_geometry import native_geometry_spec


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def generation_tcl(chain):
    from clay.layout.native_diffusion_pcell import _native_diffusion_generation_tcl
    text = _native_diffusion_generation_tcl(chain)
    if text.count('conn_gates 1') != 1:
        raise ValueError('split gate requires the supported connected-gate source profile')
    return text.replace('conn_gates 1', 'conn_gates 0')


def _contacts(chain, raw, tech):
    from clay.layout.native_buffer_access import _nm_box, _rule
    from clay.layout.native_diffusion_pcell import _native_body_position
    if tech.pdk_id != 'sky130':
        raise ValueError('split gate requires SKY130')
    grid = tech.manufacturing_grid_nm
    if type(grid) is not int or grid <= 0:
        raise ValueError('split gate manufacturing grid differs')
    poly_rules = tech.rules.rule_for('poly'); metal_rules = tech.rules.rule_for('met1')
    rules = (poly_rules.min_width, poly_rules.min_spacing, metal_rules.min_width,
             metal_rules.min_spacing, metal_rules.min_area)
    values = tuple(_rule(rule, identity) for rule, identity in zip(rules,
        ('poly.1a', 'poly.2', 'met1.1', 'met1.2', 'met1.6')))
    poly_width, poly_gap, metal_width, metal_gap, metal_area = values
    layout = kdb.Layout(); layout.read(str(raw)); top = layout.top_cell()
    if layout.dbu != .001 or top is None:
        raise ValueError('split gate raw PCell differs')
    region = lambda layer: kdb.Region(top.begin_shapes_rec(layout.layer(*layer))).merged()
    active, poly, metal, well = (region(key) for key in ((65,20),(66,20),(68,20),(64,20)))
    channels = sorted((active & poly).merged().each(), key=lambda item:item.bbox().left)
    gates = sorted((p for p in poly.each() if not (kdb.Region(p) & active).is_empty()),
                   key=lambda item:item.bbox().left)
    if len(channels) != len(chain.fingers) or len(gates) != len(channels):
        raise ValueError('split gate raw channel or poly inventory differs')
    owners = tuple(f.owner for f in chain.fingers)
    runs = []
    for index, owner in enumerate(owners):
        if not runs or runs[-1][0] != owner: runs.append([owner, index, index+1])
        else: runs[-1][2] = index+1
    if len(runs) != 4 or [end-start for _,start,end in runs] != [1,4,4,1]:
        raise ValueError('split gate source occurrence order differs')
    boundary = runs[1][2]
    groups = (range(0,boundary), range(boundary,len(channels)))
    gate_boxes = [p.bbox() for p in gates]; channel_boxes = [p.bbox() for p in channels]
    if any((kdb.Region(gates[i]) & kdb.Region(channels[i])).is_empty()
           for i in range(len(channels))):
        raise ValueError('split gate channel ownership differs')
    poly_boxes = []
    top_contacts = []
    for group in groups:
        indices = tuple(group)
        bottom = min(gate_boxes[i].bottom for i in indices)
        upper = max(gate_boxes[i].bottom for i in indices)
        if upper-bottom < poly_width or upper >= active.bbox().bottom:
            raise ValueError('split gate poly bridge has no legal interval')
        poly_boxes.append((channel_boxes[indices[0]].left, bottom,
                           channel_boxes[indices[-1]].right, upper))
        contacts=[]
        for i in indices:
            x=(channel_boxes[i].left+channel_boxes[i].right)//2
            hits=[p.bbox() for p in metal.each() if p.bbox().left <= x <= p.bbox().right
                  and p.bbox().bottom >= active.bbox().top]
            if len(hits)>1: raise ValueError('split gate top contact is ambiguous')
            if hits: contacts.append(hits[0])
        if not contacts: raise ValueError('split gate group has no native metal access')
        top_contacts.append(contacts)
    if poly_boxes[1][0] - poly_boxes[0][2] < poly_gap:
        raise ValueError('split gate poly groups violate PDK spacing')
    metal_boxes=[]; fills=[]
    for contacts in top_contacts:
        y0=max(b.bottom for b in contacts); y1=min(b.top for b in contacts)
        if y1-y0 < metal_width: raise ValueError('split gate metal bridge has no legal interval')
        for b in contacts:
            x=(b.left+b.right)//2
            half=max((b.width()+1)//2, ((metal_area + 2*b.height()*grid - 1)//(2*b.height()*grid))*grid)
            fills.append((x-half,b.bottom,x+half,b.top))
        metal_boxes.append((contacts[0].left,y0,contacts[-1].right,y1))
    if min(b[0] for b in fills[len(top_contacts[0]):]) - max(b[2] for b in fills[:len(top_contacts[0])]) < metal_gap:
        raise ValueError('split gate metal groups violate PDK spacing')
    boxes=poly_boxes+fills+metal_boxes
    if any(v % grid for box in boxes for v in box):
        raise ValueError('split gate artwork differs from PDK grid')
    ab=active.bbox(); wb=well.bbox(); bx,by=_native_body_position(ab,'p')
    commands=[_nm_box((wb.left,wb.bottom,ab.right+930,wb.top),'nwell'),
        f'box {bx/1000:.3f}um {by/1000:.3f}um {bx/1000:.3f}um {by/1000:.3f}um',
        'sky130::draw_contact 0.18 0.17 0.12 0.08 0.17 nsd nsc li horz',
        'sky130::draw_contact 0.17 0.17 0.08 0.06 0.17 li mcon metal1 full', 'label B c nsc']
    commands += [_nm_box(box,'poly') for box in poly_boxes]
    commands += [_nm_box(box,'metal1') for box in fills+metal_boxes]
    text='''crashbackups stop
drc off
gds read /work/raw.gds
load primitive
'''+ '\n'.join(commands)+'''
select top cell
gds write contacted.gds
set h [open scale_um.txt w]
puts $h [cif scale out]
close $h
flatten primitive_flat
load primitive_flat
select top cell
drc euclidean on
drc style drc(full)
drc check
drc catchup
set h [open drc.txt w]
puts $h [drc listall why]
close $h
extract path .
extract no capacitance
extract no coupling
extract no resistance
extract no length
extract all
ext2spice lvs
ext2spice merge none
ext2spice -p . -o native.spice
quit -noprompt
'''
    facts={'boundary':boundary,'poly_boxes':poly_boxes,'metal_boxes':metal_boxes,'fills':fills,
           'grid_nm':grid,'rules':[(v,r.rule_id,str(r.source)) for v,r in zip(values,rules)]}
    return text, json.loads(json.dumps(facts))


def _verify_groups(cell, chain, folder, tech):
    from clay.layout.native_diffusion_mosaic import build_native_diffusion_mosaic
    geometry = build_native_diffusion_mosaic(cell,chain,folder,tech)
    binding = geometry.joint_row_binding
    sites = [site for site in binding.expected_sites if site.allowed_roles == ('g',)]
    groups = []
    for component in binding.joint_inventory.joint_components:
        ids = {site.site_id for site in sites}.intersection(component.site_ids)
        if ids:
            if len(component.routing_component_ids) != 1:
                raise ValueError('split gate group has multiple metal components')
            groups.append(ids)
    owners = tuple(f.owner for f in chain.fingers)
    transitions = [i for i in range(1,len(owners)) if owners[i] != owners[i-1]]
    if len(transitions)!=3:
        raise ValueError('split gate source owner transitions differ')
    boundary = transitions[1]
    expected = [{f'{f.owner}[{f.occurrence:03d}]/gate{i}' for i,f in enumerate(chain.fingers) if i < boundary},
                {f'{f.owner}[{f.occurrence:03d}]/gate{i}' for i,f in enumerate(chain.fingers) if i >= boundary}]
    if len(groups) != 2 or {frozenset(g) for g in groups} != {frozenset(g) for g in expected}:
        raise ValueError('split gate physical groups differ from source occurrences')
    return geometry


def materialize(cell,bundle,chain,*,tech,output_dir,timeout_seconds=60,candidate):
    from clay.layout.native_diffusion_pcell import _check_chain, _magic
    from clay.layout.native_diffusion_source import _require_canonical_native_arrays
    from clay.layout.native_row_candidate import PRECHARGE_SPLIT_GATE, require_candidate
    if type(bundle) is not NativeGeometryBundle:
        raise TypeError('split gate requires a complete native bundle')
    if require_candidate(cell,chain,candidate) != PRECHARGE_SPLIT_GATE:
        raise ValueError('split gate candidate differs')
    bundle.require_current(cell); _check_chain(cell,chain)
    source_sha,canonical = _require_canonical_native_arrays(cell); _check_chain(canonical,chain)
    command=json.loads((bundle.output_dir/'command.json').read_text(encoding='utf8'))
    images=[x for x in command if isinstance(x,str) and re.fullmatch(r'sha256:[0-9a-f]{64}',x)]
    runtime=json.loads((bundle.output_dir/'toolchain.json').read_text(encoding='utf8'))
    if len(images)!=1 or not runtime['pdk_root'].endswith('/sky130A') or tech.pdk_id!='sky130':
        raise ValueError('split gate PDK authority differs')
    folder=Path(output_dir).resolve(); folder.mkdir(parents=True,exist_ok=False)
    (folder/'generate.tcl').write_text(generation_tcl(chain),encoding='utf8',newline='\n')
    _magic(folder,images[0],runtime['pdk_root'][:-len('/sky130A')],'generate.tcl','primitive',timeout_seconds)
    shutil.copyfile(folder/'primitive.gds',folder/'raw.gds')
    text,facts=_contacts(chain,folder/'raw.gds',tech)
    (folder/'contacts.tcl').write_text(text,encoding='utf8',newline='\n')
    _magic(folder,images[0],runtime['pdk_root'][:-len('/sky130A')],'contacts.tcl','contacts',timeout_seconds)
    observer=folder/'observer'; observer.mkdir()
    shutil.copyfile(folder/'contacted.gds',observer/'primitive.gds')
    for name in ('drc.txt','native.spice','primitive_flat.ext','scale_um.txt'):
        shutil.copyfile(folder/name,observer/name)
    device=next(d for d in cell.devices if d.instance_name==chain.fingers[0].owner)
    spec=dict(native_geometry_spec(device)); count=len(chain.fingers)
    spec.update(w_nm=count*chain.width_nm,l_nm=chain.length_nm,nf=count,m=1,mult=1,replicas=1,occurrences=count)
    observed=observe_native_primitive(observer,spec)
    _verify_groups(cell,chain,folder,tech)
    bundle.require_current(cell)
    if _require_canonical_native_arrays(cell)[0]!=source_sha:
        raise ValueError('source changed during split gate generation')
    receipt={'source_identity':native_source_identity(cell),'source_sha256':source_sha,'model':chain.model,
        'unit_width_nm':chain.width_nm,'length_nm':chain.length_nm,'image':images[0],
        'native_bundle_manifest_sha256':_sha(bundle.output_dir/'geometry.json'),
        'producer_sha256':_sha(__file__),'candidate_implementation':asdict(candidate),
        'split_rule_evidence':facts,'inputs':{str(Path(cell.source_span['file']).resolve()):source_sha},
        'outputs':{p.name:_sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    (folder/'template-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    return observed,receipt


def admits_producer_identity(receipt):
    digest=receipt.get('producer_sha256') if isinstance(receipt,dict) else None
    return isinstance(digest,str) and bool(re.fullmatch(r'[0-9a-f]{64}',digest)) and digest!='0'*64


def verify_receipt(cell,chain,folder,receipt,tech,*,candidate):
    from clay.layout.native_diffusion_source import _require_canonical_native_arrays
    from clay.layout.native_row_candidate import PRECHARGE_SPLIT_GATE, require_candidate
    if require_candidate(cell,chain,candidate)!=PRECHARGE_SPLIT_GATE:
        raise ValueError('split gate candidate differs')
    source_sha,_=_require_canonical_native_arrays(cell); folder=Path(folder)
    if (receipt.get('candidate_implementation')!=asdict(candidate) or not admits_producer_identity(receipt)
            or receipt.get('source_sha256')!=source_sha or receipt.get('source_identity')!=native_source_identity(cell)):
        raise ValueError('split gate receipt source differs')
    required=('generate.tcl','contacts.tcl','primitive.gds','raw.gds','contacted.gds',
              'drc.txt','native.spice','primitive_flat.ext','scale_um.txt')
    outputs=receipt.get('outputs')
    if type(outputs) is not dict or not set(required)<=set(outputs):
        raise ValueError('split gate receipt inventory incomplete')
    if any(_sha(folder/name)!=outputs[name] for name in required):
        raise ValueError('split gate output differs')
    text,facts=_contacts(chain,folder/'raw.gds',tech)
    if ((folder/'generate.tcl').read_text(encoding='utf8')!=generation_tcl(chain)
            or (folder/'contacts.tcl').read_text(encoding='utf8')!=text
            or receipt.get('split_rule_evidence')!=facts
            or _sha(folder/'primitive.gds')!=_sha(folder/'raw.gds')):
        raise ValueError('split gate PDK or artwork replay differs')
    _verify_groups(cell,chain,folder,tech)
