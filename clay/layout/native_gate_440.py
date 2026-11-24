"""Typed, source-declared SKY130 LVT single-unit row producer and replay.

Isolated maintainer overlay only. The variant has its own producer identity,
PDK-unit invocations, template key and complete source/geometry evidence.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

import klayout.db as kdb

from clay.layout import native_gate_440_access as access
from clay.layout import native_gate_440_composer as composer
from clay.layout import native_gate_440_planner as planner
from clay.layout.native_sky130 import NativeGeometryBundle, native_source_identity, observe_native_primitive
from clay.pdk.sky130_native_geometry import native_geometry_spec

PROFILE='sky130_lvt_single_440_v1'
OUTPUT_PROFILE='sky130_lvt_single_440_output_m12_v1'
INPUT_OUTPUT_PROFILE='sky130_lvt_single_440_output_input_m12_v1'
PINNED_IMAGE='sha256:8d846d760c47d35f961871b7e958bb03fe7a0bef4cef0d9177721d0a3535eba1'
REFERENCE_HASHES={'ports0.gds':'1095f38fa872677b1ffc9d1eae7c4b207f35204765c590f7c2a95a244a6560d4',
                  'ports1.gds':'742cf6b19f0c919caf42ce076d8a2525359d3d59c95c554d312a181e9ea18e7e',
                  'observation.json':'77f1e731150878f52cc920f4b8b223084875742cdc4adaabb851aedf57a14aa3'}
def _sha(p):return sha256(Path(p).read_bytes()).hexdigest()


def reference_paths():
    return tuple(Path(__file__).with_name('native_gate_440_reference')/name for name in sorted(REFERENCE_HASHES))


def _labels(layout,top):
    top.flatten(-1,False)
    return tuple(sorted((layout.get_info(i).layer,layout.get_info(i).datatype,shape.text.string,str(shape.text.trans))
        for i in layout.layer_indices() for shape in top.shapes(i).each() if shape.is_text()))


def verify_unit_reference(labeled,noports,image):
    """Compare new units to a retained, independently observed pinned PDK unit."""
    if image!=PINNED_IMAGE:raise ValueError('typed native unit image is unsupported')
    for path in reference_paths():
        if _sha(path)!=REFERENCE_HASHES[path.name]:raise ValueError('typed PDK reference evidence differs')
    ref=Path(__file__).with_name('native_gate_440_reference')
    for actual_path,reference_path in ((Path(labeled),ref/'ports1.gds'),(Path(noports),ref/'ports0.gds')):
        al,at=composer.read(actual_path);bl,bt=composer.read(reference_path)
        ar,br=composer.regions(al,at),composer.regions(bl,bt)
        for key in set(ar)|set(br):
            if not (ar.get(key,kdb.Region())^br.get(key,kdb.Region())).is_empty():
                raise ValueError('fresh PDK unit differs from validated native unit')
        if _labels(al,at)!=_labels(bl,bt):raise ValueError('fresh PDK unit labels differ from validated native unit')


def unit_tcl(chain,ports, *, implementation=None):
    profile=chain.generation_profile if implementation is None else implementation
    if ports not in (0,1) or profile not in (PROFILE,OUTPUT_PROFILE,INPUT_OUTPUT_PROFILE):
        raise ValueError('typed native unit invocation differs')
    return f'''crashbackups stop
drc off
load primitive
box 0 0 0 0
set params {{w {chain.width_nm/1000:.3f} l {chain.length_nm/1000:.3f} m 1 nf 1 guard 0 conn_gates 0 topc 0 botc 0 doports {ports} viagb 100}}
magic::gencell sky130::{chain.model} U0 -spice {{*}}$params
select top cell
gds write primitive.gds
puts GENERATED
quit -noprompt
'''


def _operands(cell,chain):
    from clay.layout.native_diffusion_pcell import _check_chain
    from clay.layout.native_diffusion_source import _require_canonical_native_arrays
    _source_sha,canonical=_require_canonical_native_arrays(cell)
    _check_chain(cell,chain);_check_chain(canonical,chain)
    pins={device.instance_name:dict(device.pins) for device in cell.devices}
    return _source_sha,asdict(chain),pins


def contacts_tcl(cell,chain,raw_gds,unit_labeled,tech, *, candidate=None):
    from clay.layout.native_diffusion_pcell import _native_body_position
    _sha_source,data,pins=_operands(cell,chain)
    layout=kdb.Layout();layout.read(str(raw_gds));top=layout.top_cell()
    if top is None or layout.dbu!=.001:raise ValueError('typed native raw GDS differs')
    active=kdb.Region(top.begin_shapes_rec(layout.layer(65,20))).merged()
    bx,by=_native_body_position(active.bbox(),'n')
    implementation=None
    if candidate is not None:
        from clay.layout.native_row_candidate import require_candidate
        implementation=require_candidate(cell,chain,candidate)
    text,evidence=access.build(raw_gds,data,pins,tech,unit_labeled,implementation=implementation)
    script=f'''crashbackups stop
drc off
gds read /work/raw.gds
load primitive
box {bx/1000:.3f}um {by/1000:.3f}um {bx/1000:.3f}um {by/1000:.3f}um
sky130::draw_contact 0.18 0.17 0.12 0.08 0.17 psd psc li horz
sky130::draw_contact 0.17 0.17 0.08 0.06 0.17 li mcon metal1 full
label B c psc
{text}
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
puts CONTACTED_AND_EXTRACTED
quit -noprompt
'''
    return script,evidence


def materialize(cell,bundle,chain,*,tech,output_dir,timeout_seconds=60,candidate=None):
    from clay.layout.native_diffusion_pcell import _magic
    if type(bundle) is not NativeGeometryBundle:raise TypeError('complete native bundle required')
    if getattr(tech,'pdk_id',None)!='sky130':raise ValueError('typed native producer requires the selected SKY130 tech')
    bundle.require_current(cell)
    source_sha,data,pins=_operands(cell,chain)
    implementation=None
    if candidate is not None:
        from clay.layout.native_row_candidate import require_candidate
        implementation=require_candidate(cell,chain,candidate)
    profile=chain.generation_profile if implementation is None else implementation
    command=json.loads((bundle.output_dir/'command.json').read_text(encoding='utf-8'))
    images=[v for v in command if isinstance(v,str) and re.fullmatch(r'sha256:[0-9a-f]{64}',v)]
    if len(images)!=1 or images[0]!=PINNED_IMAGE:raise ValueError('typed producer bundle image differs')
    runtime=json.loads((bundle.output_dir/'toolchain.json').read_text(encoding='utf-8'))
    if not runtime['pdk_root'].endswith('/sky130A'):raise ValueError('typed producer PDK root differs')
    pdk_root=runtime['pdk_root'][:-len('/sky130A')];folder=Path(output_dir).resolve()
    folder.mkdir(parents=True,exist_ok=False)
    for ports in (0,1):
        unit=folder/('unit-ports'+str(ports));unit.mkdir()
        (unit/'generate.tcl').write_text(unit_tcl(chain,ports,implementation=implementation),encoding='utf-8',newline='\n')
        _magic(unit,images[0],pdk_root,'generate.tcl','primitive',timeout_seconds)
    labeled=folder/'unit-ports1/primitive.gds';noports=folder/'unit-ports0/primitive.gds'
    verify_unit_reference(labeled,noports,images[0])
    raw=folder/'primitive.gds'
    fact=composer.compose(labeled,noports,raw,data,pins,implementation=implementation)
    composer.verify(labeled,noports,folder/'primitive-hierarchical.gds',raw,data,pins,implementation=implementation)
    (folder/'composition-proof.json').write_text(json.dumps(fact,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    shutil.copyfile(raw,folder/'raw.gds')
    text,evidence=contacts_tcl(cell,chain,raw,labeled,tech,candidate=candidate)
    (folder/'gate-access.json').write_text(json.dumps(evidence,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
    (folder/'contacts.tcl').write_text(text,encoding='utf-8',newline='\n')
    _magic(folder,images[0],pdk_root,'contacts.tcl','contacts',timeout_seconds)
    if profile in (OUTPUT_PROFILE,INPUT_OUTPUT_PROFILE):
        from clay.layout import native_gate_440_output as output_access
        base=folder/'base-contacted.gds'
        (folder/'contacted.gds').replace(base)
        proof=output_access.emit(cell,chain,base,folder/'contacted.gds',tech,candidate=candidate)
        (folder/'output-access.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        if profile == INPUT_OUTPUT_PROFILE:
            from clay.layout import native_gate_440_input as input_access
            intermediate=folder/'output-contacted.gds'
            (folder/'contacted.gds').replace(intermediate)
            input_proof=input_access.emit(cell,chain,intermediate,folder/'gate-access.json',folder/'contacted.gds',tech,candidate=candidate)
            (folder/'input-access.json').write_text(json.dumps(input_proof,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        drc_script="""crashbackups stop
drc off
gds read /work/contacted.gds
load primitive
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
quit -noprompt
"""
        ext_script="""crashbackups stop
drc off
gds read /work/contacted.gds
load primitive
flatten primitive_flat
load primitive_flat
select top cell
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
"""
        (folder/'output-drc.tcl').write_text(drc_script,encoding='utf-8',newline='\n')
        (folder/'output-extract.tcl').write_text(ext_script,encoding='utf-8',newline='\n')
        _magic(folder,images[0],pdk_root,'output-drc.tcl','output-drc',timeout_seconds)
        _magic(folder,images[0],pdk_root,'output-extract.tcl','output-extract',timeout_seconds)
        output_access.verify(cell,chain,base,folder/('output-contacted.gds' if profile==INPUT_OUTPUT_PROFILE else 'contacted.gds'),json.loads((folder/'output-access.json').read_text()),tech,candidate=candidate)
        if profile == INPUT_OUTPUT_PROFILE:
            input_access.verify(cell,chain,folder/'output-contacted.gds',folder/'gate-access.json',folder/'contacted.gds',json.loads((folder/'input-access.json').read_text()),tech,candidate=candidate)
    observer=folder/'observer';observer.mkdir();shutil.copyfile(folder/'contacted.gds',observer/'primitive.gds')
    for name in ('drc.txt','native.spice','primitive_flat.ext','scale_um.txt'):
        shutil.copyfile(folder/name,observer/name)
    device=next(d for d in cell.devices if d.instance_name==chain.fingers[0].owner)
    spec=dict(native_geometry_spec(device));count=len(chain.fingers)
    spec.update(w_nm=count*chain.width_nm,l_nm=chain.length_nm,nf=count,m=1,mult=1,replicas=1,occurrences=count)
    observed=observe_native_primitive(observer,spec)
    bundle.require_current(cell)
    if _operands(cell,chain)[0]!=source_sha:raise ValueError('source changed during typed generation')
    receipt={'source_identity':native_source_identity(cell),'source_sha256':source_sha,'model':chain.model,
        'unit_width_nm':chain.width_nm,'length_nm':chain.length_nm,'image':images[0],
        'native_bundle_manifest_sha256':_sha(bundle.output_dir/'geometry.json'),
        'producer_sha256':_sha(__file__),'inputs':{str(Path(cell.source_span['file']).resolve()):source_sha},
        'outputs':{p.relative_to(folder).as_posix():_sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}}
    if candidate is None: receipt['generation_profile']=chain.generation_profile
    else: receipt['candidate_implementation']=asdict(candidate)
    (folder/'template-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return observed,receipt


def admits_producer_identity(receipt):
    digest = receipt.get('producer_sha256') if isinstance(receipt, dict) else None
    return isinstance(digest, str) and bool(re.fullmatch(r'[0-9a-f]{64}', digest)) and digest != '0' * 64


def verify_receipt(cell,chain,folder,receipt,tech, *, candidate=None):
    """Replay the real unit union and source-derived contact recipe, not hashes alone."""
    if candidate is None:
        return _verify_receipt(cell,chain,folder,receipt,tech,candidate=None)
    from clay.layout.native_row_candidate import candidate_verification_scope
    with candidate_verification_scope(cell,chain,candidate):
        return _verify_receipt(cell,chain,folder,receipt,tech,candidate=candidate)


def _verify_receipt(cell,chain,folder,receipt,tech, *, candidate=None):
    folder=Path(folder);_source,data,pins=_operands(cell,chain)
    implementation=None
    if candidate is not None:
        from clay.layout.native_row_candidate import require_candidate
        implementation=require_candidate(cell,chain,candidate)
        if receipt.get('candidate_implementation') != asdict(candidate) or 'generation_profile' in receipt:
            raise ValueError('native row candidate receipt differs')
    elif receipt.get('generation_profile')!=chain.generation_profile or 'candidate_implementation' in receipt:
        raise ValueError('typed native source profile differs')
    profile=chain.generation_profile if implementation is None else implementation
    if (tech.pdk_id!='sky130' or
            not admits_producer_identity(receipt)):
        raise ValueError('typed native producer identity differs')
    required=('primitive.gds','raw.gds','primitive-hierarchical.gds','contacted.gds','contacts.tcl',
              'gate-access.json','drc.txt','native.spice','primitive_flat.ext','scale_um.txt',
              'unit-ports0/generate.tcl','unit-ports0/primitive.gds',
              'unit-ports1/generate.tcl','unit-ports1/primitive.gds')
    outputs=receipt.get('outputs')
    if type(outputs) is not dict or not set(required)<=set(outputs):
        raise ValueError('typed native receipt output inventory incomplete')
    if profile in (OUTPUT_PROFILE,INPUT_OUTPUT_PROFILE):
        required += ('base-contacted.gds','output-access.json','output-drc.tcl','output-extract.tcl',
                     'output-drc.json','output-extract.json')
        if not set(required)<=set(outputs): raise ValueError('output access receipt inventory incomplete')
        from clay.layout import native_gate_440_output as output_access
        output_access.verify(cell,chain,folder/'base-contacted.gds',folder/('output-contacted.gds' if profile==INPUT_OUTPUT_PROFILE else 'contacted.gds'),
                             json.loads((folder/'output-access.json').read_text()),tech,candidate=candidate)
        if profile==INPUT_OUTPUT_PROFILE:
            required += ('output-contacted.gds','input-access.json')
            if not set(required)<=set(outputs):raise ValueError('input access receipt inventory incomplete')
            from clay.layout import native_gate_440_input as input_access
            input_access.verify(cell,chain,folder/'output-contacted.gds',folder/'gate-access.json',folder/'contacted.gds',json.loads((folder/'input-access.json').read_text()),tech,candidate=candidate)
    for name in required:
        if _sha(folder/name)!=outputs[name]:raise ValueError('typed native output differs: '+name)
    for ports in (0,1):
        if (folder/('unit-ports'+str(ports))/'generate.tcl').read_text(encoding='utf-8')!=unit_tcl(chain,ports,implementation=implementation):
            raise ValueError('typed native unit invocation differs')
    labeled=folder/'unit-ports1/primitive.gds';noports=folder/'unit-ports0/primitive.gds'
    verify_unit_reference(labeled,noports,receipt.get('image'))
    raw=folder/'primitive.gds'
    if _sha(raw)!=_sha(folder/'raw.gds'):raise ValueError('typed native raw artifact differs')
    composer.verify(labeled,noports,folder/'primitive-hierarchical.gds',raw,data,pins,implementation=implementation)
    text,evidence=contacts_tcl(cell,chain,raw,labeled,tech,candidate=candidate)
    if ((folder/'contacts.tcl').read_text(encoding='utf-8')!=text or
            json.loads((folder/'gate-access.json').read_text(encoding='utf-8'))!=json.loads(json.dumps(evidence,default=str))):
        raise ValueError('typed native gate access recipe differs')
