"""Source-bound compact M1 access for a declared SKY130 buffer row.

The pinned native PCell emits source LI without MCON using its ``viasrc``
parameter. Existing contact construction then adds grid/PDK-derived pads;
no source, device dimensions, net roles, or raw PCell masks are rewritten.
"""
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
    old = 'doports 1 viagb 100}'
    if text.count(old) != 1:
        raise ValueError('native buffer PCell invocation differs')
    return text.replace(old, 'doports 1 viagb 100 viasrc 0}')


def _contacts(chain, raw, tech):
    from clay.layout.native_buffer_access import shared_source_contacts_tcl
    from clay.layout.native_diffusion_pcell import _native_body_position
    layout = kdb.Layout(); layout.read(str(raw)); top = layout.top_cell()
    if layout.dbu != .001 or top is None:
        raise ValueError('compact native source raw PCell differs')
    active = kdb.Region(top.begin_shapes_rec(layout.layer(65,20))).merged()
    body_x, body_y = _native_body_position(active.bbox(), 'n')
    return shared_source_contacts_tcl(layout=layout, top=top, chain=chain, tech=tech,
        body_x=body_x, body_y=body_y, compact_source=True)


def materialize(cell, bundle, chain, *, tech, output_dir, timeout_seconds=60, candidate):
    from clay.layout.native_diffusion_pcell import _check_chain, _magic
    from clay.layout.native_diffusion_source import _require_canonical_native_arrays
    from clay.layout.native_row_candidate import BUFFER_COMPACT_SOURCE, require_candidate
    if type(bundle) is not NativeGeometryBundle:
        raise TypeError('compact native source requires a complete bundle')
    if require_candidate(cell,chain,candidate) != BUFFER_COMPACT_SOURCE:
        raise ValueError('compact native source choice differs')
    bundle.require_current(cell); _check_chain(cell,chain)
    source_sha,canonical = _require_canonical_native_arrays(cell); _check_chain(canonical,chain)
    command=json.loads((bundle.output_dir/'command.json').read_text(encoding='utf8'))
    images=[x for x in command if isinstance(x,str) and re.fullmatch(r'sha256:[0-9a-f]{64}',x)]
    runtime=json.loads((bundle.output_dir/'toolchain.json').read_text(encoding='utf8'))
    if len(images)!=1 or not runtime['pdk_root'].endswith('/sky130A') or tech.pdk_id!='sky130':
        raise ValueError('compact native source PDK authority differs')
    pdk_root=runtime['pdk_root'][:-len('/sky130A')]
    folder=Path(output_dir).resolve();folder.mkdir(parents=True,exist_ok=False)
    (folder/'generate.tcl').write_text(generation_tcl(chain),encoding='utf8',newline='\n')
    _magic(folder,images[0],pdk_root,'generate.tcl','primitive',timeout_seconds)
    shutil.copyfile(folder/'primitive.gds',folder/'raw.gds')
    text,facts=_contacts(chain,folder/'raw.gds',tech)
    (folder/'contacts.tcl').write_text(text,encoding='utf8',newline='\n')
    _magic(folder,images[0],pdk_root,'contacts.tcl','contacts',timeout_seconds)
    observer=folder/'observer';observer.mkdir()
    shutil.copyfile(folder/'contacted.gds',observer/'primitive.gds')
    for name in ('drc.txt','native.spice','primitive_flat.ext','scale_um.txt'):
        shutil.copyfile(folder/name,observer/name)
    device=next(d for d in cell.devices if d.instance_name==chain.fingers[0].owner)
    spec=dict(native_geometry_spec(device));count=len(chain.fingers)
    spec.update(w_nm=count*chain.width_nm,l_nm=chain.length_nm,nf=count,m=1,mult=1,replicas=1,occurrences=count)
    observed=observe_native_primitive(observer,spec)
    bundle.require_current(cell)
    if _require_canonical_native_arrays(cell)[0]!=source_sha:
        raise ValueError('source changed during compact native generation')
    receipt={'source_identity':native_source_identity(cell),'source_sha256':source_sha,'model':chain.model,
        'unit_width_nm':chain.width_nm,'length_nm':chain.length_nm,'image':images[0],
        'native_bundle_manifest_sha256':_sha(bundle.output_dir/'geometry.json'),
        'producer_sha256':_sha(__file__),'access_profile':chain.access_profile,
        'access_rule_evidence':json.loads(json.dumps(facts)),'candidate_implementation':asdict(candidate),
        'inputs':{str(Path(cell.source_span['file']).resolve()):source_sha},
        'outputs':{p.name:_sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    (folder/'template-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    return observed,receipt


def admits_producer_identity(receipt):
    digest = receipt.get('producer_sha256') if isinstance(receipt, dict) else None
    return isinstance(digest, str) and bool(re.fullmatch(r'[0-9a-f]{64}', digest)) and digest != '0' * 64


def verify_receipt(cell,chain,folder,receipt,tech,*,candidate):
    from clay.layout.native_diffusion_source import _require_canonical_native_arrays
    from clay.layout.native_row_candidate import BUFFER_COMPACT_SOURCE, require_candidate
    if require_candidate(cell,chain,candidate)!=BUFFER_COMPACT_SOURCE:
        raise ValueError('compact native source choice differs')
    folder=Path(folder);source_sha,_=_require_canonical_native_arrays(cell)
    if (receipt.get('candidate_implementation')!=asdict(candidate)
            or not admits_producer_identity(receipt)
            or receipt.get('source_sha256')!=source_sha
            or receipt.get('source_identity')!=native_source_identity(cell)
            or receipt.get('access_profile')!=chain.access_profile):
        raise ValueError('compact native source receipt differs')
    required=('generate.tcl','contacts.tcl','primitive.gds','raw.gds','contacted.gds',
              'drc.txt','native.spice','primitive_flat.ext','scale_um.txt')
    outputs=receipt.get('outputs')
    if type(outputs) is not dict or not set(required)<=set(outputs):
        raise ValueError('compact native source receipt inventory incomplete')
    if any(_sha(folder/name)!=outputs[name] for name in required):
        raise ValueError('compact native source output differs')
    if (folder/'generate.tcl').read_text(encoding='utf8')!=generation_tcl(chain):
        raise ValueError('compact native source PCell invocation differs')
    text,facts=_contacts(chain,folder/'raw.gds',tech)
    if ((folder/'contacts.tcl').read_text(encoding='utf8')!=text
            or receipt.get('access_rule_evidence')!=json.loads(json.dumps(facts))):
        raise ValueError('compact native source contact replay differs')
    if _sha(folder/'primitive.gds')!=_sha(folder/'raw.gds'):
        raise ValueError('compact native source raw artwork differs')
