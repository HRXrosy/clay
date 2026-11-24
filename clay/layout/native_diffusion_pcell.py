"""Bounded source-driven SKY130 native diffusion PCells.

This producer supports the observed 1+2, two-gate chain with source/body on
the repeated diffusion columns. It emits a raw PCell and separate contacts,
and delegates physical extraction and DRC to the existing native observer.
"""
from collections import Counter
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import uuid

import klayout.db as kdb

from clay.layout.native_sky130 import (
    NativeGeometryBundle, _sha, native_source_identity, observe_native_primitive,
)
from clay.layout.place.native_diffusion_chain import (
    NativeDiffusionChain, compile_native_diffusion_chains, order_native_diffusion_chain,
)
from clay.pdk.sky130_native_geometry import native_geometry_spec, native_diffusion_contact_coverage


def _source_declaration_for_chain(cell, chain, *, label, intent=None, groups=None):
    from clay.layout.intent import layout_intent_from_cell_def, replay_array_intent
    from clay.layout.place.row_program import _group_map
    names = {finger.owner for finger in chain.fingers}
    resolved_groups = _group_map(cell) if groups is None else groups
    resolved_intent = layout_intent_from_cell_def(cell) if intent is None else intent
    declarations = [
        replay_array_intent(row)
        for row in resolved_intent.arrays
        if row.style == 'native_diffusion' and set(resolved_groups.get(row.group, ())) == names
    ]
    if len(declarations) > 1:
        raise ValueError(f'native diffusion {label} source declaration is ambiguous')
    return None if not declarations else declarations[0]


def _source_coverage_for_chain(cell, chain, *, intent=None, groups=None):
    """Find the declaration that owns this complete source occurrence set."""
    decl = _source_declaration_for_chain(cell, chain, label='coverage', intent=intent, groups=groups)
    return None if decl is None else decl.coverage_profile


def _source_access_for_chain(cell, chain, *, intent=None, groups=None):
    decl = _source_declaration_for_chain(cell, chain, label='access', intent=intent, groups=groups)
    return '' if decl is None else decl.access_profile


def _source_generation_for_chain(cell, chain, *, intent=None, groups=None):
    decl = _source_declaration_for_chain(cell, chain, label='generation', intent=intent, groups=groups)
    return '' if decl is None else decl.generation_profile


def _source_synthesis_for_chain(cell, chain, *, intent=None, groups=None):
    decl = _source_declaration_for_chain(cell, chain, label='synthesis', intent=intent, groups=groups)
    return '' if decl is None else decl.synthesis


def _source_tap_rail_for_chain(cell, chain, *, intent=None, groups=None):
    decl = _source_declaration_for_chain(cell, chain, label='tap_rail', intent=intent, groups=groups)
    return '' if decl is None else decl.tap_rail


def _check_chain(cell, chain, *, intent=None, groups=None, source_identity=None, specs=None):
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place.row_program import _group_map

    resolved_identity = native_source_identity(cell) if source_identity is None else source_identity
    if type(chain) is not NativeDiffusionChain or chain.source_identity != resolved_identity:
        raise ValueError('native diffusion chain source differs')
    resolved_intent = layout_intent_from_cell_def(cell) if intent is None else intent
    resolved_groups = _group_map(cell) if groups is None else groups
    if chain.coverage_profile != _source_coverage_for_chain(cell, chain, intent=resolved_intent, groups=resolved_groups):
        raise ValueError('native diffusion contact coverage differs from source declaration')
    if chain.access_profile != _source_access_for_chain(cell, chain, intent=resolved_intent, groups=resolved_groups):
        raise ValueError('native diffusion access profile differs from source declaration')
    if chain.generation_profile != _source_generation_for_chain(cell, chain, intent=resolved_intent, groups=resolved_groups):
        raise ValueError('native diffusion generation profile differs from source declaration')
    if chain.synthesis != _source_synthesis_for_chain(cell, chain, intent=resolved_intent, groups=resolved_groups):
        raise ValueError('native diffusion synthesis differs from source declaration')
    if chain.tap_rail != _source_tap_rail_for_chain(cell, chain, intent=resolved_intent, groups=resolved_groups):
        raise ValueError('native diffusion tap_rail differs from source declaration')
    if chain.synthesis in ('parameterized', 'cmos_gate'):
        if chain.coverage_profile is not None or chain.access_profile or chain.generation_profile:
            raise ValueError('parameterized native diffusion chain cannot combine legacy profiles')
        if (chain.model not in ('sky130_fd_pr__nfet_01v8', 'sky130_fd_pr__nfet_01v8_lvt',
                                'sky130_fd_pr__pfet_01v8', 'sky130_fd_pr__pfet_01v8_lvt')
                or type(chain.width_nm) is not int or chain.width_nm < 420 or chain.width_nm % 5 != 0
                or type(chain.length_nm) is not int or chain.length_nm < 150 or chain.length_nm % 5 != 0
                or not chain.fingers):
            raise ValueError('parameterized native diffusion chain has unsupported model or off-grid dimensions')
        owners = tuple(row.owner for row in chain.fingers)
        candidates = compile_native_diffusion_chains(
            cell,
            tuple(sorted(set(owners))),
            synthesis=chain.synthesis,
            tap_rail=chain.tap_rail,
            _specs=specs,
            _source_identity=resolved_identity,
        )
        if len(candidates) != 1 or order_native_diffusion_chain(
                candidates[0], owners, left_net=chain.columns[0]) != chain:
            raise ValueError('parameterized native diffusion chain does not replay from source')
        return chain
    if chain.generation_profile and (chain.generation_profile not in ('sky130_lvt_single_440_v1','sky130_lvt_single_440_output_m12_v1','sky130_lvt_single_440_output_input_m12_v1')
            or chain.coverage_profile is not None or chain.access_profile
            or (chain.model,chain.width_nm,chain.length_nm,len(chain.fingers)) !=
                ('sky130_fd_pr__nfet_01v8_lvt',2000,150,10)):
        raise ValueError('native diffusion generation profile is unsupported for source geometry')
    native_diffusion_contact_coverage(chain.model,chain.width_nm,chain.length_nm,
                                      len(chain.fingers),chain.coverage_profile)
    owners=tuple(row.owner for row in chain.fingers)
    counts=Counter(owners)
    is_buffer = len(owners)==3 and sorted(counts.values()) == [1,2]
    if chain.access_profile and chain.access_profile != 'shared_cross_coupled_m1_v1':
        gates=tuple(row.gate for row in chain.fingers);columns=chain.columns
        if not (chain.access_profile in ('shared_source_li_m1','shared_source_li_m1_far') and chain.coverage_profile is None
                and (chain.model,chain.width_nm,chain.length_nm,len(owners))==
                    ('sky130_fd_pr__nfet_01v8',500,150,3)
                and is_buffer and owners[1]==owners[2]!=owners[0]
                and gates[1]==gates[2]==columns[0] and gates[0]!=gates[1]
                and columns[1]==columns[3]==chain.body
                and len({gates[0],gates[1],columns[2],chain.body})==4):
            raise ValueError('native shared source access profile is unsupported for this source topology')
    is_logic = (len(owners), chain.model, chain.width_nm, chain.length_nm) in {
        (4,'sky130_fd_pr__pfet_01v8',800,150),
        (2,'sky130_fd_pr__nfet_01v8',420,150),
    } and len(counts)==len(owners)
    # This LVT profile was independently generated, contacted and observed
    # with the same native PDK: two four-finger input owners and one source-
    # tied dummy at each end. Exact owner/net connectivity is replayed below.
    is_lvt_input = ((len(owners),chain.model,chain.width_nm,chain.length_nm)
                    == (10,'sky130_fd_pr__nfet_01v8_lvt',2000,150)
                    and sorted(counts.values()) == [1,1,4,4])
    # Two source-audited PMOS trails use the same six-finger PDK master:
    # distinct two-finger owners between explicit source-tied end dummies.
    is_pmos_six = ((len(owners),chain.model,chain.width_nm,chain.length_nm)
                   == (6,'sky130_fd_pr__pfet_01v8',2000,150)
                   and sorted(counts.values()) == [1,1,2,2])
    # Source-bound tail profile: one eight-finger source and two equivalent
    # source-tied dummies, independently checked against the real PDK masks.
    is_tail = ((len(owners),chain.model,chain.width_nm,chain.length_nm)
               == (10,'sky130_fd_pr__nfet_01v8',500,150)
               and sorted(counts.values()) == [1,1,8])
    # The contacted SKY130 PMOS row is witnessed for two four-occurrence
    # source devices and two electrically equivalent end dummies.  Admission
    # is based on source topology and PDK geometry, never instance names.
    is_pmos_precharge = ((len(owners),chain.model,chain.width_nm,chain.length_nm)
                         == (10,'sky130_fd_pr__pfet_01v8',500,150)
                         and sorted(counts.values()) == [1,1,4,4])
    # Six distinct source owners in the reset/cross-coupled NMOS chain.
    # Both end dummies and every diffusion column remain source-bound.
    is_nmos_reset = ((len(owners),chain.model,chain.width_nm,chain.length_nm)
                     == (6,'sky130_fd_pr__nfet_01v8',2000,150)
                     and len(counts)==6 and all(count==1 for count in counts.values()))
    if chain.access_profile == 'shared_cross_coupled_m1_v1' and not is_nmos_reset:
        raise ValueError('cross-coupled access requires the six-device native reset row')
    if not (is_buffer or is_logic or is_lvt_input or is_pmos_six or is_tail or is_pmos_precharge or is_nmos_reset):
        raise ValueError('native diffusion chain has unsupported source inventory or dimensions')
    if is_buffer:
        first=next(name for name,count in counts.items() if count==1)
        second=next(name for name,count in counts.items() if count==2)
        if owners != (first,second,second):
            raise ValueError('native diffusion chain source order differs')
    candidates=compile_native_diffusion_chains(cell,tuple(sorted(set(owners))))
    if len(candidates)!=1 or order_native_diffusion_chain(candidates[0],owners,left_net=chain.columns[0]) != replace(chain,coverage_profile=None,access_profile='',generation_profile=''):
        raise ValueError('native diffusion chain does not replay from source')
    if is_buffer and not (chain.fingers[0].gate != chain.fingers[1].gate == chain.fingers[2].gate
            and chain.columns[1] == chain.columns[3] == chain.body):
        raise ValueError('native diffusion chain topology is unsupported')
    if is_logic and not (chain.columns[0] == chain.columns[-1] == chain.body
                         and len({finger.gate for finger in chain.fingers}) == 2
                         and all(chain.fingers[i].gate != chain.fingers[i+1].gate for i in range(len(owners)-1))):
        raise ValueError('native logic diffusion chain topology is unsupported')
    if is_lvt_input:
        devices={device.instance_name:device for device in cell.devices}
        first, last = devices[owners[0]], devices[owners[-1]]
        gates=tuple(finger.gate for finger in chain.fingers)
        if not (counts[owners[0]] == counts[owners[-1]] == 1
                and owners[1:5] == (owners[1],)*4
                and owners[5:9] == (owners[5],)*4
                and gates[:5] == (gates[0],)*5 and gates[5:] == (gates[5],)*5
                and gates[0] != gates[5]
                and first.pins['d'] == first.pins['s'] == chain.columns[0] == chain.columns[1]
                and last.pins['d'] == last.pins['s'] == chain.columns[-2] == chain.columns[-1]
                and chain.columns[0] == chain.columns[-1]):
            raise ValueError('native LVT input row topology is unsupported')
    if is_pmos_six:
        devices={device.instance_name:device for device in cell.devices}
        first,last=devices[owners[0]],devices[owners[-1]]
        gates=tuple(finger.gate for finger in chain.fingers)
        shared=chain.columns[0]
        if not (counts[owners[0]] == counts[owners[-1]] == 1
                and owners[1] == owners[2] and owners[3] == owners[4]
                and gates[:3] == (gates[0],)*3 and gates[3:] == (gates[3],)*3
                and gates[0] != gates[3]
                and first.pins['d'] == first.pins['s'] == shared == chain.columns[1]
                and last.pins['d'] == last.pins['s'] == shared == chain.columns[-2] == chain.columns[-1]
                and chain.columns[3] == shared and shared != chain.body
                and (chain.columns[2] == chain.body) != (chain.columns[4] == chain.body)):
            raise ValueError('native six-finger PMOS row topology is unsupported')
    if is_tail:
        devices={device.instance_name:device for device in cell.devices}
        first,last=devices[owners[0]],devices[owners[-1]]
        shared=chain.columns[0]
        if not (counts[owners[0]] == counts[owners[-1]] == 1
                and owners[1:9] == (owners[1],)*8
                and len({finger.gate for finger in chain.fingers}) == 1
                and first.pins['d'] == first.pins['s'] == shared
                and last.pins['d'] == last.pins['s'] == shared
                and shared != chain.body
                and all(chain.columns[i] == shared for i in (0,1,3,5,7,9,10))
                and all(chain.columns[i] == chain.body for i in (2,4,6,8))):
            raise ValueError('native ten-finger tail row topology is unsupported')
    if is_pmos_precharge:
        devices={device.instance_name:device for device in cell.devices}
        first,last=devices[owners[0]],devices[owners[-1]]
        body=chain.body
        if not (counts[owners[0]] == counts[owners[-1]] == 1
                and owners[1:5] == (owners[1],)*4
                and owners[5:9] == (owners[5],)*4
                and len({finger.gate for finger in chain.fingers}) == 1
                and first.pins['d'] == first.pins['s'] == body
                and last.pins['d'] == last.pins['s'] == body
                and all(chain.columns[i] == body for i in (0,1,3,5,7,9,10))
                and chain.columns[2] == chain.columns[4]
                and chain.columns[6] == chain.columns[8]
                and len({body,chain.columns[2],chain.columns[6]}) == 3):
            raise ValueError('native ten-finger PMOS row topology is unsupported')
    if is_nmos_reset:
        body=chain.body; columns=chain.columns; gates=tuple(f.gate for f in chain.fingers)
        if not (all(columns[i]==body for i in (0,1,3,5,6))
                and len({body,columns[2],columns[4],gates[1]})==4
                and gates[0]==gates[5]==body and gates[1]==gates[4]
                and gates[2]==columns[4] and gates[3]==columns[2]):
            raise ValueError('native six-finger NMOS reset row topology is unsupported')
    supported={('sky130_fd_pr__pfet_01v8',1000,150),('sky130_fd_pr__nfet_01v8',500,150)}
    if not (is_logic or is_lvt_input or is_pmos_six or is_tail or is_pmos_precharge or is_nmos_reset) and (chain.model,chain.width_nm,chain.length_nm) not in supported:
        raise ValueError('native diffusion chain model or dimensions are unsupported')
    return chain


def _connected_precharge_profile(chain):
    """Select the connected-gate template only after source chain replay."""
    owners = tuple(f.owner for f in chain.fingers)
    return ((len(owners), chain.model, chain.width_nm, chain.length_nm)
            == (10, 'sky130_fd_pr__pfet_01v8', 500, 150)
            and sorted(Counter(owners).values()) == [1, 1, 4, 4]
            and len({f.gate for f in chain.fingers}) == 1)


def _magic(folder, image, pdk_root, script, tag, timeout):
    name='clay-native-chain-'+uuid.uuid4().hex[:12]
    command=['docker','run','--rm','--network','none','--name',name,
        '-v',f'{folder.as_posix()}:/work','-w','/work','--entrypoint','bash',image,
        '-lc',f'exec magic -dnull -noconsole -rcfile {shlex.quote(pdk_root + "/sky130A/libs.tech/magic/sky130A.magicrc")} /work/{script}']
    try:
        process=subprocess.run(command,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=30)
        raise
    (folder/(tag+'.stdout')).write_bytes(process.stdout)
    (folder/(tag+'.stderr')).write_bytes(process.stderr)
    (folder/(tag+'.json')).write_text(json.dumps({'command':command,'returncode':process.returncode},indent=2)+'\n')
    if process.returncode:
        raise ValueError('native diffusion PCell generator failed: '+tag)


def _native_diffusion_generation_tcl(chain):
    """Canonical PDK invocation; no coverage arguments for the legacy default."""
    if chain.generation_profile:
        raise ValueError('native generation profile requires its distinct typed producer')
    coverage=native_diffusion_contact_coverage(chain.model,chain.width_nm,chain.length_nm,
                                               len(chain.fingers),chain.coverage_profile)
    coverage_args=''.join(f' {name} {value}' for name,value in coverage)
    count=len(chain.fingers)
    gate_top=0 if chain.access_profile and chain.access_profile != 'shared_cross_coupled_m1_v1' else 1
    return f'''crashbackups stop
drc off
load primitive
box 0 0 0 0
set params {{w {count*chain.width_nm/1000:.3f} l {chain.length_nm/1000:.3f} m 1 nf {count} guard 0 conn_gates {int(_connected_precharge_profile(chain))} topc {gate_top} botc 0 doports 1 viagb 100{coverage_args}}}
magic::gencell sky130::{chain.model} U0 -spice {{*}}$params
select top cell
gds write primitive.gds
puts GENERATED
quit -noprompt
'''


def _native_body_position(active_box, family):
    """Preserve the existing native tap placement authority for both paths."""
    return active_box.right+(540 if family=='p' else 520),(active_box.bottom+active_box.top)//2


def materialize_native_diffusion_template(cell, bundle, chain, *, output_dir, timeout_seconds=60):
    """Generate and observe one supported 3-finger native geometry template."""
    if type(bundle) is not NativeGeometryBundle:
        raise TypeError('native diffusion template requires the complete native bundle')
    bundle.require_current(cell)
    _check_chain(cell,chain)
    from clay.layout.native_diffusion_source import _require_canonical_native_arrays
    source_before,canonical=_require_canonical_native_arrays(cell)
    _check_chain(canonical,chain)
    command=json.loads((bundle.output_dir/'command.json').read_text(encoding='utf-8'))
    images=[arg for arg in command if isinstance(arg,str) and re.fullmatch(r'sha256:[0-9a-f]{64}',arg)]
    if len(images)!=1:
        raise ValueError('native bundle image authority is missing')
    image=images[0]
    runtime=json.loads((bundle.output_dir/'toolchain.json').read_text(encoding='utf-8'))
    pdk_path=runtime['pdk_root']
    if not pdk_path.endswith('/sky130A'):
        raise ValueError('native bundle PDK root differs')
    pdk_root=pdk_path[:-len('/sky130A')]
    folder=Path(output_dir).resolve(); folder.mkdir(parents=True,exist_ok=False)
    source_device=next(device for device in cell.devices if device.instance_name==chain.fingers[0].owner)
    spec=dict(native_geometry_spec(source_device))
    count=len(chain.fingers)
    spec.update(w_nm=count*chain.width_nm,l_nm=chain.length_nm,nf=count,m=1,mult=1,replicas=1,occurrences=count)
    family='p' if chain.model=='sky130_fd_pr__pfet_01v8' else 'n'
    connected_gates = _connected_precharge_profile(chain)
    (folder/'generate.tcl').write_text(_native_diffusion_generation_tcl(chain))
    _magic(folder,image,pdk_root,'generate.tcl','primitive',timeout_seconds)
    shutil.copyfile(folder/'primitive.gds',folder/'raw.gds')
    layout=kdb.Layout(); layout.read(str(folder/'raw.gds')); top=layout.top_cell()
    if layout.dbu!=.001 or top is None: raise ValueError('native PCell DBU/top differs')
    active=kdb.Region(top.begin_shapes_rec(layout.layer(65,20))).merged()
    poly=kdb.Region(top.begin_shapes_rec(layout.layer(66,20))).merged()
    metal=kdb.Region(top.begin_shapes_rec(layout.layer(68,20))).merged()
    gates=sorted((active&poly).merged().each(),key=lambda p:p.bbox().left)
    if len(gates)!=count or any((p.bbox().width(),p.bbox().height()) != (chain.length_nm,chain.width_nm) for p in gates):
        raise ValueError('native PCell physical channels differ')
    box=active.bbox(); xs=[(gates[i].bbox().left+gates[i].bbox().right)//2 for i in range(0,count,2)]
    top_fills=[]
    for i in (() if connected_gates or (chain.access_profile and chain.access_profile != 'shared_cross_coupled_m1_v1') else range(1,count,2)):
        mid=(gates[i].bbox().left+gates[i].bbox().right)//2
        tops=[p.bbox() for p in metal.each() if p.bbox().left<=mid<=p.bbox().right and p.bbox().bottom>=box.top]
        if len(tops)!=1: raise ValueError('native upper gate contact is ambiguous')
        top_box=tops[0]
        top_fills.append(f'box {(mid-185)/1000:.3f}um {top_box.bottom/1000:.3f}um {(mid+185)/1000:.3f}um {top_box.top/1000:.3f}um\npaint metal1')
    gate_y=box.bottom-360; body_x,body_y=_native_body_position(box,family)
    gate_fills='\n'.join(f'box {(x-185)/1000:.3f}um {(gate_y-115)/1000:.3f}um '
        f'{(x+185)/1000:.3f}um {(gate_y+115)/1000:.3f}um\npaint metal1' for x in xs)
    if connected_gates:
        xs=[]
        gate_fills=''
    if family=='p':
        well=kdb.Region(top.begin_shapes_rec(layout.layer(64,20))).merged()
        well_box=well.bbox()
        if well.is_empty(): raise ValueError('native PMOS PCell has no nwell')
        well_top=1810 if count == 3 else well_box.top
        well_left=box.right+90 if count == 3 else well_box.left
        body=('''box %0.3fum %0.3fum %0.3fum %0.3fum\npaint nwell\n''' %
              (well_left/1000,well_box.bottom/1000,(box.right+930)/1000,well_top/1000))
    else:
        body=''
    diffusion,contact=('nsd','nsc') if family=='p' else ('psd','psc')
    # Select the flattened cell explicitly: Magic's DRC list is selection
    # scoped and otherwise can report an empty list for the wrong cell.
    flatten_before_drc = len(chain.fingers) != 3
    before_drc = 'flatten primitive_flat\nload primitive_flat\nselect top cell' if flatten_before_drc else ''
    before_extract = '' if flatten_before_drc else 'flatten primitive_flat\nload primitive_flat'
    (folder/'contacts.tcl').write_text(f'''crashbackups stop
drc off
gds read /work/raw.gds
load primitive
{body}box {body_x/1000:.3f}um {body_y/1000:.3f}um {body_x/1000:.3f}um {body_y/1000:.3f}um
sky130::draw_contact 0.18 0.17 0.12 0.08 0.17 {diffusion} {contact} li horz
sky130::draw_contact 0.17 0.17 0.08 0.06 0.17 li mcon metal1 full
label B c {contact}
foreach x {{{' '.join(f'{x/1000:.3f}' for x in xs)}}} {{
  box ${{x}}um {gate_y/1000:.3f}um ${{x}}um {gate_y/1000:.3f}um
  sky130::draw_contact 0.17 0.17 0.08 0.08 0.17 poly pc li horz
  sky130::draw_contact 0.17 0.17 0.08 0.06 0.17 li mcon metal1 horz
}}
{gate_fills}
{chr(10).join(top_fills)}
select top cell
gds write contacted.gds
set h [open scale_um.txt w]
puts $h [cif scale out]
close $h
{before_drc}
drc euclidean on
drc style drc(full)
drc check
drc catchup
set h [open drc.txt w]
puts $h [drc listall why]
close $h
{before_extract}
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
''')
    access_evidence=None
    if chain.access_profile and chain.access_profile != 'shared_cross_coupled_m1_v1':
        from clay.layout.native_buffer_access import shared_source_contacts_tcl
        from clay.layout.tech import LayoutTech
        access_text,access_evidence=shared_source_contacts_tcl(layout=layout,top=top,chain=chain,
            tech=LayoutTech.resolve('sky130'),body_x=body_x,body_y=body_y)
        (folder/'contacts.tcl').write_text(access_text,encoding='utf-8')
    _magic(folder,image,pdk_root,'contacts.tcl','contacts',timeout_seconds)
    # Keep the raw PCell and added contacts independently available. The
    # existing observer expects the complete mask under its conventional name.
    observer=folder/'observer'; observer.mkdir()
    shutil.copyfile(folder/'contacted.gds',observer/'primitive.gds')
    for name in ('drc.txt','native.spice','primitive_flat.ext','scale_um.txt'):
        shutil.copyfile(folder/name,observer/name)
    observed=observe_native_primitive(observer,spec)
    bundle.require_current(cell)
    source_sha,_canonical= _require_canonical_native_arrays(cell)
    if source_sha != source_before:
        raise ValueError('native diffusion source changed during template generation')
    evidence={'source_identity':native_source_identity(cell),'source_sha256':source_sha,'model':chain.model,
        'unit_width_nm':chain.width_nm,'length_nm':chain.length_nm,
        'image':image,'native_bundle_manifest_sha256':_sha(bundle.output_dir/'geometry.json'),
        'producer_sha256':_sha(__file__),
        'inputs':{str(Path(cell.source_span['file']).resolve()):source_sha},
        'outputs':{path.name:_sha(path) for path in sorted(folder.iterdir()) if path.is_file()}}
    if chain.coverage_profile is not None:
        evidence['coverage_profile']=list(chain.coverage_profile)
    if chain.access_profile:
        evidence['access_profile']=chain.access_profile
        evidence['access_rule_evidence']=access_evidence
    (folder/'template-receipt.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return observed, evidence
