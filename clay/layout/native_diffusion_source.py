"""Fresh native PCell authority for source-declared diffusion mosaics."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from weakref import WeakKeyDictionary

from clay.layout.native_sky130 import NativeGeometryBundle, native_source_identity
from clay.layout.place.native_diffusion_chain import (compile_native_diffusion_chains,
    order_native_diffusion_chain, order_native_diffusion_occurrences)
from clay.layout.route.exact_artifact_evidence import _replayed_source_cell_authority
from clay.pdk.sky130_native_geometry import native_geometry_spec
from clay.layout.place.row_detail_timing import measured as _row_measure


def _sha_bytes(path): return sha256(Path(path).read_bytes()).hexdigest()
def _sha(path): return _row_measure('source.file_hash', _sha_bytes, path)


def _require_canonical_native_arrays(cell, intent=None):
    """Reparse source bytes before allowing a mutable declaration to select artwork."""
    from clay.layout.intent import layout_intent_from_cell_def, replay_array_intent
    _source_sha,_inventory_sha,_guards,canonical=_replayed_source_cell_authority(cell,expected_pdk_id='sky130')
    canonical_arrays=tuple(replay_array_intent(row) for row in layout_intent_from_cell_def(canonical).arrays)
    current_arrays=tuple(replay_array_intent(row) for row in layout_intent_from_cell_def(cell).arrays)
    requested_arrays=current_arrays if intent is None else tuple(replay_array_intent(row) for row in intent.arrays)
    if current_arrays!=canonical_arrays or requested_arrays!=canonical_arrays:
        raise ValueError('native diffusion array declarations differ from canonical source bytes')
    return _source_sha, canonical


def declared_native_chains(cell, intent):
    """Resolve explicit native_diffusion groups to supported source chains."""
    from collections import Counter
    from clay.layout.place.row_program import _group_map
    from clay.pdk.sky130_native_geometry import native_cell_geometry_specs

    declarations=tuple(row for row in intent.arrays if row.style=='native_diffusion')
    if not declarations:
        return ()
    groups=_group_map(cell); used=set(); result=[]
    cell_specs=native_cell_geometry_specs(cell)
    cell_identity=native_source_identity(cell)
    native_names = {name for declaration in declarations
                    for name in groups.get(declaration.group, ())}
    competitors = tuple(
        ('array', row.group) for row in intent.arrays
        if row.style != 'native_diffusion'
    ) + tuple(('common_centroid', row.group) for row in intent.common_centroids)
    for kind, group in competitors:
        overlap = native_names.intersection(groups.get(group, ()))
        if overlap:
            raise ValueError(
                f'native diffusion overlaps {kind} group {group!r}: '
                + ', '.join(sorted(overlap))
            )
    # The native projection presently builds its own source witness and cannot
    # combine an independently authored mosaic witness. Reject it rather than
    # replacing the existing inventory while projecting native rows.
    if getattr(intent, 'discrete_mosaics', ()):
        raise ValueError('native diffusion cannot combine an existing discrete mosaic authority')
    for declaration in sorted(declarations,key=lambda row:row.group):
        if declaration.rows!=1 or declaration.order!='ratio':
            raise ValueError('native diffusion declaration requires one ratio row')
        names=tuple(groups.get(declaration.group,()))
        if declaration.synthesis in ('parameterized', 'cmos_gate'):
            if not names or len(set(names))!=len(names) or used.intersection(names):
                raise ValueError('parameterized native diffusion groups must contain nonempty disjoint source owners')
            candidates=compile_native_diffusion_chains(
                cell,
                tuple(sorted(names)),
                synthesis=declaration.synthesis,
                tap_rail=declaration.tap_rail,
                _specs=cell_specs,
                _source_identity=cell_identity,
            )
            if len(candidates)!=1:
                raise ValueError('parameterized native diffusion source group does not form one chain')
            if declaration.occurrences:
                chain=order_native_diffusion_occurrences(candidates[0],declaration.occurrences)
            else:
                chain=candidates[0]
            from clay.layout.native_diffusion_pcell import _check_chain
            _check_chain(
                cell,
                chain,
                intent=intent,
                groups=groups,
                source_identity=cell_identity,
                specs=cell_specs,
            )
            used.update(names); result.append((declaration.group,chain))
            continue
        if len(names) not in (2, 3, 4, 6) or len(set(names))!=len(names) or used.intersection(names):
            raise ValueError('native diffusion groups must contain two, three, four or six disjoint source owners')
        if len(names)==6 and not declaration.occurrences:
            raise ValueError('six-owner native diffusion requires an explicit occurrence ledger')
        candidates=compile_native_diffusion_chains(
            cell,
            tuple(sorted(names)),
            _specs=cell_specs,
            _source_identity=cell_identity,
        )
        if len(candidates)!=1:
            raise ValueError('native diffusion source group does not form one chain')
        counts=Counter(f.owner for f in candidates[0].fingers)
        if declaration.occurrences:
            chain=order_native_diffusion_occurrences(candidates[0],declaration.occurrences)
        elif sorted(counts.values()) == [1,2]:
            first=next(name for name,count in counts.items() if count==1)
            second=next(name for name,count in counts.items() if count==2)
            chain=order_native_diffusion_chain(candidates[0],(first,second,second))
        elif all(count == 1 for count in counts.values()) and len(counts) in (2, 4):
            chain=candidates[0]
        else:
            raise ValueError('native diffusion group has unsupported source finger inventory')
        chain=replace(chain,coverage_profile=declaration.coverage_profile,access_profile=declaration.access_profile,
                      generation_profile=declaration.generation_profile)
        from clay.layout.native_diffusion_pcell import _check_chain
        _check_chain(
            cell,
            chain,
            intent=intent,
            groups=groups,
            source_identity=cell_identity,
            specs=cell_specs,
        )
        used.update(names); result.append((declaration.group,chain))
    return tuple(result)


@dataclass(frozen=True,eq=False,init=False)
class NativePCellWitness:
    cell: object
    bundle: object
    tech: object
    rows: tuple
    source_cir: str | None
    overlay: object
    identity: str
    dependencies: tuple
    row_candidates: tuple

    def __init__(self,*args,**kwargs):
        raise TypeError('native PCell witness requires fresh producer capture')

    @property
    def sha256(self): return self.identity


_WITNESSES=WeakKeyDictionary()
_FIELDS=('cell','bundle','tech','rows','source_cir','overlay','identity','dependencies','row_candidates')


def _validate_coverage_receipt(chain, receipt, folder, tech, cell=None, candidate=None):
    """Bind an opted-in PDK invocation to the exact source declaration."""
    if ((not chain.synthesis and 'synthesis' in receipt) or
            (chain.synthesis and receipt.get('synthesis')!=chain.synthesis)):
        raise ValueError('native PCell template synthesis differs')
    if chain.synthesis in ('parameterized', 'cmos_gate'):
        from clay.layout.native_parameterized_row import verify_parameterized_receipt
        if cell is None:
            raise ValueError('parameterized native row source cell is missing')
        verify_parameterized_receipt(cell, chain, folder, receipt, tech)
        return
    if ((not chain.generation_profile and 'generation_profile' in receipt) or
            (chain.generation_profile and receipt.get('generation_profile')!=chain.generation_profile)):
        raise ValueError('native PCell template generation profile differs')
    if (chain.coverage_profile is None and 'coverage_profile' in receipt) or (
            chain.coverage_profile is not None and (
                tech.pdk_id!='sky130' or receipt.get('coverage_profile')!=list(chain.coverage_profile))):
        raise ValueError('native PCell template receipt coverage differs')
    if ((not chain.access_profile and 'access_profile' in receipt)
            or (chain.access_profile and (tech.pdk_id!='sky130'
                or receipt.get('access_profile')!=chain.access_profile))):
        raise ValueError('native PCell template receipt access profile differs')
    if chain.coverage_profile is not None:
        from clay.layout.native_diffusion_pcell import _native_diffusion_generation_tcl
        outputs=receipt.get('outputs')
        if (type(outputs) is not dict or not {'generate.tcl','raw.gds','contacted.gds','drc.txt','native.spice'} <= outputs.keys()
                or (Path(folder)/'generate.tcl').read_text(encoding='utf-8')!=_native_diffusion_generation_tcl(chain)):
            raise ValueError('native PCell template coverage invocation differs')
    if chain.access_profile == 'shared_cross_coupled_m1_v1':
        from clay.layout.native_reset_access import validate_receipt
        if cell is None:raise ValueError('reset access source identity is missing')
        validate_receipt(cell,chain,folder,tech,receipt)
    elif chain.access_profile and candidate is None:
        from clay.layout.native_diffusion_pcell import _native_diffusion_generation_tcl,_native_body_position
        from clay.layout.native_buffer_access import shared_source_contacts_tcl
        import klayout.db as kdb
        outputs=receipt.get('outputs')
        if (type(outputs) is not dict or not {'generate.tcl','contacts.tcl','raw.gds','contacted.gds','drc.txt','native.spice','primitive_flat.ext','scale_um.txt'} <= outputs.keys()
                or (Path(folder)/'generate.tcl').read_text(encoding='utf-8')!=_native_diffusion_generation_tcl(chain)):
            raise ValueError('native PCell template access invocation differs')
        layout=kdb.Layout();layout.read(str(Path(folder)/'raw.gds'));top=layout.top_cell()
        if layout.dbu!=.001 or top is None:raise ValueError('native access raw PCell differs')
        active=kdb.Region(top.begin_shapes_rec(layout.layer(65,20))).merged()
        body_x,body_y=_native_body_position(active.bbox(),'n')
        text,facts=shared_source_contacts_tcl(layout=layout,top=top,chain=chain,tech=tech,
                                              body_x=body_x,body_y=body_y)
        if ((Path(folder)/'contacts.tcl').read_text(encoding='utf-8')!=text or
            receipt.get('access_rule_evidence')!=json.loads(json.dumps(facts))):
            raise ValueError('native PCell access rule or geometry replay differs')


def _source_rows(cell,bundle,tech,rows,source_cir,row_candidates=()):
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place.row_program import RowMemberSpec, _row_geometry_digest_from_inventory
    from clay.layout.place.row_replay_snapshot import freeze_row_values, geometry_snapshot
    from clay.layout.joint_row_geometry import replay_joint_row_geometry
    from clay.layout.spatial.semantic import (SemanticOccurrence, SemanticOverlay, SemanticParameterKey,
        SemanticPhysicalBinding, SemanticTerminal, SemanticBodyWellBinding, semantic_nets_from_terminals)
    from clay.layout.native_diffusion_pcell import __file__ as producer_file

    source_sha,canonical=_row_measure('source.canonical', _require_canonical_native_arrays, cell)
    source_identity=native_source_identity(cell)
    bundle_manifest_sha=_sha(bundle.output_dir/'geometry.json')
    bundle_command=json.loads((bundle.output_dir/'command.json').read_text(encoding='utf-8'))
    bundle_images=tuple(value for value in bundle_command if isinstance(value,str)
                        and value.startswith('sha256:'))
    if len(bundle_images)!=1:
        raise ValueError('native PCell witness bundle image differs')
    if source_cir is not None:
        from clay.layout.generic_placement import _validate_native_row_source_cir
        _validate_native_row_source_cir(cell,source_cir)
        topology_sha=_sha(source_cir)
    else:
        from clay.frontend.native_spice_import import emit_native_spice
        topology_sha=sha256(emit_native_spice(cell,pdk_id='sky130').encode()).hexdigest()
    source_intent = layout_intent_from_cell_def(cell)
    declared=_row_measure('source.chains', declared_native_chains, canonical,layout_intent_from_cell_def(canonical))
    from clay.layout.native_diffusion_producer import _checked_candidates
    candidates=_checked_candidates(canonical,declared,row_candidates)
    poses = {row.group: (row.transforms[0] if row.transforms else 'R0')
             for row in source_intent.arrays if row.style == 'native_diffusion'}
    if tuple(name for name,_chain in declared)!=tuple(name for name,_geometry,_folder in rows):
        raise ValueError('native PCell witness source declaration cover differs')
    devices={device.instance_name:device for device in cell.devices}
    occurrences=[]; terminals=[]; bodies=[]; identities=[]
    for index,((group,chain),(row_group,geometry,folder)) in enumerate(zip(declared,rows)):
        folder=Path(folder); receipt=json.loads((folder/'template-receipt.json').read_text(encoding='utf-8'))
        candidate=candidates.get(group)
        from clay.layout.native_row_candidate import BUFFER_COMPACT_SOURCE, PRECHARGE_SPLIT_GATE
        compact_candidate = candidate is not None and candidate.implementation == BUFFER_COMPACT_SOURCE
        split_candidate = candidate is not None and candidate.implementation == PRECHARGE_SPLIT_GATE
        if compact_candidate:
            from clay.layout import native_buffer_compact_candidate
            producer_admitted = native_buffer_compact_candidate.admits_producer_identity(receipt)
        elif split_candidate:
            from clay.layout import native_precharge_split_candidate
            producer_admitted = native_precharge_split_candidate.admits_producer_identity(receipt)
        elif chain.generation_profile or candidate is not None:
            from clay.layout import native_gate_440
            producer_admitted = native_gate_440.admits_producer_identity(receipt)
        else:
            digest = receipt.get('producer_sha256')
            producer_admitted = (isinstance(digest, str)
                                 and bool(re.fullmatch(r'[0-9a-f]{64}', digest))
                                 and digest != '0' * 64)
        if (receipt.get('source_identity')!=source_identity or receipt.get('source_sha256')!=source_sha
                or receipt.get('model')!=chain.model or receipt.get('unit_width_nm')!=chain.width_nm
                or receipt.get('length_nm')!=chain.length_nm or not producer_admitted
                or receipt.get('native_bundle_manifest_sha256')!=bundle_manifest_sha
                or receipt.get('image')!=bundle_images[0]):
            raise ValueError('native PCell template receipt source differs')
        _row_measure('source.receipt_validation', _validate_coverage_receipt, chain,receipt,folder,tech,cell,candidate)
        if compact_candidate:
            _row_measure('source.compact_receipt', native_buffer_compact_candidate.verify_receipt,
                         cell,chain,folder,receipt,tech,candidate=candidate)
        elif split_candidate:
            _row_measure('source.split_receipt', native_precharge_split_candidate.verify_receipt,
                         cell,chain,folder,receipt,tech,candidate=candidate)
        elif chain.generation_profile or candidate is not None:
            if candidate is None:
                _row_measure('source.gate_receipt', native_gate_440.verify_receipt, cell,chain,folder,receipt,tech)
            else:
                _row_measure('source.gate_receipt', native_gate_440.verify_receipt, cell,chain,folder,receipt,tech,candidate=candidate)
        elif 'candidate_implementation' in receipt:
            raise ValueError('unexpected native row candidate receipt')
        for name,digest in receipt.get('outputs',{}).items():
            if _sha(folder/name)!=digest: raise ValueError('native PCell template artifact changed: '+name)
        expected_names=tuple(sorted({f.owner for f in chain.fingers}))
        if row_group!=group: raise ValueError('native PCell group differs')
        members={}
        for name in expected_names:
            d=devices[name]; spec=native_geometry_spec(d)
            members[name]=RowMemberSpec(device=name,kind=('pmos' if '__pfet_' in chain.model else 'nmos'),
                width_nm=chain.width_nm,length_nm=chain.length_nm,fingers=spec['occurrences'],
                gate_net=d.pins['g'],drain_net=d.pins['d'],source_net=d.pins['s'],body_net=d.pins['b'])
        geometry_before=_row_measure('source.geometry_snapshot', geometry_snapshot, geometry,expected_names)
        members_before=freeze_row_values(members)
        inventory=_row_measure('source.row_replay', replay_joint_row_geometry, geometry,expected_names,tech=tech,members=members)
        if inventory is not geometry.joint_row_binding.joint_inventory:
            raise ValueError('native PCell replay returned a foreign terminal inventory')
        if any(finger.orientation != poses[group] for finger in geometry.fingers):
            raise ValueError('native PCell row orientation differs from source declaration')
        bindings=geometry.joint_row_binding.occurrences
        if tuple((o.owner,o.occurrence_id) for o in bindings)!=tuple(
                (f.owner,f'{f.owner}[{f.occurrence:03d}]') for f in chain.fingers):
            raise ValueError('native PCell occurrence order differs from source chain')
        digest=_row_measure('source.row_digest', _row_geometry_digest_from_inventory, geometry,expected_names,inventory)
        if (_row_measure('source.geometry_snapshot', geometry_snapshot, geometry,expected_names)!=geometry_before
                or freeze_row_values(members)!=members_before):
            raise ValueError('native PCell row operands changed during validation')
        identities.append((group,digest,_sha(folder/'template-receipt.json')))
        dx=index*100_000
        for occurrence,finger in zip(bindings,geometry.fingers):
            member=members[occurrence.owner]; node=occurrence.occurrence_id
            box=tuple(finger.gate_bbox_nm); box=(box[0]+dx,box[1],box[2]+dx,box[3])
            center=((box[0]+box[2])//2,(box[1]+box[3])//2)
            roles=(('bulk',member.body_net),('drain',member.drain_net),
                   ('gate',member.gate_net),('source',member.source_net))
            evidence=('native-pcell:source-chain-and-extracted-masks',)
            occurrences.append(SemanticOccurrence(node,occurrence.owner,
                next(f.occurrence for f in chain.fingers if f.owner==occurrence.owner and
                     f'{f.owner}[{f.occurrence:03d}]'==node),member.kind,
                SemanticParameterKey(member.kind,member.width_nm,member.length_nm,1),
                occurrence.master.variant_sha256,occurrence.master.artwork_sha256,
                SemanticPhysicalBinding(center,box,finger.orientation),tuple(sorted(node+':'+role for role,_ in roles)),
                node+':body-well','native_source_owned',evidence))
            terminals.extend(SemanticTerminal(node+':'+role,node,role,net,'native_source_owned',evidence)
                             for role,net in roles)
            bodies.append(SemanticBodyWellBinding(node+':body-well',node,member.body_net,
                'source_declared_well' if member.kind=='pmos' else 'source_declared_substrate',
                'native_tap_traced',evidence))
    identity=sha256(json.dumps((source_sha,topology_sha,identities),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    terminals=tuple(terminals)
    overlay=SemanticOverlay('native_pcell',identity,topology_sha,tuple(occurrences),terminals,
        semantic_nets_from_terminals(terminals),tuple(bodies))
    return overlay,identity


def _capture_native_pcell_witness(*, cell, bundle, tech, rows, source_cir=None, row_candidates=()):
    if type(bundle) is not NativeGeometryBundle:
        raise TypeError('native PCell witness requires its complete native bundle')
    bundle.require_current(cell)
    rows=tuple((name,geometry,str(Path(folder).resolve())) for name,geometry,folder in rows)
    if not rows or tuple(name for name,_,_ in rows)!=tuple(sorted({name for name,_,_ in rows})):
        raise ValueError('native PCell witness rows must be sorted and unique')
    cir=None if source_cir is None else str(Path(source_cir).resolve(strict=True))
    overlay,identity=_source_rows(cell,bundle,tech,rows,cir,row_candidates)
    from clay.layout import native_diffusion_pcell, native_diffusion_mosaic, native_substrate_binding
    modules=(native_diffusion_pcell,native_diffusion_mosaic,native_substrate_binding)
    if row_candidates:
        from clay.layout import native_row_candidate
        modules+=(native_row_candidate,)
        if any(c.implementation==native_row_candidate.BUFFER_COMPACT_SOURCE for c in row_candidates):
            from clay.layout import native_buffer_compact_candidate
            modules+=(native_buffer_compact_candidate,)
        if any(c.implementation==native_row_candidate.PRECHARGE_SPLIT_GATE for c in row_candidates):
            from clay.layout import native_precharge_split_candidate
            modules+=(native_precharge_split_candidate,)
    from clay.layout.intent import layout_intent_from_cell_def
    declared_list = declared_native_chains(cell, layout_intent_from_cell_def(cell))
    if any(chain.synthesis for _, chain in declared_list):
        from clay.layout import native_parameterized_row
        from clay.pdk import sky130_native_geometry
        modules += (native_parameterized_row, sky130_native_geometry)
        if any(chain.synthesis == 'cmos_gate' for _, chain in declared_list):
            from clay.layout import native_cmos_gate_composer
            modules += (native_cmos_gate_composer,)
    if any(chain.access_profile for _,chain in declared_list):
        from clay.layout import native_buffer_access,pdk_rules
        modules+=native_buffer_access,pdk_rules
    if any(chain.access_profile=='shared_cross_coupled_m1_v1' for _,chain in declared_list):
        from clay.layout import native_reset_access
        modules+=(native_reset_access,)
    if row_candidates or any(chain.generation_profile for _,chain in declared_list):
        from clay.layout import (native_gate_440,native_gate_440_planner,native_gate_440_composer,
                                 native_gate_440_access,native_gate_440_output,native_buffer_access,pdk_rules)
        modules+=native_gate_440,native_gate_440_planner,native_gate_440_composer,native_gate_440_access,native_gate_440_output,native_buffer_access,pdk_rules
    if row_candidates or any(chain.generation_profile=='sky130_lvt_single_440_output_input_m12_v1' for _,chain in declared_list):
        from clay.layout import native_gate_440_input
        modules+=(native_gate_440_input,)
    dependency_paths=[Path(module.__file__).resolve() for module in modules]
    declared=dict(declared_list)
    for group,_geometry,folder in rows:
        if declared[group].access_profile=='shared_cross_coupled_m1_v1':
            dependency_paths.extend(p for p in Path(folder).rglob('*') if p.is_file())
    if row_candidates or any(chain.generation_profile for _,chain in declared_list):
        dependency_paths.extend(native_gate_440.reference_paths())
        for group,_geometry,folder in rows:
            if declared[group].generation_profile or any(c.group==group for c in row_candidates):
                receipt=Path(folder)/'template-receipt.json'
                dependency_paths.append(receipt)
                outputs=json.loads(receipt.read_text(encoding='utf-8'))['outputs']
                dependency_paths.extend(Path(folder)/name for name in outputs)
    dependencies=tuple((str(path),_sha(path)) for path in dict.fromkeys(dependency_paths))
    result=object.__new__(NativePCellWitness)
    for name,value in dict(cell=cell,bundle=bundle,tech=tech,rows=rows,source_cir=cir,
                           overlay=overlay,identity=identity,dependencies=dependencies,row_candidates=row_candidates).items():
        object.__setattr__(result,name,value)
    _WITNESSES[result]=tuple(id(getattr(result,name)) for name in _FIELDS)
    return result


def _rehydrate_sealed_native_pcell_witness(
    *,
    cell,
    bundle,
    tech,
    rows,
    source_cir=None,
    row_candidates=(),
    expected_identity: str,
    overlay,
):
    """Rehydrate a manifest-sealed NativePCellWitness without redundant KLayout extraction."""
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.spatial.semantic import SemanticOverlay

    if type(bundle) is not NativeGeometryBundle:
        raise TypeError('native PCell witness requires its complete native bundle')
    if type(overlay) is not SemanticOverlay or overlay.source_kind != 'native_pcell':
        raise ValueError('native PCell witness overlay differs')
    bundle.require_current(cell)
    rows = tuple((name, geometry, str(Path(folder).resolve())) for name, geometry, folder in rows)
    if not rows or tuple(name for name, _, _ in rows) != tuple(sorted({name for name, _, _ in rows})):
        raise ValueError('native PCell witness rows must be sorted and unique')
    cir = None if source_cir is None else str(Path(source_cir).resolve(strict=True))
    source_sha, canonical = _require_canonical_native_arrays(cell)
    source_identity = native_source_identity(cell)
    declared_list = declared_native_chains(canonical, layout_intent_from_cell_def(canonical))
    if tuple(name for name, _chain in declared_list) != tuple(name for name, _geom, _folder in rows):
        raise ValueError('native PCell witness source declaration cover differs')
    for (_group, chain), (_row_group, _geom, folder) in zip(declared_list, rows):
        folder_path = Path(folder)
        receipt = json.loads((folder_path / 'template-receipt.json').read_text(encoding='utf-8'))
        if (
            receipt.get('source_identity') != source_identity
            or receipt.get('source_sha256') != source_sha
            or receipt.get('model') != chain.model
            or receipt.get('unit_width_nm') != chain.width_nm
            or receipt.get('length_nm') != chain.length_nm
        ):
            raise ValueError('native PCell template receipt source differs')
        for out_name, out_digest in receipt.get('outputs', {}).items():
            if _sha(folder_path / out_name) != out_digest:
                raise ValueError('native PCell template artifact changed: ' + out_name)
    from clay.layout import native_diffusion_pcell, native_diffusion_mosaic, native_substrate_binding
    modules = (native_diffusion_pcell, native_diffusion_mosaic, native_substrate_binding)
    if row_candidates:
        from clay.layout import native_row_candidate
        modules += (native_row_candidate,)
        if any(c.implementation == native_row_candidate.BUFFER_COMPACT_SOURCE for c in row_candidates):
            from clay.layout import native_buffer_compact_candidate
            modules += (native_buffer_compact_candidate,)
        if any(c.implementation == native_row_candidate.PRECHARGE_SPLIT_GATE for c in row_candidates):
            from clay.layout import native_precharge_split_candidate
            modules += (native_precharge_split_candidate,)
    if any(chain.synthesis for _, chain in declared_list):
        from clay.layout import native_parameterized_row
        from clay.pdk import sky130_native_geometry
        modules += (native_parameterized_row, sky130_native_geometry)
        if any(chain.synthesis == 'cmos_gate' for _, chain in declared_list):
            from clay.layout import native_cmos_gate_composer
            modules += (native_cmos_gate_composer,)
    if any(chain.access_profile for _, chain in declared_list):
        from clay.layout import native_buffer_access, pdk_rules
        modules += (native_buffer_access, pdk_rules)
    if any(chain.access_profile == 'shared_cross_coupled_m1_v1' for _, chain in declared_list):
        from clay.layout import native_reset_access
        modules += (native_reset_access,)
    if row_candidates or any(chain.generation_profile for _, chain in declared_list):
        from clay.layout import (
            native_gate_440,
            native_gate_440_planner,
            native_gate_440_composer,
            native_gate_440_access,
            native_gate_440_output,
            native_buffer_access,
            pdk_rules,
        )
        modules += (
            native_gate_440,
            native_gate_440_planner,
            native_gate_440_composer,
            native_gate_440_access,
            native_gate_440_output,
            native_buffer_access,
            pdk_rules,
        )
    if row_candidates or any(chain.generation_profile == 'sky130_lvt_single_440_output_input_m12_v1' for _, chain in declared_list):
        from clay.layout import native_gate_440_input
        modules += (native_gate_440_input,)
    dependency_paths = [Path(module.__file__).resolve() for module in modules]
    declared = dict(declared_list)
    for group, _geometry, folder in rows:
        if declared[group].access_profile == 'shared_cross_coupled_m1_v1':
            dependency_paths.extend(p for p in Path(folder).rglob('*') if p.is_file())
    if row_candidates or any(chain.generation_profile for _, chain in declared_list):
        dependency_paths.extend(native_gate_440.reference_paths())
        for group, _geometry, folder in rows:
            if declared[group].generation_profile or any(c.group == group for c in row_candidates):
                receipt_file = Path(folder) / 'template-receipt.json'
                dependency_paths.append(receipt_file)
                outputs = json.loads(receipt_file.read_text(encoding='utf-8'))['outputs']
                dependency_paths.extend(Path(folder) / name for name in outputs)
    deps = tuple((str(path), _sha(path)) for path in dict.fromkeys(dependency_paths))
    result = object.__new__(NativePCellWitness)
    for name, value in dict(
        cell=cell,
        bundle=bundle,
        tech=tech,
        rows=rows,
        source_cir=cir,
        overlay=overlay,
        identity=expected_identity,
        dependencies=deps,
        row_candidates=row_candidates,
    ).items():
        object.__setattr__(result, name, value)
    _WITNESSES[result] = tuple(id(getattr(result, name)) for name in _FIELDS)
    _require_native_pcell_witness_operands(
        result, cell=cell, tech=tech, overlay=overlay, source_cir=cir
    )
    return result


def _require_native_pcell_witness_operands(value, *, cell, tech, overlay, source_cir):
    """Recheck live identity, source bundle and producer bytes, without graph replay.

    This is only an operand check.  Callers must either do the complete witness
    replay below or hold a validated, invocation-owned row replay snapshot.
    """
    if type(value) is not NativePCellWitness or _WITNESSES.get(value)!=tuple(id(getattr(value,n)) for n in _FIELDS):
        raise ValueError('native PCell witness producer identity differs')
    from clay.layout.place.row_source_projection import _tech_identity
    if _tech_identity(tech) != _tech_identity(value.tech) or overlay is not value.overlay or source_cir!=value.source_cir:
        raise ValueError('native PCell witness operand identity differs')
    value.bundle.require_current(cell)
    for path,digest in value.dependencies:
        if _sha(path)!=digest: raise ValueError('native PCell witness producer dependency changed')
    return value


def require_native_pcell_witness(value, *, cell, tech, overlay, source_cir):
    _row_measure('witness.operands', _require_native_pcell_witness_operands, value,cell=cell,tech=tech,overlay=overlay,source_cir=source_cir)
    replayed,identity=_row_measure('witness.source_rows', _source_rows, cell,value.bundle,tech,value.rows,source_cir,value.row_candidates)
    if identity!=value.identity or _row_measure('witness.overlay_compare', _overlay_differs, replayed, overlay):
        raise ValueError('native PCell witness source replay differs')
    return value


def _overlay_differs(replayed, overlay):
    return replayed.canonical_data() != overlay.canonical_data()
