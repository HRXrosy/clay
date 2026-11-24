"""Materialize explicitly declared native diffusion rows from current source."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

from clay.layout.native_composite_row import _capture_native_composite_row
from clay.layout.native_diffusion_mosaic import build_native_diffusion_mosaic
from clay.layout.native_diffusion_pcell import materialize_native_diffusion_template, _sha
from clay.layout.native_diffusion_source import (declared_native_chains, _capture_native_pcell_witness,
                                                _require_canonical_native_arrays)
from clay.layout.place.row_program import materialize_row_program, verify_row_program_materialization
from clay.layout.place.row_source_projection import _capture_row_source_projection
from clay.layout.spatial import derive_abstract_spatial_ir, project_discrete_mosaic


# These generation receipts encode actual source owners, occurrences and nets.
_SOURCE_BOUND_LVT_PROFILES = frozenset((
    'sky130_lvt_single_440_v1',
    'sky130_lvt_single_440_output_m12_v1',
    'sky130_lvt_single_440_output_input_m12_v1',
))


def _source_access_role_key(chain):
    """Bind row-specific added access to every source occurrence and role."""
    data = asdict(chain)
    # Optional synthesis fields default to "" and must not perturb legacy LVT/candidate keys.
    if not data.get('synthesis'):
        data.pop('synthesis', None)
    if not data.get('tap_rail'):
        data.pop('tap_rail', None)
    payload=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=True)
    return sha256(payload.encode('utf8')).hexdigest()


def _native_template_key(chain, candidate=None):
    """Identify the native PCell artwork, including its actual finger count."""
    base=(chain.model, chain.width_nm, chain.length_nm, len(chain.fingers))
    profile=getattr(chain,'coverage_profile',None)
    if profile is not None:base=base+(profile,)
    if chain.access_profile:base=base+(chain.access_profile,)
    if chain.access_profile=='shared_cross_coupled_m1_v1':base=base+(_source_access_role_key(chain),)
    if chain.generation_profile:
        base=base+(chain.generation_profile,)
        if chain.generation_profile in _SOURCE_BOUND_LVT_PROFILES:
            base=base+(_source_access_role_key(chain),)
    if getattr(chain, 'synthesis', ''):
        base=base+('synthesis', chain.synthesis, chain.tap_rail, _source_access_role_key(chain))
    if candidate is not None:
        base=base+('candidate',candidate.implementation,candidate.source_sha256,candidate.group,_source_access_role_key(chain))
    return base


def _native_template_folder(chain, candidate=None):
    # Retain the complete SHA while keeping Windows/Docker bind paths short.
    if candidate is not None:
        return 'candidate-'+sha256(repr(_native_template_key(chain,candidate)).encode()).hexdigest()
    if getattr(chain, 'synthesis', ''):
        return 'param-'+_source_access_role_key(chain)
    if chain.generation_profile in _SOURCE_BOUND_LVT_PROFILES:
        return 'lvt-'+_source_access_role_key(chain)
    base=f'{chain.model}-{chain.width_nm}-{chain.length_nm}-{len(chain.fingers)}'
    if chain.coverage_profile is not None:
        source,drain=chain.coverage_profile
        base+='-source-'+source.replace('+','p').replace('-','m')+'-drain-'+drain.replace('+','p').replace('-','m')
    if chain.access_profile:base+='-access-'+chain.access_profile
    if chain.access_profile=='shared_cross_coupled_m1_v1':base+='-source-'+_source_access_role_key(chain)
    return base if not chain.generation_profile else base+'-generation-'+chain.generation_profile


def materialize_declared_native_diffusion(cell, intent, tech, bundle, *, output_dir,
                                          source_cir=None, timeout_seconds=60, row_candidates=()):
    """Generate the declared disjoint source rows against the existing bundle."""
    _require_canonical_native_arrays(cell,intent)
    chains=declared_native_chains(cell,intent)
    candidates=_checked_candidates(cell,chains,row_candidates)
    if not chains:
        return intent,None
    if any(chain.coverage_profile is not None or chain.access_profile for _,chain in chains) and tech.pdk_id!='sky130':
        raise ValueError('native diffusion contact coverage requires SKY130 PDK')
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=False)
    templates={}
    for _group,chain in chains:
        candidate=candidates.get(_group)
        key=_native_template_key(chain,candidate)
        if key not in templates:
            folder=output/_native_template_folder(chain,candidate)
            if candidate is not None:
                from clay.layout.native_row_candidate import BUFFER_COMPACT_SOURCE, PRECHARGE_SPLIT_GATE
                if candidate.implementation == BUFFER_COMPACT_SOURCE:
                    from clay.layout.native_buffer_compact_candidate import materialize as materialize_typed
                elif candidate.implementation == PRECHARGE_SPLIT_GATE:
                    from clay.layout.native_precharge_split_candidate import materialize as materialize_typed
                else:
                    from clay.layout.native_gate_440 import materialize as materialize_typed
                materialize_typed(cell,bundle,chain,tech=tech,output_dir=folder,timeout_seconds=timeout_seconds,candidate=candidate)
            elif chain.access_profile == 'shared_cross_coupled_m1_v1':
                from clay.layout.native_reset_access import materialize
                materialize(cell,bundle,chain,tech=tech,output_dir=folder,timeout_seconds=timeout_seconds)
            elif getattr(chain, 'synthesis', '') in ('parameterized', 'cmos_gate'):
                from clay.layout.native_parameterized_row import materialize_parameterized_diffusion_template
                materialize_parameterized_diffusion_template(
                    cell, bundle, chain, tech=tech, output_dir=folder, timeout_seconds=timeout_seconds
                )
            elif chain.generation_profile:
                from clay.layout.native_gate_440 import materialize as materialize_typed
                materialize_typed(cell,bundle,chain,tech=tech,output_dir=folder,timeout_seconds=timeout_seconds)
            else:
                materialize_native_diffusion_template(cell,bundle,chain,output_dir=folder,
                                                      timeout_seconds=timeout_seconds)
            templates[key]=folder
    if row_candidates:
        return _materialize_native_diffusion_templates(cell,intent,tech,bundle,chains,templates,
                                                       source_cir=source_cir,row_candidates=row_candidates)
    return _materialize_native_diffusion_templates(cell,intent,tech,bundle,chains,templates,source_cir=source_cir)


def _checked_candidates(cell, chains, row_candidates):
    from clay.layout.native_row_candidate import NativeRowCandidate, require_candidate
    if type(row_candidates) is not tuple or any(type(x) is not NativeRowCandidate for x in row_candidates):
        raise TypeError('native row candidates must be an explicit tuple')
    choices={x.group:x for x in row_candidates}
    if len(choices)!=len(row_candidates): raise ValueError('duplicate native row candidate')
    source_chains=dict(chains)
    for group,candidate in choices.items():
        if group not in source_chains: raise ValueError('candidate group is not declared')
        require_candidate(cell,source_chains[group],candidate)
    return choices


def _materialize_native_diffusion_templates(cell, intent, tech, bundle, chains, templates, *, source_cir=None, row_candidates=()):
    """Replay generated templates through the same source-bound materializer."""
    _require_canonical_native_arrays(cell,intent)
    declared = declared_native_chains(cell,intent)
    candidates=_checked_candidates(cell,declared,row_candidates)
    if any(chain.coverage_profile is not None or chain.access_profile for _,chain in declared) and tech.pdk_id!='sky130':
        raise ValueError('native diffusion contact coverage requires SKY130 PDK')
    if chains != declared or set(templates) != {_native_template_key(chain,candidates.get(group)) for group,chain in declared}:
        raise ValueError('native diffusion template inventory differs from source')
    poses = {row.group: (row.transforms[0] if row.transforms else 'R0')
             for row in intent.arrays if row.style == 'native_diffusion'}
    def source_row(group, chain):
        folder = templates[_native_template_key(chain,candidates.get(group))]
        geometry = build_native_diffusion_mosaic(cell, chain, folder, tech)
        if poses[group] != 'R0':
            from clay.layout.mosaic_geometry import transform_mosaic_geometry
            geometry = transform_mosaic_geometry(geometry, poses[group], tech=tech)
        return group, geometry, folder
    rows=tuple(source_row(group, chain) for group,chain in chains)
    witness=_capture_native_pcell_witness(cell=cell,bundle=bundle,tech=tech,rows=rows,source_cir=source_cir,row_candidates=row_candidates)
    overlay=witness.overlay
    asir=derive_abstract_spatial_ir(overlay)
    requests=tuple((group,tuple(sorted({finger.owner for finger in chain.fingers})))
                   for group,chain in chains)
    mosaics=tuple(project_discrete_mosaic(asir,mosaic_id=group,logical_owners=owners)
                  for group,owners in requests)
    projection=_capture_row_source_projection(canonical_source_cell=cell,authoring_source_cell=cell,
        tech=tech,reference_overlay=overlay,reference_asir=asir,mosaic_requests=requests,
        discrete_mosaics=mosaics,source_cir=source_cir,native_witness=witness)
    projected_intent=replace(intent,discrete_mosaics=mosaics)
    program=projection.projected_program
    geometries={name:geometry for name,geometry,_folder in rows}
    by_mosaic={mosaic.mosaic_id:mosaic for mosaic in mosaics}
    overrides={entry.owner:replace(geometries[entry.group],
        mosaic_lattice_intent=(by_mosaic[entry.group] if by_mosaic[entry.group].column_ranks else None),tech=tech)
        for entry in program.entries}
    materialization=materialize_row_program(program,tech,geometry_overrides=overrides,source_projection=projection)
    if materialization.residuals or len(materialization.realizations)!=len(chains):
        raise ValueError('declared native diffusion materialization failed: '+repr(materialization.residuals))
    verify_row_program_materialization(materialization,program=program,tech_id=tech.pdk_id)
    from clay.layout import (native_composite_row,native_diffusion_source,native_diffusion_pcell,
                             native_diffusion_mosaic,native_substrate_binding,joint_row_geometry,
                             joint_occurrence_center_basis)
    from clay.layout.place import row_program,row_source_projection
    modules=(native_composite_row,native_diffusion_source,native_diffusion_pcell,native_diffusion_mosaic,
             native_substrate_binding,joint_row_geometry,joint_occurrence_center_basis,row_program,row_source_projection)
    if candidates:
        from clay.layout import native_row_candidate
        modules+=(native_row_candidate,)
        if any(c.implementation==native_row_candidate.BUFFER_COMPACT_SOURCE for c in candidates.values()):
            from clay.layout import native_buffer_compact_candidate
            modules+=(native_buffer_compact_candidate,)
        if any(c.implementation==native_row_candidate.PRECHARGE_SPLIT_GATE for c in candidates.values()):
            from clay.layout import native_precharge_split_candidate
            modules+=(native_precharge_split_candidate,)
    if any(getattr(chain, 'synthesis', '') for _, chain in chains):
        from clay.layout import native_parameterized_row
        from clay.pdk import sky130_native_geometry
        modules += (native_parameterized_row, sky130_native_geometry)
        if any(getattr(chain, 'synthesis', '') == 'cmos_gate' for _, chain in chains):
            from clay.layout import native_cmos_gate_composer
            modules += (native_cmos_gate_composer,)
    if any(chain.access_profile for _,chain in chains):
        from clay.layout import native_buffer_access,pdk_rules
        modules+=native_buffer_access,pdk_rules
    if any(chain.access_profile=='shared_cross_coupled_m1_v1' for _,chain in chains):
        from clay.layout import native_reset_access
        modules+=(native_reset_access,)
    if candidates or any(chain.generation_profile for _,chain in chains):
        from clay.layout import (native_gate_440,native_gate_440_planner,native_gate_440_composer,
                                 native_gate_440_access,native_gate_440_output,native_buffer_access,pdk_rules)
        modules+=native_gate_440,native_gate_440_planner,native_gate_440_composer,native_gate_440_access,native_gate_440_output,native_buffer_access,pdk_rules
    if candidates or any(chain.generation_profile=='sky130_lvt_single_440_output_input_m12_v1' for _,chain in chains):
        from clay.layout import native_gate_440_input
        modules+=(native_gate_440_input,)
    paths=[Path(__file__).resolve(),*(Path(module.__file__).resolve() for module in modules)]
    if candidates or any(chain.generation_profile for _,chain in chains):
        paths.extend(native_gate_440.reference_paths())
    typed_keys={_native_template_key(chain,candidates.get(group)) for group,chain in chains if chain.generation_profile or group in candidates}
    for key,folder in templates.items():
        paths.extend(path for path in (folder.rglob('*') if key in typed_keys or 'shared_cross_coupled_m1_v1' in key else folder.iterdir()) if path.is_file())
    dependencies=tuple((str(path),_sha(path)) for path in dict.fromkeys(paths))
    first_folder=Path(rows[0][2]); receipt=first_folder/'template-receipt.json'
    producer=_capture_native_composite_row(cell=cell,native_bundle=bundle,tech=tech,projection=projection,
        program=program,materialization=materialization,geometry=materialization.realizations[0].geometry,
        receipt_path=receipt,receipt_sha256=_sha(receipt),dependencies=dependencies)
    return projected_intent,producer
