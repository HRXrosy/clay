"""Draft proof for the source-declared 440 LVT M1/M2 output row only.

This is a narrowly scoped extension of the historical 415 nm complete-stamp
envelope proof, not a general approval of metal2 or via-bearing native rows.
"""
import hashlib
import json
from pathlib import Path
import re
from dataclasses import dataclass

import klayout.db as kdb

PROFILE='sky130_lvt_single_440_output_m12_v1'
INPUT_OUTPUT_PROFILE='sky130_lvt_single_440_output_input_m12_v1'
EXTRA=frozenset(((68,44),(69,20)))
PROOF_SCHEMA='clay.native-feol-source-output-m12-proof.v1'

@dataclass(frozen=True,slots=True)
class SourceOutputFeolProof:
    schema: str
    group: str
    source_sha256: str
    source_identity: str
    template_receipt_sha256: str
    placed_geometry_sha256: str
    deck_sha256: str
    pdk_provenance_sha256: str
    engine_configuration_sha256: str
    via_redundancy: int

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _normalized_gds(path, keys):
    layout=kdb.Layout();layout.read(str(path));top=layout.top_cell()
    if layout.dbu!=.001 or top is None: raise ValueError('output FEOL template GDS differs')
    all_regions={}
    for index in layout.layer_indices():
        info=layout.get_info(index)
        region=kdb.Region(top.begin_shapes_rec(index)).merged()
        if not region.is_empty(): all_regions[(info.layer,info.datatype)]=region
    envelope=kdb.Region()
    for region in all_regions.values():envelope+=region
    if envelope.is_empty():raise ValueError('output FEOL template is empty')
    bbox=envelope.bbox()
    transform=kdb.Trans(-bbox.left,-bbox.bottom)
    return {key:all_regions.get(key,kdb.Region()).transformed(transform).merged() for key in keys}

def _source_policy(cell, source_identity):
    """Resolve the exact source CellDef through the public policy path."""
    from clay.parser import parse_clay_file
    from clay.layout.policy import policy_for_cell
    from clay.layout.route.engine_registry import route_engine_from_policy
    from clay.layout.route.canonical_input_capture import route_engine_configuration
    from clay.layout.native_sky130 import native_source_identity
    doc=parse_clay_file(cell.source_span['file'])
    source_cells=[candidate for candidate in doc.cell_defs if candidate.local_name==cell.local_name]
    if len(source_cells)!=1 or native_source_identity(source_cells[0])!=source_identity:
        raise ValueError('output FEOL policy source identity differs')
    engine=route_engine_from_policy(policy_for_cell(doc,source_cells[0]))
    if type(engine.via_redundancy) is not int or engine.via_redundancy!=2:
        raise ValueError('output FEOL requires source-declared N=2')
    config=route_engine_configuration(engine)
    config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return engine.via_redundancy,config_sha

def _routing_rule_bounds(tech,magic_text):
    """Validate actual PDK values and deck text under the 415 nm envelope."""
    from clay.layout.route.via_model import via_cut_size_nm,via_enclosure_nm
    from clay.layout.route.resource_world import _via_spacing_nm
    if (via_cut_size_nm(tech,'via1'),_via_spacing_nm(tech,'via1'),tuple(via_enclosure_nm(tech,'via1')),
            tuple(tech.via_key(1)),tech.min_space('metal2')) != (150,170,(85,85),(68,44),140):
        raise ValueError('output FEOL source PDK via/M2 rule differs')
    bands=tech.width_spacing_entries('metal2')
    if not bands or any(max(b.spacing.value,b.attached_halo_nm or 0)>415 for b in bands):
        raise ValueError('output FEOL M2 wide spacing exceeds the 415 nm envelope')
    checks=(r'spacing\s+allm2\s+allm2,obsm2,m2fill\s+140\s+touching_ok',
            r'widespacing\s+allm2\s+3005\s+allm2,obsm2,m2fill\s+280\s+touching_ok',
            r'cifspacing\s+m2_large_halo\s+m2_large_halo\s+280\s+touching_ok',
            r'width\s+v1/m1\s+260',r'spacing\s+v1\s+v1\s+60\s+touching_ok')
    if not all(re.search(expr,magic_text) for expr in checks):
        raise ValueError('output FEOL source-bound SKY130 deck differs')
    return max(max(b.spacing.value,b.attached_halo_nm or 0) for b in bands)

def admits(*,bundle,cell,stamps,geometry,placement_um,row_program,row_materialization,tech,
           first,second,magic_text):
    """Replay a source, receipt and actual layer match before widening scope.

    Returns a source-bound proof for the selected profile; returns False for
    other profiles/pairs. A malformed claimed profile raises. The caller
    retains its original fallback for False.
    """
    if tech is None or getattr(tech,'pdk_id',None)!='sky130' or row_program is None or row_materialization is None:
        return False
    base={(64,20),(65,20),(65,44),(66,20),(66,44),(67,20),(67,44),
          (68,20),(93,44),(94,20),(95,20),(125,44)}
    extras={s.stamp_id:frozenset(geometry[s.stamp_id].polys_by_layer)-base for s in (first,second)}
    # Two rows both carrying added layers require a separate pair proof;
    # retain the existing conservative fallback for that case.
    hits=[name for name,values in extras.items() if values]
    if len(hits)!=1 or extras[hits[0]]!=EXTRA:return False
    group=hits[0]
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.native_diffusion_source import declared_native_chains, _require_native_pcell_witness_operands, _require_canonical_native_arrays
    chains=dict(declared_native_chains(cell,layout_intent_from_cell_def(cell)))
    chain=chains.get(group)
    if chain is None:return False
    projection=row_materialization.source_projection
    witness=getattr(projection,'native_witness',None)
    if witness is None:
        if chain.generation_profile in (PROFILE,INPUT_OUTPUT_PROFILE):
            raise ValueError('output FEOL source witness missing')
        return False
    if (chain.generation_profile not in (PROFILE,INPUT_OUTPUT_PROFILE)
            and not any(c.group==group for c in getattr(witness,'row_candidates',()))):
        return False
    _require_native_pcell_witness_operands(witness,cell=cell,tech=tech,
        overlay=witness.overlay,source_cir=projection.source_cir)
    from clay.layout.native_row_candidate import witness_implementation
    profile,candidate=witness_implementation(cell,chain,witness,group)
    if profile not in (PROFILE,INPUT_OUTPUT_PROFILE):return False
    entries=[entry for entry in row_program.entries if entry.group==group]
    if len(entries)!=1 or set(entries[0].consumed_devices)!={f.owner for f in chain.fingers}:
        raise ValueError('output FEOL source row ownership differs')
    # Limit the proof to the original untransformed source row. The selected
    # witness is captured by the ordinary source projection, not reconstructed.
    declarations=[r for r in layout_intent_from_cell_def(cell).arrays if r.group==group]
    if len(declarations)!=1 or declarations[0].transforms not in ((),('R0',)):
        return False
    rows=[row for row in witness.rows if row[0]==group]
    if len(rows)!=1:raise ValueError('output FEOL witness row differs')
    # Both exact source row owners, including the other P/N stamp, must still
    # be admitted through the complete row-materialization/source replay.
    from clay.layout.place.native_feol_reservation import sky130_native_row_kinds
    kinds=sky130_native_row_kinds(bundle=bundle,cell=cell,stamps=stamps,geometry=geometry,
        row_program=row_program,row_materialization=row_materialization,tech=tech)
    if any(name not in kinds for name in (first.stamp_id,second.stamp_id)):
        raise ValueError('output FEOL pair has an unproved source row')
    folder=Path(rows[0][2]);receipt=json.loads((folder/'template-receipt.json').read_text())
    source_sha,_canonical=_require_canonical_native_arrays(cell)
    via_redundancy,config_sha=_source_policy(cell,chain.source_identity)
    from clay.layout import native_gate_440
    if (receipt.get('source_sha256')!=source_sha or receipt.get('source_identity')!=chain.source_identity
            or (candidate is None and receipt.get('generation_profile')!=chain.generation_profile) or
            receipt.get('native_bundle_manifest_sha256')!=_sha(bundle.output_dir/'geometry.json') or
            not native_gate_440.admits_producer_identity(receipt)):
        raise ValueError('output FEOL receipt source differs')
    if candidate is None:
        native_gate_440.verify_receipt(cell,chain,folder,receipt,tech)
    else:
        native_gate_440.verify_receipt(cell,chain,folder,receipt,tech,candidate=candidate)
    # Replay the actual placed stamp against the full profile template's added
    # layers in a translation-normalized frame; full stamp digest is checked by
    # the existing caller both here and again after composite merging.
    from clay.layout.place.edge_rail_geometry import _stamp_geometry
    from clay.layout.place.native_feol_reservation import _regions
    box,digest=_stamp_geometry(geometry[group],placement_um[group])
    stamp=next(s for s in (first,second) if s.stamp_id==group)
    if digest!=stamp.geometry_sha256:raise ValueError('output FEOL placed stamp changed')
    keys={'via1':(68,44),'metal2':(69,20)}
    actual=_regions(geometry[group],placement_um[group],(-box[0],-box[1]),keys)
    expected=_normalized_gds(folder/'contacted.gds',EXTRA)
    if any(not (actual[name]^expected[key]).is_empty() for name,key in keys.items()):
        raise ValueError('output FEOL source-owned M2/via artwork differs')
    _routing_rule_bounds(tech,magic_text)
    from clay.layout.route.resource_world import pdk_provenance_digest
    return SourceOutputFeolProof(PROOF_SCHEMA,group,source_sha,chain.source_identity,
        _sha(folder/'template-receipt.json'),digest,_sha(bundle.output_dir/'sky130A.tech'),
        pdk_provenance_digest(tech),config_sha,via_redundancy)
