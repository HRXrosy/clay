"""Source-bound channel evidence for matching within one native row.

These measurements are channel occurrences, not independent PCell footprints.
The complete source, native masks, contacts and row projection are revalidated
before capture and again at every consumption.
"""
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from weakref import WeakKeyDictionary


@dataclass(frozen=True, eq=False, init=False)
class NativeRowMatchingEvidence:
    placement: object
    cell: object
    pair: tuple
    record: tuple

    def __init__(self, *args, **kwargs):
        raise TypeError('native row matching evidence requires source-bound capture')


_CAPTURED=WeakKeyDictionary()


@dataclass(frozen=True, eq=False, init=False)
class NativeRowMatchingInventory:
    """Live pair inventory, signed separately from the mutable lookup dict."""
    pairs: tuple
    evidence: tuple

    def __init__(self,*args,**kwargs):
        raise TypeError('native row matching inventory requires source-bound capture')


_INVENTORIES=WeakKeyDictionary()


def require_native_row_matching_inventory(inventory, evidence):
    if type(inventory) is not NativeRowMatchingInventory or type(evidence) is not dict:
        raise ValueError('native row matching inventory is missing or foreign')
    captured=_INVENTORIES.get(inventory)
    if (captured is None or captured!=(inventory.pairs,tuple(id(v) for v in inventory.evidence))
            or tuple(sorted(evidence))!=inventory.pairs
            or len(inventory.pairs)!=len(inventory.evidence)
            or any(evidence[pair] is not item for pair,item in zip(inventory.pairs,inventory.evidence))):
        raise ValueError('native row matching inventory or evidence changed')
    return inventory


def _candidate_pairs(placement, cell):
    """Select from source declarations before checking the live row witness."""
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place.row_program import _group_map
    from clay.layout.native_diffusion_source import NativePCellWitness
    native=any(getattr(device,'device_class',None)=='native'
               for device in (getattr(cell,'devices',()) or ()))
    span=getattr(cell,'source_span',None)
    if (native or getattr(placement,'native_geometry',None) is not None) and isinstance(span,dict) and span.get('file'):
        # The caller may have altered or removed its in-memory declarations.
        # Discover native matching from the canonical source bytes, not from
        # an object that could hide the need for a witness.
        from clay.layout.route.exact_artifact_evidence import _replayed_source_cell_authority
        cell=_replayed_source_cell_authority(cell)[3]
    declared=tuple(decl for decl in (getattr(cell,'layout_intents',()) or ())
                   if getattr(decl,'kind',None)=='array'
                   and getattr(decl,'attributes',{}).get('style')=='native_diffusion')
    if not declared:
        return ()
    intent=layout_intent_from_cell_def(cell)
    groups=_group_map(cell)
    native_groups={row.group:tuple(groups.get(row.group,())) for row in intent.arrays
                   if row.style=='native_diffusion'}
    expected=[]
    for match in intent.matching_intents:
        if match.requires_interdigitation:
            continue
        pair=tuple(sorted(match.members))
        owners=[group for group,members in native_groups.items() if set(pair)<=set(members)]
        if len(owners)>1:raise ValueError('native row matching source owner is ambiguous')
        if owners:expected.append((pair,owners[0]))
    if not expected:
        return ()
    materialization=getattr(placement,'row_materialization',None)
    projection=getattr(materialization,'source_projection',None)
    witness=getattr(projection,'native_witness',None)
    program=getattr(placement,'row_program',None)
    if materialization is None or projection is None or type(witness) is not NativePCellWitness or program is None:
        raise ValueError('native row matching source witness is missing')
    witness_groups={name for name,_geometry,_folder in witness.rows}
    for pair,group in expected:
        entries=[entry for entry in program.entries if entry.group==group]
        if (group not in witness_groups or len(entries)!=1
                or set(entries[0].consumed_devices)!=set(native_groups[group])):
            raise ValueError('native row matching source row is partial or missing')
    return tuple(sorted({pair for pair,_group in expected}))


def _measure(placement,cell,pair):
    from clay.layout.generic_placement import verify_row_source_compiler_authority,verify_stored_row_projection
    from clay.layout.joint_row_geometry import replay_joint_row_geometry,_region,_shapes
    from clay.layout.joint_occurrence_center_basis import replay_joint_channel_centers
    from clay.layout.joint_terminal_inventory import _native_frame
    from clay.layout.joint_row_geometry import joint_d4_orientation
    from clay.layout.route.exact_artifact_evidence import _replayed_source_cell_authority
    from clay.layout.tech import LayoutTech
    from clay.pdk.sky130_native_geometry import native_geometry_spec
    import klayout.db as kdb

    if type(pair) is not tuple or len(pair)!=2 or pair!=tuple(sorted(set(pair))):
        raise ValueError('native row matching pair is malformed')
    _source,_inventory,_classes,canonical=_replayed_source_cell_authority(cell)
    tech=LayoutTech.resolve(placement.row_program.tech_id)
    program,materialization=verify_row_source_compiler_authority(placement,canonical,tech)
    if pair not in _candidate_pairs(placement,canonical):
        raise ValueError('native row matching source pair changed')
    verify_stored_row_projection(placement)
    entries=[entry for entry in program.entries if set(pair)<=set(entry.consumed_devices)]
    if len(entries)!=1:raise ValueError('native row matching owner is not unique')
    entry,=entries
    realizations=[r for r in materialization.realizations if r.entry_id==entry.entry_id]
    bindings=[b for b in placement.row_final_certificate.macro_bindings if b.entry_id==entry.entry_id]
    if len(realizations)!=1 or len(bindings)!=1:
        raise ValueError('native row matching realization or placement is missing')
    realization,=realizations;placed,=bindings
    members={m.device:m for m in program.members}
    geom=realization.geometry
    replay_joint_row_geometry(geom,tuple(sorted(entry.consumed_devices)),tech=tech,members=members)
    source=geom.joint_row_binding
    centers=replay_joint_channel_centers(source,tech=tech,members=members)
    records={r.occurrence_id:r for r in centers.records}
    dx=placed.bbox_nm[0]-realization.local_bbox_nm[0]
    dy=placed.bbox_nm[1]-realization.local_bbox_nm[1]
    if (placed.bbox_nm[2]-realization.local_bbox_nm[2],
            placed.bbox_nm[3]-realization.local_bbox_nm[3])!=(dx,dy):
        raise ValueError('native row matching placement changed scale or artwork')
    devices={d.instance_name:d for d in canonical.devices}
    rows=[]
    for owner in pair:
        if owner not in devices:raise ValueError('native row matching source owner is missing')
        spec=native_geometry_spec(devices[owner])
        signature=(spec['model'],spec['w_nm'],spec['l_nm'],spec['nf'],
                   spec['m'],spec['mult'],spec['occurrences'])
        occurrences=[]
        for occ in source.occurrences:
            if occ.owner!=owner:continue
            channel=_region(records[occ.occurrence_id].channel_geometry_nm).transformed(_native_frame(occ.master_to_local))
            box=channel.bbox()
            if channel.is_empty() or box.width()<=0 or box.height()<=0:
                raise ValueError('native row matching actual channel is empty')
            normalized=_shapes(channel.transformed(kdb.Trans(-box.left,-box.bottom)))
            occurrences.append((box.left+box.right+2*dx,box.bottom+box.top+2*dy,
                                box.width(),box.height(),normalized,joint_d4_orientation(occ.master_to_local)))
        if len(occurrences)!=spec['occurrences'] or not occurrences:
            raise ValueError('native row matching source occurrence denominator differs')
        rows.append((signature,tuple(sorted(occurrences))))
    axis=placement.matching_pair_axes.get(pair)
    if (type(axis) is not tuple or len(axis)!=2 or axis[0] not in ('x','y')
            or type(axis[1]) not in (int,float)):
        raise ValueError('native row matching source axis is absent')
    return (pair,axis,tuple(rows))


def capture_native_row_matching(placement,cell):
    result={}
    pairs = _candidate_pairs(placement,cell)
    if pairs:
        from clay.layout.place.row_replay_session import native_matching_row_scope
        from clay.layout.tech import LayoutTech
        program=placement.row_program
        with native_matching_row_scope(program=program,materialization=placement.row_materialization,
                                       tech=LayoutTech.resolve(program.tech_id)):
            for pair in pairs:
                record=_measure(placement,cell,pair)
                evidence=object.__new__(NativeRowMatchingEvidence)
                for name,value in (('placement',placement),('cell',cell),('pair',pair),('record',record)):
                    object.__setattr__(evidence,name,value)
                _CAPTURED[evidence]=(id(placement),id(cell),pair,record)
                result[pair]=evidence
    if not result:
        return result,None
    inventory=object.__new__(NativeRowMatchingInventory)
    pairs=tuple(sorted(result)); items=tuple(result[pair] for pair in pairs)
    object.__setattr__(inventory,'pairs',pairs)
    object.__setattr__(inventory,'evidence',items)
    _INVENTORIES[inventory]=(pairs,tuple(id(item) for item in items))
    return result,inventory


def replay_native_row_matching(evidence,pair):
    if type(evidence) is not NativeRowMatchingEvidence:
        raise ValueError('native row matching evidence type differs')
    captured=_CAPTURED.get(evidence)
    if captured!=(id(evidence.placement),id(evidence.cell),evidence.pair,evidence.record) or evidence.pair!=pair:
        raise ValueError('native row matching evidence origin differs')
    from clay.layout.place.row_replay_session import native_matching_row_scope
    from clay.layout.tech import LayoutTech
    program=evidence.placement.row_program
    with native_matching_row_scope(program=program,materialization=evidence.placement.row_materialization,
                                   tech=LayoutTech.resolve(program.tech_id)):
        if _measure(evidence.placement,evidence.cell,pair)!=evidence.record:
            raise ValueError('native row matching evidence replay differs')
    _pair,axis,rows=evidence.record
    signatures=[r[0] for r in rows]
    occurrences=[r[1] for r in rows]
    axis_index=0 if axis[0]=='x' else 1
    perp=1-axis_index
    centroids=[tuple(Fraction(sum(o[i] for o in group),2*len(group)) for i in (0,1))
               for group in occurrences]
    span=max(max(o[axis_index] for group in occurrences for o in group)-
             min(o[axis_index] for group in occurrences for o in group),2)
    source_axis=Fraction(Decimal(str(axis[1])))
    residual=float(max(abs((centroids[0][axis_index]+centroids[1][axis_index])/2-source_axis),
                       abs(centroids[0][perp]-centroids[1][perp]))/Fraction(span,2))
    profiles=[Counter((o[2],o[3],o[4]) for o in group) for group in occurrences]
    patterns=[]
    for index,group in enumerate(occurrences):
        center2=tuple(sum(Fraction(o[i]) for o in group)/len(group) for i in (0,1))
        pattern=[]
        for occurrence in group:
            offset=[Fraction(occurrence[i])-center2[i] for i in (0,1)]
            if index==1:offset[axis_index]=-offset[axis_index]
            pattern.append(tuple(offset))
        patterns.append(Counter(pattern))
    footprint=0.0 if (signatures[0]==signatures[1] and profiles[0]==profiles[1]
                      and patterns[0]==patterns[1]) else 1.0
    orientations=[Counter(o[5] for o in group) for group in occurrences]
    orientation=0.0 if orientations[0]==orientations[1] else 1.0
    return residual,footprint,orientation
