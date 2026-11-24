"""Current measured MOS ingredients and coordinate-free construction selections.

The complete reference frame appears only in the semantic observation. Native
constructors receive local masters, sites, accessories and interface identities.
This object depends on the original active reference lifetime and cannot import
authority from a serialized record or certify body tracing, DRC, LVS or quality.
"""
from collections import Counter
from dataclasses import asdict

from clay.frontend.reference_device_artwork import CurrentReferenceDeviceArtwork, _digest
from clay.frontend.reference_device_shapes import _map_shape, _region, _require, _shapes
from clay.frontend.source_revision import reopen_complete_source
from clay.frontend.unit_master_geometry import (
    UnitMasterSource, UnitTerminalSite, normalize_unit_master_geometry, validate_unit_master_regions,
)
from clay.layout.joint_terminal_inventory import TerminalSiteRequirement
from clay.layout.mos_mosaic_materializer import MosMosaicRawUnit, MosMosaicAccessory, MosMosaicSharedInterface
from clay.layout.place.row_replay_snapshot import freeze_row_values


def _inverse(frame):
    a,b,c,d,dx,dy = frame
    return a,c,b,d,-a*dx-c*dy,-b*dx-d*dy


def _raw_origin(record):
    points = tuple(point for _key,shapes in record.raw_master_geometry for shape in shapes for point in shape.hull_nm)
    return min(x for x,_ in points),min(y for _,y in points)


def _raw_unit(record, *, node_id, pdk_id, gds_sha256, observation_sha256, net_to_node):
    from clay.frontend._reference_device_measurement import POLY_WIRE, N_SD, P_SD
    _require(record.family in ('pmos','nmos') and record.primitive.center_basis == 'channel_bbox_center',
             'current raw MOS requires actual channel semantics')
    inverse = _inverse(record.raw_to_global)
    actual = Counter((key,shape) for key,shapes in record.raw_master_geometry for shape in shapes)
    replayed = Counter((key,_map_shape(shape,inverse)) for key,shape in record.parts)
    _require(actual == replayed, 'current raw MOS complete master multiset differs')
    sites = tuple(sorted((UnitTerminalSite(role,'gate' if role == 'gate' else 'sd',
        ((layer,tuple(sorted(_map_shape(shape,inverse) for shape in shapes))),))
        for role,layer,shapes in record.primitive.terminals),key=lambda site:site.site_id))
    _require(tuple(site.site_id for site in sites) == ('gate','sd0','sd1'), 'complete raw MOS gate/SD sites differ')
    master = normalize_unit_master_geometry(pdk_id=pdk_id,parameter_class=record.parameter_class,
        sources=(UnitMasterSource(gds_sha256,record.master_name,observation_sha256),),
        geometry_by_layer_nm=tuple((key,_shapes(_region(shapes))) for key,shapes in record.raw_master_geometry),
        terminal_sites=sites,
        unresolved_access_obligations=('joint_artwork_terminal_access_required',))
    validate_unit_master_regions(master)
    pins,nodes = dict(record.source_terminals),dict(record.physical_terminals)
    requirements = []
    for site in master.terminal_sites:
        roles = tuple(role for role in (('g',) if site.role_group == 'gate' else ('d','s'))
                      if net_to_node[pins[role]] == nodes[site.site_id])
        _require(roles, 'raw MOS site lacks its actual source terminal net alias')
        nets = {pins[role] for role in roles}
        _require(len(nets) == 1, 'raw MOS site source net assignment is ambiguous')
        key,shapes = site.geometry_by_layer_nm[0]
        query = POLY_WIRE if site.role_group == 'gate' else P_SD if record.family == 'pmos' else N_SD
        requirements.append(TerminalSiteRequirement(node_id+'/'+site.site_id,record.logical_id,node_id,
            next(iter(nets)),roles,key,query,shapes,master.variant_sha256,observation_sha256))
    ox,oy = _raw_origin(record)
    emitted = tuple((key,tuple(_map_shape(shape,(1,0,0,1,-ox,-oy)) for shape in shapes))
                    for key,shapes in record.raw_master_geometry)
    # The merged mask is only the query/site model. Every original polygon,
    # including shared edges and repetitions, remains in emitted raw artwork.
    return MosMosaicRawUnit(node_id,master,tuple(requirements),raw_geometry_by_layer_nm=emitted)


def _accessories(records, identities, accessories, provenance):
    by_path = {record.occurrence_path:record for record in records}
    result = []
    for item in accessories:
        _require(set(item.occurrence_paths) <= set(by_path), 'accessory source occurrence denominator differs')
        anchor_path = min(item.occurrence_paths,key=lambda path:identities[by_path[path].occurrence_id])
        record = by_path[anchor_path]
        inverse = _inverse(record.raw_to_global)
        ox,oy = _raw_origin(record)
        local = (*inverse[:4],inverse[4]-ox,inverse[5]-oy)
        layers = tuple((key,tuple(sorted(_map_shape(shape,local) for layer,shape in item.parts if layer == key)))
                       for key in sorted({key for key,_shape in item.parts}))
        ids = tuple(sorted(identities[by_path[path].occurrence_id] for path in item.occurrence_paths))
        result.append(MosMosaicAccessory(item.occurrence_path,identities[record.occurrence_id],ids,layers,provenance))
    return tuple(sorted(result,key=lambda item:item.part_id))


def _interfaces(records, identities, semantic_layers, checkpoint):
    from clay.layout.fixed_artwork_abutment import align_fixed_artwork_abutment
    result = []
    keys = dict(semantic_layers)
    def artwork(record):
        return tuple((key,tuple(shape for layer,shape in record.parts if key == layer))
                     for key in sorted({key for key,_shape in record.parts}))
    for index,left in enumerate(records):
        for right in records[index+1:]:
            checkpoint()
            if left.family != right.family:
                continue
            a,b = sorted((left,right),key=lambda row:(row.center2_nm[0],row.occurrence_id))
            for left_role,left_key,left_shapes in a.primitive.terminals:
                if not left_role.startswith('sd'):
                    continue
                left_region = _region(left_shapes)
                for right_role,right_key,right_shapes in b.primitive.terminals:
                    if not right_role.startswith('sd') or left_key != right_key:
                        continue
                    right_region = _region(right_shapes)
                    if left_region.interacting(right_region).is_empty():
                        continue
                    names = a.occurrence_id+'/'+left_role,b.occurrence_id+'/'+right_role
                    _require(dict(a.physical_terminals)[left_role] == dict(b.physical_terminals)[right_role],
                             'shared raw SD interface has different actual nodes: '+repr(names))
                    if (left_region & right_region).is_empty():
                        aligned = align_fixed_artwork_abutment(left_site=left_shapes,right_site=right_shapes,
                            left_channel=a.primitive.body_nm,right_channel=b.primitive.body_nm,checkpoint=checkpoint)
                        interface_type = MosMosaicSharedInterface
                    else:
                        from clay.layout.fixed_artwork_shared_access import align_fixed_artwork_shared_access
                        from clay.layout.mos_mosaic_materializer import MosMosaicSharedAccessInterface
                        aligned = align_fixed_artwork_shared_access(left_site=left_shapes,right_site=right_shapes,
                            left_channel=a.primitive.body_nm,right_channel=b.primitive.body_nm,
                            left_artwork=artwork(a),right_artwork=artwork(b),
                            contact_layer=keys['contact'],metal_layer=keys['metal1'],checkpoint=checkpoint)
                        interface_type = MosMosaicSharedAccessInterface
                    _require(aligned.translation_nm == (0,0),
                             'observed raw SD interface differs from its intrinsic complete artwork alignment: '+repr(names))
                    result.append(interface_type(identities[a.occurrence_id]+'/'+left_role,
                                                 identities[b.occurrence_id]+'/'+right_role))
    return tuple(sorted(result,key=lambda row:(row.left_site_id,row.right_site_id)))


def _overlay(records, units, identities, source_sha, observation_sha, gds_sha):
    from clay.layout.spatial.semantic import (SemanticOccurrence, SemanticOverlay, SemanticParameterKey,
        SemanticPhysicalBinding, SemanticTerminal, SemanticBodyWellBinding, semantic_nets_from_terminals)
    lookup = {unit.occurrence_id:unit for unit in units}
    occurrences,terminals,bodies = [],[],[]
    evidence = ('current-reference:'+observation_sha,)
    for record in records:
        node = identities[record.occurrence_id]
        ordinal = int(node.rsplit('[',1)[1][:-1])
        pins = dict(record.source_terminals)
        master = lookup[node].master
        roles = (('bulk','b'),('drain','d'),('gate','g'),('source','s'))
        _require(all(value % 2 == 0 for value in record.center2_nm), 'half-nm channel anchor is unsupported by source overlay')
        occurrences.append(SemanticOccurrence(node,record.logical_id,ordinal,record.family,
            SemanticParameterKey(record.parameter_class,record.primitive.width_nm,record.primitive.length_nm,1),
            master.variant_sha256,master.artwork_sha256,SemanticPhysicalBinding(
                tuple(value//2 for value in record.center2_nm),record.owned_bbox_nm,record.orientation),
            tuple(node+':'+role for role,_pin in roles),node+':body-well','current_fixed_source_terminal_partition',evidence))
        terminals.extend(SemanticTerminal(node+':'+role,node,role,pins[pin],
            'declared_not_traced' if pin == 'b' else 'current_fixed_source_terminal_partition',evidence) for role,pin in roles)
        bodies.append(SemanticBodyWellBinding(node+':body-well',node,pins['b'],
            'source_declared_body','declared_not_traced',evidence))
    terminals = tuple(terminals)
    return SemanticOverlay('reference_gds',gds_sha,source_sha,tuple(occurrences),terminals,
                           semantic_nets_from_terminals(terminals),tuple(bodies))


class CurrentReferenceMosInputs:
    def __init__(self,*args,**kwargs):
        raise TypeError('current MOS inputs require the active current reference producer')

    def _values(self):
        return freeze_row_values((self.overlay,self.units,self.accessories,self.interfaces,
            self.source_members,self.connect_model,self.provenance_sha256,self.record_node_ids,self._asir,vars(self.source_cell)))

    def require_current(self):
        self._reference.require_current()
        _require(self._source is self.source_cell and self._tech is self.tech,
                 'current MOS source/PDK object identity changed')
        _require(self._values() == self._snapshot, 'current MOS complete input values changed')

    def for_mosaic(self,mosaic,*,mosaic_scope_by_logical=()):
        from clay.layout.spatial import derive_abstract_spatial_ir, project_discrete_mosaic
        self.require_current()
        _require(type(mosaic_scope_by_logical) is tuple, 'mosaic scope requires a canonical complete tuple')
        ir = (derive_abstract_spatial_ir(self.overlay,mosaic_scope_by_logical=mosaic_scope_by_logical)
              if mosaic_scope_by_logical else self._asir)
        expected = project_discrete_mosaic(ir,mosaic_id=mosaic.mosaic_id,logical_owners=mosaic.members)
        _require(expected.canonical_data() == mosaic.canonical_data(), 'mosaic differs from complete current reference observation')
        selected = {node.node_id for node in mosaic.occurrences}
        units = tuple(unit for unit in self.units if unit.occurrence_id in selected)
        _require(len(units) == len(selected), 'mosaic current raw unit denominator differs')
        accessories = []
        for item in self.accessories:
            if selected.intersection(item.occurrence_ids):
                _require(set(item.occurrence_ids) <= selected, 'mosaic splits a physical shared gate accessory')
                accessories.append(item)
        interfaces = []
        for item in self.interfaces:
            pair = {item.left_site_id.rsplit('/',1)[0],item.right_site_id.rsplit('/',1)[0]}
            if selected.intersection(pair):
                _require(pair <= selected, 'mosaic splits a physical shared SD interface')
                interfaces.append(item)
        return dict(units=units,accessories=tuple(accessories),interfaces=tuple(interfaces))


def capture_current_reference_mos_inputs(source_cell, *, reference, tech):
    from clay.layout.place.row_program import _normalize_records
    from clay.layout.spatial import derive_abstract_spatial_ir
    _require(type(reference) is CurrentReferenceDeviceArtwork, 'current MOS inputs require the actual current reference producer')
    reference.require_current()
    data = reference.canonical_data()
    current = reopen_complete_source(source_cell)
    _require(current['source_sha256'] == data['source_sha256'] and current['pdk_id'] == tech.pdk_id
             and reference._tech is tech, 'current MOS complete source/PDK differs from reference producer')
    records = tuple(row for row in reference.records if row.family in ('nmos','pmos'))
    members = {row.device:row for row in _normalize_records(source_cell,None)}
    _require(Counter(row.logical_id for row in records) == {name:row.fingers for name,row in members.items()},
             'current complete source MOS population differs')
    identities = {}
    for name in sorted(members):
        for index,record in enumerate(sorted((row for row in records if row.logical_id == name),key=lambda row:row.occurrence_id)):
            identities[record.occurrence_id] = f'{name}[{index:03d}]'
    provenance = _digest(data)
    gds = tuple(path for path in data['input_sha256'] if path.lower().endswith('.gds'))
    _require(len(gds) == 1, 'current MOS needs one explicitly measured reference GDS')
    units = tuple(_raw_unit(row,node_id=identities[row.occurrence_id],pdk_id=tech.pdk_id,
        gds_sha256=data['input_sha256'][gds[0]],observation_sha256=provenance,net_to_node=dict(reference.net_to_node))
        for row in records)
    result = object.__new__(CurrentReferenceMosInputs)
    result._reference,result._source,result._tech = reference,source_cell,tech
    result.source_cell,result.tech = source_cell,tech
    result.units,result.source_members = units,members
    result.accessories = _accessories(records,identities,reference.gate_accessories,provenance)
    result.interfaces = _interfaces(records,identities,data['semantic_layers'],reference._session.checkpoint)
    result.overlay = _overlay(records,units,identities,current['electrical_sha256'],provenance,data['input_sha256'][gds[0]])
    result.connect_model,result.provenance_sha256 = reference.connect_model,provenance
    result.record_node_ids = tuple(sorted(identities.items()))
    result._asir = derive_abstract_spatial_ir(result.overlay)
    result._snapshot = result._values()
    result.require_current()
    return result
