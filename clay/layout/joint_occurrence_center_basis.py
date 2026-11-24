"""Optional raw-channel center evidence for a complete native joint row.

Existing rows retain raw-master extent centers. New materializers may bind this
explicit geometry basis. It is transported inside the native binding; observers
cannot select a different center convention after materialization.
"""
from dataclasses import dataclass

from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM


def _require(value, message):
    if not value:
        raise ValueError('joint channel center ' + message)


def _region(shapes):
    from clay.layout.joint_row_geometry import _region as region
    return region(shapes)


def _center(shapes):
    region = _region(shapes).merged()
    box = region.bbox()
    _require(region.size() == 1 and box.width() > 0 and box.height() > 0
             and region.area() == box.width()*box.height(), 'geometry must be one complete rectangle')
    return (box.left+box.right)//2, (box.bottom+box.top)//2


@dataclass(frozen=True, slots=True)
class JointRawChannelCenter:
    occurrence_id: str
    channel_geometry_nm: tuple[PhysicalTerminalPolygonNM, ...]
    center_nm: tuple[int, int]

    def __post_init__(self):
        from clay.layout.joint_terminal_inventory import _text, _polygons
        _text(self.occurrence_id, 'channel center occurrence')
        _polygons(self.channel_geometry_nm, 'channel center geometry')
        _require(type(self.center_nm) is tuple and len(self.center_nm) == 2
                 and all(type(value) is int for value in self.center_nm)
                 and self.center_nm == _center(self.channel_geometry_nm),
                 'record differs from its complete channel geometry')


@dataclass(frozen=True, slots=True)
class JointRawChannelCenters:
    family: str
    od_layer: tuple[int, int]
    poly_layer: tuple[int, int]
    records: tuple[JointRawChannelCenter, ...]
    basis: str = 'materialized_raw_channel_bbox_center_floor_nm'

    def __post_init__(self):
        from clay.layout.joint_terminal_inventory import _layer
        _require(self.basis == 'materialized_raw_channel_bbox_center_floor_nm', 'basis is unsupported')
        _require(self.family in ('nmos','pmos'), 'source family is unsupported')
        _layer(self.od_layer)
        _layer(self.poly_layer)
        _require(self.od_layer != self.poly_layer, 'raw PDK layer roles coincide')
        _require(type(self.records) is tuple and 0 < len(self.records) <= 4096
                 and all(type(record) is JointRawChannelCenter for record in self.records),
                 'complete record inventory is malformed')
        for record in self.records:
            record.__post_init__()
        _require(tuple(record.occurrence_id for record in self.records) ==
                 tuple(sorted({record.occurrence_id for record in self.records})),
                 'occurrence denominator is duplicated or not canonical')


def _raw_channel(master, od_layer, poly_layer, gate_site_id=''):
    from clay.layout.joint_row_geometry import _shapes
    layers = dict(master.geometry_by_layer_nm)
    _require(od_layer in layers and poly_layer in layers, 'raw master PDK layers are missing')
    gate = layers[poly_layer]
    if gate_site_id:
        site = next((site for site in master.terminal_sites if site.site_id == gate_site_id), None)
        _require(site is not None and site.role_group == 'gate', 'selected channel gate is missing')
        site_layers = dict(site.geometry_by_layer_nm)
        _require(poly_layer in site_layers, 'selected channel gate lacks PDK poly')
        gate = site_layers[poly_layer]
    return _shapes((_region(layers[od_layer]) & _region(gate)).merged())


def capture_joint_channel_centers(occurrences, *, tech, family):
    """Measure independent normalized raw-master channels at materialization."""
    from clay.layout.device_cell import _resolve_layer_keys
    keys = _resolve_layer_keys(tech, family)
    records = []
    for occurrence in occurrences:
        shapes = _raw_channel(occurrence.master, keys.active, keys.poly, occurrence.channel_site_id)
        records.append(JointRawChannelCenter(occurrence.occurrence_id, shapes, _center(shapes)))
    return JointRawChannelCenters(family, keys.active, keys.poly,
                                  tuple(sorted(records,key=lambda record:record.occurrence_id)))


def replay_joint_channel_centers(binding, *, tech=None, members=None):
    from clay.layout.device_cell import _resolve_layer_keys
    from clay.layout.tech import LayoutTech
    value = binding.quality_centers
    _require(type(value) is JointRawChannelCenters, 'typed native inventory is missing or foreign')
    value.__post_init__()
    tech = LayoutTech.resolve(binding.pdk_id) if tech is None else tech
    _require(tech.pdk_id == binding.pdk_id, 'current PDK differs')
    keys = _resolve_layer_keys(tech, value.family)
    _require((value.od_layer,value.poly_layer) == (keys.active,keys.poly), 'current PDK layer roles differ')
    records = {record.occurrence_id:record for record in value.records}
    _require(set(records) == {occ.occurrence_id for occ in binding.occurrences}, 'complete occurrence denominator differs')
    for occurrence in binding.occurrences:
        if members is not None:
            _require(members[occurrence.owner].kind == value.family, 'declared source family differs')
        shapes = _raw_channel(occurrence.master, value.od_layer, value.poly_layer, occurrence.channel_site_id)
        record = records[occurrence.occurrence_id]
        _require(record.channel_geometry_nm == shapes and record.center_nm == _center(shapes),
                 'record differs from actual raw master channel')
    return value


def joint_channel_occurrence_centers(binding, origin_nm):
    from clay.layout.joint_row_geometry import joint_d4_orientation, _shapes
    from clay.layout.joint_terminal_inventory import _native_frame
    inventory = replay_joint_channel_centers(binding)
    records = {record.occurrence_id:record for record in inventory.records}
    result = {}
    for occurrence in binding.occurrences:
        transform = _native_frame(occurrence.master_to_local)
        # Floor the actual transformed channel bbox. Transforming an already
        # rounded center can shift an odd-nm reflected channel by one nm.
        shapes = _shapes(_region(records[occurrence.occurrence_id].channel_geometry_nm).transformed(transform))
        x,y = _center(shapes)
        result.setdefault(occurrence.owner,[]).append(((x-origin_nm[0])/1000,
            (y-origin_nm[1])/1000,joint_d4_orientation(occurrence.master_to_local)))
    return tuple((owner,tuple(sorted(values))) for owner,values in sorted(result.items()))


def require_joint_stamp_center_basis(stamp):
    """Validate the recorded native choice before a quality observer reads it."""
    from clay.layout.joint_row_stamp import JointRowStampBinding
    binding = getattr(stamp, 'joint_row_stamp_binding', None)
    _require(type(binding) is JointRowStampBinding, 'bound channel center stamp is foreign')
    binding.__post_init__()
    if binding.source.quality_centers is not None:
        expected = joint_channel_occurrence_centers(binding.source, binding.native_origin_nm)
        _require(getattr(stamp,'physical_occurrences_by_member',None) == expected,
                 'bound channel center metadata differs from the native geometry basis')
