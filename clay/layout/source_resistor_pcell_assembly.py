"""Complete source-associated resistor PCells with one emission per raw shape.

Logical resistor bodies may share original marker, implant and metal polygons.
The local construction keeps that multiset intact and records every owner;
member emission views are not independent raw masters or extraction operands.
Pure plans are data. Current materialization additionally requires the live
reference producer, the complete current electrical source and installed PDK.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json

from clay.frontend.reference_device_artwork import ReferenceDeviceArtworkRecord
from clay.frontend.reference_device_shapes import observe_reference_pcell_geometry
from clay.layout.device_cell import DeviceGeom, Terminal
from clay.layout.joint_row_geometry import joint_record_counter
from clay.layout.occurrence_artwork import polygon_records

_SEAL = object()


def _require(condition, message):
    if not condition:
        raise ValueError('source resistor ' + message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _checkpoint():
    from clay.flow.execution_session import current_session
    session = current_session()
    if session is not None:
        session.checkpoint()


def _bbox(shapes):
    points = tuple(point for shape in shapes for point in shape.hull_nm)
    _require(bool(points), 'artwork is empty')
    return min(x for x, y in points), min(y for x, y in points), max(x for x, y in points), max(y for x, y in points)


@dataclass(frozen=True, slots=True)
class ResistorPCellPart:
    part_id: str
    layer: tuple
    shape: object
    owners: tuple[str, ...]
    emitter: str


@dataclass(frozen=True, slots=True)
class ResistorPCellOccurrence:
    owner: str
    occurrence_id: str
    segment_index: int
    source_member: tuple
    primitive: object
    role_to_head: tuple
    terminal_boxes: tuple
    orientation: str

    @property
    def center2_nm(self):
        return self.primitive.center2_nm

    @property
    def owned_bbox_nm(self):
        return self.primitive.owned_bbox_nm

    @property
    def bbox_nm(self):
        return self.owned_bbox_nm


def _polygons(parts):
    from clay.layout.backend import LayerPolygon, Point, Polygon
    result = []
    for part in parts:
        _checkpoint()
        for polygon in part.shape.to_region().each():
            result.append(LayerPolygon(*part.layer, Polygon([Point(point.x, point.y)
                for point in polygon.resolved_holes().each_point_hull()])))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ResistorPCellPlan:
    occurrence_path: str
    master_name: str
    raw_master_geometry: tuple
    raw_to_local: tuple
    semantic_layers: tuple
    parts: tuple[ResistorPCellPart, ...]
    occurrences: tuple[ResistorPCellOccurrence, ...]
    bbox_nm: tuple

    @property
    def members(self):
        return tuple(row.owner for row in self.occurrences)

    def canonical_data(self):
        return dict(schema='clay.source-resistor-pcell-plan/v1', **asdict(self),
            quality_admission='NOT_ESTABLISHED',
            scope='raw_multiset_and_source_parameter_terminal_association_data_only')

    @property
    def sha256(self):
        return _sha(self.canonical_data())

    def device_geometry(self):
        return DeviceGeom(list(_polygons(self.parts)), {}, 'resistor_assembly', len(self.occurrences),
            0, 0, self.bbox_nm, placement_extent=self.bbox_nm,
            orient=self.occurrences[0].orientation)

    def member_geometry(self, owner):
        rows = [row for row in self.occurrences if row.owner == owner]
        _require(len(rows) == 1, 'member owner is absent')
        row, = rows
        pins = dict(row.source_member[4])
        boxes = dict(row.terminal_boxes)
        geometry = DeviceGeom(list(_polygons(tuple(p for p in self.parts if p.emitter == owner))),
            {role: Terminal(role, values[0], pins[role]) for role, values in boxes.items()},
            'res', 1, row.primitive.length_nm, row.primitive.width_nm, row.owned_bbox_nm,
            orient=row.orientation, placement_extent=row.owned_bbox_nm)
        object.__setattr__(geometry, 'terminal_layers', {role: dict(self.semantic_layers)['metal1'] for role in boxes})
        object.__setattr__(geometry, 'terminal_accesses', {role: tuple(Terminal(role, box, pins[role])
            for box in values) for role, values in boxes.items()})
        return geometry


def _source_rows(source_members):
    _require(type(source_members) is tuple, 'source inventory must be immutable')
    _require(all(type(row) is tuple and len(row) == 6 for row in source_members), 'source member shape differs')
    rows = tuple(row for row in source_members if row[1] == 'res')
    _require(rows and len({row[0] for row in rows}) == len(rows), 'complete source resistor inventory differs')
    for row in rows:
        pins, sizing = dict(row[4]), dict(row[5])
        _require(set(pins) == {'p', 'n'} and pins['p'] != pins['n'], 'source requires distinct complete p/n pins')
        _require(Fraction(str(sizing.get('M', 1))) == 1, 'source requires one primitive per logical resistor')
        _require(Fraction(str(sizing.get('value', sizing.get('R', 0)))) > 0, 'source resistor value is missing')
    return {row[0]: row for row in rows}


def compile_resistor_pcell_plans(*, records, source_members, net_to_node,
        semantic_layers, sheet_resistance_ohm_sq):
    """Replay supplied complete raw geometry; this pure function mints no authority."""
    _checkpoint()
    _require(type(records) is tuple and all(type(row) is ReferenceDeviceArtworkRecord for row in records),
             'reference records must be immutable exact producer values')
    source = _source_rows(source_members)
    selected = tuple(row for row in records if row.family == 'resistor')
    _require(len(selected) == len(source) and {r.logical_id for r in selected} == set(source)
             and len({r.occurrence_id for r in selected}) == len(selected), 'complete source/primitive denominator differs')
    _require(type(net_to_node) is tuple and len(dict(net_to_node)) == len(net_to_node)
             and len({node for _, node in net_to_node}) == len(net_to_node), 'net mapping must be bijective')
    mapping = dict(net_to_node)
    rho = Fraction(str(sheet_resistance_ohm_sq))
    _require(rho > 0, 'PDK sheet resistance must be positive')
    groups = {}
    for row in selected:
        groups.setdefault(row.occurrence_path, []).append(row)
    plans = []
    for path, rows in sorted(groups.items()):
        _checkpoint()
        first = rows[0]
        _require(all((r.master_name, r.raw_master_geometry, r.raw_to_global, r.parts) ==
                     (first.master_name, first.raw_master_geometry, first.raw_to_global, first.parts) for r in rows),
                 'one PCell has conflicting original master/frame/parts')
        actual = observe_reference_pcell_geometry(geometry_by_layer_nm=first.raw_master_geometry,
            family='resistor', semantic_layers=semantic_layers, raw_to_global=first.raw_to_global, checkpoint=_checkpoint)
        _require(actual.parts == first.parts, 'complete original PCell part multiset differs')
        _require(sorted(r.segment_index for r in rows) == list(range(len(actual.primitives))),
                 'complete PCell segment denominator differs')
        for row in rows:
            _require(actual.primitives[row.segment_index] == row.primitive, 'recognized primitive geometry differs')
        oriented = observe_reference_pcell_geometry(geometry_by_layer_nm=first.raw_master_geometry,
            family='resistor', semantic_layers=semantic_layers,
            raw_to_global=(*first.raw_to_global[:4], 0, 0), checkpoint=_checkpoint)
        x0, y0, _x1, _y1 = _bbox(tuple(shape for _, shape in oriented.parts))
        frame = (*first.raw_to_global[:4], -x0, -y0)
        local = observe_reference_pcell_geometry(geometry_by_layer_nm=first.raw_master_geometry,
            family='resistor', semantic_layers=semantic_layers, raw_to_global=frame, checkpoint=_checkpoint)
        occurrences = []
        for row in sorted(rows, key=lambda r: r.logical_id):
            member = source[row.logical_id]
            _require(tuple(sorted(row.source_terminals)) == member[4], 'source pin declaration differs')
            value = Fraction(str(dict(member[5]).get('value', dict(member[5]).get('R'))))
            primitive = local.primitives[row.segment_index]
            _require(rho*primitive.length_nm/primitive.width_nm == value, 'source value differs from current PDK body W/L')
            physical = dict(row.physical_terminals)
            _require(set(physical) == {'head0', 'head1'} and len(set(physical.values())) == 2,
                     'physical terminal denominator differs')
            role_heads, boxes = [], []
            for role, net in sorted(row.source_terminals):
                matches = [head for head, node in physical.items() if mapping.get(net) == node]
                _require(len(matches) == 1, 'source terminal net mapping differs from physical head')
                head, = matches
                role_heads.append((role, head))
                from clay.frontend.reference_device_shapes import _region
                probe = _region(next(shapes for name, _, shapes in primitive.terminals if name == head))
                cuts = tuple(local.parts[i][1] for i in primitive.contact_part_indices
                             if (local.parts[i][1].to_region()-probe).is_empty())
                _require(cuts, 'complete terminal head has no actual routing contact')
                boxes.append((role, tuple(_bbox((shape,)) for shape in cuts)))
            occurrences.append(ResistorPCellOccurrence(row.logical_id, row.occurrence_id,
                row.segment_index, member, primitive, tuple(role_heads), tuple(boxes), row.orientation))
        parts = []
        for index, (layer, shape) in enumerate(local.parts):
            owners = tuple(row.owner for row in occurrences if index in row.primitive.owned_part_indices)
            _require(bool(owners), 'original PCell part has no logical owner')
            parts.append(ResistorPCellPart(f'part/{index}', layer, shape, owners, owners[0]))
        plans.append(ResistorPCellPlan(path, first.master_name, first.raw_master_geometry, frame,
            semantic_layers, tuple(parts), tuple(occurrences), _bbox(tuple(shape for _, shape in local.parts))))
    return tuple(sorted(plans, key=lambda plan: plan.members))


@dataclass(frozen=True, slots=True, init=False)
class CurrentResistorPCellAssembly:
    plan: ResistorPCellPlan
    producer: object
    source_members: tuple
    pdk_identity: tuple
    connect_model: object
    native_terminal_nodes: tuple
    _producer_identity: int
    _producer_report_sha256: str
    _captured_sha256: str
    _seal: object

    def __init__(self, *args, **kwargs):
        raise TypeError('source resistor assembly requires current producer materialization')

    def __deepcopy__(self, memo):
        return self

    @property
    def members(self):
        return self.plan.occurrences

    @property
    def bbox_nm(self):
        return self.plan.bbox_nm

    @property
    def pdk_id(self):
        return self.pdk_identity[0]

    def payload(self):
        return dict(schema='clay.current-resistor-pcell-assembly/v1', plan=self.plan.canonical_data(),
            source_members=self.source_members, pdk_identity=self.pdk_identity,
            reference_report_sha256=self._producer_report_sha256,
            connect_model=asdict(self.connect_model), native_terminal_nodes=self.native_terminal_nodes)

    def validate(self):
        _require(self._seal is _SEAL and type(self.plan) is ResistorPCellPlan, 'materialized assembly is foreign')
        _require(id(self.producer) == self._producer_identity
                 and self.producer._report_sha == self._producer_report_sha256,
                 'current materialization producer identity changed')
        from clay.frontend.reference_device_artwork import _owner
        _require(self.producer._active and not self.producer._closed and self.producer._owner == _owner(),
                 'current producer lifetime is closed or foreign')
        _require(_sha(self.payload()) == self._captured_sha256, 'materialized assembly changed')
        return self

    @property
    def sha256(self):
        self.validate()
        return self._captured_sha256


def _terminal_nodes(plan, model, *, polygons=None, return_nodes=False):
    from clay.layout.connectivity import extract_nodes
    from clay.frontend._reference_device_measurement import POLY_WIRE, _hits
    from clay.frontend.reference_device_shapes import _region
    import klayout.db as kdb
    artwork = {}
    if polygons is None:
        polygons = plan.device_geometry().polygons
    for polygon in polygons:
        artwork.setdefault((polygon.layer, polygon.datatype), []).append(kdb.Polygon(
            [kdb.Point(p.x, p.y) for p in polygon.polygon.vertices]))
    nodes = tuple(extract_nodes(artwork, model))
    by_node, result = {}, []
    for occurrence in plan.occurrences:
        pins = dict(occurrence.source_member[4])
        probes = {head: shapes for head, _, shapes in occurrence.primitive.terminals}
        for role, head in occurrence.role_to_head:
            node = _hits(nodes, POLY_WIRE, probes[head], _checkpoint)
            _require(by_node.setdefault(node, pins[role]) == pins[role], 'local physical node shorts different source nets')
            for box in dict(occurrence.terminal_boxes)[role]:
                region = kdb.Region(kdb.Box(*box))
                index = int(node.split(':')[1])
                metal = dict(plan.semantic_layers)['metal1']
                _require(metal in nodes[index] and (region-nodes[index][metal]).is_empty(),
                         'native routing access differs from complete terminal head node')
            result.append((occurrence.owner, role, pins[role], node))
    return (tuple(result), nodes) if return_nodes else tuple(result)


def materialize_current_resistor_pcell_assemblies(producer, *, source_cell, tech):
    from clay.frontend.reference_device_artwork import CurrentReferenceDeviceArtwork
    from clay.layout.source_mim_materialization import _technology
    from clay.layout.route.exact_artifact_evidence import _canonical_source_member_inventory
    _require(type(producer) is CurrentReferenceDeviceArtwork, 'requires exact current reference producer')
    producer.require_current()
    source = _canonical_source_member_inventory(source_cell)
    _require(source == _canonical_source_member_inventory(producer._source), 'current source electrical inventory differs')
    identity = _technology(tech)
    _require(identity == producer._pdk[0], 'current producer PDK differs')
    plans = compile_resistor_pcell_plans(records=producer.records, source_members=source,
        net_to_node=producer.net_to_node, semantic_layers=producer._pdk[1],
        sheet_resistance_ohm_sq=tech.adapter.poly_sheet_rho_ohm_sq)
    result = []
    for plan in plans:
        geometry = plan.device_geometry()
        combined = tuple(p for owner in plan.members for p in plan.member_geometry(owner).polygons)
        _require(joint_record_counter(polygon_records(geometry.polygons)) ==
                 joint_record_counter(polygon_records(combined)), 'native member emission does not conserve raw multiset')
        value = object.__new__(CurrentResistorPCellAssembly)
        for key, item in dict(plan=plan, producer=producer, source_members=source,
            pdk_identity=identity, connect_model=producer.connect_model,
            native_terminal_nodes=_terminal_nodes(plan, producer.connect_model),
            _producer_identity=id(producer), _producer_report_sha256=producer._report_sha, _seal=_SEAL).items():
            object.__setattr__(value, key, item)
        object.__setattr__(value, '_captured_sha256', _sha(value.payload()))
        result.append(value.validate())
    producer.require_current()
    return tuple(result)


def replay_current_resistor_assembly(value, *, source_cell, tech, intent=None):
    from clay.layout.source_mim_materialization import _technology
    from clay.layout.route.exact_artifact_evidence import _canonical_source_member_inventory
    _require(type(value) is CurrentResistorPCellAssembly, 'requires exact current assembly')
    value.validate()
    value.producer.require_current()
    _require(_canonical_source_member_inventory(source_cell) == value.source_members,
             'current source inventory differs from materialization')
    _require(_technology(tech) == value.pdk_identity, 'current PDK differs from materialization')
    _require(_terminal_nodes(value.plan, value.connect_model) == value.native_terminal_nodes,
             'current native terminal graph changed')
    return value
