"""Local terminal-site connectivity with distinct joint and routing graphs.

All records are immutable data. A caller supplies the independent, complete site
ledger and an explicitly bound conductor model; this module cannot authenticate
that ledger, a PDK recipe, or a raw-master variant. It replays their exact local
geometry and connectivity, not body/well, DRC, LVS, routing or source authority.

Required joint components are AND obligations. Every routing-metal component
within one joint component is an OR access alternative, retained in full. The
existing routing-metal-only PhysicalTerminalInventory is not reinterpreted.
Native extraction has bounded inputs and checks before/after each invocation;
the caller's supervisor still owns native wall/RSS interruption and peak storage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable

import klayout.db as kdb

from clay.layout.connectivity import ConnectModel, DerivedLayer, extract_nodes
from clay.layout.occurrence_channels import _shape_keys
from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM

Layer = tuple[int, int]
PolygonsNM = tuple[PhysicalTerminalPolygonNM, ...]
LayeredPolygonsNM = tuple[tuple[Layer, PolygonsNM], ...]
D4NM = tuple[int, int, int, int, int, int]
_SCOPE = 'local-device-connectivity-preflight-only/v1'
_MAX_ITEMS = 262_144
_MAX_VERTICES = 1_048_576
_MAX_COORD = 1_000_000_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('joint terminal ' + message)


def _tuple(value, field, *, nonempty=False, maximum=_MAX_ITEMS):
    _require(type(value) is tuple and len(value) <= maximum and (bool(value) or not nonempty),
             field + ' must be a bounded exact tuple')


def _text(value, field):
    _require(type(value) is str and 0 < len(value) <= 4096 and value == value.strip(),
             field + ' must be a bounded canonical string')


def _sha(value, field):
    _require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
             field + ' must be SHA256')


def _layer(value):
    _require(type(value) is tuple and len(value) == 2 and
             all(type(v) is int and 0 <= v <= 65535 for v in value), 'layer must be an exact pair')


def _polygons(value, field, work=None):
    _tuple(value, field, nonempty=True)
    vertices = 0
    for shape in value:
        _require(type(shape) is PhysicalTerminalPolygonNM, field + ' contains a foreign polygon')
        if work is not None:
            work.polygon(shape)
        vertices += len(shape.hull_nm) + sum(map(len,shape.holes_nm))
        _require(vertices <= _MAX_VERTICES,field + ' exceeds its vertex bound')
        shape.__post_init__()


def _geometry(value, field, work=None):
    _tuple(value, field, nonempty=True, maximum=4096)
    keys = []
    for row in value:
        _require(type(row) is tuple and len(row) == 2, field + ' has a malformed layer row')
        key, shapes = row
        _layer(key)
        _polygons(shapes, field, work)
        keys.append(key)
    _require(len(keys) == len(set(keys)), field + ' contains duplicate layers')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def _frame(value):
    _require(type(value) is tuple and len(value) == 6 and all(type(v) is int for v in value),
             'D4 frame must be an exact six-integer tuple')
    a,b,c,d,x,y = value
    _require(a*a+c*c == b*b+d*d == 1 and a*b+c*d == 0 and abs(a*d-b*c) == 1,
             'frame must be D4')
    _require(abs(x) <= _MAX_COORD and abs(y) <= _MAX_COORD, 'frame coordinate exceeds bound')


def _native_frame(value):
    a,b,c,d,x,y = value
    for angle in (0,90,180,270):
        for mirror in (False,True):
            candidate = kdb.ICplxTrans(1,angle,mirror,x,y)
            ex,ey = candidate*kdb.Point(1,0),candidate*kdb.Point(0,1)
            if (ex.x-x,ey.x-x,ex.y-y,ey.y-y) == (a,b,c,d):
                return candidate
    raise ValueError('joint terminal frame must be D4')


@dataclass(frozen=True, slots=True)
class TerminalSiteRequirement:
    site_id: str
    owner: str
    occurrence_id: str
    source_net: str
    allowed_roles: tuple[str, ...]
    raw_layer: Layer
    query_layer: Layer
    master_geometry_nm: PolygonsNM
    variant_sha256: str
    provenance_sha256: str

    def __post_init__(self):
        for field in ('site_id','owner','occurrence_id','source_net'):
            _text(getattr(self,field),field)
        _tuple(self.allowed_roles,'allowed roles',nonempty=True,maximum=64)
        for role in self.allowed_roles:
            _text(role,'allowed role')
        _require(self.allowed_roles == tuple(sorted(set(self.allowed_roles))), 'allowed roles must be canonical')
        _layer(self.raw_layer)
        _layer(self.query_layer)
        _polygons(self.master_geometry_nm,'master site polygons')
        _require(self.master_geometry_nm == tuple(sorted(set(self.master_geometry_nm))),
                 'master site polygons must be canonical and unique')
        _sha(self.variant_sha256,'variant')
        _sha(self.provenance_sha256,'site provenance')


@dataclass(frozen=True, slots=True)
class OccurrenceTerminalSite:
    site_id: str
    variant_sha256: str
    master_to_local: D4NM
    geometry_nm: PolygonsNM

    def __post_init__(self):
        _text(self.site_id,'site id')
        _sha(self.variant_sha256,'variant')
        _frame(self.master_to_local)
        _polygons(self.geometry_nm,'placed site polygons')
        _require(self.geometry_nm == tuple(sorted(set(self.geometry_nm))),
                 'placed site polygons must be canonical and unique')


@dataclass(frozen=True, slots=True)
class JointTerminalLimits:
    max_sites: int = 4096
    max_polygons: int = _MAX_ITEMS
    max_vertices: int = _MAX_VERTICES
    max_nodes: int = _MAX_ITEMS
    max_pair_evaluations: int = 2_000_000

    def __post_init__(self):
        caps = (4096,_MAX_ITEMS,_MAX_VERTICES,_MAX_ITEMS,2_000_000)
        for field,cap in zip(self.__dataclass_fields__,caps):
            value = getattr(self,field)
            _require(type(value) is int and 0 < value <= cap, field + ' must be a bounded positive exact integer')


class JointTerminalRejected(ValueError):
    def __init__(self, code: str, site_ids: tuple[str, ...] = (), detail: str = ''):
        self.code = code
        self.site_ids = tuple(sorted(set(site_ids)))
        self.detail = detail
        super().__init__(f'{code}: {detail}; sites={self.site_ids!r}')


@dataclass(frozen=True, slots=True)
class JointTerminalComponent:
    component_id: str
    source_net: str
    site_ids: tuple[str, ...]
    geometry_by_layer_nm: LayeredPolygonsNM
    routing_component_ids: tuple[str, ...]

    def __post_init__(self):
        _text(self.component_id,'joint component id')
        _text(self.source_net,'joint component net')
        for value,field in ((self.site_ids,'joint site ids'),(self.routing_component_ids,'routing component ids')):
            _tuple(value,field,nonempty=True)
            for item in value:
                _text(item,field)
            _require(value == tuple(sorted(set(value))),field + ' must be canonical and unique')
        _geometry(self.geometry_by_layer_nm,'joint geometry')


@dataclass(frozen=True, slots=True)
class RoutingAccessComponent:
    component_id: str
    joint_component_id: str
    geometry_by_layer_nm: LayeredPolygonsNM

    def __post_init__(self):
        _text(self.component_id,'routing component id')
        _text(self.joint_component_id,'routing joint component id')
        _geometry(self.geometry_by_layer_nm,'routing access geometry')


@dataclass(frozen=True, slots=True)
class JointTerminalInventory:
    expected_sites: tuple[TerminalSiteRequirement, ...]
    placed_sites: tuple[OccurrenceTerminalSite, ...]
    site_components: tuple[tuple[str, tuple[str, ...]], ...]
    joint_components: tuple[JointTerminalComponent, ...]
    routing_components: tuple[RoutingAccessComponent, ...]
    input_sha256: str
    artwork_sha256: str
    connect_model_sha256: str
    routing_model_sha256: str
    pdk_provenance_sha256: str
    scope: str = _SCOPE

    def __post_init__(self):
        _require(type(self.scope) is str and self.scope == _SCOPE,'inventory scope differs')
        for field in ('input_sha256','artwork_sha256','connect_model_sha256','routing_model_sha256','pdk_provenance_sha256'):
            _sha(getattr(self,field),field)
        for value,kind,field in ((self.expected_sites,TerminalSiteRequirement,'expected sites'),
                (self.placed_sites,OccurrenceTerminalSite,'placed sites'),
                (self.joint_components,JointTerminalComponent,'joint components'),
                (self.routing_components,RoutingAccessComponent,'routing components')):
            _tuple(value,field,nonempty=True)
            for row in value:
                _require(type(row) is kind,field + ' contains a foreign record')
                row.__post_init__()
        sites = {row.site_id:row for row in self.expected_sites}
        _require(tuple(sites) == tuple(sorted(sites)) and len(sites) == len(self.expected_sites),
                 'expected site ids must be canonical and unique')
        _require(tuple(row.site_id for row in self.placed_sites) == tuple(sites),'placed site denominator differs')
        joint = {row.component_id:row for row in self.joint_components}
        routing = {row.component_id:row for row in self.routing_components}
        for keys,value in ((tuple(joint),self.joint_components),(tuple(routing),self.routing_components)):
            _require(keys == tuple(sorted(keys)) and len(keys) == len(value),'component ids must be canonical and unique')
        _tuple(self.site_components,'site incidences',nonempty=True)
        incidence = {}
        for row in self.site_components:
            _require(type(row) is tuple and len(row) == 2,'site incidence must be an exact pair')
            site_id,component_ids = row
            _text(site_id,'incidence site id')
            _tuple(component_ids,'site component ids',nonempty=True)
            _require(all(type(value) is str for value in component_ids),'site component id is foreign')
            _require(component_ids == tuple(sorted(set(component_ids))) and set(component_ids) <= set(joint),
                     'site incidence components differ')
            incidence[site_id] = component_ids
        _require(tuple(incidence) == tuple(sites) and len(incidence) == len(self.site_components),
                 'site incidence denominator differs')
        for component in self.joint_components:
            _require(component.site_ids == tuple(site_id for site_id,ids in self.site_components
                                                 if component.component_id in ids),'joint site incidence differs')
            _require({sites[site_id].source_net for site_id in component.site_ids} == {component.source_net},
                     'joint component source nets differ')
            _require(component.routing_component_ids == tuple(row.component_id for row in self.routing_components
                if row.joint_component_id == component.component_id),'routing projection partition differs')
        _require(all(row.joint_component_id in joint for row in self.routing_components),
                 'routing component names an absent joint component')

    @property
    def sha256(self):
        # A content digest is not a certificate; consumers must call replay.
        self.__post_init__()
        return _digest(asdict(self))


def _model(model):
    _require(type(model) is ConnectModel,'model must be an exact ConnectModel')
    _tuple(model.metals,'model metals',nonempty=True,maximum=4096)
    for key in model.metals:
        _layer(key)
    _require(len(set(model.metals)) == len(model.metals),'model metals contain duplicates')
    for field in ('vias','extra_connect','bridges','derived'):
        _tuple(getattr(model,field),'model '+field,maximum=4096)
    for triple in model.vias + model.extra_connect:
        _require(type(triple) is tuple and len(triple) == 3,'model connection must be an exact triple')
        for key in triple:
            _layer(key)
        _require(triple[1] in model.metals and triple[2] in model.metals,
                 'model connection references an undeclared conductor')
    for pair in model.bridges:
        _require(type(pair) is tuple and len(pair) == 2,'model bridge must be an exact pair')
        for key in pair:
            _layer(key)
            _require(key in model.metals,'model bridge references an undeclared conductor')
    names,operations = set(),0
    for derived in model.derived:
        _require(type(derived) is DerivedLayer,'model derived layer is foreign')
        _layer(derived.name)
        _require(derived.name not in names,'model derived layers contain duplicates')
        names.add(derived.name)
        _tuple(derived.ops,'derived operations',nonempty=True,maximum=4096)
        operations += len(derived.ops)
        _require(operations <= 4096,'model derived operation bound exceeded')
        for index,op in enumerate(derived.ops):
            _require(type(op) is tuple and len(op)==3 and op[0] in ('and','or','not','interacting'),
                     'derived operation is malformed')
            for operand in op[1:]:
                if type(operand) is str:
                    _require(operand.startswith('$') and operand[1:].isdigit()
                             and int(operand[1:]) < index,'derived back-reference is invalid')
                else:
                    _layer(operand)


class _Work:
    def __init__(self, checkpoint, limits):
        _require(callable(checkpoint),'checkpoint must be callable')
        _require(type(limits) is JointTerminalLimits,'limits must have the exact typed value')
        limits.__post_init__()
        self.checkpoint,self.limits = checkpoint,limits
        self.polygons = self.vertices = self.pairs = 0

    def check(self):
        self.checkpoint()

    def polygon(self, shape):
        self.check()
        self.polygons += 1
        self.vertices += len(shape.hull_nm) + sum(map(len,shape.holes_nm))
        if self.polygons > self.limits.max_polygons or self.vertices > self.limits.max_vertices:
            raise JointTerminalRejected('COMPLEXITY_LIMIT_EXCEEDED',detail='polygon/vertex work bound')

    def pair(self):
        self.check()
        self.pairs += 1
        if self.pairs > self.limits.max_pair_evaluations:
            raise JointTerminalRejected('COMPLEXITY_LIMIT_EXCEEDED',detail='component incidence work bound')

    def nodes(self, nodes):
        self.check()
        if len(nodes) > self.limits.max_nodes:
            raise JointTerminalRejected('COMPLEXITY_LIMIT_EXCEEDED',detail='native node denominator')


def _prepare(*, artwork, expected_sites, placed_sites, connect_model, routing_model,
             pdk_provenance_sha256, checkpoint, limits):
    work = _Work(checkpoint,limits)
    work.check()
    _geometry(artwork,'artwork',work)
    _tuple(expected_sites,'expected sites',nonempty=True,maximum=4096)
    _tuple(placed_sites,'placed sites',nonempty=True,maximum=4096)
    if max(len(expected_sites),len(placed_sites)) > limits.max_sites:
        raise JointTerminalRejected('COMPLEXITY_LIMIT_EXCEEDED',detail='site denominator')
    for site in expected_sites:
        work.check()
        _require(type(site) is TerminalSiteRequirement,'expected site has a foreign type')
        _polygons(site.master_geometry_nm,'expected site polygons',work)
        site.__post_init__()
    for site in placed_sites:
        work.check()
        _require(type(site) is OccurrenceTerminalSite,'placed site has a foreign type')
        _polygons(site.geometry_nm,'placed site polygons',work)
        site.__post_init__()
    expected_ids,placed_ids = tuple(s.site_id for s in expected_sites),tuple(s.site_id for s in placed_sites)
    if (len(set(expected_ids)) != len(expected_ids) or len(set(placed_ids)) != len(placed_ids)
            or set(expected_ids) != set(placed_ids)):
        raise JointTerminalRejected('SITE_DENOMINATOR_MISMATCH',tuple(sorted(set(expected_ids)^set(placed_ids))))
    semantic = [(s.owner,s.occurrence_id,s.raw_layer,s.query_layer,s.master_geometry_nm) for s in expected_sites]
    _require(len(set(semantic)) == len(semantic),'expected ledger duplicates a physical site')
    _sha(pdk_provenance_sha256,'PDK provenance')
    _model(connect_model)
    _model(routing_model)
    if routing_model.extra_connect or routing_model.derived or routing_model.bridges:
        raise JointTerminalRejected('ROUTING_MODEL_SCOPE_INVALID')
    if not set(routing_model.metals) <= set(connect_model.metals):
        raise JointTerminalRejected('ROUTING_MODEL_SCOPE_INVALID',detail='routing conductors absent in joint model')
    if set(layer for layer,_ in artwork) & {item.name for item in connect_model.derived}:
        raise JointTerminalRejected('DERIVED_RAW_LAYER_COLLISION')
    canonical_art = tuple(sorted((key,tuple(sorted(shapes))) for key,shapes in artwork))
    expected = tuple(sorted(expected_sites,key=lambda row:row.site_id))
    placed = tuple(sorted(placed_sites,key=lambda row:row.site_id))
    art_sha = _digest(tuple((key,tuple(asdict(shape) for shape in shapes)) for key,shapes in canonical_art))
    model_sha,routing_sha = _digest(asdict(connect_model)),_digest(asdict(routing_model))
    input_sha = _digest({'artwork':art_sha,'expected':tuple(asdict(row) for row in expected),
        'placed':tuple(asdict(row) for row in placed),'model':model_sha,'routing_model':routing_sha,
        'pdk':pdk_provenance_sha256,'limits':asdict(limits),'scope':_SCOPE})
    return work,canonical_art,expected,placed,(input_sha,art_sha,model_sha,routing_sha,pdk_provenance_sha256)


def _region(shapes, work):
    result = kdb.Region()
    for shape in shapes:
        work.polygon(shape)
        result += shape.to_region()
    work.check()
    return result.merged()


def _node_geometry(node, work):
    result = []
    for key,region in sorted(node.items()):
        work.check()
        shapes = []
        for hull,holes in _shape_keys(region):
            shape = PhysicalTerminalPolygonNM(hull,holes)
            work.polygon(shape)
            shapes.append(shape)
        if shapes:
            result.append((key,tuple(shapes)))
    return tuple(result)


def _compile_prepared(prepared, connect_model, routing_model):
    work,artwork,expected,placed,hashes = prepared
    raw = {key:_region(shapes,work) for key,shapes in artwork}
    site_regions = {}
    for req,site in zip(expected,placed):
        work.check()
        if req.variant_sha256 != site.variant_sha256:
            raise JointTerminalRejected('SITE_VARIANT_MISMATCH',(req.site_id,))
        master = _region(req.master_geometry_nm,work)
        actual = _region(site.geometry_nm,work)
        if not (master.transformed(_native_frame(site.master_to_local)) ^ actual).is_empty():
            raise JointTerminalRejected('SITE_TRANSFORM_MISMATCH',(req.site_id,))
        missing = actual - raw.get(req.raw_layer,kdb.Region())
        if not missing.is_empty():
            raise JointTerminalRejected('SITE_RAW_COVERAGE_MISSING',(req.site_id,),f'area_nm2={missing.area()}')
        site_regions[req.site_id] = actual
    shapes = {key:list(region.each()) for key,region in raw.items()}
    work.check()
    joint_nodes = tuple(extract_nodes(shapes,connect_model))
    work.nodes(joint_nodes)
    routing_nodes = tuple(extract_nodes(shapes,routing_model))
    work.nodes(routing_nodes)
    by_joint,site_indices = {},{}
    for req in expected:
        covered,hits = kdb.Region(),[]
        for index,node in enumerate(joint_nodes):
            work.pair()
            conductor = node.get(req.query_layer)
            if conductor is not None and not (conductor & site_regions[req.site_id]).is_empty():
                hits.append(index)
                covered += conductor
                by_joint.setdefault(index,[]).append(req)
        missing = site_regions[req.site_id] - covered
        if not missing.is_empty():
            raise JointTerminalRejected('SITE_QUERY_COVERAGE_MISSING',(req.site_id,),f'area_nm2={missing.area()}')
        site_indices[req.site_id] = hits
    component_ids,joint_geometry = {},{}
    for index,reqs in by_joint.items():
        work.check()
        site_ids = tuple(sorted(req.site_id for req in reqs))
        nets = {req.source_net for req in reqs}
        if len(nets) != 1:
            raise JointTerminalRejected('JOINT_COMPONENT_MIXED_NETS',site_ids)
        geometry = _node_geometry(joint_nodes[index],work)
        joint_geometry[index] = geometry
        component_ids[index] = 'joint-terminal:' + _digest((site_ids,next(iter(nets)),
            tuple((key,tuple(asdict(s) for s in shapes)) for key,shapes in geometry),hashes[2]))
    routing_components,access_ids = [],{}
    for routing_node in routing_nodes:
        memberships = []
        for index in by_joint:
            work.pair()
            joint_node = joint_nodes[index]
            if any(key in joint_node and not (reg & joint_node[key]).is_empty() for key,reg in routing_node.items()):
                memberships.append(index)
        if not memberships:
            continue  # Unseeded fill/dummy metal creates no invented terminal.
        if len(memberships) != 1:
            raise JointTerminalRejected('ROUTING_COMPONENT_CROSSES_JOINT_COMPONENTS',
                tuple(req.site_id for index in memberships for req in by_joint[index]))
        index, = memberships
        if any(not (reg-joint_nodes[index].get(key,kdb.Region())).is_empty() for key,reg in routing_node.items()):
            raise JointTerminalRejected('ROUTING_COMPONENT_OUTSIDE_JOINT_COMPONENT',
                tuple(req.site_id for req in by_joint[index]))
        geometry = _node_geometry(routing_node,work)
        component_id = 'joint-routing-access:' + _digest((component_ids[index],
            tuple((key,tuple(asdict(s) for s in shapes)) for key,shapes in geometry),hashes[3]))
        routing_components.append(RoutingAccessComponent(component_id,component_ids[index],geometry))
        access_ids.setdefault(index,[]).append(component_id)
    joint_components = []
    for index,reqs in by_joint.items():
        work.check()
        site_ids = tuple(sorted(req.site_id for req in reqs))
        if not access_ids.get(index):
            raise JointTerminalRejected('JOINT_COMPONENT_WITHOUT_ROUTING_ACCESS',site_ids)
        joint_components.append(JointTerminalComponent(component_ids[index],reqs[0].source_net,site_ids,
            joint_geometry[index],tuple(sorted(access_ids[index]))))
    work.check()
    return JointTerminalInventory(expected,placed,
        tuple((req.site_id,tuple(sorted(component_ids[index] for index in site_indices[req.site_id]))) for req in expected),
        tuple(sorted(joint_components,key=lambda row:row.component_id)),
        tuple(sorted(routing_components,key=lambda row:row.component_id)),*hashes)


def compile_joint_terminal_inventory(*, artwork: LayeredPolygonsNM,
        expected_sites: tuple[TerminalSiteRequirement, ...], placed_sites: tuple[OccurrenceTerminalSite, ...],
        connect_model: ConnectModel, routing_model: ConnectModel, pdk_provenance_sha256: str,
        checkpoint: Callable[[],None], limits: JointTerminalLimits) -> JointTerminalInventory:
    """Replay the complete local site ledger; success remains preflight data."""
    operands = dict(artwork=artwork,expected_sites=expected_sites,placed_sites=placed_sites,
        connect_model=connect_model,routing_model=routing_model,pdk_provenance_sha256=pdk_provenance_sha256,
        checkpoint=checkpoint,limits=limits)
    return _compile_prepared(_prepare(**operands),connect_model,routing_model)


def replay_joint_terminal_inventory(*, inventory: JointTerminalInventory, **operands) -> JointTerminalInventory:
    """Recompute actual geometry and both graphs; no stored boolean is trusted."""
    _require(type(inventory) is JointTerminalInventory,'inventory must have the exact typed value')
    inventory.__post_init__()
    prepared = _prepare(**operands)
    if inventory.input_sha256 != prepared[-1][0]:
        raise JointTerminalRejected('REPLAY_INPUT_MISMATCH')
    actual = _compile_prepared(prepared,operands['connect_model'],operands['routing_model'])
    if inventory != actual:
        raise JointTerminalRejected('REPLAY_INVENTORY_MISMATCH')
    return actual
