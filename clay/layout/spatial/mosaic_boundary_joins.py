"""Coordinate-free raw SD interface requests bound to a complete mosaic.

These are construction inputs. Source declarations constrain role, family and
bulk compatibility; they do not establish a body trace or physical admission.
Only enumerated interfaces request abutment. All other edges remain isolated.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

from .intent_projection import replay_discrete_mosaic_intent


def _require(value, message):
    if not value:
        raise ValueError('mosaic boundary ' + message)


def _text(value):
    _require(type(value) is str and 0 < len(value) <= 4096 and value == value.strip(),
             'interface identity must be a bounded canonical string')


def _sha(value):
    _require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
             'interface provenance must be SHA256')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


@dataclass(frozen=True, slots=True)
class MosaicInterfaceEndpoint:
    node_id: str
    owner: str
    site_id: str
    variant_sha256: str
    roles: tuple[str, ...]

    def __post_init__(self):
        for value in (self.node_id, self.owner, self.site_id):
            _text(value)
        _sha(self.variant_sha256)
        _require(type(self.roles) is tuple and bool(self.roles)
                 and all(type(r) is str and r in ('d','s') for r in self.roles)
                 and tuple(sorted(set(self.roles))) == self.roles,
                 'interface roles must be the complete canonical SD aliases')


@dataclass(frozen=True, slots=True)
class MosaicAbutmentInterface:
    left: MosaicInterfaceEndpoint
    right: MosaicInterfaceEndpoint
    source_net: str
    family: str
    oxide: str
    body_net: str
    body_status: str
    provenance_sha256: str

    def __post_init__(self):
        for endpoint in (self.left, self.right):
            _require(type(endpoint) is MosaicInterfaceEndpoint, 'interface endpoint type differs')
            endpoint.__post_init__()
        _require(self.left.node_id != self.right.node_id, 'interface must join distinct occurrences')
        for value in (self.source_net, self.body_net):
            _text(value)
        _require(type(self.family) is str and self.family in ('nmos','pmos'), 'interface family differs')
        _require(type(self.oxide) is str and self.oxide in ('thin','thick'), 'interface oxide differs')
        _require(type(self.body_status) is str and self.body_status == 'declared_not_traced',
                 'interface body scope must remain declared_not_traced')
        _sha(self.provenance_sha256)

    def canonical_data(self):
        self.__post_init__()
        return asdict(self)

    @property
    def sha256(self):
        return _digest(self.canonical_data())


@dataclass(frozen=True, slots=True)
class MosaicSharedAccessInterface(MosaicAbutmentInterface):
    """A distinct raw SD overlap mode, derived from complete shared contacts."""
    def canonical_data(self):
        self.__post_init__()
        return {'interface_kind': 'complete_shared_contact_access', **asdict(self)}


def mosaic_interface_mode(interface):
    _require(type(interface) in (MosaicAbutmentInterface, MosaicSharedAccessInterface), 'interface type differs')
    return ('fixed_artwork_shared_access' if type(interface) is MosaicSharedAccessInterface
            else 'fixed_artwork_abutment')


@dataclass(frozen=True, slots=True)
class MosaicBoundaryJoinIntent:
    source_mosaic_sha256: str
    interfaces: tuple[MosaicAbutmentInterface | MosaicSharedAccessInterface, ...]

    def __post_init__(self):
        _sha(self.source_mosaic_sha256)
        _require(type(self.interfaces) is tuple and 0 < len(self.interfaces) <= 4096,
                 'interfaces must be a nonempty bounded exact tuple')
        pairs = []
        for item in self.interfaces:
            mosaic_interface_mode(item)
            item.__post_init__()
            pairs.append((item.left.node_id, item.right.node_id))
        _require(len(pairs) == len(set(pairs)), 'interface adjacency is duplicated')

    def canonical_data(self):
        self.__post_init__()
        version = 'v2' if any(type(item) is MosaicSharedAccessInterface for item in self.interfaces) else 'v1'
        return {'schema': 'clay.mosaic-boundary-joins/'+version,
                'source_mosaic_sha256': self.source_mosaic_sha256,
                'interfaces': [item.canonical_data() for item in self.interfaces],
                'scope': 'local_construction_request', 'undeclared_edges': 'pdk_isolated'}

    @property
    def sha256(self):
        return _digest(self.canonical_data())


def replay_mosaic_boundary_join_intent(value, *, mosaic=None, members=None):
    _require(type(value) is MosaicBoundaryJoinIntent, 'interface intent type differs')
    value.__post_init__()
    if mosaic is not None:
        mosaic = replay_discrete_mosaic_intent(mosaic)
        _require(value.source_mosaic_sha256 == mosaic.sha256, 'source mosaic differs')
        nodes = {node.node_id: node for node in mosaic.occurrences}
        pairs = tuple((a.node_id,b.node_id) for row in mosaic.rows for a,b in zip(row,row[1:]))
        requested = tuple((item.left.node_id,item.right.node_id) for item in value.interfaces)
        _require(tuple(pair for pair in pairs if pair in requested) == requested,
                 'interface order or oriented adjacency differs')
        for interface in value.interfaces:
            for endpoint in (interface.left, interface.right):
                _require(endpoint.node_id in nodes and nodes[endpoint.node_id].logical_owner == endpoint.owner,
                         'interface occurrence owner differs')
    if members is not None:
        for interface in value.interfaces:
            for endpoint in (interface.left, interface.right):
                member = members.get(endpoint.owner)
                _require(member is not None, 'interface source owner is absent')
                _require((member.kind, member.oxide, member.body_net) ==
                         (interface.family, interface.oxide, interface.body_net),
                         'interface source family, oxide or body declaration differs')
                aliases = tuple(role for role, net in (('d',member.drain_net),('s',member.source_net))
                                if net == interface.source_net)
                _require(aliases == endpoint.roles, 'interface source net/role aliases differ')
    return value


def replay_mosaic_boundary_join_inventory(values, *, mosaics=None, members=None):
    _require(type(values) is tuple and len(values) <= 4096, 'interface inventory must be bounded exact tuple')
    by_mosaic = None if mosaics is None else {m.sha256: m for m in mosaics}
    result = []
    for value in values:
        replay_mosaic_boundary_join_intent(value)
        if by_mosaic is not None:
            _require(value.source_mosaic_sha256 in by_mosaic, 'source mosaic is absent')
        result.append(replay_mosaic_boundary_join_intent(value,
            mosaic=None if by_mosaic is None else by_mosaic[value.source_mosaic_sha256], members=members))
    _require(len({v.source_mosaic_sha256 for v in result}) == len(result), 'source mosaic boundary is duplicated')
    return tuple(sorted(result, key=lambda v: v.source_mosaic_sha256))


def compile_mosaic_row_boundaries(intent, *, row, atoms, gap_nm):
    from clay.layout.place.row_boundary_joins import RowBoundaryJoin
    replay_mosaic_boundary_join_intent(intent)
    by_pair = {(item.left.node_id,item.right.node_id): item for item in intent.interfaces}
    ref = 'mosaic_boundary_joins:' + intent.sha256
    result = []
    for a,b,aa,bb in zip(row,row[1:],atoms,atoms[1:]):
        interface = by_pair.get((a.node_id,b.node_id))
        result.append(RowBoundaryJoin(aa.atom_id,bb.atom_id,
            'isolated' if interface is None else mosaic_interface_mode(interface),
            gap_nm if interface is None else 0, '' if interface is None else interface.sha256, ref))
    return tuple(result)


def merge_mosaic_boundary_join_inventories(first, second):
    result = {}
    for value in (*replay_mosaic_boundary_join_inventory(first), *replay_mosaic_boundary_join_inventory(second)):
        prior = result.get(value.source_mosaic_sha256)
        _require(prior is None or prior == value, 'conflicting source mosaic boundary declarations')
        result[value.source_mosaic_sha256] = value
    return replay_mosaic_boundary_join_inventory(tuple(result.values()))
