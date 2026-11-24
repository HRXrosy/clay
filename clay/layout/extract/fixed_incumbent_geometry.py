"""Replay fixed master geometry without materializing Cartesian atom pairs.

This is an immutable inventory of obligations, not a cached legal-placement
verdict or a source/PDK admission. Each replay checks the supplied coordinates
afresh. The optimizing Full2DCompactionGraph and its pair choices are unchanged.
"""
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

from clay.layout.spatial.collision_queries import RectangleIndex

Box = tuple[int, int, int, int]
Layer = tuple[int, int] | None
Equality = tuple[str, str, int]
AxisAuthority = tuple[str, str, str, str, str, tuple[str, ...]]


class _BoxTypeError(TypeError):
    pass


class _BoxValueError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ValueError('fixed incumbent geometry ' + message)


def _poller(checkpoint):
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError('fixed incumbent geometry checkpoint must be callable')
    def poll():
        # Callbacks and sessions belong to this operation, never to the proof.
        from clay.flow.execution_session import current_session
        session = current_session()
        if session is not None:
            session.checkpoint()
        if checkpoint is not None:
            checkpoint()
    return poll


def _boxes(values, poll):
    if not isinstance(values, Mapping) or not values:
        raise _BoxTypeError('fixed incumbent geometry boxes require a non-empty mapping')
    result = {}
    for index, (owner, box) in enumerate(values.items()):
        if index % 256 == 0:
            poll()
        if type(owner) is not str or not owner:
            raise _BoxTypeError('fixed incumbent geometry owner must be an exact nonempty string')
        if not (type(box) is tuple and len(box) == 4
                and all(type(value) is int for value in box)
                and box[0] < box[2] and box[1] < box[3]):
            raise _BoxValueError('fixed incumbent geometry bbox is malformed')
        result[owner] = box
    poll()
    return dict(sorted(result.items()))


def _footprints(rows, boxes, grid, layered, poll):
    if type(rows) is not tuple:
        raise TypeError('fixed incumbent geometry footprints require an exact tuple')
    result, count = {}, 0
    for row in rows:
        poll()
        _require(type(row) is tuple and len(row) == 2 and type(row[0]) is str
                 and row[0] in boxes and row[0] not in result
                 and type(row[1]) is tuple and bool(row[1]), 'footprint ownership is malformed')
        owner, atoms = row
        width, height = boxes[owner][2] - boxes[owner][0], boxes[owner][3] - boxes[owner][1]
        previous = None
        for atom in atoms:
            count += 1
            if count % 256 == 0:
                poll()
            _require(type(atom) is tuple and len(atom) == (6 if layered else 4)
                     and all(type(value) is int for value in atom), 'footprint atom is malformed')
            rectangle = atom[2:] if layered else atom
            _require(not layered or (atom[0] >= 0 and atom[1] >= 0), 'layer is malformed')
            _require(0 <= rectangle[0] < rectangle[2] <= width
                     and 0 <= rectangle[1] < rectangle[3] <= height
                     and rectangle[0] % grid == 0 and rectangle[1] % grid == 0,
                     'footprint atom is outside its master or off-grid')
            _require(previous is None or previous < atom, 'footprint atoms must be canonical and unique')
            previous = atom
        result[owner] = atoms
    _require(tuple(result) == tuple(sorted(result)), 'footprint owners must be canonical')
    return result


def _declarations(boxes, grid, equalities, authorities, preferences, compatible, poll):
    for rows in equalities:
        if type(rows) is not tuple:
            raise TypeError('fixed incumbent geometry equalities require exact tuples')
        for index, row in enumerate(rows):
            if index % 256 == 0:
                poll()
            _require(type(row) is tuple and len(row) == 3
                     and type(row[0]) is str and type(row[1]) is str
                     and row[0] < row[1] and row[0] in boxes and row[1] in boxes
                     and type(row[2]) is int and row[2] % grid == 0,
                     'master equality is malformed or off-grid')
    for rows, size, label in ((authorities, 6, 'authority'), (preferences, 3, 'preference'),
                               (compatible, 2, 'overlap exemption')):
        if type(rows) is not tuple:
            raise TypeError('fixed incumbent geometry ' + label + ' requires an exact tuple')
        previous, pairs = None, set()
        for index, row in enumerate(rows):
            if index % 256 == 0:
                poll()
            _require(type(row) is tuple and len(row) == size
                     and type(row[0]) is str and type(row[1]) is str
                     and row[0] < row[1] and row[0] in boxes and row[1] in boxes,
                     label + ' owner pair is malformed')
            if size > 2:
                _require(type(row[2]) is str and row[2] in ('horizontal', 'vertical'), label + ' axis is malformed')
            if size == 6:
                _require(type(row[3]) is str and type(row[4]) is str
                         and {row[3], row[4]} == {row[0], row[1]}
                         and type(row[5]) is tuple and bool(row[5])
                         and all(type(value) is str and value for value in row[5])
                         and row[5] == tuple(sorted(set(row[5]))), 'authority source IDs/endpoints are malformed')
            _require((previous is None or previous < row) and row[:2] not in pairs,
                     label + ' must be canonical and unique')
            previous = row
            pairs.add(row[:2])
    _require(not set(compatible).intersection(row[:2] for row in authorities),
             'overlap exemptions cannot also carry axis authority')
    poll()


def _atoms(boxes, plain, layered, poll):
    sizes, owners, layers = {}, {}, {}
    count = 0
    for owner, box in boxes.items():
        poll()
        width, height = box[2] - box[0], box[3] - box[1]
        atoms = layered.get(owner)
        rectangles = tuple(atom[2:] for atom in atoms) if atoms is not None else plain.get(owner, ((0, 0, width, height),))
        compound = atoms is not None or len(rectangles) != 1 or rectangles[0] != (0, 0, width, height)
        multiplicity, ordinals = Counter(), Counter()
        if atoms is not None:
            for index, atom in enumerate(atoms):
                if index % 256 == 0:
                    poll()
                multiplicity[atom[:2]] += 1
        for index, rectangle in enumerate(rectangles):
            count += 1
            if count % 256 == 0:
                poll()
            layer = None if atoms is None else atoms[index][:2]
            if not compound:
                node = owner
            elif layer is None:
                node = f'@full2d:{owner}:{index:04d}'
            else:
                node = f'@full2d:{owner}:L{layer[0]}:{layer[1]}'
                if multiplicity[layer] > 1:
                    node += f':{ordinals[layer]:04d}'
                ordinals[layer] += 1
            _require(node not in sizes and not (compound and node in boxes), 'atom node name collides with a master')
            sizes[node] = (rectangle[2] - rectangle[0], rectangle[3] - rectangle[1])
            owners[node], layers[node] = (owner, rectangle[:2]), layer
    poll()
    return (tuple(sorted(sizes)), tuple(sorted(sizes.items())),
            tuple((node, owner, offset) for node, (owner, offset) in sorted(owners.items())),
            tuple(sorted(layers.items())))


def _groups(proof, boxes, poll):
    sizes, layers, grouped = dict(proof.sizes_nm), dict(proof.node_layers), {}
    for index, (node, owner, (dx, dy)) in enumerate(proof.node_owner_offsets_nm):
        if index % 256 == 0:
            poll()
        x, y = boxes[owner][:2]
        width, height = sizes[node]
        grouped.setdefault(owner, {}).setdefault(layers[node], []).append((x + dx, y + dy, x + dx + width, y + dy + height))
    poll()
    return grouped


def _matching_layers(layer, other):
    if layer is None:
        yield from other
    else:
        if layer in other:
            yield layer
        if None in other:
            yield None


def _interactions(left, right):
    # One relation for each actually interacting layer group, including wildcard.
    for layer in left:
        for other in _matching_layers(layer, right):
            yield layer, other


def _assert_axis_dags(grouped, authorities, poll):
    # Collapse complete bipartite atom edges to (owner, layer) vertices. Traverse
    # implicit neighbors, retaining no cross-product edges even for wildcard layers.
    if not authorities:
        return
    for axis in ('horizontal', 'vertical'):
        incoming, outgoing = {}, {}
        for index, row in enumerate(authorities):
            if index % 256 == 0:
                poll()
            if row[2] == axis:
                outgoing.setdefault(row[3], []).append(row[4])
                incoming.setdefault(row[4], []).append(row[3])
        degrees = {}
        for owner, groups in grouped.items():
            for layer in groups:
                poll()
                degrees[owner, layer] = sum(
                    len(grouped[lower]) if layer is None else int(layer in grouped[lower]) + int(None in grouped[lower])
                    for lower in incoming.get(owner, ()))
        ready = deque(key for key, degree in degrees.items() if degree == 0)
        visited = 0
        while ready:
            owner, layer = ready.popleft()
            visited += 1
            if visited % 256 == 0:
                poll()
            for upper in outgoing.get(owner, ()):
                for other_layer in _matching_layers(layer, grouped[upper]):
                    poll()
                    key = upper, other_layer
                    degrees[key] -= 1
                    if degrees[key] == 0:
                        ready.append(key)
        _require(visited == len(degrees), axis + ' declaration graph contains a cycle')


def _failure(row_id, names, sources=()):
    return {'row_id': row_id, 'constraint_names': names, 'source_constraint_ids': sources}


def _physical_failures(grouped, gap, exempt, poll):
    indexes = {}
    owners = tuple(sorted(grouped))
    for index, first in enumerate(owners):
        for second in owners[index + 1:]:
            poll()
            if (first, second) in exempt:
                continue
            for left_layer, right_layer in _interactions(grouped[first], grouped[second]):
                poll()
                key = first, left_layer
                if key not in indexes:
                    indexes[key] = RectangleIndex(tuple(grouped[first][left_layer]), checkpoint=poll)
                if indexes[key].conflicts(tuple(grouped[second][right_layer]), clearance_nm=gap, checkpoint=poll):
                    yield _failure(f'four_way_disjunction:{first}<->{second}:{left_layer!r}:{right_layer!r}',
                                   ('physical_nonoverlap',), ('physical_nonoverlap',))


def _axis_failures(grouped, authorities, poll):
    if not authorities:
        return
    bounds = {}
    for owner, groups in grouped.items():
        for layer, boxes in groups.items():
            x0, y0, x1, y1 = boxes[0]
            for index, box in enumerate(boxes):
                if index % 256 == 0:
                    poll()
                x0, y0, x1, y1 = min(x0, box[0]), min(y0, box[1]), max(x1, box[2]), max(y1, box[3])
            bounds[owner, layer] = x0, y0, x1, y1
    for _, _, axis, lower, upper, sources in authorities:
        poll()
        coordinate = 0 if axis == 'horizontal' else 1
        for lower_layer, upper_layer in _interactions(grouped[lower], grouped[upper]):
            poll()
            if bounds[lower, lower_layer][coordinate + 2] > bounds[upper, upper_layer][coordinate]:
                yield _failure(f'fixed_axis:{lower}->{upper}:{axis}:{lower_layer!r}:{upper_layer!r}', sources, sources)
                break  # One witnessed failure per input declaration, not per atom pair.


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()


_FIELDS = ('seed_boxes_nm', 'nodes', 'sizes_nm', 'master_sizes_nm', 'node_owner_offsets_nm',
           'node_layers', 'minimum_clearance_nm', 'grid_nm', 'x_equalities', 'y_equalities',
           'pair_axis_authority', 'pair_axis_preference', 'overlap_compatible_pairs',
           'footprints_nm', 'layered_footprints_nm')


@dataclass(frozen=True, slots=True, init=False)
class FixedIncumbentGeometryProof:
    """Factory-owned immutable data; every public replay performs current queries.

    The seal detects mutation of this captured data. It grants no authority to
    caller-supplied source IDs, overlap exemptions, artwork, or PDK declarations.
    """
    seed_boxes_nm: tuple[tuple[str, Box], ...]
    nodes: tuple[str, ...]
    sizes_nm: tuple[tuple[str, tuple[int, int]], ...]
    master_sizes_nm: tuple[tuple[str, tuple[int, int]], ...]
    node_owner_offsets_nm: tuple[tuple[str, str, tuple[int, int]], ...]
    node_layers: tuple[tuple[str, Layer], ...]
    minimum_clearance_nm: int
    grid_nm: int
    x_equalities: tuple[Equality, ...]
    y_equalities: tuple[Equality, ...]
    pair_axis_authority: tuple[AxisAuthority, ...]
    pair_axis_preference: tuple[tuple[str, str, str], ...]
    overlap_compatible_pairs: tuple[tuple[str, str], ...]
    footprints_nm: tuple[tuple[str, tuple[Box, ...]], ...]
    layered_footprints_nm: tuple[tuple[str, tuple[tuple[int, int, int, int, int, int], ...]], ...]
    _sealed_sha256: str

    def __init__(self, *args, **kwargs):
        raise TypeError('FixedIncumbentGeometryProof requires fixed_incumbent_geometry_from_seed')

    def _payload(self):
        return {'schema': 'clay.fixed-incumbent-geometry-proof/v1',
                **{name: getattr(self, name) for name in _FIELDS}}

    def _validate(self, poll):
        poll()
        _require(type(self) is FixedIncumbentGeometryProof, 'requires an exact proof type')
        pending = [getattr(self, name) for name in _FIELDS]
        count = 0
        while pending:
            value = pending.pop()
            count += 1
            if count % 256 == 0:
                poll()
            if type(value) is tuple:
                pending.extend(value)
            else:
                _require(value is None or type(value) in (str, int), 'captured source data is no longer immutable')
        _require(type(self._sealed_sha256) is str and _digest(self._payload()) == self._sealed_sha256,
                 'captured source data changed after construction')
        poll()

    def __post_init__(self):
        self._validate(_poller(None))

    @property
    def master_nodes(self):
        return tuple(owner for owner, _ in self.master_sizes_nm)

    def payload(self):
        self.__post_init__()
        return self._payload()

    @property
    def sha256(self):
        self.__post_init__()
        return self._sealed_sha256

    def replay_diagnostic(self, boxes: Mapping[str, Box], *, checkpoint=None):
        """Return input/equality failures and bounded witnessed geometry failures.

        One physical collision is sufficient to reject; each failed declaration
        contributes at most one interactive-layer witness. No diagnostic builds
        a Cartesian list of all failed atom pairs.
        """
        poll = _poller(checkpoint)
        self._validate(poll)
        try:
            canonical = _boxes(boxes, poll)
        except (_BoxTypeError, _BoxValueError):
            return (_failure('graph_input:canonical_boxes', ('canonical_full2d_boxes',)),)
        if tuple(canonical) != self.master_nodes:
            return (_failure('graph_input:master_inventory', ('master_inventory',)),)
        failures, sizes = [], dict(self.master_sizes_nm)
        for owner, box in canonical.items():
            poll()
            if (box[2] - box[0], box[3] - box[1]) != sizes[owner]:
                failures.append(_failure('master_size:' + owner, ('master_size',)))
            if box[0] % self.grid_nm or box[1] % self.grid_nm:
                failures.append(_failure('manufacturing_grid:' + owner, ('manufacturing_grid',)))
        if not failures:
            grouped = _groups(self, canonical, poll)
            physical = next(_physical_failures(grouped, self.minimum_clearance_nm,
                            set(self.overlap_compatible_pairs), poll), None)
            if physical is not None:
                failures.append(physical)
            failures.extend(_axis_failures(grouped, self.pair_axis_authority, poll))
            for axis, rows in enumerate((self.x_equalities, self.y_equalities)):
                for index, (first, second, delta) in enumerate(rows):
                    if index % 256 == 0:
                        poll()
                    if canonical[second][axis] - canonical[first][axis] != delta:
                        label = 'x' if axis == 0 else 'y'
                        failures.append(_failure(f'{label}_equality:{first}->{second}:{delta}', (label + '_equality',)))
        self._validate(poll)
        try:
            ending = _boxes(boxes, poll)
        except (_BoxTypeError, _BoxValueError):
            ending = None
        if ending != canonical:
            failures.append(_failure('graph_input:current_boxes_changed', ('current_geometry_changed',)))
        return tuple(sorted(failures, key=lambda row: row['row_id']))

    def replay(self, boxes: Mapping[str, Box], *, checkpoint=None) -> bool:
        return not self.replay_diagnostic(boxes, checkpoint=checkpoint)


def fixed_incumbent_geometry_from_seed(
    boxes: Mapping[str, Box], *, minimum_clearance_nm: int, grid_nm: int,
    x_equalities: tuple[Equality, ...] = (), y_equalities: tuple[Equality, ...] = (),
    pair_axis_authority: tuple[AxisAuthority, ...] = (),
    pair_axis_preference: tuple[tuple[str, str, str], ...] = (),
    overlap_compatible_pairs: tuple[tuple[str, str], ...] = (),
    footprints_nm: tuple[tuple[str, tuple[Box, ...]], ...] = (),
    layered_footprints_nm: tuple[tuple[str, tuple[tuple[int, int, int, int, int, int], ...]], ...] = (),
    checkpoint=None,
) -> FixedIncumbentGeometryProof:
    """Capture the legacy factory's complete inputs without constructing pairs.

    As in the legacy factory, an illegal physical seed without axis authority
    is rejected immediately. With authority it can be represented, but replay
    still rejects any failed declared axis, physical spacing, or equality.
    """
    poll = _poller(checkpoint)
    poll()
    canonical = _boxes(boxes, poll)
    _require(type(minimum_clearance_nm) is int and minimum_clearance_nm >= 0, 'clearance must be an exact nonnegative integer')
    _require(type(grid_nm) is int and grid_nm > 0, 'grid must be an exact positive integer')
    _require(all(box[0] % grid_nm == 0 and box[1] % grid_nm == 0 for box in canonical.values()), 'seed origins are off-grid')
    plain = _footprints(footprints_nm, canonical, grid_nm, False, poll)
    layered = _footprints(layered_footprints_nm, canonical, grid_nm, True, poll)
    _declarations(canonical, grid_nm, (x_equalities, y_equalities), pair_axis_authority,
                  pair_axis_preference, overlap_compatible_pairs, poll)
    nodes, sizes, offsets, layers = _atoms(canonical, plain, layered, poll)
    proof = object.__new__(FixedIncumbentGeometryProof)
    values = (tuple(canonical.items()), nodes, sizes,
              tuple((owner, (box[2] - box[0], box[3] - box[1])) for owner, box in canonical.items()),
              offsets, layers, minimum_clearance_nm, grid_nm, x_equalities, y_equalities,
              pair_axis_authority, pair_axis_preference, overlap_compatible_pairs, footprints_nm, layered_footprints_nm)
    for name, value in zip(_FIELDS, values, strict=True):
        object.__setattr__(proof, name, value)
    object.__setattr__(proof, '_sealed_sha256', _digest(proof._payload()))
    proof._validate(poll)
    grouped = _groups(proof, canonical, poll)
    _assert_axis_dags(grouped, pair_axis_authority, poll)
    seed_exempt = set(overlap_compatible_pairs).union(row[:2] for row in pair_axis_authority)
    for failure in _physical_failures(grouped, minimum_clearance_nm, seed_exempt, poll):
        raise ValueError('fixed incumbent geometry seed is not clearance-legal: ' + failure['row_id'])
    # The caller's mapping is mutable; reject changes during this capture. Later
    # replays consume their own exact current boxes, not this caller object.
    poll()
    _require(_boxes(boxes, poll) == canonical, 'source boxes changed during construction')
    return proof
