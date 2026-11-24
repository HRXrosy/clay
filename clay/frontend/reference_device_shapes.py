"""Exact PCell recognition data; no source, PDK or execution authority is minted.

Artwork is retained as a multiset. A shared marker or metal component can have
several logical consumers; association never clips its original polygons.
The present recognition vocabulary is thick planar MOS, straight N-poly
segments under SAB/resistor markers, and one explicitly drawn CTM MIM level.
"""
from dataclasses import dataclass

import klayout.db as kdb

from clay.layout.occurrence_channels import _shape_keys
from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM
from clay.layout.terminal_body_geometry import opposing_terminal_axis as _opposing

_D4 = frozenset(((1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0),
                 (1, 0, 0, -1), (0, 1, 1, 0), (-1, 0, 0, 1), (0, -1, -1, 0)))
_LIMIT = 262144


def _require(value, message):
    if not value:
        raise ValueError('reference device geometry: ' + message)


def _shapes(region):
    return tuple(PhysicalTerminalPolygonNM(hull, holes) for hull, holes in _shape_keys(region))


def _region(shapes):
    result = kdb.Region()
    for shape in shapes:
        _require(type(shape) is PhysicalTerminalPolygonNM, 'polygon type is not exact')
        result += shape.to_region()
    return result.merged()


def _box(region):
    value = region.bbox()
    return value.left, value.bottom, value.right, value.top


def _map_shape(shape, frame):
    a, b, c, d, dx, dy = frame
    def points(ring):
        return [kdb.Point(a*x+b*y+dx, c*x+d*y+dy) for x, y in ring]
    polygon = kdb.Polygon(points(shape.hull_nm))
    for hole in shape.holes_nm:
        polygon.insert_hole(points(hole))
    transformed, = _shapes(kdb.Region(polygon))
    return transformed


def _map_region(region, frame):
    return _region(tuple(_map_shape(shape, frame) for shape in _shapes(region)))


def _rect(region, label):
    _require(len(tuple(region.each())) == 1, label + ' is not one region')
    box = region.bbox()
    _require((region ^ kdb.Region(box)).is_empty(), label + ' is not a complete rectangle')
    return box


@dataclass(frozen=True, slots=True)
class ReferencePrimitiveGeometry:
    family: str
    body_nm: tuple
    center2_nm: tuple[int, int]
    center_basis: str
    width_nm: int
    length_nm: int
    channel_length_axis: tuple[int, int]
    raw_to_global: tuple[int, ...]
    terminals: tuple
    owned_part_indices: tuple[int, ...]
    poly_part_indices: tuple[int, ...]
    contact_part_indices: tuple[int, ...]
    owned_bbox_nm: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ReferencePCellGeometry:
    parts: tuple
    primitives: tuple[ReferencePrimitiveGeometry, ...]


def _mos(regions, keys, family):
    diff, poly = regions[keys['diff']], regions[keys['poly']]
    body = (diff & poly).merged()
    _rect(body, 'MOS channel')
    implant = regions.get(keys['pplus' if family == 'pmos' else 'nplus'], kdb.Region())
    _require((diff - implant).is_empty(), 'MOS diffusion implant is incomplete')
    oxide = regions.get(keys['od2'], kdb.Region())
    _require((body - oxide).is_empty(), 'thick MOS oxide marker is incomplete')
    if family == 'pmos':
        _require((diff - regions.get(keys['nwell'], kdb.Region())).is_empty(), 'PMOS well is incomplete')
    sides = tuple(kdb.Region(item) for item in sorted((diff-poly).each(), key=lambda p: _box(kdb.Region(p))))
    axis, width, length = _opposing(body, sides)
    gate = poly.interacting(body).merged()
    return ((body, 'channel_bbox_center', axis, width, length,
             (('gate', keys['poly'], gate), ('sd0', keys['diff'], sides[0]),
              ('sd1', keys['diff'], sides[1])), (), ()),)


def _resistors(parts, regions, keys, checkpoint):
    poly = regions[keys['poly']]
    body = (poly & regions[keys['sab']] & regions[keys['res_mk']] & regions[keys['nplus']]).merged()
    polys = tuple(kdb.Region(p) for p in sorted(poly.each(), key=lambda p: _box(kdb.Region(p))))
    _require(polys and len(polys) == len(tuple(body.each())), 'resistor body/poly denominator differs')
    contact_parts = [(i, shape.to_region()) for i, (key, shape) in enumerate(parts) if key == keys['contact']]
    metal = regions.get(keys['metal1'], kdb.Region())
    output, claimed = [], set()
    for full_poly in polys:
        checkpoint()
        segment = (body & full_poly).merged()
        _rect(segment, 'resistor body')
        heads = tuple(kdb.Region(p) for p in sorted((full_poly-segment).each(), key=lambda p: _box(kdb.Region(p))))
        axis, width, length = _opposing(segment, heads)
        indices = []
        for head in heads:
            hits = [(i, cut) for i, cut in contact_parts if not (cut & head).is_empty()]
            _require(hits, 'resistor head has no contact')
            for i, cut in hits:
                _require((cut-head).is_empty() and (cut-metal).is_empty(), 'resistor contact is not enclosed by head and M1')
                _require(i not in claimed, 'contact has ambiguous segment/head owner')
                claimed.add(i)
                indices.append(i)
        poly_indices = tuple(i for i, (key, shape) in enumerate(parts)
                             if key == keys['poly'] and not (shape.to_region() & full_poly).is_empty())
        output.append((segment, 'resistor_body_bbox_center', axis, width, length,
                       (('head0', keys['poly'], heads[0]), ('head1', keys['poly'], heads[1])),
                       poly_indices, tuple(sorted(indices))))
    _require(claimed == {i for i, _ in contact_parts}, 'unowned resistor contact remains')
    return tuple(output)


def _mim(regions, keys):
    levels = tuple(level for level in range(2, 6) if f'ctm{level}' in keys
                   and not regions.get(keys[f'ctm{level}'], kdb.Region()).is_empty())
    _require(len(levels) == 1, 'MIM requires one explicit CTM level')
    level, = levels
    body = regions[keys[f'ctm{level}']]
    box = _rect(body, 'CTM plate')
    bottom, cover, via = (keys[f'metal{level}'], keys[f'metal{level+1}'], keys[f'via{level}'])
    _require((body-regions.get(bottom, kdb.Region())).is_empty(), 'MIM bottom does not cover CTM')
    cuts = regions.get(via, kdb.Region()) & body
    _require(not cuts.is_empty() and (cuts-regions.get(cover, kdb.Region())).is_empty(),
             'MIM CTM lacks complete real via/cover access')
    return ((body, 'ctm_plate_bbox_center', (1, 0), box.height(), box.width(),
             (('bottom', bottom, body), ('top', keys[f'ctm{level}'], body)), (), ()),)


def observe_reference_pcell_geometry(*, geometry_by_layer_nm, family, semantic_layers,
                                     raw_to_global, checkpoint):
    """Recognize current supplied artwork and transport every record by exact D4.

    Local MOS gate/SD probes need not carry local metal. The authority producer
    separately traces them against the complete top-level conductor graph.
    Positive channel axis denotes the raw master's increasing x/y, not D/S.
    """
    checkpoint()
    _require(type(raw_to_global) is tuple and len(raw_to_global) == 6
             and all(type(v) is int and abs(v) <= 1_000_000_000 for v in raw_to_global)
             and raw_to_global[:4] in _D4, 'native frame must be integer D4')
    _require(type(semantic_layers) is tuple and semantic_layers == tuple(sorted(semantic_layers)),
             'semantic layers must be canonical')
    keys = dict(semantic_layers)
    _require(len(keys) == len(semantic_layers), 'semantic layers have duplicate roles')
    _require(type(geometry_by_layer_nm) is tuple and geometry_by_layer_nm,
             'artwork requires a complete immutable inventory')
    parts = []
    for key, shapes in geometry_by_layer_nm:
        checkpoint()
        _require(type(key) is tuple and len(key) == 2 and all(type(v) is int and 0 <= v <= 65535 for v in key),
                 'invalid raw layer')
        _require(type(shapes) is tuple and shapes, 'empty/mutable raw polygon inventory')
        for shape in shapes:
            _require(type(shape) is PhysicalTerminalPolygonNM, 'foreign polygon record')
            shape.to_region()
            parts.append((key, shape))
    _require(len(parts) <= _LIMIT and len(dict(geometry_by_layer_nm)) == len(geometry_by_layer_nm),
             'raw polygon/layer denominator is invalid')
    parts = tuple(parts)
    regions = {key: _region(shapes) for key, shapes in geometry_by_layer_nm}
    required = {'pmos': ('diff', 'poly', 'pplus', 'nwell', 'od2'),
                'nmos': ('diff', 'poly', 'nplus', 'od2'),
                'resistor': ('poly', 'sab', 'res_mk', 'nplus', 'contact', 'metal1'),
                'capacitor': ()}
    _require(family in required, 'unsupported device family')
    for role in required[family]:
        _require(role in keys, 'semantic layer missing: ' + role)
        _require(keys[role] in regions, 'required channel/oxide/contact layer absent: ' + role)
    if family in ('pmos', 'nmos'):
        recognized = _mos(regions, keys, family)
    elif family == 'resistor':
        recognized = _resistors(parts, regions, keys, checkpoint)
    else:
        recognized = _mim(regions, keys)
    transformed = tuple((key, _map_shape(shape, raw_to_global)) for key, shape in parts)
    result, all_owned = [], set()
    a, b, c, d, _x, _y = raw_to_global
    for body, basis, axis, width, length, terminals, poly_ids, cut_ids in recognized:
        checkpoint()
        if family != 'resistor':
            owned = tuple(range(len(parts)))
        else:
            full_poly = _region(tuple(parts[i][1] for i in poly_ids))
            contacts = _region(tuple(parts[i][1] for i in cut_ids))
            metal = regions[keys['metal1']].interacting(contacts).merged()
            owned = tuple(i for i, (key, shape) in enumerate(parts)
                          if (key == keys['poly'] and i in poly_ids)
                          or (key == keys['contact'] and i in cut_ids)
                          or (key == keys['metal1'] and not (shape.to_region() & metal).is_empty())
                          or (key not in (keys['poly'], keys['contact'], keys['metal1'])
                              and not (shape.to_region() & full_poly).is_empty()))
        all_owned.update(owned)
        global_body = _map_region(body, raw_to_global)
        x0, y0, x1, y1 = _box(global_body)
        envelope = kdb.Region()
        for i in owned:
            envelope += transformed[i][1].to_region()
        result.append(ReferencePrimitiveGeometry(family, _shapes(global_body), (x0+x1, y0+y1),
            basis, width, length, (a*axis[0]+b*axis[1], c*axis[0]+d*axis[1]), raw_to_global,
            tuple((role, layer, _shapes(_map_region(probe, raw_to_global))) for role, layer, probe in terminals),
            owned, poly_ids, cut_ids, _box(envelope)))
    _require(all_owned == set(range(len(parts))), 'complete artwork has an unowned part')
    checkpoint()
    return ReferencePCellGeometry(transformed, tuple(result))
