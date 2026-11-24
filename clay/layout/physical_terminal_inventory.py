"""Source terminal coverage under the actual PDK routing-metal/via model.

These immutable records are data, not execution or source-authority tokens.
The materializer supplies the complete independent source denominator; replay
compares it with fresh artwork and a freshly derived ConnectModel. Connectivity
is limited to ConnectModel.from_tech routing metals and vias. It does not prove
poly/contact/OD, device extraction, routing accessibility, or DRC legality.

Each source-bearing electrical component retains its metal masks once. Other
artwork remains bound by the complete raw-polygon digest, without manufacturing
external terminals for unseeded fill or dummy metal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import klayout.db as kdb

from clay.layout.backend import LayerPolygon
from clay.layout.connectivity import ConnectModel, extract_nodes
from clay.layout.occurrence_channels import _canonical_ring, _raw_polygon, _shape_keys


_LIMIT = 1_000_000_000
_MAX_ITEMS = 262_144
_MAX_VERTICES = 1_048_576
_SCOPE = "ConnectModel.from_tech:routing-metal-via-only/v1"


def _require(condition, message):
    if not condition:
        raise ValueError("physical terminal " + message)


def _text(value, field):
    _require(type(value) is str and 0 < len(value) <= 4096, field + " must be a bounded exact string")


def _integer(value, field, *, minimum=-_LIMIT, maximum=_LIMIT):
    _require(type(value) is int and minimum <= value <= maximum, field + " must be a bounded exact integer")


def _tuple(value, field, *, nonempty=False, limit=_MAX_ITEMS):
    _require(type(value) is tuple and len(value) <= limit and (bool(value) or not nonempty),
             field + " must be a bounded exact tuple")
    return value


def _layer(value):
    _require(type(value) is tuple and len(value) == 2, "layer must be an exact pair")
    for item in value:
        _integer(item, "layer", minimum=0, maximum=65535)


def _point(value):
    _require(type(value) is tuple and len(value) == 2, "point must be an exact pair")
    for item in value:
        _integer(item, "coordinate")


def _box(value):
    _require(type(value) is tuple and len(value) == 4, "source bbox must be an exact tuple")
    for item in value:
        _integer(item, "source bbox coordinate")
    _require(value[0] < value[2] and value[1] < value[3], "source bbox must have positive area")


def _sha_value(value, field):
    _require(type(value) is str and len(value) == 64
             and all(char in "0123456789abcdef" for char in value), field + " must be SHA256")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _component_id(source_ids, net):
    return "physical-terminal-component:" + _digest({"source_ids": source_ids, "net": net})


def _ring(value):
    _tuple(value, "polygon contour", nonempty=True, limit=4096)
    for point in value:
        _point(point)
    _require(len(value) >= 3 and value == _canonical_ring(value), "polygon contour must be canonical")
    area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(value, value[1:] + value[:1]))
    _require(area2 != 0, "polygon contour has zero area")


@dataclass(frozen=True, slots=True, order=True)
class PhysicalTerminalPolygonNM:
    hull_nm: tuple[tuple[int, int], ...]
    holes_nm: tuple[tuple[tuple[int, int], ...], ...] = ()

    def __post_init__(self):
        _ring(self.hull_nm)
        _tuple(self.holes_nm, "polygon holes", limit=4096)
        for hole in self.holes_nm:
            _ring(hole)
        _require(self.holes_nm == tuple(sorted(set(self.holes_nm))), "polygon holes must be unique and canonical")
        _require(len(self.hull_nm) + sum(map(len, self.holes_nm)) <= 4096, "polygon exceeds vertex bound")

    def to_region(self):
        self.__post_init__()
        region = kdb.Region(_raw_polygon(self.hull_nm, self.holes_nm))
        _require(_shape_keys(region) == ((self.hull_nm, self.holes_nm),), "component polygon geometry is invalid")
        return region

    def payload(self):
        self.__post_init__()
        return {"hull_nm": self.hull_nm, "holes_nm": self.holes_nm}


@dataclass(frozen=True, slots=True)
class PhysicalTerminalSource:
    source_id: str
    member: str
    occurrence_index: int
    role: str
    net: str
    layer: tuple[int, int]
    bbox_nm: tuple[int, int, int, int]

    def __post_init__(self):
        for field in ("source_id", "member", "role", "net"):
            _text(getattr(self, field), "source " + field)
        _integer(self.occurrence_index, "source occurrence index", minimum=0, maximum=_MAX_ITEMS)
        _layer(self.layer)
        _box(self.bbox_nm)

    def payload(self):
        self.__post_init__()
        return {"source_id": self.source_id, "member": self.member,
            "occurrence_index": self.occurrence_index, "role": self.role, "net": self.net,
            "layer": self.layer, "bbox_nm": self.bbox_nm}


@dataclass(frozen=True, slots=True)
class PhysicalTerminalComponent:
    component_id: str
    net: str
    source_ids: tuple[str, ...]
    geometry_by_layer_nm: tuple[tuple[tuple[int, int], tuple[PhysicalTerminalPolygonNM, ...]], ...]
    seed_layer: tuple[int, int]
    seed_xy_nm: tuple[int, int]

    def __post_init__(self):
        _text(self.net, "component net")
        _tuple(self.source_ids, "component source ids", nonempty=True)
        for source_id in self.source_ids:
            _text(source_id, "component source id")
        _require(self.source_ids == tuple(sorted(set(self.source_ids))), "component source ids must be unique and canonical")
        _require(self.component_id == _component_id(self.source_ids, self.net), "component identity differs from its source partition")
        _tuple(self.geometry_by_layer_nm, "component geometry", nonempty=True)
        layers = []
        for row in self.geometry_by_layer_nm:
            _require(type(row) is tuple and len(row) == 2, "component geometry layer record is malformed")
            layer, shapes = row
            _layer(layer)
            layers.append(layer)
            _tuple(shapes, "component polygon inventory", nonempty=True)
            for shape in shapes:
                _require(type(shape) is PhysicalTerminalPolygonNM, "component polygon has a foreign type")
                shape.__post_init__()
            _require(shapes == tuple(sorted(set(shapes))), "component polygons must be canonical and unique")
        _require(layers == sorted(set(layers)), "component geometry layers must be canonical and unique")
        _layer(self.seed_layer)
        _point(self.seed_xy_nm)
        _require(self.seed_layer in layers, "component seed layer is outside its geometry")

    def payload(self):
        self.__post_init__()
        return {"component_id": self.component_id, "net": self.net, "source_ids": self.source_ids,
            "geometry_by_layer_nm": tuple((layer, tuple(shape.payload() for shape in shapes))
                for layer, shapes in self.geometry_by_layer_nm),
            "seed_layer": self.seed_layer, "seed_xy_nm": self.seed_xy_nm}


def _sources(value):
    _tuple(value, "source inventory", nonempty=True)
    for source in value:
        _require(type(source) is PhysicalTerminalSource, "source inventory requires exact typed sources")
        source.__post_init__()
    _require(len({row.source_id for row in value}) == len(value), "source inventory contains duplicate ids")
    semantic = {(row.member, row.occurrence_index, row.role, row.layer, row.bbox_nm) for row in value}
    _require(len(semantic) == len(value), "source inventory contains duplicate physical terminal records")
    return tuple(sorted(value, key=lambda row: row.source_id))


@dataclass(frozen=True, slots=True)
class PhysicalTerminalInventory:
    sources: tuple[PhysicalTerminalSource, ...]
    components: tuple[PhysicalTerminalComponent, ...]
    artwork_sha256: str
    connect_model_sha256: str
    pdk_provenance_sha256: str
    pdk_id: str
    manufacturing_grid_nm: int

    def __post_init__(self):
        _require(_sources(self.sources) == self.sources, "source inventory must be canonical")
        for field in ("artwork_sha256", "connect_model_sha256", "pdk_provenance_sha256"):
            _sha_value(getattr(self, field), field)
        _text(self.pdk_id, "PDK id")
        _integer(self.manufacturing_grid_nm, "manufacturing grid", minimum=1)
        _tuple(self.components, "component inventory", nonempty=True)
        by_source = {row.source_id: row for row in self.sources}
        used = []
        ids = []
        vertex_count = 0
        for component in self.components:
            _require(type(component) is PhysicalTerminalComponent, "component inventory requires exact typed components")
            component.__post_init__()
            ids.append(component.component_id)
            used.extend(component.source_ids)
            _require(set(component.source_ids) <= set(by_source), "component references an absent source")
            _require({by_source[source_id].net for source_id in component.source_ids} == {component.net},
                     "component has mixed source nets")
            _require(all(value % self.manufacturing_grid_nm == 0 for value in component.seed_xy_nm),
                     "component seed is off manufacturing grid")
            for _layer_key, shapes in component.geometry_by_layer_nm:
                vertex_count += sum(len(shape.hull_nm) + sum(map(len, shape.holes_nm)) for shape in shapes)
        _require(ids == sorted(set(ids)), "component inventory identities must be unique and canonical")
        _require(Counter(used) == Counter(by_source.keys()), "component source partition is incomplete or duplicated")
        _require(vertex_count <= _MAX_VERTICES, "component geometry exceeds total vertex bound")

    def payload(self):
        # Structural nested replay only. Electrical extraction is performed once
        # by compile/replay at the actual materialization boundary, never here.
        self.__post_init__()
        return {"schema": "clay.physical-terminal-inventory/v1", "connect_scope": _SCOPE,
            "sources": tuple(row.payload() for row in self.sources),
            "components": tuple(row.payload() for row in self.components),
            "artwork_sha256": self.artwork_sha256, "connect_model_sha256": self.connect_model_sha256,
            "pdk_provenance_sha256": self.pdk_provenance_sha256, "pdk_id": self.pdk_id,
            "manufacturing_grid_nm": self.manufacturing_grid_nm}

    @property
    def sha256(self):
        return _digest(self.payload())


def _artwork(geometry):
    polygons = geometry if type(geometry) is tuple else getattr(geometry, "polygons", None)
    _require(type(polygons) in {tuple, list} and 0 < len(polygons) <= _MAX_ITEMS,
             "geometry requires complete actual LayerPolygon artwork")
    records, shapes = [], {}
    vertices = 0
    for polygon in polygons:
        _require(type(polygon) is LayerPolygon, "artwork polygon has a foreign type")
        layer = (polygon.layer, polygon.datatype)
        _layer(layer)
        contour = getattr(polygon.polygon, "vertices", None)
        _require(type(contour) in {list, tuple} and 3 <= len(contour) <= 4096,
                 "artwork contour requires a bounded vertex sequence")
        vertices += len(contour)
        _require(vertices <= _MAX_VERTICES, "artwork exceeds total vertex bound")
        raw = tuple((point.x, point.y) for point in contour)
        for point in raw:
            _point(point)
        ring = _canonical_ring(raw)
        _ring(ring)
        actual = kdb.Polygon([kdb.Point(x, y) for x, y in ring])
        _require(actual.area() > 0, "artwork polygon has no area")
        records.append((layer, ring))
        shapes.setdefault(layer, []).append(actual)
    return _digest(tuple(sorted(records))), shapes


def _model(tech):
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.resource_world import pdk_provenance_digest

    model = ConnectModel.from_tech(tech)
    _require(bool(model.metals) and len(set(model.metals)) == len(model.metals),
             "PDK routing model has an empty or duplicate metal inventory")
    for layer in model.metals:
        _layer(layer)
    _require(not model.extra_connect and not model.derived and not model.bridges,
             "connect model exceeds routing-metal-via scope")
    for via, lower, upper in model.vias:
        _layer(via)
        _require(lower in model.metals and upper in model.metals and lower != upper,
                 "PDK via references an invalid routing layer")
    payload = {"scope": _SCOPE, "metals": model.metals, "vias": model.vias,
               "extra_connect": model.extra_connect, "derived": model.derived, "bridges": model.bridges}
    pdk_id = tech.pdk_id
    _text(pdk_id, "PDK id")
    return model, _digest(payload), pdk_provenance_digest(tech), pdk_id, pdk_manufacturing_grid_nm(tech)


def _seed(source, grid):
    coordinates = []
    for lower, upper in zip(source.bbox_nm[:2], source.bbox_nm[2:]):
        first = (lower // grid + 1) * grid
        last = ((upper - 1) // grid) * grid
        _require(first <= last, "source metal has no interior manufacturing-grid seed")
        nearest = ((lower + upper + grid) // (2 * grid)) * grid
        coordinates.append(max(first, min(last, nearest)))
    return source.layer, tuple(coordinates)


def compile_physical_terminal_inventory(*, geometry, sources, tech):
    """Compile every supplied source terminal against fresh actual connectivity.

    Complete source capture is the caller's separate obligation: an unseeded
    metal polygon alone cannot say whether it is a terminal, fill, or a dummy.
    """
    sources = _sources(sources)
    artwork_sha, shapes = _artwork(geometry)
    model, model_sha, pdk_sha, pdk_id, grid = _model(tech)
    nodes = extract_nodes(shapes, model)
    node_bbs = [
        {lay: reg.bbox() for lay, reg in node.items()}
        for node in nodes
    ]
    by_node = {}
    for source in sources:
        _require(source.layer in model.metals, "source layer is outside the routing-metal scope")
        sx0, sy0, sx1, sy1 = source.bbox_nm
        terminal = kdb.Region(kdb.Box(sx0, sy0, sx1, sy1))
        hits = []
        slay = source.layer
        for index, node in enumerate(nodes):
            bb = node_bbs[index].get(slay)
            if bb is None:
                continue
            if bb.left >= sx1 or sx0 >= bb.right or bb.bottom >= sy1 or sy0 >= bb.top:
                continue
            region = node[slay]
            if not (terminal & region).is_empty():
                hits.append(index)
        _require(len(hits) == 1, "source terminal must belong to one actual metal component")
        index, = hits
        _require((terminal - nodes[index][slay]).is_empty(),
                 "source terminal bbox is not wholly covered by actual metal artwork")
        by_node.setdefault(index, []).append(source)
    components = []
    for index, node_sources in by_node.items():
        nets = {source.net for source in node_sources}
        _require(len(nets) == 1, "physical component contains mixed source nets")
        net, = nets
        source_ids = tuple(sorted(source.source_id for source in node_sources))
        masks = tuple((layer, tuple(PhysicalTerminalPolygonNM(hull, holes)
            for hull, holes in _shape_keys(region))) for layer, region in sorted(nodes[index].items()))
        seed_source = min(node_sources, key=lambda row: row.source_id)
        seed_layer, seed_xy = _seed(seed_source, grid)
        components.append(PhysicalTerminalComponent(_component_id(source_ids, net), net, source_ids,
            masks, seed_layer, seed_xy))
    return PhysicalTerminalInventory(sources, tuple(sorted(components, key=lambda row: row.component_id)),
        artwork_sha, model_sha, pdk_sha, pdk_id, grid)


def replay_physical_terminal_inventory(*, geometry, sources, inventory, tech):
    """Deep electrical replay against independent sources, artwork and PDK data."""
    _require(type(inventory) is PhysicalTerminalInventory, "inventory requires an exact typed value")
    inventory.__post_init__()
    _require(inventory.sources == _sources(sources), "source inventory differs from the complete expected denominator")
    actual = compile_physical_terminal_inventory(geometry=geometry, sources=sources, tech=tech)
    _require(inventory == actual, "inventory source/artwork/component/PDK model or canonical seed differs")
    return inventory


__all__ = ["PhysicalTerminalSource", "PhysicalTerminalPolygonNM", "PhysicalTerminalComponent",
           "PhysicalTerminalInventory", "compile_physical_terminal_inventory", "replay_physical_terminal_inventory"]
