"""Exact supplied MIM artwork to native geometry and local access observations.

The caller supplies the PDK identity, MIM connect model and complete site/role
mapping. This adapter measures their local geometric relationship; it does not
authenticate the reference, admit a source device, or establish DRC/signoff.
No access metal, via, riser or placement coordinate is synthesized. Complete
site routing masks remain separate from DeviceGeom's representative terminals.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import hashlib
import json

import klayout.db as kdb

from clay.frontend.unit_master_geometry import (
    LayerGeometry, UnitMasterGeometry, UnitTerminalSite, validate_unit_master_regions,
)
from clay.layout.backend import LayerPolygon, Point, Polygon
from clay.layout.capacitor_unit_occurrences import CapacitorUnitOccurrenceGeometry
from clay.layout.connectivity import DerivedLayer
from clay.layout.device_cell import DeviceGeom, Terminal
from clay.layout.device_terminal import MimConnectSpec, extract_device_nodes, extract_terminal_access
from clay.layout.occurrence_channels import _shape_keys
from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM

_ROTATIONS = {"R0": 0, "R90": 1, "R180": 2, "R270": 3}
_ORIENTATIONS = tuple(_ROTATIONS) + tuple("M" + rotation for rotation in _ROTATIONS)


def _require(condition, message):
    if not condition:
        raise ValueError("MIM unit " + message)


def _pairs(value, field):
    _require(type(value) is tuple and 0 < len(value) <= 262_144,
             field + " requires a bounded nonempty exact tuple")
    _require(all(type(row) is tuple and len(row) == 2 for row in value),
             field + " requires exact pairs")
    keys = tuple(row[0] for row in value)
    _require(all(type(key) is str and 0 < len(key) <= 4096 for key in keys),
             field + " requires bounded exact string keys")
    _require(keys == tuple(sorted(set(keys))), field + " keys must be canonical and unique")
    return dict(value)


def _model_literal(value):
    if value is None or type(value) in (str, int):
        return value
    if type(value) in (tuple, list):
        return tuple(_model_literal(item) for item in value)
    if type(value) is dict:
        return tuple(sorted((_model_literal(key), _model_literal(item)) for key, item in value.items()))
    if type(value) in (MimConnectSpec, DerivedLayer):
        return tuple((field.name, _model_literal(getattr(value, field.name))) for field in fields(value))
    raise ValueError("MIM unit connect model contains unsupported data")


@dataclass(frozen=True, slots=True)
class MimSiteAccess:
    """One independent supplied site and all real routing masks of its node."""
    site_id: str
    role: str
    node_id: int
    routing_geometry_by_layer_nm: LayerGeometry

    def __post_init__(self):
        _require(type(self.role) is str and self.role in {"p", "n"}, "site role must be p or n")
        _require(type(self.node_id) is int and self.node_id in (0, 1), "site requires one of two local nodes")
        UnitTerminalSite(self.site_id, self.role, self.routing_geometry_by_layer_nm)


@dataclass(frozen=True, slots=True)
class MaterializedMimUnit:
    """Factory result, without a source-admission or immutable DeviceGeom claim.

The original master and site records are immutable. DeviceGeom is the existing
native geometry currency; consumers must replay it before relying on retained
observations after mutation, copying, banking or placement.
"""
    master: UnitMasterGeometry
    geometry: DeviceGeom
    site_accesses: tuple[MimSiteAccess, ...]
    terminal_layers: tuple[tuple[str, tuple[int, int]], ...]
    mim_model_data: tuple

    def __post_init__(self):
        _require(type(self.master) is UnitMasterGeometry, "requires an exact typed master")
        self.master.__post_init__()
        _require(type(self.geometry) is DeviceGeom and self.geometry.kind == "cap",
                 "requires native capacitor geometry")
        _require(type(self.site_accesses) is tuple, "site access inventory must be an exact tuple")
        _require(all(type(row) is MimSiteAccess for row in self.site_accesses),
                 "requires exact site access records")
        for row in self.site_accesses:
            row.__post_init__()
        _require(tuple(row.site_id for row in self.site_accesses) == tuple(
            site.site_id for site in self.master.terminal_sites), "site access denominator differs from master")
        _require({row.role for row in self.site_accesses} == {"p", "n"}, "both terminal roles are required")
        role_nodes = {role: {row.node_id for row in self.site_accesses if row.role == role} for role in ("p", "n")}
        _require(all(len(nodes) == 1 for nodes in role_nodes.values()) and
                 role_nodes["p"] != role_nodes["n"], "terminal roles must bind distinct complete nodes")
        layers = _pairs(self.terminal_layers, "terminal layers")
        _require(set(layers) == {"p", "n"}, "terminal layers require both roles")
        for role, layer in layers.items():
            _require(all(layer in dict(row.routing_geometry_by_layer_nm)
                         for row in self.site_accesses if row.role == role), "terminal layer differs from node masks")
        _require(type(self.mim_model_data) is tuple and bool(self.mim_model_data)
                 and _model_literal(self.mim_model_data) == self.mim_model_data,
                 "connect model snapshot must be immutable data")

    @property
    def mim_model_sha256(self):
        payload = json.dumps(self.mim_model_data, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _regions(geometry):
    result = {}
    for layer, shapes in geometry:
        region = kdb.Region()
        for shape in shapes:
            region += shape.to_region()
        result[layer] = region.merged()
    return result


def _retained(regions):
    return tuple((layer, tuple(PhysicalTerminalPolygonNM(hull, holes)
        for hull, holes in _shape_keys(region))) for layer, region in sorted(regions.items())
        if not region.is_empty())


def _native_polygons(regions):
    """Backend polygons have no holes: partition, then replay exact layer XOR."""
    result = []
    for layer, region in sorted(regions.items()):
        replay = kdb.Region()
        for polygon in region.each():
            pieces = polygon.decompose_trapezoids() if polygon.holes() else (polygon,)
            for piece in pieces:
                points = tuple(piece.each_point()) if isinstance(piece, kdb.SimplePolygon) else tuple(piece.each_point_hull())
                native = LayerPolygon(layer[0], layer[1], Polygon([Point(point.x, point.y) for point in points]))
                result.append(native)
                replay.insert(kdb.Polygon([kdb.Point(point.x, point.y) for point in native.polygon.vertices]))
        _require((region ^ replay).is_empty(), "native polygon conversion changed original artwork")
    return result


def _transform_regions(master, orientation):
    mirrored = orientation.startswith("M")
    rotation = orientation[1:] if mirrored else orientation
    transform = kdb.Trans(_ROTATIONS[rotation], mirrored, 0, 0)
    transformed = {layer: region.transformed(transform)
                   for layer, region in _regions(master.geometry_by_layer_nm).items()}
    dx = -min(region.bbox().left for region in transformed.values())
    dy = -min(region.bbox().bottom for region in transformed.values())
    shift = kdb.Trans(dx, dy)
    return {layer: region.transformed(shift) for layer, region in transformed.items()}, shift * transform, (dx, dy)


def _routing_masks(node, artwork, tech, spec):
    metal_order = []
    for level in range(1, 9):
        try:
            layer = tuple(tech.metal_key(level))
        except KeyError:
            continue
        if layer not in metal_order:
            metal_order.append(layer)
    result = {}
    for layer, region in node.items():
        real = spec.synthetic_to_real.get(layer, layer)
        if real not in metal_order or real not in artwork:
            continue
        overlap = region & artwork[real]
        if not overlap.is_empty():
            result[real] = result.get(real, kdb.Region()) + overlap
    return {layer: result[layer].merged() for layer in metal_order if layer in result}


def _site_node(site, transform, nodes):
    query = {layer: region.transformed(transform) for layer, region in _regions(site.geometry_by_layer_nm).items()}
    matches = [index for index, node in enumerate(nodes) if all(
        layer in node and (region - node[layer]).is_empty() for layer, region in query.items())]
    _require(len(matches) == 1, "site must be wholly contained on exactly one local electrical node")
    selected = matches[0]
    _require(not any(index != selected and any(layer in node and not (region & node[layer]).is_empty()
        for layer, region in query.items()) for index, node in enumerate(nodes)), "site intersects a foreign node")
    return selected


def _contained_box(region):
    """Choose an existing rectangular landing; never draw or fill its bbox."""
    candidates = []
    for polygon in region.each():
        for piece in polygon.decompose_trapezoids():
            box = piece.bbox()
            query = kdb.Region(box)
            if box.width() > 0 and box.height() > 0 and (query - region).is_empty():
                candidates.append((-box.area(), box.left, box.bottom, box.right, box.top))
    _require(bool(candidates), "node has no contained integer rectangular routing landing")
    return min(candidates)[1:]


def _terminal_landing(node_index, routing):
    # Existing native helpers identify a terminal by a box, with no layer. Prefer
    # real metal outside the other node's projection so that representative is
    # also unambiguous through those older interfaces. All masks are retained.
    foreign = kdb.Region()
    for index, layers in enumerate(routing):
        if index != node_index:
            for region in layers.values():
                foreign += region
    for exclusive in (True, False):
        for layer, region in routing[node_index].items():
            candidate = region - foreign if exclusive else region
            if candidate.is_empty():
                continue
            try:
                return layer, _contained_box(candidate)
            except ValueError:
                continue
    raise ValueError("MIM unit node has no existing rectangular routing access")


def materialize_mim_unit(master, *, site_roles, nets, tech, mim_spec, orient="R0",
                         quality_plate_site=None) -> MaterializedMimUnit:
    """Materialize one exact local unit against an explicitly supplied model.

    ``site_roles`` and ``nets`` are canonical exact tuples of pairs. Multiplicity,
    source parameter admission, banking and global placement belong to callers.
    The returned full site ledger must survive those callers' consumption paths.
    """
    _require(type(master) is UnitMasterGeometry, "requires an exact typed master")
    _require(type(orient) is str and orient in _ORIENTATIONS, "orientation must be D4")
    _require(type(mim_spec) is MimConnectSpec, "requires an explicit PDK MIM connect model")
    _require(master.pdk_id == getattr(tech, "pdk_id", None), "master and current technology PDK differ")
    _require(not master.unresolved_access_obligations, "master has unresolved access obligations")
    roles, net_map = _pairs(site_roles, "site roles"), _pairs(nets, "terminal nets")
    _require(tuple(roles) == tuple(site.site_id for site in master.terminal_sites), "site role denominator differs from master")
    _require(set(roles.values()) == {"p", "n"} and all(type(role) is str for role in roles.values()),
             "complete site roles must contain p and n")
    _require(set(net_map) == {"p", "n"} and all(type(net) is str and 0 < len(net) <= 4096
        for net in net_map.values()), "terminal nets require exact p/n names")
    validate_unit_master_regions(master)
    model_data, spec = _model_literal(mim_spec), deepcopy(mim_spec)
    artwork, transform, shift = _transform_regions(master, orient)
    polygons = _native_polygons(artwork)
    boxes = [region.bbox() for region in artwork.values()]
    extent = (0, 0, max(box.right for box in boxes), max(box.top for box in boxes))
    facts = CapacitorUnitOccurrenceGeometry.from_unit_extent(master.extent_nm)
    if quality_plate_site is not None:
        _require(type(quality_plate_site) is str, "quality plate site must be an exact identity")
        selected = tuple(site for site in master.terminal_sites if site.site_id == quality_plate_site)
        _require(len(selected) == 1 and selected[0].role_group == 'plate',
                 "quality plate site must identify an independently observed plate")
        plate = _regions(selected[0].geometry_by_layer_nm)
        _require(len(plate) == 1 and next(iter(plate)) in spec.metals,
                 "quality plate must occupy one actual electrical plate layer")
        region, = plate.values()
        box = region.bbox()
        _require(not region.is_empty() and (region ^ kdb.Region(box)).is_empty(),
                 "quality plate center requires one complete rectangular plate")
        facts = replace(facts, center_basis='materialized_unit_plate_bbox_center_floor_nm',
            unit_centers_nm=(((box.left+box.right)//2, (box.bottom+box.top)//2),))
    facts = facts.transformed(orient, shift, extent)
    geometry = DeviceGeom(polygons, {}, "cap", 1, 0, 0, extent,
                          orient=orient, capacitor_unit_occurrences=facts)
    nodes = extract_device_nodes(geometry, tech=tech, mim_spec=spec)
    _require(len(nodes) == 2, "artwork must extract exactly two local electrical nodes")
    routing = tuple(_routing_masks(node, artwork, tech, spec) for node in nodes)
    _require(all(routing), "every plate node requires existing real routing metal")
    accesses = tuple(MimSiteAccess(site.site_id, roles[site.site_id],
        index, _retained(routing[index])) for site in master.terminal_sites
        for index in (_site_node(site, transform, nodes),))
    role_nodes = {role: {row.node_id for row in accesses if row.role == role} for role in ("p", "n")}
    _require(all(len(ids) == 1 for ids in role_nodes.values()) and role_nodes["p"] != role_nodes["n"],
             "all sites of each role must bind the same distinct node")
    landings = {role: _terminal_landing(next(iter(ids)), routing) for role, ids in role_nodes.items()}
    terminals = {role: Terminal(role, box, net_map[role]) for role, (_layer, box) in landings.items()}
    geometry = replace(geometry, terminals=terminals, source_rail_bbox=terminals["p"].bbox,
                       drain_rail_bbox=terminals["n"].bbox)
    native, report = extract_terminal_access(geometry, tech=tech, mim_spec=spec)
    _require(not report.collapsed and not report.unmapped and set(native) == {"p", "n"}
        and all(native[role].node_id in role_nodes[role] for role in native),
        "native representative terminals do not replay their original site nodes")
    _require(_model_literal(mim_spec) == model_data == _model_literal(spec), "supplied connect model changed during materialization")
    return MaterializedMimUnit(master, geometry, accesses,
        tuple(sorted((role, layer) for role, (layer, _box) in landings.items())), model_data)


__all__ = ["MimSiteAccess", "MaterializedMimUnit", "materialize_mim_unit"]
