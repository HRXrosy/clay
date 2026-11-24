"""Same-materialization passive terminals; local node checks, not signoff.

The immutable value is minted with the native DeviceGeom before placement. It
is not a source-authority certificate or a JSON admission API. Current native
materialization/source/PDK authority remains the caller's responsibility.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from types import SimpleNamespace

from clay.layout.capacitor_unit_occurrences import CapacitorUnitOccurrenceGeometry
from clay.frontend.unit_master_geometry import UnitTerminalSite

_SEAL = object()


@dataclass(frozen=True, slots=True)
class MimMaterializedSite:
    """One independent original site in native bank coordinates, before placement."""
    unit_ordinal: int
    site_id: str
    role: str
    net: str
    orientation: str
    translation_nm: tuple[int, int]
    site_geometry_by_layer_nm: tuple
    routing_geometry_by_layer_nm: tuple

    def __post_init__(self):
        if type(self.unit_ordinal) is not int or self.unit_ordinal < 0:
            raise ValueError("MIM site requires a nonnegative exact unit ordinal")
        if (type(self.role) is not str or self.role not in ("p", "n")
                or type(self.net) is not str or not self.net or self.net != self.net.strip()
                or type(self.orientation) is not str or self.orientation not in
                ("R0", "R90", "R180", "R270", "MR0", "MR90", "MR180", "MR270")):
            raise ValueError("MIM site role/net/orientation differs")
        _point(self.translation_nm)
        for masks in (self.site_geometry_by_layer_nm, self.routing_geometry_by_layer_nm):
            UnitTerminalSite(self.site_id, self.role, masks)

    def canonical_data(self):
        self.__post_init__()
        def geometry(rows):
            return [[list(layer), [{"hull_nm": [list(point) for point in shape.hull_nm],
                "holes_nm": [[list(point) for point in ring] for ring in shape.holes_nm]} for shape in shapes]]
                for layer, shapes in rows]
        return {"unit_ordinal": self.unit_ordinal, "site_id": self.site_id, "role": self.role,
            "net": self.net, "orientation": self.orientation, "translation_nm": list(self.translation_nm),
            "site_geometry_by_layer_nm": geometry(self.site_geometry_by_layer_nm),
            "routing_geometry_by_layer_nm": geometry(self.routing_geometry_by_layer_nm)}

    def verify_nodes(self, nodes, transform=None):
        import klayout.db as kdb
        selected = None
        for masks in (self.site_geometry_by_layer_nm, self.routing_geometry_by_layer_nm):
            query = {}
            for layer, shapes in masks:
                region = kdb.Region()
                for shape in shapes:
                    if transform is None:
                        region += shape.to_region()
                    else:
                        from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM
                        region += PhysicalTerminalPolygonNM(tuple(transform(*point) for point in shape.hull_nm),
                            tuple(tuple(transform(*point) for point in ring) for ring in shape.holes_nm)).to_region()
                query[layer] = region
            matches = [index for index, node in enumerate(nodes)
                if all(layer in node and (region - node[layer]).is_empty() for layer, region in query.items())]
            if len(matches) != 1 or (selected is not None and selected != matches[0]):
                raise ValueError("MIM site/routing masks have no unique common native node")
            selected = matches[0]
            if any(index != selected and layer in node and not (region & node[layer]).is_empty()
                   for index, node in enumerate(nodes) for layer, region in query.items()):
                raise ValueError("MIM site intersects a foreign node")
        return selected


def _mim_ledger(rows, *, occurrences, terminals, nodes=None, transform=None):
    if type(rows) is not tuple or any(type(row) is not MimMaterializedSite for row in rows):
        raise TypeError("MIM site ledger requires immutable exact records")
    if not rows:
        return
    if type(occurrences) is not CapacitorUnitOccurrenceGeometry:
        raise ValueError("MIM site ledger lacks complete capacitor occurrences")
    ids = tuple((row.unit_ordinal, row.site_id) for row in rows)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("MIM site ledger contains duplicate or unordered identities")
    by_unit = {index: [] for index in range(occurrences.unit_count)}
    role_nodes = {}
    for row in rows:
        row.__post_init__()
        if (row.unit_ordinal not in by_unit or row.net != terminals.get(row.role)
                or row.orientation != occurrences.unit_transforms[row.unit_ordinal]):
            raise ValueError("MIM site ledger differs from native source/unit orientation")
        by_unit[row.unit_ordinal].append((row.site_id, row.role))
        if nodes is not None:
            node = row.verify_nodes(nodes, transform)
            if role_nodes.setdefault(row.role, node) != node:
                raise ValueError("MIM sites of one source role are disconnected")
    if (not all(value and {role for _site, role in value} == {"p", "n"} for value in by_unit.values())
            or any(value != by_unit[0] for value in by_unit.values())):
        raise ValueError("MIM complete unit/site denominator differs")
    if nodes is not None and len(set(role_nodes.values())) != 2:
        raise ValueError("MIM source roles do not remain two distinct nodes")


@dataclass(frozen=True, slots=True)
class MaterializedPassiveAccess:
    pdk_id: str
    kind: str
    model: str
    orient: str
    value: float
    multiplier: int
    # (role, net, routing layer, original terminal box, explicit access boxes).
    terminals: tuple
    # Native polygon vertices relative to the *artwork* lower-left, not a
    # reservation bbox. All records are immutable integer tuples.
    polygons: tuple
    # Series resistor bodies are not metal shorts: retain their real internal
    # metal nodes while requiring only the external p/n roles to be distinct.
    native_node_count: int
    native_metadata_json: bytes
    _seal: object = field(repr=False, compare=False)
    capacitor_unit_occurrences: CapacitorUnitOccurrenceGeometry | None = None
    mim_site_ledger: tuple[MimMaterializedSite, ...] = ()

    def __post_init__(self):
        if self._seal is not _SEAL:
            raise TypeError("passive access requires native materialization")
        if type(self.native_node_count) is not int or self.native_node_count < 2:
            raise ValueError("passive native terminal node inventory is incomplete")
        metadata = json.loads(self.native_metadata_json)
        if type(metadata) is not dict:
            raise ValueError("passive native metadata must be an object")
        stored = metadata.get("capacitor_unit_occurrences")
        facts = self.capacitor_unit_occurrences
        _mim_ledger(self.mim_site_ledger, occurrences=facts,
                    terminals={role: net for role, net, *_rest in self.terminals})
        if metadata.get("mim_site_ledger", []) != [row.canonical_data() for row in self.mim_site_ledger]:
            raise ValueError("MIM site ledger differs from native capture metadata")
        if facts is None and stored is None:
            return
        if type(facts) is not CapacitorUnitOccurrenceGeometry:
            raise TypeError("capacitor occurrence binding requires exact typed geometry")
        replayed = CapacitorUnitOccurrenceGeometry.from_data(stored)
        if replayed != facts:
            raise ValueError("capacitor occurrence binding differs from native metadata")
        if (self.kind != "cap" or type(self.multiplier) is not int
                or self.multiplier != facts.unit_count
                or self.orient != facts.bank_orientation):
            raise ValueError("capacitor occurrence binding differs from native kind, count, or orientation")

    def __deepcopy__(self, memo):
        return self


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _require(value):
    if type(value) is not MaterializedPassiveAccess or value._seal is not _SEAL:
        raise TypeError("passive access requires exact native materialization data")
    return value


def _box(value):
    if (type(value) is not tuple or len(value) != 4
            or any(type(v) is not int for v in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        raise ValueError("passive terminal box must be nonempty exact integer geometry")
    return value


def _point(value):
    if type(value) is not tuple or len(value) != 2 or any(type(v) is not int for v in value):
        raise ValueError("passive terminal access must be an exact integer point")
    return value


def _center(box):
    return ((box[0]+box[2])//2, (box[1]+box[3])//2)


def _shift_box(box, dx, dy):
    return (box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy)


def _polygon_rows(polygons):
    rows = []
    for polygon in polygons:
        vertices = tuple((p.x, p.y) for p in polygon.polygon.vertices)
        if (type(polygon.layer) is not int or type(polygon.datatype) is not int
                or len(vertices) < 3
                or any(type(v) is not int for point in vertices for v in point)):
            raise ValueError("passive artwork must have exact integer native polygons")
        rows.append((polygon.layer, polygon.datatype, vertices))
    if not rows:
        raise ValueError("passive materialization has no artwork")
    return tuple(rows)


def _origin(rows):
    return (min(x for _, _, ring in rows for x, y in ring),
            min(y for _, _, ring in rows for x, y in ring))


def _local_rows(rows, dx, dy):
    return tuple((layer, datatype, tuple((x-dx, y-dy) for x, y in ring))
                 for layer, datatype, ring in rows)


def _spec(binding, tech, *, geometry=None, record=None):
    if binding.pdk_id != str(tech.pdk_id):
        raise ValueError("passive materialization PDK differs")
    if binding.kind == "res":
        return None
    from clay.layout.route.pins import _mim_spec_and_level_for_stamp, _mim_spec_for_passive

    mim_opt = (
        getattr(geometry, "mim_option", None)
        or (record or {}).get("mim_option")
        or getattr(binding, "mim_option", None)
    )
    metal_lvl = (
        getattr(geometry, "mim_metal_level", None)
        or (record or {}).get("metal_level")
        or getattr(binding, "metal_level", None)
    )
    pex_backend = (record or {}).get("pex_backend") or getattr(binding, "pex_backend", None)
    if mim_opt is None and binding.pdk_id.lower() == "gf180mcu":
        stamp_obj = geometry
        if stamp_obj is None and getattr(binding, "polygons", None):
            drawn = {(int(layer), int(dt)): [()] for layer, dt, _ring in binding.polygons}
            stamp_obj = SimpleNamespace(polys_by_layer=drawn)
        if stamp_obj is not None:
            spec_from_stamp, _ = _mim_spec_and_level_for_stamp(tech, stamp_obj)
            if spec_from_stamp is not None:
                return spec_from_stamp
    result = _mim_spec_for_passive(
        tech,
        SimpleNamespace(
            model=binding.model,
            mim_option=mim_opt,
            metal_level=metal_lvl,
            pex_backend=pex_backend,
        ),
    )
    if result is None:
        raise ValueError("passive capacitor lacks a current PDK plate connect model")
    return result


def _node_for_box(nodes, layer, box):
    """Whole-box ownership on the explicit layer, never ANY-layer proximity."""
    import klayout.db as kdb
    region = kdb.Region(kdb.Box(*box))
    matches = [i for i, node in enumerate(nodes)
               if layer in node and (region-node[layer]).is_empty()]
    if len(matches) != 1:
        raise ValueError("passive terminal/access has no unique complete electrical node")
    selected = matches[0]
    if any(i != selected and layer in node and not (region & node[layer]).is_empty()
           for i, node in enumerate(nodes)):
        raise ValueError("passive terminal/access intersects a foreign electrical node")
    return selected


def _probe(point):
    return (point[0], point[1], point[0]+1, point[1]+1)


def _routing_layers(tech):
    layers = set()
    for level in range(1, 9):
        try:
            layers.add(tuple(tech.metal_key(level)))
        except Exception:
            continue
    return layers


def _capture_materialized_passive_access(geometry, *, record, terminal_layers, tech,
                                        terminal_accesses=None, mim_site_ledger=()):
    """Called by the native materializer on the geometry it actually consumed."""
    from clay.layout.device_terminal import extract_device_nodes
    kind = str(record.get("kind", "res"))
    if kind not in {"cap", "res"} or geometry.kind != kind:
        raise ValueError("passive materialization kind differs")
    if set(geometry.terminals) != {"p", "n"} or set(terminal_layers) != {"p", "n"}:
        raise ValueError("passive materialization requires both original p/n terminals")
    rows = _polygon_rows(geometry.polygons)
    ox, oy = _origin(rows)
    metadata = {key: getattr(geometry, key, None) for key in (
        "parallel_mim_bank", "bank_interconnect", "fixed_pcell")}
    occurrences = getattr(geometry, "capacitor_unit_occurrences", None)
    if occurrences is not None:
        if type(occurrences) is not CapacitorUnitOccurrenceGeometry:
            raise TypeError("capacitor occurrence materialization requires exact typed geometry")
        if (kind != "cap" or type(record.get("M", 1)) is not int
                or occurrences.unit_count != record.get("M", 1)
                or type(geometry.finger_count) is not int
                or occurrences.unit_count != geometry.finger_count
                or occurrences.bank_extent_nm != geometry.extent()
                or occurrences.bank_orientation != str(geometry.orient)
                or occurrences.bank_orientation != str(record.get("orient", "R0") or "R0")):
            raise ValueError("capacitor occurrence facts differ from native materialization")
    metadata["capacitor_unit_occurrences"] = (
        occurrences.canonical_data() if occurrences is not None else None)
    _mim_ledger(mim_site_ledger, occurrences=occurrences, terminals=record["pins"])
    if mim_site_ledger:
        metadata["mim_site_ledger"] = [row.canonical_data() for row in mim_site_ledger]
    plan = record.get("bank_interconnect_plan")
    if plan is not None:
        from clay.layout.bank_interconnect import replay_bank_interconnect_plan
        plan = replay_bank_interconnect_plan(plan).canonical_data()
    actual_plan = (metadata["bank_interconnect"] or {}).get("plan")
    if plan != actual_plan:
        raise ValueError("passive bank interconnect differs from native source plan")
    ports = []
    metal_layers = _routing_layers(tech)
    for role in ("p", "n"):
        terminal = geometry.terminals[role]
        if terminal.role != role or terminal.net != record["pins"].get(role):
            raise ValueError("passive terminal role/net differs from native source")
        layer = tuple(terminal_layers[role])
        if layer not in metal_layers:
            raise ValueError("passive terminal layer is not a routing metal")
        box = _box(terminal.bbox)
        accesses = []
        explicit_accesses = getattr(geometry, "terminal_accesses", {}) if terminal_accesses is None else terminal_accesses
        for access in (explicit_accesses or {}).get(role, ()):
            if access.role != role or access.net != terminal.net:
                raise ValueError("passive access role/net differs from original terminal")
            accesses.append(_shift_box(_box(access.bbox), -ox, -oy))
        ports.append((role, terminal.net, layer, _shift_box(box, -ox, -oy), tuple(accesses)))
    nodes = extract_device_nodes(
        geometry,
        tech=tech,
        mim_spec=_spec(
            SimpleNamespace(
                pdk_id=str(tech.pdk_id),
                kind=kind,
                model=str(record.get("model", "") or ""),
            ),
            tech,
            geometry=geometry,
            record=record,
        ),
    )
    _mim_ledger(mim_site_ledger, occurrences=occurrences, terminals=record["pins"], nodes=nodes)
    binding = MaterializedPassiveAccess(
        pdk_id=str(tech.pdk_id), kind=kind, model=str(record.get("model", "") or ""),
        orient=str(record.get("orient", "R0") or "R0"), value=float(record["value"]),
        multiplier=int(record.get("M", 1)), terminals=tuple(ports),
        polygons=_local_rows(rows, ox, oy), native_node_count=len(nodes),
        native_metadata_json=_json(metadata), _seal=_SEAL,
        capacitor_unit_occurrences=occurrences, mim_site_ledger=mim_site_ledger)
    if str(geometry.orient) != binding.orient:
        raise ValueError("passive native materialization orientation differs")
    role_nodes = []
    for role, net, layer, box, accesses in binding.terminals:
        original_node = _node_for_box(nodes, layer, _shift_box(box, ox, oy))
        role_nodes.append(original_node)
        for access in accesses:
            if _node_for_box(nodes, layer, _shift_box(access, ox, oy)) != original_node:
                raise ValueError("passive access is on the other original terminal node")
    if len(set(role_nodes)) != 2:
        raise ValueError("passive materialization must expose two distinct terminal nodes")
    return binding


def _check_source(binding, passive):
    expected = (binding.kind, binding.model, binding.orient, binding.value, binding.multiplier,
                {role: net for role, net, _, _, _ in binding.terminals})
    actual = (str(passive.kind), str(passive.model or ""), str(passive.orient),
              float(passive.value), int(passive.M), dict(passive.nets))
    if actual != expected:
        raise ValueError("passive source identity/terminal net differs from native materialization")


def bind_placed_passive_access(stamp, placed, *, passive, center_nm, tech):
    """Attach original role facts only after exact translated-artwork replay."""
    binding = _require(getattr(placed, "materialized_passive_access", None))
    _check_source(binding, passive)
    if placed.kind != binding.kind or placed.label != passive.name:
        raise ValueError("passive placed identity differs")
    native_metadata = json.loads(binding.native_metadata_json)
    if _json(placed.fixed_pcell) != _json(native_metadata["fixed_pcell"] or {}):
        raise ValueError("passive fixed materialization identity differs")
    rows = _polygon_rows(placed.polys)
    ox, oy = _origin(rows)
    if Counter(_local_rows(rows, ox, oy)) != Counter(binding.polygons):
        raise ValueError("passive placed artwork differs from retained native materialization")
    expected_terms = {r: (x+ox, y+oy) for r, _, _, box, _ in binding.terminals
                      for x, y in (_center(box),)}
    expected_layers = {r: layer for r, _, layer, _, _ in binding.terminals}
    expected_accesses = {r: tuple((x+ox, y+oy) for box in accesses for x, y in (_center(box),))
                         for r, _, _, _, accesses in binding.terminals if accesses}
    if ({r: _point(p) for r, p in placed.terminals.items()} != expected_terms
            or placed.term_layers != expected_layers
            or placed.terminal_accesses != expected_accesses):
        raise ValueError("passive placed terminal/access metadata differs from original native roles")
    _spec(binding, tech)
    stamp.materialized_passive_access = binding
    stamp.passive_artwork_origin_um = ((ox-center_nm[0])/1000, (oy-center_nm[1])/1000)


def _stamp_rows(stamp, position):
    from clay.layout.placed_geometry import placed_point_nm
    return tuple((layer, datatype, tuple(placed_point_nm(x, y, position) for x, y in ring))
                 for (layer, datatype), rings in stamp.polys_by_layer.items() for ring in rings)


def verified_passive_route_accesses(stamp, *, passive, position, tech):
    """Return verified final integer access points without rebuilding a cell.

    Replays the exact original artwork through the same grid normalizer, then
    extracts actual current nodes. Returned tuples contain role, net, real layer,
    final integer point and the independent role-node identity for final checking.
    """
    from clay.layout.device_terminal import extract_cellgeom_nodes
    from clay.layout.placed_geometry import placed_point_nm
    from clay.layout.route.route_cell import _normalize_materialized_geometry_to_manufacturing_grid
    from clay.layout.route.pins import _manufacturing_grid_access_on_stamp
    binding = _require(stamp.materialized_passive_access)
    _check_source(binding, passive)
    ox, oy = stamp.passive_artwork_origin_um
    by_layer = {}
    for layer, datatype, ring in binding.polygons:
        by_layer.setdefault((layer, datatype), []).append([(x/1000+ox, y/1000+oy) for x, y in ring])
    expected = SimpleNamespace(polys_by_layer=by_layer)
    _normalize_materialized_geometry_to_manufacturing_grid({passive.name: expected},
                                                          {passive.name: position}, tech)
    actual_rows = _stamp_rows(stamp, position)
    if Counter(actual_rows) != Counter(_stamp_rows(expected, position)):
        raise ValueError("passive current artwork differs from normalized native materialization")
    # Use exact absolute emit coordinates; local half-nm centroid residue must
    # not be rounded before adding the placement.
    absolute_layers = {}
    for layer, datatype, ring in actual_rows:
        absolute_layers.setdefault((layer, datatype), []).append([(x/1000, y/1000) for x, y in ring])
    nodes = extract_cellgeom_nodes(SimpleNamespace(polys_by_layer=absolute_layers),
                                  tech=tech, mim_spec=_spec(binding, tech))
    if binding.mim_site_ledger:
        native_x, native_y = binding.capacitor_unit_occurrences.bank_extent_nm[:2]
        _mim_ledger(binding.mim_site_ledger, occurrences=binding.capacitor_unit_occurrences,
            terminals=dict(passive.nets), nodes=nodes,
            transform=lambda x, y: placed_point_nm((x-native_x)/1000+ox, (y-native_y)/1000+oy, position))
    role_nodes, result = [], []
    expected_offsets, expected_accesses, expected_layers = {}, {}, {}
    for role, net, layer, box, accesses in binding.terminals:
        x, y = _center(box)
        original_point = placed_point_nm(x/1000+ox, y/1000+oy, position)
        node = _node_for_box(nodes, layer, _probe(original_point))
        original_point = _manufacturing_grid_access_on_stamp(stamp, position, layer, original_point, tech)
        if _node_for_box(nodes, layer, _probe(original_point)) != node:
            raise ValueError("passive grid selection moved the original terminal onto another node")
        role_nodes.append(node)
        expected_offsets[role] = placed_point_nm(x/1000+ox, y/1000+oy, position)
        expected_layers[role] = layer
        points = tuple(placed_point_nm(x/1000+ox, y/1000+oy, position)
                       for entry in accesses for x, y in (_center(entry),))
        if accesses:
            expected_accesses[role] = points
        for index, point in enumerate(points or (expected_offsets[role],)):
            final_point = _manufacturing_grid_access_on_stamp(stamp, position, layer, point, tech)
            if _node_for_box(nodes, layer, _probe(final_point)) != node:
                raise ValueError("passive final access is on the other terminal node")
            result.append((role, net, layer, final_point, node, index))
    observed_offsets = {r: placed_point_nm(*point, position) for r, point in stamp.terminal_offsets_um.items()}
    observed_accesses = {r: tuple(placed_point_nm(*point, position) for point in points)
                         for r, points in stamp.terminal_accesses_um.items()}
    if (observed_offsets != expected_offsets or observed_accesses != expected_accesses
            or stamp.terminal_layers != expected_layers):
        raise ValueError("passive stamp terminal/access metadata differs from retained native roles")
    if len(nodes) != binding.native_node_count or len(set(role_nodes)) != 2:
        raise ValueError("passive current artwork changed its node inventory or distinct terminal nodes")
    return tuple(result), nodes


def verify_emitted_passive_access(pin, *, expected, nodes):
    """Check the actual Pin after the shared _emit/grid-access operation."""
    role, net, layer, point, node, index = expected
    if (pin.terminal != role or pin.net != net or tuple(pin.layer) != layer
            or tuple(pin.xy_nm) != point
            or _node_for_box(nodes, layer, _probe(tuple(pin.xy_nm))) != node):
        raise ValueError("passive final emitted terminal differs from verified node/layer/access")
