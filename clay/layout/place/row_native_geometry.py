"""Bind row projection to the original native collision proof and actual artwork.

This is geometry proof data, not an execution or production-authority mint.  The
original typed certificate remains attached; the enclosing live placement
session must still authorize production consumption.  The supported scope is
the complete materialized native device inventory supplied to the factory.
Callers must pass that same explicit scope when replaying later route geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import SimpleNamespace
from typing import Mapping

from clay.layout.extract.compaction import Full2DCompactionGraph, Full2DPairConstraint
from clay.layout.extract.realize_render import _placed_stamp_layer_rectangles_nm
from clay.layout.physical_macro_contracts import _canonical_ring
from clay.layout.place.final_placement_state import final_placement_state_digest
from clay.layout.place.actual_materialized_partition import actual_materialized_partition
from clay.layout.placed_geometry import placed_point_nm
from clay.layout.place_blocks_2d import (
    FinalPlacementCertificate,
    _source_device_bboxes_nm,
    placement_certificate_is_signable,
)


_Box = tuple[int, int, int, int]
_Layer = tuple[int, int]
_Ring = tuple[tuple[int, int], ...]
_MAX_COORDINATE_NM = 10**12


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _names(value: object, field: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if (type(value) is not tuple or (nonempty and not value)
            or any(type(item) is not str or not item for item in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{field} must contain unique exact names")
    return tuple(sorted(value))


def _layer(value: object) -> _Layer:
    if (type(value) is not tuple or len(value) != 2
            or any(type(item) is not int or not 0 <= item <= 65535 for item in value)):
        raise ValueError("actual artwork layer must be an exact physical layer key")
    return value


def _point(value: object) -> tuple[int, int]:
    if (type(value) is not tuple or len(value) != 2
            or any(type(item) is not int or abs(item) > _MAX_COORDINATE_NM for item in value)):
        raise ValueError("actual artwork point must be bounded exact integer nm")
    return value


def _box(value: object) -> _Box:
    if (type(value) is not tuple or len(value) != 4
            or any(type(item) is not int or abs(item) > _MAX_COORDINATE_NM for item in value)
            or value[2] <= value[0] or value[3] <= value[1]):
        raise ValueError("native geometry bbox must be a positive exact integer rectangle")
    return value


def _envelope(boxes: tuple[_Box, ...]) -> _Box:
    if not boxes:
        raise ValueError("actual stamp envelope must not be empty")
    return (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))


def _inside(inner: _Box, outer: _Box) -> bool:
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


@dataclass(frozen=True, slots=True)
class _Stamp:
    name: str
    source: str
    members: tuple[str, ...]
    guard_sha256: str
    guard_members: tuple[str, ...]
    layers: tuple[tuple[_Layer, tuple[_Ring, ...]], ...]
    tap_child: object = None

    def payload(self) -> tuple:
        original = (self.name, self.source, self.members, self.guard_sha256, self.guard_members, self.layers)
        return original if self.tap_child is None else (*original, self.tap_child.payload())


@dataclass(frozen=True, slots=True)
class _Snapshot:
    stamps: tuple[_Stamp, ...]
    guards: tuple[tuple[tuple[str, ...], str], ...]

    def payload(self) -> tuple:
        return (tuple(stamp.payload() for stamp in self.stamps), self.guards)


def _capture(geometry: Mapping, placement_um: Mapping, intent: object) -> _Snapshot:
    if not isinstance(geometry, Mapping) or not geometry or not isinstance(placement_um, Mapping):
        raise TypeError("actual route geometry and placement must be nonempty mappings")
    if any(type(name) is not str or not name for name in geometry):
        raise ValueError("actual stamp inventory must use exact names")
    stamps = []
    for name in sorted(geometry):
        stamp = geometry[name]
        position = placement_um.get(name)
        if (type(position) is not tuple or len(position) != 2
                or any(type(item) not in {int, float} or not math.isfinite(item) for item in position)):
            raise ValueError(f"actual stamp {name!r} lacks finite placed coordinates")
        source = getattr(stamp, "source", "")
        guard_sha = getattr(stamp, "guard_ring_intent_sha256", "")
        if type(source) is not str or type(guard_sha) is not str:
            raise TypeError("actual stamp source provenance must be exact strings")
        members = _names(getattr(stamp, "composite_members", ()), "actual composite members")
        guard_members = _names(getattr(stamp, "guard_ring_declared_members", ()), "actual guard members")
        child = getattr(stamp, 'layered_placement_tap_child', None)
        if child is not None:
            from clay.layout.layered_placement_assembly import LayeredTapChild
            if type(child) is not LayeredTapChild:
                raise TypeError('actual tap marker requires exact typed ownership')
            child.__post_init__()
            if child.stamp_name != name or members:
                raise ValueError('actual tap marker identity or logical members differ')
            child = LayeredTapChild(child.stamp_name, child.construction, child.local_offset_nm)
        raw_layers = getattr(stamp, "polys_by_layer", None)
        if type(raw_layers) is not dict or not raw_layers:
            raise ValueError(f"actual stamp {name!r} has no polygon layers")
        layers = []
        for key, raw_rings in raw_layers.items():
            key = _layer(key)
            if type(raw_rings) is not list or not raw_rings:
                raise ValueError("actual stamp layers require nonempty polygon lists")
            rings = []
            for ring in raw_rings:
                if type(ring) not in {tuple, list} or len(ring) < 3:
                    raise ValueError("actual stamp polygon is degenerate")
                points = []
                for point in ring:
                    if (type(point) not in {tuple, list} or len(point) != 2
                            or any(type(item) not in {int, float} or not math.isfinite(item) for item in point)):
                        raise ValueError("actual stamp polygon requires finite coordinate pairs")
                    points.append(_point(placed_point_nm(point[0], point[1], position)))
                rings.append(_canonical_ring(tuple(points)))
            layers.append((key, tuple(sorted(rings))))  # Deliberately preserve duplicate polygons.
        stamps.append(_Stamp(name, source, members, guard_sha, guard_members, tuple(sorted(layers)), child))
    raw_guards = getattr(intent, "guard_rings", ())
    if type(raw_guards) is not tuple:
        raise TypeError("actual guard declaration inventory must be an exact tuple")
    guards = []
    for guard in raw_guards:
        members = _names(getattr(guard, "members", ()), "guard declaration members", nonempty=True)
        digest = getattr(guard, "sha256", "")
        if type(digest) is not str:
            raise TypeError("guard declaration digest must be an exact string")
        guards.append((members, digest))
    return _Snapshot(tuple(stamps), tuple(guards))


def _thaw(snapshot: _Snapshot) -> tuple[dict, dict, object]:
    """Feed the shared extractor an exact reconstruction of placed-nm polygons."""
    geometry = {}
    for stamp in snapshot.stamps:
        layers = {}
        for key, rings in stamp.layers:
            converted = []
            for ring in rings:
                points = [(x / 1000.0, y / 1000.0) for x, y in ring]
                if tuple(placed_point_nm(x, y, (0.0, 0.0)) for x, y in points) != ring:
                    raise ValueError("actual nm snapshot does not round-trip the shared extractor")
                converted.append(points)
            layers[key] = converted
        geometry[stamp.name] = SimpleNamespace(
            source=stamp.source, composite_members=stamp.members,
            guard_ring_intent_sha256=stamp.guard_sha256,
            guard_ring_declared_members=stamp.guard_members, polys_by_layer=layers,
            layered_placement_tap_child=stamp.tap_child)
    placement = {name: (0.0, 0.0) for name in geometry}
    intent = SimpleNamespace(guard_rings=tuple(
        SimpleNamespace(members=members, sha256=digest) for members, digest in snapshot.guards))
    return geometry, placement, intent


@dataclass(frozen=True, slots=True)
class _PhysicalBlock:
    actual_id: str
    native_id: str
    master: str
    bbox_nm: _Box
    device_stamps: tuple[str, ...]
    all_stamps: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Inventory:
    blocks: tuple[_PhysicalBlock, ...]
    stamp_rectangles: tuple[tuple[str, tuple[tuple[_Layer, tuple[_Box, ...]], ...]], ...]
    stamp_bboxes: tuple[tuple[str, _Box], ...]
    source_bboxes: tuple[tuple[str, _Box], ...]
    shared_guards: tuple = ()


def _require_source_bboxes(source_bboxes: object, inventory: _Inventory) -> tuple[str, ...]:
    if type(source_bboxes) is not tuple:
        raise TypeError("source bbox inventory must be an exact tuple")
    authority = dict(inventory.source_bboxes)
    names = []
    for row in source_bboxes:
        if type(row) is not tuple or len(row) != 2 or type(row[0]) is not str:
            raise TypeError("source bbox inventory requires exact named pairs")
        name, box = row
        if _box(box) != authority.get(name):
            raise ValueError("source bbox differs from actual native source geometry")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("source bbox inventory repeats a native source owner")
    return tuple(names)


def _validate_native(certificate: FinalPlacementCertificate) -> str:
    if type(certificate) is not FinalPlacementCertificate:
        raise TypeError("row geometry requires the exact original FinalPlacementCertificate")
    graph = certificate.full2d_compaction_graph
    from clay.layout.extract.fixed_incumbent_geometry import FixedIncumbentGeometryProof
    if type(graph) not in (Full2DCompactionGraph, FixedIncumbentGeometryProof):
        raise TypeError("row geometry requires the original typed full-2D collision graph")
    # Graph construction checks complete pairs, but does not re-run nested row
    # constructors.  Both checks are needed for pre-existing typed objects.
    for pair in graph.pair_constraints if type(graph) is Full2DCompactionGraph else ():
        if type(pair) is not Full2DPairConstraint:
            raise TypeError("native collision graph contains a foreign pair type")
        pair.__post_init__()
    graph.__post_init__()
    certificate.__post_init__()
    if not placement_certificate_is_signable(certificate):
        raise ValueError("retained native certificate does not satisfy the placement-to-route contract")
    return final_placement_state_digest(certificate)


def _derive(certificate: FinalPlacementCertificate, snapshot: _Snapshot) -> _Inventory:
    import klayout.db as kdb

    geometry, placement, intent = _thaw(snapshot)
    rectangles = _placed_stamp_layer_rectangles_nm(geometry, placement)
    if set(rectangles) != set(geometry):
        raise ValueError("actual geometry lacks complete nonempty stamp rectangle coverage")
    stamp_bboxes = {name: _envelope(tuple(box for boxes in layers.values() for box in boxes))
                    for name, layers in rectangles.items()}
    for stamp in snapshot.stamps:
        if set(rectangles[stamp.name]) != {key for key, _rings in stamp.layers}:
            raise ValueError("actual geometry contains an unrepresented polygon layer")
    native_boxes = dict(certificate.materialized_block_bboxes)
    native_sources = dict(certificate.materialized_block_source_members)
    master_boxes = dict(certificate.blocks)
    from clay.layout.place.physical_master_frames import physical_master_boxes
    physical_masters = physical_master_boxes(master_boxes, certificate.flexible_envelope_projections)
    members = certificate.materialized_block_members
    master_by_physical = {physical: master for master, physicals in members for physical in physicals}
    if (not native_boxes or set(native_boxes) != set(native_sources)
            or set(native_boxes) != set(master_by_physical)
            or {master for master, _physicals in members} != set(master_boxes)
            or sum(len(physicals) for _master, physicals in members) != len(master_by_physical)):
        raise ValueError("native materialized ownership must completely partition original masters")
    for name, box in native_boxes.items():
        if not _inside(_box(box), _box(physical_masters[master_by_physical[name]])):
            raise ValueError("native physical block lies outside its certified master")
    all_sources = tuple(sorted(source for sources in native_sources.values() for source in sources))
    if not all_sources or len(set(all_sources)) != len(all_sources):
        raise ValueError("native source ownership must be complete and unique")
    source_bboxes = _source_device_bboxes_nm(geometry, placement, all_sources)
    if tuple(sorted(source_bboxes.items())) != certificate.materialized_source_member_bboxes:
        raise ValueError("actual source bboxes differ from complete native source bbox authority")
    partition = actual_materialized_partition(certificate, geometry, placement, intent)
    actual_blocks, block_stamps, block_guard, guard_names = (
        partition.blocks, partition.block_stamps, partition.block_gring, partition.gring_names)
    source_members = dict(partition.source_members)
    native_by_sources = {sources: name for name, sources in native_sources.items()}
    from clay.layout.place.shared_guard_native_artwork import shared_guard_partition
    shared_guards, shared_replacements = shared_guard_partition(
        certificate, snapshot, partition, stamp_bboxes)
    replaced_masters = {master for group in shared_guards for master in group.masters}
    blocks = []
    covered_stamps = []
    for actual_id, actual_box in actual_blocks:
        device_stamps = tuple(sorted(set(block_stamps[actual_id]) - partition.tap_names))
        stamps = tuple(sorted(tuple(block_stamps[actual_id]) +
                              ((block_guard[actual_id],) if actual_id in block_guard else ())))
        covered_stamps.extend(stamps)
        sources = source_members[actual_id]
        if actual_id in shared_replacements:
            for native_id, owned_stamps in shared_replacements[actual_id]:
                device_box = _envelope(tuple(stamp_bboxes[name] for name in owned_stamps))
                blocks.append(_PhysicalBlock(actual_id, native_id, master_by_physical[native_id],
                    device_box, owned_stamps, owned_stamps, native_sources[native_id]))
            continue
        native_id = native_by_sources.get(sources)
        if native_id is None or len(sources) != len(set(sources)):
            raise ValueError("actual physical source ownership differs from native materialized ownership")
        if actual_box != native_boxes[native_id]:
            raise ValueError(
                "actual physical bbox differs from its native materialized block: "
                f"actual_id={actual_id!r}, native_id={native_id!r}, sources={sources!r}, "
                f"actual_bbox_nm={actual_box!r}, native_bbox_nm={native_boxes[native_id]!r}, "
                f"stamps={tuple((name, stamp_bboxes[name]) for name in stamps)!r}"
            )
        if actual_box != _envelope(tuple(stamp_bboxes[name] for name in stamps)):
            raise ValueError("actual extracted block bbox differs from exact stamp rectangles")
        blocks.append(_PhysicalBlock(actual_id, native_id, master_by_physical[native_id],
                                     actual_box, device_stamps, stamps, sources))
    if (len(covered_stamps) != len(set(covered_stamps)) or set(covered_stamps) != set(geometry)
            or len(blocks) != len(native_boxes) or {block.native_id for block in blocks} != set(native_boxes)
            or not guard_names <= set(covered_stamps)):
        raise ValueError("actual stamps and native physical owners require exact complete coverage")

    graph = certificate.full2d_compaction_graph
    excluded_layers = set(certificate.collision_excluded_layers)
    sizes = dict(graph.sizes_nm)
    node_layers = dict(graph.node_layers)
    from clay._native import load_native as _load_native_eng
    _eng_cpp = _load_native_eng("_engine_cpp")
    _use_cpp_cover = _eng_cpp is not None and hasattr(_eng_cpp, "boxes_covered_by_union_cpp")
    authority_boxes: dict = {}
    regions = {}
    certified_layers = {master: set() for master in master_boxes}
    for node, master, (dx, dy) in graph.node_owner_offsets_nm:
        layer = node_layers[node]
        certified_layers[master].add(layer)
        x0, y0 = master_boxes[master][:2]
        width, height = sizes[node]
        if _use_cpp_cover:
            authority_boxes.setdefault((master, layer), []).append(
                (x0 + dx, y0 + dy, x0 + dx + width, y0 + dy + height)
            )
        else:
            region = regions.setdefault((master, layer), kdb.Region())
            region.insert(kdb.Box(x0 + dx, y0 + dy, x0 + dx + width, y0 + dy + height))
    if not _use_cpp_cover:
        for region in regions.values():
            region.merge()
    actual_layers = {master: set() for master in master_boxes}
    for block in blocks:
        for stamp in block.all_stamps:
            actual_layers[block.master].update(set(rectangles[stamp]) - excluded_layers)
    for master, layers in certified_layers.items():
        if None in layers:
            if layers != {None}:
                raise ValueError("mixed wildcard and layered native owners are unsupported")
        elif master in replaced_masters:
            if not actual_layers[master] <= layers:
                raise ValueError("actual shared device layer inventory exceeds native collision layers")
        elif layers != actual_layers[master]:
            raise ValueError("actual owner layer inventory differs from certified collision layers: "
                             f"owner={master!r}, actual={tuple(sorted(actual_layers[master]))!r}, "
                             f"certified={tuple(sorted(layers))!r}, excluded={tuple(sorted(excluded_layers))!r}")
    for group in shared_guards:
        certified = set().union(*(certified_layers[master] for master in group.masters))
        actual = set().union(*(actual_layers[master] for master in group.masters))
        actual.update(set(rectangles[group.guard_stamp]) - excluded_layers)
        if certified != {None} and certified != actual:
            raise ValueError("actual joint guard and device layer inventory differs from native group")
    from .native_feol_reservation import sky130_group_spacing_master_pairs
    rail_ir = certificate.edge_rail_reservation
    group_pairs = set(sky130_group_spacing_master_pairs(rail_ir))
    stamp_owner = {stamp: block.master for block in blocks for stamp in block.device_stamps}
    for first, second in graph.overlap_compatible_pairs:
        if not actual_layers[first] & actual_layers[second]:
            continue
        pair = tuple(sorted((first, second)))
        orders = certificate.hierarchical_macro_orders
        ordered = any(row[0].startswith('authored_group_separate:') and
                      ((first in row[5] and second in row[6]) or
                       (second in row[5] and first in row[6])) for row in orders)
        if pair not in group_pairs or not ordered:
            raise ValueError("native compatible pair lacks actual disjoint layer evidence")
        # The complete four-pair FEOL proof belongs to these actual source
        # stamps, not merely to two masters that happen to share layer names.
        # _validate_native already replayed the original IR and source order.
        from .edge_rail_geometry import _stamp_geometry, _shift
        envelope_rows = [row for row in rail_ir.native_feol_pairs
                         if row.rule_id == 'sky130:group-row-envelope'
                         and tuple(sorted((row.left_master, row.right_master))) == pair]
        stamps = {stamp.stamp_id: stamp for stamp in rail_ir.stamps}
        for row in envelope_rows:
            for name, master in ((row.left_stamp, row.left_master),
                                 (row.right_stamp, row.right_master)):
                original = stamps.get(name)
                if original is None or stamp_owner.get(name) != master or original.master_id != master:
                    raise ValueError("native group spacing stamp ownership differs")
                box, digest = _stamp_geometry(geometry[name], placement[name])
                if box != _shift(original.bbox_nm, master_boxes[master][:2]) or digest != original.geometry_sha256:
                    raise ValueError("native group spacing actual stamp geometry differs")
    for block in blocks:
        for stamp in block.all_stamps:
            for layer, boxes in rectangles[stamp].items():
                if layer in excluded_layers:
                    continue
                authority_layer = None if certified_layers[block.master] == {None} else layer
                if _use_cpp_cover:
                    auth_list = authority_boxes.get((block.master, authority_layer))
                    if not auth_list or not _eng_cpp.boxes_covered_by_union_cpp(boxes, auth_list):
                        raise ValueError("actual layer rectangles lack union coverage by native collision atoms")
                else:
                    authority = regions.get((block.master, authority_layer))
                    actual = kdb.Region()
                    for box in boxes:
                        actual.insert(kdb.Box(*box))
                    if authority is None or not (actual - authority).is_empty():
                        raise ValueError("actual layer rectangles lack union coverage by native collision atoms")
    return _Inventory(
        tuple(sorted(blocks, key=lambda block: block.native_id)),
        tuple((name, tuple(sorted(layers.items()))) for name, layers in sorted(rectangles.items())),
        tuple(sorted(stamp_bboxes.items())), tuple(sorted(source_bboxes.items())), shared_guards)


@dataclass(frozen=True, slots=True, init=False)
class RowNativeGeometryBinding:
    """Immutable actual-artwork binding under an enclosing live native session."""

    certificate: FinalPlacementCertificate
    native_state_sha256: str
    actual_geometry_sha256: str
    _snapshot: _Snapshot
    _inventory: _Inventory

    def __init__(self, *args, **kwargs):
        raise TypeError("use RowNativeGeometryBinding.from_route_geometry with actual geometry")

    @classmethod
    def from_route_geometry(cls, *, certificate, geometry, placement_um, intent) -> RowNativeGeometryBinding:
        if cls is not RowNativeGeometryBinding:
            raise TypeError("row native geometry binding does not support foreign subclasses")
        native_sha = _validate_native(certificate)
        snapshot = _capture(geometry, placement_um, intent)
        inventory = _derive(certificate, snapshot)
        binding = object.__new__(cls)
        for name, value in (("certificate", certificate), ("native_state_sha256", native_sha),
                            ("actual_geometry_sha256", _sha(snapshot.payload())),
                            ("_snapshot", snapshot), ("_inventory", inventory)):
            object.__setattr__(binding, name, value)
        return binding

    def _replay(self) -> _Inventory:
        if type(self) is not RowNativeGeometryBinding or type(self.certificate) is not FinalPlacementCertificate:
            raise TypeError("row native geometry proof has a foreign type")
        if _validate_native(self.certificate) != self.native_state_sha256:
            raise ValueError("retained native certificate state differs from the actual geometry binding")
        if type(self._snapshot) is not _Snapshot or _sha(self._snapshot.payload()) != self.actual_geometry_sha256:
            raise ValueError("retained actual geometry snapshot digest differs")
        inventory = _derive(self.certificate, self._snapshot)
        if inventory != self._inventory:
            raise ValueError("retained actual geometry inventory does not replay its raw polygons")
        return inventory

    def require_actual_route_geometry(self, *, geometry, placement_um, intent) -> None:
        self._replay()
        if _capture(geometry, placement_um, intent) != self._snapshot:
            raise ValueError("actual route geometry differs from the retained placed polygon snapshot")

    @property
    def stamp_names(self) -> tuple[str, ...]:
        """Original emitted device scope; this does not authorize skipping extras."""
        return tuple(stamp.name for stamp in self._snapshot.stamps)

    def replayed_stamp_artwork_nm(self) -> tuple[
        tuple[str, tuple[str, ...], tuple[tuple[_Layer, tuple[_Ring, ...]], ...]], ...
    ]:
        """Return complete immutable placed stamp masks after native proof replay.

        A composite retains its complete source-member tuple; this accessor does
        not assign shared artwork to individual members. Polygon multiplicity is
        preserved. Consumers must still join this scope to actual route geometry
        and the enclosing placement session before claiming production evidence.
        """
        self._replay()
        return tuple((stamp.name, () if stamp.tap_child is not None else stamp.members or (stamp.name,), stamp.layers)
                     for stamp in self._snapshot.stamps)

    def require_source_bboxes(self, *, source_bboxes) -> None:
        """Join a selected source subset to exact actual native source bboxes."""
        _require_source_bboxes(source_bboxes, self._replay())

    def _rows(self, macro_bindings, inventory: _Inventory) -> tuple:
        from clay.layout.generic_placement import RowMacroBlockBinding

        if type(macro_bindings) is not tuple:
            raise TypeError("row macro bindings must be an exact tuple")
        physical_by_sources = {block.sources: block for block in inventory.blocks}
        stamp_boxes = dict(inventory.stamp_bboxes)
        rows = []
        owners = set()
        used_physical = set()
        for binding in macro_bindings:
            if type(binding) is not RowMacroBlockBinding:
                raise TypeError("row geometry requires exact RowMacroBlockBinding values")
            binding.__post_init__()
            block = physical_by_sources.get(binding.consumed_devices)
            if block is None or block.native_id in used_physical or binding.owner in owners:
                raise ValueError("row consumed devices require one unique complete native physical source owner")
            actual_bbox = _envelope(tuple(stamp_boxes[name] for name in block.device_stamps))
            if binding.bbox_nm != actual_bbox:
                raise ValueError("row bbox differs from the actual row stamp envelope")
            owners.add(binding.owner)
            used_physical.add(block.native_id)
            rows.append((binding, block))
        return tuple(rows)

    def require_row_blocks(self, *, macro_bindings, free_device_blocks) -> None:
        inventory = self._replay()
        rows = self._rows(macro_bindings, inventory)
        sources = dict(inventory.source_bboxes)
        consumed = [source for binding, _block in rows for source in binding.consumed_devices]
        free = _require_source_bboxes(free_device_blocks, inventory)
        covered = consumed + list(free)
        if len(covered) != len(set(covered)) or set(covered) != set(sources):
            raise ValueError("row consumed and free sources must exactly cover native source ownership")

    def require_row_artwork(self, *, macro_bindings, macro_polys_by_owner) -> None:
        from clay.layout.backend import LayerPolygon

        inventory = self._replay()
        rows = self._rows(macro_bindings, inventory)
        if (not isinstance(macro_polys_by_owner, Mapping)
                or set(macro_polys_by_owner) != {binding.owner for binding, _block in rows}):
            raise ValueError("row artwork inventory must exactly cover the row bindings")
        stamps = {stamp.name: stamp for stamp in self._snapshot.stamps}
        for binding, block in rows:
            expected = sorted((layer, ring) for name in block.device_stamps
                              for layer, rings in stamps[name].layers for ring in rings)
            actual = []
            polygons = macro_polys_by_owner[binding.owner]
            if type(polygons) not in {tuple, list}:
                raise TypeError("row artwork must be an explicit polygon sequence")
            for polygon in polygons:
                if type(polygon) is not LayerPolygon:
                    raise TypeError("row artwork requires exact LayerPolygon values")
                layer = _layer((polygon.layer, polygon.datatype))
                ring = _canonical_ring(tuple(_point((point.x, point.y)) for point in polygon.polygon.vertices))
                actual.append((layer, ring))
            if sorted(actual) != expected:
                raise ValueError("row artwork polygon multiset differs from actual native row stamps")

    def to_canonical_data(self) -> dict:
        inventory = self._replay()
        payload = {
            "schema": "clay.row-native-geometry-binding/v2" if inventory.shared_guards else "clay.row-native-geometry-binding/v1",
            "native_state_sha256": self.native_state_sha256,
            "actual_geometry_sha256": self.actual_geometry_sha256,
            "stamp_count": len(self._snapshot.stamps),
            "source_count": len(inventory.source_bboxes),
            "block_mapping": [{
                "actual_block": block.actual_id, "native_materialized_block": block.native_id,
                "master": block.master, "bbox_nm": list(block.bbox_nm),
                "stamps": list(block.all_stamps), "sources": list(block.sources),
            } for block in inventory.blocks],
        }
        if inventory.shared_guards:
            from dataclasses import asdict
            payload['shared_guard_mapping'] = [asdict(group) for group in inventory.shared_guards]
        return payload

    def to_canonical_bytes(self) -> bytes:
        return _bytes(self.to_canonical_data())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()
