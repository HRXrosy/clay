"""Deterministic RG5 compiler from topology declarations to rigid row macros.

``row_calculus`` describes one row macro.  This module closes the next, still
placement-independent boundary: it normalizes the cell's MOS records, assigns
each declared structural group to exactly one macro, and invokes the existing
shared-diffusion/common-centroid geometry builders once per macro.  The canonical
``place_generic`` master stores this result and later placers project it without
recompiling or invoking a second materializer.

There are no circuit names, reference coordinates, environment switches, or
router policy in this module.  All dimensions are exact integer nanometres and
all collection inputs are bounded before canonicalization.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from itertools import islice
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from clay.layout.joint_row_terminals import (
    component_terminal_contracts as _component_terminal_contracts,
    component_route_demands as _component_route_demands,
    require_terminal_member_nets as _terminal_member_nets,
    terminal_geometry_payload, require_macro_inventory, is_joint_inventory, joint_inventory_payload,
)

from .mosaic_row_compile import _compile_mosaic_macro, _mosaic_gap_nm
from .row_program_primitives import MAX_DIMENSION_NM, RowProgramError
from .row_program_ownership import expected_row_finger_owners

from .row_calculus import (
    BoundaryTerminal,
    DiffusionBoundary,
    DummyEndcapObligation,
    RowAtom,
    RowAtomKind,
    RowCalculusError,
    RowMacro,
    RowString,
    RowStructureKind,
    compile_row_macro,
)


MAX_PROGRAM_ENTRIES = 256
MAX_PROGRAM_MEMBERS = 4096
MAX_PROGRAM_RESIDUALS = 4096
MAX_GROUP_MEMBERS = 4096
MAX_TEXT_CHARS = 4096
MAX_FINGERS = 1_000_000
MAX_GEOMETRY_POLYGONS = 262_144
MAX_POLYGON_VERTICES = 4_096
MAX_MEMBER_TERMINALS = 64
MAX_GDS_LAYER_VALUE = 65_535
# Whole-materialization limits.  The per-polygon/per-realization bounds above do
# not prevent many individually legal row entries from multiplying into an
# unbounded replay workload.  These cumulative caps are checked before content
# hashing at every trust boundary.
MAX_MATERIALIZED_POLYGONS = 262_144
MAX_MATERIALIZED_VERTICES = 1_048_576


class RowProgramMaterializationError(RowProgramError):
    """A compiled program cannot be replayed by the rigid materializer."""


def _bounded(values: Iterable[Any], limit: int, field: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise RowProgramError(f"{field} must be an iterable of items, not text")
    try:
        result = tuple(islice(iter(values), limit + 1))
    except TypeError as exc:
        raise RowProgramError(f"{field} must be iterable") from exc
    if len(result) > limit:
        raise RowProgramError(f"{field} exceeds its {limit}-item cap+1 limit")
    return result


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise RowProgramError(f"{field} must be an exact str")
    result = value.strip()
    if not result and not allow_empty:
        raise RowProgramError(f"{field} must not be empty")
    if len(result) > MAX_TEXT_CHARS:
        raise RowProgramError(f"{field} exceeds {MAX_TEXT_CHARS} characters")
    return result


def _positive(value: object, field: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise RowProgramError(f"{field} must be an exact integer in [1, {maximum}]")
    return value


def _non_negative(value: object, field: str, maximum: int = MAX_DIMENSION_NM) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise RowProgramError(f"{field} must be an exact integer in [0, {maximum}]")
    return value


def _coordinate(
    value: object, field: str, *, maximum: int = MAX_DIMENSION_NM
) -> int:
    if type(value) is not int or not -maximum <= value <= maximum:
        raise RowProgramMaterializationError(
            f"{field} coordinate must be an exact integer in "
            f"[-{maximum}, {maximum}]"
        )
    return value


def _exact_text_tuple(values: Iterable[Any], field: str, limit: int) -> tuple[str, ...]:
    result = tuple(_text(item, field) for item in _bounded(values, limit, field))
    if len(set(result)) != len(result):
        raise RowProgramError(f"{field} must not contain duplicates")
    return result


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _digest(value: object, field: str) -> str:
    result = _text(value, field).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise RowProgramError(f"{field} must contain exactly 64 hexadecimal characters")
    return result


def row_terminal_sources(*, geometry: object, consumed_devices: tuple[str, ...], tech: object) -> tuple:
    """Derive the complete source terminal set from independent row evidence.

    Original replication records must cover the actual polygons as a multiset,
    and their terminals must independently equal the materializer's columns and
    per-occurrence gate contacts.  The component inventory never supplies its
    own expected denominator here.
    """
    from clay.layout.physical_terminal_inventory import PhysicalTerminalSource

    devices = _exact_text_tuple(consumed_devices, "terminal consumed devices", MAX_GROUP_MEMBERS)
    raw = getattr(geometry, "fixed_occurrence_sources", None)
    if type(raw) is not tuple or not raw:
        raise RowProgramMaterializationError("fixed row terminal source inventory is missing")
    raw = _bounded(raw, MAX_PROGRAM_MEMBERS, "terminal source occurrences")
    fingers = _bounded(getattr(geometry, "fingers", ()), MAX_PROGRAM_MEMBERS, "terminal source fingers")
    if len(raw) != len(fingers):
        raise RowProgramMaterializationError("terminal source occurrence denominator differs")
    layer = tuple(tech.metal_key(1))
    source_polygons = Counter()
    source_terminals = Counter()
    sources = []
    vertex_count = 0
    polygon_count = 0
    import klayout.db as kdb

    for index, (source, finger) in enumerate(zip(raw, fingers)):
        owner = _text(getattr(source, "owner", None), "terminal source owner")
        if owner not in devices or owner != getattr(finger, "member", None):
            raise RowProgramMaterializationError("terminal source occurrence owner differs")
        origin = getattr(source, "origin_nm", None)
        if type(origin) is not tuple or len(origin) != 2 or origin != (
            (int(finger.x0) + int(finger.x1)) // 2, int(finger.cy)
        ):
            raise RowProgramMaterializationError("terminal source occurrence frame differs")
        if getattr(source, "orientation", None) != getattr(finger, "orientation", None):
            raise RowProgramMaterializationError("terminal source occurrence orientation differs")
        records = getattr(source, "polygon_records", None)
        terminals = getattr(source, "terminals", None)
        if type(records) is not tuple or not records or type(terminals) is not tuple or not terminals:
            raise RowProgramMaterializationError("terminal source artwork/terminal records are missing")
        records = _bounded(records, MAX_MATERIALIZED_POLYGONS, "terminal source polygons")
        terminals = _bounded(terminals, MAX_MEMBER_TERMINALS, "terminal source contacts")
        metal = kdb.Region()
        for record in records:
            if type(record) is not tuple or len(record) != 3:
                raise RowProgramMaterializationError("terminal source polygon is malformed")
            ln, dt, vertices = record
            if (type(ln) is not int or not 0 <= ln <= MAX_GDS_LAYER_VALUE
                    or type(dt) is not int or not 0 <= dt <= MAX_GDS_LAYER_VALUE
                    or type(vertices) is not tuple or not 3 <= len(vertices) <= MAX_POLYGON_VERTICES):
                raise RowProgramMaterializationError("terminal source polygon is malformed")
            polygon_count += 1
            vertex_count += len(vertices)
            if polygon_count > MAX_MATERIALIZED_POLYGONS or vertex_count > MAX_MATERIALIZED_VERTICES:
                raise RowProgramMaterializationError("terminal source artwork exceeds complexity bounds")
            for point in vertices:
                if type(point) is not tuple or len(point) != 2:
                    raise RowProgramMaterializationError("terminal source polygon coordinate is malformed")
                _coordinate(point[0], "terminal source x")
                _coordinate(point[1], "terminal source y")
            source_polygons[record] += 1
            if (ln, dt) == layer:
                metal.insert(kdb.Polygon([kdb.Point(*point) for point in vertices]))
        metal.merge()
        role_ordinals: dict[str, int] = {}
        for terminal in terminals:
            if type(terminal) is not tuple or len(terminal) != 3:
                raise RowProgramMaterializationError("terminal source contact is malformed")
            role, net, box = terminal
            role = _text(role, "terminal source role")
            net = _text(net, "terminal source net")
            box = _bbox4(box, "terminal source contact")
            if not (kdb.Region(kdb.Box(*box)) - metal).is_empty():
                raise RowProgramMaterializationError("terminal source contact is outside its occurrence metal")
            ordinal = role_ordinals.get(role, 0)
            role_ordinals[role] = ordinal + 1
            sources.append(PhysicalTerminalSource(
                source_id=f"terminal:{index:06d}:{role}:{ordinal:04d}", member=owner,
                occurrence_index=index, role=role, net=net, layer=layer, bbox_nm=box,
            ))
            source_terminals[(role, net, box)] += 1
        if set(role_ordinals) != {"d", "g", "s"}:
            raise RowProgramMaterializationError("fixed row occurrence lacks the complete terminal roles")
    actual_polygons = Counter(
        (shape.layer, shape.datatype, tuple((point.x, point.y) for point in shape.polygon.vertices))
        for shape in _bounded(geometry.polygons, MAX_MATERIALIZED_POLYGONS, "terminal row polygons")
    )
    if source_polygons != actual_polygons:
        raise RowProgramMaterializationError("terminal source artwork differs from actual row polygons")
    actual_terminals = Counter(
        (column.role, column.net, tuple(column.contact_bbox))
        for column in _bounded(getattr(geometry, "columns", ()), MAX_GROUP_MEMBERS * MAX_MEMBER_TERMINALS,
                               "terminal row columns")
        if column.role in {"d", "s"}
    )
    actual_terminals.update(("g", finger.gate_net,
        _bbox4(getattr(finger, "gate_bbox_nm", None), "terminal gate contact")) for finger in fingers)
    if source_terminals != actual_terminals:
        raise RowProgramMaterializationError("terminal source denominator differs from actual columns/gate contacts")
    return tuple(sorted(sources, key=lambda item: item.source_id))


def replay_row_terminal_inventory(*, geometry: object, consumed_devices: tuple[str, ...], tech: object = None):
    """Replay required fixed-row components without an optional-field downgrade."""
    if getattr(geometry, 'joint_row_binding', None) is not None:
        from clay.layout.joint_row_geometry import replay_joint_row_geometry
        return replay_joint_row_geometry(geometry, consumed_devices, tech=tech)
    inventory = getattr(geometry, "physical_terminal_inventory", None)
    required = (getattr(geometry, "fixed_occurrence_sources", None) is not None
                or getattr(geometry, "fixed_pcell", None) is not None)
    if inventory is None and not required:
        return None
    from clay.layout.physical_terminal_inventory import PhysicalTerminalInventory, replay_physical_terminal_inventory
    from clay.layout.tech import LayoutTech
    if type(inventory) is not PhysicalTerminalInventory:
        raise RowProgramMaterializationError("fixed row physical terminal inventory is missing or foreign")
    if tech is None:
        tech = LayoutTech.resolve(inventory.pdk_id)
    sources = row_terminal_sources(geometry=geometry, consumed_devices=consumed_devices, tech=tech)
    return replay_physical_terminal_inventory(geometry=geometry, sources=sources, inventory=inventory, tech=tech)


def _require_row_terminal_member_nets(geometry: object, consumed_devices: tuple[str, ...], members: Mapping,
                                      *, tech: object = None):
    return _terminal_member_nets(geometry, consumed_devices, members, tech)


def row_geometry_sha256(
    geometry: Any, consumed_devices: Iterable[str]
) -> str:
    """Content-address the exact stored macro artwork and terminal interface.

    Bind emitted artwork, duplicate polygons and the complete terminal interface;
    fixed rows also bind the independently replayed per-owner artwork partition.
    """

    return _row_geometry_digest(geometry, consumed_devices, replay_row_terminal_inventory)


def _row_geometry_digest_from_inventory(geometry, consumed_devices, inventory):
    """Canonicalize an inventory already replayed by the private native scope."""
    return _row_geometry_digest(geometry, consumed_devices, lambda **_operands: inventory)


def _row_geometry_digest(geometry, consumed_devices, terminal_replay):
    devices = tuple(sorted(_exact_text_tuple(
        consumed_devices, "geometry consumed devices", MAX_GROUP_MEMBERS
    )))
    raw_polygons_iter = getattr(geometry, "polygons", ())
    from clay._native import load_native as _load_native_eng
    _eng_cpp = _load_native_eng("_engine_cpp")
    sorted_polygon_records = None
    if _eng_cpp is not None and hasattr(_eng_cpp, "canonical_row_polygon_records_cpp"):
        try:
            sorted_polygon_records = _eng_cpp.canonical_row_polygon_records_cpp(
                raw_polygons_iter,
                MAX_GEOMETRY_POLYGONS,
                MAX_POLYGON_VERTICES,
                MAX_MATERIALIZED_VERTICES,
                MAX_DIMENSION_NM,
            )
        except OverflowError as exc:
            msg = str(exc)
            if msg.startswith("CAP_POLYGONS:"):
                raise RowProgramError(
                    f"materialized geometry polygons exceeds its {MAX_GEOMETRY_POLYGONS}-item cap+1 limit"
                ) from exc
            if msg.startswith("CAP_VERTS:"):
                raise RowProgramError(
                    f"materialized polygon vertices exceeds its {MAX_POLYGON_VERTICES}-item cap+1 limit"
                ) from exc
            raise
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith("MAT:"):
                raise RowProgramMaterializationError(msg[4:]) from exc
            raise
    if sorted_polygon_records is None:
        raw_polygons = _bounded(
            raw_polygons_iter,
            MAX_GEOMETRY_POLYGONS,
            "materialized geometry polygons",
        )
        polygon_records = []
        cumulative_vertices = 0
        max_dim = MAX_DIMENSION_NM
        neg_max_dim = -MAX_DIMENSION_NM
        for layer_polygon in raw_polygons:
            layer = getattr(layer_polygon, "layer", None)
            datatype = getattr(layer_polygon, "datatype", None)
            if type(layer) is not int or type(datatype) is not int:
                raise RowProgramMaterializationError(
                    "materialized polygon layer/datatype must be exact integers"
                )
            poly_obj = getattr(layer_polygon, "polygon", None)
            verts_raw = getattr(poly_obj, "vertices", ())
            if type(verts_raw) is list or type(verts_raw) is tuple:
                if len(verts_raw) > MAX_POLYGON_VERTICES:
                    raise RowProgramError(
                        f"materialized polygon vertices exceeds its {MAX_POLYGON_VERTICES}-item cap+1 limit"
                    )
                raw_vertices = verts_raw
            else:
                raw_vertices = _bounded(
                    verts_raw,
                    MAX_POLYGON_VERTICES,
                    "materialized polygon vertices",
                )
            n_verts = len(raw_vertices)
            if n_verts < 3:
                raise RowProgramMaterializationError(
                    "materialized polygon must have at least three vertices"
                )
            vertices = []
            for vertex in raw_vertices:
                x = getattr(vertex, "x", None)
                if type(x) is not int or x < neg_max_dim or x > max_dim:
                    raise RowProgramMaterializationError(
                        "materialized polygon x coordinate must be an exact integer in "
                        f"[-{max_dim}, {max_dim}]"
                    )
                y = getattr(vertex, "y", None)
                if type(y) is not int or y < neg_max_dim or y > max_dim:
                    raise RowProgramMaterializationError(
                        "materialized polygon y coordinate must be an exact integer in "
                        f"[-{max_dim}, {max_dim}]"
                    )
                vertices.append((x, y))
            cumulative_vertices += n_verts
            if cumulative_vertices > MAX_MATERIALIZED_VERTICES:
                raise RowProgramMaterializationError(
                    "materialization exceeds cumulative vertex complexity cap"
                )
            polygon_records.append((layer, datatype, tuple(vertices)))
        sorted_polygon_records = tuple(sorted(polygon_records))

    member_pin_records = []
    member_pins = getattr(geometry, "member_pins", None)
    if not callable(member_pins):
        raise RowProgramMaterializationError(
            "materialized geometry exposes no member_pins accessor"
        )
    for device in devices:
        try:
            pins = member_pins(device)
        except KeyError:
            pins = {}
        if not isinstance(pins, Mapping):
            raise RowProgramMaterializationError(
                "materialized member_pins must return a bounded mapping"
            )
        items = _bounded(
            pins.items(), MAX_MEMBER_TERMINALS, "materialized member terminals"
        )
        rebuilt = []
        for role, point in items:
            role = _text(role, "materialized terminal role")
            if type(point) is not tuple or len(point) != 2:
                raise RowProgramMaterializationError(
                    "materialized terminal coordinate must be an exact integer 2-tuple"
                )
            rebuilt.append((role, (
                _coordinate(point[0], "materialized terminal x"),
                _coordinate(point[1], "materialized terminal y"),
            )))
        if len({role for role, _point in rebuilt}) != len(rebuilt):
            raise RowProgramMaterializationError(
                "materialized terminal roles must be unique"
            )
        member_pin_records.append((device, tuple(sorted(rebuilt))))

    extent = _bbox4(tuple(geometry.extent()), "materialized geometry extent")
    payload = {
        "schema": "clay.row-macro-geometry.v1",
        "consumed_devices": devices,
        "extent_nm": extent,
        "member_pins": tuple(member_pin_records),
        "polygons": sorted_polygon_records,
    }
    terminals = terminal_replay(geometry=geometry, consumed_devices=devices)
    if getattr(geometry, 'native_owned_artwork', None) is not None:
        from clay.layout.native_owned_artwork import replay_native_owned_artwork
        payload['native_owned_artwork'] = replay_native_owned_artwork(geometry, devices).payload()
    if terminals is not None:
        payload.update(terminal_geometry_payload(geometry, terminals))
    if getattr(geometry, "channel_inventory", None) is not None:
        from clay.layout.occurrence_channels import replay_row_channel_inventory

        channels = replay_row_channel_inventory(
            geometry=geometry, consumed_devices=devices,
        )
        # Resolved-master occurrences and physical channels have different
        # denominators. Seal their explicit join and the original row frame.
        payload["schema"] = "clay.row-macro-geometry.v2"
        payload["channel_inventory"] = channels.payload()
        payload["occurrence_frames"] = tuple(
            (
                _text(finger.member, "channel occurrence owner"),
                _coordinate(finger.x0, "channel occurrence x0"),
                _coordinate(finger.x1, "channel occurrence x1"),
                _coordinate(finger.cy, "channel occurrence cy"),
                _text(getattr(finger, "orientation", ""), "channel occurrence orientation", allow_empty=True),
                None if getattr(finger, "gate_bbox_nm", None) is None else
                _bbox4(finger.gate_bbox_nm, "channel occurrence terminal probe"),
            )
            for finger in geometry.fingers
        )
    if getattr(geometry, "mosaic_guest_slot_binding", None) is not None:
        from clay.layout.spatial.mosaic_guest_slot import replay_mosaic_guest_slot_geometry
        payload["mosaic_guest_slot"] = replay_mosaic_guest_slot_geometry(geometry)
    if getattr(geometry, 'mosaic_boundary_intent', None) is not None:
        from clay.layout.spatial.mosaic_boundary_joins import replay_mosaic_boundary_join_intent
        payload['mosaic_boundary_intent'] = replay_mosaic_boundary_join_intent(geometry.mosaic_boundary_intent).canonical_data()
    if getattr(geometry, 'mosaic_lattice_intent', None) is not None:
        from clay.layout.mosaic_lattice_materialization import mosaic_lattice_geometry_payload
        payload['mosaic_lattice_intent'] = mosaic_lattice_geometry_payload(geometry)
    if terminals is not None:
        payload["schema"] = "clay.row-macro-geometry.v5" if is_joint_inventory(terminals) else "clay.row-macro-geometry.v4"
    return _sha(payload)


def _is_shared_native_lattice_binding(geometry: object) -> bool:
    from clay.layout.joint_row_geometry import JointRowGeometryBinding

    binding = getattr(geometry, "joint_row_binding", None)
    if type(binding) is not JointRowGeometryBinding:
        return False
    return (
        len(binding.occurrences) > 1
        and all(bool(occ.master_site_ids) and bool(occ.channel_site_id) for occ in binding.occurrences)
        and any(part.kind == "raw_unit" and len(part.occurrence_ids) > 1 for part in binding.parts)
    )


def _replay_shared_native_lattice_entry_geometry(
    entry: object, geometry: object, *, tech=None, members=None
):
    from clay.layout.device_cell import _resolve_layer_keys
    from clay.layout.joint_row_geometry import (
        JointRowGeometryBinding,
        _region,
        joint_checkpoint,
        joint_d4_orientation,
        joint_record_counter,
        joint_row_polygons,
    )
    from clay.layout.joint_terminal_inventory import _native_frame
    from clay.layout.occurrence_artwork import polygon_records
    from clay.layout.spatial.intent_projection import replay_discrete_mosaic_intent
    from clay.layout.tech import LayoutTech

    def _req(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError("mosaic lattice " + message)

    prefix = "mosaic_channel_lattice:"
    refs = tuple(ref[len(prefix):] for ref in entry.declaration_refs if ref.startswith(prefix))
    intent = getattr(geometry, "mosaic_lattice_intent", None)
    if not refs and intent is None:
        return None
    checkpoint = joint_checkpoint()
    _req(len(refs) == 1 and intent is not None, "independent source or native intent is missing")
    intent = replay_discrete_mosaic_intent(intent)
    _req(
        bool(intent.column_ranks) and refs[0] == intent.sha256 and intent.sha256 in entry.declaration_refs,
        "complete source intent or hash differs",
    )
    _req(
        "asir:" + intent.source_asir_sha256 in entry.declaration_refs
        and entry.group == intent.mosaic_id
        and entry.consumed_devices == intent.members
        and getattr(entry.structure_kind, "value", None) == "discrete_mosaic",
        "compiled source identity differs",
    )
    binding = getattr(geometry, "joint_row_binding", None)
    _req(type(binding) is JointRowGeometryBinding, "complete joint source binding is missing")
    binding.__post_init__()
    occurrences = {occ.occurrence_id: occ for occ in binding.occurrences}
    expected_nodes = tuple(node.node_id for node in intent.occurrences)
    _req(
        expected_nodes == tuple(occ.occurrence_id for occ in binding.occurrences)
        and len(occurrences) == len(expected_nodes),
        "complete ordered occurrence cover differs",
    )
    _req(len(entry.macro.rows) == len(intent.rows), "complete compiled row cover differs")
    for row, projected in zip(entry.macro.rows, intent.rows):
        checkpoint()
        _req(len(row.atoms) == len(projected), "compiled row occurrence count differs")
        for atom, node in zip(row.atoms, projected):
            occurrence = occurrences[node.node_id]
            _req(
                occurrence.owner == node.logical_owner
                and joint_d4_orientation(occurrence.master_to_local) == node.transform
                and atom.match_key == f"mosaic:{node.node_id}:{node.transform}"
                and atom.atom_id == f"{entry.group}:r{row.row_index}:active:{node.logical_owner}:{node.node_id}",
                "compiled source occurrence, transform or order differs",
            )
    _req(
        joint_record_counter(polygon_records(geometry.polygons))
        == joint_record_counter(polygon_records(joint_row_polygons(binding.parts))),
        "actual native artwork differs from the complete physical parts",
    )
    tech = LayoutTech.resolve(binding.pdk_id) if tech is None else tech
    _req(tech.pdk_id == binding.pdk_id, "current PDK differs")
    parts = {part.part_id: part for part in binding.parts}
    points = {}
    for occurrence in binding.occurrences:
        checkpoint()
        kind = getattr(geometry, "kind", None) if members is None else members[occurrence.owner].kind
        _req(kind in ("nmos", "pmos"), "source family is unsupported")
        keys = _resolve_layer_keys(tech, kind)
        raw = parts.get(occurrence.raw_part_id)
        _req(
            raw is not None
            and raw.kind == "raw_unit"
            and occurrence.occurrence_id in raw.occurrence_ids
            and bool(occurrence.master_site_ids)
            and bool(occurrence.channel_site_id),
            "raw occurrence part association differs",
        )
        layers = dict(raw.geometry_by_layer_nm)
        _req(
            tuple(keys.active) in layers and tuple(keys.poly) in layers,
            "raw active/poly channel layers are missing",
        )
        gate_site = next(
            (s for s in occurrence.master.terminal_sites if s.site_id == occurrence.channel_site_id),
            None,
        )
        _req(gate_site is not None and gate_site.role_group == "gate", "selected channel gate is missing")
        gate_shapes = dict(gate_site.geometry_by_layer_nm).get(tuple(keys.poly))
        _req(gate_shapes is not None, "selected channel gate lacks poly geometry")
        gate_reg = _region(gate_shapes).transformed(_native_frame(occurrence.master_to_local))
        channel = (_region(layers[tuple(keys.active)]) & gate_reg).merged()
        box = channel.bbox()
        _req(
            channel.size() == 1
            and box.width() > 0
            and box.height() > 0
            and channel.area() == box.width() * box.height(),
            "raw occurrence must expose exactly one complete rectangular channel",
        )
        points[occurrence.occurrence_id] = (box.left + box.right, box.bottom + box.top)
    _req(len(set(points.values())) == len(points), "actual channel slots are duplicated")
    xs = {x: rank for rank, x in enumerate(sorted({p[0] for p in points.values()}))}
    ys = {y: rank for rank, y in enumerate(sorted({p[1] for p in points.values()}))}
    expected_x = {f"{owner}[{ordinal:03d}]": rank for owner, ordinal, rank in intent.column_ranks}
    expected_y = {node.node_id: rank for rank, row in enumerate(intent.rows) for node in row}
    _req(
        {node: xs[point[0]] for node, point in points.items()} == expected_x
        and {node: ys[point[1]] for node, point in points.items()} == expected_y,
        "actual raw channel row/column ranks differ from source",
    )
    checkpoint()
    return intent


def _replay_guest_slot_entry_geometry(entry: object, geometry: object, *, tech=None, members=None) -> None:
    from clay.layout.mosaic_boundary_materialization import replay_mosaic_boundary_entry_geometry
    from clay.layout.mosaic_lattice_materialization import replay_mosaic_lattice_entry_geometry
    try:
        replay_mosaic_boundary_entry_geometry(entry, geometry, tech=tech, members=members)
        if _is_shared_native_lattice_binding(geometry):
            _replay_shared_native_lattice_entry_geometry(entry, geometry, tech=tech, members=members)
        else:
            replay_mosaic_lattice_entry_geometry(entry, geometry, tech=tech, members=members)
    except (TypeError, ValueError) as exc:
        raise RowProgramMaterializationError(str(exc)) from exc
    refs = tuple(ref.split(":", 1)[1] for ref in entry.declaration_refs if ref.startswith("mosaic_guest_slot:"))
    binding = getattr(geometry, "mosaic_guest_slot_binding", None)
    if not refs and binding is None:
        return
    if len(refs) != 1 or binding is None:
        raise RowProgramMaterializationError("mosaic slot declaration/geometry binding is missing or ambiguous")
    from clay.layout.spatial.mosaic_guest_slot import replay_mosaic_guest_slot_geometry
    try:
        replay_mosaic_guest_slot_geometry(geometry, expected_slot_sha256=refs[0])
    except (TypeError, ValueError) as exc:
        raise RowProgramMaterializationError(str(exc)) from exc


def _geometry_complexity(geometry: Any) -> tuple[int, int]:
    polygons = _bounded(
        getattr(geometry, "polygons", ()),
        MAX_GEOMETRY_POLYGONS,
        "materialized geometry polygons",
    )
    vertices = 0
    for layer_polygon in polygons:
        vertices += len(_bounded(
            getattr(getattr(layer_polygon, "polygon", None), "vertices", ()),
            MAX_POLYGON_VERTICES,
            "materialized polygon vertices",
        ))
    return len(polygons), vertices


def _structure(value: RowStructureKind | str) -> RowStructureKind:
    if type(value) is RowStructureKind:
        return value
    if type(value) is not str:
        raise RowProgramError("structure_kind must be exact RowStructureKind or str")
    try:
        return RowStructureKind(value.strip().lower())
    except ValueError as exc:
        raise RowProgramError(f"unsupported structure_kind {value!r}") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class RowMemberSpec:
    """One exact MOS record consumed by a row entry."""

    device: str
    kind: str
    width_nm: int
    length_nm: int
    fingers: int
    gate_net: str
    drain_net: str
    source_net: str
    body_net: str = ""
    oxide: str = "thin"
    dummy_for: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _text(self.device, "device"))
        kind = _text(self.kind, "kind").lower()
        if kind not in ("nmos", "pmos"):
            raise RowProgramError("kind must normalize to nmos or pmos")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "width_nm", _positive(self.width_nm, "width_nm", MAX_DIMENSION_NM)
        )
        object.__setattr__(
            self, "length_nm", _positive(self.length_nm, "length_nm", MAX_DIMENSION_NM)
        )
        object.__setattr__(self, "fingers", _positive(self.fingers, "fingers", MAX_FINGERS))
        for field in ("gate_net", "drain_net", "source_net"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "body_net", _text(self.body_net, "body_net", allow_empty=True))
        oxide = _text(self.oxide, "oxide").lower()
        if oxide not in ("thin", "thick"):
            raise RowProgramError("oxide must be thin or thick")
        object.__setattr__(self, "oxide", oxide)
        object.__setattr__(
            self, "dummy_for", _text(self.dummy_for, "dummy_for", allow_empty=True)
        )

    @property
    def builder_tuple(
        self,
    ) -> tuple[str, str, float, float, int, str, str, str, str]:
        return (
            self.device,
            self.kind,
            self.width_nm / 1000.0,
            self.length_nm / 1000.0,
            self.fingers,
            self.gate_net,
            self.drain_net,
            self.source_net,
            self.body_net,
        )

    def payload(self) -> dict[str, object]:
        return {
            "device": self.device,
            "kind": self.kind,
            "width_nm": self.width_nm,
            "length_nm": self.length_nm,
            "fingers": self.fingers,
            "gate_net": self.gate_net,
            "drain_net": self.drain_net,
            "source_net": self.source_net,
            "body_net": self.body_net,
            "oxide": self.oxide,
            "dummy_for": self.dummy_for,
        }


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RowProgramResidual:
    code: str
    group: str
    declaration_refs: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, "residual code"))
        object.__setattr__(self, "group", _text(self.group, "residual group"))
        object.__setattr__(
            self,
            "declaration_refs",
            tuple(sorted(_exact_text_tuple(
                self.declaration_refs, "residual declaration_refs", MAX_PROGRAM_ENTRIES
            ))),
        )
        object.__setattr__(
            self,
            "devices",
            tuple(sorted(_exact_text_tuple(
                self.devices, "residual devices", MAX_GROUP_MEMBERS
            ))),
        )
        object.__setattr__(self, "detail", _text(self.detail, "residual detail", allow_empty=True))

    def payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "group": self.group,
            "declaration_refs": self.declaration_refs,
            "devices": self.devices,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RowBodyTapDelegation:
    """Exact canonical guard-ring declaration that owns a PMOS row body tap."""

    declaration_sha256: str
    members: tuple[str, ...]
    well: str
    tie_net: str
    hug: str
    role: str
    enclose: str = ""

    def __post_init__(self) -> None:
        from clay.layout.intent import GuardRingIntent

        if (
            type(self.declaration_sha256) is not str
            or self.declaration_sha256 != self.declaration_sha256.strip()
            or self.declaration_sha256 != self.declaration_sha256.lower()
        ):
            raise RowProgramError(
                "body-tap declaration_sha256 must be exact lowercase canonical text"
            )
        object.__setattr__(
            self,
            "declaration_sha256",
            _digest(self.declaration_sha256, "body-tap declaration_sha256"),
        )
        if type(self.members) is not tuple:
            raise RowProgramError("body-tap members must be an exact tuple")
        members = _exact_text_tuple(
            self.members, "body-tap members", MAX_GROUP_MEMBERS
        )
        if members != self.members:
            raise RowProgramError("body-tap members must be canonical text")
        object.__setattr__(self, "members", members)
        for field in ("well", "tie_net", "hug", "role"):
            raw = getattr(self, field)
            value = _text(raw, field)
            if value != raw:
                raise RowProgramError(f"body-tap {field} must be canonical text")
            object.__setattr__(self, field, value)
        raw_enclose = self.enclose
        object.__setattr__(
            self, "enclose", _text(self.enclose, "enclose", allow_empty=True)
        )
        if self.enclose != raw_enclose:
            raise RowProgramError("body-tap enclose must be canonical text")
        if (
            self.well != "n"
            or self.hug != "tight"
            or self.role != "tap"
            or self.enclose
        ):
            raise RowProgramError(
                "body-tap delegation requires an exact tight n-well tap ring"
            )
        replayed = GuardRingIntent(
            members=self.members,
            well=self.well,
            tie_net=self.tie_net,
            hug=self.hug,
            role=self.role,
            enclose=self.enclose,
        )
        if replayed.sha256 != self.declaration_sha256:
            raise RowProgramError(
                "body-tap delegation digest does not match its guard-ring declaration"
            )

    def payload(self) -> dict[str, object]:
        return {
            "declaration_sha256": self.declaration_sha256,
            "members": self.members,
            "well": self.well,
            "tie_net": self.tie_net,
            "hug": self.hug,
            "role": self.role,
            "enclose": self.enclose,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RowProgramEntry:
    entry_id: str
    group: str
    owner: str
    structure_kind: RowStructureKind | str
    rows: int
    order: str
    active_devices: tuple[str, ...]
    dummy_devices: tuple[str, ...]
    declaration_refs: tuple[str, ...]
    macro: RowMacro
    orientation: str = "R0"
    body_tap_delegation: RowBodyTapDelegation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _text(self.entry_id, "entry_id"))
        object.__setattr__(self, "group", _text(self.group, "group"))
        object.__setattr__(self, "owner", _text(self.owner, "owner"))
        object.__setattr__(self, "structure_kind", _structure(self.structure_kind))
        object.__setattr__(self, "rows", _positive(self.rows, "rows", 256))
        object.__setattr__(self, "order", _text(self.order, "order", allow_empty=True))
        # Active order is physical program data (e.g. ratio order keeps the diode
        # member first).  The compiler derives it canonically; the IR must not sort
        # that semantic sequence a second time.
        active = _exact_text_tuple(
            self.active_devices, "active_devices", MAX_GROUP_MEMBERS
        )
        if not active:
            raise RowProgramError("active_devices must not be empty")
        dummy = tuple(sorted(_exact_text_tuple(
            self.dummy_devices, "dummy_devices", MAX_GROUP_MEMBERS
        )))
        if set(active) & set(dummy):
            raise RowProgramError("active_devices and dummy_devices must be disjoint")
        object.__setattr__(self, "active_devices", active)
        object.__setattr__(self, "dummy_devices", dummy)
        object.__setattr__(
            self,
            "declaration_refs",
            tuple(sorted(_exact_text_tuple(
                self.declaration_refs, "declaration_refs", MAX_PROGRAM_ENTRIES
            ))),
        )
        if type(self.macro) is not RowMacro:
            raise RowProgramError("macro must be an exact RowMacro")
        if self.macro.structure_kind is not self.structure_kind:
            raise RowProgramError("entry and macro structure_kind mismatch")
        if self.macro.macro_id != self.entry_id:
            raise RowProgramError("entry_id and macro_id mismatch")
        if self.owner != self.entry_id or self.entry_id != f"row:{self.group}":
            raise RowProgramError("entry owner/id must be the canonical row:<group> owner")
        if len(self.macro.rows) != self.rows:
            raise RowProgramError("entry rows and macro row count mismatch")
        if self.declaration_refs != self.macro.declaration_refs:
            raise RowProgramError("entry and macro declaration_refs mismatch")
        if type(self.orientation) is not str or self.orientation not in (
            "R0", "R90", "R180", "R270"
        ):
            raise RowProgramError(
                "entry orientation must be one of R0/R90/R180/R270"
            )
        if self.body_tap_delegation is not None:
            if type(self.body_tap_delegation) is not RowBodyTapDelegation:
                raise RowProgramError(
                    "body_tap_delegation must be an exact RowBodyTapDelegation"
                )
            if tuple(sorted(self.body_tap_delegation.members)) != self.consumed_devices:
                raise RowProgramError(
                    "body-tap owner members must exactly equal consumed devices"
                )
        macro_active = {
            _atom_owner(atom, active)
            for row in self.macro.rows for atom in row.atoms
            if atom.kind is RowAtomKind.ACTIVE
        }
        if macro_active != set(active):
            raise RowProgramError(
                "active_devices must exactly equal macro active match-key owners"
            )

    @property
    def consumed_devices(self) -> tuple[str, ...]:
        return tuple(sorted((*self.active_devices, *self.dummy_devices)))

    def payload(self) -> dict[str, object]:
        payload = {
            "entry_id": self.entry_id,
            "group": self.group,
            "owner": self.owner,
            "structure_kind": self.structure_kind.value,
            "rows": self.rows,
            "order": self.order,
            "active_devices": self.active_devices,
            "dummy_devices": self.dummy_devices,
            "consumed_devices": self.consumed_devices,
            "declaration_refs": self.declaration_refs,
            "macro": self.macro.final_master_contract,
        }
        if self.orientation != "R0":
            payload["orientation"] = self.orientation
        if self.body_tap_delegation is not None:
            payload["body_tap_delegation"] = self.body_tap_delegation.payload()
        return payload

    @property
    def sha256(self) -> str:
        return _sha(self.payload())


@dataclass(frozen=True, slots=True, kw_only=True)
class RowCompileCertificate:
    schema: str
    cell: str
    tech_id: str
    program_sha256: str
    entry_sha256s: tuple[tuple[str, str], ...]
    consumed_devices: tuple[str, ...]
    residual_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "certificate schema"))
        object.__setattr__(self, "cell", _text(self.cell, "certificate cell"))
        object.__setattr__(self, "tech_id", _text(self.tech_id, "certificate tech_id"))
        object.__setattr__(self, "program_sha256", _digest(self.program_sha256, "program_sha256"))
        pairs = _bounded(self.entry_sha256s, MAX_PROGRAM_ENTRIES, "certificate entries")
        rebuilt_pairs: list[tuple[str, str]] = []
        for pair in pairs:
            if type(pair) is not tuple or len(pair) != 2:
                raise RowProgramError("certificate entry hashes must be exact 2-tuples")
            rebuilt_pairs.append((
                _text(pair[0], "certificate entry_id"),
                _digest(pair[1], "certificate entry sha256"),
            ))
        rebuilt_pairs.sort()
        if len({entry_id for entry_id, _digest_value in rebuilt_pairs}) != len(rebuilt_pairs):
            raise RowProgramError("certificate entry ids must be unique")
        object.__setattr__(self, "entry_sha256s", tuple(rebuilt_pairs))
        object.__setattr__(
            self, "consumed_devices",
            tuple(sorted(_exact_text_tuple(
                self.consumed_devices, "certificate consumed_devices", MAX_PROGRAM_MEMBERS
            ))),
        )
        object.__setattr__(
            self, "residual_codes",
            tuple(sorted(
                _text(item, "certificate residual code")
                for item in _bounded(
                    self.residual_codes, MAX_PROGRAM_RESIDUALS,
                    "certificate residual_codes",
                )
            )),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RowProgram:
    cell: str
    tech_id: str
    members: tuple[RowMemberSpec, ...]
    entries: tuple[RowProgramEntry, ...]
    residuals: tuple[RowProgramResidual, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _text(self.cell, "cell"))
        object.__setattr__(self, "tech_id", _text(self.tech_id, "tech_id"))
        members = _bounded(self.members, MAX_PROGRAM_MEMBERS, "program members")
        if any(type(item) is not RowMemberSpec for item in members):
            raise RowProgramError("program members must be exact RowMemberSpec values")
        members = tuple(sorted(members, key=lambda item: item.device))
        if len({item.device for item in members}) != len(members):
            raise RowProgramError("program member device names must be unique")
        entries = _bounded(self.entries, MAX_PROGRAM_ENTRIES, "program entries")
        if any(type(item) is not RowProgramEntry for item in entries):
            raise RowProgramError("program entries must be exact RowProgramEntry values")
        entries = tuple(sorted(entries, key=lambda item: item.entry_id))
        if len({item.entry_id for item in entries}) != len(entries):
            raise RowProgramError("program entry_id values must be unique")
        if len({item.owner for item in entries}) != len(entries):
            raise RowProgramError("program entry owner values must be unique")
        residuals = _bounded(self.residuals, MAX_PROGRAM_RESIDUALS, "program residuals")
        if any(type(item) is not RowProgramResidual for item in residuals):
            raise RowProgramError("program residuals must be exact RowProgramResidual values")
        residuals = tuple(sorted(residuals))
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "residuals", residuals)
        member_names = {item.device for item in members}
        consumed_list = [device for entry in entries for device in entry.consumed_devices]
        missing = tuple(sorted(set(consumed_list) - member_names))
        if missing:
            raise RowProgramError(f"entries consume absent member records {missing}")
        duplicates = tuple(sorted({device for device in consumed_list if consumed_list.count(device) > 1}))
        if duplicates:
            raise RowProgramError(f"entries consume devices more than once {duplicates}")
        member_by_name = {item.device: item for item in members}
        for entry in entries:
            active_counts: dict[str, int] = {}
            dummy_counts: dict[str, int] = {}
            for row in entry.macro.rows:
                for atom in row.atoms:
                    if atom.kind is RowAtomKind.ACTIVE:
                        owner = _atom_owner(atom, entry.active_devices)
                        weight = 2 if entry.structure_kind is RowStructureKind.COMMON_CENTROID and atom.match_key == "cc:matched-pair-unit" else 1
                        active_counts[owner] = active_counts.get(owner, 0) + weight
                    else:
                        owners = tuple(
                            device for device in entry.dummy_devices
                            if f":{device}:" in f":{atom.atom_id}:"
                        )
                        if len(owners) != 1:
                            raise RowProgramError(
                                f"dummy atom {atom.atom_id!r} lacks one exact consumed owner"
                            )
                        dummy_counts[owners[0]] = dummy_counts.get(owners[0], 0) + 1
            expected_active = {
                device: member_by_name[device].fingers for device in entry.active_devices
            }
            expected_dummy = {
                device: member_by_name[device].fingers for device in entry.dummy_devices
            }
            if active_counts != expected_active:
                raise RowProgramError(
                    f"macro active finger ownership {active_counts} != records {expected_active}"
                )
            if dummy_counts != expected_dummy:
                raise RowProgramError(
                    f"macro dummy finger ownership {dummy_counts} != records {expected_dummy}"
                )

    @property
    def consumed_devices(self) -> tuple[str, ...]:
        return tuple(sorted({d for entry in self.entries for d in entry.consumed_devices}))

    def payload(self) -> dict[str, object]:
        # Unconsumed records are intentionally absent: deleting a declaration leaves
        # no device residue in the structural program.
        consumed = set(self.consumed_devices)
        return {
            "schema": "clay.row-program.v1",
            "cell": self.cell,
            "tech_id": self.tech_id,
            "members": tuple(item.payload() for item in self.members if item.device in consumed),
            "entries": tuple(item.payload() for item in self.entries),
            "residuals": tuple(item.payload() for item in self.residuals),
            "consumed_devices": self.consumed_devices,
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json(self.payload())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    @property
    def certificate(self) -> RowCompileCertificate:
        return RowCompileCertificate(
            schema="clay.row-compile-certificate.v1",
            cell=self.cell,
            tech_id=self.tech_id,
            program_sha256=self.sha256,
            entry_sha256s=tuple((entry.entry_id, entry.sha256) for entry in self.entries),
            consumed_devices=self.consumed_devices,
            residual_codes=tuple(item.code for item in self.residuals),
        )


def _normalize_kind(raw: object) -> tuple[str, str]:
    token = _text(raw, "record kind").lower()
    if "pmos" in token or "pfet" in token:
        return "pmos", "thick" if "3v" in token or "33" in token else "thin"
    if "nmos" in token or "nfet" in token:
        return "nmos", "thick" if "3v" in token or "33" in token else "thin"
    raise RowProgramError(f"record kind {token!r} is not a MOS polarity")


def _atom_owner(atom: RowAtom, devices: Iterable[str]) -> str:
    matches = tuple(
        device for device in devices
        if f":active:{device}:" in f":{atom.atom_id}:"
    )
    if len(matches) != 1:
        raise RowProgramError(f"active atom {atom.atom_id!r} lacks one exact owner")
    return matches[0]


def _record_is_mos(raw: object) -> bool:
    if not isinstance(raw, Mapping):
        return True  # let the exact record validator produce the named error
    kind = raw.get("kind", raw.get("device_class", ""))
    if type(kind) is not str:
        return True
    token = kind.strip().lower()
    return any(part in token for part in ("nmos", "pmos", "nfet", "pfet"))


def _nm_from_record(record: Mapping[str, object], short: str) -> int:
    nm_key = f"{short}_nm"
    if nm_key in record:
        return _positive(record[nm_key], nm_key, MAX_DIMENSION_NM)
    if short not in record:
        raise RowProgramError(f"record is missing {nm_key} or {short}")
    value = record[short]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RowProgramError(f"record {short} must be a finite numeric metre value")
    scaled = float(value) * 1_000_000_000.0
    nearest = round(scaled)
    tolerance = max(1e-6, abs(scaled) * 1e-12)
    if (
        not math.isfinite(scaled)
        or not 0 < nearest <= MAX_DIMENSION_NM
        or abs(scaled - nearest) > tolerance
    ):
        raise RowProgramError(f"record {short} must normalize to exact positive nm")
    return int(nearest)


def _pin(record: Mapping[str, object], role: str) -> str:
    pins = record.get("pins", {})
    if not isinstance(pins, Mapping):
        raise RowProgramError("record pins must be a mapping")
    aliases = {"gate": ("g", "gate"), "drain": ("d", "drain"),
               "source": ("s", "source"), "body": ("b", "body")}
    value: object = ""
    for key in aliases[role]:
        if key in pins:
            value = pins[key]
            break
    return _text(value, f"{role} net", allow_empty=role == "body")


def _normalize_record(name: object, raw: object) -> RowMemberSpec:
    device = _text(name, "record device")
    if not isinstance(raw, Mapping):
        raise RowProgramError(f"record {device!r} must be a mapping")
    kind, default_oxide = _normalize_kind(raw.get("kind", raw.get("device_class", "")))
    raw_nf = raw.get("nf", raw.get("M", 1))
    if isinstance(raw_nf, bool) or not isinstance(raw_nf, (int, float)):
        raise RowProgramError(f"{device} nf must be an exact integer or integral parser number")
    numeric_nf = float(raw_nf)
    if not math.isfinite(numeric_nf) or not numeric_nf.is_integer():
        raise RowProgramError(f"{device} nf must be integral")
    nf = _positive(int(numeric_nf), f"{device} nf", MAX_FINGERS)
    oxide = raw.get("oxide", default_oxide)
    return RowMemberSpec(
        device=device,
        kind=kind,
        width_nm=_nm_from_record(raw, "W"),
        length_nm=_nm_from_record(raw, "L"),
        fingers=nf,
        gate_net=_pin(raw, "gate"),
        drain_net=_pin(raw, "drain"),
        source_net=_pin(raw, "source"),
        body_net=_pin(raw, "body"),
        oxide=_text(oxide, "oxide").lower(),
        dummy_for=_text(raw.get("dummy_for", ""), "dummy_for", allow_empty=True),
    )


def _records_from_cell(cell: object, *, native_mos: bool = False) -> Mapping[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    devices = _bounded(getattr(cell, "devices", ()) or (), MAX_PROGRAM_MEMBERS, "cell devices")
    native_specs = None
    if native_mos and any(getattr(item, "device_class", None) == "native" for item in devices):
        from clay.pdk.sky130_native_geometry import native_cell_geometry_specs
        native_specs = native_cell_geometry_specs(cell)
    for item in devices:
        name = _text(getattr(item, "instance_name", ""), "cell device name")
        sizing = getattr(item, "sizing", {}) or {}
        pins = getattr(item, "pins", {}) or {}
        record = {
            "kind": getattr(item, "device_class", ""),
            "W": sizing.get("W"),
            "L": sizing.get("L"),
            "M": sizing.get("M", 1),
            "pins": pins,
            "dummy_for": getattr(item, "dummy_for", ""),
        }
        if native_specs is not None and getattr(item, "device_class", None) == "native":
            spec = native_specs[name]
            if spec['family'] == 'mos':
                # W in the native source is total per PCell; RowMemberSpec is
                # per finger.  Use the existing PDK contract, including all
                # nf/m/mult occurrences, rather than reparsing X-card units.
                record.update(kind=spec['model'], W_nm=spec['w_nm'] // spec['nf'],
                              L_nm=spec['l_nm'], nf=spec['occurrences'],
                              native_total_w_nm=spec['w_nm'])
        if hasattr(item, "oxide"):
            record["oxide"] = getattr(item, "oxide")
        result[name] = record
    return MappingProxyType(result)


def _normalize_records(
    cell: object, device_records: Mapping[str, Mapping[str, object]] | None,
    *, native_mos: bool = False,
) -> tuple[RowMemberSpec, ...]:
    source_records = _records_from_cell(cell, native_mos=native_mos)
    raw = device_records if device_records is not None else source_records
    if not isinstance(raw, Mapping):
        raise RowProgramError("device_records must be a mapping")
    if device_records is not None:
        native_names = {name for name, value in source_records.items()
                        if 'native_total_w_nm' in value}
        if native_names and not native_names <= set(raw):
            raise RowProgramError("circuit/native source MOS inventory differs")
        bound_records: dict[object, object] = {}
        for name, record in raw.items():
            source_record = source_records.get(name)
            if not isinstance(record, Mapping):
                bound_records[name] = record
                continue
            if source_record is not None:
                circuit_is_mos = _record_is_mos(record)
                source_is_mos = _record_is_mos(source_record)
                if circuit_is_mos != source_is_mos:
                    raise RowProgramError(
                        f"{name} circuit/source MOS classifications differ"
                    )
            if source_record is not None and circuit_is_mos and source_is_mos:
                if 'native_total_w_nm' in source_record:
                    if (_nm_from_record(record, 'W') != source_record['native_total_w_nm']
                            or _nm_from_record(record, 'L') != source_record['L_nm']
                            or any(_pin(record, role) != _pin(source_record, role)
                                   for role in ('gate', 'drain', 'source', 'body'))
                            or str(record.get('model', '')).casefold() !=
                               str(source_record['kind']).casefold()):
                        raise RowProgramError(f"{name} circuit/native source geometry or nets differ")
                    # X-card nf/m are owned by the validated PDK source contract;
                    # the legacy SPICE records are only an independent cross-check.
                    bound_records[name] = source_record
                    continue
                circuit_kind, _circuit_oxide = _normalize_kind(
                    record.get("kind", record.get("device_class", ""))
                )
                source_kind, _source_oxide = _normalize_kind(
                    source_record.get(
                        "kind", source_record.get("device_class", "")
                    )
                )
                if circuit_kind != source_kind:
                    raise RowProgramError(
                        f"{name} circuit/source MOS polarities differ"
                    )
                record = dict(record)
                record["kind"] = source_record["kind"]
                if "oxide" in source_record:
                    record["oxide"] = source_record["oxide"]
            bound_records[name] = record
        raw = MappingProxyType(bound_records)
    items = _bounded(raw.items(), MAX_PROGRAM_MEMBERS, "device_records")
    normalized = tuple(
        sorted(
            (
                _normalize_record(name, record)
                for name, record in items
                if _record_is_mos(record)
            ),
            key=lambda item: item.device,
        )
    )

    # A first-class @fold changes the physical finger plan while preserving the
    # electrical total width.  The device-cell and route-stamp paths already
    # consume this declaration; row-final must see the SAME plan or a declared
    # common centroid is rejected against the stale schematic M=1 record.
    from clay.layout.annotations import Fold
    from clay.layout.extract.fold import DEFAULT_H_MAX_UM

    raw_annotations = getattr(cell, "_layout_annotations", {}) or {}
    folded: list[RowMemberSpec] = []
    for member in normalized:
        declarations = [
            annotation
            for annotation in raw_annotations.get(member.device, ())
            if isinstance(annotation, Fold)
        ]
        if not declarations:
            folded.append(member)
            continue
        if len(declarations) != 1:
            raise RowProgramError(
                f"{member.device} has multiple @fold declarations"
            )
        h_max_um = declarations[0].h_max_um
        if h_max_um is None:
            h_max_um = DEFAULT_H_MAX_UM
        h_max_nm_float = float(h_max_um) * 1000.0
        h_max_nm = round(h_max_nm_float)
        if (
            not math.isfinite(h_max_nm_float)
            or h_max_nm <= 0
            or abs(h_max_nm_float - h_max_nm) > 1e-6
        ):
            raise RowProgramError(
                f"{member.device} @fold height must normalize to exact positive nm"
            )
        total_width_nm = member.width_nm * member.fingers
        planned_fingers = max(
            member.fingers,
            (total_width_nm + int(h_max_nm) - 1) // int(h_max_nm),
        )
        if total_width_nm % planned_fingers:
            raise RowProgramError(
                f"{member.device} @fold plan cannot conserve exact integer-nm "
                f"width {total_width_nm} across {planned_fingers} fingers"
            )
        folded.append(
            replace(
                member,
                width_nm=total_width_nm // planned_fingers,
                fingers=planned_fingers,
            )
        )
    return tuple(folded)


def _group_map(cell: object) -> dict[str, tuple[str, ...]]:
    raw = getattr(cell, "group_members", None)
    if raw is None:
        from clay.frontend.craft_infer import group_members
        raw = group_members(cell)
    if not isinstance(raw, Mapping):
        raise RowProgramError("cell group_members must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for group, members in _bounded(raw.items(), MAX_PROGRAM_ENTRIES, "group map"):
        g = _text(group, "group label")
        names = tuple(sorted(_exact_text_tuple(members, f"group {g} members", MAX_GROUP_MEMBERS)))
        result[g] = names
    return result


def _intent_tuple(intent: object, field: str) -> tuple[object, ...]:
    return _bounded(getattr(intent, field, ()) or (), MAX_PROGRAM_ENTRIES, field)


def _guard_ring_candidates(
    intent: object,
    group_devices: tuple[str, ...],
) -> tuple[str, ...]:
    """Devices co-owned by the smallest declared ring containing the row group."""

    active = set(group_devices)
    islands: list[tuple[str, ...]] = []
    for ring in _intent_tuple(intent, "guard_rings"):
        raw_members = getattr(ring, "members", ()) or ()
        members = tuple(sorted(_exact_text_tuple(
            raw_members, "guard-ring members", MAX_GROUP_MEMBERS
        )))
        if active and active.issubset(members):
            islands.append(members)
    if not islands:
        return ()
    island = min(islands, key=lambda members: (len(members), members))
    return tuple(device for device in island if device not in active)


def _body_tap_delegation(
    intent: object,
    consumed_devices: tuple[str, ...],
    by_name: Mapping[str, RowMemberSpec],
) -> RowBodyTapDelegation | None:
    """Delegate only to one exact same-net outer tap-ring declaration."""

    from clay.layout.intent import GuardRingIntent

    raw_rings = getattr(intent, "guard_rings", ())
    if type(raw_rings) is not tuple:
        raise TypeError("intent guard_rings must be an exact tuple")
    expected_members = tuple(sorted(consumed_devices))
    candidates: list[RowBodyTapDelegation] = []
    for raw in raw_rings:
        if type(raw) is not GuardRingIntent:
            raise TypeError("intent guard_rings must contain exact GuardRingIntent values")
        ring = GuardRingIntent(
            members=raw.members,
            well=raw.well,
            tie_net=raw.tie_net,
            hug=raw.hug,
            role=raw.role,
            enclose=raw.enclose,
        )
        if tuple(sorted(ring.members)) != expected_members:
            continue
        specs = tuple(by_name.get(device) for device in expected_members)
        if any(spec is None for spec in specs):
            continue
        if (
            ring.well != "n"
            or ring.hug != "tight"
            or ring.role != "tap"
            or ring.enclose
            or any(spec.kind != "pmos" for spec in specs)
            or any(spec.body_net != spec.source_net for spec in specs)
            or any(spec.body_net != ring.tie_net for spec in specs)
        ):
            continue
        candidates.append(RowBodyTapDelegation(
            declaration_sha256=ring.sha256,
            members=ring.members,
            well=ring.well,
            tie_net=ring.tie_net,
            hug=ring.hug,
            role=ring.role,
            enclose=ring.enclose,
        ))
    if len(candidates) > 1:
        raise RowProgramError(
            "multiple exact guard-ring declarations claim one row body tap"
        )
    return candidates[0] if candidates else None


def _declaration_ref(kind: str, group: str, rows: int = 0, suffix: str = "") -> str:
    parts = [kind, group]
    if rows:
        parts.append(str(rows))
    if suffix:
        parts.append(suffix)
    return ":".join(parts)


def _ordered_active(members: Iterable[RowMemberSpec], order: str) -> tuple[RowMemberSpec, ...]:
    items = tuple(members)
    if order.strip().lower() == "ratio":
        return tuple(sorted(items, key=lambda m: (m.gate_net != m.drain_net, m.device)))
    return tuple(sorted(items, key=lambda m: m.device))


def _residual(code: str, group: str, ref: str, devices: Iterable[str], detail: str) -> RowProgramResidual:
    return RowProgramResidual(
        code=code, group=group, declaration_refs=(ref,), devices=tuple(devices), detail=detail
    )


def _validate_active(
    group: str,
    names: tuple[str, ...],
    by_name: Mapping[str, RowMemberSpec],
    structure: RowStructureKind,
    ref: str,
) -> tuple[tuple[RowMemberSpec, ...] | None, RowProgramResidual | None]:
    if not names:
        return None, _residual("GROUP_UNKNOWN", group, ref, (), "group has no topology members")
    missing = tuple(name for name in names if name not in by_name)
    if missing:
        return None, _residual(
            "MEMBER_RECORD_MISSING", group, ref, missing, "topology member lacks a normalized MOS record"
        )
    members = tuple(by_name[name] for name in names)
    if structure is RowStructureKind.COMMON_CENTROID and len(members) != 2:
        return None, _residual(
            "COMMON_CENTROID_MEMBER_COUNT", group, ref, names,
            "common-centroid requires exactly two active devices",
        )
    geometries = {
        (m.kind, m.width_nm, m.length_nm, m.oxide)
        if structure is RowStructureKind.SHARED_OD
        else (m.kind, m.width_nm, m.length_nm, m.fingers, m.oxide)
        for m in members
    }
    if len(geometries) != 1:
        return None, _residual(
            "MEMBER_GEOMETRY_MISMATCH", group, ref, names,
            "one rigid row requires compatible polarity, W, L, multiplicity and oxide",
        )
    source_nets = {m.source_net for m in members}
    if len(source_nets) != 1:
        return None, _residual(
            "SOURCE_NET_MISMATCH", group, ref, names,
            "members do not share one source diffusion net",
        )
    if structure is RowStructureKind.SHARED_OD:
        gates = {m.gate_net for m in members}
        if len(gates) != 1:
            return None, _residual(
                "GATE_NET_MISMATCH", group, ref, names,
                "shared-OD member builder requires one shared gate net",
            )
    return members, None


def _dummy_matches(
    member: RowMemberSpec,
    active: tuple[RowMemberSpec, ...],
    group: str,
    tie: str,
) -> bool:
    first = active[0]
    if member.dummy_for != group:
        return False
    if (member.kind, member.width_nm, member.length_nm, member.oxide) != (
        first.kind, first.width_nm, first.length_nm, first.oxide
    ):
        return False
    if tie == "rail":
        return member.gate_net == member.drain_net == member.source_net == first.source_net
    if tie == "gate_off":
        return (
            member.gate_net == member.source_net == first.source_net
            and member.drain_net in {item.drain_net for item in active}
        )
    return False


def _resolve_dummies(
    *,
    group: str,
    rows: int,
    structure: RowStructureKind,
    active: tuple[RowMemberSpec, ...],
    all_members: tuple[RowMemberSpec, ...],
    declarations: tuple[object, ...],
    topology_candidates: tuple[str, ...],
) -> tuple[tuple[RowMemberSpec, ...], tuple[DummyEndcapObligation, ...], tuple[str, ...], RowProgramResidual | None]:
    enabled = []
    for item in declarations:
        if _text(getattr(item, "target", ""), "dummy target", allow_empty=True) != group:
            continue
        share = getattr(item, "share_diffusion", True)
        if type(share) is not bool:
            raise RowProgramError("dummy share_diffusion must be an exact bool")
        if share:
            enabled.append(item)
    if not enabled:
        return (), (), (), None
    if len(enabled) != 1:
        ref = _declaration_ref("dummy_flank", group)
        return (), (), (), _residual(
            "DUMMY_DECLARATION_CONFLICT", group, ref, (),
            "one row entry accepts one exact dummy-flank declaration",
        )
    decl = enabled[0]
    per_end = getattr(decl, "per_end", 1)
    if type(per_end) is not int or per_end < 1:
        raise RowProgramError("dummy per_end must be a positive exact int")
    tie = _text(getattr(decl, "tie", ""), "dummy tie").lower()
    base = f"dummy_flank:{group}:{tie}:{per_end}"
    topology_set = set(topology_candidates)
    candidates = tuple(sorted(
        (
            item for item in all_members
            if (item.dummy_for == group or item.device in topology_set)
            and _dummy_matches_relaxed(item, active, group, tie)
        ),
        key=lambda item: item.device,
    ))
    if not candidates:
        return (), (), (), _residual(
            "DUMMY_MEMBER_MISSING", group, base, (),
            "declared dummy flank has no topology-matched normalized dummy device",
        )
    expected_each = per_end * rows
    if structure is RowStructureKind.SHARED_OD:
        actual_each = sum(item.fingers // 2 for item in candidates if item.fingers % 2 == 0)
        valid_counts = all(item.fingers % 2 == 0 for item in candidates)
    elif tie == "rail":
        divisor = 2 * rows
        actual_each = sum(item.fingers // divisor for item in candidates if item.fingers % divisor == 0) * rows
        valid_counts = all(item.fingers % divisor == 0 for item in candidates)
    else:
        actual_each = sum(item.fingers for item in candidates) // 2
        valid_counts = all(item.fingers % rows == 0 for item in candidates)
        # A drain-exposed CC dummy is placed at one reflected end per row.
        expected_each = per_end * rows
    if not valid_counts or actual_each != expected_each:
        return (), (), (), _residual(
            "DUMMY_COUNT_MISMATCH", group, base,
            tuple(item.device for item in candidates),
            f"declared per-end row count {expected_each} != materializable count {actual_each}",
        )
    left_ref, right_ref = f"{base}:left", f"{base}:right"
    obligations = (
        DummyEndcapObligation(side="left", declaration_ref=left_ref, count=expected_each),
        DummyEndcapObligation(side="right", declaration_ref=right_ref, count=expected_each),
    )
    return candidates, obligations, (base, left_ref, right_ref), None


def _dummy_matches_relaxed(
    member: RowMemberSpec,
    active: tuple[RowMemberSpec, ...],
    group: str,
    tie: str,
) -> bool:
    """Electrical/geometry half of dummy matching.

    Association is independently proven either by normalized ``dummy_for`` or by
    membership in the same declared guard-ring island.  This function therefore
    never guesses from a device name.
    """

    if member.dummy_for == group:
        return _dummy_matches(member, active, group, tie)
    first = active[0]
    if (member.kind, member.width_nm, member.length_nm, member.oxide) != (
        first.kind, first.width_nm, first.length_nm, first.oxide
    ):
        return False
    if tie == "rail":
        return member.gate_net == member.drain_net == member.source_net == first.source_net
    if tie == "gate_off":
        return (
            member.gate_net == member.source_net == first.source_net
            and member.drain_net in {item.drain_net for item in active}
        )
    return False


def _atom(
    *,
    atom_id: str,
    index: int,
    member: RowMemberSpec,
    ref: str,
    kind: RowAtomKind,
    match_key: str,
    left_net: str,
    left_terminal: BoundaryTerminal,
    right_net: str,
    right_terminal: BoundaryTerminal,
    dummy_ref: str | None = None,
) -> RowAtom:
    atom_width = max(2, member.length_nm)
    span = max(1, min(atom_width - 1, member.length_nm // 4))
    refs = (ref,) if dummy_ref is None else (ref, dummy_ref)
    return RowAtom(
        atom_id=atom_id,
        sequence_index=index,
        width_nm=atom_width,
        height_nm=member.width_nm,
        left=DiffusionBoundary(
            net=left_net, terminal=left_terminal,
            share_span_nm=span, declaration_ref=ref,
        ),
        right=DiffusionBoundary(
            net=right_net, terminal=right_terminal,
            share_span_nm=span, declaration_ref=ref,
        ),
        match_key=match_key,
        kind=kind,
        declaration_refs=refs,
        dummy_obligation_ref=dummy_ref,
    )


def _row_atoms(
    *,
    group: str,
    row: int,
    active: tuple[tuple[RowMemberSpec, int, int], ...],
    dummies: tuple[RowMemberSpec, ...],
    per_end: int,
    ref: str,
    left_ref: str | None,
    right_ref: str | None,
    row_count: int,
    structure: RowStructureKind,
) -> tuple[RowAtom, ...]:
    specs: list[
        tuple[
            str, RowMemberSpec, RowAtomKind, str, str | None,
            str, BoundaryTerminal, str, BoundaryTerminal,
        ]
    ] = []
    if dummies:
        rail_dummies = all(item.gate_net == item.drain_net == item.source_net for item in dummies)
        if rail_dummies:
            left_expanded = tuple(
                item for item in dummies for _ in range(item.fingers // (2 * row_count))
            )
        else:
            left_owner = active[0][0]
            if structure is RowStructureKind.COMMON_CENTROID and row_count == 1:
                right_owner = next(
                    member
                    for member, _local_index, _weight in active
                    if member.device != left_owner.device
                )
            else:
                right_owner = active[-1][0]
            left_expanded = tuple(
                item
                for item in dummies
                if item.drain_net == left_owner.drain_net
                for _ in range(item.fingers // row_count)
            )
        for ordinal, member in enumerate(left_expanded):
            left_net = member.drain_net if ordinal == 0 and not rail_dummies else member.source_net
            match_key = (
                f"cc:dummy-flank-pair:{ordinal}"
                if structure is RowStructureKind.COMMON_CENTROID
                and row_count == 1
                and not rail_dummies
                else f"dummy:{member.device}:{ordinal}"
            )
            specs.append((
                f"dummy:left:{member.device}:{ordinal}", member, RowAtomKind.DUMMY,
                match_key, left_ref,
                left_net,
                BoundaryTerminal.DRAIN if left_net == member.drain_net and not rail_dummies
                else BoundaryTerminal.SOURCE,
                member.source_net, BoundaryTerminal.SOURCE,
            ))
    previous_right_net: str | None = None
    for signal_position, (member, local_index, weight) in enumerate(active):
        if structure is RowStructureKind.SHARED_OD:
            left_net = member.source_net if previous_right_net is None else previous_right_net
            right_net = member.drain_net if local_index % 2 == 0 else member.source_net
            left_terminal = (
                BoundaryTerminal.DRAIN if left_net == member.drain_net
                else BoundaryTerminal.SOURCE
            )
            right_terminal = (
                BoundaryTerminal.DRAIN if right_net == member.drain_net
                else BoundaryTerminal.SOURCE
            )
            previous_right_net = right_net
        else:
            # A common-centroid atom is one electrically closed same-device
            # finger PAIR: source at both exterior edges, drain internal to the
            # pair.  This is the smallest point-symmetric unit for every even nf;
            # individual fingers are not point-paired for nf=2 mod 4.
            left_net = right_net = member.source_net
            left_terminal = right_terminal = BoundaryTerminal.SOURCE
        label = (
            f"active:{member.device}:pair:{local_index // 2}"
            if weight == 2 else f"active:{member.device}:{local_index}"
        )
        specs.append((
            label, member,
            RowAtomKind.ACTIVE,
            "cc:matched-pair-unit" if weight == 2 else member.device,
            None,
            left_net, left_terminal, right_net, right_terminal,
        ))
    if dummies:
        if rail_dummies:
            right_expanded = tuple(
                item for item in reversed(dummies)
                for _ in range(item.fingers // (2 * row_count))
            )
        else:
            right_expanded = tuple(
                item for item in reversed(dummies) if item.drain_net == right_owner.drain_net
                for _ in range(item.fingers // row_count)
            )
        for ordinal, member in enumerate(right_expanded):
            is_outer = ordinal == len(right_expanded) - 1 and not rail_dummies
            reflected_ordinal = len(right_expanded) - 1 - ordinal
            match_key = (
                f"cc:dummy-flank-pair:{reflected_ordinal}"
                if structure is RowStructureKind.COMMON_CENTROID
                and row_count == 1
                and not rail_dummies
                else f"dummy:{member.device}:{reflected_ordinal}"
            )
            specs.append((
                f"dummy:right:{member.device}:{ordinal}", member, RowAtomKind.DUMMY,
                match_key,
                right_ref,
                member.source_net, BoundaryTerminal.SOURCE,
                member.drain_net if is_outer else member.source_net,
                BoundaryTerminal.DRAIN if is_outer else BoundaryTerminal.SOURCE,
            ))
    return tuple(
        _atom(
            atom_id=f"{group}:r{row}:{label}", index=index, member=member,
            ref=ref, kind=kind, match_key=match_key,
            dummy_ref=dummy_ref, left_net=left_net, left_terminal=left_terminal,
            right_net=right_net, right_terminal=right_terminal,
        )
        for index, (
            label, member, kind, match_key, dummy_ref,
            left_net, left_terminal, right_net, right_terminal,
        ) in enumerate(specs)
    )


def _compile_macro(
    *,
    group: str,
    structure: RowStructureKind,
    row_count: int,
    active: tuple[RowMemberSpec, ...],
    dummies: tuple[RowMemberSpec, ...],
    dummy_obligations: tuple[DummyEndcapObligation, ...],
    declaration_refs: tuple[str, ...],
) -> RowMacro:
    ref = declaration_refs[0]
    obligations = {item.side.value: item.declaration_ref for item in dummy_obligations}
    if dummies:
        per_end = dummy_obligations[0].count // row_count
    else:
        per_end = 0
    active_rows: list[tuple[tuple[RowMemberSpec, int, int], ...]] = []
    if structure is RowStructureKind.SHARED_OD:
        flat = tuple(
            (member, local_index, 1)
            for member in active
            for local_index in range(member.fingers)
        )
        per_row = len(flat) // row_count
        active_rows = [
            flat[row * per_row:(row + 1) * per_row]
            for row in range(row_count)
        ]
    elif row_count == 2:
        a, b = active
        labels = tuple(
            label
            for pair_index in range(a.fingers // 2)
            for label in ((a if pair_index % 2 == 0 else b),) * 2
        )
        counts = {a.device: 0, b.device: 0}
        first_fingers: list[tuple[RowMemberSpec, int]] = []
        for member in labels:
            first_fingers.append((member, counts[member.device]))
            counts[member.device] += 1
        second_fingers: list[tuple[RowMemberSpec, int]] = []
        for member in (b if item is a else a for item in labels):
            second_fingers.append((member, counts[member.device]))
            counts[member.device] += 1
        active_rows = [
            tuple((row[index][0], row[index][1], 2) for index in range(0, len(row), 2))
            for row in (first_fingers, second_fingers)
        ]
    else:
        a, b = active
        pair_count = a.fingers // 2
        if pair_count % 2 == 0:
            pair_labels = tuple(
                label
                for _ in range(pair_count // 2)
                for label in (a, b, b, a)
            )
        else:
            first_half = tuple(a if index % 2 == 0 else b for index in range(pair_count))
            pair_labels = first_half + tuple(reversed(first_half))
        labels = tuple(label for member in pair_labels for label in (member, member))
        counts = {a.device: 0, b.device: 0}
        one_row_fingers: list[tuple[RowMemberSpec, int]] = []
        for member in labels:
            one_row_fingers.append((member, counts[member.device]))
            counts[member.device] += 1
        active_rows = [tuple(
            (one_row_fingers[index][0], one_row_fingers[index][1], 2)
            for index in range(0, len(one_row_fingers), 2)
        )]

    rows: list[RowString] = []
    for row in range(row_count):
        atoms = _row_atoms(
            group=group, row=row, active=active_rows[row], dummies=dummies,
            per_end=per_end, ref=ref,
            left_ref=obligations.get("left"), right_ref=obligations.get("right"),
            row_count=row_count, structure=structure,
        )
        rows.append(RowString(
            row_id=f"{group}:row:{row}", row_index=row, atoms=atoms,
            declaration_refs=declaration_refs,
        ))
    macro = compile_row_macro(
        macro_id=f"row:{group}", structure_kind=structure, rows=rows,
        row_pitch_nm=max(item.width_nm for item in active),
        declaration_refs=declaration_refs,
        dummy_endcap_obligations=dummy_obligations,
    )
    assert macro is not None
    return macro


def compile_row_program(
    cell: object,
    intent: object,
    tech: object,
    *,
    device_records: Mapping[str, Mapping[str, object]] | None = None,
) -> RowProgram:
    """Compile row structures from normalized topology and declarative intent.

    A common-centroid declaration is the stronger structural authority for its
    group and supersedes the overlapping ``@array(... ABBA ...)`` declaration.
    Invalid declarations become named residuals; no free-device fallback is
    silently inserted into the program.
    """

    cell_name = _text(
        getattr(cell, "cell_name", getattr(cell, "local_name", "")), "cell name"
    )
    tech_id = _text(getattr(tech, "pdk_id", getattr(tech, "name", "")), "tech identity")
    members = _normalize_records(cell, device_records, native_mos=(tech_id == 'sky130'))
    by_name = {item.device: item for item in members}
    groups = _group_map(cell)
    raw_cc_declarations = getattr(intent, "common_centroids", ())
    if type(raw_cc_declarations) is not tuple:
        raise TypeError("intent common_centroids must be an exact tuple")
    from clay.layout.intent import (
        replay_array_intent,
        replay_common_centroid_intent,
    )
    cc_decls = tuple(
        replay_common_centroid_intent(declaration)
        for declaration in raw_cc_declarations
    )

    raw_array_declarations = getattr(intent, "arrays", ())
    if type(raw_array_declarations) is not tuple:
        raise TypeError("intent arrays must be an exact tuple")
    array_decls = tuple(
        replay_array_intent(declaration)
        for declaration in raw_array_declarations
    )
    native_groups = {decl.group for decl in array_decls if decl.style == "native_diffusion"}
    conflicting_groups = native_groups & (
        {decl.group for decl in cc_decls}
        | {decl.group for decl in array_decls if decl.style == "shared_diffusion"}
    )
    if conflicting_groups:
        raise RowProgramError(
            "native diffusion conflicts with another row owner: "
            + ", ".join(sorted(conflicting_groups))
        )
    dummy_decls = _intent_tuple(intent, "dummy_flanks")
    from clay.layout.spatial.intent_projection import (
        replay_discrete_mosaic_inventory,
    )
    mosaic_decls = replay_discrete_mosaic_inventory(
        getattr(intent, "discrete_mosaics", ()) or ()
    )
    from clay.layout.spatial.mosaic_boundary_joins import replay_mosaic_boundary_join_inventory
    boundary_intents = replay_mosaic_boundary_join_inventory(
        getattr(intent, "mosaic_boundary_joins", ()), mosaics=mosaic_decls, members=by_name)
    boundaries_by_mosaic = {item.source_mosaic_sha256: item for item in boundary_intents}
    from clay.layout.spatial.mosaic_guest_slot import replay_mosaic_guest_slot_inventory
    guest_slots = replay_mosaic_guest_slot_inventory(getattr(intent, "mosaic_guest_slots", ()))
    slots_by_mosaic = {row.source_mosaic_sha256: row for row in guest_slots}
    if not set(slots_by_mosaic) <= {row.sha256 for row in mosaic_decls}:
        raise RowProgramError("guest slot has no exact mosaic source declaration")
    mosaic_claimed = {
        member for declaration in mosaic_decls for member in declaration.members
    }

    selected: dict[str, tuple[RowStructureKind, int, str, str, str, str]] = {}
    residuals: list[RowProgramResidual] = []
    cc_options: dict[str, set[tuple[int, str, str, str]]] = {}
    for decl in cc_decls:
        group = _text(getattr(decl, "group", ""), "common-centroid group")
        rows = getattr(decl, "rows", 2)
        if type(rows) is not int or rows not in (1, 2):
            ref = _declaration_ref("common_centroid", group, rows if type(rows) is int else 0)
            residuals.append(_residual(
                "UNSUPPORTED_ROW_COUNT", group, ref, (), "rigid CC builder supports rows 1 or 2"
            ))
            continue
        orientation = getattr(decl, "orientation", "R0")
        if type(orientation) is not str or orientation not in (
            "R0", "R90", "R180", "R270"
        ):
            ref = _declaration_ref("common_centroid", group, rows, "invalid-orientation")
            residuals.append(_residual(
                "UNSUPPORTED_COMMON_CENTROID_ORIENTATION", group, ref, (),
                "rigid CC builder supports R0/R90/R180/R270",
            ))
            continue
        first_member = getattr(decl, "first_member", "")
        if type(first_member) is not str:
            ref = _declaration_ref("common_centroid", group, rows, "invalid-first-member")
            residuals.append(_residual(
                "INVALID_COMMON_CENTROID_FIRST_MEMBER", group, ref, (),
                "common-centroid first_member must be an exact string",
            ))
            continue
        ref = _declaration_ref(
            "common_centroid", group, rows,
            ":".join(part for part in (
                orientation if orientation != "R0" else "",
                f"first-member={first_member}" if first_member else "",
            ) if part),
        )
        cc_options.setdefault(group, set()).add(
            (rows, orientation, first_member, ref)
        )
    cc_groups: set[str] = set()
    for group, options in sorted(cc_options.items()):
        cc_groups.add(group)
        group_devices = set(groups.get(group, ()))
        if group_devices and group_devices <= mosaic_claimed:
            continue
        if group_devices & mosaic_claimed:
            refs = tuple(sorted(ref for _rows, _orientation, _first_member, ref in options))
            residuals.append(RowProgramResidual(
                code="MOSAIC_STRUCTURAL_OVERLAP",
                group=group,
                declaration_refs=refs,
                devices=tuple(sorted(group_devices & mosaic_claimed)),
                detail="a projected mosaic only partially covers an existing row owner",
            ))
            continue
        if len(options) != 1:
            refs = tuple(
                sorted(ref for _rows, _orientation, _first_member, ref in options)
            )
            residuals.append(RowProgramResidual(
                code="STRUCTURAL_DECLARATION_CONFLICT", group=group,
                declaration_refs=refs, devices=(),
                detail="distinct common-centroid declarations name one group",
            ))
            continue
        rows, orientation, first_member, ref = next(iter(options))
        selected[group] = (
            RowStructureKind.COMMON_CENTROID,
            rows,
            "common_centroid",
            orientation,
            first_member,
            ref,
        )
    array_options: dict[str, set[tuple[int, str, str]]] = {}
    for decl in array_decls:
        group = _text(getattr(decl, "group", ""), "array group")
        style = _text(getattr(decl, "style", ""), "array style", allow_empty=True).lower()
        if style != "shared_diffusion":
            continue
        if group in cc_groups:
            continue
        rows = getattr(decl, "rows", 1)
        if type(rows) is not int or not 1 <= rows <= 256:
            raise RowProgramError("array rows must be an exact integer in [1, 256]")
        order = _text(getattr(decl, "order", ""), "array order", allow_empty=True)
        if order.strip().lower() not in ("", "ratio"):
            ref = _declaration_ref("array", group, rows, order)
            residuals.append(_residual(
                "UNSUPPORTED_SHARED_OD_ORDER", group, ref, (),
                "shared-OD rigid builder accepts ratio or empty order",
            ))
            continue
        ref = _declaration_ref("array", group, rows, order or "none")
        array_options.setdefault(group, set()).add((rows, order, ref))
    for group, options in sorted(array_options.items()):
        group_devices = set(groups.get(group, ()))
        if group_devices and group_devices <= mosaic_claimed:
            continue
        if group_devices & mosaic_claimed:
            refs = tuple(sorted(ref for _rows, _order, ref in options))
            residuals.append(RowProgramResidual(
                code="MOSAIC_STRUCTURAL_OVERLAP",
                group=group,
                declaration_refs=refs,
                devices=tuple(sorted(group_devices & mosaic_claimed)),
                detail="a projected mosaic only partially covers an existing row owner",
            ))
            continue
        if len(options) != 1:
            refs = tuple(sorted(ref for _rows, _order, ref in options))
            residuals.append(RowProgramResidual(
                code="STRUCTURAL_DECLARATION_CONFLICT", group=group,
                declaration_refs=refs, devices=(),
                detail="distinct shared-OD declarations name one group",
            ))
            continue
        rows, order, ref = next(iter(options))
        selected[group] = (
            RowStructureKind.SHARED_OD, rows, order, "R0", "", ref
        )

    entries: list[RowProgramEntry] = []
    consumed: set[str] = set()
    for declaration in mosaic_decls:
        missing = tuple(sorted(set(declaration.members) - set(by_name)))
        if missing:
            residuals.append(_residual(
                "MOSAIC_MEMBER_RECORD_MISSING",
                declaration.mosaic_id,
                f"discrete_mosaic:{declaration.mosaic_id}",
                missing,
                "projected mosaic owner lacks a normalized MOS record",
            ))
            continue
        expected_counts = {
            member: by_name[member].fingers for member in declaration.members
        }
        actual_counts = {
            member: sum(
                occurrence.logical_owner == member
                for occurrence in declaration.occurrences
            )
            for member in declaration.members
        }
        if actual_counts != expected_counts:
            residuals.append(_residual(
                "MOSAIC_OCCURRENCE_COUNT_MISMATCH",
                declaration.mosaic_id,
                f"discrete_mosaic:{declaration.mosaic_id}",
                declaration.members,
                f"projected counts {actual_counts!r} != topology {expected_counts!r}",
            ))
            continue
        try:
            macro = _compile_mosaic_macro(
                declaration,
                by_name=by_name,
                tech=tech,
                guest_slot=slots_by_mosaic.get(declaration.sha256),
                boundary_intent=boundaries_by_mosaic.get(declaration.sha256),
            )
        except (RowProgramError, RowCalculusError) as exc:
            residuals.append(_residual(
                "MOSAIC_ROW_CALCULUS_REJECTED",
                declaration.mosaic_id,
                f"discrete_mosaic:{declaration.mosaic_id}",
                declaration.members,
                str(exc),
            ))
            continue
        entry = RowProgramEntry(
            entry_id=f"row:{declaration.mosaic_id}",
            group=declaration.mosaic_id,
            owner=f"row:{declaration.mosaic_id}",
            structure_kind=RowStructureKind.DISCRETE_MOSAIC,
            rows=len(declaration.rows),
            order="row_major",
            active_devices=declaration.members,
            dummy_devices=(),
            declaration_refs=macro.declaration_refs,
            macro=macro,
        )
        entries.append(entry)
        consumed.update(declaration.members)
    for group, (
        structure, rows, order, orientation, first_member, ref
    ) in sorted(selected.items()):
        active, issue = _validate_active(group, groups.get(group, ()), by_name, structure, ref)
        if issue is not None:
            residuals.append(issue)
            continue
        assert active is not None
        active = _ordered_active(active, order)
        if structure is RowStructureKind.COMMON_CENTROID and first_member:
            if first_member not in {item.device for item in active}:
                residuals.append(_residual(
                    "COMMON_CENTROID_FIRST_MEMBER_NOT_IN_GROUP",
                    group,
                    ref,
                    tuple(item.device for item in active),
                    "common-centroid first_member must name one active group member",
                ))
                continue
            active = tuple(sorted(
                active,
                key=lambda item: (item.device != first_member, item.device),
            ))
        if structure is RowStructureKind.COMMON_CENTROID and (
            active[0].fingers < 2 or active[0].fingers % 2 != 0
        ):
            residuals.append(_residual(
                "COMMON_CENTROID_FINGER_PAIRING", group, ref,
                tuple(item.device for item in active),
                "common-centroid requires equal even finger counts of at least two",
            ))
            continue
        if (
            structure is RowStructureKind.COMMON_CENTROID
            and rows == 1
            and active[0].fingers % 4 != 0
        ):
            residuals.append(_residual(
                "COMMON_CENTROID_ONE_ROW_DIVISIBILITY", group, ref,
                tuple(item.device for item in active),
                "one-row common-centroid requires per-device fingers divisible by four",
            ))
            continue
        if structure is RowStructureKind.SHARED_OD:
            total_fingers = sum(item.fingers for item in active)
            if total_fingers % rows:
                residuals.append(_residual(
                    "SHARED_OD_ROW_DIVISIBILITY", group, ref,
                    tuple(item.device for item in active),
                    f"total fingers {total_fingers} are not divisible by rows {rows}",
                ))
                continue
            odd_interior = tuple(item.device for item in active[:-1] if item.fingers % 2)
            if odd_interior:
                residuals.append(_residual(
                    "SHARED_OD_DRAIN_BOUNDARY", group, ref, odd_interior,
                    "an odd interior member ends on drain and cannot abut the next source",
                ))
                continue
        dummies, obligations, dummy_refs, issue = _resolve_dummies(
            group=group, rows=rows, structure=structure, active=active,
            all_members=members, declarations=dummy_decls,
            topology_candidates=_guard_ring_candidates(
                intent, tuple(item.device for item in active)
            ),
        )
        if issue is not None:
            residuals.append(issue)
            continue
        declaration_refs = tuple(sorted((ref, *dummy_refs)))
        requested = {item.device for item in active} | {item.device for item in dummies}
        body_tap_delegation = _body_tap_delegation(
            intent, tuple(sorted(requested)), by_name
        )
        overlap = tuple(sorted(requested & consumed))
        if overlap:
            residuals.append(_residual(
                "DEVICE_DOUBLE_CONSUMED", group, ref, overlap,
                "a device is already owned by an earlier canonical row entry",
            ))
            continue
        try:
            macro = _compile_macro(
                group=group, structure=structure, row_count=rows, active=active,
                dummies=dummies, dummy_obligations=obligations,
                declaration_refs=declaration_refs,
            )
        except RowCalculusError as exc:
            residuals.append(_residual(
                "ROW_CALCULUS_REJECTED", group, ref, tuple(sorted(requested)), str(exc)
            ))
            continue
        entry = RowProgramEntry(
            entry_id=f"row:{group}", group=group, owner=f"row:{group}",
            structure_kind=structure, rows=rows, order=order,
            active_devices=tuple(item.device for item in active),
            dummy_devices=tuple(item.device for item in dummies),
            declaration_refs=declaration_refs, macro=macro,
            orientation=orientation,
            body_tap_delegation=body_tap_delegation,
        )
        entries.append(entry)
        consumed.update(requested)
    return RowProgram(
        cell=cell_name, tech_id=tech_id, members=members,
        entries=tuple(entries), residuals=tuple(residuals),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RowMacroRealization:
    entry_id: str
    owner: str
    offset_nm: tuple[int, int]
    sequence: tuple[tuple[str, ...], ...]
    consumed_devices: tuple[str, ...]
    local_bbox_nm: tuple[int, int, int, int]
    placed_bbox_nm: tuple[int, int, int, int]
    geometry_sha256: str
    geometry: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _text(self.entry_id, "realization entry_id"))
        object.__setattr__(self, "owner", _text(self.owner, "realization owner"))
        if type(self.offset_nm) is not tuple or len(self.offset_nm) != 2:
            raise RowProgramMaterializationError("offset_nm must be an exact 2-tuple")
        if any(type(value) is not int for value in self.offset_nm):
            raise RowProgramMaterializationError("offset_nm values must be exact integers")
        raw_rows = _bounded(self.sequence, 256, "realization sequence rows")
        rebuilt_rows: list[tuple[str, ...]] = []
        for row in raw_rows:
            if type(row) is not tuple:
                raise RowProgramMaterializationError("realization sequence rows must be exact tuples")
            rebuilt_rows.append(_exact_text_tuple(
                row, "realization sequence atoms", MAX_GROUP_MEMBERS * 256
            ))
        if not rebuilt_rows or any(not row for row in rebuilt_rows):
            raise RowProgramMaterializationError("realization sequence rows must not be empty")
        object.__setattr__(self, "sequence", tuple(rebuilt_rows))
        object.__setattr__(
            self, "consumed_devices",
            tuple(sorted(_exact_text_tuple(
                self.consumed_devices, "realization consumed_devices", MAX_GROUP_MEMBERS
            ))),
        )
        local = _bbox4(self.local_bbox_nm, "local_bbox_nm")
        placed = _bbox4(self.placed_bbox_nm, "placed_bbox_nm")
        expected_placed = (
            local[0] + self.offset_nm[0], local[1] + self.offset_nm[1],
            local[2] + self.offset_nm[0], local[3] + self.offset_nm[1],
        )
        if placed != expected_placed:
            raise RowProgramMaterializationError(
                "placed_bbox_nm must be local_bbox_nm translated by offset_nm"
            )
        object.__setattr__(self, "local_bbox_nm", local)
        object.__setattr__(self, "placed_bbox_nm", placed)
        digest = _digest(self.geometry_sha256, "realization geometry_sha256")
        actual_digest = row_geometry_sha256(self.geometry, self.consumed_devices)
        if digest != actual_digest:
            raise RowProgramMaterializationError(
                "realization geometry_sha256 does not bind stored geometry"
            )
        object.__setattr__(self, "geometry_sha256", digest)


def _bbox4(value: object, field: str) -> tuple[int, int, int, int]:
    if type(value) is not tuple or len(value) != 4:
        raise RowProgramMaterializationError(f"{field} must be an exact 4-tuple")
    x0, y0, x1, y1 = tuple(
        _coordinate(item, field) for item in value
    )
    if x1 <= x0 or y1 <= y0:
        raise RowProgramMaterializationError(f"{field} must have positive area")
    return value


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RowMaterializationResidual:
    code: str
    entry_id: str
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, "materialization residual code"))
        object.__setattr__(self, "entry_id", _text(self.entry_id, "materialization entry_id"))
        object.__setattr__(self, "detail", _text(self.detail, "materialization detail"))


@dataclass(frozen=True, slots=True, kw_only=True)
class RowMaterializationCertificate:
    schema: str
    program_sha256: str
    tech_id: str
    entries: tuple[
        tuple[
            str, str, tuple[int, int], tuple[tuple[str, ...], ...], tuple[str, ...],
            tuple[int, int, int, int], tuple[int, int, int, int], str,
        ], ...
    ]
    consumed_devices: tuple[str, ...]
    residuals: tuple[tuple[str, str], ...]
    source_projection_sha256: str | None = None

    def __post_init__(self) -> None:
        schema = _text(self.schema, "materialization certificate schema")
        if schema not in {
            "clay.row-materialization-certificate.v2",
            "clay.row-materialization-certificate.v3",
        }:
            raise RowProgramMaterializationError("unsupported materialization certificate schema")
        if schema == "clay.row-materialization-certificate.v2":
            if self.source_projection_sha256 is not None:
                raise RowProgramMaterializationError("legacy materialization cannot bind source projection")
        else:
            _digest(self.source_projection_sha256, "source_projection_sha256")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self, "program_sha256", _digest(self.program_sha256, "program_sha256")
        )
        object.__setattr__(self, "tech_id", _text(self.tech_id, "certificate tech_id"))
        raw_entries = _bounded(
            self.entries, MAX_PROGRAM_ENTRIES, "materialization certificate entries"
        )
        rebuilt_entries = []
        aggregate_devices: set[str] = set()
        for item in raw_entries:
            if type(item) is not tuple or len(item) != 8:
                raise RowProgramMaterializationError(
                    "materialization certificate entries must be exact 8-tuples"
                )
            entry_id = _text(item[0], "certificate entry_id")
            owner = _text(item[1], "certificate owner")
            offset = item[2]
            if type(offset) is not tuple or len(offset) != 2 or any(type(v) is not int for v in offset):
                raise RowProgramMaterializationError("certificate offset must be an exact integer 2-tuple")
            sequence_rows = _bounded(item[3], 256, "certificate sequence rows")
            sequence = tuple(
                _exact_text_tuple(row, "certificate sequence atoms", MAX_GROUP_MEMBERS * 256)
                if type(row) is tuple else _raise_bad_sequence_row()
                for row in sequence_rows
            )
            devices = tuple(sorted(_exact_text_tuple(
                item[4], "certificate entry consumed_devices", MAX_GROUP_MEMBERS
            )))
            local = _bbox4(item[5], "certificate local bbox")
            placed = _bbox4(item[6], "certificate placed bbox")
            expected = (
                local[0] + offset[0], local[1] + offset[1],
                local[2] + offset[0], local[3] + offset[1],
            )
            if placed != expected:
                raise RowProgramMaterializationError(
                    "certificate placed bbox is not local bbox translated by offset"
                )
            geometry_sha256 = _digest(
                item[7], "certificate entry geometry_sha256"
            )
            aggregate_devices.update(devices)
            rebuilt_entries.append((
                entry_id, owner, offset, sequence, devices, local, placed,
                geometry_sha256,
            ))
        rebuilt_entries.sort(key=lambda item: item[0])
        if len({item[0] for item in rebuilt_entries}) != len(rebuilt_entries):
            raise RowProgramMaterializationError("certificate entry ids must be unique")
        object.__setattr__(self, "entries", tuple(rebuilt_entries))
        consumed = tuple(sorted(_exact_text_tuple(
            self.consumed_devices, "certificate consumed_devices", MAX_PROGRAM_MEMBERS
        )))
        if consumed != tuple(sorted(aggregate_devices)):
            raise RowProgramMaterializationError(
                "certificate consumed_devices must equal the union of entry consumption"
            )
        object.__setattr__(self, "consumed_devices", consumed)
        raw_residuals = _bounded(
            self.residuals, MAX_PROGRAM_RESIDUALS, "certificate residuals"
        )
        residuals: list[tuple[str, str]] = []
        for item in raw_residuals:
            if type(item) is not tuple or len(item) != 2:
                raise RowProgramMaterializationError(
                    "certificate residuals must be exact 2-tuples"
                )
            residuals.append((
                _text(item[0], "certificate residual entry_id"),
                _text(item[1], "certificate residual code"),
            ))
        object.__setattr__(self, "residuals", tuple(sorted(residuals)))

    def payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "program_sha256": self.program_sha256,
            "tech_id": self.tech_id,
            "entries": self.entries,
            "consumed_devices": self.consumed_devices,
            "residuals": self.residuals,
        }
        if self.source_projection_sha256 is not None:
            payload["source_projection_sha256"] = self.source_projection_sha256
        return payload

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json(self.payload())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class RowProgramMaterialization:
    realizations: tuple[RowMacroRealization, ...]
    residuals: tuple[RowMaterializationResidual, ...]
    certificate: RowMaterializationCertificate
    source_projection: object | None = None

    def __post_init__(self) -> None:
        realizations = _bounded(
            self.realizations, MAX_PROGRAM_ENTRIES, "materialization realizations"
        )
        residuals = _bounded(
            self.residuals, MAX_PROGRAM_RESIDUALS, "materialization residuals"
        )
        if any(type(item) is not RowMacroRealization for item in realizations):
            raise RowProgramMaterializationError("realizations must be exact RowMacroRealization values")
        if any(type(item) is not RowMaterializationResidual for item in residuals):
            raise RowProgramMaterializationError("residuals must be exact RowMaterializationResidual values")
        if type(self.certificate) is not RowMaterializationCertificate:
            raise RowProgramMaterializationError("certificate must be exact RowMaterializationCertificate")
        object.__setattr__(self, "realizations", tuple(realizations))
        object.__setattr__(self, "residuals", tuple(sorted(residuals)))
        verify_row_program_materialization(self)


def verify_row_program_materialization(
    materialization: RowProgramMaterialization,
    *,
    program: RowProgram | None = None,
    tech_id: str | None = None,
) -> bool:
    """Deep-replay one stored materialization against its signed geometry.

    This is intentionally called again at each placement projection boundary:
    frozen wrapper objects do not make the geometry builders' polygon lists or
    pin maps immutable after construction.
    """

    return _verify_row_program_materialization(materialization, program=program, tech_id=tech_id)


def _verify_row_program_materialization(materialization, *, program=None, tech_id=None, entry_replay=None):
    # Only the invocation owner supplies an entry replayer. The public API above
    # always independently reconstructs current geometry and terminal graphs.
    if type(materialization) is not RowProgramMaterialization:
        raise RowProgramMaterializationError(
            "materialization must be an exact RowProgramMaterialization"
        )
    certificate = materialization.certificate
    if type(certificate) is not RowMaterializationCertificate:
        raise RowProgramMaterializationError(
            "materialization certificate must be exact"
        )
    if program is not None and type(program) is not RowProgram:
        raise RowProgramMaterializationError("program must be an exact RowProgram")
    # Rebuild the certificate to reject post-construction field corruption.
    replayed_certificate = RowMaterializationCertificate(
        schema=certificate.schema,
        program_sha256=certificate.program_sha256,
        tech_id=certificate.tech_id,
        entries=certificate.entries,
        consumed_devices=certificate.consumed_devices,
        residuals=certificate.residuals,
        source_projection_sha256=certificate.source_projection_sha256,
    )
    if replayed_certificate.to_canonical_bytes() != certificate.to_canonical_bytes():
        raise RowProgramMaterializationError(
            "materialization certificate canonical replay mismatch"
        )
    if materialization.source_projection is None:
        if replayed_certificate.source_projection_sha256 is not None:
            raise RowProgramMaterializationError("materialization source projection is missing")
    else:
        from clay.layout.place.row_source_projection import require_row_source_projection

        projection = require_row_source_projection(materialization.source_projection, program=program)
        if replayed_certificate.source_projection_sha256 != projection.sha256:
            raise RowProgramMaterializationError("materialization source projection digest differs")
    if program is not None:
        if replayed_certificate.program_sha256 != program.sha256:
            raise RowProgramMaterializationError(
                "materialization certificate does not bind the current row program"
            )
        if replayed_certificate.tech_id != program.tech_id:
            raise RowProgramMaterializationError(
                "materialization certificate tech does not match the row program"
            )
        if replayed_certificate.consumed_devices != program.consumed_devices:
            raise RowProgramMaterializationError(
                "materialization consumption does not match the current row program"
            )
        program_entries = {item.entry_id: item for item in program.entries}
        if set(program_entries) != {item[0] for item in replayed_certificate.entries}:
            raise RowProgramMaterializationError(
                "materialization entry set does not match the current row program"
            )
        for record in replayed_certificate.entries:
            entry = program_entries[record[0]]
            expected_sequence = tuple(
                tuple(atom.atom_id for atom in row.atoms) for row in entry.macro.rows
            )
            if (
                record[1] != entry.owner
                or record[3] != expected_sequence
                or record[4] != entry.consumed_devices
            ):
                raise RowProgramMaterializationError(
                    f"materialization entry does not match row program: {entry.entry_id}"
                )
    if tech_id is not None:
        exact_tech_id = _text(tech_id, "expected materialization tech_id")
        if replayed_certificate.tech_id != exact_tech_id:
            raise RowProgramMaterializationError(
                "materialization certificate does not bind the expected technology"
            )
    by_entry = {item[0]: item for item in replayed_certificate.entries}
    if len(by_entry) != len(replayed_certificate.entries):
        raise RowProgramMaterializationError(
            "materialization certificate entry ids are not unique"
        )
    if len(materialization.realizations) != len(by_entry):
        raise RowProgramMaterializationError(
            "stored realization count does not match materialization certificate"
        )
    seen = set()
    cumulative_polygons = 0
    cumulative_vertices = 0
    for realization in materialization.realizations:
        if type(realization) is not RowMacroRealization:
            raise RowProgramMaterializationError(
                "stored realizations must be exact RowMacroRealization values"
            )
        entry = by_entry.get(realization.entry_id)
        if entry is None or realization.entry_id in seen:
            raise RowProgramMaterializationError(
                "stored realization entry binding is missing or duplicated"
            )
        seen.add(realization.entry_id)
        polygon_count, vertex_count = _geometry_complexity(realization.geometry)
        cumulative_polygons += polygon_count
        if cumulative_polygons > MAX_MATERIALIZED_POLYGONS:
            raise RowProgramMaterializationError(
                "materialization exceeds cumulative polygon complexity cap"
            )
        cumulative_vertices += vertex_count
        if cumulative_vertices > MAX_MATERIALIZED_VERTICES:
            raise RowProgramMaterializationError(
                "materialization exceeds cumulative vertex complexity cap"
            )
        if program is not None:
            _replay_guest_slot_entry_geometry(
                program_entries[realization.entry_id], realization.geometry,
                members={member.device: member for member in program.members},
            )
            if entry_replay is None:
                _require_row_terminal_member_nets(
                    realization.geometry, realization.consumed_devices,
                    {member.device: member for member in program.members},
                )
        # Certificate entries are tuples, not source declarations. Without a
        # current program the geometry hash below replays its local slot binding;
        # it does not claim to establish fresh source authority.
        try:
            actual_geometry_sha256 = (row_geometry_sha256(realization.geometry, realization.consumed_devices)
                if entry_replay is None else entry_replay(realization,
                    None if program is None else program_entries[realization.entry_id]))
        except (ValueError, TypeError) as exc:
            raise RowProgramMaterializationError(
                f"stored realization geometry changed: {realization.entry_id}: {exc}"
            ) from exc
        if actual_geometry_sha256 != realization.geometry_sha256:
            raise RowProgramMaterializationError(
                f"stored realization geometry changed: {realization.entry_id}"
            )
        expected = (
            realization.entry_id,
            realization.owner,
            realization.offset_nm,
            realization.sequence,
            realization.consumed_devices,
            realization.local_bbox_nm,
            realization.placed_bbox_nm,
            realization.geometry_sha256,
        )
        if entry != expected:
            raise RowProgramMaterializationError(
                f"stored realization certificate mismatch: {realization.entry_id}"
            )
    residuals = tuple(
        (item.entry_id, item.code) for item in sorted(materialization.residuals)
    )
    if residuals != replayed_certificate.residuals:
        raise RowProgramMaterializationError(
            "stored materialization residuals do not match certificate"
        )
    return True


def _raise_bad_sequence_row() -> tuple[str, ...]:
    raise RowProgramMaterializationError("certificate sequence rows must be exact tuples")


def _rebuild_program_exact(program: RowProgram) -> RowProgram:
    """Finite deep replay before any geometry-side effect.

    Frozen dataclasses can still be corrupted through ``object.__setattr__`` or
    hostile deserialization.  Reconstruct every nested row through
    ``compile_row_macro`` so its cap+1 probes and exact nested-type checks run
    before a rigid builder is invoked.
    """

    raw_members = _bounded(program.members, MAX_PROGRAM_MEMBERS, "deep replay members")
    members = tuple(
        RowMemberSpec(
            device=item.device, kind=item.kind, width_nm=item.width_nm,
            length_nm=item.length_nm, fingers=item.fingers,
            gate_net=item.gate_net, drain_net=item.drain_net,
            source_net=item.source_net, body_net=item.body_net,
            oxide=item.oxide, dummy_for=item.dummy_for,
        )
        for item in raw_members
        if type(item) is RowMemberSpec
    )
    if len(members) != len(raw_members):
        raise RowProgramMaterializationError("deep replay requires exact RowMemberSpec values")
    entries: list[RowProgramEntry] = []
    for item in _bounded(program.entries, MAX_PROGRAM_ENTRIES, "deep replay entries"):
        if type(item) is not RowProgramEntry:
            raise RowProgramMaterializationError("deep replay requires exact RowProgramEntry values")
        macro = item.macro
        if type(macro) is not RowMacro:
            raise RowProgramMaterializationError("deep replay requires an exact RowMacro")
        rebuilt_macro = compile_row_macro(
            macro_id=macro.macro_id,
            structure_kind=macro.structure_kind,
            rows=macro.rows,
            row_pitch_nm=macro.row_pitch_nm,
            declaration_refs=macro.declaration_refs,
            dummy_endcap_obligations=macro.dummy_endcap_obligations,
        )
        if rebuilt_macro is None:
            raise RowProgramMaterializationError("deep replay deleted a supposedly live macro")
        entries.append(RowProgramEntry(
            entry_id=item.entry_id, group=item.group, owner=item.owner,
            structure_kind=item.structure_kind, rows=item.rows, order=item.order,
            active_devices=item.active_devices, dummy_devices=item.dummy_devices,
            declaration_refs=item.declaration_refs, macro=rebuilt_macro,
            orientation=item.orientation,
            body_tap_delegation=(
                RowBodyTapDelegation(**item.body_tap_delegation.payload())
                if item.body_tap_delegation is not None
                else None
            ),
        ))
    residuals: list[RowProgramResidual] = []
    for item in _bounded(program.residuals, MAX_PROGRAM_RESIDUALS, "deep replay residuals"):
        if type(item) is not RowProgramResidual:
            raise RowProgramMaterializationError("deep replay requires exact residual values")
        residuals.append(RowProgramResidual(
            code=item.code, group=item.group, declaration_refs=item.declaration_refs,
            devices=item.devices, detail=item.detail,
        ))
    return RowProgram(
        cell=program.cell, tech_id=program.tech_id, members=members,
        entries=tuple(entries), residuals=tuple(residuals),
    )


def _offset_map(
    entries: tuple[RowProgramEntry, ...], values: Mapping[str, tuple[int, int]] | None
) -> dict[str, tuple[int, int]]:
    if values is None:
        return {entry.owner: (0, 0) for entry in entries}
    if not isinstance(values, Mapping):
        raise RowProgramMaterializationError("offsets_nm must be a mapping")
    items = _bounded(values.items(), MAX_PROGRAM_ENTRIES, "offsets_nm")
    owners = {entry.owner for entry in entries}
    unknown = sorted(_text(owner, "offset owner") for owner, _point in items if owner not in owners)
    if unknown:
        raise RowProgramMaterializationError(f"unknown owner in offsets_nm: {tuple(unknown)}")
    result = {entry.owner: (0, 0) for entry in entries}
    for owner, point in items:
        if type(point) is not tuple or len(point) != 2 or any(type(v) is not int for v in point):
            raise RowProgramMaterializationError("every offset must be an exact integer 2-tuple")
        result[str(owner)] = point
    return result


def _geometry_override_map(
    entries: tuple[RowProgramEntry, ...], values: Mapping[str, object] | None
) -> dict[str, object]:
    """Validate exact-owner rigid artwork supplied by a bound materializer.

    The override changes only PDK artwork.  RowProgram retains ownership,
    sequence, footprint, terminal, complexity, and content-hash verification
    below, so a caller cannot bypass the structural certificate.
    """

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise RowProgramMaterializationError("geometry_overrides must be a mapping")
    items = _bounded(
        values.items(), MAX_PROGRAM_ENTRIES, "geometry_overrides"
    )
    by_owner = {entry.owner: entry for entry in entries}
    unknown = sorted(
        _text(owner, "geometry override owner")
        for owner, _geometry in items
        if owner not in by_owner
    )
    if unknown:
        raise RowProgramMaterializationError(
            f"unknown owner in geometry_overrides: {tuple(unknown)}"
        )
    result: dict[str, object] = {}
    for owner, geometry in items:
        exact_owner = _text(owner, "geometry override owner")
        if by_owner[exact_owner].structure_kind not in {
            RowStructureKind.COMMON_CENTROID,
            RowStructureKind.DISCRETE_MOSAIC,
        }:
            raise RowProgramMaterializationError(
                "geometry override requires a common-centroid row owner"
            )
        if geometry is None:
            raise RowProgramMaterializationError(
                "geometry override must contain one materialized geometry"
            )
        result[exact_owner] = geometry
    return result


def materialize_row_program(
    program: RowProgram,
    tech: object,
    *,
    offsets_nm: Mapping[str, tuple[int, int]] | None = None,
    geometry_overrides: Mapping[str, object] | None = None,
    source_projection: object | None = None,
) -> RowProgramMaterialization:
    """Invoke one existing rigid geometry builder for each compiled row entry."""

    if type(program) is not RowProgram:
        raise RowProgramMaterializationError("program must be an exact RowProgram")
    try:
        program = _rebuild_program_exact(program)
    except (RowProgramError, RowCalculusError) as exc:
        if isinstance(exc, RowProgramMaterializationError):
            raise
        raise RowProgramMaterializationError(f"deep replay rejected program: {exc}") from exc
    if program.residuals:
        raise RowProgramMaterializationError(
            "row program has compile residuals; refusing partial/free-device fallback"
        )
    if source_projection is not None:
        from clay.layout.place.row_source_projection import require_row_source_projection

        require_row_source_projection(source_projection, program=program)
    tech_id = _text(getattr(tech, "pdk_id", getattr(tech, "name", "")), "tech identity")
    if tech_id != program.tech_id:
        raise RowProgramMaterializationError(
            f"tech identity mismatch: compiled {program.tech_id!r}, replay {tech_id!r}"
        )
    offsets = _offset_map(program.entries, offsets_nm)
    overrides = _geometry_override_map(program.entries, geometry_overrides)
    by_name = {item.device: item for item in program.members}
    realizations: list[RowMacroRealization] = []
    residuals: list[RowMaterializationResidual] = []
    cumulative_polygons = 0
    cumulative_vertices = 0
    for entry in program.entries:
        active = tuple(by_name[name] for name in entry.active_devices)
        dummy = tuple(by_name[name] for name in entry.dummy_devices)
        oxide = active[0].oxide
        body_tap_owner_sha256 = (
            entry.body_tap_delegation.declaration_sha256
            if entry.body_tap_delegation is not None
            else ""
        )
        if entry.body_tap_delegation is not None and (
            any(item.kind != "pmos" for item in (*active, *dummy))
            or any(item.body_net != item.source_net for item in (*active, *dummy))
            or any(
                item.body_net != entry.body_tap_delegation.tie_net
                for item in (*active, *dummy)
            )
        ):
            raise RowProgramMaterializationError(
                "body-tap delegation no longer matches source-tied PMOS members"
            )
        try:
            if entry.owner in overrides:
                geometry = overrides[entry.owner]
            elif entry.structure_kind is RowStructureKind.DISCRETE_MOSAIC:
                residuals.append(RowMaterializationResidual(
                    code="MOSAIC_GEOMETRY_OVERRIDE_REQUIRED",
                    entry_id=entry.entry_id,
                    detail=(
                        "a projected mosaic requires one evidence-bound geometry override"
                    ),
                ))
                continue
            elif entry.structure_kind is RowStructureKind.SHARED_OD:
                from clay.layout.device_cell import build_shared_diffusion_array
                flank = tuple((item.builder_tuple, item.fingers // 2) for item in dummy)
                geometry = build_shared_diffusion_array(
                    tech, tuple(item.builder_tuple for item in active),
                    order=entry.order or "ratio", oxide=oxide,
                    fold_rows=entry.rows, flank_dummies=flank,
                    body_tap_owner_sha256=body_tap_owner_sha256,
                )
            else:
                from clay.layout.common_centroid_array import build_common_centroid_array
                geometry = build_common_centroid_array(
                    tech, tuple(item.builder_tuple for item in active), rows=entry.rows,
                    oxide=oxide, flank_dummies=tuple(item.builder_tuple for item in dummy),
                    orientation=entry.orientation,
                    body_tap_owner_sha256=body_tap_owner_sha256,
                )
        except ValueError as exc:
            residuals.append(RowMaterializationResidual(
                code="MATERIALIZER_REJECTED", entry_id=entry.entry_id, detail=str(exc)
            ))
            continue
        active_set = set(entry.active_devices)
        dummy_set = set(entry.dummy_devices)
        expected_owner_sequence_tuple = expected_row_finger_owners(entry)
        try:
            raw_fingers = _bounded(
                getattr(geometry, "fingers", ()), MAX_PROGRAM_MEMBERS * 256,
                "materialized fingers",
            )
            actual_owner_sequence: list[str] = []
            for finger in raw_fingers:
                owner = _text(
                    getattr(finger, "member", ""),
                    "materialized finger owner", allow_empty=True,
                )
                if owner in active_set or owner in dummy_set:
                    actual_owner_sequence.append(owner)
                    continue
                if not owner and len(dummy_set) == 1:
                    # The shared-OD builder historically records an empty owner for
                    # end-cap fingers.  With exactly one consumed dummy, ownership is
                    # nevertheless unique (not guessed): no other declared device can
                    # own an anonymous builder finger.  Count and left/right position
                    # are still checked by the full sequence comparison below.
                    actual_owner_sequence.append(next(iter(dummy_set)))
                    continue
                if not owner:
                    raise RowProgramMaterializationError(
                        "anonymous materialized dummy finger has no unique declared owner"
                    )
                actual_owner_sequence.append(owner)
        except RowProgramError as exc:
            residuals.append(RowMaterializationResidual(
                code="MATERIALIZED_SEQUENCE_UNMEASURABLE",
                entry_id=entry.entry_id,
                detail=str(exc),
            ))
            continue
        except RowProgramMaterializationError as exc:
            residuals.append(RowMaterializationResidual(
                code="MATERIALIZED_SEQUENCE_UNMEASURABLE",
                entry_id=entry.entry_id,
                detail=str(exc),
            ))
            continue
        actual_owner_sequence_tuple = tuple(actual_owner_sequence)
        if actual_owner_sequence_tuple != expected_owner_sequence_tuple:
            residuals.append(RowMaterializationResidual(
                code="MATERIALIZED_SEQUENCE_MISMATCH",
                entry_id=entry.entry_id,
                detail=(
                    f"program full owner sequence {expected_owner_sequence_tuple!r} != "
                    f"builder full owner sequence {actual_owner_sequence_tuple!r}"
                ),
            ))
            continue
        if entry.structure_kind is RowStructureKind.DISCRETE_MOSAIC:
            expected_transforms = tuple(
                atom.match_key.rsplit(":", 1)[-1]
                for row in entry.macro.rows
                for atom in row.atoms
            )
            actual_transforms = tuple(
                _text(
                    getattr(finger, "orientation", ""),
                    "materialized mosaic occurrence transform",
                )
                for finger in raw_fingers
            )
            if actual_transforms != expected_transforms:
                residuals.append(RowMaterializationResidual(
                    code="MATERIALIZED_MOSAIC_TRANSFORM_MISMATCH",
                    entry_id=entry.entry_id,
                    detail=(
                        f"program transforms {expected_transforms!r} != "
                        f"builder transforms {actual_transforms!r}"
                    ),
                ))
                continue
        try:
            _replay_guest_slot_entry_geometry(entry, geometry, tech=tech, members=by_name)
            local_bbox = _bbox4(tuple(geometry.extent()), "materialized local bbox")
        except (AttributeError, TypeError, RowProgramMaterializationError) as exc:
            residuals.append(RowMaterializationResidual(
                code="MATERIALIZED_FOOTPRINT_UNMEASURABLE",
                entry_id=entry.entry_id,
                detail=str(exc),
            ))
            continue
        dx, dy = offsets[entry.owner]
        placed_bbox = (
            local_bbox[0] + dx, local_bbox[1] + dy,
            local_bbox[2] + dx, local_bbox[3] + dy,
        )
        object.__setattr__(geometry, "composite_members", entry.consumed_devices)
        polygon_count, vertex_count = _geometry_complexity(geometry)
        cumulative_polygons += polygon_count
        cumulative_vertices += vertex_count
        if cumulative_polygons > MAX_MATERIALIZED_POLYGONS:
            raise RowProgramMaterializationError(
                "materialization exceeds cumulative polygon complexity cap"
            )
        if cumulative_vertices > MAX_MATERIALIZED_VERTICES:
            raise RowProgramMaterializationError(
                "materialization exceeds cumulative vertex complexity cap"
            )
        _require_row_terminal_member_nets(geometry, entry.consumed_devices, by_name, tech=tech)
        realizations.append(RowMacroRealization(
            entry_id=entry.entry_id, owner=entry.owner,
            offset_nm=offsets[entry.owner], sequence=entry.macro.sequence,
            consumed_devices=entry.consumed_devices,
            local_bbox_nm=local_bbox, placed_bbox_nm=placed_bbox,
            geometry_sha256=row_geometry_sha256(
                geometry, entry.consumed_devices
            ),
            geometry=geometry,
        ))
    realizations_tuple = tuple(realizations)
    residuals_tuple = tuple(sorted(residuals))
    certificate = RowMaterializationCertificate(
        schema=("clay.row-materialization-certificate.v2" if source_projection is None
                else "clay.row-materialization-certificate.v3"),
        program_sha256=program.sha256,
        tech_id=program.tech_id,
        entries=tuple(
            (
                item.entry_id, item.owner, item.offset_nm, item.sequence,
                item.consumed_devices, item.local_bbox_nm, item.placed_bbox_nm,
                item.geometry_sha256,
            )
            for item in realizations_tuple
        ),
        consumed_devices=tuple(sorted({
            device for item in realizations_tuple for device in item.consumed_devices
        })),
        residuals=tuple((item.entry_id, item.code) for item in residuals_tuple),
        source_projection_sha256=(None if source_projection is None else source_projection.sha256),
    )
    return RowProgramMaterialization(
        realizations=realizations_tuple, residuals=residuals_tuple, certificate=certificate,
        source_projection=source_projection,
    )


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RowTerminalAccessContract:
    member: str
    role: str
    net: str
    layer: tuple[int, int]
    accesses_nm: tuple[tuple[int, int], ...]
    component_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", _text(self.component_id, "terminal component id", allow_empty=True))
        object.__setattr__(self, "member", _text(self.member, "terminal member"))
        object.__setattr__(self, "role", _text(self.role, "terminal role"))
        object.__setattr__(self, "net", _text(self.net, "terminal net"))
        if (
            type(self.layer) is not tuple
            or len(self.layer) != 2
            or any(
                type(value) is not int
                or not 0 <= value <= MAX_GDS_LAYER_VALUE
                for value in self.layer
            )
        ):
            raise RowProgramMaterializationError(
                "terminal layer must be an exact bounded non-negative integer pair"
            )
        points = _bounded(
            self.accesses_nm, MAX_MEMBER_TERMINALS, "terminal accesses"
        )
        if not points:
            raise RowProgramMaterializationError(
                "terminal access contract must expose at least one point"
            )
        rebuilt = []
        for point in points:
            if type(point) is not tuple or len(point) != 2:
                raise RowProgramMaterializationError(
                    "terminal access must be an exact integer nm pair"
                )
            rebuilt.append((
                _coordinate(point[0], "terminal access x"),
                _coordinate(point[1], "terminal access y"),
            ))
        if len(set(rebuilt)) != len(rebuilt):
            raise RowProgramMaterializationError("terminal accesses contain duplicates")
        object.__setattr__(self, "accesses_nm", tuple(sorted(rebuilt)))

    def payload(self) -> dict[str, object]:
        payload = {
            "member": self.member,
            "role": self.role,
            "net": self.net,
            "layer": self.layer,
            "accesses_nm": self.accesses_nm,
        }
        if self.component_id:
            payload["component_id"] = self.component_id
        return payload


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RowRouteDemandContract:
    net: str
    members: tuple[str, ...]
    terminal_count: int
    access_count: int
    required_component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.required_component_ids) is not tuple:
            raise RowProgramMaterializationError("required component ids must be an exact tuple")
        components = _exact_text_tuple(self.required_component_ids, "required component ids",
                                       MAX_GROUP_MEMBERS * MAX_MEMBER_TERMINALS)
        if components != tuple(sorted(components)):
            raise RowProgramMaterializationError("required component ids must be canonical")
        object.__setattr__(self, "net", _text(self.net, "route demand net"))
        members = tuple(sorted(_exact_text_tuple(
            self.members, "route demand members", MAX_GROUP_MEMBERS
        )))
        if not members:
            raise RowProgramMaterializationError(
                "route demand must own at least one member"
            )
        object.__setattr__(self, "members", members)
        object.__setattr__(
            self,
            "terminal_count",
            _positive(
                self.terminal_count,
                "route demand terminal_count",
                MAX_GROUP_MEMBERS * MAX_MEMBER_TERMINALS,
            ),
        )
        object.__setattr__(
            self,
            "access_count",
            _positive(
                self.access_count,
                "route demand access_count",
                MAX_GROUP_MEMBERS * MAX_MEMBER_TERMINALS,
            ),
        )

    def payload(self) -> dict[str, object]:
        payload = {
            "net": self.net,
            "members": self.members,
            "terminal_count": self.terminal_count,
            "access_count": self.access_count,
        }
        if self.required_component_ids:
            payload["required_component_ids"] = self.required_component_ids
        return payload

    @property
    def component_count(self) -> int:
        return len(self.required_component_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class RowMacroContract:
    """Closed pre-placement interface for one exact rigid row materialization."""

    macro_id: str
    owner: str
    entry_sha256: str
    geometry_sha256: str
    structure_kind: RowStructureKind | str
    active_members: tuple[str, ...]
    dummy_members: tuple[str, ...]
    consumed_members: tuple[str, ...]
    local_bbox_nm: tuple[int, int, int, int]
    terminal_accesses: tuple[RowTerminalAccessContract, ...]
    local_symmetry_axes_2nm: tuple[tuple[str, int], ...]
    body_tie_domain: tuple[str, ...]
    legal_orientations: tuple[str, ...]
    route_demand: tuple[RowRouteDemandContract, ...]
    terminal_components: object = None
    joint_inventory: object = None
    joint_manufacturing_grid_nm: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "macro_id", _text(self.macro_id, "macro_id"))
        object.__setattr__(self, "owner", _text(self.owner, "macro owner"))
        object.__setattr__(
            self, "entry_sha256", _digest(self.entry_sha256, "entry_sha256")
        )
        object.__setattr__(
            self,
            "geometry_sha256",
            _digest(self.geometry_sha256, "geometry_sha256"),
        )
        object.__setattr__(self, "structure_kind", _structure(self.structure_kind))
        active = _exact_text_tuple(
            self.active_members, "macro active_members", MAX_GROUP_MEMBERS
        )
        dummy = tuple(sorted(_exact_text_tuple(
            self.dummy_members, "macro dummy_members", MAX_GROUP_MEMBERS
        )))
        consumed = tuple(sorted(_exact_text_tuple(
            self.consumed_members, "macro consumed_members", MAX_GROUP_MEMBERS
        )))
        if set(active) & set(dummy) or consumed != tuple(sorted((*active, *dummy))):
            raise RowProgramMaterializationError(
                "macro consumed members must equal disjoint active and dummy members"
            )
        object.__setattr__(self, "active_members", active)
        object.__setattr__(self, "dummy_members", dummy)
        object.__setattr__(self, "consumed_members", consumed)
        object.__setattr__(self, "local_bbox_nm", _bbox4(
            self.local_bbox_nm, "macro local bbox"
        ))
        terminals = _bounded(
            self.terminal_accesses,
            MAX_GROUP_MEMBERS * MAX_MEMBER_TERMINALS,
            "macro terminal accesses",
        )
        if any(type(item) is not RowTerminalAccessContract for item in terminals):
            raise RowProgramMaterializationError(
                "macro terminal accesses must contain exact contracts"
            )
        terminals = tuple(sorted(terminals))
        if not terminals:
            raise RowProgramMaterializationError(
                "macro must expose at least one terminal access"
            )
        terminal_keys = [(item.member, item.role, item.component_id) for item in terminals]
        if len(terminal_keys) != len(set(terminal_keys)):
            raise RowProgramMaterializationError(
                "macro terminal member/role/component keys must be unique"
            )
        if any(item.member not in consumed for item in terminals):
            raise RowProgramMaterializationError(
                "macro terminal access names an unconsumed member"
            )
        missing_active_terminals = tuple(
            sorted(set(active) - {item.member for item in terminals})
        )
        if missing_active_terminals:
            raise RowProgramMaterializationError(
                "macro terminal access is missing active member "
                f"{missing_active_terminals[0]!r}"
            )
        object.__setattr__(self, "terminal_accesses", terminals)
        axes = _bounded(self.local_symmetry_axes_2nm, 2, "macro symmetry axes")
        rebuilt_axes: list[tuple[str, int]] = []
        for item in axes:
            if (
                type(item) is not tuple
                or len(item) != 2
                or item[0] not in ("x", "y")
            ):
                raise RowProgramMaterializationError(
                    "macro symmetry axis must be ('x'|'y', exact doubled-nm int)"
                )
            rebuilt_axes.append((
                item[0],
                _coordinate(
                    item[1],
                    "macro symmetry axis",
                    maximum=2 * MAX_DIMENSION_NM,
                ),
            ))
        if len({axis for axis, _value in rebuilt_axes}) != len(rebuilt_axes):
            raise RowProgramMaterializationError("macro symmetry axes contain duplicates")
        object.__setattr__(self, "local_symmetry_axes_2nm", tuple(sorted(rebuilt_axes)))
        object.__setattr__(
            self,
            "body_tie_domain",
            tuple(sorted(_exact_text_tuple(
                self.body_tie_domain, "macro body tie domain", MAX_GROUP_MEMBERS
            ))),
        )
        orientations = _exact_text_tuple(
            self.legal_orientations, "macro legal orientations", 8
        )
        if not orientations or any(item not in ("R0", "R90", "R180", "R270") for item in orientations):
            raise RowProgramMaterializationError("macro has unsupported legal orientation")
        object.__setattr__(self, "legal_orientations", tuple(sorted(orientations)))
        demand = _bounded(
            self.route_demand, MAX_GROUP_MEMBERS * MAX_MEMBER_TERMINALS,
            "macro route demand",
        )
        if any(type(item) is not RowRouteDemandContract for item in demand):
            raise RowProgramMaterializationError(
                "macro route demand must contain exact contracts"
            )
        demand = tuple(sorted(demand))
        if not demand:
            raise RowProgramMaterializationError(
                "macro must expose at least one route demand"
            )
        if len({item.net for item in demand}) != len(demand):
            raise RowProgramMaterializationError("macro route demand nets must be unique")
        object.__setattr__(self, "route_demand", demand)
        require_macro_inventory(self.terminal_components, self.joint_inventory, terminals, demand,
                                grid=self.joint_manufacturing_grid_nm)

    def payload(self) -> dict[str, object]:
        payload = {
            "schema": "clay.row-macro-contract/v1" if self.terminal_components is None else "clay.row-macro-contract/v2",
            "macro_id": self.macro_id,
            "owner": self.owner,
            "entry_sha256": self.entry_sha256,
            "geometry_sha256": self.geometry_sha256,
            "structure_kind": self.structure_kind.value,
            "active_members": self.active_members,
            "dummy_members": self.dummy_members,
            "consumed_members": self.consumed_members,
            "local_bbox_nm": self.local_bbox_nm,
            "terminal_accesses": tuple(item.payload() for item in self.terminal_accesses),
            "local_symmetry_axes_2nm": self.local_symmetry_axes_2nm,
            "body_tie_domain": self.body_tie_domain,
            "legal_orientations": self.legal_orientations,
            "route_demand": tuple(item.payload() for item in self.route_demand),
        }
        if self.terminal_components is not None:
            payload["terminal_components"] = self.terminal_components.payload()
        if self.joint_inventory is not None:
            payload.update(schema='clay.row-macro-contract/v3', joint_inventory=joint_inventory_payload(self.joint_inventory),
                           joint_manufacturing_grid_nm=self.joint_manufacturing_grid_nm)
        return payload

    @property
    def structure_fingerprint(self) -> str:
        return _sha(self.payload())


@dataclass(frozen=True, slots=True, kw_only=True)
class RowMacroContractSet:
    graph_sha256: str
    program_sha256: str
    materialization_sha256: str
    tech_id: str
    macros: tuple[RowMacroContract, ...]

    def __post_init__(self) -> None:
        for field in ("graph_sha256", "program_sha256", "materialization_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "tech_id", _text(self.tech_id, "contract tech_id"))
        macros = _bounded(self.macros, MAX_PROGRAM_ENTRIES, "row macro contracts")
        if any(type(item) is not RowMacroContract for item in macros):
            raise RowProgramMaterializationError(
                "row macro contract set requires exact macro contracts"
            )
        macros = tuple(sorted(macros, key=lambda item: item.macro_id))
        if len({item.macro_id for item in macros}) != len(macros):
            raise RowProgramMaterializationError("row macro contract ids must be unique")
        object.__setattr__(self, "macros", macros)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.row-macro-contract-set/v1",
            "graph_sha256": self.graph_sha256,
            "program_sha256": self.program_sha256,
            "materialization_sha256": self.materialization_sha256,
            "tech_id": self.tech_id,
            "macros": tuple(item.payload() for item in self.macros),
        }

    @property
    def sha256(self) -> str:
        return _sha(self.payload())


def compile_row_macro_contracts(
    graph: object,
    program: RowProgram,
    materialization: RowProgramMaterialization,
    *,
    tech: object,
) -> RowMacroContractSet:
    """Close graph, exact row geometry, terminals, and route demand before solve."""

    from clay.layout.physical_macro_graph import PhysicalMacroGraph

    if type(graph) is not PhysicalMacroGraph:
        raise RowProgramMaterializationError(
            "graph must be an exact PhysicalMacroGraph"
        )
    if type(program) is not RowProgram:
        raise RowProgramMaterializationError("program must be an exact RowProgram")
    from clay.layout.place.row_replay_session import require_current_row_materialization, current_row_terminal_inventory
    require_current_row_materialization(
        materialization, program=program, tech_id=program.tech_id
    )
    if materialization.residuals:
        raise RowProgramMaterializationError(
            "cannot compile macro contracts from a partial materialization"
        )
    if graph.cell != program.cell:
        raise RowProgramMaterializationError(
            "physical graph and row program name different cells"
        )
    tech_id = _text(
        getattr(tech, "pdk_id", getattr(tech, "name", "")), "contract tech identity"
    )
    if tech_id != program.tech_id:
        raise RowProgramMaterializationError(
            "contract technology differs from the row program"
        )
    try:
        layer = tuple(tech.metal_key(1))
    except Exception as exc:
        raise RowProgramMaterializationError(
            "contract technology exposes no routable metal1 layer"
        ) from exc
    if (
        len(layer) != 2
        or any(type(value) is not int or value < 0 for value in layer)
    ):
        raise RowProgramMaterializationError(
            "contract metal1 layer must contain exact non-negative integers"
        )

    graph_macros = {item.macro_id: item for item in graph.macros}
    entries = {item.entry_id: item for item in program.entries}
    realizations = {item.entry_id: item for item in materialization.realizations}
    members = {item.device: item for item in program.members}
    contracts: list[RowMacroContract] = []
    for entry_id in sorted(entries):
        entry = entries[entry_id]
        realization = realizations.get(entry_id)
        if realization is None:
            raise RowProgramMaterializationError(
                f"row realization missing for contract {entry_id}"
            )
        macro_id = f"matched:{entry.group}"
        graph_macro = graph_macros.get(macro_id)
        if graph_macro is None or graph_macro.members != tuple(
            sorted(entry.active_devices)
        ):
            raise RowProgramMaterializationError(
                f"graph membership differs from exact row contract {macro_id}"
            )
        unknown_consumed = tuple(
            sorted(set(entry.consumed_devices) - set(graph.devices))
        )
        if unknown_consumed:
            raise RowProgramMaterializationError(
                f"row contract consumes unknown graph member {unknown_consumed[0]!r}"
            )

        inventory = current_row_terminal_inventory(
            realization.geometry, entry.consumed_devices, members, tech=tech,
        )
        terminals: list[RowTerminalAccessContract] = []
        net_demands: dict[str, dict[str, object]] = {}
        for member_name in (() if inventory is not None else entry.consumed_devices):
            member = members[member_name]
            try:
                pins = realization.geometry.member_pins(member_name)
            except KeyError:
                if member_name in entry.active_devices:
                    raise RowProgramMaterializationError(
                        f"materialized active member exposes no pins: {member_name}"
                    )
                pins = {}
            if not isinstance(pins, Mapping):
                raise RowProgramMaterializationError(
                    f"materialized member pins are not a mapping: {member_name}"
                )
            if member_name in entry.active_devices and not pins:
                raise RowProgramMaterializationError(
                    f"materialized active member exposes no pins: {member_name}"
                )
            net_by_role = {
                "g": member.gate_net,
                "d": member.drain_net,
                "s": member.source_net,
                "b": member.body_net,
            }
            for raw_role, raw_point in sorted(pins.items()):
                role = _text(raw_role, "materialized terminal role")
                net = net_by_role.get(role, "")
                if not net:
                    raise RowProgramMaterializationError(
                        f"materialized terminal {member_name}.{role} has no declared net"
                    )
                point = tuple(raw_point)
                terminal = RowTerminalAccessContract(
                    member=member_name,
                    role=role,
                    net=net,
                    layer=(layer[0], layer[1]),
                    accesses_nm=(point,),
                )
                terminals.append(terminal)
                demand = net_demands.setdefault(
                    net, {"members": set(), "terminals": 0, "accesses": 0}
                )
                demand["members"].add(member_name)  # type: ignore[union-attr]
                demand["terminals"] = int(demand["terminals"]) + 1
                demand["accesses"] = int(demand["accesses"]) + len(
                    terminal.accesses_nm
                )
        route_demand = tuple(
            RowRouteDemandContract(
                net=net,
                members=tuple(sorted(data["members"])),  # type: ignore[arg-type]
                terminal_count=int(data["terminals"]),
                access_count=int(data["accesses"]),
            )
            for net, data in sorted(net_demands.items())
        )
        if inventory is not None:
            from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
            terminals = list(_component_terminal_contracts(inventory, grid=pdk_manufacturing_grid_nm(tech)))
            route_demand = _component_route_demands(terminals)
        x0, y0, x1, y1 = realization.local_bbox_nm
        axes = (
            (("x", x0 + x1), ("y", y0 + y1))
            if entry.structure_kind is RowStructureKind.COMMON_CENTROID
            else ()
        )
        contracts.append(RowMacroContract(
            macro_id=macro_id,
            owner=entry.owner,
            entry_sha256=entry.sha256,
            geometry_sha256=realization.geometry_sha256,
            structure_kind=entry.structure_kind,
            active_members=entry.active_devices,
            dummy_members=entry.dummy_devices,
            consumed_members=entry.consumed_devices,
            local_bbox_nm=realization.local_bbox_nm,
            terminal_accesses=tuple(terminals),
            local_symmetry_axes_2nm=axes,
            body_tie_domain=tuple(sorted({
                members[name].body_net
                for name in entry.consumed_devices
                if members[name].body_net
            })),
            legal_orientations=(entry.orientation,),
            route_demand=route_demand,
            terminal_components=None if is_joint_inventory(inventory) else inventory,
            joint_inventory=inventory if is_joint_inventory(inventory) else None,
            joint_manufacturing_grid_nm=pdk_manufacturing_grid_nm(tech) if is_joint_inventory(inventory) else 1,
        ))
    return RowMacroContractSet(
        graph_sha256=graph.sha256,
        program_sha256=program.sha256,
        materialization_sha256=materialization.certificate.sha256,
        tech_id=program.tech_id,
        macros=tuple(contracts),
    )


__all__ = [
    "MAX_GROUP_MEMBERS",
    "MAX_PROGRAM_ENTRIES",
    "MAX_PROGRAM_MEMBERS",
    "RowCompileCertificate",
    "RowMacroContract",
    "RowMacroContractSet",
    "RowMacroRealization",
    "RowMaterializationCertificate",
    "RowMaterializationResidual",
    "RowMemberSpec",
    "RowProgram",
    "RowProgramEntry",
    "RowProgramError",
    "RowProgramMaterialization",
    "RowProgramMaterializationError",
    "RowProgramResidual",
    "RowRouteDemandContract",
    "RowTerminalAccessContract",
    "compile_row_macro_contracts",
    "compile_row_program",
    "materialize_row_program",
    "row_geometry_sha256",
    "row_terminal_sources",
    "replay_row_terminal_inventory",
    "verify_row_program_materialization",
]
