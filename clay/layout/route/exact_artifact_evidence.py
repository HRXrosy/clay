"""Typed evidence for one exact emitted route artifact.

The router's raw ``tagged`` list is not automatically an emitted ownership
manifest: :func:`emit_labelled_gds` may merge fragments, add same-net heal/fill
material, and snap polygons.  This module therefore accepts an ownership
manifest only together with an emitter-produced scope certificate proving that
those transformations were a no-op in geometry for the exact route artifact.

The resulting bundle is deliberately small.  It joins one immutable GDS, one
exhaustive router-owned manifest, and already-computed via/layer reports.  It
never invokes a router, emitter, DRC deck, LVS deck, or checker itself.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from clay.layout.route.truth_manifest import TruthManifest
from clay.layout.route.via_provenance_check import route_manifest_sha256

LayerKey = Tuple[int, int]
BBoxNm = Tuple[int, int, int, int]
RouteManifestEntry = Tuple[str, LayerKey, BBoxNm]

_SCHEMA = "clay.exact_emitted_route_artifact.v1"
_MAX_ENTRIES = 1_000_000


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _canonical_int_ring(points: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """Canonical non-degenerate polygon ring for emitter-input replay."""

    ring = tuple((int(x), int(y)) for x, y in points)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3 or len(set(ring)) < 3:
        raise ValueError("emitted device snapshot contains a degenerate polygon")
    twice_area = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1])
    )
    if twice_area == 0:
        raise ValueError("emitted device snapshot contains a degenerate polygon")
    candidates = []
    for sequence in (ring, tuple(reversed(ring))):
        candidates.extend(
            sequence[index:] + sequence[:index] for index in range(len(sequence))
        )
    return min(candidates)


def _emitted_device_snapshot_sha256(
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Any],
) -> str:
    """Digest every exact placed polygon handed to the canonical GDS emitter.

    The digest is label- and layer-aware, invariant to ring start/winding and
    mapping order, and uses the emitter's own placed-nanometre coordinate
    currency.  It lets downstream GDS ownership replay reject a truncated or
    substituted in-memory stamp even when the remaining polygons still occur in
    the final flat GDS.
    """

    from clay.layout.placed_geometry import placed_point_nm

    if not isinstance(geometry, Mapping) or not isinstance(placement_um, Mapping):
        raise TypeError("emitted device snapshot requires geometry and placement mappings")
    if type(geometry) is dict and type(placement_um) is dict:
        from clay._native import load_native as _load_native_eng

        _eng = _load_native_eng("_engine_cpp")
        if _eng is not None and hasattr(_eng, "canonical_emitted_device_snapshot_records_cpp"):
            return _sha256_payload(
                _eng.canonical_emitted_device_snapshot_records_cpp(geometry, placement_um)
            )
    records = []
    for raw_name, stamp in sorted(geometry.items(), key=lambda item: str(item[0])):
        name = str(raw_name)
        raw_members = getattr(stamp, "composite_members", ())
        if raw_members is None:
            raw_members = ()
        if (
            type(raw_members) is not tuple
            or len(raw_members) != len(set(raw_members))
            or any(type(member) is not str or not member for member in raw_members)
        ):
            raise ValueError(
                f"emitted device snapshot {name!r} has malformed composite members"
            )
        composite_members = tuple(raw_members)
        position = placement_um.get(raw_name)
        if position is None and name != raw_name:
            position = placement_um.get(name)
        if position is None:
            continue  # mirrors the canonical emitter's non-emission rule
        if type(position) not in (tuple, list) or len(position) != 2:
            raise ValueError("emitted device snapshot has invalid placement")
        layers = getattr(stamp, "polys_by_layer", None)
        if not isinstance(layers, Mapping):
            raise TypeError("emitted device snapshot stamp lacks polys_by_layer")
        for raw_key, raw_rings in sorted(
            layers.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
        ):
            try:
                key = (int(raw_key[0]), int(raw_key[1]))
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError("emitted device snapshot has invalid layer key") from exc
            if not isinstance(raw_rings, (tuple, list)):
                raise TypeError("emitted device snapshot rings must be a sequence")
            for raw_ring in raw_rings:
                try:
                    points = tuple(
                        placed_point_nm(float(x), float(y), tuple(position))
                        for x, y in tuple(raw_ring)
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "emitted device snapshot contains an invalid polygon"
                    ) from exc
                records.append(
                    (name, composite_members, key, _canonical_int_ring(points))
                )
    return _sha256_payload(tuple(sorted(records)))


def _terminal_pin_snapshot_sha256(pins_by_net: Mapping[str, Any]) -> str:
    """Digest the exact terminal accesses handed to route and emit."""

    from clay.layout.route.terminal_identity import pin_binding_payload

    if not isinstance(pins_by_net, Mapping):
        raise TypeError("terminal pin snapshot requires a net mapping")
    records = []
    for raw_net, raw_pins in sorted(pins_by_net.items()):
        if type(raw_net) is not str or not raw_net:
            raise TypeError("terminal pin snapshot net must be non-empty")
        if type(raw_pins) not in (tuple, list):
            raise TypeError("terminal pin snapshot values must be exact sequences")
        for pin in raw_pins:
            net = getattr(pin, "net", None)
            device = getattr(pin, "device", None)
            terminal = getattr(pin, "terminal", None)
            layer = getattr(pin, "layer", None)
            point = getattr(pin, "xy_nm", None)
            if point is None:
                xy_um = getattr(pin, "xy_um", None)
                if (
                    type(xy_um) is tuple
                    and len(xy_um) == 2
                    and all(
                        type(value) in (int, float) and math.isfinite(float(value))
                        for value in xy_um
                    )
                ):
                    point = tuple(
                        int(round(float(value) * 1000.0)) for value in xy_um
                    )
            if (
                type(net) is not str
                or net != raw_net
                or type(device) is not str
                or not device
                or type(terminal) is not str
                or not terminal
                or type(layer) is not tuple
                or len(layer) != 2
                or any(type(value) is not int for value in layer)
                or type(point) is not tuple
                or len(point) != 2
                or any(type(value) is not int for value in point)
            ):
                raise TypeError("terminal pin snapshot contains a noncanonical pin")
            record = (net, device, terminal, layer, point)
            if pin_binding_payload(pin) is not None:
                # Preserve the exact legacy record. Typed accesses additionally
                # bind the complete source ledger, independent of seed position.
                record += (pin.terminal_binding.canonical_tuple(),)
            records.append(record)
            if len(records) > _MAX_ENTRIES:
                raise ValueError("terminal pin snapshot is too large")
    return _sha256_payload(tuple(sorted(records)))


def _guard_ring_member_classes_sha256(value: Sequence[Any]) -> str:
    """Canonical digest of source-compiled guard-ring member classes.

    The exact emitter seals this digest alongside the GDS/device snapshot.  A
    final checker may replay the typed inventory, but it cannot replace MOS with
    passive (or vice versa) after emission without invalidating the scope.
    """

    if type(value) is not tuple:
        raise TypeError("guard-ring member classes must be an exact tuple")
    rows = []
    for row in value:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or not row[0]
            or row[0] != row[0].strip()
            or type(row[1]) is not str
            or row[1] not in ("mos", "passive")
        ):
            raise TypeError("guard-ring member classes must be exact typed pairs")
        rows.append((row[0], row[1]))
    canonical = tuple(sorted(rows))
    if len(set(name for name, _kind in canonical)) != len(canonical):
        raise ValueError("guard-ring member classes contain duplicate members")
    if tuple(value) != canonical:
        raise ValueError("guard-ring member classes must be canonical and sorted")
    return _sha256_payload(canonical)


_SOURCE_MOS_CLASSES = frozenset(
    {"nmos", "pmos", "nmos3v", "pmos3v", "nch", "pch", "nch3", "pch3", "nd", "pd"}
)


def _canonical_source_scalar(value: Any) -> Any:
    if value is None or type(value) in (str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("source member inventory contains a non-finite value")
        return value
    raise TypeError(
        "source member inventory values must be exact str/int/float/None scalars"
    )


def _canonical_source_member_inventory(source_cell: Any) -> Tuple[Any, ...]:
    """Canonical topology/sizing inventory of one parsed source ``CellDef``."""

    if source_cell is None:
        return ()
    raw_devices = getattr(source_cell, "devices", None)
    if type(raw_devices) not in (tuple, list):
        raise TypeError("source cell devices must be an exact tuple/list")
    if len(raw_devices) > _MAX_ENTRIES:
        raise ValueError("source member inventory is too large")
    rows = []
    for device in raw_devices:
        name = getattr(device, "instance_name", None)
        device_class = getattr(device, "device_class", None)
        if (
            type(name) is not str
            or not name
            or name != name.strip()
            or type(device_class) is not str
            or not device_class
            or device_class != device_class.strip()
        ):
            raise TypeError("source member inventory contains an invalid identity")
        pins = getattr(device, "pins", None)
        sizing = getattr(device, "sizing", None)
        if not isinstance(pins, Mapping) or not isinstance(sizing, Mapping):
            raise TypeError("source member inventory lacks pins or sizing")
        canonical_pins = []
        for role, net in pins.items():
            if (
                type(role) is not str
                or not role
                or role != role.strip()
                or type(net) is not str
                or not net
                or net != net.strip()
            ):
                raise TypeError("source member inventory contains an invalid pin")
            canonical_pins.append((role, net))
        canonical_sizing = []
        for key, value in sizing.items():
            if type(key) is not str or not key or key != key.strip():
                raise TypeError("source member inventory contains an invalid sizing key")
            canonical_sizing.append((key, _canonical_source_scalar(value)))
        pdk_variant = getattr(device, "pdk_variant", None)
        source_model = getattr(device, "source_model", None)
        if pdk_variant is not None and type(pdk_variant) is not str:
            raise TypeError("source member inventory has an invalid PDK variant")
        if source_model is not None and type(source_model) is not str:
            raise TypeError("source member inventory has an invalid source model")
        record = (
            name,
            device_class,
            pdk_variant,
            source_model,
            tuple(sorted(canonical_pins)),
            tuple(sorted(canonical_sizing)),
        )
        if device_class == "native":
            parameters = getattr(device, "native_parameters", None)
            if (
                type(parameters) is not tuple
                or not parameters
                or any(
                    type(pair) is not tuple
                    or len(pair) != 2
                    or any(type(value) is not str for value in pair)
                    for pair in parameters
                )
            ):
                raise TypeError("source native parameters require exact string pairs")
            # Native W/L/multiplicity live in their original parameter tokens,
            # not portable sizing. Preserve ordinary inventory bytes unchanged.
            record += (parameters,)
        rows.append(record)
    canonical = tuple(sorted(rows))
    if len({row[0] for row in canonical}) != len(canonical):
        raise ValueError("source member inventory contains duplicate instances")
    return canonical


def _source_member_inventory_sha256(source_cell: Any) -> str:
    return _sha256_payload(_canonical_source_member_inventory(source_cell))


def _guard_ring_member_classes_from_source_cell(
    source_cell: Any,
) -> Tuple[Tuple[str, str], ...]:
    rows = []
    for row in _canonical_source_member_inventory(source_cell):
        name, raw_class = row[0], row[1]
        device_class = raw_class.lower()
        if device_class in _SOURCE_MOS_CLASSES:
            rows.append((name, "mos"))
        elif device_class in ("res", "cap"):
            rows.append((name, "passive"))
    return tuple(sorted(rows))


def _guard_ring_member_polarities_from_source_cell(
    source_cell: Any,
) -> Tuple[Tuple[str, str], ...]:
    """Canonical MOS polarity/passive role for guard-ring replay.

    The source inventory digest already seals the exact device class.  This
    projection preserves the polarity needed to distinguish a solid shared
    PMOS N-well carrier from a passive/foreign isolation N-well ring.
    """

    nmos_classes = frozenset({"nmos", "nmos3v", "nch", "nch3", "nd"})
    pmos_classes = frozenset({"pmos", "pmos3v", "pch", "pch3", "pd"})
    rows = []
    for row in _canonical_source_member_inventory(source_cell):
        name, raw_class = row[0], row[1]
        device_class = raw_class.lower()
        if device_class in nmos_classes:
            rows.append((name, "nmos"))
        elif device_class in pmos_classes:
            rows.append((name, "pmos"))
        elif device_class in ("res", "cap"):
            rows.append((name, "passive"))
    return tuple(sorted(rows))


def _replayed_source_cell_authority(
    source_cell: Any,
    *,
    expected_pdk_id: Optional[str] = None,
) -> Tuple[str, str, Tuple[Tuple[str, str], ...], Any]:
    """Reparse the source bytes and bind the exact CellDef inventory.

    A bridged routing view is deliberately not an authority here: it is a public,
    caller-replaceable projection.  The emitter and final checker both replay the
    source file named by the parser span and require its selected cell to equal the
    supplied effective ``CellDef`` inventory.
    """

    if source_cell is None:
        return _sha256_payload(None), _sha256_payload(()), (), None
    local_name = getattr(source_cell, "local_name", None)
    source_span = getattr(source_cell, "source_span", None)
    if type(local_name) is not str or not local_name or local_name != local_name.strip():
        raise TypeError("source member inventory lacks a canonical cell name")
    if not isinstance(source_span, Mapping):
        raise TypeError("source member inventory lacks parser source authority")
    raw_path = source_span.get("file")
    if type(raw_path) is not str or not raw_path:
        raise TypeError("source member inventory lacks a source file")
    path = Path(raw_path).resolve()
    before = path.read_bytes()
    from clay.parser import parse_clay_file

    parsed = parse_clay_file(str(path))
    after = path.read_bytes()
    if before != after:
        raise ValueError("source member inventory changed while it was replayed")
    matches = tuple(
        cell
        for cell in tuple(getattr(parsed, "cell_defs", ()) or ())
        if getattr(cell, "local_name", None) == local_name
    )
    if len(matches) != 1:
        raise ValueError("source member inventory cell identity is not unique")
    if expected_pdk_id is not None and parsed.pdk_id != expected_pdk_id:
        raise ValueError("source member inventory PDK differs from emission authority")
    if any(
        getattr(device, "device_class", None) == "native"
        for cell in (source_cell, matches[0])
        for device in tuple(getattr(cell, "devices", ()) or ())
    ):
        from clay.frontend.native_spice_import import native_cell_signature

        # The parsed file owns the PDK. This also keeps the existing complete
        # all-native source contract, including model, terminal and token rules.
        expected_native = native_cell_signature(source_cell, pdk_id=parsed.pdk_id)
        replayed_native = native_cell_signature(matches[0], pdk_id=parsed.pdk_id)
        if expected_native != replayed_native:
            raise ValueError("source native inventory differs from canonical source bytes")
    expected = _canonical_source_member_inventory(source_cell)
    replayed = _canonical_source_member_inventory(matches[0])
    if expected != replayed:
        raise ValueError("source member inventory differs from canonical source bytes")
    canonical_cell = matches[0]
    return (
        hashlib.sha256(before).hexdigest(),
        _sha256_payload(replayed),
        _guard_ring_member_classes_from_source_cell(canonical_cell),
        canonical_cell,
    )


def _source_cell_authority(
    source_cell: Any,
) -> Tuple[str, str, Tuple[Tuple[str, str], ...]]:
    """Return source identity while keeping the canonical replay internal."""

    source_sha256, inventory_sha256, member_classes, _canonical_cell = (
        _replayed_source_cell_authority(source_cell)
    )
    return source_sha256, inventory_sha256, member_classes


def _compiled_guard_ring_member_classes(
    devices: Sequence[Any],
    passives: Sequence[Any],
) -> Tuple[Tuple[str, str], ...]:
    """Replay member class from canonical source-compiled route inputs.

    ``devices`` and ``passives`` are the same bridged netlist views consumed by
    routing/emission.  No caller-provided ``name -> class`` assertion is
    accepted at this authority boundary.
    """

    if type(devices) is not tuple or type(passives) is not tuple:
        raise TypeError("source device/passive inventories must be exact tuples")
    rows = []
    for inventory, member_class in ((devices, "mos"), (passives, "passive")):
        for item in inventory:
            name = getattr(item, "name", None)
            if type(name) is not str or not name or name != name.strip():
                raise TypeError("source member inventory contains an invalid name")
            kind = getattr(item, "kind", None)
            if type(kind) is not str or not kind:
                raise TypeError(f"source member {name!r} lacks an exact kind")
            normalized = kind.lower()
            if member_class == "mos" and normalized not in ("nmos", "pmos"):
                raise ValueError(
                    f"source device {name!r} has non-MOS kind {kind!r}"
                )
            if member_class == "passive" and normalized not in ("res", "cap"):
                raise ValueError(
                    f"source passive {name!r} has non-passive kind {kind!r}"
                )
            rows.append((name, member_class))
    canonical = tuple(sorted(rows))
    if len({name for name, _kind in canonical}) != len(canonical):
        raise ValueError("source member inventories overlap or contain duplicates")
    return canonical


def _require_native_source_route_projection(
    source_cell: Any,
    devices: Tuple[Any, ...],
    passives: Tuple[Any, ...],
    native_geometry: Any,
    pdk_id: Any,
) -> None:
    """Replay the existing native bridge without assigning guard-ring classes.

    The installed native provider currently admits complete SKY130 native cells.
    Its live bundle owns the actual model, dimensions, terminal roles and observed
    well conductor. A routed kind/name assertion cannot substitute for it.
    """
    from clay.layout.native_sky130 import (
        NativeGeometryBundle,
        NativeRouteDevice,
        native_route_devices,
    )

    if type(native_geometry) is not NativeGeometryBundle or pdk_id != "sky130":
        raise ValueError("native exact emission requires the bound SKY130 geometry authority")
    if type(devices) is not tuple or type(passives) is not tuple or passives:
        raise ValueError("native source route projection requires one complete device inventory")
    if any(type(device) is not NativeRouteDevice for device in devices):
        raise TypeError("native source route projection requires exact NativeRouteDevice records")
    names = tuple(device.name for device in devices)
    if any(type(name) is not str for name in names) or len(set(names)) != len(names):
        raise ValueError("native source route projection contains invalid or duplicate names")
    # native_route_devices replays require_current, every source/native spec,
    # runtime/artifact/producer identity, and the original observed well role.
    expected = tuple(native_route_devices(source_cell, native_geometry))
    expected_by_name = {device.name: device for device in expected}
    if set(names) != set(expected_by_name):
        raise ValueError("native source route projection member denominator differs")
    for actual in devices:
        original = expected_by_name[actual.name]
        for field in fields(NativeRouteDevice):
            value, wanted = getattr(actual, field.name), getattr(original, field.name)
            if type(value) is not type(wanted):
                raise TypeError(
                    f"native source route projection {actual.name!r}.{field.name} has a noncanonical type"
                )
            if type(value) is dict and any(
                type(key) is not str or type(net) is not str
                for key, net in value.items()
            ):
                raise TypeError("native source route projection terminal maps require exact strings")
            if value != wanted:
                raise ValueError(
                    f"native source route projection {actual.name!r}.{field.name} differs from its geometry authority"
                )


def _canonical_ring(points: Sequence[Any]) -> Tuple[Tuple[int, int], ...]:
    ring = tuple((int(point.x), int(point.y)) for point in points)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if not ring:
        return ()
    forward = tuple(ring[index:] + ring[:index] for index in range(len(ring)))
    reverse_ring = tuple(reversed(ring))
    reverse = tuple(
        reverse_ring[index:] + reverse_ring[:index]
        for index in range(len(reverse_ring))
    )
    return min(forward + reverse)


def _canonical_region(region: Any) -> Tuple[Any, ...]:
    merged = region.merged()
    polygons = []
    for polygon in merged.each():
        hull = _canonical_ring(tuple(polygon.each_point_hull()))
        holes = tuple(
            sorted(
                _canonical_ring(tuple(polygon.each_point_hole(index)))
                for index in range(polygon.holes())
            )
        )
        polygons.append((hull, holes))
    return tuple(sorted(polygons))


def _net_layer_regions_sha256(regions: Mapping[str, Mapping[LayerKey, Any]]) -> str:
    """Canonical geometric-union digest, invariant to route fragmentation."""
    payload = []
    for net in sorted(regions):
        for raw_key in sorted(regions[net]):
            key = (int(raw_key[0]), int(raw_key[1]))
            canonical = _canonical_region(regions[net][raw_key])
            if canonical:
                payload.append((str(net), key, canonical))
    return _sha256_payload(payload)


def _cell_polygon_union_sha256(layout: Any, cell: Any) -> str:
    """Digest every polygon union in one flat emitter cell, excluding labels."""
    import klayout.db as kdb

    payload = []
    for layer_index in layout.layer_indexes():
        info = layout.get_info(layer_index)
        region = kdb.Region(cell.begin_shapes_rec(layer_index))
        canonical = _canonical_region(region)
        if canonical:
            payload.append(((int(info.layer), int(info.datatype)), canonical))
    return _sha256_payload(sorted(payload))


def _read_gds_polygon_union_sha256(path: Path, top_cell: str) -> str:
    """Parse the just-written GDS and replay its full polygon union digest."""
    import klayout.db as kdb

    layout = kdb.Layout()
    try:
        layout.read(str(Path(path).resolve()))
    except Exception as exc:
        raise ValueError("exact emitted artifact is not a readable GDS") from exc
    cell = layout.cell(str(top_cell))
    if cell is None:
        raise ValueError(f"exact emitted GDS lacks top cell {top_cell!r}")
    return _cell_polygon_union_sha256(layout, cell)


def _entries(value: Sequence[Any]) -> Tuple[RouteManifestEntry, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("route ownership manifest must be a bounded tuple/list")
    if len(value) > _MAX_ENTRIES:
        raise ValueError(f"route ownership manifest exceeds {_MAX_ENTRIES} entries")
    out = []
    for index, entry in enumerate(value):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise TypeError(f"malformed route ownership entry at index {index}")
        net = str(entry[0]).strip()
        if not net:
            raise ValueError(f"route ownership entry {index} has unknown owner")
        try:
            key = (int(entry[1][0]), int(entry[1][1]))
            box = tuple(int(v) for v in entry[2])
        except Exception as exc:
            raise TypeError(f"malformed route ownership entry at index {index}") from exc
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"route ownership entry {index} has invalid half-open bbox")
        out.append((net, key, box))
    canonical = tuple(out)
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate route ownership manifest entry")
    return canonical


def route_owned_manifest_from_routed_geometry(routed: Any) -> Tuple[RouteManifestEntry, ...]:
    """Build exhaustive ownership entries from the router's actual rectangles.

    This is a geometry/provenance operation, not an optimality certificate.  It
    accepts every route-owned rectangle exactly once and fails closed if a route
    shape cannot be represented by the emitter's rectangular manifest currency.
    The canonical emitter independently compares the resulting per-net/layer
    unions with the router output and with the written exact GDS.
    """

    import klayout.db as kdb

    entries = set()
    nets = getattr(routed, "nets", None)
    if not isinstance(nets, Mapping):
        raise TypeError("routed geometry lacks a net mapping")
    for raw_net, net_route in sorted(nets.items(), key=lambda item: str(item[0])):
        net = str(raw_net).strip()
        if not net:
            raise ValueError("routed geometry contains an ownerless net")
        for wire in (getattr(net_route, "wires", ()) or ()):
            key = (int(wire.layer), int(wire.datatype))
            points = tuple(
                (int(point.x), int(point.y))
                for point in wire.polygon.vertices
            )
            if len(points) < 4:
                raise ValueError("route-owned polygon has fewer than four vertices")
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            box = (min(xs), min(ys), max(xs), max(ys))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("route-owned polygon has an empty bbox")
            polygon_region = kdb.Region(kdb.Polygon([
                kdb.Point(x, y) for x, y in points
            ]))
            box_region = kdb.Region(kdb.Box(*box))
            if not (
                (polygon_region - box_region).is_empty()
                and (box_region - polygon_region).is_empty()
            ):
                raise ValueError(
                    "route-owned polygon is not exactly representable by one bbox"
                )
            entries.add((net, key, box))
            if len(entries) > _MAX_ENTRIES:
                raise ValueError(
                    f"route ownership manifest exceeds {_MAX_ENTRIES} entries"
                )
    return _entries(tuple(sorted(entries)))


@dataclass(frozen=True, init=False)
class RouteEmissionScopeCertificate:
    """Emitter proof that declared ownership equals emitted route geometry.

    Merge-only fragmentation changes are harmless because the proof compares
    per-net/per-layer geometric unions.  Any emitter-added heal/fill material or
    any manufacturing-grid movement makes the proof non-exhaustive and therefore
    cannot produce :class:`ExactEmittedArtifact`.
    """

    allocated_union_matches_router: bool
    allocated_route_manifest_sha256: str
    emitted_route_manifest_sha256: str
    allocated_route_union_sha256: str
    router_route_union_sha256: str
    emitted_route_union_sha256: str
    emitted_device_snapshot_sha256: str
    terminal_pin_snapshot_sha256: str
    source_cell_source_sha256: str
    source_member_inventory_sha256: str
    guard_ring_member_classes_sha256: str
    pre_transform_geometry_sha256: str
    post_transform_geometry_sha256: str
    written_geometry_sha256: str
    exact_gds_sha256: str
    top_cell: str
    heal_shape_count: int
    upper_fill_shape_count: int
    post_assembly_fill_shape_count: int
    snapped_shape_count: int

    def _validate(self) -> None:
        if type(self.allocated_union_matches_router) is not bool:
            raise TypeError("allocated_union_matches_router must be bool")
        for name in (
            "heal_shape_count",
            "upper_fill_shape_count",
            "post_assembly_fill_shape_count",
            "snapped_shape_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "allocated_route_manifest_sha256",
            "emitted_route_manifest_sha256",
            "allocated_route_union_sha256",
            "router_route_union_sha256",
            "emitted_route_union_sha256",
            "emitted_device_snapshot_sha256",
            "terminal_pin_snapshot_sha256",
            "source_cell_source_sha256",
            "source_member_inventory_sha256",
            "guard_ring_member_classes_sha256",
            "pre_transform_geometry_sha256",
            "post_transform_geometry_sha256",
            "written_geometry_sha256",
            "exact_gds_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not str(self.top_cell).strip():
            raise ValueError("scope certificate lacks a top cell")

    @property
    def emitter_added_shape_count(self) -> int:
        return (
            self.heal_shape_count
            + self.upper_fill_shape_count
            + self.post_assembly_fill_shape_count
        )

    @property
    def exhaustive(self) -> bool:
        return bool(
            self.allocated_union_matches_router
            and self.allocated_route_manifest_sha256
            == self.emitted_route_manifest_sha256
            and self.allocated_route_union_sha256
            == self.router_route_union_sha256
            == self.emitted_route_union_sha256
            and self.pre_transform_geometry_sha256
            == self.post_transform_geometry_sha256
            == self.written_geometry_sha256
            and self.emitter_added_shape_count == 0
            and self.snapped_shape_count == 0
        )


@dataclass(frozen=True)
class EmittedRouteOwnershipManifest:
    """One artifact-bound, exhaustive router-owned geometry manifest."""

    entries: Tuple[RouteManifestEntry, ...]
    artifact_sha256: str
    route_manifest_sha256: str
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        canonical = _entries(self.entries)
        object.__setattr__(self, "entries", canonical)
        if self.schema != _SCHEMA:
            raise ValueError(f"unsupported emitted route manifest schema {self.schema!r}")
        digest = route_manifest_sha256(canonical)
        if digest != self.route_manifest_sha256:
            raise ValueError("emitted route manifest digest does not replay")
        if len(self.artifact_sha256) != 64:
            raise ValueError("emitted route manifest lacks an exact artifact SHA-256")


@dataclass(frozen=True, init=False)
class ExactEmittedArtifact:
    """Exact GDS plus the only route manifest certified exhaustive for it."""

    exact_gds_path: Path
    truth_manifest: TruthManifest
    route_manifest: EmittedRouteOwnershipManifest
    scope: RouteEmissionScopeCertificate
    well_merge_certificates: Tuple[Any, ...]

    def _validate(self) -> None:
        path = Path(self.exact_gds_path).resolve()
        object.__setattr__(self, "exact_gds_path", path)
        if type(self.truth_manifest) is not TruthManifest:
            raise TypeError("exact emitted artifact requires TruthManifest")
        if type(self.route_manifest) is not EmittedRouteOwnershipManifest:
            raise TypeError("exact emitted artifact requires emitted ownership manifest")
        if type(self.scope) is not RouteEmissionScopeCertificate or not self.scope.exhaustive:
            raise ValueError("emitted route ownership is not exhaustive")
        from clay.layout.spec.exact_structure_profile import WellMergeCertificate

        if type(self.well_merge_certificates) is not tuple or any(
            type(item) is not WellMergeCertificate
            for item in self.well_merge_certificates
        ):
            raise TypeError("well merge certificates must be an exact typed tuple")
        if len({item.layer_key for item in self.well_merge_certificates}) != len(
            self.well_merge_certificates
        ):
            raise ValueError("well merge certificates must have unique layer keys")
        self.scope._validate()
        self.truth_manifest.bind_verdict_gds(path)
        self.truth_manifest.bind_claimed_sha256(
            self.scope.exact_gds_sha256,
            what="emitter scope certificate",
        )
        self.truth_manifest.bind_claimed_sha256(
            self.route_manifest.artifact_sha256,
            what="emitted route ownership manifest",
        )
        if (
            self.scope.allocated_route_manifest_sha256
            != self.route_manifest.route_manifest_sha256
            or self.scope.emitted_route_manifest_sha256
            != self.route_manifest.route_manifest_sha256
        ):
            raise ValueError("emitter scope certificate does not bind the route manifest")
        claimed_route = self.truth_manifest.route_manifest_sha256
        if (
            claimed_route is not None
            and claimed_route != self.route_manifest.route_manifest_sha256
        ):
            from clay.layout.route.truth_manifest import ArtifactDigestMismatch

            raise ArtifactDigestMismatch(
                "signoff route manifest does not bind the exact emitted artifact: "
                f"expected {self.route_manifest.route_manifest_sha256}, "
                f"got {claimed_route}"
            )

    def bind_truth_manifest(self, truth_manifest: TruthManifest) -> "ExactEmittedArtifact":
        """Return the same emitted proof bound to stronger immutable provenance."""
        if type(truth_manifest) is not TruthManifest:
            raise TypeError("exact emitted artifact requires TruthManifest")
        rebound = object.__new__(ExactEmittedArtifact)
        object.__setattr__(rebound, "exact_gds_path", self.exact_gds_path)
        object.__setattr__(rebound, "truth_manifest", truth_manifest)
        object.__setattr__(rebound, "route_manifest", self.route_manifest)
        object.__setattr__(rebound, "scope", self.scope)
        object.__setattr__(
            rebound, "well_merge_certificates", self.well_merge_certificates
        )
        rebound.validate()
        return rebound

    def validate(self) -> None:
        self._validate()
        self.scope._validate()
        self.truth_manifest.validate_current_artifact()
        self.truth_manifest.bind_claimed_sha256(
            self.route_manifest.artifact_sha256,
            what="emitted route ownership manifest",
        )
        if route_manifest_sha256(self.route_manifest.entries) != self.route_manifest.route_manifest_sha256:
            raise ValueError("emitted route manifest digest no longer replays")


def _finalize_verified_emission(
    path: Path,
    entries: Sequence[Any],
    *,
    top_cell: str,
    allocated_route_union_sha256: str,
    router_route_union_sha256: str,
    emitted_route_union_sha256: str,
    emitted_device_snapshot_sha256: str,
    terminal_pin_snapshot_sha256: str,
    pre_transform_geometry_sha256: str,
    post_transform_geometry_sha256: str,
    heal_shape_count: int,
    upper_fill_shape_count: int,
    post_assembly_fill_shape_count: int,
    snapped_shape_count: int,
    source_cell: Any = None,
    source_devices: Sequence[Any] = (),
    source_passives: Sequence[Any] = (),
    source_native_geometry: Any = None,
    source_pdk_id: Optional[str] = None,
    source_tech: Any = None,
    emitted_device_geometry: Optional[Mapping[str, Any]] = None,
    emitted_device_placement_um: Optional[Mapping[str, Any]] = None,
    well_merge_certificates: Sequence[Any] = (),
    transformation_evidence: Sequence[Mapping[str, Any]] = (),
) -> ExactEmittedArtifact:
    """Emitter-only finalizer: parse/replay the written GDS before issuing proof.

    This intentionally is not a public constructor.  Callers cannot assert a
    success boolean or hand in a certificate.  The canonical emitter supplies
    its independently measured pre/post unions, while this finalizer computes
    the artifact and written-geometry digests itself.
    """
    canonical = _entries(entries)
    path = Path(path).resolve()
    truth = TruthManifest.from_artifact(path)
    written_geometry_sha256 = _read_gds_polygon_union_sha256(path, top_cell)
    manifest_sha256 = route_manifest_sha256(canonical)
    (
        source_cell_source_sha256,
        source_member_inventory_sha256,
        source_member_classes,
        canonical_source_cell,
    ) = _replayed_source_cell_authority(
        source_cell,
        expected_pdk_id=(
            source_pdk_id
            if any(getattr(device, "device_class", None) == "native"
                   for device in tuple(getattr(source_cell, "devices", ()) or ()))
            else None
        ),
    )
    projected_devices, projected_passives = tuple(source_devices), tuple(source_passives)
    if any(getattr(device, "device_class", None) == "native"
           for device in tuple(getattr(canonical_source_cell, "devices", ()) or ())):
        _require_native_source_route_projection(
            canonical_source_cell, projected_devices, projected_passives,
            source_native_geometry, source_pdk_id,
        )
        # Native declarations have no established guard-ring class projection.
        # The complete native inventory was checked above; do not invent one.
        projected_devices, projected_passives = (), ()
    elif source_native_geometry is not None:
        raise ValueError("non-native exact emission received unexpected native geometry authority")
    if source_cell is not None:
        projected_member_classes = _compiled_guard_ring_member_classes(
            projected_devices, projected_passives
        )
        canonical_names = {name for name, _member_class in source_member_classes}
        projected_source_members = tuple(
            row for row in projected_member_classes if row[0] in canonical_names
        )
        if projected_source_members != source_member_classes:
            raise ValueError(
                "source member inventory differs from routed device/passive projection"
            )
    scope = object.__new__(RouteEmissionScopeCertificate)
    for name, value in (
        ("allocated_union_matches_router", allocated_route_union_sha256 == router_route_union_sha256),
        ("allocated_route_manifest_sha256", manifest_sha256),
        ("emitted_route_manifest_sha256", manifest_sha256),
        ("allocated_route_union_sha256", str(allocated_route_union_sha256)),
        ("router_route_union_sha256", str(router_route_union_sha256)),
        ("emitted_route_union_sha256", str(emitted_route_union_sha256)),
        ("emitted_device_snapshot_sha256", str(emitted_device_snapshot_sha256)),
        ("terminal_pin_snapshot_sha256", str(terminal_pin_snapshot_sha256)),
        ("source_cell_source_sha256", source_cell_source_sha256),
        ("source_member_inventory_sha256", source_member_inventory_sha256),
        (
            "guard_ring_member_classes_sha256",
            _guard_ring_member_classes_sha256(source_member_classes),
        ),
        ("pre_transform_geometry_sha256", str(pre_transform_geometry_sha256)),
        ("post_transform_geometry_sha256", str(post_transform_geometry_sha256)),
        ("written_geometry_sha256", written_geometry_sha256),
        ("exact_gds_sha256", truth.exact_gds_sha256),
        ("top_cell", str(top_cell)),
        ("heal_shape_count", heal_shape_count),
        ("upper_fill_shape_count", upper_fill_shape_count),
        ("post_assembly_fill_shape_count", post_assembly_fill_shape_count),
        ("snapped_shape_count", snapped_shape_count),
    ):
        object.__setattr__(scope, name, value)
    scope._validate()
    if not scope.exhaustive:
        evidence = tuple(dict(item) for item in transformation_evidence)
        raise ValueError(
            "emitter transformations prevent an exhaustive exact-route manifest: "
            f"heal={scope.heal_shape_count} upper_fill={scope.upper_fill_shape_count} "
            f"post_fill={scope.post_assembly_fill_shape_count} "
            f"snapped={scope.snapped_shape_count}; "
            "transform_evidence="
            + json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        )
    manifest = EmittedRouteOwnershipManifest(
        entries=canonical,
        artifact_sha256=truth.exact_gds_sha256,
        route_manifest_sha256=manifest_sha256,
    )
    artifact = object.__new__(ExactEmittedArtifact)
    from clay.layout.spec.exact_structure_profile import (
        WellMergeCertificate,
        _region_sha256,
    )
    import klayout.db as kdb

    typed_well_certificates = tuple(well_merge_certificates)
    if any(type(item) is not WellMergeCertificate for item in typed_well_certificates):
        raise TypeError("emitter well merge certificates must be exact")
    check_layout = kdb.Layout()
    check_layout.read(str(path))
    check_cell = check_layout.cell(str(top_cell))
    if check_cell is None:
        raise ValueError("exact emitted GDS lacks the certified top cell")
    boundary_tech = source_tech
    if source_tech is not None and source_pdk_id != source_tech.pdk_id:
        raise ValueError("assembly boundary technology differs from emitter PDK")
    if source_pdk_id is not None:
        from clay.layout.tech import LayoutTech

        try:
            boundary_tech = LayoutTech.resolve(source_pdk_id)
        except KeyError:
            # Synthetic exact-accounting callers have no registered physical
            # PDK. They cannot declare an authoritative boundary purpose.
            from clay.layout.assembly_boundary import _boundary_key

            if source_tech is not None and _boundary_key(source_tech) is not None:
                raise ValueError("assembly boundary requires a registered emitter PDK")
    if boundary_tech is not None:
        from clay.layout.assembly_boundary import require_assembly_boundary

        if source_pdk_id != boundary_tech.pdk_id:
            raise ValueError("assembly boundary technology differs from emitter PDK")
        if _emitted_device_snapshot_sha256(
            emitted_device_geometry, emitted_device_placement_um
        ) != emitted_device_snapshot_sha256:
            raise ValueError("assembly boundary original device snapshot does not replay")
        require_assembly_boundary(
            check_layout, check_cell, boundary_tech,
            emitted_device_geometry, emitted_device_placement_um,
        )
    for certificate in typed_well_certificates:
        layer_index = check_layout.find_layer(*certificate.layer_key)
        written = (
            kdb.Region()
            if layer_index is None
            else kdb.Region(check_cell.begin_shapes_rec(layer_index))
        )
        written.merge()
        if _region_sha256(written) != certificate.output_region_sha256:
            raise ValueError("well merge certificate output differs from written GDS")
    object.__setattr__(artifact, "exact_gds_path", truth.exact_gds_path)
    object.__setattr__(artifact, "truth_manifest", truth)
    object.__setattr__(artifact, "route_manifest", manifest)
    object.__setattr__(artifact, "scope", scope)
    object.__setattr__(artifact, "well_merge_certificates", typed_well_certificates)
    artifact.validate()
    return artifact


@dataclass(frozen=True)
class ExactArtifactEvidenceBundle:
    """Precomputed evidence joined to exactly one GDS and route manifest."""

    artifact: ExactEmittedArtifact
    via_provenance_report: Any
    layer_route_profile_report: Any
    exact_structure_profile_report: Optional[Any] = None
    exact_structure_expectation: Optional[Any] = None
    row_final_placement_certificate: Optional[Any] = None
    final_placement_certificate: Optional[Any] = None
    final_placement_um: Optional[Mapping[str, Any]] = None
    final_placement_geometry: Optional[Mapping[str, Any]] = None
    route_cell_result: Optional[Any] = None
    lvs_reference_equivalence_certificate: Optional[Any] = None
    lvs_verdict: Optional[Any] = None
    drc_verdict: Optional[Any] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        from clay.layout.route.drc_gate import DrcOracleCertificate, DrcVerdict
        from clay.layout.route.lvs_gate import LvsOracleCertificate, LvsVerdict
        from clay.layout.route.truth_manifest import (
            ArtifactDigestMismatch,
            final_placement_artwork_sha256,
            sha256_path,
        )
        from clay.layout.place_blocks_2d import (
            FinalPlacementCertificate,
            final_placement_state_digest,
        )
        from clay.layout.route.via_provenance_check import ViaProvenanceReport
        from clay.layout.generic_placement import RowFinalPlacementCertificate
        from clay.layout.spec.exact_structure_profile import (
            ExactStructureExpectation,
            ExactStructureProfileReport,
        )
        from clay.layout.spec.geometric_fidelity import LayerRouteProfileReport

        if type(self.artifact) is not ExactEmittedArtifact:
            raise TypeError("exact evidence bundle requires ExactEmittedArtifact")
        self.artifact.validate()
        truth = self.artifact.truth_manifest
        expected_artifact = self.artifact.truth_manifest.exact_gds_sha256
        expected_manifest = self.artifact.route_manifest.route_manifest_sha256
        if truth.route_manifest_sha256 is not None and (
            truth.route_manifest_sha256 != expected_manifest
        ):
            raise ArtifactDigestMismatch(
                "signoff route manifest does not bind the exact evidence bundle"
            )

        final_certificate = self.final_placement_certificate
        final_placement = self.final_placement_um
        final_geometry = self.final_placement_geometry
        if truth.final_placement_sha256 is not None or (
            truth.final_placement_certificate_sha256 is not None
        ):
            if type(final_certificate) is not FinalPlacementCertificate:
                raise TypeError(
                    "exact evidence bundle requires typed FinalPlacementCertificate"
                )
            if not isinstance(final_placement, Mapping) or not isinstance(
                final_geometry, Mapping
            ):
                raise TypeError(
                    "exact evidence bundle requires actual final placement and geometry"
                )
            certificate_sha256 = final_placement_state_digest(final_certificate)
            if certificate_sha256 != truth.final_placement_certificate_sha256:
                raise ArtifactDigestMismatch(
                    "signoff final placement certificate does not replay"
                )
            artwork_sha256 = final_placement_artwork_sha256(
                final_placement, final_geometry
            )
            if artwork_sha256 != truth.final_placement_sha256:
                raise ArtifactDigestMismatch(
                    "signoff final placement artwork does not replay"
                )

        def validate_verdict_file(
            path_value: Any,
            claimed_sha256: Any,
            expected_sha256: Any,
            *,
            what: str,
        ) -> None:
            if not path_value:
                raise ArtifactDigestMismatch(f"signoff verdict lacks {what} path")
            path = Path(path_value).resolve()
            try:
                if path.stat().st_size <= 0:
                    raise ArtifactDigestMismatch(
                        f"signoff {what} must be a non-empty file: {path}"
                    )
                actual = sha256_path(path)
            except ArtifactDigestMismatch:
                raise
            except OSError as exc:
                raise ArtifactDigestMismatch(
                    f"signoff {what} is missing or unreadable: {path}"
                ) from exc
            if actual != claimed_sha256 or actual != expected_sha256:
                raise ArtifactDigestMismatch(
                    f"signoff {what} bytes do not replay: "
                    f"manifest={expected_sha256}, verdict={claimed_sha256}, "
                    f"actual={actual}"
                )
        for name, report, exact_type in (
            ("via provenance", self.via_provenance_report, ViaProvenanceReport),
            ("layer route profile", self.layer_route_profile_report, LayerRouteProfileReport),
        ):
            if type(report) is not exact_type:
                raise TypeError(f"exact evidence bundle requires typed {name} report")
            artifact_hash = str(getattr(report, "artifact_sha256", "") or "")
            manifest_hash = str(getattr(report, "route_manifest_sha256", "") or "")
            if artifact_hash != expected_artifact:
                raise ValueError(f"{name} report does not bind the same exact GDS")
            if manifest_hash != expected_manifest:
                raise ValueError(f"{name} report does not bind the same route manifest")

        structure_report = self.exact_structure_profile_report
        structure_expectation = self.exact_structure_expectation
        row_certificate = self.row_final_placement_certificate
        if structure_report is not None:
            if type(structure_report) is not ExactStructureProfileReport:
                raise TypeError(
                    "exact evidence bundle requires typed structure profile report"
                )
            if structure_report.artifact_sha256 != expected_artifact:
                raise ValueError(
                    "structure profile report does not bind the same exact GDS"
                )
            if structure_report.route_manifest_sha256 != expected_manifest:
                raise ValueError(
                    "structure profile report does not bind the same route manifest"
                )
            if type(row_certificate) is not RowFinalPlacementCertificate:
                raise TypeError(
                    "exact structure profile requires typed row final placement certificate"
                )
            if type(structure_expectation) is not ExactStructureExpectation:
                raise TypeError(
                    "exact structure profile requires typed structure expectation"
                )
            if structure_report.expectation_sha256 != structure_expectation.sha256:
                raise ValueError(
                    "structure profile report does not bind the supplied expectation"
                )
            if structure_report.row_final_certificate_sha256 != row_certificate.sha256:
                raise ValueError(
                    "structure profile report does not bind the supplied row final certificate"
                )
            if structure_report.row_program_sha256 != row_certificate.program_sha256:
                raise ValueError(
                    "structure profile row program differs from the row final certificate"
                )
            if (
                structure_expectation.row_program_sha256
                != row_certificate.program_sha256
                or structure_expectation.row_final_certificate_sha256
                != row_certificate.sha256
            ):
                raise ValueError(
                    "structure expectation does not bind the supplied final row master"
                )
            if (
                truth.row_final_placement_sha256 is not None
                and truth.row_final_placement_sha256 != row_certificate.sha256
            ):
                raise ArtifactDigestMismatch(
                    "signoff row-final placement certificate does not bind the bundle"
                )
        elif row_certificate is not None and type(row_certificate) is not RowFinalPlacementCertificate:
            raise TypeError(
                "row_final_placement_certificate must be exact when supplied"
            )
        elif structure_expectation is not None:
            raise ValueError(
                "exact structure expectation requires its measured structure report"
            )

        if self.lvs_verdict is not None:
            if type(self.lvs_verdict) is not LvsVerdict:
                raise TypeError("exact evidence bundle requires typed LvsVerdict")
            if type(self.lvs_verdict.ran) is not bool or type(self.lvs_verdict.matched) is not bool:
                raise TypeError("precomputed LVS verdict has non-boolean state")
            if not isinstance(self.lvs_verdict.unmatched, dict):
                raise TypeError("precomputed LVS verdict has malformed unmatched evidence")
            for key in ("nets", "devices"):
                if not isinstance(self.lvs_verdict.unmatched.get(key, []), (tuple, list)):
                    raise TypeError(
                        f"precomputed LVS verdict has malformed unmatched {key} evidence"
                    )
            if not self.lvs_verdict.gds_path:
                raise ValueError("precomputed LVS verdict lacks its GDS path")
            lvs_certificate = self.lvs_verdict.oracle_certificate
            if type(lvs_certificate) is not LvsOracleCertificate:
                raise ValueError("precomputed LVS lacks an oracle certificate")
            lvs_certificate._validate()
            if (
                self.lvs_verdict.full_extraction is not True
                or lvs_certificate.full_extraction is not True
            ):
                raise ValueError(
                    "exact evidence bundle requires full_extraction=True"
                )
            if lvs_certificate.returncode != 0:
                raise ValueError("precomputed LVS oracle returncode is nonzero")
            if (
                self.lvs_verdict.returncode != lvs_certificate.returncode
                or self.lvs_verdict.ran != lvs_certificate.ran
                or self.lvs_verdict.verbatim != lvs_certificate.verbatim
                or self.lvs_verdict.matched != lvs_certificate.matched
                or tuple(self.lvs_verdict.unmatched.get("nets", ()))
                != lvs_certificate.unmatched_nets
                or tuple(self.lvs_verdict.unmatched.get("devices", ()))
                != lvs_certificate.unmatched_devices
                or self.lvs_verdict.full_extraction
                != lvs_certificate.full_extraction
            ):
                raise ValueError(
                    "precomputed LVS verdict does not replay its oracle certificate"
                )
            truth.validate_signoff_provenance()
            validate_verdict_file(
                self.lvs_verdict.gds_path,
                self.lvs_verdict.gds_sha256,
                lvs_certificate.gds_sha256,
                what="exact emitted GDS",
            )
            validate_verdict_file(
                self.lvs_verdict.ref_path,
                self.lvs_verdict.ref_sha256,
                lvs_certificate.ref_sha256,
                what="reference netlist",
            )
            validate_verdict_file(
                self.lvs_verdict.deck_path,
                self.lvs_verdict.deck_sha256,
                lvs_certificate.deck_sha256,
                what="LVS deck",
            )
            validate_verdict_file(
                self.lvs_verdict.lvsdb_path,
                self.lvs_verdict.lvsdb_sha256,
                lvs_certificate.lvsdb_sha256,
                what="full-extraction LVS database",
            )
            validate_verdict_file(
                self.lvs_verdict.extracted_netlist_path,
                self.lvs_verdict.extracted_netlist_sha256,
                lvs_certificate.extracted_netlist_sha256,
                what="extracted netlist",
            )
            for name, certificate_hash, manifest_hash in (
                ("GDS", lvs_certificate.gds_sha256, truth.exact_gds_sha256),
                ("reference netlist", lvs_certificate.ref_sha256, truth.netlist_sha256),
                ("LVS deck", lvs_certificate.deck_sha256, truth.lvs_deck_sha256),
                (
                    "full-extraction LVS database",
                    lvs_certificate.lvsdb_sha256,
                    truth.full_extraction_lvsdb_sha256,
                ),
            ):
                if certificate_hash != manifest_hash:
                    raise ArtifactDigestMismatch(
                        f"LVS oracle certificate binds a different signoff {name}"
                    )
        if self.drc_verdict is not None:
            if type(self.drc_verdict) is not DrcVerdict:
                raise TypeError("exact evidence bundle requires typed DrcVerdict")
            if type(self.drc_verdict.ran) is not bool or type(self.drc_verdict.clean) is not bool:
                raise TypeError("precomputed DRC verdict has non-boolean state")
            if type(self.drc_verdict.total) is not int or self.drc_verdict.total < 0:
                raise ValueError("precomputed DRC verdict has an invalid total")
            if not isinstance(self.drc_verdict.counts_by_rule, dict) or any(
                type(count) is not int or count < 0
                for count in self.drc_verdict.counts_by_rule.values()
            ):
                raise TypeError("precomputed DRC verdict has malformed rule counts")
            if not self.drc_verdict.gds_path:
                raise ValueError("precomputed DRC verdict lacks its GDS path")
            if self.drc_verdict.ran:
                drc_certificate = self.drc_verdict.oracle_certificate
                if type(drc_certificate) is not DrcOracleCertificate:
                    raise ValueError("precomputed DRC lacks an oracle certificate")
                drc_certificate._validate()
                if (
                    self.drc_verdict.returncode != drc_certificate.returncode
                    or self.drc_verdict.clean != drc_certificate.clean
                    or self.drc_verdict.total != drc_certificate.total
                    or tuple(sorted(self.drc_verdict.counts_by_rule.items()))
                    != drc_certificate.counts_by_rule
                ):
                    raise ValueError(
                        "precomputed DRC verdict does not replay its oracle certificate"
                    )
                truth.validate_signoff_provenance()
                validate_verdict_file(
                    self.drc_verdict.gds_path,
                    self.drc_verdict.gds_sha256,
                    drc_certificate.gds_sha256,
                    what="exact emitted GDS",
                )
                validate_verdict_file(
                    self.drc_verdict.deck_path,
                    self.drc_verdict.deck_sha256,
                    drc_certificate.deck_sha256,
                    what="DRC deck",
                )
                validate_verdict_file(
                    self.drc_verdict.report_path,
                    self.drc_verdict.report_sha256,
                    drc_certificate.report_sha256,
                    what="DRC report database",
                )
                if (
                    drc_certificate.gds_sha256 != truth.exact_gds_sha256
                    or drc_certificate.deck_sha256 != truth.drc_deck_sha256
                ):
                    raise ArtifactDigestMismatch(
                        "DRC oracle certificate binds different GDS/deck bytes"
                    )

    def validate_reference_equivalence(self, cell_def: Any) -> bool:
        """Replay source-CIR -> current cell IR -> actual oracle reference."""
        from clay.layout.route.lvs_gate import (
            LvsReferenceEquivalenceCertificate,
        )
        from clay.layout.route.truth_manifest import (
            ArtifactDigestMismatch,
            sha256_path,
        )

        certificate = self.lvs_reference_equivalence_certificate
        if type(certificate) is not LvsReferenceEquivalenceCertificate:
            raise TypeError(
                "exact signoff requires an LVS reference equivalence certificate"
            )
        truth = self.artifact.truth_manifest
        source_path = truth.source_netlist_path
        reference_path = truth.reference_netlist_path
        if source_path is None or reference_path is None:
            raise ArtifactDigestMismatch(
                "exact signoff lacks source/reference netlist paths"
            )
        source_path = Path(source_path).resolve()
        reference_path = Path(reference_path).resolve()
        if (
            source_path != certificate.source_cir_path
            or sha256_path(source_path) != certificate.source_cir_sha256
            or truth.source_netlist_sha256 != certificate.source_cir_sha256
        ):
            raise ArtifactDigestMismatch(
                "LVS reference equivalence certificate binds another source CIR"
            )
        try:
            generated_text = reference_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ArtifactDigestMismatch(
                "generated LVS reference is missing or unreadable"
            ) from exc
        if (
            sha256_path(reference_path)
            != certificate.generated_reference_sha256
            or truth.netlist_sha256
            != certificate.generated_reference_sha256
        ):
            raise ArtifactDigestMismatch(
                "LVS reference equivalence certificate binds another generated reference"
            )
        certificate.replay(
            source_path,
            cell_def,
            generated_reference_text=generated_text,
        )
        return True

    def validate_for_route_cell_result(
        self, result: Any, cell_def: Any
    ) -> bool:
        """Replay this bundle against the exact current canonical route result."""
        from clay.layout.generic_placement import RowFinalPlacementCertificate
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.place_blocks_2d import (
            FinalPlacementCertificate,
            final_placement_state_digest,
        )
        from clay.layout.route.allocated_route_manifest import (
            replay_exact_allocated_route_manifest,
        )
        from clay.layout.route.route_cell import (
            RouteCellResult,
            _final_placement_provenance_digest,
            build_final_placement_resource_evidence,
        )
        from clay.layout.route.truth_manifest import (
            ArtifactDigestMismatch,
            final_placement_artwork_sha256,
        )

        self.validate()
        if type(result) is not RouteCellResult:
            raise TypeError(
                "exact signoff requires the current typed RouteCellResult"
            )
        if not (
            self.route_cell_result is result
            or (
                type(self.route_cell_result) is RouteCellResult
                and final_placement_artwork_sha256(
                    self.route_cell_result.placement_um,
                    self.route_cell_result.geometry,
                )
                == final_placement_artwork_sha256(
                    result.placement_um,
                    result.geometry,
                )
                and route_owned_manifest_from_routed_geometry(
                    self.route_cell_result.routed
                )
                == route_owned_manifest_from_routed_geometry(result.routed)
            )
        ):
            raise ValueError(
                "exact evidence bundle is not bound to the current RouteCellResult"
            )
        from clay.layout.row_terminal_probes import require_routing_row_terminal_probes as require_row_terminal_probes

        require_row_terminal_probes(
            tech=result.tech, geometry=result.geometry, placement_um=result.placement_um,
            pins_by_net=result.pins_by_net, row_program=result.row_program,
            row_materialization=result.row_materialization,
            row_final_certificate=result.row_final_placement_certificate,
            source_cell=cell_def,
            allow_fixed_multilayer=bool(getattr(getattr(result,'selected_route_engine',None),
                'consumes_fixed_multilayer_joint_terminals',False)),
            multilayer_engine=getattr(result,'selected_route_engine',None),
        )
        truth = self.artifact.truth_manifest
        current_certificate = result.final_placement_certificate
        if type(current_certificate) is not FinalPlacementCertificate or not (
            self.final_placement_certificate is current_certificate
            or (
                type(self.final_placement_certificate)
                is FinalPlacementCertificate
                and final_placement_state_digest(
                    self.final_placement_certificate
                )
                == final_placement_state_digest(current_certificate)
            )
        ):
            raise ValueError(
                "exact evidence bundle does not bind the current final placement certificate"
            )
        current_certificate_sha256 = final_placement_state_digest(
            current_certificate
        )
        if (
            not truth.final_placement_certificate_sha256
            or current_certificate_sha256
            != truth.final_placement_certificate_sha256
        ):
            raise ArtifactDigestMismatch(
                "current final placement certificate differs from signoff provenance"
            )
        from clay.layout.place.resource_feedback import (
            PlacementResourceClosureCertificate,
        )

        closure = getattr(result, "placement_resource_closure", None)
        certificate_closure = getattr(
            current_certificate, "resource_closure_certificate", None
        )
        closure_required = (
            result.candidate_pool is not None and result.resource_plan is not None
        )
        if closure_required:
            if (
                getattr(current_certificate, "resource_status", "UNAVAILABLE")
                != "PROVEN_FEASIBLE"
                or not isinstance(closure, PlacementResourceClosureCertificate)
                or not (
                    certificate_closure is closure
                    or certificate_closure == closure
                )
            ):
                raise ArtifactDigestMismatch(
                    "required RG4 resource closure is missing from the route result"
                )
        if closure is not None:
            if not isinstance(closure, PlacementResourceClosureCertificate):
                raise TypeError("route result has a foreign resource closure")
            if not (
                certificate_closure is closure
                or certificate_closure == closure
            ):
                raise ArtifactDigestMismatch(
                    "resource closure does not bind the final placement certificate"
                )
            closure.replay(
                pool=result.candidate_pool,
                plan=result.resource_plan,
                final_placement_certificate=current_certificate,
            )
        current_artwork_sha256 = final_placement_artwork_sha256(
            result.placement_um, result.geometry
        )
        if (
            not truth.final_placement_sha256
            or current_artwork_sha256 != truth.final_placement_sha256
        ):
            raise ArtifactDigestMismatch(
                "current placement/geometry differs from signoff provenance"
            )
        final_resource_evidence = build_final_placement_resource_evidence(
            result.placement_um,
            result.geometry,
            layout_intent_from_cell_def(cell_def),
            result.tech,
            final_placement_certificate=current_certificate,
        )
        replayed_final_sha256 = _final_placement_provenance_digest(
            final_resource_evidence, current_certificate
        )
        if replayed_final_sha256 != current_certificate_sha256:
            raise ArtifactDigestMismatch(
                "final placement certificate does not replay current placement/geometry"
            )

        row_certificate = result.row_final_placement_certificate
        if (
            type(row_certificate) is not RowFinalPlacementCertificate
            or not (
                self.row_final_placement_certificate is row_certificate
                or (
                    type(self.row_final_placement_certificate)
                    is RowFinalPlacementCertificate
                    and self.row_final_placement_certificate.sha256
                    == row_certificate.sha256
                )
            )
            or not truth.row_final_placement_sha256
            or row_certificate.sha256 != truth.row_final_placement_sha256
        ):
            raise ArtifactDigestMismatch(
                "current row-final placement certificate differs from signoff provenance"
            )
        native_row_binding = row_certificate.native_geometry_binding
        if native_row_binding is not None:
            if not (
                native_row_binding.certificate is current_certificate
                or (
                    type(native_row_binding.certificate)
                    is FinalPlacementCertificate
                    and final_placement_state_digest(
                        native_row_binding.certificate
                    )
                    == current_certificate_sha256
                )
            ):
                raise ArtifactDigestMismatch(
                    "row geometry binds a different retained native certificate"
                )
            # Replay the original device scope; later rail/guard artwork stays
            # in the full artifact and is subject to its own signoff checks.
            native_row_binding.require_actual_route_geometry(
                geometry={name: result.geometry[name] for name in native_row_binding.stamp_names},
                placement_um={name: result.placement_um[name] for name in native_row_binding.stamp_names},
                intent=layout_intent_from_cell_def(cell_def),
            )
        if (
            getattr(result.row_program, "sha256", None)
            != row_certificate.program_sha256
        ):
            raise ArtifactDigestMismatch(
                "current row program does not replay the row-final certificate"
            )
        materialization_sha256 = getattr(
            getattr(result.row_materialization, "certificate", None),
            "sha256",
            None,
        )
        if row_certificate.macro_bindings and (
            materialization_sha256 != row_certificate.materialization_sha256
        ):
            raise ArtifactDigestMismatch(
                "current row materialization does not replay the row-final certificate"
            )

        has_pool = result.candidate_pool is not None
        has_plan = result.resource_plan is not None
        if has_pool != has_plan:
            raise ArtifactDigestMismatch(
                "route result has a partial candidate-pool/plan lineage"
            )
        if has_pool:
            replayed_manifest = replay_exact_allocated_route_manifest(
                candidate_pool=result.candidate_pool,
                resource_plan=result.resource_plan,
                materialization_certificate=(
                    result.routed.allocation_replay_certificate
                ),
                routing_world=result.routing_world_provenance,
                tech=result.tech,
            )
        else:
            replayed_manifest = route_owned_manifest_from_routed_geometry(
                result.routed
            )
        if replayed_manifest != self.artifact.route_manifest.entries:
            raise ArtifactDigestMismatch(
                "current allocated route/world replay differs from emitted manifest"
            )
        if (
            route_manifest_sha256(replayed_manifest)
            != self.artifact.route_manifest.route_manifest_sha256
        ):
            raise ArtifactDigestMismatch(
                "current allocated route/world replay has a stale manifest digest"
            )
        from clay.layout.route.via_provenance_check import (
            terminal_claims_from_pins,
            terminal_claims_sha256,
        )

        current_terminal_claims = terminal_claims_from_pins(
            result.pins_by_net, result.tech
        )
        reported_terminal_count = getattr(
            self.via_provenance_report, "terminal_claim_count", 0
        )
        reported_terminal_sha256 = str(
            getattr(self.via_provenance_report, "terminal_claims_sha256", "") or ""
        )
        if (
            current_terminal_claims
            or reported_terminal_count
            or reported_terminal_sha256
        ) and (
            reported_terminal_count != len(current_terminal_claims)
            or reported_terminal_sha256
            != terminal_claims_sha256(current_terminal_claims)
        ):
            raise ArtifactDigestMismatch(
                "current physical terminal claims differ from via provenance"
            )
        return True


__all__ = [
    "EmittedRouteOwnershipManifest",
    "ExactArtifactEvidenceBundle",
    "ExactEmittedArtifact",
    "RouteEmissionScopeCertificate",
    "route_owned_manifest_from_routed_geometry",
]
