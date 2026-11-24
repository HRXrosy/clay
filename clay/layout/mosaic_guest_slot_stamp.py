"""Join a source slot to current normalized host/guest artwork and owner frames.

This is a geometry projection, not a fresh-source or placement admission issuer.
No reference coordinates enter this module.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import SimpleNamespace

from clay.layout.mosaic_guest_slot_projection import (
    MosaicGuestSlotProjection, canonical_mosaic_guest_slot_projections,
)
from clay.layout.spatial.mosaic_guest_slot import (
    MosaicGuestSlotGeometryBinding, _polygon_records, _records_box,
    replay_mosaic_guest_slot_geometry, replay_mosaic_guest_slot_inventory,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _stamp_records(stamp, position_um):
    layers = getattr(stamp, "polys_by_layer", None)
    _require(type(layers) is dict and layers, "guest-slot current stamp has no polygon inventory")
    _require(len(position_um) == 2 and all(math.isfinite(v) for v in position_um),
             "guest-slot stamp origin is not finite")
    records = []
    for layer, rings in layers.items():
        _require(type(layer) is tuple and len(layer) == 2
                 and all(type(v) is int for v in layer), "guest-slot stamp layer identity differs")
        for ring in rings:
            _require(len(ring) >= 3 and all(len(point) == 2 and all(math.isfinite(v) for v in point)
                                         for point in ring), "guest-slot stamp polygon is malformed")
            points = tuple((int(round((x+position_um[0])*1000)), int(round((y+position_um[1])*1000)))
                           for x, y in ring)
            records.append((*layer, points))
    _records_box(tuple(records))
    return tuple(records)


@dataclass(frozen=True, slots=True)
class MosaicGuestSlotStampBinding:
    materialization: MosaicGuestSlotGeometryBinding
    native_origin_nm: tuple[int, int]

    def __post_init__(self):
        if type(self.materialization) is not MosaicGuestSlotGeometryBinding:
            raise TypeError("guest-slot stamp has a foreign materialization")
        self.materialization.__post_init__()
        _require(type(self.native_origin_nm) is tuple and len(self.native_origin_nm) == 2
                 and all(type(v) is int for v in self.native_origin_nm),
                 "guest-slot native stamp origin must be two exact integer nm")


def bind_mosaic_guest_slot_stamp(native_geometry, stamp, *, native_origin_nm):
    """Preserve the already verified native binding through the array bridge."""
    binding = getattr(native_geometry, "mosaic_guest_slot_binding", None)
    if binding is None:
        return
    replay_mosaic_guest_slot_geometry(native_geometry)
    tag = MosaicGuestSlotStampBinding(binding, native_origin_nm)
    actual = _stamp_records(stamp, tuple(v/1000 for v in native_origin_nm))
    _require(Counter(actual) == Counter(_polygon_records(native_geometry.polygons)),
             "guest-slot array bridge changed native artwork")
    stamp.mosaic_guest_slot_stamp_binding = tag


def _normalized_units(units, *, origin_nm, placement_um, tech):
    """Use the actual native normalizer, retaining each occurrence's polygons."""
    from clay.layout.route.route_cell import _normalize_materialized_geometry_to_manufacturing_grid
    geometry = {}
    for node, records in units.items():
        layers = {}
        for layer, datatype, points in records:
            layers.setdefault((layer, datatype), []).append(
                [((x-origin_nm[0])/1000, (y-origin_nm[1])/1000) for x, y in points])
        geometry[node] = SimpleNamespace(polys_by_layer=layers)
    positions = {node: placement_um for node in units}
    _normalize_materialized_geometry_to_manufacturing_grid(geometry, positions, tech)
    return {node: _stamp_records(stamp, placement_um) for node, stamp in geometry.items()}


def _owner_records(records, owner_box):
    return tuple(sorted((layer, datatype, tuple((x-owner_box[0], y-owner_box[1]) for x, y in points))
                        for layer, datatype, points in records))


def build_mosaic_guest_slot_projections(intent, *, solver_blocks, block_stamps,
        master_members, owner_logicals, geometry, placement_um, minimum_clearance_nm, tech):
    """Rebuild relative operands from complete current native polygon/frame joins."""
    slots = replay_mosaic_guest_slot_inventory(getattr(intent, "mosaic_guest_slots", ()))
    tagged = {name: getattr(stamp, "mosaic_guest_slot_stamp_binding", None)
              for name, stamp in geometry.items()
              if getattr(stamp, "mosaic_guest_slot_stamp_binding", None) is not None}
    if not slots:
        _require(not tagged, "undeclared guest-slot materialization stamp")
        return ()
    _require(type(minimum_clearance_nm) is int and minimum_clearance_nm >= 0,
             "guest-slot source gap must be exact nonnegative integer nm")
    blocks = dict(solver_blocks)
    _require(len(blocks) == len(solver_blocks) and set(blocks) == set(owner_logicals),
             "guest-slot live owner partition differs")
    for box in blocks.values():
        _require(type(box) is tuple and len(box) == 4 and all(type(v) is int for v in box)
                 and box[0] < box[2] and box[1] < box[3], "guest-slot solver owner box differs")

    def one_owner(members):
        candidates = [owner for owner, logicals in owner_logicals.items()
                      if tuple(sorted(logicals)) == tuple(sorted(members))]
        _require(len(candidates) == 1, "guest-slot needs one exact logical owner partition")
        return candidates[0]

    def one_stamp(owner):
        members = master_members.get(owner, (owner,))
        names = [name for member in members for name in block_stamps.get(member, ())]
        _require(len(names) == 1 and names[0] in geometry and names[0] in placement_um,
                 "guest-slot requires one complete current owner stamp")
        return names[0]

    rows, consumed = [], set()
    for slot in slots:
        candidates = [(name, tag) for name, tag in tagged.items()
                      if type(tag) is MosaicGuestSlotStampBinding and tag.materialization.slot == slot]
        _require(len(candidates) == 1, "guest-slot current source/materialization binding is missing or ambiguous")
        host_stamp, tag = candidates[0]
        tag.__post_init__()
        materialization = tag.materialization
        _require(materialization.source_gap_nm == minimum_clearance_nm,
                 "guest-slot materialization and current source gap differ")
        host_owner = one_owner(materialization.mosaic.members)
        _require(one_stamp(host_owner) == host_stamp, "guest-slot host stamp ownership differs")
        match = re.fullmatch(r"(.+)\[(\d{3,})\]", slot.guest_node)
        _require(match is not None and int(match.group(2)) == 0,
                 "guest-slot requires one exact singleton source occurrence")
        guest_owner = one_owner((match.group(1),))
        guest_stamp = one_stamp(guest_owner)
        _require(guest_owner != host_owner, "guest-slot host and guest ownership collapse")
        shifts = dict(materialization.plan.node_dx_nm)
        units = {node: tuple((layer, datatype, tuple((x+shifts[node], y) for x, y in points))
                            for layer, datatype, points in records)
                 for node, _x, _y, records in materialization.baseline_units}
        normalized = _normalized_units(units, origin_nm=tag.native_origin_nm,
            placement_um=placement_um[host_stamp], tech=tech)
        complete_host = tuple(record for records in normalized.values() for record in records)
        _require(Counter(complete_host) == Counter(_stamp_records(geometry[host_stamp], placement_um[host_stamp])),
                 "guest-slot current normalized host artwork differs from native units")
        guest_box = materialization.guest_extent_nm
        guest_origin = ((guest_box[0]+guest_box[2])/2, (guest_box[1]+guest_box[3])/2)
        normalized_guest = _normalized_units({slot.guest_node: materialization.guest_polygon_records},
            origin_nm=guest_origin, placement_um=placement_um[guest_stamp], tech=tech)[slot.guest_node]
        _require(Counter(normalized_guest) == Counter(_stamp_records(geometry[guest_stamp], placement_um[guest_stamp])),
                 "guest-slot current normalized guest artwork differs from native inventory")
        host_box, guest_owner_box = blocks[host_owner], blocks[guest_owner]
        owner_units = {node: _owner_records(records, host_box) for node, records in normalized.items()}
        owner_guest = _owner_records(normalized_guest, guest_owner_box)
        boundaries = tuple((boundary.left_node, _records_box(owner_units[boundary.left_node]),
                            boundary.right_node, _records_box(owner_units[boundary.right_node]))
                           for boundary in slot.boundaries)
        physical_identity = _hash({"native_materialization_sha256": materialization.sha256,
                                  "host_units_owner_nm": owner_units, "guest_owner_nm": owner_guest})
        rows.append(MosaicGuestSlotProjection(
            obligation_id=f"mosaic_guest_slot:{slot.sha256}", source_slot_sha256=slot.sha256,
            geometry_binding_sha256=physical_identity, host_master=host_owner, guest_master=guest_owner,
            host_size_nm=(host_box[2]-host_box[0], host_box[3]-host_box[1]),
            guest_size_nm=(guest_owner_box[2]-guest_owner_box[0], guest_owner_box[3]-guest_owner_box[1]),
            axis="horizontal", boundaries=boundaries, guest_envelope_nm=_records_box(owner_guest),
            minimum_clearance_nm=minimum_clearance_nm))
        consumed.add(host_stamp)
    _require(consumed == set(tagged), "guest-slot stamp source cover is incomplete")
    return canonical_mosaic_guest_slot_projections(tuple(sorted(rows, key=lambda row: row.obligation_id)),
        dims={owner: (box[2]-box[0], box[3]-box[1]) for owner, box in blocks.items()})
