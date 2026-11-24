"""Observe final logical bboxes and native occurrences without granting quality.

The caller supplies the actual final materialization in its native placed frame.
This snapshot binds its complete logical denominator to the exact certificate
state; it does not measure polygon equivalence or grant acceptance/signoff.
Implementation, source and PDK identities belong to the calling producer.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

from clay.layout.place.final_placement_state import (
    final_placement_state_digest, final_placement_state_payload,
)
from clay.layout.place.placement_certificates import FinalPlacementCertificate
from clay.layout.place.placement_geometry import _source_device_bboxes_nm
from clay.layout.place.reference_quality_geometry import collect_live_source_occurrences

_KINDS = {"nmos", "pmos", "capacitor", "cap", "mimcap", "resistor", "res", "pnp", "npn"}
_ORIENTATIONS = {"R0", "R90", "R180", "R270", "MR0", "MR90", "MR180", "MR270"}
_STATE_SCHEMAS = {f"clay.final-placement-state.v{version}"
                  for version in (2, 3, 4, 5, 6, 7, 9, *range(12, 36))}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _basis(kind):
    return ("materialized_resistor_bank_center" if kind in {"res", "resistor"}
            else "materialized_unit_metadata")


def _bbox(value, *, sequence_type=tuple):
    return (type(value) is sequence_type and len(value) == 4
            and all(type(coordinate) is int for coordinate in value)
            and value[0] < value[2] and value[1] < value[3])


def _inside(inner, outer):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def _state_boxes(raw, label):
    _require(type(raw) is list and raw, f"final state {label} bbox inventory is required")
    _require(all(type(row) is list and len(row) == 2 and type(row[0]) is str and row[0]
                 and row[0] == row[0].strip() and _bbox(row[1], sequence_type=list) for row in raw),
             f"final state {label} bbox rows differ")
    names = [row[0] for row in raw]
    _require(names == sorted(set(names)), f"final state {label} bbox inventory is not canonical")
    return dict(raw)


def _state_core(state):
    """Validate the stable core and the physical-to-master coordinate join."""
    blocks = _state_boxes(state.get("blocks"), "master")
    from clay.layout.place.physical_master_frames import physical_master_boxes, projections_from_payload
    schema = state.get('schema')
    flexible = bool(state.get('flexible_envelope_projections'))
    _require((flexible if schema == 'clay.final-placement-state.v35'
              else True if schema == 'clay.final-placement-state.v36'
              else not flexible), 'final state flexible physical authority schema differs')
    if schema == 'clay.final-placement-state.v36':
        digest = state.get('native_cross_row_final_proof_sha256')
        _require(type(digest) is str and len(digest) == 64
                 and all(char in '0123456789abcdef' for char in digest),
                 'final state native cross-row proof digest differs')
    physical_masters = physical_master_boxes(
        blocks, projections_from_payload(state.get("flexible_envelope_projections", [])))
    frame = state.get("bbox_nm")
    _require(_bbox(frame, sequence_type=list) and all(_inside(box, frame) for box in blocks.values()),
             "final state master bbox escapes its core frame")
    _require(type(state.get("declared_channel_nm")) is int,
             "final state declared channel must be an exact integer")
    bands = state.get("bands")
    _require(type(bands) is list and all(type(row) is list and len(row) == 2
                 and type(row[0]) is str and row[0] in blocks and type(row[1]) is int for row in bands),
             "final state band inventory differs from master blocks")
    _require(bands == sorted(bands) and len({row[0] for row in bands}) == len(bands),
             "final state bands must be canonical and unique")
    physical = _state_boxes(state.get("materialized_block_bboxes"), "physical")
    members = state.get("materialized_block_members")
    _require(type(members) is list and members, "final state materialized block ownership is required")
    masters, owner_of = [], {}
    for entry in members:
        _require(type(entry) is list and len(entry) == 2 and type(entry[0]) is str
                 and entry[0] in blocks and type(entry[1]) is list and entry[1]
                 and all(type(name) is str and name in physical for name in entry[1]),
                 "final state materialized block ownership differs")
        _require(entry[1] == sorted(set(entry[1])), "final state physical block members are not canonical")
        masters.append(entry[0])
        for name in entry[1]:
            _require(name not in owner_of and _inside(physical[name], physical_masters[entry[0]]),
                     "final state physical block has repeated owner or escapes master bbox")
            owner_of[name] = entry[0]
    _require(masters == sorted(blocks) and set(owner_of) == set(physical),
             "final state physical ownership must completely partition master blocks")
    return physical


@dataclass(frozen=True, slots=True)
class FinalLogicalGeometry:
    """Native occurrence frame and certified physical source envelope.

    A shared stamp can give several logical members the same envelope. The
    separately observed owned-artwork inventory defines their quality bboxes.
    """
    logical_id: str
    kind: str
    bbox_nm: tuple[int, int, int, int]
    occurrence_basis: str
    occurrences: tuple[tuple[int, int, str], ...]

    def __post_init__(self):
        _require(type(self.logical_id) is str and self.logical_id
                 and self.logical_id == self.logical_id.strip(), "logical id differs")
        _require(type(self.kind) is str and self.kind in _KINDS, "logical kind differs")
        _require(_bbox(self.bbox_nm), "logical bbox must be an immutable exact integer tuple")
        allowed_bases = {_basis(self.kind)}
        if self.kind in {'res', 'resistor'}:
            allowed_bases.add('materialized_resistor_body_center')
        _require(type(self.occurrence_basis) is str and self.occurrence_basis in allowed_bases,
                 "logical occurrence basis differs from kind")
        _require(type(self.occurrences) is tuple and self.occurrences,
                 "logical occurrence inventory must be an immutable nonempty tuple")
        for row in self.occurrences:
            _require(type(row) is tuple and len(row) == 3
                     and type(row[0]) is int and type(row[1]) is int
                     and type(row[2]) is str and row[2] in _ORIENTATIONS,
                     "logical occurrence point/orientation differs")
            _require(self.bbox_nm[0] <= row[0] <= self.bbox_nm[2]
                     and self.bbox_nm[1] <= row[1] <= self.bbox_nm[3],
                     "logical occurrence escapes source bbox")
        _require(self.occurrences == tuple(sorted(self.occurrences)),
                 "logical occurrence inventory must be canonical")
        if self.occurrence_basis == 'materialized_resistor_body_center':
            _require(len(self.occurrences) == 1, 'resistor body basis requires one actual primitive')
        elif self.kind in {"res", "resistor"}:
            _require(len(self.occurrences) == 1 and self.occurrences[0][:2] == (
                (self.bbox_nm[0] + self.bbox_nm[2]) // 2,
                (self.bbox_nm[1] + self.bbox_nm[3]) // 2), "resistor occurrence basis requires one bank center")

    def canonical_data(self):
        return {"logical_id": self.logical_id, "kind": self.kind,
                "bbox_nm": list(self.bbox_nm), "occurrence_basis": self.occurrence_basis,
                "occurrence_count": len(self.occurrences),
                "occurrences": [list(row) for row in self.occurrences]}


@dataclass(frozen=True, slots=True)
class FinalQualityGeometryObservation:
    _final_state_json: str
    final_placement_state_sha256: str
    logical_geometry: tuple[FinalLogicalGeometry, ...]
    source_artwork_bboxes_nm: tuple[tuple[str, tuple[int, int, int, int]], ...] = ()

    def __post_init__(self):
        """Validate record consistency; construction does not prove observation."""
        _require(type(self._final_state_json) is str, "final state must be canonical JSON text")
        try:
            state = json.loads(self._final_state_json)
        except (ValueError, TypeError) as exc:
            raise ValueError("final state must be valid JSON") from exc
        _require(type(state) is dict and type(state.get("schema")) is str
                 and state["schema"] in _STATE_SCHEMAS,
                 "final state schema differs")
        _require(_json(state) == self._final_state_json, "final state JSON must be canonical")
        _require(type(self.final_placement_state_sha256) is str
                 and hashlib.sha256(self._final_state_json.encode()).hexdigest() == self.final_placement_state_sha256,
                 "final state digest differs from payload")
        physical = _state_core(state)
        _require(type(self.logical_geometry) is tuple and self.logical_geometry
                 and all(type(row) is FinalLogicalGeometry for row in self.logical_geometry),
                 "logical geometry inventory must be immutable exact records")
        for row in self.logical_geometry:
            row.__post_init__()
        names = tuple(row.logical_id for row in self.logical_geometry)
        _require(names == tuple(sorted(set(names))), "logical geometry inventory must be canonical and unique")
        owned = state.get("materialized_block_source_members")
        _require(type(owned) is list and owned, "final state requires complete source ownership")
        owners, sources = [], []
        for entry in owned:
            _require(type(entry) is list and len(entry) == 2 and type(entry[0]) is str
                     and entry[0] and type(entry[1]) is list and entry[1]
                     and all(type(source) is str and source for source in entry[1]),
                     "final state source ownership differs")
            _require(entry[1] == sorted(set(entry[1])), "final state source inventory is not canonical")
            owners.append(entry[0])
            sources.extend(entry[1])
        _require(owners == sorted(set(owners)) and tuple(sorted(sources)) == names,
                 "final state source ownership differs from complete logical inventory")
        _require(set(owners) == set(physical), "final state source ownership differs from physical blocks")
        boxes = state.get("materialized_source_member_bboxes")
        _require(type(boxes) is list and all(type(entry) is list and len(entry) == 2
                 and type(entry[0]) is str and _bbox(entry[1], sequence_type=list) for entry in boxes),
                 "final state source bbox inventory differs")
        _require(boxes == [[row.logical_id, list(row.bbox_nm)] for row in self.logical_geometry],
                 "final state bbox inventory differs from complete logical geometry")
        source_owner = {source: owner for owner, sources in owned for source in sources}
        _require(all(_inside(box, physical[source_owner[name]]) for name, box in boxes),
                 "final state logical bbox escapes its physical owner")
        artwork = self.source_artwork_bboxes_nm
        _require(type(artwork) is tuple, 'owned artwork inventory must be an immutable exact tuple')
        if artwork:
            _require(all(type(row) is tuple and len(row) == 2 and type(row[0]) is str
                         and _bbox(row[1]) for row in artwork), 'owned artwork bbox records differ')
            _require(tuple(name for name, _box in artwork) == names,
                     'owned artwork complete logical inventory differs')
            envelopes = {row.logical_id: row.bbox_nm for row in self.logical_geometry}
            _require(all(_inside(box, envelopes[name]) for name, box in artwork),
                     'owned artwork bbox escapes its certified physical source envelope')

    def canonical_data(self):
        data = {"schema": "clay.final-quality-geometry-observation/v1",
                "geometry_substrate": "logical_bboxes_and_materialized_occurrences",
                "final_placement_state": json.loads(self._final_state_json),
                "final_placement_state_sha256": self.final_placement_state_sha256,
                "logical_count": len(self.logical_geometry),
                "occurrence_count": sum(len(row.occurrences) for row in self.logical_geometry),
                "logical_geometry": [row.canonical_data() for row in self.logical_geometry]}
        if self.source_artwork_bboxes_nm:
            data['schema'] = 'clay.final-quality-geometry-observation/v2'
            data['geometry_substrate'] = 'physical_source_envelopes_owned_artwork_and_materialized_occurrences'
            data['source_artwork_bboxes_nm'] = [[name, list(box)]
                for name, box in self.source_artwork_bboxes_nm]
        return data

    @property
    def sha256(self):
        return hashlib.sha256(_json(self.canonical_data()).encode()).hexdigest()


def _kinds(logical_kinds):
    _require(type(logical_kinds) is dict and logical_kinds,
             "complete logical kind inventory is required")
    _require(all(type(name) is str and name and name == name.strip()
                 and type(kind) is str and kind in _KINDS
                 for name, kind in logical_kinds.items()), "logical kind inventory differs")
    return tuple(sorted(logical_kinds.items()))


def _observe_rows(*, geometry, placement_um, kinds, signed_boxes):
    from clay.layout.owned_artwork import quality_source_artwork_bboxes_nm
    names = tuple(name for name, _kind in kinds)
    boxes = _source_device_bboxes_nm(geometry, placement_um, names)
    _require(tuple(sorted(boxes.items())) == signed_boxes,
             "current source bbox geometry differs from complete final certificate")
    occurrences = collect_live_source_occurrences(
        geometry=geometry, placement_um=placement_um, logical_kinds=dict(kinds),
        source_device_bboxes=boxes)
    _require(set(occurrences) == set(names), "complete current occurrence inventory differs")
    body_centers = {owner for stamp in geometry.values()
        if getattr(stamp, 'source_resistor_stamp_binding', None) is not None
        for owner in stamp.composite_members}
    rows = tuple(FinalLogicalGeometry(
        logical_id=name, kind=kind, bbox_nm=boxes[name],
        occurrence_basis=('materialized_resistor_body_center' if name in body_centers else _basis(kind)),
        occurrences=occurrences[name]) for name, kind in kinds)
    artwork = quality_source_artwork_bboxes_nm(
        geometry=geometry, placement_um=placement_um, source_devices=names)
    return rows, tuple(sorted(artwork.items()))


def source_logical_kinds(cell):
    """Resolve one complete parsed primitive inventory for geometry consumers."""
    from clay.layout.annotate_to_solver import kind_polarity_oxide

    kinds = {}
    for device in cell.devices:
        name, declared_kind = device.instance_name, device.device_class
        if type(name) is not str or not name or name in kinds:
            raise ValueError("parsed final device inventory requires unique nonempty names")
        if type(declared_kind) is not str or not declared_kind:
            raise ValueError("parsed final device inventory requires declared kinds")
        kinds[name] = kind_polarity_oxide(declared_kind)[0] or declared_kind
    _kinds(kinds)
    return kinds


def observe_materialized_source(*, cell, compact):
    """Observe a parsed source through its actual final materialization result."""
    kinds = source_logical_kinds(cell)
    placement_um, geometry = compact.ring_inclusive
    return observe_final_quality_geometry(certificate=compact.final_placement_certificate,
        geometry=geometry, placement_um=placement_um, logical_kinds=kinds)


def observe_final_quality_geometry(*, certificate, geometry, placement_um, logical_kinds):
    """Capture a complete, immutable observation from the supplied final frame.

    Use the final materializer's ring-inclusive geometry/placement pair. Filtering
    nonlogical guard stamps is harmless only when every logical source retains
    the same certified bbox and native occurrence metadata; no frame is guessed.
    Resistor rows denote whole-bank centers, not a segment-count measurement.
    """
    if type(certificate) is not FinalPlacementCertificate:
        raise TypeError("final geometry observation requires an exact FinalPlacementCertificate")
    _require(isinstance(geometry, Mapping) and isinstance(placement_um, Mapping),
             "current geometry and placement inventory must be mappings")
    kinds = _kinds(logical_kinds)
    state_json = _json(final_placement_state_payload(certificate))
    state_digest = final_placement_state_digest(certificate)
    _require(hashlib.sha256(state_json.encode()).hexdigest() == state_digest,
             "final certificate state changed during observation")
    owned = tuple(sorted(source for _owner, sources in certificate.materialized_block_source_members
                         for source in sources))
    names = tuple(name for name, _kind in kinds)
    _require(owned == names, "complete final source ownership differs from logical inventory")
    signed_boxes = certificate.materialized_source_member_bboxes
    _require(tuple(name for name, _box in signed_boxes) == names,
             "complete final source bbox inventory is required")
    first = _observe_rows(geometry=geometry, placement_um=placement_um, kinds=kinds,
                         signed_boxes=signed_boxes)
    repeated = _observe_rows(geometry=geometry, placement_um=placement_um, kinds=kinds,
                             signed_boxes=signed_boxes)
    _require(first == repeated, "current occurrence or owned artwork geometry changed during observation")
    _require(_kinds(logical_kinds) == kinds, "logical kind inventory changed during observation")
    _require(_json(final_placement_state_payload(certificate)) == state_json,
             "final certificate state changed during observation")
    rows, artwork = first
    return FinalQualityGeometryObservation(state_json, state_digest, rows, artwork)
