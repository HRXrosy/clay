"""Immutable generated seed data and exact native-input joins.

This module does not establish input provenance or placement authority. A flow
that accepts external candidates must establish that authority before creating
this request. There are no reference geometry readers or solver calls here.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from threading import get_ident


SCHEMA = "clay.generated-placement-incumbent-seed/v1"
SCOPE = "CANDIDATE_SEED_ONLY_REQUIRES_FRESH_SOURCE_AND_NATIVE_REPLAY"
_STATE_FIELDS = frozenset({"max_time_s", "w_cap", "source_seed_positions", "certified_source_seed"})
_FRAME_JOIN = ContextVar("clay_generated_incumbent_frame_join", default=None)


def _execution_owner():
    import asyncio
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), get_ident(), task


def _current_frame_join():
    state = _FRAME_JOIN.get()
    if state is not None and (not state["active"] or state["owner"] != _execution_owner()):
        raise ValueError("generated incumbent frame scope is closed or belongs to another invocation")
    return state


@contextmanager
def scoped_generated_incumbent_frame(request, owner_logicals, *, phase="initial_master", adjacency_projection=None):
    """Bind one explicit native materialization phase to immutable seed data."""
    if type(request) is not GeneratedIncumbentSeed:
        raise TypeError("incumbent frame scope requires immutable native seed data")
    if phase not in {"initial_master", "exact_materialization_replay"}:
        raise ValueError("generated incumbent frame phase differs")
    if adjacency_projection is not None:
        if type(adjacency_projection) is not MaterializedAdjacencyScopeProjection:
            raise TypeError("incumbent requires an actual typed materialized adjacency projection")
        if phase != "exact_materialization_replay":
            raise ValueError("materialized adjacency projection requires exact phase")
    if _current_frame_join() is not None:
        raise ValueError("generated incumbent frame scopes cannot nest")
    owners = owner_partition(owner_logicals)
    request.require_owner_partition(owners)
    state = {"owner": _execution_owner(), "active": True, "consumed": False,
             "request": request, "owners_bytes": canonical_bytes(owners), "phase": phase,
             "adjacency_projection": adjacency_projection}
    token = _FRAME_JOIN.set(state)
    try:
        yield
        _current_frame_join()
        if not state["consumed"]:
            raise ValueError("generated incumbent frame scope completed without native consumption")
    finally:
        state["active"] = False
        _FRAME_JOIN.reset(token)


def current_generated_incumbent_phase():
    state = _current_frame_join()
    return None if state is None else state["phase"]


def current_generated_incumbent_adjacency_projection():
    state = _current_frame_join()
    projection = None if state is None else state["adjacency_projection"]
    return None if projection is None else projection.canonical_data()


def require_current_generated_incumbent_frame(frame):
    state = _current_frame_join()
    if state is None:
        return None
    request = state["request"]
    from clay.layout.place.solver_execution_policy import record_incumbent_frame_observation
    from clay.layout.place_blocks_2d import _diagnostic_safe_canonical_value
    captured = {key: _diagnostic_safe_canonical_value(value) for key, value in frame.items()}
    projection = current_generated_incumbent_adjacency_projection()
    observation = record_incumbent_frame_observation({
        "schema": "clay.generated-incumbent-frame-observation/v1", "phase": state["phase"],
        "request_sha256": request.sha256, "canonical_frame": captured,
        "canonical_frame_sha256": digest(captured), "adjacency_projection": projection,
        "adjacency_projection_sha256": digest(projection), "status": "NOT_EVALUATED", "error": None})
    try:
        expected = request.positions_for_frame(frame, json.loads(state["owners_bytes"]),
            phase=state["phase"], adjacency_projection=state["adjacency_projection"])
        actual = _positions(frame["source_seed_positions"], expected)
        if actual != expected or frame["certified_source_seed"] is not True:
            raise ValueError("generated incumbent actual source seed differs before native execution")
    except BaseException as error:
        if observation is not None:
            observation.update(status="REJECTED_FRAME_JOIN", error={"type": type(error).__name__, "message": str(error)})
        raise
    if observation is not None:
        observation["status"] = "ACCEPTED_FRAME_JOIN"
    state["consumed"] = True
    return dict(expected)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def owner_partition(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError("incumbent owner partition must be a nonempty mapping")
    rows, seen = {}, set()
    for owner, members in sorted(value.items()):
        if (type(owner) is not str or not owner or not isinstance(members, (list, tuple))
                or not members or any(type(name) is not str or not name for name in members)
                or len(set(members)) != len(members) or seen.intersection(members)):
            raise ValueError("incumbent owner logical partition differs")
        rows[owner] = tuple(sorted(members))
        seen.update(members)
    return rows


def _pairs(value, label):
    if (not isinstance(value, (list, tuple)) or any(not isinstance(row, (list, tuple))
            or len(row) != 2 for row in value)):
        raise ValueError("incumbent " + label + " pair inventory differs")
    result = dict(value)
    if len(result) != len(value):
        raise ValueError("incumbent " + label + " repeats an owner")
    return result


def _positions(value, owners):
    from clay.layout.place_blocks_2d import _canonical_source_seed_positions
    return _canonical_source_seed_positions(value, ids=tuple(sorted(owners)))


@dataclass(frozen=True, slots=True, init=False)
class MaterializedAdjacencyScopeProjection:
    """Replayable native geometry data; construction grants no G4 authority.

The actual materializer supplies this object to the exact frame scope. JSON
reconstruction is used only to check already-bound native stage evidence.
"""
    _payload_bytes: bytes
    _layers_bytes: bytes
    _owners_bytes: bytes

    def __init__(self, *, scope_offsets_nm, provenance):
        from types import SimpleNamespace
        from clay.layout.placed_geometry import placed_point_nm
        from clay.layout.place_blocks_2d import (
            _diagnostic_canonical_value as canonical, _layered_collision_footprints_from_geometry,
        )
        fields = {"schema", "producer", "master_blocks", "master_members", "materialized_stamps",
            "materialized_rings", "source_members", "rigid_nonmos_macro_plans", "geometry_and_placement",
            "placed_geometry_sha256", "tech_layer_name", "lookup_status", "excluded_layers",
            "absolute_scope_bboxes_nm", "materialized_blocks", "component_refinement_owners"}
        if (type(provenance) is not dict or set(provenance) != fields
                or provenance["schema"] != "clay.materialized-adjacency-scope-provenance/v1"
                or provenance["producer"] != "clay.layout.place_blocks_2d._materialized_adjacency_scope_bboxes_nm"
                or provenance["tech_layer_name"] != "nwell" or provenance["lookup_status"] != "RESOLVED"
                or digest(provenance["geometry_and_placement"]) != provenance["placed_geometry_sha256"]):
            raise ValueError("materialized adjacency projection provenance differs")
        masters = _pairs(provenance["master_blocks"], "projection masters")
        members = _pairs(provenance["master_members"], "projection members")
        stamps = _pairs(provenance["materialized_stamps"], "projection stamps")
        rings = _pairs(provenance["materialized_rings"], "projection rings")
        geometry_rows = _pairs(provenance["geometry_and_placement"], "projection geometry")
        source = tuple(provenance["source_members"])
        excluded = tuple(tuple(layer) for layer in provenance["excluded_layers"])
        if (not masters or set(members) != set(masters) or tuple(sorted(set(source))) != source
                or len(excluded) != 1 or len(excluded[0]) != 2 or any(type(n) is not int for n in excluded[0])):
            raise ValueError("materialized adjacency ownership or PDK layer denominator differs")
        member_ids = [member for group in members.values() for member in group]
        if not member_ids or len(member_ids) != len(set(member_ids)) or not set(member_ids) <= set(stamps):
            raise ValueError("materialized adjacency member ownership differs")
        consumed = {stamp for member in member_ids for stamp in stamps[member]}
        consumed.update(rings[member] for member in member_ids if member in rings)
        if set(geometry_rows) != consumed:
            raise ValueError("materialized adjacency consumed geometry denominator differs")
        geometry, placement, structural_boxes = {}, {}, {}
        for name, raw in geometry_rows.items():
            data = _pairs(raw, "projection stamp fields")
            if set(data) != {"polys_by_layer", "placement_um", "composite_members"}:
                raise ValueError("materialized adjacency stamp fields differ")
            point = tuple(data["placement_um"])
            if len(point) != 2 or any(type(n) not in (int, float) for n in point):
                raise ValueError("materialized adjacency placement differs")
            layers = {}
            for key, polygons in data["polys_by_layer"]:
                layer = tuple(key)
                if len(layer) != 2 or any(type(n) is not int for n in layer) or layer in layers:
                    raise ValueError("materialized adjacency geometry layer differs")
                layers[layer] = [tuple(tuple(vertex) for vertex in polygon) for polygon in polygons]
            geometry[name] = SimpleNamespace(polys_by_layer=layers, composite_members=tuple(data["composite_members"]))
            placement[name] = point
        def envelope(boxes):
            return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes))
        block_scopes = {}
        for member in member_ids:
            names = set(stamps[member]) | ({rings[member]} if member in rings else set())
            points = [placed_point_nm(float(x), float(y), placement[name])
                for name in sorted(names) for layer, polygons in geometry[name].polys_by_layer.items()
                if layer not in excluded for polygon in polygons for x, y in polygon]
            if not points:
                raise ValueError("materialized adjacency structural geometry is empty")
            block_scopes[member] = (min(x for x, _ in points), min(y for _, y in points),
                                    max(x for x, _ in points), max(y for _, y in points))
        offsets, projected_owners = {}, {}
        for owner, group in members.items():
            master = tuple(masters[owner])
            box = envelope([block_scopes[member] for member in group])
            if (len(master) != 4 or any(type(n) is not int for n in master)
                    or not (master[0] <= box[0] < box[2] <= master[2] and master[1] <= box[1] < box[3] <= master[3])):
                raise ValueError("materialized adjacency scope escapes its master")
            structural_boxes[owner] = box
            offsets[owner] = (box[0]-master[0], box[1]-master[1], box[2]-master[0], box[3]-master[1])
            projected_owners[owner] = tuple(sorted({logical for member in group for name in stamps[member]
                for logical in (name, *geometry[name].composite_members) if logical in set(source)}))
        if (canonical_bytes(canonical(structural_boxes)) != canonical_bytes(provenance["absolute_scope_bboxes_nm"])
                or canonical_bytes(canonical(offsets)) != canonical_bytes(canonical(scope_offsets_nm))):
            raise ValueError("materialized adjacency actual polygon projection differs")
        # Reuse the real native abstraction, including per-component refinement
        # and exact ring rectangles. Equal bounding boxes alone are insufficient.
        layers = _layered_collision_footprints_from_geometry(
            solver_blocks=tuple((name, tuple(box)) for name, box in masters.items()),
            materialized_blocks=tuple((name, tuple(box)) for name, box in provenance["materialized_blocks"]),
            block_stamps={name: tuple(group) for name, group in stamps.items()}, block_gring=rings,
            geometry=geometry, placement_um=placement,
            master_members={name: tuple(group) for name, group in members.items()},
            component_refinement_owners=tuple(provenance["component_refinement_owners"]))
        if set(dict(layers)) != set(masters):
            raise ValueError("materialized adjacency layered geometry coverage differs")
        payload = {"schema": "clay.materialized-adjacency-scope-projection/v1",
            "scope_offsets_nm": canonical(offsets), "provenance": provenance}
        object.__setattr__(self, "_payload_bytes", canonical_bytes(payload))
        object.__setattr__(self, "_layers_bytes", canonical_bytes(canonical(layers)))
        object.__setattr__(self, "_owners_bytes", canonical_bytes(owner_partition(projected_owners)))

    @classmethod
    def from_data(cls, data):
        if (type(data) is not dict or set(data) != {"schema", "scope_offsets_nm", "provenance"}
                or data["schema"] != "clay.materialized-adjacency-scope-projection/v1"):
            raise ValueError("materialized adjacency projection schema differs")
        return cls(scope_offsets_nm=_pairs(data["scope_offsets_nm"], "projection scopes"), provenance=data["provenance"])

    def canonical_data(self):
        return json.loads(self._payload_bytes)

    @property
    def sha256(self):
        return hashlib.sha256(self._payload_bytes).hexdigest()

    def expected_frame(self, request, *, phase):
        if phase != "exact_materialization_replay":
            raise ValueError("materialized adjacency projection requires exact phase")
        expected = request.canonical_data()["native_frame"]
        if canonical_bytes(expected["layered_collision_footprints"]) != self._layers_bytes:
            raise ValueError("materialized adjacency full layered artwork identity differs")
        if canonical_bytes(request.owner_logicals) != self._owners_bytes:
            raise ValueError("materialized adjacency source owner partition differs")
        data = self.canonical_data()
        old, new = _pairs(expected["adjacency_scope_offsets_nm"], "initial scopes"), _pairs(data["scope_offsets_nm"], "materialized scopes")
        if set(new) != set(old) or set(new) != set(request.owner_logicals):
            raise ValueError("materialized adjacency scope owner inventory differs")
        active = {name for left, right, _limit in expected["adjacency_limits_nm"] for name in (left, right)}
        if any(canonical_bytes(new[name]) != canonical_bytes(old[name]) for name in active):
            raise ValueError("materialized adjacency active endpoint scope differs")
        expected["adjacency_scope_offsets_nm"] = data["scope_offsets_nm"]
        return expected


def frame_data(frame):
    from clay.layout.place_blocks_2d import _native_stage_frame_fields, _diagnostic_canonical_value
    if not isinstance(frame, Mapping) or set(frame) != _native_stage_frame_fields():
        raise ValueError("complete incumbent native frame fields differ")
    return {name: _diagnostic_canonical_value(value) for name, value in frame.items()}


def frame_contract_digest(frame):
    data = frame_data(frame)
    state_fields = _STATE_FIELDS - {"w_cap"} if data["strict_width_cap"] or data["exact_outline_width"] else _STATE_FIELDS
    return digest({key: value for key, value in data.items() if key not in state_fields})


def _checked_payload(payload):
    from clay.layout.place_blocks_2d import _native_stage_frame_fields
    fields = {"schema", "scope", "promotion", "g4_status", "positions_nm", "owner_logicals",
              "native_frame", "production_evidence"}
    if (type(payload) is not dict or set(payload) != fields or payload["schema"] != SCHEMA
            or payload["scope"] != SCOPE or payload["promotion"] != "REJECT"
            or payload["g4_status"] != "NOT_RUN"):
        raise ValueError("incumbent seed payload schema or scope differs")
    owners = owner_partition(_pairs(payload["owner_logicals"], "owner partition"))
    positions = _positions(_pairs(payload["positions_nm"], "positions"), owners)
    frame = payload["native_frame"]
    if type(frame) is not dict or set(frame) != _native_stage_frame_fields():
        raise ValueError("incumbent stored native frame fields differ")
    dims = _pairs(frame["dims"], "dimensions")
    if (set(dims) != set(owners) or set(frame["ids"]) != set(owners)
            or len(frame["ids"]) != len(owners)
            or any(not isinstance(size, (tuple, list)) or len(size) != 2
                   or any(type(n) is not int or n <= 0 for n in size) for size in dims.values())):
        raise ValueError("incumbent owner dimensions differ")
    evidence = payload["production_evidence"]
    if (type(evidence) is not dict or set(evidence) != {
            "model_proto_sha256", "promotion_gate_sha256", "outer_replay_sha256"}
            or any(type(value) is not str or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value) for value in evidence.values())):
        raise ValueError("incumbent production evidence identities differ")
    return positions, owners


def validate_native_incumbent_join(*, payload, current_frame, owner_logicals, phase="initial_master", adjacency_projection=None):
    """Return copied XY only when the complete current native contract matches."""
    positions, owners = _checked_payload(payload)
    if owner_partition(owner_logicals) != owners:
        raise ValueError("incumbent logical owner partition identity differs")
    current = frame_data(current_frame)
    expected = payload["native_frame"]
    if adjacency_projection is not None:
        if type(adjacency_projection) is not MaterializedAdjacencyScopeProjection:
            raise TypeError("incumbent materialized adjacency projection type differs")
        expected = adjacency_projection.expected_frame(GeneratedIncumbentSeed(payload), phase=phase)
    state_fields = _STATE_FIELDS
    if expected["strict_width_cap"] or expected["exact_outline_width"]:
        state_fields = state_fields - {"w_cap"}
    differing = sorted(key for key in current if key not in state_fields
                       and canonical_bytes(current[key]) != canonical_bytes(expected[key]))
    if differing:
        raise ValueError("incumbent native frame contract identity differs: " + ", ".join(differing))
    return dict(positions)


def require_retained_generated_incumbent(*, payload, current_frame, owner_logicals, final_positions):
    expected = validate_native_incumbent_join(payload=payload, current_frame=current_frame,
                                              owner_logicals=owner_logicals)
    actual = _positions(final_positions, expected)
    if actual != expected:
        raise ValueError("retained incumbent candidate positions differ")


@dataclass(frozen=True, slots=True, init=False)
class GeneratedIncumbentSeed:
    """Closed immutable native data. Construction does not grant provenance."""
    _payload_bytes: bytes

    def __init__(self, payload):
        _checked_payload(payload)
        object.__setattr__(self, "_payload_bytes", canonical_bytes(payload))

    def canonical_data(self):
        return json.loads(self._payload_bytes)

    @property
    def sha256(self):
        return hashlib.sha256(self._payload_bytes).hexdigest()

    @property
    def positions(self):
        return _checked_payload(self.canonical_data())[0]

    @property
    def owner_logicals(self):
        return _checked_payload(self.canonical_data())[1]

    def require_owner_partition(self, owner_logicals):
        if owner_partition(owner_logicals) != self.owner_logicals:
            raise ValueError("incumbent current owner partition differs")

    def positions_for_frame(self, frame, owner_logicals, *, phase="initial_master", adjacency_projection=None):
        return validate_native_incumbent_join(payload=self.canonical_data(), current_frame=frame,
                                              owner_logicals=owner_logicals, phase=phase, adjacency_projection=adjacency_projection)

    def require_recorded_frame(self, data, *, phase="initial_master", adjacency_projection=None):
        """Check the same contract after the native recorder canonicalizes it."""
        from clay.layout.place_blocks_2d import _native_stage_frame_fields, _diagnostic_canonical_value
        expected = self.canonical_data()["native_frame"]
        if adjacency_projection is not None:
            if type(adjacency_projection) is not MaterializedAdjacencyScopeProjection:
                raise TypeError("recorded materialized adjacency projection type differs")
            expected = adjacency_projection.expected_frame(self, phase=phase)
        if type(data) is not dict or set(data) != _native_stage_frame_fields():
            raise ValueError("recorded incumbent native frame fields differ")
        omitted = _STATE_FIELDS - {"w_cap"} if expected["strict_width_cap"] or expected["exact_outline_width"] else _STATE_FIELDS
        if (canonical_bytes({k: v for k, v in data.items() if k not in omitted}) !=
                canonical_bytes({k: v for k, v in expected.items() if k not in omitted})
                or canonical_bytes(data["source_seed_positions"]) != canonical_bytes(_diagnostic_canonical_value(self.positions))
                or data["certified_source_seed"] is not True):
            raise ValueError("recorded incumbent frame or actual source seed identity differs")

    def require_final_positions(self, final_positions):
        if _positions(final_positions, self.owner_logicals) != self.positions:
            raise ValueError("retained incumbent final candidate positions differ")
