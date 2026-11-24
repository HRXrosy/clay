"""Opt-in execution limits for the main banded area search.

The default context leaves all native parameters untouched. A scoped wall-clock
cap can make timeout and incumbent selection machine-dependent; it does not
change the generic default deterministic search or any model constraints.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Callable, Iterator
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json


@dataclass(frozen=True, slots=True)
class AreaSearchExecutionPolicy:
    stop_after_first_solution: bool
    max_wall_time_s: float
    num_search_workers: int | None = None
    max_deterministic_time: float | None = None
    random_seed: int | None = None
    canonical_tie_break: bool = False

    def __post_init__(self) -> None:
        if type(self.stop_after_first_solution) is not bool:
            raise TypeError("stop_after_first_solution must be an exact bool")
        if type(self.max_wall_time_s) not in (int, float):
            raise TypeError("max_wall_time_s must be a real number")
        if not math.isfinite(self.max_wall_time_s) or self.max_wall_time_s <= 0:
            raise ValueError("max_wall_time_s must be finite and positive")
        if self.num_search_workers is not None:
            if type(self.num_search_workers) is not int or self.num_search_workers <= 0:
                raise ValueError("num_search_workers must be a positive exact integer")
        if self.max_deterministic_time is not None:
            if type(self.max_deterministic_time) not in (int, float):
                raise TypeError("max_deterministic_time must be a real number")
            if not math.isfinite(self.max_deterministic_time) or self.max_deterministic_time <= 0:
                raise ValueError("max_deterministic_time must be finite and positive")
        if self.random_seed is not None:
            if type(self.random_seed) is not int or self.random_seed < 0:
                raise ValueError("random_seed must be a non-negative exact integer")
        if type(self.canonical_tie_break) is not bool:
            raise TypeError("canonical_tie_break must be an exact bool")


_AREA_SEARCH_POLICY: ContextVar[AreaSearchExecutionPolicy | None] = ContextVar(
    "clay_area_search_execution_policy", default=None,
)

class _IncumbentReplaySession:
    __slots__ = ("_completed", "_observed", "_indexed_geometry")

    def __init__(self, *, indexed_geometry=False):
        if type(indexed_geometry) is not bool:
            raise TypeError("indexed_geometry must be an exact bool")
        self._indexed_geometry = indexed_geometry
        self._completed = []
        self._observed = []

    @property
    def completed_records(self):
        return deepcopy(tuple(self._completed))

    @property
    def observed_frames(self):
        return deepcopy(tuple(self._observed))


_INCUMBENT_REPLAY: ContextVar[_IncumbentReplaySession | None] = ContextVar(
    "clay_incumbent_replay", default=None,
)


@contextmanager
def scoped_incumbent_replay(*, indexed_geometry=False) -> Iterator[_IncumbentReplaySession]:
    """Replay the supplied seed, optionally using a fixed geometry proof.

    The opt-in changes only geometric replay storage, retaining the complete
    original CP and independent native hard checks. Default evidence is unchanged.
    """
    session = _IncumbentReplaySession(indexed_geometry=indexed_geometry)
    token = _INCUMBENT_REPLAY.set(session)
    try:
        yield session
    finally:
        _INCUMBENT_REPLAY.reset(token)


def incumbent_replay_enabled() -> bool:
    return _INCUMBENT_REPLAY.get() is not None


def indexed_incumbent_geometry_enabled() -> bool:
    session = _INCUMBENT_REPLAY.get()
    return session is not None and session._indexed_geometry


def record_incumbent_frame_observation(observation):
    """Snapshot a native frame before its join; this is diagnostic data only.

    The generic join owns the returned private row and updates only its result
    status/error. Readers receive copies through session.observed_frames.
    """
    session = _INCUMBENT_REPLAY.get()
    if session is None:
        return None
    if not isinstance(observation, Mapping):
        raise TypeError("incumbent frame observation must be a mapping")
    row = deepcopy(dict(observation))
    session._observed.append(row)
    return row


def _complete_incumbent_replay_stage(*, phase, frame, positions, graph,
                                     model, solver, status, primitive_checks,
                                     full_hard_audit, cohesion_caps, original_model_bytes):
    """Called only after the native stage's full outer/caps/graph success."""
    session = _INCUMBENT_REPLAY.get()
    if session is None:
        raise ValueError("incumbent replay completion requires its active scope")
    if (phase not in {"initial_master", "exact_materialization_replay", "standalone_banded"}
            or not primitive_checks or any(row["status"] != "PASS" for row in primitive_checks)
            or set(full_hard_audit) != {"failed_rows", "inventory", "spec"}
            or full_hard_audit["failed_rows"]
            # Native inventory lists declared relations/caps. A plain banded
            # model can have none while its complete physical checks pass.
            or type(full_hard_audit["inventory"]) not in (tuple, list)
            or not full_hard_audit["spec"]):
        raise ValueError("incumbent replay cannot complete without all native evidence")
    from clay.layout.place_blocks_2d import _diagnostic_canonical_value
    from clay.layout.place.generated_incumbent import current_generated_incumbent_adjacency_projection
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
    canonical_frame = {key: _diagnostic_canonical_value(value) for key, value in frame.items()}
    adjacency_projection = current_generated_incumbent_adjacency_projection()
    model_bytes = model.Proto().SerializeToString(deterministic=True)
    parameters_bytes = solver.parameters.SerializeToString(deterministic=True)
    response_bytes = solver.ResponseProto().SerializeToString(deterministic=True)
    row = {
        "schema": "clay.native-incumbent-replay-stage/v1", "phase": phase,
        "solver_status": solver.StatusName(status), "optimization_status": "NOT_RUN",
        "aggregate_status": "NOT_RUN", "primitive_status": "PASS",
        "full_hard_status": "PASS", "caps_status": "PASS", "graph_status": "PASS",
        "model_bytes": model_bytes, "parameters_bytes": parameters_bytes, "response_bytes": response_bytes,
        "original_model_bytes": original_model_bytes,
        "original_model_proto_sha256": hashlib.sha256(original_model_bytes).hexdigest(),
        "model_proto_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "solver_parameters_sha256": hashlib.sha256(parameters_bytes).hexdigest(),
        "solver_response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "candidate_positions_nm": dict(positions),
        "candidate_positions_sha256": digest(_diagnostic_canonical_value(positions)),
        "canonical_frame": canonical_frame, "canonical_frame_sha256": digest(canonical_frame),
        "adjacency_projection": adjacency_projection,
        "adjacency_projection_sha256": digest(adjacency_projection),
        "graph_sha256": graph.sha256, "primitive_checks": deepcopy(primitive_checks),
        "primitive_checks_sha256": digest(_diagnostic_canonical_value(primitive_checks)),
        "full_hard_audit": deepcopy(full_hard_audit),
        "full_hard_audit_sha256": digest(_diagnostic_canonical_value(full_hard_audit)),
        "hierarchical_cohesion_span_caps_nm": _diagnostic_canonical_value(cohesion_caps),
        "hierarchical_cohesion_span_caps_sha256": digest(_diagnostic_canonical_value(cohesion_caps)),
    }
    session._completed.append(row)


def prepare_incumbent_replay_model(model, *, coordinate_variables, positions):
    """Clone the complete CP, changing only its objective and coordinate hints.

    The caller must actually Solve with fix_variables_to_their_hinted_value and
    independently check the response and the full native hard constraints.
    This helper neither solves nor creates placement evidence.
    """
    if (not isinstance(coordinate_variables, Mapping) or not coordinate_variables
            or not isinstance(positions, Mapping) or set(positions) != set(coordinate_variables)):
        raise ValueError("incumbent replay requires the complete coordinate inventory")
    proto = model.Proto()
    hints = []
    for name in sorted(coordinate_variables):
        variables, point = coordinate_variables[name], positions[name]
        if (type(name) is not str or not name or type(variables) not in (tuple, list)
                or len(variables) != 2 or type(point) not in (tuple, list) or len(point) != 2
                or any(type(value) is not int or value < 0 for value in point)):
            raise ValueError("incumbent replay coordinates must be exact nonnegative integers")
        for axis, variable, value in zip(("x", "y"), variables, point):
            index = variable.Index()
            if (type(index) is not int or not 0 <= index < len(proto.variables)
                    or variable.Name() != proto.variables[index].name
                    or variable.Name() not in (axis + "_" + name, "full2d_" + axis + "_" + name)):
                raise ValueError("incumbent replay coordinate does not identify its model variable")
            domain = proto.variables[index].domain
            if not any(lo <= value <= hi for lo, hi in zip(domain[::2], domain[1::2])):
                raise ValueError("incumbent replay coordinate is outside the original domain")
            hints.append((index, value))
    if len({index for index, _value in hints}) != len(hints):
        raise ValueError("incumbent replay repeats a coordinate variable")
    before = proto.SerializeToString(deterministic=True)
    replay = model.Clone()
    replay.ClearObjective()
    replay.Proto().ClearField("solution_hint")
    for index, value in hints:
        replay.AddHint(replay.GetIntVarFromProtoIndex(index), value)
    restored = type(proto)()
    restored.CopyFrom(replay.Proto())
    for field in ("objective", "floating_point_objective", "solution_hint"):
        restored.ClearField(field)
        if proto.HasField(field):
            getattr(restored, field).CopyFrom(getattr(proto, field))
    if (restored.SerializeToString(deterministic=True) != before
            or proto.SerializeToString(deterministic=True) != before):
        raise ValueError("incumbent replay changed the original model")
    invalid = replay.Validate()
    if invalid:
        raise ValueError("invalid incumbent replay model: " + invalid)
    return replay


def require_incumbent_replay_response(model, solver, status, *, positions, coordinate_variables):
    """Check actual response domains, all original primitives and exact X/Y."""
    from ortools.sat.python import cp_model
    from clay.layout.place.production_model_evidence import _check_proto_solution, _expression_value
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        raise ValueError("incumbent replay requires actual SAT; solver_status=" + solver.StatusName(status))
    try:
        response = solver.ResponseProto()
    except RuntimeError as exc:
        raise ValueError("incumbent replay has no actual solver response") from exc
    solution = tuple(response.solution)
    if response.status != status or len(solution) != len(model.Proto().variables):
        raise ValueError("incumbent replay response status/variable inventory differs")
    checks = _check_proto_solution(model.Proto(), solution)
    # Primary area construction has an integer product, unlike the linear
    # fixed-outline adapter. Evaluate that original primitive exactly here.
    for row in checks:
        if row.get("status") == "UNSUPPORTED_CP_PRIMITIVE:int_prod":
            index = int(row["row_id"].split(":")[1])
            constraint = model.Proto().constraints[index]
            active = all(solution[literal] == 1 if literal >= 0 else solution[-literal-1] == 0
                         for literal in constraint.enforcement_literal)
            product = math.prod(_expression_value(expr, solution) for expr in constraint.int_prod.exprs)
            valid = not active or _expression_value(constraint.int_prod.target, solution) == product
            row["status"] = "PASS" if valid else "FAIL"
    if not checks or any(row["status"] != "PASS" for row in checks):
        raise ValueError("incumbent replay original CP primitive check failed")
    for name, point in positions.items():
        if tuple(solution[var.Index()] for var in coordinate_variables[name]) != tuple(point):
            raise ValueError("incumbent replay response changed supplied coordinates")
    return checks


@contextmanager
def scoped_area_search_execution_policy(policy: AreaSearchExecutionPolicy) -> Iterator[None]:
    """Restore the previous policy on every exit, including nested contexts."""
    if type(policy) is not AreaSearchExecutionPolicy:
        raise TypeError("area search execution policy must be an AreaSearchExecutionPolicy")
    token = _AREA_SEARCH_POLICY.set(policy)
    try:
        yield
    finally:
        _AREA_SEARCH_POLICY.reset(token)


def canonical_tie_break_enabled() -> bool:
    policy = _AREA_SEARCH_POLICY.get()
    return policy is not None and policy.canonical_tie_break


def apply_area_search_execution_policy(solver) -> Callable[[], None] | None:
    """Apply scoped execution fields and return their exact restoration callback.

    The caller must restore in a finally around the single intended Solve; a
    later phase may reuse the same solver. No policy means no parameter access.
    """
    policy = _AREA_SEARCH_POLICY.get()
    if policy is None:
        return None
    parameters = solver.parameters
    fields = ("stop_after_first_solution", "max_time_in_seconds")
    if policy.num_search_workers is not None:
        fields += ("num_search_workers",)
    if policy.max_deterministic_time is not None:
        fields += ("max_deterministic_time",)
    if policy.random_seed is not None:
        fields += ("random_seed",)
    previous = {name: (parameters.HasField(name), getattr(parameters, name)) for name in fields}

    def restore() -> None:
        for name, (was_present, value) in previous.items():
            if was_present:
                setattr(parameters, name, value)
            else:
                parameters.ClearField(name)

    try:
        parameters.stop_after_first_solution = policy.stop_after_first_solution
        if policy.num_search_workers is not None:
            parameters.num_search_workers = policy.num_search_workers
        if policy.max_deterministic_time is not None:
            parameters.max_deterministic_time = min(
                parameters.max_deterministic_time, policy.max_deterministic_time,
            )
        if policy.random_seed is not None:
            parameters.random_seed = policy.random_seed
        parameters.max_time_in_seconds = min(
            parameters.max_time_in_seconds, policy.max_wall_time_s,
        )
    except BaseException:
        restore()
        raise
    return restore
