"""Actual bounded reference reconstruction, followed by conditional Q search.

Only this executor's fresh Solve calls can mint a candidate record. Public
reference targets and current owner-local geometry are data: the caller still
authenticates their original sources and materialization. Structure is explicitly
conditional on unchanged freshly checked artwork and consumption; actual native
certification, G4 and strict evaluation remain mandatory upper gates.

Centroid polygons define a sufficient inner domain; labelled class X, greedy Y,
contained-frame envelopes and module midpoint relations are exact. Fixed logical
patterns and D4 counts are checked before solving. Complete final evaluation
and separate current artwork/consumption conditions remain mandatory. Q begins
only after a complete conditional witness. No diagnostic package, saved
candidate, coordinate seed, window or alternate model input is consumed.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from fractions import Fraction
import json
import math
import os
import sys
import threading
import time

from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model

from clay.layout.place import generated_cohesion as primitives
from clay.layout.place.production_model_evidence import _check_proto_solution
from clay.layout.reference_quality_contract import ReferenceQualityContract, RigidQualityGeometryIR
from clay.layout.reference_quality_metrics import evaluate_reference_quality


_PREFIX = "reference_quality_"
_PPM = 1_000_000
_SAFE = (1 << 62) - 1
_RADIUS = 49_996
_AXIS_ERROR = Fraction(500_000_001, 500_000_000)
_SCOPE = {
    "source_authority": "NOT_GRANTED",
    "graph_full_outer_status": "REQUIRED_FROM_CALLER",
    "original_model_fixed_hint_certification": "NOT_RUN",
    "actual_g4_and_strict": "REQUIRED_FROM_CALLER_ON_SAME_CANDIDATE",
    "full_domain_optimality": "NOT_CLAIMED",
    "full_domain_infeasibility": "NOT_CLAIMED",
    "structure_scope": "CONDITIONAL_UNCHANGED_FRESH_ARTWORK_AND_MATERIALIZED_CONSUMPTION",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _owner():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    # Keep the actual lifetime objects alive. Integer thread IDs and id(task)
    # can both be recycled after their original execution contexts terminate.
    return os.getpid(), threading.current_thread(), task


def _rational(value):
    return [value.numerator, value.denominator]


def _hull(points):
    ordered = sorted(set(points))
    lower, upper = [], []
    for result, sequence in ((lower, ordered), (upper, reversed(ordered))):
        for point in sequence:
            while len(result) >= 2:
                a, b = result[-2:]
                cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
                if cross > 0:
                    break
                result.pop()
            result.append(point)
    return lower[:-1] + upper[:-1]


def _inner_polygon():
    """Floating angles only propose vertices; integer checks certify the hull."""
    points = []
    for index in range(128):
        angle = 2 * math.pi * index / 128
        point = [int(round(_RADIUS * math.cos(angle))), int(round(_RADIUS * math.sin(angle)))]
        while sum(value * value for value in point) > _RADIUS * _RADIUS:
            axis = 0 if abs(point[0]) >= abs(point[1]) else 1
            point[axis] -= 1 if point[axis] > 0 else -1
        points.append(tuple(point))
    vertices = _hull(points)
    _require(len(vertices) == 128 and all(x * x + y * y <= _RADIUS * _RADIUS for x, y in vertices),
             "inner polygon vertex count or exact disk containment differs")
    facets = []
    for first, second in zip(vertices, vertices[1:] + vertices[:1]):
        a, b = second[1] - first[1], first[0] - second[0]
        divisor = math.gcd(abs(a), abs(b))
        _require(divisor > 0, "inner polygon has a degenerate edge")
        a, b = a // divisor, b // divisor
        c = a * first[0] + b * first[1]
        _require(c > 0 and all(a * x + b * y <= c for x, y in vertices),
                 "inner polygon does not have a complete supporting facet inventory")
        facets.append({"a": a, "b": b, "c": c})
    result = {"radius_ppm": _RADIUS, "vertices": [list(point) for point in vertices], "facets": facets}
    return {**result, "sha256": primitives._digest(result)}


def _integer_halfplane(coefficients, rhs):
    """Clear exact rationals and floor the RHS after coefficient gcd reduction."""
    _require(type(coefficients) is dict and bool(coefficients) and type(rhs) in (int, Fraction),
             "halfplane requires exact rational operands")
    _require(all(type(index) is int and index >= 0 and type(value) in (int, Fraction)
                 for index, value in coefficients.items()), "halfplane coordinate operands are malformed")
    values = {index: Fraction(value) for index, value in sorted(coefficients.items()) if value}
    _require(bool(values), "centroid halfplane has no coordinate terms")
    rhs = Fraction(rhs)
    denominator = math.lcm(rhs.denominator, *(value.denominator for value in values.values()))
    scaled = [value.numerator * (denominator // value.denominator) for value in values.values()]
    scaled_rhs = rhs.numerator * (denominator // rhs.denominator)
    divisor = math.gcd(*(abs(value) for value in scaled))
    reduced, upper = [value // divisor for value in scaled], scaled_rhs // divisor
    _require(all(abs(value) <= _SAFE for value in (*reduced, upper)), "centroid facet coefficients exceed int64 safety")
    return {"variable_indices": list(values), "coefficients": reduced, "upper_bound": upper,
            "clearing_denominator": denominator, "coefficient_gcd": divisor, "unreduced_rhs": scaled_rhs}


def _binding(*, coordinate_indices, dims, declared_groups, geometry, contract):
    if type(geometry) is not RigidQualityGeometryIR or type(contract) is not ReferenceQualityContract:
        raise TypeError("candidate identity requires exact public contract and live geometry IR")
    geometry.__post_init__()
    contract.__post_init__()
    indices = primitives._pairs(coordinate_indices, minimum=0, label="coordinate indices")
    dimensions = primitives._pairs(dims, minimum=1, label="dimensions")
    primitives._geometry({name: (0, 0) for name in dimensions}, dimensions, declared_groups)
    _require({name: (width, height) for name, width, height in geometry.owners} == dimensions
             and set(indices) == set(dimensions), "live geometry/dimension/coordinate identity differs")
    _require(Counter(row.logical_id for row in geometry.occurrences)
             == Counter({row.logical_id: len(row.occurrence_ids) for row in contract.logical_population}),
             "complete public/current occurrence population differs")
    return primitives._json_bytes({"coordinate_indices": indices, "dims": dimensions, "declared_groups": declared_groups,
        "geometry": geometry.canonical_data(), "contract": contract.canonical_data()})


def _compile_centroids(*, domain_bytes, indices, geometry, contract, frame):
    """Append unconditional facets, preserving every original assumption/hint.

For n occurrences the affine residual is the exact mean of unrounded normalized
owner-local points minus the public reference centroid. Rounding each point and
then its mean changes an axis by less than 1.000000002 ppm under these bounds;
the retained 49996-ppm hull consequently lies safely inside native acceptance.
"""
    _require(sys.implementation.name == "cpython" and sys.version_info >= (3, 10)
             and (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.rounds) == (2, 53, 1),
             "centroid search requires bounded nearest-binary64 CPython arithmetic")
    extents = tuple(frame[axis + 2] - frame[axis] for axis in (0, 1))
    _require(all(0 < value <= _PPM for value in extents) and len(geometry.occurrences) <= _PPM,
             "centroid search frame or occurrence count exceeds bounded arithmetic proof")
    model = primitives._model(domain_bytes)
    original = model.Proto()
    _require(not any(row.name.startswith(_PREFIX) for row in original.variables)
             and not any(row.name.startswith(_PREFIX) for row in original.constraints), "quality model is already augmented")
    model.ClearObjective()
    polygon = _inner_polygon()
    populations = {("logical", row.logical_id): (row.logical_id,) for row in contract.logical_population}
    populations.update({("class", row.class_id): row.logical_ids for row in contract.class_population})
    inventory = []
    for target in contract.location_targets:
        names = populations[target.scope, target.target_id]
        occurrences = tuple(row for row in geometry.occurrences if row.logical_id in names)
        counts = Counter(row.owner_id for row in occurrences)
        count = len(occurrences)
        _require(0 < count <= _PPM, "centroid population is outside bounded arithmetic")
        axes = []
        for axis in (0, 1):
            factor = Fraction(_PPM, count * extents[axis])
            constant = factor * sum(row.center_offset_nm[axis] - frame[axis] for row in occurrences)
            constant -= target.centroid_ppm[axis]
            axes.append({"constant": _rational(constant), "terms": [
                {"owner": owner, "coordinate_index": indices[owner][axis], "occurrence_count": multiplicity,
                 "coefficient": _rational(factor * multiplicity)} for owner, multiplicity in sorted(counts.items())]})
        planes = []
        for number, facet in enumerate(polygon["facets"]):
            coefficients, rhs = Counter(), Fraction(facet["c"])
            for factor, axis in zip((facet["a"], facet["b"]), axes):
                rhs -= factor * Fraction(*axis["constant"])
                for term in axis["terms"]:
                    coefficients[term["coordinate_index"]] += factor * Fraction(*term["coefficient"])
            plane = _integer_halfplane(dict(coefficients), rhs)
            activity = abs(plane["upper_bound"])
            for index, coefficient in zip(plane["variable_indices"], plane["coefficients"]):
                domain = model.Proto().variables[index].domain
                activity += abs(coefficient) * max(abs(int(domain[0])), abs(int(domain[-1])))
            _require(activity <= _SAFE, "centroid facet activity exceeds int64 safety")
            expression = sum(coefficient * model.GetIntVarFromProtoIndex(index)
                for index, coefficient in zip(plane["variable_indices"], plane["coefficients"]))
            constraint_index = len(model.Proto().constraints)
            model.Add(expression <= plane["upper_bound"]).WithName(
                f"{_PREFIX}{target.scope}:{target.target_id}:facet:{number}")
            planes.append({**plane, "constraint_index": constraint_index, "activity_bound": activity})
        inventory.append({"scope": target.scope, "target_id": target.target_id, "occurrence_count": count,
            "reference_centroid_ppm": list(target.centroid_ppm), "axes": axes, "facet_planes": planes})
    original = primitives._model(domain_bytes).Proto()
    primitives._require_prefix(original, model.Proto())
    raw = primitives._bytes(model.Proto())
    primitives._model(raw)
    metadata = {"scope": "SUFFICIENT_CENTROID_INNER_DOMAIN", "full_acceptance_equivalence": False,
        "metric_count": len(inventory), "facet_count": sum(len(row["facet_planes"]) for row in inventory),
        "added_assumption_count": 0, "polygon": polygon, "metrics": inventory,
        "axis_affine_error_strict_bound": _rational(_AXIS_ERROR), "native_centroid_limit_ppm": contract.centroid_limit_ppm,
        "logical_count": len(contract.logical_population), "class_count": len(contract.class_population),
        "occurrence_count": len(geometry.occurrences), "frame_nm": list(frame), "model_sha256": primitives._sha(raw)}
    return model, metadata


def _add_no_good(model, indices, positions, number):
    """Exclude exactly one complete owner placement, independent of auxiliaries."""
    _require(set(indices) == set(positions), "rejection cut must cover every owner coordinate")
    first = len(model.Proto().constraints)
    changed = []
    for owner in sorted(indices):
        for axis, index in enumerate(indices[owner]):
            flag = model.NewBoolVar(f"{_PREFIX}no_good:{number}:{owner}:{axis}")
            model.Add(model.GetIntVarFromProtoIndex(index) != positions[owner][axis]).OnlyEnforceIf(flag)
            changed.append(flag)
    model.Add(sum(changed) >= 1)
    return {"rejected_positions": {owner: list(positions[owner]) for owner in sorted(positions)},
            "constraint_indices": list(range(first, len(model.Proto().constraints))),
            "scope": "EXACT_COMPLETE_COORDINATE_ASSIGNMENT_ONLY"}


def _compile_quality_model(*, domain_bytes, indices, geometry, contract, frame):
    from clay.layout.place.class_x_rank_constraints import append_class_x_rank_constraints
    from clay.layout.place.class_y_rank_constraints import append_class_y_rank_constraints, validate_rigid_quality_patterns
    from clay.layout.place.module_relation_constraints import append_module_relation_constraints
    from clay.layout.place.quality_location_constraints import append_quality_location_constraints
    invariants = validate_rigid_quality_patterns(geometry, contract)
    model, centroids = _compile_centroids(domain_bytes=domain_bytes, indices=indices,
        geometry=geometry, contract=contract, frame=frame)
    columns = append_class_x_rank_constraints(model, coordinate_indices=indices,
        geometry=geometry, contract=contract)
    rows = append_class_y_rank_constraints(model, coordinate_indices=indices, geometry=geometry, contract=contract)
    relations = append_module_relation_constraints(model, coordinate_indices=indices,
        geometry=geometry, contract=contract, frame_nm=tuple(frame))
    locations = append_quality_location_constraints(model, coordinate_indices=indices,
        geometry=geometry, contract=contract, frame_nm=tuple(frame))
    return model, centroids, columns, relations, rows, locations, invariants


def _attempt_parameters(original, *, deterministic, wall, phase):
    parameters, _changes = primitives._execution_parameters(original,
        max_deterministic_time=deterministic, max_wall_time_s=wall)
    parameters.stop_after_first_solution = phase == "FEASIBILITY"
    changes = {}
    for field in ("max_deterministic_time", "max_time_in_seconds", "stop_after_first_solution"):
        before = {"present": original.HasField(field), "value": getattr(original, field) if original.HasField(field) else None}
        after = {"present": parameters.HasField(field), "value": getattr(parameters, field)}
        if before != after:
            changes[field] = {"before": before, "after": after}
    return parameters, changes


def _primitive_replay(*, original, model, response, indices, dimensions, groups, q_index):
    solution = tuple(response.solution)
    _require(len(solution) == len(model.Proto().variables), "actual response complete variable inventory differs")
    primitives._require_prefix(original.Proto(), model.Proto())
    original_checks = _check_proto_solution(original.Proto(), solution[:len(original.Proto().variables)])
    additional_checks = primitives._additional_checks(model.Proto(), solution,
        len(original.Proto().variables), len(original.Proto().constraints))
    _require(primitives._checks_status(original_checks) == "PASS"
             and primitives._checks_status(additional_checks) == "PASS", "actual response fails complete primitive replay")
    positions = {owner: tuple(solution[index] for index in pair) for owner, pair in indices.items()}
    q_value = primitives.evaluate_squared_distance(positions=positions, dims=dimensions, declared_groups=groups)
    objective = 0 if q_index is None else q_value
    # CP-SAT's response scalar can differ by one binary64 ULP after presolve
    # scaling (an actual integer Q=100 response returned 100.00000000000001).
    # Every coordinate, square, sum and improvement bound above remains exact;
    # this tolerance checks only the solver's non-authoritative display scalar.
    tolerance = 0.0 if q_index is None else math.ulp(float(objective))
    _require((q_index is None or solution[q_index] == q_value)
             and abs(response.objective_value - float(objective)) <= tolerance,
             f"actual objective or Q definition differs from exact integer replay: index={q_index}, "
             f"variable={None if q_index is None else solution[q_index]}, Q={q_value}, "
             f"objective={response.objective_value!r}, expected={objective}")
    if response.status == cp_model.OPTIMAL:
        _require(abs(response.best_objective_bound - float(objective)) <= tolerance, "actual optimal objective bound differs")
    return positions, q_value, {"original_primitive_status": "PASS", "additional_primitive_status": "PASS",
        "objective_integer_value": objective, "objective_scalar_tolerance": tolerance,
        "original_checks": original_checks, "additional_checks": additional_checks}


def _quality_replay(*, positions, geometry, contract, frame, dimensions, groups, caps, centroid_search, location_search, solution):
    from clay.layout.place.quality_location_constraints import replay_location_search
    boxes = [(x, y, x + dimensions[owner][0], y + dimensions[owner][1]) for owner, (x, y) in positions.items()]
    actual_frame = (min(box[0] for box in boxes), min(box[1] for box in boxes),
                    max(box[2] for box in boxes), max(box[3] for box in boxes))
    _require(actual_frame == tuple(frame), "actual candidate differs from exact generated A outline")
    for group, cap in zip(groups, caps):
        _require(tuple(cap["owners"]) == group and all(
            max(positions[owner][axis] + dimensions[owner][axis] for owner in group)
            - min(positions[owner][axis] for owner in group) <= cap["span_nm"][axis] for axis in (0, 1)),
            "actual candidate differs from complete A group caps")
    report = evaluate_reference_quality(positions=positions, geometry=geometry, contract=contract, frame_nm=tuple(frame),
        required_structure_status="pass", current_structure_status="pass")
    replay_location_search(report, centroid_search, location_search, solution, contract)
    return report


_MINT = object()


class GeneratedReferenceQualityExecution:
    """Immutable actual-Solve record tied to one process, thread and async task.

This API provenance guard is not protection against arbitrary in-process code
execution. Public data and this record do not authenticate upper source inputs.
"""
    __slots__ = ("_seal", "_origin", "_data", "_artifacts", "_binding")

    def __new__(cls, *args, **kwargs):
        raise TypeError("reference quality execution requires actual execute_generated_reference_quality")

    def __setattr__(self, name, value):
        raise TypeError("reference quality execution is immutable")

    def __reduce_ex__(self, protocol):
        raise TypeError("reference quality execution cannot be reconstructed")

    def _check_origin(self):
        current = _owner()
        _require(self._seal is _MINT and self._origin[0] == current[0]
                 and self._origin[1] is current[1] and self._origin[2] is current[2],
                 "reference quality execution process/thread/task identity differs")

    def evidence(self):
        self._check_origin()
        return json.loads(self._data)

    def artifacts(self):
        self._check_origin()
        return dict(self._artifacts)


def execute_generated_reference_quality(*, original_model_bytes, original_parameters_bytes, original_response_bytes,
        coordinate_indices, dims, declared_groups, geometry, contract,
        max_deterministic_time, max_wall_time_s, max_attempts=16):
    """Search fresh A's complete hard domain under an explicit public contract."""
    started = time.perf_counter()
    _require(type(max_attempts) is int and max_attempts > 0, "max_attempts must be a positive exact integer")
    _require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                 for value in (max_deterministic_time, max_wall_time_s)), "quality execution budgets must be finite and positive")
    binding = _binding(coordinate_indices=coordinate_indices, dims=dims, declared_groups=declared_groups,
        geometry=geometry, contract=contract)
    original = primitives._model(original_model_bytes)
    _require(not original.Proto().objective.domain and not original.Proto().HasField("floating_point_objective"),
             "original objective hard domain or floating objective cannot be replaced")
    parameters = primitives._parse(original_parameters_bytes, sat_parameters_pb2.SatParameters())
    response = primitives._response(original_response_bytes)
    _require(response.status in (cp_model.FEASIBLE, cp_model.OPTIMAL), "original response has no incumbent")
    original_checks = _check_proto_solution(original.Proto(), tuple(response.solution))
    _require(primitives._checks_status(original_checks) == "PASS", "original response fails complete original primitive/assumption replay")
    indices = primitives._pairs(coordinate_indices, minimum=0, label="coordinate indices")
    _require(all(index < len(response.solution) for pair in indices.values() for index in pair), "original coordinate inventory differs")
    points = {owner: tuple(response.solution[index] for index in pair) for owner, pair in indices.items()}
    points, dimensions, groups = primitives._geometry(points, dims, declared_groups)
    indices = primitives._coordinates(original.Proto(), indices, dimensions, points)
    original_q = primitives.evaluate_squared_distance(positions=points, dims=dimensions, declared_groups=groups)
    domain = primitives._generated_domain(original=original, solution=tuple(response.solution), indices=indices,
        points=points, dimensions=dimensions, groups=groups)
    search_model, centroid_search, class_x_search, module_relation_search, class_y_search, location_search, invariants = _compile_quality_model(
        domain_bytes=domain["model_bytes"], indices=indices,
        geometry=geometry, contract=contract, frame=domain["bbox_nm"])
    artifacts = {"original-model.pb": original_model_bytes, "original-parameters.pb": original_parameters_bytes,
        "original-response.pb": original_response_bytes, "generated-domain-model.pb": domain["model_bytes"],
        "quality-model.pb": primitives._bytes(search_model.Proto()),
        "contract.json": primitives._json_bytes(contract.canonical_data()), "geometry.json": primitives._json_bytes(geometry.canonical_data())}
    det_limit = min(parameters.max_deterministic_time, max_deterministic_time)
    wall_limit = min(parameters.max_time_in_seconds, max_wall_time_s)
    attempts, accepted = [], None
    deterministic_used = 0.0
    search_result = "BOUNDED_SEARCH_UNRESOLVED"
    q_status = "Q_OPTIMIZATION_UNRESOLVED"
    q_build_failure = None
    operands = {"coordinate_indices": indices, "dims": dimensions, "generated_positions": points, "declared_groups": groups}
    for number in range(max_attempts):
        phase = "FEASIBILITY" if accepted is None else "Q"
        model, q_control = search_model.Clone(), None
        if accepted is not None:
            input_bytes = primitives._bytes(model.Proto())
            try:
                q_control = primitives._build_squared_control(model_bytes=input_bytes, parameters_bytes=original_parameters_bytes,
                    expected_model_sha256=primitives._sha(input_bytes), **operands, expected_operand_sha256=primitives._digest(operands))
            except OverflowError as error:
                # Wide original domains can make conservative Q square bounds
                # unsupported even though the accepted placement is tiny and
                # fully verified. No Q Solve occurred and no Q artifact exists.
                # Replay/model identity failures remain exceptions below.
                q_status = "Q_MODEL_ARITHMETIC_UNSUPPORTED"
                q_build_failure = {"phase": "Q", "exception_type": "OverflowError", "message": str(error),
                                   "accepted_attempt_retained": accepted["attempt"]}
                break
            model = primitives._model(q_control["model_bytes"])
            model.Add(model.GetIntVarFromProtoIndex(q_control["q_variable_index"]) <= accepted["q_value"] - 1)
        remaining_det = det_limit - deterministic_used
        remaining_wall = wall_limit - (time.perf_counter() - started)
        if remaining_det <= 0 or remaining_wall <= 0:
            break
        actual_parameters, changes = _attempt_parameters(parameters,
            deterministic=remaining_det, wall=remaining_wall, phase=phase)
        model_bytes, parameter_bytes = primitives._bytes(model.Proto()), primitives._bytes(actual_parameters)
        primitives._model(model_bytes)
        primitives._require_prefix(original.Proto(), model.Proto())
        solver = cp_model.CpSolver()
        solver.parameters.CopyFrom(actual_parameters)
        solve_started = time.perf_counter()
        status = solver.Solve(model)
        elapsed = time.perf_counter() - solve_started
        actual_response = solver.ResponseProto()
        response_bytes = primitives._bytes(actual_response)
        _require(actual_response.status == status and primitives._bytes(model.Proto()) == model_bytes
                 and primitives._bytes(solver.parameters) == parameter_bytes,
                 "actual quality solver model/parameters/response identity differs")
        primitives._response(response_bytes)
        deterministic_used += actual_response.deterministic_time
        names = {role: f"attempt-{number:03d}-{role}.pb" for role in ("model", "parameters", "response")}
        artifacts.update({names["model"]: model_bytes, names["parameters"]: parameter_bytes, names["response"]: response_bytes})
        attempt = {"attempt": number, "phase": phase, "solver_status": primitives._STATUS[status], "artifacts": names,
            "parameter_changes": changes, "actual_wall_time_seconds": elapsed,
            "response_wall_time_seconds": actual_response.wall_time, "response_deterministic_time": actual_response.deterministic_time,
            "objective_value": actual_response.objective_value, "best_objective_bound": actual_response.best_objective_bound,
            "q_variable_index": None if q_control is None else q_control["q_variable_index"],
            "q_upper_bound": None if accepted is None else accepted["q_value"] - 1, "outcome": "NO_CANDIDATE"}
        attempts.append(attempt)
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            if status == cp_model.INFEASIBLE:
                if accepted is None:
                    search_result = "NO_WITNESS_IN_SUFFICIENT_CENTROID_DOMAIN"
                else:
                    q_status = "NO_BETTER_QUALITY_WITNESS_IN_SUFFICIENT_CENTROID_DOMAIN"
            break
        candidate, q_value, replay = _primitive_replay(original=original, model=model, response=actual_response,
            indices=indices, dimensions=dimensions, groups=groups, q_index=attempt["q_variable_index"])
        quality = _quality_replay(positions=candidate, geometry=geometry, contract=contract, frame=domain["bbox_nm"],
            dimensions=dimensions, groups=groups, caps=domain["caps"], centroid_search=centroid_search,
            location_search=location_search, solution=tuple(actual_response.solution))
        attempt.update(**replay, candidate_positions=candidate, q_value=q_value, quality_evaluation=quality)
        if quality["strict_status"] == "pass":
            _require(accepted is None or q_value < accepted["q_value"], "Q candidate does not improve the retained quality witness")
            attempt["outcome"] = "QUALITY_ACCEPTED"
            accepted = attempt
            search_result = "COMPLETE_CONDITIONAL_QUALITY_WITNESS"
        else:
            attempt["outcome"] = "QUALITY_REJECTED"
            attempt["rejection_cut"] = _add_no_good(search_model, indices, candidate, number)
    report = {**_SCOPE, "schema": "clay.generated-reference-quality-execution/v1", "status": "NO_CANDIDATE",
        "execution_origin": "ACTUAL_LOCAL_CP_SAT_SOLVES", "original_evidence_origin": "CALLER_PROVIDED_BYTES_PRIMITIVELY_REPLAYED",
        "original_candidate_positions": points, "original_q_value": original_q,
        "coordinate_indices": indices, "dims": dimensions, "declared_groups": groups,
        "owner_count": len(dimensions), "active_owner_count": len({owner for group in groups for owner in group}),
        "fixed_bbox_nm": domain["bbox_nm"], "group_caps_nm": domain["caps"],
        "generated_domain_inventory": domain["inventory"], "a_generated_domain_checks": domain["a_checks"],
        "original_response_checks": original_checks, "original_assumptions": list(original.Proto().assumptions),
        "contract_sha256": contract.sha256, "geometry_sha256": geometry.sha256, "centroid_search": centroid_search,
        "class_x_search": class_x_search,
        "module_relation_search": module_relation_search,
        "class_y_search": class_y_search, "location_search": location_search, "fixed_pattern_invariants": invariants,
        "requested_budgets": {"max_deterministic_time": max_deterministic_time, "max_wall_time_s": max_wall_time_s, "max_attempts": max_attempts},
        "actual_wall_time_seconds": time.perf_counter() - started, "response_deterministic_time": deterministic_used,
        "attempts": attempts, "accepted_attempt": None, "search_result": search_result, "q_optimization_status": q_status,
        "terminal_solver_status": attempts[-1]["solver_status"] if attempts else "NOT_RUN",
        "artifact_sha256": {name: primitives._sha(value) for name, value in artifacts.items()}}
    if q_build_failure is not None:
        report["q_build_failure"] = q_build_failure
    if accepted is not None:
        report.update(status="QUALITY_FEASIBLE", accepted_attempt=accepted["attempt"], candidate_positions=accepted["candidate_positions"],
            solver_status=accepted["solver_status"], q_value=accepted["q_value"], quality_evaluation=accepted["quality_evaluation"],
            quality_status="CONDITIONAL_COMPLETE_QUALITY_PASS", original_primitive_status="PASS", additional_primitive_status="PASS",
            movement_from_A_nm=sum(abs(accepted["candidate_positions"][owner][axis] - points[owner][axis]) for owner in points for axis in (0, 1)))
    else:
        report["solver_status"] = report["terminal_solver_status"]
    execution = object.__new__(GeneratedReferenceQualityExecution)
    for name, value in {"_seal": _MINT, "_origin": _owner(), "_data": primitives._json_bytes(report),
                        "_artifacts": tuple(sorted(artifacts.items())), "_binding": binding}.items():
        object.__setattr__(execution, name, value)
    return execution


def require_generated_reference_quality_candidate(execution, *, original_model_bytes, original_parameters_bytes,
        original_response_bytes, coordinate_indices, dims, declared_groups, geometry, contract):
    """Bind a live execution to current caller operands and replay its accepted point."""
    if type(execution) is not GeneratedReferenceQualityExecution:
        raise TypeError("candidate requires a live reference quality execution")
    execution._check_origin()
    artifacts = execution.artifacts()
    for role, value in (("model", original_model_bytes), ("parameters", original_parameters_bytes), ("response", original_response_bytes)):
        _require(type(value) is bytes and artifacts[f"original-{role}.pb"] == value, f"original {role} identity differs")
    binding = _binding(coordinate_indices=coordinate_indices, dims=dims, declared_groups=declared_groups,
        geometry=geometry, contract=contract)
    _require(binding == execution._binding, "reference quality contract/geometry/operand identity differs")
    report = execution.evidence()
    _require(report["status"] == "QUALITY_FEASIBLE", "reference quality execution has no verified candidate")
    _require(report["artifact_sha256"] == {name: primitives._sha(value) for name, value in artifacts.items()},
             "actual execution artifact identity differs")
    original = primitives._model(original_model_bytes)
    initial_response = primitives._response(original_response_bytes)
    indices = primitives._pairs(coordinate_indices, minimum=0, label="coordinate indices")
    points = {owner: tuple(initial_response.solution[index] for index in pair)
              for owner, pair in indices.items()}
    points, dimensions, groups = primitives._geometry(points, dims, declared_groups)
    indices = primitives._coordinates(original.Proto(), indices, dimensions, points)
    domain = primitives._generated_domain(original=original, solution=tuple(initial_response.solution),
        indices=indices, points=points, dimensions=dimensions, groups=groups)
    initial_model, centroids, columns, relations, rows, locations, invariants = _compile_quality_model(domain_bytes=domain["model_bytes"],
        indices=indices, geometry=geometry, contract=contract, frame=domain["bbox_nm"])
    _require(domain["model_bytes"] == artifacts["generated-domain-model.pb"]
             and primitives._bytes(initial_model.Proto()) == artifacts["quality-model.pb"]
             and primitives._digest(centroids) == primitives._digest(report["centroid_search"])
             and primitives._digest(columns) == primitives._digest(report["class_x_search"])
             and primitives._digest(relations) == primitives._digest(report["module_relation_search"])
             and primitives._digest(rows) == primitives._digest(report["class_y_search"])
             and primitives._digest(locations) == primitives._digest(report["location_search"])
             and primitives._digest(invariants) == primitives._digest(report["fixed_pattern_invariants"]),
             "current compiled quality model or metric inventory identity differs")
    attempt = report["attempts"][report["accepted_attempt"]]
    model = primitives._model(artifacts[attempt["artifacts"]["model"]])
    response = primitives._response(artifacts[attempt["artifacts"]["response"]])
    _require(response.status in (cp_model.FEASIBLE, cp_model.OPTIMAL), "accepted actual response has no candidate")
    positions, q_value, _replay = _primitive_replay(original=original, model=model, response=response,
        indices=coordinate_indices, dimensions=dims, groups=declared_groups, q_index=attempt["q_variable_index"])
    quality = _quality_replay(positions=positions, geometry=geometry, contract=contract, frame=report["fixed_bbox_nm"],
        dimensions=dims, groups=declared_groups, caps=report["group_caps_nm"], centroid_search=report["centroid_search"],
        location_search=report["location_search"], solution=tuple(response.solution))
    _require(quality["strict_status"] == "pass" and primitives._digest(quality) == primitives._digest(report["quality_evaluation"])
             and q_value == report["q_value"] and primitives._digest(positions) == primitives._digest(report["candidate_positions"]),
             "retained candidate complete quality or coordinate identity differs")
    return positions


__all__ = ["GeneratedReferenceQualityExecution", "execute_generated_reference_quality",
           "require_generated_reference_quality_candidate"]
