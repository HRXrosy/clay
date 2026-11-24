"""Actual bounded generated-cohesion execution, without source authority.

The caller supplies original CP evidence from its own verified fresh source
pipeline. This module replays that original response, derives A's exact outline
and all group span caps, then actually solves and replays Q. Only that execution
can return the process/thread-bound candidate record. No JSON reconstruction,
reference data, alternative coordinate input, cache, or authority registry is
provided. Upper full geometry/native replay and original-model fixed-hint
certification remain mandatory. Q is not a strict-similarity objective.

The integer square construction/replay is ported from the independently tested
generated-group squared diagnostic. Neither this module nor its dependencies
import the diagnostic or LDO packages.
"""

from collections.abc import Mapping
import hashlib
import itertools
import json
import math
import os
import re
import threading
import time

from ortools.sat import cp_model_pb2, sat_parameters_pb2
from ortools.sat.python import cp_model


INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
_PREFIX = "generated_squared_cohesion_"
_SCOPE = {"source_authority": "NOT_GRANTED", "graph_full_outer_status": "REQUIRED_FROM_CALLER",
          "original_model_fixed_hint_certification": "NOT_RUN"}
_STATUS = {cp_model.UNKNOWN: "UNKNOWN", cp_model.MODEL_INVALID: "MODEL_INVALID",
           cp_model.FEASIBLE: "FEASIBLE", cp_model.INFEASIBLE: "INFEASIBLE",
           cp_model.OPTIMAL: "OPTIMAL"}


def _bytes(proto):
    return proto.SerializeToString(deterministic=True)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("ascii"))


def _bound_sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("binding SHA must be an exact lowercase SHA-256")
    return value


def _int64(value, label):
    if type(value) is not int or not INT64_MIN <= value <= INT64_MAX:
        raise OverflowError(f"{label} exceeds signed int64 bounds")
    return value


def _finite_fields(message):
    for field, value in message.ListFields():
        values = value if field.label == field.LABEL_REPEATED else (value,)
        for item in values:
            if field.message_type is not None:
                _finite_fields(item)
            elif type(item) is float and not math.isfinite(item):
                raise ValueError("protobuf contains a nonfinite scalar")


def _parse(raw, proto):
    if type(raw) is not bytes:
        raise TypeError("protobuf input must be exact bytes")
    proto.ParseFromString(raw)
    if _bytes(proto) != raw:
        raise ValueError("protobuf input does not round-trip canonically")
    _finite_fields(proto)
    return proto


def _model(raw):
    if not raw:
        raise ValueError("original CP model bytes are empty")
    model = cp_model.CpModel()
    _parse(raw, model.Proto())
    error = model.Validate()
    if error:
        raise ValueError(f"invalid CP model: {error}")
    return model


def _pairs(mapping, *, minimum, label):
    if not isinstance(mapping, Mapping) or not mapping:
        raise TypeError(f"{label} must be a nonempty named mapping")
    result = {}
    for owner, values in mapping.items():
        if (type(owner) is not str or not owner or type(values) not in (list, tuple)
                or len(values) != 2 or any(type(v) is not int for v in values)
                or any(v < minimum or v > INT64_MAX for v in values)):
            raise ValueError(f"{label} requires exact owner names and integer pairs")
        result[owner] = tuple(values)
    return dict(sorted(result.items()))


def _geometry(positions, dims, declared_groups):
    points = _pairs(positions, minimum=0, label="generated positions")
    dimensions = _pairs(dims, minimum=1, label="dimensions")
    if set(points) != set(dimensions):
        raise ValueError("complete geometry owner inventory differs")
    groups = declared_groups
    if type(groups) is not tuple or not groups:
        raise ValueError("declared groups must be a nonempty canonical tuple")
    for group in groups:
        if (type(group) is not tuple or len(group) < 2
                or any(type(owner) is not str for owner in group)
                or group != tuple(sorted(set(group)))
                or not set(group) <= set(dimensions)):
            raise ValueError("declared group has unknown, duplicate or unordered owners")
    if groups != tuple(sorted(set(groups))):
        raise ValueError("declared group inventory is duplicated or unordered")
    return points, dimensions, groups


def evaluate_squared_distance(*, positions, dims, declared_groups):
    """Exact twice-centre arithmetic, preserving nested-group multiplicity."""
    points, dimensions, groups = _geometry(positions, dims, declared_groups)
    centres = {owner: tuple(_int64(2 * points[owner][axis] + dimensions[owner][axis],
                                  "twice-centre") for axis in (0, 1))
               for owner in dimensions}
    total = 0
    for group in groups:
        for left, right in itertools.combinations(group, 2):
            for axis in (0, 1):
                delta = _int64(centres[left][axis] - centres[right][axis], "centre delta")
                total = _int64(total + _int64(delta * delta, "square"), "sum Q")
    return total


def _in_domain(value, domain):
    return any(lo <= value <= hi for lo, hi in zip(domain[::2], domain[1::2]))


def _coordinates(proto, indices, dimensions, points):
    result = _pairs(indices, minimum=0, label="coordinate indices")
    if set(result) != set(dimensions):
        raise ValueError("complete coordinate owner inventory differs")
    flat = [index for pair in result.values() for index in pair]
    if len(flat) != len(set(flat)) or any(i >= len(proto.variables) for i in flat):
        raise ValueError("coordinate variables are missing or aliased")
    expected_names = {f"{axis}_{owner}" for owner in dimensions for axis in ("x", "y")}
    coordinate_names = [v.name for v in proto.variables if v.name.startswith(("x_", "y_"))]
    if set(coordinate_names) != expected_names or len(coordinate_names) != len(expected_names):
        raise ValueError("original CP coordinate name/owner inventory differs")
    for owner, pair in result.items():
        for axis, index in enumerate(pair):
            variable = proto.variables[index]
            if variable.name != f"{'xy'[axis]}_{owner}":
                raise ValueError("coordinate axis/owner identity differs")
            if not _in_domain(points[owner][axis], variable.domain):
                raise ValueError("generated coordinate is outside the actual CP domain")
            if variable.domain[0] < 0:
                raise ValueError("negative native coordinate domains are unsupported")
    return result


def _without_objective(proto):
    duplicate = type(proto)()
    duplicate.CopyFrom(proto)
    duplicate.ClearField("objective")
    duplicate.ClearField("floating_point_objective")
    return _bytes(duplicate)


def _build_squared_control(*, model_bytes, parameters_bytes, expected_model_sha256,
                          coordinate_indices, dims, generated_positions,
                          declared_groups, expected_operand_sha256):
    """Append square and generated-movement definitions; replace only objective."""
    if type(model_bytes) is not bytes or _sha(model_bytes) != _bound_sha(expected_model_sha256):
        raise ValueError("original model SHA binding differs")
    points, dimensions, groups = _geometry(generated_positions, dims, declared_groups)
    indices = _pairs(coordinate_indices, minimum=0, label="coordinate indices")
    operands = {"coordinate_indices": indices, "dims": dimensions,
                "generated_positions": points, "declared_groups": groups}
    if _digest(operands) != _bound_sha(expected_operand_sha256):
        raise ValueError("complete operand digest binding differs")
    model = _model(model_bytes)
    _parse(parameters_bytes, sat_parameters_pb2.SatParameters())
    original = type(model.Proto())()
    original.CopyFrom(model.Proto())
    _coordinates(original, indices, dimensions, points)
    if any(v.name.startswith(_PREFIX) for v in original.variables):
        raise ValueError("model already contains squared-control auxiliary variables")
    if original.HasField("floating_point_objective"):
        raise ValueError("floating original objective is unsupported")

    # Compute every bound using unbounded Python integers before modifying CP.
    centre_domains, movement_terms, movement_upper = {}, [], 0
    for owner, pair in indices.items():
        for axis, index in enumerate(pair):
            domain = original.variables[index].domain
            centre_domains[owner, axis] = tuple(
                _int64(2 * end + dimensions[owner][axis], "twice-centre bound")
                for end in (domain[0], domain[-1]))
            upper = _int64(max(abs(domain[0] - points[owner][axis]),
                               abs(domain[-1] - points[owner][axis])), "movement bound")
            movement_upper = _int64(movement_upper + upper, "sum movement bound")
            movement_terms.append({"owner": owner, "axis": axis, "upper_bound": upper})
    plan, q_upper = [], 0
    for group_index, group in enumerate(groups):
        for left, right in itertools.combinations(group, 2):
            for axis in (0, 1):
                a, b = centre_domains[left, axis], centre_domains[right, axis]
                low = _int64(a[0] - b[1], "delta lower bound")
                high = _int64(a[1] - b[0], "delta upper bound")
                square_upper = _int64(max(abs(low), abs(high)) ** 2, "square upper bound")
                q_upper = _int64(q_upper + square_upper, "sum Q bound")
                plan.append({"group_index": group_index, "owners": (left, right),
                             "axis": axis, "delta_domain_nm": (low, high),
                             "square_upper_bound": square_upper})

    square_vars = []
    for number, term in enumerate(plan):
        left, right = term["owners"]
        axis = term["axis"]
        delta = model.NewIntVar(*term["delta_domain_nm"], _PREFIX + f"delta_{number}")
        square = model.NewIntVar(0, term["square_upper_bound"], _PREFIX + f"square_{number}")
        a = model.GetIntVarFromProtoIndex(indices[left][axis])
        b = model.GetIntVarFromProtoIndex(indices[right][axis])
        model.Add(delta == 2 * a + dimensions[left][axis] - 2 * b - dimensions[right][axis])
        model.AddMultiplicationEquality(square, [delta, delta])
        term.update(delta_index=delta.Index(), square_index=square.Index())
        square_vars.append(square)
    q = model.NewIntVar(0, q_upper, _PREFIX + "sum")
    model.Add(q == sum(square_vars))
    movements = []
    for number, term in enumerate(movement_terms):
        owner, axis = term["owner"], term["axis"]
        movement = model.NewIntVar(0, term["upper_bound"], _PREFIX + f"movement_{number}")
        coord = model.GetIntVarFromProtoIndex(indices[owner][axis])
        model.AddAbsEquality(movement, coord - points[owner][axis])
        term["variable_index"] = movement.Index()
        movements.append(movement)
    movement = model.NewIntVar(0, movement_upper, _PREFIX + "movement_sum")
    model.Add(movement == sum(movements))
    model.Minimize(q)
    raw = _bytes(model.Proto())
    _model(raw)
    prefix = type(original)()
    prefix.CopyFrom(model.Proto())
    del prefix.variables[len(original.variables):]
    del prefix.constraints[len(original.constraints):]
    if _without_objective(prefix) != _without_objective(original):
        raise ValueError("augmentation modified the original complete CP prefix")
    return {"stage": "Q",
            **operands, "operand_sha256": expected_operand_sha256,
            "declared_groups_sha256": _digest(groups),
            "original_model_bytes": model_bytes, "original_model_sha256": _sha(model_bytes),
            "model_bytes": raw, "model_sha256": _sha(raw),
            "parameters_bytes": parameters_bytes, "parameters_sha256": _sha(parameters_bytes),
            "original_variable_count": len(original.variables),
            "original_constraint_count": len(original.constraints),
            "q_variable_index": q.Index(), "q_upper_bound": q_upper,
            "movement_variable_index": movement.Index(), "movement_upper_bound": movement_upper,
            "pair_occurrence_count": len(plan) // 2, "square_term_count": len(plan),
            "pair_terms": tuple(plan), "movement_terms": tuple(movement_terms)}


def _response(raw):
    response = _parse(raw, cp_model_pb2.CpSolverResponse())
    if response.status not in _STATUS:
        raise ValueError("unknown solver response status")
    return response


def _verified_control(control):
    if not isinstance(control, Mapping):
        raise TypeError("control must be a complete builder result")
    expected = _build_squared_control(
        model_bytes=control["original_model_bytes"], parameters_bytes=control["parameters_bytes"],
        expected_model_sha256=control["original_model_sha256"],
        coordinate_indices=control["coordinate_indices"], dims=control["dims"],
        generated_positions=control["generated_positions"], declared_groups=control["declared_groups"],
        expected_operand_sha256=control["operand_sha256"])
    if dict(control) != expected:
        raise ValueError("control model, prefix, operands or evidence identity differs")
    return _model(expected["model_bytes"]), _model(expected["original_model_bytes"])


def _expression(expr, solution):
    return int(expr.offset) + sum(int(coefficient) * (
        solution[index] if index >= 0 else -solution[-index - 1])
        for index, coefficient in zip(expr.vars, expr.coeffs))


def _additional_checks(proto, solution, variable_start, constraint_start):
    def literal(index):
        return solution[index] == 1 if index >= 0 else solution[-index - 1] == 0
    checks = [{"row_id": f"cp_domain:{i}",
               "status": "PASS" if _in_domain(solution[i], proto.variables[i].domain) else "FAIL"}
              for i in range(variable_start, len(proto.variables))]
    for index in range(constraint_start, len(proto.constraints)):
        row = proto.constraints[index]
        kind = row.WhichOneof("constraint")
        status = "PASS"
        if kind not in {"linear", "lin_max", "int_prod"}:
            status = "UNSUPPORTED_CP_PRIMITIVE:" + str(kind)
        elif all(literal(value) for value in row.enforcement_literal):
            body = getattr(row, kind)
            if kind == "linear":
                value = sum(int(c) * solution[v] for v, c in zip(body.vars, body.coeffs))
                valid = _in_domain(value, body.domain)
            elif kind == "lin_max":
                valid = _expression(body.target, solution) == max(
                    _expression(expr, solution) for expr in body.exprs)
            else:
                # Python integers replay signed products exactly, without float
                # objectives, int64 wraparound, or a square-root approximation.
                product = math.prod(_expression(expr, solution) for expr in body.exprs)
                valid = _expression(body.target, solution) == product
            if not valid:
                status = "FAIL"
        checks.append({"row_id": f"cp_primitive:{index}", "kind": kind, "status": status})
    return checks


def _checks_status(checks):
    if any(row["status"] == "FAIL" for row in checks):
        return "FAIL"
    if any(row["status"] != "PASS" for row in checks):
        return "UNSUPPORTED"
    return "PASS"


def _replay_squared_control(*, control, response_bytes):
    """Replay original and added CP primitives; never inherit an outer/G4 pass."""
    model, original = _verified_control(control)
    response = _response(response_bytes)
    result = {**_SCOPE, "solver_status": _STATUS[response.status],
              "model_sha256": control["model_sha256"], "response_sha256": _sha(response_bytes),
              "original_primitive_status": "NOT_EVALUATED",
              "additional_primitive_status": "NOT_EVALUATED",
              "graph_full_outer_caps_status": "NOT_EVALUATED"}
    if response.status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return {**result, "status": "NO_CANDIDATE"}
    solution = tuple(response.solution)
    if len(solution) != len(model.Proto().variables):
        raise ValueError("response variable inventory differs from complete control CP")
    from clay.layout.place.production_model_evidence import _check_proto_solution

    original_checks = _check_proto_solution(
        original.Proto(), solution[:control["original_variable_count"]])
    additional_checks = _additional_checks(model.Proto(), solution,
        control["original_variable_count"], control["original_constraint_count"])
    original_status, additional_status = _checks_status(original_checks), _checks_status(additional_checks)
    result.update(original_primitive_status=original_status,
                  additional_primitive_status=additional_status,
                  original_checks=original_checks, additional_checks=additional_checks)
    if original_status != "PASS" or additional_status != "PASS":
        return {**result, "status": "FAIL" if "FAIL" in (original_status, additional_status) else "UNSUPPORTED"}
    points = {owner: tuple(solution[i] for i in pair)
              for owner, pair in control["coordinate_indices"].items()}
    q_value = evaluate_squared_distance(positions=points, dims=control["dims"],
                                        declared_groups=control["declared_groups"])
    movement = sum(abs(points[owner][axis] - control["generated_positions"][owner][axis])
                   for owner in points for axis in (0, 1))
    if (solution[control["q_variable_index"]] != q_value
            or solution[control["movement_variable_index"]] != movement):
        return {**result, "status": "FAIL", "additional_primitive_status": "FAIL",
                "error": "direct Q/generated-movement integer replay differs"}
    integer_objective = q_value
    if response.objective_value != float(integer_objective):
        return {**result, "status": "FAIL", "error": "response objective scalar differs from integer replay"}
    if response.status == cp_model.OPTIMAL and response.best_objective_bound != float(integer_objective):
        return {**result, "status": "FAIL", "error": "OPTIMAL response bound differs from integer objective"}
    return {**result, "status": "PASS", "candidate_positions": points,
            "q_value": q_value, "movement_value": movement,
            "optimality_authority": "ACTUAL_Q_EXECUTION_RECORD_REQUIRED"}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _require_prefix(original, augmented):
    prefix = type(augmented)()
    prefix.CopyFrom(augmented)
    if len(prefix.variables) < len(original.variables) or len(prefix.constraints) < len(original.constraints):
        raise ValueError("original complete CP prefix was removed")
    del prefix.variables[len(original.variables):]
    del prefix.constraints[len(original.constraints):]
    if _without_objective(prefix) != _without_objective(original):
        raise ValueError("original complete CP prefix identity differs")


def _generated_domain(*, original, solution, indices, points, dimensions, groups):
    """Preserve A's actual occupied bbox and both spans of every complete group."""
    model = original.Clone()
    before = original.Proto()
    extended = list(solution)
    inventory = []
    caps = []
    if any(var.name.startswith("generated_cohesion_domain_") for var in before.variables):
        raise ValueError("generated cohesion domain already present")

    def envelope(owners, axis, name):
        variables = [model.GetIntVarFromProtoIndex(indices[owner][axis]) for owner in owners]
        low_bound = min(before.variables[indices[owner][axis]].domain[0] for owner in owners)
        high_bound = max(before.variables[indices[owner][axis]].domain[-1] for owner in owners)
        end_low = min(_int64(before.variables[indices[owner][axis]].domain[0] + dimensions[owner][axis],
                            "envelope lower bound") for owner in owners)
        end_high = max(_int64(before.variables[indices[owner][axis]].domain[-1] + dimensions[owner][axis],
                             "envelope upper bound") for owner in owners)
        actual_low = min(points[owner][axis] for owner in owners)
        actual_high = max(_int64(points[owner][axis] + dimensions[owner][axis], "A envelope") for owner in owners)
        low = model.NewIntVar(low_bound, high_bound, name + "_min")
        high = model.NewIntVar(end_low, end_high, name + "_max")
        if (low.Index(), high.Index()) != (len(extended), len(extended) + 1):
            raise ValueError("generated envelope variable prefix differs")
        extended.extend((actual_low, actual_high))
        model.AddMinEquality(low, variables)
        model.AddMaxEquality(high, [variable + dimensions[owner][axis]
                                   for variable, owner in zip(variables, owners)])
        return low, high, actual_low, actual_high

    bbox = [0, 0, 0, 0]
    for axis in (0, 1):
        start = len(model.Proto().constraints)
        low, high, a_low, a_high = envelope(tuple(dimensions), axis, f"generated_cohesion_domain_outline_{axis}")
        model.Add(low == a_low)
        model.Add(high == a_high)
        bbox[axis], bbox[axis + 2] = a_low, a_high
        inventory.append({"kind": "GENERATED_A_EXACT_OUTLINE", "axis": axis,
            "minimum_nm": a_low, "maximum_nm": a_high,
            "constraint_indices": list(range(start, len(model.Proto().constraints)))})
    for number, group in enumerate(groups):
        spans = []
        for axis in (0, 1):
            start = len(model.Proto().constraints)
            low, high, a_low, a_high = envelope(group, axis, f"generated_cohesion_domain_group_{number}_{axis}")
            cap = _int64(a_high - a_low, "A group span")
            model.Add(high - low <= cap)
            spans.append(cap)
            inventory.append({"kind": "GENERATED_A_GROUP_SPAN_CAP", "axis": axis, "owners": group,
                "cap_nm": cap, "constraint_indices": list(range(start, len(model.Proto().constraints)))})
        caps.append({"owners": group, "span_nm": tuple(spans)})
    _require_prefix(before, model.Proto())
    # This stage has no objective transformation at all.
    if _bytes(before.objective) != _bytes(model.Proto().objective):
        raise ValueError("generated domain changed the original objective")
    raw = _bytes(model.Proto())
    _model(raw)
    from clay.layout.place.production_model_evidence import _check_proto_solution
    checks = _check_proto_solution(model.Proto(), tuple(extended))
    if _checks_status(checks) != "PASS":
        raise ValueError("original A fails generated outline/cap primitives")
    return {"model_bytes": raw, "bbox_nm": tuple(bbox), "caps": tuple(caps),
            "inventory": tuple(inventory), "a_checks": tuple(checks)}


def _execution_parameters(original, *, max_deterministic_time, max_wall_time_s):
    budgets = (max_deterministic_time, max_wall_time_s)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in budgets):
        raise ValueError("generated cohesion budgets must be finite and positive")
    actual = sat_parameters_pb2.SatParameters()
    actual.CopyFrom(original)
    actual.max_deterministic_time = min(original.max_deterministic_time, max_deterministic_time)
    actual.max_time_in_seconds = min(original.max_time_in_seconds, max_wall_time_s)
    actual.stop_after_first_solution = False
    fields = ("max_deterministic_time", "max_time_in_seconds", "stop_after_first_solution")
    changes = {}
    for field in fields:
        before = {"present": original.HasField(field), "value": getattr(original, field) if original.HasField(field) else None}
        after = {"present": actual.HasField(field), "value": getattr(actual, field)}
        if before != after:
            changes[field] = {"before": before, "after": after}
    a, b = sat_parameters_pb2.SatParameters(), sat_parameters_pb2.SatParameters()
    a.CopyFrom(original)
    b.CopyFrom(actual)
    for field in fields:
        a.ClearField(field)
        b.ClearField(field)
    if _bytes(a) != _bytes(b):
        raise ValueError("undeclared generated cohesion parameter change")
    return actual, changes


_EXECUTION_MINT = object()


class GeneratedCohesionExecution:
    """Immutable same-process/thread evidence of this module's actual Q Solve.

    This API provenance guard is not a cryptographic barrier against arbitrary
    in-process code execution. Source authority always remains with the caller.
    """
    __slots__ = ("_seal", "_pid", "_thread", "_data", "_artifacts", "_geometry")

    def __new__(cls, *args, **kwargs):
        raise TypeError("generated cohesion execution requires actual execute_generated_cohesion")

    def __setattr__(self, name, value):
        raise TypeError("generated cohesion execution is immutable")

    def __reduce_ex__(self, protocol):
        raise TypeError("generated cohesion execution cannot be reconstructed")

    def _check_origin(self):
        if self._seal is not _EXECUTION_MINT:
            raise ValueError("generated cohesion execution mint identity differs")
        if (self._pid, self._thread) != (os.getpid(), threading.current_thread()):
            raise ValueError("generated cohesion execution process/thread differs")

    def evidence(self):
        self._check_origin()
        return json.loads(self._data)

    def artifacts(self):
        self._check_origin()
        return dict(self._artifacts)


def execute_generated_cohesion(*, original_model_bytes, original_parameters_bytes, original_response_bytes,
                              coordinate_indices, dims, declared_groups,
                              max_deterministic_time, max_wall_time_s):
    """Replay caller-provided original evidence, then perform exactly one Q Solve.

    Only original response coordinates are admitted as A. The caller must join
    those original bytes and all geometry/source operands to its fresh authority
    before using this execution in an upper certification pipeline.
    """
    original = _model(original_model_bytes)
    if original.Proto().objective.domain:
        raise ValueError("original objective domain is a hard bound and cannot be replaced by Q")
    parameters = _parse(original_parameters_bytes, sat_parameters_pb2.SatParameters())
    response = _response(original_response_bytes)
    if response.status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        raise ValueError("original response has no generated incumbent")
    if original.Proto().HasField("floating_point_objective"):
        raise ValueError("floating original objective is unsupported")
    solution = tuple(response.solution)
    from clay.layout.place.production_model_evidence import _check_proto_solution
    checks = _check_proto_solution(original.Proto(), solution)
    if _checks_status(checks) != "PASS":
        raise ValueError("original response fails complete original primitive/assumption replay")
    indices = _pairs(coordinate_indices, minimum=0, label="coordinate indices")
    if any(index >= len(solution) for pair in indices.values() for index in pair):
        raise ValueError("original response coordinate inventory differs")
    points = {name: tuple(solution[index] for index in pair) for name, pair in indices.items()}
    points, dimensions, groups = _geometry(points, dims, declared_groups)
    indices = _coordinates(original.Proto(), indices, dimensions, points)
    # Reject unsafe arithmetic before mutating an augmented model or solving.
    original_q = evaluate_squared_distance(positions=points, dims=dimensions, declared_groups=groups)
    domain = _generated_domain(original=original, solution=solution, indices=indices,
        points=points, dimensions=dimensions, groups=groups)
    actual_parameters, changes = _execution_parameters(parameters,
        max_deterministic_time=max_deterministic_time, max_wall_time_s=max_wall_time_s)
    parameter_bytes = _bytes(actual_parameters)
    operands = {"coordinate_indices": indices, "dims": dimensions, "generated_positions": points,
                "declared_groups": groups}
    control = _build_squared_control(model_bytes=domain["model_bytes"], parameters_bytes=parameter_bytes,
        expected_model_sha256=_sha(domain["model_bytes"]), **operands, expected_operand_sha256=_digest(operands))
    q_model = _model(control["model_bytes"])
    _require_prefix(original.Proto(), q_model.Proto())
    solver = cp_model.CpSolver()
    solver.parameters.CopyFrom(actual_parameters)
    start = time.perf_counter()
    status = solver.Solve(q_model)
    elapsed = time.perf_counter() - start
    q_response = solver.ResponseProto()
    response_bytes = _bytes(q_response)
    if (q_response.status != status or _bytes(q_model.Proto()) != control["model_bytes"]
            or _bytes(solver.parameters) != parameter_bytes):
        raise ValueError("actual Q solver model/parameter/response identity differs")
    replay = _replay_squared_control(control=control, response_bytes=response_bytes)
    artifacts = {"original-model.pb": original_model_bytes,
        "original-parameters.pb": original_parameters_bytes, "original-response.pb": original_response_bytes,
        "generated-domain-model.pb": domain["model_bytes"], "q-model.pb": control["model_bytes"],
        "q-parameters.pb": parameter_bytes, "q-response.pb": response_bytes}
    report = {**_SCOPE, "schema": "clay.generated-cohesion-execution/v1",
        "status": "NO_CANDIDATE", "solver_status": _STATUS[status],
        "original_evidence_origin": "CALLER_PROVIDED_SOLVER_BYTES_MATHEMATICALLY_REPLAYED",
        "q_execution_origin": "ACTUAL_LOCAL_CP_SAT_SOLVE",
        "original_candidate_positions": points, "original_q_value": original_q,
        "coordinate_indices": indices, "dims": dimensions, "declared_groups": groups,
        "fixed_bbox_nm": domain["bbox_nm"], "group_caps_nm": domain["caps"],
        "generated_domain_inventory": domain["inventory"], "a_generated_domain_checks": domain["a_checks"],
        "original_response_checks": checks, "original_assumptions": list(original.Proto().assumptions),
        "owner_count": len(dimensions), "active_owner_count": len({owner for group in groups for owner in group}),
        "pair_occurrence_count": control["pair_occurrence_count"], "square_term_count": control["square_term_count"],
        "pair_terms": control["pair_terms"], "q_variable_index": control["q_variable_index"],
        "movement_variable_index": control["movement_variable_index"],
        "parameter_changes": changes, "requested_budgets": {"max_deterministic_time": max_deterministic_time,
                                                             "max_wall_time_s": max_wall_time_s},
        "actual_wall_time_seconds": elapsed, "response_wall_time_seconds": q_response.wall_time,
        "response_deterministic_time": q_response.deterministic_time,
        "objective_value": q_response.objective_value, "best_objective_bound": q_response.best_objective_bound,
        "original_primitive_status": replay["original_primitive_status"],
        "additional_primitive_status": replay["additional_primitive_status"], "q_primitive_replay": replay,
        "artifact_sha256": {name: _sha(data) for name, data in artifacts.items()}}
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        if replay["status"] != "PASS":
            raise ValueError("actual Q response fails complete integer primitive replay")
        report.update(status="CANDIDATE_PRIMITIVES_PASS", candidate_positions=replay["candidate_positions"],
            q_value=replay["q_value"], movement_from_A_nm=replay["movement_value"])
    elif replay["status"] != "NO_CANDIDATE":
        raise ValueError("non-incumbent Q response exposed a candidate")
    record = object.__new__(GeneratedCohesionExecution)
    for name, value in {"_seal": _EXECUTION_MINT, "_pid": os.getpid(), "_thread": threading.current_thread(),
        "_data": _json_bytes(report), "_artifacts": tuple(sorted(artifacts.items())),
        "_geometry": _json_bytes({"coordinate_indices": indices, "dims": dimensions, "declared_groups": groups})}.items():
        object.__setattr__(record, name, value)
    return record


def require_generated_cohesion_candidate(execution, *, original_model_bytes, original_parameters_bytes,
                                        original_response_bytes, coordinate_indices, dims, declared_groups):
    """Join current caller evidence to the exact original execution; return copied XY."""
    if type(execution) is not GeneratedCohesionExecution:
        raise TypeError("candidate requires the live generated cohesion execution")
    execution._check_origin()
    artifacts = execution.artifacts()
    for role, value in (("model", original_model_bytes), ("parameters", original_parameters_bytes),
                        ("response", original_response_bytes)):
        if type(value) is not bytes or value != artifacts[f"original-{role}.pb"]:
            raise ValueError(f"generated cohesion original {role} identity differs")
    try:
        indices = _pairs(coordinate_indices, minimum=0, label="coordinate indices")
        dimensions = _pairs(dims, minimum=1, label="dimensions")
        groups = declared_groups
        _geometry({owner: (0, 0) for owner in dimensions}, dimensions, groups)
        identity = _json_bytes({"coordinate_indices": indices, "dims": dimensions, "declared_groups": groups})
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("generated cohesion geometry identity differs") from exc
    if identity != execution._geometry:
        raise ValueError("generated cohesion complete geometry identity differs")
    report = execution.evidence()
    if report["status"] != "CANDIDATE_PRIMITIVES_PASS":
        raise ValueError("generated cohesion execution has no verified Q candidate")
    return {owner: tuple(xy) for owner, xy in report["candidate_positions"].items()}


__all__ = ["GeneratedCohesionExecution", "execute_generated_cohesion",
           "require_generated_cohesion_candidate", "evaluate_squared_distance"]
