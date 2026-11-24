"""Diagnostic full-background assumption interaction probes.

Every non-target production assumption remains enabled.  Target rows are
toggled through positive/negative assumption literals on cloned models, so the
probe never deletes constraints or mutates the production model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any, Callable


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _literal_parts(raw_literal: int) -> tuple[int, bool]:
    if type(raw_literal) is not int:
        raise TypeError("CP-SAT assumption literal must be exact")
    return (
        (raw_literal, True)
        if raw_literal >= 0
        else (-raw_literal - 1, False)
    )


def _canonical_assumption_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    required = {
        "row_kind",
        "row_id",
        "family",
        "obligation_id",
        "declaration_source",
        "assumption_literal",
        "assumption_index",
        "witness_status",
    }
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("complete assumption row inventory is required")
    output = []
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("assumption row schema differs")
        row = dict(raw)
        if row["row_kind"] != "ASSUMPTION":
            raise ValueError("interaction matrix accepts assumption rows only")
        if any(
            type(row[field]) is not str or not row[field]
            for field in required - {"assumption_index"}
        ):
            raise TypeError("assumption row text is malformed")
        if type(row["assumption_index"]) is not int or row["assumption_index"] < 0:
            raise TypeError("assumption row index is malformed")
        output.append(row)
    output.sort(key=lambda row: row["row_id"])
    if (
        len({row["row_id"] for row in output}) != len(output)
        or len({row["assumption_literal"] for row in output}) != len(output)
        or len({row["assumption_index"] for row in output}) != len(output)
    ):
        raise ValueError("assumption inventory is not one-to-one")
    return tuple(json.loads(_canonical_bytes(output).decode("ascii")))


def solve_full_background_assumption_case(
    *,
    model: Any,
    assumption_rows: Sequence[Mapping[str, object]],
    assumption_overrides: Mapping[str, bool],
    max_deterministic_time: float,
    solver_factory: Callable[[], Any] | None = None,
) -> dict[str, object]:
    """Solve one complete-background case with only named rows toggled."""

    from ortools.sat.python import cp_model

    rows = _canonical_assumption_rows(assumption_rows)
    if not isinstance(assumption_overrides, Mapping) or not assumption_overrides:
        raise ValueError("one or more named assumption overrides are required")
    overrides = dict(assumption_overrides)
    if any(type(key) is not str or not key for key in overrides) or any(
        type(enabled) is not bool for enabled in overrides.values()
    ):
        raise TypeError("assumption overrides are malformed")
    by_row_id = {str(row["row_id"]): row for row in rows}
    if not set(overrides) <= set(by_row_id):
        raise ValueError("assumption override row is absent from production inventory")
    if (
        not isinstance(max_deterministic_time, (int, float))
        or isinstance(max_deterministic_time, bool)
        or not math.isfinite(float(max_deterministic_time))
        or float(max_deterministic_time) <= 0.0
    ):
        raise ValueError("interaction solve budget must be positive")

    live_by_name: dict[str, int] = {}
    for raw_literal in tuple(model.Proto().assumptions):
        index, positive = _literal_parts(int(raw_literal))
        variable = model.GetBoolVarFromProtoIndex(index)
        name = variable.Name()
        if not positive or not name or name in live_by_name:
            raise ValueError("live production assumption inventory is malformed")
        live_by_name[name] = index
    expected = {
        str(row["assumption_literal"]): int(row["assumption_index"])
        for row in rows
    }
    if expected != live_by_name:
        raise ValueError("attributed rows differ from live production assumptions")

    assignments: dict[str, bool] = {}
    row_id_by_index: dict[int, str] = {}
    for row in rows:
        row_id = str(row["row_id"])
        enabled = overrides.get(row_id, True)
        assignments[row_id] = enabled
        index = int(row["assumption_index"])
        row_id_by_index[index] = row_id

    # Decide feasibility with constants first.  CP-SAT must preserve assumptions
    # during presolve so that it can return a core; that restriction is valuable
    # only after infeasibility is known and can make a satisfiable toggle remain
    # UNKNOWN.  Fixed Boolean equalities are logically identical here and allow
    # the full presolve for the SAT/UNSAT decision.
    trial = model.Clone()
    trial.ClearObjective()
    trial.ClearAssumptions()
    for row in rows:
        variable = trial.GetBoolVarFromProtoIndex(int(row["assumption_index"]))
        trial.Add(variable == int(assignments[str(row["row_id"])]))
    solver = (solver_factory or cp_model.CpSolver)()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_deterministic_time = float(max_deterministic_time)
    solver.parameters.use_energetic_reasoning_in_no_overlap_2d = True
    status = solver.Solve(trial)
    status_name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        verdict = "SAT"
        failed_rows: list[dict[str, object]] = []
    elif status == cp_model.INFEASIBLE:
        verdict = "INFEASIBLE"
        core_trial = model.Clone()
        core_trial.ClearObjective()
        core_trial.ClearAssumptions()
        literals = []
        for row in rows:
            row_id = str(row["row_id"])
            variable = core_trial.GetBoolVarFromProtoIndex(
                int(row["assumption_index"])
            )
            if assignments[row_id]:
                literals.append(variable)
            else:
                # A toggled-off row is background state, not a candidate cause
                # of the remaining conflict.  Make that state a hard constant
                # so the returned sufficient core can contain enabled rows only.
                core_trial.Add(variable == 0)
        core_trial.AddAssumptions(literals)
        core_solver = (solver_factory or cp_model.CpSolver)()
        core_solver.parameters.CopyFrom(solver.parameters)
        core_status = core_solver.Solve(core_trial)
        if core_status != cp_model.INFEASIBLE:
            raise RuntimeError(
                "fixed-assignment INFEASIBLE decision did not reproduce with "
                "assumption provenance"
            )
        failed_rows = []
        for raw_literal in core_solver.SufficientAssumptionsForInfeasibility():
            index, positive = _literal_parts(int(raw_literal))
            row_id = row_id_by_index.get(index)
            if row_id is None:
                raise RuntimeError(
                    "interaction solver returned an unmapped sufficient-core row"
                )
            source = by_row_id[row_id]
            failed_rows.append({
                "row_id": row_id,
                "obligation_id": source["obligation_id"],
                "declaration_source": source["declaration_source"],
                "assumption_literal": source["assumption_literal"],
                "assumption_index": source["assumption_index"],
                "witness_status": source["witness_status"],
                "assumed_enabled": positive,
            })
        failed_rows.sort(key=lambda row: row["row_id"])
    elif status == cp_model.UNKNOWN:
        verdict = "UNKNOWN"
        failed_rows = []
    else:
        raise RuntimeError(f"interaction solver failed with {status_name}")
    payload: dict[str, object] = {
        "schema": "clay.full-background-assumption-case/v1",
        "diagnostic_only": True,
        "model_proto_sha256": hashlib.sha256(
            model.Proto().SerializeToString(deterministic=True)
        ).hexdigest(),
        "complete_assumption_row_count": len(rows),
        "complete_assumption_rows_sha256": hashlib.sha256(
            _canonical_bytes(rows)
        ).hexdigest(),
        "assumption_assignments": assignments,
        "overrides": dict(sorted(overrides.items())),
        "disabled_row_ids": sorted(
            row_id for row_id, enabled in assignments.items() if not enabled
        ),
        "decision_mode": "FIXED_ASSIGNMENT_PRESOLVE",
        "solver_status": status_name,
        "verdict": verdict,
        "failed_rows": failed_rows,
        "candidate_solution_values": (
            list(solver.ResponseProto().solution) if verdict == "SAT" else None
        ),
        "solved_model_proto_sha256": (
            hashlib.sha256(trial.Proto().SerializeToString(deterministic=True)).hexdigest()
            if verdict == "SAT" else None
        ),
    }
    payload["artifact_sha256"] = hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()
    return payload


def solve_full_background_assumption_matrix(
    *,
    model: Any,
    assumption_rows: Sequence[Mapping[str, object]],
    target_row_ids: Sequence[str],
    max_deterministic_time: float,
    solver_factory: Callable[[], Any] | None = None,
    sat_case_sink: Callable[..., None] | None = None,
) -> dict[str, object]:
    """Solve the complete truth table for ``target_row_ids``.

    The input model is never modified.  Each case clones it, clears only the
    objective and assumption list, then re-adds every production assumption:
    non-targets positively, enabled targets positively, and disabled targets
    negatively.  Consequently all unguarded hard background constraints remain
    byte-identical and active in every cell.

    ``sat_case_sink`` can inspect an existing incumbent, using keyword arguments
    ``target_mask``, ``model``, ``solver``, and ``solver_status``. It must not
    mutate the solved model. The full positive mask is the only case eligible
    for a production evidence adapter; other cases remain diagnostics.
    """

    from ortools.sat.python import cp_model

    rows = _canonical_assumption_rows(assumption_rows)
    targets = tuple(target_row_ids)
    if (
        not targets
        or len(targets) > 12
        or len(set(targets)) != len(targets)
        or any(type(row_id) is not str or not row_id for row_id in targets)
    ):
        raise ValueError("target assumption row IDs are malformed")
    by_row_id = {str(row["row_id"]): row for row in rows}
    if not set(targets) <= set(by_row_id):
        raise ValueError("target assumption row is absent from production inventory")
    if (
        not isinstance(max_deterministic_time, (int, float))
        or isinstance(max_deterministic_time, bool)
        or not math.isfinite(float(max_deterministic_time))
        or float(max_deterministic_time) <= 0.0
    ):
        raise ValueError("interaction solve budget must be positive")

    live_by_index: dict[int, tuple[str, Any]] = {}
    live_by_name: dict[str, int] = {}
    for raw_literal in tuple(model.Proto().assumptions):
        index, positive = _literal_parts(int(raw_literal))
        variable = model.GetBoolVarFromProtoIndex(index)
        name = variable.Name()
        if not positive or not name or index in live_by_index or name in live_by_name:
            raise ValueError("live production assumption inventory is malformed")
        live_by_index[index] = (name, variable)
        live_by_name[name] = index
    expected = {
        str(row["assumption_literal"]): int(row["assumption_index"])
        for row in rows
    }
    if expected != live_by_name:
        raise ValueError("attributed rows differ from live production assumptions")
    row_id_by_index = {
        int(row["assumption_index"]): str(row["row_id"])
        for row in rows
    }
    proto_sha = hashlib.sha256(
        model.Proto().SerializeToString(deterministic=True)
    ).hexdigest()
    cases: list[dict[str, object]] = []
    for mask in range(1 << len(targets)):
        enabled_targets = {
            row_id for bit, row_id in enumerate(targets) if mask & (1 << bit)
        }
        trial = model.Clone()
        trial.ClearObjective()
        trial.ClearAssumptions()
        assignments: dict[str, bool] = {}
        literals = []
        for row in rows:
            row_id = str(row["row_id"])
            enabled = row_id not in targets or row_id in enabled_targets
            assignments[row_id] = enabled
            variable = trial.GetBoolVarFromProtoIndex(int(row["assumption_index"]))
            literals.append(variable if enabled else variable.Not())
        trial.AddAssumptions(literals)
        solved_proto_sha = hashlib.sha256(
            trial.Proto().SerializeToString(deterministic=True)
        ).hexdigest()
        solver = (solver_factory or cp_model.CpSolver)()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_deterministic_time = float(max_deterministic_time)
        solver.parameters.use_energetic_reasoning_in_no_overlap_2d = True
        status = solver.Solve(trial)
        status_name = solver.StatusName(status)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            verdict = "SAT"
            failed_rows: list[dict[str, object]] = []
            if sat_case_sink is not None:
                sat_case_sink(
                    target_mask=mask, model=trial, solver=solver,
                    solver_status=status,
                )
                if solved_proto_sha != hashlib.sha256(
                    trial.Proto().SerializeToString(deterministic=True)
                ).hexdigest():
                    raise ValueError("SAT evidence observer mutated the solved model")
        elif status == cp_model.INFEASIBLE:
            verdict = "INFEASIBLE"
            core_indices = tuple(solver.SufficientAssumptionsForInfeasibility())
            failed_rows = []
            for raw_literal in core_indices:
                index, positive = _literal_parts(int(raw_literal))
                row_id = row_id_by_index.get(index)
                if row_id is None:
                    raise RuntimeError(
                        "interaction solver returned an unmapped sufficient-core row"
                    )
                source = by_row_id[row_id]
                failed_rows.append({
                    "row_id": row_id,
                    "obligation_id": source["obligation_id"],
                    "declaration_source": source["declaration_source"],
                    "assumption_literal": source["assumption_literal"],
                    "witness_status": source["witness_status"],
                    "assumed_enabled": positive,
                })
            failed_rows.sort(key=lambda row: row["row_id"])
        elif status == cp_model.UNKNOWN:
            verdict = "UNKNOWN"
            failed_rows = []
        else:
            raise RuntimeError(
                f"interaction solver failed with {status_name}"
            )
        cases.append({
            "target_mask": mask,
            "enabled_target_row_ids": [
                row_id for row_id in targets if row_id in enabled_targets
            ],
            "assumption_assignments": assignments,
            "solver_status": status_name,
            "solved_model_proto_sha256": solved_proto_sha,
            "verdict": verdict,
            "failed_rows": failed_rows,
        })
    payload: dict[str, object] = {
        "schema": "clay.full-background-assumption-interaction/v1",
        "diagnostic_only": True,
        "model_proto_sha256": proto_sha,
        "complete_assumption_row_count": len(rows),
        "complete_assumption_row_ids": [row["row_id"] for row in rows],
        "complete_assumption_rows_sha256": hashlib.sha256(
            _canonical_bytes(rows)
        ).hexdigest(),
        "target_row_ids": list(targets),
        "cases": cases,
    }
    payload["artifact_sha256"] = hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()
    return payload


__all__ = [
    "solve_full_background_assumption_case",
    "solve_full_background_assumption_matrix",
]
