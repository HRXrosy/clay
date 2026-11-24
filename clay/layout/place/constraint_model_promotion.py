"""Fail-closed promotion contract for placement constraint-model evidence.

Reduced assumption experiments are useful diagnostics, but they are not a
certificate for the production constraint set.  This module keeps those claims
separate and admits grounding only after the complete production row inventory
is present in the solver and independently covered by hard replay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any


_INVENTORY_SCHEMA = "clay.placement-production-row-inventory/v1"
_PROMOTION_SCHEMA = "clay.constraint-model-promotion/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ROW_KINDS = frozenset(("ASSUMPTION", "HARD_BACKGROUND"))
_SAT_STATUSES = frozenset(("FEASIBLE", "OPTIMAL"))


class ConstraintModelPromotionError(RuntimeError):
    """Raised when a caller tries to ground an unpromoted constraint model."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _data_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _canonical_rows(rows: object, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError(f"{label} must contain production rows")
    canonical: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    assumption_literals: set[str] = set()
    assumption_indices: set[int] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError(f"{label} row is not an object")
        row = dict(raw)
        kind = row.get("row_kind")
        required = {
            "row_kind",
            "row_id",
            "family",
            "obligation_id",
            "declaration_source",
        }
        if kind == "ASSUMPTION":
            required |= {
                "assumption_literal",
                "assumption_index",
                "witness_status",
            }
        if kind not in _ROW_KINDS or set(row) != required:
            raise ValueError(f"{label} row schema differs")
        for field in required - {"assumption_index"}:
            if type(row[field]) is not str or not row[field]:
                raise TypeError(f"{label} row {field} is not one exact string")
        identity = (str(kind), str(row["row_id"]))
        if identity in identities:
            raise ValueError(f"{label} has duplicate row identity")
        identities.add(identity)
        if kind == "ASSUMPTION":
            literal = row["assumption_literal"]
            index = row["assumption_index"]
            if type(index) is not int or index < 0:
                raise TypeError(f"{label} assumption index is malformed")
            if literal in assumption_literals or index in assumption_indices:
                raise ValueError(f"{label} assumption binding is not one-to-one")
            assumption_literals.add(literal)
            assumption_indices.add(index)
            if row["witness_status"] not in {
                "PASS",
                "FAIL",
                "NOT_EVALUATED",
                "UNMEASURABLE",
            }:
                raise ValueError(f"{label} witness status differs")
        canonical.append(row)
    canonical.sort(key=lambda row: (row["row_kind"], row["row_id"]))
    return json.loads(_canonical_bytes(canonical).decode("ascii"))


def build_production_constraint_inventory(
    *,
    stage_input_sha256: str,
    model_proto_sha256: str,
    projected_rows: Sequence[Mapping[str, object]],
    materialized_rows: Sequence[Mapping[str, object]],
    active_solver_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bind projection, lowering, and live-solver views of one full model.

    The three row sequences intentionally arrive through separate arguments.
    Matching self-reported digests are not enough: their canonical row objects
    must be byte-identical, and the inventory must name always-on hard
    background rows as well as assumption-labelled rows.
    """

    stage_sha = _require_sha256(stage_input_sha256, field="stage input SHA")
    proto_sha = _require_sha256(model_proto_sha256, field="model proto SHA")
    projected = _canonical_rows(projected_rows, label="projected inventory")
    materialized = _canonical_rows(
        materialized_rows, label="materialized inventory"
    )
    active = _canonical_rows(active_solver_rows, label="active solver inventory")
    if projected != materialized or projected != active:
        raise ValueError(
            "projection, materialized, and active solver row inventories differ"
        )
    if not any(row["row_kind"] == "HARD_BACKGROUND" for row in projected):
        raise ValueError("production inventory lacks named hard background rows")
    row_ids = [[row["row_kind"], row["row_id"]] for row in projected]
    row_ids_sha = _data_sha256(row_ids)
    rows_sha = _data_sha256(projected)
    payload: dict[str, object] = {
        "schema": _INVENTORY_SCHEMA,
        "stage_input_sha256": stage_sha,
        "model_proto_sha256": proto_sha,
        "rows": projected,
        "row_count": len(projected),
        "row_ids_sha256": row_ids_sha,
        "rows_sha256": rows_sha,
        "expected_projection_row_ids_sha256": row_ids_sha,
        "materialized_model_row_ids_sha256": row_ids_sha,
        "active_solver_row_ids_sha256": row_ids_sha,
    }
    payload["artifact_sha256"] = _data_sha256(payload)
    return payload


def _validate_inventory(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("production row inventory is not an object")
    inventory = dict(value)
    expected_fields = {
        "schema",
        "stage_input_sha256",
        "model_proto_sha256",
        "rows",
        "row_count",
        "row_ids_sha256",
        "rows_sha256",
        "expected_projection_row_ids_sha256",
        "materialized_model_row_ids_sha256",
        "active_solver_row_ids_sha256",
        "artifact_sha256",
    }
    if set(inventory) != expected_fields or inventory["schema"] != _INVENTORY_SCHEMA:
        raise ValueError("production row inventory schema differs")
    rows = _canonical_rows(inventory["rows"], label="production inventory")
    row_ids = [[row["row_kind"], row["row_id"]] for row in rows]
    row_ids_sha = _data_sha256(row_ids)
    rows_sha = _data_sha256(rows)
    if (
        inventory["row_count"] != len(rows)
        or inventory["row_ids_sha256"] != row_ids_sha
        or inventory["rows_sha256"] != rows_sha
        or inventory["expected_projection_row_ids_sha256"] != row_ids_sha
        or inventory["materialized_model_row_ids_sha256"] != row_ids_sha
        or inventory["active_solver_row_ids_sha256"] != row_ids_sha
    ):
        raise ValueError("production row inventory does not replay")
    _require_sha256(inventory["stage_input_sha256"], field="stage input SHA")
    _require_sha256(inventory["model_proto_sha256"], field="model proto SHA")
    reported_artifact_sha = inventory.pop("artifact_sha256")
    if reported_artifact_sha != _data_sha256(inventory):
        raise ValueError("production row inventory artifact digest differs")
    inventory["artifact_sha256"] = reported_artifact_sha
    inventory["rows"] = rows
    return inventory


def _reduced_model_blocker(value: object) -> tuple[dict[str, object] | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        raise TypeError("reduced model evidence is not an object")
    reduced = dict(value)
    if reduced.get("claim") != "REDUCED_MODEL_SAT":
        raise ValueError("reduced model evidence claim differs")
    if reduced.get("solver_status") not in _SAT_STATUSES:
        raise ValueError("reduced model SAT evidence lacks a solver incumbent")
    if reduced.get("diagnostic_only") is not True:
        raise ValueError("reduced model evidence must be diagnostic-only")
    if type(reduced.get("row_count")) is not int or reduced["row_count"] <= 0:
        raise ValueError("reduced model evidence row count is malformed")
    _require_sha256(reduced.get("row_ids_sha256"), field="reduced row IDs SHA")
    return reduced, "REDUCED_MODEL_SAT_IS_DIAGNOSTIC_ONLY"


def _validate_sufficient_core(
    value: object,
    *,
    inventory: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("full-model INFEASIBLE evidence lacks its sufficient core")
    core = dict(value)
    expected_fields = {
        "kind",
        "solver_reported_core_literal_count",
        "row_count",
        "rows",
        "row_ids_sha256",
        "rows_sha256",
    }
    if set(core) != expected_fields or core["kind"] not in {
        "SUFFICIENT_NOT_MINIMAL",
        "SUFFICIENT_MINIMAL",
    }:
        raise ValueError("full-model sufficient core schema differs")
    raw_rows = core["rows"]
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("full-model sufficient core rows are malformed")
    row_fields = {
        "row_id",
        "assumption_literal",
        "assumption_index",
        "obligation_id",
        "declaration_source",
        "witness_status",
    }
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != row_fields:
            raise ValueError("full-model sufficient core row schema differs")
        row = dict(raw)
        if any(
            type(row[field]) is not str or not row[field]
            for field in row_fields - {"assumption_index"}
        ) or type(row["assumption_index"]) is not int or row["assumption_index"] < 0:
            raise TypeError("full-model sufficient core row is malformed")
        rows.append(row)
    rows.sort(key=lambda row: row["row_id"])
    if (
        not rows
        or len({row["row_id"] for row in rows}) != len(rows)
        or len({row["assumption_literal"] for row in rows}) != len(rows)
        or len({row["assumption_index"] for row in rows}) != len(rows)
    ):
        raise ValueError("full-model sufficient core is empty or not one-to-one")
    if (
        core["solver_reported_core_literal_count"] != len(rows)
        or core["row_count"] != len(rows)
        or core["row_ids_sha256"]
        != _data_sha256([row["row_id"] for row in rows])
        or core["rows_sha256"] != _data_sha256(rows)
    ):
        raise ValueError("full-model sufficient core does not replay completely")
    if inventory is not None:
        assumption_by_id = {
            row["row_id"]: row
            for row in inventory["rows"]
            if row["row_kind"] == "ASSUMPTION"
        }
        for row in rows:
            production = assumption_by_id.get(row["row_id"])
            if production is None or any(
                row[field] != production[field] for field in row_fields
            ):
                raise ValueError(
                    "full-model sufficient core contains a foreign or drifted row"
                )
    core["rows"] = rows
    return core


def build_constraint_model_promotion_receipt(
    *,
    reduced_model: Mapping[str, object] | None,
    production_inventory: Mapping[str, object] | None,
    full_model: Mapping[str, object] | None,
    full_hard_replay: Mapping[str, object] | None,
    required_adapter: bool = False,
) -> dict[str, object]:
    """Evaluate the independent claims; production callers require the adapter."""

    reduced, reduced_blocker = _reduced_model_blocker(reduced_model)
    inventory = (
        None
        if production_inventory is None
        else _validate_inventory(production_inventory)
    )
    blockers: list[str] = []
    if inventory is None:
        blockers.append("MISSING_PRODUCTION_ROW_INVENTORY")

    model = None if full_model is None else dict(full_model)
    if model is None:
        blockers.append("MISSING_FULL_MODEL_SAT")
    else:
        if model.get("model_kind") != "PRODUCTION_FULL_MODEL":
            blockers.append("FULL_MODEL_KIND_INVALID")
        claim = model.get("claim")
        if claim == "FULL_MODEL_INFEASIBLE":
            if model.get("solver_status") != "INFEASIBLE":
                raise ValueError(
                    "full-model INFEASIBLE claim contradicts solver status"
                )
            model["sufficient_core"] = _validate_sufficient_core(
                model.get("sufficient_core"), inventory=inventory
            )
            blockers.append("FULL_MODEL_INFEASIBLE")
        elif claim == "FULL_MODEL_SAT":
            if model.get("solver_status") not in _SAT_STATUSES:
                blockers.append("FULL_MODEL_NOT_SAT")
            candidate = model.get("candidate_positions_sha256")
            if type(candidate) is not str or _SHA256_RE.fullmatch(candidate) is None:
                blockers.append("FULL_MODEL_CANDIDATE_MISSING")
        else:
            blockers.append("FULL_MODEL_CLAIM_INVALID")
        if inventory is not None and (
            model.get("stage_input_sha256") != inventory["stage_input_sha256"]
            or model.get("model_proto_sha256") != inventory["model_proto_sha256"]
            or model.get("production_row_count") != inventory["row_count"]
            or model.get("production_row_ids_sha256")
            != inventory["row_ids_sha256"]
            or model.get("production_rows_sha256") != inventory["rows_sha256"]
        ):
            blockers.append("FULL_MODEL_INVENTORY_BINDING_MISMATCH")

    replay = None if full_hard_replay is None else dict(full_hard_replay)
    if replay is None:
        blockers.append("MISSING_FULL_HARD_REPLAY_PASS")
    else:
        if (
            replay.get("claim") != "FULL_HARD_REPLAY_PASS"
            or replay.get("status") != "PASS"
        ):
            blockers.append("FULL_HARD_REPLAY_NOT_PASS")
        if replay.get("failed_rows") != []:
            blockers.append("FULL_HARD_REPLAY_HAS_FAILURES")
        if replay.get("uncovered_production_row_ids") != []:
            blockers.append("FULL_HARD_REPLAY_HAS_UNCOVERED_ROWS")
        native_sha = replay.get("native_replay_inventory_sha256")
        if type(native_sha) is not str or _SHA256_RE.fullmatch(native_sha) is None:
            blockers.append("FULL_HARD_REPLAY_INVENTORY_MISSING")
        coverage_rows = replay.get("coverage_rows")
        if (
            type(coverage_rows) is not list
            or replay.get("coverage_map_sha256") != _data_sha256(coverage_rows)
        ):
            blockers.append("FULL_HARD_REPLAY_COVERAGE_INVALID")
        if inventory is not None:
            expected_ids = [row["row_id"] for row in inventory["rows"]]
            covered_ids = (
                [row.get("production_row_id") for row in coverage_rows]
                if type(coverage_rows) is list
                and all(isinstance(row, Mapping) for row in coverage_rows)
                else []
            )
            if (
                replay.get("checked_production_row_ids_sha256")
                != inventory["row_ids_sha256"]
                or covered_ids != expected_ids
            ):
                blockers.append("FULL_HARD_REPLAY_COVERAGE_INCOMPLETE")
        if model is not None and (
            replay.get("candidate_positions_sha256")
            != model.get("candidate_positions_sha256")
        ):
            blockers.append("FULL_MODEL_REPLAY_CANDIDATE_MISMATCH")

    adapter_model_fields = {
        "evidence_adapter", "expected_primitive_count", "materialized_primitive_count",
        "active_primitive_count", "variable_count", "assumption_indices",
        "primitive_bindings", "primitive_bindings_sha256", "stage_inputs",
        "candidate_positions", "captured_model_proto_sha256",
    }
    adapter_replay_fields = {
        "primitive_checks", "native_replay", "model_proto_sha256", "stage_input_sha256",
    }
    if (
        required_adapter
        or adapter_model_fields.intersection(model or {})
        or adapter_replay_fields.intersection(replay or {})
    ):
        blockers.extend(_production_adapter_blockers(model or {}, replay, inventory))

    if reduced_blocker is not None:
        blockers.append(reduced_blocker)
    payload: dict[str, object] = {
        "schema": _PROMOTION_SCHEMA,
        "status": "PROMOTABLE" if not blockers else "REJECT",
        "blockers": blockers,
        "evidence": {
            "reduced_model": reduced,
            "production_inventory": inventory,
            "full_model": model,
            "full_hard_replay": replay,
        },
    }
    payload["artifact_sha256"] = _data_sha256(payload)
    return payload


def _production_adapter_blockers(
    model: Mapping[str, object], replay: Mapping[str, object] | None,
    inventory: Mapping[str, object] | None,
) -> list[str]:
    """Recheck the concrete adapter payload as well as its summary hashes."""
    blockers = []
    if model.get("evidence_adapter") != "clay.fixed-outline-production-evidence/v1":
        return ["PRODUCTION_EVIDENCE_ADAPTER_UNSUPPORTED"]
    if replay is None or inventory is None:
        return ["PRODUCTION_ADAPTER_EVIDENCE_INCOMPLETE"]
    if (
        replay.get("model_proto_sha256") != model.get("model_proto_sha256")
        or replay.get("stage_input_sha256") != model.get("stage_input_sha256")
        or _data_sha256(model.get("stage_inputs")) != model.get("stage_input_sha256")
        or _data_sha256(model.get("candidate_positions")) != model.get("candidate_positions_sha256")
    ):
        blockers.append("PRODUCTION_ADAPTER_IDENTITY_MISMATCH")
    primitive_count = model.get("expected_primitive_count")
    variable_count = model.get("variable_count")
    assumptions = model.get("assumption_indices")
    if (
        type(primitive_count) is not int or primitive_count <= 0
        or primitive_count != model.get("materialized_primitive_count")
        or primitive_count != model.get("active_primitive_count")
        or type(variable_count) is not int or variable_count <= 0
        or type(assumptions) is not list
        or any(type(index) is not int or index < 0 for index in assumptions)
        or len(set(assumptions)) != len(assumptions)
    ):
        return [*blockers, "PRODUCTION_ADAPTER_PRIMITIVE_INVENTORY_INVALID"]
    expected_checks = {
        *(f"cp_primitive:{i}" for i in range(primitive_count)),
        *(f"cp_domain:{i}" for i in range(variable_count)),
        *(f"cp_assumption:{i}" for i in assumptions),
    }
    checks = replay.get("primitive_checks")
    if (
        type(checks) is not list
        or any(not isinstance(row, Mapping) for row in checks)
        or len(checks) != len(expected_checks)
        or {row.get("row_id") for row in checks} != expected_checks
        or any(row.get("status") != "PASS" for row in checks)
    ):
        blockers.append("PRODUCTION_ADAPTER_PRIMITIVE_REPLAY_INCOMPLETE")
    bindings = model.get("primitive_bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != {row["row_id"] for row in inventory["rows"]}
        or any(type(ids) is not list or not ids or any(type(name) is not str for name in ids)
               for ids in bindings.values())
        or _data_sha256(bindings) != model.get("primitive_bindings_sha256")
        or set().union(*(set(ids) for ids in bindings.values())) != expected_checks
    ):
        blockers.append("PRODUCTION_ADAPTER_PRIMITIVE_BINDINGS_INVALID")
    native = replay.get("native_replay")
    native_ids: set[str] = set()
    if (
        not isinstance(native, Mapping)
        or native.get("status") != "PASS" or native.get("failed_rows") != []
        or _data_sha256(native) != replay.get("native_replay_inventory_sha256")
        or type(native.get("inventory")) is not list
        or type(native.get("background_checks")) is not list
        or any(not isinstance(row, Mapping) for row in (*native["inventory"], *native["background_checks"]))
    ):
        blockers.append("PRODUCTION_ADAPTER_NATIVE_REPLAY_INVALID")
    else:
        native_ids = {row.get("row_id") for row in native["inventory"]}
        native_ids.update(row.get("row_id") for row in native["background_checks"] if row.get("status") == "PASS")
    coverage = replay.get("coverage_rows")
    if isinstance(bindings, Mapping) and type(coverage) is list:
        for row in coverage:
            if (
                not isinstance(row, Mapping)
                or row.get("primitive_check_ids") != bindings.get(row.get("production_row_id"))
                or type(row.get("native_replay_row_ids")) is not list
                or not row["native_replay_row_ids"]
                or any(name not in native_ids for name in row["native_replay_row_ids"])
            ):
                blockers.append("PRODUCTION_ADAPTER_REPLAY_COVERAGE_INCOMPLETE")
                break
    return blockers


def require_groundable_constraint_model(receipt: Mapping[str, object]) -> None:
    """Fail closed unless ``receipt`` replays as one promotable full model."""

    if not isinstance(receipt, Mapping):
        raise ConstraintModelPromotionError("constraint-model receipt is absent")
    value = dict(receipt)
    reported_sha = value.pop("artifact_sha256", None)
    if (
        value.get("schema") != _PROMOTION_SCHEMA
        or reported_sha != _data_sha256(value)
    ):
        raise ConstraintModelPromotionError("constraint-model receipt is invalid")
    blockers = value.get("blockers")
    if value.get("status") != "PROMOTABLE" or blockers != []:
        rendered = ",".join(blockers) if type(blockers) is list else "UNKNOWN"
        raise ConstraintModelPromotionError(
            f"constraint model is not groundable: {rendered}"
        )


__all__ = [
    "ConstraintModelPromotionError",
    "build_constraint_model_promotion_receipt",
    "build_production_constraint_inventory",
    "require_groundable_constraint_model",
]
