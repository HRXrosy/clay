"""Typed provenance and canonical receipts for governed intent-basis search.

This module deliberately contains no placement or routing behavior.  It is the
information-flow boundary between design/reference observations and intent
hypotheses that physical consumers may later lower.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any


_POLICY_SCHEMA = "clay.intent-acceptance-policy/v1"
_ATOM_SCHEMA = "clay.intent-atom/v1"
_BASIS_SCHEMA = "clay.intent-basis/v1"
_CONSUMPTION_SCHEMA = "clay.intent-consumption-certificate/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REFERENCE_COORDINATE_KEYS = frozenset(
    {
        "x",
        "y",
        "xy",
        "bbox",
        "centroid",
        "origin",
        "offset",
        "offset_nm",
        "coordinates",
    }
)
_REFERENCE_PAYLOAD_SCHEMAS = {
    "group_floorplan.cell": frozenset({"column", "group", "row", "tier"}),
}
_REFERENCE_REQUIRED_PAYLOAD_FIELDS = {
    "group_floorplan.cell": frozenset({"column", "group", "tier"}),
}


class ClaimScope(str, Enum):
    REFERENCE_CONDITIONED_SINGLE_CIRCUIT = (
        "reference_conditioned_single_circuit"
    )
    REFERENCE_FREE_SINGLE_CIRCUIT = "reference_free_single_circuit"
    HELD_OUT_GENERALIZATION = "held_out_generalization"


class InformationOrigin(str, Enum):
    DESIGN_INPUT = "design_input"
    TARGET_REFERENCE_OBSERVATION = "target_reference_observation"
    TRAINING_REFERENCE_OBSERVATION = "training_reference_observation"
    GENERATED_REALIZATION = "generated_realization"
    EVALUATION = "evaluation"


class MinimalityScope(str, Enum):
    GLOBAL = "global"
    DELETION_MINIMAL = "deletion_minimal"
    BOUNDED_SEARCH = "bounded_search"
    UNPROVEN = "unproven"


def _canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256(data: object) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _require_exact_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{field} must be an exact non-empty string")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256 digest")
    return value


def _require_exact_tuple(value: object, *, field: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be an exact tuple")
    return value


def _require_sorted_unique_strings(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    rows = _require_exact_tuple(value, field=field)
    if any(type(item) is not str or not item for item in rows):
        raise TypeError(f"{field} must contain exact non-empty strings")
    if (not allow_empty and not rows) or tuple(sorted(set(rows))) != rows:
        raise ValueError(f"{field} must contain sorted unique values")
    return rows


def _require_schema_fields(
    data: object,
    *,
    expected: frozenset[str],
    schema: str,
) -> dict[str, object]:
    if type(data) is not dict:
        raise TypeError("canonical replay data must be an exact dict")
    keys = frozenset(data)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise ValueError(
            "canonical schema fields disagree: "
            f"missing={missing}, unknown={unknown}"
        )
    if data.get("schema") != schema:
        raise ValueError(f"canonical schema must be {schema!r}")
    return data


def _json_value(value: object, *, path: str = "payload") -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is list:
        return [_json_value(item, path=f"{path}[]") for item in value]
    if type(value) is dict:
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be exact strings")
            normalized[key] = _json_value(item, path=f"{path}.{key}")
        return normalized
    raise TypeError(
        f"{path} must contain only exact JSON values; got {type(value).__name__}"
    )


def _find_reference_coordinate_key(value: object) -> str | None:
    if type(value) is dict:
        for key, item in value.items():
            if key in _REFERENCE_COORDINATE_KEYS:
                return key
            nested = _find_reference_coordinate_key(item)
            if nested is not None:
                return nested
    elif type(value) is list:
        for item in value:
            nested = _find_reference_coordinate_key(item)
            if nested is not None:
                return nested
    return None


def _validate_reference_payload(kind: str, payload: dict[str, object]) -> None:
    allowed = _REFERENCE_PAYLOAD_SCHEMAS.get(kind)
    if allowed is None:
        raise ValueError(
            f"reference-derived intent atom kind {kind!r} has no payload schema"
        )
    unknown = frozenset(payload) - allowed
    if unknown:
        raise ValueError(
            "reference-derived intent atom contains fields outside its payload "
            f"schema: {sorted(unknown)}"
        )
    missing = _REFERENCE_REQUIRED_PAYLOAD_FIELDS[kind] - frozenset(payload)
    if missing:
        raise ValueError(
            "reference-derived intent atom lacks required payload fields: "
            f"{sorted(missing)}"
        )
    group = payload.get("group")
    if type(group) is not str or not group:
        raise TypeError("group_floorplan.cell group must be an exact non-empty string")
    for field in ("column", "row", "tier"):
        if field not in payload:
            continue
        value = payload[field]
        if type(value) is not int or value < 0:
            raise TypeError(
                f"group_floorplan.cell {field} must be an exact non-negative integer"
            )


def _list_of_strings(value: object, *, field: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise TypeError(f"{field} canonical data must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class IntentAcceptancePolicy:
    circuit_id: str
    source_sha256: str
    pdk_snapshot_sha256: str
    pdk_installed_authority_sha256: str
    fixed_outline_width_nm: int
    maximum_area_nm2: int
    stage_budgets_s: tuple[tuple[str, int], ...]
    required_exact_gates: tuple[str, ...]
    forbidden_side_effects: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_exact_string(self.circuit_id, field="circuit_id")
        _require_sha256(self.source_sha256, field="source_sha256")
        _require_sha256(self.pdk_snapshot_sha256, field="pdk_snapshot_sha256")
        _require_sha256(
            self.pdk_installed_authority_sha256,
            field="pdk_installed_authority_sha256",
        )
        if type(self.fixed_outline_width_nm) is not int or self.fixed_outline_width_nm <= 0:
            raise TypeError("fixed_outline_width_nm must be an exact positive integer")
        if type(self.maximum_area_nm2) is not int or self.maximum_area_nm2 <= 0:
            raise TypeError("maximum_area_nm2 must be an exact positive integer")

        budget_rows = _require_exact_tuple(
            self.stage_budgets_s,
            field="stage_budgets_s",
        )
        normalized_budget_rows: list[tuple[str, int]] = []
        for row in budget_rows:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("stage_budgets_s rows must be exact 2-tuples")
            stage, seconds = row
            _require_exact_string(stage, field="stage budget name")
            if type(seconds) is not int or seconds <= 0:
                raise TypeError("stage budget seconds must be an exact positive integer")
            normalized_budget_rows.append((stage, seconds))
        if tuple(sorted(set(normalized_budget_rows))) != tuple(normalized_budget_rows):
            raise ValueError("stage_budgets_s must contain sorted unique rows")

        _require_sorted_unique_strings(
            self.required_exact_gates,
            field="required_exact_gates",
            allow_empty=False,
        )
        _require_sorted_unique_strings(
            self.forbidden_side_effects,
            field="forbidden_side_effects",
            allow_empty=True,
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _POLICY_SCHEMA,
            "circuit_id": self.circuit_id,
            "source_sha256": self.source_sha256,
            "pdk_snapshot_sha256": self.pdk_snapshot_sha256,
            "pdk_installed_authority_sha256": self.pdk_installed_authority_sha256,
            "fixed_outline_width_nm": self.fixed_outline_width_nm,
            "maximum_area_nm2": self.maximum_area_nm2,
            "stage_budgets_s": [list(row) for row in self.stage_budgets_s],
            "required_exact_gates": list(self.required_exact_gates),
            "forbidden_side_effects": list(self.forbidden_side_effects),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "IntentAcceptancePolicy":
        row = _require_schema_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "circuit_id",
                    "source_sha256",
                    "pdk_snapshot_sha256",
                    "pdk_installed_authority_sha256",
                    "fixed_outline_width_nm",
                    "maximum_area_nm2",
                    "stage_budgets_s",
                    "required_exact_gates",
                    "forbidden_side_effects",
                }
            ),
            schema=_POLICY_SCHEMA,
        )
        raw_budgets = row["stage_budgets_s"]
        if type(raw_budgets) is not list:
            raise TypeError("stage_budgets_s canonical data must be a list")
        budgets: list[tuple[str, int]] = []
        for budget in raw_budgets:
            if type(budget) is not list or len(budget) != 2:
                raise TypeError("stage_budgets_s canonical rows must be 2-lists")
            budgets.append((budget[0], budget[1]))  # type: ignore[arg-type]
        return cls(
            circuit_id=row["circuit_id"],  # type: ignore[arg-type]
            source_sha256=row["source_sha256"],  # type: ignore[arg-type]
            pdk_snapshot_sha256=row["pdk_snapshot_sha256"],  # type: ignore[arg-type]
            pdk_installed_authority_sha256=row[
                "pdk_installed_authority_sha256"
            ],  # type: ignore[arg-type]
            fixed_outline_width_nm=row["fixed_outline_width_nm"],  # type: ignore[arg-type]
            maximum_area_nm2=row["maximum_area_nm2"],  # type: ignore[arg-type]
            stage_budgets_s=tuple(budgets),
            required_exact_gates=_list_of_strings(
                row["required_exact_gates"],
                field="required_exact_gates",
            ),
            forbidden_side_effects=_list_of_strings(
                row["forbidden_side_effects"],
                field="forbidden_side_effects",
            ),
        )


@dataclass(frozen=True, order=True)
class IntentAtom:
    atom_id: str
    kind: str
    payload_json: str
    origin: InformationOrigin
    cost_bits: int
    required_stages: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_exact_string(self.atom_id, field="atom_id")
        _require_exact_string(self.kind, field="kind")
        if type(self.payload_json) is not str:
            raise TypeError("payload_json must be an exact string")
        try:
            payload = json.loads(self.payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("payload_json must contain canonical JSON") from exc
        payload = _json_value(payload)
        if type(payload) is not dict:
            raise TypeError("intent atom payload must be a JSON object")
        if _canonical_json(payload) != self.payload_json:
            raise ValueError("payload_json must contain canonical JSON")
        if type(self.origin) is not InformationOrigin:
            raise TypeError("origin must be an exact InformationOrigin")
        if self.origin in {
            InformationOrigin.GENERATED_REALIZATION,
            InformationOrigin.EVALUATION,
        }:
            raise ValueError(
                f"{self.origin.value} facts cannot be promoted to intent atoms"
            )
        if self.origin in {
            InformationOrigin.TARGET_REFERENCE_OBSERVATION,
            InformationOrigin.TRAINING_REFERENCE_OBSERVATION,
        }:
            key = _find_reference_coordinate_key(payload)
            if key is not None:
                raise ValueError(
                    "reference-derived intent atom contains forbidden coordinate "
                    f"key {key!r}"
                )
            _validate_reference_payload(self.kind, payload)
        if type(self.cost_bits) is not int or self.cost_bits < 0:
            raise TypeError("cost_bits must be an exact non-negative integer")
        _require_sorted_unique_strings(
            self.required_stages,
            field="required_stages",
            allow_empty=False,
        )

    @classmethod
    def from_payload(
        cls,
        *,
        atom_id: str,
        kind: str,
        payload: dict[str, object],
        origin: InformationOrigin,
        cost_bits: int,
        required_stages: tuple[str, ...],
    ) -> "IntentAtom":
        if type(payload) is not dict:
            raise TypeError("payload must be an exact dict")
        normalized = _json_value(payload)
        return cls(
            atom_id=atom_id,
            kind=kind,
            payload_json=_canonical_json(normalized),
            origin=origin,
            cost_bits=cost_bits,
            required_stages=required_stages,
        )

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.payload_json)
        if type(value) is not dict:  # guarded by from_payload for public creation
            raise ValueError("intent atom payload must be a JSON object")
        return value

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _ATOM_SCHEMA,
            "atom_id": self.atom_id,
            "kind": self.kind,
            "payload": self.payload,
            "origin": self.origin.value,
            "cost_bits": self.cost_bits,
            "required_stages": list(self.required_stages),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "IntentAtom":
        row = _require_schema_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "atom_id",
                    "kind",
                    "payload",
                    "origin",
                    "cost_bits",
                    "required_stages",
                }
            ),
            schema=_ATOM_SCHEMA,
        )
        raw_origin = row["origin"]
        if type(raw_origin) is not str:
            raise TypeError("origin canonical data must be a string")
        try:
            origin = InformationOrigin(raw_origin)
        except ValueError as exc:
            raise ValueError(f"unknown information origin {raw_origin!r}") from exc
        payload = row["payload"]
        if type(payload) is not dict:
            raise TypeError("payload canonical data must be an exact dict")
        return cls.from_payload(
            atom_id=row["atom_id"],  # type: ignore[arg-type]
            kind=row["kind"],  # type: ignore[arg-type]
            payload=payload,
            origin=origin,
            cost_bits=row["cost_bits"],  # type: ignore[arg-type]
            required_stages=_list_of_strings(
                row["required_stages"],
                field="required_stages",
            ),
        )


@dataclass(frozen=True)
class IntentBasis:
    circuit_id: str
    claim_scope: ClaimScope
    base_candidate_sha256: str
    acceptance_policy_sha256: str
    atoms: tuple[IntentAtom, ...]
    target_reference_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_exact_string(self.circuit_id, field="circuit_id")
        if type(self.claim_scope) is not ClaimScope:
            raise TypeError("claim_scope must be an exact ClaimScope")
        _require_sha256(
            self.base_candidate_sha256,
            field="base_candidate_sha256",
        )
        _require_sha256(
            self.acceptance_policy_sha256,
            field="acceptance_policy_sha256",
        )
        rows = _require_exact_tuple(self.atoms, field="atoms")
        if any(type(atom) is not IntentAtom for atom in rows):
            raise TypeError("atoms must contain exact IntentAtom values")
        atom_ids = tuple(atom.atom_id for atom in rows)
        if tuple(sorted(set(atom_ids))) != atom_ids:
            raise ValueError("atoms must contain sorted unique atom IDs")

        origins = frozenset(atom.origin for atom in rows)
        if (
            self.claim_scope is ClaimScope.HELD_OUT_GENERALIZATION
            and InformationOrigin.TARGET_REFERENCE_OBSERVATION in origins
        ):
            raise ValueError(
                "held-out basis cannot contain a target-reference observation"
            )
        if self.claim_scope is ClaimScope.REFERENCE_FREE_SINGLE_CIRCUIT and (
            InformationOrigin.TARGET_REFERENCE_OBSERVATION in origins
            or InformationOrigin.TRAINING_REFERENCE_OBSERVATION in origins
        ):
            raise ValueError(
                "reference-free basis cannot contain a reference observation"
            )

        receipt = self.target_reference_receipt_sha256
        if receipt is not None:
            _require_sha256(receipt, field="target_reference_receipt_sha256")
            if self.claim_scope is not ClaimScope.REFERENCE_CONDITIONED_SINGLE_CIRCUIT:
                raise ValueError(
                    "target-reference receipt is allowed only for a "
                    "reference-conditioned basis"
                )
        if (
            InformationOrigin.TARGET_REFERENCE_OBSERVATION in origins
            and receipt is None
        ):
            raise ValueError(
                "target-reference observations require a bound reference receipt"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _BASIS_SCHEMA,
            "circuit_id": self.circuit_id,
            "claim_scope": self.claim_scope.value,
            "base_candidate_sha256": self.base_candidate_sha256,
            "acceptance_policy_sha256": self.acceptance_policy_sha256,
            "target_reference_receipt_sha256": self.target_reference_receipt_sha256,
            "atoms": [atom.canonical_data() for atom in self.atoms],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "IntentBasis":
        row = _require_schema_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "circuit_id",
                    "claim_scope",
                    "base_candidate_sha256",
                    "acceptance_policy_sha256",
                    "target_reference_receipt_sha256",
                    "atoms",
                }
            ),
            schema=_BASIS_SCHEMA,
        )
        raw_scope = row["claim_scope"]
        if type(raw_scope) is not str:
            raise TypeError("claim_scope canonical data must be a string")
        try:
            scope = ClaimScope(raw_scope)
        except ValueError as exc:
            raise ValueError(f"unknown claim scope {raw_scope!r}") from exc
        raw_atoms = row["atoms"]
        if type(raw_atoms) is not list:
            raise TypeError("atoms canonical data must be a list")
        receipt = row["target_reference_receipt_sha256"]
        if receipt is not None and type(receipt) is not str:
            raise TypeError(
                "target_reference_receipt_sha256 must be null or a string"
            )
        return cls(
            circuit_id=row["circuit_id"],  # type: ignore[arg-type]
            claim_scope=scope,
            base_candidate_sha256=row["base_candidate_sha256"],  # type: ignore[arg-type]
            acceptance_policy_sha256=row[
                "acceptance_policy_sha256"
            ],  # type: ignore[arg-type]
            target_reference_receipt_sha256=receipt,
            atoms=tuple(IntentAtom.from_data(atom) for atom in raw_atoms),
        )


@dataclass(frozen=True, order=True)
class IntentConsumptionCertificate:
    stage: str
    basis_sha256: str
    consumed_atom_ids: tuple[str, ...]
    normalized_projection_sha256: str

    def __post_init__(self) -> None:
        _require_exact_string(self.stage, field="stage")
        _require_sha256(self.basis_sha256, field="basis_sha256")
        _require_sorted_unique_strings(
            self.consumed_atom_ids,
            field="consumed_atom_ids",
            allow_empty=True,
        )
        _require_sha256(
            self.normalized_projection_sha256,
            field="normalized_projection_sha256",
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _CONSUMPTION_SCHEMA,
            "stage": self.stage,
            "basis_sha256": self.basis_sha256,
            "consumed_atom_ids": list(self.consumed_atom_ids),
            "normalized_projection_sha256": self.normalized_projection_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "IntentConsumptionCertificate":
        row = _require_schema_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "stage",
                    "basis_sha256",
                    "consumed_atom_ids",
                    "normalized_projection_sha256",
                }
            ),
            schema=_CONSUMPTION_SCHEMA,
        )
        return cls(
            stage=row["stage"],  # type: ignore[arg-type]
            basis_sha256=row["basis_sha256"],  # type: ignore[arg-type]
            consumed_atom_ids=_list_of_strings(
                row["consumed_atom_ids"],
                field="consumed_atom_ids",
            ),
            normalized_projection_sha256=row[
                "normalized_projection_sha256"
            ],  # type: ignore[arg-type]
        )
