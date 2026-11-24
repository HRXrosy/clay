"""Typed immutable identity and admission checks for intent experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re


_SCHEMA = "clay.intent-experiment-spec/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
# G3 is the final-projection closure gate.  It cannot be honestly sealed until
# G6 has emitted the final GDS, so exact admission binds the cheap/static gates
# plus G4/G5 and re-closes G3 from the G6 artifact.  Receipts created by the
# older protocol may still carry an extra pre-exact G3 row.
_EXACT_PREREQUISITE_GATES = frozenset({"G0", "G1", "G2", "G4", "G5"})


class ExperimentStage(str, Enum):
    STATIC = "static"
    PLACEMENT = "placement"
    ROUTE_EMIT_MICRO = "route_emit_micro"
    EXACT = "exact"


def _require_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise TypeError(f"{field} must be an exact non-empty canonical string")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256 digest")
    return value


def _string_tuple(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be an exact tuple")
    if any(type(item) is not str or not item for item in value):
        raise TypeError(f"{field} must contain exact non-empty strings")
    if (not allow_empty and not value) or tuple(sorted(set(value))) != value:
        raise ValueError(f"{field} must contain sorted unique values")
    return value


def _digest_rows(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be an exact tuple")
    rows: list[tuple[str, str]] = []
    for raw in value:
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError(f"{field} rows must be exact 2-tuples")
        name, digest = raw
        _require_string(name, field=f"{field} name")
        _require_sha256(digest, field=f"{field} digest")
        rows.append((name, digest))
    if (not allow_empty and not rows) or tuple(sorted(set(rows))) != tuple(rows):
        raise ValueError(f"{field} must contain sorted unique rows")
    if len({name for name, _digest in rows}) != len(rows):
        raise ValueError(f"{field} must contain sorted unique names")
    return tuple(rows)


def _budget_rows(value: object) -> tuple[tuple[str, int], ...]:
    if type(value) is not tuple:
        raise TypeError("budgets_s must be an exact tuple")
    rows: list[tuple[str, int]] = []
    for raw in value:
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError("budgets_s rows must be exact 2-tuples")
        name, seconds = raw
        _require_string(name, field="budget name")
        if type(seconds) is not int or seconds <= 0:
            raise TypeError("budget seconds must be an exact positive integer")
        rows.append((name, seconds))
    if not rows or tuple(sorted(set(rows))) != tuple(rows) or len(
        {name for name, _seconds in rows}
    ) != len(rows):
        raise ValueError("budgets_s must contain sorted unique rows")
    return tuple(rows)


def _canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256(data: object) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IntentExperimentSpec:
    circuit_id: str
    stage: ExperimentStage
    source_path: str
    source_sha256: str
    basis_sha256: str
    acceptance_policy_sha256: str
    implementation_sha256: tuple[tuple[str, str], ...]
    pdk_snapshot_sha256: str
    pdk_installed_authority_sha256: str
    budgets_s: tuple[tuple[str, int], ...]
    output_root: str
    forbidden_side_effects: tuple[str, ...]
    prerequisite_gate_receipts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.circuit_id, field="circuit_id")
        if type(self.stage) is not ExperimentStage:
            raise TypeError("stage must be an exact ExperimentStage")
        _require_string(self.source_path, field="source_path")
        _require_sha256(self.source_sha256, field="source_sha256")
        _require_sha256(self.basis_sha256, field="basis_sha256")
        _require_sha256(
            self.acceptance_policy_sha256,
            field="acceptance_policy_sha256",
        )
        _digest_rows(
            self.implementation_sha256,
            field="implementation_sha256",
            allow_empty=False,
        )
        _require_sha256(self.pdk_snapshot_sha256, field="pdk_snapshot_sha256")
        _require_sha256(
            self.pdk_installed_authority_sha256,
            field="pdk_installed_authority_sha256",
        )
        _budget_rows(self.budgets_s)
        _require_string(self.output_root, field="output_root")
        _string_tuple(
            self.forbidden_side_effects,
            field="forbidden_side_effects",
            allow_empty=True,
        )
        receipts = _digest_rows(
            self.prerequisite_gate_receipts,
            field="prerequisite_gate_receipts",
            allow_empty=True,
        )
        if self.stage is ExperimentStage.EXACT:
            actual = {gate for gate, _digest in receipts}
            if not _EXACT_PREREQUISITE_GATES <= actual:
                missing = sorted(_EXACT_PREREQUISITE_GATES - actual)
                raise ValueError(
                    "exact experiment requires prerequisite gate receipts "
                    f"G0-G2 and G4-G5; missing={missing}"
                )

    @property
    def output_path(self) -> Path:
        return Path(self.output_root)

    def require_fresh_output_root(self) -> None:
        if self.output_path.exists():
            raise FileExistsError(
                f"refusing to reuse intent experiment output root: {self.output_root}"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "circuit_id": self.circuit_id,
            "stage": self.stage.value,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "basis_sha256": self.basis_sha256,
            "acceptance_policy_sha256": self.acceptance_policy_sha256,
            "implementation_sha256": [
                list(row) for row in self.implementation_sha256
            ],
            "pdk_snapshot_sha256": self.pdk_snapshot_sha256,
            "pdk_installed_authority_sha256": self.pdk_installed_authority_sha256,
            "budgets_s": [list(row) for row in self.budgets_s],
            "output_root": self.output_root,
            "forbidden_side_effects": list(self.forbidden_side_effects),
            "prerequisite_gate_receipts": [
                list(row) for row in self.prerequisite_gate_receipts
            ],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @property
    def cache_key(self) -> str:
        data = self.canonical_data()
        del data["output_root"]
        return _sha256(data)

    @classmethod
    def from_data(cls, data: object) -> "IntentExperimentSpec":
        if type(data) is not dict:
            raise TypeError("experiment replay data must be an exact dict")
        expected = frozenset(
            {
                "schema",
                "circuit_id",
                "stage",
                "source_path",
                "source_sha256",
                "basis_sha256",
                "acceptance_policy_sha256",
                "implementation_sha256",
                "pdk_snapshot_sha256",
                "pdk_installed_authority_sha256",
                "budgets_s",
                "output_root",
                "forbidden_side_effects",
                "prerequisite_gate_receipts",
            }
        )
        keys = frozenset(data)
        if keys != expected:
            raise ValueError(
                "experiment canonical schema fields disagree: "
                f"missing={sorted(expected - keys)}, unknown={sorted(keys - expected)}"
            )
        if data["schema"] != _SCHEMA:
            raise ValueError(f"experiment schema must be {_SCHEMA!r}")
        raw_stage = data["stage"]
        if type(raw_stage) is not str:
            raise TypeError("stage canonical data must be a string")
        try:
            stage = ExperimentStage(raw_stage)
        except ValueError as exc:
            raise ValueError(f"unknown experiment stage {raw_stage!r}") from exc

        def digest_rows(field: str) -> tuple[tuple[str, str], ...]:
            raw = data[field]
            if type(raw) is not list or any(
                type(row) is not list or len(row) != 2 for row in raw
            ):
                raise TypeError(f"{field} canonical data must be a list of 2-lists")
            return tuple((row[0], row[1]) for row in raw)  # type: ignore[misc]

        raw_budgets = data["budgets_s"]
        if type(raw_budgets) is not list or any(
            type(row) is not list or len(row) != 2 for row in raw_budgets
        ):
            raise TypeError("budgets_s canonical data must be a list of 2-lists")
        raw_side_effects = data["forbidden_side_effects"]
        if type(raw_side_effects) is not list or any(
            type(item) is not str for item in raw_side_effects
        ):
            raise TypeError("forbidden_side_effects canonical data must be a list")
        return cls(
            circuit_id=data["circuit_id"],  # type: ignore[arg-type]
            stage=stage,
            source_path=data["source_path"],  # type: ignore[arg-type]
            source_sha256=data["source_sha256"],  # type: ignore[arg-type]
            basis_sha256=data["basis_sha256"],  # type: ignore[arg-type]
            acceptance_policy_sha256=data[
                "acceptance_policy_sha256"
            ],  # type: ignore[arg-type]
            implementation_sha256=digest_rows("implementation_sha256"),
            pdk_snapshot_sha256=data["pdk_snapshot_sha256"],  # type: ignore[arg-type]
            pdk_installed_authority_sha256=data[
                "pdk_installed_authority_sha256"
            ],  # type: ignore[arg-type]
            budgets_s=tuple(
                (row[0], row[1]) for row in raw_budgets
            ),  # type: ignore[misc]
            output_root=data["output_root"],  # type: ignore[arg-type]
            forbidden_side_effects=tuple(raw_side_effects),
            prerequisite_gate_receipts=digest_rows(
                "prerequisite_gate_receipts"
            ),
        )


__all__ = ["ExperimentStage", "IntentExperimentSpec"]
