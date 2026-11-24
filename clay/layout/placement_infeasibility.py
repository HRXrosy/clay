"""Typed, lossless CP-SAT placement infeasibility diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WITNESS_STATUSES = frozenset(("PASS", "FAIL", "NOT_EVALUATED", "UNMEASURABLE"))


class PlacementInfeasibilityContractError(ValueError):
    """A solver core cannot be traced back to declared obligations."""


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PlacementInfeasibilityContractError(f"{label} must be one exact string")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ConstraintObligationTrace:
    """One assumption row bound to its named obligation and source contract."""

    row_id: str
    assumption_literal: str
    assumption_index: int
    obligation_id: str
    declaration_source: str
    witness_status: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.row_id, "core row id"),
            (self.assumption_literal, "assumption literal"),
            (self.obligation_id, "obligation id"),
            (self.declaration_source, "declaration source"),
            (self.witness_status, "witness status"),
        ):
            _text(value, label)
        if self.witness_status not in _WITNESS_STATUSES:
            raise PlacementInfeasibilityContractError(
                "witness status is not one typed state"
            )
        if type(self.assumption_index) is not int or self.assumption_index < 0:
            raise PlacementInfeasibilityContractError(
                "assumption index must be one non-negative exact integer"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "assumption_literal": self.assumption_literal,
            "assumption_index": self.assumption_index,
            "obligation_id": self.obligation_id,
            "declaration_source": self.declaration_source,
            "witness_status": self.witness_status,
        }

    @classmethod
    def from_data(cls, value: object) -> "ConstraintObligationTrace":
        fields = {
            "row_id",
            "assumption_literal",
            "assumption_index",
            "obligation_id",
            "declaration_source",
            "witness_status",
        }
        if type(value) is not dict or set(value) != fields:
            raise PlacementInfeasibilityContractError(
                "sufficient core row schema differs"
            )
        result = cls(**value)
        if result.canonical_data() != value:
            raise PlacementInfeasibilityContractError(
                "sufficient core row is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class SufficientCoreDiagnostic:
    """A sufficient CP core whose every row remains mechanically attributable."""

    solver_status: str
    stage: str
    family_core: tuple[str, ...]
    row_ids: tuple[str, ...]
    rows: tuple[ConstraintObligationTrace, ...]
    core_sha256: str

    def __post_init__(self) -> None:
        _text(self.solver_status, "solver status")
        _text(self.stage, "solver stage")
        if self.solver_status != "INFEASIBLE":
            raise PlacementInfeasibilityContractError(
                "sufficient core requires INFEASIBLE solver status"
            )
        if (
            type(self.family_core) is not tuple
            or not self.family_core
            or self.family_core != tuple(sorted(set(self.family_core)))
            or any(type(item) is not str or not item for item in self.family_core)
        ):
            raise PlacementInfeasibilityContractError(
                "sufficient core family mapping is not exact"
            )
        if (
            type(self.row_ids) is not tuple
            or not self.row_ids
            or self.row_ids != tuple(sorted(set(self.row_ids)))
            or any(type(item) is not str or not item for item in self.row_ids)
        ):
            raise PlacementInfeasibilityContractError(
                "sufficient core row ids are not exact"
            )
        if type(self.rows) is not tuple or not self.rows:
            raise PlacementInfeasibilityContractError(
                "sufficient core row mapping cannot be empty"
            )
        if any(type(row) is not ConstraintObligationTrace for row in self.rows):
            raise PlacementInfeasibilityContractError(
                "sufficient core row mapping is not typed"
            )
        if tuple(row.row_id for row in self.rows) != self.row_ids:
            raise PlacementInfeasibilityContractError(
                "sufficient core row mapping differs from solver row ids"
            )
        if len({row.assumption_index for row in self.rows}) != len(self.rows):
            raise PlacementInfeasibilityContractError(
                "sufficient core assumption indices are not unique"
            )
        if len({row.assumption_literal for row in self.rows}) != len(self.rows):
            raise PlacementInfeasibilityContractError(
                "sufficient core assumption literals are not unique"
            )
        if tuple(sorted({row.row_id.split(":", 1)[0] for row in self.rows})) != (
            self.family_core
        ):
            raise PlacementInfeasibilityContractError(
                "sufficient core row families differ from family core"
            )
        if type(self.core_sha256) is not str or _SHA256_RE.fullmatch(
            self.core_sha256
        ) is None:
            raise PlacementInfeasibilityContractError(
                "sufficient core digest is malformed"
            )
        expected_digest = hashlib.sha256(repr(self.row_ids).encode()).hexdigest()
        if self.core_sha256 != expected_digest:
            raise PlacementInfeasibilityContractError(
                "sufficient core digest differs from solver row ids"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.placement-sufficient-core/v2",
            "solver_status": self.solver_status,
            "stage": self.stage,
            "family_core": list(self.family_core),
            "row_ids": list(self.row_ids),
            "rows": [row.canonical_data() for row in self.rows],
            "core_sha256": self.core_sha256,
        }

    @classmethod
    def from_data(cls, value: object) -> "SufficientCoreDiagnostic":
        fields = {
            "schema",
            "solver_status",
            "stage",
            "family_core",
            "row_ids",
            "rows",
            "core_sha256",
        }
        if type(value) is not dict or set(value) != fields:
            raise PlacementInfeasibilityContractError(
                "sufficient core schema differs"
            )
        if value["schema"] != "clay.placement-sufficient-core/v2":
            raise PlacementInfeasibilityContractError(
                "sufficient core schema version differs"
            )
        result = cls(
            solver_status=value["solver_status"],
            stage=value["stage"],
            family_core=tuple(value["family_core"]),
            row_ids=tuple(value["row_ids"]),
            rows=tuple(
                ConstraintObligationTrace.from_data(row) for row in value["rows"]
            ),
            core_sha256=value["core_sha256"],
        )
        if result.canonical_data() != value:
            raise PlacementInfeasibilityContractError(
                "sufficient core is not canonical"
            )
        return result


class PlacementInfeasibilityError(RuntimeError):
    """Typed stop retaining a sufficient core through placement wrappers."""

    def __init__(self, message: str, diagnostic: SufficientCoreDiagnostic) -> None:
        _text(message, "placement infeasibility message")
        if type(diagnostic) is not SufficientCoreDiagnostic:
            raise PlacementInfeasibilityContractError(
                "placement infeasibility lacks one typed sufficient core"
            )
        self.diagnostic = diagnostic
        super().__init__(message)


__all__ = [
    "ConstraintObligationTrace",
    "PlacementInfeasibilityContractError",
    "PlacementInfeasibilityError",
    "SufficientCoreDiagnostic",
]
