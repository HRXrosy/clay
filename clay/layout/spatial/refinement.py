"""Strict round-trip refinement admission for Abstract Spatial IR.

This is a soundness/refinement gate, not a claimed Galois connection.  It
separates positive obligations, forbidden structure, exact typed overlays, and
continuous admission bands.  Diagnostics are intentionally absent from the
decision path.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from clay.layout.spatial.abstract_ir import (
    AbstractSpatialIR,
    HardInvariant,
    SpatialFactKind,
    SpatialIrSchemaError,
    _canonical_decimal,
    _canonical_text,
    _decimal_text,
    _exact_tuple,
    _reject_coordinate_key,
)


class SpatialViolationCode(str, Enum):
    NODE_SET_MISMATCH = "node_set_mismatch"
    MISSING_REQUIRED_ORDER = "missing_required_order"
    FORBIDDEN_ORDER_PRESENT = "forbidden_order_present"
    MISSING_REQUIRED_FACT = "missing_required_fact"
    FORBIDDEN_FACT_PRESENT = "forbidden_fact_present"
    EXTRA_EXACT_FACT = "extra_exact_fact"
    MISSING_METRIC = "missing_metric"
    METRIC_UNIT_MISMATCH = "metric_unit_mismatch"
    BAND_VIOLATION = "band_violation"


@dataclass(frozen=True, slots=True)
class AdmissibleBand:
    """Inclusive authoritative interval for one coordinate-free observation."""

    metric_id: str
    lower: Decimal | None
    upper: Decimal | None
    unit: str

    def __post_init__(self) -> None:
        metric_id = _canonical_text(self.metric_id, field="band metric_id")
        _reject_coordinate_key(metric_id, field=f"band metric {metric_id}")
        unit = _canonical_text(self.unit, field="band unit")
        lower = (
            None
            if self.lower is None
            else _canonical_decimal(self.lower, field="band lower")
        )
        upper = (
            None
            if self.upper is None
            else _canonical_decimal(self.upper, field="band upper")
        )
        if lower is None and upper is None:
            raise SpatialIrSchemaError("admissible band requires at least one bound")
        if lower is not None and upper is not None and lower > upper:
            raise SpatialIrSchemaError("admissible band lower exceeds upper")
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "unit", unit)

    def canonical_data(self) -> dict[str, str | None]:
        return {
            "metric_id": self.metric_id,
            "lower": None if self.lower is None else _decimal_text(self.lower),
            "upper": None if self.upper is None else _decimal_text(self.upper),
            "unit": self.unit,
        }


def _canonical_relations(
    value: object,
    *,
    field: str,
    nodes: frozenset[str],
) -> tuple[tuple[str, str], ...]:
    rows = _exact_tuple(value, field=field)
    result: set[tuple[str, str]] = set()
    for row in rows:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError(f"{field} must contain exact two-item tuples")
        before, after = row
        if type(before) is not str or type(after) is not str:
            raise TypeError(f"{field} endpoints must be exact strings")
        if before not in nodes or after not in nodes:
            raise SpatialIrSchemaError(f"{field} names an unknown node")
        if before == after:
            raise SpatialIrSchemaError(f"{field} cannot contain a self relation")
        result.add((before, after))
    return tuple(sorted(result))


def _canonical_facts(
    value: object,
    *,
    field: str,
    nodes: frozenset[str],
) -> tuple[HardInvariant, ...]:
    rows = _exact_tuple(value, field=field)
    if any(type(fact) is not HardInvariant for fact in rows):
        raise TypeError(f"{field} must contain exact HardInvariant values")
    if any(
        binding.node not in nodes
        for fact in rows
        for binding in fact.bindings  # type: ignore[union-attr]
    ):
        raise SpatialIrSchemaError(f"{field} contains an unknown ASIR node")
    signatures = [fact.semantic_signature for fact in rows]  # type: ignore[union-attr]
    if len(set(signatures)) != len(signatures):
        raise SpatialIrSchemaError(f"{field} semantic signatures must be unique")
    return tuple(sorted(rows, key=lambda fact: fact.semantic_signature))  # type: ignore[union-attr]


@dataclass(frozen=True, slots=True)
class SpatialRequirementSet:
    """Closed strict-gate requirements bound to one reference ASIR identity."""

    reference_asir_sha256: str
    expected_nodes: tuple[str, ...]
    required_horizontal: tuple[tuple[str, str], ...] = ()
    forbidden_horizontal: tuple[tuple[str, str], ...] = ()
    required_vertical: tuple[tuple[str, str], ...] = ()
    forbidden_vertical: tuple[tuple[str, str], ...] = ()
    required_facts: tuple[HardInvariant, ...] = ()
    forbidden_facts: tuple[HardInvariant, ...] = ()
    exact_fact_kinds: tuple[SpatialFactKind, ...] = ()
    bands: tuple[AdmissibleBand, ...] = ()

    def __post_init__(self) -> None:
        if type(self.reference_asir_sha256) is not str or len(self.reference_asir_sha256) != 64:
            raise SpatialIrSchemaError("reference_asir_sha256 must be SHA-256")
        raw_nodes = _exact_tuple(self.expected_nodes, field="expected_nodes")
        if any(type(node) is not str or not node for node in raw_nodes):
            raise SpatialIrSchemaError("expected_nodes must contain canonical strings")
        if len(set(raw_nodes)) != len(raw_nodes):
            raise SpatialIrSchemaError("expected_nodes must be unique")
        nodes = tuple(sorted(raw_nodes))  # type: ignore[arg-type]
        known = frozenset(nodes)

        object.__setattr__(self, "expected_nodes", nodes)
        for field in (
            "required_horizontal",
            "forbidden_horizontal",
            "required_vertical",
            "forbidden_vertical",
        ):
            object.__setattr__(
                self,
                field,
                _canonical_relations(getattr(self, field), field=field, nodes=known),
            )
        for field in ("required_facts", "forbidden_facts"):
            object.__setattr__(
                self,
                field,
                _canonical_facts(getattr(self, field), field=field, nodes=known),
            )

        raw_kinds = _exact_tuple(self.exact_fact_kinds, field="exact_fact_kinds")
        if any(type(kind) is not SpatialFactKind for kind in raw_kinds):
            raise TypeError("exact_fact_kinds must contain exact SpatialFactKind values")
        object.__setattr__(
            self,
            "exact_fact_kinds",
            tuple(sorted(set(raw_kinds), key=lambda kind: kind.value)),  # type: ignore[union-attr]
        )

        raw_bands = _exact_tuple(self.bands, field="bands")
        if any(type(band) is not AdmissibleBand for band in raw_bands):
            raise TypeError("bands must contain exact AdmissibleBand values")
        if len({band.metric_id for band in raw_bands}) != len(raw_bands):  # type: ignore[union-attr]
            raise SpatialIrSchemaError("band metric ids must be unique")
        object.__setattr__(
            self,
            "bands",
            tuple(sorted(raw_bands, key=lambda band: band.metric_id)),  # type: ignore[union-attr]
        )

    @classmethod
    def strict_from_reference(
        cls,
        reference: AbstractSpatialIR,
        *,
        bands: tuple[AdmissibleBand, ...] = (),
    ) -> "SpatialRequirementSet":
        if type(reference) is not AbstractSpatialIR:
            raise TypeError("reference must be an exact AbstractSpatialIR")
        all_pairs = {
            (before, after)
            for before in reference.nodes
            for after in reference.nodes
            if before != after
        }
        required_h = set(reference.horizontal.closure_edges())
        required_v = set(reference.vertical.closure_edges())
        return cls(
            reference_asir_sha256=reference.sha256,
            expected_nodes=reference.nodes,
            required_horizontal=tuple(sorted(required_h)),
            forbidden_horizontal=tuple(sorted(all_pairs - required_h)),
            required_vertical=tuple(sorted(required_v)),
            forbidden_vertical=tuple(sorted(all_pairs - required_v)),
            required_facts=reference.hard_facts,
            exact_fact_kinds=tuple(SpatialFactKind),
            bands=bands,
        )


@dataclass(frozen=True, slots=True)
class SpatialViolation:
    code: SpatialViolationCode
    subject: str
    detail: str

    def __post_init__(self) -> None:
        if type(self.code) is not SpatialViolationCode:
            raise TypeError("code must be an exact SpatialViolationCode")
        _canonical_text(self.subject, field="violation subject")
        _canonical_text(self.detail, field="violation detail")

    def canonical_data(self) -> dict[str, str]:
        return {"code": self.code.value, "subject": self.subject, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class SpatialGateResult:
    passed: bool
    violations: tuple[SpatialViolation, ...]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("passed must be an exact bool")
        rows = _exact_tuple(self.violations, field="violations")
        if any(type(row) is not SpatialViolation for row in rows):
            raise TypeError("violations must contain exact SpatialViolation values")
        canonical = tuple(
            sorted(rows, key=lambda row: (row.code.value, row.subject, row.detail))  # type: ignore[union-attr]
        )
        if self.passed != (not canonical):
            raise SpatialIrSchemaError("passed must equal the absence of violations")
        object.__setattr__(self, "violations", canonical)

    def canonical_data(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "violations": [row.canonical_data() for row in self.violations],
        }


def _fact_subject(fact: HardInvariant) -> str:
    roles = ",".join(f"{binding.role}={binding.node}" for binding in fact.bindings)
    return f"{fact.kind.value}:{roles}"


def evaluate_spatial_requirements(
    requirements: SpatialRequirementSet,
    candidate: AbstractSpatialIR,
) -> SpatialGateResult:
    """Evaluate all strict requirements without first-error short circuiting."""

    if type(requirements) is not SpatialRequirementSet:
        raise TypeError("requirements must be an exact SpatialRequirementSet")
    if type(candidate) is not AbstractSpatialIR:
        raise TypeError("candidate must be an exact AbstractSpatialIR")

    violations: list[SpatialViolation] = []
    if candidate.nodes != requirements.expected_nodes:
        violations.append(
            SpatialViolation(
                SpatialViolationCode.NODE_SET_MISMATCH,
                "nodes",
                f"expected={requirements.expected_nodes!r}; candidate={candidate.nodes!r}",
            )
        )

    candidate_h = set(candidate.horizontal.closure_edges())
    candidate_v = set(candidate.vertical.closure_edges())
    for axis, present, required, forbidden in (
        (
            "horizontal",
            candidate_h,
            requirements.required_horizontal,
            requirements.forbidden_horizontal,
        ),
        (
            "vertical",
            candidate_v,
            requirements.required_vertical,
            requirements.forbidden_vertical,
        ),
    ):
        for before, after in sorted(set(required) - present):
            violations.append(
                SpatialViolation(
                    SpatialViolationCode.MISSING_REQUIRED_ORDER,
                    f"{axis}:{before}->{after}",
                    "required strict order is absent",
                )
            )
        for before, after in sorted(set(forbidden) & present):
            violations.append(
                SpatialViolation(
                    SpatialViolationCode.FORBIDDEN_ORDER_PRESENT,
                    f"{axis}:{before}->{after}",
                    "candidate contains a forbidden strict order",
                )
            )

    candidate_by_signature = {
        fact.semantic_signature: fact for fact in candidate.hard_facts
    }
    required_by_signature = {
        fact.semantic_signature: fact for fact in requirements.required_facts
    }
    forbidden_by_signature = {
        fact.semantic_signature: fact for fact in requirements.forbidden_facts
    }
    for signature, fact in sorted(
        required_by_signature.items(), key=lambda row: row[0]
    ):
        if signature not in candidate_by_signature:
            violations.append(
                SpatialViolation(
                    SpatialViolationCode.MISSING_REQUIRED_FACT,
                    _fact_subject(fact),
                    "required hard invariant is absent",
                )
            )
    for signature, fact in sorted(
        forbidden_by_signature.items(), key=lambda row: row[0]
    ):
        if signature in candidate_by_signature:
            violations.append(
                SpatialViolation(
                    SpatialViolationCode.FORBIDDEN_FACT_PRESENT,
                    _fact_subject(fact),
                    "candidate contains a forbidden hard invariant",
                )
            )

    exact_kinds = frozenset(requirements.exact_fact_kinds)
    allowed_exact = {
        signature
        for signature, fact in required_by_signature.items()
        if fact.kind in exact_kinds
    }
    for signature, fact in sorted(
        candidate_by_signature.items(), key=lambda row: row[0]
    ):
        if fact.kind in exact_kinds and signature not in allowed_exact:
            violations.append(
                SpatialViolation(
                    SpatialViolationCode.EXTRA_EXACT_FACT,
                    _fact_subject(fact),
                    "candidate adds structure in an exact fact kind",
                )
            )

    observations = {item.metric_id: item for item in candidate.observations}
    for band in requirements.bands:
        observation = observations.get(band.metric_id)
        if observation is None:
            violations.append(
                SpatialViolation(
                    SpatialViolationCode.MISSING_METRIC,
                    band.metric_id,
                    "required admission metric is absent",
                )
            )
            continue
        if observation.unit != band.unit:
            violations.append(
                SpatialViolation(
                    SpatialViolationCode.METRIC_UNIT_MISMATCH,
                    band.metric_id,
                    f"expected unit {band.unit!r}; candidate uses {observation.unit!r}",
                )
            )
            continue
        below = band.lower is not None and observation.value < band.lower
        above = band.upper is not None and observation.value > band.upper
        if below or above:
            violations.append(
                SpatialViolation(
                    SpatialViolationCode.BAND_VIOLATION,
                    band.metric_id,
                    "value "
                    f"{_decimal_text(observation.value)} is outside "
                    f"[{None if band.lower is None else _decimal_text(band.lower)},"
                    f"{None if band.upper is None else _decimal_text(band.upper)}]",
                )
            )

    canonical = tuple(
        sorted(violations, key=lambda row: (row.code.value, row.subject, row.detail))
    )
    return SpatialGateResult(passed=not canonical, violations=canonical)


__all__ = [
    "AdmissibleBand",
    "SpatialGateResult",
    "SpatialRequirementSet",
    "SpatialViolation",
    "SpatialViolationCode",
    "evaluate_spatial_requirements",
]
