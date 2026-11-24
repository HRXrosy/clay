"""Strict structure-gated physical-build orchestration.

The placement passed to routing must be the placement measured against the
reference.  This module owns that handoff and deliberately knows nothing about
circuit coordinates or area targets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


PHYSICAL_BUILD_STAGE_ORDER = (
    "materialize",
    "place",
    "strict_structure_gate",
    "route",
    "emit",
    "drc",
    "lvs",
    "pex",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PASS_COMPONENT_FIELDS = (
    "device_structure_status",
    "current_pcell_structure_status",
    "macro_organization_status",
    "required_structure_status",
)


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class PhysicalBuildSpec:
    """Identity required by one strict pre-route admission."""

    circuit_key: str
    evidence_circuit_id: str
    source_sha256: str
    topology_sha256: str
    reference_gds_sha256: str
    accepted_gate_schemas: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("circuit_key", "evidence_circuit_id"):
            value = getattr(self, field)
            if type(value) is not str or not value:
                raise ValueError(f"{field} must be one exact non-empty string")
        for field in (
            "source_sha256",
            "topology_sha256",
            "reference_gds_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if (
            type(self.accepted_gate_schemas) is not tuple
            or not self.accepted_gate_schemas
            or any(type(value) is not str or not value for value in self.accepted_gate_schemas)
            or self.accepted_gate_schemas != tuple(sorted(set(self.accepted_gate_schemas)))
        ):
            raise ValueError("accepted gate schemas must be canonical and unique")


@dataclass(frozen=True, slots=True)
class PlacementHandoff:
    """The exact placement state that similarity and routing must share."""

    circuit_key: str
    placement_state_sha256: str
    placement_receipt_sha256: str
    payload: Any

    def __post_init__(self) -> None:
        if type(self.circuit_key) is not str or not self.circuit_key:
            raise ValueError("placement circuit key must be exact")
        _require_sha256(
            self.placement_state_sha256,
            field="placement_state_sha256",
        )
        _require_sha256(
            self.placement_receipt_sha256,
            field="placement_receipt_sha256",
        )


@dataclass(frozen=True, slots=True)
class StrictStructureAdmission:
    """A passing diagnostic receipt bound to one immutable placement."""

    circuit_key: str
    gate_schema: str
    gate_receipt_sha256: str
    placement_state_sha256: str
    placement_receipt_sha256: str
    reference_gds_sha256: str
    status: str = "pass"

    def __post_init__(self) -> None:
        if self.status != "pass":
            raise ValueError("strict structure admission can only represent pass")
        if type(self.circuit_key) is not str or not self.circuit_key:
            raise ValueError("strict admission circuit key must be exact")
        if type(self.gate_schema) is not str or not self.gate_schema:
            raise ValueError("strict admission gate schema must be exact")
        for field in (
            "gate_receipt_sha256",
            "placement_state_sha256",
            "placement_receipt_sha256",
            "reference_gds_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)


@dataclass(frozen=True, slots=True)
class PhysicalBuildResult:
    placement: PlacementHandoff
    strict_structure_admission: StrictStructureAdmission
    downstream_result: Any


class StrictStructureGateRejected(ValueError):
    """The measured placement may not enter routing."""


def _gate_receipt_sha256(payload: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise StrictStructureGateRejected(
            "strict structure receipt is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def require_strict_structure_admission(
    *,
    spec: PhysicalBuildSpec,
    placement: PlacementHandoff,
    gate_payload: Mapping[str, object],
) -> StrictStructureAdmission:
    """Validate a strict receipt without consulting area as an admission axis."""

    if type(spec) is not PhysicalBuildSpec:
        raise TypeError("physical build spec has a foreign type")
    if type(placement) is not PlacementHandoff:
        raise TypeError("placement handoff has a foreign type")
    if placement.circuit_key != spec.circuit_key:
        raise StrictStructureGateRejected("placement circuit identity differs")
    if type(gate_payload) is not dict:
        raise StrictStructureGateRejected(
            "strict structure receipt must be one exact object"
        )

    schema = gate_payload.get("schema")
    if schema not in spec.accepted_gate_schemas:
        raise StrictStructureGateRejected("strict structure receipt schema differs")
    if (
        gate_payload.get("scope") != "pre-route-placement-only"
        or gate_payload.get("diagnostic_only") is not True
        or gate_payload.get("routing_status") != "not_run_pre_route_gate"
    ):
        raise StrictStructureGateRejected(
            "strict structure receipt is not a pre-route diagnostic"
        )

    identities = (
        ("source_sha256", spec.source_sha256, "source"),
        ("topology_sha256", spec.topology_sha256, "topology"),
        ("reference_gds_sha256", spec.reference_gds_sha256, "reference GDS"),
        (
            "placement_state_sha256",
            placement.placement_state_sha256,
            "placement state",
        ),
        (
            "placement_receipt_sha256",
            placement.placement_receipt_sha256,
            "placement receipt",
        ),
    )
    for field, expected, label in identities:
        try:
            measured = _require_sha256(gate_payload.get(field), field=field)
        except ValueError as exc:
            raise StrictStructureGateRejected(str(exc)) from exc
        if measured != expected:
            raise StrictStructureGateRejected(f"strict gate {label} differs")

    blockers = gate_payload.get("strict_structural_blockers")
    if type(blockers) is not list or any(
        type(value) is not str or not value for value in blockers
    ):
        raise StrictStructureGateRejected(
            "strict structure blockers are malformed"
        )
    strict_status = gate_payload.get("strict_structural_status")
    routing_admission = gate_payload.get("routing_admission_status")
    if strict_status != "pass" or routing_admission != "pass" or blockers:
        detail = ",".join(blockers) if blockers else "strict structural status"
        raise StrictStructureGateRejected(f"strict structure gate rejected: {detail}")
    failed_components = tuple(
        field
        for field in _PASS_COMPONENT_FIELDS
        if gate_payload.get(field) != "pass"
    )
    if failed_components:
        raise StrictStructureGateRejected(
            "strict structure components failed: " + ",".join(failed_components)
        )

    return StrictStructureAdmission(
        circuit_key=spec.circuit_key,
        gate_schema=str(schema),
        gate_receipt_sha256=_gate_receipt_sha256(gate_payload),
        placement_state_sha256=placement.placement_state_sha256,
        placement_receipt_sha256=placement.placement_receipt_sha256,
        reference_gds_sha256=spec.reference_gds_sha256,
    )


def execute_physical_build(
    *,
    spec: PhysicalBuildSpec,
    place_stage: Callable[[], PlacementHandoff],
    strict_structure_stage: Callable[[PlacementHandoff], Mapping[str, object]],
    route_emit_signoff_stage: Callable[
        [PlacementHandoff, StrictStructureAdmission], Any
    ],
) -> PhysicalBuildResult:
    """Run placement, strict comparison, then the one downstream heavy stage."""

    placement = place_stage()
    if type(placement) is not PlacementHandoff:
        raise TypeError("place stage returned a foreign handoff")
    gate_payload = strict_structure_stage(placement)
    admission = require_strict_structure_admission(
        spec=spec,
        placement=placement,
        gate_payload=gate_payload,
    )
    downstream = route_emit_signoff_stage(placement, admission)
    return PhysicalBuildResult(
        placement=placement,
        strict_structure_admission=admission,
        downstream_result=downstream,
    )


__all__ = [
    "PHYSICAL_BUILD_STAGE_ORDER",
    "PhysicalBuildResult",
    "PhysicalBuildSpec",
    "PlacementHandoff",
    "StrictStructureAdmission",
    "StrictStructureGateRejected",
    "execute_physical_build",
    "require_strict_structure_admission",
]
