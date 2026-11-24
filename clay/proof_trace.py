"""Runtime proof-trace extraction for proof-driven Clay.

This module is intentionally small: it turns verifier assertions into
the source-obligation and obligation-row shape mirrored in Lean's
`ProofStatus.lean`. It does not issue a proof token; it only builds the
trace data that a certificate layer can check.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from .ir import AssertionResult, Device, VerifyResult


CURRENT_SCHEMA_VERSION = 2
VERIFICATION_STATUSES = frozenset(
    {
        "pass",
        "fail",
        "skip",
        "pending",
        "unsupported",
        "no_convergence",
        "error",
    }
)


@dataclass(frozen=True)
class SourceObligationRecord:
    obligation_id: str
    required: bool = True


@dataclass(frozen=True)
class ObligationRowRecord:
    required: bool
    status: str
    obligation_id: str
    source_obligation_id: str
    trace_id: str


@dataclass(frozen=True)
class ProofTrace:
    cell_name: str
    sources: list[SourceObligationRecord]
    rows: list[ObligationRowRecord]


@dataclass(frozen=True)
class ProofTokenRecord:
    schema_version: int
    cell: str
    status: str
    pdk_corner: str
    temp: float
    obligation_ids: list[str]
    source_obligation_ids: list[str]
    trace_ids: list[str]
    trace_digest: str


@dataclass(frozen=True)
class CompilerTraceEventRecord:
    phase: str
    source_obligation_id: str
    emitted_row_source_obligation_id: str
    emitted_row_obligation_id: str


@dataclass(frozen=True)
class CompilerTraceRecord:
    trace_id: str
    source_spans: list[str]
    events: list[CompilerTraceEventRecord]
    source_obligation_ids: list[str]
    emitted_row_source_obligation_ids: list[str]
    emitted_row_obligation_ids: list[str]


@dataclass(frozen=True)
class CompilerRunManifest:
    schema_version: int
    cell: str
    program_id: str
    circuit_id: str
    trace: CompilerTraceRecord
    certificate: ProofTrace
    token: ProofTokenRecord | None


@dataclass(frozen=True)
class MeasureObservationRecord:
    obligation_id: str
    trace_id: str
    status: str
    value_text: str


@dataclass(frozen=True)
class SpiceOracleManifest:
    schema_version: int
    cell: str
    netlist_path: str
    log_path: str
    run_status: str
    log_digest: str
    observations: list[MeasureObservationRecord]
    oracle_trace_row_trace_ids: list[str]
    certificate: ProofTrace
    token: ProofTokenRecord | None


@dataclass(frozen=True)
class DeviceModelEnvelopeRecord:
    model_name: str
    corner: str
    parameter_digest: str


@dataclass(frozen=True)
class PdkAdequacyManifest:
    schema_version: int
    cell: str
    pdk_id: str
    device_models: list[DeviceModelEnvelopeRecord]
    rule_deck: str
    signoff_rows: list[ObligationRowRecord]
    certificate: ProofTrace
    token: ProofTokenRecord | None


@dataclass(frozen=True)
class SpecificationObservationRecord:
    measure_name: str
    obligation_id: str
    required: bool = True


@dataclass(frozen=True)
class ObservationBasisManifest:
    schema_version: int
    cell: str
    specifications: list[SpecificationObservationRecord]
    functional_roles: list[dict[str, str]]
    signoff_checks: list[dict[str, str]]
    derived_measures: list[dict[str, str]]
    certificate: ProofTrace
    token: ProofTokenRecord | None


@dataclass(frozen=True)
class RepairStepRecord:
    operator_name: str
    target_obligation_id: str
    before_trace_id: str
    after_trace_id: str


@dataclass(frozen=True)
class RepairOperatorManifest:
    schema_version: int
    cell: str
    steps: list[RepairStepRecord]
    final_rows: list[ObligationRowRecord]
    certificate: ProofTrace
    token: ProofTokenRecord | None


@dataclass(frozen=True)
class BoundaryEvidenceManifest:
    schema_version: int
    cell: str
    relative_certificate: ProofTrace
    compiler: CompilerRunManifest
    spice: SpiceOracleManifest
    pdk: PdkAdequacyManifest
    observations: ObservationBasisManifest
    repairs: RepairOperatorManifest
    certificate_digest: str
    token: ProofTokenRecord | None


def proof_trace_from_verify_result(result: VerifyResult) -> ProofTrace:
    sources = _sources_from_verify_result(result)
    rows: list[ObligationRowRecord] = []
    for index, assertion in enumerate(result.assertions):
        source_id = assertion.source_obligation_id
        if not source_id:
            raise ValueError(
                "AssertionResult is missing source_obligation_id; "
                "proof traces require proof-driven Clay verifier output"
            )
        rows.append(_row_from_assertion(result.cell_name, index, assertion))
    return ProofTrace(cell_name=result.cell_name, sources=sources, rows=rows)


def certification_status(trace: ProofTrace) -> str:
    _validate_row_statuses(trace.rows)
    if not _has_required(trace.rows):
        return "no_obligation"
    if _all_required_pass(trace.rows):
        return "pass"
    return _priority_blocking_status(trace.rows)


def certification_status_with_sources(trace: ProofTrace) -> str:
    if _aggregate_covers_required_sources(trace):
        return certification_status(trace)
    return "pending"


def emits_proof_token_with_sources(trace: ProofTrace) -> bool:
    return certification_status_with_sources(trace) == "pass"


def proof_token_from_trace(
    trace: ProofTrace,
    *,
    pdk_corner: str = "",
    temp: float = 27.0,
    trace_digest: str | None = None,
) -> ProofTokenRecord | None:
    if not emits_proof_token_with_sources(trace):
        return None
    required_rows = _required_rows(trace.rows)
    return ProofTokenRecord(
        schema_version=CURRENT_SCHEMA_VERSION,
        cell=trace.cell_name,
        status="pass",
        pdk_corner=pdk_corner,
        temp=temp,
        obligation_ids=[row.obligation_id for row in required_rows],
        source_obligation_ids=_source_obligation_ids(trace.sources),
        trace_ids=[row.trace_id for row in required_rows],
        trace_digest=trace_digest or proof_trace_digest(trace),
    )


def proof_token_from_verify_result(
    result: VerifyResult,
    *,
    pdk_corner: str = "",
    temp: float = 27.0,
    trace_digest: str | None = None,
) -> ProofTokenRecord | None:
    return proof_token_from_trace(
        proof_trace_from_verify_result(result),
        pdk_corner=pdk_corner,
        temp=temp,
        trace_digest=trace_digest,
    )


def compiler_trace_from_proof_trace(
    trace: ProofTrace,
    *,
    trace_id: str | None = None,
    source_spans: list[str] | None = None,
) -> CompilerTraceRecord:
    required_rows = _required_rows(trace.rows)
    events = [
        CompilerTraceEventRecord(
            phase="emitVerifierRow",
            source_obligation_id=row.source_obligation_id,
            emitted_row_source_obligation_id=row.source_obligation_id,
            emitted_row_obligation_id=row.obligation_id,
        )
        for row in required_rows
    ]
    return CompilerTraceRecord(
        trace_id=trace_id or f"compile:{_proof_atom(trace.cell_name)}",
        source_spans=list(source_spans or []),
        events=events,
        source_obligation_ids=_source_obligation_ids(trace.sources),
        emitted_row_source_obligation_ids=[
            row.source_obligation_id for row in required_rows
        ],
        emitted_row_obligation_ids=[row.obligation_id for row in required_rows],
    )


def compiler_trace_preserves_required_source_coverage(
    compiler_trace: CompilerTraceRecord,
    proof_trace: ProofTrace,
) -> bool:
    required_source_ids = _source_obligation_ids(proof_trace.sources)
    required_rows = _required_rows(proof_trace.rows)
    required_row_source_ids = [
        row.source_obligation_id for row in required_rows
    ]
    required_row_ids = [row.obligation_id for row in required_rows]
    return (
        compiler_trace.source_obligation_ids == required_source_ids
        and compiler_trace.emitted_row_source_obligation_ids
        == required_row_source_ids
        and compiler_trace.emitted_row_obligation_ids == required_row_ids
        and compiler_trace.source_obligation_ids
        == compiler_trace.emitted_row_source_obligation_ids
        and [event.source_obligation_id for event in compiler_trace.events]
        == compiler_trace.emitted_row_source_obligation_ids
        and [
            event.emitted_row_source_obligation_id
            for event in compiler_trace.events
        ]
        == compiler_trace.emitted_row_source_obligation_ids
        and [event.emitted_row_obligation_id for event in compiler_trace.events]
        == compiler_trace.emitted_row_obligation_ids
    )


def compiler_run_manifest_from_proof_trace(
    trace: ProofTrace,
    *,
    program_id: str | None = None,
    circuit_id: str | None = None,
    pdk_corner: str = "",
    temp: float = 27.0,
    trace_id: str | None = None,
    source_spans: list[str] | None = None,
) -> CompilerRunManifest:
    compiler_trace = compiler_trace_from_proof_trace(
        trace,
        trace_id=trace_id,
        source_spans=source_spans,
    )
    return CompilerRunManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        cell=trace.cell_name,
        program_id=program_id or trace.cell_name,
        circuit_id=circuit_id or trace.cell_name,
        trace=compiler_trace,
        certificate=trace,
        token=proof_token_from_trace(
            trace,
            pdk_corner=pdk_corner,
            temp=temp,
            trace_digest=proof_trace_digest(trace),
        ),
    )


def compiler_run_manifest_is_certified(
    manifest: CompilerRunManifest,
) -> bool:
    return (
        manifest.schema_version == CURRENT_SCHEMA_VERSION
        and manifest.program_id == manifest.circuit_id
        and manifest.token is not None
        and compiler_trace_preserves_required_source_coverage(
            manifest.trace,
            manifest.certificate,
        )
    )


def spice_oracle_manifest_from_verify_result(
    result: VerifyResult,
    *,
    pdk_corner: str = "",
    temp: float = 27.0,
) -> SpiceOracleManifest:
    trace = proof_trace_from_verify_result(result)
    observations = [
        MeasureObservationRecord(
            obligation_id=row.obligation_id,
            trace_id=row.trace_id,
            status=row.status,
            value_text=_assertion_value_text(assertion),
        )
        for row, assertion in zip(trace.rows, result.assertions)
    ]
    return SpiceOracleManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        cell=result.cell_name,
        netlist_path=result.netlist_path,
        log_path=result.log_path,
        run_status="error" if result.simulator_error else "pass",
        log_digest=_verify_result_log_digest(result),
        observations=observations,
        oracle_trace_row_trace_ids=[row.trace_id for row in trace.rows],
        certificate=trace,
        token=proof_token_from_trace(
            trace,
            pdk_corner=pdk_corner,
            temp=temp,
            trace_digest=proof_trace_digest(trace),
        ),
    )


def spice_oracle_manifest_is_certified(manifest: SpiceOracleManifest) -> bool:
    rows = manifest.certificate.rows
    supported = all(
        any(
            observation.obligation_id == row.obligation_id
            and observation.trace_id == row.trace_id
            and observation.status == row.status
            for observation in manifest.observations
        )
        for row in rows
    )
    return (
        manifest.schema_version == CURRENT_SCHEMA_VERSION
        and manifest.run_status == "pass"
        and supported
        and manifest.oracle_trace_row_trace_ids
        == [row.trace_id for row in rows]
        and manifest.token is not None
    )


def pdk_adequacy_manifest_from_trace(
    trace: ProofTrace,
    *,
    pdk_id: str,
    pdk_corner: str,
    devices: list[Device],
    rule_deck: str,
    signoff_rows: list[ObligationRowRecord] | None = None,
    temp: float = 27.0,
) -> PdkAdequacyManifest:
    return PdkAdequacyManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        cell=trace.cell_name,
        pdk_id=pdk_id,
        device_models=[
            DeviceModelEnvelopeRecord(
                model_name=device.pdk_variant or device.device_class,
                corner=pdk_corner,
                parameter_digest=_device_parameter_digest(device),
            )
            for device in devices
        ],
        rule_deck=rule_deck,
        signoff_rows=list(signoff_rows or []),
        certificate=trace,
        token=proof_token_from_trace(
            trace,
            pdk_corner=pdk_corner,
            temp=temp,
            trace_digest=proof_trace_digest(trace),
        ),
    )


def pdk_adequacy_manifest_is_certified(
    manifest: PdkAdequacyManifest,
) -> bool:
    certificate_rows = {
        (row.obligation_id, row.trace_id) for row in manifest.certificate.rows
    }
    signoff_rows_covered = all(
        (row.obligation_id, row.trace_id) in certificate_rows
        for row in manifest.signoff_rows
    )
    return (
        manifest.schema_version == CURRENT_SCHEMA_VERSION
        and len(manifest.device_models) > 0
        and signoff_rows_covered
        and certification_status_with_sources(manifest.certificate) == "pass"
        and manifest.token is not None
    )


def observation_basis_manifest_from_trace(
    trace: ProofTrace,
    *,
    specification_measure_names: list[str],
    functional_roles: list[dict[str, str]] | None = None,
    signoff_checks: list[dict[str, str]] | None = None,
    derived_measures: list[dict[str, str]] | None = None,
    pdk_corner: str = "",
    temp: float = 27.0,
) -> ObservationBasisManifest:
    required_sources = [
        source for source in trace.sources if source.required
    ]
    specifications = [
        SpecificationObservationRecord(
            measure_name=measure_name,
            obligation_id=source.obligation_id,
            required=source.required,
        )
        for measure_name, source in zip(
            specification_measure_names,
            required_sources,
        )
    ]
    return ObservationBasisManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        cell=trace.cell_name,
        specifications=specifications,
        functional_roles=list(functional_roles or []),
        signoff_checks=list(signoff_checks or []),
        derived_measures=list(derived_measures or []),
        certificate=trace,
        token=proof_token_from_trace(
            trace,
            pdk_corner=pdk_corner,
            temp=temp,
            trace_digest=proof_trace_digest(trace),
        ),
    )


def observation_basis_manifest_from_runtime_evidence(
    trace: ProofTrace,
    *,
    specification_measure_names: list[str],
    cell=None,
    multisetup_result=None,
    signoff_checks: list[dict[str, str]] | None = None,
    pdk_corner: str = "",
    temp: float = 27.0,
) -> ObservationBasisManifest:
    functional_roles = (
        functional_role_observations_from_cell(cell) if cell is not None else []
    )
    derived_measures = (
        derived_measure_observations_from_multisetup_result(multisetup_result)
        if multisetup_result is not None
        else []
    )
    return observation_basis_manifest_from_trace(
        trace,
        specification_measure_names=specification_measure_names,
        functional_roles=functional_roles,
        signoff_checks=signoff_checks,
        derived_measures=derived_measures,
        pdk_corner=pdk_corner,
        temp=temp,
    )


def derived_measure_observations_from_multisetup_result(result) -> list[dict[str, str]]:
    source_ids_by_name = {}
    for assertion in tuple(result.derived_assertions):
        source_id = assertion.source_obligation_id
        if not source_id:
            raise ValueError(
                "derived assertion is missing source_obligation_id"
            )
        source_ids_by_name[assertion.name] = source_id

    observations: list[dict[str, str]] = []
    for name, value in dict(result.derived_measures).items():
        if name not in source_ids_by_name:
            raise ValueError(
                f"derived measure {name!r} has no derived assertion source_obligation_id"
            )
        observations.append(
            {
                "name": str(name),
                "value": f"{value:g}",
                "source_obligation_id": source_ids_by_name[name],
            }
        )
    return observations


def verify_result_from_multisetup_result(result) -> VerifyResult:
    setup_assertions = [
        assertion
        for setup_result in dict(result.per_setup).values()
        for assertion in list(getattr(setup_result, "assertions", []))
    ]
    simulator_errors = [
        f"{name}: {setup_result.simulator_error}"
        for name, setup_result in dict(result.per_setup).items()
        if getattr(setup_result, "simulator_error", None)
    ]
    return VerifyResult(
        cell_name=str(result.cell_name),
        measures=dict(result.derived_measures),
        assertions=setup_assertions + list(result.derived_assertions),
        log_path=_multisetup_path_summary(result, "log_path"),
        netlist_path=_multisetup_path_summary(result, "netlist_path"),
        simulator_error="; ".join(simulator_errors) if simulator_errors else None,
    )


def functional_role_observations_from_cell(cell) -> list[dict[str, str]]:
    direct_roles = getattr(cell, "functional_roles", None)
    roles = list(direct_roles) if direct_roles else list(
        getattr(cell, "__dict__", {}).get("_functional_roles", [])
    )
    observations: list[dict[str, str]] = []
    for role in roles:
        attributes = getattr(role, "attributes", {}) or {}
        source_id = attributes.get("source_obligation_id")
        if not source_id:
            raise ValueError(
                f"functional role {getattr(role, 'name', '?')!r} "
                "is missing source_obligation_id"
            )
        observations.append(
            {
                "kind": str(getattr(role, "kind", "")),
                "name": str(getattr(role, "name", "")),
                "source_obligation_id": str(source_id),
            }
        )
    return observations


def observation_basis_manifest_is_certified(
    manifest: ObservationBasisManifest,
) -> bool:
    specification_source_ids = {
        observation.obligation_id
        for observation in manifest.specifications
        if observation.required
    }
    functional_role_source_ids = _dict_source_obligation_ids(
        manifest.functional_roles
    )
    signoff_source_ids = _dict_source_obligation_ids(manifest.signoff_checks)
    derived_source_ids = _derived_measure_source_ids(manifest.derived_measures)
    observed_required_source_ids = (
        specification_source_ids
        | functional_role_source_ids
        | signoff_source_ids
        | derived_source_ids
    )
    required_source_ids = set(_source_obligation_ids(manifest.certificate.sources))
    has_observation_basis = bool(observed_required_source_ids)
    return (
        manifest.schema_version == CURRENT_SCHEMA_VERSION
        and has_observation_basis
        and required_source_ids <= observed_required_source_ids
        and certification_status_with_sources(manifest.certificate) == "pass"
        and manifest.token is not None
    )


def repair_operator_manifest_from_trace(
    trace: ProofTrace,
    *,
    repair_steps: list[RepairStepRecord],
    pdk_corner: str = "",
    temp: float = 27.0,
) -> RepairOperatorManifest:
    return RepairOperatorManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        cell=trace.cell_name,
        steps=list(repair_steps),
        final_rows=list(trace.rows),
        certificate=trace,
        token=proof_token_from_trace(
            trace,
            pdk_corner=pdk_corner,
            temp=temp,
            trace_digest=proof_trace_digest(trace),
        ),
    )


def repair_operator_manifest_is_certified(
    manifest: RepairOperatorManifest,
) -> bool:
    certificate_rows = {
        (row.obligation_id, row.trace_id) for row in manifest.certificate.rows
    }
    steps_covered = all(
        (step.target_obligation_id, step.after_trace_id) in certificate_rows
        for step in manifest.steps
    )
    return (
        manifest.schema_version == CURRENT_SCHEMA_VERSION
        and steps_covered
        and certification_status_with_sources(manifest.certificate) == "pass"
        and manifest.token is not None
    )


def boundary_evidence_manifest_from_parts(
    *,
    relative_certificate: ProofTrace,
    compiler: CompilerRunManifest,
    spice: SpiceOracleManifest,
    pdk: PdkAdequacyManifest,
    observations: ObservationBasisManifest,
    repairs: RepairOperatorManifest,
    pdk_corner: str = "",
    temp: float = 27.0,
) -> BoundaryEvidenceManifest:
    digest = proof_trace_digest(relative_certificate)
    return BoundaryEvidenceManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        cell=relative_certificate.cell_name,
        relative_certificate=relative_certificate,
        compiler=compiler,
        spice=spice,
        pdk=pdk,
        observations=observations,
        repairs=repairs,
        certificate_digest=digest,
        token=proof_token_from_trace(
            relative_certificate,
            pdk_corner=pdk_corner,
            temp=temp,
            trace_digest=digest,
        ),
    )


def boundary_evidence_manifest_from_runtime_evidence(
    *,
    verify_result: VerifyResult,
    pdk_id: str,
    pdk_corner: str,
    devices: list[Device],
    rule_deck: str,
    specification_measure_names: list[str],
    cell=None,
    multisetup_result=None,
    signoff_checks: list[dict[str, str]] | None = None,
    repair_steps: list[RepairStepRecord] | None = None,
    temp: float = 27.0,
) -> BoundaryEvidenceManifest:
    trace = proof_trace_from_verify_result(verify_result)
    return boundary_evidence_manifest_from_parts(
        relative_certificate=trace,
        compiler=compiler_run_manifest_from_proof_trace(
            trace,
            pdk_corner=pdk_corner,
            temp=temp,
        ),
        spice=spice_oracle_manifest_from_verify_result(
            verify_result,
            pdk_corner=pdk_corner,
            temp=temp,
        ),
        pdk=pdk_adequacy_manifest_from_trace(
            trace,
            pdk_id=pdk_id,
            pdk_corner=pdk_corner,
            temp=temp,
            devices=devices,
            rule_deck=rule_deck,
            signoff_rows=trace.rows,
        ),
        observations=observation_basis_manifest_from_runtime_evidence(
            trace,
            specification_measure_names=specification_measure_names,
            cell=cell,
            multisetup_result=multisetup_result,
            signoff_checks=signoff_checks,
            pdk_corner=pdk_corner,
            temp=temp,
        ),
        repairs=repair_operator_manifest_from_trace(
            trace,
            repair_steps=list(repair_steps or []),
            pdk_corner=pdk_corner,
            temp=temp,
        ),
        pdk_corner=pdk_corner,
        temp=temp,
    )


def boundary_evidence_manifest_certificates_aligned(
    manifest: BoundaryEvidenceManifest,
) -> bool:
    return all(
        component.certificate == manifest.relative_certificate
        for component in _boundary_evidence_components(manifest)
    )


def boundary_evidence_manifest_is_certified(
    manifest: BoundaryEvidenceManifest,
) -> bool:
    return (
        manifest.schema_version == CURRENT_SCHEMA_VERSION
        and manifest.token is not None
        and manifest.token.trace_digest == manifest.certificate_digest
        and boundary_evidence_manifest_certificates_aligned(manifest)
        and _boundary_evidence_component_tokens_match(manifest)
        and compiler_run_manifest_is_certified(manifest.compiler)
        and spice_oracle_manifest_is_certified(manifest.spice)
        and pdk_adequacy_manifest_is_certified(manifest.pdk)
        and observation_basis_manifest_is_certified(manifest.observations)
        and repair_operator_manifest_is_certified(manifest.repairs)
    )


def proof_trace_digest(trace: ProofTrace) -> str:
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "cell": trace.cell_name,
        "sources": [
            {
                "obligation_id": source.obligation_id,
                "required": source.required,
            }
            for source in trace.sources
        ],
        "rows": [
            {
                "required": row.required,
                "status": row.status,
                "obligation_id": row.obligation_id,
                "source_obligation_id": row.source_obligation_id,
                "trace_id": row.trace_id,
            }
            for row in trace.rows
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _row_from_assertion(
    cell_name: str,
    index: int,
    assertion: AssertionResult,
) -> ObligationRowRecord:
    assert assertion.source_obligation_id is not None
    return ObligationRowRecord(
        required=True,
        status=_verification_status_from_assertion(assertion),
        obligation_id=(
            f"row:{_proof_atom(cell_name)}:{index}:{_proof_atom(assertion.name)}"
        ),
        source_obligation_id=assertion.source_obligation_id,
        trace_id=(
            f"verify:{_proof_atom(cell_name)}:{index}:"
            f"{_proof_atom(assertion.name)}"
        ),
    )


def _sources_from_verify_result(result: VerifyResult) -> list[SourceObligationRecord]:
    compiler_sources = result.__dict__.get("_compiler_source_obligations")
    if compiler_sources is not None:
        return [
            SourceObligationRecord(
                obligation_id=str(source["obligation_id"]),
                required=bool(source.get("required", True)),
            )
            for source in compiler_sources
        ]
    compiler_source_ids = result.__dict__.get("_compiler_source_obligation_ids")
    if compiler_source_ids is not None:
        return [
            SourceObligationRecord(obligation_id=str(source_id))
            for source_id in compiler_source_ids
        ]
    return [
        SourceObligationRecord(obligation_id=assertion.source_obligation_id)
        for assertion in result.assertions
        if assertion.source_obligation_id
    ]


def _verification_status_from_assertion(assertion: AssertionResult) -> str:
    if assertion.category == "pass" and assertion.passed:
        return "pass"
    if assertion.category == "spec_violation":
        return "fail"
    if assertion.category == "measure_missing":
        return "pending"
    if assertion.category == "simulator_error":
        return "error"
    return "error"


def _assertion_value_text(assertion: AssertionResult) -> str:
    return "" if assertion.actual is None else str(assertion.actual)


def _verify_result_log_digest(result: VerifyResult) -> str:
    payload = {
        "cell": result.cell_name,
        "log_path": result.log_path,
        "netlist_path": result.netlist_path,
        "simulator_error": result.simulator_error,
        "measures": dict(result.measures),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _multisetup_path_summary(result, attribute: str) -> str:
    paths = [
        str(getattr(setup_result, attribute, ""))
        for setup_result in dict(result.per_setup).values()
        if getattr(setup_result, attribute, "")
    ]
    return "multi_setup:" + ",".join(paths) if paths else "multi_setup"


def _device_parameter_digest(device: Device) -> str:
    payload = {
        "instance_name": device.instance_name,
        "device_class": device.device_class,
        "pdk_variant": device.pdk_variant,
        "pins": device.pins,
        "sizing": device.sizing,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _has_required(rows: list[ObligationRowRecord]) -> bool:
    return any(row.required for row in rows)


def _validate_row_statuses(rows: list[ObligationRowRecord]) -> None:
    for row in rows:
        if row.status not in VERIFICATION_STATUSES:
            raise ValueError(
                f"unknown verification status for {row.obligation_id}: "
                f"{row.status!r}"
            )


def _all_required_pass(rows: list[ObligationRowRecord]) -> bool:
    return all((not row.required) or row.status == "pass" for row in rows)


def _priority_blocking_status(rows: list[ObligationRowRecord]) -> str:
    for status in ("fail", "error", "no_convergence", "unsupported", "pending"):
        if any(row.required and row.status == status for row in rows):
            return status
    return "error"


def _required_rows(rows: list[ObligationRowRecord]) -> list[ObligationRowRecord]:
    return [row for row in rows if row.required]


def _aggregate_covers_required_sources(trace: ProofTrace) -> bool:
    covered = {row.source_obligation_id for row in _required_rows(trace.rows)}
    return all(
        (not source.required) or source.obligation_id in covered
        for source in trace.sources
    )


def _boundary_evidence_components(manifest: BoundaryEvidenceManifest):
    return (
        manifest.compiler,
        manifest.spice,
        manifest.pdk,
        manifest.observations,
        manifest.repairs,
    )


def _boundary_evidence_component_tokens_match(
    manifest: BoundaryEvidenceManifest,
) -> bool:
    for component in _boundary_evidence_components(manifest):
        if component.schema_version != CURRENT_SCHEMA_VERSION:
            return False
        if component.token is None:
            return False
        if component.token.trace_digest != manifest.certificate_digest:
            return False
    return True


def _source_obligation_ids(sources: list[SourceObligationRecord]) -> list[str]:
    return [source.obligation_id for source in sources if source.required]


def _derived_measure_source_ids(
    derived_measures: list[dict[str, str]],
) -> set[str]:
    return _dict_source_obligation_ids(derived_measures)


def _dict_source_obligation_ids(items: list[dict[str, str]]) -> set[str]:
    return {
        str(item["source_obligation_id"])
        for item in items
        if item.get("source_obligation_id")
    }


def _proof_atom(value: str) -> str:
    atom = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return atom or "unnamed"
