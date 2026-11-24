from __future__ import annotations

from clay.core_contract import (
    ContractCertificate,
    DesignDomain,
    ObservationRow,
    ObservationSignature,
    SourceObligation,
    VerificationStatus,
    valid_certificate,
)


def certificate_from_proof_trace(
    trace,
    *,
    observation_kinds: dict[str, str] | None = None,
    units: dict[str, str] | None = None,
) -> ContractCertificate:
    observation_kinds = observation_kinds or {}
    units = units or {}
    metadata = _source_metadata(trace, observation_kinds, units)
    source_obligations = tuple(
        SourceObligation(
            id=source.obligation_id,
            observation_id=source.obligation_id,
            observation_kind=metadata[source.obligation_id][0],
            unit=metadata[source.obligation_id][1],
            required=source.required,
        )
        for source in trace.sources
    )
    observation_basis = tuple(
        ObservationSignature(
            observation_id=source.obligation_id,
            observation_kind=metadata[source.obligation_id][0],
            unit=metadata[source.obligation_id][1],
        )
        for source in trace.sources
    )
    observations = tuple(
        ObservationRow(
            source_obligation_id=row.source_obligation_id,
            observation_id=row.source_obligation_id,
            observation_kind=metadata[row.source_obligation_id][0],
            unit=metadata[row.source_obligation_id][1],
            status=VerificationStatus(row.status),
            required=row.required,
        )
        for row in trace.rows
    )
    return ContractCertificate(
        intent_id=trace.cell_name,
        domain=DesignDomain(id=trace.cell_name),
        source_obligations=source_obligations,
        observation_basis=observation_basis,
        observations=observations,
    )


def core_certificate_valid_from_trace(
    trace,
    *,
    observation_kinds=None,
    units=None,
) -> bool:
    return valid_certificate(
        certificate_from_proof_trace(
            trace,
            observation_kinds=observation_kinds,
            units=units,
        )
    )


def _source_metadata(
    trace,
    observation_kinds: dict[str, str],
    units: dict[str, str],
) -> dict[str, tuple[str, str]]:
    metadata = {}
    for source in trace.sources:
        source_id = source.obligation_id
        if source_id not in observation_kinds:
            raise ValueError(f"missing observation kind for {source_id}")
        if source_id not in units:
            raise ValueError(f"missing unit for {source_id}")
        metadata[source_id] = (observation_kinds[source_id], units[source_id])
    return metadata
