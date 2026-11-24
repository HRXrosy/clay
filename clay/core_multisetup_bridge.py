from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from clay.core_contract import (
    ContractCertificate,
    DesignDomain,
    ObservationRow,
    ObservationSignature,
    SourceObligation,
    valid_certificate,
)


@dataclass
class SetupCoreTrace:
    setup_name: str
    sources: tuple[SourceObligation, ...]
    rows: tuple[ObservationRow, ...]


def certificate_from_setup_traces(
    intent_id: str,
    domain_id: str,
    setup_traces: Iterable[SetupCoreTrace],
    observation_basis: Iterable[ObservationSignature],
) -> ContractCertificate:
    traces = tuple(setup_traces)
    basis = tuple(observation_basis)

    source_obligations: list[SourceObligation] = []
    observations: list[ObservationRow] = []
    seen_source_ids: set[str] = set()

    for trace in traces:
        for source in tuple(trace.sources):
            if source.id in seen_source_ids:
                raise ValueError(f"duplicate source obligation id: {source.id}")
            seen_source_ids.add(source.id)
            source_obligations.append(source)
        observations.extend(tuple(trace.rows))

    _require_basis_coverage(tuple(source_obligations), basis)

    return ContractCertificate(
        intent_id=intent_id,
        domain=DesignDomain(id=domain_id),
        source_obligations=tuple(source_obligations),
        observation_basis=basis,
        observations=tuple(observations),
    )


def core_certificate_valid_from_setup_traces(
    intent_id: str,
    domain_id: str,
    setup_traces: Iterable[SetupCoreTrace],
    observation_basis: Iterable[ObservationSignature],
) -> bool:
    return valid_certificate(
        certificate_from_setup_traces(
            intent_id,
            domain_id,
            setup_traces,
            observation_basis,
        )
    )


def _require_basis_coverage(
    source_obligations: tuple[SourceObligation, ...],
    observation_basis: tuple[ObservationSignature, ...],
) -> None:
    basis_signatures = {
        (signature.observation_id, signature.observation_kind, signature.unit)
        for signature in observation_basis
    }
    for source in source_obligations:
        source_signature = (source.observation_id, source.observation_kind, source.unit)
        if source_signature not in basis_signatures:
            raise ValueError(
                "missing observation basis for source obligation "
                f"{source.id}: {source.observation_id}/{source.observation_kind}/{source.unit}"
            )
