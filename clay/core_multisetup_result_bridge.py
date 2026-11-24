from __future__ import annotations

from typing import Iterable

from clay.core_bridge import certificate_from_proof_trace
from clay.core_contract import ContractCertificate, ObservationSignature
from clay.core_multisetup_bridge import (
    SetupCoreTrace,
    certificate_from_setup_traces,
    core_certificate_valid_from_setup_traces,
)
from clay.core_observation_extract import (
    ObservationMetadata,
    observation_metadata_maps,
)
from clay.proof_trace import proof_trace_from_verify_result


def setup_core_traces_from_multisetup_result(
    result,
    metadata: Iterable[ObservationMetadata],
) -> tuple[SetupCoreTrace, ...]:
    observation_kinds, units = observation_metadata_maps(tuple(metadata))
    setup_traces: list[SetupCoreTrace] = []

    for setup_name, verify_result in _iter_setup_verify_results(result):
        trace = proof_trace_from_verify_result(verify_result)
        certificate = certificate_from_proof_trace(
            trace,
            observation_kinds=observation_kinds,
            units=units,
        )
        setup_traces.append(
            SetupCoreTrace(
                setup_name=setup_name,
                sources=certificate.source_obligations,
                rows=certificate.observations,
            )
        )

    return tuple(setup_traces)


def certificate_from_multisetup_result(
    result,
    metadata: Iterable[ObservationMetadata],
) -> ContractCertificate:
    metadata_items = tuple(metadata)
    setup_traces = setup_core_traces_from_multisetup_result(result, metadata_items)
    return certificate_from_setup_traces(
        result.cell_name,
        result.cell_name,
        setup_traces,
        _observation_basis(metadata_items),
    )


def core_certificate_valid_from_multisetup_result(
    result,
    metadata: Iterable[ObservationMetadata],
) -> bool:
    metadata_items = tuple(metadata)
    return core_certificate_valid_from_setup_traces(
        result.cell_name,
        result.cell_name,
        setup_core_traces_from_multisetup_result(result, metadata_items),
        _observation_basis(metadata_items),
    )


def _iter_setup_verify_results(result):
    if hasattr(result, "per_setup"):
        yield from tuple(result.per_setup.items())
        return

    for setup_result in tuple(result.setup_results):
        yield setup_result.setup_name, setup_result.verify_result


def _observation_basis(
    metadata: Iterable[ObservationMetadata],
) -> tuple[ObservationSignature, ...]:
    return tuple(
        ObservationSignature(
            item.source_obligation_id,
            item.observation_kind,
            item.unit,
        )
        for item in metadata
    )
