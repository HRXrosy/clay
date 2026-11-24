from __future__ import annotations

from typing import Iterable

from clay.core_bridge import (
    certificate_from_proof_trace,
    core_certificate_valid_from_trace,
)
from clay.core_contract import ContractCertificate
from clay.core_observation_extract import (
    ObservationMetadata,
    observation_metadata_maps,
)
from clay.proof_trace import (
    boundary_evidence_manifest_from_runtime_evidence,
    proof_trace_from_verify_result,
)


def certificate_from_verify_result(
    verify_result,
    metadata: Iterable[ObservationMetadata],
) -> ContractCertificate:
    trace = proof_trace_from_verify_result(verify_result)
    observation_kinds, units = observation_metadata_maps(metadata)
    return certificate_from_proof_trace(
        trace,
        observation_kinds=observation_kinds,
        units=units,
    )


def core_certificate_valid_from_verify_result(
    verify_result,
    metadata: Iterable[ObservationMetadata],
) -> bool:
    trace = proof_trace_from_verify_result(verify_result)
    observation_kinds, units = observation_metadata_maps(metadata)
    return core_certificate_valid_from_trace(
        trace,
        observation_kinds=observation_kinds,
        units=units,
    )


def boundary_evidence_manifest_from_verify_result(
    verify_result,
    *,
    pdk_id: str,
    pdk_corner: str,
    devices,
    rule_deck: str,
    specification_measure_names: list[str],
    cell=None,
    multisetup_result=None,
    signoff_checks=None,
    repair_steps=None,
    temp: float = 27.0,
):
    return boundary_evidence_manifest_from_runtime_evidence(
        verify_result=verify_result,
        pdk_id=pdk_id,
        pdk_corner=pdk_corner,
        devices=list(devices),
        rule_deck=rule_deck,
        specification_measure_names=specification_measure_names,
        cell=cell,
        multisetup_result=multisetup_result,
        signoff_checks=signoff_checks,
        repair_steps=repair_steps,
        temp=temp,
    )
