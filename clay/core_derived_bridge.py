from __future__ import annotations

from types import SimpleNamespace

from clay.core_bridge import (
    certificate_from_proof_trace,
    core_certificate_valid_from_trace,
)
from clay.core_contract import ContractCertificate
from clay.core_observation_extract import ObservationMetadata, observation_metadata_maps
from clay.proof_trace import proof_trace_from_verify_result

try:
    from clay.core_verify_metadata import metadata_from_verify_result as _core_metadata
except ImportError:
    _core_metadata = None


def certificate_from_derived_assertions(
    result,
    *,
    units: dict[str, str] | None = None,
    kinds: dict[str, str] | None = None,
) -> ContractCertificate:
    derived_assertions = tuple(result.derived_assertions)
    verify_result = _verify_result_from_derived_assertions(
        result.cell_name,
        derived_assertions,
    )
    trace = proof_trace_from_verify_result(verify_result)
    observation_kinds, observation_units = _metadata_maps(
        verify_result,
        units=units,
        kinds=kinds,
    )
    return certificate_from_proof_trace(
        trace,
        observation_kinds=observation_kinds,
        units=observation_units,
    )


def core_certificate_valid_from_derived_assertions(
    result,
    *,
    units=None,
    kinds=None,
) -> bool:
    derived_assertions = tuple(result.derived_assertions)
    verify_result = _verify_result_from_derived_assertions(
        result.cell_name,
        derived_assertions,
    )
    trace = proof_trace_from_verify_result(verify_result)
    observation_kinds, observation_units = _metadata_maps(
        verify_result,
        units=units,
        kinds=kinds,
    )
    return core_certificate_valid_from_trace(
        trace,
        observation_kinds=observation_kinds,
        units=observation_units,
    )


def _verify_result_from_derived_assertions(
    cell_name: str,
    derived_assertions: tuple[object, ...],
):
    return SimpleNamespace(
        cell_name=cell_name,
        assertions=list(derived_assertions),
    )


def _metadata_maps(
    verify_result,
    *,
    units: dict[str, str] | None,
    kinds: dict[str, str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    if _core_metadata is not None:
        return observation_metadata_maps(
            _core_metadata(verify_result, units=units, kinds=kinds)
        )

    return observation_metadata_maps(
        _local_metadata_from_verify_result(
            verify_result,
            units=units,
            kinds=kinds,
        )
    )


def _local_metadata_from_verify_result(
    verify_result,
    *,
    units: dict[str, str] | None,
    kinds: dict[str, str] | None,
) -> tuple[ObservationMetadata, ...]:
    metadata: list[ObservationMetadata] = []
    seen_source_ids: set[str] = set()

    for assertion in verify_result.assertions:
        source_id = getattr(assertion, "source_obligation_id", None)
        if not source_id:
            raise ValueError(
                "AssertionResult is missing source_obligation_id; "
                "derived assertion certificates require source obligations"
            )
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate source_obligation_id: {source_id}")
        seen_source_ids.add(source_id)

        assertion_name = str(getattr(assertion, "name"))
        metadata.append(
            ObservationMetadata(
                source_obligation_id=source_id,
                observation_kind=_override_value(
                    kinds,
                    source_id,
                    assertion_name,
                    assertion_name,
                ),
                unit=_override_value(
                    units,
                    source_id,
                    assertion_name,
                    "unitless",
                ),
            )
        )

    return tuple(metadata)


def _override_value(
    overrides: dict[str, str] | None,
    source_obligation_id: str,
    assertion_name: str,
    default: str,
) -> str:
    if overrides is None:
        return default
    if source_obligation_id in overrides:
        return overrides[source_obligation_id]
    if assertion_name in overrides:
        return overrides[assertion_name]
    return default
