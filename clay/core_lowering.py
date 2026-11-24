from __future__ import annotations

from dataclasses import dataclass

from clay.core_contract import (
    ContractCertificate,
    DesignDomain,
    ObservationRow,
    ObservationSignature,
    SourceObligation,
    VerificationStatus,
)


@dataclass(frozen=True)
class LegacyCleanFragment:
    domain_id: str
    intent_id: str
    source_obligation_id: str
    required: bool
    observation_id: str
    observation_kind: str
    unit: str
    status: str = "pass"


def _require_clean_fragment_ids(fragment: LegacyCleanFragment) -> None:
    for field_name in (
        "source_obligation_id",
        "observation_id",
        "observation_kind",
        "unit",
    ):
        if not getattr(fragment, field_name):
            raise ValueError(f"missing {field_name}")


def lower_clean_fragment_obligation(
    fragment: LegacyCleanFragment,
) -> SourceObligation:
    _require_clean_fragment_ids(fragment)
    return SourceObligation(
        id=fragment.source_obligation_id,
        observation_id=fragment.observation_id,
        observation_kind=fragment.observation_kind,
        unit=fragment.unit,
        required=fragment.required,
    )


def lower_clean_fragment_row(fragment: LegacyCleanFragment) -> ObservationRow:
    _require_clean_fragment_ids(fragment)
    return ObservationRow(
        source_obligation_id=fragment.source_obligation_id,
        observation_id=fragment.observation_id,
        observation_kind=fragment.observation_kind,
        unit=fragment.unit,
        status=VerificationStatus(fragment.status),
        required=fragment.required,
    )


def lower_clean_fragment(fragment: LegacyCleanFragment) -> ContractCertificate:
    _require_clean_fragment_ids(fragment)
    return ContractCertificate(
        intent_id=fragment.intent_id,
        domain=DesignDomain(id=fragment.domain_id),
        source_obligations=(lower_clean_fragment_obligation(fragment),),
        observation_basis=(
            ObservationSignature(
                observation_id=fragment.observation_id,
                observation_kind=fragment.observation_kind,
                unit=fragment.unit,
            ),
        ),
        observations=(lower_clean_fragment_row(fragment),),
    )
