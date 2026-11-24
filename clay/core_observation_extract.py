from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from clay.core_alignment import (
    CoreReadiness,
    LegacyEvidenceShape,
    classify_legacy_evidence,
)
from clay.core_lowering import LegacyCleanFragment


@dataclass(frozen=True)
class ObservationMetadata:
    source_obligation_id: str
    observation_kind: str
    unit: str


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    return True


def metadata_from_clean_fragment(
    fragment: LegacyCleanFragment,
) -> ObservationMetadata:
    for field_name in ("source_obligation_id", "observation_kind", "unit"):
        if not _present(getattr(fragment, field_name)):
            raise ValueError(f"missing {field_name}")

    return ObservationMetadata(
        source_obligation_id=fragment.source_obligation_id,
        observation_kind=fragment.observation_kind,
        unit=fragment.unit,
    )


def metadata_from_evidence_shape(
    shape: LegacyEvidenceShape,
) -> ObservationMetadata:
    readiness = classify_legacy_evidence(shape)
    if readiness is not CoreReadiness.CORE_READY:
        raise ValueError(
            "observation metadata requires CORE_READY evidence shape; "
            f"got {readiness.value}"
        )

    return ObservationMetadata(
        source_obligation_id=shape.source_obligation_id or "",
        observation_kind=shape.observation_kind or "",
        unit=shape.unit or "",
    )


def observation_metadata_maps(
    items: Iterable[ObservationMetadata],
) -> tuple[dict[str, str], dict[str, str]]:
    observation_kinds: dict[str, str] = {}
    units: dict[str, str] = {}

    for item in items:
        source_id = item.source_obligation_id
        if source_id in observation_kinds:
            raise ValueError(f"duplicate source_obligation_id: {source_id}")
        observation_kinds[source_id] = item.observation_kind
        units[source_id] = item.unit

    return observation_kinds, units
