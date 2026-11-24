from __future__ import annotations

from clay.core_observation_extract import ObservationMetadata


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


def metadata_from_verify_result(
    verify_result,
    *,
    units: dict[str, str] | None = None,
    kinds: dict[str, str] | None = None,
) -> tuple[ObservationMetadata, ...]:
    metadata: list[ObservationMetadata] = []
    seen_source_ids: set[str] = set()

    for assertion in verify_result.assertions:
        source_id = assertion.source_obligation_id
        if not source_id:
            raise ValueError("AssertionResult is missing source_obligation_id")
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate source_obligation_id: {source_id}")
        seen_source_ids.add(source_id)

        assertion_name = assertion.name
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
