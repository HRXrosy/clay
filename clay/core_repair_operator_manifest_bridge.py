from __future__ import annotations

from typing import Any

from clay.core_contract import ObservationRow, RepairStepEvidence, VerificationStatus
from clay.core_repair_bridge import LegacyRepairAttempt
from clay.core_repair_manifest_bridge import (
    LegacyRepairManifest,
    repair_steps_from_manifest,
    valid_repair_manifest,
)


def _find_row_by_trace_id(rows: Any, trace_id: str | None) -> Any | None:
    if trace_id is None:
        return None
    for row in rows:
        if getattr(row, "trace_id", None) == trace_id:
            return row
    return None


def _resolve_step_row(
    *,
    step: Any,
    explicit_row_attr: str,
    trace_id_attr: str,
    candidate_rows: Any,
) -> Any | None:
    explicit_row = getattr(step, explicit_row_attr, None)
    if explicit_row is not None:
        return explicit_row
    return _find_row_by_trace_id(
        candidate_rows,
        getattr(step, trace_id_attr, None),
    )


def _operator_steps(manifest: Any) -> Any:
    if hasattr(manifest, "repair_steps"):
        return getattr(manifest, "repair_steps")
    return getattr(manifest, "steps")


def _target_matches_operator_row(
    *,
    raw_row: Any | None,
    core_row: ObservationRow,
    target_obligation_id: str,
) -> bool:
    return (
        core_row.source_obligation_id == target_obligation_id
        or getattr(raw_row, "obligation_id", None) == target_obligation_id
    )


def _legacy_attempt_from_operator_step(
    step: Any,
    *,
    before_rows: Any,
    final_rows: Any,
    index: int,
) -> LegacyRepairAttempt:
    repair_id = getattr(step, "repair_id", getattr(step, "operator_name", None))
    if repair_id is None:
        raise ValueError(f"operator repair step at index {index} is missing repair id")
    target_obligation_id = getattr(step, "target_obligation_id")
    raw_before_row = _resolve_step_row(
        step=step,
        explicit_row_attr="before_row",
        trace_id_attr="before_trace_id",
        candidate_rows=before_rows,
    )
    raw_after_row = _resolve_step_row(
        step=step,
        explicit_row_attr="after_row",
        trace_id_attr="after_trace_id",
        candidate_rows=final_rows,
    )
    before_row = _core_row_from_candidate(raw_before_row)
    after_row = _core_row_from_candidate(raw_after_row)

    if before_row is None:
        raise ValueError(f"operator repair step at index {index} is missing before row")
    if after_row is None:
        raise ValueError(f"operator repair step at index {index} is missing after row")
    if not _target_matches_operator_row(
        raw_row=raw_before_row,
        core_row=before_row,
        target_obligation_id=target_obligation_id,
    ):
        raise ValueError(
            f"operator repair step at index {index} before row target mismatch"
        )
    if not _target_matches_operator_row(
        raw_row=raw_after_row,
        core_row=after_row,
        target_obligation_id=target_obligation_id,
    ):
        raise ValueError(
            f"operator repair step at index {index} after row target mismatch"
        )
    if before_row.source_obligation_id != after_row.source_obligation_id:
        raise ValueError(
            f"operator repair step at index {index} source obligation mismatch"
        )
    if getattr(before_row, "status", None) != VerificationStatus.FAIL:
        raise ValueError(
            f"operator repair step at index {index} before row must be fail"
        )
    if getattr(after_row, "status", None) != VerificationStatus.PASS:
        raise ValueError(
            f"operator repair step at index {index} after row must be pass"
        )

    return LegacyRepairAttempt(
        repair_id=repair_id,
        target_obligation_id=after_row.source_obligation_id,
        before_row=before_row,
        after_row=after_row,
    )


def _core_row_from_candidate(row: Any | None) -> ObservationRow | None:
    if row is None:
        return None
    if isinstance(row, ObservationRow):
        return row
    status = getattr(row, "status", None)
    return ObservationRow(
        source_obligation_id=getattr(row, "source_obligation_id"),
        observation_id=getattr(row, "source_obligation_id"),
        observation_kind=getattr(row, "observation_kind", "legacy_repair"),
        unit=getattr(row, "unit", "status"),
        status=VerificationStatus(status),
        required=getattr(row, "required", True),
    )


def legacy_repair_manifest_from_operator_manifest(manifest: Any) -> LegacyRepairManifest:
    before_rows = getattr(manifest, "before_rows", ())
    final_rows = getattr(manifest, "final_rows")
    repair_attempts = tuple(
        _legacy_attempt_from_operator_step(
            step,
            before_rows=before_rows,
            final_rows=final_rows,
            index=index,
        )
        for index, step in enumerate(_operator_steps(manifest))
    )
    legacy_manifest = LegacyRepairManifest(repair_attempts=repair_attempts)
    if repair_attempts and not valid_repair_manifest(legacy_manifest):
        raise ValueError("operator repair manifest does not lower to a valid manifest")
    return legacy_manifest


def repair_steps_from_operator_manifest(
    manifest: Any,
) -> tuple[RepairStepEvidence, ...]:
    return repair_steps_from_manifest(
        legacy_repair_manifest_from_operator_manifest(manifest)
    )


def valid_operator_repair_manifest(manifest: Any) -> bool:
    try:
        legacy_manifest = legacy_repair_manifest_from_operator_manifest(manifest)
    except (AttributeError, TypeError, ValueError):
        return False
    return valid_repair_manifest(legacy_manifest)
