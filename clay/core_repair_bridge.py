from __future__ import annotations

from dataclasses import dataclass

from clay.core_contract import ObservationRow, RepairStepEvidence, VerificationStatus


@dataclass(frozen=True)
class LegacyRepairAttempt:
    repair_id: str
    target_obligation_id: str
    before_row: ObservationRow
    after_row: ObservationRow | None


def _matches_target(row: ObservationRow | None, target_obligation_id: str) -> bool:
    return (
        row is not None
        and row.source_obligation_id == target_obligation_id
    )


def valid_repair_attempt(attempt: LegacyRepairAttempt) -> bool:
    target = attempt.target_obligation_id
    return (
        _matches_target(attempt.before_row, target)
        and attempt.before_row.status == VerificationStatus.FAIL
        and _matches_target(attempt.after_row, target)
        and attempt.after_row.status == VerificationStatus.PASS
    )


def repair_step_from_legacy_attempt(
    attempt: LegacyRepairAttempt,
) -> RepairStepEvidence:
    if attempt.after_row is None:
        raise ValueError("legacy repair attempt is missing after evidence")
    if not _matches_target(attempt.before_row, attempt.target_obligation_id):
        raise ValueError("legacy repair attempt before row target mismatch")
    if not _matches_target(attempt.after_row, attempt.target_obligation_id):
        raise ValueError("legacy repair attempt after row target mismatch")
    if not valid_repair_attempt(attempt):
        raise ValueError("legacy repair attempt must repair fail status to pass status")

    return RepairStepEvidence(
        id=attempt.repair_id,
        source_obligation_id=attempt.target_obligation_id,
    )
