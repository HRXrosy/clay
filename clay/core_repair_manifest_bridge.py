from __future__ import annotations

from dataclasses import dataclass

from clay.core_contract import RepairStepEvidence
from clay.core_repair_bridge import (
    LegacyRepairAttempt,
    repair_step_from_legacy_attempt,
    valid_repair_attempt,
)


@dataclass(frozen=True)
class LegacyRepairManifest:
    repair_attempts: tuple[LegacyRepairAttempt, ...]


def valid_repair_manifest(manifest: LegacyRepairManifest) -> bool:
    return bool(manifest.repair_attempts) and all(
        valid_repair_attempt(attempt)
        for attempt in manifest.repair_attempts
    )


def repair_steps_from_manifest(
    manifest: LegacyRepairManifest,
) -> tuple[RepairStepEvidence, ...]:
    steps: list[RepairStepEvidence] = []
    for index, attempt in enumerate(manifest.repair_attempts):
        if not valid_repair_attempt(attempt):
            raise ValueError(f"invalid repair attempt at index {index}")
        steps.append(repair_step_from_legacy_attempt(attempt))

    return tuple(steps)
