from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CoreReadiness(str, Enum):
    CORE_READY = "CORE_READY"
    LOWERABLE_WITH_WORK = "LOWERABLE_WITH_WORK"
    EXCLUDED = "EXCLUDED"


@dataclass(frozen=True)
class LegacyEvidenceShape:
    source_obligation_id: str | None = None
    required: bool | None = None
    observation_id: str | None = None
    status: Any | None = None
    observation_kind: str | None = None
    unit: str | None = None
    feature_kind: str | None = "spec_verify"


_EXCLUDED_FEATURE_KINDS = frozenset(
    {
        "dynamic_python_escape",
        "untyped_role_stash",
        "simulator_side_effect",
    }
)


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    return True


def excluded_reason(shape: LegacyEvidenceShape) -> str | None:
    if not _present(shape.source_obligation_id):
        return "missing source_obligation_id"

    if shape.feature_kind in _EXCLUDED_FEATURE_KINDS:
        return f"excluded feature_kind: {shape.feature_kind}"

    if shape.feature_kind != "spec_verify":
        return f"unknown feature_kind: {shape.feature_kind}"

    for field_name in ("required", "observation_id", "status"):
        if not _present(getattr(shape, field_name)):
            return f"missing {field_name}"

    return None


def classify_legacy_evidence(shape: LegacyEvidenceShape) -> CoreReadiness:
    if excluded_reason(shape) is not None:
        return CoreReadiness.EXCLUDED

    if _present(shape.observation_kind) and _present(shape.unit):
        return CoreReadiness.CORE_READY

    return CoreReadiness.LOWERABLE_WITH_WORK
