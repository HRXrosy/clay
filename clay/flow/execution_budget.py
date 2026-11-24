"""Invocation budget values and cooperative cancellation, independent of tools."""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping


EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
WALL_BUDGET_EXCEEDED = "WALL_BUDGET_EXCEEDED"
DETERMINISTIC_BUDGET_EXCEEDED = "DETERMINISTIC_BUDGET_EXCEEDED"
_STOP_REASONS = frozenset({
    EXECUTION_CANCELLED, WALL_BUDGET_EXCEEDED, DETERMINISTIC_BUDGET_EXCEEDED,
})


def _validate_seconds(value: float, name: str, *, allow_zero: bool = False) -> float:
    requirement = "finite nonnegative" if allow_zero else "finite positive"
    message = f"{name} must be a {requirement} number"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(message)
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(message)
    return result


def _json_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate finite JSON data and detach every nested mutable value."""
    if not isinstance(value, Mapping):
        raise TypeError("details must be a mapping of JSON values")
    return json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class ExecutionBudget:
    """Optional positive limits; deterministic work is explicitly charged."""

    wall_seconds: float | None = None
    deterministic_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("wall_seconds", "deterministic_seconds"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _validate_seconds(value, name))

    def to_dict(self) -> dict[str, float | None]:
        return {
            "wall_seconds": self.wall_seconds,
            "deterministic_seconds": self.deterministic_seconds,
        }


class ExecutionStopped(RuntimeError):
    """A named cancellation or budget stop with detached flow diagnostics."""

    def __init__(self, reason: str, **details: Any) -> None:
        if reason not in _STOP_REASONS:
            raise ValueError(f"unknown execution stop reason: {reason!r}")
        self.reason = reason
        self._details = _json_snapshot(details)
        super().__init__(reason)

    def flow_error_data(self) -> dict[str, Any]:
        return {"reason": self.reason, **_json_snapshot(self._details)}


class CancellationToken:
    """Thread-safe one-way cancellation; the first supplied reason is retained."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: str | None = None

    def cancel(self, reason: str = "requested") -> None:
        if not isinstance(reason, str):
            raise TypeError("cancellation reason must be a string")
        with self._lock:
            if self._reason is None:
                self._reason = reason

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    @property
    def cancelled(self) -> bool:
        return self.reason is not None


_CANCELLATION_REASON_PRODUCER = CancellationToken.reason.fget
