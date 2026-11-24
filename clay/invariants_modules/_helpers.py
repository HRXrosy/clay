"""
Basic numeric helpers for invariant runtime (v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 1450-1508.
"""

from __future__ import annotations

import math
from typing import Any, Optional


def _as_float(value: Any) -> Optional[float]:
    """Best-effort numeric coercion.

    The parser may store the target either as a Python float (from a
    `number` token) or as a string carrying a unit suffix that the
    transformer did not unwrap. We accept both, returning None when the
    value isn't representable as a finite float."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        # Strip whitespace and try a direct float() coercion. This
        # captures values like '1.5' or '1e3' but NOT '1MHz' — the
        # parser is responsible for unit unwrapping. Refusing to
        # silently interpret unit suffixes here keeps the contract
        # explicit: invariants run on already-resolved SI quantities.
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _relative_error(actual: float, target: float) -> float:
    """Signed relative error: (actual - target) / max(|target|, eps).

    Positive means `actual` exceeds `target`; negative means it
    underperforms. The `eps` denominator floor prevents a divide-by-
    zero for `target == 0` cases (rare but legal — e.g. a zero
    `iref` declaration is degenerate but not a runtime crash)."""
    denom = max(abs(target), 1e-30)
    return (actual - target) / denom


def _within_tolerance(actual: float, target: float, tolerance: float) -> bool:
    """`|actual - target| / |target| <= tolerance`, with `target == 0`
    handled by absolute tolerance against `tolerance` itself."""
    if math.isnan(actual) or math.isnan(target):
        return False
    if abs(target) < 1e-30:
        return abs(actual) <= tolerance
    return abs(_relative_error(actual, target)) <= tolerance


def _format_value(v: Optional[float]) -> str:
    """Compact representation for log lines."""
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if abs(v) >= 1e3 or (abs(v) < 1e-2 and v != 0.0):
        return f"{v:.3e}"
    return f"{v:.4g}"
