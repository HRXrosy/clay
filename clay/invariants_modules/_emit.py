"""
Shared value-check emitter + AC gain resolver (v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 2117-2202.

`_emit_value_violation` is the single canonical builder every per-keyword
check function routes through for scalar comparisons. `_resolve_ac_gain_linear`
turns one of three measure keys into a linear V/V gain.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from ._helpers import _format_value, _relative_error, _within_tolerance
from ._violation import InvariantViolation


# ─── Shared value-check emitter ─────────────────────────────────


def _emit_value_violation(
    *,
    role,
    attr: str,
    target: float,
    actual: Optional[float],
    source: str,
    tolerance: float,
    suggest_fn,
    measure_missing_hint: str,
) -> InvariantViolation:
    """Build the InvariantViolation for a scalar comparison.

    `actual=None` produces a measure_missing entry with the supplied
    hint. A finite actual within the tolerance band produces a skipped
    entry (informational pass). A mismatch produces the full structured
    error with `suggest_fn(target, actual)` as the suggested adjustment.
    """
    if actual is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=None, relative_error=None,
            tolerance=tolerance, category="measure_missing",
            message=(
                f"{role.kind} {role.name!r}: target {attr}="
                f"{_format_value(target)} declared but simulator did not "
                f"produce the supporting measure. {measure_missing_hint}"
            ),
            suggested_adjustment=measure_missing_hint,
        )
    if _within_tolerance(actual, target, tolerance):
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=actual,
            relative_error=_relative_error(actual, target),
            tolerance=tolerance, category="skipped",
            message=(
                f"{role.kind} {role.name!r}: {attr}={_format_value(target)} "
                f"matches actual={_format_value(actual)} within "
                f"±{tolerance*100:g}%"
            ),
            extras={"actual_source": source},
        )
    rel = _relative_error(actual, target)
    direction = "exceeds" if rel > 0 else "underperforms"
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name, target_attr=attr,
        target_value=target, actual_value=actual, relative_error=rel,
        tolerance=tolerance, category="mismatch",
        message=(
            f"{role.kind} {role.name!r}: {attr}={_format_value(target)} "
            f"but actual={_format_value(actual)} (rel_err={rel*100:.1f}%, "
            f"{direction} target by {abs(rel)*100:.1f}%, "
            f"tolerance=±{tolerance*100:g}%)"
        ),
        suggested_adjustment=suggest_fn(target, actual),
        extras={"actual_source": source},
    )


# ─── Helpers: resolve AC gain in linear form ─────────────────────


def _resolve_ac_gain_linear(
    role_name: str, ac_result: Mapping[str, float]
) -> tuple[Optional[float], str]:
    """Pick a low-frequency voltage gain in LINEAR form.

    Returns `(value, source_key)` or `(None, "")` when no gain measure
    is available. The conversion from dB to linear uses 10**(dB/20).
    """
    keyed_lin = f"gain_{role_name}"
    if keyed_lin in ac_result and math.isfinite(ac_result[keyed_lin]):
        return float(ac_result[keyed_lin]), keyed_lin
    if "gain_dc_lin" in ac_result and math.isfinite(ac_result["gain_dc_lin"]):
        return float(ac_result["gain_dc_lin"]), "gain_dc_lin"
    if "gain_dc" in ac_result and math.isfinite(ac_result["gain_dc"]):
        # gain_dc is in dB by convention (vdb()). Convert to linear so
        # the tolerance band applies to the V/V ratio rather than the
        # dB scale (a 20% tolerance on dB is far too loose for gain
        # comparison).
        return 10.0 ** (float(ac_result["gain_dc"]) / 20.0), "gain_dc(dB→lin)"
    return None, ""
