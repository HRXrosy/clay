"""
First-batch (v0.0.86 Round 62) functional-role check functions
(v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 1740-2114.

Six checks:
  * `_check_input_transconductor`   — gm_target vs OP-derived gm
  * `_check_output_buffer`          — gain_target vs low-freq AC gain
  * `_check_feedforward_path`       — gain vs low-freq AC gain
  * `_check_dominant_pole_cap`      — pole_freq_target vs -3dB freq
  * `_check_compensation_zero`      — zero_freq_target vs zero freq
  * `_check_tail_current_source`    — iref vs measured OP current
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from ._emit import _emit_value_violation, _resolve_ac_gain_linear
from ._helpers import _as_float, _format_value, _relative_error, _within_tolerance
from ._suggestions import (
    _suggest_gain_adjustment,
    _suggest_gm_adjustment,
    _suggest_iref_adjustment,
    _suggest_pole_adjustment,
    _suggest_zero_adjustment,
)
from ._violation import _TARGET_ATTR_BY_KIND, InvariantViolation


# ─── Per-keyword check functions ────────────────────────────────


def _check_input_transconductor(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare a declared `gm_target` against the OP-derived gm.

    Resolution order for the actual gm value:
      1. A direct `gm_<role_name>` measure on `op_result` (driver may
         emit one via `.print @MX[gm]`; not required by the IR).
      2. A direct `gm` key on `op_result` (single-device convention).
      3. Derived from `id_<role_name>` + `vov_<role_name>` via
         `gm = 2 * id / vov` (long-channel saturation approximation).
      4. None — surface a `measure_missing` violation so the LLM sees
         that the runtime can't validate this target without extra
         instrumentation.
    """
    attr = _TARGET_ATTR_BY_KIND["input_transconductor"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"input_transconductor {role.name!r}: no {attr!r} "
                f"attribute declared; runtime has no target to check"
            ),
            suggested_adjustment="declare `gm_target = <value>` in the role body",
        )

    actual: Optional[float] = None
    fallback_used: str = ""
    # 1) Per-role gm measure
    keyed = f"gm_{role.name}"
    if keyed in op_result and math.isfinite(op_result[keyed]):
        actual = float(op_result[keyed])
        fallback_used = keyed
    # 2) Single-device gm
    elif "gm" in op_result and math.isfinite(op_result["gm"]):
        actual = float(op_result["gm"])
        fallback_used = "gm"
    # 3) Derived gm = 2 * id / vov
    else:
        id_key  = f"id_{role.name}"
        vov_key = f"vov_{role.name}"
        if (
            id_key in op_result and vov_key in op_result
            and math.isfinite(op_result[id_key])
            and math.isfinite(op_result[vov_key])
            and op_result[vov_key] != 0.0
        ):
            actual = 2.0 * float(op_result[id_key]) / float(op_result[vov_key])
            fallback_used = f"derived 2*{id_key}/{vov_key}"

    if actual is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=None, relative_error=None,
            tolerance=tolerance, category="measure_missing",
            message=(
                f"input_transconductor {role.name!r}: target gm_target="
                f"{_format_value(target)} declared but ngspice did not "
                f"produce gm/{role.name}, gm, or id/vov pair to derive "
                f"gm. Add `.print op @m{role.name}[gm]` to the testbench."
            ),
            suggested_adjustment=(
                f"emit a per-device gm measure (e.g. `gm_{role.name}`) "
                f"or supply both `id_{role.name}` and `vov_{role.name}` "
                f"so gm can be derived as 2*id/vov"
            ),
            extras={"resolution_attempts": [keyed, "gm", "id/vov"]},
        )

    if _within_tolerance(actual, target, tolerance):
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=actual,
            relative_error=_relative_error(actual, target),
            tolerance=tolerance, category="skipped",
            message=(
                f"input_transconductor {role.name!r}: gm_target="
                f"{_format_value(target)} matches actual="
                f"{_format_value(actual)} within ±{tolerance*100:g}%"
            ),
            extras={"actual_source": fallback_used},
        )

    rel = _relative_error(actual, target)
    direction = "exceeds" if rel > 0 else "underperforms"
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name, target_attr=attr,
        target_value=target, actual_value=actual,
        relative_error=rel,
        tolerance=tolerance, category="mismatch",
        message=(
            f"input_transconductor {role.name!r}: gm_target="
            f"{_format_value(target)} but actual gm={_format_value(actual)} "
            f"(rel_err={rel*100:.1f}%, {direction} target by "
            f"{abs(rel)*100:.1f}%, tolerance=±{tolerance*100:g}%)"
        ),
        suggested_adjustment=_suggest_gm_adjustment(target, actual),
        extras={"actual_source": fallback_used},
    )


def _check_output_buffer(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `gain_target` against low-frequency AC gain.

    Resolution order:
      1. Per-role gain measure `gain_<role_name>` (linear).
      2. `gain_dc_lin` (linear-domain copy of `gain_dc`, if the
         testbench emits one).
      3. `gain_dc` (in dB) converted to linear via 10**(dB/20).

    The dB→linear conversion uses the SAME tolerance band on the linear
    quantity. A 20% tolerance on a target of 100 V/V means actual ∈
    [80, 120] V/V (≈ -2 dB and +1.6 dB around the target).
    """
    attr = _TARGET_ATTR_BY_KIND["output_buffer"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"output_buffer {role.name!r}: no {attr!r} attribute "
                f"declared; runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `gain_target = <value>` in the role body"
            ),
        )

    actual, source = _resolve_ac_gain_linear(role.name, ac_result)
    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_gain_adjustment,
        measure_missing_hint=(
            "ngspice .meas did not produce gain_<name>, gain_dc_lin, "
            "or gain_dc; check AC testbench measure list"
        ),
    )


def _check_feedforward_path(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `gain` against low-frequency AC gain.

    Same resolution path as `output_buffer` — both check a voltage
    transfer ratio. The role surface form distinguishes them but the
    measured quantity is the same."""
    attr = _TARGET_ATTR_BY_KIND["feedforward_path"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"feedforward_path {role.name!r}: no {attr!r} attribute "
                f"declared; runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `gain = <value>` in the role body"
            ),
        )

    actual, source = _resolve_ac_gain_linear(role.name, ac_result)
    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_gain_adjustment,
        measure_missing_hint=(
            "ngspice .meas did not produce gain_<name>, gain_dc_lin, "
            "or gain_dc; check AC testbench measure list"
        ),
    )


def _check_dominant_pole_cap(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `pole_freq_target` against the simulated -3dB freq.

    Resolution order:
      1. Per-role `pole_freq_<role_name>` measure.
      2. Global `pole_freq_dominant` (e.g. -3dB find).
      3. Derived from `gbw / gain_lin` (single-pole approximation:
         dominant pole = GBW / open-loop DC gain).
    """
    attr = _TARGET_ATTR_BY_KIND["dominant_pole_cap"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"dominant_pole_cap {role.name!r}: no {attr!r} "
                f"declared; runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `pole_freq_target = <value>Hz` in the role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    keyed = f"pole_freq_{role.name}"
    if keyed in ac_result and math.isfinite(ac_result[keyed]):
        actual = float(ac_result[keyed])
        source = keyed
    elif "pole_freq_dominant" in ac_result and math.isfinite(
        ac_result["pole_freq_dominant"]
    ):
        actual = float(ac_result["pole_freq_dominant"])
        source = "pole_freq_dominant"
    else:
        gbw = ac_result.get("gbw")
        # gain_dc is in dB by convention; convert to linear when used.
        gain_lin = _resolve_ac_gain_linear("__dom_pole__", ac_result)[0]
        if (
            gbw is not None and math.isfinite(gbw)
            and gain_lin is not None and gain_lin != 0.0
        ):
            actual = float(gbw) / abs(gain_lin)
            source = f"derived gbw/{_format_value(gain_lin)}"

    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_pole_adjustment,
        measure_missing_hint=(
            "ngspice .meas did not produce pole_freq_<name>, "
            "pole_freq_dominant, nor both gbw and gain_dc to derive "
            "the dominant pole; check AC testbench measure list"
        ),
    )


def _check_compensation_zero(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `zero_freq_target` against simulated zero freq.

    Resolution order:
      1. Per-role `zero_freq_<role_name>` measure.
      2. Global `zero_freq` measure (when only one zero is interesting).
    No fallback derivation — a compensation zero needs an explicit phase
    or magnitude probe to localize; the runtime surfaces a measure_missing
    when neither is present.
    """
    attr = _TARGET_ATTR_BY_KIND["compensation_zero"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"compensation_zero {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `zero_freq_target = <value>Hz` in the role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    keyed = f"zero_freq_{role.name}"
    if keyed in ac_result and math.isfinite(ac_result[keyed]):
        actual = float(ac_result[keyed])
        source = keyed
    elif "zero_freq" in ac_result and math.isfinite(ac_result["zero_freq"]):
        actual = float(ac_result["zero_freq"])
        source = "zero_freq"

    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_zero_adjustment,
        measure_missing_hint=(
            "ngspice .meas did not produce zero_freq_<name> nor "
            "zero_freq; the compensation zero needs an explicit phase- "
            "or magnitude-derived measure"
        ),
    )


def _check_tail_current_source(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `iref` against measured OP current.

    Resolution order:
      1. Per-role `i_<role_name>` measure (e.g. `i_Itail`).
      2. Per-role `iref_<role_name>` measure.
      3. Global `iref` / `itail` measure.
    All measurements compared by absolute value (current direction
    convention varies across testbenches; we care about magnitude).
    """
    attr = _TARGET_ATTR_BY_KIND["tail_current_source"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"tail_current_source {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `iref = <value>A` in the role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    candidates = [
        f"i_{role.name}", f"iref_{role.name}",
        "iref", "itail",
    ]
    for key in candidates:
        if key in op_result and math.isfinite(op_result[key]):
            actual = float(op_result[key])
            source = key
            break

    # Current magnitude is the comparable quantity; both target and
    # actual are coerced to absolute values before the tolerance check.
    # The relative error preserves its sign for the message.
    if actual is None:
        return _emit_value_violation(
            role=role, attr=attr, target=target, actual=None,
            source=source, tolerance=tolerance,
            suggest_fn=_suggest_iref_adjustment,
            measure_missing_hint=(
                "ngspice .meas did not produce i_<name>, iref_<name>, "
                "iref, or itail; add `.meas op iref find i(Iref) at=0` "
                "to the testbench"
            ),
        )
    return _emit_value_violation(
        role=role, attr=attr, target=abs(target), actual=abs(actual),
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_iref_adjustment,
        measure_missing_hint="",
    )
