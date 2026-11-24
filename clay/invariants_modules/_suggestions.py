"""
Suggestion-helper functions (LLM-targeted hints) for invariant runtime
(v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py`:
  * Lines 2205-2273 — first-batch (gm/gain/pole/zero/iref)
  * Lines 3216-3303 — second-batch (ro_boost/pole_tracking/dual_feedforward/
                                     loop_bw/mirror/effective_R)
  * Lines 3929-4045 — third-batch (cmrr/droop/regulated_cascode/boost_gain/
                                    iq/dynamic_bias/dynamic_bias_drift)
  * Lines 4047-4115 — fourth-batch (vref/dropout/divide_ratio/temp_coeff)

Each `_suggest_*_adjustment(target, actual)` returns a structured
human-readable hint the LLM consumes alongside the structured
relative_error number.
"""

from __future__ import annotations

from ._helpers import _relative_error


# ─── First-batch suggestion helpers (Round 62 / v0.0.86) ─────────────


def _suggest_gm_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"actual gm is below target. Options: increase W (raises gm "
            f"linearly), increase Id (raises gm as sqrt), or pick a "
            f"higher gm/Id intent in sizing."
        )
    return (
        f"actual gm exceeds target. Options: shrink W, reduce Id, or "
        f"pick a lower gm/Id intent."
    )


def _suggest_gain_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"actual gain is below target. Increase intrinsic gain (longer "
            f"L on the input pair / cascode) or add a second gain stage."
        )
    return (
        f"actual gain exceeds target. Shorter L or lower gm/Id reduces "
        f"intrinsic gain; check if the target was too conservative."
    )


def _suggest_pole_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"actual pole frequency is below target — likely too much "
            f"compensation capacitance. Reduce Cc or move the pole-setting "
            f"node by reducing parasitic load."
        )
    return (
        f"actual pole frequency exceeds target — compensation may be too "
        f"light. Increase Cc to push the dominant pole down."
    )


def _suggest_zero_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"actual zero frequency is below target — nulling resistor too "
            f"large. Reduce Rz (the textbook value is 1/gm_m2)."
        )
    return (
        f"actual zero frequency exceeds target — nulling resistor too "
        f"small. Increase Rz to move the zero closer to GBW."
    )


def _suggest_iref_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"actual tail current is below target. Increase the reference "
            f"current source value or raise the mirror ratio N."
        )
    return (
        f"actual tail current exceeds target. Reduce the reference current "
        f"source value or lower the mirror ratio."
    )


# ─── v0.0.90 (Round 67) — second-batch suggestion helpers ──────────


def _suggest_ro_boost_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"cascode boost below target. Increase cascode device L "
            f"(raises ro_casc), increase gm_casc (wider W or higher Id "
            f"on the cascode), or stack a second cascode level."
        )
    return (
        f"cascode boost exceeds target. Either the target was too "
        f"conservative or the cascode device is over-sized; shorter L "
        f"or smaller W on the cascode reduces the boost."
    )


def _suggest_pole_tracking_adjustment(target: float, actual: float) -> str:
    return (
        f"cross-corner pole drift {actual*100:.1f}% exceeds bound "
        f"{target*100:.1f}%. Replace fixed metal cap with a PVT-tracking "
        f"cap (gate-oxide MOSCAP) or add a Cc-mirror that tracks "
        f"transistor gm across corners."
    )


def _suggest_dual_feedforward_adjustment(
    sub_attr: str, target: float, actual: float,
) -> str:
    if sub_attr == "symmetry":
        return (
            f"symmetry ratio {actual:.3g} differs from target {target:.3g}. "
            f"Match the W/L of the primary and secondary feedforward "
            f"devices, or scale Id on the underperforming path."
        )
    rel = _relative_error(actual, target)
    leg = sub_attr.replace("gm_", "")
    if rel < 0:
        return (
            f"{leg} feedforward gm is below target. Increase W on the "
            f"{leg}-leg device or raise its bias current."
        )
    return (
        f"{leg} feedforward gm exceeds target. Shrink W on the "
        f"{leg}-leg device or reduce its bias current."
    )


def _suggest_loop_bw_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"loop bandwidth below target. Increase charge-pump current "
            f"Icp (raises loop gain linearly), reduce loop-filter "
            f"resistor R1, or shrink C1 (lowest order of effect)."
        )
    return (
        f"loop bandwidth exceeds target. Reduce Icp or raise R1/C1 to "
        f"push the loop pole down."
    )


def _suggest_mirror_adjustment(target: float, actual: float, branch: str) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"branch {branch!r} current is below target. Increase the "
            f"mirror device W/L ratio on the {branch} leg, or shorten L."
        )
    return (
        f"branch {branch!r} current exceeds target. Reduce the mirror "
        f"device W/L ratio on the {branch} leg, or increase L."
    )


def _suggest_effective_R_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"effective R below target — gm is too high (R≈1/gm). "
            f"Shrink the diode-connected device W, lower its Id, or "
            f"pick a higher gm/Id intent."
        )
    return (
        f"effective R exceeds target — gm too low. Increase W, raise "
        f"Id, or pick a lower gm/Id intent."
    )


# ─── v0.0.92 (Round 68) — third-batch suggestion helpers ───────────


def _suggest_cmrr_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"actual CMRR is below target — common-mode rejection too "
            f"weak. Increase the CMFB amplifier gain (longer L on the "
            f"sense devices, or stack a second-stage on the CMFB), or "
            f"tighten device matching between the differential pair "
            f"output legs."
        )
    return (
        f"actual CMRR exceeds target — the spec was conservative. "
        f"Either accept the headroom or simplify the CMFB amplifier "
        f"(fewer stages, smaller W) to recover power/area."
    )


def _suggest_droop_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel > 0:
        return (
            f"actual hold droop exceeds target. Increase the hold "
            f"capacitor C_hold (slows leakage-driven droop linearly), "
            f"size up the sample/hold switch L to lower subthreshold "
            f"leakage, or move the gate drive farther from the "
            f"threshold during hold phase."
        )
    return (
        f"actual hold droop is below target — the spec was generous. "
        f"Either accept the margin or shrink C_hold to recover area."
    )


def _suggest_regulated_cascode_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"regulated-cascode boost below target. Increase the "
            f"auxiliary amplifier gain (A_boost) — longer L on the "
            f"boost-amp input pair, add a second stage, or boost its "
            f"bias current. Also check the cascode device's own ro "
            f"is not loading the boost amp output."
        )
    return (
        f"regulated-cascode boost exceeds target — either the spec was "
        f"conservative or the boost amp is over-designed. Reduce "
        f"A_boost (shorter L on the auxiliary amp) to recover area / "
        f"current."
    )


def _suggest_boost_gain_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"auxiliary-amp gain below target. Increase L on the input "
            f"pair (raises intrinsic gain), add a cascode load to the "
            f"boost amp, or stack a second gain stage. Verify the "
            f"input pair has enough Vov headroom (low gm/Id wins gain)."
        )
    return (
        f"auxiliary-amp gain exceeds target — likely over-designed. "
        f"Shorter L on the input pair or lower bias current reduces "
        f"intrinsic gain and recovers area."
    )


def _suggest_iq_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"quiescent current below target — class-AB stage is starved "
            f"in idle. Increase the bias-loop reference current, raise "
            f"the translinear-loop core current, or grow W on the "
            f"push/pull devices so the idle bias point lands above the "
            f"sub-threshold knee."
        )
    return (
        f"quiescent current exceeds target — class-AB stage burns too "
        f"much idle power. Reduce the bias-loop reference, shrink the "
        f"translinear-loop devices, or pick a lower Vov target on the "
        f"push/pull devices."
    )


def _suggest_dynamic_bias_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"dynamic bias output below target. Increase the beta-multiplier "
            f"core resistor (raises I_ref linearly), or widen the bias-loop "
            f"mirror ratio. Check that the bias-loop start-up circuit has "
            f"pulled the loop out of the zero-current operating point."
        )
    return (
        f"dynamic bias output exceeds target. Reduce the beta-multiplier "
        f"core resistor or shrink the mirror-output W. Verify the bias "
        f"loop is not running with insufficient supply headroom (forces "
        f"the mirror into triode and over-drives the output)."
    )


def _suggest_dynamic_bias_drift_adjustment(
    target: float, actual: float,
) -> str:
    return (
        f"cross-corner PVT drift {actual*100:.1f}% exceeds bound "
        f"{target*100:.1f}%. Tighten the tracking loop: use a "
        f"beta-multiplier with a tighter PTAT slope, replace fixed "
        f"resistors with poly-resistor mix tuned for PVT, or add a "
        f"temperature-compensated trim leg. Also check the start-up "
        f"circuit isn't biased differently across corners."
    )


# ─── v0.0.94 (Round 71) — fourth-batch suggestion helpers ──────────


def _suggest_vref_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"actual vref below target. Increase the PTAT-to-CTAT gain "
            f"ratio (raise the PTAT branch's series-resistor or scale "
            f"the bipolar emitter-area ratio N higher), or check the "
            f"CTAT diode-stack count. Verify start-up circuit pulled "
            f"the loop out of the zero-current operating point."
        )
    return (
        f"actual vref exceeds target. Reduce the PTAT-to-CTAT gain "
        f"ratio (lower the PTAT branch resistor or reduce the bipolar "
        f"emitter-area ratio N), or remove a CTAT diode-stack stage. "
        f"Check that the op-amp offset isn't pushing the reference up."
    )


def _suggest_dropout_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel > 0:
        return (
            f"actual dropout exceeds target — pass device drops too "
            f"much under full load. Increase W on the pass transistor "
            f"(lowers Rds_on linearly), reduce the load current spec, "
            f"or pick a lower-VT pass device variant. Check the "
            f"pass-device gate drive is at its rail-extremum."
        )
    return (
        f"actual dropout is below target — pass device has headroom "
        f"to spare. Shrink W to recover area, or relax the load "
        f"current spec if the design allows."
    )


def _suggest_divide_ratio_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"divide ratio below target — feedback fraction too high. "
            f"Increase Rtop (raises numerator), or decrease Rbot (raises "
            f"denominator weight of Rtop). Verify the divider isn't "
            f"loaded by the error-amp input bias current."
        )
    return (
        f"divide ratio exceeds target — feedback fraction too low. "
        f"Decrease Rtop or increase Rbot. Check for matching offset "
        f"in the poly-resistor pair and inter-digitate the layout."
    )


def _suggest_temp_coeff_adjustment(target: float, actual: float) -> str:
    rel = _relative_error(actual, target)
    if rel < 0:
        return (
            f"PTAT temp coefficient below target — current slope too "
            f"flat. Increase the bipolar emitter-area ratio N (PTAT "
            f"slope scales with ln(N)), reduce the PTAT series resistor "
            f"(raises the ΔVbe / R current), or check that the loop "
            f"is biased above the threshold knee at all corners."
        )
    return (
        f"PTAT temp coefficient exceeds target — current slope too "
        f"steep. Decrease N, increase the series resistor, or add a "
        f"CTAT cancellation leg to flatten the slope."
    )
