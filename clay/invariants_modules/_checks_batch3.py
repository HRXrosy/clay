"""
Third-batch (v0.0.92 Round 68 Agent G+H) functional-role check functions
(v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 3306-3926.

Six checks:
  * `_check_common_mode_feedback`   — cmrr_target (dB) vs measured CMRR
  * `_check_sample_hold_switch`     — hold_droop_target (V/s) vs droop
  * `_check_regulated_cascode`      — ro_boost_target vs measured boost
  * `_check_gain_boost_amp`         — boost_gain_target (dB) vs aux amp
  * `_check_output_class_AB`        — iq_target (A) vs quiescent current
  * `_check_dynamic_bias`           — iref_target + cross-corner drift
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from ._emit import _emit_value_violation
from ._helpers import _as_float, _relative_error
from ._suggestions import (
    _suggest_boost_gain_adjustment,
    _suggest_cmrr_adjustment,
    _suggest_droop_adjustment,
    _suggest_dynamic_bias_adjustment,
    _suggest_dynamic_bias_drift_adjustment,
    _suggest_iq_adjustment,
    _suggest_regulated_cascode_adjustment,
)
from ._violation import _TARGET_ATTR_BY_KIND, InvariantViolation


# ─── v0.0.92 (Round 68) — real check functions for third batch ─────
#
# Six new functional-role keywords (`common_mode_feedback`,
# `sample_hold_switch`, `regulated_cascode`, `gain_boost_amp`,
# `output_class_AB`, `dynamic_bias`) now perform real ngspice OP / AC
# result comparison against their declared target attributes, matching
# the v0.0.86 / v0.0.90 quality bar:
#
#   * Each check resolves the actual quantity from `op_result` /
#     `ac_result` via a documented fallback chain.
#   * A finite actual value within ±tolerance produces `category="skipped"`
#     (informational pass) with `actual_value` populated and an
#     `extras["actual_source"]` recording the resolution path.
#   * An actual value outside the band produces `category="mismatch"`
#     with `relative_error` + a per-keyword structured
#     `suggested_adjustment` hint.
#   * A missing measure produces `category="measure_missing"` carrying
#     a hint that names the supporting ngspice measure(s) needed to
#     land the check.
#
# Semantic notes per keyword:
#
#   common_mode_feedback — `cmrr_target` is the CMFB rejection ratio
#     in dB. Resolution: direct `cmrr_<name>` (dB) measure or derived
#     from `gain_dm_<name>` / `gain_cm_<name>` linear gains via
#     CMRR = 20*log10(|A_dm / A_cm|). Tolerance is applied to the dB
#     value directly (a ±20% tolerance on 60 dB → 48..72 dB; this
#     matches how CMFB targets are typically spec'd).
#
#   sample_hold_switch — `hold_droop_target` is the V/s droop during
#     the hold phase. Resolution: direct `droop_<name>` measure or
#     derived from `delta_vout_<name>` / `delta_t_<name>` pair. The
#     check uses absolute-value comparison (sign convention varies).
#
#   regulated_cascode — `ro_boost_target` is a UNITLESS multiplier.
#     Resolution: direct `ro_boost_<name>`, ratio
#     `ro_<name>/ro_<attached_to>`, or textbook derivation
#     `gm_<name>*ro_<name>*A_boost` (A_boost from `boost_gain_<boost_amp>`
#     when declared, defaulting to 1). The boost amplifier itself
#     is checked separately by `gain_boost_amp`.
#
#   gain_boost_amp — `boost_gain_target` is the auxiliary amp's AC
#     open-loop gain in dB. Resolution mirrors `output_buffer` but
#     allows the role to declare the target in dB; the resolver
#     converts dB to linear once and compares in linear space (a 20%
#     tolerance on 100 V/V is the canonical convention; dB tolerance
#     would be far too loose).
#
#   output_class_AB — `iq_target` is the quiescent (idle-state) current
#     drawn through the push or pull device. Resolution: direct
#     `iq_<name>`, `i_<push_device>` from attribute, or `id_<push_device>`.
#     Tolerance comparison uses absolute values.
#
#   dynamic_bias — `iref_target` is the tracking bias output current
#     in A. Resolution: direct `i_<name>` (single corner) or per-corner
#     `iref_<name>_<corner>` (cross-corner check) with optional
#     `tracking_drift_max` bound on the spread/mean fraction across
#     observed corners (analogous to `pole_tracking_cap`'s drift mode).


# ─── common_mode_feedback ──────────────────────────────────────────


def _check_common_mode_feedback(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `cmrr_target` (dB) against measured CMRR.

    Resolution order:
      1. Direct `cmrr_<role.name>` from `ac_result` (already in dB).
      2. Derived from per-leg differential/common-mode gain pair
         `gain_dm_<role.name>` / `gain_cm_<role.name>` via
         `CMRR_dB = 20*log10(|A_dm / A_cm|)`. Both keys are expected
         in LINEAR V/V.
      3. Global `cmrr` (dB) fallback when only one CMFB is present.

    The check operates in dB space: target and actual are both dB
    quantities, and the tolerance band applies to the dB number
    directly. A ±20% tolerance on 60 dB allows 48..72 dB — the
    canonical convention for CMRR specs.
    """
    attr = _TARGET_ATTR_BY_KIND["common_mode_feedback"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"common_mode_feedback {role.name!r}: no {attr!r} "
                f"declared; runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `cmrr_target = <value>dB` in the role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    keyed = f"cmrr_{role.name}"
    # 1) Per-role direct CMRR measure (dB).
    if keyed in ac_result and math.isfinite(ac_result[keyed]):
        actual = float(ac_result[keyed])
        source = keyed
    else:
        # 2) Derive from differential and common-mode gain pair.
        gdm_key = f"gain_dm_{role.name}"
        gcm_key = f"gain_cm_{role.name}"
        if (
            gdm_key in ac_result and gcm_key in ac_result
            and math.isfinite(ac_result[gdm_key])
            and math.isfinite(ac_result[gcm_key])
            and ac_result[gcm_key] != 0.0
        ):
            ratio = abs(float(ac_result[gdm_key])) / abs(float(ac_result[gcm_key]))
            if ratio > 0.0:
                actual = 20.0 * math.log10(ratio)
                source = f"derived 20*log10(|{gdm_key}|/|{gcm_key}|)"
        # 3) Global cmrr fallback.
        if actual is None and "cmrr" in ac_result and math.isfinite(
            ac_result["cmrr"]
        ):
            actual = float(ac_result["cmrr"])
            source = "cmrr"

    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_cmrr_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce cmrr_{role.name!r} (dB), "
            f"gain_dm_{role.name!r}/gain_cm_{role.name!r} (V/V pair), "
            f"nor a global `cmrr`. Add `.meas ac cmrr_{role.name} "
            f"param='20*log10(vdb(out_dm)/vdb(out_cm))'` or run a "
            f"common-mode AC loop testbench."
        ),
    )


# ─── sample_hold_switch ────────────────────────────────────────────


def _check_sample_hold_switch(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `hold_droop_target` (V/s) against measured droop.

    Resolution order:
      1. Direct `droop_<role.name>` (V/s) from `op_result`. The
         transient testbench is expected to emit a single droop figure
         (typically `(v(hold)_end - v(hold)_start) / hold_duration`).
      2. Derived from `delta_vout_<role.name>` (V) and
         `delta_t_<role.name>` (s) via `delta_vout / delta_t`.
      3. Global `droop` fallback when only one sample/hold cell exists.

    Comparisons use absolute values — droop sign depends on whether
    the cap leakage is positive or negative and which switch polarity
    the testbench probes. The spec is on magnitude.
    """
    attr = _TARGET_ATTR_BY_KIND["sample_hold_switch"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"sample_hold_switch {role.name!r}: no {attr!r} "
                f"declared; runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `hold_droop_target = <value>V/s` in the role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    keyed = f"droop_{role.name}"
    # 1) Direct droop measure.
    if keyed in op_result and math.isfinite(op_result[keyed]):
        actual = abs(float(op_result[keyed]))
        source = keyed
    else:
        # 2) Derived from delta_vout / delta_t pair.
        dv_key = f"delta_vout_{role.name}"
        dt_key = f"delta_t_{role.name}"
        if (
            dv_key in op_result and dt_key in op_result
            and math.isfinite(op_result[dv_key])
            and math.isfinite(op_result[dt_key])
            and op_result[dt_key] != 0.0
        ):
            actual = abs(float(op_result[dv_key]) / float(op_result[dt_key]))
            source = f"derived {dv_key}/{dt_key}"
        # 3) Global droop fallback.
        elif "droop" in op_result and math.isfinite(op_result["droop"]):
            actual = abs(float(op_result["droop"]))
            source = "droop"

    return _emit_value_violation(
        role=role, attr=attr, target=abs(target), actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_droop_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce droop_{role.name!r} (V/s) or "
            f"delta_vout_{role.name!r}/delta_t_{role.name!r} pair. "
            f"Add `.meas tran droop_{role.name} param='(v(sh_end)-"
            f"v(sh_start))/(t_end-t_start)'` to the transient "
            f"sample/hold testbench."
        ),
    )


# ─── regulated_cascode ─────────────────────────────────────────────


def _check_regulated_cascode(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `ro_boost_target` against measured cascode boost.

    `ro_boost_target` is a UNITLESS multiplier (the auxiliary-amp-
    boosted output resistance divided by the baseline-stage ro).
    Resolution mirrors `cascode_helper` but accounts for the boost amp:

      1. Direct `ro_boost_<role.name>` (the testbench emits one
         already-computed boost ratio from a two-sweep comparison).
      2. Derived from `ro_<role.name> / ro_<attached_to>` — the actual
         small-signal output resistance ratio.
      3. Textbook derivation `gm_<role.name> * ro_<role.name> *
         A_boost` where `A_boost` comes from
         `boost_gain_<boost_amp>` (the auxiliary amp's open-loop
         gain in linear V/V). Defaults to 1 when not measured (the
         "raw cascode boost" baseline).

    Returns `measure_missing` when none of the three paths resolves.
    """
    attr = _TARGET_ATTR_BY_KIND["regulated_cascode"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"regulated_cascode {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `ro_boost_target = <multiplier>` in the role body"
            ),
        )

    attached_to = role.attributes.get("attached_to", "")
    boost_amp = role.attributes.get("boost_amp", "")
    actual: Optional[float] = None
    source = ""

    # 1) Per-role direct boost measure.
    direct = f"ro_boost_{role.name}"
    if direct in op_result and math.isfinite(op_result[direct]):
        actual = float(op_result[direct])
        source = direct
    elif direct in ac_result and math.isfinite(ac_result[direct]):
        actual = float(ac_result[direct])
        source = direct
    else:
        # 2) ro(post)/ro(base) ratio.
        ro_self = f"ro_{role.name}"
        ro_base = f"ro_{attached_to}" if attached_to else ""
        if (
            ro_self in op_result and ro_base and ro_base in op_result
            and math.isfinite(op_result[ro_self])
            and math.isfinite(op_result[ro_base])
            and op_result[ro_base] != 0.0
        ):
            actual = float(op_result[ro_self]) / float(op_result[ro_base])
            source = f"derived {ro_self}/{ro_base}"
        else:
            # 3) gm*ro*A_boost textbook derivation.
            gm_key = f"gm_{role.name}"
            ro_key = f"ro_{role.name}"
            if (
                gm_key in op_result and ro_key in op_result
                and math.isfinite(op_result[gm_key])
                and math.isfinite(op_result[ro_key])
            ):
                a_boost = 1.0
                boost_src = "default(1)"
                if boost_amp:
                    boost_key_lin = f"boost_gain_{boost_amp}"
                    boost_key_db = f"gain_dc_{boost_amp}"
                    if (
                        boost_key_lin in ac_result
                        and math.isfinite(ac_result[boost_key_lin])
                    ):
                        a_boost = abs(float(ac_result[boost_key_lin]))
                        boost_src = boost_key_lin
                    elif (
                        boost_key_db in ac_result
                        and math.isfinite(ac_result[boost_key_db])
                    ):
                        a_boost = 10.0 ** (float(ac_result[boost_key_db]) / 20.0)
                        boost_src = f"{boost_key_db}(dB→lin)"
                actual = (
                    float(op_result[gm_key])
                    * float(op_result[ro_key])
                    * a_boost
                )
                source = f"derived {gm_key}*{ro_key}*A_boost({boost_src})"

    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_regulated_cascode_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce ro_boost_{role.name!r}, "
            f"ro_{role.name!r}/ro_{attached_to!r}, nor "
            f"gm_{role.name!r}*ro_{role.name!r}. Add a "
            f".meas op ro_boost_{role.name} from a two-step sweep "
            f"(boost-amp on vs off), or emit per-device gm/ro "
            f"measures and a boost_gain_{boost_amp or '<amp>'} value."
        ),
    )


# ─── gain_boost_amp ────────────────────────────────────────────────


def _check_gain_boost_amp(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `boost_gain_target` (dB) against the auxiliary
    amp's AC gain.

    The role's target is given in dB (the conventional way to spec a
    gain-boost auxiliary amplifier — typical values 30..60 dB). The
    runtime resolves a LINEAR actual via the standard AC gain
    resolution chain and converts both target (dB→lin) and actual
    (already linear) to the same space before applying the tolerance
    band. This matches the `output_buffer` convention: tolerance
    applies to V/V, not dB.

    Resolution order:
      1. `gain_<role.name>` (linear V/V) — direct measure.
      2. `gain_dc_<role.name>` (dB) — direct measure, dB→lin.
      3. Global `gain_dc` (dB) fallback when only one boost amp is
         present.
    """
    attr = _TARGET_ATTR_BY_KIND["gain_boost_amp"]
    target_db = _as_float(role.attributes.get(attr))
    if target_db is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"gain_boost_amp {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `boost_gain_target = <value>dB` in the role body"
            ),
        )

    target_lin = 10.0 ** (target_db / 20.0)

    actual: Optional[float] = None
    source = ""
    # 1) Per-role linear gain measure.
    keyed_lin = f"gain_{role.name}"
    if keyed_lin in ac_result and math.isfinite(ac_result[keyed_lin]):
        actual = float(ac_result[keyed_lin])
        source = keyed_lin
    else:
        # 2) Per-role gain in dB.
        keyed_db = f"gain_dc_{role.name}"
        if keyed_db in ac_result and math.isfinite(ac_result[keyed_db]):
            actual = 10.0 ** (float(ac_result[keyed_db]) / 20.0)
            source = f"{keyed_db}(dB→lin)"
        # 3) Global gain_dc fallback.
        elif "gain_dc" in ac_result and math.isfinite(ac_result["gain_dc"]):
            actual = 10.0 ** (float(ac_result["gain_dc"]) / 20.0)
            source = "gain_dc(dB→lin)"

    return _emit_value_violation(
        role=role, attr=attr, target=target_lin, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_boost_gain_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce gain_{role.name!r} (linear V/V) "
            f"or gain_dc_{role.name!r} (dB) on the auxiliary amp. Add "
            f".meas ac gain_{role.name} find vdb(boost_out) at=1k to "
            f"the auxiliary-amp AC testbench."
        ),
    )


# ─── output_class_AB ───────────────────────────────────────────────


def _check_output_class_AB(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `iq_target` (A) against quiescent current.

    `iq_target` is the IDLE-STATE current drawn through the push (or
    pull) device of a class-AB output stage. In a properly biased
    class-AB stage |I_push| ≈ |I_pull| ≈ I_q. The runtime resolves
    the actual via these paths:

      1. Direct `iq_<role.name>` measure.
      2. `id_<push_device>` (or its absolute value) where `push_device`
         is declared on the role body.
      3. `id_<pull_device>` (same idea) as a secondary fallback.
      4. Global `iq` measure.

    All comparisons use absolute values — the testbench convention for
    push device drain current sign varies (some testbenches probe
    `i(Vdd, Mpush)`, others `i(out, Mpush)`). The class-AB spec is
    on magnitude.
    """
    attr = _TARGET_ATTR_BY_KIND["output_class_AB"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"output_class_AB {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `iq_target = <value>A` in the role body"
            ),
        )

    push_dev = role.attributes.get("push_device", "")
    pull_dev = role.attributes.get("pull_device", "")
    actual: Optional[float] = None
    source = ""

    # 1) Direct iq measure.
    candidates: list[str] = [f"iq_{role.name}"]
    if push_dev:
        # Push-device current resolution: prefer dedicated `i_<push>` then
        # transistor drain `id_<push>`.
        candidates.extend([f"i_{push_dev}", f"id_{push_dev}"])
    if pull_dev:
        candidates.extend([f"i_{pull_dev}", f"id_{pull_dev}"])
    candidates.append("iq")

    for key in candidates:
        if key in op_result and math.isfinite(op_result[key]):
            actual = abs(float(op_result[key]))
            source = key
            break

    return _emit_value_violation(
        role=role, attr=attr, target=abs(target), actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_iq_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce iq_{role.name!r}, "
            f"i_{push_dev or '<push>'} / id_{push_dev or '<push>'}, "
            f"nor a global `iq`. Add `.meas op iq_{role.name} "
            f"find abs(i(Mpush)) at=0` (with the class-AB stage in "
            f"its idle bias point)."
        ),
    )


# ─── dynamic_bias ──────────────────────────────────────────────────


def _check_dynamic_bias(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `iref_target` (A) against measured bias current,
    with optional cross-corner tracking drift check.

    The runtime supports both single-corner and multi-corner verifies:

      * Single-corner path: `i_<role.name>` (or `iref_<role.name>`)
        in `op_result` gives a single current value; compared to
        `iref_target` within tolerance using absolute values.
      * Multi-corner path: when `op_result` carries multiple
        `iref_<role.name>_<corner>` keys (e.g. `iref_Idyn_ss`,
        `iref_Idyn_tt`, `iref_Idyn_ff`), the runtime computes the
        max-min drift fraction across observed corners. If the role
        declares `tracking_drift_max` (e.g. 0.10 for ±10% PVT tracking)
        the drift is compared against the bound; otherwise drift is
        reported informationally and the comparison falls back to the
        single-corner mean.

    Returns `measure_missing` only when neither path resolves.
    """
    attr = _TARGET_ATTR_BY_KIND["dynamic_bias"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"dynamic_bias {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `iref_target = <value>A` in the role body"
            ),
        )

    # Collect cross-corner per-corner currents (canonical key form
    # `iref_<role.name>_<corner>`).
    corner_prefix = f"iref_{role.name}_"
    corner_values: list[tuple[str, float]] = []
    for key, val in op_result.items():
        if key.startswith(corner_prefix) and isinstance(val, (int, float)):
            v = float(val)
            if math.isfinite(v):
                corner_values.append((key[len(corner_prefix):], v))

    tracking_drift_max = _as_float(role.attributes.get("tracking_drift_max"))

    # Cross-corner drift path: requires the role declared the bound
    # AND at least 2 corners are observed in the OP result.
    if tracking_drift_max is not None and len(corner_values) >= 2:
        values = [abs(v) for _, v in corner_values]
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        drift_frac = (spread / mean) if mean != 0.0 else 0.0
        if drift_frac <= tracking_drift_max * (1.0 + tolerance):
            return InvariantViolation(
                role_kind=role.kind, role_name=role.name,
                target_attr="tracking_drift_max",
                target_value=tracking_drift_max, actual_value=drift_frac,
                relative_error=_relative_error(drift_frac, tracking_drift_max),
                tolerance=tolerance, category="skipped",
                message=(
                    f"dynamic_bias {role.name!r}: cross-corner bias "
                    f"drift={drift_frac*100:.2f}% (over "
                    f"{len(corner_values)} corners) within bound "
                    f"tracking_drift_max={tracking_drift_max*100:.2f}%"
                ),
                extras={
                    "actual_source": "cross_corner_drift",
                    "corners_observed": [c for c, _ in corner_values],
                    "mean_iref": mean,
                    "iref_target": target,
                },
            )
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name,
            target_attr="tracking_drift_max",
            target_value=tracking_drift_max, actual_value=drift_frac,
            relative_error=_relative_error(drift_frac, tracking_drift_max),
            tolerance=tolerance, category="mismatch",
            message=(
                f"dynamic_bias {role.name!r}: cross-corner bias drift "
                f"{drift_frac*100:.2f}% exceeds declared bound "
                f"tracking_drift_max={tracking_drift_max*100:.2f}% "
                f"(over {len(corner_values)} corners)"
            ),
            suggested_adjustment=_suggest_dynamic_bias_drift_adjustment(
                tracking_drift_max, drift_frac,
            ),
            extras={
                "actual_source": "cross_corner_drift",
                "corners_observed": [c for c, _ in corner_values],
                "mean_iref": mean,
                "iref_target": target,
            },
        )

    # Single-corner path: read `i_<name>` / `iref_<name>` / `iref` (or
    # fall back to the cross-corner mean if any corner samples
    # exist).
    actual: Optional[float] = None
    source = ""
    for key in (f"i_{role.name}", f"iref_{role.name}", "iref"):
        if key in op_result and math.isfinite(op_result[key]):
            actual = abs(float(op_result[key]))
            source = key
            break
    if actual is None and corner_values:
        values = [abs(v) for _, v in corner_values]
        actual = sum(values) / len(values)
        source = f"mean over {len(corner_values)} corners"

    return _emit_value_violation(
        role=role, attr=attr, target=abs(target), actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_dynamic_bias_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce i_{role.name!r} (single-corner), "
            f"iref_{role.name!r}, nor any iref_{role.name}_<corner> "
            f"keys (cross-corner). Add `.meas op i_{role.name} "
            f"find i(Idyn) at=0`, or run per-corner OP sweeps for "
            f"tracking-drift mode."
        ),
    )
