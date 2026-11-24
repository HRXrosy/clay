"""
Second-batch (v0.0.90 Round 67) functional-role check functions
(v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 2275-3213.

Six checks:
  * `_check_cascode_helper`          — ro_boost_target vs measured cascode boost
  * `_check_pole_tracking_cap`       — target_pole + cross-corner drift
  * `_check_dual_feedforward_pair`   — gm_primary/secondary + symmetry
  * `_check_loop_filter_pole`        — target_loop_bw + deferred-skip
  * `_check_current_mirror`          — ref_current + per-branch fan-out
  * `_check_diode_load`              — gm_target + effective_R cross-check
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from ._emit import _emit_value_violation
from ._helpers import _as_float, _format_value, _relative_error
from ._suggestions import (
    _suggest_dual_feedforward_adjustment,
    _suggest_effective_R_adjustment,
    _suggest_gm_adjustment,
    _suggest_iref_adjustment,
    _suggest_loop_bw_adjustment,
    _suggest_mirror_adjustment,
    _suggest_pole_adjustment,
    _suggest_pole_tracking_adjustment,
    _suggest_ro_boost_adjustment,
)
from ._violation import _TARGET_ATTR_BY_KIND, InvariantViolation


# ─── v0.0.90 (Round 67) — real check functions for second batch ─────
#
# The six new keywords (cascode_helper, pole_tracking_cap,
# dual_feedforward_pair, loop_filter_pole, current_mirror, diode_load)
# now perform real ngspice OP / AC result comparison against their
# declared target attributes, matching the v0.0.86 first-batch
# (input_transconductor / output_buffer / ...) quality bar:
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
#     a measure_missing_hint that names the supporting ngspice
#     measure(s) needed to land the check.
#
# Two keywords (`loop_filter_pole`, partially) remain deferred because
# the closed-loop PLL testbench infrastructure isn't plumbed yet — the
# check function still resolves an actual value when the LLM supplies a
# direct `loop_bw_<name>` measure (so the check works in test fixtures),
# but falls back to a `skipped + extras.notes.deferred` when no direct
# measure exists rather than attempting an unrealistic open-loop
# approximation. The Round 67 brief flagged this explicitly.


# ─── cascode_helper ──────────────────────────────────────────────


def _check_cascode_helper(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `ro_boost_target` against measured cascode boost.

    `ro_boost_target` is a UNITLESS multiplier (e.g. 100 means the
    cascode should boost ro by 100x vs the bare-stage baseline). The
    runtime resolves the actual boost via these paths:

      1. Per-role direct measure `ro_boost_<role.name>` (the testbench
         emits one already-computed boost ratio).
      2. Derived from `ro_<role.name>` / `ro_<attached_to>` —
         post-cascode output resistance divided by baseline (the
         `attached_to` attribute names the device the cascode is
         stacked on top of).
      3. Derived from `gm_<role.name>` × `ro_<role.name>` ×
         `ro_<attached_to>` — the textbook cascode boost
         `boost ≈ gm_casc * ro_casc` (used when the explicit baseline
         ratio isn't measured but the underlying small-signal params
         are available).

    Returns `measure_missing` when none of the three paths resolves.
    """
    attr = _TARGET_ATTR_BY_KIND["cascode_helper"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"cascode_helper {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `ro_boost_target = <multiplier>` in the role body"
            ),
        )

    attached_to = role.attributes.get("attached_to", "")
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
            # 3) gm_casc * ro_casc textbook boost approximation.
            gm_key = f"gm_{role.name}"
            ro_key = f"ro_{role.name}"
            if (
                gm_key in op_result and ro_key in op_result
                and math.isfinite(op_result[gm_key])
                and math.isfinite(op_result[ro_key])
            ):
                actual = float(op_result[gm_key]) * float(op_result[ro_key])
                source = f"derived {gm_key}*{ro_key}"

    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_ro_boost_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce ro_boost_{role.name!r}, "
            f"ro_{role.name!r}/ro_{attached_to!r}, nor "
            f"gm_{role.name!r}*ro_{role.name!r}; add "
            f".meas op ro_boost_{role.name} from a two-step "
            f"baseline-vs-cascode sweep, or emit per-device "
            f"gm/ro measures"
        ),
    )


# ─── pole_tracking_cap ──────────────────────────────────────────


def _check_pole_tracking_cap(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `target_pole` against measured pole, with optional
    cross-corner drift check via `pole_drift_max`.

    The runtime supports both single-corner and multi-corner verifies:

      * Single-corner path: `pole_<role.name>` in `ac_result` gives the
        single pole value; compared to `target_pole` within tolerance.
      * Multi-corner path: when the AC result carries multiple
        `pole_<role.name>_<corner>` keys (e.g. `pole_Ctk_ss`,
        `pole_Ctk_tt`, `pole_Ctk_ff`), the runtime computes the
        max-min drift across all matching corners as a fraction of the
        mean, and compares it against `pole_drift_max` (declared as a
        fractional drift e.g. 0.15 for ±15%). The reported `actual_value`
        is the worst-case drift.

    The single-corner check uses the target_pole vs measured pole
    tolerance; the multi-corner check tests the drift fraction itself
    against the declared `pole_drift_max` (which is the *bound* —
    actual drift must NOT exceed it).
    """
    attr = _TARGET_ATTR_BY_KIND["pole_tracking_cap"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"pole_tracking_cap {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `target_pole = <value>Hz` in the role body"
            ),
        )

    # Look for any multi-corner pole entries that match the role name
    # prefix. `track_corners` (if declared) is informational — the
    # runtime accepts any corner names present in `ac_result`.
    corner_prefix = f"pole_{role.name}_"
    corner_values: list[tuple[str, float]] = []
    for key, val in ac_result.items():
        if key.startswith(corner_prefix) and isinstance(val, (int, float)):
            v = float(val)
            if math.isfinite(v):
                corner_values.append((key[len(corner_prefix):], v))

    pole_drift_max = _as_float(role.attributes.get("pole_drift_max"))

    # Multi-corner drift path takes priority when the AC result carries
    # cross-corner pole measures (and the role declared the bound).
    if pole_drift_max is not None and len(corner_values) >= 2:
        values = [v for _, v in corner_values]
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        drift_frac = (spread / mean) if mean != 0.0 else 0.0
        # Compare drift fraction to the declared `pole_drift_max` BOUND:
        # actual <= bound is a pass. Use the bound as the "target"
        # for the violation envelope so the LLM sees both numbers.
        if drift_frac <= pole_drift_max * (1.0 + tolerance):
            return InvariantViolation(
                role_kind=role.kind, role_name=role.name, target_attr="pole_drift_max",
                target_value=pole_drift_max, actual_value=drift_frac,
                relative_error=_relative_error(drift_frac, pole_drift_max),
                tolerance=tolerance, category="skipped",
                message=(
                    f"pole_tracking_cap {role.name!r}: cross-corner pole "
                    f"drift={drift_frac*100:.2f}% (over "
                    f"{len(corner_values)} corners) within bound "
                    f"pole_drift_max={pole_drift_max*100:.2f}%"
                ),
                extras={
                    "actual_source": "cross_corner_drift",
                    "corners_observed": [c for c, _ in corner_values],
                    "mean_pole": mean,
                },
            )
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr="pole_drift_max",
            target_value=pole_drift_max, actual_value=drift_frac,
            relative_error=_relative_error(drift_frac, pole_drift_max),
            tolerance=tolerance, category="mismatch",
            message=(
                f"pole_tracking_cap {role.name!r}: cross-corner pole "
                f"drift={drift_frac*100:.2f}% exceeds declared bound "
                f"pole_drift_max={pole_drift_max*100:.2f}% "
                f"(over {len(corner_values)} corners)"
            ),
            suggested_adjustment=_suggest_pole_tracking_adjustment(
                pole_drift_max, drift_frac,
            ),
            extras={
                "actual_source": "cross_corner_drift",
                "corners_observed": [c for c, _ in corner_values],
                "mean_pole": mean,
            },
        )

    # Single-corner path: compare target_pole to a single pole value.
    actual: Optional[float] = None
    source = ""
    keyed = f"pole_{role.name}"
    if keyed in ac_result and math.isfinite(ac_result[keyed]):
        actual = float(ac_result[keyed])
        source = keyed
    elif "pole_freq_dominant" in ac_result and math.isfinite(
        ac_result["pole_freq_dominant"]
    ):
        actual = float(ac_result["pole_freq_dominant"])
        source = "pole_freq_dominant"

    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_pole_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce pole_{role.name!r} (single-corner) "
            f"or pole_{role.name}_<corner> entries (cross-corner). Add "
            f".meas ac pole_{role.name} when find vdb(out) val=-3 to the "
            f"AC testbench, or run per-corner sweeps for drift mode."
        ),
    )


# ─── dual_feedforward_pair ──────────────────────────────────────


def _check_dual_feedforward_pair(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `gm_primary` + `gm_secondary` against measured
    per-path gm values, optionally checking the symmetry ratio.

    The role declares two gm targets (one per feedforward path) and an
    optional `symmetry_target` (defaults to `gm_secondary / gm_primary`
    if absent — i.e. perfect declared symmetry). The check resolves:

      * `gm_primary_<role.name>` from `op_result` for the primary leg gm
      * `gm_secondary_<role.name>` for the secondary leg gm
      * Falls back to plain `gm_<role.name>_primary` / `gm_<role.name>_secondary`
        if the underscore-suffix form isn't present.

    Returns:
      * `skipped` when both legs match their targets AND the symmetry
        ratio matches `symmetry_target` (within tolerance).
      * `mismatch` keyed on the WORST of the three sub-checks (primary
        gm, secondary gm, symmetry).
      * `measure_missing` when either leg's gm can't be resolved.

    The reported `target_value` / `actual_value` correspond to the
    *worst* sub-check so the LLM sees the most important deviation.
    """
    gm_primary_target = _as_float(role.attributes.get("gm_primary"))
    gm_secondary_target = _as_float(role.attributes.get("gm_secondary"))
    if gm_primary_target is None and gm_secondary_target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name,
            target_attr="gm_primary",
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"dual_feedforward_pair {role.name!r}: neither "
                f"'gm_primary' nor 'gm_secondary' declared; runtime "
                f"has no target to check"
            ),
            suggested_adjustment=(
                "declare gm_primary and gm_secondary on the role body"
            ),
        )

    # Resolve per-leg gm.
    def _resolve_leg(leg: str) -> tuple[Optional[float], str]:
        # Try the leg-suffix and prefix forms.
        candidates = [
            f"gm_{leg}_{role.name}",
            f"gm_{role.name}_{leg}",
        ]
        for k in candidates:
            if k in op_result and math.isfinite(op_result[k]):
                return float(op_result[k]), k
        return None, ""

    gm_primary_actual, src_p = _resolve_leg("primary")
    gm_secondary_actual, src_s = _resolve_leg("secondary")

    if gm_primary_actual is None and gm_secondary_actual is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name,
            target_attr="gm_primary",
            target_value=gm_primary_target, actual_value=None,
            relative_error=None,
            tolerance=tolerance, category="measure_missing",
            message=(
                f"dual_feedforward_pair {role.name!r}: gm_primary="
                f"{_format_value(gm_primary_target)} / gm_secondary="
                f"{_format_value(gm_secondary_target)} declared but "
                f"neither leg gm resolved. Add .meas op gm_primary_"
                f"{role.name} find @M<primary>[gm] and the secondary "
                f"counterpart to the testbench."
            ),
            suggested_adjustment=(
                f"emit per-leg gm measures: gm_primary_{role.name} and "
                f"gm_secondary_{role.name}"
            ),
            extras={
                "resolution_attempts": [
                    f"gm_primary_{role.name}", f"gm_{role.name}_primary",
                    f"gm_secondary_{role.name}", f"gm_{role.name}_secondary",
                ],
            },
        )

    # Score each declared target leg, pick the worst.
    sub_errors: list[tuple[str, float, float, float]] = []
    if gm_primary_target is not None and gm_primary_actual is not None:
        rel = _relative_error(gm_primary_actual, gm_primary_target)
        sub_errors.append(("gm_primary", gm_primary_target, gm_primary_actual, rel))
    if gm_secondary_target is not None and gm_secondary_actual is not None:
        rel = _relative_error(gm_secondary_actual, gm_secondary_target)
        sub_errors.append(("gm_secondary", gm_secondary_target, gm_secondary_actual, rel))

    # Symmetry check: actual ratio vs declared / inferred symmetry_target.
    symmetry_target = _as_float(role.attributes.get("symmetry_target"))
    if (
        symmetry_target is None
        and gm_primary_target is not None and gm_secondary_target is not None
        and gm_primary_target != 0.0
    ):
        symmetry_target = gm_secondary_target / gm_primary_target

    sym_actual: Optional[float] = None
    if (
        gm_primary_actual is not None and gm_secondary_actual is not None
        and gm_primary_actual != 0.0
    ):
        sym_actual = gm_secondary_actual / gm_primary_actual
        if symmetry_target is not None:
            sym_rel = _relative_error(sym_actual, symmetry_target)
            sub_errors.append(("symmetry", symmetry_target, sym_actual, sym_rel))

    if not sub_errors:
        # Defensive: target + actual partially resolved but no comparable pair.
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name,
            target_attr="gm_primary",
            target_value=gm_primary_target, actual_value=gm_primary_actual,
            relative_error=None,
            tolerance=tolerance, category="measure_missing",
            message=(
                f"dual_feedforward_pair {role.name!r}: partial leg "
                f"resolution; cannot compute any sub-check."
            ),
            suggested_adjustment="emit both gm_primary and gm_secondary measures",
        )

    # Pick the worst sub-check (largest |relative_error|).
    sub_errors.sort(key=lambda e: -abs(e[3]))
    worst_attr, worst_target, worst_actual, worst_rel = sub_errors[0]
    extras = {
        "actual_source": (
            f"primary={src_p or 'n/a'}, secondary={src_s or 'n/a'}"
        ),
        "sub_errors": [
            {"attr": a, "target": t, "actual": ac, "rel_err": r}
            for (a, t, ac, r) in sub_errors
        ],
        "symmetry_target": symmetry_target,
        "symmetry_actual": sym_actual,
    }

    if abs(worst_rel) <= tolerance:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name,
            target_attr=worst_attr,
            target_value=worst_target, actual_value=worst_actual,
            relative_error=worst_rel,
            tolerance=tolerance, category="skipped",
            message=(
                f"dual_feedforward_pair {role.name!r}: all checks "
                f"within ±{tolerance*100:g}% (worst {worst_attr}="
                f"{_format_value(worst_actual)} vs target "
                f"{_format_value(worst_target)})"
            ),
            extras=extras,
        )

    direction = "exceeds" if worst_rel > 0 else "underperforms"
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name,
        target_attr=worst_attr,
        target_value=worst_target, actual_value=worst_actual,
        relative_error=worst_rel,
        tolerance=tolerance, category="mismatch",
        message=(
            f"dual_feedforward_pair {role.name!r}: worst sub-check "
            f"{worst_attr}={_format_value(worst_target)} but actual="
            f"{_format_value(worst_actual)} (rel_err={worst_rel*100:.1f}%, "
            f"{direction} target by {abs(worst_rel)*100:.1f}%, "
            f"tolerance=±{tolerance*100:g}%)"
        ),
        suggested_adjustment=_suggest_dual_feedforward_adjustment(
            worst_attr, worst_target, worst_actual,
        ),
        extras=extras,
    )


# ─── loop_filter_pole ──────────────────────────────────────────


def _check_loop_filter_pole(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `target_loop_bw` against measured PLL loop bandwidth.

    NOTE (v0.0.90 Round 67): The full closed-loop PLL AC testbench is
    not yet plumbed (charge-pump + VCO + divider chain require their
    own behavioral models and the open-loop response extraction is
    spec'd but not yet implemented). This check function therefore:

      * Resolves a directly-supplied `loop_bw_<role.name>` measure
        when the testbench emits one (typical for hand-written test
        fixtures and the future closed-loop infra).
      * Falls back to `loop_filter_bw` global key when only one PLL
        is present.
      * Surfaces `skipped + extras["deferred"]=True` when no direct
        measure is available — rather than attempting a degenerate
        open-loop approximation that would mislead the LLM.

    The full closed-loop derivation (via the crossover-frequency
    extraction on the loop gain) lands in a follow-up alongside the
    PLL testbench infrastructure (see spec §15.5).
    """
    attr = _TARGET_ATTR_BY_KIND["loop_filter_pole"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"loop_filter_pole {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `target_loop_bw = <value>Hz` in the role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    keyed = f"loop_bw_{role.name}"
    if keyed in ac_result and math.isfinite(ac_result[keyed]):
        actual = float(ac_result[keyed])
        source = keyed
    elif "loop_filter_bw" in ac_result and math.isfinite(
        ac_result["loop_filter_bw"]
    ):
        actual = float(ac_result["loop_filter_bw"])
        source = "loop_filter_bw"

    if actual is None:
        # Honest deferred-path skip: the closed-loop AC testbench
        # required to actually measure PLL loop bandwidth is not yet
        # plumbed. Return `skipped` with `deferred=True` so the LLM
        # sees the check IS aware of the target but explicitly cannot
        # land it on this v0.0.90 simulator. This is preferable to a
        # `measure_missing` (which would imply the testbench should
        # produce the measure and didn't) or a `mismatch` (which would
        # require an actual value the runtime doesn't have).
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"loop_filter_pole {role.name!r}: target_loop_bw="
                f"{_format_value(target)}Hz declared, but the closed-loop "
                f"PLL AC testbench is deferred to a follow-up. Supply a "
                f"direct loop_bw_{role.name} measure in the testbench to "
                f"land the check now."
            ),
            suggested_adjustment=(
                f"emit a direct loop_bw_{role.name} measure from a "
                f"closed-loop PLL AC sweep (charge-pump + VCO + divider "
                f"behavioral models required)"
            ),
            extras={
                "deferred": True,
                "deferred_reason": "PLL closed-loop AC testbench not plumbed",
            },
        )

    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_loop_bw_adjustment,
        measure_missing_hint="",
    )


# ─── current_mirror ────────────────────────────────────────────


def _check_current_mirror(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `ref_current` + per-branch fan-out currents.

    `ref_current` is the reference leg's DC current. The role's
    `fan_out_to` attribute names the downstream branches that should
    each carry `ref_current * multipliers[i]` (multipliers default to
    1.0 per branch when not declared).

    Resolution:
      * Read `i_<role.name>` for the reference leg actual.
      * For each name in `fan_out_to`, read `i_<branch_name>`.
      * Each branch's expected current = ref_current * multiplier_i.
      * Pick the worst-off branch by relative_error for the violation
        envelope; pass when ALL branches are within tolerance.

    Surface conventions follow the v0.0.86 `tail_current_source` check:
    currents compared by absolute value (sign convention varies).
    """
    attr = _TARGET_ATTR_BY_KIND["current_mirror"]
    ref_current = _as_float(role.attributes.get(attr))
    if ref_current is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"current_mirror {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `ref_current = <value>A` in the role body"
            ),
        )

    fan_out = role.attributes.get("fan_out_to", []) or []
    if isinstance(fan_out, str):
        # Defensive: lark may surface a single bare identifier as a
        # plain string when the user forgets the parenthesized list.
        fan_out = [fan_out]
    multipliers = role.attributes.get("multipliers", []) or []
    if not isinstance(multipliers, list):
        multipliers = [multipliers]

    # Reference leg current.
    ref_actual: Optional[float] = None
    ref_source = ""
    for k in (f"i_{role.name}", f"iref_{role.name}", "iref"):
        if k in op_result and math.isfinite(op_result[k]):
            ref_actual = float(op_result[k])
            ref_source = k
            break

    # Per-branch resolution: each branch in fan_out_to gets its own
    # expected/actual entry. Each entry's relative_error is signed.
    branch_records: list[dict] = []
    for i, branch in enumerate(fan_out):
        mult = 1.0
        if i < len(multipliers):
            m_val = _as_float(multipliers[i])
            if m_val is not None:
                mult = m_val
        expected = abs(ref_current) * abs(mult)
        actual: Optional[float] = None
        bsource = ""
        for k in (f"i_{branch}", f"id_{branch}"):
            if k in op_result and math.isfinite(op_result[k]):
                actual = abs(float(op_result[k]))
                bsource = k
                break
        rel: Optional[float] = None
        if actual is not None:
            rel = _relative_error(actual, expected)
        branch_records.append({
            "branch": branch, "multiplier": mult,
            "expected": expected, "actual": actual,
            "rel_err": rel, "source": bsource,
        })

    # If there are NO fan-out branches, fall back to a plain
    # reference-leg-only check (equivalent to tail_current_source).
    if not branch_records:
        if ref_actual is None:
            return InvariantViolation(
                role_kind=role.kind, role_name=role.name, target_attr=attr,
                target_value=ref_current, actual_value=None,
                relative_error=None,
                tolerance=tolerance, category="measure_missing",
                message=(
                    f"current_mirror {role.name!r}: ref_current="
                    f"{_format_value(ref_current)} declared but no "
                    f"`i_{role.name}` or `iref` measure resolved. Add "
                    f".meas op i_{role.name} find i(I{role.name}) at=0."
                ),
                suggested_adjustment=(
                    f"emit i_{role.name} or iref measure in the testbench"
                ),
            )
        rel = _relative_error(abs(ref_actual), abs(ref_current))
        if abs(rel) <= tolerance:
            return InvariantViolation(
                role_kind=role.kind, role_name=role.name, target_attr=attr,
                target_value=abs(ref_current), actual_value=abs(ref_actual),
                relative_error=rel,
                tolerance=tolerance, category="skipped",
                message=(
                    f"current_mirror {role.name!r}: ref_current="
                    f"{_format_value(ref_current)} matches actual="
                    f"{_format_value(ref_actual)} within ±{tolerance*100:g}%"
                ),
                extras={"actual_source": ref_source},
            )
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=abs(ref_current), actual_value=abs(ref_actual),
            relative_error=rel,
            tolerance=tolerance, category="mismatch",
            message=(
                f"current_mirror {role.name!r}: ref_current="
                f"{_format_value(ref_current)} but actual="
                f"{_format_value(ref_actual)} "
                f"(rel_err={rel*100:.1f}%, tolerance=±{tolerance*100:g}%)"
            ),
            suggested_adjustment=_suggest_iref_adjustment(
                abs(ref_current), abs(ref_actual),
            ),
            extras={"actual_source": ref_source},
        )

    # All-branches must resolve OR we surface measure_missing.
    unresolved = [r for r in branch_records if r["actual"] is None]
    if unresolved:
        names = [r["branch"] for r in unresolved]
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=abs(ref_current), actual_value=None,
            relative_error=None,
            tolerance=tolerance, category="measure_missing",
            message=(
                f"current_mirror {role.name!r}: ref_current="
                f"{_format_value(ref_current)} declared with "
                f"{len(branch_records)} fan-out branches, but "
                f"{len(unresolved)} branch current(s) unresolved: "
                f"{', '.join(names)}. Add `.meas op i_<branch>` for each."
            ),
            suggested_adjustment=(
                f"emit i_<branch_name> measures for: {', '.join(names)}"
            ),
            extras={"branches": branch_records},
        )

    # Worst-branch picker.
    branch_records.sort(key=lambda r: -abs(r["rel_err"]))
    worst = branch_records[0]
    worst_rel = float(worst["rel_err"])
    if abs(worst_rel) <= tolerance:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name,
            target_attr=f"i_{worst['branch']}",
            target_value=worst["expected"], actual_value=worst["actual"],
            relative_error=worst_rel,
            tolerance=tolerance, category="skipped",
            message=(
                f"current_mirror {role.name!r}: all "
                f"{len(branch_records)} fan-out branches within "
                f"±{tolerance*100:g}% of ref_current * multiplier "
                f"(worst {worst['branch']!r}: actual="
                f"{_format_value(worst['actual'])} vs "
                f"expected={_format_value(worst['expected'])})"
            ),
            extras={
                "actual_source": worst["source"],
                "ref_current": ref_current,
                "branches": branch_records,
            },
        )

    direction = "exceeds" if worst_rel > 0 else "underperforms"
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name,
        target_attr=f"i_{worst['branch']}",
        target_value=worst["expected"], actual_value=worst["actual"],
        relative_error=worst_rel,
        tolerance=tolerance, category="mismatch",
        message=(
            f"current_mirror {role.name!r}: branch {worst['branch']!r} "
            f"expected={_format_value(worst['expected'])} (ref_current "
            f"* multiplier={worst['multiplier']:g}) but actual="
            f"{_format_value(worst['actual'])} "
            f"(rel_err={worst_rel*100:.1f}%, {direction} target by "
            f"{abs(worst_rel)*100:.1f}%, tolerance=±{tolerance*100:g}%)"
        ),
        suggested_adjustment=_suggest_mirror_adjustment(
            worst["expected"], worst["actual"], worst["branch"],
        ),
        extras={
            "actual_source": worst["source"],
            "ref_current": ref_current,
            "branches": branch_records,
        },
    )


# ─── diode_load ────────────────────────────────────────────────


def _check_diode_load(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `gm_target` (and optional `effective_R_target`)
    against measured per-device gm.

    A diode-connected MOS has effective small-signal resistance
    `R_eff ≈ 1/gm` (in parallel with `ro`, but ro >> 1/gm for typical
    sky130 biasing). The check resolves gm via the same path as
    `_check_input_transconductor` (per-role gm key, then derived from
    id/vov), then optionally cross-checks `1/gm` against
    `effective_R_target` if declared.

    Resolution priority:
      * `gm_target` primary check (mandatory if `effective_R_target`
        is also absent — must have at least one target).
      * `effective_R_target` cross-check (uses 1/gm if gm resolved;
        falls back to a direct `R_<role.name>` measure if testbench
        emits one).
    """
    gm_target = _as_float(role.attributes.get("gm_target"))
    R_target = _as_float(role.attributes.get("effective_R_target"))
    if gm_target is None and R_target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name,
            target_attr="gm_target",
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"diode_load {role.name!r}: neither 'gm_target' nor "
                f"'effective_R_target' declared; runtime has no target "
                f"to check"
            ),
            suggested_adjustment=(
                "declare `gm_target = <value>A/V` and/or "
                "`effective_R_target = <value>Ohm` on the role body"
            ),
        )

    # Resolve gm via the input_transconductor's resolution chain.
    gm_actual: Optional[float] = None
    gm_source = ""
    keyed = f"gm_{role.name}"
    if keyed in op_result and math.isfinite(op_result[keyed]):
        gm_actual = float(op_result[keyed])
        gm_source = keyed
    elif "gm" in op_result and math.isfinite(op_result["gm"]):
        gm_actual = float(op_result["gm"])
        gm_source = "gm"
    else:
        id_key = f"id_{role.name}"
        vov_key = f"vov_{role.name}"
        if (
            id_key in op_result and vov_key in op_result
            and math.isfinite(op_result[id_key])
            and math.isfinite(op_result[vov_key])
            and op_result[vov_key] != 0.0
        ):
            gm_actual = 2.0 * float(op_result[id_key]) / float(op_result[vov_key])
            gm_source = f"derived 2*{id_key}/{vov_key}"

    # Sub-check 1: gm_target vs gm_actual.
    sub_errors: list[tuple[str, float, Optional[float], Optional[float]]] = []
    if gm_target is not None:
        if gm_actual is None:
            return InvariantViolation(
                role_kind=role.kind, role_name=role.name,
                target_attr="gm_target",
                target_value=gm_target, actual_value=None,
                relative_error=None,
                tolerance=tolerance, category="measure_missing",
                message=(
                    f"diode_load {role.name!r}: gm_target="
                    f"{_format_value(gm_target)} declared but ngspice "
                    f"did not produce gm_{role.name}, gm, or id/vov to "
                    f"derive gm. Add `.print op @m{role.name}[gm]`."
                ),
                suggested_adjustment=(
                    f"emit a per-device gm measure gm_{role.name} or "
                    f"supply id_{role.name} + vov_{role.name}"
                ),
                extras={"resolution_attempts": [keyed, "gm", "id/vov"]},
            )
        rel = _relative_error(gm_actual, gm_target)
        sub_errors.append(("gm_target", gm_target, gm_actual, rel))

    # Sub-check 2: effective_R_target. Direct R measure preferred, else
    # 1/gm cross-check.
    R_actual: Optional[float] = None
    R_source = ""
    if R_target is not None:
        R_key = f"R_{role.name}"
        R_eff_key = f"R_eff_{role.name}"
        if R_key in op_result and math.isfinite(op_result[R_key]):
            R_actual = float(op_result[R_key])
            R_source = R_key
        elif R_eff_key in op_result and math.isfinite(op_result[R_eff_key]):
            R_actual = float(op_result[R_eff_key])
            R_source = R_eff_key
        elif gm_actual is not None and gm_actual != 0.0:
            R_actual = 1.0 / gm_actual
            R_source = f"derived 1/{gm_source}"

        if R_actual is None:
            return InvariantViolation(
                role_kind=role.kind, role_name=role.name,
                target_attr="effective_R_target",
                target_value=R_target, actual_value=None,
                relative_error=None,
                tolerance=tolerance, category="measure_missing",
                message=(
                    f"diode_load {role.name!r}: effective_R_target="
                    f"{_format_value(R_target)} declared but no direct "
                    f"R measure and no gm to invert. Add an R-extraction "
                    f"measure or a per-device gm probe."
                ),
                suggested_adjustment=(
                    f"emit R_{role.name} or gm_{role.name} measure"
                ),
            )
        rel = _relative_error(R_actual, R_target)
        sub_errors.append(("effective_R_target", R_target, R_actual, rel))

    if not sub_errors:
        # Defensive: should be unreachable given the early-return guard.
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name,
            target_attr="gm_target",
            target_value=gm_target, actual_value=gm_actual,
            relative_error=None,
            tolerance=tolerance, category="skipped",
            message=f"diode_load {role.name!r}: no comparable target+actual pair",
        )

    sub_errors.sort(key=lambda e: -abs(e[3] or 0.0))
    worst_attr, worst_target, worst_actual, worst_rel = sub_errors[0]
    extras = {
        "actual_source": gm_source if worst_attr == "gm_target" else R_source,
        "sub_errors": [
            {"attr": a, "target": t, "actual": ac, "rel_err": r}
            for (a, t, ac, r) in sub_errors
        ],
        "gm_actual": gm_actual,
        "R_actual": R_actual,
    }

    if abs(worst_rel) <= tolerance:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name,
            target_attr=worst_attr,
            target_value=worst_target, actual_value=worst_actual,
            relative_error=worst_rel,
            tolerance=tolerance, category="skipped",
            message=(
                f"diode_load {role.name!r}: all checks within "
                f"±{tolerance*100:g}% (worst {worst_attr}="
                f"{_format_value(worst_target)} vs actual="
                f"{_format_value(worst_actual)})"
            ),
            extras=extras,
        )

    direction = "exceeds" if worst_rel > 0 else "underperforms"
    suggest_fn = (
        _suggest_gm_adjustment if worst_attr == "gm_target"
        else _suggest_effective_R_adjustment
    )
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name,
        target_attr=worst_attr,
        target_value=worst_target, actual_value=worst_actual,
        relative_error=worst_rel,
        tolerance=tolerance, category="mismatch",
        message=(
            f"diode_load {role.name!r}: {worst_attr}="
            f"{_format_value(worst_target)} but actual="
            f"{_format_value(worst_actual)} (rel_err={worst_rel*100:.1f}%, "
            f"{direction} target by {abs(worst_rel)*100:.1f}%, "
            f"tolerance=±{tolerance*100:g}%)"
        ),
        suggested_adjustment=suggest_fn(worst_target, worst_actual),
        extras=extras,
    )
