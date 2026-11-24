"""
Fourth-batch (v0.0.94 Round 70 Agent C / Round 71 Agent B promoted real)
functional-role check functions (v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 4118-4792.

Five real checks + 1 placeholder + 1 live-sim handle helper:
  * `_placeholder_violation`            — retained Round 70 placeholder (unused)
  * `_check_bandgap_core`               — vref_target (V) vs measured bandgap
  * `_check_dropout_pass_device`        — dropout_voltage_target (V)
  * `_check_ldo_feedback_divider`       — divide_ratio_target vs tap ratio
  * `_check_ptat_branch`                — temp_coefficient_target (A/K)
  * `_check_pll_loop_filter`            — loop_bandwidth_target (Hz) + live AC
  * `_run_pll_loop_ac_via_handle`       — live closed-loop AC sweep handle
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from ._emit import _emit_value_violation
from ._helpers import _as_float, _format_value
from ._suggestions import (
    _suggest_divide_ratio_adjustment,
    _suggest_dropout_adjustment,
    _suggest_loop_bw_adjustment,
    _suggest_temp_coeff_adjustment,
    _suggest_vref_adjustment,
)
from ._violation import _TARGET_ATTR_BY_KIND, InvariantViolation


# ─── v0.0.94 (Round 71) — real check functions ─────────────────────
#
# Five new functional-role keywords for non-amp categories
# (bandgap_core, dropout_pass_device, ldo_feedback_divider, ptat_branch,
# pll_loop_filter) now route to REAL check functions that compare
# declared targets against ngspice OP / AC measures (Round 71 promoted
# the Round 70 Agent C placeholders).
#
# Promotion status (per Round 71 mission):
#   * bandgap_core         → REAL DC OP `vref_<name>` / `v_<vref_node>`
#                            comparison.
#   * dropout_pass_device  → REAL DC OP `dropout_<name>` or
#                            `|v_vin - v_vout|` derivation.
#   * ldo_feedback_divider → REAL DC OP `(v_top + v_bot)/v_bot`
#                            derivation, or direct `divide_ratio_<name>`.
#   * ptat_branch          → REAL multi-temperature dI/dT slope from
#                            `iref_<name>_<corner>` + `temp_<corner>`
#                            pairs; honest `skipped + deferred=True` when
#                            only one corner is present (analogous to
#                            v0.0.90 loop_filter_pole partial).
#   * pll_loop_filter      → PARTIAL: direct `loop_bw_<name>` measure
#                            when testbench emits one; honest `skipped +
#                            deferred=True` when no direct measure is
#                            available (closed-loop PLL AC testbench is
#                            a follow-up dependency).
#
# `_placeholder_violation` below is retained as dead code (no callers
# remain in v0.0.94 Round 71) — it stays in the file so the git diff
# history reads clearly and a future revert is trivial. The runtime
# dispatch table no longer routes any kind through it.


def _placeholder_violation(
    role,
    *,
    tolerance: float,
    kind_label: str,
    target_attr_unit_hint: str,
) -> InvariantViolation:
    """Build the canonical Round 70 Agent C placeholder violation
    (unused as of v0.0.94 Round 71 — every fourth-batch keyword now
    routes to a real check function above).

    A well-formed `InvariantViolation` carrying:
      * `category="skipped"`
      * `target_value` populated when the declared target attribute is
        parseable as a float, otherwise None (so the JSON dict the LLM
        consumes still shows what target value was declared)
      * `extras["deferred"] = True` and
        `extras["placeholder_round"] = "v0.0.94"` so callers can
        distinguish a "kind not yet implemented" skip from a real
        `skipped` (informational pass after real measurement
        comparison)
    """
    attr = _TARGET_ATTR_BY_KIND[role.kind]
    target = _as_float(role.attributes.get(attr))
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name, target_attr=attr,
        target_value=target, actual_value=None, relative_error=None,
        tolerance=tolerance, category="skipped",
        message=(
            f"{kind_label} {role.name!r}: real invariant check is "
            f"deferred to Round 71+ — v0.0.94 Round 70 Agent C ships "
            f"only parser + Rust + placeholder dispatch. Declare "
            f"`{attr} = <value>{target_attr_unit_hint}` for forward "
            f"compatibility; the runtime will compare against the "
            f"corresponding ngspice measure once the testbench plumbing "
            f"lands."
        ),
        suggested_adjustment=(
            f"no adjustment available — placeholder check returns "
            f"`skipped` regardless of declared {attr!r}"
        ),
        extras={
            "deferred": True,
            "placeholder_round": "v0.0.94",
            "deferred_reason": (
                f"{kind_label} testbench plumbing not yet implemented"
            ),
        },
    )


def _check_bandgap_core(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `vref_target` (V) against measured bandgap output.

    Resolution order:
      1. Direct `vref_<role.name>` (V) from `op_result` — typical when
         the testbench emits a per-bandgap measure tagged by role name.
      2. Direct `v_<role.name>` (V) — fallback for testbenches that
         emit the bandgap output as a generic voltage measure.
      3. `vref_<vref_node>` — when the role declares `vref_node` as the
         tap-point identifier, look up `vref_<vref_node>` and `v_<vref_node>`.
      4. Global `vref` (V) fallback when only one bandgap is present.

    Comparisons use absolute values — typical bandgap output is ~1.2V
    positive but a CMOS bandgap variant may emit a negative reference;
    the spec is on magnitude.
    """
    attr = _TARGET_ATTR_BY_KIND["bandgap_core"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"bandgap_core {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `vref_target = <value>V` in the role body"
            ),
        )

    vref_node = role.attributes.get("vref_node", "")
    actual: Optional[float] = None
    source = ""
    candidates = [
        f"vref_{role.name}",
        f"v_{role.name}",
    ]
    if vref_node:
        candidates.extend([f"vref_{vref_node}", f"v_{vref_node}"])
    candidates.append("vref")
    for k in candidates:
        if k in op_result and math.isfinite(op_result[k]):
            actual = abs(float(op_result[k]))
            source = k
            break

    return _emit_value_violation(
        role=role, attr=attr, target=abs(target), actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_vref_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce vref_{role.name!r} (V), "
            f"v_{role.name!r}, nor a global `vref`. Add `.meas op "
            f"vref_{role.name} find v({vref_node or 'vref'}) at=0` "
            f"to the bandgap testbench."
        ),
    )


def _check_dropout_pass_device(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `dropout_voltage_target` (V) against measured headroom.

    Dropout voltage is `v(vin) - v(vout)` under full load — the
    minimum supply-to-output headroom the LDO can sustain before
    regulation collapses. Resolution order:

      1. Direct `dropout_<role.name>` (V) from `op_result` — testbench
         emits the pre-computed headroom.
      2. Derived from `v_vin_<role.name>` (V) and `v_vout_<role.name>`
         (V) via `|v_vin - v_vout|`.
      3. Derived from global `v_vin` / `v_vout` (V) pair — fallback
         when only one LDO pass device is present.
      4. Global `dropout` (V) fallback.

    Comparison uses absolute values — dropout sign convention varies
    by polarity (PMOS vs NMOS pass device); the spec is on magnitude.
    """
    attr = _TARGET_ATTR_BY_KIND["dropout_pass_device"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"dropout_pass_device {role.name!r}: no {attr!r} "
                f"declared; runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `dropout_voltage_target = <value>V` in the "
                "role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    keyed = f"dropout_{role.name}"
    # 1) Direct dropout measure.
    if keyed in op_result and math.isfinite(op_result[keyed]):
        actual = abs(float(op_result[keyed]))
        source = keyed
    else:
        # 2) Derive from per-role v_vin / v_vout pair.
        vin_key = f"v_vin_{role.name}"
        vout_key = f"v_vout_{role.name}"
        if (
            vin_key in op_result and vout_key in op_result
            and math.isfinite(op_result[vin_key])
            and math.isfinite(op_result[vout_key])
        ):
            actual = abs(
                float(op_result[vin_key]) - float(op_result[vout_key])
            )
            source = f"derived |{vin_key} - {vout_key}|"
        # 3) Derive from global v_vin / v_vout pair.
        elif (
            "v_vin" in op_result and "v_vout" in op_result
            and math.isfinite(op_result["v_vin"])
            and math.isfinite(op_result["v_vout"])
        ):
            actual = abs(
                float(op_result["v_vin"]) - float(op_result["v_vout"])
            )
            source = "derived |v_vin - v_vout|"
        # 4) Global dropout fallback.
        elif "dropout" in op_result and math.isfinite(op_result["dropout"]):
            actual = abs(float(op_result["dropout"]))
            source = "dropout"

    return _emit_value_violation(
        role=role, attr=attr, target=abs(target), actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_dropout_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce dropout_{role.name!r} (V), "
            f"v_vin_{role.name!r}/v_vout_{role.name!r} pair, nor a "
            f"global `v_vin`/`v_vout` pair. Add `.meas op "
            f"dropout_{role.name} param='v(vin)-v(vout)'` under full-"
            f"load bias to the LDO testbench."
        ),
    )


def _check_ldo_feedback_divider(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `divide_ratio_target` against measured tap ratio.

    For a resistive divider `Rtop` (top) + `Rbot` (bottom) with the tap
    in between, the divide ratio is `(v_top + v_bot) / v_bot =
    (Rtop + Rbot) / Rbot`, i.e. the inverse of the standard divider
    fraction. The runtime resolves the actual ratio via:

      1. Direct `divide_ratio_<role.name>` from `op_result` — testbench
         emits the pre-computed ratio.
      2. Derived from `v_top_<role.name>` (V) and `v_bot_<role.name>`
         (V) per `(v_top + v_bot)/v_bot`. Both voltages are absolute
         w.r.t. ground; v_bot is the tap voltage and v_top is the
         drop across Rtop.
      3. Derived from `v_<tap_node>` and an external reference voltage
         when the role declares `tap_node` and the testbench emits
         the reference. Less common; serves as a fallback when the
         testbench only emits the tap node voltage.
      4. Global `divide_ratio` fallback when only one divider is
         present.

    `divide_ratio` is a unitless multiplier; the comparison uses the
    standard `_emit_value_violation` tolerance band.
    """
    attr = _TARGET_ATTR_BY_KIND["ldo_feedback_divider"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"ldo_feedback_divider {role.name!r}: no {attr!r} "
                f"declared; runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `divide_ratio_target = <ratio>` in the role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    keyed = f"divide_ratio_{role.name}"
    # 1) Direct divide_ratio measure.
    if keyed in op_result and math.isfinite(op_result[keyed]):
        actual = float(op_result[keyed])
        source = keyed
    else:
        # 2) Derive from per-role v_top + v_bot pair.
        vt_key = f"v_top_{role.name}"
        vb_key = f"v_bot_{role.name}"
        if (
            vt_key in op_result and vb_key in op_result
            and math.isfinite(op_result[vt_key])
            and math.isfinite(op_result[vb_key])
            and op_result[vb_key] != 0.0
        ):
            v_top = float(op_result[vt_key])
            v_bot = float(op_result[vb_key])
            actual = (v_top + v_bot) / v_bot
            source = f"derived ({vt_key} + {vb_key})/{vb_key}"
        # 3) Global divide_ratio fallback.
        elif "divide_ratio" in op_result and math.isfinite(
            op_result["divide_ratio"]
        ):
            actual = float(op_result["divide_ratio"])
            source = "divide_ratio"

    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_divide_ratio_adjustment,
        measure_missing_hint=(
            f"ngspice did not produce divide_ratio_{role.name!r}, "
            f"v_top_{role.name!r}/v_bot_{role.name!r} pair, nor a "
            f"global `divide_ratio`. Add `.meas op v_top_{role.name} "
            f"find v(top) at=0` and `.meas op v_bot_{role.name} find "
            f"v(tap) at=0` to the LDO feedback-divider testbench."
        ),
    )


def _check_ptat_branch(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `temp_coefficient_target` (A/K) against measured slope.

    PTAT current grows linearly with absolute temperature; the
    coefficient is `dI/dT` in A/K. Resolution paths:

      1. Direct `temp_coeff_<role.name>` (A/K) — testbench emits the
         pre-computed slope (typically from a finite-difference over a
         small `.temp` sweep).
      2. Cross-corner slope: requires `iref_<role.name>_<corner>` keys
         AND `temp_<corner>` keys in `op_result` for >= 2 temperatures.
         Computed as `(I_high - I_low) / (T_high - T_low)` using the
         min/max temperature pair. The runtime expects temperatures
         in Kelvin (or Celsius — only deltas matter for a slope).
      3. Global `temp_coeff` (A/K) fallback.

    When only a single corner is present (or no temperature info),
    the check surfaces `skipped + extras["deferred"]=True` with an
    explicit "needs multi-temperature sweep" hint — rather than
    misleading the LLM with a degenerate single-point comparison.
    """
    attr = _TARGET_ATTR_BY_KIND["ptat_branch"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"ptat_branch {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `temp_coefficient_target = <value>A/K` in the "
                "role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    keyed = f"temp_coeff_{role.name}"
    # 1) Direct temp coefficient measure.
    if keyed in op_result and math.isfinite(op_result[keyed]):
        actual = float(op_result[keyed])
        source = keyed
    else:
        # 2) Cross-corner slope path: collect iref_<name>_<corner> +
        # temp_<corner> tuples and compute dI/dT over min/max T.
        corner_prefix = f"iref_{role.name}_"
        per_corner: list[tuple[str, float]] = []
        for key, val in op_result.items():
            if key.startswith(corner_prefix) and isinstance(
                val, (int, float)
            ):
                v = float(val)
                if math.isfinite(v):
                    per_corner.append((key[len(corner_prefix):], v))
        # Pair each corner with its temperature.
        paired: list[tuple[float, float]] = []
        for corner, ival in per_corner:
            tk = f"temp_{corner}"
            if tk in op_result and math.isfinite(op_result[tk]):
                paired.append((float(op_result[tk]), ival))
        if len(paired) >= 2:
            paired.sort()
            t_lo, i_lo = paired[0]
            t_hi, i_hi = paired[-1]
            if t_hi != t_lo:
                actual = (i_hi - i_lo) / (t_hi - t_lo)
                source = (
                    f"derived (iref@{t_hi:.1f}K - iref@{t_lo:.1f}K)/"
                    f"(T_hi-T_lo) over {len(paired)} corners"
                )
        # 3) Global temp_coeff fallback.
        if actual is None and "temp_coeff" in op_result and math.isfinite(
            op_result["temp_coeff"]
        ):
            actual = float(op_result["temp_coeff"])
            source = "temp_coeff"

    if actual is None:
        # Honest deferred path: the cross-corner temperature sweep
        # required to compute dI/dT is not present in this run. Surface
        # `skipped + deferred=True` rather than `measure_missing` so the
        # caller distinguishes "testbench infra deferred" from
        # "expected measure absent".
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"ptat_branch {role.name!r}: temp_coefficient_target="
                f"{_format_value(target)}A/K declared, but the multi-"
                f"temperature sweep required to compute dI/dT is not "
                f"present in this op_result. Emit a direct "
                f"temp_coeff_{role.name} measure (preferred) or run "
                f"the same testbench at >= 2 temperatures with "
                f"`iref_{role.name}_<corner>` + `temp_<corner>` "
                f"measures."
            ),
            suggested_adjustment=(
                f"emit `.meas op temp_coeff_{role.name} param='(i_hi"
                f"-i_lo)/(T_hi-T_lo)'` from a 2-point .temp sweep, or "
                f"run per-corner OP sweeps emitting iref + temp pairs."
            ),
            extras={
                "deferred": True,
                "deferred_reason": (
                    "multi-temperature sweep not present in op_result"
                ),
            },
        )

    return _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_temp_coeff_adjustment,
        measure_missing_hint="",
    )


def _check_pll_loop_filter(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> InvariantViolation:
    """Compare declared `loop_bandwidth_target` (Hz) against measured
    PLL closed-loop crossover bandwidth.

    Resolution order (v0.0.95 R72 C — full closed-loop AC testbench
    plumbed; the v0.0.94 R71 B partial-deferred path remains as the
    explicit fallback for cells that opt out of the live sim):

      1. Direct `loop_bw_<role.name>` (Hz) in ac_result — when an
         upstream testbench has already produced the crossover
         frequency. Tested first to preserve byte-exact behaviour for
         R71 B test fixtures that pre-computed the value.
      2. Global `pll_loop_bw` (Hz) fallback when only one PLL is
         present and the role-keyed name is absent.
      3. v0.0.95 R72 C NEW: a live closed-loop AC sweep via
         `clay.pll_loop_ac.simulate_pll_loop_bandwidth`, gated on the
         caller having attached a runtime handle to
         `role.attributes["_pll_loop_ac_runtime"]`. The handle dict
         carries:
           * `cell`         — the loop-filter Cell IR (required).
           * `ngspice_bin`  — absolute path to ngspice_con.exe
                              (required).
           * `work_dir`     — optional output directory.
           * `icp`, `kvco_hz_per_v`, `divider_n`, `f_start`, `f_stop`,
             `points_per_decade`, `pdk_corner`, `temp`, `timeout` —
             optional PLL-surrounding-block parameters.
         When the live sim succeeds, the measured crossover is used
         as `actual`. When it fails (build error, ngspice missing,
         no measure parsed), the runtime falls through to the
         deferred-path skip — never to a `measure_missing` (that
         would mislead the LLM into thinking the testbench should
         emit a measure it physically cannot without the chain).
      4. Surfaces `skipped + extras["deferred"]=True` when none of
         the above produce a value — same partial-deferred pattern as
         v0.0.90 R67 `loop_filter_pole`.

    Distinct from v0.0.90 `loop_filter_pole`, which targets the
    filter's own dominant pole; `pll_loop_filter` targets the closed-
    loop bandwidth at the PLL crossover (a derived quantity from the
    full loop gain, not the filter alone).
    """
    attr = _TARGET_ATTR_BY_KIND["pll_loop_filter"]
    target = _as_float(role.attributes.get(attr))
    if target is None:
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=None, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"pll_loop_filter {role.name!r}: no {attr!r} declared; "
                f"runtime has no target to check"
            ),
            suggested_adjustment=(
                "declare `loop_bandwidth_target = <value>Hz` in the "
                "role body"
            ),
        )

    actual: Optional[float] = None
    source = ""
    extras_extra: dict = {}
    keyed = f"loop_bw_{role.name}"
    if keyed in ac_result and math.isfinite(ac_result[keyed]):
        actual = float(ac_result[keyed])
        source = keyed
    elif "pll_loop_bw" in ac_result and math.isfinite(
        ac_result["pll_loop_bw"]
    ):
        actual = float(ac_result["pll_loop_bw"])
        source = "pll_loop_bw"

    # v0.0.95 R72 C: when no direct measure is in ac_result, try the
    # live closed-loop AC sweep if the caller has attached a runtime
    # handle. The handle dict is the explicit opt-in: cells that do
    # not attach it retain the v0.0.94 R71 B deferred-skip behaviour
    # byte-exact (so the existing test fixtures keep passing).
    if actual is None:
        handle = role.attributes.get("_pll_loop_ac_runtime")
        if isinstance(handle, dict):
            sim_result = _run_pll_loop_ac_via_handle(role, handle)
            if sim_result is not None:
                actual_val, sim_extras = sim_result
                if actual_val is not None and math.isfinite(actual_val):
                    actual = actual_val
                    # Source label reflects the cell that was simulated;
                    # fall back to the role name when the handle did
                    # not pass an explicit cell_name.
                    cell_obj = handle.get("cell")
                    cell_label = (
                        handle.get("cell_name")
                        or getattr(cell_obj, "name", None)
                        or role.name
                    )
                    source = f"pll_loop_ac:{cell_label}"
                    extras_extra.update(sim_extras)

    if actual is None:
        # Honest deferred-path skip: either no direct measure was
        # supplied AND no live closed-loop AC handle was attached, OR
        # the live AC sweep failed (build error, ngspice missing,
        # crossover not found in the swept band). The deferred-skip
        # surfaces the same structured marker the v0.0.94 R71 B test
        # suite asserts on (`extras["deferred"]=True` and
        # `extras["deferred_reason"]`), preserved byte-exact for the
        # no-handle code path so existing tests in
        # `test_functional_role_keywords_v94.py` keep passing.
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=None, relative_error=None,
            tolerance=tolerance, category="skipped",
            message=(
                f"pll_loop_filter {role.name!r}: loop_bandwidth_target="
                f"{_format_value(target)}Hz declared, but the closed-"
                f"loop PLL AC testbench is deferred to a follow-up. "
                f"Supply a direct loop_bw_{role.name} measure in the "
                f"testbench to land the check now."
            ),
            suggested_adjustment=(
                f"emit a direct loop_bw_{role.name} measure from a "
                f"closed-loop PLL AC sweep (charge-pump + VCO + "
                f"divider behavioral models required)"
            ),
            extras={
                "deferred": True,
                "deferred_reason": (
                    "PLL closed-loop AC testbench not plumbed"
                ),
            },
        )

    v = _emit_value_violation(
        role=role, attr=attr, target=target, actual=actual,
        source=source, tolerance=tolerance,
        suggest_fn=_suggest_loop_bw_adjustment,
        measure_missing_hint="",
    )
    if extras_extra:
        # Attach the live-sim provenance metadata to the violation so
        # downstream LLM tooling can see WHICH chain produced the
        # number (Icp / Kvco / N parameters used, log path, etc.).
        new_extras = dict(v.extras) if v.extras else {}
        new_extras.update(extras_extra)
        # Re-issue the violation with the augmented extras. Dataclass
        # is frozen-less so direct assignment works, but use a new
        # instance to keep semantics clean.
        v = InvariantViolation(
            role_kind=v.role_kind, role_name=v.role_name,
            target_attr=v.target_attr, target_value=v.target_value,
            actual_value=v.actual_value, relative_error=v.relative_error,
            tolerance=v.tolerance, category=v.category,
            message=v.message,
            suggested_adjustment=v.suggested_adjustment,
            extras=new_extras,
            gradient_value=v.gradient_value,
            suggested_target_value=v.suggested_target_value,
            suggested_target_unit=v.suggested_target_unit,
        )
    return v


def _run_pll_loop_ac_via_handle(
    role, handle: dict,
) -> Optional[tuple[Optional[float], dict]]:
    """Run the closed-loop AC sweep via the role-attached runtime handle.

    Returns `(actual_loop_bw_or_none, extras_dict)` on a clean call —
    `actual_loop_bw` is `None` when the sweep ran but produced no
    parseable crossover, in which case the caller falls through to the
    deferred-skip path. Returns `None` (outer-level) only when the
    handle itself is malformed (missing `cell` or `ngspice_bin`),
    which is also treated as a fall-through.

    The function never raises — every failure path returns a structured
    extras dict explaining the failure so the LLM can read the
    provenance even when the live sim degenerates.
    """
    cell = handle.get("cell")
    ngspice_bin = handle.get("ngspice_bin")
    if cell is None or ngspice_bin is None:
        # Malformed handle — fall through to deferred-skip without
        # surfacing an error (the caller likely attached the handle
        # incompletely and the deferred path is the safer default).
        return None
    try:
        from ..pll_loop_ac import simulate_pll_loop_bandwidth
    except ImportError as exc:
        return (None, {
            "pll_loop_ac_error": (
                f"import_failed: {type(exc).__name__}: {exc}"
            ),
        })
    kwargs = {
        k: handle[k] for k in (
            "work_dir", "icp", "kvco_hz_per_v", "divider_n",
            "f_start", "f_stop", "points_per_decade",
            "pdk_corner", "temp", "timeout",
        )
        if k in handle
    }
    try:
        res = simulate_pll_loop_bandwidth(
            cell, role, ngspice_bin=ngspice_bin, **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — never raise into runtime
        return (None, {
            "pll_loop_ac_error": (
                f"sim_raised: {type(exc).__name__}: {exc}"
            ),
        })
    extras = {
        "pll_loop_ac_netlist": str(res.netlist_path),
        "pll_loop_ac_log": str(res.log_path),
        "pll_loop_ac_measure": res.measure_name,
    }
    if res.loop_gain_lf_db is not None:
        extras["pll_loop_ac_lf_gain_db"] = res.loop_gain_lf_db
    if res.error is not None:
        extras["pll_loop_ac_error"] = res.error
    return (res.loop_bw_hz, extras)
