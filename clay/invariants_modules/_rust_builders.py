"""
Rust-side tagged-tuple → byte-exact Python InvariantViolation rebuilders
(v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 497-1449.

The Rust side supplies pre-computed numerics; the Python side does
every f-string format so the byte-equality contract is preserved.
Each per-kind branch mirrors the verbatim Python check function's
message + suggested_adjustment + extras template exactly.
"""

from __future__ import annotations

from typing import Optional

from ._helpers import _format_value
from ._suggestions import (
    _suggest_boost_gain_adjustment,
    _suggest_cmrr_adjustment,
    _suggest_divide_ratio_adjustment,
    _suggest_droop_adjustment,
    _suggest_dropout_adjustment,
    _suggest_dynamic_bias_adjustment,
    _suggest_gain_adjustment,
    _suggest_gm_adjustment,
    _suggest_iq_adjustment,
    _suggest_iref_adjustment,
    _suggest_loop_bw_adjustment,
    _suggest_pole_adjustment,
    _suggest_regulated_cascode_adjustment,
    _suggest_ro_boost_adjustment,
    _suggest_temp_coeff_adjustment,
    _suggest_vref_adjustment,
    _suggest_zero_adjustment,
)
from ._violation import InvariantViolation


def _build_violation_from_rust(
    role,
    result: tuple,
    tolerance: float,
) -> Optional[InvariantViolation]:
    """Convert the Rust tagged tuple into an `InvariantViolation` whose
    `message` / `suggested_adjustment` / `extras` are byte-exact with
    the Python check function that would have produced this outcome.

    The Rust side supplies pre-computed numerics; the Python side does
    every f-string format so the byte-equality contract is preserved.
    """
    if not isinstance(result, tuple) or not result:
        return None
    tag = result[0]
    if tag == "skipped_no_target":
        # ("skipped_no_target", attr, suggested_adjustment)
        if len(result) != 3:
            return None
        _, attr, suggested = result
        return _python_skipped_no_target(role, attr, suggested, tolerance)
    if tag == "measure_missing":
        # ("measure_missing", attr, target_value, hint_marker, [attempts])
        if len(result) != 5:
            return None
        _, attr, target_value, hint, attempts = result
        return _python_measure_missing(
            role, attr, float(target_value), hint, list(attempts), tolerance,
        )
    if tag == "skipped_in_tolerance":
        # ("skipped_in_tolerance", attr, target, actual, rel_err, source)
        if len(result) != 6:
            return None
        _, attr, target_value, actual_value, rel_err, source = result
        return _python_skipped_in_tolerance(
            role, attr, float(target_value), float(actual_value),
            float(rel_err), source, tolerance,
        )
    if tag == "mismatch":
        # ("mismatch", attr, target, actual, rel_err, source)
        if len(result) != 6:
            return None
        _, attr, target_value, actual_value, rel_err, source = result
        return _python_mismatch(
            role, attr, float(target_value), float(actual_value),
            float(rel_err), source, tolerance,
        )
    # R86 B (Tier B.5 closure) — deferred-skip outcome for the 3
    # closed-loop / multi-T keywords (`loop_filter_pole`, `ptat_branch`,
    # `pll_loop_filter`) when the target is declared but no usable
    # measure resolves AND the missing measure is from a deferred
    # testbench path. Distinct from `measure_missing` (the latter is
    # "testbench should have produced this measure and didn't"; this is
    # "testbench infra to plumb this check isn't built yet").
    if tag == "skipped_deferred":
        # ("skipped_deferred", attr, target_value, kind_marker)
        if len(result) != 4:
            return None
        _, attr, target_value, kind_marker = result
        return _python_skipped_deferred(
            role, attr, float(target_value), kind_marker, tolerance,
        )
    return None


def _python_skipped_no_target(
    role, attr: str, suggested_adjustment: str, tolerance: float,
) -> InvariantViolation:
    """Rebuild the canonical `skipped` violation a check function emits
    when the target attribute is absent. The message template matches
    each Python check's `f"{kind} {role.name!r}: no {attr!r}..."` form."""
    kind = role.kind
    if kind == "input_transconductor":
        message = (
            f"input_transconductor {role.name!r}: no {attr!r} "
            f"attribute declared; runtime has no target to check"
        )
    elif kind == "output_buffer":
        message = (
            f"output_buffer {role.name!r}: no {attr!r} attribute "
            f"declared; runtime has no target to check"
        )
    elif kind == "compensation_zero":
        message = (
            f"compensation_zero {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "bandgap_core":
        message = (
            f"bandgap_core {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "dropout_pass_device":
        message = (
            f"dropout_pass_device {role.name!r}: no {attr!r} "
            f"declared; runtime has no target to check"
        )
    elif kind == "feedforward_path":
        # Python: `f"feedforward_path {name!r}: no {attr!r} attribute "
        #          f"declared; runtime has no target to check"`
        message = (
            f"feedforward_path {role.name!r}: no {attr!r} attribute "
            f"declared; runtime has no target to check"
        )
    elif kind == "dominant_pole_cap":
        # Python: `f"dominant_pole_cap {name!r}: no {attr!r} "
        #          f"declared; runtime has no target to check"`
        message = (
            f"dominant_pole_cap {role.name!r}: no {attr!r} "
            f"declared; runtime has no target to check"
        )
    elif kind == "tail_current_source":
        message = (
            f"tail_current_source {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "cascode_helper":
        message = (
            f"cascode_helper {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "pole_tracking_cap":
        message = (
            f"pole_tracking_cap {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "current_mirror":
        # Python: `f"current_mirror {role.name!r}: no {attr!r} declared; "
        #          f"runtime has no target to check"`
        message = (
            f"current_mirror {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "diode_load":
        # Python's `_check_diode_load` "neither / nor" message names BOTH
        # target attributes. The Rust path passes `attr="gm_target"` but
        # the Python wrapper rebuilds the canonical multi-attribute
        # message. The suggested_adjustment matches Python verbatim.
        message = (
            f"diode_load {role.name!r}: neither 'gm_target' nor "
            f"'effective_R_target' declared; runtime has no target "
            f"to check"
        )
    elif kind == "sample_hold_switch":
        message = (
            f"sample_hold_switch {role.name!r}: no {attr!r} "
            f"declared; runtime has no target to check"
        )
    elif kind == "gain_boost_amp":
        message = (
            f"gain_boost_amp {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "dynamic_bias":
        message = (
            f"dynamic_bias {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    # ─── R85 B (BL-012 cont) — 5 more keyword no-target messages ────
    elif kind == "dual_feedforward_pair":
        # Python custom message: "neither 'gm_primary' nor
        # 'gm_secondary' declared; runtime has no target to check".
        # The Rust path emits this branch ONLY when neither leg is
        # declared (so the attr field is the canonical "gm_primary").
        message = (
            f"dual_feedforward_pair {role.name!r}: neither "
            f"'gm_primary' nor 'gm_secondary' declared; runtime "
            f"has no target to check"
        )
    elif kind == "common_mode_feedback":
        message = (
            f"common_mode_feedback {role.name!r}: no {attr!r} "
            f"declared; runtime has no target to check"
        )
    elif kind == "regulated_cascode":
        message = (
            f"regulated_cascode {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "output_class_AB":
        message = (
            f"output_class_AB {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "ldo_feedback_divider":
        message = (
            f"ldo_feedback_divider {role.name!r}: no {attr!r} "
            f"declared; runtime has no target to check"
        )
    # ─── R86 B (Tier B.5 closure) — 3 more keyword no-target msgs ──
    elif kind == "loop_filter_pole":
        # Python: f"loop_filter_pole {name!r}: no {attr!r} declared; "
        #          f"runtime has no target to check"
        message = (
            f"loop_filter_pole {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "ptat_branch":
        message = (
            f"ptat_branch {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    elif kind == "pll_loop_filter":
        message = (
            f"pll_loop_filter {role.name!r}: no {attr!r} declared; "
            f"runtime has no target to check"
        )
    else:  # pragma: no cover - subset guards above
        message = f"{kind} {role.name!r}: no {attr!r} declared"
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name, target_attr=attr,
        target_value=None, actual_value=None, relative_error=None,
        tolerance=tolerance, category="skipped",
        message=message,
        suggested_adjustment=suggested_adjustment,
    )


def _python_measure_missing(
    role, attr: str, target: float, hint_marker: str,
    resolution_attempts: list, tolerance: float,
) -> InvariantViolation:
    """Rebuild a `measure_missing` violation. Each supported kind has its
    own measure-missing message template — we dispatch on kind and
    `hint_marker` to pick the canonical text."""
    kind = role.kind
    if kind == "input_transconductor":
        message = (
            f"input_transconductor {role.name!r}: target gm_target="
            f"{_format_value(target)} declared but ngspice did not "
            f"produce gm/{role.name}, gm, or id/vov pair to derive "
            f"gm. Add `.print op @m{role.name}[gm]` to the testbench."
        )
        suggested = (
            f"emit a per-device gm measure (e.g. `gm_{role.name}`) "
            f"or supply both `id_{role.name}` and `vov_{role.name}` "
            f"so gm can be derived as 2*id/vov"
        )
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=None, relative_error=None,
            tolerance=tolerance, category="measure_missing",
            message=message, suggested_adjustment=suggested,
            extras={"resolution_attempts": resolution_attempts},
        )
    if kind == "output_buffer":
        hint = (
            "ngspice .meas did not produce gain_<name>, gain_dc_lin, "
            "or gain_dc; check AC testbench measure list"
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "compensation_zero":
        hint = (
            "ngspice .meas did not produce zero_freq_<name> nor "
            "zero_freq; the compensation zero needs an explicit phase- "
            "or magnitude-derived measure"
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "bandgap_core":
        # hint_marker = "bandgap_core:<vref_node>"
        vref_node = ""
        if hint_marker.startswith("bandgap_core:"):
            vref_node = hint_marker[len("bandgap_core:"):]
        hint = (
            f"ngspice did not produce vref_{role.name!r} (V), "
            f"v_{role.name!r}, nor a global `vref`. Add `.meas op "
            f"vref_{role.name} find v({vref_node or 'vref'}) at=0` "
            f"to the bandgap testbench."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "dropout_pass_device":
        hint = (
            f"ngspice did not produce dropout_{role.name!r} (V), "
            f"v_vin_{role.name!r}/v_vout_{role.name!r} pair, nor a "
            f"global `v_vin`/`v_vout` pair. Add `.meas op "
            f"dropout_{role.name} param='v(vin)-v(vout)'` under full-"
            f"load bias to the LDO testbench."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "feedforward_path":
        hint = (
            "ngspice .meas did not produce gain_<name>, gain_dc_lin, "
            "or gain_dc; check AC testbench measure list"
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "dominant_pole_cap":
        hint = (
            "ngspice .meas did not produce pole_freq_<name>, "
            "pole_freq_dominant, nor both gbw and gain_dc to derive "
            "the dominant pole; check AC testbench measure list"
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "tail_current_source":
        # tail_current_source uses _emit_value_violation with this
        # hint string; the Python check passes `target=abs(target)` so
        # the message text shows the absolute value (already done
        # Rust-side via target_abs).
        hint = (
            "ngspice .meas did not produce i_<name>, iref_<name>, "
            "iref, or itail; add `.meas op iref find i(Iref) at=0` "
            "to the testbench"
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "cascode_helper":
        # hint_marker = "cascode_helper:<attached_to>"
        attached_to = ""
        if hint_marker.startswith("cascode_helper:"):
            attached_to = hint_marker[len("cascode_helper:"):]
        hint = (
            f"ngspice did not produce ro_boost_{role.name!r}, "
            f"ro_{role.name!r}/ro_{attached_to!r}, nor "
            f"gm_{role.name!r}*ro_{role.name!r}; add "
            f".meas op ro_boost_{role.name} from a two-step "
            f"baseline-vs-cascode sweep, or emit per-device "
            f"gm/ro measures"
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "pole_tracking_cap":
        hint = (
            f"ngspice did not produce pole_{role.name!r} (single-corner) "
            f"or pole_{role.name}_<corner> entries (cross-corner). Add "
            f".meas ac pole_{role.name} when find vdb(out) val=-3 to the "
            f"AC testbench, or run per-corner sweeps for drift mode."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "current_mirror":
        # Python no-fan-out measure_missing message is CUSTOM (not the
        # `_emit_value_violation` template). It uses `ref_current=`
        # literal and a specific `Add .meas op i_<name>` hint string.
        # The Rust path sends the SIGNED ref_current as target_value
        # so the `_format_value(ref_current)` interpolation byte-matches
        # the Python check at line ~2281 in `_check_current_mirror`.
        message = (
            f"current_mirror {role.name!r}: ref_current="
            f"{_format_value(target)} declared but no "
            f"`i_{role.name}` or `iref` measure resolved. Add "
            f".meas op i_{role.name} find i(I{role.name}) at=0."
        )
        suggested = (
            f"emit i_{role.name} or iref measure in the testbench"
        )
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=None, relative_error=None,
            tolerance=tolerance, category="measure_missing",
            message=message,
            suggested_adjustment=suggested,
        )
    if kind == "diode_load":
        # Python: custom message naming the resolution chain
        # `gm_<name>, gm, or id/vov`. `extras["resolution_attempts"]`
        # uses ["gm_<name>", "gm", "id/vov"] from Rust.
        message = (
            f"diode_load {role.name!r}: gm_target="
            f"{_format_value(target)} declared but ngspice "
            f"did not produce gm_{role.name}, gm, or id/vov to "
            f"derive gm. Add `.print op @m{role.name}[gm]`."
        )
        suggested = (
            f"emit a per-device gm measure gm_{role.name} or "
            f"supply id_{role.name} + vov_{role.name}"
        )
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=None, relative_error=None,
            tolerance=tolerance, category="measure_missing",
            message=message, suggested_adjustment=suggested,
            extras={"resolution_attempts": resolution_attempts},
        )
    if kind == "sample_hold_switch":
        # Routes through generic `_emit_value_violation` template.
        hint = (
            f"ngspice did not produce droop_{role.name!r} (V/s) or "
            f"delta_vout_{role.name!r}/delta_t_{role.name!r} pair. "
            f"Add `.meas tran droop_{role.name} param='(v(sh_end)-"
            f"v(sh_start))/(t_end-t_start)'` to the transient "
            f"sample/hold testbench."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "gain_boost_amp":
        # Routes through generic `_emit_value_violation` template.
        # `target` is the linear V/V value (dB→lin done Rust-side).
        hint = (
            f"ngspice did not produce gain_{role.name!r} (linear V/V) "
            f"or gain_dc_{role.name!r} (dB) on the auxiliary amp. Add "
            f".meas ac gain_{role.name} find vdb(boost_out) at=1k to "
            f"the auxiliary-amp AC testbench."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "dynamic_bias":
        # Routes through generic `_emit_value_violation` template.
        # Single-corner path only — multi-corner trigger deferred
        # to Python pre-Rust via Rust None return.
        hint = (
            f"ngspice did not produce i_{role.name!r} (single-corner), "
            f"iref_{role.name!r}, nor any iref_{role.name}_<corner> "
            f"keys (cross-corner). Add `.meas op i_{role.name} "
            f"find i(Idyn) at=0`, or run per-corner OP sweeps for "
            f"tracking-drift mode."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    # ─── R85 B (BL-012 cont) — 4 more keyword measure_missing ─────
    if kind == "common_mode_feedback":
        # Generic `_emit_value_violation` template.
        hint = (
            f"ngspice did not produce cmrr_{role.name!r} (dB), "
            f"gain_dm_{role.name!r}/gain_cm_{role.name!r} (V/V pair), "
            f"nor a global `cmrr`. Add `.meas ac cmrr_{role.name} "
            f"param='20*log10(vdb(out_dm)/vdb(out_cm))'` or run a "
            f"common-mode AC loop testbench."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "regulated_cascode":
        # hint_marker = "regulated_cascode:<attached_to>|<boost_amp>"
        attached_to = ""
        boost_amp = ""
        prefix = "regulated_cascode:"
        if hint_marker.startswith(prefix):
            rest = hint_marker[len(prefix):]
            bar = rest.find("|")
            if bar >= 0:
                attached_to = rest[:bar]
                boost_amp = rest[bar + 1:]
        # Python's hint uses `attached_to!r` (None when absent — but
        # role.attributes.get returns "" for absent string attrs in
        # the verbatim Python). For the boost_amp slot Python uses
        # `boost_amp or '<amp>'` which renders the literal sentinel
        # when boost_amp is empty.
        hint = (
            f"ngspice did not produce ro_boost_{role.name!r}, "
            f"ro_{role.name!r}/ro_{attached_to!r}, nor "
            f"gm_{role.name!r}*ro_{role.name!r}. Add a "
            f".meas op ro_boost_{role.name} from a two-step sweep "
            f"(boost-amp on vs off), or emit per-device gm/ro "
            f"measures and a boost_gain_{boost_amp or '<amp>'} value."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "output_class_AB":
        # hint_marker = "output_class_AB:<push_dev>"
        push_dev = ""
        prefix = "output_class_AB:"
        if hint_marker.startswith(prefix):
            push_dev = hint_marker[len(prefix):]
        hint = (
            f"ngspice did not produce iq_{role.name!r}, "
            f"i_{push_dev or '<push>'} / id_{push_dev or '<push>'}, "
            f"nor a global `iq`. Add `.meas op iq_{role.name} "
            f"find abs(i(Mpush)) at=0` (with the class-AB stage in "
            f"its idle bias point)."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    if kind == "ldo_feedback_divider":
        hint = (
            f"ngspice did not produce divide_ratio_{role.name!r}, "
            f"v_top_{role.name!r}/v_bot_{role.name!r} pair, nor a "
            f"global `divide_ratio`. Add `.meas op v_top_{role.name} "
            f"find v(top) at=0` and `.meas op v_bot_{role.name} find "
            f"v(tap) at=0` to the LDO feedback-divider testbench."
        )
        return _emit_value_violation_byteexact_missing(
            role=role, attr=attr, target=target,
            tolerance=tolerance, hint=hint,
        )
    # pragma: no cover - subset guards above
    return _emit_value_violation_byteexact_missing(
        role=role, attr=attr, target=target,
        tolerance=tolerance, hint="",
    )


def _emit_value_violation_byteexact_missing(
    *, role, attr: str, target: float, tolerance: float, hint: str,
) -> InvariantViolation:
    """Mirror `_emit_value_violation` measure-missing branch byte-exactly.

    Python's `_emit_value_violation(actual=None, ...)` produces a
    measure_missing entry with message
    `f"{role.kind} {role.name!r}: target {attr}={_format_value(target)} "
    `f"declared but simulator did not produce the supporting measure. "
    `f"{measure_missing_hint}"`. The R82 B output_buffer / compensation_zero
    / bandgap_core / dropout_pass_device checks all route through this
    helper.
    """
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name, target_attr=attr,
        target_value=target, actual_value=None, relative_error=None,
        tolerance=tolerance, category="measure_missing",
        message=(
            f"{role.kind} {role.name!r}: target {attr}="
            f"{_format_value(target)} declared but simulator did not "
            f"produce the supporting measure. {hint}"
        ),
        suggested_adjustment=hint,
    )


def _rewrite_rust_source_marker(source: str) -> str:
    """Rewrite Rust source-string sentinels into the byte-equal Python
    source string. R83 B introduced one sentinel and R84 B introduces
    a second:

      * `derived_gbw_over_gain:<gain_lin_raw>` for `dominant_pole_cap`
        derived path. Rust emits the raw f64 in `{}` format; Python
        rebuilds `f"derived gbw/{_format_value(gain_lin)}"` so the
        embedded `_format_value` (Python's `%.4g` / `%.3e` mixed
        format) renders byte-equal to the verbatim Python checker.

      * `current_mirror_signed_actual:<signed_a>|<real_source>` for
        `current_mirror` no-fan-out path. Python's message text uses
        `_format_value(ref_actual)` (signed) while
        `actual_value=abs(ref_actual)`. The Rust path classifies on
        abs values for correct rel_err math, but preserves the
        signed value in the sentinel so the wrapper can render the
        signed value in the message. This rewrite strips the sentinel
        and returns just the `<real_source>` part so
        `extras["actual_source"]` stays byte-equal to Python's
        `ref_source` (the raw measure key like `i_Mmir` / `iref`).

    All other sources pass through unchanged."""
    prefix = "derived_gbw_over_gain:"
    if source.startswith(prefix):
        try:
            gain_lin = float(source[len(prefix):])
        except ValueError:
            return source
        return f"derived gbw/{_format_value(gain_lin)}"
    cm_prefix = "current_mirror_signed_actual:"
    if source.startswith(cm_prefix):
        # Sentinel format: current_mirror_signed_actual:<signed_a>|<real_source>
        # The wrapper that NEEDS the signed actual reads it via
        # `_extract_current_mirror_signed_actual` below; this rewrite
        # just strips the sentinel for the `extras["actual_source"]`
        # field.
        rest = source[len(cm_prefix):]
        bar = rest.find("|")
        if bar >= 0:
            return rest[bar + 1:]
        return source
    return source


def _extract_current_mirror_signed_actual(
    source: str,
) -> Optional[float]:
    """Return the signed actual value embedded in a
    `current_mirror_signed_actual:` sentinel source string, or None
    when the source has no sentinel.

    Used by the `current_mirror` Python wrapper to render the signed
    actual in the message text while still surfacing the abs value in
    the `actual_value` field (matching Python's verbatim check)."""
    prefix = "current_mirror_signed_actual:"
    if not source.startswith(prefix):
        return None
    rest = source[len(prefix):]
    bar = rest.find("|")
    if bar < 0:
        return None
    try:
        return float(rest[:bar])
    except ValueError:
        return None


def _python_skipped_in_tolerance(
    role, attr: str, target: float, actual: float,
    rel_err: float, source: str, tolerance: float,
) -> InvariantViolation:
    """Rebuild a `skipped` (in-tolerance pass) violation byte-exactly.

    `input_transconductor` has its own message template (mentions the
    kind name verbatim); every other supported kind routes through
    `_emit_value_violation`'s skipped branch — same template, but the
    kind name is interpolated from `role.kind` so a single helper
    suffices for output_buffer / compensation_zero / bandgap_core /
    dropout_pass_device / feedforward_path / dominant_pole_cap /
    tail_current_source / cascode_helper / pole_tracking_cap.
    """
    # R83 B: rewrite any Rust source-string sentinel (e.g.
    # dominant_pole_cap's `derived_gbw_over_gain:<gain_lin>`) into the
    # byte-equal Python source string. R82 B kinds use literal source
    # strings so this is a no-op for them. R84 B current_mirror also
    # embeds a `current_mirror_signed_actual:<signed_a>|<src>` sentinel
    # — we preserve the original BEFORE the strip so the wrapper can
    # recover the signed actual for the message text.
    orig_source = source
    source = _rewrite_rust_source_marker(source)
    kind = role.kind
    if kind == "input_transconductor":
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=actual,
            relative_error=rel_err,
            tolerance=tolerance, category="skipped",
            message=(
                f"input_transconductor {role.name!r}: gm_target="
                f"{_format_value(target)} matches actual="
                f"{_format_value(actual)} within ±{tolerance*100:g}%"
            ),
            extras={"actual_source": source},
        )
    if kind == "current_mirror":
        # Python no-fan-out skipped message uses `ref_current=` literal
        # AND `_format_value(ref_actual)` (signed). The Rust sentinel
        # `current_mirror_signed_actual:<signed_a>|<src>` was already
        # stripped by `_rewrite_rust_source_marker` above into `source`,
        # so we read the signed value from the ORIGINAL source before
        # the strip.
        signed_actual = _extract_current_mirror_signed_actual(orig_source)
        msg_actual = signed_actual if signed_actual is not None else actual
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=actual,
            relative_error=rel_err,
            tolerance=tolerance, category="skipped",
            message=(
                f"current_mirror {role.name!r}: ref_current="
                f"{_format_value(target)} matches actual="
                f"{_format_value(msg_actual)} within ±{tolerance*100:g}%"
            ),
            extras={"actual_source": source},
        )
    if kind == "diode_load":
        # Python: "all checks within ±X% (worst gm_target=... vs
        # actual=...)" with extras carrying `sub_errors`, `gm_actual`,
        # `R_actual` (None for single-target).
        sub_errors = [
            {"attr": "gm_target", "target": target, "actual": actual,
             "rel_err": rel_err},
        ]
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=actual,
            relative_error=rel_err,
            tolerance=tolerance, category="skipped",
            message=(
                f"diode_load {role.name!r}: all checks within "
                f"±{tolerance*100:g}% (worst gm_target="
                f"{_format_value(target)} vs actual="
                f"{_format_value(actual)})"
            ),
            extras={
                "actual_source": source,
                "sub_errors": sub_errors,
                "gm_actual": actual,
                "R_actual": None,
            },
        )
    # output_buffer / compensation_zero / bandgap_core /
    # dropout_pass_device / sample_hold_switch / gain_boost_amp /
    # dynamic_bias → `_emit_value_violation` skipped template.
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name, target_attr=attr,
        target_value=target, actual_value=actual,
        relative_error=rel_err,
        tolerance=tolerance, category="skipped",
        message=(
            f"{role.kind} {role.name!r}: {attr}={_format_value(target)} "
            f"matches actual={_format_value(actual)} within "
            f"±{tolerance*100:g}%"
        ),
        extras={"actual_source": source},
    )


def _python_mismatch(
    role, attr: str, target: float, actual: float,
    rel_err: float, source: str, tolerance: float,
) -> InvariantViolation:
    """Rebuild a `mismatch` violation byte-exactly.

    `input_transconductor` has its own message template (mentions
    `gm_target=` verbatim). Every other supported kind routes through
    `_emit_value_violation`'s mismatch template (`{attr}=...` form). The
    `suggested_adjustment` is computed via the same `_suggest_*` helper
    the verbatim Python checker uses.
    """
    # R83 B: rewrite any Rust source-string sentinel before the
    # extras map is constructed (e.g. dominant_pole_cap derived path).
    # R84 B current_mirror embeds a signed-actual sentinel — preserve
    # the original so the wrapper can recover the signed value.
    orig_source = source
    source = _rewrite_rust_source_marker(source)
    direction = "exceeds" if rel_err > 0 else "underperforms"
    kind = role.kind
    if kind == "input_transconductor":
        suggested = _suggest_gm_adjustment(target, actual)
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=actual, relative_error=rel_err,
            tolerance=tolerance, category="mismatch",
            message=(
                f"input_transconductor {role.name!r}: gm_target="
                f"{_format_value(target)} but actual gm={_format_value(actual)} "
                f"(rel_err={rel_err*100:.1f}%, {direction} target by "
                f"{abs(rel_err)*100:.1f}%, tolerance=±{tolerance*100:g}%)"
            ),
            suggested_adjustment=suggested,
            extras={"actual_source": source},
        )
    if kind == "current_mirror":
        # Python no-fan-out mismatch message uses `ref_current=`
        # literal AND `_format_value(ref_actual)` (signed). The Rust
        # sentinel preserves the signed actual; read it before the
        # rewrite stripped it.
        signed_actual = _extract_current_mirror_signed_actual(orig_source)
        msg_actual = signed_actual if signed_actual is not None else actual
        # Python passes `abs(ref_current), abs(ref_actual)` to
        # `_suggest_iref_adjustment` — `target` and `actual` are already
        # the abs values from Rust's `target_abs` / `a.abs()`.
        suggested = _suggest_iref_adjustment(target, actual)
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=actual, relative_error=rel_err,
            tolerance=tolerance, category="mismatch",
            message=(
                f"current_mirror {role.name!r}: ref_current="
                f"{_format_value(target)} but actual="
                f"{_format_value(msg_actual)} "
                f"(rel_err={rel_err*100:.1f}%, "
                f"tolerance=±{tolerance*100:g}%)"
            ),
            suggested_adjustment=suggested,
            extras={"actual_source": source},
        )
    if kind == "diode_load":
        # Python: "diode_load {name}: gm_target=... but actual=... ..."
        # with extras carrying sub_errors / gm_actual / R_actual=None.
        suggested = _suggest_gm_adjustment(target, actual)
        sub_errors = [
            {"attr": "gm_target", "target": target, "actual": actual,
             "rel_err": rel_err},
        ]
        return InvariantViolation(
            role_kind=role.kind, role_name=role.name, target_attr=attr,
            target_value=target, actual_value=actual, relative_error=rel_err,
            tolerance=tolerance, category="mismatch",
            message=(
                f"diode_load {role.name!r}: gm_target="
                f"{_format_value(target)} but actual="
                f"{_format_value(actual)} "
                f"(rel_err={rel_err*100:.1f}%, "
                f"{direction} target by {abs(rel_err)*100:.1f}%, "
                f"tolerance=±{tolerance*100:g}%)"
            ),
            suggested_adjustment=suggested,
            extras={
                "actual_source": source,
                "sub_errors": sub_errors,
                "gm_actual": actual,
                "R_actual": None,
            },
        )
    # Map kind → suggest_fn (matches the verbatim per-kind Python checker).
    if kind == "output_buffer":
        suggested = _suggest_gain_adjustment(target, actual)
    elif kind == "compensation_zero":
        suggested = _suggest_zero_adjustment(target, actual)
    elif kind == "bandgap_core":
        suggested = _suggest_vref_adjustment(target, actual)
    elif kind == "dropout_pass_device":
        suggested = _suggest_dropout_adjustment(target, actual)
    elif kind == "feedforward_path":
        # feedforward_path Python checker routes through
        # _suggest_gain_adjustment (same as output_buffer — both check
        # a voltage transfer ratio).
        suggested = _suggest_gain_adjustment(target, actual)
    elif kind == "dominant_pole_cap":
        suggested = _suggest_pole_adjustment(target, actual)
    elif kind == "tail_current_source":
        suggested = _suggest_iref_adjustment(target, actual)
    elif kind == "cascode_helper":
        suggested = _suggest_ro_boost_adjustment(target, actual)
    elif kind == "pole_tracking_cap":
        # Single-corner path uses _suggest_pole_adjustment (the
        # multi-corner _suggest_pole_tracking_adjustment branch never
        # routes through Rust — Rust returns None for that case).
        suggested = _suggest_pole_adjustment(target, actual)
    elif kind == "sample_hold_switch":
        suggested = _suggest_droop_adjustment(target, actual)
    elif kind == "gain_boost_amp":
        suggested = _suggest_boost_gain_adjustment(target, actual)
    elif kind == "dynamic_bias":
        # Single-corner path uses _suggest_dynamic_bias_adjustment
        # (multi-corner _suggest_dynamic_bias_drift_adjustment never
        # routes through Rust — Rust returns None when the trigger
        # fires).
        suggested = _suggest_dynamic_bias_adjustment(target, actual)
    # ─── R85 B (BL-012 cont) suggest_fn map ────────────────────────
    elif kind == "common_mode_feedback":
        suggested = _suggest_cmrr_adjustment(target, actual)
    elif kind == "regulated_cascode":
        suggested = _suggest_regulated_cascode_adjustment(target, actual)
    elif kind == "output_class_AB":
        suggested = _suggest_iq_adjustment(target, actual)
    elif kind == "ldo_feedback_divider":
        suggested = _suggest_divide_ratio_adjustment(target, actual)
    # ─── R86 B (Tier B.5 closure) suggest_fn map ───────────────────
    elif kind == "loop_filter_pole":
        # Both PLL keywords share `_suggest_loop_bw_adjustment` (the
        # Python check at lines 2642-2647 routes through
        # `_emit_value_violation` with this suggest_fn).
        suggested = _suggest_loop_bw_adjustment(target, actual)
    elif kind == "ptat_branch":
        suggested = _suggest_temp_coeff_adjustment(target, actual)
    elif kind == "pll_loop_filter":
        # Same suggest_fn as loop_filter_pole — both target a loop
        # bandwidth quantity (in Hz) and the suggested-adjustment text
        # is identical (Python check at line 4540-4543).
        suggested = _suggest_loop_bw_adjustment(target, actual)
    else:  # pragma: no cover - subset guards above
        suggested = ""
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name, target_attr=attr,
        target_value=target, actual_value=actual, relative_error=rel_err,
        tolerance=tolerance, category="mismatch",
        message=(
            f"{role.kind} {role.name!r}: {attr}={_format_value(target)} "
            f"but actual={_format_value(actual)} (rel_err={rel_err*100:.1f}%, "
            f"{direction} target by {abs(rel_err)*100:.1f}%, "
            f"tolerance=±{tolerance*100:g}%)"
        ),
        suggested_adjustment=suggested,
        extras={"actual_source": source},
    )


def _python_skipped_deferred(
    role, attr: str, target: float, kind_marker: str, tolerance: float,
) -> InvariantViolation:
    """Rebuild a `skipped + extras["deferred"]=True` violation byte-exactly.

    R86 B kinds emit this when the target is declared but the
    underlying testbench infra (closed-loop AC for `loop_filter_pole` /
    `pll_loop_filter`, multi-temperature sweep for `ptat_branch`) is
    not plumbed. The Python verbatim checker uses a kind-specific
    message + suggested_adjustment template and stamps
    `extras["deferred"]=True` plus `extras["deferred_reason"]=<reason>`
    so downstream tooling can distinguish "infra deferred" from
    "measure absent from a plumbed testbench".
    """
    if kind_marker == "loop_filter_pole":
        message = (
            f"loop_filter_pole {role.name!r}: target_loop_bw="
            f"{_format_value(target)}Hz declared, but the closed-loop "
            f"PLL AC testbench is deferred to a follow-up. Supply a "
            f"direct loop_bw_{role.name} measure in the testbench to "
            f"land the check now."
        )
        suggested = (
            f"emit a direct loop_bw_{role.name} measure from a "
            f"closed-loop PLL AC sweep (charge-pump + VCO + divider "
            f"behavioral models required)"
        )
        deferred_reason = "PLL closed-loop AC testbench not plumbed"
    elif kind_marker == "ptat_branch":
        message = (
            f"ptat_branch {role.name!r}: temp_coefficient_target="
            f"{_format_value(target)}A/K declared, but the multi-"
            f"temperature sweep required to compute dI/dT is not "
            f"present in this op_result. Emit a direct "
            f"temp_coeff_{role.name} measure (preferred) or run "
            f"the same testbench at >= 2 temperatures with "
            f"`iref_{role.name}_<corner>` + `temp_<corner>` "
            f"measures."
        )
        suggested = (
            f"emit `.meas op temp_coeff_{role.name} param='(i_hi"
            f"-i_lo)/(T_hi-T_lo)'` from a 2-point .temp sweep, or "
            f"run per-corner OP sweeps emitting iref + temp pairs."
        )
        deferred_reason = (
            "multi-temperature sweep not present in op_result"
        )
    elif kind_marker == "pll_loop_filter":
        message = (
            f"pll_loop_filter {role.name!r}: loop_bandwidth_target="
            f"{_format_value(target)}Hz declared, but the closed-"
            f"loop PLL AC testbench is deferred to a follow-up. "
            f"Supply a direct loop_bw_{role.name} measure in the "
            f"testbench to land the check now."
        )
        suggested = (
            f"emit a direct loop_bw_{role.name} measure from a "
            f"closed-loop PLL AC sweep (charge-pump + VCO + "
            f"divider behavioral models required)"
        )
        deferred_reason = (
            "PLL closed-loop AC testbench not plumbed"
        )
    else:  # pragma: no cover - kind_marker is one of the above
        message = (
            f"{role.kind} {role.name!r}: deferred (no kind_marker template)"
        )
        suggested = ""
        deferred_reason = "deferred"
    return InvariantViolation(
        role_kind=role.kind, role_name=role.name, target_attr=attr,
        target_value=target, actual_value=None, relative_error=None,
        tolerance=tolerance, category="skipped",
        message=message,
        suggested_adjustment=suggested,
        extras={
            "deferred": True,
            "deferred_reason": deferred_reason,
        },
    )
