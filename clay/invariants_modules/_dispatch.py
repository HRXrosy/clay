"""
Dispatch table + public-API entry points for invariant runtime
(v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 4794-4953.

Surfaces:
  * `_CHECKERS`                       — 23-keyword kind→check fn dispatch
  * `_functional_roles_of(cell)`      — extract roles from Cell IR
  * `check_functional_invariants(...)` — main public entry point
  * `filter_real_violations(...)`     — keep only mismatch+measure_missing
"""

from __future__ import annotations

from typing import Mapping

from ._checks_batch1 import (
    _check_compensation_zero,
    _check_dominant_pole_cap,
    _check_feedforward_path,
    _check_input_transconductor,
    _check_output_buffer,
    _check_tail_current_source,
)
from ._checks_batch2 import (
    _check_cascode_helper,
    _check_current_mirror,
    _check_diode_load,
    _check_dual_feedforward_pair,
    _check_loop_filter_pole,
    _check_pole_tracking_cap,
)
from ._checks_batch3 import (
    _check_common_mode_feedback,
    _check_dynamic_bias,
    _check_gain_boost_amp,
    _check_output_class_AB,
    _check_regulated_cascode,
    _check_sample_hold_switch,
)
from ._checks_batch4 import (
    _check_bandgap_core,
    _check_dropout_pass_device,
    _check_ldo_feedback_divider,
    _check_pll_loop_filter,
    _check_ptat_branch,
)
from ._rust_delegation import _try_rust_check_functional_role
from ._violation import (
    DEFAULT_TOLERANCE,
    _KNOWN_KINDS,
    InvariantViolation,
)


# ─── Dispatch table + public API ────────────────────────────────


# v0.0.90 (Round 67): _CHECKERS extended to 12 entries — first batch six
# unchanged (v0.0.86), second batch six now route to REAL check functions
# that compare declared targets against ngspice OP / AC quantities.
# loop_filter_pole still emits a `skipped + deferred=True` when no
# direct loop_bw_<name> measure is available (PLL closed-loop AC
# testbench infrastructure is the follow-up dependency).
#
# v0.0.92 (Round 68 Agent G+H): _CHECKERS extended to 18 entries —
# third batch six route to REAL check functions (Agent H promoted the
# Agent G placeholders to real measurement comparison).
#
# v0.0.94 (Round 70 Agent C): _CHECKERS extended to 23 entries — fourth
# batch five originally routed to PLACEHOLDER check functions that
# returned a `category="skipped"` informational violation with
# `extras["deferred"]=True` and `extras["placeholder_round"]="v0.0.94"`.
#
# v0.0.94 (Round 71 Agent B): the fourth-batch five PLACEHOLDERS are
# promoted to REAL check functions that compare declared targets
# against ngspice OP/AC measures, mirroring the v0.0.92 Round 68 Agent H
# promotion pattern. Three of the five (bandgap_core, dropout_pass_device,
# ldo_feedback_divider) land single-corner DC OP comparison directly;
# ptat_branch lands a multi-temperature dI/dT slope comparison with an
# honest `skipped + deferred=True` when only one corner is present;
# pll_loop_filter lands a direct `loop_bw_<name>` measure path with
# `skipped + deferred=True` when the closed-loop PLL AC testbench is
# not yet plumbed (same partial pattern as v0.0.90 loop_filter_pole).
_CHECKERS = {
    # First batch (v0.0.86) — real checks
    "input_transconductor": _check_input_transconductor,
    "output_buffer":        _check_output_buffer,
    "feedforward_path":     _check_feedforward_path,
    "dominant_pole_cap":    _check_dominant_pole_cap,
    "compensation_zero":    _check_compensation_zero,
    "tail_current_source":  _check_tail_current_source,
    # Second batch (v0.0.90 Round 67) — real checks (loop_filter_pole
    # still partially deferred when no direct measure is available).
    "cascode_helper":        _check_cascode_helper,
    "pole_tracking_cap":     _check_pole_tracking_cap,
    "dual_feedforward_pair": _check_dual_feedforward_pair,
    "loop_filter_pole":      _check_loop_filter_pole,
    "current_mirror":        _check_current_mirror,
    "diode_load":            _check_diode_load,
    # Third batch (v0.0.92 Round 68 Agent G+H) — real checks
    "common_mode_feedback":  _check_common_mode_feedback,
    "sample_hold_switch":    _check_sample_hold_switch,
    "regulated_cascode":     _check_regulated_cascode,
    "gain_boost_amp":        _check_gain_boost_amp,
    "output_class_AB":       _check_output_class_AB,
    "dynamic_bias":          _check_dynamic_bias,
    # Fourth batch (v0.0.94 Round 70 Agent C → Round 71 Agent B promoted
    # to real checks). pll_loop_filter retains the partial-deferred path
    # (same pattern as v0.0.90 Round 67 loop_filter_pole) when no direct
    # closed-loop measure is supplied; ptat_branch falls back to
    # `skipped + deferred=True` when fewer than two temperature corners
    # are present in op_result.
    "bandgap_core":          _check_bandgap_core,
    "dropout_pass_device":   _check_dropout_pass_device,
    "ldo_feedback_divider":  _check_ldo_feedback_divider,
    "ptat_branch":           _check_ptat_branch,
    "pll_loop_filter":       _check_pll_loop_filter,
}


def _functional_roles_of(cell) -> list:
    """Return the list of `ParsedFunctionalRole` attached to a Cell IR.

    Two shapes are accepted:
      * `cell.functional_roles` (a hypothetical future direct field)
      * `cell.__dict__['_functional_roles']` (compiler-stashed attribute)

    The compiler stashes the list on `__dict__` to avoid having to add a
    new field to `ir.Cell` (which is Agent B's scope this round). The
    runtime accepts either shape so a future migration is trivial.
    """
    direct = getattr(cell, "functional_roles", None)
    if direct:
        return list(direct)
    return list(cell.__dict__.get("_functional_roles", []))


def check_functional_invariants(
    cell,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[InvariantViolation]:
    """Compare every functional role's declared target against ngspice.

    Args:
        cell:        compiled Cell IR. Either carries `functional_roles`
                     directly or has them stashed on `__dict__`.
        op_result:   `{name: float}` map of OP-analysis measures
                     (currents, gm, Vov, ...).
        ac_result:   `{name: float}` map of AC-analysis measures
                     (gain_dc, gbw, phase_at_gbw, pole_freq_*, ...).
        tolerance:   relative tolerance band (default 0.20 = ±20%).

    Returns:
        Flat `list[InvariantViolation]`. Empty list means every declared
        role passed (within tolerance) — including the case where the
        cell has zero functional roles.

    No exceptions. Unknown role kinds get a `skipped` violation. Missing
    measures get a `measure_missing` violation. Out-of-tolerance values
    get a `mismatch` violation with a `suggested_adjustment` hint.
    """
    if tolerance <= 0:
        raise ValueError(
            f"tolerance must be positive, got {tolerance!r}"
        )
    op = dict(op_result) if op_result is not None else {}
    ac = dict(ac_result) if ac_result is not None else {}

    out: list[InvariantViolation] = []
    for role in _functional_roles_of(cell):
        kind = getattr(role, "kind", None)
        if kind not in _KNOWN_KINDS:
            out.append(InvariantViolation(
                role_kind=str(kind), role_name=getattr(role, "name", "?"),
                target_attr="", target_value=None, actual_value=None,
                relative_error=None, tolerance=tolerance,
                category="skipped",
                message=(
                    f"unknown functional-role kind {kind!r}; runtime "
                    f"can only check {sorted(_KNOWN_KINDS)}"
                ),
                suggested_adjustment=(
                    f"use one of the supported kinds: "
                    f"{sorted(_KNOWN_KINDS)}"
                ),
            ))
            continue
        # v0.1.06 R82 B: Rust delegation hook (BL-012). Try Rust first
        # for the FIVE-keyword subset when `CLAY_USE_RUST_INVARIANTS=1`.
        # The Rust path returns a byte-equal `InvariantViolation` or
        # None; on None the Python check function runs verbatim. All
        # other 18 kinds always run the Python checker.
        rust_v = _try_rust_check_functional_role(role, op, ac, tolerance)
        if rust_v is not None:
            out.append(_attach_source_obligation_id(rust_v, role))
            continue
        checker = _CHECKERS[kind]
        out.append(
            _attach_source_obligation_id(
                checker(role, op, ac, tolerance),
                role,
            )
        )
    return out


def filter_real_violations(
    items: list[InvariantViolation],
) -> list[InvariantViolation]:
    """Return only `mismatch` and `measure_missing` entries.

    Convenience helper for callers (LLMs, dashboards, CI) that want
    the "real" failures separated from the informational `skipped`
    entries the runtime emits for in-tolerance passes."""
    return [v for v in items if v.category != "skipped"]


def _attach_source_obligation_id(
    violation: InvariantViolation,
    role,
) -> InvariantViolation:
    source_id = getattr(role, "attributes", {}).get("source_obligation_id")
    if source_id and violation.source_obligation_id is None:
        violation.source_obligation_id = str(source_id)
    return violation
