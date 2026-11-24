"""
v0.0.90 gradient-augmented retry feedback (v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 4955-5331.

Stitches the v0.0.89 Adjoint Network `gradient_on_functional_target`
API onto v0.0.86 mismatch violations so the LLM sees a NUMERIC
suggested_target_value alongside the structured relative_error.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from ._dispatch import check_functional_invariants
from ._violation import DEFAULT_TOLERANCE, InvariantViolation


# ─── v0.0.90: Gradient-augmented retry feedback ────────────────────
#
# v0.0.86 invariant runtime surfaces "mismatch" as a text-level
# `suggested_adjustment` ("Increase Cc to push the dominant pole down").
# v0.0.89 Adjoint Network extension shipped `gradient_on_functional_target`
# returning a precise `∂(DC gain)/∂(target_attr)` number. v0.0.90 STITCHES
# the two so a mismatch violation surfaces a NUMERIC retry signal:
#
#     gradient_value         = ∂metric/∂target  (from adjoint)
#     suggested_target_value = target + step    (Newton-style)
#     suggested_target_unit  = canonical SI unit string
#
# Newton-step formula (per the brief):
#
#     suggested_target_value = current_target + (gradient_direction)
#                              * (spec_distance / gradient_magnitude)
#
# where `gradient_direction = sign(gradient_value)` and
# `spec_distance = abs(target_value - actual_value)`. Degenerate
# gradient magnitudes are clipped to a small epsilon so the suggestion
# stays finite; the resulting step is a *local linearization* — the LLM
# treats it as a one-step Newton hint, not a guaranteed converged value.
#
# Why dispatch ONLY on category="mismatch":
#   * `skipped`         — already passing; no retry needed → no gradient.
#   * `measure_missing` — no actual_value to compute spec_distance →
#                         no Newton step is meaningful.
#
# This conservative dispatch keeps the gradient compute cost zero on
# the in-tolerance and unobservable paths. The augmentation is also a
# soft-failure step: any exception inside the gradient bridge call is
# caught, the violation is left with `gradient_value=None`, and the
# extras dict records the failure reason so the LLM still gets the
# original text-level suggestion.

# Canonical SI unit string per target_attr. The unit is informational
# (the runtime never rescales magnitudes — every value flowing through
# `InvariantViolation.target_value` / `actual_value` is already in SI),
# but it lets the LLM render the suggestion as a human-readable string
# ("gm_target = 250e-6 A/V") rather than a raw float.
_TARGET_ATTR_UNIT: dict[str, str] = {
    "gm_target":         "A/V",
    "gain_target":       "V/V",
    "gain":              "V/V",
    "pole_freq_target":  "Hz",
    "zero_freq_target":  "Hz",
    "iref":              "A",
}

# The v0.0.89 Adjoint surface lowers exactly these target_attrs to a
# single device knob; pass-through attributes (`gain` on
# feedforward_path, `zero_freq_target` on compensation_zero) return
# `success=False` with `reason="passthrough"` and we surface no
# numeric suggestion (the text-level hint stays the only signal).
_GRADIENT_SUPPORTED_TARGETS: frozenset[str] = frozenset({
    "gm_target",
    "gain_target",
    "pole_freq_target",
    "iref",
})


def _newton_step_target(
    target_value: float,
    actual_value: float,
    gradient: float,
    *,
    eps: float = 1e-30,
    max_rel_step: float = 4.0,
) -> Optional[float]:
    """Newton-style suggested target value (per the v0.0.90 brief).

    Formula (brief, verbatim):
        spec_distance = abs(target_value - actual_value)
        gradient_direction = sign(gradient)         (+1, -1, or 0)
        suggested_target_value = current_target
                                 + gradient_direction
                                   * (spec_distance / |gradient|)

    The brief's formula is a one-step linearization of the inverse
    relationship: given `gradient = ∂metric/∂target`, the LLM wants to
    move `target` in the direction the gradient says will close the gap
    between the declared target and the simulator's actual. The step
    magnitude `spec_distance / |gradient|` is the classic Newton scale:
    "how far must target move to produce `spec_distance` change in
    metric, at the local slope `gradient`".

    Returns None when:
      * `target_value` or `actual_value` is non-finite.
      * `gradient` is zero or non-finite (no direction information).
      * The Newton step would push the suggestion past
        `(1+max_rel_step)*target` or below `target/(1+max_rel_step)` —
        clipping protects the LLM from a degenerate-OP gradient
        producing an order-of-magnitude jump on a single retry. This is
        a soft safety rail: the gradient is locally accurate, so a
        Newton step farther than ~5x the current value is almost
        certainly noise rather than signal.

    Note on direction convention:
      The brief specifies +sign(gradient) (not a sign-flip for
      `actual > target`). The intent is that the gradient's own sign
      tells the LLM which direction to push the target. If the LLM
      sees the wrong sign in practice on a specific (target_attr,
      cell topology) pair, that's a function of how the v0.0.89
      adjoint defines `∂metric/∂target` for that target_attr — not a
      problem for this formula. The numeric magnitude is always
      sensible (`spec_distance / |gradient|` is bounded by the clip).
    """
    if target_value is None or actual_value is None:
        return None
    if not (math.isfinite(target_value) and math.isfinite(actual_value)):
        return None
    if gradient is None or not math.isfinite(gradient):
        return None
    if abs(gradient) < eps:
        return None
    spec_distance = abs(target_value - actual_value)
    direction = 1.0 if gradient > 0 else -1.0
    step = direction * spec_distance / max(abs(gradient), eps)
    suggested = target_value + step
    # Clip to a sane band around the original target so a degenerate
    # OP doesn't yield an order-of-magnitude jump. Both positive and
    # negative targets are clipped to a +/- max_rel_step factor.
    if target_value > 0:
        lo = target_value / (1.0 + max_rel_step)
        hi = target_value * (1.0 + max_rel_step)
        suggested = max(lo, min(hi, suggested))
    elif target_value < 0:
        lo = target_value * (1.0 + max_rel_step)
        hi = target_value / (1.0 + max_rel_step)
        suggested = max(lo, min(hi, suggested))
    if not math.isfinite(suggested):
        return None
    return float(suggested)


def augment_violation_with_gradient(
    violation: InvariantViolation,
    cell_or_handle,
    *,
    in_node: str = "in",
    out_node: str = "out",
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
    step_rel: float = 0.05,
) -> InvariantViolation:
    """Stitch v0.0.89 Adjoint gradient onto a v0.0.86 mismatch entry.

    For ``category="mismatch"`` violations on a v0.0.89-supported
    target_attr (`gm_target` / `gain_target` / `pole_freq_target` /
    `iref`), call ``clay._ir_bridge.gradient_on_functional_target`` to
    resolve ``∂(DC gain)/∂(target)`` at the OP and turn that number plus
    the spec distance into a Newton-step ``suggested_target_value``.

    The function MUTATES the passed violation in-place AND returns it,
    so callers can chain. ``skipped`` / ``measure_missing`` violations
    are returned untouched (no gradient compute). Pass-through
    target_attrs (``gain``, ``zero_freq_target``) also skip — the
    Adjoint surface doesn't lower them to a single device knob.

    Soft-failure contract: any exception from the bridge call lands as
    ``gradient_value=None`` + ``extras["_gradient_error"]`` recording
    the failure reason. The text-level ``suggested_adjustment`` stays
    intact so the LLM still has SOMETHING to consume even when the
    adjoint compute degenerates.

    Args:
        violation:      The InvariantViolation to augment. Mutated.
        cell_or_handle: A compiled ``clay.ir.Cell`` or a
                        ``clay_ir.CellGraphHandle``. The bridge handles
                        both shapes; passing a handle saves the
                        ``cell_to_hypergraph`` rebuild on each call when
                        the caller has multiple mismatches on the same
                        cell.
        in_node:        Net name for the AC input probe (default
                        ``"in"``).
        out_node:       Net name for the AC output probe (default
                        ``"out"``).
        supply_vdd:     Supply rail (volts) for the OP solve.
        pdk_id:         PDK identifier (``"sky130"``, ``"gf180mcu"``).
        step_rel:       Finite-difference step relative size for the
                        FD fallback path (unused on the adjoint default
                        path).

    Returns:
        The same `InvariantViolation` object with
        `gradient_value` / `suggested_target_value` /
        `suggested_target_unit` populated when augmentation succeeded.
    """
    # Only the mismatch category carries enough information to compute
    # a Newton step. Other categories pass through untouched.
    if violation.category != "mismatch":
        return violation
    if violation.target_attr not in _GRADIENT_SUPPORTED_TARGETS:
        # Pass-through attribute (`gain` / `zero_freq_target`) — no
        # single-knob lowering; we leave the violation as v0.0.86.
        violation.extras.setdefault(
            "_gradient_dispatch", "passthrough_target_attr",
        )
        return violation
    if (
        violation.target_value is None
        or violation.actual_value is None
        or not math.isfinite(violation.target_value)
        or not math.isfinite(violation.actual_value)
    ):
        # Defensive: a mismatch entry should always carry both, but
        # belt-and-suspenders matter for malformed inputs from future
        # extension surfaces.
        violation.extras.setdefault(
            "_gradient_dispatch", "non_finite_values",
        )
        return violation

    # Lazy import to keep `clay.invariants` importable on PDK-less /
    # extension-less environments. The bridge itself returns None when
    # `_HAS_CLAY_IR=False` and we surface a soft no-op in that case.
    try:
        from .._ir_bridge import gradient_on_functional_target
    except ImportError as exc:  # noqa: BLE001
        violation.extras.setdefault(
            "_gradient_error", f"bridge_import_failed: {exc}",
        )
        return violation

    try:
        grad_dict = gradient_on_functional_target(
            cell_or_handle,
            violation.role_name,
            violation.target_attr,
            in_node,
            out_node,
            supply_vdd=supply_vdd,
            pdk_id=pdk_id,
            step_rel=step_rel,
        )
    except Exception as exc:  # noqa: BLE001
        violation.extras.setdefault(
            "_gradient_error", f"{type(exc).__name__}: {exc}",
        )
        return violation

    if grad_dict is None:
        # `_HAS_CLAY_IR=False`, handle was None, or the cell didn't
        # convert. Soft no-op.
        violation.extras.setdefault(
            "_gradient_error", "bridge_returned_none",
        )
        return violation
    if not grad_dict.get("success", False):
        # Adjoint reported failure (pass-through attr, unknown role,
        # degenerate OP, etc.). Surface the reason so the LLM sees why
        # no numeric suggestion was produced.
        violation.extras.setdefault(
            "_gradient_error",
            f"adjoint_failed: {grad_dict.get('reason', 'unknown')}",
        )
        return violation

    grad_val = grad_dict.get("gradient")
    if grad_val is None or not isinstance(grad_val, (int, float)):
        violation.extras.setdefault("_gradient_error", "non_numeric_gradient")
        return violation
    grad_val = float(grad_val)
    if not math.isfinite(grad_val):
        violation.extras.setdefault("_gradient_error", "non_finite_gradient")
        return violation

    suggested = _newton_step_target(
        target_value=violation.target_value,
        actual_value=violation.actual_value,
        gradient=grad_val,
    )

    violation.gradient_value = grad_val
    violation.suggested_target_value = suggested
    violation.suggested_target_unit = _TARGET_ATTR_UNIT.get(
        violation.target_attr,
    )
    # Record the resolution method for downstream tooling. Don't
    # clobber an existing entry — extras may already carry other keys.
    violation.extras.setdefault(
        "_gradient_method", grad_dict.get("method", "adjoint"),
    )
    violation.extras.setdefault(
        "_gradient_node_id", grad_dict.get("node_id", -1),
    )
    return violation


def augment_violations_with_gradients(
    violations: list[InvariantViolation],
    cell_or_handle,
    *,
    in_node: str = "in",
    out_node: str = "out",
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
    step_rel: float = 0.05,
) -> list[InvariantViolation]:
    """Apply :func:`augment_violation_with_gradient` to every entry in
    ``violations`` in-place. Returns the same list (handy for chaining
    into a JSON dump). Each entry is augmented independently so a
    failure on one entry doesn't prevent the others from getting a
    gradient.
    """
    for v in violations:
        augment_violation_with_gradient(
            v,
            cell_or_handle,
            in_node=in_node,
            out_node=out_node,
            supply_vdd=supply_vdd,
            pdk_id=pdk_id,
            step_rel=step_rel,
        )
    return violations


def check_functional_invariants_with_gradient(
    cell,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    in_node: str = "in",
    out_node: str = "out",
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
    step_rel: float = 0.05,
    cell_or_handle=None,
) -> list[InvariantViolation]:
    """One-call surface: run the v0.0.86 invariant check AND augment
    every mismatch entry with a Newton-step suggested_target_value.

    This is the convenience entry point downstream tooling reaches for
    when it wants the gradient-augmented retry signal in one round trip.
    The two-step decomposition (check, then augment) is preserved so
    callers that already hold a list of violations (e.g. parsed from a
    marker-section JSON dump) can call ``augment_violations_with_gradients``
    directly without re-running the invariant check.

    Args:
        cell:          Compiled Cell IR. Either carries
                       `functional_roles` directly or has them stashed
                       on `__dict__`.
        op_result:     OP-analysis measure dict.
        ac_result:     AC-analysis measure dict.
        tolerance:     Relative tolerance band (default 0.20 = ±20%).
        in_node:       AC input probe net name.
        out_node:      AC output probe net name.
        supply_vdd:    Supply rail (volts).
        pdk_id:        PDK identifier.
        step_rel:      FD step relative size.
        cell_or_handle: Optional override — if the caller has already
                       built a `clay_ir.CellGraphHandle` for the cell
                       (e.g. via `cell_to_hypergraph`), passing it here
                       avoids the redundant rebuild on each gradient
                       call. Defaults to `cell` (the bridge auto-builds
                       the handle from the Cell).
    """
    out = check_functional_invariants(
        cell,
        op_result=op_result,
        ac_result=ac_result,
        tolerance=tolerance,
    )
    target = cell_or_handle if cell_or_handle is not None else cell
    return augment_violations_with_gradients(
        out,
        target,
        in_node=in_node,
        out_node=out_node,
        supply_vdd=supply_vdd,
        pdk_id=pdk_id,
        step_rel=step_rel,
    )
