"""
Rust delegation hook (BL-012 Tier B.5 23/23) for invariant runtime
(v0.1.14 R91 B split).

Extracted verbatim from `clay/invariants.py` lines 242-495.

When `CLAY_USE_RUST_INVARIANTS=1` AND the `clay_ir` Rust extension is
built AND exposes `try_rust_check_functional_role`, ALL TWENTY-THREE
supported functional-role checks delegate to Rust first (R86 B Tier
B.5 closure: 23/23 = 100% Rust co-impl). The Rust path returns a
tagged tuple carrying pre-computed numerics; the Python wrapper in
`_rust_builders.py` rebuilds the `InvariantViolation` using the SAME
message templates the verbatim Python check uses — so the user-facing
`message` / `suggested_adjustment` / `extras` are byte-equal to what
Python's path would produce.

Honest-scope deferrals (R86 B partial paths kept in Python):
  * `pll_loop_filter` with `_pll_loop_ac_runtime` runtime handle on
    `role.attributes` → live closed-loop AC sweep via ngspice. The
    handle is a dict (not numeric/string), already dropped at the
    PyO3 boundary; `_r84b_should_defer_to_python` (extended in R86 B)
    detects the handle Python-side BEFORE delegation and routes to
    verbatim Python so the live-sim provenance extras are produced
    by the same code path that v0.0.95 R72 C plumbed.
"""

from __future__ import annotations

import math
import os
from typing import Mapping, Optional

from ._helpers import _as_float
from ._violation import InvariantViolation


# ─── v0.1.06–v0.1.09 R82 B + R83 B + R84 B + R85 B + R86 B: Rust delegation hook (BL-012) ──
#
# Strangler-fig contract: the 23 existing Python check functions are
# byte-exact preserved. The TWENTY-THREE supported kinds — five from
# R82 B (input_transconductor, output_buffer, compensation_zero,
# bandgap_core, dropout_pass_device), five from R83 B (feedforward_path,
# dominant_pole_cap, tail_current_source, cascode_helper,
# pole_tracking_cap single-corner path), five from R84 B
# (current_mirror no-fan-out path, diode_load single-target gm_target
# path, sample_hold_switch, gain_boost_amp dB→linear, dynamic_bias
# single-corner path), five from R85 B (dual_feedforward_pair
# no-target-only path, common_mode_feedback, regulated_cascode,
# output_class_AB, ldo_feedback_divider), plus the LAST THREE from
# R86 B (loop_filter_pole, ptat_branch, pll_loop_filter — full ports
# except for `pll_loop_filter`'s live closed-loop sim runtime handle
# which still defers to Python) — have an ADDITIVE delegation hook at
# the dispatch site in `check_functional_invariants`; when Rust
# returns None (extension missing, env flag off, pole_tracking_cap
# multi-corner trigger, dynamic_bias multi-corner trigger,
# current_mirror multi-branch fan-out trigger, diode_load multi-target
# trigger, dual_feedforward_pair worst-sub-error picker trigger,
# pll_loop_filter live-sim handle trigger, or any boundary exception),
# the Python checker runs verbatim.
#
# R86 B introduces a fifth outcome tag for `loop_filter_pole`/
# `ptat_branch`/`pll_loop_filter`: `skipped_deferred`. This surfaces
# when the target is declared but the underlying testbench infra
# (closed-loop AC or multi-T sweep) is not plumbed — Python's verbatim
# check stamps `extras["deferred"]=True` + `extras["deferred_reason"]`
# and the Rust path produces a `SkippedDeferred` variant the wrapper
# turns into the same byte-equal violation.

_HAS_CLAY_IR_INVARIANTS: bool = False
try:
    import clay_ir as _clay_ir_ext  # type: ignore[import-not-found]
    _HAS_CLAY_IR_INVARIANTS = hasattr(
        _clay_ir_ext, "try_rust_check_functional_role"
    )
except ImportError:
    _clay_ir_ext = None  # type: ignore[assignment]


# The TWENTY-THREE-keyword Rust subset (R86 B Tier B.5 closure: 23/23
# = 100% Rust co-impl). The Python wrapper checks membership before
# crossing the PyO3 boundary so out-of-subset kinds skip the call site
# cleanly. With R86 B the subset matches `_KNOWN_KINDS` 1:1.
_RUST_INVARIANT_SUBSET: frozenset[str] = frozenset({
    # R82 B
    "input_transconductor",
    "output_buffer",
    "compensation_zero",
    "bandgap_core",
    "dropout_pass_device",
    # R83 B
    "feedforward_path",
    "dominant_pole_cap",
    "tail_current_source",
    "cascode_helper",
    "pole_tracking_cap",
    # R84 B
    "current_mirror",
    "diode_load",
    "sample_hold_switch",
    "gain_boost_amp",
    "dynamic_bias",
    # R85 B
    "dual_feedforward_pair",
    "common_mode_feedback",
    "regulated_cascode",
    "output_class_AB",
    "ldo_feedback_divider",
    # R86 B (Tier B.5 closure — 23/23 = 100%)
    "loop_filter_pole",
    "ptat_branch",
    "pll_loop_filter",
})


def _rust_invariants_enabled() -> bool:
    """True iff the user opted into the Rust invariant path AND the
    extension surface is available."""
    if not _HAS_CLAY_IR_INVARIANTS:
        return False
    flag = os.environ.get("CLAY_USE_RUST_INVARIANTS", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _role_attrs_split(role) -> tuple[list, list]:
    """Split role.attributes into (numeric_pairs, string_pairs) lists for
    the PyO3 boundary. Non-finite numerics and non-(num|str) values are
    dropped (matches Python `_as_float` semantics)."""
    numerics: list = []
    strings: list = []
    for k, v in role.attributes.items():
        coerced = _as_float(v)
        if coerced is not None:
            numerics.append((str(k), coerced))
        elif isinstance(v, str):
            strings.append((str(k), v))
    return numerics, strings


def _op_ac_dict_to_pairs(d: Mapping[str, float]) -> list:
    """Coerce an op/ac dict into a list of (str, float) pairs, dropping
    non-finite / non-numeric values (PyO3 boundary doesn't accept None
    in the value slot)."""
    pairs: list = []
    if not d:
        return pairs
    for k, v in d.items():
        if isinstance(v, (int, float)) and math.isfinite(v):
            pairs.append((str(k), float(v)))
    return pairs


def _r84b_should_defer_to_python(role) -> bool:
    """Pre-Rust guards for R84 B / R86 B kinds whose complex paths stay in Python.

    Returns True when the role's declared attributes trigger a complex
    path that the Rust port does NOT cover. The dispatch site then
    skips the Rust call and runs verbatim Python so byte-equality is
    preserved.

    Triggers:

      * `current_mirror` with non-empty `fan_out_to` → multi-branch
        fan-out path (custom worst-branch picker, per-branch extras
        list). The Rust port handles ONLY the no-fan-out path.

      * `diode_load` with `effective_R_target` declared (with or
        without `gm_target`) → multi-target sub-error picker. The Rust
        port handles ONLY the single-target gm_target path.

      * R86 B: `pll_loop_filter` with `_pll_loop_ac_runtime` handle
        attached → live closed-loop AC sweep path. The handle is a
        dict containing the Cell IR, ngspice_bin path, and PLL-
        surrounding-block parameters — too complex for the Rust port,
        so the verbatim Python path runs the live sim. The handle is
        a dict (not numeric/string), so it's already dropped at the
        PyO3 boundary — this guard detects its presence Python-side
        BEFORE delegation.

    Other multi-corner triggers (dynamic_bias) are detected
    Rust-side via the `tracking_drift_max` + corner-count check;
    those return None from Rust and fall through to Python via the
    normal None-handling path (no Python-side pre-guard needed).
    """
    kind = getattr(role, "kind", None)
    if kind == "current_mirror":
        attrs = getattr(role, "attributes", {}) or {}
        fan_out = attrs.get("fan_out_to", None)
        # Empty / None / empty string → no fan-out. Non-empty list /
        # non-empty string → multi-branch path → defer.
        if fan_out:
            if isinstance(fan_out, (list, tuple)) and len(fan_out) == 0:
                return False
            if isinstance(fan_out, str) and fan_out == "":
                return False
            return True
        return False
    if kind == "diode_load":
        attrs = getattr(role, "attributes", {}) or {}
        eff_R = _as_float(attrs.get("effective_R_target"))
        if eff_R is not None:
            return True
        return False
    # R86 B: pll_loop_filter live closed-loop sim handle defers to Python.
    # The handle is `role.attributes["_pll_loop_ac_runtime"]` (a dict).
    # When present, the Python check function runs `simulate_pll_loop_bandwidth`
    # via `_run_pll_loop_ac_via_handle` and augments extras with the
    # `pll_loop_ac_netlist` / `pll_loop_ac_log` / `pll_loop_ac_measure`
    # provenance keys — too complex (boot ngspice, parse log) for Rust.
    if kind == "pll_loop_filter":
        attrs = getattr(role, "attributes", {}) or {}
        handle = attrs.get("_pll_loop_ac_runtime")
        if isinstance(handle, dict):
            return True
        return False
    return False


def _try_rust_check_functional_role(
    role,
    op_result: Mapping[str, float],
    ac_result: Mapping[str, float],
    tolerance: float,
) -> Optional[InvariantViolation]:
    """Delegate one functional-role check to Rust when applicable.

    Returns:
      * `None` when the Rust path is not applicable (env var unset,
        extension missing, kind out of subset, boundary error, or the
        R84 B pre-Rust guard detected a complex path that stays in
        Python). The caller MUST run the Python checker verbatim.
      * `InvariantViolation` rebuilt from Rust's tagged tuple result.
        The message text and `suggested_adjustment` are constructed
        Python-side using the same templates the verbatim Python
        checker uses, so the result is byte-equal.
    """
    # Lazy import to break the import cycle: _rust_builders imports
    # _python_skipped_no_target / _python_measure_missing / etc.
    # which live in _rust_builders.py and which depend on this file's
    # helpers. Reaching `_build_violation_from_rust` only at call time
    # avoids the top-level circular import.
    from ._rust_builders import _build_violation_from_rust

    if not _rust_invariants_enabled():
        return None
    if _clay_ir_ext is None:
        return None
    kind = getattr(role, "kind", None)
    if kind not in _RUST_INVARIANT_SUBSET:
        return None
    name = getattr(role, "name", "")
    if not isinstance(name, str):
        return None
    # R84 B pre-Rust guards: detect complex paths that aren't covered
    # by the Rust port and defer back to verbatim Python.
    if _r84b_should_defer_to_python(role):
        return None

    try:
        numerics, strings = _role_attrs_split(role)
        op_pairs = _op_ac_dict_to_pairs(op_result)
        ac_pairs = _op_ac_dict_to_pairs(ac_result)
        result = _clay_ir_ext.try_rust_check_functional_role(
            kind, name, numerics, strings, op_pairs, ac_pairs, float(tolerance),
        )
    except Exception:
        # Defensive: any boundary error falls back to Python so the
        # 23-keyword byte-exact path stays the contract anchor.
        return None
    if result is None:
        return None

    return _build_violation_from_rust(role, result, tolerance)
