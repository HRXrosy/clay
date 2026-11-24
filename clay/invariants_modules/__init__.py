"""
Clay functional-role invariant runtime — split package (v0.1.14 R91 B).

The invariants implementation was split out of `clay/invariants.py`
(formerly 5331 LOC) into this package in v0.1.14 (R91 Agent B). This
package re-exports the full public + internal surface so the
`clay/invariants.py` shim (preserved as < 200 LOC re-export) keeps
every existing import path byte-exact.

Module layout:
  * `_violation.py`        — InvariantViolation dataclass + role-kind constants
  * `_helpers.py`          — basic numeric helpers (_as_float etc.)
  * `_op_parser.py`        — ngspice OP-quantity log parser
  * `_suggestions.py`      — all _suggest_*_adjustment helpers (16 fns)
  * `_emit.py`             — _emit_value_violation + _resolve_ac_gain_linear
  * `_rust_delegation.py`  — Rust delegation hook (BL-012 Tier B.5 23/23)
  * `_rust_builders.py`    — Byte-exact Python rebuilders for Rust tagged tuples
  * `_checks_batch1.py`    — first-batch 6 checks (v0.0.86 Round 62)
  * `_checks_batch2.py`    — second-batch 6 checks (v0.0.90 Round 67)
  * `_checks_batch3.py`    — third-batch 6 checks (v0.0.92 Round 68)
  * `_checks_batch4.py`    — fourth-batch 5 checks + placeholder + PLL handle
  * `_dispatch.py`         — _CHECKERS dispatch + check_functional_invariants
  * `_gradient.py`         — v0.0.90 gradient-augmented retry feedback

The shim re-exports the verbatim public + internal surface so callers
keep working:

  * `from clay.invariants import DEFAULT_TOLERANCE, InvariantViolation`
  * `from clay.invariants import check_functional_invariants`
  * `from clay.invariants import _CHECKERS, _KNOWN_KINDS, _TARGET_ATTR_BY_KIND`
  * `from clay.invariants import _check_pll_loop_filter`
  * `from clay.invariants import _HAS_CLAY_IR_INVARIANTS, _RUST_INVARIANT_SUBSET`
  * `from clay.invariants import parse_op_quantity_lines, merge_op_quantities`
  * `from clay.invariants import filter_real_violations, _functional_roles_of`
  * `from clay.invariants import augment_violation_with_gradient`
  * `from clay.invariants import _newton_step_target`
"""

from __future__ import annotations


# ─── Re-export the full surface ──────────────────────────────────


# Violation dataclass + role-kind constants
from ._violation import (  # noqa: F401
    DEFAULT_TOLERANCE,
    InvariantViolation,
    VerificationStatus,
    _KNOWN_KINDS,
    _TARGET_ATTR_BY_KIND,
    category_to_verification_status,
    contributes_to_proof_token,
)


# Numeric helpers
from ._helpers import (  # noqa: F401
    _as_float,
    _format_value,
    _relative_error,
    _within_tolerance,
)


# OP log parser + merge
from ._op_parser import (  # noqa: F401
    _OP_QUANTITY_LINE_RE,
    _device_name_matches_role,
    merge_op_quantities,
    parse_op_quantity_lines,
)


# Suggestion helpers
from ._suggestions import (  # noqa: F401
    _suggest_boost_gain_adjustment,
    _suggest_cmrr_adjustment,
    _suggest_divide_ratio_adjustment,
    _suggest_droop_adjustment,
    _suggest_dropout_adjustment,
    _suggest_dual_feedforward_adjustment,
    _suggest_dynamic_bias_adjustment,
    _suggest_dynamic_bias_drift_adjustment,
    _suggest_effective_R_adjustment,
    _suggest_gain_adjustment,
    _suggest_gm_adjustment,
    _suggest_iq_adjustment,
    _suggest_iref_adjustment,
    _suggest_loop_bw_adjustment,
    _suggest_mirror_adjustment,
    _suggest_pole_adjustment,
    _suggest_pole_tracking_adjustment,
    _suggest_regulated_cascode_adjustment,
    _suggest_ro_boost_adjustment,
    _suggest_temp_coeff_adjustment,
    _suggest_vref_adjustment,
    _suggest_zero_adjustment,
)


# Shared value-check emitter + AC gain resolver
from ._emit import (  # noqa: F401
    _emit_value_violation,
    _resolve_ac_gain_linear,
)


# Rust delegation hook (BL-012 Tier B.5 23/23) — entry surface
from ._rust_delegation import (  # noqa: F401
    _HAS_CLAY_IR_INVARIANTS,
    _RUST_INVARIANT_SUBSET,
    _clay_ir_ext,
    _op_ac_dict_to_pairs,
    _r84b_should_defer_to_python,
    _role_attrs_split,
    _rust_invariants_enabled,
    _try_rust_check_functional_role,
)


# Rust-side tagged-tuple → byte-exact Python rebuilders
from ._rust_builders import (  # noqa: F401
    _build_violation_from_rust,
    _emit_value_violation_byteexact_missing,
    _extract_current_mirror_signed_actual,
    _python_measure_missing,
    _python_mismatch,
    _python_skipped_deferred,
    _python_skipped_in_tolerance,
    _python_skipped_no_target,
    _rewrite_rust_source_marker,
)


# First-batch (v0.0.86) check functions
from ._checks_batch1 import (  # noqa: F401
    _check_compensation_zero,
    _check_dominant_pole_cap,
    _check_feedforward_path,
    _check_input_transconductor,
    _check_output_buffer,
    _check_tail_current_source,
)


# Second-batch (v0.0.90) check functions
from ._checks_batch2 import (  # noqa: F401
    _check_cascode_helper,
    _check_current_mirror,
    _check_diode_load,
    _check_dual_feedforward_pair,
    _check_loop_filter_pole,
    _check_pole_tracking_cap,
)


# Third-batch (v0.0.92) check functions
from ._checks_batch3 import (  # noqa: F401
    _check_common_mode_feedback,
    _check_dynamic_bias,
    _check_gain_boost_amp,
    _check_output_class_AB,
    _check_regulated_cascode,
    _check_sample_hold_switch,
)


# Fourth-batch (v0.0.94) check functions + placeholder + PLL handle
from ._checks_batch4 import (  # noqa: F401
    _check_bandgap_core,
    _check_dropout_pass_device,
    _check_ldo_feedback_divider,
    _check_pll_loop_filter,
    _check_ptat_branch,
    _placeholder_violation,
    _run_pll_loop_ac_via_handle,
)


# Dispatch table + public-API entry points
from ._dispatch import (  # noqa: F401
    _CHECKERS,
    _functional_roles_of,
    check_functional_invariants,
    filter_real_violations,
)


# Gradient-augmented retry feedback
from ._gradient import (  # noqa: F401
    _GRADIENT_SUPPORTED_TARGETS,
    _TARGET_ATTR_UNIT,
    _newton_step_target,
    augment_violation_with_gradient,
    augment_violations_with_gradients,
    check_functional_invariants_with_gradient,
)


# ─── Public __all__ ──────────────────────────────────────────────


__all__ = [
    # Public surface (v0.0.62 / v0.0.86)
    "DEFAULT_TOLERANCE",
    "InvariantViolation",
    "VerificationStatus",
    "category_to_verification_status",
    "contributes_to_proof_token",
    "check_functional_invariants",
    "filter_real_violations",
    "merge_op_quantities",
    "parse_op_quantity_lines",
    # Public gradient surface (v0.0.90)
    "augment_violation_with_gradient",
    "augment_violations_with_gradients",
    "check_functional_invariants_with_gradient",
]
