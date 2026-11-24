"""
Clay functional-role invariant runtime — backward-compatibility shim (v0.1.14).

The invariants implementation was split out of this single file
(formerly 5331 LOC) into `clay/invariants_modules/` in v0.1.14
(R91 Agent B). This module is preserved as a re-export shim so every
existing import path continues to resolve:

  * `from clay.invariants import DEFAULT_TOLERANCE, InvariantViolation`
  * `from clay.invariants import check_functional_invariants`
  * `from clay.invariants import filter_real_violations`
  * `from clay.invariants import merge_op_quantities, parse_op_quantity_lines`
  * `from clay.invariants import _CHECKERS, _KNOWN_KINDS, _TARGET_ATTR_BY_KIND`
  * `from clay.invariants import _check_pll_loop_filter` and the
    other 22 per-keyword `_check_*` functions
  * `from clay.invariants import _HAS_CLAY_IR_INVARIANTS, _RUST_INVARIANT_SUBSET`
  * `from clay.invariants import _newton_step_target,
    augment_violation_with_gradient,
    augment_violations_with_gradients,
    check_functional_invariants_with_gradient`

Every symbol the legacy single-file module exposed is re-exported
from `clay.invariants_modules` here. The split-module layout is
documented in `clay/invariants_modules/__init__.py`.

The shim preserves byte-exact behavior on every public + internal
surface:
  * R82 B / R83 B / R84 B / R85 B / R86 B BL-012 Tier B.5 23/23
    Rust delegation hook (`_try_rust_check_functional_role`,
    `_build_violation_from_rust`, kind-specific Python rebuilders)
  * R85 D `_emit_value_violation` byte-exact template
  * Round-62 / Round-67 / Round-68 / Round-71 23 check functions
  * v0.0.87 OP-quantity log parser + merge
  * v0.0.90 gradient-augmented retry signal (Newton-step suggestions)
  * v0.0.95 R72 C live closed-loop PLL AC handle path

See `clay/invariants_modules/` for the actual implementation modules.
"""

from __future__ import annotations

# Re-export the full public + internal surface from the split package.
# The wildcard import gives us `__all__` (the public surface);
# explicitly re-importing the underscore-prefixed names below covers
# the internals that the legacy module exposed (tests reach into
# `_CHECKERS`, `_KNOWN_KINDS`, `_check_pll_loop_filter`,
# `_HAS_CLAY_IR_INVARIANTS`, etc.).
from .invariants_modules import *  # noqa: F401, F403

# Violation dataclass + role-kind constants.
from .invariants_modules._violation import (  # noqa: F401
    DEFAULT_TOLERANCE,
    InvariantViolation,
    VerificationStatus,
    _KNOWN_KINDS,
    _TARGET_ATTR_BY_KIND,
    category_to_verification_status,
    contributes_to_proof_token,
)

# Numeric helpers.
from .invariants_modules._helpers import (  # noqa: F401
    _as_float,
    _format_value,
    _relative_error,
    _within_tolerance,
)

# OP log parser + merge.
from .invariants_modules._op_parser import (  # noqa: F401
    _OP_QUANTITY_LINE_RE,
    _device_name_matches_role,
    merge_op_quantities,
    parse_op_quantity_lines,
)

# Shared value-check emitter + AC gain resolver.
from .invariants_modules._emit import (  # noqa: F401
    _emit_value_violation,
    _resolve_ac_gain_linear,
)

# Rust delegation hook (BL-012 Tier B.5 23/23).
from .invariants_modules._rust_delegation import (  # noqa: F401
    _HAS_CLAY_IR_INVARIANTS,
    _RUST_INVARIANT_SUBSET,
    _clay_ir_ext,
    _op_ac_dict_to_pairs,
    _r84b_should_defer_to_python,
    _role_attrs_split,
    _rust_invariants_enabled,
    _try_rust_check_functional_role,
)

# Rust-side tagged-tuple → byte-exact Python rebuilders.
from .invariants_modules._rust_builders import (  # noqa: F401
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

# 23 per-keyword check functions.
from .invariants_modules._checks_batch1 import (  # noqa: F401
    _check_compensation_zero,
    _check_dominant_pole_cap,
    _check_feedforward_path,
    _check_input_transconductor,
    _check_output_buffer,
    _check_tail_current_source,
)
from .invariants_modules._checks_batch2 import (  # noqa: F401
    _check_cascode_helper,
    _check_current_mirror,
    _check_diode_load,
    _check_dual_feedforward_pair,
    _check_loop_filter_pole,
    _check_pole_tracking_cap,
)
from .invariants_modules._checks_batch3 import (  # noqa: F401
    _check_common_mode_feedback,
    _check_dynamic_bias,
    _check_gain_boost_amp,
    _check_output_class_AB,
    _check_regulated_cascode,
    _check_sample_hold_switch,
)
from .invariants_modules._checks_batch4 import (  # noqa: F401
    _check_bandgap_core,
    _check_dropout_pass_device,
    _check_ldo_feedback_divider,
    _check_pll_loop_filter,
    _check_ptat_branch,
    _placeholder_violation,
    _run_pll_loop_ac_via_handle,
)

# Dispatch table + public-API entry points.
from .invariants_modules._dispatch import (  # noqa: F401
    _CHECKERS,
    _functional_roles_of,
    check_functional_invariants,
    filter_real_violations,
)

# Gradient-augmented retry feedback (v0.0.90).
from .invariants_modules._gradient import (  # noqa: F401
    _GRADIENT_SUPPORTED_TARGETS,
    _TARGET_ATTR_UNIT,
    _newton_step_target,
    augment_violation_with_gradient,
    augment_violations_with_gradients,
    check_functional_invariants_with_gradient,
)

# Suggestion helpers (legacy reach-in path — tests + downstream tooling).
from .invariants_modules._suggestions import (  # noqa: F401
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
