"""
clay.combinators — algebraic spatial combinators (v0.1.12 split).

The combinator slice was split out of the legacy single-file
`clay/stage_combinators.py` (4225 LOC) in v0.1.12 (R89 Agent C). The
public API is byte-exact preserved: `from clay.stage_combinators
import *` and every `from clay.stage_combinators import <name>` path
keeps working through the shim in `clay/stage_combinators.py`, which
re-exports this package.

Split layout:
  * `_common.py`              — `StageRef`, `TemplateBuilder`,
                                `stage` / `_resolve`, `_is_finite`,
                                shared testbench-resolution helpers.
  * `cascade.py`              — `cascade`, `cascade_chain`, lowering,
                                auto-bias injection.
  * `feedforward.py`          — `feedforward`, lowering.
  * `compensate.py`           — `compensate`, lowering.
  * `compensate_autosize.py`  — Newton + bisection + Adjoint search.
  * `mirror_match.py`         — `mirror_match`, lowering,
                                `MirrorMismatchViolation`,
                                `check_mirror_match_invariant`,
                                `inject_mirror_match_measures`.
  * `mirror_match_runtime.py` — `run_mirror_match_check` (closed-loop
                                ngspice check) + probe builders / parsers.
  * `decouple.py`             — `decouple`, lowering, auto-sizing.
  * `sample_and_hold.py`      — `sample_and_hold`, lowering, auto-sizing.
  * `_lower.py`               — `lower_to_template` dispatch.
"""

from __future__ import annotations

# ─── Public API surface ────────────────────────────────────────────
#
# Every re-export here is the same symbol the legacy single-file
# `clay/stage_combinators.py` exposed at module top-level. The order
# below mirrors the original section ordering for grep-friendly diffing.

from ._common import (
    StageRef,
    TemplateBuilder,
    _bootstrap_combinator_extras,
    _is_finite,
    _leftmost_leaf_template_name,
    _resolve,
    _resolve_autosize_testbench,
    stage,
)
from .cascade import (
    _builder_accepts_kw,
    _inject_cascade_auto_bias,
    _lower_cascade,
    _merge_cascade_interface,
    _merge_spec_targets,
    _split_substage_kwargs,
    _stitch_devices,
    _union_ports,
    cascade,
    cascade_chain,
)
from .feedforward import (
    _lower_feedforward,
    feedforward,
)
from .compensate import (
    _lower_compensate,
    compensate,
)
from .compensate_autosize import (
    _AUTOSIZE_CC_MAX,
    _AUTOSIZE_CC_MIN,
    _AUTOSIZE_MAX_ITERS,
    _AUTOSIZE_NEWTON_CLIP_FACTOR,
    _AUTOSIZE_NEWTON_EMPIRICAL_SLOPE_DEG_PER_DECADE,
    _AUTOSIZE_NEWTON_MAX_ITERS,
    _AUTOSIZE_PM_TOL,
    _AUTOSIZE_SEED_CC,
    _AUTOSIZE_SEED_RZ,
    _adjoint_direction_hint,
    _adjoint_pm_cc_via_op_resolve,
    _adjoint_slope_pm_per_log10_cc,
    _autosize_compensate,
    _autosize_io_nodes,
    _bisection_compensate,
    _build_compensate_probe_cell,
    _measure_pm,
    _newton_step_compensate,
    _probe_pm_with_trace,
)
from .mirror_match import (
    MirrorMismatchViolation,
    _collect_internal_nets,
    _device_supplies_shared_rail,
    _format_mirror_print_directives,
    _identify_mirror_pairs,
    _lower_mirror_match,
    _mirror_pairs_of,
    _resolve_device_current,
    _suggest_mirror_adjustment,
    check_mirror_match_invariant,
    inject_mirror_match_measures,
    mirror_match,
)
from .mirror_match_runtime import (
    _PROBE_LINE_RE,
    _branch_currents_from_probes,
    _build_mirror_op_probe_lines,
    _default_mirror_match_testbench,
    _inject_mirror_probe_lines_into_netlist,
    _parse_probe_log,
    _pdk_model_for_device,
    run_mirror_match_check,
)
from .decouple import (
    _DECOUPLE_ADJUST_DB_PER_DECADE_CAP,
    _DECOUPLE_ADJUST_DB_PER_DECADE_W,
    _DECOUPLE_ADJUST_MAX_FACTOR,
    _DECOUPLE_PASS_L,
    _DECOUPLE_SEED_CAP,
    _DECOUPLE_SEED_W,
    _autosize_decouple,
    _build_decouple_probe_cell,
    _decouple_newton_adjust,
    _lower_decouple,
    _measure_psrr,
    decouple,
)
from .sample_and_hold import (
    _SH_ADJUST_MAX_FACTOR,
    _SH_BW_W_GAIN_PER_DECADE,
    _SH_OFF_LEAK_CURRENT,
    _SH_PASS_L,
    _SH_SEED_HOLD_CAP,
    _SH_SEED_SWITCH_W,
    _lower_sample_and_hold,
    _sample_hold_newton_adjust,
    sample_and_hold,
)
from ._lower import lower_to_template


__all__ = [
    # Common
    "StageRef",
    "TemplateBuilder",
    "stage",
    # Cascade
    "cascade",
    "cascade_chain",
    # Feedforward
    "feedforward",
    # Compensate
    "compensate",
    # Mirror match
    "mirror_match",
    "MirrorMismatchViolation",
    "check_mirror_match_invariant",
    "inject_mirror_match_measures",
    "run_mirror_match_check",
    # Decouple
    "decouple",
    # Sample-and-hold
    "sample_and_hold",
    # Lowering dispatch
    "lower_to_template",
]
