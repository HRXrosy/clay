"""
Algebraic spatial combinators — backward-compatibility shim (v0.1.12).

The combinator implementation was split out of this single file
(formerly 4225 LOC) into `clay/combinators/` in v0.1.12 (R89 Agent C).
This module is preserved as a re-export shim so every existing import
path continues to resolve:

  * `from clay.stage_combinators import *`
  * `from clay.stage_combinators import cascade, compensate, ...`
  * `from clay.stage_combinators import _autosize_compensate`  (internals)

Every symbol the legacy single-file module exposed is re-exported from
`clay.combinators` here. The split-module layout is documented in
`clay/combinators/__init__.py`.

For the four legacy entry surfaces:

  * `stage_a >> stage_b`               — sequential cascade
  * `feedforward(a, b, gain=k)`        — cross-stage feedforward gm path
  * `compensate(stage, miller=Cc, ...)` — frequency compensation wrap
  * `mirror_match(stage_a, stage_b)`   — operating-point coupling

And the v0.0.95 / v0.0.99 additions:

  * `decouple(stage, ...)`            — supply-rail decoupling
  * `sample_and_hold(stage, ...)`     — track-and-hold combinator

See `clay/combinators/` for the actual implementation modules.
"""

from __future__ import annotations

# Re-export the full public + internal surface from the split package.
# The wildcard import gives us `__all__` (the public surface);
# explicitly re-importing the underscore-prefixed names below covers
# the internals that the legacy module exposed (tests reach into
# `_build_mirror_op_probe_lines`, `_measure_psrr`, `_autosize_*`, etc.).
from .combinators import *  # noqa: F401, F403

# Legacy transitive re-exports: the single-file `stage_combinators.py`
# imported these from `clay.stage_interface` and `clay.ir` at module
# top-level, which made them accessible via `clay.stage_combinators.X`.
# `clay.combinator_algebra.from_composed_stage` reaches for
# `clay.stage_combinators.ComposedStage` specifically; the shim keeps
# that path live so the algebra-bridge import doesn't break.
from .ir import Cell, Device, Measure, Spec, Testbench  # noqa: F401
from .stage_interface import (  # noqa: F401
    ComposeError,
    ComposeTypeError,
    ComposedStage,
    PortDecl,
    SpecTarget,
    StageInterface,
    _interface_to_composed_leaf,
    check_compose,
    get_stage_interface,
    has_stage_interface,
    register_stage_interface,
)

# Underscore-prefixed re-exports — `from clay.combinators import *` does
# NOT pull these in (they're filtered by the `_` prefix convention),
# but the legacy module made them importable individually, so we list
# them explicitly here.
from .combinators._common import (  # noqa: F401
    _bootstrap_combinator_extras,
    _is_finite,
    _leftmost_leaf_template_name,
    _resolve,
    _resolve_autosize_testbench,
)
from .combinators.cascade import (  # noqa: F401
    _builder_accepts_kw,
    _inject_cascade_auto_bias,
    _lower_cascade,
    _merge_cascade_interface,
    _merge_spec_targets,
    _split_substage_kwargs,
    _stitch_devices,
    _union_ports,
)
from .combinators.feedforward import _lower_feedforward  # noqa: F401
from .combinators.compensate import _lower_compensate  # noqa: F401
from .combinators.compensate_autosize import (  # noqa: F401
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
from .combinators.mirror_match import (  # noqa: F401
    _collect_internal_nets,
    _device_supplies_shared_rail,
    _format_mirror_print_directives,
    _identify_mirror_pairs,
    _lower_mirror_match,
    _mirror_pairs_of,
    _resolve_device_current,
    _suggest_mirror_adjustment,
)
from .combinators.mirror_match_runtime import (  # noqa: F401
    _PROBE_LINE_RE,
    _branch_currents_from_probes,
    _build_mirror_op_probe_lines,
    _default_mirror_match_testbench,
    _inject_mirror_probe_lines_into_netlist,
    _parse_probe_log,
    _pdk_model_for_device,
)
from .combinators.decouple import (  # noqa: F401
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
)
from .combinators.sample_and_hold import (  # noqa: F401
    _SH_ADJUST_MAX_FACTOR,
    _SH_BW_W_GAIN_PER_DECADE,
    _SH_OFF_LEAK_CURRENT,
    _SH_PASS_L,
    _SH_SEED_HOLD_CAP,
    _SH_SEED_SWITCH_W,
    _lower_sample_and_hold,
    _sample_hold_newton_adjust,
)
