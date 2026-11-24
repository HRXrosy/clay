"""
Shared types + helpers used across the combinators split (v0.1.12).

This module is the dependency root for `clay/combinators/`: every other
file in this directory imports from here (and never the other way
around). It carries:

  * `StageRef` — the union of input shapes every combinator accepts.
  * `TemplateBuilder` — the `(cell_name, spec, **kwargs) -> Cell`
    contract every lowered combinator returns.
  * `stage` / `_resolve` — the `StageRef -> ComposedStage` normaliser.
  * `_is_finite` — the small "no NaN / no inf" guard used by every
    combinator's argument validation.
  * Testbench-resolution helpers (`_leftmost_leaf_template_name`,
    `_resolve_autosize_testbench`) shared by `compensate_autosize.py`
    and `decouple.py`.
  * `_bootstrap_combinator_extras` — registers the
    `common_source_pmos_input` StageInterface at module import (idempotent).

The R89 C split preserves byte-exact behaviour from the v0.0.99
single-file `clay/stage_combinators.py` — `from clay.stage_combinators
import *` continues to surface everything in this module via the shim
+ `__init__.py` re-export chain.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

from ..ir import Cell
from ..stage_interface import (
    ComposedStage,
    PortDecl,
    SpecTarget,
    StageInterface,
    _interface_to_composed_leaf,
    get_stage_interface,
    has_stage_interface,
    register_stage_interface,
)


# ─── Bootstrap extra interfaces needed by combinator targets ────────
#
# The v0.0.87 first cohort (6 templates) covers diff-input topologies.
# To exercise `>>` cascade at the e2e level we additionally need a
# single-ended input stage so a typical 2-stage opamp can chain into
# a 3rd CS gain stage:
#
#   two_stage_miller_pmos (diff_in → single_ended_out)
#       >> common_source_pmos_input (single_ended_in → single_ended_out)
#
# `common_source_pmos_input` already has a builder and testbench
# registered in `clay.stage_templates`; it just lacked a typed
# interface. We register that interface here so combinator tests
# don't have to redo it inline.

def _bootstrap_combinator_extras() -> None:
    """Register StageInterface declarations the combinator slice needs.

    Currently registers:
      * `common_source_pmos_input` — single-ended in/out CS stage.

    Idempotent re-registration is OK; the underlying
    `register_stage_interface` short-circuits when the declaration
    matches.
    """
    if has_stage_interface("common_source_pmos_input"):
        return
    register_stage_interface(StageInterface(
        template_name="common_source_pmos_input",
        input_ports=[
            PortDecl(name="in", role="signal_in", polarity="single_ended"),
        ],
        output_ports=[
            PortDecl(name="out", role="signal_out", polarity="single_ended"),
        ],
        bias_ports=[],
        supply_ports=[
            PortDecl(name="vdd", role="supply", polarity="none"),
            PortDecl(name="vss", role="supply", polarity="none"),
        ],
        spec_targets={
            "gain_dc": SpecTarget("gain_dc", "db", ">=", 20.0),
            "gbw":     SpecTarget("gbw",     "freq", ">=", 5e5),
            "pm":      SpecTarget("pm",      "angle", ">=", 45.0),
        },
        compatible_compensation=[],
    ))


# Trigger the bootstrap once at import time. Idempotent — re-imports
# of this module short-circuit on the registry check.
_bootstrap_combinator_extras()

# ─── Public entry points ────────────────────────────────────────────

StageRef = Union[str, StageInterface, ComposedStage]
"""Anything a combinator accepts as an input stage:

  * `str` — registered template_name (looked up via
    `get_stage_interface`).
  * `StageInterface` — bare typed declaration (wrapped as a leaf).
  * `ComposedStage` — passed through.
"""


# Callable contract every lowered combinator returns. Declared here
# so each split module can reference it without circular re-imports.
TemplateBuilder = Callable[..., Cell]


def stage(ref: StageRef) -> ComposedStage:
    """Normalise a stage reference to a `ComposedStage`.

    Equivalent to `_resolve(ref)` but exported for users who want an
    explicit conversion. Useful when threading a stage through a list
    comprehension or when the inferred type is ambiguous.
    """
    return _resolve(ref)


def _resolve(ref: StageRef) -> ComposedStage:
    """Normalise any `StageRef` to a `ComposedStage`.

    String → registry lookup → wrap as leaf.
    StageInterface → wrap as leaf.
    ComposedStage → return as-is.
    """
    if isinstance(ref, ComposedStage):
        return ref
    if isinstance(ref, StageInterface):
        return _interface_to_composed_leaf(ref)
    if isinstance(ref, str):
        if not has_stage_interface(ref):
            raise KeyError(
                f"no StageInterface registered for {ref!r}. Either pass a "
                f"StageInterface instance or register {ref!r} first via "
                "clay.stage_interface.register_stage_interface()."
            )
        return _interface_to_composed_leaf(get_stage_interface(ref))
    raise TypeError(
        f"combinator argument must be str | StageInterface | "
        f"ComposedStage, got {type(ref).__name__}"
    )


def _is_finite(x: float) -> bool:
    """Return True iff x is a finite real number (no NaN, no inf)."""
    return x == x and x not in (float("inf"), float("-inf"))


def _leftmost_leaf_template_name(composed: ComposedStage) -> Optional[str]:
    """Walk to the leftmost leaf and return its raw `template_name`.

    For a leaf the `parts[0]` is the registered template name (a str).
    For composites we recurse into `parts[0]` (left side / source side
    for FF). Returns None if the walk runs into an unexpected shape.
    """
    cur: Any = composed
    while isinstance(cur, ComposedStage):
        if cur.kind == "leaf":
            if cur.parts and isinstance(cur.parts[0], str):
                return cur.parts[0]
            return None
        if not cur.parts:
            return None
        cur = cur.parts[0]
    if isinstance(cur, str):
        return cur
    return None


def _resolve_autosize_testbench(
    inner: ComposedStage,
) -> Tuple[Optional[Callable], Optional[float]]:
    """Locate the `build_ac_testbench` / `build_ac_testbench_cs` companion
    for the wrapped stage.

    Walks the composition tree until it hits a leaf, then tries to import
    `clay.templates.<leaf_template_name>` and look up its testbench
    builder. The lookup is heuristic and best-effort:

      * `two_stage_miller_pmos` / `two_stage_miller_nmos` → `build_ac_testbench`
      * `common_source_pmos_input` → `build_ac_testbench_cs` (in `common_source` module)
      * other templates: fall through to `None` (auto-sizing degrades).

    Returns `(builder, supply_volt_or_None)`. The supply is left as None
    when the testbench module handles auto-resolve via `_supply.resolve_supply`.
    """
    # Walk to a leaf template_name. For composites we use the LEFTMOST
    # leaf — the testbench rig of the input stage matches the composed
    # cell's external port shape.
    leaf_name = _leftmost_leaf_template_name(inner)
    if leaf_name is None:
        return None, None

    if leaf_name in ("two_stage_miller_pmos",):
        try:
            from ..templates.two_stage_miller_pmos import build_ac_testbench
            return build_ac_testbench, None
        except ImportError:
            return None, None
    if leaf_name in ("two_stage_miller_nmos",):
        try:
            from ..templates.two_stage_miller_nmos import build_ac_testbench
            return build_ac_testbench, None
        except ImportError:
            return None, None
    if leaf_name in ("common_source_pmos_input",):
        try:
            from ..templates.common_source import build_ac_testbench_cs
            return build_ac_testbench_cs, None
        except ImportError:
            return None, None

    return None, None
