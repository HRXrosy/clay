"""
Combinator lowering dispatch — `ComposedStage` → builder (v0.0.89 + v0.1.12 split).

Hosts the top-level `lower_to_template` function. Each kind dispatches
into the matching `_lower_<kind>` in its sibling module. The lazy
imports prevent circular dependencies between combinator modules
(every `_lower_<kind>` calls back into `lower_to_template` for nested
compositions).
"""

from __future__ import annotations

from ..stage_interface import ComposedStage
from ._common import TemplateBuilder


# ─── Lowering: ComposedStage → builder ──────────────────────────────


def lower_to_template(composed: ComposedStage) -> TemplateBuilder:
    """Lower a `ComposedStage` tree to a Python builder function.

    The returned callable has the same signature as a base stage
    builder — `(cell_name: str, spec: Spec, **kwargs) -> Cell` — so it
    plugs into `clay.verifier.verify`, the CLI's cell_def path, and
    every existing testbench builder without further surface change.

    Lowering strategy by `kind`:
      * `leaf`       → return the registered base builder directly.
      * `cascade`    → return a new builder that calls both sub-builders,
                       merges their device lists with junction wiring,
                       and inserts AC coupling + DC bias resistors at
                       each cascade junction so multi-stage compositions
                       converge to a sane operating point.
      * `feedforward`→ inflates a real gm path (PMOS / NMOS) from
                       source.input → sink.output. The polarity choice
                       lands on `composed.meta['ff_polarity']`.
      * `compensate` → inserts a Miller cap (and optional nulling
                       resistor) between the inner stage's primary
                       internal high-Z node and its output.
      * `mirror`     → ties the two stages' bias rails together by
                       routing one stage's local bias net to the other's
                       (shared rail recorded on `composed.meta`).
      * `decouple`   → inserts a series isolation PMOS + decoupling cap
                       on the inner stage's supply rail (v0.0.95).
      * `sample_and_hold` → inserts an NMOS pass switch (gate driven by
                       clkS) plus a hold cap on the inner stage's output
                       net (v0.0.99).
    """
    if composed.kind == "leaf":
        template_name = composed.parts[0]
        from ..stage_templates import get_stage
        return get_stage(template_name)

    if composed.kind == "cascade":
        from .cascade import _lower_cascade
        left, right = composed.parts  # type: ignore[misc]
        return _lower_cascade(left, right, composed.interface)

    if composed.kind == "feedforward":
        from .feedforward import _lower_feedforward
        source, sink = composed.parts  # type: ignore[misc]
        return _lower_feedforward(source, sink, composed.meta)

    if composed.kind == "compensate":
        from .compensate import _lower_compensate
        (inner,) = composed.parts  # type: ignore[misc]
        return _lower_compensate(inner, composed.meta)

    if composed.kind == "mirror":
        from .mirror_match import _lower_mirror_match
        first, second = composed.parts  # type: ignore[misc]
        return _lower_mirror_match(first, second, composed.meta)

    if composed.kind == "decouple":
        from .decouple import _lower_decouple
        (inner,) = composed.parts  # type: ignore[misc]
        return _lower_decouple(inner, composed.meta)

    if composed.kind == "sample_and_hold":
        # v0.0.99 (Round 75 C): track-and-hold combinator wraps the
        # inner stage's output with an NMOS pass switch + hold cap.
        from .sample_and_hold import _lower_sample_and_hold
        (inner,) = composed.parts  # type: ignore[misc]
        return _lower_sample_and_hold(inner, composed.meta)

    raise AssertionError(
        f"lower_to_template: unhandled kind={composed.kind!r}"
    )
