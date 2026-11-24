"""clay.elaborate — desugar the rich EDALLM .clay surface DOWN to the
clean-room 2-primitive verified core (Cell + Constraint@Locus).

The clean-room kernel (separate Lean repo) accepts a minimal surface:

    leaf|node interface { <role> <port>, ... } model "<m>"
    [composition { nets { ... } inst <child> bind { ... } ... }]
    [constraints { constraint <stratum> <domain> <checker> "<pred>", ... }]

This package emits exactly that surface from a `clay.ir.Cell` (post-compile,
devices populated). Structural constructs (cells, device primitives,
instantiation, hierarchy) become Cell/Composition/Instance; constraint-like
constructs (spec, sizing, annotate, layout-intent) become Constraint @ Locus.
Orchestration (flow/quality/policy/verify) is process, not structure — it is
honestly NOT coerced into the structural core (deferred, recorded only).

Entry point: ``emit_cell(cell) -> str`` returns parseDesign-accepted text.
"""

from .to_core import (
    emit_cell,
    emit_leaf,
    emit_node,
    emit_constraints,
    DeferredRecord,
)

__all__ = [
    "emit_cell",
    "emit_leaf",
    "emit_node",
    "emit_constraints",
    "DeferredRecord",
]
