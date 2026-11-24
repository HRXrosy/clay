"""Clay flow automation + management.

Turns Clay from "language + point tools" into "language + flow automation +
management". The design's flow and acceptance contract live in the Clay source
as ``flow {}`` / ``quality {}`` blocks (see clay/grammar.lark); this package
compiles them into a data-driven :class:`CircuitSpec` and runs a generic,
incremental, recorded flow over the universal analog steps
(compile -> layout -> drc -> lvs -> pex -> post -> intent -> sign).

Extreme restraint against case-by-case hardcoding: everything circuit-specific
is DATA discovered from the case packages, never ``if circuit == "..."`` code.
"""

from clay.flow.spec import (  # noqa: F401
    CircuitSpec,
    ConsistencySpec,
    FlowContext,
    FlowStep,
    GateSpec,
    KpiSpec,
    StepResult,
    compile_flow_spec,
    discover_registry,
)

__all__ = [
    "CircuitSpec",
    "ConsistencySpec",
    "FlowContext",
    "FlowStep",
    "GateSpec",
    "KpiSpec",
    "StepResult",
    "compile_flow_spec",
    "discover_registry",
]
