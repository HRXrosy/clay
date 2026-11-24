"""
Feedforward combinator — cross-stage gm path (v0.0.89 + v0.1.12 split).

`feedforward(source, sink, gain=k)` declares a small-signal gm path
from `source.input` to `sink.output`. Both stages must be structurally
compatible (same `check_compose` rules as `>>`). The lowering inserts
a single MOSFET (or pair, for diff inputs) sourced from `vdd` or
`vss` depending on the source-stage input polarity.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..ir import Cell, Device, Spec
from ..stage_interface import (
    ComposeTypeError,
    ComposedStage,
    check_compose,
)
from ._common import StageRef, TemplateBuilder, _is_finite, _resolve


# ─── Feedforward ────────────────────────────────────────────────────

def feedforward(
    source: StageRef,
    sink: StageRef,
    *,
    gain: float = 1.0,
) -> ComposedStage:
    """`feedforward(a, b, gain=k)` — cross-stage feedforward path.

    Declares a gm-path from `source.input` (first stage's input) to
    `sink.output` (final stage's output) with the given small-signal
    gain weight. Both stages must already be in cascade — `feedforward`
    is a structural modifier on top of an existing cascade composition,
    not a replacement for `>>`.

    Type-check rules:
      * `source.input_ports` and `sink.output_ports` must have
        compatible polarity multisets. For the common 1-input → 1-output
        case both must be single-ended; for differential inputs the FF
        path resolves to a differential pair (handled in lowering).
      * `gain` must be a finite real number.

    Lowering inserts a single MOSFET (or pair, for diff inputs) sized
    to provide the requested transconductance contribution:
      * For PMOS-input source stages the FF device is a PMOS sourced
        from `vdd` with gate at `source.input` and drain at
        `sink.output`. The DC bias mirrors the existing source-stage
        tail bias (`pbias`) so the FF device sits in saturation
        without any extra kwarg from the caller.
      * For NMOS-input source stages the FF device is an NMOS sinking
        to `vss` with gate at `source.input` and drain at
        `sink.output`. The DC bias mirrors the existing `nbias` rail.
      * The polarity choice is decided at lowering time by inspecting
        the *source* stage's first input PMOS/NMOS sizing block; the
        decision lands on `meta['ff_polarity']` so downstream tooling
        can introspect.

    Meta fields populated:
      * `meta['gain']`         — caller-supplied weight.
      * `meta['ff_polarity']` — `'pmos'` / `'nmos'` (set at lowering).
    """
    s = _resolve(source)
    k = _resolve(sink)

    # FF path needs symmetric endpoints on both stages — the same
    # structural check the cascade uses (output count + polarity).
    errors = check_compose(s.interface, k.interface)
    if errors:
        # Reuse the same error vocabulary — the FF path inherits the
        # cascade's structural requirements.
        raise ComposeTypeError(errors)

    if not isinstance(gain, (int, float)) or not _is_finite(float(gain)):
        raise ValueError(
            f"feedforward(gain=...) must be a finite real number, "
            f"got {gain!r}"
        )

    # The FF composition keeps the SINK's external surface — a FF path
    # is a modifier on `sink`, not a new top-level chain.
    composed = ComposedStage(
        kind="feedforward",
        interface=k.interface,
        parts=(s, k),
        meta={"gain": float(gain)},
    )
    return composed


# ─── Feedforward lowering (v0.0.89) ────────────────────────────────


def _lower_feedforward(
    source: ComposedStage,
    sink: ComposedStage,
    meta: Dict[str, Any],
) -> TemplateBuilder:
    """Build the feedforward builder.

    Strategy: build the SINK stage's Cell IR through its own lowered
    builder, then append one PMOS or NMOS gm device whose gate connects
    to the SOURCE stage's input port and whose drain connects to the
    sink stage's output port. The polarity (`pmos` vs `nmos`) is
    decided from the source stage's first input device polarity:
      * PMOS-input source stage → PMOS feedforward device sourced from
        vdd; gate bias comes from the source stage's natural input DC
        bias (which the testbench rig already provides).
      * NMOS-input source stage → NMOS feedforward device sinking to
        vss; same gate-bias inheritance.

    For the differential-input case (2 input ports on `source`) the
    feedforward expands into a *pair* of devices — one per polarity-
    matched input — drain-merged onto the sink's output. The output is
    single-ended for every current first-cohort sink stage so the pair
    naturally merges at `out`.

    The transconductance contribution is controlled by the `gain`
    kwarg (recorded on `composed.meta['gain']`). We translate it into
    a width multiplier: `M_ff = max(1, round(gain * 4))` so a `gain=1`
    feedforward gets 4× the minimum drive, `gain=0.5` gets 2×, etc.
    The exact mapping is chosen so the FF path's gm contribution is
    of the same order as the sink's CS gm — i.e. a real signal, not a
    nuisance device. Future rounds will plumb a proper gm-target.

    Implementation notes:
      * The FF device's source rail mirrors the polarity choice (PMOS →
        vdd, NMOS → vss). Bulk follows source for both. This is the
        same convention every stage_template uses internally, so the
        cascade-time net rename map applies naturally.
      * The FF device's gate is the source stage's FIRST input port
        name (e.g. `inn` on a diff-input source, `in` on a SE-input
        source). The mapping uses the *composed* surface so the caller
        sees the FF wired to whatever input net the cell exposes.
    """
    # The sink's own builder produces the structural baseline; we
    # decorate its output with the FF device.
    from ._lower import lower_to_template
    inner_builder = lower_to_template(sink)
    ff_gain = float(meta.get("gain", 1.0))

    # Decide polarity from the source stage's input device type. The
    # canonical way is to peek at the LEFT stage's lowered Cell — but
    # to avoid an unnecessary build, we use the source's interface
    # `compatible_compensation` + `template_name` to infer. Every
    # first-cohort stage uses PMOS inputs (the standard sky130 default);
    # the explicit NMOS-input template `two_stage_miller_nmos` is the
    # one exception. We default to PMOS unless the template name carries
    # `_nmos_` or starts with `nmos_`.
    src_name = source.template_name.lower()
    ff_polarity = "nmos" if (
        "_nmos" in src_name or src_name.startswith("nmos_")
    ) else "pmos"

    # Record the polarity choice back on `meta` (mutating the frozen
    # dataclass field is OK because dataclass field defaults are
    # mutable dicts — we update the dict, not the dataclass).
    meta["ff_polarity"] = ff_polarity

    # Width / length choices for the FF device. These mirror typical
    # CS-stage sizing and scale by the `gain` weight.
    ff_W_unit = 4e-6                 # base width per multiplier
    ff_L = 0.5e-6                    # short channel for high gm/Id

    # Convert ff_gain → integer multiplier (clamped at ≥1) so the device
    # has a meaningful drive contribution.
    ff_M = max(1, int(round(ff_gain * 4)))

    # The source stage's first input port name — the FF gate net.
    if not source.input_ports:
        raise ValueError(
            f"feedforward source stage {source.template_name!r} has no "
            "input_ports; cannot wire a FF path."
        )
    ff_gate_nets = [p.name for p in source.input_ports]

    # The sink stage's first output port — the FF drain net.
    if not sink.output_ports:
        raise ValueError(
            f"feedforward sink stage {sink.template_name!r} has no "
            "output_ports; cannot wire a FF path."
        )
    ff_drain_net = sink.output_ports[0].name

    def _builder(cell_name: str, spec: Spec, **kwargs: Any) -> Cell:
        # Allow the caller to override the FF device sizing via the
        # `ff_W` / `ff_L` / `ff_M` kwargs. The base inner builder
        # consumes the rest.
        ff_W_override = kwargs.pop("ff_W", ff_W_unit)
        ff_L_override = kwargs.pop("ff_L", ff_L)
        ff_M_override = kwargs.pop("ff_M", ff_M)

        inner_cell = inner_builder(cell_name, spec, **kwargs)

        # The FF device(s) — one per source input port. For a
        # differential source the two devices share the drain (sink
        # output) so they form a diff-pair-style FF; for a single-ended
        # source there's just one device.
        ff_devices: List[Device] = []
        for idx, gate_net in enumerate(ff_gate_nets):
            instance = f"Mff_{idx}" if len(ff_gate_nets) > 1 else "Mff"
            if ff_polarity == "pmos":
                pins = {
                    "d": ff_drain_net,
                    "g": gate_net,
                    "s": "vdd",
                    "b": "vdd",
                }
            else:
                pins = {
                    "d": ff_drain_net,
                    "g": gate_net,
                    "s": "vss",
                    "b": "vss",
                }
            ff_devices.append(Device(
                instance_name=instance,
                device_class=ff_polarity,
                pdk_variant=None,
                pins=pins,
                sizing={
                    "W": float(ff_W_override),
                    "L": float(ff_L_override),
                    "M": float(ff_M_override),
                },
            ))

        return Cell(
            name=inner_cell.name,
            ports=list(inner_cell.ports),
            spec=spec,
            devices=list(inner_cell.devices) + ff_devices,
            pdk_id=inner_cell.pdk_id,
        )

    _builder.__name__ = (
        f"feedforward__{source.template_name}__{sink.template_name}"
    )
    return _builder
