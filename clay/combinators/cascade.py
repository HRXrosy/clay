"""
Cascade combinator (`>>`) — sequential stage composition (v0.0.89 + v0.1.12 split).

Surfaces `cascade(a, b)`, the ergonomic `cascade_chain([a, b, c, ...])`,
the cascade lowering `_lower_cascade`, and the v0.0.89 auto-bias
injection `_inject_cascade_auto_bias`. Helpers for substage kwarg
routing and device stitching live here too so the cascade module owns
the full junction-wiring story.

The cascade junction protocol:
  * `__junc_<i>` is the internal net that fuses `left.output_ports[i]`
    with `right.input_ports[i]`.
  * Each cascade builder defers to `lower_to_template` (imported lazily
    inside the builder to break import cycles) for both substages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..ir import Cell, Device, Spec
from ..stage_interface import (
    ComposeTypeError,
    ComposedStage,
    PortDecl,
    SpecTarget,
    StageInterface,
    check_compose,
)
from ._common import StageRef, TemplateBuilder, _resolve


# ─── Cascade (`>>`) ─────────────────────────────────────────────────

def cascade(left: StageRef, right: StageRef) -> ComposedStage:
    """`left >> right` — sequential cascade combinator.

    Type-checks via `check_compose(left, right)`; raises
    `ComposeTypeError` on failure (carries the structured error list).

    The composed surface inherits:
      * `input_ports`   from `left`
      * `output_ports`  from `right`
      * `bias_ports`    = `left.bias` ∪ `right.bias` (deduplicated by
        port name)
      * `supply_ports`  = `left.supply` ∪ `right.supply` (deduplicated)
      * `spec_targets`  — KPI promises optimistically take the BEST
        bound across the two stages (max for `>=` promises, min for
        `<=`); equal-direction conflicts are resolved by stronger
        bound, mixed-direction KPIs are dropped silently (the
        downstream type-check surfaces them).
      * `compatible_compensation` — union of both stages' tags.
    """
    a = _resolve(left)
    b = _resolve(right)

    errors = check_compose(a.interface, b.interface)
    if errors:
        raise ComposeTypeError(errors)

    composed_iface = _merge_cascade_interface(a.interface, b.interface)
    return ComposedStage(
        kind="cascade",
        interface=composed_iface,
        parts=(a, b),
        meta={},
    )


def _merge_cascade_interface(
    a: StageInterface, b: StageInterface,
) -> StageInterface:
    """Build the cascade's StageInterface surface.

    Used by `cascade` once `check_compose` has cleared structural
    compatibility. Renames the composition `template_name` to
    `'<a>__cascade__<b>'` so the registry never collides with the
    base templates.
    """
    bias = _union_ports(a.bias_ports, b.bias_ports)
    supplies = _union_ports(a.supply_ports, b.supply_ports)
    targets = _merge_spec_targets(a.spec_targets, b.spec_targets)
    comps = sorted(
        set(a.compatible_compensation) | set(b.compatible_compensation)
    )
    return StageInterface(
        template_name=f"{a.template_name}__cascade__{b.template_name}",
        input_ports=list(a.input_ports),
        output_ports=list(b.output_ports),
        bias_ports=bias,
        supply_ports=supplies,
        spec_targets=targets,
        compatible_compensation=comps,
    )


def _union_ports(xs: List[PortDecl], ys: List[PortDecl]) -> List[PortDecl]:
    """Union two port lists, deduplicating by port name.

    Preserves order from `xs` first then `ys`. Duplicates by name are
    skipped (first occurrence wins).
    """
    seen: set[str] = set()
    out: List[PortDecl] = []
    for p in list(xs) + list(ys):
        if p.name in seen:
            continue
        seen.add(p.name)
        out.append(p)
    return out


def _merge_spec_targets(
    a: Dict[str, SpecTarget], b: Dict[str, SpecTarget],
) -> Dict[str, SpecTarget]:
    """Merge two KPI promise maps.

    For overlapping KPIs the STRONGER bound wins (max for `>=`, min for
    `<=`). Conflicting directions (one `>=`, one `<=`) drop the entry
    silently — the combinator caller can re-declare a clean target on
    the result if they need to pin a specific value.
    """
    out: Dict[str, SpecTarget] = dict(a)
    for kpi, st in b.items():
        if kpi not in out:
            out[kpi] = st
            continue
        existing = out[kpi]
        if existing.op != st.op:
            del out[kpi]   # mixed-direction → drop
            continue
        if existing.op == ">=":
            stronger = max(existing.target_value, st.target_value)
        elif existing.op == "<=":
            stronger = min(existing.target_value, st.target_value)
        else:
            # `=`, `>`, `<` — keep the existing entry as-is
            continue
        out[kpi] = SpecTarget(
            kpi=kpi, dimension=existing.dimension,
            op=existing.op, target_value=stronger,
        )
    return out


# ─── Cascade chain (ergonomic nested cascade) ──────────────────────


def cascade_chain(stages: List[StageRef]) -> ComposedStage:
    """`cascade_chain([a, b, c, ...])` — n-stage left-associative cascade.

    Equivalent to ``((a >> b) >> c) >> ...`` but reads as a single
    expression for arbitrary chain lengths. Nested cascades round-trip
    through `lower_to_template` exactly the same way — the helper is
    purely surface sugar.

    Raises:
      * `ValueError` if `stages` is empty.
      * `ComposeTypeError` if any adjacent pair fails `check_compose`.
    """
    if len(stages) == 0:
        raise ValueError("cascade_chain() requires at least one stage")
    if len(stages) == 1:
        return _resolve(stages[0])
    composed = _resolve(stages[0])
    for nxt in stages[1:]:
        composed = cascade(composed, nxt)
    return composed


# ─── Cascade lowering ───────────────────────────────────────────────


def _lower_cascade(
    left: ComposedStage,
    right: ComposedStage,
    composed_iface: StageInterface,
) -> TemplateBuilder:
    """Build the cascade builder that fuses both sub-stages' Cell IRs.

    The returned builder:
      1. Calls `left`'s builder with a sub-cell name `<cell_name>_s1`.
      2. Calls `right`'s builder with `<cell_name>_s2`.
      3. Builds the merged device list:
         * Junction wiring — each `left.output_ports[i]` net is renamed
           to a junction node which `right.input_ports[i]` also points
           to. The shared net adopts `right`'s input port name when the
           composed cell's input ports come from `left` (so the
           external surface name `inn` / `inp` / `in` on `left` stays
           intact; `out` from `left` is the one renamed).
         * Internal nets of `right` get a `s2_` prefix to avoid
           collision with `left`'s internal nets (e.g. both stages
           might use `nbias`).
         * Device instance names get the same `s1_` / `s2_` prefix.
         * Supplies (`vdd`, `vss`) are NOT prefixed — they are the
           globally shared supply rails.
      4. Returns a fresh `Cell` whose port surface matches
         `composed_iface`.

    kwargs are routed to BOTH builders by default. Sub-stage-specific
    kwargs can be selected via the `s1_<kwarg>` / `s2_<kwarg>` prefix
    convention; kwargs without a prefix go to whichever builder
    declares them, falling back to "send to both that accept it".
    Unknown kwargs raise TypeError just like a normal builder.
    """
    from ._lower import lower_to_template
    left_builder = lower_to_template(left)
    right_builder = lower_to_template(right)

    # External port surface of the cascade
    composed_ports: List[str] = composed_iface.all_port_names

    # The set of supply names that survive un-prefixed.
    supply_names = {p.name for p in composed_iface.supply_ports}

    # Junction nets: each (left_output_port, right_input_port) pair
    # collapses to a single internal node `__junc_<i>`. The chosen
    # name avoids collision with any external port (the composed
    # surface uses `left.input` + `right.output`, neither of which can
    # start with `__junc_`). Storing the mapping per-side keeps the
    # rename pass linear.
    left_junction_map: Dict[str, str] = {}
    right_junction_map: Dict[str, str] = {}
    for idx, (li, ri) in enumerate(zip(left.output_ports, right.input_ports)):
        junction_net = f"__junc_{idx}"
        left_junction_map[li.name] = junction_net
        right_junction_map[ri.name] = junction_net

    # Composed-cell external port set drives the auto-bias decision —
    # the cell's external `out` is the natural sink for the DC feedback
    # resistor at each `__junc_<i>` so the bias loop closes inside the
    # composed cell rather than relying on the testbench rig.
    composed_output_names = [p.name for p in composed_iface.output_ports]

    def _builder(cell_name: str, spec: Spec, **kwargs: Any) -> Cell:
        # Split kwargs into per-substage buckets if the caller used
        # the `s1_` / `s2_` prefix convention; otherwise route the
        # kwarg to both builders and let each accept-or-reject.
        # `auto_bias=True` (opt-in) injects an AC cap + DC bias resistor
        # at each cascade junction so multi-stage compositions whose
        # external testbench feedback cannot reach the junction (e.g.
        # `two_stage_miller_pmos >> common_source_pmos_input` under the
        # diff-input AC testbench) still settle DC properly. The default
        # `auto_bias=False` preserves v0.0.88 byte-exact lowering for
        # the cascade types whose external testbench already closes a
        # bias loop through both stages (e.g. CS >> CS under the CS
        # self-bias testbench).
        auto_bias = bool(kwargs.pop("auto_bias", False))
        # Allow callers to override the auto-bias cap / resistor values.
        # Defaults are tuned for sky130 1.8 V: `C_ac=1uF` is large enough
        # to be an AC short well below 1 Hz and not load the dominant
        # pole; `R_bias=1G` mimics the standalone CS testbench's R_fb.
        c_ac_default = float(kwargs.pop("auto_bias_cap", 1.0e-6))
        r_bias_default = float(kwargs.pop("auto_bias_res", 1.0e9))
        s1_kwargs, s2_kwargs = _split_substage_kwargs(
            left_builder, right_builder, kwargs,
        )
        # Propagate auto_bias into nested cascade sub-builders. The
        # `_split_substage_kwargs` routing relies on signature inspection
        # which only sees `auto_bias` on cascade-style `_builder`s; we
        # still want nested cascades to inject their own junction
        # auto-bias devices, so we forward the flag explicitly. Leaf
        # builders ignore unknown kwargs (they raise on truly unknown
        # ones — but `auto_bias` is recognised either as a cascade flag
        # or, if not, gets dropped here before reaching a leaf).
        if auto_bias:
            if _builder_accepts_kw(left_builder, "auto_bias"):
                s1_kwargs.setdefault("auto_bias", True)
            if _builder_accepts_kw(right_builder, "auto_bias"):
                s2_kwargs.setdefault("auto_bias", True)

        # Build the two sub-cells with a dummy spec each — only the
        # composed cell carries the user-facing spec; the substage
        # cells are stitched together at the device level so their
        # individual specs are not used in the verify pipeline.
        left_cell = left_builder(f"{cell_name}_s1", spec, **s1_kwargs)
        right_cell = right_builder(f"{cell_name}_s2", spec, **s2_kwargs)

        # Stitch the two device lists.
        merged_devices = _stitch_devices(
            left_cell.devices,
            right_cell.devices,
            left_junction_map=left_junction_map,
            right_junction_map=right_junction_map,
            supply_names=supply_names,
            left_external_ports=left.interface.all_port_names,
            right_external_ports=right.interface.all_port_names,
        )

        if auto_bias:
            # The auto-bias scheme inserts at each `__junc_<i>` an AC
            # coupling cap and a DC feedback resistor from the composed
            # cell's `out` port. The right-stage's *first* input gate
            # gets rewired to the bias-shifted side of the cap so DC
            # is set by the feedback path while AC is fully coupled.
            merged_devices = _inject_cascade_auto_bias(
                merged_devices,
                left_junction_map=left_junction_map,
                right_junction_map=right_junction_map,
                composed_outputs=composed_output_names,
                c_ac=c_ac_default,
                r_bias=r_bias_default,
            )

        return Cell(
            name=cell_name,
            ports=composed_ports,
            spec=spec,
            devices=merged_devices,
            pdk_id=left_cell.pdk_id,
        )

    _builder.__name__ = (
        f"cascade__{left.template_name}__{right.template_name}"
    )
    return _builder


def _builder_accepts_kw(builder: TemplateBuilder, kw_name: str) -> bool:
    """True if `builder`'s signature accepts `kw_name` (named or **kwargs).

    Used to decide whether to forward cascade-level flags (e.g.
    `auto_bias`) into a nested sub-builder. Leaf stage builders that
    do NOT declare `**kwargs` ignore flag entirely (signature inspection
    fails → return False).
    """
    import inspect
    try:
        sig = inspect.signature(builder)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if kw_name in params:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _split_substage_kwargs(
    left_builder: TemplateBuilder,
    right_builder: TemplateBuilder,
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Route caller kwargs to the two sub-builders.

    Routing rules:
      * `s1_<name>=...` → goes to left builder as `<name>=...`
      * `s2_<name>=...` → goes to right builder as `<name>=...`
      * unprefixed `<name>=...` → goes to whichever builder accepts it,
        as discovered via `inspect.signature`. Sent to both if both
        accept (gm_id_diff, etc. usually mean the same thing in both).
    """
    import inspect

    s1: Dict[str, Any] = {}
    s2: Dict[str, Any] = {}

    try:
        left_sig = inspect.signature(left_builder)
        right_sig = inspect.signature(right_builder)
        left_params = set(left_sig.parameters)
        right_params = set(right_sig.parameters)
        left_has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in left_sig.parameters.values()
        )
        right_has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in right_sig.parameters.values()
        )
    except (TypeError, ValueError):
        # If signature inspection fails (e.g. C-callable), send all
        # kwargs to BOTH builders and let them raise.
        return dict(kwargs), dict(kwargs)

    for key, value in kwargs.items():
        if key.startswith("s1_"):
            s1[key[3:]] = value
            continue
        if key.startswith("s2_"):
            s2[key[3:]] = value
            continue
        sent = False
        if key in left_params or left_has_kwargs:
            s1[key] = value
            sent = True
        if key in right_params or right_has_kwargs:
            s2[key] = value
            sent = True
        if not sent:
            # Send to both anyway — let the substage raise a TypeError
            # itself so the diagnostic carries the original signature.
            s1[key] = value
            s2[key] = value

    return s1, s2


def _stitch_devices(
    left_devs: List[Device],
    right_devs: List[Device],
    *,
    left_junction_map: Dict[str, str],
    right_junction_map: Dict[str, str],
    supply_names: set[str],
    left_external_ports: List[str],
    right_external_ports: List[str],
) -> List[Device]:
    """Merge two device lists with junction wiring + collision-safe renames.

    Net renaming rules per stage:
      * Supply nets (`vdd` / `vss`) survive un-prefixed — they are the
        globally shared supply rails.
      * For the LEFT stage:
          - Left's output ports get rewritten to the corresponding
            `__junc_<i>` internal net (the cascade junction).
          - Left's other external ports (inputs / biases) survive
            un-prefixed — they are the composed cell's external ports.
          - Left's internal nets get an `s1_` prefix.
      * For the RIGHT stage:
          - Right's input ports get rewritten to the corresponding
            `__junc_<i>` internal net.
          - Right's other external ports (outputs / biases) survive
            un-prefixed.
          - Right's internal nets get an `s2_` prefix.
      * Instance names always get `s1_` / `s2_` prefixed (regardless
        of supply / external status) — collisions on `Iref`, `Mbn`
        etc. across templates are the common case.
    """
    left_external_set = set(left_external_ports)
    right_external_set = set(right_external_ports)

    def _rename_left(net: str) -> str:
        if net in left_junction_map:
            return left_junction_map[net]
        if net in supply_names:
            return net
        if net in left_external_set:
            return net
        return f"s1_{net}"

    def _rename_right(net: str) -> str:
        if net in right_junction_map:
            return right_junction_map[net]
        if net in supply_names:
            return net
        if net in right_external_set:
            return net
        return f"s2_{net}"

    out: List[Device] = []
    for d in left_devs:
        new_pins = {pin_name: _rename_left(net)
                    for pin_name, net in d.pins.items()}
        out.append(Device(
            instance_name=f"s1_{d.instance_name}",
            device_class=d.device_class,
            pdk_variant=d.pdk_variant,
            pins=new_pins,
            sizing=dict(d.sizing),
        ))
    for d in right_devs:
        new_pins = {pin_name: _rename_right(net)
                    for pin_name, net in d.pins.items()}
        out.append(Device(
            instance_name=f"s2_{d.instance_name}",
            device_class=d.device_class,
            pdk_variant=d.pdk_variant,
            pins=new_pins,
            sizing=dict(d.sizing),
        ))
    return out


# ─── Cascade auto-bias (v0.0.89) ────────────────────────────────────


def _inject_cascade_auto_bias(
    devices: List[Device],
    *,
    left_junction_map: Dict[str, str],
    right_junction_map: Dict[str, str],
    composed_outputs: List[str],
    c_ac: float,
    r_bias: float,
) -> List[Device]:
    """Insert AC-coupling + DC-bias devices at each cascade junction.

    The cascade lowering glues each pair of `(left.output, right.input)`
    nets into a single `__junc_<i>` net. For shallow cascades (CS → CS)
    where the external testbench feedback closes a global bias loop
    through both stages, this is sufficient — the junction settles to a
    sensible operating point under the testbench's R_fb feedback.

    Deeper cascades (`two_stage_miller_pmos >> common_source_pmos_input`
    under the differential AC testbench) have no such global loop:
    inn / inp are driven from VCM via VCVS pairs, and the testbench
    leaves the cascade open-loop. The junction node — which is the
    high-impedance output of stage 1's CS final and the high-impedance
    gate of stage 2's CS input — has no DC path to a known voltage. The
    smallest mirror mismatch then drives the node to a supply rail and
    the cascade collapses (gain_dc < 0 dB).

    Auto-bias inserts a local self-bias rig at every junction:

        __junc_<i> ──Cac──→ s2_in_bias_<i> ──Rbias──→ <composed.out>
                                  │
                                  └──→ (rewired into right-stage's
                                        original input devices)

    Every right-stage device whose pins refer to `__junc_<i>` get their
    references to that net redirected to a fresh `s2_in_bias_<i>` net.
    The cap `Cac` (1 µF default) is a short well below 1 Hz so the AC
    signal couples cleanly. The resistor `Rbias` (1 GΩ default) provides
    a high-impedance DC path from the composed cell's output back to
    the bias node — at AC the resistor is effectively open relative to
    the cap, so it doesn't load the small-signal response. The bias
    loop closes locally for the right stage's input transistor without
    disturbing the cascade gain.

    The function is a pure netlist transformation — it appends new
    devices (Cac / Rbias) and rewrites pins on existing devices, but
    never removes or resizes pre-existing transistors. The bias-shifted
    net `s2_in_bias_<i>` is added to the right-stage's local internal
    namespace so it never collides with any prefixed-up netname.

    Args:
        devices: post-`_stitch_devices` flat list (already merged).
        left_junction_map, right_junction_map: same maps the cascade
            wiring built. Used here only for the `__junc_<i>` net
            membership check.
        composed_outputs: external `output_ports` of the composed cell.
            The first one is used as the DC feedback source for every
            junction (typically `'out'` for opamp shapes).
        c_ac, r_bias: device values for the injected cap / resistor.
    """
    if not composed_outputs:
        # Edge case: a cascade whose composed surface has no `signal_out`
        # has nothing to anchor the bias loop on — skip silently.
        return devices

    # Use the FIRST output port as the feedback source. For every
    # current first-cohort interface this is the canonical 'out' net.
    bias_anchor = composed_outputs[0]

    # Collect every junction net name to be rebiased.
    junction_nets: List[str] = sorted(set(left_junction_map.values()))

    if not junction_nets:
        return devices

    # Build the rename map for right-stage devices: each __junc_<i> on a
    # right-stage device gate (or other input pin) → s2_in_bias_<i>.
    # We identify right-stage devices by the `s2_` prefix on their
    # instance_name; that's the post-stitch contract.
    bias_node_for_junction: Dict[str, str] = {
        junc: f"s2_in_bias_{junc.removeprefix('__junc_')}"
        for junc in junction_nets
    }

    rebiased: List[Device] = []
    for d in devices:
        if not d.instance_name.startswith("s2_"):
            rebiased.append(d)
            continue
        # Only rewire pins that refer to a cascade junction net. We do
        # NOT rewrite pins that point at supply nets or at unrelated
        # internal `s2_`-prefixed nets.
        needs_rewrite = any(
            net in bias_node_for_junction for net in d.pins.values()
        )
        if not needs_rewrite:
            rebiased.append(d)
            continue
        new_pins = {
            pin_name: (
                bias_node_for_junction[net]
                if net in bias_node_for_junction
                else net
            )
            for pin_name, net in d.pins.items()
        }
        rebiased.append(Device(
            instance_name=d.instance_name,
            device_class=d.device_class,
            pdk_variant=d.pdk_variant,
            pins=new_pins,
            sizing=dict(d.sizing),
        ))

    # Append one (Cac, Rbias) pair per junction.
    for junc, bias_node in bias_node_for_junction.items():
        suffix = junc.removeprefix("__junc_")
        rebiased.append(Device(
            instance_name=f"Cac_junc_{suffix}",
            device_class="cap",
            pdk_variant=None,
            pins={"p": junc, "n": bias_node},
            sizing={"value": c_ac},
        ))
        rebiased.append(Device(
            instance_name=f"Rbias_junc_{suffix}",
            device_class="res",
            pdk_variant=None,
            pins={"p": bias_anchor, "n": bias_node},
            sizing={"value": r_bias},
        ))

    return rebiased
