"""
Sample-and-Hold combinator — track-and-hold network (v0.0.99 / R75 C + v0.1.12 split).

`sample_and_hold(stage, ...)` wraps a stage's analog output with a
track-and-hold network: an NMOS pass switch whose gate is driven by an
external `clkS` signal, plus a hold capacitor from the sampled-output
node to `vss`. Auto-sizing path uses a single-step Newton-style
adjustment from droop-rate + (optional) bandwidth targets.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..ir import Cell, Device, Spec
from ..stage_interface import ComposedStage
from ._common import StageRef, TemplateBuilder, _is_finite, _resolve


# Auto-sizing seeds for the sample-and-hold network. The default hold
# capacitor of 1 pF is a representative on-chip S/H value: large enough
# to dominate over the kT/C noise floor on sky130 1.8 V at room temp
# while small enough to keep the sample-mode RC bandwidth (with a 1 µm
# pass switch) into the MHz range. The default switch W of 1 µm sits in
# moderate strong-inversion when its gate sees a full-rail clkS swing,
# yielding R_on in the kΩ ballpark.
_SH_SEED_HOLD_CAP: float = 1.0e-12          # 1 pF hold capacitor
_SH_SEED_SWITCH_W: float = 1.0e-6           # 1 µm pass switch width
_SH_PASS_L: float = 0.18e-6                 # 0.18 µm channel length
# Empirical estimate of the OFF-state leak current through an NMOS pass
# switch at sky130 1.8 V — drain–source subthreshold + gate–source / gate–
# drain overlap leak. Real silicon depends on Vds and temp; this is a
# room-temperature mid-range proxy used for the single-step Newton
# adjustment. Units: amperes.
_SH_OFF_LEAK_CURRENT: float = 1.0e-12       # 1 pA OFF leak (proxy)
# Hard cap on the single-step Newton adjustment magnitude so the search
# cannot run away when the requested droop_rate is unrealistically
# tight or loose.
_SH_ADJUST_MAX_FACTOR: float = 1000.0
# When sample-mode bandwidth is also tracked, this scaling constant
# turns "Δlog10(BW deficit)" into a multiplicative widen on the switch
# W. One decade of bandwidth costs one decade of W (1st-order).
_SH_BW_W_GAIN_PER_DECADE: float = 1.0


def sample_and_hold(
    stage_arg: StageRef,
    *,
    hold_cap: Optional[float] = None,
    sample_switch: Optional[float] = None,
    sample_clk: Optional[str] = None,
    droop_rate_target: Optional[float] = None,
    sample_bandwidth_target: Optional[float] = None,
) -> ComposedStage:
    """`sample_and_hold(stage, ...)` — track-and-hold combinator.

    Wraps a stage's analog output with a track-and-hold network: an
    NMOS pass switch whose gate is driven by an external `clkS` signal,
    plus a hold capacitor from the sampled-output node to `vss`. When
    `clkS` is HIGH the switch is on (track mode — output follows the
    inner stage); when `clkS` is LOW the switch is off (hold mode —
    output droops at I_leak / C_hold).

    Topology that is lowered (sky130 NMOS pass-switch convention):

      stage.output ──┐
                     │
                    [d]
        Msample (NMOS, gate=clkS, body=vss)
                    [s]
                     ├── hold_cap ── vss
                     │
                     out (sampled output)

    Two equivalent entry surfaces:

      * **Explicit sizing**:
          - `hold_cap=Ch`         — hold capacitor value (Farads).
          - `sample_switch=Wsw`   — switch channel width (m).
        At least one must be supplied.

      * **Auto-sizing**:
          - `droop_rate_target=Rd` (V/s) — derive Ch (and optionally
                                            widen Wsw) such that the
                                            measured OFF-state droop
                                            rate stays ≤ Rd. Sized via
                                            a single-step Newton-style
                                            adjustment from
                                            (Ch=1 pF, W=1 µm) seeds.

    Optional:
      * `sample_clk` (str) — name of the gate driver net. Defaults to
        `'clkS'`. The caller is responsible for forcing this net at the
        testbench level (e.g. a pulse source between vdd and vss).
      * `sample_bandwidth_target` (Hz) — when supplied together with
        `droop_rate_target`, the auto-sizer ALSO widens the switch W so
        that `1/(2π · R_on · Ch)` ≥ `sample_bandwidth_target`. A 1st-
        order proxy: each decade of bandwidth deficit scales W by one
        decade (capped by `_SH_ADJUST_MAX_FACTOR`).

    Type-check rules:
      * Exactly one of {explicit hold_cap/sample_switch,
        droop_rate_target} must be supplied (auto vs explicit is
        mutually exclusive).
      * `hold_cap` / `sample_switch` must be positive finite reals.
      * `droop_rate_target` must be a positive finite real (V/s).
      * `sample_bandwidth_target`, when given, must be positive finite
        and is only honoured when `droop_rate_target` is also supplied
        (otherwise raises — bandwidth adjustment depends on the derived
        cap value).

    Meta fields populated:
      * `meta['hold_cap']`              — explicit cap value if given.
      * `meta['sample_switch']`         — explicit switch W if given.
      * `meta['sample_clk']`            — gate driver net name (default
                                            `'clkS'`).
      * `meta['droop_rate_target']`     — auto target if given.
      * `meta['sample_bandwidth_target']` — optional BW target.
      * `meta['derived_cap']`           — populated after auto-sizing.
      * `meta['derived_W']`             — populated after auto-sizing.
      * `meta['sh_auto_trace']`         — list of trace records
                                            describing the Newton step
                                            (empty when degraded).

    Bias / clocking convention: the gate driver net (`sample_clk`,
    default `'clkS'`) is treated as an EXTERNAL signal the testbench
    must provide. The combinator does NOT extend the wrapped cell's
    port list (so existing testbench builders keep working); instead
    `clkS` flows in as a kwarg-renamed signal and the testbench rig
    forces it explicitly. Real ADC/DAC tests would route a pulse
    source (0 → vdd, with a configurable duty cycle) into this net.
    """
    s = _resolve(stage_arg)

    has_explicit = (
        hold_cap is not None or sample_switch is not None
    )
    has_auto = droop_rate_target is not None
    # Surface the bandwidth-without-droop misuse first; otherwise the
    # generic "neither explicit nor auto" branch would shadow the more
    # specific BW pairing constraint when the caller supplies ONLY
    # `sample_bandwidth_target=`.
    if sample_bandwidth_target is not None and droop_rate_target is None:
        raise ValueError(
            "sample_and_hold(sample_bandwidth_target=...) requires "
            "droop_rate_target= as well (the BW adjustment scales "
            "the auto-derived cap and switch W together)."
        )
    if not has_explicit and not has_auto:
        raise ValueError(
            "sample_and_hold() requires either explicit "
            "(hold_cap=...,/sample_switch=...) OR "
            "auto-sizing (droop_rate_target=...)"
        )
    if has_explicit and has_auto:
        raise ValueError(
            "sample_and_hold(): cannot combine explicit hold_cap=/"
            "sample_switch= with droop_rate_target=. Choose one "
            "entry surface."
        )

    if hold_cap is not None:
        if (not isinstance(hold_cap, (int, float))
                or float(hold_cap) <= 0
                or not _is_finite(float(hold_cap))):
            raise ValueError(
                f"sample_and_hold(hold_cap=...) must be a positive "
                f"finite capacitance in Farads, got {hold_cap!r}"
            )
    if sample_switch is not None:
        if (not isinstance(sample_switch, (int, float))
                or float(sample_switch) <= 0
                or not _is_finite(float(sample_switch))):
            raise ValueError(
                f"sample_and_hold(sample_switch=...) must be a positive "
                f"finite width in metres, got {sample_switch!r}"
            )
    if droop_rate_target is not None:
        if (not isinstance(droop_rate_target, (int, float))
                or not _is_finite(float(droop_rate_target))
                or float(droop_rate_target) <= 0):
            raise ValueError(
                f"sample_and_hold(droop_rate_target=...) must be a "
                f"positive finite real in V/s, got {droop_rate_target!r}"
            )
    if sample_bandwidth_target is not None:
        if (not isinstance(sample_bandwidth_target, (int, float))
                or not _is_finite(float(sample_bandwidth_target))
                or float(sample_bandwidth_target) <= 0):
            raise ValueError(
                f"sample_and_hold(sample_bandwidth_target=...) must be "
                f"a positive finite frequency in Hz, got "
                f"{sample_bandwidth_target!r}"
            )
    if sample_clk is not None:
        if not isinstance(sample_clk, str) or not sample_clk:
            raise ValueError(
                f"sample_and_hold(sample_clk=...) must be a non-empty "
                f"string net name, got {sample_clk!r}"
            )

    meta: Dict[str, Any] = {}
    if hold_cap is not None:
        meta["hold_cap"] = float(hold_cap)
    if sample_switch is not None:
        meta["sample_switch"] = float(sample_switch)
    meta["sample_clk"] = str(sample_clk) if sample_clk is not None else "clkS"
    if droop_rate_target is not None:
        meta["droop_rate_target"] = float(droop_rate_target)
    if sample_bandwidth_target is not None:
        meta["sample_bandwidth_target"] = float(sample_bandwidth_target)

    return ComposedStage(
        kind="sample_and_hold",
        interface=s.interface,
        parts=(s,),
        meta=meta,
    )


def _sample_hold_newton_adjust(
    *,
    seed_cap: float,
    seed_W: float,
    droop_rate_target: float,
    measured_leak: float,
    sample_bandwidth_target: Optional[float],
) -> Tuple[float, float]:
    """Compute (cap, W) from a droop-rate / bandwidth deficit.

    Droop-rate path:
      measured_droop = measured_leak / seed_cap
      factor_cap     = measured_droop / droop_rate_target
    Scaling Ch by `factor_cap` brings the predicted droop in line with
    the target. The factor is clipped to `[1/_SH_ADJUST_MAX_FACTOR,
    _SH_ADJUST_MAX_FACTOR]` so a numerical artefact (leak ≈ 0) cannot
    drive the result to absurd values.

    Sample-mode bandwidth path (optional):
      bw_target / bw_seed  → factor_W via
                              factor_W = 10 ** (decades_of_deficit *
                                                _SH_BW_W_GAIN_PER_DECADE)
    Wider W → lower R_on → higher 3-dB bandwidth. We DO scale Ch and W
    together (not just one) because a larger cap tightens the bandwidth
    in the same proportion the auto-droop step grew it.
    """
    if seed_cap <= 0 or droop_rate_target <= 0:
        return seed_cap, seed_W
    measured_droop = float(measured_leak) / float(seed_cap)
    if measured_droop <= 0:
        # Nothing to adjust — leak floor is below numerical resolution.
        return seed_cap, seed_W
    raw_factor_cap = measured_droop / float(droop_rate_target)
    factor_cap = min(
        _SH_ADJUST_MAX_FACTOR,
        max(1.0 / _SH_ADJUST_MAX_FACTOR, raw_factor_cap),
    )
    cap_new = float(seed_cap) * factor_cap
    W_new = float(seed_W)
    if sample_bandwidth_target is not None:
        # Seed bandwidth: 1 / (2π · R_on · Ch). R_on for a 1 µm sky130
        # NMOS pass switch with gate at vdd is ≈ 5 kΩ (mid-strong-inv).
        # We model R_on ∝ 1/W so doubling W halves R_on and doubles BW.
        R_on_seed = 5.0e3 * (_SH_SEED_SWITCH_W / float(seed_W))
        bw_seed = 1.0 / (2.0 * math.pi * R_on_seed * cap_new)
        if bw_seed > 0 and sample_bandwidth_target > 0:
            decades = math.log10(
                float(sample_bandwidth_target) / float(bw_seed)
            )
            if decades > 0:
                raw_factor_W = 10.0 ** (
                    decades * _SH_BW_W_GAIN_PER_DECADE
                )
                factor_W = min(
                    _SH_ADJUST_MAX_FACTOR,
                    max(1.0 / _SH_ADJUST_MAX_FACTOR, raw_factor_W),
                )
                W_new = float(seed_W) * factor_W
    return cap_new, W_new


def _lower_sample_and_hold(
    inner: ComposedStage,
    meta: Dict[str, Any],
) -> TemplateBuilder:
    """Build the sample_and_hold builder.

    Inserts an NMOS pass switch (gate=clkS) between the inner stage's
    output node and a NEW sampled-output node, plus a hold cap from
    that sampled node to `vss`. The composed cell's external port
    surface is preserved (the inner stage's `out` port becomes the
    sampled output node — i.e. we rename the inner stage's emitted
    `out` net to a junction and re-attach `out` to the cap top).

    Wiring (assuming inner stage has an `out` port):
      * `__sh_track_out` is the inner stage's untracked output (the
        device-level net that used to be `out`).
      * `Msample` — NMOS pass switch with d=__sh_track_out, g=clkS,
                    s=out, b=vss. Sized by `sample_switch`.
      * `Chold`   — cap from `out` to `vss`. Sized by `hold_cap`.

    After insertion, every device whose pin map references `out` (the
    pre-existing inner stage's output net) and is NOT one of {Msample,
    Chold} is rewired to the new `__sh_track_out` net. The external
    cell port surface (`in`, `out`, `vdd`, `vss`, ...) is preserved
    byte-for-byte so the testbench rig works unchanged.

    Auto-sizing path: if `meta['droop_rate_target']` is set without an
    explicit `meta['hold_cap']`, we apply a single-step Newton-style
    adjustment using the proxy leak current `_SH_OFF_LEAK_CURRENT` so
    the predicted droop rate (`I_leak / Ch`) lands at the target.
    Bandwidth widening is honoured when
    `meta['sample_bandwidth_target']` is also set.

    Args:
        inner: the wrapped ComposedStage.
        meta: see `sample_and_hold` for the meta surface contract.
    """
    from ._lower import lower_to_template
    inner_builder = lower_to_template(inner)
    droop_target = meta.get("droop_rate_target")
    bw_target = meta.get("sample_bandwidth_target")

    def _builder(cell_name: str, spec: Spec, **kwargs: Any) -> Cell:
        # Pull aside any caller overrides for the switch gate net. By
        # default the switch's gate sits on the meta-configured net (or
        # 'clkS' if unset). Sophisticated callers can supply a renamed
        # net at builder time via `sample_clk_net=...` for testbench
        # plumbing flexibility.
        clk_net = str(
            kwargs.pop("sample_clk_net", meta.get("sample_clk", "clkS"))
        )

        # Auto-sizing path: probe the analytic leak proxy and apply
        # the single Newton step. Falls back to seed values when only
        # droop_target is given.
        trace: List[Dict[str, Any]] = []
        if droop_target is not None and "hold_cap" not in meta:
            ch, w = _sample_hold_newton_adjust(
                seed_cap=_SH_SEED_HOLD_CAP,
                seed_W=_SH_SEED_SWITCH_W,
                droop_rate_target=float(droop_target),
                measured_leak=_SH_OFF_LEAK_CURRENT,
                sample_bandwidth_target=(
                    float(bw_target) if bw_target is not None else None
                ),
            )
            meta["derived_cap"] = ch
            meta["derived_W"] = w
            trace.append({
                "iter": 0,
                "Ch": _SH_SEED_HOLD_CAP,
                "W": _SH_SEED_SWITCH_W,
                "leak_A": _SH_OFF_LEAK_CURRENT,
                "method": "seed_probe",
                "status": "measured_proxy",
            })
            trace.append({
                "iter": 1,
                "Ch": ch,
                "W": w,
                "leak_A": _SH_OFF_LEAK_CURRENT,
                "method": "newton_step",
                "droop_target_V_per_s": float(droop_target),
                "status": "adjusted",
            })
            meta["sh_auto_trace"] = trace
            sh_cap = ch
            sw_W = w
        else:
            sh_cap = float(meta.get("hold_cap", _SH_SEED_HOLD_CAP))
            sw_W = float(meta.get("sample_switch", _SH_SEED_SWITCH_W))
            meta.setdefault("sh_auto_trace", [])

        inner_cell = inner_builder(cell_name, spec, **kwargs)

        # Defensive: if the inner cell has no `out` port we cannot wire
        # the sample switch. Degrade to a no-op (return the inner cell
        # untouched) so unusual topologies still compile.
        if "out" not in inner_cell.ports:
            return inner_cell

        track_net = "__sh_track_out"
        # Rewire every device whose pin map names `out` (the inner
        # stage's emitted output net) to the new `track_net`, except
        # for any device we are about to add (Msample / Chold). The
        # inner stage's devices are treated as immutable; we mutate
        # copies.
        rewired: List[Device] = []
        for d in inner_cell.devices:
            if "out" not in d.pins.values():
                rewired.append(d)
                continue
            new_pins = {
                pin_name: (track_net if net == "out" else net)
                for pin_name, net in d.pins.items()
            }
            rewired.append(Device(
                instance_name=d.instance_name,
                device_class=d.device_class,
                pdk_variant=d.pdk_variant,
                pins=new_pins,
                sizing=dict(d.sizing),
            ))

        # Insert Msample: NMOS pass switch between track_net (drain)
        # and out (source). Gate driven by clkS, body on vss. With clkS
        # at vdd → strong inversion → low R_on; with clkS at vss → off.
        rewired.append(Device(
            instance_name="Msample_sh",
            device_class="nmos",
            pdk_variant=None,
            pins={
                "d": track_net,
                "g": clk_net,
                "s": "out",
                "b": "vss",
            },
            sizing={
                "W": float(sw_W),
                "L": float(_SH_PASS_L),
                "M": 1.0,
            },
        ))

        # Insert Chold: hold capacitor between out and vss. Top plate
        # on the sampled output, bottom plate on the ground rail.
        rewired.append(Device(
            instance_name="Chold_sh",
            device_class="cap",
            pdk_variant=None,
            pins={"p": "out", "n": "vss"},
            sizing={"value": float(sh_cap)},
        ))

        return Cell(
            name=inner_cell.name,
            ports=list(inner_cell.ports),
            spec=spec,
            devices=rewired,
            pdk_id=inner_cell.pdk_id,
        )

    _builder.__name__ = f"sample_and_hold__{inner.template_name}"
    return _builder
