"""
PLL transient testbench builder (v0.0.37, last AnalogGym non-amplifier
gap closure).

Sibling of:
  * `build_ac_testbench`             (differential opamps)
  * `build_ldo_testbench`            (LDO line / load regulation)
  * `build_bandgap_testbench`        (voltage references)
  * `build_charge_pump_testbench`    (CP UP/DN current matching)
  * `build_sensor_testbench`         (PTAT temperature / front-end)

A PLL has a fundamentally different testbench shape than every cell
class above:

  1. It has a REFERENCE CLOCK input port (`ref_in`) that must be driven
     by a PULSE waveform, not a DC source or a small-signal VCVS.
  2. It has a CLOCK OUTPUT port (`clk_out`) whose transient waveform is
     the figure of merit — frequency, amplitude, phase.
  3. The native PLL measurements (lock time, phase error, jitter,
     reference spur suppression) are spectral / time-domain figures
     that require long transient runs.

A FULL closed-loop PLL transient — drive ref_in, watch the PFD generate
UP/DN pulses, the CP integrate them onto the loop filter, the VCO track
the loop voltage, the divider feed back to the PFD, loop locks — is
notoriously difficult to converge in pure ngspice batch mode. Reasons:

  * The loop is high-Q with timescales spanning ns (VCO period) and µs
    (lock time). Transient stepsize must resolve the fast edges; the
    run length must cover the slow lock dynamics. Total step count
    blows up.
  * Charge-pump current matching at sky130 / 1.8 V is sensitive; small
    asymmetries (~µA) accumulate over thousands of ref cycles into
    large vctrl drift that the loop must correct.
  * Without an `ind` primitive and varactor model, AnalogGym's LC-tank
    VCO is inexpressible (see analoggym_pll_benchmark.clay header for
    the three independent blockers). Ring VCO substitutes, but ring
    VCO jitter is high and lock dynamics differ from LC.

`build_pll_testbench` therefore ships an OPEN-LOOP / DEGRADED testbench
that produces ONE meaningful transient measurement: the VCO oscillates
under externally-driven control voltage with the reference clock
applied to `ref_in`. This proves:

  * The cell's bias chain converges under sky130 1.8 V supply.
  * The reference clock reaches the cell (ref_in pp ≈ vdd at the
    drive node).
  * The VCO output port (`clk_out`) shows meaningful activity OR is
    a defined DC node (depending on whether the cell internally wires
    ref_in → ... → clk_out, which the current Clay PLL skeleton does
    NOT — it composes only a loop-filter opamp instance).

Wiring:

  * VDD / VSS        — DC supplies on `vdd` / `vss` ports.
  * VREF_CLK         — PULSE on `ref_in`. Default 1 MHz reference,
                        50% duty cycle, 100 ps rise/fall (full-rail
                        digital clock).
  * VCP_BIAS         — DC on `cp_out` port at vdd/2. The current PLL
                        skeleton uses `cp_out` as the loop-filter
                        opamp's non-inverting input, so a DC bias at
                        the midpoint puts the opamp in its linear
                        region. Auto-detected — only emitted if the
                        cell declares the port.
  * VBIAS_LF         — DC on `vbias_lf` port at vdd/2. The opamp's
                        inverting input in the same skeleton. Same
                        auto-detect treatment.
  * (no explicit vctrl drive) — `vctrl` is the loop-filter output and
                        the VCO control input. We do NOT drive it
                        because the cell's internal opamp produces
                        this signal. If the cell does NOT have a
                        `vctrl` port (e.g. a future direct-ref-to-VCO
                        skeleton), the testbench works just as well.
  * CCLKOUT, CVCTRL  — small load capacitors on `clk_out` (and `vctrl`
                        if exposed) so dangling-net warnings from
                        ngspice don't show up. These caps are tiny
                        (1 fF) so they don't perturb any real
                        oscillation if the cell DOES wire ref→clk_out.

Analyses:
  * `OpAnalysis` — operating point.
  * `TranAnalysis` — long transient covering many ref cycles. Default
                     5 µs at 1 ns step (5000 timepoints). Step is
                     coarse-grained: a ring VCO at sky130 oscillates
                     around 100 MHz, so 1 ns / 5 µs gives ~500 VCO
                     cycles to integrate over. Pulse rise/fall (100
                     ps) is still resolved on PULSE source itself —
                     the step subsamples the edges but the source IR
                     handles the on/off transitions explicitly.

Default measures (transient aggregates over the steady-state window
[t_settle, t_end]):

  * `clk_out_pp`      — PP(v(clk_out)). PRIMARY oscillation health
                         signal — large pp means the VCO is swinging
                         meaningfully; near-zero pp means the loop
                         did not light up or the cell does not actually
                         route ref→clk_out (which is the v0.0.37 case
                         for the documented skeleton).
  * `clk_out_avg`     — AVG(v(clk_out)). DC operating point of the
                         output node. For an oscillating clock this
                         is the symmetric DC; for a dangling node it
                         is whatever floating value ngspice settled
                         on (usually 0 with a 1 fF cap to vss).
  * `clk_out_max`     — MAX(v(clk_out)). Peak high of the output.
  * `clk_out_min`     — MIN(v(clk_out)). Peak low of the output.
  * `clk_out_rms`     — RMS(v(clk_out)). Useful for distinguishing
                         square-wave (≈ vdd/2) from sine (≈ 0.7 × vdd)
                         from quiet (≈ 0). Aggregate over the
                         steady-state window.
  * `vctrl_avg`       — AVG(v(vctrl)). Settled control voltage. Only
                         emitted if the cell declares a `vctrl` port.
  * `vctrl_pp`        — PP(v(vctrl)). Ripple on the loop-filter
                         output. Smaller is better.
  * `ref_in_pp`       — PP(v(ref_in)). Sanity check — should be
                         approximately vdd (full-rail digital clock).
                         Confirms the PULSE drive is reaching the
                         port.
  * `fb_clk_pp`       — PP(v(fb_clk)). v0.0.40 loop-closure indicator.
                         Only emitted if the cell declares an `fb_clk`
                         port. A toggling fb_clk proves the divider
                         (and thus the clock path through the VCO) is
                         active. Near-zero fb_clk_pp means the
                         feedback loop has not energized — usually a
                         missing supply rail or a divider clock-edge
                         problem, not a phase-lock issue.

These are all `tran` analysis measures with explicit `over time from
t_settle to t_end` aggregates. Ordering convention: aggregates BEFORE
any `find` so ngspice's sticky-state .meas quirk does not silently
zero out subsequent AVGs (same pattern as the LDO / bandgap / CP /
sensor testbenches).

Why no native lock-time / jitter measure:

  Lock time needs a step in reference frequency followed by waveform
  fitting to a settling exponential. Jitter is the standard deviation
  of zero-crossing intervals over a long capture. Both require a
  post-processing pipeline that goes beyond ngspice's `.meas`
  primitives: a fitted-exponential FROM=/TO=/ rise=/fall= grammar for
  lock time, and per-cycle crossing extraction + statistics for
  jitter. Wiring those at the testbench layer would require either:

  (a) Adding ngspice's `.fourier` analysis output parsing to the
      driver — possible but driver-layer work, not testbench-layer.
  (b) Adding a Python-side post-process step that re-reads the
      transient `.raw` file and computes the figures — requires a
      raw-file reader, also driver-layer.

  v0.0.37 ships the pp / avg / rms / max / min suite as the
  best-available proxy: a healthy oscillation has pp ≈ vdd, avg ≈
  vdd/2, rms in the (0.4 × vdd, 0.6 × vdd) band. Out-of-band values
  reveal "the loop didn't oscillate" or "the cell's clk_out is
  dangling" without requiring spectral analysis.

Why this is the LAST non-amp AnalogGym gap closure:

  * v0.0.32 closed LDO + amplifier baselines.
  * v0.0.35 closed bandgap + sensor (front-end) + charge pump.
  * v0.0.36 closed digital primitives (inv / nand2 / dff) so the
    PFD + divider stop being stubs.
  * v0.0.37 closes PLL via this testbench. After this, every AnalogGym
    benchmark in the v0.0.32+ supported set has a runnable e2e test
    (some still skipped for inductor / varactor / OpenAccess-only
    netlist reasons).
"""

from __future__ import annotations

from ..ir import (
    Cell, Device, Measure, OpAnalysis, Source, Testbench, TranAnalysis,
)


# ─── Defaults ──────────────────────────────────────────────────────
# Picked to match the sky130 ring-VCO behavior measured in
# tmp_clay_rundir/pll_vco_test_* during testbench development. At
# vctrl = 1.0 V the ring VCO oscillates with pp ≈ 1.4 V and average
# period in the 5-10 ns range.

DEFAULT_VDD_NOM = 1.8           # sky130 nominal supply
DEFAULT_F_REF = 1e6             # 1 MHz reference clock
DEFAULT_PFD_HIGH_FRAC = 1.0     # PFD high level = vdd (full rail)
DEFAULT_REF_RISE = 1e-10        # 100 ps rise/fall on ref_in
DEFAULT_REF_DELAY = 0.0         # no leading idle window
DEFAULT_DC_BIAS_FRAC = 0.5      # cp_out / vbias_lf default to vdd/2
DEFAULT_TRAN_STEP = 1e-9        # 1 ns transient step
DEFAULT_TRAN_END = 5e-6         # 5 µs total — ~5 ref cycles at 1 MHz,
                                # ~500 VCO cycles at 100 MHz
DEFAULT_T_SETTLE = 1e-6         # exclude the first 1 µs from steady-
                                # state aggregates (bias chain ramp-up)
DEFAULT_LOAD_CAP = 1e-15        # 1 fF on clk_out / vctrl if dangling


# Port names the heuristic looks for to classify a cell as PLL-shaped.
# A PLL cell MUST have both `ref_in` (reference clock input) AND
# `clk_out` (VCO output) ports. Other PLL-related ports (`cp_out`,
# `vbias_lf`, `vctrl`, `fb_clk`) are optional — auto-detected and
# wired only if present.
_REQUIRED_PLL_PORTS = ("ref_in", "clk_out")

# Optional PLL bias / internal-signal ports that the testbench drives
# at vdd/2 when present. Order matters for source emission (vctrl is
# the loop-filter output, so we drive it ONLY if no other path biases
# it — see _emit_bias_sources).
_OPTIONAL_BIAS_PORTS = ("cp_out", "vbias_lf")

# Optional output-like ports that need a small load cap if they would
# otherwise dangle (no device touches them inside the cell). The
# testbench scans the cell for dangling ports — see _emit_load_caps.
_OUTPUT_LIKE_PORTS = ("clk_out", "vctrl", "fb_clk")


def find_pll_ports(cell: Cell) -> dict[str, str]:
    """Return the {logical_name → port_name} map for the cell's PLL
    surface, or an empty dict if the cell is not PLL-shaped.

    A PLL-shaped cell MUST declare both `ref_in` and `clk_out` ports.
    Optional logical keys (only present when the cell has the matching
    port) are: `cp_out`, `vbias_lf`, `vctrl`, `fb_clk`.

    The returned dict carries the actual port names so the testbench
    builder can wire its sources / load caps to whatever the cell
    author used. Returns `{}` if the cell does not declare both
    `ref_in` and `clk_out` — that's the heuristic the CLI uses to
    fall back to the AC testbench / other rigs.
    """
    port_set = set(cell.ports)
    if not all(p in port_set for p in _REQUIRED_PLL_PORTS):
        return {}
    out: dict[str, str] = {"ref_in": "ref_in", "clk_out": "clk_out"}
    for opt in (*_OPTIONAL_BIAS_PORTS, "vctrl", "fb_clk"):
        if opt in port_set:
            out[opt] = opt
    return out


def _net_touched_by_devices(cell: Cell, net: str) -> bool:
    """Return True iff any device on the cell drives or senses `net`.

    A net that no device pin references is "dangling" from ngspice's
    POV — the simulator may converge to it floating but the result is
    not physically meaningful. The testbench adds a small load cap
    to vss for any dangling output-like port so the netlist is
    well-formed.
    """
    for d in cell.devices:
        if net in d.pins.values():
            return True
    return False


def build_pll_testbench(
    cell: Cell,
    *,
    vdd_nom: float | None = None,
    f_ref: float = DEFAULT_F_REF,
    pfd_high_frac: float = DEFAULT_PFD_HIGH_FRAC,
    ref_rise: float = DEFAULT_REF_RISE,
    ref_delay: float = DEFAULT_REF_DELAY,
    dc_bias_frac: float = DEFAULT_DC_BIAS_FRAC,
    tran_step: float = DEFAULT_TRAN_STEP,
    tran_end: float = DEFAULT_TRAN_END,
    t_settle: float = DEFAULT_T_SETTLE,
    load_cap: float = DEFAULT_LOAD_CAP,
) -> Testbench:
    """Build the default PLL transient testbench.

    Wiring:
      * VDD       — voltage_dc, `vdd` ↔ 0, dc=vdd_nom.
      * VSS       — voltage_dc, `vss` ↔ 0, dc=0.
      * VREF_CLK  — voltage_pulse, `ref_in` ↔ 0. PULSE clock at
                    `f_ref` Hz, 50% duty cycle, rail-to-rail. The
                    PULSE period is `1 / f_ref`, pulse width is
                    `period / 2`, rise / fall is `ref_rise`.
      * VCP_BIAS  — voltage_dc, `cp_out` ↔ 0, dc=vdd_nom*dc_bias_frac.
                    Auto-detected — only emitted if the cell has a
                    `cp_out` port. Drives the loop-filter opamp's
                    non-inverting input (in the current PLL skeleton).
      * VBIAS_LF  — voltage_dc, `vbias_lf` ↔ 0, dc=vdd_nom*dc_bias_frac.
                    Auto-detected — only emitted if the cell has a
                    `vbias_lf` port. The opamp's virtual-ground
                    reference.

    Extra devices (load caps for dangling output ports):
      * CCLKOUT   — cap on `clk_out` to vss, value=load_cap (default
                    1 fF). Only added if no device inside the cell
                    touches `clk_out` — i.e. the cell doesn't actually
                    wire ref_in → ... → clk_out internally. For a
                    cell that DOES wire clk_out (e.g. a future full-
                    PLL composition), the small cap is omitted to
                    avoid perturbing the output stage.
      * CVCTRL    — cap on `vctrl` to vss, same auto-detect logic.
                    The current PLL skeleton DOES wire vctrl (it's
                    the loop-filter opamp output) so this cap is
                    typically not emitted.

    Analyses:
      * OpAnalysis — bias-chain operating point.
      * TranAnalysis(step=tran_step, t_end=tran_end) — covers many
        reference cycles plus VCO settling.

    Default measures (aggregates over [t_settle, tran_end]):
      * clk_out_pp, clk_out_avg, clk_out_max, clk_out_min, clk_out_rms
      * vctrl_pp, vctrl_avg (only if the cell has a `vctrl` port)
      * ref_in_pp (sanity check — confirms PULSE drive is reaching
        the port)

    Args:
        cell: the PLL cell IR. MUST declare `vdd`, `vss`, `ref_in`,
            `clk_out` ports. Optional ports `cp_out`, `vbias_lf`,
            `vctrl`, `fb_clk` are auto-detected.
        vdd_nom: nominal supply in volts.
        f_ref: reference clock frequency in Hz.
        pfd_high_frac: reference clock high level as a fraction of
            vdd_nom (default 1.0 = full rail).
        ref_rise: rise/fall time of the reference clock in seconds.
        ref_delay: initial delay before the first reference edge.
        dc_bias_frac: DC bias level for `cp_out` / `vbias_lf` ports
            as a fraction of vdd_nom (default 0.5 = midpoint).
        tran_step: transient time step in seconds.
        tran_end: total transient window in seconds.
        t_settle: time after which steady-state aggregates are
            computed. Excludes the bias-chain ramp-up window.
        load_cap: capacitance value for the dangling-net load caps.
            Default 1 fF — small enough to not perturb real signals.

    Returns:
        A Testbench with VDD/VSS supplies, VREF_CLK PULSE source,
        optional DC bias sources (auto-detected), optional load caps
        (auto-detected), OP + TRAN analyses, and the default measure
        suite.

    Raises:
        ValueError: if the cell does not declare the required ports
            (`vdd`, `vss`, `ref_in`, `clk_out`).
    """
    # ─── Required-port validation ──────────────────────────────
    if "vdd" not in cell.ports:
        raise ValueError(
            f"PLL testbench: cell {cell.name!r} has no `vdd` port "
            f"(ports={cell.ports}). The testbench wires the cell's "
            f"vdd to a DC supply — a PLL cell must declare a vdd port."
        )
    if "vss" not in cell.ports:
        raise ValueError(
            f"PLL testbench: cell {cell.name!r} has no `vss` port "
            f"(ports={cell.ports}). The testbench wires the cell's "
            f"vss to ground."
        )
    pll_map = find_pll_ports(cell)
    if not pll_map:
        raise ValueError(
            f"PLL testbench: cell {cell.name!r} has no PLL surface "
            f"ports (ports={cell.ports}). The PLL testbench requires "
            f"both `ref_in` (reference clock input) and `clk_out` "
            f"(VCO output) ports. For a non-PLL cell, use a different "
            f"testbench (build_ac_testbench for opamps, "
            f"build_ldo_testbench for LDOs, build_charge_pump_testbench "
            f"for CPs, build_bandgap_testbench for voltage references, "
            f"build_sensor_testbench for sensors)."
        )

    # ─── Source list ────────────────────────────────────────────
    if vdd_nom is None:
        from ._supply import resolve_supply
        vdd_nom = resolve_supply(cell, default=DEFAULT_VDD_NOM)
    period = 1.0 / f_ref
    pulse_width = period / 2.0
    ref_high = vdd_nom * pfd_high_frac
    dc_bias = vdd_nom * dc_bias_frac

    sources: list[Source] = [
        Source("voltage_dc", "VDD",
               pins={"p": "vdd", "n": "0"},
               params={"dc": vdd_nom}),
        Source("voltage_dc", "VSS",
               pins={"p": "vss", "n": "0"},
               params={"dc": 0.0}),
        # Reference clock PULSE. Full-rail digital clock, 50% duty.
        Source("voltage_pulse", "VREF_CLK",
               pins={"p": pll_map["ref_in"], "n": "0"},
               params={
                   "v_low": 0.0,
                   "v_high": ref_high,
                   "td": ref_delay,
                   "t_rise": ref_rise,
                   "t_fall": ref_rise,
                   "pulse_width": pulse_width,
                   "period": period,
               }),
    ]

    # Optional bias DC sources on internal nets exposed as outer ports.
    # Auto-detect: drive the port AS A DC bias ONLY when no device inside
    # the cell touches it. Two use cases:
    #
    #   (a) v0.0.32 skeleton — cp_out / vbias_lf are outer ports purely
    #       to satisfy the composition compiler's "every port_map RHS
    #       must be an outer port" rule. They have no internal driver,
    #       so the testbench supplies a DC bias.
    #
    #   (b) v0.0.39 PllTop closure — the same outer port IS driven
    #       internally (the charge pump pulls current into cp_out, and
    #       the loop filter's R1+C1+C2 actually form the loop filter).
    #       Driving it from the testbench would short out the closed
    #       loop, so we skip the source. Same heuristic as the load-cap
    #       path below.
    if "cp_out" in pll_map and not _net_touched_by_devices(cell, pll_map["cp_out"]):
        sources.append(Source(
            kind="voltage_dc",
            instance_name="VCP_BIAS",
            pins={"p": pll_map["cp_out"], "n": "0"},
            params={"dc": dc_bias},
        ))
    if "vbias_lf" in pll_map and not _net_touched_by_devices(cell, pll_map["vbias_lf"]):
        sources.append(Source(
            kind="voltage_dc",
            instance_name="VBIAS_LF",
            pins={"p": pll_map["vbias_lf"], "n": "0"},
            params={"dc": dc_bias},
        ))

    # ─── Load caps for dangling output-like ports ──────────────
    # Output ports that no device inside the cell touches would
    # otherwise dangle in the netlist. Add a small cap (1 fF default)
    # to vss so ngspice sees a defined node. For a cell that DOES wire
    # the output internally (e.g. clk_out connected to a VCO inside),
    # we skip the cap so the small parasitic doesn't perturb the
    # output stage.
    extra_devices: list[Device] = []
    for port_name in _OUTPUT_LIKE_PORTS:
        if port_name in cell.ports and not _net_touched_by_devices(cell, port_name):
            extra_devices.append(Device(
                instance_name=f"CLOAD_{port_name.upper()}",
                device_class="cap",
                pdk_variant=None,
                pins={"p": port_name, "n": "vss"},
                sizing={"value": load_cap},
            ))

    # ─── Analyses ──────────────────────────────────────────────
    analyses: list = [
        OpAnalysis(),
        TranAnalysis(step=tran_step, t_end=tran_end),
    ]

    # ─── Measure suite ─────────────────────────────────────────
    measures: list[Measure] = _pll_measures(
        pll_map=pll_map,
        cell=cell,
        t_settle=t_settle,
        t_end=tran_end,
    )

    return Testbench(
        cell=cell,
        sources=sources,
        extra_devices=extra_devices,
        analyses=analyses,
        measures=measures,
        pdk_corner="tt",
        supply_volt=vdd_nom,
    )


def _pll_measures(
    *,
    pll_map: dict[str, str],
    cell: Cell,
    t_settle: float,
    t_end: float,
) -> list[Measure]:
    """Build the .meas list for a PLL testbench.

    Measure set (all aggregates over the steady-state window
    [t_settle, t_end]):

      * clk_out_pp   — PP(v(clk_out)). Primary oscillation indicator.
      * clk_out_max  — MAX(v(clk_out)). Output peak high.
      * clk_out_min  — MIN(v(clk_out)). Output peak low.
      * clk_out_avg  — AVG(v(clk_out)). Output DC operating point.
      * clk_out_rms  — RMS(v(clk_out)). Signal energy proxy.
      * vctrl_pp     — PP(v(vctrl)). Only if cell has vctrl port.
      * vctrl_avg    — AVG(v(vctrl)). Only if cell has vctrl port.
      * vctrl_max    — MAX(v(vctrl)). Only if cell has vctrl port.
      * vctrl_min    — MIN(v(vctrl)). Only if cell has vctrl port.
      * ref_in_pp    — PP(v(ref_in)). Sanity check on PULSE drive.

    All measures use `analysis_ref='tran'` since the testbench runs a
    TRAN analysis.

    Order convention: MAX → MIN → AVG → RMS → PP, with all aggregates
    BEFORE any find-style measures (matches the LDO / bandgap / CP /
    sensor testbench order to dodge ngspice's sticky-state .meas
    quirk).
    """
    out: list[Measure] = []

    # clk_out is always present (required by find_pll_ports).
    clk = pll_map["clk_out"]
    out.append(Measure(
        name="clk_out_max",
        analysis_ref="tran",
        expr=f"MAX(v({clk})) over time from {t_settle:g} to {t_end:g}",
    ))
    out.append(Measure(
        name="clk_out_min",
        analysis_ref="tran",
        expr=f"MIN(v({clk})) over time from {t_settle:g} to {t_end:g}",
    ))
    out.append(Measure(
        name="clk_out_avg",
        analysis_ref="tran",
        expr=f"AVG(v({clk})) over time from {t_settle:g} to {t_end:g}",
    ))
    out.append(Measure(
        name="clk_out_rms",
        analysis_ref="tran",
        expr=f"RMS(v({clk})) over time from {t_settle:g} to {t_end:g}",
    ))
    out.append(Measure(
        name="clk_out_pp",
        analysis_ref="tran",
        expr=f"PP(v({clk})) over time from {t_settle:g} to {t_end:g}",
    ))

    # vctrl is optional — only emit measures if the port exists.
    if "vctrl" in pll_map:
        vctrl = pll_map["vctrl"]
        out.append(Measure(
            name="vctrl_max",
            analysis_ref="tran",
            expr=f"MAX(v({vctrl})) over time from {t_settle:g} to {t_end:g}",
        ))
        out.append(Measure(
            name="vctrl_min",
            analysis_ref="tran",
            expr=f"MIN(v({vctrl})) over time from {t_settle:g} to {t_end:g}",
        ))
        out.append(Measure(
            name="vctrl_avg",
            analysis_ref="tran",
            expr=f"AVG(v({vctrl})) over time from {t_settle:g} to {t_end:g}",
        ))
        out.append(Measure(
            name="vctrl_pp",
            analysis_ref="tran",
            expr=f"PP(v({vctrl})) over time from {t_settle:g} to {t_end:g}",
        ))

    # ref_in PP is the sanity-check measure (verifies PULSE drive).
    ref = pll_map["ref_in"]
    out.append(Measure(
        name="ref_in_pp",
        analysis_ref="tran",
        expr=f"PP(v({ref})) over time from {t_settle:g} to {t_end:g}",
    ))

    # v0.0.40: fb_clk pp is the divider-output activity indicator. A
    # well-formed PLL has the divider running on clk_out and producing
    # a fb_clk that toggles at clk_out / N. Near-zero fb_clk_pp means
    # the divider's DFF isn't sampling — usually a clock-path or
    # supply-rail problem, not a lock issue. Only emit the measure if
    # the cell actually declares an `fb_clk` port (PLL skeletons that
    # hide the feedback internally don't need it).
    if "fb_clk" in pll_map:
        fb_clk = pll_map["fb_clk"]
        out.append(Measure(
            name="fb_clk_pp",
            analysis_ref="tran",
            expr=f"PP(v({fb_clk})) over time from {t_settle:g} to {t_end:g}",
        ))

    return out
