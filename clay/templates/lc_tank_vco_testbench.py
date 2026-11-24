"""
LC-tank VCO transient testbench builder (v0.0.57).

Companion to the v0.0.56 `build_lc_tank_vco` cell builder. The base
cell ships a structurally-correct cross-coupled NMOS LC tank, but a
perfectly differential topology has a symmetric DC operating point —
ngspice batch mode converges to that equilibrium and transient stays
flat. The v0.0.56 docstring called this out and deferred actual
oscillation verification to v0.0.57.

This testbench breaks the symmetry with a current-pulse kick on
`clk_out` so transient observes real LC oscillation. The kick is
deterministic, lives on the testbench side (no cell modification), and
is short enough (≤ a few ns) that it neither sets the steady-state
amplitude nor distorts the eventual frequency. Once the cross-coupled
pair's negative resistance overcomes tank loss, the kick is irrelevant
— the loop sustains itself.

Wiring
------

  * VDD      — voltage_dc on `vdd` ↔ 0, dc=`vdd_nom` (auto-resolved
               from the PDK adapter if not supplied).
  * VSS      — voltage_dc on `vss` ↔ 0, dc=0.
  * VCTRL    — voltage_dc on `vctrl` ↔ 0, dc=`vctrl_dc`. Default
               `vctrl_dc` ≈ vdd_nom (sky130: 1.7 V). High vctrl keeps
               the tail NMOS deep in saturation so I_tail and hence
               gm of the cross-coupled pair are large — necessary for
               startup margin.
  * IKICK    — current_pulse, ground ↔ `clk_out`. PULSE of `kick_amp`
               amperes for `kick_width` seconds starting at `kick_delay`.
               Default 50 mA × 2 ns at t=1 ns. The kick pulls a small
               net charge out of clk_out, which the cross-coupled pair
               then amplifies into full-rail oscillation.

Analyses
--------

  * OpAnalysis — bias chain check.
  * TranAnalysis — long transient covering startup + settled steady
    state. Default 2 µs at 25 ps step:
      * 2 µs gives ~800 VCO cycles at the ~400 MHz target — plenty
        of settling time before the measurement window starts.
      * 25 ps step resolves the rising edge of the PULSE kick AND
        leaves >40 samples per VCO cycle at 400 MHz, enough for
        clean PP / RMS / crossing-time extraction.

Measures
--------

All measures aggregate over the settled window `[t_settle, t_end]`
(default [1.5 µs, 2.0 µs]).

  * `clk_out_pp`   — PP(v(clk_out)). PRIMARY oscillation indicator.
                     Healthy oscillation ≥ 0.5 V; symmetric DC = 0.
  * `clk_out_max`  — MAX(v(clk_out)). Peak high (can exceed vdd
                     because of LC flyback).
  * `clk_out_min`  — MIN(v(clk_out)). Peak low.
  * `clk_out_avg`  — AVG(v(clk_out)). DC offset of the oscillation.
                     For a symmetric tank centered at vdd, this is
                     approximately vdd_nom.
  * `clk_out_rms`  — RMS(v(clk_out)). Energy proxy. A clean sine at
                     ~400 MHz oscillating between rails has rms in
                     the (0.5 × vdd, 0.7 × vdd) band.

Why no native frequency measure
-------------------------------

ngspice's WHEN ... CROSS=N extension is not exposed in Clay's
neutral measure grammar (`clay.measure_expr`). A single
`time when v(clk_out) = vdd_nom` measure returns the first crossing
only — not enough to compute a period without ngspice-side
arithmetic. Frequency extraction is done outside the testbench in
the e2e test by either:

  (a) post-processing the netlist with an ngspice-specific .control
      script that captures multiple CROSS=N crossings and divides,
      or
  (b) reading the raw waveform via the upcoming raw-file reader.

Both paths are out of this testbench's scope. The pp/rms suite is
sufficient to assert "real oscillation occurred" — which is the
v0.0.57 contract.

Honest scope
------------

v0.0.57 verifies one thing: with the right combination of sizing
overrides (lower R_tank_series for higher Q, slightly larger W_osc
for more gm) and a current-pulse kick to break DC symmetry,
the cross-coupled NMOS LC-tank VCO does sustain real differential
oscillation in pure ngspice batch mode. The cell builder structure
shipped in v0.0.56 is unchanged; the kick lives at the testbench
layer; the e2e test wires sizing overrides via the cell_def's
`sizing { ... }` block so the .clay surface and the Python builder
stay consistent.

What is NOT covered:
  * Frequency-tuning via vctrl — needs a real varactor in the tank,
    deferred to a v0.0.58+ varactor primitive item.
  * Phase noise — needs .fourier post-processing or a Python-side
    raw-file FFT, driver-layer work.
  * Process / temp corner sweep beyond tt/27 °C — the testbench
    supports it through the standard `corners { }` / `temps { }`
    blocks, but the prompt's scope is tt/27 °C for the demo.
"""

from __future__ import annotations

from ..ir import (
    Cell, Device, Measure, OpAnalysis, Source, Testbench, TranAnalysis,
)

# ─── Defaults ──────────────────────────────────────────────────────
# Tuned to match the empirical "deterministic oscillation" envelope on
# sky130 (1.8 V) with the LC-VCO sizing overrides documented in
# examples/lc_tank_vco_demo.clay:
#   W_osc=40 µm, L_osc=0.5 µm, R_tank_series=0.5 Ω, vctrl_dc=1.7 V.
# Under these settings the cross-coupled pair's gm·R_tank product
# comfortably exceeds the 2× startup threshold and oscillation locks in
# by ~500 ns (well within the 1.5 µs t_settle).

DEFAULT_VCTRL_FRAC = 0.94       # vctrl ≈ 0.94 × vdd → high tail current
DEFAULT_KICK_AMP = 50e-3        # 50 mA peak — enough to break symmetry
                                # but small enough not to perturb final
                                # amplitude.
DEFAULT_KICK_DELAY = 1e-9       # start kick at t = 1 ns (after OP point
                                # is established).
DEFAULT_KICK_WIDTH = 2e-9       # 2 ns wide pulse
DEFAULT_KICK_EDGE = 100e-12     # 100 ps rise / fall — much faster than
                                # any LC time-constant.
DEFAULT_TRAN_STEP = 25e-12      # 25 ps step → >40 samples per cycle at
                                # 400 MHz target.
DEFAULT_TRAN_END = 2e-6         # 2 µs total — ~800 VCO cycles, plenty
                                # of settling time.
DEFAULT_T_SETTLE = 1.5e-6       # exclude first 1.5 µs from aggregates
                                # so the measure window samples the
                                # settled steady state, not the kick
                                # transient.
DEFAULT_VDD_FALLBACK = 1.8      # fallback supply if no PDK adapter
                                # exposes vdd_nom.


# Logical name of the cell's primary output port. The LC-VCO cell
# ships with `clk_out` (one phase of the differential tank); a future
# fully-differential variant would expose both phases as `clk_out_p`
# / `clk_out_n` and this testbench would need a small generalisation.
_OUTPUT_PORT = "clk_out"

# Required ports the cell must declare for this testbench to wire up
# correctly. `clk_out` (output), `vctrl` (tail bias), `vdd`, `vss`.
_REQUIRED_PORTS = ("clk_out", "vctrl", "vdd", "vss")


def find_lc_vco_ports(cell: Cell) -> bool:
    """Return True iff `cell` matches the LC-VCO testbench's port shape.

    The LC-VCO testbench requires exactly the port set declared by
    `build_lc_tank_vco`: `(clk_out, vctrl, vdd, vss)`. This is the
    same surface a ring VCO declares; what distinguishes the LC-VCO
    use case is the topology style (`lc_tank_vco`) attached to the
    cell_def, NOT the port surface itself. The CLI's port-shape
    heuristic ladder still routes through `find_pll_ports` first
    (which requires `ref_in`), so this helper is used by the stage
    template registry's testbench callback, not by the CLI ladder.

    Returns True if every required port is present.
    """
    return all(p in cell.ports for p in _REQUIRED_PORTS)


def build_lc_tank_vco_testbench(
    cell: Cell,
    *,
    vdd_nom: float | None = None,
    vctrl_frac: float = DEFAULT_VCTRL_FRAC,
    vctrl_dc: float | None = None,
    vbias_tail_dc: float | None = None,
    kick_amp: float = DEFAULT_KICK_AMP,
    kick_delay: float = DEFAULT_KICK_DELAY,
    kick_width: float = DEFAULT_KICK_WIDTH,
    kick_edge: float = DEFAULT_KICK_EDGE,
    tran_step: float = DEFAULT_TRAN_STEP,
    tran_end: float = DEFAULT_TRAN_END,
    t_settle: float = DEFAULT_T_SETTLE,
) -> Testbench:
    """Build a transient testbench for the cross-coupled NMOS LC-tank VCO.

    Wires VDD / VSS supplies, a DC control voltage on `vctrl`, and a
    short current-pulse kick on `clk_out` that breaks the symmetric
    DC operating point so transient observes real oscillation.

    Args:
        cell: the LC-VCO cell IR. MUST declare `vdd`, `vss`, `vctrl`,
            `clk_out` ports.
        vdd_nom: nominal supply in volts. If None, auto-resolves from
            the cell's PDK adapter (sky130 → 1.8 V, gf180mcu → 3.3 V).
        vctrl_frac: vctrl DC level as a fraction of vdd_nom. Default
            0.94 → vctrl ≈ 1.7 V on sky130. Higher vctrl → more
            tail current → easier startup. Used iff `vctrl_dc` is
            None.
        vctrl_dc: explicit vctrl DC level in volts. Overrides
            `vctrl_frac` when supplied.
        kick_amp: peak current of the startup pulse in amperes.
            Default 50 mA — empirically large enough to break the
            symmetric DC OP, small enough that the kick is gone
            within a few ns and doesn't set the final amplitude.
        kick_delay: time at which the kick fires (seconds).
        kick_width: kick duration (seconds).
        kick_edge: rise / fall time of the kick (seconds).
        tran_step: transient time step in seconds.
        tran_end: total transient window in seconds.
        t_settle: lower bound of the aggregate measurement window.
            Aggregates run over [t_settle, tran_end].

    Returns:
        A Testbench with VDD / VSS / VCTRL / IKICK sources, an OP +
        TRAN analysis pair, and the default oscillation-health
        measure suite.

    Raises:
        ValueError: if the cell does not declare the required ports
            (`vdd`, `vss`, `vctrl`, `clk_out`).
    """
    # ─── Required-port validation ──────────────────────────────
    missing = [p for p in _REQUIRED_PORTS if p not in cell.ports]
    if missing:
        raise ValueError(
            f"LC-tank VCO testbench: cell {cell.name!r} is missing "
            f"required ports {missing!r}. The testbench wires the "
            f"cell's {list(_REQUIRED_PORTS)} ports to supplies / "
            f"control / kick. Cell port list: {cell.ports}."
        )

    # ─── Supply resolution ────────────────────────────────────
    if vdd_nom is None:
        from ._supply import resolve_supply
        vdd_nom = resolve_supply(cell, default=DEFAULT_VDD_FALLBACK)
    if vctrl_dc is None:
        vctrl_dc = float(vdd_nom) * float(vctrl_frac)

    # v0.0.60: if the cell declares a `vbias_tail` port (the
    # decouple_vbias_tail=True path), the testbench wires a separate
    # DC source. The default tail bias is set to a fixed voltage that
    # keeps Mtail comfortably in saturation independent of vctrl:
    #   * sky130 1.8 V → ~0.9 V tail bias (Vgs ≈ 0.9 V, well above
    #     Vth_n ≈ 0.5 V → solid overdrive)
    #   * gf180mcu 3.3 V → ~1.5 V tail bias
    # The 0.5 × vdd anchor scales gracefully across PDKs without
    # forcing PDK-specific defaults into this builder.
    has_vbias_tail = "vbias_tail" in cell.ports
    if has_vbias_tail and vbias_tail_dc is None:
        vbias_tail_dc = 0.5 * float(vdd_nom)

    # ─── Source list ───────────────────────────────────────────
    # Period of the kick PULSE is set far beyond tran_end so only the
    # first rising edge fires inside the simulation window. The
    # cross-coupled pair's negative resistance takes over once the
    # symmetry is broken; the kick itself is irrelevant to the
    # steady-state amplitude.
    big_period = tran_end * 1000.0

    sources: list[Source] = [
        Source(
            kind="voltage_dc",
            instance_name="VDD",
            pins={"p": "vdd", "n": "0"},
            params={"dc": float(vdd_nom)},
        ),
        Source(
            kind="voltage_dc",
            instance_name="VSS",
            pins={"p": "vss", "n": "0"},
            params={"dc": 0.0},
        ),
        Source(
            kind="voltage_dc",
            instance_name="VCTRL",
            pins={"p": "vctrl", "n": "0"},
            params={"dc": float(vctrl_dc)},
        ),
        # Current pulse from ground to clk_out. Pulls a small charge
        # OUT of clk_out (positive `v_high` with `p=0, n=clk_out`
        # means conventional current flows from 0 into clk_out via
        # the source's internal direction, which raises clk_out's
        # voltage briefly). Either polarity works — what matters is
        # breaking the dV(clk_out)/dt = dV(outn)/dt symmetry so the
        # cross-coupled pair's small-signal loop gain has something
        # to amplify.
        Source(
            kind="current_pulse",
            instance_name="IKICK",
            pins={"p": "0", "n": "clk_out"},
            params={
                "v_low": 0.0,
                "v_high": float(kick_amp),
                "td": float(kick_delay),
                "t_rise": float(kick_edge),
                "t_fall": float(kick_edge),
                "pulse_width": float(kick_width),
                "period": big_period,
            },
        ),
    ]

    # v0.0.60: append a VBIAS_TAIL DC source if the cell decoupled
    # the tail bias from vctrl. The fixed tail bias holds Mtail in
    # saturation across the full vctrl swing.
    if has_vbias_tail:
        sources.append(
            Source(
                kind="voltage_dc",
                instance_name="VBIAS_TAIL",
                pins={"p": "vbias_tail", "n": "0"},
                params={"dc": float(vbias_tail_dc)},
            )
        )

    # ─── Analyses ──────────────────────────────────────────────
    analyses: list = [
        OpAnalysis(),
        TranAnalysis(step=tran_step, t_end=tran_end),
    ]

    # ─── Measure suite ─────────────────────────────────────────
    measures = _lc_vco_measures(t_settle=t_settle, t_end=tran_end)

    return Testbench(
        cell=cell,
        sources=sources,
        extra_devices=[],
        analyses=analyses,
        measures=measures,
        pdk_corner="tt",
        supply_volt=float(vdd_nom),
    )


def _lc_vco_measures(*, t_settle: float, t_end: float) -> list[Measure]:
    """Build the `.meas` list for the LC-VCO testbench.

    All measures aggregate over the settled steady-state window
    `[t_settle, t_end]`. Order convention follows the existing
    PLL / LDO / CP / sensor testbenches: MAX → MIN → AVG → RMS → PP,
    aggregates before any find-style measures (dodges ngspice's
    sticky-state .meas quirk where a failed earlier .meas can zero
    out subsequent AVG-style ones).
    """
    out: list[Measure] = []
    out.append(Measure(
        name="clk_out_max",
        analysis_ref="tran",
        expr=(
            f"MAX(v({_OUTPUT_PORT})) over time "
            f"from {t_settle:g} to {t_end:g}"
        ),
    ))
    out.append(Measure(
        name="clk_out_min",
        analysis_ref="tran",
        expr=(
            f"MIN(v({_OUTPUT_PORT})) over time "
            f"from {t_settle:g} to {t_end:g}"
        ),
    ))
    out.append(Measure(
        name="clk_out_avg",
        analysis_ref="tran",
        expr=(
            f"AVG(v({_OUTPUT_PORT})) over time "
            f"from {t_settle:g} to {t_end:g}"
        ),
    ))
    out.append(Measure(
        name="clk_out_rms",
        analysis_ref="tran",
        expr=(
            f"RMS(v({_OUTPUT_PORT})) over time "
            f"from {t_settle:g} to {t_end:g}"
        ),
    ))
    out.append(Measure(
        name="clk_out_pp",
        analysis_ref="tran",
        expr=(
            f"PP(v({_OUTPUT_PORT})) over time "
            f"from {t_settle:g} to {t_end:g}"
        ),
    ))
    return out
