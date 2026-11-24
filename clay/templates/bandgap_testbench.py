"""
Bandgap / voltage-reference testbench builder (v0.0.35).

The default AC testbench (`build_ac_testbench` in `two_stage_miller_pmos.py`)
is shaped for differential opamp characterization — it drives an
inn/inp/out/vdd/vss cell with a differential VCVS pair and measures
gain_dc / gbw at the `out` node. That rig cannot run on a voltage
reference because:

  1. A bandgap has no differential inputs — it's self-biased.
  2. A bandgap typically has TWO reference outputs (vref1/vref2 in the
     AnalogGym dual-output sub-threshold CMOS reference) and NO single
     `out` port.
  3. The native bandgap measurement is V(vref) at multiple temperatures,
     which yields the temperature coefficient TC = (V_pp / V_avg) / dT.
     None of the AC measures (gain_dc, gbw, phase_at_gbw) apply.

`build_bandgap_testbench` produces a rig in which:

  * VDD is a DC source on the cell's `vdd` port (nominal 1.8 V).
  * VSS is a DC source on the cell's `vss` port (0 V).
  * NO additional stimulus — the bandgap is self-biased through its
    internal MOSFET loops.

Analyses:
  * `.op` for the operating point.
  * A trivial DC sweep on VDD around the nominal value (3 points, step
    0.01 V). This is purely a workaround for ngspice's limitation that
    `.meas op` is not supported — only `.meas dc / .meas ac / .meas tran`
    can extract scalar values. By sweeping VDD over a tight range with
    a step matching the range, every sweep point is at the nominal Vdd
    and the `.meas dc find ... at=vdd_nom` extraction works.

Measures (auto-derived from the cell's port list):
  * For every cell port matching the pattern `vref*` (case-insensitive),
    we emit four measures:
      - `vrefN`         : v(vrefN) at the nominal Vdd
      - `vrefN_max`     : MAX(v(vrefN)) over the trivial sweep
      - `vrefN_min`     : MIN(v(vrefN)) over the trivial sweep
      - `vrefN_avg`     : AVG(v(vrefN)) over the trivial sweep
    Since the sweep is tight (3 points all at Vdd_nom ± 0.01 V), the
    max/min/avg values are essentially identical to the `at=vdd_nom`
    sample — they exist for symmetry with the LDO testbench and to
    expose a uniform interface for downstream temperature-coefficient
    post-processing (see v0.0.35 docstring on TC derivation).

Temperature sweep integration:
  * `tb.temp` is plumbed through by the CLI (v0.0.24). When a verify
    job declares `temps { -40C; 27C; 85C; }`, the CLI builds this
    testbench once and overrides `tb.temp` at each temperature point;
    each point runs as a separate ngspice sim and produces its own
    Vref values. The verifier reports per-temp measures so the user
    can derive TC = (Vref_max - Vref_min) / (T_max - T_min)
    post-process if they want.

  * Per-temperature TC derivation is NOT performed by this testbench
    in v0.0.35; doing so would require coordination between separate
    sim runs (cross-temp aggregation) which sits at the verifier
    layer, not the testbench layer. The CLI prints per-point Vref
    numbers and the user can compute TC from those. A future v0.0.36+
    `derived_measure` feature spanning the temp sweep axis would
    automate this.

The testbench accepts optional kwargs to override the supply nominal
and sweep characteristics, but the defaults match AnalogGym's
dual-output sub-threshold reference (vdda = 1.8 V).
"""

from __future__ import annotations

import re

from ..ir import (
    Cell, DcAnalysis, Measure, OpAnalysis, Source, Testbench,
)


# ─── Defaults ──────────────────────────────────────────────────────
# Picked to match AnalogGym's bandgap testbench (vdda = 1.8 V nominal).
DEFAULT_VDD_NOM = 1.8         # nominal supply for sub-threshold bandgap
DEFAULT_VDD_SWEEP_DELTA = 0.01  # ±0.01 V around nominal — keeps the sweep
                                # tight enough to be effectively a single
                                # operating point.
DEFAULT_VDD_SWEEP_STEP = 0.01


# Regex for matching a cell port that carries a voltage-reference
# output. Matches `vref`, `vref1`, `vref2`, `vref_a`, etc.
# Case-insensitive. The `vref_ports` field on the testbench exposes
# the matched port list to downstream tooling.
_VREF_PORT_RE = re.compile(r"^vref[\w]*$", re.IGNORECASE)


def find_vref_ports(cell: Cell) -> list[str]:
    """Return the list of `vref*` ports declared on the cell, in
    declaration order.

    A "bandgap-shaped" cell is identified by having at least one port
    whose name starts with `vref` (case-insensitive). This function is
    also used by the CLI's port-shape heuristic to decide whether to
    pick the bandgap testbench over the default differential AC rig.
    """
    return [p for p in cell.ports if _VREF_PORT_RE.match(p)]


def build_bandgap_testbench(
    cell: Cell,
    *,
    vdd_nom: float | None = None,
    vdd_sweep_delta: float = DEFAULT_VDD_SWEEP_DELTA,
    vdd_sweep_step: float = DEFAULT_VDD_SWEEP_STEP,
) -> Testbench:
    """Build the default bandgap / voltage-reference testbench.

    `vdd_nom=None` auto-resolves from `cell.pdk_id` (sky130 → 1.8 V,
    gf180mcu → 3.3 V).

    Wiring:
      * VDD — voltage_dc, `vdd` ↔ 0, dc=vdd_nom. The supply rail.
      * VSS — voltage_dc, `vss` ↔ 0, dc=0. Fixed ground.

    No additional stimulus, no feedback loops, no load — the bandgap
    is self-biased through its internal MOSFET sub-threshold loop.

    The analyses list is `[OpAnalysis(), DcAnalysis(VDD)]` where the
    DcAnalysis is a tight sweep around `vdd_nom` (3 points by default).
    This is a workaround for ngspice's limitation that `.meas op` is
    not a supported analysis-type: `.meas dc / .meas ac / .meas tran`
    are the only kinds that can extract a scalar. A 3-point sweep is
    the cheapest analysis that lets `.meas dc find v(vrefN) at=vdd_nom`
    work.

    Measures auto-generate from the cell's `vref*` port list. For
    each `vrefN` port found, four measures are emitted:
      * vrefN          — v(vrefN) at the sweep point closest to vdd_nom
      * vrefN_max      — MAX(v(vrefN)) over the trivial sweep
      * vrefN_min      — MIN(v(vrefN)) over the trivial sweep
      * vrefN_avg      — AVG(v(vrefN)) over the trivial sweep

    Args:
        cell: the bandgap cell IR. Must declare a `vdd` port, a `vss`
            port, and at least one `vref*` port. If no `vref*` ports
            are found a ValueError is raised — this testbench has
            nothing to measure on a cell without reference outputs.
        vdd_nom: nominal supply voltage in volts.
        vdd_sweep_delta: half-width of the DC sweep around vdd_nom.
            Default 0.01 V. The sweep range is
            `[vdd_nom - delta, vdd_nom + delta]`.
        vdd_sweep_step: step size for the DC sweep, in volts.

    Returns:
        A Testbench with VDD/VSS sources, OP + DC analyses, and
        per-vref-port measures. Caller (typically the CLI) is
        responsible for overriding `tb.temp` and `tb.pdk_corner` for
        each requested sweep point.

    Raises:
        ValueError: if the cell has no `vref*` ports, or if the cell
            is missing a `vdd` / `vss` port (the testbench is
            unwireable without those).
    """
    if "vdd" not in cell.ports:
        raise ValueError(
            f"bandgap testbench: cell {cell.name!r} has no `vdd` port "
            f"(ports={cell.ports}). The testbench wires the cell's "
            f"vdd to a DC supply — a bandgap cell must declare a vdd "
            f"port."
        )
    if "vss" not in cell.ports:
        raise ValueError(
            f"bandgap testbench: cell {cell.name!r} has no `vss` port "
            f"(ports={cell.ports}). The testbench wires the cell's "
            f"vss to ground."
        )
    if vdd_nom is None:
        from ._supply import resolve_supply
        vdd_nom = resolve_supply(cell, default=DEFAULT_VDD_NOM)
    vref_ports = find_vref_ports(cell)
    if not vref_ports:
        raise ValueError(
            f"bandgap testbench: cell {cell.name!r} has no `vref*` "
            f"ports (ports={cell.ports}). The bandgap testbench has "
            f"nothing to measure on a cell without reference outputs. "
            f"For a non-reference cell, use a different testbench "
            f"(e.g. build_ac_testbench for opamps, build_ldo_testbench "
            f"for LDOs)."
        )
    sources: list[Source] = [
        Source("voltage_dc", "VDD",
               pins={"p": "vdd", "n": "0"},
               params={"dc": vdd_nom}),
        Source("voltage_dc", "VSS",
               pins={"p": "vss", "n": "0"},
               params={"dc": 0.0}),
    ]
    # Tight DC sweep around vdd_nom. The sweep itself is a workaround
    # for ngspice's lack of `.meas op` support; the bandgap reference
    # value should be essentially constant across this tight Vdd range
    # (a well-designed bandgap has line-regulation << 1 mV/V).
    sweep_lo = vdd_nom - vdd_sweep_delta
    sweep_hi = vdd_nom + vdd_sweep_delta
    # ngspice prefixes voltage sources with `V`, so the IR's `VDD`
    # becomes `VVDD` in the netlist.
    analyses: list = [
        OpAnalysis(),
        DcAnalysis(
            sweep_source="VVDD",
            start=sweep_lo,
            stop=sweep_hi,
            step=vdd_sweep_step,
        ),
    ]
    measures: list[Measure] = _vref_measures(
        vref_ports, sweep_lo=sweep_lo, sweep_hi=sweep_hi, vdd_nom=vdd_nom,
    )
    return Testbench(
        cell=cell,
        sources=sources,
        extra_devices=[],
        analyses=analyses,
        measures=measures,
        pdk_corner="tt",
        supply_volt=vdd_nom,
    )


def _vref_measures(
    vref_ports: list[str],
    *,
    sweep_lo: float,
    sweep_hi: float,
    vdd_nom: float,
) -> list[Measure]:
    """Build the .meas list for a bandgap testbench.

    For each `vrefN` port, emit four measures:
      * vrefN          — v(vrefN) at the nominal Vdd (the headline
                         reference value)
      * vrefN_max      — MAX(v(vrefN)) over the trivial sweep
      * vrefN_min      — MIN(v(vrefN)) over the trivial sweep
      * vrefN_avg      — AVG(v(vrefN)) over the trivial sweep

    All four use the `dc` analysis_ref (ngspice's `.meas dc`) since
    the testbench includes a DcAnalysis on VDD. Without the DC sweep
    we couldn't emit `.meas` lines at all (ngspice rejects `.meas op`).
    """
    # Order matters: ngspice's .meas processor has a sticky state quirk
    # where a `find` before an `AVG` causes AVG to report "out of
    # interval" and produce no value. Emitting all aggregates BEFORE
    # the find dodges this — pattern confirmed against the LDO
    # testbench's working measure order. The bandgap testbench follows
    # the same convention: MAX → MIN → AVG → find.
    out: list[Measure] = []
    for port in vref_ports:
        out.append(Measure(
            name=f"{port}_max",
            analysis_ref="dc",
            expr=f"MAX(v({port})) over dc_sweep from {sweep_lo:g} to {sweep_hi:g}",
        ))
        out.append(Measure(
            name=f"{port}_min",
            analysis_ref="dc",
            expr=f"MIN(v({port})) over dc_sweep from {sweep_lo:g} to {sweep_hi:g}",
        ))
        out.append(Measure(
            name=f"{port}_avg",
            analysis_ref="dc",
            expr=f"AVG(v({port})) over dc_sweep from {sweep_lo:g} to {sweep_hi:g}",
        ))
        out.append(Measure(
            name=port,
            analysis_ref="dc",
            expr=f"v({port}) @ dc_sweep={vdd_nom:g}",
        ))
    return out
