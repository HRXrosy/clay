"""
Sensor / front-end testbench builder (v0.0.35 follow-on).

The default AC testbench (`build_ac_testbench` in `two_stage_miller_pmos.py`)
is shaped for differential opamp characterization — it drives an
inn/inp/out/vdd/vss cell with a differential VCVS pair and measures
gain_dc / gbw at the `out` node. That rig cannot run on a single-ended
sensor front end (PTAT temperature sensor, light sensor, etc.) because:

  1. A sensor has NO differential inputs — its "input" is a physical
     variable (temperature, illumination) that the circuit modulates
     via device parameters, not a port the testbench wires.
  2. A sensor has ONE single-ended output (vsense / tsense_out /
     sensor_out) and zero inputs. Port shape is therefore `(vsense,
     vdd, vss)` — strictly 3 ports.
  3. The native sensor measurement is V(vsense) AT MULTIPLE
     TEMPERATURES, which yields the temperature coefficient (the LSB,
     in V/°C, between two temperature points). None of the AC measures
     (gain_dc, gbw, phase_at_gbw) apply.

`build_sensor_testbench` produces a rig in which:

  * VDD is a DC source on the cell's `vdd` port (nominal 1.8 V).
  * VSS is a DC source on the cell's `vss` port (0 V).
  * NO additional stimulus — the sensor is self-biased through its
    internal stack (e.g. the PTAT 2-MOS cell's M0 source-follower
    above an M1 diode at vsense).

Analyses:
  * `.op` for the operating point.
  * A trivial DC sweep on VDD around the nominal value (3 points,
    step 0.01 V). Same workaround as the bandgap testbench: ngspice's
    `.meas op` is not supported (only `.meas dc / .meas ac / .meas
    tran` can extract scalar values), so a 3-point sweep gives us a
    `.meas dc find v(vsense) at=vdd_nom` extraction at the nominal
    operating point. The sweep is intentionally narrow so the sensor's
    sub-threshold operating point doesn't perturb between sample
    points — a well-designed sensor has line-regulation << 1 mV/V.

ngspice convergence note:
  * Deep sub-threshold sensors at temperature corners (-20 / 120 °C)
    occasionally push ngspice's solver into bad Newton steps. The
    standard ngspice options (`gmin = 1e-12`, `reltol = 1e-3`)
    typically suffice; if a sensor cell needs looser knobs, a future
    driver-level extension would surface them through the testbench
    IR. v0.0.35 relies on ngspice's defaults — the e2e test
    documents any convergence behaviour observed at the AnalogGym
    PTAT corners.

Measures (auto-derived from the cell's sensor-output port list):
  * For every cell port matching the pattern `vsense*`, `tsense*`,
    or `sensor_out*` (case-insensitive), we emit four measures:
      - `<port>`         : v(<port>) at the nominal Vdd
      - `<port>_max`     : MAX(v(<port>)) over the trivial sweep
      - `<port>_min`     : MIN(v(<port>)) over the trivial sweep
      - `<port>_avg`     : AVG(v(<port>)) over the trivial sweep
    Same shape as the bandgap testbench's per-`vref*` measure suite.
    Since the sweep is tight (3 points all at Vdd_nom ± 0.01 V), the
    max/min/avg values are essentially identical to the `at=vdd_nom`
    sample — they exist for symmetry with the LDO/bandgap testbenches
    and to expose a uniform interface for downstream temperature
    sweep aggregation.

Temperature sweep integration:
  * `tb.temp` is plumbed through by the CLI. When a verify job
    declares `temps { -20C; 25C; 75C; 120C; }`, the CLI builds this
    testbench once and overrides `tb.temp` at each temperature
    point; each point runs as a separate ngspice sim and produces
    its own V(vsense) value. The verifier reports per-temp measures
    so the user (or a downstream post-processor) can derive the LSB:
        lsb := (vsense_at_high_temp - vsense_at_low_temp)
               / (temp_high - temp_low)
    The cross-temp derivation is NOT done by this testbench in
    v0.0.35 — it would require coordination between separate sim
    runs (cross-temp aggregation) which sits at the verifier layer,
    not the testbench layer. The CLI prints per-point V(vsense)
    numbers and the user can compute the LSB from those.

The testbench accepts optional kwargs to override the supply nominal
and sweep characteristics, but the defaults match the AnalogGym
PTAT_SENSOR testbench (vdda = 1.8 V).
"""

from __future__ import annotations

import re

from ..ir import (
    Cell, DcAnalysis, Measure, OpAnalysis, Source, Testbench,
)


# ─── Defaults ──────────────────────────────────────────────────────
# Picked to match the AnalogGym PTAT_SENSOR testbench (vdda = 1.8 V
# nominal). Sub-threshold MOSFET sensors operate at any supply that
# keeps the stack above its conduction threshold — 1.8 V is generous
# for the 2-T cell.
DEFAULT_VDD_NOM = 1.8         # nominal supply for sub-threshold sensors
DEFAULT_VDD_SWEEP_DELTA = 0.01  # ±0.01 V around nominal — keeps the sweep
                                # tight enough to be effectively a single
                                # operating point.
DEFAULT_VDD_SWEEP_STEP = 0.01


# Regex for matching a cell port that carries a sensor read-out.
# Matches `vsense`, `vsense1`, `tsense_out`, `sensor_out`, etc.
# Case-insensitive. The `sensor_ports` returned by `find_sensor_ports`
# is exposed to the CLI's port-shape heuristic.
_SENSOR_PORT_RE = re.compile(
    r"^(?:vsense|tsense|sensor_out)[\w]*$", re.IGNORECASE,
)


def find_sensor_ports(cell: Cell) -> list[str]:
    """Return the list of sensor-output ports declared on the cell, in
    declaration order.

    A "sensor-shaped" cell is identified by having at least one port
    whose name starts with `vsense`, `tsense`, or `sensor_out`
    (case-insensitive). This function is also used by the CLI's
    port-shape heuristic to decide whether to pick the sensor
    testbench over the bandgap / CP / differential AC rigs.

    Note: a bandgap cell with `vref*` ports is NOT matched by this
    regex — the bandgap heuristic must remain a sibling check. The
    CLI ladder is ordered (vref → bandgap, vsense → sensor, ...)
    so a cell with both `vref` and `vsense` ports (rare) would route
    to the bandgap branch first.
    """
    return [p for p in cell.ports if _SENSOR_PORT_RE.match(p)]


def build_sensor_testbench(
    cell: Cell,
    *,
    vdd_nom: float | None = None,
    vdd_sweep_delta: float = DEFAULT_VDD_SWEEP_DELTA,
    vdd_sweep_step: float = DEFAULT_VDD_SWEEP_STEP,
) -> Testbench:
    """Build the default sensor / front-end testbench.

    `vdd_nom=None` auto-resolves from `cell.pdk_id`.

    Wiring:
      * VDD — voltage_dc, `vdd` ↔ 0, dc=vdd_nom. The supply rail.
      * VSS — voltage_dc, `vss` ↔ 0, dc=0. Fixed ground.

    No additional stimulus, no feedback loops, no load — the sensor
    is self-biased through its internal stack (e.g. PTAT 2-MOS cell:
    M0 source-follower on top of M1 diode at vsense).

    The analyses list is `[OpAnalysis(), DcAnalysis(VVDD)]` where the
    DcAnalysis is a tight sweep around `vdd_nom` (3 points by
    default). This is a workaround for ngspice's limitation that
    `.meas op` is not a supported analysis-type: `.meas dc / .meas
    ac / .meas tran` are the only kinds that can extract a scalar.
    A 3-point sweep is the cheapest analysis that lets
    `.meas dc find v(vsense) at=vdd_nom` work.

    Measures auto-generate from the cell's sensor-output port list.
    For each `<port>` matched by `find_sensor_ports`, four measures
    are emitted: `<port>`, `<port>_max`, `<port>_min`, `<port>_avg`.

    Args:
        cell: the sensor cell IR. Must declare a `vdd` port, a `vss`
            port, and at least one sensor-output port (matching
            `vsense*` / `tsense*` / `sensor_out*`). If no such port
            is found, ValueError is raised.
        vdd_nom: nominal supply voltage in volts.
        vdd_sweep_delta: half-width of the DC sweep around vdd_nom.
            Default 0.01 V. The sweep range is
            `[vdd_nom - delta, vdd_nom + delta]`.
        vdd_sweep_step: step size for the DC sweep, in volts.

    Returns:
        A Testbench with VDD/VSS sources, OP + DC analyses, and
        per-sensor-port measures. Caller (typically the CLI) is
        responsible for overriding `tb.temp` and `tb.pdk_corner` for
        each requested sweep point.

    Raises:
        ValueError: if the cell has no sensor-output ports, or if the
            cell is missing a `vdd` / `vss` port (the testbench is
            unwireable without those).
    """
    if "vdd" not in cell.ports:
        raise ValueError(
            f"sensor testbench: cell {cell.name!r} has no `vdd` port "
            f"(ports={cell.ports}). The testbench wires the cell's "
            f"vdd to a DC supply — a sensor cell must declare a vdd "
            f"port."
        )
    if "vss" not in cell.ports:
        raise ValueError(
            f"sensor testbench: cell {cell.name!r} has no `vss` port "
            f"(ports={cell.ports}). The testbench wires the cell's "
            f"vss to ground."
        )
    if vdd_nom is None:
        from ._supply import resolve_supply
        vdd_nom = resolve_supply(cell, default=DEFAULT_VDD_NOM)
    sensor_ports = find_sensor_ports(cell)
    if not sensor_ports:
        raise ValueError(
            f"sensor testbench: cell {cell.name!r} has no "
            f"`vsense*` / `tsense*` / `sensor_out*` ports "
            f"(ports={cell.ports}). The sensor testbench has nothing "
            f"to measure on a cell without sensor outputs. For a "
            f"non-sensor cell, use a different testbench (e.g. "
            f"build_bandgap_testbench for voltage references, "
            f"build_ac_testbench for opamps, build_ldo_testbench for "
            f"LDOs)."
        )
    sources: list[Source] = [
        Source("voltage_dc", "VDD",
               pins={"p": "vdd", "n": "0"},
               params={"dc": vdd_nom}),
        Source("voltage_dc", "VSS",
               pins={"p": "vss", "n": "0"},
               params={"dc": 0.0}),
    ]
    # Tight DC sweep around vdd_nom. Same workaround as the bandgap
    # testbench (ngspice rejects .meas op; .meas dc is the supported
    # path for scalar extraction at an operating point).
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
    measures: list[Measure] = _sensor_measures(
        sensor_ports, sweep_lo=sweep_lo, sweep_hi=sweep_hi, vdd_nom=vdd_nom,
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


def _sensor_measures(
    sensor_ports: list[str],
    *,
    sweep_lo: float,
    sweep_hi: float,
    vdd_nom: float,
) -> list[Measure]:
    """Build the .meas list for a sensor testbench.

    For each sensor-output `<port>`, emit four measures:
      * <port>          — v(<port>) at the nominal Vdd (the headline
                          read-out value)
      * <port>_max      — MAX(v(<port>)) over the trivial sweep
      * <port>_min      — MIN(v(<port>)) over the trivial sweep
      * <port>_avg      — AVG(v(<port>)) over the trivial sweep

    All four use the `dc` analysis_ref (ngspice's `.meas dc`) since
    the testbench includes a DcAnalysis on VDD. Without the DC sweep
    we couldn't emit `.meas` lines at all (ngspice rejects `.meas
    op`).

    Order: MAX → MIN → AVG → FIND. ngspice's .meas processor has a
    sticky state quirk where a `find` before an `AVG` causes AVG to
    report "out of interval" and produce no value. Emitting all
    aggregates BEFORE the find dodges this — same convention as the
    bandgap testbench (which lifted the order from the LDO testbench
    after confirming the bug pattern there).
    """
    out: list[Measure] = []
    for port in sensor_ports:
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
