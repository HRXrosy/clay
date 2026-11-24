"""
LDO-specific testbench builder (v0.0.33).

The default AC testbench (`build_ac_testbench` in `two_stage_miller_pmos.py`)
is shaped for differential opamp characterization: VCVS pair drives
inp/inn with opposite-phase AC, plus a load cap on `out`. That rig
cannot measure LDO-relevant specs (line regulation, load regulation,
dropout voltage, PSRR) because:

  1. There is no Vin sweep capability — Vdd is a fixed DC supply.
  2. There is no load current source — `out` is an open node loaded
     only by a small cap.
  3. The DUT's `inn` port (Vfb) is driven by the VCVS as the −
     differential input, not connected to `out` in unity-gain
     feedback.

`build_ldo_testbench` produces a rig in which:

  * Vdd is a dedicated, possibly-swept supply (`VIN` source).
  * Vref is a dedicated DC reference at `inp` (`VREF` source).
  * `out` is loaded by a current source (`ILOAD`) sinking a
    configurable steady-state load current.
  * `inn` is shorted to `out` (unity-gain feedback) through a 0 V
    voltage source — this is the standard LDO-loop closing trick.
    At DC the 0 V source is a wire; it is also a wire at AC (no
    series inductor used here, since the loop-gain AC analysis
    uses different stimuli).

The four LDO drive modes (`dc_vin_sweep`, `dc_iload_sweep`,
`tran_load_step`, `tran_line_step`) are *rewrites* of this base
testbench, performed by
`clay.verifier.rewrite_testbench_for_drive_mode`. The base testbench
itself emits a default DC sweep on Vin so a single-setup verify
without `drive { }` blocks already produces meaningful line-
regulation numbers.

Measures emitted by default:
  * vout_at_nom      — Vout at the nominal Vin (set via `vin_nom`)
  * vout_max         — max Vout across the Vin sweep range
  * vout_min         — min Vout across the Vin sweep range
  * vout_avg         — average Vout across the sweep range
  * vout_pp          — peak-to-peak Vout across the sweep range
  * line_regulation  — pp(Vout) / (Vin_max - Vin_min), in V/V
  * load_regulation  — present only after dc_iload_sweep rewrite;
                       pp(Vout) / (Iload_max - Iload_min), in V/A
  * dropout_voltage  — Vin - Vout at the lowest Vin point of the
                       sweep (where the regulator falls out of
                       regulation)

The exact measures emitted depend on which drive mode rewrote the
testbench. The base testbench's measure list is a SUPERSET of what
any single drive mode will produce; the rewriter prunes / adjusts
as needed.
"""

from __future__ import annotations

from ..ir import (
    AcAnalysis, Cell, DcAnalysis, Device, Measure, OpAnalysis, Source,
    Spec, Testbench, TranAnalysis,
)


# ─── Default constants for the base testbench ──────────────────────
# Picked to match AnalogGym ldo_simple's testbench numbers so the
# benchmark cell verifies against the same operating point AnalogGym
# uses. Callers can override at build time.

DEFAULT_VIN_NOM = 2.0      # 2.0 V nominal input (AnalogGym ldo_simple)
DEFAULT_VIN_MIN = 1.6      # sweep lower bound (just into dropout for 1.8V ref)
DEFAULT_VIN_MAX = 3.0      # sweep upper bound
DEFAULT_VIN_STEP = 0.01    # sweep step
DEFAULT_VREF = 1.8         # AnalogGym ldo_simple Vref
DEFAULT_ILOAD_NOM = 1e-3   # 1 mA steady-state load
DEFAULT_ILOAD_MIN = 1e-5   # 10 uA min load (load regulation sweep)
DEFAULT_ILOAD_MAX = 1e-2   # 10 mA max load
DEFAULT_ILOAD_STEP = 1e-5  # 10 uA step
DEFAULT_CL = 1e-9          # 1 nF output cap (LDO stability load)
DEFAULT_TRAN_STEP = 1e-7   # 100 ns transient step
DEFAULT_TRAN_END = 1e-4    # 100 us transient end time


def build_ldo_testbench(
    cell: Cell,
    *,
    vin_nom: float | None = None,
    vin_min: float = DEFAULT_VIN_MIN,
    vin_max: float = DEFAULT_VIN_MAX,
    vin_step: float = DEFAULT_VIN_STEP,
    vref: float = DEFAULT_VREF,
    iload_nom: float = DEFAULT_ILOAD_NOM,
    iload_min: float = DEFAULT_ILOAD_MIN,
    iload_max: float = DEFAULT_ILOAD_MAX,
    iload_step: float = DEFAULT_ILOAD_STEP,
    CL: float = DEFAULT_CL,
) -> Testbench:
    """Build the default LDO testbench.

    Wiring:
      * VIN     — voltage_dc, `vdd` ↔ 0, dc=vin_nom. Becomes the swept
                  source under `dc_vin_sweep` rewrite (and a stepped
                  source under `tran_line_step`).
      * VSS     — voltage_dc, `vss` ↔ 0, dc=0. Fixed ground.
      * VREF    — voltage_dc, `inp` ↔ 0, dc=vref. Reference voltage
                  for the error amp's + input.
      * VFB     — voltage_dc, `inn` ↔ `out`, dc=0. Unity-gain
                  feedback — acts as a wire connecting Vfb (= inn)
                  to Vreg (= out). The 0 V source doubles as an
                  ammeter for the feedback current.
      * ILOAD   — current_dc, `out` ↔ `vss`, dc=iload_nom. Steady-
                  state load. Becomes the swept source under
                  `dc_iload_sweep` and stepped under
                  `tran_load_step`.
      * CL      — cap, `out` ↔ `vss`, value=CL. Output bulk
                  capacitance (LDO stability ESL/ESR is not modeled
                  in this v0).

    The base testbench's default analysis is a DC sweep on VIN from
    `vin_min` to `vin_max` — running it as-is (no drive_mode rewrite)
    already produces line-regulation measures. To run a load
    regulation, transient load step, or transient line step, declare
    a `setup` block in the .clay verify and let the verifier rewrite
    the testbench.

    Args:
        cell: the LDO cell IR. Expected ports: inn (Vfb), inp (Vref),
            out (Vreg), vdd, vss. (Matches the AnalogGym ldo_simple
            shape.)
        vin_nom: nominal input voltage in volts.
        vin_min, vin_max, vin_step: DC sweep bounds on Vin.
        vref: reference voltage at `inp`.
        iload_nom: nominal load current in amps.
        iload_min, iload_max, iload_step: DC sweep bounds on Iload.
        CL: output capacitance in farads.

    Returns:
        A Testbench with VIN/VSS/VREF/VFB/ILOAD sources, a CL load
        cap, an OP-point + DC-sweep analysis pair, and a default
        measure list aimed at line regulation. The rewriter swaps
        analyses / measures / sources as the drive mode demands.

    `vin_nom=None` auto-resolves from `cell.pdk_id`.
    """
    if vin_nom is None:
        from ._supply import resolve_supply
        # LDO needs Vin > Vout (regulated). On sky130 1.8 V supply we
        # take the DEFAULT_VIN_NOM=2.0 V (0.2 V dropout headroom over
        # the typical 1.8 V regulated output). On higher-supply PDKs
        # (gf180mcu 3.3 V), we let the supply win — that's already
        # enough headroom for a 1.8 V output and matches the cell's
        # device family.
        resolved = resolve_supply(cell, default=DEFAULT_VIN_NOM)
        vin_nom = max(resolved, DEFAULT_VIN_NOM)
    sources: list[Source] = [
        # VIN: this is the swept source for dc_vin_sweep / tran_line_step.
        # Default DC level is the nominal Vin.
        Source("voltage_dc", "VIN",
               pins={"p": "vdd", "n": "0"},
               params={"dc": vin_nom}),
        Source("voltage_dc", "VSS",
               pins={"p": "vss", "n": "0"},
               params={"dc": 0.0}),
        # VREF: the reference voltage at the error amp's + input.
        Source("voltage_dc", "VREF",
               pins={"p": "inp", "n": "0"},
               params={"dc": vref}),
        # VFB: 0 V source between inn and out. Acts as a wire (unity-
        # gain feedback) AND an ammeter for the feedback current. The
        # 0 V trick is the standard LDO-loop closing pattern — direct
        # wire works too, but the explicit source keeps the netlist
        # readable.
        Source("voltage_dc", "VFB",
               pins={"p": "inn", "n": "out"},
               params={"dc": 0.0}),
        # ILOAD: current sink at the output. Default is the nominal
        # load. Gets rewritten by dc_iload_sweep / tran_load_step.
        Source("current_dc", "ILOAD",
               pins={"p": "out", "n": "vss"},
               params={"dc": iload_nom}),
    ]
    extras: list[Device] = [
        # CL: output bulk capacitance. Necessary for LDO stability
        # and for any transient analysis to settle properly.
        Device("CL", "cap", None,
               pins={"p": "out", "n": "vss"},
               sizing={"value": CL}),
    ]
    # Default analyses: OP point + DC sweep of VIN (line regulation
    # is the "default question" an LDO testbench answers).
    #
    # ngspice convention: voltage sources are emitted with a `V` prefix
    # (the IR's `VIN` becomes `VVIN` in the netlist), so the DC sweep
    # must reference `VVIN`, not `VIN`. Same for current sources (`I`
    # prefix → `IILOAD` for an IR-named `ILOAD`). The rewriter respects
    # this convention too.
    analyses: list = [
        OpAnalysis(),
        DcAnalysis(sweep_source="VVIN",
                   start=vin_min, stop=vin_max, step=vin_step),
    ]
    # Default measures: line regulation suite — vout's max / min /
    # avg / pp over the Vin sweep range, plus a dropout estimate.
    # The rewriter swaps these out for load regulation when
    # `dc_iload_sweep` fires; for transient drives the measures
    # become time-domain settling / overshoot.
    measures: list[Measure] = _default_line_reg_measures(vin_min, vin_max)
    return Testbench(
        cell=cell,
        sources=sources,
        extra_devices=extras,
        analyses=analyses,
        measures=measures,
        pdk_corner="tt",
        supply_volt=vin_nom,
    )


def _default_line_reg_measures(vin_min: float, vin_max: float) -> list[Measure]:
    """Standard line-regulation measure set for the LDO DC-sweep rig.

    Emits five `.meas` lines (max / min / avg / pp of vout, plus a
    `find` at vin_min for the dropout-voltage proxy) and one
    derived `line_regulation` placeholder. The placeholder is a
    `find v(out) at=vin_min` (deterministic ngspice .meas) that the
    verifier post-processor replaces with `vout_pp / (vin_max -
    vin_min)` before any assertion fires. Both forms are emitted
    so the caller can choose: assert against vout_pp directly
    (deterministic measure) or assert against line_regulation
    (post-processed derivation).
    """
    return [
        # MAX / MIN / AVG / PP of vout over the Vin sweep range.
        # `over dc_sweep from <vin_min> to <vin_max>` would be the
        # neutral way to scope; the ngspice emitter applies the
        # FROM=/TO= to the sweep variable's natural axis (dc_sweep
        # for DC analyses).
        Measure("vout_max", "dc",
                f"MAX(v(out)) over dc_sweep from {vin_min:g} to {vin_max:g}"),
        Measure("vout_min", "dc",
                f"MIN(v(out)) over dc_sweep from {vin_min:g} to {vin_max:g}"),
        Measure("vout_avg", "dc",
                f"AVG(v(out)) over dc_sweep from {vin_min:g} to {vin_max:g}"),
        Measure("vout_pp", "dc",
                f"PP(v(out)) over dc_sweep from {vin_min:g} to {vin_max:g}"),
        # Dropout proxy: Vin at the bottom of the sweep minus Vout
        # at that same point. ngspice reports this through two
        # successive finds; we emit the Vout-at-vin_min one here
        # and let derived expressions compute Vin - Vout if needed.
        Measure("vout_at_vmin", "dc",
                f"v(out) @ dc_sweep={vin_min:g}"),
        # line_regulation placeholder. ngspice emits a bare
        # `find v(out) at=<vin_min>`; the verifier post-processor
        # (`_post_process_ldo_regulation`) overrides this value
        # with the proper `vout_pp / (vin_max - vin_min)` ratio.
        # The expression's `over dc_sweep from ... to ...` clause
        # encodes the sweep range so the post-processor can recover
        # it without a separate metadata channel.
        Measure("line_regulation", "dc",
                f"v(out) @ dc_sweep={vin_min:g}"),
    ]


def _default_load_reg_measures(iload_min: float, iload_max: float) -> list[Measure]:
    """Standard load-regulation measure set for the Iload-sweep rig.

    Same shape as `_default_line_reg_measures` but the sweep axis is
    the load current. `load_regulation = pp(vout) / (iload_max -
    iload_min)`, in V/A.
    """
    return [
        Measure("vout_max", "dc",
                f"MAX(v(out)) over dc_sweep from {iload_min:g} to {iload_max:g}"),
        Measure("vout_min", "dc",
                f"MIN(v(out)) over dc_sweep from {iload_min:g} to {iload_max:g}"),
        Measure("vout_avg", "dc",
                f"AVG(v(out)) over dc_sweep from {iload_min:g} to {iload_max:g}"),
        Measure("vout_pp", "dc",
                f"PP(v(out)) over dc_sweep from {iload_min:g} to {iload_max:g}"),
        Measure("vout_at_imin", "dc",
                f"v(out) @ dc_sweep={iload_min:g}"),
    ]


def _default_tran_measures(t_end: float) -> list[Measure]:
    """Standard transient-step measure set.

    Captures vout's MAX / MIN / final value across the transient
    window. The rewriter that installs a transient drive mode is
    responsible for setting up an Iload or Vin source that actually
    steps — the measures here just read the response.
    """
    return [
        # Across the full transient window.
        Measure("vout_max", "tran",
                f"MAX(v(out)) over time from 0 to {t_end:g}"),
        Measure("vout_min", "tran",
                f"MIN(v(out)) over time from 0 to {t_end:g}"),
        # Final-value sample at t_end (settled response).
        Measure("vout_final", "tran",
                f"v(out) @ time={t_end:g}"),
        # Initial-value sample at t=0 (pre-step baseline).
        Measure("vout_initial", "tran",
                f"v(out) @ time=0"),
    ]
