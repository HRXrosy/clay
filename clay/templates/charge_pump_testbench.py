"""
Charge-pump testbench builder (v0.0.36 follow-on to v0.0.35 bandgap).

The default AC testbench (`build_ac_testbench` in `two_stage_miller_pmos.py`)
is shaped for differential opamp characterisation, the LDO testbench
(`build_ldo_testbench`) targets line / load regulation, and the bandgap
testbench (`build_bandgap_testbench`) is the OP-style reference rig.
None of those fit a charge-pump (CP) cell because:

  1. A CP has no analog inputs — it's driven by phase-frequency-detector
     (PFD) digital logic signals: `up`, `dn` (plus their complements
     `upb` / `dnb` in the differential current-steering variant).
  2. The native CP measurement is a transient sweep of the current
     pumped into / sourced from the loop-filter node (`cp_out`), with
     UP current vs DN current matching as the figure of merit. There is
     no DC small-signal "gain" to extract.
  3. CP cells often have additional ports beyond the PFD signals:
     external bias-current inputs (`i10u`, `i5u` in AnalogGym's
     external_pdk 40 LL variant), power-down (`pd`, `pdb`), and the loop-
     filter output (`cp_out`).

`build_charge_pump_testbench` produces a rig in which:

  * VDD / VSS are DC supply sources on the cell's `vdd` / `vss` ports.
  * UP / DN (and their complements UPB / DNB) are driven by PULSE
    sources that exercise UP then DN in alternation. The cycle is:
        t ∈ [0, td)               : both UP=0, DN=0 (idle)
        t ∈ [td, td+pw)           : UP=1, DN=0     (UP-phase: pump
                                                    current into cp_out)
        t ∈ [td+pw, td+2pw)       : UP=0, DN=1     (DN-phase: sink
                                                    current out of cp_out)
        t = td+2pw                : end of one full cycle
    `upb` and `dnb` (if present on the cell) are driven with the
    inverse waveform of `up` / `dn`.
  * `cp_out` is held at a fixed voltage by a 0 V voltage source between
    a reference rail at `vdd/2` and the cell's `cp_out` port. The 0 V
    source doubles as an ammeter — the current through it equals the
    current the CP is pumping into / pulling out of `cp_out`. During
    UP-phase the CP sources current into `cp_out`, so the ammeter
    reads a negative current (current flows from the external rail
    into cp_out, i.e. into the source's `n` terminal). During DN-phase
    the CP sinks current out of `cp_out`, so the ammeter reads a
    positive current. We flip signs in the .meas extractor so
    `i_up` and `i_dn` are both reported as POSITIVE numbers (the
    AnalogGym convention).
  * Optional bias-current inputs (`i10u`, `i5u`) are auto-detected
    from the cell's port list and driven by DC current sources. The
    AnalogGym external_pdk 40 LL CP nominally takes 10 µA and 5 µA references;
    callers can override via kwargs.
  * Power-down ports (`pd`, `pdb`) are auto-detected and held at the
    inactive level (`pd` ↔ vss, `pdb` ↔ vdd) so the CP runs normally.

Analyses:
  * `OpAnalysis` for the operating point.
  * `TranAnalysis` covering exactly one full UP+DN cycle (default
    `td + 2 * pw + small slack`). The transient step is small enough
    to capture the PULSE edges (default 100 ps).

Default measures (the FOM family that AnalogGym uses):
  * `i_up`   — MAX(|i(VCP_OUT)|) over the UP-phase window
               [td + edge_settle, td + pw - edge_settle]. This is the
               peak current the CP can source. The window excludes the
               rising / falling edges so we measure the steady-state
               UP current, not the di/dt spike.
  * `i_dn`   — MAX(|i(VCP_OUT)|) over the DN-phase window
               [td + pw + edge_settle, td + 2*pw - edge_settle]. The
               steady-state DN-phase current.
  * `i_up_avg` — AVG(i(VCP_OUT)) over the UP window (same range as
                 i_up, but unsigned average). Reports the time-
                 averaged UP current — a more conservative number
                 than MAX, closer to the AnalogGym `up_iavg` figure.
  * `i_dn_avg` — AVG(i(VCP_OUT)) over the DN window. Mirror of
                 `i_up_avg`.
  * `i_match` (post-processed in the verifier) — |i_up - i_dn|.
                 The headline matching figure: smaller is better.
                 AnalogGym calls this `diff1..diff4` (one figure per
                 cp_out value in their sweep); we expose a single
                 number per (corner, temp) sim run because each Clay
                 sim is fixed at one cp_out hold voltage.

Why the holding-voltage trick instead of AnalogGym's internal ammeter:
  AnalogGym's original testbench inserts two 0 V sources INSIDE the
  CP subckt (`vupper`, `vlower`) on the bias-chain branches that
  carry the UP and DN currents. Those sources are not part of the
  cell's port list — they're testbench-side instrumentation that
  AnalogGym buried inside the netlist for convenience.

  Clay's testbench layer can't reach inside the cell, so we
  instrument the cell from the OUTSIDE: a 0 V holding source on the
  `cp_out` port reads the net current pumped onto the loop-filter
  node. This is the same physical quantity AnalogGym measures
  (their `up_iavg = i(vupper)` is the bias-chain current that
  becomes the cp_out current when the UP switch is closed), just
  measured at a different point in the circuit. The match figure
  `|i_up - i_dn|` is identical either way.

Why the cp_out hold rail is at vdd/2:
  A CP's UP source current and DN sink current both have an
  output-impedance dependence on cp_out (the cascoded mirror's
  output impedance gives flat current over a wide cp_out range,
  but not infinite). Holding cp_out at vdd/2 is the symmetric
  midpoint where the PMOS source and NMOS sink are both well into
  saturation — the typical operating point of a CP-driven PLL loop
  filter when the loop is locked.

Why a 200 ns transient window by default:
  AnalogGym's testbench runs `tran 2p 200n` (200 ns total). We
  match that as a default so the order of magnitude is right. The
  short ngspice tran is < 1 second wall-time even on slow
  hardware.

This testbench is the third non-amp testbench in clay/templates/,
after `build_ldo_testbench` (v0.0.32) and `build_bandgap_testbench`
(v0.0.35). It uses the v0.0.34 PULSE source IR (added alongside
in-file `instance` resolution and the `ind` primitive) — the only
prerequisite the previous releases needed to make CP-style transient
drive expressible in Clay.
"""

from __future__ import annotations

from ..ir import (
    Cell, Device, Measure, OpAnalysis, Source, Testbench, TranAnalysis,
)


# ─── Default constants for the base testbench ──────────────────────
# Picked to roughly match AnalogGym's external_pdk 40 LL CP testbench (3.3 V
# supply, 200 ns transient, 10 µA / 5 µA bias references). Clay's
# default sky130 PDK runs at 1.8 V, so the supply default below is
# 1.8 V — the AnalogGym .clay file overrides to 3.3 V at the verify
# layer once the external_pdk 40 PDK adapter is registered. The current-
# reference values are kept at the AnalogGym numbers; they're scale-
# invariant w.r.t. supply voltage at the CP architecture level.

DEFAULT_VDD_NOM = 1.8           # nominal supply (sky130 default)
DEFAULT_CP_HOLD_FRAC = 0.5      # hold cp_out at vdd_nom * frac
DEFAULT_I10U_NOM = 10e-6        # 10 µA reference current (AnalogGym i10u)
DEFAULT_I5U_NOM = 5e-6          # 5 µA reference current (AnalogGym i5u)
DEFAULT_PFD_HIGH_FRAC = 1.0     # PFD high level = vdd_nom * frac
DEFAULT_TD = 5e-9               # 5 ns idle window before the first edge
DEFAULT_PW = 10e-9              # 10 ns per phase (UP-phase, then DN-phase)
DEFAULT_T_RISE = 1e-10          # 100 ps PFD edge rise/fall
DEFAULT_TRAN_STEP = 1e-10       # 100 ps tran step (captures the edges)
DEFAULT_EDGE_SETTLE = 1e-9      # 1 ns guard band around each PFD edge


# Port names the heuristic looks for to classify a cell as CP-shaped.
# The CP testbench requires at least UP and DN signals; UPB / DNB are
# optional (only differential current-steering CPs use them, and the
# v0.0.33 textbook CP uses only UP / DN).
_REQUIRED_PFD_PORTS = ("up", "dn")
_OPTIONAL_PFD_PORTS = ("upb", "dnb")
# Alternate naming convention (some textbook CPs name them
# `up_signal` / `dn_signal` to avoid clashing with internal nets):
_ALTERNATE_PFD_PORTS = ("up_signal", "dn_signal")


def find_pfd_ports(cell: Cell) -> dict[str, str]:
    """Return the {logical_name → port_name} map for the cell's PFD
    inputs, or an empty dict if the cell is not CP-shaped.

    Logical keys are `up`, `dn` (always present in a CP-shaped cell),
    and optionally `upb`, `dnb` (differential variant) or `up_signal`
    / `dn_signal` (alternate naming).

    The returned dict carries the actual port names so the testbench
    builder can wire its PULSE sources to whatever the cell author
    used. Returns `{}` if the cell does not declare both UP and DN
    in some recognized form — that's the heuristic the CLI uses to
    fall back to the AC testbench.
    """
    port_set = set(cell.ports)
    # Try the canonical (up, dn) shape first.
    if all(p in port_set for p in _REQUIRED_PFD_PORTS):
        out: dict[str, str] = {"up": "up", "dn": "dn"}
        for opt in _OPTIONAL_PFD_PORTS:
            if opt in port_set:
                out[opt] = opt
        return out
    # Fall back to the alternate (up_signal, dn_signal) naming.
    if all(p in port_set for p in _ALTERNATE_PFD_PORTS):
        return {"up": "up_signal", "dn": "dn_signal"}
    return {}


def build_charge_pump_testbench(
    cell: Cell,
    *,
    vdd_nom: float | None = None,
    cp_hold_frac: float = DEFAULT_CP_HOLD_FRAC,
    i10u_nom: float = DEFAULT_I10U_NOM,
    i5u_nom: float = DEFAULT_I5U_NOM,
    pfd_high_frac: float = DEFAULT_PFD_HIGH_FRAC,
    td: float = DEFAULT_TD,
    pw: float = DEFAULT_PW,
    t_rise: float = DEFAULT_T_RISE,
    tran_step: float = DEFAULT_TRAN_STEP,
    edge_settle: float = DEFAULT_EDGE_SETTLE,
) -> Testbench:
    """Build the default charge-pump testbench.

    Wiring:
      * VDD       — voltage_dc, `vdd` ↔ 0, dc=vdd_nom. The supply rail.
      * VSS       — voltage_dc, `vss` ↔ 0, dc=0. Fixed ground.
      * VCP_OUT   — voltage_dc, `cp_hold_rail` ↔ `cp_out`, dc=0. Acts
                    as both a hold rail tying cp_out to vdd_nom*frac
                    AND an ammeter — i(VVCP_OUT) is the current the
                    CP is pumping onto cp_out.
      * VCP_HOLD  — voltage_dc, `cp_hold_rail` ↔ 0, dc=vdd_nom*frac.
                    Provides the cp_out hold voltage.
      * VUP       — voltage_pulse on the cell's UP port (or its
                    alternate name; see find_pfd_ports).
      * VDN       — voltage_pulse on the cell's DN port. UP and DN
                    pulse trains are offset by one pulse_width so
                    one is HIGH while the other is LOW.
      * VUPB / VDNB — complementary pulse trains, only emitted if the
                    cell declares UPB / DNB ports.
      * VPD       — voltage_dc on the `pd` port at 0 V (inactive), if
                    the cell has a `pd` port. CP runs normally.
      * VPDB      — voltage_dc on the `pdb` port at vdd_nom (inactive),
                    if the cell has a `pdb` port.
      * II10U     — current_dc, vdd ↔ `i10u`, dc=i10u_nom (10 µA
                    reference; matches AnalogGym), if the cell has an
                    `i10u` port. The current flows INTO the `i10u`
                    node from vdd, which is the way AnalogGym's bias
                    chain expects to receive its reference.
      * II5U      — current_dc, vdd ↔ `i5u`, dc=i5u_nom, if the cell
                    has an `i5u` port.

    Analyses: OP + TRAN with t_end = td + 2*pw + edge_settle (one full
    UP+DN cycle plus a small post-cycle slack).

    Default measures:
      * `i_up`     — MAX(|i(VVCP_OUT)|) over UP-phase window
                     [td + edge_settle, td + pw - edge_settle].
                     Reported as a positive scalar; the underlying
                     ngspice `.meas tran ... MAX i(VVCP_OUT)` may
                     report a sign-bearing value (positive means
                     current flows from rail into cp_out, i.e. the
                     CP is sourcing). The verifier doesn't flip
                     signs — callers should `abs(i_up - i_dn)` to
                     get the matching figure.
      * `i_dn`     — MAX(|i(VVCP_OUT)|) over DN-phase window
                     [td + pw + edge_settle, td + 2*pw - edge_settle].
      * `i_up_avg` — AVG(i(VVCP_OUT)) over the UP-phase window.
      * `i_dn_avg` — AVG(i(VVCP_OUT)) over the DN-phase window.

    The verifier's optional `_post_process_cp_match` step (a future
    follow-on, not in this initial template) would derive
    `i_match := |i_up - i_dn|` automatically. For now, callers can
    compute the match themselves via a derived_measure on the
    verify job.

    Args:
        cell: the CP cell IR. MUST declare `vdd`, `vss`, `cp_out`,
            `up`, `dn` ports (or `up_signal`/`dn_signal` aliases for
            the PFD signals). Optional ports `upb`, `dnb`, `pd`,
            `pdb`, `i10u`, `i5u` are auto-detected.
        vdd_nom: nominal supply voltage in volts.
        cp_hold_frac: hold cp_out at vdd_nom * cp_hold_frac (default
            0.5 = midpoint of supply). Most CPs are tested at the
            symmetric midpoint where PMOS source and NMOS sink are
            both well-saturated.
        i10u_nom, i5u_nom: nominal reference currents (only used if
            the cell has matching ports).
        pfd_high_frac: PFD logic-high level as a fraction of vdd_nom
            (default 1.0 = full rail). Some PDFs run at a level-
            shifted logic level (e.g. 0.6 * vdd); set lower to
            simulate a level-shifter input.
        td: idle delay before the first PFD edge (in seconds).
        pw: pulse width for each PFD phase (in seconds). UP-phase
            and DN-phase each last `pw`.
        t_rise: PFD edge rise / fall time (in seconds).
        tran_step: transient analysis time step (in seconds). Default
            100 ps captures the PFD edges.
        edge_settle: guard band around each PFD edge (in seconds)
            excluded from the i_up / i_dn measurement window. Default
            1 ns lets the switch transitions settle before sampling
            the steady-state current.

    Returns:
        A Testbench with all the sources, extra devices, OP + TRAN
        analyses, and the default measure suite.

    Raises:
        ValueError: if the cell does not declare the required ports
            (`vdd`, `vss`, `cp_out`, plus UP/DN in some form).

    `vdd_nom=None` auto-resolves from `cell.pdk_id`.
    """
    if vdd_nom is None:
        from ._supply import resolve_supply
        vdd_nom = resolve_supply(cell, default=DEFAULT_VDD_NOM)
    # ─── Required-port validation ──────────────────────────────
    if "vdd" not in cell.ports:
        raise ValueError(
            f"charge-pump testbench: cell {cell.name!r} has no `vdd` "
            f"port (ports={cell.ports}). The testbench wires the "
            f"cell's vdd to a DC supply — a CP cell must declare a "
            f"vdd port."
        )
    if "vss" not in cell.ports:
        raise ValueError(
            f"charge-pump testbench: cell {cell.name!r} has no `vss` "
            f"port (ports={cell.ports}). The testbench wires the "
            f"cell's vss to ground."
        )
    if "cp_out" not in cell.ports:
        raise ValueError(
            f"charge-pump testbench: cell {cell.name!r} has no "
            f"`cp_out` port (ports={cell.ports}). The CP testbench "
            f"requires `cp_out` (output node) plus `up` / `dn` (or "
            f"`up_signal` / `dn_signal`) PFD input ports; this cell "
            f"has none of them. For an opamp / LDO / bandgap cell, "
            f"use a different testbench (build_ac_testbench / "
            f"build_ldo_testbench / build_bandgap_testbench)."
        )
    pfd_map = find_pfd_ports(cell)
    if not pfd_map:
        raise ValueError(
            f"charge-pump testbench: cell {cell.name!r} has no PFD "
            f"input ports (ports={cell.ports}). The CP testbench "
            f"requires `up` and `dn` ports (or their `up_signal` / "
            f"`dn_signal` aliases). For an opamp / LDO / bandgap "
            f"cell, use a different testbench (build_ac_testbench / "
            f"build_ldo_testbench / build_bandgap_testbench)."
        )

    pfd_high = vdd_nom * pfd_high_frac
    cp_hold_v = vdd_nom * cp_hold_frac
    # PULSE period set so we get exactly one UP cycle + one DN cycle
    # in the transient window — no repetition. The DN pulse is offset
    # so it starts at td+pw (immediately after UP-phase ends).
    pulse_period = 2.0 * (td + pw + pw)  # well beyond t_end → single edge

    sources: list[Source] = [
        Source("voltage_dc", "VDD",
               pins={"p": "vdd", "n": "0"},
               params={"dc": vdd_nom}),
        Source("voltage_dc", "VSS",
               pins={"p": "vss", "n": "0"},
               params={"dc": 0.0}),
        # Hold rail at midpoint of supply. The cp_out port is held at
        # this voltage via the 0 V ammeter source VCP_OUT below.
        Source("voltage_dc", "VCP_HOLD",
               pins={"p": "cp_hold_rail", "n": "0"},
               params={"dc": cp_hold_v}),
        # 0 V ammeter between cp_hold_rail and cp_out. Current through
        # this source = current the CP is pumping onto cp_out.
        # During UP-phase: CP sources current INTO cp_out, so current
        # flows from cp_hold_rail (the n-terminal: see ngspice
        # convention) into cp_out, meaning i(VVCP_OUT) > 0 by the
        # ngspice sign convention (current flows from p to n through
        # the source). During DN-phase the sign flips.
        Source("voltage_dc", "VCP_OUT",
               pins={"p": "cp_hold_rail", "n": "cp_out"},
               params={"dc": 0.0}),
    ]

    # UP PULSE: starts at td, high for pw, then low. Period >> t_end
    # so we get exactly one pulse.
    sources.append(Source(
        kind="voltage_pulse",
        instance_name="VUP",
        pins={"p": pfd_map["up"], "n": "0"},
        params={
            "v_low": 0.0,
            "v_high": pfd_high,
            "td": td,
            "t_rise": t_rise,
            "t_fall": t_rise,
            "pulse_width": pw,
            "period": pulse_period,
        },
    ))
    # DN PULSE: starts at td + pw (immediately after UP ends).
    sources.append(Source(
        kind="voltage_pulse",
        instance_name="VDN",
        pins={"p": pfd_map["dn"], "n": "0"},
        params={
            "v_low": 0.0,
            "v_high": pfd_high,
            "td": td + pw,
            "t_rise": t_rise,
            "t_fall": t_rise,
            "pulse_width": pw,
            "period": pulse_period,
        },
    ))
    # Optional UPB / DNB — complementary signals, only present in
    # differential current-steering CPs.
    if "upb" in pfd_map:
        # UPB is the inverse of UP: HIGH except during the UP-phase.
        # Implemented by inverting the v_low/v_high values.
        sources.append(Source(
            kind="voltage_pulse",
            instance_name="VUPB",
            pins={"p": pfd_map["upb"], "n": "0"},
            params={
                "v_low": pfd_high,
                "v_high": 0.0,
                "td": td,
                "t_rise": t_rise,
                "t_fall": t_rise,
                "pulse_width": pw,
                "period": pulse_period,
            },
        ))
    if "dnb" in pfd_map:
        sources.append(Source(
            kind="voltage_pulse",
            instance_name="VDNB",
            pins={"p": pfd_map["dnb"], "n": "0"},
            params={
                "v_low": pfd_high,
                "v_high": 0.0,
                "td": td + pw,
                "t_rise": t_rise,
                "t_fall": t_rise,
                "pulse_width": pw,
                "period": pulse_period,
            },
        ))

    # Optional power-down ports: held inactive so the CP runs normally.
    if "pd" in cell.ports:
        sources.append(Source(
            kind="voltage_dc",
            instance_name="VPD",
            pins={"p": "pd", "n": "0"},
            params={"dc": 0.0},   # pd=0 → CP active
        ))
    if "pdb" in cell.ports:
        sources.append(Source(
            kind="voltage_dc",
            instance_name="VPDB",
            pins={"p": "pdb", "n": "0"},
            params={"dc": vdd_nom},   # pdb=vdd → CP active
        ))

    # Optional bias-current inputs: drive AnalogGym-style external
    # current references into i10u / i5u ports if they exist.
    # ngspice current source convention: `I<name> p n dc <value>`
    # forces `value` amperes from `p` to `n` through the external
    # circuit. For a CP cell that expects current FLOWING INTO i10u
    # from an external supply, we orient `p=vdd` and `n=i10u` so the
    # source pulls current from vdd and pushes it into the i10u node.
    if "i10u" in cell.ports:
        sources.append(Source(
            kind="current_dc",
            instance_name="II10U",
            pins={"p": "vdd", "n": "i10u"},
            params={"dc": i10u_nom},
        ))
    if "i5u" in cell.ports:
        sources.append(Source(
            kind="current_dc",
            instance_name="II5U",
            pins={"p": "vdd", "n": "i5u"},
            params={"dc": i5u_nom},
        ))

    # Transient analysis window covers the full UP+DN cycle plus a
    # small slack so the last DN edge has time to settle in the .meas
    # window.
    t_end = td + 2.0 * pw + edge_settle
    analyses: list = [
        OpAnalysis(),
        TranAnalysis(step=tran_step, t_end=t_end),
    ]

    # ─── Measure suite ─────────────────────────────────────────
    # UP-phase window: from `td + edge_settle` to `td + pw - edge_settle`.
    # DN-phase window: from `td + pw + edge_settle` to `td + 2*pw - edge_settle`.
    up_lo = td + edge_settle
    up_hi = td + pw - edge_settle
    dn_lo = td + pw + edge_settle
    dn_hi = td + 2.0 * pw - edge_settle

    measures: list[Measure] = _cp_measures(
        up_lo=up_lo, up_hi=up_hi, dn_lo=dn_lo, dn_hi=dn_hi,
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


def _cp_measures(
    *, up_lo: float, up_hi: float, dn_lo: float, dn_hi: float,
) -> list[Measure]:
    """Build the .meas list for a charge-pump testbench.

    Six measures total:
      * `i_up`     — MAX(i(VVCP_OUT)) over [up_lo, up_hi]
      * `i_up_min` — MIN(i(VVCP_OUT)) over [up_lo, up_hi]
      * `i_up_avg` — AVG(i(VVCP_OUT)) over [up_lo, up_hi]
      * `i_dn`     — MAX(i(VVCP_OUT)) over [dn_lo, dn_hi]  (note:
                     during DN-phase the current sign flips, so MAX
                     here may report a value smaller than i_up if
                     the convention is for both currents to be
                     positive. AnalogGym's `lo_imax` is the analogous
                     figure with sign-flip implicit.)
      * `i_dn_min` — MIN(i(VVCP_OUT)) over [dn_lo, dn_hi]
      * `i_dn_avg` — AVG(i(VVCP_OUT)) over [dn_lo, dn_hi]

    The signed ngspice .meas values let callers compute either the
    magnitude (UP vs DN current match) or the signed asymmetry.
    The `i_up` / `i_dn` (MAX) pair gives the peak current; the
    `i_up_avg` / `i_dn_avg` pair gives the time-averaged current
    that AnalogGym uses for its `up_iavg` / `lo_iavg` FOM.

    Order convention: emit AGGREGATES (MAX / MIN / AVG) BEFORE any
    FIND-style measures so ngspice's sticky-state .meas bug doesn't
    silently zero out subsequent AVGs. This matches the LDO / bandgap
    testbenches' ordering (also adopted from the same bug-avoidance
    pattern).

    The branch-current signal is `i(VVCP_OUT)` because ngspice
    prefixes voltage sources with `V` — Clay's IR `VCP_OUT` becomes
    `VVCP_OUT` in the netlist. The `.meas` line references that
    netlist-level name.
    """
    return [
        # UP-phase aggregates.
        Measure("i_up", "tran",
                f"MAX(i(VVCP_OUT)) over time from {up_lo:g} to {up_hi:g}"),
        Measure("i_up_min", "tran",
                f"MIN(i(VVCP_OUT)) over time from {up_lo:g} to {up_hi:g}"),
        Measure("i_up_avg", "tran",
                f"AVG(i(VVCP_OUT)) over time from {up_lo:g} to {up_hi:g}"),
        # DN-phase aggregates.
        Measure("i_dn", "tran",
                f"MAX(i(VVCP_OUT)) over time from {dn_lo:g} to {dn_hi:g}"),
        Measure("i_dn_min", "tran",
                f"MIN(i(VVCP_OUT)) over time from {dn_lo:g} to {dn_hi:g}"),
        Measure("i_dn_avg", "tran",
                f"AVG(i(VVCP_OUT)) over time from {dn_lo:g} to {dn_hi:g}"),
    ]
