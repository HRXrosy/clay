"""Testbench-builder cluster split out of ``clay/_ir_bridge.py``.

This module hosts the family of ``_build_*_testbench`` functions that wrap
a Rust-emitted ``.subckt`` block in a complete ngspice testbench deck. Each
builder is a thin orchestrator over the corresponding
``clay_ir._build_*_testbench_rust`` delegate; the Python side only marshals
``functional_roles`` and converts the corner-file ``Path`` to a POSIX string.

Why these live in a focused module while the dynamic extension state stays
in ``_ir_bridge``:

  * The Rust-extension globals (``_HAS_CLAY_IR`` / ``_clay_ir``) and the
    ``_require_clay_ir`` / ``_roles_to_rust_payload`` helpers remain
    canonical in ``clay._ir_bridge``. Test fixtures monkeypatch
    ``clay._ir_bridge._HAS_CLAY_IR`` / ``._clay_ir`` to exercise the
    hard-error and dispatch branches, so every builder here resolves that
    state through the ``_ir_bridge`` module object at call time
    (``_ir_bridge._require_clay_ir()`` etc.) rather than capturing a stale
    binding at import time. This keeps the patch semantics byte-identical.

  * ``clay._ir_bridge`` re-exports every name defined here at the spot the
    functions used to be defined, so all existing
    ``from clay._ir_bridge import _build_*`` import sites keep working
    unchanged.

The ``import clay._ir_bridge`` at module scope is a module reference only —
no attribute is read during import, so the circular relationship
(``_ir_bridge`` imports this module to re-export, this module imports
``_ir_bridge`` for shared state) resolves cleanly: the partially
initialized module object is sufficient because attribute access is
deferred to call time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from . import _ir_bridge
from .ir import ParsedFunctionalRole


def _build_ac_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    input_node: str,
    output_node: str,
    supply_v: float,
    cload: float = 1e-12,
    f_start: float = 1.0,
    f_stop: float = 1e9,
    points_per_decade: int = 10,
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Wrap a Rust-emitted `.subckt` block in a minimal AC testbench.

    The DUT is driven from a DC + AC voltage source on `input_node`,
    with a `cload` cap on `output_node`. The .meas lines extract the
    DC magnitude (`gain_dc` = vdb(out) at 1 Hz) and the unity-gain
    frequency (`gbw` = first frequency where vdb crosses 0 on the
    falling edge). These are the two measures the MDP reward needs
    today; richer transient / phase-margin measures are deferred to
    v0.0.74+.

    Supply rails: `vdd`/`vss` ports on the subckt are wired to the
    testbench's `Vdd` (= supply_v) and `Vss` (= 0). Input bias is
    set to mid-rail (`supply_v/2`) so the CS-amp's M1 sits in
    saturation under the default sky130/gf180 device sizing.

    See also :func:`_build_ac_pm_testbench` (v0.0.74) which extends
    this block with a phase-margin .meas + degrees-mode AC scan.

    v0.0.98 R75 B: when `functional_roles` is None or empty AND the
    `clay_ir._build_ac_testbench_rust` delegate is available, dispatch
    to the Rust emitter.

    v0.1.11 R87 D: the Rust delegate now accepts `functional_roles` as
    an optional payload (matching the R85 D inject-helper tuple shape)
    and internally calls `emit_functional_role_op_lines` +
    `emit_v92_functional_role_lines` so the v0.0.87 / v0.0.92 inject
    layers are byte-equal in Rust. The Python fallback that re-emitted
    the deck is gone — this function unconditionally delegates to
    `clay_ir._build_ac_testbench_rust`, passing the marshalled roles
    payload when non-empty.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. Test
    fixtures that previously patched the predicate alone to exercise
    the Python emission path still work because the extension is loaded
    and the predicate, not `_HAS_CLAY_IR`, controls the Rust dispatch
    branch.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_ac_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        input_node=input_node,
        output_node=output_node,
        supply_v=supply_v,
        cload=cload,
        f_start=f_start,
        f_stop=f_stop,
        points_per_decade=points_per_decade,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


# ─── v0.0.74: extended testbench builders ─────────────────────────


def _build_ac_pm_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    input_node: str,
    output_node: str,
    supply_v: float,
    cload: float = 1e-12,
    f_start: float = 1.0,
    f_stop: float = 1e9,
    points_per_decade: int = 10,
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Same AC testbench as :func:`_build_ac_testbench`, plus a phase-
    margin measure.

    Adds two extra ngspice control-deck commands:

      * ``set units=degrees`` — flips ``vp()`` from radians (the
        ngspice default on most builds) to degrees. Without this the
        phase numbers come back as ~π at DC instead of ~180 and the
        downstream `180 - |phase|` math is unit-confused.
      * ``meas ac phase_deg find vp(out) when vdb(out)=0 fall=1`` —
        picks the phase at the unity-gain frequency (the same crossing
        the AC test bench uses for ``gbw``).

    Then computes ``pm = 180 - abs(phase_deg)`` via a `let` statement
    so the returned measure name is just ``pm`` (consistent with the
    rest of the bridge's metric naming). For a single-pole CS-amp the
    raw phase at GBW is in the 130–160° range, yielding a 20–50° PM —
    consistent with the very-low-headroom diode-load topology.

    Returned measures: ``gain_dc``, ``gbw``, ``phase_deg``, ``pm``.

    v0.0.98 R75 B: same Rust-delegate pattern as
    :func:`_build_ac_testbench`. Empty-roles call lands on
    ``clay_ir._build_ac_pm_testbench_rust`` byte-for-byte identical.

    v0.1.11 R87 D: the Rust delegate now accepts `functional_roles` as
    an optional payload and internally calls
    `emit_functional_role_op_lines` + `emit_v92_functional_role_lines`
    so the v0.0.87 / v0.0.92 inject layers are byte-equal in Rust. The
    Python fallback that re-emitted the deck is gone — this function
    unconditionally delegates to `clay_ir._build_ac_pm_testbench_rust`,
    passing the marshalled roles payload when non-empty.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. See
    :func:`_build_ac_testbench` for the soft-fail / predicate-patch
    contract.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_ac_pm_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        input_node=input_node,
        output_node=output_node,
        supply_v=supply_v,
        cload=cload,
        f_start=f_start,
        f_stop=f_stop,
        points_per_decade=points_per_decade,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


def _build_tran_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    input_node: str,
    output_node: str,
    supply_v: float,
    cload: float = 1e-12,
    t_step: float = 1e-10,
    t_stop: float = 1e-7,
    pulse_low_v: Optional[float] = None,
    pulse_high_v: Optional[float] = None,
    pulse_delay: float = 5e-9,
    pulse_rise: float = 1e-10,
    pulse_fall: float = 1e-10,
    pulse_width: float = 2e-8,
    pulse_period: float = 5e-8,
    meas_from: float = 8e-9,
    meas_to: float = 2.5e-8,
    settling_band: float = 0.005,
    settling_avg_from: Optional[float] = None,
    settling_avg_to: Optional[float] = None,
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Build a transient testbench that pulses ``input_node`` and
    measures the slew rate on ``output_node``.

    The input is a small-signal VPULSE centred at mid-rail
    (``supply_v/2``) with a default ±50 mV swing — enough to walk the
    DUT through one rise/fall cycle without leaving the device's
    saturation region. Default sweep: 0.1 ns step, 100 ns stop.

    Slew is captured via the canonical ``max_at`` / ``min_at`` pair:

        meas tran v_max max v(out)   from=… to=…
        meas tran v_min min v(out)   from=… to=…
        meas tran t_max max_at v(out) from=… to=…
        meas tran t_min min_at v(out) from=… to=…
        let slew_rate = (v_max-v_min)/abs(t_max-t_min)

    The ``from``/``to`` window is the half-period that contains one
    full output transition; bounding the window keeps the v_max/v_min
    samples from landing on a steady-state plateau (which would give
    a degenerate ``t_max == t_min`` and a ``/0`` slew rate).

    Settling time (v0.0.78): computed against the late-window output
    average ``v_final`` (the asymptote the output approaches after the
    first rising step). A ±``settling_band`` × 100 % band is drawn
    around ``v_final``; ``settling_time`` is the latest time at which
    ``v(out)`` is touching either band edge, minus the time of the
    pulse rise. Default band: 0.5 % (i.e. ``settling_band=0.005``).

        let v_final = mean(v(out)[N_late:])      ; late-window asymptote
        let v_upper = v_final * (1 + settling_band)
        let v_lower = v_final * (1 - settling_band)
        meas tran t_step_start when v(in)=mid rise=1
        meas tran t_upper_last when v(out)=v_upper cross=last
        meas tran t_lower_last when v(out)=v_lower cross=last
        let settling_time = max(t_upper_last, t_lower_last) - t_step_start

    The ``cross=last`` qualifier picks the LAST crossing of each band
    edge in the transient window — i.e. the latest moment ``v(out)``
    enters the band and stays inside until ``t_stop``. The two band
    edges are tracked separately because the response can over-shoot
    (upper edge crosses last) or under-shoot (lower edge crosses last);
    the max of the two is the conservative settling estimate.

    Returned measures: ``v_max``, ``v_min``, ``t_max``, ``t_min``,
    ``slew_rate`` (V/s), ``v_final`` (V), ``settling_time`` (s).

    Honest deferred work: a clean 10%/90% slew measure requires the
    swing to be known at deck-compile time — ngspice's ``.meas …
    when`` doesn't accept variable expressions. v0.0.74 sticks with
    the (v_max-v_min)/(t_max-t_min) approximation; closer-to-textbook
    slew (10/90 of swing) would need an outer driver that runs the
    sim once for swing detection, then again with hard thresholds.
    The v0.0.78 settling-time measure leans on ``cross=last`` which is
    a load-bearing ngspice 35+ feature; older builds without LAST
    support return NaN through ``_parse_measure_log``.

    v0.0.98 R75 B: same Rust-delegate pattern as
    :func:`_build_ac_testbench`. Empty-roles call lands on
    ``clay_ir._build_tran_testbench_rust`` byte-for-byte identical.

    v0.1.11 R87 D: the Rust delegate now accepts `functional_roles` as
    an optional payload and internally calls
    `emit_functional_role_op_lines` + `emit_v92_functional_role_lines`
    so the v0.0.87 / v0.0.92 inject layers are byte-equal in Rust. The
    Python fallback that re-emitted the deck is gone — this function
    unconditionally delegates to `clay_ir._build_tran_testbench_rust`,
    passing the marshalled roles payload when non-empty.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. See
    :func:`_build_ac_testbench` for the soft-fail / predicate-patch
    contract.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_tran_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        input_node=input_node,
        output_node=output_node,
        supply_v=supply_v,
        cload=cload,
        t_step=t_step,
        t_stop=t_stop,
        pulse_low_v=pulse_low_v,
        pulse_high_v=pulse_high_v,
        pulse_delay=pulse_delay,
        pulse_rise=pulse_rise,
        pulse_fall=pulse_fall,
        pulse_width=pulse_width,
        pulse_period=pulse_period,
        meas_from=meas_from,
        meas_to=meas_to,
        settling_band=settling_band,
        settling_avg_from=settling_avg_from,
        settling_avg_to=settling_avg_to,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


def _build_dc_sweep_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    input_node: str,
    output_node: str,
    supply_v: float,
    cload: float = 1e-12,
    sweep_start: float = 0.0,
    sweep_stop: Optional[float] = None,
    sweep_step: float = 0.01,
    vdd_sweep_frac: float = 0.10,
    vdd_sweep_step: float = 0.01,
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Build a DC sweep testbench for a 4-port single-ended cell.

    Two sequential DC analyses share the same .control block:

      1. **Input sweep** (v0.0.74 behaviour, kept for backward compat).
         Walks ``input_node`` from ``sweep_start`` to ``sweep_stop``
         (defaults to ``supply_v``) at ``sweep_step`` resolution, reads
         V(``output_node``) at the mid-rail point → ``vout_at_mid``.
      2. **Supply sweep** (v0.0.79 PSRR-DC). Walks ``Vdd`` from
         ``supply_v*(1-vdd_sweep_frac)`` to ``supply_v*(1+vdd_sweep_frac)``
         at ``vdd_sweep_step`` resolution while ``input_node`` stays at
         the mid-rail DC bias. The output's response to the supply
         excursion gives the DC supply rejection metric:
         ``psrr_dc = 20*log10((v_out_at_vdd_max - v_out_at_vdd_min) /
         (vdd_max - vdd_min))``.

    Returned measures:

      * ``vout_at_mid`` — V(``output_node``) at V(``input_node``) =
        supply_v / 2 (input sweep result).
      * ``v_out_at_vdd_min`` — V(``output_node``) when Vdd =
        supply_v*(1-vdd_sweep_frac).
      * ``v_out_at_vdd_max`` — V(``output_node``) when Vdd =
        supply_v*(1+vdd_sweep_frac).
      * ``psrr_dc`` — supply-to-output gain in dB.  A well-rejected
        supply lands negative dB; output that tracks the supply 1:1
        lands at ~0 dB.  This is the *linearised secant* PSRR
        (endpoint-secant), preserved as the v0.0.79 contract for
        callers that read it.
      * ``v_out_window_max`` / ``v_out_window_min`` — extrema of
        V(``output_node``) across the supply sweep window
        ``[vdd_lo, vdd_hi]``.
      * ``vdd_at_v_out_max`` / ``vdd_at_v_out_min`` — Vdd values where
        those extrema are attained (via `.meas dc max_at / min_at`).
      * ``psrr_dc_worst`` — worst-case PSRR in dB:
        ``20*log10(|v_out_window_max - v_out_window_min| / Δvdd)``.
        For monotonic V(``output_node``) vs Vdd this collapses to the
        endpoint-secant ``psrr_dc``; for non-monotonic cells it
        dominates the endpoint-secant magnitude and surfaces the
        worst rejection inside the window.  This is still a window-
        secant, not the pointwise derivative ``max|∂V_out/∂Vdd|``;
        an exact pointwise worst-case PSRR would require deriv
        tracking ngspice ``.meas`` does not support out of the box.

    The sweep cap (``cload``) is wired in but has no AC effect at DC;
    it stays for parity with the other testbenches so the netlist
    shape is stable across testbench_kind values.

    v0.0.98 R75 B: same Rust-delegate pattern as the other single-ended
    builders. Empty-roles dispatch to
    ``clay_ir._build_dc_sweep_testbench_rust`` byte-for-byte identical.

    v0.1.11 R87 D: the Rust delegate now accepts `functional_roles` as
    an optional payload and internally calls
    `emit_functional_role_op_lines` + `emit_v92_functional_role_lines`
    so the v0.0.87 / v0.0.92 inject layers are byte-equal in Rust. The
    Python fallback that re-emitted the deck is gone — this function
    unconditionally delegates to `clay_ir._build_dc_sweep_testbench_rust`,
    passing the marshalled roles payload when non-empty.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. See
    :func:`_build_ac_testbench` for the soft-fail / predicate-patch
    contract.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_dc_sweep_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        input_node=input_node,
        output_node=output_node,
        supply_v=supply_v,
        cload=cload,
        sweep_start=sweep_start,
        sweep_stop=sweep_stop,
        sweep_step=sweep_step,
        vdd_sweep_frac=vdd_sweep_frac,
        vdd_sweep_step=vdd_sweep_step,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


def _build_ac_diff_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    supply_v: float,
    cload: float = 1e-12,
    f_start: float = 1.0,
    f_stop: float = 1e9,
    points_per_decade: int = 10,
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Build an AC testbench for a 5-port differential opamp.

    Driving convention (Razavi §9 textbook differential AC sweep):

      * `Vcm` sets the common-mode bias at mid-rail (supply_v / 2).
      * `Vsig` is the single AC source — `dc 0 ac 1`.
      * Two VCVS (E-elements) split Vsig into `+0.5*Vsig` at `inp`
        and `-0.5*Vsig` at `inn`, both centred on `vcm`. Differential
        input swing is therefore `1.0 V` at AC = 1 (matches the
        single-ended testbench's `vdb(out)/vdb(in)` conventions so
        downstream gain_dc/gbw measures stay comparable across
        testbench kinds).
      * Load cap on `out`, AC sweep, same `gain_dc` / `gbw` .meas
        as the single-ended template.

    Port order on `xDUT` is `inn inp out vdd vss` — matches the stdlib
    cell convention (`TwoStageMillerPMOS`, `TwoStageMillerNMOS`,
    `FoldedCascode`, etc. all declare ports in that order).

    v0.1.05 R81 D: same Rust-delegate pattern as the single-ended
    builders. Empty-roles call lands on
    ``clay_ir._build_ac_diff_testbench_rust`` byte-for-byte identical;
    non-empty falls back here so the v0.0.87 / v0.0.92 inject layers
    stay live unchanged.

    v0.1.12 R88 B: the Rust delegate now accepts `functional_roles` as
    an optional payload (matching the R85 D inject-helper tuple shape)
    and internally calls `emit_functional_role_op_lines` +
    `emit_v92_functional_role_lines` so the v0.0.87 / v0.0.92 inject
    layers are byte-equal in Rust. The Python fallback that re-emitted
    the deck is gone — this function unconditionally delegates to
    `clay_ir._build_ac_diff_testbench_rust`, passing the marshalled
    roles payload when non-empty.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. See
    :func:`_build_ac_testbench` for the soft-fail / predicate-patch
    contract.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_ac_diff_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        supply_v=supply_v,
        cload=cload,
        f_start=f_start,
        f_stop=f_stop,
        points_per_decade=points_per_decade,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


def _build_ac_fully_diff_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    supply_v: float,
    cload: float = 1e-12,
    f_start: float = 1.0,
    f_stop: float = 1e9,
    points_per_decade: int = 10,
    r_cmfb_sense: float = 1e6,
    cmfb_ea_gain: float = 1e6,
    cmfb_ea_bw: float = float("inf"),
    cmfb_probe_ac: float = 1e-3,
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Build an AC testbench for a 6-port fully-differential opamp.

    Driving + sensing convention (textbook FD-opamp AC sweep):

      * `Vcm` sets `vcm_target` at mid-rail (supply_v / 2). v0.0.81
        adds a tiny AC perturbation (``cmfb_probe_ac`` V, default
        1 mV) so the CMFB error amp sees a known stimulus — this is
        what `cmfb_phase_margin` measures.
      * `Vsig` is the single AC source — `dc 0 ac 1`.
      * Two VCVS (E1/E2) split Vsig into `+0.5*Vsig` at `inp` and
        `-0.5*Vsig` at `inn`, both centred on `vcm_target`. The diff
        input swing therefore equals `1.0 V` at AC = 1 — same convention
        as the 5-port differential template so `gain_dc` numbers
        are comparable across testbench kinds.
      * `Rcm1`/`Rcm2` are equal sense resistors (1 MΩ default) that
        average `outn`/`outp` into a `vcm_sensed` node. The high
        resistance keeps the sense path from loading the differential
        signal mode.
      * `E_cmfb` is the CMFB error amplifier that drives `vcmfb` from
        `(vcm_target - vcm_sensed)`. Two modes:

          - ``cmfb_ea_bw=inf`` (v0.0.77 default): single VCVS with
            gain ``cmfb_ea_gain``. Behaviour bit-identical to v0.0.77
            / 78 / 79 / 80.
          - ``cmfb_ea_bw`` finite (v0.0.81 finite-gain model): VCVS
            with gain ``cmfb_ea_gain`` drives an internal node
            ``vcmfb_pre``; an explicit RC pole (``R_cmfb_pole`` = 1 Ω
            and ``C_cmfb_pole`` = 1/(2π·bw) F) shapes the EA transfer
            function as H(s) = gain / (1 + s/(2π·bw)). This gives a
            measurable phase margin at the EA's unity-gain crossover.

        With no dedicated CMFB port on the DUT, this regulator acts via
        the resistor-leakage path (sufficient to set the DC CM operating
        point for most stdlib FD topologies).
      * `E_diff` synthesises `diff_out = outp - outn` so the existing
        `vdb(...)` .meas lines keep working without dialect changes.
      * `Cload_p`/`Cload_n` load both outputs symmetrically — the FD
        opamp sees a balanced load (this is the convention even on
        silicon test benches where one output drives ESD probes).

    Port order on `xDUT` is `inn inp outn outp vdd vss` — matches the
    naming convention every detected FD cell follows (and what
    `_detect_port_shape` returns in `input_ports`/`output_ports`).

    v0.0.81 additions:

      * ``cmfb_ea_bw`` — single-pole bandwidth of the CMFB error amp.
        Default ``inf`` preserves the ideal v0.0.77 behaviour bit-
        exactly (no extra nodes, no extra components). A finite value
        introduces an explicit RC pole so the EA's frequency response
        rolls off; this is what makes ``cmfb_phase_margin`` a number
        instead of NaN.
      * ``cmfb_probe_ac`` — AC excitation on ``vcm_target`` (V).
        Default 1 mV; small enough not to disturb the differential
        gain measurement (Vsig AC = 1 swamps it by 60 dB), large
        enough for the .meas line to resolve the EA's transfer
        function. The differential `gain_dc` / `gbw` numbers are
        therefore not affected by this probe to within four decimal
        places.

    Returned measures: ``gain_dc``, ``gbw``, ``cmfb_phase_margin``,
    ``cmfb_cm_offset``.
    With ``cmfb_ea_bw=inf`` (default) ``cmfb_phase_margin`` lands NaN
    because the EA has no unity-gain crossover. Opt in to a finite
    ``cmfb_ea_bw`` to make it real.

    ``cmfb_cm_offset`` is the quantitative CMFB CM-regulation residual
    at the AC sweep's DC point: ``(V(outp) + V(outn))/2 − vcm_target``.
    With the default ideal EA (``cmfb_ea_gain=1e6``) it lands ~0 V;
    finite EA gain leaves a measurable non-zero offset that scales
    roughly as ``vcm_target / cmfb_ea_gain`` for a single-rail-leakage
    loop. Use this to demonstrate that finite EA gain degrades CM
    accuracy quantitatively.

    v0.1.06 R82 C: same Rust-delegate pattern as the single-ended +
    5-port diff builders. Empty-roles call lands on
    ``clay_ir._build_ac_fully_diff_testbench_rust`` byte-for-byte
    identical; non-empty falls back here so the v0.0.87 / v0.0.92
    inject layers stay live unchanged.

    v0.1.12 R88 B: the Rust delegate now accepts `functional_roles` as
    an optional payload (matching the R85 D inject-helper tuple shape)
    and internally calls `emit_functional_role_op_lines` +
    `emit_v92_functional_role_lines` so the v0.0.87 / v0.0.92 inject
    layers are byte-equal in Rust. The Python fallback that re-emitted
    the deck is gone — this function unconditionally delegates to
    `clay_ir._build_ac_fully_diff_testbench_rust`, passing the marshalled
    roles payload when non-empty.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. See
    :func:`_build_ac_testbench` for the soft-fail / predicate-patch
    contract.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_ac_fully_diff_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        supply_v=supply_v,
        cload=cload,
        f_start=f_start,
        f_stop=f_stop,
        points_per_decade=points_per_decade,
        r_cmfb_sense=r_cmfb_sense,
        cmfb_ea_gain=cmfb_ea_gain,
        cmfb_ea_bw=cmfb_ea_bw,
        cmfb_probe_ac=cmfb_probe_ac,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


def _build_ac_fully_diff_cmrr_cm_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    supply_v: float,
    cload: float = 1e-12,
    f_start: float = 1.0,
    f_stop: float = 1e9,
    points_per_decade: int = 10,
    r_cmfb_sense: float = 1e6,
    cmfb_ea_gain: float = 1e6,
    cmfb_ea_bw: float = float("inf"),
    cm_drive_ac: float = 1.0,
) -> str:
    """Build the common-mode-drive leg of the 2-run CMRR sweep for a
    6-port fully-differential opamp.

    Same DUT instantiation + CMFB regulator topology as
    `_build_ac_fully_diff_testbench`, but the differential drive
    (Vsig + E1 + E2) is replaced by a pure common-mode source: both
    inp and inn are tied directly to `vcm_target`, which carries an
    AC amplitude `cm_drive_ac` (default 1 V — matches the diff-leg's
    Vsig AC=1 so |gain_cm| lands on the same dBV scale as |gain_dm|).

    Returned measure: `gain_cm_dc` — the common-mode output amplitude
    in dB at f_start. The orchestrator
    `simulate_fully_diff_cmrr_2run` pairs this with the diff leg's
    `gain_dc` to compute CMRR = 20·log10(gain_dm_linear / gain_cm_linear)
    = gain_dc_dB − gain_cm_dB.

    The CMFB regulator stays in place because removing it would let the
    CM output drift to the rails; v0.0.79's `dc_sweep` CMRR uses the
    same approach (drive vcm, keep CMFB, measure |dV(diff_out)/dV(vcm)|).

    v0.1.07 R83 C: delegates to `clay_ir._build_ac_fully_diff_cmrr_cm_testbench_rust`
    byte-for-byte when the Rust extension is available. The Python
    builder has no `functional_roles` parameter so the delegate is
    unconditional (no inject-layer fallback to preserve).

    v0.1.12 R88 B: strip the Python fallback deck. The Rust delegate
    is now the only path; `_require_clay_ir()` hard-errors when missing
    so the predicate-gate branch is gone too.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. See
    :func:`_build_ac_testbench` for the soft-fail / predicate-patch
    contract.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_ac_fully_diff_cmrr_cm_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        supply_v=supply_v,
        cload=cload,
        f_start=f_start,
        f_stop=f_stop,
        points_per_decade=points_per_decade,
        r_cmfb_sense=r_cmfb_sense,
        cmfb_ea_gain=cmfb_ea_gain,
        cmfb_ea_bw=cmfb_ea_bw,
        cm_drive_ac=cm_drive_ac,
    )


def _build_tran_diff_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    supply_v: float,
    cload: float = 1e-12,
    t_step: float = 1e-10,
    t_stop: float = 1e-7,
    pulse_swing: float = 0.05,
    pulse_delay: float = 5e-9,
    pulse_rise: float = 1e-10,
    pulse_fall: float = 1e-10,
    pulse_width: float = 2e-8,
    pulse_period: float = 5e-8,
    meas_from: float = 8e-9,
    meas_to: float = 2.5e-8,
    settling_band: float = 0.005,
    settling_avg_from: Optional[float] = None,
    settling_avg_to: Optional[float] = None,
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Build a transient testbench for a 5-port differential opamp.

    Driving convention (Razavi §9 textbook differential step response):

      * `Vcm` sets the common-mode bias at mid-rail (supply_v / 2).
      * `Vsig` is a VPULSE source on `sig` swinging
        ``[-pulse_swing, +pulse_swing]`` around 0. Default ±50 mV
        single-ended swing — translates to ±50 mV differential
        excursion at inp/inn after the E1/E2 split.
      * E1/E2 split `sig` into ``+0.5*sig`` at `inp` and ``-0.5*sig``
        at `inn`, both centred on vcm. Differential input excursion
        therefore equals ``1.0 × pulse_swing`` at the peak (matches
        the v0.0.75 AC builder's ±0.5 split convention).
      * Load cap on `out`; same .meas measure set as the single-ended
        builder so the returned dict shape is invariant across port
        shapes.

    Port order on `xDUT` is `inn inp out vdd vss` — matches the stdlib
    diff opamp convention (TwoStageMillerPMOS / TwoStageMillerNMOS /
    FoldedCascode all declare ports in that order).

    v0.1.05 R81 D: same Rust-delegate pattern as the single-ended
    builders. Empty-roles call lands on
    ``clay_ir._build_tran_diff_testbench_rust`` byte-for-byte identical;
    non-empty falls back here so the v0.0.87 / v0.0.92 inject layers
    stay live unchanged.

    v0.1.12 R88 B: the Rust delegate now accepts `functional_roles` as
    an optional payload (matching the R85 D inject-helper tuple shape)
    and internally calls `emit_functional_role_op_lines` +
    `emit_v92_functional_role_lines` so the v0.0.87 / v0.0.92 inject
    layers are byte-equal in Rust. The Python fallback that re-emitted
    the deck is gone — this function unconditionally delegates to
    `clay_ir._build_tran_diff_testbench_rust`, passing the marshalled
    roles payload when non-empty.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. See
    :func:`_build_ac_testbench` for the soft-fail / predicate-patch
    contract.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_tran_diff_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        supply_v=supply_v,
        cload=cload,
        t_step=t_step,
        t_stop=t_stop,
        pulse_swing=pulse_swing,
        pulse_delay=pulse_delay,
        pulse_rise=pulse_rise,
        pulse_fall=pulse_fall,
        pulse_width=pulse_width,
        pulse_period=pulse_period,
        meas_from=meas_from,
        meas_to=meas_to,
        settling_band=settling_band,
        settling_avg_from=settling_avg_from,
        settling_avg_to=settling_avg_to,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


def _build_tran_fully_diff_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    supply_v: float,
    cload: float = 1e-12,
    t_step: float = 1e-10,
    t_stop: float = 1e-7,
    pulse_swing: float = 0.05,
    pulse_delay: float = 5e-9,
    pulse_rise: float = 1e-10,
    pulse_fall: float = 1e-10,
    pulse_width: float = 2e-8,
    pulse_period: float = 5e-8,
    meas_from: float = 8e-9,
    meas_to: float = 2.5e-8,
    settling_band: float = 0.005,
    settling_avg_from: Optional[float] = None,
    settling_avg_to: Optional[float] = None,
    r_cmfb_sense: float = 1e6,
    cmfb_ea_gain: float = 1e6,
    cmfb_ea_bw: float = float("inf"),
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Build a transient testbench for a 6-port fully-differential opamp.

    Driving + sensing convention (mirrors v0.0.77's AC fully-diff
    builder, with the AC source swapped for a VPULSE):

      * `Vcm` sets `vcm_target` at mid-rail (supply_v / 2).
      * `Vsig` is a VPULSE on `sig` swinging
        ``[-pulse_swing, +pulse_swing]`` around 0.
      * E1/E2 split `sig` into ``+0.5*sig`` at `inp` and ``-0.5*sig``
        at `inn`, centred on `vcm_target`.
      * `Rcm1`/`Rcm2` (1 MΩ default) average `outn`/`outp` into
        `vcm_sensed`; `E_cmfb` is the high-gain VCVS regulator that
        drives `vcmfb` from `(vcm_target - vcm_sensed)`. Same
        regulator topology as the v0.0.77 AC fully-diff builder.
      * `E_diff` synthesises `diff_out = outp - outn` so the
        single-node .meas lines work unchanged.
      * `Cload_p`/`Cload_n` load both outputs symmetrically.

    Port order on `xDUT` is `inn inp outn outp vdd vss` — matches the
    convention every detected FD cell follows.

    v0.0.82 addition: ``cmfb_cm_offset`` (V) — the time-averaged CM
    regulation residual ``(V(outp) + V(outn))/2 − vcm_target`` over the
    settling window. Ideal EA → ~0; finite EA gain → measurable
    non-zero offset.

    Honest deferred scope (NOT in this builder):

      * CMFB stability — the 1 MΩ / 1e6-gain VCVS loop is uncompensated.
        Adequate for the DC operating point + AC magnitude, but the
        CMFB pole interacts with the main amp's transient response.
        For tight CMFB tuning a finite-gain phase-compensated EA is
        needed.
      * CM-mode settling time — the new ``cmfb_cm_offset`` averages
        over the settling window, giving the steady-state CM residual;
        the CM step-response settling time itself (analogous to the
        diff_out ``settling_time``) is not measured.

    v0.1.06 R82 C: same Rust-delegate pattern as the single-ended +
    5-port diff builders. Empty-roles call lands on
    ``clay_ir._build_tran_fully_diff_testbench_rust`` byte-for-byte
    identical; non-empty falls back here so the v0.0.87 / v0.0.92
    inject layers stay live unchanged.

    v0.1.12 R88 B: the Rust delegate now accepts `functional_roles` as
    an optional payload (matching the R85 D inject-helper tuple shape)
    and internally calls `emit_functional_role_op_lines` +
    `emit_v92_functional_role_lines` so the v0.0.87 / v0.0.92 inject
    layers are byte-equal in Rust. The Python fallback that re-emitted
    the deck is gone — this function unconditionally delegates to
    `clay_ir._build_tran_fully_diff_testbench_rust`, passing the
    marshalled roles payload when non-empty.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. See
    :func:`_build_ac_testbench` for the soft-fail / predicate-patch
    contract.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_tran_fully_diff_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        supply_v=supply_v,
        cload=cload,
        t_step=t_step,
        t_stop=t_stop,
        pulse_swing=pulse_swing,
        pulse_delay=pulse_delay,
        pulse_rise=pulse_rise,
        pulse_fall=pulse_fall,
        pulse_width=pulse_width,
        pulse_period=pulse_period,
        meas_from=meas_from,
        meas_to=meas_to,
        settling_band=settling_band,
        settling_avg_from=settling_avg_from,
        settling_avg_to=settling_avg_to,
        r_cmfb_sense=r_cmfb_sense,
        cmfb_ea_gain=cmfb_ea_gain,
        cmfb_ea_bw=cmfb_ea_bw,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


def _build_dc_sweep_diff_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    supply_v: float,
    cload: float = 1e-12,
    vdd_sweep_frac: float = 0.10,
    vdd_sweep_step: float = 0.01,
    vcm_sweep_frac: float = 0.10,
    vcm_sweep_step: float = 0.01,
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Build a DC sweep testbench for a 5-port differential opamp.

    Driving convention (mirrors v0.0.75 AC diff template + v0.0.78 tran
    diff template):

      * `Vcm` sets the common-mode bias at mid-rail (supply_v / 2)
        during the supply sweep; becomes the swept source during the
        CM sweep.
      * `Vsig` is held at DC 0 — no differential drive during DC sweep.
        E1/E2 split `sig` into ±0.5 at `inp`/`inn` around `vcm`, so
        with `sig=0` both inputs sit at the CM bias.
      * Port order on `xDUT` is `inn inp out vdd vss`.

    Returned measures:

      * ``v_out_at_vdd_min`` / ``v_out_at_vdd_max`` — V(out) at the
        supply-sweep extrema. ``psrr_dc`` = 20*log10(|ΔV(out)/ΔV(vdd)|)
        is the linearised endpoint-secant PSRR (v0.0.79 contract).
      * ``v_out_at_vcm_min`` / ``v_out_at_vcm_max`` — V(out) at the
        CM-sweep extrema. ``cmrr_dc`` = 20*log10(|ΔV(out)/ΔV(vcm)|)
        is the linearised endpoint-secant CMRR (v0.0.79 contract).
      * ``v_out_window_max`` / ``v_out_window_min`` /
        ``vdd_at_v_out_max`` / ``vdd_at_v_out_min`` /
        ``psrr_dc_worst`` — v0.0.80 worst-case PSRR over the
        supply-sweep window via ``.meas dc max / min / max_at /
        min_at``. ``psrr_dc_worst`` collapses to ``psrr_dc`` when
        V(out) vs Vdd is monotonic; for non-monotonic cells it
        dominates the endpoint-secant magnitude.
      * ``v_out_cm_window_max`` / ``v_out_cm_window_min`` /
        ``vcm_at_v_out_max`` / ``vcm_at_v_out_min`` /
        ``cmrr_dc_worst`` — v0.0.80 worst-case CMRR over the CM-sweep
        window. Same window-secant interpretation as PSRR worst-case.

    v0.1.16 R93 D: same Rust-delegate pattern as the AC + tran diff +
    fully-diff DC sweep builders. The empty-roles path lands on
    ``clay_ir._build_dc_sweep_diff_testbench_rust`` byte-for-byte
    identical to the original v0.0.79 Python emission; non-empty roles
    flow through the same Rust delegate via the marshalled payload
    (Rust's ``extend_role_inject_lines`` reproduces the v0.0.87 /
    v0.0.92 inject layers byte-for-byte).

    v0.1.18 R96 D mass-deletion strip. The Python fallback that R93 D
    kept alive as `_build_dc_sweep_diff_testbench_pure_python` (the
    public escape hatch for a just-added Rust delegate) has been stripped
    — this function unconditionally delegates to the Rust port, matching
    the R88 B fully-diff DC sweep posture. Byte-equality across the
    Python reference and Rust port was asserted by R93 D + R94 F (51 +
    53 byte-equal tests across `tests/test_rust_testbench_diff_v128.py`)
    before strip. Raises `RuntimeError` when the Rust extension is
    missing (`_require_clay_ir` hard-error contract from R84 C).
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_dc_sweep_diff_testbench_rust(  # type: ignore[union-attr]
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        supply_v=supply_v,
        cload=cload,
        vdd_sweep_frac=vdd_sweep_frac,
        vdd_sweep_step=vdd_sweep_step,
        vcm_sweep_frac=vcm_sweep_frac,
        vcm_sweep_step=vcm_sweep_step,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


def _build_dc_sweep_diff_testbench_pure_python(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    supply_v: float,
    cload: float = 1e-12,
    vdd_sweep_frac: float = 0.10,
    vdd_sweep_step: float = 0.01,
    vcm_sweep_frac: float = 0.10,
    vcm_sweep_step: float = 0.01,
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """5-port diff DC sweep builder — R86 C-style backward-compat shim.

    v0.1.18 R96 D mass-deletion strip. The pure-Python implementation
    body (R93 D / v0.1.16) has been removed; this function unconditionally
    delegates to `clay_ir._build_dc_sweep_diff_testbench_rust`, the
    byte-equal Rust port landed by R93 D. Kept as a public symbol so
    pre-strip imports from `tests/test_rust_testbench_diff_v128.py` and
    any downstream consumers continue to work — the function now produces
    the same byte sequence as `_build_dc_sweep_diff_testbench` (both
    delegate to the same Rust path). Raises `RuntimeError` when the Rust
    extension is missing.

    Mirror of the R86 C `_emit_functional_role_op_lines` strip pattern
    and the R95 D `_expand_mirror` / `_expand_cs` strip pattern.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_dc_sweep_diff_testbench_rust(  # type: ignore[union-attr]
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        supply_v=supply_v,
        cload=cload,
        vdd_sweep_frac=vdd_sweep_frac,
        vdd_sweep_step=vdd_sweep_step,
        vcm_sweep_frac=vcm_sweep_frac,
        vcm_sweep_step=vcm_sweep_step,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )


def _build_dc_sweep_fully_diff_testbench(
    subckt_text: str,
    cell_name: str,
    *,
    corner_file: Optional[Path],
    supply_v: float,
    cload: float = 1e-12,
    vdd_sweep_frac: float = 0.10,
    vdd_sweep_step: float = 0.01,
    vcm_sweep_frac: float = 0.10,
    vcm_sweep_step: float = 0.01,
    r_cmfb_sense: float = 1e6,
    cmfb_ea_gain: float = 1e6,
    cmfb_ea_bw: float = float("inf"),
    functional_roles: Optional[Sequence[ParsedFunctionalRole]] = None,
) -> str:
    """Build a DC sweep testbench for a 6-port fully-differential opamp.

    Driving + sensing convention (mirrors v0.0.77 AC fully-diff + v0.0.78
    tran fully-diff templates, with the AC / VPULSE source replaced by
    a DC 0 source so the cell sees a pure CM input during both
    sub-sweeps):

      * `Vcm` sets `vcm_target` at mid-rail; becomes the swept source
        during the CM sweep.
      * `Vsig` is held at DC 0 — no differential drive during DC sweep.
      * E1/E2 split `sig` into ±0.5 around `vcm_target` so both inputs
        sit at the CM bias.
      * `Rcm1`/`Rcm2` + `E_cmfb` regulate the CM output via the same
        topology as the AC/tran fully-diff builders. The regulator's
        DC operating point lands ``vcm_sensed`` close to ``vcm_target``
        as long as the diff pair stays in saturation across both
        sub-sweep ranges.
      * `E_diff` synthesises ``diff_out = outp - outn`` so .meas
        targets a single node — same parser code path as the single-
        ended and 5-port DC sweep builders.

    Port order on `xDUT` is `inn inp outn outp vdd vss`.

    Returned measures: same `v_out_at_vdd_*` / `psrr_dc` /
    `v_out_at_vcm_*` / `cmrr_dc` set as the 5-port builder, but the
    `v_out_at_*` values are read from `diff_out` rather than `out`.

    v0.0.82 addition: ``cmfb_cm_offset`` (V) — the CM regulation
    residual ``(V(outp) + V(outn))/2 − vcm_target`` averaged over the
    Vdd sweep window. Ideal CM regulation is supply-independent so the
    average is a clean scalar; finite EA gain leaves a measurable
    non-zero offset.

    Honest deferred scope:

      * The CMFB regulator is uncompensated — fine for the DC operating
        point but the CMFB pole would interact with a real CMFB EA's
        AC behaviour. PSRR-DC / CMRR-DC measure the static supply /
        CM-to-diff_out gain only, so this is in-scope; the AC PSRR / AC
        CMRR sweep would need its own template.
      * `vout_at_mid` is NOT emitted on the FD path (the FD cell has no
        single-ended `out` to measure at a mid-rail input). It lands
        NaN via the wanted-list passthrough — same convention as the
        diff builder.

    v0.1.07 R83 C: same Rust-delegate pattern as the AC + tran fully-diff
    builders. Empty-roles call lands on
    ``clay_ir._build_dc_sweep_fully_diff_testbench_rust`` byte-for-byte
    identical; non-empty falls back here so the v0.0.87 / v0.0.92 inject
    layers stay live unchanged.

    v0.1.12 R88 B: the Rust delegate now accepts `functional_roles` as
    an optional payload (matching the R85 D inject-helper tuple shape)
    and internally calls `emit_functional_role_op_lines` +
    `emit_v92_functional_role_lines` so the v0.0.87 / v0.0.92 inject
    layers are byte-equal in Rust. The Python fallback that re-emitted
    the deck is gone — this function unconditionally delegates to
    `clay_ir._build_dc_sweep_fully_diff_testbench_rust`, passing the
    marshalled roles payload when non-empty.

    v0.1.07 R84 C: hard-errors when `_HAS_CLAY_IR` is False. See
    :func:`_build_ac_testbench` for the soft-fail / predicate-patch
    contract.
    """
    _ir_bridge._require_clay_ir()
    return _ir_bridge._clay_ir._build_dc_sweep_fully_diff_testbench_rust(
        subckt_text,
        cell_name,
        corner_file=corner_file.as_posix() if corner_file is not None else None,
        supply_v=supply_v,
        cload=cload,
        vdd_sweep_frac=vdd_sweep_frac,
        vdd_sweep_step=vdd_sweep_step,
        vcm_sweep_frac=vcm_sweep_frac,
        vcm_sweep_step=vcm_sweep_step,
        r_cmfb_sense=r_cmfb_sense,
        cmfb_ea_gain=cmfb_ea_gain,
        cmfb_ea_bw=cmfb_ea_bw,
        functional_roles=(
            _ir_bridge._roles_to_rust_payload(functional_roles)
            if functional_roles
            else None
        ),
    )
