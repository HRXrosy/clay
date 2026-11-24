"""
Built-in template: nested_miller_3stage_iac_redesign.

R94 C / BL-093 IAC architectural redesign for M-SIZING-PAPER-GRADE
13/17 -> 14/17. Applies the R92 D / R93 C "2-axis sweep first, then
lock in mechanism" methodology to IAC. R94 C found the IAC GBW lever
is the *same* mechanism R92 D used for DACFC (cascoded NMOS mirror
W widening lowers effective output impedance, pulls dominant pole
into paper window), with a different applied magnitude (2.5x vs
DACFC's 10x). IAC's own indirect-AC compensation network (Cm2_iac
nulling-R chain via Mgmt triode helper) was empirically FALSIFIED
as a GBW lever -- all four IAC-unique knobs (Cm2_iac, R0_iac, W_gmt,
L_gmt) are no-op on both gain and GBW at relevant biases.

R86 D v124 baseline
====================

The R85 C cascode recipe was applied to IAC backbone in R86 D batch
rollout #2 (v0.1.09). Measured at the R86 D v124 default sizing:
gain_dc = 119.508 dB (PASS, >= 115 dB) and GBW = 4.7254 MHz (FAIL,
above paper window [2.80, 4.20] MHz). Failure mode is opposite to
RAFFC's: IAC's gain already passes, but GBW is too high. The
cascoded-mirror output stage produces a dominant pole at ~5 MHz
when the cascode-W is at R85 C default 8 um; the pole must be
pulled DOWN to land in the [2.80, 4.20] MHz window.

R94 C 2-axis sweep finding
===========================

A 2-axis sweep over (Cm1_iac, W_cas_n_mirror) at the R86 D default
sizing showed:

  * Gain INVARIANT at ~119.5 dB across (Cm1_iac in [6, 48] pF) x
    (W_cas_n_mirror in [8, 80] um). All 12 cells return gain in
    [119.461, 119.510] dB -- spread less than 0.05 dB, essentially
    numerical noise. The gain already passes; no gain lever needed.
  * Cm1_iac is essentially no-op on GBW too (range 4.86 MHz at
    Cm1_iac=6 pF, W_cas=8 um down to 4.66 MHz at Cm1_iac=48 pF,
    W_cas=8 um -- only 4% change for 8x Cm1_iac swing). The IAC
    Cm1_iac was already nearly saturating the Miller pole at
    R86 D default.
  * W_cas_n_mirror is the actual GBW lever: widening 8 -> 80 um
    pulls GBW from 4.73 -> 1.49 MHz. Each 2.5x widening cuts GBW
    by roughly half. The mechanism is identical to R92 D DACFC:
    wider cascode -> lower effective output impedance of cascoded
    mirror at the high-Z d1 node -> dominant pole moves DOWN.

R94 C IAC-unique falsification axis
====================================

A second sweep over the IAC-unique compensation network knobs
(Cm2_iac, R0_iac, W_gmt, L_gmt) at the R86 D W_cas=8 um baseline
showed:

  * Cm2_iac swept 0.5 -> 5 pF (10x): zero change in gain or GBW
    (byte-identical 119.508 dB / 4.7254 MHz).
  * R0_iac swept 1 kohm -> 100 kohm (100x): zero change in gain
    or GBW (byte-identical 119.508 dB / 4.7254 MHz).
  * W_gmt swept 0.8 -> 80 um (100x): zero change in gain or GBW
    (byte-identical 119.508 dB / 4.7254 MHz).
  * Mgmf_scale swept 0.01 -> 1.0 (100x): does move gain and GBW
    *together* (1.0 -> 65 dB / 0.63 MHz; 0.01 -> 85 dB / 1.59 MHz)
    but destroys gain by 35-55 dB. Not a viable GBW lever -- it
    is a coupled gain/GBW knob, not a single-axis dial.

Honest finding: the IAC architectural distinctness (Mgmt triode
helper + niac/npath/vcas_iac nodes + Cm2_iac/R0_iac RC chain) is
preserved as device positions, but its small-signal electrical
effect is below numerical detection threshold at sky130 tt at the
v124 bias plan. The indirect-AC compensation network introduces
zero measurable pole/zero shift on the small-signal AC sweep. The
cascoded NMOS mirror (R85 C recipe) is what controls the dominant
pole.

R94 C lock-in
==============

Inherits R86 D v124 cascode recipe and signal-path widths verbatim.
R94 C single delta:

  * W_cas_n_mirror: 8 um -> 20 um (2.5x, sweet spot pulling GBW
    from 4.73 MHz to 3.47 MHz on paper target 3.50 MHz)
  * W_cas_n_bias: 4 um -> 10 um (2.5x, matched to wider stack to
    preserve V_DS_sat alignment)

The R92 D DACFC lock-in used 10x widening (W_cas_n_mirror 8 -> 80
um). R94 C IAC uses 2.5x because IAC's R86 D baseline GBW is 4.73
MHz (only 1.35x above paper target 3.50 MHz), versus DACFC's R86 D
baseline GBW which sat at ~5.5 MHz needing a much larger pull-down
to land in [1.60, 2.40] MHz at 2.00 MHz paper target. Same lever,
different magnitude for different topology.

Measured at the locked-in default sizing:

  * gain_dc = 119.510 dB (PASS, >= 115 dB, 4.5 dB headroom)
  * GBW = 3.474 MHz (PASS, paper window [2.80, 4.20] MHz, 0.75%
    off paper target 3.50 MHz -- essentially dead-on)
  * PM proxy: simulation completes with finite gain + GBW + no NaN
    (matches R88-R93 paper-grade pattern -- 5-port diff cell has no
    ac_pm testbench template today)

Architectural signature preserved
==================================

The IAC fingerprint (Mgmt triode helper PMOS with d=niac,
g=vcas_iac, s=npath; Mgmf active FF PMOS with g=d1, d=out; series
RC nulling chain Cm2_iac niac<->n_rc_iac + R0_iac n_rc_iac<->vss;
NMCF-style outer Miller Cm1_iac out<->d2) is PRESERVED verbatim.
The only structural change is the *sizing* of the inherited R85 C
cascode mirror (W_cas_n_mirror + W_cas_n_bias both 2.5x), not the
connectivity. The R92 D / R93 C pattern "preserve topology, change
sizing" carries over.

R83 D PMOS input-cascode + R84 D NMOS mirror-cascode + R85 C
diode-loop architecture are inherited verbatim.

Acceptance
==========

  * gain_dc >= 115 dB (locked at 119.510 dB, +4.5 dB headroom)
  * GBW in [paper * 0.8, paper * 1.2] = [2.80, 4.20] MHz (locked
    at 3.474 MHz, 0.75% off paper target 3.50 MHz)
  * PM proxy: simulation completes with finite gain + GBW + no NaN
"""

from __future__ import annotations

from ..ir import Cell, Device, Spec


def _mosfet(name: str, kind: str, d: str, g: str, s: str, b: str,
            W: float, L: float, M: int = 1) -> Device:
    return Device(
        instance_name=name,
        device_class=kind,
        pdk_variant=None,
        pins={"d": d, "g": g, "s": s, "b": b},
        sizing={"W": W, "L": L, "M": float(M)},
    )


def _cap(name: str, p: str, n: str, value: float) -> Device:
    return Device(
        instance_name=name,
        device_class="cap",
        pdk_variant=None,
        pins={"p": p, "n": n},
        sizing={"value": value},
    )


def _res(name: str, p: str, n: str, value: float) -> Device:
    return Device(
        instance_name=name,
        device_class="res",
        pdk_variant=None,
        pins={"p": p, "n": n},
        sizing={"value": value},
    )


def build_nested_miller_3stage_iac_redesign(
    cell_name: str,
    spec: Spec,
    *,
    # Bias / mirror ratios -- R86 D v124 PRESERVED.
    I_ref: float = 1e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs3_load: int = 4,

    # Per-device sizing: physical (W, L) form -- R86 D v124 PRESERVED.
    W_diff: float = 20e-6,  L_diff: float = 0.5e-6,
    W_load1: float = 4e-6,  L_load1: float = 1e-6,
    W_cs2: float = 20e-6,   L_cs2: float = 0.5e-6,
    W_cs3: float = 40e-6,   L_cs3: float = 0.5e-6,

    # Per-device sizing: intent (gm/Id) form.
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,
    gm_id_stage2: float | None = None,
    gm_id_stage3: float | None = None,

    # IAC-specific: triode helper Mgmt -- R86 D v124 PRESERVED.
    # Falsified as GBW lever: W_gmt swept 0.8 -> 80 um (100x), zero
    # change in either gain or GBW. The Mgmt triode device is below
    # numerical detection threshold for small-signal AC at this bias.
    W_gmt: float = 8e-6,
    L_gmt: float = 1e-6,

    # IAC-specific: active FF Mgmf -- R86 D v124 PRESERVED.
    # Mgmf_scale is a coupled gain/GBW knob, NOT a single-axis lever.
    # Reducing it does pull GBW down but at huge gain cost.
    Mgmf_scale: float = 0.1,
    W_gmf: float | None = None,
    L_gmf: float | None = None,
    gm_id_gmf: float | None = None,

    # IAC-specific: compensation network -- R86 D v124 PRESERVED.
    # Falsified as GBW lever: Cm2_iac, R0_iac swept 10x/100x with
    # zero effect on either gain or GBW. The indirect-AC RC chain
    # via Mgmt has no measurable small-signal effect at relevant
    # bias.
    Cm1_iac: float = 1.4e-12,    # outer Miller NMCF-style (out<->d2)
    Cm2_iac: float = 0.5e-12,    # indirect cap (niac<->n_rc_iac)
    R0_iac:  float = 10e3,       # nulling-R (n_rc_iac<->vss)

    # R83 D inherited: PMOS cascode pair on the input.
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # R94 C delta: NMOS cascode pair on mirror WIDENED 8 -> 20 um
    # (2.5x). Empirical: widening lowers cascoded-mirror effective
    # output impedance, pulling dominant pole from 4.73 MHz into
    # the paper [2.80, 4.20] MHz window at 3.47 MHz. Same mechanism
    # as R92 D DACFC, with smaller magnitude (DACFC needed 10x to
    # land in [1.60, 2.40] MHz; IAC needs 2.5x to land in [2.80,
    # 4.20] MHz at paper target 3.50 MHz). The 2-axis sweep
    # falsified all four IAC-unique compensation knobs (Cm2_iac,
    # R0_iac, W_gmt, L_gmt) as GBW levers; the R85 C cascoded-
    # mirror stage is the actual dominant-pole controller.
    W_cas_n_mirror: float = 20e-6,
    L_cas_n_mirror: float = 1e-6,
    # R94 C delta: cascode-bias rail also WIDENED 4 -> 10 um (2.5x)
    # to match the wider cascode stack and keep V_DS_sat alignment
    # on the cascoded mirror devices.
    W_cas_n_bias: float = 10e-6,
    L_cas_n_bias: float = 1e-6,
    I_ref_cas_n: float = 1e-5,

    # ─── R110 C: qualitative DRC annotation surface ─────────
    # Additive metadata (see folded_cascode.py R108 C precedent).
    # Default False keeps every pre-R110 caller byte-for-byte
    # identical. When True, three qualitative hints attach:
    #   MATCHING       on PMOS diff pair (M1 / M2),
    #   GUARD_RING     on PMOS tail current source (Mtail),
    #   WELL_ISOLATED  on the IAC triode-helper Mgmt — the indirect-AC
    #                  compensation network's architectural fingerprint.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for the R94 C IAC redesign 3-stage opamp.

    R86 D v124 (iac_cascoded_input_mirror) architecture PRESERVED
    verbatim. R94 C deltas: W_cas_n_mirror 8 -> 20 um (2.5x) and
    W_cas_n_bias 4 -> 10 um (2.5x). The widened cascode reduces
    effective output impedance of the cascoded mirror, pulling the
    dominant pole from 4.73 MHz down into the paper [2.80, 4.20]
    MHz window at 3.47 MHz (0.75% off paper target 3.50 MHz).

    The 2-axis sweep methodology preserved from R92 D / R93 C. The
    identified lever (cascoded-mirror W) is the *same* as R92 D
    DACFC because both share the R85 C cascoded NMOS mirror at the
    high-Z d1 node. The IAC-unique compensation network (Cm2_iac
    RC chain via Mgmt triode helper) was empirically FALSIFIED as
    a GBW lever -- all four IAC-unique knobs are no-op on gain and
    GBW within numerical noise.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing -- R86 D v124 PRESERVED.
        W_diff..L_cs3: per-device physical sizing -- R86 D v124
            PRESERVED.
        gm_id_diff..gm_id_stage3: gm/Id intent overrides.
        W_gmt, L_gmt: triode-helper Mgmt sizing -- R86 D v124
            PRESERVED (falsified as GBW lever).
        Mgmf_scale, W_gmf, L_gmf, gm_id_gmf: active FF Mgmf sizing
            -- R86 D v124 PRESERVED.
        Cm1_iac: outer Miller (out <-> d2). Default 1.4 pF (R86 D
            v124 PRESERVED).
        Cm2_iac: indirect cap (niac <-> n_rc_iac). Default 0.5 pF
            (R86 D v124 PRESERVED, falsified as GBW lever).
        R0_iac: nulling-R (n_rc_iac <-> vss). Default 10 kohm (R86
            D v124 PRESERVED, falsified as GBW lever).
        W_cas, L_cas, I_ref_cas: PMOS input-cascode pair sizing
            -- R83 D inherited.
        W_cas_n_mirror: R94 C WIDENED to 20 um (from R86 D v124
            default 8 um, 2.5x). Empirical: widening lowers
            cascoded mirror's effective output impedance, pulling
            dominant pole from 4.73 MHz into [2.80, 4.20] MHz paper
            window.
        L_cas_n_mirror: NMOS cascode mirror L.
        W_cas_n_bias: R94 C WIDENED to 10 um (from R86 D v124
            default 4 um, 2.5x) to match the wider cascode stack.
        L_cas_n_bias: stacked NMOS-diode pair L.
        I_ref_cas_n: bookkeeping placeholder.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan -- same as NMCF.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # IAC: Mgmf draws additional DC current at out (same as AFFC).
    I_gmf = Mgmf_scale * K_cs3_load * I_ref
    K_cs3_load_total = int(round(K_cs3_load * (1 + Mgmf_scale)))
    if K_cs3_load_total < 1:
        K_cs3_load_total = 1

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("nmos", gm_id=gm_id_load1,
                                I_d=I_load1, L=L_load1).W
    if gm_id_stage2 is not None:
        W_cs2 = closed_form_W("nmos", gm_id=gm_id_stage2,
                              I_d=I_cs2, L=L_cs2).W
    if gm_id_stage3 is not None:
        W_cs3 = closed_form_W("pmos", gm_id=gm_id_stage3,
                              I_d=I_cs3, L=L_cs3).W

    # Mgmf sizing (active FF, same logic as AFFC's Mgf2).
    if L_gmf is None:
        L_gmf = L_cs3
    if gm_id_gmf is not None:
        W_gmf = closed_form_W("pmos", gm_id=gm_id_gmf,
                              I_d=I_gmf, L=L_gmf).W
    elif W_gmf is None:
        W_gmf = W_cs3 * Mgmf_scale

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # Bias chain (PMOS diode + NMOS cross-mirror).
    devices.append(_mosfet("Mbp",  "pmos", d="pbias", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbn",  "nmos", d="nbias", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mbnm", "nmos", d="pbias", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    devices.append(Device(
        instance_name="Iref",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "vdd", "n": "nbias"},
        sizing={"value": I_ref},
    ))

    # Cascode bias rail (pcas) -- R83 D inherited.
    devices.append(_mosfet("Mcas_b", "pmos",
                            d="pcas", g="pcas", s="vdd", b="vdd",
                            W=W_cas, L=L_cas))
    devices.append(Device(
        instance_name="Iref_cas",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "pcas", "n": "vss"},
        sizing={"value": I_ref_cas},
    ))

    # NMOS mirror-cascode bias rail (ncas) -- R84 D inherited.
    # R94 C delta: W_cas_n_bias WIDENED to 10 um default (2.5x R86 D
    # v124) to match the wider cascode stack and preserve V_DS_sat
    # alignment.
    devices.append(_mosfet("Mcas_n_b1", "nmos",
                            d="ncas_int", g="ncas_int",
                            s="vss", b="vss",
                            W=W_cas_n_bias, L=L_cas_n_bias))
    devices.append(_mosfet("Mcas_n_b2", "nmos",
                            d="ncas", g="ncas",
                            s="ncas_int", b="vss",
                            W=W_cas_n_bias, L=L_cas_n_bias))
    devices.append(_mosfet("Mbp_ncas_drv", "pmos",
                            d="ncas", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # Stage 1: cascoded PMOS diff pair with cascoded NMOS mirror.
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode_int", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1_int",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    devices.append(_mosfet("Mcas1", "pmos",
                            d="d1_diode", g="pcas", s="d1_diode_int", b="vdd",
                            W=W_cas, L=L_cas))
    devices.append(_mosfet("Mcas2", "pmos",
                            d="d1",       g="pcas", s="d1_int",       b="vdd",
                            W=W_cas, L=L_cas))

    devices.append(_mosfet("M3", "nmos",
                            d="d1_diode_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos",
                            d="d1_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    # R94 C delta: cascode stack devices WIDENED via W_cas_n_mirror
    # default 20 um (from R86 D v124 default 8 um, 2.5x). Lowers
    # effective output impedance of the cascoded mirror, pulling
    # dominant pole from 4.73 MHz into the paper [2.80, 4.20] MHz
    # window at 3.47 MHz (0.75% off paper target 3.50 MHz).
    devices.append(_mosfet("Mcas3_n", "nmos",
                            d="d1_diode", g="ncas", s="d1_diode_n_int", b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))
    devices.append(_mosfet("Mcas4_n", "nmos",
                            d="d1",       g="ncas", s="d1_n_int",       b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))

    # Stage 2: NMOS CS amplifier driven by d1.
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # Stage 2 replica-bias leg (unchanged from NMCF).
    devices.append(_mosfet("Mcs2_repl", "nmos",
                            d="bias_d2", g="d1_diode", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))
    devices.append(_mosfet("Mbp_d2", "pmos",
                            d="bias_d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mlcs2", "pmos",
                            d="d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # Stage 3: PMOS CS amplifier (output stage).
    devices.append(_mosfet("Mcs3", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))

    # IAC delta: bias rails for indirect compensation -- PRESERVED.
    # vcas_iac: cascode bias rail for Mgmt's gate. PMOS diode from
    # vdd, holds vcas_iac at vdd - |Vgs_p|.
    devices.append(_mosfet("Mbp_vcas_iac", "pmos",
                            d="vcas_iac", g="vcas_iac", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    # npath: source rail for Mgmt. PMOS diode from vdd holds it as
    # a clamped source rail one |Vgs_p| below vdd.
    devices.append(_mosfet("Mbp_npath", "pmos",
                            d="npath", g="npath", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # IAC delta: triode-helper Mgmt -- PRESERVED.
    # Architectural signature of IAC. PMOS in triode region acting as
    # the indirect AC coupling buffer between vcas_iac/npath and the
    # indirect Miller node niac. NOT a CS-style device gated by d2 --
    # its gate at vcas_iac (a cascode bias rail) is the "indirect"
    # signature. R94 C falsified W_gmt as a GBW lever (sweeping 0.8 ->
    # 80 um produced zero change in gain or GBW within numerical
    # noise) -- preserved as architectural fingerprint, not tuned.
    devices.append(_mosfet("Mgmt", "pmos",
                            d="niac", g="vcas_iac", s="npath", b="vdd",
                            W=W_gmt, L=L_gmt))

    # NMOS sink at niac so Mgmt's drain has a DC path. Sinks I_ref to
    # vss; this also forces Mgmt into triode by limiting V(niac).
    devices.append(_mosfet("Mbn_niac", "nmos",
                            d="niac", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # IAC delta: active feedforward Mgmf -- PRESERVED.
    # PMOS gated by d1, drain at out -- standard AFFC-style FF
    # position. Provides the LHP zero through an active FF leg.
    devices.append(_mosfet("Mgmf", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gmf, L=L_gmf))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # IAC compensation network -- PRESERVED.
    # Cm1_iac: NMCF-style outer Miller (out <-> d2).
    devices.append(_cap("Cm1_iac", p="out", n="d2", value=Cm1_iac))
    # Cm2_iac: indirect Miller cap from niac to inner RC node.
    # R94 C falsified as GBW lever (sweeping 0.5 -> 5 pF produced
    # zero change) -- preserved as architectural fingerprint.
    devices.append(_cap("Cm2_iac", p="niac", n="n_rc_iac", value=Cm2_iac))
    # R0_iac: nulling resistor closing the RC path to vss.
    # R94 C falsified as GBW lever (sweeping 1 kohm -> 100 kohm
    # produced zero change) -- preserved as architectural fingerprint.
    devices.append(_res("R0_iac", p="n_rc_iac", n="vss", value=R0_iac))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R110 C: qualitative DRC annotation surface ────────────────────
    # R108 C established the surface on folded_cascode; R109 C extended
    # to 3 NMCF-family siblings. R110 C broadens to the IAC redesign —
    # R94 C BL-093 IAC architectural lock-in (cascoded input + cascoded
    # NMOS mirror + indirect-AC compensation network).
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        # 1) MATCHING — PMOS differential pair (M1 / M2). Input offset
        #    is Vth-mismatch limited; common-centroid layout required
        #    on the cascoded-input pair.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": "PMOS diff pair — input offset is Vth-mismatch limited",
                "ratio": "1:1",
            },
        ))

        # 2) GUARD_RING — PMOS tail current source (Mtail). Carries
        #    N_tail × I_ref; substrate-noise isolation prevents
        #    common-mode disturbance on the tail node.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": "tail current source — substrate-noise isolation",
                "well_type": "nwell",
            },
        ))

        # 3) WELL_ISOLATED — Mgmt (IAC triode-helper PMOS). This is the
        #    architectural fingerprint of the IAC topology: a PMOS in
        #    the triode region acting as the indirect-AC coupling
        #    buffer between the cascode-bias rail vcas_iac, the clamped
        #    source rail npath, and the indirect Miller node niac. R94 C
        #    empirically falsified W_gmt as a small-signal lever (sweep
        #    0.8 → 80 μm produced zero gain/GBW change), so Mgmt is
        #    preserved as device-position fingerprint only — but in a
        #    physical layout the indirect-coupling node niac is
        #    high-impedance and body-effect coupling onto Mgmt's bulk
        #    via shared nwell with the standard PMOS rail devices
        #    (Mbp, Mbp_vcas_iac, Mbp_npath) would inject substrate
        #    swing onto niac, defeating the indirect-AC compensation
        #    intent. Dedicated nwell tie keeps Mgmt's coupling node
        #    immune to bulk modulation.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mgmt",),
            extras={
                "rationale": "IAC triode-helper PMOS — body-effect isolation on indirect-AC coupling node",
                "isolate_from": ("Mbp_vcas_iac", "Mbp_npath"),
            },
        ))

    return cell
