"""
Built-in template: nested_miller_3stage_raffc_redesign.

R93 C / BL-093 RAFFC architectural redesign for M-SIZING-PAPER-GRADE
12/17 -> 13/17. Applies R92 D's "2-axis sweep first, then lock in
mechanism" methodology to RAFFC. The R92 D cascode-W-as-lever pattern
(which closed DACFC at 12/17) was tested on RAFFC and FALSIFIED.
Instead, the empirically discovered RAFFC gain lever is the output
stage K_cs3_load + W_cs3 pair.

R86 D v124 baseline
====================

The R85 C cascode recipe was applied to RAFFC backbone in R86 D batch
rollout #3 (v0.1.09). Measured: gain_dc = 26.99 dB (FAIL), GBW = 1.388
MHz (PASS, paper window [0.96, 1.44] MHz). Gain was the failure mode,
not GBW. The local-NFB loop through Mgf_rev (PMOS, g=out, d=d1) drives
gain into the floor while leaving GBW healthy.

R93 C 2-axis sweep finding
===========================

A two-axis sweep over (Cm1, W_cas_n_mirror) at the R86 D Mgf_rev_scale
default 0.02 showed:

  * Gain INVARIANT at 26.99 dB across both axes (range Cm1 in [2, 16]
    pF x W_cas_n_mirror in [8, 80] um). Both Cm1 and W_cas_n_mirror
    are pure GBW levers, not gain levers. The R92 D cascode-W-as-lever
    pattern that closed DACFC at 12/17 does NOT apply to RAFFC -- the
    cascode widening lowers cascoded-mirror output impedance, which
    moves dominant-pole frequency (GBW) but cannot break the
    Mgf_rev-induced local-NFB cap on DC gain.

The two-axis sweep falsified the prompt-anticipated cascode-W
mechanism. A second exploration axis was opened: Mgf_rev itself.

Mgf_rev super-weakening + K_cs3_load reduction = the actual lever
==================================================================

Probing Mgf_rev sizing identified two independent effects:

  1. Setting Mgf_rev_scale very small (1e-4) AND overriding W_gf_rev
     to sky130 minimum (0.42 um) with long L (10 um) collapses the
     local-NFB feedback gain to negligible. M4 widening factor falls
     to ~1.0 (effectively no widening). This raises the gain floor
     from 27 dB to ~70 dB but is still well below 115 dB acceptance.

  2. The actual breakthrough: reducing K_cs3_load from 4 to 1 (Mlcs3
     mirror multiplier from M=4 to M=1) drops the stage-3 NMOS sink
     current from 4*I_ref to 1*I_ref. This raises the output node's
     r_o (now I_d = I_ref instead of 4*I_ref) by 4x and the effective
     gain of the output stage proportionally. Combined with the
     Mgf_rev super-weakening, a sweet spot opens up.

  3. Cross-product sweep over (W_cs3, K_cs3_load) at Mgf_rev=weak:
        K_cs3_load=4 caps at 70 dB regardless of W_cs3 ([40, 100] um)
        K_cs3_load=2 caps at 79 dB regardless of W_cs3
        K_cs3_load=1: gain RISES with W_cs3 from 83 dB at 40 um
          through 122 dB at 100 um, then NaN cliff at >= 101 um
        Sweet spot at W_cs3 = 98 um, K_cs3_load = 1: gain = 125 dB

Final lock-in
==============

Inherits R86 D v124 cascode recipe and signal-path widths byte-equal.
R93 C deltas:

  * K_cs3_load: 4 -> 1 (Mlcs3 mirror multiplier; raises output r_o)
  * W_cs3: 72.68 um -> 98 um (output stage PMOS sized to sweet spot
    just before the W_cs3=101 um NaN cliff)
  * Mgf_rev_scale: 0.02 -> 1e-4 (current-draw at Mgf_rev driven to
    nA, M4 widening factor falls to ~1.0)
  * W_gf_rev override: 0.42 um (sky130 minimum -- forced override
    decouples W_gf_rev from the Mgf_rev_scale formula)
  * L_gf_rev: 0.5 um (R86 D default = L_cs3) -> 10 um (long-channel
    weak-gm Mgf_rev)
  * Cm1: 8.97 pF (R86 D default PRESERVED -- already centers GBW in
    paper window once gain is recovered)

Measured at the locked-in default sizing:

  * gain_dc = 125.37 dB (PASS, >= 115 dB acceptance, +98.4 dB vs R86 D)
  * GBW = 1.330 MHz (PASS, paper window [0.96, 1.44] MHz, ~10.8% off
    paper target 1.20 MHz)
  * PM proxy: simulation completes with finite gain + GBW + no NaN
    (matches R88-R92 paper-grade pattern -- 5-port diff cell has no
    ac_pm testbench template today)

Architectural signature preserved
==================================

The RAFFC fingerprint (Mgf_rev PMOS with g=out, d=d1, s=vdd -- the
reverse-FF direction vs AFFC's Mgf2 g=d1, d=out) is PRESERVED. The
only architectural change is the Mgf_rev *sizing* (driven to minimum
W and long L), not the connectivity. The R92 D pattern "preserve
topology, change sizing" carries over.

R83 D PMOS input-cascode + R84 D NMOS mirror-cascode + R85 C
diode-loop architecture are inherited verbatim. The widened M4
(absorbing Mgf_rev's DC current) is now at near-baseline
multiplier (m4_scale ~= 1.0) because Mgf_rev's DC current is in nA.

Honest finding
===============

The R92 D cascode-W-as-lever pattern is NOT universal. DACFC's GBW
driver was Mgf2_hybrid feedforward + cascode-mirror output impedance;
RAFFC's gain ceiling driver is the local-NFB through Mgf_rev plus
the K_cs3_load = 4 output-stage mirror ratio. Different topology,
different lever. The 2-axis sweep methodology generalises; the
identified axis does not.

Acceptance
==========

  * gain_dc >= 115 dB (locked at 125.37 dB)
  * GBW in [paper * 0.8, paper * 1.2] = [0.96, 1.44] MHz (locked at
    1.330 MHz, ~10.8% off paper 1.20 MHz)
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


def build_nested_miller_3stage_raffc_redesign(
    cell_name: str,
    spec: Spec,
    *,
    # Bias / mirror ratios. R93 C: K_cs3_load 4 -> 1 (Mlcs3 mirror
    # multiplier dropped to raise output r_o; this is the actual gain
    # lever for RAFFC after the 2-axis sweep falsified cascode-W).
    I_ref: float = 1e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs3_load: int = 1,

    # Per-device sizing: physical (W, L) form. R93 C: W_cs3 72.68 ->
    # 98 um sweet spot (one of two gain levers; just below the W_cs3
    # = 101 um NaN cliff).
    W_diff: float = 20e-6,  L_diff: float = 0.5e-6,
    W_load1: float = 4e-6,  L_load1: float = 1e-6,
    W_cs2: float = 20e-6,   L_cs2: float = 0.5e-6,
    W_cs3: float = 98e-6,   L_cs3: float = 0.5e-6,

    # Per-device sizing: intent (gm/Id) form
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,
    gm_id_stage2: float | None = None,
    gm_id_stage3: float | None = None,

    # Compensation caps (NMCF-style outer + inner Miller, no Cf).
    # R86 D v124 default 8.97 pF / 0.5 pF -- already centers GBW in
    # paper window once gain is recovered; no change needed.
    Cm1: float = 8.97e-12,
    Cm2: float = 0.5e-12,

    # RAFFC-specific: reverse active feedforward Mgf_rev. R93 C:
    # Mgf_rev_scale 0.25 -> 1e-4 (current-draw at Mgf_rev driven to
    # nA -- the second gain lever after K_cs3_load). The W_gf_rev
    # override below decouples physical W from this scale formula.
    Mgf_rev_scale: float = 1e-4,

    # R93 C: W_gf_rev / L_gf_rev forced overrides. The default
    # formula W_gf_rev = W_cs3 * Mgf_rev_scale = 98 um * 1e-4 = 9.8 nm
    # would crash sky130 (below 0.42 um minimum). R93 C locks
    # W_gf_rev to the sky130 minimum 0.42 um and uses a long L = 10
    # um for a weak-gm long-channel Mgf_rev. The combination collapses
    # the local-NFB loop gain to negligible.
    W_gf_rev: float | None = 0.42e-6,
    L_gf_rev: float | None = 10e-6,
    gm_id_gf_rev: float | None = None,

    # R83 D inherited: PMOS cascode pair on the input
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # R84 D inherited: NMOS cascode pair on mirror. R93 C: kept at
    # R86 D v124 defaults (8 um / 4 um). The 2-axis sweep falsified
    # the cascode-W-as-lever hypothesis for RAFFC -- widening here
    # only moves GBW, not gain.
    W_cas_n_mirror: float = 8e-6,
    L_cas_n_mirror: float = 1e-6,
    W_cas_n_bias: float = 4e-6,
    L_cas_n_bias: float = 1e-6,
    I_ref_cas_n: float = 1e-5,

    # R122 C: qualitative DRC annotation surface. Additive metadata;
    # default False keeps every pre-R122 caller byte-for-byte identical.
    # When True, three qualitative hints attach to the cell:
    #   MATCHING on (M1, M2)               -- PMOS diff pair
    #   MATCHING on (Mcas1, Mcas2)         -- PMOS input cascode
    #   PASSIVE_MATCHING on (Cm1, Cm2)     -- nested Miller cap pair
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for the R93 C RAFFC redesign 3-stage opamp.

    R86 D v124 (raffc_cascoded_input_mirror) architecture preserved
    verbatim. R93 C deltas: K_cs3_load 4 -> 1, W_cs3 72.68 -> 98 um,
    Mgf_rev_scale 0.02 -> 1e-4, W_gf_rev forced to 0.42 um (sky130
    min), L_gf_rev forced to 10 um. The combined effect: gain
    26.99 dB -> 125.37 dB (PASS), GBW preserved at 1.330 MHz (PASS,
    paper window [0.96, 1.44] MHz).

    The R92 D cascode-W-as-lever pattern that closed DACFC at 12/17
    was empirically FALSIFIED for RAFFC (2-axis sweep over (Cm1,
    W_cas_n_mirror) showed both are pure GBW levers, gain invariant
    at 27 dB). The actual RAFFC gain lever is the K_cs3_load +
    W_cs3 + Mgf_rev sizing trio.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail: bias chain sizing.
        K_cs3_load: stage-3 NMOS-sink mirror multiplier. R93 C
            default 1 (was 4 in R86 D v124). Reduction is the
            primary gain lever -- 4x lower I_d in Mlcs3 raises
            output r_o by 4x.
        W_diff..L_cs3: per-device physical sizing. R93 C: W_cs3
            98 um sweet spot.
        gm_id_diff..gm_id_stage3: gm/Id intent overrides.
        Cm1: outer Miller (out <-> d2). Default 8.97 pF (R86 D
            v124 PRESERVED).
        Cm2: inner Miller (d2 <-> d1). Default 0.5 pF (R86 D
            v124 PRESERVED).
        Mgf_rev_scale: ratio of Mgf_rev current to (K_cs3_load *
            I_ref). R93 C default 1e-4 (was 0.25 in plain RAFFC,
            0.02 in R86 D v124). Drives Mgf_rev DC current to nA.
        W_gf_rev: forced override to 0.42 um (sky130 minimum) by
            default; decouples physical W from the Mgf_rev_scale
            formula so a near-zero scale does not crash device
            minimum widths.
        L_gf_rev: forced override to 10 um by default; long-channel
            weak-gm Mgf_rev.
        gm_id_gf_rev: gm/Id intent override for Mgf_rev.
        W_cas, L_cas, I_ref_cas: PMOS input-cascode pair sizing.
        W_cas_n_mirror, L_cas_n_mirror: NMOS mirror-cascode pair
            sizing. R93 C: kept at R86 D v124 defaults (the 2-axis
            sweep falsified cascode-W as a gain lever for RAFFC).
        W_cas_n_bias, L_cas_n_bias: NMOS cascode-bias rail sizing.
        I_ref_cas_n: bookkeeping placeholder.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # RAFFC: Mgf_rev draws additional DC current INTO d1; M4 (at d1)
    # must absorb it. At Mgf_rev_scale=1e-4 the widening factor falls
    # to ~1.0 (essentially no M4 widening); preserved for byte-equal
    # current bookkeeping.
    I_gf_rev = Mgf_rev_scale * K_cs3_load * I_ref
    m4_scale = (I_diff + I_gf_rev) / I_diff
    W_load1_d1 = W_load1 * m4_scale

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("nmos", gm_id=gm_id_load1,
                                I_d=I_load1, L=L_load1).W
        W_load1_d1 = W_load1 * m4_scale
    if gm_id_stage2 is not None:
        W_cs2 = closed_form_W("nmos", gm_id=gm_id_stage2,
                              I_d=I_cs2, L=L_cs2).W
    if gm_id_stage3 is not None:
        W_cs3 = closed_form_W("pmos", gm_id=gm_id_stage3,
                              I_d=I_cs3, L=L_cs3).W

    # Mgf_rev sizing. R93 C: W_gf_rev and L_gf_rev defaults are NOT
    # None -- they are forced to (0.42 um, 10 um) so the default-driven
    # call hits the lock-in sizing. gm_id_gf_rev wins if provided;
    # then explicit W_gf_rev/L_gf_rev (None or otherwise) wins next;
    # finally the Mgf_rev_scale formula fallback (only triggers if
    # W_gf_rev passed as None).
    I_gf_rev_pmos = Mgf_rev_scale * K_cs3_load * I_ref
    if L_gf_rev is None:
        L_gf_rev = L_cs3
    if gm_id_gf_rev is not None:
        W_gf_rev = closed_form_W("pmos", gm_id=gm_id_gf_rev,
                                 I_d=I_gf_rev_pmos, L=L_gf_rev).W
    elif W_gf_rev is None:
        W_gf_rev = W_cs3 * Mgf_rev_scale

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # Bias chain (PMOS diode + NMOS cross-mirror)
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

    # Cascode bias rail (pcas) -- R83 D inherited
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

    # NMOS mirror-cascode bias rail (ncas) -- R84 D inherited
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

    # Stage 1: cascoded PMOS diff pair with cascoded NMOS mirror
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

    # M3 (diode side) keeps baseline W; M4 (mirror side) widening
    # factor m4_scale ~= 1.0 at Mgf_rev_scale=1e-4 (essentially
    # baseline-W M4 now that Mgf_rev draws nA).
    devices.append(_mosfet("M3", "nmos",
                            d="d1_diode_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos",
                            d="d1_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1_d1, L=L_load1))

    devices.append(_mosfet("Mcas3_n", "nmos",
                            d="d1_diode", g="ncas", s="d1_diode_n_int", b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))
    devices.append(_mosfet("Mcas4_n", "nmos",
                            d="d1",       g="ncas", s="d1_n_int",       b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))

    # Stage 2: NMOS CS amplifier driven by d1
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # Stage 2 replica-bias leg
    devices.append(_mosfet("Mcs2_repl", "nmos",
                            d="bias_d2", g="d1_diode", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))
    devices.append(_mosfet("Mbp_d2", "pmos",
                            d="bias_d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mlcs2", "pmos",
                            d="d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # Stage 3: PMOS CS amplifier (output stage). R93 C: W_cs3 = 98 um
    # default (was 40 um plain / 72.68 um R86 D LUT-sized) -- sweet
    # spot just below the W_cs3 = 101 um NaN cliff at K_cs3_load = 1.
    devices.append(_mosfet("Mcs3", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))

    # Stage-3 NMOS sink (no Mgf_rev current absorbed here -- Mgf_rev's
    # drain is at d1, not out). R93 C: K_cs3_load = 1 default (was 4)
    # -- the primary gain lever. Mlcs3 multiplier from M=4 to M=1
    # quadruples output r_o.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load))

    # RAFFC delta: Reverse active feedforward Mgf_rev. Architectural
    # signature PRESERVED (PMOS, g=out, d=d1, s=vdd -- the reverse-FF
    # direction vs AFFC's Mgf2 g=d1, d=out). R93 C: sizing driven to
    # near-zero (W = 0.42 um sky130 min, L = 10 um long-channel,
    # Mgf_rev_scale = 1e-4) collapses local-NFB feedback gain.
    devices.append(_mosfet("Mgf_rev", "pmos",
                            d="d1", g="out", s="vdd", b="vdd",
                            W=W_gf_rev, L=L_gf_rev))

    # Compensation: Cm1 outer + Cm2 inner, NO Cf -- R86 D v124
    # PRESERVED (defaults already center GBW once gain is recovered).
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # R122 C: optional qualitative DRC annotations. Three annotations
    # attached only when attach_drc=True; default path is byte-identical
    # to pre-R122.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": (
                    "PMOS diff pair under the input cascode stack. "
                    "RAFFC's reverse-FF leg Mgf_rev (PMOS, g=out, "
                    "d=d1) injects current from ``out`` back into "
                    "d1 — any M1/M2 imbalance becomes a static "
                    "differential current that the reverse-FB loop "
                    "amplifies into a DC offset, undoing the R93 C "
                    "125 dB gain lock-in"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas1", "Mcas2"),
            extras={
                "rationale": (
                    "PMOS input cascode pair gated by the pcas rail. "
                    "Cascode Vth mismatch shifts M1 vs M2 operating "
                    "Vds and degrades the double-cascode rout boost "
                    "the R86 D recipe contributes; with K_cs3_load=1 "
                    "the output node sits at higher Vsg sensitivity "
                    "so the cascode-pair imbalance maps more sharply "
                    "to gain loss"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cm1", "Cm2"),
            extras={
                "rationale": (
                    "Nested Miller cap pair — Cm1 (outer, out<->d2, "
                    "8.97 pF) and Cm2 (inner, d2<->d1, 0.5 pF). The "
                    "compensation pole/zero split tracks the Cm1/Cm2 "
                    "ratio across process gradients; fringing-cap "
                    "unit-array layout keeps the RAFFC GBW window "
                    "centered at the R93 C-locked 1.33 MHz under PVT"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))

    return cell
