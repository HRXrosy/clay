"""
Built-in template: nested_miller_3stage_acbc_cascoded_input_mirror.

Peng ACBC 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** added on top of the active-cap +
active-feedforward ACBC backbone. R86 D batch rollout #1 — applies
the R85 C cross-variant cascode recipe (telescopic PMOS input
cascode + DFCFC1-style regulated NMOS mirror with diode loop at
high-Z node ``d1``) to a STRUCTURALLY DIFFERENT NMCF-class variant:
ACBC's active-capacitance compensation (4th NMOS transconductor
Mgm4 + bias-comp node ``nb``).

Why this variant
================

R85 C v123 landed the DFCFC2 cross-variant generalisation; R86 D
extends to 5 NMCF variants in parallel. ACBC is the natural first
because:

  1. ACBC has a **standard NMCF stage 1** (plain PMOS diff-pair +
     NMOS diode mirror) — the R85 C cascode recipe drops in
     verbatim with NO topology re-engineering.
  2. ACBC's distinguishing compensation lives in the compensation
     network (Mgm4 + nb + Mnb_diode + Mnb_drv + Cm2 re-routed
     out↔nb), all downstream of stage 1. The cascode recipe at
     stage 1 doesn't disturb any ACBC architectural device.
  3. ACBC paper GBW target is **1.5 MHz** (per AnalogGym
     GraphAMPACBC.GBW_target). Cm1 should be in the same family
     as R84 D TCFC's 5 pF (paper 1 MHz) — empirical sweep needed.
  4. ACBC's Mgf2 (shared with AFFC) — primary FF — gates from d1
     to out. Same DC near-cancellation risk as DFCFC2 at full
     scale; expect Mgf2_scale ≈ 0.1-0.3 lands DC gain.

Architectural devices preserved verbatim
=========================================

Every ACBC fingerprint is preserved:

  1. **Mgf2** (PMOS, g=d1, d=out, s=vdd) — primary active
     feedforward, shared with AFFC.
  2. **Mgm4** (NMOS, g=d2, d=nb, s=vss) — 4th active
     transconductor (active-capacitance gain element).
  3. **Mnb_diode** (PMOS diode at nb) + **Mnb_drv** (PMOS
     mirror at pbias) — bias-comp chain.
  4. **Compensation**: Cm1 (out↔d1 skip-over) + Cm2 (nb↔d2
     active-cap routing). No Cf.

R85 C cascode recipe added
==========================

  * **Mcas1 / Mcas2** PMOS cascode pair between input pair drains
    and mirror summing nodes, gated by ``pcas`` rail.
  * **Mcas_b** + **Iref_cas** — pcas rail bias (R83 D inherited).
  * **Mcas3_n / Mcas4_n** NMOS cascode pair on mirror legs, gated
    by ``ncas`` rail; M3/M4 share gate at HIGH-Z node ``d1_diode``
    (DFCFC1-style regulated cascode loop, R84 D inherited).
  * **Mcas_n_b1** + **Mcas_n_b2** stacked NMOS-diode pair for ncas
    rail; **Mbp_ncas_drv** PMOS mirror at pbias sources I_ref.

The mirror diode loop closes at the high-Z signal node ``d1_diode``
(M3, M4 with g=d1_diode), exactly as in the R84 D / R85 C recipe.

Honest scope
============

ACBC paper GBW = 1.5 MHz. Initial Cm1 estimate 5 pF × (1/1.5)
≈ 3.3 pF. Mgf2_scale = 0.1 (R85 C inherited tune for the active
FF gating from d1). The active-cap loop (Mgm4 + nb + Cm2) is
small-signal effect at AC band; doesn't disturb DC gain expectation
of ~120-128 dB from the double-cascode + plain ACBC stages 2/3.

All seven W's (LUT/heuristic) BYTE-EQUAL with R85 C v123.

Knobs
-----

Inherits every kwarg from `build_nested_miller_3stage_acbc` PLUS
the cascode-related kwargs from R85 C:

  W_cas, L_cas, I_ref_cas
      PMOS input cascode pair sizing (R83 D pattern).
  W_cas_n_mirror, L_cas_n_mirror
      NMOS mirror-cascode pair sizing (R84 D pattern).
  W_cas_n_bias, L_cas_n_bias
      Stacked bias-diode pair sizing for the ncas rail.
  I_ref_cas_n
      Bookkeeping placeholder — actual ncas-rail current set by
      Mbp_ncas_drv at I_ref via the pbias mirror.
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


def build_nested_miller_3stage_acbc_cascoded_input_mirror(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Bias / mirror ratios ───────────────────────────────
    I_ref: float = 1e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs3_load: int = 4,

    # ─── Per-device sizing: physical (W, L) form ────────────
    W_diff: float = 20e-6,  L_diff: float = 0.5e-6,
    W_load1: float = 4e-6,  L_load1: float = 1e-6,
    W_cs2: float = 20e-6,   L_cs2: float = 0.5e-6,
    W_cs3: float = 40e-6,   L_cs3: float = 0.5e-6,

    # ─── Per-device sizing: intent (gm/Id) form ─────────────
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,
    gm_id_stage2: float | None = None,
    gm_id_stage3: float | None = None,

    # ─── Compensation caps (Cm1, Cm2 only — no Cf in ACBC) ──
    Cm1: float = 3.3e-12,
    Cm2: float = 0.5e-12,

    # ─── ACBC-specific: active feedforward Mgf2 (shared with AFFC) ─
    Mgf2_scale: float = 0.1,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # ─── ACBC-specific: active bias-comp transconductor Mgm4 ─
    Mgm4_scale: float = 1.0,
    W_gm4: float | None = None,
    L_gm4: float | None = None,
    gm_id_gm4: float | None = None,

    # ─── ACBC-specific: bias-comp node diode (Mnb_diode) ────
    W_bp_nb: float = 4e-6,
    L_bp_nb: float = 1e-6,

    # ─── R83 D inherited: PMOS cascode pair on the input ────
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # ─── R84 D inherited: NMOS cascode pair on mirror ───────
    W_cas_n_mirror: float = 4e-6,
    L_cas_n_mirror: float = 1e-6,
    W_cas_n_bias: float = 4e-6,
    L_cas_n_bias: float = 1e-6,
    I_ref_cas_n: float = 1e-5,

    # ─── R110 C: qualitative DRC annotation surface ─────────
    # Additive metadata (see folded_cascode.py R108 C precedent).
    # Default False keeps every pre-R110 caller byte-for-byte
    # identical. When True, three qualitative hints attach:
    #   MATCHING       on PMOS diff pair (M1 / M2),
    #   GUARD_RING     on PMOS tail current source (Mtail),
    #   WELL_ISOLATED  on the PMOS input cascode pair (Mcas1 / Mcas2) —
    #                  the architectural marker of the cascoded-input
    #                  recipe layered on the ACBC active-cap backbone.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Peng ACBC 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.

    Same 5-port surface as the plain ACBC cell. The cascode pair
    (Mcas1/Mcas2) and the NMOS mirror cascode pair (Mcas3_n/Mcas4_n)
    stack on the stage-1 input pair and the M3/M4 mirror legs
    respectively. The ACBC active-cap signature (Mgm4 + nb + Mnb_diode
    + Mnb_drv + Cm1 skip-over + Cm2 to nb) is preserved verbatim.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # ACBC: Mgf2 draws additional DC current.
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref
    K_cs3_load_total = K_cs3_load * (1 + Mgf2_scale)
    if K_cs3_load_total < 1:
        K_cs3_load_total = 1

    # ACBC: Mgm4 sinks I_gm4 into nb; Mnb_drv supplies it.
    I_gm4 = Mgm4_scale * I_cs2
    M_nb_drv = max(int(round(I_gm4 / I_ref)), 1)

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

    # Mgf2 sizing
    if L_gf2 is None:
        L_gf2 = L_cs3
    if gm_id_gf2 is not None:
        W_gf2 = closed_form_W("pmos", gm_id=gm_id_gf2,
                              I_d=I_gf2, L=L_gf2).W
    elif W_gf2 is None:
        W_gf2 = W_cs3 * Mgf2_scale

    # Mgm4 sizing
    if L_gm4 is None:
        L_gm4 = L_cs2
    if gm_id_gm4 is not None:
        W_gm4 = closed_form_W("nmos", gm_id=gm_id_gm4,
                              I_d=I_gm4, L=L_gm4).W
    elif W_gm4 is None:
        W_gm4 = W_cs2 * Mgm4_scale

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # ─── Bias chain (PMOS diode + NMOS cross-mirror) ─────────
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

    # ─── Cascode bias rail (pcas) — R83 D inherited ─────────
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

    # ─── NMOS mirror-cascode bias rail (ncas) — R84 D inherited ─
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

    # ─── Stage 1: cascoded PMOS diff pair with cascoded NMOS mirror ─
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode_int", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1_int",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    # PMOS cascode pair (R83 D inherited).
    devices.append(_mosfet("Mcas1", "pmos",
                            d="d1_diode", g="pcas", s="d1_diode_int", b="vdd",
                            W=W_cas, L=L_cas))
    devices.append(_mosfet("Mcas2", "pmos",
                            d="d1",       g="pcas", s="d1_int",       b="vdd",
                            W=W_cas, L=L_cas))

    # NMOS mirror load (R84 D regulated-cascode mirror).
    devices.append(_mosfet("M3", "nmos",
                            d="d1_diode_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos",
                            d="d1_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    # NMOS cascode pair on top of the mirror legs.
    devices.append(_mosfet("Mcas3_n", "nmos",
                            d="d1_diode", g="ncas", s="d1_diode_n_int", b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))
    devices.append(_mosfet("Mcas4_n", "nmos",
                            d="d1",       g="ncas", s="d1_n_int",       b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))

    # ─── Stage 2: NMOS CS amplifier driven by d1 ─────────────
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # ─── Stage 2 replica-bias leg ────────────────────────────
    devices.append(_mosfet("Mcs2_repl", "nmos",
                            d="bias_d2", g="d1_diode", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))
    devices.append(_mosfet("Mbp_d2", "pmos",
                            d="bias_d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mlcs2", "pmos",
                            d="d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # ─── Stage 3: PMOS CS amplifier (output stage) ───────────
    devices.append(_mosfet("Mcs3", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))

    # ─── ACBC delta 1: Active feedforward Mgf2 (g=d1, d=out) ──
    devices.append(_mosfet("Mgf2", "pmos", d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # NMOS sink at out — widened to absorb both Mcs3 and Mgf2 DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── ACBC delta 2: Active bias-comp node nb + Mgm4 ──────
    devices.append(_mosfet("Mnb_diode", "pmos",
                            d="nb", g="nb", s="vdd", b="vdd",
                            W=W_bp_nb, L=L_bp_nb))
    devices.append(_mosfet("Mnb_drv", "pmos",
                            d="nb", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=M_nb_drv))
    devices.append(_mosfet("Mgm4", "nmos",
                            d="nb", g="d2", s="vss", b="vss",
                            W=W_gm4, L=L_gm4))

    # ─── Compensation: Cm1 skip-over + Cm2 to nb (no Cf) ────
    # ACBC distinctive routing preserved verbatim:
    #   Cm1: out ↔ d1 (skip-over outer Miller)
    #   Cm2: nb ↔ d2 (inner Miller to bias-comp node)
    devices.append(_cap("Cm1", p="out", n="d1", value=Cm1))
    devices.append(_cap("Cm2", p="nb",  n="d2", value=Cm2))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R110 C: qualitative DRC annotation surface ────────────────────
    # R108 C established the surface on folded_cascode; R109 C extended
    # to 3 NMCF-family siblings. R110 C broadens to the ACBC cascoded-
    # input-mirror variant — the R86 D batch rollout #1 of the R85 C
    # cross-variant cascode recipe applied over the active-capacitance
    # ACBC backbone.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        # 1) MATCHING — PMOS differential pair (M1 / M2). Inherited
        #    from the ACBC backbone; input offset is Vth-mismatch
        #    limited.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": "PMOS diff pair — input offset is Vth-mismatch limited",
                "ratio": "1:1",
            },
        ))

        # 2) GUARD_RING — PMOS tail current source (Mtail). Carries
        #    N_tail × I_ref; substrate noise on Mtail surfaces as a
        #    tail-node common-mode disturbance and is amplified by the
        #    differential-to-common-mode coupling of the cascoded input
        #    pair.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": "tail current source — substrate-noise isolation",
                "well_type": "nwell",
            },
        ))

        # 3) WELL_ISOLATED — PMOS input cascode pair (Mcas1 / Mcas2).
        #    This is the architectural marker of the cascoded-input-
        #    mirror recipe applied over the ACBC backbone. Mcas1 / Mcas2
        #    sit above the diff-pair drains (sources tied to M1/M2
        #    drains, gates tied to pcas rail, drains feeding the
        #    cascoded NMOS mirror summing nodes d1_diode / d1). Their
        #    output impedance is what gives the stage its first-stage
        #    gain advantage over the plain ACBC backbone — body-effect
        #    coupling through a shared nwell with the diff-pair PMOS
        #    M1/M2 modulates |Vth_p| and degrades the cascode gm·ro.
        #    A dedicated nwell tie isolates the cascode pair from the
        #    diff-pair PMOS bulk.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mcas1", "Mcas2"),
            extras={
                "rationale": "PMOS input cascode pair — body-effect isolation from diff-pair PMOS",
                "isolate_from": ("M1", "M2"),
            },
        ))

    return cell
