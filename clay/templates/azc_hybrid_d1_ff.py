"""
Built-in template: nested_miller_3stage_azc_hybrid_d1_ff.

R89 B / BL-093 hybrid d1-gated FF retrofit batch — AZC variant.
Rolls out R88 C's NMCNR-validated retrofit pattern (cascoded mirror
+ d1-gated Mgf2-style FF → +30-35 dB gain lift) to Qu2017 AZC.

Structural finding (R87 C / R88 C closure)
==========================================

R88 C pilot retrofit on NMCNR validated R87 C's structural finding:
the R85 C cascode recipe REQUIRES a d1-gated Mgf2-style PMOS FF
(PMOS, g=d1, d=out, s=vdd) to realise its +30-35 dB gain product.
NMCNR cascoded baseline 83.29 dB → NMCNR + hybrid d1-FF 119.36 dB
(+36.07 dB lift, GBW landed at 1.230 MHz inside paper +/-20% window).

R89 B AZC retrofit: direct graft
=================================

AZC's R87 C cascoded baseline has NO active FF at all — only the
charge-redistribution archetype trio (Mazc1 / Mazc2_diode / Mazc3)
+ quad-R mesh + triple-cap network. R87 C measured AZC gain_dc =
83.29 dB (exactly the FAIL-variant ceiling, identical to NMCNR
cascoded), confirming AZC has no compensating gain path.

R89 B AZC retrofit is **direct graft**: just add Mgf2_hybrid (g=d1)
to the AZC backbone — no original FF to preserve. The experimental
risk is whether AZC's quad-R nulling mesh + triple-cap network at
internal nodes (nazc_bias, nazc_hub, nazc_cap0/1/2) interferes with
the new d1→out direct gain path.

Structurally there's no interference: AZC's mesh sits between d2,
nazc_*, and vss; Mgf2_hybrid sits between d1 and out. The two
networks share only vdd / vss / d2 (d2 enters the AZC mesh via
Mazc1.g=d2). The d1→out path through Mgf2_hybrid is independent.

Architectural devices preserved verbatim
=========================================

AZC fingerprint that survives R89 B retrofit:

  1. **Mazc1** (NMOS, d=nazc1, g=d2, s=vss).
  2. **Mazc2_diode** (PMOS diode at nazc1).
  3. **Mazc3** (NMOS, d=nazc_hub, g=nazc_cap2, s=vss).
  4. **Razc0-Razc3** quad-R mesh.
  5. **Cazc0/1/2** triple-cap network (inner caps to vss).
  6. **DC bias paths** Mbp_azc_bias + Mbn_azc_hub + Mbn_azc_cap0.

R85 C cascode recipe inherited verbatim
=======================================

  * Mcas1 / Mcas2 PMOS cascode pair on input.
  * Mcas_b + Iref_cas — pcas rail bias.
  * Mcas3_n / Mcas4_n NMOS cascode pair on mirror legs. M3/M4
    gates closed at ``d1_diode`` (DFCFC1-style regulated cascode
    loop).
  * Mcas_n_b1 + Mcas_n_b2 + Mbp_ncas_drv ncas rail.

R89 B delta
===========

  * **Mgf2_hybrid** (PMOS, g=d1, d=out, s=vdd) — NEW d1-FF grafted
    from R88 C NMCNR pattern.
  * **Mlcs3** sized by K_cs3_load * (1 + Mgf2_scale) — wider NMOS
    sink absorbs Mgf2_hybrid's DC current at out (AZC has no
    original FF so only one delta scale).

If gain_dc lands in [115, 120] dB AND GBW stays in [2.40, 3.60] MHz,
R88 C structural finding generalises across variants with quad-R
nulling mesh compensation. M-SIZING-PAPER-GRADE 9/17 → 10/17 (AZC
counted as PASS).

Honest scope
============

AZC paper recipe has NO active FF at all — the charge-redistribution
trio + quad-R + triple-cap is the entire compensation. Adding
Mgf2_hybrid is purely a Clay engineering addition motivated by R88 C
structural finding, not paper replication. The R88 C lock-in Cm1 =
10 pF for 1.20 MHz target scales (roughly) to Cazc0 ~4.0 pF for AZC's
3.00 MHz paper target, but real GBW depends on AZC's mesh-dominated
pole structure — Cazc0 needs re-tune via sweep.
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


def build_nested_miller_3stage_azc_hybrid_d1_ff(
    cell_name: str,
    spec: Spec,
    *,
    # Bias / mirror ratios
    I_ref: float = 1e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs3_load: int = 4,

    # Per-device sizing: physical (W, L) form
    W_diff: float = 20e-6,  L_diff: float = 0.5e-6,
    W_load1: float = 4e-6,  L_load1: float = 1e-6,
    W_cs2: float = 20e-6,   L_cs2: float = 0.5e-6,
    W_cs3: float = 40e-6,   L_cs3: float = 0.5e-6,

    # Per-device sizing: intent (gm/Id) form
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,
    gm_id_stage2: float | None = None,
    gm_id_stage3: float | None = None,

    # AZC-specific: triple-MOSFET archetype sizing
    W_azc: float = 4e-6,
    L_azc: float = 1e-6,

    # AZC-specific: quad-R nulling-resistor mesh
    Razc0: float = 10e3,
    Razc1: float = 10e3,
    Razc2: float = 10e3,
    Razc3: float = 10e3,

    # AZC-specific: triple-cap compensation network
    Cazc0: float = 1.5e-12,   # outer Miller-like (out <-> nazc_bias)
    Cazc1: float = 0.5e-12,   # inner cap to vss
    Cazc2: float = 0.5e-12,   # inner cap to vss

    # R89 B hybrid d1-FF retrofit (R88 C inherited DFCFC1/DFCFC2 Mgf2)
    Mgf2_scale: float = 0.1,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # R89 B AZC GBW-tune helper. AZC's native quad-R + triple-cap
    # mesh has no direct out<->high-Z-node Miller path; the
    # dominant pole is set by the output stage's load + gm.
    # Cm_helper provides an out<->d2 NMCF-style outer Miller for
    # GBW control. Default 0 (disabled, AZC native behaviour).
    # Documented as R89 B engineering addition, not AZC paper.
    Cm_helper: float = 0.0,

    # R83 D inherited: PMOS cascode pair on the input
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # R84 D inherited: NMOS cascode pair on mirror
    W_cas_n_mirror: float = 4e-6,
    L_cas_n_mirror: float = 1e-6,
    W_cas_n_bias: float = 4e-6,
    L_cas_n_bias: float = 1e-6,
    I_ref_cas_n: float = 1e-5,

    # R118 C: optional qualitative DRC annotations
    # When True, attach three R111 C-vocabulary annotations on the
    # AZC + R89 B hybrid d1-FF direct-graft retrofit:
    #   * MATCHING on the PMOS diff pair (M1, M2). Vth mismatch maps
    #     directly to input-referred offset; AZC has no auto-zero
    #     loop so diff-pair mismatch propagates unattenuated through
    #     stage 1 and then through the d1->out Mgf2_hybrid path.
    #   * MATCHING on the NMOS mirror cascode pair (Mcas3_n, Mcas4_n).
    #     The R85 C cascode mirror with DFCFC1-style diode loop at
    #     d1_diode closes a regulated-cascode feedback through these
    #     two devices; mismatch unbalances the loop and re-introduces
    #     offset at d1, partially cancelling the cascode gain product
    #     that the AZC direct-graft retrofit relies on.
    #   * PASSIVE_MATCHING on the inner-cap pair (Cazc1, Cazc2). Both
    #     terminate at vss and form the symmetric pair of the AZC
    #     triple-cap compensation network; their ratio sets two of
    #     the three nazc_cap poles in the AZC mesh-dominated pole
    #     structure, so process-tracking matching is required to
    #     keep the dominant-pole location predictable across PVT.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for AZC 3-stage opamp with R85 C cascode recipe,
    triple-MOSFET archetype + quad-R mesh + triple-cap network
    PRESERVED, AND a NEW Mgf2_hybrid (g=d1) grafted from R88 C NMCNR
    pattern (R89 B direct-graft retrofit).

    AZC signature (triple-MOSFET + quad-R + triple-cap) preserved
    verbatim. NEW: Mgf2_hybrid (PMOS, g=d1, d=out) + Mlcs3 widened
    by (1 + Mgf2_scale) to absorb the FF DC current.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (NMCF baseline inherited).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the baseline signal path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four signal-path devices.
        W_azc, L_azc: triple-MOSFET archetype sizing.
        Razc0-Razc3: quad-R nulling-resistor mesh.
        Cazc0: outer Miller-like (out ↔ nazc_bias). Default 1.5 pF.
        Cazc1, Cazc2: inner caps to vss. Default 0.5 pF each.
        Mgf2_scale: R89 B hybrid d1-FF scale relative to Mcs3 stage-3
            CS amp. Default 0.1 (R88 C tested-good value).
        W_gf2, L_gf2, gm_id_gf2: Mgf2_hybrid sizing overrides.
        Cm_helper: R89 B engineering addition — optional out<->d2
            NMCF-style outer Miller for GBW tune. Default 0 (disabled,
            AZC native behaviour). When > 0, slows dominant pole to
            fit paper GBW window. Honest scope: AZC's native quad-R
            + triple-cap mesh has no out<->high-Z-node Miller path so
            cannot satisfy paper GBW via native caps alone.
        W_cas, L_cas, I_ref_cas: PMOS input-cascode pair sizing.
        W_cas_n_mirror, L_cas_n_mirror: NMOS mirror-cascode pair sizing.
        W_cas_n_bias, L_cas_n_bias: stacked NMOS-diode pair sizing.
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

    # R89 B delta: Mgf2_hybrid draws additional DC current at out.
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref
    K_cs3_load_total = int(round(K_cs3_load * (1 + Mgf2_scale)))
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

    # R89 B delta: Mgf2_hybrid (g=d1) sizing.
    if L_gf2 is None:
        L_gf2 = L_cs3
    if gm_id_gf2 is not None:
        W_gf2 = closed_form_W("pmos", gm_id=gm_id_gf2,
                              I_d=I_gf2, L=L_gf2).W
    elif W_gf2 is None:
        W_gf2 = W_cs3 * Mgf2_scale

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # Bias chain
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

    # Cascode bias rail (pcas)
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

    # NMOS mirror-cascode bias rail (ncas)
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

    devices.append(_mosfet("M3", "nmos",
                            d="d1_diode_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos",
                            d="d1_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

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

    # Stage 3: PMOS CS amplifier (output stage)
    devices.append(_mosfet("Mcs3", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))

    # R89 B delta: hybrid d1-gated Mgf2-style active feedforward
    # (PMOS, g=d1, d=out, s=vdd). R88 C NMCNR inherited pattern.
    devices.append(_mosfet("Mgf2_hybrid", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # Stage-3 NMOS sink widened by (1 + Mgf2_scale).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # AZC delta: triple-MOSFET archetype trio (PRESERVED VERBATIM)
    devices.append(_mosfet("Mazc1", "nmos",
                            d="nazc1", g="d2", s="vss", b="vss",
                            W=W_azc, L=L_azc))
    devices.append(_mosfet("Mazc2_diode", "pmos",
                            d="nazc1", g="nazc1", s="vdd", b="vdd",
                            W=W_azc, L=L_azc))
    devices.append(_mosfet("Mazc3", "nmos",
                            d="nazc_hub", g="nazc_cap2", s="vss", b="vss",
                            W=W_azc, L=L_azc))

    # AZC delta: quad-R nulling-resistor mesh (PRESERVED VERBATIM)
    devices.append(_res("Razc0", p="nazc_bias", n="nazc_cap0",
                        value=Razc0))
    devices.append(_res("Razc1", p="nazc_bias", n="nazc_cap1",
                        value=Razc1))
    devices.append(_res("Razc2", p="nazc_hub",  n="nazc_cap1",
                        value=Razc2))
    devices.append(_res("Razc3", p="nazc_hub",  n="nazc_cap2",
                        value=Razc3))

    # AZC delta: triple-cap compensation network (PRESERVED VERBATIM)
    devices.append(_cap("Cazc0", p="out", n="nazc_bias", value=Cazc0))
    devices.append(_cap("Cazc1", p="nazc_cap1", n="vss", value=Cazc1))
    devices.append(_cap("Cazc2", p="nazc_cap2", n="vss", value=Cazc2))

    # R89 B engineering addition: optional out<->d2 NMCF-style outer
    # Miller for GBW tune. Disabled by default (Cm_helper = 0). When
    # > 0, provides a direct Miller path from out to stage-2 output,
    # slowing dominant pole to fit paper GBW window.
    if Cm_helper > 0:
        devices.append(_cap("Cm_helper", p="out", n="d2",
                            value=Cm_helper))

    # DC bias paths for AZC internal nodes.
    devices.append(_mosfet("Mbp_azc_bias", "pmos",
                            d="nazc_bias", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbn_azc_hub", "nmos",
                            d="nazc_hub", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mbn_azc_cap0", "nmos",
                            d="nazc_cap0", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # R118 C: optional DRC annotations
    # MATCHING (diff pair) / MATCHING (NMOS mirror cascode) /
    # PASSIVE_MATCHING (Cazc1, Cazc2). See kwarg docs above for the
    # per-annotation rationale.
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
                    "PMOS diff pair — Vth mismatch maps directly to "
                    "input-referred offset. AZC has no auto-zero loop "
                    "so diff-pair mismatch propagates unattenuated "
                    "through stage 1 and then through the d1->out "
                    "Mgf2_hybrid (R89 B direct-graft d1-FF) path"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas3_n", "Mcas4_n"),
            extras={
                "rationale": (
                    "NMOS mirror cascode pair (R85 C / R84 D recipe) "
                    "closes a DFCFC1-style regulated-cascode feedback "
                    "through d1_diode. Mismatch unbalances the loop "
                    "and re-introduces offset at d1, partially "
                    "cancelling the cascode gain product the AZC "
                    "direct-graft retrofit relies on"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cazc1", "Cazc2"),
            extras={
                "rationale": (
                    "AZC inner-cap pair — both terminate at vss and "
                    "form the symmetric pair of the AZC triple-cap "
                    "compensation network; their ratio sets two of "
                    "the three nazc_cap poles in the AZC mesh-"
                    "dominated pole structure, so process-tracking "
                    "matching is required to keep the dominant-pole "
                    "location predictable across PVT"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))

    return cell
