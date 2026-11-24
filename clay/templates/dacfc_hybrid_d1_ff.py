"""
Built-in template: nested_miller_3stage_dacfc_hybrid_d1_ff.

R89 B / BL-093 hybrid d1-gated FF retrofit batch — DACFC variant.
Rolls out R88 C's NMCNR-validated retrofit pattern (cascoded mirror
+ d1-gated Mgf2-style FF → +30-35 dB gain lift) to Song DACFC.

Structural finding (R87 C / R88 C closure)
==========================================

R88 C pilot retrofit on NMCNR validated R87 C's structural finding:
the R85 C cascode recipe REQUIRES a d1-gated Mgf2-style PMOS FF
(PMOS, g=d1, d=out, s=vdd) to realise its +30-35 dB gain product.
NMCNR cascoded baseline 83.29 dB → NMCNR + hybrid d1-FF 119.36 dB
(+36.07 dB lift, GBW landed at 1.230 MHz inside paper +/-20% window).

R89 B DACFC retrofit: dual-FF
==============================

DACFC's R87 C cascoded baseline retains an active FF Mgmf2 (PMOS,
g=d2, d=out, s=vdd) — but Mgmf2 is gated by ``d2`` (stage-2 output)
NOT by ``d1`` (stage-1 output). This means Mgmf2 is part of the
main A1·A2·A3 path. DACFC also has input-coupled FF pair (Mgmf1_n /
Mgmf1_p) that bypasses the diff pair entirely, but these feed
forward INTO stage-1 internal nodes (nga1_a / nga1_b), not directly
to out. DACFC's R87 C measured gain_dc = 84.90 dB confirms neither
FF lifts gain past the 83-87 dB ceiling.

R89 B DACFC retrofit is **dual-FF**: keep DACFC's existing Mgmf2
(g=d2) AND its input-coupled gmf1 pair verbatim, AND add a NEW
Mgf2_hybrid (g=d1) directly grafted from R88 C NMCNR pattern. The
three FF paths operate on different signal sources:

  * Mgmf1_n / Mgmf1_p : g=inn/inp, d=nga1_a/nga1_b — input-coupled
                       FF into stage-1 cascode nodes (unchanged).
  * Mgmf2            : g=d2, d=out — main-path output FF (unchanged).
  * Mgf2_hybrid      : g=d1, d=out — NEW, R88 C inherited; provides
                       the direct d1→out gain path that unlocks the
                       cascoded mirror's +30 dB gain product.

Mlcs3 NMOS output sink widened by (1 + Mgmf2_scale + Mgf2_scale)
to absorb both Mgmf2 and Mgf2_hybrid DC currents.

Architectural devices preserved verbatim
=========================================

DACFC fingerprint that survives R89 B retrofit:

  1. **Mga1_a** (PMOS, d=nga1_a, g=bias_d2, s=vdd) — matched gma1.
  2. **Mga1_b** (PMOS, d=nga1_b, g=bias_d2, s=vdd) — matched pair.
  3. **Mgmf1_n** (PMOS, d=nga1_a, g=inn, s=vdd) — input-coupled FF.
  4. **Mgmf1_p** (PMOS, d=nga1_b, g=inp, s=vdd) — input-coupled FF.
  5. **Mga2** (PMOS, d=nga2, g=bias_d2, s=vdd) — single buffer.
  6. **Mgmf2** (PMOS, d=out, g=d2, s=vdd) — original DACFC output FF.
  7. **Cm1** (cap, out<->d2) — NMCF-style outer Miller.
  8. **Cm2_dac** (cap, nga1_a<->vss) — inner cap to vss.
  9. **DC bias sinks** Mlcs_ga1_a, Mlcs_ga1_b, Mlcs_ga2.

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
  * **Mlcs3** sized by K_cs3_load * (1 + Mgmf2_scale + Mgf2_scale)
    — wider NMOS sink absorbs DC current from BOTH Mgmf2 and
    Mgf2_hybrid at the output node.

If gain_dc lands in [115, 120] dB AND GBW stays in [1.60, 2.40] MHz,
R88 C structural finding generalises across DACFC's doubly-
differential FF backbone. M-SIZING-PAPER-GRADE 9/17 → 10/17 (DACFC
counted as PASS).

Honest scope
============

DACFC paper recipe has Mgmf2 (g=d2) + input-coupled gmf1 pair as
its complete FF network. Adding Mgf2_hybrid (g=d1) is an
architectural cross-graft, not paper replication. Three-FF
concurrent operation is a Clay engineering choice — documented as
additive R89 B delta. The input-coupled gmf1 pair adds DACFC-
specific complexity vs AZ's simpler dual-FF case; the doubly-
differential FF should still bypass the diff pair as before since
Mgf2_hybrid is independent.
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


def build_nested_miller_3stage_dacfc_hybrid_d1_ff(
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

    # Compensation cap (NMCF-style outer)
    Cm1: float = 2.0e-12,    # outer Miller (out <-> d2)

    # DACFC-specific: matched gma1 PMOS pair (cascode)
    W_gma1: float = 8e-6,
    L_gma1: float = 0.5e-6,

    # DACFC-specific: input-coupled gmf1_PMOS pair
    W_gmf1_in: float = 8e-6,
    L_gmf1_in: float = 1e-6,

    # DACFC-specific: single gma2 PMOS buffer
    W_gma2: float = 4e-6,
    L_gma2: float = 1e-6,

    # DACFC-specific: original output-stage FF Mgmf2 (g=d2)
    Mgmf2_scale: float = 0.1,
    W_gmf2: float | None = None,
    L_gmf2: float | None = None,
    gm_id_gmf2: float | None = None,

    # DACFC-specific: inner cap to vss
    Cm2_dac: float = 0.5e-12,

    # R89 B hybrid d1-FF retrofit (R88 C inherited DFCFC1/DFCFC2 Mgf2)
    Mgf2_scale: float = 0.1,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

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
    # DACFC + R89 B hybrid d1-FF dual-FF retrofit. DACFC's
    # doubly-differential FF backbone exposes THREE matched device
    # pairs that all sit on critical mismatch-amplified paths:
    #   * MATCHING on the PMOS diff pair (M1, M2). Vth mismatch maps
    #     directly to input-referred offset; the new d1-FF
    #     (Mgf2_hybrid) directly amplifies d1 mismatch to out at
    #     +30 dB, on top of the original Mgmf2 (g=d2) DACFC path.
    #   * MATCHING on the DACFC matched gma1 PMOS pair (Mga1_a,
    #     Mga1_b). These active-cascode buffers drive nga1_a/nga1_b
    #     symmetric stage-1 cascode nodes — mismatch unbalances the
    #     DACFC fully-differential cascode and re-injects offset
    #     into the d2 path.
    #   * MATCHING on the DACFC input-coupled FF pair (Mgmf1_n,
    #     Mgmf1_p). These PMOS bypass the diff pair from
    #     inn/inp directly into nga1_a/nga1_b stage-1 internal
    #     nodes; mismatch in this FF pair injects differential
    #     offset upstream of stage-1's auto-cancellation and
    #     contaminates the doubly-differential FF backbone.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for Song DACFC 3-stage opamp with R85 C cascode
    recipe, doubly-differential FF network PRESERVED, AND a NEW
    Mgf2_hybrid (g=d1) grafted from R88 C NMCNR pattern (R89 B
    dual-FF retrofit).

    DACFC signature (matched gma1 pair + input-coupled gmf1 FF pair
    + Mga2 buffer + Mgmf2 output FF) preserved verbatim. NEW:
    Mgf2_hybrid (PMOS, g=d1, d=out) + Mlcs3 widened by (1 +
    Mgmf2_scale + Mgf2_scale).

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain.
        W_diff..L_cs3: per-device physical sizing.
        gm_id_diff..gm_id_stage3: gm/Id intent overrides.
        Cm1: outer Miller (out ↔ d2). Default 2.0 pF.
        W_gma1, L_gma1: matched gma1 PMOS pair (active cascode).
        W_gmf1_in, L_gmf1_in: input-coupled gmf1 PMOS pair.
        W_gma2, L_gma2: single gma2 PMOS buffer.
        Mgmf2_scale: original DACFC Mgmf2 (g=d2) scale. Default 0.1.
        W_gmf2, L_gmf2, gm_id_gmf2: original Mgmf2 sizing overrides.
        Cm2_dac: inner cap to vss (nga1_a ↔ vss). Default 0.5 pF.
        Mgf2_scale: R89 B hybrid d1-FF scale. Default 0.1.
        W_gf2, L_gf2, gm_id_gf2: Mgf2_hybrid sizing overrides.
        W_cas, L_cas, I_ref_cas: PMOS input-cascode pair sizing.
        W_cas_n_mirror, L_cas_n_mirror: NMOS mirror-cascode pair.
        W_cas_n_bias, L_cas_n_bias: stacked NMOS-diode pair.
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

    # DACFC original Mgmf2 draws DC current at out (preserved).
    I_gmf2 = Mgmf2_scale * K_cs3_load * I_ref
    # R89 B delta: NEW Mgf2_hybrid draws additional DC current at out.
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref
    K_cs3_load_total = int(round(
        K_cs3_load * (1.0 + Mgmf2_scale + Mgf2_scale)
    ))
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

    # Original Mgmf2 (g=d2) sizing.
    if L_gmf2 is None:
        L_gmf2 = L_cs3
    if gm_id_gmf2 is not None:
        W_gmf2 = closed_form_W("pmos", gm_id=gm_id_gmf2,
                               I_d=I_gmf2, L=L_gmf2).W
    elif W_gmf2 is None:
        W_gmf2 = W_cs3 * Mgmf2_scale

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

    # Stage 2 replica-bias leg (bias_d2 reused as DACFC gma1 gate)
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

    # DACFC delta: matched gma1 PMOS pair (active cascode) — PRESERVED
    devices.append(_mosfet("Mga1_a", "pmos",
                            d="nga1_a", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))
    devices.append(_mosfet("Mga1_b", "pmos",
                            d="nga1_b", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))

    # DACFC delta: input-coupled gmf1_PMOS pair — PRESERVED
    devices.append(_mosfet("Mgmf1_n", "pmos",
                            d="nga1_a", g="inn", s="vdd", b="vdd",
                            W=W_gmf1_in, L=L_gmf1_in))
    devices.append(_mosfet("Mgmf1_p", "pmos",
                            d="nga1_b", g="inp", s="vdd", b="vdd",
                            W=W_gmf1_in, L=L_gmf1_in))

    # DACFC delta: single gma2 PMOS buffer — PRESERVED
    devices.append(_mosfet("Mga2", "pmos",
                            d="nga2", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma2, L=L_gma2))

    # DACFC delta: output-stage FF Mgmf2 (g=d2 not d1) — PRESERVED.
    devices.append(_mosfet("Mgmf2", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf2, L=L_gmf2))

    # R89 B delta: hybrid d1-gated Mgf2-style active feedforward
    # (PMOS, g=d1, d=out, s=vdd). R88 C NMCNR inherited pattern.
    devices.append(_mosfet("Mgf2_hybrid", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # DC bias sinks for the cascode / buffer drain nodes
    devices.append(_mosfet("Mlcs_ga1_a", "nmos",
                            d="nga1_a", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga1_b", "nmos",
                            d="nga1_b", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga2", "nmos",
                            d="nga2", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf2 + Mgf2_hybrid
    # DC currents (R89 B delta: total scale is now 1 + Mgmf2_scale +
    # Mgf2_scale).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # Compensation: Cm1 outer (NMCF-style) + Cm2_dac to vss
    devices.append(_cap("Cm1",     p="out",    n="d2",  value=Cm1))
    devices.append(_cap("Cm2_dac", p="nga1_a", n="vss", value=Cm2_dac))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # R118 C: optional DRC annotations
    # Three MATCHING annotations covering the diff pair, the DACFC
    # matched gma1 PMOS active-cascode pair, and the DACFC input-
    # coupled FF pair. See kwarg docs above for the per-annotation
    # rationale.
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
                    "input-referred offset. The R89 B d1-FF "
                    "(Mgf2_hybrid, g=d1, d=out) directly amplifies d1 "
                    "mismatch to out at +30 dB, on top of the "
                    "original Mgmf2 (g=d2) DACFC path"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mga1_a", "Mga1_b"),
            extras={
                "rationale": (
                    "DACFC matched gma1 PMOS pair (active cascode) — "
                    "drives the symmetric stage-1 cascode nodes "
                    "nga1_a/nga1_b; mismatch unbalances the DACFC "
                    "fully-differential cascode and re-injects "
                    "offset into the d2 path via Mga2 buffer"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mgmf1_n", "Mgmf1_p"),
            extras={
                "rationale": (
                    "DACFC input-coupled FF pair — Mgmf1_n (g=inn) / "
                    "Mgmf1_p (g=inp) bypass the diff pair from "
                    "inn/inp directly into nga1_a/nga1_b stage-1 "
                    "internal nodes; mismatch injects differential "
                    "offset upstream of stage-1 auto-cancellation and "
                    "contaminates the doubly-differential FF backbone"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
