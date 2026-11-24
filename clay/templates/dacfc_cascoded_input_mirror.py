"""
Built-in template: nested_miller_3stage_dacfc_cascoded_input_mirror.

Song DACFC 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** following the R85 C cross-variant
cascode recipe. R87 C batch rollout #5 — applies the R85 C cascode
recipe (telescopic PMOS input cascode + DFCFC1-style regulated NMOS
mirror with diode loop at high-Z node ``d1``) to DACFC's doubly-
differential feedforward backbone.

Why this variant
================

DACFC has plain NMCF stage 1 (PMOS diff pair + NMOS diode mirror) -
R85 C cascode recipe drops in verbatim. DACFC's distinguishing
compensation is doubly-differential:

  1. Matched gma1 PMOS pair (Mga1_a + Mga1_b, gated by bias_d2).
  2. Input-coupled gmf1_PMOS pair (Mgmf1_n + Mgmf1_p, gated by
     inn / inp respectively). Feeds forward from inputs directly,
     bypassing the diff pair.
  3. Single gma2 PMOS buffer (g=bias_d2).
  4. Output-stage FF Mgmf2 (PMOS, g=d2, d=out) - same role as
     AFFC's Mgf2 BUT gated by d2 not d1.
  5. NMCF-style outer Miller Cm1 (out<->d2) + Cm2_dac (nga1_a<->vss)
     inner cap to vss.

DACFC paper GBW: AnalogGym does not list DACFC in ckt_graphs.py;
the design_variables file uses C0=1.5 pF and C1=0.7 pF with
500 pF load. Use ~2 MHz as conservative target (between ACBC's
1.5 MHz and IAC's 3.5 MHz).

Architectural devices preserved verbatim
=========================================

DACFC fingerprint that survives R85 C cascode swap:

  1. **Mga1_a** (PMOS, d=nga1_a, g=bias_d2, s=vdd).
  2. **Mga1_b** (PMOS, d=nga1_b, g=bias_d2, s=vdd) - matched pair.
  3. **Mgmf1_n** (PMOS, d=nga1_a, g=inn, s=vdd).
  4. **Mgmf1_p** (PMOS, d=nga1_b, g=inp, s=vdd) - input-coupled FF.
  5. **Mga2** (PMOS, d=nga2, g=bias_d2, s=vdd) - single buffer.
  6. **Mgmf2** (PMOS, d=out, g=d2, s=vdd) - output FF (g=d2 not d1).
  7. **Cm1** (cap, out<->d2) NMCF-style outer Miller.
  8. **Cm2_dac** (cap, nga1_a<->vss) inner cap to vss.
  9. **DC bias sinks** Mlcs_ga1_a, Mlcs_ga1_b, Mlcs_ga2.

R85 C cascode recipe applied
============================

  * Mcas1 / Mcas2 PMOS cascode pair on input.
  * Mcas_b + Iref_cas - pcas rail bias.
  * Mcas3_n / Mcas4_n NMOS cascode pair on mirror legs. M3/M4
    gates closed at ``d1_diode``.
  * Mcas_n_b1 + Mcas_n_b2 + Mbp_ncas_drv ncas rail.

Honest scope
============

DACFC's input-coupled FF pair bypasses the diff pair entirely;
this creates a parallel signal path that may not be helped much
by the cascoded mirror's boost at d1. Expected gain_dc ~120-128 dB
mostly from cascoded mirror; the FF path's contribution is largely
unchanged. GBW retune required as cascoded mirror moves the
dominant pole.

All seven W's BYTE-EQUAL with R86 D v124 / R85 C v123.
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


def build_nested_miller_3stage_dacfc_cascoded_input_mirror(
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

    # DACFC-specific: output-stage FF Mgmf2
    Mgmf2_scale: float = 0.1,
    W_gmf2: float | None = None,
    L_gmf2: float | None = None,
    gm_id_gmf2: float | None = None,

    # DACFC-specific: inner cap to vss
    Cm2_dac: float = 0.5e-12,

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

    # ─── R114 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # DACFC cascoded-input-mirror variant:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Tail current source sits adjacent to the NMOS mirror legs and
    #     the cascoded NMOS-cascode pair; nwell guard ring tied to vdd
    #     keeps supply-coupled substrate noise off the high-Z d1 node.
    #   * SYMMETRIC_LAYOUT on the DACFC input-coupled feedforward pair
    #     (Mgmf1_n / Mgmf1_p). DACFC's distinguishing structural feature
    #     is a PMOS pair whose gates tap inn / inp directly — drains
    #     fold into nga1_a / nga1_b respectively. CMRR / PSRR at the
    #     output is dominated by the symmetry of THIS pair (not just
    #     the diff pair), because the FF pair injects a parallel signal
    #     path that bypasses the diff pair entirely. Razavi §11.3 — for
    #     CMRR the gain-element symmetry must hold at the LAYOUT level;
    #     the FF pair must sit on a vertical mirror axis anchored at the
    #     tail node so the inn / inp drain currents see identical
    #     parasitic environments.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Song DACFC 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.
    DACFC doubly-differential FF signature preserved verbatim.
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # DACFC: Mgmf2 draws additional DC current at out.
    I_gmf2 = Mgmf2_scale * K_cs3_load * I_ref
    K_cs3_load_total = int(round(K_cs3_load * (1 + Mgmf2_scale)))
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

    # Mgmf2 sizing.
    if L_gmf2 is None:
        L_gmf2 = L_cs3
    if gm_id_gmf2 is not None:
        W_gmf2 = closed_form_W("pmos", gm_id=gm_id_gmf2,
                               I_d=I_gmf2, L=L_gmf2).W
    elif W_gmf2 is None:
        W_gmf2 = W_cs3 * Mgmf2_scale

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

    # DACFC delta: matched gma1 PMOS pair (active cascode)
    devices.append(_mosfet("Mga1_a", "pmos",
                            d="nga1_a", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))
    devices.append(_mosfet("Mga1_b", "pmos",
                            d="nga1_b", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))

    # DACFC delta: input-coupled gmf1_PMOS pair (FF pair)
    devices.append(_mosfet("Mgmf1_n", "pmos",
                            d="nga1_a", g="inn", s="vdd", b="vdd",
                            W=W_gmf1_in, L=L_gmf1_in))
    devices.append(_mosfet("Mgmf1_p", "pmos",
                            d="nga1_b", g="inp", s="vdd", b="vdd",
                            W=W_gmf1_in, L=L_gmf1_in))

    # DACFC delta: single gma2 PMOS buffer
    devices.append(_mosfet("Mga2", "pmos",
                            d="nga2", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma2, L=L_gma2))

    # DACFC delta: output-stage FF Mgmf2 (g=d2 not d1)
    devices.append(_mosfet("Mgmf2", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf2, L=L_gmf2))

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

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf2 DC currents.
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

    # ─── R114 C: optional DRC annotations ─────────────────
    # Three annotations spanning MATCHING / GUARD_RING / SYMMETRIC_LAYOUT.
    # See the kwarg docs block above for the per-annotation rationale.
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
                    "PMOS diff pair — Vth mismatch directly maps to "
                    "input-referred offset; common-centroid required"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "PMOS tail current source (M=N_tail) sits adjacent "
                    "to the NMOS mirror and NMOS cascode legs; nwell "
                    "guard ring tied to vdd shields the high-Z d1 node "
                    "from supply-coupled substrate noise"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.SYMMETRIC_LAYOUT,
            devices=("Mgmf1_n", "Mgmf1_p"),
            extras={
                "rationale": (
                    "DACFC input-coupled feedforward pair — gates tap "
                    "inn / inp directly and drains fold into nga1_a / "
                    "nga1_b; CMRR/PSRR at the output is dominated by "
                    "the symmetry of THIS pair (parallel signal path "
                    "bypasses the diff pair); layout must respect a "
                    "vertical mirror axis anchored at the tail node"
                ),
                "axis": "vertical",
                "anchor": "tail",
            },
        ))

    return cell
