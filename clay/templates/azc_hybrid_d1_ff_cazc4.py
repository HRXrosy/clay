"""
Built-in template: nested_miller_3stage_azc_hybrid_d1_ff_cazc4.

R90 C / BL-093 AZC GBW co-redesign via Cazc4 Miller path.

R89 B finding
=============

R89 B v127 AZC retrofit (azc_hybrid_d1_ff_v127.py) achieved:

  * gain_dc = 119.51 dB (PASS, > 115 dB acceptance)
  * GBW    = 4.35 MHz (FAIL, +45% off paper 3.00 MHz target;
              paper window [2.40, 3.60] MHz)

R89 B engineering attempt: add Cm_helper (out <-> d2) NMCF-style outer
Miller cap. Even at Cm_helper = 5.0 pF the dominant pole was pinned by
the d1-FF feedforward zero floor at ~3.67 MHz; GBW could not be pulled
below ~4.35 MHz.

Architectural diagnosis (R89 B follow-up):

  AZC's native quad-R + triple-cap mesh has NO direct out <-> high-Z-
  node Miller path. Cazc0 (out <-> nazc_bias) is a Miller cap, but
  nazc_bias is a low-Z DC bias node — Miller multiplication is small.
  The cascoded mirror raised stage-1 output impedance at d1, and the
  d1-FF Mgf2_hybrid created a fast direct d1 -> out gain path; both
  push the dominant pole UP, not DOWN. Without a Miller cap that sees
  high gain across it, the dominant pole frequency is set by output-
  node load capacitance only and lands ~10x above paper target.

R90 C delta: Cazc4 (nazc1 <-> out)
===================================

nazc1 is the AZC's most accessible high-Z node:

  * Mazc1.drain (NMOS, g=d2, d=nazc1, s=vss) -- pulls down nazc1 with
    transconductance gm_Mazc1.
  * Mazc2_diode.drain (PMOS diode, d=g=nazc1, s=vdd) -- pulls up
    nazc1, configured as diode connection (R = 1/gm_Mazc2).

The nazc1 node is structurally a half-impedance node (~1/gm_Mazc2 due
to the diode tying its DC operating point), which gives Cazc4 a Miller
multiplication ratio comparable to:

  |A_nazc1_to_out| ~ (gm_stage2 / gm_Mazc2_diode) * A_stage3
                  ~ moderate multiplication (factor of ~5-15 vs the
                    near-zero multiplication that Cazc0 sees because
                    nazc_bias is a low-Z bias node).

Adding Cazc4 thus introduces a structurally new Miller path that
lowers the dominant pole frequency without disturbing the d1 -> out
direct path (which sits on the OTHER side of stage 2).

R90 C parameter sweep (single-shot per Cazc4 value):
  * Cazc4 = 0.5, 1.0, 2.0, 3.0, 4.0, 5.0 pF
  * Mgf2_scale = 0.10 (R88 C / R89 B lock-in, byte-equal v126/v127)
  * All other sizing byte-equal v127 (R89 B AZC baseline preserved)

Acceptance:
  * GBW in [2.40, 3.60] MHz (paper 3.00 MHz +/-20%)
  * gain preserved >= 115 dB
  * Honest scope: topology co-design risk — Cazc4 may also affect gain
    via direct feedforward path, and the GBW floor from d1-FF feed-
    forward zero may still pin GBW above target. Empirical sweep
    needed to confirm.

Architectural devices preserved verbatim (byte-equal v127)
==========================================================

R89 B AZC fingerprint that survives R90 C addition:

  1. Mazc1 (NMOS, d=nazc1, g=d2, s=vss) -- AZC charge-redistribution
     archetype trio #1.
  2. Mazc2_diode (PMOS, d=g=nazc1, s=vdd) -- archetype trio #2.
  3. Mazc3 (NMOS, d=nazc_hub, g=nazc_cap2, s=vss) -- archetype #3.
  4. Razc0-Razc3 quad-R nulling-resistor mesh.
  5. Cazc0/1/2 triple-cap compensation network.
  6. DC bias paths Mbp_azc_bias + Mbn_azc_hub + Mbn_azc_cap0.
  7. R85 C cascode recipe (Mcas1/2 PMOS input + Mcas3_n/4_n NMOS
     mirror with DFCFC1-style diode loop at d1_diode).
  8. R88 C / R89 B Mgf2_hybrid (PMOS, g=d1, d=out, s=vdd) and
     Mlcs3 widened by (1 + Mgf2_scale).
  9. Optional Cm_helper (out <-> d2) preserved as kwarg (default 0
     for v128 sweeps; v127 had default 5.0 pF). Empirical sweep
     starts without Cm_helper to isolate Cazc4's effect.

R90 C delta:
  * Cazc4 (cap, nazc1 <-> out) -- the SOLE structural addition.
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


def build_nested_miller_3stage_azc_hybrid_d1_ff_cazc4(
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

    # R90 C delta: Cazc4 Miller cap (nazc1 <-> out). Introduces a new
    # Miller path through the AZC charge-redistribution mesh's high-Z
    # node nazc1, intended to pull the dominant pole frequency down
    # into the paper +/-20% GBW window. Default 0 (disabled, byte-
    # exact v127 behaviour).
    Cazc4: float = 0.0,

    # R89 B hybrid d1-FF retrofit (R88 C inherited DFCFC1/DFCFC2 Mgf2)
    Mgf2_scale: float = 0.1,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # R89 B engineering addition preserved (default 0 for v128 to
    # isolate Cazc4 effect; v127 default 5 pF used Cm_helper alone).
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

    # R122 C: qualitative DRC annotation surface. Additive metadata;
    # default False keeps every pre-R122 caller byte-for-byte identical.
    # When True, three qualitative hints attach to the cell:
    #   MATCHING on (M1, M2)               -- PMOS diff pair
    #   MATCHING on (Mcas1, Mcas2)         -- PMOS input cascode
    #   MATCHING on (Mcas3_n, Mcas4_n)     -- NMOS mirror cascode
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for AZC 3-stage opamp with R89 B retrofit AND a
    new R90 C Cazc4 Miller cap (nazc1 <-> out).

    R89 B AZC v127 signature preserved byte-exact: triple-MOSFET
    archetype + quad-R nulling-mesh + triple-cap network + R85 C
    cascode recipe + Mgf2_hybrid d1-FF. R90 C delta: Cazc4 cap
    introduced between AZC's high-Z node nazc1 and output.

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
        Cazc0: outer Miller-like (out <-> nazc_bias). Default 1.5 pF.
        Cazc1, Cazc2: inner caps to vss. Default 0.5 pF each.
        Cazc4: R90 C delta -- Miller cap (nazc1 <-> out). Default 0
            (disabled, byte-exact v127 behaviour). When > 0, introduces
            a new Miller path through nazc1 (AZC charge-redistribution
            archetype high-Z node) intended to lower the dominant pole
            frequency for GBW tune.
        Mgf2_scale: R89 B hybrid d1-FF scale. Default 0.1.
        W_gf2, L_gf2, gm_id_gf2: Mgf2_hybrid sizing overrides.
        Cm_helper: R89 B engineering addition (out <-> d2). Default 0
            in v128 (was 5 pF in v127 default). Set > 0 only when a
            two-cap combination is being explored.
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

    # R89 B inherited: hybrid d1-gated Mgf2-style active feedforward.
    devices.append(_mosfet("Mgf2_hybrid", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # Stage-3 NMOS sink widened by (1 + Mgf2_scale).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # AZC delta: triple-MOSFET archetype trio (PRESERVED VERBATIM).
    devices.append(_mosfet("Mazc1", "nmos",
                            d="nazc1", g="d2", s="vss", b="vss",
                            W=W_azc, L=L_azc))
    devices.append(_mosfet("Mazc2_diode", "pmos",
                            d="nazc1", g="nazc1", s="vdd", b="vdd",
                            W=W_azc, L=L_azc))
    devices.append(_mosfet("Mazc3", "nmos",
                            d="nazc_hub", g="nazc_cap2", s="vss", b="vss",
                            W=W_azc, L=L_azc))

    # AZC delta: quad-R nulling-resistor mesh (PRESERVED VERBATIM).
    devices.append(_res("Razc0", p="nazc_bias", n="nazc_cap0",
                        value=Razc0))
    devices.append(_res("Razc1", p="nazc_bias", n="nazc_cap1",
                        value=Razc1))
    devices.append(_res("Razc2", p="nazc_hub",  n="nazc_cap1",
                        value=Razc2))
    devices.append(_res("Razc3", p="nazc_hub",  n="nazc_cap2",
                        value=Razc3))

    # AZC delta: triple-cap compensation network (PRESERVED VERBATIM).
    devices.append(_cap("Cazc0", p="out", n="nazc_bias", value=Cazc0))
    devices.append(_cap("Cazc1", p="nazc_cap1", n="vss", value=Cazc1))
    devices.append(_cap("Cazc2", p="nazc_cap2", n="vss", value=Cazc2))

    # R90 C delta: Cazc4 Miller cap (nazc1 <-> out). Introduces a new
    # Miller path through the AZC mesh's high-Z archetype-trio node
    # nazc1. Disabled by default (Cazc4 = 0) for byte-exact v127
    # behaviour; sweep mode runs Cazc4 = 0.5 .. 5 pF to find GBW-target
    # value.
    if Cazc4 > 0:
        devices.append(_cap("Cazc4", p="nazc1", n="out", value=Cazc4))

    # R89 B inherited: optional out <-> d2 NMCF-style outer Miller.
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
                    "AZC's charge-redistribution mesh (Razc0-Razc3 + "
                    "Cazc0/1/2 + R90 C Cazc4 Miller path) routes the "
                    "stage-1 differential through high-Z node nazc1; "
                    "M1/M2 Vth mismatch contaminates the redistribution "
                    "balance and propagates as DC offset through the "
                    "R89 B Mgf2_hybrid d1-FF amplification path"
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
                    "Cascode Vth mismatch shifts the M1 vs M2 Vds "
                    "operating point and defeats the rout boost the "
                    "R83 D cascoded-input recipe contributes — critical "
                    "for the >115 dB gain target preserved through the "
                    "AZC retrofit"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas3_n", "Mcas4_n"),
            extras={
                "rationale": (
                    "NMOS mirror cascode pair (R84 D inherited recipe). "
                    "Mismatch unbalances the cascode rout between the "
                    "d1_diode and d1 legs, leaking stage-1 differential "
                    "into the mirror's common-mode path and modulating "
                    "the AZC mesh's DC operating point at the high-Z "
                    "nazc1 node"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
