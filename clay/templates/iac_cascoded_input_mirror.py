"""
Built-in template: nested_miller_3stage_iac_cascoded_input_mirror.

Peng IAC 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** added on top of the indirect-AC IAC
backbone. R86 D batch rollout #2 — applies the R85 C cross-variant
cascode recipe (telescopic PMOS input cascode + DFCFC1-style
regulated NMOS mirror with diode loop at high-Z node ``d1``) to a
STRUCTURALLY DIFFERENT NMCF-class variant: IAC's indirect AC
compensation via triode-region helper Mgmt + RC nulling chain.

Why this variant
================

  1. IAC has a **standard NMCF stage 1** (plain PMOS diff-pair +
     NMOS diode mirror) — the R85 C cascode recipe drops in
     verbatim with NO topology re-engineering.
  2. IAC's distinguishing compensation lives in the compensation
     network (triode helper Mgmt + niac/npath/vcas_iac/n_rc_iac
     nodes + Cm2_iac series cap + R0_iac nulling-R), all downstream
     of stage 1.
  3. IAC paper GBW target is **3.5 MHz** (per AnalogGym
     GraphAMPIAC.GBW_target). Cm1_iac scales as 1/GBW from R84 D
     TCFC's 5 pF → Cm1_iac ≈ 5 pF × (1/3.5) ≈ 1.4 pF estimate.
  4. IAC's Mgmf (active FF, gate=d1, drain=out) is same shape as
     ACBC/DFCFC2/AFFC's Mgf2; same DC near-cancellation risk at
     full scale; Mgmf_scale = 0.1 inherited from R85 C.

Architectural devices preserved verbatim
=========================================

  1. **Mgmt** (PMOS, d=niac, g=vcas_iac, s=npath) — triode-helper
     for indirect AC routing.
  2. **Mbp_vcas_iac** (PMOS diode at vcas_iac) — Mgmt gate bias.
  3. **Mbp_npath** (PMOS diode at npath) — Mgmt source rail.
  4. **Mbn_niac** (NMOS sink at niac) — DC current path for Mgmt.
  5. **Mgmf** (PMOS, g=d1, d=out, s=vdd) — primary active
     feedforward.
  6. **Compensation**: Cm1_iac (out↔d2 outer Miller, NMCF-style)
     + Cm2_iac (niac↔n_rc_iac indirect cap) + R0_iac (n_rc_iac↔vss
     nulling-R).

R85 C cascode recipe added
==========================

Same telescopic PMOS input cascode + cascoded NMOS mirror as R85 C.
Mirror diode loop closes at high-Z node ``d1_diode`` (M3, M4 with
g=d1_diode).

Honest scope
============

IAC paper GBW = 3.5 MHz. Initial Cm1_iac estimate 1.4 pF (5 pF / 3.5).
Mgmf_scale = 0.1 (R85 C inherited tune). The IAC RC nulling network
(Cm2_iac + R0_iac at niac/n_rc_iac) is small-signal effect at AC
band; doesn't disturb DC gain expectation of ~120-128 dB from the
double-cascode + plain IAC stages 2/3.

All seven W's (LUT/heuristic) BYTE-EQUAL with R85 C v123.
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


def build_nested_miller_3stage_iac_cascoded_input_mirror(
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

    # ─── IAC-specific: triode helper Mgmt ───────────────────
    W_gmt: float = 8e-6,
    L_gmt: float = 1e-6,

    # ─── IAC-specific: active FF Mgmf ───────────────────────
    Mgmf_scale: float = 0.1,
    W_gmf: float | None = None,
    L_gmf: float | None = None,
    gm_id_gmf: float | None = None,

    # ─── IAC-specific: compensation network ─────────────────
    Cm1_iac: float = 1.4e-12,
    Cm2_iac: float = 0.5e-12,
    R0_iac: float = 10e3,

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

    # ─── R120 C: optional qualitative DRC annotations ───────
    # When True, attach three layout-prompt annotations capturing the
    # cascoded-input IAC's pair-match constraints. Unlike plain IAC,
    # this variant inserts a PMOS input cascode pair (Mcas1/Mcas2)
    # and an NMOS mirror cascode pair (Mcas3_n/Mcas4_n) — both pairs
    # are pair-match critical for the cascode gain product. We pick:
    #   * MATCHING (Mcas1, Mcas2) — telescopic PMOS input cascode;
    #     Vds mismatch unbalances the cascode output impedance and
    #     re-introduces stage-1 offset at d1.
    #   * CURRENT_MIRROR_MATCHING (M3, M4) — stage-1 NMOS mirror
    #     (diode at d1_diode, mirror at d1) — same offset/imbalance
    #     concern as plain IAC, but here the cascoded mirror amplifies
    #     the gain consequence of any imbalance.
    #   * MATCHING (Mcas3_n, Mcas4_n) — NMOS mirror cascode pair
    #     (gates closed at d1_diode, DFCFC1-style diode loop).
    #     Mismatch unbalances the cascode output impedance and erodes
    #     the +15-20 dB cascoded-mirror gain boost.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Peng IAC 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.

    Same 5-port surface as the plain IAC cell. The IAC indirect-AC
    signature (Mgmt + niac/npath/vcas_iac + Cm2_iac + R0_iac +
    Mgmf + Cm1_iac out↔d2) is preserved verbatim.
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # IAC: Mgmf draws additional DC current at out.
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

    # Mgmf sizing
    if L_gmf is None:
        L_gmf = L_cs3
    if gm_id_gmf is not None:
        W_gmf = closed_form_W("pmos", gm_id=gm_id_gmf,
                              I_d=I_gmf, L=L_gmf).W
    elif W_gmf is None:
        W_gmf = W_cs3 * Mgmf_scale

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

    # ─── IAC delta: bias rails for indirect compensation ────
    devices.append(_mosfet("Mbp_vcas_iac", "pmos",
                            d="vcas_iac", g="vcas_iac", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbp_npath", "pmos",
                            d="npath", g="npath", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # ─── IAC delta: triode-helper Mgmt ──────────────────────
    devices.append(_mosfet("Mgmt", "pmos",
                            d="niac", g="vcas_iac", s="npath", b="vdd",
                            W=W_gmt, L=L_gmt))

    # NMOS sink at niac so Mgmt's drain has a DC path.
    devices.append(_mosfet("Mbn_niac", "nmos",
                            d="niac", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # ─── IAC delta: active feedforward Mgmf ─────────────────
    devices.append(_mosfet("Mgmf", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gmf, L=L_gmf))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── IAC compensation network ───────────────────────────
    devices.append(_cap("Cm1_iac", p="out", n="d2", value=Cm1_iac))
    devices.append(_cap("Cm2_iac", p="niac", n="n_rc_iac", value=Cm2_iac))
    devices.append(_res("R0_iac", p="n_rc_iac", n="vss", value=R0_iac))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R120 C: optional DRC annotations ───────────────────
    # MATCHING (Mcas1, Mcas2) / CURRENT_MIRROR_MATCHING (M3, M4) /
    # MATCHING (Mcas3_n, Mcas4_n). See the kwarg docs block above for
    # the per-annotation rationale.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas1", "Mcas2"),
            extras={
                "rationale": (
                    "PMOS input cascode pair (R85 C recipe) — "
                    "telescopic cascode lifts stage-1 output impedance "
                    "only when Mcas1 / Mcas2 share Vds bias; Vth or "
                    "W/L mismatch unbalances the cascode output "
                    "impedance and re-introduces offset at d1, "
                    "partially cancelling the cascode gain product "
                    "the IAC indirect-AC compensation relies on for "
                    "the 120-128 dB DC gain envelope"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "Stage-1 NMOS mirror — M3 (g=d1_diode) is the "
                    "unit reference, M4 (g=d1_diode, d to mirror "
                    "cascode) is the mirror leg. Cascoded mirror "
                    "amplifies the gain consequence of any mirror "
                    "imbalance: a 1% W/L mismatch at the unit cell "
                    "shows up as a ~1% I_d imbalance through the "
                    "cascode, then a ~20 dB-amplified offset at d1"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas3_n", "Mcas4_n"),
            extras={
                "rationale": (
                    "NMOS mirror cascode pair (R84 D regulated "
                    "cascode) — gates closed at d1_diode (DFCFC1-style "
                    "diode loop). Mismatched Mcas3_n / Mcas4_n "
                    "unbalances the cascode output impedance and "
                    "erodes the +15-20 dB DC gain bump from the "
                    "cascoded NMOS mirror, the architectural payoff "
                    "of the cascoded-input IAC variant"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
