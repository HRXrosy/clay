"""
Built-in template: nested_miller_3stage_lec_cascoded_input_mirror.

Qu LEC 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** added on top of the local-embedded-
compensation LEC backbone. R86 D batch rollout #4 — applies the
R85 C cross-variant cascode recipe to LEC's local-RC compensation
network (Mlea + Mleb + Rlec + Clec_gnd at internal node nlec).

Why this variant
================

  1. LEC has a **standard NMCF stage 1** (plain PMOS diff-pair +
     NMOS diode mirror). The R85 C cascode recipe drops in
     verbatim.
  2. LEC's distinguishing compensation lives downstream (Mlea
     PMOS local error amp + Mleb NMOS local buffer + Rlec + cap
     to vss Clec_gnd + Cm1 skip-over routing).
  3. LEC paper GBW target is **3 MHz** (per AnalogGym
     GraphAMPLEC.GBW_target). Cm1 estimate 5 pF × (1/3) ≈ 1.67 pF.
  4. LEC's Mgf_lec (active FF, gate=d1, drain=out) is same shape
     as ACBC/DFCFC2/AFFC/IAC's primary FF. Mgf_lec_scale = 0.1
     inherited from R85 C.

Architectural devices preserved verbatim
=========================================

  1. **Mlea** (PMOS, g=d1, d=nlec, s=vdd) — local error amplifier.
  2. **Mlcs_nlec** (NMOS sink at nlec) — DC current path for Mlea.
  3. **Mleb** (NMOS, g=nlec, d=nleb, s=vss) — local error buffer.
  4. **Mlcs_leb** (PMOS source at nleb) — DC current path for Mleb.
  5. **Clec_gnd** (cap, p=nlec, n=vss) — LEC architectural signature
     (compensation cap to vss).
  6. **Rlec** (res, p=nlec, n=nleb) — local resistor.
  7. **Mgf_lec** (PMOS, g=d1, d=out) — active feedforward.
  8. **Compensation**: Cm1 (out↔d1 skip-over only; no Cm2, no Cf).

R85 C cascode recipe added
==========================

Same telescopic PMOS input cascode + cascoded NMOS mirror as R85 C.
Mirror diode loop closes at high-Z node ``d1_diode``.

Honest scope
============

LEC paper GBW = 3 MHz. Initial Cm1 estimate 1.67 pF. The local-RC
network (Clec_gnd + Rlec at nlec/nleb) is small-signal effect at
AC band; doesn't disturb DC gain expectation of ~120-128 dB.

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


def build_nested_miller_3stage_lec_cascoded_input_mirror(
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

    # ─── Compensation: skip-over Cm1 only ───────────────────
    Cm1: float = 1.67e-12,

    # ─── LEC-specific: active FF Mgf_lec ───────────────────
    Mgf_lec_scale: float = 0.1,
    W_gf_lec: float | None = None,
    L_gf_lec: float | None = None,
    gm_id_gf_lec: float | None = None,

    # ─── LEC-specific: local error amplifier Mlea (PMOS) ────
    Mlea_scale: float = 1.0,
    W_lea: float | None = None,
    L_lea: float | None = None,
    gm_id_lea: float | None = None,

    # ─── LEC-specific: local error buffer Mleb (NMOS) ───────
    Mleb_scale: float = 1.0,
    W_leb: float | None = None,
    L_leb: float | None = None,
    gm_id_leb: float | None = None,

    # ─── LEC-specific: local R-C filter ─────────────────────
    Rlec: float = 10e3,
    Clec_gnd: float = 0.5e-12,

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

    # ─── R117 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # cascoded LEC 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1, M2). Vth mismatch maps
    #     to input-referred offset; the LEC local-RC error loop
    #     amplifies any residual imbalance at d1.
    #   * GUARD_RING on Mtail (PMOS tail, M=N_tail). Carries the entire
    #     stage-1 bias current; nwell guard ring tied to vdd isolates
    #     the high-Z d1 node — which drives both Mgf_lec (active FF)
    #     and Mlea (local error amp) — from supply-coupled substrate
    #     noise.
    #   * MATCHING on the PMOS input cascode pair (Mcas1, Mcas2). The
    #     R85 C telescopic input cascode lifts stage-1 output impedance
    #     only when Mcas1/Mcas2 share Vds bias; mismatch unbalances the
    #     cascode and partially cancels the +15-20 dB gain boost that
    #     justifies the cascode recipe.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Qu LEC 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.

    Same 5-port surface as the plain LEC cell. The LEC local-RC
    signature (Mlea + Mleb + Rlec + Clec_gnd cap-to-vss + Cm1
    skip-over) is preserved verbatim.
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # LEC: Mgf_lec draws additional DC current at out.
    I_gf_lec = Mgf_lec_scale * K_cs3_load * I_ref
    K_cs3_load_total = K_cs3_load * (1 + Mgf_lec_scale)
    if K_cs3_load_total < 1:
        K_cs3_load_total = 1

    # LEC: Mlea + Mleb each draw additional DC current at nlec / nleb.
    I_lea = Mlea_scale * K_cs3_load * I_ref
    I_leb = Mleb_scale * I_cs2
    K_nlec_sink = max(int(round(I_lea / I_ref)), 1)
    K_leb_sink = max(int(round(I_leb / I_ref)), 1)

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

    # Mgf_lec sizing
    if L_gf_lec is None:
        L_gf_lec = L_cs3
    if gm_id_gf_lec is not None:
        W_gf_lec = closed_form_W("pmos", gm_id=gm_id_gf_lec,
                                 I_d=I_gf_lec, L=L_gf_lec).W
    elif W_gf_lec is None:
        W_gf_lec = W_cs3 * Mgf_lec_scale

    # Mlea sizing
    if L_lea is None:
        L_lea = L_cs3
    if gm_id_lea is not None:
        W_lea = closed_form_W("pmos", gm_id=gm_id_lea,
                              I_d=I_lea, L=L_lea).W
    elif W_lea is None:
        W_lea = W_cs3 * Mlea_scale

    # Mleb sizing
    if L_leb is None:
        L_leb = L_cs2
    if gm_id_leb is not None:
        W_leb = closed_form_W("nmos", gm_id=gm_id_leb,
                              I_d=I_leb, L=L_leb).W
    elif W_leb is None:
        W_leb = W_cs2 * Mleb_scale

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

    # ─── LEC delta: Local error amplifier Mlea ──────────────
    devices.append(_mosfet("Mlea", "pmos", d="nlec", g="d1", s="vdd", b="vdd",
                            W=W_lea, L=L_lea))
    devices.append(_mosfet("Mlcs_nlec", "nmos",
                            d="nlec", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_nlec_sink))

    # ─── LEC delta: Local error buffer Mleb ─────────────────
    devices.append(_mosfet("Mleb", "nmos", d="nleb", g="nlec", s="vss", b="vss",
                            W=W_leb, L=L_leb))
    devices.append(_mosfet("Mlcs_leb", "pmos",
                            d="nleb", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=K_leb_sink))

    # ─── LEC delta: Local R-C filter ────────────────────────
    devices.append(_cap("Clec_gnd", p="nlec", n="vss", value=Clec_gnd))
    devices.append(_res("Rlec",     p="nlec", n="nleb", value=Rlec))

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

    # ─── LEC delta: Active feedforward Mgf_lec ──────────────
    devices.append(_mosfet("Mgf_lec", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf_lec, L=L_gf_lec))

    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── Compensation: Cm1 skip-over only ───────────────────
    devices.append(_cap("Cm1", p="out", n="d1", value=Cm1))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R117 C: optional DRC annotations ─────────────────
    # MATCHING / GUARD_RING / MATCHING (PMOS input cascode). See
    # kwarg docs above for the per-annotation rationale.
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
                    "input-referred offset. The LEC local-RC error "
                    "loop (Mlea + Mleb + Rlec + Clec_gnd) amplifies "
                    "any residual imbalance at d1, so common-centroid "
                    "matching is required"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "PMOS tail current source (M=N_tail) carries the "
                    "entire stage-1 bias current; nwell guard ring tied "
                    "to vdd isolates the high-Z d1 node — which drives "
                    "both Mgf_lec (active FF, g=d1) and Mlea (local "
                    "error amp PMOS, g=d1) — from supply-coupled "
                    "substrate noise"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas1", "Mcas2"),
            extras={
                "rationale": (
                    "PMOS input cascode pair (R85 C telescopic recipe) "
                    "— lifts stage-1 output impedance only when "
                    "Mcas1/Mcas2 share Vds bias. Mismatch unbalances "
                    "the cascode and partially cancels the +15-20 dB "
                    "DC gain boost that justifies the cascode recipe"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
