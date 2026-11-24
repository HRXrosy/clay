"""
Built-in template: nested_miller_3stage_raffc_cascoded_input_mirror.

Alfio RAFFC 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** added on top of the reverse-direction
RAFFC backbone. R86 D batch rollout #3 — applies the R85 C
cross-variant cascode recipe to RAFFC's reverse-FF compensation
(PMOS Mgf_rev with g=out, d=d1 — direction reversed vs AFFC).

Why this variant
================

  1. RAFFC has a **standard NMCF stage 1** (plain PMOS diff-pair +
     NMOS diode mirror, but M4 widened to absorb Mgf_rev's drain
     current at d1).
  2. The R85 C cascode recipe drops in at stage 1; the M4-widening
     is reapplied AFTER the cascode is added because Mgf_rev's
     drain still lands on d1 (the HIGH-Z signal node above the
     cascode).
  3. RAFFC paper GBW target is **1.2 MHz** (per AnalogGym
     GraphAMPRAFFC.GBW_target). Cm1 estimate from R84 D's 5 pF:
     5 pF × (1/1.2) ≈ 4.17 pF.
  4. RAFFC's Mgf_rev is in the LOCAL feedback direction (g=out,
     d=d1), so its DC effect on gain is different from AFFC-style
     FF — it forms a local negative-feedback loop. Mgf_rev_scale
     default at 0.25 in plain RAFFC, kept here.

Architectural devices preserved verbatim
=========================================

  1. **Mgf_rev** (PMOS, d=d1, g=out, s=vdd) — reverse FF device
     (direction REVERSED vs AFFC).
  2. M4 widened by (1 + Mgf_rev_scale) to absorb Mgf_rev's DC
     drain current at d1. Note: in the cascoded variant the M4
     widening still applies — Mgf_rev's drain pin is d1 (the
     high-Z signal node above the cascode), so M4 (sink leg)
     must absorb the current at d1_n_int through Mcas4_n.
  3. **Compensation**: Cm1 (out↔d2 outer Miller) + Cm2 (d2↔d1
     inner Miller). No Cf.

R85 C cascode recipe added
==========================

Same telescopic PMOS input cascode + cascoded NMOS mirror as R85 C.
Mirror diode loop closes at high-Z node ``d1_diode`` (M3, M4 with
g=d1_diode).

Honest scope
============

RAFFC paper GBW = 1.2 MHz. Initial Cm1 estimate 4.17 pF. The reverse
FF creates a local negative-feedback loop from out back to d1; the
cascoded mirror at d1 raises the impedance at d1, which strengthens
the local feedback. Expect gain_dc to land slightly below the +60-66
dB cascode-product delta seen in TCFC/DFCFC2 because of the local
feedback. Expected landing: 110-125 dB.

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


def build_nested_miller_3stage_raffc_cascoded_input_mirror(
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

    # ─── Compensation caps ──────────────────────────────────
    Cm1: float = 4.17e-12,
    Cm2: float = 0.5e-12,

    # ─── RAFFC-specific: reverse active FF Mgf_rev ──────────
    Mgf_rev_scale: float = 0.25,
    W_gf_rev: float | None = None,
    L_gf_rev: float | None = None,
    gm_id_gf_rev: float | None = None,

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

    # ─── R121 C: qualitative DRC annotation surface ────────
    # Additive metadata. Default False keeps every pre-R121
    # caller byte-for-byte identical. When True, three
    # qualitative hints attach to the cell:
    #   MATCHING         on (M1, M2)         — PMOS diff pair
    #   MATCHING         on (Mcas1, Mcas2)   — PMOS input cascode pair
    #   PASSIVE_MATCHING on (Cm1, Cm2)       — nested Miller caps
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for an Alfio RAFFC 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.

    Same 5-port surface as the plain RAFFC cell. The RAFFC reverse-FF
    signature (Mgf_rev with g=out, d=d1 + M4 widening) is preserved
    verbatim.
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # RAFFC: Mgf_rev draws additional DC current INTO d1; M4 (at d1)
    # must absorb it. Compute the widening factor.
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

    # Mgf_rev sizing
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

    # M3 (diode side) keeps baseline W; M4 (mirror side) WIDENED to
    # absorb Mgf_rev's DC drain current at d1.
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

    # Stage-3 NMOS sink (no Mgf_rev current absorbed here — Mgf_rev's
    # drain is at d1, not out).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load))

    # ─── RAFFC delta: Reverse active feedforward Mgf_rev ─────
    # PMOS gated by out (global output) with drain at d1 (stage-1
    # output) — direction REVERSED vs AFFC.
    devices.append(_mosfet("Mgf_rev", "pmos",
                            d="d1", g="out", s="vdd", b="vdd",
                            W=W_gf_rev, L=L_gf_rev))

    # ─── Compensation: Cm1 outer + Cm2 inner, NO Cf ─────────
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R121 C: optional qualitative DRC annotations ─────────
    # Two MATCHING + one PASSIVE_MATCHING covering the diff pair, the
    # PMOS input cascode pair, and the nested Miller cap pair that
    # together set RAFFC's reverse-FF feedback loop balance. Default
    # attach_drc=False keeps the pre-R121 emit byte-identical.
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
                    "RAFFC's reverse-FF leg (Mgf_rev) injects current "
                    "from out back into d1 — any M1/M2 imbalance "
                    "becomes a static differential current the reverse "
                    "feedback amplifies into a DC offset"
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
                    "Cascode Vth mismatch shifts the M1 vs M2 Vgs "
                    "balance, degrading the double-cascode rout boost "
                    "and partially defeating the high-DC-gain target "
                    "of the R86 D cascoded-input recipe"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cm1", "Cm2"),
            extras={
                "rationale": (
                    "Nested Miller cap pair — Cm1 (outer, out↔d2) and "
                    "Cm2 (inner, d2↔d1). The compensation pole/zero "
                    "split depends on the Cm1/Cm2 ratio tracking over "
                    "process gradients; fringing-cap unit-array layout "
                    "is required so RAFFC's reverse-FF loop sees the "
                    "designed phase margin across PVT"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))

    return cell
