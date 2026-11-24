"""
Built-in template: nested_miller_3stage_pfc_cascoded_input_mirror.

Ramos PFC 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** added on top of the pole-zero-following
PFC backbone. R86 D batch rollout #5 — applies the R85 C cross-variant
cascode recipe to PFC's oversized-FF compensation (PMOS Mgmf1 with
oversize × Mcs3 width).

Why this variant
================

  1. PFC has a **standard NMCF stage 1** (plain PMOS diff-pair +
     NMOS diode mirror) — the R85 C cascode recipe drops in
     verbatim.
  2. PFC's distinguishing compensation is just the OVERSIZED Mgmf1
     (PMOS, g=d1, d=out, W = PFC_oversize × W_cs3) plus widened
     output sink. Trivial structural delta.
  3. PFC paper GBW target is **5 MHz** (per AnalogGym
     GraphAMPPFC.GBW_target, line 3186). Cm1 estimate from R84 D's
     5 pF: 5 pF × (1/5) = 1.0 pF.
  4. PFC's Mgmf1 is the FF device gated by d1 / drained at out —
     same DC cancellation risk as AFFC/ACBC/DFCFC2/IAC/LEC. But
     PFC's signature is the OVERSIZE factor (default 2.0 in plain
     PFC, raising current consumption). For the cascoded variant
     we keep PFC_oversize at a conservative 0.1 — same logic as
     Mgf2_scale=0.1 in R85 C, since the DC cancellation mechanism
     is identical and PFC's "oversize" architectural intent is
     preserved at the documented kwarg level even if numerically
     reduced for DC gain survival.

Architectural devices preserved verbatim
=========================================

  1. **Mgmf1** (PMOS, g=d1, d=out, s=vdd, W=PFC_oversize × W_cs3) —
     oversized active FF. The architectural signature is the
     "PFC_oversize" kwarg presence and routing pattern.
  2. **Mlcs3** widened by (1 + PFC_oversize) at out.
  3. **Compensation**: Cm1 (out↔d2 outer Miller) + Cm2 (d2↔d1
     inner Miller). No Cf.

R85 C cascode recipe added
==========================

Same telescopic PMOS input cascode + cascoded NMOS mirror as R85 C.
Mirror diode loop closes at high-Z node ``d1_diode``.

Honest scope
============

PFC paper GBW = 5 MHz. Initial Cm1 estimate 1.0 pF. PFC_oversize =
0.1 (R85 C-inherited DC-gain-preserving tune; numerically identical
to other variants' Mgf*_scale = 0.1). This is a conservative
deviation from PFC paper's "oversize" intent — preserved
architecturally via the kwarg but tuned for DC gain survival in the
cascoded backbone. R87+ may revisit with the oversize numerical
default raised, e.g. PFC_oversize=0.5-1.0 + Cm1 retune.

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


def build_nested_miller_3stage_pfc_cascoded_input_mirror(
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
    Cm1: float = 1.0e-12,
    Cm2: float = 0.5e-12,

    # ─── PFC-specific: oversized active FF Mgmf1 ────────────
    # R85 C-inherited DC-preserving tune (default 0.1 in cascoded
    # variant vs plain PFC's 2.0). Preserves PFC's architectural
    # "oversize" kwarg semantically.
    PFC_oversize: float = 0.1,
    W_gmf1: float | None = None,
    L_gmf1: float | None = None,
    gm_id_gmf1: float | None = None,

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
    #   MATCHING on (M1, M2)                — PMOS diff pair
    #   MATCHING on (Mcas1, Mcas2)          — PMOS input cascode pair
    #   MATCHING on (Mcas3_n, Mcas4_n)      — NMOS mirror cascode pair
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Ramos PFC 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.

    Same 5-port surface as the plain PFC cell. The PFC oversized-FF
    signature (Mgmf1 + Mlcs3 widening) is preserved verbatim.
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # PFC: Mgmf1 draws additional DC current at out.
    I_gmf1 = PFC_oversize * I_cs3
    K_cs3_load_total = int(round(K_cs3_load * (1 + PFC_oversize)))
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

    # Mgmf1 sizing
    if L_gmf1 is None:
        L_gmf1 = L_cs3
    if gm_id_gmf1 is not None:
        W_gmf1 = closed_form_W("pmos", gm_id=gm_id_gmf1,
                               I_d=I_gmf1, L=L_gmf1).W
    elif W_gmf1 is None:
        W_gmf1 = W_cs3 * PFC_oversize

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

    # ─── PFC delta: Oversized active FF Mgmf1 ───────────────
    devices.append(_mosfet("Mgmf1", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gmf1, L=L_gmf1))

    # NMOS sink at out — widened to absorb both Mcs3 and Mgmf1 DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

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
    # Three MATCHING annotations cover the three stacked matched-pair
    # groups that define the R86 D cascoded-input + cascoded-mirror
    # recipe layered on the PFC backbone. Default attach_drc=False
    # keeps pre-R121 emit byte-identical.
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
                    "Vth mismatch maps to input-referred offset. The "
                    "PFC oversized Mgmf1 path amplifies any stage-1 "
                    "asymmetry into a DC error at out, so common-"
                    "centroid M1/M2 matching is the layout prerequisite"
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
                    "Mismatch in cascode Vth changes the small-signal "
                    "gate-source voltage at M1 vs M2, defeating the "
                    "double-cascode rout boost the R86 D recipe is "
                    "designed to deliver"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas3_n", "Mcas4_n"),
            extras={
                "rationale": (
                    "NMOS mirror cascode pair (DFCFC1-style regulated "
                    "loop closing at d1_diode). Mismatch breaks the "
                    "cascode output-impedance balance between the "
                    "diode-side and mirror-side legs, leaking stage-1 "
                    "differential signal into the mirror's common-mode "
                    "path"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
