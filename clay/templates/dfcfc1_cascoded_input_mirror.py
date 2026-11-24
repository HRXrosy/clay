"""
Built-in template: nested_miller_3stage_dfcfc1_cascoded_input_mirror.

Leung DFCFC1 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** following the R85 C cross-variant
cascode recipe. R87 C batch rollout #1 — applies the R85 C cascode
recipe (telescopic PMOS input cascode + DFCFC1-style regulated NMOS
mirror with diode loop at high-Z node ``d1``) to the DFCFC1 backbone.

Important deviation from plain DFCFC1
=====================================

The plain DFCFC1 template (``clay/templates/leung_dfcfc1.py``) already
carries ITS OWN cascoded NMOS mirror (``Mcas1_n_load`` /
``Mcas1_n_diode`` + ``Mcas_bias`` diode + ``vcas`` rail). That
internal cascode uses **node names** (``d1_int`` / ``d1_diode_int``,
``vcas``) that overlap structurally with the R85 C cascoded-mirror
recipe (which uses ``d1_n_int`` / ``d1_diode_n_int``, ``ncas``).

R87 C strategy: **replace DFCFC1's own cascoded mirror with the R85 C
recipe verbatim**. The R85 C recipe is functionally equivalent — both
boost the output impedance at d1 by g_m_cas * r_o_M4 — but uses the
canonical R85 C node naming + explicit stacked ncas bias rail
(Mcas_n_b1 + Mcas_n_b2 + Mbp_ncas_drv). DFCFC1's architectural
DELTA vs plain NMCF (active feedforward Mgf2) is preserved verbatim
in the compensation network; only the mirror cascode topology is
unified with the cross-variant recipe.

Why this variant
================

DFCFC1's active feedforward Mgf2 (PMOS, g=d1, d=out, s=vdd) is shared
with AFFC and provides the primary LHP zero. Paper GBW = 1.05 MHz.
Initial Cm1 ≈ 4.5 pF (similar order to ACBC's 5 pF for 1.5 MHz).
Mgf2_scale = 0.1 (R85 C inherited tune for DC-preserving active FF).

Architectural devices preserved verbatim
=========================================

DFCFC1 fingerprint that survives R85 C cascode swap:

  1. **Mgf2** (PMOS, g=d1, d=out, s=vdd) — primary active
     feedforward; shared with AFFC.
  2. **Compensation**: Cm1 (out↔d2) + Cm2 (d2↔d1) NMCF-style
     routing; Cf dropped (Mgf2 provides the LHP zero structurally).

R85 C cascode recipe applied
============================

  * **Mcas1 / Mcas2** PMOS cascode pair between input pair drains
    and mirror summing nodes (g=pcas, source at d1_diode_int /
    d1_int).
  * **Mcas_b** + **Iref_cas** — pcas rail bias (R83 D inherited).
  * **Mcas3_n / Mcas4_n** NMOS cascode pair on mirror legs (g=ncas,
    source at d1_diode_n_int / d1_n_int). M3/M4 gates closed at
    ``d1_diode`` (DFCFC1-style regulated cascode loop, R84 D
    inherited).
  * **Mcas_n_b1** + **Mcas_n_b2** stacked NMOS-diode pair for ncas
    rail; **Mbp_ncas_drv** PMOS mirror at pbias sources I_ref.

Honest scope
============

DFCFC1's own ``vcas`` rail + ``Mcas1_n_load`` / ``Mcas1_n_diode`` /
``Mcas_bias`` devices are **dropped** in this cascoded variant. The
R85 C cascoded mirror replaces them functionally. Plain DFCFC1
template is unchanged (byte-exact preserved).

All seven W's (LUT/heuristic) BYTE-EQUAL with R86 D v124 / R85 C v123.
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


def build_nested_miller_3stage_dfcfc1_cascoded_input_mirror(
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

    # Compensation caps (NMCF-style routing; Cf dropped)
    Cm1: float = 4.5e-12,
    Cm2: float = 0.5e-12,

    # DFCFC1-specific: active feedforward Mgf2
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

    # ─── R113 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # DFCFC1 cascoded-input-mirror variant:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset.
    #     Common-centroid required.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries 2× I_diff of bias current and sits adjacent to the
    #     NMOS mirror M3/M4 + NMOS cascode Mcas3_n/Mcas4_n; nwell
    #     guard ring (tied to vdd) prevents substrate coupling into
    #     the high-Z d1 node.
    #   * WELL_ISOLATED on the PMOS input cascode pair (Mcas1 / Mcas2).
    #     R83 D inherited common-gate PMOS cascode between input drains
    #     and the mirror summing nodes. Body-effect coupling through
    #     the shared nwell would degrade the cascode's r_o boost;
    #     dedicated nwell for the cascode pair (separate from Mtail's
    #     and the diff pair's) is required.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Leung DFCFC1 3-stage opamp with the R85 C
    cascode recipe applied (replacing DFCFC1's own internal cascoded
    mirror with the canonical R85 C structure).
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # DFCFC1: Mgf2 draws additional DC current.
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

    # Mgf2 sizing
    if L_gf2 is None:
        L_gf2 = L_cs3
    if gm_id_gf2 is not None:
        W_gf2 = closed_form_W("pmos", gm_id=gm_id_gf2,
                              I_d=I_gf2, L=L_gf2).W
    elif W_gf2 is None:
        W_gf2 = W_cs3 * Mgf2_scale

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

    # Cascode bias rail (pcas) - R83 D inherited
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

    # NMOS mirror-cascode bias rail (ncas) - R84 D inherited
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

    # DFCFC1 delta: Active feedforward Mgf2 (g=d1, d=out)
    devices.append(_mosfet("Mgf2", "pmos", d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # NMOS sink at out - widened to absorb both Mcs3 and Mgf2 DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # Compensation: Cm1 outer (out<->d2) + Cm2 inner (d2<->d1), NO Cf.
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R113 C: optional DRC annotations ─────────────────
    # Three annotations spanning three R111 C kinds. See the kwarg
    # docs block above for rationale.
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
                    "guard ring tied to vdd prevents substrate "
                    "coupling into the high-Z d1 node"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mcas1", "Mcas2"),
            extras={
                "rationale": (
                    "PMOS input cascode pair — body-effect coupling "
                    "through the shared nwell with the diff pair would "
                    "degrade the cascode r_o boost; isolate Mcas1/Mcas2 "
                    "in a dedicated nwell separate from M1/M2 and Mtail"
                ),
                "well_type": "nwell",
            },
        ))

    return cell
