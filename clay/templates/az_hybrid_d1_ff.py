"""
Built-in template: nested_miller_3stage_az_hybrid_d1_ff.

R89 B / BL-093 hybrid d1-gated FF retrofit batch — AZ variant.
Rolls out R88 C's NMCNR-validated retrofit pattern (cascoded mirror
+ d1-gated Mgf2-style FF → +30-35 dB gain lift) to Yan AZ.

Structural finding (R87 C / R88 C closure)
==========================================

R88 C pilot retrofit on NMCNR validated R87 C's structural finding:
the R85 C cascode recipe (telescopic PMOS input cascode + cascoded
NMOS mirror with DFCFC1-style diode loop at high-Z node ``d1``)
REQUIRES a d1-gated Mgf2-style PMOS feedforward (PMOS, g=d1, d=out,
s=vdd) to realise its full +30-35 dB gain product. NMCNR cascoded
baseline 83.29 dB → NMCNR + hybrid d1-FF 119.36 dB (+36.07 dB lift,
GBW landed at 1.230 MHz inside paper +/-20% window [0.96, 1.44] MHz).

R89 B extends the same retrofit pattern to three more FAIL variants
(AZ / AZC / DACFC). This file is the AZ retrofit.

AZ retrofit approach: dual-FF
==============================

AZ's R87 C cascoded baseline retains an active FF Mgmf2 (PMOS, g=d2,
d=out, s=vdd) — but Mgmf2 is gated by ``d2`` (stage-2 output),
NOT by ``d1`` (stage-1 output). This means Mgmf2 is part of the main
A1·A2·A3 signal path, not a direct d1→out bypass. AZ's R87 C measured
gain_dc = 84.90 dB confirms Mgmf2 alone doesn't lift gain past the
83-87 dB ceiling characteristic of FAIL variants.

R89 B AZ retrofit is **dual-FF**: keep AZ's existing Mgmf2 (g=d2)
verbatim, AND add a NEW Mgf2_hybrid (g=d1) directly grafted from the
R88 C NMCNR pattern. The two FFs operate on different stages:

  * Mgmf2  : g=d2, d=out — completes the existing AZ paper recipe;
             provides LHP zero via FF leg.
  * Mgf2_hybrid : g=d1, d=out — NEW, R88 C inherited; provides the
             direct d1→out gain path that unlocks the cascoded
             mirror's +30 dB gain product.

Mlcs3 NMOS output sink widened by (1 + Mgmf2_scale + Mgf2_scale)
to absorb both FF DC currents into the output node.

Architectural devices preserved verbatim
=========================================

AZ fingerprint that survives R89 B retrofit:

  1. **Mgmb1** (NMOS, d=naz_drv, g=naz_cap, s=vss) — auto-zero
     sensing device (unchanged).
  2. **Mgmb2** (NMOS, d=naz_pair, g=naz_drv, s=vss) — auto-zero
     mirror device (unchanged).
  3. **Mgmf2** (PMOS, g=d2, d=out, s=vdd) — original AZ active FF
     PRESERVED VERBATIM.
  4. **Raz** (resistor, naz_drv <-> naz_cap) — primary nulling R.
  5. **Caz** (cap, naz_cap <-> naz_pair) — auto-zero series cap.
  6. **Caz_outer** (cap, out <-> d2) — NMCF-style outer Miller.
  7. **DC bias paths** Mbn_naz_pair + Mbp_naz_cap (unchanged).

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

If gain_dc lands in [115, 120] dB AND GBW stays in [4.00, 6.00] MHz,
R88 C structural finding generalises across cross-FF-structure
variants. M-SIZING-PAPER-GRADE 9/17 → 10/17 (AZ counted as PASS).

Honest scope
============

AZ paper recipe has Mgmf2 (g=d2) as its only active FF. Adding
Mgf2_hybrid (g=d1) is an architectural cross-graft, not paper
replication. Both FFs running concurrently is a Clay engineering
choice — documented as additive R89 B delta. At default
Mgf2_scale = 0.10 + Mgmf2_scale = 0.10, total stage-3 NMOS sink
widens by (1 + 0.10 + 0.10) = 1.20× — modest area / power overhead.
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


def build_nested_miller_3stage_az_hybrid_d1_ff(
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

    # AZ-specific: auto-zero NMOS pair sizing
    W_gmb: float = 4e-6,
    L_gmb: float = 1e-6,

    # AZ-specific: original active FF Mgmf2 (g=d2, paper recipe)
    Mgmf2_scale: float = 0.1,
    W_gmf2: float | None = None,
    L_gmf2: float | None = None,
    gm_id_gmf2: float | None = None,

    # AZ-specific: compensation network
    Caz_outer: float = 1.0e-12,  # outer Miller (out<->d2)
    Caz:       float = 0.5e-12,  # auto-zero cap (naz_cap<->naz_pair)
    Raz:       float = 10e3,     # primary nulling-R (naz_drv<->naz_cap)

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
    # AZ + R89 B hybrid d1-FF retrofit:
    #   * MATCHING on the PMOS diff pair (M1, M2). Vth mismatch maps
    #     directly to input-referred offset; AZ's auto-zero loop
    #     cancels DC offset over the chopper cycle, but short-term
    #     mismatch still seeds the d1 settling error that the new
    #     d1-FF (Mgf2_hybrid) directly amplifies to out.
    #   * MATCHING on the AZ NMOS pair (Mgmb1, Mgmb2). Reciprocal
    #     gate/drain coupling (Mgmb1 g=naz_cap d=naz_drv; Mgmb2
    #     g=naz_drv d=naz_pair) makes the AZ feedback sensitive to
    #     Vth mismatch — a mismatched pair would skew the auto-zero
    #     reference and leak residual DC offset to out via Mgmf2's
    #     d2->out path.
    #   * GUARD_RING on Mtail (PMOS tail, M=N_tail). Carries the
    #     entire stage-1 bias current; nwell guard ring tied to vdd
    #     isolates the high-Z d1 node — now drives BOTH Mcs2 (main
    #     path) AND Mgf2_hybrid (NEW d1-FF, g=d1, d=out) — from
    #     supply-coupled substrate noise. With the R89 B d1-FF in
    #     place, d1 noise sees a +30 dB direct path to out, so this
    #     guard ring matters more than on the cascoded-only baseline.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for AZ 3-stage opamp with R85 C cascode recipe,
    original AZ Mgmf2 (g=d2) PRESERVED, AND a NEW Mgf2_hybrid (g=d1)
    grafted from R88 C NMCNR pattern (R89 B dual-FF retrofit).

    AZ signature (auto-zero NMOS pair + nulling R + active FF Mgmf2)
    preserved verbatim. NEW: Mgf2_hybrid (PMOS, g=d1, d=out) + Mlcs3
    widened by (1 + Mgmf2_scale + Mgf2_scale) to absorb both FF DC
    currents.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (NMCF baseline inherited).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the baseline signal path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four signal-path devices.
        W_gmb, L_gmb: AZ auto-zero NMOS pair sizing.
        Mgmf2_scale: original AZ Mgmf2 (g=d2) scale. Default 0.1
            (paper recipe).
        W_gmf2, L_gmf2, gm_id_gmf2: original Mgmf2 sizing overrides.
        Caz_outer: outer Miller cap (out ↔ d2). Default 1.0 pF.
        Caz: auto-zero series cap (naz_cap ↔ naz_pair). Default 0.5 pF.
        Raz: primary nulling-R (naz_drv ↔ naz_cap). Default 10 kOhm.
        Mgf2_scale: R89 B hybrid d1-FF scale relative to Mcs3 stage-3
            CS amp. Default 0.1 (R88 C tested-good value).
        W_gf2, L_gf2, gm_id_gf2: Mgf2_hybrid sizing overrides.
        W_cas, L_cas, I_ref_cas: PMOS input-cascode pair sizing
            (R83 D inherited).
        W_cas_n_mirror, L_cas_n_mirror: NMOS mirror-cascode pair
            sizing (R84 D inherited).
        W_cas_n_bias, L_cas_n_bias: stacked NMOS-diode pair sizing
            for the ncas bias rail.
        I_ref_cas_n: bookkeeping placeholder; actual ncas-rail
            current set by Mbp_ncas_drv at I_ref via pbias mirror.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # AZ original Mgmf2 draws DC current at out (preserved).
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

    # AZ delta: auto-zero NMOS pair (PRESERVED VERBATIM)
    devices.append(_mosfet("Mgmb1", "nmos",
                            d="naz_drv", g="naz_cap", s="vss", b="vss",
                            W=W_gmb, L=L_gmb))
    devices.append(_mosfet("Mgmb2", "nmos",
                            d="naz_pair", g="naz_drv", s="vss", b="vss",
                            W=W_gmb, L=L_gmb))

    # AZ original active FF Mgmf2 (PMOS, g=d2, d=out) — PRESERVED.
    devices.append(_mosfet("Mgmf2", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf2, L=L_gmf2))

    # R89 B delta: hybrid d1-gated Mgf2-style active feedforward
    # (PMOS, g=d1, d=out, s=vdd). R88 C NMCNR inherited pattern.
    devices.append(_mosfet("Mgf2_hybrid", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf2 + Mgf2_hybrid
    # DC currents (R89 B delta: total scale is now 1 + Mgmf2_scale +
    # Mgf2_scale).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # AZ compensation: Caz_outer (NMCF-style outer Miller) + Caz +
    # Raz (auto-zero series).
    devices.append(_cap("Caz_outer", p="out", n="d2", value=Caz_outer))
    devices.append(_cap("Caz", p="naz_cap", n="naz_pair", value=Caz))
    devices.append(_res("Raz", p="naz_drv", n="naz_cap", value=Raz))

    # DC bias paths for AZ internal nodes.
    devices.append(_mosfet("Mbn_naz_pair", "nmos",
                            d="naz_pair", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mbp_naz_cap", "pmos",
                            d="naz_cap", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # R118 C: optional DRC annotations
    # MATCHING (diff pair) / MATCHING (AZ NMOS pair) / GUARD_RING
    # (Mtail). See kwarg docs above for the per-annotation rationale.
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
                    "input-referred offset. AZ auto-zero cancels DC "
                    "offset over the chopper cycle but short-term "
                    "mismatch still seeds the d1 settling error that "
                    "the R89 B d1-FF (Mgf2_hybrid, g=d1, d=out) "
                    "directly amplifies to out"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mgmb1", "Mgmb2"),
            extras={
                "rationale": (
                    "AZ NMOS pair with reciprocal gate/drain coupling "
                    "(Mgmb1 g=naz_cap d=naz_drv; Mgmb2 g=naz_drv "
                    "d=naz_pair) — Vth mismatch skews the auto-zero "
                    "reference and leaks residual DC offset to out via "
                    "the Mgmf2 d2->out path that AZ retains"
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
                    "to vdd isolates the high-Z d1 node — now drives "
                    "BOTH Mcs2 (main path) AND Mgf2_hybrid (R89 B "
                    "d1-FF, g=d1, d=out) — from supply-coupled "
                    "substrate noise. With the d1-FF in place, d1 "
                    "noise sees a +30 dB direct path to out"
                ),
                "well_type": "nwell",
            },
        ))

    return cell
