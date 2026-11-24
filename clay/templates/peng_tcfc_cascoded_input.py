"""
Built-in template: nested_miller_3stage_tcfc_cascoded_input.

Peng TCFC 3-stage opamp with **cascoded PMOS input pair** added.
Preserves every TCFC architectural signature (active feedforward Mgmf,
cascoded NMOS sink Mcas_n with diode-biased Mlcs3 chain, internal node
nd3, no Cf) but folds the stage-1 differential pair one rail deeper:
each input PMOS feeds an additional PMOS cascode device whose drain
reaches the NMOS mirror load. The cascode pair sits between the input
pair drains (`d1_int`, `d1_diode_int`) and the mirror's traditional
summing nodes (`d1`, `d1_diode`).

Why this variant
================

R82 D landed a LUT-sized plain Peng TCFC at GBW = 0.973 MHz (paper
target 1.0 MHz, paper threshold +/-20% PASS) but gain_dc = 88.5 dB,
far short of the paper's 130 dB AnalogGym headline. The gap is not a
sizing problem — plain 3-stage NMC + active FF + cascoded sink is
structurally bounded by ``gm_pmos · (ro_pmos || ro_nmos)`` at stage-1,
~30-35 dB on sky130 1v8. R83 D adds a PMOS cascode pair in series
with each input device, recovering a textbook cascode gain product
``gm_pmos_cas · ro_pmos_cas · ro_pmos`` at stage-1 (~+15-25 dB on
sky130 tt). Same structural pattern used by
``nested_miller_3stage_nmcnr_cascoded_input`` (R68 Agent I).

Cascoding the input pair also reduces the Miller-multiplication burden
on the gate-to-drain capacitance of the input PMOS — the cascode
common-gate device buffers the input pair's drain from the high-swing
d1 / d1_diode mirror node, so the parasitic ``Cgd`` of M1/M2 sees a
much smaller AC voltage swing at its drain side.

Topology delta vs `nested_miller_3stage_tcfc`
==============================================

Plain TCFC stage 1 (5 devices):

    Mtail (PMOS, d=tail)             — tail current source
    M1    (PMOS, d=d1_diode, g=inp)  — input pair half A
    M2    (PMOS, d=d1,       g=inn)  — input pair half B
    M3    (NMOS, d=d1_diode, g=d1_diode, diode-mirror) — mirror load
    M4    (NMOS, d=d1,       g=d1_diode, mirror)        — mirror load

Cascoded-input TCFC stage 1 (8 devices — adds 2 PMOS cascodes + 1
PMOS cascode bias diode + 1 cascode bias current source):

    Mtail   (PMOS, d=tail)                            — tail current
    M1      (PMOS, d=d1_diode_int, g=inp, s=tail)     — input pair A
    M2      (PMOS, d=d1_int,       g=inn, s=tail)     — input pair B
    Mcas1   (PMOS, d=d1_diode, g=pcas, s=d1_diode_int) — cascode A
    Mcas2   (PMOS, d=d1,       g=pcas, s=d1_int)      — cascode B
    Mcas_b  (PMOS, d=pcas, g=pcas, s=vdd, diode)      — pcas bias diode
    Iref_cas (cur_src, p=pcas, n=vss)                 — pcas bias sink
    M3      (NMOS, d=d1_diode, g=d1_diode, diode)     — mirror load A
    M4      (NMOS, d=d1,       g=d1_diode, mirror)    — mirror load B

The cascode-gate rail ``pcas`` sits one |Vgs_p| below vdd (sized by
Mcas_b's diode-drop with Iref_cas pulling current to vss); this puts
the cascode device sources at ``pcas + |Vgs_p|`` ≈ vdd - 2|Vgs_p|,
which is one |Vgs_p| of headroom below the input-pair source node.
On sky130 1v8 with |Vthp| ≈ 1.0 V and ``Vov`` ≈ 0.13 V (gm/Id ≈ 15
moderate-inv) the cascode pair sits comfortably in saturation.

Every other TCFC delta vs plain NMCF is preserved verbatim:

  1. Active feedforward Mgmf (PMOS, g=d2, d=out, s=vdd) — drives the
     output from the stage-2 output node, bypassing only stage 3.
  2. Cascoded NMOS sink at stage-3 output: Mcas_n (NMOS, d=out, s=nd3,
     g=nbias) + Mlcs3 (NMOS, d=nd3, s=vss, g=nd3_diode) + Mlcs3_diode
     (NMOS-diode at nd3_diode) + Mlcs3_diode_drv (PMOS-mirror pushes
     I_ref into nd3_diode). The internal node ``nd3`` is the TCFC-
     distinct net of the sink chain.
  3. Compensation: TWO Miller caps only (Cm1 outer + Cm2 inner). No
     Cf — the active Mgmf + cascoded sink combine to provide the PM
     recovery that NMCF gets from Cf.

Combined architectural signature
================================

A cell built from this template has BOTH:

  - The TCFC architectural fingerprint: Mgmf (active FF gated by d2),
    Mcas_n (NMOS cascode at output), nd3 (internal node between Mcas_n
    and Mlcs3), Mlcs3_diode + Mlcs3_diode_drv (diode-bias chain), and
    NO Cf.
  - The input-cascode fingerprint: Mcas1 / Mcas2 PMOS cascode pair, a
    `pcas` bias rail driven by Mcas_b + Iref_cas, two new internal
    nodes d1_int / d1_diode_int.

This is the FIRST stdlib stage that combines TCFC's compensation
architecture with a cascoded input — meant to close the gain_dc gap
on AnalogGym Peng_TCFC_Pin_3 toward the 130 dB headline target while
keeping the GBW = 1.0 MHz LUT-sized result from R82 D inside the
paper threshold +/-20% window.

Honest scope (what you should expect)
=====================================

Stage-1 gain rises by roughly ``20 log10(gm_pc · ro_pc)`` ≈ 15-25 dB
on sky130 tt at the default sizing (gm_pc ≈ 100 µS, ro_pc ≈ 1 MΩ).
The downstream stages and compensation network are unchanged, so:

  - gain_dc: TCFC baseline (~88.5 dB at LUT-sized v120) + 15-25 dB
    delta from the cascode pair. Expected landing zone is roughly
    100-115 dB. The 130 dB paper headline still likely out of reach
    without further upgrades (e.g. cascoded mirror load too, or a
    4th-stage gain boost), but moving from 88.5 dB to ≥ 100 dB is
    a meaningful step toward paper-grade and satisfies the R83 D
    acceptance criterion.
  - gbw: roughly unchanged (set by gm_3 / Cm1 in the outer Miller
    loop, both unchanged).
  - pm:  roughly unchanged (compensation network — Cm1 + Cm2 +
    active Mgmf + cascoded NMOS sink — identical to plain TCFC).

Knobs
-----

Cascoded-input TCFC adds two new knobs on top of the plain TCFC
template:

  W_cas: float = 20e-6
      Cascode PMOS device width. Default matches W_diff so the
      cascode and input-pair PMOS have similar Vov at the same
      drain current.
  L_cas: float = 0.5e-6
      Cascode PMOS channel length.

A small `I_ref_cas` current source (default 1 µA) bleeds DC through
Mcas_b to set the pcas rail. The sink sits on vss so the cascode
gate rides one |Vgs_p| below vdd regardless of the input pair
operating point.

All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load,
W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3,
gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3, Cm1, Cm2,
Mgmf_scale, W_gmf, L_gmf, gm_id_gmf, W_cas_n, L_cas_n) are inherited
verbatim from plain TCFC.
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


def build_nested_miller_3stage_tcfc_cascoded_input(
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

    # ─── Compensation caps (Cm1, Cm2 only — no Cf in TCFC) ──
    Cm1: float = 1.5e-12,
    Cm2: float = 0.5e-12,

    # ─── TCFC-specific: active feedforward (Mgmf) ───────────
    Mgmf_scale: float = 1.0,
    W_gmf: float | None = None,
    L_gmf: float | None = None,
    gm_id_gmf: float | None = None,

    # ─── TCFC-specific: NMOS cascode at output sink (Mcas_n) ─
    W_cas_n: float = 4e-6,
    L_cas_n: float = 1e-6,

    # ─── NEW vs plain TCFC: PMOS cascode pair on the input ──
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # R122 C: qualitative DRC annotation surface. Additive metadata;
    # default False keeps every pre-R122 caller byte-for-byte identical.
    # When True, three qualitative hints attach to the cell:
    #   MATCHING on (M1, M2)              -- PMOS diff pair
    #   MATCHING on (Mcas1, Mcas2)        -- PMOS input cascode
    #   CURRENT_MIRROR_MATCHING on (M3, M4) -- NMOS mirror load (no NMOS cascode here)
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Peng TCFC 3-stage opamp with a cascoded
    PMOS input pair.

    Same 5-port surface as the plain TCFC cell. The cascode pair
    Mcas1 / Mcas2 sits between the input pair drains (`d1_int`,
    `d1_diode_int`) and the NMOS mirror summing nodes (`d1`,
    `d1_diode`), with both cascode gates tied to a shared `pcas`
    rail driven by Mcas_b + Iref_cas. Every other TCFC architectural
    device (Mgmf, Mcas_n, nd3, Mlcs3_diode, Mlcs3_diode_drv) is
    preserved verbatim from `build_nested_miller_3stage_tcfc`.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (inherited from plain TCFC).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the baseline signal path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four signal-path devices.
        Cm1, Cm2: nested Miller cap values (no Cf in TCFC).
        Mgmf_scale: ratio of Mgmf W to Mcs3 W (default 1.0).
        W_gmf, L_gmf: optional explicit overrides for Mgmf sizing.
        gm_id_gmf: gm/Id intent override for Mgmf.
        W_cas_n, L_cas_n: NMOS cascode device sizing at the output sink
            (default 4 µm / 1 µm, matches Mbn sizing).
        W_cas, L_cas: PMOS input-cascode pair sizing. Default matches
            the input-pair sizing so both halves see similar Vov.
        I_ref_cas: bleed current through the cascode bias diode
            Mcas_b. Sets pcas one |Vgs_p| below vdd.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as plain TCFC).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # TCFC: Mgmf draws additional DC current proportional to Mgmf_scale.
    I_gmf = Mgmf_scale * K_cs3_load * I_ref

    # The NMOS sink leg must absorb Mcs3 + Mgmf currents. Cascoded sink
    # mirror ratio is bumped internally to keep the output DC OP near
    # the same V(out) as plain TCFC.
    K_cs3_load_total = K_cs3_load * (1 + Mgmf_scale)

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

    # Mgmf sizing: gm/Id override beats W_gmf which beats Mgmf_scale.
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

    # ─── Cascode bias rail (pcas) ────────────────────────────
    # PMOS diode Mcas_b sits between vdd and pcas; Iref_cas pulls
    # current to vss. pcas settles at vdd - |Vgs_p|(Mcas_b @ I_ref_cas).
    # Sized with the same width as the cascode pair so |Vov| is
    # consistent at the bias-replication target current.
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

    # ─── Stage 1: cascoded PMOS diff pair with NMOS mirror load ─
    # Input pair drains route to the new internal nodes
    # `d1_diode_int` / `d1_int` (NOT directly to the mirror).
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode_int", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1_int",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    # Cascode pair: common-gate (g=pcas) PMOS between input drains
    # and the mirror summing nodes. Sources = input pair drains,
    # drains = original mirror summing nodes d1_diode / d1.
    devices.append(_mosfet("Mcas1", "pmos",
                            d="d1_diode", g="pcas", s="d1_diode_int", b="vdd",
                            W=W_cas, L=L_cas))
    devices.append(_mosfet("Mcas2", "pmos",
                            d="d1",       g="pcas", s="d1_int",       b="vdd",
                            W=W_cas, L=L_cas))

    # NMOS mirror load — unchanged (drains land at d1_diode / d1).
    devices.append(_mosfet("M3", "nmos", d="d1_diode", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos", d="d1",       g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    # ─── Stage 2: NMOS CS amplifier driven by d1 ─────────────
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # ─── Stage 2 replica-bias leg (unchanged from plain TCFC) ─
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

    # ─── TCFC delta 1: Active feedforward device Mgmf ───────
    # Same as plain TCFC: PMOS gated by d2 (stage-2 output), drain
    # to global output, source to vdd. Bypasses ONLY stage 3.
    devices.append(_mosfet("Mgmf", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf, L=L_gmf))

    # ─── TCFC delta 2: Cascoded NMOS sink at output ──────────
    # Identical to plain TCFC. Mcas_n between out and nd3, Mlcs3
    # sinks nd3 to vss, biased by Mlcs3_diode + Mlcs3_diode_drv.
    devices.append(_mosfet("Mcas_n", "nmos",
                            d="out", g="nbias", s="nd3", b="vss",
                            W=W_cas_n, L=L_cas_n,
                            M=K_cs3_load_total))
    devices.append(_mosfet("Mlcs3", "nmos",
                            d="nd3", g="nd3_diode", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))
    devices.append(_mosfet("Mlcs3_diode", "nmos",
                            d="nd3_diode", g="nd3_diode",
                            s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs3_diode_drv", "pmos",
                            d="nd3_diode", g="pbias",
                            s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # ─── Compensation: Cm1 + Cm2 (no Cf, same as plain TCFC) ─
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))

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
                    "PMOS diff pair under the R83 D input cascode "
                    "stack. TCFC's Mgmf active feedforward at out and "
                    "Mcas_n cascoded NMOS sink both depend on stage-1 "
                    "differential balance — Vth mismatch translates "
                    "to input-referred offset that the feedforward "
                    "amplifies into the output stage"
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
                    "Mismatch in cascode Vth changes the diff pair's "
                    "operating Vds and defeats the rout boost the "
                    "R83 D cascoded-input recipe contributes — the "
                    "stage-1 gain recovery toward the AnalogGym "
                    "Peng TCFC 130 dB headline collapses"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "NMOS stage-1 diode mirror (M3 diode, M4 mirror "
                    "leg). Unlike the cascoded-input-mirror sibling "
                    "this variant keeps the NMOS mirror non-cascoded, "
                    "so mirror ratio error injects a static "
                    "differential current into d1 that propagates "
                    "directly into the Mgmf feedforward path; "
                    "common-centroid unit-cell array required to hold "
                    "the 1:1 ratio across process gradients"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
