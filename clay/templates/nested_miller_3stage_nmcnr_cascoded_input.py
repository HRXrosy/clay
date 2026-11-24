"""
Built-in template: nested_miller_3stage_nmcnr_cascoded_input.

NMCNR (Nested Miller Compensation with Nulling Resistor) opamp with a
**cascoded PMOS input pair**. Same compensation network as Leung's
plain NMCNR (Cm1 / Cm2 / Rz1 all sharing the rz_n internal node), but
the stage-1 differential pair is folded one rail deeper: each input
PMOS feeds an additional PMOS cascode device whose drain reaches the
NMOS mirror load. The cascode pair sits between the input pair drain
nodes (`d1_int`, `d1_diode_int`) and the mirror's traditional summing
nodes (`d1`, `d1_diode`).

Why this variant
================

Plain NMCNR's stage-1 gain is bounded by ``gm_pmos · ro_pmos`` of the
input pair alone — the NMOS mirror's output impedance ``ro_nmos``
appears in parallel with the input PMOS ``ro``, so the stage-1
small-signal gain is ``gm_pmos · (ro_pmos || ro_nmos)``. Adding a
common-gate PMOS cascode in series with each input device boosts the
stage-1 output impedance to ``ro_pmos_cas · gm_pmos_cas · ro_pmos``,
recovering a textbook cascode gain product. The downstream signal path
(NMOS mirror load, stage 2, stage 3, and the entire NMCNR Miller / Rz1
compensation network) is unchanged.

Cascoding the input pair also reduces the Miller-multiplication burden
on the gate-to-drain capacitance of the input PMOS — the cascode
common-gate device buffers the input pair's drain from the high-swing
d1 / d1_diode mirror node, so the parasitic ``Cgd`` of M1/M2 sees a
much smaller AC voltage swing at its drain side. In the AnalogGym
NMCNR-class targets this is the cheapest way to increase DC gain
without touching the Miller compensation budget.

Topology delta vs `nested_miller_3stage_nmcnr`
==============================================

Plain NMCNR stage 1 (5 devices):

    Mtail (PMOS, d=tail)             — tail current source
    M1    (PMOS, d=d1_diode, g=inp)  — input pair half A
    M2    (PMOS, d=d1,       g=inn)  — input pair half B
    M3    (NMOS, d=d1_diode, g=d1_diode, diode-mirror) — mirror load
    M4    (NMOS, d=d1,       g=d1_diode, mirror)        — mirror load

Cascoded-input NMCNR stage 1 (8 devices — adds 2 PMOS cascodes + 1
PMOS cascode bias diode):

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
On sky130 1v8 with |Vthp| ≈ 1.0 V and ``Vov`` ≈ 0.2 V the cascode pair
sits comfortably in saturation.

The mirror summing nodes ``d1`` and ``d1_diode`` keep their original
names so the rest of the cell (stage 2 gate ``g=d1``, replica bias
gate ``g=d1_diode``, inner Miller cap Cm2 endpoint ``n=d1``) is
unchanged byte-for-byte. The cascode pair introduces TWO new internal
nodes: ``d1_int`` and ``d1_diode_int`` (the input-pair drains pulled
out from under the cascode).

Stage 2, stage 3, and the NMCNR compensation block (Cm1 / Cm2 / Rz1
with shared node rz_n) are inherited verbatim from the plain NMCNR
template. The architectural signature distinguishing this variant
from plain NMCNR is therefore:

  - exactly TWO PMOS cascode devices named Mcas1 / Mcas2
  - a `pcas` bias rail driven by Mcas_b + Iref_cas
  - two new internal nodes d1_int / d1_diode_int between the input
    pair drains and the mirror summing nodes

The standalone `Rz1` resistor + the rz_n shared node remain — both
the input-cascode AND the nulling-R compensation signatures coexist.

Honest scope (what you should expect)
=====================================

Stage-1 gain rises by roughly ``20 log10(gm_pc · ro_pc)`` ≈ 15–25 dB
on sky130 tt at the default sizing (gm_pc ≈ 100 µS, ro_pc ≈ 1 MΩ).
The downstream stages and compensation network are unchanged, so:

  - gain_dc: NMCNR baseline + 15-25 dB delta (template default sizing
    targets a measurable ≥ 5 dB gain bump vs plain NMCNR at matched
    bias / Cm1 / Cm2 / Rz1).
  - gbw: roughly unchanged (set by gm_3 / Cm1, both unchanged).
  - pm:  roughly unchanged (compensation network identical).

The test `test_nmcnr_cascoded_input_higher_gain_than_baseline` in
`tests/test_nmcnr_cascoded_input_stdlib.py` checks the gain delta.

Knobs
-----

Cascoded-input NMCNR adds two new knobs on top of the plain NMCNR
baseline:

  W_cas: float = 20e-6
      Cascode PMOS device width. Default matches W_diff so the
      cascode and input-pair PMOS have similar Vov.
  L_cas: float = 0.5e-6
      Cascode PMOS channel length.

A small `Iref_cas` current source (default 1 µA) bleeds DC through
Mcas_b to set the pcas rail. The sink sits on vss so the cascode
gate rides one |Vgs_p| below vdd regardless of the input pair
operating point.

All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load,
W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3,
gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3, Cm1, Cm2, Rz1)
are inherited verbatim from plain NMCNR.
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


def build_nested_miller_3stage_nmcnr_cascoded_input(
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

    # ─── NMCNR compensation: TWO Miller caps + nulling R ────
    Cm1: float = 1.5e-12,
    Cm2: float = 0.5e-12,
    Rz1: float = 5e3,

    # ─── Cascode pair (new vs plain NMCNR) ──────────────────
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # ─── R120 C: optional qualitative DRC annotations ───────
    # When True, attach three layout-prompt annotations capturing the
    # NMCNR-cascoded-input variant's signal-path pair matches and the
    # NMCNR compensation cap-pair match constraint:
    #   * MATCHING (M1, M2) — PMOS diff pair Vth mismatch dominates
    #     input-referred offset; the PMOS input cascode lifts gain but
    #     does NOT cancel diff-pair Vth mismatch (cascode only buffers
    #     Vds, not Vgs).
    #   * MATCHING (Mcas1, Mcas2) — telescopic PMOS input cascode;
    #     Vds mismatch unbalances the cascode output impedance and
    #     reduces the cascode gain product the NMCNR variant relies
    #     on for the +15-25 dB stage-1 boost vs plain NMCNR.
    #   * PASSIVE_MATCHING (Cm1, Cm2) — NMCNR Miller cap pair; both
    #     share the rz_n nulling-R node (Cm1: rz_n↔d2 outer; Cm2:
    #     rz_n↔d1 inner). The Rz1 nulling-R's RHP-zero migration
    #     depends on the Cm1/Cm2 ratio being preserved over process
    #     gradients; fringing-cap unit arrays make the two caps
    #     track on the same gradient.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for an NMCNR 3-stage opamp with a cascoded
    PMOS input pair.

    Same 5-port surface as the plain NMCNR cell. The cascode pair
    Mcas1 / Mcas2 sits between the input pair drains (`d1_int`,
    `d1_diode_int`) and the NMOS mirror summing nodes (`d1`,
    `d1_diode`), with both cascode gates tied to a shared `pcas`
    rail driven by Mcas_b + Iref_cas.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec.
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (inherited from NMCNR).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the baseline signal path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four signal-path devices.
        Cm1, Cm2, Rz1: NMCNR compensation knobs (see plain NMCNR).
        W_cas, L_cas: cascode-pair PMOS sizing. Default matches the
            input-pair sizing so both halves see similar Vov.
        I_ref_cas: bleed current through the cascode bias diode
            Mcas_b. Sets pcas one |Vgs_p| below vdd.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents — same NMCNR bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

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

    # ─── Stage 2 replica-bias leg (unchanged from NMCNR) ─────
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
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                           W=W_bn, L=L_bn, M=K_cs3_load))

    # ─── Compensation: NMCNR (nulling-R + 2 Miller caps) ─────
    # Identical to plain NMCNR — Cm1/Cm2 share the rz_n node and
    # Rz1 connects rz_n to out.
    devices.append(_cap("Cm1", p="rz_n", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="rz_n", n="d1", value=Cm2))
    devices.append(_res("Rz1", p="rz_n", n="out", value=Rz1))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R120 C: optional DRC annotations ───────────────────
    # MATCHING (M1, M2) / MATCHING (Mcas1, Mcas2) /
    # PASSIVE_MATCHING (Cm1, Cm2). See the kwarg docs block above for
    # the per-annotation rationale.
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
                    "input-referred offset; the PMOS input cascode "
                    "(Mcas1/Mcas2) lifts stage-1 gain but does NOT "
                    "cancel diff-pair Vth mismatch (the cascode only "
                    "buffers the input pair's Vds, not Vgs). Common-"
                    "centroid layout of M1/M2 is therefore the "
                    "dominant offset control"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas1", "Mcas2"),
            extras={
                "rationale": (
                    "Telescopic PMOS input cascode pair — the "
                    "+15-25 dB stage-1 gain boost vs plain NMCNR "
                    "depends on Mcas1/Mcas2 sharing Vds bias. "
                    "Mismatched Mcas1/Mcas2 unbalances the cascode "
                    "output impedance and reduces the cascode gain "
                    "product, eroding the architectural payoff of "
                    "the cascoded-input variant"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cm1", "Cm2"),
            extras={
                "rationale": (
                    "NMCNR Miller cap pair — both share the rz_n "
                    "node (Cm1: rz_n↔d2 outer; Cm2: rz_n↔d1 inner). "
                    "The Rz1 nulling-R's RHP-zero migration depends "
                    "on the Cm1/Cm2 ratio being preserved over "
                    "process gradients; fringing-cap unit arrays "
                    "make the two caps track on the same gradient "
                    "and preserve the nulling-R zero placement"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))

    return cell
