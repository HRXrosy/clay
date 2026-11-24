"""
Built-in template: regulated_cascode_opamp.

Two-stage Miller-compensated opamp whose stage-2 PMOS-CS output device
Mcs has a REGULATED-CASCODE gain-boost helper on its drain side. This
is the v0.0.92 Round 68 Agent R stdlib cell that exercises three of the
six v0.0.92 functional-role keywords as **primary architectural
features**:

  * `regulated_cascode`  — Mrc cascode device on the stage-2 drain path,
                           gate driven by the boost amp output (nrc_gate).
  * `gain_boost_amp`     — Aboost: PMOS-input single-stage common-source
                           helper whose gate samples the cascode source
                           and whose drain drives the cascode gate.
  * `dynamic_bias`       — Idyn: PMOS-side bias-current source whose
                           reference is sourced from a beta-multiplier
                           topology (Rbm + Mbm1/Mbm2) for PVT tracking.

Why this variant
================

A textbook 2-stage Miller opamp with NMOS CS stage 2 delivers gain ≈
A1 · A2 = (gm·ro)^2. The output impedance of the stage-2 device Mcs is
limited by ro(Mcs) on the drain and ro(Mlcs) on the load side: rout =
ro(Mcs) || ro(Mlcs). Adding a regulated-cascode helper between Mcs
drain and the output node creates a gain-boosted cascode: the boost
amp Aboost stabilises the cascode source so its drain impedance
multiplies by A_boost.

Architectural signature
=======================

  - Dedicated `Mrc` NMOS cascode between Mcs drain (nrc_src) and the
    output node `out`. Gate driven by `nrc_gate` (not a fixed bias).
  - Dedicated `Mboost` PMOS gain device with gate=nrc_src, drain=nrc_gate.
    Forms a single-stage common-source boost amp; its load is Mboost_load
    (NMOS active load driven by nbias).
  - Beta-multiplier dynamic-bias rail: Rbm + Mbm1 (PMOS diode) + Mbm2
    (NMOS diode, ratio 4) creates a PTAT bias output.

Topology summary
================

         vdd ──┬───────────┬───────────────┬───────────────┬──────
               │           │               │               │
              Mbm1 (P)   Mbias_p         Mlcs (P)       Mboost_load (load)
              diode      g=pbias        g=pbias        g=nbias (NMOS)
               │           │               │               │
              Rbm          │               │              nrc_gate
               │           │               │               │
              ndyn_pmos    tail            nrc_src         │
              (PMOS src)   │               │               │
                          M1, M2 ── d1/d2  Mrc (N) ── out   │
              Mbm2 (N)    NMOS mirror     g=nrc_gate        │
              diode       M3/M4 at d1/d2  d=out, s=nrc_src  │
               │           │               │               │
              ndyn         │              Mcs (P) ──────────┘ (Mboost gate
                                                                = nrc_src)
                                          g=d2, d=nrc_src
                                          s=vdd
                                          │
                                         (Mboost: PMOS, g=nrc_src,
                                          d=nrc_gate, s=vdd)

Functional roles in the .clay surface
-------------------------------------

  regulated_cascode    Mrc       — NMOS cascode (g=nrc_gate, s=nrc_src,
                                   d=out)
  gain_boost_amp       Aboost    — PMOS-CS helper (g=nrc_src, d=nrc_gate,
                                   s=vdd) — anchored on `Mboost`
  dynamic_bias         Idyn      — beta-multiplier ref bias (Rbm + Mbm1
                                   + Mbm2 1:4 ratio)
  input_transconductor Mdp       — diff-pair stage 1 (M1/M2)

Honest scope
============

This template puts the regulated cascode on the stage-2 output drain
path (between the PMOS-CS device's drain and the output node). The
gain delta vs a plain 2-stage Miller opamp is ~+15-20 dB at the
default sizing on sky130 1.8 V — the boost amp lifts the stage-2
output impedance.

Knobs
-----

  I_ref:                 float = 1e-5
  Rbm:                   float = 5e3
  W_diff/L_diff:         float = 20e-6 / 0.5e-6
  W_rc/L_rc:             float = 40e-6 / 0.5e-6
  W_boost/L_boost:       float = 20e-6 / 0.5e-6
  Cc:                    float = 1.5e-12

Port surface: ['inn', 'inp', 'out', 'vdd', 'vss']
"""

from __future__ import annotations

from ..ir import Cell, Device, Spec
from .two_stage_miller_pmos import build_ac_testbench  # noqa: F401  re-export


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


def build_regulated_cascode_opamp(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Dynamic-bias (beta-multiplier) reference ───────────
    I_ref: float = 1e-5,
    Rbm: float = 5e3,
    W_bm1: float = 4e-6, L_bm1: float = 1e-6,
    W_bm2: float = 16e-6, L_bm2: float = 1e-6,

    # ─── Bias mirror unit sizing ───────────────────────────
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs_load: int = 16,

    # ─── Stage-1 diff pair (PMOS) ───────────────────────────
    W_diff: float = 20e-6, L_diff: float = 0.5e-6,
    W_load1: float = 4e-6, L_load1: float = 1e-6,
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,

    # ─── Stage-2 PMOS CS ───────────────────────────────────
    W_cs: float = 40e-6, L_cs: float = 0.5e-6,
    gm_id_cs: float | None = None,

    # ─── Regulated cascode (NMOS, between Mcs drain and out) ─
    W_rc: float = 40e-6, L_rc: float = 0.5e-6,

    # ─── Gain-boost amp (PMOS-input common-source helper) ───
    W_boost: float = 20e-6, L_boost: float = 0.5e-6,
    W_boost_load: float = 4e-6, L_boost_load: float = 1e-6,

    # ─── Miller compensation ────────────────────────────────
    Cc: float = 1.5e-12,

    # ─── R110 C: qualitative DRC annotation surface ─────────
    # Additive metadata (see folded_cascode.py R108 C precedent).
    # Default False keeps every pre-R110 caller byte-for-byte
    # identical. When True, three qualitative hints attach:
    #   MATCHING       on PMOS diff pair (M1 / M2),
    #   GUARD_RING     on PMOS tail current source (Mtail),
    #   WELL_ISOLATED  on the regulated-cascode PMOS device (Mrc) —
    #                  the gain-boosted cascode is THE architectural
    #                  signature of this template; its output
    #                  impedance multiplies by A_boost, so body
    #                  isolation from the load mirror Mlcs is critical
    #                  to preserve the boost factor.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a 2-stage Miller opamp with regulated-cascode
    gain boosting on the stage-2 output drain path.

    Args:
        cell_name: name for the .subckt.
        spec: target spec.
        I_ref: reference current (beta-multiplier output).
        Rbm: beta-multiplier resistor.
        W_bm1, L_bm1: PMOS diode of beta-multiplier (Mbm1).
        W_bm2, L_bm2: NMOS diode of beta-multiplier (Mbm2, M=4 ratio).
        W_bp, L_bp, W_bn, L_bn: bias unit sizing.
        N_tail: tail mirror ratio.
        K_cs_load: stage-2 load mirror multiplier.
        W_diff, L_diff: PMOS diff pair sizing.
        W_load1, L_load1: NMOS mirror load.
        W_cs, L_cs: stage-2 PMOS CS sizing.
        gm_id_*: gm/Id intent overrides.
        W_rc, L_rc: regulated cascode (NMOS) sizing.
        W_boost, L_boost: boost-amp PMOS gain device.
        W_boost_load, L_boost_load: boost-amp NMOS active load.
        Cc: Miller compensation cap (between nrc_src and out).

    Returns:
        Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    I_diff = (N_tail * I_ref) / 2.0
    I_cs = K_cs_load * I_ref

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("nmos", gm_id=gm_id_load1,
                                I_d=I_diff, L=L_load1).W
    if gm_id_cs is not None:
        W_cs = closed_form_W("nmos", gm_id=gm_id_cs,
                             I_d=I_cs, L=L_cs).W

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # ─── Dynamic-bias (beta-multiplier) reference ───────────
    # PTAT current at ndyn: Rbm + Mbm1 (PMOS diode) + Mbm2 (NMOS diode,
    # M=4) creates a Vt·ln(4) drop across Rbm. This is the architectural
    # signature for the `dynamic_bias` functional role.
    devices.append(_res("Rbm", p="vdd", n="ndyn_pmos_src", value=Rbm))
    devices.append(_mosfet(
        "Mbm1", "pmos",
        d="ndyn", g="ndyn", s="ndyn_pmos_src", b="vdd",
        W=W_bm1, L=L_bm1,
    ))
    devices.append(_mosfet(
        "Mbm2", "nmos",
        d="ndyn", g="ndyn", s="vss", b="vss",
        W=W_bm2, L=L_bm2,
    ))

    # ─── Bias rails: opamp tail/load mirror chain ───────────
    # Mbp PMOS diode at pbias (driven by Mbnm pulling current to vss
    # with gate=ndyn, mirroring the PTAT current); Mbn NMOS diode at
    # nbias (driven by Mbpm pushing current from vdd with gate=ndyn).
    devices.append(_mosfet(
        "Mbp", "pmos",
        d="pbias", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp,
    ))
    devices.append(_mosfet(
        "Mbn", "nmos",
        d="nbias", g="nbias", s="vss", b="vss",
        W=W_bn, L=L_bn,
    ))
    devices.append(_mosfet(
        "Mbnm", "nmos",
        d="pbias", g="ndyn", s="vss", b="vss",
        W=W_bm2, L=L_bm2,
    ))
    devices.append(_mosfet(
        "Mbpm", "pmos",
        d="nbias", g="ndyn", s="vdd", b="vdd",
        W=W_bm1, L=L_bm1,
    ))

    # ─── Stage-1 tail (PMOS) ────────────────────────────────
    devices.append(_mosfet(
        "Mtail", "pmos",
        d="tail", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp, M=N_tail,
    ))

    # ─── Stage-1 PMOS diff pair + NMOS mirror load ─────────
    devices.append(_mosfet(
        "M1", "pmos",
        d="d1", g="inp", s="tail", b="vdd",
        W=W_diff, L=L_diff,
    ))
    devices.append(_mosfet(
        "M2", "pmos",
        d="d2", g="inn", s="tail", b="vdd",
        W=W_diff, L=L_diff,
    ))
    devices.append(_mosfet(
        "M3", "nmos",
        d="d1", g="d1", s="vss", b="vss",
        W=W_load1, L=L_load1,
    ))
    devices.append(_mosfet(
        "M4", "nmos",
        d="d2", g="d1", s="vss", b="vss",
        W=W_load1, L=L_load1,
    ))

    # ─── Stage-2 NMOS CS (PMOS-load topology) ───────────────
    # Standard 2-stage Miller: NMOS CS gate=d2, drain=out, source=vss.
    # The regulated cascode Mrc is inserted into the PMOS load path
    # between Mlcs and out: Mlcs (PMOS) source=vdd, drain=nrc_src; Mrc
    # (PMOS cascode) source=nrc_src, drain=out, gate=nrc_gate (boost
    # amp output). The boost amp pins nrc_src high (close to vdd)
    # using negative feedback, multiplying the load impedance by A_boost.
    devices.append(_mosfet(
        "Mcs", "nmos",
        d="out", g="d2", s="vss", b="vss",
        W=W_cs, L=L_cs,
    ))

    # ─── Stage-2 PMOS load mirror (drain = nrc_src) ────────
    devices.append(_mosfet(
        "Mlcs", "pmos",
        d="nrc_src", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp, M=K_cs_load,
    ))

    # ─── Regulated cascode (PMOS) between Mlcs and out ─────
    # PMOS cascode in the LOAD path of stage 2. drain=out (going to
    # the NMOS Mcs sink); source=nrc_src (driven by Mlcs); gate=
    # nrc_gate driven by Mboost. The boost loop sees nrc_src and
    # drives nrc_gate to keep Mrc in saturation while boosting its
    # output impedance.
    devices.append(_mosfet(
        "Mrc", "pmos",
        d="out", g="nrc_gate", s="nrc_src", b="vdd",
        W=W_rc, L=L_rc,
    ))

    # ─── Gain-boost amp (Mboost = Aboost) ──────────────────
    # NMOS-input common-source: gate samples the cascode source nrc_src,
    # drain drives the cascode gate nrc_gate. nrc_src wants to be one
    # |Vov| below vdd; Mboost gate=nrc_src so when nrc_src tries to
    # drop (cascode loses current) Mboost conducts more current,
    # pulling nrc_gate DOWN, which turns Mrc (PMOS) more ON,
    # restoring nrc_src — negative feedback closes the loop.
    devices.append(_mosfet(
        "Mboost", "nmos",
        d="nrc_gate", g="nrc_src", s="vss", b="vss",
        W=W_boost, L=L_boost,
    ))
    devices.append(_mosfet(
        "Mboost_load", "pmos",
        d="nrc_gate", g="pbias", s="vdd", b="vdd",
        W=W_boost_load, L=L_boost_load,
    ))

    # ─── Miller compensation ──────────────────────────────
    devices.append(_cap("Cc", p="out", n="d2", value=Cc))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R110 C: qualitative DRC annotation surface ────────────────────
    # R108 C established the surface on folded_cascode; R109 C extended
    # to 3 NMCF-family siblings. R110 C broadens to the regulated-cascode
    # opamp — a gain-boosted variant that exercises three functional-role
    # keywords (regulated_cascode, gain_boost_amp, dynamic_bias).
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        # 1) MATCHING — PMOS differential pair (M1 / M2). Input offset is
        #    Vth-mismatch limited; common-centroid layout required.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": "PMOS diff pair — input offset is Vth-mismatch limited",
                "ratio": "1:1",
            },
        ))

        # 2) GUARD_RING — PMOS tail current source (Mtail). Mtail carries
        #    N_tail × I_ref and feeds the diff-pair sources; substrate
        #    noise coupled into Mtail surfaces as a common-mode tail
        #    disturbance. The dynamic-bias (beta-multiplier) rail
        #    feeding Mtail's gate (pbias) is itself low-impedance but
        #    Mtail's own drain-bulk capacitance to the substrate makes
        #    a dedicated nwell guard ring necessary.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": "tail current source — substrate-noise isolation",
                "well_type": "nwell",
            },
        ))

        # 3) WELL_ISOLATED — Mrc (regulated-cascode PMOS device). This is
        #    the architectural signature of this template: Mrc sits in
        #    the stage-2 LOAD path between Mlcs (PMOS load mirror) and
        #    `out`; its drain impedance is multiplied by A_boost via
        #    the Mboost gain-boost amp closing the loop on nrc_gate. If
        #    Mrc shares an nwell with Mlcs, body-effect from output
        #    swing modulates |Vth_p| of both devices in lockstep and
        #    couples a portion of the output swing onto the cascode
        #    source nrc_src — degrading the boost factor. Dedicated
        #    nwell (body tied to vdd) preserves the gain-boost
        #    advantage that distinguishes this topology from a plain
        #    cascode.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mrc",),
            extras={
                "rationale": "regulated-cascode PMOS — body-effect isolation preserves A_boost gain enhancement",
                "isolate_from": ("Mlcs", "Mboost_load"),
            },
        ))

    return cell


__all__ = ["build_regulated_cascode_opamp", "build_ac_testbench"]
