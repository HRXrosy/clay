"""
Built-in template: nested_miller_3stage_cfcc.

Three-stage opamp with **Cascoded Feedforward Compensation** (CFCC) —
Sau et al's cascode-indirect Miller compensation. A single Miller cap
terminates at a cascode source node rather than directly at the stage-2
output, decoupling the high-impedance node from Miller-cap parasitics.

Reference
=========

The CFCC topology is the cascode-indirect Miller scheme analysed in
Sau's NMCF-family paper (one of the three nested-3-stage variants
benchmarked in AnalogGym alongside Fan 2005 SMC and Hoi Lee 2003
AFFC). The headline insight: insert a cascode device between the
stage-3 input transistor's drain and the output node, then terminate
the Miller cap at the cascode source (not at the output). The cascode
buffer current-shunts the Miller cap so the high-Z output node sees
only a small reflected impedance, recovering pole-splitting without
loading the output with the full Miller cap.

Topology delta vs `nested_miller_3stage`
========================================

The signal path through stage 1 and stage 2 is identical to the NMCF
baseline (PMOS diff pair → NMOS-mirror load → NMOS CS stage 2 with
PMOS replica-bias load → output stage). Only stage 3 + compensation
network differ.

Stage 3 deltas
--------------

NMCF baseline stage 3:
  Mcs3  : PMOS, d=out,   g=d2, s=vdd, b=vdd     (CS input device)
  Mlcs3 : NMOS, d=out,   g=nbias, s=vss, b=vss  (current sink)

CFCC stage 3 (cascode added):
  Mcs3  : PMOS, d=cas_p, g=d2,        s=vdd,    b=vdd
  Mcas  : PMOS, d=out,   g=pbias_cas, s=cas_p,  b=vdd     (NEW cascode device)
  Mlcs3 : NMOS, d=out,   g=nbias,     s=vss,    b=vss

`cas_p` is a new internal net (the cascode source). The cascode gate
is driven by `pbias_cas`, a **level-shifted** PMOS bias one |Vgs_p|
below the existing `pbias` reference (v0.0.53). The bias chain adds
two new devices:

  Mbp_cas : PMOS diode, d/g=pbias_cas, s=pbias, b=pbias
            (body tied to source — separate n-well — so the |Vgs_p|
            drop does NOT include body-effect Vth shift)
  Mbnm_cas: NMOS mirror, d=pbias_cas, g=nbias, s=vss, b=vss
            (sinks I_ref out of pbias_cas → defines the |Vgs_p|
            drop across Mbp_cas)

This puts pbias_cas ≈ vdd − 2|Vgs_p| ≈ 1.3 V on gf180mcu 3.3 V
(vs ≈ 2.3 V with the v0.0.52 default cascode_gate='pbias'). Mcas's
gate-source bias |Vgs_cas| = cas_p − pbias_cas ≈ 1 V, deep into
strong inversion, so cas_p settles at ≈ pbias_cas + |Vgs_cas| ≈ 2.15 V.
This gives Mcs3 (drain=cas_p, source=vdd) a Vds budget of
vdd − cas_p ≈ 1.15 V — comfortable saturation — while Mcas's own
Vds = cas_p − out ≈ 0.5 V is at the cascode Vds_sat knee but
positive (PMOS saturated).

v0.0.52's `cascode_gate = 'pbias'` default put Mcas's gate at
~2.3 V and forced cas_p into the deep-triode regime (cas_p pinned
near vdd, Mcs3 squeezed). v0.0.53 swaps the default to the
level-shifted `pbias_cas` net. Callers wanting the old behavior
can pass `cascode_gate='pbias'` explicitly; callers wanting a
fully-tuned third-party bias net can pass any other string. The
sizing knobs `W_bp_cas` / `L_bp_cas` set the level-shift diode's
geometry.

Compensation network deltas
---------------------------

NMCF baseline has THREE compensation devices:

  Cm1 : cap, p=out, n=d2     (outer Miller, wraps stage 3 directly)
  Cm2 : cap, p=d2,  n=d1     (inner Miller, wraps stage 2)
  Cf  : cap, p=out, n=d1     (passive feedforward, LHP zero)

CFCC keeps ONE Miller cap, re-routed through the cascode source:

  Cm1 : cap, p=cas_p, n=d2   (Miller, but at cascode source node)

Cm2 and Cf are removed — the cascode-source termination of Cm1
provides the dominant feedback path Sau analyses, and the textbook
Cf was specifically there to cancel the RHP zero introduced by Cm2
in the NMCF scheme. With Cm2 gone, the RHP zero never forms.

Why cascoded-Miller helps (qualitative)
---------------------------------------

In a textbook Miller-compensated 3-stage amp, the Miller cap sees
the full output-stage gain stage_3_gain, multiplying the effective
cap by that gain. This is great for pole-splitting (the dominant
pole moves to low frequency) but bad for the high-frequency response:
the Miller cap appears at the output as a large effective load,
reducing GBW and adding a RHP zero from the direct feedforward path
through the cap.

The cascoded-Miller trick: insert a current buffer (cascode device)
between Mcs3's drain and the output node. The Miller cap now sits
at the cascode source — a low-impedance node from the output side.
Current flowing into Cm1 is shunted directly to vdd through Mcas
without dragging the high-Z output. The RHP zero from the
direct feedforward path through Cm1 is suppressed because Mcas
isolates Cm1 from the output node capacitively.

Net effects (relative to plain NMCF):
  - Dominant pole moves to similar frequency (still Miller-multiplied)
  - RHP zero suppressed (cascode current-buffers the feedforward)
  - Output stage drives a lower effective Miller load → potentially
    higher GBW, depending on Mcas Vds budget

Schematic (compensation + stage 3 delta only)
---------------------------------------------

  NMCF:                              CFCC (Sau):
    Mcs3 drain → out                   Mcs3 drain → cas_p
    Cm1: out ↔ d2                      Mcas: cas_p → out (PMOS cascode)
    Cm2: d2 ↔ d1                       Cm1: cas_p ↔ d2 (Miller at cas src)
    Cf:  out ↔ d1                      (no Cm2, no Cf)

Honest scope (what you should NOT expect)
-----------------------------------------

CFCC is a **cascode-augmented** compensation scheme. The cascode
adds a degree of freedom but also a constraint: the cascode device
needs Vds_sat headroom. v0.0.52 shipped with `cascode_gate='pbias'`
(reusing the existing PMOS-diode reference); on gf180mcu 3.3 V this
forced Mcas into deep triode (cas_p pinned near vdd, leaving Mcs3
no Vds budget) and the PVT result was GBW 48-84 MHz but PM tight
(~20° = ~0.36 rad at GBW vs NMCF's ~60°). v0.0.53 replaces that
default with a **level-shifted `pbias_cas` net** generated by an
extra PMOS-diode/NMOS-mirror pair: pbias_cas ≈ vdd − 2|Vgs_p|. With
that bias point Mcas operates in saturation, the cascode-buffer
action engages, the high-frequency pole moves further out, and the
PM recovers — while GBW degrades modestly. The trade is documented
in CHANGELOG v0.0.53. On low-supply PDKs (sky130 at 1.8 V) the
2-diode stack may not fit — callers can pass `cascode_gate='pbias'`
to recover the v0.0.52 sky130-friendly behavior, or wire to any
external bias net.

The distinguishing test is the materialised SPICE netlist: CFCC
contains exactly ONE Miller cap (Cm1, p=cas_p, n=d2) AND a cascode
device Mcas (PMOS, d=out, g=pbias_cas, s=cas_p). NMCF has no Mcas
and three compensation caps; SMC has Cm1 on a different routing
(d1↔out) and no Mcas; AFFC has the Mgf2 active feedforward and
all three NMCF caps. Tests assert these invariants.

Knobs
-----

CFCC exposes the cascode device + bias-shifter sizing on top of the
NMCF baseline (v0.0.53):

  W_cas: float = 30e-6             — cascode PMOS width
  L_cas: float = 0.5e-6            — cascode PMOS length
  W_bp_cas: float = 4e-6           — level-shift PMOS-diode width
  L_bp_cas: float = 1e-6           — level-shift PMOS-diode length
  cascode_gate: str = 'pbias_cas'  — net name to use as the cascode gate
                                       (default routes to the new
                                       level-shifted bias; pass 'pbias'
                                       to recover the v0.0.52 default,
                                       or any other net for a hand-wired
                                       bias)

All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load,
W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3,
gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3) are inherited
verbatim from the NMCF baseline. The kwargs Cm2 and Cf from the
NMCF baseline are **not exposed** because the CFCC topology does
not contain those devices — the only Miller cap value is `Cm1`.
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


def build_nested_miller_3stage_cfcc(
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

    # ─── CFCC compensation: ONE Miller cap at cascode source ─
    Cm1: float = 1.15e-12,  # single cascode-indirect Miller (cas_p ↔ d2)

    # ─── CFCC-specific: cascode device ──────────────────────
    W_cas: float = 30e-6,
    L_cas: float = 0.5e-6,
    # ─── CFCC-specific: cascode bias level-shifter (v0.0.53) ─
    # Extra PMOS-diode device that drops one |Vgs_p| from `pbias` to
    # produce `pbias_cas`. Sized at the same W/L as the existing Mbp
    # diode by default so both diodes pass I_ref at the same Vov.
    # Smaller W → larger |Vgs_p| drop → pbias_cas pulled further from
    # vdd → Mcas has more Vds budget (but Mcs3's Vds budget shrinks
    # proportionally — narrow the diff between defaults if PM doesn't
    # recover enough).
    W_bp_cas: float = 4e-6,
    L_bp_cas: float = 1e-6,
    cascode_gate: str = "pbias_cas",
    # R109 C: qualitative DRC annotation surface (additive metadata,
    # see folded_cascode.py R108 C precedent). Default False keeps
    # every pre-R109 caller byte-for-byte identical (no `_drc_annotations`
    # key in __dict__, no behavior change in SPICE emit / sizing /
    # verifier). When True, three qualitative hints attach:
    #   MATCHING       on PMOS diff pair (M1 / M2),
    #   GUARD_RING     on PMOS tail current source (Mtail),
    #   WELL_ISOLATED  on PMOS cascode device (Mcas) — the CFCC-defining
    #                  cascode whose body-effect coupling to the load
    #                  mirror would degrade output-stage ro.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Sau CFCC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC variants. CFCC differs from NMCF in two places:
    a cascode device Mcas inserted between Mcs3 drain and the output
    node, and the Miller cap terminating at the cascode source.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see NMCF baseline docstring).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four NMCF baseline devices.
        Cm1: cascode-indirect Miller cap value. Default 1.15 pF matches
            the AnalogGym Sau_CFCC reference netlist.
        W_cas, L_cas: cascode PMOS sizing. Default targets gf180mcu
            (3.3 V) operating point; tighten for sky130 (1.8 V).
        W_bp_cas, L_bp_cas: level-shift PMOS-diode sizing (v0.0.53).
            Stacks below Mbp to generate `pbias_cas` ≈ vdd − 2|Vgs_p|.
        cascode_gate: net name to use as cascode gate. Default
            'pbias_cas' routes through the v0.0.53 level-shifter.
            Pass 'pbias' for the v0.0.52 default (cascode in deep
            triode but works on sky130 1.8 V), or any external bias
            net name.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0     # each PMOS diff-pair leg
    I_load1 = I_diff                    # each NMOS mirror leg
    I_cs2 = I_diff                      # ≈ I_M3 stage-2 NMOS CS current
    I_cs3 = K_cs3_load * I_ref          # stage-3 PMOS CS current

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

    # ─── Cascode bias level-shifter (v0.0.53) ─────────────────
    # The v0.0.52 default `cascode_gate='pbias'` put Mcas's gate
    # at pbias ≈ vdd − |Vgs_p| (≈ 2.3 V on gf180mcu 3.3 V), which
    # forced cas_p to pin near vdd and Mcas into deep triode —
    # cascode current-buffer action was suppressed and PM was tight
    # (~20° at GBW). v0.0.53 adds an extra PMOS-diode stacked below
    # Mbp to produce `pbias_cas` ≈ vdd − 2|Vgs_p| (≈ 1.3 V on
    # gf180mcu 3.3 V). Mbnm_cas mirrors Mbn at the cascode-bias
    # node so the level-shift diode passes the same I_ref as the
    # original bias diode → both Vgs_p drops are well-defined.
    #
    # Body of Mbp_cas tied to its own source (separate n-well) so
    # the |Vgs_p| drop does NOT include body-effect Vth shift.
    # If body=vdd, Vsb_Mbp_cas = vdd − pbias > 0 would raise |Vth_p|
    # by ~γ·sqrt(Vsb), pulling pbias_cas ~0.3 V lower than intended
    # and shoving Mcas back into deep triode from the other side.
    # The body=source trick mirrors the working pattern from
    # `folded_cascode.py` (MbpC, lines 287-290) and
    # `telescopic_cascode.py` (Mpc_diag, lines 263-267).
    devices.append(_mosfet("Mbp_cas", "pmos",
                            d="pbias_cas", g="pbias_cas",
                            s="pbias", b="pbias",
                            W=W_bp_cas, L=L_bp_cas))
    devices.append(_mosfet("Mbnm_cas", "nmos",
                            d="pbias_cas", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # ─── Stage 1: PMOS diff pair with NMOS mirror load ───────
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M3", "nmos", d="d1_diode", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos", d="d1",       g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

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

    # ─── Stage 3 with PMOS cascode (CFCC architectural delta) ─
    # Mcs3 drains to cas_p (the cascode source), NOT directly to out.
    devices.append(_mosfet("Mcs3", "pmos", d="cas_p", g="d2", s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))

    # Mcas: PMOS cascode device, current-buffers between cas_p and out.
    # Gate biased at `cascode_gate` (default 'pbias' — uses the existing
    # PMOS-diode reference; override for a level-shifted cascode bias).
    devices.append(_mosfet("Mcas", "pmos", d="out", g=cascode_gate,
                            s="cas_p", b="vdd",
                            W=W_cas, L=L_cas))

    # NMOS sink at the output (unchanged from NMCF baseline).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load))

    # ─── Compensation: ONE Miller cap at cascode source ──────
    # Cm1 spans cas_p (cascode source, low-Z from output side) to d2
    # (stage-2 output). The cascode current-buffers the Miller cap from
    # the high-Z output node, suppressing the textbook RHP feedforward
    # zero while still providing pole-splitting via Miller multiplication.
    devices.append(_cap("Cm1", p="cas_p", n="d2", value=Cm1))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R109 C: qualitative DRC annotation surface ────────────────────
    # R108 C established the surface on folded_cascode; R109 C extends the
    # template inventory consumed by the layout prompt's round-trip path.
    # Annotations live in cell.__dict__["_drc_annotations"] only — SPICE
    # emit (which iterates cell.devices) is unaffected.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        # 1) MATCHING — PMOS differential pair (M1 / M2). The Sau CFCC
        #    inherits its stage-1 diff pair from the NMCF baseline; the
        #    input offset is set by M1/M2 Vth-mismatch (Hastings Ch. 8
        #    common-centroid prescription).
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": "PMOS diff pair — input offset is Vth-mismatch limited",
                "ratio": "1:1",
            },
        ))

        # 2) GUARD_RING — PMOS tail current source (Mtail). Carries
        #    N_tail × I_ref into the diff-pair sources; substrate-noise
        #    coupling on the tail node converts to common-mode disturbance
        #    that the cascoded Miller does not reject (CFCC's cascode
        #    helps PM, not PSRR). A dedicated nwell guard ring stops
        #    neighboring switching activity from modulating I_tail.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": "tail current source — substrate-noise isolation",
                "well_type": "nwell",
            },
        ))

        # 3) WELL_ISOLATED — CFCC cascode device Mcas. This is the
        #    architectural marker of the Sau CFCC topology (cascode-
        #    indirect Miller). Mcas sits between Mcs3's drain (cas_p
        #    node) and the output, current-buffering the Miller cap.
        #    If Mcas shares an nwell with the Stage-3 input device
        #    Mcs3, body-effect coupling from out swing modulates |Vth|
        #    and degrades the cascode buffer's output impedance —
        #    nullifying the cascode advantage that defines CFCC. A
        #    dedicated nwell tie (body=vdd, isolated from Mcs3 well)
        #    preserves the cascode action.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mcas",),
            extras={
                "rationale": "CFCC cascode — body-effect isolation from stage-3 PMOS Mcs3",
                "isolate_from": ("Mcs3",),
            },
        ))

    return cell
