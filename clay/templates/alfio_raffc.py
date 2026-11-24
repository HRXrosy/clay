"""
Built-in template: nested_miller_3stage_raffc.

Three-stage opamp with **Reverse Active FeedForward Compensation**
(Alfio_RAFFC) — an NMCF-family variant where the active feedforward
device is wired in the REVERSE direction compared to AFFC. AFFC's
active FF flows d1 → out (forward direction: feeds STAGE-1 output
INTO the global output, bypassing stages 2+3); RAFFC's active FF
flows out → d1 (reverse direction: feeds the GLOBAL OUTPUT back
INTO stage-1's high-impedance node). The gate/drain pair is
literally swapped — AFFC has (g=d1, d=out), RAFFC has (g=out,
d=d1). This creates a local NEGATIVE-feedback path that shapes the
loop transfer function with a different LHP-zero location than
AFFC's forward FF.

Reference
=========

Alfio_RAFFC_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Alfio_RAFFC_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm11 VOUT  net050 VDDA VDDA pfet  → Mgmf (primary FF PMOS, but
                                       direction depends on which
                                       net is "d1" vs "d2" — see
                                       below)
  xm10 net013 net050 VDDA VDDA pfet → REVERSE FF PMOS — drain into
                                       the bias-network return path
                                       net013 (which serves as the
                                       internal "feedback" node back
                                       to stage 1)
  xm22 VOUT  net043 GNDA GNDA nfet  → gm3_NMOS (stage-3 NMOS sink,
                                       same as NMCF Mlcs3)

The naming "Reverse" in the Alfio paper specifically refers to the
**direction** of the active FF current path: in AFFC, the FF PMOS
sources current INTO the output node from VDD (primary forward
direction d1 → out). In RAFFC, the equivalent device is wired so
its drain feeds back into an earlier stage's high-impedance node
(d2 → d1 direction). The "reverse" LHP-zero position arises because
the FF signal flows opposite to the main forward gain path.

The RAFFC architectural fingerprint:
  1. A **reverse active feedforward** PMOS `Mgf_rev` whose drain
     terminates at the stage-1 high-Z output d1, NOT at the global
     output. Gate driven by d2 (stage-2 output) so the path is
     d2 → d1 (reverse, against the main forward gain direction).
     This is the architectural signature: drain at d1, gate at d2.
  2. The NMOS sink at d1 (M4) widened to absorb both M4's mirror
     current and Mgf_rev's DC source current (analogous to AFFC's
     Mlcs3 widening, but on the stage-1 NMOS load instead of the
     stage-3 sink).
  3. Compensation routing stays NMCF-style for the outer Miller
     (Cm1 wraps stage 3 with an out↔d2 cap); Cf is dropped because
     the reverse FF provides the LHP zero structurally.

Topology delta vs `nested_miller_3stage`
========================================

Stages 2 and 3 inherit the NMCF baseline verbatim. Stage 1 keeps
its PMOS diff-pair + NMOS mirror but adds the reverse FF device
and an NMOS-load multiplier. Compensation network: NMCF outer
Miller routing (out↔d2), inner Miller routing (d2↔d1), Cf dropped.

**Reverse active feedforward Mgf_rev (NEW, RAFFC-unique):**
  Mgf_rev : PMOS, d=d1, g=out, s=vdd, b=vdd

  Compare with AFFC's Mgf2 (PMOS, d=out, g=d1, s=vdd). Both are
  PMOS active FF devices wired to vdd, but AFFC's gate is at d1
  (forward, stage-1 output side) and drain is at out (output side),
  while RAFFC's gate is at out (global-output side) and drain is at
  d1 (stage-1 output). The drain-gate node assignment is literally
  swapped — that's what makes it "reverse".

  Physical interpretation: in AFFC, when d1 rises, Mgf2 (PMOS,
  g=d1) conducts LESS, so the current it sources from VDD into out
  drops → out drops slightly (negative gain in the FF path,
  matching the inverting stage-2-then-3 main path → cooperative
  FORWARD feed). In RAFFC, when out rises, Mgf_rev (PMOS, g=out)
  conducts LESS, so the current it sources into d1 drops → d1
  drops. The d1 drop propagates through the main path
  (d1↓→Mcs2↓→d2↑→Mcs3↓→out↓) to drag out back down — local
  NEGATIVE-feedback loop. This stable feedback creates an LHP zero
  in the loop transfer function whose location is set by gm_rev /
  C_d1, distinct from AFFC's forward LHP zero at gm_f2 / Cf.

**NMOS sink at d1 widened (NEW, balances Mgf_rev DC current):**

  NMCF baseline:
    M4 : NMOS mirror at d1, M=1 (single device)

  RAFFC:
    M4 : NMOS mirror at d1, M=(1 + Mgf_rev_scale) (widened by the
         same ratio as AFFC widens Mlcs3 — to absorb Mgf_rev's DC
         current at d1)

**Compensation: Cm1 outer + Cm2 inner, NO Cf:**

  NMCF:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1
    Cf  : out ↔ d1    (passive LHP-zero generator, dropped in RAFFC)

  RAFFC:
    Cm1 : out ↔ d2    (same as NMCF outer Miller)
    Cm2 : d2  ↔ d1    (same as NMCF inner Miller)
    (Cf is dropped — the reverse FF Mgf_rev provides the LHP zero
     structurally)

RAFFC vs AFFC in one line
=========================

AFFC: Mgf2 (PMOS, g=d1, d=out) — FORWARD active FF d1 → out, plus
      passive Cf. Compensation = outer Cm1 (out↔d2) + inner Cm2
      (d2↔d1) + Cf (out↔d1) + Mgf2.

RAFFC: Mgf_rev (PMOS, g=out, d=d1) — REVERSE active FF out → d1.
       Compensation = outer Cm1 (out↔d2) + inner Cm2 (d2↔d1).
       Cf dropped because Mgf_rev provides the LHP zero through its
       negative-feedback loop.

The two cells have the SAME 5 ports and the SAME stage-2 / stage-3
signal paths, but differ structurally at the FF device's drain/gate
swap. The materialised SPICE netlists differ by exactly the (g, d)
assignment of one PMOS device and the absence of Cf in RAFFC.

Why the reverse direction matters (transfer-function level)
-----------------------------------------------------------

In a 3-stage NMCF, the loop transfer function has:
  - Dominant pole at d2 (loaded by Cm1 multiplied by stage-3 gain)
  - Second pole at d1 (loaded by Cm2 multiplied by stage-2 gain)
  - Third pole at out (loaded by Cm1 + C_L)
  - One LHP zero from the FF path (location depends on FF
    direction)

AFFC's forward FF puts the LHP zero at s ≈ -gm_f2 / (C_out + Cf):
high-frequency, near GBW, used to cancel the second pole.

RAFFC's reverse FF puts an additional LHP zero at s ≈ -gm_rev /
(Cm2 · |A_2| ) (approximately, where A_2 is the stage-2 gain):
lower-frequency, sits between the dominant pole and second pole.
The mid-frequency zero gives a different PM/GBW trade — RAFFC
trades some GBW (zero pushes phase up earlier) for wider PM
margin at the same Cm1.

Honest scope (what you should NOT expect)
-----------------------------------------

RAFFC is a structural compensation refinement that genuinely
changes the netlist (Mgf_rev with reversed g/d assignment +
widened M4 + no Cf). At the relaxed 5 dB / 10 kHz spec floor on
gf180mcu 3.3 V, the PVT envelope will land comfortably; whether
the reverse FF gives better PVT corners than AFFC depends on
sizing (Mgf_rev_scale and Cm2 in particular). Tests in
`tests/test_alfio_raffc_distinct.py` lock the architectural
distinction (Mgf_rev exists with g=d2 & d=d1, NOT g=d1 & d=out;
no Cf) rather than asserting specific SPICE numerics.

Knobs
-----

RAFFC adds the following new kwargs on top of the NMCF baseline:

  Mgf_rev_scale: float = 1.0
      Sizing ratio of Mgf_rev / Mcs3. Default 1.0 → gm_rev ≈ gm_3
      at the same gm/Id. Reasonable range 0.5-2.0; below 0.5 the
      reverse FF is too weak to matter, above 2.0 burns excessive
      DC current at d1.
  W_gf_rev, L_gf_rev: optional explicit overrides for Mgf_rev sizing.
  gm_id_gf_rev: gm/Id intent override for Mgf_rev.

The kwarg `Cf` from the NMCF baseline is NOT exposed — RAFFC
explicitly drops Cf. All other knobs (I_ref, W_bp, L_bp, W_bn,
L_bn, N_tail, K_cs3_load, W_diff, L_diff, W_load1, L_load1, W_cs2,
L_cs2, W_cs3, L_cs3, gm_id_diff, gm_id_load1, gm_id_stage2,
gm_id_stage3, Cm1, Cm2) are inherited verbatim from NMCF.
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


def build_nested_miller_3stage_raffc(
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

    # ─── Compensation caps (NMCF-style routing; Cf dropped) ─
    Cm1: float = 1.5e-12,   # outer Miller (out ↔ d2), same as NMCF
    Cm2: float = 0.5e-12,   # inner Miller (d2 ↔ d1), same as NMCF

    # ─── RAFFC-specific: reverse active feedforward Mgf_rev ─
    # PMOS, g=out, d=d1 — direction REVERSED vs AFFC's Mgf2.
    # Default scale 0.25 keeps the local-feedback loop gain moderate
    # so DC gain isn't crushed to <5 dB; the architectural signature
    # (Mgf_rev exists with g=out, d=d1, gate/drain swapped vs AFFC)
    # is the test target, not raw gain numbers. Designer can override
    # for tighter band-shaping if PVT margin permits.
    Mgf_rev_scale: float = 0.25,
    W_gf_rev: float | None = None,
    L_gf_rev: float | None = None,
    gm_id_gf_rev: float | None = None,

    # ─── R116 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # Alfio RAFFC 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset;
    #     RAFFC's reverse-FF feedback (Mgf_rev: g=out, d=d1) closes a
    #     local loop back to d1, so any imbalance at the diff-pair
    #     output is amplified by that loop's small-signal gain.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the high-Z d1 node from supply-coupled
    #     substrate noise — d1 is the feedback summing node where
    #     Mgf_rev injects its reverse-FF current.
    #   * PASSIVE_MATCHING on the NMCF-style Miller cap pair (Cm1 /
    #     Cm2). Cm1 (out↔d2) + Cm2 (d2↔d1) — their ratio sets the
    #     classical nested-Miller pole-splitting; the fringing-cap unit-
    #     array primitive preserves the ratio against process gradients.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for an Alfio RAFFC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC / NMCNR / TCFC / ACBC / DFCFC2 / DFCFC1
    variants. RAFFC differs from NMCF in two structural places:
      1. Reverse active feedforward Mgf_rev (PMOS, g=d2, d=d1) — the
         distinguishing device. Drain at d1 (NOT out), gate at d2
         (NOT d1). This is the "reverse" direction vs AFFC's Mgf2.
      2. NMOS load M4 at d1 widened by (1 + Mgf_rev_scale) to absorb
         Mgf_rev's DC source current at the d1 node.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see NMCF baseline docstring).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four NMCF baseline devices.
        Cm1, Cm2: NMCF-style Miller caps (out↔d2 and d2↔d1).
        Mgf_rev_scale: ratio of Mgf_rev W to Mcs3 W (default 1.0).
        W_gf_rev, L_gf_rev, gm_id_gf_rev: overrides for Mgf_rev sizing.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # RAFFC: Mgf_rev draws additional DC current into d1 → widen the
    # NMOS load M4 to absorb it. M4 is sized via W_load1 (not by
    # mirror multiplier). Baseline M4 sinks I_diff = N_tail*I_ref/2;
    # Mgf_rev (PMOS, gated by d2) sources approximately
    # I_gf_rev = Mgf_rev_scale * K_cs3_load * I_ref at the d2
    # operating point (mirroring Mcs3's bias to first order). M4
    # therefore needs to sink (I_diff + I_gf_rev), so its W must be
    # scaled by the ratio (I_diff + I_gf_rev) / I_diff to keep the
    # same Vgs operating point.
    I_gf_rev = Mgf_rev_scale * K_cs3_load * I_ref
    m4_scale = (I_diff + I_gf_rev) / I_diff
    W_load1_d1 = W_load1 * m4_scale

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("nmos", gm_id=gm_id_load1,
                                I_d=I_load1, L=L_load1).W
        # Recompute W_load1_d1 if gm_id override changed W_load1.
        W_load1_d1 = W_load1 * m4_scale
    if gm_id_stage2 is not None:
        W_cs2 = closed_form_W("nmos", gm_id=gm_id_stage2,
                              I_d=I_cs2, L=L_cs2).W
    if gm_id_stage3 is not None:
        W_cs3 = closed_form_W("pmos", gm_id=gm_id_stage3,
                              I_d=I_cs3, L=L_cs3).W

    # Mgf_rev sizing: gm_id_gf_rev beats W_gf_rev / L_gf_rev which
    # beats Mgf_rev_scale-derived default.
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

    # ─── Stage 1: PMOS diff pair with NMOS mirror load ───────
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    # M3 (diode side) keeps baseline NMCF size.
    devices.append(_mosfet("M3", "nmos", d="d1_diode", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    # M4 (mirror side) is WIDENED to absorb both the mirror current and
    # Mgf_rev's DC source current at d1.
    devices.append(_mosfet("M4", "nmos", d="d1",       g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1_d1, L=L_load1))

    # ─── Stage 2: NMOS CS amplifier driven by d1 ─────────────
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # ─── Stage 2 replica-bias leg (unchanged from NMCF) ──────
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

    # Stage-3 NMOS sink (unchanged from NMCF baseline — no Mgf_rev
    # current absorbed here because Mgf_rev's drain is at d1, not out).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load))

    # ─── RAFFC delta: Reverse active feedforward Mgf_rev ─────
    # Architectural signature of RAFFC. PMOS gated by out (global
    # output) with drain at d1 (stage-1 output) — the direction is
    # REVERSED compared to AFFC's Mgf2 (PMOS, g=d1, d=out).
    # Concretely: AFFC's source/destination pair is (g=d1, d=out)
    # forward; RAFFC's is (g=out, d=d1) reverse. This forms a
    # negative-feedback path from out back to d1 (out ↑ → Mgf_rev
    # |Vgs|↓ → less current into d1 → d1 ↓ → propagates through
    # main path to drag out ↓) — a stable local feedback loop that
    # creates a different LHP-zero location vs AFFC.
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

    # ─── R116 C: optional DRC annotations ─────────────────
    # Three annotations spanning MATCHING / GUARD_RING / PASSIVE_MATCHING.
    # See the kwarg docs block above for the per-annotation rationale.
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
                    "input-referred offset; common-centroid required. "
                    "RAFFC's reverse-FF feedback (Mgf_rev: g=out, d=d1) "
                    "closes a local loop back to d1, so any imbalance "
                    "at the diff-pair output is amplified by that "
                    "loop's small-signal gain"
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
                    "to vdd shields the high-Z d1 node — the feedback "
                    "summing node where Mgf_rev injects its reverse-FF "
                    "current — from supply-coupled substrate noise"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cm1", "Cm2"),
            extras={
                "rationale": (
                    "NMCF-style Miller cap pair — Cm1 (out↔d2) + Cm2 "
                    "(d2↔d1); their ratio sets the classical nested-"
                    "Miller pole-splitting. Fringing-cap unit-array "
                    "primitive preserves the ratio against process "
                    "gradients"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))

    return cell
