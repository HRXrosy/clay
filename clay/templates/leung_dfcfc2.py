"""
Built-in template: nested_miller_3stage_dfcfc2.

Three-stage opamp with **Damped Feed-Forward Compensation v2**
(Leung_DFCFC2, Leung 2003 v2) — the dual-feedforward variant of the
NMCF family. Two active feedforward PMOS devices both gated by the
stage-1 output `d1`: Mgf2 drains to the output (like AFFC), and
Mgf_damp drains to a new internal "damping" node `nd_damp`. A
secondary Miller cap C_damp from d1 to `nd_damp` (NOT to out)
creates a damping path that flattens the second pole.

Reference
=========

Leung_DFCFC2_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Leung_DFCFC2_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm12 VOUT  net050 VDDA VDDA pfet  → Mgf2 (active FF PMOS, gated by
                                       d1, drains to out — like AFFC)
  xm10 net2  net050 VDDA VDDA pfet  → Mgf_damp (DFCFC2-unique PMOS,
                                       gated by d1, drains to the
                                       damping node nd_damp)
  xm22 net2  VB4    GNDA GNDA nfet  → Mlcs_damp (NMOS sink of Mgf_damp's
                                       DC current, gate=nbias)

Plus the usual diff pair, stage-2 PMOS CS, stage-3 PMOS CS + NMOS
sink, and two Miller caps (C1 from d1 to VOUT, C2 from d1 to
nd_damp).

The DFCFC2 architectural fingerprint:
  1. A SECOND active feedforward PMOS `Mgf_damp` — same gate (d1) as
     Mgf2 but a DIFFERENT drain (the damping node nd_damp). This is
     what makes it "dual" feedforward.
  2. A NEW internal node `nd_damp` carrying signal from the second
     feedforward path. NMCF / AFFC / TCFC / SMC / CFCC / NMCNR / ACBC
     all have no such node.
  3. A SECOND active sink `Mlcs_damp` providing the DC current path for
     Mgf_damp from nd_damp to vss.
  4. The inner Miller cap `Cm2_damp` routes from d1 to nd_damp (the new
     node), NOT from d2 to d1 like NMCF. This is the "damping" path —
     the cap loads nd_damp with stage-1 signal, and the second pole
     (at d2) is partially decoupled from the outer feedback loop.

Topology delta vs `nested_miller_3stage`
========================================

Stage 1, stage 2, and stage 3 (signal path including Mcs3 and the
single NMOS sink Mlcs3) inherit the NMCF baseline. The deltas:

**Active feedforward Mgf2 (NEW, shared with AFFC):**
  Mgf2 : PMOS, d=out, g=d1, s=vdd

**Active feedforward Mgf_damp (NEW, unique to DFCFC2):**
  Mgf_damp : PMOS, d=nd_damp, g=d1, s=vdd

**Active sink Mlcs_damp (NEW, unique to DFCFC2):**
  Mlcs_damp : NMOS, d=nd_damp, g=nbias, s=vss

  Provides the DC current path for Mgf_damp. Sized to absorb
  Mgf_damp's DC current (same trick as widening Mlcs3 for Mgf2 in
  AFFC, but as a separate device because nd_damp is not connected to
  out).

**Compensation: TWO Miller caps (no Cf):**
  NMCF:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1
    Cf  : out ↔ d1    (FF cap for RHP-zero cancellation)

  DFCFC2:
    Cm1      : out      ↔ d1   (skip-over outer Miller, spans stages 2+3
                                — same routing as SMC/AFFC's outer cap
                                because Mgf2 provides the active FF path)
    Cm2_damp : nd_damp  ↔ d1   (damping Miller cap to the new internal
                                node; this is the DFCFC2 architectural
                                signature)
    (Cf is dropped — the dual active feedforward Mgf2 + Mgf_damp combine
     with the damping path through Cm2_damp / nd_damp to suppress the
     RHP zero without needing a passive Cf)

Schematic (compensation + new devices only)
-------------------------------------------

  NMCF (compensation):              DFCFC2 (Leung 2003 v2):
    Cm1 out↔d2                        Mgf2 PMOS (g=d1, d=out)
    Cm2 d2↔d1                         Mgf_damp PMOS (g=d1, d=nd_damp)
    Cf  out↔d1                        Mlcs_damp NMOS (d=nd_damp, g=nbias)
                                      Cm1      out↔d1 (outer Miller)
                                      Cm2_damp nd_damp↔d1 (damping cap)

Why dual feedforward + damping helps (transfer-function level)
--------------------------------------------------------------

The single-FF AFFC scheme moves the LHP zero set by gm_f2 / gm_3 to
provide phase margin. But the second pole at d2 (driven by gm_3 /
C_d2) sits between the dominant pole and GBW — if you push GBW too
high, p2 starts eating phase margin again. Leung 2003's DFCFC2
inserts a second feedforward path through Mgf_damp that drives a
parallel low-Z node nd_damp; the damping cap Cm2_damp injects
stage-1 signal into nd_damp directly, which through Mlcs_damp's
small-signal gm_load damps the high-frequency peak that would
otherwise form at p2. Net effect:

  - Two LHP zeros (one from each Mgf) instead of one
  - Damped second pole (lower Q from the parallel nd_damp path)
  - Wider GBW achievable at fixed phase margin compared to AFFC

Note: the "damping" terminology refers to the damping of the
amplifier's small-signal response, not literal physical damping.
The mechanism is parallel-pole-cancellation through the second
feedforward path acting as a series-feedback loop at d1.

Honest scope (what you should NOT expect)
-----------------------------------------

DFCFC2 is a structural compensation refinement that genuinely changes
the netlist (new active Mgf_damp + Mlcs_damp + new internal node
nd_damp + re-routed damping cap). At the relaxed 5 dB / 10 kHz spec
floor on gf180mcu 3.3 V, the PVT envelope will land comfortably;
the actual physical benefit of the damping path (PM improvement) is
sensitive to Mgf_damp_scale and Cm2_damp value. Tests in
`tests/test_leung_dfcfc2_distinct.py` lock the architectural
distinction (Mgf_damp + Mlcs_damp exist, nd_damp net exists, Cm2_damp
routes to nd_damp not d2, Cf does NOT exist) rather than asserting
specific SPICE numerics.

Knobs
-----

DFCFC2 adds the following new kwargs on top of the NMCF baseline:

  Mgf2_scale: float = 1.0
      Sizing ratio of Mgf2 / Mcs3. Default 1.0 → gm_f2 ≈ gm_3 at
      the same gm/Id (same as AFFC).
  W_gf2, L_gf2: optional explicit overrides for Mgf2 sizing.
  Mgf_damp_scale: float = 0.5
      Sizing ratio of Mgf_damp / Mcs3. Default 0.5 → the second
      feedforward path is half-strength compared to Mgf2 (Leung's
      typical choice for v2 since the damping path's primary
      function is pole-damping, not gain contribution).
  W_gf_damp, L_gf_damp: optional explicit overrides.
  Cm2_damp: float = 0.5e-12
      Damping Miller cap value (d1 ↔ nd_damp). Default 0.5 pF.

The kwarg `Cf` from the NMCF baseline is NOT exposed — DFCFC2
explicitly drops Cf. The kwarg `Cm2` is RE-INTERPRETED in DFCFC2
as the damping cap Cm2_damp (since the textbook Cm2 routing d1↔d2
doesn't exist in DFCFC2). The kwarg Cm2_damp is also exposed for
clarity.

All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load,
W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3,
gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3, Cm1) are
inherited verbatim from NMCF.
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


def build_nested_miller_3stage_dfcfc2(
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

    # ─── Compensation caps (Cm1 outer + Cm2_damp damping) ───
    Cm1: float = 1.5e-12,
    Cm2_damp: float = 0.5e-12,

    # ─── DFCFC2-specific: active feedforward Mgf2 (shared with AFFC) ─
    Mgf2_scale: float = 1.0,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # ─── DFCFC2-specific: second feedforward Mgf_damp ───────
    # Drains into the damping node nd_damp instead of out. Default
    # scale 0.5 → half the strength of Mgf2 (Leung's typical v2
    # choice; the damping path's primary role is pole-damping, not
    # gain contribution).
    Mgf_damp_scale: float = 0.5,
    W_gf_damp: float | None = None,
    L_gf_damp: float | None = None,
    gm_id_gf_damp: float | None = None,

    # ─── R116 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # Leung DFCFC2 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset; the
    #     dual feedforward paths (Mgf2 to out, Mgf_damp to nd_damp) are
    #     both gated by d1 and amplify any imbalance at M2's drain.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the high-Z d1 node — the common gate for both
    #     feedforward devices and one terminal of Cm1 + Cm2_damp — from
    #     supply-coupled substrate noise.
    #   * SYMMETRIC_LAYOUT on the dual feedforward pair (Mgf2 /
    #     Mgf_damp). Both PMOS devices are gated by d1 and span the
    #     same gm contribution to the small-signal transfer function;
    #     the damping behavior depends on Mgf2's gm tracking Mgf_damp's
    #     gm over PVT. A vertical symmetric axis with anchor at d1
    #     (the shared gate) gives both devices the same gate-routing
    #     parasitic, preserving the gm ratio.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Leung DFCFC2 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC / NMCNR / TCFC / ACBC variants. DFCFC2
    differs from NMCF in three structural places:
      1. Active feedforward Mgf2 (PMOS, g=d1, d=out) — shared with AFFC
      2. Second active feedforward Mgf_damp (PMOS, g=d1, d=nd_damp) —
         the dual-FF distinguishing device
      3. Active sink Mlcs_damp (NMOS, d=nd_damp, g=nbias) — DC current
         path for the second FF
      4. Re-routed compensation: Cm1 (out↔d1 skip-over) + Cm2_damp
         (nd_damp↔d1 damping cap); Cf dropped

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see NMCF baseline docstring).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four NMCF baseline devices.
        Cm1: outer Miller cap (out ↔ d1 skip-over). Default 1.5 pF.
        Cm2_damp: damping Miller cap (nd_damp ↔ d1). Default 0.5 pF.
        Mgf2_scale: ratio of Mgf2 W to Mcs3 W (default 1.0).
        W_gf2, L_gf2: optional explicit overrides for Mgf2 sizing.
        gm_id_gf2: gm/Id intent override for Mgf2.
        Mgf_damp_scale: ratio of Mgf_damp W to Mcs3 W (default 0.5).
        W_gf_damp, L_gf_damp: optional explicit overrides.
        gm_id_gf_damp: gm/Id intent override for Mgf_damp.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # DFCFC2: Mgf2 draws additional DC current → widened NMOS sink at
    # out (same trick as AFFC). Mgf_damp draws DC current too, but it
    # is absorbed by a SEPARATE NMOS sink Mlcs_damp (not the output
    # sink, because Mgf_damp drains to nd_damp not out).
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref
    I_gf_damp = Mgf_damp_scale * K_cs3_load * I_ref

    # NMOS sink at output (Mlcs3) widened by (1 + Mgf2_scale) to absorb
    # Mcs3 + Mgf2 DC currents at out — same as AFFC.
    K_cs3_load_total = K_cs3_load * (1 + Mgf2_scale)
    # NMOS sink at nd_damp (Mlcs_damp) sized to match Mgf_damp DC
    # current. K multiplier rounded up.
    K_damp_load = max(int(round(I_gf_damp / I_ref)), 1)

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

    # Mgf_damp sizing
    if L_gf_damp is None:
        L_gf_damp = L_cs3
    if gm_id_gf_damp is not None:
        W_gf_damp = closed_form_W("pmos", gm_id=gm_id_gf_damp,
                                  I_d=I_gf_damp, L=L_gf_damp).W
    elif W_gf_damp is None:
        W_gf_damp = W_cs3 * Mgf_damp_scale

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
    devices.append(_mosfet("M3", "nmos", d="d1_diode", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos", d="d1",       g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

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

    # ─── DFCFC2 delta 1: Active feedforward Mgf2 (shared w/AFFC) ──
    # Same structural role as AFFC's Mgf2 — PMOS in parallel with
    # Mcs3 but gated by d1 instead of d2. Creates the primary LHP
    # zero.
    devices.append(_mosfet("Mgf2", "pmos", d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # NMOS sink at output — widened to absorb both Mcs3 and Mgf2 DC
    # currents (same trick as AFFC).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── DFCFC2 delta 2: SECOND active feedforward Mgf_damp ──
    # The architectural signature of DFCFC2. PMOS gated by d1 (same as
    # Mgf2) but draining to a NEW internal node nd_damp, not out. This
    # is what makes it "dual" feedforward.
    devices.append(_mosfet("Mgf_damp", "pmos",
                            d="nd_damp", g="d1", s="vdd", b="vdd",
                            W=W_gf_damp, L=L_gf_damp))

    # NMOS sink at nd_damp: provides the DC current path for Mgf_damp.
    # Sized to match Mgf_damp's DC current via K_damp_load multiplier
    # on Mbn. Sources the small-signal damping action through its 1/gm
    # loading.
    devices.append(_mosfet("Mlcs_damp", "nmos",
                            d="nd_damp", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_damp_load))

    # ─── Compensation: Cm1 outer + Cm2_damp damping (no Cf) ──
    # DFCFC2's distinctive routing:
    #   Cm1: out ↔ d1 (skip-over outer Miller, spans stages 2+3 like
    #        AFFC because both have active FF from d1 to out)
    #   Cm2_damp: nd_damp ↔ d1 (damping Miller cap to the new internal
    #        node — this is the DFCFC2 architectural signature)
    # Cf is dropped — the dual active feedforward + damping path
    # together provide PM recovery structurally.
    devices.append(_cap("Cm1",      p="out",     n="d1", value=Cm1))
    devices.append(_cap("Cm2_damp", p="nd_damp", n="d1", value=Cm2_damp))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R116 C: optional DRC annotations ─────────────────
    # Three annotations spanning MATCHING / GUARD_RING / SYMMETRIC_LAYOUT.
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
                    "The dual feedforward paths (Mgf2 to out, Mgf_damp "
                    "to nd_damp) are both gated by d1 and amplify any "
                    "imbalance at M2's drain"
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
                    "to vdd shields the high-Z d1 node — the common "
                    "gate for both feedforward devices and one terminal "
                    "of Cm1 + Cm2_damp — from supply-coupled substrate "
                    "noise"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.SYMMETRIC_LAYOUT,
            devices=("Mgf2", "Mgf_damp"),
            extras={
                "rationale": (
                    "Dual feedforward PMOS pair — both gated by d1 and "
                    "contributing parallel gm to the small-signal "
                    "transfer function. Damping behavior depends on "
                    "Mgf2's gm tracking Mgf_damp's gm over PVT; a "
                    "vertical symmetric axis with anchor at d1 (the "
                    "shared gate) gives both devices the same gate-"
                    "routing parasitic, preserving the gm ratio"
                ),
                "axis": "vertical",
                "anchor": "d1",
            },
        ))

    return cell
