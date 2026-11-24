"""
Built-in template: nested_miller_3stage_acbc.

Three-stage opamp with **Active Capacitance + Bias Compensation**
(Peng_ACBC) — an NMCF-family compensation scheme that augments
classical Miller compensation with an auxiliary active bias-comp
loop. A 4th transconductor (gm4, NMOS) gated by the stage-2 output
feeds current into a dedicated bias-compensation node `nb`, which
is held at a PMOS-diode-set voltage by a separate bias chain. The
inner Miller cap terminates at `nb` instead of vdd; the active
gm4 transconductor in parallel with the bias-comp node makes the
effective Miller capacitance much larger than the physical Cm — the
"active capacitance" effect.

Reference
=========

Peng_ACBC_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Peng_ACBC_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm12 VOUT net050 VDDA VDDA pfet      → Mgf2 (active feedforward
                                          PMOS gated by d1, like AFFC)
  xm24 net1 net043 GNDA GNDA nfet      → Mgm4 (active bias-comp
                                          NMOS gated by d2; sinks
                                          current into the active-cap
                                          node nb)
  xm59 net1 net1 VDDA VDDA pfet        → Mnb_diode (PMOS diode at nb)
  xm8  net1 net013 VDDA VDDA pfet      → Mnb_drv (PMOS mirror sourcing
                                          current into nb from pbias)

Plus the usual diff pair, stage-2 PMOS CS, stage-3 PMOS CS, and two
Miller caps (C0 from d1 to VOUT, C1 from stage-2 output net049 to
the active-cap node net1).

The ACBC architectural fingerprint:
  1. An **active bias-comp node** `nb` (= AnalogGym `net1`) held at a
     PMOS-diode-set voltage (≈ vdd − |Vgs_p|) by Mnb_diode +
     Mnb_drv. NMCF / AFFC / TCFC / SMC / CFCC / NMCNR have no such
     dedicated bias-comp node.
  2. A **4th active transconductor** `Mgm4` (NMOS, gate=d2, drain=nb,
     source=vss) which sinks signal current into the bias-comp node
     proportional to V(d2). This is the "active capacitance" half of
     the name — gm4 multiplies the small physical Cm into a much
     larger effective Miller load on the d2 node.
  3. An **active feedforward** PMOS `Mgf2` gated by d1 (like AFFC),
     sourcing current into the output. Both ACBC and AFFC share this
     device; ACBC's distinction is the bias-comp loop on top.
  4. The **inner Miller cap** Cm2 terminates at `nb` instead of d1
     (NMCF) or vdd (canonical Miller). This routing is what makes the
     active-cap multiplication actually appear at d2.

Topology delta vs `nested_miller_3stage`
========================================

Stage 1 inherits the NMCF baseline verbatim. The deltas:

**Bias-comp chain (new):**
  Mnb_diode : PMOS diode at nb (PMOS-diode reference for the active-cap node)
  Mnb_drv   : PMOS mirror sourcing I_ref into nb from pbias

  `nb` settles at ≈ vdd − |Vgs_p|. The pair of devices (diode + driver)
  is structurally similar to TCFC's nd3_diode + Mlcs3_diode_drv chain
  but routed PMOS-up to vdd, not NMOS-down to vss.

**Active feedforward Mgf2 (NEW, shared with AFFC):**
  Mgf2 : PMOS, d=out, g=d1, s=vdd, b=vdd

  Gate driven by d1 (stage-1 output) so the feedforward path bypasses
  stages 2 AND 3. Identical to HoiLee_AFFC's Mgf2. ACBC adds gm4 on top
  of this, making it a strict superset of AFFC in device count.

**Active bias-comp transconductor Mgm4 (NEW, unique to ACBC):**
  Mgm4 : NMOS, d=nb, g=d2, s=vss, b=vss

  NMOS gain device driven by d2; sinks signal current into the nb
  node. Combined with the inner Miller cap Cm2 routed nb↔d2, this
  forms the active-cap loop: when V(d2) goes up by ΔV, gm4 sinks
  Δi = gm4·ΔV out of nb, which has to be supplied by the small
  PMOS-diode impedance (1/gm_p), so V(nb) drops by Δi/gm_p =
  gm4·ΔV/gm_p. Cm2 sees V(d2) - V(nb) = (1 + gm4/gm_p)·ΔV → looks
  like Cm2·(1 + gm4/gm_p) to d2. With gm4/gm_p ≈ 5-10, this is the
  "active capacitance" boost over physical Cm2.

**Compensation: TWO Miller caps, inner routed to nb (no Cf):**
  NMCF:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1
    Cf  : out ↔ d1    (FF cap for RHP-zero cancellation)

  ACBC:
    Cm1 : out ↔ d1    (skip-over outer Miller, like SMC's signal path —
                       but with the d2-bias-comp loop adding active
                       multiplication this routing makes physical sense)
    Cm2 : nb  ↔ d2    (inner Miller routed to active-cap node)
    (Cf is dropped — the active feedforward Mgf2 + the bias-comp loop
     handle PM recovery structurally, similar to TCFC)

Stage 3 sink (unchanged from NMCF baseline):
  Mlcs3 : NMOS, d=out, g=nbias, s=vss   (single device sink — ACBC
                                          uses bias-comp loop instead
                                          of cascoded sink for PM)

The NMOS sink is widened internally (like AFFC) to absorb both Mcs3
and Mgf2 DC currents.

Schematic (compensation + new devices only)
-------------------------------------------

  NMCF (compensation):              ACBC (Peng):
    Cm1 out↔d2 / Cm2 d2↔d1            Cm1 out↔d1 (skip-over outer)
    Cf  out↔d1                        Cm2 nb↔d2 (inner at bias-comp node)
                                      Mgf2 PMOS, g=d1, d=out (active FF)
                                      Mgm4 NMOS, g=d2, d=nb (active gm4)
                                      Mnb_diode + Mnb_drv (bias-comp chain)

Why ACBC matters (transfer-function level)
------------------------------------------

The classical Miller cap Cm2 between d2 and d1 multiplies by stage-2
gain only — order 30-50 dB on a typical gf180mcu sizing. ACBC's
bias-comp loop multiplies Cm2 by gm4 / gm_p (the PMOS-diode
impedance ratio), which on a typical sizing is 5-10× more. Net
effect: Cm2's pole-splitting action is stronger per pF of physical
silicon, so smaller compensation caps suffice for the same dominant-
pole location → less area, less load on d2.

Combined with the AFFC-style active feedforward Mgf2 → out, the loop
gets a structural LHP zero at gm_f2 · g_o3 / s · C_out from Mgf2
(same as AFFC), plus the active-cap-boosted dominant pole, plus the
suppressed RHP zero from Cm1's skip-over routing (no Cm2 between d2
and d1 to create one). Three structural compensation mechanisms in
one cell — this is what makes ACBC a "compensation kitchen sink"
relative to its simpler siblings.

Net effect on PVT numbers (qualitative):
  - DC gain: slightly increased over NMCF/AFFC (the bias-comp loop
    raises output impedance through the gm4 path's effect on d2's
    load impedance)
  - GBW: depends on Cm1 ; with the same physical Cm1=1.5 pF and
    skip-over routing GBW is in the same ballpark as SMC (similar
    Miller multiplication via stage-2 gain only on that path)
  - PM: improved relative to plain NMCF due to active feedforward Mgf2
    contributing a LHP zero

Honest scope (what you should NOT expect)
-----------------------------------------

ACBC is a structurally-rich compensation scheme that genuinely
changes the netlist (new active gm4 transconductor, new bias-comp
chain, new internal node nb, re-routed Miller caps). At the relaxed
5 dB / 10 kHz spec floor on gf180mcu 3.3 V, the PVT envelope will
land comfortably; whether the LHP zero from Mgf2 produces a
measurable PM bump vs NMCF baseline depends on gm4 / gm_p ratio and
sizing details. Tests in `tests/test_peng_acbc_distinct.py` lock the
architectural distinction (Mgm4 + Mgf2 exist, nb net exists, Mnb_diode
exists, Cf does NOT exist) rather than asserting specific SPICE
numerics.

Knobs
-----

ACBC adds the following new kwargs on top of the NMCF baseline:

  Mgf2_scale: float = 1.0
      Sizing ratio of Mgf2 / Mcs3. Default 1.0 → gm_f2 ≈ gm_3 at
      the same gm/Id (like AFFC). Sizing range 0.5–2.0 reasonable.
  W_gf2, L_gf2: optional explicit overrides for Mgf2 sizing.
  Mgm4_scale: float = 1.0
      Sizing ratio of Mgm4 / Mcs2. Default 1.0 → gm_4 ≈ gm_2. Larger
      values increase the active-cap multiplication factor at the
      cost of more DC current through the bias-comp chain.
  W_gm4, L_gm4: optional explicit overrides for Mgm4 sizing.
  W_bp_nb, L_bp_nb: float = 4 µm / 1 µm
      PMOS-diode sizing for Mnb_diode (the bias-comp node reference).

The kwargs `Cf` from the NMCF baseline is NOT exposed — ACBC
explicitly drops Cf. All other kwargs (I_ref, W_bp, L_bp, W_bn,
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


def build_nested_miller_3stage_acbc(
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

    # ─── Compensation caps (Cm1, Cm2 only — no Cf in ACBC) ──
    Cm1: float = 1.5e-12,
    Cm2: float = 0.5e-12,

    # ─── ACBC-specific: active feedforward Mgf2 (shared with AFFC) ─
    Mgf2_scale: float = 1.0,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # ─── ACBC-specific: active bias-comp transconductor Mgm4 ─
    # NMOS, gate=d2, drain=nb, source=vss. Mgm4_scale sets W ratio
    # vs Mcs2 (default 1.0 → gm_4 ≈ gm_2). Larger increases the
    # active-cap multiplication but adds DC current to the bias-comp
    # chain.
    Mgm4_scale: float = 1.0,
    W_gm4: float | None = None,
    L_gm4: float | None = None,
    gm_id_gm4: float | None = None,

    # ─── ACBC-specific: bias-comp node diode (Mnb_diode) ────
    # PMOS-diode reference setting V(nb) ≈ vdd − |Vgs_p|. Sized
    # identically to Mbp by default so both diodes pass I_ref at
    # the same |Vov|.
    W_bp_nb: float = 4e-6,
    L_bp_nb: float = 1e-6,

    # ─── R115 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # Peng ACBC 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the high-Z d1 node and the bias-comp node nb
    #     from supply-coupled substrate noise injected by Mgm4's switching
    #     drain current at nb.
    #   * MATCHING on the NMOS stage-1 mirror legs (M3 / M4). Both legs
    #     share gate `d1_diode` at identical W/L; 1:1 mirror — pure
    #     MATCHING (not CURRENT_MIRROR_MATCHING which targets 1:N
    #     unit-cell arrays). Common-centroid required so the mirror
    #     ratio is preserved across local-mismatch / OD-edge gradients;
    #     ACBC's active-cap loop assumes a balanced stage-1 mirror as
    #     the reference for the gm4 bias-comp feedback.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Peng ACBC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC / NMCNR / TCFC variants. ACBC differs from
    NMCF in four structural places:
      1. Active feedforward Mgf2 (PMOS, g=d1, d=out) — shared with AFFC
      2. Active bias-comp transconductor Mgm4 (NMOS, g=d2, d=nb) —
         unique to ACBC
      3. Bias-comp chain: Mnb_diode (PMOS-diode at nb) + Mnb_drv (PMOS
         driver pushing I_ref into nb from pbias) — unique to ACBC
      4. Re-routed Miller caps: Cm1 (out↔d1 skip-over) + Cm2 (nb↔d2);
         Cf dropped

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see NMCF baseline docstring).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four NMCF baseline devices.
        Cm1, Cm2: Miller cap values (no Cf).
        Mgf2_scale: ratio of Mgf2 W to Mcs3 W (default 1.0). gm/Id
            intent override via gm_id_gf2.
        W_gf2, L_gf2: optional explicit overrides for Mgf2 sizing.
        Mgm4_scale: ratio of Mgm4 W to Mcs2 W (default 1.0). gm/Id
            intent override via gm_id_gm4.
        W_gm4, L_gm4: optional explicit overrides for Mgm4 sizing.
        W_bp_nb, L_bp_nb: PMOS-diode sizing for Mnb_diode (default
            4 µm / 1 µm, matches Mbp).

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # ACBC: Mgf2 draws additional DC current proportional to Mgf2_scale,
    # absorbed by widening the NMOS sink Mlcs3 (same as AFFC).
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref
    K_cs3_load_total = K_cs3_load * (1 + Mgf2_scale)

    # ACBC: Mgm4 draws DC current into nb proportional to Mgm4_scale ·
    # I_cs2. This current must be supplied by Mnb_drv into nb. We size
    # Mnb_drv (PMOS mirror of Mbp) with a multiplier matching gm4's
    # current so the bias-comp loop is current-balanced at DC.
    I_gm4 = Mgm4_scale * I_cs2
    # Mnb_drv multiplier: floor(I_gm4 / I_ref). Round up to keep nb
    # in saturation (PMOS-diode delivers slightly more current than
    # gm4 sinks; the extra is sunk by Mnb_diode below).
    M_nb_drv = max(int(round(I_gm4 / I_ref)), 1)

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

    # Mgf2 sizing: gm_id_gf2 beats W_gf2 / L_gf2 which beats
    # Mgf2_scale-derived default.
    if L_gf2 is None:
        L_gf2 = L_cs3
    if gm_id_gf2 is not None:
        W_gf2 = closed_form_W("pmos", gm_id=gm_id_gf2,
                              I_d=I_gf2, L=L_gf2).W
    elif W_gf2 is None:
        W_gf2 = W_cs3 * Mgf2_scale

    # Mgm4 sizing: gm_id_gm4 beats W_gm4 / L_gm4 which beats
    # Mgm4_scale-derived default. Mgm4 is NMOS so uses 'nmos' solver.
    if L_gm4 is None:
        L_gm4 = L_cs2
    if gm_id_gm4 is not None:
        W_gm4 = closed_form_W("nmos", gm_id=gm_id_gm4,
                              I_d=I_gm4, L=L_gm4).W
    elif W_gm4 is None:
        W_gm4 = W_cs2 * Mgm4_scale

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

    # ─── ACBC delta 1: Active feedforward Mgf2 ──────────────
    # Same structural role as AFFC's Mgf2 — PMOS in parallel with Mcs3
    # but gated by d1 instead of d2. Creates the structural LHP zero
    # that pushes phase margin without needing a passive Cf cap.
    devices.append(_mosfet("Mgf2", "pmos", d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # NMOS sink at output — widened to absorb both Mcs3 and Mgf2 DC
    # currents (same trick as AFFC).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── ACBC delta 2: Active bias-comp node nb + Mgm4 ──────
    # The architectural signature of ACBC. Mgm4 sinks signal current
    # from d2 into the bias-comp node nb; Mnb_diode + Mnb_drv hold nb
    # at vdd − |Vgs_p|. Combined with Cm2 routed nb↔d2, this forms the
    # "active capacitance" loop: the small physical Cm2 sees an
    # effective multiplication by (1 + gm_4 / gm_p) at d2.
    #
    # Mnb_diode: PMOS diode at nb, sets V(nb) ≈ vdd − |Vgs_p_Mnb_diode|.
    # Body tied to source so the |Vgs_p| drop is well-defined (no
    # body-effect Vth shift).
    devices.append(_mosfet("Mnb_diode", "pmos",
                            d="nb", g="nb", s="vdd", b="vdd",
                            W=W_bp_nb, L=L_bp_nb))

    # Mnb_drv: PMOS mirror of Mbp, sources DC current into nb to balance
    # gm4's sink. Multiplier M_nb_drv chosen so I_Mnb_drv ≈ I_gm4 at
    # the bias point.
    devices.append(_mosfet("Mnb_drv", "pmos",
                            d="nb", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=M_nb_drv))

    # Mgm4: NMOS active transconductor — the bias-comp loop's gain
    # element. Gate=d2, drain=nb, source=vss. Acts as a gm-cell that
    # multiplies Cm2's effective Miller capacitance.
    devices.append(_mosfet("Mgm4", "nmos",
                            d="nb", g="d2", s="vss", b="vss",
                            W=W_gm4, L=L_gm4))

    # ─── Compensation: Cm1 skip-over + Cm2 to nb (no Cf) ────
    # ACBC's distinctive routing:
    #   Cm1: out ↔ d1 (skip-over outer Miller, spans stage 2 + 3 like
    #        Fan SMC — but with the active-cap loop on Cm2 this routing
    #        gives a structurally different transfer function)
    #   Cm2: nb ↔ d2 (inner Miller to the bias-comp node, gets multiplied
    #        by (1 + gm_4 / gm_p) due to the active-cap loop)
    # Cf is dropped — the active feedforward Mgf2 + bias-comp loop
    # combine to suppress the RHP zero without needing Cf.
    devices.append(_cap("Cm1", p="out", n="d1", value=Cm1))
    devices.append(_cap("Cm2", p="nb",  n="d2", value=Cm2))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R115 C: optional DRC annotations ─────────────────
    # Three annotations spanning MATCHING / GUARD_RING / MATCHING.
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
                    "PMOS tail current source (M=N_tail) carries the "
                    "entire stage-1 bias current; nwell guard ring tied "
                    "to vdd shields the high-Z d1 node and the bias-comp "
                    "node nb from supply-coupled substrate noise injected "
                    "by Mgm4's switching drain current at nb"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "NMOS stage-1 mirror legs (1:1) — common-centroid "
                    "required so mirror ratio is preserved across local-"
                    "mismatch / OD-edge gradients; ACBC's active-cap loop "
                    "assumes a balanced stage-1 mirror as the reference "
                    "for the gm4 bias-comp feedback"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
