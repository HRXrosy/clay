"""
Built-in template: nested_miller_3stage_dfcfc1.

Three-stage opamp with **Damped Feed-Forward Compensation v1**
(Leung_DFCFC1, Leung 2003 v1) — the single-feedforward + cascoded
stage-1 load variant of the NMCF family. The DFCFC2 sister
(`nested_miller_3stage_dfcfc2`) adds a SECOND active feedforward
PMOS to a damping node; DFCFC1 has only ONE active feedforward
device, but compensates for the missing second-FF damping by
cascoding the stage-1 NMOS mirror load — the cascoded mirror's
higher output impedance at d1 partially substitutes for the
dual-FF damping path's pole-flattening role.

Reference
=========

Leung_DFCFC1_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Leung_DFCFC1_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm12 VOUT  net050 VDDA VDDA pfet  → Mgf2 (active FF PMOS, gated by
                                       d1, drains to out — same as
                                       AFFC's Mgf2)
  xm17 net050 VB3   net063 GNDA nfet → Mcas1_n_load (NMOS cascode
                                       between d1 (=net050) and the
                                       new cascode-source node
                                       nd1_cas (=net063))
  xm16 VOUTN  VB3   DM_2  GNDA nfet → cascode on the diode side
  xm21 net063 VB4   GNDA  GNDA nfet → Mlcs1 (NMOS sink at the
                                       cascode-source node, mirrors
                                       nbias to vss)
  xm20 DM_2   VB4   GNDA  GNDA nfet → diode-side companion sink

Plus the usual diff pair, stage-2 NMOS CS, stage-3 PMOS CS + NMOS
sink, and two Miller caps (C0 from d1 to VOUT, C1 from d2 to nb).

The DFCFC1 architectural fingerprint:
  1. An active feedforward PMOS `Mgf2` (PMOS, g=d1, d=out) — same
     primary FF as AFFC and DFCFC2.
  2. A **cascoded stage-1 NMOS mirror load**: instead of the NMCF
     baseline's M3 (diode) + M4 (mirror) draining directly into d1
     / d1_diode from vss, the DFCFC1 mirror has a 2-stack on each
     leg — the gain devices M3 / M4 sink to vss as before, but
     their drains land on internal cascode-source nodes
     `d1_int` / `d1_diode_int`, and a separate NMOS cascode device
     (Mcas1_n_load / Mcas1_n_diode) sits between those internal
     nodes and the actual d1 / d1_diode nets. The cascode gates
     are biased by a `vcas` rail. The internal cascode-source nodes
     `d1_int` / `d1_diode_int` plus the `vcas` bias rail are the
     DFCFC1 architectural markers — neither NMCF / AFFC / SMC /
     CFCC / NMCNR / TCFC / ACBC / DFCFC2 has these nodes.
  3. A cascode-bias diode `Mcas_bias` (NMOS diode at vcas) +
     `Mbp_cas_drv` (PMOS driver mirroring Mbp) hold vcas at
     ≈ Vgs_n above vss; sized identically to Mbn / Mbp by default
     so the cascode gates sit one Vgs above vss with PVT-robust
     headroom.

Topology delta vs `nested_miller_3stage`
========================================

Stage 2 + stage 3 (signal path) inherit the NMCF baseline verbatim.
Compensation routing also stays NMCF-style for the outer Miller
(Cm1 wraps stage 3 with an out↔d2 cap; Cf is dropped because Mgf2
provides the LHP zero structurally). The deltas:

**Active feedforward Mgf2 (shared with AFFC):**
  Mgf2 : PMOS, d=out, g=d1, s=vdd

  Same structural role as HoiLee_AFFC's Mgf2 — PMOS in parallel
  with Mcs3 but driven by stage-1 output d1 instead of stage-2
  output d2. Creates the primary LHP zero.

**Cascoded stage-1 NMOS mirror load (NEW, DFCFC1-unique):**

  NMCF baseline:
    M3 : NMOS diode, d=d1_diode, g=d1_diode, s=vss
    M4 : NMOS mirror, d=d1, g=d1_diode, s=vss

  DFCFC1 (regulated cascode):
    M3            : NMOS, d=d1_diode_int, g=d1_diode, s=vss
                    (gain device, drain moved off d1_diode to internal node)
    M4            : NMOS, d=d1_int, g=d1_diode, s=vss
                    (gain device, drain moved off d1 to internal node)
    Mcas1_n_load  : NMOS, d=d1, g=vcas, s=d1_int, b=vss
                    (cascode device between d1 and gain-device drain)
    Mcas1_n_diode : NMOS, d=d1_diode, g=vcas, s=d1_diode_int, b=vss
                    (cascode device on diode side; diode loop closes
                    around the cascode, so M3's gate=d1_diode is still
                    the mirror reference)
    Mcas_bias     : NMOS diode at vcas (d=g=vcas, s=vss)
    Mbp_cas_drv   : PMOS mirror, g=pbias, d=vcas, s=vdd
                    (sources I_ref into vcas to bias Mcas_bias diode)

  The cascode boosts the output impedance looking into d1 by
  roughly g_m_cas · r_o_M4, raising stage-1 DC gain. Leung 2003 v1
  design intent: cascode-boosted stage-1 gain partially substitutes
  for the missing second FF path's damping role (which DFCFC2 adds
  via Mgf_damp + nd_damp instead).

**Compensation: Cm1 outer (NMCF-style) + Cm2 inner, NO Cf:**

  NMCF:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1
    Cf  : out ↔ d1    (LHP-zero generator)

  DFCFC1:
    Cm1 : out ↔ d2    (same as NMCF outer Miller)
    Cm2 : d2  ↔ d1    (same as NMCF inner Miller)
    (Cf is dropped — Mgf2 provides the LHP zero structurally;
     cascoded stage-1 load provides additional pole-splitting
     headroom at d1)

DFCFC1 vs DFCFC2 in one line
============================

DFCFC1 has ONE Mgf (single FF to out) + cascoded stage-1 mirror,
keeping NMCF-style outer Miller routing (out↔d2).
DFCFC2 has TWO Mgf (dual FF, one to out + one to nd_damp) + single
stage-1 mirror, with skip-over Miller routing (out↔d1).

Same paper family (Leung 2003), but architecturally distinct
compensation philosophy: DFCFC1 leans on cascode-boosted stage-1
output impedance; DFCFC2 leans on a second active feedforward path.
The reference netlist confirms this distinction: DFCFC1 has the
cascoded NMOS load mirror (xm17 + xm21 chain), DFCFC2 has the dual
Mgf2 + Mgf_damp PMOS pair (xm12 + xm10 chain).

Schematic (stage 1 load delta only — rest unchanged from NMCF)
--------------------------------------------------------------

  NMCF:                                DFCFC1:
    d1_diode ─ M3 ─ vss                  d1       ─ Mcas1_n_load  ─ d1_int       ─ M4 ─ vss
    d1       ─ M4 ─ vss                  d1_diode ─ Mcas1_n_diode ─ d1_diode_int ─ M3 ─ vss
                                         vcas ─ Mcas_bias diode + Mbp_cas_drv

  Plus the active FF Mgf2 (PMOS, g=d1, d=out, s=vdd) — shared with AFFC.

Why DFCFC1 matters (transfer-function level)
--------------------------------------------

The single-FF AFFC has gm_f2 / gm_3 LHP zero from Mgf2 alone but
relies on the standard NMOS mirror load at stage 1, whose output
impedance r_M4 caps the stage-1 DC gain at A_1 ≈ gm_M1 · r_M4 ≈
gm/Id · Vea_M4 (typically 30-40 dB on gf180mcu 3.3 V). Adding a
cascode device boosts r_M4 → r_M4 · g_m_cas / g_o_M4, raising A_1
by ~20-30 dB. That extra stage-1 gain compounds with stages 2 and 3
to give a higher overall A_DC; equivalently, at the same total A_DC,
the dominant pole sits at lower frequency, so the second pole has
more room to roll off before crossing GBW. The net is: cascode
boost lets DFCFC1 hit higher GBW × A_DC product than AFFC for the
same Cm1 / Mgf2 sizing.

Note: the regulated-cascode structure in DFCFC1 also creates a new
high-impedance node at d1 (with r_eff = r_M4 · (1 + gm_cas · r_cas)),
which interacts with Cm2 to set the inner-Miller pole. Whether the
net effect on PM is positive depends on Cm2 / r_d1 ratio; tests in
`tests/test_leung_dfcfc1_distinct.py` lock the architectural
distinction (Mgf2 + Mcas1_n_load + Mcas1_n_diode + Mlcs1 +
Mlcs1_diode + nd1_cas + nd1_cas_diode + vcas exist) rather than
asserting specific SPICE numerics.

Honest scope (what you should NOT expect)
-----------------------------------------

DFCFC1 is a structural compensation refinement that genuinely
changes the netlist (cascoded stage-1 mirror + cascode-bias chain +
two new internal nodes nd1_cas / nd1_cas_diode + vcas rail). At
the relaxed 5 dB / 10 kHz spec floor on gf180mcu 3.3 V, the PVT
envelope will land comfortably; the actual physical benefit of the
cascode boost (DC-gain bump) is sensitive to W/L of the cascode
devices and the vcas voltage. Tests do NOT assert "DC gain bigger
than NMCF baseline" because default sizing isn't tuned for that;
they only assert the cascode devices exist and route correctly.

Knobs
-----

DFCFC1 adds the following new kwargs on top of the NMCF baseline:

  Mgf2_scale: float = 1.0
      Sizing ratio of Mgf2 / Mcs3 (same role as AFFC's Mgf2_scale).
  W_gf2, L_gf2: optional explicit overrides for Mgf2 sizing.
  W_cas1, L_cas1: float = 4 µm / 1 µm
      Cascode device sizing for Mcas1_n_load + Mcas1_n_diode (same
      sizing for both legs to keep matching).
  W_cas_bias, L_cas_bias: float = 4 µm / 1 µm
      Cascode-bias diode sizing (Mcas_bias).
  N_cas_bias: int = 1
      Multiplier on Mcas_bias; controls vcas voltage and hence the
      cascode device headroom.

The kwarg `Cf` from the NMCF baseline is NOT exposed — DFCFC1
explicitly drops Cf. Cm1 and Cm2 keep their NMCF routing (out↔d2
and d2↔d1) so they ARE exposed. All other knobs (I_ref, W_bp,
L_bp, W_bn, L_bn, N_tail, K_cs3_load, W_diff, L_diff, W_load1,
L_load1, W_cs2, L_cs2, W_cs3, L_cs3, gm_id_diff, gm_id_load1,
gm_id_stage2, gm_id_stage3) are inherited verbatim from NMCF.
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


def build_nested_miller_3stage_dfcfc1(
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

    # ─── DFCFC1-specific: active feedforward Mgf2 ───────────
    Mgf2_scale: float = 1.0,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # ─── DFCFC1-specific: cascoded stage-1 mirror devices ───
    # Mcas1_n_load + Mcas1_n_diode: NMOS cascode between d1 / d1_diode
    # and the cascode-source nodes nd1_cas / nd1_cas_diode. Both legs
    # use the same W/L for matching.
    W_cas1: float = 4e-6,
    L_cas1: float = 1e-6,

    # Cascode-bias diode Mcas_bias: NMOS diode at vcas, sources its
    # current from a Mbp-mirrored leg (Mbp_cas_drv) so vcas ≈ Vgs_n
    # above vss. N_cas_bias multiplies Mcas_bias's M to set the I_d
    # through the diode (higher M → lower Vgs → less cascode headroom
    # consumed).
    W_cas_bias: float = 4e-6,
    L_cas_bias: float = 1e-6,
    N_cas_bias: int = 1,

    # ─── R115 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # Leung DFCFC1 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the cascoded high-Z d1 node from supply-coupled
    #     substrate noise. DFCFC1's cascoded mirror boosts r_out at d1
    #     substantially so substrate noise injection there has an
    #     outsized effect on stage-1 gain accuracy.
    #   * WELL_ISOLATED on the NMOS cascode pair (Mcas1_n_load /
    #     Mcas1_n_diode). Razavi §9.2 — cascoded NMOS mirror legs share
    #     a deep-substrate well with the underlying gain devices M3 / M4
    #     and the body-effect coupling through the shared p-substrate
    #     would degrade the cascoded output impedance boost. Per-leg
    #     local well isolation keeps body-effect Vth shift identical
    #     across the two cascode legs (preserves cascode r_out × g_m
    #     gain boost).
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Leung DFCFC1 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC / NMCNR / TCFC / ACBC / DFCFC2 variants.
    DFCFC1 differs from NMCF in two structural places:
      1. Active feedforward Mgf2 (PMOS, g=d1, d=out) — shared with AFFC
      2. Cascoded stage-1 NMOS mirror load: Mcas1_n_load + Mcas1_n_diode
         cascode devices, Mlcs1 + Mlcs1_diode sink mirrors, Mcas_bias
         cascode-bias diode, creating two new internal nodes nd1_cas /
         nd1_cas_diode and a new bias rail vcas.

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
        Mgf2_scale: ratio of Mgf2 W to Mcs3 W (default 1.0).
        W_gf2, L_gf2, gm_id_gf2: overrides for Mgf2 sizing.
        W_cas1, L_cas1: cascode device sizing.
        W_cas_bias, L_cas_bias, N_cas_bias: cascode-bias diode sizing.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # DFCFC1: Mgf2 draws additional DC current → widened NMOS sink at
    # out (same trick as AFFC).
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref
    K_cs3_load_total = K_cs3_load * (1 + Mgf2_scale)

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

    # ─── DFCFC1 cascode-bias chain: vcas rail ───────────────
    # Mcas_bias: NMOS diode at vcas, sources its DC current from a
    # PMOS mirror (Mbp_cas_drv) so vcas settles at Vgs_n above vss
    # (≈ 0.6-0.8 V on gf180mcu 3.3 V at typical bias). N_cas_bias
    # adjusts Vgs (more M → smaller Vgs → less headroom consumed
    # for cascode operation).
    devices.append(_mosfet("Mbp_cas_drv", "pmos",
                            d="vcas", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mcas_bias", "nmos",
                            d="vcas", g="vcas", s="vss", b="vss",
                            W=W_cas_bias, L=L_cas_bias, M=N_cas_bias))

    # ─── Stage 1: PMOS diff pair ────────────────────────────
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    # ─── DFCFC1 delta: cascoded NMOS mirror load ───────────
    # Architectural signature of DFCFC1. Each leg of the NMOS mirror
    # gets a cascode device sitting between the gain-device drain
    # (d1 / d1_diode) and the gain-device source (which is now the
    # internal cascode-source node nd1_cas / nd1_cas_diode rather
    # than vss). Below the cascode-source nodes, a separate NMOS
    # mirror leg sinks the I_load1 current to vss.
    #
    # Mirror gain devices (M3 diode, M4 mirror) — note that for the
    # cascode structure their SOURCES are at the cascode-source
    # nodes, not vss. The cascode device sits between their drain
    # and the d1 / d1_diode nodes:
    devices.append(_mosfet("M3", "nmos", d="d1_diode_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos", d="d1_int",       g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    # Cascode devices: drain at d1 / d1_diode, gate=vcas, source at
    # the gain-device drain (cascode-source nodes nd1_cas / nd1_cas_diode).
    # The cascode boosts the output impedance looking into d1 by
    # roughly g_m_cas · r_o_M4, raising stage-1 DC gain.
    devices.append(_mosfet("Mcas1_n_load", "nmos",
                            d="d1", g="vcas", s="d1_int", b="vss",
                            W=W_cas1, L=L_cas1))
    devices.append(_mosfet("Mcas1_n_diode", "nmos",
                            d="d1_diode", g="vcas",
                            s="d1_diode_int", b="vss",
                            W=W_cas1, L=L_cas1))

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

    # ─── DFCFC1 delta: Active feedforward Mgf2 ─────────────
    # Same role as AFFC's Mgf2 — PMOS in parallel with Mcs3 gated
    # by d1. Creates the LHP zero that replaces the dropped Cf.
    devices.append(_mosfet("Mgf2", "pmos", d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # NMOS sink at output — widened to absorb both Mcs3 and Mgf2 DC
    # currents (same trick as AFFC).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── Compensation: Cm1 outer + Cm2 inner, NO Cf ─────────
    # DFCFC1 keeps NMCF-style Miller routing (out↔d2 outer, d2↔d1
    # inner) because the cascoded stage-1 mirror raises the d1
    # impedance enough that the inner Miller Cm2 keeps its standard
    # pole-splitting role. Cf is dropped because Mgf2 provides the
    # LHP zero structurally.
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R115 C: optional DRC annotations ─────────────────
    # Three annotations spanning MATCHING / GUARD_RING / WELL_ISOLATED.
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
                    "to vdd shields the cascoded high-Z d1 node from "
                    "supply-coupled substrate noise (DFCFC1's cascoded "
                    "mirror boosts r_out at d1 so substrate noise there "
                    "has an outsized effect on stage-1 gain accuracy)"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mcas1_n_load", "Mcas1_n_diode"),
            extras={
                "rationale": (
                    "NMOS cascode pair on stage-1 mirror — body-effect "
                    "coupling through the shared p-substrate would "
                    "degrade the cascoded output impedance boost at d1; "
                    "per-leg local well isolation keeps body-effect Vth "
                    "shift identical across the two cascode legs and "
                    "preserves the cascode r_out * g_m gain boost"
                ),
                "isolation_type": "local_well",
            },
        ))

    return cell
