"""
Built-in template: nested_miller_3stage_dfcfc2_cascoded_input_mirror.

Leung_DFCFC2 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** added on top of the dual-feedforward
DFCFC2 backbone. R85 C cross-variant generalisation of R84 D's Peng
TCFC recipe (telescopic input cascode + cascoded NMOS mirror load
with DFCFC1-style mirror diode loop at the high-Z node ``d1``).

Why this variant
================

R84 D landed the Peng TCFC + double-sided cascode at gain_dc =
129.106 dB / GBW = 1.117 MHz (BL-093 cont PASS; only ~0.9 dB short
of paper 130 dB headline). R85 C asks: does the **same recipe**
generalise to a STRUCTURALLY DIFFERENT NMCF-class variant? DFCFC2
is the natural cross-variant target because:

  1. DFCFC2 has **dual active feedforwards** (Mgf2 + Mgf_damp) and
     a damping node nd_damp — structurally distinct from TCFC's
     single Mgmf + cascoded output sink Mcas_n + internal node nd3.
  2. DFCFC2's compensation network is also distinct: Cm1 (out↔d1
     skip-over) + Cm2_damp (nd_damp↔d1 damping cap), vs TCFC's
     Cm1 (out↔d2 outer) + Cm2 (d2↔d1 inner).
  3. DFCFC2 paper GBW target is **4 MHz** (vs TCFC's 1 MHz; 4× higher).
     The cascode recipe needs to preserve a different GBW window.
  4. Mgf2 / Mgf_damp gate node is ``d1`` — same as TCFC's Mgmf, so
     the cascode recipe's "d1 is the high-Z signal node above the
     cascode" structural assumption holds.

If the R84 D recipe lands DFCFC2 with gain_dc ≥ 120 dB (paper-10 dB
acceptance) AND GBW ∈ [3.2 MHz, 4.8 MHz], then M-SIZING-PAPER-GRADE
goes 3/17 → 4/17 and R85+ can safely roll the recipe out to the
remaining 13 NMCF-class variants.

Topology delta vs `nested_miller_3stage_dfcfc2`
================================================

Baseline DFCFC2 stage 1 (5 devices):

    Mtail   (PMOS, d=tail)                            — tail current
    M1      (PMOS, d=d1_diode, g=inp, s=tail)         — input pair A
    M2      (PMOS, d=d1,       g=inn, s=tail)         — input pair B
    M3      (NMOS, d=d1_diode, g=d1_diode, s=vss)     — mirror diode A
    M4      (NMOS, d=d1,       g=d1_diode, s=vss)     — mirror gain B

R85 C cascoded-input + cascoded-mirror DFCFC2 stage 1 (12 devices):

    Mtail        (PMOS, d=tail)                          — tail current
    M1           (PMOS, d=d1_diode_int, g=inp, s=tail)   — input pair A
    M2           (PMOS, d=d1_int,       g=inn, s=tail)   — input pair B
    Mcas1        (PMOS, d=d1_diode, g=pcas, s=d1_diode_int) — pmos cascode A
    Mcas2        (PMOS, d=d1,       g=pcas, s=d1_int)    — pmos cascode B
    Mcas_b       (PMOS, d=pcas, g=pcas, s=vdd, diode)    — pcas bias diode
    Iref_cas     (cur_src, p=pcas, n=vss)                — pcas bias sink
    M3           (NMOS, d=d1_diode_n_int, g=d1_diode, s=vss) — mirror gain A
                 *(diode reference at d1_diode, the HIGH-Z node above the
                   nmos cascode — DFCFC1-style regulated mirror)*
    M4           (NMOS, d=d1_n_int, g=d1_diode, s=vss)   — mirror gain B
    Mcas3_n      (NMOS, d=d1_diode, g=ncas, s=d1_diode_n_int) — nmos cascode A
    Mcas4_n      (NMOS, d=d1,       g=ncas, s=d1_n_int)       — nmos cascode B
    Mcas_n_b1    (NMOS, d=ncas_int, g=ncas_int, s=vss, diode) — lower bias diode
    Mcas_n_b2    (NMOS, d=ncas, g=ncas, s=ncas_int, diode)    — upper bias diode
                 (stacked pair => ncas = 2·Vgs_n above vss)
    Mbp_ncas_drv (PMOS, d=ncas, g=pbias, s=vdd)               — sources I_ref into ncas

The mirror **diode loop closes DFCFC1-style** at the high-Z signal
node ``d1_diode`` (NOT at the internal cascode-source node
``d1_diode_n_int``). M3 / M4 share gate=d1_diode, exactly as in
DFCFC1's cascoded mirror pattern. This is the recipe that delivered
R84 D's +24 dB gain product on the TCFC backbone.

DFCFC2 architectural devices preserved verbatim
================================================

Every DFCFC2 dual-feedforward signature is preserved:

  1. **Mgf2** (PMOS, g=d1, d=out, s=vdd) — primary active feedforward,
     drains to the output node. Same role as AFFC's Mgf2.
  2. **Mgf_damp** (PMOS, g=d1, d=nd_damp, s=vdd) — DFCFC2-unique
     second active feedforward, drains to the damping node nd_damp.
  3. **Mlcs_damp** (NMOS, d=nd_damp, g=nbias) — DFCFC2-unique NMOS
     sink providing the DC current path for Mgf_damp.
  4. **Compensation**: Cm1 (out↔d1 skip-over) + Cm2_damp (nd_damp↔d1
     damping cap). No Cf.

Mgf2 / Mgf_damp gate node ``d1`` is preserved as the high-Z signal
node above the NMOS cascode — the cascode recipe lifts the
small-signal impedance at d1 without changing the DC operating
point of d1 (Vgs_M3 = V(d1_diode), Mcas3_n's drain still settles
at the same V(d1) as in plain DFCFC2 since the cascode only adds
ro to the impedance, not DC voltage drop).

Combined architectural signature
================================

A cell built from this template has THREE cascode signatures plus
TWO DFCFC2-unique active devices:

  - **DFCFC2 dual-feedforward fingerprint**: Mgf2 (PMOS, g=d1, d=out),
    Mgf_damp (PMOS, g=d1, d=nd_damp), Mlcs_damp (NMOS sink), Cm1 +
    Cm2_damp routing, NO Cf.
  - **R83 D input-cascode fingerprint**: Mcas1 / Mcas2 PMOS cascode
    pair, `pcas` bias rail driven by Mcas_b + Iref_cas, two internal
    nodes d1_int / d1_diode_int.
  - **R84 D mirror-cascode fingerprint**: Mcas3_n / Mcas4_n NMOS
    cascode pair, `ncas` bias rail driven by Mcas_n_b1 + Mcas_n_b2 +
    Mbp_ncas_drv (PMOS mirror leg slaved to pbias), two internal
    nodes d1_n_int / d1_diode_n_int.

Honest scope (what you should expect)
=====================================

Stage-1 gain rises by roughly the same +60-66 dB delta that R84 D
delivered on TCFC (double-sided cascode product is structurally
identical at stage 1). Plain DFCFC2 baseline at v0.0.84 sizing is
gain_dc = 62.5 dB; R85 C target is gain_dc ≥ 120 dB (R85 C
acceptance = paper 130 dB - 10 dB).

DFCFC2 paper GBW is 4 MHz (vs TCFC's 1 MHz). Cm1 must be scaled
down accordingly — at fixed gm_3, GBW scales as 1/Cm1, so Cm1 ~
1/4 of TCFC's 5 pF = 1.25 pF puts GBW in the right ballpark. The
cascoded-mirror also raises the d1 impedance which may shift the
dominant pole; some Cm1 retune may be needed.

Knobs
-----

Inherits every kwarg from `build_nested_miller_3stage_dfcfc2` PLUS
the cascode-related kwargs from R84 D:

  W_cas, L_cas, I_ref_cas
      PMOS input cascode pair sizing (R83 D pattern).
  W_cas_n_mirror, L_cas_n_mirror
      NMOS mirror-cascode pair sizing (R84 D pattern).
  W_cas_n_bias, L_cas_n_bias
      Stacked bias-diode pair sizing for the ncas rail.
  I_ref_cas_n
      Bookkeeping placeholder — actual ncas-rail current set by
      Mbp_ncas_drv at I_ref via the pbias mirror.
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


def build_nested_miller_3stage_dfcfc2_cascoded_input_mirror(
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
    Cm1: float = 1.25e-12,
    Cm2_damp: float = 0.5e-12,

    # ─── DFCFC2-specific: active feedforward Mgf2 (shared with AFFC) ─
    Mgf2_scale: float = 1.0,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # ─── DFCFC2-specific: second feedforward Mgf_damp ───────
    Mgf_damp_scale: float = 0.5,
    W_gf_damp: float | None = None,
    L_gf_damp: float | None = None,
    gm_id_gf_damp: float | None = None,

    # ─── R83 D inherited: PMOS cascode pair on the input ────
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # ─── R84 D inherited: NMOS cascode pair on mirror ───────
    W_cas_n_mirror: float = 4e-6,
    L_cas_n_mirror: float = 1e-6,
    W_cas_n_bias: float = 4e-6,
    L_cas_n_bias: float = 1e-6,
    I_ref_cas_n: float = 1e-5,

    # ─── R113 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # DFCFC2 cascoded-input-mirror variant:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Same diff-pair
    #     offset-via-Vth-mismatch argument as the DFCFC1 sibling.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Same substrate-isolation argument as the DFCFC1 sibling.
    #   * WELL_ISOLATED on the NMOS mirror cascode pair (Mcas3_n /
    #     Mcas4_n). The NMOS cascode stack lifts d1 impedance via
    #     g_m_cas·r_o; body-effect coupling through the shared
    #     p-substrate with M3 / M4 would degrade that r_o boost.
    #     The DFCFC2 variant differs from DFCFC1 here: DFCFC1's
    #     critical cascode is the PMOS input cascode (Mcas1/Mcas2);
    #     DFCFC2's dominant gain limiter at d1 is the NMOS cascode
    #     side once the dual-feedforward Mgf2+Mgf_damp is at play.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Leung DFCFC2 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.

    Same 5-port surface as the plain DFCFC2 cell. The cascode pair
    (Mcas1/Mcas2) and the NMOS mirror cascode pair (Mcas3_n/Mcas4_n)
    stack on the stage-1 input pair and the M3/M4 mirror legs
    respectively. The DFCFC2 dual-feedforward signature (Mgf2 + Mgf_damp
    + Mlcs_damp + nd_damp + Cm1 + Cm2_damp routing) is preserved verbatim.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (inherited from NMCF baseline).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the baseline signal path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four signal-path devices.
        Cm1: outer Miller cap (out ↔ d1 skip-over). Default 1.25 pF
            (scaled down from R84 D's TCFC 5 pF by 4× since DFCFC2
            paper GBW is 4 MHz vs TCFC's 1 MHz).
        Cm2_damp: damping Miller cap (nd_damp ↔ d1). Default 0.5 pF.
        Mgf2_scale, W_gf2, L_gf2, gm_id_gf2: Mgf2 sizing knobs.
        Mgf_damp_scale, W_gf_damp, L_gf_damp, gm_id_gf_damp: Mgf_damp
            sizing knobs.
        W_cas, L_cas, I_ref_cas: PMOS input-cascode pair sizing
            (R83 D inherited).
        W_cas_n_mirror, L_cas_n_mirror: NMOS mirror-cascode pair
            sizing (R84 D inherited).
        W_cas_n_bias, L_cas_n_bias: stacked NMOS-diode pair sizing
            for the ncas bias rail.
        I_ref_cas_n: bookkeeping placeholder — actual ncas-rail
            current is set by Mbp_ncas_drv (PMOS mirror at pbias)
            forcing I_ref through Mcas_n_b1+b2.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as plain DFCFC2).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # DFCFC2: Mgf2 + Mgf_damp draw additional DC current.
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref
    I_gf_damp = Mgf_damp_scale * K_cs3_load * I_ref

    # NMOS sink at out (Mlcs3) widened by (1 + Mgf2_scale).
    K_cs3_load_total = K_cs3_load * (1 + Mgf2_scale)
    # NMOS sink at nd_damp (Mlcs_damp) sized to match Mgf_damp DC current.
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

    # ─── Cascode bias rail (pcas) — R83 D inherited ─────────
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

    # ─── NMOS mirror-cascode bias rail (ncas) — R84 D inherited ─
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

    # ─── Stage 1: cascoded PMOS diff pair with cascoded NMOS mirror ─
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode_int", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1_int",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    # PMOS cascode pair (R83 D inherited): common-gate (g=pcas) PMOS
    # between input drains and the mirror summing nodes.
    devices.append(_mosfet("Mcas1", "pmos",
                            d="d1_diode", g="pcas", s="d1_diode_int", b="vdd",
                            W=W_cas, L=L_cas))
    devices.append(_mosfet("Mcas2", "pmos",
                            d="d1",       g="pcas", s="d1_int",       b="vdd",
                            W=W_cas, L=L_cas))

    # NMOS mirror load (R84 D regulated-cascode mirror): M3 / M4 are
    # NMOS gain devices, both gated by d1_diode (the high-Z signal
    # node above the cascode, DFCFC1-style).
    devices.append(_mosfet("M3", "nmos",
                            d="d1_diode_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos",
                            d="d1_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    # NMOS cascode pair on top of the mirror legs.
    devices.append(_mosfet("Mcas3_n", "nmos",
                            d="d1_diode", g="ncas", s="d1_diode_n_int", b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))
    devices.append(_mosfet("Mcas4_n", "nmos",
                            d="d1",       g="ncas", s="d1_n_int",       b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))

    # ─── Stage 2: NMOS CS amplifier driven by d1 ─────────────
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # ─── Stage 2 replica-bias leg (unchanged from DFCFC2) ──────
    # Mcs2_repl gate is d1_diode (the high-Z signal node above the
    # NMOS cascode, same as the DFCFC1-style regulated mirror's
    # reference). Same pattern as R84 D TCFC.
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

    # ─── DFCFC2 delta 1: Active feedforward Mgf2 (g=d1, d=out) ──
    devices.append(_mosfet("Mgf2", "pmos", d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # NMOS sink at output — widened to absorb both Mcs3 and Mgf2 DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── DFCFC2 delta 2: Second active feedforward Mgf_damp ──
    # PMOS gated by d1 (same as Mgf2) but draining to a new internal
    # node nd_damp. This is the dual-FF distinguishing device.
    devices.append(_mosfet("Mgf_damp", "pmos",
                            d="nd_damp", g="d1", s="vdd", b="vdd",
                            W=W_gf_damp, L=L_gf_damp))

    # NMOS sink at nd_damp: provides the DC current path for Mgf_damp.
    devices.append(_mosfet("Mlcs_damp", "nmos",
                            d="nd_damp", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_damp_load))

    # ─── Compensation: Cm1 outer + Cm2_damp damping (no Cf) ──
    # DFCFC2's distinctive routing preserved verbatim:
    #   Cm1: out ↔ d1 (skip-over outer Miller)
    #   Cm2_damp: nd_damp ↔ d1 (damping Miller cap to nd_damp)
    devices.append(_cap("Cm1",      p="out",     n="d1", value=Cm1))
    devices.append(_cap("Cm2_damp", p="nd_damp", n="d1", value=Cm2_damp))

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
                    "to the NMOS mirror / NMOS cascode and the dual-FF "
                    "active devices Mgf2 + Mgf_damp; nwell guard ring "
                    "tied to vdd prevents substrate coupling into the "
                    "high-Z d1 node and the damping node nd_damp"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mcas3_n", "Mcas4_n"),
            extras={
                "rationale": (
                    "NMOS mirror cascode pair — lifts d1 impedance "
                    "via g_m_cas·r_o; body-effect coupling through "
                    "the shared p-substrate with M3/M4 would degrade "
                    "the r_o boost; isolate Mcas3_n/Mcas4_n in a "
                    "dedicated pwell separate from M3/M4"
                ),
                "well_type": "pwell",
            },
        ))

    return cell
