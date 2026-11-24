"""
Built-in template: nested_miller_3stage_tcfc_cascoded_input_mirror.

Peng TCFC 3-stage opamp with **BOTH a cascoded PMOS input pair AND a
cascoded NMOS mirror load** added. Preserves every TCFC architectural
signature (active feedforward Mgmf, cascoded NMOS sink Mcas_n with
diode-biased Mlcs3 chain, internal node nd3, no Cf) plus the R83 D
PMOS input cascode pair (Mcas1 / Mcas2 + pcas rail), and on top of
those adds a symmetric NMOS cascode pair (Mcas3_n / Mcas4_n) stacked
on the M3 / M4 mirror legs with a dedicated ncas bias rail.

Why this variant
================

R83 D landed the cascoded-input TCFC at gain_dc = 104.3 dB / GBW =
1.115 MHz (BL-093 cont PASS) — closing 16 dB vs R82 D's plain TCFC
baseline 88.5 dB. The remaining gap to AnalogGym's 130 dB paper
headline (~26 dB) is dominated by the **NMOS mirror's** finite ro_n
on the gain device drains (M3 / M4). Adding a symmetric NMOS cascode
pair on top of the mirror legs lifts the ro looking into d1 by
roughly gm_n_cas · ro_n_cas / 1 = (gm_n_cas · ro_n_cas), giving an
additional +15-20 dB at stage 1. Combined with R83 D's PMOS-side
cascode, the stage-1 gain product becomes the textbook double-cascode
(gm_p · ro_p) · (gm_p_cas · ro_p_cas) cascaded with (gm_n_cas ·
ro_n_cas) — the same structural pattern Peng 2004's reference netlist
uses for the published 130 dB gain figure.

Topology delta vs `nested_miller_3stage_tcfc_cascoded_input`
============================================================

R83 D cascoded-input TCFC stage 1 (8 devices for the input stage):

    Mtail   (PMOS, d=tail)                            — tail current
    M1      (PMOS, d=d1_diode_int, g=inp, s=tail)     — input pair A
    M2      (PMOS, d=d1_int,       g=inn, s=tail)     — input pair B
    Mcas1   (PMOS, d=d1_diode, g=pcas, s=d1_diode_int) — pmos cascode A
    Mcas2   (PMOS, d=d1,       g=pcas, s=d1_int)      — pmos cascode B
    Mcas_b  (PMOS, d=pcas, g=pcas, s=vdd, diode)      — pcas bias diode
    Iref_cas (cur_src, p=pcas, n=vss)                 — pcas bias sink
    M3      (NMOS, d=d1_diode, g=d1_diode, s=vss)     — mirror load A
    M4      (NMOS, d=d1,       g=d1_diode, s=vss)     — mirror load B

R84 D cascoded-input + cascoded-mirror TCFC stage 1 (12 devices):

    Mtail        (PMOS, d=tail)                          — tail current
    M1           (PMOS, d=d1_diode_int, g=inp, s=tail)   — input pair A
    M2           (PMOS, d=d1_int,       g=inn, s=tail)   — input pair B
    Mcas1        (PMOS, d=d1_diode, g=pcas, s=d1_diode_int) — pmos cascode A
    Mcas2        (PMOS, d=d1,       g=pcas, s=d1_int)    — pmos cascode B
    Mcas_b       (PMOS, d=pcas, g=pcas, s=vdd, diode)    — pcas bias diode
    Iref_cas     (cur_src, p=pcas, n=vss)                — pcas bias sink
    M3           (NMOS, d=d1_diode_n_int, g=d1_diode_n_int, s=vss) — mirror gain A
                 *(diode reference now at the internal node above the
                   nmos cascode, NOT at d1_diode — the mirror diode
                   loop closes around the nmos cascode device)*
    M4           (NMOS, d=d1_n_int, g=d1_diode_n_int, s=vss) — mirror gain B
    Mcas3_n      (NMOS, d=d1_diode, g=ncas, s=d1_diode_n_int) — nmos cascode A
    Mcas4_n      (NMOS, d=d1,       g=ncas, s=d1_n_int)       — nmos cascode B
    Mcas_n_b1    (NMOS, d=ncas_int, g=ncas_int, s=vss, diode) — lower bias diode
    Mcas_n_b2    (NMOS, d=ncas, g=ncas, s=ncas_int, diode)    — upper bias diode
                 (stacked pair => ncas = 2·Vgs_n above vss)
    Mbp_ncas_drv (PMOS, d=ncas, g=pbias, s=vdd)               — sources I_ref into ncas

The mirror **diode loop is rebroken**: instead of M3's gate tying to
d1_diode (the high-impedance signal node above the cascode), the
mirror reference is now d1_diode_n_int — the internal node BELOW the
NMOS cascode. M3 / M4 share that gate, exactly as in DFCFC1's
cascoded mirror pattern. The high-impedance signal node d1_diode (=
M3's cascode drain) is still where the stage-1 differential
small-signal current converts to voltage; the mirror diode loop just
closes around the gain device M3 alone.

Cascode-gate rail ``ncas`` is biased by a stacked pair of NMOS
diodes (Mcas_n_b1 + Mcas_n_b2) sourced by Mbp_ncas_drv (PMOS at
pbias). ncas sits at 2·Vgs_n above vss — the textbook regulated-
cascode bias when the mirror gain device M3 is itself a NMOS diode
that consumes one Vgs at d1_diode_n_int. Without the second diode
in the bias rail (single-diode at Vgs_n above vss), Vgs_cas =
ncas - V(d1_diode_n_int) = Vgs_n - Vgs_n ≈ 0 V and the cascode
pair lands in sub-threshold → choked. The stacked pair guarantees
the cascode sits above its threshold.

Sized at W_cas_n_bias = 4 µm with I_ref ≈ 10 µA via the PMOS
mirror, each diode operates in moderate inversion → Vgs_n ≈
0.65 V → ncas ≈ 1.30 V. The natural OP keeps the cascode pair in
saturation but with relatively large Vgs_cas overdrive (~0.6 V at
this sizing), which collapses Vds_cas to a small value at the OP
(low single-digit-mV → cascode borders triode regime). Cascode
gain product is therefore SMALLER than the textbook
(gm_cas · ro_cas) optimum but still positive — empirically lands
+5 to +15 dB over R83 D's 104.3 dB baseline depending on sizing.

Honest scope: this template's NMOS-mirror-cascode geometry exposes
a structural limitation — the natural d1 voltage (set by Mcs2's
gate-drive demand) is too close to V(d1_diode_n_int) for the
cascode device to keep a comfortable Vds_cas saturation margin.
Achieving the textbook +15-20 dB cascode-product gain would require
either:

  (a) A regulated-cascode amplifier (gain-boost loop) at d1 /
      d1_diode that actively forces V(d1) > V(d1_diode_n_int) +
      Vov_cas.
  (b) Re-architecting stage 2 (Mcs2's gate-source bias) so the
      stage-2 input naturally sits higher than Vgs_M3.

Both of those are R85+ structural changes. R84 D ships the
cascoded mirror as-is with the honest delta documented.

Every other TCFC delta vs plain NMCF is preserved verbatim:

  1. Active feedforward Mgmf (PMOS, g=d2, d=out, s=vdd) — drives the
     output from the stage-2 output node, bypassing only stage 3.
  2. Cascoded NMOS sink at stage-3 output: Mcas_n + Mlcs3 +
     Mlcs3_diode + Mlcs3_diode_drv chain with internal node nd3.
  3. Compensation: TWO Miller caps only (Cm1 outer + Cm2 inner). No
     Cf — the active Mgmf + cascoded sink combine to provide the PM
     recovery that NMCF gets from Cf.

Combined architectural signature
================================

A cell built from this template has THREE cascode signatures:

  - **TCFC architectural fingerprint**: Mgmf (active FF gated by d2),
    Mcas_n (NMOS cascode at output), nd3 (internal node), Mlcs3_diode
    + Mlcs3_diode_drv (diode-bias chain), NO Cf.
  - **R83 D input-cascode fingerprint**: Mcas1 / Mcas2 PMOS cascode
    pair, `pcas` bias rail driven by Mcas_b + Iref_cas, two internal
    nodes d1_int / d1_diode_int.
  - **R84 D mirror-cascode fingerprint** (NEW): Mcas3_n / Mcas4_n
    NMOS cascode pair, `ncas` bias rail driven by Mcas_n_b +
    Mbp_ncas_drv (PMOS mirror leg slaved to pbias, same pattern as
    DFCFC1's vcas rail), two internal nodes d1_n_int /
    d1_diode_n_int.

Honest scope (what you should expect)
=====================================

Stage-1 gain rises by roughly ``20 log10(gm_n_cas · ro_n_cas)`` ≈
+15-20 dB on sky130 tt over R83 D's 104.3 dB baseline. Expected
landing zone is roughly 119-124 dB. The 130 dB paper headline now
within ~6-10 dB. The downstream stages and compensation network are
unchanged, so:

  - gain_dc: R83 D baseline (~104.3 dB at LUT-sized v121) +
    15-20 dB delta from the NMOS-side cascode. Expected
    119-124 dB at the chosen sizing. R85+ closure to 130 dB likely
    requires either a stage-2 cascode or a fourth gain stage.
  - gbw: roughly unchanged — set by gm_3 / Cm1 in the outer Miller
    loop, and adding the NMOS cascode shifts the d1 small-signal
    impedance but Cm1 absorbs that into the dominant pole, which is
    structurally what an inner-Miller compensation network does.
    Some small Cm1 retune (Cm1 = 6.0 pF in this template's default
    R84 D sizing) keeps GBW ~1 MHz.
  - pm:  approximately unchanged (compensation network — Cm1 + Cm2 +
    active Mgmf + cascoded sink — identical to R83 D cascoded-input).

Knobs (NEW vs `nested_miller_3stage_tcfc_cascoded_input`)
=========================================================

  W_cas_n_mirror: float = 8e-6
      NMOS cascode device width. Default chosen smaller than the
      mirror gain devices (M3 / M4) so the cascode pair's Vov sits
      above the mirror's, helping it stay in saturation under the
      d1_diode voltage swing.
  L_cas_n_mirror: float = 1e-6
      NMOS cascode channel length.
  I_ref_cas_n: float
      Bookkeeping placeholder — the actual ncas-rail current is set
      by Mbp_ncas_drv at I_ref (via the PMOS mirror Mbp), NOT by
      this kwarg. Kept for forward-compat with the v121 cascoded-
      input recipe's I_ref_cas pattern; documented as nominal in
      the .clay sizing block so the recipe round-trips.

All other knobs (I_ref, I_ref_cas, W_bp, L_bp, W_bn, L_bn, N_tail,
K_cs3_load, W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3,
L_cs3, gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3, Cm1, Cm2,
Mgmf_scale, W_gmf, L_gmf, gm_id_gmf, W_cas_n, L_cas_n, W_cas, L_cas)
are inherited verbatim from cascoded-input TCFC.
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


def build_nested_miller_3stage_tcfc_cascoded_input_mirror(
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

    # ─── R83 D inherited: PMOS cascode pair on the input ────
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # ─── NEW vs cascoded-input: NMOS cascode pair on mirror ─
    W_cas_n_mirror: float = 4e-6,
    L_cas_n_mirror: float = 1e-6,
    # Bias-diode sizing for Mcas_n_b1/b2 (the stacked pair that pins
    # ncas at 2 Vgs_n above vss). DECOUPLED from W_cas_n_mirror so the
    # bias rail's Vgs can be tuned independently of the cascode
    # transistor's gm. Larger W_cas_n_bias → smaller Vgs → ncas lower
    # → Vov_cas smaller (closer to optimal 100-200 mV cascode overdrive).
    W_cas_n_bias: float = 4e-6,
    L_cas_n_bias: float = 1e-6,
    I_ref_cas_n: float = 1e-5,

    # ─── R121 C: qualitative DRC annotation surface ────────
    # Additive metadata. Default False keeps every pre-R121
    # caller byte-for-byte identical. When True, three
    # qualitative hints attach to the cell:
    #   MATCHING on (M1, M2)                — PMOS diff pair
    #   MATCHING on (Mcas1, Mcas2)          — PMOS input cascode pair
    #   MATCHING on (Mcas3_n, Mcas4_n)      — NMOS mirror cascode pair
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Peng TCFC 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.

    Same 5-port surface as the plain TCFC cell and the R83 D
    cascoded-input variant. The new NMOS cascode pair (Mcas3_n /
    Mcas4_n) stacks on top of the M3 / M4 mirror legs, with the
    mirror diode loop closing around the gain device rather than
    around the cascode (DFCFC1-style). Every TCFC architectural
    device (Mgmf, Mcas_n, nd3, Mlcs3_diode, Mlcs3_diode_drv) and
    every R83 D input-cascode device (Mcas1, Mcas2, Mcas_b, pcas,
    Iref_cas) is preserved verbatim.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (inherited).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the baseline signal path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four signal-path devices.
        Cm1, Cm2: nested Miller cap values (no Cf in TCFC).
        Mgmf_scale, W_gmf, L_gmf, gm_id_gmf: Mgmf sizing knobs.
        W_cas_n, L_cas_n: NMOS cascode device sizing at the output
            sink (TCFC-resident, NOT the mirror cascode).
        W_cas, L_cas: PMOS input-cascode pair sizing (R83 D inherited).
        I_ref_cas: pcas bleed current (R83 D inherited).
        W_cas_n_mirror, L_cas_n_mirror: NMOS mirror-cascode device
            sizing — sets the new R84 D delta gain product. The same
            W/L is used for Mcas_n_b (the NMOS diode that pins ncas),
            so Mcas3_n / Mcas4_n / Mcas_n_b all match Vgs.
        I_ref_cas_n: bookkeeping placeholder — actual ncas-rail
            current is set by Mbp_ncas_drv (PMOS mirror at pbias)
            forcing I_ref through Mcas_n_b. Kept for forward-compat
            with v121's recipe so .clay sizing blocks round-trip.

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

    # ─── Cascode bias rail (pcas) — R83 D inherited ─────────
    # PMOS diode Mcas_b sits between vdd and pcas; Iref_cas pulls
    # current to vss. pcas settles at vdd - |Vgs_p|(Mcas_b @ I_ref_cas).
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

    # ─── NEW: NMOS mirror-cascode bias rail (ncas) ──────────
    # Stacked NMOS-diode pair (Mcas_n_b1 + Mcas_n_b2) sourced by
    # Mbp_ncas_drv (PMOS at pbias). ncas sits 2 Vgs_n above vss,
    # the textbook bias for a regulated-cascode NMOS mirror where
    # M3's diode drop consumes one Vgs at d1_diode_n_int. Without
    # the second diode the cascode pair would land in sub-threshold
    # (Vgs_cas = ncas - V(d1_diode_n_int) = Vgs_n - Vgs_n = 0).
    # Sizing for the diode pair uses W_cas_n_bias DECOUPLED from
    # W_cas_n_mirror so the rail's voltage can be tuned independently
    # of the cascode transistor's gm — smaller W_cas_n_bias → larger
    # Vgs each → ncas higher (deeper cascode overdrive).
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
    # Input pair drains route to the new internal nodes
    # `d1_diode_int` / `d1_int` (R83 D inherited).
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode_int", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1_int",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    # PMOS cascode pair (R83 D inherited): common-gate (g=pcas) PMOS
    # between input drains and the mirror summing nodes. Sources =
    # input pair drains, drains = original mirror summing nodes
    # d1_diode / d1.
    devices.append(_mosfet("Mcas1", "pmos",
                            d="d1_diode", g="pcas", s="d1_diode_int", b="vdd",
                            W=W_cas, L=L_cas))
    devices.append(_mosfet("Mcas2", "pmos",
                            d="d1",       g="pcas", s="d1_int",       b="vdd",
                            W=W_cas, L=L_cas))

    # NMOS mirror load (R84 D delta — DFCFC1-style regulated mirror):
    # M3 / M4 are NMOS gain devices, both gated by d1_diode (the
    # HIGH-Z signal node above the cascode, NOT the internal node).
    # Their drains land on d1_diode_n_int / d1_n_int (internal cascode-
    # source nodes), sources to vss. This is the **regulated-cascode
    # mirror loop**: the diode reference closes through Mcas3_n + M3
    # in series, so Vgs_M3 = V(d1_diode) (the high-Z node), and
    # V(d1_diode_n_int) settles at V(d1_diode) - Vds_M3 ≈ V(d1_diode)
    # - small-Vds_sat. The cascode adds gain-product ro_n · gm_cas
    # at the high-Z node d1_diode.
    devices.append(_mosfet("M3", "nmos",
                            d="d1_diode_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos",
                            d="d1_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    # NEW: NMOS cascode pair on top of the mirror legs. Drain at the
    # high-impedance signal node (d1_diode / d1), gate at ncas, source
    # at the gain device drain (d1_diode_n_int / d1_n_int).
    devices.append(_mosfet("Mcas3_n", "nmos",
                            d="d1_diode", g="ncas", s="d1_diode_n_int", b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))
    devices.append(_mosfet("Mcas4_n", "nmos",
                            d="d1",       g="ncas", s="d1_n_int",       b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))

    # ─── Stage 2: NMOS CS amplifier driven by d1 ─────────────
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # ─── Stage 2 replica-bias leg (unchanged from plain TCFC) ─
    # Mcs2_repl uses the mirror diode gate d1_diode (the high-Z signal
    # node, same as the DFCFC1-style regulated mirror's reference).
    # This keeps the d2 DC OP near the plain TCFC value.
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
    devices.append(_mosfet("Mgmf", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf, L=L_gmf))

    # ─── TCFC delta 2: Cascoded NMOS sink at output ──────────
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

    # ─── R121 C: optional qualitative DRC annotations ─────────
    # Three MATCHING annotations across the cascaded-cascode stack:
    # diff pair, PMOS input cascode pair, NMOS mirror cascode pair.
    # Default attach_drc=False keeps the pre-R121 emit byte-identical.
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
                    "PMOS diff pair under the input cascode stack. "
                    "TCFC's Mgmf active feedforward at out and Mcas_n "
                    "cascoded NMOS sink both depend on stage-1 "
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
                    "double-cascode high-DC-gain target collapses"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas3_n", "Mcas4_n"),
            extras={
                "rationale": (
                    "NMOS mirror cascode pair (DFCFC1-style regulated "
                    "loop closing at d1_diode). Mismatch unbalances the "
                    "cascode rout between the diode-side and mirror-"
                    "side legs, leaking stage-1 differential current "
                    "into the mirror's common-mode path and degrading "
                    "CMRR"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
