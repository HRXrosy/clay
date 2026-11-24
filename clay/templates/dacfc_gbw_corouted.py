"""
Built-in template: nested_miller_3stage_dacfc_gbw_corouted.

R91 C / BL-093 DACFC GBW co-redesign via Mgmf1 drain re-route.

R89 B finding
=============

R89 B v127 DACFC retrofit (dacfc_hybrid_d1_ff_v127.py) achieved:

  * gain_dc = 118.31 dB (PASS, > 115 dB acceptance)
  * GBW    = 5.53 MHz   (FAIL, +176% off paper 2.00 MHz; paper window
              [1.60, 2.40] MHz)

R87 C v125 DACFC (no d1-FF) gave gain = 84.90 dB FAIL but GBW =
2.12 MHz PASS (in window). R89 B v127 adding Mgf2_hybrid (g=d1, d=out)
lifted gain by +33.4 dB but pushed GBW from 2.12 MHz to 5.53 MHz
(2.6x up). The d1-FF creates a fast direct d1->out path that bypasses
stages 2-3, effectively raising the unity-gain frequency past the
Miller-cap dominant pole.

Architectural diagnosis (R91 C hypothesis)
==========================================

In v127 the input-coupled gmf1 pair (Mgmf1_n, Mgmf1_p) has its drains
on nga1_a and nga1_b. nga1_a/nga1_b are sunk by Mlcs_ga1_a/Mlcs_ga1_b
NMOS bias devices (g=nbias), which are dead-end current sinks --
nga1_a/nga1_b never feed back into the main signal path. So Mgmf1 in
v127 is a structural no-op for the small-signal gain.

The R91 attack: re-route Mgmf1_n.drain and Mgmf1_p.drain from
nga1_a/nga1_b to ``out``. This makes the input-coupled FF pair an
active feedforward DIRECTLY to out (parallel to Mgf2_hybrid and
Mgmf2). The hypothesis is that with Mgmf1 actually injecting current
at out, Cm1 (out <-> d2) sees a higher effective Miller multiplication
factor, pulling the dominant pole frequency down.

Honest scope: this is the prompt-specified attack vector. The opposite
outcome is also plausible -- ADDING more feedforward at ``out`` may
INCREASE GBW further, not decrease it (Miller cap dominant pole
depends on g_m_stage3 * R_out, while feedforward zeros are independent
parallel paths). The sweep across Mgmf1_co_scale {0.0, 0.5, 1.0, 1.5}
together with Cm1 sweep {2, 4, 6, 8, 12, 16 pF} measures both effects
empirically.

R91 C delta
===========

  * Mgmf1_n drain: ``nga1_a`` -> ``out``
  * Mgmf1_p drain: ``nga1_b`` -> ``out``
  * Mlcs3 widened by (1 + Mgmf2_scale + Mgf2_scale + 2 * Mgmf1_co_scale)
    to absorb the two extra PMOS source currents now flowing to out.
  * NEW kwarg ``Mgmf1_co_scale``: float in [0, 2], default 1.0. Scales
    Mgmf1_n/p W relative to W_cs3 (parallels the Mgmf2_scale /
    Mgf2_scale convention). Mgmf1_co_scale = 0 effectively disables
    the Mgmf1 pair (W set to a small floor), letting us isolate the
    co-routing's contribution.
  * Mlcs_ga1_a / Mlcs_ga1_b sinks are PRESERVED (they now have to sink
    only Mga1_a / Mga1_b currents alone, instead of the doubled
    Mga1_a + Mgmf1_n / Mga1_b + Mgmf1_p current pair in v127).

All other devices (R85 C cascode recipe + DACFC matched gma1 pair +
Mga2 buffer + Mgmf2 g=d2 + Mgf2_hybrid g=d1 + Cm1 + Cm2_dac) are
PRESERVED VERBATIM byte-equal v127.

Architectural devices preserved verbatim (byte-equal v127)
==========================================================

R89 B DACFC fingerprint that survives R91 C addition:

  1. Mga1_a / Mga1_b matched PMOS pair (g=bias_d2, d=nga1_a / nga1_b)
     -- DACFC matched gma1 active cascode -- PRESERVED.
  2. Mga2 single PMOS buffer (g=bias_d2, d=nga2) -- PRESERVED.
  3. Mgmf2 (PMOS, g=d2, d=out) -- original DACFC output FF, PRESERVED.
  4. Mgf2_hybrid (PMOS, g=d1, d=out) -- R89 B d1-FF, PRESERVED.
  5. Mlcs_ga1_a / Mlcs_ga1_b / Mlcs_ga2 nga DC bias sinks -- PRESERVED.
  6. Cm1 (out <-> d2) -- NMCF-style outer Miller -- PRESERVED.
  7. Cm2_dac (nga1_a <-> vss) -- inner cap -- PRESERVED.
  8. R85 C cascode recipe (Mcas1/2 PMOS input + Mcas3_n/4_n NMOS
     mirror with DFCFC1-style diode loop at d1_diode) -- PRESERVED.

R91 C v128 delta
================

  * Mgmf1_n.drain: nga1_a -> out (re-routed)
  * Mgmf1_p.drain: nga1_b -> out (re-routed)
  * Mlcs3 widening accounting for new currents at out
  * Mgmf1_co_scale knob (default 1.0 = byte-equal sizing to v127's W_gmf1_in)

The structural mutation is minimal: 2 device pin renames + 1 Mlcs3
multiplier change. No bias-chain disturbance.

Acceptance
==========

  * gain_dc >= 115 dB (preserve v127 PASS at 118.31 dB)
  * GBW in [1.60, 2.40] MHz (paper 2.00 MHz +/-20%)
  * PM proxy: simulation completes with finite gain + GBW + no NaN/inf
    (matches R88 C / R89 B / R90 C pattern -- 5-port diff cell has no
    ac_pm testbench template today; the GBW window + gain preservation
    + lack of oscillation are the available stability proxies)
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


def build_nested_miller_3stage_dacfc_gbw_corouted(
    cell_name: str,
    spec: Spec,
    *,
    # Bias / mirror ratios
    I_ref: float = 1e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs3_load: int = 4,

    # Per-device sizing: physical (W, L) form
    W_diff: float = 20e-6,  L_diff: float = 0.5e-6,
    W_load1: float = 4e-6,  L_load1: float = 1e-6,
    W_cs2: float = 20e-6,   L_cs2: float = 0.5e-6,
    W_cs3: float = 40e-6,   L_cs3: float = 0.5e-6,

    # Per-device sizing: intent (gm/Id) form
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,
    gm_id_stage2: float | None = None,
    gm_id_stage3: float | None = None,

    # Compensation cap (NMCF-style outer)
    Cm1: float = 2.0e-12,    # outer Miller (out <-> d2)

    # DACFC-specific: matched gma1 PMOS pair (cascode)
    W_gma1: float = 8e-6,
    L_gma1: float = 0.5e-6,

    # DACFC-specific: input-coupled gmf1_PMOS pair (R91 C re-routed)
    W_gmf1_in: float = 8e-6,
    L_gmf1_in: float = 1e-6,

    # DACFC-specific: single gma2 PMOS buffer
    W_gma2: float = 4e-6,
    L_gma2: float = 1e-6,

    # DACFC-specific: original output-stage FF Mgmf2 (g=d2)
    Mgmf2_scale: float = 0.1,
    W_gmf2: float | None = None,
    L_gmf2: float | None = None,
    gm_id_gmf2: float | None = None,

    # DACFC-specific: inner cap to vss
    Cm2_dac: float = 0.5e-12,

    # R89 B hybrid d1-FF retrofit (R88 C inherited DFCFC1/DFCFC2 Mgf2)
    Mgf2_scale: float = 0.1,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # R91 C delta: Mgmf1 co-route scale knob. Default 1.0 leaves the
    # Mgmf1 pair at its v127 sizing (W_gmf1_in unchanged) but RE-ROUTES
    # the drains from nga1_a/nga1_b to out. Set to 0 to effectively
    # disable Mgmf1 (W collapsed to a small floor of 0.5 um).
    Mgmf1_co_scale: float = 1.0,

    # R83 D inherited: PMOS cascode pair on the input
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # R84 D inherited: NMOS cascode pair on mirror
    W_cas_n_mirror: float = 4e-6,
    L_cas_n_mirror: float = 1e-6,
    W_cas_n_bias: float = 4e-6,
    L_cas_n_bias: float = 1e-6,
    I_ref_cas_n: float = 1e-5,

    # R122 C: qualitative DRC annotation surface. Additive metadata;
    # default False keeps every pre-R122 caller byte-for-byte identical.
    # When True, three qualitative hints attach to the cell:
    #   MATCHING on (M1, M2)               -- PMOS diff pair
    #   MATCHING on (Mga1_a, Mga1_b)       -- DACFC matched gma1 pair
    #   MATCHING on (Mgmf1_n, Mgmf1_p)     -- DACFC input-coupled FF pair
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for Song DACFC 3-stage opamp with R85 C cascode
    recipe + R89 B Mgf2_hybrid d1-FF + R91 C Mgmf1 drain re-route to
    out (R91 C / BL-093 GBW co-redesign attempt).

    DACFC signature (matched gma1 pair + Mga2 buffer + Mgmf2 output
    FF) preserved verbatim. R91 C delta: Mgmf1_n.drain and Mgmf1_p.drain
    re-routed from nga1_a/nga1_b to out, plus Mlcs3 widened to absorb
    the additional PMOS source currents at out.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain.
        W_diff..L_cs3: per-device physical sizing.
        gm_id_diff..gm_id_stage3: gm/Id intent overrides.
        Cm1: outer Miller (out <-> d2). Default 2.0 pF.
        W_gma1, L_gma1: matched gma1 PMOS pair (active cascode).
        W_gmf1_in, L_gmf1_in: input-coupled gmf1 PMOS pair.
        W_gma2, L_gma2: single gma2 PMOS buffer.
        Mgmf2_scale: original DACFC Mgmf2 (g=d2) scale. Default 0.1.
        W_gmf2, L_gmf2, gm_id_gmf2: original Mgmf2 sizing overrides.
        Cm2_dac: inner cap to vss (nga1_a <-> vss). Default 0.5 pF.
        Mgf2_scale: R89 B hybrid d1-FF scale. Default 0.1.
        W_gf2, L_gf2, gm_id_gf2: Mgf2_hybrid sizing overrides.
        Mgmf1_co_scale: R91 C scale for the re-routed Mgmf1 pair.
            Default 1.0 keeps W_gmf1_in unchanged. Set to 0 to disable
            (W collapsed to 0.5 um floor).
        W_cas, L_cas, I_ref_cas: PMOS input-cascode pair sizing.
        W_cas_n_mirror, L_cas_n_mirror: NMOS mirror-cascode pair.
        W_cas_n_bias, L_cas_n_bias: stacked NMOS-diode pair.
        I_ref_cas_n: bookkeeping placeholder.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # DACFC original Mgmf2 draws DC current at out (preserved).
    I_gmf2 = Mgmf2_scale * K_cs3_load * I_ref
    # R89 B delta: Mgf2_hybrid draws additional DC current at out.
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref
    # R91 C delta: each re-routed Mgmf1 device contributes a PMOS source
    # current to out. Estimate the ratio of Mgmf1 to Mcs3 W as the
    # current ratio (Mgmf1_co_scale * W_gmf1_in / W_cs3) and apply 2x
    # for the n + p pair. Floored at 0 if Mgmf1_co_scale = 0.
    if Mgmf1_co_scale > 0:
        # The "effective" Mgmf1 current contribution at out, expressed
        # as a multiple of K_cs3_load * I_ref so it composes cleanly
        # into K_cs3_load_total.
        i_mgmf1_per_device_ratio = (
            (Mgmf1_co_scale * W_gmf1_in) / max(W_cs3, 1e-12)
        )
        mgmf1_total_scale = 2.0 * i_mgmf1_per_device_ratio
    else:
        mgmf1_total_scale = 0.0
    K_cs3_load_total = int(round(
        K_cs3_load * (1.0 + Mgmf2_scale + Mgf2_scale + mgmf1_total_scale)
    ))
    if K_cs3_load_total < 1:
        K_cs3_load_total = 1

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

    # Original Mgmf2 (g=d2) sizing.
    if L_gmf2 is None:
        L_gmf2 = L_cs3
    if gm_id_gmf2 is not None:
        W_gmf2 = closed_form_W("pmos", gm_id=gm_id_gmf2,
                               I_d=I_gmf2, L=L_gmf2).W
    elif W_gmf2 is None:
        W_gmf2 = W_cs3 * Mgmf2_scale

    # R89 B delta: Mgf2_hybrid (g=d1) sizing.
    if L_gf2 is None:
        L_gf2 = L_cs3
    if gm_id_gf2 is not None:
        W_gf2 = closed_form_W("pmos", gm_id=gm_id_gf2,
                              I_d=I_gf2, L=L_gf2).W
    elif W_gf2 is None:
        W_gf2 = W_cs3 * Mgf2_scale

    # R91 C delta: apply Mgmf1_co_scale to W_gmf1_in. When 0, collapse
    # to a 0.5 um floor (effectively disable the device but keep ngspice
    # happy with non-zero geometry).
    if Mgmf1_co_scale > 0:
        W_gmf1_co = W_gmf1_in * Mgmf1_co_scale
    else:
        W_gmf1_co = 0.5e-6

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # Bias chain
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

    # Cascode bias rail (pcas)
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

    # NMOS mirror-cascode bias rail (ncas)
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

    # Stage 1: cascoded PMOS diff pair with cascoded NMOS mirror
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode_int", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1_int",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    devices.append(_mosfet("Mcas1", "pmos",
                            d="d1_diode", g="pcas", s="d1_diode_int", b="vdd",
                            W=W_cas, L=L_cas))
    devices.append(_mosfet("Mcas2", "pmos",
                            d="d1",       g="pcas", s="d1_int",       b="vdd",
                            W=W_cas, L=L_cas))

    devices.append(_mosfet("M3", "nmos",
                            d="d1_diode_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos",
                            d="d1_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    devices.append(_mosfet("Mcas3_n", "nmos",
                            d="d1_diode", g="ncas", s="d1_diode_n_int", b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))
    devices.append(_mosfet("Mcas4_n", "nmos",
                            d="d1",       g="ncas", s="d1_n_int",       b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))

    # Stage 2: NMOS CS amplifier driven by d1
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # Stage 2 replica-bias leg (bias_d2 reused as DACFC gma1 gate)
    devices.append(_mosfet("Mcs2_repl", "nmos",
                            d="bias_d2", g="d1_diode", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))
    devices.append(_mosfet("Mbp_d2", "pmos",
                            d="bias_d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mlcs2", "pmos",
                            d="d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # Stage 3: PMOS CS amplifier (output stage)
    devices.append(_mosfet("Mcs3", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))

    # DACFC delta: matched gma1 PMOS pair (active cascode) -- PRESERVED
    devices.append(_mosfet("Mga1_a", "pmos",
                            d="nga1_a", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))
    devices.append(_mosfet("Mga1_b", "pmos",
                            d="nga1_b", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))

    # R91 C delta: input-coupled gmf1_PMOS pair -- DRAINS RE-ROUTED to
    # ``out`` (was nga1_a / nga1_b in v127). W scaled by Mgmf1_co_scale.
    devices.append(_mosfet("Mgmf1_n", "pmos",
                            d="out", g="inn", s="vdd", b="vdd",
                            W=W_gmf1_co, L=L_gmf1_in))
    devices.append(_mosfet("Mgmf1_p", "pmos",
                            d="out", g="inp", s="vdd", b="vdd",
                            W=W_gmf1_co, L=L_gmf1_in))

    # DACFC delta: single gma2 PMOS buffer -- PRESERVED
    devices.append(_mosfet("Mga2", "pmos",
                            d="nga2", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma2, L=L_gma2))

    # DACFC delta: output-stage FF Mgmf2 (g=d2) -- PRESERVED.
    devices.append(_mosfet("Mgmf2", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf2, L=L_gmf2))

    # R89 B inherited: hybrid d1-gated Mgf2-style active feedforward.
    devices.append(_mosfet("Mgf2_hybrid", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # DC bias sinks for the cascode / buffer drain nodes. After R91 C
    # re-route, Mlcs_ga1_a/b only have to sink Mga1_a/b currents alone
    # (Mgmf1 currents now flow to out via the wider Mlcs3 below).
    devices.append(_mosfet("Mlcs_ga1_a", "nmos",
                            d="nga1_a", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga1_b", "nmos",
                            d="nga1_b", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga2", "nmos",
                            d="nga2", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf2 + Mgf2_hybrid +
    # 2x re-routed Mgmf1 DC currents at out. R91 C delta: total scale
    # is now 1 + Mgmf2_scale + Mgf2_scale + 2 * (Mgmf1_co_scale *
    # W_gmf1_in / W_cs3).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # Compensation: Cm1 outer (NMCF-style) + Cm2_dac to vss -- PRESERVED.
    devices.append(_cap("Cm1",     p="out",    n="d2",  value=Cm1))
    devices.append(_cap("Cm2_dac", p="nga1_a", n="vss", value=Cm2_dac))

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
                    "PMOS diff pair under the input cascode stack. "
                    "The R91 C Mgmf1 drain re-route to ``out`` couples "
                    "stage-1 imbalance directly into the output node "
                    "via push-pull feedforward; M1/M2 Vth mismatch "
                    "becomes a stage-3 DC offset that the cascode "
                    "rout would otherwise filter"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mga1_a", "Mga1_b"),
            extras={
                "rationale": (
                    "DACFC matched gma1 PMOS pair (active cascode) — "
                    "drives the symmetric stage-1 cascode nodes "
                    "nga1_a/nga1_b. Mismatch unbalances the DACFC "
                    "fully-differential cascode and re-injects the "
                    "imbalance into the d2 path via the Mga2 buffer"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mgmf1_n", "Mgmf1_p"),
            extras={
                "rationale": (
                    "DACFC input-coupled FF pair — Mgmf1_n (g=inn) / "
                    "Mgmf1_p (g=inp) with R91 C drains re-routed to "
                    "``out``. Mismatch injects differential offset "
                    "directly into the output node, where the wider "
                    "Mlcs3 sink cannot reject it; contaminates the "
                    "GBW co-design path under empirical sweep"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
