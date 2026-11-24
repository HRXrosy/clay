"""
Built-in template: nested_miller_3stage_dacfc_gbw_v201.

R107 Agent D / BL-093 DACFC GBW v201 — falsification variant of v200
(R105 D) that switches the Mlcs3 output-sink sizing from integer-
rounded ``M = int(round(K_cs3_load_total))`` to FLOATING-POINT
fractional W (``W_mlcs3 = W_bn * K_cs3_load_total_float``, M=1).

Hypothesis under test (R106 C audit doc)
========================================

R106 C's coordinate-localization sweep over Mgmf1_co_scale in
[0.75, 0.90] @ 0.025 resolution found a sharp 21.5 dB gain cliff
between Mgmf1_co_scale = 0.825 (PASS, 118.206 dB) and 0.850 (FAIL,
96.648 dB). R106 C's primary mechanism hypothesis: the
``K_cs3_load_total = int(round(...))`` calculation in v200's builder
flips the Mlcs3 NMOS-sink mirror multiplier from 5 -> 6 EXACTLY at
the half-step boundary 4 * 1.37162 = 5.487 -> rounds to 5 vs
4 * 1.37713 = 5.509 -> rounds to 6. The integer step changes the
DC current balance at the output node by 20% (5 unit currents ->
6 unit currents), which can drive Mcs3 / Mgmf2 / Mgf2_hybrid /
Mgmf1 pair out of saturation -> r_o crash -> 21 dB gain drop.

This v201 variant kills the integer rounding entirely. The
mirror-multiplier scaling is now applied through fractional W (the
SPICE driver's ``W={W:g}`` emit accepts arbitrary float; only the
``m={M}`` flag is cast to int via ``int(dev.sizing.get("M", 1))``).
So instead of:

    M_mlcs3 = int(round(K_cs3_load * (1 + ... + mgmf1_total_scale)))

we use:

    W_mlcs3 = W_bn * K_cs3_load * (1 + ... + mgmf1_total_scale)
    M_mlcs3 = 1

This makes the effective channel area W*L*M scale CONTINUOUSLY with
Mgmf1_co_scale. If R106 C's int-round hypothesis is correct, the
sharp cliff at (0.825, 0.850] should DISAPPEAR — the gain curve
should degrade smoothly across the same sweep range (a few dB drop
per 0.025-step, not a 21 dB single-step). If the cliff PERSISTS even
with continuous Mlcs3 scaling, R106 C's hypothesis is FALSIFIED and
a deeper mechanism (e.g., gm-contribution sign-flip from Mgmf1 pair,
pole-zero crossing, loop-stability boundary) must be the cause.

This is the ONLY difference between v201 and v200. Every other knob,
device, sizing literal, and architectural recipe is preserved
verbatim from v200.

Why fractional W is byte-equivalent to fractional M for DC analysis
-------------------------------------------------------------------

In SPICE's BSIM4 (sky130 nfet/pfet), a transistor's drain current
under fixed Vgs/Vds is proportional to W*M/L (M is the parallel
finger count; W is per-finger width). The product W*M is what
determines the gm and ro at the operating point. For DC analysis,
W = 4u, M = 12 and W = 48u, M = 1 produce IDENTICAL device behavior
at the same Vgs/Vds (modulo parasitic capacitance differences that
only matter at AC, which is not what's driving the v200 cliff).

So v201 with W_mlcs3 = 4u * 5.487 = 21.949u, M = 1 is electrically
equivalent (to first order) to v200 with W_mlcs3 = 4u, M = 5 — same
W*M product (21.949u vs 20u, a 9.7% discrepancy that REPLACES the
20% v200 step from M=5 to M=6 with continuous scaling).

If R106 C's hypothesis #1 is right, we'd see:

  * v200 Mgmf1_co_scale=0.825: M_mlcs3=5 (= int(round(5.487)))
    => W*M = 4u * 5 = 20.000u  =>  cliff PASS  (gain=118.21 dB)
  * v200 Mgmf1_co_scale=0.850: M_mlcs3=6 (= int(round(5.509)))
    => W*M = 4u * 6 = 24.000u  =>  cliff FAIL  (gain=96.65 dB)

vs:

  * v201 Mgmf1_co_scale=0.825: W_mlcs3 = 4u * 5.487 = 21.946u, M=1
    => W*M = 21.946u  =>  gain in between (smooth)
  * v201 Mgmf1_co_scale=0.850: W_mlcs3 = 4u * 5.509 = 22.034u, M=1
    => W*M = 22.034u  =>  gain in between (smooth)

The hypothesis #1 prediction: v201 will show monotonic degradation
across the whole sweep, with NO sharp boundary at 0.825 / 0.850.

R107 D acceptance and falsification expectations
================================================

* PRIMARY HYPOTHESIS CONFIRMED: v201 fine sweep across
  [0.75, 0.95] @ 0.025 shows smooth monotonic gain degradation (no
  step > ~3 dB per 0.025-step). Cliff coordinates v200 (0.825, 0.850]
  vs v201 NONE -- mechanism CONFIRMED.

* PRIMARY HYPOTHESIS FALSIFIED: v201 still shows a sharp cliff at
  some coordinate in [0.75, 0.95]. The structural mechanism is NOT
  the int-rounding of M; deeper structure must be identified
  (RHP zero crossing? Loop-instability boundary? gm sign-flip?).

DO NOT modify v200. v200 is the baseline-of-record for the
falsification record.
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


def build_nested_miller_3stage_dacfc_gbw_v201(
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
    Cm1: float = 2.0e-12,

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

    # R89 B Mgf2_hybrid d1-FF (preserved at v127 default 0.14 by R92 D)
    Mgf2_scale: float = 0.14,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # R91 C delta inherited: Mgmf1 co-route scale.
    Mgmf1_co_scale: float = 1.0,

    # R83 D inherited: PMOS cascode pair on input
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # R92 D widening adopted as v200 default
    W_cas_n_mirror: float = 80e-6,
    L_cas_n_mirror: float = 1e-6,
    W_cas_n_bias: float = 40e-6,
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
    """Build a Cell IR for the R107 v201 falsification variant of v200.

    The ONLY difference vs v200: Mlcs3 sizing is computed as a
    floating-point W * M product instead of an int-rounded M with
    fixed W. See module docstring for the falsification rationale.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    I_gmf2 = Mgmf2_scale * K_cs3_load * I_ref
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref

    # R91 C delta: re-routed Mgmf1 PMOS pair contributes source current
    # at ``out``. Estimate the per-device ratio (Mgmf1_co_scale * W_gmf1_in
    # / W_cs3) and apply 2x for the n + p pair.
    if Mgmf1_co_scale > 0:
        i_mgmf1_per_device_ratio = (
            (Mgmf1_co_scale * W_gmf1_in) / max(W_cs3, 1e-12)
        )
        mgmf1_total_scale = 2.0 * i_mgmf1_per_device_ratio
    else:
        mgmf1_total_scale = 0.0

    # R107 D falsification delta: keep K_cs3_load_total as a FLOAT.
    # v200 cast this to int via round(); v201 keeps continuous scaling.
    K_cs3_load_total_float = (
        K_cs3_load * (1.0 + Mgmf2_scale + Mgf2_scale + mgmf1_total_scale)
    )
    # Floor at a minimum scaling so Mlcs3 cannot vanish. v200 used
    # ``if K_cs3_load_total < 1: K_cs3_load_total = 1``; mirror that
    # here as ``max(K_cs3_load_total_float, 1.0)``.
    if K_cs3_load_total_float < 1.0:
        K_cs3_load_total_float = 1.0

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

    if L_gmf2 is None:
        L_gmf2 = L_cs3
    if gm_id_gmf2 is not None:
        W_gmf2 = closed_form_W("pmos", gm_id=gm_id_gmf2,
                               I_d=I_gmf2, L=L_gmf2).W
    elif W_gmf2 is None:
        W_gmf2 = W_cs3 * Mgmf2_scale

    if L_gf2 is None:
        L_gf2 = L_cs3
    if gm_id_gf2 is not None:
        W_gf2 = closed_form_W("pmos", gm_id=gm_id_gf2,
                              I_d=I_gf2, L=L_gf2).W
    elif W_gf2 is None:
        W_gf2 = W_cs3 * Mgf2_scale

    if Mgmf1_co_scale > 0:
        W_gmf1_co = W_gmf1_in * Mgmf1_co_scale
    else:
        W_gmf1_co = 0.5e-6

    # R107 D delta: Mlcs3 effective W*M product applied entirely via W.
    # M held at 1 (integer; SPICE driver casts via int()).
    W_mlcs3_eff = W_bn * K_cs3_load_total_float

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

    # NMOS mirror-cascode bias rail (ncas) -- R92 D widened defaults.
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

    # Stage 2 replica-bias leg
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

    # DACFC matched gma1 PMOS pair
    devices.append(_mosfet("Mga1_a", "pmos",
                            d="nga1_a", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))
    devices.append(_mosfet("Mga1_b", "pmos",
                            d="nga1_b", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))

    # R91 C delta inherited: Mgmf1 PMOS pair DRAINS at out.
    devices.append(_mosfet("Mgmf1_n", "pmos",
                            d="out", g="inn", s="vdd", b="vdd",
                            W=W_gmf1_co, L=L_gmf1_in))
    devices.append(_mosfet("Mgmf1_p", "pmos",
                            d="out", g="inp", s="vdd", b="vdd",
                            W=W_gmf1_co, L=L_gmf1_in))

    # DACFC single gma2 PMOS buffer
    devices.append(_mosfet("Mga2", "pmos",
                            d="nga2", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma2, L=L_gma2))

    # DACFC output-stage FF Mgmf2 (g=d2)
    devices.append(_mosfet("Mgmf2", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf2, L=L_gmf2))

    # R89 B hybrid d1-gated Mgf2-style active feedforward
    devices.append(_mosfet("Mgf2_hybrid", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # DC bias sinks
    devices.append(_mosfet("Mlcs_ga1_a", "nmos",
                            d="nga1_a", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga1_b", "nmos",
                            d="nga1_b", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga2", "nmos",
                            d="nga2", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # R107 D delta: Mlcs3 uses float W instead of integer M. The
    # effective W*M scales continuously with Mgmf1_co_scale.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_mlcs3_eff, L=L_bn, M=1))

    # Compensation
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
                    "v201 keeps the v200 topology but replaces the "
                    "integer-rounded Mlcs3 mirror multiplier with a "
                    "continuous float W. M1/M2 mismatch still maps to "
                    "input-referred offset; the floating-point Mlcs3 "
                    "scaling does not alter the matching requirement"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mga1_a", "Mga1_b"),
            extras={
                "rationale": (
                    "DACFC matched gma1 PMOS pair (active cascode). "
                    "v201's continuous Mlcs3 W scaling is intended to "
                    "smooth the v200 21 dB gain cliff; if the gma1 "
                    "pair is mismatched, any residual cliff structure "
                    "would be aliased with stage-1 cascode imbalance "
                    "rather than the M-multiplier hypothesis under test"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mgmf1_n", "Mgmf1_p"),
            extras={
                "rationale": (
                    "DACFC input-coupled FF pair under v201's "
                    "falsification routing (drains at ``out``, "
                    "continuous Mlcs3 scaling). Mismatch in this pair "
                    "would inject a co-variate confound into the "
                    "Mgmf1_co_scale sweep that R106 C / R107 D use to "
                    "isolate the int-round mechanism — pair-matching "
                    "must be enforced for a clean falsification record"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
