"""
Built-in template: nested_miller_3stage_dacfc_gbw_v200.

R105 Agent D / BL-093 DACFC GBW v200 — Mgmf1 drain re-route ``out`` +
Cm1 dominant-pole + optional cascode widening, layered on the R91 C +
R92 D mechanisms.

Context: M-SIZING-PAPER-GRADE is already at 14/17 DONE (per R94 close)
via the R92 D ``dacfc_mgf2_reduced_v128`` cascode-widening lever. The
R105 prompt to advance 11/17 -> 12/17 is severely stale; the actual
target value is now 14/17 -> 15/17 for any *new* cohort variant. The
DACFC variant covered by the v200 template is therefore a 2nd
falsification-vs-falsification record on the same Mgmf1-drain-routing
hypothesis that R91 C tested in v128 ``dacfc_gbw_corouted``.

R91 C v128 finding recap
========================

  * Mgmf1_n.drain / Mgmf1_p.drain re-routed from nga1_a / nga1_b to
    ``out``. Mlcs3 widened to absorb the new PMOS source currents.
  * Sweep Cm1 in {2, 4, 6, 8, 12, 16, 32, 80} pF x Mgmf1_co_scale in
    {0.0, 0.5, 1.0}. Best GBW = 4.70 MHz, min GBW = 5.32 MHz, GBW
    movement vs Cm1 < 1.5%.
  * At Mgmf1_co_scale = 1.0 the input-coupled FF pair contributes
    push-pull current at ``out``, *crushing* gain from 118.31 dB ->
    96.71 dB. The structural mechanism is parallel feedforward at
    ``out`` lowering effective output impedance.
  * Empirical conclusion: Mgf2_hybrid (PMOS, g=d1, d=out) is the
    actual GBW driver. The Mgmf1 drain re-route is a no-op on the
    dominant pole; it only damages gain.

R92 D v128 follow-up: the working mechanism turned out to be cascode-
stack widening (W_cas_n_mirror 8 -> 80 um, W_cas_n_bias 4 -> 40 um at
v127 Mgf2_scale = 0.14, Mgmf1 v127 routing). Measured: gain = 118.345
dB / GBW = 2.025 MHz, both PASS at locked-in default sizing.

R105 v200 delta
===============

The v200 template ships the Mgmf1 -> out re-route from R91 C v128 as
its **default topology**, BUT exposes the cascode-widening kwargs from
R92 D v128 so the two attack vectors can be composed empirically. This
asks the falsifiable question: does the R92 D cascode-widening lever
still work *with* the Mgmf1 routing crushed gain, or does the R91 C
gain loss dominate even with R92 D widening?

Three configurable axes:

  1. ``Mgmf1_co_scale`` (default 1.0 = full R91 C re-route, drains at
     ``out``). Set 0.0 to disable Mgmf1 (W floored to 0.5 um while
     drains stay at ``out`` — distinguishes routing from sizing).
  2. ``W_cas_n_mirror`` (default 80 um = R92 D widening). Set 8 um to
     recover R91 C baseline.
  3. ``W_cas_n_bias`` (default 40 um = R92 D widening). Set 4 um to
     recover R91 C baseline.

The Cm1 default is preserved at 6 pF (R89 B v127 default) so the
sweep results are directly comparable to v128. The v200 template is
otherwise byte-equal v128 ``dacfc_gbw_corouted`` (same R85 C cascode
recipe, same DACFC matched gma1 pair, same Mga2 buffer, same Mgmf2
g=d2, same Mgf2_hybrid g=d1).

Acceptance and falsification expectation
=========================================

R105 prompt acceptance: gain_dc >= 115 dB AND GBW in paper window
[1.60, 2.40] MHz (paper 2.0 MHz +/-20%).

Predicted outcomes (based on R91 C + R92 D ground truth):

  * v200 default (Mgmf1_co_scale=1.0 + R92 D cascode widening): gain
    likely FAILS (Mgmf1 re-route crushes ~22 dB on top of v127
    baseline). Falsification confirms R91 C finding survives R92 D
    cascode widening — the structural gain-loss mechanism is
    independent of the GBW lever.
  * v200 with Mgmf1_co_scale=0 (drains at ``out`` but sizing collapsed):
    expect gain restored close to v127 (~118 dB) AND GBW in paper
    window (cascode widening drives the pole down). This *should*
    match the R92 D result but via the R91 C drain routing — a
    cleaner separation of routing-vs-sizing.

If v200 with Mgmf1_co_scale=0 + cascode widening PASSES both: this is
not a new milestone advance (M-SIZING-PAPER-GRADE already DONE at
14/17 via R92 D), but a documented *equivalence* between two
structurally-distinct templates closing the same cohort variant. The
test gate asserts gain >= 115 dB AND GBW window; honest 14/17 hold
preserved either way.

If v200 default (Mgmf1_co_scale=1.0) PASSES both: this would be a
R91 C falsification *break*. Mark CHANGELOG accordingly.

If both configurations FAIL: honest 14/17 hold confirmed; document the
structural reason.

Architectural devices preserved verbatim
=========================================

R85 C cascode recipe + DACFC backbone:

  1. Mga1_a / Mga1_b matched PMOS pair (g=bias_d2, d=nga1_a / nga1_b)
     -- DACFC matched gma1 active cascode -- PRESERVED.
  2. Mga2 single PMOS buffer (g=bias_d2, d=nga2) -- PRESERVED.
  3. Mgmf2 (PMOS, g=d2, d=out) -- original DACFC output FF -- PRESERVED.
  4. Mgf2_hybrid (PMOS, g=d1, d=out) -- R89 B d1-FF -- PRESERVED.
  5. Mlcs_ga1_a / Mlcs_ga1_b / Mlcs_ga2 nga DC bias sinks -- PRESERVED.
  6. Cm1 (out <-> d2) -- NMCF-style outer Miller -- PRESERVED.
  7. Cm2_dac (nga1_a <-> vss) -- inner cap -- PRESERVED.
  8. R85 C cascode recipe (Mcas1/2 PMOS input + Mcas3_n/4_n NMOS
     mirror with DFCFC1-style diode loop at d1_diode) -- PRESERVED
     with WIDENED W on the cascode stack devices (R92 D defaults).

R105 v200 delta vs v128 dacfc_gbw_corouted
============================================

  * W_cas_n_mirror default 8 -> 80 um (R92 D widening adopted)
  * W_cas_n_bias default 4 -> 40 um (R92 D widening adopted)
  * Mgmf1_n.drain / Mgmf1_p.drain -> ``out`` (R91 C routing PRESERVED)
  * All other kwargs byte-equal v128 dacfc_gbw_corouted defaults
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


def build_nested_miller_3stage_dacfc_gbw_v200(
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

    # R89 B Mgf2_hybrid d1-FF (preserved at v127 default 0.14 by R92 D
    # finding; reduction below 0.13 hits the empirically-confirmed gain
    # cliff)
    Mgf2_scale: float = 0.14,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # R91 C delta inherited: Mgmf1 co-route scale (default 1.0 = full
    # re-route, drains at ``out`` with W = W_gmf1_in). Set to 0 to
    # collapse W to 0.5 um floor while keeping drains at ``out`` -- this
    # distinguishes the routing axis from the sizing axis.
    Mgmf1_co_scale: float = 1.0,

    # R83 D inherited: PMOS cascode pair on input
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # R92 D widening adopted as v200 default: NMOS cascode pair WIDENED
    # from R91 C's 8 um -> 80 um (10x). Empirical: lowers cascoded
    # mirror's effective output impedance, pulling dominant pole from
    # ~5.5 MHz into [1.6, 2.4] MHz paper window.
    W_cas_n_mirror: float = 80e-6,
    L_cas_n_mirror: float = 1e-6,
    # R92 D widening adopted as v200 default: cascode bias rail WIDENED
    # 4 um -> 40 um (10x) to match the wider stack.
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
    """Build a Cell IR for the R105 v200 DACFC variant.

    Topology = R85 C cascode + DACFC matched gma1 pair + Mgmf1 drains
    re-routed to ``out`` (R91 C delta) + R89 B Mgf2_hybrid d1-FF + R92 D
    cascode-stack widening defaults. Asks the empirical question of
    whether R91 C's gain-crash mechanism survives R92 D's GBW lever.

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
        Mgf2_scale: R89 B Mgf2_hybrid scale. Default 0.14 (v127 preserved;
            reduction below 0.13 falls off R92 D's empirical gain cliff).
        W_gf2, L_gf2, gm_id_gf2: Mgf2_hybrid sizing overrides.
        Mgmf1_co_scale: R91 C re-route scale. Default 1.0 (full sizing
            at W_gmf1_in). Set to 0 to floor W at 0.5 um while keeping
            drains at ``out``.
        W_cas, L_cas, I_ref_cas: PMOS input-cascode pair sizing.
        W_cas_n_mirror: NMOS cascode mirror W (R92 D widened default
            80 um).
        L_cas_n_mirror: NMOS cascode mirror L.
        W_cas_n_bias: NMOS cascode bias rail W (R92 D widened default
            40 um).
        L_cas_n_bias: stacked NMOS-diode pair L.
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

    # DACFC matched gma1 PMOS pair (active cascode) -- PRESERVED
    devices.append(_mosfet("Mga1_a", "pmos",
                            d="nga1_a", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))
    devices.append(_mosfet("Mga1_b", "pmos",
                            d="nga1_b", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))

    # R91 C delta inherited: Mgmf1 input-coupled PMOS pair DRAINS at
    # ``out`` (not nga1_a/nga1_b). W scaled by Mgmf1_co_scale.
    devices.append(_mosfet("Mgmf1_n", "pmos",
                            d="out", g="inn", s="vdd", b="vdd",
                            W=W_gmf1_co, L=L_gmf1_in))
    devices.append(_mosfet("Mgmf1_p", "pmos",
                            d="out", g="inp", s="vdd", b="vdd",
                            W=W_gmf1_co, L=L_gmf1_in))

    # DACFC single gma2 PMOS buffer -- PRESERVED
    devices.append(_mosfet("Mga2", "pmos",
                            d="nga2", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma2, L=L_gma2))

    # DACFC output-stage FF Mgmf2 (g=d2) -- PRESERVED
    devices.append(_mosfet("Mgmf2", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf2, L=L_gmf2))

    # R89 B hybrid d1-gated Mgf2-style active feedforward.
    devices.append(_mosfet("Mgf2_hybrid", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # DC bias sinks for the cascode / buffer drain nodes.
    devices.append(_mosfet("Mlcs_ga1_a", "nmos",
                            d="nga1_a", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga1_b", "nmos",
                            d="nga1_b", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga2", "nmos",
                            d="nga2", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # Stage-3 NMOS sink widened for Mcs3 + Mgmf2 + Mgf2_hybrid + 2x
    # re-routed Mgmf1 currents at ``out``.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # Compensation: Cm1 outer + Cm2_dac to vss -- PRESERVED.
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
                    "v200 ships the R91 C Mgmf1->out re-route as the "
                    "default topology — M1/M2 Vth mismatch propagates "
                    "through both the cascoded mirror and the direct "
                    "Mgmf1 push-pull path into a stage-3 DC offset "
                    "that the R92 D widening lever cannot reject"
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
                    "fully-differential cascode regardless of the "
                    "v200 cascode-widening or Mgmf1 routing axes"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mgmf1_n", "Mgmf1_p"),
            extras={
                "rationale": (
                    "DACFC input-coupled FF pair under the v200 "
                    "default routing (drains at ``out``). Mismatch "
                    "injects differential push-pull current into the "
                    "output node and crosses the empirical gain cliff "
                    "the v201 falsification variant probes — pair "
                    "matching is prerequisite for any clean v200/v201 "
                    "comparison"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
