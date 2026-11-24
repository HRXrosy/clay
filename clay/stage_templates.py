"""
Stage template registry.

A "stage template" is the Python function that produces a Cell IR for
a given topology style. The .clay surface declares
`topology { style = 'foo'; }` and the compiler looks up `foo` here.

This is the bridge between the user-facing DSL (text) and the
Python-implemented circuit-skeleton library (currently in
`clay/templates/`). Future versions will allow stage templates
themselves to be authored in .clay, at which point this Python registry
becomes one of several backends.

A stage builder is `(spec, **kwargs) -> Cell`. It must accept at least
`cell_name` and `spec`; other kwargs are forwarded from the cell_def
sizing block.

v0.0.86 (P2 first slice): every stage_template MAY ALSO carry a
`StageInterface` declaration (input/output/bias/supply port types +
KPI promises + compensation tags) registered alongside the builder.
The interface is what algebraic spatial combinators (`stage_a >>
stage_b`, `feedforward(stage_a, stage_b)`) will type-check against
when the combinator runtime lands in a follow-up round. This slice
ships the typed surface only; combinator runtime is a later round.
"""

from __future__ import annotations

from typing import Callable, Optional

from .ir import Cell, Spec, Testbench
from .stage_interface import (
    PortDecl,
    SpecTarget,
    StageInterface,
    register_stage_interface,
)


StageBuilder = Callable[..., Cell]
TestbenchBuilder = Callable[[Cell], Testbench]


_REGISTRY: dict[str, StageBuilder] = {}
_TESTBENCH_REGISTRY: dict[str, TestbenchBuilder] = {}


def register_stage(
    style: str,
    builder: StageBuilder,
    *,
    testbench: Optional[TestbenchBuilder] = None,
) -> None:
    """Register a stage template under a public `style` name.

    Args:
        style: name used in `topology { style = '...'; }` in .clay source
        builder: function (spec, **kwargs) -> Cell
        testbench: optional default testbench builder for cells produced
            by this stage. If provided, the CLI's cell_def path will use
            this testbench instead of falling back to the generic AC
            testbench (which assumes differential inn/inp ports).

    Idempotent re-registration is OK (same builder for same name).
    Conflicting builders for the same name raise.
    """
    if style in _REGISTRY:
        if _REGISTRY[style] is builder:
            # If a testbench is being supplied on a re-registration with
            # the SAME builder, accept it (idempotent re-register).
            if testbench is not None:
                _TESTBENCH_REGISTRY[style] = testbench
            return
        raise RuntimeError(
            f"stage style {style!r} already registered to "
            f"{_REGISTRY[style].__name__}; refusing to overwrite with "
            f"{builder.__name__}"
        )
    _REGISTRY[style] = builder
    if testbench is not None:
        _TESTBENCH_REGISTRY[style] = testbench


def get_stage(style: str) -> StageBuilder:
    """Return the builder for a registered style; raise if unknown."""
    if style not in _REGISTRY:
        raise KeyError(
            f"unknown topology style {style!r}. "
            f"Registered: {sorted(_REGISTRY)}. "
            f"Register new stages via clay.stage_templates.register_stage()."
        )
    return _REGISTRY[style]


def get_stage_testbench(style: str) -> Optional[TestbenchBuilder]:
    """Return the default testbench builder for `style`, or None.

    Unlike `get_stage`, this returns None for unregistered styles and
    for styles that registered without a testbench (legacy stages).
    Callers should fall back to their own default in that case.
    """
    return _TESTBENCH_REGISTRY.get(style)


def list_styles() -> list[str]:
    return sorted(_REGISTRY)


# ─── Built-in stage registrations ────────────────────────────

def _bootstrap_builtin() -> None:
    """Lazy import to avoid circular import at clay package init time."""
    from .templates.common_source import (
        build_ac_testbench_cs, build_common_source_pmos_input,
    )
    from .templates.folded_cascode import build_folded_cascode
    from .templates.fully_differential import (
        build_ac_testbench_fda, build_fully_differential_pmos_input,
    )
    from .templates.nested_miller_3stage import (
        build_nested_miller_3stage_feedforward,
    )
    from .templates.nested_miller_3stage_nmcf import (
        build_nested_miller_3stage_nmcf,
    )
    from .templates.hoilee_affc import (
        build_nested_miller_3stage_affc,
    )
    from .templates.fan_smc import (
        build_nested_miller_3stage_smc,
    )
    from .templates.sau_cfcc import (
        build_nested_miller_3stage_cfcc,
    )
    from .templates.leung_nmcnr import (
        build_nested_miller_3stage_nmcnr,
    )
    from .templates.nested_miller_3stage_nmcnr_cascoded_input import (
        build_nested_miller_3stage_nmcnr_cascoded_input,
    )
    from .templates.affc_dual_miller_skip import (
        build_nested_miller_3stage_affc_dual_miller_skip,
    )
    from .templates.single_stage_dual_feedforward import (
        build_single_stage_dual_feedforward,
    )
    from .templates.peng_tcfc import (
        build_nested_miller_3stage_tcfc,
    )
    from .templates.peng_tcfc_cascoded_input import (
        build_nested_miller_3stage_tcfc_cascoded_input,
    )
    from .templates.peng_tcfc_cascoded_input_mirror import (
        build_nested_miller_3stage_tcfc_cascoded_input_mirror,
    )
    from .templates.peng_acbc import (
        build_nested_miller_3stage_acbc,
    )
    from .templates.leung_dfcfc2 import (
        build_nested_miller_3stage_dfcfc2,
    )
    from .templates.dfcfc2_cascoded_input_mirror import (
        build_nested_miller_3stage_dfcfc2_cascoded_input_mirror,
    )
    # v0.1.09 Round 86 Agent D: BL-093 batch rollout 5 NMCF variants.
    # R85 C cascode recipe (telescopic PMOS input + cascoded NMOS mirror
    # with DFCFC1-style diode loop at high-Z node d1) applied to 5 more
    # NMCF-class variants: ACBC, IAC, RAFFC, LEC, PFC. All preserve their
    # respective architectural signatures verbatim at stage-2 / stage-3 /
    # compensation network levels.
    from .templates.acbc_cascoded_input_mirror import (
        build_nested_miller_3stage_acbc_cascoded_input_mirror,
    )
    from .templates.iac_cascoded_input_mirror import (
        build_nested_miller_3stage_iac_cascoded_input_mirror,
    )
    from .templates.raffc_cascoded_input_mirror import (
        build_nested_miller_3stage_raffc_cascoded_input_mirror,
    )
    from .templates.lec_cascoded_input_mirror import (
        build_nested_miller_3stage_lec_cascoded_input_mirror,
    )
    from .templates.pfc_cascoded_input_mirror import (
        build_nested_miller_3stage_pfc_cascoded_input_mirror,
    )
    from .templates.dfcfc1_cascoded_input_mirror import (
        build_nested_miller_3stage_dfcfc1_cascoded_input_mirror,
    )
    from .templates.nmcnr_cascoded_input_mirror import (
        build_nested_miller_3stage_nmcnr_cascoded_input_mirror,
    )
    from .templates.az_cascoded_input_mirror import (
        build_nested_miller_3stage_az_cascoded_input_mirror,
    )
    from .templates.azc_cascoded_input_mirror import (
        build_nested_miller_3stage_azc_cascoded_input_mirror,
    )
    from .templates.dacfc_cascoded_input_mirror import (
        build_nested_miller_3stage_dacfc_cascoded_input_mirror,
    )
    # v0.1.11 Round 88 Agent C: NMCNR hybrid d1-gated FF retrofit pilot.
    # Validates R87 C structural finding (cascoded mirror REQUIRES d1-
    # gated Mgf2-style FF for +30-35 dB gain product). Adds DFCFC2/
    # DFCFC1 inherited Mgf2 (PMOS, g=d1, d=out, s=vdd) onto NMCNR's
    # nulling-R + 2-Miller-cap-to-rz_n backbone. NMCNR baseline (R87 C)
    # 83.29 dB → expected 115-120 dB.
    from .templates.nmcnr_hybrid_d1_ff import (
        build_nested_miller_3stage_nmcnr_hybrid_d1_ff,
    )
    # v0.1.12 Round 89 Agent B: AZ / AZC / DACFC hybrid d1-gated FF
    # retrofit batch (BL-093 9/17 -> 11/17 or 12/17 attempt). Same
    # retrofit pattern validated by R88 C on NMCNR, rolled to three
    # more FAIL variants. AZ + DACFC: dual-FF (preserve original Mgmf2
    # + add Mgf2_hybrid). AZC: direct graft (no original FF).
    from .templates.az_hybrid_d1_ff import (
        build_nested_miller_3stage_az_hybrid_d1_ff,
    )
    from .templates.azc_hybrid_d1_ff import (
        build_nested_miller_3stage_azc_hybrid_d1_ff,
    )
    from .templates.dacfc_hybrid_d1_ff import (
        build_nested_miller_3stage_dacfc_hybrid_d1_ff,
    )
    # v0.1.13 Round 90 Agent C: AZC GBW co-redesign via Cazc4 Miller
    # path. R89 B AZC v127 landed gain=119.51 dB PASS but GBW=4.35 MHz
    # (+45% off paper 3.0 MHz, FAIL). Adds Cazc4 cap (nazc1 <-> out)
    # to introduce a new Miller path through AZC's archetype-trio
    # high-Z node nazc1, pulling the dominant pole down to land GBW
    # in paper [2.4, 3.6] MHz window. R89 B byte-exact preserved.
    from .templates.azc_hybrid_d1_ff_cazc4 import (
        build_nested_miller_3stage_azc_hybrid_d1_ff_cazc4,
    )
    # v0.1.14 Round 91 Agent C: DACFC GBW co-redesign via Mgmf1 drain
    # re-route. R89 B DACFC v127 landed gain=118.31 dB PASS but
    # GBW=5.53 MHz (+176% off paper 2.0 MHz, FAIL). R91 C delta:
    # re-route Mgmf1_n/p drains from nga1_a/nga1_b to out, so the
    # input-coupled FF pair actively contributes at the output node
    # where Cm1 (out <-> d2) can dominate the pole.
    from .templates.dacfc_gbw_corouted import (
        build_nested_miller_3stage_dacfc_gbw_corouted,
    )
    # v0.1.15 Round 92 Agent D: DACFC cascode-stack widening to land
    # M-SIZING-PAPER-GRADE 12/17. R91 C identified Mgf2_hybrid (PMOS,
    # g=d1, d=out) as actual GBW driver. R92 D empirical sweep found
    # the Mgf2-reduction hypothesis falsified (cliff at Mgf2<0.13)
    # but the cascode-widening axis succeeded: widening the NMOS
    # cascode stack and its bias rail by 10x (W_cas_n_mirror 8 -> 80
    # um, W_cas_n_bias 4 -> 40 um) pulls the dominant pole into the
    # paper window via output-impedance reduction. Mgf2_scale stays
    # at v127 default 0.14. All v127 device connectivity preserved.
    from .templates.dacfc_mgf2_reduced import (
        build_nested_miller_3stage_dacfc_mgf2_reduced,
    )
    # R93 C / BL-093 v128 RAFFC architectural redesign for
    # M-SIZING-PAPER-GRADE 12/17 -> 13/17. The R92 D cascode-W-as-lever
    # pattern was empirically FALSIFIED on RAFFC (2-axis sweep over
    # (Cm1, W_cas_n_mirror) showed both are pure GBW levers, gain
    # invariant at 27 dB). The actual RAFFC gain lever discovered:
    # K_cs3_load 4 -> 1 + W_cs3 72.68 -> 98 um sweet spot + Mgf_rev
    # super-weakening (W=0.42um, L=10um, scale=1e-4). Measured:
    # gain=125.37 dB PASS / GBW=1.330 MHz PASS.
    from .templates.raffc_redesign import (
        build_nested_miller_3stage_raffc_redesign,
    )
    # R94 C / BL-093 v128 IAC architectural redesign for
    # M-SIZING-PAPER-GRADE 13/17 -> 14/17. The 2-axis sweep over
    # (Cm1_iac, W_cas_n_mirror) at R86 D v124 baseline (gain=119.5 dB
    # PASS, GBW=4.73 MHz FAIL above paper window [2.80, 4.20] MHz)
    # showed gain INVARIANT at 119.5 dB across all 12 cells, W_cas_n_mirror
    # is the GBW lever. A second sweep falsified all four IAC-unique
    # compensation knobs (Cm2_iac, R0_iac, W_gmt, L_gmt) as GBW levers
    # -- they are byte-identical no-ops on gain/GBW within numerical
    # noise. The lever turns out the same mechanism as R92 D DACFC
    # (cascoded NMOS mirror output impedance), with smaller magnitude
    # (2.5x widening vs 10x). R94 C delta: W_cas_n_mirror 8 -> 20 um,
    # W_cas_n_bias 4 -> 10 um. Measured: gain=119.510 dB PASS / GBW=3.474
    # MHz PASS (0.75% off paper target 3.50 MHz).
    from .templates.iac_redesign import (
        build_nested_miller_3stage_iac_redesign,
    )
    from .templates.leung_dfcfc1 import (
        build_nested_miller_3stage_dfcfc1,
    )
    from .templates.alfio_raffc import (
        build_nested_miller_3stage_raffc,
    )
    from .templates.qu_lec import (
        build_nested_miller_3stage_lec,
    )
    from .templates.ramos_pfc import (
        build_nested_miller_3stage_pfc,
    )
    from .templates.qu2017_azc import (
        build_nested_miller_3stage_azc,
    )
    from .templates.song_dacfc import (
        build_nested_miller_3stage_dacfc,
    )
    from .templates.peng_iac import (
        build_nested_miller_3stage_iac,
    )
    from .templates.tan_clia import (
        build_nested_miller_3stage_clia,
    )
    from .templates.yan_az import (
        build_nested_miller_3stage_az,
    )
    from .templates.telescopic_cascode import build_telescopic_cascode
    # v0.0.92 Round 68 Agent R — 3 stdlib opamps anchored on the new
    # third-batch functional-role keywords.
    from .templates.regulated_cascode_opamp import (
        build_regulated_cascode_opamp,
    )
    from .templates.class_AB_output_opamp import (
        build_class_AB_output_opamp,
    )
    from .templates.dynamic_bias_opamp import (
        build_dynamic_bias_opamp,
    )
    from .templates.two_stage_miller_pmos import (
        build_ac_testbench, build_two_stage_miller_pmos,
    )
    from .templates.two_stage_miller_pmos_explicit_bias import (
        build_ac_testbench as build_ac_testbench_explicit_bias,
        build_two_stage_miller_pmos_explicit_bias,
    )
    from .templates.two_stage_miller_nmos import build_two_stage_miller_nmos
    from .templates.lc_tank_vco import build_lc_tank_vco
    from .templates.lc_tank_vco_testbench import build_lc_tank_vco_testbench

    register_stage("two_stage_miller_pmos", build_two_stage_miller_pmos)
    register_stage("two_stage_miller_nmos", build_two_stage_miller_nmos)
    register_stage(
        "two_stage_miller_pmos_explicit_bias",
        build_two_stage_miller_pmos_explicit_bias,
        testbench=build_ac_testbench_explicit_bias,
    )
    register_stage("folded_cascode",        build_folded_cascode)
    register_stage("telescopic_cascode",    build_telescopic_cascode)
    register_stage(
        "common_source_pmos_input",
        build_common_source_pmos_input,
        testbench=build_ac_testbench_cs,
    )
    register_stage(
        "fully_differential_pmos_input",
        build_fully_differential_pmos_input,
        testbench=build_ac_testbench_fda,
    )
    register_stage(
        "nested_miller_3stage_feedforward",
        build_nested_miller_3stage_feedforward,
        testbench=build_ac_testbench,
    )
    # v0.0.91: explicit NMCF baseline (no feedforward cap). Distinct from
    # `_feedforward` above: this variant drops Cf so combinator algebra
    # can layer `feedforward(...)` as an explicit composition rather
    # than inheriting it. See clay/templates/nested_miller_3stage_nmcf.py
    # for the full topology — same signal path as `_feedforward`, only
    # the compensation network differs (Cm1 + Cm2, no Cf).
    register_stage(
        "nested_miller_3stage_nmcf",
        build_nested_miller_3stage_nmcf,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_affc",
        build_nested_miller_3stage_affc,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_smc",
        build_nested_miller_3stage_smc,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_cfcc",
        build_nested_miller_3stage_cfcc,
        testbench=build_ac_testbench,
    )
    # v0.0.56: NMCNR (nulling-R Miller) and TCFC (active feedforward +
    # cascoded sink). Both extend the NMCF baseline structurally:
    #   - NMCNR adds a series nulling resistor Rz1 in the Miller path
    #     (architectural marker: `res` device + `rz_n` shared net).
    #   - TCFC adds active feedforward Mgmf + cascoded NMOS sink
    #     (Mcas_n + Mlcs3 + diode chain) creating internal node `nd3`.
    # See clay/templates/leung_nmcnr.py and clay/templates/peng_tcfc.py
    # for the full topology descriptions.
    register_stage(
        "nested_miller_3stage_nmcnr",
        build_nested_miller_3stage_nmcnr,
        testbench=build_ac_testbench,
    )
    # v0.0.91 Round 68 Agent I: NMCNR + cascoded PMOS input pair.
    # Same compensation network as plain NMCNR (Cm1 + Cm2 sharing
    # rz_n + Rz1 nulling resistor) but the stage-1 differential pair
    # is folded one rail deeper: Mcas1/Mcas2 PMOS cascodes sit
    # between the input pair drains (d1_int / d1_diode_int) and the
    # NMOS mirror summing nodes. A `pcas` bias rail driven by
    # Mcas_b + Iref_cas sets the cascode common-gate. The
    # architectural signature is the cascode pair + the pcas rail
    # (no other NMCF-family variant in the stdlib has this).
    # See clay/templates/nested_miller_3stage_nmcnr_cascoded_input.py
    # for the full topology description. Realises the v0.0.90
    # holdout task "NMCNR + cascoded input" from fictional to a
    # concrete stdlib stage.
    register_stage(
        "nested_miller_3stage_nmcnr_cascoded_input",
        build_nested_miller_3stage_nmcnr_cascoded_input,
        testbench=build_ac_testbench,
    )
    # v0.0.91 Round 68 Agent L: 2 fictional-combo stdlib cells.
    # The v0.0.90 real-LLM DeepSeek run included three "fictional"
    # holdout tasks; Agent I landed the NMCNR + cascoded-input combo
    # in this same round (see above). Agent L lands the remaining
    # two fictional combos so future LLM runs have stdlib stage
    # templates to reach for by name.
    #
    #   - affc_dual_miller_skip: AFFC backbone (active feedforward
    #     gm_f2 + canonical signal path) + DUAL skip-over Miller caps
    #     Cm1_skip (d1↔out) and Cm2_skip (d1↔d2). Both Miller caps
    #     share an endpoint at d1 (stage-1 output) — the natural
    #     high-impedance node for maximum Miller multiplication.
    #     Cf (plain AFFC's passive feedforward) is dropped.
    #   - single_stage_dual_feedforward: Textbook single-stage
    #     diff-pair + NMOS mirror baseline + TWO symmetric PMOS
    #     feedforward devices (Mff_p, Mff_n) cross-coupled to the
    #     differential inputs and feeding the output directly from
    #     vdd. Doubles output-stage Gm for +6 dB / 2× GBW boost.
    register_stage(
        "nested_miller_3stage_affc_dual_miller_skip",
        build_nested_miller_3stage_affc_dual_miller_skip,
        testbench=build_ac_testbench,
    )
    register_stage(
        "single_stage_dual_feedforward",
        build_single_stage_dual_feedforward,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_tcfc",
        build_nested_miller_3stage_tcfc,
        testbench=build_ac_testbench,
    )
    # v0.1.07 Round 83 Agent D: TCFC + cascoded PMOS input pair. R82 D
    # landed LUT-sized plain TCFC at GBW=0.973 MHz (paper threshold +/-20%
    # PASS) but gain_dc=88.5 dB << paper 130 dB headline. Stage-1 gain is
    # structurally bounded by gm_pmos*(ro_pmos||ro_nmos) ~30-35 dB on
    # sky130 1v8 — sizing alone cannot close the gap. This variant adds
    # PMOS cascode pair (Mcas1/Mcas2) between input pair drains and
    # mirror summing nodes, recovering textbook cascode gain product
    # +15-25 dB at stage 1. All other TCFC architectural devices (Mgmf
    # active FF, Mcas_n cascoded NMOS sink, nd3, Mlcs3_diode chain) are
    # preserved verbatim. See clay/templates/peng_tcfc_cascoded_input.py.
    register_stage(
        "nested_miller_3stage_tcfc_cascoded_input",
        build_nested_miller_3stage_tcfc_cascoded_input,
        testbench=build_ac_testbench,
    )
    # v0.1.07 Round 84 Agent D: TCFC + cascoded PMOS input + cascoded NMOS
    # mirror load. R83 D landed cascoded-input TCFC at gain_dc=104.3 dB
    # / GBW=1.115 MHz (BL-093 cont PASS) but ~26 dB short of paper 130 dB.
    # The remaining gap is dominated by the NMOS mirror's finite ro_n on
    # M3/M4. R84 D stacks a symmetric NMOS cascode pair (Mcas3_n/Mcas4_n)
    # on the mirror legs with an ncas bias rail, recovering the textbook
    # cascode gain product gm_n_cas * ro_n_cas at the mirror side (+15-20
    # dB). Combined with the R83 D PMOS cascode, stage 1 now has a
    # double-cascode gain product matching Peng 2004's reference netlist
    # pattern. See clay/templates/peng_tcfc_cascoded_input_mirror.py.
    register_stage(
        "nested_miller_3stage_tcfc_cascoded_input_mirror",
        build_nested_miller_3stage_tcfc_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    # v0.0.57: ACBC (active capacitance + bias compensation) and DFCFC2
    # (dual feedforward + damping path). Both extend the NMCF baseline
    # with structurally distinct compensation:
    #   - ACBC: 4th active gm-cell Mgm4 (NMOS, g=d2, d=nb) + bias-comp
    #     chain (Mnb_diode + Mnb_drv) creating internal node `nb`;
    #     active feedforward Mgf2 (PMOS, g=d1, d=out) shared with AFFC.
    #   - DFCFC2: dual feedforward (Mgf2 to out + Mgf_damp to nd_damp)
    #     plus damping cap Cm2_damp at the new internal node nd_damp,
    #     active sink Mlcs_damp.
    # See clay/templates/peng_acbc.py and clay/templates/leung_dfcfc2.py
    # for the full topology descriptions.
    register_stage(
        "nested_miller_3stage_acbc",
        build_nested_miller_3stage_acbc,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_dfcfc2",
        build_nested_miller_3stage_dfcfc2,
        testbench=build_ac_testbench,
    )
    # v0.1.08 Round 85 Agent C: DFCFC2 + cascoded PMOS input + cascoded
    # NMOS mirror load. Cross-variant generalisation of R84 D's TCFC
    # cascode recipe (telescopic input cascode + DFCFC1-style regulated
    # NMOS mirror at high-Z node d1). DFCFC2 has dual active feedforwards
    # (Mgf2 + Mgf_damp) and a damping node nd_damp — structurally
    # distinct from TCFC's Mgmf single-FF + cascoded output sink. Target
    # gain_dc >= 120 dB (paper - 10 dB) while keeping GBW in DFCFC2's
    # paper +/-20% window [3.2 MHz, 4.8 MHz]. See
    # clay/templates/dfcfc2_cascoded_input_mirror.py.
    register_stage(
        "nested_miller_3stage_dfcfc2_cascoded_input_mirror",
        build_nested_miller_3stage_dfcfc2_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    # v0.1.09 Round 86 Agent D: 5 NMCF variants with R85 C cascode recipe
    # applied as a batch rollout. All five (ACBC / IAC / RAFFC / LEC / PFC)
    # have plain NMCF stage 1 — the R85 C cascode recipe (telescopic PMOS
    # input cascode + cascoded NMOS mirror with DFCFC1-style diode loop at
    # high-Z node d1) drops in verbatim. Each variant's architectural
    # signature (compensation network, active FFs, bias-comp nodes) is
    # preserved verbatim at stages 2-3 + compensation. Target gain_dc >=
    # paper - 10 dB AND GBW in paper +/-20% window. BL-093 batch test
    # M-SIZING-PAPER-GRADE 4/17 → 7-9/17 (depending on 3/5 or 5/5 PASS).
    # See clay/templates/<variant>_cascoded_input_mirror.py for full
    # topology descriptions per variant.
    register_stage(
        "nested_miller_3stage_acbc_cascoded_input_mirror",
        build_nested_miller_3stage_acbc_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_iac_cascoded_input_mirror",
        build_nested_miller_3stage_iac_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_raffc_cascoded_input_mirror",
        build_nested_miller_3stage_raffc_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_lec_cascoded_input_mirror",
        build_nested_miller_3stage_lec_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_pfc_cascoded_input_mirror",
        build_nested_miller_3stage_pfc_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    # v0.1.10 Round 87 Agent C: 5 more NMCF variants applying R85 C
    # cascode recipe as a batch rollout (BL-093 cont). All five
    # (DFCFC1 / NMCNR / AZ / AZC / DACFC) have plain NMCF stage 1 —
    # the R85 C cascode recipe drops in verbatim. Each variant's
    # architectural signature (compensation network, active FFs,
    # bias-comp nodes) is preserved verbatim at stages 2-3 +
    # compensation. Target gain_dc >= paper - 10 dB AND GBW in
    # paper +/-20% window. BL-093 batch test M-SIZING-PAPER-GRADE
    # 7/17 → 10/17 (if 3/5 PASS).
    # NB: DFCFC1's own internal cascoded mirror (Mcas1_n_load /
    # Mcas1_n_diode + vcas rail) is replaced by the R85 C recipe
    # (canonical node naming + explicit stacked ncas bias rail).
    # See clay/templates/<variant>_cascoded_input_mirror.py for
    # full topology descriptions per variant.
    register_stage(
        "nested_miller_3stage_dfcfc1_cascoded_input_mirror",
        build_nested_miller_3stage_dfcfc1_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_nmcnr_cascoded_input_mirror",
        build_nested_miller_3stage_nmcnr_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_az_cascoded_input_mirror",
        build_nested_miller_3stage_az_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_azc_cascoded_input_mirror",
        build_nested_miller_3stage_azc_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_dacfc_cascoded_input_mirror",
        build_nested_miller_3stage_dacfc_cascoded_input_mirror,
        testbench=build_ac_testbench,
    )
    # v0.1.11 Round 88 Agent C: NMCNR hybrid d1-gated FF retrofit pilot.
    # BL-093 R87 C closure surfaced a structural finding: cascoded mirror
    # recipe (R85 C) REQUIRES a d1-gated Mgf2-style active feedforward
    # companion to realise its +30-35 dB gain product. PASS variants
    # (DFCFC2/ACBC/LEC/PFC/DFCFC1) ALL have a d1-gated FF; FAIL variants
    # (NMCNR/AZ/AZC/DACFC/RAFFC) have NONE. R88 C pilot retrofits NMCNR
    # by grafting Mgf2_hybrid (PMOS, g=d1, d=out) onto its nulling-R +
    # 2-Miller-cap-to-rz_n backbone. R87 C NMCNR baseline 83.29 dB →
    # expected 115-120 dB (PASS validates structural finding, unlocks
    # AZ / AZC / DACFC retrofit in R89+).
    register_stage(
        "nested_miller_3stage_nmcnr_hybrid_d1_ff",
        build_nested_miller_3stage_nmcnr_hybrid_d1_ff,
        testbench=build_ac_testbench,
    )
    # v0.1.12 Round 89 Agent B: AZ / AZC / DACFC hybrid d1-gated FF
    # retrofit batch. BL-093 batch rollout of R88 C-validated retrofit
    # pattern. AZ + DACFC dual-FF (preserve Mgmf2 g=d2 + add Mgf2_hybrid
    # g=d1). AZC direct graft (no original FF, just add Mgf2_hybrid).
    # Acceptance: >= 2/3 PASS to advance M-SIZING-PAPER-GRADE 9/17 ->
    # 11/17 or 12/17.
    register_stage(
        "nested_miller_3stage_az_hybrid_d1_ff",
        build_nested_miller_3stage_az_hybrid_d1_ff,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_azc_hybrid_d1_ff",
        build_nested_miller_3stage_azc_hybrid_d1_ff,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_dacfc_hybrid_d1_ff",
        build_nested_miller_3stage_dacfc_hybrid_d1_ff,
        testbench=build_ac_testbench,
    )
    # v0.1.13 Round 90 Agent C: AZC GBW co-redesign via Cazc4 Miller
    # path. Additive variant of nested_miller_3stage_azc_hybrid_d1_ff
    # with one new device (Cazc4 cap, nazc1 <-> out). All other devices
    # byte-exact v127.
    register_stage(
        "nested_miller_3stage_azc_hybrid_d1_ff_cazc4",
        build_nested_miller_3stage_azc_hybrid_d1_ff_cazc4,
        testbench=build_ac_testbench,
    )
    # v0.1.14 Round 91 Agent C: DACFC GBW co-redesign via Mgmf1 drain
    # re-route. Additive variant of nested_miller_3stage_dacfc_hybrid_d1_ff
    # with Mgmf1_n/p drains moved from nga1_a/nga1_b to out, plus Mlcs3
    # widened to absorb the additional PMOS source currents at out.
    register_stage(
        "nested_miller_3stage_dacfc_gbw_corouted",
        build_nested_miller_3stage_dacfc_gbw_corouted,
        testbench=build_ac_testbench,
    )
    # v0.1.15 Round 92 Agent D: DACFC cascode-stack widening for
    # 12/17 closure. Additive variant of nested_miller_3stage_dacfc_hybrid_d1_ff
    # with W_cas_n_mirror 8 -> 80 um (10x) and W_cas_n_bias 4 -> 40 um
    # (10x). The widening lowers effective output impedance of the
    # cascoded mirror, pulling dominant pole from ~5.5 MHz into the
    # paper [1.6, 2.4] MHz window. Mgf2_scale PRESERVED at v127
    # default 0.14 (reduction below 0.13 hits a gain cliff that was
    # empirically falsified during R92 D sweep). All v127 connectivity
    # preserved (Mgmf1 drains stay at nga1_a/nga1_b). Measured:
    # gain=118.345 dB PASS / GBW=2.025 MHz PASS.
    register_stage(
        "nested_miller_3stage_dacfc_mgf2_reduced",
        build_nested_miller_3stage_dacfc_mgf2_reduced,
        testbench=build_ac_testbench,
    )
    # R93 C / BL-093 RAFFC redesign for M-SIZING-PAPER-GRADE 13/17.
    # Inherits R86 D v124 cascode recipe verbatim; R93 C deltas are
    # K_cs3_load = 1 (Mlcs3 minimum), W_cs3 = 98 um sweet spot,
    # Mgf_rev_scale = 1e-4 + W_gf_rev = 0.42 um sky130 min + L_gf_rev
    # = 10 um long-channel. The combined effect raises gain
    # 26.99 dB -> 125.37 dB while leaving GBW at 1.330 MHz (paper
    # window [0.96, 1.44] MHz). RAFFC architectural signature
    # (Mgf_rev PMOS g=out, d=d1) PRESERVED -- only sizing changed.
    register_stage(
        "nested_miller_3stage_raffc_redesign",
        build_nested_miller_3stage_raffc_redesign,
        testbench=build_ac_testbench,
    )
    # R94 C / BL-093 IAC redesign for M-SIZING-PAPER-GRADE 14/17.
    # Inherits R86 D v124 cascode recipe and signal-path widths
    # verbatim; R94 C single delta is W_cas_n_mirror 8 -> 20 um (2.5x)
    # and W_cas_n_bias 4 -> 10 um (2.5x). The widened cascode reduces
    # effective output impedance of the cascoded mirror, pulling the
    # dominant pole from 4.73 MHz into the paper [2.80, 4.20] MHz
    # window at 3.47 MHz (0.75% off paper target 3.50 MHz). Gain
    # already passed at R86 D baseline (119.5 dB); only GBW needed
    # repair. IAC architectural signature (Mgmt triode helper + niac/
    # npath/vcas_iac/n_rc_iac nodes + Cm2_iac/R0_iac RC chain + Mgmf
    # active FF) PRESERVED verbatim -- only sizing changed.
    register_stage(
        "nested_miller_3stage_iac_redesign",
        build_nested_miller_3stage_iac_redesign,
        testbench=build_ac_testbench,
    )
    # R105 D / BL-093 v200 DACFC GBW co-redesign vs. cascode-widening
    # composition. Ships R91 C Mgmf1 -> out drain re-route as the
    # default topology, layered on R92 D cascode-widening defaults
    # (W_cas_n_mirror=80, W_cas_n_bias=40). Tests whether R91 C's gain-
    # crash mechanism survives R92 D's GBW lever; predicted FAIL at
    # Mgmf1_co_scale=1.0 (gain crushed), PASS at Mgmf1_co_scale=0 +
    # cascode widening (gain restored + GBW pulled in by R92 D mechanism).
    # M-SIZING-PAPER-GRADE already DONE at 14/17 via R94 close; v200 is
    # a 2nd falsification record on the routing-vs-sizing decomposition.
    from .templates.dacfc_gbw_integer_scaling import (
        build_nested_miller_3stage_dacfc_gbw_v200,
    )
    register_stage(
        "nested_miller_3stage_dacfc_gbw_v200",
        build_nested_miller_3stage_dacfc_gbw_v200,
        testbench=build_ac_testbench,
    )
    # R107 Agent D v201: falsification variant of v200 that switches the
    # Mlcs3 output-sink sizing from integer-rounded M to FLOATING-POINT
    # fractional W. Tests R106 C's hypothesis that the v200 cliff at
    # Mgmf1_co_scale = (0.825, 0.850] is caused by the
    # `int(round(K_cs3_load_total))` 5->6 step at the half-step boundary.
    # If v201's continuous-scaling sweep shows smooth degradation,
    # mechanism CONFIRMED; if cliff persists, deeper structure flagged.
    from .templates.dacfc_gbw_continuous_scaling import (
        build_nested_miller_3stage_dacfc_gbw_v201,
    )
    register_stage(
        "nested_miller_3stage_dacfc_gbw_v201",
        build_nested_miller_3stage_dacfc_gbw_v201,
        testbench=build_ac_testbench,
    )
    # v0.0.59: DFCFC1 (single FF + cascoded stage-1 mirror), RAFFC
    # (reverse-direction active FF), LEC (local embedded compensation
    # with R-C-to-vss). All three extend the NMCF baseline with
    # structurally distinct compensation architectures:
    #   - DFCFC1: Mgf2 (shared with AFFC) + cascoded NMOS mirror at
    #     stage 1 (Mcas1_n_load + Mcas1_n_diode + Mcas_bias + vcas
    #     rail), creating internal cascode-source nodes d1_int /
    #     d1_diode_int. NMCF-style Cm1 routing (out↔d2) preserved.
    #   - RAFFC: Mgf_rev (PMOS, g=d2, d=d1) — reverse-direction active
    #     FF vs AFFC's forward (g=d1, d=out). NMCF-style Cm1 routing
    #     (out↔d2) preserved; Cf dropped.
    #   - LEC: local error amplifier Mlea (PMOS, g=d1, d=nlec) +
    #     local buffer Mleb (NMOS, g=nlec, d=nleb) + local R-C filter
    #     (Clec_gnd cap to vss + Rlec res nlec↔nleb). Stage 2 gate
    #     re-routed from d1 to nleb. Cm1 skip-over routing (out↔d1);
    #     Cm2 + Cf dropped.
    # See clay/templates/leung_dfcfc1.py, alfio_raffc.py, qu_lec.py
    # for the full topology descriptions.
    register_stage(
        "nested_miller_3stage_dfcfc1",
        build_nested_miller_3stage_dfcfc1,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_raffc",
        build_nested_miller_3stage_raffc,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_lec",
        build_nested_miller_3stage_lec,
        testbench=build_ac_testbench,
    )
    # v0.0.60: PFC (Ramos pole-zero following compensation), AZC
    # (Qu2017 auto-zero compensation with triple Miller + quad nulling-R
    # mesh), DACFC (Song doubly-differential active-cascode FF). All
    # three extend the NMCF baseline with structurally distinct
    # compensation architectures:
    #   - PFC: oversized Mgmf1 (PMOS, g=d1, d=out, W = 4 × Mcs3 by
    #     default) — same routing as AFFC's Mgf2 but oversized so the
    #     LHP zero auto-follows the non-dominant pole across PVT.
    #   - AZC: triple AZC MOSFET archetype (Mazc1 NMOS + Mazc2_diode
    #     PMOS + Mazc3 NMOS) + quad nulling-R mesh (Razc0-Razc3 with
    #     two hub nodes nazc_bias / nazc_hub) + triple compensation
    #     cap network (Cazc0 to out + Cazc1/Cazc2 to vss). Replaces
    #     NMCF's Cm1/Cm2/Cf entirely.
    #   - DACFC: matched gma1 PMOS pair (Mga1_a + Mga1_b, g=bias_d2)
    #     PLUS input-coupled gmf1_PMOS pair (Mgmf1_n / Mgmf1_p, gated
    #     directly by inn / inp) PLUS single gma2 buffer PLUS single
    #     Mgmf2 (g=d2, d=out). Five FF devices total. Cm2_dac (nga1_a
    #     ↔ vss) replaces NMCF's Cm2.
    # See clay/templates/ramos_pfc.py, qu2017_azc.py, song_dacfc.py
    # for the full topology descriptions.
    register_stage(
        "nested_miller_3stage_pfc",
        build_nested_miller_3stage_pfc,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_azc",
        build_nested_miller_3stage_azc,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_dacfc",
        build_nested_miller_3stage_dacfc,
        testbench=build_ac_testbench,
    )
    # v0.0.61: IAC (Peng indirect AC compensation via triode-region PMOS
    # helper Mgmt + active FF + series RC chain to vss), CLIA (Tan
    # capacitor-less indirect amplifier — gm21 PMOS + gm23 NMOS mirror
    # pair forming a current-mode local impedance loop, NO Miller cap),
    # AZ (Yan auto-zero — simpler subset of AZC with Mgmb1/Mgmb2 pair +
    # single nulling-R + single auto-zero cap in series with Miller).
    # All three close the cohort to 16/17 distinct paper variants:
    #   - IAC: Mgmt (PMOS, d=niac, g=vcas_iac, s=npath) triode helper +
    #     Mgmf (PMOS, g=d1, d=out) active FF + Cm1_iac out↔d2 outer
    #     Miller + Cm2_iac niac↔n_rc_iac indirect cap + R0_iac
    #     n_rc_iac↔vss nulling.
    #   - CLIA: Mgm21_PMOS (g=d2) + Mgm23_NMOS_a (diode at ngm21) +
    #     Mgm23_NMOS_b (mirror) + Mgmf (PMOS, g=d2, d=out) + Cclia_gnd
    #     cap-to-vss (NOT Miller position) + Rclia nulling. NO Cm1, NO
    #     Cm2, NO Cf — literally capacitor-less in Miller position.
    #   - AZ: Mgmb1 (d=naz_drv, g=naz_cap) + Mgmb2 (d=naz_pair,
    #     g=naz_drv) + Mgmf2 (PMOS, g=d2, d=out) + Caz_outer out↔d2 +
    #     Caz naz_cap↔naz_pair (series with Miller) + Raz nulling.
    # See clay/templates/peng_iac.py, tan_clia.py, yan_az.py for the full
    # topology descriptions.
    register_stage(
        "nested_miller_3stage_iac",
        build_nested_miller_3stage_iac,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_clia",
        build_nested_miller_3stage_clia,
        testbench=build_ac_testbench,
    )
    register_stage(
        "nested_miller_3stage_az",
        build_nested_miller_3stage_az,
        testbench=build_ac_testbench,
    )
    # v0.0.92 Round 68 Agent R: 3 stdlib opamps that USE the new
    # third-batch functional-role keywords (common_mode_feedback,
    # sample_hold_switch, regulated_cascode, gain_boost_amp,
    # output_class_AB, dynamic_bias) as primary architectural features.
    # Each cell anchors ≥ 3 of the 6 third-batch keywords on real
    # devices in the template — so the v0.0.92 invariant runtime has
    # actual cells to check declared targets against.
    register_stage(
        "regulated_cascode_opamp",
        build_regulated_cascode_opamp,
        testbench=build_ac_testbench,
    )
    register_stage(
        "class_AB_output_opamp",
        build_class_AB_output_opamp,
        testbench=build_ac_testbench,
    )
    register_stage(
        "dynamic_bias_opamp",
        build_dynamic_bias_opamp,
        testbench=build_ac_testbench,
    )
    # v0.0.56: cross-coupled NMOS LC-tank VCO. First stage template that
    # uses the `ind` device primitive (shipped in v0.0.55) end-to-end
    # through a real circuit topology.
    #
    # v0.0.57: register the dedicated transient testbench
    # (build_lc_tank_vco_testbench). The testbench wires VDD / VSS /
    # VCTRL supplies plus a short current-pulse kick on `clk_out` that
    # breaks the symmetric DC operating point, so transient observes
    # real oscillation rather than a flat DC line. Port surface is
    # `(clk_out, vctrl, vdd, vss)` — same as a ring VCO, so the CLI's
    # PLL-shape heuristic (which requires `ref_in`) does NOT fire on
    # this cell. A future PLL composition that wraps the LC-VCO inside
    # a `ref_in`-bearing cell would route through `build_pll_testbench`
    # instead.
    register_stage(
        "lc_tank_vco",
        build_lc_tank_vco,
        testbench=build_lc_tank_vco_testbench,
    )

    _bootstrap_builtin_interfaces()


# ─── StageInterface declarations (v0.0.86, P2 first slice) ───
#
# Surface-level typed declarations for 6 representative stage_templates.
# These do NOT alter the underlying builder semantics; they merely
# declare the input/output/bias/supply port structure + KPI promises
# in a form the future combinator runtime can type-check.
#
# Scope of this slice: 2-stage Miller (PMOS + NMOS inputs), folded
# cascode, telescopic cascode, NMCF feedforward, and AFFC. Every
# remaining stage_template (NMC subset closed in v0.0.61, LC-tank VCO,
# common-source, fully-differential, etc.) will land interfaces in
# subsequent rounds once the combinator surface stabilises.

def _bootstrap_builtin_interfaces() -> None:
    """Register StageInterfaces for the v0.0.86 first-slice cohort.

    Every stage in this cohort happens to share the canonical
    diff-input + single-ended-output opamp surface
    (`inn`, `inp`, `out`, `vdd`, `vss`). The shared shape is captured
    in `_DIFF_OPAMP_PORTS` so adding a new stage to the cohort is a
    one-line change.

    KPI promises (`spec_targets`) are the LOWER-BOUND envelope the
    builder's gm/Id-intent surface can hit on a 1.8 V sky130 design
    when its tail/load multipliers are routed sanely. The combinator
    runtime (future) uses these to triage obviously-wrong asks before
    materialising a sim.

    `compatible_compensation` is the list of compensation strategies
    the template ALREADY EMBEDS — it's metadata for the combinator,
    not a runtime config. Stages with built-in Miller compensation
    advertise `'miller'`; the 3-stage NMCF variants advertise
    `'nested_miller'` + the variant-specific tag.
    """
    diff_in = [
        PortDecl(name="inn", role="signal_in",
                 polarity="differential_neg"),
        PortDecl(name="inp", role="signal_in",
                 polarity="differential_pos"),
    ]
    se_out = [
        PortDecl(name="out", role="signal_out", polarity="single_ended"),
    ]
    supplies = [
        PortDecl(name="vdd", role="supply", polarity="none"),
        PortDecl(name="vss", role="supply", polarity="none"),
    ]

    # ── two_stage_miller_pmos ─────────────────────────────────
    register_stage_interface(StageInterface(
        template_name="two_stage_miller_pmos",
        input_ports=list(diff_in),
        output_ports=list(se_out),
        bias_ports=[],          # all bias internal; pbias/nbias not on surface
        supply_ports=list(supplies),
        spec_targets={
            "gain_dc": SpecTarget("gain_dc", "db", ">=", 25.0),
            "gbw":     SpecTarget("gbw",     "freq", ">=", 1e6),
            "pm":      SpecTarget("pm",      "angle", ">=", 45.0),
        },
        compatible_compensation=["miller"],
    ))

    # ── two_stage_miller_nmos ─────────────────────────────────
    register_stage_interface(StageInterface(
        template_name="two_stage_miller_nmos",
        input_ports=list(diff_in),
        output_ports=list(se_out),
        bias_ports=[],
        supply_ports=list(supplies),
        spec_targets={
            "gain_dc": SpecTarget("gain_dc", "db", ">=", 25.0),
            "gbw":     SpecTarget("gbw",     "freq", ">=", 1e6),
            "pm":      SpecTarget("pm",      "angle", ">=", 45.0),
        },
        compatible_compensation=["miller"],
    ))

    # ── folded_cascode ────────────────────────────────────────
    # Single-stage opamp with cascoded load. No internal Miller; combinator
    # users typically wrap it with an explicit second stage if higher
    # gain is required.
    register_stage_interface(StageInterface(
        template_name="folded_cascode",
        input_ports=list(diff_in),
        output_ports=list(se_out),
        bias_ports=[],
        supply_ports=list(supplies),
        spec_targets={
            "gain_dc": SpecTarget("gain_dc", "db", ">=", 40.0),
            "gbw":     SpecTarget("gbw",     "freq", ">=", 5e5),
            "pm":      SpecTarget("pm",      "angle", ">=", 60.0),
        },
        compatible_compensation=[],   # no built-in compensation
    ))

    # ── telescopic_cascode ────────────────────────────────────
    register_stage_interface(StageInterface(
        template_name="telescopic_cascode",
        input_ports=list(diff_in),
        output_ports=list(se_out),
        bias_ports=[],
        supply_ports=list(supplies),
        spec_targets={
            "gain_dc": SpecTarget("gain_dc", "db", ">=", 50.0),
            "gbw":     SpecTarget("gbw",     "freq", ">=", 5e5),
            "pm":      SpecTarget("pm",      "angle", ">=", 60.0),
        },
        compatible_compensation=[],
    ))

    # ── nested_miller_3stage_feedforward (canonical NMCF) ─────
    register_stage_interface(StageInterface(
        template_name="nested_miller_3stage_feedforward",
        input_ports=list(diff_in),
        output_ports=list(se_out),
        bias_ports=[],
        supply_ports=list(supplies),
        spec_targets={
            "gain_dc": SpecTarget("gain_dc", "db", ">=", 80.0),
            "gbw":     SpecTarget("gbw",     "freq", ">=", 1e6),
            "pm":      SpecTarget("pm",      "angle", ">=", 45.0),
        },
        compatible_compensation=["nested_miller", "feedforward"],
    ))

    # ── nested_miller_3stage_affc (Ho-Lee AFFC) ───────────────
    register_stage_interface(StageInterface(
        template_name="nested_miller_3stage_affc",
        input_ports=list(diff_in),
        output_ports=list(se_out),
        bias_ports=[],
        supply_ports=list(supplies),
        spec_targets={
            "gain_dc": SpecTarget("gain_dc", "db", ">=", 80.0),
            "gbw":     SpecTarget("gbw",     "freq", ">=", 1e6),
            "pm":      SpecTarget("pm",      "angle", ">=", 45.0),
        },
        compatible_compensation=["nested_miller", "active_feedforward"],
    ))


_bootstrap_builtin()
