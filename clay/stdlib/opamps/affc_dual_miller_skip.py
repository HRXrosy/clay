"""AFFCDualMillerSkip — AFFC opamp with dual skip-over Miller caps.

Combines Hoi-Lee 2003's AFFC active feedforward backbone (gm_f2
device, see `clay.templates.hoilee_affc`) with Fan 2005's
skip-over Miller-cap routing concept (see `clay.templates.fan_smc`)
generalised to TWO independent skip caps. Both Miller caps share an
endpoint at d1 (stage-1 output) — the natural high-impedance node
for maximum Miller multiplication — and route to different downstream
nodes (out / d2). The active feedforward device Mgf2 is retained
verbatim from AFFC.

Same 5-port surface as the AFFC baseline; the cap renaming
(Cm1_skip / Cm2_skip) is internal. See
`clay.templates.affc_dual_miller_skip` for the full topology
description and sizing knobs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.affc_dual_miller_skip import (
    build_nested_miller_3stage_affc_dual_miller_skip,
)
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("AFFCDualMillerSkip", testbench=build_ac_testbench)
def AFFCDualMillerSkip(spec: Spec, **kwargs) -> Cell:
    """Build a Cell IR for an AFFC opamp with dual skip-over Miller caps.

    kwargs are forwarded to the template builder. In addition to all
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load,
    gm_id_diff / load1 / stage2 / stage3) and the AFFC knobs
    (Mgf2_scale, W_gf2, L_gf2, gm_id_gf2), this cell accepts:

      Cm1_skip: float = 1.5e-12    — skip cap A (d1 ↔ out)
      Cm2_skip: float = 0.5e-12    — skip cap B (d1 ↔ d2)

    Note: Cm1, Cm2, Cf kwargs from the AFFC baseline are NOT accepted
    here. The compensation network is structurally different — both
    Miller caps share the d1 endpoint and use distinct instance names
    (Cm1_skip / Cm2_skip) to disambiguate the cohort.
    """
    return build_nested_miller_3stage_affc_dual_miller_skip(
        cell_name=kwargs.pop("cell_name", "AFFCDualMillerSkip"),
        spec=spec,
        **kwargs,
    )
