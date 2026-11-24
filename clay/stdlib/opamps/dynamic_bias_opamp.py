"""DynamicBiasOpamp — 2-stage Miller opamp with beta-multiplier dynamic bias + S/H.

Two-stage Miller-compensated opamp whose tail and stage-2 load currents
are derived from a beta-multiplier dynamic-bias rail (PTAT current via
Mbm1/Mbm2 + Rbm). A sample/hold switch (Msh) and hold cap (Csh) sit on
the bias-storage node, demonstrating a switched-bias dynamic-bias use
case.

Architectural roles available in the .clay surface:
  - dynamic_bias        Idyn      (beta-multiplier output ndyn)
  - sample_hold_switch  Msh       (NMOS S/H gate switch on phi1)
  - tail_current_source Mtail     (opamp tail, legacy keyword)
  - dominant_pole_cap   Cc        (Miller cap, legacy keyword)
  - input_transconductor Mdp      (diff pair, legacy keyword)

See `clay.templates.dynamic_bias_opamp` for the full topology description
and sizing knobs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.dynamic_bias_opamp import (
    build_dynamic_bias_opamp,
)
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("DynamicBiasOpamp", testbench=build_ac_testbench)
def DynamicBiasOpamp(spec: Spec, **kwargs) -> Cell:
    """Build a Cell IR for a 2-stage Miller opamp with beta-multiplier
    dynamic bias rail and sample/hold switch.

    kwargs are forwarded to the template builder for sizing overrides:
    I_ref, Rbm, W_bm1, L_bm1, W_bm2, L_bm2, W_sh, L_sh, Csh, W_bp, L_bp,
    W_bn, L_bn, N_tail, K_cs_load, W_diff, L_diff, W_load1, L_load1,
    gm_id_diff, gm_id_load1, W_cs, L_cs, gm_id_cs, Cc.
    """
    return build_dynamic_bias_opamp(
        cell_name=kwargs.pop("cell_name", "DynamicBiasOpamp"),
        spec=spec,
        **kwargs,
    )
