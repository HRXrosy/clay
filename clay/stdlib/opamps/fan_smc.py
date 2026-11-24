"""FanSMC — 3-stage opamp with single Miller compensation (Fan 2005).

Fan 2005 Single Miller Compensation (SMC) topology. Same signal path
as the textbook NMCF (`nested_miller_3stage_feedforward`) but the
compensation network is reduced to a single Miller cap Cm1 spanning
stage-1 output directly to stage-3 output (skip-over compensation).
The inner cap Cm2 and the feedforward cap Cf from NMCF are both
removed.

Same 5-port surface as the NMCF baseline; the compensation-network
delta is internal. See `clay.templates.fan_smc` for the full topology
description, the Fan 2005 transfer-function argument, and the
`gm2_boost` kwarg that keeps the single Miller cap's zero in LHP.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.fan_smc import build_nested_miller_3stage_smc
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("FanSMC", testbench=build_ac_testbench)
def FanSMC(spec: Spec, **kwargs) -> Cell:
    """Build a Fan SMC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    gm_id_diff / load1 / stage2 / stage3), the SMC cell accepts one
    new knob:

      gm2_boost: float = 1.5   — widen stage-2 NMOS CS by this factor
                                  to push gm_2 above gm_3 (Fan 2005
                                  LHP-zero constraint). 1.0 disables.

    Note: Cm2 and Cf kwargs from the NMCF baseline are NOT accepted
    here because the SMC topology has no Cm2 or Cf devices.
    """
    return build_nested_miller_3stage_smc(
        cell_name=kwargs.pop("cell_name", "FanSMC"),
        spec=spec,
        **kwargs,
    )
