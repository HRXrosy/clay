"""HoiLeeAFFC — 3-stage NMCF opamp with active gm_f2 feedforward.

HoiLee 2003 Active FeedForward Compensation (AFFC) topology. Extends
the textbook NMCF (`nested_miller_3stage_feedforward`) with an extra
active PMOS device `Mgf2` whose gate is connected to the stage-1
output and whose drain feeds the output node — bypassing stage 2 and
creating an LHP zero in the loop transfer function that is distinct
from the passive Cf cap.

Same 5-port surface as the NMCF baseline; the additional feedforward
device and rebalanced stage-3 sink are internal. See
`clay.templates.hoilee_affc` for the full topology and bias plan.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.hoilee_affc import build_nested_miller_3stage_affc
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("HoiLeeAFFC", testbench=build_ac_testbench)
def HoiLeeAFFC(spec: Spec, **kwargs) -> Cell:
    """Build a HoiLee AFFC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    Cm2, Cf, gm_id_diff / load1 / stage2 / stage3), the AFFC cell
    accepts three new knobs for the feedforward device:

      Mgf2_scale: float = 1.0   — ratio of Mgf2 W to Mcs3 W
      W_gf2, L_gf2: optional explicit overrides
      gm_id_gf2: gm/Id intent override for Mgf2
    """
    return build_nested_miller_3stage_affc(
        cell_name=kwargs.pop("cell_name", "HoiLeeAFFC"),
        spec=spec,
        **kwargs,
    )
