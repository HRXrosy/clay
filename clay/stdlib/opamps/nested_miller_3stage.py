"""NestedMiller3StageFeedforward — 3-stage NMCF opamp with feedforward path.

Canonical AnalogGym NMCF benchmark topology. Same 5-port surface as the
2-stage Miller cells; the extra stage and three compensation caps are
internal. See `clay.templates.nested_miller_3stage` for the full
topology, bias plan, and compensation analysis.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.nested_miller_3stage import (
    build_nested_miller_3stage_feedforward,
)
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("NestedMiller3StageFeedforward", testbench=build_ac_testbench)
def NestedMiller3StageFeedforward(spec: Spec, **kwargs) -> Cell:
    """Build a 3-stage NMCF opamp Cell IR.

    kwargs are forwarded to the template builder for sizing overrides:
    I_ref, W_diff, L_diff, W_cs2, L_cs2, W_cs3, L_cs3, W_load1, L_load1,
    N_tail, K_cs2_load, K_cs3_load, Cm1, Cm2, Cf, and the gm_id_* intent
    knobs (gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3).
    """
    return build_nested_miller_3stage_feedforward(
        cell_name=kwargs.pop("cell_name", "NestedMiller3StageFeedforward"),
        spec=spec,
        **kwargs,
    )
