"""TwoStageMillerNMOS — 2-stage Miller-compensated opamp with NMOS input pair."""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.two_stage_miller_nmos import (
    build_ac_testbench, build_two_stage_miller_nmos,
)


@register_cell("TwoStageMillerNMOS", testbench=build_ac_testbench)
def TwoStageMillerNMOS(spec: Spec, **kwargs) -> Cell:
    """Build a 2-stage Miller NMOS-input opamp Cell IR.

    kwargs are forwarded to the template builder for sizing overrides:
    I_ref, W_diff, L_diff, W_load1, L_load1, W_cs, L_cs, Cc, K_cs_load, etc.
    """
    return build_two_stage_miller_nmos(
        cell_name=kwargs.pop("cell_name", "TwoStageMillerNMOS"),
        spec=spec,
        **kwargs,
    )
