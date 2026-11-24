"""TelescopicCascode — single-stage telescopic cascode opamp (NMOS input)."""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.telescopic_cascode import build_telescopic_cascode
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("TelescopicCascode", testbench=build_ac_testbench)
def TelescopicCascode(spec: Spec, **kwargs) -> Cell:
    """Build a telescopic cascode opamp Cell IR.

    kwargs are forwarded to the template builder for sizing overrides:
    I_ref, W_diff, L_diff, W_nc, L_nc, W_pc, L_pc, W_load, L_load,
    W_tail, L_tail, M_branch, M_tail, and the W_b* / L_b* bias-chain
    unit sizes.
    """
    return build_telescopic_cascode(
        cell_name=kwargs.pop("cell_name", "TelescopicCascode"),
        spec=spec,
        **kwargs,
    )
