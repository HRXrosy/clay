"""FoldedCascode — single-stage folded-cascode opamp with PMOS input pair."""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.folded_cascode import build_folded_cascode
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("FoldedCascode", testbench=build_ac_testbench)
def FoldedCascode(spec: Spec, **kwargs) -> Cell:
    """Build a single-stage folded-cascode opamp Cell IR.

    kwargs are forwarded to the template builder for sizing overrides:
    I_ref, W_diff, L_diff, W_load_p, L_load_p, W_load_p_cas, L_load_p_cas,
    W_cs_n, L_cs_n, W_bot_n, L_bot_n, N_tail, N_load, N_bot, etc.
    """
    return build_folded_cascode(
        cell_name=kwargs.pop("cell_name", "FoldedCascode"),
        spec=spec,
        **kwargs,
    )
