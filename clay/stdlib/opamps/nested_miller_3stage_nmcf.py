"""NestedMiller3StageNMCF — 3-stage NMCF baseline (no feedforward cap).

The textbook Leung-Mok NMCF starting point: two nested Miller caps
Cm1 (outer) and Cm2 (inner), no Cf. This is the explicit baseline
against which feedforward / nulling-R / active-FF variants are
compared.

Same 5-port surface as the `_feedforward` variant; the absence of
Cf is the only topology delta. See
`clay.templates.nested_miller_3stage_nmcf` for the full topology.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.nested_miller_3stage_nmcf import (
    build_nested_miller_3stage_nmcf,
)
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("NestedMiller3StageNMCF", testbench=build_ac_testbench)
def NestedMiller3StageNMCF(spec: Spec, **kwargs) -> Cell:
    """Build a Cell IR for the NMCF baseline 3-stage opamp.

    kwargs are forwarded to the template builder. Same kwarg surface
    as the `_feedforward` variant EXCEPT `Cf` is not accepted (the
    NMCF baseline does not contain a feedforward cap).

    Accepted kwargs: I_ref, W_bp, L_bp, W_bn, L_bn, N_tail,
    K_cs3_load, W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2,
    W_cs3, L_cs3, gm_id_diff, gm_id_load1, gm_id_stage2,
    gm_id_stage3, Cm1, Cm2.
    """
    return build_nested_miller_3stage_nmcf(
        cell_name=kwargs.pop("cell_name", "NestedMiller3StageNMCF"),
        spec=spec,
        **kwargs,
    )
