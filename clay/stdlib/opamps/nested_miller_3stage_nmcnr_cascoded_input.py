"""LeungNMCNRCascodedInput — NMCNR 3-stage opamp with cascoded input pair.

Combines Leung's NMCNR compensation (Cm1 + Cm2 sharing the shared
internal node rz_n + a nulling resistor Rz1 between rz_n and out)
with a cascoded PMOS input pair. The cascode devices Mcas1 / Mcas2
sit between the input pair drains and the NMOS mirror summing
nodes, lifting stage-1 gain by roughly the cascode product
``gm_pc · ro_pc`` while keeping the downstream signal path
unchanged.

Same 5-port surface as the plain NMCNR cell; the cascode pair +
pcas bias rail are internal. See
`clay.templates.nested_miller_3stage_nmcnr_cascoded_input` for the
full topology description and sizing knobs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.nested_miller_3stage_nmcnr_cascoded_input import (
    build_nested_miller_3stage_nmcnr_cascoded_input,
)
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("LeungNMCNRCascodedInput", testbench=build_ac_testbench)
def LeungNMCNRCascodedInput(spec: Spec, **kwargs) -> Cell:
    """Build a Cell IR for an NMCNR 3-stage opamp with cascoded input.

    kwargs are forwarded to the template builder. In addition to all
    plain NMCNR kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    Cm2, Rz1, gm_id_diff / load1 / stage2 / stage3), this cell
    accepts:

      W_cas: float = 20e-6       — cascode PMOS width (m)
      L_cas: float = 0.5e-6      — cascode PMOS length (m)
      I_ref_cas: float = 1e-6    — pcas-rail bias current (A)
    """
    return build_nested_miller_3stage_nmcnr_cascoded_input(
        cell_name=kwargs.pop("cell_name", "LeungNMCNRCascodedInput"),
        spec=spec,
        **kwargs,
    )
