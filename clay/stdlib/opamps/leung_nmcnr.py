"""LeungNMCNR — 3-stage opamp with Miller cap + nulling resistor.

Leung Nested Miller compensation with Nulling Resistor (NMCNR). Same
signal path as the textbook NMCF (`nested_miller_3stage_feedforward`)
but the compensation network adds a series nulling resistor `Rz1`
between a shared Miller-cap node `rz_n` and the output. Both Miller
caps Cm1 (outer) and Cm2 (inner) terminate at `rz_n`; the feedforward
cap Cf is removed because the nulling-R handles RHP-zero cancellation
through a different mechanism.

Same 5-port surface as the NMCF baseline; the rz_n node + Rz1
resistor are internal. See `clay.templates.leung_nmcnr` for the full
topology description and the Rz1 / Cm1 / Cm2 sizing knobs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.leung_nmcnr import build_nested_miller_3stage_nmcnr
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("LeungNMCNR", testbench=build_ac_testbench)
def LeungNMCNR(spec: Spec, **kwargs) -> Cell:
    """Build a Leung NMCNR 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    Cm2, gm_id_diff / load1 / stage2 / stage3), the NMCNR cell accepts
    one new knob:

      Rz1: float = 5e3   — nulling resistance in series with Cm1 (Ω)

    Note: Cf kwarg from the NMCF baseline is NOT accepted here because
    the NMCNR topology does not contain Cf (the nulling-R handles
    RHP-zero cancellation instead).
    """
    return build_nested_miller_3stage_nmcnr(
        cell_name=kwargs.pop("cell_name", "LeungNMCNR"),
        spec=spec,
        **kwargs,
    )
