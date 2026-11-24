"""LcTankVCO — cross-coupled NMOS LC-tank voltage-controlled oscillator.

v0.0.56's headline cell — the first Clay stdlib cell built on the `ind`
device primitive (v0.0.55). Used as a drop-in replacement for the
ring oscillator inside the AnalogGym PLL benchmark when the target
spec calls for an LC-tank rather than a current-starved ring VCO.

Port list: (clk_out, vctrl, vdd, vss). The clk_out port carries one
phase of the differential tank (outp); the opposite phase (outn) is an
internal net. A future fully-differential PLL wrapper can expose both
phases as outer ports.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.lc_tank_vco import build_lc_tank_vco


@register_cell("LcTankVCO")
def LcTankVCO(spec: Spec, **kwargs) -> Cell:
    """Build a cross-coupled NMOS LC-tank VCO Cell IR.

    No `testbench=` parameter is passed to @register_cell because the
    natural rig for this cell is the PLL transient testbench
    (`clay.templates.pll_testbench.build_pll_testbench`), which the CLI
    selects via the PLL-shape heuristic on the cell's port list rather
    than via stdlib's per-cell testbench registry. That keeps the PLL
    testbench attached to the (clk_out, vctrl, ...) port surface
    uniformly across the LC and ring variants.

    kwargs are forwarded to the template builder for sizing overrides:
      L_tank, C_tank, R_tank_series, W_osc, L_osc, W_tail, L_tail,
      I_bias, pdk_id.
    """
    return build_lc_tank_vco(
        cell_name=kwargs.pop("cell_name", "LcTankVCO"),
        spec=spec,
        **kwargs,
    )
