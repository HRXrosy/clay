"""SauCFCC — 3-stage opamp with cascoded-Miller feedforward compensation.

Sau et al's Cascoded Feedforward Compensation (CFCC) topology. Same
signal path as the textbook NMCF (`nested_miller_3stage_feedforward`)
through stages 1 and 2, but stage 3 inserts a PMOS cascode between
the input transistor's drain and the output node, and the Miller cap
terminates at the cascode source (low-Z from the output side) rather
than spanning the full output stage.

Same 5-port surface as the NMCF baseline; the compensation-network +
stage-3 delta is internal. See `clay.templates.sau_cfcc` for the full
topology description, the cascode-buffering argument, and the
`cascode_gate` / `W_cas` / `L_cas` kwargs for cascode sizing.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.sau_cfcc import build_nested_miller_3stage_cfcc
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("SauCFCC", testbench=build_ac_testbench)
def SauCFCC(spec: Spec, **kwargs) -> Cell:
    """Build a Sau CFCC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    gm_id_diff / load1 / stage2 / stage3), the CFCC cell accepts three
    new knobs:

      W_cas, L_cas: float    — cascode PMOS sizing (default 30 µm / 0.5 µm)
      cascode_gate: str      — net name for cascode gate (default 'pbias')

    Note: Cm2 and Cf kwargs from the NMCF baseline are NOT accepted
    here because the CFCC topology has no Cm2 or Cf devices.
    """
    return build_nested_miller_3stage_cfcc(
        cell_name=kwargs.pop("cell_name", "SauCFCC"),
        spec=spec,
        **kwargs,
    )
