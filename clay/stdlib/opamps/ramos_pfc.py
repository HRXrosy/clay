"""RamosPFC — 3-stage opamp with pole-zero following compensation.

Ramos 2004 Pole-zero Following Compensation (PFC). Same signal
path as the textbook NMCF (`nested_miller_3stage_feedforward`) and
same compensation cap routing (Cm1 out↔d2, Cm2 d2↔d1), but the
active feedforward PMOS Mgmf1 is intentionally OVERSIZED (default
2× the stage-3 PMOS Mcs3 — the AnalogGym netlist uses 3-5× but on
gf180mcu 3.3 V a 2× factor balances the architectural property with
the relaxed 5 dB / 10 kHz spec floor; range 1.5-8× is reasonable). The oversizing makes gmf1 dominate the
FF path and track bias strongly, so the LHP zero from gmf1 / C_out
follows the non-dominant pole at gm3 / C_L across PVT.

  AFFC's Mgf2:  PMOS, g=d1, d=out (W = 1× Mcs3 by default)
  PFC's Mgmf1:  PMOS, g=d1, d=out (W = 2× Mcs3 by default)

Same architectural device position as AFFC's Mgf2, but the W/L
philosophy is opposite (oversize on purpose, not minimum-sized).
The stage-3 NMOS sink Mlcs3 is widened by (1 + PFC_oversize) to
absorb Mgmf1's larger DC source current.

Same 5-port surface as the NMCF baseline; the oversized FF device
and the widened sink are internal. See `clay.templates.ramos_pfc`
for the full topology description and the PFC_oversize / Mgmf1
kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.ramos_pfc import build_nested_miller_3stage_pfc
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("RamosPFC", testbench=build_ac_testbench)
def RamosPFC(spec: Spec, **kwargs) -> Cell:
    """Build a Ramos PFC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    Cm2, gm_id_diff / load1 / stage2 / stage3), the PFC cell accepts
    these new knobs:

      PFC_oversize: float = 2.0   — ratio of Mgmf1 W to Mcs3 W
      W_gmf1, L_gmf1: optional explicit overrides for Mgmf1 sizing
      gm_id_gmf1: gm/Id intent override for Mgmf1

    Note: Cf kwarg from the NMCF baseline is NOT accepted here
    because the PFC topology does not contain Cf (Mgmf1's oversized
    FF provides the LHP zero and forces it to track the non-dominant
    pole).
    """
    return build_nested_miller_3stage_pfc(
        cell_name=kwargs.pop("cell_name", "RamosPFC"),
        spec=spec,
        **kwargs,
    )
