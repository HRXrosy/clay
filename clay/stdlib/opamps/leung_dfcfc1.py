"""LeungDFCFC1 — 3-stage opamp with single FF + cascoded stage-1 mirror.

Leung 2003 v1 Damped Feed-Forward Compensation (DFCFC1). Same
signal path as the textbook NMCF
(`nested_miller_3stage_feedforward`) but adds two structural
features that distinguish it from all 8 prior NMCF-family variants:

  1. An active feedforward PMOS device `Mgf2` (PMOS, g=d1, d=out)
     — same structural role as HoiLee_AFFC's Mgf2 (primary FF path).
  2. A cascoded NMOS mirror at stage 1: gain devices M3 / M4
     drain to internal cascode-source nodes `d1_int` /
     `d1_diode_int`, with cascode devices `Mcas1_n_load` /
     `Mcas1_n_diode` (gate=vcas) sitting between those internal
     nodes and the actual d1 / d1_diode high-impedance nets. A
     cascode-bias diode `Mcas_bias` + driver `Mbp_cas_drv` set
     `vcas` ≈ Vgs_n above vss.

DFCFC1 vs DFCFC2 in one sentence: DFCFC1 has ONE Mgf + cascoded
stage-1 mirror with NMCF-style Cm1 routing (out↔d2); DFCFC2 has TWO
Mgf (dual FF) + single stage-1 mirror with skip-over Cm1 routing
(out↔d1). Same Leung 2003 paper family, structurally distinct
compensation philosophy.

Same 5-port surface as the NMCF baseline; the new cascode devices,
bias rail, and internal cascode-source nodes are internal. See
`clay.templates.leung_dfcfc1` for the full topology description
and the Mgf2 / W_cas1 / Mcas_bias kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.leung_dfcfc1 import build_nested_miller_3stage_dfcfc1
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("LeungDFCFC1", testbench=build_ac_testbench)
def LeungDFCFC1(spec: Spec, **kwargs) -> Cell:
    """Build a Leung DFCFC1 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    Cm2, gm_id_diff / load1 / stage2 / stage3), the DFCFC1 cell
    accepts these new knobs:

      Mgf2_scale: float = 1.0     — ratio of Mgf2 W to Mcs3 W
      W_gf2, L_gf2: optional explicit overrides for Mgf2 sizing
      gm_id_gf2: gm/Id intent override for Mgf2
      W_cas1, L_cas1: float = 4 µm / 1 µm — cascode device sizing
      W_cas_bias, L_cas_bias: float = 4 µm / 1 µm — cascode-bias
                                                    diode sizing
      N_cas_bias: int = 1 — multiplier on Mcas_bias

    Note: Cf kwarg from the NMCF baseline is NOT accepted here
    because the DFCFC1 topology does not contain Cf (Mgf2 provides
    the LHP zero structurally).
    """
    return build_nested_miller_3stage_dfcfc1(
        cell_name=kwargs.pop("cell_name", "LeungDFCFC1"),
        spec=spec,
        **kwargs,
    )
