"""PengTCFC — 3-stage opamp with active feedforward + cascoded sink.

Peng 2004 Topology-cascoded Feedforward Compensation (TCFC). Same
stage-1 + stage-2 signal path as the textbook NMCF
(`nested_miller_3stage_feedforward`) but the output stage adds:

  1. An active feedforward PMOS device `Mgmf` in parallel with Mcs3,
     also gated by d2 (parallel output transconductance).
  2. A cascoded NMOS sink at the output: cascode device `Mcas_n`
     between out and a new internal node `nd3`, current source
     `Mlcs3` between nd3 and vss, biased by a separate diode chain
     (Mlcs3_diode + Mlcs3_diode_drv).
  3. The feedforward cap Cf is dropped — Mgmf + cascode handle the
     PM recovery structurally.

Same 5-port surface as the NMCF baseline; the new devices are
internal. See `clay.templates.peng_tcfc` for the full topology
description and the Mgmf / W_cas_n / Mgmf_scale kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.peng_tcfc import build_nested_miller_3stage_tcfc
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("PengTCFC", testbench=build_ac_testbench)
def PengTCFC(spec: Spec, **kwargs) -> Cell:
    """Build a Peng TCFC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    Cm2, gm_id_diff / load1 / stage2 / stage3), the TCFC cell accepts
    these new knobs:

      Mgmf_scale: float = 1.0   — ratio of Mgmf W to Mcs3 W
      W_gmf, L_gmf: optional explicit overrides for Mgmf sizing
      gm_id_gmf: gm/Id intent override for Mgmf
      W_cas_n: float = 4e-6     — NMOS cascode device width
      L_cas_n: float = 1e-6     — NMOS cascode device length

    Note: Cf kwarg from the NMCF baseline is NOT accepted here because
    the TCFC topology does not contain Cf (the active feedforward +
    cascoded sink handle PM recovery without a passive Cf).
    """
    return build_nested_miller_3stage_tcfc(
        cell_name=kwargs.pop("cell_name", "PengTCFC"),
        spec=spec,
        **kwargs,
    )
