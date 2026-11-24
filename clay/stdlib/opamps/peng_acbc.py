"""PengACBC — 3-stage opamp with active capacitance + bias compensation.

Peng Active Capacitance + Bias Compensation (ACBC). Same stage-1 +
stage-2 signal path as the textbook NMCF
(`nested_miller_3stage_feedforward`) but the compensation network
adds four structural features:

  1. An active feedforward PMOS device `Mgf2` (PMOS, g=d1, d=out)
     — same structural role as HoiLee_AFFC's Mgf2.
  2. A dedicated bias-comp node `nb` held at ≈ vdd − |Vgs_p| by a
     PMOS-diode + PMOS-driver pair (`Mnb_diode` + `Mnb_drv`).
  3. An active bias-comp transconductor `Mgm4` (NMOS, g=d2, d=nb)
     — the 4th gm-cell that sinks signal current into nb and
     multiplies Cm2's effective Miller capacitance.
  4. Re-routed Miller caps: Cm1 (out↔d1 skip-over) + Cm2 (nb↔d2,
     terminates at the active-cap node). Cf is dropped.

Same 5-port surface as the NMCF baseline; the new devices and node
are internal. See `clay.templates.peng_acbc` for the full topology
description and the Mgf2 / Mgm4 / W_bp_nb kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.peng_acbc import build_nested_miller_3stage_acbc
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("PengACBC", testbench=build_ac_testbench)
def PengACBC(spec: Spec, **kwargs) -> Cell:
    """Build a Peng ACBC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    Cm2, gm_id_diff / load1 / stage2 / stage3), the ACBC cell accepts
    these new knobs:

      Mgf2_scale: float = 1.0   — ratio of Mgf2 W to Mcs3 W (active FF)
      W_gf2, L_gf2: optional explicit overrides for Mgf2 sizing
      gm_id_gf2: gm/Id intent override for Mgf2
      Mgm4_scale: float = 1.0   — ratio of Mgm4 W to Mcs2 W (active gm4)
      W_gm4, L_gm4: optional explicit overrides for Mgm4 sizing
      gm_id_gm4: gm/Id intent override for Mgm4
      W_bp_nb: float = 4e-6     — PMOS-diode width for the nb reference
      L_bp_nb: float = 1e-6     — PMOS-diode length for the nb reference

    Note: Cf kwarg from the NMCF baseline is NOT accepted here because
    the ACBC topology does not contain Cf (the active feedforward +
    bias-comp loop handle PM recovery without a passive Cf).
    """
    return build_nested_miller_3stage_acbc(
        cell_name=kwargs.pop("cell_name", "PengACBC"),
        spec=spec,
        **kwargs,
    )
