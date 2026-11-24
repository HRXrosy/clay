"""AlfioRAFFC — 3-stage opamp with reverse active feedforward.

Alfio Reverse Active FeedForward Compensation (RAFFC). Same signal
path as the textbook NMCF (`nested_miller_3stage_feedforward`) but
the active feedforward device is wired in the REVERSE direction
compared to AFFC:

  AFFC's Mgf2: PMOS, g=d1, d=out (forward: stage 1 → output)
  RAFFC's Mgf_rev: PMOS, g=out, d=d1 (reverse: output → stage 1,
                                      local negative-feedback)

The reverse direction creates a different LHP-zero location in the
loop transfer function (mid-frequency instead of high-frequency)
and a structurally different compensation behaviour. The NMCF Cf
cap is dropped because Mgf_rev provides the LHP zero structurally.

Same 5-port surface as the NMCF baseline; the reverse-FF device
and the widened NMOS load at d1 are internal. See
`clay.templates.alfio_raffc` for the full topology description
and the Mgf_rev / Mgf_rev_scale kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.alfio_raffc import build_nested_miller_3stage_raffc
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("AlfioRAFFC", testbench=build_ac_testbench)
def AlfioRAFFC(spec: Spec, **kwargs) -> Cell:
    """Build an Alfio RAFFC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    Cm2, gm_id_diff / load1 / stage2 / stage3), the RAFFC cell
    accepts these new knobs:

      Mgf_rev_scale: float = 1.0   — ratio of Mgf_rev W to Mcs3 W
      W_gf_rev, L_gf_rev: optional explicit overrides for Mgf_rev
      gm_id_gf_rev: gm/Id intent override for Mgf_rev

    Note: Cf kwarg from the NMCF baseline is NOT accepted here
    because the RAFFC topology does not contain Cf (Mgf_rev provides
    the LHP zero structurally via the reverse-direction feedforward).
    """
    return build_nested_miller_3stage_raffc(
        cell_name=kwargs.pop("cell_name", "AlfioRAFFC"),
        spec=spec,
        **kwargs,
    )
