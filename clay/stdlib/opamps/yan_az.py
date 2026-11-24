"""YanAZ — 3-stage opamp with auto-zero compensation (simpler subset of AZC).

Yan Auto-Zero (AZ). Same signal path as the textbook NMCF
(`nested_miller_3stage_feedforward`) but with a SIMPLER auto-zero
compensation network than the full Qu2017 AZC:

  1. Auto-zero NMOS pair Mgmb1 (d=naz_drv, g=naz_cap) + Mgmb2
     (d=naz_pair, g=naz_drv). The reciprocal gate/drain coupling
     (Mgmb1 senses cap, Mgmb2 gated by drv) is the AZ pair
     signature.
  2. Active feedforward Mgmf2 (PMOS, g=d2, d=out) — provides LHP
     zero through FF leg.
  3. Single nulling-R Raz (naz_drv↔naz_cap) + single auto-zero
     cap Caz (naz_cap↔naz_pair) in series with the Miller path.
     Caz_outer (out↔d2) NMCF-style preserved.

AZ vs AZC in one sentence:
  AZC has THREE compensation MOSFETs (Mazc1+Mazc2_diode+Mazc3) +
      FOUR nulling-R (Razc0-Razc3 forming quad-R mesh with 2 hub
      nodes) + THREE compensation caps (Cazc0 outer Miller-like +
      Cazc1/Cazc2 to vss).
  AZ has TWO nulling MOSFETs (Mgmb1+Mgmb2) + ONE primary nulling-R
     (Raz) + ONE auto-zero cap (Caz in series with Miller path,
     NOT directly to vss) + outer Miller Caz_outer (out↔d2).
     Strict subset of AZC — half the MOSFETs, quarter the
     resistors, third of the caps.

Same 5-port surface as the NMCF baseline; the auto-zero internal
nodes (naz_drv, naz_cap, naz_pair) are internal. See
`clay.templates.yan_az` for the full topology description and the
W_gmb / Mgmf2_scale / Caz_outer / Caz / Raz kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.yan_az import build_nested_miller_3stage_az
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("YanAZ", testbench=build_ac_testbench)
def YanAZ(spec: Spec, **kwargs) -> Cell:
    """Build a Yan AZ 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load,
    gm_id_diff / load1 / stage2 / stage3), the AZ cell accepts these
    new knobs:

      W_gmb, L_gmb: float = 4 µm / 1 µm
                           — auto-zero NMOS pair sizing (matched
                             across Mgmb1 / Mgmb2)
      Mgmf2_scale: float = 1.0   — ratio of Mgmf2 W to Mcs3 W
      W_gmf2, L_gmf2, gm_id_gmf2: optional overrides for Mgmf2
      Caz_outer: float = 1.5 pF   — outer Miller cap (out↔d2)
      Caz: float = 0.5 pF          — auto-zero cap
                                     (naz_cap↔naz_pair, series with
                                     Miller)
      Raz: float = 10 kΩ           — primary nulling-R
                                     (naz_drv↔naz_cap)

    Note: Cm1, Cm2, Cf kwargs from the NMCF baseline are NOT
    accepted here — AZ uses Caz_outer (out↔d2) for outer Miller
    and replaces Cm2/Cf with the auto-zero pair + Caz series with
    Miller path.
    """
    return build_nested_miller_3stage_az(
        cell_name=kwargs.pop("cell_name", "YanAZ"),
        spec=spec,
        **kwargs,
    )
