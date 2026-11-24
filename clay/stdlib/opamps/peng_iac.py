"""PengIAC — 3-stage opamp with indirect AC compensation.

Peng Indirect AC Compensation (IAC). Same signal path as the
textbook NMCF (`nested_miller_3stage_feedforward`) but the inner
Miller cap is replaced by an INDIRECT path routed through a
triode-region PMOS helper (Mgmt) + series RC chain to vss:

  1. Triode-helper PMOS Mgmt (d=niac, g=vcas_iac, s=npath) —
     provides the indirect AC path. Gate at a cascode bias rail
     (NOT d2 — the indirect signature), source at intermediate
     bias rail npath, drain at the indirect Miller node niac.
  2. Active feedforward Mgmf (PMOS, g=d1, d=out) — provides LHP
     zero through standard AFFC-style FF leg.
  3. Series RC chain on indirect node: Cm2_iac (niac↔n_rc_iac) +
     R0_iac (n_rc_iac↔vss). Outer Miller Cm1_iac (out↔d2)
     NMCF-style preserved.
  4. NMCF's Cm2 (d2↔d1) and Cf (out↔d1) dropped — the indirect
     path + active FF + RC nulling replace them.

IAC vs AFFC vs LEC in one sentence:
  AFFC has ONE FF (Mgf2 PMOS, g=d1, d=out) + Cm1/Cm2/Cf NMCF caps.
  LEC has local error amp + buffer + cap-to-vss + Rlec + Cm1
       skip-over routing (d1↔out).
  IAC has triode helper Mgmt (g=vcas_iac NOT d2) + active FF Mgmf
      (g=d1) + indirect Miller node niac + series RC chain to vss
      + Cm1_iac NMCF-style outer.

Same 5-port surface as the NMCF baseline; the indirect node niac,
vcas_iac/npath bias rails, and n_rc_iac inner-RC node are
internal. See `clay.templates.peng_iac` for the full topology
description and the W_gmt / Mgmf_scale / Cm1_iac / Cm2_iac /
R0_iac kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.peng_iac import build_nested_miller_3stage_iac
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("PengIAC", testbench=build_ac_testbench)
def PengIAC(spec: Spec, **kwargs) -> Cell:
    """Build a Peng IAC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load,
    gm_id_diff / load1 / stage2 / stage3), the IAC cell accepts these
    new knobs:

      W_gmt, L_gmt: float = 8 µm / 1 µm
                           — triode-helper Mgmt sizing
      Mgmf_scale: float = 1.0   — ratio of Mgmf W to Mcs3 W
      W_gmf, L_gmf, gm_id_gmf: optional overrides for Mgmf sizing
      Cm1_iac: float = 1.5 pF   — outer Miller cap (out↔d2)
      Cm2_iac: float = 0.5 pF   — indirect cap (niac↔n_rc_iac)
      R0_iac:  float = 10 kΩ    — nulling-R (n_rc_iac↔vss)

    Note: Cm1, Cm2, Cf kwargs from the NMCF baseline are NOT
    accepted here — IAC replaces Cm2 and Cf entirely with the
    indirect path + RC chain, and uses Cm1_iac for the outer
    Miller position.
    """
    return build_nested_miller_3stage_iac(
        cell_name=kwargs.pop("cell_name", "PengIAC"),
        spec=spec,
        **kwargs,
    )
