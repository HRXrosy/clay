"""TanCLIA — 3-stage opamp with capacitor-less indirect amplifier compensation.

Tan Capacitor-Less Indirect Amplifier (CLIA). Same signal path as
the textbook NMCF (`nested_miller_3stage_feedforward`) but the
ENTIRE Miller compensation network is replaced by an active
current-mode local impedance amplifier. NO Cm1, NO Cm2, NO Cf —
literally capacitor-less in the Miller position:

  1. gm21 PMOS (Mgm21_PMOS, g=d2, d=ngm21) — senses stage-2 output
     and converts to current into gm23 NMOS mirror diode.
  2. gm23 NMOS diode (Mgm23_NMOS_a, d=g=ngm21) — diode side of the
     current mirror. Sets the local loop reference.
  3. gm23 NMOS mirror (Mgm23_NMOS_b, d=nclia_drv, g=ngm21) —
     mirror side. Drives current-mode impedance back into the
     inner RC node, completing the local loop without any Miller
     cap.
  4. Active FF Mgmf (PMOS, g=d2, d=out) — provides LHP zero
     through FF leg.
  5. ONE cap Cclia_gnd to vss (NOT a Miller cap — on an internal
     RC node) + ONE nulling-R Rclia (between gm23 mirror drain
     and cap node).

CLIA vs LEC vs AZC in one sentence:
  LEC has local error amp + buffer + cap-to-vss + Rlec + Cm1
      skip-over routing (an outer Miller cap d1↔out exists).
  AZC has triple Mazc + quad-R + triple cap (2 to vss, 1 outer
      Miller-like out↔nazc_bias).
  CLIA has gm21 PMOS + gm23 NMOS pair (current-mode local loop)
       + Cclia_gnd cap-to-vss on internal RC node + Rclia nulling.
       **NO Miller-position cap at all** — literally
       capacitor-less, the architecturally most distinct of all
       NMCF-family variants.

Same 5-port surface as the NMCF baseline; the gm21/gm23 trio
internal nodes (ngm21, nclia_drv, nclia_cap) are internal. See
`clay.templates.tan_clia` for the full topology description and
the W_gm21 / W_gm23 / Mgmf_scale / Cclia_gnd / Rclia kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.tan_clia import build_nested_miller_3stage_clia
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("TanCLIA", testbench=build_ac_testbench)
def TanCLIA(spec: Spec, **kwargs) -> Cell:
    """Build a Tan CLIA 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load,
    gm_id_diff / load1 / stage2 / stage3), the CLIA cell accepts these
    new knobs:

      W_gm21, L_gm21: float = 8 µm / 1 µm   — gm21 PMOS sizing
      W_gm23, L_gm23: float = 8 µm / 1 µm   — gm23 NMOS pair sizing
      Mgmf_scale: float = 1.0   — ratio of Mgmf W to Mcs3 W
      W_gmf, L_gmf, gm_id_gmf: optional overrides for Mgmf sizing
      Cclia_gnd: float = 1.0 pF   — cap-to-vss on inner RC node
                                     (NOT a Miller cap)
      Rclia: float = 5 kΩ          — nulling-R (nclia_drv↔nclia_cap)

    Note: Cm1, Cm2, Cf kwargs from the NMCF baseline are NOT
    accepted here — CLIA drops ALL three (capacitor-less in the
    Miller position by definition).
    """
    return build_nested_miller_3stage_clia(
        cell_name=kwargs.pop("cell_name", "TanCLIA"),
        spec=spec,
        **kwargs,
    )
