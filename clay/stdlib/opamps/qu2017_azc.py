"""Qu2017AZC — 3-stage opamp with auto-zero compensation.

Qu 2017 Auto-Zero Compensation (AZC). Same signal path as the
textbook NMCF (`nested_miller_3stage_feedforward`) but replaces the
NMCF Cm1 / Cm2 / Cf compensation network with a triple Miller-cap
+ quad nulling-resistor charge-redistribution mesh:

  1. Three AZC MOSFETs forming the charge-redistribution archetype:
     - Mazc1 (NMOS, d=nazc1, g=d2) — triggers redistribution from d2
     - Mazc2_diode (PMOS diode at nazc1) — DC bias clamp
     - Mazc3 (NMOS, d=nazc_hub, g=nazc_cap2) — closes the loop
  2. Four AZC nulling resistors (Razc0 / Razc1 / Razc2 / Razc3)
     forming a quad-R mesh with two shared hub nodes (nazc_bias,
     nazc_hub). Razc0 + Razc1 share endpoint nazc_bias; Razc2 +
     Razc3 share endpoint nazc_hub.
  3. Three AZC compensation caps:
     - Cazc0 (out ↔ nazc_bias) — outer Miller-like cap to out
     - Cazc1 (nazc_cap1 ↔ vss) — inner cap to vss (AZC signature)
     - Cazc2 (nazc_cap2 ↔ vss) — inner cap to vss (AZC signature)

AZC vs LEC in one sentence: LEC has ONE compensation cap to vss
(Clec_gnd) + ONE local resistor (Rlec); AZC has TWO compensation
caps to vss + FOUR nulling resistors forming a quad-R mesh — the
charge-redistribution archetype trio.

Same 5-port surface as the NMCF baseline; the AZC mesh nodes
(nazc1, nazc_hub, nazc_bias, nazc_cap0, nazc_cap1, nazc_cap2) are
internal. See `clay.templates.qu2017_azc` for the full topology
description and the W_azc / Razc* / Cazc* kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.qu2017_azc import build_nested_miller_3stage_azc
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("Qu2017AZC", testbench=build_ac_testbench)
def Qu2017AZC(spec: Spec, **kwargs) -> Cell:
    """Build a Qu2017 AZC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load,
    gm_id_diff / load1 / stage2 / stage3), the AZC cell accepts
    these new knobs:

      W_azc, L_azc: float = 4 µm / 1 µm   — AZC MOSFET trio sizing
      Razc0, Razc1, Razc2, Razc3: float = 10 kΩ each
                                            — quad-R mesh values
      Cazc0: float = 1.5 pF   — outer Miller-like cap (out↔nazc_bias)
      Cazc1: float = 0.5 pF   — inner cap to vss (nazc_cap1↔vss)
      Cazc2: float = 0.5 pF   — inner cap to vss (nazc_cap2↔vss)

    Note: Cf, Cm1, Cm2 kwargs from the NMCF baseline are NOT accepted
    here because the AZC topology drops all three (the AZC mesh
    replaces the entire NMCF compensation network).
    """
    return build_nested_miller_3stage_azc(
        cell_name=kwargs.pop("cell_name", "Qu2017AZC"),
        spec=spec,
        **kwargs,
    )
