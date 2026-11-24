"""QuLEC — 3-stage opamp with local embedded compensation.

Qu Local Embedded Compensation (LEC). Same signal path as the
textbook NMCF (`nested_miller_3stage_feedforward`) but adds a local
amplifier + local R-C-to-vss network in the compensation path:

  1. Local error amplifier `Mlea` (PMOS, g=d1, d=nlec) — taps off
     stage-1 output d1 and drives a new internal node nlec.
  2. Local error buffer `Mleb` (NMOS, g=nlec, d=nleb) — inverts
     the local amp's drive; nleb tracks d1 through the local R-C
     filter at nlec.
  3. Local R-C filter: `Clec_gnd` (cap, p=nlec, n=vss) and `Rlec`
     (res, p=nlec, n=nleb). The cap referenced to vss is the LEC
     architectural signature — no other NMCF-family variant has a
     compensation cap to vss.
  4. Mlea g=d1 (local-loop branch attaches as parallel signal path
     on d1); Mcs2's gate stays at d1 directly so stage-2 DC
     operating point is preserved.
  5. Active feedforward `Mgf_lec` (PMOS, g=d1, d=out) — shared
     with AFFC's Mgf2; provides the LHP zero structurally.
  6. Re-routed Miller: Cm1 (out↔d1 skip-over only); Cm2 + Cf
     dropped (the local R-C network takes over the inner-Miller
     pole-splitting role).

Note: AnalogGym's upstream Qu_LEC_Pin_3 spice netlist is empty
(zero-byte); this implementation is reconstructed from the
companion spectre netlist's topology. The LEC architectural
fingerprint is locked by `tests/test_qu_lec_distinct.py`.

Same 5-port surface as the NMCF baseline; the new local-loop
devices, internal nodes (nlec, nleb), and local R-C filter are
internal. See `clay.templates.qu_lec` for the full topology
description and the Mlea / Mleb / Rlec / Clec_gnd kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.qu_lec import build_nested_miller_3stage_lec
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("QuLEC", testbench=build_ac_testbench)
def QuLEC(spec: Spec, **kwargs) -> Cell:
    """Build a Qu LEC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    gm_id_diff / load1 / stage2 / stage3), the LEC cell accepts
    these new knobs:

      Mgf_lec_scale: float = 1.0   — ratio of Mgf_lec W to Mcs3 W
      W_gf_lec, L_gf_lec, gm_id_gf_lec: overrides for Mgf_lec
      Mlea_scale: float = 1.0      — ratio of Mlea W to Mcs3 W
      W_lea, L_lea, gm_id_lea: overrides for Mlea sizing
      Mleb_scale: float = 1.0      — ratio of Mleb W to Mcs2 W
      W_leb, L_leb, gm_id_leb: overrides for Mleb sizing
      Rlec: float = 10e3           — local resistor (nlec↔nleb)
      Clec_gnd: float = 0.5e-12    — local cap to vss (nlec↔vss)

    Note: Cf and Cm2 kwargs from the NMCF baseline are NOT accepted
    here because the LEC topology drops both (Mgf_lec provides the
    LHP zero; the local R-C network takes over the inner-Miller
    pole-splitting role).
    """
    return build_nested_miller_3stage_lec(
        cell_name=kwargs.pop("cell_name", "QuLEC"),
        spec=spec,
        **kwargs,
    )
