"""SongDACFC — 3-stage opamp with doubly-differential active-cascode FF.

Song Doubly-differential Active-Cascode FeedForward Compensation
(DACFC). Same signal path as the textbook NMCF
(`nested_miller_3stage_feedforward`) but the FF network is a
doubly-differential structure:

  1. Matched gma1 PMOS pair (Mga1_a, Mga1_b) — both gated by
     bias_d2 (stage-2 replica node), drains at two separate
     cascode nodes nga1_a / nga1_b. Half 1 of doubly-differential.
  2. Input-coupled gmf1_PMOS pair (Mgmf1_n, Mgmf1_p) — gated
     directly by inn and inp respectively (NOT by d1 like AFFC's
     Mgf2), drains coupled to the SAME nga1_a / nga1_b nodes.
     Half 2 of doubly-differential — the input-coupling is
     DACFC's distinguishing architectural choice.
  3. Single gma2 PMOS buffer (Mga2) — additional cascode buffer
     between the gma1 pair and the output sense network.
  4. Single Mgmf2 output FF (PMOS, g=d2, d=out) — compare AFFC
     (g=d1, d=out): DACFC's gmf2 is gated by d2, NOT d1.
  5. Cm1 (out↔d2) NMCF-style outer Miller; Cm2_dac (nga1_a↔vss)
     inner cap to vss on cascode side; Cf dropped.

DACFC vs AFFC vs PFC in one sentence:
  AFFC has ONE FF (g=d1, d=out, W=1×).
  PFC has ONE FF (g=d1, d=out, W=4× oversized).
  DACFC has FIVE FF devices: matched pair (g=bias_d2) + input pair
  (g=inn, g=inp) + single buffer (g=bias_d2) + output single
  (g=d2). Input-coupled pair is the distinguishing architectural
  choice.

Same 5-port surface as the NMCF baseline; the new internal nodes
(nga1_a, nga1_b, nga2) are internal. See `clay.templates.song_dacfc`
for the full topology description and the W_gma1 / W_gmf1_in /
Mgmf2_scale / Cm2_dac kwargs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.song_dacfc import build_nested_miller_3stage_dacfc
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("SongDACFC", testbench=build_ac_testbench)
def SongDACFC(spec: Spec, **kwargs) -> Cell:
    """Build a Song DACFC 3-stage opamp Cell IR.

    kwargs are forwarded to the template builder. In addition to the
    NMCF baseline kwargs (I_ref, W_*, L_*, N_tail, K_cs3_load, Cm1,
    gm_id_diff / load1 / stage2 / stage3), the DACFC cell accepts
    these new knobs:

      W_gma1, L_gma1: float = 8 µm / 0.5 µm
                          — matched pair sizing (Mga1_a + Mga1_b)
      W_gmf1_in, L_gmf1_in: float = 8 µm / 1 µm
                          — input-coupled pair sizing
                            (Mgmf1_n + Mgmf1_p)
      W_gma2, L_gma2: float = 4 µm / 1 µm   — single buffer sizing
      Mgmf2_scale: float = 1.0   — ratio of Mgmf2 W to Mcs3 W
      W_gmf2, L_gmf2, gm_id_gmf2: optional overrides for Mgmf2
      Cm2_dac: float = 0.5 pF   — inner cap to vss (nga1_a↔vss)

    Note: Cf kwarg from the NMCF baseline is NOT accepted here
    because the DACFC topology drops Cf (the doubly-differential
    FF network + Mgmf2 replace it).
    """
    return build_nested_miller_3stage_dacfc(
        cell_name=kwargs.pop("cell_name", "SongDACFC"),
        spec=spec,
        **kwargs,
    )
