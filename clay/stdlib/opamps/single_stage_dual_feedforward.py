"""SingleStageDualFeedforward — single-stage opamp with TWO feedforward gm paths.

Takes the textbook single-stage differential opamp baseline (PMOS
input pair + NMOS mirror load + single high-Z output node) and adds
TWO symmetric, cross-coupled PMOS feedforward devices (Mff_p, Mff_n)
feeding the output directly from vdd. Each feedforward device's
drain current injects into out alongside the main mirror's current,
boosting differential gain and GBW (typically +6 dB / 2× over the
baseline at Mff_scale=1.0).

Same 5-port surface as the textbook 2-stage Miller cells; the
feedforward pair is internal. See
`clay.templates.single_stage_dual_feedforward` for the full topology
description, the differential-mode signal-path math, and the
common-mode rejection trade-off.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.single_stage_dual_feedforward import (
    build_single_stage_dual_feedforward,
)
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("SingleStageDualFeedforward", testbench=build_ac_testbench)
def SingleStageDualFeedforward(spec: Spec, **kwargs) -> Cell:
    """Build a Cell IR for a single-stage opamp with dual feedforward gm.

    kwargs are forwarded to the template builder. Accepted kwargs:

      I_ref: float = 1e-5                — bias chain reference current (A)
      W_bp, L_bp, W_bn, L_bn             — bias mirror sizing
      N_tail: int = 4                    — tail mirror ratio
      W_diff, L_diff                     — input pair PMOS sizing
      W_load, L_load                     — NMOS mirror sizing
      gm_id_diff, gm_id_load             — gm/Id intent overrides

    Plus three new feedforward-specific knobs:

      Mff_scale: float = 1.0    — ratio of feedforward PMOS W to input
                                   pair W. Default 1.0 yields gm_ff ≈
                                   gm_M1 (+6 dB gain bump).
      W_ff, L_ff                — optional explicit overrides for
                                   feedforward device sizing.
      gm_id_ff                  — gm/Id intent override for Mff_p / Mff_n.
    """
    return build_single_stage_dual_feedforward(
        cell_name=kwargs.pop("cell_name", "SingleStageDualFeedforward"),
        spec=spec,
        **kwargs,
    )
