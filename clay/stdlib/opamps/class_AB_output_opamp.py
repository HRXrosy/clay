"""ClassABOutputOpamp — 2-stage Miller opamp with class-AB push/pull output.

Two-stage Miller-compensated opamp with a class-AB push-pull output
stage (NMOS push + PMOS pull source-followers sharing a translinear
bias loop) plus a CMFB error amp that regulates the bias-rail common
mode to fix the quiescent current across PVT.

Architectural roles available in the .clay surface:
  - output_class_AB     Mab     (push/pull pair Mab_push / Mab_pull)
  - common_mode_feedback Acmfb  (bias-rail CMFB amp)
  - dominant_pole_cap   Cc      (Miller cap, legacy keyword)
  - input_transconductor Mdp    (stage-1 diff pair)
  - tail_current_source Mtail   (stage-1 tail, legacy)

See `clay.templates.class_AB_output_opamp` for the full topology
description and sizing knobs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.class_AB_output_opamp import (
    build_class_AB_output_opamp,
)
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("ClassABOutputOpamp", testbench=build_ac_testbench)
def ClassABOutputOpamp(spec: Spec, **kwargs) -> Cell:
    """Build a Cell IR for a 2-stage Miller opamp with class-AB output
    stage and bias-rail CMFB.

    kwargs are forwarded to the template builder for sizing overrides:
    I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, M_cs2_load, W_diff, L_diff,
    W_load1, L_load1, gm_id_diff, gm_id_load1, W_cs2, L_cs2, gm_id_cs2,
    W_ab_diode, L_ab_diode, W_ab_push, L_ab_push, W_ab_pull, L_ab_pull,
    W_cmfb_amp, L_cmfb_amp, W_cmfb_load, L_cmfb_load, Cc.
    """
    return build_class_AB_output_opamp(
        cell_name=kwargs.pop("cell_name", "ClassABOutputOpamp"),
        spec=spec,
        **kwargs,
    )
