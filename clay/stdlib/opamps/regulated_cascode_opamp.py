"""RegulatedCascodeOpamp — telescopic cascode with gain-boost regulated cascode.

Single-stage telescopic cascode opamp augmented with a regulated-cascode
gain-boost on the NMOS cascode of the output branch. The boost amp
Aboost (PMOS-input common-source helper) drives the cascode gate so the
cascode source voltage stays pinned regardless of d_aux current —
multiplying the cascode's gm·ro by the boost amp's open-loop gain.

The dynamic-bias rail (beta-multiplier Mbm1/Mbm2 + Rbm) supplies the
PTAT bias current so the cell tracks across PVT corners.

Architectural roles available in the .clay surface:
  - regulated_cascode  Mrc      (cascode helper, gate=nrc_gate)
  - gain_boost_amp     Aboost   (drives Mrc gate, samples d_aux)
  - dynamic_bias       Idyn     (beta-multiplier ref bias)
  - input_transconductor Mdp    (diff-pair stage 1)

See `clay.templates.regulated_cascode_opamp` for the full topology
description and sizing knobs.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.regulated_cascode_opamp import (
    build_regulated_cascode_opamp,
)
from ...templates.two_stage_miller_pmos import build_ac_testbench


@register_cell("RegulatedCascodeOpamp", testbench=build_ac_testbench)
def RegulatedCascodeOpamp(spec: Spec, **kwargs) -> Cell:
    """Build a Cell IR for a telescopic-cascode opamp with regulated-
    cascode gain boost.

    kwargs are forwarded to the template builder for sizing overrides:
    I_ref, Rbm, W_bm1, L_bm1, W_bm2, L_bm2, W_bp, L_bp, W_bn, L_bn,
    N_tail, M_load, W_diff, L_diff, gm_id_diff, W_rc, L_rc, W_rc_dummy,
    L_rc_dummy, W_btm, L_btm, M_btm, W_pload, L_pload, W_boost, L_boost,
    W_boost_load, L_boost_load, Cc.
    """
    return build_regulated_cascode_opamp(
        cell_name=kwargs.pop("cell_name", "RegulatedCascodeOpamp"),
        spec=spec,
        **kwargs,
    )
