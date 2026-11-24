"""FullyDifferentialPMOSInput — single-stage fully-differential amplifier
with PMOS input pair and NMOS active loads.

First fully-differential (two-output) cell in Clay's standard library.
Port list is `(inn, inp, out_p, out_n, vdd, vss)` — TWO output ports
that carry the differential signal.

See clay.templates.fully_differential for the topology spec, bias plan,
and the v0.0.22 pseudo-CMFB approximation (deferred to a follow-up
release for a real CMFB loop).
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.fully_differential import (
    build_ac_testbench_fda, build_fully_differential_pmos_input,
)


@register_cell("FullyDifferentialPMOSInput", testbench=build_ac_testbench_fda)
def FullyDifferentialPMOSInput(spec: Spec, **kwargs) -> Cell:
    """Build a single-stage fully-differential PMOS-input amp Cell IR.

    kwargs are forwarded to the template builder for sizing overrides:
    I_ref, W_bp, L_bp, N_tail, W_diff, L_diff, W_load, L_load,
    gm_id_diff, gm_id_load, R_cmfb, R_div, CL.
    """
    return build_fully_differential_pmos_input(
        cell_name=kwargs.pop("cell_name", "FullyDifferentialPMOSInput"),
        spec=spec,
        **kwargs,
    )
