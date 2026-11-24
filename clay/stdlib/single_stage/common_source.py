"""CommonSourcePMOSInput — single-stage common-source amplifier, PMOS input.

The simplest practical amplifier topology and the first non-differential
cell in Clay's standard library. Port list is `(in, out, vdd, vss)` —
no differential pair.
"""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.common_source import (
    build_ac_testbench_cs, build_common_source_pmos_input,
)


@register_cell("CommonSourcePMOSInput", testbench=build_ac_testbench_cs)
def CommonSourcePMOSInput(spec: Spec, **kwargs) -> Cell:
    """Build a single-stage common-source PMOS-input amp Cell IR.

    kwargs are forwarded to the template builder for sizing overrides:
    I_ref, W_bn, L_bn, N_load, W_gain, L_gain, gm_id_input, CL.
    """
    return build_common_source_pmos_input(
        cell_name=kwargs.pop("cell_name", "CommonSourcePMOSInput"),
        spec=spec,
        **kwargs,
    )
