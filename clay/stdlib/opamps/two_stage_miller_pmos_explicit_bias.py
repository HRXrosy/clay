"""TwoStageMillerPMOSExplicitBias — variant of `TwoStageMillerPMOS`
whose bias chain is declared in .clay source via a `bias_tree { ... }`
block (v0.0.26 / G2).

Signal path is identical to the canonical PMOS Miller two-stager;
only the bias generator is user-authored. See
`clay/templates/two_stage_miller_pmos_explicit_bias.py` for the
device-level template."""

from __future__ import annotations

from ...ir import Cell, Spec
from ...module import register_cell
from ...templates.two_stage_miller_pmos_explicit_bias import (
    build_ac_testbench, build_two_stage_miller_pmos_explicit_bias,
)


@register_cell("TwoStageMillerPMOSExplicitBias", testbench=build_ac_testbench)
def TwoStageMillerPMOSExplicitBias(spec: Spec, **kwargs) -> Cell:
    """Cell entry-point for the explicit-bias PMOS Miller variant.

    kwargs are forwarded to the template builder. In addition to the
    sizing kwargs accepted by the baseline (`I_ref`, `W_*`, `L_*`,
    `gm_id_*`, `Cc`, `K_cs_load`, ...), this cell ALSO accepts a
    `bias_tree` kwarg — a list of `clay.ir.BiasNode` declarations
    that the builder converts into the bias-chain devices. When the
    cell is reached via the cell_def compiler path the bias_tree is
    propagated automatically; when invoked directly from Python you
    must pass it explicitly.
    """
    return build_two_stage_miller_pmos_explicit_bias(
        cell_name=kwargs.pop("cell_name", "TwoStageMillerPMOSExplicitBias"),
        spec=spec,
        **kwargs,
    )
