"""
clay.stdlib.opamps — standard-library opamp cells (package).

This package is one Clay module that aggregates multiple opamp topologies.
Each topology lives in its own file (.py) inside this directory; the
__init__.py declares the package and re-imports each topology file so all
@register_cell decorators run.
"""

from __future__ import annotations

from ...module import package


# Declare the Clay package once at the directory level.
package("clay.stdlib.opamps", version="0.0.2")

# Import each topology submodule. When this __init__ is re-executed via
# importlib.reload() (e.g. after reset_registry() in tests), submodules
# would otherwise stay cached in sys.modules and skip their @register_cell
# decorators. Explicitly reload each.
import importlib as _il      # noqa: E402
import sys as _sys           # noqa: E402

def _force_import(submod_name: str):
    full = f"{__name__}.{submod_name}"
    if full in _sys.modules:
        _il.reload(_sys.modules[full])
    else:
        _il.import_module(full)


_force_import("two_stage_miller")

# Optional topology files (added by subagents in v0.0.10). Wrapped in try
# so missing ones do not break the package.
for _opt in (
    "folded_cascode",
    "telescopic_cascode",
    "two_stage_miller_nmos",
    "two_stage_miller_pmos_explicit_bias",
    "nested_miller_3stage",
    "nested_miller_3stage_nmcf",
    "hoilee_affc",
    "fan_smc",
    "sau_cfcc",
    "leung_nmcnr",
    "nested_miller_3stage_nmcnr_cascoded_input",
    "affc_dual_miller_skip",
    "single_stage_dual_feedforward",
    "peng_tcfc",
    "peng_acbc",
    "leung_dfcfc2",
    "leung_dfcfc1",
    "alfio_raffc",
    "qu_lec",
    "ramos_pfc",
    "qu2017_azc",
    "song_dacfc",
    "peng_iac",
    "tan_clia",
    "yan_az",
    # v0.0.92 Round 68 Agent R — 3 stdlib opamps exercising the new
    # third-batch functional-role keywords as primary architectural
    # features (regulated_cascode, gain_boost_amp, dynamic_bias,
    # output_class_AB, common_mode_feedback, sample_hold_switch).
    "regulated_cascode_opamp",
    "class_AB_output_opamp",
    "dynamic_bias_opamp",
):
    try:
        _force_import(_opt)
    except Exception:
        pass
