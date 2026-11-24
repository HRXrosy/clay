"""
clay.stdlib.vcos — standard-library voltage-controlled oscillator cells
(package).

Mirrors the structure of `clay.stdlib.opamps` but holds VCO topologies
used as building blocks of phase-locked loops (PLLs). v0.0.56 ships
the cross-coupled NMOS LC-tank VCO; the existing ring-oscillator VCO
remains a textual `.clay` cell (`examples/pll_subcells/ring_vco.clay`)
that the PLL composer instantiates directly — it does NOT live in this
package.

Each topology lives in its own file (.py) inside this directory; the
__init__.py declares the package and re-imports each topology file so
every @register_cell decorator runs at package import time.
"""

from __future__ import annotations

from ...module import package


# Declare the Clay package once at the directory level.
package("clay.stdlib.vcos", version="0.0.1")

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


_force_import("lc_tank_vco")
