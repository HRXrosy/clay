"""
clay.stdlib.single_stage — standard-library single-ended (non-differential)
amplifier cells (package).

This package mirrors the structure of `clay.stdlib.opamps` but holds
non-differential single-stage topologies (common-source, common-drain,
common-gate, source follower, cascode, ...). Each topology lives in its
own file (.py) inside this directory; the __init__.py declares the
package and re-imports each topology file so all @register_cell
decorators run.
"""

from __future__ import annotations

from ...module import package


# Declare the Clay package once at the directory level.
package("clay.stdlib.single_stage", version="0.0.1")

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


_force_import("common_source")
