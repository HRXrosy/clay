"""Thin compatibility shim over the per-PDK adapters.

The per-PDK facts now live in one module each (``clay/pdk/open_pdk.py``,
``open_pdk.py``, ``open_pdk.py``, ``open_pdk.py``, ``open_pdk.py``), registered on
the single ``clay.pdk.PdkAdapter`` via ``register_pdk``. This module keeps the old
``real_pdk_recipes`` function surface (``preamble`` / ``device_card`` /
``has_recipe`` / ``get_recipe`` / ``runnable_pdks`` / ...) so the portable harness
and the pre-sim helpers keep working unchanged -- but every fact is sourced from
the adapters, NOT a hand-maintained recipe table. There is one home per PDK.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from clay.pdk import (  # noqa: F401  (Device re-exported for back-compat imports)
    Device, PdkAdapter, all_adapters, get_pdk, load_all, runnable_ids,
)

# Recipe is an alias of the unified adapter now (back-compat type name).
Recipe = PdkAdapter


def _commercial() -> Dict[str, PdkAdapter]:
    """Adapters that carry a runnable commercial HSPICE-lib binding."""
    return {a.pdk_id.lower(): a for a in all_adapters() if a.model_lib_rel}


class _RecipeMap(dict):
    """RECIPES is the live commercial-adapter map (rebuilt from the registry)."""
    def __missing__(self, key):
        return _commercial()[key]

    def __iter__(self):
        return iter(_commercial())

    def __contains__(self, key):
        return key in _commercial()

    def items(self):
        return _commercial().items()

    def values(self):
        return _commercial().values()

    def keys(self):
        return _commercial().keys()

    def get(self, key, default=None):
        return _commercial().get(key, default)


RECIPES = _RecipeMap()


def has_recipe(pdk_id: str) -> bool:
    return pdk_id.lower() in _commercial()


def get_recipe(pdk_id: str) -> PdkAdapter:
    return _commercial()[pdk_id.lower()]


def runnable_pdks() -> Tuple[str, ...]:
    return tuple(runnable_ids())


def resolve_pdk_dir(pdk_id: str) -> Optional[Path]:
    a = _commercial().get(pdk_id.lower())
    return a.root() if a else None


def model_lib_path(pdk_id: str) -> Optional[Path]:
    a = _commercial().get(pdk_id.lower())
    return a.model_lib_path() if a else None


def pdk_available(pdk_id: str) -> bool:
    a = _commercial().get(pdk_id.lower())
    return bool(a and a.lib_available())


def preamble(pdk_id: str, corner: Optional[str] = None) -> str:
    return get_recipe(pdk_id).sim_preamble(corner)


def device_card(pdk_id: str, which: str, inst: str, nodes: str, *, w: float, l: float,
                w_unit: str = "u", l_unit: str = "u") -> str:
    return get_recipe(pdk_id).device_card(
        which, inst, nodes, w=w, l=l, w_unit=w_unit, l_unit=l_unit
    )


__all__ = [
    "Device", "Recipe", "RECIPES", "has_recipe", "get_recipe", "runnable_pdks",
    "resolve_pdk_dir", "model_lib_path", "pdk_available", "preamble", "device_card",
]
