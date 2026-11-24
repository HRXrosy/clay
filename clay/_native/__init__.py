"""Loader for Clay's C++20 native PnR acceleration modules.

All production placement and routing hot paths execute through the compiled
C++20 native extensions (``_place_cpp``, ``_pathfinder_cpp``, ``_gridless_cpp``)
fail-closed by default so production flows never silently degrade to legacy
pure-Python loops.

Set ``CLAY_DISABLE_CPP_PNR=1`` in the environment ONLY when explicitly running
A/B differential parity tests or benchmarks against the pure-Python reference.
"""
from __future__ import annotations

import importlib
import os
from typing import Any, Optional

REQUIRED_NATIVE_MODULES: tuple[str, ...] = (
    "_engine_cpp",
    "_place_cpp",
    "_pathfinder_cpp",
    "_gridless_cpp",
)

_CACHE: dict[str, Optional[Any]] = {}


class NativePnRUnavailableError(ImportError):
    """Raised when a required C++20 native PnR extension cannot be loaded."""


def cpp_pnr_disabled() -> bool:
    val = os.environ.get("CLAY_DISABLE_CPP_PNR", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def load_native(module_name: str) -> Optional[Any]:
    """Load ``clay._native.<module_name>`` fail-closed unless explicitly disabled.

    Returns ``None`` ONLY when ``CLAY_DISABLE_CPP_PNR=1`` is set (A/B parity /
    benchmarking override). Otherwise raises :class:`NativePnRUnavailableError`
    when the compiled C++20 module cannot be imported.
    """
    if cpp_pnr_disabled():
        return None
    if module_name in _CACHE:
        cached = _CACHE[module_name]
        if cached is not None:
            return cached
    try:
        mod = importlib.import_module(f"clay._native.{module_name}")
    except ImportError as exc:
        raise NativePnRUnavailableError(
            f"Required C++20 native PnR module 'clay._native.{module_name}' could not be loaded "
            f"({exc}). Build all extensions via `python cpp/clay_pnr/build.py --target all`, "
            "or set `CLAY_DISABLE_CPP_PNR=1` explicitly for pure-Python differential testing."
        ) from exc
    _CACHE[module_name] = mod
    return mod


def verify_native_modules() -> dict[str, str]:
    """Verify all required C++20 PnR extensions load and return their file paths."""
    loaded: dict[str, str] = {}
    for name in REQUIRED_NATIVE_MODULES:
        mod = load_native(name)
        if mod is None:
            raise NativePnRUnavailableError(
                f"C++20 native PnR module {name!r} is disabled via CLAY_DISABLE_CPP_PNR=1"
            )
        loaded[name] = str(getattr(mod, "__file__", "<builtin>"))
    return loaded


def clear_native_cache() -> None:
    _CACHE.clear()
