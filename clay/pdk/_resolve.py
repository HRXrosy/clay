"""Shared PDK-install-root resolution for the per-PDK adapters.

Several proprietary-PDK adapters (open_pdk, open_pdk, open_pdk, ...) each
need to locate their gitignored install directory using the same precedence
order. This module holds the single canonical implementation so the adapters
do not hand-roll byte-identical copies (engineering treaty rule 2: a workaround
appearing twice sinks to a shared layer).

Resolution order (highest priority first):
  1. the adapter's explicit env var (``env_var``), when declared.  A non-empty
     invalid value fails closed; it is never replaced with another installation;
  2. the generic ``CLAY_PDK_PATH`` env var, but ONLY when ``CLAY_PDK_ID``
     selects this adapter (``CLAY_PDK_ID == pdk_id``, case-insensitive), and it
     names an existing dir;
  3. the in-tree ``fallback`` directory, if it exists;
  4. otherwise ``None``.

The environment is read at call time (not import time) so callers that
monkeypatch ``os.environ`` -- or that pass a freshly-recomputed ``fallback`` --
observe the change.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = ["resolve_root", "resolve_root_via_config", "root_override_declared"]


def root_override_declared(env_var: str, pdk_id: str) -> bool:
    """Whether a non-empty environment override selects this PDK.

    This deliberately reports declaration rather than validity.  A caller can
    therefore distinguish a clean checkout with no proprietary install from an
    explicit but invalid authority that must fail closed.
    """

    if os.environ.get(env_var):
        return True
    return (
        os.environ.get("CLAY_PDK_ID", "").lower() == pdk_id.lower()
        and bool(os.environ.get("CLAY_PDK_PATH"))
    )


def resolve_root(env_var: str, pdk_id: str, fallback: Path) -> Optional[Path]:
    """Locate a PDK install root; see module docstring for the precedence order."""
    explicit = os.environ.get(env_var)
    if explicit:
        path = Path(explicit)
        return path if path.is_dir() else None
    if os.environ.get("CLAY_PDK_ID", "").lower() == pdk_id:
        generic = os.environ.get("CLAY_PDK_PATH")
        if generic:
            path = Path(generic)
            return path if path.is_dir() else None
    if fallback.is_dir():
        return fallback
    return None


def resolve_root_via_config(env_var: str, pdk_id: str) -> Optional[Path]:
    """Locate a closed/notional PDK install root that has NO in-tree fallback.

    Used by the proprietary adapters (open_pdk, gf180mcu, open_pdk) whose install
    directory is never shipped in-tree, so they resolve through the global
    ``ClayConfig`` instead of a fallback path. Precedence (highest first):

      1. the adapter's explicit env var (``env_var``);
      2. ``ClayConfig.pdk_root`` when ``ClayConfig.pdk_id == pdk_id``;
      3. otherwise ``None``.

    Read at call time so a monkeypatched ``os.environ`` -- or a reconfigured
    ``ClayConfig`` -- is observed (engineering-treaty rule 2: a workaround seen
    twice sinks to a shared layer; here it was hand-rolled in THREE adapters).
    """
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    # lazy import keeps this module free of an import cycle through clay.config
    from ..config import get_config
    cfg = get_config()
    if cfg.pdk_id == pdk_id and cfg.pdk_root is not None:
        return cfg.pdk_root
    return None
