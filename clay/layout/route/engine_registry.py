"""General NAME -> :class:`RouteEngine` strategy registry.

Consumes a design's declared ``@route_policy(engine=NAME, kv...)`` (carried on
:class:`clay.layout.policy.EffectivePolicy` as ``route_engine`` /
``route_engine_params``) into a constructed engine instance for
``route_cell(..., engine=...)``. Data-driven: the engine is selected by NAME
from this one registry -- never a per-circuit branch -- and the extra declared
kv are the engine's own constructor kwargs (e.g. ``time_budget_sec``), so a new
engine strategy plugs in by registering a name, not by threading new flags.

Contract:
  * no name (None / "") -> ``None``: the caller keeps route_cell's
    byte-identical production default (``PathFinderEngine(widen=True)``).
  * a known name -> the constructed instance (kwargs applied; a wrong kwarg
    raises the constructor's own TypeError naming it).
  * an UNKNOWN name -> :class:`UnknownRouteEngineError` naming the name and
    the registry (NEVER a silent fallback -- a declared engine that cannot be
    honored must fail loudly, not quietly route with a different strategy).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class UnknownRouteEngineError(ValueError):
    """A declared @route_policy engine name has no registered strategy."""


def route_engine_registry() -> Dict[str, type]:
    """The one name -> engine-class map (lazy imports keep module load light)."""
    from .engine_gridless import GridlessEngine
    from .engine_pathfinder import PathFinderEngine
    from .engine_channel import ChannelRouteEngine
    from .engine_imported import ImportedRouteEngine

    return {
        "pathfinder": PathFinderEngine,
        "gridless": GridlessEngine,
        "channel": ChannelRouteEngine,
        "imported": ImportedRouteEngine,
    }


def make_route_engine(name: Optional[str], **params: Any):
    """Construct the RouteEngine strategy declared by ``name`` (+ ctor kwargs).

    ``None``/empty -> None (use route_cell's default). ``pathfinder`` mirrors
    the production default construction (``widen=True``) unless the declaration
    overrides it, so declaring the default engine explicitly stays faithful to
    the undeclared path.
    """
    if name is None or not str(name).strip():
        return None
    key = str(name).strip().lower()
    registry = route_engine_registry()
    cls = registry.get(key)
    if cls is None:
        raise UnknownRouteEngineError(
            f"@route_policy engine={name!r} is not a registered RouteEngine "
            f"strategy; known engines: {sorted(registry)}"
        )
    kwargs = dict(params)
    if key == "pathfinder":
        kwargs.setdefault("widen", True)
    return cls(**kwargs)


def route_engine_from_policy(eff_policy) -> Optional[Any]:
    """The engine instance a resolved :class:`EffectivePolicy` declares, or None.

    Reads ``route_engine`` (name) + ``route_engine_params`` (sorted kv tuple)
    -- the ONE consumption seam the generic flow and the ota_dev build share.
    """
    name = getattr(eff_policy, "route_engine", "") or ""
    if not name:
        return None
    params = dict(getattr(eff_policy, "route_engine_params", ()) or ())
    return make_route_engine(name, **params)


__all__ = [
    "UnknownRouteEngineError",
    "make_route_engine",
    "route_engine_from_policy",
    "route_engine_registry",
]
