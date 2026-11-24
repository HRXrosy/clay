"""v2-lang: resolve a cell's EFFECTIVE place/route policy.

A `policy NAME { ... }` Clay block declares the GENERAL place/route strategy
+ convergence defaults; a cell selects one via `flow { policy = NAME; }`.
This module merges that global policy with the cell's LOCAL `layout_intent`
(local wins on conflict) into a frozen :class:`EffectivePolicy` the layout
engine consumes.

Honest scope (mirrors docs/superpowers/round-v2lang-policy.md):

  * ACTIVELY consumed downstream NOW: ``place_iters`` / ``route_iters`` ->
    ``place_to_intent`` / ``route_to_intent`` ``max_iters``; ``tol`` -> the
    convergence criteria tolerance. ``learn`` -> a flag the v3 memory reads.
    ``@place_policy(engine=)`` -> the placement engine (milp | seqpair).
    ``@place_policy(aspect=, gap=)`` -> the canonical compaction optimization
    targets (``aspect_target`` / ``gap_um`` fed first to
    ``maybe_envelope_replace`` and then to the structure-aware stage).  Declared
    band/anchor/channel structure independently activates
    ``materialize_declared_compact_override``; with no aspect target it preserves
    the incoming aspect.  A cell with neither policy compaction nor craft structure
    stays on the raw byte-identical placement.
    ``@route_policy(engine=, kv...)`` -> the RouteEngine strategy by NAME
    (resolved through ``clay.layout.route.engine_registry``; extra kv are
    engine constructor kwargs, e.g. ``time_budget_sec``). These were
    previously "recorded provenance"; R7d made them CONSUMED so a design's
    .clay -- not caller parameters in a harness -- carries the decisions.
  * RECORDED as provenance (honored as the engine grows, not faked as
    already-active): the remaining ``@place_policy`` / ``@route_policy``
    strategy fields (symmetry / pack / route_layers / route_tracks /
    shield_mode). The placer already does mirror_x + rows; the policy makes
    it explicit and overridable.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Optional

from ..ir import ParsedPolicyDecl


_RETIRED_ROUTE_STYLE_KEYS = frozenset({
    "hv_discipline",
    "m2_track_assign",
    "over_cell_routing",
    "over_cell_outside_cost",
})


@dataclass(frozen=True)
class EffectivePolicy:
    """The merged place/route policy a cell's layout engine consumes.

    ``source`` records provenance: ``"default"`` when no policy was selected,
    the policy name when one resolved, and ``"<name>+local"`` when the cell's
    local ``layout_intent`` overrode one or more fields.
    """

    place_iters: int = 6
    route_iters: int = 4
    tol: float = 0.10
    symmetry: str = "mirror_x"
    pack: str = "rows"
    placement_engine: str = "milp"   # "milp" (exact) | "seqpair" (constraint-graph, scalable)
    # envelope-min compaction knobs from @place_policy(aspect=, gap=). None ->
    # no policy-driven envelope re-pack; declared band/anchor/channel structure
    # can still invoke its independent canonical materializer.
    aspect_target: Optional[float] = None
    gap_um: Optional[float] = None
    # Optional exact physical width of the normalized rigid-block outline.  This
    # is a coordinate-free size constraint, not a placement coordinate.
    outline_width_um: Optional[float] = None
    channel_strategy: str = "declared"
    # Coordinate-free structural selector used only to disambiguate which
    # declared common-centroid group owns the final master's left column.
    left_column_group: str = ""
    route_layers: tuple = ()
    route_tracks: str = "net_disjoint"
    shield_mode: str = "on_request"
    # RouteEngine strategy from @route_policy(engine=NAME, kv...). "" -> no
    # override -> route_cell's byte-identical PathFinder default. The extra kv
    # (constructor kwargs, e.g. time_budget_sec) ride as a sorted kv tuple so
    # the frozen dataclass stays hashable.
    route_engine: str = ""
    route_engine_params: tuple = ()
    learn: bool = False
    source: str = "default"


def aspect_policy_binding(aspect_target: object) -> tuple[bool, str]:
    """Return the canonical aspect-obligation bit and its replay digest."""

    if aspect_target is None:
        target = None
    else:
        if type(aspect_target) not in (int, float):
            raise TypeError("aspect target must be an exact integer, float, or None")
        target = float(aspect_target)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("aspect target must be positive finite")
    payload = {
        "aspect_target": target,
        "schema": "clay.placement-aspect-policy/v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return target is not None, hashlib.sha256(encoded).hexdigest()


def outline_width_policy_binding(width_nm: object) -> tuple[Optional[int], str]:
    """Validate and hash the optional exact rigid-block outline width."""

    if width_nm is None:
        width = None
    else:
        if type(width_nm) is not int or width_nm <= 0:
            raise ValueError("outline width must be a positive exact integer nm")
        width = width_nm
    payload = {
        "outline_width_nm": width,
        "schema": "clay.placement-outline-width-policy/v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return width, hashlib.sha256(encoded).hexdigest()


def _declared_number(policy_name: str, block: str, key: str, value) -> Optional[float]:
    """Coerce a DECLARED policy kv to float; absent -> None; non-numeric ->
    a NAMED error (a declared design decision must never be silently dropped)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"policy {policy_name!r}: @{block} {key}={value!r} is not numeric"
        ) from None


def _coerce_scalar(value):
    """Best-effort scalar coercion for engine-constructor kv (parser hands every
    SIGNED_NUMBER/CNAME through as str): numeric -> float, on/off booleans ->
    bool, anything else (incl. tuples from id-lists) unchanged."""
    if isinstance(value, tuple):
        return value
    s = str(value).strip()
    low = s.lower()
    if low in ("on", "true", "yes"):
        return True
    if low in ("off", "false", "no"):
        return False
    try:
        return float(s)
    except ValueError:
        return s


def _local_tol(layout_intent) -> Optional[float]:
    """Extract a LOCAL convergence tolerance from a cell's layout_intent.

    Override precedence for ``tol`` (local wins on conflict):
      1. an explicit numeric ``tol`` attribute on the intent, if present;
      2. otherwise the TIGHTEST (smallest) ``rel_tol`` across the intent's
         parasitic budgets -- a designer who declared a tighter matched-net
         balance tolerance is asking the loop to converge at least that hard.

    Returns ``None`` when the intent declares nothing tol-like (no override).
    """
    if layout_intent is None:
        return None
    explicit = getattr(layout_intent, "tol", None)
    if isinstance(explicit, (int, float)):
        return float(explicit)
    budgets = getattr(layout_intent, "budgets", None) or ()
    rel_tols = [
        float(b.rel_tol)
        for b in budgets
        if isinstance(getattr(b, "rel_tol", None), (int, float))
    ]
    if rel_tols:
        return min(rel_tols)
    return None


def resolve_policy(
    policy_decl: Optional[ParsedPolicyDecl],
    layout_intent=None,
) -> EffectivePolicy:
    """Merge a global ``policy {}`` with a cell's local ``layout_intent``.

    Start from defaults, overlay the policy's ``@converge`` (place_iters /
    route_iters / tol) + ``@place_policy`` / ``@route_policy`` strategy fields
    + ``@learn``; then let the LOCAL ``layout_intent`` override on conflict.

    Local-override semantics (local wins): currently the ``tol`` field is
    overridable by the cell -- a cell whose ``layout_intent`` declares a
    tighter tolerance (explicit ``tol`` or the tightest budget ``rel_tol``)
    pulls ``tol`` to its local value. ``place_iters`` / ``route_iters`` /
    strategy fields take the policy value (no local source for them yet).

    When ``policy_decl is None`` the defaults are returned with
    ``source="default"`` (a local tol override still applies, marked
    ``"default+local"``).
    """
    eff = EffectivePolicy()

    if policy_decl is not None:
        conv = policy_decl.converge or {}
        place_iters = conv.get("place_iters", eff.place_iters)
        route_iters = conv.get("route_iters", eff.route_iters)
        tol = conv.get("tol", eff.tol)

        place = policy_decl.place_policy or {}
        route = policy_decl.route_policy or {}
        layers = route.get("layers", eff.route_layers)
        if not isinstance(layers, tuple):
            layers = (layers,) if layers else ()

        # @place_policy(aspect=, gap=): the declared envelope-compaction knobs.
        # A declared-but-non-numeric value is a NAMED error -- declared intent
        # must never be silently dropped.
        aspect = _declared_number(policy_decl.name, "place_policy", "aspect",
                                  place.get("aspect"))
        gap = _declared_number(policy_decl.name, "place_policy", "gap",
                               place.get("gap"))
        outline_width = _declared_number(
            policy_decl.name,
            "place_policy",
            "outline_width",
            place.get("outline_width"),
        )
        if outline_width is not None and (
            not math.isfinite(outline_width) or outline_width <= 0.0
        ):
            raise ValueError(
                f"policy {policy_decl.name!r}: @place_policy outline_width "
                "must be positive finite"
            )
        channel_strategy = place.get("channel_strategy", eff.channel_strategy)
        if type(channel_strategy) is not str or channel_strategy not in {
            "declared",
            "block_gap_only",
        }:
            raise ValueError(
                f"policy {policy_decl.name!r}: @place_policy channel_strategy "
                "must be declared or block_gap_only"
            )

        # @route_policy(engine=NAME, kv...): the engine name plus its
        # constructor kwargs (every kv that is not a recorded strategy field).
        engine_name = str(route.get("engine", "") or "")
        retired = sorted(_RETIRED_ROUTE_STYLE_KEYS.intersection(route))
        if retired:
            raise ValueError(
                f"policy {policy_decl.name!r}: retired @route_policy key "
                f"{retired[0]!r}; routing direction, tracks, and over-cell "
                "legality are derived from PDK and certified resource plans"
            )
        _strategy_keys = {"engine", "layers", "tracks", "shield"}
        engine_params = tuple(sorted(
            (k, _coerce_scalar(v))
            for k, v in route.items() if k not in _strategy_keys
        ))

        eff = replace(
            eff,
            place_iters=int(place_iters),
            route_iters=int(route_iters),
            tol=float(tol),
            symmetry=str(place.get("symmetry", eff.symmetry)),
            pack=str(place.get("pack", eff.pack)),
            placement_engine=str(place.get("engine", eff.placement_engine)),
            aspect_target=aspect,
            gap_um=gap,
            outline_width_um=outline_width,
            channel_strategy=channel_strategy,
            left_column_group=str(place.get("left_column", "") or ""),
            route_layers=layers,
            route_tracks=str(route.get("tracks", eff.route_tracks)),
            shield_mode=str(route.get("shield", eff.shield_mode)),
            route_engine=engine_name,
            route_engine_params=engine_params,
            learn=bool(policy_decl.learn),
            source=policy_decl.name,
        )

    # Local layout_intent override (local wins on conflict). Only a TIGHTER
    # local tol takes effect -- a looser local tol does not relax the policy.
    local_tol = _local_tol(layout_intent)
    if local_tol is not None and local_tol < eff.tol:
        eff = replace(eff, tol=local_tol, source=f"{eff.source}+local")

    return eff


def policy_for_cell(clay_file, cell_def) -> EffectivePolicy:
    """Resolve the EffectivePolicy for one cell of a parsed ClayFile.

    Reads ``cell_def.flow_decl.policy_ref``, looks it up in
    ``clay_file.policies``, and resolves it against the cell's local
    ``layout_intent`` (via :func:`resolve_policy`). Robust to every missing
    piece (no flow_decl, no policy_ref, unknown policy name, no layout_intent)
    -- returns the defaults in that case.
    """
    flow_decl = getattr(cell_def, "flow_decl", None)
    policy_ref = getattr(flow_decl, "policy_ref", None) if flow_decl else None

    policy_decl = None
    if policy_ref:
        policies = getattr(clay_file, "policies", None) or {}
        policy_decl = policies.get(policy_ref)

    layout_intent = None
    try:
        from .intent import layout_intent_from_cell_def

        layout_intent = layout_intent_from_cell_def(cell_def)
    except Exception:
        layout_intent = None

    return resolve_policy(policy_decl, layout_intent=layout_intent)
