"""Place/route learning memory (v3) -- the "smarter with use" flywheel for the
layout convergence loops.

After a converged place / route run the engine records ``(intent_context, loop
knobs, outcome)``; for a NEW cell whose intent-context MATCHES a prior one, it
retrieves a blended seed to initialise the loop's knobs (fewer iterations / a
better initial layout). Every converged design makes the engine smarter -- the
cross-episode flywheel, applied to the layout loops.

Compose, don't duplicate
=========================
This is a THIN ADAPTER over :mod:`clay.selfboot.cross_episode_store`. The
persistence, fingerprint-equality ranking, and inverse-distance warm-start blend
are reused verbatim from :class:`CrossEpisodeStore`; this module adds only:

  * :func:`intent_context_fingerprint` -- a STABLE :class:`TopologyFingerprint`
    over a layout intent-context (cell + phase + the structural intent digest +
    the policy strategy fields), so two layout runs blend only when their
    intent-context matches exactly;
  * :class:`LayoutMemory` -- a wrapper holding a store + an optional disk path,
    with ``record`` / ``seed`` / ``save`` / ``load`` keyed to the layout knob
    maps (``grid_nm`` / ``rel_tol`` for placement; ``spacing_extra_nm`` for
    routing).

Honesty discipline (project rule): every seed this returns is the inverse-spec-
distance blend of PRIOR converged runs' real knobs -- nothing is fabricated. An
empty / non-matching history returns ``None`` and the loop falls back to its
engine defaults byte-for-byte, so accumulation can never make a cold run worse.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from clay.selfboot.cross_episode_store import (
    CrossEpisodeStore,
    EpisodeRecord,
    TopologyFingerprint,
)

# Owned-knob sets per layout phase (the keys the seed blends + the loops thread).
# Placement converges (grid_nm, rel_tol); routing converges (spacing_extra_nm).
_PLACE_KNOBS = ("grid_nm", "rel_tol")
_ROUTE_KNOBS = ("spacing_extra_nm",)

_PHASE_KNOBS: Dict[str, Sequence[str]] = {
    "place": _PLACE_KNOBS,
    "route": _ROUTE_KNOBS,
}

_DEVICE_CLASS = "layout"


# ─── intent-context fingerprint ───────────────────────────────────────────────


def _structural_digest(layout_intent: Any) -> str:
    """A canonical, deterministic digest of a layout intent's STRUCTURE.

    Built from the sorted matched-pair labels (``diff_pairs``), the sorted
    common-centroid / mirror group memberships (``mirror_pairs`` + any
    ``place_groups`` device sets), so two cells with the same structural intent
    produce the same digest regardless of the order the pairs were declared.
    Robust to a duck-typed intent missing any field (treated as empty).
    """
    diff_pairs = getattr(layout_intent, "diff_pairs", ()) or ()
    mirror_pairs = getattr(layout_intent, "mirror_pairs", ()) or ()
    place_groups = getattr(layout_intent, "place_groups", ()) or ()

    # Matched pairs: sort each pair's nets + sort the pair set (order-insensitive).
    diff_sig = sorted(
        tuple(sorted(str(n) for n in pair)) for pair in diff_pairs
    )
    cc_sig = sorted(
        tuple(sorted(str(n) for n in pair)) for pair in mirror_pairs
    )
    group_sig = sorted(
        (
            str(getattr(g, "group", "")),
            tuple(sorted(str(d) for d in (getattr(g, "devices", ()) or ()))),
        )
        for g in place_groups
    )
    payload = repr((diff_sig, cc_sig, group_sig))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _policy_strategy_sig(effective_policy: Any) -> str:
    """A canonical signature of the policy's place/route STRATEGY fields.

    Reads ``symmetry`` / ``pack`` / ``route_tracks`` off the EffectivePolicy
    (defaulting gracefully when absent). Two contexts with different placement
    strategy must NOT blend (a mirror_x placement's grid is not a free-pack
    placement's), so the strategy participates in the fingerprint.
    """
    symmetry = str(getattr(effective_policy, "symmetry", ""))
    pack = str(getattr(effective_policy, "pack", ""))
    route_tracks = str(getattr(effective_policy, "route_tracks", ""))
    return f"{symmetry}|{pack}|{route_tracks}"


def intent_context_fingerprint(
    cell_name: str,
    effective_policy: Any,
    layout_intent: Any,
    *,
    phase: str,
    pdk: str = "gf180mcu",
) -> TopologyFingerprint:
    """Build a STABLE :class:`TopologyFingerprint` for a layout intent-context.

    ``topology_id`` is a deterministic signature of the intent context: the cell
    name + the phase (``"place"`` / ``"route"``) + a canonical digest of the
    structural intent (sorted matched-pair labels + sorted common-centroid group
    memberships) + the policy strategy fields (symmetry / pack / route_tracks).
    Two layout runs blend (in ``warm_start``) ONLY when this fingerprint is equal
    -- i.e. same cell, same phase, same structural intent, same strategy.

    ``owned_knobs`` is the phase's knob set (``("grid_nm", "rel_tol")`` for
    placement; ``("spacing_extra_nm",)`` for routing); ``device_class`` is
    ``"layout"``; ``pdk`` carries the process family (a gf180 grid does not
    transfer to sky130).
    """
    if phase not in _PHASE_KNOBS:
        raise ValueError(
            f"intent_context_fingerprint: phase must be one of "
            f"{sorted(_PHASE_KNOBS)}, got {phase!r}"
        )
    struct = _structural_digest(layout_intent)
    strat = _policy_strategy_sig(effective_policy)
    topology_id = f"{cell_name}|{phase}|{struct}|{strat}"
    return TopologyFingerprint(
        topology_id=topology_id,
        device_class=_DEVICE_CLASS,
        owned_knobs=tuple(_PHASE_KNOBS[phase]),
        pdk=str(pdk),
    )


# ─── the layout memory adapter ────────────────────────────────────────────────


class LayoutMemory:
    """A thin wrapper over :class:`CrossEpisodeStore` for the layout loops.

    Holds a store + an optional disk path. ``record`` writes a converged (or
    near-miss) run's knobs; ``seed`` retrieves the inverse-distance blend of
    matching PRIOR converged runs' knobs for a fingerprint (or ``None`` when the
    history has no match). ``save`` / ``load`` delegate to the store.
    """

    def __init__(
        self,
        store: Optional[CrossEpisodeStore] = None,
        *,
        path: Optional[Path | str] = None,
    ) -> None:
        self._store = store if store is not None else CrossEpisodeStore()
        self._path = Path(path) if path is not None else None
        self._counter = 0

    @property
    def store(self) -> CrossEpisodeStore:
        return self._store

    @property
    def path(self) -> Optional[Path]:
        return self._path

    # ── record ────────────────────────────────────────────────────────────────

    def record(
        self,
        fingerprint: TopologyFingerprint,
        knobs: Mapping[str, float],
        *,
        converged: bool,
        iters: int,
        residual: float,
        target: Optional[Mapping[str, float]] = None,
    ) -> EpisodeRecord:
        """Record one finished layout run's experience.

        Builds an :class:`EpisodeRecord` (``converged_sizing`` = the loop knobs,
        ``best_loss`` = the final residual, ``passed`` = converged, ``iterations``
        = iters, ``spec`` = ``target`` or ``{}``), adds it to the store, and
        persists to the disk path when one is set. Returns the record.

        All numeric values are coerced to finite floats (the store's contract);
        a non-finite residual is recorded as ``best_loss=None`` (no usable loss).
        """
        knob_map = {str(k): float(v) for k, v in dict(knobs).items()}
        self._counter += 1
        episode_id = f"{fingerprint.topology_id}#{self._counter}"
        rec = EpisodeRecord(
            episode_id=episode_id,
            fingerprint=fingerprint,
            spec={str(k): float(v) for k, v in dict(target or {}).items()},
            converged_sizing=knob_map,
            role_feedback={"iterations": int(iters), "converged": bool(converged)},
            best_loss=float(residual),
            passed=bool(converged),
            iterations=int(iters),
            timestamp=float(time.time()),
        )
        self._store.add(rec)
        if self._path is not None:
            self._store.save(self._path)
        return rec

    # ── seed ────────────────────────────────────────────────────────────────────

    def seed(
        self,
        fingerprint: TopologyFingerprint,
        *,
        target: Optional[Mapping[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """Retrieve the blended converged knobs (the seed) for a fingerprint.

        Runs :meth:`CrossEpisodeStore.warm_start` with ``passed_only=True`` (only
        CONVERGED prior runs seed) and returns its blended ``sizing`` -- the
        inverse-spec-distance weighted mean of the matching episodes' converged
        knobs over the fingerprint's owned knobs. Returns ``None`` when no
        matching converged history exists (a cold query): the caller then falls
        back to the engine defaults.
        """
        query = self._store.warm_start(
            fingerprint,
            dict(target or {}),
            passed_only=True,
        )
        if query.is_cold or not query.sizing:
            return None
        return {str(k): float(v) for k, v in query.sizing.items()}

    # ── persistence (delegate to the store) ─────────────────────────────────────

    def save(self, path: Optional[Path | str] = None) -> None:
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("LayoutMemory.save: no path given and none configured")
        self._store.save(target)
        self._path = target

    @classmethod
    def load(cls, path: Path | str) -> "LayoutMemory":
        store = CrossEpisodeStore.load(path)
        return cls(store, path=path)


__all__ = [
    "intent_context_fingerprint",
    "LayoutMemory",
]
