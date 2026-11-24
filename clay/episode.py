"""clay.episode — process-level episode-recording spine (Episode-Spine R1).

The self-bootstrapping north star is that every design episode leaves a
transferable trace: a later episode warm-starts from what earlier ones
learned.  Two pieces of that spine already exist:

  * ``clay.distance_to_sat.verify_with_distance`` — produces a
    distance-bearing verdict (continuous satisfaction loss + per-spec
    distances) every time a cell is verified, AND
  * ``clay.selfboot.cross_episode_store`` — a persistent,
    fingerprint-keyed store of :class:`EpisodeRecord` that answers
    ``warm_start(fingerprint, spec)``.

What was missing is the LINK: nothing in the production verify path wrote
an :class:`EpisodeRecord` back into a store.  ``run_episode_and_record``
exists but only the self-boot flywheel and its own tests call it — a
generic ``verify_with_distance`` call left no trace.

This module is that link, and ONLY that link.  It is deliberately tiny,
default-on, and side-effect-free on the verdict:

  * :func:`get_default_store` — the process-level
    :class:`CrossEpisodeStore` every ``verify_with_distance`` appends to
    (lazily constructed, shared within the process).
  * :func:`record_episode` — best-effort: turn a
    :class:`DistanceVerifyResult` into an :class:`EpisodeRecord` and
    append it to a store.  NEVER raises; any failure is swallowed and the
    caller's verdict is unchanged.
  * :func:`recording_enabled` — reads the ``CLAY_RECORD_EPISODES`` env
    flag (default ON).  A caller (or a test) can disable recording
    process-wide by setting ``CLAY_RECORD_EPISODES=0``.

Honesty discipline (project rule): every number an EpisodeRecord written
here carries comes from the verdict the verifier actually produced — the
measured per-spec distances and the aggregate satisfaction loss.  Nothing
is fabricated.  When a measure was missing the recorded distance is the
honest ``inf`` (stored as ``None`` so the JSON round-trips), never a
plausible-looking placeholder.

Blast radius: this module is NOT imported by ``verify()`` or
``VerifyResult``.  It is reached only from the additive
``verify_with_distance`` path, and only as a best-effort tail call whose
failure cannot change a verdict.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from .selfboot.cross_episode_store import (
    CrossEpisodeStore,
    EpisodeRecord,
    TopologyFingerprint,
)


# ─── Env flag ─────────────────────────────────────────────────────────

# Recording is ON by default.  Set CLAY_RECORD_EPISODES to one of the
# falsey tokens below to disable it process-wide (e.g. a benchmark that
# does not want its verify calls to accumulate into the default store).
_OFF_TOKENS = frozenset({"0", "false", "no", "off", ""})


def recording_enabled() -> bool:
    """Return True when episode recording is enabled (the default).

    Controlled by the ``CLAY_RECORD_EPISODES`` environment variable.
    Any value other than a falsey token (``0`` / ``false`` / ``no`` /
    ``off`` / empty) — and the *unset* case — enables recording.  The
    check is intentionally read at call time (not import time) so a test
    can toggle the flag with ``monkeypatch.setenv`` without re-importing.
    """
    raw = os.environ.get("CLAY_RECORD_EPISODES")
    if raw is None:
        return True
    return raw.strip().lower() not in _OFF_TOKENS


# ─── Process-level default store ──────────────────────────────────────

_DEFAULT_STORE: Optional[CrossEpisodeStore] = None
_STORE_LOCK = threading.Lock()
# Monotone counter so each recorded episode gets a unique id within the
# process even when the caller does not supply one.  Guarded by the same
# lock as the store so the (id, add) pair is atomic.
_EPISODE_SEQ = 0


def get_default_store() -> CrossEpisodeStore:
    """Return the process-level :class:`CrossEpisodeStore`.

    Lazily constructed on first use and shared for the lifetime of the
    process.  Callers that want isolation (tests, parallel benchmarks)
    should pass their own store to :func:`record_episode` instead of
    relying on this one, or call :func:`reset_default_store`.
    """
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _STORE_LOCK:
            if _DEFAULT_STORE is None:
                _DEFAULT_STORE = CrossEpisodeStore()
    return _DEFAULT_STORE


def reset_default_store() -> None:
    """Drop the process-level store (next :func:`get_default_store`
    rebuilds an empty one).  Primarily for test isolation."""
    global _DEFAULT_STORE
    with _STORE_LOCK:
        _DEFAULT_STORE = None


def _next_episode_id() -> str:
    global _EPISODE_SEQ
    _EPISODE_SEQ += 1
    return f"verify-{_EPISODE_SEQ:06d}"


# ─── Verdict → EpisodeRecord ──────────────────────────────────────────


def _fingerprint_for_verdict(verdict: Any, *, pdk: str) -> TopologyFingerprint:
    """Derive a best-effort fingerprint from a DistanceVerifyResult.

    A generic ``verify_with_distance`` verdict does not carry sizing
    knobs (it is a measure-side artifact), so the owned-knob set is
    empty — the recorded episode is still a useful spec/outcome trace
    even though it cannot seed a warm-start blend (the store skips
    empty-converged-sizing records when blending, which is correct).

    ``topology_id`` is the cell name; ``device_class`` is a coarse tag
    so two episodes on the same cell collate.  Both are read off the
    underlying ``VerifyResult`` / verdict — never invented.
    """
    cell_name = str(getattr(verdict, "cell_name", "?") or "?")
    return TopologyFingerprint(
        topology_id=cell_name,
        device_class="cell",
        owned_knobs=(),
        pdk=str(pdk),
    )


def episode_record_from_verdict(
    verdict: Any,
    *,
    episode_id: Optional[str] = None,
    pdk: str = "sky130",
    iterations: int = 1,
    timestamp: float = 0.0,
) -> EpisodeRecord:
    """Build an :class:`EpisodeRecord` from a ``DistanceVerifyResult``.

    The record captures exactly what the verdict measured:

      * ``spec``        — the target value per spec field (the contract
                          the episode aimed at).
      * ``role_feedback`` — per-spec distance-to-sat at the measured
                          point (the honest continuous signal), plus the
                          measured values and the outcome flag.
      * ``best_loss``   — the aggregate satisfaction loss (``None`` when
                          it was non-finite, i.e. a measure was missing).
      * ``passed``      — whether every spec was satisfied.

    No simulator number is fabricated; every value is read off the
    verdict the verifier produced.
    """
    if episode_id is None:
        episode_id = _next_episode_id()
    fp = _fingerprint_for_verdict(verdict, pdk=pdk)

    violations = list(getattr(verdict, "violations", []) or [])
    spec: dict[str, float] = {}
    per_spec_distance: dict[str, Any] = {}
    measures: dict[str, float] = {}
    for v in violations:
        name = getattr(v, "spec_name", None)
        if name is None:
            continue
        name = str(name)
        target = getattr(v, "target", None)
        if isinstance(target, (int, float)):
            ft = float(target)
            # EpisodeRecord.spec rejects non-finite values; skip those.
            if ft == ft and ft not in (float("inf"), float("-inf")):
                spec[name] = ft
        dist = getattr(v, "distance", None)
        if isinstance(dist, (int, float)):
            fd = float(dist)
            per_spec_distance[name] = (
                None if fd in (float("inf"), float("-inf")) or fd != fd else fd
            )
        measured = getattr(v, "measured", None)
        if isinstance(measured, (int, float)):
            fm = float(measured)
            if fm == fm and fm not in (float("inf"), float("-inf")):
                measures[name] = fm

    raw_loss = getattr(verdict, "satisfaction_loss", None)
    best_loss: Optional[float] = None
    if isinstance(raw_loss, (int, float)):
        fl = float(raw_loss)
        if fl == fl and fl not in (float("inf"), float("-inf")):
            best_loss = fl

    gradient = getattr(verdict, "gradient", None)
    role_feedback: dict[str, Any] = {
        "per_metric_distance": per_spec_distance,
        "measures": measures,
        "satisfaction_loss": (
            None
            if not isinstance(raw_loss, (int, float))
            or float(raw_loss) != float(raw_loss)
            or float(raw_loss) in (float("inf"), float("-inf"))
            else float(raw_loss)
        ),
        "source": "verify_with_distance",
    }
    if isinstance(gradient, dict) and gradient:
        # Carry the adjoint direction as opaque context (the store treats
        # role_feedback as free-form).  Non-finite components are dropped
        # by _json_safe on serialization.
        role_feedback["satisfaction_gradient"] = {
            str(k): float(v)
            for k, v in gradient.items()
            if isinstance(v, (int, float))
        }

    return EpisodeRecord(
        episode_id=episode_id,
        fingerprint=fp,
        spec=spec,
        converged_sizing={},
        role_feedback=role_feedback,
        best_loss=best_loss,
        passed=bool(getattr(verdict, "satisfied", False)),
        iterations=int(iterations),
        timestamp=float(timestamp),
    )


def record_episode(
    verdict: Any,
    *,
    store: Optional[CrossEpisodeStore] = None,
    episode_id: Optional[str] = None,
    pdk: str = "sky130",
    iterations: int = 1,
    timestamp: float = 0.0,
) -> Optional[EpisodeRecord]:
    """Best-effort: append an :class:`EpisodeRecord` for ``verdict``.

    This is the spine hook ``verify_with_distance`` calls.  It is
    CONTRACTUALLY side-effect-free on the verdict:

      * If recording is disabled (``CLAY_RECORD_EPISODES`` falsey) it is
        a no-op returning ``None``.
      * Any exception building or appending the record is swallowed; the
        function returns ``None`` and the caller's verdict is unchanged.

    Returns the :class:`EpisodeRecord` that was appended (so a caller /
    test can inspect it), or ``None`` when nothing was recorded.

    Args:
      verdict: a ``DistanceVerifyResult`` (duck-typed: anything exposing
        ``cell_name`` / ``violations`` / ``satisfaction_loss`` /
        ``satisfied`` works).
      store: target store.  Defaults to the process-level
        :func:`get_default_store`.
      episode_id: explicit id (auto-generated unique id when None).
      pdk / iterations / timestamp: passed through to the record.
    """
    if not recording_enabled():
        return None
    try:
        target_store = store if store is not None else get_default_store()
        with _STORE_LOCK:
            rec = episode_record_from_verdict(
                verdict,
                episode_id=episode_id,
                pdk=pdk,
                iterations=iterations,
                timestamp=timestamp,
            )
            target_store.add(rec)
        return rec
    except Exception:  # noqa: BLE001 — never let recording change a verdict
        return None


__all__ = [
    "recording_enabled",
    "get_default_store",
    "reset_default_store",
    "episode_record_from_verdict",
    "record_episode",
]
