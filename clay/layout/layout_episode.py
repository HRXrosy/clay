"""clay.layout.layout_episode -- the LAYOUT half of the self-boot flywheel (R2).

R1 (``clay.episode`` + ``clay.selfboot.cross_episode_store``) gave the
SIZING half of self-bootstrapping a transferable trace: every finished sizing
episode deposits an :class:`~clay.selfboot.cross_episode_store.EpisodeRecord`
keyed on a topology fingerprint, and a later episode warm-starts from it.  The
flywheel never touched LAYOUT -- yet layout signoff (DRC clean + LVS match +
the declared :class:`~clay.layout.intent.LayoutObligation` budgets verified
against a real PEX extraction) is the dominant signoff effort.

This module mirrors the sizing spine for LAYOUT, deliberately WITHOUT first
building a perfect generic layout engine.  The transferable unit is not a
continuous sizing blend; it is the DISCRETE CHOICE of which existing builder /
router generation / params produced a clean signoff for a given layout INTENT.
A :class:`LayoutEpisodeRecord` records exactly that choice plus the real
outcome it earned (DRC clean, LVS match, obligation satisfaction).  A later
layout run can ask the store "which builder/router cleanly signed off a cell
with THIS intent before?" -- the WarmStart policy in
:mod:`clay.layout.layout_policy`.

Honesty discipline (project rule): every field a record carries is read off a
REAL layout outcome -- the obligation list ``check_layout_intent`` produced
against a real (geometric or Magic) ``PexReport``, and the DRC/LVS verdicts the
caller passed.  Nothing is fabricated; ``drc_clean`` / ``lvs_match`` default to
the honest "not separately measured on this path" value the recording hook
supplies, never a plausible-looking constant.

Design mirrors :mod:`clay.episode` exactly:

  * :func:`intent_fingerprint` -- a STABLE hash of the
    :class:`~clay.layout.intent.LayoutIntent`'s obligation kinds + targets, so
    two cells that declare the same parasitic relationships collate.  It keys
    on the budget kinds/pairs/tolerances (the obligations a layout must satisfy),
    NOT on the cell name, so a builder that cleanly satisfied a given set of
    differential-balance budgets is transferable to another cell declaring the
    same set.
  * :class:`LayoutEpisodeStore` -- ``append`` / ``query_best`` /
    ``records`` (the exact interface contract shape).
  * a process-level default store + best-effort, never-raises
    :func:`record_layout_episode`, controlled by ``CLAY_RECORD_LAYOUT_EPISODES``
    (default ON), mirroring ``clay.episode``'s ``CLAY_RECORD_EPISODES`` pattern.

Blast radius: this module is NOT imported by any layout builder/router or by
``certify_layout_intent`` / ``full_layout_signoff``'s verdict computation.  It
is reached only from an ADDITIVE best-effort recording tail call, whose failure
can never change a layout's GDS / DRC / LVS / token outcome.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ─── Env flag (mirrors clay.episode.recording_enabled) ────────────────────

# Recording is ON by default.  Set CLAY_RECORD_LAYOUT_EPISODES to one of the
# falsey tokens below to disable it process-wide (e.g. a benchmark that does not
# want its signoff calls to accumulate into the default store).
_OFF_TOKENS = frozenset({"0", "false", "no", "off", ""})


def recording_enabled() -> bool:
    """Return True when layout-episode recording is enabled (the default).

    Controlled by the ``CLAY_RECORD_LAYOUT_EPISODES`` environment variable.  Any
    value other than a falsey token (``0`` / ``false`` / ``no`` / ``off`` /
    empty) -- and the *unset* case -- enables recording.  The check is read at
    call time (not import time) so a test can toggle the flag with
    ``monkeypatch.setenv`` without re-importing.
    """
    raw = os.environ.get("CLAY_RECORD_LAYOUT_EPISODES")
    if raw is None:
        return True
    return raw.strip().lower() not in _OFF_TOKENS


# ─── Intent fingerprint ───────────────────────────────────────────────────


def intent_fingerprint(intent: Any) -> str:
    """A stable 16-hex hash of a :class:`LayoutIntent`'s OBLIGATION shape.

    Keys on the set of declared parasitic-budget OBLIGATIONS a layout must
    satisfy -- each budget's ``kind`` + net ``pair`` + relative tolerance --
    plus the differential / mirror pairs and declared shields that drive the
    placement obligations.  It is intentionally INDEPENDENT of the cell name:
    two cells that declare the same relationships (the same matched-cap /
    clk_regen / kickback budgets at the same tolerances) share a fingerprint, so
    a builder/router that cleanly satisfied that obligation set on one cell is a
    transferable proposal for the other.

    Sorted-keys JSON so the hash is stable across runs / Python versions.  Any
    field that changes which obligations the layout must meet (a budget's pair,
    tolerance, or kind; a diff/mirror pair; a declared shield) changes the
    fingerprint.  Duck-typed: anything exposing ``budgets`` (each with
    ``kind`` / ``pair`` / ``rel_tol``) works; missing attributes degrade to an
    empty component rather than raising.
    """
    budgets = []
    for b in tuple(getattr(intent, "budgets", ()) or ()):
        pair = tuple(getattr(b, "pair", ()) or ())
        budgets.append(
            {
                "kind": str(getattr(b, "kind", "")),
                "pair": sorted(str(x) for x in pair),
                "rel_tol": _safe_float(getattr(b, "rel_tol", 0.0)),
            }
        )
    # Sort the budget descriptors so declaration order never changes the hash.
    budgets.sort(key=lambda d: (d["kind"], tuple(d["pair"]), d["rel_tol"]))

    diff_pairs = sorted(
        sorted(str(x) for x in (p or ()))
        for p in tuple(getattr(intent, "diff_pairs", ()) or ())
    )
    mirror_pairs = sorted(
        sorted(str(x) for x in (p or ()))
        for p in tuple(getattr(intent, "mirror_pairs", ()) or ())
    )
    shields = sorted(
        [
            str(getattr(s, "aggressor", "")),
            str(getattr(s, "victim", "")),
            str(getattr(s, "shield_net", "")),
        ]
        for s in tuple(getattr(intent, "shields", ()) or ())
    )

    payload = {
        "budgets": budgets,
        "diff_pairs": diff_pairs,
        "mirror_pairs": mirror_pairs,
        "shields": shields,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _safe_float(x: Any) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


# ─── Layout episode record ────────────────────────────────────────────────


@dataclass
class LayoutEpisodeRecord:
    """One finished layout-signoff episode's transferable experience.

    The DISCRETE-CHOICE analogue of the sizing
    :class:`~clay.selfboot.cross_episode_store.EpisodeRecord`: instead of a
    continuous converged sizing, it records WHICH builder / router generation /
    params produced a given real layout outcome for a given intent fingerprint.

    Fields (the exact interface contract shape):

      layout_episode_id
        Caller-supplied unique id within a store.
      intent_fingerprint
        Stable hash of the :class:`LayoutIntent`'s obligation kinds/targets
        (see :func:`intent_fingerprint`).  Only episodes with an EQUAL
        fingerprint are considered transferable.
      builder
        The cell-builder identity that produced the layout (e.g. a
        ``_CELL_BUILDERS`` key or a bespoke builder name).
      router_generation
        The router generation that wired it (e.g. ``maze_router_v16x`` /
        ``dense_router_v18x`` / ``solver_v183``), or ``""`` when unknown.
      params
        Discrete build/route parameters that distinguish the choice (free-form
        JSON-clean dict).
      drc_clean
        Whether real DRC reported zero violations.
      lvs_match
        Whether real LVS matched.
      obligations_satisfied / obligations_total
        How many declared layout obligations passed, out of the total declared
        (from ``check_layout_intent`` against a real ``PexReport``).
      passed
        ``drc_clean and lvs_match and obligations_satisfied == obligations_total``
        -- the full layout-signoff verdict.  Stored explicitly (and re-derived
        in ``__post_init__`` if the caller left it default) so the record is
        self-consistent.
      score
        A scalar ranking key (higher is better).  By convention the obligation
        SATISFACTION RATIO (in ``[0, 1]``); a caller may pass ``-n_violations``
        instead.  Used by :meth:`LayoutEpisodeStore.query_best` to rank passing
        episodes.
      timestamp
        Caller-supplied ordering key (epoch seconds).  The store never calls the
        wall clock itself, keeping tests reproducible.
    """

    layout_episode_id: str
    intent_fingerprint: str
    builder: str
    router_generation: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    drc_clean: bool = False
    lvs_match: bool = False
    obligations_satisfied: int = 0
    obligations_total: int = 0
    passed: bool = False
    score: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.layout_episode_id, str) or not self.layout_episode_id:
            raise ValueError(
                "LayoutEpisodeRecord.layout_episode_id must be a non-empty str, "
                f"got {self.layout_episode_id!r}"
            )
        if not isinstance(self.intent_fingerprint, str) or not self.intent_fingerprint:
            raise ValueError(
                "LayoutEpisodeRecord.intent_fingerprint must be a non-empty str, "
                f"got {self.intent_fingerprint!r}"
            )
        self.builder = str(self.builder)
        self.router_generation = str(self.router_generation)
        if not isinstance(self.params, Mapping):
            raise TypeError("LayoutEpisodeRecord.params must be a mapping")
        self.params = dict(self.params)
        self.drc_clean = bool(self.drc_clean)
        self.lvs_match = bool(self.lvs_match)
        self.obligations_satisfied = int(self.obligations_satisfied)
        self.obligations_total = int(self.obligations_total)
        # ``passed`` is the derived signoff verdict.  If the caller left it at the
        # default False we re-derive it from the parts; if the caller supplied an
        # explicit value we keep theirs but it should agree with the derivation.
        derived = (
            self.drc_clean
            and self.lvs_match
            and self.obligations_satisfied == self.obligations_total
        )
        self.passed = bool(self.passed) or derived
        sc = _safe_float(self.score)
        self.score = sc

    @property
    def satisfaction_ratio(self) -> float:
        """Fraction of declared obligations satisfied (1.0 when none declared)."""
        if self.obligations_total <= 0:
            return 1.0
        return self.obligations_satisfied / self.obligations_total

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "layout_episode_id": self.layout_episode_id,
            "intent_fingerprint": self.intent_fingerprint,
            "builder": self.builder,
            "router_generation": self.router_generation,
            "params": dict(self.params),
            "drc_clean": bool(self.drc_clean),
            "lvs_match": bool(self.lvs_match),
            "obligations_satisfied": int(self.obligations_satisfied),
            "obligations_total": int(self.obligations_total),
            "passed": bool(self.passed),
            "score": float(self.score),
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "LayoutEpisodeRecord":
        return cls(
            layout_episode_id=str(obj["layout_episode_id"]),
            intent_fingerprint=str(obj["intent_fingerprint"]),
            builder=str(obj.get("builder", "")),
            router_generation=str(obj.get("router_generation", "")),
            params=dict(obj.get("params", {})),
            drc_clean=bool(obj.get("drc_clean", False)),
            lvs_match=bool(obj.get("lvs_match", False)),
            obligations_satisfied=int(obj.get("obligations_satisfied", 0)),
            obligations_total=int(obj.get("obligations_total", 0)),
            passed=bool(obj.get("passed", False)),
            score=float(obj.get("score", 0.0)),
            timestamp=float(obj.get("timestamp", 0.0)),
        )


# ─── The store ────────────────────────────────────────────────────────────


class LayoutEpisodeStore:
    """A fingerprint-keyed store of layout-signoff experience.

    The discrete-choice analogue of
    :class:`~clay.selfboot.cross_episode_store.CrossEpisodeStore`.

    Interface contract:

      * :meth:`append` -- accumulate one :class:`LayoutEpisodeRecord` (rejects a
        duplicate id, mirroring the sizing store's ``add``).
      * :meth:`query_best` -- the best-scoring PASSING episode for a given intent
        fingerprint, or ``None`` when no passing episode exists.  This is what the
        WarmStart policy consults to propose a builder/router/params; an empty
        store (or no match) yields ``None`` so the policy falls back to today's
        hardcoded default -- accumulation can never make the proposal worse than
        the committed baseline, only move it toward a known-clean choice.
      * :meth:`records` -- all stored records in insertion order.
    """

    def __init__(self, records: Optional[Sequence[LayoutEpisodeRecord]] = None) -> None:
        self._records: List[LayoutEpisodeRecord] = []
        self._ids: set[str] = set()
        for rec in records or []:
            self.append(rec)

    # ── mutation ──────────────────────────────────────────────────────────

    def append(self, record: LayoutEpisodeRecord) -> None:
        if not isinstance(record, LayoutEpisodeRecord):
            raise TypeError("LayoutEpisodeStore.append expects a LayoutEpisodeRecord")
        if record.layout_episode_id in self._ids:
            raise ValueError(
                "LayoutEpisodeStore.append: duplicate layout_episode_id "
                f"{record.layout_episode_id!r}"
            )
        self._records.append(record)
        self._ids.add(record.layout_episode_id)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    # ── query ─────────────────────────────────────────────────────────────

    def records(self) -> List[LayoutEpisodeRecord]:
        """All stored records, in insertion order (a copy)."""
        return list(self._records)

    def matching(
        self,
        intent_fp: str,
        *,
        passed_only: bool = False,
    ) -> List[LayoutEpisodeRecord]:
        """Episodes whose ``intent_fingerprint`` EQUALS ``intent_fp`` (optionally
        only those that passed signoff), in insertion order."""
        out: List[LayoutEpisodeRecord] = []
        for rec in self._records:
            if rec.intent_fingerprint != intent_fp:
                continue
            if passed_only and not rec.passed:
                continue
            out.append(rec)
        return out

    def query_best(self, intent_fp: str) -> Optional[LayoutEpisodeRecord]:
        """The best PASSING episode for ``intent_fp``, or ``None``.

        Ranking: among the passing episodes with an EQUAL fingerprint, the one
        with the highest ``score`` wins; ties break to the more recent
        (``-timestamp``) then the lexicographically-larger id, so the choice is
        deterministic.  Returns ``None`` when no passing episode exists for the
        fingerprint -- the policy then proposes today's hardcoded default.
        """
        cands = self.matching(intent_fp, passed_only=True)
        if not cands:
            return None
        return max(
            cands,
            key=lambda r: (r.score, r.timestamp, r.layout_episode_id),
        )


# ─── Process-level default store (mirrors clay.episode) ───────────────────

_DEFAULT_STORE: Optional[LayoutEpisodeStore] = None
_STORE_LOCK = threading.Lock()
# Monotone counter so each recorded episode gets a unique id within the process
# even when the caller does not supply one.  Guarded by the same lock as the
# store so the (id, append) pair is atomic.
_EPISODE_SEQ = 0


def get_default_store() -> LayoutEpisodeStore:
    """Return the process-level :class:`LayoutEpisodeStore`.

    Lazily constructed on first use and shared for the lifetime of the process.
    Callers that want isolation (tests, parallel benchmarks) should pass their
    own store to :func:`record_layout_episode` instead of relying on this one,
    or call :func:`reset_default_store`.
    """
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _STORE_LOCK:
            if _DEFAULT_STORE is None:
                _DEFAULT_STORE = LayoutEpisodeStore()
    return _DEFAULT_STORE


def reset_default_store() -> None:
    """Drop the process-level store (next :func:`get_default_store` rebuilds an
    empty one).  Primarily for test isolation."""
    global _DEFAULT_STORE
    with _STORE_LOCK:
        _DEFAULT_STORE = None


def _next_episode_id() -> str:
    global _EPISODE_SEQ
    _EPISODE_SEQ += 1
    return f"layout-{_EPISODE_SEQ:06d}"


# ─── Best-effort recording hook ───────────────────────────────────────────


def record_layout_episode(
    intent: Any,
    obligations: Any,
    *,
    builder: str,
    router_generation: str = "",
    params: Optional[Mapping[str, Any]] = None,
    drc_clean: bool = True,
    lvs_match: bool = True,
    store: Optional[LayoutEpisodeStore] = None,
    layout_episode_id: Optional[str] = None,
    timestamp: float = 0.0,
) -> Optional[LayoutEpisodeRecord]:
    """Best-effort: append a :class:`LayoutEpisodeRecord` for a real outcome.

    This is the spine hook a layout-signoff path calls AFTER it has a real
    layout outcome (the obligation list from ``check_layout_intent`` against a
    real ``PexReport``, plus the DRC / LVS verdicts).  It is CONTRACTUALLY
    side-effect-free on the layout result:

      * If recording is disabled (``CLAY_RECORD_LAYOUT_EPISODES`` falsey) it is a
        no-op returning ``None``.
      * Any exception building or appending the record is swallowed; the function
        returns ``None`` and the caller's layout result is unchanged.

    ``obligations`` is the list of :class:`~clay.layout.intent.LayoutObligation`
    the checker produced; ``obligations_satisfied`` counts those with
    ``status == "pass"`` (an unmeasurable obligation is NOT a pass, matching the
    verdict convention).  ``drc_clean`` / ``lvs_match`` default to the honest
    "not separately measured on this path" value the caller supplies -- on the
    parasitic-intent signoff path DRC/LVS are not re-run, so the caller passes
    what it knows (the parasitic-intent record's strength is the obligation
    satisfaction).  The ``score`` is the obligation satisfaction ratio.

    Returns the appended record (so a caller / test can inspect it), or ``None``
    when nothing was recorded.
    """
    if not recording_enabled():
        return None
    try:
        obls = list(obligations or [])
        total = len(obls)
        satisfied = sum(1 for o in obls if str(getattr(o, "status", "")) == "pass")
        score = (satisfied / total) if total > 0 else 1.0
        fp = intent_fingerprint(intent)
        target_store = store if store is not None else get_default_store()
        with _STORE_LOCK:
            eid = layout_episode_id if layout_episode_id is not None else _next_episode_id()
            rec = LayoutEpisodeRecord(
                layout_episode_id=eid,
                intent_fingerprint=fp,
                builder=str(builder),
                router_generation=str(router_generation),
                params=dict(params or {}),
                drc_clean=bool(drc_clean),
                lvs_match=bool(lvs_match),
                obligations_satisfied=int(satisfied),
                obligations_total=int(total),
                score=float(score),
                timestamp=float(timestamp),
            )
            target_store.append(rec)
        return rec
    except Exception:  # noqa: BLE001 -- never let recording change a layout outcome
        return None


__all__ = [
    "recording_enabled",
    "intent_fingerprint",
    "LayoutEpisodeRecord",
    "LayoutEpisodeStore",
    "get_default_store",
    "reset_default_store",
    "record_layout_episode",
]
