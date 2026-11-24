"""clay.layout.layout_policy -- WarmStart policy for LAYOUT builder/router choice.

The LAYOUT half of self-bootstrapping (R2).  R1 learned over SIZING; this policy
learns over the DISCRETE layout CHOICE -- which existing builder / router
generation / params to pick for a given layout intent -- WITHOUT first building a
perfect generic engine.

The policy is an INSTANCE of the same trusted-oracle / untrusted-proposal split
the sizing flywheel proved (``formal/lean/Clay/EpisodeSoundness.lean``): the
trusted base is the layout SIGNOFF ORACLE (real DRC / LVS / PEX obligations);
this policy is the UNTRUSTED proposal that can only suggest which committed
builder to try.  It can never sign off a layout itself, so a wrong proposal is
caught by the oracle exactly the way a wrong sizing guess is.

DEFAULT-PRESERVING CONTRACT (the load-bearing property): when the store has no
relevant PASSING episode for an intent's fingerprint, :func:`propose_layout_choice`
returns :data:`DEFAULT_LAYOUT_PROPOSAL` -- the sentinel meaning "use today's
hardcoded choice".  Its ``builder`` / ``router_generation`` / ``params`` are the
neutral "unspecified -> caller keeps its committed default" markers, so a caller
that consults the policy and finds ``is_default`` makes the SAME build call it
makes today.  Nothing about a layout's GDS / DRC / LVS output changes by default;
the policy only diverges from today's behavior once REAL experience (a passing
episode with this fingerprint) exists, and even then only as advice a caller may
follow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .layout_episode import (
    LayoutEpisodeRecord,
    LayoutEpisodeStore,
    get_default_store,
)


@dataclass(frozen=True)
class LayoutProposal:
    """A WarmStart proposal of which builder / router / params to use.

    ``is_default`` is True for the sentinel :data:`DEFAULT_LAYOUT_PROPOSAL` --
    the policy found NO relevant passing experience, so the caller must use its
    committed hardcoded choice (the default layout path is byte-identical to
    today).  When ``is_default`` is False the proposal names the builder /
    router / params of the best-scoring passing episode for the intent.

    ``source_episode_id`` / ``score`` carry the provenance of a learned proposal
    (None / 0.0 for the default sentinel) so a caller can log WHY the policy
    diverged from the default.
    """

    builder: Optional[str]
    router_generation: Optional[str]
    params: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = True
    source_episode_id: Optional[str] = None
    score: float = 0.0

    @classmethod
    def from_episode(cls, rec: LayoutEpisodeRecord) -> "LayoutProposal":
        """Build a (non-default) proposal from a passing episode record."""
        return cls(
            builder=rec.builder,
            router_generation=rec.router_generation,
            params=dict(rec.params),
            is_default=False,
            source_episode_id=rec.layout_episode_id,
            score=float(rec.score),
        )


# The sentinel returned when the store has no relevant passing episode.  Its
# ``builder`` / ``router_generation`` are None ("unspecified -> caller keeps its
# committed default") and ``is_default`` is True.  A caller that receives this
# makes the SAME build/route call it makes today, so the default layout output is
# unchanged.  Frozen + module-level so identity (`proposal is DEFAULT_LAYOUT_PROPOSAL`)
# is a valid "did the policy learn anything?" check.
DEFAULT_LAYOUT_PROPOSAL = LayoutProposal(
    builder=None,
    router_generation=None,
    params={},
    is_default=True,
    source_episode_id=None,
    score=0.0,
)


def propose_layout_choice(
    intent_fingerprint: str,
    *,
    store: Optional[LayoutEpisodeStore] = None,
) -> LayoutProposal:
    """Propose a builder / router / params for an intent fingerprint.

    WarmStart semantics, mirroring the sizing store's
    ``warm_start(fingerprint, spec)``:

      * When ``store`` has a PASSING episode whose ``intent_fingerprint`` equals
        ``intent_fingerprint``, return a proposal carrying the best-scoring such
        episode's builder / router / params (``is_default == False``).
      * Otherwise -- empty store, or no passing episode for this fingerprint --
        return :data:`DEFAULT_LAYOUT_PROPOSAL` (``is_default == True``), the
        sentinel telling the caller to keep its committed hardcoded choice.  This
        is the DEFAULT-PRESERVING guarantee: with no experience the proposal is
        the today-behavior sentinel, so nothing changes by default.

    ``store`` defaults to the process-level
    :func:`clay.layout.layout_episode.get_default_store`.  The lookup is read-only
    (it never mutates the store) and never raises on a missing match -- it simply
    returns the default sentinel.
    """
    target_store = store if store is not None else get_default_store()
    best = target_store.query_best(str(intent_fingerprint))
    if best is None:
        return DEFAULT_LAYOUT_PROPOSAL
    return LayoutProposal.from_episode(best)


__all__ = [
    "LayoutProposal",
    "DEFAULT_LAYOUT_PROPOSAL",
    "propose_layout_choice",
]
