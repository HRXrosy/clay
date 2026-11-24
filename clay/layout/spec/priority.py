"""Intent PRIORITY axis -- tier + superiority data model (STEP 1).

This module introduces the priority axis: the orthogonal companion to the
hardness axis already carried by :class:`clay.layout.spec.packs.SpecRule`.

Two orthogonal axes
-------------------
    hardness  -- a property of ONE obligation: does violating it gate? can the
                 oracle observe it? is the capability present? (-> 5 states).
    priority  -- a RELATION BETWEEN obligations: when two obligations conflict,
                 which one yields? (-> tier lexicographic order + within-tier
                 defeasible superiority).

STEP 1 is annotation only: this module supplies the tier taxonomy and a pure
derivation function ``tier_for``. It changes NO verdict behavior; there is no
resolver and nothing is ever marked yielded yet. The resolver (STEP 3+) builds
on the invariant locked here: a physical-law obligation always derives to the
LAW tier, the floor of the order, so it can never later be marked yielded.

Pure stdlib, zero coordinates. Tier classification is a DEFAULT TABLE derived
from hardness / checker / kind; a rule (or, later, a language-level
``@priority`` declaration) may declare ``tier`` explicitly to override it. Every
override therefore carries provenance and is auditable -- tier is a declaration,
never a hardcode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


# The four declarable tiers, in lexicographic (highest-priority-first) order.
# Index 0 == LAW is the floor of the order: nothing outranks a physical law.
VALID_TIER = ("LAW", "CORRECTNESS", "QUALITY", "OBJECTIVE")


@dataclass(frozen=True)
class Priority:
    """The priority annotation of one obligation.

    Fields
    ------
    tier : str
        One of :data:`VALID_TIER`. The cross-tier lexicographic rank.
    superior_to : frozenset
        Within-tier defeasible-superiority out-edges: the obligation keys this
        obligation defeats (a partial order, not a total order). Empty by
        default -- absence of a declared superiority is honest, not an implicit
        win-by-arrival.
    """

    tier: str
    superior_to: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIER:
            raise ValueError(
                f"invalid tier {self.tier!r}; expected one of {list(VALID_TIER)}"
            )


# Default tier derived from a rule's hardness class. This is a TABLE, not a
# hardcode: a rule may declare ``tier`` to override it.
#
#   hard            -> CORRECTNESS by default; promoted to LAW when the rule's
#                      checker is a physical-law checker (see below).
#   quantified_soft -> QUALITY (quantified but the degree is negotiable).
#   qualitative_soft-> QUALITY (advisory by nature).
#   unsupported     -> None: a capability gap is ORTHOGONAL to tier; it does not
#                      enter the lexicographic order at all.
_DEFAULT_TIER = {
    "hard": "CORRECTNESS",
    "quantified_soft": "QUALITY",
    # the F1 gating geometric-quality class enters the priority axis at the
    # QUALITY tier (a gating tier, not advisory) so banding cannot silently
    # override a geometric-fidelity obligation (F1 (3) / plan section 2.4).
    "fidelity": "QUALITY",
    "qualitative_soft": "QUALITY",
    "unsupported": None,
}


# Hard obligations whose checker is one of these are promoted from CORRECTNESS
# to LAW: violating them yields a physically illegal layout, not merely an
# intent-infidelity. This is the set the future resolver's law-never-yielded
# guard keys on.
_PHYSICAL_LAW_CHECKERS = frozenset(
    {"drc", "lvs_connectivity", "rail_isolation", "check_shorts"}
)


# Default tier for net-level LayoutObligation objects, keyed by their ``kind``.
# Net-level obligations carry no SpecRule, so their tier is derived from kind
# (with a future language-level ``@priority`` declaration able to override).
_DEFAULT_TIER_BY_KIND = {
    # LAW: violating it yields a physically illegal layout. ring_clearance is
    # LAW-derived -- if two guard rings overlap their rails merge at LVS (the
    # dense x ring P4 fault), so insufficient ring clearance is a rail short, a
    # physical-law violation, not merely a quality regression.
    "ring_clearance": "LAW",
    # CORRECTNESS: intent fidelity that must hold.
    "matched_cc_balance": "CORRECTNESS",
    "body_tie": "CORRECTNESS",
    "shield": "CORRECTNESS",
    "clk_regen": "CORRECTNESS",
    "kickback": "CORRECTNESS",
    "order": "CORRECTNESS",
    "abut": "CORRECTNESS",
    "fix": "CORRECTNESS",
    "equal_size": "CORRECTNESS",
    # QUALITY: degree-negotiable quality that may gracefully degrade.
    "lde_budget": "QUALITY",
    "parasitic_balance": "QUALITY",
    "parasitic_cap": "QUALITY",
    "common_centroid": "QUALITY",
    "interdig": "QUALITY",
    "match_length": "QUALITY",
    "wire_width": "QUALITY",
    "density": "QUALITY",
    # OBJECTIVE: optimization targets, constrained by the three tiers above.
    "place_group_realized": "OBJECTIVE",
}


def _tier_from_hardness_and_checker(
    hardness: str, checker: Optional[str]
) -> Optional[str]:
    """Derive a tier from a (hardness, checker) pair using the default tables."""
    if hardness == "hard" and checker in _PHYSICAL_LAW_CHECKERS:
        return "LAW"
    return _DEFAULT_TIER.get(hardness, "CORRECTNESS")


def tier_for(rule_or_view) -> Optional[str]:
    """Derive the tier of a rule or obligation view.

    Resolution order (all auditable):
      1. an explicit ``tier`` declared on the object (override; must be valid);
      2. for SpecRule-like objects: hardness + checker via the default tables,
         with physical-law checkers promoted to LAW;
      3. for net-level obligation views: ``kind`` via _DEFAULT_TIER_BY_KIND.

    Returns None only for ``unsupported`` hardness, which is orthogonal to tier
    (a capability gap, not a priority rank).
    """
    # 1. explicit declaration wins (provenance-bearing override).
    declared = getattr(rule_or_view, "tier", None)
    if declared is not None:
        if declared not in VALID_TIER:
            raise ValueError(
                f"invalid declared tier {declared!r}; expected one of "
                f"{list(VALID_TIER)}"
            )
        return declared

    # 2. SpecRule-like: derive from hardness + checker.
    hardness = getattr(rule_or_view, "hardness", None)
    if hardness is not None:
        checker = getattr(rule_or_view, "checker", None)
        return _tier_from_hardness_and_checker(hardness, checker)

    # 3. net-level obligation view: derive from kind.
    kind = getattr(rule_or_view, "kind", None)
    if kind is not None:
        return _DEFAULT_TIER_BY_KIND.get(kind, "CORRECTNESS")

    raise TypeError(
        "tier_for expects a SpecRule (hardness/checker) or an obligation view "
        "(kind); got an object exposing neither"
    )


# --------------------------------------------------------------------------- #
# STEP 2 -- unified, read-only obligation view over BOTH existing chains.      #
# --------------------------------------------------------------------------- #
#
# Clay enumerates obligations through two separate, already-canonical chains:
#
#   * the NET-LEVEL chain -- ``clay.layout.intent_obligations.check_layout_intent``
#     produces ``LayoutObligation`` objects (kind / pair / measured_imbalance /
#     rel_tol / missing_nets). It carries no hardness or checker symbol; its tier
#     derives from ``kind`` via _DEFAULT_TIER_BY_KIND.
#   * the SPEC-PACK chain -- ``clay.layout.spec.packs.apply_packs`` appends
#     provenance-bearing obligation DICTS to ``net.obligations`` (rule / pack /
#     metric / effect / object / hardness / checker / members). Its tier derives
#     from (hardness, checker) via _DEFAULT_TIER / _PHYSICAL_LAW_CHECKERS.
#
# The future resolver (STEP 3) must adjudicate obligations from BOTH chains
# against each other, so it needs to see them in ONE shape. ``unify_obligations``
# builds that single read-only view. It does NOT rebuild geometry, re-run any
# checker, or mutate either input chain -- it only re-projects the two existing
# enumerations into a common, priority-annotated record. This honours the rule
# that every downstream consumer reads the same canonical artifact via no second
# path: the view is a projection, not a parallel obligation model.


# The pack obligation dict's natural "kind" symbol. The spec-pack chain has no
# ``kind`` key (the net-level chain does); its closest equivalent is the physical
# EFFECT the rule targets, falling back through object / metric / rule so a pack
# obligation always has a stable, design-sourced kind symbol.
_PACK_KIND_KEYS = ("effect", "object", "metric", "rule")


@dataclass(frozen=True)
class ObligationView:
    """A read-only, priority-annotated view of one obligation from either chain.

    Fields
    ------
    ob_id : tuple
        Stable identity ``(source, kind, frozenset(members))``. The same
        obligation maps to the same ob_id across calls (no arrival index, no
        object identity), so the resolver can refer to obligations by a value key.
    source : str
        Which chain produced it: ``"net"`` (LayoutObligation) or ``"pack"``
        (apply_packs obligation dict).
    kind : str
        The obligation's kind symbol -- the LayoutObligation ``kind`` for a net
        view, or the pack rule's targeted effect (see _PACK_KIND_KEYS) for a pack
        view.
    members : tuple of str
        The design-sourced member names this obligation relates (the net-level
        ``pair`` or the pack dict ``members``).
    hardness : str or None
        The pack obligation's declared hardness; ``None`` for a net-level
        obligation (which carries no hardness symbol).
    checker : str or None
        The pack obligation's named checker; ``None`` for a net-level obligation.
    priority : Priority or None
        The derived priority annotation (tier via :func:`tier_for`). ``None`` only
        when the obligation's tier is orthogonal (unsupported hardness).
    """

    ob_id: tuple
    source: str
    kind: str
    members: Tuple[str, ...]
    hardness: Optional[str]
    checker: Optional[str]
    priority: Optional[Priority]


def _make_ob_id(source: str, kind: str, members: Iterable[str]) -> tuple:
    """Stable identity for one obligation: (source, kind, frozenset(members))."""
    return (source, kind, frozenset(str(m) for m in members))


def _priority_for(view_like) -> Optional[Priority]:
    """Derive the Priority annotation of a view-like object via ``tier_for``.

    Returns ``None`` when the derived tier is ``None`` (unsupported hardness is
    orthogonal to the lexicographic tier order)."""
    tier = tier_for(view_like)
    if tier is None:
        return None
    return Priority(tier=tier)


class _PackTierShim:
    """Adapt a pack obligation dict to the (hardness, checker) shape tier_for reads.

    ``tier_for`` derives a pack obligation's tier from its hardness + checker (and
    an explicit ``tier`` override when present). The shim exposes exactly those
    attributes off the dict without mutating it."""

    __slots__ = ("hardness", "checker", "tier")

    def __init__(self, ob: dict) -> None:
        self.hardness = ob.get("hardness")
        self.checker = ob.get("checker")
        # apply_packs does not propagate an explicit tier into the dict today, but
        # honour one if a caller threaded it through (override stays auditable).
        self.tier = ob.get("tier")


class _NetTierShim:
    """Adapt a net-level LayoutObligation to the ``kind`` shape tier_for reads."""

    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind


def _pack_kind(ob: dict) -> str:
    """The kind symbol for a pack obligation dict (effect / object / metric / rule)."""
    for key in _PACK_KIND_KEYS:
        val = ob.get(key)
        if val:
            return str(val)
    return "obligation"


def unify_obligations(layout_obligations, pack_obligations) -> List[ObligationView]:
    """Project both obligation chains into one read-only list of ObligationView.

    Parameters
    ----------
    layout_obligations : iterable of LayoutObligation
        The net-level chain output (``check_layout_intent``). Each carries a
        ``kind`` and a ``pair`` (its members); no hardness / checker symbol.
    pack_obligations : iterable of dict
        The spec-pack chain output (``apply_packs`` -> ``net.obligations``). Each
        carries rule / pack / metric / effect / object / hardness / checker /
        members provenance.

    Returns
    -------
    list of ObligationView
        One view per input obligation, in net-then-pack order (deterministic).
        Net views first so the cap/structural net chain stays ahead of the
        spec-pack chain, mirroring the existing enumeration order. ``ob_id`` is
        the stable value key ``(source, kind, frozenset(members))``; ``priority``
        derives from the existing default tables via :func:`tier_for`.

    Read-only: neither input chain is mutated and no geometry / checker is re-run.
    """
    views: List[ObligationView] = []

    # NET-LEVEL chain: LayoutObligation(kind, pair, ...). Members = the pair.
    for ob in layout_obligations or ():
        kind = str(getattr(ob, "kind"))
        members = tuple(str(m) for m in getattr(ob, "pair", ()))
        views.append(
            ObligationView(
                ob_id=_make_ob_id("net", kind, members),
                source="net",
                kind=kind,
                members=members,
                hardness=None,
                checker=None,
                priority=_priority_for(_NetTierShim(kind)),
            )
        )

    # SPEC-PACK chain: provenance-bearing dicts in net.obligations.
    for ob in pack_obligations or ():
        kind = _pack_kind(ob)
        members = tuple(str(m) for m in (ob.get("members") or ()))
        views.append(
            ObligationView(
                ob_id=_make_ob_id("pack", kind, members),
                source="pack",
                kind=kind,
                members=members,
                hardness=ob.get("hardness"),
                checker=ob.get("checker"),
                priority=_priority_for(_PackTierShim(ob)),
            )
        )

    return views


# --------------------------------------------------------------------------- #
# STEP 3 -- resolve_priority: the independent adjudication stage.              #
# --------------------------------------------------------------------------- #
#
# This is the priority axis' adjudication stage. It sits AFTER obligation
# enumeration (the two chains, unified by ``unify_obligations``) and BEFORE the
# signoff verdict aggregation (``clay.signoff_verdict.is_blocking``). It consumes
# the OBLIGATION SET + the CONFLICT PAIRS (which obligations cannot both be
# satisfied; the conflict sources live in the construction plan section 3.3 and
# are an INPUT here, not geometry). It produces, per obligation, an effective
# decision -- "active" or "yielded(by=<winner>,basis=<reason>)" -- plus a list of
# explicit CONFLICT verdicts for the same-tier incomparable conflicts it cannot
# resolve. It rebuilds no geometry and runs no checker: it adjudicates already
# enumerated obligations against each other. This honours the rule that every
# downstream consumer reads the same canonical artifact via no second path.
#
# Adjudication law (construction plan section 1 structures 2/3/4 + section 2.3):
#   1. cross-tier  -> the higher tier (lower VALID_TIER index) wins; the lower
#      yields, basis "cross-tier lexicographic". A LAW-tier obligation is the
#      floor of the order and is NEVER marked yielded (the law-never-yielded
#      invariant locked at STEP 1, enforced here).
#   2. same tier + declared superiority (A.superior_to contains B) -> B yields,
#      basis "declared superiority".
#   3. same tier + no superiority -> specificity tie-break: the narrower scope
#      (fewer members) wins, basis "more-specific scope".
#   4. same tier + no superiority + equal specificity -> INCOMPARABLE: no silent
#      pick. An explicit CONFLICT verdict naming both intents is emitted; neither
#      is yielded. This is no-silent-priority, the dual of no-silent-pass.


# The CONFLICT status the unresolved-conflict verdict carries. STEP 4 adds
# ``VerificationStatus.CONFLICT`` and folds it into the signoff blocking policy;
# until then this resolves to an already-blocking real-defect status so the
# verdict is honestly blocking (no-silent-priority) and so STEP 4 upgrades the
# status in one place without changing this resolver. The blocker_reason carries
# the full conflict semantics either way.
def _conflict_status():
    from clay.core_contract import VerificationStatus

    return getattr(VerificationStatus, "CONFLICT", VerificationStatus.FAIL)


def _tier_rank(view) -> Optional[int]:
    """The lexicographic rank (index) of a view's tier; None if no tier."""
    pri = getattr(view, "priority", None)
    if pri is None or pri.tier is None:
        return None
    return VALID_TIER.index(pri.tier)


def _is_law(view) -> bool:
    pri = getattr(view, "priority", None)
    return pri is not None and pri.tier == "LAW"


def _declares_superiority_over(winner, loser) -> bool:
    """True iff ``winner`` declares superiority over ``loser`` (superior_to edge)."""
    pri = getattr(winner, "priority", None)
    if pri is None:
        return False
    return loser.ob_id in pri.superior_to


def _yielded(winner, basis: str) -> str:
    """The effective-decision string for an obligation that yields to ``winner``."""
    return f"yielded(by={winner.ob_id},basis={basis})"


def _waiver_kind_label(view) -> str:
    """A readable kind label for a waiver reason (hyphenated, tier-annotated).

    e.g. a ring_clearance view at LAW tier reads "ring-clearance(LAW)"; a density
    view at QUALITY tier reads "density". Underscores become hyphens; a LAW winner
    is annotated "(LAW)" so the audit trail names why it was unwaivable."""
    kind = str(getattr(view, "kind", "obligation")).replace("_", "-")
    pri = getattr(view, "priority", None)
    if pri is not None and pri.tier == "LAW":
        return f"{kind}(LAW)"
    return kind


def _make_waiver(loser, winner, basis: str):
    """Build the auditable Waiver record for ``loser`` yielding to ``winner``."""
    reason = (
        f"{_waiver_kind_label(loser)} yielded to {_waiver_kind_label(winner)}, "
        f"basis={basis}"
    )
    return Waiver(
        loser=loser.ob_id, winner=winner.ob_id, basis=basis, reason=reason
    )


@dataclass(frozen=True)
class Waiver:
    """An auditable record that one obligation yielded to another (external_drc model).

    A waiver is the registered, signoff-visible trail of a RESOLVED priority
    conflict: the loser was not silently dropped, it yielded to a named winner on
    a named basis. The downstream signoff aggregation lowers the yielded
    obligation's effective ``required`` (STEP 4) and surfaces this record so the
    let-off is auditable, never silent. A LAW-tier obligation can never be the
    ``loser`` of a waiver (law-never-yielded / LAW is not waivable).

    Fields
    ------
    loser : tuple
        The ob_id of the obligation that yielded.
    winner : tuple
        The ob_id of the obligation it yielded to.
    basis : str
        Why it yielded ("cross-tier lexicographic" / "declared superiority" /
        "more-specific scope").
    reason : str
        A human-readable, signoff-visible sentence (e.g. "density yielded to
        ring-clearance(LAW), basis=cross-tier lexicographic").
    """

    loser: tuple
    winner: tuple
    basis: str
    reason: str


@dataclass(frozen=True)
class PriorityResolution:
    """The result of one ``resolve_priority`` pass.

    Fields
    ------
    effective : dict
        ``ob_id -> "active" | "yielded(by=<winner ob_id>,basis=<reason>)"``. The
        per-obligation effective decision the downstream signoff aggregation
        consumes (a yielded obligation's effective ``required`` is later lowered;
        STEP 4). A LAW-tier obligation is never "yielded".
    conflict_verdicts : list of CheckerVerdict
        One verdict per UNRESOLVED same-tier incomparable conflict. Each is an
        explicit, blocking verdict naming both intents -- the no-silent-priority
        record. Empty when every conflict was adjudicated.
    waivers : list of Waiver
        One auditable :class:`Waiver` per RESOLVED conflict where an obligation
        yielded. The signoff-visible let-off trail (no silent drop). Empty when
        no obligation yielded.
    """

    effective: dict
    conflict_verdicts: list
    waivers: list = field(default_factory=list)


def resolve_priority(obligations, *, conflicts) -> "PriorityResolution":
    """Adjudicate conflicting obligations by tier / superiority / specificity.

    Parameters
    ----------
    obligations : iterable of ObligationView
        The unified obligation set (see :func:`unify_obligations`). Every
        obligation starts ``"active"``.
    conflicts : iterable of (ObligationView, ObligationView)
        Pairs of obligations that cannot both be satisfied. The conflict SOURCE
        (declaration-level static rules / implementation-level fault_intent) is
        upstream of this resolver; here a conflict pair is an input.

    Returns
    -------
    PriorityResolution
        ``effective`` (per-obligation decision) + ``conflict_verdicts`` (the
        unresolved same-tier incomparable conflicts as explicit blocking
        verdicts). No geometry is rebuilt; no checker is re-run.
    """
    from clay.signoff_verdict import CheckerVerdict

    effective = {}
    for ob in obligations or ():
        effective[ob.ob_id] = "active"

    conflict_verdicts = []
    waivers = []

    for a, b in conflicts or ():
        # ensure both endpoints have an effective entry even if not listed in
        # ``obligations`` (defensive; keeps the result self-describing).
        effective.setdefault(a.ob_id, "active")
        effective.setdefault(b.ob_id, "active")

        rank_a, rank_b = _tier_rank(a), _tier_rank(b)

        # 1. cross-tier: the higher tier (lower rank index) wins. A LAW side is
        #    never the loser (law-never-yielded).
        if rank_a is not None and rank_b is not None and rank_a != rank_b:
            if rank_a < rank_b:
                winner, loser = a, b
            else:
                winner, loser = b, a
            # law-never-yielded guard: a LAW obligation can never be the loser.
            # By construction LAW is index 0 (the floor), so it is always the
            # winner of any cross-tier pair; this assertion documents and locks
            # the invariant rather than silently trusting it.
            if _is_law(loser):
                raise AssertionError(
                    "law-never-yielded violated: a LAW-tier obligation "
                    f"{loser.ob_id} was selected to yield to {winner.ob_id}"
                )
            effective[loser.ob_id] = _yielded(
                winner, "cross-tier lexicographic"
            )
            waivers.append(_make_waiver(loser, winner, "cross-tier lexicographic"))
            continue

        # below here: same tier (or one/both untiered). An untiered obligation
        # (unsupported hardness) is orthogonal to the order: it cannot win or
        # lose a priority contest, so such a pair falls through to CONFLICT only
        # when truly incomparable. For the same-tier path we need both ranks.
        if rank_a is None or rank_b is None or rank_a != rank_b:
            # one side is untiered -> not comparable by tier and no within-tier
            # relation applies; report it as an explicit conflict (no silent pick).
            conflict_verdicts.append(_make_conflict_verdict(a, b, CheckerVerdict))
            continue

        # 2. same tier + declared superiority.
        if _declares_superiority_over(a, b):
            effective[b.ob_id] = _yielded(a, "declared superiority")
            waivers.append(_make_waiver(b, a, "declared superiority"))
            continue
        if _declares_superiority_over(b, a):
            effective[a.ob_id] = _yielded(b, "declared superiority")
            waivers.append(_make_waiver(a, b, "declared superiority"))
            continue

        # 3. same tier + no superiority -> specificity (fewer members = narrower).
        len_a, len_b = len(a.members), len(b.members)
        if len_a != len_b:
            if len_a < len_b:
                winner, loser = a, b
            else:
                winner, loser = b, a
            effective[loser.ob_id] = _yielded(winner, "more-specific scope")
            waivers.append(_make_waiver(loser, winner, "more-specific scope"))
            continue

        # 4. same tier + no superiority + equal specificity -> incomparable.
        #    No silent pick: emit an explicit CONFLICT verdict naming both.
        conflict_verdicts.append(_make_conflict_verdict(a, b, CheckerVerdict))

    return PriorityResolution(
        effective=effective,
        conflict_verdicts=conflict_verdicts,
        waivers=waivers,
    )


# --------------------------------------------------------------------------- #
# STEP 7 -- declaration-level conflict source (static mutual-exclusion rules).  #
# --------------------------------------------------------------------------- #
#
# The DECLARATION-LEVEL conflict source (construction plan section 3.3) -- the
# dual of the implementation-level ``impl_conflicts_from_geometry``. It catches a
# KNOWN-SHAPE conflict BEFORE any geometry runs, purely from the obligation set:
# two obligations whose kinds are mutually exclusive over a SHARED resource
# (members) cannot both be satisfied, so the resolver should adjudicate them at
# adjudication time rather than letting the contradiction explode at the oracle.
#
# The motivating retrofit is dense x ring (plan section 4): a density-fold packing
# obligation (QUALITY) and a ring-clearance obligation (LAW-derived) over the same
# guard-ring nets are mutually exclusive -- packing the rings closer to raise
# density is exactly what violates ring clearance and merges the rails at LVS. The
# static rule flags that pair at the declaration stage; the SAME resolve_priority
# then adjudicates it cross-tier (density yields to ring-clearance(LAW)) and emits
# an auditable waiver, instead of a late LVS rail-short failure.
#
# Honest incompleteness (plan section 6): a static rule can only enumerate
# KNOWN-SHAPE conflicts; conflicts it cannot name statically are left to the
# implementation-level diagnose + the oracle. It does NOT pretend the declaration
# stage enumerates every conflict (that would be its own silent failure). Pure
# symbolic, zero coordinates: it reads obligation kinds + members, never geometry.


# Declarative mutual-exclusion rules: unordered pairs of obligation kinds that
# cannot both be satisfied when they relate a SHARED resource (overlapping
# members). A frozenset entry {K1, K2} means "an obligation of kind K1 and one of
# kind K2 over shared members are mutually exclusive". This is a TABLE, auditable
# and extendable; it is NOT a per-circuit special-case (it keys on kinds, not on a
# named circuit or spec). The resolver -- not this table -- decides who yields.
_MUTUAL_EXCLUSION_RULES = (
    # dense x ring: density-fold packing vs ring-clearance over the same guard
    # rings. Packing rings closer to raise density is exactly what violates ring
    # clearance and merges the rails at LVS (the P4 fault).
    frozenset({"density", "ring_clearance"}),
)


def _members_overlap(a, b) -> bool:
    """True iff two obligation views relate at least one shared member (resource).

    A declaration-level mutual exclusion only fires when the two obligations
    actually contend for the SAME resource; disjoint members are not in conflict.
    """
    return bool(set(a.members) & set(b.members))


def declaration_conflicts(obligations):
    """Enumerate declaration-level, known-shape mutual-exclusion conflict pairs.

    The declaration-level conflict source feeding :func:`resolve_priority`. It runs
    on the unified obligation set (see :func:`unify_obligations`) BEFORE any
    geometry, and returns the conflict pairs the static mutual-exclusion table
    (:data:`_MUTUAL_EXCLUSION_RULES`) identifies between obligations that contend
    for a shared resource.

    Parameters
    ----------
    obligations : iterable of ObligationView
        The unified obligation set.

    Returns
    -------
    list of (ObligationView, ObligationView)
        One pair per detected mutual-exclusion conflict (kinds in a rule, members
        overlapping), ready to hand to ``resolve_priority`` as its ``conflicts``
        input. Honest emptiness: a pair whose kinds are not in any rule, or whose
        members are disjoint, is not invented as a conflict (no silent pick). Only
        KNOWN-SHAPE conflicts are returned; the rest are left to the
        implementation-level source + the oracle (plan section 6).

    Reads obligation kinds + members only; no geometry, no checker re-run.
    """
    views = list(obligations or ())
    pairs = []
    seen = set()
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            a, b = views[i], views[j]
            kinds = frozenset({a.kind, b.kind})
            if kinds not in _MUTUAL_EXCLUSION_RULES:
                continue
            if not _members_overlap(a, b):
                continue
            key = tuple(sorted((str(a.ob_id), str(b.ob_id))))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b))
    return pairs


# --------------------------------------------------------------------------- #
# STEP 5 -- impl-conflict-named: implementation-level conflict via fault_intent #
# --------------------------------------------------------------------------- #
#
# The IMPLEMENTATION-LEVEL conflict source -- the dual of the declaration-level
# source (construction plan section 3.3). Declaration-level catches a known-shape
# conflict BEFORE geometry runs; this catches conflicts AFTER geometry by grouping
# already-occurred FAIL obligations that share one root decision.
#
# Mechanism (plan 3.3 + 5.5): when geometry has run and MULTIPLE FAIL obligations
# map to the same ``fault_intent.RootCause.decision`` (the same column/resource the
# router co-located their nets into), they are ONE implementation-level conflict
# cluster. We REUSE ``fault_intent.diagnose`` (its column set-cover IS the MCS /
# minimal conflict set) to find the root decisions, and ``validate_counterfactuals``
# (its self-verification) to prove the cluster's single-step fix is geometrically
# sound. The cluster then feeds ``resolve_priority``'s ``conflicts`` as ONE
# conflict pair -- the SAME resolver, no second adjudication path. It rebuilds no
# geometry: it reads the already-run routed cell + the already-produced obligations.


@dataclass(frozen=True)
class ImplConflictCluster:
    """One implementation-level conflict cluster (plan section 3.3 / 5.5).

    A set of FAIL obligations the router co-located into ONE root decision (the
    same fault_intent column). The cluster is NAMED by the RootCause it violates.

    Fields
    ------
    decision : str
        The ``RootCause.decision`` that groups the members (the shared column /
        resource the conflict roots in).
    intent_violated : str
        The named intent the RootCause violates (e.g. "rail isolation ...",
        "@route_policy(tracks=net_disjoint) ..."). This is what makes the cluster
        a NAMED conflict rather than an anonymous geometry failure.
    obligations : tuple of ObligationView
        The FAIL obligations that share this root decision (>= 2; a single
        obligation is not a between-obligation conflict).
    counterfactual_net : str
        The net the RootCause's single-step fix relocates.
    counterfactual_sound : bool
        Whether ``validate_counterfactuals`` confirmed that relocation actually
        clears the named net's symptoms in the real geometry (self-verified, not
        asserted).
    """

    decision: str
    intent_violated: str
    obligations: tuple
    counterfactual_net: str
    counterfactual_sound: bool


def _obligation_view_from_layout_obligation(ob) -> "ObligationView":
    """Project a net-level FAIL LayoutObligation into the canonical ObligationView.

    Reuses the same net-chain projection as :func:`unify_obligations` (source
    "net", kind from ``.kind``, members from ``.pair``, tier via the kind default
    table) so the resolver sees impl-level conflict obligations in exactly the same
    canonical shape as every other obligation -- no second obligation model."""
    kind = str(getattr(ob, "kind"))
    members = tuple(str(m) for m in getattr(ob, "pair", ()))
    return ObligationView(
        ob_id=_make_ob_id("net", kind, members),
        source="net",
        kind=kind,
        members=members,
        hardness=None,
        checker=None,
        priority=_priority_for(_NetTierShim(kind)),
    )


def _obligation_nets(ob) -> frozenset:
    """The non-empty net set a FAIL obligation relates (its ``pair``)."""
    return frozenset(str(m) for m in getattr(ob, "pair", ()) if str(m))


def _root_for_obligation(ob, root_causes):
    """The RootCause whose co-located nets cover this obligation's nets, or None.

    An obligation maps to a root decision when EVERY net it relates was co-located
    into that root's column (the router's shared decision is what broke it). Reading
    provenance off the diagnosis -- never a counterfactual replay (fault_intent's
    first trick)."""
    nets = _obligation_nets(ob)
    if not nets:
        return None
    for rc in root_causes:
        rc_nets = frozenset(rc.nets_involved)
        if nets <= rc_nets:
            return rc
    return None


def impl_conflicts_from_geometry(routed, obligations, *, criteria=None):
    """Group already-occurred FAIL obligations that share one fault_intent root.

    The implementation-level conflict source feeding :func:`resolve_priority`.

    Parameters
    ----------
    routed : routed-cell-like
        An ALREADY-RUN routed cell carrying ``tagged`` geometry (and union-find
        ``shorts``). Read-only; nothing is re-routed or re-extracted.
    obligations : iterable of LayoutObligation
        The already-produced net-level obligations. Only those with
        ``status == "fail"`` participate (a pass / unmeasurable obligation is not
        an implementation-level conflict).
    criteria : optional
        Passed through to ``diagnose`` for diff-pair intent attribution.

    Returns
    -------
    (conflict_pairs, clusters)
        ``conflict_pairs`` -- a list of ``(ObligationView, ObligationView)`` pairs,
        EXACTLY ONE pair per cluster, ready to hand to ``resolve_priority`` as its
        ``conflicts`` input (so N shared-root FAIL obligations produce ONE conflict
        verdict, not N). ``clusters`` -- the named :class:`ImplConflictCluster`
        list (one per shared root decision), each self-validated.

    A cluster forms only when >= 2 FAIL obligations share ONE
    ``RootCause.decision``. Honest emptiness: a FAIL obligation whose nets were not
    co-located maps to no root and invents no conflict (no silent pick).
    """
    from clay.layout.fault_intent import diagnose, validate_counterfactuals

    fails = [ob for ob in (obligations or ()) if getattr(ob, "status", None) == "fail"]
    if not fails:
        return [], []

    diagnosis = diagnose(routed, criteria=criteria)
    if not diagnosis.root_causes:
        return [], []

    # self-verify the counterfactuals ONCE over the diagnosis; index by decision.
    _soundness, per_root = validate_counterfactuals(routed, diagnosis)
    sound_by_decision = {}
    for decision, info in per_root:
        sound_by_decision[decision] = bool(info and info.get("exact"))

    # group FAIL obligations by the shared RootCause.decision they map onto.
    rc_by_decision = {rc.decision: rc for rc in diagnosis.root_causes}
    grouped: dict = {}
    for ob in fails:
        rc = _root_for_obligation(ob, diagnosis.root_causes)
        if rc is None:
            continue  # not co-located into any root -> no impl-level conflict.
        grouped.setdefault(rc.decision, []).append(ob)

    clusters: List[ImplConflictCluster] = []
    conflict_pairs: list = []
    for decision in sorted(grouped):
        members = grouped[decision]
        if len(members) < 2:
            continue  # a single obligation is not a between-obligation conflict.
        rc = rc_by_decision[decision]
        views = tuple(_obligation_view_from_layout_obligation(ob) for ob in members)
        clusters.append(
            ImplConflictCluster(
                decision=decision,
                intent_violated=rc.intent_violated,
                obligations=views,
                counterfactual_net=rc.counterfactual_net,
                counterfactual_sound=sound_by_decision.get(decision, False),
            )
        )
        # ONE conflict pair per cluster (representative endpoints) so the cluster
        # feeds the resolver as a SINGLE conflict -- N shared-root FAILs -> one
        # CONFLICT verdict, not N. The cluster object retains all members for audit.
        conflict_pairs.append((views[0], views[1]))

    return conflict_pairs, clusters


def _conflict_obligation_label(view) -> str:
    """A human-readable label for an obligation in a conflict reason."""
    members = ",".join(view.members) if view.members else "-"
    return f"{view.kind}({members})"


def _make_conflict_verdict(a, b, checker_verdict_cls):
    """Build the explicit, blocking CONFLICT verdict for an incomparable pair.

    The verdict names BOTH intents, their tier, and the incomparable reason --
    the no-silent-priority record. A stable obligation_id binds it to the
    unordered pair of ob_ids."""
    pri_a = getattr(a, "priority", None)
    tier = pri_a.tier if pri_a is not None and pri_a.tier is not None else "?"
    label_a = _conflict_obligation_label(a)
    label_b = _conflict_obligation_label(b)
    reason = (
        f"priority CONFLICT: {label_a} and {label_b} are both at tier {tier} "
        "with no declared superiority and equal specificity -- incomparable; "
        "declare a superiority or change an intent (no arrival-order tie-break)."
    )
    # stable, order-independent id for the conflicting pair.
    pair_key = tuple(sorted((str(a.ob_id), str(b.ob_id))))
    return checker_verdict_cls(
        checker="resolve_priority",
        obligation_id=f"conflict::{pair_key[0]}::{pair_key[1]}",
        status=_conflict_status(),
        evidence={
            "intent_a": str(a.ob_id),
            "intent_b": str(b.ob_id),
            "tier": tier,
            "reason": "no declared superiority, equal specificity",
        },
        blocker_reason=reason,
        required=True,
    )
