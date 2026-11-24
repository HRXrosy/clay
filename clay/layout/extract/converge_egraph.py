"""Phase 3 of the INTENT-EXTRACTION CONVERGENCE METHOD: the GLOBAL, ORDER-
INDEPENDENT minimal-intent search (the e-graph / two-part-MDL extraction).

WHERE THIS FITS (vs. relax.converge)
====================================
``clay.layout.extract.relax.converge`` is GREEDY delta-debugging (Zeller-
Hildebrandt): it drops one ``@fix`` pin at a time and keeps the drop iff the
equivalence oracle still holds, reaching a 1-MINIMAL fixpoint. That fixpoint is
ORDER-DEPENDENT (which 1-minimal you land on depends on the drop order) and only
LOCALLY irreducible -- "no single remaining pin can be dropped", which is weaker
than "the globally shortest intent".

This module builds the GLOBAL search the survey (docs/superpowers/specs/
2026-06-06-abstraction-mdl-convergence-survey.html) recommends:

    lattice of pinned-subsets (frame, Galois backbone, survey 1)
  + oracle-guarded equivalence-preserving rewrites populating the lattice
        (search, survey 2 + 4 -- the inverse of CEGAR / delta-debugging, but
         NON-DESTRUCTIVE: every admissible subset coexists, no drop order is
         committed)
  + two-part-MDL extraction (stopping / minimality, survey 3)

The order-independence payoff: because we explore the subset LATTICE rather than
commit to a greedy chain, the returned minimum is the same regardless of which
relaxations we would have tried first. We then COMPARE against
``relax.converge``'s greedy result on the SAME input and report whether the
global search beats greedy (fewer pins / lower MDL).

THE "E-GRAPH" AT THIS SCALE
===========================
The survey's headline structure is an e-graph populated by equality saturation.
Here the only relaxation is "drop the @fix pin of device d" (every device has the
same single rewrite), so the set of admissible intent encodings is exactly the
set of admissible PINNED SUBSETS S of devices -- a subset LATTICE. A full
egg-style e-graph (union-find + congruence closure over a rich rewrite algebra)
is overkill at this scale: there is no shared-subexpression structure to
deduplicate, the congruence is trivial, and the "extraction" is just "pick the
min-MDL admissible subset". So we realize the e-graph as the SUBSET LATTICE +
MEMOIZED admissibility. We state this honestly rather than build machinery the
problem does not need. (If the rewrite vocabulary grows -- multiple competing
abstract relations per pin, shared relation primitives -- a real e-graph would
start to earn its keep; see the roadmap caveat at the bottom.)

ADMISSIBILITY (downward-monotone-ish)
=====================================
A pinned subset S is ADMISSIBLE iff ``equivalence_to_anchor(place_relaxed(intent,
S), intent, care_mask)`` holds: with only S pinned, the abstract relations re-
derive every dropped device within the care-mask. Dropping MORE pins is
(generally) HARDER -- removing a pin can only remove information the placer had --
so admissibility is approximately DOWNWARD-CLOSED in the "more pins" direction
(equivalently, upward-closed toward the full anchor, which is always admissible).
We exploit this for branch-and-bound pruning, but we do NOT ASSUME it as a hard
invariant: the oracle is a semantic placer call and a relation could in principle
re-derive a device better with fewer pins, so every candidate is still tested.
The monotonicity is a HEURISTIC for ordering/pruning, not a soundness crutch.

ORACLE COST
===========
Each admissibility test is one ``place_relaxed`` call = one deterministic placer
run. This DOMINATES cost (the survey's main open feasibility risk, 6.2). We
MEMOIZE by ``frozenset(subset)`` so each distinct subset is placed at most once.
The exact search is EXPONENTIAL in the number of relaxable devices (optimal
DAG/e-graph extraction is NP-complete, survey 4.4-ii) and is therefore GATED to
small N; beyond that we fall back to a beam / local search that keeps the
order-independence of the SEARCH SPACE but makes the final pick approximate.

This module does NOT edit relax.py / the placers / library_device / grammar /
design.clay. It IMPORTS relax.py's oracle (``equivalence_to_anchor``), placer
(``place_relaxed``), anchor reader, relation attribution, and the greedy
``converge`` (for the global-vs-greedy comparison).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from clay.layout.extract.relax import (
    _anchor_centroids,
    _relations_governing,
    converge as _greedy_converge,
    equivalence_to_anchor,
    place_relaxed,
)
from clay.layout.intent_basis_search import BasisEvaluation, search_minimal_basis

# --------------------------------------------------------------------------- #
# Default MDL code weights (TUNABLE -- see ``MdlCostModel``).
# --------------------------------------------------------------------------- #
# A pinned @fix(x, y) coordinate costs many bits (two explicit reals); an active
# abstract relation costs a few bits (a primitive name + its operands). The
# residual term encodes the leftover positional slack the placer did not nail
# under the care-mask. These are the survey's two-part code L(I) + L(R|I,P).
_DEFAULT_PIN_COST = 16.0       # bits to encode one explicit @fix coordinate pair
_DEFAULT_RELATION_COST = 4.0   # bits to encode one active abstract relation
_DEFAULT_RESIDUAL_SCALE = 1.0  # bits per um of encoded residual deviation


# --------------------------------------------------------------------------- #
# 1. The relaxation lattice: candidate relaxations + what covers each drop.
# --------------------------------------------------------------------------- #
@dataclass
class RelaxationLattice:
    """The search space = choosing a PINNED SUBSET ``S`` of devices.

    * ``all_devices``    -- every device with a reference bbox (the lattice carrier).
    * ``relaxable``      -- devices whose @fix COULD be dropped because some abstract
      relation plausibly covers them (``relations_for`` non-empty). Dropping a pin
      with no covering relation can only ever break equivalence, so such devices are
      forced-pinned and excluded from the search dimension (a free, sound prune).
    * ``forced_pinned``  -- ``all_devices - relaxable``: pins no relation covers.
    * ``relations_for``  -- per device, the abstract relation(s) (matching / equal_
      size / band) that would re-derive its position if its pin were dropped. This is
      the "which abstract relation covers each candidate" annotation the task asks
      for; it is a pure read of the intent (no placement).

    The intent for a pinned subset ``S`` is "abstract relations for ALL devices +
    @fix only on ``S``" (exactly ``relax.emit_relaxed(intent, S)``). The lattice top
    is ``S = all_devices`` (the anchor, always admissible); we search downward.
    """

    all_devices: Tuple[str, ...]
    relaxable: Tuple[str, ...]
    forced_pinned: Tuple[str, ...]
    relations_for: Dict[str, List[str]]

    @property
    def n_relaxable(self) -> int:
        return len(self.relaxable)


def relaxation_lattice(intent: Any) -> RelaxationLattice:
    """Enumerate the candidate relaxations = the relaxable-device search dimension.

    A device is a relaxation CANDIDATE iff (a) it has a reference bbox and (b) at
    least one abstract relation could cover it (``_relations_governing`` non-empty).
    Devices with no covering relation are FORCED-PINNED: dropping their pin can only
    break equivalence, so excluding them from the search dimension is a sound,
    free prune (it never removes an admissible subset, because any subset that drops
    a forced-pinned device is inadmissible by construction -- nothing re-derives it).

    Returns the :class:`RelaxationLattice` carrying the carrier set, the relaxable
    dimension, the forced-pinned residual floor, and the per-device covering
    relations.
    """
    all_devices = tuple(sorted(_anchor_centroids(intent)))
    relations_for: Dict[str, List[str]] = {
        d: list(_relations_governing(intent, d)) for d in all_devices
    }
    relaxable = tuple(d for d in all_devices if relations_for[d])
    forced = tuple(d for d in all_devices if not relations_for[d])
    return RelaxationLattice(
        all_devices=all_devices,
        relaxable=relaxable,
        forced_pinned=forced,
        relations_for=relations_for,
    )


# --------------------------------------------------------------------------- #
# 2. The two-part MDL cost model (TUNABLE).
# --------------------------------------------------------------------------- #
@dataclass
class MdlCostModel:
    """Two-part MDL code weights L(I) + L(R | I, P) (survey 3). All TUNABLE.

    * ``pin_cost``       -- bits per explicit @fix coordinate pair in ``S``.
    * ``relation_cost``  -- bits per ACTIVE abstract relation (one whose membership
      the intent declares and that governs at least one device).
    * ``residual_scale`` -- bits per micron of encoded residual deviation
      (the don't-care slack the placer did not nail exactly under the care-mask).

    Rationale (survey 3.3): an explicit pinned coordinate is expensive (two reals);
    an abstract relation is cheap and general (a primitive name + operands), so it
    SHOULD cost far less than the pins it replaces -- otherwise "minimal" is
    meaningless (survey 6.2: "MDL code is only as good as the prior"). The defaults
    encode pin >> relation; tune to your vocabulary's real marginal description
    length.
    """

    pin_cost: float = _DEFAULT_PIN_COST
    relation_cost: float = _DEFAULT_RELATION_COST
    residual_scale: float = _DEFAULT_RESIDUAL_SCALE


def _active_relation_count(intent: Any) -> int:
    """Number of ACTIVE abstract relations the intent declares.

    Counts each declared matched/diff/mirror/interdigitated pair, each equal_size
    group, and the band partition (one relation per distinct band that has >= 2
    members). This is L(I)'s relation term -- it is INDEPENDENT of which subset is
    pinned (the relations are emitted for ALL devices in the relaxed form), so it is
    a fixed offset across subsets; we keep it explicit so the cost model is honest
    about the full L(I), not just the pin term.
    """
    n = 0
    for attr in ("matched_pairs", "diff_pairs", "mirror_pairs", "interdigitated_pairs"):
        for ab in getattr(intent, attr, ()) or ():
            if len(ab) == 2:
                n += 1
    for g in getattr(intent, "equal_size_groups", ()) or ():
        if len({str(m) for m in g}) >= 2:
            n += 1
    bands = getattr(intent, "bands", {}) or {}
    band_members: Dict[Any, int] = {}
    for _, b in bands.items():
        band_members[b] = band_members.get(b, 0) + 1
    n += sum(1 for b, c in band_members.items() if c >= 2)
    return n


def mdl_cost(
    intent: Any,
    pinned_subset: Set[str],
    *,
    care_mask_um: float,
    model: Optional[MdlCostModel] = None,
    oracle: Optional[Callable[[Set[str]], Tuple[bool, float]]] = None,
) -> float:
    """Two-part MDL objective for the intent that pins ``pinned_subset`` (survey 3).

    ::

        mdl = L(I)                              + L(R | I, P)
            = pin_cost * |S|                       residual_scale * residual_um
            + relation_cost * (#active relations)

    * ``L(I)`` -- the intent description length: ``pin_cost`` per @fix pin actually
      kept (``|pinned_subset ∩ all_devices|``) PLUS ``relation_cost`` per active
      abstract relation (a fixed offset across subsets; included for an honest full
      L(I)). Fewer pins -> shorter intent -> lower cost: the pin term DECREASES
      monotonically as pins drop, which is the pressure toward abstraction.
    * ``L(R | I, P)`` -- the residual: ``residual_scale`` times the placer's max
      per-device centroid deviation from the anchor under the relaxed intent (the
      don't-care slack still needed to recover R exactly). Obtained from the oracle's
      ``max_dev``; for an INADMISSIBLE subset the residual is the deviation that
      exceeds the mask. If no ``oracle`` is supplied the residual term is taken as 0
      (pure L(I) mode -- useful for ranking admissible subsets, which all have
      residual <= care_mask by definition).

    ``oracle`` (when given) maps a pinned subset to ``(is_equivalent, max_dev_um)``;
    inject it to avoid a real placer run in fast tests, or let ``converge_global``
    pass its memoized oracle. The cost model is fully TUNABLE via ``model``.
    """
    m = model or MdlCostModel()
    anchor = set(_anchor_centroids(intent))
    n_pins = len(set(pinned_subset) & anchor)
    n_relations = _active_relation_count(intent)

    l_intent = m.pin_cost * n_pins + m.relation_cost * n_relations

    residual_um = 0.0
    if oracle is not None:
        _, max_dev = oracle(set(pinned_subset))
        if math.isfinite(max_dev):
            residual_um = max_dev
    l_residual = m.residual_scale * residual_um

    return l_intent + l_residual


# --------------------------------------------------------------------------- #
# 3. The global search: exact branch-and-bound (small N) + beam fallback.
# --------------------------------------------------------------------------- #
@dataclass
class GlobalResult:
    """The outcome of the global, order-independent search.

    * ``pinned_subset``  -- the MIN-MDL admissible pinned subset (the global minimal
      intent's pins).
    * ``dropped``        -- ``all_devices - pinned_subset``: the abstracted devices.
    * ``mdl``            -- the two-part MDL cost of ``pinned_subset``.
    * ``total``          -- |all devices with a reference bbox|.
    * ``oracle_calls``   -- number of DISTINCT subsets placed (memoized admissibility
      tests); the cost the survey says dominates.
    * ``method``         -- ``'exact-bnb'`` or ``'beam'`` (which engine produced it).
    * ``care_mask_um``   -- the don't-care radius used.
    * ``relations_for_dropped`` -- per dropped device, the abstract relation(s) that
      cover it (the attribution).
    * ``greedy`` / ``greedy_mdl`` -- the greedy ``relax.converge`` comparison.
    * ``beats_greedy``   -- did global find FEWER pins or LOWER MDL than greedy?
    """

    pinned_subset: Tuple[str, ...]
    dropped: Tuple[str, ...]
    mdl: float
    total: int
    oracle_calls: int
    method: str
    care_mask_um: float
    relations_for_dropped: Dict[str, List[str]] = field(default_factory=dict)
    greedy: Optional[Tuple[str, ...]] = None
    greedy_mdl: Optional[float] = None
    beats_greedy: Optional[bool] = None


# How many relaxable devices the exact branch-and-bound will handle before the
# beam fallback kicks in (2**12 = 4096 subsets is the worst-case lattice size).
_EXACT_MAX_RELAXABLE = 12


class _MemoOracle:
    """Memoized admissibility oracle: each distinct pinned subset placed at most once.

    Wraps the (placer + equivalence) pipeline keyed by ``frozenset(subset)`` so the
    EXPONENTIAL search never re-places a subset. ``calls`` counts the distinct
    placer runs actually performed (cache misses) -- the true oracle cost.

    ``equiv_fn`` defaults to the real ``relax`` pipeline (place then compare); inject
    a stub ``equiv_fn(subset) -> (ok, max_dev)`` for fast/deterministic tests.
    """

    def __init__(
        self,
        intent: Any,
        care_mask_um: float,
        *,
        tech: str = "gf180mcu",
        equiv_fn: Optional[Callable[[FrozenSet[str]], Tuple[bool, float]]] = None,
    ) -> None:
        self._intent = intent
        self._mask = care_mask_um
        self._tech = tech
        self._equiv_fn = equiv_fn
        self._cache: Dict[FrozenSet[str], Tuple[bool, float]] = {}
        self.calls = 0

    def _place_and_compare(self, subset: FrozenSet[str]) -> Tuple[bool, float]:
        placed = place_relaxed(self._intent, set(subset), tech=self._tech)
        return equivalence_to_anchor(placed, self._intent, self._mask)

    def __call__(self, subset: Set[str]) -> Tuple[bool, float]:
        key = frozenset(subset)
        if key in self._cache:
            return self._cache[key]
        if self._equiv_fn is not None:
            res = self._equiv_fn(key)
        else:
            res = self._place_and_compare(key)
        self._cache[key] = res
        self.calls += 1
        return res


def converge_global(
    intent: Any,
    care_mask_um: float,
    *,
    tech: str = "gf180mcu",
    model: Optional[MdlCostModel] = None,
    budget: int = _EXACT_MAX_RELAXABLE,
    beam_width: int = 8,
    equiv_fn: Optional[Callable[[FrozenSet[str]], Tuple[bool, float]]] = None,
    compare_greedy: bool = True,
) -> GlobalResult:
    """Find the GLOBAL min-MDL admissible pinned subset, ORDER-INDEPENDENTLY.

    Builds the :func:`relaxation_lattice`, then searches the pinned-subset lattice
    for the admissible subset minimizing :func:`mdl_cost`:

      * EXACT branch-and-bound (:func:`_exact_bnb`) when the number of RELAXABLE
        devices is ``<= budget`` (default 12 -> <= 4096 subsets). Admissibility is
        downward-monotone-ish, so the anchor is the always-admissible top and we
        prune by the L(I) lower bound. This is correct-but-exponential (optimal
        e-graph extraction is NP-complete, survey 4.4-ii) hence the gate.
      * BEAM / local search (:func:`_beam_search`) fallback when relaxable > budget:
        polynomial oracle calls, order-independent search space, approximate pick.

    Every distinct subset is placed AT MOST ONCE (``frozenset`` memoization), so
    ``oracle_calls`` counts true placer runs. Inject ``equiv_fn(frozenset) ->
    (ok, max_dev)`` to stub the placer for fast/deterministic tests.

    When ``compare_greedy`` (default), also runs the GREEDY ``relax.converge`` on the
    SAME input and reports ``beats_greedy`` (global found fewer pins OR lower MDL) --
    the order-independence payoff. The greedy comparison uses a real placer run
    UNLESS ``equiv_fn`` is injected, in which case it is skipped (the stub does not
    drive the greedy loop) and ``greedy``/``beats_greedy`` are left ``None`` with the
    caller expected to compare against an independently-computed greedy subset.

    Returns a :class:`GlobalResult`.

    HONEST CAVEATS (the survey's, restated):
      * Optimal DAG/e-graph extraction is NP-COMPLETE; the exact search is
        EXPONENTIAL and gated to small N (``budget``).
      * The ORACLE CALL DOMINATES cost -- each candidate is a placer run; we memoize
        by ``frozenset(subset)`` but cannot avoid testing distinct admissible subsets.
      * The "e-graph" here is the subset lattice + memoized admissibility; a full
        egg-style e-graph is overkill at this single-rewrite scale (stated up top).
    """
    m = model or MdlCostModel()
    lattice = relaxation_lattice(intent)
    oracle = _MemoOracle(intent, care_mask_um, tech=tech, equiv_fn=equiv_fn)

    # The generic basis search uses integral description lengths.  Preserve the
    # public floating-point MDL model by using micro-bit integers for extraction,
    # then recompute the published result with ``mdl_cost`` below.  The active-
    # relation and forced-pin terms are constant across this search dimension and
    # therefore do not affect which subset is minimal.
    mdl_scale = 1_000_000
    pin_cost_units = int(round(m.pin_cost * mdl_scale))
    if pin_cost_units < 0:
        raise ValueError("pin_cost must be non-negative")
    atom_costs = tuple((device, pin_cost_units) for device in lattice.relaxable)
    forced_pinned = frozenset(lattice.forced_pinned)

    def evaluate_basis(selected: FrozenSet[str]) -> BasisEvaluation:
        pinned = frozenset(selected | forced_pinned)
        ok, max_dev = oracle(set(pinned))
        residual_units = (
            max(0, int(round(m.residual_scale * max_dev * mdl_scale)))
            if math.isfinite(max_dev)
            else 0
        )
        evidence_data = {
            "accepted": bool(ok),
            "max_dev_um": repr(max_dev),
            "pinned_subset": sorted(pinned),
        }
        evidence_sha256 = hashlib.sha256(
            json.dumps(
                evidence_data,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return BasisEvaluation(
            accepted=bool(ok),
            failed_gate=None if ok else "equivalence_to_anchor",
            evidence_sha256=evidence_sha256,
            residual_cost_bits=residual_units,
        )

    basis_result = search_minimal_basis(
        atom_costs=atom_costs,
        evaluate=evaluate_basis,
        exact_limit=budget,
        beam_width=beam_width,
    )
    best_subset = frozenset(basis_result.selected_atom_ids) | forced_pinned
    best_mdl = mdl_cost(
        intent,
        set(best_subset),
        care_mask_um=care_mask_um,
        model=m,
        oracle=oracle,
    )
    method = basis_result.method

    all_dev = set(lattice.all_devices)
    dropped = tuple(sorted(all_dev - set(best_subset)))
    rel_for_dropped = {d: lattice.relations_for.get(d, []) for d in dropped}

    result = GlobalResult(
        pinned_subset=tuple(sorted(best_subset)),
        dropped=dropped,
        mdl=best_mdl,
        total=len(all_dev),
        oracle_calls=basis_result.oracle_calls,
        method=method,
        care_mask_um=care_mask_um,
        relations_for_dropped=rel_for_dropped,
    )

    # ---- global-vs-greedy comparison (the order-independence payoff) ---------- #
    if compare_greedy and equiv_fn is None:
        greedy = _greedy_converge(intent, care_mask_um, tech=tech)
        greedy_pins = set(greedy.pinned_residual)
        greedy_mdl = mdl_cost(
            intent, greedy_pins, care_mask_um=care_mask_um, model=m, oracle=oracle
        )
        result.greedy = greedy.pinned_residual
        result.greedy_mdl = greedy_mdl
        result.beats_greedy = (
            len(best_subset) < len(greedy_pins) or best_mdl < greedy_mdl
        )

    return result


__all__ = [
    "RelaxationLattice",
    "relaxation_lattice",
    "MdlCostModel",
    "mdl_cost",
    "GlobalResult",
    "converge_global",
]
