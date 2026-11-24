"""Schematic <-> layout consistency checker for the layout-intent contract.

The layout-intent contract (:mod:`clay.layout.intent`) declares a cell's
net-level parasitic intent -- which differential pairs must stay matched -- and
:mod:`clay.layout.intent_obligations` measures that intent on a REAL PEX
extraction. That answers "did the layout deliver what the layout contract
declared?".

This module answers the prior, stronger question: does the layout contract
itself honour what the SCHEMATIC declares? The schematic-level transistor
symmetry (a differential pair, a cross-coupled latch -- devices that play
mirror-image roles in the CIRCUIT) is the GROUND TRUTH of intended symmetry.
A :class:`~clay.layout.intent.LayoutIntent` could silently drop one of those
pairs (a symmetry the schematic intended but the layout never budgeted for), or
could declare a matched pair the schematic never asked for. Either is a
schematic/layout inconsistency that passes every PEX obligation yet still means
the layout does not faithfully realise the circuit.

The check is three-way:

  * COVERAGE   -- every schematic-declared diff pair has a corresponding
    ``matched_cc_balance`` budget in the intent;
  * EXTRA      -- any layout matched pair the schematic did NOT declare;
  * DELIVERED  -- (when obligations are supplied) whether each covered pair's
    matched obligation actually PASSED on the real layout.

Where the schematic-declared pairs come from
---------------------------------------------
The schematic side has a real symmetry source:
:func:`clay.layout.symmetry_derive.derive_symmetry_groups`, which reads the
``topology`` block (device class + sizing + the nets each pin connects) and
returns :class:`~clay.layout.symmetry_derive.SymmetryGroup` objects whose
``differential_nets`` ARE the circuit-derived differential net pairs. The
faithful comparator schematic export
(:data:`clay.schematic.export.COMPARATOR_DIFF_PAIRS`) is the same set, hand-kept
from the topology. :func:`schematic_pairs_from_symmetry_groups` extracts the net
pairs out of derived groups so a caller that has a parsed cell can feed the real
circuit symmetry straight in.

This module does NOT itself parse a netlist into a schematic -- it never
fabricates a schematic source. The caller supplies the schematic-declared pairs
(from ``derive_symmetry_groups`` on a real cell, from the schematic export
constant, or from an explicit designer declaration), and this checker verifies
the layout intent against them.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .intent import LayoutIntent, LayoutObligation


# A pair is compared UNORDERED: ``(a, b)`` and ``(b, a)`` are the same diff pair.
_PairKey = FrozenSet[str]


def _pair_key(pair: Sequence[str]) -> _PairKey:
    """Unordered comparison key for a net pair; rejects a degenerate self-pair."""
    a, b = tuple(pair)
    if a == b:
        raise ValueError(f"a differential pair must name two distinct nets, got {pair!r}")
    return frozenset((str(a), str(b)))


def _ordered(pair: Sequence[str]) -> Tuple[str, str]:
    """Canonical, display-stable ordering of a net pair (sorted)."""
    a, b = tuple(pair)
    return (str(a), str(b)) if str(a) <= str(b) else (str(b), str(a))


def schematic_pairs_from_symmetry_groups(
    groups: Iterable[object],
) -> Tuple[Tuple[str, str], ...]:
    """Pull the differential net pairs out of circuit-derived symmetry groups.

    Helper for the real schematic-symmetry source: given an iterable of
    :class:`~clay.layout.symmetry_derive.SymmetryGroup` (the output of
    :func:`~clay.layout.symmetry_derive.derive_symmetry_groups` on a parsed
    ``cell``), collect each group's ``differential_nets`` into a deduplicated,
    order-stable tuple of net pairs. These pairs are the SCHEMATIC's ground-truth
    intended symmetry and are exactly what
    :func:`check_schematic_layout_consistency` expects as ``schematic_pairs``.

    Pairs are deduplicated by UNORDERED net membership (a pair shared across
    merged groups appears once); the first occurrence's net order is preserved.
    Accepts any object exposing a ``differential_nets`` attribute, so it does not
    hard-depend on the concrete dataclass.
    """
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for g in groups:
        for pair in getattr(g, "differential_nets", ()) or ():
            a, b = tuple(pair)
            if a == b:
                continue
            key = frozenset((str(a), str(b)))
            if key in seen:
                continue
            seen.add(key)
            out.append((str(a), str(b)))
    return tuple(out)


def _matched_pairs_of_intent(intent: LayoutIntent) -> List[Tuple[str, str]]:
    """The diff pairs the layout intent treats as matched.

    A layout pair is "matched" iff it carries a ``matched_cc_balance`` budget --
    that budget IS the layout's declaration that the two nets must stay
    parasitically balanced. ``intent.diff_pairs`` is the same set in practice (the
    deriver emits one budget per diff pair), but the budget list is the
    authoritative source: it is what the obligation checker measures, so a pair
    with a matched budget is the pair the layout actually committed to delivering.
    """
    return [tuple(b.pair) for b in intent.budgets if b.kind == "matched_cc_balance"]


def _delivered_status(
    pair_key: _PairKey,
    obligations: Sequence[LayoutObligation],
) -> str:
    """Verdict for one covered pair from the checked obligations.

    Looks up the ``matched_cc_balance`` obligation for ``pair_key`` (matched
    UNORDERED) and returns its terminal state: ``"pass"`` / ``"fail"`` /
    ``"unmeasurable"``. If no matched obligation names this pair, returns
    ``"unmeasurable"`` -- the layout declared the pair but no extraction verdict
    exists for it, which (like an absent-net obligation) must NOT read as a pass.
    """
    for o in obligations:
        if o.kind != "matched_cc_balance":
            continue
        if frozenset((str(o.pair[0]), str(o.pair[1]))) == pair_key:
            return o.status
    return "unmeasurable"


def check_schematic_layout_consistency(
    schematic_pairs: Iterable[Sequence[str]],
    intent: LayoutIntent,
    obligations: Optional[Sequence[LayoutObligation]] = None,
) -> Dict[str, object]:
    """Check that the layout intent honours the schematic-declared symmetry.

    ``schematic_pairs`` is the GROUND TRUTH set of differential net pairs the
    SCHEMATIC declares (an iterable of net 2-tuples -- e.g. from
    :func:`schematic_pairs_from_symmetry_groups` on circuit-derived symmetry
    groups, from :data:`clay.schematic.export.COMPARATOR_DIFF_PAIRS`, or supplied
    explicitly by the designer). ``intent`` is the
    :class:`~clay.layout.intent.LayoutIntent` whose ``matched_cc_balance`` budgets
    are what the layout DELIVERS. ``obligations`` are the checked
    :class:`~clay.layout.intent.LayoutObligation` list from
    :func:`~clay.layout.intent_obligations.check_layout_intent` against a real PEX
    report; when omitted, only the structural coverage is checked.

    Pair comparison is UNORDERED throughout: net order within a pair never
    matters.

    Returns a dict with:

      * ``consistent``       -- ``bool``. True iff no schematic pair is missing
        from the layout AND (when ``obligations`` is given) every covered pair was
        DELIVERED with a ``"pass"`` obligation. EXTRA layout pairs do NOT by
        themselves break consistency (the schematic is ground truth for what MUST
        be matched; an extra layout pair is reported for review, not a violation).
      * ``covered``          -- the schematic pairs that DO have a matched budget
        in the layout intent (display-ordered net pairs).
      * ``missing_in_layout``-- schematic-declared pairs with NO matched budget in
        the intent: a symmetry the schematic intended that the layout silently
        dropped. This is the blocking inconsistency.
      * ``extra_in_layout``  -- layout matched pairs the schematic did NOT declare:
        a pair the layout treats as matched but the circuit never asked for.
      * ``delivered``        -- ``{pair: "pass"|"fail"|"unmeasurable"}`` for each
        COVERED pair, present only when ``obligations`` was supplied (else ``{}``).
        The key is the canonical sorted net-pair tuple.
    """
    sch_keys: Dict[_PairKey, Tuple[str, str]] = {}
    for p in schematic_pairs:
        sch_keys.setdefault(_pair_key(p), _ordered(p))

    lay_keys: Dict[_PairKey, Tuple[str, str]] = {}
    for p in _matched_pairs_of_intent(intent):
        lay_keys.setdefault(_pair_key(p), _ordered(p))

    covered_keys = [k for k in sch_keys if k in lay_keys]
    missing_keys = [k for k in sch_keys if k not in lay_keys]
    extra_keys = [k for k in lay_keys if k not in sch_keys]

    covered = [sch_keys[k] for k in covered_keys]
    missing_in_layout = [sch_keys[k] for k in missing_keys]
    extra_in_layout = [lay_keys[k] for k in extra_keys]

    delivered: Dict[Tuple[str, str], str] = {}
    if obligations is not None:
        for k in covered_keys:
            delivered[sch_keys[k]] = _delivered_status(k, obligations)

    consistent = not missing_in_layout
    if obligations is not None:
        consistent = consistent and all(
            status == "pass" for status in delivered.values()
        )

    return {
        "consistent": bool(consistent),
        "covered": covered,
        "missing_in_layout": missing_in_layout,
        "extra_in_layout": extra_in_layout,
        "delivered": delivered,
    }


__all__ = [
    "check_schematic_layout_consistency",
    "schematic_pairs_from_symmetry_groups",
]
