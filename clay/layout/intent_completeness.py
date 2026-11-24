"""NO-SILENT-IGNORE completeness gate for layout intents.

The soundness rule this module enforces: *no declared layout intent may be
"parsed but silently ignored."* A cell that DECLARES an intent (a matched-cap
budget, an LDE budget, an interdigitation, a common-centroid, a place-group, a
symmetric/adjacent/well-isolated annotation, a shield, ...) is making a CLAIM
about its layout. If nothing ever CHECKS that claim, the layout silently fails to
honor it and nothing flags the dishonesty -- the proof token signs anyway.

This module is the META-CHECK that makes silent-ignore impossible. It does NOT
measure any geometry or parasitics itself (the measurement layer lives in
``clay.layout.intent_obligations`` + ``clay.layout.geometric_intent_checks``).
It compares two sets:

  * the DECLARED-intent set  -- every intent surface a cell declares, read off
    its :class:`~clay.layout.intent.LayoutIntent` (net-level: budgets, LDE
    budgets, shields, place-groups, resistance budgets, objective/area) and,
    when supplied, its :class:`~clay.layout.annotations.AnnotatedCell`
    (device-level: matching / interdigitated / symmetric / adjacent /
    well_isolated / common_centroid / guard_ring);
  * the OBLIGATION set        -- every :class:`~clay.layout.intent.LayoutObligation`
    actually produced by the checkers for this cell.

For every declared intent with NO covering obligation, this module emits a
BLOCKING obligation of kind ``unchecked_intent:<kind>`` (status
``unmeasurable``, ``measured_imbalance=None``) carrying a human-readable reason.
Because that obligation is non-pass, it withholds the proof token exactly the way
a failing DRC row does. So the outcome is binary and honest: either a real check
PASSES, or the token is BLOCKED with an explicit "this intent is declared but not
verified" reason. There is no third "silently passed" outcome.

GENERIC: the gate is driven by the DECLARED-intent set vs the OBLIGATION set.
There is no per-circuit / cell-name branching anywhere in this module -- adding a
new declared intent kind to a cell automatically makes the gate demand a
matching obligation, and adding a new obligation checker automatically satisfies
that demand the moment it emits an obligation for the declared target.

All of this reads only the intent dataclasses + the obligation list, so it is
unit-testable WITHOUT Magic/Docker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .intent import LayoutIntent, LayoutObligation

# ─────────────────────────────────────────────────────────────────────────────
# Declared-intent enumeration
# ─────────────────────────────────────────────────────────────────────────────
#
# A DeclaredIntent is one atomic claim a cell makes about its layout. ``kind`` is
# the intent family (the user-facing keyword normalised: ``matched_cc_balance``,
# ``lde:well_edge``, ``interdigitated``, ``common_centroid``, ``place_group``,
# ``symmetric``, ``adjacent``, ``well_isolated``, ``shield``, ...). ``targets`` is
# the unordered set of the nets/devices the claim is about, used to match a
# declared intent to the obligation that checks THAT specific instance (so a cell
# with two interdigitated pairs needs an obligation for EACH pair, not just one).


@dataclass(frozen=True)
class DeclaredIntent:
    """One atomic declared layout-intent claim, awaiting a covering obligation."""

    kind: str
    targets: FrozenSet[str]
    # A short surface label for the reason string (e.g. "M_in_p~M_in_n").
    label: str
    declaration_ids: Tuple[str, ...] = ()

    @property
    def key(self) -> Tuple[str, FrozenSet[str], Tuple[str, ...]]:
        return (self.kind, self.targets, self.declaration_ids)


def _label(targets: Sequence[str]) -> str:
    return "~".join(str(t) for t in targets) if targets else "(cell)"


# Obligation-kind PREFIX -> declared-intent FAMILY. An obligation whose ``kind``
# starts with one of these prefixes COVERS a declared intent of the mapped family
# (when the targets also match). The prefix form is deliberately permissive so a
# new geometric checker may name its kind ``interdigitated``, ``interdig:abba``,
# or ``geom:interdigitated`` and still be recognised as covering an
# ``interdigitated`` declaration -- the completeness gate does not need to be
# updated when the measurement layer picks a concrete kind string. Pure-cap kinds
# (matched_cc_balance / parasitic_cap / kickback / clk_regen) and ``lde:`` are
# matched by their own families.
_KIND_FAMILY_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("interdigitated", "interdigitated"),
    ("interdig", "interdigitated"),
    ("matching", "matching"),
    ("common_centroid", "common_centroid"),
    ("common-centroid", "common_centroid"),
    ("commoncentroid", "common_centroid"),
    ("place_group", "place_group"),
    ("placegroup", "place_group"),
    ("symmetric", "symmetric"),
    ("symmetry", "symmetric"),
    ("adjacent", "adjacent"),
    ("well_isolated", "well_isolated"),
    ("well-isolated", "well_isolated"),
    ("guard_ring", "guard_ring"),
    ("guardring", "guard_ring"),
    ("equal_size", "equal_size"),
    ("equalsize", "equal_size"),
    ("order", "order"),
    ("abut", "abut"),
    ("fix", "fix"),
    ("shield", "shield"),
    ("series_res", "series_res"),
)


def _obligation_family(kind: str) -> str:
    """Map an obligation ``kind`` to its declared-intent FAMILY.

    The cap families (matched_cc_balance / parasitic_cap / kickback / clk_regen)
    and the ``lde:<metric>`` family are their own kinds. Everything else is
    resolved through :data:`_KIND_FAMILY_PREFIXES` so a geometric checker's
    concrete kind string maps back to the family it covers without this module
    needing to know the exact string.
    """
    k = str(kind)
    kl = k.lower()
    if kl.startswith("lde:"):
        return k  # the declared LDE intent uses the same ``lde:<metric>`` kind
    if kl in ("matched_cc_balance", "parasitic_cap", "kickback", "clk_regen"):
        return kl
    for prefix, family in _KIND_FAMILY_PREFIXES:
        if kl.startswith(prefix):
            return family
    return kl


def enumerate_declared_intents(
    intent: LayoutIntent,
    annotated=None,
) -> List[DeclaredIntent]:
    """Every atomic declared layout-intent claim for a cell.

    Reads the net-level :class:`LayoutIntent` and (when supplied) the
    device-level :class:`~clay.layout.annotations.AnnotatedCell`. Returns one
    :class:`DeclaredIntent` per atomic claim, deduplicated by ``(kind, targets)``.

    Net-level surfaces:

      * each :class:`~clay.layout.intent.ParasiticBudget` -> its own kind
        (matched_cc_balance / parasitic_cap / kickback / clk_regen);
      * each :class:`~clay.layout.intent.LdeBudget` -> ``lde:<metric>``;
      * each :class:`~clay.layout.intent.Shield` -> ``shield``;
      * each :class:`~clay.layout.intent.PlaceGroup` -> ``place_group``;
      * each :class:`~clay.layout.intent.ResistanceBudget` -> ``series_res``.

    Device-level surfaces (from the ``annotate`` block; only when ``annotated`` is
    passed): ``@matching`` -> ``matching``, ``@interdigitated`` ->
    ``interdigitated``, ``@common_centroid`` -> ``common_centroid``,
    ``@symmetric`` -> ``symmetric``, ``@adjacent`` -> ``adjacent``,
    ``@well_isolated`` -> ``well_isolated``, and ``@guard_ring`` -> ``guard_ring``.
    Reciprocal matching declarations collapse to one unordered claim while their
    two directed declaration IDs remain independently required for coverage.

    Objective / area are placement DIRECTIVES (best-effort optimisation goals,
    not verifiable obligations) and are intentionally NOT enumerated as claims
    that must be checked -- they never gate the token, consistent with how
    :class:`PlaceGroup` is documented. ``place_group`` IS enumerated because a
    declared floorplan group is a concrete, checkable placement claim.
    """
    seen: set = set()
    out: List[DeclaredIntent] = []

    def _add(
        kind: str,
        targets: Sequence[str],
        declaration_ids: Sequence[str] = (),
    ) -> None:
        tset = frozenset(str(t) for t in targets)
        ids = tuple(sorted(str(item) for item in declaration_ids))
        key = (kind, tset, ids)
        if key in seen:
            return
        seen.add(key)
        out.append(
            DeclaredIntent(
                kind=kind,
                targets=tset,
                label=_label(list(targets)),
                declaration_ids=ids,
            )
        )

    for b in intent.budgets:
        _add(b.kind, tuple(b.pair))
    for lb in getattr(intent, "lde_budgets", ()) or ():
        _add(f"lde:{lb.metric}", tuple(lb.pair))
    for sh in intent.shields:
        _add("shield", (sh.aggressor, sh.victim))
    from .intent import place_group_floorplan_row

    for g in getattr(intent, "place_groups", ()) or ():
        if place_group_floorplan_row(g) is None and getattr(g, "col", None) is None:
            continue
        # A place-group's identity is BOTH its group name AND its device
        # membership: a placement obligation may name the group by its GROUP NAME
        # (e.g. ``place_group_realized`` with pair ``(group, row-label)``) or by a
        # device subset. Including the group name in the target set lets either
        # form satisfy coverage (``_covers`` matches on set overlap), so the gate
        # is decoupled from which identity the geometric checker chose.
        targets = tuple(g.devices) + ((g.group,) if g.group else ())
        _add("place_group", targets if targets else (g.group,))
    for rb in getattr(intent, "resistance_budgets", ()) or ():
        _add("series_res", tuple(rb.pair))
    for (a, b, _rel) in getattr(intent, "order_constraints", ()) or ():
        _add("order", (a, b))
    for (a, b) in getattr(intent, "abut_pairs", ()) or ():
        _add("abut", (a, b))
    for matching in getattr(intent, "matching_intents", ()) or ():
        if (
            matching.requires_interdigitation
            and matching.declaration_ids
            == matching.interdigitation_declaration_ids
        ):
            continue
        _add("matching", matching.members, matching.declaration_ids)

    if annotated is not None:
        _add_device_annotation_intents(annotated, _add)

    return out


# The ``annotate``-block annotation classes whose presence is a declared,
# checkable LAYOUT claim. Keyed by the class NAME (string) so this module does not
# hard-import the annotation classes (keeps the dependency one-directional and the
# gate importable even if the annotations module is reorganised).
_ANNOTATION_CLASS_TO_KIND: Dict[str, str] = {
    "Matching": "matching",
    "Interdigitated": "interdigitated",
    "CommonCentroid": "common_centroid",
    "Symmetric": "symmetric",
    "Adjacent": "adjacent",
    "WellIsolated": "well_isolated",
    "GuardRing": "guard_ring",
    "EqualSize": "equal_size",
    "Fix": "fix",
}


def _annotation_targets(target: str, ann) -> Tuple[str, ...]:
    """Resolve the target set for one device-level annotation.

    The annotation is keyed by the device ``target`` it was declared on; most
    annotation classes also carry a partner / device list (``other`` /
    ``devices`` / ``partner``). The target set is the union so a pairwise
    ``M_in_p: @interdigitated(M_in_n, ...)`` claims the unordered ``{M_in_p,
    M_in_n}`` and matches an obligation produced for either ordering.
    """
    targets = [str(target)]
    for attr in ("other", "partner", "with_device", "neighbor", "device"):
        val = getattr(ann, attr, None)
        if val:
            targets.append(str(val))
    devices = getattr(ann, "devices", None)
    if devices:
        targets.extend(str(d) for d in devices)
    return tuple(targets)


def _add_device_annotation_intents(annotated, _add) -> None:
    """Enumerate the device-level ``annotate``-block claims onto the declared set."""
    annotations = getattr(annotated, "annotations", None)
    if not isinstance(annotations, dict):
        return
    matching_by_pair: Dict[Tuple[str, str], List[str]] = {}
    for target, anns in annotations.items():
        for ann in anns or ():
            kind = _ANNOTATION_CLASS_TO_KIND.get(type(ann).__name__)
            if kind is None:
                continue
            if kind == "matching":
                pair = tuple(sorted((str(target), str(getattr(ann, "other", "")))))
                declaration_id = (
                    f"matching:{getattr(annotated, 'cell_name', '')}:"
                    f"{target}->{getattr(ann, 'other', '')}"
                )
                matching_by_pair.setdefault(pair, []).append(declaration_id)
                continue
            _add(kind, _annotation_targets(str(target), ann))
    for pair in sorted(matching_by_pair):
        _add("matching", pair, tuple(matching_by_pair[pair]))


# ─────────────────────────────────────────────────────────────────────────────
# Coverage: is every declared intent backed by an obligation?
# ─────────────────────────────────────────────────────────────────────────────


def _obligation_pair(declared: DeclaredIntent) -> Tuple[str, str]:
    """A well-formed 2-tuple identity label for an ``unchecked_intent`` obligation.

    The synthetic blocking obligation carries no measured pair imbalance, but the
    signoff row builder reads ``obligation.pair`` as exactly two elements. A
    declared intent with two or more targets uses its first two (sorted, stable);
    a single-target or cell-level claim pads the second slot so the row stays
    well-formed. ``d.label`` carries the FULL target set for the human reason.
    """
    targets = sorted(declared.targets)
    if len(targets) >= 2:
        return (targets[0], targets[1])
    if len(targets) == 1:
        return (targets[0], "(cell)")
    return ("(cell)", "(cell)")


def _obligation_targets(ob: LayoutObligation) -> FrozenSet[str]:
    return frozenset(str(t) for t in ob.pair)


def _covers(declared: DeclaredIntent, ob: LayoutObligation) -> bool:
    """True iff obligation ``ob`` is a check OF the declared intent ``declared``.

    An obligation covers a declared intent when (1) the obligation's kind maps to
    the SAME family as the declared intent's kind, and (2) their target sets match
    -- either identically, or one is a subset of the other (a place-group
    obligation may name a representative subset of the declared device list, and a
    pairwise annotation obligation may name just the ordered pair). The subset
    rule never lets an UNRELATED obligation cover a declaration: the families must
    already agree and the targets must overlap as sub/superset.

    PLACE-GROUP relaxation: a placement checker (e.g. ``place_group_realized``)
    may identify the group by its GROUP NAME paired with a row/column LABEL
    (``(group, "row2_colNone")``) rather than by its device membership. Since the
    declared intent's targets include the group name, a NON-EMPTY intersection is
    accepted as coverage for the place_group family -- the realized obligation
    naming THIS group proves the group was processed. The family agreement gate
    already prevents an unrelated obligation from matching.
    """
    if _obligation_family(ob.kind) != _obligation_family(declared.kind):
        return False
    ob_t = _obligation_targets(ob)
    dt = declared.targets
    if _obligation_family(declared.kind) == "matching":
        # Direct matching is declaration-complete and member-complete.  A proxy
        # carrying one reciprocal direction, or only a subset of the members,
        # cannot cover the authored claim.
        return (
            ob_t == dt
            and frozenset(getattr(ob, "declaration_ids", ()) or ())
            == frozenset(declared.declaration_ids)
        )
    if not dt:
        # A cell-level claim (no specific targets) is covered by any obligation of
        # its family.
        return True
    if not ob_t:
        return False
    if _obligation_family(declared.kind) == "place_group":
        # Group-name OR device-membership overlap suffices (see docstring).
        return bool(ob_t & dt)
    return ob_t == dt or ob_t <= dt or dt <= ob_t


# ─────────────────────────────────────────────────────────────────────────────
# Enforcement policy: which declared-intent families the gate BLOCKS on.
# ─────────────────────────────────────────────────────────────────────────────
#
# The gate blocks a declared intent that has NO covering obligation ONLY when its
# family is ENFORCED here. Two families are deliberately NOT enforced, because the
# project already records them through a DOCUMENTED non-blocking provenance path
# (so they are not "silently" ignored -- their handling is explicit and visible):
#
#   * ``shield``      -- a declared shield is realised geometrically and CHECKED
#     by ``clay.layout.intent_obligations.check_shield_intent``, which needs a
#     ROUTED cell (tagged shapes), NOT the cap-only PexReport the certify path
#     carries. Blocking every cap-only certify on a missing routed cell would
#     de-certify every comparator signoff (whose intent declares clk->regen
#     shields) even when the shields are present in the routed layout. The shield
#     verdict belongs to the routed-cell signoff, not the parasitic certify, so
#     the cap-path gate records-but-does-not-block it.
#   * ``series_res``  -- Clay has no PEX resistance-imbalance metric, so a series
#     resistance budget is, by an EXPLICIT design decision (see
#     ``clay.layout.intent._RESISTANCE_BUDGET_KINDS`` + ``ResistanceBudget``),
#     recorded as a declared-but-unverified intent rather than a permanently-
#     unmeasurable block on every sign-off. The completeness gate honours that
#     existing, documented contract instead of overriding it.
#
# Everything else IS enforced: a declared matched_cc_balance / parasitic_cap /
# kickback / clk_regen budget, an ``lde:*`` budget, a ``place_group``, and the
# device-level geometric annotations (interdigitated / common_centroid /
# symmetric / adjacent / well_isolated / guard_ring) MUST be backed by a passing
# obligation or the token is withheld. These are the families that, today, are
# "parsed and carried but not consumed" -- the genuine silent-ignores the gate
# exists to make impossible.
_UNENFORCED_FAMILIES: FrozenSet[str] = frozenset({"shield", "series_res"})


def completeness_obligations(
    intent: LayoutIntent,
    obligations: Sequence[LayoutObligation],
    *,
    annotated=None,
) -> List[LayoutObligation]:
    """The BLOCKING ``unchecked_intent:<kind>`` obligations for this cell.

    For every declared intent (see :func:`enumerate_declared_intents`) that NO
    obligation in ``obligations`` covers (see :func:`_covers`), emit one
    :class:`LayoutObligation` of kind ``unchecked_intent:<declared-kind>`` with
    ``measured_imbalance=None`` (status ``unmeasurable``) and a ``missing_nets``
    reason of the form ``("declared_but_unchecked",)``. Such an obligation is
    non-pass, so once it is folded into the proof-token gate the token is WITHHELD
    -- a silently-ignored intent can no longer slip through as a free pass.

    Returns an EMPTY list when every declared intent is covered (the gate is a
    no-op on a fully-checked cell -- it only ever ADDS blocking rows, never
    removes or alters a real obligation). ``rel_tol`` on the synthetic obligation
    is a sentinel ``1.0`` (never used: an unmeasurable obligation has no measured
    value to compare).
    """
    declared = enumerate_declared_intents(intent, annotated=annotated)
    obs = list(obligations)
    extra: List[LayoutObligation] = []
    for d in declared:
        # Skip the families recorded through a documented non-blocking provenance
        # path (shield -> routed-cell check; series_res -> declared-but-unverified
        # ResistanceBudget). They are NOT silently ignored -- their handling is
        # explicit -- so the gate does not double-block them here.
        if _obligation_family(d.kind) in _UNENFORCED_FAMILIES:
            continue
        if any(_covers(d, ob) for ob in obs):
            continue
        extra.append(
            LayoutObligation(
                kind=f"unchecked_intent:{d.kind}",
                # ``LayoutObligation.pair`` is consumed as an exactly-2-tuple by the
                # signoff row builder; pad a single-target / cell-level claim so the
                # row stays well-formed (the pair is only an identity label here, the
                # obligation carries no measured pair imbalance).
                pair=_obligation_pair(d),
                measured_imbalance=None,
                rel_tol=1.0,
                missing_nets=("declared_but_unchecked", f"intent:{d.label}"),
                declaration_ids=d.declaration_ids,
            )
        )
    return extra


def assert_no_silent_ignore(
    intent: LayoutIntent,
    obligations: Sequence[LayoutObligation],
    *,
    annotated=None,
) -> List[LayoutObligation]:
    """Return ``obligations`` PLUS any ``unchecked_intent`` blocking rows.

    The single call the certify / signoff layer uses: it appends the completeness
    gate's blocking rows to the real obligation list so a downstream verdict /
    proof-token gate sees BOTH the measured obligations and the
    "declared-but-unchecked" blocks in one homogeneous list. A cell whose every
    declared intent is checked gets its obligation list back unchanged.
    """
    return list(obligations) + completeness_obligations(
        intent, obligations, annotated=annotated
    )


def completeness_report(
    intent: LayoutIntent,
    obligations: Sequence[LayoutObligation],
    *,
    annotated=None,
) -> Dict[str, object]:
    """A human/machine-readable audit of declared-intent coverage for one cell.

    Returns ``{"cell", "declared": [...], "covered": [...], "unchecked": [...],
    "complete": bool}`` where each ``declared`` entry records its kind, label, and
    whether an obligation covers it. ``complete`` is True iff every declared
    intent is covered (the token-eligible state for the completeness axis).
    """
    declared = enumerate_declared_intents(intent, annotated=annotated)
    obs = list(obligations)
    covered: List[Dict[str, object]] = []
    unchecked: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    for d in declared:
        is_covered = any(_covers(d, ob) for ob in obs)
        row = {
            "kind": d.kind,
            "label": d.label,
            "declaration_ids": list(d.declaration_ids),
            "covered": is_covered,
        }
        rows.append(row)
        (covered if is_covered else unchecked).append(row)
    return {
        "cell": intent.cell,
        "declared": rows,
        "covered": covered,
        "unchecked": unchecked,
        "complete": not unchecked,
    }


__all__ = [
    "DeclaredIntent",
    "enumerate_declared_intents",
    "completeness_obligations",
    "assert_no_silent_ignore",
    "completeness_report",
]
