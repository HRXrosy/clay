"""
Layout-aware repair operators — the LAYOUT_AWARE arm of the repair algebra.

The committed registry seed (``clay.repair._registry._OPERATOR_SEED``) covers
the sizing / bias / compensation / topology-preserving classes but leaves the
``RepairClass.LAYOUT_AWARE`` class empty: there was no operator that maps a
violated *layout* obligation (``clay.layout.intent.LayoutObligation``) onto a
legal routing move. This module supplies that arm without mutating the
committed seed — it defines three operators keyed for the layout-obligation
kinds and exposes a registry-builder the main session can wire.

The three operators correspond to the designer's three point-3 repair moves
for differential-pair coupling imbalance, in priority order:

  ① ``symmetrize_routing`` — make a differential pair's coupling symmetric so
     the aggressor injection cancels in the difference. This is the PRIMARY
     fix for any imbalance obligation (matched_cc_balance / kickback /
     clk_regen): symmetry removes the *imbalance* directly without lengthening
     wires or adding capacitance. knob: ``diff_routing_symmetry``.

  ② ``insert_gnd_shield`` — drop a grounded shield between an aggressor and the
     victim / regen pair, cutting the coupling that drives the imbalance.
     Applies to aggressor-driven obligations (clk_regen / kickback).
     knob: ``gnd_shield``.

  ③ ``increase_spacing`` — pull the aggressor and victim routing apart to lower
     the coupling magnitude. Lowest priority because more spacing lengthens
     wires and raises the line's own capacitance. knob: ``wire_spacing``.

All three carry ``repair_class=RepairClass.LAYOUT_AWARE``.

Like the rest of ``clay.repair``, a layout-repair proposal lives strictly on
the repair axis: ``propose_layout_repair`` NEVER certifies a verification PASS.
A passing obligation yields a NO-OP; the verifier remains the only authority
that turns a layout obligation green.

This module imports the registry, status and layout-intent contracts
READ-ONLY; it edits no committed module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

from clay.layout.intent import LayoutObligation

from ._registry import (
    RepairClass,
    RepairOperator,
    RepairRegistry,
    _DEFERRED_CLASSES,
    default_registry,
)
from ._status import RepairStatus

# ─── Layout obligation kinds (mirrors clay.layout.intent._BUDGET_KINDS) ──────
#
# Kept as module constants so the operator keys and the dispatch table cannot
# drift from the obligation kinds the intent layer emits.
KIND_MATCHED = "matched_cc_balance"
KIND_KICKBACK = "kickback"
KIND_CLK_REGEN = "clk_regen"
# A declared node-cap budget (`@budget(parasitic_cap, ...)`) is measured as a
# cap-balance imbalance, so it is repaired the same way matched_cc_balance is
# (routing symmetry — no single aggressor to shield).
KIND_PARASITIC_CAP = "parasitic_cap"

# Canonical target attribute every layout obligation repairs: the RELATIVE
# coupling imbalance between the two nets of the obligation's pair. Using a
# single canonical attr keeps the (role_kind, target_attr) registry keys
# stable and lets the kind-only index resolve every obligation.
_LAYOUT_TARGET_ATTR = "coupling_imbalance"


# ─── The three LAYOUT_AWARE operators ────────────────────────────────────────
#
# role_kind is the layout-obligation kind (NOT a functional role) so these keys
# never collide with the committed functional-role seed. monotone_sign is -1
# for all three: increasing the knob (more symmetry / more shielding / more
# spacing) LOWERS the coupling imbalance toward its budget.

SYMMETRIZE_ROUTING = RepairOperator(
    role_kind=KIND_MATCHED,
    target_attr=_LAYOUT_TARGET_ATTR,
    knobs=("diff_routing_symmetry", "track_mirroring", "via_balance"),
    repair_class=RepairClass.LAYOUT_AWARE,
    monotone_sign=-1,  # more routing symmetry -> less imbalance
)

INSERT_GND_SHIELD = RepairOperator(
    role_kind=KIND_CLK_REGEN,
    target_attr=_LAYOUT_TARGET_ATTR,
    knobs=("gnd_shield", "shield_coverage", "shield_net"),
    repair_class=RepairClass.LAYOUT_AWARE,
    monotone_sign=-1,  # more shield coverage -> less aggressor coupling
)

INCREASE_SPACING = RepairOperator(
    role_kind=KIND_KICKBACK,
    target_attr=_LAYOUT_TARGET_ATTR,
    knobs=("wire_spacing", "keepout"),
    repair_class=RepairClass.LAYOUT_AWARE,
    monotone_sign=-1,  # more spacing -> less coupling magnitude
)

LAYOUT_OPERATORS: Tuple[RepairOperator, ...] = (
    SYMMETRIZE_ROUTING,
    INSERT_GND_SHIELD,
    INCREASE_SPACING,
)


# ─── Per-obligation-kind operator ordering ──────────────────────────────────
#
# Each obligation kind maps to the operators that can close it, in priority
# order (the proposer picks the first):
#   * matched_cc_balance -> symmetry only (no single aggressor to shield; the
#     defect is a routing-asymmetry between the two matched nets).
#   * clk_regen / kickback -> symmetrize first (cancel in the difference), then
#     a gnd shield (cut the aggressor coupling), then more spacing (last
#     resort: it raises C / lengthens wires).
_OPERATOR_ORDER: dict[str, Tuple[RepairOperator, ...]] = {
    KIND_MATCHED: (SYMMETRIZE_ROUTING,),
    KIND_PARASITIC_CAP: (SYMMETRIZE_ROUTING,),
    KIND_CLK_REGEN: (SYMMETRIZE_ROUTING, INSERT_GND_SHIELD, INCREASE_SPACING),
    KIND_KICKBACK: (SYMMETRIZE_ROUTING, INSERT_GND_SHIELD, INCREASE_SPACING),
}

# Human-readable knob -> directive verb, used to build the proposal directive.
_DIRECTIVE_VERB: dict[str, str] = {
    "diff_routing_symmetry": "symmetrize",
    "gnd_shield": "shield",
    "wire_spacing": "space",
}


# ─── Registration ────────────────────────────────────────────────────────────


def register_layout_operators(registry: RepairRegistry) -> RepairRegistry:
    """Register the three LAYOUT_AWARE operators on an existing registry.

    Returns the same registry (mutated) for chaining. The main session can
    call this on its own ``default_registry()`` instance to wire the layout
    arm without this module touching the committed ``_OPERATOR_SEED``.
    """
    for op in LAYOUT_OPERATORS:
        registry.register(op)
    return registry


def layout_repair_registry() -> RepairRegistry:
    """A fresh ``default_registry()`` extended with the layout operators.

    After this call ``RepairClass.LAYOUT_AWARE`` is no longer empty: the
    returned registry resolves every layout-obligation kind to its operator.
    """
    return register_layout_operators(default_registry())


def layout_operators_in(registry: RepairRegistry) -> Tuple[RepairOperator, ...]:
    """Every LAYOUT_AWARE operator currently registered in ``registry``."""
    return tuple(
        op for op in registry.operators()
        if op.repair_class is RepairClass.LAYOUT_AWARE
    )


# ─── Actionability of the LAYOUT_AWARE class ─────────────────────────────────


def layout_repair_class_actionable() -> bool:
    """True iff the LAYOUT_AWARE repair class is actionable (not deferred).

    A repair class is *deferred* when it sits in
    ``_registry._DEFERRED_CLASSES`` (currently only ``topology_changing``).
    LAYOUT_AWARE is a supported class, so a proposed layout repair can be
    acted on rather than parked as REPAIR_UNSUPPORTED.
    """
    return RepairClass.LAYOUT_AWARE not in _DEFERRED_CLASSES


# ─── Proposal ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayoutRepairProposal:
    """A concrete layout-routing repair move for one failing obligation.

    Lives on the repair axis only. ``progress_metric`` is the amount of
    imbalance (``obligation.over_budget``) the move is expected to remove; it
    is NOT a verification result. ``certifies_verification_pass`` is wired to
    the repair-status invariant and is always False.
    """

    operator: RepairOperator
    knob: str
    directive: str
    pair: Tuple[str, str]
    progress_metric: float
    kind: str
    status: RepairStatus = RepairStatus.REPAIRABLE
    preserved_obligations: Tuple[str, ...] = (
        "schematic_connectivity",
        "measurement_obligations",
    )
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def certifies_verification_pass(self) -> bool:
        """Always False — a repair proposal never asserts a verify PASS."""
        return self.status.certifies_verification_pass

    def as_dict(self) -> dict:
        """Plain-dict view (the contract ``propose_layout_repair`` returns)."""
        return {
            "operator": self.operator,
            "knob": self.knob,
            "directive": self.directive,
            "pair": self.pair,
            "progress_metric": self.progress_metric,
            "kind": self.kind,
            "status": self.status,
            "preserved_obligations": self.preserved_obligations,
        }


def _build_directive(operator: RepairOperator, pair: Tuple[str, str]) -> str:
    """Human-readable directive for an operator + net pair."""
    a, b = pair
    verb = _DIRECTIVE_VERB.get(operator.primary_knob, operator.primary_knob)
    if operator is SYMMETRIZE_ROUTING:
        return f"symmetrize {a}/{b} differential routing"
    if operator is INSERT_GND_SHIELD:
        return f"shield clk from {a},{b}"
    if operator is INCREASE_SPACING:
        return f"increase spacing between {a} and {b}"
    return f"{verb} {a},{b}"


def propose_layout_repair(
    obl: LayoutObligation,
) -> Union[dict, RepairStatus, None]:
    """Propose a layout-routing repair for one layout obligation.

    Behaviour:
      * A PASSING obligation has nothing to repair -> returns ``None`` (NO-OP).
      * A FAILING obligation -> returns a proposal dict::

            {
              "operator": <RepairOperator>,           # LAYOUT_AWARE
              "knob": <primary knob name>,
              "directive": <human text>,
              "pair": (netA, netB),
              "progress_metric": obl.over_budget,     # imbalance to remove
              "kind": obl.kind,
              "status": RepairStatus.REPAIRABLE,
              "preserved_obligations": (...),
            }

        The operator is chosen by ``_OPERATOR_ORDER[obl.kind]`` (first =
        highest priority): symmetrize_routing for every kind, falling back to
        insert_gnd_shield then increase_spacing for the aggressor-driven kinds.
      * An obligation of an unknown kind -> ``RepairStatus.NO_REPAIR_OPERATOR``.

    The returned proposal NEVER certifies a verification pass: its embedded
    ``RepairStatus`` is REPAIRABLE and
    ``RepairStatus.REPAIRABLE.certifies_verification_pass`` is False.
    """
    # NO-OP: a satisfied obligation needs no repair. Returning None keeps the
    # "already satisfied" signal distinct from "broken algebra".
    if obl.status == "pass":
        return None

    order = _OPERATOR_ORDER.get(obl.kind)
    if not order:
        # No layout operator covers this kind.
        return RepairStatus.NO_REPAIR_OPERATOR

    operator = order[0]
    proposal = LayoutRepairProposal(
        operator=operator,
        knob=operator.primary_knob,
        directive=_build_directive(operator, obl.pair),
        pair=obl.pair,
        progress_metric=obl.over_budget,
        kind=obl.kind,
        status=RepairStatus.REPAIRABLE,
    )
    return proposal.as_dict()


def _obligation_from_serialized(entry: dict) -> LayoutObligation:
    """Rebuild a :class:`LayoutObligation` from one serialized arbiter entry.

    The arbiter hook (:func:`clay.layout.intent.evaluate_intent_for_flow`)
    returns obligations as plain JSON-serializable dicts (kind / pair /
    measured_imbalance / rel_tol / status / over_budget / missing_nets). The
    SignStep's optional auto-repair retry rebuilds the obligation object here so
    it can route through the unchanged :func:`propose_layout_repair` algebra.
    """
    pair = entry["pair"]
    measured = entry.get("measured_imbalance")
    return LayoutObligation(
        kind=str(entry["kind"]),
        pair=(str(pair[0]), str(pair[1])),
        measured_imbalance=None if measured is None else float(measured),
        rel_tol=float(entry["rel_tol"]),
        missing_nets=tuple(str(n) for n in entry.get("missing_nets", ())),
    )


def propose_repairs_for_flow(arbiter_result: dict) -> list[dict]:
    """Repair proposals for every non-passing obligation in an arbiter result.

    Given the dict :func:`clay.layout.intent.evaluate_intent_for_flow` returns,
    this is the enforcing seam the engine's ``SignStep`` calls when the intent
    verdict is ``"fail"``: it proposes a concrete layout-routing move for each
    obligation that did NOT pass, in repair priority order, so the step can
    optionally apply one and re-run the flow.

    A PASSING obligation yields no proposal (``propose_layout_repair`` returns a
    NO-OP for it). An UNMEASURABLE obligation (a needed net absent from the
    extraction) still surfaces a proposal whose ``progress_metric`` is ``0.0`` --
    the imbalance is unknown, so there is no over-budget magnitude, but the
    routing move (symmetrize the pair) is still the actionable response. An
    obligation of a kind no operator covers is dropped (no actionable move).

    Each element is the plain-dict proposal ``propose_layout_repair`` returns;
    the list is ordered by the arbiter's obligation order. NEVER certifies a
    verification pass — every proposal's status is REPAIRABLE.
    """
    proposals: list[dict] = []
    for entry in arbiter_result.get("obligations", []):
        if str(entry.get("status")) == "pass":
            continue
        obl = _obligation_from_serialized(entry)
        proposal = propose_layout_repair(obl)
        if isinstance(proposal, dict):
            proposals.append(proposal)
    return proposals


__all__ = [
    # operators
    "SYMMETRIZE_ROUTING",
    "INSERT_GND_SHIELD",
    "INCREASE_SPACING",
    "LAYOUT_OPERATORS",
    # registration
    "register_layout_operators",
    "layout_repair_registry",
    "layout_operators_in",
    # actionability
    "layout_repair_class_actionable",
    # proposal
    "LayoutRepairProposal",
    "propose_layout_repair",
    "propose_repairs_for_flow",
    # kind constants
    "KIND_MATCHED",
    "KIND_KICKBACK",
    "KIND_CLK_REGEN",
    "KIND_PARASITIC_CAP",
]
