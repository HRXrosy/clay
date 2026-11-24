"""Multi-step, HONEST layout-repair convergence loop for the comparator.

``clay.repair.layout_apply.close_layout_repair_loop`` closes ONE edge of the
layout-intent contract: it repairs a single obligation (``clk_regen``) by turning
one real router knob -- the grounded clk shield strip
(``build_routed_comparator(shield=True)``). This module generalizes that single
fix into a CONVERGENCE LOOP that iterates over ALL failing obligations, while
staying HONEST about which obligation kinds actually have a parameterized router
knob behind them and which can only be PROPOSED.

The honesty rule, made explicit
-------------------------------
A kind is "knob-addressable" iff a real, applicable build knob realizes its
repair. Today TWO knobs exist:

    KNOB_FOR_KIND = {
        "clk_regen": "shield",                       # grounded clk-shield strip
        "kickback":  "shield",
        "matched_cc_balance": "symmetric_matched_routing",  # mirror-route the pair
    }

The ``symmetric_matched_routing`` knob makes a differential pair's routing an exact
x-mirror, so the matched pair's two nets carry balanced parasitics -- the real fix
for a ``matched_cc_balance`` imbalance that a routing asymmetry caused. Any other
kind (e.g. one a proposer would address with ``increase_spacing``) still has NO
knob and stays PROPOSAL-ONLY: the loop attaches ``propose_layout_repair``'s output
so a human/router can act on it, but it NEVER claims it ``fixed``. Fabricating a
"fix" for a kind with no knob would be a false repair claim; the loop refuses to.
The loop also refuses to claim ``fixed`` for a knob-addressable obligation that
was ALREADY within budget in the degraded baseline (it was never broken).

Operator->knob attribution honesty
----------------------------------
``propose_layout_repair`` names ``symmetrize_routing`` FIRST for ``clk_regen`` /
``kickback`` (symmetry is its preferred proposal). But the move the loop actually
APPLIES for those kinds is the grounded shield, whose operator is
``insert_gnd_shield`` -- so the realized fix is attributed to ``insert_gnd_shield``,
NOT ``symmetrize_routing``. For ``matched_cc_balance`` the realized move genuinely
IS symmetric routing, so its fix is attributed to ``symmetrize_routing`` (there is
no aggressor to shield for a matched-cap balance). Attribution follows the knob
that was turned, never the proposer's first-named operator.

Repair-axis only
----------------
Like the rest of ``clay.repair``, this loop lives on the repair axis. It reports
which obligations a knob brought within budget and which still need work; it
NEVER certifies a verification PASS. The verifier remains the only authority.

The extraction goes through the real ``MagicPexBackend`` (Magic ext2spice in
Docker), so callers must gate on
``clay.layout.pex.magic_runner.magic_available()``. The loop is bounded by
``max_iters`` and terminates early on all-knob-addressable-pass or on no-progress.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from clay.layout.intent import LayoutIntent, LayoutObligation
from clay.repair._layout_ops import (
    INSERT_GND_SHIELD,
    KIND_CLK_REGEN,
    KIND_KICKBACK,
    KIND_MATCHED,
    SYMMETRIZE_ROUTING,
    propose_layout_repair,
)
from clay.repair.layout_apply import _CELL_NAME

# ─── The HONEST apply-knob registry ──────────────────────────────────────────
#
# Maps an obligation KIND -> the name of the real, parameterized router knob that
# realizes its repair. Two real router knobs exist today:
#
#   * "shield" -- the grounded clk-shield strip
#     (``build_routed_comparator(shield=...)``) cuts an aggressor's unequal
#     coupling, addressing the two aggressor-driven kinds:
#       - clk_regen: clk couples unequally into the regen pair; the shield cuts it.
#       - kickback:  the same grounded strip reduces a dynamic line's unequal
#                    coupling onto the input pair.
#   * "symmetric_matched_routing" -- the matched-pair routing-symmetry knob
#     (``build_routed_comparator(symmetric_matched_routing=...)``). When ON the
#     differential pairs are routed as exact x-mirrors, so each matched pair's two
#     nets carry balanced parasitics; this addresses the matched_cc_balance kind
#     (a routing asymmetry between the two matched nets). It is NOT a shield and
#     NOT an aggressor cut -- the realized operator is symmetrize_routing.
#
# A kind is "knob-addressable" iff it is a key here -- the honesty contract: the
# loop only claims a kind ``fixed`` when a real build knob realizes its repair.
KNOB_FOR_KIND: Dict[str, str] = {
    KIND_CLK_REGEN: "shield",
    KIND_KICKBACK: "shield",
    KIND_MATCHED: "symmetric_matched_routing",
}

# The operator each knob actually realizes. For the shield knob, propose_layout_
# repair names symmetrize_routing FIRST for clk_regen/kickback, but the APPLIED
# move is the grounded shield -> attribute the realized fix to insert_gnd_shield.
# The matched-routing knob genuinely realizes symmetrize_routing (it makes the
# differential pair's routing symmetric), so it is attributed to that operator --
# NOT to insert_gnd_shield (there is no aggressor to shield for matched balance).
_KNOB_OPERATOR = {
    "shield": INSERT_GND_SHIELD,
    "symmetric_matched_routing": SYMMETRIZE_ROUTING,
}

# The value each knob takes to ENABLE its fix. Both are boolean today, and both
# follow the same pattern: the DEGRADED baseline turns the knob OFF (False) and the
# fix turns it ON (True). For "shield" ON draws the grounded strip; for
# "symmetric_matched_routing" ON keeps the differential routing an exact mirror.
_KNOB_ENABLE_VALUE = {"shield": True, "symmetric_matched_routing": True}

# Every build knob the loop can turn, in build_routed_comparator's kwarg names. The
# degraded baseline builds with all of them OFF (the opposite of their enable value).
_ALL_KNOBS = ("shield", "symmetric_matched_routing")

# Imbalance must drop by at least this RELATIVE fraction of the prior imbalance to
# count as "progress". Below it, a rebuild is treated as a no-op and the loop
# stops rather than spinning. RELATIVE, per the project-wide threshold rule.
_MIN_REL_PROGRESS = 0.01


def _knob_addressable(obl: LayoutObligation) -> bool:
    """True iff this obligation's kind has a real router knob in KNOB_FOR_KIND."""
    return obl.kind in KNOB_FOR_KIND


def _failing(obls: List[LayoutObligation]) -> List[LayoutObligation]:
    return [o for o in obls if o.status == "fail"]


def _improved(prev: float, cur: float) -> bool:
    """Did the imbalance drop by at least the relative-progress floor?"""
    if prev <= 0.0:
        return False
    return (prev - cur) / prev >= _MIN_REL_PROGRESS


def _operator_name(op) -> str:
    """A stable, human name for a RepairOperator (its target obligation kind)."""
    return getattr(op, "role_kind", str(op))


def _extract_and_check(
    intent: LayoutIntent,
    *,
    tmp_dir,
    gds_name: str,
    **knobs: object,
) -> List[LayoutObligation]:
    """Build the comparator at the given knob settings, emit GDS, run REAL Magic
    PEX, and check the intent -- the loop's DECLARE -> build -> extract -> CHECK leg.

    A thin generalization of ``layout_apply.extract_and_check`` that threads EVERY
    build knob the loop owns (``shield`` AND ``symmetric_matched_routing``) into
    ``build_routed_comparator``, instead of the single ``shield`` knob the
    single-obligation driver forwards. Any knob the caller omits keeps
    ``build_routed_comparator``'s committed default (so an all-default call stays
    byte-identical to the committed layout). One real extraction per call.
    """
    from clay.layout.comparator_router import (
        build_routed_comparator,
        emit_routed_comparator,
    )
    from clay.layout.intent_obligations import check_layout_intent
    from clay.layout.pex import MagicPexBackend

    tmp_dir = Path(tmp_dir)
    gds = tmp_dir / gds_name
    cell = build_routed_comparator(**knobs)
    emit_routed_comparator(cell, str(gds), cell_name=_CELL_NAME)
    report = MagicPexBackend().extract(str(gds), _CELL_NAME, "gf180mcu")
    return check_layout_intent(intent, report)


def run_layout_repair_loop(
    intent: LayoutIntent,
    *,
    tmp_dir,
    max_iters: int = 4,
) -> Dict[str, object]:
    """Iterate the layout-repair APPLY loop over ALL failing obligations.

    Algorithm (repair-axis only, bounded, honest)::

      0. EXTRACT+CHECK the DEGRADED layout: build with every knob OFF (each knob
         set to the opposite of its enable value -- ``shield=False`` AND
         ``symmetric_matched_routing=False``) and check the intent -> the initial
         obligation set. clk_regen and the desymmetrized matched pair start failing.
      1. While there are knob-addressable failing obligations AND progress is
         still being made AND iterations remain:
           a. For each failing obligation whose kind is in ``KNOB_FOR_KIND``,
              set that knob to its enabling value (``shield=True`` and/or
              ``symmetric_matched_routing=True``). One rebuild can satisfy SEVERAL
              knob-addressable obligations at once (the shield strip helps
              clk_regen and kickback; mirror routing rebalances the matched pair),
              so the distinct knobs are pooled and a SINGLE rebuild is done.
           b. RE-EXTRACT + RE-CHECK with the pooled knobs set.
           c. Record each knob-addressable obligation's before/after imbalance.
           d. Stop early if every knob-addressable obligation now passes, or if
              the worst knob-addressable imbalance did not improve (no-progress).
      2. Classify the FINAL obligation set:
           * ``fixed``         -- obligations a knob brought to pass: kind in
                                  KNOB_FOR_KIND, it STARTED over budget in the
                                  degraded baseline, and it now passes -- attributed
                                  to the knob's operator (shield -> insert_gnd_shield,
                                  symmetric_matched_routing -> symmetrize_routing).
                                  A knob-addressable obligation that was never
                                  broken (already within budget) is NOT claimed
                                  fixed -- it appears in neither bucket;
           * ``proposal_only`` -- failing obligations whose kind has NO knob;
                                  each carries ``propose_layout_repair``'s output
                                  so a router/human can act -- NEVER claimed fixed;
           * ``still_failing`` -- knob-addressable obligations still failing after
                                  the loop exhausted its knobs/iterations.

    Returns::

        {
          "iterations": int,                 # rebuild rounds actually run
          "knobs_applied": ["shield", ...],  # distinct knobs the loop turned
          "fixed": [ {kind, pair, operator, before_imbalance,
                      after_imbalance, rel_tol, knob}, ... ],
          "proposal_only": [ {kind, pair, measured_imbalance, rel_tol,
                              proposal: <propose_layout_repair output>}, ... ],
          "still_failing": [ {kind, pair, measured_imbalance, rel_tol}, ... ],
          "converged": bool,                 # all knob-addressable now pass
          "initial": [LayoutObligation, ...],
          "final":   [LayoutObligation, ...],
        }

    ``converged`` is True iff NO knob-addressable obligation is still failing
    (proposal-only kinds, having no knob, do not block convergence -- the loop is
    honest that it cannot fix them and reports them separately). The loop NEVER
    certifies a verification pass.
    """
    tmp_dir = Path(tmp_dir)

    # ── 0. DEGRADED baseline: every knob off. ────────────────────────────────
    # All knobs OFF = the opposite of their enable value (shield omitted, matched
    # routing desymmetrized) -> clk_regen AND matched_cc_balance start over budget.
    degraded_kwargs = {k: not _KNOB_ENABLE_VALUE[k] for k in _ALL_KNOBS}
    initial = _extract_and_check(
        intent, tmp_dir=tmp_dir, gds_name="layout_loop_initial.gds", **degraded_kwargs
    )
    # Snapshot the degraded imbalance per obligation, keyed by (kind, pair) for
    # before/after deltas. The key MUST include the pair: a cell has several
    # matched_cc_balance obligations (one per diff pair), so keying by kind alone
    # would collapse them and misattribute the degraded baseline of one pair to
    # another. (kind, pair) is unique per obligation.
    before_imbalance: Dict[tuple, float] = {
        (o.kind, o.pair): o.measured_imbalance for o in initial
    }

    current = initial
    applied_knobs: List[str] = []
    knob_state: Dict[str, object] = {}  # knobs currently set ON
    iterations = 0

    # ── 1. CONVERGENCE LOOP over knob-addressable failing obligations. ───────
    while iterations < max_iters:
        failing = _failing(current)
        addressable = [o for o in failing if _knob_addressable(o)]
        if not addressable:
            break  # nothing left that a knob can touch -> done iterating

        # Pool the DISTINCT knobs the failing obligations call for. A single
        # rebuild with the pooled knobs can fix several obligations at once.
        wanted = {
            KNOB_FOR_KIND[o.kind]: _KNOB_ENABLE_VALUE[KNOB_FOR_KIND[o.kind]]
            for o in addressable
        }
        # If every wanted knob is already ON, another rebuild cannot help.
        if all(knob_state.get(k) == v for k, v in wanted.items()):
            break

        # Worst knob-addressable imbalance going in (no-progress comparison).
        worst_before = max(o.measured_imbalance for o in addressable)

        # APPLY: turn on the pooled knobs and rebuild + re-extract + re-check.
        knob_state.update(wanted)
        for k in wanted:
            if k not in applied_knobs:
                applied_knobs.append(k)
        # Every knob: ON if currently set on (in knob_state), else its degraded
        # (OFF) value -- so a knob the loop has not yet turned stays desymmetrized
        # / unshielded rather than silently reverting to its committed default.
        kwargs = {
            k: knob_state.get(k, not _KNOB_ENABLE_VALUE[k]) for k in _ALL_KNOBS
        }
        iterations += 1
        current = _extract_and_check(
            intent,
            tmp_dir=tmp_dir,
            gds_name=f"layout_loop_iter{iterations}.gds",
            **kwargs,
        )

        # No-progress guard: if the worst knob-addressable imbalance did not
        # improve, stop rather than spin (the knob has no further leverage).
        new_addressable = [o for o in _failing(current) if _knob_addressable(o)]
        if not new_addressable:
            break  # all knob-addressable obligations now pass
        worst_after = max(o.measured_imbalance for o in new_addressable)
        if not _improved(worst_before, worst_after):
            break

    final = current

    # ── 2. CLASSIFY the final obligation set. ────────────────────────────────
    fixed: List[Dict[str, object]] = []
    proposal_only: List[Dict[str, object]] = []
    still_failing: List[Dict[str, object]] = []

    for o in final:
        if _knob_addressable(o):
            before = before_imbalance.get((o.kind, o.pair))
            # Did this obligation actually START over budget in the degraded
            # baseline? Only then can a knob have FIXED it. A knob-addressable
            # obligation that was already within budget (e.g. a matched pair the
            # degraded build never desymmetrized) was never broken -> it is neither
            # ``fixed`` nor ``still_failing``; claiming it ``fixed`` would invent a
            # repair that never happened.
            started_failing = before is not None and before > o.rel_tol
            if o.status == "pass":
                if not started_failing:
                    continue  # never broken -> no repair to claim
                # A knob brought a genuinely-failing obligation to pass: attribute
                # it to the knob's operator (shield -> insert_gnd_shield; matched
                # routing -> symmetrize_routing).
                knob = KNOB_FOR_KIND[o.kind]
                op = _KNOB_OPERATOR.get(knob)
                fixed.append(
                    {
                        "kind": o.kind,
                        "pair": o.pair,
                        "operator": _operator_name(op) if op is not None else knob,
                        "knob": knob,
                        "before_imbalance": before,
                        "after_imbalance": o.measured_imbalance,
                        "rel_tol": o.rel_tol,
                    }
                )
            else:
                # Knob-addressable but still failing after the loop ran out.
                still_failing.append(
                    {
                        "kind": o.kind,
                        "pair": o.pair,
                        "measured_imbalance": o.measured_imbalance,
                        "rel_tol": o.rel_tol,
                    }
                )
        else:
            # No knob for this kind. If it fails, it is PROPOSAL-ONLY: attach the
            # proposer's move but NEVER claim it fixed. A passing no-knob
            # obligation needs nothing.
            if o.status == "fail":
                proposal_only.append(
                    {
                        "kind": o.kind,
                        "pair": o.pair,
                        "measured_imbalance": o.measured_imbalance,
                        "rel_tol": o.rel_tol,
                        "proposal": propose_layout_repair(o),
                    }
                )

    # Converged iff NO knob-addressable obligation is still failing. Proposal-only
    # kinds have no knob, so they do not block convergence -- the loop is honest
    # that it cannot fix them and reports them in their own bucket.
    converged = len(still_failing) == 0

    return {
        "iterations": iterations,
        "knobs_applied": applied_knobs,
        "fixed": fixed,
        "proposal_only": proposal_only,
        "still_failing": still_failing,
        "converged": converged,
        "initial": initial,
        "final": final,
    }


__all__ = ["KNOB_FOR_KIND", "run_layout_repair_loop"]
