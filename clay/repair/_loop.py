"""
Repair loop — bounded, strict-descent repair iteration with an explicit
finite retry budget.

Per the theory doc's "Repair completeness theorem shape": termination
requires strict descent on the discretized distance OR a finite retry
budget. This loop enforces BOTH — it stops on the first of:

  * the violation set is satisfied (distance reaches zero),
  * a step fails to make strict progress (the discretized distance did
    not strictly decrease), or
  * the retry budget is exhausted.

The loop never runs a simulator itself. The caller supplies an
``apply_and_reverify`` callable that takes a :class:`RepairProposal` and
returns a fresh list of :class:`InvariantViolation` (the result of
applying the proposed knob move and re-running the oracle). This keeps
the loop unit-testable with a synthetic stub AND usable with the real
verifier later (the main session wires the real callable — see the
package report). The loop also requires a ``propose`` callable so the
proposer (and its registry) is injected rather than hard-wired.

Local safety: every accepted step must pass
:func:`repair_is_locally_safe` against the pre-step violation set, so a
move that reduces the headline distance but regresses a previously-
satisfied obligation is rejected and stops the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence, Union

from clay.invariants_modules._violation import InvariantViolation

from ._distance import repair_is_locally_safe, violation_distance
from ._proposal import RepairProposal, propose_repair
from ._status import RepairStatus

# Type aliases for the injected callables.
ApplyAndReverify = Callable[[RepairProposal], Sequence[InvariantViolation]]
Proposer = Callable[
    [InvariantViolation],
    Union[RepairProposal, RepairStatus],
]

DEFAULT_RETRY_BUDGET: int = 5

# Minimum strict-descent the discretized distance must achieve per step
# for the step to count as progress. A step that reduces distance by less
# than this is treated as "no progress" and stops the loop (prevents an
# asymptotic non-terminating crawl).
DEFAULT_MIN_DESCENT: float = 1e-9


class StopReason(Enum):
    """Why the repair loop halted."""

    SATISFIED = "satisfied"            # distance reached zero
    NO_PROGRESS = "no_progress"        # a step failed strict descent
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROPOSAL = "no_proposal"        # no actionable proposal this round
    UNSAFE = "unsafe"                  # a step regressed a preserved obligation


@dataclass
class RepairStep:
    """One iteration record of the loop (for traceability)."""

    iteration: int
    proposal: RepairProposal
    distance_before: float
    distance_after: float
    accepted: bool


@dataclass
class RepairResult:
    """Outcome of a :class:`RepairLoop.run`.

    Fields:
      * `status`        — :attr:`RepairStatus.REPAIRED` only when the
                          loop drove the distance to zero; otherwise the
                          most-informative non-success status reached
                          (REPAIRABLE if it descended but did not finish,
                          else the proposer's blocking status).
      * `stop_reason`   — :class:`StopReason`.
      * `iterations`    — number of apply+reverify rounds executed.
      * `budget`        — the configured retry budget.
      * `initial_distance` / `final_distance`.
      * `steps`         — per-iteration :class:`RepairStep` records.
      * `final_violations` — the violation set after the last accepted
                          step (or the input set if none was accepted).
    """

    status: RepairStatus
    stop_reason: StopReason
    iterations: int
    budget: int
    initial_distance: float
    final_distance: float
    steps: list[RepairStep] = field(default_factory=list)
    final_violations: list[InvariantViolation] = field(default_factory=list)

    @property
    def made_progress(self) -> bool:
        return self.final_distance < self.initial_distance


@dataclass
class RepairLoop:
    """Bounded strict-descent repair driver.

    Args:
      * `apply_and_reverify` — caller-provided callable that applies a
                          proposal and returns the fresh violation set.
                          Unit tests pass a synthetic stub; the main
                          session wires the real verifier here.
      * `propose`        — proposer callable (defaults to
                          :func:`propose_repair` with the default
                          registry). Injectable for tests / custom
                          registries.
      * `budget`         — finite retry budget (default 5). Exposed on the
                          result so callers can see the bound.
      * `min_descent`    — minimum strict distance decrease per step to
                          count as progress.
    """

    apply_and_reverify: ApplyAndReverify
    propose: Proposer = propose_repair
    budget: int = DEFAULT_RETRY_BUDGET
    min_descent: float = DEFAULT_MIN_DESCENT

    def _select_target(
        self, violations: Sequence[InvariantViolation],
    ) -> Optional[InvariantViolation]:
        """Pick the violation with the largest per-set distance to repair.

        Repairing the worst-offending obligation first is a greedy
        strict-descent heuristic; ties break on input order.
        """
        worst: Optional[InvariantViolation] = None
        worst_d = 0.0
        for v in violations:
            d = violation_distance([v])
            if d > worst_d:
                worst_d = d
                worst = v
        return worst

    def run(
        self, violations: Sequence[InvariantViolation],
        *, weights=None,
    ) -> RepairResult:
        """Drive the repair loop to a stop condition.

        Returns a :class:`RepairResult`. The loop:
          1. Computes the current distance; if zero, stops SATISFIED.
          2. Selects the worst violation and asks the proposer for a move.
             A non-REPAIRABLE proposer result stops the loop NO_PROPOSAL
             (carrying the proposer's blocking status).
          3. Applies the move via ``apply_and_reverify`` and recomputes
             distance.
          4. Accepts the step only if it is locally safe AND strictly
             descends by at least ``min_descent``; else stops
             NO_PROGRESS / UNSAFE.
          5. Repeats until satisfied, no progress, or budget exhausted.
        """
        current = list(violations)
        initial_dist = violation_distance(current, weights)
        steps: list[RepairStep] = []

        if initial_dist <= 0.0:
            # Already at distance zero: no active defect to repair. The
            # repair goal trivially holds (iterations==0 signals no move
            # was applied). Still a repair-axis fact, never a PASS.
            return RepairResult(
                status=RepairStatus.REPAIRED,
                stop_reason=StopReason.SATISFIED,
                iterations=0,
                budget=self.budget,
                initial_distance=initial_dist,
                final_distance=initial_dist,
                steps=steps,
                final_violations=current,
            )

        blocking_status = RepairStatus.REPAIRABLE
        for it in range(1, self.budget + 1):
            dist_before = violation_distance(current, weights)
            target = self._select_target(current)
            if target is None:
                # No violation carries positive distance -> satisfied.
                return self._result(
                    RepairStatus.REPAIRED, StopReason.SATISFIED,
                    it - 1, initial_dist, dist_before, steps, current,
                )

            proposal = self.propose(target)
            if not isinstance(proposal, RepairProposal):
                # Proposer could not act: carry its blocking status out.
                blocking_status = proposal
                return self._result(
                    blocking_status, StopReason.NO_PROPOSAL,
                    it - 1, initial_dist, dist_before, steps, current,
                )

            after = list(self.apply_and_reverify(proposal))
            dist_after = violation_distance(after, weights)

            safe = repair_is_locally_safe(
                dist_before, dist_after,
                before_violations=current, after_violations=after,
            )
            strict = (dist_before - dist_after) >= self.min_descent

            if not safe:
                steps.append(RepairStep(it, proposal, dist_before, dist_after, False))
                return self._result(
                    blocking_status, StopReason.UNSAFE,
                    it, initial_dist, dist_before, steps, current,
                )
            if not strict:
                steps.append(RepairStep(it, proposal, dist_before, dist_after, False))
                return self._result(
                    blocking_status, StopReason.NO_PROGRESS,
                    it, initial_dist, dist_before, steps, current,
                )

            # Accept the step.
            steps.append(RepairStep(it, proposal, dist_before, dist_after, True))
            current = after

            if dist_after <= 0.0:
                # Distance-to-satisfaction reached zero: every active,
                # observable obligation is within tolerance. This is the
                # repair-progress goal (a measure_missing / pending entry
                # carries a nonzero penalty, so a zero distance cannot
                # hide an unobservable defect). REPAIRED here is a
                # repair-axis fact only — it does NOT certify a
                # verification PASS (see RepairStatus).
                return self._result(
                    RepairStatus.REPAIRED,
                    StopReason.SATISFIED,
                    it, initial_dist, dist_after, steps, current,
                )

        # Budget exhausted with progress made but not satisfied.
        final_dist = violation_distance(current, weights)
        return self._result(
            blocking_status, StopReason.BUDGET_EXHAUSTED,
            self.budget, initial_dist, final_dist, steps, current,
        )

    def _result(
        self, status, stop_reason, iterations, initial_dist, final_dist,
        steps, current,
    ) -> RepairResult:
        return RepairResult(
            status=status,
            stop_reason=stop_reason,
            iterations=iterations,
            budget=self.budget,
            initial_distance=initial_dist,
            final_distance=final_dist,
            steps=steps,
            final_violations=list(current),
        )
