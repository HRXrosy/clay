"""
Per-run budget cap for the real tSS-BO global explorer (v0.0.68 Phase 3
second slice).

Mirrors the v0.0.59 ``tools.llm_authoring.budget`` design: a single
:class:`BoBudgetTracker` instance covers one optimizer run, enforces
hard caps before the heavy work starts, and exposes a single-line
summary so a stray "I'll just bump the cap one nudge" never goes
unnoticed.

Why a separate tracker (and not the LLM one)
--------------------------------------------

Two reasons:

  1. **Different cost surface.** The LLM tracker prices in USD per
     million tokens. A BO run's costs are wall-clock seconds and
     ngspice invocations — no tokens, no vendor pricing table. The
     unit conversion is meaningless to share with the LLM tracker.
  2. **Different default policy.** The LLM tracker defaults to a
     **per-run** USD ceiling because "one task, one model" is the
     unit of work. A BO run's unit of work is one optimizer-loop
     ``run(max_iters=N)`` call; the caps are per-loop, not per-task,
     and the natural ceilings are iteration count and wall-clock.

The two trackers share the same conservative-by-default ethos: every
default is low enough that a user who passes ``--optimizer tssbo``
without explicit caps cannot accidentally burn an hour of CPU.

The four caps
-------------

* ``max_bo_iters``        — hard ceiling on optimizer iterations
                            for one ``run(max_iters=N)``. The harness
                            already takes ``max_iters`` as an arg; the
                            tracker enforces that the **caller's**
                            requested ``max_iters`` does not exceed
                            this budget cap.
* ``max_wall_clock_sec``  — wall-clock seconds across the whole run.
                            The loop calls :meth:`check_wall_clock`
                            before each iteration; the first call
                            past the deadline refuses with
                            :class:`BudgetExceededError`.
* ``max_ngspice_calls``   — total ngspice invocations across the
                            whole run. Each successful simulator
                            call records one against the budget.
                            Refuses to dispatch a sim that would
                            tip past the cap.
* ``max_bo_proposals``    — total ``propose()`` calls. Bounded just
                            in case an optimizer reproposes the
                            same point on every iteration (which
                            costs CPU but no ngspice traffic).

All caps are constructor arguments — no environment-driven overrides.
A stray env var should not be able to widen a cap silently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ─── Exception ───────────────────────────────────────────────────────


class BudgetExceededError(RuntimeError):
    """Raised when a tSS-BO call would push the tracker over a cap.

    Subclass of :class:`RuntimeError` so callers that want to catch
    "any benign budget refusal" can do so without also swallowing a
    transport / parser failure.
    """


# ─── Tracker ─────────────────────────────────────────────────────────


@dataclass
class BoBudgetTracker:
    """Per-run accumulator with hard caps on iters / wall / sim calls.

    Lifecycle:

      1. Construct with explicit caps (or accept the conservative
         defaults).
      2. Call :meth:`start_clock` at the start of the run to pin the
         wall-clock origin.
      3. Before authorising work that costs an iter / a sim, call
         :meth:`check_iter` / :meth:`check_ngspice_call`. Both raise
         :class:`BudgetExceededError` if the budget would be tripped.
      4. After the work succeeds, call :meth:`record_iter` /
         :meth:`record_ngspice_call`.
      5. :meth:`summary` returns a one-line human-readable digest.

    Defaults: 100 iters, 5 minutes wall, 100 ngspice calls, 500
    proposals. A v0.0.68 user who hits any of these by accident has
    almost certainly mis-configured the loop and should see the
    refusal in stderr, not a silent CPU runaway.
    """

    # Per-run ceilings.
    max_bo_iters: int = 100
    max_wall_clock_sec: float = 300.0  # 5 minutes
    max_ngspice_calls: int = 100
    max_bo_proposals: int = 500

    # Live counters.
    total_iters: int = 0
    total_ngspice_calls: int = 0
    total_proposals: int = 0

    # Wall-clock origin, set by :meth:`start_clock`. ``None`` means
    # ``check_wall_clock`` is a no-op (the caller hasn't started the
    # clock; useful for fixed-budget tests that don't care about
    # elapsed wall time).
    _t0: Optional[float] = field(default=None, repr=False)

    # Optional label for stderr summary lines.
    label: str = "(unlabeled)"

    # ── Lifecycle ───────────────────────────────────────────────────

    def start_clock(self) -> None:
        """Pin the wall-clock origin to monotonic time.

        Idempotent: calling twice does NOT reset the clock — the
        first call sticks. This lets callers safely
        ``start_clock`` defensively at the top of a chain that may
        already have been started elsewhere.
        """
        if self._t0 is None:
            self._t0 = time.monotonic()

    def reset_clock(self) -> None:
        """Force the wall-clock origin back to the next
        :meth:`start_clock`. Used by tests that want a clean slate.
        """
        self._t0 = None

    # ── Pre-flight checks ────────────────────────────────────────────

    def check_max_iters(self, requested_max_iters: int) -> None:
        """Refuse if the loop's requested ``max_iters`` exceeds the
        per-run iter cap. Call once at the top of the loop —
        cheaper than tripping the per-iter check on iteration N.
        """
        if requested_max_iters > self.max_bo_iters:
            raise BudgetExceededError(
                f"BO max_iters cap exceeded: requested "
                f"{requested_max_iters} > cap {self.max_bo_iters}. "
                f"Raise max_bo_iters or reduce the loop's max_iters."
            )

    def check_iter(self) -> None:
        """Refuse if recording one more iter would tip past the cap."""
        if self.total_iters + 1 > self.max_bo_iters:
            raise BudgetExceededError(
                f"BO iter cap exceeded: {self.total_iters} + 1 > "
                f"cap {self.max_bo_iters}. Raise max_bo_iters."
            )

    def check_ngspice_call(self) -> None:
        """Refuse if recording one more sim would tip past the cap."""
        if self.total_ngspice_calls + 1 > self.max_ngspice_calls:
            raise BudgetExceededError(
                f"BO ngspice call cap exceeded: "
                f"{self.total_ngspice_calls} + 1 > cap "
                f"{self.max_ngspice_calls}. Raise max_ngspice_calls."
            )

    def check_proposal(self) -> None:
        """Refuse if recording one more propose() would tip past the
        cap. The propose cap is the soft outer limit — a runaway
        optimizer that re-proposes on every check_budget catches
        before the iter cap fires."""
        if self.total_proposals + 1 > self.max_bo_proposals:
            raise BudgetExceededError(
                f"BO proposal cap exceeded: {self.total_proposals} + 1 "
                f"> cap {self.max_bo_proposals}. Raise max_bo_proposals."
            )

    def check_wall_clock(self) -> None:
        """Refuse if the elapsed wall-clock has crossed the per-run
        cap. No-op if :meth:`start_clock` was never called."""
        if self._t0 is None:
            return
        elapsed = time.monotonic() - self._t0
        if elapsed > self.max_wall_clock_sec:
            raise BudgetExceededError(
                f"BO wall-clock cap exceeded: elapsed "
                f"{elapsed:.1f}s > cap "
                f"{self.max_wall_clock_sec:.1f}s. Raise "
                f"max_wall_clock_sec."
            )

    # ── Recording (called after work succeeds) ──────────────────────

    def record_iter(self) -> None:
        self.total_iters += 1

    def record_ngspice_call(self) -> None:
        self.total_ngspice_calls += 1

    def record_proposal(self) -> None:
        self.total_proposals += 1

    # ── Summary ──────────────────────────────────────────────────────

    @property
    def elapsed_sec(self) -> float:
        """Wall-clock seconds since :meth:`start_clock`. Returns 0.0
        when the clock hasn't been started."""
        if self._t0 is None:
            return 0.0
        return time.monotonic() - self._t0

    def summary(self) -> str:
        """One-line human summary of the tracker state."""
        return (
            f"[{self.label}] iters={self.total_iters}/"
            f"{self.max_bo_iters} ngspice={self.total_ngspice_calls}/"
            f"{self.max_ngspice_calls} proposals={self.total_proposals}/"
            f"{self.max_bo_proposals} wall={self.elapsed_sec:.1f}s/"
            f"{self.max_wall_clock_sec:.1f}s"
        )
