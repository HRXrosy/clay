"""
Deterministic stub optimizers — exercise the ``OptimizerInterface``
contract without paying for a real BO library.

Three stubs ship in v0.0.67 Phase 3 first slice:

  * :class:`RandomSearchStub` — uniform random sample within each
    knob's bounds. Reproducible given a seed; honors ``log_scale``
    for knobs that span orders of magnitude.
  * :class:`GridSearchStub` — regular grid enumeration; cycles back to
    grid origin after exhausting the cartesian product so the harness
    can run forever without an early exit.
  * :class:`GradientFreeStub` — returns the same fixed sizing every
    call. Sanity-check baseline for the loop plumbing.

None of these are intended to find good designs. They exist so the
harness contract has at least three independent implementations to
test against and so the CLI has a smoke-test path that doesn't depend
on the tSS-BO library (deferred to v0.0.68+).

Each stub:

  * accepts a list of ``SizingBounds`` defining the knob list and
    per-knob feasible range (random / grid),
  * accepts a seedable random source (random / grid scramble),
  * never raises on degenerate input — empty bounds list yields a
    no-op optimizer that returns an empty dict on every ``propose``
    (the harness then records a sizing-empty entry and moves on).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Mapping, Sequence

from .harness import (
    OptimizerHistory,
    OptimizerHistoryEntry,
    Sizing,
    SizingBounds,
)


# ─── Random search ────────────────────────────────────────────────


@dataclass
class RandomSearchStub:
    """Uniform random sample within bounds, seeded for reproducibility.

    Construction:
      * ``bounds`` — list of ``SizingBounds`` (one per knob). Empty
        list → ``propose`` returns ``{}`` on every call (the harness
        records sizing-empty entries; useful for testing the no-knob
        guard rail).
      * ``seed`` — passed to ``random.Random``. Two instances with the
        same seed produce identical proposal sequences for the same
        sequence of ``propose`` calls.

    The stub is stateless beyond its PRNG: ``propose`` ignores the
    history entirely; ``update`` is a no-op. That makes it the
    simplest possible non-trivial ``OptimizerInterface`` and a useful
    contract reference for new optimizers.
    """

    bounds: Sequence[SizingBounds] = field(default_factory=tuple)
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def propose(self, history: OptimizerHistory) -> Sizing:
        # history is intentionally unused — random search is stateless.
        _ = history
        out: dict[str, float] = {}
        for b in self.bounds:
            if b.log_scale:
                lo = math.log(b.low)
                hi = math.log(b.high)
                val = math.exp(self._rng.uniform(lo, hi))
            else:
                val = self._rng.uniform(b.low, b.high)
            out[b.knob] = val
        return out

    def update(
        self,
        history: OptimizerHistory,
        sizing: Sizing,
        result: OptimizerHistoryEntry,
    ) -> None:
        # Stateless: ignore feedback. Real BO would refit a surrogate
        # here from history.entries.
        _ = (history, sizing, result)


# ─── Grid search ──────────────────────────────────────────────────


@dataclass
class GridSearchStub:
    """Enumerate a regular grid over the bounds; cycle on overflow.

    Each knob is split into ``points_per_axis`` equally-spaced samples
    (linear or logarithmic per the bound). The optimizer cycles
    through the cartesian product in row-major order; after the last
    combination it wraps back to the first so ``run(max_iters > N)``
    is well-defined (the harness can ask for more iters than the grid
    has — useful for testing exhaustion).

    Construction:
      * ``bounds`` — knob list. Empty → ``propose`` returns ``{}``.
      * ``points_per_axis`` — same for every axis; must be ≥ 2 (a
        single-point axis is a degenerate special case; reject so
        tests catch the configuration error). Defaults to 3.
      * ``seed`` — unused for ordering (the grid is deterministic).
        Kept on the constructor so all three stubs share a uniform
        seedable surface for the CLI.
    """

    bounds: Sequence[SizingBounds] = field(default_factory=tuple)
    points_per_axis: int = 3
    seed: int = 0
    _index: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.bounds and self.points_per_axis < 2:
            raise ValueError(
                f"GridSearchStub.points_per_axis must be >= 2, got "
                f"{self.points_per_axis!r}"
            )

    @property
    def grid_size(self) -> int:
        """Total number of cartesian product points; 0 if no bounds."""
        if not self.bounds:
            return 0
        return self.points_per_axis ** len(self.bounds)

    def _index_to_sizing(self, idx: int) -> dict[str, float]:
        out: dict[str, float] = {}
        if not self.bounds:
            return out
        n = self.points_per_axis
        # Convert flat idx → per-axis indices (row-major / little-endian
        # over the bound list).
        remaining = idx
        per_axis_idx: List[int] = []
        for _ in self.bounds:
            per_axis_idx.append(remaining % n)
            remaining //= n
        # Now per_axis_idx[i] is the grid index along bounds[i].
        for b, axis_i in zip(self.bounds, per_axis_idx):
            # Fraction along the axis in [0, 1].
            t = axis_i / (n - 1)
            if b.log_scale:
                lo = math.log(b.low)
                hi = math.log(b.high)
                out[b.knob] = math.exp(lo + t * (hi - lo))
            else:
                out[b.knob] = b.low + t * (b.high - b.low)
        return out

    def propose(self, history: OptimizerHistory) -> Sizing:
        _ = history
        if not self.bounds:
            return {}
        sizing = self._index_to_sizing(self._index)
        # Increment & wrap — cycles if the harness asks for more
        # iters than grid_size.
        self._index = (self._index + 1) % self.grid_size
        return sizing

    def update(
        self,
        history: OptimizerHistory,
        sizing: Sizing,
        result: OptimizerHistoryEntry,
    ) -> None:
        _ = (history, sizing, result)


# ─── Trivial no-op optimizer ─────────────────────────────────────


@dataclass
class GradientFreeStub:
    """Returns the same fixed sizing every iteration. Sanity-check.

    Useful for:
      * Plumbing tests — confirm the harness runs the same sizing N
        times and produces N entries with identical knob dicts.
      * Determinism baseline — every call returns the canonical
        sizing, no PRNG state, no grid index.

    Construction:
      * ``fixed_sizing`` — the dict to return on every ``propose``.
        Empty → returns ``{}`` (degenerate but legal).
      * ``seed`` — unused; kept on the constructor for uniform CLI
        wiring across the three stubs.
    """

    fixed_sizing: Mapping[str, float] = field(default_factory=dict)
    seed: int = 0

    def propose(self, history: OptimizerHistory) -> Sizing:
        _ = history
        return dict(self.fixed_sizing)

    def update(
        self,
        history: OptimizerHistory,
        sizing: Sizing,
        result: OptimizerHistoryEntry,
    ) -> None:
        _ = (history, sizing, result)


# ─── Convenience factory ──────────────────────────────────────────


def build_stub(
    kind: str,
    *,
    bounds: Sequence[SizingBounds] = (),
    fixed_sizing: Mapping[str, float] | None = None,
    seed: int = 0,
    points_per_axis: int = 3,
):
    """Build a stub by string tag — used by the CLI's --optimizer flag.

    Recognized tags: ``'stub-random'``, ``'stub-grid'``, ``'stub-noop'``.
    Unknown tags raise ValueError with the recognized set. The bounds
    / fixed_sizing kwargs are forwarded only to the relevant stub —
    extras are silently ignored so the CLI doesn't have to branch.
    """
    if kind == "stub-random":
        return RandomSearchStub(bounds=tuple(bounds), seed=seed)
    if kind == "stub-grid":
        return GridSearchStub(
            bounds=tuple(bounds),
            points_per_axis=points_per_axis,
            seed=seed,
        )
    if kind == "stub-noop":
        return GradientFreeStub(
            fixed_sizing=dict(fixed_sizing or {}),
            seed=seed,
        )
    raise ValueError(
        f"unknown stub optimizer kind {kind!r}; "
        f"choose one of 'stub-random', 'stub-grid', 'stub-noop'"
    )
