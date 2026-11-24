"""
Adapter shim bridging Clay's :class:`OptimizerInterface` to the real
tSS-BO library (Truncated Subspace Sampling Bayesian Optimization, from
DATE 2024 / ``github.com/PascalInn/tSS-BO``).

v0.0.68 Phase 3 second slice. Mirrors the v0.0.59 LLM real-API safety
pattern: real tSS-BO is **opt-in only**, gated behind four independent
checks. Stubs from v0.0.67 stay the default — calling
:class:`TSsBoAdapter` without the opt-in env set raises a clean
``RuntimeError`` that callers catch and fall back to a stub.

The 4-layer opt-in
------------------

Every real-BO call must clear all four:

  1. **Env var.** ``CLAY_USE_TSSBO=1`` (truthy) in the environment.
     Centralised in :data:`OPT_IN_ENV` for a single point of audit.
  2. **CLI flag.** ``--optimizer tssbo`` on the command line — the
     ``run.py`` dispatch refuses to route to this adapter without it,
     even if the env var is set.
  3. **Package present.** The tSS-BO Python package must import. The
     adapter probes ``import tssbo`` (with a ``CLAY_TSSBO_MODULE``
     override for the canonical module name once the upstream
     library nails it down). Missing import → ``RuntimeError``.
  4. **Budget.** A :class:`BoBudgetTracker` instance covers the run.
     Caps on iters / wall-clock / ngspice calls. Tripping any cap
     raises :class:`BudgetExceededError` BEFORE the heavy work fires.

If any layer fails, the adapter raises a ``RuntimeError`` (or
``BudgetExceededError`` for the budget layer). The CLI catches both
and exits 2. Callers that want a graceful stub fallback wrap the
constructor in ``try / except RuntimeError`` and instantiate a stub
on the except branch.

Mocking surface for tests
-------------------------

The actual tSS-BO library is heavy (torch, scipy, gpytorch). Pulling
it in by default would balloon CI dependencies and break the
strangler-fig discipline. The adapter therefore goes through a thin
``_TSsBoBackend`` abstraction:

  * Real backend: created by :func:`_create_real_backend`, which
    imports the configured tSS-BO module and constructs the BO
    surrogate. Untested in CI by default (no real lib installed).
  * Mock backend: :class:`_MockTSsBoBackend` ships canned proposals
    that depend on the sizing bounds; tests inject it via the
    ``backend=`` constructor argument so the full adapter contract
    can be exercised without touching the real library.

Tests cover every gate (env, flag, package, budget) by either
mocking the env / ``sys.modules`` entry or passing a pre-configured
backend.

What this slice does NOT do
---------------------------

* No real tSS-BO surrogate refit. The adapter wires the API; the
  refit budget tuning is left to v0.0.69+ once we have empirical
  evidence on real Clay cells.
* No PVT-aware proposals — single-corner BO only. Multi-corner
  weighting comes after PVT-aware reward shaping lands.
* No Adjoint Network handoff — that's v0.0.69+ when AST-level
  small-signal extraction is ready.
"""

from __future__ import annotations

import importlib
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence

from .budget import BoBudgetTracker, BudgetExceededError
from .harness import (
    OptimizerHistory,
    OptimizerHistoryEntry,
    Sizing,
    SizingBounds,
)


# ─── Opt-in env var (Layer 1) ─────────────────────────────────────────


OPT_IN_ENV: str = "CLAY_USE_TSSBO"
"""Env var that authorises real tSS-BO dispatch. Truthy values:
``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive). Any other
value — including unset — routes through the opt-in error path."""


MODULE_NAME_ENV: str = "CLAY_TSSBO_MODULE"
"""Optional override for the importable module name. The upstream
project (`github.com/PascalInn/tSS-BO`) has not settled on a single
canonical PyPI name; if a user installs the lib under a non-standard
name they can pin it here. Defaults to ``tssbo``."""


_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})


def _is_opted_in() -> bool:
    """Return True iff the opt-in env var is set to a truthy value."""
    return os.environ.get(OPT_IN_ENV, "").strip().lower() in _TRUTHY


def _resolve_module_name() -> str:
    """The Python module name to ``importlib.import_module``.

    Defaults to ``tssbo``. Users with a non-canonical install can set
    ``CLAY_TSSBO_MODULE=their_local_name`` to override.
    """
    return os.environ.get(MODULE_NAME_ENV, "").strip() or "tssbo"


# ─── Backend abstraction ─────────────────────────────────────────────


class _TSsBoBackend:
    """Thin protocol the adapter speaks to the underlying BO library.

    Two methods:

      * ``suggest(history)`` — return the next sizing dict to try.
        The adapter forwards Clay's ``OptimizerHistory`` (already in
        the format the harness records), so backends that want only
        the last feasible reward can scan ``history.feasible_entries()``
        without re-deriving anything.

      * ``observe(sizing, reward, feasible)`` — record one
        evaluation. Reward is the harness's signed slack scalar (NaN
        on infeasible); ``feasible`` is the harness's coarse mask.

    Subclasses implement both methods. The mock implementation below
    is the contract reference; the real implementation in
    :func:`_create_real_backend` adapts the upstream tSS-BO API to
    the same shape.
    """

    def suggest(self, history: OptimizerHistory) -> dict[str, float]:
        raise NotImplementedError("subclass must implement suggest()")

    def observe(
        self,
        sizing: Mapping[str, float],
        reward: float,
        feasible: bool,
    ) -> None:
        raise NotImplementedError("subclass must implement observe()")


# ─── Mock backend (used by tests) ─────────────────────────────────────


@dataclass
class _MockTSsBoBackend(_TSsBoBackend):
    """Canned BO backend for tests. Returns a deterministic sequence
    of proposals derived from the sizing bounds.

    Two flavours of test coverage:

      * **Bound-respecting**: every proposal stays within
        [low, high] for each knob. The default sequence is the
        midpoint of each bound (matches GridSearchStub's behaviour
        when given a single-point axis).

      * **History-aware**: after each :meth:`observe`, the backend
        biases the next :meth:`suggest` toward the better of the
        last two rewards. This lets tests assert that the adapter
        actually forwards observations (a static-midpoint mock
        would pass the round-trip test trivially even if the
        adapter dropped every observation on the floor).
    """

    bounds: Sequence[SizingBounds] = field(default_factory=tuple)

    # The "trajectory" the mock walks down. Index 0 = midpoint;
    # each :meth:`observe` advances one step toward the more rewarded
    # half of each bound.
    _step: int = field(init=False, default=0)
    _last_reward: float = field(init=False, default=float("nan"))
    _last_sizing: dict[str, float] = field(init=False, default_factory=dict)
    _bias_high: dict[str, bool] = field(init=False, default_factory=dict)

    # Observation counter — public so tests can assert observe() fired
    # the expected number of times.
    observed_count: int = field(init=False, default=0)
    suggested_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # Initial bias is "toward midpoint" — first :meth:`observe`
        # decides whether each knob biases high or low.
        for b in self.bounds:
            self._bias_high[b.knob] = True

    def suggest(self, history: OptimizerHistory) -> dict[str, float]:
        self.suggested_count += 1
        _ = history
        out: dict[str, float] = {}
        # Walk each knob from its midpoint toward the biased end. The
        # step size shrinks logarithmically (1/2, 1/4, 1/8, ...) so
        # the trajectory converges. This isn't a real BO surrogate —
        # it's just enough state to assert the adapter forwards
        # observations correctly.
        shrink = 0.5 ** max(self._step, 0)
        for b in self.bounds:
            mid = 0.5 * (b.low + b.high)
            half_span = 0.5 * (b.high - b.low)
            target = b.high if self._bias_high[b.knob] else b.low
            # Move (1 - shrink) of the way from midpoint to the
            # biased endpoint; clamp to bounds.
            val = mid + (target - mid) * (1.0 - shrink)
            val = max(b.low, min(b.high, val))
            out[b.knob] = val
        return out

    def observe(
        self,
        sizing: Mapping[str, float],
        reward: float,
        feasible: bool,
    ) -> None:
        self.observed_count += 1
        _ = feasible
        # If this reward is better than the previous one, lock in the
        # bias direction; if it's worse, flip the bias.
        if math.isnan(self._last_reward):
            self._last_reward = reward
            self._last_sizing = dict(sizing)
            self._step += 1
            return
        if not math.isnan(reward) and reward > self._last_reward:
            # Direction along each knob was good: maintain bias.
            pass
        else:
            # Flip biases — toggling pulls the trajectory away from a
            # bad region.
            for k in self._bias_high:
                self._bias_high[k] = not self._bias_high[k]
        self._last_reward = reward
        self._last_sizing = dict(sizing)
        self._step += 1


# ─── Real backend factory ────────────────────────────────────────────


def _create_real_backend(
    module_name: str,
    bounds: Sequence[SizingBounds],
    seed: int,
) -> _TSsBoBackend:
    """Import the real tSS-BO module and adapt it to the
    :class:`_TSsBoBackend` protocol.

    The upstream library (`github.com/PascalInn/tSS-BO`) has not
    pinned its public API on PyPI yet. From the DATE 2024 paper
    summary, the expected surface is:

      * Construct a ``TSsBoOptimizer(bounds=..., seed=...)`` where
        ``bounds`` is a list of ``(low, high)`` pairs in knob order.
      * Call ``opt.suggest()`` to get the next sample (returns a
        list/ndarray in knob order).
      * Call ``opt.observe(x, y)`` to record one evaluation with
        the reward ``y`` (higher = better; the lib's surrogate
        flips signs as needed for an internal minimisation
        convention).

    The wrapper below adapts the Clay dict-of-knob-name format to
    the list-of-floats format the upstream expects, and back.

    If the upstream API turns out to differ (it will: the lib is
    research-grade), the adapter has a single point of change here
    rather than threading the API across the harness.
    """
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"real tSS-BO not enabled - set {OPT_IN_ENV}=1 and install "
            f"package (failed to import {module_name!r}: {exc!r}). "
            f"Override the module name via ${MODULE_NAME_ENV} if your "
            f"install uses a different importable name."
        ) from exc

    # The upstream's constructor is expected to live at
    # ``<module>.TSsBoOptimizer`` (or ``<module>.Optimizer`` as a
    # fallback). If neither is found, surface a clear error rather
    # than silently picking up a wrong name.
    ctor = (
        getattr(mod, "TSsBoOptimizer", None)
        or getattr(mod, "Optimizer", None)
    )
    if ctor is None:
        raise RuntimeError(
            f"real tSS-BO module {module_name!r} imported but neither "
            f"`TSsBoOptimizer` nor `Optimizer` class was found. The "
            f"upstream API may have changed; pin a specific version "
            f"or override the adapter."
        )

    knob_order = [b.knob for b in bounds]
    bound_pairs = [(b.low, b.high) for b in bounds]

    try:
        opt_inst = ctor(bounds=bound_pairs, seed=seed)
    except TypeError as exc:
        # Fall back to a positional construction signature in case
        # upstream uses ``ctor(bounds)`` without keyword names.
        try:
            opt_inst = ctor(bound_pairs)
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(
                f"failed to construct tSS-BO optimizer instance: "
                f"{type(ctor).__name__}({exc!r}); fallback "
                f"positional construction also failed: {exc2!r}"
            ) from exc2

    class _RealBackend(_TSsBoBackend):
        def suggest(self, history: OptimizerHistory) -> dict[str, float]:
            _ = history
            # Upstream's ``suggest()`` returns a list in knob order.
            x = opt_inst.suggest()
            return {k: float(v) for k, v in zip(knob_order, x)}

        def observe(
            self,
            sizing: Mapping[str, float],
            reward: float,
            feasible: bool,
        ) -> None:
            _ = feasible
            # Re-pack to the upstream's list order. If the reward is
            # NaN (infeasible iteration), forward a large penalty so
            # the surrogate avoids the region without crashing on
            # NaN-handling.
            x = [float(sizing[k]) for k in knob_order]
            y = (
                reward if math.isfinite(reward) else -1.0e6
            )
            opt_inst.observe(x, y)

    return _RealBackend()


# ─── Public adapter ──────────────────────────────────────────────────


class TSsBoAdapter:
    """Real tSS-BO optimizer wired to Clay's :class:`OptimizerInterface`.

    Construction flow (the 4-layer opt-in):

      1. **Layer 1 — env var.** ``CLAY_USE_TSSBO=1`` must be set in
         the environment. Otherwise raises ``RuntimeError`` with the
         canonical message ``"real tSS-BO not enabled - set
         CLAY_USE_TSSBO=1 and install package"`` (callers catch and
         fall back to a stub).
      2. **Layer 2 — CLI flag.** The CLI dispatch in
         :mod:`tools.clay_optimize.run` is the second layer; this
         adapter trusts the CLI to have already verified the flag.
         Direct programmatic use should pass ``enabled_by_cli=True``
         to explicitly acknowledge the flag layer (defaults to False
         so a stray import doesn't accidentally clear it).
      3. **Layer 3 — package.** :func:`_create_real_backend` is
         called; an ImportError is wrapped into the same canonical
         ``RuntimeError`` message so the failure mode is uniform.
      4. **Layer 4 — budget.** A :class:`BoBudgetTracker` instance
         (passed via ``budget=`` or constructed with defaults) is
         consulted before every iteration's work.

    Construction args:
      * ``bounds`` — list of :class:`SizingBounds` defining the
        knob list and per-knob feasible range.
      * ``seed`` — forwarded to the upstream tSS-BO constructor.
      * ``budget`` — optional :class:`BoBudgetTracker`. If omitted,
        a tracker with default caps (100 iters / 5 min / 100 sims /
        500 proposals) is constructed.
      * ``enabled_by_cli`` — must be True for the CLI to dispatch.
        Defaults to False so direct programmatic use is also
        flag-gated; the CLI sets it after verifying ``--optimizer
        tssbo``.
      * ``backend`` — optional pre-built backend. Tests inject the
        :class:`_MockTSsBoBackend` here to avoid importing the real
        library; production code leaves it None.

    Public surface mirrors :class:`OptimizerInterface` from
    :mod:`.harness`:

      * :meth:`propose(history) -> Sizing`
      * :meth:`update(history, sizing, result) -> None`

    Both methods consult :attr:`budget` and refuse if a cap is
    tripped — see :class:`BoBudgetTracker` for the four caps.
    """

    # Canonical error string. Tests assert on a substring of this
    # message; keep stable.
    OPT_IN_ERROR_MSG: str = (
        "real tSS-BO not enabled - set CLAY_USE_TSSBO=1 and "
        "install package"
    )

    def __init__(
        self,
        bounds: Sequence[SizingBounds],
        *,
        seed: int = 0,
        budget: Optional[BoBudgetTracker] = None,
        enabled_by_cli: bool = False,
        backend: Optional[_TSsBoBackend] = None,
        source_declared: bool = False,
    ) -> None:
        # ── Layer 1: env var ────────────────────────────────────────
        if not source_declared and not _is_opted_in():
            raise RuntimeError(self.OPT_IN_ERROR_MSG)

        # ── Layer 2: CLI flag ───────────────────────────────────────
        # Direct programmatic use is allowed but must explicitly
        # acknowledge the flag layer. The CLI sets ``enabled_by_cli``
        # after verifying ``--optimizer tssbo``.
        if not source_declared and not enabled_by_cli:
            raise RuntimeError(
                self.OPT_IN_ERROR_MSG
                + " (CLI flag layer: pass enabled_by_cli=True or use "
                "the `--optimizer tssbo` CLI dispatch)"
            )

        if not bounds:
            raise ValueError(
                "TSsBoAdapter requires at least one SizingBounds; "
                "empty bounds list is not supported."
            )

        self.bounds = tuple(bounds)
        self.seed = seed
        self.source_declared = bool(source_declared)
        self.budget = budget if budget is not None else BoBudgetTracker(
            label="tssbo-adapter",
        )

        # ── Layer 3: package present ────────────────────────────────
        if backend is None and not source_declared:
            module_name = _resolve_module_name()
            backend = _create_real_backend(
                module_name, self.bounds, self.seed,
            )
        self._backend = backend

        # Start the wall-clock so subsequent budget checks have a
        # reference. Idempotent — safe if the tracker has been
        # pre-started elsewhere.
        self.budget.start_clock()

        # Internal RNG for any logic that needs randomness OUTSIDE
        # the BO surrogate (rare; the upstream lib owns its own
        # seeded PRNG).
        self._rng = random.Random(seed)

    # ── OptimizerInterface ──────────────────────────────────────────

    def propose(self, history: OptimizerHistory) -> Sizing:
        """Ask the BO surrogate for the next sizing.

        Layer 4 (budget) fires here: ``check_proposal`` and
        ``check_wall_clock`` both run before the surrogate is
        consulted. Both can raise :class:`BudgetExceededError`.

        The returned dict is guaranteed to contain exactly the knobs
        in :attr:`bounds` and to respect each knob's bounds (the
        adapter clamps any drifted-out-of-bounds proposal the
        backend returns).
        """
        if self._backend is None:
            raise RuntimeError(
                "source-declared tSS-BO uses solve_constrained(), not ask/tell propose()"
            )
        self.budget.check_wall_clock()
        self.budget.check_proposal()
        raw = self._backend.suggest(history)
        out: dict[str, float] = {}
        for b in self.bounds:
            v = raw.get(b.knob)
            if v is None or not math.isfinite(v):
                # Backend returned NaN / missing knob — fall back to
                # the bound midpoint so the harness's cell builder
                # always gets a valid number.
                v = 0.5 * (b.low + b.high)
            # Clamp to bound. tSS-BO proposals can drift outside the
            # constraint set by a fraction of a percent due to the
            # truncated subspace sampling; clamp defensively.
            v = max(b.low, min(b.high, v))
            out[b.knob] = float(v)
        self.budget.record_proposal()
        return out

    def update(
        self,
        history: OptimizerHistory,
        sizing: Sizing,
        result: OptimizerHistoryEntry,
    ) -> None:
        """Push one evaluation result back into the BO surrogate.

        Layer 4 (budget) fires here for the iter / ngspice counters.
        :class:`BudgetExceededError` propagates so the harness sees
        the refusal — but the harness already records the entry
        BEFORE calling ``update``, so the failed iter is in
        ``history`` regardless.
        """
        _ = history
        self.budget.check_wall_clock()
        self.budget.check_iter()
        # Only count an ngspice call if the simulator actually ran
        # (the harness sets ``measures`` non-empty iff the sim
        # produced output; pre-filter rejections have measures={}).
        if result.measures:
            self.budget.check_ngspice_call()
        # Forward the observation to the backend. Reward is the
        # signed-slack scalar the harness computed; feasible is the
        # harness's coarse mask.
        self._backend.observe(
            dict(sizing), float(result.reward), bool(result.is_feasible),
        )
        # Record budget post-success so a refused call (above)
        # doesn't bump the counters.
        self.budget.record_iter()
        if result.measures:
            self.budget.record_ngspice_call()

    def solve_constrained(
        self,
        funct: Callable[[Any], Any],
        *,
        n_max: int,
        outdim: int,
        batch_size: int = 4,
        n_candidates: int = 80,
        n_resample: int = 5,
        init_x: Any = None,
        init_y: Any = None,
    ):
        """Dispatch the source-declared closed-loop constrained API.

        This path is authorized by explicit ``.clay`` intent and therefore
        does not consult the legacy process-global opt-in flag.  It calls the
        checked-in DATE-2024 implementation through ``tssbo_solver``.
        """
        if not self.source_declared:
            raise RuntimeError("solve_constrained requires source-declared tSS-BO intent")
        self.budget.check_max_iters(int(n_max))
        from .tssbo_solver import minimize_constrained

        return minimize_constrained(
            funct,
            dim=len(self.bounds),
            n_max=int(n_max),
            outdim=int(outdim),
            batch_size=int(batch_size),
            n_candidates=int(n_candidates),
            n_resample=int(n_resample),
            seed=int(self.seed),
            init_x=init_x,
            init_y=init_y,
        )


# ─── Convenience factory (used by the CLI) ───────────────────────────


def build_tssbo_adapter(
    bounds: Sequence[SizingBounds],
    *,
    seed: int = 0,
    budget: Optional[BoBudgetTracker] = None,
    backend: Optional[_TSsBoBackend] = None,
) -> TSsBoAdapter:
    """Build a :class:`TSsBoAdapter` with the CLI flag layer marked
    as satisfied. Used by ``tools.clay_optimize.run`` after it has
    verified the ``--optimizer tssbo`` flag. Direct programmatic
    callers can use this when they intend to bypass the CLI.

    Layers 1 / 3 / 4 still fire (env / package / budget). Only the
    CLI-flag layer is short-circuited.
    """
    return TSsBoAdapter(
        bounds=bounds,
        seed=seed,
        budget=budget,
        enabled_by_cli=True,
        backend=backend,
    )


def build_tssbo_adapter_from_intent(
    bounds: Sequence[SizingBounds],
    *,
    seed: int,
    budget: Optional[BoBudgetTracker] = None,
) -> TSsBoAdapter:
    """Build the constrained adapter authorized by compiled ``.clay`` intent."""
    return TSsBoAdapter(
        bounds=bounds,
        seed=seed,
        budget=budget,
        source_declared=True,
    )


__all__ = [
    "OPT_IN_ENV",
    "MODULE_NAME_ENV",
    "TSsBoAdapter",
    "build_tssbo_adapter",
    "build_tssbo_adapter_from_intent",
    "BudgetExceededError",
    "_MockTSsBoBackend",
    "_TSsBoBackend",
    "_create_real_backend",
    "_is_opted_in",
    "_resolve_module_name",
]
