"""
Optimizer harness — orchestration glue between an optimizer (BO / grid /
random / Adjoint) and the Clay verify pipeline.

The harness owns the **loop** (propose → build cell → semantic check →
simulate → score → update). The optimizer owns **search strategy**
(how to propose the next sizing given history). This decoupling is
the whole point of v0.0.67 Phase 3 first slice: it lets tSS-BO drop
in at v0.0.68+ without touching any of the cell construction / verify
plumbing.

Three integration points the optimizer sees through ``OptimizerLoop``:

  1. ``propose(history)`` — given everything seen so far (including
     failed iterations), return the next sizing to try.
  2. The harness pre-filters the proposed sizing through the Phase 2
     Rust semantic check whenever ``clay_ir`` is available; obviously
     infeasible sizings get recorded with ``is_feasible=False`` and a
     non-zero ``violations`` list, and ngspice is skipped to save the
     budget. The optimizer's ``update`` still gets called so it can
     learn the negative sample.
  3. ``update(history, sizing, result)`` — the harness pushes the
     just-recorded entry to the optimizer so it can update its
     surrogate / posterior / grid state.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Iterable, List, Mapping, Optional, Protocol,
    Sequence, runtime_checkable,
)


# ─── Knob types ───────────────────────────────────────────────────


Sizing = Mapping[str, float]
"""Ordered map of knob name → physical value.

A sizing is what the optimizer proposes and what the cell builder
consumes as kwargs (e.g. ``build_two_stage_miller_pmos(name, spec,
gm_id_diff=15.0, gm_id_load1=10.0, gm_id_cs=12.0)``). Stored as a
plain dict in practice; the alias makes signatures self-documenting.
"""


@dataclass(frozen=True)
class SizingBounds:
    """Per-knob lower / upper bound used by the optimizer to sample.

    Used by ``RandomSearchStub`` for uniform sampling and by
    ``GridSearchStub`` for axis enumeration. The real tSS-BO wrapper
    (v0.0.68+) will consume the same bound list to set its truncated
    subspace prior.

    A bound is **inclusive** at both endpoints. ``log_scale=True`` tells
    samplers to draw uniformly in log space (typical for W / I_ref /
    Cc that span orders of magnitude); the default linear uniform is
    appropriate for gm/Id targets that live in [3, 25].
    """

    knob: str
    low: float
    high: float
    log_scale: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(
                f"SizingBounds[{self.knob!r}] must be finite, "
                f"got low={self.low!r}, high={self.high!r}"
            )
        if self.high < self.low:
            raise ValueError(
                f"SizingBounds[{self.knob!r}]: high={self.high!r} "
                f"must be >= low={self.low!r}"
            )
        if self.log_scale and (self.low <= 0.0 or self.high <= 0.0):
            raise ValueError(
                f"SizingBounds[{self.knob!r}]: log_scale=True requires "
                f"both bounds positive, got low={self.low!r}, "
                f"high={self.high!r}"
            )


# ─── Spec / reward types ─────────────────────────────────────────


@dataclass(frozen=True)
class SpecTarget:
    """One target the optimizer tries to satisfy.

    Mirrors the shape of ``clay.ir.SpecField`` deliberately so the
    target list can be derived from a parsed Clay file in later
    versions without re-typing. The harness only consumes ``name``,
    ``op``, and ``value`` — units are informational and only used in
    the CLI / report rendering.

    Supported operators: ``>=``, ``<=``, ``>``, ``<``, ``=``.
    """

    name: str
    op: str
    value: float
    units: str = ""

    def __post_init__(self) -> None:
        if self.op not in (">=", "<=", ">", "<", "="):
            raise ValueError(
                f"SpecTarget[{self.name!r}]: unsupported op {self.op!r}; "
                f"choose one of >=, <=, >, <, ="
            )
        if not math.isfinite(self.value):
            raise ValueError(
                f"SpecTarget[{self.name!r}]: value must be finite, "
                f"got {self.value!r}"
            )


# ─── History records ─────────────────────────────────────────────


@dataclass(frozen=True)
class OptimizerHistoryEntry:
    """One iteration's record in the optimizer history.

    Fields:
      * ``iteration``    — 0-indexed iteration counter.
      * ``sizing``       — the knob dict the optimizer proposed.
      * ``measures``     — ngspice-extracted (or post-processed) scalar
                           measure dict. Empty if pre-filter rejected
                           or the simulator errored.
      * ``violations``   — list of semantic-violation strings from the
                           Phase 2 pre-filter. Empty when ``clay_ir``
                           isn't available (the harness treats absent
                           pre-filter as "no info", not "infeasible").
      * ``is_feasible``  — True iff the pre-filter accepted the sizing
                           AND the simulator produced measures. False
                           on any rejection path so optimizers can
                           use it as a coarse mask.
      * ``reward``       — scalar reward (higher = closer to target).
                           NaN if the iteration produced no measures.
      * ``simulator_error`` — non-None when the driver failed at the
                              run / parse step. Surface for debugging.
      * ``elapsed_sec``  — wall-clock seconds spent on this iteration
                           (cell build + pre-filter + simulate +
                           reward). Useful when ranking optimizer
                           strategies on compute budget.
    """

    iteration: int
    sizing: Mapping[str, float]
    measures: Mapping[str, float] = field(default_factory=dict)
    violations: Sequence[str] = field(default_factory=tuple)
    is_feasible: bool = False
    reward: float = float("nan")
    simulator_error: Optional[str] = None
    elapsed_sec: float = 0.0


@dataclass
class OptimizerHistory:
    """Append-only history of an optimizer run.

    The optimizer sees the full history on every ``propose`` call (it
    is free to ignore most of it). The CLI consumes it for the final
    table; tests inspect it to assert invariants.

    Convenience views:
      * :meth:`best_entry` — highest-reward feasible entry, or the
        highest-reward entry overall if none are feasible. None when
        the history is empty.
      * :meth:`feasible_entries` — only the entries that cleared the
        pre-filter AND produced measures.
    """

    entries: List[OptimizerHistoryEntry] = field(default_factory=list)

    def append(self, entry: OptimizerHistoryEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def feasible_entries(self) -> List[OptimizerHistoryEntry]:
        return [e for e in self.entries if e.is_feasible]

    def best_entry(self) -> Optional[OptimizerHistoryEntry]:
        if not self.entries:
            return None
        # Prefer feasible entries; among them pick the highest reward
        # (NaN-safe — NaN compares False against any finite, so it
        # never wins). Fall back to the overall highest-reward entry
        # (even if that's also NaN — caller can detect via .measures
        # being empty).
        feas = self.feasible_entries()
        pool = feas if feas else self.entries

        def _key(e: OptimizerHistoryEntry) -> float:
            return e.reward if not math.isnan(e.reward) else -math.inf

        return max(pool, key=_key)


# ─── Optimizer protocol ──────────────────────────────────────────


@runtime_checkable
class OptimizerInterface(Protocol):
    """The contract every optimizer satisfies.

    Two methods:

      * ``propose(history)`` — given the prefix of evaluations, return
        the next sizing dict. Called BEFORE the harness runs the cell.
        The optimizer is free to ignore the history (random search) or
        consume only the last entry (gradient descent) or every entry
        (BO surrogate refit). Must return a non-empty dict whose keys
        match the optimizer's knob list.

      * ``update(history, sizing, result)`` — called AFTER the harness
        records the result. The history already contains ``result``
        as its last entry; we pass ``sizing`` and ``result`` directly
        as a convenience so stateful optimizers don't have to re-scan
        the tail. May be a no-op for stateless optimizers.

    Implementations are NOT required to be deterministic, but the
    stubs in :mod:`stub_optimizers` are seeded so tests can assert
    reproducibility.
    """

    def propose(self, history: OptimizerHistory) -> Sizing:  # pragma: no cover - protocol
        ...

    def update(
        self,
        history: OptimizerHistory,
        sizing: Sizing,
        result: OptimizerHistoryEntry,
    ) -> None:  # pragma: no cover - protocol
        ...


# ─── Reward function ─────────────────────────────────────────────


def default_reward(
    measures: Mapping[str, float],
    targets: Sequence[SpecTarget],
) -> float:
    """Default reward: aggregate signed slack across all targets.

    Sign convention (higher = better, target met = >= 0):

      * For ``>=`` / ``>`` targets: slack = (actual - target) /
        max(|target|, eps). Positive means the target is met with
        margin; negative means the actual is short.
      * For ``<=`` / ``<`` targets: slack = (target - actual) /
        max(|target|, eps). Same sign convention.
      * For ``=`` targets: slack = -|actual - target| /
        max(|target|, eps). Always non-positive; closer to zero means
        closer to target.

    Aggregation: the reward is the **min** of the per-target slacks
    (worst-case bottleneck). The min is a natural choice for "all
    targets must be satisfied" — it stays negative until every target
    has positive slack, and rises in step with the worst-violating
    target. Replace with a weighted sum or a logarithmic barrier when
    you have per-target weights (left to v0.0.68+ when tSS-BO arrives).

    Returns NaN when any required measure is absent from ``measures``
    (the harness records the iteration as infeasible). NaN propagates
    through ``best_entry`` cleanly — feasible entries always rank
    above NaN-reward ones.
    """
    if not targets:
        return 0.0
    eps = 1e-12
    slacks: List[float] = []
    for t in targets:
        actual = measures.get(t.name)
        if actual is None or not math.isfinite(actual):
            return float("nan")
        denom = max(abs(t.value), eps)
        if t.op in (">=", ">"):
            slack = (actual - t.value) / denom
        elif t.op in ("<=", "<"):
            slack = (t.value - actual) / denom
        else:  # op == "="
            slack = -abs(actual - t.value) / denom
        slacks.append(slack)
    return min(slacks)


# ─── Optimizer loop ──────────────────────────────────────────────


# Type aliases for the loop's callables. Kept loose (Any) because the
# Clay IR is heavy enough that pulling its types in for a callable
# signature wouldn't add anything readable.
CellBuilder = Callable[[Sizing], Any]
"""Given a sizing dict, build a clay.ir.Cell. The harness is dialect-
agnostic; the builder closes over the cell name, base spec, PDK, and
any non-optimizable kwargs."""

TestbenchBuilder = Callable[[Any], Any]
"""Given a Cell, build a clay.ir.Testbench. The harness threads the
builder through; default for v0.0.67 is two_stage_miller_pmos's
build_ac_testbench."""

Simulator = Callable[[Any, Path], Mapping[str, Any]]
"""Given a Testbench and a work dir, return a dict with at least:

  * ``measures`` — ``Mapping[str, float]`` of extracted scalars.
  * ``simulator_error`` — ``Optional[str]`` non-None on driver failure.

This is the only seam the harness exposes for mocking out ngspice in
tests. The default points at ``clay.verifier.verify`` and consumes its
VerifyResult."""


class OptimizerLoop:
    """Top-level orchestrator.

    Construct once with cell + testbench + simulator + targets, then
    call ``run(optimizer, max_iters=...)`` to execute the loop.

    The loop is hermetic: it does NOT mutate the optimizer's state on
    any path except via ``optimizer.update``. It does NOT raise on
    individual iteration failures — every exception inside the cell
    build / pre-filter / simulator step gets caught, recorded as an
    ``OptimizerHistoryEntry`` with ``is_feasible=False``, and the
    optimizer's ``update`` still fires so the optimizer can learn from
    the failure.

    Args:
      cell_builder: ``Sizing -> Cell``. Called once per iteration.
      testbench_builder: ``Cell -> Testbench``. Called once per
        iteration; the harness doesn't cache it because some testbench
        builders depend on the cell's resolved PDK.
      simulator: ``(Testbench, work_dir) -> dict``. Defaults to the
        ``clay.verifier.verify`` wrapper; tests pass a stub.
      targets: list of ``SpecTarget`` — the reward function consumes
        these.
      semantic_check_fn: optional ``(Sizing, Cell) -> tuple[list[str],
        bool]``. Returns ``(violation_strings, is_feasible)``. The
        boolean controls whether ngspice runs — ``False`` skips the
        sim. Defaults to the **informational** check that always
        returns ``is_feasible=True`` so v0.0.67's first slice doesn't
        regress on cells the static analyzer can't yet model (the
        textbook two_stage_miller_pmos has a known ``MissingLoad`` on
        its bias tail that's not actually a defect — Phase 2 §
        cascode-stack feasibility doesn't cover NMOS-mirror sink
        chains, so blocking on it would be a false negative). Pass
        ``semantic_check_fn=strict_semantic_check`` to enable hard
        rejection at the optimizer level once the Phase 2 LUT is
        more complete.
      work_dir: root dir for ngspice intermediate files. Each iteration
        gets a fresh sub-dir keyed on iteration index.
      reward_fn: defaults to ``default_reward``. Test injection point
        for alternative shaping experiments.
      supply_vdd: nominal supply voltage used by the semantic check.
        Defaults to 1.8 (sky130 nominal); the caller can override for
        gf180mcu or other PDKs.
      pdk_id: PDK identifier forwarded to the semantic check. Default
        matches sky130; respected when the test injects a custom
        semantic_check_fn that wants to switch on PDK.
    """

    def __init__(
        self,
        *,
        cell_builder: CellBuilder,
        testbench_builder: TestbenchBuilder,
        simulator: Simulator,
        targets: Sequence[SpecTarget],
        semantic_check_fn: Optional[
            Callable[[Sizing, Any], tuple[list[str], bool]]
        ] = None,
        work_dir: Path | str = "tmp_clay_optimize",
        reward_fn: Callable[
            [Mapping[str, float], Sequence[SpecTarget]], float
        ] = default_reward,
        supply_vdd: float = 1.8,
        pdk_id: str = "sky130",
    ) -> None:
        self.cell_builder = cell_builder
        self.testbench_builder = testbench_builder
        self.simulator = simulator
        self.targets = tuple(targets)
        self.semantic_check_fn = (
            semantic_check_fn
            if semantic_check_fn is not None
            else _default_semantic_check_factory(supply_vdd, pdk_id)
        )
        self.work_dir = Path(work_dir)
        self.reward_fn = reward_fn
        self.supply_vdd = supply_vdd
        self.pdk_id = pdk_id

    def run(
        self,
        optimizer: OptimizerInterface,
        max_iters: int,
        *,
        history: Optional[OptimizerHistory] = None,
    ) -> OptimizerHistory:
        """Drive the loop for at most ``max_iters`` iterations.

        Returns the populated history. If the same ``history`` is
        passed across multiple ``run`` calls (e.g. checkpoint /
        resume), iteration indices continue from the last entry's
        ``iteration + 1``.
        """
        if max_iters <= 0:
            raise ValueError(f"max_iters must be > 0, got {max_iters}")
        if history is None:
            history = OptimizerHistory()
        start_idx = (
            history.entries[-1].iteration + 1 if history.entries else 0
        )
        for offset in range(max_iters):
            iter_idx = start_idx + offset
            entry = self._run_one_iter(optimizer, history, iter_idx)
            history.append(entry)
            # Let the optimizer learn from this entry. The harness
            # tolerates exceptions inside `update` — the optimizer may
            # raise on degenerate input, but we don't abort the loop.
            try:
                optimizer.update(history, entry.sizing, entry)
            except Exception:  # noqa: BLE001
                # Silent on purpose: the harness's job is to keep the
                # loop running even if the optimizer's internal state
                # update raises. The next propose() call still gets
                # the full history.
                pass
        return history

    # ─── Internals ────────────────────────────────────────────────

    def _run_one_iter(
        self,
        optimizer: OptimizerInterface,
        history: OptimizerHistory,
        iter_idx: int,
    ) -> OptimizerHistoryEntry:
        t0 = time.monotonic()
        # 1. Optimizer proposes the next sizing. Failures here are
        # surfaced as zero-data entries; the optimizer state stays
        # untouched so the next call still gets a full history.
        try:
            sizing = dict(optimizer.propose(history))
        except Exception as exc:  # noqa: BLE001
            return OptimizerHistoryEntry(
                iteration=iter_idx,
                sizing={},
                violations=(f"optimizer.propose raised: {exc!r}",),
                is_feasible=False,
                elapsed_sec=time.monotonic() - t0,
            )

        # 2. Build the cell. A builder exception means the proposed
        # sizing was structurally invalid — record and move on.
        try:
            cell = self.cell_builder(sizing)
        except Exception as exc:  # noqa: BLE001
            return OptimizerHistoryEntry(
                iteration=iter_idx,
                sizing=sizing,
                violations=(f"cell_builder raised: {exc!r}",),
                is_feasible=False,
                elapsed_sec=time.monotonic() - t0,
            )

        # 3. Phase 2 cheap pre-filter. Skip ngspice if the static
        # analyzer flags the sizing as infeasible — saves the sim
        # budget for promising candidates. The semantic_check_fn
        # signature is intentionally permissive (sizing + cell) so
        # tests can inject deterministic rejection logic without
        # touching the Rust path.
        try:
            violations, feasible_static = self.semantic_check_fn(sizing, cell)
        except Exception as exc:  # noqa: BLE001
            return OptimizerHistoryEntry(
                iteration=iter_idx,
                sizing=sizing,
                violations=(f"semantic_check raised: {exc!r}",),
                is_feasible=False,
                elapsed_sec=time.monotonic() - t0,
            )

        if not feasible_static:
            # Don't burn a sim run on a sizing the static analyzer
            # already rejected. Reward stays NaN so the best_entry
            # tracker ignores it; the optimizer sees the full
            # violation list as a learning signal.
            return OptimizerHistoryEntry(
                iteration=iter_idx,
                sizing=sizing,
                violations=tuple(violations),
                is_feasible=False,
                elapsed_sec=time.monotonic() - t0,
            )

        # 4. Build testbench, run simulator. Each iteration gets a
        # fresh sub-dir so concurrent runs don't clobber each other's
        # logs.
        iter_work = self.work_dir / f"iter_{iter_idx:04d}"
        try:
            tb = self.testbench_builder(cell)
            sim_out = self.simulator(tb, iter_work)
        except Exception as exc:  # noqa: BLE001
            return OptimizerHistoryEntry(
                iteration=iter_idx,
                sizing=sizing,
                violations=tuple(violations),
                is_feasible=False,
                simulator_error=f"{type(exc).__name__}: {exc}",
                elapsed_sec=time.monotonic() - t0,
            )

        measures = dict(sim_out.get("measures") or {})
        sim_error = sim_out.get("simulator_error")
        if sim_error is not None:
            return OptimizerHistoryEntry(
                iteration=iter_idx,
                sizing=sizing,
                violations=tuple(violations),
                is_feasible=False,
                simulator_error=str(sim_error),
                elapsed_sec=time.monotonic() - t0,
            )

        # 5. Score.
        reward = self.reward_fn(measures, self.targets)
        # An iteration is feasible iff the pre-filter accepted it AND
        # the simulator ran AND the reward is a finite scalar (no
        # missing measures).
        is_feasible = math.isfinite(reward)
        return OptimizerHistoryEntry(
            iteration=iter_idx,
            sizing=sizing,
            measures=measures,
            violations=tuple(violations),
            is_feasible=is_feasible,
            reward=reward,
            simulator_error=None,
            elapsed_sec=time.monotonic() - t0,
        )


# ─── Semantic-check helpers ─────────────────────────────────────


def _default_semantic_check_factory(
    supply_vdd: float, pdk_id: str
) -> Callable[[Sizing, Any], tuple[list[str], bool]]:
    """Return an **informational** semantic-check callable.

    The default policy is "report violations but never block": the
    returned callable runs the Phase 2 Rust analyzer (when available),
    forwards every violation string to the optimizer, and ALWAYS sets
    ``is_feasible=True``. The optimizer can read history entries'
    ``violations`` field to penalize known-bad sizings, but the
    harness still asks ngspice for an actual measurement — because
    Phase 2 currently has false-negatives on textbook templates
    (e.g. the ``MissingLoad`` flag fires on the bias tail of the
    PMOS-input two-stage Miller opamp, which is a perfectly working
    circuit).

    Switch to :func:`strict_semantic_check_factory` to opt in to
    skipping ngspice on any flagged sizing. Strict mode is preferable
    once the Phase 2 LUT covers NMOS-mirror sink chains.

    When ``clay_ir`` isn't available the callable returns
    ``([], True)`` (no info, no block) — Python-only environments
    still run the loop.
    """

    def _check(sizing: Sizing, cell: Any) -> tuple[list[str], bool]:
        _ = sizing  # informational: violations come from the cell graph
        try:
            from clay._rust_bridge import cell_to_hypergraph, has_rust_ir
        except ImportError:
            return ([], True)
        if not has_rust_ir():
            return ([], True)
        try:
            handle = cell_to_hypergraph(cell)
        except Exception:  # noqa: BLE001
            return ([], True)
        if handle is None:
            return ([], True)
        try:
            violations = handle.semantic_check(supply_vdd, pdk_id)
        except Exception as exc:  # noqa: BLE001
            # Even on a Rust-side crash we keep the loop going. The
            # exception message lands in the entry's violations list
            # so the optimizer / debugger can see it.
            return ([f"semantic_check raised: {exc!r}"], True)
        return (list(violations), True)

    return _check


def strict_semantic_check_factory(
    supply_vdd: float, pdk_id: str
) -> Callable[[Sizing, Any], tuple[list[str], bool]]:
    """Return a **strict** semantic-check callable.

    Same routing as the informational factory, but ``is_feasible``
    flips to ``False`` whenever the Phase 2 analyzer returns any
    violations. Use when the Phase 2 LUT is known to be complete
    enough for the cell family under test, or in adversarial /
    fuzz-style explorations where false positives are preferable to
    burning the simulator budget on garbage sizings.
    """

    def _check(sizing: Sizing, cell: Any) -> tuple[list[str], bool]:
        _ = sizing
        try:
            from clay._rust_bridge import cell_to_hypergraph, has_rust_ir
        except ImportError:
            return ([], True)
        if not has_rust_ir():
            return ([], True)
        try:
            handle = cell_to_hypergraph(cell)
        except Exception:  # noqa: BLE001
            return ([], True)
        if handle is None:
            return ([], True)
        try:
            violations = handle.semantic_check(supply_vdd, pdk_id)
        except Exception as exc:  # noqa: BLE001
            return ([f"semantic_check raised: {exc!r}"], False)
        return (list(violations), len(violations) == 0)

    return _check
