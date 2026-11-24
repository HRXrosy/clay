"""Deterministic minimal-subset search over an injected acceptance oracle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
import itertools
import re
from typing import Callable

from clay.layout.intent_basis import MinimalityScope
from clay.layout.intent_closure import (
    ConsumerProjection,
    IntentEntailment,
    evaluate_intent_closure,
)
from clay.layout.intent_facts import IntentFactSet, SourceInventory


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ORACLE_STAGES = frozenset(f"G{index}" for index in range(8))
_BASIS_CLOSURE_STAGES = frozenset({"G2", "G3"})


def _require_sha256(value: object, *, field: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256 digest")


def _validate_closure_binding(
    *,
    closure_receipt_sha256: object,
    oracle_max_stage: object,
) -> None:
    if closure_receipt_sha256 is None and oracle_max_stage is None:
        return
    if closure_receipt_sha256 is None or oracle_max_stage is None:
        raise ValueError(
            "closure_receipt_sha256 and oracle_max_stage must be bound together"
        )
    _require_sha256(
        closure_receipt_sha256,
        field="closure_receipt_sha256",
    )
    if type(oracle_max_stage) is not str or oracle_max_stage not in _ORACLE_STAGES:
        raise ValueError("oracle_max_stage must be one of G0 through G7")


def _require_sorted_unique_strings(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be an exact tuple")
    if any(type(item) is not str or not item for item in value):
        raise TypeError(f"{field} must contain exact non-empty strings")
    if (not allow_empty and not value) or tuple(sorted(set(value))) != value:
        raise ValueError(f"{field} must contain sorted unique values")
    return value


@dataclass(frozen=True)
class BasisEvaluation:
    accepted: bool
    failed_gate: str | None
    evidence_sha256: str
    residual_cost_bits: int = 0
    closure_receipt_sha256: str | None = None
    oracle_max_stage: str | None = None

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be an exact bool")
        if self.accepted:
            if self.failed_gate is not None:
                raise ValueError("accepted evaluation must have failed_gate=None")
        elif type(self.failed_gate) is not str or not self.failed_gate:
            raise ValueError("rejected evaluation must name a failed_gate")
        _require_sha256(self.evidence_sha256, field="evidence_sha256")
        if type(self.residual_cost_bits) is not int or self.residual_cost_bits < 0:
            raise TypeError(
                "residual_cost_bits must be an exact non-negative integer"
            )
        _validate_closure_binding(
            closure_receipt_sha256=self.closure_receipt_sha256,
            oracle_max_stage=self.oracle_max_stage,
        )


@total_ordering
@dataclass(frozen=True)
class BasisCounterexample:
    removed_atom_ids: tuple[str, ...]
    failed_gate: str
    evidence_sha256: str
    closure_receipt_sha256: str | None = None
    oracle_max_stage: str | None = None

    def __post_init__(self) -> None:
        _require_sorted_unique_strings(
            self.removed_atom_ids,
            field="removed_atom_ids",
            allow_empty=False,
        )
        if type(self.failed_gate) is not str or not self.failed_gate:
            raise TypeError("failed_gate must be an exact non-empty string")
        _require_sha256(self.evidence_sha256, field="evidence_sha256")
        _validate_closure_binding(
            closure_receipt_sha256=self.closure_receipt_sha256,
            oracle_max_stage=self.oracle_max_stage,
        )

    def _sort_key(self) -> tuple[tuple[str, ...], str, str, str, str]:
        return (
            self.removed_atom_ids,
            self.failed_gate,
            self.evidence_sha256,
            self.closure_receipt_sha256 or "",
            self.oracle_max_stage or "",
        )

    def __lt__(self, other: object) -> bool:
        if type(other) is not BasisCounterexample:
            return NotImplemented
        return self._sort_key() < other._sort_key()


@dataclass(frozen=True)
class BasisSearchResult:
    selected_atom_ids: tuple[str, ...]
    intent_cost_bits: int
    oracle_calls: int
    cache_hits: int
    method: str
    minimality_scope: MinimalityScope
    counterexamples: tuple[BasisCounterexample, ...]
    budget_exhausted: bool
    closure_receipt_sha256: str | None = None
    oracle_max_stage: str | None = None

    def __post_init__(self) -> None:
        _require_sorted_unique_strings(
            self.selected_atom_ids,
            field="selected_atom_ids",
            allow_empty=True,
        )
        for field_name in ("intent_cost_bits", "oracle_calls", "cache_hits"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise TypeError(
                    f"{field_name} must be an exact non-negative integer"
                )
        if type(self.method) is not str or not self.method:
            raise TypeError("method must be an exact non-empty string")
        if type(self.minimality_scope) is not MinimalityScope:
            raise TypeError("minimality_scope must be an exact MinimalityScope")
        if type(self.counterexamples) is not tuple or any(
            type(item) is not BasisCounterexample for item in self.counterexamples
        ):
            raise TypeError("counterexamples must be an exact tuple of counterexamples")
        if tuple(sorted(set(self.counterexamples))) != self.counterexamples:
            raise ValueError("counterexamples must contain sorted unique rows")
        if type(self.budget_exhausted) is not bool:
            raise TypeError("budget_exhausted must be an exact bool")
        _validate_closure_binding(
            closure_receipt_sha256=self.closure_receipt_sha256,
            oracle_max_stage=self.oracle_max_stage,
        )
        closure_bound = self.closure_receipt_sha256 is not None
        for row in self.counterexamples:
            if closure_bound and row.closure_receipt_sha256 is None:
                raise ValueError(
                    "closure-bound result requires every counterexample binding"
                )
            if not closure_bound and row.closure_receipt_sha256 is not None:
                raise ValueError(
                    "legacy result cannot contain a closure-bound counterexample"
                )
            if closure_bound and int(row.oracle_max_stage[1:]) > int(
                self.oracle_max_stage[1:]
            ):
                raise ValueError(
                    "counterexample oracle stage exceeds the closure-bound result"
                )
        selected = frozenset(self.selected_atom_ids)
        if any(not set(row.removed_atom_ids) <= selected for row in self.counterexamples):
            raise ValueError("counterexample removes an atom outside the selected basis")
        proved_singletons = {
            row.removed_atom_ids
            for row in self.counterexamples
            if len(row.removed_atom_ids) == 1
        }
        required_singletons = {(atom_id,) for atom_id in self.selected_atom_ids}
        deletion_proven = proved_singletons == required_singletons
        if self.method == "exact-bnb":
            expected_scope = (
                MinimalityScope.UNPROVEN
                if self.budget_exhausted
                else MinimalityScope.GLOBAL
            )
        elif self.method == "beam":
            expected_scope = (
                MinimalityScope.DELETION_MINIMAL
                if deletion_proven and not self.budget_exhausted
                else MinimalityScope.BOUNDED_SEARCH
            )
        else:
            raise ValueError(f"unsupported intent-basis search method {self.method!r}")
        if self.minimality_scope is not expected_scope:
            raise ValueError(
                "minimality scope disagrees with search method, budget completion, "
                "and recorded counterexamples"
            )
        if expected_scope is MinimalityScope.DELETION_MINIMAL:
            if proved_singletons != required_singletons:
                raise ValueError(
                    "completed minimality scope lacks one counterexample per selected atom"
                )


class _OracleBudgetExhausted(RuntimeError):
    pass


class _MemoizedEvaluator:
    def __init__(
        self,
        evaluate: Callable[[frozenset[str]], BasisEvaluation],
        *,
        max_oracle_calls: int | None,
    ) -> None:
        self._evaluate = evaluate
        self._maximum = max_oracle_calls
        self._cache: dict[tuple[str, ...], BasisEvaluation] = {}
        self.oracle_calls = 0
        self.cache_hits = 0
        self.budget_exhausted = False

    def __call__(self, selected: tuple[str, ...]) -> BasisEvaluation:
        if selected in self._cache:
            self.cache_hits += 1
            return self._cache[selected]
        if self._maximum is not None and self.oracle_calls >= self._maximum:
            self.budget_exhausted = True
            raise _OracleBudgetExhausted
        result = self._evaluate(frozenset(selected))
        if type(result) is not BasisEvaluation:
            raise TypeError("evaluate must return an exact BasisEvaluation")
        self._cache[selected] = result
        self.oracle_calls += 1
        return result

    def cached(self, selected: tuple[str, ...]) -> BasisEvaluation | None:
        return self._cache.get(selected)


def _validate_inputs(
    *,
    atom_costs: object,
    evaluate: object,
    exact_limit: object,
    beam_width: object,
    max_oracle_calls: object,
) -> tuple[tuple[tuple[str, int], ...], Callable[[frozenset[str]], BasisEvaluation]]:
    if type(atom_costs) is not tuple:
        raise TypeError("atom_costs must be an exact tuple")
    normalized: list[tuple[str, int]] = []
    for row in atom_costs:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("atom_costs rows must be exact 2-tuples")
        atom_id, cost = row
        if type(atom_id) is not str or not atom_id:
            raise TypeError("atom IDs must be exact non-empty strings")
        if type(cost) is not int or cost < 0:
            raise TypeError("atom costs must be exact non-negative integers")
        normalized.append((atom_id, cost))
    if tuple(sorted(set(normalized))) != tuple(normalized) or len(
        {atom_id for atom_id, _ in normalized}
    ) != len(normalized):
        raise ValueError("atom_costs must contain sorted unique atom IDs")
    if not callable(evaluate):
        raise TypeError("evaluate must be callable")
    if type(exact_limit) is not int or exact_limit < 0:
        raise TypeError("exact_limit must be an exact non-negative integer")
    if type(beam_width) is not int or beam_width <= 0:
        raise TypeError("beam_width must be an exact positive integer")
    if max_oracle_calls is not None and (
        type(max_oracle_calls) is not int or max_oracle_calls <= 0
    ):
        raise TypeError("max_oracle_calls must be null or an exact positive integer")
    return tuple(normalized), evaluate  # type: ignore[return-value]


def _base_cost(selected: tuple[str, ...], costs: dict[str, int]) -> int:
    return sum(costs[atom_id] for atom_id in selected)


def _score(
    selected: tuple[str, ...],
    evaluation: BasisEvaluation,
    costs: dict[str, int],
) -> tuple[int, int, tuple[str, ...]]:
    return (
        _base_cost(selected, costs) + evaluation.residual_cost_bits,
        len(selected),
        selected,
    )


def _counterexamples_for_selected(
    selected: tuple[str, ...],
    oracle: _MemoizedEvaluator,
) -> tuple[tuple[BasisCounterexample, ...], bool]:
    counterexamples: list[BasisCounterexample] = []
    every_removal_rejected = True
    for atom_id in selected:
        candidate = tuple(item for item in selected if item != atom_id)
        try:
            evaluation = oracle(candidate)
        except _OracleBudgetExhausted:
            return tuple(sorted(set(counterexamples))), False
        if evaluation.accepted:
            every_removal_rejected = False
            continue
        assert evaluation.failed_gate is not None
        counterexamples.append(
            BasisCounterexample(
                removed_atom_ids=(atom_id,),
                failed_gate=evaluation.failed_gate,
                evidence_sha256=evaluation.evidence_sha256,
                closure_receipt_sha256=evaluation.closure_receipt_sha256,
                oracle_max_stage=evaluation.oracle_max_stage,
            )
        )
    return tuple(sorted(set(counterexamples))), every_removal_rejected


def _exact_search(
    atom_ids: tuple[str, ...],
    costs: dict[str, int],
    oracle: _MemoizedEvaluator,
) -> tuple[tuple[str, ...], BasisEvaluation, bool]:
    # Evaluate the declared upper bound first so an interrupted search can still
    # return a verified candidate instead of silently inventing one.
    best_selected: tuple[str, ...] | None = None
    best_evaluation: BasisEvaluation | None = None
    try:
        upper = oracle(atom_ids)
    except _OracleBudgetExhausted:  # max_oracle_calls is validated positive
        raise RuntimeError("oracle budget ended before the upper bound was evaluated")
    if upper.accepted:
        best_selected = atom_ids
        best_evaluation = upper

    candidates: list[tuple[int, tuple[str, ...]]] = []
    for count in range(len(atom_ids) + 1):
        for selected in itertools.combinations(atom_ids, count):
            candidates.append((_base_cost(selected, costs), selected))
    candidates.sort(key=lambda item: (item[0], item[1]))

    complete = True
    for lower_bound, selected in candidates:
        if best_selected is not None and best_evaluation is not None:
            incumbent = _score(best_selected, best_evaluation, costs)
            # residual_cost_bits is non-negative.  This is a proof-preserving
            # branch-and-bound cut, including the stable ID tie break.
            candidate_lower_bound = (lower_bound, len(selected), selected)
            if candidate_lower_bound >= incumbent:
                continue
        try:
            evaluation = oracle(selected)
        except _OracleBudgetExhausted:
            complete = False
            break
        if not evaluation.accepted:
            continue
        if best_selected is None or best_evaluation is None or _score(
            selected, evaluation, costs
        ) < _score(best_selected, best_evaluation, costs):
            best_selected = selected
            best_evaluation = evaluation

    if best_selected is None or best_evaluation is None:
        raise ValueError("no accepted intent basis was found")
    return best_selected, best_evaluation, complete


def _bounded_search(
    atom_ids: tuple[str, ...],
    costs: dict[str, int],
    oracle: _MemoizedEvaluator,
    *,
    beam_width: int,
) -> tuple[tuple[str, ...], BasisEvaluation]:
    try:
        current_evaluation = oracle(atom_ids)
    except _OracleBudgetExhausted:
        raise RuntimeError("oracle budget ended before the upper bound was evaluated")
    if not current_evaluation.accepted:
        raise ValueError("the full intent basis is not accepted")
    current = atom_ids

    chunk_size = max(2, (len(current) + 1) // 2)
    while chunk_size >= 2 and not oracle.budget_exhausted:
        changed = False
        for start in range(0, len(current), chunk_size):
            removed = frozenset(current[start : start + chunk_size])
            candidate = tuple(atom for atom in current if atom not in removed)
            try:
                evaluation = oracle(candidate)
            except _OracleBudgetExhausted:
                break
            if evaluation.accepted and _score(candidate, evaluation, costs) < _score(
                current, current_evaluation, costs
            ):
                current = candidate
                current_evaluation = evaluation
                changed = True
                break
        if oracle.budget_exhausted:
            break
        if not changed:
            chunk_size //= 2
        elif len(current) < chunk_size:
            chunk_size = max(1, len(current))

    changed = True
    while changed and not oracle.budget_exhausted:
        changed = False
        for atom_id in current:
            candidate = tuple(atom for atom in current if atom != atom_id)
            try:
                evaluation = oracle(candidate)
            except _OracleBudgetExhausted:
                break
            if evaluation.accepted and _score(candidate, evaluation, costs) < _score(
                current, current_evaluation, costs
            ):
                current = candidate
                current_evaluation = evaluation
                changed = True
                break

    # A bounded one-swap pass permits a lower-cost omitted atom to replace a
    # retained expensive atom without turning this fallback into an exponential
    # second exact search.
    omitted = tuple(atom for atom in atom_ids if atom not in frozenset(current))
    swaps: list[tuple[int, tuple[str, ...]]] = []
    for removed in current:
        for added in omitted:
            candidate = tuple(sorted((set(current) - {removed}) | {added}))
            swaps.append((_base_cost(candidate, costs), candidate))
    for _, candidate in sorted(set(swaps), key=lambda item: (item[0], item[1]))[
        :beam_width
    ]:
        try:
            evaluation = oracle(candidate)
        except _OracleBudgetExhausted:
            break
        if evaluation.accepted and _score(candidate, evaluation, costs) < _score(
            current, current_evaluation, costs
        ):
            current = candidate
            current_evaluation = evaluation

    return current, current_evaluation


def search_minimal_basis(
    *,
    atom_costs: tuple[tuple[str, int], ...],
    evaluate: Callable[[frozenset[str]], BasisEvaluation],
    exact_limit: int = 12,
    beam_width: int = 8,
    max_oracle_calls: int | None = None,
) -> BasisSearchResult:
    """Search a finite atom basis under a pure, memoized acceptance oracle.

    Small dimensions use proof-preserving branch-and-bound.  Larger dimensions
    use deterministic chunk deletion, single deletion, and a bounded one-swap
    pass.  Completed single-removal counterexamples are reported separately from
    the physical acceptance verdict.
    """

    rows, evaluator = _validate_inputs(
        atom_costs=atom_costs,
        evaluate=evaluate,
        exact_limit=exact_limit,
        beam_width=beam_width,
        max_oracle_calls=max_oracle_calls,
    )
    atom_ids = tuple(atom_id for atom_id, _ in rows)
    costs = dict(rows)
    oracle = _MemoizedEvaluator(
        evaluator,
        max_oracle_calls=max_oracle_calls,
    )

    if len(atom_ids) <= exact_limit:
        method = "exact-bnb"
        selected, evaluation, complete = _exact_search(atom_ids, costs, oracle)
        counterexamples, _ = _counterexamples_for_selected(selected, oracle)
        scope = (
            MinimalityScope.GLOBAL
            if complete and not oracle.budget_exhausted
            else MinimalityScope.UNPROVEN
        )
    else:
        method = "beam"
        selected, evaluation = _bounded_search(
            atom_ids,
            costs,
            oracle,
            beam_width=beam_width,
        )
        counterexamples, deletion_proven = _counterexamples_for_selected(
            selected,
            oracle,
        )
        scope = (
            MinimalityScope.DELETION_MINIMAL
            if deletion_proven and not oracle.budget_exhausted
            else MinimalityScope.BOUNDED_SEARCH
        )

    return BasisSearchResult(
        selected_atom_ids=selected,
        intent_cost_bits=_score(selected, evaluation, costs)[0],
        oracle_calls=oracle.oracle_calls,
        cache_hits=oracle.cache_hits,
        method=method,
        minimality_scope=scope,
        counterexamples=counterexamples,
        budget_exhausted=oracle.budget_exhausted,
        closure_receipt_sha256=evaluation.closure_receipt_sha256,
        oracle_max_stage=evaluation.oracle_max_stage,
    )


def closure_basis_evaluator(
    *,
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
    entailments: tuple[IntentEntailment, ...],
    projections: tuple[ConsumerProjection, ...],
    maximum_stage: str,
    residual_cost_bits: int = 0,
) -> Callable[[frozenset[str]], BasisEvaluation]:
    """Adapt pure fact closure to the minimal-basis acceptance oracle."""

    if type(fact_set) is not IntentFactSet:
        raise TypeError("fact_set must be an exact IntentFactSet")
    if type(source_inventory) is not SourceInventory:
        raise TypeError("source_inventory must be an exact SourceInventory")
    if type(entailments) is not tuple:
        raise TypeError("entailments must be an exact tuple")
    if type(projections) is not tuple:
        raise TypeError("projections must be an exact tuple")
    if type(maximum_stage) is not str or maximum_stage not in _ORACLE_STAGES:
        raise ValueError("maximum_stage must be one of G0 through G7")
    if maximum_stage not in _BASIS_CLOSURE_STAGES:
        raise ValueError(
            "ORACLE_SCOPE: closure basis evaluator supports only G2 and G3"
        )
    if type(residual_cost_bits) is not int or residual_cost_bits != 0:
        raise ValueError(
            "semantic closure residual_cost_bits must be exactly zero"
        )

    def evaluate(selected: frozenset[str]) -> BasisEvaluation:
        if type(selected) is not frozenset or any(
            type(atom_id) is not str or not atom_id for atom_id in selected
        ):
            raise TypeError(
                "selected atoms must be an exact frozenset of non-empty strings"
            )
        receipt = evaluate_intent_closure(
            fact_set=fact_set,
            source_inventory=source_inventory,
            entailments=entailments,
            selected_atom_ids=tuple(sorted(selected)),
            projections=projections,
            maximum_stage=maximum_stage,
        )
        return BasisEvaluation(
            accepted=receipt.accepted,
            failed_gate=receipt.first_blocker,
            evidence_sha256=receipt.sha256,
            residual_cost_bits=0,
            closure_receipt_sha256=receipt.sha256,
            oracle_max_stage=receipt.maximum_stage,
        )

    return evaluate


__all__ = [
    "BasisCounterexample",
    "BasisEvaluation",
    "BasisSearchResult",
    "closure_basis_evaluator",
    "search_minimal_basis",
]
