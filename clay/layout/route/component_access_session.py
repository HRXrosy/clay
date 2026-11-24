"""Solve-local ownership of dynamic component-access query results.

The caller supplies exact world keys and completed proofs. This module knows
nothing about an engine, PDK, geometry evaluator, search policy or wall clock.
Fixed-world legality and event construction remain separate from these dynamic
working sets. Replacing the session starts a new route-world lifetime.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, Iterator, NamedTuple, TypeVar


Rect = tuple[int, int, int, int]
GeometrySignature = tuple[tuple[int, tuple[Rect, ...]], ...]
ForbiddenSites = tuple[tuple[int, int, int, int], ...]
Rejections = tuple[tuple[tuple[str, str], int], ...]
RejectionSamples = tuple[tuple[tuple[Any, ...], int, Rect], ...]
Partition = tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]]
State = TypeVar("State")


@dataclass(frozen=True)
class PartitionContext:
    committed_generation: int
    net: str
    attempt_cuts: GeometrySignature
    forbidden_sites: ForbiddenSites


class SiteLegalityKey(NamedTuple):
    committed_generation: int
    net: str
    x: int
    y: int
    lo: int
    single_model: int
    local_attempt_cuts: tuple[Rect, ...]
    forbidden: bool
    static_key: Hashable


class EscapeWorldKey(NamedTuple):
    net: str
    x: int
    y: int
    gi: int
    atoms_identity: int
    committed_generation: int
    attempt_cuts: GeometrySignature
    attempt_metal: GeometrySignature
    forbidden_sites: ForbiddenSites
    history_generation: int
    present_factor: float


class ViaLegalityProof(NamedTuple):
    legal: bool
    rejections: Rejections
    samples: RejectionSamples


class EscapeProof(NamedTuple, Generic[State]):
    states: tuple[State, ...]
    rejections: Rejections
    samples: RejectionSamples
    component_details: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class QueryStats:
    site_hits: int = 0
    site_misses: int = 0
    partition_hits: int = 0
    partition_misses: int = 0
    escape_hits: int = 0
    escape_misses: int = 0


@dataclass(eq=False)
class EscapeScope(Generic[State]):
    """Opaque working set for one signal edge or rail drop, issued by a session."""

    _owner: object = field(repr=False)
    _proofs: dict[Hashable, EscapeProof[State]] = field(
        default_factory=dict, repr=False
    )


class ComponentAccessSession(Generic[State]):
    """Own query reuse, invalidation, bounded storage and nested scope lifetime.

    The site capacity preserves the existing max-expansions-derived policy.
    A broad partition-context change does not flush local site proofs: their
    keys already bind local cuts and forbidden membership. A committed-index
    rebuild flushes both working sets, retaining solve-total diagnostic counts.
    Escape keys bind committed/history/attempt worlds independently.
    """

    def __init__(self, *, max_expansions: int) -> None:
        self._site_capacity = max(1, min(int(max_expansions), 100_000))
        self._sites: OrderedDict[Hashable, ViaLegalityProof] = OrderedDict()
        self._partition_context: PartitionContext | None = None
        self._partitions: dict[Hashable, Partition] = {}
        self._scope_owner = object()
        self._active_scope: EscapeScope[State] | None = None
        # Internal mutable counts avoid allocating a stats snapshot per landing.
        self._counts = dict.fromkeys(QueryStats.__dataclass_fields__, 0)

    @property
    def stats(self) -> QueryStats:
        """A detached, immutable diagnostic snapshot; never search authority."""
        return QueryStats(**self._counts)

    def bind_partition(self, context: PartitionContext) -> None:
        if context != self._partition_context:
            self._partitions.clear()
            self._partition_context = context

    def committed_world_changed(self) -> None:
        self._partition_context = None
        self._partitions.clear()
        self._sites.clear()

    def lookup_site(self, key: Hashable) -> ViaLegalityProof | None:
        proof = self._sites.pop(key, None)
        if proof is None:
            self._counts["site_misses"] += 1
        else:
            self._sites[key] = proof
            self._counts["site_hits"] += 1
        return proof

    def store_site(
        self, key: Hashable, proof: ViaLegalityProof, *,
        max_expansions: int | None = None,
    ) -> None:
        # The engine budget is mutable. Preserve its publication-time capacity
        # policy, including when a direct helper caller changes it mid-session.
        if max_expansions is not None:
            self._site_capacity = max(1, min(int(max_expansions), 100_000))
        self._sites[key] = proof
        while len(self._sites) > self._site_capacity:
            self._sites.popitem(last=False)

    def lookup_partition(self, key: Hashable) -> Partition | None:
        partition = self._partitions.get(key)
        self._counts[
            "partition_misses" if partition is None else "partition_hits"
        ] += 1
        return partition

    def store_partition(self, key: Hashable, partition: Partition) -> None:
        self._partitions[key] = partition

    def new_escape_scope(self) -> EscapeScope[State]:
        return EscapeScope(self._scope_owner)

    @property
    def escape_scope_active(self) -> bool:
        return self._active_scope is not None

    @contextmanager
    def escape_scope(
        self, scope: EscapeScope[State] | None = None,
    ) -> Iterator[None]:
        """Activate a fresh drop scope, or reenter one edge's existing scope.

        Nested scopes restore their parent on every exit, including watchdogs.
        A scope cannot be reused after route ingestion replaces its session.
        """
        if scope is None:
            scope = self.new_escape_scope()
        if scope._owner is not self._scope_owner:
            raise ValueError("escape scope belongs to a different query session")
        previous = self._active_scope
        self._active_scope = scope
        try:
            yield
        finally:
            self._active_scope = previous

    def lookup_escape(self, key: Hashable) -> EscapeProof[State] | None:
        if self._active_scope is None:
            return None
        proof = self._active_scope._proofs.get(key)
        if proof is not None:
            self._counts["escape_hits"] += 1
        return proof

    def store_escape(self, key: Hashable, proof: EscapeProof[State]) -> None:
        if self._active_scope is None:
            raise RuntimeError("escape proof requires an active query scope")
        self._active_scope._proofs[key] = proof
        # Preserve historical semantics: only completed escape computations count
        # as misses. Site/partition counters instead count lookup attempts.
        self._counts["escape_misses"] += 1
