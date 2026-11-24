"""Queryable PLACEMENT-engine capability -- the placement dual of
:class:`clay.layout.route.engine.RouteEngine` (spec Wave B3).

Before this, placement capability was NOT queryable: an @anchor / @channel / @rail_affinity
that the CP spine could not honour was SILENTLY DROPPED. This module makes the CP spine's
capability honest and QUERYABLE (``satisfies() -> frozenset[kind]``), so the placement
PRE-handshake surfaces a REPORTED UNSUPPORTED (5-state) for a placement kind the backend
does not satisfy -- exactly the routing-side handshake, on the placement side.

The CP envelope-min spine (``synth.placement.place_envelope_min``) honestly satisfies
``{rigid_macro, relative_order, abutment, symmetry, region_affinity, keep_together,
alignment}`` and NOT ``{edge_affinity, channel, keepout, spacing}`` -- so @anchor (edge_affinity)
/ @channel become reported UNSUPPORTED rather than silently dropped. Wave B4 opts the
edge-affinity capability in (``extra_kinds``) once the rail-affinity lowering materialises
it -- a flag-gated extension, byte-identical while OFF.
"""
from __future__ import annotations

import abc
from typing import Any, FrozenSet, List, Sequence

from clay.layout.place.constraints import PLACEMENT_CONSTRAINT_KINDS, is_hard
from clay.signoff_verdict import CheckerVerdict, is_blocking

__all__ = [
    "PlacementEngine",
    "CPEnvelopePlacementEngine",
    "cp_placement_backend",
    "preplace_capability_gate",
    "PlacementUnsupportedError",
]

# the kinds the CP envelope-min placer TRULY enforces today (spec B3). A superset of the
# closed set would be a lie; a subset is the honest capability that surfaces the gaps.
_CP_SATISFIES: FrozenSet[str] = frozenset(
    {
        "rigid_macro",
        "relative_order",
        "abutment",
        "symmetry",
        "region_affinity",
        "keep_together",
        "alignment",
    }
)


class PlacementEngine(abc.ABC):
    """A placement backend's queryable capability face. ``satisfies()`` returns the
    CLOSED placement kinds this backend ENFORCES (never merely records) -- always a
    subset of :data:`PLACEMENT_CONSTRAINT_KINDS`. A QUERY, never a mutation."""

    name: str = "placement_engine"

    @abc.abstractmethod
    def satisfies(self) -> FrozenSet[str]:  # pragma: no cover - abstract
        ...


class CPEnvelopePlacementEngine(PlacementEngine):
    """The CP envelope-min spine placer's capability (``place_envelope_min`` /
    ``cp_group_placement_pos``). Honestly enforces the structural 6; ``extra_kinds``
    lets Wave B4 opt in ``edge_affinity`` once the rail-affinity term is wired
    (flag-gated -- OFF by default keeps every existing capability report byte-stable)."""

    name = "cp_envelope"

    def __init__(self, *, extra_kinds: Sequence[str] = ()) -> None:
        extra = frozenset(extra_kinds)
        bad = extra - PLACEMENT_CONSTRAINT_KINDS
        if bad:
            raise ValueError(
                f"extra_kinds {sorted(bad)} are not in the closed placement set "
                f"{sorted(PLACEMENT_CONSTRAINT_KINDS)}"
            )
        self._extra = extra

    def satisfies(self) -> FrozenSet[str]:
        return _CP_SATISFIES | self._extra


def cp_placement_backend(*, edge_affinity: bool = False) -> CPEnvelopePlacementEngine:
    """The CP spine placement backend. ``edge_affinity=True`` (Wave B4, flag-gated) adds
    the ``edge_affinity`` capability once the rail-affinity lowering materialises it."""
    return CPEnvelopePlacementEngine(
        extra_kinds=("edge_affinity",) if edge_affinity else ()
    )


class PlacementUnsupportedError(RuntimeError):
    """Typed 5-state stop: the placement backend cannot satisfy a HARD/required
    placement kind declared by an intent. Mirrors ``route_cell.RouteUnsupportedError``
    -- the placer must not silently drop a required structural request."""

    def __init__(self, backend: Any, verdicts: Sequence[CheckerVerdict]) -> None:
        from clay.core_contract import VerificationStatus

        self.status = VerificationStatus.UNSUPPORTED
        self.backend_name = str(getattr(backend, "name", type(backend).__name__))
        self.verdicts = list(verdicts)
        kinds = sorted({v.evidence.get("kind", "?") for v in self.verdicts})
        self.unsupported_kinds = kinds
        super().__init__(
            f"placement capability handshake: backend {self.backend_name!r} does NOT "
            f"satisfy required placement kind(s) for intent(s) {kinds} -- stopping "
            "before place (5-state UNSUPPORTED), never silently dropping them."
        )


def preplace_capability_gate(
    intent: Any, backend: PlacementEngine, *, raise_on_block: bool = False
) -> List[CheckerVerdict]:
    """PRE-handshake on the placement side: for every declared CRAFT intent, ask its
    registered implementation's ``capabilities(decl, backend)`` whether ``backend`` can
    satisfy the generic kinds it lowers to. Returns the list of NON-PASS verdicts
    (reported UNSUPPORTED / advisory), so a caller renders one honest 5-state report
    rather than the placer silently dropping @anchor / @channel / @rail_affinity.

    Feeds the registry's ``capabilities()`` INTO the capability handshake (the Wave A
    invariant). ``raise_on_block=True`` raises :class:`PlacementUnsupportedError` when
    any verdict BLOCKS (a required UNSUPPORTED) -- else it is report-only."""
    from clay.layout.intent_registry import get
    import clay.layout.intent_impls_craft  # noqa: F401  (ensure registration)

    verdicts: List[CheckerVerdict] = []
    for kind in sorted(_craft_fields()):
        decls = getattr(intent, kind, ()) or ()
        if not decls:
            continue
        impl = get(kind)
        if impl is None:
            continue
        for decl in decls:
            v = impl.capabilities(decl, backend)
            verdicts.append(v)

    blocking = [v for v in verdicts if is_blocking(v)]
    if raise_on_block and blocking:
        raise PlacementUnsupportedError(backend, blocking)
    return [v for v in verdicts if v.status.value != "pass"]


def _craft_fields() -> FrozenSet[str]:
    """The declarative craft fields the registry keys on (derived, never hand-listed)."""
    from clay.layout.intent_impls_craft import CRAFT_IMPLS

    return frozenset(impl.kind for impl in CRAFT_IMPLS)
