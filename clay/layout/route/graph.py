"""RoutingConstraintGraph -- the CLOSED routing input object (spec §4.3, Wave B2).

Today ``route_cell`` hands the engine a FLAT BAG of positional args
(``route_engine.route(pins_by_net, obstacles, cons, tech)``). This wraps that bag in
ONE frozen, queryable object so the router consumes a closed contract, not a loose
argument list -- and so a capability handshake can ask the graph ``kinds()`` (the set of
constraint kinds present) against a backend's ``satisfies()`` BEFORE routing.

It is an ADAPTER, not a second path: :meth:`from_route_inputs` builds it from the exact
existing inputs and :meth:`route_inputs` returns them verbatim, so the engine call is
byte-identical (a parity test asserts round-trip equality).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet, Mapping, Tuple

__all__ = ["RoutingConstraintGraph"]


@dataclass(frozen=True)
class RoutingConstraintGraph:
    """The closed routing input: mandatory connectivity (``pins_by_net``) + the
    declarative OVERLAY (``constraints``, from :mod:`clay.layout.route.constraints`) +
    the ``obstacles`` / optional ``preroutes`` / ``rails``. Frozen + hashable-ish
    (tuples), so it can be logged / compared / handshaked deterministically."""

    pins_by_net: Mapping[str, Any]
    constraints: Tuple[Any, ...] = ()
    obstacles: Tuple[Any, ...] = ()
    preroutes: Mapping[str, Any] = field(default_factory=dict)
    rails: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_route_inputs(
        cls,
        pins_by_net: Mapping[str, Any],
        obstacles: Any = (),
        constraints: Any = (),
        *,
        preroutes: Mapping[str, Any] | None = None,
        rails: Mapping[str, Any] | None = None,
    ) -> "RoutingConstraintGraph":
        """Build from the exact positional inputs ``route_cell`` passes the engine."""
        return cls(
            pins_by_net=pins_by_net,
            constraints=tuple(constraints or ()),
            obstacles=tuple(obstacles or ()),
            preroutes=dict(preroutes or {}),
            rails=dict(rails or {}),
        )

    def route_inputs(self) -> Tuple[Mapping[str, Any], Tuple[Any, ...], Tuple[Any, ...]]:
        """The ``(pins_by_net, obstacles, constraints)`` triple the engine consumes --
        returned verbatim so ``engine.route(*graph.route_inputs(), tech)`` is
        byte-identical to the pre-graph call."""
        return self.pins_by_net, self.obstacles, self.constraints

    def kinds(self) -> FrozenSet[str]:
        """The set of routing-constraint KINDS present (for the capability handshake vs
        a backend's ``satisfies()``)."""
        return frozenset(str(getattr(c, "kind", "?")) for c in self.constraints)

    def hard_kinds(self) -> FrozenSet[str]:
        """The KINDS of the HARD constraints present (the ones the handshake gates on)."""
        from clay.layout.route.constraints import is_hard

        return frozenset(
            str(getattr(c, "kind", "?")) for c in self.constraints if is_hard(c)
        )
