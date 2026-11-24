"""The faithful-router CONTRACT (interface), plus its I/O types.

The engine is a STRATEGY behind this interface; the CONTRACT below is the invariant that
holds regardless of which algorithm (multi-layer A* / ILP / ...) implements it. This is
what lets us swap or add routing algorithms without touching intent, constraints, or
verification -- and what stops the engine from accreting per-circuit special cases (any
such case goes behind a new RouteEngine strategy, never an inline branch).

CONTRACT -- ``route(pins_by_net, obstacles, constraints, pdk, *, preroutes=None,
resource_plan=None) -> RoutedCell`` guarantees:
  1. Connectivity: every net in ``pins_by_net`` is drawn as ONE connected component
     joining all its pins (the LVS precondition).
  2. Hard constraints: every constraint in ``constraints`` is SATISFIED, or it appears in
     ``RoutedCell.unsatisfied`` and ``success`` is False. The engine NEVER silently
     relaxes a constraint to claim success (cf. dcsolve reporting conflicts).
  3. DRC-legal: emitted geometry obeys the PDK rules.  Obstacles are layer-local:
     different-layer metal may pass over device artwork, while same-layer wire and
     via landing pads honour net-aware ownership (only the owning net may touch an
     owned terminal; unowned/foreign device metal remains blocked) -- by
     construction, not by hope.
  4. Secondary objectives (short wire, few vias) optimise ONLY the degrees of freedom the
     constraints leave free (don't-care); they never override a hard constraint.
  5. Honest failure: if a net cannot be connected or constraints conflict, that is
     reported (``unsatisfied`` / ``nets[*].connected=False`` / ``shorts``), never faked.
  6. PREROUTE (same-net seed geometry -- the production-router power-routing
     mechanism): ``preroutes`` maps ``net -> list[LayerPolygon]`` (nm-int rects, the
     emit currency) of metal that ALREADY EXISTS on that net. Semantics:
       a. the preroute rasterises into the net's INITIAL route tree -- a pin that
          reaches the preroute-containing tree is connected;
       b. every OTHER net treats the preroute as an OWNED hard keep-out (the exact
          owner semantics of device terminal pads: foreign nets hold their
          width-aware min-space halo, the owning net may land on it);
       c. the preroute rects are merged VERBATIM into the net's emitted ``wires``
          (through the same per-net GDS merge as routed metal);
       d. the connectivity / short oracles count the preroute as the owning net's
          metal (a foreign overlap surfaces in ``shorts``, never silently merged).
     A preroute on a layer OUTSIDE the routing graph (unknown, or a thick/top
     metal the fine grid does not route) raises ValueError -- rejected explicitly,
     never silently dropped. ``preroutes=None`` (the default) is byte-identical to
     the pre-preroute engine. The engine never INVENTS preroute geometry; content
     (incl. the internal connectedness of multi-rect preroutes) is the caller's.
  7. RESOURCE PLAN (compiler-selected complete-net allocation): ``resource_plan``
     is additive and defaults to ``None``. An engine may consume a non-empty plan
     only when it declares ``consumes_resource_plan = True`` and materializes the
     selected topology/tracks/landings. A non-consuming engine raises the typed
     :class:`RouteResourcePlanUnsupportedError`; it may never silently ignore a
     selected plan. A consuming engine still raises the typed
     :class:`RouteResourcePlanIncompleteError` unless every selected net has one
     byte-complete allocated materialization program. ``None`` preserves the
     historical call byte-for-byte.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from clay.layout.route.constraints import Layer, RouteConstraint
from clay.layout.route.terminal_identity import PhysicalTerminalPinBinding, pin_binding_payload

_XY = Tuple[float, float]


@dataclass(frozen=True)
class Pin:
    """One terminal pin to be connected. ``xy_um`` is the ABSOLUTE pin centre (device's
    placed centroid + the terminal's centroid-relative offset). ``xy_nm`` optionally
    carries the exact integer-nm source point when one exists, avoiding a lossy
    nm->float-um->nm round trip. ``layer`` is the pin's access metal layer."""
    net: str
    device: str
    terminal: str
    xy_um: _XY
    layer: Layer
    xy_nm: Optional[Tuple[int, int]] = None
    terminal_binding: Optional[PhysicalTerminalPinBinding] = None
    # A compiler-created terminal without native metal must be reached by an
    # actual wire centerline, not merely covered by an offset escape-via pad.
    requires_wire_endpoint: bool = False

    def __post_init__(self) -> None:
        if type(self.requires_wire_endpoint) is not bool:
            raise TypeError("Pin.requires_wire_endpoint must be an exact bool")
        if self.requires_wire_endpoint and self.xy_nm is None:
            raise ValueError("A required wire endpoint needs an exact integer-nm point")
        if self.terminal_binding is not None:
            pin_binding_payload(self)
        if self.xy_nm is None:
            return
        if (
            len(self.xy_nm) != 2
            or any(type(value) is not int for value in self.xy_nm)
        ):
            raise ValueError("Pin.xy_nm must be an exact integer-nm point")
        round_trip = tuple(
            int(round(float(value) * 1000.0)) for value in self.xy_um
        )
        if round_trip != self.xy_nm:
            raise ValueError(
                "Pin.xy_nm must agree with the rounded xy_um access point"
            )


@dataclass(frozen=True)
class Obstacle:
    """A keep-out rectangle on a layer (device artwork, guard ring, committed wire). The
    engine rasterises these into per-layer blockage so routes avoid them by construction.

    ``owner_net`` makes an obstacle NET-AWARE (the pin-access model): if set, this rect
    is that net's own terminal/connection region -- it does NOT block ``owner_net`` (the
    net connects there) but DOES block every OTHER net. Left None (the default) it is a
    hard keep-out for all nets (device body, guard ring).

    ``kind`` is the layout-model provenance tag (``Conductor.kind``), threaded
    through losslessly so an engine can tell literally-DRAWN artwork from
    routing-model ABSTRACTIONS (``clay.layout.model.ROUTING_ABSTRACTION_KINDS``:
    min-space-bloated cap-plate collars / footprint via bans / no-route-under
    keep-outs). Abstractions block and price exactly like artwork, but must
    never be credited as DRAWN metal in emitted-geometry judgments (gap
    coverage / junction contact / band detection) -- model-only metal cannot
    merge anything in the artifact the deck reads. ``None`` (the default and
    every legacy constructor) means unknown provenance, treated as drawn."""
    x0: float
    y0: float
    x1: float
    y1: float
    layer: Layer
    owner_net: Optional[str] = None
    kind: Optional[str] = None
    # Compiler-owned identity of one immutable device-terminal conductor seam.
    # It is deliberately independent of bbox/electrical component identity:
    # adjacent same-net device instances may touch but remain distinct seams.
    terminal_conductor_ref: Optional[str] = None


@dataclass
class NetRoute:
    """The realized route of one net."""
    net: str
    pins: List[Pin] = field(default_factory=list)
    wires: list = field(default_factory=list)   # list[LayerPolygon] (device_cell geometry currency)
    connected: bool = False
    length_um: float = 0.0


@dataclass
class RoutedCell:
    """The router's output -- honest by construction."""
    nets: Dict[str, NetRoute] = field(default_factory=dict)
    shorts: list = field(default_factory=list)               # detected same-layer cross-net shorts (target: [])
    unsatisfied: List[RouteConstraint] = field(default_factory=list)  # hard constraints NOT satisfied
    success: bool = False

    @property
    def wires(self) -> list:
        out: list = []
        for nr in self.nets.values():
            out.extend(nr.wires)
        return out

    @property
    def n_connected(self) -> int:
        return sum(1 for nr in self.nets.values() if nr.connected)


class RouteUnsat(Exception):
    """Raised when a route is structurally impossible and the caller asked to fail hard
    (default is to return a RoutedCell with success=False + populated ``unsatisfied``)."""


class RouteResourcePlanUnsupportedError(RuntimeError):
    """Typed 5-state stop for an engine that cannot consume a selected plan."""

    def __init__(self, engine: object, resource_plan: object) -> None:
        from clay.core_contract import VerificationStatus

        self.status = VerificationStatus.UNSUPPORTED
        self.engine_name = str(getattr(engine, "name", type(engine).__name__))
        self.resource_plan = resource_plan
        self.candidate_pool_hash = str(
            getattr(resource_plan, "candidate_pool_hash", "")
        )
        self.selected_candidate_ids = tuple(
            str(getattr(selected, "candidate_id", ""))
            for selected in (getattr(resource_plan, "selected_nets", ()) or ())
        )
        super().__init__(
            f"route engine {self.engine_name!r} does not consume non-empty "
            f"RouteResourcePlan {self.candidate_pool_hash!r} "
            "(5-state UNSUPPORTED); the selected plan was not routed"
        )


class RouteResourcePlanIncompleteError(RuntimeError):
    """A consuming engine received a selected plan without materialization IR."""

    def __init__(self, engine: object, resource_plan: object) -> None:
        from clay.core_contract import VerificationStatus

        self.status = VerificationStatus.UNMEASURABLE
        self.state = "INCOMPLETE"
        self.engine_name = str(getattr(engine, "name", type(engine).__name__))
        self.resource_plan = resource_plan
        self.candidate_pool_hash = str(
            getattr(resource_plan, "candidate_pool_hash", "")
        )
        self.selected_candidate_ids = tuple(
            str(getattr(selected, "candidate_id", ""))
            for selected in (getattr(resource_plan, "selected_nets", ()) or ())
        )
        super().__init__(
            f"route engine {self.engine_name!r} consumes resource plans, but "
            f"RouteResourcePlan {self.candidate_pool_hash!r} is INCOMPLETE; "
            "the selected plan has no byte-complete materialization program"
        )


def require_resource_plan_support(engine: object, resource_plan: object) -> None:
    """Reject a non-empty plan unless ``engine`` explicitly consumes the contract.

    The exact :class:`RouteResourcePlan` type is checked lazily to keep the base
    engine module independent of the planner at import time. ``None`` and an exact
    empty selection are inert; every other accepted plan must be materialized by an
    engine that explicitly opts in.
    """

    if resource_plan is None:
        return
    from clay.layout.route.resource_plan import RouteResourcePlan

    if not isinstance(resource_plan, RouteResourcePlan):
        raise TypeError("resource_plan must be RouteResourcePlan or None")
    if not resource_plan.selected_nets:
        return
    if not bool(getattr(engine, "consumes_resource_plan", False)):
        raise RouteResourcePlanUnsupportedError(engine, resource_plan)
    expected_programs = {
        (str(selected.net), str(selected.candidate_id))
        for selected in resource_plan.selected_nets
    }
    actual_programs = {
        (str(program.net), str(program.candidate_id))
        for program in resource_plan.allocated_programs
    }
    if (
        not resource_plan.materialization_complete
        or actual_programs != expected_programs
        or len(resource_plan.allocated_programs) != len(expected_programs)
    ):
        raise RouteResourcePlanIncompleteError(engine, resource_plan)


class RouteEngine(abc.ABC):
    """The faithful-router interface. Implementations are strategies; the CONTRACT in the
    module docstring is the invariant. NO implementation may branch on a circuit name."""

    consumes_resource_plan = False

    @abc.abstractmethod
    def route(
        self,
        pins_by_net: Dict[str, List[Pin]],
        obstacles: Sequence[Obstacle],
        constraints: Sequence[RouteConstraint],
        pdk,
        *,
        preroutes: Optional[Dict[str, list]] = None,
        resource_plan: Optional[object] = None,
    ) -> RoutedCell:
        """Connect ``pins_by_net`` avoiding ``obstacles``, satisfying ``constraints``,
        under ``pdk`` (a :class:`clay.layout.tech.LayoutTech`) rules. ``preroutes``
        (``net -> list[LayerPolygon]``, nm ints) is same-net seed geometry -- see
        contract point 6. ``resource_plan`` is a replay-verified selected complete-
        net allocation; see contract point 7. Both defaults preserve the historical
        four-argument call."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

    def satisfies(self) -> FrozenSet[str]:
        """The CONSTRAINT KINDS (``constraints.CONSTRAINT_KINDS`` members) this
        strategy can actually MATERIALIZE -- the backend half of the capability
        handshake. The default is EMPTY: a strategy claims NOTHING until it opts
        in per kind, so a subclass that forgets to declare capabilities gates
        every HARD constraint rather than silently emitting a GDS that ignores
        it. Consumed by the route_cell PRE-gate (a HARD constraint of a kind NOT
        in this set STOPS compilation with a 5-state UNSUPPORTED verdict before
        emit). Always a subset of ``CONSTRAINT_KINDS``. This is a QUERY, never a
        per-circuit branch."""
        return frozenset()
