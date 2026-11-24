"""Shared data structures, geometry primitives, and negotiation types for GridlessEngine."""
from __future__ import annotations

import hashlib
import heapq
import json
import sys
import weakref
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from clay._native import load_native
from clay.layout.backend import LayerPolygon, Polygon
from clay.layout.route.constraints import (
    Access,
    Direction,
    LayerSet,
    Length,
    Mirror,
    Shield,
    Spacing,
    TrackOrder,
    TrunkLayer,
    TrunkLayerDeclarationError,
    Via,
    Width,
)
from clay.layout.route.engine import Obstacle, Pin
try:
    from clay.layout.route.engine_maze import _NM_PER_UM
except ImportError:
    _NM_PER_UM = 1000
from clay.layout.route.terminal_identity import (
    pin_fact as canonical_pin_fact,
    pin_provenance,
)


def _eg_attr(name: str, default: Any) -> Any:
    """Resolve module-level attributes via engine_gridless facade so monkeypatches apply."""
    mod = sys.modules.get("clay.layout.route.engine_gridless")
    return getattr(mod, name, default) if mod is not None else default

Rect = Tuple[int, int, int, int]


_GRIDLESS_REPLAY_ISSUER_AUTHORITY = object()
_GRIDLESS_REPLAY_RECEIPTS: Dict[
    int, Tuple[weakref.ReferenceType[Any], str]
] = {}
_MAX_GRIDLESS_REPLAY_SELECTED_CANDIDATES = 4096


def _allocation_geometry_digest(
    geometry_records: Sequence[Tuple[str, int, int, Rect]],
) -> str:
    payload = [
        [net, layer, datatype, list(bbox)]
        for net, layer, datatype, bbox in geometry_records
    ]
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _gridless_replay_receipt_payload(
    certificate: "GridlessAllocationReplayCertificate",
) -> str:
    payload = {
        "candidate_pool_hash": certificate.candidate_pool_hash,
        "current_pdk_provenance_digest": (
            certificate.current_pdk_provenance_digest
        ),
        "geometry_digest": certificate.geometry_digest,
        "materialization_complete": certificate.materialization_complete,
        "resource_plan_hash": certificate.resource_plan_hash,
        "routing_world_provenance_digests": list(
            certificate.routing_world_provenance_digests
        ),
        "selected_candidates": [
            list(item) for item in certificate.selected_candidates
        ],
        "verification_scope": certificate.verification_scope,
    }
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _is_lower_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


@dataclass(frozen=True, init=False)
class GridlessAllocationReplayCertificate:
    """Exact provenance for the allocation program materialized by Gridless.

    The plan hash commits to every selected track, landing site, owned-component
    reference, and allocation-program field.  ``geometry_digest`` independently
    commits to the emitted layer rectangles.  This is a deterministic replay
    certificate, not an LVS/connectivity verdict.
    """

    candidate_pool_hash: str
    resource_plan_hash: str
    selected_candidates: Tuple[Tuple[str, str], ...]
    geometry_digest: str
    materialization_complete: bool
    current_pdk_provenance_digest: str
    routing_world_provenance_digests: Tuple[str, ...]
    verification_scope: str
    _issuer_authority: object = field(repr=False, compare=False, init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "GridlessAllocationReplayCertificate is a module-issued capability"
        )


def _verify_gridless_allocation_replay_certificate(
    certificate: GridlessAllocationReplayCertificate,
) -> None:
    """Validate one live engine-issued runtime-materialization capability."""

    if type(certificate) is not GridlessAllocationReplayCertificate:
        raise TypeError(
            "materialization certificate must be GridlessAllocationReplayCertificate"
        )
    if (
        getattr(certificate, "_issuer_authority", None)
        is not _GRIDLESS_REPLAY_ISSUER_AUTHORITY
    ):
        raise TypeError(
            "materialization certificate must be a module-issued capability"
        )
    for name in (
        "candidate_pool_hash",
        "resource_plan_hash",
        "geometry_digest",
        "current_pdk_provenance_digest",
    ):
        if not _is_lower_sha256(getattr(certificate, name, None)):
            raise TypeError(f"materialization certificate {name} is malformed")
    selected = getattr(certificate, "selected_candidates", None)
    if (
        type(selected) is not tuple
        or len(selected) > _MAX_GRIDLESS_REPLAY_SELECTED_CANDIDATES
        or any(
            type(item) is not tuple
            or len(item) != 2
            or any(type(value) is not str or not value for value in item)
            for item in selected
        )
        or len(selected) != len(set(selected))
    ):
        raise TypeError(
            "materialization certificate selected candidates are malformed"
        )
    worlds = getattr(certificate, "routing_world_provenance_digests", None)
    if (
        type(worlds) is not tuple
        or len(worlds) > _MAX_GRIDLESS_REPLAY_SELECTED_CANDIDATES
        or any(not _is_lower_sha256(value) for value in worlds)
        or worlds != tuple(sorted(set(worlds)))
    ):
        raise TypeError(
            "materialization certificate routing-world digests are malformed"
        )
    if type(certificate.materialization_complete) is not bool:
        raise TypeError(
            "materialization certificate completion state is malformed"
        )
    if (
        type(certificate.verification_scope) is not str
        or not certificate.verification_scope.strip()
    ):
        raise TypeError("materialization certificate scope is malformed")
    issued = _GRIDLESS_REPLAY_RECEIPTS.get(id(certificate))
    if (
        issued is None
        or issued[0]() is not certificate
        or issued[1] != _gridless_replay_receipt_payload(certificate)
    ):
        raise TypeError(
            "materialization certificate is not an intact module-issued capability"
        )


def _via_cut_space_nm(contract, via_name: Optional[str], fallback_nm: int) -> int:
    """Cut-to-cut spacing (nm) for ``via_name`` from the SAME PDK rule source the
    deck's VIA{k}.S.1 check consumes: the parsed ``via{k}_space`` deck rule when
    the PDK exposes parsed rules (external_pdk/external_pdk synthesized decks), else the foundry
    ``via_rule`` spacing (gf180 real-deck path), else the caller's fallback (the
    lower metal's min-space -- the historical PDK-value-free model, kept only for
    a PDK that declares no cut spacing at all). No constants of its own."""
    if via_name:
        try:
            parsed = contract.drc_rules() or {}
        except Exception:
            parsed = {}
        e = parsed.get(f"{via_name}_space")
        if isinstance(e, dict):
            v = e.get("value_nm")
            if v is None and e.get("value_um") is not None:
                v = round(float(e["value_um"]) * 1000.0)
            if v:
                return int(v)
        try:
            vr = contract.via_rule(via_name)
        except Exception:
            vr = None
        sp = getattr(vr, "spacing", None)
        if sp is None:
            sp = getattr(vr, "spacing_nm", None)
        if sp is not None:
            v = getattr(sp, "value", sp)
            if v:
                return int(v)
    return int(fallback_nm)

# Constraint kinds this strategy SATISFIES; anything else is recorded unsatisfied
# (honest P1 reporting), but the net is still routed for connectivity.
_HANDLED_KINDS = frozenset({"layer_set", "width", "mirror", "length", "spacing"})
_CONSTRAINT_TYPES = (
    LayerSet, Width, Mirror, Spacing, Shield, Length, TrackOrder, Access,
    Direction, TrunkLayer, Via,
)


class TrunkLayerResolutionError(TrunkLayerDeclarationError):
    """A trunk advisory names a layer absent from the active routing stack."""

# Search cost weights (relative; a per-nm STEP keeps the Dijkstra cost ~ wirelength).
_STEP_COST_PER_NM = 1
# A corner is priced in ROUTING PITCHES (min_width + min_space), NOT a fixed nm
# constant -- mirroring the pitch-scaled ``_TRUNK_SPAN_PITCHES`` / ``_CONG_WEIGHT``
# idioms below so the corner price stays commensurate with the pitch-scaled
# ``step`` / ``cong`` terms across PDKs and cell sizes. A fixed 6000nm charge was
# outbid by congestion on a ~106um cell (a single base-round congestion cell
# already costs ``pres_fac0 * _CONG_WEIGHT`` == 8 pitch-widths), so the A* bought
# gratuitous jogs to dodge committed metal. At 40 pitches a corner dominates ~5
# base-round congestion cells (straightness wins small congestion ties) yet stays
# BELOW one layer escalation ``_VIA_COST`` (a turn must never cost more than a via,
# or the router would via instead of turning). The per-corner cost is
# ``_TURN_COST_PITCHES * self._res_cell`` (nm), computed per solve from the tech
# pitch (``_turn_cost``); general, keyed to no circuit and no reference number.
_TURN_COST_PITCHES = 40
_OFF_PREF_COST_PER_NM = 1      # extra per-nm against a layer's preferred direction
_VIA_COST = 40000             # one adjacent-layer escalation (drop a via stack)
# One exact-goal index visit evaluates a 3-axis subtree bound, a 3-axis point
# distance, and up to two 3-axis child bounds.  The separable transform work model
# counts one scalar relaxation, so compare like operations rather than treating
# one branch-and-bound node as one relaxation.
_EXACT_GOAL_INDEX_VISIT_WORK = 12
# M1 routing is heavily penalised: a net climbs to M2+ at its pin and routes OVER the
# cell. (Per-nm so it scales with any attempted M1 run.) M1 stays pin-access only.
_M1_ROUTE_COST_PER_NM = 50

# --- wireless via escape (FIX: dense-pack pin-access) --------------------------
# A dense pack routinely blocks the M1->M2 escape via at the pin CENTRE, and the
# pin can emit NO wire at all when it sits within min_space of a foreign finger
# (the wire's own half-width pokes into the gap). The escape is an OFFSET via whose
# M1 landing pad clears the foreign metal while STILL COVERING the pin centre --
# it overlaps the pin's own metal, so the via connects with no wire leg. General:
# derived from the via clearance geometry, no per-cell / per-PDK constant (see
# ``_escape_landing``).

# --- deliberate trunk-layer climb (FIX: long output net over the cell) ---------
# Ported generally from PathFinderEngine's trunk model: a net's TRUNK metal rises
# with its pin-bbox span, so a net spanning the cell (a compact pack's output /
# power trunk) climbs to an upper metal instead of contending on M1/M2 over the
# output stage. The span-per-level is measured in ROUTING PITCHES (min_width +
# min_space) so it scales with any PDK, not a hard nm constant.
_TRUNK_SPAN_PITCHES = 90        # pin-bbox span (in pitches) that promotes one metal
_TRUNK_BELOW_COST_PER_NM = 4  # extra per-nm, per metal-level a net runs BELOW its trunk

# --- negotiated congestion (PathFinder) ----------------------------------------
# Foreign COMMITTED ROUTES have two distinct roles: exact closed-rectangle contact
# is HARD-illegal in the search graph, while their positive min-space band remains
# SOFT so rip-up/history can negotiate channels. Fixed device/rail/guard artwork
# remains hard with its full deck clearance. The congestion of an edge is measured
# over a coarse RESOURCE grid (cell = one routing pitch): for
# every FOREIGN metal band the edge overlaps, PRESENT congestion (pres_fac per shared
# resource cell) plus accumulated HISTORY (chronic contention) is added. Scaled by
# the pitch * per-nm step so it is commensurate with wirelength and PDK-scale-free --
# a legal min-space-clean route (edges exactly min_space apart -> bands just touch)
# pays ZERO. Uniform across nets: NO per-net / per-circuit branch.
_CONG_WEIGHT = 4                # relative weight of one shared-resource-cell of congestion
_COST_FACTOR_SCALE = 1000       # fixed-point scale for declared price multipliers


@dataclass(frozen=True)
class _ExactGoalPotentialNode:
    """One deterministic node in a weighted-Manhattan target-component index."""

    point: Tuple[int, int, int]
    terminal_cost: int
    bounds: Tuple[int, int, int, int, int, int]
    minimum_terminal_cost: int
    left: Optional["_ExactGoalPotentialNode"] = None
    right: Optional["_ExactGoalPotentialNode"] = None


class _ExactGoalPotentialIndex:
    """Exact nearest-target potential without a Cartesian-grid materialization.

    The target access domain is a physical component represented by point
    alternatives.  Index those alternatives directly, as the reference
    analytical router does, instead of scanning every goal on each heap update
    or allocating an ``x*y*layer`` distance table.  The branch-and-bound lower
    bound combines weighted-L1 distance to a subtree box with that subtree's
    cheapest terminal-via cost, so ``value`` is bit-for-bit the historical
    ``min(_heuristic + terminal_cost)`` result, not an approximation.
    """

    def __init__(
        self,
        terminal_costs: Mapping[Tuple[int, int, int], int],
        *,
        watchdog: Optional[Any] = None,
        step_cost: int = _STEP_COST_PER_NM,
    ) -> None:
        if not terminal_costs:
            raise ValueError("goal potential index requires at least one target")
        self._watchdog = watchdog
        self._step_cost = step_cost
        self._work = 0
        items = tuple(
            sorted(
                (
                    int(goal[0]) * self._step_cost,
                    int(goal[1]) * self._step_cost,
                    int(goal[2]) * _VIA_COST,
                    int(terminal_cost),
                )
                for goal, terminal_cost in terminal_costs.items()
            )
        )
        self._root = self._build(items)
        assert self._root is not None

    def _check(self) -> None:
        if self._work % 4096 == 0 and self._watchdog is not None:
            self._watchdog()
        self._work += 1

    @property
    def work(self) -> int:
        """Deterministic index-node visits, including construction and queries."""
        return int(self._work)

    def _build(
        self,
        items: Sequence[Tuple[int, int, int, int]],
    ) -> Optional[_ExactGoalPotentialNode]:
        if not items:
            return None
        self._check()
        ranges = tuple(
            max(item[axis] for item in items)
            - min(item[axis] for item in items)
            for axis in range(3)
        )
        axis = max(range(3), key=lambda candidate: (ranges[candidate], -candidate))
        ordered = tuple(sorted(
            items,
            key=lambda item: (
                item[axis], item[0], item[1], item[2], item[3]
            ),
        ))
        middle = len(ordered) // 2
        item = ordered[middle]
        left = self._build(ordered[:middle])
        right = self._build(ordered[middle + 1:])
        children = tuple(child for child in (left, right) if child is not None)
        x0 = min((item[0], *(child.bounds[0] for child in children)))
        y0 = min((item[1], *(child.bounds[1] for child in children)))
        z0 = min((item[2], *(child.bounds[2] for child in children)))
        x1 = max((item[0], *(child.bounds[3] for child in children)))
        y1 = max((item[1], *(child.bounds[4] for child in children)))
        z1 = max((item[2], *(child.bounds[5] for child in children)))
        minimum_terminal_cost = min((
            item[3],
            *(child.minimum_terminal_cost for child in children),
        ))
        return _ExactGoalPotentialNode(
            point=(item[0], item[1], item[2]),
            terminal_cost=item[3],
            bounds=(x0, y0, z0, x1, y1, z1),
            minimum_terminal_cost=minimum_terminal_cost,
            left=left,
            right=right,
        )

    @staticmethod
    def _box_distance(
        point: Tuple[int, int, int],
        bounds: Tuple[int, int, int, int, int, int],
    ) -> int:
        total = 0
        for axis in range(3):
            value = point[axis]
            lo = bounds[axis]
            hi = bounds[axis + 3]
            if value < lo:
                total += lo - value
            elif value > hi:
                total += value - hi
        return total

    def value(self, x: int, y: int, gi: int) -> int:
        query = (
            int(x) * self._step_cost,
            int(y) * self._step_cost,
            int(gi) * _VIA_COST,
        )
        best: Optional[int] = None
        stack = [self._root]
        while stack:
            node = stack.pop()
            self._check()
            lower_bound = (
                self._box_distance(query, node.bounds)
                + node.minimum_terminal_cost
            )
            if best is not None and lower_bound >= best:
                continue
            candidate = (
                sum(abs(query[axis] - node.point[axis]) for axis in range(3))
                + node.terminal_cost
            )
            if best is None or candidate < best:
                best = candidate
            children = []
            for order, child in enumerate((node.left, node.right)):
                if child is None:
                    continue
                child_bound = (
                    self._box_distance(query, child.bounds)
                    + child.minimum_terminal_cost
                )
                if best is None or child_bound < best:
                    children.append((child_bound, order, child))
            # LIFO: push the weaker child first so the tighter bound is visited
            # first. Stable child order makes equal-bound traversal deterministic.
            for _bound, _order, child in sorted(children, reverse=True):
                stack.append(child)
        assert best is not None
        return best


def _norm(rc: Rect) -> Rect:
    a, b, c, d = rc
    lo_x, hi_x = (a, c) if a <= c else (c, a)
    lo_y, hi_y = (b, d) if b <= d else (d, b)
    return (lo_x, lo_y, hi_x, hi_y)


def _pin_xy_nm(pin: Pin) -> Tuple[int, int]:
    """Return a pin's exact source point when available."""

    if pin.xy_nm is not None:
        return pin.xy_nm
    scale = _eg_attr("_NM_PER_UM", _NM_PER_UM)
    return (
        int(round(pin.xy_um[0] * scale)),
        int(round(pin.xy_um[1] * scale)),
    )


def _pin_order_key(pin: Pin) -> Tuple[Any, ...]:
    """Stable total order for negotiation inputs, independent of caller order."""

    x_nm, y_nm = _pin_xy_nm(pin)
    layer = tuple(pin.layer) if isinstance(pin.layer, tuple) else (pin.layer,)
    layer_key: Tuple[Any, ...]
    if all(type(value) is int for value in layer):
        layer_key = (0, tuple(layer))
    else:
        layer_key = (1, tuple(str(value) for value in layer))
    return (
        x_nm,
        y_nm,
        layer_key,
        str(pin.device),
        str(pin.terminal),
        str(pin.net),
    )


def _group_route_pin_key(net: str, pin: object) -> Tuple[object, ...]:
    """Join group accesses to the exact runtime component and its source ledger."""
    binding = getattr(pin, "terminal_binding", None)
    return (net, pin_provenance(pin), tuple(pin.xy_nm),
            tuple(getattr(pin, "layer", getattr(pin, "layer_key", ()))),
            () if binding is None else binding.canonical_tuple())


def _obstacle_order_key(obstacle: Obstacle) -> Tuple[Any, ...]:
    """Stable total order for fixed geometry and its ownership provenance."""
    layer = (
        tuple(obstacle.layer)
        if isinstance(obstacle.layer, tuple)
        else (obstacle.layer,)
    )
    layer_key = (
        (0, tuple(layer))
        if all(type(value) is int for value in layer)
        else (1, tuple(str(value) for value in layer))
    )
    scale = _eg_attr("_NM_PER_UM", _NM_PER_UM)
    rect = _norm((
        int(round(obstacle.x0 * scale)),
        int(round(obstacle.y0 * scale)),
        int(round(obstacle.x1 * scale)),
        int(round(obstacle.y1 * scale)),
    ))
    return (
        layer_key,
        rect,
        "" if obstacle.owner_net is None else str(obstacle.owner_net),
        "" if obstacle.kind is None else str(obstacle.kind),
        (
            ""
            if obstacle.terminal_conductor_ref is None
            else str(obstacle.terminal_conductor_ref)
        ),
    )


def _negotiation_net_order(
    net_ids: Sequence[str],
    *,
    priority_nets: Set[str],
    pin_count_by_net: Mapping[str, int],
    terminal_atom_count_by_net: Optional[Mapping[str, int]] = None,
    pin_span_by_net: Optional[Mapping[str, int]] = None,
    fixed_endpoint_nets: Optional[Set[str]] = None,
) -> Tuple[str, ...]:
    """Fixed wire endpoints, priority tier, access complexity, fanout and span.

    Exact owned terminal atoms drive component-access event enumeration, so nets
    with more of them receive the emptier routing world before schematic fanout
    is considered.  A mandatory physical wire endpoint has no covering-via or
    alternate-terminal freedom, so it precedes movable rail access routing.
    Fixed rail artwork remains in the hard obstacle world throughout; this
    orders only the new routed wires and does not move or relax a declared rail.
    Among nets in the same endpoint tier, declared rails retain their priority.
    A longer
    generated-terminal span follows fanout, and net id is the deterministic final
    tie break.  No hash/insertion order or circuit/net-name special case enters
    the solve.
    """

    priority = frozenset(str(net) for net in priority_nets)
    fixed_endpoints = frozenset(str(net) for net in (fixed_endpoint_nets or ()))
    counts = {
        str(net): max(0, int(count))
        for net, count in pin_count_by_net.items()
    }
    terminal_atoms = {
        str(net): max(0, int(count))
        for net, count in (terminal_atom_count_by_net or {}).items()
    }
    spans = {
        str(net): max(0, int(span))
        for net, span in (pin_span_by_net or {}).items()
    }
    return tuple(sorted(
        {str(net) for net in net_ids},
        key=lambda net: (
            0 if net in fixed_endpoints else 1,
            0 if net in priority else 1,
            -terminal_atoms.get(net, 0),
            -counts.get(net, 0),
            -spans.get(net, 0),
            net,
        ),
    ))


def _topology_terminal_count(pins: Sequence[Pin]) -> int:
    """Count physical components, retaining legacy terminal identity when untyped."""

    return len({pin_provenance(pin) for pin in pins})


def _topology_terminal_span_nm(pins: Sequence[Pin]) -> int:
    """Manhattan span of schematic terminal-component centres in generated layout."""

    points_by_terminal: Dict[str, List[Tuple[int, int]]] = {}
    for pin in pins:
        points_by_terminal.setdefault(
            pin_provenance(pin), []
        ).append(_pin_xy_nm(pin))
    centres = tuple(
        (
            (min(x for x, _y in points) + max(x for x, _y in points)) // 2,
            (min(y for _x, y in points) + max(y for _x, y in points)) // 2,
        )
        for _terminal, points in sorted(points_by_terminal.items())
    )
    if len(centres) < 2:
        return 0
    return (
        max(x for x, _y in centres)
        - min(x for x, _y in centres)
        + max(y for _x, y in centres)
        - min(y for _x, y in centres)
    )


def _deck_open_route_nets(
    report: "DeckIsolationReport",
    ordered: Sequence[str],
) -> Tuple[str, ...]:
    """Map deck split/unresolved ownership claims back to route-net names.

    ``cross_net_isolated`` remains the sole hard short predicate.  Ambiguous
    same-net components and unresolved owned probes are instead physical OPEN
    evidence: they must participate in reroute and connectivity convergence even
    when the graph model marked the corresponding route connected.
    """

    claimed_open = {str(net).upper() for net in report.ambiguous_nets}
    claimed_open.update(
        str(net).upper()
        for net, _layer, _x, _y in report.unresolved_probes
    )
    return tuple(
        net for net in ordered if str(net).upper() in claimed_open
    )


def _negotiation_snapshot_score(
    *,
    unconnected: Sequence[str],
    shorts: Sequence[Tuple[str, str]],
    deck_report: Optional["DeckIsolationReport"] = None,
    deck_spacing_count: int,
    overused_count: int,
    advisory_spacing_count: int,
    length_violation_count: int = 0,
) -> Tuple[int, ...]:
    """Lexicographic feasibility score for a bounded negotiation snapshot.

    Deck-connected cross-net isolation is ranked before omitted geometry: an
    isolated partial route is a more truthful LVS bring-up artifact than a
    highly connected merged component. ``probe_complete`` is deliberately not
    the no-short predicate; graph opens plus deck split/unresolved owners are
    represented by ``unconnected``. Native same-layer short incidence is
    reported separately and has no score tier.
    Component severity and all tie breaks are stable under pair/input order.
    """
    _ = shorts  # Native incidence is diagnostic-only negotiation output.
    deck_component_sizes = tuple(
        len(group) for group in (
            deck_report.merged_net_groups if deck_report is not None else ()
        )
    )
    return (
        int(
            deck_report is not None
            and not deck_report.cross_net_isolated
        ),
        max(deck_component_sizes, default=0),
        sum(deck_component_sizes),
        len(deck_component_sizes),
        len(unconnected),
        int(length_violation_count),
        int(deck_spacing_count),
        int(overused_count),
        int(advisory_spacing_count),
    )


def _negotiation_conflict_victims(
    ordered: Sequence[str], owner_groups, *, protected_nets: Set[str],
) -> Set[str]:
    """Keep one stable committed owner and reroute the other conflict owners.

    During the bounded first-round spine-protection window, a declared rail/spine
    owner remains committed when present; otherwise the earliest declared owner
    is the stable obstacle.  This mirrors anaroute's DRC-driven single-net replay:
    the later owner first searches against the already committed world.  Once that
    protection window expires, a persistent ordinary conflict is the exact evidence
    needed to move both sides; otherwise the early owner would be frozen forever.
    Pre-emptively ripping both sides in round zero multiplies work and discards a
    legal incumbent before there is evidence that it must move.
    Singleton groups represent a movable net against fixed artwork or one of its
    own deck-visible residuals, so that owner remains a victim.  If two protected
    spines conflict in the protected round, declared order provides the
    deterministic tie break.
    """
    rank = {str(net): index for index, net in enumerate(ordered)}
    protected = frozenset(str(net) for net in protected_nets)
    victims: Set[str] = set()
    for group in owner_groups:
        owners = sorted(
            {str(net) for net in group if str(net) in rank},
            key=lambda net: rank[net],
        )
        if len(owners) <= 1:
            victims.update(owners)
            continue
        if not protected:
            victims.update(owners)
            continue
        protected_owners = [net for net in owners if net in protected]
        stable_owner = protected_owners[0] if protected_owners else owners[0]
        victims.update(net for net in owners if net != stable_owner)
    return victims


def _negotiation_protected_nets(
    rail_nets: Set[str], *, iteration: int,
) -> Set[str]:
    """Give declared spines one present-congestion round, not permanent immunity."""
    if int(iteration) != 0:
        return set()
    return {str(net) for net in rail_nets}


_COMMITTED_BLOCKER_REJECTION_CATEGORIES = frozenset({
    "committed_metal_contact",
    "committed_via_pad_contact",
    "committed_cut_contact",
    "committed_metal_spacing",
    "committed_via_pad_spacing",
    "committed_cut_spacing",
})


def _negotiation_committed_blocker_components(
    *,
    ordered: Sequence[str],
    open_nets: Sequence[str],
    route_stops: Mapping[str, "NegotiationRouteStop"],
    committed: Mapping[str, Any],
    protected_nets: Set[str],
) -> Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...]:
    """Build canonical open/blocker components from exact committed contacts.

    Only the three typed committed-contact categories can direct replay.  An
    owner must still name a connected route in the current book, must differ
    from the failed net, and must not have this round's bounded spine
    protection.  Fixed artwork, current-attempt geometry, unowned/unknown names,
    and stale disconnected owners therefore remain diagnostics only.
    """

    rank = {str(net): index for index, net in enumerate(ordered)}
    open_set = {
        str(net) for net in open_nets if str(net) in rank
    }
    protected = {str(net) for net in protected_nets}
    adjacency: Dict[str, Set[str]] = {}

    for open_net in sorted(open_set, key=lambda net: rank[net]):
        stop = route_stops.get(open_net)
        if stop is None:
            continue
        try:
            details = dict(stop.details)
        except (TypeError, ValueError):
            continue
        samples = details.get("rejection_samples", ())
        if not isinstance(samples, (tuple, list)):
            continue
        for sample in samples:
            try:
                fields = dict(sample)
            except (TypeError, ValueError):
                continue
            category = fields.get("category")
            owner = fields.get("owner")
            if (
                category not in _COMMITTED_BLOCKER_REJECTION_CATEGORIES
                or not isinstance(owner, str)
                or owner == open_net
                or owner in protected
                or owner not in rank
            ):
                continue
            owner_route = committed.get(owner)
            if owner_route is None or not bool(
                getattr(owner_route, "connected", False)
            ):
                continue
            adjacency.setdefault(open_net, set()).add(owner)
            adjacency.setdefault(owner, set()).add(open_net)

    components = []
    visited: Set[str] = set()
    for seed in sorted(open_set, key=lambda net: rank[net]):
        if seed in visited or seed not in adjacency:
            continue
        frontier = [seed]
        component: Set[str] = set()
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.extend(sorted(
                adjacency.get(current, ()),
                key=lambda net: rank[net],
                reverse=True,
            ))
        visited.update(component)
        component_opens = tuple(
            net for net in ordered if net in component and net in open_set
        )
        component_blockers = tuple(
            net for net in ordered if net in component and net not in open_set
        )
        if component_opens and component_blockers:
            components.append((component_opens, component_blockers))
    return tuple(components)


def _negotiation_reroute_in_place(
    *,
    ordered: Sequence[str],
    victims: Set[str],
    book,
    committed,
    route_one,
    blocker_components: Sequence[
        Tuple[Sequence[str], Sequence[str]]
    ] = (),
    atomic_route_groups: Tuple[Tuple[str, ...], ...] = (),
    route_stops: Optional[Mapping[str, "NegotiationRouteStop"]] = None,
    protected_nets: Optional[Set[str]] = None,
) -> None:
    """Settle ordinary victims, then replay directed blocker components.

    Non-component victims first retain the historical Gauss-Seidel contract:
    later old demand stays present until that owner is rerouted.  Each directed
    component is then removed atomically, with its open nets routing in canonical
    order before its connected blockers.  Declared atomic route groups (currently
    hard Mirror primary/partner pairs) close each component before removal, so a
    partner cannot be replayed against a stale failed primary and then miss the
    newly routed primary for the rest of the round.  Cross-component membership
    is defensively merged so every net receives at most one attempt per round.
    """
    rank = {str(net): index for index, net in enumerate(ordered)}
    if type(atomic_route_groups) is not tuple:
        raise TypeError("atomic route groups must be one exact tuple")
    atomic_groups = []
    atomic_members: Set[str] = set()
    for raw_group in atomic_route_groups:
        if (
            type(raw_group) is not tuple
            or len(raw_group) < 2
            or any(type(net) is not str or net not in rank for net in raw_group)
            or len(set(raw_group)) != len(raw_group)
            or atomic_members.intersection(raw_group)
        ):
            raise ValueError("atomic route groups must be disjoint exact net tuples")
        group = frozenset(raw_group)
        atomic_groups.append(group)
        atomic_members.update(group)

    component_specs = []
    for open_group, blocker_group in blocker_components:
        open_members = {str(net) for net in open_group if str(net) in rank}
        blocker_members = {
            str(net) for net in blocker_group
            if str(net) in rank and str(net) not in open_members
        }
        for atomic_group in atomic_groups:
            if atomic_group & open_members:
                open_members.update(atomic_group)
                blocker_members.difference_update(atomic_group)
            elif atomic_group & blocker_members:
                blocker_members.update(atomic_group)
        if open_members or blocker_members:
            component_specs.append((open_members, blocker_members))

    # Two directed blocker components can be joined only by an atomic pair whose
    # members appeared on opposite sides.  Merge that closure before touching the
    # book; otherwise the first component could recommit half the pair while the
    # second still holds stale geometry for its mate.
    index = 0
    while index < len(component_specs):
        open_members, blocker_members = component_specs[index]
        probe = index + 1
        while probe < len(component_specs):
            other_open, other_blockers = component_specs[probe]
            if (open_members | blocker_members) & (other_open | other_blockers):
                open_members.update(other_open)
                blocker_members.update(other_blockers)
                blocker_members.difference_update(open_members)
                component_specs.pop(probe)
                continue
            probe += 1
        index += 1

    component_members: Set[str] = set()
    for open_members, blocker_members in component_specs:
        component_members.update(open_members)
        component_members.update(blocker_members)

    for net in ordered:
        if net not in victims or net in component_members:
            continue
        book.remove(net)
        committed.pop(net, None)
        route_one(net)

    for open_members, blocker_members in component_specs:
        for group in (open_members, blocker_members):
            for net in sorted(group, key=lambda value: rank[value]):
                book.remove(net)
                committed.pop(net, None)
        while True:
            ordered_opens = tuple(sorted(
                open_members, key=lambda value: rank[value]
            ))
            for net in ordered_opens:
                route_one(net)
            failed_opens = tuple(
                net for net in ordered_opens
                if not bool(getattr(committed.get(net), "connected", False))
            )
            if not failed_opens or route_stops is None:
                break
            discovered_components = _negotiation_committed_blocker_components(
                ordered=ordered,
                open_nets=failed_opens,
                route_stops=route_stops,
                committed=committed,
                protected_nets=(protected_nets or set()),
            )
            discovered = {
                str(net)
                for _opens, blockers in discovered_components
                for net in blockers
                if str(net) in rank
            }
            new_blockers = discovered - open_members - blocker_members
            for atomic_group in atomic_groups:
                if atomic_group & new_blockers:
                    new_blockers.update(atomic_group - open_members)
            if not new_blockers:
                break
            # Keep every causal blocker removed until the whole open/atomic group
            # succeeds.  Recommitting an earlier blocker here would merely hide
            # the next blocker and reproduce one-layer-per-round convergence.
            for net in sorted(open_members, key=lambda value: rank[value]):
                book.remove(net)
                committed.pop(net, None)
            for net in sorted(new_blockers, key=lambda value: rank[value]):
                book.remove(net)
                committed.pop(net, None)
            blocker_members.update(new_blockers)
            component_members.update(new_blockers)
        for net in sorted(blocker_members, key=lambda value: rank[value]):
            route_one(net)


def _sep(a: Rect, b: Rect) -> Tuple[int, int]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 < bx0:
        sx = bx0 - ax1
    elif bx1 < ax0:
        sx = ax0 - bx1
    else:
        sx = 0
    if ay1 < by0:
        sy = by0 - ay1
    elif by1 < ay0:
        sy = ay0 - by1
    else:
        sy = 0
    return sx, sy


def _closed_rects_touch(a: Rect, b: Rect) -> bool:
    """Closed axis-aligned rectangles overlap or touch on both axes."""
    return _sep(_norm(a), _norm(b)) == (0, 0)


def _marker_selected_conductor_components(
    literals: Sequence[Tuple[Tuple[int, int], Optional[str], Rect]],
    *,
    conductor_layer: Tuple[int, int],
    marker_layer: Tuple[int, int],
) -> Tuple[Tuple[Optional[str], Rect], ...]:
    """Select whole fixed conductor components interacting a MIM marker.

    Foundry ``metal.interacting(marker)`` operates on connected same-layer
    geometry, not just the rectangle directly under the marker.  Flooding closed
    rectangle contact reproduces that component obligation for the exact fixed
    literal world retained by Gridless.
    """

    conductors = tuple(sorted(
        ((owner, _norm(rect)) for layer, owner, rect in literals
         if tuple(layer) == tuple(conductor_layer)),
        key=lambda item: (
            item[0] is not None,
            "" if item[0] is None else str(item[0]),
            item[1],
        ),
    ))
    markers = tuple(sorted(
        _norm(rect) for layer, _owner, rect in literals
        if tuple(layer) == tuple(marker_layer)
    ))
    selected = {
        index
        for index, (_owner, rect) in enumerate(conductors)
        if any(_closed_rects_touch(rect, marker) for marker in markers)
    }
    frontier = list(sorted(selected))
    while frontier:
        current = frontier.pop(0)
        current_rect = conductors[current][1]
        for index, (_owner, rect) in enumerate(conductors):
            if index in selected or not _closed_rects_touch(current_rect, rect):
                continue
            selected.add(index)
            frontier.append(index)
    return tuple(conductors[index] for index in sorted(selected))


def _rects_clear(a: Rect, b: Rect, min_gap: int) -> bool:
    """Edge-to-edge separation >= ``min_gap`` on EITHER axis (a same-layer wire to
    the side OR above another metal is DRC-legal)."""
    sx, sy = _sep(_norm(a), _norm(b))
    return sx >= min_gap or sy >= min_gap


def _seg_bbox(ax: int, ay: int, bx: int, by: int, half: int) -> Rect:
    lo_x, hi_x = (ax, bx) if ax <= bx else (bx, ax)
    lo_y, hi_y = (ay, by) if ay <= by else (by, ay)
    return (lo_x - half, lo_y - half, hi_x + half, hi_y + half)


def _contact_line_end_repair_rects(
    joined,
    contacts,
    *,
    line_end_nm: int,
    eol_edge_max_nm: int,
) -> Tuple[Rect, ...]:
    """Exact directional M1-over-CO repairs for short route-created ends.

    This is the router-side twin of the synthesized deck's M1 line-end rule:
    ``metal.enclosing(contact, line_end, projection)`` filtered to metal edges
    shorter than ``eol_edge_max``.  Only the deficient short boundary edge is
    extended; long rail sides remain under the ordinary contact-enclosure rule.
    The returned rectangles are allocation requests, never emitter fill.
    """
    import klayout.db as kdb

    line_end = int(line_end_nm)
    edge_max = int(eol_edge_max_nm)
    if line_end <= 0 or edge_max <= 0 or joined.is_empty() or contacts.is_empty():
        return ()

    violations = joined.enclosing_check(
        contacts,
        line_end,
        False,
        kdb.Region.Projection,
    )
    short_edges = tuple(joined.edges().with_length(0, edge_max, False).each())

    def coordinates(edge):
        return (
            int(edge.p1.x),
            int(edge.p1.y),
            int(edge.p2.x),
            int(edge.p2.y),
        )

    def overlaps(first, boundary) -> bool:
        ax, ay, bx, by = coordinates(first)
        cx, cy, dx, dy = coordinates(boundary)
        if ay == by and cy == dy and ay == cy:
            return max(min(ax, bx), min(cx, dx)) < min(max(ax, bx), max(cx, dx))
        if ax == bx and cx == dx and ax == cx:
            return max(min(ay, by), min(cy, dy)) < min(max(ay, by), max(cy, dy))
        return False

    repairs: Set[Rect] = set()
    for pair in violations.each():
        first = pair.first
        second = pair.second
        boundary = next(
            (edge for edge in short_edges if overlaps(first, edge)),
            None,
        )
        if boundary is None:
            continue
        fx0, fy0, fx1, fy1 = coordinates(first)
        sx0, sy0, sx1, sy1 = coordinates(second)
        bx0, by0, bx1, by1 = coordinates(boundary)
        if fy0 == fy1 and sy0 == sy1:
            missing = line_end - abs(fy0 - sy0)
            if missing <= 0:
                continue
            lo_x, hi_x = sorted((bx0, bx1))
            if fy0 > sy0:
                repairs.add((lo_x, fy0, hi_x, fy0 + missing))
            else:
                repairs.add((lo_x, fy0 - missing, hi_x, fy0))
        elif fx0 == fx1 and sx0 == sx1:
            missing = line_end - abs(fx0 - sx0)
            if missing <= 0:
                continue
            lo_y, hi_y = sorted((by0, by1))
            if fx0 > sx0:
                repairs.add((fx0, lo_y, fx0 + missing, hi_y))
            else:
                repairs.add((fx0 - missing, lo_y, fx0, hi_y))
    return tuple(sorted(repairs))


class _SpatialIndex:
    """A uniform bucket grid over axis-aligned rects for near-neighbour queries.

    Legality is O(local) instead of O(all-obstacles): a rect is a spacing threat to
    a query bbox only if it falls within the query grown by the clearance, so only
    the buckets that box overlaps need scanning. Bucket rows are exact-filtered by
    closed bbox intersection before they reach a DRC consumer; sharing one coarse
    bucket is not itself a candidate fact. Rebuilt cheaply; obstacles are static,
    committed metal is re-indexed per rip-up round."""

    __slots__ = ("cell", "_buckets", "_native", "_exported_buckets")

    def __init__(self, cell: int = 6000) -> None:
        self.cell = max(1, int(cell))
        cpp = load_native("_gridless_cpp")
        if cpp is not None:
            self._native = cpp.NativeSpatialIndex(self.cell)
            self._buckets: Optional[
                Dict[Tuple[int, int], List[Tuple[Optional[str], Rect]]]
            ] = None
        else:
            self._native = None
            self._buckets = {}
        self._exported_buckets = None

    def __getstate__(self) -> Dict[str, Any]:
        native = self._native
        buckets = self._buckets
        if native is not None and (
            buckets is None
            or buckets is self._exported_buckets
            or type(buckets).__name__ == "mappingproxy"
        ):
            return {
                "cell": self.cell,
                "entries": native.export_entries(),
                "buckets": None,
                "sealed": type(buckets).__name__ == "mappingproxy",
            }
        # Preserve exact buckets mapping (unwrap MappingProxyType if sealed)
        raw_buckets = (
            dict(buckets)
            if buckets is not None and type(buckets).__name__ == "mappingproxy"
            else buckets
        )
        return {
            "cell": self.cell,
            "entries": None,
            "buckets": raw_buckets,
            "sealed": type(buckets).__name__ == "mappingproxy",
        }

    def __setstate__(self, state: Any) -> None:
        # Support both current dict state and legacy (None, slot_dict) tuples
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state = state[1]
        self.cell = max(1, int(state.get("cell", 6000)))
        self._exported_buckets = None
        sealed = bool(state.get("sealed", False))
        entries = state.get("entries")
        if entries is not None:
            cpp = load_native("_gridless_cpp")
            if cpp is not None:
                native = cpp.NativeSpatialIndex(self.cell)
                for owner, rect in entries:
                    native.add(owner, rect)
                self._native = native
                self._buckets = None
            else:
                self._native = None
                self._buckets = {}
                for owner, rect in entries:
                    self.add(owner, rect)
            if sealed:
                from types import MappingProxyType
                self._buckets = MappingProxyType(self.buckets)
            return
        raw_buckets = state.get("buckets")
        if raw_buckets is None:
            raw_buckets = state.get("_buckets")
        self._native = None
        if raw_buckets is None:
            self._buckets = {}
        elif sealed:
            from types import MappingProxyType
            self._buckets = MappingProxyType(raw_buckets)
        else:
            self._buckets = raw_buckets

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_SpatialIndex":
        import copy as _copy
        native = self._native
        buckets = self._buckets
        if native is not None and buckets is None:
            cloned_native = native.clone()
            if cloned_native is not None:
                out = object.__new__(_SpatialIndex)
                out.cell = self.cell
                out._native = cloned_native
                out._buckets = None
                out._exported_buckets = None
                memo[id(self)] = out
                return out
            out = _SpatialIndex(self.cell)
            memo[id(self)] = out
            for owner, rect in native.export_entries():
                out.add(_copy.deepcopy(owner, memo), _copy.deepcopy(rect, memo))
            return out
        out = object.__new__(_SpatialIndex)
        out.cell = self.cell
        out._exported_buckets = None
        memo[id(self)] = out
        if native is not None and (
            buckets is self._exported_buckets
            or type(buckets).__name__ == "mappingproxy"
        ):
            cloned_native = native.clone()
            if cloned_native is not None:
                out._native = cloned_native
                if type(buckets).__name__ == "mappingproxy":
                    from types import MappingProxyType
                    out._buckets = MappingProxyType(
                        {k: _copy.deepcopy(v, memo) for k, v in buckets.items()}
                    )
                else:
                    copied_b = _copy.deepcopy(buckets, memo)
                    out._buckets = copied_b
                    out._exported_buckets = copied_b
                return out
        out._native = None
        if buckets is None:
            out._buckets = {}
        elif type(buckets).__name__ == "mappingproxy":
            from types import MappingProxyType
            out._buckets = MappingProxyType(
                {k: _copy.deepcopy(v, memo) for k, v in buckets.items()}
            )
        else:
            out._buckets = _copy.deepcopy(buckets, memo)
        return out

    @property
    def buckets(self) -> Dict[Tuple[int, int], List[Tuple[Optional[str], Rect]]]:
        b = self._buckets
        if b is None:
            b = self._native.export_buckets() if self._native is not None else {}
            self._buckets = b
            self._exported_buckets = b
        return b

    @buckets.setter
    def buckets(self, value: Any) -> None:
        self._buckets = value
        # Preserve native acceleration when sealing with MappingProxyType or
        # restoring our exported dict; drop native if caller replaces with an
        # unrelated mapping.
        if (
            self._native is not None
            and value is not self._exported_buckets
            and type(value).__name__ != "mappingproxy"
        ):
            self._native = None

    def add(self, owner: Optional[str], rect: Rect, *, diagnostic_work=None) -> None:
        # An interrupted diagnostic add may leave this index partly filled.
        # Its caller must stage the index locally until the operation is safe.
        native = self._native
        if diagnostic_work is None and self._buckets is None and native is not None:
            native.add(owner, rect)
            return
        x0, y0, x1, y1 = _norm(rect)
        c = self.cell
        buckets = self.buckets
        try:
            for bx in range(x0 // c, x1 // c + 1):
                for by in range(y0 // c, y1 // c + 1):
                    if diagnostic_work is not None:
                        diagnostic_work("index.add.bucket")
                    buckets.setdefault((bx, by), []).append((owner, rect))
        except BaseException:
            if diagnostic_work is not None:
                self._native = None
            raise
        if native is not None:
            native.add(owner, rect)

    def _query_python(self, bbox: Rect, *, diagnostic_work=None):
        x0, y0, x1, y1 = bbox
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        c = self.cell
        buckets = self.buckets
        bx0, bx1 = x0 // c, x1 // c
        by0, by1 = y0 // c, y1 // c
        if diagnostic_work is None and bx0 == bx1 and by0 == by1:
            row = buckets.get((bx0, by0))
            if not row:
                return
            first_claim = None
            seen_single: Optional[Set[Tuple[Optional[str], int]]] = None
            for item in row:
                rx0, ry0, rx1, ry1 = item[1]
                if rx0 > rx1:
                    rx0, rx1 = rx1, rx0
                if ry0 > ry1:
                    ry0, ry1 = ry1, ry0
                if rx1 < x0 or x1 < rx0 or ry1 < y0 or y1 < ry0:
                    continue
                claim_id = (item[0], id(item[1]))
                if first_claim is None:
                    first_claim = claim_id
                elif seen_single is None:
                    if claim_id == first_claim:
                        continue
                    seen_single = {first_claim, claim_id}
                elif claim_id in seen_single:
                    continue
                else:
                    seen_single.add(claim_id)
                yield item
            return
        seen: Set[Tuple[Optional[str], int]] = set()
        for bx in range(bx0, bx1 + 1):
            for by in range(by0, by1 + 1):
                if diagnostic_work is not None:
                    diagnostic_work("index.query.bucket")
                for item in buckets.get((bx, by), ()):
                    if diagnostic_work is not None:
                        diagnostic_work("index.query.row")
                    rx0, ry0, rx1, ry1 = item[1]
                    if rx0 > rx1:
                        rx0, rx1 = rx1, rx0
                    if ry0 > ry1:
                        ry0, ry1 = ry1, ry0
                    if rx1 < x0 or x1 < rx0 or ry1 < y0 or y1 < ry0:
                        continue
                    # One physical Rect may deliberately be indexed under two
                    # ownership claims.  Geometry identity alone would let the
                    # first owner hide the second and false-green cross-net
                    # legality; only duplicate bucket entries of the SAME claim
                    # are elided.
                    claim_id = (item[0], id(item[1]))
                    if claim_id in seen:
                        continue
                    seen.add(claim_id)
                    yield item

    def query(self, bbox: Rect, *, diagnostic_work=None):
        if diagnostic_work is None and self._native is not None:
            return self._native.query(bbox)
        return self._query_python(bbox, diagnostic_work=diagnostic_work)

    def scan_metal_conflicts(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        space_nm: int,
        own_owner: Optional[str],
        exclude_owner: Optional[str] = None,
    ) -> List[Tuple[Optional[str], Rect, bool]]:
        """Exact Euclidean spacing + closed-touch conflict scan."""
        native = self._native
        if native is not None:
            return native.scan_metal_conflicts(
                x0, y0, x1, y1, space_nm, own_owner, exclude_owner
            )
        box = _norm((x0, y0, x1, y1))
        grow = space_nm if space_nm > 0 else 0
        q = (box[0] - grow, box[1] - grow, box[2] + grow, box[3] + grow)
        spsq = space_nm * space_nm
        out: List[Tuple[Optional[str], Rect, bool]] = []
        for owner, rect in self.query(q):
            if owner == own_owner or (exclude_owner is not None and owner == exclude_owner):
                continue
            nb = _norm(rect)
            sx, sy = _sep(box, nb)
            if sx == 0 and sy == 0:
                out.append((owner, nb, True))
            elif space_nm > 0 and sx * sx + sy * sy < spsq:
                out.append((owner, nb, False))
        return out

    def scan_cut_conflicts(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        via_space_nm: int,
        own_owner: Optional[str],
    ) -> Tuple[int, Tuple[Tuple[Optional[str], Rect], ...], Tuple[Tuple[Optional[str], Rect], ...]]:
        """Single-pass cut scan returning (present_same, foreign_contacts, spacing_violations)."""
        native = self._native
        if native is not None:
            return native.scan_cut_conflicts(x0, y0, x1, y1, via_space_nm, own_owner)
        cut = _norm((x0, y0, x1, y1))
        q = (
            cut[0] - via_space_nm,
            cut[1] - via_space_nm,
            cut[2] + via_space_nm,
            cut[3] + via_space_nm,
        )
        present_same = 0
        foreign_contacts: List[Tuple[Optional[str], Rect]] = []
        spacing_violations: List[Tuple[Optional[str], Rect]] = []
        for owner, other in self.query(q):
            nb = _norm(other)
            if owner == own_owner:
                if nb == cut:
                    present_same += 1
                elif not _rects_clear(cut, nb, via_space_nm):
                    spacing_violations.append((owner, nb))
                continue
            if _sep(cut, nb) == (0, 0):
                foreign_contacts.append((owner, nb))
            elif not _rects_clear(cut, nb, via_space_nm):
                spacing_violations.append((owner, nb))
        key = lambda item: ("" if item[0] is None else str(item[0]), item[1])
        foreign_contacts.sort(key=key)
        spacing_violations.sort(key=key)
        return present_same, tuple(foreign_contacts), tuple(spacing_violations)


@dataclass(frozen=True)
class _WireSeg:
    gi: int
    ax: int
    ay: int
    bx: int
    by: int
    # Zero preserves the historical per-net/layer width table.  A positive
    # value freezes one local branch at the exact half-width used by search;
    # this is required when a minimum-width tap shares the same layer as a
    # separately materialized wide trunk/rail.
    half_width_nm: int = 0


@dataclass(frozen=True)
class _ViaSite:
    x: int
    y: int
    lo: int  # graph layer gi_low (bridges lo <-> lo+1)
    n: int = 0  # per-site via redundancy: 0 -> engine default; 1 -> single functional
    #             via (a sealed-pin escape the N x N reliability array could not fit)


@dataclass
class _HardConnectivityWitness:
    """Hard-legal pattern paths that can be embedded in residual candidate axes."""

    proofs: List[
        Tuple[
            Tuple[int, int, int],
            Tuple[int, int, int],
            Tuple[Tuple[int, int, int], ...],
        ]
    ] = field(default_factory=list)

    def prove(
        self,
        a: Tuple[int, int, int],
        b: Tuple[int, int, int],
        *,
        segs: Sequence[_WireSeg],
        vias: Sequence[_ViaSite],
    ) -> None:
        endpoint_a = tuple(int(value) for value in a)
        endpoint_b = tuple(int(value) for value in b)
        nodes = {endpoint_a, endpoint_b}
        for seg in segs:
            nodes.add((int(seg.ax), int(seg.ay), int(seg.gi)))
            nodes.add((int(seg.bx), int(seg.by), int(seg.gi)))
        for via in vias:
            nodes.add((int(via.x), int(via.y), int(via.lo)))
            nodes.add((int(via.x), int(via.y), int(via.lo + 1)))
        proof = (endpoint_a, endpoint_b, tuple(sorted(nodes)))
        if proof not in self.proofs:
            self.proofs.append(proof)

    @staticmethod
    def _crosses(
        a: Tuple[int, int, int],
        b: Tuple[int, int, int],
        starts: Set[Tuple[int, int, int]],
        goals: Set[Tuple[int, int, int]],
    ) -> bool:
        return (
            (a in starts and b in goals)
            or (b in starts and a in goals)
        )

    def augment_candidate_graph(
        self,
        starts: Sequence[Tuple[int, int, int]],
        goals: Set[Tuple[int, int, int]],
        xs: Sequence[int],
        ys: Sequence[int],
        layers: Set[int],
    ) -> Tuple[List[int], List[int], bool]:
        """Add the fewest exact pattern axes that embed one active-domain proof."""
        start_set = set(starts)
        base_xs = set(int(value) for value in xs)
        base_ys = set(int(value) for value in ys)
        candidates = []
        for a, b, nodes in self.proofs:
            if not self._crosses(a, b, start_set, goals):
                continue
            if any(gi not in layers for _x, _y, gi in nodes):
                continue
            proof_xs = {x for x, _y, _gi in nodes}
            proof_ys = {y for _x, y, _gi in nodes}
            candidates.append((
                len(proof_xs - base_xs) + len(proof_ys - base_ys),
                nodes,
                proof_xs,
                proof_ys,
            ))
        if not candidates:
            return sorted(base_xs), sorted(base_ys), False
        _added, _nodes, proof_xs, proof_ys = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        return (
            sorted(base_xs | proof_xs),
            sorted(base_ys | proof_ys),
            True,
        )

    def connects_graph(
        self,
        starts: Sequence[Tuple[int, int, int]],
        goals: Set[Tuple[int, int, int]],
        xs: Sequence[int],
        ys: Sequence[int],
        layers: Set[int],
    ) -> bool:
        start_set = set(starts)
        x_set = set(xs)
        y_set = set(ys)
        return any(
            self._crosses(a, b, start_set, goals)
            and all(
                x in x_set and y in y_set and gi in layers
                for x, y, gi in nodes
            )
            for a, b, nodes in self.proofs
        )


@dataclass
class _PatternSearchIncumbent:
    """Best hard-legal congested pattern available to a soft-cost A* probe.

    The closed-form pattern remains a complete feasible route.  Residual A* is
    therefore an improvement search, not a connectivity oracle: it may replace
    this incumbent with a strictly cheaper path, but exhausting its deterministic
    work authority must not turn the already-proved connection into an OPEN.
    """

    _rank: Optional[Tuple[Any, ...]] = None
    score: Optional[int] = None
    start: Optional[
        Tuple[Tuple[int, int, int], Optional[_ViaSite]]
    ] = None
    goal: Optional[
        Tuple[Tuple[int, int, int], Optional[_ViaSite]]
    ] = None
    segs: Tuple[_WireSeg, ...] = ()
    vias: Tuple[_ViaSite, ...] = ()

    @staticmethod
    def _access_key(
        item: Tuple[Tuple[int, int, int], Optional[_ViaSite]],
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        state, via = item
        if via is None:
            return (*state, 0, 0, 0, 0, 0)
        return (*state, 1, via.x, via.y, via.lo, via.n)

    def consider(
        self,
        *,
        score: int,
        candidate_index: int,
        start: Tuple[Tuple[int, int, int], Optional[_ViaSite]],
        goal: Tuple[Tuple[int, int, int], Optional[_ViaSite]],
        segs: Sequence[_WireSeg],
        vias: Sequence[_ViaSite],
    ) -> None:
        candidate_segs = tuple(segs)
        candidate_vias = tuple(vias)
        rank = (
            int(score),
            int(candidate_index),
            self._access_key(start),
            self._access_key(goal),
            tuple(
                (seg.gi, seg.ax, seg.ay, seg.bx, seg.by)
                for seg in candidate_segs
            ),
            tuple(
                (via.x, via.y, via.lo, via.n) for via in candidate_vias
            ),
        )
        if self._rank is not None and rank >= self._rank:
            return
        self._rank = rank
        self.score = int(score)
        self.start = start
        self.goal = goal
        self.segs = candidate_segs
        self.vias = candidate_vias

    def applies_to(
        self,
        starts: Sequence[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ],
        goals: Dict[Tuple[int, int, int], Optional[_ViaSite]],
        layers: Set[int],
    ) -> bool:
        if self.score is None or self.start is None or self.goal is None:
            return False
        start_keys = {self._access_key(item) for item in starts}
        goal_keys = {
            self._access_key(item) for item in goals.items()
        }
        access_via_ids = {
            id(via)
            for via in (self.start[1], self.goal[1])
            if via is not None
        }
        return (
            self._access_key(self.start) in start_keys
            and self._access_key(self.goal) in goal_keys
            and all(seg.gi in layers for seg in self.segs)
            and all(
                id(via) in access_via_ids
                or (via.lo in layers and via.lo + 1 in layers)
                for via in self.vias
            )
        )


@dataclass(frozen=True)
class _EmittedNetGeometry:
    """Explicit correspondence between routed segments and complete emit output."""

    segment_wires: Tuple[LayerPolygon, ...]
    wires: Tuple[LayerPolygon, ...]
    length_um: float


@dataclass(frozen=True)
class _DeckNegotiationSnapshot:
    """Canonical exact-emission view and its deck-connected isolation verdict."""

    literal_rects: Tuple["DeckLiteralRect", ...]
    report: "DeckIsolationReport"


@dataclass(frozen=True)
class _OwnedTerminalAtom:
    """One exact, literally drawn owned-metal rectangle in the route world.

    This is deliberately only a canonical obstacle projection.  It is not an
    owned-conductor certificate or a cached electrical component: component
    membership is recomputed from exact closed-touch geometry for each pin
    access against the current immutable route-world projection.
    """

    gi: int
    owner: str
    rect: Rect
    kind: Optional[str]
    terminal_conductor_ref: Optional[str]


@dataclass
class _AccessStaticDisconnectionRecorder:
    """Invocation-local observation; never part of engine checkpoint state."""
    scope: Optional[str] = None
    elapsed_ms: Optional[float] = None


@dataclass(frozen=True)
class _PlanarEndpointDomain:
    """Original endpoint and its exact, source-owned wire-centre regions.

    A required or materialized state is a singleton. Otherwise ``regions``
    contains inclusive manufacturing-grid bounds paired with the literal atom
    that authorizes each cap. Overlapping atoms remain separate witnesses; no
    component bounding box supplies metal.
    """

    node: Tuple[int, int, int]
    grid: int
    singleton: Optional[Tuple[int, int, int]]
    regions: Tuple[Tuple[_OwnedTerminalAtom, Rect], ...]
    excluded: frozenset = frozenset()
    locked_planar: bool = False

    def without_exact(self, key: Tuple[int, ...]) -> "_PlanarEndpointDomain":
        """Exclude one full access identity without clipping its owning atom."""
        return _PlanarEndpointDomain(
            self.node, self.grid, self.singleton, self.regions,
            self.excluded | {tuple(key)}, self.locked_planar,
        )

    def owning_atoms(self, state: Tuple[int, int, int]) -> Tuple[_OwnedTerminalAtom, ...]:
        """Return exact throat-reachable atom witnesses for an on-grid cap."""
        if self.singleton is not None or self.grid <= 0 or len(state) != 3:
            return ()
        x, y, gi = state
        if (gi != self.node[2] or x % self.grid or y % self.grid
                or tuple(state) == self.node
                or (*state, 0, 0, 0, 0, 0) in self.excluded):
            return ()
        return tuple(atom for atom, (x0, y0, x1, y1) in self.regions
                     if x0 <= x <= x1 and y0 <= y <= y1)


def _advance_planar_region_step(
    state, path, next_rectangle, *, route_world, net, grid, xs, ys,
    turn_cost, charge_event, event_run_cost, settle_prefix,
):
    """Advance one projected leg using the caller's routing world and ledger.

    The world supplies the original wire oracle; callbacks retain the active
    search's event budget and fixed-axis pricing.  The world, net and callbacks
    must describe the same search invocation and ledger.  This operation
    neither publishes geometry nor decides endpoint ownership.
    """
    if len(path) < 2:
        return None
    if not charge_event():
        return None
    ux, uy, ul, ulast = state[:4]
    rectangle, _pending, _destination, run_x, run_y, run_dir = state[4:10]
    vx, vy = path[1][0] * grid, path[1][1] * grid
    vdir = 1 if uy == vy else 2
    if not route_world._wire_legal(ul, ux, uy, vx, vy, net):
        return None
    signed_dir = ((1 if vx > ux else -1) if vdir == 1
                  else (2 if vy > uy else -2))
    static_cost = 0
    start_x, start_y = run_x, run_y
    if not run_dir:
        start_x, start_y = ux, uy
    elif signed_dir != run_dir:
        static_cost = event_run_cost(ul, (run_x, run_y), (ux, uy))
        if static_cost is None:
            return None
        start_x, start_y = ux, uy
    # Settle only the completed prefix of the original fixed-axis partition.
    boundary = settle_prefix((start_x, start_y), (vx, vy), xs, ys)
    if boundary is not None:
        prefix_cost = event_run_cost(ul, (start_x, start_y), boundary)
        if prefix_cost is None:
            return None
        static_cost += prefix_cost
        start_x, start_y = boundary
    turn = turn_cost if ulast != 0 and vdir != ulast else 0
    rest = tuple(path[2:])
    rect = next_rectangle if not rest and next_rectangle >= 0 else rectangle
    return (vx, vy, ul, vdir, rect, rest, next_rectangle,
            start_x, start_y, signed_dir), static_cost + turn, None


def _region_via_goal_waypoint(path, *, grid, layer, goal_set, target_domain):
    """Return the first actual projected state that already reaches the goal."""
    for index, (x, y) in enumerate(path[1:], 1):
        state = (x * grid, y * grid, layer)
        if (state in goal_set or
                (layer == target_domain.node[2] and target_domain.owning_atoms(state))):
            return index
    return None


def _region_via_goal_prefix_lower_bound(
    path, goal_index, *, grid, ux, uy, ulast, rate, turn_cost, base, pending,
):
    """Lower-bound only the projected path prefix ending at an earlier goal."""
    gx, gy = path[goal_index]
    first_direction = 1 if path[1][1] * grid == uy else 2
    goal_turn = turn_cost if ulast and first_direction != ulast else 0
    return base + pending + (abs(gx * grid - ux) + abs(gy * grid - uy)) * rate + goal_turn


def _region_via_target_prunable(
    path, *, grid, layer, other_layer, goal_set, target_domain,
    destination_owners, rect_saturated, via_successor_last_dir, protect_goal=True,
):
    """Discard a via projection only when its actual exit direction is full.

    A projected leg can reach a terminal before the via is taken, including
    at its intermediate elbow. Such a goal is not subject to the next layer's
    entry limit.
    """
    waypoint_fn = _eg_attr("_region_via_goal_waypoint", _region_via_goal_waypoint)
    if (protect_goal and waypoint_fn(
            path, grid=grid, layer=layer, goal_set=goal_set,
            target_domain=target_domain) is not None):
        return False
    x, y = path[-1]
    direction = 1 if path[-1][1] == path[-2][1] else 2
    direction = via_successor_last_dir(direction)
    return not any(
        not rect_saturated(other_layer, int(owner), x * grid, y * grid, direction)
        for owner, _rect in destination_owners
    )


class _CartesianSearchGraph:
    """The original fixed-axis graph and its packed adjacent-edge identities."""

    def __init__(self, xs: Sequence[int], ys: Sequence[int],
                 layers: Sequence[int], max_gi: int):
        self.xs, self.ys, self.layers = xs, ys, layers
        self.xi = {x: i for i, x in enumerate(xs)}
        self.yi = {y: i for i, y in enumerate(ys)}
        self.layer_index = {gi: i for i, gi in enumerate(layers)}
        self.max_gi = max_gi
        self.plane = len(xs) * len(ys)
        self.horizontal_edges = len(layers) * len(ys) * max(0, len(xs) - 1)
        self.vertical_edges = len(layers) * max(0, len(ys) - 1) * len(xs)
        self.via_pair_index = {
            lower: index for index, lower in enumerate(
                lower for lower, upper in zip(layers, layers[1:])
                if upper == lower + 1
            )
        }

    def physical_node_id(self, x: int, y: int, gi: int) -> int:
        return (self.layer_index[gi] * self.plane
                + self.yi[y] * len(self.xs) + self.xi[x])

    def embeds_planar_edge(self, gi: int, ax: int, ay: int, bx: int, by: int) -> bool:
        """Packed planar IDs represent only adjacent, fixed-axis transitions."""
        if gi not in self.layer_index:
            return False
        if ay == by and ay in self.yi and ax in self.xi and bx in self.xi:
            return abs(self.xi[ax] - self.xi[bx]) == 1
        if ax == bx and ax in self.xi and ay in self.yi and by in self.yi:
            return abs(self.yi[ay] - self.yi[by]) == 1
        return False

    def planar_edge_id(self, gi: int, ax: int, ay: int, bx: int, by: int) -> int:
        layer_pos = self.layer_index[gi]
        if ay == by:
            return ((layer_pos * len(self.ys) + self.yi[ay]) * max(0, len(self.xs) - 1)
                    + min(self.xi[ax], self.xi[bx]))
        return (self.horizontal_edges
                + (layer_pos * max(0, len(self.ys) - 1) + min(self.yi[ay], self.yi[by]))
                * len(self.xs) + self.xi[ax])

    def planar_cache_key(self, gi: int, ax: int, ay: int, bx: int, by: int) -> Optional[int]:
        """Only fixed-axis adjacent edges may use the packed Cartesian memo."""
        if not self.embeds_planar_edge(gi, ax, ay, bx, by):
            return None
        return self.planar_edge_id(gi, ax, ay, bx, by)

    def via_edge_id(self, x: int, y: int, lo: int) -> int:
        return (self.horizontal_edges + self.vertical_edges
                + self.via_pair_index[lo] * self.plane
                + self.yi[y] * len(self.xs) + self.xi[x])

    def planar(self, x: int, y: int):
        cx, cy = self.xi[x], self.yi[y]
        if cx + 1 < len(self.xs):
            yield (self.xs[cx + 1], y, 1)
        if cx - 1 >= 0:
            yield (self.xs[cx - 1], y, 1)
        if cy + 1 < len(self.ys):
            yield (x, self.ys[cy + 1], 2)
        if cy - 1 >= 0:
            yield (x, self.ys[cy - 1], 2)

    def adjacent_layers(self, gi: int):
        for other in (gi - 1, gi + 1):
            if 0 <= other <= self.max_gi and other in self.layer_index:
                yield other


def _owned_terminal_atom_key(atom: _OwnedTerminalAtom) -> Tuple[Any, ...]:
    return (
        int(atom.gi),
        str(atom.owner),
        tuple(_norm(atom.rect)),
        "" if atom.kind is None else str(atom.kind),
        (
            ""
            if atom.terminal_conductor_ref is None
            else str(atom.terminal_conductor_ref)
        ),
    )


class NegotiationStopReason(str, Enum):
    """Deterministic reasons an unplanned feasibility route can stop."""

    INVALID_LAYER_ACCESS = "INVALID_LAYER_ACCESS"
    NO_LEGAL_ROUTE = "NO_LEGAL_ROUTE"
    EDGE_EXPANSION_BUDGET = "EDGE_EXPANSION_BUDGET"
    SOLVE_WORK_BUDGET = "SOLVE_WORK_BUDGET"
    WALL_CLOCK_WATCHDOG = "WALL_CLOCK_WATCHDOG"
    DECK_NET_MERGE = "DECK_NET_MERGE"


@dataclass(frozen=True)
class NegotiationRouteStop:
    """Typed, content-addressed per-net stop from negotiated congestion."""

    reason: NegotiationStopReason
    net: str
    phase: str
    evidence: Tuple[Any, ...] = ()
    details: Tuple[Tuple[str, Any], ...] = ()

    def canonical_json(self) -> str:
        return json.dumps({
            "details": {
                str(key): value
                for key, value in self.details
            },
            "evidence": list(self.evidence),
            "net": self.net,
            "phase": self.phase,
            "reason": self.reason.value,
            "schema": "clay.negotiation-route-stop.v2",
        }, sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def content_ref(self) -> str:
        return f"{self.reason.value}:{self.net}:{self.sha256()}"


class NegotiationWatchdogStop(RuntimeError):
    """Non-emitting typed STOP when the outer wall-clock watchdog expires."""

    def __init__(self, stop: NegotiationRouteStop) -> None:
        if (
            type(stop) is not NegotiationRouteStop
            or stop.reason is not NegotiationStopReason.WALL_CLOCK_WATCHDOG
        ):
            raise TypeError("NegotiationWatchdogStop requires a watchdog route stop")
        self.stop = stop
        super().__init__(stop.canonical_json())


class GroupRouteBudgetStop(ValueError):
    """Retain a group-local budget stop across the public flow boundary."""

    def __init__(self, message: str, diagnostic: Dict[str, Any]) -> None:
        self._diagnostic = diagnostic
        super().__init__(message)

    def flow_error_data(self) -> Dict[str, Any]:
        return dict(self._diagnostic)


class GroupRouteLandingUnavailable(ValueError):
    """The bounded group interface has no legal policy-compatible landing."""

    def __init__(self, group: str, net: str, via_redundancy: int) -> None:
        self.group, self.net, self.via_redundancy = group, net, via_redundancy
        super().__init__(f"group-route port {group!r}:{net!r} has no legal landing")

    def flow_error_data(self) -> Dict[str, Any]:
        return {"schema": "clay.group-route-landing-unavailable.v1",
                "reason": "EXECUTION_INCOMPLETE",
                "failure_kind": "GROUP_ROUTE_LANDING_UNAVAILABLE",
                "group": self.group, "net": self.net,
                "via_redundancy": self.via_redundancy}


@dataclass
class _RoutedNet:
    name: str
    segs: List[_WireSeg] = field(default_factory=list)
    vias: List[_ViaSite] = field(default_factory=list)
    connected: bool = False


class _Book:
    """Committed metal, via landing pads and cuts, keyed by owning net."""

    def __init__(self, n_layers: int) -> None:
        self.metal: Dict[int, List[Tuple[str, Rect]]] = {i: [] for i in range(n_layers)}
        self.via_pads: Dict[int, List[Tuple[str, Rect]]] = {
            i: [] for i in range(n_layers)
        }
        self.cuts: List[Tuple[str, Rect, int]] = []  # (net, cut_rect, gi_low)
        self.persistent: Dict[
            str,
            Tuple[
                Dict[int, List[Rect]],
                List[Tuple[Rect, int]],
                Dict[int, List[Rect]],
            ],
        ] = {}
        self._persistent_metal_cache: Dict[
            Tuple[str, int], Tuple[int, int, FrozenSet[Rect]]
        ] = {}

    def is_persistent_metal(
        self, gi: int, owner: Optional[str], rect: Rect
    ) -> bool:
        """Return True iff ``rect`` belongs to ``owner``'s frozen persistent metal."""
        if owner is None:
            return False
        entry = self.persistent.get(owner)
        if entry is None:
            return False
        rects = entry[0].get(int(gi))
        if not rects:
            return False
        cache = getattr(self, "_persistent_metal_cache", None)
        if cache is None:
            cache = {}
            self._persistent_metal_cache = cache
        cache_key = (str(owner), int(gi))
        list_id = id(rects)
        list_len = len(rects)
        cached = cache.get(cache_key)
        if cached is None or cached[0] != list_id or cached[1] != list_len:
            norm_set = frozenset(
                (
                    min(int(r[0]), int(r[2])),
                    min(int(r[1]), int(r[3])),
                    max(int(r[0]), int(r[2])),
                    max(int(r[1]), int(r[3])),
                )
                for r in rects
            )
            cache[cache_key] = (list_id, list_len, norm_set)
        else:
            norm_set = cached[2]
        nr = (
            min(int(rect[0]), int(rect[2])),
            min(int(rect[1]), int(rect[3])),
            max(int(rect[0]), int(rect[2])),
            max(int(rect[1]), int(rect[3])),
        )
        return nr in norm_set

    def add(self, name: str, metal_by_gi: Dict[int, List[Rect]],
            cuts: List[Tuple[Rect, int]],
            via_pads_by_gi: Optional[Dict[int, List[Rect]]] = None) -> None:
        for gi, rects in metal_by_gi.items():
            for r in rects:
                self.metal[gi].append((name, r))
        for gi, rects in (via_pads_by_gi or {}).items():
            for r in rects:
                self.via_pads[gi].append((name, r))
        for (cut, lo) in cuts:
            self.cuts.append((name, cut, lo))

    def remove(self, name: str) -> None:
        for gi in self.metal:
            self.metal[gi] = [(n, r) for (n, r) in self.metal[gi] if n != name]
            self.via_pads[gi] = [
                (n, r) for (n, r) in self.via_pads[gi] if n != name
            ]
        self.cuts = [(n, r, lo) for (n, r, lo) in self.cuts if n != name]
        persistent = self.persistent.get(name)
        if persistent is not None:
            metal_by_gi, cuts, via_pads_by_gi = persistent
            self.add(name, metal_by_gi, cuts, via_pads_by_gi)

    def set_persistent(
        self,
        name: str,
        metal_by_gi: Dict[int, List[Rect]],
        cuts: List[Tuple[Rect, int]],
        via_pads_by_gi: Dict[int, List[Rect]],
    ) -> None:
        """Install one always-present pre-global route interface for ``name``."""

        self.persistent.pop(name, None)
        self.remove(name)
        frozen_metal = {
            int(gi): [tuple(rect) for rect in rects]
            for gi, rects in metal_by_gi.items()
        }
        frozen_cuts = [(tuple(rect), int(lo)) for rect, lo in cuts]
        frozen_pads = {
            int(gi): [tuple(rect) for rect in rects]
            for gi, rects in via_pads_by_gi.items()
        }
        self.persistent[name] = (frozen_metal, frozen_cuts, frozen_pads)
        self.add(name, frozen_metal, frozen_cuts, frozen_pads)


class GroupRouteFullFeedbackUnavailable(ValueError):
    """Full-prefix feedback needs the installed transactional physical caller."""



def _rect(key: Tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> LayerPolygon:
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    return LayerPolygon(layer=int(key[0]), datatype=int(key[1]),
                        polygon=Polygon.rectangle(int(lo_x), int(lo_y),
                                                  int(hi_x), int(hi_y)))


def _pop(lst: list, item) -> None:
    for idx in range(len(lst) - 1, -1, -1):
        if lst[idx] == item:
            lst.pop(idx)
            return

