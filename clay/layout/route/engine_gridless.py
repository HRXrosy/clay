"""GridlessEngine -- a shape-based, gridless over-the-cell RouteEngine strategy.

This is ANOTHER strategy behind the :class:`clay.layout.route.engine.RouteEngine`
contract (see that module's docstring). Where
:class:`~clay.layout.route.engine_pathfinder.PathFinderEngine` routes on ONE uniform
grid whose pitch is ``max(min_width + min_space)`` and rounds every pin/obstacle edge
to the nearest node, this strategy routes on CONTINUOUS nm coordinates at the exact
DRC-min clearance -- the lever the seam study (V3) identified as the only way to pack
below ~1.74x, where the uniform grid collapses two sub-pitch terminals onto the same
node (VOUT opens / GND merges).

It is a FRESH reimplementation of the ideas in the retired
``clay/layout/gridless_over_cell_router_v178.py`` (candidate-line feature-driven Hanan
grid + wire/via clearance predicates + multi-layer Dijkstra + bounded greedy rip-up),
lifted UP to the engine interface so it consumes the SAME micron-float
:class:`~clay.layout.route.engine.Pin` / :class:`~clay.layout.route.engine.Obstacle`
inputs and the SAME :class:`~clay.layout.route.engine_maze._Tech` PDK worldview the
other engines use -- NO private physical world, NO per-circuit branch, NO circuit-name
special case. The clearance-offset kernel is also available in Rust
(``clay/layout/_rs_bridge_clearance_offset_v180.py``); this engine keeps a generic
pure-Python candidate-line builder (not capped at 5 layers) so it works on any PDK's
routable stack, and can be swapped to the Rust hot path where the layer count matches.

Honesty contract (identical to the retired module + the RouteEngine contract):
  * PURE search + geometry. The engine NEVER asserts a DRC/LVS verdict. Its clearance
    model (edge-to-edge >= min_space per layer, cut spacing between same-kind cuts) is
    its OWN legality predicate; the real KLayout DRC/LVS deck is the INDEPENDENT oracle.
  * If a net cannot be connected it is reported ``connected=False`` (never faked); a
    constraint this strategy does not satisfy is recorded in ``unsatisfied`` and
    ``success`` is False.
  * Deterministic pure function of ``(pins_by_net, obstacles, constraints, pdk)``.

Scope (honest): rectilinear only, a rectilinear MST for fan-out (not an optimal
Steiner tree), and bounded deterministic negotiated congestion.  It is a
FEASIBILITY engine -- connectivity-first, DRC-min clearance -- with declared
Width carriers, exact vertical Mirror pairs, and bounded planar length tuning;
it is not a reference-shape copier.

Emitted-geometry legality model (R7a): via landing pads are drawn at
``max(metal min-width, cut + 2 * deck-enforced enclosure)`` per side, resolved
through the PDK contract surface (the SAME rule values the signoff deck consumes);
the ONE ``_via_pad_half`` number feeds legality, congestion, occupancy and the
emitted polygons. Own-net artwork obeys TOUCH-OR-CLEAR under the deck's net-blind
space() -- enforced SOFTLY (an escalating congestion price; hard-banding sealed
real pin accesses) and judged EXACTLY at convergence. Via cut spacing follows the
PDK ``via{k}_space`` rule, and convergence additionally requires ZERO
DECK-VISIBLE sub-min-space residuals -- judged at the DECK'S OWN semantics
(merged drawn geometry, net-blind, euclidean, at the parsed rule values;
``_space_residual_scan``). The engine's deliberately-STRICTER exact detector
(min-width swath coverage, junction-width on merged contacts) stays an ADVISORY
diagnostic reported as ``advisory_residuals`` on the result -- it never blocks
convergence or ``success`` and is never silently dropped (the old strict stop
burned entire route budgets negotiating residuals the real deck never flags);
the real KLayout deck remains the independent oracle.
"""
from __future__ import annotations

import heapq
import hashlib
import json
import math
import time
import weakref
from array import array
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import islice
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from clay.flow.execution_session import current_session
from clay.layout.backend import LayerPolygon, Polygon
from clay.layout.minimum_area import minimum_area_square_side_nm
from clay.layout.route.component_access_session import (
    ComponentAccessSession,
    EscapeProof,
    EscapeWorldKey,
    PartitionContext,
    SiteLegalityKey,
    ViaLegalityProof,
)
from clay.layout.route.terminal_identity import (
    pin_fact as canonical_pin_fact,
    pin_provenance,
)
from clay.layout.route.constraints import (
    Access,
    Direction,
    LayerSet,
    Length,
    Mirror,
    RouteConstraint,
    Shield,
    Spacing,
    TrackOrder,
    TrunkLayer,
    TrunkLayerDeclarationError,
    Via,
    Width,
)
from clay.layout.route.engine import (
    NetRoute,
    Obstacle,
    Pin,
    RoutedCell,
    RouteEngine,
    RouteResourcePlanIncompleteError,
    require_resource_plan_support,
)
from clay.layout.route.spacing_intent import (
    check_spacing_geometry,
    compile_spacing_rules,
    fixed_spacing_conflicts,
    spacing_blockers,
    spacing_nm,
)
from clay.layout.route.floorplan_containment import (
    RoutingEnvelope,
    require_routed_containment,
)
from clay.layout.route.metal_junction_legalization import (
    legalize_island_junctions,
)
from clay.layout.route.engine_maze import (
    _NM_PER_UM,
    _Tech,
    _via_cut_size_nm,
    _via_name_for,
)
from clay.layout.route.via_model import (
    landing_rect_for_cut,
    subtract_rects,
    via_enclosure_nm,
)


from clay.layout.route.gridless_types import (
    Rect,
    _GRIDLESS_REPLAY_ISSUER_AUTHORITY,
    _GRIDLESS_REPLAY_RECEIPTS,
    _MAX_GRIDLESS_REPLAY_SELECTED_CANDIDATES,
    _allocation_geometry_digest,
    _gridless_replay_receipt_payload,
    _is_lower_sha256,
    GridlessAllocationReplayCertificate,
    _verify_gridless_allocation_replay_certificate,
    _via_cut_space_nm,
    _HANDLED_KINDS,
    _CONSTRAINT_TYPES,
    TrunkLayerResolutionError,
    _STEP_COST_PER_NM,
    _TURN_COST_PITCHES,
    _OFF_PREF_COST_PER_NM,
    _VIA_COST,
    _EXACT_GOAL_INDEX_VISIT_WORK,
    _M1_ROUTE_COST_PER_NM,
    _TRUNK_SPAN_PITCHES,
    _TRUNK_BELOW_COST_PER_NM,
    _CONG_WEIGHT,
    _COST_FACTOR_SCALE,
    _ExactGoalPotentialNode,
    _ExactGoalPotentialIndex,
    _norm,
    _pin_xy_nm,
    _pin_order_key,
    _group_route_pin_key,
    _obstacle_order_key,
    _negotiation_net_order,
    _topology_terminal_count,
    _topology_terminal_span_nm,
    _deck_open_route_nets,
    _negotiation_snapshot_score,
    _negotiation_conflict_victims,
    _negotiation_protected_nets,
    _COMMITTED_BLOCKER_REJECTION_CATEGORIES,
    _negotiation_committed_blocker_components,
    _negotiation_reroute_in_place,
    _sep,
    _closed_rects_touch,
    _marker_selected_conductor_components,
    _rects_clear,
    _seg_bbox,
    _contact_line_end_repair_rects,
    _SpatialIndex,
    _WireSeg,
    _ViaSite,
    _HardConnectivityWitness,
    _PatternSearchIncumbent,
    _EmittedNetGeometry,
    _DeckNegotiationSnapshot,
    _OwnedTerminalAtom,
    _AccessStaticDisconnectionRecorder,
    _PlanarEndpointDomain,
    _advance_planar_region_step,
    _region_via_goal_waypoint,
    _region_via_goal_prefix_lower_bound,
    _region_via_target_prunable,
    _CartesianSearchGraph,
    _owned_terminal_atom_key,
    NegotiationStopReason,
    NegotiationRouteStop,
    NegotiationWatchdogStop,
    GroupRouteBudgetStop,
    GroupRouteLandingUnavailable,
    _RoutedNet,
    _Book,
    GroupRouteFullFeedbackUnavailable,
    _rect,
    _pop,
)
from clay.layout.route.gridless_resource_plan import _GridlessResourcePlanMixin
from clay.layout.route.gridless_legality import _GridlessLegalityMixin
from clay.layout.route.gridless_owned_escape import _GridlessOwnedEscapeMixin
from clay.layout.route.gridless_group_route import _GridlessGroupRouteMixin
from clay.layout.route.gridless_planar_search import _GridlessPlanarSearchMixin
from clay.layout.route.gridless_rail_drop import _GridlessRailDropMixin
from clay.layout.route.gridless_emit_drc import _GridlessEmitDrcMixin


class GridlessEngine(
    _GridlessResourcePlanMixin,
    _GridlessLegalityMixin,
    _GridlessOwnedEscapeMixin,
    _GridlessGroupRouteMixin,
    _GridlessPlanarSearchMixin,
    _GridlessRailDropMixin,
    _GridlessEmitDrcMixin,
    RouteEngine,
):
    """A gridless, shape-based over-the-cell :class:`RouteEngine` strategy.

    Parameters
    ----------
    time_budget_sec : float
        Outer wall-clock watchdog for the whole feasibility solve.  Deterministic
        work budgets own emitted geometry; expiry raises a named
        :class:`NegotiationWatchdogStop` and emits no timing-dependent partial route.
    max_iters : int
        Max negotiated-congestion rip-up/reroute rounds. If congestion is not
        resolved by then the BEST round seen is returned (``success=False`` + the
        remaining shorts / unconnected nets) -- honest non-convergence, never faked.
    pres_fac0 : float
        Initial present-congestion factor (cost of one currently-shared resource
        cell). Small at first so nets find natural channels.
    pres_fac_mult : float
        Per-round multiplier that raises the present-congestion factor, forcing nets
        off contested resources as rounds progress.
    hist_fac : float
        History increment/weight: each round a resource is over-used its history
        grows by ``hist_fac`` and history costs ``hist_fac`` per unit in the search,
        so chronic contention becomes steadily more expensive across rounds.
    fail_cap : int
        Accepted for back-compat; no longer used as a pruning authority.  A route
        failure is evidence only about the current committed/congestion world and
        remains retryable in every bounded negotiation round.  Only an input-static
        invalid layer-access proof is permanently excluded from reroute victims.
    max_expansions : int
        FLOOR on the per-edge A* node-expansion budget. The ACTUAL budget for an
        edge is ``max(max_expansions, |states|)`` capped at ``expansion_cap``, where
        ``|states| = len(xs) * len(ys) * n_allowed_layers * 3`` is the size of the
        edge's search graph (the 3 is the ``last_dir`` multiplier). A budget equal to
        the state-graph size makes the A* COMPLETE -- it can exhaust the local graph,
        so a long-span edge (a cell-spanning power trunk / far bias pin) is bounded by
        its OWN grid, not a one-size cap that starved it and left the net honestly
        unconnected. Deterministic (count-based), never wall-clock.
    expansion_cap : int
        Hard ceiling on the per-edge budget above, so a pathological huge window still
        terminates. Generous by default -- real routable edges find the goal long
        before exhausting their graph.
    prefer_target_search : bool
        Explicitly prefer the existing target-directed A* over its optional
        hard-connectivity flood. All edge legality and work limits remain in
        force. False retains the historical strategy; finite-budget outcomes
        and causal diagnostics may differ when True.
    max_ripups, per_net_rip_budget : int
        Accepted for back-compat; unused by the negotiated-congestion solver (the
        old bounded greedy rip-up they bounded is replaced by history negotiation).
    """

    # WORLD-MODEL CAPABILITY: this strategy consumes the FULL artwork conductor
    # projection (upper-metal stamp pads, net-tagged, + cut-layer riser rects).
    # route_cell keys the extended ``build_layout_model`` projection on this flag
    # so the default PathFinder inputs stay byte-identical (opt-in worldview,
    # same pattern as ``trunk_layers``).
    consumes_artwork_conductors = True
    # A replayed joint obligation with one routing-metal component lowers
    # losslessly to one physical binding. Its whole exact metal component stays
    # in the existing access search. Independent OR islands are not supported by
    # this Pin-only strategy and reject before any partial lowering.
    consumes_single_component_joint_terminals = True
    consumes_fixed_multilayer_joint_terminals = True

    # SAFETY CAPABILITY (unconditional): the canonical model and post-legalizer
    # must use real-polygon, unique-net device-M1 provenance for every Gridless
    # route. This is a class invariant, not a policy/style flag.
    strict_device_m1_provenance = True
    # FINALIZATION CAPABILITY: after Gridless has committed its ordinary
    # rectangle route, route_cell may reconcile the exact placed terminal
    # components with that route and commit any PDK-required M1 seam closure to
    # the owning NetRoute before containment/short/manifest replay.  The
    # independent emitter then observes this geometry; it never authors it.
    materializes_terminal_interface_m1 = True

    # WORLD-MODEL CAPABILITY: this strategy consumes DECLARED SUPPLY RAILS as a
    # net's routing SPINE (``rails=`` argument to ``route()``): each source pin of
    # a rail net drops LOCALLY onto the owned rail metal instead of joining
    # pin-to-pin toward one hub. route_cell keys the rail hand-off on this flag so
    # engines without the spine path (the default PathFinder) are never passed the
    # ``rails=`` kwarg and stay byte-identical (same opt-in pattern as
    # ``consumes_artwork_conductors``).
    consumes_rail_spine = True
    # PHYSICAL-DOMAIN CAPABILITY: canonical route_cell hands this engine the
    # materializer-derived final floorplan rectangle. Search legality consumes
    # the complete wire/via outline; no convenience halo may enlarge this world.
    consumes_routing_envelope = True

    # EXACT ALLOCATION CAPABILITY (RG3): a non-empty RouteResourcePlan takes a
    # separate, deterministic materialization path.  It never enters the legacy
    # pattern/MST/A*/negotiation pipeline below and therefore consumes none of its
    # style switches. None/an empty selection follows the bounded residual path,
    # whose direction bias is derived directly from the PDK preferred axes.
    consumes_resource_plan = True
    consumes_diagnostic_resource_program = True
    # FEASIBILITY CAPABILITY: with no resource plan this engine runs its bounded,
    # deterministic negotiated-congestion solve.  route_cell uses this handshake
    # only to authorize explicitly NON_SIGNOFF bring-up diagnostics.
    supports_deterministic_negotiation = True
    # The ordinary producer's _emit_net extends wire endpoints by half-width.
    # Exact resource-plan tracks are a separate flush-ended producer.
    negotiated_wire_square_ends = True
    # HIERARCHICAL ACCESS CAPABILITY: a non-empty group-route contract is
    # consumed before global negotiation by reserving one emitted via landing
    # for every topology-derived physical group port.
    consumes_group_route_ports = True

    def satisfies(self) -> frozenset:
        """The constraint kinds this gridless strategy MATERIALIZES: exactly
        ``_HANDLED_KINDS`` == {layer_set, width, mirror, length, spacing}. LayerSet caps a net's
        routable metals, Width draws the emitted half-width, and Mirror routes a
        primary before materializing and verifying its partner as an exact
        physical reflection. Length inserts a bounded, PDK-grid-derived planar
        tuning detour and is independently remeasured from final emitted
        rectangles. Spacing constrains both wires and via landing pads during
        search and is remeasured against returned metal and fixed terminals.
        ``TrunkLayer`` is a SOFT advisory consumed by the
        negotiated path, not a hard capability claim. Every OTHER kind
        (shield/track_order/access/direction/via) is
        recorded ``unsatisfied`` when routed here (honest P1 reporting), so a HARD
        one must be caught by the PRE-gate BEFORE emit -- this method is that
        gate's query. (Promotes the formerly-dead ``_HANDLED_KINDS`` to
        load-bearing.) A subset of ``constraints.CONSTRAINT_KINDS`` by
        construction."""
        return _HANDLED_KINDS

    @staticmethod
    def _route_stop_causal_details(
        stop: NegotiationRouteStop,
        details: Mapping[str, Any],
        failure_kind: str,
    ) -> Tuple[Tuple[str, Any], ...]:
        """Return a bounded, named replay surface for one unresolved edge.

        ``NegotiationRouteStop.details`` also carries transaction inventories that
        can be thousands of rows long.  The outer watchdog frontier must not copy
        those inventories, but a failure kind without its endpoint obligation is
        not actionable.  Retain only fixed-cardinality endpoint/search metadata;
        for the static-connectivity proof, name the five scalar counters already
        stored in the stop evidence.
        """

        def _exact_pairs(raw: Any) -> Optional[Dict[str, Any]]:
            if type(raw) is not tuple:
                return None
            pairs: Dict[str, Any] = {}
            for item in raw:
                if type(item) is not tuple or len(item) != 2:
                    return None
                key, value = item
                if type(key) is not str or key in pairs:
                    return None
                pairs[key] = value
            return pairs

        def _endpoint(raw: Any) -> Optional[Tuple[Tuple[str, Any], ...]]:
            pairs = _exact_pairs(raw)
            if pairs is None:
                return None
            if set(pairs) != {"device", "terminal", "xy_nm", "gi"}:
                return None
            xy_nm = pairs["xy_nm"]
            if (
                type(pairs["device"]) is not str
                or type(pairs["terminal"]) is not str
                or type(xy_nm) is not tuple
                or len(xy_nm) != 2
                or any(type(value) is not int for value in xy_nm)
                or type(pairs["gi"]) is not int
            ):
                return None
            return (
                ("device", pairs["device"]),
                ("terminal", pairs["terminal"]),
                ("xy_nm", xy_nm),
                ("gi", pairs["gi"]),
            )

        def _conflict_witness(
            raw: Any,
            *,
            tuple_widths: Mapping[str, int],
            scalar_names: Set[str],
        ) -> Optional[Tuple[Tuple[str, Any], ...]]:
            pairs = _exact_pairs(raw)
            if pairs is None or set(pairs) != set(tuple_widths) | scalar_names:
                return None
            for name in scalar_names:
                if type(pairs[name]) is not int:
                    return None
            for name, width in tuple_widths.items():
                value = pairs[name]
                if (
                    type(value) is not tuple
                    or len(value) != int(width)
                    or any(type(item) is not int for item in value)
                ):
                    return None
            return tuple(sorted(pairs.items(), key=lambda item: item[0]))

        rows: List[Tuple[str, Any]] = []
        for key in ("source", "goal"):
            endpoint = _endpoint(details.get(key))
            if endpoint is not None:
                rows.append((key, endpoint))

        window = details.get("window")
        if (
            type(window) is tuple
            and len(window) == 4
            and all(type(value) is int for value in window)
        ):
            rows.append(("window", window))
        allowed_layers = details.get("allowed_layers")
        if (
            type(allowed_layers) is tuple
            and len(allowed_layers) <= 32
            and all(type(value) is int for value in allowed_layers)
        ):
            rows.append(("allowed_layers", allowed_layers))

        for key in (
            "trunk_gi",
            "source_access_state_count",
            "goal_access_state_count",
            "nearest_source_access_state_count",
            "nearest_goal_access_state_count",
            "local_source_access_state_count",
            "frontier_graph_states",
            "frontier_graph_cap",
            "frontier_local_stage_count",
            "frontier_complete_stage_count",
            "frontier_local_max_graph_states",
            "frontier_complete_max_graph_states",
        ):
            value = details.get(key)
            if type(value) is int:
                rows.append((key, value))
        for key in (
            "signal_access_scope", "source_access_scope",
            "frontier_projection_scope",
        ):
            value = details.get(key)
            if type(value) is str:
                rows.append((key, value))

        conflict_schemas = {
            "candidate_via_pad_conflict": (
                {"left_via": 4, "right_via": 4},
                {"gi", "left_index", "right_index"},
            ),
            "candidate_via_wire_conflict": (
                {"via": 4, "wire": 6},
                {"gi", "segment_index", "via_index"},
            ),
            "candidate_via_attempt_conflict": (
                {"blocker": 4, "pad": 4, "via": 4},
                {"gi", "via_index"},
            ),
        }
        for key, (tuple_widths, scalar_names) in conflict_schemas.items():
            witness = _conflict_witness(
                details.get(key),
                tuple_widths=tuple_widths,
                scalar_names=scalar_names,
            )
            if witness is not None:
                rows.append((key, witness))

        if failure_kind == "static_disconnected" and len(stop.evidence) >= 5:
            static_values = stop.evidence[:5]
            if all(type(value) is int for value in static_values):
                rows.extend(zip(
                    ("connectivity_nodes", "connectivity_edges", "xs", "ys", "layers"),
                    static_values,
                ))
        return tuple(rows)

    @staticmethod
    def _route_stop_frontier(
        stops: Mapping[str, NegotiationRouteStop],
    ) -> Tuple[Tuple[Any, ...], ...]:
        """Bounded causal summary of the currently unresolved net frontier."""
        rows = []
        for name, stop in sorted(stops.items(), key=lambda item: item[0]):
            details = dict(stop.details)
            failure_kind = str(details.get("failure_kind", ""))
            mirror_failure_kind = str(details.get("mirror_failure_kind", ""))
            if mirror_failure_kind:
                failure_kind = (
                    f"{failure_kind}:{mirror_failure_kind}"
                    if failure_kind
                    else mirror_failure_kind
                )
            blockers = []
            for raw in details.get("rejection_samples", ()):
                try:
                    sample = dict(raw)
                except (TypeError, ValueError):
                    continue
                blockers.append((
                    str(sample.get("category", "")),
                    str(sample.get("owner", "")),
                    str(sample.get("blocker_kind", "") or ""),
                    int(sample.get("gi", 0) or 0),
                    int(sample.get("rejection_count", 0) or 0),
                ))
            rows.append((
                str(name),
                stop.reason.value,
                str(stop.phase),
                failure_kind,
                tuple(sorted(set(blockers))[:8]),
                GridlessEngine._route_stop_causal_details(
                    stop, details, failure_kind
                ),
            ))
        return tuple(rows[:64])

    # ----------------------------------------------------------- hard mirror
    @staticmethod
    def _index_mirrors(
        mirrors: Sequence[Mirror],
        pins_by_net: Mapping[str, Sequence[Pin]],
        unsatisfied: List[RouteConstraint],
    ) -> Dict[str, Mirror]:
        """Return valid hard pairs keyed by their materialized partner net.

        A gridless reflection is exact in integer-nanometre geometry.  Therefore
        ``2 * axis`` must itself be an integer nanometre coordinate.  Invalid or
        overlapping declarations stay explicit in ``unsatisfied``; a canonical
        ordering makes the result independent of declaration order.
        """
        by_b: Dict[str, Mirror] = {}
        used: Set[str] = set()
        ordered = sorted(
            mirrors,
            key=lambda item: (
                str(item.net_a),
                str(item.net_b),
                repr(item.axis_x_um),
            ),
        )
        for constraint in ordered:
            try:
                twice_axis = float(constraint.axis_x_um) * 2.0 * _NM_PER_UM
                axis_valid = (
                    math.isfinite(twice_axis)
                    and abs(twice_axis - round(twice_axis)) <= 1.0e-6
                )
            except (TypeError, ValueError, OverflowError):
                axis_valid = False
            a = str(constraint.net_a)
            b = str(constraint.net_b)
            valid = (
                axis_valid
                and a != b
                and a in pins_by_net
                and b in pins_by_net
                and a not in used
                and b not in used
            )
            if not valid:
                if constraint not in unsatisfied:
                    unsatisfied.append(constraint)
                continue
            used.update((a, b))
            by_b[b] = constraint
        return by_b

    @staticmethod
    def _match_pair_order(
        order: Sequence[str],
        length_constraints: Sequence[Length],
        mirror_by_b: Mapping[str, Mirror],
        reverse: bool = False,
    ) -> List[str]:
        if not length_constraints:
            return list(order)
        mirror_pairs = {
            frozenset((str(m.net_a), str(m.net_b)))
            for m in mirror_by_b.values()
        }
        pos = {n: idx for idx, n in enumerate(order)}
        paired_by_first: Dict[str, str] = {}
        used: Set[str] = set()
        for constraint in sorted(
            length_constraints,
            key=lambda item: (str(item.net_a), str(item.net_b)),
        ):
            a, b = str(constraint.net_a), str(constraint.net_b)
            if a == b or frozenset((a, b)) in mirror_pairs:
                continue
            if a not in pos or b not in pos or a in used or b in used:
                continue
            first, second = (b, a) if reverse else (a, b)
            paired_by_first[first] = second
            used.add(a)
            used.add(b)
        if not paired_by_first:
            return list(order)
        anchor_map: Dict[str, Tuple[str, str]] = {}
        for first, second in paired_by_first.items():
            anchor = first if pos[first] < pos[second] else second
            anchor_map[anchor] = (first, second)
        out_order: List[str] = []
        emitted: Set[str] = set()
        for n in order:
            if n in emitted:
                continue
            pair = anchor_map.get(n)
            if pair is not None:
                f_net, s_net = pair
                out_order.append(f_net)
                out_order.append(s_net)
                emitted.add(f_net)
                emitted.add(s_net)
            else:
                out_order.append(n)
                emitted.add(n)
        return out_order

    def _compute_match_layer_surcharge(
        self,
        net: str,
        length_constraints: Sequence[Length],
        committed: Mapping[str, _RoutedNet],
        declared_trunk_gi: Mapping[str, int],
    ) -> Dict[int, int]:
        partners: List[str] = []
        for constraint in length_constraints:
            a, b = str(constraint.net_a), str(constraint.net_b)
            if a == net and b != net:
                partners.append(b)
            elif b == net and a != net:
                partners.append(a)
        if not partners:
            return {}
        partner_len_by_gi: Dict[int, int] = {}
        total_len = 0
        for mate in sorted(set(partners)):
            routed = committed.get(mate)
            if routed is None or not getattr(routed, "connected", False):
                continue
            for seg in routed.segs:
                seg_len = abs(int(seg.bx) - int(seg.ax)) + abs(int(seg.by) - int(seg.ay))
                if seg_len <= 0:
                    continue
                gi = int(seg.gi)
                partner_len_by_gi[gi] = partner_len_by_gi.get(gi, 0) + seg_len
                total_len += seg_len
        if total_len < 1000:
            return {}
        own_trunk = declared_trunk_gi.get(net)
        surcharge: Dict[int, int] = {}
        n_layers = int(getattr(self, "_n_layers", 0) or 0)
        for gi in range(n_layers):
            if own_trunk is not None and gi == own_trunk:
                continue
            used_gi = partner_len_by_gi.get(gi, 0)
            if used_gi == 0:
                surcharge[gi] = 3
            elif used_gi * 6 < total_len:
                surcharge[gi] = 1
        return surcharge

    @staticmethod
    def _mirror_order(
        order: Sequence[str], mirror_by_b: Mapping[str, Mirror]
    ) -> List[str]:
        """Hoist each primary/partner pair, with the primary routed first.

        Reserving the pair before unconstrained signals follows the same
        correct-by-construction ordering used by the reference analytical
        router.  With no Mirror declarations this returns an identical copy.
        """
        if not mirror_by_b:
            return list(order)
        by_a = {constraint.net_a: b for b, constraint in mirror_by_b.items()}
        present = set(order)
        result: List[str] = []
        for net in order:
            partner = by_a.get(net)
            if partner is None:
                continue
            result.append(net)
            if partner in present:
                result.append(partner)
        for net in order:
            if net not in result:
                result.append(net)
        return result

    @staticmethod
    def _mirror_axis_twice_nm(constraint: Mirror) -> int:
        return int(round(float(constraint.axis_x_um) * 2.0 * _NM_PER_UM))

    @classmethod
    def _mirror_nodes_match(
        cls,
        constraint: Mirror,
        primary_nodes: Sequence[Tuple[int, int, int]],
        partner_nodes: Sequence[Tuple[int, int, int]],
    ) -> bool:
        axis2 = cls._mirror_axis_twice_nm(constraint)
        reflected = sorted(
            (axis2 - int(x), int(y), int(gi))
            for x, y, gi in primary_nodes
        )
        return reflected == sorted(
            (int(x), int(y), int(gi)) for x, y, gi in partner_nodes
        )

    @classmethod
    def _reflect_routed_net(
        cls, primary: _RoutedNet, partner: str, constraint: Mirror
    ) -> _RoutedNet:
        axis2 = cls._mirror_axis_twice_nm(constraint)
        return _RoutedNet(
            name=str(partner),
            segs=[
                _WireSeg(
                    int(segment.gi),
                    axis2 - int(segment.ax),
                    int(segment.ay),
                    axis2 - int(segment.bx),
                    int(segment.by),
                    int(getattr(segment, "half_width_nm", 0)),
                )
                for segment in primary.segs
            ],
            vias=[
                _ViaSite(
                    axis2 - int(via.x),
                    int(via.y),
                    int(via.lo),
                    int(via.n),
                )
                for via in primary.vias
            ],
            connected=bool(primary.connected),
        )

    def _materialize_mirror_partner(
        self,
        constraint: Mirror,
        primary: Optional[_RoutedNet],
        primary_nodes: Sequence[Tuple[int, int, int]],
        partner_nodes: Sequence[Tuple[int, int, int]],
        partner_layers: Sequence[int],
    ) -> Optional[_RoutedNet]:
        """Build one exact reflected route and validate every emitted primitive."""
        self._mirror_failure_details: Tuple[Tuple[str, Any], ...] = ()
        if primary is None or not primary.connected:
            self._mirror_failure_details = ((
                "mirror_failure_kind", "primary_unrouted"
            ),)
            return None
        axis2 = self._mirror_axis_twice_nm(constraint)
        expected_nodes = tuple(sorted(
            (axis2 - int(x), int(y), int(gi))
            for x, y, gi in primary_nodes
        ))
        actual_nodes = tuple(sorted(
            (int(x), int(y), int(gi)) for x, y, gi in partner_nodes
        ))
        if expected_nodes != actual_nodes:
            expected_counts = Counter(expected_nodes)
            actual_counts = Counter(actual_nodes)
            missing = tuple(sorted((expected_counts - actual_counts).elements()))
            unexpected = tuple(sorted((actual_counts - expected_counts).elements()))
            self._mirror_failure_details = (
                ("mirror_failure_kind", "endpoint_mismatch"),
                ("mirror_axis_twice_nm", int(axis2)),
                ("mirror_primary_node_count", len(expected_nodes)),
                ("mirror_partner_node_count", len(actual_nodes)),
                ("mirror_missing_partner_nodes", missing[:8]),
                ("mirror_unexpected_partner_nodes", unexpected[:8]),
            )
            return None
        partner = self._reflect_routed_net(
            primary, constraint.net_b, constraint
        )
        allowed = set(int(layer) for layer in partner_layers)
        used_layers = {segment.gi for segment in primary.segs}
        for via in primary.vias:
            used_layers.update((via.lo, via.lo + 1))
        unavailable = tuple(sorted(layer for layer in used_layers if layer not in allowed))
        if unavailable:
            self._mirror_failure_details = (
                ("mirror_failure_kind", "layer_set_mismatch"),
                ("mirror_unavailable_layers", unavailable),
            )
            return None
        width_mismatches = tuple(sorted(
            layer
            for layer in used_layers
            if (
            self._hw(constraint.net_a, layer)
            != self._hw(constraint.net_b, layer)
            )
        ))
        if width_mismatches:
            self._mirror_failure_details = (
                ("mirror_failure_kind", "width_mismatch"),
                ("mirror_width_mismatch_layers", width_mismatches),
            )
            return None

        # The primary is already committed.  Testing the reflected partner in
        # that exact world reserves the second corridor against fixed artwork,
        # prior routes, the routing envelope, via cuts and landing pads.
        self._rebuild_metal_index()
        for segment in partner.segs:
            if not self._wire_legal(
                segment.gi,
                segment.ax,
                segment.ay,
                segment.bx,
                segment.by,
                partner.name,
            ):
                self._mirror_failure_details = (
                    ("mirror_failure_kind", "reflected_segment_illegal"),
                    (
                        "mirror_illegal_segment",
                        (segment.gi, segment.ax, segment.ay, segment.bx, segment.by),
                    ),
                )
                return None
        for via in partner.vias:
            if (
                via.lo not in allowed
                or via.lo + 1 not in allowed
                or not self._via_legal(
                    via.x,
                    via.y,
                    via.lo,
                    partner.name,
                    self._site_n(via),
                )
            ):
                self._mirror_failure_details = (
                    ("mirror_failure_kind", "reflected_via_illegal"),
                    (
                        "mirror_illegal_via",
                        (via.x, via.y, via.lo, via.n),
                    ),
                )
                return None
        if self._candidate_mim_metal_failure(
            partner.name, partner.segs, partner.vias, prior=_RoutedNet(partner.name),
        ):
            self._mirror_failure_details = (
                ("mirror_failure_kind", "reflected_mim_metal_clearance"),
            )
            return None
        return partner

    def _mirror_report(
        self,
        constraint: Mirror,
        primary: _RoutedNet,
        partner: _RoutedNet,
    ) -> dict:
        """Compare final emitted rectangles, including via pads and cuts."""
        def emitted_rectangles(route: _RoutedNet) -> Counter:
            result: Counter = Counter()
            for polygon in self._emit_net(route).wires:
                points = tuple(
                    (int(point.x), int(point.y))
                    for point in polygon.polygon.vertices
                )
                xs = tuple(point[0] for point in points)
                ys = tuple(point[1] for point in points)
                result[
                    (
                        (int(polygon.layer), int(polygon.datatype)),
                        (min(xs), min(ys), max(xs), max(ys)),
                    )
                ] += 1
            return result

        axis2 = self._mirror_axis_twice_nm(constraint)
        primary_rects = emitted_rectangles(primary)
        expected: Counter = Counter()
        for (layer, (x0, y0, x1, y1)), count in primary_rects.items():
            expected[(layer, (axis2 - x1, y0, axis2 - x0, y1))] += count
        actual = emitted_rectangles(partner)
        intersection = sum((expected & actual).values())
        union = sum((expected | actual).values())
        satisfied = bool(
            primary.connected
            and partner.connected
            and expected == actual
        )
        return {
            "net_a": str(constraint.net_a),
            "net_b": str(constraint.net_b),
            "axis_x_um": float(constraint.axis_x_um),
            "axis_x_realized_um": float(constraint.axis_x_um),
            "axis_snap_um": 0.0,
            "mode": "exact-mirror" if satisfied else "deviating",
            "n_rects_a": sum(primary_rects.values()),
            "n_rects_b": sum(actual.values()),
            "n_mirror_match": intersection,
            "deviation_frac": (
                0.0 if union == 0 else 1.0 - intersection / union
            ),
            "satisfied": satisfied,
        }

    @classmethod
    def _materialized_mirror_report(
        cls,
        constraint: Mirror,
        primary: NetRoute,
        partner: NetRoute,
    ) -> dict:
        """Compare the exact rectangles emitted by two allocated programs.

        Resource-plan replay does not own an A* path that can be reflected after
        selection.  Its hard mirror contract is therefore the closed emitted
        geometry itself: every track slab, landing pad, and via cut selected for
        the partner must equal the reflection of the primary program.
        """

        def emitted_rectangles(route: NetRoute) -> Counter:
            result: Counter = Counter()
            for polygon in route.wires:
                points = tuple(
                    (int(point.x), int(point.y))
                    for point in polygon.polygon.vertices
                )
                if not points:
                    continue
                xs = tuple(point[0] for point in points)
                ys = tuple(point[1] for point in points)
                result[
                    (
                        (int(polygon.layer), int(polygon.datatype)),
                        (min(xs), min(ys), max(xs), max(ys)),
                    )
                ] += 1
            return result

        axis2 = cls._mirror_axis_twice_nm(constraint)
        primary_rects = emitted_rectangles(primary)
        expected: Counter = Counter()
        for (layer, (x0, y0, x1, y1)), count in primary_rects.items():
            expected[(layer, (axis2 - x1, y0, axis2 - x0, y1))] += count
        actual = emitted_rectangles(partner)
        intersection = sum((expected & actual).values())
        union = sum((expected | actual).values())
        satisfied = bool(
            primary.connected
            and partner.connected
            and expected == actual
        )
        return {
            "net_a": str(constraint.net_a),
            "net_b": str(constraint.net_b),
            "axis_x_um": float(constraint.axis_x_um),
            "axis_x_realized_um": float(constraint.axis_x_um),
            "axis_snap_um": 0.0,
            "mode": "exact-mirror" if satisfied else "deviating",
            "n_rects_a": sum(primary_rects.values()),
            "n_rects_b": sum(actual.values()),
            "n_mirror_match": intersection,
            "deviation_frac": (
                0.0 if union == 0 else 1.0 - intersection / union
            ),
            "satisfied": satisfied,
        }

    @staticmethod
    def _materialized_length_report(
        constraint: Length,
        routes: Mapping[str, NetRoute],
    ) -> dict:
        """Measure one hard length pair from the exact rectangles to be emitted.

        The metric deliberately matches ``RoutedGeometry.routed_length`` used by
        intent signoff: every tagged wire/pad/cut rectangle contributes its
        long-axis run.  The router therefore cannot satisfy one private
        centre-line estimate and later fail the exact-GDS obligation.
        """

        tolerance = float(constraint.tol_rel)
        if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
            raise ValueError("length tolerance must be finite in [0, 1)")

        def routed_length(net: str) -> int:
            route = routes.get(str(net))
            if route is None or not route.connected or not route.wires:
                return 0
            total = 0
            for polygon in route.wires:
                points = tuple(
                    (int(point.x), int(point.y))
                    for point in polygon.polygon.vertices
                )
                if not points:
                    continue
                xs = tuple(point[0] for point in points)
                ys = tuple(point[1] for point in points)
                total += max(max(xs) - min(xs), max(ys) - min(ys))
            return int(total)

        length_a = routed_length(str(constraint.net_a))
        length_b = routed_length(str(constraint.net_b))
        denominator = max(length_a, length_b)
        imbalance = (
            1.0
            if denominator <= 0
            else abs(length_a - length_b) / denominator
        )
        return {
            "net_a": str(constraint.net_a),
            "net_b": str(constraint.net_b),
            "length_a_nm": length_a,
            "length_b_nm": length_b,
            "relative_imbalance": imbalance,
            "satisfied": bool(
                denominator > 0
                and length_a > 0
                and length_b > 0
                and imbalance <= tolerance
            ),
        }

    @staticmethod
    def _validate_trunk_layers(
        constraints: Sequence[RouteConstraint], tech: _Tech,
    ) -> None:
        """Resolve every advisory before either negotiated or replay routing.

        The exact resource-plan path intentionally ignores a valid SOFT trunk
        preference, but it must not silently label an invalid layer as handled.
        """
        for constraint in constraints:
            if not isinstance(constraint, TrunkLayer):
                continue
            if tech.resolve_layer_to_gi(constraint.layer) is None:
                raise TrunkLayerResolutionError(
                    f"trunk layer for {constraint.net!r} is absent from "
                    f"{tech.pdk.pdk_id}: {constraint.layer!r}"
                )

    def __init__(
        self,
        *,
        time_budget_sec: float = 1800.0,
        max_iters: int = 40,
        pres_fac0: float = 2.0,
        pres_fac_mult: float = 2.0,
        hist_fac: float = 1.0,
        fail_cap: int = 2,
        max_ripups: int = 200,
        per_net_rip_budget: int = 6,
        max_expansions: int = 400_000,
        expansion_cap: int = 8_000_000,
        via_redundancy: int = 1,
        debug: bool = False,
        prefer_target_search: bool = False,
        allow_single_via_escape: bool = False,
    ) -> None:
        if type(prefer_target_search) is not bool:
            raise TypeError("prefer_target_search must be an explicit bool")
        if type(allow_single_via_escape) is not bool:
            raise TypeError("allow_single_via_escape must be an explicit bool")
        self.prefer_target_search = prefer_target_search
        self.allow_single_via_escape = allow_single_via_escape
        self.time_budget_sec = float(time_budget_sec)
        self.max_iters = max(1, int(max_iters))
        self.pres_fac0 = float(pres_fac0)
        self.pres_fac_mult = float(pres_fac_mult)
        self.hist_fac = float(hist_fac)
        self.fail_cap = int(fail_cap)
        self.max_ripups = int(max_ripups)
        self.per_net_rip_budget = int(per_net_rip_budget)
        self.max_expansions = int(max_expansions)
        # Hard ceiling on the per-edge A* expansion budget (see ``_route_edge``).
        self.expansion_cap = int(expansion_cap)
        # VIA-REDUNDANCY (declared via @route_policy(via_redundancy=N)): draw an
        # N x N cut ARRAY at every layer transition instead of a single via -- the
        # reference's EM / contact-resistance / open-defect reliability practice
        # (measured: reference median 4 cuts/connection, min 2; a single-via
        # router leaves median 1). The policy parser hands the value through as a
        # float; coerce to a >=1 int. N=1 (the undeclared default) -> arr=0 in the
        # pad-grow + cut helpers -> a single cut per site, byte-identical to the
        # pre-redundancy path. The array is DRC-legal by construction (see
        # ``route()``: pad grown to enclose the whole array + enclosure; cuts at
        # exactly via{k}_space), and if the grown footprint cannot clear, the
        # via's normal legality rejects that placement (recorded honestly, never a
        # silent over-stuff).
        self.via_redundancy = max(1, int(round(float(via_redundancy))))
        self.debug = bool(debug)
        # cut rects placed by the CURRENT net attempt (escape vias + earlier MST
        # edges), keyed by gi_low -- they are not in the book yet, but the deck's
        # net-blind via space() judges them; reset per ``_route_net`` attempt.
        self._attempt_cuts: Dict[int, List[Rect]] = {}
        # ARTWORK via cuts (cut-layer obstacles: passive-stamp risers), keyed by
        # gi_low; populated at ``route()`` ingestion. Net-blind participants of
        # the VIA{k}.S.1 model exactly like committed / attempt cuts.
        self._obs_cuts: Dict[int, List[Rect]] = {}
        # Immutable fixed-artwork cuts are indexed once by ``route()`` ingestion.
        # Direct helper tests may replace ``_obs_cuts`` without this companion;
        # their exact full-scan fallback is retained by ``_local_fixed_cuts``.
        self._obs_cut_idx: Dict[int, _SpatialIndex] = {}
        # Exact provenance is parallel to ``_obs_cuts`` so the residual hot path
        # and legacy rect-only unit states stay byte-compatible.  A fixed cut that
        # seals a terminal can then be named in a typed STOP instead of becoming
        # an anonymous ``<unowned>`` blocker.
        self._obs_cut_provenance: Dict[
            Tuple[int, Rect],
            Tuple[Optional[str], Optional[str], Optional[str]],
        ] = {}
        # Canonical exact-nm literal world retained for the deck-connected
        # isolation oracle.  It is observational only: above-graph layers never
        # enter route indexes or A* legality.
        self._fixed_deck_literal_rects: Tuple[
            Tuple[Tuple[int, int], Optional[str], Rect], ...
        ] = ()
        self._fixed_wide_source_key = None
        self._fixed_wide_geometry_indexes = {}
        self._wide_geometry_entries_by_gi = {}
        self._wide_geometry_indexes = {}
        self._wide_index_metal_index = None
        self._wide_geometry_index_builds = 0
        # Canonical FEOL contact cuts are not routing-stack connections, but a
        # PDK M1 line-end rule consumes their exact rectangles during final
        # route allocation.  Keep them separate from deck-isolation literals.
        self._fixed_contact_rects: Tuple[Rect, ...] = ()
        # metal rects placed by the CURRENT net attempt's EARLIER MST edges
        # (+ spatial index, rebuilt per edge). A net's own in-progress metal is
        # invisible to hard legality by design (edges connect through shared pin
        # nodes), but the deck's space() is net-blind: successive edges taking
        # near-parallel tracks at 5-250nm offsets leave same-net corner gaps the
        # deck flags and rip-up reproduces (the measured 11-residual livelock:
        # jittered duplicate trunk runs). Banded proximity to own attempt metal
        # is therefore PRICED like the own-artwork band -- overlap (merge) stays
        # free, so the A* aligns onto its own earlier track instead of jittering
        # beside it.
        self._attempt_metal: Dict[int, List[Rect]] = {}
        self._attempt_idx: Dict[int, _SpatialIndex] = {}
        # Conflict-directed retries may exclude an exact via site/model from one
        # search branch.  This is solve-local state, never a candidate cap: every
        # complementary branch remains represented until success or typed watchdog.
        self._search_forbidden_via_sites: Set[
            Tuple[int, int, int, int]
        ] = set()
        # Committed via-pad ownership is an engine invariant, including before
        # ``route()`` prepares a routing world (direct helper/replay calls see an
        # empty world rather than an absent safety index).
        self._via_pad_idx: Dict[int, _SpatialIndex] = {}
        self._via_pad_provenance: Dict[
            int, frozenset[Tuple[Optional[str], Rect]]
        ] = {}
        self._cut_idx: Dict[int, _SpatialIndex] = {}
        # One immutable pre-scan may be shared by the immediately adjacent A*
        # legality and static-price calls.  It is populated only inside one
        # ``_route_edge`` transition evaluation and restored immediately after.
        self._active_static_metal_scans = None
        self._active_static_cut_scans = None
        self._active_static_obstacle_scans = None
        # model-only keep-out rects per gi (populated at ``route()`` ingestion
        # from ``Obstacle.kind``); never credited as drawn metal in the
        # emitted-geometry judgments. Empty for hand-built engine states.
        self._abstract_rects: Dict[int, Set[Rect]] = {}
        # Claim-specific index of literally drawn planar fixed artwork.  Unlike
        # ``_abstract_rects`` this keeps a real conductor when an abstraction has
        # the same bbox; final electrical contact must be decided per source claim,
        # not by a rect-only summary. Empty for hand-built engine states.
        self._literal_planar_idx: Dict[int, _SpatialIndex] = {}
        # Canonical, read-only projection of exact drawn owned metal.  It is
        # populated only by the unplanned route-world ingestion and never imports
        # CandidatePool / OwnedConductorComponentIndex evidence.
        self._owned_terminal_atoms: Tuple[_OwnedTerminalAtom, ...] = ()
        self._owned_terminal_group_indexes: Dict[
            Tuple[str, int],
            Tuple[
                Tuple[_OwnedTerminalAtom, ...],
                Tuple[_OwnedTerminalAtom, ...],
                _SpatialIndex,
                Dict[Rect, int],
            ],
        ] = {}
        # Immutable route-world artwork/fixed-cut frontiers used by exact owned
        # component via access.  This is only a query-result accelerator: mutable
        # attempt/committed cuts are overlaid on every access, and ``route()``
        # clears it whenever it ingests a new world.
        self._owned_atom_static_via_events: Dict[
            Tuple[Any, ...], Tuple[Tuple[int, ...], Tuple[int, ...]]
        ] = {}
        # Dynamic access query reuse has one solve-local owner. Geometry and
        # search policy stay here; keys explicitly bind their mutable world.
        self._committed_index_generation = 0
        self._component_access: ComponentAccessSession[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ] = ComponentAccessSession(max_expansions=self.max_expansions)
        self._history_generation = 0
        self._via_static_legality_cache: OrderedDict[
            Tuple[Any, ...], Any
        ] = OrderedDict()
        self._wire_static_legality_cache: OrderedDict[
            Tuple[Any, ...], Any
        ] = OrderedDict()
        self._via_static_legality_cache_hits_total = 0
        self._via_static_legality_cache_misses_total = 0
        self._wire_static_legality_cache_hits_total = 0
        self._wire_static_legality_cache_misses_total = 0
        # per-net emitted HALF-WIDTH (nm, per gi) materialized from the received
        # ``Width`` constraints (``route()``); a net absent here draws at the
        # layer min half-width (``self._half_w``) -> the no-Width path stays
        # byte-identical. Consumed uniformly by every drawn-footprint helper via
        # ``_hw``. Empty for hand-built engine states.
        self._net_half_w: Dict[str, List[int]] = {}
        # DECLARED SUPPLY RAILS (materialized @trunk(edge=) spines): net -> list of
        # (gi_rail, rail_rect_nm). Empty until one routing world is ingested.
        self._rail_by_net: Dict[str, List[Tuple[int, Rect]]] = {}
        # Direct helper/replay calls use the same deterministic work ceiling as a
        # full route. ``route()`` resets both at the start of each solve.
        self._solve_expansions_total = 0
        self._solve_expansion_budget = self.max_iters * self.expansion_cap
        self._watchdog_deadline = None
        self._astar_watchdog_details: Tuple[Tuple[str, Any], ...] = ()
        self._access_watchdog_details: Tuple[Tuple[str, Any], ...] = ()
        self._route_watchdog_details: Tuple[Tuple[str, Any], ...] = ()
        self._potential_transform_switches_total = 0
        # Exact obstacle-free goal potentials depend only on the irregular graph
        # axes, allowed layers, and terminal lower bounds.  Negotiated reroutes
        # repeatedly revisit those identical logical edges while only blockers and
        # history change.  Retain an LRU working set bounded by the already-declared
        # nominal A* state authority; this avoids rebuilding the separable transform
        # without introducing an independent cache-size knob or changing costs.
        self._exact_potential_cache: OrderedDict[
            Tuple[Any, ...], Tuple[int, List[int]]
        ] = OrderedDict()
        self._exact_potential_cache_elements = 0
        self._potential_transform_cache_hits_total = 0
        self._potential_transform_cache_misses_total = 0

    def _watchdog_check(self, net: str, phase: str) -> None:
        """Abort without returning geometry when the declared time ceiling expires.

        Deterministic expansion/iteration budgets continue to decide every emitted
        shape.  This wall-clock check is only an outer safety stop: it raises a
        typed value and therefore cannot select or emit a timing-dependent partial
        snapshot.
        """
        session = current_session()
        if session is not None:
            session.checkpoint()
        deadline = getattr(self, "_watchdog_deadline", None)
        if deadline is None or time.monotonic() < deadline:
            return
        phase_name = str(phase)
        # The active access transaction appends its complete endpoint domains to
        # ``_route_failure_details``.  Those can contain thousands of records and
        # are already summarized by the access/A* progress maps.  A watchdog stop
        # keeps the stable route obligation identity without duplicating that
        # unbounded payload in the fail-close ledger.
        transaction_payload_keys = {
            "access_attempts",
            "access_failures",
            "forbidden_via_sites",
            "goal_access",
            "source_access",
        }
        obligation_details = tuple(
            (key, value)
            for key, value in getattr(self, "_route_failure_details", ())
            if key not in transaction_payload_keys
        )
        # A causal-frontier A* replay is selected from these exact blocker
        # aggregates.  Omitting them from a watchdog stop leaves only grid-size
        # counters and makes the next bounded repair axis unprovable.  The rows
        # are already canonical and collapse all repeated candidate hits by exact
        # blocker identity, so retaining the complete set does not copy the
        # unbounded endpoint-transaction inventories excluded above.
        causal_details = []
        if getattr(self, "_active_edge_ctx", None) is not None:
            self._flush_active_edge_ctx()
        rejections = self._hard_rejection_evidence()
        rejection_samples = self._hard_rejection_sample_evidence()
        if rejections:
            causal_details.append(("rejections", rejections))
        if rejection_samples:
            causal_details.append(("rejection_samples", rejection_samples))
        obligation_details += tuple(causal_details)
        if phase_name.startswith("astar"):
            # A* commonly executes inside the combined endpoint-access
            # transaction.  Preserve that transaction's deterministic domain
            # cursor in the typed stop so a timeout distinguishes one expensive
            # search from conflict-directed domain replay.  Only merge the
            # details while the transaction is actively inside its callback;
            # otherwise they may describe a completed earlier edge.
            access_details = tuple(
                getattr(self, "_access_watchdog_details", ())
            )
            if dict(access_details).get("subphase") != "route_attempt":
                access_details = ()
            phase_details = obligation_details + access_details + tuple(
                getattr(self, "_astar_watchdog_details", ())
            )
        elif phase_name == "access_selection":
            phase_details = obligation_details + tuple(
                getattr(self, "_access_watchdog_details", ())
            )
        elif phase_name.startswith("component_access"):
            query_stats = self._component_access.stats
            phase_details = obligation_details + (
                (
                    "component_access_partition_cache_hits_total",
                    int(query_stats.partition_hits),
                ),
                (
                    "component_access_partition_cache_misses_total",
                    int(query_stats.partition_misses),
                ),
                (
                    "component_access_via_cache_hits_total",
                    int(query_stats.site_hits),
                ),
                (
                    "component_access_via_cache_misses_total",
                    int(query_stats.site_misses),
                ),
                (
                    "component_escape_state_cache_hits_total",
                    int(query_stats.escape_hits),
                ),
                (
                    "component_escape_state_cache_misses_total",
                    int(query_stats.escape_misses),
                ),
                (
                    "via_static_legality_cache_hits_total",
                    int(self._via_static_legality_cache_hits_total),
                ),
                (
                    "via_static_legality_cache_misses_total",
                    int(self._via_static_legality_cache_misses_total),
                ),
                (
                    "wire_static_legality_cache_hits_total",
                    int(self._wire_static_legality_cache_hits_total),
                ),
                (
                    "wire_static_legality_cache_misses_total",
                    int(self._wire_static_legality_cache_misses_total),
                ),
            )
        else:
            phase_details = ()
        detail_map = dict(getattr(self, "_route_watchdog_details", ()))
        detail_map.update(dict(phase_details))
        details = tuple(sorted(detail_map.items(), key=lambda item: item[0]))
        raise NegotiationWatchdogStop(NegotiationRouteStop(
            NegotiationStopReason.WALL_CLOCK_WATCHDOG,
            str(net),
            str(phase),
            (int(round(max(0.0, self.time_budget_sec) * 1000.0)),),
            details,
        ))

    # ------------------------------------------------------------- width table
    def _hw(self, net: str, gi: int) -> int:
        """Emitted HALF-WIDTH (nm) for ``net`` on graph layer ``gi``: the per-net
        ``Width``-derived half-width when the net carries a (supra-min) Width,
        else the layer min half-width (``self._half_w``).

        ONE number feeds hard legality, congestion, occupancy AND the emitted
        polygon (the module's single-emitted-geometry rule), so a widened rail
        also RESERVES its true clearance and cannot silently short a neighbour.
        The min-width DRC obligations (junction-width, merge fills, spacing
        swaths) deliberately keep ``self._half_w`` -- that is the PDK min feature
        size, not the drawn wire width."""
        override = getattr(self, "_active_half_width_overrides", {}).get(
            (str(net), int(gi))
        )
        if override is not None:
            return max(self._half_w[gi], int(override))
        nw = self._net_half_w.get(net)
        if nw is not None:
            return nw[gi]
        return self._half_w[gi]

    def _segment_half_width(self, net: str, seg: _WireSeg) -> int:
        """Exact emitted/search half-width carried by ``seg``."""
        explicit = int(getattr(seg, "half_width_nm", 0))
        if explicit > 0:
            return max(self._half_w[seg.gi], explicit)
        return self._hw(net, seg.gi)

    def _wire_segment(
        self,
        net: str,
        gi: int,
        ax: int,
        ay: int,
        bx: int,
        by: int,
    ) -> _WireSeg:
        """Create a segment and freeze only an active local-width override."""
        explicit = 0
        if (str(net), int(gi)) in getattr(
            self, "_active_half_width_overrides", {}
        ):
            explicit = self._hw(net, gi)
        return _WireSeg(gi, ax, ay, bx, by, explicit)

    def _width_half_table(
        self, constraints: Sequence[RouteConstraint], half_w: List[int],
        n_layers: int,
        *,
        carrier_gi_by_net: Optional[Mapping[str, int]] = None,
    ) -> Tuple[Dict[str, List[int]], Set[str], Dict[str, Width]]:
        """Build the per-net emitted half-width table from the received ``Width``
        constraints. Returns ``(net_half_w, widened_nets, cons_by_net)``:

          * a plain Width has the historical uniform semantics:
            ``net_half_w[net][gi] = max(layer min half-width,
            round(width_um * _NM_PER_UM / 2))``;
          * when ``carrier_gi_by_net`` names the effective layer of a declared
            ``TrunkLayer``, only that structural trunk carrier is widened and all
            local access/transport layers retain their PDK minimum half-width.
            This matches the exact shared-TRUNK/STAR materializer, whose local
            branches and staircases are explicitly minimum-width;
          * a Width AT OR BELOW min-width floors at min (never widened DOWN), so
            it stays byte-identical to the no-Width path; only a SUPRA-min Width
            widens a net's drawn footprint.
          * ``widened_nets`` -- nets whose target exceeds min on some layer (the
            ones whose materialization is verifiable / reportable).
          * ``cons_by_net`` -- the (max-width) constraint object per net, so an
            unmaterialized Width can be reported ``unsatisfied`` verbatim.

        Pure function of the constraints + PDK half-widths + structural carrier
        map: no net-name / circuit branch, deterministic, no wall-clock."""
        width_um_by_net: Dict[str, float] = {}
        cons_by_net: Dict[str, Width] = {}
        for c in constraints:
            if isinstance(c, Width):
                w = float(c.width_um)
                prev = width_um_by_net.get(c.net)
                if prev is None or w > prev:
                    width_um_by_net[c.net] = w
                    cons_by_net[c.net] = c
        net_half_w: Dict[str, List[int]] = {}
        widened: Set[str] = set()
        for net, w_um in width_um_by_net.items():
            target_half = round(w_um * _NM_PER_UM / 2)
            carrier_gi = (carrier_gi_by_net or {}).get(net)
            if carrier_gi is None:
                per_gi = [
                    max(half_w[gi], target_half) for gi in range(n_layers)
                ]
            else:
                if not 0 <= int(carrier_gi) < n_layers:
                    raise ValueError("width carrier layer is outside routing stack")
                per_gi = list(half_w)
                per_gi[int(carrier_gi)] = max(
                    half_w[int(carrier_gi)], target_half
                )
            net_half_w[net] = per_gi
            if any(per_gi[gi] > half_w[gi] for gi in range(n_layers)):
                widened.add(net)
        return net_half_w, widened, cons_by_net

    # ---------------------------------------------------------------- contract
    def route(
        self,
        pins_by_net: Dict[str, List[Pin]],
        obstacles: Sequence[Obstacle],
        constraints: Sequence[RouteConstraint],
        pdk,
        *,
        preroutes: Optional[Dict[str, list]] = None,
        rails: Optional[Dict[str, Sequence[Tuple[Any, Tuple[float, float, float, float]]]]] = None,
        resource_plan: Optional[object] = None,
        diagnostic_resource_program: Optional[object] = None,
        diagnostic_full_pins_by_net: Optional[Dict[str, List[Pin]]] = None,
        routing_envelope_nm: Optional[object] = None,
        group_route_contract: Optional[object] = None,
        group_demand_topology_plan: Optional[object] = None,
    ) -> RoutedCell:
        if getattr(self, '_experimental_fixed_index_seal_budget', None) is not None:
            from clay.layout.route.fixed_index_seal import FixedIndexSealStop, _Scope
            scope = getattr(self, '_fixed_index_seal_scope', None)
            if type(scope) is not _Scope or scope.engine is not self:
                raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
            scope.enter_route()
        # Reset on every call: engines may be reused for unrelated cells.
        self._intent_spacing_rules = compile_spacing_rules(constraints, pins_by_net)
        routing_envelope = (
            None
            if routing_envelope_nm is None
            else RoutingEnvelope.coerce(routing_envelope_nm)
        )
        self._routing_envelope = routing_envelope
        if resource_plan is not None and diagnostic_resource_program is not None:
            raise TypeError(
                "resource_plan and diagnostic_resource_program are mutually exclusive"
            )
        if group_route_contract is not None:
            from clay.layout.route.group_ports import GroupRouteRoutingContract

            if type(group_route_contract) is not GroupRouteRoutingContract:
                raise TypeError("group-route routing contract has a foreign type")
            if resource_plan is not None or diagnostic_resource_program is not None:
                raise TypeError(
                    "group-route landing allocation is unavailable in resource-plan replay"
                )
        if group_demand_topology_plan is not None and (
                group_route_contract is None or
                not getattr(self, '_experimental_group_demand_guidance', False)):
            raise TypeError('group demand topology needs the enabled group-route consumer')
        if diagnostic_resource_program is not None:
            self._preflight_diagnostic_resource_program(
                diagnostic_resource_program
            )
            tech = _Tech(pdk)
            self._manufacturing_grid_nm = tech.manufacturing_grid_nm
            self._validate_trunk_layers(constraints, tech)
            try:
                materialized = self._materialize_resource_plan(
                    pins_by_net,
                    constraints,
                    pdk,
                    tech,
                    diagnostic_resource_program,
                    obstacles=obstacles,
                    rails=rails,
                    preroutes=preroutes,
                    diagnostic=True,
                    diagnostic_full_pins_by_net=diagnostic_full_pins_by_net,
                )
            except RouteResourcePlanIncompleteError:
                raise
            except Exception as exc:
                raise RouteResourcePlanIncompleteError(
                    self, diagnostic_resource_program
                ) from exc
            if routing_envelope is not None:
                require_routed_containment(materialized, routing_envelope)
                materialized.routing_envelope_nm = routing_envelope.bbox_nm
            return materialized
        if diagnostic_full_pins_by_net is not None:
            raise TypeError(
                "diagnostic_full_pins_by_net requires diagnostic_resource_program"
            )
        self._preflight_resource_plan(resource_plan)
        require_resource_plan_support(self, resource_plan)
        tech = _Tech(pdk)
        self._manufacturing_grid_nm = tech.manufacturing_grid_nm
        self._validate_trunk_layers(constraints, tech)
        if resource_plan is not None and resource_plan.selected_nets:
            try:
                materialized = self._materialize_resource_plan(
                    pins_by_net,
                    constraints,
                    pdk,
                    tech,
                    resource_plan,
                    obstacles=obstacles,
                    rails=rails,
                    preroutes=preroutes,
                )
            except RouteResourcePlanIncompleteError:
                raise
            except Exception as exc:
                try:
                    self._raise_plan_incomplete(resource_plan)
                except RouteResourcePlanIncompleteError as error:
                    raise error from exc
            if routing_envelope is not None:
                require_routed_containment(materialized, routing_envelope)
                materialized.routing_envelope_nm = routing_envelope.bbox_nm
            return materialized
        fixed_conflicts = fixed_spacing_conflicts(
            pins_by_net, constraints, obstacles, _Tech(pdk), rails,
        ) if self._intent_spacing_rules else []
        if fixed_conflicts:
            out = RoutedCell(nets={net: NetRoute(net, pins=list(pins), connected=False)
                                   for net, pins in pins_by_net.items()},
                             unsatisfied=[c for c, _ in fixed_conflicts], success=False)
            out.spacing_constraints = tuple(c for c in constraints if isinstance(c, Spacing))
            out.spacing_reports = [report for _, report in fixed_conflicts]
            out.route_stops = tuple(NegotiationRouteStop(
                NegotiationStopReason.NO_LEGAL_ROUTE, c.net_a, "fixed_spacing", (),
                (("net_pair", (c.net_a, c.net_b)), ("required_nm", report["required_gap_nm"]),
                 ("minimum_gap_nm", report["minimum_gap_nm"]),
                 ("closest_boxes", report["closest_boxes"])),
            ) for c, report in fixed_conflicts)
            return out
        self._watchdog_deadline = (
            time.monotonic() + max(0.0, self.time_budget_sec)
        )
        self._astar_watchdog_details = ()
        self._access_watchdog_details = ()
        self._route_watchdog_details = ()
        self._potential_transform_switches_total = 0
        self._exact_potential_cache = OrderedDict()
        self._exact_potential_cache_elements = 0
        self._potential_transform_cache_hits_total = 0
        self._potential_transform_cache_misses_total = 0
        # The unplanned feasibility path owns its ordering.  Canonicalizing here
        # leaves exact resource-plan replay byte-for-byte isolated above while
        # making MST roots, A* sources, bounds and emitted net order independent
        # of caller dict/list insertion order.
        pins_by_net = {
            str(net): sorted(list(pins), key=_pin_order_key)
            for net, pins in sorted(pins_by_net.items(), key=lambda item: str(item[0]))
        }
        if pins_by_net:
            from clay.layout.route.deck_isolation import _uppercase_net_set
            _uppercase_net_set(
                tuple(pins_by_net), source="Gridless negotiation"
            )
        obstacles = sorted(list(obstacles), key=_obstacle_order_key)
        out = RoutedCell()
        out.mirror_reports = []
        out.length_reports = []
        # two-tier residual reporting (``_space_residual_scan``): the STRICT
        # advisory count/pairs are never silently dropped; deck_residual_pairs
        # is the deck-visible subset that gates ``success``.
        out.advisory_residuals = 0
        out.advisory_residual_pairs = []
        out.deck_residual_pairs = []

        n_layers = tech.n_layers
        # per-graph-layer half-width + spacing (nm), indexed by gi.
        half_w = [tech.gi_min_width(gi) // 2 for gi in range(n_layers)]
        space = [int(tech.min_space_nm[tech._gi_to_pdk[gi]]) for gi in range(n_layers)]
        # via geometry per gi_low: cut size, per-side enclosure, pad half on lo/hi.
        via_cut: Dict[int, int] = {}
        via_pad_half: Dict[int, Tuple[int, int]] = {}  # (below-half, above-half)
        # SINGLE-via (N=1) pad halves per level -- the fallback pad for a sealed-pin
        # escape when the declared N x N reliability array cannot land (dense pack).
        # Equal to via_pad_half when the engine default is 1 (arr == 0).
        via_pad_half1: Dict[int, Tuple[int, int]] = {}
        via_space: Dict[int, int] = {}
        via_pitch: Dict[int, int] = {}  # redundant-array cut centre-to-centre (nm)
        # The PDK CONTRACT surface (drc_rules / via_rule / via_size) lives on the
        # underlying LayoutTech, NOT on the ``_Tech`` routing view -- passing the
        # view to ``via_model`` silently degraded every enclosure to (0, 0), the
        # R7a flush-pad defect family (VIA1.EN.1 x26 + M2.W.1/M3.W.1 on the real
        # open_pdk deck). Same resolution ``engine_maze._via_env_rule_nm`` uses.
        contract = getattr(tech, "pdk", tech)
        grid = max(1, int(self._manufacturing_grid_nm))

        def landing_half(gi: int, enclosure_half: int) -> int:
            layer_name = tech.metal_names[tech._gi_to_pdk[gi]]
            area_side = minimum_area_square_side_nm(contract.min_area(layer_name))
            half = max((tech.gi_min_width(gi) + 1) // 2,
                       (area_side + 1) // 2, enclosure_half)
            return ((half + grid - 1) // grid) * grid

        N = self.via_redundancy
        for lo in range(n_layers - 1):
            cut = _via_cut_size_nm(tech, lo)
            via_cut[lo] = cut
            vname = _via_name_for(tech, lo)
            if vname is not None:
                rb, ra = via_enclosure_nm(contract, vname)
            else:
                rb = ra = 0
            ch = (cut + 1) // 2
            # cut-to-cut spacing: the deck's via{k}_space rule when the PDK
            # declares one; lower metal min-space stays the last-resort fallback.
            vsp = _via_cut_space_nm(contract, vname, space[lo])
            via_space[lo] = vsp
            # VIA-REDUNDANCY array pitch: cut + via{k}_space is the centre-to-centre
            # step of adjacent cuts, so their edge-to-edge gap is exactly the
            # via-space rule (DRC-legal). Force the pitch EVEN so an N x N array of
            # integer cut centres stays EXACTLY symmetric about the via site (the
            # enclosure is then identical on all four sides); rounding the pitch UP
            # only widens the gap, never below the rule.
            pitch = cut + vsp
            if pitch % 2:
                pitch += 1
            via_pitch[lo] = pitch
            # site centre -> outermost cut centre of the N x N array (0 when N=1).
            arr = (N - 1) * pitch // 2
            # Each layer's landing satisfies its own minimum width and area,
            # encloses the complete cut array, and rounds its half-size outward
            # to the manufacturing grid. One per-side model feeds legality,
            # congestion, occupancy and emitted polygons. The N=1 fallback uses
            # the same contract without the array extent; cut count is unchanged.
            via_pad_half[lo] = (
                landing_half(lo, arr + ch + int(rb)),
                landing_half(lo + 1, arr + ch + int(ra)),
            )
            # SINGLE-via pad (arr == 0): the escape fallback's landing footprint.
            via_pad_half1[lo] = (
                landing_half(lo, ch + int(rb)),
                landing_half(lo + 1, ch + int(ra)),
            )

        self._tech = tech
        self._n_layers = n_layers
        self._half_w = half_w
        self._space = space
        self._via_cut = via_cut
        self._via_pad_half = via_pad_half
        self._via_pad_half1 = via_pad_half1
        self._via_space = via_space
        self._via_pitch = via_pitch
        self._max_gi = n_layers - 1

        # --- DECLARED SUPPLY RAILS -> net -> [(gi_rail, rail_rect_nm)]. Each entry
        #     is a materialized @trunk(edge=) rail (owned metal seeded into the
        #     obstacle model): a device pin of a rail net drops LOCALLY onto it
        #     (``_route_net_to_rail``) rather than joining pin-to-pin toward a hub.
        #     A layer outside this PDK's routing graph is skipped (honest no-op).
        #     ``rails`` None/empty -> ``_rail_by_net`` empty -> the MST path below
        #     is byte-identical for every net.
        rail_by_net: Dict[str, List[Tuple[int, Rect]]] = {}
        for rnet, rlist in sorted(
            (rails or {}).items(), key=lambda item: str(item[0])
        ):
            acc: List[Tuple[int, Rect]] = []
            for (layer, bbox_um) in rlist or ():
                gi = tech.resolve_layer_to_gi(layer)
                if gi is None:
                    continue
                bx0u, by0u, bx1u, by1u = bbox_um
                acc.append((gi, _norm((
                    round(bx0u * _NM_PER_UM), round(by0u * _NM_PER_UM),
                    round(bx1u * _NM_PER_UM), round(by1u * _NM_PER_UM)))))
            if acc:
                rail_by_net[str(rnet)] = sorted(acc)
        self._rail_by_net = rail_by_net

        # --- constraints: LayerSet + Width + exact Mirror/Length + soft Direction
        layer_set_terms_by_net: Dict[str, List[Tuple[LayerSet, Set[int]]]] = {}
        layer_set_by_net: Dict[str, List[int]] = {}
        layer_set_constraints_by_net: Dict[str, Tuple[LayerSet, ...]] = {}
        declared_trunk_options_by_net: Dict[str, Set[Tuple[int, str]]] = {}
        declared_trunk_by_net: Dict[str, int] = {}
        declared_trunk_axis_by_net: Dict[str, Tuple[int, str]] = {}
        mirror_constraints: List[Mirror] = []
        length_constraints: List[Length] = []
        direction_preference_by_gi: Dict[int, Tuple[str, float]] = {}
        unsatisfied: List[RouteConstraint] = []
        for c in constraints:
            if isinstance(c, LayerSet):
                gis = []
                for lay in c.layers:
                    gi = tech.resolve_layer_to_gi(lay)
                    if gi is not None:
                        gis.append(gi)
                layer_set_terms_by_net.setdefault(c.net, []).append(
                    (c, set(gis))
                )
            elif isinstance(c, Width):
                # MATERIALIZED, not silent-passed: the per-net emitted half-width
                # table is built from every Width below (``_width_half_table``)
                # and drawn through ``_hw``. Honesty (no silent pass) is closed at
                # the end of the solve -- a supra-min Width whose net never
                # materialized it (unconnected, or drew no wire) is reported
                # ``unsatisfied`` there, NOT recorded satisfied here.
                pass
            elif isinstance(c, Mirror):
                mirror_constraints.append(c)
            elif isinstance(c, Spacing):
                # Valid hard pairs have entered the search rule table above.
                # Invalid/missing pairs fail the independent final measurement.
                if c.hardness == "hard" and spacing_nm(c, pins_by_net) is None:
                    unsatisfied.append(c)
            elif isinstance(c, Length):
                tolerance = float(c.tol_rel)
                if (
                    type(c.net_a) is not str
                    or type(c.net_b) is not str
                    or not c.net_a
                    or not c.net_b
                    or c.net_a == c.net_b
                    or c.net_a not in pins_by_net
                    or c.net_b not in pins_by_net
                    or not math.isfinite(tolerance)
                    or tolerance < 0.0
                    or tolerance >= 1.0
                ):
                    unsatisfied.append(c)
                    continue
                length_constraints.append(c)
            elif isinstance(c, TrunkLayer):
                if getattr(c, "hardness", "soft") == "hard":
                    unsatisfied.append(c)
                    continue
                gi = tech.resolve_layer_to_gi(c.layer)
                assert gi is not None  # validated before either route path
                axis = str(c.orient).strip().upper()
                declared_trunk_options_by_net.setdefault(c.net, set()).add(
                    (gi, axis if axis in {"H", "V"} else "")
                )
            elif isinstance(c, Direction):
                axis = str(c.axis).strip().upper()
                weight = float(c.weight)
                if axis not in {"H", "V"} or not math.isfinite(weight) or weight < 0.0:
                    # Malformed advisory input remains typed evidence; a valid
                    # soft preference is consumed below and never becomes a hard
                    # exact failure.
                    unsatisfied.append(c)
                    continue
                if weight == 0.0:
                    continue
                gi = tech._pdk_to_gi.get(int(c.layer_n))
                if gi is not None:
                    # The last declaration for a layer wins, matching the shared
                    # Direction contract used by the other routing engines.
                    direction_preference_by_gi[gi] = (axis, weight)
            else:
                # Shield / TrackOrder / Access / Via /
                # unsupported hard kinds remain honest unsatisfied obligations.
                unsatisfied.append(c)

        mirror_by_b = self._index_mirrors(
            mirror_constraints, pins_by_net, unsatisfied
        )
        mirror_by_a = {
            constraint.net_a: constraint
            for constraint in mirror_by_b.values()
        }
        matched_partners: Dict[str, Set[str]] = {}
        for constraint in (*mirror_constraints, *length_constraints):
            na, nb = str(constraint.net_a), str(constraint.net_b)
            if na and nb and na != nb:
                matched_partners.setdefault(na, set()).add(nb)
                matched_partners.setdefault(nb, set()).add(na)
        self._matched_partners_by_net: Dict[str, Tuple[str, ...]] = {
            net_name: tuple(sorted(mates))
            for net_name, mates in matched_partners.items()
        }
        self._tuned_1pin_baseline_nets: Set[str] = set()

        # Repeated hard LayerSets are a logical conjunction.  Keep an explicit
        # empty result distinct from "no LayerSet" so impossible intersections
        # cannot regain the full routing stack through a truthiness fallback.
        for net, terms in sorted(layer_set_terms_by_net.items()):
            allowed = set(range(n_layers))
            for _constraint, term in terms:
                allowed.intersection_update(term)
            layer_set_by_net[net] = sorted(allowed)
            canonical_constraints = tuple(sorted(
                (constraint for constraint, _term in terms),
                key=lambda constraint: tuple(sorted(
                    repr(layer) for layer in constraint.layers
                )),
            ))
            layer_set_constraints_by_net[net] = canonical_constraints
            if not allowed:
                for constraint in canonical_constraints:
                    if constraint not in unsatisfied:
                        unsatisfied.append(constraint)

        # Soft trunk declarations are advisory alternatives.  Resolve all of
        # them first, then select one canonical (gi, axis) pair; input ordering
        # must not become an implicit style priority.
        for net, options in sorted(declared_trunk_options_by_net.items()):
            gi, axis = min(options)
            declared_trunk_by_net[net] = gi
            if axis:
                declared_trunk_axis_by_net[net] = (gi, axis)

        # --- obstacles -> per-gi foreign/owned rects (nm). Non-metal layers (poly,
        #     active, below M1) do not map to a routing gi -> skipped (the router
        #     routes on metals, passing over sub-M1 artwork freely).
        #     EXCEPTION (R7 tail): CUT-layer artwork (a passive stamp's riser via
        #     cuts). The deck's VIA{k}.S.1 is NET-BLIND over ALL cuts; dropping
        #     these here made them invisible to the cut-spacing legality and the
        #     exact residual detector -- the measured VIA2.S.1 was a routed via2
        #     cut 170nm (< the 260nm via2_space rule) from a MIM riser via2 cut
        #     the engine never saw. Ingested into ``_obs_cuts`` per via level.
        via_lo_by_key: Dict[Tuple[int, int], int] = {}
        for lo in range(n_layers - 1):
            vk = tech.via_key_for(lo)
            if vk is not None:
                via_lo_by_key[tuple(vk)] = lo
        obs_by_gi: Dict[int, List[Tuple[Optional[str], Rect]]] = {
            gi: [] for gi in range(n_layers)
        }
        obs_provenance: Dict[
            Tuple[int, Optional[str], Rect],
            Tuple[Optional[str], Optional[str]],
        ] = {}
        obs_provenance_options: Dict[
            Tuple[int, Optional[str], Rect],
            Set[Tuple[Optional[str], Optional[str]]],
        ] = {}
        planar_claim_options: Dict[
            Tuple[int, Rect],
            Set[Tuple[Optional[str], Optional[str], Optional[str]]],
        ] = {}
        obs_cuts: Dict[int, List[Rect]] = {}
        obs_cut_provenance_options: Dict[
            Tuple[int, Rect],
            Set[Tuple[Optional[str], Optional[str], Optional[str]]],
        ] = {}
        # ROUTING-MODEL ABSTRACTIONS (R7 junction round): obstacles whose
        # provenance kind marks them as model-only keep-outs, NOT drawn artwork
        # (the min-space-BLOATED cap-plate collars / no-route-under bans --
        # ``clay.layout.model.ROUTING_ABSTRACTION_KINDS``). They stay FULL
        # participants of hard legality, congestion pricing and candidate
        # lines, but the emitted-geometry judgments (gap coverage / junction
        # contact / same-net band detection) must not credit them as metal:
        # model-only metal merges nothing in the artifact the deck reads. The
        # measured escape: a net25 merge fill rode its bloated plate collar and
        # only corner point-touched the TRUE drawn plate (deck M4.W.1 x2 +
        # M4.S.1 x2 at one kiss), while the collar 'covered' the junction in
        # the old model world.
        from clay.layout.model import ROUTING_ABSTRACTION_KINDS
        from clay.layout.route.deck_isolation import DeckConnectStack
        deck_stack = DeckConnectStack.from_tech(tech.pdk)
        self._deck_connect_stack = deck_stack
        deck_conductor_layers = set(deck_stack.conductors)
        deck_cut_layers = {item.cut for item in deck_stack.connections}
        deck_marker_layers = {item.top_marker for item in deck_stack.mim_carves}
        # Physical selectors needed by complete-union MIM checks are literal
        # artwork even if they do not participate in electrical connectivity.
        mim_geometry_rules = getattr(tech.pdk, "mim_metal_geometry_rules", None)
        if callable(mim_geometry_rules):
            deck_marker_layers.update(rule.marker_layer for rule in mim_geometry_rules())
        try:
            contact_layer = tuple(tech.pdk.contact_key())
        except Exception:
            contact_layer = None
        fixed_deck_literals: List[
            Tuple[Tuple[int, int], Optional[str], Rect]
        ] = []
        fixed_contact_rects: List[Rect] = []
        abstract_rects: Dict[int, Set[Rect]] = {}
        literal_planar_claims: Dict[
            int, Set[Tuple[Optional[str], Rect]]
        ] = {}
        for ob in obstacles:
            r = (round(ob.x0 * _NM_PER_UM), round(ob.y0 * _NM_PER_UM),
                 round(ob.x1 * _NM_PER_UM), round(ob.y1 * _NM_PER_UM))
            rn_ = _norm(r)
            lk = tuple(ob.layer) if isinstance(ob.layer, tuple) else None
            kind = getattr(ob, "kind", None)
            if kind == "device_contact" and lk == contact_layer:
                fixed_contact_rects.append(rn_)
            if (
                kind not in ROUTING_ABSTRACTION_KINDS
                and kind != "device_projection"
                and lk is not None
            ):
                if lk in deck_conductor_layers:
                    fixed_deck_literals.append((lk, ob.owner_net, rn_))
                elif lk in deck_cut_layers or lk in deck_marker_layers:
                    fixed_deck_literals.append((lk, None, rn_))
            if getattr(ob, "kind", None) == "device_projection":
                # The exact planner owns any active-area utilization objective.
                # A projection is never a hard obstacle: upper metal may pass
                # over device artwork, while landing legality remains governed
                # by exact owned-conductor provenance.
                continue
            gi = tech.resolve_layer_to_gi(ob.layer)
            if gi is not None:
                obs_by_gi[gi].append((ob.owner_net, rn_))
                obs_provenance_options.setdefault(
                    (gi, ob.owner_net, rn_), set()
                ).add(
                    (
                        getattr(ob, "kind", None),
                        getattr(ob, "terminal_conductor_ref", None),
                    )
                )
                planar_claim_options.setdefault((gi, rn_), set()).add(
                    (
                        ob.owner_net,
                        getattr(ob, "kind", None),
                        getattr(ob, "terminal_conductor_ref", None),
                    )
                )
                if getattr(ob, "kind", None) in ROUTING_ABSTRACTION_KINDS:
                    abstract_rects.setdefault(gi, set()).add(rn_)
                else:
                    literal_planar_claims.setdefault(gi, set()).add(
                        (ob.owner_net, rn_)
                    )
                continue
            lo = via_lo_by_key.get(lk) if lk is not None else None
            if lo is not None:
                cut_rect = _norm(r)
                obs_cuts.setdefault(lo, []).append(cut_rect)
                obs_cut_provenance_options.setdefault(
                    (lo, cut_rect), set()
                ).add(
                    (
                        ob.owner_net,
                        getattr(ob, "kind", None),
                        getattr(ob, "terminal_conductor_ref", None),
                    )
                )

        # ``rails=`` is a materialized-conductor boundary, not merely a set of
        # A* goal coordinates.  Canonical ``route_cell`` also projects its rail
        # stamps through ``obstacles``; direct engine callers are not required to
        # duplicate that second representation.  Seed every otherwise-absent rail
        # into the exact same owned planar/deck views as drawn obstacle metal so:
        #
        # * two drops landing on one rail are deck-connected;
        # * foreign nets see the rail as hard owned artwork; and
        # * the deck-isolation snapshot judges rail contacts/shorts.
        #
        # Exact owner+rect dedup keeps canonical route_cell byte-identical when its
        # projected obstacle already supplied the materialized conductor.  The
        # obligation is generic over net/layer/PDK and derived solely from the
        # declared rail geometry.
        for rail_owner, owned_rails in sorted(rail_by_net.items()):
            for rail_gi, rail_rect in owned_rails:
                rail_rect = _norm(rail_rect)
                existing_claims = literal_planar_claims.setdefault(
                    rail_gi, set()
                )
                if (rail_owner, rail_rect) in existing_claims:
                    continue
                rail_layer = tuple(tech.gi_to_metal_key(rail_gi))
                fixed_deck_literals.append(
                    (rail_layer, rail_owner, rail_rect)
                )
                obs_by_gi[rail_gi].append((rail_owner, rail_rect))
                obs_provenance_options.setdefault(
                    (rail_gi, rail_owner, rail_rect), set()
                ).add(("power_rail", None))
                planar_claim_options.setdefault(
                    (rail_gi, rail_rect), set()
                ).add((rail_owner, "power_rail", None))
                existing_claims.add((rail_owner, rail_rect))

        self._fixed_deck_literal_rects = tuple(sorted(
            set(fixed_deck_literals),
            key=lambda item: (
                tuple(item[0]),
                item[1] is not None,
                "" if item[1] is None else str(item[1]),
                tuple(item[2]),
            ),
        ))
        self._fixed_contact_rects = tuple(sorted(set(fixed_contact_rects)))
        mim_carve_bottom_blockers_by_gi: Dict[
            int, Tuple[Tuple[Optional[str], Rect, str], ...]
        ] = {}
        for carve in deck_stack.mim_carves:
            gi = tech.resolve_layer_to_gi(tuple(carve.bottom))
            if gi is None:
                continue
            marker_blockers = tuple(
                (owner, tuple(rect), "mim_top_marker")
                for layer, owner, rect in self._fixed_deck_literal_rects
                if tuple(layer) == tuple(carve.top_marker)
            )
            selected_bottom = tuple(
                (owner, tuple(rect), "mim_selected_bottom_component")
                for owner, rect in _marker_selected_conductor_components(
                    self._fixed_deck_literal_rects,
                    conductor_layer=tuple(carve.bottom),
                    marker_layer=tuple(carve.top_marker),
                )
            )
            additions = marker_blockers + (() if carve.marker_only else selected_bottom)
            if not additions:
                continue
            prior = mim_carve_bottom_blockers_by_gi.get(gi, ())
            mim_carve_bottom_blockers_by_gi[gi] = tuple(sorted(
                set(prior) | set(additions),
                key=lambda item: (
                    item[0] is not None,
                    "" if item[0] is None else str(item[0]),
                    item[1],
                    item[2],
                ),
            ))
        self._mim_carve_bottom_blockers_by_gi = (
            mim_carve_bottom_blockers_by_gi
        )
        mim_forbidden_by_lo: Dict[
            int, Tuple[Tuple[Optional[str], Rect], ...]
        ] = {}
        mim_cut_rules = tuple(
            contract.mim_forbidden_cut_touch_rules()
            if hasattr(contract, "mim_forbidden_cut_touch_rules")
            else ()
        )
        for cut_layer, conductor_layer, marker_layer in mim_cut_rules:
            lo = via_lo_by_key.get(tuple(cut_layer))
            if lo is None:
                raise ValueError(
                    "PDK MIM forbidden cut is outside the active routing stack: "
                    f"{tuple(cut_layer)}"
                )
            selected = _marker_selected_conductor_components(
                self._fixed_deck_literal_rects,
                conductor_layer=tuple(conductor_layer),
                marker_layer=tuple(marker_layer),
            )
            if selected:
                prior = mim_forbidden_by_lo.get(lo, ())
                mim_forbidden_by_lo[lo] = tuple(sorted(
                    set(prior) | set(selected),
                    key=lambda item: (
                        item[0] is not None,
                        "" if item[0] is None else str(item[0]),
                        item[1],
                    ),
                ))
        self._mim_forbidden_cut_touch_by_lo = mim_forbidden_by_lo
        mim_clearance_by_gi: Dict[
            int, Tuple[Tuple[Optional[str], Rect, int], ...]
        ] = {}
        mim_clearance_rules = tuple(
            contract.mim_bottom_routing_clearance_rules()
            if hasattr(contract, "mim_bottom_routing_clearance_rules")
            else ()
        )
        for conductor_layer, marker_layer, clearance_nm in mim_clearance_rules:
            gi = tech.resolve_layer_to_gi(tuple(conductor_layer))
            if gi is None:
                raise ValueError(
                    "PDK MIM bottom conductor is outside the active routing "
                    f"stack: {tuple(conductor_layer)}"
                )
            selected = _marker_selected_conductor_components(
                self._fixed_deck_literal_rects,
                conductor_layer=tuple(conductor_layer),
                marker_layer=tuple(marker_layer),
            )
            if not selected:
                continue
            prior = mim_clearance_by_gi.get(gi, ())
            additions = tuple(
                (owner, rect, int(clearance_nm)) for owner, rect in selected
            )
            mim_clearance_by_gi[gi] = tuple(sorted(
                set(prior) | set(additions),
                key=lambda item: (
                    item[0] is not None,
                    "" if item[0] is None else str(item[0]),
                    item[1],
                    item[2],
                ),
            ))
        self._mim_bottom_routing_clearance_by_gi = mim_clearance_by_gi
        mim_derived_unrelated_by_gi: Dict[
            int, Tuple[Tuple[str, Rect, int, str, str], ...]
        ] = {}
        mim_metal_rules = tuple(
            contract.mim_metal_geometry_rules()
            if hasattr(contract, "mim_metal_geometry_rules")
            else ()
        )
        for rule in mim_metal_rules:
            if getattr(rule, "predicate", None) != "derived_unrelated":
                continue
            gi = tech.resolve_layer_to_gi(tuple(rule.metal_layer))
            grow_nm = int(getattr(rule, "grow_nm", 0) or 0)
            mim_spacing_nm = int(rule.spacing_nm)
            metric = str(getattr(rule, "metric", "euclidian") or "euclidian").lower()
            if gi is None or int(self._space[gi]) <= grow_nm:
                continue
            selected = _marker_selected_conductor_components(
                self._fixed_deck_literal_rects,
                conductor_layer=tuple(rule.metal_layer),
                marker_layer=tuple(rule.marker_layer),
            )
            markers = tuple(sorted(
                _norm(rect)
                for layer, _owner, rect in self._fixed_deck_literal_rects
                if tuple(layer) == tuple(rule.marker_layer)
            ))
            additions_unrelated = []
            for m_rect in markers:
                touching = [
                    (owner, c_rect)
                    for owner, c_rect in selected
                    if _closed_rects_touch(c_rect, m_rect)
                ]
                touching_owners = {owner for owner, _ in touching}
                if len(touching_owners) != 1 or None in touching_owners:
                    continue
                plate_owner = str(next(iter(touching_owners)))
                for _owner, c_rect in touching:
                    ix0 = max(c_rect[0], m_rect[0])
                    iy0 = max(c_rect[1], m_rect[1])
                    ix1 = min(c_rect[2], m_rect[2])
                    iy1 = min(c_rect[3], m_rect[3])
                    if ix0 < ix1 and iy0 < iy1:
                        derived_rect = (
                            ix0 - grow_nm,
                            iy0 - grow_nm,
                            ix1 + grow_nm,
                            iy1 + grow_nm,
                        )
                        additions_unrelated.append((
                            plate_owner,
                            derived_rect,
                            mim_spacing_nm,
                            metric,
                            str(rule.rule_id),
                        ))
            if additions_unrelated:
                prior_unrel = mim_derived_unrelated_by_gi.get(gi, ())
                mim_derived_unrelated_by_gi[gi] = tuple(sorted(
                    set(prior_unrel) | set(additions_unrelated)
                ))
        self._mim_derived_unrelated_clearance_by_gi = (
            mim_derived_unrelated_by_gi
        )
        mim_top_via_clearance_by_lo: Dict[
            int, Tuple[Tuple[Optional[str], Rect, int], ...]
        ] = {}
        mim_top_via_rules = tuple(
            contract.mim_top_via_marker_clearance_rules()
            if hasattr(contract, "mim_top_via_marker_clearance_rules")
            else ()
        )
        for cut_layer, marker_layer, clearance_nm in mim_top_via_rules:
            lo = via_lo_by_key.get(tuple(cut_layer))
            if lo is None:
                raise ValueError(
                    "PDK MIM top-via clearance cut is outside the active "
                    f"routing stack: {tuple(cut_layer)}"
                )
            additions = tuple(
                (owner, tuple(rect), int(clearance_nm))
                for layer, owner, rect in self._fixed_deck_literal_rects
                if tuple(layer) == tuple(marker_layer)
            )
            if not additions:
                continue
            prior = mim_top_via_clearance_by_lo.get(lo, ())
            mim_top_via_clearance_by_lo[lo] = tuple(sorted(
                set(prior) | set(additions),
                key=lambda item: (
                    item[0] is not None,
                    "" if item[0] is None else str(item[0]),
                    item[1],
                    item[2],
                ),
            ))
        self._mim_top_via_marker_clearance_by_lo = (
            mim_top_via_clearance_by_lo
        )
        for key, options in sorted(
            obs_provenance_options.items(),
            key=lambda item: (
                int(item[0][0]),
                item[0][1] is not None,
                "" if item[0][1] is None else str(item[0][1]),
                tuple(item[0][2]),
            ),
        ):
            ordered_options = sorted(
                options,
                key=lambda value: tuple(
                    "" if item is None else str(item) for item in value
                ),
            )
            kinds, terminal_refs = tuple(zip(*ordered_options))
            kind = kinds[0] if len(set(kinds)) == 1 else None
            terminal_ref = (
                terminal_refs[0]
                if kind is not None and len(set(terminal_refs)) == 1
                else None
            )
            obs_provenance[key] = (kind, terminal_ref)
        obs_cut_provenance = {}
        for key, options in sorted(obs_cut_provenance_options.items()):
            ordered_options = sorted(
                options,
                key=lambda value: tuple(
                    "" if item is None else str(item) for item in value
                ),
            )
            owners, kinds, terminal_refs = tuple(zip(*ordered_options))
            owner = owners[0] if len(set(owners)) == 1 else None
            kind = kinds[0] if len(set(kinds)) == 1 else None
            # Owner and terminal seam are one attribution claim.  If duplicate
            # exact artwork disagrees on owner, retaining a coincident ref would
            # fabricate a device-terminal identity that no source asserted.
            terminal_ref = (
                terminal_refs[0]
                if owner is not None and len(set(terminal_refs)) == 1
                else None
            )
            obs_cut_provenance[key] = (owner, kind, terminal_ref)
        self._obs_by_gi = obs_by_gi
        self._obs_provenance = obs_provenance
        self._obs_cuts = obs_cuts
        self._obs_cut_provenance = obs_cut_provenance
        self._obs_cut_idx = {}
        for lo, rects in sorted(obs_cuts.items()):
            idx = _SpatialIndex()
            for rect in rects:
                idx.add(None, rect)
            self._obs_cut_idx[int(lo)] = idx
        self._abstract_rects = abstract_rects
        self._literal_planar_idx = {}
        for gi, claims in sorted(literal_planar_claims.items()):
            idx = _SpatialIndex()
            for owner, rect in sorted(
                claims,
                key=lambda item: (
                    item[0] is not None,
                    "" if item[0] is None else str(item[0]),
                    tuple(item[1]),
                ),
            ):
                idx.add(owner, rect)
            self._literal_planar_idx[int(gi)] = idx
        # The negotiated path may recover only LITERALLY DRAWN planar metal.
        # Cut-layer records live in ``obs_cut_provenance`` and therefore cannot
        # enter this view.  If any coincident provenance option marked a rect as a
        # routing abstraction, ``abstract_rects`` excludes the exact rect even
        # when the canonical kind was cleared by a duplicate conflict.
        owned_terminal_atoms = []
        for (gi, rect), claims in sorted(planar_claim_options.items()):
            ordered_claims = tuple(sorted(
                claims,
                key=lambda value: tuple(
                    "" if item is None else str(item) for item in value
                ),
            ))
            owners, kinds, terminal_refs = tuple(zip(*ordered_claims))
            # Eligibility is one atomic attribution claim over the exact planar
            # rectangle.  The obstacle legality model may retain every duplicate
            # owner, but component escape must not choose whichever owner happens
            # to match the routed net.  Cross-owner or cross-seam duplicates are
            # therefore absent from the owned-atom view, order-independently.
            if (
                len(set(owners)) != 1
                or owners[0] is None
                or len(set(terminal_refs)) != 1
                or any(kind in ROUTING_ABSTRACTION_KINDS for kind in kinds)
                or tuple(_norm(rect)) in abstract_rects.get(int(gi), set())
            ):
                continue
            kind = kinds[0] if len(set(kinds)) == 1 else None
            owned_terminal_atoms.append(_OwnedTerminalAtom(
                int(gi),
                str(owners[0]),
                tuple(_norm(rect)),
                kind,
                terminal_refs[0],
            ))
        self._owned_terminal_atoms = tuple(sorted(
            owned_terminal_atoms, key=_owned_terminal_atom_key
        ))
        self._owned_terminal_group_indexes = {}
        # spatial index of the STATIC obstacles per gi (O(local) legality).
        self._obs_idx: Dict[int, _SpatialIndex] = {}
        for gi, rects in obs_by_gi.items():
            idx = _SpatialIndex()
            for (owner, r) in rects:
                idx.add(owner, r)
            self._obs_idx[gi] = idx
        self._fixed_wide_source_key = None
        self._ensure_fixed_wide_geometry_indexes()
        self._wide_index_metal_index = None
        self._owned_atom_static_via_events = {}
        self._component_access = ComponentAccessSession(
            max_expansions=self.max_expansions
        )
        self._history_generation = 0
        self._via_static_legality_cache = OrderedDict()
        self._wire_static_legality_cache = OrderedDict()
        self._via_static_legality_cache_hits_total = 0
        self._via_static_legality_cache_misses_total = 0
        self._wire_static_legality_cache_hits_total = 0
        self._wire_static_legality_cache_misses_total = 0
        # committed-metal index per gi (rebuilt from the book per net attempt).
        self._metal_idx: Dict[int, _SpatialIndex] = {}
        # Committed via cuts indexed separately by bridged lower-layer index.
        # Cut spacing is net-blind while ownership decides whether a committed
        # cut is a hard same-net conflict or a soft foreign capacity use.
        self._cut_idx: Dict[int, _SpatialIndex] = {}
        # Subset of committed metal that is an actual via landing pad.  The main
        # metal index rejects electrical contact with every foreign shape;
        # this pad index keeps the landing-pad ownership contract explicit too.
        self._via_pad_idx: Dict[int, _SpatialIndex] = {}

        # --- routing window. Canonical route_cell supplies the exact final
        # floorplan rectangle. Legacy direct callers without that contract retain
        # the historical pins/artwork + halo domain for API compatibility.
        xs: List[int] = []
        ys: List[int] = []
        for pins in pins_by_net.values():
            for p in pins:
                px, py = _pin_xy_nm(p)
                xs.append(px)
                ys.append(py)
        for gi in obs_by_gi:
            for (_o, (x0, y0, x1, y1)) in obs_by_gi[gi]:
                xs += [x0, x1]
                ys += [y0, y1]
        if not xs:
            out.unsatisfied = unsatisfied
            out.spacing_reports = check_spacing_geometry(
                out, constraints, obstacles, tech, rails
            )
            out.success = not unsatisfied
            if routing_envelope is not None:
                out.routing_envelope_nm = routing_envelope.bbox_nm
            return out
        if routing_envelope is not None:
            self._bounds = routing_envelope.bbox_nm
        else:
            margin = max(half_w) + max(space) + max(
                (max(vp) for vp in via_pad_half.values()), default=0) + 2000
            self._bounds = (min(xs) - margin, min(ys) - margin,
                            max(xs) + margin, max(ys) + margin)

        # --- per-net pin nodes (x, y, gi) and allowed layers.
        usable = list(range(n_layers))
        net_nodes: Dict[str, List[Tuple[int, int, int]]] = {}
        net_node_pins: Dict[str, List[Pin]] = {}
        net_layers: Dict[str, List[int]] = {}
        self._required_wire_endpoints = set()
        invalid_layer_access_nets: Set[str] = set()
        for net, pins in pins_by_net.items():
            lg = (
                list(layer_set_by_net[net])
                if net in layer_set_by_net
                else list(usable)
            )
            net_layers[net] = lg
            nodes: List[Tuple[int, int, int]] = []
            node_pins: List[Pin] = []
            for p in pins:
                px, py = _pin_xy_nm(p)
                seed = tech.pin_seed(
                    p.layer, lg, strict_layers=net in layer_set_by_net
                )
                if seed is None:
                    invalid_layer_access_nets.add(net)
                    continue
                gi, _ladder = seed
                nodes.append((px, py, gi))
                node_pins.append(p)
                if getattr(p, "requires_wire_endpoint", False):
                    self._required_wire_endpoints.add((net, (px, py, gi)))
            net_nodes[net] = nodes
            net_node_pins[net] = node_pins
            if net in invalid_layer_access_nets:
                for constraint in layer_set_constraints_by_net.get(net, ()):
                    if constraint not in unsatisfied:
                        unsatisfied.append(constraint)

        # --- TRUNK-LAYER climb: each net's long run is biased UP to a trunk metal
        #     chosen by its pin-bbox span (in routing pitches). A cell-spanning net
        #     (the output / a power trunk) climbs over the output stage instead of
        #     crawling on M1/M2; a local net keeps trunk 0 (no bias). PDK-scaled.
        gi_ref = 1 if n_layers > 1 else 0
        pitch = max(1, 2 * half_w[gi_ref] + space[gi_ref])
        span_per_level = _TRUNK_SPAN_PITCHES * pitch
        trunk_gi: Dict[str, int] = {}
        for net, nodes in net_nodes.items():
            allowed = net_layers[net]
            declared = declared_trunk_by_net.get(net)
            if declared is not None and declared in allowed:
                trunk_gi[net] = declared
                continue
            if not allowed:
                trunk_gi[net] = 0
                continue
            if len(nodes) < 2:
                fallback = 0
                trunk_gi[net] = min(
                    allowed,
                    key=lambda gi: (abs(gi - fallback), gi),
                )
                continue
            xs2 = [c[0] for c in nodes]
            ys2 = [c[1] for c in nodes]
            span = (max(xs2) - min(xs2)) + (max(ys2) - min(ys2))
            top = max(allowed)
            fallback = max(0, min(top, span // span_per_level))
            if declared is None:
                # Preserve the historical automatic-trunk choice for nets
                # without a structural declaration, snapped into a possibly
                # sparse hard LayerSet.
                trunk_gi[net] = min(
                    allowed,
                    key=lambda gi: (abs(gi - fallback), gi),
                )
            else:
                # A soft declared trunk may be excluded by a hard LayerSet.
                # Snap the existing deterministic fallback to the nearest
                # actually allowed layer, so the hard Width is carried by
                # real routed geometry instead of widening an unused gi.
                trunk_gi[net] = min(
                    allowed,
                    key=lambda gi: (abs(gi - fallback), gi),
                )
        self._trunk_gi = trunk_gi
        self._declared_trunk_gi_by_net = declared_trunk_by_net
        self._declared_trunk_axis_by_net = declared_trunk_axis_by_net
        self._direction_preference_by_gi = direction_preference_by_gi

        # --- WIDTH materialization: a plain Width retains the historical
        #     uniform footprint.  When a net also declares a resolvable soft
        #     TrunkLayer, its effective trunk gi above is the sole supra-min
        #     carrier; local access/riser layers stay at their PDK minimum.  The
        #     same ``_hw`` value still feeds legality, congestion, occupancy and
        #     emit.  If a hard LayerSet excludes the declared gi, ``trunk_gi`` is
        #     the deterministic allowed-layer fallback and carries the width.
        width_carrier_gi_by_net = {
            net: trunk_gi[net]
            for net in sorted(declared_trunk_by_net)
            if net in trunk_gi
        }
        self._net_half_w, width_widened, width_cons_by_net = (
            self._width_half_table(
                constraints,
                half_w,
                n_layers,
                carrier_gi_by_net=width_carrier_gi_by_net,
            )
        )

        # Mandatory physical wire endpoints precede movable rail access routes;
        # otherwise an early power connection can fence off the only permitted
        # boundary endpoint before negotiation has a chance to move that wire.
        # Fixed rail artwork is already in the hard world and is never removed.
        # Declared rail/spine nets retain priority within each endpoint tier.
        # Within a tier, route exact
        # terminal-access-heavy nets before the committed world gets crowded;
        # schematic fanout, generated span and net id are deterministic tie
        # breaks.
        terminal_atom_count_by_net = Counter(
            str(atom.owner)
            for atom in self._owned_terminal_atoms
            if type(atom.terminal_conductor_ref) is str
            and atom.terminal_conductor_ref
        )
        ordered = _negotiation_net_order(
            tuple(pins_by_net),
            priority_nets=set(rail_by_net),
            pin_count_by_net={
                str(net): _topology_terminal_count(pins)
                for net, pins in pins_by_net.items()
            },
            terminal_atom_count_by_net=terminal_atom_count_by_net,
            pin_span_by_net={
                str(net): _topology_terminal_span_nm(pins)
                for net, pins in pins_by_net.items()
            },
            fixed_endpoint_nets={
                net for net, _node in self._required_wire_endpoints
            },
        )
        ordered = self._mirror_order(ordered, mirror_by_b)
        ordered = self._match_pair_order(
            ordered, length_constraints, mirror_by_b
        )

        # --- resource grid + history for negotiated congestion -----------------
        self._res_cell = pitch
        self._history: Dict[Tuple[int, int, int], float] = {}

        # --- negotiated-congestion (PathFinder) rip-up / reroute solve ---------
        # Foreign device ARTWORK stays a HARD keep-out (real blockage).  Foreign
        # COMMITTED METAL contact is always hard, while its positive
        # min-space band remains SOFT congestion so a net can NEGOTIATE a channel
        # instead of the greedy engine's "route fails ->
        # pick one victim -> re-queue" thrash that tore down already-routed nets.
        # history[resource] rises each round a resource is over-used; pres_fac rises
        # each round. Uniform across nets (no per-net / per-circuit branch).
        book = _Book(n_layers)
        self._book = book
        if getattr(self, '_post_selection_observation', None) is not None:
            from clay.layout.route.group_endpoint_observation import bind_route_book
            bind_route_book(self, book)
        committed: Dict[str, _RoutedNet] = {}
        self._pres_fac = self.pres_fac0
        self._solve_expansions_total = 0
        self._solve_expansion_budget = self.max_iters * self.expansion_cap
        self._group_route_required_states = {}
        self._layered_portal_wire_states = set()
        self._group_route_landing_baselines = {}
        self._group_route_global_interfaces_by_net = {}
        self._group_route_local_segment_keys_by_net = {}
        self._group_route_local_via_keys_by_net = {}
        self._group_route_global_nodes_by_net = {}
        if group_route_contract is None:
            if getattr(self, '_experimental_fixed_index_seal_budget', None) is not None:
                from clay.layout.route.fixed_index_seal import seal_if_enabled
                seal_if_enabled(self)
            if '_route_net' not in vars(self) and getattr(type(self), '_route_net', None) is GridlessEngine._route_net:
                from clay.layout.route.planar_precheck import verify_static_terminal_escapability
                verify_static_terminal_escapability(
                    self, pins_by_net, net_nodes, net_node_pins, net_layers
                )
        if group_route_contract is not None:
            group_nets = {port.net for port in group_route_contract.ports}
            mirrored_nets = {
                net
                for partner, mirror in mirror_by_b.items()
                for net in (partner, mirror.net_a)
            }
            overlap = tuple(sorted(group_nets.intersection(mirrored_nets)))
            if overlap:
                raise ValueError(
                    "group-route landing allocation does not yet support hard "
                    f"mirror nets {overlap!r}"
                )
            if getattr(self, '_experimental_fixed_index_seal_budget', None) is not None:
                from clay.layout.route.fixed_index_seal import seal_if_enabled
                seal_if_enabled(self)
            out.group_route_landing_certificate = (
                self._reserve_group_route_landings(
                    group_route_contract,
                    pins_by_net,
                    net_nodes,
                    net_node_pins,
                    net_layers,
                    book,
                    ordered_nets=ordered,
                    **({'topology_plan': group_demand_topology_plan}
                       if group_demand_topology_plan is not None else {}),
                )
            )
        if getattr(self, '_post_selection_observation', None) is not None:
            from clay.layout.route.group_endpoint_observation import stop_if_unobserved
            stop_if_unobserved(self)
        nonretryable_nets: Set[str] = set()
        route_stop_by_net: Dict[str, NegotiationRouteStop] = {}
        route_rank = {net: index for index, net in enumerate(ordered)}
        route_invocation = 0
        negotiation_round = 0

        def _route_one(net: str) -> None:
            """Route a single net under the CURRENT congestion + commit (or record an
            honest unconnected placeholder). Trivial 0/1-pin nets draw no metal."""
            nonlocal route_invocation
            route_invocation += 1
            route_watchdog_details = [
                ("committed_net_count", len(committed)),
                (
                    "connected_net_count",
                    sum(1 for routed in committed.values() if routed.connected),
                ),
                ("negotiation_round", int(negotiation_round)),
                ("net_order_index", int(route_rank[net])),
                ("ordered_net_count", len(ordered)),
                ("route_invocation", int(route_invocation)),
                ("route_stop_count", len(route_stop_by_net)),
                ("solve_expansion_budget", int(self._solve_expansion_budget)),
                ("solve_expansions_total", int(self._solve_expansions_total)),
            ]
            if route_stop_by_net:
                route_watchdog_details.append((
                    "route_stop_frontier",
                    self._route_stop_frontier(route_stop_by_net),
                ))
            self._route_watchdog_details = tuple(sorted(
                route_watchdog_details, key=lambda item: item[0]
            ))
            self._watchdog_check(net, "route_net")
            nodes = net_nodes[net]
            if net in invalid_layer_access_nets:
                nonretryable_nets.add(net)
                route_stop_by_net[net] = NegotiationRouteStop(
                    NegotiationStopReason.INVALID_LAYER_ACCESS,
                    net,
                    "pin_access",
                    tuple(net_layers.get(net, ())),
                )
                committed[net] = _RoutedNet(name=net, connected=False)
                return
            mirror = mirror_by_b.get(net)
            if mirror is not None:
                self._hard_rejections = {}
                self._hard_rejection_samples = {}
                reflected = self._materialize_mirror_partner(
                    mirror,
                    committed.get(mirror.net_a),
                    net_nodes.get(mirror.net_a, ()),
                    nodes,
                    net_layers.get(net, ()),
                )
                if reflected is not None:
                    route_stop_by_net.pop(net, None)
                    self._commit(book, committed, reflected)
                    return
                route_stop_by_net[net] = NegotiationRouteStop(
                    NegotiationStopReason.NO_LEGAL_ROUTE,
                    net,
                    "mirror_partner",
                    (
                        str(mirror.net_a),
                        float(mirror.axis_x_um),
                    ),
                    tuple(sorted((
                        ("failure_kind", "mirror_reflection_illegal"),
                        (
                            "rejection_samples",
                            self._hard_rejection_sample_evidence(),
                        ),
                        *getattr(self, "_mirror_failure_details", ()),
                    ), key=lambda item: item[0])),
                )
                committed[net] = _RoutedNet(name=net, connected=False)
                return
            if len(nodes) <= 1 and not self._requires_singleton_route(net):
                routed_trivial = self._with_group_route_landing_baseline(
                    _RoutedNet(name=net, connected=True)
                )
                if self._candidate_mim_metal_failure(
                    net, routed_trivial.segs, routed_trivial.vias, prior=_RoutedNet(net),
                ):
                    route_stop_by_net[net] = NegotiationRouteStop(
                        NegotiationStopReason.NO_LEGAL_ROUTE, net, "group_route_landing",
                        (), (("failure_kind", "candidate_mim_metal_clearance"),),
                    )
                    committed[net] = _RoutedNet(name=net, connected=False)
                    return
                self._commit(book, committed, routed_trivial)
                return
            _t = time.monotonic()
            self._edge_fail = None
            self._hard_rejections: Dict[Tuple[str, str], int] = {}
            self._hard_rejection_samples = {}
            self._route_failure_details: Tuple[Tuple[str, Any], ...] = ()
            previous_mirror_legality = getattr(
                self, "_active_mirror_legality", None
            )
            previous_mirror_pair_conflict = getattr(
                self, "_mirror_pair_conflict_seen", False
            )
            previous_match_layer_surcharge = getattr(
                self, "_active_match_layer_surcharge", None
            )
            self._active_mirror_legality = mirror_by_a.get(net)
            self._mirror_pair_conflict_seen = False
            self._active_match_layer_surcharge = self._compute_match_layer_surcharge(
                net,
                length_constraints,
                committed,
                getattr(self, "_declared_trunk_gi_by_net", {}),
            )
            try:
                rn = self._route_net(
                    net,
                    nodes,
                    net_layers[net],
                    net_node_pins.get(net),
                )
            finally:
                mirror_pair_conflict_seen = bool(
                    self._mirror_pair_conflict_seen
                )
                self._active_mirror_legality = previous_mirror_legality
                self._mirror_pair_conflict_seen = (
                    previous_mirror_pair_conflict
                )
                self._active_match_layer_surcharge = previous_match_layer_surcharge
            if rn is None and mirror_pair_conflict_seen:
                self._edge_fail = (
                    "mirror_pair_in_transaction_conflict",
                    self._hard_rejection_evidence(),
                )
            self._watchdog_check(net, "route_net")
            if self.debug:
                print(f"  [gridless] {net:8s} pins={len(nodes):2d} "
                      f"{'OK ' if rn is not None else 'FAIL'} "
                      f"{time.monotonic() - _t:5.1f}s pres={self._pres_fac:.1f}",
                      flush=True)
            if rn is not None:
                rn = self._with_group_route_landing_baseline(rn)
                if self._candidate_mim_metal_failure(
                    net, rn.segs, rn.vias, prior=_RoutedNet(net),
                ):
                    self._edge_fail = ("candidate_mim_metal_clearance", self._hard_rejection_evidence())
                    rn = None
            if rn is not None:
                route_stop_by_net.pop(net, None)
                self._commit(book, committed, rn)
            else:
                edge_fail = getattr(self, "_edge_fail", None)
                if edge_fail is None:
                    edge_fail = (
                        "pin_access",
                        self._hard_rejection_evidence(),
                    )
                reason = {
                    "budget": NegotiationStopReason.EDGE_EXPANSION_BUDGET,
                    "solve_budget": NegotiationStopReason.SOLVE_WORK_BUDGET,
                }.get(
                    edge_fail[0] if edge_fail else None,
                    NegotiationStopReason.NO_LEGAL_ROUTE,
                )
                detail_map = dict(
                    getattr(self, "_route_failure_details", ())
                )
                detail_map["failure_kind"] = (
                    str(edge_fail[0]) if edge_fail else "pin_access"
                )
                detail_map["rejection_samples"] = (
                    self._hard_rejection_sample_evidence()
                )
                route_stop_by_net[net] = NegotiationRouteStop(
                    reason,
                    net,
                    "route_net",
                    tuple(edge_fail[1:]) if edge_fail else (),
                    tuple(sorted(detail_map.items(), key=lambda item: item[0])),
                )
                committed[net] = _RoutedNet(name=net, connected=False)

        # ROUND 0: route every net once (present congestion from earlier nets in
        # THIS round already spreads them; the positive spacing band is soft so
        # negotiation can clean it up without permitting electrical contact).
        for net in ordered:
            _route_one(net)

        best_committed: Dict[str, _RoutedNet] = {}
        best_route_stops: Dict[str, NegotiationRouteStop] = {}
        best_deck_snapshot: Optional[_DeckNegotiationSnapshot] = None
        best_score: Optional[Tuple[int, ...]] = None
        converged = False

        def _clone_committed(
            source: Mapping[str, _RoutedNet],
        ) -> Dict[str, _RoutedNet]:
            return {
                n: _RoutedNet(
                    name=r.name,
                    segs=list(r.segs),
                    vias=list(r.vias),
                    connected=r.connected,
                )
                for n, r in source.items()
            }

        def _snapshot(
            score: Tuple[int, ...],
            deck_snapshot: _DeckNegotiationSnapshot,
        ) -> None:
            nonlocal best_committed, best_route_stops, best_deck_snapshot, best_score
            if not deck_snapshot.report.cross_net_isolated:
                raise AssertionError("deck-merged negotiation snapshot cannot be saved")
            best_score = score
            best_committed = _clone_committed(committed)
            best_route_stops = dict(route_stop_by_net)
            best_deck_snapshot = deck_snapshot

        for _it in range(self.max_iters):
            # DETECT: over-used resources (min-space footprint sharing, occ >= 2),
            # real same-layer shorts (the sep==0 oracle), EXACT sub-min-space
            # residuals split STRICT-advisory / DECK-VISIBLE (the deck-DRC
            # spacing goal), and unconnected nets.
            occ = self._occupancy(committed)
            overused = {c for c, nets in occ.items() if len(nets) >= 2}
            shorts = self._detect_shorts(committed)
            spacing_all, spacing = self._space_residual_scan(committed)
            deck_snapshot = self._deck_negotiation_snapshot(committed)
            graph_unconn = {
                n for n in pins_by_net
                if not committed.get(n, _RoutedNet(name=n)).connected
            }
            graph_unconn.update(
                _deck_open_route_nets(deck_snapshot.report, ordered)
            )
            unconn = [n for n in ordered if n in graph_unconn]
            if (
                not unconn
                and deck_snapshot.report.cross_net_isolated
                and deck_snapshot.report.probe_complete
                and any(a == b for a, b in spacing)
            ):
                # A complete world may contain a gap the existing final-world
                # legalizer can close before negotiation destroys that world.
                # Unary witnesses also include fixed foreign/cut conflicts;
                # they authorize one trial, never a presumed same-owner fix.
                repaired = self._try_complete_world_legalization(
                    committed, net_layers, ordered, spacing,
                    mirrors=tuple(mirror_by_b.values()),
                    length_constraints=length_constraints,
                )
                if repaired is not None:
                    committed = repaired
                book = self._book
                # Remeasure even a rejected trial: its private scans replaced
                # the observer join, and rollback keeps index generations
                # monotonic. These adjacent returns belong to this live round.
                occ = self._occupancy(committed)
                overused = {c for c, nets in occ.items() if len(nets) >= 2}
                shorts = self._detect_shorts(committed)
                spacing_all, spacing = self._space_residual_scan(committed)
                deck_snapshot = self._deck_negotiation_snapshot(committed)
                graph_unconn = {
                    n for n in pins_by_net
                    if not committed.get(n, _RoutedNet(name=n)).connected
                }
                graph_unconn.update(
                    _deck_open_route_nets(deck_snapshot.report, ordered)
                )
                unconn = [n for n in ordered if n in graph_unconn]
            if (
                not unconn
                and deck_snapshot.report.cross_net_isolated
                and deck_snapshot.report.probe_complete
                and any(a != b for a, b in spacing)
            ):
                repaired = self._try_complete_world_wire_displacement(
                    committed, net_layers, ordered, spacing,
                    net_nodes_by_net=net_nodes,
                    mirrors=tuple(mirror_by_b.values()),
                    length_constraints=length_constraints,
                )
                if repaired is not None:
                    committed = repaired
                book = self._book
                # Trial geometry never owns the observer's round join. Both
                # accepted and rejected trials return to this exact route frame
                # for adjacent spacing/deck measurements before snapshot/rip-up.
                occ = self._occupancy(committed)
                overused = {c for c, nets in occ.items() if len(nets) >= 2}
                shorts = self._detect_shorts(committed)
                spacing_all, spacing = self._space_residual_scan(committed)
                deck_snapshot = self._deck_negotiation_snapshot(committed)
                graph_unconn = {
                    n for n in pins_by_net
                    if not committed.get(n, _RoutedNet(name=n)).connected
                }
                graph_unconn.update(
                    _deck_open_route_nets(deck_snapshot.report, ordered)
                )
                unconn = [n for n in ordered if n in graph_unconn]
            length_tuning_eligible = bool(
                length_constraints
                and not unconn
                and deck_snapshot.report.cross_net_isolated
                and not spacing
            )
            if length_tuning_eligible:
                length_reports = self._materialize_length_constraints(
                    committed, length_constraints
                )
                # Length tuning rebuilds the committed-world book only after an
                # exact deck-clean candidate wins. Keep negotiation on that
                # same truth, then remeasure every outer-loop predicate.
                book = self._book
                occ = self._occupancy(committed)
                overused = {c for c, nets in occ.items() if len(nets) >= 2}
                shorts = self._detect_shorts(committed)
                spacing_all, spacing = self._space_residual_scan(committed)
                deck_snapshot = self._deck_negotiation_snapshot(committed)
                graph_unconn = {
                    n for n in pins_by_net
                    if not committed.get(n, _RoutedNet(name=n)).connected
                }
                graph_unconn.update(
                    _deck_open_route_nets(deck_snapshot.report, ordered)
                )
                unconn = [n for n in ordered if n in graph_unconn]
            else:
                length_reports = tuple(
                    self._length_report_for_committed(
                        constraint, committed
                    )
                    for constraint in length_constraints
                )
            length_violations = tuple(
                report
                for constraint, report in zip(length_constraints, length_reports)
                if getattr(constraint, "hardness", "hard") == "hard"
                and not report["satisfied"]
            )
            score = _negotiation_snapshot_score(
                unconnected=unconn,
                shorts=shorts,
                deck_report=deck_snapshot.report,
                deck_spacing_count=len(spacing),
                overused_count=len(overused),
                advisory_spacing_count=len(spacing_all),
                length_violation_count=len(length_violations),
            )
            if (
                deck_snapshot.report.cross_net_isolated
                and (best_score is None or score < best_score)
            ):
                _snapshot(score, deck_snapshot)
            if self.debug:
                print(f"  [gridless iter {_it}] unconn={len(unconn)} "
                      f"shorts={len(shorts)} spacing={len(spacing)} "
                      f"advisory={len(spacing_all)} overused={len(overused)} "
                      f"pres={self._pres_fac:.1f}", flush=True)
            # GOAL: net-complete + exact deck-connected cross-net isolation + 0
            # DECK-VISIBLE sub-min-space residuals (the deck-DRC spacing
            # target, judged at the DECK'S OWN semantics: merged, net-blind,
            # euclidean, parsed rule values -- ``_space_residual_scan``).
            # Connectivity alone used to be the stop; the "min-space residual
            # is not an LVS defect" tolerance emitted real M2.S.1/M3.S.1 deck
            # violations (R7a), so spacing-clean joined the goal. The engine's
            # STRICTER advisory detector then became the stop and gates burned
            # the whole route budget (~5400s, budget-maxed) negotiating
            # residuals the real deck never flags (measured: success=False +
            # residuals>0 with deck DRC total=0) -- the stop is now
            # DECK-ALIGNED; strict stays an advisory diagnostic (reported as
            # ``advisory_residuals``, never silently dropped). The coarse
            # rasterised ``overused`` stays the escalation signal only (it
            # over-approximates; exact residuals are the judgment).
            if (
                not unconn
                and deck_snapshot.report.cross_net_isolated
                and not spacing
                and not length_violations
            ):
                converged = True
                break
            if _it == self.max_iters - 1:
                break
            # ESCALATE: chronic contention and exact deck-visible DRC witnesses
            # get persistent local history.  Same-net residuals have one owner
            # and therefore cannot appear in ``overused``; without the exact
            # witness cells their victim rip-up reproduced identical geometry
            # every round with zero memory.
            spacing_history_cells = set(getattr(
                self, "_last_deck_spacing_history_cells", frozenset()
            ))
            history_cells = overused | spacing_history_cells
            for c in history_cells:
                self._history[c] = self._history.get(c, 0.0) + self.hist_fac
            if history_cells:
                self._history_generation = int(getattr(
                    self, "_history_generation", 0
                )) + 1
            self._pres_fac *= self.pres_fac_mult
            # VICTIMS: exact deck-merged owner groups + nets in a DECK-VISIBLE
            # spacing residual + unconnected nets.  Coarse over-use remains a
            # history PRICE only: it deliberately over-approximates spacing and
            # therefore cannot authorize ripping a connected net without an exact
            # deck witness (the measured GF180 case otherwise pulled an unrelated
            # ntail route into every replay round).
            # Dynamic route failure is evidence only about the current world, so
            # every open net remains a victim; only input-static layer-access
            # contradictions are nonretryable. Native same-layer incidence is
            # diagnostic only.
            # Exact merge groups select victims but never price history: only the
            # over-used resource cells above do that.
            conflict_groups = []
            route_name_by_deck = {
                str(net).upper(): str(net) for net in ordered
            }
            for group in deck_snapshot.report.merged_net_groups:
                routed_group = frozenset(
                    route_name_by_deck[owner.upper()]
                    for owner in group
                    if owner.upper() in route_name_by_deck
                )
                if routed_group:
                    conflict_groups.append(routed_group)
            conflict_groups.extend(frozenset(pair) for pair in spacing)
            protected_nets = _negotiation_protected_nets(
                set(rail_by_net), iteration=_it
            )
            victims = _negotiation_conflict_victims(
                ordered,
                conflict_groups,
                protected_nets=protected_nets,
            )
            for n in unconn:
                if n not in nonretryable_nets:
                    victims.add(n)
            if length_tuning_eligible:
                for report in length_violations:
                    shorter = (
                        str(report["net_a"])
                        if int(report["length_a_nm"])
                        <= int(report["length_b_nm"])
                        else str(report["net_b"])
                    )
                    if shorter not in nonretryable_nets:
                        victims.add(shorter)
            # A Mirror pair is one routing transaction.  Ripping either side
            # without its mate would leave a stale reflection in the committed
            # world, so conflict and open-net recovery always couples the pair.
            for partner, mirror in mirror_by_b.items():
                if partner in victims or mirror.net_a in victims:
                    victims.update((partner, mirror.net_a))
            blocker_components = _negotiation_committed_blocker_components(
                ordered=ordered,
                open_nets=unconn,
                route_stops=route_stop_by_net,
                committed=committed,
                protected_nets=protected_nets,
            )
            negotiation_round = _it + 1
            _negotiation_reroute_in_place(
                ordered=ordered,
                victims=victims,
                book=book,
                committed=committed,
                route_one=_route_one,
                blocker_components=blocker_components,
                atomic_route_groups=tuple(
                    (str(mirror.net_a), str(partner))
                    for partner, mirror in sorted(mirror_by_b.items())
                ),
                route_stops=route_stop_by_net,
                protected_nets=protected_nets,
            )

        # Every net receives an honest entry; fixed work/iteration budgets above
        # are deterministic functions of the declared policy and exact inputs.
        for net in pins_by_net:
            if net not in committed:
                route_stop_by_net[net] = NegotiationRouteStop(
                    NegotiationStopReason.SOLVE_WORK_BUDGET,
                    net,
                    "solve",
                    (
                        self._solve_expansions_total,
                        self._solve_expansion_budget,
                    ),
                )
                committed[net] = _RoutedNet(name=net, connected=False)
        # Every negotiation iteration detects and snapshots before either
        # converging or reaching the fixed iteration bound.  There is no route
        # mutation between that detect step and this point, so an unconditional
        # whole-world rescan merely repeats the final loop-top scan.  A missing
        # entry above carries no geometry, so it cannot change spacing or shorts.

        has_isolated_snapshot = best_deck_snapshot is not None
        failed_merge_groups = (
            ()
            if has_isolated_snapshot
            else tuple(deck_snapshot.report.merged_net_groups)
        )
        emit_from = best_committed if has_isolated_snapshot else {}
        emit_route_stops = (
            best_route_stops if has_isolated_snapshot else route_stop_by_net
        )
        open_state_from = emit_from if has_isolated_snapshot else committed
        # The per-net watchdog context above describes the last route invocation,
        # not the negotiation snapshot selected for emission.  A deadline reached
        # during final legalization must name the exact convergence predicates
        # that kept a fully-routed design in negotiation; otherwise a 45/45 case
        # reports only a stale net cursor and the next repair axis is guesswork.
        # Keep both the final-round score and the best isolated score because the
        # latter is the geometry ``emit_from`` is about to legalize.
        self._route_watchdog_details = tuple(sorted((
            ("best_snapshot_score", (
                None if best_score is None else tuple(best_score)
            )),
            ("committed_net_count", len(emit_from)),
            (
                "connected_net_count",
                sum(1 for routed in emit_from.values() if routed.connected),
            ),
            (
                "current_advisory_spacing_count",
                len(spacing_all),
            ),
            (
                "current_deck_cross_net_isolated",
                bool(deck_snapshot.report.cross_net_isolated),
            ),
            ("current_deck_spacing_count", len(spacing)),
            (
                "current_length_violation_count",
                len(length_violations),
            ),
            ("current_overused_count", len(overused)),
            ("current_short_pairs", tuple(sorted(shorts))),
            ("current_snapshot_score", tuple(score)),
            ("current_unconnected_nets", tuple(unconn)),
            ("negotiation_round", int(negotiation_round)),
            ("route_invocation", int(route_invocation)),
            ("route_stop_count", len(emit_route_stops)),
            (
                "solve_expansion_budget",
                int(self._solve_expansion_budget),
            ),
            (
                "solve_expansions_total",
                int(self._solve_expansions_total),
            ),
        ), key=lambda item: item[0]))
        # --- FINAL MERGE-OR-CLEAR MATERIALIZATION (R7 tail) on the snapshot
        #     actually emitted: close every same-net band gap that is legally
        #     closable against the FINAL world (the per-attempt legalization
        #     judged a world rip-up later changed). See ``_legalize_committed``.
        if ordered:
            self._watchdog_check(ordered[0], "final_legalize")
        for net in tuple(sorted(emit_from)):
            emit_from[net] = self._with_group_route_landing_baseline(
                emit_from[net]
            )
        prelegalized_emit = _clone_committed(emit_from)
        self._legalize_final_emission(emit_from, net_layers)
        # Final merge-or-clear may add primary-side metal.  Re-materialize each
        # partner from that final primary geometry, then judge the deck snapshot;
        # this keeps legalization from silently degrading a hard Mirror.
        for partner, mirror in sorted(mirror_by_b.items()):
            original = emit_from.get(partner)
            without_partner = dict(emit_from)
            without_partner.pop(partner, None)
            self._rebuild_committed_state(without_partner)
            reflected = self._materialize_mirror_partner(
                mirror,
                without_partner.get(mirror.net_a),
                net_nodes.get(mirror.net_a, ()),
                net_nodes.get(partner, ()),
                net_layers.get(partner, ()),
            )
            if reflected is not None:
                emit_from[partner] = reflected
            elif original is not None:
                emit_from[partner] = original
            self._rebuild_committed_state(emit_from)
        pre_length_deck_snapshot = self._deck_negotiation_snapshot(emit_from)
        _pre_length_advisory, pre_length_spacing = (
            self._space_residual_scan(emit_from)
        )
        if (
            all(
                emit_from.get(net, _RoutedNet(name=net)).connected
                for net in pins_by_net
            )
            and
            pre_length_deck_snapshot.report.cross_net_isolated
            and not _deck_open_route_nets(
                pre_length_deck_snapshot.report, ordered
            )
            and not pre_length_spacing
        ):
            self._materialize_length_constraints(
                emit_from, length_constraints
            )
        emitted_deck_snapshot = self._deck_negotiation_snapshot(emit_from)
        if (
            has_isolated_snapshot
            and not emitted_deck_snapshot.report.cross_net_isolated
        ):
            emit_from = prelegalized_emit
            self._rebuild_committed_state(emit_from)
            emitted_deck_snapshot = self._deck_negotiation_snapshot(emit_from)
        best_deck_snapshot = (
            emitted_deck_snapshot
            if (
                has_isolated_snapshot
                and emitted_deck_snapshot.report.cross_net_isolated
            )
            else None
        )
        if ordered:
            self._watchdog_check(ordered[0], "final_emit")
        self._best_committed = emit_from  # diagnostics only (never read by contract)
        self._best_deck_snapshot = best_deck_snapshot
        emitted_deck_open_nets = set(
            _deck_open_route_nets(emitted_deck_snapshot.report, ordered)
        )

        # --- emit geometry per net from the BEST round + assemble RoutedCell.
        emitted_by_net: Dict[str, _EmittedNetGeometry] = {}
        for net, pins in pins_by_net.items():
            self._watchdog_check(net, "final_emit")
            rn = emit_from.get(net) or _RoutedNet(name=net, connected=False)
            nr = NetRoute(net=net, pins=list(pins))
            nr.connected = bool(
                rn.connected and net not in emitted_deck_open_nets
            )
            emitted = self._emit_net(rn)
            emitted_by_net[net] = emitted
            nr.wires = list(emitted.wires)
            nr.length_um = emitted.length_um
            out.nets[net] = nr

        # Bind global upper-layer portal centerlines to the exact segment
        # polygons emitted above.  A same-net via pad is not a wire witness.
        # The local key exclusion is conservative if identical geometry was
        # also emitted by a local closure; it never promotes a local wire.
        from clay.layout.route.group_ports import GroupRouteGlobalWireWitness

        certificate = getattr(out, "group_route_landing_certificate", None)
        portal_targets = {
            (row.net, tuple(row.portal_layer_key), tuple(row.portal_xy_nm))
            for row in (() if certificate is None else certificate.landings)
            if row.portal_layer_key is not None
        }
        global_wires = []
        for net, emitted in sorted(emitted_by_net.items()):
            rn = emit_from.get(net)
            if rn is None:
                continue
            if len(rn.segs) != len(emitted.segment_wires):
                raise ValueError("group-route segment emitter inventory differs")
            local_keys = self._group_route_local_segment_keys_by_net.get(
                net, frozenset()
            )
            for index, (seg, poly) in enumerate(
                zip(rn.segs, emitted.segment_wires)
            ):
                if index % 256 == 0:
                    self._watchdog_check(net, "final_group_global_wire_witness")
                layer = (int(poly.layer), int(poly.datatype))
                a, b = (int(seg.ax), int(seg.ay)), (int(seg.bx), int(seg.by))
                if (
                    (net, layer, a) not in portal_targets
                    and (net, layer, b) not in portal_targets
                ):
                    continue
                key = (
                    min(a, b),
                    max(a, b),
                    int(seg.gi),
                    int(self._segment_half_width(net, seg)),
                )
                if key in local_keys or a == b:
                    continue
                points = tuple((int(p.x), int(p.y)) for p in poly.polygon.vertices)
                if len(points) != 4:
                    raise ValueError(
                        "group-route global emitted wire is not rectangular"
                    )
                xs, ys = zip(*points)
                global_wires.append(
                    GroupRouteGlobalWireWitness(
                        net=str(net),
                        layer_key=layer,
                        start_xy_nm=a,
                        end_xy_nm=b,
                        bbox_nm=(min(xs), min(ys), max(xs), max(ys)),
                    )
                )
        out.group_route_global_wire_witnesses = tuple(sorted(global_wires))

        # Preserve the actual router-owned via sites for the final GDS consumer.
        # PCell-owned contacts are not represented by _ViaSite and are excluded.
        via_sites = []
        for net in sorted(pins_by_net):
            routed_net = emit_from.get(net) or _RoutedNet(name=net, connected=False)
            for via in routed_net.vias:
                self._require_policy_via(via)
                cut_key = self._tech.via_key_for(via.lo)
                if cut_key is None:
                    raise ValueError("router-owned via has no PDK cut layer")
                cuts = []
                for poly in self._via_polys(via)[2:]:
                    xs = [int(point.x) for point in poly.polygon.vertices]
                    ys = [int(point.y) for point in poly.polygon.vertices]
                    cuts.append((min(xs), min(ys), max(xs), max(ys)))
                allowed_cut_counts = (
                    (1, self.via_redundancy ** 2)
                    if self._single_via_escape_allowed()
                    else (self.via_redundancy ** 2,)
                )
                if len(cuts) not in allowed_cut_counts or len(set(cuts)) != len(cuts):
                    raise ValueError("router-owned via has incomplete array geometry")
                via_sites.append((str(net), tuple(cut_key), tuple(sorted(cuts))))
        out.router_via_redundancy = self.via_redundancy
        out.router_via_allow_single_escape = bool(self._single_via_escape_allowed())
        out.router_via_sites = tuple(via_sites)

        # Mirror is accepted only on the exact geometry returned to the caller.
        # Valid declarations always receive one report; a failed final comparison
        # remains an explicit unsatisfied hard constraint and forces success false.
        for partner, mirror in sorted(mirror_by_b.items()):
            primary_route = emit_from.get(mirror.net_a) or _RoutedNet(
                name=mirror.net_a, connected=False
            )
            partner_route = emit_from.get(partner) or _RoutedNet(
                name=partner, connected=False
            )
            report = self._mirror_report(
                mirror, primary_route, partner_route
            )
            out.mirror_reports.append(report)
            if not report["satisfied"] and mirror not in unsatisfied:
                unsatisfied.append(mirror)

        for constraint in sorted(
            length_constraints,
            key=lambda item: (str(item.net_a), str(item.net_b)),
        ):
            report = self._materialized_length_report(constraint, out.nets)
            out.length_reports.append(report)
            if (
                getattr(constraint, "hardness", "hard") == "hard"
                and not report["satisfied"]
                and constraint not in unsatisfied
            ):
                unsatisfied.append(constraint)

        out.unsatisfied = unsatisfied
        out.spacing_reports = check_spacing_geometry(
            out, constraints, obstacles, tech, rails
        )

        # WIDTH honesty (NO silent pass): a supra-min Width whose net did NOT
        # materialize the declared width is reported ``unsatisfied`` -- either
        # the net is unconnected (no room to route at the declared width) or it
        # connected but drew NO supra-min wire seg to carry the width.  A plain
        # Width widens every layer, while Width + TrunkLayer widens only the
        # effective structural carrier gi; a local min-width access segment is
        # therefore not evidence that the declared trunk width materialized.
        # The independent ``check_power_rail_width`` remains the geometric width
        # oracle.
        for net, wc in width_cons_by_net.items():
            nr = out.nets.get(net)
            connected = nr is not None and nr.connected
            scoped_carrier = width_carrier_gi_by_net.get(net)
            carrier_gis = (
                {scoped_carrier}
                if scoped_carrier is not None
                # Without a structural trunk, every actual route layer can
                # carry the width. A layer whose PDK minimum already meets the
                # target is just as valid as one widened above that minimum.
                else set(range(n_layers))
            )
            has_supra_min_carrier = False
            rn_emit = emit_from.get(net)
            if nr is not None and rn_emit is not None:
                emitted = emitted_by_net[net]
                if len(emitted.segment_wires) != len(rn_emit.segs):
                    raise AssertionError(
                        "emitted segment evidence count differs from routed segments"
                    )
                for segment, wire in zip(
                    rn_emit.segs,
                    emitted.segment_wires,
                ):
                    emitted_gi = tech.resolve_layer_to_gi(
                        (wire.layer, wire.datatype)
                    )
                    if emitted_gi != segment.gi:
                        raise AssertionError(
                            "emitted segment evidence layer differs from routed segment"
                        )
                    if segment.gi not in carrier_gis:
                        continue
                    xs = [point.x for point in wire.polygon.vertices]
                    ys = [point.y for point in wire.polygon.vertices]
                    narrow = min(max(xs) - min(xs), max(ys) - min(ys))
                    if narrow >= 2 * self._hw(net, segment.gi):
                        has_supra_min_carrier = True
                        break
            if not has_supra_min_carrier:
                for gi, rect in self._rail_by_net.get(net, ()):
                    if gi not in carrier_gis:
                        continue
                    x0, y0, x1, y1 = _norm(rect)
                    if min(x1 - x0, y1 - y0) >= 2 * self._hw(net, gi):
                        has_supra_min_carrier = True
                        break
            if (
                not connected
                or (net in width_widened and not has_supra_min_carrier)
            ):
                unsatisfied.append(wc)

        # HONEST success semantics, judged on the geometry the caller RECEIVES
        # (not a loop-break flag over a superseded round): all nets connected,
        # exact deck-connected cross-net isolation, zero DECK-VISIBLE sub-min-space
        # residuals (wires, own-artwork bands, cut spacing incl. artwork cuts;
        # judged at the deck's merged/net-blind/euclidean semantics --
        # ``_space_residual_scan``), and no unsatisfied constraint. success
        # means DECK-ALIGNED clean; the engine's STRICTER detector verdict is
        # kept verbatim as ``advisory_residuals`` / ``advisory_residual_pairs``
        # -- advisory, never a success blocker, never silently dropped. The
        # real KLayout deck remains the independent oracle.
        native_diagnostic_from = (
            emit_from if has_isolated_snapshot else committed
        )
        out.shorts = self._detect_shorts(native_diagnostic_from)
        out.unsatisfied = unsatisfied
        all_conn = all(nr.connected for nr in out.nets.values())
        final_all, final_deck = self._space_residual_scan(emit_from)
        out.success = (
            all_conn
            and emitted_deck_snapshot.report.cross_net_isolated
            and not final_deck
            and not unsatisfied
        )
        out.advisory_residuals = len(final_all)
        out.advisory_residual_pairs = final_all
        out.deck_residual_pairs = final_deck
        deck_merge_stops = tuple(
            NegotiationRouteStop(
                NegotiationStopReason.DECK_NET_MERGE,
                group[0],
                "final_snapshot",
                (tuple(group),),
                (("merged_nets", tuple(group)),),
            )
            for group in failed_merge_groups
        )
        open_stops = tuple(
            emit_route_stops.get(net) or NegotiationRouteStop(
                NegotiationStopReason.NO_LEGAL_ROUTE,
                net,
                "final_snapshot",
            )
            for net, net_route in out.nets.items()
            if (
                not open_state_from.get(net, _RoutedNet(name=net)).connected
                or (
                    has_isolated_snapshot
                    and net in emitted_deck_open_nets
                )
            )
        )
        out.route_stops = tuple(sorted(
            (*deck_merge_stops, *open_stops),
            key=lambda stop: stop.canonical_json(),
        ))
        out.route_stop_refs = tuple(
            stop.content_ref for stop in out.route_stops
        )
        if routing_envelope is not None:
            require_routed_containment(out, routing_envelope)
            out.routing_envelope_nm = routing_envelope.bbox_nm
        return out

    def _unrequired_planar_endpoint_domain(
        self, net, node, grid, *, work, diagnostic_work,
    ) -> _PlanarEndpointDomain:
        """Construct the ordinary domain after the caller resolves requirements."""
        x, y, gi = node
        atoms = (self._owned_terminal_component(net, x, y, gi) if work is None else
                 self._owned_terminal_component(net, x, y, gi,
                                                diagnostic_work=diagnostic_work))
        if not atoms:
            return _PlanarEndpointDomain(node, grid, node, ())
        width = int(self._tech.gi_min_width(gi))
        half = int(self._hw(net, gi))
        if width <= 0 or half <= 0 or grid <= 0:
            return _PlanarEndpointDomain(node, grid, node, ())
        if work is None:
            reached = {index for index, atom in enumerate(atoms)
                       if self._point_in_closed_rect(x, y, atom.rect)}
        else:
            reached = set()
            for index, atom in enumerate(atoms):
                work("domain.seed_atom")
                if self._point_in_closed_rect(x, y, atom.rect):
                    reached.add(index)
        pending = list(sorted(reached))
        for index in pending:
            self._watchdog_check(net, "planar_component_access")
            for other, atom in enumerate(atoms):
                if work is not None:
                    work("domain.throat_pair")
                if (other not in reached and self._owned_landing_has_minimum_throat(
                        atoms[index].rect, atom.rect, width)):
                    reached.add(other)
                    pending.append(other)

        regions = []
        for index in sorted(reached):
            if work is not None:
                work("domain.region_atom")
            self._watchdog_check(net, "planar_component_access")
            x0, y0, x1, y1 = _norm(atoms[index].rect)
            lo_x = -(-(x0 + half) // grid) * grid
            lo_y = -(-(y0 + half) // grid) * grid
            hi_x = ((x1 - half) // grid) * grid
            hi_y = ((y1 - half) // grid) * grid
            if lo_x <= hi_x and lo_y <= hi_y:
                regions.append((atoms[index], (lo_x, lo_y, hi_x, hi_y)))
        return _PlanarEndpointDomain(node, grid, None, tuple(regions))

    @staticmethod
    def _failure_endpoint(
        pin: Optional[Pin],
        node: Tuple[int, int, int],
    ) -> Tuple[Tuple[str, Any], ...]:
        """Immutable terminal identity used only by typed STOP diagnostics."""
        x, y, gi = node
        return (
            ("device", "" if pin is None else str(pin.device)),
            ("terminal", "" if pin is None else str(pin.terminal)),
            ("xy_nm", (int(x), int(y))),
            ("gi", int(gi)),
        )

    def _begin_route_failure_context(
        self,
        *,
        phase: str,
        source_pin: Optional[Pin],
        source: Tuple[int, int, int],
        goal_pin: Optional[Pin],
        goal: Tuple[int, int, int],
        window: Rect,
        allowed_layers: Sequence[int],
        trunk_gi: int,
    ) -> None:
        """Reset edge-local diagnostics and name the exact search obligation."""
        self._hard_rejections = {}
        self._hard_rejection_samples = {}
        self._route_failure_details = (
            ("phase", str(phase)),
            ("source", self._failure_endpoint(source_pin, source)),
            ("goal", self._failure_endpoint(goal_pin, goal)),
            ("window", tuple(_norm(window))),
            ("allowed_layers", tuple(sorted(set(int(v) for v in allowed_layers)))),
            ("trunk_gi", int(trunk_gi)),
        )

    def _requires_singleton_route(self, net: str) -> bool:
        """Whether a strategy must construct source-declared singleton artwork."""
        return False

    def _route_net(self, net: str, nodes: List[Tuple[int, int, int]],
                   lg: List[int],
                   node_pins: Optional[Sequence[Pin]] = None,
                   ) -> Optional[_RoutedNet]:
        self._rebuild_metal_index()
        rn = _RoutedNet(name=net)
        pins = list(node_pins or ())
        if len(pins) != len(nodes):
            pins = [None] * len(nodes)
        nodes, pins, endpoint_members = (
            self._contract_owned_component_endpoints(net, nodes, pins)
        )
        # RAIL-SPINE path: a net that declares a materialized @trunk(edge=) rail
        # is connected by a SHORT LOCAL DROP of each pin onto the owned rail metal
        # (the distributed spine), NOT an MST over the pins toward one hub node.
        rail = self._rail_by_net.get(net)
        if rail:
            return self._route_net_to_rail(
                net,
                nodes,
                lg,
                rn,
                rail,
                node_pins=pins,
                endpoint_members=endpoint_members,
            )
        edges = self._mst_edges(nodes)
        bx0, by0, bx1, by1 = self._bounds
        trunk = self._trunk_gi.get(net, 0)
        # PER-PIN ACCESS STATES: each pin enters the graph at its seed-metal node
        # AND (when they exist) at wireless escape landings on the metal above --
        # pin-local plus the complete exact drawn-component domain, with the
        # reliability model chosen across that whole component. These are
        # CANDIDATES, not geometry: each MST edge selects one
        # finite endpoint pair transactionally, reserves only that pair during
        # search, and commits only the winning route's vias.  They are generated
        # FRESH per edge (and retry) because earlier edges change ``_attempt_cuts``;
        # caching legal via sites before the MST would make later access stale.
        self._attempt_cuts = {}
        self._attempt_metal = {}
        self._attempt_idx = {}
        materialized_access_states: Dict[int, Tuple[int, int, int]] = {}
        planar_materialized_endpoints: Set[int] = set()
        regional_endpoint_materialized = False
        for (i, j) in edges:
            self._last_accepted_group_source_union = False
            self._last_accepted_group_source_candidate = None
            a, b = nodes[i], nodes[j]
            # PER-EDGE local window: the two endpoints' bbox grown by a bounded
            # margin (so a full-cell power net's MST decomposes into many CHEAP
            # local edges, not one whole-cell grid). Detours stay within the margin.
            axs = (a[0], b[0])
            ays = (a[1], b[1])
            span = max(max(axs) - min(axs), max(ays) - min(ays))
            margin = min(30000, max(12000, span // 2))
            window = (max(bx0, min(axs) - margin), max(by0, min(ays) - margin),
                      min(bx1, max(axs) + margin), min(by1, max(ays) + margin))
            self._begin_route_failure_context(
                phase="signal_edge",
                source_pin=pins[i],
                source=a,
                goal_pin=pins[j],
                goal=b,
                window=window,
                allowed_layers=lg,
                trunk_gi=trunk,
            )
            details = dict(self._route_failure_details)
            if len(endpoint_members[i]) > 1:
                details["source_owned_component_members"] = endpoint_members[i]
            if len(endpoint_members[j]) > 1:
                details["goal_owned_component_members"] = endpoint_members[j]
            self._route_failure_details = tuple(sorted(details.items()))
            # A wireless escape is emitted geometry too. Keep only access states
            # on the net's hard LayerSet so fallback A* cannot regain an off-cap
            # upper state after pattern-first correctly deferred.
            # Replays of this edge share one exact world-keyed access cache.  The
            # key includes committed generation, attempt geometry, conflict branch,
            # and history, so a changed world still recomputes while an unchanged
            # refinement stage cannot rebuild the same terminal domain.
            edge_access_scope = self._component_access.new_escape_scope()

            def _fresh_edge_access():
                with self._component_access.escape_scope(edge_access_scope):
                    def endpoint_access(index, endpoint, node):
                        materialized = materialized_access_states.get(index)
                        if materialized is not None:
                            # A prior MST edge already emitted the conductor/via at
                            # this exact state.  Reuse that physical junction without
                            # re-emitting its access via; a fresh alternative would
                            # join only the logical endpoint while splitting the
                            # emitted route into separate deck components.
                            return [(materialized, None)]
                        return self._endpoint_access(
                            endpoint=endpoint,
                            net=net,
                            node=node,
                            allowed_layers=lg,
                        )

                    starts = endpoint_access(i, "source", a)
                    goals = endpoint_access(j, "goal", b)
                    return starts, goals

            edge_starts, edge_goals = _fresh_edge_access()
            edge_materialized_terminal_states = tuple(sorted({
                tuple(materialized_access_states[endpoint_index])
                for endpoint_index in (i, j)
                if endpoint_index in materialized_access_states
            } | {
                tuple(required)
                for endpoint_index in (i, j)
                if (
                    required := getattr(
                        self, "_group_route_required_states", {}
                    ).get((net, tuple(nodes[endpoint_index])))
                ) is not None
            }))
            if not edge_starts or not edge_goals:
                failed_index = i if not edge_starts else j
                failed_node = nodes[failed_index]
                details = dict(self._route_failure_details)
                failed_endpoint = self._failure_endpoint(
                    pins[failed_index], failed_node
                )
                details["phase"] = "signal_pin_access"
                details["source"] = failed_endpoint
                details["goal"] = failed_endpoint
                details["window"] = tuple(_norm(self._bounds))
                details["access_state_count"] = 0
                self._route_failure_details = tuple(sorted(details.items()))
                self._edge_fail = (
                    "pin_access_no_state",
                    failed_endpoint,
                    self._hard_rejection_evidence(),
                )
                return None
            # #1 PATTERN-FIRST: a clean closed-form planar route (straight / L both
            # ways / Z with one mid track) between the two PIN endpoints, judged by
            # hard legality plus the shared present/history objective and emitted
            # directly only when the candidate set has a congestion-free member --
            # skipping the dense per-edge A* every multi-pin signal net paid. The
            # dense ``_candidate_lines`` ingests EVERY in-window device-finger edge,
            # so a signal net's tall-window edge built a ~234x254 grid (80248
            # expansions, the measured 100-470s/net); a clean edge needs none of it.
            xs, ys = self._signal_candidate_lines(a[:2], b[:2], window)
            def _attempt_edge(selected_starts, selected_goals):
                self._precheck_envelope_disconnected = False
                connectivity_witness = _HardConnectivityWitness()
                pattern_incumbent = _PatternSearchIncumbent()
                previous_witness = getattr(
                    self, "_active_hard_connectivity_witness", None
                )
                previous_incumbent = getattr(
                    self, "_active_pattern_search_incumbent", None
                )
                self._active_hard_connectivity_witness = connectivity_witness
                self._active_pattern_search_incumbent = pattern_incumbent
                try:
                    found_edge = self._signal_edge_pattern(
                        net,
                        a,
                        b,
                        selected_starts,
                        tuple(selected_goals.items()),
                        lg=lg,
                        trunk_gi=trunk,
                    )
                finally:
                    self._active_hard_connectivity_witness = previous_witness
                    self._active_pattern_search_incumbent = previous_incumbent
                if found_edge is not None:
                    return found_edge
                # #5 SPARSE FALLBACK: the clean pattern is BLOCKED -> search a
                # BOUNDED, finger-density-INDEPENDENT candidate grid (endpoints +
                # a capped uniform/coarse fill), NOT the dense obstacle-edge set.
                active_starts = [state for state, _via in selected_starts]
                active_goals = set(selected_goals)
                attempt_xs, attempt_ys, connectivity_proven = (
                    connectivity_witness.augment_candidate_graph(
                        active_starts,
                        active_goals,
                        xs,
                        ys,
                        set(lg),
                    )
                )
                previous_witness = getattr(
                    self, "_active_hard_connectivity_witness", None
                )
                self._active_hard_connectivity_witness = (
                    connectivity_witness if connectivity_proven else None
                )
                self._active_pattern_search_incumbent = pattern_incumbent
                try:
                    return self._route_edge(
                        selected_starts,
                        selected_goals,
                        net,
                        attempt_xs,
                        attempt_ys,
                        lg,
                        trunk,
                    )
                finally:
                    self._active_hard_connectivity_witness = previous_witness
                    self._active_pattern_search_incumbent = previous_incumbent

            frontier_starts, frontier_goals = (
                self._nearest_access_pair_frontier(
                    edge_starts, edge_goals
                )
            )
            complete_source_count = len(edge_starts)
            complete_goal_count = len(edge_goals)

            def _search_signal_domain(
                domain_starts,
                domain_goals,
                scope: str,
            ):
                nonlocal regional_endpoint_materialized
                details = dict(self._route_failure_details)
                details["source_access_state_count"] = (
                    complete_source_count
                )
                details["goal_access_state_count"] = complete_goal_count
                details["nearest_source_access_state_count"] = len(
                    frontier_starts
                )
                details["nearest_goal_access_state_count"] = len(
                    frontier_goals
                )
                details["signal_access_scope"] = str(scope)
                self._route_failure_details = tuple(
                    sorted(details.items())
                )
                self._last_accepted_access_states = None
                previous_materialized = getattr(
                    self, "_active_materialized_access_states", ()
                )
                self._active_materialized_access_states = (
                    edge_materialized_terminal_states
                )
                missing_routed_net = object()
                previous_routed_net = getattr(
                    self, "_active_access_routed_net", missing_routed_net
                )
                missing_endpoint_domains = object()
                previous_endpoint_domains = getattr(
                    self, "_active_endpoint_access_domains", missing_endpoint_domains
                )
                self._active_endpoint_access_domains = (
                    self._planar_endpoint_domain(
                        net, a, materialized_state=materialized_access_states.get(i)
                    ),
                    self._planar_endpoint_domain(
                        net, b, materialized_state=materialized_access_states.get(j)
                    ),
                )
                self._active_access_routed_net = rn
                missing_source_layers = object()
                previous_source_layers = getattr(self, '_active_group_source_union_layers', missing_source_layers)
                self._active_group_source_union_layers = tuple(lg)
                try:
                    result = self._route_access_transactions(
                        domain_starts,
                        {s3: via for (s3, via) in domain_goals},
                        net,
                        trunk,
                        _attempt_edge,
                        prevalidated=True,
                    )
                    if result is not None:
                        regional_endpoint_materialized |= any(
                            getattr(self, "_last_accepted_planar", (False, False))
                        )
                        planar_materialized_endpoints.update(
                            index for index, selected in zip(
                                (i, j), getattr(self, "_last_accepted_planar", (False, False))
                            ) if selected
                        )
                    return result
                finally:
                    if previous_source_layers is missing_source_layers:
                        del self._active_group_source_union_layers
                    else:
                        self._active_group_source_union_layers = previous_source_layers
                    if previous_routed_net is missing_routed_net:
                        del self._active_access_routed_net
                    else:
                        self._active_access_routed_net = previous_routed_net
                    if previous_endpoint_domains is missing_endpoint_domains:
                        del self._active_endpoint_access_domains
                    else:
                        self._active_endpoint_access_domains = previous_endpoint_domains
                    self._active_materialized_access_states = (
                        previous_materialized
                    )

            use_frontier = (
                frontier_starts != tuple(edge_starts)
                or frontier_goals != tuple(edge_goals)
            )
            signal_stages = (
                (
                    ("nearest_frontier", frontier_starts, frontier_goals),
                    ("complete_fallback", tuple(edge_starts), tuple(edge_goals)),
                )
                if use_frontier
                else (("complete", tuple(edge_starts), tuple(edge_goals)),)
            )
            found = None
            winning_starts = tuple(edge_starts)
            winning_goals = tuple(edge_goals)
            failure_kind = None
            shared_materialized_state = self._shared_materialized_access_state(
                edge_starts,
                edge_goals,
            )
            if (shared_materialized_state is not None
                    and ({i, j} & planar_materialized_endpoints)):
                # A previously selected component point is only a physical MST
                # junction while its actual wire and original terminal remain
                # connected. Equality of search coordinates alone is not proof.
                if (any(not self._owned_planar_access_joins_conductor(
                            net, nodes[index], shared_materialized_state,
                            rn.segs, rn.vias)
                        for index in (i, j) if index in planar_materialized_endpoints)
                        or not self._candidate_deck_connects_probes(
                            net, rn.segs, rn.vias, (a, b), include_fixed=True)):
                    self._edge_fail = ("planar_shared_endpoint_proof_failed", a, b)
                    return None
            if shared_materialized_state is not None:
                materialized_row = (shared_materialized_state, None)
                found = ([], [])
                winning_starts = (materialized_row,)
                winning_goals = (materialized_row,)
                self._last_accepted_access_states = (
                    shared_materialized_state,
                    shared_materialized_state,
                )
                self._last_accepted_terminal_probes = (
                    shared_materialized_state,
                    shared_materialized_state,
                )
            else:
                for stage_index, (scope, stage_starts, stage_goals) in enumerate(
                    signal_stages
                ):
                    if stage_index:
                        edge_starts, edge_goals = _fresh_edge_access()
                        complete_source_count = len(edge_starts)
                        complete_goal_count = len(edge_goals)
                        frontier_starts, frontier_goals = (
                            self._nearest_access_pair_frontier(
                                edge_starts, edge_goals
                            )
                        )
                        stage_starts = tuple(edge_starts)
                        stage_goals = tuple(edge_goals)
                    found = _search_signal_domain(
                        stage_starts, stage_goals, scope
                    )
                    failure_kind = (
                        None
                        if (
                            found is not None
                            or getattr(self, "_edge_fail", None) is None
                        )
                        else self._edge_fail[0]
                    )
                    if found is not None:
                        winning_starts = tuple(stage_starts)
                        winning_goals = tuple(stage_goals)
                        break
                    if failure_kind == "solve_budget":
                        break
            precheck_disconnected = bool(
                found is None and getattr(self, "_precheck_envelope_disconnected", False)
            )
            # The sparse first pass is deliberately obstacle-density-independent.
            # If it exhausts, try local causal strata before the complete legacy
            # projection.  Each pass retains descending rejection frequency.
            # A local per-edge cap stops only that pass; the global solve budget
            # remains authoritative across both passes without being reset.
            if (
                found is None
                and not precheck_disconnected
                and failure_kind not in {"solve_budget", "budget"}
            ):
                admitted_access_states = tuple(
                    state for state, _via in (*edge_starts, *edge_goals))
                refinement_passes = self._signal_causal_frontier_passes(
                    net,
                    xs,
                    ys,
                    window,
                    lg,
                    admitted_access_states,
                )
                priority_passes = ()
                if getattr(self, "_active_group_route_local_closure", False):
                    # Freeze these axes from the same sparse-search evidence as
                    # the original stages.  Do not run them before an original
                    # successful refinement: earlier port geometry stays intact.
                    priority_passes = self._group_endpoint_layer_frontier_prepasses(
                        net, xs, ys, window, lg, admitted_access_states,
                    )
                for projection_scope, refinement_stages in refinement_passes:
                    for refinement_index, (refined_xs, refined_ys) in enumerate(
                        refinement_stages,
                        start=1,
                    ):
                        details = dict(self._route_failure_details)
                        details["frontier_projection_scope"] = projection_scope
                        details.pop("frontier_graph_states", None)
                        details.pop("frontier_graph_cap", None)
                        self._route_failure_details = tuple(sorted(details.items()))
                        if not self._causal_frontier_graph_within_budget(
                            refined_xs,
                            refined_ys,
                            lg,
                            nominal_work=True,
                        ):
                            failure_kind = "budget"
                            break
                        xs, ys = refined_xs, refined_ys
                        edge_starts, edge_goals = _fresh_edge_access()
                        complete_source_count = len(edge_starts)
                        complete_goal_count = len(edge_goals)
                        frontier_starts, frontier_goals = (
                            self._nearest_access_pair_frontier(
                                edge_starts, edge_goals
                            )
                        )
                        found = _search_signal_domain(
                            edge_starts,
                            edge_goals,
                            (
                                f"{projection_scope}_refined_"
                                f"{refinement_index}_of_{len(refinement_stages)}"
                            ),
                        )
                        if found is not None:
                            winning_starts = tuple(edge_starts)
                            winning_goals = tuple(edge_goals)
                            break
                        failure_kind = (
                            None
                            if getattr(self, "_edge_fail", None) is None
                            else self._edge_fail[0]
                        )
                        if failure_kind in {"solve_budget", "budget"}:
                            break
                    if found is not None or failure_kind == "solve_budget":
                        break
                original_stop = getattr(self, "_edge_fail", None)
                if (found is None and failure_kind == "budget" and priority_passes
                        and isinstance(original_stop, tuple)
                        and original_stop[-1] == "causal_frontier_graph"):
                    original_details = tuple(self._route_failure_details)
                    # The original stages have all had their normal opportunity.
                    # At their allocation gate, try endpoint-layer projections
                    # under the same full search world and cumulative budgets.
                    for projection_scope, stages in priority_passes:
                        for index, (refined_xs, refined_ys) in enumerate(stages, start=1):
                            details = dict(self._route_failure_details)
                            details["frontier_projection_scope"] = projection_scope
                            details.pop("frontier_graph_states", None)
                            details.pop("frontier_graph_cap", None)
                            self._route_failure_details = tuple(sorted(details.items()))
                            if not self._causal_frontier_graph_within_budget(
                                    refined_xs, refined_ys, lg, nominal_work=True):
                                break
                            xs, ys = refined_xs, refined_ys
                            edge_starts, edge_goals = _fresh_edge_access()
                            complete_source_count = len(edge_starts)
                            complete_goal_count = len(edge_goals)
                            frontier_starts, frontier_goals = self._nearest_access_pair_frontier(
                                edge_starts, edge_goals)
                            found = _search_signal_domain(
                                edge_starts, edge_goals,
                                f"{projection_scope}_refined_{index}_of_{len(stages)}")
                            if found is not None:
                                winning_starts = tuple(edge_starts)
                                winning_goals = tuple(edge_goals)
                                break
                            priority_failure = getattr(self, "_edge_fail", None)
                            if priority_failure is not None and priority_failure[0] in {"budget", "solve_budget"}:
                                break
                        if found is not None:
                            break
                        if (getattr(self, "_edge_fail", None) is not None
                                and self._edge_fail[0] == "solve_budget"):
                            failure_kind = "solve_budget"
                            break
                    if found is None and failure_kind != "solve_budget":
                        self._edge_fail = original_stop
                        details = dict(original_details)
                        details["endpoint_layer_prepass_solve_expansions_total"] = int(
                            getattr(self, "_solve_expansions_total", 0))
                        self._route_failure_details = tuple(sorted(details.items()))
            # Exhausting a bounded local graph is not a proof on the
            # whole routing envelope. Retain the complete immutable blocker
            # evidence and expose its real escape boundaries before giving up.
            # Bounds grow monotonically; unchanged bounds or the existing work
            # budget terminate this search without weakening any predicate.
            escape_window_index = 0
            while (
                found is None
                and not precheck_disconnected
                and self._local_search_domain_exhausted(failure_kind)
            ):
                expanded_window = self._causal_blocker_escape_window(window, lg)
                if expanded_window == window:
                    break
                previous_window = window
                window = expanded_window
                escape_window_index += 1
                details = dict(self._route_failure_details)
                details["local_exhausted_window"] = previous_window
                details["local_search_failure_kind"] = failure_kind
                if failure_kind == "static_disconnected":
                    details["local_static_disconnected_window"] = previous_window
                details["window"] = window
                details["candidate_window_scope"] = "immutable_blocker_escape"
                # The next sparse graph has no frontier projection yet.  If it
                # reaches a work limit, do not attach the prior window's plan.
                for key in (
                    "frontier_projection_scope",
                    "frontier_graph_states",
                    "frontier_graph_cap",
                    "frontier_local_stage_count",
                    "frontier_complete_stage_count",
                    "frontier_local_max_graph_states",
                    "frontier_complete_max_graph_states",
                ):
                    details.pop(key, None)
                self._route_failure_details = tuple(sorted(details.items()))
                xs, ys = self._signal_candidate_lines(a[:2], b[:2], window)
                edge_starts, edge_goals = _fresh_edge_access()
                complete_source_count, complete_goal_count = len(edge_starts), len(edge_goals)
                found = _search_signal_domain(
                    edge_starts, edge_goals, f"complete_escape_window_{escape_window_index}")
                if found is not None:
                    winning_starts, winning_goals = tuple(edge_starts), tuple(edge_goals)
                    break
                failure_kind = (self._edge_fail[0]
                                if getattr(self, "_edge_fail", None) is not None else None)
                if failure_kind in {"solve_budget", "budget"}:
                    break
                admitted_access_states = tuple(
                    state for state, _via in (*edge_starts, *edge_goals))
                # Retain the sparse-window evidence, but defer the costly
                # priority projection until the original scopes hit the cap.
                expanded_priority_basis = None
                if getattr(self, "_active_group_route_local_closure", False):
                    expanded_priority_basis = (
                        tuple(xs), tuple(ys), admitted_access_states,
                        dict(getattr(self, "_hard_rejection_samples", {})),
                    )
                refinement_passes = self._signal_causal_frontier_passes(
                    net, xs, ys, window, lg, admitted_access_states,
                )
                for projection_scope, refinement_stages in refinement_passes:
                    for refinement_index, (refined_xs, refined_ys) in enumerate(
                        refinement_stages, 1,
                    ):
                        details = dict(self._route_failure_details)
                        details["frontier_projection_scope"] = projection_scope
                        details.pop("frontier_graph_states", None)
                        details.pop("frontier_graph_cap", None)
                        self._route_failure_details = tuple(sorted(details.items()))
                        if not self._causal_frontier_graph_within_budget(
                                refined_xs, refined_ys, lg, nominal_work=True):
                            failure_kind = "budget"
                            break
                        xs, ys = refined_xs, refined_ys
                        edge_starts, edge_goals = _fresh_edge_access()
                        complete_source_count, complete_goal_count = len(edge_starts), len(edge_goals)
                        found = _search_signal_domain(
                            edge_starts, edge_goals,
                            f"{projection_scope}_escape_window_{escape_window_index}_refined_{refinement_index}")
                        if found is not None:
                            winning_starts, winning_goals = tuple(edge_starts), tuple(edge_goals)
                            break
                        failure_kind = (self._edge_fail[0]
                                        if getattr(self, "_edge_fail", None) is not None else None)
                        if failure_kind in {"solve_budget", "budget"}:
                            break
                    if found is not None or failure_kind == "solve_budget":
                        break
                original_stop = getattr(self, "_edge_fail", None)
                if (found is None and failure_kind == "budget"
                        and expanded_priority_basis is not None
                        and isinstance(original_stop, tuple)
                        and original_stop[-1] == "causal_frontier_graph"):
                    original_details = tuple(self._route_failure_details)
                    basis_xs, basis_ys, basis_states, basis_samples = expanded_priority_basis
                    self._watchdog_check(net, "group_expanded_endpoint_prepass")
                    priority_passes = self._group_endpoint_layer_frontier_prepasses(
                        net, basis_xs, basis_ys, window, lg, basis_states,
                        rejection_samples=basis_samples,
                    )
                    self._watchdog_check(net, "group_expanded_endpoint_prepass")
                    # Original local and complete scopes have already run.
                    for projection_scope, stages in priority_passes:
                        for index, (refined_xs, refined_ys) in enumerate(stages, start=1):
                            details = dict(self._route_failure_details)
                            details["frontier_projection_scope"] = projection_scope
                            details.pop("frontier_graph_states", None)
                            details.pop("frontier_graph_cap", None)
                            self._route_failure_details = tuple(sorted(details.items()))
                            if not self._causal_frontier_graph_within_budget(
                                    refined_xs, refined_ys, lg, nominal_work=True):
                                break
                            xs, ys = refined_xs, refined_ys
                            edge_starts, edge_goals = _fresh_edge_access()
                            complete_source_count = len(edge_starts)
                            complete_goal_count = len(edge_goals)
                            frontier_starts, frontier_goals = self._nearest_access_pair_frontier(
                                edge_starts, edge_goals)
                            found = _search_signal_domain(
                                edge_starts, edge_goals,
                                f"{projection_scope}_refined_{index}_of_{len(stages)}")
                            if found is not None:
                                winning_starts = tuple(edge_starts)
                                winning_goals = tuple(edge_goals)
                                break
                            priority_failure = getattr(self, "_edge_fail", None)
                            if priority_failure is not None and priority_failure[0] in {"budget", "solve_budget"}:
                                break
                        if found is not None:
                            break
                        if (getattr(self, "_edge_fail", None) is not None
                                and self._edge_fail[0] == "solve_budget"):
                            failure_kind = "solve_budget"
                            break
                    if found is None and failure_kind != "solve_budget":
                        self._edge_fail = original_stop
                        details = dict(original_details)
                        details["endpoint_layer_prepass_solve_expansions_total"] = int(
                            getattr(self, "_solve_expansions_total", 0))
                        self._route_failure_details = tuple(sorted(details.items()))
            # A pin seed can lie on the edge of a wider, genuinely drawn
            # terminal component.  If the complete ordinary group-local search
            # exhausted, try a finite set of wire centres inside that component.
            # Keep this separate from the shared-state shortcut above: an
            # inferred search coordinate is never a materialized connection.
            if (found is None
                    and self._local_search_domain_exhausted(failure_kind)
                    and getattr(self, "_active_group_route_local_closure", False)):
                planar_starts = (() if i in materialized_access_states else
                    self._owned_planar_access_candidates(net, a))
                planar_goals = (() if j in materialized_access_states else
                    self._owned_planar_access_candidates(net, b))
                if planar_starts or planar_goals:
                    edge_starts, edge_goals = _fresh_edge_access()
                    start_states = {state for state, _via in edge_starts}
                    goal_states = {state for state, _via in edge_goals}
                    # Never turn a shared search coordinate into a zero-wire
                    # proof. Existing materialized endpoint handling is above.
                    shared = ((set(planar_starts) | start_states)
                              & (set(planar_goals) | goal_states))
                    start_trials = tuple(edge_starts) + tuple(
                        (state, None) for state in planar_starts
                        if state not in shared and state not in start_states)
                    goal_trials = tuple(edge_goals) + tuple(
                        (state, None) for state in planar_goals
                        if state not in shared and state not in goal_states)
                    if (len(start_trials) > len(edge_starts)
                            or len(goal_trials) > len(edge_goals)):
                        found = _search_signal_domain(
                            start_trials, goal_trials, "owned_planar_fallback")
                        if found is not None:
                            selected = getattr(self, "_last_accepted_access_states", None)
                            actual_segs, actual_vias = found
                            valid = isinstance(selected, tuple) and len(selected) == 2
                            if valid:
                                for node, state, candidates in (
                                    (a, selected[0], planar_starts),
                                    (b, selected[1], planar_goals),
                                ):
                                    if (state in candidates and not
                                            self._owned_planar_access_joins_conductor(
                                                net, node, state, actual_segs,
                                                tuple(rn.vias) + tuple(actual_vias))):
                                        valid = False
                                        break
                            if valid:
                                valid = self._candidate_deck_connects_probes(
                                    net, tuple(rn.segs) + tuple(actual_segs),
                                    tuple(rn.vias) + tuple(actual_vias),
                                    (a, b), include_fixed=True)
                            if valid:
                                winning_starts, winning_goals = start_trials, goal_trials
                                planar_materialized_endpoints.update(
                                    index for index, state, candidates in (
                                        (i, selected[0], planar_starts),
                                        (j, selected[1], planar_goals),
                                    ) if state in candidates)
                            else:
                                found = None
                                self._edge_fail = ("planar_endpoint_proof_failed", a, b)
            if found is None:
                if self.debug:
                    print(f"  [gridless] {net}: edge {a}->{b} FAIL "
                          f"{getattr(self, '_edge_fail', None)}", flush=True)
                return None  # caller rips up a blocker + re-queues.
            edge_starts = list(winning_starts)
            edge_goals = list(winning_goals)
            segs, vias = found
            accepted_access_states = getattr(
                self, "_last_accepted_access_states", None
            )
            protected_access_states: Tuple[Tuple[int, int, int], ...] = ()
            protected_terminal_probes: Tuple[Tuple[int, int, int], ...] = ()
            if (
                isinstance(accepted_access_states, tuple)
                and len(accepted_access_states) == 2
                and all(
                    isinstance(state, tuple) and len(state) == 3
                    for state in accepted_access_states
                )
            ):
                protected_access_states = tuple(
                    tuple(int(value) for value in state)
                    for state in accepted_access_states
                )
                for endpoint_index, state in zip(
                    (i, j), protected_access_states
                ):
                    canonical_state = state
                    previous_state = materialized_access_states.get(
                        endpoint_index
                    )
                    if (
                        previous_state is not None
                        and previous_state != canonical_state
                    ):
                        self._edge_fail = (
                            "materialized_endpoint_access_changed",
                            endpoint_index,
                            previous_state,
                            canonical_state,
                        )
                        return None
                    materialized_access_states[endpoint_index] = canonical_state
                accepted_terminal_probes = getattr(
                    self, "_last_accepted_terminal_probes", None
                )
                if (
                    isinstance(accepted_terminal_probes, tuple)
                    and len(accepted_terminal_probes) == 2
                    and all(
                        isinstance(state, tuple) and len(state) == 3
                        for state in accepted_terminal_probes
                    )
                ):
                    protected_terminal_probes = tuple(
                        tuple(int(value) for value in state)
                        for state in accepted_terminal_probes
                    )
                else:
                    replayed_probes = self._candidate_terminal_probe_states(
                        net,
                        protected_access_states,
                        materialized_access_states=(
                            edge_materialized_terminal_states
                        ),
                    )
                    if replayed_probes is not None:
                        protected_terminal_probes = tuple(replayed_probes)
            # Each MST edge is searched atomically against the earlier attempt
            # ledger, but its landing-pad model is still chosen inside that one
            # edge.  Before committing it, satisfy the pad-spacing obligation for
            # the complete same-net via group accumulated across prior edges.
            # Earlier geometry stays immutable.  New internal hops may shrink; a
            # selected endpoint-access may do so only when this exact transaction
            # domain explicitly offers n=1 at the same authorized landing.
            prior_via_count = len(rn.vias)
            combined_vias = list(rn.vias) + list(vias)
            access_vias = tuple(
                access_via
                for _state, access_via in (
                    tuple(edge_starts) + tuple(edge_goals)
                )
                if access_via is not None
            )
            access_via_ids = {id(access_via) for access_via in access_vias}
            authorized_single_sites = {
                (access_via.x, access_via.y, access_via.lo)
                for access_via in access_vias
                if self._site_n(access_via) == 1
            }

            def via_covers_access_state(
                via: _ViaSite,
                state: Tuple[int, int, int],
            ) -> bool:
                x, y, gi = state
                below_half, above_half = self._via_pad_half_for(via)
                if int(gi) == int(via.lo):
                    half = int(below_half)
                elif int(gi) == int(via.lo) + 1:
                    half = int(above_half)
                else:
                    return False
                return self._point_in_closed_rect(
                    int(x),
                    int(y),
                    (
                        int(via.x) - half,
                        int(via.y) - half,
                        int(via.x) + half,
                        int(via.y) + half,
                    ),
                )

            mutable_indices = {
                index
                for index in range(prior_via_count, len(combined_vias))
                if (
                    id(combined_vias[index]) not in access_via_ids
                    or (
                        self._site_n(combined_vias[index]) != 1
                        and (
                            combined_vias[index].x,
                            combined_vias[index].y,
                            combined_vias[index].lo,
                        ) in authorized_single_sites
                    )
                )
                and not any(
                    via_covers_access_state(combined_vias[index], state)
                    for state in protected_access_states
                )
            }
            resolved_vias = self._resolve_pattern_via_models(
                net,
                combined_vias,
                mutable_indices=mutable_indices,
                segs=tuple(rn.segs) + tuple(segs),
            )
            if resolved_vias is None:
                candidate_segs = tuple(rn.segs) + tuple(segs)
                repair_segs = self._materialize_candidate_via_wire_merges(
                    net,
                    combined_vias,
                    candidate_segs,
                    prior_via_count=prior_via_count,
                    prior_seg_count=len(rn.segs),
                    immutable_via_indices=(
                        set(range(len(combined_vias))) - mutable_indices
                    ),
                )
                if repair_segs:
                    repaired_candidate_segs = candidate_segs + repair_segs
                    resolved_vias = self._resolve_pattern_via_models(
                        net,
                        combined_vias,
                        mutable_indices=mutable_indices,
                        segs=repaired_candidate_segs,
                    )
                    if resolved_vias is not None:
                        segs = list(segs) + list(repair_segs)
            if resolved_vias is None:
                repaired = self._resolve_candidate_same_net_fills(
                    net, rn, segs, vias,
                    mutable_indices={index-prior_via_count for index in mutable_indices},
                )
                if repaired is not None:
                    segs, vias = repaired
                    resolved_vias = list(rn.vias) + list(vias)
            if resolved_vias is None:
                candidate_segs = tuple(rn.segs) + tuple(segs)
                pad_conflicts = self._pattern_via_pad_visible_conflicts(
                    net,
                    combined_vias,
                    candidate_segs,
                )
                wire_conflicts = self._pattern_via_wire_conflicts(
                    net,
                    combined_vias,
                    candidate_segs,
                )
                details = dict(self._route_failure_details)
                details.update(dict(self._candidate_via_conflict_details(
                    net,
                    combined_vias,
                    candidate_segs,
                    pad_conflicts,
                    wire_conflicts,
                )))
                self._route_failure_details = tuple(sorted(details.items()))
                self._edge_fail = (
                    "candidate_via_pad_spacing_conflict",
                    (
                        pad_conflicts,
                        wire_conflicts,
                    ),
                )
                if self.debug:
                    print(
                        f"  [gridless] {net}: edge {a}->{b} FAIL "
                        f"{self._edge_fail}",
                        flush=True,
                    )
                return None
            combined_segs = tuple(rn.segs) + tuple(segs)

            def resolved_geometry_covers_state(
                state: Tuple[int, int, int],
            ) -> bool:
                x, y, gi = state
                return any(
                    int(seg.gi) == int(gi)
                    and self._point_in_closed_rect(
                        int(x),
                        int(y),
                        _seg_bbox(
                            seg.ax,
                            seg.ay,
                            seg.bx,
                            seg.by,
                            self._segment_half_width(net, seg),
                        ),
                    )
                    for seg in combined_segs
                ) or any(
                    via_covers_access_state(via, state)
                    for via in resolved_vias
                )

            if not all(
                resolved_geometry_covers_state(state)
                for state in protected_access_states
            ):
                self._edge_fail = (
                    "candidate_endpoint_access_missing_after_group_resolution",
                    len(protected_access_states),
                )
                return None
            vias = resolved_vias[prior_via_count:]
            if (
                protected_access_states
                and self._candidate_access_deck_witness_required(
                    protected_access_states
                )
                and not self._candidate_access_deck_connected(
                    net,
                    segs,
                    vias,
                    protected_access_states,
                )
            ):
                self._edge_fail = (
                    "candidate_endpoint_deck_disconnected_after_group_resolution",
                    len(protected_access_states),
                )
                return None
            if (
                protected_access_states
                and (
                    len(protected_terminal_probes) != 2
                    or not self._candidate_terminal_deck_connected_after_group_resolution(
                        net,
                        rn.segs,
                        segs,
                        resolved_vias,
                        protected_terminal_probes,
                    )
                )
            ):
                self._edge_fail = (
                    "candidate_terminal_deck_disconnected_after_group_resolution",
                    len(protected_access_states),
                )
                return None
            if ({i, j} & planar_materialized_endpoints):
                # Model resolution may change via footprints. Replay the actual
                # original source endpoints and the selected planar wire after
                # resolution, including all prior MST geometry, before commit.
                if (any(not self._owned_planar_access_joins_conductor(
                            net, nodes[index], materialized_access_states[index],
                            combined_segs, resolved_vias)
                        for index in (i, j) if index in planar_materialized_endpoints)
                        or not self._candidate_terminal_deck_connected_after_group_resolution(
                            net, rn.segs, segs, resolved_vias, (a, b))):
                    self._edge_fail = ("planar_terminal_after_group_resolution", a, b)
                    return None
            if self._candidate_mim_metal_failure(net, segs, vias, prior=rn):
                self._edge_fail = (
                    "candidate_mim_metal_clearance_after_group_resolution",
                    self._hard_rejection_evidence(),
                )
                return None
            if (getattr(self, '_active_group_route_local_closure', False)
                    and getattr(self, '_last_accepted_group_source_union', False)):
                baseline = getattr(self, '_active_group_route_union_baseline', None)
                accepted = getattr(self, '_last_accepted_group_source_candidate', None)
                if (not isinstance(baseline, _RoutedNet) or baseline.name != net
                        or accepted is None or accepted[0] is not self._book
                        or accepted[1] is not baseline or accepted[2] is not rn):
                    self._edge_fail = ('candidate_source_union_unknown', 'missing_baseline')
                    return None
                _, _, _, accepted_segs, accepted_vias, accepted_probes, accepted_states = accepted
                if (len(segs) < len(accepted_segs) or len(vias) != len(accepted_vias)
                        or any(a is not b for a, b in zip(segs, accepted_segs))
                        or any(a is not b for a, b in zip(vias, accepted_vias))):
                    self._edge_fail = ('candidate_source_union_unknown', 'outer_candidate_changed')
                    return None
                previous = _RoutedNet(net, segs=list(baseline.segs) + list(rn.segs),
                                      vias=list(baseline.vias) + list(rn.vias))
                accepted_route = _RoutedNet(net, segs=list(previous.segs) + list(accepted_segs),
                                            vias=list(previous.vias) + list(accepted_vias))
                final = _RoutedNet(net, segs=list(previous.segs) + list(segs),
                                   vias=list(previous.vias) + list(vias))
                before_metal = self._net_metal(previous)[0]
                accepted_metal = self._net_metal(accepted_route)[0]
                final_metal = self._net_metal(final)[0]
                current = _RoutedNet(net, segs=list(rn.segs) + list(segs),
                                     vias=list(rn.vias) + list(vias))
                accepted_report = self._deck_negotiation_snapshot({net: accepted_route}).report
                final_report = self._deck_negotiation_snapshot({net: final}).report
                if (self._authenticated_owned_book_metal(net, current) is None
                        or not self._ordinary_repair_preserves_world(net, before_metal, final_metal)
                        or not self._repair_preserves_world(net, before_metal, final_metal)
                        or not self._ordinary_repair_preserves_world(net, accepted_metal, final_metal)
                        or not self._repair_preserves_world(net, accepted_metal, final_metal)
                        or not final_report.cross_net_isolated
                        or not set(final_report.unresolved_probes) <= set(accepted_report.unresolved_probes)
                        or not set(final_report.ambiguous_nets) <= set(accepted_report.ambiguous_nets)
                        or not self._candidate_deck_connects_probes(net, final.segs, final.vias, accepted_probes, include_fixed=True)
                        or not self._candidate_deck_connects_probes(net, final.segs, final.vias, accepted_states, include_fixed=True)):
                    self._edge_fail = ('candidate_source_union_unknown', 'outer_resolution')
                    return None
            rn.segs.extend(segs)
            rn.vias.extend(vias)
            for v in vias:
                self._add_attempt_cut(v)
            # register this edge's metal so LATER edges of the same net pay the
            # net-blind band price near it (overlap/merge stays free) -- the
            # jittered-parallel-track self-gap fix. Index rebuilt per edge
            # (metal is static within one edge's A*).
            for s in segs:
                self._attempt_metal.setdefault(s.gi, []).append(
                    _seg_bbox(s.ax, s.ay, s.bx, s.by, self._hw(net, s.gi)))
            for v in vias:
                bh2, ah2 = self._via_pad_half_for(v)
                self._attempt_metal.setdefault(v.lo, []).append(
                    (v.x - bh2, v.y - bh2, v.x + bh2, v.y + bh2))
                self._attempt_metal.setdefault(v.lo + 1, []).append(
                    (v.x - ah2, v.y - ah2, v.x + ah2, v.y + ah2))
            self._attempt_idx = {}
            for gi2, rects2 in self._attempt_metal.items():
                idx2 = _SpatialIndex()
                for r2 in rects2:
                    idx2.add(net, r2)
                self._attempt_idx[gi2] = idx2
        # DEDUP identical escape vias a pin shares across several MST edges.
        seen_v: Set[Tuple[int, int, int]] = set()
        uniq: List[_ViaSite] = []
        for v in rn.vias:
            k = (v.x, v.y, v.lo)
            if k in seen_v:
                continue
            seen_v.add(k)
            uniq.append(v)
        rn.vias = uniq
        # MERGE-OR-CLEAR materialization (R7 closure): the band price steers
        # the A*, but where the search graph offers no alternative the attempt
        # still commits same-net metal in the OPEN band (0, min_space) of its
        # own geometry -- rip-up re-prices and re-produces near-identical
        # jittered stubs (the measured 10-pair plateau across pres 2->64).
        # Close every such gap NOW by metallising it (deck-merge semantics),
        # so the committed artwork SATISFIES the touch-or-clear obligation the
        # exact detector judges, instead of endlessly re-negotiating it.
        self._legalize_same_net(net, rn)
        if regional_endpoint_materialized and getattr(self, "_active_group_route_local_closure", False):
            from .source_junction_kernel import classify_candidate_source_junction
            status, repair = classify_candidate_source_junction(
                self, net, _RoutedNet(name=net), rn.segs, rn.vias, lg,
                (tuple(nodes[0]), tuple(nodes[1])),
            )
            if status == "REPAIRED":
                rn.segs, rn.vias = repair
            if status in {"NO_NEW_SOURCE_WIDTH", "REPAIRED"}:
                if any(not self._candidate_deck_connects_probes(
                        net, rn.segs, rn.vias, (tuple(nodes[0]), tuple(node)), include_fixed=True
                ) for node in nodes[1:]):
                    status = "UNKNOWN"
            if status not in {"NO_NEW_SOURCE_WIDTH", "REPAIRED"}:
                self._edge_fail = ("symbolic_domain_incomplete", "source_junction_unverified")
                return None
        rn.connected = True
        return rn



_FIXED_INDEX_SEAL_PRODUCERS = (GridlessEngine.route, GridlessEngine._watchdog_check,
                               _SpatialIndex, _SpatialIndex.__init__, _SpatialIndex.add,
                               _SpatialIndex.query, _norm)
