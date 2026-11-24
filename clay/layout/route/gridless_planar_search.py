"""Planar endpoint domains, region A* search, multi-access transactions, and causal frontier mixin."""
from __future__ import annotations

from array import array as _base_array
import builtins
from collections import Counter, OrderedDict
from dataclasses import replace
from functools import lru_cache
import heapq as _base_heapq
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from clay._native import load_native
from clay.layout.route.component_access_session import PartitionContext
from clay.layout.route.engine import Pin
from clay.layout.route.engine_maze import _NM_PER_UM as _base_NM_PER_UM
from clay.layout.route.gridless_types import (
    NegotiationRouteStop,
    NegotiationStopReason,
    Rect,
    _AccessStaticDisconnectionRecorder,
    _CONG_WEIGHT,
    _COST_FACTOR_SCALE,
    _CartesianSearchGraph,
    _EXACT_GOAL_INDEX_VISIT_WORK,
    _ExactGoalPotentialIndex,
    _HardConnectivityWitness,
    _M1_ROUTE_COST_PER_NM,
    _OFF_PREF_COST_PER_NM,
    _OwnedTerminalAtom,
    _PatternSearchIncumbent,
    _PlanarEndpointDomain,
    _RoutedNet as _base_RoutedNet,
    _STEP_COST_PER_NM,
    _SpatialIndex,
    _TRUNK_BELOW_COST_PER_NM,
    _TRUNK_SPAN_PITCHES,
    _TURN_COST_PITCHES,
    _VIA_COST,
    _ViaSite,
    _WireSeg,
    _advance_planar_region_step as _base_advance_planar_region_step,
    _closed_rects_touch as _base_closed_rects_touch,
    _eg_attr,
    _norm,
    _owned_terminal_atom_key,
    _rects_clear as _base_rects_clear,
    _region_via_goal_prefix_lower_bound as _base_region_via_goal_prefix_lower_bound,
    _region_via_goal_waypoint as _base_region_via_goal_waypoint,
    _region_via_target_prunable as _base_region_via_target_prunable,
    _seg_bbox,
    _sep as _base_sep,
)
from clay.layout.route.region_event_geometry import (
    project_portal as _ORIG_PROJECT_PORTAL,
    project_target as _ORIG_PROJECT_TARGET,
    settled_axis_prefix as _ORIG_SETTLED_AXIS_PREFIX,
)
from clay.layout.route.via_model import landing_rect_for_cut, subtract_rects

_closed_rects_touch = _base_closed_rects_touch
_base_bytearray = builtins.bytearray

_EDGE_CTX_FORBIDDEN_OVERRIDES = (
    "_wire_legal",
    "_wire_legal_single",
    "_wire_static_legality",
    "_compute_wire_static_legality",
    "_static_or_direct_obstacle_scan",
    "_clear_obstacles",
    "_reject_intent_spacing",
    "_reject_foreign_committed_metal_spacing",
    "_reject_foreign_committed_metal_contact",
    "_scan_committed_metal",
    "_static_or_direct_committed_metal_scan",
    "_via_legal",
    "_via_legal_with_active_scans",
    "_evaluate_via_static_legality",
    "_compute_via_static_legality",
    "_reject_fixed_same_net_via_pad_spacing",
    "_reject_committed_via_pad_spacing",
    "_scan_committed_cuts",
    "_static_or_direct_committed_cut_scan",
    "_local_fixed_cuts",
    "_raw_gap_metallised",
    "_wide_spacing_conflicts",
    "_wide_spacing_neighbors",
    "_wide_indexes",
    "_edge_objective",
    "_edge_cost",
    "_edge_congestion",
    "_box_congestion",
    "_via_congestion",
    "_via_cut_congestion",
    "_history_congestion_units",
    "_functional_single_via_penalty",
    "_terminal_via_objective",
    "_via_successor_last_dir",
    "_hw",
    "_record_hard_rejection",
    "_reject_mim_bottom_routing_clearance",
    "_reject_mim_derived_unrelated_clearance",
    "_reject_mim_carve_bottom_interaction",
    "_reject_group_route_local_wire",
    "_reject_group_route_global_wire",
    "_reject_group_route_local_via",
    "_reject_group_route_global_via",
)


@lru_cache(maxsize=64)
def _pareto_euclid_pairs(sp: int) -> Tuple[Tuple[int, int], ...]:
    """Maximal integer (sx, sy) corners with sx*sx + sy*sy < sp*sp."""
    if sp <= 0:
        return ((0, 0),)
    limit = sp * sp - 1
    pairs = []
    prev_sx = -1
    for sy in range(sp - 1, -1, -1):
        sx = math.isqrt(limit - sy * sy)
        if sx > prev_sx:
            pairs.append((sx, sy))
            prev_sx = sx
    return tuple(pairs)


@lru_cache(maxsize=64)
def _euclid_poly_template(sp: int) -> Tuple[Tuple[bool, int, bool, int], ...]:
    """Axis-aligned SimplePolygon boundary offsets matching all Pareto boxes."""
    ps = _pareto_euclid_pairs(int(sp))
    if len(ps) <= 1:
        return ()
    tpl = []
    for i, (dx, dy) in enumerate(ps):
        if i == 0:
            tpl.append((False, -dx, True, dy))
        tpl.append((True, dx, True, dy))
        if i + 1 < len(ps):
            tpl.append((True, dx, True, ps[i + 1][1]))
    for i in range(len(ps) - 1, -1, -1):
        dx, dy = ps[i]
        tpl.append((True, dx, False, -dy))
        if i > 0:
            tpl.append((True, ps[i - 1][0], False, -dy))
    for i, (dx, dy) in enumerate(ps):
        tpl.append((False, -dx, False, -dy))
        if i + 1 < len(ps):
            tpl.append((False, -dx, False, -ps[i + 1][1]))
    for i in range(len(ps) - 1, 0, -1):
        dx, dy = ps[i]
        tpl.append((False, -dx, True, dy))
        tpl.append((False, -ps[i - 1][0], True, dy))
    return tuple(tpl)


def _reduce_1d_pareto(
    points: Sequence[Tuple[int, int]], rate: int,
) -> Tuple[Tuple[int, int], ...]:
    """Return the exact minimal subset for ``min_i(v_i + |p - p_i| * rate)``."""
    best_at: Dict[int, int] = {}
    for p, v in points:
        prev = best_at.get(p)
        if prev is None or v < prev:
            best_at[p] = v
    kept_left: List[Tuple[int, int]] = []
    best_l: Optional[int] = None
    for p, v in sorted(best_at.items()):
        if best_l is None or v < best_l + p * rate:
            kept_left.append((p, v))
            val = v - p * rate
            if best_l is None or val < best_l:
                best_l = val
    kept: List[Tuple[int, int]] = []
    best_r: Optional[int] = None
    for p, v in reversed(kept_left):
        if best_r is None or v < best_r - p * rate:
            kept.append((p, v))
            val = v + p * rate
            if best_r is None or val < best_r:
                best_r = val
    kept.reverse()
    return tuple(kept)


def _eval_1d_pareto(
    pareto: Tuple[Tuple[int, int], ...], coord: int, rate: int,
) -> int:
    p0, v0 = pareto[0]
    best = v0 + (p0 - coord if coord < p0 else coord - p0) * rate
    for i in range(1, len(pareto)):
        pi, vi = pareto[i]
        cand = vi + (pi - coord if coord < pi else coord - pi) * rate
        if cand < best:
            best = cand
    return best


def _build_exact_2d_potential_fn(
    projected: Mapping[Tuple[int, int, int], int],
    rate: int,
    fallback_index: _ExactGoalPotentialIndex,
):
    """Build an algebraically exact O(1)-quadrant / 1D-Pareto L1 goal potential."""
    cpp = load_native("_gridless_cpp")
    if cpp is not None and projected:
        try:
            return cpp.Native2DPotential(projected, int(rate))
        except (OverflowError, TypeError):
            pass
    pts = [(gx, gy, cost) for (gx, gy, _), cost in projected.items()]
    gx0 = gx1 = pts[0][0]
    gy0 = gy1 = pts[0][1]
    sw_k = se_k = nw_k = ne_k = None
    south_pts: List[Tuple[int, int]] = []
    north_pts: List[Tuple[int, int]] = []
    west_pts: List[Tuple[int, int]] = []
    east_pts: List[Tuple[int, int]] = []
    for gx, gy, c in pts:
        if gx < gx0:
            gx0 = gx
        elif gx > gx1:
            gx1 = gx
        if gy < gy0:
            gy0 = gy
        elif gy > gy1:
            gy1 = gy
        gx_r, gy_r = gx * rate, gy * rate
        c_sw = c + gx_r + gy_r
        c_se = c - gx_r + gy_r
        c_nw = c + gx_r - gy_r
        c_ne = c - gx_r - gy_r
        if sw_k is None or c_sw < sw_k:
            sw_k = c_sw
        if se_k is None or c_se < se_k:
            se_k = c_se
        if nw_k is None or c_nw < nw_k:
            nw_k = c_nw
        if ne_k is None or c_ne < ne_k:
            ne_k = c_ne
        south_pts.append((gx, c + gy_r))
        north_pts.append((gx, c - gy_r))
        west_pts.append((gy, c + gx_r))
        east_pts.append((gy, c - gx_r))
    south_p = _reduce_1d_pareto(south_pts, rate)
    north_p = _reduce_1d_pareto(north_pts, rate)
    west_p = _reduce_1d_pareto(west_pts, rate)
    east_p = _reduce_1d_pareto(east_pts, rate)

    def eval_potential(x: int, y: int) -> int:
        if x <= gx0:
            if y <= gy0:
                return sw_k - (x + y) * rate
            if y >= gy1:
                return nw_k + (-x + y) * rate
            return _eval_1d_pareto(west_p, y, rate) - x * rate
        if x >= gx1:
            if y <= gy0:
                return se_k + (x - y) * rate
            if y >= gy1:
                return ne_k + (x + y) * rate
            return _eval_1d_pareto(east_p, y, rate) + x * rate
        if y <= gy0:
            return _eval_1d_pareto(south_p, x, rate) - y * rate
        if y >= gy1:
            return _eval_1d_pareto(north_p, x, rate) + y * rate
        return fallback_index.value(x, y, 0)

    return eval_potential


def _insert_linf_violation_box(
    excluded,
    db_mod,
    rect: Tuple[int, int, int, int],
    half: int,
    sp: int,
) -> None:
    """Exact centerline unit squares violating ``_rects_clear(..., sp)``."""
    rx0, ry0, rx1, ry1 = _norm(rect)
    box_cls = db_mod.Box
    if sp <= 0:
        excluded.insert(box_cls(rx0 - half, ry0 - half, rx1 + half + 1, ry1 + half + 1))
    else:
        excluded.insert(
            box_cls(
                rx0 - half - sp + 1,
                ry0 - half - sp + 1,
                rx1 + half + sp,
                ry1 + half + sp,
            )
        )


def _insert_exact_euclid_violation_boxes(
    excluded,
    db_mod,
    rect: Tuple[int, int, int, int],
    half: int,
    sp: int,
) -> None:
    """Exact centerline unit squares violating ``sx*sx + sy*sy < sp*sp``."""
    rx0, ry0, rx1, ry1 = _norm(rect)
    if sp <= 0:
        excluded.insert(
            db_mod.Box(rx0 - half, ry0 - half, rx1 + half + 1, ry1 + half + 1)
        )
        return
    bx0 = rx0 - half
    by0 = ry0 - half
    bx1 = rx1 + half + 1
    by1 = ry1 + half + 1
    tpl = _euclid_poly_template(int(sp))
    if not tpl:
        dx, dy = _pareto_euclid_pairs(int(sp))[0]
        excluded.insert(db_mod.Box(bx0 - dx, by0 - dy, bx1 + dx, by1 + dy))
        return
    pt_cls = db_mod.Point
    excluded.insert(
        db_mod.SimplePolygon(
            [
                pt_cls((bx1 if ux else bx0) + sdx, (by1 if uy else by0) + sdy)
                for ux, sdx, uy, sdy in tpl
            ]
        )
    )


class _GridlessPlanarSearchMixin:
    """Endpoint domains, candidate lines, region A* search, and access transactions."""

    def _pin_access(self, net: str, px: int, py: int, gi: int,
                    ) -> List[Tuple[Tuple[int, int, int], Optional[_ViaSite]]]:
        """The routing START/GOAL states for a pin: the pin node on its seed metal
        (route on M1 / via up at will), PLUS -- when a wireless escape via exists --
        a landing node on gi+1 whose fixed via (recorded) connects down to the pin.
        A dense-pack pin that can emit NO M1 wire still reaches the graph through the
        offset escape landing.  Independently, exact drawn same-net metal may carry
        access to canonical via anchors anywhere on the pin's closed-touch component;
        an individually legal local landing must not hide that complete domain.
        A smaller functional landing may coexist with a default array at the same
        authorized anchor when it reduces negotiated congestion; the common route
        objective carries the reliability preference. Component alternatives contain
        ONLY their upper state + landing via; no fictitious remote seed-metal state is
        exposed. Every alternative therefore remains physically connected through the
        pin."""
        required = getattr(self, "_group_route_required_states", {}).get(
            (net, (px, py, gi))
        )
        if (net, (px, py, gi)) in getattr(self, "_required_wire_endpoints", ()):
            if required is not None and tuple(required) != (px, py, gi):
                raise ValueError("A required wire endpoint cannot be redirected to a group portal")
            # Unlike a real device conductor, a virtual endpoint has no native
            # metal to bridge an offset landing back to its declared access.
            # The ordinary graph must emit the final wire to this exact state.
            return [((px, py, gi), None)]
        if required is not None:
            return [(tuple(required), None)]
        states: List[Tuple[Tuple[int, int, int], Optional[_ViaSite]]] = [
            ((px, py, gi), None)
        ]
        via_states = []
        if gi < self._max_gi:
            component_states = self._owned_component_escape_states(
                net,
                px,
                py,
                gi,
            )
            landing = self._escape_landing(net, px, py, gi)
            if landing is not None:
                ex, ey, en = landing
                if en == 0 or not any(
                    via is not None and via.n == 0
                    for _state, via in component_states
                ):
                    via_states.append((
                        (ex, ey, gi + 1),
                        _ViaSite(ex, ey, gi, en),
                    ))
            via_states.extend(component_states)
        states.extend(via_states)
        unique = {self._access_state_key(item): item for item in states}
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _shared_materialized_access_state(
        starts: Sequence[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ],
        goals: Sequence[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ],
    ) -> Optional[Tuple[int, int, int]]:
        """Return one exact state already materialized for both endpoints."""

        start_states = {
            tuple(state) for state, via in starts if via is None
        }
        goal_states = {
            tuple(state) for state, via in goals if via is None
        }
        shared = tuple(sorted(start_states & goal_states))
        return None if not shared else shared[0]

    def _endpoint_access(
        self,
        *,
        endpoint: str,
        net: str,
        node: Tuple[int, int, int],
        allowed_layers: Sequence[int],
    ) -> List[Tuple[Tuple[int, int, int], Optional[_ViaSite]]]:
        """Fresh route-edge access plus endpoint-scoped STOP evidence.

        The attempt-cut ledger changes after every committed MST edge, so signal
        access cannot be cached per net attempt.  Generate it only after the edge's
        failure context exists, then move the component diagnostic under a stable
        source/goal key.  This prevents the second endpoint (or the next context
        reset) from overwriting which exact pin/component supplied the evidence.
        """
        prefix = "component_escape_"
        details = dict(getattr(self, "_route_failure_details", ()))
        details.pop(f"{endpoint}_component_escape", None)
        for key in tuple(details):
            if key.startswith(prefix):
                del details[key]
        self._route_failure_details = tuple(sorted(details.items()))

        x, y, gi = node
        states = [
            state
            for state in self._pin_access(net, x, y, gi)
            if state[0][2] in allowed_layers
        ]
        details = dict(getattr(self, "_route_failure_details", ()))
        component = tuple(sorted(
            (key[len(prefix):], details.pop(key))
            for key in tuple(details)
            if key.startswith(prefix)
        ))
        if component:
            details[f"{endpoint}_component_escape"] = component
        self._route_failure_details = tuple(sorted(details.items()))
        return states

    def _planar_endpoint_domain(
        self, net: str, node: Tuple[int, int, int], *,
        materialized_state: Optional[Tuple[int, int, int]] = None,
        diagnostic_work=None,
    ) -> _PlanarEndpointDomain:
        """Derive cap regions from the literal seed component and throat graph.

        The optional callback debits explicitly named Python work events.  It
        is not a total time, memory, input-validation, or geometry budget.
        New diagnostic indexes remain local and are never put in the route cache.
        """
        if diagnostic_work is not None:
            if not callable(diagnostic_work):
                raise TypeError("diagnostic work must be callable")
            def work(event):
                self._endpoint_diagnostic_step(net, diagnostic_work, event)
            work("domain.enter")
        else:
            work = None
        node = tuple(node)
        x, y, gi = node
        grid = int(self._manufacturing_grid_nm)
        key = (net, node)
        required = getattr(self, "_group_route_required_states", {}).get(key)
        if key in getattr(self, "_required_wire_endpoints", ()):
            if required is not None and tuple(required) != node:
                raise ValueError("A required wire endpoint cannot be redirected to a group portal")
            required = node
        if materialized_state is not None:
            materialized_state = tuple(materialized_state)
            if required is not None and tuple(required) != materialized_state:
                raise ValueError("A materialized endpoint cannot change its required group state")
            required = materialized_state
        if required is not None:
            return _PlanarEndpointDomain(node, grid, tuple(required), ())
        return self._unrequired_planar_endpoint_domain(
            net, node, grid, work=work, diagnostic_work=diagnostic_work,
        )

    def _owned_planar_access_candidates(
        self, net: str, node: Tuple[int, int, int],
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Finite historical trials within the exact owned endpoint domain."""
        domain = self._planar_endpoint_domain(net, node)
        if domain.singleton is not None:
            return ()

        def grid_points(low, high):
            midpoint = ((low + high) // (2 * domain.grid)) * domain.grid
            return tuple(sorted({low, midpoint, high}))

        candidates = set()
        for _atom, (x0, y0, x1, y1) in domain.regions:
            self._watchdog_check(net, "planar_component_access")
            for px in grid_points(x0, x1):
                for py in grid_points(y0, y1):
                    state = (px, py, domain.node[2])
                    if state != domain.node:
                        candidates.add(state)
        return tuple(sorted(candidates))

    def _via_layer_pad_rect(
        self, via: _ViaSite, gi: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return the exact via landing-pad rectangle on layer ``gi``, if any."""
        below_half, above_half = self._via_pad_half_for(via)
        if int(gi) == int(via.lo):
            half = int(below_half)
        elif int(gi) == int(via.lo) + 1:
            half = int(above_half)
        else:
            return None
        return (
            int(via.x) - half,
            int(via.y) - half,
            int(via.x) + half,
            int(via.y) + half,
        )

    def _owned_planar_access_joins_wire(
        self, net: str, node: Tuple[int, int, int],
        selected: Tuple[int, int, int], segs: Sequence[_WireSeg],
    ) -> bool:
        """Recheck exact membership and an actual minimum-width wire contact."""
        atoms = self._planar_endpoint_domain(net, node).owning_atoms(selected)
        if not atoms:
            return False
        width = int(self._tech.gi_min_width(node[2]))
        for segment in segs:
            if segment.gi != selected[2]:
                continue
            rect = _seg_bbox(segment.ax, segment.ay, segment.bx, segment.by,
                             self._segment_half_width(net, segment))
            if (self._point_in_closed_rect(selected[0], selected[1], rect)
                    and any(self._point_in_closed_rect(selected[0], selected[1], atom.rect)
                            and self._owned_landing_has_minimum_throat(atom.rect, rect, width)
                            for atom in atoms)):
                return True
        return False

    def _owned_planar_access_joins_conductor(
        self, net: str, node: Tuple[int, int, int],
        selected: Tuple[int, int, int], segs: Sequence[_WireSeg],
        vias: Sequence[_ViaSite] = (),
    ) -> bool:
        """Accept either a planar wire segment or a vertical via landing pad."""
        if self._owned_planar_access_joins_wire(net, node, selected, segs):
            return True
        if not vias:
            return False
        atoms = self._planar_endpoint_domain(net, node).owning_atoms(selected)
        if not atoms:
            return False
        width = int(self._tech.gi_min_width(node[2]))
        sx, sy, sgi = int(selected[0]), int(selected[1]), int(selected[2])
        for via in vias:
            rect = self._via_layer_pad_rect(via, sgi)
            if rect is None:
                continue
            if (self._point_in_closed_rect(sx, sy, rect)
                    and any(self._point_in_closed_rect(sx, sy, atom.rect)
                            and self._owned_landing_has_minimum_throat(atom.rect, rect, width)
                            for atom in atoms)):
                return True
        return False

    # ------------------------------------------------------------- candidate grid
    def _on_manufacturing_grid_lines(
        self,
        values: Sequence[int],
        lower: int,
        upper: int,
        *,
        preserve_frontier_sides: bool = False,
    ) -> List[int]:
        """Project candidate axes to the PDK grid without losing a frontier side."""

        grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
        if grid == 1:
            return sorted(set(int(value) for value in values))
        projected: Set[int] = set()
        for raw in values:
            value = int(raw)
            floor = (value // grid) * grid
            ceil = floor if value == floor else floor + grid
            candidates = (
                (floor, ceil)
                if preserve_frontier_sides
                else (floor if (value - floor) * 2 < grid else ceil,)
            )
            for candidate in candidates:
                if int(lower) <= candidate <= int(upper):
                    projected.add(candidate)
        return sorted(projected)

    def _candidate_lines(self, anchors: Sequence[Tuple[int, int]], net: str,
                         window: Rect) -> Tuple[List[int], List[int]]:
        """The candidate routing lines inside a local ``window`` (the net's pin bbox
        grown by a routing margin).

        Lines sit at: every terminal coordinate (exact, gridless pin access); every
        FIXED-OBSTACLE (device artwork) edge offset by its hard clearance (so a wire
        can hug min-space to immovable metal); and a UNIFORM PITCH FILL grid across
        the window (aligned to a global origin so tracks are consistent net-to-net).

        Foreign COMMITTED-METAL edges are DELIBERATELY NOT all turned into initial
        candidate lines here: exact contact is hard at transition time and records
        causal blocker geometry, while positive proximity is queried directly as
        ``_box_congestion``. The initial grid must NOT blow up with the complete
        committed-metal count -- that explosion is what made the pure-Python A*
        wander the whole cell. The ordinary fill grid stays metal-count-independent;
        the hard-rejection evidence separately retains each causal committed blocker
        for bounded victim replay. Because a fill track is >= one pitch
        (>= min_space) from any foreign centreline, routing on it is min-space clean
        by construction."""
        wx0, wy0, wx1, wy1 = _norm(window)
        local_iface = (
            getattr(self, "_active_group_route_local_interface", None)
            if bool(getattr(self, "_active_group_route_local_closure", False))
            else None
        )
        if local_iface is not None:
            ex0, ey0, ex1, ey1 = (int(v) for v in local_iface[0])
            if ex0 <= ex1 and ey0 <= ey1 and all(
                ex0 <= int(ax) <= ex1 and ey0 <= int(ay) <= ey1
                for (ax, ay) in anchors
            ):
                if max(wx0, ex0) <= min(wx1, ex1):
                    wx0, wx1 = max(wx0, ex0), min(wx1, ex1)
                if max(wy0, ey0) <= min(wy1, ey1):
                    wy0, wy1 = max(wy0, ey0), min(wy1, ey1)
                window = (wx0, wy0, wx1, wy1)
        xs: Set[int] = set()
        ys: Set[int] = set()
        for (ax, ay) in anchors:
            xs.add(ax)
            ys.add(ay)
        rlayers = range(1, self._n_layers) if self._n_layers > 1 else range(self._n_layers)
        max_space = max(self._space[i] for i in rlayers)
        max_pad = max((max(self._via_pad_half[lo]) for lo in self._via_pad_half),
                      default=max(self._half_w[i] for i in rlayers))

        def _emit_edges(x0, y0, x1, y1, off_a, off_b):
            for clear in (off_a, off_b):
                for xv in (x0 - clear, x1 + clear):
                    if wx0 <= xv <= wx1:
                        xs.add(xv)
                for yv in (y0 - clear, y1 + clear):
                    if wy0 <= yv <= wy1:
                        ys.add(yv)

        # FIXED device obstacles -> exact hard-clearance offset lines (HARD keep-out;
        # the router must be able to sit exactly min-space from immovable artwork).
        # OWN-net rects also emit lines: under TOUCH-OR-CLEAR (R7a) the legal
        # positions near own artwork are flush-touch (wire/pad edge exactly on the
        # rect edge), inside-overlap, or >= min-space clear -- the old blanket
        # owner==net exemption needed no lines because EVERYTHING near own metal
        # was legal; with the band rule and no touch lines the A* had no legal
        # coordinates near its own tap rows / access tabs (the measured gnd/vout
        # budget-exhaustion failure: ~200k wander rejections, 0 legal passage).
        if local_iface is not None:
            ex0, ey0, ex1, ey1 = (int(v) for v in local_iface[0])
            px, py = int(local_iface[1][0]), int(local_iface[1][1])
            for xv in (ex0, ex1, px):
                if wx0 <= xv <= wx1:
                    xs.add(xv)
            for yv in (ey0, ey1, py):
                if wy0 <= yv <= wy1:
                    ys.add(yv)
            grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 1) or 1))
            min_hw = max(1, min((int(h) for h in self._half_w), default=1))
            use_single = bool(
                getattr(self, "_single_via_escape_allowed", lambda: False)()
            )
            via_pad1 = getattr(self, "_via_pad_half1", None) or self._via_pad_half
            local_offs: Set[int] = set()
            for g_idx in range(self._n_layers):
                hw_g = int(self._half_w[g_idx])
                if g_idx > 0 and hw_g >= 4 * min_hw and g_idx != int(local_iface[1][2]):
                    continue
                local_offs.add(hw_g)
                local_offs.add(int(self._hw(net, g_idx)))
                if g_idx in self._via_pad_half:
                    bh_g, ah_g = self._via_pad_half[g_idx]
                    cut_ext = int(self._via_cut_extent(g_idx, None))
                    local_offs.add(int(bh_g))
                    local_offs.add(max(int(bh_g), cut_ext))
                    if use_single and g_idx in via_pad1:
                        bh1, ah1 = via_pad1[g_idx]
                        cut_ext1 = int(self._via_cut_extent(g_idx, 1))
                        local_offs.add(max(int(bh1), cut_ext1))
                    if (
                        g_idx + 1 < self._n_layers
                        and int(self._half_w[g_idx + 1]) < 4 * min_hw
                    ):
                        local_offs.add(int(ah_g))
                        local_offs.add(max(int(ah_g), int(bh_g), cut_ext))
                        if use_single and g_idx in via_pad1:
                            bh1, ah1 = via_pad1[g_idx]
                            cut_ext1 = int(self._via_cut_extent(g_idx, 1))
                            local_offs.add(max(int(ah1), int(bh1), cut_ext1))
            for off in local_offs:
                xv_lo = -(-(ex0 + off) // grid) * grid
                xv_hi = ((ex1 - off) // grid) * grid
                for xv in (xv_lo, xv_hi):
                    if wx0 <= xv <= wx1 and ex0 <= xv <= ex1:
                        xs.add(xv)
                yv_lo = -(-(ey0 + off) // grid) * grid
                yv_hi = ((ey1 - off) // grid) * grid
                for yv in (yv_lo, yv_hi):
                    if wy0 <= yv <= wy1 and ey0 <= yv <= ey1:
                        ys.add(yv)

        for gi in range(self._n_layers):
            hw = self._half_w[gi]
            sp = self._space[gi]
            oidx = self._obs_idx.get(gi)
            if oidx is not None:
                for (owner, (x0, y0, x1, y1)) in oidx.query(window):
                    if owner == net:
                        _emit_edges(x0, y0, x1, y1, hw, hw + sp)       # abut-touch / clear (wire)
                        _emit_edges(x0, y0, x1, y1, -hw, 0)            # inside-flush / edge-ride
                        _emit_edges(x0, y0, x1, y1, max_pad, max_pad + sp)  # abut-touch / clear (pad)
                        continue
                    _emit_edges(x0, y0, x1, y1, hw + sp, max_pad + sp)
            if self._mim_derived_unrelated_static_active() and (
                bool(getattr(self, "_negotiation_port_legalize", False))
                or bool(getattr(self, "_search_forbidden_via_sites", ()))
                or bool(getattr(self, "_search_forbidden_wire_edges", ()))
            ):
                mim_pad_halves = {hw}
                if gi in self._via_pad_half:
                    mim_pad_halves.add(int(self._via_pad_half[gi][0]))
                if (gi - 1) in self._via_pad_half:
                    mim_pad_halves.add(int(self._via_pad_half[gi - 1][1]))
                for (
                    m_owner,
                    (mx0, my0, mx1, my1),
                    m_clear,
                    _m_metric,
                    _m_rule,
                ) in getattr(
                    self, "_mim_derived_unrelated_clearance_by_gi", {}
                ).get(gi, ()):
                    if m_owner == net:
                        continue
                    req = int(m_clear)
                    max_off = max(mim_pad_halves) + req
                    if (
                        mx1 + max_off < wx0
                        or mx0 - max_off > wx1
                        or my1 + max_off < wy0
                        or my0 - max_off > wy1
                    ):
                        continue
                    for ph in sorted(mim_pad_halves):
                        _emit_edges(mx0, my0, mx1, my1, ph + req, ph + req)
            # OWN in-attempt metal -> ONE track-alignment line per rect axis:
            # the CENTER coordinate (a wire box's center IS its track). Without
            # this coordinate a later edge of the same net cannot ride the
            # earlier track exactly (per-edge windows emit different
            # obstacle-offset lines), so the own-band price had no zero-cost
            # alternative to steer to -- the 10-residual plateau of jittered
            # parallel same-net tracks. Centers ONLY: emitting offset families
            # per rect exploded the grid (1618x1587 lines -> the 8M-expansion
            # budget cap -> gnd FAIL).
            aidx = self._attempt_idx.get(gi)
            if aidx is not None:
                for (_own, (x0, y0, x1, y1)) in aidx.query(window):
                    cx = (x0 + x1) // 2
                    cy = (y0 + y1) // 2
                    if wx0 <= cx <= wx1:
                        xs.add(cx)
                    if wy0 <= cy <= wy1:
                        ys.add(cy)

        # UNIFORM PITCH FILL (bounded, committed-metal-independent), aligned to 0 so
        # tracks are consistent across MST edges and across nets.
        pitch = self._res_cell
        gx = (wx0 // pitch) * pitch
        while gx <= wx1:
            if gx >= wx0:
                xs.add(gx)
            gx += pitch
        gy = (wy0 // pitch) * pitch
        while gy <= wy1:
            if gy >= wy0:
                ys.add(gy)
            gy += pitch

        xs.add(wx0); xs.add(wx1); ys.add(wy0); ys.add(wy1)
        env_xs, env_ys = self._group_envelope_clearance_lines(wx0, wy0, wx1, wy1)
        xs.update(env_xs)
        ys.update(env_ys)
        return (
            self._on_manufacturing_grid_lines(xs, wx0, wx1),
            self._on_manufacturing_grid_lines(ys, wy0, wy1),
        )

    # ------------------------------------------------------------- 2-pin A*
    def _edge_cost(
        self,
        net: str,
        gi: int,
        dx: int,
        dy: int,
        trunk: int = 0,
    ) -> int:
        length = abs(dx) + abs(dy)
        cost = length * _STEP_COST_PER_NM
        declared_axis_record = self._declared_trunk_axis_by_net.get(net)
        declared_axis = (
            declared_axis_record[1]
            if (
                declared_axis_record is not None
                and gi == trunk == declared_axis_record[0]
            )
            else None
        )
        direction_preference = getattr(
            self, "_direction_preference_by_gi", {}
        ).get(gi)
        pref_h = (
            declared_axis == "H"
            if declared_axis is not None
            else (
                direction_preference[0] == "H"
                if direction_preference is not None
                else self._tech.preferred_horizontal(gi)
            )
        )
        moving_h = dy == 0
        if pref_h != moving_h:
            weight = (
                direction_preference[1]
                if declared_axis is None and direction_preference is not None
                else 1.0
            )
            cost += round(length * _OFF_PREF_COST_PER_NM * weight)
        if gi == 0:
            cost += length * _M1_ROUTE_COST_PER_NM
        # TRUNK bias: routing BELOW this net's trunk metal costs progressively more,
        # so a cell-spanning net climbs to its trunk layer and runs OVER the cell.
        # When a net explicitly declares TrunkLayer, deviating above its declared
        # carrier also pays the per-level trunk-deviation rate so wide trunks stay on
        # their declared carrier unless blocked.
        trunk_delta = self._effective_trunk_delta(net, gi, trunk)
        if trunk_delta:
            cost += length * _TRUNK_BELOW_COST_PER_NM * trunk_delta
        if not bool(getattr(self, "_active_group_route_local_closure", False)):
            match_surcharge = getattr(self, "_active_match_layer_surcharge", None)
            if match_surcharge:
                cost += length * int(match_surcharge.get(gi, 0))
        elif gi > 0:
            declared_trunks = getattr(self, "_declared_trunk_gi_by_net", {})
            local_iface = getattr(self, "_active_group_route_local_interface", None)
            declared_gi = declared_trunks.get(net)
            if (
                declared_gi is not None
                and isinstance(local_iface, tuple)
                and len(local_iface) >= 2
                and len(local_iface[1]) >= 3
                and int(local_iface[1][2]) < int(declared_gi)
                and gi < int(declared_gi)
            ):
                # When a single-group port must exit on its landing layer (unlayered portal),
                # still incentivize interior runs to climb onto the declared TrunkLayer.
                cost += length * _M1_ROUTE_COST_PER_NM * (int(declared_gi) - gi)
        return cost

    def _effective_trunk_delta(self, net: str, gi: int, trunk: int) -> int:
        """Trunk-layer distance penalty, keeping matched nets on declared trunks even inside groups."""
        declared_trunks = getattr(self, "_declared_trunk_gi_by_net", {})
        matched = bool(getattr(self, "_matched_partners_by_net", {}).get(net))
        in_local = bool(getattr(self, "_active_group_route_local_closure", False))
        declared_active = net in declared_trunks and (not in_local or matched)
        delta = abs(trunk - gi) if declared_active else max(0, trunk - gi)
        if matched and net in declared_trunks and gi < int(declared_trunks[net]):
            delta += 2 * (int(declared_trunks[net]) - gi)
        return delta

    def _edge_objective(
        self,
        net: str,
        gi: int,
        ax: int,
        ay: int,
        bx: int,
        by: int,
        trunk: int = 0,
    ) -> int:
        """PDK-derived planar objective shared by residual A* and patterns."""
        return self._edge_cost(net, gi, bx - ax, by - ay, trunk)

    def _pattern_objective(
        self, net: str, segs: Sequence[_WireSeg],
        vias: Sequence[_ViaSite], trunk: int = 0,
        *, congestion: Optional[int] = None,
    ) -> int:
        """A* equivalent score for one already-legal closed-form pattern.

        Pattern-first is an acceleration, not a second routing policy. It sees
        the same present/history congestion, turn price and via price that A*
        sees; otherwise every rip-up round
        deterministically re-selects the same foreign-metal crossing before
        A* gets a chance to negotiate it away.
        """
        score = (
            self._pattern_congestion(net, segs, vias)
            if congestion is None else int(congestion)
        )
        last_dir = 0
        for seg in segs:
            score += self._edge_objective(
                net, seg.gi, seg.ax, seg.ay, seg.bx, seg.by, trunk)
            direction = 1 if seg.ay == seg.by else 2
            if last_dir and direction != last_dir:
                score += self._turn_cost()
            last_dir = direction
        for via in vias:
            score += _VIA_COST + self._functional_single_via_penalty(via)
        return score

    def _functional_single_via_penalty(self, via: _ViaSite) -> int:
        """Soft reliability price for one access that drops declared redundancy.

        Reuse one adjacent-layer escalation price: clean or lightly congested
        routing therefore retains the declared array, while negotiated present /
        history pressure can eventually buy the already-supported functional single.
        The price is uniform across nets and PDKs and introduces no new absolute
        geometry threshold.
        """
        if self.via_redundancy > 1 and self._site_n(via) == 1:
            return _VIA_COST
        return 0

    def _terminal_via_objective(self, net: str, via: _ViaSite) -> int:
        """A* terminal-via price shared by starts, goals and exact potentials."""
        return (
            _VIA_COST
            + self._functional_single_via_penalty(via)
            + self._via_congestion(
                via.x, via.y, via.lo, net, self._site_n(via)
            )
        )

    def _preferred_via_site(
        self,
        net: str,
        x: int,
        y: int,
        lo: int,
    ) -> Optional[_ViaSite]:
        """Select the declared array when legal, or single escape when authorized."""
        default = _ViaSite(x, y, lo)
        if self._via_legal(x, y, lo, net):
            if not self._single_via_escape_allowed() or not hasattr(self, "_pres_fac"):
                return default
            default_congestion = self._via_congestion(x, y, lo, net)
            single = _ViaSite(x, y, lo, 1)
            single_penalty = self._functional_single_via_penalty(single)
            if (
                default_congestion > single_penalty
                and self._via_legal(x, y, lo, net, n=1)
                and (
                    single_penalty
                    + self._via_congestion(x, y, lo, net, n=1)
                    < default_congestion
                )
            ):
                return single
            return default
        if self._single_via_escape_allowed() and self._via_legal(x, y, lo, net, n=1):
            return _ViaSite(x, y, lo, 1)
        return None

    def _pattern_via_pad_conflicts(
        self,
        vias: Sequence[_ViaSite],
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Raw sub-rule landing-pad gaps inside one pattern candidate."""
        _sep = _eg_attr("_sep", _base_sep)
        pads = []
        for via in vias:
            below, above = self._via_pad_half_for(via)
            pads.append((
                (via.lo, (via.x - below, via.y - below,
                          via.x + below, via.y + below)),
                (via.lo + 1, (via.x - above, via.y - above,
                              via.x + above, via.y + above)),
            ))
        conflicts = set()
        for left in range(len(vias)):
            for right in range(left + 1, len(vias)):
                for left_gi, left_pad in pads[left]:
                    for right_gi, right_pad in pads[right]:
                        if left_gi != right_gi:
                            continue
                        sx, sy = _sep(_norm(left_pad), _norm(right_pad))
                        if not (sx or sy):
                            continue
                        space = self._space[left_gi]
                        if sx * sx + sy * sy < space * space:
                            conflicts.add((left, right, left_gi))
        return tuple(sorted(conflicts))

    def _raw_gap_metallised(
        self,
        gi: int,
        left: Rect,
        right: Rect,
        sx: int,
        sy: int,
        candidate_metal: Mapping[int, Sequence[Rect]],
    ) -> bool:
        """Whether exact drawn metal removes every facing edge in a raw band."""
        gap = self._gap_box(left, right, sx, sy)
        gx0, gy0, gx1, gy1 = gap
        if gx1 <= gx0 or gy1 <= gy0:
            return False
        cover = []
        for source in (
            candidate_metal.get(gi, ()),
            self._attempt_metal.get(gi, ()),
        ):
            for rect in source:
                rr = _norm(rect)
                if (rr[0] < gx1 and gx0 < rr[2]
                        and rr[1] < gy1 and gy0 < rr[3]):
                    cover.append(rr)
        abstract = self._abstract_rects.get(gi, ())
        obstacle_index = self._obs_idx.get(gi)
        if obstacle_index is not None:
            for _owner, rect in obstacle_index.query(gap):
                if rect in abstract:
                    continue
                rr = _norm(rect)
                if (rr[0] < gx1 and gx0 < rr[2]
                        and rr[1] < gy1 and gy0 < rr[3]):
                    cover.append(rr)
        committed_index = self._metal_idx.get(gi)
        if committed_index is not None:
            for _owner, rect in committed_index.query(gap):
                rr = _norm(rect)
                if (rr[0] < gx1 and gx0 < rr[2]
                        and rr[1] < gy1 and gy0 < rr[3]):
                    cover.append(rr)
        for rr in cover:
            if (rr[0] <= gx0 and rr[1] <= gy0
                    and rr[2] >= gx1 and rr[3] >= gy1):
                return True
        return not subtract_rects(gap, cover)

    def _pattern_via_pad_visible_conflicts(
        self,
        net: str,
        vias: Sequence[_ViaSite],
        segs: Sequence[_WireSeg] = (),
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Landing-pad gaps that remain visible after exact metal union."""
        _sep = _eg_attr("_sep", _base_sep)
        pads: Dict[Tuple[int, int], Rect] = {}
        candidate_metal: Dict[int, List[Rect]] = {}
        for via_index, via in enumerate(vias):
            below, above = self._via_pad_half_for(via)
            for gi, pad in (
                (via.lo, (via.x - below, via.y - below,
                          via.x + below, via.y + below)),
                (via.lo + 1, (via.x - above, via.y - above,
                              via.x + above, via.y + above)),
            ):
                normalized = _norm(pad)
                pads[(via_index, gi)] = normalized
                candidate_metal.setdefault(gi, []).append(normalized)
        for seg in segs:
            candidate_metal.setdefault(seg.gi, []).append(_norm(_seg_bbox(
                seg.ax,
                seg.ay,
                seg.bx,
                seg.by,
                self._segment_half_width(net, seg),
            )))

        visible = []
        for left_index, right_index, gi in self._pattern_via_pad_conflicts(vias):
            left = pads[(left_index, gi)]
            right = pads[(right_index, gi)]
            sx, sy = _sep(left, right)
            if not self._raw_gap_metallised(
                gi,
                left,
                right,
                sx,
                sy,
                candidate_metal,
            ):
                visible.append((left_index, right_index, gi))
        return tuple(visible)

    def _pattern_via_wire_conflicts(
        self,
        net: str,
        vias: Sequence[_ViaSite],
        segs: Sequence[_WireSeg],
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Deck-visible open landing-pad gaps to wires in one route group."""
        _sep = _eg_attr("_sep", _base_sep)
        metal: Dict[int, List[Rect]] = {}
        pads = []
        for via_index, via in enumerate(vias):
            below, above = self._via_pad_half_for(via)
            for gi, pad in (
                (via.lo, (via.x - below, via.y - below,
                          via.x + below, via.y + below)),
                (via.lo + 1, (via.x - above, via.y - above,
                              via.x + above, via.y + above)),
            ):
                normalized = _norm(pad)
                pads.append((via_index, gi, normalized))
                metal.setdefault(gi, []).append(normalized)
        wires = []
        for seg_index, seg in enumerate(segs):
            box = _norm(_seg_bbox(
                seg.ax,
                seg.ay,
                seg.bx,
                seg.by,
                self._segment_half_width(net, seg),
            ))
            wires.append((seg_index, seg.gi, box))
            metal.setdefault(seg.gi, []).append(box)

        conflicts = set()
        pair_work = 0
        for via_index, pad_gi, pad in pads:
            for seg_index, wire_gi, wire in wires:
                if pair_work % 256 == 0:
                    self._watchdog_check(net, 'candidate_via_wire_conflicts')
                pair_work += 1
                if pad_gi != wire_gi:
                    continue
                sx, sy = _sep(pad, wire)
                if not (sx or sy):
                    continue
                space = self._space[pad_gi]
                if sx * sx + sy * sy >= space * space:
                    continue
                if self._raw_gap_metallised(
                    pad_gi,
                    pad,
                    wire,
                    sx,
                    sy,
                    metal,
                ):
                    continue
                conflicts.add((via_index, seg_index, pad_gi))
        return tuple(sorted(conflicts))

    def _candidate_via_conflict_details(
        self,
        net: str,
        vias: Sequence[_ViaSite],
        segs: Sequence[_WireSeg],
        pad_conflicts: Sequence[Tuple[int, int, int]],
        wire_conflicts: Sequence[Tuple[int, int, int]],
    ) -> Tuple[Tuple[str, Any], ...]:
        """Bound the first exact same-net via conflict for typed replay."""

        def via_identity(via: _ViaSite) -> Tuple[int, int, int, int]:
            return (int(via.x), int(via.y), int(via.lo), int(via.n))

        details = []
        if pad_conflicts:
            left_index, right_index, gi = min(pad_conflicts)
            details.append(("candidate_via_pad_conflict", tuple(sorted((
                ("gi", int(gi)),
                ("left_index", int(left_index)),
                ("left_via", via_identity(vias[left_index])),
                ("right_index", int(right_index)),
                ("right_via", via_identity(vias[right_index])),
            )))))
        if wire_conflicts:
            via_index, segment_index, gi = min(wire_conflicts)
            seg = segs[segment_index]
            first, second = sorted((
                (int(seg.ax), int(seg.ay)),
                (int(seg.bx), int(seg.by)),
            ))
            details.append(("candidate_via_wire_conflict", tuple(sorted((
                ("gi", int(gi)),
                ("segment_index", int(segment_index)),
                ("via", via_identity(vias[via_index])),
                ("via_index", int(via_index)),
                ("wire", (
                    int(seg.gi),
                    first[0],
                    first[1],
                    second[0],
                    second[1],
                    int(self._segment_half_width(net, seg)),
                )),
            )))))
        attempt_conflicts = []
        for via_index, via in enumerate(vias):
            for gi, pad, blocker in self._via_pad_attempt_conflicts(net, via):
                attempt_conflicts.append((
                    int(via_index),
                    int(gi),
                    tuple(_norm(pad)),
                    tuple(_norm(blocker)),
                ))
        if attempt_conflicts:
            via_index, gi, pad, blocker = min(attempt_conflicts)
            details.append(("candidate_via_attempt_conflict", tuple(sorted((
                ("blocker", blocker),
                ("gi", gi),
                ("pad", pad),
                ("via", via_identity(vias[via_index])),
                ("via_index", via_index),
            )))))
        return tuple(sorted(details, key=lambda item: item[0]))

    def _materialize_candidate_via_wire_merges(
        self,
        net: str,
        vias: Sequence[_ViaSite],
        segs: Sequence[_WireSeg],
        *,
        prior_via_count: int,
        prior_seg_count: int,
        immutable_via_indices: Set[int],
    ) -> Optional[Tuple[_WireSeg, ...]]:
        """Build an atomic same-net bridge for cumulative via/wire gaps.

        One MST edge is searched against the previous edges' attempt ledger,
        but the complete via group is admitted only after that search.  The new
        landing pad can therefore face prior same-net wire across the deck's
        open spacing band even though both pieces were individually legal.
        Final ``_legalize_same_net`` runs too late to repair such a rejected
        group.  Reuse its exact merge-or-clear primitives here, against a local
        candidate-metal index, and return only validated repair segments.  No
        routed-net or attempt ledger is mutated until the caller replays the
        unchanged via resolver and commits the whole edge transaction.
        """

        _sep = _eg_attr("_sep", _base_sep)
        if not (
            0 <= prior_via_count <= len(vias)
            and 0 <= prior_seg_count <= len(segs)
            and all(0 <= index < len(vias) for index in immutable_via_indices)
        ):
            return None
        candidate_segs = list(segs)
        repair_segs: List[_WireSeg] = []
        own_idx: Dict[int, _SpatialIndex] = {}

        def add_own(gi: int, rect: Rect) -> None:
            own_idx.setdefault(gi, _SpatialIndex()).add(net, _norm(rect))

        for seg in candidate_segs:
            add_own(
                seg.gi,
                _seg_bbox(
                    seg.ax,
                    seg.ay,
                    seg.bx,
                    seg.by,
                    self._segment_half_width(net, seg),
                ),
            )
        for via in vias:
            below_half, above_half = self._via_pad_half_for(via)
            add_own(via.lo, (
                via.x - below_half,
                via.y - below_half,
                via.x + below_half,
                via.y + below_half,
            ))
            add_own(via.lo + 1, (
                via.x - above_half,
                via.y - above_half,
                via.x + above_half,
                via.y + above_half,
            ))

        conflicts = self._pattern_via_wire_conflicts(
            net,
            vias,
            candidate_segs,
        )
        bridge_limit = 4 * len(conflicts)
        bridge_count = 0
        while True:
            conflicts = self._pattern_via_wire_conflicts(
                net,
                vias,
                candidate_segs,
            )
            if not conflicts:
                return tuple(repair_segs)
            if bridge_count >= bridge_limit:
                return None
            progressed = False
            for via_index, segment_index, gi in conflicts:
                if (
                    via_index not in immutable_via_indices
                    or (
                        via_index < prior_via_count
                        and segment_index < prior_seg_count
                    )
                ):
                    continue
                via = vias[via_index]
                seg = candidate_segs[segment_index]
                below_half, above_half = self._via_pad_half_for(via)
                if gi == via.lo:
                    pad_half = below_half
                elif gi == via.lo + 1:
                    pad_half = above_half
                else:
                    return None
                pad = _norm((
                    via.x - pad_half,
                    via.y - pad_half,
                    via.x + pad_half,
                    via.y + pad_half,
                ))
                wire = _seg_bbox(
                    seg.ax,
                    seg.ay,
                    seg.bx,
                    seg.by,
                    self._segment_half_width(net, seg),
                )
                sx, sy = _sep(pad, wire)
                if not (sx or sy):
                    continue
                fill = self._merge_fill_rect(gi, pad, wire, sx, sy)
                if fill is None:
                    continue
                new_segs = self._fill_segs(net, gi, fill)
                actual_boxes = tuple(
                    _seg_bbox(
                        item.ax,
                        item.ay,
                        item.bx,
                        item.by,
                        self._segment_half_width(net, item),
                    )
                    for item in new_segs
                )
                if not self._fill_legal(
                    net,
                    gi,
                    actual_boxes,
                    own_idx=own_idx,
                ):
                    continue
                candidate_segs.extend(new_segs)
                repair_segs.extend(new_segs)
                for item, actual_box in zip(new_segs, actual_boxes):
                    add_own(item.gi, actual_box)
                bridge_count += 1
                progressed = True
                break
            if not progressed:
                return None

    def _resolve_candidate_same_net_fills(
        self,
        net: str,
        prior: Optional[_RoutedNet],
        segs: Sequence[_WireSeg],
        vias: Sequence[_ViaSite],
        *,
        mutable_indices: Set[int],
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """Try the existing late legalizer on actual prior and candidate objects.

        The list containers are local; original wire/via objects and prior
        models remain unchanged. Only gaps involving candidate metal (or a
        validated fill descended from it) are eligible. Resolve the complete
        group with prior models immutable; the caller still replays all selected
        endpoint, changed-model cut and deck checks before acceptance.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        if not isinstance(prior, _RoutedNet) or prior.name != net:
            return None
        candidate = _RoutedNet(net, segs=list(segs), vias=list(vias))
        candidate_metal, _candidate_cuts = self._net_metal(candidate)
        active_metal = {
            (int(gi), _norm(rect))
            for gi, rects in candidate_metal.items() for rect in rects
        }
        if not active_metal:
            return None
        trial = _RoutedNet(
            net, segs=list(prior.segs) + list(segs),
            vias=list(prior.vias) + list(vias),
        )
        original_segs, original_vias = tuple(trial.segs), tuple(trial.vias)
        missing = object()
        saved_rejections = getattr(self, "_hard_rejections", missing)
        saved_samples = getattr(self, "_hard_rejection_samples", missing)
        saved_generation = getattr(self, "_hard_rejection_generation", missing)
        self._hard_rejections = (
            {} if saved_rejections is missing else dict(saved_rejections)
        )
        self._hard_rejection_samples = (
            {} if saved_samples is missing else dict(saved_samples)
        )
        try:
            prior_count = len(prior.vias)
            before_metal = self._net_metal(trial)[0]
            orig_wire_conflicts = self._pattern_via_wire_conflicts(
                net, trial.vias, trial.segs,
            )
            orig_attempt_conflicts = tuple(
                (index, gi, pad, blocker)
                for index, via in enumerate(trial.vias)
                for gi, pad, blocker in self._via_pad_attempt_conflicts(
                    net, via, candidate_segs=trial.segs,
                )
            )
            checked_unchanged_union = False
            if orig_wire_conflicts or orig_attempt_conflicts:
                checked_unchanged_union = True
                if (
                    self._complete_owned_union_clears_layers(
                        net, trial.vias, trial.segs, set(before_metal),
                    )
                    and self._ordinary_repair_preserves_world(
                        net, before_metal, before_metal,
                    )
                    and self._repair_preserves_world(
                        net, before_metal, before_metal,
                    )
                ):
                    orig_resolver_args = dict(
                        mutable_indices={prior_count + i for i in mutable_indices},
                        segs=trial.segs,
                    )
                    if orig_wire_conflicts:
                        orig_resolver_args["allow_complete_owned_union"] = True
                    if orig_attempt_conflicts:
                        orig_resolver_args["owned_attempt_prior"] = prior
                    orig_resolved = self._resolve_pattern_via_models(
                        net, trial.vias, **orig_resolver_args,
                    )
                    if (
                        orig_resolved is not None
                        and len(orig_resolved) == len(original_vias)
                        and all(
                            orig_resolved[i] is via
                            for i, via in enumerate(prior.vias)
                        )
                    ):
                        return trial.segs[len(prior.segs):], orig_resolved[prior_count:]

            book = getattr(self, "_book", None)
            baseline = getattr(self, "_active_group_route_union_baseline", None)
            if baseline is None:
                baseline = getattr(self, "_group_route_landing_baselines", {}).get(net)
            auth_base_metal = None
            include_baseline = False
            if (
                book is not None
                and net in book.persistent
                and isinstance(baseline, _RoutedNet)
                and baseline.name == net
                and (baseline.segs or baseline.vias)
            ):
                prior_has_baseline = (
                    len(prior.segs) >= len(baseline.segs)
                    and len(prior.vias) >= len(baseline.vias)
                    and all(a is b for a, b in zip(prior.segs, baseline.segs))
                    and all(a is b for a, b in zip(prior.vias, baseline.vias))
                )
                if not prior_has_baseline:
                    auth_base_metal = self._authenticated_owned_book_metal(net, trial)
                    if auth_base_metal is not None:
                        include_baseline = True

            if include_baseline:
                legalize_trial = _RoutedNet(
                    net,
                    segs=list(baseline.segs) + list(trial.segs),
                    vias=list(baseline.vias) + list(trial.vias),
                )
                legalize_orig_segs = tuple(legalize_trial.segs)
                legalize_orig_vias = tuple(legalize_trial.vias)
                self._legalize_same_net(
                    net, legalize_trial, active_metal=active_metal,
                )
                if (
                    len(legalize_trial.segs) < len(legalize_orig_segs)
                    or any(
                        item is not original
                        for item, original in zip(
                            legalize_trial.segs, legalize_orig_segs
                        )
                    )
                    or len(legalize_trial.vias) != len(legalize_orig_vias)
                    or any(
                        item is not original
                        for item, original in zip(
                            legalize_trial.vias, legalize_orig_vias
                        )
                    )
                ):
                    return None
                trial.segs.extend(legalize_trial.segs[len(legalize_orig_segs):])
            else:
                self._legalize_same_net(net, trial, active_metal=active_metal)
                if (
                    len(trial.segs) < len(original_segs)
                    or any(
                        item is not original
                        for item, original in zip(trial.segs, original_segs)
                    )
                    or len(trial.vias) != len(original_vias)
                    or any(
                        item is not original
                        for item, original in zip(trial.vias, original_vias)
                    )
                ):
                    return None

            after_metal = self._net_metal(trial)[0]
            wire_conflicts = self._pattern_via_wire_conflicts(net, trial.vias, trial.segs)
            attempt_conflicts = tuple(
                (index, gi, pad, blocker)
                for index, via in enumerate(trial.vias)
                for gi, pad, blocker in self._via_pad_attempt_conflicts(
                    net, via, candidate_segs=trial.segs,
                )
            )
            if len(trial.segs) == len(original_segs):
                # A no-op is useful only to prove real raw conflicts false;
                # unchanged clean candidates keep their original fast path.
                if checked_unchanged_union or not (wire_conflicts or attempt_conflicts):
                    return None
                if not self._complete_owned_union_clears_layers(
                    net, trial.vias, trial.segs, set(after_metal),
                ):
                    return None
            world_before = before_metal
            world_after = after_metal
            if include_baseline and auth_base_metal:
                world_before = {
                    gi: (*auth_base_metal.get(gi, ()), *before_metal.get(gi, ()))
                    for gi in (set(auth_base_metal) | set(before_metal))
                }
                world_after = {
                    gi: (*auth_base_metal.get(gi, ()), *after_metal.get(gi, ()))
                    for gi in (set(auth_base_metal) | set(after_metal))
                }
            if (not self._ordinary_repair_preserves_world(net, world_before, world_after)
                    or not self._repair_preserves_world(net, world_before, world_after)):
                return None
            resolver_args = dict(
                mutable_indices={prior_count + i for i in mutable_indices},
                segs=trial.segs,
            )
            if wire_conflicts:
                resolver_args['allow_complete_owned_union'] = True
            if attempt_conflicts:
                resolver_args['owned_attempt_prior'] = prior
            resolved = self._resolve_pattern_via_models(net, trial.vias, **resolver_args)
            if (
                resolved is None or len(resolved) != len(original_vias)
                or any(resolved[i] is not via
                       for i, via in enumerate(prior.vias))
            ):
                return None
            return trial.segs[len(prior.segs):], resolved[prior_count:]
        finally:
            for name, value in (
                ("_hard_rejections", saved_rejections),
                ("_hard_rejection_samples", saved_samples),
                ("_hard_rejection_generation", saved_generation),
            ):
                if value is missing:
                    self.__dict__.pop(name, None)
                else:
                    setattr(self, name, value)

    def _via_pad_attempt_conflicts(
        self,
        net: str,
        via: _ViaSite,
        *,
        candidate_segs: Sequence[_WireSeg] = (),
    ) -> Tuple[Tuple[int, Rect, Rect], ...]:
        """Deck-visible landing gaps from one access via to prior MST metal.

        Access enumeration runs after earlier edges of the same net have been
        committed into ``_attempt_idx``.  A via model is a bottom/cut/top
        transaction: an open sub-rule band from either landing pad to that exact
        prior metal is illegal unless other drawn metal fully covers the raw band
        and the deck therefore sees one merged polygon.  This is the same-net
        union-spacing obligation used by the final residual oracle, applied before
        a re-anchored single is exported into the A* access domain.
        """
        _sep = _eg_attr("_sep", _base_sep)
        below, above = self._via_pad_half_for(via)
        conflicts = set()
        for gi, half in ((via.lo, below), (via.lo + 1, above)):
            pad = _norm((
                via.x - half,
                via.y - half,
                via.x + half,
                via.y + half,
            ))
            attempt_index = self._attempt_idx.get(gi)
            if attempt_index is None:
                continue
            space = self._space[gi]
            query = (
                pad[0] - space,
                pad[1] - space,
                pad[2] + space,
                pad[3] + space,
            )
            for owner, blocker_raw in attempt_index.query(query):
                if owner != net:
                    continue
                blocker = _norm(blocker_raw)
                sx, sy = _sep(pad, blocker)
                if not (sx or sy):
                    continue
                if sx * sx + sy * sy >= space * space:
                    continue
                gap = self._gap_box(pad, blocker, sx, sy)
                gx0, gy0, gx1, gy1 = gap
                if gx1 <= gx0 or gy1 <= gy0:
                    conflicts.add((gi, pad, blocker))
                    continue

                cover = []
                for seg in candidate_segs:
                    if seg.gi != gi:
                        continue
                    rr = _seg_bbox(
                        seg.ax,
                        seg.ay,
                        seg.bx,
                        seg.by,
                        self._segment_half_width(net, seg),
                    )
                    if (rr[0] < gx1 and gx0 < rr[2]
                            and rr[1] < gy1 and gy0 < rr[3]):
                        cover.append(rr)
                for rect in self._attempt_metal.get(gi, ()):
                    rr = _norm(rect)
                    if (rr[0] < gx1 and gx0 < rr[2]
                            and rr[1] < gy1 and gy0 < rr[3]):
                        cover.append(rr)
                abstract = self._abstract_rects.get(gi, ())
                obstacle_index = self._obs_idx.get(gi)
                if obstacle_index is not None:
                    for _obstacle_owner, rect in obstacle_index.query(gap):
                        if rect in abstract:
                            continue
                        rr = _norm(rect)
                        if (rr[0] < gx1 and gx0 < rr[2]
                                and rr[1] < gy1 and gy0 < rr[3]):
                            cover.append(rr)
                committed_index = self._metal_idx.get(gi)
                if committed_index is not None:
                    for _committed_owner, rect in committed_index.query(gap):
                        rr = _norm(rect)
                        if (rr[0] < gx1 and gx0 < rr[2]
                                and rr[1] < gy1 and gy0 < rr[3]):
                            cover.append(rr)
                if subtract_rects(gap, cover):
                    conflicts.add((gi, pad, blocker))
        return tuple(sorted(conflicts))

    def _resolve_pattern_via_models(
        self,
        net: str,
        vias: Sequence[_ViaSite],
        *,
        mutable_indices: Optional[Set[int]] = None,
        segs: Sequence[_WireSeg] = (),
        allow_complete_owned_union: bool = False,
        owned_attempt_prior: Optional[_RoutedNet] = None,
    ) -> Optional[List[_ViaSite]]:
        """Resolve combined candidate conflicts under the selected via policy."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        current = list(vias)
        for via in current:
            self._require_policy_via(via)
        if not self._single_via_escape_allowed():
            if self._pattern_via_pad_visible_conflicts(net, current, segs):
                return None
            wire_conflicts = self._pattern_via_wire_conflicts(net, current, segs)
            if wire_conflicts:
                if not allow_complete_owned_union or not self._complete_owned_union_clears_wire_conflicts(
                    net, current, segs, wire_conflicts,
                ):
                    return None
            attempt_conflicts = tuple(
                (index, gi, pad, blocker)
                for index, via in enumerate(current)
                for gi, pad, blocker in self._via_pad_attempt_conflicts(
                    net, via, candidate_segs=segs,
                )
            )
            if attempt_conflicts and not self._owned_prior_attempt_conflicts_clear(
                net, owned_attempt_prior, current, segs, attempt_conflicts,
            ):
                return None
            return current

        mutable = (
            set(range(len(current)))
            if mutable_indices is None
            else set(mutable_indices)
        )
        def unresolved_state(trial_vias: List[_ViaSite]):
            v_conf = self._pattern_via_pad_visible_conflicts(net, trial_vias, segs)
            w_conf = self._pattern_via_wire_conflicts(net, trial_vias, segs)
            if w_conf and allow_complete_owned_union and self._complete_owned_union_clears_wire_conflicts(
                net, trial_vias, segs, w_conf,
            ):
                w_conf = ()
            a_rows = tuple(
                (idx, gi, pad, blocker)
                for idx, v in enumerate(trial_vias)
                for gi, pad, blocker in self._via_pad_attempt_conflicts(
                    net, v, candidate_segs=segs,
                )
            )
            if a_rows and self._owned_prior_attempt_conflicts_clear(
                net, owned_attempt_prior, trial_vias, segs, a_rows,
            ):
                a_rows = ()
            a_conf = tuple(sorted({row[0] for row in a_rows}))
            return v_conf, w_conf, a_conf

        via_conflicts, wire_conflicts, attempt_conflicts = unresolved_state(current)
        while via_conflicts or wire_conflicts or attempt_conflicts:
            conflict_count = (
                len(via_conflicts) + len(wire_conflicts) + len(attempt_conflicts)
            )
            options = []
            participants = sorted(
                {index for conflict in via_conflicts for index in conflict[:2]}
                | {conflict[0] for conflict in wire_conflicts}
                | set(attempt_conflicts)
            )
            for index in participants:
                if index not in mutable:
                    continue
                via = current[index]
                if self._site_n(via) == 1:
                    continue
                if not self._via_legal(via.x, via.y, via.lo, net, n=1):
                    continue
                trial = list(current)
                trial[index] = _ViaSite(via.x, via.y, via.lo, 1)
                rem_via, rem_wire, rem_att = unresolved_state(trial)
                rem_count = len(rem_via) + len(rem_wire) + len(rem_att)
                if rem_count > conflict_count:
                    continue
                signature = tuple((item.x, item.y, item.lo, item.n) for item in trial)
                options.append((
                    rem_count,
                    sum(self._site_n(item) == 1 for item in trial),
                    signature,
                    trial,
                    rem_via,
                    rem_wire,
                    rem_att,
                ))
            if not options:
                return None
            _rc, _sc, _sig, current, via_conflicts, wire_conflicts, attempt_conflicts = min(
                options,
                key=lambda item: (item[0], item[1], item[2]),
            )
        return current

    def _complete_owned_union_clears_wire_conflicts(self, net, vias, segs, conflicts):
        """Use literal same-owner geometry only after a scoped fill transaction."""
        if not conflicts:
            return False
        return self._complete_owned_union_clears_layers(
            net, vias, segs, {int(row[2]) for row in conflicts},
        )

    def _authenticated_owned_book_metal(self, net, current_route):
        """Bind book metal, pads and cut ledgers to actual emitted route owners."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        book = getattr(self, '_book', None)
        if book is None or not isinstance(current_route, _RoutedNet) or current_route.name != net:
            return None
        work = 0
        def checkpoint():
            nonlocal work
            if work % 256 == 0:
                self._watchdog_check(net, 'candidate_owned_book')
            work += 1
        def kernel_checkpoint():
            self._watchdog_check(net, 'candidate_owned_book')
        def flatten(data):
            result = Counter()
            for gi, rectangles in data.items():
                for rect in rectangles:
                    checkpoint()
                    result[(int(gi), _norm(rect))] += 1
            return result
        def cut_rows(cuts):
            result = Counter()
            for rect, lo in cuts:
                checkpoint()
                result[(int(lo), _norm(rect))] += 1
            return result
        def emitted_matches(route, expected_metal):
            metal_layers = {tuple(self._tech.gi_to_metal_key(gi)): int(gi)
                            for gi in expected_metal}
            cut_layers = {}
            expected_cuts = Counter()
            for via in route.vias:
                checkpoint()
                try:
                    self._require_policy_via(via)
                except ValueError:
                    return False
                raw_key = self._tech.via_key_for(via.lo)
                if raw_key is None:
                    return False
                key = tuple(raw_key)
                if key in metal_layers:
                    return False
                cut_layers[key] = int(via.lo)
                count = self._site_n(via)
                if count is None:
                    count = self.via_redundancy
                centers = tuple(self._via_cut_centers(via.x, via.y, via.lo, count))
                if len(centers) != count * count or len(set(centers)) != len(centers):
                    return False
                for cx, cy in centers:
                    checkpoint()
                    expected_cuts[(int(via.lo), _norm(landing_rect_for_cut(
                        cx, cy, self._via_cut[via.lo], 0,
                    )))] += 1
            actual_metal, actual_cuts = Counter(), Counter()
            self._watchdog_check(net, 'candidate_owned_book')
            polygons = self._emit_net(route, checkpoint=kernel_checkpoint).wires
            self._watchdog_check(net, 'candidate_owned_book')
            for polygon in polygons:
                checkpoint()
                layer = (int(polygon.layer), int(polygon.datatype))
                vertices = tuple((int(p.x), int(p.y)) for p in polygon.polygon.vertices)
                if len(vertices) != 4:
                    return False
                xs, ys = zip(*vertices)
                box = (min(xs), min(ys), max(xs), max(ys))
                if set(vertices) != {(box[0],box[1]),(box[2],box[1]),
                                     (box[2],box[3]),(box[0],box[3])}:
                    return False
                if layer in metal_layers:
                    actual_metal[(metal_layers[layer], box)] += 1
                elif layer in cut_layers:
                    actual_cuts[(cut_layers[layer], box)] += 1
                else:
                    return False
            return actual_metal == flatten(expected_metal) and actual_cuts == expected_cuts
        self._watchdog_check(net, 'candidate_owned_book')
        current_metal, current_cuts = self._net_metal(current_route, checkpoint=kernel_checkpoint)
        current_pads = self._net_via_pads(current_route, checkpoint=kernel_checkpoint)
        self._watchdog_check(net, 'candidate_owned_book')
        if not emitted_matches(current_route, current_metal):
            return None
        persistent = book.persistent.get(net)
        baseline = getattr(self, '_active_group_route_union_baseline', None)
        if baseline is None:
            baseline = getattr(self, '_group_route_landing_baselines', {}).get(net)
        base_metal, base_cuts, base_pads = {}, [], {}
        if persistent is not None:
            if not isinstance(baseline, _RoutedNet) or baseline.name != net:
                return None
            base_metal, base_cuts = self._net_metal(baseline, checkpoint=kernel_checkpoint)
            base_pads = self._net_via_pads(baseline, checkpoint=kernel_checkpoint)
            if (flatten(base_metal) != flatten(persistent[0])
                    or cut_rows(base_cuts) != cut_rows(persistent[1])
                    or flatten(base_pads) != flatten(persistent[2])
                    or not emitted_matches(baseline, base_metal)):
                return None
        elif baseline is not None and (baseline.segs or baseline.vias):
            return None
        def booked_rows(index):
            result = Counter()
            for gi, rows in index.items():
                for owner, rect in rows:
                    checkpoint()
                    if owner == net:
                        result[(int(gi), _norm(rect))] += 1
            return result
        def bounded(booked, required, candidate):
            return all(required[key] <= booked[key] <= required[key] + candidate[key]
                       for key in (booked.keys() | required.keys() | candidate.keys()))
        booked_cuts = Counter()
        for owner, rect, lo in book.cuts:
            checkpoint()
            if owner == net:
                booked_cuts[(int(lo), _norm(rect))] += 1
        if (not bounded(booked_rows(book.metal), flatten(base_metal), flatten(current_metal))
                or not bounded(booked_rows(book.via_pads), flatten(base_pads), flatten(current_pads))
                or not bounded(booked_cuts, cut_rows(base_cuts), cut_rows(current_cuts))):
            return None
        # Actual foreign conductors may neither duplicate nor partially touch
        # the authenticated route. Retain ordinary positive spacing as a soft
        # negotiation residual; only closed contact is rejected here.
        foreign_metal = {}
        for index in (book.metal, book.via_pads):
            for gi, rows in index.items():
                for owner, rect in rows:
                    checkpoint()
                    if owner != net:
                        foreign_metal.setdefault(int(gi), _SpatialIndex()).add(owner, _norm(rect))
        for gi in set(base_metal) | set(current_metal):
            index = foreign_metal.get(int(gi))
            if index is None:
                continue
            for rect in (*base_metal.get(gi, ()), *current_metal.get(gi, ())):
                checkpoint()
                for _owner, other in index.query(_norm(rect)):
                    checkpoint()
                    if _closed_rects_touch(_norm(rect), _norm(other)):
                        return None
        # The existing committed cut gate uses per-site outer envelopes. Use
        # the identical hard-contact rule against foreign book cuts here; these
        # ledger bounds are never added to the conductor union.
        foreign_cuts = {}
        for owner, rect, lo in book.cuts:
            checkpoint()
            if owner != net:
                foreign_cuts.setdefault(int(lo), _SpatialIndex()).add(owner, _norm(rect))
        for lo, rect in (cut_rows(base_cuts) + cut_rows(current_cuts)):
            checkpoint()
            index = foreign_cuts.get(lo)
            if index is None:
                continue
            for _owner, other in index.query(rect):
                checkpoint()
                if _closed_rects_touch(rect, _norm(other)):
                    return None
        return base_metal

    def _complete_owned_union_clears_layers(self, net, vias, segs, layers):
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        if not layers or not segs or getattr(self, '_tech', None) is None:
            return False
        literals = getattr(self, '_fixed_deck_literal_rects', None)
        if not literals:
            return False
        route = _RoutedNet(net, segs=list(segs), vias=list(vias))
        metal = self._net_metal(route, checkpoint=lambda: self._watchdog_check(
            net, 'candidate_owned_union'))[0]
        baseline = self._authenticated_owned_book_metal(net, route)
        if baseline is None:
            return False
        for gi in sorted(set(layers) | set(baseline)):
            self._watchdog_check(net, 'candidate_owned_union')
            rects = (*metal.get(gi, ()), *baseline.get(gi, ()))
            if not rects:
                return False
            key = tuple(self._tech.gi_to_metal_key(gi))
            abstract = self._abstract_rects.get(gi, ())
            owned = []
            for index, (layer, owner, rect) in enumerate(literals):
                if index % 256 == 0:
                    self._watchdog_check(net, 'candidate_owned_union')
                if tuple(layer) == key and owner == net and rect not in abstract:
                    owned.append(rect)
            if not self._merged_same_net_metal_is_rule_clear(
                [*rects, *owned], minimum_width=2*self._half_w[gi],
                minimum_space=self._space[gi],
                checkpoint=lambda: self._watchdog_check(net, 'candidate_owned_union'),
            ):
                return False
            self._watchdog_check(net, 'candidate_owned_union')
        return True

    def _owned_prior_attempt_conflicts_clear(self, net, prior, vias, segs, conflicts):
        """Reconcile only exact prior emitted pad/segment attempt witnesses."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        if (not isinstance(prior, _RoutedNet) or prior.name != net or not conflicts
                or len(segs) < len(prior.segs) or len(vias) < len(prior.vias)
                or any(item is not old for item, old in zip(segs, prior.segs))
                or any(item is not old for item, old in zip(vias, prior.vias))):
            return False
        proof_work = 0
        def check_proof():
            nonlocal proof_work
            if proof_work % 256 == 0:
                self._watchdog_check(net, 'candidate_owned_attempt_proof')
            proof_work += 1
        prior_metal = self._net_metal(prior, checkpoint=lambda: self._watchdog_check(
            net, 'candidate_owned_attempt_input'))[0]
        prior_wires = {}
        for segment in prior.segs:
            check_proof()
            prior_wires.setdefault(segment.gi, set()).add(_norm(_seg_bbox(
                segment.ax, segment.ay, segment.bx, segment.by,
                self._segment_half_width(net, segment),
            )))
        prior_emitted = {}
        for gi, rectangles in prior_metal.items():
            emitted = prior_emitted.setdefault(gi, set())
            for rectangle in rectangles:
                check_proof()
                emitted.add(_norm(rectangle))
        for index, gi, pad, blocker in conflicts:
            check_proof()
            if index >= len(prior.vias):
                return False
            via = prior.vias[index]
            below, above = self._via_pad_half_for(via)
            half = below if gi == via.lo else above if gi == via.lo + 1 else None
            if half is None or _norm(pad) != _norm((
                via.x-half, via.y-half, via.x+half, via.y+half,
            )):
                return False
            emitted = prior_emitted.get(gi, ())
            if (_norm(pad) not in emitted or _norm(blocker) not in emitted
                    or _norm(blocker) not in prior_wires.get(gi, ())):
                return False
            idx = getattr(self, '_attempt_idx', {}).get(gi)
            if idx is None:
                return False
            owners = set()
            for owner, rect in idx.query(_norm(blocker)):
                check_proof()
                if _norm(rect) == _norm(blocker):
                    owners.add(owner)
            if owners != {net}:
                return False
        return self._complete_owned_union_clears_layers(
            net, vias, segs, {int(gi) for _index, gi, _pad, _blocker in conflicts},
        )

    def _candidate_new_owned_spacing_wire(self, net, prior, segs, vias):
        """Map a new full-union spacing witness to one exact new wire chain.

        Only real emitted route rectangles and same-owner canonical fixed
        literals participate. A dirty prior, missing literal authority, a
        marker with no wire witness, or distinct supporting chains fails closed.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        import klayout.db as kdb

        literals = getattr(self, '_fixed_deck_literal_rects', None)
        if (not isinstance(prior, _RoutedNet) or prior.name != net or not segs
                or not literals or getattr(self, '_tech', None) is None):
            return None
        old = self._net_metal(prior, checkpoint=lambda: self._watchdog_check(
            net, 'candidate_owned_spacing_input'))[0]
        combined_route = _RoutedNet(
            net, segs=[*prior.segs, *segs], vias=[*prior.vias, *vias],
        )
        combined = self._net_metal(combined_route, checkpoint=lambda: self._watchdog_check(
            net, 'candidate_owned_spacing_input'))[0]
        baseline = self._authenticated_owned_book_metal(net, combined_route)
        if baseline is None:
            return None
        chains = {}
        grouped = {}
        scan_work = 0
        def check_scan():
            nonlocal scan_work
            if scan_work % 256 == 0:
                self._watchdog_check(net, 'candidate_owned_spacing_retry')
            scan_work += 1
        for index, segment in enumerate(segs):
            check_scan()
            if segment.ax == segment.bx and segment.ay != segment.by:
                orientation, constant = 'v', segment.ax
                start, stop = sorted((segment.ay, segment.by))
            elif segment.ay == segment.by and segment.ax != segment.bx:
                orientation, constant = 'h', segment.ay
                start, stop = sorted((segment.ax, segment.bx))
            else:
                continue
            half = self._segment_half_width(net, segment)
            grouped.setdefault((segment.gi, orientation, constant, half), []).append(
                (start, stop, index))
        for key, intervals in sorted(grouped.items()):
            runs = []
            for start, stop, index in sorted(intervals):
                check_scan()
                if runs and start <= runs[-1][1]:
                    runs[-1][1] = max(stop, runs[-1][1])
                    runs[-1][2].append(index)
                else:
                    runs.append([start, stop, [index]])
            gi, orientation, constant, half = key
            for start, stop, indices in runs:
                check_scan()
                wire = (_WireSeg(gi, start, constant, stop, constant, half)
                        if orientation == 'h' else
                        _WireSeg(gi, constant, start, constant, stop, half))
                identity = self._wire_search_identity(net, wire)
                for index in indices:
                    chains[index] = identity

        def edge_on_box(edge, box):
            x0, y0, x1, y1 = _norm(box)
            a, b = edge.p1, edge.p2
            if a.y == b.y and a.y in (y0, y1):
                return min(max(a.x, b.x), x1) > max(min(a.x, b.x), x0)
            if a.x == b.x and a.x in (x0, x1):
                return min(max(a.y, b.y), y1) > max(min(a.y, b.y), y0)
            return False

        found = set()
        for gi in sorted(set(combined) | set(baseline)):
            rects = combined.get(gi, ())
            self._watchdog_check(net, 'candidate_owned_spacing_geometry')
            key = tuple(self._tech.gi_to_metal_key(gi))
            abstract = self._abstract_rects.get(gi, ())
            fixed = []
            for layer, owner, rect in literals:
                check_scan()
                if tuple(layer) == key and owner == net and rect not in abstract:
                    fixed.append(rect)
            before = [*old.get(gi, ()), *baseline.get(gi, ()), *fixed]
            if not before or not self._merged_same_net_metal_is_rule_clear(
                before, minimum_width=2*self._half_w[gi], minimum_space=self._space[gi],
                checkpoint=lambda: self._watchdog_check(net, 'candidate_owned_spacing_geometry'),
            ):
                return None
            region = kdb.Region()
            for rectangle in (*rects, *baseline.get(gi, ()), *fixed):
                check_scan()
                region.insert(kdb.Box(*_norm(rectangle)))
            region.merge()
            self._watchdog_check(net, 'candidate_owned_spacing_geometry')
            width_errors = region.width_check(int(2*self._half_w[gi]))
            self._watchdog_check(net, 'candidate_owned_spacing_geometry')
            if not width_errors.is_empty():
                return None
            space_errors = region.space_check(int(self._space[gi]))
            self._watchdog_check(net, 'candidate_owned_spacing_geometry')
            for pair in space_errors.each():
                witnesses = set()
                for index, segment in enumerate(segs):
                    check_scan()
                    if segment.gi != gi or index not in chains:
                        continue
                    box = _seg_bbox(segment.ax, segment.ay, segment.bx, segment.by,
                                    self._segment_half_width(net, segment))
                    if edge_on_box(pair.first, box) or edge_on_box(pair.second, box):
                        witnesses.add(chains[index])
                if len(witnesses) != 1:
                    return None
                found.update(witnesses)
            self._watchdog_check(net, 'candidate_owned_spacing_geometry')
        return next(iter(found)) if len(found) == 1 else None

    def _pattern_congestion(
        self, net: str, segs: Sequence[_WireSeg], vias: Sequence[_ViaSite],
    ) -> int:
        """Present/history price shared by every pattern-first fast path."""
        score = sum(
            self._edge_congestion(
                seg.gi, seg.ax, seg.ay, seg.bx, seg.by, net
            )
            for seg in segs
        )
        score += sum(
            self._via_congestion(
                via.x, via.y, via.lo, net, self._site_n(via)
            )
            for via in vias
        )
        return score

    def _heuristic(self, x: int, y: int, gi: int,
                   goal: Tuple[int, int, int]) -> int:
        gx, gy, gl = goal
        return (abs(x - gx) + abs(y - gy)) * _STEP_COST_PER_NM + abs(gi - gl) * _VIA_COST

    def _cached_exact_multigoal_potential(
        self,
        xs: Sequence[int],
        ys: Sequence[int],
        layers: Sequence[int],
        goal_terminal_cost: Mapping[Tuple[int, int, int], int],
        net: str,
    ) -> List[int]:
        """Reuse a bit-identical goal transform across negotiated replays.

        The transform is obstacle-free and independent of present/history cost;
        its complete semantic key is therefore the graph axes, layer coordinates,
        and per-goal terminal lower bounds.  Cache authority is measured in table
        elements and bounded by ``max_expansions``, the engine's existing nominal
        A* work declaration.  Least-recently-used eviction is deterministic for a
        deterministic route trace, and a table larger than that authority simply
        runs uncached.
        """
        key = (
            tuple(int(value) for value in xs),
            tuple(int(value) for value in ys),
            tuple(int(value) for value in layers),
            tuple(sorted(
                (
                    tuple(int(value) for value in goal),
                    int(cost),
                )
                for goal, cost in goal_terminal_cost.items()
            )),
        )
        cache = self._exact_potential_cache
        cached = cache.pop(key, None)
        if cached is not None:
            cache[key] = cached
            self._potential_transform_cache_hits_total += 1
            return cached[1]

        self._potential_transform_cache_misses_total += 1
        potential = self._exact_multigoal_potential(
            xs, ys, layers, goal_terminal_cost, net
        )
        element_count = len(potential)
        capacity = max(0, int(self.max_expansions))
        if element_count > capacity:
            return potential
        while (
            cache
            and self._exact_potential_cache_elements + element_count > capacity
        ):
            _old_key, (old_count, _old_value) = cache.popitem(last=False)
            self._exact_potential_cache_elements -= int(old_count)
        if element_count <= capacity:
            cache[key] = (element_count, potential)
            self._exact_potential_cache_elements += element_count
        return potential

    def _exact_multigoal_potential(
        self,
        xs: Sequence[int],
        ys: Sequence[int],
        layers: Sequence[int],
        goal_terminal_cost: Mapping[Tuple[int, int, int], int],
        net: str,
    ) -> List[int]:
        """Exact obstacle-free cost-to-go on an irregular Cartesian graph.

        The old multi-goal A* evaluated ``min(heuristic(state, goal))`` by
        scanning every access alternative on every heap push *and* pop.  Exact
        owned-component throat enumeration can legitimately expose hundreds of
        alternatives, turning a small graph search into ``O(A* work * goals)``.

        Manhattan distance is separable.  Seed every goal with its terminal-via
        cost, then perform forward/backward min-plus transforms along x, y and
        layer.  The result is bit-for-bit the old minimum heuristic at every
        graph coordinate, but each A* lookup is O(1).  Callers only materialize
        this table when the graph already fits the engine's declared state cap.
        """
        nx = len(xs)
        ny = len(ys)
        nl = len(layers)
        if not nx or not ny or not nl or not goal_terminal_cost:
            return []
        plane = nx * ny
        if plane * nl * 3 > self.expansion_cap:
            raise ValueError(
                "multigoal potential graph exceeds expansion cap"
            )

        cpp = load_native("_gridless_cpp")
        if cpp is not None and "_watchdog_check" not in self.__dict__:
            self._watchdog_check(net, "astar_potential")
            try:
                result = cpp.exact_distance_transform_cpp(
                    xs, ys, layers, goal_terminal_cost,
                    _STEP_COST_PER_NM, _VIA_COST,
                )
            except OverflowError:
                result = None
            if result is not None:
                self._watchdog_check(net, "astar_potential")
                return result

        work = 0

        def watchdog() -> None:
            nonlocal work
            if work % 4096 == 0:
                self._watchdog_check(net, "astar_potential")
            work += 1

        def ordered_axis(
            values: Sequence[int],
        ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
            ordered = []
            index_by_value = {}
            previous = None
            for index, raw_value in enumerate(values):
                watchdog()
                value = int(raw_value)
                if previous is not None and value <= previous:
                    raise ValueError(
                        "multigoal potential axes must be strictly increasing"
                    )
                ordered.append(value)
                index_by_value[value] = index
                previous = value
            return tuple(ordered), index_by_value

        ordered_xs, x_index = ordered_axis(xs)
        ordered_ys, y_index = ordered_axis(ys)
        ordered_layers, layer_index = ordered_axis(layers)
        maximum_terminal_cost = None
        for terminal_cost in goal_terminal_cost.values():
            watchdog()
            terminal_cost = int(terminal_cost)
            if (
                maximum_terminal_cost is None
                or terminal_cost > maximum_terminal_cost
            ):
                maximum_terminal_cost = terminal_cost
        assert maximum_terminal_cost is not None
        upper_bound = (
            maximum_terminal_cost
            + (ordered_xs[-1] - ordered_xs[0]) * _STEP_COST_PER_NM
            + (ordered_ys[-1] - ordered_ys[0]) * _STEP_COST_PER_NM
            + (ordered_layers[-1] - ordered_layers[0]) * _VIA_COST
            + 1
        )
        watchdog()
        potential = [upper_bound] * (plane * nl)
        # Seeding is a commutative per-index minimum, so mapping iteration order
        # cannot affect the table.  Avoiding ``sorted(items())`` prevents an O(G)
        # duplicate allocation before the typed watchdog can run.
        for goal, terminal_cost in goal_terminal_cost.items():
            watchdog()
            gx, gy, gl = goal
            index = (
                layer_index[int(gl)] * plane
                + y_index[int(gy)] * nx
                + x_index[int(gx)]
            )
            potential[index] = min(potential[index], int(terminal_cost))

        # Exact 1-D irregular-coordinate distance transforms, composed over the
        # three separable dimensions.  Sorted loop order is part of determinism.
        for layer_offset in range(0, plane * nl, plane):
            for y_offset in range(0, plane, nx):
                row = layer_offset + y_offset
                for x_pos in range(1, nx):
                    watchdog()
                    index = row + x_pos
                    candidate = (
                        potential[index - 1]
                        + (ordered_xs[x_pos] - ordered_xs[x_pos - 1])
                        * _STEP_COST_PER_NM
                    )
                    if candidate < potential[index]:
                        potential[index] = candidate
                for x_pos in range(nx - 2, -1, -1):
                    watchdog()
                    index = row + x_pos
                    candidate = (
                        potential[index + 1]
                        + (ordered_xs[x_pos + 1] - ordered_xs[x_pos])
                        * _STEP_COST_PER_NM
                    )
                    if candidate < potential[index]:
                        potential[index] = candidate

        for layer_offset in range(0, plane * nl, plane):
            for x_pos in range(nx):
                for y_pos in range(1, ny):
                    watchdog()
                    index = layer_offset + y_pos * nx + x_pos
                    candidate = (
                        potential[index - nx]
                        + (ordered_ys[y_pos] - ordered_ys[y_pos - 1])
                        * _STEP_COST_PER_NM
                    )
                    if candidate < potential[index]:
                        potential[index] = candidate
                for y_pos in range(ny - 2, -1, -1):
                    watchdog()
                    index = layer_offset + y_pos * nx + x_pos
                    candidate = (
                        potential[index + nx]
                        + (ordered_ys[y_pos + 1] - ordered_ys[y_pos])
                        * _STEP_COST_PER_NM
                    )
                    if candidate < potential[index]:
                        potential[index] = candidate

        for y_pos in range(ny):
            for x_pos in range(nx):
                offset = y_pos * nx + x_pos
                for layer_pos in range(1, nl):
                    watchdog()
                    index = layer_pos * plane + offset
                    candidate = (
                        potential[index - plane]
                        + (ordered_layers[layer_pos]
                           - ordered_layers[layer_pos - 1])
                        * _VIA_COST
                    )
                    if candidate < potential[index]:
                        potential[index] = candidate
                for layer_pos in range(nl - 2, -1, -1):
                    watchdog()
                    index = layer_pos * plane + offset
                    candidate = (
                        potential[index + plane]
                        + (ordered_layers[layer_pos + 1]
                           - ordered_layers[layer_pos])
                        * _VIA_COST
                    )
                    if candidate < potential[index]:
                        potential[index] = candidate
        return potential

    def _turn_cost(self) -> int:
        """Per-corner A* penalty in COST units: ``_TURN_COST_PITCHES`` routing
        pitches (one pitch == ``self._res_cell`` nm == min_width + min_space), so
        a corner scales with the PDK routing pitch exactly like the congestion
        (``_CONG_WEIGHT``) and trunk (``_TRUNK_SPAN_PITCHES``) terms rather than a
        fixed nm constant a pitch-scaled congestion term on a large cell could
        outbid. General, keyed to no circuit."""
        return _TURN_COST_PITCHES * self._res_cell

    @staticmethod
    def _via_successor_last_dir(pre_via_last_dir: int) -> int:
        """The ``last_dir`` a state carries THROUGH a via layer change: the
        incoming planar direction is PRESERVED (not reset to 0), so a corner that
        straddles a via is still charged ``_turn_cost`` -- closing the free-kink-
        at-a-via gap (a multi-via net used to kink for free at every via). Returns
        the incoming direction unchanged."""
        return pre_via_last_dir

    def _planar_event_geometry(self, net: str, domain: _PlanarEndpointDomain):
        """Build a conservative candidate region from the current route world."""
        class RegionBudgetStop(Exception):
            pass

        work = 0

        def checkpoint():
            nonlocal work
            if self._solve_expansions_total >= self._solve_expansion_budget:
                raise RegionBudgetStop("solve_budget")
            if work >= self.max_expansions:
                raise RegionBudgetStop("budget")
            work += 1
            self._solve_expansions_total += 1
            self._watchdog_check(net, "planar_region_geometry")

        try:
            return self._planar_event_geometry_bounded(net, domain, checkpoint)
        except RegionBudgetStop as stop:
            self._edge_fail = (str(stop), work, self.max_expansions,
                               self._solve_expansions_total, self._solve_expansion_budget,
                               "planar_region_geometry")
            return None

    def _planar_event_geometry_bounded(self, net, domain, checkpoint):
        return self._planar_event_geometry_for_layer(
            net, domain.node[2], domain.grid, checkpoint,
        )

    def _planar_event_geometry_for_layer(self, net, gi, grid, checkpoint):
        """Build the same conservative planar region for an explicit active layer."""
        from .region_event_geometry import conservative_center_region, build_region_geometry

        checkpoint()

        half = int(self._hw(net, gi))
        ordinary = int(self._space[gi])
        bounds = tuple(_norm(self._bounds))
        envelope = getattr(self, "_routing_envelope", None)
        if envelope is not None:
            ex0, ey0, ex1, ey1 = envelope.bbox_nm
            bounds = (max(bounds[0], ex0 + half), max(bounds[1], ey0 + half),
                      min(bounds[2], ex1 - half), min(bounds[3], ey1 - half))
        if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
            return None
        # The wire oracle rejects strict spacing violations; equality is legal.
        # On integer coordinates its forbidden closed interval ends one nm
        # before the clearance boundary.  Keep zero-spacing behavior unchanged.
        halo = half + ordinary - 1 if ordinary > 0 else half + grid
        query = (bounds[0] - halo, bounds[1] - halo,
                 bounds[2] + halo, bounds[3] + halo)
        blockers = []
        fixed_index = getattr(self, "_obs_idx", {}).get(gi)
        literal_index = getattr(self, "_literal_planar_idx", {}).get(gi)
        fixed_rows = (() if fixed_index is None else fixed_index.query(query))
        literal_rows = (self._fixed_literal_metal_rows().get(gi, ()) if literal_index is None
                        else literal_index.query(query))
        for rows in (fixed_rows, literal_rows, self._book.metal.get(gi, ()),
                     self._book.via_pads.get(gi, ())):
            for owner, rect in rows:
                checkpoint()
                if owner != net:
                    blockers.append((tuple(_norm(rect)), halo))
        checkpoint()
        for space, core, _window, attached in self._wide_indexes(True).get(gi, ()):
            extra = half + int(space) + grid
            wide_query = (bounds[0] - extra, bounds[1] - extra,
                          bounds[2] + extra, bounds[3] + extra)
            for index in (core, attached):
                for owner, rect in index.query(wide_query):
                    checkpoint()
                    if owner != net:
                        blockers.append((tuple(_norm(rect)), extra))
        if self._mim_derived_unrelated_static_active():
            for m_owner, m_rect, m_clear, _m_metric, _m_rule in getattr(
                self, "_mim_derived_unrelated_clearance_by_gi", {}
            ).get(gi, ()):
                checkpoint()
                if m_owner != net and int(m_clear) > 0:
                    blockers.append((
                        tuple(_norm(m_rect)),
                        half + int(m_clear) - 1,
                    ))
        region = conservative_center_region(bounds, tuple(blockers), grid_nm=grid,
                                            checkpoint=checkpoint)
        geometry = build_region_geometry(region, checkpoint=checkpoint)
        index = _SpatialIndex(cell=max(32, 1000 // max(1, int(grid))))
        for number, (x0, y0, x1, y1) in enumerate(geometry.rectangles):
            checkpoint()
            index.add(str(number), (x0, y0, x1 - 1, y1 - 1))
        portals = [[] for _ in geometry.rectangles]
        for portal in geometry.portals:
            checkpoint()
            portals[portal.source].append(portal)
        return geometry, tuple(tuple(row) for row in portals), index

    def _planar_via_event_regions(self, net, lo, lower, upper, grid, checkpoint):
        """Candidate via-centre regions in the two current planar worlds.

        Rectangle halos use the actual selected array footprint, current fixed
        and booked conductors, named spacing and cut ledgers.  They deliberately
        overexclude; remaining candidates still require the full via oracle.
        """
        import klayout.db as kdb
        from .region_event_geometry import conservative_center_region, build_region_geometry

        bounds = tuple(_norm(self._bounds))
        spacing_partners = getattr(self, "_intent_spacing_rules", {}).get(net, {})
        blockers = []
        single_allowed = self._single_via_escape_allowed()
        pad_halves = (
            self._via_pad_half1[lo]
            if single_allowed
            else self._via_pad_half[lo]
        )
        for gi, half in ((lo, pad_halves[0]), (lo + 1, pad_halves[1])):
            space = max(int(self._space[gi]), max(spacing_partners.values(), default=0))
            halo = int(half) + space + grid
            query = (bounds[0] - halo, bounds[1] - halo,
                     bounds[2] + halo, bounds[3] + halo)
            fixed = getattr(self, "_obs_idx", {}).get(gi)
            literal = getattr(self, "_literal_planar_idx", {}).get(gi)
            for rows in (() if fixed is None else fixed.query(query),
                         self._fixed_literal_metal_rows().get(gi, ()) if literal is None else literal.query(query),
                         self._book.metal.get(gi, ()), self._book.via_pads.get(gi, ())):
                for owner, rect in rows:
                    checkpoint()
                    if owner != net:
                        blockers.append((tuple(_norm(rect)), halo))
            for wide_space, core, _window, attached in self._wide_indexes(True).get(gi, ()):
                wide_halo = int(half) + int(wide_space) + grid
                wide_query = (bounds[0] - wide_halo, bounds[1] - wide_halo,
                              bounds[2] + wide_halo, bounds[3] + wide_halo)
                for index in (core, attached):
                    for owner, rect in index.query(wide_query):
                        checkpoint()
                        if owner != net:
                            blockers.append((tuple(_norm(rect)), wide_halo))
            if self._mim_derived_unrelated_static_active():
                for m_owner, m_rect, m_clear, _m_metric, _m_rule in getattr(
                    self, "_mim_derived_unrelated_clearance_by_gi", {}
                ).get(gi, ()):
                    checkpoint()
                    if m_owner != net and int(m_clear) > 0:
                        blockers.append((
                            tuple(_norm(m_rect)),
                            int(half) + int(m_clear) - 1,
                        ))
        cut_halo = (
            self._via_cut_extent(lo, 1 if single_allowed else None)
            + int(self._via_space[lo])
            + grid
        )
        cut_query = (bounds[0] - cut_halo, bounds[1] - cut_halo,
                     bounds[2] + cut_halo, bounds[3] + cut_halo)
        for rect in self._local_fixed_cuts(lo, cut_query):
            checkpoint()
            blockers.append((tuple(_norm(rect)), cut_halo))
        cut_index = self._cut_idx.get(lo)
        if cut_index is not None:
            for _owner, rect in cut_index.query(cut_query):
                checkpoint()
                blockers.append((tuple(_norm(rect)), cut_halo))
        for rect in self._attempt_cuts.get(lo, ()):
            checkpoint()
            blockers.append((tuple(_norm(rect)), cut_halo))
        candidates = conservative_center_region(bounds, tuple(blockers), grid_nm=grid,
                                                checkpoint=checkpoint)
        for graph in (lower, upper):
            mask = kdb.Region()
            for rect in graph[0].rectangles:
                checkpoint()
                mask.insert(kdb.Box(*rect))
            candidates &= mask.merged()
            checkpoint()
        return build_region_geometry(candidates, checkpoint=checkpoint).rectangles

    def _planar_exact_hard_precheck(
        self,
        starts: Sequence[Tuple[Tuple[int, int, int], Optional[_ViaSite]]],
        goals: Mapping[Tuple[int, int, int], Optional[_ViaSite]],
        net: str,
        ordered_layers: List[int],
        target_domain: Optional[_PlanarEndpointDomain],
        effective_target_regions: Tuple[Any, ...],
        *,
        xs: Sequence[int] = (),
        ys: Sequence[int] = (),
        source_domain: Optional[_PlanarEndpointDomain] = None,
        effective_source_regions: Tuple[Any, ...] = (),
    ) -> Tuple[bool, float]:
        """Conservative multilayer reachability under exact hard wire/via rules.

        Fixed obstacles, wide-metal cores, and cuts use ``_rects_clear`` (L_inf),
        matching ``_clear_obstacles``, ``_wide_spacing_conflicts``, and
        ``_scan_committed_cuts``.  Committed via pads and enforced committed
        wires use exact integer Euclidean circle boxes matching
        ``_scan_committed_metal`` (``sx*sx + sy*sy < sp*sp``).  Therefore every
        legal ``_wire_legal_single`` / ``_via_legal`` path lies inside
        ``(F_wire, F_via)`` with zero false disconnections.
        """
        import klayout.db as db
        from clay.layout.route.planar_precheck import _polygon_has_mfg_grid_point

        t0 = time.perf_counter()
        if not starts or (not goals and not effective_target_regions):
            return False, (time.perf_counter() - t0) * 1000.0

        bx0, by0, bx1, by1 = _norm(self._bounds)
        if xs:
            bx0, bx1 = min(bx0, min(xs)), max(bx1, max(xs))
        if ys:
            by0, by1 = min(by0, min(ys)), max(by1, max(ys))
        for (sx, sy, _sgi), _v in starts:
            bx0, by0 = min(bx0, sx), min(by0, sy)
            bx1, by1 = max(bx1, sx), max(by1, sy)
        for gx, gy, _ggi in goals:
            bx0, by0 = min(bx0, gx), min(by0, gy)
            bx1, by1 = max(bx1, gx), max(by1, gy)
        for _atom, (rx0, ry0, rx1, ry1) in effective_target_regions:
            bx0, by0 = min(bx0, rx0), min(by0, ry0)
            bx1, by1 = max(bx1, rx1), max(by1, ry1)
        for _atom, (rx0, ry0, rx1, ry1) in effective_source_regions:
            bx0, by0 = min(bx0, rx0), min(by0, ry0)
            bx1, by1 = max(bx1, rx1), max(by1, ry1)

        envelope = getattr(self, "_routing_envelope", None)
        env_bbox = envelope.bbox_nm if envelope is not None else None
        local_iface = (
            getattr(self, "_active_group_route_local_interface", None)
            if bool(getattr(self, "_active_group_route_local_closure", False))
            else None
        )
        neg_legalize = bool(getattr(self, "_negotiation_port_legalize", False))
        grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
        layer_set = set(ordered_layers)
        wide_map = self._wide_indexes(True)

        use_single_via = bool(self._single_via_escape_allowed())

        def _build_and_check(use_exact_euclid: bool):
            insert_committed = (
                _insert_exact_euclid_violation_boxes
                if use_exact_euclid
                else _insert_linf_violation_box
            )
            f_wire: Dict[int, Any] = {}
            for gi in ordered_layers:
                base_hw = int(self._half_w[gi])
                half = int(self._hw(net, gi))
                sp = int(self._space[gi])
                lx0, ly0, lx1, ly1 = bx0, by0, bx1, by1
                if env_bbox is not None:
                    lx0 = max(lx0, env_bbox[0] + half)
                    ly0 = max(ly0, env_bbox[1] + half)
                    lx1 = min(lx1, env_bbox[2] - half)
                    ly1 = min(ly1, env_bbox[3] - half)
                if local_iface is not None:
                    ex0, ey0, ex1, ey1 = (int(v) for v in local_iface[0])
                    px, py = int(local_iface[1][0]), int(local_iface[1][1])
                    lx0 = max(lx0, ex0)
                    ly0 = max(ly0, ey0)
                    lx1 = min(lx1, ex1)
                    ly1 = min(ly1, ey1)
                    if lx0 > lx1 or ly0 > ly1:
                        f_wire[gi] = db.Region()
                        continue
                    outer = db.Region()
                    ix0 = max(lx0, ex0 + half)
                    iy0 = max(ly0, ey0 + half)
                    ix1 = min(lx1, ex1 - half)
                    iy1 = min(ly1, ey1 - half)
                    if ix0 <= ix1 and iy0 <= iy1:
                        outer.insert(db.Box(ix0, iy0, ix1 + 1, iy1 + 1))
                    sx0 = max(lx0, ex0 if px == ex0 else ex0 + half, px - half)
                    sy0 = max(ly0, ey0 if py == ey0 else ey0 + half, py - half)
                    sx1 = min(lx1, ex1 if px == ex1 else ex1 - half, px + half)
                    sy1 = min(ly1, ey1 if py == ey1 else ey1 - half, py + half)
                    if sx0 <= sx1 and sy0 <= sy1:
                        outer.insert(db.Box(sx0, sy0, sx1 + 1, sy1 + 1))
                    outer.merge()
                    if outer.is_empty():
                        f_wire[gi] = db.Region()
                        continue
                else:
                    if lx0 > lx1 or ly0 > ly1:
                        f_wire[gi] = db.Region()
                        continue
                    outer = db.Region(db.Box(lx0, ly0, lx1 + 1, ly1 + 1))
                excl = db.Region()
                q = (
                    lx0 - half - sp - 5,
                    ly0 - half - sp - 5,
                    lx1 + half + sp + 5,
                    ly1 + half + sp + 5,
                )
                obs_idx = getattr(self, "_obs_idx", {}).get(gi)
                if obs_idx is not None:
                    for owner, rect in obs_idx.query(q):
                        if owner != net:
                            _insert_linf_violation_box(excl, db, rect, half, sp)
                for _owner, blocker, clearance in getattr(
                    self, "_mim_bottom_routing_clearance_by_gi", {}
                ).get(gi, ()):
                    _insert_linf_violation_box(excl, db, blocker, half, int(clearance))
                euclid_raw = db.Region() if use_exact_euclid else None
                for owner, rect in self._book.via_pads.get(gi, ()):
                    if owner != net:
                        if (
                            rect[2] >= q[0]
                            and rect[0] <= q[2]
                            and rect[3] >= q[1]
                            and rect[1] <= q[3]
                        ):
                            nr = _norm(rect)
                            if euclid_raw is not None and nr[0] < nr[2] and nr[1] < nr[3]:
                                euclid_raw.insert(db.Box(nr[0], nr[1], nr[2], nr[3]))
                            else:
                                insert_committed(excl, db, nr, half, sp)
                for owner, rect in self._book.metal.get(gi, ()):
                    if owner != net:
                        if (
                            rect[2] >= q[0]
                            and rect[0] <= q[2]
                            and rect[3] >= q[1]
                            and rect[1] <= q[3]
                        ):
                            enforce_sp = (
                                self._persistent_metal_requires_hard_space(
                                    gi, owner, rect
                                )
                                or half > base_hw
                                or (
                                    owner is not None
                                    and int(self._hw(str(owner), gi)) > base_hw
                                )
                            )
                            if enforce_sp:
                                nr = _norm(rect)
                                if euclid_raw is not None and nr[0] < nr[2] and nr[1] < nr[3]:
                                    euclid_raw.insert(db.Box(nr[0], nr[1], nr[2], nr[3]))
                                else:
                                    insert_committed(excl, db, nr, half, sp)
                            else:
                                _insert_linf_violation_box(excl, db, rect, half, 0)
                if euclid_raw is not None and not euclid_raw.is_empty():
                    for poly in euclid_raw.merged().each():
                        for trap in poly.decompose_trapezoids(db.Polygon.TD_vtrapezoids):
                            tb = trap.bbox()
                            insert_committed(
                                excl, db, (tb.left, tb.bottom, tb.right, tb.top), half, sp
                            )
                for wspace, core, _win, _attached in wide_map.get(gi, ()):
                    wq = (
                        lx0 - half - int(wspace) - 5,
                        ly0 - half - int(wspace) - 5,
                        lx1 + half + int(wspace) + 5,
                        ly1 + half + int(wspace) + 5,
                    )
                    for owner, rect in core.query(wq):
                        if owner != net:
                            _insert_linf_violation_box(excl, db, rect, half, int(wspace))
                free = outer - excl.merged()
                free.min_coherence = True
                free.merge()
                free.min_coherence = True
                f_wire[gi] = free

            f_via: Dict[int, Any] = {}
            for lo in ordered_layers:
                if (lo + 1) not in layer_set or lo not in self._via_pad_half:
                    continue
                bh, ah = (
                    self._via_pad_half1[lo]
                    if use_single_via
                    else self._via_pad_half[lo]
                )
                bh, ah = int(bh), int(ah)
                ext = int(self._via_cut_extent(lo, 1 if use_single_via else None))
                vsp = int(self._via_space[lo])
                max_h = max(bh, ah, ext)
                vx0, vy0, vx1, vy1 = bx0, by0, bx1, by1
                if env_bbox is not None:
                    vx0 = max(vx0, env_bbox[0] + max_h)
                    vy0 = max(vy0, env_bbox[1] + max_h)
                    vx1 = min(vx1, env_bbox[2] - max_h)
                    vy1 = min(vy1, env_bbox[3] - max_h)
                if local_iface is not None:
                    ex0, ey0, ex1, ey1 = (int(v) for v in local_iface[0])
                    vx0 = max(vx0, ex0 + max_h)
                    vy0 = max(vy0, ey0 + max_h)
                    vx1 = min(vx1, ex1 - max_h)
                    vy1 = min(vy1, ey1 - max_h)
                if vx0 > vx1 or vy0 > vy1:
                    f_via[lo] = db.Region()
                    continue
                v_outer = db.Region(db.Box(vx0, vy0, vx1 + 1, vy1 + 1))
                v_excl = db.Region()
                for gi, ph in ((lo, bh), (lo + 1, ah)):
                    sp = int(self._space[gi])
                    vq = (
                        vx0 - ph - sp - 5,
                        vy0 - ph - sp - 5,
                        vx1 + ph + sp + 5,
                        vy1 + ph + sp + 5,
                    )
                    obs_idx = getattr(self, "_obs_idx", {}).get(gi)
                    if obs_idx is not None:
                        for owner, rect in obs_idx.query(vq):
                            if owner != net:
                                _insert_linf_violation_box(v_excl, db, rect, ph, sp)
                    for _owner, blocker, clearance in getattr(
                        self, "_mim_bottom_routing_clearance_by_gi", {}
                    ).get(gi, ()):
                        _insert_linf_violation_box(v_excl, db, blocker, ph, int(clearance))
                    v_euclid_raw = db.Region() if use_exact_euclid else None
                    for rows in (
                        self._book.metal.get(gi, ()),
                        self._book.via_pads.get(gi, ()),
                    ):
                        for owner, rect in rows:
                            if owner != net:
                                if (
                                    rect[2] >= vq[0]
                                    and rect[0] <= vq[2]
                                    and rect[3] >= vq[1]
                                    and rect[1] <= vq[3]
                                ):
                                    nr = _norm(rect)
                                    if v_euclid_raw is not None and nr[0] < nr[2] and nr[1] < nr[3]:
                                        v_euclid_raw.insert(db.Box(nr[0], nr[1], nr[2], nr[3]))
                                    else:
                                        insert_committed(v_excl, db, nr, ph, sp)
                    if v_euclid_raw is not None and not v_euclid_raw.is_empty():
                        for poly in v_euclid_raw.merged().each():
                            for trap in poly.decompose_trapezoids(db.Polygon.TD_vtrapezoids):
                                tb = trap.bbox()
                                insert_committed(
                                    v_excl, db, (tb.left, tb.bottom, tb.right, tb.top), ph, sp
                                )
                    for wspace, core, _win, _attached in wide_map.get(gi, ()):
                        wq = (
                            vx0 - ph - int(wspace) - 5,
                            vy0 - ph - int(wspace) - 5,
                            vx1 + ph + int(wspace) + 5,
                            vy1 + ph + int(wspace) + 5,
                        )
                        for owner, rect in core.query(wq):
                            if owner != net:
                                _insert_linf_violation_box(
                                    v_excl, db, rect, ph, int(wspace)
                                )
                cq = (
                    vx0 - ext - vsp - 5,
                    vy0 - ext - vsp - 5,
                    vx1 + ext + vsp + 5,
                    vy1 + ext + vsp + 5,
                )
                cut_idx = self._cut_idx.get(lo)
                if cut_idx is not None:
                    for owner, rect in cut_idx.query(cq):
                        nr = _norm(rect)
                        if owner != net or (nr[2] - nr[0]) != 2 * ext or (nr[3] - nr[1]) != 2 * ext:
                            _insert_linf_violation_box(v_excl, db, nr, ext, vsp)
                for rect in getattr(self, "_attempt_cuts", {}).get(lo, ()):
                    nr = _norm(rect)
                    if (nr[2] - nr[0]) != 2 * ext or (nr[3] - nr[1]) != 2 * ext:
                        _insert_linf_violation_box(v_excl, db, nr, ext, vsp)
                for rect in self._local_fixed_cuts(lo, cq):
                    if (rect[2] - rect[0]) != 2 * ext or (rect[3] - rect[1]) != 2 * ext:
                        _insert_linf_violation_box(v_excl, db, rect, ext, vsp)
                raw_v = v_outer - v_excl.merged()
                raw_v.min_coherence = True
                raw_v.merge()
                raw_v.min_coherence = True
                f_via[lo] = raw_v

            r_pts = {gi: db.Region() for gi in ordered_layers}
            r_wire = {gi: db.Region() for gi in ordered_layers}
            for (sx, sy, sgi), _v in starts:
                if sgi in r_pts:
                    r_pts[sgi].insert(db.Box(sx, sy, sx + 1, sy + 1))
            if effective_source_regions and source_domain is not None:
                sgi = source_domain.node[2]
                if sgi in r_pts:
                    for _atom, (rx0, ry0, rx1, ry1) in effective_source_regions:
                        r_pts[sgi].insert(db.Box(rx0, ry0, rx1 + 1, ry1 + 1))
            for gi in ordered_layers:
                if not r_pts[gi].is_empty():
                    r_pts[gi].merge()
                    w = f_wire[gi].interacting(r_pts[gi])
                    w.min_coherence = True
                    r_wire[gi] = w

            g_reg = {gi: db.Region() for gi in ordered_layers}
            for gx, gy, ggi in goals:
                if ggi in g_reg:
                    g_reg[ggi].insert(db.Box(gx, gy, gx + 1, gy + 1))
            if effective_target_regions and target_domain is not None:
                tgi = target_domain.node[2]
                if tgi in g_reg:
                    for _atom, (rx0, ry0, rx1, ry1) in effective_target_regions:
                        g_reg[tgi].insert(db.Box(rx0, ry0, rx1 + 1, ry1 + 1))
            for gi in ordered_layers:
                if not g_reg[gi].is_empty():
                    g_reg[gi].merge()

            used = {
                (src_gi, dst_gi): db.Region()
                for lo in f_via
                for src_gi, dst_gi in ((lo, lo + 1), (lo + 1, lo))
            }
            while True:
                for gi in ordered_layers:
                    if g_reg[gi].is_empty():
                        continue
                    cur = r_wire[gi] + r_pts[gi]
                    if not cur.is_empty() and not (cur & g_reg[gi]).is_empty():
                        return True, f_wire, f_via, r_pts, r_wire, g_reg
                progressed = False
                for lo, f_v in f_via.items():
                    if f_v.is_empty():
                        continue
                    for src_gi, dst_gi in ((lo, lo + 1), (lo + 1, lo)):
                        src = r_wire[src_gi] + r_pts[src_gi]
                        if src.is_empty():
                            continue
                        v_cand = (f_v & src) - used[(src_gi, dst_gi)]
                        if v_cand.is_empty():
                            continue
                        v_cand.merge()
                        if grid > 1:
                            v_hit = db.Region()
                            for poly in v_cand.each():
                                if _polygon_has_mfg_grid_point(poly, grid):
                                    v_hit.insert(poly)
                            if v_hit.is_empty():
                                used[(src_gi, dst_gi)] = (
                                    used[(src_gi, dst_gi)] + v_cand
                                ).merged()
                                continue
                        else:
                            v_hit = v_cand
                        v_hit.min_coherence = True
                        used[(src_gi, dst_gi)] = (
                            used[(src_gi, dst_gi)] + v_cand
                        ).merged()
                        r_pts[dst_gi] = (r_pts[dst_gi] + v_hit).merged()
                        cand_w = f_wire[dst_gi] - r_wire[dst_gi]
                        if not cand_w.is_empty():
                            cand_w.min_coherence = True
                            nw = cand_w.interacting(v_hit)
                            if not nw.is_empty():
                                nw.min_coherence = True
                                r_wire[dst_gi] = (r_wire[dst_gi] + nw).merged()
                                r_wire[dst_gi].min_coherence = True
                        progressed = True
                if not progressed:
                    break
            return False, f_wire, f_via, r_pts, r_wire, g_reg

        # Fast-accept gate: 1-box L_inf exclusion is a strict superset of
        # Euclidean spacing, so reachability under L_inf proves reachability
        # under the exact hard rules without constructing circle staircases.
        linf_ok, *_ = _build_and_check(False)
        if linf_ok:
            return True, (time.perf_counter() - t0) * 1000.0

        exact_ok, F_wire, F_via, reached_pts, reached_wire, goal_reg = _build_and_check(True)
        if exact_ok:
            return True, (time.perf_counter() - t0) * 1000.0

        def _multilayer_reach(seed_pts: Dict[int, Any]) -> Dict[int, Any]:
            r_pts = {gi: seed_pts[gi].dup() for gi in ordered_layers}
            r_wire = {gi: db.Region() for gi in ordered_layers}
            for gi in ordered_layers:
                if not r_pts[gi].is_empty():
                    r_pts[gi].merge()
                    w = F_wire[gi].interacting(r_pts[gi])
                    w.min_coherence = True
                    r_wire[gi] = w
            u_via = {
                (s_gi, d_gi): db.Region()
                for lo in F_via
                for s_gi, d_gi in ((lo, lo + 1), (lo + 1, lo))
            }
            while True:
                prog = False
                for lo, f_v in F_via.items():
                    if f_v.is_empty():
                        continue
                    for s_gi, d_gi in ((lo, lo + 1), (lo + 1, lo)):
                        s_cur = r_wire[s_gi] + r_pts[s_gi]
                        if s_cur.is_empty():
                            continue
                        vc = (f_v & s_cur) - u_via[(s_gi, d_gi)]
                        if vc.is_empty():
                            continue
                        vc.merge()
                        if grid > 1:
                            vh = db.Region()
                            for poly in vc.each():
                                if _polygon_has_mfg_grid_point(poly, grid):
                                    vh.insert(poly)
                            if vh.is_empty():
                                u_via[(s_gi, d_gi)] = (
                                    u_via[(s_gi, d_gi)] + vc
                                ).merged()
                                continue
                        else:
                            vh = vc
                        vh.min_coherence = True
                        u_via[(s_gi, d_gi)] = (u_via[(s_gi, d_gi)] + vc).merged()
                        r_pts[d_gi] = (r_pts[d_gi] + vh).merged()
                        cw = F_wire[d_gi] - r_wire[d_gi]
                        if not cw.is_empty():
                            cw.min_coherence = True
                            nw = cw.interacting(vh)
                            if not nw.is_empty():
                                nw.min_coherence = True
                                r_wire[d_gi] = (r_wire[d_gi] + nw).merged()
                                r_wire[d_gi].min_coherence = True
                        prog = True
                if not prog:
                    break
            return {
                gi: (r_wire[gi] + r_pts[gi]).merged()
                for gi in ordered_layers
                if not (r_wire[gi] + r_pts[gi]).is_empty()
            }

        start_comp = {
            gi: (reached_wire[gi] + reached_pts[gi]).merged()
            for gi in ordered_layers
            if not (reached_wire[gi] + reached_pts[gi]).is_empty()
        }
        goal_comp = _multilayer_reach(goal_reg)
        start_area = sum(r.area() for r in start_comp.values())
        goal_area = sum(r.area() for r in goal_comp.values())
        ordered_comps = (
            [goal_comp, start_comp]
            if (0 < goal_area <= start_area or start_area == 0)
            else [start_comp, goal_comp]
        )

        def _has_committed_rejection() -> bool:
            return any(
                str(cat).startswith("committed_")
                for cat, _owner in getattr(self, "_hard_rejections", {}).keys()
            )

        grow_probe = max(grid, 2 * grid)
        for comp in ordered_comps:
            for gi, reg in comp.items():
                if reg.is_empty():
                    continue
                bb = reg.bbox()
                half = int(self._hw(net, gi))
                sp = int(self._space[gi])
                reg_grown = reg.sized(grow_probe)
                wq = (
                    bb.left - half - sp - grow_probe - 5,
                    bb.bottom - half - sp - grow_probe - 5,
                    bb.right + half + sp + grow_probe + 5,
                    bb.top + half + sp + grow_probe + 5,
                )
                base_hw = int(self._half_w[gi])
                by_owner_euclid: Dict[Any, Any] = {}
                by_owner_linf: Dict[Any, Any] = {}
                for owner, rect in self._book.via_pads.get(gi, ()):
                    if (
                        owner != net
                        and rect[2] >= wq[0]
                        and rect[0] <= wq[2]
                        and rect[3] >= wq[1]
                        and rect[1] <= wq[3]
                    ):
                        nr = _norm(rect)
                        if nr[0] < nr[2] and nr[1] < nr[3]:
                            by_owner_euclid.setdefault(owner, db.Region()).insert(
                                db.Box(nr[0], nr[1], nr[2], nr[3])
                            )
                        else:
                            _insert_exact_euclid_violation_boxes(
                                by_owner_linf.setdefault(owner, db.Region()),
                                db,
                                nr,
                                half,
                                sp,
                            )
                for owner, rect in self._book.metal.get(gi, ()):
                    if (
                        owner != net
                        and rect[2] >= wq[0]
                        and rect[0] <= wq[2]
                        and rect[3] >= wq[1]
                        and rect[1] <= wq[3]
                    ):
                        enforce_sp = (
                            self._persistent_metal_requires_hard_space(
                                gi, owner, rect
                            )
                            or half > base_hw
                            or (
                                owner is not None
                                and int(self._hw(str(owner), gi)) > base_hw
                            )
                        )
                        if enforce_sp:
                            nr = _norm(rect)
                            if nr[0] < nr[2] and nr[1] < nr[3]:
                                by_owner_euclid.setdefault(owner, db.Region()).insert(
                                    db.Box(nr[0], nr[1], nr[2], nr[3])
                                )
                            else:
                                _insert_exact_euclid_violation_boxes(
                                    by_owner_linf.setdefault(owner, db.Region()),
                                    db,
                                    nr,
                                    half,
                                    sp,
                                )
                        else:
                            _insert_linf_violation_box(
                                by_owner_linf.setdefault(owner, db.Region()),
                                db,
                                rect,
                                half,
                                0,
                            )
                all_owners = list(dict.fromkeys(
                    list(by_owner_euclid.keys()) + list(by_owner_linf.keys())
                ))
                for owner in all_owners:
                    v_reg = by_owner_linf.get(owner, db.Region())
                    raw_o = by_owner_euclid.get(owner)
                    if raw_o is not None and not raw_o.is_empty():
                        for poly in raw_o.merged().each():
                            for trap in poly.decompose_trapezoids(
                                db.Polygon.TD_vtrapezoids
                            ):
                                tb = trap.bbox()
                                _insert_exact_euclid_violation_boxes(
                                    v_reg,
                                    db,
                                    (tb.left, tb.bottom, tb.right, tb.top),
                                    half,
                                    sp,
                                )
                    hit = reg_grown & v_reg.merged()
                    if not hit.is_empty():
                        recorded_owner = False
                        for poly in hit.each():
                            for trap in poly.decompose_trapezoids(
                                db.Polygon.TD_vtrapezoids
                            ):
                                tb = trap.bbox()
                                mx = -(-tb.left // grid) * grid
                                my = -(-tb.bottom // grid) * grid
                                if mx < tb.right and my < tb.top:
                                    wb = (
                                        mx - half,
                                        my - half,
                                        mx + half,
                                        my + half,
                                    )
                                    self._reject_foreign_committed_metal_spacing(
                                        gi,
                                        wb,
                                        net,
                                        candidate_half=half,
                                        require_supra_minimum_endpoint=True,
                                    )
                                    self._reject_foreign_committed_metal_contact(
                                        gi,
                                        wb,
                                        net,
                                        candidate_half=half,
                                    )
                                    recorded_owner = True
                                    break
                            if recorded_owner:
                                break
                for lo in (gi, gi - 1):
                    if lo not in F_via:
                        continue
                    bh, ah = (
                        self._via_pad_half1[lo]
                        if use_single_via
                        else self._via_pad_half[lo]
                    )
                    bh, ah = int(bh), int(ah)
                    ext = int(self._via_cut_extent(lo, 1 if use_single_via else None))
                    vsp = int(self._via_space[lo])
                    cand_choke = db.Region()
                    for m_gi, ph in ((lo, bh), (lo + 1, ah)):
                        msp = int(self._space[m_gi])
                        mq = (
                            bb.left - ph - msp - 5,
                            bb.bottom - ph - msp - 5,
                            bb.right + ph + msp + 5,
                            bb.top + ph + msp + 5,
                        )
                        choke_raw = db.Region()
                        for rows in (
                            self._book.via_pads.get(m_gi, ()),
                            self._book.metal.get(m_gi, ()),
                        ):
                            for owner, rect in rows:
                                if (
                                    owner != net
                                    and rect[2] >= mq[0]
                                    and rect[0] <= mq[2]
                                    and rect[3] >= mq[1]
                                    and rect[1] <= mq[3]
                                ):
                                    nr = _norm(rect)
                                    if nr[0] < nr[2] and nr[1] < nr[3]:
                                        choke_raw.insert(db.Box(nr[0], nr[1], nr[2], nr[3]))
                                    else:
                                        _insert_exact_euclid_violation_boxes(
                                            cand_choke, db, nr, ph, msp
                                        )
                        if not choke_raw.is_empty():
                            for poly in choke_raw.merged().each():
                                for trap in poly.decompose_trapezoids(
                                    db.Polygon.TD_vtrapezoids
                                ):
                                    tb = trap.bbox()
                                    _insert_exact_euclid_violation_boxes(
                                        cand_choke,
                                        db,
                                        (tb.left, tb.bottom, tb.right, tb.top),
                                        ph,
                                        msp,
                                    )
                    cq = (
                        bb.left - ext - vsp - 5,
                        bb.bottom - ext - vsp - 5,
                        bb.right + ext - vsp + 5 + 2 * vsp,
                        bb.top + ext - vsp + 5 + 2 * vsp,
                    )
                    cut_idx = self._cut_idx.get(lo)
                    if cut_idx is not None:
                        for owner, rect in cut_idx.query(cq):
                            if owner != net:
                                _insert_linf_violation_box(
                                    cand_choke, db, rect, ext, vsp
                                )
                    choked = reg & cand_choke.merged()
                    if not choked.is_empty():
                        probed_v = 0
                        for poly in choked.each():
                            for trap in poly.decompose_trapezoids(
                                db.Polygon.TD_vtrapezoids
                            ):
                                tb = trap.bbox()
                                px = -(-tb.left // grid) * grid
                                py = -(-tb.bottom // grid) * grid
                                if px < tb.right and py < tb.top:
                                    if self._evaluate_via_static_legality(
                                        px, py, lo, net, 1 if use_single_via else None
                                    ):
                                        self._via_legal(
                                            px, py, lo, net, n=1 if use_single_via else None
                                        )
                                        probed_v += 1
                                        if probed_v >= 16:
                                            break
                            if probed_v >= 16:
                                break
                        if probed_v == 0:
                            for poly in choked.each():
                                for trap in poly.decompose_trapezoids(
                                    db.Polygon.TD_vtrapezoids
                                ):
                                    tb = trap.bbox()
                                    px = -(-tb.left // grid) * grid
                                    py = -(-tb.bottom // grid) * grid
                                    if px < tb.right and py < tb.top:
                                        for m_gi, ph in ((lo, bh), (lo + 1, ah)):
                                            pad = (px - ph, py - ph, px + ph, py + ph)
                                            self._reject_committed_via_pad_spacing(
                                                m_gi, pad, net, candidate_half=ph
                                            )
                                            self._reject_foreign_committed_metal_contact(
                                                m_gi, pad, net, candidate_half=ph
                                            )
                                        probed_v += 1
                                        if probed_v >= 8:
                                            break
                                if probed_v >= 8:
                                    break
            if _has_committed_rejection():
                break

        if not getattr(self, "_hard_rejections", None):
            for comp in ordered_comps:
                for gi, reg in comp.items():
                    if not reg.is_empty():
                        bb = reg.bbox()
                        mid_x = ((bb.left + bb.right - 1) // (2 * grid)) * grid
                        mid_y = ((bb.bottom + bb.top - 1) // (2 * grid)) * grid
                        if gi in F_via:
                            self._via_legal(
                                mid_x, mid_y, gi, net, n=1 if use_single_via else None
                            )
                        self._wire_legal(gi, mid_x, mid_y, mid_x + grid, mid_y, net)
                        break
                if getattr(self, "_hard_rejections", None):
                    break

        return False, (time.perf_counter() - t0) * 1000.0

    def _build_native_edge_context(
        self,
        net: str,
        trunk: int,
        neg_pending_hist: bool,
        hist_scale_num: int,
        ordered_layers: Sequence[int] = (),
        *,
        _allow_mirror: bool = True,
    ) -> Tuple[Any, Any, Any]:
        """Build a per-_route_edge C++20 NativeEdgeContext when all guards hold."""
        cpp = load_native("_gridless_cpp")
        if cpp is None:
            return None, None, None
        net_s = str(net)
        if (
            getattr(self, "_search_forbidden_wire_edges", ())
            or bool(getattr(self, "_active_group_route_local_closure", False))
            or getattr(self, "_group_route_global_interfaces_by_net", {}).get(net_s)
            or getattr(self, "_layered_portal_wire_states", ())
            or getattr(self, "_intent_spacing_rules", {}).get(net)
            or type(getattr(self, "_history", None)) is not dict
            or getattr(self, "_tech", None) is None
            or getattr(self, "_half_w", None) is None
            or getattr(self, "_space", None) is None
            or getattr(self, "_res_cell", None) is None
            or getattr(self, "_pres_fac", None) is None
            or getattr(self, "_via_pad_half", None) is None
            or getattr(self, "_via_space", None) is None
            or getattr(self, "_via_cut", None) is None
            or _eg_attr("_rects_clear", _base_rects_clear) is not _base_rects_clear
            or _eg_attr("_sep", _base_sep) is not _base_sep
            or _eg_attr("_closed_rects_touch", _base_closed_rects_touch) is not _base_closed_rects_touch
        ):
            return cpp, None, None
        inst_dict = self.__dict__
        for name in _EDGE_CTX_FORBIDDEN_OVERRIDES:
            if name in inst_dict:
                return cpp, None, None
        mirror_ctx = None
        mirror_axis2 = 0
        mirror_intent_space = 0
        mirror = getattr(self, "_active_mirror_legality", None)
        if mirror is not None and net_s == str(mirror.net_a):
            if (
                not _allow_mirror
                or "_reject_active_mirror_pair_box" in inst_dict
                or "_mirror_axis_twice_nm" in inst_dict
                or getattr(self, "_intent_spacing_rules", {}).get(str(mirror.net_b))
            ):
                return cpp, None, None
            _mc, mirror_ctx, _mh = self._build_native_edge_context(
                str(mirror.net_b),
                trunk,
                False,
                hist_scale_num,
                ordered_layers,
                _allow_mirror=False,
            )
            if mirror_ctx is None:
                return cpp, None, None
            mirror_axis2 = int(self._mirror_axis_twice_nm(mirror))
            mirror_intent_space = int(
                getattr(self, "_intent_spacing_rules", {})
                .get(str(mirror.net_a), {})
                .get(str(mirror.net_b), 0)
            )
        env = getattr(self, "_routing_envelope", None)
        if env is not None and type(env).__module__ != "clay.layout.route.floorplan_containment":
            return cpp, None, None

        n_layers = int(getattr(self, "_n_layers", 0))
        if n_layers <= 0 or len(self._half_w) < n_layers or len(self._space) < n_layers:
            return cpp, None, None
        allowed_set = set(ordered_layers) if ordered_layers else set(range(n_layers))
        obs_idx_map = getattr(self, "_obs_idx", None) or {}
        metal_idx_map = getattr(self, "_metal_idx", None) or {}
        att_idx_map = getattr(self, "_attempt_idx", None) or {}
        obs_cut_map = getattr(self, "_obs_cut_idx", None) or {}
        cut_idx_map = getattr(self, "_cut_idx", None) or {}
        for gi in range(n_layers):
            for idx in (obs_idx_map.get(gi), metal_idx_map.get(gi), att_idx_map.get(gi)):
                if idx is not None and (type(idx) is not _SpatialIndex or idx._native is None):
                    return cpp, None, None
        for lo in range(max(0, n_layers - 1)):
            for idx in (obs_cut_map.get(lo), cut_idx_map.get(lo)):
                if idx is not None and (type(idx) is not _SpatialIndex or idx._native is None):
                    return cpp, None, None

        self._ensure_fixed_wide_geometry_indexes()
        wide_by_gi = self._wide_indexes(True)
        min_wide_map = getattr(self, "_min_wide_ge_by_gi", {})

        hgrid = self._get_native_history_grid(cpp, force_fresh=True)
        if getattr(self, "_history", None) and hgrid is None:
            # Overflowed int64 SAT range -> stay on Python bigint history
            return cpp, None, None

        book = getattr(self, "_book", None)
        persistent_nets = getattr(book, "persistent", ()) if book is not None else ()
        pers_always_hard = bool(getattr(self, "_negotiation_port_legalize", False))
        net_half_w = getattr(self, "_net_half_w", {})
        via_pad_prov = getattr(self, "_via_pad_provenance", {})
        abstract_rects = getattr(self, "_abstract_rects", {})
        attempt_metal = getattr(self, "_attempt_metal", {})
        attempt_cuts = getattr(self, "_attempt_cuts", {})
        obs_cuts = getattr(self, "_obs_cuts", {})
        via_pad_half = self._via_pad_half
        via_pad_half1 = getattr(self, "_via_pad_half1", None) or via_pad_half
        via_space = self._via_space
        via_cut = self._via_cut

        mim_bot_map = getattr(self, "_mim_bottom_routing_clearance_by_gi", {})
        mim_der_active = self._mim_derived_unrelated_static_active()
        mim_der_map = (
            getattr(self, "_mim_derived_unrelated_clearance_by_gi", {})
            if mim_der_active
            else {}
        )
        mim_crv_map = getattr(self, "_mim_carve_bottom_blockers_by_gi", {})
        mim_fct_map = getattr(self, "_mim_forbidden_cut_touch_by_lo", {})
        mim_tvm_map = getattr(self, "_mim_top_via_marker_clearance_by_lo", {})

        declared_axis_rec = getattr(self, "_declared_trunk_axis_by_net", {}).get(net)
        dir_pref_map = getattr(self, "_direction_preference_by_gi", {})
        match_surcharge = getattr(self, "_active_match_layer_surcharge", None) or {}

        layer_specs = []
        for gi in range(n_layers):
            hw_i = int(self._hw(net, gi))
            min_hw_i = int(self._half_w[gi])
            sp_i = int(self._space[gi])
            base_rate = (
                _STEP_COST_PER_NM
                + (_M1_ROUTE_COST_PER_NM if gi == 0 else 0)
                + _TRUNK_BELOW_COST_PER_NM * self._effective_trunk_delta(net, gi, trunk)
                + int(match_surcharge.get(gi, 0))
            )
            decl_axis = (
                declared_axis_rec[1]
                if (
                    declared_axis_rec is not None
                    and gi == trunk == declared_axis_rec[0]
                )
                else None
            )
            dpref = dir_pref_map.get(gi)
            pref_h = (
                decl_axis == "H"
                if decl_axis is not None
                else (
                    dpref[0] == "H"
                    if dpref is not None
                    else bool(self._tech.preferred_horizontal(gi))
                )
            )
            off_w = (
                float(dpref[1])
                if decl_axis is None and dpref is not None
                else 1.0
            )
            oidx = obs_idx_map.get(gi)
            midx = metal_idx_map.get(gi)
            aidx = att_idx_map.get(gi)

            widened_owners = tuple(
                owner for owner, nw in net_half_w.items() if nw[gi] > min_hw_i
            ) if net_half_w else None
            pers_owners = (
                tuple((p_owner, pers_always_hard) for p_owner in persistent_nets)
                if persistent_nets
                else None
            )
            min_w = min_wide_map.get(gi)
            w_entries_raw = wide_by_gi.get(gi, ())
            w_entries_arg = None
            if w_entries_raw:
                w_list = []
                for w_sp, c_idx, _win_idx, a_idx in w_entries_raw:
                    if (
                        type(c_idx) is not _SpatialIndex
                        or c_idx._native is None
                        or type(a_idx) is not _SpatialIndex
                        or a_idx._native is None
                    ):
                        return cpp, None, None
                    w_list.append((int(w_sp), c_idx._native, a_idx._native))
                w_entries_arg = tuple(w_list)

            mb_raw = mim_bot_map.get(gi)
            mb_arg = (
                tuple((mo, _norm(mblk), int(mclr)) for mo, mblk, mclr in mb_raw)
                if mb_raw
                else None
            )
            md_raw = mim_der_map.get(gi)
            md_arg = (
                tuple(
                    (mo, _norm(mblk), int(mclr), mm == "euclidian", mrid)
                    for mo, mblk, mclr, mm, mrid in md_raw
                    if mo != net
                )
                if md_raw
                else None
            ) or None
            mc_raw = mim_crv_map.get(gi)
            mc_arg = (
                tuple((mo, _norm(mblk), mk) for mo, mblk, mk in mc_raw)
                if mc_raw
                else None
            )

            layer_specs.append((
                hw_i,
                min_hw_i,
                sp_i,
                int(base_rate),
                pref_h,
                off_w,
                oidx._native if oidx is not None else None,
                midx._native if midx is not None else None,
                aidx._native if aidx is not None else None,
                attempt_metal.get(gi) or None,
                abstract_rects.get(gi) or None,
                via_pad_prov.get(gi) or None,
                widened_owners or None,
                pers_owners,
                int(min_w) if min_w is not None else None,
                w_entries_arg,
                mb_arg,
                md_arg,
                mc_arg,
            ))

        forb_sites = getattr(self, "_search_forbidden_via_sites", None)
        forb_by_lo_0: Dict[int, List[Tuple[int, int]]] = {}
        forb_by_lo_1: Dict[int, List[Tuple[int, int]]] = {}
        if forb_sites:
            for fs in forb_sites:
                if len(fs) >= 4:
                    if fs[3] == 1:
                        forb_by_lo_1.setdefault(int(fs[2]), []).append((int(fs[0]), int(fs[1])))
                    else:
                        forb_by_lo_0.setdefault(int(fs[2]), []).append((int(fs[0]), int(fs[1])))

        via_specs = []
        for lo in range(max(0, n_layers - 1)):
            if lo not in via_pad_half or lo not in via_space or lo not in via_cut:
                if lo in allowed_set and (lo + 1) in allowed_set:
                    return cpp, None, None
                via_specs.append((
                    0, 0, 0, 0, 0, 0, 0,
                    None, None, None, None, None, None, None, None,
                ))
                continue
            bh0, ah0 = via_pad_half[lo]
            bh1, ah1 = via_pad_half1.get(lo, (bh0, ah0))
            ext0 = int(self._via_cut_extent(lo, None))
            ext1 = int(self._via_cut_extent(lo, 1))
            vsp_i = int(via_space[lo])
            oc_idx = obs_cut_map.get(lo)
            fc_fb = (
                None
                if oc_idx is not None
                else (tuple(_norm(r) for r in obs_cuts.get(lo, ())) or None)
            )
            c_idx = cut_idx_map.get(lo)
            mf_raw = mim_fct_map.get(lo)
            mf_arg = tuple((mo, _norm(mblk)) for mo, mblk in mf_raw) if mf_raw else None
            mt_raw = mim_tvm_map.get(lo)
            mt_arg = (
                tuple((mo, _norm(mblk), int(mclr)) for mo, mblk, mclr in mt_raw)
                if mt_raw
                else None
            )
            via_specs.append((
                int(bh0), int(ah0), ext0,
                int(bh1), int(ah1), ext1,
                vsp_i,
                oc_idx._native if oc_idx is not None else None,
                fc_fb,
                c_idx._native if c_idx is not None else None,
                attempt_cuts.get(lo) or None,
                mf_arg,
                mt_arg,
                forb_by_lo_0.get(lo) or None,
                forb_by_lo_1.get(lo) or None,
            ))

        rc_i = max(1, int(self._res_cell))
        pres_i = int(round(self._pres_fac * _COST_FACTOR_SCALE))
        mul_scale = rc_i * _STEP_COST_PER_NM * _CONG_WEIGHT
        single_ok = bool(self._single_via_escape_allowed())
        single_pen = _VIA_COST if self.via_redundancy > 1 else 0
        via_dirs = (
            int(self._via_successor_last_dir(0)),
            int(self._via_successor_last_dir(1)),
            int(self._via_successor_last_dir(2)),
        )
        try:
            edge_ctx = cpp.NativeEdgeContext(
                net=net,
                n_layers=n_layers,
                grid=max(1, int(getattr(self, "_manufacturing_grid_nm", 1))),
                rc=rc_i,
                pres=pres_i,
                mul_scale=mul_scale,
                cost_factor_scale=_COST_FACTOR_SCALE,
                turn_cost=int(self._turn_cost()),
                via_cost=int(_VIA_COST),
                step_cost=int(_STEP_COST_PER_NM),
                off_pref_cost=int(_OFF_PREF_COST_PER_NM),
                single_escape_ok=single_ok,
                single_via_penalty=int(single_pen),
                via_succ_dirs=via_dirs,
                env_bbox=tuple(int(v) for v in env.bbox_nm) if env is not None else None,
                hist_grid=hgrid,
                neg_pending_hist=bool(neg_pending_hist),
                hist_scale_num=int(hist_scale_num),
                layers=tuple(layer_specs),
                vias=tuple(via_specs),
                mirror_ctx=mirror_ctx,
                mirror_axis2=mirror_axis2,
                mirror_intent_space=mirror_intent_space,
            )
        except (OverflowError, TypeError, ValueError):
            return cpp, None, None
        return cpp, edge_ctx, hgrid

    def _route_edge(
        self,
        starts: Sequence[Tuple[Tuple[int, int, int], Optional[_ViaSite]]],
        goals: Dict[Tuple[int, int, int], Optional[_ViaSite]],
        net: str, xs: List[int], ys: List[int], lg: List[int],
        trunk: int = 0,
        *,
        hard_connectivity_witness: Optional[_HardConnectivityWitness] = None,
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """Multi-source / multi-target A*: EITHER endpoint may enter the graph at its
        pin node OR at a wireless escape landing (a fixed via that connects to the
        pin). ``starts`` / ``goals`` carry each state's optional pre/post via; the
        chosen path's entry + exit vias are prepended/appended to the emitted route."""
        prev_ctx = getattr(self, "_active_edge_ctx", None)
        prev_net = getattr(self, "_active_edge_ctx_net", None)
        prev_cpp = getattr(self, "_active_edge_ctx_cpp", None)
        prev_bumped = getattr(self, "_active_edge_ctx_gen_bumped", 0)
        prev_bypass = getattr(self, "_active_edge_ctx_bypass", False)
        prev_hgrid = getattr(self, "_active_history_grid", None)
        try:
            _advance_planar_region_step = _eg_attr("_advance_planar_region_step", _base_advance_planar_region_step)
            _region_via_goal_prefix_lower_bound = _eg_attr("_region_via_goal_prefix_lower_bound", _base_region_via_goal_prefix_lower_bound)
            _region_via_goal_waypoint = _eg_attr("_region_via_goal_waypoint", _base_region_via_goal_waypoint)
            _region_via_target_prunable = _eg_attr("_region_via_target_prunable", _base_region_via_target_prunable)
            array = _eg_attr("array", _base_array)
            bytearray = _eg_attr("bytearray", _base_bytearray)
            heapq = _eg_attr("heapq", _base_heapq)
            if hard_connectivity_witness is None:
                hard_connectivity_witness = getattr(
                    self, "_active_hard_connectivity_witness", None
                )
            self._selected_access_pair = None
            # LayerSet is a hard contract.  Entry/exit access states do not enlarge
            # the routable layer set: a wireless M2 escape cannot authorize M2 when
            # the net is explicitly capped to M1.
            allowed = set(lg)
            single_escape_ok = self._single_via_escape_allowed()
            def policy_access(via):
                return via is None or (
                    type(via.n) is int
                    and (
                        via.n in (0, self.via_redundancy)
                        or (via.n == 1 and single_escape_ok)
                    )
                )
            starts = [(s3, via) for (s3, via) in starts if s3[2] in allowed and policy_access(via)]
            goals = {s3: via for (s3, via) in goals.items() if s3[2] in allowed and policy_access(via)}
            if not starts or not goals:
                return None

            def via_key(via: Optional[_ViaSite]) -> Tuple[int, int, int, int, int]:
                if via is None:
                    return (0, 0, 0, 0, 0)
                return (1, via.x, via.y, via.lo, via.n)

            starts = sorted(starts, key=lambda item: (item[0], via_key(item[1])))
            goals = dict(sorted(
                goals.items(), key=lambda item: (item[0], via_key(item[1]))
            ))
            pattern_incumbent = getattr(
                self, "_active_pattern_search_incumbent", None
            )
            if not (
                isinstance(pattern_incumbent, _PatternSearchIncumbent)
                and pattern_incumbent.applies_to(starts, goals, allowed)
                and self._candidate_forbidden_wire(
                    net, pattern_incumbent.segs
                ) is None
            ):
                pattern_incumbent = None

            # The physical edge and all hard/soft costs are undirected.  Seed A*
            # from the smaller endpoint domain and put the larger component behind
            # the exact goal index, matching anaroute's access-component/spatial-
            # target split.  Otherwise a late fan-out edge can enqueue thousands of
            # already-connected conductor points before doing useful work (GF180
            # vout: 3185 starts versus one new terminal).  Preserve the caller's
            # start/goal orientation during reconstruction; this changes only search
            # direction, never endpoint ownership or emitted geometry semantics.
            endpoint_domains = getattr(self, "_active_endpoint_access_domains", (None, None))
            if not isinstance(endpoint_domains, tuple) or len(endpoint_domains) != 2:
                endpoint_domains = (None, None)
            def _planar_domain_rank(domain) -> int:
                if domain is None:
                    return 0
                if domain.singleton is None and domain.regions:
                    return 2
                if (
                    bool(getattr(self, "_negotiation_port_legalize", False))
                    and domain.locked_planar
                    and domain.singleton is not None
                    and domain.singleton[2] == domain.node[2]
                ):
                    return 1
                return 0

            source_domain, target_domain = endpoint_domains
            source_rank = _planar_domain_rank(source_domain)
            target_rank = _planar_domain_rank(target_domain)
            search_reversed = False
            prefer_reverse = pattern_incumbent is None and source_rank > target_rank
            if prefer_reverse or (
                pattern_incumbent is None
                and source_rank == target_rank
                and len(starts) > len(goals)
            ):
                reversed_goals = {state: via for state, via in starts}
                # A Mapping goal domain can carry only one access identity per state.
                # Do not collapse distinct same-state via alternatives just to reverse
                # the search; the ordinary complete transaction remains authoritative.
                if len(reversed_goals) == len(starts):
                    starts = list(goals.items())
                    goals = dict(sorted(
                        reversed_goals.items(),
                        key=lambda item: (item[0], via_key(item[1])),
                    ))
                    search_reversed = True
                    source_domain, target_domain = target_domain, source_domain
                    source_rank, target_rank = target_rank, source_rank

            self._precheck_envelope_disconnected = False
            start_coords = [s3 for (s3, _v) in starts]
            goal_coords = sorted(goals)
            xs = sorted(set(xs) | {c[0] for c in start_coords} | {c[0] for c in goal_coords})
            ys = sorted(set(ys) | {c[1] for c in start_coords} | {c[1] for c in goal_coords})
            ordered_layers = sorted(allowed)

            if self._solve_expansions_total >= self._solve_expansion_budget and pattern_incumbent is None:
                self._edge_fail = (
                    "solve_budget",
                    self._solve_expansions_total,
                    self._solve_expansion_budget,
                    len(xs),
                    len(ys),
                    len(allowed),
                )
                return None

            event_graph = None
            event_graphs = {}
            effective_target_regions = ()
            effective_source_regions = ()
            planar_target_active = (
                pattern_incumbent is None
                and target_rank > 0
                and target_domain.node[2] in allowed
            )
            if planar_target_active:
                effective_target_regions = (
                    target_domain.regions
                    if target_domain.regions
                    else ((None, (
                        target_domain.singleton[0],
                        target_domain.singleton[1],
                        target_domain.singleton[0],
                        target_domain.singleton[1],
                    )),)
                )
                if source_rank > 0 and source_domain.node[2] in allowed:
                    effective_source_regions = (
                        source_domain.regions
                        if source_domain.regions
                        else ((None, (
                            source_domain.singleton[0],
                            source_domain.singleton[1],
                            source_domain.singleton[0],
                            source_domain.singleton[1],
                        )),)
                    )

            precheck_passed = False
            if (
                planar_target_active
                and not getattr(self, "_active_mirror_legality", None)
                and (goals or effective_target_regions)
            ):
                from clay.layout.route.planar_precheck import fast_planar_via_precheck
                precheck_source_domain = (
                    source_domain
                    if planar_target_active and source_rank > 0 and source_domain.node[2] in allowed
                    else None
                )
                reachable, precheck_ms = fast_planar_via_precheck(
                    self,
                    starts,
                    goals,
                    net,
                    ordered_layers,
                    target_domain if planar_target_active else None,
                    effective_target_regions,
                    xs=xs,
                    ys=ys,
                    source_domain=precheck_source_domain,
                    effective_source_regions=effective_source_regions,
                )
                if reachable:
                    reachable, exact_ms = self._planar_exact_hard_precheck(
                        starts,
                        goals,
                        net,
                        ordered_layers,
                        target_domain if planar_target_active else None,
                        effective_target_regions,
                        xs=xs,
                        ys=ys,
                        source_domain=precheck_source_domain,
                        effective_source_regions=effective_source_regions,
                    )
                    precheck_ms += exact_ms
                if not reachable:
                    observation = getattr(self, "_access_static_disconnection_evidence", None)
                    if type(observation) is _AccessStaticDisconnectionRecorder:
                        observation.scope = "planar_region"
                        observation.elapsed_ms = round(precheck_ms, 3)
                    self._precheck_envelope_disconnected = bool(
                        not getattr(self, "_search_forbidden_via_sites", ())
                        and not getattr(self, "_search_forbidden_wire_edges", ())
                    )
                    self._edge_fail = (
                        "static_disconnected",
                        0,
                        0,
                        len(xs),
                        len(ys),
                        len(ordered_layers),
                        self._hard_rejection_evidence(),
                    )
                    details = dict(getattr(self, "_route_failure_details", ()))
                    details["precheck_pruned_ms"] = round(precheck_ms, 3)
                    self._route_failure_details = tuple(sorted(details.items()))
                    return None
                precheck_passed = True

            if planar_target_active:
                event_graph = self._planar_event_geometry(net, target_domain)
                if event_graph is None and (getattr(self, "_edge_fail", None) or (None,))[0] in {"budget", "solve_budget"}:
                    return None
                if event_graph is not None:
                    event_graphs[target_domain.node[2]] = event_graph
                    if source_rank > 0 and source_domain.node[2] in allowed:
                        if source_domain.node[2] not in event_graphs:
                            src_graph = self._planar_event_geometry(net, source_domain)
                            if src_graph is None and (getattr(self, "_edge_fail", None) or (None,))[0] in {"budget", "solve_budget"}:
                                return None
                            if src_graph is not None:
                                event_graphs[source_domain.node[2]] = src_graph
                else:
                    effective_target_regions = ()
                    effective_source_regions = ()

            def incumbent_result():
                assert pattern_incumbent is not None
                assert pattern_incumbent.start is not None
                assert pattern_incumbent.goal is not None
                self._edge_fail = None
                self._record_selected_access_pair(
                    pattern_incumbent.start, pattern_incumbent.goal
                )
                return list(pattern_incumbent.segs), list(pattern_incumbent.vias)

            graph = _CartesianSearchGraph(xs, ys, ordered_layers, self._max_gi)
            xi, yi = graph.xi, graph.yi
            goal_set = set(goal_coords)

            # DETERMINISTIC, GRID-SCALED expansion budget: sized to THIS edge's search
            # graph (|xs|*|ys|*|layers|*3 last_dir states) so a long-span edge -- a
            # cell-spanning power trunk, a far bias-pin net -- can search its OWN grid to
            # completion instead of dying on a one-size cap (the root cause of the flaky
            # gnd/net25 opens). Count-based -> a pure function of the inputs; NO wall-clock
            # inside the A* loop (that timing dependence was the non-determinism seed).
            local_budget = min(
                self.expansion_cap,
                max(
                    self.max_expansions,
                    len(xs) * len(ys) * max(1, len(allowed)) * 3,
                ),
            )
            solve_work_before_edge = self._solve_expansions_total
            remaining_solve_budget = max(
                0,
                self._solve_expansion_budget - self._solve_expansions_total,
            )
            if remaining_solve_budget == 0:
                if pattern_incumbent is not None:
                    return incumbent_result()
                self._edge_fail = (
                    "solve_budget",
                    self._solve_expansions_total,
                    self._solve_expansion_budget,
                    len(xs),
                    len(ys),
                    len(allowed),
                )
                return None
            budget = min(local_budget, remaining_solve_budget)
            event_cap = (
                min(
                    self.expansion_cap,
                    max(
                        local_budget * (16 if precheck_passed else 3),
                        6_400_000 if precheck_passed else 1_200_000,
                    ),
                )
                if self.expansion_cap > local_budget
                else local_budget
            )
            event_budget = min(event_cap, remaining_solve_budget)
            if pattern_incumbent is not None:
                # A closed-form route has already proved feasibility.  Residual A*
                # now owns only one deterministic opportunity to improve its SOFT
                # present/history price.  Size that opportunity by the exposed
                # access/axis interface, not by the Cartesian x*y*layer*direction
                # state volume: every endpoint alternative and candidate axis gets
                # one expansion of authority before the legal incumbent is retained.
                # The outer PathFinder still raises history and reranks all patterns
                # on later rounds, so keeping the incumbent cannot freeze one crossing.
                improvement_budget = max(
                    1,
                    len(start_coords)
                    + len(goal_coords)
                    + len(xs)
                    + len(ys)
                    + len(allowed),
                )
                budget = min(budget, improvement_budget)
            # per-corner charge, priced in routing pitches (``_turn_cost``); computed
            # once per edge so it is not re-derived per relaxation.
            turn_cost = self._turn_cost()

            potential_node_count = len(xs) * len(ys) * len(ordered_layers)
            graph_within_potential_cap = (
                potential_node_count * 3 <= self.expansion_cap
            )
            indexed_goal_domain = len(goal_coords) >= 4
            potential_mode = (
                "indexed_exact"
                if indexed_goal_domain
                else ("direct" if graph_within_potential_cap else "bbox")
            )
            geometry_mode = "direct"
            expansions = 0
            event_work = 0
            event_budget_stopped = False
            terminal_access_priced = 0

            def publish_watchdog_details() -> None:
                self._astar_watchdog_details = tuple(sorted((
                    ("expansions", int(expansions)),
                    ("geometry_mode", str(geometry_mode)),
                    ("goals", len(goal_coords)),
                    ("layers", len(ordered_layers)),
                    ("potential_mode", str(potential_mode)),
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
                    (
                        "potential_transform_switches_total",
                        int(self._potential_transform_switches_total),
                    ),
                    (
                        "potential_transform_cache_hits_total",
                        int(self._potential_transform_cache_hits_total),
                    ),
                    (
                        "potential_transform_cache_misses_total",
                        int(self._potential_transform_cache_misses_total),
                    ),
                    ("starts", len(start_coords)),
                    ("terminal_access_priced", int(terminal_access_priced)),
                    ("xs", len(xs)),
                    ("ys", len(ys)),
                )))

            publish_watchdog_details()

            def terminal_via_lower_bound(via: Optional[_ViaSite]) -> int:
                """Cheap admissible cost of one virtual terminal-access edge.

                Present/history congestion is non-negative and belongs to the exact
                edge evaluation when that access candidate reaches the frontier.  It
                must not be scanned eagerly for every point in a large conductor
                component before A* performs its first expansion.
                """
                if via is None:
                    return 0
                return _VIA_COST + self._functional_single_via_penalty(via)

            goal_terminal_lower_bound: Dict[Tuple[int, int, int], int] = {}
            for goal_index, (goal, via) in enumerate(goals.items()):
                if goal_index % 4096 == 0:
                    self._watchdog_check(net, "astar_potential")
                goal_terminal_lower_bound[goal] = terminal_via_lower_bound(via)

            layer_index = graph.layer_index
            potential_plane = graph.plane
            transform_work = (
                potential_node_count  # table initialization
                + len(xs) + len(ys) + len(ordered_layers)  # axis validation
                + 2 * len(goal_coords) + 1  # max, allocation check, seeds
                + 2 * len(ordered_layers) * len(ys) * max(0, len(xs) - 1)
                + 2 * len(ordered_layers) * len(xs) * max(0, len(ys) - 1)
                + 2 * len(ys) * len(xs) * max(
                    0, len(ordered_layers) - 1
                )
            )
            if indexed_goal_domain:
                # anaroute pattern: the target component is indexed once and every
                # source/relaxation asks for its nearest target.  Clay's fresh
                # implementation keeps the historical weighted-Manhattan value
                # exact (including per-goal terminal-via price), so this changes
                # only work complexity -- never A* ordering or emitted geometry.
                indexed_potential = _ExactGoalPotentialIndex(
                    goal_terminal_lower_bound,
                    watchdog=lambda: self._watchdog_check(
                        net, "astar_potential"
                    ),
                )
                indexed_potential_cache: Optional[Dict[int, int]] = {}
                indexed_potential_cache_limit = min(
                    potential_node_count,
                    max(0, int(self.max_expansions)),
                )
                indexed_query_work = 0
                indexed_transform_disabled = False
                exact_potential: List[int] = []

                def potential_value(x: int, y: int, gi: int) -> int:
                    return exact_potential[
                        layer_index[gi] * potential_plane
                        + yi[y] * len(xs)
                        + xi[x]
                    ]

                def hmin(x: int, y: int, gi: int) -> int:
                    nonlocal indexed_potential_cache
                    nonlocal indexed_query_work, indexed_transform_disabled
                    nonlocal exact_potential, potential_mode
                    if exact_potential:
                        return potential_value(x, y, gi)
                    node_index = (
                        layer_index[gi] * potential_plane
                        + yi[y] * len(xs)
                        + xi[x]
                    )
                    if indexed_potential_cache is not None:
                        cached = indexed_potential_cache.get(node_index)
                        if cached is not None:
                            return cached
                    work_before = indexed_potential.work
                    value = indexed_potential.value(x, y, gi)
                    indexed_query_work += indexed_potential.work - work_before
                    if (
                        graph_within_potential_cap
                        and not indexed_transform_disabled
                        and indexed_query_work * _EXACT_GOAL_INDEX_VISIT_WORK
                        > transform_work
                    ):
                        try:
                            candidate_potential = self._cached_exact_multigoal_potential(
                                xs,
                                ys,
                                ordered_layers,
                                goal_terminal_lower_bound,
                                net,
                            )
                        except MemoryError:
                            # The index remains exact and bounded.  A failed optional
                            # transform is never retried, so memory pressure cannot
                            # create an allocation loop or alter candidate authority.
                            indexed_transform_disabled = True
                            potential_mode = "indexed_exact_allocation_fallback"
                            publish_watchdog_details()
                        else:
                            exact_potential = candidate_potential
                            indexed_potential_cache = None
                            self._potential_transform_switches_total += 1
                            potential_mode = "exact_after_index"
                            publish_watchdog_details()
                            return potential_value(x, y, gi)
                    if (
                        indexed_potential_cache is not None
                        and len(indexed_potential_cache)
                        < indexed_potential_cache_limit
                    ):
                        try:
                            indexed_potential_cache[node_index] = value
                        except MemoryError:
                            # Query memoization is optional; the bounded goal index
                            # remains the exact source of truth.
                            indexed_potential_cache = None
                    return value
            elif graph_within_potential_cap:
                # Start with direct exact goal scans.  An adjacent easy edge then
                # pays only for the coordinates it actually queries, regardless of
                # the unused Cartesian grid size.  Once observed goal-comparison
                # work would exceed the deterministic table build, switch to the
                # bit-identical exact transform.  Because both forms return the
                # same integer potential, the switch cannot affect A* ordering.
                exact_potential: List[int] = []
                direct_goal_work = 0
                direct_potential_cache: Optional[Dict[int, int]] = None
                direct_potential_cache_limit = min(
                    potential_node_count,
                    max(0, int(self.max_expansions)),
                )

                def potential_value(x: int, y: int, gi: int) -> int:
                    return exact_potential[
                        layer_index[gi] * potential_plane
                        + yi[y] * len(xs)
                        + xi[x]
                    ]

                def hmin(x: int, y: int, gi: int) -> int:
                    nonlocal exact_potential, direct_goal_work, potential_mode
                    nonlocal direct_potential_cache
                    if exact_potential:
                        return potential_value(x, y, gi)
                    node_index = (
                        layer_index[gi] * potential_plane
                        + yi[y] * len(xs)
                        + xi[x]
                    )
                    if direct_potential_cache is not None:
                        cached = direct_potential_cache.get(node_index)
                        if cached is not None:
                            return cached
                    next_goal_work = direct_goal_work + len(goal_coords)
                    if next_goal_work > transform_work:
                        potential_mode = "exact"
                        publish_watchdog_details()
                        direct_potential_cache = None
                        exact_potential = self._cached_exact_multigoal_potential(
                            xs,
                            ys,
                            ordered_layers,
                            goal_terminal_lower_bound,
                            net,
                        )
                        return potential_value(x, y, gi)
                    self._watchdog_check(net, "astar_potential")
                    value = None
                    for goal_index, goal in enumerate(goal_coords):
                        if goal_index and goal_index % 4096 == 0:
                            self._watchdog_check(net, "astar_potential")
                        candidate = (
                            self._heuristic(x, y, gi, goal)
                            + goal_terminal_lower_bound[goal]
                        )
                        if value is None or candidate < value:
                            value = candidate
                    assert value is not None
                    direct_goal_work = next_goal_work
                    if (
                        direct_potential_cache is not None
                        and len(direct_potential_cache)
                        < direct_potential_cache_limit
                    ):
                        try:
                            direct_potential_cache[node_index] = value
                        except MemoryError:
                            # The memo is an acceleration only.  Release it and keep
                            # the exact direct/transform policy without retrying.
                            direct_potential_cache = None
                    return value
            else:
                # A sparse first-pass grid may deliberately exceed the materialized
                # graph cap and rely on the bounded A* expansion ceiling.  Do not
                # allocate an unbounded potential table in that case.  Distance to
                # the goal-domain bounding box plus the cheapest terminal is a
                # deterministic, consistent lower bound, so A* optimal stopping is
                # preserved without dropping a single access alternative.
                goal_x0 = min(goal[0] for goal in goal_coords)
                goal_x1 = max(goal[0] for goal in goal_coords)
                goal_y0 = min(goal[1] for goal in goal_coords)
                goal_y1 = max(goal[1] for goal in goal_coords)
                goal_l0 = min(goal[2] for goal in goal_coords)
                goal_l1 = max(goal[2] for goal in goal_coords)
                minimum_terminal_cost = min(goal_terminal_lower_bound.values())

                def interval_distance(value: int, lo: int, hi: int) -> int:
                    if value < lo:
                        return lo - value
                    if value > hi:
                        return value - hi
                    return 0

                def hmin(x: int, y: int, gi: int) -> int:
                    return (
                        (
                            interval_distance(x, goal_x0, goal_x1)
                            + interval_distance(y, goal_y0, goal_y1)
                        )
                        * _STEP_COST_PER_NM
                        + interval_distance(gi, goal_l0, goal_l1) * _VIA_COST
                        + minimum_terminal_cost
                    )

            if event_graph is not None:
                # Relax obstacles and directional/congestion penalties, but keep
                # mandatory layer-dependent wire costs and via distances.  Every
                # physical path visits a cheapest wire layer; its full Manhattan
                # length costs at least that layer's rate, and it must pay the vias
                # from its current layer through that layer to the target layer.
                # Minimize over all allowed layers and both finite and region goals.
                layer_rates = {
                    layer: (_STEP_COST_PER_NM
                            + (_M1_ROUTE_COST_PER_NM if layer == 0 else 0)
                            + _TRUNK_BELOW_COST_PER_NM * self._effective_trunk_delta(net, layer, trunk))
                    for layer in ordered_layers
                }
                layer_indexes = {}
                layer_potentials = {}
                for layer in ordered_layers:
                    projected = {}
                    for (gx, gy, goal_layer), terminal_cost in goal_terminal_lower_bound.items():
                        key = (gx, gy, 0)
                        cost = terminal_cost + abs(layer - goal_layer) * _VIA_COST
                        projected[key] = min(projected.get(key, cost), cost)
                    idx = _ExactGoalPotentialIndex(
                        projected, step_cost=layer_rates[layer],
                        watchdog=lambda: self._watchdog_check(net, "astar_potential"),
                    )
                    layer_indexes[layer] = idx
                    layer_potentials[layer] = _build_exact_2d_potential_fn(
                        projected, layer_rates[layer], idx
                    )
                layer_bound_cache = {}
                n_gi = max(max(ordered_layers, default=0), int(target_domain.node[2])) + 1
                target_gi = int(target_domain.node[2])
                ordered_layer_rates = tuple(
                    (layer, layer_rates[layer], layer_potentials[layer], abs(layer - target_gi) * _VIA_COST)
                    for layer in ordered_layers
                )
                target_boxes = tuple(box for _atom, box in effective_target_regions)
                single_target_box = target_boxes[0] if len(target_boxes) == 1 else None
                _inf_h = 1 << 60
                cpp_hmin_mod = load_native("_gridless_cpp")
                native_hmin = None
                if (
                    cpp_hmin_mod is not None
                    and target_boxes
                    and ordered_layer_rates
                    and all(
                        isinstance(pot_fn, cpp_hmin_mod.Native2DPotential)
                        for _layer, _rate, pot_fn, _via_tail in ordered_layer_rates
                    )
                ):
                    try:
                        native_hmin = cpp_hmin_mod.NativeLayerHmin(
                            ordered_layer_rates,
                            target_boxes,
                            _VIA_COST,
                            n_gi,
                            max(0, int(self.max_expansions)),
                        )
                    except (OverflowError, TypeError, ValueError):
                        native_hmin = None

                if native_hmin is not None:
                    hmin = native_hmin
                else:
                    def hmin(x: int, y: int, gi: int) -> int:
                        key = (x, y)
                        bounds = layer_bound_cache.get(key)
                        if bounds is None:
                            if single_target_box is not None:
                                tx0, ty0, tx1, ty1 = single_target_box
                                dx = tx0 - x if x < tx0 else (x - tx1 if x > tx1 else 0)
                                dy = ty0 - y if y < ty0 else (y - ty1 if y > ty1 else 0)
                                distance = dx + dy
                            else:
                                distance = min(
                                    max(x0 - x, 0, x - x1) + max(y0 - y, 0, y - y1)
                                    for x0, y0, x1, y1 in target_boxes
                                )
                            b = [_inf_h] * n_gi
                            for layer, rate, pot_fn, via_tail in ordered_layer_rates:
                                val = distance * rate + via_tail
                                idx_val = pot_fn(x, y)
                                b[layer] = idx_val if idx_val < val else val
                            for g in range(1, n_gi):
                                cand = b[g - 1] + _VIA_COST
                                if cand < b[g]:
                                    b[g] = cand
                            for g in range(n_gi - 2, -1, -1):
                                cand = b[g + 1] + _VIA_COST
                                if cand < b[g]:
                                    b[g] = cand
                            bounds = tuple(b)
                            if len(layer_bound_cache) < max(0, int(self.max_expansions)):
                                layer_bound_cache[key] = bounds
                        return bounds[gi]

            neg_pending_hist = bool(getattr(self, "_history", None)) and (
                bool(getattr(self, "_negotiation_port_legalize", False))
                or not bool(getattr(self, "_active_group_route_local_closure", False))
            )
            hw_by_gi = (
                {gi: int(self._hw(net, gi)) for gi in ordered_layers}
                if neg_pending_hist
                else {}
            )
            sp_half_by_gi = (
                {gi: int(self._space[gi]) // 2 for gi in ordered_layers}
                if neg_pending_hist
                else {}
            )
            rc_hist = max(1, int(getattr(self, "_res_cell", 1) or 1))
            hist_scale_num = rc_hist * _STEP_COST_PER_NM * _CONG_WEIGHT
            pending_hist_cache: Dict[Tuple[int, int, int, int, int], int] = {}

            _cpp_mod, _edge_ctx, _edge_hgrid = self._build_native_edge_context(
                net, trunk, neg_pending_hist, hist_scale_num, ordered_layers
            )
            if _edge_ctx is not None:
                self._active_edge_ctx = _edge_ctx
                self._active_edge_ctx_net = str(net)
                self._active_edge_ctx_cpp = _cpp_mod
                self._active_edge_ctx_gen_bumped = 0
                self._active_edge_ctx_bypass = False
            if _edge_hgrid is not None:
                self._active_history_grid = _edge_hgrid

            def pending_history_lb(gi: int, ax: int, ay: int, bx: int, by: int) -> int:
                if not neg_pending_hist or (ax == bx and ay == by):
                    return 0
                if _edge_ctx is not None:
                    c_val = _edge_ctx.pending_history_lb(gi, ax, ay, bx, by)
                    if c_val is not None:
                        return c_val
                pad = hw_by_gi[gi] + sp_half_by_gi[gi]
                x0, x1 = (ax, bx) if ax <= bx else (bx, ax)
                y0, y1 = (ay, by) if ay <= by else (by, ay)
                bx0, by0, bx1, by1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
                ckey = (
                    gi,
                    bx0 // rc_hist,
                    by0 // rc_hist,
                    (bx1 - 1) // rc_hist,
                    (by1 - 1) // rc_hist,
                )
                cached = pending_hist_cache.get(ckey)
                if cached is not None:
                    return cached
                units = self._history_congestion_units(gi, (bx0, by0, bx1, by1))
                val = (units * hist_scale_num) // _COST_FACTOR_SCALE if units else 0
                if len(pending_hist_cache) < 65536:
                    pending_hist_cache[ckey] = val
                return val

            _local_closure = bool(getattr(self, "_active_group_route_local_closure", False))
            _match_surcharge = getattr(self, "_active_match_layer_surcharge", None) or {}
            unit_cost_by_gi = {
                gi: (
                    _STEP_COST_PER_NM
                    + (_M1_ROUTE_COST_PER_NM if gi == 0 else 0)
                    + _TRUNK_BELOW_COST_PER_NM * self._effective_trunk_delta(net, gi, trunk)
                    + (0 if _local_closure else int(_match_surcharge.get(gi, 0)))
                )
                for gi in ordered_layers
            }

            def state_hmin(state) -> int:
                if len(state) <= 4:
                    return hmin(*state[:3])
                gi = state[2]
                unit_cost = unit_cost_by_gi[gi]
                pending_cost = (
                    (abs(state[0] - state[7]) + abs(state[1] - state[8])) * unit_cost
                    + pending_history_lb(gi, state[7], state[8], state[0], state[1])
                    if state[9] else 0
                )
                if len(state) == 10 and state[6] < -1 and target_domain is not None:
                    v_other = -state[6] - 2
                    if state[5]:
                        end_x = state[5][-1][0] * target_domain.grid
                        end_y = state[5][-1][1] * target_domain.grid
                        rem_cost = (abs(end_x - state[0]) + abs(end_y - state[1])) * unit_cost
                    else:
                        end_x, end_y, rem_cost = state[0], state[1], 0
                    end3 = (end_x, end_y, gi)
                    if not (
                        end3 in goal_set
                        or (gi == target_domain.node[2] and bool(target_domain.owning_atoms(end3)))
                    ):
                        return pending_cost + rem_cost + _VIA_COST + hmin(end_x, end_y, v_other)
                return hmin(*state[:3]) + pending_cost

            # Static geometry/cost does not depend on A*'s incoming ``last_dir``.
            # A hard search can expand horizontal and vertical arrival states at the
            # same coordinate, historically repeating every spatial-index query up
            # to four times per undirected edge.  The hard-connectivity preflight is
            # already required to judge these exact physical edges; every non-trivial
            # within-cap graph populates bounded signed-64 storage for edge ids,
            # non-negative priced cost / -1 immutable-illegal / -2 hard-legal but
            # unpriced, and three committed-present lanes. A* consumes it
            # direct-mapped over undirected H/V/via ids.
            # Slots reuse ``max_expansions`` as their existing work authority. Keys,
            # costs, and three present-scan lanes top out near 16 MB by default rather
            # than a full graph-sized Python cache.  Preflight shares exact hard
            # geometry and its committed-present scan with A*, but leaves the
            # objective/present/history price to edges A* actually requests;
            # collisions only recompute/overwrite and cannot change search.
            # The memo is allowed only when this graph already fits the existing
            # expansion cap; an over-cap sparse search remains direct and bounded.
            horizontal_edge_count = graph.horizontal_edges
            vertical_edge_count = graph.vertical_edges
            via_pair_index = graph.via_pair_index
            via_edge_count = len(via_pair_index) * potential_plane
            packed_edge_count = (
                horizontal_edge_count + vertical_edge_count + via_edge_count
            )
            packed_slot_count = min(
                packed_edge_count,
                max(0, int(self.max_expansions)),
            )
            static_geometry_work = 0
            packed_geometry_disabled = False
            packed_edge_keys = None
            packed_edge_costs = None
            packed_edge_present = None
            preflight_tested_edges = None
            packed_unknown = -1
            packed_illegal_cost = -1
            packed_legal_unpriced_cost = -2
            packed_present_unknown = -1
            packed_cost_max = (1 << 63) - 1
            static_geometry_phase = "astar_geometry"
            via_model_by_edge: Dict[int, int] = {}

            def enable_packed_geometry(*, force: bool = False) -> None:
                nonlocal geometry_mode
                nonlocal packed_geometry_disabled
                nonlocal packed_edge_keys, packed_edge_costs, packed_edge_present
                nonlocal preflight_tested_edges
                if (
                    packed_edge_keys is not None
                    or packed_geometry_disabled
                    or not graph_within_potential_cap
                    or packed_slot_count == 0
                    or (
                        not force
                        and static_geometry_work <= packed_slot_count
                    )
                ):
                    return
                self._watchdog_check(net, static_geometry_phase)
                new_keys = None
                new_costs = None
                new_present = None
                try:
                    new_keys = array("q", [packed_unknown]) * packed_slot_count
                    self._watchdog_check(net, static_geometry_phase)
                    new_costs = array("q", [0]) * packed_slot_count
                    self._watchdog_check(net, static_geometry_phase)
                    new_present = (
                        array("q", [packed_present_unknown])
                        * (packed_slot_count * 3)
                    )
                except MemoryError:
                    # Allocation is atomic with respect to search semantics: discard
                    # every partial array and remain on the exact direct predicates
                    # for this edge.  Never retry, so memory pressure cannot spin.
                    new_keys = None
                    new_costs = None
                    new_present = None
                    packed_geometry_disabled = True
                    preflight_tested_edges = None
                    geometry_mode = "direct_allocation_fallback"
                    publish_watchdog_details()
                    return
                packed_edge_keys = new_keys
                packed_edge_costs = new_costs
                packed_edge_present = new_present
                preflight_tested_edges = None
                geometry_mode = "packed"
                publish_watchdog_details()

            planar_edge_id = graph.planar_edge_id

            def store_packed_present(index: int, value: int) -> None:
                """Store only non-negative totals representable by signed-64."""
                numeric = int(value)
                if 0 <= numeric <= packed_cost_max:
                    packed_edge_present[index] = numeric

            def planar_static_cost(
                gi: int,
                ax: int,
                ay: int,
                bx: int,
                by: int,
            ) -> Optional[int]:
                nonlocal static_geometry_work
                edge_id = graph.planar_cache_key(gi, ax, ay, bx, by)
                static_geometry_work += 1
                if static_geometry_work % 4096 == 0:
                    self._watchdog_check(net, static_geometry_phase)
                enable_packed_geometry(force=(
                    edge_id is not None and preflight_tested_edges is not None
                    and bool(preflight_tested_edges[edge_id])
                ))
                slot = -1
                hard_legal = False
                if packed_edge_keys is not None and edge_id is not None:
                    slot = edge_id % packed_slot_count
                    if packed_edge_keys[slot] == edge_id:
                        cached = int(packed_edge_costs[slot])
                        if cached == packed_illegal_cost:
                            return None
                        if cached == packed_legal_unpriced_cost:
                            hard_legal = True
                        else:
                            return cached
                previous_scans = self._active_static_metal_scans
                active_scans = {}
                if hard_legal and packed_edge_present is not None:
                    present = int(packed_edge_present[slot * 3])
                    if present != packed_present_unknown:
                        box = _seg_bbox(
                            ax, ay, bx, by, self._hw(net, gi)
                        )
                        active_scans[(int(gi), tuple(_norm(box)), str(net))] = (
                            present,
                            (),
                        )
                # Lazy: fixed artwork is judged before the first committed-metal
                # lookup; legality then fills this one-transition cache and price
                # reuses it.
                self._active_static_metal_scans = active_scans
                try:
                    if not hard_legal:
                        rejection_generation_before = getattr(
                            self, "_hard_rejection_generation", 0
                        )
                        if not self._wire_legal(gi, ax, ay, bx, by, net):
                            # Cache only an evidence-backed immutable geometry
                            # rejection. A test/instrumentation override may return
                            # False without the engine's hard-rejection witness and
                            # is not safe to memoize.
                            if (
                                packed_edge_keys is not None and edge_id is not None
                                and getattr(self, "_hard_rejection_generation", 0)
                                > rejection_generation_before
                            ):
                                packed_edge_keys[slot] = int(edge_id)
                                packed_edge_costs[slot] = packed_illegal_cost
                            return None
                    cost = (
                        self._edge_objective(net, gi, ax, ay, bx, by, trunk)
                        + self._edge_congestion(gi, ax, ay, bx, by, net)
                    )
                finally:
                    self._active_static_metal_scans = previous_scans
                if packed_edge_keys is not None and edge_id is not None and 0 <= cost <= packed_cost_max:
                    packed_edge_keys[slot] = int(edge_id)
                    packed_edge_costs[slot] = int(cost)
                return cost

            def planar_hard_legal(
                gi: int,
                ax: int,
                ay: int,
                bx: int,
                by: int,
            ) -> bool:
                """Exact topology predicate without A*'s deferred soft price."""
                nonlocal static_geometry_work
                edge_id = graph.planar_cache_key(gi, ax, ay, bx, by)
                static_geometry_work += 1
                if static_geometry_work % 4096 == 0:
                    self._watchdog_check(net, static_geometry_phase)
                enable_packed_geometry(force=(
                    edge_id is not None and preflight_tested_edges is not None
                    and bool(preflight_tested_edges[edge_id])
                ))
                slot = -1
                if packed_edge_keys is not None and edge_id is not None:
                    slot = edge_id % packed_slot_count
                    if packed_edge_keys[slot] == edge_id:
                        return int(packed_edge_costs[slot]) != packed_illegal_cost
                previous_scans = self._active_static_metal_scans
                active_scans = {}
                self._active_static_metal_scans = active_scans
                try:
                    rejection_generation_before = getattr(
                        self, "_hard_rejection_generation", 0
                    )
                    legal = self._wire_legal(gi, ax, ay, bx, by, net)
                finally:
                    self._active_static_metal_scans = previous_scans
                if packed_edge_keys is not None and edge_id is not None and (
                    legal
                    or getattr(self, "_hard_rejection_generation", 0)
                    > rejection_generation_before
                ):
                    packed_edge_keys[slot] = int(edge_id)
                    packed_edge_costs[slot] = (
                        packed_legal_unpriced_cost
                        if legal
                        else packed_illegal_cost
                    )
                    present_slot = slot * 3
                    for offset in range(3):
                        packed_edge_present[present_slot + offset] = (
                            packed_present_unknown
                        )
                    if legal:
                        box = _seg_bbox(
                            ax, ay, bx, by, self._hw(net, gi)
                        )
                        scan = active_scans.get((
                            int(gi), tuple(_norm(box)), str(net)
                        ))
                        if scan is not None:
                            store_packed_present(present_slot, scan[0])
                return legal

            def via_static_cost(x: int, y: int, lo: int) -> Optional[int]:
                nonlocal static_geometry_work
                edge_id = via_edge_id(x, y, lo)
                static_geometry_work += 1
                if static_geometry_work % 4096 == 0:
                    self._watchdog_check(net, static_geometry_phase)
                enable_packed_geometry(force=(
                    preflight_tested_edges is not None
                    and bool(preflight_tested_edges[edge_id])
                ))
                slot = -1
                hard_legal = False
                selected_model_n = 0
                if packed_edge_keys is not None:
                    slot = edge_id % packed_slot_count
                    if packed_edge_keys[slot] == edge_id:
                        cached = int(packed_edge_costs[slot])
                        if cached == packed_illegal_cost:
                            return None
                        if cached == packed_legal_unpriced_cost:
                            hard_legal = True
                            selected_model_n = via_model_by_edge.get(edge_id, 0)
                        else:
                            return cached
                previous_metal_scans = self._active_static_metal_scans
                previous_cut_scans = self._active_static_cut_scans
                active_metal_scans = {}
                active_cut_scans = {}
                if hard_legal and packed_edge_present is not None:
                    present_slot = slot * 3
                    bh, ah = (
                        self._via_pad_half1[lo]
                        if selected_model_n == 1
                        else self._via_pad_half[lo]
                    )
                    metal_specs = (
                        (lo, bh, int(packed_edge_present[present_slot])),
                        (
                            lo + 1,
                            ah,
                            int(packed_edge_present[present_slot + 1]),
                        ),
                    )
                    for layer, half, present in metal_specs:
                        if present == packed_present_unknown:
                            continue
                        pad = (x - half, y - half, x + half, y + half)
                        active_metal_scans[(
                            int(layer), tuple(_norm(pad)), str(net)
                        )] = (present, ())
                    cut_present = int(packed_edge_present[present_slot + 2])
                    if cut_present != packed_present_unknown:
                        ext = self._via_cut_extent(
                            lo, 1 if selected_model_n == 1 else None
                        )
                        cut = (x - ext, y - ext, x + ext, y + ext)
                        active_cut_scans[(
                            int(lo), tuple(_norm(cut)), str(net)
                        )] = (cut_present, (), ())
                self._active_static_metal_scans = active_metal_scans
                self._active_static_cut_scans = active_cut_scans
                try:
                    if not hard_legal:
                        rejection_generation_before = getattr(
                            self, "_hard_rejection_generation", 0
                        )
                        default_legal = self._via_legal(x, y, lo, net)
                        selected_model_n = 0
                        if not default_legal:
                            single_legal = (
                                single_escape_ok
                                and self._via_legal(x, y, lo, net, n=1)
                            )
                            if single_legal:
                                selected_model_n = 1
                            else:
                                if (
                                    packed_edge_keys is not None
                                    and getattr(self, "_hard_rejection_generation", 0)
                                    > rejection_generation_before
                                ):
                                    packed_edge_keys[slot] = int(edge_id)
                                    packed_edge_costs[slot] = packed_illegal_cost
                                return None
                    model_n = selected_model_n
                    if model_n == 1:
                        single = _ViaSite(x, y, lo, 1)
                        cost = (
                            _VIA_COST
                            + self._functional_single_via_penalty(single)
                            + self._via_congestion(x, y, lo, net, 1)
                        )
                    else:
                        default_congestion = self._via_congestion(x, y, lo, net)
                        cost = _VIA_COST + default_congestion
                        if single_escape_ok:
                            single = _ViaSite(x, y, lo, 1)
                            single_penalty = self._functional_single_via_penalty(single)
                            if (
                                default_congestion > single_penalty
                                and self._via_legal(x, y, lo, net, n=1)
                            ):
                                single_cost = (
                                    _VIA_COST
                                    + single_penalty
                                    + self._via_congestion(x, y, lo, net, n=1)
                                )
                                if single_cost < cost:
                                    cost = single_cost
                                    model_n = 1
                finally:
                    self._active_static_metal_scans = previous_metal_scans
                    self._active_static_cut_scans = previous_cut_scans
                via_model_by_edge[edge_id] = model_n
                if packed_edge_keys is not None and 0 <= cost <= packed_cost_max:
                    packed_edge_keys[slot] = int(edge_id)
                    packed_edge_costs[slot] = int(cost)
                return cost

            def via_hard_legal(x: int, y: int, lo: int) -> bool:
                """Exact via topology predicate without A*'s deferred soft price."""
                nonlocal static_geometry_work
                edge_id = via_edge_id(x, y, lo)
                static_geometry_work += 1
                if static_geometry_work % 4096 == 0:
                    self._watchdog_check(net, static_geometry_phase)
                enable_packed_geometry(force=(
                    preflight_tested_edges is not None
                    and bool(preflight_tested_edges[edge_id])
                ))
                slot = -1
                if packed_edge_keys is not None:
                    slot = edge_id % packed_slot_count
                    if packed_edge_keys[slot] == edge_id:
                        return int(packed_edge_costs[slot]) != packed_illegal_cost
                previous_metal_scans = self._active_static_metal_scans
                previous_cut_scans = self._active_static_cut_scans
                active_metal_scans = {}
                active_cut_scans = {}
                self._active_static_metal_scans = active_metal_scans
                self._active_static_cut_scans = active_cut_scans
                try:
                    rejection_generation_before = getattr(
                        self, "_hard_rejection_generation", 0
                    )
                    legal = self._via_legal(x, y, lo, net)
                    model_n = 0
                    if not legal and single_escape_ok:
                        legal = self._via_legal(x, y, lo, net, n=1)
                        if legal:
                            model_n = 1
                finally:
                    self._active_static_metal_scans = previous_metal_scans
                    self._active_static_cut_scans = previous_cut_scans
                if legal:
                    via_model_by_edge[edge_id] = model_n
                if packed_edge_keys is not None and (
                    legal
                    or getattr(self, "_hard_rejection_generation", 0)
                    > rejection_generation_before
                ):
                    packed_edge_keys[slot] = int(edge_id)
                    packed_edge_costs[slot] = (
                        packed_legal_unpriced_cost
                        if legal
                        else packed_illegal_cost
                    )
                    present_slot = slot * 3
                    for offset in range(3):
                        packed_edge_present[present_slot + offset] = (
                            packed_present_unknown
                        )
                    if legal:
                        bh, ah = (
                            self._via_pad_half1[lo]
                            if model_n == 1
                            else self._via_pad_half[lo]
                        )
                        for offset, (layer, half) in enumerate((
                            (lo, bh),
                            (lo + 1, ah),
                        )):
                            pad = (x - half, y - half, x + half, y + half)
                            scan = active_metal_scans.get((
                                int(layer), tuple(_norm(pad)), str(net)
                            ))
                            if scan is not None:
                                store_packed_present(
                                    present_slot + offset,
                                    scan[0],
                                )
                        ext = self._via_cut_extent(
                            lo, 1 if model_n == 1 else None
                        )
                        cut = (x - ext, y - ext, x + ext, y + ext)
                        cut_scan = active_cut_scans.get((
                            int(lo), tuple(_norm(cut)), str(net)
                        ))
                        if cut_scan is not None:
                            store_packed_present(
                                present_slot + 2,
                                cut_scan[0],
                            )
                return legal

            via_edge_id = graph.via_edge_id
            planar = graph.planar
            adjacent_layers = graph.adjacent_layers

            # A turn-aware state multiplies a physically sealed graph by up to three
            # incoming-direction states before it can prove that no path exists.  First
            # decide only the hard topological question on the exact same candidate
            # nodes and the exact same hard-geometry predicates.  Legal/illegal results
            # populate the existing exact memo, but objective/history/present prices
            # are deferred until A* requests a legal edge. Thus soft congestion cannot
            # become hard legality, while a disconnected graph consumes neither price
            # work nor solve-expansion authority.
            connectivity_nodes = 0
            connectivity_edges = 0

            def publish_connectivity_watchdog_details() -> None:
                self._astar_watchdog_details = tuple(sorted((
                    ("connectivity_edges", int(connectivity_edges)),
                    ("connectivity_nodes", int(connectivity_nodes)),
                    ("expansions", int(expansions)),
                    ("geometry_mode", str(geometry_mode)),
                    ("goals", len(goal_coords)),
                    ("layers", len(ordered_layers)),
                    ("potential_mode", str(potential_mode)),
                    (
                        "potential_transform_switches_total",
                        int(self._potential_transform_switches_total),
                    ),
                    ("starts", len(start_coords)),
                    ("xs", len(xs)),
                    ("ys", len(ys)),
                )))

            physical_node_id = graph.physical_node_id

            def hard_connectivity_preflight() -> Optional[bool]:
                nonlocal connectivity_nodes, connectivity_edges
                nonlocal preflight_tested_edges
                nonlocal static_geometry_phase
                unrepresented_edges = False
                try:
                    reached = bytearray(potential_node_count)
                    tested_edges = bytearray(packed_edge_count)
                except MemoryError:
                    # This proof is an optimization, never a new routing
                    # prerequisite.  Drop either partial dense allocation, restore
                    # the ordinary A* diagnostic surface, and execute the original
                    # bounded search exactly once without retrying the preflight.
                    static_geometry_phase = "astar_geometry"
                    publish_watchdog_details()
                    return True
                start_node_ids = {
                    physical_node_id(s3[0], s3[1], s3[2])
                    for s3, _via in starts
                }
                goal_node_ids = {
                    physical_node_id(goal[0], goal[1], goal[2])
                    for goal in goal_coords
                }
                if len(start_node_ids) <= len(goal_node_ids):
                    seed_node_ids = start_node_ids
                    target_node_ids = goal_node_ids
                else:
                    seed_node_ids = goal_node_ids
                    target_node_ids = start_node_ids
                frontier = sorted(seed_node_ids)
                for node_id in frontier:
                    reached[node_id] = 1

                static_geometry_phase = "astar_connectivity"
                publish_connectivity_watchdog_details()
                self._watchdog_check(net, "astar_connectivity")
                hard_connected = any(
                    node_id in target_node_ids for node_id in frontier
                )
                if not hard_connected:
                    # A non-trivial connected solve would allocate this same bounded
                    # memo on A*'s first repeated edge.  Allocate before preflight so
                    # its exact judgments fill the cache.  start==goal remains
                    # allocation-free; MemoryError retains the direct path.
                    enable_packed_geometry(force=True)
                frontier_index = 0
                while not hard_connected and frontier_index < len(frontier):
                    if connectivity_nodes and connectivity_nodes % 1024 == 0:
                        publish_connectivity_watchdog_details()
                        self._watchdog_check(net, "astar_connectivity")
                    node_id = frontier[frontier_index]
                    frontier_index += 1
                    layer_pos, plane_offset = divmod(node_id, potential_plane)
                    y_pos, x_pos = divmod(plane_offset, len(xs))
                    ux = xs[x_pos]
                    uy = ys[y_pos]
                    ul = ordered_layers[layer_pos]
                    connectivity_nodes += 1

                    for vx, vy, _vdir in planar(ux, uy):
                        if not graph.embeds_planar_edge(ul, ux, uy, vx, vy):
                            # This flood can only prove facts about the fixed-axis
                            # graph. Never alias another graph's edge to its packed
                            # ID, or claim disconnection when that graph was omitted.
                            unrepresented_edges = True
                            continue
                        edge_id = planar_edge_id(ul, ux, uy, vx, vy)
                        if tested_edges[edge_id]:
                            continue
                        tested_edges[edge_id] = 1
                        connectivity_edges += 1
                        if not planar_hard_legal(ul, ux, uy, vx, vy):
                            continue
                        target_id = physical_node_id(vx, vy, ul)
                        if not reached[target_id]:
                            reached[target_id] = 1
                            frontier.append(target_id)
                            if target_id in target_node_ids:
                                hard_connected = True
                                break
                    if hard_connected:
                        break

                    for other in adjacent_layers(ul):
                        lo = min(ul, other)
                        edge_id = via_edge_id(ux, uy, lo)
                        if tested_edges[edge_id]:
                            continue
                        tested_edges[edge_id] = 1
                        connectivity_edges += 1
                        if not via_hard_legal(ux, uy, lo):
                            continue
                        target_id = physical_node_id(ux, uy, other)
                        if not reached[target_id]:
                            reached[target_id] = 1
                            frontier.append(target_id)
                            if target_id in target_node_ids:
                                hard_connected = True
                                break

                publish_connectivity_watchdog_details()
                self._watchdog_check(net, "astar_connectivity")
                static_geometry_phase = "astar_geometry"
                if (
                    hard_connected
                    and packed_edge_keys is None
                    and not packed_geometry_disabled
                ):
                    preflight_tested_edges = tested_edges
                return None if not hard_connected and unrepresented_edges else hard_connected

            hard_connectivity_proven = bool(
                hard_connectivity_witness is not None
                and not getattr(self, "_search_forbidden_wire_edges", ())
                and hard_connectivity_witness.connects_graph(
                    start_coords, goal_set, xs, ys, allowed
                )
            )
            # Large terminal components already have an exact spatial goal index, and
            # a refined graph larger than the declared nominal A* work is itself too
            # large for an unguided whole-graph proof.  Follow anaroute's search
            # boundary here: generate hard-legal neighbours only when A* pops a node.
            # A separate flood fill can otherwise evaluate the whole x*y*layer graph
            # before the first A* expansion (the GF180 vout failure had 768 indexed
            # goals; Bandgap net0167 reached 551,936 connectivity nodes with 0 A*
            # expansions).  Skipping this optional proof does not relax a hard
            # predicate or a work budget; bounded A* below uses the same exact
            # wire/via legality and fails closed when no legal path exists.
            use_hard_connectivity_preflight = (
                graph_within_potential_cap
                and event_graph is None
                and potential_node_count <= self.max_expansions
                and not indexed_goal_domain
                and not hard_connectivity_proven
                # Old captured engines do not run the new constructor on unpickle.
                # Missing configuration therefore retains the historical strategy.
                and not getattr(self, "prefer_target_search", False)
            )
            def _cpp_wide_wire_cb(w_gi: int, w_ax: int, w_ay: int, w_bx: int, w_by: int):
                prev_scans = self._active_static_metal_scans
                active_scans = {}
                self._active_static_metal_scans = active_scans
                self._active_edge_ctx_bypass = True
                try:
                    gen_before = getattr(self, "_hard_rejection_generation", 0)
                    legal = self._wire_legal(w_gi, w_ax, w_ay, w_bx, w_by, net)
                    rej_inc = getattr(self, "_hard_rejection_generation", 0) > gen_before
                finally:
                    self._active_edge_ctx_bypass = False
                    self._active_static_metal_scans = prev_scans
                pres = -1
                if legal:
                    box = _seg_bbox(w_ax, w_ay, w_bx, w_by, self._hw(net, w_gi))
                    scan = active_scans.get((int(w_gi), tuple(_norm(box)), str(net)))
                    if scan is not None:
                        pres = int(scan[0])
                return (legal, pres, rej_inc)

            def _cpp_wide_via_cb(w_x: int, w_y: int, w_lo: int, w_nf: int):
                prev_m = self._active_static_metal_scans
                prev_c = self._active_static_cut_scans
                am, ac = {}, {}
                self._active_static_metal_scans = am
                self._active_static_cut_scans = ac
                self._active_edge_ctx_bypass = True
                n_opt = 1 if w_nf == 1 else None
                try:
                    gen_before = getattr(self, "_hard_rejection_generation", 0)
                    legal = self._via_legal(w_x, w_y, w_lo, net, n=n_opt)
                    rej_inc = getattr(self, "_hard_rejection_generation", 0) > gen_before
                finally:
                    self._active_edge_ctx_bypass = False
                    self._active_static_metal_scans = prev_m
                    self._active_static_cut_scans = prev_c
                p_lo = p_hi = p_cut = -1
                if legal:
                    bh_w, ah_w = (
                        self._via_pad_half1[w_lo]
                        if w_nf == 1
                        else self._via_pad_half[w_lo]
                    )
                    net_s = str(net)
                    s_lo = am.get((int(w_lo), (w_x - bh_w, w_y - bh_w, w_x + bh_w, w_y + bh_w), net_s))
                    s_hi = am.get((int(w_lo) + 1, (w_x - ah_w, w_y - ah_w, w_x + ah_w, w_y + ah_w), net_s))
                    ext_w = self._via_cut_extent(w_lo, n_opt)
                    s_cut = ac.get((int(w_lo), (w_x - ext_w, w_y - ext_w, w_x + ext_w, w_y + ext_w), net_s))
                    if s_lo is not None:
                        p_lo = int(s_lo[0])
                    if s_hi is not None:
                        p_hi = int(s_hi[0])
                    if s_cut is not None:
                        p_cut = int(s_cut[0])
                return (legal, p_lo, p_hi, p_cut, rej_inc)

            def _cpp_wide_cong_cb(kind: int, a: int, b: int, c: int, d: int, e: int) -> int:
                self._active_edge_ctx_bypass = True
                try:
                    if kind == 0:
                        return int(self._edge_congestion(a, b, c, d, e, net))
                    if kind == 1:
                        return int(self._via_congestion(a, b, c, net, 1 if d == 1 else None))
                    return int(self._terminal_via_objective(net, _ViaSite(a, b, c, d)))
                finally:
                    self._active_edge_ctx_bypass = False

            use_cpp_cartesian_preflight = (
                _edge_ctx is not None
                and bytearray is _base_bytearray
                and array is _base_array
                and "_watchdog_check" not in self.__dict__
            )
            if use_hard_connectivity_preflight and use_cpp_cartesian_preflight:
                def _cpp_preflight_wd(phase_id: int, c_nodes: int, c_edges: int, c_packed: bool) -> None:
                    nonlocal connectivity_nodes, connectivity_edges, geometry_mode
                    connectivity_nodes = c_nodes
                    connectivity_edges = c_edges
                    geometry_mode = "packed" if c_packed else "direct"
                    publish_connectivity_watchdog_details()
                    self._watchdog_check(
                        net, "astar_connectivity" if phase_id == 0 else "astar_geometry"
                    )

                p_start_ids = [physical_node_id(s3[0], s3[1], s3[2]) for s3, _via in starts]
                p_goal_ids = [physical_node_id(g[0], g[1], g[2]) for g in goal_coords]
                c_status, connectivity_nodes, connectivity_edges, c_packed = (
                    _edge_ctx.run_hard_connectivity_preflight(
                        xs,
                        ys,
                        ordered_layers,
                        p_start_ids,
                        p_goal_ids,
                        int(self.max_expansions),
                        graph_within_potential_cap,
                        _cpp_preflight_wd,
                        _cpp_wide_wire_cb,
                        _cpp_wide_via_cb,
                    )
                )
                if c_status == -1:
                    geometry_mode = "direct"
                    publish_watchdog_details()
                    connectivity_result = True
                else:
                    geometry_mode = "packed" if c_packed else "direct"
                    connectivity_result = bool(c_status)
            else:
                connectivity_result = (
                    hard_connectivity_preflight()
                    if use_hard_connectivity_preflight else None
                )
            if connectivity_result is False:
                if pattern_incumbent is not None:
                    return incumbent_result()
                observation = getattr(self, "_access_static_disconnection_evidence", None)
                if type(observation) is _AccessStaticDisconnectionRecorder:
                    observation.scope = "finite_candidate_graph"
                    observation.elapsed_ms = None
                self._edge_fail = (
                    "static_disconnected",
                    connectivity_nodes,
                    connectivity_edges,
                    len(xs),
                    len(ys),
                    len(ordered_layers),
                    self._hard_rejection_evidence(),
                )
                return None

            use_cpp_cartesian_astar = (
                _edge_ctx is not None
                and event_graph is None
                and bytearray is _base_bytearray
                and array is _base_array
                and heapq is _base_heapq
                and "_watchdog_check" not in self.__dict__
                and "_cached_exact_multigoal_potential" not in self.__dict__
                and "_heuristic" not in self.__dict__
            )
            if use_cpp_cartesian_astar:
                force_packed_astar = bool(
                    graph_within_potential_cap
                    and (hard_connectivity_proven or indexed_goal_domain)
                )
                if force_packed_astar:
                    geometry_mode = "packed"
                publish_watchdog_details()

                def _cpp_astar_wd(phase_id: int, cur_exp: int, cur_tp: int, c_packed: bool) -> None:
                    nonlocal expansions, terminal_access_priced, geometry_mode
                    expansions = cur_exp
                    terminal_access_priced = cur_tp
                    geometry_mode = "packed" if c_packed else "direct"
                    publish_watchdog_details()
                    if phase_id == 1:
                        self._watchdog_check(net, "astar_geometry")
                    elif phase_id == 2:
                        self._watchdog_check(net, "astar")
                    else:
                        self._watchdog_check(net, "astar_terminal_access")

                starts_spec = [
                    (
                        xi[s3[0]],
                        yi[s3[1]],
                        layer_index[s3[2]],
                        via is not None,
                        0 if via is None else int(via.x),
                        0 if via is None else int(via.y),
                        0 if via is None else int(via.lo),
                        0 if via is None else int(via.n),
                        terminal_via_lower_bound(via),
                    )
                    for s3, via in starts
                ]
                goals_spec = [
                    (
                        xi[g[0]],
                        yi[g[1]],
                        layer_index[g[2]],
                        goals[g] is not None,
                        0 if goals[g] is None else int(goals[g].x),
                        0 if goals[g] is None else int(goals[g].y),
                        0 if goals[g] is None else int(goals[g].lo),
                        0 if goals[g] is None else int(goals[g].n),
                        goal_terminal_lower_bound[g],
                    )
                    for g in goal_coords
                ]
                a_status, expansions, terminal_access_priced, a_packed, root_s_idx, path_nodes = (
                    _edge_ctx.run_cartesian_astar(
                        xs,
                        ys,
                        ordered_layers,
                        starts_spec,
                        goals_spec,
                        int(budget),
                        int(self._solve_expansions_total),
                        int(self._solve_expansion_budget),
                        int(self.max_expansions),
                        graph_within_potential_cap,
                        indexed_goal_domain,
                        None if pattern_incumbent is None else int(pattern_incumbent.score),
                        force_packed_astar,
                        _cpp_astar_wd,
                        (_cpp_wide_wire_cb, _cpp_wide_via_cb, _cpp_wide_cong_cb),
                    )
                )
                geometry_mode = "packed" if a_packed else "direct"
                publish_watchdog_details()
                if a_status == 0:
                    self._solve_expansions_total = getattr(
                        self, "_solve_expansions_total", 0
                    ) + expansions
                    nodes_out = list(path_nodes)
                    goal_node = nodes_out[-1]
                    root_via = starts[root_s_idx][1] if root_s_idx >= 0 else None
                    goal_via = goals.get((goal_node[0], goal_node[1], goal_node[2]))
                    if search_reversed:
                        nodes_out.reverse()
                        ev = goal_via
                        gv = root_via
                    else:
                        ev = root_via
                        gv = goal_via
                    segs_out: List[_WireSeg] = []
                    vias_out: List[_ViaSite] = []
                    if ev is not None:
                        vias_out.append(ev)
                    for i in range(1, len(nodes_out)):
                        ax, ay, al = nodes_out[i - 1][0], nodes_out[i - 1][1], nodes_out[i - 1][2]
                        bx, by, bl = nodes_out[i][0], nodes_out[i][1], nodes_out[i][2]
                        if al != bl:
                            tnode = nodes_out[i - 1] if search_reversed else nodes_out[i]
                            vlo = tnode[4]
                            vn = tnode[5]
                            vias_out.append(
                                _ViaSite(ax, ay, vlo, vn)
                                if vlo >= 0
                                else _ViaSite(ax, ay, min(al, bl))
                            )
                        elif (ax, ay) != (bx, by):
                            segs_out.append(self._wire_segment(net, al, ax, ay, bx, by))
                    if gv is not None:
                        vias_out.append(gv)
                    if nodes_out:
                        self._record_selected_access_pair(
                            ((nodes_out[0][0], nodes_out[0][1], nodes_out[0][2]), ev),
                            ((nodes_out[-1][0], nodes_out[-1][1], nodes_out[-1][2]), gv),
                        )
                    return segs_out, vias_out
                if a_status == 1:
                    self._solve_expansions_total = getattr(
                        self, "_solve_expansions_total", 0
                    ) + expansions
                    return incumbent_result()
                if a_status in (3, 4):
                    self._edge_fail = (
                        "solve_budget" if a_status == 4 else "budget",
                        expansions,
                        budget,
                        self._solve_expansions_total,
                        self._solve_expansion_budget,
                        len(xs),
                        len(ys),
                        len(allowed),
                    )
                    self._solve_expansions_total = getattr(
                        self, "_solve_expansions_total", 0
                    ) + expansions
                    return None
                self._edge_fail = (
                    "exhausted",
                    expansions,
                    len(xs),
                    len(ys),
                    len(allowed),
                    self._hard_rejection_evidence(),
                )
                self._solve_expansions_total = getattr(
                    self, "_solve_expansions_total", 0
                ) + expansions
                return None

            if graph_within_potential_cap and (
                hard_connectivity_proven or indexed_goal_domain
            ):
                # A large indexed access component is already known to require a
                # non-trivial search.  Allocate the existing bounded per-edge cache
                # before its first expansion so turn-state revisits reuse exact DRC
                # judgments; waiting until every graph edge had been touched left
                # the live compact-OTA bias net in ``geometry_mode=direct`` for the
                # whole watchdog interval.
                enable_packed_geometry(force=True)
            publish_watchdog_details()

            State = Tuple[int, int, int, int]  # (x, y, gi, last_dir)
            dist: Dict[State, int] = {}
            prev: Dict[State, Optional[State]] = {}
            entry_via: Dict[State, Optional[_ViaSite]] = {}
            # Keep the historical (f, state) ordering exactly; queued g is only a
            # trailing stale-entry witness and therefore cannot perturb tie-breaks.
            # At equal cost, expand ordinary states before deferred layer-graph
            # construction.  Historical ordering among ordinary states is unchanged.
            pq: List[Tuple[int, int, State, int]] = []
            # Terminal access is a virtual edge from a super-source/super-target.
            # Queue its cheap structural lower bound first; exact via congestion is
            # evaluated only when that virtual edge reaches the global frontier.
            # This is the access-component split used by analytical routers: retain
            # every candidate without paying the source*goal Cartesian cost up front.
            start_frontier: List[
                Tuple[int, Tuple[int, int, int], Tuple[int, ...], int]
            ] = []
            region_frontier: List[
                Tuple[int, int, str, State, int, Tuple[Tuple[int, int], ...], int]
            ] = []
            region_seq = 0
            _region_event_geometry_mod = None
            _project_portal = None
            _project_target = None
            _settled_axis_prefix = None
            _cpp_via_projector = None
            if event_graph is not None:
                from . import region_event_geometry as _region_event_geometry_mod
                _project_portal = _region_event_geometry_mod.project_portal
                _project_target = _region_event_geometry_mod.project_target
                _settled_axis_prefix = _region_event_geometry_mod.settled_axis_prefix
            for start_index, (s3, via) in enumerate(starts):
                lower_bound = terminal_via_lower_bound(via)
                heapq.heappush(
                    start_frontier,
                    (
                        lower_bound + hmin(*s3),
                        s3,
                        via_key(via),
                        start_index,
                    ),
                )
            initial_start_frontier = list(start_frontier)
            exact_terminal_cache: Dict[Tuple[int, ...], int] = {}

            def exact_terminal_cost(via: Optional[_ViaSite]) -> int:
                nonlocal terminal_access_priced
                if via is None:
                    return 0
                key = via_key(via)
                cached = exact_terminal_cache.get(key)
                if cached is not None:
                    return cached
                publish_watchdog_details()
                self._watchdog_check(net, "astar_terminal_access")
                value = self._terminal_via_objective(net, via)
                exact_terminal_cache[key] = value
                terminal_access_priced += 1
                publish_watchdog_details()
                self._watchdog_check(net, "astar_terminal_access")
                return value

            event_work_by_kind = {}

            def charge_event(kind='region_step'):
                nonlocal event_work, event_budget_stopped
                if (expansions >= budget
                        or expansions + event_work >= event_budget
                        or self._solve_expansions_total + expansions >= self._solve_expansion_budget):
                    event_budget_stopped = True
                    return False
                event_work += 1
                event_work_by_kind[kind] = event_work_by_kind.get(kind, 0) + 1
                self._solve_expansions_total += 1
                if event_work & 255 == 1:
                    self._watchdog_check(net, "planar_region_successors")
                return True

            # Prices depend on this edge search's immutable net, axes, trunk,
            # congestion and Book.  Never carry this memo across searches.
            event_run_cache = {}
            price_cache_hits = 0
            static_price_steps = 0

            def charge_static_price(_kind='axis_price'):
                nonlocal event_budget_stopped, static_price_steps
                if self._solve_expansions_total + expansions >= self._solve_expansion_budget:
                    event_budget_stopped = True
                    return False
                self._solve_expansions_total += 1
                static_price_steps += 1
                if static_price_steps & 255 == 0:
                    self._watchdog_check(net, "planar_region_axis_price")
                return True

            from .axis_price_cache import AxisPriceCache
            def _fb_region_price_fn(p_gi, p_ax, p_ay, p_bx, p_by):
                self._active_edge_ctx_bypass = True
                try:
                    return (
                        self._edge_objective(net, p_gi, p_ax, p_ay, p_bx, p_by, trunk)
                        + self._edge_congestion(p_gi, p_ax, p_ay, p_bx, p_by, net)
                    )
                finally:
                    self._active_edge_ctx_bypass = False

            def _fb_region_wire_fn(w_gi, w_ax, w_ay, w_bx, w_by):
                self._active_edge_ctx_bypass = True
                try:
                    return bool(self._wire_legal(w_gi, w_ax, w_ay, w_bx, w_by, net))
                finally:
                    self._active_edge_ctx_bypass = False

            def _region_cpp_wd_cb(wd_kind: int) -> None:
                if wd_kind == 0:
                    self._watchdog_check(net, "planar_region_axis_price")
                elif wd_kind == 1:
                    self._watchdog_check(net, "planar_region_price_cache")
                else:
                    self._watchdog_check(net, "planar_region_successors")

            axis_price_cache = AxisPriceCache(
                xs, ys,
                lambda gi, ax, ay, bx, by: (
                    self._edge_objective(net, gi, ax, ay, bx, by, trunk)
                    + self._edge_congestion(gi, ax, ay, bx, by, net)),
                charge_static_price,
            )
            _use_cpp_region_axis = (
                _edge_ctx is not None
                and axis_price_cache._native is not None
            )
            _use_cpp_region_step = (
                _use_cpp_region_axis
                and _advance_planar_region_step is _base_advance_planar_region_step
                and _settled_axis_prefix is _ORIG_SETTLED_AXIS_PREFIX
            )
            _max_run_cache_i = max(0, int(self.max_expansions))

            def event_run_cost(gi, start, end):
                nonlocal price_cache_hits, static_price_steps, event_budget_stopped
                if start == end:
                    return 0
                if _use_cpp_region_axis:
                    rem = max(0, self._solve_expansion_budget - (self._solve_expansions_total + expansions))
                    total, charged, static_price_steps, price_cache_hits, bstop = axis_price_cache.run_ctx(
                        gi,
                        start,
                        end,
                        _edge_ctx,
                        rem,
                        static_price_steps,
                        price_cache_hits,
                        _max_run_cache_i,
                        _region_cpp_wd_cb,
                        _fb_region_price_fn,
                    )
                    self._solve_expansions_total += charged
                    if bstop:
                        event_budget_stopped = True
                    return total
                key = (gi, start, end)
                cached = event_run_cache.get(key)
                if cached is not None:
                    price_cache_hits += 1
                    if price_cache_hits & 1023 == 0:
                        self._watchdog_check(net, "planar_region_price_cache")
                    return cached
                total = axis_price_cache.run(gi, start, end)
                if total is None:
                    return None
                if len(event_run_cache) < _max_run_cache_i:
                    event_run_cache[key] = total
                return total

            def _cpp_advance_step(cur_u, cur_path, cur_dest):
                nonlocal event_work, static_price_steps, price_cache_hits, event_budget_stopped
                rem = max(0, self._solve_expansion_budget - (self._solve_expansions_total + expansions))
                step_pair, ev_ch, pr_ch, static_price_steps, price_cache_hits, bstop = (
                    _edge_ctx.advance_region_step_ctx(
                        cur_u,
                        cur_path,
                        cur_dest,
                        target_domain.grid,
                        axis_price_cache._native,
                        expansions,
                        budget,
                        event_work,
                        event_budget,
                        rem,
                        static_price_steps,
                        price_cache_hits,
                        _max_run_cache_i,
                        _region_cpp_wd_cb,
                        _fb_region_wire_fn,
                        _fb_region_price_fn,
                    )
                )
                if ev_ch:
                    event_work += ev_ch
                    event_work_by_kind['region_step'] = event_work_by_kind.get('region_step', 0) + ev_ch
                if ev_ch or pr_ch:
                    self._solve_expansions_total += ev_ch + pr_ch
                if bstop:
                    event_budget_stopped = True
                if step_pair is None:
                    return None
                return (step_pair[0], step_pair[1], None)

            via_region_cache = {}
            static_build_steps = 0
            class EventBudgetStop(Exception):
                pass

            def checkpoint_static_build():
                nonlocal event_budget_stopped, static_build_steps
                if self._solve_expansions_total + expansions >= self._solve_expansion_budget:
                    event_budget_stopped = True
                    raise EventBudgetStop()
                self._solve_expansions_total += 1
                static_build_steps += 1
                if static_build_steps & 255 == 0:
                    self._watchdog_check(net, "planar_region_static_build")

            neg_bucket_dedup = event_graph is not None

            def ensure_event_graph(layer):
                if layer in event_graphs:
                    return event_graphs[layer]
                layer_domain = replace(target_domain, node=(target_domain.node[0], target_domain.node[1], layer))
                try:
                    graph = self._planar_event_geometry_bounded(net, layer_domain, checkpoint_static_build)
                except EventBudgetStop:
                    return None
                if graph is not None:
                    event_graphs[layer] = graph
                return graph

            def via_regions(lo):
                cached = via_region_cache.get(lo)
                if cached is not None:
                    return cached
                if ensure_event_graph(lo) is None or ensure_event_graph(lo + 1) is None:
                    return {}
                try:
                    rectangles = self._planar_via_event_regions(
                        net, lo, event_graphs[lo], event_graphs[lo + 1],
                        target_domain.grid, checkpoint_static_build)
                    by_layer = {lo: {}, lo + 1: {}}
                    for rect in rectangles:
                        x0, y0, x1, y1 = rect
                        for layer in (lo, lo + 1):
                            for owner, _box in event_graphs[layer][2].query((x0, y0, x1 - 1, y1 - 1)):
                                checkpoint_static_build()
                                by_layer[layer].setdefault(int(owner), []).append(rect)
                except EventBudgetStop:
                    return {}
                via_region_cache[lo] = by_layer
                return by_layer

            restrict_rect_entries = not bool(
                getattr(self, "_search_forbidden_wire_edges", ())
            )
            rect_bucket_nm = 140
            if event_graph is not None:
                _cpp_mod_vp = load_native("_gridless_cpp")
                if (
                    _cpp_mod_vp is not None
                    and isinstance(hmin, _cpp_mod_vp.NativeLayerHmin)
                    and _project_target is _ORIG_PROJECT_TARGET
                    and _eg_attr("_region_via_goal_waypoint", _base_region_via_goal_waypoint) is _base_region_via_goal_waypoint
                    and "_via_successor_last_dir" not in self.__dict__
                ):
                    _t_node = target_domain.node
                    _t_sing_none = target_domain.singleton is None and target_domain.grid > 0
                    _t_regs = (
                        tuple(r for _atom, r in target_domain.regions)
                        if _t_sing_none
                        else ()
                    )
                    _t_excl = (
                        tuple(
                            (entry[0], entry[1])
                            for entry in target_domain.excluded
                            if (
                                len(entry) == 8
                                and entry[2] == _t_node[2]
                                and entry[3] == 0 and entry[4] == 0
                                and entry[5] == 0 and entry[6] == 0
                                and entry[7] == 0
                            )
                        )
                        if _t_sing_none and target_domain.excluded
                        else ()
                    )
                    _cpp_via_projector = _cpp_mod_vp.NativeViaTargetProjector(
                        goal_set=goal_set,
                        target_node=_t_node,
                        target_singleton_none=_t_sing_none,
                        target_regions=_t_regs,
                        target_excluded_xy=_t_excl,
                        hmin=hmin,
                        grid=int(target_domain.grid),
                        turn_cost=int(turn_cost),
                        via_cost=int(_VIA_COST),
                        rect_bucket_nm=rect_bucket_nm,
                        neg_bucket_dedup=bool(neg_bucket_dedup),
                        via_succ_dir_1=int(self._via_successor_last_dir(1)),
                        via_succ_dir_2=int(self._via_successor_last_dir(2)),
                    )
            saturation_pruned = False
            rect_entry_expansions: Dict[Tuple[int, int, int, int, int], int] = {}
            rect_via_expansions: Dict[Tuple[int, int, int], int] = {}
            _use_cpp_region_frontier = (
                _cpp_via_projector is not None
                and heapq is _base_heapq
                and _region_via_goal_prefix_lower_bound is _base_region_via_goal_prefix_lower_bound
                and _region_via_target_prunable is _base_region_via_target_prunable
            )
            _cpp_other_caps: Dict[int, Optional[int]] = {}

            def rect_entry_cap(layer: int) -> Optional[int]:
                if not restrict_rect_entries or layer not in event_graphs:
                    return None
                if not neg_bucket_dedup:
                    if target_domain is not None and layer == target_domain.node[2]:
                        return None
                    if len(event_graphs[layer][0].rectangles) <= 512:
                        return None
                    return 2
                if target_domain is not None and layer == target_domain.node[2]:
                    return 4
                return 3

            def rect_saturated(
                layer: int, rect_id: int, x_nm: int = 0, y_nm: int = 0, ulast: int = 0
            ) -> bool:
                nonlocal saturation_pruned
                cap = rect_entry_cap(layer)
                if cap is None:
                    return False
                if neg_bucket_dedup:
                    key = (
                        layer,
                        rect_id,
                        int(ulast),
                        x_nm // rect_bucket_nm,
                        y_nm // rect_bucket_nm,
                    )
                else:
                    key = (layer, rect_id, 0, 0, 0)
                if rect_entry_expansions.get(key, 0) >= cap:
                    saturation_pruned = True
                    return True
                return False

            def rect_via_saturated(layer: int, other_layer: int, rect_id: int) -> bool:
                nonlocal saturation_pruned
                if not restrict_rect_entries or neg_bucket_dedup or layer not in event_graphs:
                    return False
                if target_domain is not None and layer == target_domain.node[2]:
                    return False
                if len(event_graphs[layer][0].rectangles) <= 512:
                    return False
                if rect_via_expansions.get((layer, other_layer, rect_id), 0) >= 2:
                    saturation_pruned = True
                    return True
                return False

            via_target_cache: Dict[
                Tuple[int, int, int, Tuple[int, int]],
                Tuple[bool, Tuple[Any, ...]],
            ] = {}
            clipped_via_cache: Dict[
                Tuple[int, int, int],
                Tuple[Tuple[int, int, int, int], ...],
            ] = {}
            other_owner_cache: Dict[
                Tuple[int, int, int],
                Tuple[int, ...],
            ] = {}
            waypoint_cache: Dict[
                Tuple[int, Tuple[Tuple[int, int], ...]],
                Optional[int],
            ] = {}
            portal_cache: Dict[
                Tuple[int, int, Tuple[int, int]],
                Tuple[Any, ...],
            ] = {}
            goal_target_cache: Dict[
                Tuple[int, Tuple[int, int]],
                Tuple[Any, ...],
            ] = {}
            cross_via_cache: Dict[
                Tuple[int, int, int],
                Optional[Tuple[int, Any]],
            ] = {}
            _cross_via_missing = object()

            def successors(u: State):
                """Yield each legal single-segment or single-via A* transition."""
                nonlocal saturation_pruned
                ux, uy, ul, ulast = u[:4]
                if len(u) == 4:
                    for vx, vy, vdir in planar(ux, uy):
                        static_cost = planar_static_cost(ul, ux, uy, vx, vy)
                        if static_cost is None:
                            continue
                        turn = turn_cost if (ulast != 0 and vdir != ulast) else 0
                        yield (vx, vy, ul, vdir), static_cost + turn, None
                    for other in adjacent_layers(ul):
                        lo = min(ul, other)
                        static_cost = via_static_cost(ux, uy, lo)
                        if static_cost is None:
                            continue
                        yield (ux, uy, other, self._via_successor_last_dir(ulast)), static_cost, lo
                    if (event_graph is not None
                            and ux % target_domain.grid == 0 and uy % target_domain.grid == 0
                            and ensure_event_graph(ul) is not None):
                        _geometry, _portals, region_index = event_graphs[ul]
                        px, py = ux // target_domain.grid, uy // target_domain.grid
                        for owner, _rect in region_index.query((px, py, px, py)):
                            owner_id = int(owner)
                            if rect_saturated(ul, owner_id, ux, uy, ulast):
                                continue
                            if not charge_event('grid_to_region'):
                                return
                            yield (ux, uy, ul, ulast, owner_id, (), -1, ux, uy, 0), 0, None
                    return

                project_portal = _project_portal
                project_target = _project_target
                settled_axis_prefix = _settled_axis_prefix
                geometry, portals, _region_index = event_graphs[ul]
                rectangle, pending, destination, run_x, run_y, run_dir = u[4:10]
                grid = target_domain.grid
                point = (ux // grid, uy // grid)

                def follow(path, next_rectangle):
                    if _use_cpp_region_step:
                        return _cpp_advance_step(u, path, next_rectangle)
                    return _advance_planar_region_step(
                        u, path, next_rectangle, route_world=self, net=net,
                        grid=grid, xs=xs, ys=ys, turn_cost=turn_cost,
                        charge_event=charge_event, event_run_cost=event_run_cost,
                        settle_prefix=settled_axis_prefix,
                    )

                def cross_via(other):
                    if not charge_event('via_check'):
                        return
                    lo = min(ul, other)
                    vkey = (ux, uy, lo)
                    via_info = cross_via_cache.get(vkey, _cross_via_missing)
                    if via_info is _cross_via_missing:
                        cpp_cv = _edge_ctx.via_static_cost(ux, uy, lo) if _edge_ctx is not None else None
                        if cpp_cv is False:
                            cross_via_cache[vkey] = None
                            return
                        if cpp_cv is not None:
                            via_cost_val, model_n = cpp_cv
                            via_tag = (lo, 1) if model_n == 1 else lo
                            via_info = (via_cost_val, via_tag)
                            cross_via_cache[vkey] = via_info
                        else:
                            rejection_generation_before = getattr(
                                self, "_hard_rejection_generation", 0
                            )
                            prev_metal_scans = self._active_static_metal_scans
                            prev_cut_scans = self._active_static_cut_scans
                            self._active_static_metal_scans = {}
                            self._active_static_cut_scans = {}
                            prev_byp = getattr(self, "_active_edge_ctx_bypass", False)
                            if _edge_ctx is not None:
                                self._active_edge_ctx_bypass = True
                            try:
                                default_legal = self._via_legal(ux, uy, lo, net)
                                model_n = 0
                                if not default_legal:
                                    if not (single_escape_ok and self._via_legal(ux, uy, lo, net, n=1)):
                                        if (
                                            getattr(self, "_hard_rejection_generation", 0)
                                            > rejection_generation_before
                                        ):
                                            cross_via_cache[vkey] = None
                                        return
                                    model_n = 1
                                if model_n == 1:
                                    single = _ViaSite(ux, uy, lo, 1)
                                    via_cost_val = (
                                        _VIA_COST
                                        + self._functional_single_via_penalty(single)
                                        + self._via_congestion(ux, uy, lo, net, 1)
                                    )
                                else:
                                    default_cong = self._via_congestion(ux, uy, lo, net)
                                    via_cost_val = _VIA_COST + default_cong
                                    if single_escape_ok:
                                        single = _ViaSite(ux, uy, lo, 1)
                                        single_pen = self._functional_single_via_penalty(single)
                                        if default_cong > single_pen and self._via_legal(ux, uy, lo, net, n=1):
                                            single_cost_val = (
                                                _VIA_COST
                                                + single_pen
                                                + self._via_congestion(ux, uy, lo, net, 1)
                                            )
                                            if single_cost_val < via_cost_val:
                                                via_cost_val = single_cost_val
                                                model_n = 1
                            finally:
                                if _edge_ctx is not None:
                                    self._active_edge_ctx_bypass = prev_byp
                                self._active_static_metal_scans = prev_metal_scans
                                self._active_static_cut_scans = prev_cut_scans
                            via_tag = (lo, 1) if model_n == 1 else lo
                            via_info = (via_cost_val, via_tag)
                            cross_via_cache[vkey] = via_info
                    elif via_info is None:
                        return
                    else:
                        via_cost_val, via_tag = via_info
                    pending_cost = event_run_cost(ul, (run_x, run_y), (ux, uy)) if run_dir else 0
                    if pending_cost is None:
                        return
                    next_ulast = self._via_successor_last_dir(ulast)
                    okey = (other, point[0], point[1])
                    owners = other_owner_cache.get(okey)
                    if owners is None:
                        owners = tuple(
                            int(owner)
                            for owner, _rect in event_graphs[other][2].query(
                                (point[0], point[1], point[0], point[1])
                            )
                        )
                        other_owner_cache[okey] = owners
                    for owner_id in owners:
                        if rect_saturated(other, owner_id, ux, uy, next_ulast):
                            continue
                        if not charge_event('via_destination'):
                            return
                        yield (ux, uy, other, next_ulast,
                               owner_id, (), -1, ux, uy, 0), pending_cost + via_cost_val, via_tag

                if len(u) == 11:
                    nonlocal region_seq
                    other = u[10]
                    u_base = u[:10]
                    u_du = dist[u]
                    u_rate = layer_rates[ul]
                    pending_lb = (
                        (abs(ux - run_x) + abs(uy - run_y)) * u_rate
                        + pending_history_lb(ul, run_x, run_y, ux, uy)
                        if run_dir else 0
                    )
                    check_other_sat = (
                        neg_bucket_dedup
                        and restrict_rect_entries
                        and rect_entry_cap(other) is not None
                    )
                    _other_reg_idx_fast = event_graphs[other][2] if check_other_sat else None
                    if (
                        _use_cpp_region_frontier
                        and (
                            _other_reg_idx_fast is None
                            or (
                                type(_other_reg_idx_fast) is _SpatialIndex
                                and _other_reg_idx_fast._native is not None
                            )
                        )
                    ):
                        px, py = point
                        with_owners_fast = _other_reg_idx_fast is not None
                        if _cpp_via_projector.has_cached_target(
                            ul, other, rectangle, px, py, with_owners_fast
                        ):
                            clipped_arg = None
                        else:
                            clip_key = (ul, other, rectangle)
                            clipped_fast = clipped_via_cache.get(clip_key)
                            if clipped_fast is None:
                                lo = min(ul, other)
                                rx0, ry0, rx1, ry1 = geometry.rectangles[rectangle]
                                candidates = via_regions(lo).get(ul, {}).get(rectangle, ())
                                seen_c = set()
                                c_list = []
                                for target_idx, (tx0, ty0, tx1, ty1) in enumerate(candidates):
                                    if target_idx & 255 == 255:
                                        self._watchdog_check(net, "planar_region_via_targets")
                                    lx = rx0 if rx0 >= tx0 else tx0
                                    ly = ry0 if ry0 >= ty0 else ty0
                                    hx = rx1 if rx1 <= tx1 else tx1
                                    hy = ry1 if ry1 <= ty1 else ty1
                                    if lx < hx and ly < hy:
                                        box = (lx, ly, hx, hy)
                                        if box not in seen_c:
                                            seen_c.add(box)
                                            c_list.append(box)
                                clipped_fast = tuple(c_list)
                                if not event_budget_stopped:
                                    clipped_via_cache[clip_key] = clipped_fast
                            clipped_arg = clipped_fast
                        rx0, ry0, rx1, ry1 = geometry.rectangles[rectangle]
                        cutoff = best_goal[0] if best_goal is not None else None
                        pattern_cutoff = pattern_incumbent.score if pattern_incumbent is not None else None
                        base_lb = u_du + pending_lb
                        other_cap = rect_entry_cap(other) if neg_bucket_dedup else None
                        _cpp_other_caps[other] = other_cap
                        fmask = _cpp_via_projector.enqueue_via_targets(
                            ul, other, rectangle, px, py,
                            rx0, ry0, rx1, ry1,
                            clipped_arg,
                            _other_reg_idx_fast._native if with_owners_fast else None,
                            u_base, u_du, ulast, u_rate, base_lb,
                            cutoff, pattern_cutoff, other_cap,
                            event_budget_stopped,
                        )
                        if fmask & 2:
                            saturation_pruned = True
                        if fmask & 1:
                            yield from cross_via(other)
                        return
                    cache_key = (ul, other, rectangle, point)
                    cached_targets = via_target_cache.get(cache_key)
                    if cached_targets is None:
                        clip_key = (ul, other, rectangle)
                        clipped = clipped_via_cache.get(clip_key)
                        if clipped is None:
                            lo = min(ul, other)
                            rx0, ry0, rx1, ry1 = geometry.rectangles[rectangle]
                            candidates = via_regions(lo).get(ul, {}).get(rectangle, ())
                            seen_c = set()
                            c_list = []
                            for target_idx, (tx0, ty0, tx1, ty1) in enumerate(candidates):
                                if target_idx & 255 == 255:
                                    self._watchdog_check(net, "planar_region_via_targets")
                                lx = rx0 if rx0 >= tx0 else tx0
                                ly = ry0 if ry0 >= ty0 else ty0
                                hx = rx1 if rx1 <= tx1 else tx1
                                hy = ry1 if ry1 <= ty1 else ty1
                                if lx < hx and ly < hy:
                                    box = (lx, ly, hx, hy)
                                    if box not in seen_c:
                                        seen_c.add(box)
                                        c_list.append(box)
                            clipped = tuple(c_list)
                            if not event_budget_stopped:
                                clipped_via_cache[clip_key] = clipped
                        px, py = point
                        _other_reg_idx = event_graphs[other][2] if check_other_sat else None
                        if (
                            _cpp_via_projector is not None
                            and (
                                _other_reg_idx is None
                                or (type(_other_reg_idx) is _SpatialIndex and _other_reg_idx._native is not None)
                            )
                        ):
                            rx0, ry0, rx1, ry1 = geometry.rectangles[rectangle]
                            cached_targets = _cpp_via_projector.project(
                                ul, other, px, py,
                                rx0, ry0, rx1, ry1,
                                clipped,
                                _other_reg_idx._native if _other_reg_idx is not None else None,
                            )
                        else:
                            has_cross = False
                            seen_via_paths = set()
                            ends_seen = {point}
                            records = []
                            for lx, ly, hx, hy in clipped:
                                if lx <= px < hx and ly <= py < hy:
                                    has_cross = True
                                    continue
                                ex_g = lx if px < lx else (hx - 1 if px >= hx else px)
                                ey_g = ly if py < ly else (hy - 1 if py >= hy else py)
                                end_pt = (ex_g, ey_g)
                                if end_pt in ends_seen:
                                    continue
                                ends_seen.add(end_pt)
                                for path in project_target(geometry, rectangle, point, (lx, ly, hx, hy)):
                                    if len(path) == 1:
                                        has_cross = True
                                    elif path not in seen_via_paths:
                                        seen_via_paths.add(path)
                                        epx, epy = path[-1]
                                        ex, ey = epx * grid, epy * grid
                                        end3 = (ex, ey, ul)
                                        if neg_bucket_dedup:
                                            wkey = (ul, path)
                                            goal_waypoint = waypoint_cache.get(wkey)
                                            if goal_waypoint is None and wkey not in waypoint_cache:
                                                goal_waypoint = _region_via_goal_waypoint(
                                                    path, grid=grid, layer=ul, goal_set=goal_set,
                                                    target_domain=target_domain,
                                                )
                                                waypoint_cache[wkey] = goal_waypoint
                                        else:
                                            is_endpoint_goal = (
                                                end3 in goal_set or (
                                                    ul == target_domain.node[2]
                                                    and bool(target_domain.owning_atoms(end3))
                                                )
                                            )
                                            goal_waypoint = len(path) - 1 if is_endpoint_goal else None
                                        is_end_goal = goal_waypoint == len(path) - 1
                                        last_vdir = 1 if path[-1][1] == path[-2][1] else 2
                                        other_ulast = self._via_successor_last_dir(last_vdir)
                                        if goal_waypoint is None and check_other_sat:
                                            okey = (other, epx, epy)
                                            owners = other_owner_cache.get(okey)
                                            if owners is None:
                                                owners = tuple(
                                                    int(owner)
                                                    for owner, _box in event_graphs[other][2].query((epx, epy, epx, epy))
                                                )
                                                other_owner_cache[okey] = owners
                                        else:
                                            owners = ()
                                        extra_turn = turn_cost if len(path) > 2 else 0
                                        if goal_waypoint is not None and not is_end_goal:
                                            tail_lb = None
                                        elif is_end_goal:
                                            tail_lb = hmin(ex, ey, ul)
                                        else:
                                            tail_lb = _VIA_COST + hmin(ex, ey, other)
                                        records.append((
                                            path, ex, ey, path[1][1] * grid, goal_waypoint,
                                            is_end_goal, other_ulast, owners, ex // rect_bucket_nm, ey // rect_bucket_nm,
                                            extra_turn, tail_lb,
                                        ))
                            cached_targets = (has_cross, tuple(records))
                        if not event_budget_stopped:
                            via_target_cache[cache_key] = cached_targets
                    has_cross, records = cached_targets
                    if has_cross:
                        yield from cross_via(other)
                    cutoff = best_goal[0] if best_goal is not None else None
                    pattern_cutoff = pattern_incumbent.score if pattern_incumbent is not None else None
                    base_lb = u_du + pending_lb
                    other_cap = rect_entry_cap(other) if neg_bucket_dedup else None
                    for (
                        path, ex, ey, p1_y_nm, goal_waypoint,
                        is_end_goal, other_ulast, owners, ebx, eby, extra_turn, tail_lb,
                    ) in records:
                        if goal_waypoint is None and other_cap is not None:
                            any_open = False
                            for owner_id in owners:
                                if rect_entry_expansions.get((other, owner_id, other_ulast, ebx, eby), 0) < other_cap:
                                    any_open = True
                                    break
                            if not any_open:
                                if owners:
                                    saturation_pruned = True
                                continue
                        if tail_lb is None:
                            lb_f = _region_via_goal_prefix_lower_bound(
                                path, goal_waypoint, grid=grid, ux=ux, uy=uy,
                                ulast=ulast, rate=u_rate, turn_cost=turn_cost,
                                base=u_du, pending=pending_lb,
                            )
                        else:
                            first_vdir = 1 if p1_y_nm == uy else 2
                            turn1 = turn_cost if (ulast != 0 and first_vdir != ulast) else 0
                            lb_f = (
                                base_lb
                                + (abs(ex - ux) + abs(ey - uy)) * u_rate
                                + turn1
                                + extra_turn
                                + tail_lb
                            )
                        if cutoff is not None and lb_f > cutoff:
                            continue
                        if pattern_cutoff is not None and lb_f >= pattern_cutoff:
                            continue
                        region_seq += 1
                        heapq.heappush(
                            region_frontier,
                            (lb_f, region_seq, 'via_target', u_base, u_du, path, -other - 2),
                        )
                    return

                if pending:
                    step = follow((point,) + pending, destination)
                    if step is not None:
                        yield step
                    return
                if destination < -1:
                    yield from cross_via(-destination - 2)
                    return
                if ux in xi and uy in yi:
                    cost = event_run_cost(ul, (run_x, run_y), (ux, uy)) if run_dir else 0
                    if cost is not None:
                        yield (ux, uy, ul, ulast), cost, None
                ul_cap = rect_entry_cap(ul)
                portal_key = (ul, rectangle, point)
                portal_records = portal_cache.get(portal_key)
                if portal_records is None:
                    rect_portals = portals[rectangle]
                    if (
                        _cpp_mod is not None
                        and project_portal is _ORIG_PROJECT_PORTAL
                        and type(geometry.rectangles) is tuple
                        and (len(rect_portals) < 256 or "_watchdog_check" not in self.__dict__)
                    ):
                        portal_rows = tuple(
                            (
                                p.source,
                                p.target,
                                p.axis == "x",
                                p.direction,
                                p.boundary,
                                p.lo,
                                p.hi,
                            )
                            for p in rect_portals
                        )
                        portal_records = _cpp_mod.batch_project_portals_cpp(
                            geometry.rectangles,
                            portal_rows,
                            ul,
                            grid,
                            rect_bucket_nm,
                            neg_bucket_dedup,
                            point[0],
                            point[1],
                        )
                    else:
                        seen_steps = set()
                        precs = []
                        for ordinal, portal in enumerate(rect_portals):
                            if ordinal & 255 == 255:
                                self._watchdog_check(net, "planar_region_portals")
                            ptarget = portal.target
                            for path in project_portal(geometry, portal, point):
                                step_key = (path, ptarget)
                                if step_key in seen_steps:
                                    continue
                                seen_steps.add(step_key)
                                fx, fy = path[-1][0] * grid, path[-1][1] * grid
                                end_vdir = 1 if path[-1][1] == path[-2][1] else 2
                                rkey = (
                                    (ul, ptarget, int(end_vdir), fx // rect_bucket_nm, fy // rect_bucket_nm)
                                    if neg_bucket_dedup
                                    else (ul, ptarget, 0, 0, 0)
                                )
                                precs.append((path, ptarget, rkey))
                        portal_records = tuple(precs)
                    portal_cache[portal_key] = portal_records
                for path, ptarget, rkey in portal_records:
                    if ul_cap is not None and rect_entry_expansions.get(rkey, 0) >= ul_cap:
                        saturation_pruned = True
                        continue
                    if not charge_event('portal'):
                        return
                    step = follow(path, ptarget)
                    if step is not None:
                        yield step
                if ul == target_domain.node[2]:
                    goal_key = (rectangle, point)
                    goal_paths = goal_target_cache.get(goal_key)
                    if goal_paths is None:
                        if (
                            _cpp_mod is not None
                            and project_target is _ORIG_PROJECT_TARGET
                            and type(geometry.rectangles) is tuple
                            and (len(effective_target_regions) < 256 or "_watchdog_check" not in self.__dict__)
                        ):
                            t_grid_rects = tuple(
                                (x0 // grid, y0 // grid, x1 // grid + 1, y1 // grid + 1)
                                for _atom, (x0, y0, x1, y1) in effective_target_regions
                            )
                            goal_paths = _cpp_mod.batch_project_goal_targets_cpp(
                                geometry.rectangles,
                                rectangle,
                                t_grid_rects,
                                point[0],
                                point[1],
                            )
                        else:
                            seen_goal_paths = set()
                            gpaths = []
                            for ordinal, (_atom, (x0, y0, x1, y1)) in enumerate(effective_target_regions):
                                if ordinal & 255 == 255:
                                    self._watchdog_check(net, "planar_region_goal_targets")
                                target = (x0 // grid, y0 // grid, x1 // grid + 1, y1 // grid + 1)
                                for path in project_target(geometry, rectangle, point, target):
                                    if path not in seen_goal_paths:
                                        seen_goal_paths.add(path)
                                        gpaths.append(path)
                            goal_paths = tuple(gpaths)
                        goal_target_cache[goal_key] = goal_paths
                    for path in goal_paths:
                        if not charge_event('goal_target'):
                            return
                        step = follow(path, rectangle)
                        if step is not None:
                            yield step
                for other in adjacent_layers(ul):
                    if rect_via_saturated(ul, other, rectangle):
                        continue
                    yield (*u, other), 0, None

            best_goal: Optional[Tuple[int, State]] = None

            while True:
                rf_top = (
                    _cpp_via_projector.frontier_top_f()
                    if _use_cpp_region_frontier
                    else (region_frontier[0][0] if region_frontier else None)
                )
                if not (pq or start_frontier or rf_top is not None):
                    if (
                        best_goal is None
                        and pattern_incumbent is None
                        and restrict_rect_entries
                        and saturation_pruned
                    ):
                        # Spatial/rectangle entry caps are an incomplete acceleration.
                        # Never conclude infeasibility ("exhausted") when any entry
                        # was pruned by saturation; rerun unpruned within the same
                        # declared expansion/event/solve budgets.
                        restrict_rect_entries = False
                        saturation_pruned = False
                        dist.clear()
                        prev.clear()
                        entry_via.clear()
                        pq.clear()
                        region_frontier.clear()
                        if _use_cpp_region_frontier:
                            _cpp_via_projector.clear_frontier()
                        rect_entry_expansions.clear()
                        rect_via_expansions.clear()
                        region_seq = 0
                        start_frontier[:] = initial_start_frontier
                        if not start_frontier:
                            break
                    else:
                        break
                queue_lower_bound = min(
                    pq[0][0] if pq else float("inf"),
                    start_frontier[0][0] if start_frontier else float("inf"),
                    rf_top if rf_top is not None else float("inf"),
                )
                if (
                    pattern_incumbent is not None
                    and queue_lower_bound >= pattern_incumbent.score
                    and (
                        best_goal is None
                        or best_goal[0] >= pattern_incumbent.score
                    )
                ):
                    self._solve_expansions_total = getattr(
                        self, "_solve_expansions_total", 0
                    ) + expansions
                    return incumbent_result()
                if best_goal is not None and queue_lower_bound > best_goal[0]:
                    self._solve_expansions_total = getattr(
                        self, "_solve_expansions_total", 0
                    ) + expansions
                    return self._reconstruct(
                        net,
                        best_goal[1],
                        prev,
                        entry_via,
                        goals,
                        reverse_search=search_reversed,
                    )
                if (
                    start_frontier
                    and start_frontier[0][0] == queue_lower_bound
                    and (
                        not pq
                        or start_frontier[0][0] <= pq[0][0]
                    )
                ):
                    _lower_f, s3, _access_key, start_index = heapq.heappop(
                        start_frontier
                    )
                    via = starts[start_index][1]
                    base = exact_terminal_cost(via)
                    s0: State = (s3[0], s3[1], s3[2], 0)
                    if base < dist.get(s0, float("inf")):
                        dist[s0] = base
                        prev[s0] = None
                        entry_via[s0] = via
                        heapq.heappush(pq, (base + hmin(*s3), 0, s0, base))
                    continue
                if (
                    rf_top is not None
                    and rf_top == queue_lower_bound
                    and (
                        not pq
                        or rf_top < pq[0][0]
                        or pq[0][1] >= 1
                    )
                ):
                    if _use_cpp_region_frontier:
                        popped = _cpp_via_projector.frontier_pop(
                            dist,
                            restrict_rect_entries,
                            _cpp_other_caps if restrict_rect_entries else None,
                        )
                        if popped is False:
                            continue
                        if popped is True:
                            saturation_pruned = True
                            continue
                        u, r_du, r_path, r_dest, pop_sat = popped
                        if pop_sat:
                            saturation_pruned = True
                        r_kind = 'via_target'
                    else:
                        _rlb_f, _rseq, r_kind, u, r_du, r_path, r_dest = heapq.heappop(
                            region_frontier
                        )
                        if r_du > dist.get(u, float("inf")):
                            continue
                        if restrict_rect_entries and r_dest < -1:
                            r_other = -r_dest - 2
                            if rect_entry_cap(r_other) is not None:
                                epx, epy = r_path[-1]
                                if _region_via_target_prunable(
                                    r_path, grid=target_domain.grid, layer=u[2],
                                    other_layer=r_other, goal_set=goal_set,
                                    target_domain=target_domain,
                                    destination_owners=event_graphs[r_other][2].query((epx, epy, epx, epy)),
                                    rect_saturated=rect_saturated,
                                    via_successor_last_dir=self._via_successor_last_dir,
                                    protect_goal=neg_bucket_dedup,
                                ):
                                    continue
                    if charge_event(r_kind):
                        if _use_cpp_region_step:
                            r_step = _cpp_advance_step(u, r_path, r_dest)
                        else:
                            r_step = _advance_planar_region_step(
                                u, r_path, r_dest, route_world=self, net=net,
                                grid=target_domain.grid, xs=xs, ys=ys, turn_cost=turn_cost,
                                charge_event=charge_event, event_run_cost=event_run_cost,
                                settle_prefix=_region_event_geometry_mod.settled_axis_prefix,
                            )
                        if r_step is not None:
                            rv, r_cost, _rvia = r_step
                            rnd = r_du + r_cost
                            if rnd < dist.get(rv, float("inf")):
                                dist[rv] = rnd
                                prev[rv] = u
                                entry_via[rv] = None
                                heapq.heappush(pq, (rnd + state_hmin(rv), 0, rv, rnd))
                    if event_budget_stopped:
                        self._solve_expansions_total += expansions
                        if best_goal is not None:
                            return self._reconstruct(
                                net, best_goal[1], prev, entry_via, goals,
                                reverse_search=search_reversed,
                            )
                        if pattern_incumbent is not None:
                            return incumbent_result()
                        kind = ("solve_budget" if self._solve_expansions_total >= self._solve_expansion_budget
                                else "budget")
                        effective_edge_cap = event_budget if expansions < budget else budget
                        self._route_failure_details += (("planar_region_event_work", event_work),
                            ("planar_region_event_work_by_kind", tuple(sorted(event_work_by_kind.items()))))
                        self._edge_fail = (kind, expansions, effective_edge_cap, solve_work_before_edge,
                                           self._solve_expansion_budget, len(xs), len(ys),
                                           len(allowed), "planar_region_events")
                        return None
                    continue
                if expansions % 1024 == 0:
                    publish_watchdog_details()
                    self._watchdog_check(net, "astar")
                _f, _deferred, u, queued_g = heapq.heappop(pq)
                ux, uy, ul, ulast = u[:4]
                du = dist.get(u, float("inf"))
                if queued_g != du:
                    continue
                if (
                    pattern_incumbent is not None
                    and _f >= pattern_incumbent.score
                    and (
                        best_goal is None
                        or best_goal[0] >= pattern_incumbent.score
                    )
                ):
                    self._solve_expansions_total = getattr(
                        self, "_solve_expansions_total", 0
                    ) + expansions
                    return incumbent_result()
                if best_goal is not None and _f > best_goal[0]:
                    self._solve_expansions_total = getattr(
                        self, "_solve_expansions_total", 0) + expansions
                    return self._reconstruct(
                        net,
                        best_goal[1],
                        prev,
                        entry_via,
                        goals,
                        reverse_search=search_reversed,
                    )
                goal3 = (ux, uy, ul)
                symbolic_goal = (event_graph is not None and len(u) == 10
                                 and bool(target_domain.owning_atoms(goal3)))
                if len(u) != 11 and (goal3 in goal_set or symbolic_goal):
                    pending_cost = (
                        event_run_cost(ul, (u[7], u[8]), (ux, uy))
                        if len(u) > 4 and u[9]
                        else 0
                    )
                    if pending_cost is not None:
                        candidate = (du + pending_cost + exact_terminal_cost(goals.get(goal3)), u)
                        if best_goal is None or candidate < best_goal:
                            best_goal = candidate
                if restrict_rect_entries and len(u) >= 10:
                    b_dedup = neg_bucket_dedup
                    settled_in_rect = (not u[5]) and (
                        u[6] >= -1 if b_dedup else u[6] == -1
                    )
                    if len(u) == 10:
                        if settled_in_rect:
                            if rect_saturated(ul, u[4], ux, uy, ulast):
                                continue
                        elif u[6] >= 0 and u[6] != u[4]:
                            if b_dedup and u[5]:
                                fx = u[5][-1][0] * target_domain.grid
                                fy = u[5][-1][1] * target_domain.grid
                                prev_py = u[5][-2][1] if len(u[5]) >= 2 else (uy // target_domain.grid)
                                end_vdir = 1 if u[5][-1][1] == prev_py else 2
                                if rect_saturated(ul, u[6], fx, fy, end_vdir):
                                    continue
                            elif not b_dedup and rect_saturated(ul, u[6]):
                                continue
                    elif len(u) == 11 and (not u[5]) and u[6] == -1:
                        if rect_via_saturated(ul, u[10], u[4]):
                            continue
                if (expansions >= budget or expansions + event_work >= event_budget
                        or event_budget_stopped
                        or self._solve_expansions_total + expansions >= self._solve_expansion_budget):
                    if (
                        best_goal is not None
                        and (
                            pattern_incumbent is None
                            or best_goal[0] < pattern_incumbent.score
                        )
                    ):
                        self._solve_expansions_total = getattr(
                            self, "_solve_expansions_total", 0
                        ) + expansions
                        return self._reconstruct(
                            net,
                            best_goal[1],
                            prev,
                            entry_via,
                            goals,
                            reverse_search=search_reversed,
                        )
                    if pattern_incumbent is not None:
                        self._solve_expansions_total = getattr(
                            self, "_solve_expansions_total", 0
                        ) + expansions
                        return incumbent_result()
                    solve_exhausted = (
                        self._solve_expansions_total + expansions
                        >= self._solve_expansion_budget
                    )
                    effective_edge_cap = (
                        event_budget
                        if (event_graph is not None and expansions < budget)
                        else budget
                    )
                    self._edge_fail = (
                        "solve_budget" if solve_exhausted else "budget",
                        expansions,
                        effective_edge_cap,
                        solve_work_before_edge if event_graph is not None else self._solve_expansions_total,
                        self._solve_expansion_budget,
                        len(xs),
                        len(ys),
                        len(allowed),
                    )
                    if event_graph is not None:
                        self._edge_fail += ("planar_region_events",)
                        self._route_failure_details += (("planar_region_event_work", event_work),
                            ("planar_region_event_work_by_kind", tuple(sorted(event_work_by_kind.items()))))
                    self._solve_expansions_total = getattr(
                        self, "_solve_expansions_total", 0) + expansions
                    return None
                if restrict_rect_entries and len(u) >= 10:
                    settled_in_rect = (not u[5]) and (
                        u[6] >= -1 if neg_bucket_dedup else u[6] == -1
                    )
                    if len(u) == 10 and settled_in_rect:
                        rkey = (
                            (
                                ul,
                                u[4],
                                int(ulast),
                                ux // rect_bucket_nm,
                                uy // rect_bucket_nm,
                            )
                            if neg_bucket_dedup
                            else (ul, u[4], 0, 0, 0)
                        )
                        rect_entry_expansions[rkey] = rect_entry_expansions.get(rkey, 0) + 1
                        if _use_cpp_region_frontier:
                            _cpp_via_projector.inc_rect_entry(
                                rkey[0], rkey[1], rkey[2], rkey[3], rkey[4]
                            )
                    elif len(u) == 11 and (not u[5]) and u[6] == -1 and not neg_bucket_dedup:
                        vkey = (ul, u[10], u[4])
                        rect_via_expansions[vkey] = rect_via_expansions.get(vkey, 0) + 1
                expansions += 1
                for v, step_cost, via_lo in successors(u):
                    nd = du + step_cost
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        prev[v] = u
                        if via_lo is None:
                            entry_via[v] = None
                        elif isinstance(via_lo, tuple):
                            entry_via[v] = _ViaSite(ux, uy, via_lo[0], via_lo[1])
                        else:
                            entry_via[v] = _ViaSite(
                                ux, uy, via_lo,
                                (via_model_by_edge.get(via_edge_id(ux, uy, via_lo), 0)
                                 if ux in xi and uy in yi else 0),
                            )
                        heapq.heappush(pq, (nd + state_hmin(v), int(len(v) == 11), v, nd))
                if event_budget_stopped:
                    self._solve_expansions_total += expansions
                    if best_goal is not None:
                        return self._reconstruct(
                            net, best_goal[1], prev, entry_via, goals,
                            reverse_search=search_reversed,
                        )
                    if pattern_incumbent is not None:
                        return incumbent_result()
                    kind = ("solve_budget" if self._solve_expansions_total >= self._solve_expansion_budget
                            else "budget")
                    self._route_failure_details += (("planar_region_event_work", event_work),
                        ("planar_region_event_work_by_kind", tuple(sorted(event_work_by_kind.items()))))
                    effective_edge_cap = event_budget if expansions < budget else budget
                    self._edge_fail = (kind, expansions, effective_edge_cap, solve_work_before_edge,
                                       self._solve_expansion_budget, len(xs), len(ys),
                                       len(allowed), "planar_region_events")
                    return None
            if best_goal is not None:
                if (
                    pattern_incumbent is not None
                    and best_goal[0] >= pattern_incumbent.score
                ):
                    self._solve_expansions_total = getattr(
                        self, "_solve_expansions_total", 0
                    ) + expansions
                    return incumbent_result()
                self._solve_expansions_total = getattr(
                    self, "_solve_expansions_total", 0
                ) + expansions
                return self._reconstruct(
                    net,
                    best_goal[1],
                    prev,
                    entry_via,
                    goals,
                    reverse_search=search_reversed,
                )
            if pattern_incumbent is not None:
                self._solve_expansions_total = getattr(
                    self, "_solve_expansions_total", 0
                ) + expansions
                return incumbent_result()
            # queue exhausted: NO legal path exists in this edge's graph (sealed) --
            # diagnostic surface only (read by debug logging; never by logic).
            self._edge_fail = (
                "exhausted",
                expansions,
                len(xs),
                len(ys),
                len(allowed),
                self._hard_rejection_evidence(),
            )
            self._solve_expansions_total = getattr(
                self, "_solve_expansions_total", 0) + expansions
            return None
        finally:
            if getattr(self, "_active_edge_ctx", None) is not None:
                self._flush_active_edge_ctx()
            if prev_ctx is None:
                self.__dict__.pop("_active_edge_ctx", None)
                self.__dict__.pop("_active_edge_ctx_net", None)
                self.__dict__.pop("_active_edge_ctx_cpp", None)
                self.__dict__.pop("_active_edge_ctx_gen_bumped", None)
                self.__dict__.pop("_active_edge_ctx_bypass", None)
            else:
                self._active_edge_ctx = prev_ctx
                self._active_edge_ctx_net = prev_net
                self._active_edge_ctx_cpp = prev_cpp
                self._active_edge_ctx_gen_bumped = prev_bumped
                self._active_edge_ctx_bypass = prev_bypass
            if prev_hgrid is None:
                self.__dict__.pop("_active_history_grid", None)
            else:
                self._active_history_grid = prev_hgrid

    def _reconstruct(
        self,
        net,
        goal_state,
        prev,
        entry_via,
        goals,
        *,
        reverse_search: bool = False,
    ):
        state_chain = []
        cur = goal_state
        root = None
        while cur is not None:
            state_chain.append(cur)
            root = cur
            cur = prev[cur]
        state_chain.reverse()
        if reverse_search:
            state_chain.reverse()
        chain: List[Tuple[int, int, int]] = [
            (state[0], state[1], state[2]) for state in state_chain
        ]
        segs: List[_WireSeg] = []
        vias: List[_ViaSite] = []
        # entry escape via (pin -> start landing), if the path began on an escape.
        if reverse_search:
            ev = goals.get((goal_state[0], goal_state[1], goal_state[2]))
            gv = entry_via.get(root) if root is not None else None
        else:
            ev = entry_via.get(root) if root is not None else None
            gv = goals.get((goal_state[0], goal_state[1], goal_state[2]))
        if ev is not None:
            vias.append(ev)
        for i in range(1, len(chain)):
            ax, ay, al = chain[i - 1]
            bx, by, bl = chain[i]
            if al != bl:
                transition_state = (
                    state_chain[i - 1] if reverse_search else state_chain[i]
                )
                transition_via = entry_via.get(transition_state)
                vias.append(
                    transition_via
                    if transition_via is not None
                    else _ViaSite(ax, ay, min(al, bl))
                )
            elif (ax, ay) != (bx, by):
                segs.append(self._wire_segment(net, al, ax, ay, bx, by))
        # exit escape via (goal landing -> pin), if the goal was an escape landing.
        if gv is not None:
            vias.append(gv)
        if root is not None:
            self._record_selected_access_pair(
                ((chain[0][0], chain[0][1], chain[0][2]), ev),
                ((chain[-1][0], chain[-1][1], chain[-1][2]), gv),
            )
        return segs, vias

    # ------------------------------------------------------------- MST fan-out
    @staticmethod
    def _mst_edges(nodes: Sequence[Tuple[int, int, int]]):
        n = len(nodes)
        if n == 2:
            return [(0, 1)]
        in_tree = [False] * n
        in_tree[0] = True
        edges = []

        def d(i, j):
            return (abs(nodes[i][0] - nodes[j][0]) + abs(nodes[i][1] - nodes[j][1])
                    + abs(nodes[i][2] - nodes[j][2]) * 1000)

        for _ in range(n - 1):
            best = None
            best_d = 1 << 62
            for i in range(n):
                if not in_tree[i]:
                    continue
                for j in range(n):
                    if in_tree[j]:
                        continue
                    dd = d(i, j)
                    if dd < best_d:
                        best_d = dd
                        best = (i, j)
            i, j = best
            in_tree[j] = True
            edges.append((i, j))
        return edges

    def _contract_owned_component_endpoints(
        self,
        net: str,
        nodes: Sequence[Tuple[int, int, int]],
        pins: Sequence[Optional[Pin]],
    ) -> Tuple[
        List[Tuple[int, int, int]],
        List[Optional[Pin]],
        List[Tuple[Tuple[Tuple[str, Any], ...], ...]],
    ]:
        """Contract endpoints already joined by one exact owned-metal component.

        The complete canonical atom tuple is the equivalence authority.  A bbox,
        terminal ref, cached component, resource-plan fact, or placement fact is
        insufficient.  Endpoints without one authorized component remain distinct,
        even when their coordinates coincide.  The returned member evidence keeps
        every contracted terminal named deterministically while the MST sees only
        physical connected components.
        """
        endpoint_records = []
        for endpoint_index, (raw_node, pin) in enumerate(zip(nodes, pins)):
            if endpoint_index % 64 == 0:
                self._watchdog_check(net, "component_endpoint_contraction")
            node = tuple(int(value) for value in raw_node)
            component = self._owned_terminal_component(net, *node)
            endpoint_key = (
                int(node[0]),
                int(node[1]),
                int(node[2]),
                "" if pin is None else str(pin.device),
                "" if pin is None else str(pin.terminal),
            )
            atom_key = tuple(sorted(
                _owned_terminal_atom_key(atom) for atom in component
            ))
            # The ordinal makes every unauthorized endpoint a singleton.  It is
            # deliberately absent from ordering/evidence, so input reorder cannot
            # perturb distinguishable endpoints' canonical output.
            group_key = (
                ("owned_component", atom_key)
                if atom_key
                else ("endpoint", endpoint_index)
            )
            endpoint_records.append((
                group_key,
                endpoint_key,
                node,
                pin,
                self._failure_endpoint(pin, node),
            ))

        grouped: Dict[Tuple[Any, ...], List[Tuple[Any, ...]]] = {}
        for record_index, record in enumerate(endpoint_records):
            if record_index % 64 == 0:
                self._watchdog_check(net, "component_endpoint_contraction")
            grouped.setdefault(record[0], []).append(record)

        physical_endpoints = []
        for group_index, members in enumerate(grouped.values()):
            if group_index % 64 == 0:
                self._watchdog_check(net, "component_endpoint_contraction")
            ordered_members = tuple(sorted(members, key=lambda item: item[1]))
            representative = ordered_members[0]
            physical_endpoints.append((
                representative[1],
                representative[2],
                representative[3],
                tuple(member[4] for member in ordered_members),
            ))
        physical_endpoints.sort(key=lambda item: item[0])
        return (
            [item[1] for item in physical_endpoints],
            [item[2] for item in physical_endpoints],
            [item[3] for item in physical_endpoints],
        )

    def _rebuild_metal_index(self) -> None:
        """Re-index the committed foreign metal per gi from the book. Own-net metal
        is excluded from legality (owner==net), and MST edges connect through shared
        pin nodes -- so a net never needs to SEE its own in-progress metal, and the
        index is a per-net-attempt snapshot of the committed nets only."""
        self._committed_index_generation = int(getattr(
            self, "_committed_index_generation", 0
        )) + 1
        self._component_access.committed_world_changed()
        self._metal_idx = {}
        for gi, rects in self._book.metal.items():
            if not rects:
                continue
            idx = _SpatialIndex()
            for (owner, r) in rects:
                idx.add(owner, r)
            self._metal_idx[gi] = idx
        self._via_pad_idx = {}
        via_pad_provenance: Dict[
            int, Set[Tuple[Optional[str], Rect]]
        ] = {}
        for gi, rects in self._book.via_pads.items():
            if not rects:
                continue
            idx = _SpatialIndex()
            for (owner, r) in rects:
                idx.add(owner, r)
                via_pad_provenance.setdefault(gi, set()).add(
                    (owner, tuple(_norm(r)))
                )
            self._via_pad_idx[gi] = idx
        self._via_pad_provenance = {
            gi: frozenset(claims)
            for gi, claims in via_pad_provenance.items()
        }
        self._cut_idx = {}
        for owner, r, lo in self._book.cuts:
            idx = self._cut_idx.get(lo)
            if idx is None:
                self._cut_idx[lo] = idx = _SpatialIndex()
            idx.add(owner, r)

    def _add_attempt_cut(self, v: _ViaSite) -> None:
        ext = self._via_cut_extent(v.lo, self._site_n(v))
        self._attempt_cuts.setdefault(v.lo, []).append(
            (v.x - ext, v.y - ext, v.x + ext, v.y + ext))

    @staticmethod
    def _access_state_key(
        item: Tuple[Tuple[int, int, int], Optional[_ViaSite]],
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        state, via = item
        if via is None:
            return (*state, 0, 0, 0, 0, 0)
        return (*state, 1, via.x, via.y, via.lo, via.n)

    @staticmethod
    def _access_state_evidence(
        item: Tuple[Tuple[int, int, int], Optional[_ViaSite]],
    ) -> Tuple[Tuple[str, Any], ...]:
        state, via = item
        return (
            ("state", tuple(int(value) for value in state)),
            (
                "via_site",
                (
                    None
                    if via is None
                    else (int(via.x), int(via.y), int(via.lo), int(via.n))
                ),
            ),
        )

    def _record_selected_access_pair(
        self,
        start: Tuple[Tuple[int, int, int], Optional[_ViaSite]],
        goal: Tuple[Tuple[int, int, int], Optional[_ViaSite]],
    ) -> None:
        """Publish exact endpoint identities selected by one combined search."""
        self._selected_access_pair = (
            self._access_state_key(start),
            self._access_state_key(goal),
        )

    def _endpoint_access_vias_cut_clear(
        self,
        left: Optional[_ViaSite],
        right: Optional[_ViaSite],
    ) -> bool:
        """Whether two mandatory endpoint vias can coexist in one route.

        Endpoint access is validated independently against the immutable cut
        ledger before the multi-source search.  That does not prove that the
        selected start and goal vias are mutually clear.  Keep this predicate
        exactly aligned with ``_candidate_via_group_conflicts``' cut branch;
        landing-pad gaps are deliberately excluded because the returned route
        metal may legally cover those gaps.
        """
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        if left is None or right is None:
            return True
        left_identity = self._via_search_identity(
            left.x, left.y, left.lo, self._site_n(left)
        )
        right_identity = self._via_search_identity(
            right.x, right.y, right.lo, self._site_n(right)
        )
        if left_identity == right_identity:
            return True
        if int(left.lo) != int(right.lo):
            return True
        left_ext = self._via_cut_extent(left.lo, self._site_n(left))
        right_ext = self._via_cut_extent(right.lo, self._site_n(right))
        left_cut = (
            left.x - left_ext,
            left.y - left_ext,
            left.x + left_ext,
            left.y + left_ext,
        )
        right_cut = (
            right.x - right_ext,
            right.y - right_ext,
            right.x + right_ext,
            right.y + right_ext,
        )
        return left_cut == right_cut or _rects_clear(
            left_cut,
            right_cut,
            self._via_space[left.lo],
        )

    def _candidate_via_group_conflicts(
        self,
        vias: Sequence[_ViaSite],
        net: str,
        segs: Sequence[_WireSeg] = (),
    ) -> Tuple[
        Tuple[
            Tuple[int, int, int, int],
            Tuple[int, int, int, int],
        ],
        ...,
    ]:
        """Exact pairwise cut or landing-pad conflicts in one returned route.

        The path search checks each via against the immutable attempt ledger, but
        it does not carry all earlier path cuts in its A* state.  Atomic validation
        can therefore discover a same-path conflict only after reconstruction.
        Return only the first canonical physical disjunction needed for a complete
        conflict-directed retry; its two branches cover every replacement path, so
        collecting later pairs would add quadratic work without information.  Do
        not reinterpret it as endpoint-pair infeasibility.
        """
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        ordered = tuple(sorted(
            {
                self._via_search_identity(
                    via.x,
                    via.y,
                    via.lo,
                    self._site_n(via),
                ): via
                for via in vias
            }.items()
        ))
        ordered_vias = [via for _key, via in ordered]
        pad_conflicts = self._pattern_via_pad_visible_conflicts(
            net, ordered_vias, segs
        )
        if pad_conflicts:
            self._watchdog_check(net, "candidate_via_group_conflicts")
            left_index, right_index, _gi = pad_conflicts[0]
            return ((ordered[left_index][0], ordered[right_index][0]),)

        comparisons = 0
        for left_index, (left_key, left) in enumerate(ordered):
            left_ext = self._via_cut_extent(
                left.lo, self._site_n(left)
            )
            left_cut = (
                left.x - left_ext,
                left.y - left_ext,
                left.x + left_ext,
                left.y + left_ext,
            )
            for right_key, right in ordered[left_index + 1:]:
                if comparisons % 1024 == 0:
                    self._watchdog_check(
                        net, "candidate_via_group_conflicts"
                    )
                comparisons += 1
                if int(left.lo) != int(right.lo):
                    continue
                right_ext = self._via_cut_extent(
                    right.lo, self._site_n(right)
                )
                right_cut = (
                    right.x - right_ext,
                    right.y - right_ext,
                    right.x + right_ext,
                    right.y + right_ext,
                )
                if (
                    left_cut != right_cut
                    and not _rects_clear(
                        left_cut,
                        right_cut,
                        self._via_space[left.lo],
                    )
                ):
                    return ((left_key, right_key),)
        return ()

    def _route_access_multi_transaction(
        self,
        starts: Sequence[Tuple[Tuple[int, int, int], Optional[_ViaSite]]],
        goals: Sequence[Tuple[Tuple[int, int, int], Optional[_ViaSite]]],
        net: str,
        attempt,
        *,
        prevalidated: bool = False,
        materialized_access_states: Sequence[Tuple[int, int, int]] = (),
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """One virtual multi-source/goal search for an expanded access domain.

        More than two alternatives on either endpoint can only come from exact
        component/event access (the local domain is pin + at most one escape).
        Running the full search once per Cartesian pair makes a C-atom terminal pay
        O(C^2) searches outside the solve budget.  Instead, validate each optional
        access independently against the base cut ledger, pass every legal state to
        the existing multi-source/multi-goal search once, then atomically validate
        only the vias it selected.  A returned internal-via conflict first branches
        on the exact physical disjunction (forbid its left site OR its right site)
        and retries the same selected endpoint pair.  Only a conflict between two
        mandatory endpoint vias proves that pair infeasible.  The remaining endpoint
        Cartesian domain is partitioned around the ACTUALLY selected pair into two
        disjoint residual domains.  Breadth-first replay remains one search in the
        ordinary no-conflict case; all finite conflict branches remain represented
        until success or the typed watchdog.

        A complete-union MIM failure additionally supplies candidate-scoped
        diversification.  It does not prove a site physically forbidden: a new
        complete path can retain the site and remove its gap with a real bridge.
        Static legality caches never receive these contextual exclusions.
        """
        self._last_accepted_access_states = None
        self._last_accepted_terminal_probes = None
        self._last_accepted_planar = (False, False)
        self._last_accepted_group_source_union = False
        self._last_accepted_group_source_candidate = None
        materialized_access_states = tuple(
            tuple(int(value) for value in state)
            for state in materialized_access_states
        )
        base_cuts = {
            int(lo): list(cuts)
            for lo, cuts in self._attempt_cuts.items()
        }
        base_forbidden = set(getattr(
            self, "_search_forbidden_via_sites", set()
        ))
        base_forbidden_wires = set(getattr(
            self, "_search_forbidden_wire_edges", set()
        ))
        base_details = tuple(getattr(self, "_route_failure_details", ()))
        absent_static_evidence = object()
        prior_static_evidence = getattr(
            self, "_access_static_disconnection_evidence", absent_static_evidence
        )
        aggregate_rejections: Dict[Tuple[str, str], int] = dict(
            getattr(self, "_hard_rejections", {})
        )
        aggregate_samples = {
            cause: (int(count), tuple(extent))
            for cause, (count, extent) in getattr(
                self, "_hard_rejection_samples", {}
            ).items()
        }
        access_failures = []
        query_stats = self._component_access.stats
        access_progress: Dict[str, Any] = {
            "active_forbidden_site_count": 0,
            "active_forbidden_wire_count": 0,
            "active_goal_count": 0,
            "active_source_count": 0,
            "attempt_count": 0,
            "candidate_failure_count": 0,
            "candidate_segment_count": 0,
            "candidate_via_count": 0,
            "candidate_via_cursor": 0,
            "component_access_via_cache_hits_total": int(
                query_stats.site_hits
            ),
            "component_access_via_cache_misses_total": int(
                query_stats.site_misses
            ),
            "component_access_partition_cache_hits_total": int(
                query_stats.partition_hits
            ),
            "component_access_partition_cache_misses_total": int(
                query_stats.partition_misses
            ),
            "component_escape_state_cache_hits_total": int(
                query_stats.escape_hits
            ),
            "component_escape_state_cache_misses_total": int(
                query_stats.escape_misses
            ),
            "via_static_legality_cache_hits_total": int(
                self._via_static_legality_cache_hits_total
            ),
            "via_static_legality_cache_misses_total": int(
                self._via_static_legality_cache_misses_total
            ),
            "wire_static_legality_cache_hits_total": int(
                self._wire_static_legality_cache_hits_total
            ),
            "wire_static_legality_cache_misses_total": int(
                self._wire_static_legality_cache_misses_total
            ),
            "domain_count": 0,
            "domain_cursor": 0,
            "endpoint_pair_count": 0,
            "endpoint_pair_cut_conflict_count": 0,
            "goal_access_count": len(goals),
            "initial_domain_count": 0,
            "last_branch_site_count": 0,
            "last_branch_wire_count": 0,
            "last_candidate_failure_kind": "none",
            "last_group_conflict_count": 0,
            "last_residual_goal_count": 0,
            "last_residual_source_count": 0,
            "potential_transform_switches_total": int(
                self._potential_transform_switches_total
            ),
            "potential_transform_cache_hits_total": int(
                self._potential_transform_cache_hits_total
            ),
            "potential_transform_cache_misses_total": int(
                self._potential_transform_cache_misses_total
            ),
            "prevalidated": bool(prevalidated),
            "prevalidation_cursor": 0,
            "prevalidation_endpoint": "none",
            "prevalidation_total": len(starts) + len(goals),
            "source_access_count": len(starts),
            "subphase": "initialize",
        }

        def publish_access_progress(subphase: str, **updates: Any) -> None:
            query_stats = self._component_access.stats
            access_progress.update(updates)
            access_progress["subphase"] = str(subphase)
            access_progress["potential_transform_switches_total"] = int(
                self._potential_transform_switches_total
            )
            access_progress["potential_transform_cache_hits_total"] = int(
                self._potential_transform_cache_hits_total
            )
            access_progress["potential_transform_cache_misses_total"] = int(
                self._potential_transform_cache_misses_total
            )
            access_progress["component_access_via_cache_hits_total"] = int(
                query_stats.site_hits
            )
            access_progress["component_access_via_cache_misses_total"] = int(
                query_stats.site_misses
            )
            access_progress[
                "component_access_partition_cache_hits_total"
            ] = int(query_stats.partition_hits)
            access_progress[
                "component_access_partition_cache_misses_total"
            ] = int(query_stats.partition_misses)
            access_progress[
                "component_escape_state_cache_hits_total"
            ] = int(query_stats.escape_hits)
            access_progress[
                "component_escape_state_cache_misses_total"
            ] = int(query_stats.escape_misses)
            access_progress[
                "via_static_legality_cache_hits_total"
            ] = int(self._via_static_legality_cache_hits_total)
            access_progress[
                "via_static_legality_cache_misses_total"
            ] = int(self._via_static_legality_cache_misses_total)
            access_progress[
                "wire_static_legality_cache_hits_total"
            ] = int(self._wire_static_legality_cache_hits_total)
            access_progress[
                "wire_static_legality_cache_misses_total"
            ] = int(self._wire_static_legality_cache_misses_total)
            self._access_watchdog_details = tuple(sorted(
                access_progress.items(), key=lambda item: item[0]
            ))

        # This transaction owns the endpoint-selection outcome.  In particular,
        # an empty legal domain must not inherit a solve-budget STOP from an
        # earlier sparse/refined replay.
        self._edge_fail = None
        endpoint_domains = getattr(
            self, "_active_endpoint_access_domains", (None, None)
        )
        if (not isinstance(endpoint_domains, tuple) or len(endpoint_domains) != 2
                or any(domain is not None and not isinstance(domain, _PlanarEndpointDomain)
                       for domain in endpoint_domains)):
            raise TypeError("endpoint access domains must contain two planar domains")

        def has_planar(domain):
            return domain is not None and domain.singleton is None and bool(domain.regions)

        def selected_member(items, domain, key):
            matches = tuple(item for item in items if self._access_state_key(item) == key)
            if len(matches) == 1:
                is_locked_planar = bool(
                    domain is not None and domain.locked_planar
                    and domain.singleton == tuple(matches[0][0])
                    and matches[0][1] is None
                )
                return matches[0], is_locked_planar
            if matches:
                return None, False
            if (has_planar(domain) and isinstance(key, tuple) and len(key) == 8
                    and all(type(value) is int for value in key)
                    and tuple(key[3:]) == (0, 0, 0, 0, 0)
                    and domain.owning_atoms(tuple(key[:3]))):
                return (tuple(key[:3]), None), True
            return None, False

        def merge_rejections() -> None:
            for key, count in self._hard_rejections.items():
                aggregate_rejections[key] = (
                    aggregate_rejections.get(key, 0) + int(count)
                )
            for cause, (count, extent) in self._hard_rejection_samples.items():
                previous = aggregate_samples.get(cause)
                if previous is None:
                    aggregate_samples[cause] = (int(count), tuple(extent))
                else:
                    old_count, old_extent = previous
                    aggregate_samples[cause] = (
                        int(old_count) + int(count),
                        (
                            min(old_extent[0], extent[0]),
                            min(old_extent[1], extent[1]),
                            max(old_extent[2], extent[2]),
                            max(old_extent[3], extent[3]),
                        ),
                    )

        def legal_access(items, endpoint):
            legal = []
            for index, item in enumerate(items):
                if index % 64 == 0:
                    publish_access_progress(
                        "prevalidate_access",
                        prevalidation_endpoint=str(endpoint),
                        prevalidation_cursor=int(index),
                    )
                    self._watchdog_check(net, "access_selection")
                self._attempt_cuts = {
                    int(lo): list(cuts) for lo, cuts in base_cuts.items()
                }
                self._search_forbidden_via_sites = set(base_forbidden)
                self._hard_rejections = {}
                self._hard_rejection_samples = {}
                _state, via = item
                if via is None or self._via_legal(
                    via.x, via.y, via.lo, net, n=self._site_n(via)
                ):
                    legal.append(item)
                    continue
                access_failures.append((
                    ("endpoint", endpoint),
                    ("access", self._access_state_evidence(item)),
                    ("failure_kind", "pin_access_blocked"),
                    ("rejections", self._hard_rejection_evidence()),
                    ("blockers", self._hard_rejection_sample_evidence()),
                ))
                merge_rejections()
            publish_access_progress(
                "prevalidate_access",
                prevalidation_endpoint=str(endpoint),
                prevalidation_cursor=len(items),
            )
            return legal

        attempts = []
        try:
            publish_access_progress("initialize")
            self._watchdog_check(net, "access_selection")
            if prevalidated:
                # ``_pin_access`` / rail-goal enumeration immediately precedes
                # the internal transaction call and has already judged every
                # optional via against this exact base ledger.  Do not repeat
                # that O(access-domain) work; the winning route's vias are still
                # atomically revalidated below before any cut is committed.
                initial_starts = tuple(starts)
                initial_goals = tuple(goals)
            else:
                initial_starts = tuple(legal_access(starts, "source"))
                initial_goals = tuple(legal_access(goals, "goal"))

            # A virtual multi-source/multi-goal search can select any Cartesian
            # endpoint pair.  Independently legal access vias are not necessarily
            # mutually cut-clear; allowing such pairs into one domain makes the
            # post-search atomic validator reject them one-by-one (the measured
            # GF180 SUB blow-up was 223 x 890 endpoint alternatives).  Partition
            # the complete compatible relation into deterministic bicliques:
            # every pair in each search domain is cut-clear, every compatible
            # pair appears exactly once, and no candidate is truncated.  Cut
            # conflicts are local, so discover only neighbouring goal cuts with
            # a per-layer spatial index rather than scanning the full Cartesian
            # product (VOUT exposes 12,488 x 790 alternatives with zero nearby
            # conflicts).
            pair_count = len(initial_starts) * len(initial_goals)
            pair_conflicts = 0
            goal_cut_indices: Dict[int, _SpatialIndex] = {}
            for goal_index, (_goal_state, goal_via) in enumerate(
                initial_goals
            ):
                if goal_via is None:
                    continue
                goal_ext = self._via_cut_extent(
                    goal_via.lo, self._site_n(goal_via)
                )
                goal_cut = (
                    goal_via.x - goal_ext,
                    goal_via.y - goal_ext,
                    goal_via.x + goal_ext,
                    goal_via.y + goal_ext,
                )
                goal_cut_indices.setdefault(
                    int(goal_via.lo), _SpatialIndex()
                ).add(str(goal_index), goal_cut)

            conflicts_by_start: List[Set[int]] = [
                set() for _start in initial_starts
            ]
            conflicts_by_goal: List[Set[int]] = [
                set() for _goal in initial_goals
            ]
            local_pair_checks = 0
            for start_index, (_start_state, start_via) in enumerate(
                initial_starts
            ):
                if start_index > 0 and start_index % 256 == 0:
                    publish_access_progress(
                        "prepartition_endpoint_pairs",
                        endpoint_pair_count=pair_count,
                        endpoint_pair_cut_conflict_count=pair_conflicts,
                    )
                    self._watchdog_check(net, "access_selection")
                if start_via is None:
                    continue
                goal_index = goal_cut_indices.get(int(start_via.lo))
                if goal_index is None:
                    continue
                start_ext = self._via_cut_extent(
                    start_via.lo, self._site_n(start_via)
                )
                start_cut = (
                    start_via.x - start_ext,
                    start_via.y - start_ext,
                    start_via.x + start_ext,
                    start_via.y + start_ext,
                )
                search_gap = max(0, int(self._via_space[start_via.lo]) - 1)
                query = (
                    start_cut[0] - search_gap,
                    start_cut[1] - search_gap,
                    start_cut[2] + search_gap,
                    start_cut[3] + search_gap,
                )
                for raw_goal_index, _goal_cut in goal_index.query(query):
                    local_pair_checks += 1
                    if local_pair_checks % 4096 == 0:
                        publish_access_progress(
                            "prepartition_endpoint_pairs",
                            endpoint_pair_count=pair_count,
                            endpoint_pair_cut_conflict_count=pair_conflicts,
                        )
                        self._watchdog_check(net, "access_selection")
                    candidate_goal_index = int(raw_goal_index)
                    goal_via = initial_goals[candidate_goal_index][1]
                    if self._endpoint_access_vias_cut_clear(
                        start_via, goal_via
                    ):
                        continue
                    conflicts_by_start[start_index].add(candidate_goal_index)
                    conflicts_by_goal[candidate_goal_index].add(start_index)
                    pair_conflicts += 1

            starts_by_forbidden_goals: Dict[Tuple[int, ...], List[int]] = {}
            for start_index, conflicts in enumerate(conflicts_by_start):
                starts_by_forbidden_goals.setdefault(
                    tuple(sorted(conflicts)), []
                ).append(start_index)
            goals_by_forbidden_starts: Dict[Tuple[int, ...], List[int]] = {}
            for goal_index, conflicts in enumerate(conflicts_by_goal):
                goals_by_forbidden_starts.setdefault(
                    tuple(sorted(conflicts)), []
                ).append(goal_index)

            use_goal_groups = (
                len(goals_by_forbidden_starts)
                < len(starts_by_forbidden_goals)
            )
            if use_goal_groups:
                domains = [
                    (
                        tuple(
                            start
                            for index, start in enumerate(initial_starts)
                            if index not in forbidden_starts
                        ),
                        tuple(
                            initial_goals[index] for index in goal_indices
                        ),
                        frozenset(),
                    )
                    for forbidden_start_indices, goal_indices
                    in goals_by_forbidden_starts.items()
                    for forbidden_starts in (set(forbidden_start_indices),)
                    if len(forbidden_starts) < len(initial_starts)
                ]
            else:
                domains = [
                    (
                        tuple(
                            initial_starts[index] for index in start_indices
                        ),
                        tuple(
                            goal
                            for index, goal in enumerate(initial_goals)
                            if index not in forbidden_goals
                        ),
                        frozenset(),
                    )
                    for forbidden_goal_indices, start_indices
                    in starts_by_forbidden_goals.items()
                    for forbidden_goals in (set(forbidden_goal_indices),)
                    if len(forbidden_goals) < len(initial_goals)
                ]
            # Preserve the existing domains and their stop semantics. New wire
            # alternatives carry their own restrictions and local-budget scope.
            # Keep the existing finite-compatible bicliques and their order.
            # A symbolic domain is offered to each attempt but has no complete
            # coverage proof here, including if no finite biclique exists.
            domains = [(*domain, frozenset(), False, *endpoint_domains)
                       for domain in domains]
            seen_domains = set()
            neg_group_active = bool(
                getattr(self, "_active_group_route_local_closure", False)
                and getattr(self, "_negotiation_port_legalize", False)
            )

            def domain_identity(starts_, goals_, forbidden_, wires_, added_, source_, goal_):
                key = (
                    tuple(self._access_state_key(item) for item in starts_),
                    tuple(self._access_state_key(item) for item in goals_),
                    tuple(sorted(forbidden_)), tuple(sorted(wires_)),
                    source_, goal_,
                )
                return key + (bool(added_),) if neg_group_active else key

            queued_domains = ({domain_identity(*domain) for domain in domains}
                              if neg_group_active else set())
            publish_access_progress(
                "domain_ready",
                domain_count=len(domains),
                endpoint_pair_count=pair_count,
                endpoint_pair_cut_conflict_count=pair_conflicts,
                initial_domain_count=len(domains),
                prevalidation_cursor=(
                    0 if prevalidated else len(starts) + len(goals)
                ),
                prevalidation_endpoint=(
                    "skipped" if prevalidated else "goal"
                ),
            )

            def append_domain(
                domain_starts,
                domain_goals,
                forbidden,
                forbidden_wires,
                added_wire_branch,
                source_domain=None,
                goal_domain=None,
            ) -> None:
                if ((not domain_starts and not has_planar(source_domain))
                        or (not domain_goals and not has_planar(goal_domain))):
                    return
                key = domain_identity(domain_starts, domain_goals, forbidden,
                                      forbidden_wires, added_wire_branch,
                                      source_domain, goal_domain)
                if key in seen_domains or (neg_group_active and key in queued_domains):
                    return
                if neg_group_active:
                    queued_domains.add(key)
                domains.append((
                    tuple(domain_starts),
                    tuple(domain_goals),
                    frozenset(forbidden),
                    frozenset(forbidden_wires),
                    bool(added_wire_branch),
                    source_domain,
                    goal_domain,
                ))

            domain_cursor = 0
            terminal_failure = None
            pending_wire_budget_failure = None
            pending_group_source_union_unknown = False
            region_budget_work = {}
            region_budget_breakdowns = {}
            pending_symbolic_incomplete = any(
                has_planar(domain) for domain in endpoint_domains
            )
            while domain_cursor < len(domains):
                (
                    active_starts,
                    active_goals,
                    active_forbidden,
                    active_forbidden_wires,
                    added_wire_branch,
                    source_domain,
                    goal_domain,
                ) = domains[domain_cursor]
                domain_cursor += 1
                publish_access_progress(
                    "domain_key",
                    active_forbidden_site_count=len(active_forbidden),
                    active_forbidden_wire_count=len(active_forbidden_wires),
                    active_goal_count=len(active_goals),
                    active_source_count=len(active_starts),
                    domain_count=len(domains),
                    domain_cursor=domain_cursor - 1,
                )
                domain_key = domain_identity(active_starts, active_goals,
                                             active_forbidden, active_forbidden_wires,
                                             added_wire_branch, source_domain, goal_domain)
                if domain_key in seen_domains:
                    continue
                seen_domains.add(domain_key)
                if not active_starts or not active_goals:
                    # Current search requires finite seed/goal states. Preserve
                    # a symbolic-only residual as unsearched rather than
                    # claiming the transaction domain was exhausted.
                    pending_symbolic_incomplete = True
                    continue
                publish_access_progress("domain_start")
                self._watchdog_check(net, "access_selection")
                self._attempt_cuts = {
                    int(lo): list(cuts) for lo, cuts in base_cuts.items()
                }
                self._search_forbidden_via_sites = (
                    set(base_forbidden) | set(active_forbidden)
                )
                self._search_forbidden_wire_edges = (
                    set(base_forbidden_wires) | set(active_forbidden_wires)
                )
                self._edge_fail = None
                self._access_static_disconnection_evidence = _AccessStaticDisconnectionRecorder()
                self._selected_access_pair = None
                self._hard_rejections = {}
                self._hard_rejection_samples = {}
                access_details = (
                    (
                        "source_access",
                        tuple(self._access_state_evidence(item)
                              for item in active_starts),
                    ),
                    (
                        "goal_access",
                        tuple(self._access_state_evidence(item)
                              for item in active_goals),
                    ),
                    (
                        "forbidden_via_sites",
                        tuple(sorted(active_forbidden)),
                    ),
                    (
                        "forbidden_wire_edges",
                        tuple(sorted(active_forbidden_wires)),
                    ),
                )
                self._route_failure_details = base_details + access_details
                attempt_num = int(access_progress["attempt_count"]) + 1
                if neg_group_active and attempt_num > 24:
                    pending_symbolic_incomplete = True
                    break
                publish_access_progress(
                    "route_attempt",
                    attempt_count=attempt_num,
                )
                missing_endpoint_domains = object()
                previous_endpoint_domains = getattr(
                    self, "_active_endpoint_access_domains", missing_endpoint_domains
                )
                self._active_endpoint_access_domains = (source_domain, goal_domain)
                try:
                    found = attempt(active_starts, dict(active_goals))
                finally:
                    if previous_endpoint_domains is missing_endpoint_domains:
                        del self._active_endpoint_access_domains
                    else:
                        self._active_endpoint_access_domains = previous_endpoint_domains
                if found is None:
                    edge_fail = getattr(self, "_edge_fail", None) or (
                        "access_exhausted",
                        self._hard_rejection_evidence(),
                    )
                    if len(edge_fail) == 9 and edge_fail[8] == "planar_region_events":
                        work = dict(self._route_failure_details).get("planar_region_event_work")
                        if work is not None:
                            region_budget_work[edge_fail] = work
                            breakdown = dict(self._route_failure_details).get('planar_region_event_work_by_kind')
                            if breakdown is not None:
                                region_budget_breakdowns[edge_fail] = breakdown
                    static_info = getattr(self, "_access_static_disconnection_evidence", None)
                    scope_details = ()
                    if (edge_fail[0] == "static_disconnected"
                            and type(static_info) is _AccessStaticDisconnectionRecorder
                            and static_info.scope in {"planar_region", "finite_candidate_graph"}):
                        scope_details = (("static_disconnection_scope", static_info.scope),)
                        if static_info.scope == "planar_region":
                            scope_details += (("precheck_pruned_ms", static_info.elapsed_ms),)
                    attempts.append(access_details + scope_details + (
                        ("failure_kind", str(edge_fail[0])),
                        ("rejections", self._hard_rejection_evidence()),
                        ("blockers", self._hard_rejection_sample_evidence()),
                    ))
                    merge_rejections()
                    if edge_fail[0] == "budget" and added_wire_branch:
                        pending_wire_budget_failure = edge_fail
                    terminal_failure = (
                        edge_fail
                        if edge_fail[0] in {"solve_budget", "budget"}
                        else pending_wire_budget_failure or edge_fail
                    )
                    if edge_fail[0] == "solve_budget" or (
                        edge_fail[0] == "budget" and not added_wire_branch
                    ):
                        domains = []
                        break
                    # This whole disjoint domain has no route, but a sibling
                    # partition created by an earlier conflict may still contain
                    # feasible pairs.
                    continue

                segs, vias = found
                candidate_segs = list(segs)
                candidate_vias = list(vias)
                publish_access_progress(
                    "candidate_prepare",
                    candidate_segment_count=len(candidate_segs),
                    candidate_via_count=len(candidate_vias),
                    candidate_via_cursor=0,
                )
                selected_pair = getattr(self, "_selected_access_pair", None)
                selected_access = ()
                selected_planar = (False, False)
                if (
                    isinstance(selected_pair, tuple)
                    and len(selected_pair) == 2
                ):
                    selected_start_key, selected_goal_key = selected_pair
                    selected_start, start_planar = selected_member(
                        active_starts, source_domain, selected_start_key
                    )
                    selected_goal, goal_planar = selected_member(
                        active_goals, goal_domain, selected_goal_key
                    )
                    if selected_start is not None and selected_goal is not None:
                        selected_access = (selected_start, selected_goal)
                        selected_planar = (start_planar, goal_planar)
                access_vias = tuple(
                    access_via
                    for _state, access_via in (
                        tuple(active_starts) + tuple(active_goals)
                    )
                    if access_via is not None
                )
                access_via_ids = {
                    id(access_via) for access_via in access_vias
                }
                authorized_single_sites = {
                    (access_via.x, access_via.y, access_via.lo)
                    for access_via in access_vias
                    if self._site_n(access_via) == 1
                }
                selected_vias = tuple(
                    via
                    for _state, via in selected_access
                    if via is not None
                )
                selected_via_ids = {id(via) for via in selected_vias}
                mutable_indices = {
                    index
                    for index, candidate_via in enumerate(candidate_vias)
                    if (
                        id(candidate_via) not in selected_via_ids
                        and (
                            id(candidate_via) not in access_via_ids
                            or (
                                self._site_n(candidate_via) != 1
                                and (
                                    candidate_via.x,
                                    candidate_via.y,
                                    candidate_via.lo,
                                ) in authorized_single_sites
                            )
                        )
                    )
                }

                def candidate_covers_selected_state(item) -> bool:
                    (x, y, gi), _access_via = item
                    for seg in candidate_segs:
                        if int(seg.gi) != int(gi):
                            continue
                        if self._point_in_closed_rect(
                            int(x),
                            int(y),
                            _seg_bbox(
                                seg.ax,
                                seg.ay,
                                seg.bx,
                                seg.by,
                                self._segment_half_width(net, seg),
                            ),
                        ):
                            return True
                    for via in candidate_vias:
                        rect = self._via_layer_pad_rect(via, int(gi))
                        if rect is not None and self._point_in_closed_rect(
                            int(x), int(y), rect
                        ):
                            return True
                    return False

                # anaroute's via-selector contract: the terminal landing is part
                # of the winning path, not a side-channel suggestion.  A search
                # callback must publish one exact selected access pair and return
                # both selected access-via objects in the atomic candidate.  If it
                # drops one, accepting the remaining graph path would mark the MST
                # edge connected while emitting no conductor at that terminal.
                via_group_ok = bool(selected_access) and all(
                    any(candidate is selected for candidate in candidate_vias)
                    for selected in selected_vias
                )
                if via_group_ok and any(selected_planar):
                    # Recompute the source component against the live engine
                    # and require actual candidate metal at each selected cap.
                    for planar, domain, (state, _via) in zip(
                        selected_planar, (source_domain, goal_domain), selected_access
                    ):
                        if planar and not self._owned_planar_access_joins_conductor(
                            net, domain.node, state, candidate_segs, candidate_vias
                        ):
                            via_group_ok = False
                            break
                    if via_group_ok and source_domain is not None and goal_domain is not None:
                        prior = getattr(self, "_active_access_routed_net", None)
                        prior_segs = () if prior is None else tuple(prior.segs)
                        prior_vias = () if prior is None else tuple(prior.vias)
                        via_group_ok = self._candidate_deck_connects_probes(
                            net, prior_segs + tuple(candidate_segs),
                            prior_vias + tuple(candidate_vias),
                            (source_domain.node, goal_domain.node), include_fixed=True,
                        )
                    elif via_group_ok:
                        via_group_ok = False
                failed_via_identity = None
                failed_wire_identity = None
                group_failure_kind = (
                    "candidate_via_cut_conflict"
                    if via_group_ok
                    else "candidate_endpoint_access_missing"
                )
                if via_group_ok:
                    for index, via in enumerate(sorted(
                        candidate_vias,
                        key=lambda item: (
                            int(item.lo), int(item.x), int(item.y), int(item.n)
                        ),
                    )):
                        if index % 64 == 0:
                            publish_access_progress(
                                "candidate_via_validate",
                                candidate_via_cursor=int(index),
                            )
                            self._watchdog_check(net, "access_selection")
                        if not self._via_legal(
                            via.x, via.y, via.lo, net, n=self._site_n(via)
                        ):
                            via_group_ok = False
                            failed_via_identity = self._via_search_identity(
                                via.x,
                                via.y,
                                via.lo,
                                self._site_n(via),
                            )
                            break
                        self._add_attempt_cut(via)
                if via_group_ok:
                    # Preserve the returned via models through exact cut-group
                    # validation.  Only a cut-clean transaction may enumerate a
                    # smaller landing model; otherwise model shrink would hide a
                    # cut conflict that requires conflict-directed path replay.
                    self._attempt_cuts = {
                        int(lo): list(cuts) for lo, cuts in base_cuts.items()
                    }
                    resolved_candidate_vias = self._resolve_pattern_via_models(
                        net,
                        candidate_vias,
                        mutable_indices=mutable_indices,
                        segs=candidate_segs,
                    )
                    if resolved_candidate_vias is None:
                        # A mandatory endpoint landing cannot move or shrink.
                        # Its own returned path can nevertheless face that pad
                        # across a legal-to-fill same-net notch.  The outer MST
                        # group uses this exact physical merge, but cannot reach
                        # it if this transaction rejects the winner first.
                        # Materialize and validate the bridge here atomically;
                        # foreign artwork still makes the normal resolver fail.
                        repair_segs = self._materialize_candidate_via_wire_merges(
                            net,
                            candidate_vias,
                            candidate_segs,
                            prior_via_count=0,
                            prior_seg_count=0,
                            immutable_via_indices=(
                                set(range(len(candidate_vias))) - mutable_indices
                            ),
                        )
                        if repair_segs:
                            repaired_segs = candidate_segs + list(repair_segs)
                            repaired_vias = self._resolve_pattern_via_models(
                                net,
                                candidate_vias,
                                mutable_indices=mutable_indices,
                                segs=repaired_segs,
                            )
                            if repaired_vias is not None:
                                candidate_segs = repaired_segs
                                resolved_candidate_vias = repaired_vias
                    if resolved_candidate_vias is None:
                        prior = getattr(self, "_active_access_routed_net", None)
                        repaired_candidate = self._resolve_candidate_same_net_fills(
                            net, prior, candidate_segs, candidate_vias,
                            mutable_indices=mutable_indices,
                        )
                        if repaired_candidate is not None:
                            candidate_segs, resolved_candidate_vias = repaired_candidate
                    if resolved_candidate_vias is None:
                        via_group_ok = False
                        group_failure_kind = (
                            "candidate_via_landing_conflict"
                        )
                        wire_conflicts = self._pattern_via_wire_conflicts(
                            net,
                            candidate_vias,
                            candidate_segs,
                        )
                        publish_access_progress(
                            "candidate_landing_conflict",
                            last_candidate_landing_conflict=(
                                self._candidate_via_conflict_details(
                                    net,
                                    candidate_vias,
                                    candidate_segs,
                                    self._pattern_via_pad_visible_conflicts(
                                        net, candidate_vias, candidate_segs
                                    ),
                                    wire_conflicts,
                                )
                            ),
                        )
                        if wire_conflicts:
                            via = candidate_vias[wire_conflicts[0][0]]
                            failed_wire_identity = self._wire_search_identity(
                                net, candidate_segs[wire_conflicts[0][1]]
                            )
                            failed_via_identity = self._via_search_identity(
                                via.x,
                                via.y,
                                via.lo,
                                self._site_n(via),
                            )
                        else:
                            for via in sorted(
                                candidate_vias,
                                key=lambda item: (
                                    int(item.lo),
                                    int(item.x),
                                    int(item.y),
                                    int(item.n),
                                ),
                            ):
                                if self._via_pad_attempt_conflicts(net, via):
                                    failed_via_identity = (
                                        self._via_search_identity(
                                            via.x,
                                            via.y,
                                            via.lo,
                                            self._site_n(via),
                                        )
                                    )
                                    break
                    else:
                        models_changed = (
                            resolved_candidate_vias != candidate_vias
                        )
                        candidate_vias = resolved_candidate_vias
                        if not all(
                            any(candidate is selected for candidate in candidate_vias)
                            for selected in selected_vias
                        ):
                            via_group_ok = False
                            group_failure_kind = (
                                "candidate_endpoint_access_missing"
                            )
                        if not models_changed:
                            for via in candidate_vias:
                                self._add_attempt_cut(via)
                        elif via_group_ok:
                            for index, via in enumerate(sorted(
                                candidate_vias,
                                key=lambda item: (
                                    int(item.lo),
                                    int(item.x),
                                    int(item.y),
                                    int(item.n),
                                ),
                            )):
                                if index % 64 == 0:
                                    publish_access_progress(
                                        "candidate_via_revalidate",
                                        candidate_via_cursor=int(index),
                                    )
                                    self._watchdog_check(
                                        net, "access_selection"
                                    )
                                if not self._via_legal(
                                    via.x,
                                    via.y,
                                    via.lo,
                                    net,
                                    n=self._site_n(via),
                                ):
                                    via_group_ok = False
                                    group_failure_kind = (
                                        "candidate_via_cut_conflict"
                                    )
                                    failed_via_identity = (
                                        self._via_search_identity(
                                            via.x,
                                            via.y,
                                            via.lo,
                                            self._site_n(via),
                                        )
                                    )
                                    break
                                self._add_attempt_cut(via)
                if via_group_ok:
                    pad_conflicts = self._pattern_via_pad_visible_conflicts(
                        net,
                        candidate_vias,
                        candidate_segs,
                    )
                    wire_conflicts = self._pattern_via_wire_conflicts(
                        net,
                        candidate_vias,
                        candidate_segs,
                    )
                    if pad_conflicts or wire_conflicts:
                        via_group_ok = False
                        group_failure_kind = (
                            "candidate_via_landing_conflict"
                        )
                        if wire_conflicts:
                            via_index = wire_conflicts[0][0]
                            via = candidate_vias[via_index]
                            failed_wire_identity = self._wire_search_identity(
                                net, candidate_segs[wire_conflicts[0][1]]
                            )
                            failed_via_identity = self._via_search_identity(
                                via.x,
                                via.y,
                                via.lo,
                                self._site_n(via),
                            )
                # Wire exclusions diversify search after a path-dependent
                # conflict. They are not physical constraints on a completed
                # candidate. A subdivided path can reproduce the span while a
                # different via or repaired metal union resolves the original
                # conflict. Physical and endpoint checks govern acceptance.
                if via_group_ok and not all(
                    candidate_covers_selected_state(item)
                    for item in selected_access
                ):
                    via_group_ok = False
                    group_failure_kind = "candidate_endpoint_access_missing"
                if via_group_ok:
                    mim_failure = self._candidate_mim_metal_failure(
                        net, candidate_segs, candidate_vias,
                    )
                    if mim_failure:
                        via_group_ok = False
                        group_failure_kind = "candidate_mim_metal_clearance"
                        failed_via_identity, failed_wire_identity = self._mim_candidate_retry_identities(
                            net, candidate_segs, candidate_vias, mim_failure,
                        )
                if via_group_ok and not self._candidate_access_deck_connected(
                    net,
                    candidate_segs,
                    candidate_vias,
                    tuple(state for state, _via in selected_access),
                ):
                    via_group_ok = False
                    group_failure_kind = "candidate_endpoint_deck_disconnected"
                if via_group_ok and not self._candidate_terminal_deck_connected(
                    net,
                    candidate_segs,
                    candidate_vias,
                    tuple(state for state, _via in selected_access),
                    materialized_access_states=materialized_access_states,
                ):
                    via_group_ok = False
                    group_failure_kind = "candidate_terminal_deck_disconnected"
                # A later group-local MST edge may contain no new via while its
                # wire approaches a previously materialized same-net array.  The
                # edge-only validator above cannot see that earlier pad.  Check
                # the same combined geometry and physical merge used by the
                # outer MST validator before committing this access candidate.
                # Only branch on a wire from this edge; the earlier via and
                # earlier route are immutable.  Keep the ordinary first choice
                # whenever the existing combined merge would accept it.
                prior = getattr(self, "_active_access_routed_net", None)
                def _covers_selected_state(via, state):
                    x, y, gi = state
                    below, above = self._via_pad_half_for(via)
                    if int(gi) == int(via.lo):
                        half = int(below)
                    elif int(gi) == int(via.lo) + 1:
                        half = int(above)
                    else:
                        return False
                    return self._point_in_closed_rect(
                        int(x), int(y),
                        (int(via.x)-half, int(via.y)-half,
                         int(via.x)+half, int(via.y)+half))
                if (via_group_ok
                        and getattr(self, "_active_group_route_local_closure", False)
                        and prior is not None and prior.vias
                        and candidate_segs):
                    combined_vias = list(prior.vias) + list(candidate_vias)
                    prior_via_count = len(prior.vias)
                    prior_seg_count = len(prior.segs)
                    group_mutable = {
                        index for index in mutable_indices
                        if not any(_covers_selected_state(candidate_vias[index], state)
                                   for state, _access in selected_access)
                    }
                    combined_mutable = {prior_via_count + index for index in group_mutable}
                    combined_segs = tuple(prior.segs) + tuple(candidate_segs)
                    resolved = self._resolve_pattern_via_models(
                        net, combined_vias, mutable_indices=combined_mutable, segs=combined_segs)
                    if resolved is None:
                        repair = self._materialize_candidate_via_wire_merges(
                            net, combined_vias, combined_segs,
                            prior_via_count=prior_via_count,
                            prior_seg_count=prior_seg_count,
                            immutable_via_indices=(set(range(len(combined_vias))) - combined_mutable))
                        if repair:
                            resolved = self._resolve_pattern_via_models(
                                net, combined_vias, mutable_indices=combined_mutable,
                                segs=combined_segs + tuple(repair))
                    if resolved is None:
                        # Prove that the same candidate can be repaired by the
                        # outer transaction. The outer caller repeats that
                        # scoped trial and atomically transports its fills.
                        repaired = self._resolve_candidate_same_net_fills(
                            net, prior, candidate_segs, candidate_vias,
                            mutable_indices=group_mutable,
                        )
                        if repaired is not None:
                            resolved = list(prior.vias) + list(repaired[1])
                    if resolved is None:
                        combined_wires = self._pattern_via_wire_conflicts(
                            net, combined_vias, combined_segs)
                        combined_pads = self._pattern_via_pad_visible_conflicts(
                            net, combined_vias, combined_segs)
                        if (combined_wires and not combined_pads
                                and all(via_index < prior_via_count
                                        and index >= prior_seg_count
                                        for via_index, index, _layer in combined_wires)):
                            via_group_ok = False
                            group_failure_kind = "candidate_prior_via_new_wire_conflict"
                            failed_wire_identity = self._wire_search_identity(
                                net, candidate_segs[combined_wires[0][1] - prior_seg_count])
                            # A prior via is not a candidate route alternative.
                            failed_via_identity = None
                        elif (combined_wires and not combined_pads
                              and all(via_index < prior_via_count
                                      and index < prior_seg_count
                                      for via_index, index, _layer in combined_wires)):
                            retry_wire = self._candidate_new_owned_spacing_wire(
                                net, prior, candidate_segs, candidate_vias,
                            )
                            if retry_wire is not None:
                                via_group_ok = False
                                group_failure_kind = 'candidate_new_owned_spacing'
                                failed_wire_identity = retry_wire
                                failed_via_identity = None
                if via_group_ok and getattr(self, '_active_group_source_union_required', False):
                    from .source_junction_kernel import repair_group_edge_union
                    saved_cuts = self._attempt_cuts
                    self._attempt_cuts = {int(lo): list(cuts) for lo, cuts in base_cuts.items()}
                    try:
                        proposal = (repair_group_edge_union(
                            self, net, prior, candidate_segs, candidate_vias,
                            self._active_group_source_union_layers,
                            (source_domain.node, goal_domain.node),
                            tuple(state for state, _via in selected_access))
                            if (source_domain is not None and goal_domain is not None
                                and getattr(self, '_active_group_source_union_layers', None) is not None)
                            else None)
                    finally:
                        self._attempt_cuts = saved_cuts
                    if proposal is None:
                        via_group_ok = False
                        group_failure_kind = 'candidate_source_union_unknown'
                        pending_group_source_union_unknown = True
                    else:
                        candidate_segs, candidate_vias = proposal
                        if (not all(candidate_covers_selected_state(item) for item in selected_access)
                                or self._candidate_mim_metal_failure(net, candidate_segs, candidate_vias)
                                or not self._candidate_access_deck_connected(
                                    net, candidate_segs, candidate_vias,
                                    tuple(state for state, _via in selected_access))
                                or not self._candidate_terminal_deck_connected(
                                    net, candidate_segs, candidate_vias,
                                    tuple(state for state, _via in selected_access),
                                    materialized_access_states=materialized_access_states)):
                            via_group_ok = False
                            group_failure_kind = 'candidate_source_union_unknown'
                            pending_group_source_union_unknown = True
                        else:
                            self._last_accepted_group_source_union = True
                            self._last_accepted_group_source_candidate = (
                                self._book, getattr(self, '_active_group_route_union_baseline', None),
                                prior, tuple(candidate_segs), tuple(candidate_vias),
                                (tuple(source_domain.node), tuple(goal_domain.node)),
                                tuple(tuple(state) for state, _via in selected_access),
                            )
                if via_group_ok:
                    self._last_accepted_access_states = tuple(
                        tuple(int(value) for value in state)
                        for state, _via in selected_access
                    )
                    self._last_accepted_planar = tuple(selected_planar)
                    self._last_accepted_terminal_probes = getattr(
                        self, "_last_candidate_terminal_probes", None
                    )
                    return candidate_segs, candidate_vias

                edge_fail = (
                    group_failure_kind,
                    self._hard_rejection_evidence(),
                )
                attempts.append(access_details + (
                    ("failure_kind", edge_fail[0]),
                    ("rejections", self._hard_rejection_evidence()),
                    ("blockers", self._hard_rejection_sample_evidence()),
                ))
                merge_rejections()
                selected_pair = getattr(self, "_selected_access_pair", None)
                if (
                    not isinstance(selected_pair, tuple)
                    or len(selected_pair) != 2
                ):
                    terminal_failure = (
                        "access_selection_identity_missing",
                        self._hard_rejection_evidence(),
                    )
                    break
                selected_start_key, selected_goal_key = selected_pair
                selected_start, start_planar = selected_member(
                    active_starts, source_domain, selected_start_key
                )
                selected_goal, goal_planar = selected_member(
                    active_goals, goal_domain, selected_goal_key
                )
                if selected_start is None or selected_goal is None:
                    terminal_failure = (
                        "access_selection_identity_invalid",
                        self._hard_rejection_evidence(),
                    )
                    break
                mandatory_via_sites = {
                    self._via_search_identity(
                        via.x,
                        via.y,
                        via.lo,
                        self._site_n(via),
                    )
                    for _state, via in (selected_start, selected_goal)
                    if via is not None
                }
                group_conflicts = self._candidate_via_group_conflicts(
                    candidate_vias,
                    net,
                    candidate_segs,
                )
                branch_sites = ()
                if group_conflicts:
                    # One conflict is sufficient to partition every possible
                    # legal replacement path: it must omit at least one member.
                    branch_sites = tuple(
                        site
                        for site in group_conflicts[0]
                        if site not in mandatory_via_sites
                    )
                elif (
                    failed_via_identity is not None
                    and failed_via_identity not in mandatory_via_sites
                ):
                    # Defensive complete branch for a returned internal via that
                    # is already illegal against the immutable base ledger.
                    branch_sites = (failed_via_identity,)

                other_starts = tuple(
                    item for item in active_starts if item is not selected_start
                )
                other_goals = tuple(
                    item for item in active_goals if item is not selected_goal
                )
                branch_wires = (
                    () if failed_wire_identity is None
                    else (failed_wire_identity,)
                )
                publish_access_progress(
                    "candidate_partition",
                    candidate_failure_count=(
                        int(access_progress["candidate_failure_count"]) + 1
                    ),
                    last_branch_site_count=len(set(branch_sites)),
                    last_branch_wire_count=len(branch_wires),
                    last_candidate_failure_kind=str(edge_fail[0]),
                    last_group_conflict_count=len(group_conflicts),
                    last_residual_goal_count=len(other_goals),
                    last_residual_source_count=len(other_starts),
                )
                def locked_domain(domain, access, planar):
                    if domain is None:
                        return None
                    return _PlanarEndpointDomain(
                        domain.node, domain.grid, tuple(access[0]), (),
                        locked_planar=planar,
                    )

                locked_source = locked_domain(source_domain, selected_start, start_planar)
                locked_goal = locked_domain(goal_domain, selected_goal, goal_planar)
                for site in sorted(set(branch_sites)):
                    append_domain(
                        (selected_start,),
                        (selected_goal,),
                        set(active_forbidden) | {site},
                        active_forbidden_wires,
                        added_wire_branch,
                        locked_source,
                        locked_goal,
                    )

                # A wire conflict depends on the reconstructed path. Retain the
                # via (including mandatory endpoint vias) in an extra branch.
                # Existing merge, via branches, and residual domains stay intact.
                for wire_identity in sorted(branch_wires):
                    append_domain(
                        (selected_start,), (selected_goal,), active_forbidden,
                        set(active_forbidden_wires) | {wire_identity},
                        True,
                        locked_source,
                        locked_goal,
                    )

                # Disjoint exact partition of (S x G) \ {(s*, g*)}:
                #   ((S - s*) x G) U ({s*} x (G - g*)).
                # FIFO makes shallow alternatives win deterministically.
                if other_starts or has_planar(source_domain):
                    append_domain(
                        other_starts,
                        active_goals,
                        active_forbidden,
                        active_forbidden_wires,
                        added_wire_branch,
                        (source_domain.without_exact(selected_start_key)
                         if has_planar(source_domain) else source_domain),
                        goal_domain,
                    )
                if other_goals or has_planar(goal_domain):
                    append_domain(
                        (selected_start,),
                        other_goals,
                        active_forbidden,
                        active_forbidden_wires,
                        added_wire_branch,
                        locked_source,
                        (goal_domain.without_exact(selected_goal_key)
                         if has_planar(goal_domain) else goal_domain),
                    )
                # A sibling may still succeed after a new child's local stop,
                # but exhausting the remaining siblings cannot prove the
                # unsearched child impossible. Keep its typed budget outcome.
                terminal_failure = pending_wire_budget_failure or edge_fail
        finally:
            if prior_static_evidence is absent_static_evidence:
                self.__dict__.pop("_access_static_disconnection_evidence", None)
            else:
                self._access_static_disconnection_evidence = prior_static_evidence
            self._attempt_cuts = {
                int(lo): list(cuts) for lo, cuts in base_cuts.items()
            }
            self._search_forbidden_via_sites = set(base_forbidden)
            self._search_forbidden_wire_edges = set(base_forbidden_wires)
            self._selected_access_pair = None

        self._precheck_envelope_disconnected = bool(
            len(attempts) == 1 and getattr(self, "_precheck_envelope_disconnected", False)
        )
        self._hard_rejections = aggregate_rejections
        self._hard_rejection_samples = aggregate_samples
        self._route_failure_details = base_details + (
            ("access_failures", tuple(access_failures)),
            ("access_attempts", tuple(attempts)),
        )
        if terminal_failure is not None and len(terminal_failure) == 9 and terminal_failure[8] == "planar_region_events":
            work = region_budget_work.get(terminal_failure)
            if work is not None:
                self._route_failure_details += (("planar_region_event_work", work),)
                breakdown = region_budget_breakdowns.get(terminal_failure)
                if breakdown is not None:
                    self._route_failure_details += (("planar_region_event_work_by_kind", breakdown),)
        if (pending_group_source_union_unknown and
                (terminal_failure is None or terminal_failure[0] not in {'budget', 'solve_budget'})):
            self._edge_fail = ('candidate_source_union_unknown', terminal_failure,
                               self._hard_rejection_evidence())
        elif (pending_symbolic_incomplete and
                (terminal_failure is None or terminal_failure[0] not in {"budget", "solve_budget"})):
            self._edge_fail = (
                "symbolic_domain_incomplete", terminal_failure,
                self._hard_rejection_evidence(),
            )
        else:
            self._edge_fail = terminal_failure or (
                "access_exhausted",
                self._hard_rejection_evidence(),
            )
        return None

    def _route_access_transactions(
        self,
        starts: Sequence[Tuple[Tuple[int, int, int], Optional[_ViaSite]]],
        goals: Mapping[Tuple[int, int, int], Optional[_ViaSite]],
        net: str,
        trunk: int,
        attempt,
        *,
        prevalidated: bool = False,
        materialized_access_states: Optional[
            Sequence[Tuple[int, int, int]]
        ] = None,
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """Canonical endpoint access through the complete multi-domain solver."""
        del trunk  # access selection is feasibility-only, never a local optimum.
        ordered_starts = sorted(starts, key=self._access_state_key)
        ordered_goals = sorted(goals.items(), key=self._access_state_key)
        if not ordered_starts or not ordered_goals:
            domains = getattr(self, "_active_endpoint_access_domains", (None, None))
            if (isinstance(domains, tuple) and len(domains) == 2
                    and all(items or (isinstance(domain, _PlanarEndpointDomain)
                                      and domain.singleton is None and domain.regions)
                            for items, domain in zip((ordered_starts, ordered_goals), domains))):
                self._edge_fail = ("symbolic_domain_incomplete", ())
            return None
        if materialized_access_states is None:
            materialized_access_states = getattr(
                self, "_active_materialized_access_states", ()
            )
        return self._route_access_multi_transaction(
            ordered_starts,
            ordered_goals,
            net,
            attempt,
            prevalidated=prevalidated,
            materialized_access_states=materialized_access_states,
        )

    # -------------------------------------------------- signal-net MST edge (#1/#5)
    def _signal_edge_pattern(
        self, net: str, a: Tuple[int, int, int], b: Tuple[int, int, int],
        acc_a: Optional[Sequence[Tuple[Tuple[int, int, int], Optional[_ViaSite]]]] = None,
        acc_b: Optional[Sequence[Tuple[Tuple[int, int, int], Optional[_ViaSite]]]] = None,
        lg: Optional[Sequence[int]] = None,
        trunk_gi: Optional[int] = None,
        *,
        hard_connectivity_witness: Optional[_HardConnectivityWitness] = None,
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """PATTERN-FIRST closed-form connection of ONE signal-net MST edge: CLIMB
        each pin's column up to a routing metal, then a few deterministic PLANAR
        routes -- STRAIGHT, single-bend L (both orientations), Z with one mid track
        (x and y) -- between the two climbed columns on that layer.

        Judged with the engine's OWN hard legality (``_via_legal`` for the climb
        stack, ``_wire_legal`` for the planar run -- foreign device-artwork keep-out +
        via-cut spacing; own metal exempt; NO new legality model) and the SAME
        present/history price as A*. Preferred-axis shapes are ranked with the shared
        PDK/A* objective. The function returns the SAME
        ``(segs, vias)`` shape as ``_route_edge``, so a clean edge skips the
        dense per-edge A* on the obstacle-edge candidate grid (the measured 234x254
        grid / 80248 expansions / 100-470s per multi-pin signal net --
        ``_candidate_lines`` ingests every in-window device-finger edge).

        The planar run rides a ROUTING metal (>= M2), NEVER the M1 pin/device layer
        (a long signal run on M1 is cost-penalized -- ``_edge_cost``'s
        ``_M1_ROUTE_COST``); both pins climb UP to it (never a dig-down), exactly the
        rail drop's climb-and-run geometry (``_straight_rail_drop``).

        The FIRST climb hop REUSES the transaction-selected pin-access ESCAPE via
        (``acc_a`` / ``acc_b`` from ``_pin_access``; recomputed if not supplied).
        The selected escape alone is reserved in ``_attempt_cuts`` for this bounded
        attempt, so reusing it keeps the drop hard-legal AND emits the pin's own
        access via. A blocked climb column,
        blocked corridor, or pattern set with no congestion-free member returns None
        and the caller falls back to the bounded SPARSE local search
        (``_signal_candidate_lines`` + ``_route_edge``). Keyed ONLY on being an MST
        edge (a net with no declared rail) -- no per-circuit branch; no wall-clock."""
        self._selected_access_pair = None
        if hard_connectivity_witness is None:
            hard_connectivity_witness = getattr(
                self, "_active_hard_connectivity_witness", None
            )
        pattern_search_incumbent = getattr(
            self, "_active_pattern_search_incumbent", None
        )
        ax, ay, agi = a
        bx, by, bgi = b
        if max(agi, bgi) > self._max_gi:
            return None                     # pin above top routable metal -> search
        # ROUTE LAYER: a routing metal (>= M2 when a pin is on M1), never below a
        # pin. Both pins climb UP to it; a single-metal tech (no headroom) stays put.
        rl = max(agi, bgi, 1)
        if rl > self._max_gi:
            rl = max(agi, bgi)
        # Pattern-first is subject to the hard LayerSet. Exact trunk/layer choices
        # are owned by the certified resource plan; this residual path does not
        # synthesize an implicit upper-layer plan from a style switch.
        target = rl
        allowed = sorted(set(lg) if lg is not None
                         else set(range(self._n_layers)))
        allowed_layers = set(allowed)
        upper = [gi for gi in allowed
                 if gi >= target and gi <= self._max_gi]
        if not upper:
            # The LayerSet-filtered fallback may route lower or fail honestly.
            return None
        declared = self._declared_trunk_gi_by_net.get(net)
        rl = declared if declared in upper else upper[0]
        if acc_a is None:
            acc_a = self._pin_access(net, ax, ay, agi)
        if acc_b is None:
            acc_b = self._pin_access(net, bx, by, bgi)
        vias: List[_ViaSite] = []
        cols: List[Tuple[int, int]] = []
        selected_access: List[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ] = []
        for (px, py, pgi), acc in (((ax, ay, agi), acc_a), ((bx, by, bgi), acc_b)):
            cx, cy = px, py
            lo = pgi
            direct = next(
                (
                    item for item in acc
                    if item[0] == (px, py, pgi) and item[1] is None
                ),
                None,
            )
            selected = direct
            if lo < rl:
                # FIRST hop pgi -> pgi+1: reuse the RESERVED escape via (identical
                # cut -> hard-legal + it covers the pin); a fresh centre via would
                # violate the net-blind via-space rule against its own reservation.
                escape = next((item for item in acc if item[1] is not None), None)
                if escape is not None:
                    _escape_state, esc = escape
                    vias.append(esc)
                    cx, cy = esc.x, esc.y
                    lo = esc.lo + 1
                    selected = escape
                elif direct is not None:
                    if not {lo, lo + 1}.issubset(allowed_layers):
                        return None
                    selected_via = self._preferred_via_site(net, px, py, lo)
                    if selected_via is None:
                        return None
                    vias.append(selected_via)
                    lo += 1
                else:
                    return None
            elif direct is None:
                # The active transaction partition may intentionally contain only
                # an offset escape state.  Pattern routing on the pin layer cannot
                # fabricate the absent direct state as its selected identity; A*
                # must consume the actual active access domain instead.
                return None
            # REMAINING stack straight up the (possibly offset) climb column to rl.
            while lo < rl:
                if not {lo, lo + 1}.issubset(allowed_layers):
                    return None
                selected_via = self._preferred_via_site(net, cx, cy, lo)
                if selected_via is None:
                    return None
                vias.append(selected_via)
                lo += 1
            cols.append((cx, cy))
            selected_access.append(selected)

        access_via_ids = {
            id(via) for _state, via in selected_access if via is not None
        }

        def resolve_candidate_vias(
            candidate_vias: Sequence[_ViaSite],
            candidate_segs: Sequence[_WireSeg] = (),
        ) -> Optional[List[_ViaSite]]:
            # A transaction-selected terminal-access via is an identity in the
            # active access domain, not a model placeholder.  Final closed-form
            # candidates may shrink only their internal climb/bridge vias; if the
            # locked accesses themselves cannot coexist, this pattern is rejected.
            mutable_indices = {
                index for index, via in enumerate(candidate_vias)
                if id(via) not in access_via_ids
            }
            return self._resolve_pattern_via_models(
                net,
                candidate_vias,
                mutable_indices=mutable_indices,
                segs=candidate_segs,
            )

        resolved_vias = resolve_candidate_vias(vias)
        if resolved_vias is None:
            return None
        vias = resolved_vias

        def selected_result(got):
            self._record_selected_access_pair(
                selected_access[0], selected_access[1]
            )
            return got

        def record_connectivity_witness(got) -> None:
            if hard_connectivity_witness is None:
                return
            access_via_ids = {
                id(via) for _state, via in selected_access if via is not None
            }
            candidate_segs, candidate_vias = got
            internal_vias = tuple(
                via for via in candidate_vias if id(via) not in access_via_ids
            )
            # Residual A* uses only the declared default via model. Functional
            # n=1 access vias are already embodied by their upper endpoint state;
            # an internal n=1 hop is not a candidate-graph edge and cannot prove it.
            if any(via.n == 1 for via in internal_vias):
                return
            hard_connectivity_witness.prove(
                selected_access[0][0],
                selected_access[1][0],
                segs=candidate_segs,
                vias=internal_vias,
            )

        def record_pattern_incumbent(
            score: int,
            candidate_index: int,
            got: Tuple[List[_WireSeg], List[_ViaSite]],
        ) -> None:
            if not isinstance(
                pattern_search_incumbent, _PatternSearchIncumbent
            ):
                return
            pattern_search_incumbent.consider(
                score=score,
                candidate_index=candidate_index,
                start=selected_access[0],
                goal=selected_access[1],
                segs=got[0],
                vias=got[1],
            )

        (cax, cay), (cbx, cby) = cols
        # SHARED climb column (endpoints distinct only in layer): the stack ties them.
        if (cax, cay) == (cbx, cby):
            return selected_result(([], vias)) if vias else None
        grid_nm = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
        if any(
            coordinate % grid_nm
            for coordinate in (cax, cay, cbx, cby)
        ):
            # Endpoint access is materializer-owned.  Pattern routing cannot
            # silently move an off-grid terminal; leave it to the exact residual
            # path, whose final grid certificate will fail closed with ownership.
            return None

        def grid_midpoint(first: int, second: int) -> int:
            values = self._on_manufacturing_grid_lines(
                ((int(first) + int(second)) // 2,),
                min(int(first), int(second)),
                max(int(first), int(second)),
            )
            return values[0] if values else int(first)

        mx = grid_midpoint(cax, cbx)
        my = grid_midpoint(cay, cby)
        # Candidate corner-lists on the route layer, STRAIGHTEST first for stable
        # ties; every legal shape is ranked by the shared A*/congestion objective.
        patterns: List[List[Tuple[int, int, int, int]]] = []
        if cax == cbx or cay == cby:
            patterns.append([(cax, cay, cbx, cby)])                     # STRAIGHT
        patterns.append([(cax, cay, cbx, cay), (cbx, cay, cbx, cby)])   # L horiz-first
        patterns.append([(cax, cay, cax, cby), (cax, cby, cbx, cby)])   # L vert-first
        patterns.append([(cax, cay, mx, cay), (mx, cay, mx, cby),
                         (mx, cby, cbx, cby)])                          # Z via mid-x
        patterns.append([(cax, cay, cax, my), (cax, my, cbx, my),
                         (cbx, my, cbx, cby)])                          # Z via mid-y
        # The per-segment direction preference is a PDK fact, not a design mode:
        # horizontal legs use an H-preferred metal and vertical legs a V-preferred
        # metal whenever the LayerSet admits the adjacent pair. Candidate ranking
        # uses the same PDK/congestion objective as residual A*.
        allowed = set(lg) if lg else set(range(self._n_layers))
        allowed.add(rl)
        hv = self._hv_layers(net, rl, allowed)
        if hv is not None:
            gi_h, gi_v = hv
            ranked_hv = []
            for candidate_idx, corners in enumerate(patterns):
                got = self._hv_decompose(net, corners, vias, gi_h, gi_v, rl)
                if got is None:
                    continue
                segs, candidate_vias = got
                candidate_vias = resolve_candidate_vias(candidate_vias, segs)
                if candidate_vias is None:
                    continue
                got = (segs, candidate_vias)
                congestion = self._pattern_congestion(net, segs, candidate_vias)
                score = self._pattern_objective(
                    net,
                    segs,
                    candidate_vias,
                    int(trunk_gi or 0),
                    congestion=congestion,
                )
                ranked_hv.append((score, candidate_idx, congestion, got))
            if ranked_hv:
                for _score, _index, _congestion, got in ranked_hv:
                    record_connectivity_witness(got)
                if all(item[2] > 0 for item in ranked_hv):
                    best = min(
                        ranked_hv, key=lambda item: (item[0], item[1])
                    )
                    record_pattern_incumbent(best[0], best[1], best[3])
                    return None
                return selected_result(
                    min(ranked_hv, key=lambda item: (item[0], item[1]))[3]
                )
            # The preferred-axis decomposition was applicable but every such
            # shape was blocked. Defer to negotiated residual A* rather than
            # silently emitting an off-preference single-layer pattern.
            return None
        # A one-axis PDK or a capped LayerSet has no adjacent opposite-axis pair.
        # Keep the bounded residual fallback legal and deterministic.
        ranked_single = []
        for candidate_idx, corners in enumerate(patterns):
            segs: List[_WireSeg] = []
            ok = True
            for (sx, sy, ex, ey) in corners:
                if (sx, sy) == (ex, ey):
                    continue
                if not self._wire_legal(rl, sx, sy, ex, ey, net):
                    ok = False
                    break
                segs.append(self._wire_segment(net, rl, sx, sy, ex, ey))
            if ok and segs:
                candidate_vias = resolve_candidate_vias(vias, segs)
                if candidate_vias is None:
                    continue
                got = (segs, candidate_vias)
                congestion = self._pattern_congestion(
                    net, segs, candidate_vias
                )
                score = self._pattern_objective(
                    net,
                    segs,
                    candidate_vias,
                    int(trunk_gi or 0),
                    congestion=congestion,
                )
                ranked_single.append((score, candidate_idx, congestion, got))
        if ranked_single:
            for _score, _index, _congestion, got in ranked_single:
                record_connectivity_witness(got)
            if all(item[2] > 0 for item in ranked_single):
                best = min(
                    ranked_single, key=lambda item: (item[0], item[1])
                )
                record_pattern_incumbent(best[0], best[1], best[3])
                return None
            return selected_result(
                min(ranked_single, key=lambda item: (item[0], item[1]))[3]
            )
        return None

    # ------------------------------------------------- HV per-direction layering
    def _hv_layers(self, net: str, route_gi: int,
                   allowed: Set[int]) -> Optional[Tuple[int, int]]:
        """The (gi_h, gi_v) layer pair for HV discipline around ``route_gi`` (where
        both pins climbed): the H-preferred and V-preferred metals a horizontal
        and a vertical leg must sit on.

        The PDK's declared preferred axes ALTERNATE (V, H, V, H ...), so ``route_gi``
        and its immediate neighbours have OPPOSITE preference: whichever axis
        ``route_gi`` itself prefers is kept ON ``route_gi`` (that leg needs no extra
        via off the climb stack), and the perpendicular axis takes the ADJACENT
        opposite-preference metal -- preferring the metal ABOVE (upper metals are the
        over-cell routing resource; the metal below is the penalised pin/device
        layer). Returns None when no adjacent opposite-axis metal is inside the net's
        ``allowed`` LayerSet (an honest cap block -> the caller keeps the plain
        single-layer pattern; never a silent off-cap route)."""
        declared = self._declared_trunk_axis_by_net.get(net)

        def pref_h(gi: int) -> bool:
            if declared is not None and gi == declared[0]:
                return declared[1] == "H"
            direction_preference = getattr(
                self, "_direction_preference_by_gi", {}
            ).get(gi)
            if direction_preference is not None:
                return direction_preference[0] == "H"
            return self._tech.preferred_horizontal(gi)

        def _neighbor(want_h: bool) -> Optional[int]:
            for g in (route_gi + 1, route_gi - 1):
                if 0 <= g <= self._max_gi and g in allowed and pref_h(g) == want_h:
                    return g
            return None

        if pref_h(route_gi):
            gi_h = route_gi
            gi_v = _neighbor(False)
            return None if gi_v is None else (gi_h, gi_v)
        gi_v = route_gi
        gi_h = _neighbor(True)
        return None if gi_h is None else (gi_h, gi_v)

    def _hv_bridge_via(self, x: int, y: int, a: int, b: int,
                       net: str) -> Optional[_ViaSite]:
        """A single via bridging ADJACENT graph layers ``a`` / ``b`` at ``(x, y)``,
        judged by the engine's OWN hard via legality and shared soft reliability /
        congestion objective. None when neither declared nor functional artwork can
        land."""
        lo = min(a, b)
        return self._preferred_via_site(net, x, y, lo)

    def _box_overlaps_foreign(self, gi: int, box: Rect, net: str) -> bool:
        """True iff ``box`` has zero-distance contact with foreign committed metal.

        Foreign device artwork is already a hard keep-out in ``_clear_obstacles``.
        Positive same-layer min-space proximity remains a congestion price so the
        search can negotiate a channel, but overlap or edge contact is electrical
        connectivity and must never be emitted by Gridless.
        """
        _sep = _eg_attr("_sep", _base_sep)
        midx = self._metal_idx.get(gi)
        if midx is None:
            return False
        nb = _norm(box)
        if (
            _sep is _base_sep
            and type(midx) is _SpatialIndex
            and midx._native is not None
        ):
            return midx._native.has_foreign_contact(
                nb[0], nb[1], nb[2], nb[3], net
            )
        for (owner, r) in midx.query(nb):
            if owner == net:
                continue
            sx, sy = _sep(nb, _norm(r))
            if sx == 0 and sy == 0:
                return True
        return False

    def _box_touches_foreign_via_pad(
        self, gi: int, box: Rect, net: str,
    ) -> bool:
        """Whether ``box`` contacts a foreign committed via landing pad.

        Closed-interval contact is electrical contact, so an edge touch rejects.
        The dedicated pad index intentionally records the via-pad subset even
        though Gridless also rejects contact with ordinary foreign wire.
        """
        _sep = _eg_attr("_sep", _base_sep)
        idx = self._via_pad_idx.get(gi)
        if idx is None:
            return False
        nb = _norm(box)
        if (
            _sep is _base_sep
            and type(idx) is _SpatialIndex
            and idx._native is not None
        ):
            return idx._native.has_foreign_contact(
                nb[0], nb[1], nb[2], nb[3], net
            )
        for (owner, r) in idx.query(nb):
            if owner == net:
                continue
            sx, sy = _sep(nb, _norm(r))
            if sx == 0 and sy == 0:
                return True
        return False

    def _via_pads_overlap_foreign(self, v: _ViaSite, net: str) -> bool:
        """True iff either landing pad of via ``v`` overlaps foreign committed metal
        (the bend-via M2 pad landing ON a foreign horizontal trunk -- the short HV
        discipline would otherwise trade the vertical-wire crossing for)."""
        bh, ah = self._via_pad_half_for(v)
        return (self._box_overlaps_foreign(
                    v.lo, (v.x - bh, v.y - bh, v.x + bh, v.y + bh), net)
                or self._box_overlaps_foreign(
                    v.lo + 1, (v.x - ah, v.y - ah, v.x + ah, v.y + ah), net))

    def _hv_decompose(
        self, net: str, corners: Sequence[Tuple[int, int, int, int]],
        climb_vias: List[_ViaSite], gi_h: int, gi_v: int, route_gi: int,
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """Re-layer ONE candidate corner-chain under HV discipline: every horizontal
        leg on ``gi_h``, every vertical leg on ``gi_v``, joined by a bend via wherever
        the running layer changes; the chain starts and ends at ``route_gi`` (the
        climbed columns), so the two pins' climb stacks still tie in.

        Judged with the engine's OWN hard legality only (``_wire_legal`` per leg,
        ``_via_legal`` per bend); a fully-legal decomposition is returned in
        the SAME ``(segs, vias)`` shape ``_route_edge`` returns. Any blocked leg or
        sealed bend via -> None (the caller tries the next shape, then the plain
        single-layer pattern). A diagonal leg (neither pure H nor V) is not a pattern
        product; it defers (None). No per-circuit branch; no wall-clock."""
        segs: List[_WireSeg] = []
        vias: List[_ViaSite] = list(climb_vias)
        cur = route_gi
        last_end: Optional[Tuple[int, int]] = None
        for (sx, sy, ex, ey) in corners:
            if (sx, sy) == (ex, ey):
                continue                          # zero-length leg
            if sy == ey:
                target = gi_h
            elif sx == ex:
                target = gi_v
            else:
                return None                        # not a rectilinear leg
            if cur != target:
                v = self._hv_bridge_via(sx, sy, cur, target, net)
                if v is None:
                    return None
                vias.append(v)
                cur = target
            if not self._wire_legal(target, sx, sy, ex, ey, net):
                return None
            segs.append(self._wire_segment(net, target, sx, sy, ex, ey))
            last_end = (ex, ey)
        if not segs:
            return None
        # return the running layer to route_gi at the final column so the second
        # pin's climb stack (which lands on route_gi) connects to the chain.
        if cur != route_gi and last_end is not None:
            v = self._hv_bridge_via(last_end[0], last_end[1], cur, route_gi, net)
            if v is None:
                return None
            vias.append(v)
        return segs, vias

    def _signal_candidate_lines(
        self, a2: Tuple[int, int], b2: Tuple[int, int], window: Rect,
    ) -> Tuple[List[int], List[int]]:
        """SPARSE, finger-density-INDEPENDENT candidate lines for ONE signal-net MST
        edge whose closed-form pattern was blocked -- the ``_candidate_lines``
        analogue of the rail drop's ``_rail_candidate_lines`` cap.

        The general ``_candidate_lines`` ingests EVERY foreign device-finger edge in
        the per-edge window, so a multi-pin signal net's window built a ~234x254 grid
        (the ``|xs|*|ys|*layers*3`` A* budget scales with it -> 80248 expansions,
        100-470s per net). Instead the fallback searches a BOUNDED grid built from
        ONLY the two endpoints plus a UNIFORM/COARSE pitch fill whose step widens so
        the line COUNT stays capped on EACH axis however large the window -- no
        obstacle-edge lines, so the graph does NOT scale with in-window finger
        density. Hard legality (``_wire_legal`` / ``_via_legal``) still rejects metal
        on a finger at expansion time; the sparse grid only bounds WHERE the edge may
        turn. Aligned to a global origin so tracks stay consistent net-to-net; no
        per-circuit branch. The exact pin-access / escape-landing coordinates are
        merged in by ``_route_edge`` (start/goal coords)."""
        wx0, wy0, wx1, wy1 = _norm(window)
        ax, ay = a2
        bx, by = b2
        pitch = self._res_cell
        _LINE_CAP = 24                      # per-axis cap -> a bounded grid
        xs: Set[int] = {ax, bx, wx0, wx1}
        ys: Set[int] = {ay, by, wy0, wy1}
        xspan = max(1, wx1 - wx0)
        xstep = max(pitch, xspan // _LINE_CAP)
        gx = (wx0 // xstep) * xstep
        while gx <= wx1:
            if gx >= wx0:
                xs.add(gx)
            gx += xstep
        yspan = max(1, wy1 - wy0)
        ystep = max(pitch, yspan // _LINE_CAP)
        gy = (wy0 // ystep) * ystep
        while gy <= wy1:
            if gy >= wy0:
                ys.add(gy)
            gy += ystep
        env_xs, env_ys = self._group_envelope_clearance_lines(wx0, wy0, wx1, wy1)
        xs.update(env_xs)
        ys.update(env_ys)
        return (
            self._on_manufacturing_grid_lines(xs, wx0, wx1),
            self._on_manufacturing_grid_lines(ys, wy0, wy1),
        )

    def _certified_group_envelopes(self) -> Tuple[Rect, ...]:
        """Return deduplicated certified ``@route_group`` envelopes during global routing."""
        if bool(getattr(self, "_active_group_route_local_closure", False)):
            return ()
        interfaces = getattr(self, "_group_route_global_interfaces_by_net", None)
        if not interfaces:
            return ()
        envelopes: Set[Rect] = set()
        for rows in interfaces.values():
            for env, _portal in rows:
                envelopes.add((int(env[0]), int(env[1]), int(env[2]), int(env[3])))
        return tuple(sorted(envelopes))

    def _group_envelope_clearance_lines(
        self, wx0: int, wy0: int, wx1: int, wy1: int
    ) -> Tuple[Set[int], Set[int]]:
        """Emit exterior wire/pad clearance lines and corridor midpoints around certified group envelopes."""
        envelopes = self._certified_group_envelopes()
        if not envelopes or not self._half_w:
            return set(), set()
        min_hw = max(1, min(int(h) for h in self._half_w))
        offsets: Set[int] = set()
        for gi in range(self._n_layers):
            hw = int(self._half_w[gi])
            if gi > 0 and hw >= 4 * min_hw:
                continue
            sp = int(self._space[gi])
            offsets.add(hw)
            offsets.add(hw + sp)
            if gi in self._via_pad_half:
                bh, ah = self._via_pad_half[gi]
                cut_ext = int(self._via_cut_extent(gi, None))
                bh_eff = max(int(bh), cut_ext)
                offsets.add(int(bh))
                offsets.add(bh_eff)
                offsets.add(bh_eff + sp)
                if gi + 1 < self._n_layers and int(self._half_w[gi + 1]) < 4 * min_hw:
                    ah_eff = max(int(ah), cut_ext)
                    offsets.add(int(ah))
                    offsets.add(ah_eff)
                    offsets.add(ah_eff + int(self._space[gi + 1]))
        grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 1) or 1))
        xs: Set[int] = set()
        ys: Set[int] = set()
        for ex0, ey0, ex1, ey1 in envelopes:
            for off in offsets:
                xv_lo = ((ex0 - off) // grid) * grid
                xv_hi = -(-(ex1 + off) // grid) * grid
                for xv in (xv_lo, xv_hi):
                    if wx0 <= xv <= wx1:
                        xs.add(xv)
                yv_lo = ((ey0 - off) // grid) * grid
                yv_hi = -(-(ey1 + off) // grid) * grid
                for yv in (yv_lo, yv_hi):
                    if wy0 <= yv <= wy1:
                        ys.add(yv)
        for idx_a, (ax0, ay0, ax1, ay1) in enumerate(envelopes):
            for bx0, by0, bx1, by1 in envelopes[idx_a + 1:]:
                if ax1 < bx0 and max(ay0, by0) <= min(ay1, by1):
                    mid_x = ((ax1 + bx0) // (2 * grid)) * grid
                    if wx0 <= mid_x <= wx1:
                        xs.add(mid_x)
                elif bx1 < ax0 and max(ay0, by0) <= min(ay1, by1):
                    mid_x = ((bx1 + ax0) // (2 * grid)) * grid
                    if wx0 <= mid_x <= wx1:
                        xs.add(mid_x)
                if ay1 < by0 and max(ax0, bx0) <= min(ax1, bx1):
                    mid_y = ((ay1 + by0) // (2 * grid)) * grid
                    if wy0 <= mid_y <= wy1:
                        ys.add(mid_y)
                elif by1 < ay0 and max(ax0, bx0) <= min(ax1, bx1):
                    mid_y = ((by1 + ay0) // (2 * grid)) * grid
                    if wy0 <= mid_y <= wy1:
                        ys.add(mid_y)
        return xs, ys

    def _causal_blocker_frontier_lines(
        self,
        net: str,
        xs: Sequence[int],
        ys: Sequence[int],
        window: Rect,
        allowed_layers: Sequence[int],
        *,
        minimum_rejection_count: int = 0,
        locality_envelope: Optional[Rect] = None,
        rejection_samples: Optional[Mapping[Tuple[Any, ...], Tuple[int, Rect]]] = None,
    ) -> Tuple[List[int], List[int]]:
        """Add exact hard-blocker skirt lines after a sparse search exhausts.

        The initial signal grid intentionally stays independent of device-finger
        density.  If that bounded grid cannot route, its legality checks have
        already identified the complete set of exact causal blocker rectangles,
        including committed geometry that remains immutable for the duration of
        this edge search.  One refinement pass may therefore add the exact
        touch-or-clear centre lines around those blockers.  This is evidence-driven
        graph refinement, not a denser initial enumeration and not a placement-move
        witness.

        No blocker is sliced or count-capped.  Every retained causal rectangle
        contributes deterministic frontiers for the candidate primitives that
        can touch it: wire/via pads for metal blockers and via cuts for cut
        blockers.  The existing A* expansion budget remains the sole solve bound.
        """
        wx0, wy0, wx1, wy1 = _norm(window)
        locality = (
            None if locality_envelope is None else _norm(locality_envelope)
        )
        refined_xs = set(int(value) for value in xs)
        refined_ys = set(int(value) for value in ys)
        allowed = set(int(value) for value in allowed_layers)
        samples = (getattr(self, "_hard_rejection_samples", {})
                   if rejection_samples is None else rejection_samples)

        for cause in sorted(samples):
            rejection_count, _candidate_extent = samples[cause]
            if int(rejection_count) < int(minimum_rejection_count):
                continue
            (
                category,
                _owner,
                gi,
                candidate_half,
                required_space,
                blocker_rect,
                _blocker_kind,
                _terminal_ref,
            ) = cause
            gi = int(gi)
            if (
                gi not in allowed
                or required_space is None
                or blocker_rect is None
            ):
                continue
            offsets: Set[int] = set()
            strict_contact = category.endswith("_contact")
            if category in {
                "fixed_metal",
                "fixed_same_net_via_pad_spacing",
                "committed_metal_contact",
                "committed_via_pad_contact",
                "committed_metal_spacing",
                "committed_via_pad_spacing",
            }:
                candidate_extent = (
                    self._hw(net, gi)
                    if candidate_half is None
                    else int(candidate_half)
                )
            elif category in {
                "fixed_cut",
                "attempt_cut",
                "committed_self_cut",
                "committed_cut_contact",
                "committed_cut_spacing",
            }:
                candidate_extent = (
                    self._via_cut_extent(gi, None)
                    if candidate_half is None
                    else int(candidate_half)
                )
            else:
                continue
            offsets.add(
                candidate_extent
                + int(required_space)
                + (1 if strict_contact else 0)
            )

            for offset in sorted(offsets):
                x_frontiers, y_frontiers = self._clearance_frontier_coordinates(
                    blocker_rect, offset
                )
                # A local first pass uses both coordinates of the exact grown
                # blocker.  Closed touch, the actual primitive footprint and the
                # strict-contact extra nm all retain their existing authority.
                # The default projection remains complete for legacy callers.
                if locality is not None and (
                    x_frontiers[1] < locality[0]
                    or x_frontiers[0] > locality[2]
                    or y_frontiers[1] < locality[1]
                    or y_frontiers[0] > locality[3]
                ):
                    continue
                for xv in x_frontiers:
                    if wx0 <= xv <= wx1:
                        refined_xs.add(xv)
                for yv in y_frontiers:
                    if wy0 <= yv <= wy1:
                        refined_ys.add(yv)
        return (
            self._on_manufacturing_grid_lines(
                refined_xs, wx0, wx1, preserve_frontier_sides=True
            ),
            self._on_manufacturing_grid_lines(
                refined_ys, wy0, wy1, preserve_frontier_sides=True
            ),
        )

    @staticmethod
    def _local_search_domain_exhausted(failure_kind: Optional[str]) -> bool:
        """Only completed local searches can request an immutable-boundary replay.

        A disconnected candidate graph and a fully exhausted A* graph both
        describe the searched window, not the entire physical envelope. Work
        limits and unrelated typed failures retain their original authority.
        """
        return failure_kind in ("exhausted", "static_disconnected", "symbolic_domain_incomplete")

    def _causal_blocker_escape_window(
        self, window: Rect, allowed_layers: Sequence[int],
    ) -> Rect:
        """Expose real immutable blocker edges beyond a failed local graph.

        This changes only candidate coordinates, never geometry or legality.
        A finite local window contained inside a MIM cut domain cannot prove
        that its M1 terminal has no legal escape beyond that domain's edge.
        """
        wx0, wy0, wx1, wy1 = _norm(window)
        bx0, by0, bx1, by1 = _norm(getattr(self, "_bounds", window))
        envelope = getattr(self, "_routing_envelope", None)
        if envelope is not None:
            bx0, by0 = max(bx0, envelope.x0), max(by0, envelope.y0)
            bx1, by1 = min(bx1, envelope.x1), min(by1, envelope.y1)
        if bool(getattr(self, "_active_group_route_local_closure", False)):
            local_iface = getattr(self, "_active_group_route_local_interface", None)
            if local_iface is not None:
                ex0, ey0, ex1, ey1 = (int(v) for v in local_iface[0])
                if wx0 >= ex0 and wy0 >= ey0 and wx1 <= ex1 and wy1 <= ey1:
                    bx0, by0 = max(bx0, ex0), max(by0, ey0)
                    bx1, by1 = min(bx1, ex1), min(by1, ey1)
        allowed = set(int(value) for value in allowed_layers)
        quantum = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
        x0, y0, x1, y1 = wx0, wy0, wx1, wy1
        for cause in getattr(self, "_hard_rejection_samples", {}):
            category, _owner, gi, half, spacing, blocker, _kind, _ref = cause
            if (blocker is None or int(gi) not in allowed
                    or not str(category).startswith(("fixed_", "mim_"))):
                continue
            rx0, ry0, rx1, ry1 = _norm(blocker)
            if rx1 < wx0 or rx0 > wx1 or ry1 < wy0 or ry0 > wy1:
                continue
            clearance = max(0, int(half or 0)) + max(0, int(spacing or 0))
            left = ((rx0 - clearance) // quantum) * quantum
            bottom = ((ry0 - clearance) // quantum) * quantum
            right = -(-(rx1 + clearance) // quantum) * quantum
            top = -(-(ry1 + clearance) // quantum) * quantum
            x0, y0 = min(x0, left), min(y0, bottom)
            x1, y1 = max(x1, right), max(y1, top)
        return (max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1))

    def _causal_blocker_frontier_stages(
        self,
        net: str,
        xs: Sequence[int],
        ys: Sequence[int],
        window: Rect,
        allowed_layers: Sequence[int],
        *,
        locality_envelope: Optional[Rect] = None,
        rejection_samples: Optional[Mapping[Tuple[Any, ...], Tuple[int, Rect]]] = None,
    ) -> Tuple[Tuple[List[int], List[int]], ...]:
        """Cumulative DRC-frontier strata in geometric frequency bands.

        Every exact causal blocker retains authority.  Frequency only orders when
        its frontier enters the graph: repeatedly hit obstacles form the first
        small refinement, then power-of-two thresholds monotonically add the
        remaining evidence.  The number of A* replays is therefore logarithmic in
        the largest observed rejection count instead of linear in the number of
        distinct counts.  Threshold one still materializes every sampled blocker;
        identical grids are elided deterministically.
        """
        samples = (getattr(self, "_hard_rejection_samples", {})
                   if rejection_samples is None else rejection_samples)
        maximum_count = max(
            (int(count) for count, _extent in samples.values()),
            default=0,
        )
        thresholds = tuple(
            1 << shift
            for shift in range(maximum_count.bit_length() - 1, -1, -1)
        )
        base = (tuple(sorted(set(xs))), tuple(sorted(set(ys))))
        seen = {base}
        stages = []
        locality_options = (
            {} if locality_envelope is None
            else {"locality_envelope": locality_envelope}
        )
        if rejection_samples is not None:
            locality_options["rejection_samples"] = rejection_samples
        for threshold in thresholds:
            refined_xs, refined_ys = self._causal_blocker_frontier_lines(
                net,
                xs,
                ys,
                window,
                allowed_layers,
                minimum_rejection_count=threshold,
                **locality_options,
            )
            key = (tuple(refined_xs), tuple(refined_ys))
            if key in seen:
                continue
            seen.add(key)
            stages.append((refined_xs, refined_ys))
        return tuple(stages)

    def _signal_causal_frontier_passes(
        self,
        net: str,
        xs: Sequence[int],
        ys: Sequence[int],
        window: Rect,
        allowed_layers: Sequence[int],
        access_states: Sequence[Tuple[int, int, int]],
    ) -> Tuple[Tuple[str, Tuple[Tuple[List[int], List[int]], ...]], ...]:
        """Try local causal strata before the unchanged complete projection.

        Component access can extend beyond the nominal sparse window.  Include
        every admitted endpoint and existing axis extremum in the locality
        envelope; never use that envelope to restrict legal search geometry.
        Every legacy frequency stage remains available as a separate fallback,
        including after a local stage reaches its existing per-edge work cap.
        """
        complete = self._causal_blocker_frontier_stages(
            net, xs, ys, window, allowed_layers
        )
        local = complete
        if complete:
            wx0, wy0, wx1, wy1 = _norm(window)
            domain_xs = [wx0, wx1, *map(int, xs)]
            domain_ys = [wy0, wy1, *map(int, ys)]
            for x, y, _gi in access_states:
                domain_xs.append(int(x))
                domain_ys.append(int(y))
            locality = (
                min(domain_xs), min(domain_ys), max(domain_xs), max(domain_ys)
            )
            local = self._causal_blocker_frontier_stages(
                net, xs, ys, window, allowed_layers,
                locality_envelope=locality,
            )
        passes = (
            (("complete", complete),)
            if local == complete
            else (("local", local), ("complete", complete))
        )
        # These are planned stage counts/projected graph maxima, not executed
        # work; A* also admits access and pattern-proof coordinates.  A collapsed
        # identical pass has no local stages.  Actual work still accumulates in
        # the existing solve counter across both passes.
        details = dict(getattr(self, "_route_failure_details", ()))
        layer_states = max(1, len(set(map(int, allowed_layers)))) * 3
        for scope in ("local", "complete"):
            stages = next((rows for name, rows in passes if name == scope), ())
            details[f"frontier_{scope}_stage_count"] = len(stages)
            details[f"frontier_{scope}_max_graph_states"] = max(
                (len(stage_xs) * len(stage_ys) * layer_states
                 for stage_xs, stage_ys in stages),
                default=0,
            )
        self._route_failure_details = tuple(sorted(details.items()))
        return passes

    def _group_endpoint_layer_frontier_prepasses(
        self,
        net: str,
        xs: Sequence[int],
        ys: Sequence[int],
        window: Rect,
        allowed_layers: Sequence[int],
        access_states: Sequence[Tuple[int, int, int]],
        *,
        rejection_samples: Optional[Mapping[Tuple[Any, ...], Tuple[int, Rect]]] = None,
    ) -> Tuple[Tuple[str, Tuple[Tuple[List[int], List[int]], ...]], ...]:
        """Prioritize nearby endpoint-layer blocker axes, retaining full search.

        This projects only candidate axis events.  It does not restrict the
        routing window, the allowed route layers, blocker legality or the
        complete causal passes that follow it.  All admitted endpoint states
        define the corridor, including remote physical component accesses.
        """
        states = tuple(tuple(map(int, state)) for state in access_states)
        if not states:
            return ()
        pitch = int(self._res_cell)
        if pitch <= 0:
            return ()
        admitted = set(map(int, allowed_layers))
        endpoint_layers = tuple(sorted({state[2] for state in states} & admitted))
        if not endpoint_layers:
            return ()
        locality = (min(state[0] for state in states) - pitch,
                    min(state[1] for state in states) - pitch,
                    max(state[0] for state in states) + pitch,
                    max(state[1] for state in states) + pitch)
        projections = ((endpoint_layers,) + tuple((gi,) for gi in endpoint_layers)
                       if len(endpoint_layers) > 1 else (endpoint_layers,))
        result = []
        for index, layers in enumerate(projections):
            options = {"locality_envelope": locality}
            if rejection_samples is not None:
                options["rejection_samples"] = rejection_samples
            stages = self._causal_blocker_frontier_stages(
                net, xs, ys, window, layers, **options)
            if stages:
                result.append((f"endpoint_layer_prepass_{index}", stages))
        return tuple(result)

    def _causal_frontier_graph_within_budget(
        self,
        xs: Sequence[int],
        ys: Sequence[int],
        allowed_layers: Sequence[int],
        *,
        nominal_work: bool = False,
    ) -> bool:
        """Fail typed before allocating a pathological refined Cartesian graph."""
        state_count = (
            len(xs)
            * len(ys)
            * max(1, len(set(int(value) for value in allowed_layers)))
            * 3
        )
        graph_cap = (
            min(int(self.expansion_cap), int(self.max_expansions))
            if nominal_work
            else int(self.expansion_cap)
        )
        if state_count <= graph_cap:
            return True
        self._edge_fail = (
            "budget",
            0,
            graph_cap,
            getattr(self, "_solve_expansions_total", 0),
            getattr(self, "_solve_expansion_budget", 0),
            len(xs),
            len(ys),
            len(set(int(value) for value in allowed_layers)),
            "causal_frontier_graph",
        )
        self._route_failure_details += (
            ("frontier_graph_states", int(state_count)),
            ("frontier_graph_cap", int(graph_cap)),
        )
        return False

