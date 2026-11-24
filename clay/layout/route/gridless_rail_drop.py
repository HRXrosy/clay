"""Rail-spine drop routing, committed-state rebuild, and length-constraint tuning mixin."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from clay.layout.route.constraints import HARD, Length, SOFT
from clay.layout.route.engine import NetRoute, Pin
from clay.layout.route.engine_maze import _NM_PER_UM as _base_NM_PER_UM
from clay.layout.route.gridless_coupling import (
    CouplingContext,
    has_third_party_non_ground_conductors,
)
from clay.layout.route.gridless_types import (
    NegotiationRouteStop,
    NegotiationStopReason,
    Rect,
    _Book,
    _COMMITTED_BLOCKER_REJECTION_CATEGORIES,
    _HardConnectivityWitness,
    _OwnedTerminalAtom,
    _PatternSearchIncumbent,
    _PlanarEndpointDomain,
    _RoutedNet as _base_RoutedNet,
    _SpatialIndex,
    _VIA_COST,
    _ViaSite,
    _WireSeg,
    _closed_rects_touch,
    _deck_open_route_nets,
    _eg_attr,
    _norm,
    _rects_clear as _base_rects_clear,
    _seg_bbox,
    _sep as _base_sep,
)


class _GridlessRailDropMixin:
    """Power/ground rail-spine drop routing and post-commit length retuning."""

    # ------------------------------------------------------- rail-spine drop
    @staticmethod
    def _nearest_rail_point(x: int, y: int,
                            rail: Sequence[Tuple[int, Rect]]) -> Tuple[int, int, int]:
        """The (gi_rail, gx, gy) landing on the NEAREST declared rail rect for a pin
        at (x, y): ``gx`` is the pin x clamped into the rail's span (a straight-down
        drop keeps the same x), ``gy`` the pin y snapped to the rail's NEAR edge
        (the shortest vertical reach onto the rail band). Manhattan-nearest across
        every rail this net declares (usually one)."""
        best: Optional[Tuple[int, int, int, int, int, int, int, int]] = None
        for (rgi, (rx0, ry0, rx1, ry1)) in rail:
            gx = min(max(x, rx0), rx1)
            gy = min(max(y, ry0), ry1)
            d = abs(x - gx) + abs(y - gy)
            candidate = (d, rgi, rx0, ry0, rx1, ry1, gx, gy)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        _d, rgi, _rx0, _ry0, _rx1, _ry1, gx, gy = best
        return rgi, gx, gy

    @staticmethod
    def _rail_rect_at_point(
        gx: int, gy: int, rgi: int, rail: Sequence[Tuple[int, Rect]],
    ) -> Optional[Rect]:
        """Select an actual owned landing rectangle in nearest-rail tie order."""
        matching = []
        for gi, raw_rect in rail:
            rect = _norm(raw_rect)
            if gi == rgi and rect[0] <= gx <= rect[2] and rect[1] <= gy <= rect[3]:
                matching.append(rect)
        return min(matching) if matching else None

    def _rail_axis_at_point(
        self, gx: int, gy: int, rgi: int, rail: Sequence[Tuple[int, Rect]],
    ) -> str:
        """Resolve the long axis of the actual owned rail at a landing.

        The rectangle tie-break matches ``_nearest_rail_point``. A missing
        rectangle retains the historical horizontal-rail helper convention;
        it does not create a landing or authorize any metal.
        """
        rect = self._rail_rect_at_point(gx, gy, rgi, rail)
        if rect is None:
            return "H"
        rx0, ry0, rx1, ry1 = rect
        return "V" if ry1 - ry0 > rx1 - rx0 else "H"

    def _rail_goal_set(
        self, gx: int, gy: int, rgi: int, rail: Sequence[Tuple[int, Rect]],
    ) -> Dict[Tuple[int, int, int], Optional[_ViaSite]]:
        """A bounded goal set on both the local rail edge and its interior.

        The historical straight-drop
        projection ``(gx, gy)`` plus six routing-pitch samples on each side along
        the owned rail (clamped to its span). These edge goals cover the existing
        six-pitch local transverse window. Every goal is ON the owned rail metal, so
        the A* may land ANYWHERE on the LOCAL rail segment (robust to a blocked exact
        column) while the drop stays local. The rail is owned metal -- landing on it
        joins the net; no goal-side via is needed (all goal vias are ``None``).

        The near edge alone is not a complete cross-section: a foreign shape
        may legally abut the rail's spacing boundary while forbidding every
        newly emitted pad centered on that edge. Retain those historical goals
        and add manufacturing-grid points on the actual owned band's center.
        This changes only candidate coordinates, never fixed geometry, the
        route budget, or the via/wire/deck legality checks.
        """
        goals: Dict[Tuple[int, int, int], Optional[_ViaSite]] = {(gx, gy, rgi): None}
        edge_rect = self._rail_rect_at_point(gx, gy, rgi, rail)
        if edge_rect is not None:
            rx0, ry0, rx1, ry1 = edge_rect
            horizontal = rx1 - rx0 >= ry1 - ry0
            pitch = self._res_cell
            for k in range(1, 7):
                if horizontal:
                    for sx in (gx - k * pitch, gx + k * pitch):
                        goals[(min(max(sx, rx0), rx1), gy, rgi)] = None
                else:
                    for sy in (gy - k * pitch, gy + k * pitch):
                        goals[(gx, min(max(sy, ry0), ry1), rgi)] = None
        for rail_gi, raw_rect in rail:
            rx0, ry0, rx1, ry1 = _norm(raw_rect)
            if rail_gi != rgi or not (rx0 <= gx <= rx1 and ry0 <= gy <= ry1):
                continue
            env_xs, env_ys = self._group_envelope_clearance_lines(
                rx0, ry0, rx1, ry1
            )
            horizontal = rx1 - rx0 >= ry1 - ry0
            if horizontal:
                raw_along = set(
                    min(max(gx + k * self._res_cell, rx0), rx1)
                    for k in range(-6, 7)
                )
                raw_along.update(env_xs)
                along = self._on_manufacturing_grid_lines(
                    tuple(sorted(raw_along)), rx0, rx1
                )
                across = self._on_manufacturing_grid_lines(
                    ((ry0 + ry1) // 2,), ry0, ry1,
                    preserve_frontier_sides=True)
                for sx in env_xs:
                    if rx0 <= sx <= rx1:
                        goals[(int(sx), gy, rgi)] = None
                goals.update({(sx, sy, rgi): None for sx in along for sy in across})
            else:
                raw_along = set(
                    min(max(gy + k * self._res_cell, ry0), ry1)
                    for k in range(-6, 7)
                )
                raw_along.update(env_ys)
                along = self._on_manufacturing_grid_lines(
                    tuple(sorted(raw_along)), ry0, ry1
                )
                across = self._on_manufacturing_grid_lines(
                    ((rx0 + rx1) // 2,), rx0, rx1,
                    preserve_frontier_sides=True)
                for sy in env_ys:
                    if ry0 <= sy <= ry1:
                        goals[(gx, int(sy), rgi)] = None
                goals.update({(sx, sy, rgi): None for sx in across for sy in along})
        return goals

    def _far_rail_pattern_goals(
        self,
        gx: int,
        gy: int,
        rgi: int,
        rail: Sequence[Tuple[int, Rect]],
        local_window: Rect,
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Return bounded, deterministic owned-rail portals outside local A*.

        A full-width edge rail is already emitted same-net conductor.  A fixed
        cross-cut can separate a terminal from every goal in the six-pitch local
        slot even though a closed-form upper-metal dogleg to another point on that
        same rail is legal.  Do not widen the residual graph: sample a constant-size
        portal set across the exact owned span for pattern routing only.
        """
        wx0, wy0, wx1, wy1 = _norm(local_window)
        candidates = set()
        _PORTAL_INTERVAL_CAP = 24
        for rail_gi, raw_rect in rail:
            rx0, ry0, rx1, ry1 = _norm(raw_rect)
            if rail_gi != rgi:
                continue
            horizontal = rx1 - rx0 >= ry1 - ry0
            if horizontal:
                if not ry0 <= gy <= ry1:
                    continue
                low, high = rx0, rx1
                window_low, window_high = wx0, wx1
            else:
                if not rx0 <= gx <= rx1:
                    continue
                low, high = ry0, ry1
                window_low, window_high = wy0, wy1
            span = max(1, high - low)
            raw_coords = {
                low + (span * index) // _PORTAL_INTERVAL_CAP
                for index in range(_PORTAL_INTERVAL_CAP + 1)
            }
            env_xs, env_ys = self._group_envelope_clearance_lines(
                rx0, ry0, rx1, ry1
            )
            raw_coords.update(env_xs if horizontal else env_ys)
            for raw_coordinate in sorted(raw_coords):
                coordinates = self._on_manufacturing_grid_lines(
                    (raw_coordinate,), low, high
                )
                candidates.update(
                    (
                        (int(coordinate), int(gy), int(rgi))
                        if horizontal else
                        (int(gx), int(coordinate), int(rgi))
                    )
                    for coordinate in coordinates
                    if coordinate < window_low or coordinate > window_high
                )
        return tuple(
            sorted(
                candidates,
                key=lambda goal: (abs(goal[0] - gx) + abs(goal[1] - gy), goal),
            )
        )

    def _far_rail_portal_pattern(
        self,
        net: str,
        pin: Tuple[int, int, int],
        starts: Sequence[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ],
        rgi: int,
        gx: int,
        gy: int,
        rail: Sequence[Tuple[int, Rect]],
        local_window: Rect,
        allowed_layers: Sequence[int],
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """Try legal closed-form doglegs to far points on one owned rail.

        This is a pattern selector, not a second search graph.  The source access
        domain is exact, the goal conductor is already owned by ``net``, and every
        candidate is judged by the same via/wire legality as ordinary signal
        patterns.  The first legal result in stable distance order is sufficient;
        all returned patterns are zero-congestion hard-legal candidates.
        """
        goals = self._far_rail_pattern_goals(
            gx, gy, rgi, rail, local_window
        )
        if not goals or not starts:
            return None
        for goal in goals:
            ordered_starts = tuple(
                sorted(
                    starts,
                    key=lambda item: (
                        abs(item[0][0] - goal[0])
                        + abs(item[0][1] - goal[1]),
                        self._access_state_key(item),
                    ),
                )
            )
            for start in ordered_starts:
                found = self._signal_edge_pattern(
                    net,
                    pin,
                    goal,
                    acc_a=(start,),
                    acc_b=((goal, None),),
                    lg=allowed_layers,
                    trunk_gi=rgi,
                )
                if found is not None:
                    return found
        return None

    def _rail_attempt_tree_pattern(
        self,
        net: str,
        pin: Tuple[int, int, int],
        starts: Sequence[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ],
        attempt_segs: Sequence[_WireSeg],
        allowed_layers: Sequence[int],
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """Join one rail drop to exact same-net metal from an earlier drop.

        A remote owned-rail portal is a necessary fallback when a local drop slot
        is cut, but independently replaying that portal for every pin creates a
        stack of cell-spanning parallel conductors.  Earlier drops in this one
        route attempt are already an exact connected tree to the declared rail.
        Project the current pin onto their emitted segment centerlines and try the
        ordinary hard-legal closed-form signal pattern to that tree before adding
        another remote branch.

        The authority is strictly solve-local: ``attempt_segs`` is reset with the
        rail-net transaction and contains only this net's previously committed
        drops.  No historical route, foreign net, sampled coordinate, or retained
        rejection participates.  Failure changes no ledger and leaves the existing
        owned-rail portal and residual-search fallbacks intact.
        """
        if not starts or not attempt_segs:
            return None
        x, y, gi = pin
        allowed = set(allowed_layers)
        goals = set()
        for seg in attempt_segs:
            if seg.gi not in allowed:
                continue
            ax, ay, bx, by = seg.ax, seg.ay, seg.bx, seg.by
            if ay == by:
                goal = (min(max(x, min(ax, bx)), max(ax, bx)), ay, seg.gi)
            elif ax == bx:
                goal = (ax, min(max(y, min(ay, by)), max(ay, by)), seg.gi)
            else:
                continue
            goals.add(goal)
        for goal in sorted(
            goals,
            key=lambda item: (
                abs(item[0] - x) + abs(item[1] - y),
                item,
            ),
        ):
            if goal == (x, y, gi):
                return [], []
            ordered_starts = tuple(
                sorted(
                    starts,
                    key=lambda item: (
                        abs(item[0][0] - goal[0])
                        + abs(item[0][1] - goal[1]),
                        self._access_state_key(item),
                    ),
                )
            )
            for start in ordered_starts:
                found = self._signal_edge_pattern(
                    net,
                    pin,
                    goal,
                    acc_a=(start,),
                    acc_b=((goal, None),),
                    lg=allowed_layers,
                    trunk_gi=goal[2],
                )
                if found is not None:
                    return found
        return None

    def _has_live_rail_pattern_blocker(self) -> bool:
        """Whether rejection evidence names a blocker still live in this world.

        A far owned-rail portal is a bounded alternative to the local A* slot, not
        an unconditional cell-spanning preference.  Authorize it only after the
        ordinary pattern/access attempt records one exact blocker rectangle and
        that same owner+rectangle is still present in the corresponding live
        fixed- or committed-route index.  Soft congestion, stale samples, and
        synthetic transaction evidence therefore retain the historical local
        replay order.
        """
        samples = getattr(self, "_hard_rejection_samples", {})
        for cause in sorted(samples):
            category, owner, gi, _half, _space, blocker_rect, *_rest = cause
            if blocker_rect is None:
                continue
            expected = tuple(_norm(blocker_rect))
            if category == "group_route_global_interior":
                interfaces = getattr(
                    self, "_group_route_global_interfaces_by_net", {}
                ).get(str(owner), ())
                if any(
                    tuple(_norm(envelope)) == expected
                    for envelope, _portal in interfaces
                ):
                    return True
                continue
            if category == "fixed_metal":
                indexes = (getattr(self, "_obs_idx", {}).get(int(gi)),)
            elif category in {
                "committed_metal_contact",
                "committed_metal_spacing",
            }:
                indexes = (getattr(self, "_metal_idx", {}).get(int(gi)),)
            elif category in {
                "committed_via_pad_contact",
                "committed_via_pad_spacing",
            }:
                indexes = (getattr(self, "_via_pad_idx", {}).get(int(gi)),)
            elif category in {
                "committed_cut_contact",
                "committed_cut_spacing",
            }:
                indexes = (getattr(self, "_cut_idx", {}).get(int(gi)),)
            else:
                continue
            index = indexes[0]
            if index is None:
                continue
            if any(
                (
                    ("<unowned>" if live_owner is None else str(live_owner))
                    == str(owner)
                    and tuple(_norm(live_rect)) == expected
                )
                for live_owner, live_rect in index.query(expected)
            ):
                return True
        return False

    def _has_live_fixed_rail_pattern_blocker(self) -> bool:
        """Backward-compatible spelling for direct diagnostic callers."""
        return self._has_live_rail_pattern_blocker()

    def _nearest_access_frontier(
        self,
        starts: Sequence[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ],
        goals: Mapping[
            Tuple[int, int, int], Optional[_ViaSite]
        ],
    ) -> Tuple[
        Tuple[Tuple[int, int, int], Optional[_ViaSite]], ...
    ]:
        """Return every source attaining the minimum target-distance bound.

        A drawn terminal component can expose hundreds of dense legal via
        anchors inside a few microns.  They are one electrical component, but
        merging every anchor coordinate into the first A* graph recreates a
        Cartesian whole-component search.  Analytical routers start from the
        component points nearest the target index.  Clay uses the same pattern
        with its exact obstacle-free weighted-Manhattan lower bound, retaining
        all ties.  The caller preserves the complete source domain as a later
        fail-close fallback, so this frontier is an ordering stage, never a cap.
        """
        ordered_starts = tuple(sorted(starts, key=self._access_state_key))
        ordered_goals = tuple(
            sorted(goals.items(), key=self._access_state_key)
        )
        if not ordered_starts or not ordered_goals:
            return ordered_starts

        def terminal_lower_bound(via: Optional[_ViaSite]) -> int:
            if via is None:
                return 0
            return _VIA_COST + self._functional_single_via_penalty(via)

        ranked = []
        for start in ordered_starts:
            state, via = start
            lower_bound = terminal_lower_bound(via) + min(
                self._heuristic(*state, goal_state)
                + terminal_lower_bound(goal_via)
                for goal_state, goal_via in ordered_goals
            )
            ranked.append((int(lower_bound), start))
        minimum = min(item[0] for item in ranked)
        return tuple(start for score, start in ranked if score == minimum)

    @staticmethod
    def _access_cloud_is_cartesian_dense(
        items: Sequence[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ],
    ) -> bool:
        """Whether access points outnumber their independent planar axes."""
        points = tuple(items)
        if not points:
            return False
        axis_count = len({item[0][0] for item in points}) + len({
            item[0][1] for item in points
        })
        return len(points) > axis_count

    def _nearest_access_pair_frontier(
        self,
        starts: Sequence[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ],
        goals: Sequence[
            Tuple[Tuple[int, int, int], Optional[_ViaSite]]
        ],
    ) -> Tuple[
        Tuple[Tuple[Tuple[int, int, int], Optional[_ViaSite]], ...],
        Tuple[Tuple[Tuple[int, int, int], Optional[_ViaSite]], ...],
    ]:
        """Nearest-target staging for dense source and/or goal components."""
        ordered_starts = tuple(sorted(starts, key=self._access_state_key))
        ordered_goals = tuple(sorted(goals, key=self._access_state_key))
        if not ordered_starts or not ordered_goals:
            return ordered_starts, ordered_goals
        source_dense = self._access_cloud_is_cartesian_dense(ordered_starts)
        goal_dense = self._access_cloud_is_cartesian_dense(ordered_goals)
        if source_dense and goal_dense:
            # Two dense components require a joint pair frontier; independently
            # minimizing each side can select a locally near but globally inferior
            # access pair.  The unified domain lets the shared route objective pick
            # the real component escape (including a farther low-congestion seed).
            return ordered_starts, ordered_goals
        frontier_starts = (
            self._nearest_access_frontier(
                ordered_starts,
                {state: via for state, via in ordered_goals},
            )
            if source_dense
            else ordered_starts
        )
        frontier_goals = (
            self._nearest_access_frontier(
                ordered_goals,
                {state: via for state, via in ordered_starts},
            )
            if goal_dense
            else ordered_goals
        )
        return frontier_starts, frontier_goals

    def _rail_drop_window(
        self, x: int, y: int, gx: int, gy: int, *, rail_axis: str = "H",
    ) -> Rect:
        """The LOCAL per-drop search window for ONE rail drop: the pin ``(x, y)``
        and its straight-down rail projection ``(gx, gy)`` grown by a bounded
        margin, clamped to the cell bounds.

        A rail drop is a LOCAL VERTICAL SLOT onto the owned edge rail. Its
        x-CORRIDOR must be a few routing pitches (room to sidestep ONE blocking
        finger) and INDEPENDENT of the vertical reach; the y-window spans the full
        pin->rail reach so a blocked column can still detour ALONG the drop.
        Decoupling the two is what keeps ``_candidate_lines`` bounded: coupling the
        x-corridor to the vertical span saturated the margin to 30um -> a 60um-wide
        window -> ~107 uniform-fill x-lines + every in-window device finger -> the
        ``|xs|*|ys|*layers*3`` A* budget blew up and the supply-net drops
        budget-exhausted (the 9/15 -> 0/15 regressor of 86ed4cf). This makes each
        drop TRACTABLE (searchable), NOT SHORT -- the drop still spans the full
        vertical reach; shortening it is the layout job (loose placement), not the
        router's. For a vertical rail, transpose the same bounded construction:
        the drop travels horizontally and its local transverse slot is in y.
        Direct callers without rail geometry retain the horizontal-rail default.
        """
        if rail_axis not in {"H", "V"}:
            raise ValueError(f"unknown rail axis: {rail_axis!r}")
        orig_x, orig_y, orig_gx, orig_gy = x, y, gx, gy
        orig_bx0, orig_by0, orig_bx1, orig_by1 = self._bounds
        bx0, by0, bx1, by1 = orig_bx0, orig_by0, orig_bx1, orig_by1
        if rail_axis == "V":
            x, y, gx, gy = y, x, gy, gx
            bx0, by0, bx1, by1 = by0, bx0, by1, bx1
        axs = (x, gx)
        ays = (y, gy)
        # x-CORRIDOR: a few routing pitches each side (~3.36um at the gf180 560nm
        # pitch) -- room to sidestep ONE blocking finger -- INDEPENDENT of the
        # vertical reach. y-WINDOW: half the pin->rail reach (bounded), so a blocked
        # column can still detour ALONG the drop. Decoupled so a 12-119um drop no
        # longer saturates the x-margin to 30um (the ~110-fill-line explosion).
        x_margin = 6 * self._res_cell
        y_margin = min(30000, max(12000, (max(ays) - min(ays)) // 2))
        window = (
            max(bx0, min(axs) - x_margin), max(by0, min(ays) - y_margin),
            min(bx1, max(axs) + x_margin), min(by1, max(ays) + y_margin),
        )
        real_win = (
            (window[1], window[0], window[3], window[2])
            if rail_axis == "V"
            else window
        )
        envelopes = self._certified_group_envelopes()
        if envelopes:
            wx0, wy0, wx1, wy1 = real_win
            changed = True
            while changed:
                changed = False
                for ex0, ey0, ex1, ey1 in envelopes:
                    if rail_axis == "V":
                        r_lo, r_hi = min(orig_x, orig_gx), max(orig_x, orig_gx)
                        if max(r_lo, ex0) < min(r_hi, ex1) and ey0 <= wy1 and ey1 >= wy0:
                            new_wy0 = min(wy0, max(orig_by0, ey0 - x_margin))
                            new_wy1 = max(wy1, min(orig_by1, ey1 + x_margin))
                            if new_wy0 != wy0 or new_wy1 != wy1:
                                wy0, wy1 = new_wy0, new_wy1
                                changed = True
                    else:
                        r_lo, r_hi = min(orig_y, orig_gy), max(orig_y, orig_gy)
                        if max(r_lo, ey0) < min(r_hi, ey1) and ex0 <= wx1 and ex1 >= wx0:
                            new_wx0 = min(wx0, max(orig_bx0, ex0 - x_margin))
                            new_wx1 = max(wx1, min(orig_bx1, ex1 + x_margin))
                            if new_wx0 != wx0 or new_wx1 != wx1:
                                wx0, wx1 = new_wx0, new_wx1
                                changed = True
            real_win = (wx0, wy0, wx1, wy1)
        return real_win

    def _rail_candidate_lines(
        self, pin: Tuple[int, int], rail_pt: Tuple[int, int], window: Rect,
        *, rail_axis: str = "H",
    ) -> Tuple[List[int], List[int]]:
        """SPARSE, finger-density-INDEPENDENT candidate lines for ONE rail drop.

        A rail drop is a near-straight LOCAL slot: the pin climbs UP one metal
        (``_escape_landing`` / a path via) and runs OVER the cell on an upper metal
        to land on the owned rail, so the M1 device fingers it passes are NOT
        candidate turn points. The general ``_candidate_lines`` ingests EVERY foreign
        obstacle edge in the window, and a rail drop's window spans the full (tall)
        pin->rail reach -- so a dense band produced ~1286 candidate y-lines, a
        ~10^6-state graph PER DROP, and 25 drops became the measured 5717s gnd flood
        (the per-edge A* budget is ``|xs|*|ys|*layers*3``, so a bigger graph is a
        LONGER search).

        Instead the drop searches a BOUNDED grid built from ONLY: the pin + rail
        anchors, a UNIFORM PITCH FILL across the (S1-bounded) x-corridor, and a
        COARSE fill across the vertical reach whose step widens so the y-line COUNT
        stays capped however tall the drop. No obstacle-edge lines -> the graph does
        NOT scale with in-window finger density. Hard legality
        (``_wire_legal`` / ``_via_legal``) still rejects placing metal on a finger at
        expansion time; the sparse grid only bounds WHERE the drop may turn. Aligned
        to a global origin so tracks stay consistent net-to-net; no per-circuit
        branch. The escape-landing / rail-goal coordinates are merged in by
        ``_route_edge`` (start/goal coords), so the exact entry + landing are always
        present. A vertical owned rail uses the exact transposed construction;
        this bounds the line count across a horizontal drop's full reach too.
        """
        if rail_axis not in {"H", "V"}:
            raise ValueError(f"unknown rail axis: {rail_axis!r}")
        wx0, wy0, wx1, wy1 = _norm(window)
        env_xs, env_ys = self._group_envelope_clearance_lines(wx0, wy0, wx1, wy1)
        px, py = pin
        gx, gy = rail_pt
        if rail_axis == "V":
            px, py, gx, gy = py, px, gy, gx
            wx0, wy0, wx1, wy1 = wy0, wx0, wy1, wx1
            env_xs, env_ys = env_ys, env_xs
        pitch = self._res_cell
        xs: Set[int] = {px, gx, wx0, wx1, *env_xs}
        ys: Set[int] = {py, gy, wy0, wy1, *env_ys}
        # x-CORRIDOR: uniform pitch fill (the S1 window is a few pitches wide -> a
        # bounded line count), origin-aligned so jog tracks are consistent net-to-net.
        _X_LINE_CAP = 24
        xspan = max(1, wx1 - wx0)
        xstep = max(pitch, xspan // _X_LINE_CAP)
        if xstep > pitch:
            local_lo = ((px - 6 * pitch) // pitch) * pitch
            local_hi = px + 6 * pitch
            gxl = local_lo
            while gxl <= local_hi:
                if wx0 <= gxl <= wx1:
                    xs.add(gxl)
                gxl += pitch
        gxi = (wx0 // xstep) * xstep
        while gxi <= wx1:
            if gxi >= wx0:
                xs.add(gxi)
            gxi += xstep
        # y-REACH: a COARSE fill capped to a bounded line count regardless of how
        # tall the drop is (a whole-cell drop must not become a whole-cell grid). The
        # drop runs mostly straight on an upper metal, so few turn rows suffice.
        _Y_LINE_CAP = 24
        reach = max(1, wy1 - wy0)
        ystep = max(pitch, reach // _Y_LINE_CAP)
        gyi = (wy0 // ystep) * ystep
        while gyi <= wy1:
            if gyi >= wy0:
                ys.add(gyi)
            gyi += ystep
        lines = (
            self._on_manufacturing_grid_lines(xs, wx0, wx1),
            self._on_manufacturing_grid_lines(ys, wy0, wy1),
        )
        return (lines[1], lines[0]) if rail_axis == "V" else lines

    def _straight_rail_drop(
        self, net: str, pin: Tuple[int, int, int], rgi: int, gx: int, gy: int,
        lg: Sequence[int],
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """PATTERN-FIRST closed-form rail drop: a straight drop of one pin
        onto its owned rail metal, with NO per-drop A* search.

        A rail drop whose column is CLEAR has a deterministic geometry -- a via
        STACK straight up the pin's OWN column (``gi`` -> the rail layer ``rgi``) +
        ONE vertical wire on the rail layer from the climbed column to the rail near
        edge ``gy`` -- so it needs none of ``_route_edge``'s dense candidate grid /
        expansion search (the measured 5717s gnd flood: 25 clean drops each building
        a ~10^6-state graph; and the vdd expansions=1 OPEN: the array-escape START
        sealed). The straight geometry is judged with the engine's OWN hard-legality
        (``_via_legal`` / ``_wire_legal`` -- foreign device-artwork keep-out + via
        cut spacing; own rail metal is exempt) and the SAME present/history price as
        A*; a congested straight candidate defers to the bounded search. NO new
        legality model, wall-clock choice, or per-circuit branch.

        A pin SEALED at its exact centre (the measured vdd (88492,80156): the dense
        pmos fingers box the pin so no via lands at offset 0) climbs through the
        pin-access ESCAPE via instead -- an OFFSET landing (<= a via pad half) that
        still COVERS the pin, so the drop rides a clear column a few hundred nm off
        the pin x. The rail is full-width, so the offset column still lands on it;
        the drop stays effectively straight (one bounded escape offset, no jog).

        When the pin-local column is sealed, the exact drawn terminal component
        may already carry the net to many legal via landings.  Those landings are
        a via-selector domain, not Cartesian A* axes: rank every real DRC-clean
        column to the declared rail and enter A* only when no straight candidate
        exists.  This is the same completeness boundary as ``_pin_access`` -- no
        coordinate is sampled or invented, and the original connected conductor
        is the authority for every remote landing.

        Returns ``(segs, vias)`` in the SAME shape ``_route_edge`` returns (the caller
        appends them + registers the cuts identically), or None when the straight
        geometry is blocked or congested -- the caller then falls back to the bounded
        local search. The actual rail rectangle determines the straight direction:
        a horizontal band takes a vertical wire and a vertical band a horizontal
        wire. Both use the same via domain, legality and objective below.
        """
        x, y, gi = pin
        allowed = set(lg)
        climb_layers = set(range(min(gi, rgi), max(gi, rgi) + 1))
        if gi not in allowed or rgi not in allowed or not climb_layers <= allowed:
            return None
        if rgi < gi:
            return None

        rail = tuple(getattr(self, "_active_rail_pattern_domain", ()))

        def rail_landing(cx: int, cy: int) -> Optional[Tuple[int, int]]:
            if rail:
                candidates = []
                for rail_index, (rail_gi, raw_rect) in enumerate(rail):
                    rx0, ry0, rx1, ry1 = _norm(raw_rect)
                    if rail_gi != rgi:
                        continue
                    horizontal = rx1 - rx0 >= ry1 - ry0
                    if horizontal:
                        if not rx0 <= cx <= rx1:
                            continue
                        landing_x, landing_y = cx, min(max(cy, ry0), ry1)
                    else:
                        if not ry0 <= cy <= ry1:
                            continue
                        landing_x, landing_y = min(max(cx, rx0), rx1), cy
                    candidates.append((
                        abs(cx - landing_x) + abs(cy - landing_y),
                        rx0,
                        ry0,
                        rx1,
                        ry1,
                        rail_index,
                        landing_x,
                        landing_y,
                    ))
                if not candidates:
                    return None
                return min(candidates)[-2:]
            # Direct unit tests and historical callers pass the already-resolved
            # projection only.  Preserve that interface without pretending an
            # offset column lies on an unknown rail span.
            return (cx, gy) if x == gx else None

        def straight_candidate(
            cx: int,
            cy: int,
            first_via: Optional[_ViaSite],
        ) -> Optional[Tuple[int, int, Tuple[_WireSeg, ...], Tuple[_ViaSite, ...]]]:
            landing = rail_landing(cx, cy)
            if landing is None:
                return None
            landing_x, landing_y = landing
            vias: List[_ViaSite] = []
            lo = gi
            if gi < rgi:
                if first_via is None or first_via.lo != gi:
                    return None
                vias.append(first_via)
                lo = gi + 1
            while lo < rgi:
                selected_via = self._preferred_via_site(net, cx, cy, lo)
                if selected_via is None:
                    return None
                vias.append(selected_via)
                lo += 1
            segs: List[_WireSeg] = []
            if (cx, cy) != landing:
                if not self._wire_legal(
                    rgi, cx, cy, landing_x, landing_y, net
                ):
                    return None
                segs.append(
                    self._wire_segment(
                        net, rgi, cx, cy, landing_x, landing_y
                    )
                )
            if (
                bool(getattr(self, "_active_require_width_carrier", False))
                and not any(
                    self._hw(net, seg.gi) > self._half_w[seg.gi]
                    for seg in segs
                )
            ):
                # A via-only shortcut can be electrically connected while drawing
                # no conductor that materializes a declared supra-min Width.  Such
                # a candidate is incomplete, not a successful rail drop: defer to
                # the ordinary access/A* path, which can route an exact widened
                # carrier.  The final Width honesty gate remains authoritative.
                return None
            congestion = self._pattern_congestion(net, segs, vias)
            objective = self._pattern_objective(
                net,
                segs,
                vias,
                rgi,
                congestion=congestion,
            )
            return objective, congestion, tuple(segs), tuple(vias)

        congested_candidates = []

        # First preserve the cheapest common case: the pin's own column (or its
        # bounded local escape) reaches the rail without enumerating the component.
        if gi == rgi:
            local = straight_candidate(x, y, None)
        else:
            local_via = self._preferred_via_site(net, x, y, gi)
            cx, cy = x, y
            if local_via is None:
                landing = self._escape_landing(net, x, y, gi)
                if landing is not None:
                    cx, cy, model_n = landing
                    local_via = _ViaSite(cx, cy, gi, model_n)
            local = (
                None
                if local_via is None
                else straight_candidate(cx, cy, local_via)
            )
        if local is not None:
            objective, congestion, segs, vias = local
            if congestion == 0:
                return list(segs), list(vias)
            congested_candidates.append((objective, segs, vias))

        # anaroute-style via selection over the COMPLETE exact component domain.
        # The selector stays linear in candidate count; it never crosses every
        # candidate x with every candidate y to manufacture a route graph.
        best_clean = None
        seen_vias = set()
        if gi < rgi:
            component_states = self._owned_component_escape_states(
                net, x, y, gi
            )

            # Order the complete, exact via domain by an admissible objective
            # lower bound.  This is branch-and-bound, not candidate sampling:
            # every candidate that can tie or improve the incumbent is still
            # checked.  Congestion and the not-yet-selected upper via models
            # have non-negative prices, so omitting them cannot overestimate
            # the final pattern objective.
            bounded_states = []
            for candidate_index, (state, via) in enumerate(component_states):
                if candidate_index % 128 == 0:
                    self._watchdog_check(net, "rail_pattern_via_selection")
                if (
                    via is None
                    or via.lo != gi
                    or state != (via.x, via.y, gi + 1)
                    or state[2] not in allowed
                ):
                    continue
                via_key = self._access_state_key((state, via))
                if via_key in seen_vias:
                    continue
                seen_vias.add(via_key)
                landing = rail_landing(via.x, via.y)
                if landing is None:
                    continue
                lower_bound = (
                    self._edge_objective(
                        net,
                        rgi,
                        via.x,
                        via.y,
                        landing[0],
                        landing[1],
                        rgi,
                    )
                    + _VIA_COST
                    + self._functional_single_via_penalty(via)
                    + max(0, rgi - gi - 1) * _VIA_COST
                )
                bounded_states.append((lower_bound, via_key, state, via))

            bounded_states.sort(key=lambda item: (item[0], item[1]))
            for candidate_index, (
                lower_bound,
                via_key,
                _state,
                via,
            ) in enumerate(bounded_states):
                if candidate_index % 128 == 0:
                    self._watchdog_check(net, "rail_pattern_via_selection")
                if best_clean is not None and lower_bound > best_clean[0]:
                    break
                candidate = straight_candidate(via.x, via.y, via)
                if candidate is None:
                    continue
                objective, congestion, segs, vias = candidate
                stable = (
                    objective,
                    via_key,
                    tuple((s.gi, s.ax, s.ay, s.bx, s.by) for s in segs),
                    tuple((v.x, v.y, v.lo, v.n) for v in vias),
                    segs,
                    vias,
                )
                if congestion == 0:
                    if best_clean is None or stable < best_clean:
                        best_clean = stable
                else:
                    congested_candidates.append((objective, segs, vias))
        if best_clean is not None:
            _score, _key, _seg_key, _via_key, segs, vias = best_clean
            return list(segs), list(vias)

        collector = getattr(self, "_active_congested_rail_patterns", None)
        if isinstance(collector, list) and congested_candidates:
            objective, segs, vias = min(
                congested_candidates,
                key=lambda item: (
                    item[0],
                    tuple((s.gi, s.ax, s.ay, s.bx, s.by) for s in item[1]),
                    tuple((v.x, v.y, v.lo, v.n) for v in item[2]),
                ),
            )
            collector.append((objective, segs, vias))
        return None

    def _route_net_to_rail(
        self, net: str, nodes: List[Tuple[int, int, int]], lg: List[int],
        rn: _RoutedNet, rail: Sequence[Tuple[int, Rect]],
        node_pins: Optional[Sequence[Optional[Pin]]] = None,
        endpoint_members: Optional[
            Sequence[Tuple[Tuple[Tuple[str, Any], ...], ...]]
        ] = None,
    ) -> Optional[_RoutedNet]:
        """Route minimum-width local taps when the wide rail is already real.

        ``Width + TrunkLayer`` scopes a supra-minimum width to the structural
        carrier.  A declared rail is that carrier; applying its width to every
        same-layer pin drop turns a compact, DRC-legal access channel into a
        false blockage.  Activate a local search-width override only when the
        received owned rail rectangle itself materializes the full declared
        width.  Segments created under the override freeze that exact width, so
        legality, occupancy, final deck snapshots, and emission stay identical.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        overrides = self._rail_local_half_width_overrides(net, lg, rail)
        with self._half_width_scope(overrides):
            return self._route_net_to_rail_impl(
                net,
                nodes,
                lg,
                rn,
                rail,
                node_pins=node_pins,
                endpoint_members=endpoint_members,
            )

    def _rail_local_half_width_overrides(
        self, net: str, lg: Sequence[int], rail: Sequence[Tuple[int, Rect]],
    ) -> Dict[Tuple[str, int], int]:
        """Use the existing actual-rail and allowed-layer width proof."""
        overrides: Dict[Tuple[str, int], int] = {}
        declared_widths = self._net_half_w.get(net)
        if declared_widths is not None:
            allowed = set(lg)
            rail_materializes_width = any(
                0 <= int(rail_gi) < len(declared_widths)
                and int(rail_gi) in allowed
                and declared_widths[int(rail_gi)] > self._half_w[int(rail_gi)]
                and min(
                    int(rail_rect[2]) - int(rail_rect[0]),
                    int(rail_rect[3]) - int(rail_rect[1]),
                ) >= 2 * int(declared_widths[int(rail_gi)])
                for rail_gi, rail_rect in rail
            )
            if rail_materializes_width:
                # An edge rail is supplied as owned geometry rather than as a
                # TrunkLayer constraint, so its plain Width table is uniform.
                # Once any received rail rectangle proves that it is the wide
                # carrier, every layer used by its local drops is access/riser
                # geometry and returns to the PDK minimum width.
                for gi, target_half in enumerate(declared_widths):
                    if gi not in allowed or target_half <= self._half_w[gi]:
                        continue
                    overrides[(str(net), int(gi))] = self._half_w[gi]

        return overrides

    @contextmanager
    def _half_width_scope(
        self, overrides: Mapping[Tuple[str, int], int], *,
        attribute: str = "_active_half_width_overrides",
    ):
        missing = object()
        previous = getattr(self, attribute, missing)
        active = {} if previous is missing else dict(previous)
        active.update(overrides)
        setattr(self, attribute, active)
        try:
            yield
        finally:
            if getattr(self, '_post_selection_observation', None) is None:
                if previous is missing:
                    delattr(self, attribute)
                else:
                    setattr(self, attribute, previous)
            else:
                import sys
                original = sys.exception()
                try:
                    if previous is missing:
                        delattr(self, attribute)
                    else:
                        setattr(self, attribute, previous)
                except BaseException as ending:
                    if original is None:
                        raise
                    from clay.layout.route.group_endpoint_observation import record_observation_unwind
                    record_observation_unwind(original, ending)

    def _rail_drop_access_order_proxy(
        self,
        net: str,
        x: int,
        y: int,
        gi: int,
        *,
        layer_access_ok: bool,
    ) -> Tuple[int, ...]:
        """Cheap deterministic fail-first rank for rail-drop planning.

        Planning order is a heuristic, not a legality oracle.  Running the full
        dynamic ``_pin_access`` universe for every drop here duplicates the real
        per-drop DRC pass after attempt cuts start changing.  Fixed owned-component
        cardinality/span is the correct stable proxy: a missing layer path sorts
        first as an immediate failure, then smaller/fewer terminal conductors sort
        before broad components.  The actual drop still recomputes exhaustive
        access and all via/DRC obligations.
        """
        if not layer_access_ok:
            return (0, 0, 0, 0)
        component = tuple(self._owned_terminal_component(net, x, y, gi))
        if not component:
            return (1, 0, 0, 0)
        x0 = min(atom.rect[0] for atom in component)
        y0 = min(atom.rect[1] for atom in component)
        x1 = max(atom.rect[2] for atom in component)
        y1 = max(atom.rect[3] for atom in component)
        total_area = sum(
            max(0, atom.rect[2] - atom.rect[0])
            * max(0, atom.rect[3] - atom.rect[1])
            for atom in component
        )
        return (
            1 + len(component),
            max(0, x1 - x0) * max(0, y1 - y0),
            int(total_area),
            max(max(0, x1 - x0), max(0, y1 - y0)),
        )

    def _admit_rail_drop(
        self, net: str, prior: _RoutedNet, found,
        source: Tuple[int, int, int], rail: Sequence[Tuple[int, Rect]],
    ) -> Optional[Tuple[List[_WireSeg], List[_ViaSite]]]:
        """Validate one rail candidate against all earlier drops, without commit.

        Access selection has already chosen every via model.  Keep those models
        and prior objects immutable; only the existing candidate-scoped metal
        repair may add segments.  The pin seed and actual declared rail are the
        endpoint obligation, including for fast paths without access transactions.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        self._watchdog_check(net, "rail_drop_admission")
        if prior.name != net or found is None or not rail:
            return None
        segs, vias = list(found[0]), list(found[1])
        original_vias = tuple(prior.vias) + tuple(vias)
        missing = object()
        saved = {
            name: getattr(self, name, missing)
            for name in ("_attempt_cuts", "_hard_rejections",
                         "_hard_rejection_samples", "_hard_rejection_generation")
        }
        self._attempt_cuts = {
            gi: list(rects) for gi, rects in self._attempt_cuts.items()
        }
        self._hard_rejections = dict(getattr(self, "_hard_rejections", {}))
        self._hard_rejection_samples = dict(getattr(self, "_hard_rejection_samples", {}))
        try:
            # Only new cuts are checked against the copied prior ledger.  Each
            # selected cut is then reserved for the other cuts in this candidate.
            for via in vias:
                self._watchdog_check(net, "rail_drop_admission")
                if not self._via_legal(via.x, via.y, via.lo, net, n=self._site_n(via)):
                    return None
                self._add_attempt_cut(via)
            combined_segs = list(prior.segs) + segs
            combined_vias = list(original_vias)
            resolved = self._resolve_pattern_via_models(
                net, combined_vias, mutable_indices=set(), segs=combined_segs,
            )
            if resolved is None:
                repair = self._materialize_candidate_via_wire_merges(
                    net, combined_vias, combined_segs,
                    prior_via_count=len(prior.vias), prior_seg_count=len(prior.segs),
                    immutable_via_indices=set(range(len(combined_vias))),
                )
                if repair is not None:
                    segs.extend(repair)
                    combined_segs = list(prior.segs) + segs
                    resolved = self._resolve_pattern_via_models(
                        net, combined_vias, mutable_indices=set(), segs=combined_segs,
                    )
                if resolved is None:
                    repaired = self._resolve_candidate_same_net_fills(
                        net, prior, segs, vias, mutable_indices=set(),
                    )
                    if repaired is None:
                        return None
                    segs, vias = repaired
                    combined_segs = list(prior.segs) + segs
                    resolved = list(prior.vias) + vias
            if len(resolved) != len(original_vias) or any(
                current is not original
                for current, original in zip(resolved, original_vias)
            ):
                return None
            if self._candidate_mim_metal_failure(net, segs, vias, prior=prior):
                return None
            # Source is this drop's canonical pin seed, never a stale selected
            # access from an earlier transaction.  Each goal lies inside one of
            # the actual rail rectangles.  All real rails are present in the
            # deck, so attachment through a previously admitted drop is valid.
            for gi, rect in rail:
                self._watchdog_check(net, "rail_drop_admission")
                x0, y0, x1, y1 = _norm(rect)
                goal = ((x0 + x1) // 2, (y0 + y1) // 2, gi)
                if self._candidate_deck_connects_probes(
                    net, combined_segs, resolved, (source, goal),
                    include_fixed=True, additional_fixed_metal=rail,
                ):
                    self._watchdog_check(net, "rail_drop_admission")
                    return segs, vias
            return None
        finally:
            for name, value in saved.items():
                if value is missing:
                    self.__dict__.pop(name, None)
                else:
                    setattr(self, name, value)

    def _route_net_to_rail_impl(
        self, net: str, nodes: List[Tuple[int, int, int]], lg: List[int],
        rn: _RoutedNet, rail: Sequence[Tuple[int, Rect]],
        node_pins: Optional[Sequence[Optional[Pin]]] = None,
        endpoint_members: Optional[
            Sequence[Tuple[Tuple[Tuple[str, Any], ...], ...]]
        ] = None,
    ) -> Optional[_RoutedNet]:
        """Connect every source pin of a RAIL-declared net by a SHORT LOCAL DROP
        onto the owned rail metal -- the generic materializer of a declared
        ``@trunk(edge=)`` intent.

        The rail (a full-cell-width owned M-k band, seeded into the obstacle model
        as OWNED metal) is the net's DISTRIBUTED SPINE. Each pin lands on it near
        its OWN x, so an N-pin supply net decomposes into N cheap, LOCAL pin->rail
        edges -- each A* searches only its own small window -- instead of ONE
        whole-cell MST toward a single hub pin (the measured supply-net bottleneck:
        gnd ripped + rerouted ~1000s/iter, vdd FAIL at ~826k A* expansions, which
        starved the whole negotiation). A node already ON the rail (the injected
        rail-access hub pin) needs no drop. If ANY pin cannot reach the rail the
        whole net attempt returns None (rip-up + retry under the negotiation, the
        SAME honest semantics as the MST path -- never a silent partial). No per-net
        / per-circuit branch (keyed only on a declared rail); deterministic; no
        wall-clock inside the A* loop."""

        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        pins = list(node_pins or ())
        if len(pins) != len(nodes):
            pins = [None] * len(nodes)
        member_groups = list(endpoint_members or ())
        if len(member_groups) != len(nodes):
            member_groups = [
                (self._failure_endpoint(pin, node),)
                for pin, node in zip(pins, nodes)
            ]

        def _attach_member_evidence(index: int) -> None:
            if not (0 <= index < len(member_groups)):
                return
            members = member_groups[index]
            if len(members) <= 1:
                return
            details = dict(self._route_failure_details)
            details["source_owned_component_members"] = members
            self._route_failure_details = tuple(sorted(details.items()))

        allowed = set(lg)
        legal_rail = tuple(item for item in rail if item[0] in allowed)
        if not legal_rail or any(gi not in allowed for (_x, _y, gi) in nodes):
            if nodes:
                source = nodes[0]
                source_pin = pins[0]
                self._begin_route_failure_context(
                    phase="rail_layer_access",
                    source_pin=source_pin,
                    source=source,
                    goal_pin=None,
                    goal=source,
                    window=self._bounds,
                    allowed_layers=lg,
                    trunk_gi=source[2],
                )
                _attach_member_evidence(0)
                self._edge_fail = (
                    "pin_access_no_layer",
                    self._failure_endpoint(source_pin, source),
                )
            return None

        def _on_rail(x: int, y: int, gi: int) -> bool:
            for (rgi, (rx0, ry0, rx1, ry1)) in legal_rail:
                if gi == rgi and rx0 <= x <= rx1 and ry0 <= y <= ry1:
                    return True
            return False

        def _group_on_rail(index: int) -> bool:
            """True when any exact member of a contracted endpoint is on rail."""
            if not (0 <= index < len(member_groups)):
                return False
            for member in member_groups[index]:
                facts = dict(member)
                xy_nm = facts.get("xy_nm")
                gi = facts.get("gi")
                if (
                    isinstance(xy_nm, tuple)
                    and len(xy_nm) == 2
                    and isinstance(gi, int)
                    and _on_rail(int(xy_nm[0]), int(xy_nm[1]), int(gi))
                ):
                    return True
            return False

        drops = [
            (index, x, y, gi)
            for index, (x, y, gi) in enumerate(nodes)
            if not _on_rail(x, y, gi) and not _group_on_rail(index)
        ]
        self._attempt_cuts = {}
        self._attempt_metal = {}
        self._attempt_idx = {}
        if not drops:
            rn.connected = True
            return rn

        # Prepare terminal-local access obligations against the empty attempt
        # ledger, then route the fewest-choice drop first.  This is a generic
        # fail-first ordering (pin identity + integer geometry are stable tie
        # breakers), not a power-net/circuit special case.
        plans = []
        for index, x, y, gi in drops:
            rgi, gx, gy = self._nearest_rail_point(x, y, legal_rail)
            rail_axis = self._rail_axis_at_point(gx, gy, rgi, legal_rail)
            climb_layers = set(range(min(gi, rgi), max(gi, rgi) + 1))
            window = self._rail_drop_window(x, y, gx, gy, rail_axis=rail_axis)
            layer_access_ok = climb_layers <= allowed
            access_proxy = self._rail_drop_access_order_proxy(
                net,
                x,
                y,
                gi,
                layer_access_ok=layer_access_ok,
            )
            pin = pins[index]
            plans.append(
                (
                    (
                        *access_proxy,
                        "" if pin is None else str(pin.device),
                        "" if pin is None else str(pin.terminal),
                        int(x),
                        int(y),
                        int(gi),
                        int(index),
                    ),
                    (
                        index,
                        x,
                        y,
                        gi,
                        rgi,
                        gx,
                        gy,
                        window,
                        layer_access_ok,
                        rail_axis,
                    ),
                )
            )
        ordered_plans = [plan for _key, plan in sorted(plans, key=lambda item: item[0])]

        def _reset_attempt() -> None:
            rn.segs = []
            rn.vias = []
            self._attempt_cuts = {}
            self._attempt_metal = {}
            self._attempt_idx = {}

        def _admit_drop(found, source):
            admitted = self._admit_rail_drop(net, rn, found, source, legal_rail)
            if admitted is None:
                self._edge_fail = ("rail_drop_admission", tuple(source))
            return admitted

        def _commit_drop(found, source) -> bool:
            admitted = _admit_drop(found, source)
            if admitted is None:
                return False
            segs, vias = admitted
            # Build every new resource before publishing any part of the drop.
            # An exception (including cancellation) leaves the prior aliases and
            # their contents intact.
            next_segs, next_vias = rn.segs + segs, rn.vias + vias
            next_cuts = {gi: list(rects) for gi, rects in self._attempt_cuts.items()}
            next_metal = {gi: list(rects) for gi, rects in self._attempt_metal.items()}
            for via in vias:
                ext = self._via_cut_extent(via.lo, self._site_n(via))
                next_cuts.setdefault(via.lo, []).append(
                    (via.x - ext, via.y - ext, via.x + ext, via.y + ext)
                )
            for seg in segs:
                next_metal.setdefault(seg.gi, []).append(
                    _seg_bbox(
                        seg.ax,
                        seg.ay,
                        seg.bx,
                        seg.by,
                        self._segment_half_width(net, seg),
                    )
                )
            for via in vias:
                bh2, ah2 = self._via_pad_half_for(via)
                next_metal.setdefault(via.lo, []).append(
                    (via.x - bh2, via.y - bh2, via.x + bh2, via.y + bh2)
                )
                next_metal.setdefault(via.lo + 1, []).append(
                    (via.x - ah2, via.y - ah2, via.x + ah2, via.y + ah2)
                )
            next_idx = {}
            for gi2, rects2 in next_metal.items():
                idx2 = _SpatialIndex()
                for rect2 in rects2:
                    idx2.add(net, rect2)
                next_idx[gi2] = idx2
            self._watchdog_check(net, "rail_drop_commit")
            rn.segs, rn.vias = next_segs, next_vias
            self._attempt_cuts = next_cuts
            self._attempt_metal = next_metal
            self._attempt_idx = next_idx
            return True

        attempt_cut_deferred_to_reorder = False

        def _route_drop(
            plan,
            *,
            defer_attempt_cut_to_reorder: bool = False,
        ) -> bool:
            with self._component_access.escape_scope():
                return _route_drop_in_component_escape_scope(
                    plan,
                    defer_attempt_cut_to_reorder=defer_attempt_cut_to_reorder,
                )

        def _route_drop_in_component_escape_scope(
            plan,
            *,
            defer_attempt_cut_to_reorder: bool = False,
        ) -> bool:
            nonlocal attempt_cut_deferred_to_reorder
            (
                index,
                x,
                y,
                gi,
                rgi,
                gx,
                gy,
                window,
                layer_access_ok,
                rail_axis,
            ) = plan
            self._begin_route_failure_context(
                phase="rail_drop",
                source_pin=pins[index],
                source=(x, y, gi),
                goal_pin=None,
                goal=(gx, gy, rgi),
                window=window,
                allowed_layers=lg,
                trunk_gi=rgi,
            )
            _attach_member_evidence(index)
            if not layer_access_ok:
                self._edge_fail = (
                    "pin_access_no_layer",
                    self._failure_endpoint(pins[index], (x, y, gi)),
                )
                return False
            # Access is solve-state dependent: previous drops add exact cut
            # reservations, and a refined replay can change the legal event cells.
            # Recompute BOTH endpoint access and the rail goal set for each search;
            # never reuse a pre-rejection snapshot.
            congested_pattern = None
            candidate_goal_state = None

            def _fresh_drop_access():
                starts = self._endpoint_access(
                    endpoint="source",
                    net=net,
                    node=(x, y, gi),
                    allowed_layers=lg,
                )
                goals = {
                    state: via
                    for state, via in self._rail_goal_set(
                        gx, gy, rgi, legal_rail
                    ).items()
                    if state[2] in allowed
                }
                if candidate_goal_state is not None:
                    goals.setdefault(candidate_goal_state, None)
                details = dict(self._route_failure_details)
                details["source_access_state_count"] = len(starts)
                details["goal_access_state_count"] = len(goals)
                self._route_failure_details = tuple(sorted(details.items()))
                return starts, goals

            # #1 PATTERN-FIRST straight drop: a CLEAN vertical column onto the rail
            # is a deterministic closed-form drop (via stack up the pin's OWN column
            # + one vertical wire), so it must NOT pay the dense per-drop A* search
            # -- that search is the measured 5717s gnd flood (25 clean drops each on
            # a ~10^6-state graph) and the vdd expansions=1 OPEN (array-escape start
            # sealed). Emitted directly when hard-legal AND congestion-free; a fixed
            # block or provisional/history price enters the bounded search.
            congested_patterns = []
            previous_congested_patterns = getattr(
                self, "_active_congested_rail_patterns", None
            )
            self._active_congested_rail_patterns = congested_patterns
            previous_rail_pattern_domain = getattr(
                self, "_active_rail_pattern_domain", ()
            )
            self._active_rail_pattern_domain = legal_rail
            previous_require_width_carrier = getattr(
                self, "_active_require_width_carrier", False
            )
            try:
                width_carrier_gis = {
                    candidate_gi
                    for candidate_gi in range(self._n_layers)
                    if self._hw(net, candidate_gi) > self._half_w[candidate_gi]
                }
                width_already_materialized = any(
                    seg.gi in width_carrier_gis for seg in rn.segs
                )
                if not width_already_materialized:
                    width_already_materialized = any(
                        rail_gi in width_carrier_gis
                        and min(
                            rail_rect[2] - rail_rect[0],
                            rail_rect[3] - rail_rect[1],
                        ) >= 2 * self._hw(net, rail_gi)
                        for rail_gi, rail_rect in legal_rail
                    )
                require_width_carrier = bool(
                    width_carrier_gis and not width_already_materialized
                )
                self._active_require_width_carrier = require_width_carrier
                found = self._straight_rail_drop(
                    net, (x, y, gi), rgi, gx, gy, lg
                )
            finally:
                self._active_congested_rail_patterns = (
                    previous_congested_patterns
                )
                self._active_rail_pattern_domain = (
                    previous_rail_pattern_domain
                )
                self._active_require_width_carrier = (
                    previous_require_width_carrier
                )
            if found is not None:
                found = _admit_drop(found, (x, y, gi))
            if congested_patterns:
                congested_pattern = min(
                    congested_patterns,
                    key=lambda item: (
                        item[0],
                        tuple(
                            (seg.gi, seg.ax, seg.ay, seg.bx, seg.by)
                            for seg in item[1]
                        ),
                        tuple(
                            (via.x, via.y, via.lo, via.n)
                            for via in item[2]
                        ),
                    ),
                )
                _score, candidate_segs, candidate_vias = congested_pattern
                if candidate_segs:
                    last_seg = candidate_segs[-1]
                    candidate_goal_state = (
                        last_seg.bx,
                        last_seg.by,
                        last_seg.gi,
                    )
                elif candidate_vias:
                    last_via = candidate_vias[-1]
                    candidate_goal_state = (
                        last_via.x,
                        last_via.y,
                        last_via.lo + 1,
                    )
                else:
                    candidate_goal_state = (x, y, gi)
                if not _on_rail(*candidate_goal_state):
                    candidate_goal_state = None
            if found is None:
                # A straight pattern rejected solely by cuts from earlier drops
                # in this same net is an ordering conflict, not yet a residual
                # path obligation.  Let the existing adjacent-drop replay consume
                # that exact evidence before building any A* access domains.  The
                # replay disables this shortcut, so residual A* remains the
                # complete fallback if the alternate order has the same conflict.
                straight_rejections = tuple(
                    getattr(self, "_hard_rejections", {})
                )
                if (
                    defer_attempt_cut_to_reorder
                    and not congested_patterns
                    and straight_rejections
                    and all(
                        category == "attempt_cut"
                        for category, _owner in straight_rejections
                    )
                ):
                    attempt_cut_deferred_to_reorder = True
                    self._edge_fail = (
                        "access_exhausted",
                        self._hard_rejection_evidence(),
                    )
                    return False
                starts, goals = _fresh_drop_access()
                if not starts:
                    self._edge_fail = (
                        "pin_access_no_state",
                        self._failure_endpoint(pins[index], (x, y, gi)),
                        self._hard_rejection_evidence(),
                    )
                    return False
                if not goals:
                    self._edge_fail = (
                        "rail_goal_no_state",
                        self._failure_endpoint(None, (gx, gy, rgi)),
                        self._hard_rejection_evidence(),
                    )
                    return False
                pattern_incumbent = _PatternSearchIncumbent()

                def _pattern_for_domain(domain_starts, domain_goals):
                    incumbent = _PatternSearchIncumbent()
                    if congested_pattern is None:
                        return incumbent
                    score, candidate_segs, raw_candidate_vias = (
                        congested_pattern
                    )
                    candidate_vias = list(raw_candidate_vias)
                    selected_start = None
                    if candidate_vias:
                        first_via = candidate_vias[0]
                        selected_start = next((
                            item
                            for item in domain_starts
                            if (
                                item[0]
                                == (
                                    first_via.x,
                                    first_via.y,
                                    first_via.lo + 1,
                                )
                                and item[1] == first_via
                            )
                        ), None)
                        if selected_start is not None:
                            candidate_vias[0] = selected_start[1]
                    if selected_start is None:
                        selected_start = next((
                            item
                            for item in domain_starts
                            if item == ((x, y, gi), None)
                        ), None)

                    goal_state = candidate_goal_state
                    selected_goal = (
                        (goal_state, domain_goals[goal_state])
                        if (
                            goal_state in domain_goals
                            and domain_goals[goal_state] is None
                        )
                        else None
                    )
                    if selected_start is not None and selected_goal is not None:
                        incumbent.consider(
                            score=score,
                            candidate_index=0,
                            start=selected_start,
                            goal=selected_goal,
                            segs=candidate_segs,
                            vias=candidate_vias,
                        )
                    return incumbent

                def _attempt_drop_edge(selected_starts, selected_goals):
                    previous_incumbent = getattr(
                        self, "_active_pattern_search_incumbent", None
                    )
                    self._active_pattern_search_incumbent = pattern_incumbent
                    try:
                        return self._route_edge(
                            selected_starts,
                            selected_goals,
                            net,
                            xs,
                            ys,
                            lg,
                            rgi,
                        )
                    finally:
                        self._active_pattern_search_incumbent = (
                            previous_incumbent
                        )
                # LOCAL per-drop window: the pin + its rail projection grown by a
                # bounded margin (so a blocked column detours locally), NEVER the
                # whole cell.
                # SPARSE candidate lines: bounded + finger-density-INDEPENDENT (the
                # drop escapes up and runs over the M1 fingers, so their edges are
                # not turn points). ``_candidate_lines`` here ingested every finger
                # edge in the tall window -> the 5717s gnd flood (a ~10^6-state
                # graph per drop).
                if rail_axis == "V":
                    xs, ys = self._rail_candidate_lines(
                        (x, y), (gx, gy), window, rail_axis=rail_axis,
                    )
                else:
                    xs, ys = self._rail_candidate_lines((x, y), (gx, gy), window)
                # bias the drop to climb TO the rail layer (trunk == rail gi):
                # routing BELOW the rail costs more so the pin rises onto the spine
                # and STOPS there. NOT ``self._trunk_gi`` -- a whole-cell supply
                # net's span-derived trunk can sit ABOVE the rail and would bias the
                # short drop to overshoot past its own rail. The rail IS the trunk.
                wx0, wy0, wx1, wy1 = _norm(window)
                local_starts = tuple(
                    item
                    for item in starts
                    if (
                        wx0 <= item[0][0] <= wx1
                        and wy0 <= item[0][1] <= wy1
                    )
                )
                if not local_starts:
                    local_starts = tuple(starts)
                nearest_starts = self._nearest_access_frontier(
                    starts, goals
                )
                complete_source_count = len(starts)

                def _search_access_domain(
                    domain_starts,
                    domain_goals,
                    scope: str,
                ):
                    nonlocal pattern_incumbent
                    details = dict(self._route_failure_details)
                    details["source_access_state_count"] = (
                        complete_source_count
                    )
                    details["local_source_access_state_count"] = len(
                        local_starts
                    )
                    details["nearest_source_access_state_count"] = len(
                        nearest_starts
                    )
                    details["source_access_scope"] = str(scope)
                    self._route_failure_details = tuple(
                        sorted(details.items())
                    )
                    pattern_incumbent = _pattern_for_domain(
                        domain_starts, domain_goals
                    )
                    missing_routed_net = object()
                    previous_routed_net = getattr(
                        self, "_active_access_routed_net", missing_routed_net
                    )
                    self._active_access_routed_net = rn
                    try:
                        candidate = self._route_access_transactions(
                            domain_starts,
                            domain_goals,
                            net,
                            rgi,
                            _attempt_drop_edge,
                            prevalidated=True,
                        )
                    finally:
                        if previous_routed_net is missing_routed_net:
                            del self._active_access_routed_net
                        else:
                            self._active_access_routed_net = previous_routed_net
                    if (
                        candidate is not None
                        and require_width_carrier
                        and not any(
                            seg.gi in width_carrier_gis
                            for seg in candidate[0]
                        )
                    ):
                        # Do not let a zero-wire access transaction silently
                        # consume the last chance to realize this net's hard
                        # Width obligation.
                        return None
                    return (None if candidate is None else
                            _admit_drop(candidate, (x, y, gi)))

                # anaroute component-access pattern, freshly implemented in Clay:
                # first expose every source on the exact nearest-target lower-bound
                # frontier, then the window-local slice when it adds authority, and
                # finally the complete component domain.  No fixed candidate cap is
                # involved and no legal access is discarded.  A deterministic edge
                # budget can fail one earlier stage without suppressing the complete
                # fallback; only the global solve budget ends the obligation.
                staged_domains = []
                seen_access_domains = set()

                def _append_access_stage(scope, domain_starts):
                    key = tuple(
                        self._access_state_key(item)
                        for item in domain_starts
                    )
                    if not key or key in seen_access_domains:
                        return
                    seen_access_domains.add(key)
                    staged_domains.append((str(scope), tuple(domain_starts)))

                # Keep the historical unified search for a sparse/one-dimensional
                # source side.  Nearest-target staging is useful when the source
                # component is a dense two-dimensional access cloud: only then do
                # its independent x/y coordinates recreate a much larger Cartesian
                # graph.  This structural threshold has no device/PDK count cap.
                dense_access_cloud = self._access_cloud_is_cartesian_dense(
                    starts
                )
                if dense_access_cloud and len(nearest_starts) < len(starts):
                    _append_access_stage(
                        "nearest_frontier", nearest_starts
                    )
                if dense_access_cloud and len(local_starts) < len(starts):
                    _append_access_stage("window_local", local_starts)
                _append_access_stage(
                    "complete" if not staged_domains else "complete_fallback",
                    starts,
                )

                found = None
                failure_kind = None
                far_pattern_attempted = False
                for stage_index, (scope, domain_starts) in enumerate(
                    staged_domains
                ):
                    if stage_index:
                        starts, goals = _fresh_drop_access()
                        complete_source_count = len(starts)
                        local_starts = tuple(
                            item
                            for item in starts
                            if (
                                wx0 <= item[0][0] <= wx1
                                and wy0 <= item[0][1] <= wy1
                            )
                        ) or tuple(starts)
                        nearest_starts = self._nearest_access_frontier(
                            starts, goals
                        )
                        if scope == "nearest_frontier":
                            domain_starts = nearest_starts
                        elif scope == "window_local":
                            domain_starts = local_starts
                        else:
                            domain_starts = tuple(starts)
                    if (
                        (
                            stage_index > 0
                            or scope in {"complete", "complete_fallback"}
                        )
                        and not far_pattern_attempted
                        and self._has_live_rail_pattern_blocker()
                    ):
                        far_pattern_attempted = True
                        tree_pattern = self._rail_attempt_tree_pattern(
                            net,
                            (x, y, gi),
                            starts,
                            rn.segs,
                            lg,
                        )
                        if tree_pattern is not None and _commit_drop(
                            tree_pattern, (x, y, gi)
                        ):
                            return True
                        far_pattern = self._far_rail_portal_pattern(
                            net,
                            (x, y, gi),
                            starts,
                            rgi,
                            gx,
                            gy,
                            legal_rail,
                            window,
                            lg,
                        )
                        if far_pattern is not None and _commit_drop(
                            far_pattern, (x, y, gi)
                        ):
                            return True
                    found = _search_access_domain(
                        domain_starts,
                        goals,
                        scope,
                    )
                    failure_kind = (
                        None
                        if (
                            found is not None
                            or getattr(self, "_edge_fail", None) is None
                        )
                        else self._edge_fail[0]
                    )
                    if found is not None or failure_kind == "solve_budget":
                        break
                if (
                    found is None
                    and failure_kind not in {"solve_budget", "budget"}
                ):
                    refinement_stages = (
                        self._causal_blocker_frontier_stages(
                            net, xs, ys, window, lg
                        )
                    )
                    for refinement_index, (
                        refined_xs,
                        refined_ys,
                    ) in enumerate(refinement_stages, start=1):
                        if not self._causal_frontier_graph_within_budget(
                            refined_xs,
                            refined_ys,
                            lg,
                            nominal_work=True,
                        ):
                            failure_kind = "budget"
                            break
                        xs, ys = refined_xs, refined_ys
                        starts, goals = _fresh_drop_access()
                        complete_source_count = len(starts)
                        found = _search_access_domain(
                            starts,
                            goals,
                            (
                                "complete_refined_"
                                f"{refinement_index}_of_"
                                f"{len(refinement_stages)}"
                            ),
                        )
                        if found is not None:
                            break
                        failure_kind = (
                            None
                            if getattr(self, "_edge_fail", None) is None
                            else self._edge_fail[0]
                        )
                        if failure_kind in {"solve_budget", "budget"}:
                            break
            if (
                found is not None
                and require_width_carrier
                and not any(
                    seg.gi in width_carrier_gis for seg in found[0]
                )
            ):
                found = None
            if found is None:
                if self.debug:
                    print(f"  [gridless] {net}: drop {(x, y, gi)}->rail FAIL "
                          f"{getattr(self, '_edge_fail', None)}", flush=True)
                return False
            return _commit_drop(found, (x, y, gi))

        def _run_drop_order(
            order,
            *,
            defer_attempt_cut_to_reorder: bool = False,
        ):
            _reset_attempt()
            for position, plan in enumerate(order):
                if not _route_drop(
                    plan,
                    defer_attempt_cut_to_reorder=defer_attempt_cut_to_reorder,
                ):
                    return False, position
            return True, None

        active_order = ordered_plans
        routed_all, failed_position = _run_drop_order(
            active_order,
            defer_attempt_cut_to_reorder=True,
        )
        original_order_deferred = attempt_cut_deferred_to_reorder
        # One evidence-gated local backtrack: an exact attempt-cut blocker proves
        # that the current drop conflicts with some earlier drop in this attempt.
        # The deliberately bounded heuristic swaps only the adjacent predecessor
        # once and replays from an empty ledger; it does not claim blocker ownership.
        # Fixed artwork, no-state, and solve-budget failures never trigger
        # speculative extra search.
        if (
            not routed_all
            and failed_position is not None
            and failed_position > 0
            and (
                getattr(self, "_edge_fail", None) is None
                or self._edge_fail[0] not in {"solve_budget", "budget"}
            )
            and any(
                category == "attempt_cut"
                for category, _owner in self._hard_rejections
            )
        ):
            retry_order = list(ordered_plans)
            retry_order[failed_position - 1], retry_order[failed_position] = (
                retry_order[failed_position],
                retry_order[failed_position - 1],
            )
            active_order = retry_order
            routed_all, failed_position = _run_drop_order(active_order)
        if not routed_all and original_order_deferred:
            # The swap is a bounded ordering heuristic.  If it fails, replay the
            # original order once with the shortcut disabled so its residual A*
            # detour remains part of the complete search authority.
            active_order = ordered_plans
            routed_all, failed_position = _run_drop_order(active_order)
        if not routed_all:
            primary_edge_fail = getattr(self, "_edge_fail", None)
            primary_failure_details = tuple(
                getattr(self, "_route_failure_details", ())
            )
            merged_rejections = dict(getattr(self, "_hard_rejections", {}))
            merged_samples = dict(
                getattr(self, "_hard_rejection_samples", {})
            )

            def _probe_committed_blocker(category: str, owner: str) -> bool:
                return (
                    category in _COMMITTED_BLOCKER_REJECTION_CATEGORIES
                    and owner not in {net, "<unowned>"}
                )

            def _merge_probe_samples(probe_samples) -> None:
                for cause, (count, candidate_extent) in sorted(
                    probe_samples.items()
                ):
                    category, owner = cause[:2]
                    if not _probe_committed_blocker(category, owner):
                        continue
                    previous = merged_samples.get(cause)
                    if previous is None:
                        merged_samples[cause] = (
                            int(count), tuple(candidate_extent)
                        )
                        continue
                    previous_count, previous_extent = previous
                    merged_samples[cause] = (
                        int(previous_count) + int(count),
                        (
                            min(previous_extent[0], candidate_extent[0]),
                            min(previous_extent[1], candidate_extent[1]),
                            max(previous_extent[2], candidate_extent[2]),
                            max(previous_extent[3], candidate_extent[3]),
                        ),
                    )

            failure_kind = (
                None if primary_edge_fail is None else primary_edge_fail[0]
            )
            has_committed_blocker_sample = any(
                cause[0] in _COMMITTED_BLOCKER_REJECTION_CATEGORIES
                for cause in merged_samples
            )
            if (
                failed_position is not None
                and failure_kind not in {"solve_budget", "budget"}
                and has_committed_blocker_sample
            ):
                # Evidence-only post-sweep: later drops are independent diagnostic
                # obligations, never alternative route authority.  A successful
                # probe contributes no blocker evidence; every probe's geometry is
                # discarded before the next one and again before returning.
                for plan in active_order[failed_position + 1:]:
                    _reset_attempt()
                    probe_failed = not _route_drop(plan)
                    probe_edge_fail = getattr(self, "_edge_fail", None)
                    probe_failure_kind = (
                        None if probe_edge_fail is None else probe_edge_fail[0]
                    )
                    if probe_failure_kind in {"solve_budget", "budget"}:
                        break
                    if not probe_failed:
                        continue
                    for (category, owner), count in sorted(
                        getattr(self, "_hard_rejections", {}).items()
                    ):
                        if not _probe_committed_blocker(category, owner):
                            continue
                        key = (category, owner)
                        merged_rejections[key] = (
                            int(merged_rejections.get(key, 0)) + int(count)
                        )
                    _merge_probe_samples(
                        getattr(self, "_hard_rejection_samples", {})
                    )

            _reset_attempt()
            self._edge_fail = primary_edge_fail
            self._route_failure_details = primary_failure_details
            self._hard_rejections = dict(sorted(merged_rejections.items()))
            self._hard_rejection_samples = dict(sorted(merged_samples.items()))
            return None

        # DEDUP identical escape vias a pin shares across drops.
        seen_v: Set[Tuple[int, int, int]] = set()
        uniq: List[_ViaSite] = []
        for v in rn.vias:
            k = (v.x, v.y, v.lo)
            if k in seen_v:
                continue
            seen_v.add(k)
            uniq.append(v)
        rn.vias = uniq
        self._legalize_same_net(net, rn)
        rn.connected = True
        return rn

    # ------------------------------------------------------- same-net merge
    def _rebuild_committed_state(
        self,
        committed: Mapping[str, _RoutedNet],
    ) -> None:
        """Rebuild routing book and spatial indices from one committed truth."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        book = _Book(self._n_layers)
        for name, baseline in sorted(
            getattr(self, "_group_route_landing_baselines", {}).items()
        ):
            metal_by_gi, cuts = self._net_metal(baseline)
            book.set_persistent(
                name,
                metal_by_gi,
                cuts,
                self._net_via_pads(baseline),
            )
        for name in sorted(committed):
            rn = committed[name]
            metal_by_gi, cuts = self._net_metal(rn)
            book.add(name, metal_by_gi, cuts, self._net_via_pads(rn))
        self._book = book
        self._rebuild_metal_index()

    def _length_report_for_committed(
        self,
        constraint: Length,
        committed: Mapping[str, _RoutedNet],
    ) -> dict:
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        routes: Dict[str, NetRoute] = {}
        for net in (str(constraint.net_a), str(constraint.net_b)):
            routed = committed.get(net) or _RoutedNet(name=net, connected=False)
            emitted = self._emit_net(routed)
            routes[net] = NetRoute(
                net=net,
                wires=list(emitted.wires),
                connected=bool(routed.connected),
                length_um=float(emitted.length_um),
            )
        return self._materialized_length_report(constraint, routes)

    @staticmethod
    def _has_via_bounded_drop(short_route: _RoutedNet, segment: _WireSeg, target_gi: int) -> bool:
        """True when both endpoints of ``segment`` already have a cut touching ``target_gi``."""
        p0 = (int(segment.ax), int(segment.ay))
        p1 = (int(segment.bx), int(segment.by))
        if p0 == p1:
            return False
        via_sites = {
            (int(v.x), int(v.y))
            for v in short_route.vias
            if int(v.lo) == int(target_gi)
        }
        return p0 in via_sites and p1 in via_sites

    def _coupling_axes_with_uniform_grid(
        self,
        base_axes: Sequence[int],
        orig_axis: int,
        lo_bound: int,
        hi_bound: int,
        half: int,
    ) -> Tuple[int, ...]:
        """Augment sparse Hanan lines with bounded uniform samples inside ``seg_bounds``."""
        grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 5)))
        merged = set(int(v) for v in base_axes) | {int(orig_axis)}
        span = int(hi_bound) - int(lo_bound) - 2 * int(half)
        if span > 0:
            step = max(grid * 40, ((span // 60) // grid) * grid)
            cur = ((int(lo_bound) + int(half) + grid - 1) // grid) * grid
            end = int(hi_bound) - int(half)
            while cur <= end:
                merged.add(cur)
                cur += step
        return tuple(sorted(merged))

    def _check_tuning_wire_legal(
        self,
        replacement: Sequence[_WireSeg],
        short_net: str,
        iface,
    ) -> bool:
        """Check ``self._wire_legal`` for every segment in ``replacement`` under ``iface``."""
        prev_local_closure = getattr(self, "_active_group_route_local_closure", False)
        prev_local_iface = getattr(self, "_active_group_route_local_interface", None)
        try:
            if iface is not None:
                self._active_group_route_local_closure = True
                self._active_group_route_local_interface = iface
            return not any(
                not self._wire_legal(
                    item.gi,
                    item.ax,
                    item.ay,
                    item.bx,
                    item.by,
                    short_net,
                )
                for item in replacement
            )
        finally:
            self._active_group_route_local_closure = prev_local_closure
            self._active_group_route_local_interface = prev_local_iface

    def _has_tuning_own_conflict(
        self,
        short_route: _RoutedNet,
        short_net: str,
        segment: _WireSeg,
        replacement: Sequence[_WireSeg],
        target_gi: int,
        half: int,
        raw_lo: int,
        raw_hi: int,
    ) -> bool:
        """True if ``replacement`` violates spacing against own-net wires or vias."""
        min_sp_check = int(self._space[int(target_gi)])
        legs_check = replacement[1:-1] if len(replacement) >= 5 else replacement
        is_drop_layer = int(target_gi) != int(segment.gi)
        raw_lo_eff, raw_hi_eff = (0, 0) if is_drop_layer else (raw_lo, raw_hi)
        other_segs = [
            (
                s_other,
                _seg_bbox(
                    s_other.ax,
                    s_other.ay,
                    s_other.bx,
                    s_other.by,
                    self._segment_half_width(short_net, s_other),
                ),
            )
            for s_idx, s_other in enumerate(short_route.segs)
            if not (raw_lo_eff <= s_idx < raw_hi_eff)
            and int(s_other.gi) == int(target_gi)
        ]
        anchors = {(int(segment.ax), int(segment.ay)), (int(segment.bx), int(segment.by))}
        non_anchor_vias: List[Rect] = []
        for v in short_route.vias:
            if int(target_gi) not in (int(v.lo), int(v.lo) + 1):
                continue
            if (int(v.x), int(v.y)) in anchors:
                continue
            bh, ah = self._via_pad_half_for(v)
            vh = int(bh if int(v.lo) == int(target_gi) else ah)
            non_anchor_vias.append((int(v.x) - vh, int(v.y) - vh, int(v.x) + vh, int(v.y) + vh))

        for leg_idx, leg in enumerate(legs_check):
            lbox = _seg_bbox(leg.ax, leg.ay, leg.bx, leg.by, half)
            for s_other, sbox in other_segs:
                gx = max(0, max(lbox[0] - sbox[2], sbox[0] - lbox[2]))
                gy = max(0, max(lbox[1] - sbox[3], sbox[1] - lbox[3]))
                if gx >= min_sp_check or gy >= min_sp_check:
                    continue
                if gx == 0 and gy == 0 and len(replacement) == 3 and leg_idx in (0, 2):
                    anchor_pt = (segment.ax, segment.ay) if leg_idx == 0 else (segment.bx, segment.by)
                    ox = min(lbox[2], sbox[2]) - max(lbox[0], sbox[0])
                    oy = min(lbox[3], sbox[3]) - max(lbox[1], sbox[1])
                    if (
                        anchor_pt in ((s_other.ax, s_other.ay), (s_other.bx, s_other.by))
                        and max(ox, oy)
                        <= 2 * max(half, self._segment_half_width(short_net, s_other)) + 5
                    ):
                        continue
                return True
            for vbox in non_anchor_vias:
                gx = max(0, max(lbox[0] - vbox[2], vbox[0] - lbox[2]))
                gy = max(0, max(lbox[1] - vbox[3], vbox[1] - lbox[3]))
                if gx < min_sp_check and gy < min_sp_check:
                    return True
        return False

    def _materialize_length_pair(
        self,
        constraint: Length,
        committed: Dict[str, _RoutedNet],
    ) -> bool:
        """Insert the shortest legal single-U detour that closes one matched pair.

        For explicit hard length constraints or isolated two-net fixtures, balances
        the materialized length metric.  For soft ``matched_cc_balance`` pairs in
        multi-net layouts, balances 2.5D shielded parasitic coupling capacitance and
        allows a zero-extra-via ``gi - 1`` detour between existing endpoint vias when
        same-layer meandering alone cannot close the coupling gap.
        """

        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        net_a = str(constraint.net_a)
        net_b = str(constraint.net_b)
        coupling_mode = bool(
            getattr(constraint, "hardness", HARD) == SOFT
            and has_third_party_non_ground_conductors(self, committed, net_a, net_b)
        )
        report = self._length_report_for_committed(constraint, committed)
        route_a = committed.get(net_a)
        route_b = committed.get(net_b)
        if (
            route_a is None
            or route_b is None
            or not route_a.connected
            or not route_b.connected
        ):
            return False

        self._rebuild_committed_state(committed)
        coupling_ctx: Optional[CouplingContext] = None
        base_cap_imbal = 0.0
        trigger_tol = float(constraint.tol_rel)
        if coupling_mode:
            coupling_ctx = CouplingContext(self, committed, net_a, net_b)
            base_cap_imbal, cap_a, cap_b = coupling_ctx.evaluate()
            trigger_tol = max(float(constraint.tol_rel), 0.075)
            if base_cap_imbal <= trigger_tol:
                return True
            short_net, long_net = (net_a, net_b) if cap_a <= cap_b else (net_b, net_a)
        else:
            if report["satisfied"]:
                return True
            if report["length_a_nm"] <= report["length_b_nm"]:
                short_net, long_net = net_a, net_b
            else:
                short_net, long_net = net_b, net_a

        short_route = committed.get(short_net)
        long_route = committed.get(long_net)
        if short_route is None or long_route is None or not short_route.segs:
            return False

        bounds = tuple(_norm(self._bounds))
        ifaces = tuple(
            getattr(self, "_group_route_global_interfaces_by_net", {}).get(
                short_net, ()
            )
        )
        pinned_points: Set[Tuple[int, int, Optional[int]]] = {
            (int(xy[0]), int(xy[1]), int(xy[2]) if len(xy) >= 3 else None)
            for xy in getattr(self, "_net_nodes", {}).get(short_net, ())
        }
        via_points: Set[Tuple[int, int, int]] = set()
        for v in short_route.vias:
            vx, vy, vlo = int(v.x), int(v.y), int(v.lo)
            via_points.add((vx, vy, vlo))
            via_points.add((vx, vy, vlo + 1))
        portal_points: Set[Tuple[int, int, Optional[int]]] = {
            (int(p[0]), int(p[1]), int(p[2]) if len(p) >= 3 else None)
            for _env, p in ifaces
        }

        def _is_protected(x: int, y: int, gi: int) -> bool:
            return (
                (x, y, gi) in via_points
                or (x, y, gi) in pinned_points
                or (x, y, None) in pinned_points
                or (x, y, gi) in portal_points
                or (x, y, None) in portal_points
            )

        def _coalesce_segs(segs):
            # Track junction degree and protected pins/vias per layer so an unrelated
            # pin or lower-layer via/junction underneath cannot split an upper wire.
            deg: Dict[Tuple[int, int, int], int] = {}
            for s in segs:
                sgi = int(s.gi)
                p0 = (sgi, int(s.ax), int(s.ay))
                p1 = (sgi, int(s.bx), int(s.by))
                deg[p0] = deg.get(p0, 0) + 1
                deg[p1] = deg.get(p1, 0) + 1
            out_segs = []
            raw_spans = []
            for raw_idx, s in enumerate(segs):
                if not out_segs:
                    out_segs.append(s)
                    raw_spans.append([raw_idx, raw_idx + 1])
                    continue
                prev = out_segs[-1]
                sgi = int(s.gi)
                p_shared = (int(prev.bx), int(prev.by))
                if (
                    prev.gi == s.gi
                    and int(getattr(prev, "half_width_nm", 0))
                    == int(getattr(s, "half_width_nm", 0))
                    and p_shared == (int(s.ax), int(s.ay))
                    and not _is_protected(p_shared[0], p_shared[1], sgi)
                    and deg.get((sgi, p_shared[0], p_shared[1]), 0) == 2
                    and (
                        (prev.ay == prev.by == s.ay == s.by
                         and (prev.bx - prev.ax) * (s.bx - s.ax) > 0)
                        or (prev.ax == prev.bx == s.ax == s.bx
                            and (prev.by - prev.ay) * (s.by - s.ay) > 0)
                    )
                ):
                    out_segs[-1] = _WireSeg(
                        prev.gi,
                        prev.ax,
                        prev.ay,
                        s.bx,
                        s.by,
                        int(getattr(prev, "half_width_nm", 0)),
                    )
                    raw_spans[-1][1] = raw_idx + 1
                else:
                    out_segs.append(s)
                    raw_spans.append([raw_idx, raw_idx + 1])
            return tuple(out_segs), tuple((lo, hi) for lo, hi in raw_spans)

        base_segs, raw_spans = _coalesce_segs(tuple(short_route.segs))

        def _seg_group_iface(seg: _WireSeg):
            ax_i, ay_i, bx_i, by_i = int(seg.ax), int(seg.ay), int(seg.bx), int(seg.by)
            for env, portal in ifaces:
                x0, y0, x1, y1 = int(env[0]), int(env[1]), int(env[2]), int(env[3])
                if (
                    x0 <= ax_i <= x1
                    and y0 <= ay_i <= y1
                    and x0 <= bx_i <= x1
                    and y0 <= by_i <= y1
                ):
                    return (env, portal)
            return None

        candidates = []
        diagnostic_counts = {
            "candidate_count": 0,
            "candidate_tried": 0,
            "hard_legality_rejected": 0,
            "deck_merge_rejected": 0,
            "deck_open_rejected": 0,
            "deck_spacing_rejected": 0,
            "native_short_rejected": 0,
        }

        def publish_diagnostics(phase: str) -> None:
            self._route_failure_details = tuple(sorted({
                "phase": str(phase),
                "length_net_a": net_a,
                "length_net_b": net_b,
                "length_a_nm": int(report["length_a_nm"]),
                "length_b_nm": int(report["length_b_nm"]),
                **diagnostic_counts,
            }.items()))

        def _segment_endpoint_max_half(segment: _WireSeg, gi: int, half: int) -> int:
            max_h = int(half)
            endpoints = {(int(segment.ax), int(segment.ay)), (int(segment.bx), int(segment.by))}
            for v in short_route.vias:
                if (int(v.x), int(v.y)) in endpoints and int(gi) in (int(v.lo), int(v.lo) + 1):
                    bh, ah = self._via_pad_half_for(v)
                    max_h = max(max_h, int(bh if int(v.lo) == int(gi) else ah))
            for s_other in short_route.segs:
                if int(s_other.gi) != int(gi):
                    continue
                if (int(s_other.ax), int(s_other.ay)) in endpoints or (int(s_other.bx), int(s_other.by)) in endpoints:
                    max_h = max(max_h, int(self._segment_half_width(short_net, s_other)))
            return max_h

        def tuning_replacement(
            segment: _WireSeg,
            tuning_axis: int,
            half: int = 0,
            iface=None,
            target_gi: Optional[int] = None,
        ):
            gi = int(segment.gi) if target_gi is None else int(target_gi)
            explicit_half = (
                0
                if gi != int(segment.gi)
                else int(getattr(segment, "half_width_nm", 0))
            )
            min_sp = int(self._space[gi])
            grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 5)))
            end_half = _segment_endpoint_max_half(segment, gi, half)
            min_gap = ((2 * half + min_sp + grid - 1) // grid) * grid
            end_clearance = ((half + end_half + min_sp + grid - 1) // grid) * grid
            ax_i, ay_i, bx_i, by_i = (
                int(segment.ax),
                int(segment.ay),
                int(segment.bx),
                int(segment.by),
            )
            span = abs(bx_i - ax_i) + abs(by_i - ay_i)
            orig_axis_i = ay_i if ay_i == by_i else ax_i
            if span < min_gap or abs(int(tuning_axis) - orig_axis_i) < end_clearance:
                return None
            if iface is not None and gi == int(segment.gi):
                # Clear connected same-layer wire/via pads by >= min_sp so neither
                # own_conflict nor _space_residual_scan rejects the approach stubs.
                full_stub = end_clearance
                floor_stub = ((half + 50 + grid - 1) // grid) * grid
                if span >= 2 * full_stub + min_gap:
                    min_stub = full_stub
                elif span >= 2 * floor_stub + min_gap:
                    min_stub = (((span - min_gap) // 2) // grid) * grid
                else:
                    return None
                if ay_i == by_i:
                    step_sign = 1 if bx_i > ax_i else -1
                    sx0 = ax_i + step_sign * min_stub
                    sx1 = bx_i - step_sign * min_stub
                    coords = (
                        (ax_i, ay_i, sx0, ay_i),
                        (sx0, ay_i, sx0, tuning_axis),
                        (sx0, tuning_axis, sx1, tuning_axis),
                        (sx1, tuning_axis, sx1, by_i),
                        (sx1, by_i, bx_i, by_i),
                    )
                else:
                    step_sign = 1 if by_i > ay_i else -1
                    sy0 = ay_i + step_sign * min_stub
                    sy1 = by_i - step_sign * min_stub
                    coords = (
                        (ax_i, ay_i, ax_i, sy0),
                        (ax_i, sy0, tuning_axis, sy0),
                        (tuning_axis, sy0, tuning_axis, sy1),
                        (tuning_axis, sy1, bx_i, sy1),
                        (bx_i, sy1, bx_i, by_i),
                    )
                return tuple(
                    _WireSeg(gi, ax0, ay0, bx0, by0, explicit_half)
                    for ax0, ay0, bx0, by0 in coords
                )
            if segment.ay == segment.by:
                coords = (
                    (segment.ax, segment.ay, segment.ax, tuning_axis),
                    (segment.ax, tuning_axis, segment.bx, tuning_axis),
                    (segment.bx, tuning_axis, segment.bx, segment.by),
                )
            else:
                coords = (
                    (segment.ax, segment.ay, tuning_axis, segment.ay),
                    (tuning_axis, segment.ay, tuning_axis, segment.by),
                    (tuning_axis, segment.by, segment.bx, segment.by),
                )
            return tuple(
                _WireSeg(gi, ax0, ay0, bx0, by0, explicit_half)
                for ax0, ay0, bx0, by0 in coords
            )

        previous_attempt_idx = self._attempt_idx
        self._attempt_idx = {}
        try:
            publish_diagnostics("length_tuning_domain")
            base_xs, base_ys = self._candidate_lines(
                (), short_net, bounds
            )
            for segment_index, segment in enumerate(base_segs):
                self._watchdog_check(short_net, "length_tuning_domain")
                if segment.ax != segment.bx and segment.ay != segment.by:
                    continue
                iface = _seg_group_iface(segment)
                seg_bounds = (
                    (int(iface[0][0]), int(iface[0][1]), int(iface[0][2]), int(iface[0][3]))
                    if iface is not None
                    else bounds
                )
                target_gis = [int(segment.gi)]
                if (
                    coupling_mode
                    and int(segment.gi) >= 1
                    and self._has_via_bounded_drop(short_route, segment, int(segment.gi) - 1)
                ):
                    target_gis.append(int(segment.gi) - 1)

                original_axis = (
                    segment.ay if segment.ay == segment.by else segment.ax
                )
                raw_lo, raw_hi = raw_spans[segment_index]
                for target_gi in target_gis:
                    half = (
                        self._hw(short_net, target_gi)
                        if target_gi != int(segment.gi)
                        else self._segment_half_width(short_net, segment)
                    )
                    if segment.ay == segment.by:
                        raw_axes = base_ys
                        lo_b, hi_b = seg_bounds[1], seg_bounds[3]
                    else:
                        raw_axes = base_xs
                        lo_b, hi_b = seg_bounds[0], seg_bounds[2]
                    orthogonal_axes = (
                        self._coupling_axes_with_uniform_grid(
                            raw_axes, int(original_axis), lo_b, hi_b, int(half)
                        )
                        if coupling_mode
                        else tuple(sorted(set(raw_axes) | {int(original_axis)}))
                    )
                    for axis_index, tuning_axis in enumerate(orthogonal_axes):
                        if axis_index % 128 == 0:
                            self._watchdog_check(
                                short_net, "length_tuning_domain"
                            )
                        if tuning_axis == original_axis:
                            continue
                        replacement = tuning_replacement(
                            segment, tuning_axis, half, iface, target_gi=target_gi
                        )
                        if replacement is None:
                            continue
                        legs_to_check = (
                            replacement[1:-1]
                            if iface is not None and len(replacement) >= 5
                            else replacement
                        )
                        if any(
                            not (
                                seg_bounds[0] <= bbox[0]
                                and seg_bounds[1] <= bbox[1]
                                and bbox[2] <= seg_bounds[2]
                                and bbox[3] <= seg_bounds[3]
                            )
                            for bbox in (
                                _norm(_seg_bbox(
                                    item.ax,
                                    item.ay,
                                    item.bx,
                                    item.by,
                                    half,
                                ))
                                for item in legs_to_check
                            )
                        ):
                            continue
                        if coupling_mode:
                            assert coupling_ctx is not None
                            if self._has_tuning_own_conflict(
                                short_route, short_net, segment,
                                replacement, target_gi, half, raw_lo, raw_hi,
                            ):
                                continue
                            if not self._check_tuning_wire_legal(
                                replacement, short_net, iface
                            ):
                                continue
                            cand_route = _RoutedNet(
                                name=short_route.name,
                                segs=[
                                    *short_route.segs[:raw_lo],
                                    *replacement,
                                    *short_route.segs[raw_hi:],
                                ],
                                vias=list(short_route.vias),
                                connected=True,
                            )
                            touched_gis = tuple(sorted({int(segment.gi), int(target_gi)}))
                            cand_imbal, _ca, _cb = coupling_ctx.evaluate(
                                override_net=short_net,
                                override_route=cand_route,
                                touched_gis=touched_gis,
                            )
                            if cand_imbal >= base_cap_imbal:
                                continue
                            same_layer = int(target_gi) == int(segment.gi)
                            goal_tol = float(constraint.tol_rel)
                            tier = (
                                0
                                if (same_layer and cand_imbal <= goal_tol * 0.7)
                                else (1 if cand_imbal <= goal_tol else 2)
                            )
                            added_len = 2 * abs(int(tuning_axis) - int(original_axis))
                            candidates.append((
                                (
                                    tier,
                                    round(cand_imbal, 3),
                                    0 if same_layer else 1,
                                    added_len,
                                    segment_index,
                                    target_gi,
                                    int(tuning_axis),
                                ),
                                segment_index,
                                target_gi,
                                int(tuning_axis),
                            ))
                        else:
                            candidate_short_length = (
                                int(report[
                                    "length_a_nm"
                                    if short_net == net_a
                                    else "length_b_nm"
                                ])
                                + 2 * abs(int(tuning_axis) - int(original_axis))
                                + (len(replacement) - (raw_hi - raw_lo)) * 2 * int(half)
                            )
                            candidate_long_length = int(report[
                                "length_b_nm" if short_net == net_a else "length_a_nm"
                            ])
                            denominator = max(
                                candidate_short_length, candidate_long_length
                            )
                            relative_imbalance = (
                                abs(candidate_short_length - candidate_long_length)
                                / denominator
                            )
                            if relative_imbalance > float(constraint.tol_rel):
                                continue
                            candidates.append((
                                (
                                    abs(candidate_short_length - candidate_long_length),
                                    denominator,
                                    segment_index,
                                    int(tuning_axis),
                                ),
                                segment_index,
                                target_gi,
                                int(tuning_axis),
                            ))
        finally:
            self._attempt_idx = previous_attempt_idx
        diagnostic_counts["candidate_count"] = len(candidates)
        publish_diagnostics("length_tuning")

        for _sort_key, segment_index, target_gi, tuning_axis in sorted(
            candidates,
            key=lambda item: item[0],
        ):
            diagnostic_counts["candidate_tried"] += 1
            publish_diagnostics("length_tuning")
            self._watchdog_check(short_net, "length_tuning")
            segment = base_segs[segment_index]
            half = (
                self._hw(short_net, target_gi)
                if target_gi != int(segment.gi)
                else self._segment_half_width(short_net, segment)
            )
            iface = _seg_group_iface(segment)
            replacement = tuning_replacement(
                segment, tuning_axis, half, iface, target_gi=target_gi
            )
            if replacement is None:
                continue
            if not self._check_tuning_wire_legal(replacement, short_net, iface):
                diagnostic_counts["hard_legality_rejected"] += 1
                continue
            raw_lo, raw_hi = raw_spans[segment_index]
            if self._has_tuning_own_conflict(
                short_route, short_net, segment,
                replacement, target_gi, half, raw_lo, raw_hi,
            ):
                diagnostic_counts["hard_legality_rejected"] += 1
                continue
            candidate = _RoutedNet(
                name=short_route.name,
                segs=[
                    *short_route.segs[:raw_lo],
                    *replacement,
                    *short_route.segs[raw_hi:],
                ],
                vias=list(short_route.vias),
                connected=True,
            )
            candidate_world = dict(committed)
            candidate_world[short_net] = candidate
            self._rebuild_committed_state(candidate_world)
            deck_snapshot = self._deck_negotiation_snapshot(candidate_world)
            if not deck_snapshot.report.cross_net_isolated:
                diagnostic_counts["deck_merge_rejected"] += 1
                continue
            if _deck_open_route_nets(
                deck_snapshot.report, tuple(sorted(candidate_world))
            ):
                diagnostic_counts["deck_open_rejected"] += 1
                continue
            _advisory, deck_spacing = self._space_residual_scan(candidate_world)
            if deck_spacing:
                diagnostic_counts["deck_spacing_rejected"] += 1
                continue
            if self._detect_shorts(candidate_world):
                diagnostic_counts["native_short_rejected"] += 1
                continue
            committed[short_net] = candidate
            if iface is not None and len(getattr(self, "_net_nodes", {}).get(short_net, ())) <= 1:
                if not hasattr(self, "_tuned_1pin_baseline_nets") or self._tuned_1pin_baseline_nets is None:
                    self._tuned_1pin_baseline_nets = set()
                self._tuned_1pin_baseline_nets.add(short_net)
            self._rebuild_committed_state(committed)
            return True

        self._rebuild_committed_state(committed)
        return False

    def _materialize_length_constraints(
        self,
        committed: Dict[str, _RoutedNet],
        constraints: Sequence[Length],
    ) -> Tuple[dict, ...]:
        """Reach a deterministic fixed point, then report every final pair."""

        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        ordered = tuple(sorted(
            constraints,
            key=lambda item: (str(item.net_a), str(item.net_b)),
        ))
        for _pass in range(max(1, len(ordered))):
            changed = False
            for constraint in ordered:
                net_a, net_b = str(constraint.net_a), str(constraint.net_b)
                before = self._length_report_for_committed(
                    constraint, committed
                )
                coupling_mode = bool(
                    getattr(constraint, "hardness", HARD) == SOFT
                    and has_third_party_non_ground_conductors(
                        self, committed, net_a, net_b
                    )
                )
                if not coupling_mode and before["satisfied"]:
                    continue
                prev_state = committed.get(net_a), committed.get(net_b)
                if self._materialize_length_pair(constraint, committed):
                    after = self._length_report_for_committed(
                        constraint, committed
                    )
                    after_state = committed.get(net_a), committed.get(net_b)
                    changed = changed or after != before or after_state != prev_state
            if not changed:
                break
        return tuple(
            self._length_report_for_committed(constraint, committed)
            for constraint in ordered
        )

