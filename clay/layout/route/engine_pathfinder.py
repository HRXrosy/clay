"""PathFinderEngine -- negotiated-congestion (PathFinder) rip-up/reroute router.

This is ANOTHER strategy behind the :class:`clay.layout.route.engine.RouteEngine`
contract (see that module's docstring), built ON TOP of the inherited maze
machinery in :class:`clay.layout.route.engine_maze.MazeRouteEngine`. It keeps the
exact same grid, obstacle model, A*, geometry emit, and short oracle; it changes
ONLY how cross-net contention is resolved.

The problem it fixes
====================
The base maze router routes nets one-by-one GREEDILY: each routed net HARD-BLOCKS
later nets' cells (its committed halo). On a congested channel a later net can
fail to connect even though a feasible GLOBAL routing exists -- a net that routes
fine ALONE starves because earlier nets took its only channels. That is an
ARTEFACT of the route order, not a real infeasibility.

The method (classic PathFinder / Nair negotiated congestion)
============================================================
Let nets SHARE cells at a COST rather than hard-blocking each other, then
rip-up-and-reroute every net under a rising congestion penalty until no cell is
shared (no shorts):

  * HARD blockage stays hard: device ``obstacles`` (with this net's width-aware
    halo, minus this net's OWN owned terminal pads) are never enterable -- exactly
    the base ``_route_net`` net-aware obstacle model.
  * SOFT congestion: a cell's entry cost when net B routes is
        (1 + history[cell] * hist_fac) * (1 + occupancy_excluding_B[cell] * pres_fac)
    where ``occupancy[cell]`` counts how many OTHER nets currently draw metal on
    that cell. This is fed through the inherited ``self._astar(cell_cost=...)`` hook
    -- other nets' wires are PENALISED, never hard-blocked, so rip-up can negotiate.
  * Iterate: each round RIPS UP all nets and reroutes every net (the base's
    deterministic difficulty order) under the current cost, then rebuilds
    ``occupancy``. Overused cells (occupancy >= 2 on drawn metal) get
    ``history[cell] += hist_fac`` and the present-congestion factor ``pres_fac``
    is multiplied up. When NO cell is overused the routing is short-free -> done.

Honesty (cf. the project's hard rule: never spin a broken result)
=================================================================
If it does NOT converge within ``max_iters`` it returns the BEST round seen, with
``success=False`` and the REMAINING shorts populated (or a net left
``connected=False`` if its pin is genuinely unreachable even at high penalty). It
NEVER fakes convergence and NEVER drops a net silently.

Constraint coverage is INHERITED unchanged: LayerSet / Width / Spacing /
Direction are satisfied (Direction with the base's EXACT parsing, fed to the
shared ``_astar`` ``dir_pref`` soft cost -- including the base's run-extent
comb contract (engine_maze ``_DIR_RUN_FREE``: anti-command runs beyond the jog
allowance escalate, seeded from the net's existing same-layer extent) -- AT
THE NEGOTIATION'S SCALE: in round k
the effective weight is ``weight x pres_fac_mult^k``, the same factor the
present-congestion cost rises by, so the commanded-axis preference is never
diluted by escalation -- round 0 is exactly the declared static weight, and
weight=0 / no Direction is byte-identical); Mirror is satisfied through the
SHARED base machinery (route net_a, adopt its reflection for net_b when the
reflection is hard-legal AND currently unoccupied, else soft-guide net_b's
search toward the reflection; exact mirror or honest unsatisfied + a
quantified per-pair asymmetry report in ``RoutedCell.mirror_reports`` -- see
the base module docstring); Shield / Length / TrackOrder / Access / Via are
recorded unsatisfied (so ``success`` is False) -- the same honest P1
reporting as the base.

ZERO per-circuit logic: ``route`` is a pure, deterministic function of
``(pins_by_net, obstacles, constraints, pdk)`` -- no circuit names, no net-name
special cases, no wall-clock / randomness.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

from clay.layout.backend import LayerPolygon, Polygon

from clay.layout.route.constraints import (
    Direction,
    LayerSet,
    Mirror,
    RouteConstraint,
    Spacing,
    TrunkLayer,
    Width,
)
from clay.layout.route.engine import (
    NetRoute,
    Obstacle,
    Pin,
    RoutedCell,
    require_resource_plan_support,
)
from clay.layout.route.engine_maze import (
    _CHECKPOINT_INTERVAL,
    _MIRROR_HARD_PREFER_PENALTY,
    MazeRouteEngine,
    _Tech,
    _active_checkpoint_cb,
    _build_native_grid,
    _via_cut_size_nm,
    _via_name_for,
)
from clay.layout.route.gridless_types import _via_cut_space_nm
from clay.layout.route.via_model import subtract_rects, via_landing_feasible

_NM_PER_UM = 1000.0

Cell = Tuple[int, int, int]  # (ix, iy, gi)

# --- deliberate multi-layer (trunk) assignment ---------------------------------
# A real router routes a long net's TRUNK on a higher metal (shorter, less
# congested), dropping to M1 only to access pins -- which is WHY a real layout uses
# its whole metal stack instead of collapsing every net onto M1+M2. We reproduce
# that: each net gets a target trunk graph-layer by its pin-bbox SPAN (in grid
# pitches); ``cell_cost`` then makes layers BELOW the trunk progressively costlier
# for that net, so its long run climbs (paying one via at each end) while a short
# local net (trunk 0) stays low. Tunable; swept empirically against real DRC.
_SPAN_PER_LEVEL = 90      # grid pitches of pin-bbox span that promote the trunk one metal
_TRUNK_BELOW_PENALTY = 0.3  # per-cell extra cost per metal-level a net routes BELOW its trunk


class PathFinderEngine(MazeRouteEngine):
    """A :class:`RouteEngine` strategy: negotiated-congestion (PathFinder)
    rip-up/reroute on top of the inherited maze machinery. Same RouteEngine
    contract; resolves the greedy-ordering congestion the base engine cannot.

    Parameters
    ----------
    max_iters : int
        Max rip-up/reroute rounds. If congestion is not resolved by then the best
        round is returned with ``success=False`` and the remaining shorts (HONEST
        non-convergence; never faked).
    pres_fac0 : float
        Initial present-congestion factor (how much a currently-shared cell costs
        on top of the base 1). Small at first so nets find natural channels.
    pres_fac_mult : float
        Per-round multiplier that raises the present-congestion factor, forcing
        nets off contested cells as rounds progress.
    hist_fac : float
        History increment added to a cell each round it is overused, AND the weight
        of accumulated history in the cost. Persistent congestion (a chronically
        contested cell) thus gets steadily more expensive across rounds.
    margin_tracks, max_expansions :
        Forwarded to :class:`MazeRouteEngine` (grid margin, per-search budget).
    """

    # Device M1 ownership is electrical-island provenance, never a per-rectangle
    # first-pin guess.  PathFinder consumes those owned obstacles in both search
    # and emit clipping, so the strict projection is an unconditional safety
    # capability just as it is for Gridless.  In particular, touching pieces of
    # one passive terminal must inherit the terminal net together; otherwise the
    # emitter can clip the real landing against a falsely-unowned sibling piece.
    strict_device_m1_provenance = True

    # The exact emitter is an observer, not a route producer.  Commit the same
    # PDK-derived terminal/contact closure as Gridless at the route-cell seam,
    # before short checks and the immutable route manifest are captured.
    materializes_terminal_interface_m1 = True

    # Consume the certified placement floorplan envelope (bounds grid, pin snap,
    # via landings, relocation, and widening strictly inside the declared box).
    consumes_routing_envelope = True

    # Consume physical upper-layer artwork (cap plates, upper device artwork,
    # via-cut keepouts) and certified edge rail placement plans. Leave
    # whole-footprint synthetic cap-column guards off so parked MIM terminals
    # stay accessible on the PathFinder grid while genuinely over-cell caps
    # still emit column guards + cap_plate_via via _cap_is_over_cell.
    consumes_artwork_conductors = True
    consumes_rail_placement_plan = True
    consumes_cap_column_guards = False

    def __init__(
        self,
        *,
        max_iters: int = 24,
        pres_fac0: float = 0.5,
        pres_fac_mult: float = 1.8,
        hist_fac: float = 1.0,
        widen: bool = False,
        trunk_layers: bool = False,
        over_active_penalty: float = 0.0,
        over_active_min_level: int = 3,
        over_active_rects_um: Optional[Sequence[Tuple[float, float, float, float]]] = None,
        margin_tracks: int = 8,
        max_expansions: int = 2_000_000,
    ) -> None:
        super().__init__(margin_tracks=margin_tracks, max_expansions=max_expansions)
        # at least one round (route every net once) -- a 0/negative budget would
        # otherwise emit nothing; clamp so the result is always a real routing.
        self.max_iters = max(1, int(max_iters))
        self.pres_fac0 = float(pres_fac0)
        self.pres_fac_mult = float(pres_fac_mult)
        self.hist_fac = float(hist_fac)
        # widen=True: ROUTE THIN (min-width centrelines -> clean, all nets connected
        # by negotiation), then WIDEN each net to its Width target where the channel
        # has room (clearance-driven; DRC-clean by construction). This gives wide
        # reference-matching trunks AND keeps min-space, which routing-at-trunk-width
        # on the fine grid cannot (adjacent-track trunks overlap). Width then means
        # the widen TARGET, not the routing width.
        self.widen = bool(widen)
        # trunk_layers=True: assign each net a TRUNK metal by pin-bbox span and bias its
        # long run to climb there. It spreads metal across M1..M_top (the layer-
        # distribution goal). LVS HISTORY, kept because each stage taught a contract:
        #   * v0.4.13: climb/via-stack/widen interaction made CROSS-NET SHORTS (external_pdk
        #     3 -> 17 unmatched). Cured by the 2026-06-10/11 emit contracts
        #     (foreign-clip min-space inflation, width-exact emit, sub-min residual
        #     filter + retry-at-min-width).
        #   * 2026-06-11..12: ONE residual orphan island ($10 = MM7 drain riser) --
        #     which turned out to be a FALSE open: the synthesized LVS deck hardcoded
        #     its connectivity ladder at metal3, so the trunk's geometrically complete
        #     M4 column (verified by direct GDS probe) was invisible to extraction.
        #     Fixed in the CHECKER (klayout_lvs_from_pdk full-stack ladder, bounded at
        #     the routable stack); the router geometry needed no change.
        #   * Current measured state (2026-06-12): external_pdk trunk LVS unmatched ==
        #     default-route unmatched EXACTLY (the remaining account is the
        #     mode-independent external_pdk cap/res extraction gap -- unrecognized MIM +
        #     cap M4/M5 plate keep-outs absent from the obstacle model); gf180 trunk
        #     is LVS-matched. See test_trunk_layers_lvs_parity_tsmc_gf180.
        # DRC-clean was NEVER evidence of correctness here (the partial deck does not
        # trace connectivity). DEFAULT OFF as policy: layers are demand-driven
        # resources, not a KPI. Default routing is unaffected.
        self.trunk_layers = bool(trunk_layers)
        # over_active_penalty > 0: a SOFT per-cell cost added when an UPPER metal
        # (graph level >= over_active_min_level's 0-based gi) routes OVER device
        # active diffusion. This implements the upper-metal-not-over-active
        # obligation as a routing PREFERENCE: the router prefers PLANAR channels
        # (off-active cells) for its long upper-metal runs, falling back to
        # over-active routing only when no planar channel exists (it is SOFT, never
        # a hard keep-out -- connectivity is never traded). The active region is
        # supplied as um rects (the device OD footprints, Clay's own geometry);
        # M1/M2 (local interconnect, which must cross the cells it connects) pay
        # nothing because over_active_min_level defaults to M3. Default 0.0 adds
        # nothing -> byte-identical route. The penalty is derived from the
        # obligation (a planarity preference), NOT a per-net / per-circuit value.
        self.over_active_penalty = float(over_active_penalty)
        self.over_active_min_level = int(over_active_min_level)
        self.over_active_rects_um = list(over_active_rects_um or [])

    # ------------------------------------------------------------------ contract
    def route(
        self,
        pins_by_net: Dict[str, List[Pin]],
        obstacles: Sequence[Obstacle],
        constraints: Sequence[RouteConstraint],
        pdk,
        *,
        preroutes: Optional[Dict[str, list]] = None,
        resource_plan: Optional[object] = None,
        routing_envelope_nm: Optional[object] = None,
    ) -> RoutedCell:
        require_resource_plan_support(self, resource_plan)
        chk = _active_checkpoint_cb()
        if chk is not None:
            chk()
        envelope = None
        envelope_bbox: Optional[Tuple[int, int, int, int]] = None
        if routing_envelope_nm is not None:
            from clay.layout.route.floorplan_containment import RoutingEnvelope

            envelope = RoutingEnvelope.coerce(routing_envelope_nm)
            envelope_bbox = envelope.bbox_nm
        tech = _Tech(pdk)
        out = RoutedCell()
        if envelope is not None:
            out.routing_envelope_nm = envelope.bbox_nm

        # --- preroute (contract point 6; the INHERITED maze helpers) -----------
        # Validate (ValueError on an out-of-graph layer) + fold the rects into the
        # obstacle sequence as OWNED hard keep-outs: foreign nets hold their
        # width-aware halo (hard -- a preroute is EXISTING metal, never something
        # negotiation may share), the owning net may land on it. preroutes=None
        # keeps everything byte-identical.
        pre_plan = self._preroute_plan(tech, preroutes)
        pre_wires_by_net: Dict[str, list] = {
            n: list(v) for n, v in (preroutes or {}).items() if v
        }
        if pre_plan:
            obstacles = list(obstacles) + self._preroute_obstacles(tech, pre_plan)

        # --- index constraints by net + collect honest-unsatisfied ones --------
        # (IDENTICAL parsing to MazeRouteEngine.route: LayerSet -> per-net allowed
        # graph layers; Width -> per-net width; Spacing -> recorded clearances;
        # Direction -> the per-layer dir_pref soft cost for the shared _astar;
        # Mirror/Shield/Length/TrackOrder/Access/Via -> recorded unsatisfied.)
        layer_set_by_net: Dict[str, List[int]] = {}
        layer_set_constraint_by_net: Dict[str, LayerSet] = {}
        width_by_net: Dict[str, float] = {}
        spacing_pairs: List[Spacing] = []
        mirror_list: List[Mirror] = []
        # graph layer -> (preferred axis 'H'|'V', weight): the Direction soft
        # cost. LAST Direction per layer wins (explicit overrides the
        # conventional table) -- the base engine's exact semantics.
        direction_pref: Dict[int, Tuple[str, float]] = {}
        unsatisfied: List[RouteConstraint] = []

        for c in constraints:
            if isinstance(c, LayerSet):
                gis = []
                for lay in c.layers:
                    gi = tech.resolve_layer_to_gi(lay)
                    if gi is not None:
                        gis.append(gi)
                if not gis:
                    # restricted to layers none of which are routable -> honest.
                    unsatisfied.append(c)
                else:
                    layer_set_by_net[c.net] = sorted(set(gis))
                    layer_set_constraint_by_net[c.net] = c
            elif isinstance(c, Width):
                width_by_net[c.net] = float(c.width_um)
            elif isinstance(c, Spacing):
                spacing_pairs.append(c)
            elif isinstance(c, Direction):
                axis = str(c.axis).upper()
                if axis not in ("H", "V") or float(c.weight) < 0.0:
                    # nonsense axis / negative weight: cannot honour -> honest.
                    unsatisfied.append(c)
                elif float(c.weight) > 0.0:
                    gi = tech._pdk_to_gi.get(int(c.layer_n))
                    if gi is not None:
                        direction_pref[gi] = (axis, float(c.weight))
                    # else: metal outside the routing stack -> nothing to bias,
                    # vacuously satisfied (soft preference on an absent layer).
                # weight == 0: an explicitly DISABLED preference -> inert by
                # contract (byte-identical route), satisfied trivially.
            elif isinstance(c, Mirror):
                mirror_list.append(c)
            elif isinstance(c, TrunkLayer):
                # This strategy has no trunk-layer cost.  A SOFT advisory is an
                # inert preference; a caller-forced HARD override is unsupported.
                if getattr(c, "hardness", "soft") == "hard":
                    unsatisfied.append(c)
            else:
                # Shield / Length / TrackOrder / Access / Via: not handled
                # by THIS strategy -> recorded, never silently dropped.
                unsatisfied.append(c)

        # valid Mirror pairs keyed by the MIRRORED net (the SHARED base
        # validation); invalid ones are recorded unsatisfied and perturb nothing.
        mirror_by_b = self._index_mirrors(mirror_list, pins_by_net, unsatisfied)

        # --- nothing to route? (identical to the base) -------------------------
        # (a preroute-only call still emits its verbatim metal + runs the oracle)
        if not pins_by_net and not pre_plan:
            out.unsatisfied = unsatisfied
            out.success = not unsatisfied
            out.mirror_reports = []
            out.negotiation_rounds = 0
            out.best_overused_cells = ()
            out.best_pad_contended_cells = ()
            out.convergence_failure_reasons = (
                ()
                if out.success
                else (f"unsatisfied_constraints={[repr(c) for c in unsatisfied]}",)
            )
            return out

        # --- build the grid from the pin + obstacle hull (inherited) -----------
        # (obstacles already include the preroute rects -> the hull covers them)
        grid = self._make_grid(
            tech, pins_by_net, obstacles, routing_envelope_nm=envelope_bbox
        )
        pre_cells: Dict[str, Set[Cell]] = {
            n: self._preroute_seed_cells(grid, rects)
            for n, rects in pre_plan.items()
        }

        # OVER-ACTIVE SOFT COST (upper-metal-not-over-active obligation): rasterise
        # the device active-diffusion rects (Clay's own OD geometry, um) into the
        # (ix,iy) cell set they cover, and the graph-layer (gi) floor at/above which
        # the over-active penalty applies. A cell on an upper metal lying over this
        # set pays ``self.over_active_penalty`` in ``cell_cost`` so the search
        # PREFERS planar (off-active) channels; it is SOFT (never a keep-out), so
        # connectivity is never traded. Empty / penalty 0 -> no cost (byte-stable).
        over_active_cells: Set[Tuple[int, int]] = set()
        over_active_gi_floor = tech.n_layers  # no level penalised by default
        if self.over_active_penalty > 0.0 and self.over_active_rects_um:
            for (x0, y0, x1, y1) in self.over_active_rects_um:
                self._raster_rect(grid, over_active_cells, x0, y0, x1, y1, 0)
            gi_floor = tech._pdk_to_gi.get(int(self.over_active_min_level))
            if gi_floor is not None:
                over_active_gi_floor = int(gi_floor)
            else:
                over_active_cells = set()  # upper level not in the routing stack

        # --- parse obstacles into per-layer rects (inherited model) ------------
        # Keep raw rects; rasterise PER NET with that net's width-aware halo below
        # (so a wide rail reserves its real half-width vs artwork). owner_net is
        # carried so a net may land on its OWN terminal pad.
        via_lo_by_key: Dict[Tuple[int, int], int] = {}
        for gl in range(max(0, tech.n_layers - 1)):
            vk = tech.via_key_for(gl)
            if vk is not None:
                via_lo_by_key[tuple(vk)] = gl
        obs_by_gi: Dict[int, List[Tuple[float, float, float, float, Optional[str]]]] = {
            gi: [] for gi in range(tech.n_layers)
        }
        via_obs_by_gi: Dict[
            int,
            List[Tuple[float, float, float, float, Optional[str], Optional[str]]],
        ] = {gl: [] for gl in range(max(0, tech.n_layers - 1))}
        for ob in obstacles:
            gi = tech.resolve_layer_to_gi(ob.layer)
            if gi is not None:
                obs_by_gi[gi].append((ob.x0, ob.y0, ob.x1, ob.y1, ob.owner_net))
                continue
            if isinstance(ob.layer, (tuple, list)) and len(ob.layer) == 2:
                cut_gl = via_lo_by_key.get((int(ob.layer[0]), int(ob.layer[1])))
                if cut_gl is not None:
                    via_obs_by_gi[cut_gl].append((
                        ob.x0, ob.y0, ob.x1, ob.y1, ob.owner_net,
                        getattr(ob, "kind", None),
                    ))
        # riser-pad clearance (inherited contract): out-of-graph pins' seed-layer
        # ladder pads are wider than the wire width -> owned hard obstacles, so
        # negotiation can never park a foreign wire on the adjacent track.
        for (gi, x0, y0, x1, y1, owner) in self._riser_pad_obstacles(
                grid, tech, pins_by_net, layer_set_by_net):
            obs_by_gi[gi].append((x0, y0, x1, y1, owner))

        # --- per-net static facts (width, allowed layers, start/target cells) --
        usable = list(range(tech.n_layers))
        route_pins_by_net: Dict[str, List[Pin]] = {
            n: list(plist) for n, plist in pins_by_net.items()
        }
        mim_clear_by_gi, mim_forbidden_cut_by_gi = self._mim_bottom_components(
            tech, pdk, obstacles
        )

        # --- deterministic difficulty order (inherited rule) -------------------
        def _net_difficulty(n: str) -> float:
            ps = pins_by_net[n]
            if len(ps) < 2:
                return 0.0
            xs = [p.xy_um[0] for p in ps]
            ys = [p.xy_um[1] for p in ps]
            return (max(xs) - min(xs)) + (max(ys) - min(ys))

        ordered_nets = sorted(pins_by_net.keys(), key=lambda n: (-_net_difficulty(n), n))
        ordered_nets = self._mirror_order(
            ordered_nets, mirror_by_b, hoist_pairs_first=True)
        net_to_id: Dict[str, int] = {n: i for i, n in enumerate(ordered_nets)}
        mirror_by_a: Dict[str, Mirror] = {mc.net_a: mc for mc in mirror_by_b.values()}

        _custom_pf_search = (
            getattr(type(self), "_route_one_net_soft", None) is not PathFinderEngine._route_one_net_soft
            or "_route_one_net_soft" in self.__dict__
            or getattr(type(self), "_astar", None) is not MazeRouteEngine._astar
            or "_astar" in self.__dict__
        )
        _custom_via_infeasible = (
            getattr(type(self), "_via_infeasible_cells", None) is not PathFinderEngine._via_infeasible_cells
            or "_via_infeasible_cells" in self.__dict__
        )
        native_grid = (
            None
            if _custom_pf_search
            else _build_native_grid(grid, tech, pdk, self._via_env_rule_nm)
        )
        if native_grid is not None:
            native_grid.set_over_active(
                self.over_active_rects_um if (self.over_active_penalty > 0.0 and over_active_gi_floor < tech.n_layers) else None,
                int(over_active_gi_floor),
                float(self.over_active_penalty),
            )

        def _upload_native_obstacles() -> None:
            if native_grid is None:
                return
            flat_obs = []
            for gi, rects in obs_by_gi.items():
                for (x0, y0, x1, y1, owner) in rects:
                    oid = -1 if owner is None else net_to_id.get(owner, -2)
                    flat_obs.append((gi, float(x0), float(y0), float(x1), float(y1), oid))
            flat_vobs = []
            for gl, vrects in via_obs_by_gi.items():
                for (x0, y0, x1, y1, owner, kind) in vrects:
                    oid = -1 if owner is None else net_to_id.get(owner, -2)
                    flat_vobs.append((gl, float(x0), float(y0), float(x1), float(y1), oid, kind == "cap_plate_via"))
            flat_mclr = []
            for gi, entries in mim_clear_by_gi.items():
                for (bx0, by0, bx1, by1, clr, own) in entries:
                    oid = -1 if own is None else net_to_id.get(own, -2)
                    flat_mclr.append((gi, int(bx0), int(by0), int(bx1), int(by1), int(clr), oid))
            flat_mcut = []
            for gl, rects in mim_forbidden_cut_by_gi.items():
                for (bx0, by0, bx1, by1) in rects:
                    flat_mcut.append((gl, int(bx0), int(by0), int(bx1), int(by1)))
            native_grid.set_static_obstacles(
                flat_obs, flat_vobs, flat_mclr, flat_mcut, envelope_bbox
            )

        def _build_net_states(cur_pins_by_net: Dict[str, List[Pin]]):
            if native_grid is not None:
                _upload_native_obstacles()
            net_layer_gis_m: Dict[str, List[int]] = {}
            net_width_nm_m: Dict[str, int] = {}
            net_pin_cells_m: Dict[str, List[Cell]] = {}
            net_width_at_m: Dict[str, object] = {}
            trunk_target_m: Dict[str, int] = {}
            invalid_nets_m: Set[str] = set()
            extra_unsat_m: List[RouteConstraint] = []
            mim_active_m: Dict[str, Dict[int, List[Tuple[int, int, int, int, int]]]] = {}
            net_blocked_m: Dict[str, Dict[int, Set[Tuple[int, int]]]] = {}

            for net, pins in cur_pins_by_net.items():
                if chk is not None:
                    chk()
                lg = sorted(set(layer_set_by_net.get(net) or usable))
                net_layer_gis_m[net] = lg
                w_nm = self._net_wire_width_nm(tech, lg, width_by_net.get(net))
                trunk_target_m[net] = w_nm
                route_w = min(tech.gi_min_width(gi) for gi in lg) if self.widen else w_nm
                net_width_nm_m[net] = route_w

                pin_xy_by_gi: Dict[int, List[Tuple[int, int]]] = {}
                for p in pins:
                    pgi = tech.resolve_layer_to_gi(p.layer)
                    if pgi is not None:
                        pin_xy_by_gi.setdefault(pgi, []).append((
                            int(round(float(p.xy_um[0]) * _NM_PER_UM)),
                            int(round(float(p.xy_um[1]) * _NM_PER_UM)),
                        ))
                mim_for_net: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
                for gi_m, entries in mim_clear_by_gi.items():
                    net_pins_gi = pin_xy_by_gi.get(gi_m, ())
                    active_entries: List[Tuple[int, int, int, int, int]] = []
                    for (bx0, by0, bx1, by1, clr, own) in entries:
                        if own is not None and own == net and any(
                            bx0 <= px <= bx1 and by0 <= py <= by1
                            for (px, py) in net_pins_gi
                        ):
                            continue
                        active_entries.append((bx0, by0, bx1, by1, clr))
                    if active_entries:
                        mim_for_net[gi_m] = active_entries
                if mim_for_net:
                    mim_active_m[net] = mim_for_net

                strict_layers = net in layer_set_by_net
                pin_cells = [
                    self._pin_cell(
                        grid, tech, p, lg,
                        strict_layers=strict_layers,
                        envelope_bbox=envelope_bbox,
                    )
                    for p in pins
                ]
                if any(cell is None for cell in pin_cells):
                    invalid_nets_m.add(net)
                    constraint = layer_set_constraint_by_net.get(net)
                    if constraint is not None and constraint not in extra_unsat_m:
                        extra_unsat_m.append(constraint)
                    net_pin_cells_m[net] = []
                    net_width_at_m[net] = (
                        lambda _ix, _iy, _gi, _width=route_w: _width
                    )
                else:
                    net_pin_cells_m[net] = [cell for cell in pin_cells if cell is not None]
                    net_width_at_m[net] = self._make_width_at(
                        grid, tech, net_pin_cells_m[net], net_width_nm_m[net]
                    )

                if native_grid is not None:
                    _own_floor = int(trunk_target_m[net]) if self.widen else 0
                    native_grid.prepare_net(
                        net_to_id[net],
                        int(net_width_nm_m[net]),
                        _own_floor,
                        lg,
                        net_pin_cells_m[net],
                        pin_xy_by_gi,
                        0,
                        True,
                        not _custom_via_infeasible,
                    )
                    net_blocked_m[net] = {}
                else:
                    blocked: Dict[int, Set[Tuple[int, int]]] = {}
                    for gi, rects in obs_by_gi.items():
                        cells: Set[Tuple[int, int]] = set()
                        halo = self._obstacle_halo_nm(tech, gi, net_width_nm_m[net])
                        pad_extra = self._via_pad_extra_halo_nm(tech, gi, net_width_nm_m[net])
                        for (x0, y0, x1, y1, owner) in rects:
                            if owner is not None and owner == net:
                                continue
                            h = halo + (pad_extra if owner is None else 0)
                            self._raster_rect(grid, cells, x0, y0, x1, y1, h)
                        mim_rects = mim_for_net.get(gi)
                        if mim_rects:
                            w_half = max(net_width_nm_m[net], tech.gi_min_width(gi)) // 2
                            for (bx0, by0, bx1, by1, clr) in mim_rects:
                                cand_m: Set[Tuple[int, int]] = set()
                                self._raster_rect(
                                    grid, cand_m,
                                    bx0 / _NM_PER_UM, by0 / _NM_PER_UM,
                                    bx1 / _NM_PER_UM, by1 / _NM_PER_UM,
                                    int(clr) + w_half,
                                )
                                for (ix_m, iy_m) in cand_m:
                                    cx_m, cy_m = grid.cell_xy(ix_m, iy_m)
                                    dx_m = max(0, bx0 - (cx_m + w_half), (cx_m - w_half) - bx1)
                                    dy_m = max(0, by0 - (cy_m + w_half), (cy_m - w_half) - by1)
                                    if dx_m < clr and dy_m < clr:
                                        cells.add((ix_m, iy_m))
                        if envelope_bbox is not None:
                            edge_half = max(net_width_nm_m[net], tech.gi_min_width(gi)) // 2
                            self._mark_envelope_blocked_cells(
                                grid, envelope_bbox, edge_half, cells
                            )
                        blocked[gi] = cells
                    net_blocked_m[net] = blocked

            net_via_inf_m: Dict[str, Dict[int, Set[Tuple[int, int]]]] = {}
            if native_grid is None or _custom_via_infeasible:
                for net in cur_pins_by_net:
                    if chk is not None:
                        chk()
                    _own_floor = int(trunk_target_m[net]) if self.widen else 0
                    vi_net = self._via_infeasible_cells(
                        grid, tech, pdk, obs_by_gi, net,
                        net_layer_gis_m[net], net_width_at_m[net], _own_floor,
                        envelope_bbox=envelope_bbox,
                        via_obs_by_gi=via_obs_by_gi,
                        mim_clear_by_gi=mim_active_m.get(net),
                        mim_forbidden_cut_by_gi=mim_forbidden_cut_by_gi,
                    )
                    net_via_inf_m[net] = vi_net
                    if native_grid is not None and vi_net:
                        flat_vi = [
                            (ix_v, iy_v, gl_v)
                            for gl_v, cset in vi_net.items()
                            for (ix_v, iy_v) in cset
                        ]
                        if flat_vi:
                            native_grid.set_net_via_blocked_cells(net_to_id[net], flat_vi)

            # Hard Fix #2 (Defense 1): disambiguate colliding/off-pad-blocked pin cells
            # onto an unblocked cell strictly inside their own device conductor pad.
            reloc_pin_nets = self._relocate_blocked_owned_pin_cells(
                grid=grid,
                tech=tech,
                pins_by_net=cur_pins_by_net,
                net_pin_cells=net_pin_cells_m,
                net_layer_gis=net_layer_gis_m,
                net_width_nm=net_width_nm_m,
                obs_by_gi=obs_by_gi,
                net_obstacle_blocked=net_blocked_m if native_grid is None else None,
                net_via_infeasible=net_via_inf_m if (native_grid is None or _custom_via_infeasible) else None,
                envelope_bbox=envelope_bbox,
                native_grid=native_grid,
                net_to_id=net_to_id,
            )
            for rnet in reloc_pin_nets:
                if rnet in invalid_nets_m:
                    continue
                net_width_at_m[rnet] = self._make_width_at(
                    grid, tech, net_pin_cells_m[rnet], net_width_nm_m[rnet]
                )
                if native_grid is not None:
                    r_pin_xy_by_gi: Dict[int, List[Tuple[int, int]]] = {}
                    for p in cur_pins_by_net[rnet]:
                        pgi = tech.resolve_layer_to_gi(p.layer)
                        if pgi is not None:
                            r_pin_xy_by_gi.setdefault(pgi, []).append((
                                int(round(float(p.xy_um[0]) * _NM_PER_UM)),
                                int(round(float(p.xy_um[1]) * _NM_PER_UM)),
                            ))
                    _own_floor = int(trunk_target_m[rnet]) if self.widen else 0
                    native_grid.prepare_net(
                        net_to_id[rnet],
                        int(net_width_nm_m[rnet]),
                        _own_floor,
                        net_layer_gis_m[rnet],
                        net_pin_cells_m[rnet],
                        r_pin_xy_by_gi,
                        0,
                        True,
                        not _custom_via_infeasible,
                    )
                    if _custom_via_infeasible and net_via_inf_m.get(rnet):
                        flat_vi = [
                            (ix_v, iy_v, gl_v)
                            for gl_v, cset in net_via_inf_m[rnet].items()
                            for (ix_v, iy_v) in cset
                        ]
                        if flat_vi:
                            native_grid.set_net_via_blocked_cells(net_to_id[rnet], flat_vi)

            return (
                net_layer_gis_m,
                net_width_nm_m,
                net_pin_cells_m,
                net_width_at_m,
                trunk_target_m,
                invalid_nets_m,
                extra_unsat_m,
                mim_active_m,
                net_blocked_m,
                net_via_inf_m,
            )

        (
            net_layer_gis,
            net_width_nm,
            net_pin_cells,
            net_width_at,
            trunk_target,
            invalid_layer_access_nets,
            layer_unsat,
            mim_active_by_net,
            net_obstacle_blocked,
            net_via_infeasible,
        ) = _build_net_states(route_pins_by_net)

        escape_risers_by_net = self._escape_sandwiched_pins(
            grid=grid,
            tech=tech,
            pdk=pdk,
            pins_by_net=route_pins_by_net,
            net_layer_gis=net_layer_gis,
            net_pin_cells=net_pin_cells,
            net_obstacle_blocked=net_obstacle_blocked,
            net_via_infeasible=net_via_infeasible,
            invalid_layer_access_nets=invalid_layer_access_nets,
            layer_set_by_net=layer_set_by_net,
            pre_cells=pre_cells,
            obs_by_gi=obs_by_gi,
            via_obs_by_gi=via_obs_by_gi,
            mim_active_by_net=mim_active_by_net,
            mim_forbidden_cut_by_gi=mim_forbidden_cut_by_gi,
            envelope_bbox=envelope_bbox,
            native_grid=native_grid,
            net_to_id=net_to_id,
        )
        if escape_risers_by_net:
            (
                net_layer_gis,
                net_width_nm,
                net_pin_cells,
                net_width_at,
                trunk_target,
                invalid_layer_access_nets,
                layer_unsat,
                mim_active_by_net,
                net_obstacle_blocked,
                net_via_infeasible,
            ) = _build_net_states(route_pins_by_net)

        for _c in layer_unsat:
            if _c not in unsatisfied:
                unsatisfied.append(_c)

        # --- per-net TRUNK layer assignment (deliberate multi-layer routing) ---
        trunk_by_net: Dict[str, int] = {}
        top_gi = tech.n_layers - 1
        if self.trunk_layers:
            for _n, _pcs in net_pin_cells.items():
                if len(_pcs) < 2:
                    trunk_by_net[_n] = 0
                    continue
                _xs = [c[0] for c in _pcs]; _ys = [c[1] for c in _pcs]
                _span = max(max(_xs) - min(_xs), max(_ys) - min(_ys))  # pitches
                _lg = net_layer_gis.get(_n) or list(range(tech.n_layers))
                _cap = max(_lg)
                trunk_by_net[_n] = max(0, min(top_gi, _cap, _span // _SPAN_PER_LEVEL))
            if native_grid is not None:
                for _n, _tg in trunk_by_net.items():
                    native_grid.set_net_trunk(net_to_id[_n], int(_tg))

        # --- negotiated-congestion rip-up / reroute loop -----------------------
        history: Dict[Cell, float] = {}
        via_pad_history: Dict[Cell, float] = {}
        pres_fac = self.pres_fac0

        pin_owners: Dict[Cell, Set[str]] = {}
        active_pin_net_ids: List[int] = []
        for _pn in ordered_nets:
            if _pn in invalid_layer_access_nets:
                continue
            if len(pins_by_net[_pn]) <= 1 and not pre_cells.get(_pn):
                continue
            active_pin_net_ids.append(net_to_id[_pn])
            if native_grid is None:
                for _pc in net_pin_cells[_pn]:
                    pin_owners.setdefault(_pc, set()).add(_pn)
        active_pin_net_id_set = set(active_pin_net_ids)

        best_metal: Optional[Dict[str, Set[Cell]]] = None
        best_wirecells: Dict[str, Set[Cell]] = {}
        best_connected: Dict[str, bool] = {}
        best_mirror_partial: Dict[str, Tuple[int, int]] = {}
        best_overused: List[Cell] = []
        best_pad_contended: List[Tuple[Cell, Optional[str], Tuple[str, ...]]] = []
        best_score: Optional[Tuple[int, int, int, int]] = None
        rounds_run = 0
        converged = False

        if native_grid is not None:
            for _it in range(self.max_iters):
                if chk is not None:
                    chk()
                rounds_run = _it + 1
                metal_cells: Dict[str, Set[Cell]] = {}
                wire_cells_by_net: Dict[str, Set[Cell]] = {}
                connected: Dict[str, bool] = {}
                reserved_by_b: Dict[str, Set[Cell]] = {}
                oob_reserved: Set[Cell] = set()
                mirror_partial: Dict[str, Tuple[int, int]] = {}

                native_grid.begin_round(active_pin_net_ids)
                dir_scale = (pres_fac / self.pres_fac0) if self.pres_fac0 > 0.0 else 1.0
                dir_pref_round = (
                    {gi: (ax, w * dir_scale)
                     for gi, (ax, w) in direction_pref.items()}
                    if direction_pref else None
                )

                for net_id, net in enumerate(ordered_nets):
                    if chk is not None:
                        chk()
                    pins = pins_by_net[net]
                    lg = net_layer_gis[net]
                    pin_cells = net_pin_cells[net]
                    seeds = pre_cells.get(net)
                    if net in invalid_layer_access_nets:
                        connected[net] = False
                        metal_cells[net] = set(seeds or ())
                        wire_cells_by_net[net] = set()
                        if seeds:
                            native_grid.commit_net_route(net_id, (), seeds, False)
                        continue
                    if len(pins) <= 1 and not seeds:
                        connected[net] = True
                        metal_cells[net] = set()
                        wire_cells_by_net[net] = set()
                        continue
                    if not pins:
                        connected[net] = True
                        metal_cells[net] = set(seeds)
                        wire_cells_by_net[net] = set()
                        native_grid.commit_net_route(net_id, (), seeds, False)
                        continue

                    if net_id in active_pin_net_id_set:
                        native_grid.mark_net_starting(net_id)

                    mirror_cells = reserved_by_b.get(net)
                    _mc_b = mirror_by_b.get(net)

                    def _native_is_blocked(ix: int, iy: int, gi: int, _nid=net_id) -> bool:
                        return bool(native_grid.is_blocked_cell(_nid, ix, iy, gi))

                    if (
                        mirror_cells
                        and not seeds
                        and self._mirror_fully_legal(
                            grid, mirror_cells, pin_cells, lg, _native_is_blocked
                        )
                    ):
                        wire_cells, ok = set(mirror_cells), True
                    else:
                        if mirror_cells:
                            _pinset = set(pin_cells)
                            _blocked_refl = sum(
                                1 for (ix, iy, gi) in mirror_cells
                                if (ix, iy, gi) not in _pinset
                                and (
                                    gi not in lg
                                    or not grid.in_bounds(ix, iy)
                                    or _native_is_blocked(ix, iy, gi)
                                )
                            )
                            if _blocked_refl and _mc_b is not None:
                                mirror_partial[net] = (
                                    _blocked_refl, len(mirror_cells))
                        wire_cells, ok = native_grid.route_one_net_soft(
                            net_id,
                            pin_cells,
                            lg,
                            seeds,
                            pres_fac=pres_fac,
                            hist_fac=self.hist_fac,
                            trunk_gi=trunk_by_net.get(net, 0),
                            use_via_cost=(_it > 0),
                            dir_pref=dir_pref_round,
                            mirror_cells=mirror_cells,
                            mirror_penalty=_MIRROR_HARD_PREFER_PENALTY if mirror_cells else 0.0,
                            max_expansions=self.max_expansions,
                            window_ladder=True,
                            use_pf_cost=True,
                            check_committed=False,
                            checkpoint_cb=chk,
                            checkpoint_interval=_CHECKPOINT_INTERVAL,
                        )

                    connected[net] = ok
                    native_grid.commit_net_route(net_id, wire_cells, seeds, ok)
                    if ok:
                        metal_cells[net] = set(wire_cells) | set(seeds or ())
                        wire_cells_by_net[net] = set(wire_cells)
                        _mc_a = mirror_by_a.get(net)
                        if _mc_a is not None and net not in mirror_by_b:
                            _kk = self._mirror_ix_sum(grid, _mc_a.axis_x_um)
                            _refl = {
                                (_kk - ix, iy, gi)
                                for (ix, iy, gi) in metal_cells[net]
                            }
                            _b = _mc_a.net_b
                            _in_refl = []
                            _oob_own: Set[Cell] = set()
                            for c_r in _refl:
                                if grid.in_bounds(c_r[0], c_r[1]) and 0 <= c_r[2] < tech.n_layers:
                                    _in_refl.append(c_r)
                                elif c_r not in oob_reserved:
                                    oob_reserved.add(c_r)
                                    _oob_own.add(c_r)
                            _own = native_grid.reserve_mirror_corridor(net_to_id[_b], _in_refl)
                            if _oob_own:
                                _own |= _oob_own
                            reserved_by_b[_b] = _own
                    else:
                        metal_cells[net] = set(seeds or ())
                        wire_cells_by_net[net] = set()

                round_conv, overused, pad_contended_raw = native_grid.finish_round(self.hist_fac)
                n_short_cells = len(overused)
                n_unconnected = sum(1 for ok in connected.values() if not ok)
                n_pad_contended = len(pad_contended_raw)
                score = (n_short_cells + n_unconnected, n_short_cells, n_unconnected,
                         n_pad_contended)
                if best_score is None or score < best_score:
                    best_score = score
                    best_metal = {k: set(v) for k, v in metal_cells.items()}
                    best_wirecells = {k: set(v) for k, v in wire_cells_by_net.items()}
                    best_connected = dict(connected)
                    best_mirror_partial = dict(mirror_partial)
                    best_overused = list(overused)
                    best_pad_contended = sorted(
                        (
                            c_p,
                            ordered_nets[mo_id] if mo_id >= 0 else None,
                            tuple(sorted(ordered_nets[oid] for oid in f_ids)),
                        )
                        for (c_p, mo_id, f_ids) in pad_contended_raw
                    )
                if round_conv:
                    converged = True
                    break
                pres_fac *= self.pres_fac_mult
        else:
            for _it in range(self.max_iters):
                if chk is not None:
                    chk()
                rounds_run = _it + 1
                metal_cells = {}
                wire_cells_by_net = {}
                connected = {}
                occupancy: Dict[Cell, int] = {}
                pad_keepout: Dict[Cell, Set[str]] = {}
                exact_pad_keepout: Dict[Cell, Set[str]] = {}
                pad_keepout_vias: Dict[Cell, Set[Tuple[str, Cell]]] = {}
                reserved_by_b = {}
                reserved_owner: Dict[Cell, str] = {}
                mirror_partial = {}

                dir_scale = (pres_fac / self.pres_fac0) if self.pres_fac0 > 0.0 else 1.0
                dir_pref_round = (
                    {gi: (ax, w * dir_scale)
                     for gi, (ax, w) in direction_pref.items()}
                    if direction_pref else None
                )

                for net in ordered_nets:
                    if chk is not None:
                        chk()
                    pins = pins_by_net[net]
                    lg = net_layer_gis[net]
                    pin_cells = net_pin_cells[net]
                    blocked = net_obstacle_blocked[net]
                    seeds = pre_cells.get(net)
                    if net in invalid_layer_access_nets:
                        connected[net] = False
                        metal_cells[net] = set(seeds or ())
                        wire_cells_by_net[net] = set()
                        continue
                    _vi = net_via_infeasible.get(net) or {}
                    via_blocked = self._make_via_blocked(_vi) if _vi else None

                    if len(pins) <= 1 and not seeds:
                        connected[net] = True
                        metal_cells[net] = set()
                        wire_cells_by_net[net] = set()
                        continue
                    if not pins:
                        connected[net] = True
                        metal_cells[net] = set(seeds)
                        wire_cells_by_net[net] = set()
                        continue

                    def is_blocked(
                        ix: int, iy: int, gi: int,
                        _blk=blocked, _ro=reserved_owner, _self=net,
                    ) -> bool:
                        if (ix, iy) in _blk.get(gi, ()):
                            return True
                        own = _ro.get((ix, iy, gi))
                        return own is not None and own != _self

                    _tg = trunk_by_net.get(net, 0)

                    def cell_cost(
                        ix: int, iy: int, gi: int, _tg=_tg, _self=net,
                        _oac=over_active_cells, _oag=over_active_gi_floor,
                        _oap=self.over_active_penalty, _pk=pad_keepout,
                    ) -> float:
                        cell = (ix, iy, gi)
                        occ = occupancy.get(cell, 0)
                        owners = _pk.get(cell)
                        pad_occ = sum(1 for o in owners if o != _self) if owners else 0
                        base = 1.0 + history.get(cell, 0.0) * self.hist_fac
                        trunk_pen = _TRUNK_BELOW_PENALTY * max(0, _tg - gi)
                        over_pen = (
                            _oap if (gi >= _oag and (ix, iy) in _oac) else 0.0
                        )
                        return (base * (1.0 + (occ + pad_occ) * pres_fac) - 1.0
                                + trunk_pen + over_pen)

                    mirror_cells = reserved_by_b.get(net)
                    _mc_b = mirror_by_b.get(net)
                    if (
                        mirror_cells
                        and not seeds
                        and self._mirror_fully_legal(
                            grid, mirror_cells, pin_cells, lg, is_blocked
                        )
                    ):
                        wire_cells, ok = set(mirror_cells), True
                    else:
                        if mirror_cells:
                            _pinset = set(pin_cells)
                            _blocked_refl = sum(
                                1 for (ix, iy, gi) in mirror_cells
                                if (ix, iy, gi) not in _pinset
                                and (
                                    gi not in lg
                                    or not grid.in_bounds(ix, iy)
                                    or is_blocked(ix, iy, gi)
                                )
                            )
                            if _blocked_refl and _mc_b is not None:
                                mirror_partial[net] = (
                                    _blocked_refl, len(mirror_cells))
                            _base_cost = cell_cost

                            def cell_cost(
                                ix: int, iy: int, gi: int,
                                _m=mirror_cells, _f=_base_cost,
                            ) -> float:
                                return _f(ix, iy, gi) + (
                                    0.0 if (ix, iy, gi) in _m
                                    else _MIRROR_HARD_PREFER_PENALTY
                                )

                        via_cost = (
                            None
                            if _it == 0
                            else self._make_via_cost(
                                grid=grid,
                                tech=tech,
                                net=net,
                                width_at=net_width_at.get(net),
                                occupancy=occupancy,
                                history=history,
                                via_pad_history=via_pad_history,
                                pin_owners=pin_owners,
                                routed_nets=metal_cells,
                                pres_fac=pres_fac,
                            )
                        )
                        wire_cells, ok = self._route_one_net_soft(
                            grid=grid,
                            tech=tech,
                            pin_cells=pin_cells,
                            layer_gis=lg,
                            is_blocked=is_blocked,
                            cell_cost=cell_cost,
                            dir_pref=dir_pref_round,
                            seed_cells=seeds,
                            via_blocked=via_blocked,
                            via_cost=via_cost,
                        )
                    connected[net] = ok
                    if ok:
                        metal_cells[net] = set(wire_cells) | set(seeds or ())
                        wire_cells_by_net[net] = set(wire_cells)
                        for c in wire_cells:
                            occupancy[c] = occupancy.get(c, 0) + 1
                        for c in self._via_pad_keepout_sources(
                            grid, tech, wire_cells, net_width_at.get(net),
                            exact_diagonal=False,
                        ):
                            pad_keepout.setdefault(c, set()).add(net)
                        exact_sources = self._via_pad_keepout_sources(
                            grid, tech, wire_cells, net_width_at.get(net),
                            exact_diagonal=True,
                        )
                        for c, vcells in exact_sources.items():
                            exact_pad_keepout.setdefault(c, set()).add(net)
                            pkv = pad_keepout_vias.setdefault(c, set())
                            for vc in vcells:
                                pkv.add((net, vc))
                        _mc_a = mirror_by_a.get(net)
                        if _mc_a is not None and net not in mirror_by_b:
                            _kk = self._mirror_ix_sum(grid, _mc_a.axis_x_um)
                            _refl = {
                                (_kk - ix, iy, gi)
                                for (ix, iy, gi) in metal_cells[net]
                            }
                            _b = _mc_a.net_b
                            _own: Set[Cell] = set()
                            for c in _refl:
                                if c in reserved_owner or occupancy.get(c, 0) > 0:
                                    continue
                                reserved_owner[c] = _b
                                _own.add(c)
                            reserved_by_b[_b] = _own
                    else:
                        metal_cells[net] = set(seeds or ())
                        wire_cells_by_net[net] = set()

                overused = [c for c, n in occupancy.items() if n >= 2]
                n_short_cells = len(overused)
                n_unconnected = sum(1 for ok in connected.values() if not ok)

                metal_owner: Dict[Cell, str] = {}
                for _mn, _mc in metal_cells.items():
                    for _c in _mc:
                        metal_owner.setdefault(_c, _mn)
                pad_contended: List[Cell] = []
                for c, owners in exact_pad_keepout.items():
                    mo = metal_owner.get(c)
                    if mo is not None and any(o != mo for o in owners):
                        pad_contended.append(c)

                n_pad_contended = len(pad_contended)
                score = (n_short_cells + n_unconnected, n_short_cells, n_unconnected,
                         n_pad_contended)
                if best_score is None or score < best_score:
                    best_score = score
                    best_metal = {k: set(v) for k, v in metal_cells.items()}
                    best_wirecells = {k: set(v) for k, v in wire_cells_by_net.items()}
                    best_connected = dict(connected)
                    best_mirror_partial = dict(mirror_partial)
                    best_overused = sorted(overused)
                    best_pad_contended = sorted(
                        (
                            c,
                            metal_owner.get(c),
                            tuple(
                                sorted(
                                    o
                                    for o in exact_pad_keepout[c]
                                    if o != metal_owner.get(c)
                                )
                            ),
                        )
                        for c in pad_contended
                    )

                if not overused and not pad_contended:
                    converged = True
                    break

                for c in overused:
                    history[c] = history.get(c, 0.0) + self.hist_fac
                for c in pad_contended:
                    history[c] = history.get(c, 0.0) + self.hist_fac
                    mo = metal_owner.get(c)
                    for via_net, vcell in pad_keepout_vias.get(c, ()):
                        if via_net != mo:
                            via_pad_history[vcell] = (
                                via_pad_history.get(vcell, 0.0) + self.hist_fac
                            )
                pres_fac *= self.pres_fac_mult

        # --- emit from the BEST round (honest: best, never a fabricated win) ----
        assert best_metal is not None  # we ran >=1 iteration (max_iters>=1 guard)
        # widen mode: fatten the thin routed centrelines to each net's Width target
        # wherever the channel has room (clearance-driven; DRC-clean by construction).
        if self.widen:
            extra_mim_clr = {
                (gi_m, n_m): rects_m
                for n_m, by_gi_m in mim_active_by_net.items()
                for gi_m, rects_m in by_gi_m.items()
            }
            emit_width_at = self._widen_to_clearance(
                grid, tech, best_metal, obs_by_gi, trunk_target,
                routing_envelope_nm=envelope_bbox,
                extra_obs_clearance_by_net_gi=extra_mim_clr or None,
            )
        elif envelope_bbox is not None:
            emit_width_at = {
                n: self._clamp_width_fn_to_envelope(
                    grid, net_width_at[n], envelope_bbox
                )
                for n in net_width_at
            }
        else:
            emit_width_at = net_width_at

        # --- width-aware cross-net occupancy for via-relocation extensions ----
        # The inherited emit may RELOCATE a clip-broken via with extension metal
        # (see engine_maze._emit_wires): such metal occupies cells the search
        # never reserved, so it must hold the symmetric width-aware clearance
        # from every OTHER net's metal -- including extensions other nets drew
        # earlier in THIS loop (the shared ``extra_owner`` map).
        _pitch = grid.pitch
        _cellowner: Dict[Cell, str] = {}
        for _mn, _mc in best_metal.items():
            for _c in _mc:
                _cellowner.setdefault(_c, _mn)
        for _mn, _mc in pre_cells.items():
            for _c in _mc:
                _cellowner.setdefault(_c, _mn)
        _w_all_max = max(
            [int(v) for v in trunk_target.values()]
            or [tech.gi_min_width(0)]
        )
        _extra_owner: Dict[Cell, Tuple[str, int]] = {}
        _extra_bridge_rects: Dict[str, Dict[Tuple[int, int], List[Tuple[int, int, int, int]]]] = {}

        # Precompute per-owner min-space-inflated metal rects ONCE across _cellowner
        # (instead of recomputing emit_width_at + cell_xy per net * per cell).
        _base_metal_rects_by_net: Dict[str, Dict[Tuple[int, int], List[Tuple[int, int, int, int]]]] = {}
        _ms_by_gi_cache = [int(tech.min_space_nm[tech._gi_to_pdk[l]]) for l in range(tech.n_layers)]
        _key_by_gi_cache = [tech.gi_to_metal_key(l) for l in range(tech.n_layers)]
        _gx0, _gy0, _gp = grid.x0, grid.y0, grid.pitch
        for c, own in _cellowner.items():
            jx, jy, l = c
            fn = emit_width_at.get(own)
            w_o = (int(fn(jx, jy, l)) if fn is not None
                   else int(trunk_target.get(own, _w_all_max)))
            reach = w_o // 2 + _ms_by_gi_cache[l]
            cx = _gx0 + jx * _gp
            cy = _gy0 + jy * _gp
            _base_metal_rects_by_net.setdefault(own, {}).setdefault(
                _key_by_gi_cache[l], []
            ).append((cx - reach, cy - reach, cx + reach, cy + reach))

        def _foreign_metal_rects_for(net_name: str):
            """Per-layer min-space-inflated keep-out rects (nm) of every OTHER net's
            metal -- the relational foreign-metal set the via landing pad must hold
            ``min_space`` to. Each foreign cell reserves ``+/- (foreign_half_width +
            min_space)`` about its centre, so a via pad kept clear of these holds the
            PDK same-layer spacing to that net's metal edge. PDK-sourced (the foreign
            wire width from ``emit_width_at`` / ``trunk_target`` + min_space); no
            per-PDK constant. The net's OWN cells are excluded (it may land on them)."""
            out_r: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
            for own, by_key in _base_metal_rects_by_net.items():
                if own == net_name:
                    continue
                for key, rlist in by_key.items():
                    out_r.setdefault(key, []).extend(rlist)
            for c, (own, w_e) in _extra_owner.items():
                if own == net_name:
                    continue
                jx, jy, l = c
                reach = int(w_e) // 2 + _ms_by_gi_cache[l]
                cx = _gx0 + jx * _gp
                cy = _gy0 + jy * _gp
                out_r.setdefault(_key_by_gi_cache[l], []).append(
                    (cx - reach, cy - reach, cx + reach, cy + reach))
            for own, by_key in _extra_bridge_rects.items():
                if own == net_name:
                    continue
                for key, rlist in by_key.items():
                    gi_b = tech.resolve_layer_to_gi(key)
                    ms_b = _ms_by_gi_cache[gi_b] if gi_b is not None else 0
                    bucket = out_r.setdefault(key, [])
                    for (bx0, by0, bx1, by1) in rlist:
                        bucket.append(
                            (bx0 - ms_b, by0 - ms_b, bx1 + ms_b, by1 + ms_b)
                        )
            return out_r

        def _make_taken(net_name: str):
            def _taken(jx: int, jy: int, l: int, w_ext: int) -> bool:
                ms = tech.min_space_nm[tech._gi_to_pdk[l]]
                rmax = max(1, int(math.ceil(
                    (w_ext / 2 + ms + _w_all_max / 2) / _pitch)))
                for dx in range(-rmax, rmax + 1):
                    for dy in range(-rmax, rmax + 1):
                        c = (jx + dx, jy + dy, l)
                        own = _cellowner.get(c)
                        w_o = None
                        if own is not None and own != net_name:
                            fn = emit_width_at.get(own)
                            w_o = (int(fn(jx + dx, jy + dy, l)) if fn is not None
                                   else int(trunk_target.get(own, _w_all_max)))
                        else:
                            eo = _extra_owner.get(c)
                            if eo is not None and eo[0] != net_name:
                                w_o = eo[1]
                        if w_o is None:
                            continue
                        r = max(abs(dx), abs(dy))
                        if r * _pitch < (w_ext + w_o) // 2 + ms:
                            return True
                return False
            return _taken

        for net in ordered_nets:
            pins = pins_by_net[net]
            pre = pre_wires_by_net.get(net)
            nr = NetRoute(net=net, pins=list(pins))
            ok = best_connected.get(net, False)
            if not ok:
                # genuinely unreachable pin even at high penalty -> honest failure:
                # connected=False, NO fabricated wire (the preroute is INPUT --
                # it already exists -- so it stays in the as-built picture).
                nr.connected = False
                nr.wires = list(pre) if pre else []
                nr.length_um = 0.0
                out.nets[net] = nr
                continue
            wire_cells = best_wirecells.get(net, set())
            if not wire_cells:
                # 0/1-pin nets (and pin-less preroute owners) draw no branch;
                # the verbatim preroute metal still flows through (contract 6c).
                nr.connected = True
                nr.wires = list(pre) if pre else []
                nr.length_um = 0.0
                out.nets[net] = nr
                continue
            # foreign keep-out rects (nm, per GDS layer) this net does NOT own ->
            # the (inherited) emit clips its pads/bridges/WIDENED metal/vias so no
            # foreign cap-plate (or device pad) is grazed, even by this net's own
            # seed-cell pad that the search cannot block. The widen target lives in
            # emit_width_at, so the widen pass cannot fatten metal over a plate.
            foreign_rects = self._foreign_emit_rects(tech, obs_by_gi, net)
            for gi_m, mim_rects in mim_active_by_net.get(net, {}).items():
                mkey = tech.gi_to_metal_key(gi_m)
                bucket = foreign_rects.setdefault(mkey, [])
                for (bx0, by0, bx1, by1, clr) in mim_rects:
                    bucket.append((bx0 - clr, by0 - clr, bx1 + clr, by1 + clr))
            # FOREIGN ROUTED-metal keep-out (the min-spacing obligation's emit face
            # for the via landing PAD): every OTHER net's metal cell, inflated by the
            # layer min_space, so the inherited via-landing emit holds the PDK
            # same-layer spacing from its rule pad to foreign metal (clip on the
            # foreign-facing side if enclosure+min_width allow, else relocate /
            # suppress -- never a silent spacing breach). Built from the FINAL
            # all-nets occupancy (``_cellowner`` + relocation ``_extra_owner``), so a
            # via sees foreign metal from nets routed both before AND after it. Every
            # dimension PDK-sourced (pitch + min_space); no per-PDK constant.
            foreign_metal_rects = _foreign_metal_rects_for(net)
            reloc: List[Tuple[int, int, int, int]] = []
            # D3: collect any via the envelope obligation SUPPRESSES WHOLE at
            # emit time (device-pad intrusion / buried cut, no legal relocation).
            # A suppressed via in this net's TREE is a real geometric OPEN, so its
            # ``connected`` flag must NOT stay True over it (native<->oracle
            # fidelity). D2 hard-forbids the device-pad cells in the search, so on
            # the happy path this stays empty and connected is unchanged; it is the
            # honest downgrade for any suppression the search approximation misses.
            via_suppressed: List[Tuple[int, int, int]] = []
            nr.wires, nr.length_um = self._emit_wires(
                grid, tech, wire_cells, net_pin_cells[net], trunk_target[net],
                width_at=emit_width_at[net],
                foreign_rects=foreign_rects,
                cell_taken=_make_taken(net), reloc_cells=reloc,
                foreign_metal_rects=foreign_metal_rects,
                suppressed_out=via_suppressed,
                routing_envelope_nm=envelope_bbox,
            )
            for (jx, jy, l, w_e) in reloc:
                # extension cells become this net's metal: later nets' taken
                # checks see them, and the short oracle counts them.
                _extra_owner[(jx, jy, l)] = (net, int(w_e))
                best_metal.setdefault(net, set()).add((jx, jy, l))
            nr.wires.extend(self._emit_pin_risers(
                grid, tech, route_pins_by_net[net], net_layer_gis[net], foreign_rects,
                strict_layers=net in layer_set_by_net,
            ))
            esc_wires = escape_risers_by_net.get(net)
            if esc_wires:
                nr.wires.extend(esc_wires)
            # contract 6c: the preroute rects join the net's wires VERBATIM (no
            # clip -- pre-existing metal) through the same per-net GDS merge.
            if pre:
                nr.wires.extend(pre)
            added_bridges = self._repair_clipped_terminal_gaps(
                tech, net, nr.wires, pins_by_net[net], obs_by_gi,
                foreign_rects, foreign_metal_rects,
                routing_envelope_nm=envelope_bbox,
            )
            if added_bridges:
                _extra_bridge_rects[net] = added_bridges
            # D3: downgrade the flag ONLY if a suppressed via actually leaves the
            # net's EMITTED metal geometrically OPEN (a load-bearing via). A
            # suppressed via the net routes around (a redundant / relocated-
            # elsewhere transition) must NOT falsely downgrade -- the native flag
            # must EQUAL the emitted-geometry truth, not a raw suppress count (the
            # anchor OTA has a suppressed via whose net stays ONE via-aware
            # component; a raw-count downgrade there made native disagree with the
            # via-aware oracle -- the exact fidelity the flag must uphold). The
            # via-aware openness check runs only when a suppression occurred
            # (byte-stable otherwise: no suppress -> connected=True as before).
            nr.connected = not (via_suppressed and self._emitted_net_open(tech, nr.wires))
            out.nets[net] = nr

        # preroute-only nets (no pins): existing metal, emitted verbatim,
        # vacuously connected -- and COUNTED by the short oracle below.
        for net in sorted(n for n in pre_wires_by_net if n not in pins_by_net):
            nr = NetRoute(net=net, pins=[])
            nr.connected = True
            nr.wires = list(pre_wires_by_net[net])
            out.nets[net] = nr
            best_metal[net] = set(pre_cells.get(net, ()))

        # --- spacing satisfaction bookkeeping (inherited semantics) ------------
        from clay.layout.route.wire_endpoints import materialize_required_wire_endpoints

        endpoint_failed = materialize_required_wire_endpoints(
            out.nets, pins_by_net, obstacles, tech
        )
        for net in endpoint_failed:
            best_connected[net] = False

        # a Spacing is satisfiable only if BOTH nets routed AND they do not share a
        # cell. Negotiated congestion guarantees the no-share part on convergence;
        # if either net failed or convergence left a short, record it unsatisfied.
        routed_ok = {n: best_connected.get(n, False) for n in pins_by_net}
        shorts = self._detect_shorts(best_metal)
        for sp in spacing_pairs:
            both = routed_ok.get(sp.net_a, False) and routed_ok.get(sp.net_b, False)
            if not both or shorts:
                if sp not in unsatisfied:
                    unsatisfied.append(sp)

        # --- Mirror satisfaction (shared base report): exact or honest ---------
        # Judged on the EMITTED round's metal; any asymmetry is quantified in
        # the per-pair report -- never a silent claim of symmetry.
        out.mirror_reports = []
        for b_net in sorted(mirror_by_b):
            mc_b = mirror_by_b[b_net]
            rep = self._mirror_report(
                grid, mc_b,
                best_metal.get(mc_b.net_a, set()),
                best_metal.get(b_net, set()),
                bool(best_connected.get(mc_b.net_a, False)),
                bool(best_connected.get(b_net, False)),
            )
            # HONEST partial: reflection cells that could not be mirrored because
            # they collide with FIXED blockage (device artwork / immovable pins).
            _bp = best_mirror_partial.get(b_net)
            if _bp is not None:
                rep["n_reflection_cells_blocked"] = _bp[0]
                rep["n_reflection_cells"] = _bp[1]
            out.mirror_reports.append(rep)
            if not rep["satisfied"] and mc_b not in unsatisfied:
                unsatisfied.append(mc_b)

        # --- final honest verdict ---------------------------------------------
        out.shorts = shorts
        out.unsatisfied = unsatisfied
        all_connected = all(nr.connected for nr in out.nets.values())
        # success requires real convergence (no overuse) AND every net connected
        # AND no recorded-unsatisfied constraint. Non-convergence within max_iters
        # leaves shorts populated and success False -- NEVER faked.
        out.success = converged and all_connected and not shorts and not unsatisfied
        out.negotiation_rounds = rounds_run
        out.best_overused_cells = tuple(best_overused)
        out.best_pad_contended_cells = tuple(best_pad_contended)
        reasons: List[str] = []
        if not converged:
            reasons.append(
                f"negotiated_congestion_unconverged(rounds={rounds_run}, "
                f"overused={len(best_overused)}, pad_contended={len(best_pad_contended)})"
            )
        unconn = sorted(n for n, nr in out.nets.items() if not nr.connected)
        if unconn:
            reasons.append(f"unconnected_nets={unconn}")
        if shorts:
            reasons.append(f"shorts={len(shorts)}")
        if unsatisfied:
            reasons.append(
                f"unsatisfied_constraints={[repr(c) for c in unsatisfied]}"
            )
        out.convergence_failure_reasons = tuple(reasons)
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mark_envelope_blocked_cells(
        grid,
        envelope_bbox: Tuple[int, int, int, int],
        edge_half_nm: int,
        sink: Set[Tuple[int, int]],
    ) -> None:
        """Mark grid cells whose distance to any floorplan edge is narrower than
        ``edge_half_nm`` so search never enters boundary tracks that would protrude."""
        ex0, ey0, ex1, ey1 = envelope_bbox
        if (ex1 - ex0) < 2 * edge_half_nm or (ey1 - ey0) < 2 * edge_half_nm:
            return
        vx_lo = max(0, int(math.ceil((ex0 + edge_half_nm - grid.x0) / grid.pitch)))
        vx_hi = min(grid.nx - 1, int(math.floor((ex1 - edge_half_nm - grid.x0) / grid.pitch)))
        vy_lo = max(0, int(math.ceil((ey0 + edge_half_nm - grid.y0) / grid.pitch)))
        vy_hi = min(grid.ny - 1, int(math.floor((ey1 - edge_half_nm - grid.y0) / grid.pitch)))
        if vx_lo > vx_hi or vy_lo > vy_hi:
            return
        for ix in range(0, vx_lo):
            for iy in range(grid.ny):
                sink.add((ix, iy))
        for ix in range(vx_hi + 1, grid.nx):
            for iy in range(grid.ny):
                sink.add((ix, iy))
        for ix in range(vx_lo, vx_hi + 1):
            for iy in range(0, vy_lo):
                sink.add((ix, iy))
            for iy in range(vy_hi + 1, grid.ny):
                sink.add((ix, iy))

    @staticmethod
    def _clamp_width_fn_to_envelope(
        grid,
        width_fn,
        envelope_bbox: Tuple[int, int, int, int],
    ):
        """Clamp a width lookup function so emitted half-widths never exceed the
        declared floorplan envelope."""
        ex0, ey0, ex1, ey1 = envelope_bbox

        def clamped(ix: int, iy: int, gi: int) -> int:
            w = int(width_fn(ix, iy, gi))
            cx, cy = grid.cell_xy(ix, iy)
            edge_clear = min(cx - ex0, ex1 - cx, cy - ey0, ey1 - cy)
            return min(w, 2 * max(0, int(edge_clear)))

        return clamped

    @staticmethod
    def _pin_cell(
        grid,
        tech: _Tech,
        p: Pin,
        layer_gis: List[int],
        *,
        strict_layers: bool = False,
        envelope_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Cell]:
        """Resolve a pin to its physical ``(ix, iy, gi)`` seed/target cell.

        An off-cap/unknown layer returns ``None``; terminal access is never moved
        to the net's first allowed layer. When ``envelope_bbox`` is given, the
        snapped track is clamped inside the envelope by at least half the seed
        layer's minimum width so terminal metal stays inside the floorplan.
        """
        ix, iy = grid.snap_index(float(p.xy_um[0]), float(p.xy_um[1]))
        seed = tech.pin_seed(
            p.layer, layer_gis, strict_layers=strict_layers
        )
        if seed is None:
            return None
        gi, _ladder = seed
        if envelope_bbox is not None:
            ex0, ey0, ex1, ey1 = envelope_bbox
            edge_half = max(1, tech.gi_min_width(gi) // 2)
            ix_lo = max(0, int(math.ceil((ex0 + edge_half - grid.x0) / grid.pitch)))
            ix_hi = min(grid.nx - 1, int(math.floor((ex1 - edge_half - grid.x0) / grid.pitch)))
            iy_lo = max(0, int(math.ceil((ey0 + edge_half - grid.y0) / grid.pitch)))
            iy_hi = min(grid.ny - 1, int(math.floor((ey1 - edge_half - grid.y0) / grid.pitch)))
            if ix_lo <= ix_hi:
                ix = max(ix_lo, min(ix_hi, ix))
            if iy_lo <= iy_hi:
                iy = max(iy_lo, min(iy_hi, iy))
        return (ix, iy, gi)

    # NOTE: ``_make_width_at`` (per-net tapered width fn) lives on the BASE
    # :class:`~clay.layout.route.engine_maze.MazeRouteEngine` and is INHERITED here
    # -- ``self._make_width_at(...)`` below resolves to that single definition (no
    # duplicate copy). The base uses it under its opt-in ``taper`` flag; PathFinder
    # always tapers (the neck fits tight terminals, the trunk matches the reference
    # per-net width).

    def _net_footprint(self, grid, tech: _Tech, wire_cells: Set[Cell], width_at) -> Set[Cell]:
        """Clearance footprint of a net's metal using its LOCAL (tapered) width:
        each metal cell dilated by ceil((local_half_width + half_min_space)/pitch)
        tracks, per layer. Two nets' footprints overlap iff their wire EDGES are
        within the min-space, so using this footprint (not the bare metal cell) as
        the negotiated-congestion occupancy makes rip-up enforce SPACING, not merely
        same-cell non-overlap. A NECKED cell reserves a small footprint (lets two
        nets land on tight adjacent terminals); a TRUNK cell reserves full clearance
        (keeps parallel channel runs min-space apart)."""
        pitch = grid.pitch
        foot: Set[Cell] = set()
        # via cells (a cell occupied on two graph-adjacent layers carries a cut):
        # their landing PAD (cut + 2*enclosure) is WIDER than the wire, so they
        # reserve a larger footprint -- otherwise the negotiated congestion lets a
        # foreign net land within min_space of the pad edge (the gf180 M*.2a family
        # the via pad creates). The half-extent used is max(wire_half, pad_half),
        # all PDK-sourced (cut size + enclosure), no per-PDK constant.
        via_low_at: Dict[Tuple[int, int], int] = {}
        by_xy: Dict[Tuple[int, int], Set[int]] = {}
        for (ix, iy, gi) in wire_cells:
            by_xy.setdefault((ix, iy), set()).add(gi)
        for (ix, iy), gis in by_xy.items():
            for gi in gis:
                if (gi + 1) in gis:
                    via_low_at[(ix, iy)] = gi

        def _pad_half(ix: int, iy: int, gi: int) -> int:
            """The via landing pad half-width on layer ``gi`` at this cell, or 0 if
            no via here / no enclosure owed -- so a non-via cell keeps the wire half."""
            gl = via_low_at.get((ix, iy))
            if gl is None:
                return 0
            if gi == gl:
                enc = self._via_env_rule_nm(tech, gl)[0]
            elif gi == gl + 1:
                enc = self._via_env_rule_nm(tech, gl)[1]
            else:
                return 0
            if enc <= 0:
                return 0
            vsz = _via_cut_size_nm(tech, gl)
            return vsz // 2 + int(enc)

        for (ix, iy, gi) in wire_cells:
            w = int(width_at(ix, iy, gi))
            half = max(w // 2, _pad_half(ix, iy, gi))
            min_space = tech.min_space_nm[tech._gi_to_pdk[gi]]
            r = max(0, int(math.ceil((half + min_space / 2) / pitch)))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    jx, jy = ix + dx, iy + dy
                    if grid.in_bounds(jx, jy):
                        foot.add((jx, jy, gi))
        return foot

    def _via_pad_extra_halo_nm(self, tech: _Tech, gi: int, wire_width_nm: int) -> int:
        """The EXTRA keep-out reach (nm) a VIA LANDING PAD on graph-layer ``gi``
        adds BEYOND the wire-centreline halo, for the R3 routed-metal-vs-fixed-
        artwork obligation.

        A via touching ``gi`` may land at ``gi-1<->gi`` or ``gi<->gi+1``; on each
        side the pad half is ``ceil(cut/2) + enclosure`` for that via. The wire
        halo already reserves ``min_space + wire_half``; the pad only adds reach
        where ``pad_half > wire_half`` (the min-width-wire case). Return the
        largest such surplus over the (gi-1, gi+1) adjacent vias. DEFAULT-SAFE:
        when no via at ``gi`` exceeds the wire (external_pdk wide wires) -> 0, so the
        fixed-artwork halo is byte-identical. Every term is PDK-sourced (the SAME
        ``_via_env_rule_nm`` / ``_via_cut_size_nm`` the emit uses); no per-PDK
        constant."""
        from clay.layout.route.engine_maze import _via_cut_size_nm
        wire_half = max(0, int(wire_width_nm) // 2)
        extra = 0
        for gl in (gi - 1, gi):
            if gl < 0 or (gl + 1) >= tech.n_layers:
                continue
            try:
                rb, ra = self._via_env_rule_nm(tech, gl)
            except Exception:
                rb, ra = 0, 0
            try:
                vsz = _via_cut_size_nm(tech, gl)
            except Exception:
                vsz = 0
            if not vsz:
                continue
            # the pad on THIS gi is the below-side of via (gl,gl+1) when gl==gi,
            # else the above-side when gl==gi-1.
            enc = ra if gl == (gi - 1) else rb
            pad_half = (int(vsz) + 1) // 2 + max(0, int(enc))
            extra = max(extra, pad_half - wire_half)
        return max(0, extra)

    @staticmethod
    def _emitted_net_open(tech: _Tech, wires) -> bool:
        """True when a net's EMITTED wires form MORE THAN ONE via-aware component
        (a geometric OPEN) -- the SAME physics as the independent oracle
        (``anchor_ota._geometric_open_nets``): per-layer metal islands union
        same-layer on direct overlap and CROSS-layer ONLY through a via cut that
        overlaps both. Rect-based (the emitted geometry is all axis-aligned
        rectangles), so no KLayout dependency on this native path. Used by D3 to
        decide whether an emit-time via SUPPRESSION actually severed the net (a
        load-bearing via) versus a via the net routes around (redundant). 0/1
        island -> not open."""
        metal_keys: Set[Tuple[int, int]] = set()
        for gi in range(tech.n_layers):
            try:
                metal_keys.add(tuple(tech.gi_to_metal_key(gi)))
            except Exception:
                pass
        via_keys: Set[Tuple[int, int]] = set()
        for gl in range(max(0, tech.n_layers - 1)):
            try:
                vk = tech.via_key_for(gl)
            except Exception:
                vk = None
            if vk:
                via_keys.add(tuple(vk))
        metal: List[Tuple[Tuple[int, int], Tuple[int, int, int, int]]] = []
        vias: List[Tuple[int, int, int, int]] = []
        for lp in wires or ():
            key = (int(lp.layer), int(lp.datatype))
            xs = [int(v.x) for v in lp.polygon.vertices]
            ys = [int(v.y) for v in lp.polygon.vertices]
            rect = (min(xs), min(ys), max(xs), max(ys))
            if key in via_keys:
                vias.append(rect)
            elif key in metal_keys:
                metal.append((key, rect))
        n = len(metal)
        if n <= 1:
            return False  # 0 or 1 island -> trivially one component
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def ov(a, b) -> bool:  # touch-inclusive overlap
            return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

        for i in range(n):
            for j in range(i + 1, n):
                if metal[i][0] == metal[j][0] and ov(metal[i][1], metal[j][1]):
                    union(i, j)
        for vr in vias:
            touch = [i for i in range(n) if ov(metal[i][1], vr)]
            for t in touch[1:]:
                union(touch[0], t)
        return len({find(i) for i in range(n)}) > 1

    @staticmethod
    def _make_via_blocked(via_infeasible: Dict[int, Set[Tuple[int, int]]]):
        """Bind the per-net infeasible-via map into a ``fn(ix, iy, gi_low) ->
        bool`` for the search (Task D2). True forbids the layer transition."""
        def via_blocked(ix: int, iy: int, gi_low: int, _vi=via_infeasible) -> bool:
            s = _vi.get(gi_low)
            return s is not None and (ix, iy) in s
        return via_blocked

    @staticmethod
    def _mim_bottom_components(
        tech: _Tech,
        pdk,
        obstacles: Sequence[Obstacle],
    ) -> Tuple[
        Dict[int, List[Tuple[int, int, int, int, int, Optional[str]]]],
        Dict[int, List[Tuple[int, int, int, int]]],
    ]:
        """Resolve PDK-declared MIM bottom-plate routing clearance and forbidden
        cut-touch components directly from literal artwork + marker obstacles."""
        from clay.layout.model import ROUTING_ABSTRACTION_KINDS

        contract = getattr(tech, "pdk", pdk)
        clear_rules = tuple(
            contract.mim_bottom_routing_clearance_rules()
            if hasattr(contract, "mim_bottom_routing_clearance_rules")
            else ()
        )
        cut_rules = tuple(
            contract.mim_forbidden_cut_touch_rules()
            if hasattr(contract, "mim_forbidden_cut_touch_rules")
            else ()
        )
        if not clear_rules and not cut_rules:
            return {}, {}

        needed_metals: Set[Tuple[int, int]] = set()
        needed_markers: Set[Tuple[int, int]] = set()
        for cond_layer, marker_layer, _clr in clear_rules:
            needed_metals.add(tuple(cond_layer))
            needed_markers.add(tuple(marker_layer))
        for _cut_layer, cond_layer, marker_layer in cut_rules:
            needed_metals.add(tuple(cond_layer))
            needed_markers.add(tuple(marker_layer))

        markers_by_layer: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
        conductors_by_layer: Dict[
            Tuple[int, int],
            List[Tuple[int, int, int, int, Optional[str]]],
        ] = {}
        for ob in obstacles:
            if getattr(ob, "kind", None) in ROUTING_ABSTRACTION_KINDS:
                continue
            if not isinstance(ob.layer, (tuple, list)) or len(ob.layer) != 2:
                continue
            lk = (int(ob.layer[0]), int(ob.layer[1]))
            if lk not in needed_markers and lk not in needed_metals:
                continue
            rect = (
                int(round(min(ob.x0, ob.x1) * _NM_PER_UM)),
                int(round(min(ob.y0, ob.y1) * _NM_PER_UM)),
                int(round(max(ob.x0, ob.x1) * _NM_PER_UM)),
                int(round(max(ob.y0, ob.y1) * _NM_PER_UM)),
            )
            if lk in needed_markers:
                markers_by_layer.setdefault(lk, []).append(rect)
            if lk in needed_metals:
                conductors_by_layer.setdefault(lk, []).append((*rect, ob.owner_net))

        def _touch(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
            return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

        def _selected(cond_layer: Tuple[int, int], marker_layer: Tuple[int, int]):
            mrects = markers_by_layer.get(marker_layer, ())
            crects = conductors_by_layer.get(cond_layer, ())
            if not mrects or not crects:
                return []
            picked = [False] * len(crects)
            queue: List[int] = []
            for i, c in enumerate(crects):
                if any(_touch(c[:4], m) for m in mrects):
                    picked[i] = True
                    queue.append(i)
            while queue:
                idx = queue.pop()
                ci = crects[idx]
                for j, cj in enumerate(crects):
                    if picked[j]:
                        continue
                    if ci[4] == cj[4] and _touch(ci[:4], cj[:4]):
                        picked[j] = True
                        queue.append(j)
            return [crects[i] for i, flag in enumerate(picked) if flag]

        mim_clear_by_gi: Dict[
            int, List[Tuple[int, int, int, int, int, Optional[str]]]
        ] = {}
        for cond_layer, marker_layer, clearance_nm in clear_rules:
            gi = tech.resolve_layer_to_gi(cond_layer)
            if gi is None or int(clearance_nm) <= 0:
                continue
            for (bx0, by0, bx1, by1, own) in _selected(
                tuple(cond_layer), tuple(marker_layer)
            ):
                mim_clear_by_gi.setdefault(gi, []).append(
                    (bx0, by0, bx1, by1, int(clearance_nm), own)
                )

        via_lo_by_key: Dict[Tuple[int, int], int] = {}
        for gl in range(max(0, tech.n_layers - 1)):
            vk = tech.via_key_for(gl)
            if vk is not None:
                via_lo_by_key[tuple(vk)] = gl

        mim_forbidden_cut_by_gi: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for cut_layer, cond_layer, marker_layer in cut_rules:
            gl = via_lo_by_key.get(tuple(cut_layer))
            if gl is None:
                continue
            for (bx0, by0, bx1, by1, _own) in _selected(
                tuple(cond_layer), tuple(marker_layer)
            ):
                mim_forbidden_cut_by_gi.setdefault(gl, []).append(
                    (bx0, by0, bx1, by1)
                )

        return mim_clear_by_gi, mim_forbidden_cut_by_gi

    def _via_infeasible_cells(
        self, grid, tech: _Tech, pdk, obs_by_gi, net: str,
        layer_gis: List[int], width_at, own_width_floor: int = 0,
        *,
        envelope_bbox: Optional[Tuple[int, int, int, int]] = None,
        via_obs_by_gi: Optional[
            Dict[int, Sequence[Tuple[float, float, float, float, Optional[str], Optional[str]]]]
        ] = None,
        mim_clear_by_gi: Optional[
            Dict[int, Sequence[Tuple[int, int, int, int, int]]]
        ] = None,
        mim_forbidden_cut_by_gi: Optional[
            Dict[int, Sequence[Tuple[int, int, int, int]]]
        ] = None,
    ) -> Dict[int, Set[Tuple[int, int]]]:
        """The ``{gi_low: {(ix, iy)}}`` cells where a via between ``gi_low`` and
        ``gi_low+1`` would have a pad-infeasible landing for ``net`` -- the
        DEVICE-pad-intrusion verdict of the SHARED
        :func:`clay.layout.route.via_model.via_landing_feasible`, the SAME
        judgment ``_emit_via_obligated._emit_landing`` enforces at emit time --
        together with physical via-cut keepouts (``cap_plate_via`` and net-blind
        ``device_cut`` spacing), PDK ``mim_forbidden_cut_touch`` and
        ``mim_bottom_routing_clearance`` pad obligations.
        The search hard-forbids these transitions (Task D2, closing F1) so it
        stops committing a via the emit path then suppresses.

        Predicate inputs mirror the emit call, in the SEARCH direction:
          * own cover = a SQUARE of the net's EMIT wire width at the cell on each
            side: the tapered routing width raised to ``own_width_floor`` (the
            widen target in widen mode, 0 otherwise -- see the caller). This gives
            the SAME coverage decision the emit's per-cell pad gives (a min-width
            wire owes the full landing pad; a wire that already covers the cut owes
            none -- the emit minimality), WITHOUT the width-form neck short-circuit,
            so the gate fires on exactly the device intrusion the emit path
            suppresses. Using the WIDEN width here is essential: widen mode routes
            thin but EMITS widened metal, so a thin own-cover would over-owe the
            pad and FALSE-block a via emit actually covers.
          * foreign DEVICE rects = ``obs_by_gi`` rects this net does not own
            (raw um -> nm); the per-layer min-space inflation is applied INSIDE
            the predicate (``min_space_nm_by_gi``), the mapping convention that
            matches the emit path's PRE-inflated sets edge for edge.
          * foreign ROUTED metal = None: no committed metal exists in search, and
            a routed-metal breach is a tolerated spacing residual, never a
            suppression -- so it must not hard-block.
        cut/enclosure are INJECTED from the engine resolver (``_via_cut_size_nm``
        / ``_via_env_rule_nm``) so predicate == executor numbers by construction
        (never the stricter ``via_name`` lookup -- see ``via_model`` D2 notes).

        Only DEVICE-pad-intrusion (ok=False) is recorded: it is the sole verdict
        that drives an emit suppress-into-open. ``no-cut-size`` / ``cut-exceeds-
        neck`` are not blocked (they are deck accounts, not opens), and the
        legacy no-enclosure path owes no landing at all. Evaluated only at cells
        within pad reach of a foreign device rect (device-adjacent), so a route
        with no nearby device is byte-identical to the pre-D2 search."""
        out: Dict[int, Set[Tuple[int, int]]] = {}
        allowed = set(layer_gis)
        for gi_low in range(tech.n_layers - 1):
            if gi_low not in allowed or (gi_low + 1) not in allowed:
                continue
            vsz = _via_cut_size_nm(tech, gi_low)
            if not vsz:
                continue
            cut_h = max(1, (int(vsz) + 1) // 2)
            rb, ra = self._via_env_rule_nm(tech, gi_low)
            infeasible: Set[Tuple[int, int]] = set()
            raw_via_obs = via_obs_by_gi.get(gi_low, ()) if via_obs_by_gi else ()
            if raw_via_obs:
                contract = getattr(tech, "pdk", pdk)
                fallback_sp = int(tech.min_space_nm[tech._gi_to_pdk[gi_low]])
                vsp = _via_cut_space_nm(contract, _via_name_for(tech, gi_low), fallback_sp)
                reach_nm = cut_h + max(0, int(vsp))
                for (x0, y0, x1, y1, owner, kind) in raw_via_obs:
                    bx0 = int(round(min(x0, x1) * _NM_PER_UM))
                    by0 = int(round(min(y0, y1) * _NM_PER_UM))
                    bx1 = int(round(max(x0, x1) * _NM_PER_UM))
                    by1 = int(round(max(y0, y1) * _NM_PER_UM))
                    cand_cut: Set[Tuple[int, int]] = set()
                    self._raster_rect(grid, cand_cut, x0, y0, x1, y1, reach_nm)
                    for (ix, iy) in cand_cut:
                        cx, cy = grid.cell_xy(ix, iy)
                        if (
                            kind != "cap_plate_via"
                            and owner is not None
                            and owner == net
                            and (cx - cut_h, cy - cut_h, cx + cut_h, cy + cut_h)
                            == (bx0, by0, bx1, by1)
                        ):
                            continue
                        dx = max(0, bx0 - (cx + cut_h), (cx - cut_h) - bx1)
                        dy = max(0, by0 - (cy + cut_h), (cy - cut_h) - by1)
                        if (dx == 0 and dy == 0) or math.hypot(dx, dy) < vsp:
                            infeasible.add((ix, iy))
            for (bx0, by0, bx1, by1) in (
                mim_forbidden_cut_by_gi.get(gi_low, ())
                if mim_forbidden_cut_by_gi
                else ()
            ):
                cand_f: Set[Tuple[int, int]] = set()
                self._raster_rect(
                    grid, cand_f,
                    bx0 / _NM_PER_UM, by0 / _NM_PER_UM,
                    bx1 / _NM_PER_UM, by1 / _NM_PER_UM,
                    cut_h,
                )
                for (ix, iy) in cand_f:
                    cx, cy = grid.cell_xy(ix, iy)
                    if not (
                        cx + cut_h < bx0
                        or bx1 < cx - cut_h
                        or cy + cut_h < by0
                        or by1 < cy - cut_h
                    ):
                        infeasible.add((ix, iy))
            if mim_clear_by_gi:
                for gs, enc in ((gi_low, rb), (gi_low + 1, ra)):
                    mim_rects = mim_clear_by_gi.get(gs, ())
                    if not mim_rects:
                        continue
                    pad_enc = cut_h + max(0, int(enc))
                    for (bx0, by0, bx1, by1, clr) in mim_rects:
                        cand_mc: Set[Tuple[int, int]] = set()
                        self._raster_rect(
                            grid, cand_mc,
                            bx0 / _NM_PER_UM, by0 / _NM_PER_UM,
                            bx1 / _NM_PER_UM, by1 / _NM_PER_UM,
                            int(clr) + max(pad_enc, own_width_floor // 2),
                        )
                        for (ix, iy) in cand_mc:
                            if (ix, iy) in infeasible:
                                continue
                            cx, cy = grid.cell_xy(ix, iy)
                            pad_g = max(
                                pad_enc,
                                max(int(width_at(ix, iy, gs)), own_width_floor) // 2,
                            )
                            dx = max(0, bx0 - (cx + pad_g), (cx - pad_g) - bx1)
                            dy = max(0, by0 - (cy + pad_g), (cy - pad_g) - by1)
                            if dx < clr and dy < clr:
                                infeasible.add((ix, iy))
            if rb > 0 or ra > 0:
                ms_by_gi = {
                    gs: int(tech.min_space_nm[tech._gi_to_pdk[gs]])
                    for gs in (gi_low, gi_low + 1)
                }
                # foreign DEVICE rects (raw nm) on the two via layers, keyed by gi.
                dev_by_gi: Dict[int, List[Tuple[int, int, int, int]]] = {
                    gi_low: [], gi_low + 1: [],
                }
                for gs in (gi_low, gi_low + 1):
                    for (x0, y0, x1, y1, owner) in obs_by_gi.get(gs, ()):  # um
                        if owner is not None and owner == net:
                            continue  # this net's OWN terminal pad -> may land on it
                        dev_by_gi[gs].append((
                            int(round(min(x0, x1) * _NM_PER_UM)),
                            int(round(min(y0, y1) * _NM_PER_UM)),
                            int(round(max(x0, x1) * _NM_PER_UM)),
                            int(round(max(y0, y1) * _NM_PER_UM)),
                        ))
                pad_half = (int(vsz) + 1) // 2 + max(int(rb), int(ra))
                if envelope_bbox is not None and (rb > 0 or ra > 0 or vsz > 0):
                    cut_h = max(1, (int(vsz) + 1) // 2)
                    via_reach = max(
                        cut_h + max(0, int(rb)),
                        cut_h + max(0, int(ra)),
                    )
                    self._mark_envelope_blocked_cells(
                        grid, envelope_bbox, via_reach, infeasible
                    )
                if dev_by_gi[gi_low] or dev_by_gi[gi_low + 1]:
                    # candidate cells: only those from which a pad could reach a device.
                    cand: Set[Tuple[int, int]] = set()
                    for gs in (gi_low, gi_low + 1):
                        reach = pad_half + ms_by_gi[gs]
                        for (x0, y0, x1, y1, owner) in obs_by_gi.get(gs, ()):  # um
                            if owner is not None and owner == net:
                                continue
                            self._raster_rect(grid, cand, x0, y0, x1, y1, reach)
                    # Pre-inflate device holes by layer min_space so candidate own-cover
                    # squares reflect what _foreign_emit_rects clipping actually leaves.
                    dev_holes_by_gi = {
                        gs: [
                            (
                                x0 - ms_by_gi[gs],
                                y0 - ms_by_gi[gs],
                                x1 + ms_by_gi[gs],
                                y1 + ms_by_gi[gs],
                            )
                            for (x0, y0, x1, y1) in dev_by_gi[gs]
                        ]
                        for gs in (gi_low, gi_low + 1)
                    }
                    for (ix, iy) in cand:
                        if (ix, iy) in infeasible:
                            continue
                        cx, cy = grid.cell_xy(ix, iy)
                        hb = max(1, max(int(width_at(ix, iy, gi_low)), own_width_floor) // 2)
                        ht = max(1, max(int(width_at(ix, iy, gi_low + 1)), own_width_floor) // 2)
                        own_b = [(cx - hb, cy - hb, cx + hb, cy + hb)]
                        if dev_holes_by_gi[gi_low]:
                            own_b = subtract_rects(own_b[0], dev_holes_by_gi[gi_low])
                        own_a = [(cx - ht, cy - ht, cx + ht, cy + ht)]
                        if dev_holes_by_gi[gi_low + 1]:
                            own_a = subtract_rects(own_a[0], dev_holes_by_gi[gi_low + 1])
                        feas = via_landing_feasible(
                            pdk, gi_low, cx, cy,
                            own_cover_below=own_b,
                            own_cover_above=own_a,
                            foreign_device_rects_by_gi={
                                gi_low: dev_by_gi[gi_low],
                                gi_low + 1: dev_by_gi[gi_low + 1],
                            },
                            foreign_routed_rects_by_gi=None,
                            min_space_nm_by_gi=ms_by_gi,
                            min_width_nm_by_gi=None,
                            cut_size_nm=int(vsz),
                            enclosure_below_nm=int(rb),
                            enclosure_above_nm=int(ra),
                        )
                        if not feas.ok and feas.reason == "device-pad-intrusion":
                            infeasible.add((ix, iy))
            if infeasible:
                out[gi_low] = infeasible
        return out

    @staticmethod
    def _via_keepout_offsets(
        pitch: int,
        pad_half: int,
        wire_half: int,
        ms: int,
        *,
        exact_diagonal: bool = True,
    ) -> List[Tuple[int, int]]:
        """Relative ``(dx, dy)`` track offsets within foreign-facing reach of a
        via landing pad.

        Orthogonal offsets keep the full symmetric radius ``r = ceil((pad_half +
        ms/2)/pitch)``. When ``exact_diagonal=True``, diagonal offsets (``dx != 0
        and dy != 0``) are filtered by exact 2D corner-to-corner Euclidean gap
        against a worst-case conductor half-width ``max(pad_half, ms/2)`` so
        DRC-clean diagonal cells are not marked contended.
        """
        if pad_half <= wire_half:
            return []
        r = max(0, int(math.ceil((pad_half + ms / 2) / pitch)))
        f_half = max(float(pad_half), ms / 2.0)
        offsets: List[Tuple[int, int]] = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if exact_diagonal and dx != 0 and dy != 0:
                    gx = max(0.0, abs(dx) * pitch - pad_half - f_half)
                    gy = max(0.0, abs(dy) * pitch - pad_half - f_half)
                    if math.hypot(gx, gy) >= ms:
                        continue
                offsets.append((dx, dy))
        return offsets

    def _via_pad_keepout_sources(
        self,
        grid,
        tech: _Tech,
        wire_cells: Set[Cell],
        width_at=None,
        *,
        exact_diagonal: bool = True,
    ) -> Dict[Cell, List[Cell]]:
        """Return ``{keepout_cell: [(vx, vy, gi_low), ...]}`` linking every cell
        covered by an emergent via landing pad's min-spacing reach back to the
        originating via transition cell(s)."""
        pitch = grid.pitch
        by_xy: Dict[Tuple[int, int], Set[int]] = {}
        for (ix, iy, gi) in wire_cells:
            by_xy.setdefault((ix, iy), set()).add(gi)
        sources: Dict[Cell, List[Cell]] = {}
        for (ix, iy), gis in by_xy.items():
            for gl in gis:
                if (gl + 1) not in gis:
                    continue
                rb, ra = self._via_env_rule_nm(tech, gl)
                vsz = _via_cut_size_nm(tech, gl)
                vcell = (ix, iy, gl)
                for (g, enc) in ((gl, rb), (gl + 1, ra)):
                    if enc <= 0:
                        continue
                    pad_half = vsz // 2 + int(enc)
                    wire_half = (
                        int(width_at(ix, iy, g)) // 2
                        if width_at is not None else 0
                    )
                    ms = tech.min_space_nm[tech._gi_to_pdk[g]]
                    for dx, dy in self._via_keepout_offsets(
                        pitch, pad_half, wire_half, ms, exact_diagonal=exact_diagonal
                    ):
                        jx, jy = ix + dx, iy + dy
                        if grid.in_bounds(jx, jy):
                            sources.setdefault((jx, jy, g), []).append(vcell)
        return sources

    def _via_pad_keepout_cells(
        self,
        grid,
        tech: _Tech,
        wire_cells: Set[Cell],
        width_at=None,
        *,
        exact_diagonal: bool = False,
    ) -> Set[Cell]:
        """The FOREIGN-FACING tracks a net's EMERGENT via landing pads must keep
        clear of OTHER nets' same-layer metal -- the negotiated-congestion face of
        the metal min-spacing obligation for via pads."""
        return set(
            self._via_pad_keepout_sources(
                grid, tech, wire_cells, width_at, exact_diagonal=exact_diagonal
            )
        )

    def _make_via_cost(
        self,
        *,
        grid,
        tech: _Tech,
        net: str,
        width_at,
        occupancy: Dict[Cell, int],
        history: Dict[Cell, float],
        via_pad_history: Dict[Cell, float],
        pin_owners: Dict[Cell, Set[str]],
        routed_nets: Dict[str, Set[Cell]],
        pres_fac: float,
    ):
        """Return a per-via transition penalty fn(ix, iy, gl) -> float pricing
        pad-keepout overlap onto foreign routed metal or unrouted foreign pins,
        plus accumulated history on the via transition cell itself."""
        stencils: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]] = {}

        def _stencil(gl: int, wh_lo: int, wh_hi: int):
            key = (gl, wh_lo, wh_hi)
            offs = stencils.get(key)
            if offs is not None:
                return offs
            rb, ra = self._via_env_rule_nm(tech, gl)
            vsz = _via_cut_size_nm(tech, gl)
            items: List[Tuple[int, int, int]] = []
            for g, enc, wh in ((gl, rb, wh_lo), (gl + 1, ra, wh_hi)):
                if enc <= 0:
                    continue
                ph = vsz // 2 + int(enc)
                ms = tech.min_space_nm[tech._gi_to_pdk[g]]
                for dx, dy in self._via_keepout_offsets(
                    grid.pitch, ph, wh, ms, exact_diagonal=True
                ):
                    items.append((dx, dy, g))
            offs = tuple(items)
            stencils[key] = offs
            return offs

        def via_cost(ix: int, iy: int, gl: int) -> float:
            w_lo_h = int(width_at(ix, iy, gl)) // 2 if width_at is not None else 0
            w_hi_h = int(width_at(ix, iy, gl + 1)) // 2 if width_at is not None else 0
            cost = 0.0
            vh = via_pad_history.get((ix, iy, gl), 0.0)
            if vh > 0.0:
                cost += vh * self.hist_fac * (1.0 + pres_fac)
            for dx, dy, g in _stencil(gl, w_lo_h, w_hi_h):
                jx, jy = ix + dx, iy + dy
                if not grid.in_bounds(jx, jy):
                    continue
                c = (jx, jy, g)
                f_occ = occupancy.get(c, 0)
                if f_occ == 0:
                    po = pin_owners.get(c)
                    if po and any(o != net and o not in routed_nets for o in po):
                        f_occ = 1
                if f_occ > 0:
                    base = 1.0 + history.get(c, 0.0) * self.hist_fac
                    cost += base * (f_occ * pres_fac)
            return cost

        return via_cost

    def _route_one_net_soft(
        self,
        *,
        grid,
        tech: _Tech,
        pin_cells: List[Cell],
        layer_gis: List[int],
        is_blocked,
        cell_cost,
        dir_pref: Optional[Dict[int, Tuple[str, float]]] = None,
        seed_cells: Optional[Set[Cell]] = None,
        via_blocked=None,
        via_cost=None,
    ) -> Tuple[Set[Cell], bool]:
        """Route ONE net's pins into a single connected tree with the soft
        congestion cost (no hard cross-net blockage). Sink-by-sink multi-source A*
        seeded from the growing tree -- the same connectivity construction as the
        base ``_route_net``, but with ``cell_cost`` instead of a hard ``committed``
        wall. ``dir_pref`` is the per-layer Direction soft cost, forwarded
        verbatim to the shared ``_astar`` (None -> byte-identical cost).

        ``seed_cells`` (preroute, contract 6a) replaces the first-pin seeding:
        the tree STARTS as the preroute cells and EVERY pin routes onto it (so
        no pin-to-pin tree can leave the trunk floating); the returned
        ``wire_cells`` hold only the drawn branches -- the preroute itself is
        emitted verbatim by the caller, never re-painted cell by cell.
        Returns ``(wire_cells, connected)``; on an unreachable pin returns
        ``(set(), False)`` -- NO fabricated wire.

        Each sink is routed by a (STEP 2) widen-on-failure A* ladder: a bounded
        per-net window that DOUBLES the slack on each miss and finishes with a FINAL
        UNBOUNDED search, so a sink is reported unconnected only when it is truly
        unroutable (honesty preserved)."""
        if seed_cells:
            tree: Set[Cell] = set(seed_cells)
            wire_cells: Set[Cell] = set()
            targets = list(pin_cells)
        else:
            tree = set()
            tree.add(pin_cells[0])
            wire_cells = set(tree)
            targets = pin_cells[1:]

        # tree memory for the Direction comb contract (see engine_maze
        # ``_DIR_RUN_FREE``): per-cell anti-command extent of the net's
        # existing same-layer metal. None without an explicit Direction ->
        # byte-identical path.
        dir_runs = None
        if dir_pref:
            dir_runs = {}
            self._update_dir_runs(dir_runs, tree, tree, dir_pref)

        def _sink_cell_cost(ix: int, iy: int, gi: int) -> float:
            base = float(cell_cost(ix, iy, gi)) if cell_cost is not None else 0.0
            if via_cost is None or (ix, iy, gi) in wire_cells:
                return base
            extra = 0.0
            if gi > 0 and (ix, iy, gi - 1) in wire_cells:
                extra += max(0.0, float(via_cost(ix, iy, gi - 1)))
            if gi + 1 < tech.n_layers and (ix, iy, gi + 1) in wire_cells:
                extra += max(0.0, float(via_cost(ix, iy, gi)))
            return base + extra

        eff_cell_cost = _sink_cell_cost if via_cost is not None else cell_cost

        for target in targets:
            if target in tree:
                if seed_cells:
                    # the pin sits ON the trunk: emit its pad cell so the
                    # terminal landing is drawn (same-net metal over the rect).
                    wire_cells.add(target)
                continue

            # STEP 2: widen-on-failure A* ladder. Start bounded to the seeds+target
            # window (max of +8 tracks or +25% of the box span), DOUBLE the slack on
            # each miss, then a FINAL UNBOUNDED search -> ``None`` only when truly
            # unroutable (honesty preserved). Deterministic: fixed ladder, no clock.
            path = None
            span = self._sink_box_span(tree, target)
            slack0 = max(8, span // 4)
            for slack in (slack0, slack0 * 2, slack0 * 4, None):
                path = self._astar(
                    grid=grid,
                    tech=tech,
                    sources=tree,
                    target=target,
                    layer_gis=layer_gis,
                    is_blocked=is_blocked,
                    cell_cost=eff_cell_cost,
                    dir_pref=dir_pref,
                    dir_runs=dir_runs,
                    via_blocked=via_blocked,
                    via_cost=via_cost,
                    window_slack=slack,
                )
                if path is not None:
                    break

            if path is None:
                return set(), False
            for c in path:
                tree.add(c)
                wire_cells.add(c)
            if dir_runs is not None:
                self._update_dir_runs(dir_runs, tree, path, dir_pref)
        return wire_cells, True

    # ---------------------------------------------------------- STEP 2 helpers
    @staticmethod
    def _sink_box_span(tree: Set[Cell], target: Cell) -> int:
        """Larger side (in tracks) of the (ix,iy) bounding box of ``tree`` +
        ``target`` -- the base the per-net A* window slack scales off (STEP 2)."""
        tx, ty, _tg = target
        min_x = max_x = tx
        min_y = max_y = ty
        for (ix, iy, _gi) in tree:
            if ix < min_x:
                min_x = ix
            elif ix > max_x:
                max_x = ix
            if iy < min_y:
                min_y = iy
            elif iy > max_y:
                max_y = iy
        return max(max_x - min_x, max_y - min_y)

    @staticmethod
    def _pin_is_trapped(
        grid,
        ix: int,
        iy: int,
        gi: int,
        lg: Sequence[int],
        blocked: Dict[int, Set[Tuple[int, int]]],
        via_infeasible: Dict[int, Set[Tuple[int, int]]],
    ) -> bool:
        """True if snapped pin cell ``(ix, iy, gi)`` sits in a dead-end pocket."""
        if (ix, iy) not in blocked.get(gi, set()) and (ix, iy) not in via_infeasible.get(gi, set()):
            return False
        lg_set = set(lg)
        max_pocket = 64
        start = (ix, iy, gi)
        seen: Set[Cell] = {start}
        q = deque([start])
        while q:
            cx, cy, cg = q.popleft()
            if cg != gi:
                return False
            if len(seen) > max_pocket:
                return False
            if cx == 0 or cy == 0 or cx == grid.nx - 1 or cy == grid.ny - 1:
                if (cx, cy) not in blocked.get(cg, set()):
                    return False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if not grid.in_bounds(nx, ny):
                    continue
                if (nx, ny) not in blocked.get(cg, set()):
                    nxt = (nx, ny, cg)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            for gl, dg in ((cg, 1), (cg - 1, -1)):
                ng = cg + dg
                if ng not in lg_set:
                    continue
                if (cx, cy) in via_infeasible.get(gl, set()):
                    continue
                if (cx, cy) in blocked.get(ng, set()):
                    continue
                nxt = (cx, cy, ng)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return True

    def _escape_sandwiched_pins(
        self,
        *,
        grid,
        tech: _Tech,
        pdk,
        pins_by_net: Dict[str, List[Pin]],
        net_layer_gis: Dict[str, List[int]],
        net_pin_cells: Dict[str, List[Cell]],
        net_obstacle_blocked: Dict[str, Dict[int, Set[Tuple[int, int]]]],
        net_via_infeasible: Dict[str, Dict[int, Set[Tuple[int, int]]]],
        invalid_layer_access_nets: Set[str],
        layer_set_by_net: Dict[str, List[int]],
        pre_cells: Dict[str, Set[Cell]],
        obs_by_gi: Dict[int, List[Tuple[float, float, float, float, Optional[str]]]],
        via_obs_by_gi: Dict[int, List[Tuple[float, float, float, float, Optional[str], Optional[str]]]],
        mim_active_by_net: Dict[str, Dict[int, List[Tuple[int, int, int, int, int]]]],
        mim_forbidden_cut_by_gi: Dict[int, Sequence[Tuple[int, int, int, int]]],
        envelope_bbox: Optional[Tuple[int, int, int, int]],
        native_grid=None,
        net_to_id: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[LayerPolygon]]:
        """Promote pins trapped on a snapped metal layer via an exact-pin-center
        ``M_k -> V_k -> M_{k+1}`` riser when foreign clearance holds at ``(px, py)``.
        """
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

        mfg_grid = max(1, int(pdk_manufacturing_grid_nm(tech.pdk)))
        contract = getattr(tech, "pdk", pdk)

        def _ceil_g(val: int) -> int:
            return ((int(val) + mfg_grid - 1) // mfg_grid) * mfg_grid

        def _floor_g(val: int) -> int:
            return (int(val) // mfg_grid) * mfg_grid

        def _box_in_env(box: Tuple[int, int, int, int]) -> bool:
            if envelope_bbox is None:
                return True
            ex0, ey0, ex1, ey1 = envelope_bbox
            return box[0] >= ex0 and box[1] >= ey0 and box[2] <= ex1 and box[3] <= ey1

        def _obs_clear(
            box: Tuple[int, int, int, int],
            rects: Sequence[Tuple[float, float, float, float, Optional[str]]],
            clr: int,
            own_net: str,
        ) -> bool:
            bx0, by0, bx1, by1 = box
            for x0, y0, x1, y1, owner in rects:
                if owner is not None and owner == own_net:
                    continue
                ox0 = int(round(min(x0, x1) * _NM_PER_UM))
                oy0 = int(round(min(y0, y1) * _NM_PER_UM))
                ox1 = int(round(max(x0, x1) * _NM_PER_UM))
                oy1 = int(round(max(y0, y1) * _NM_PER_UM))
                if bx1 + clr > ox0 and ox1 + clr > bx0 and by1 + clr > oy0 and oy1 + clr > by0:
                    return False
            return True

        def _mim_clear(
            box: Tuple[int, int, int, int],
            entries: Sequence[Tuple[int, int, int, int, int]],
        ) -> bool:
            bx0, by0, bx1, by1 = box
            for ox0, oy0, ox1, oy1, clr in entries:
                if bx1 + clr > ox0 and ox1 + clr > bx0 and by1 + clr > oy0 and oy1 + clr > by0:
                    return False
            return True

        def _via_clear(
            box: Tuple[int, int, int, int],
            vrects: Sequence[Tuple[float, float, float, float, Optional[str], Optional[str]]],
            vsp: int,
            own_net: str,
            forbidden: Sequence[Tuple[int, int, int, int]],
        ) -> bool:
            bx0, by0, bx1, by1 = box
            for ox0, oy0, ox1, oy1 in forbidden:
                if bx1 > ox0 and ox1 > bx0 and by1 > oy0 and oy1 > by0:
                    return False
            for x0, y0, x1, y1, owner, kind in vrects:
                if kind == "cap_plate_via":
                    continue
                if owner is not None and owner == own_net and kind != "escape_via":
                    continue
                ox0 = int(round(min(x0, x1) * _NM_PER_UM))
                oy0 = int(round(min(y0, y1) * _NM_PER_UM))
                ox1 = int(round(max(x0, x1) * _NM_PER_UM))
                oy1 = int(round(max(y0, y1) * _NM_PER_UM))
                if bx1 + vsp > ox0 and ox1 + vsp > bx0 and by1 + vsp > oy0 and oy1 + vsp > by0:
                    return False
            return True

        pin_cell_owners: Dict[Cell, Set[str]] = {}
        for _n, _pcs in net_pin_cells.items():
            if _n in invalid_layer_access_nets:
                continue
            if len(pins_by_net.get(_n, ())) <= 1 and not pre_cells.get(_n):
                continue
            for _c in _pcs:
                pin_cell_owners.setdefault(_c, set()).add(_n)

        risers_by_net: Dict[str, List[LayerPolygon]] = {}
        for net in sorted(pins_by_net):
            if net in invalid_layer_access_nets:
                continue
            pins = pins_by_net[net]
            if len(pins) <= 1 and not pre_cells.get(net):
                continue
            lg = net_layer_gis[net]
            blocked = net_obstacle_blocked.get(net) or {}
            via_inf = net_via_infeasible.get(net) or {}
            pcells = net_pin_cells[net]
            nid = net_to_id.get(net, -1) if (native_grid is not None and net_to_id is not None) else -1
            for idx, (p, pcell) in enumerate(zip(pins, pcells)):
                pgi = tech.resolve_layer_to_gi(p.layer)
                if pgi is None:
                    continue
                ix, iy, gi = pcell
                if gi != pgi or (gi + 1) not in lg:
                    continue
                vkey = tech.via_key_for(gi)
                if vkey is None:
                    continue
                trapped = (
                    bool(native_grid.pin_is_trapped(nid, ix, iy, gi, lg))
                    if (native_grid is not None and nid >= 0)
                    else self._pin_is_trapped(grid, ix, iy, gi, lg, blocked, via_inf)
                )
                colliding = len(pin_cell_owners.get(pcell, ())) >= 2
                if not trapped and not colliding:
                    continue
                if colliding and not trapped:
                    # Only promote a colliding pin when it sits on an owned physical
                    # conductor pad (never promote synthetic obstacle-free overlapping pins).
                    px_raw = int(round(float(p.xy_um[0]) * _NM_PER_UM))
                    py_raw = int(round(float(p.xy_um[1]) * _NM_PER_UM))
                    has_owned_pad = any(
                        own is not None and own == net
                        and int(round(min(x0, x1) * _NM_PER_UM)) - 1 <= px_raw <= int(round(max(x0, x1) * _NM_PER_UM)) + 1
                        and int(round(min(y0, y1) * _NM_PER_UM)) - 1 <= py_raw <= int(round(max(y0, y1) * _NM_PER_UM)) + 1
                        for (x0, y0, x1, y1, own) in obs_by_gi.get(gi, ())
                    )
                    if not has_owned_pad:
                        continue
                vsz = _via_cut_size_nm(tech, gi)
                if not vsz:
                    continue
                cut_h = max(1, (int(vsz) + 1) // 2)
                rb, ra = self._via_env_rule_nm(tech, gi)
                mw_lo = int(tech.gi_min_width(gi))
                mw_hi = int(tech.gi_min_width(gi + 1))
                ms_lo = int(tech.min_space_nm[tech._gi_to_pdk[gi]])
                ms_hi = int(tech.min_space_nm[tech._gi_to_pdk[gi + 1]])
                vsp = _via_cut_space_nm(contract, _via_name_for(tech, gi), ms_lo)

                px = int(round(float(p.xy_um[0]) * _NM_PER_UM / mfg_grid)) * mfg_grid
                py = int(round(float(p.xy_um[1]) * _NM_PER_UM / mfg_grid)) * mfg_grid
                lo_h = _ceil_g(max((mw_lo + 1) // 2, cut_h + max(0, int(rb))))
                ma_hi = 0
                min_area_fn = getattr(contract, "min_area", None)
                if callable(min_area_fn):
                    try:
                        ma_hi = int(min_area_fn(f"metal{tech._gi_to_pdk[gi + 1]}") or 0)
                    except Exception:
                        ma_hi = 0
                hi_area_h = _ceil_g(int(math.ceil(math.sqrt(ma_hi) / 2.0))) if ma_hi > 0 else 0
                hi_pad_h = _ceil_g(max((mw_hi + 1) // 2, cut_h + max(0, int(ra)), hi_area_h))

                key_lo = tech.gi_to_metal_key(gi)
                key_hi = tech.gi_to_metal_key(gi + 1)
                p_promoted = replace(p, layer=tuple(key_hi))
                hi_cell = self._pin_cell(
                    grid, tech, p_promoted, lg,
                    strict_layers=net in layer_set_by_net,
                    envelope_bbox=envelope_bbox,
                )
                if hi_cell is None or hi_cell[2] != gi + 1:
                    continue
                if colliding and not trapped:
                    hi_owners = pin_cell_owners.get(hi_cell)
                    if hi_owners and any(o != net for o in hi_owners):
                        continue
                ix_hi, iy_hi, _ = hi_cell
                hi_blk = (
                    bool(native_grid.is_blocked_cell(nid, ix_hi, iy_hi, gi + 1))
                    if (native_grid is not None and nid >= 0)
                    else ((ix_hi, iy_hi) in blocked.get(gi + 1, set()))
                )
                if hi_blk:
                    continue

                cx_hi, cy_hi = grid.cell_xy(ix_hi, iy_hi)
                mw_hi_h = _ceil_g((mw_hi + 1) // 2)
                lo_box = (px - lo_h, py - lo_h, px + lo_h, py + lo_h)
                cut_box = (px - cut_h, py - cut_h, px + cut_h, py + cut_h)
                hi_box = (
                    _floor_g(min(px - hi_pad_h, cx_hi - mw_hi_h)),
                    _floor_g(min(py - hi_pad_h, cy_hi - mw_hi_h)),
                    _ceil_g(max(px + hi_pad_h, cx_hi + mw_hi_h)),
                    _ceil_g(max(py + hi_pad_h, cy_hi + mw_hi_h)),
                )
                if not (_box_in_env(lo_box) and _box_in_env(hi_box)):
                    continue
                if not _obs_clear(lo_box, obs_by_gi.get(gi, ()), ms_lo, net):
                    continue
                if not _obs_clear(hi_box, obs_by_gi.get(gi + 1, ()), ms_hi, net):
                    continue
                net_mim = mim_active_by_net.get(net, {})
                if not _mim_clear(lo_box, net_mim.get(gi, ())):
                    continue
                if not _mim_clear(hi_box, net_mim.get(gi + 1, ())):
                    continue
                if not _via_clear(
                    cut_box,
                    via_obs_by_gi.get(gi, ()),
                    int(vsp),
                    net,
                    mim_forbidden_cut_by_gi.get(gi, ()),
                ):
                    continue

                pins[idx] = p_promoted
                pcells[idx] = hi_cell
                if not any(c_o == pcell for c_o in pcells):
                    pin_cell_owners.get(pcell, set()).discard(net)
                pin_cell_owners.setdefault(hi_cell, set()).add(net)
                risers_by_net.setdefault(net, []).extend([
                    LayerPolygon(layer=key_lo[0], datatype=key_lo[1], polygon=Polygon.rectangle(*lo_box)),
                    LayerPolygon(layer=vkey[0], datatype=vkey[1], polygon=Polygon.rectangle(*cut_box)),
                    LayerPolygon(layer=key_hi[0], datatype=key_hi[1], polygon=Polygon.rectangle(*hi_box)),
                ])
                obs_by_gi[gi].append((
                    lo_box[0] / _NM_PER_UM, lo_box[1] / _NM_PER_UM,
                    lo_box[2] / _NM_PER_UM, lo_box[3] / _NM_PER_UM,
                    net,
                ))
                obs_by_gi[gi + 1].append((
                    hi_box[0] / _NM_PER_UM, hi_box[1] / _NM_PER_UM,
                    hi_box[2] / _NM_PER_UM, hi_box[3] / _NM_PER_UM,
                    net,
                ))
                via_obs_by_gi[gi].append((
                    cut_box[0] / _NM_PER_UM, cut_box[1] / _NM_PER_UM,
                    cut_box[2] / _NM_PER_UM, cut_box[3] / _NM_PER_UM,
                    net, "escape_via",
                ))
        return risers_by_net


__all__ = ["PathFinderEngine"]
