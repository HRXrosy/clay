"""MazeRouteEngine -- a faithful multi-layer A*/Dijkstra maze router.

This is ONE strategy behind the :class:`clay.layout.route.engine.RouteEngine`
contract (see that module's docstring). It implements the survey-converged,
MAGICAL-style grid-maze recipe:

  * a routing GRID whose pitch is ``min_width + min_spacing`` resolved from the
    PDK (:class:`clay.layout.tech.LayoutTech`) -- so any on-grid wire is
    same-layer spacing-legal BY CONSTRUCTION ("DRC-by-pitch"); per-layer
    preferred direction is a soft cost, never a hard block;
  * a multi-layer graph over ``(ix, iy, layer)`` -- in-plane neighbours plus VIA
    edges that connect the same ``(ix, iy)`` on adjacent usable metal layers
    (cost = a via penalty); A* with a Manhattan-distance heuristic + a heapq
    priority queue;
  * OBSTACLE-AWARE BY CONSTRUCTION -- every :class:`~clay.layout.route.engine.Obstacle`
    rect is rasterised into BLOCKED grid cells on its layer (plus a spacing+width
    halo), and a route may never enter a blocked cell, so wires never run over
    device artwork on that layer (the whole point: the legacy router ran straight
    over devices);
  * CONNECTIVITY -- every net's pins are joined into ONE component by routing
    sink-by-sink onto the net's growing route tree (each new pin does a
    multi-source search seeded from every cell already on the tree);
  * honest reporting -- a net that cannot reach a pin (boxed in) is left
    ``connected=False`` with NO fabricated wire; unhandled / violated hard
    constraints go into ``RoutedCell.unsatisfied`` and force ``success=False``.

ZERO per-circuit logic: ``route`` is a pure function of
``(pins_by_net, obstacles, constraints, pdk)``. No circuit names, no inline
special cases -- any genuinely new behaviour belongs behind a new RouteEngine
strategy or a new (rare) constraint type, never an inline branch here.

Constraint coverage (P1)
========================
SATISFIED by this engine:
  * :class:`~clay.layout.route.constraints.LayerSet` -- the net's search is
    restricted to the allowed metal layers; if no allowed layer is usable the
    constraint is reported unsatisfied.
  * :class:`~clay.layout.route.constraints.Width` -- the net's wires are drawn at
    ``max(width_um, pdk metal-min)`` and the width halo is reserved as blockage
    for other nets.
  * :class:`~clay.layout.route.constraints.Spacing` -- enforced conservatively by
    inflating the committed blockage of one net of the pair by ``min_um`` so the
    other net's grid search cannot come within that clearance.
  * :class:`~clay.layout.route.constraints.Direction` -- a per-layer preferred-axis
    SOFT cost: an in-plane step against the layer's declared axis pays an extra
    ``weight x base step cost`` (REPLACING the built-in parity prior on that
    layer), and anti-command RUNS beyond a small jog allowance pay an escalated
    per-step cost seeded from the net's existing same-layer extent (the comb
    contract -- see ``_DIR_RUN_FREE``), so accumulated daisy chains migrate to
    a commanded layer instead of crawling against the command. weight=0 (or
    no Direction at all) is byte-identical to the pre-Direction engine; a layer
    outside the routing stack is vacuously satisfied; an invalid axis / negative
    weight is recorded unsatisfied (never silently dropped).
  * :class:`~clay.layout.route.constraints.Mirror` -- the anaroute-style
    symmetric-pair recipe (the axis is DESIGN data -- layout intent / symmetry
    solve -- never a measured reference coordinate): ``net_a`` routes first at
    normal cost; the REFLECTION of its routed cells about ``axis_x_um`` is
    ``net_b``'s candidate, validated cell-by-cell (bounds / obstacles /
    committed metal / pin coverage). Fully legal -> adopted verbatim (the same
    emit machinery, so the pair's geometry is a strict per-rect mirror about
    the snapped axis). Partially illegal -> the reflection becomes a SOFT
    guide (off-mirror cells pay ``_MIRROR_GUIDE_PENALTY``) and net_b
    re-searches. SATISFIED only if the final cell sets ARE exact mirror
    images; anything less keeps the Mirror in ``unsatisfied`` AND quantifies
    the asymmetry in ``RoutedCell.mirror_reports`` (deviation_frac /
    hausdorff_um) -- the engine never claims symmetry it did not draw. An
    invalid Mirror (unknown net, net_a == net_b, a net in two pairs, non-finite
    axis) is recorded unsatisfied and perturbs NOTHING.

RECORDED-UNSATISFIED (honest -- not yet handled by this strategy):
  Shield, Length, TrackOrder, Access, Via. These appear in
  ``RoutedCell.unsatisfied`` (so ``success`` is False) rather than being silently
  ignored. A later strategy (e.g. a symmetric/ILP router) satisfies them.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from clay._native import load_native
from clay.layout.backend import LayerPolygon, Polygon
from clay.layout.route.gridless_types import _via_cut_space_nm
from clay.layout.route.via_model import (
    landing_rect_for_cut,
    subtract_rects as _subtract_rects,
    via_enclosure_nm,
    via_landing_feasible,
)
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
    RouteEngine,
    RoutedCell,
    require_resource_plan_support,
)

LayerKey = Tuple[int, int]
Cell = Tuple[int, int, int]  # (ix, iy, layer_index)

# Constraint kinds this strategy can SATISFY; anything else is recorded as
# unsatisfied (honest P1 reporting). Mandatory connectivity is NOT a constraint
# (it arrives as pins_by_net), so it is not in this set.
_HANDLED_KINDS = frozenset({"layer_set", "width", "spacing", "direction", "mirror"})

# Cost knobs (dimensionless multipliers / penalties on grid-pitch units). These
# tune the SECONDARY objective (short wire, few vias, honour preferred direction)
# inside the don't-care freedom the constraints leave; they never override a hard
# constraint.
_WRONG_DIR_PENALTY = 0.5   # extra cost for a step against a layer's preferred dir
_VIA_PENALTY = 3.0         # extra cost for a layer change (discourage via churn)
# Mirror soft guide: when net_b cannot adopt net_a's reflection verbatim, every
# cell OFF the reflected candidate pays this extra entry cost, so the re-search
# hugs the mirror wherever it is legal. Non-negative -> the Manhattan heuristic
# stays admissible. Inert (never constructed) without a valid Mirror constraint.
_MIRROR_GUIDE_PENALTY = 1.0

# Mirror HARD-PREFER (PathFinder by-construction consumer): the reserved
# reflection corridor is a hard keep-out for every other net, so it is FREE for
# net_b; this large off-corridor penalty makes net_b take the reserved corridor
# wherever legal and detour off it ONLY around fixed device blockage (exactly
# where an exact mirror is impossible). Non-negative -> heuristic stays
# admissible. Inert without a valid Mirror constraint.
_MIRROR_HARD_PREFER_PENALTY = 50.0

# --- Direction run-extent escalation (the comb contract) ----------------------
# A per-STEP anti-command penalty cannot price a DAISY CHAIN: a row of same-net
# pins joined sink-by-sink with short anti-axis hops, each locally cheaper than
# a via pair, accumulates into a long anti-command comb across SEPARATE A*
# calls (the test4 measured offenders: net10 580um H@M1, power combs ~630um
# H@M1, with M1 commanded V). Axis discipline is a property of RUNS, so the
# Direction contract prices runs: on a commanded layer, anti-command travel
# beyond a small jog allowance pays an escalated per-step cost. The run length
# is CUMULATIVE anti-command travel since the layer was entered (reset only by
# a via -- wiggling cannot dodge it) and it SEEDS from the net's existing
# same-layer connected metal (tree memory: the extent of the component a new
# branch grows from), so the chain's Nth short hop is priced like the long run
# it really extends. Active ONLY under an explicit Direction (dir_pref); the
# no-Direction path is byte-identical.
_DIR_RUN_FREE = 2      # anti-command jog allowance (grid cells) before escalation
_DIR_RUN_PENALTY = 2.5  # extra per-step cost (x weight) beyond the allowance
_CHECKPOINT_INTERVAL = 4096


def _active_checkpoint_cb():
    """Return active ExecutionSession.checkpoint callback, or None."""
    from clay.flow.execution_session import current_session

    session = current_session()
    return session.checkpoint if session is not None else None


def _build_native_grid(grid: "_Grid", tech: "_Tech", pdk, env_rule_fn):
    """Create a C++20 NativePathFinderGrid matching ``(grid, tech, pdk)``, or None."""
    mod = load_native("_pathfinder_cpp")
    if mod is None:
        return None
    if tech.n_layers <= 0 or tech.n_layers > 30:
        return None
    if int(grid.nx) * int(grid.ny) * int(tech.n_layers) * 4 >= (1 << 31) - 1:
        return None
    pref_horiz = [bool(tech.preferred_horizontal(gi)) for gi in range(tech.n_layers)]
    min_width = [int(tech.gi_min_width(gi)) for gi in range(tech.n_layers)]
    min_space = [int(tech.min_space_nm[tech._gi_to_pdk[gi]]) for gi in range(tech.n_layers)]
    via_cut: List[int] = []
    via_eb: List[int] = []
    via_ea: List[int] = []
    via_sp: List[int] = []
    contract = getattr(tech, "pdk", pdk)
    for gl in range(max(0, tech.n_layers - 1)):
        try:
            vsz = int(_via_cut_size_nm(tech, gl) or 0)
        except Exception:
            vsz = 0
        try:
            rb, ra = env_rule_fn(tech, gl)
        except Exception:
            rb, ra = 0, 0
        fallback_sp = int(tech.min_space_nm[tech._gi_to_pdk[gl]])
        vsp = int(_via_cut_space_nm(contract, _via_name_for(tech, gl), fallback_sp) or 0)
        via_cut.append(vsz)
        via_eb.append(int(rb or 0))
        via_ea.append(int(ra or 0))
        via_sp.append(vsp)
    return mod.NativePathFinderGrid(
        int(grid.nx),
        int(grid.ny),
        int(tech.n_layers),
        int(grid.x0),
        int(grid.y0),
        int(grid.pitch),
        pref_horiz,
        min_width,
        min_space,
        via_cut,
        via_eb,
        via_ea,
        via_sp,
    )


class _Tech:
    """A thin read-through view of a :class:`~clay.layout.tech.LayoutTech` that
    pre-resolves the routing facts the maze needs -- the authoritative contiguous
    router-layer prefix (GDS key, min-width/spacing, and declared preferred axis),
    the wider fixed-conductor stack needed for terminal access, the global grid
    pitch, and the via keys. Everything is read from the PDK; nothing is hardcoded."""

    def __init__(self, pdk) -> None:
        self.pdk = pdk
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

        self.manufacturing_grid_nm = pdk_manufacturing_grid_nm(pdk)
        names = set(pdk.layer_names())

        def metal_name(n: int) -> Optional[str]:
            for cand in (f"metal{n}", f"met{n}", f"m{n}"):
                if cand in names:
                    return cand
            return None

        # Discover usable metals: a key exists AND width+spacing rules resolve.
        # (Some PDKs expose a metal-key for a top metal that carries no parsed
        # width/spacing rule -- gf180 metal4/5; such a layer has no well-defined
        # pitch so we cannot route on it. Honest: skip it.)
        self.metal_keys: Dict[int, LayerKey] = {}
        self.metal_names: Dict[int, str] = {}
        self.min_width_nm: Dict[int, int] = {}
        self.min_space_nm: Dict[int, int] = {}
        n = 1
        while True:
            try:
                key = pdk.metal_key(n)
            except Exception:
                break
            nm = metal_name(n)
            w = self._safe(pdk.min_width, nm) if nm else None
            s = self._safe(pdk.min_space, nm) if nm else None
            if nm is not None and w is not None and s is not None:
                self.metal_keys[n] = key
                self.metal_names[n] = nm
                self.min_width_nm[n] = int(w)
                self.min_space_nm[n] = int(s)
            n += 1
            if n > 32:  # defensive bound
                break

        if not self.metal_keys:
            raise ValueError(
                f"{getattr(pdk, 'pdk_id', pdk)!r} exposes no routable metal layer "
                "(needs a metal key + min-width + min-spacing)"
            )

        # One authoritative router stack is shared with explicit resource-world
        # enumeration.  A layermap conductor is not thereby a router layer: every
        # graph layer must belong to the contiguous prefix carrying a declared PDK
        # preferred axis.  Conductors above that prefix remain in ``metal_keys`` so
        # fixed device/MIM artwork can still be resolved and reached by an emitted
        # via ladder, but negotiated routing cannot allocate planar transport on
        # them.  The shared resolver also rejects an axis hole below a higher
        # declared router layer instead of guessing an alternating direction.
        from clay.layout.route.resource_world import _routing_layer_facts

        routing_facts = _routing_layer_facts(
            pdk, manufacturing_grid_nm=self.manufacturing_grid_nm
        )
        self.metal_order: List[int] = [
            int(fact.layer_n) for fact in routing_facts
        ]
        self._preferred_axis_by_pdk: Dict[int, str] = {
            int(fact.layer_n): str(fact.preferred_axis)
            for fact in routing_facts
        }
        for fact in routing_facts:
            layer_n = int(fact.layer_n)
            if (
                layer_n not in self.metal_keys
                or tuple(self.metal_keys[layer_n]) != tuple(fact.key)
            ):
                raise ValueError(
                    f"authoritative routing layer metal{layer_n} is absent from "
                    "the router conductor stack"
                )
        self.n_layers: int = len(self.metal_order)
        # graph layer-index (0..n_layers-1) <-> PDK metal index (1-based)
        self._gi_to_pdk: List[int] = list(self.metal_order)
        self._pdk_to_gi: Dict[int, int] = {p: i for i, p in enumerate(self.metal_order)}

        # Global grid pitch = the MAX over usable layers of (min_width+min_space):
        # one uniform integer grid so a via connects coincident (x,y) on adjacent
        # layers, and an on-grid wire clears the same-layer min-spacing on EVERY
        # layer (conservative => DRC-by-pitch on all layers at once).
        self.pitch_nm: int = max(
            self.min_width_nm[p] + self.min_space_nm[p] for p in self.metal_order
        )

        # Via keys between graph-adjacent layers (gi -> via key for gi<->gi+1). The
        # PDK via index is the LOWER metal's PDK index.
        self.via_keys: Dict[int, Optional[LayerKey]] = {}
        for gi in range(self.n_layers - 1):
            lower_pdk = self._gi_to_pdk[gi]
            try:
                self.via_keys[gi] = pdk.via_key(lower_pdk)
            except Exception:
                self.via_keys[gi] = None

    @staticmethod
    def _safe(fn, arg):
        try:
            return fn(arg)
        except Exception:
            return None

    # ---- layer-index <-> PDK metal helpers ----
    def gi_to_metal_key(self, gi: int) -> LayerKey:
        return self.metal_keys[self._gi_to_pdk[gi]]

    def resolve_layer_to_pdk(self, layer) -> Optional[int]:
        """PDK metal index (1-based) for a GDS key, over ALL discovered metals --
        including thick/top metals OUTSIDE the fine routing graph."""
        lk = tuple(layer)
        for pidx, key in self.metal_keys.items():
            if tuple(key) == lk:
                return pidx
        return None

    def pin_seed(
        self, layer, layer_gis: List[int], *, strict_layers: bool = False,
    ) -> Optional[Tuple[int, List[int]]]:
        """Seed graph layer for a pin + the VIA LADDER levels to its true layer.

        In-graph pin -> ``(gi, [])``. A pin on a discovered metal ABOVE the fine
        graph (thick/top metal, e.g. the external_pdk M5/M6 MIM plates) seeds at the TOP
        allowed graph layer and returns the PDK via LEVELS ``top_pdk..pin_pdk-1``
        the emit must ladder through -- the old silent reseed-at-layer-0 buried
        such pins under the very keep-outs protecting the plates (length=0 nets)
        and never drew the via stack the plate connection physically needs.
        A known in-graph pin outside ``layer_gis`` is unusable: silently moving
        the terminal to ``layer_gis[0]`` would fabricate electrical access.  An
        out-of-graph upper-metal ladder remains available only without an explicit
        LayerSet (``strict_layers=False``). Unknown layers are also unusable."""
        gi = self.resolve_layer_to_gi(layer)
        if gi is not None:
            return (gi, []) if gi in layer_gis else None
        pdk = self.resolve_layer_to_pdk(layer)
        if pdk is not None and layer_gis and not strict_layers:
            top_gi = max(layer_gis)
            top_pdk = self._gi_to_pdk[top_gi]
            if pdk > top_pdk:
                return top_gi, list(range(top_pdk, pdk))
        return None

    def gi_min_width(self, gi: int) -> int:
        return self.min_width_nm[self._gi_to_pdk[gi]]

    def via_key_for(self, gi_low: int) -> Optional[LayerKey]:
        return self.via_keys.get(gi_low)

    def preferred_horizontal(self, gi: int) -> bool:
        """PDK-declared preferred direction for an authoritative graph layer."""

        return self._preferred_axis_by_pdk[self._gi_to_pdk[gi]] == "H"

    def resolve_layer_to_gi(self, layer) -> Optional[int]:
        """Resolve a contract ``Layer`` (a GDS ``(layer, datatype)`` tuple OR a
        role name like ``'M1'`` / ``'metal1'`` / ``'met1'`` / ``'m1'``) to a graph
        layer-index, or ``None`` if it is not a usable routing layer."""
        # GDS key?
        if isinstance(layer, tuple) and len(layer) == 2 and all(
            isinstance(v, int) for v in layer
        ):
            for gi in range(self.n_layers):
                if self.gi_to_metal_key(gi) == tuple(layer):
                    return gi
            return None
        # role name?
        if isinstance(layer, str):
            s = layer.strip().lower()
            pdk_idx: Optional[int] = None
            for prefix in ("metal", "met", "m"):
                if s.startswith(prefix):
                    tail = s[len(prefix):]
                    if tail.isdigit():
                        pdk_idx = int(tail)
                        break
            if pdk_idx is None and s.isdigit():
                pdk_idx = int(s)
            if pdk_idx is not None:
                return self._pdk_to_gi.get(pdk_idx)
        return None


# The engine's PUBLIC coordinate currency (Pin.xy_um, Obstacle.x0..y1) is MICRONS
# -- consistent with the rest of the Python layout system (device geometry is um
# floats). The routing grid works in nm (PDK pitch is nm, wire widths are nm) and
# emits nm-int LayerPolygon (= GDS dbu). This factor converts at the single um->nm
# boundary (snap_index / grid bounds / obstacle raster); cell_xy stays nm for emit.
_NM_PER_UM = 1000.0


class _Grid:
    """The integer routing grid: a uniform pitch in x/y, an origin, and the
    cell<->coordinate mapping. Coordinates are PDK nm (ints); the public um inputs
    are converted to nm at the boundary (see _NM_PER_UM)."""

    def __init__(self, tech: _Tech, x0: int, y0: int, nx: int, ny: int) -> None:
        self.tech = tech
        self.pitch = tech.pitch_nm
        self.x0 = x0
        self.y0 = y0
        self.nx = nx
        self.ny = ny

    def cell_xy(self, ix: int, iy: int) -> Tuple[int, int]:
        return (self.x0 + ix * self.pitch, self.y0 + iy * self.pitch)

    def snap_index(self, x_um: float, y_um: float) -> Tuple[int, int]:
        # PUBLIC coords are MICRONS; the grid is nm. The um->nm conversion lives at
        # this single boundary -- callers pass um (Pin.xy_um), the grid stays nm.
        x = x_um * _NM_PER_UM
        y = y_um * _NM_PER_UM
        ix = int(round((x - self.x0) / self.pitch))
        iy = int(round((y - self.y0) / self.pitch))
        ix = max(0, min(self.nx - 1, ix))
        iy = max(0, min(self.ny - 1, iy))
        return (ix, iy)

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny


class MazeRouteEngine(RouteEngine):
    """Multi-layer A*/Dijkstra maze router (a :class:`RouteEngine` strategy).

    See the module docstring for the algorithm and constraint coverage. The
    public surface is the contract method :meth:`route`.

    Parameters
    ----------
    margin_tracks : int
        Extra grid tracks added around the pin/obstacle bounding box so a route
        may detour outside the tight hull (default 8).
    max_expansions : int
        Per-search node-expansion budget (bounded effort). A search that exhausts
        it returns no path -> the net is honestly reported ``connected=False``.
    taper : bool
        OPT-IN wire tapering (default False -> current behaviour). When True, each
        net's wire is necked to the PDK min-width within a neck radius of any of
        ITS OWN pins and runs at the full requested trunk width elsewhere. The
        per-cell width drives BOTH the inter-net commit halo (a necked cell
        reserves a small halo, so a different net's close terminal is not
        over-blocked) AND the emitted geometry. Device-obstacle clearance stays
        UNIFORM (trunk-based) -- only the inter-net halo + emit taper neck.
    max_reorder : int
        OPT-IN re-route rounds (default 0 -> current single-pass behaviour). When
        > 0, after the first difficulty-order pass, if any net is unconnected the
        unconnected nets are moved to the FRONT of the order (keeping their
        relative difficulty order) and EVERY net is re-routed from scratch; this
        repeats up to ``max_reorder`` rounds or until all nets connect. The BEST
        round seen (most-connected, then fewest-shorts, then fewest-unsatisfied)
        is emitted. Fully deterministic. HONEST: if it never connects all nets it
        emits the best round with ``success=False`` and the real shorts/unconnected.
    """

    def satisfies(self) -> frozenset:
        """The constraint kinds this maze strategy MATERIALIZES: exactly
        ``_HANDLED_KINDS`` == {layer_set, width, spacing, direction, mirror}
        (LayerSet -> per-net allowed metals; Width -> requested width; Spacing ->
        pairwise clearance; Direction -> soft per-layer bias; Mirror -> reflected
        route). Every other kind (shield/length/track_order/access/via) is
        recorded ``unsatisfied`` when routed here, so a HARD one is caught by the
        route_cell PRE-gate before emit -- this method is that gate's query. A
        subset of ``constraints.CONSTRAINT_KINDS`` by construction."""
        return _HANDLED_KINDS

    def __init__(
        self,
        *,
        margin_tracks: int = 8,
        max_expansions: int = 2_000_000,
        taper: bool = False,
        max_reorder: int = 0,
    ) -> None:
        self.margin_tracks = int(margin_tracks)
        self.max_expansions = int(max_expansions)
        self.taper = bool(taper)
        self.max_reorder = max(0, int(max_reorder))

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
    ) -> RoutedCell:
        require_resource_plan_support(self, resource_plan)
        tech = _Tech(pdk)
        out = RoutedCell()

        # --- preroute (contract point 6): validate + fold into the world -------
        # plan: net -> [(gi, x0, y0, x1, y1)] nm bboxes (raises on an out-of-graph
        # layer -- explicit rejection, never a silent drop). The preroute enters
        # the obstacle sequence as OWNED rects (the device-pad owner semantics):
        # foreign nets hold their width-aware halo from it, the owning net may
        # land on it -- and the grid hull grows to cover it. preroutes=None keeps
        # everything byte-identical (no copies, no extra entries).
        pre_plan = self._preroute_plan(tech, preroutes)
        pre_wires_by_net: Dict[str, list] = {
            n: list(v) for n, v in (preroutes or {}).items() if v
        }
        if pre_plan:
            obstacles = list(obstacles) + self._preroute_obstacles(tech, pre_plan)

        # --- index constraints by net + collect honest-unsatisfied ones --------
        layer_set_by_net: Dict[str, List[int]] = {}   # net -> allowed graph gi list
        layer_set_constraint_by_net: Dict[str, LayerSet] = {}
        width_by_net: Dict[str, float] = {}           # net -> requested width (um)
        spacing_pairs: List[Spacing] = []
        mirror_list: List[Mirror] = []
        # graph layer -> (preferred axis 'H'|'V', weight): the Direction soft cost.
        # LAST Direction per layer wins (explicit overrides the conventional table).
        direction_pref: Dict[int, Tuple[str, float]] = {}
        unsatisfied: List[RouteConstraint] = []

        for c in constraints:
            kind = getattr(c, "kind", None)
            if isinstance(c, LayerSet):
                gis = []
                for lay in c.layers:
                    gi = tech.resolve_layer_to_gi(lay)
                    if gi is not None:
                        gis.append(gi)
                if not gis:
                    # the net is restricted to layers none of which are routable:
                    # cannot satisfy -> honest.
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
            elif kind in _HANDLED_KINDS:  # pragma: no cover - defensive
                pass
            elif isinstance(c, TrunkLayer):
                # This strategy has no trunk-layer cost.  A SOFT advisory is an
                # inert preference; a caller-forced HARD override is unsupported.
                if getattr(c, "hardness", "soft") == "hard":
                    unsatisfied.append(c)
            else:
                # Shield / Length / TrackOrder / Access / Via: not handled
                # by THIS strategy -> recorded, never silently dropped.
                unsatisfied.append(c)

        # valid Mirror pairs keyed by the MIRRORED net (net_b); invalid ones are
        # already recorded unsatisfied (and perturb nothing downstream).
        mirror_by_b = self._index_mirrors(mirror_list, pins_by_net, unsatisfied)

        # A real terminal layer is not relocatable routing freedom.  Reject a net
        # whose pin has no physical seed under its explicit LayerSet instead of
        # inventing an access point on the first allowed metal.
        usable_layers = list(range(tech.n_layers))
        invalid_layer_access_nets: Set[str] = set()
        for net, pins in pins_by_net.items():
            strict = net in layer_set_by_net
            lg = layer_set_by_net.get(net) or usable_layers
            if any(tech.pin_seed(pin.layer, lg, strict_layers=strict) is None
                   for pin in pins):
                invalid_layer_access_nets.add(net)
                constraint = layer_set_constraint_by_net.get(net)
                if constraint is not None and constraint not in unsatisfied:
                    unsatisfied.append(constraint)

        # --- nothing to route? -------------------------------------------------
        # (a preroute-only call still emits its verbatim metal + runs the oracle)
        if not pins_by_net and not pre_plan:
            out.unsatisfied = unsatisfied
            out.success = not unsatisfied
            out.mirror_reports = []
            return out

        # --- build the grid from the pin + obstacle hull -----------------------
        # (obstacles already include the preroute rects -> the hull covers them)
        grid = self._make_grid(tech, pins_by_net, obstacles)
        pre_cells: Dict[str, Set[Cell]] = {
            n: self._preroute_seed_cells(grid, rects)
            for n, rects in pre_plan.items()
        }

        # --- parse obstacles into per-layer rects ------------------------------
        # OBSTACLE-AWARE BY CONSTRUCTION. We keep the raw rects (not pre-rasterised
        # cells) and rasterise them PER NET below with a width-aware halo, so a wide
        # rail / MIM strap reserves its real half-width vs artwork (a fixed
        # min-width halo let wide wires overrun devices by several um).
        obs_by_gi: Dict[int, List[Tuple[float, float, float, float, Optional[str]]]] = {
            gi: [] for gi in range(tech.n_layers)
        }
        for ob in obstacles:
            gi = tech.resolve_layer_to_gi(ob.layer)
            if gi is None:
                continue  # obstacle on a non-routing layer -> irrelevant to maze
            obs_by_gi[gi].append((ob.x0, ob.y0, ob.x1, ob.y1, ob.owner_net))
        # riser-pad clearance: out-of-graph pins' seed-layer ladder pads are wider
        # than the wire width the halo model accounts for -> owned obstacles.
        for (gi, x0, y0, x1, y1, owner) in self._riser_pad_obstacles(
                grid, tech, pins_by_net, layer_set_by_net):
            obs_by_gi[gi].append((x0, y0, x1, y1, owner))

        # net -> the spacing clearances it must hold against OTHER nets (the
        # max requested with each partner). Applied as extra halo when committing.
        spacing_req: Dict[str, Dict[str, float]] = {}
        for sp in spacing_pairs:
            spacing_req.setdefault(sp.net_a, {})[sp.net_b] = max(
                spacing_req.get(sp.net_a, {}).get(sp.net_b, 0.0), sp.min_um
            )
            spacing_req.setdefault(sp.net_b, {})[sp.net_a] = max(
                spacing_req.get(sp.net_b, {}).get(sp.net_a, 0.0), sp.min_um
            )

        # --- route each net ----------------------------------------------------
        # deterministic order: the HARDEST nets first (widest pin bounding box ->
        # longest trunk, least routing freedom), so they claim channel space before
        # the cell congests; ties broken by name. A generic difficulty metric, not a
        # per-circuit rule. (Order-INDEPENDENCE proper is the negotiated-congestion
        # job; this greedy order merely routes more nets before space runs out.)
        def _net_difficulty(n: str) -> float:
            ps = pins_by_net[n]
            if len(ps) < 2:
                return 0.0
            xs = [p.xy_um[0] for p in ps]
            ys = [p.xy_um[1] for p in ps]
            return (max(xs) - min(xs)) + (max(ys) - min(ys))

        ordered_nets = sorted(pins_by_net.keys(), key=lambda n: (-_net_difficulty(n), n))
        # a stable difficulty RANK per net (lower = harder) so the reorder loop can
        # keep nets in their relative difficulty order when it moves a subset.
        difficulty_rank: Dict[str, int] = {n: i for i, n in enumerate(ordered_nets)}

        # the widest wire any net will draw -> each net's commit halo must reserve the
        # PARTNER's half-width too (SYMMETRIC inter-net clearance). Without it, two wide
        # wires of different nets pack closer than min-space (the 731 spacing violations:
        # A reserves A/2+space, but a later wider B's edge lands inside that).
        _usable = list(range(tech.n_layers))
        max_wire_width_nm = 0
        for _n in ordered_nets:
            _lg = sorted(set(layer_set_by_net.get(_n) or _usable))
            max_wire_width_nm = max(
                max_wire_width_nm, self._net_wire_width_nm(tech, _lg, width_by_net.get(_n))
            )

        # the widest via LANDING PAD any net may drop -> the partner term a via-pad
        # halo must reserve for the FOREIGN net facing it. A via pad (cut + 2x
        # enclosure) is WIDER than the widest wire, so two foreign nets' pads packed
        # at (this net's pad_half + min_space + max_WIRE_half) still land their pad
        # EDGES below min_space apart (the gf180 M2.2a cross-net via-pad family,
        # gap=230 vs M2 min_space 280). The pad-half partner term closes that: the
        # clearance becomes symmetric between two PADS, not pad-vs-wire. PDK-sourced
        # (cut size + enclosure), no per-PDK constant; degrades to the wire term when
        # a level declares no enclosure (pad collapses to the cut, narrower).
        max_via_pad_half_nm = 0
        for _gi in range(tech.n_layers - 1):
            _vsz = _via_cut_size_nm(tech, _gi)
            _rb, _ra = self._via_env_rule_nm(tech, _gi)
            for _enc in (_rb, _ra):
                if _enc > 0:
                    max_via_pad_half_nm = max(max_via_pad_half_nm, _vsz // 2 + int(_enc))

        chk = _active_checkpoint_cb()
        if chk is not None:
            chk()
        _custom_maze_search = (
            getattr(type(self), "_astar", None) is not MazeRouteEngine._astar
            or "_astar" in self.__dict__
            or getattr(type(self), "_route_net", None) is not MazeRouteEngine._route_net
            or "_route_net" in self.__dict__
        )
        native_grid = (
            None
            if _custom_maze_search
            else _build_native_grid(grid, tech, pdk, self._via_env_rule_nm)
        )
        net_to_id: Dict[str, int] = {n: i for i, n in enumerate(ordered_nets)}
        maze_pin_cells_by_net: Dict[str, List[Cell]] = {}
        maze_lg_by_net: Dict[str, List[int]] = {}
        maze_w_by_net: Dict[str, int] = {}
        for net in ordered_nets:
            if net in invalid_layer_access_nets:
                continue
            pins = pins_by_net[net]
            if not pins or (len(pins) == 1 and not pre_cells.get(net)):
                continue
            lg = sorted(set(layer_set_by_net.get(net) or _usable))
            maze_lg_by_net[net] = lg
            maze_w_by_net[net] = self._net_wire_width_nm(tech, lg, width_by_net.get(net))
            pcells = []
            for p in pins:
                ix, iy = grid.snap_index(float(p.xy_um[0]), float(p.xy_um[1]))
                seed = tech.pin_seed(
                    p.layer, lg, strict_layers=layer_set_by_net.get(net) is not None
                )
                if seed is not None:
                    pcells.append((ix, iy, seed[0]))
            maze_pin_cells_by_net[net] = pcells
        if native_grid is not None:
            flat_obs = []
            for gi, rects in obs_by_gi.items():
                for (x0, y0, x1, y1, owner) in rects:
                    if owner is None:
                        oid = -1
                    else:
                        oid = net_to_id.get(owner, -2)
                    flat_obs.append((gi, float(x0), float(y0), float(x1), float(y1), oid))
            native_grid.set_static_obstacles(flat_obs, None, None, None, None)
            for net, pcells in maze_pin_cells_by_net.items():
                native_grid.prepare_net(
                    net_to_id[net], maze_w_by_net[net], 0, maze_lg_by_net[net], pcells, None, 0, False, False
                )

        # Hard Fix #2 (Defense 1): disambiguate colliding/foreign-overlapped pin cells
        # onto an unblocked cell strictly inside the pin's own device conductor pad.
        maze_reloc_nets = self._relocate_blocked_owned_pin_cells(
            grid=grid,
            tech=tech,
            pins_by_net=pins_by_net,
            net_pin_cells=maze_pin_cells_by_net,
            net_layer_gis=maze_lg_by_net,
            net_width_nm=maze_w_by_net,
            obs_by_gi=obs_by_gi,
            net_obstacle_blocked=None,
            net_via_infeasible=None,
            envelope_bbox=None,
            native_grid=native_grid,
            net_to_id=net_to_id,
        )
        if native_grid is not None and maze_reloc_nets:
            for net in maze_reloc_nets:
                native_grid.prepare_net(
                    net_to_id[net],
                    maze_w_by_net[net],
                    0,
                    maze_lg_by_net[net],
                    maze_pin_cells_by_net[net],
                    None,
                    0,
                    False,
                    False,
                )

        # --- single-pass router (one deterministic order -> a full routing) -----
        # Factored so the OPT-IN reorder loop can call it repeatedly with a fresh
        # ``committed`` / ``metal_cells`` each round (full rip-up, never carrying
        # last round's blockage). Returns (nets, metal_cells, shorts, routed_ok).
        def _route_pass(order: List[str]):
            committed: Dict[int, Dict[Tuple[int, int], str]] = {
                gi: {} for gi in range(tech.n_layers)
            }
            if native_grid is not None:
                native_grid.reset_committed()
            pass_nets: Dict[str, NetRoute] = {}
            pass_metal: Dict[str, Set[Cell]] = {}
            pass_ok: Dict[str, bool] = {}
            # Mirror pairing: net_a routes BEFORE net_b in every pass (the
            # adjusted order is identical when no valid Mirror exists).
            for net in self._mirror_order(order, mirror_by_b):
                if chk is not None:
                    chk()
                pins = pins_by_net[net]
                allowed = layer_set_by_net.get(net)  # None => all usable layers
                width_um = width_by_net.get(net)
                if net in invalid_layer_access_nets:
                    # The terminal is unusable under this LayerSet, but preroute
                    # metal is caller-owned geometry that already exists. Keep it
                    # verbatim in the failed NetRoute and in the oracle occupancy;
                    # connectivity failure must never erase as-built metal.
                    pass_nets[net] = NetRoute(
                        net=net,
                        pins=list(pins),
                        wires=list(pre_wires_by_net.get(net, ())),
                        connected=False,
                    )
                    pass_metal[net] = set(pre_cells.get(net, ()))
                    pass_ok[net] = False
                    continue
                # the mirrored candidate for a net_b whose partner routed in
                # THIS pass: the reflection of the partner's metal about the
                # snapped axis (None when no Mirror / partner unrouted).
                mirror_cells: Optional[Set[Cell]] = None
                mc = mirror_by_b.get(net)
                if mc is not None and pass_ok.get(mc.net_a) and pass_metal.get(mc.net_a):
                    k_sum = self._mirror_ix_sum(grid, mc.axis_x_um)
                    mirror_cells = {
                        (k_sum - ix, iy, gi)
                        for (ix, iy, gi) in pass_metal[mc.net_a]
                    }
                nr, net_metal = self._route_net(
                    net=net,
                    pins=pins,
                    grid=grid,
                    tech=tech,
                    obstacle_rects=obs_by_gi,
                    committed=committed,
                    allowed_gis=allowed,
                    width_um=width_um,
                    spacing_against=spacing_req.get(net, {}),
                    max_wire_width_nm=max_wire_width_nm,
                    max_via_pad_half_nm=max_via_pad_half_nm,
                    dir_pref=direction_pref or None,
                    seed_cells=pre_cells.get(net),
                    preroute_wires=pre_wires_by_net.get(net),
                    mirror_cells=mirror_cells,
                    native_grid=native_grid,
                    net_id=net_to_id.get(net, -1),
                    checkpoint_cb=chk,
                    pin_cells_override=maze_pin_cells_by_net.get(net) if net in maze_reloc_nets else None,
                )
                pass_nets[net] = nr
                pass_metal[net] = net_metal
                pass_ok[net] = nr.connected
            # preroute-only nets (no pins): existing metal, emitted verbatim,
            # vacuously connected (nothing to join) -- and COUNTED by the short
            # oracle (their cells are real metal a foreign net can collide with).
            for net in sorted(n for n in pre_wires_by_net if n not in pins_by_net):
                nr = NetRoute(net=net, pins=[])
                nr.connected = True
                nr.wires = list(pre_wires_by_net[net])
                pass_nets[net] = nr
                pass_metal[net] = set(pre_cells.get(net, ()))
                pass_ok[net] = True
            shorts = self._detect_shorts(pass_metal)
            return pass_nets, pass_metal, shorts, pass_ok

        # --- reorder loop (OPT-IN; max_reorder=0 -> exactly one pass) -----------
        # Run the difficulty order once. If any net is unconnected and max_reorder
        # is positive, move the unconnected nets to the FRONT (in their relative
        # difficulty order) and re-route every net from scratch; repeat up to
        # max_reorder rounds or until all nets connect. Keep the BEST round seen.
        order = list(ordered_nets)
        best_nets, best_metal, best_shorts, best_ok = _route_pass(order)
        best_unsat_extra = self._spacing_unsatisfied(spacing_pairs, best_ok, best_shorts)
        # rank: most-connected, then fewest-shorts, then fewest-unsatisfied (lower
        # tuple = better). Deterministic; no randomness / time.
        best_score = self._pass_score(best_ok, best_shorts, best_unsat_extra)

        seen_orders: Set[Tuple[str, ...]] = {tuple(order)}
        rounds = 0
        while (
            self.max_reorder > 0
            and rounds < self.max_reorder
            and any(not ok for ok in best_ok.values())
        ):
            # nets unconnected in the BEST round so far -> push to the front,
            # preserving their relative difficulty order; the rest follow (also in
            # difficulty order). Deterministic given best_ok.
            # preroute-only nets carry no pins -> never part of the route order.
            unconnected = sorted(
                (n for n, ok in best_ok.items() if not ok and n in difficulty_rank),
                key=lambda n: difficulty_rank[n],
            )
            rest = sorted(
                (n for n, ok in best_ok.items() if ok and n in difficulty_rank),
                key=lambda n: difficulty_rank[n],
            )
            order = unconnected + rest
            rounds += 1
            sig = tuple(order)
            if sig in seen_orders:
                break  # fixed point -- reordering produced no new permutation
            seen_orders.add(sig)

            nets, metal, shorts, ok = _route_pass(order)
            unsat_extra = self._spacing_unsatisfied(spacing_pairs, ok, shorts)
            score = self._pass_score(ok, shorts, unsat_extra)
            if score < best_score:
                best_score = score
                best_nets, best_metal, best_shorts, best_ok = nets, metal, shorts, ok
                best_unsat_extra = unsat_extra
            if not any(not v for v in best_ok.values()):
                break  # all connected -> done

        out.nets = best_nets
        from clay.layout.route.wire_endpoints import materialize_required_wire_endpoints

        endpoint_failed = materialize_required_wire_endpoints(
            out.nets, pins_by_net, obstacles, tech
        )
        for net in endpoint_failed:
            best_ok[net] = False
        if endpoint_failed:
            best_unsat_extra = self._spacing_unsatisfied(spacing_pairs, best_ok, best_shorts)
        # --- spacing satisfaction bookkeeping ---------------------------------
        # The committed-halo construction guarantees the clearance geometrically;
        # but if either net of a pair failed to route, we cannot CLAIM the pair is
        # satisfied -> record it unsatisfied (and success is already False from the
        # connectivity miss).
        for sp in best_unsat_extra:
            if sp not in unsatisfied:
                unsatisfied.append(sp)

        # --- short detection (same-layer cross-net METAL overlap) --------------
        out.shorts = best_shorts

        # --- Mirror satisfaction: exact reflection or honest unsatisfied -------
        # Judged on the EMITTED pass's metal (best_metal); the per-pair report
        # quantifies any asymmetry -- never a silent claim of symmetry.
        out.mirror_reports = []
        for b_net in sorted(mirror_by_b):
            mc = mirror_by_b[b_net]
            rep = self._mirror_report(
                grid, mc,
                best_metal.get(mc.net_a, set()),
                best_metal.get(b_net, set()),
                bool(best_ok.get(mc.net_a, False)),
                bool(best_ok.get(b_net, False)),
            )
            out.mirror_reports.append(rep)
            if not rep["satisfied"] and mc not in unsatisfied:
                unsatisfied.append(mc)

        out.unsatisfied = unsatisfied
        all_connected = all(nr.connected for nr in out.nets.values())
        out.success = all_connected and not out.shorts and not unsatisfied
        return out

    # ------------------------------------------------------------------ pass scoring
    @staticmethod
    def _spacing_unsatisfied(
        spacing_pairs: List[Spacing],
        routed_ok: Dict[str, bool],
        shorts: List[dict],
    ) -> List[Spacing]:
        """The Spacing constraints a pass could NOT satisfy: either endpoint net
        failed to route (cannot CLAIM the clearance) -> honest unsatisfied. The
        halo construction guarantees the geometric clearance when both routed; a
        present short is surfaced separately by the short oracle."""
        out: List[Spacing] = []
        for sp in spacing_pairs:
            if not (routed_ok.get(sp.net_a, False) and routed_ok.get(sp.net_b, False)):
                out.append(sp)
        return out

    @staticmethod
    def _pass_score(
        routed_ok: Dict[str, bool],
        shorts: List[dict],
        unsat_extra: List[Spacing],
    ) -> Tuple[int, int, int]:
        """Deterministic round-quality key (LOWER is better): (#unconnected,
        #shorts, #unsatisfied). The reorder loop keeps the round with the smallest
        key -- most-connected first, then fewest shorts, then fewest unsatisfied."""
        n_unconnected = sum(1 for ok in routed_ok.values() if not ok)
        return (n_unconnected, len(shorts), len(unsat_extra))

    # ------------------------------------------------------------------ mirror
    # The Mirror-constraint machinery is SHARED: PathFinderEngine inherits every
    # helper below (one implementation, two consuming strategies).
    @staticmethod
    def _index_mirrors(
        mirrors: List[Mirror],
        pins_by_net: Dict[str, List[Pin]],
        unsatisfied: List[RouteConstraint],
    ) -> Dict[str, Mirror]:
        """Validate Mirror constraints -> ``{net_b: Mirror}``. Honest rejection
        (recorded unsatisfied, never silently dropped) for: a net not present in
        ``pins_by_net``, ``net_a == net_b``, a non-finite axis, or a net already
        claimed by an earlier Mirror (first wins, deterministic input order)."""
        by_b: Dict[str, Mirror] = {}
        used: Set[str] = set()
        for mc in mirrors:
            a, b = mc.net_a, mc.net_b
            try:
                axis_ok = math.isfinite(float(mc.axis_x_um))
            except (TypeError, ValueError):
                axis_ok = False
            if (
                a == b
                or a not in pins_by_net
                or b not in pins_by_net
                or not axis_ok
                or a in used
                or b in used
            ):
                unsatisfied.append(mc)
                continue
            used.update((a, b))
            by_b[b] = mc
        return by_b

    @staticmethod
    def _mirror_ix_sum(grid: _Grid, axis_x_um: float) -> int:
        """The reflection in GRID INDEX space: ``ix' = K - ix`` with
        ``K = round(2 * (axis_nm - grid.x0) / pitch)``. The snapped axis is
        ``grid.x0 + K * pitch / 2`` (within half a pitch of the request; the
        snap offset is reported, never hidden)."""
        axis_nm = float(axis_x_um) * _NM_PER_UM
        return int(round(2.0 * (axis_nm - grid.x0) / grid.pitch))

    @staticmethod
    def _mirror_order(
        order: List[str],
        mirror_by_b: Dict[str, Mirror],
        hoist_pairs_first: bool = False,
    ) -> List[str]:
        """The route order with every mirrored net (net_b) moved DIRECTLY after
        its primary (net_a). Identity (a fresh copy) when no valid Mirror exists
        -- the byte-stability bar.

        ``hoist_pairs_first`` (PathFinder by-construction consumer): every
        (net_a, net_b) pair is also hoisted to the FRONT of the order, ahead of
        all unconstrained nets, so net_a's reflection CORRIDOR is reserved
        before congestion from other nets can fill it. Pairs are emitted in the
        relative net_a difficulty order they already had (stable). Default
        ``False`` preserves the maze best-effort consumer's byte-stable order
        (net_b after net_a in place, no hoist)."""
        if not mirror_by_b:
            return list(order)
        a_to_b = {mc.net_a: b for b, mc in mirror_by_b.items()}
        present = set(order)
        out: List[str] = []
        if hoist_pairs_first:
            # pairs first, in net_a's existing relative order; then the rest.
            for n in order:
                if n in a_to_b:
                    out.append(n)
                    b = a_to_b[n]
                    if b in present:
                        out.append(b)
            for n in order:
                if n not in out:
                    out.append(n)
            return out
        for n in order:
            if n in mirror_by_b:
                continue  # emitted right after its primary below
            out.append(n)
            b = a_to_b.get(n)
            if b is not None and b in present:
                out.append(b)
        # defensive: a b whose a is absent from this order (cannot happen for
        # validated pairs, both nets are in pins_by_net) keeps its slot order.
        for n in order:
            if n not in out:
                out.append(n)
        return out

    @staticmethod
    def _mirror_fully_legal(
        grid: _Grid,
        cells: Set[Cell],
        pin_cells: List[Cell],
        layer_gis: List[int],
        is_blocked,
    ) -> bool:
        """Cell-by-cell validation of a reflected candidate: every cell must be
        in bounds, on an allowed layer, and unblocked (the net's own pin cells
        are enterable by the same target exemption the search grants); EVERY pin
        cell must lie ON the candidate (pin coverage -- otherwise the candidate
        cannot connect the net and only soft guidance applies)."""
        allowed = set(layer_gis)
        pinset = set(pin_cells)
        if not all(pc in cells for pc in pin_cells):
            return False
        for (ix, iy, gi) in cells:
            if gi not in allowed or not grid.in_bounds(ix, iy):
                return False
            if (ix, iy, gi) not in pinset and is_blocked(ix, iy, gi):
                return False
        return True

    @classmethod
    def _mirror_report(
        cls,
        grid: _Grid,
        mc: Mirror,
        a_cells: Set[Cell],
        b_cells: Set[Cell],
        connected_a: bool,
        connected_b: bool,
    ) -> dict:
        """The honest per-pair asymmetry account (always emitted for a valid
        Mirror, satisfied or not): cell-set deviation of net_b vs the reflection
        of net_a. ``satisfied`` iff both nets routed AND the sets are EXACT
        mirror images. ``deviation_frac`` = 1 - |B & M| / |B | M| (0 = perfect);
        ``hausdorff_um`` = symmetric Hausdorff distance in the grid metric
        (in-plane Manhattan + layer hops, scaled by the pitch; None when one
        side is empty and the other is not)."""
        k_sum = cls._mirror_ix_sum(grid, mc.axis_x_um)
        m = {(k_sum - ix, iy, gi) for (ix, iy, gi) in a_cells}
        inter = len(m & b_cells)
        union = len(m | b_cells)
        deviation = 0.0 if union == 0 else 1.0 - inter / union
        haus_um: Optional[float]
        if not m and not b_cells:
            haus_um = 0.0
        elif not m or not b_cells:
            haus_um = None  # undefined: one side has no metal at all
        else:
            def _h(src: Set[Cell], dst: Set[Cell]) -> int:
                worst = 0
                for (x, y, g) in src:
                    best = min(
                        abs(x - u) + abs(y - v) + abs(g - w) for (u, v, w) in dst
                    )
                    if best > worst:
                        worst = best
                return worst
            haus_um = max(_h(b_cells, m), _h(m, b_cells)) * grid.pitch / 1000.0
        satisfied = bool(connected_a and connected_b and b_cells == m)
        if not connected_a:
            mode = "partner-unrouted"
        elif not connected_b:
            mode = "unrouted"
        elif satisfied:
            mode = "exact-mirror"
        else:
            mode = "deviating"
        snapped_um = (2 * grid.x0 + k_sum * grid.pitch) / 2.0 / _NM_PER_UM
        return {
            "net_a": mc.net_a,
            "net_b": mc.net_b,
            "axis_x_um": float(mc.axis_x_um),
            "axis_x_snapped_um": snapped_um,
            "axis_snap_um": abs(snapped_um - float(mc.axis_x_um)),
            "mode": mode,
            "n_cells_a": len(a_cells),
            "n_cells_b": len(b_cells),
            "n_mirror_match": inter,
            "deviation_frac": deviation,
            "hausdorff_um": haus_um,
            "satisfied": satisfied,
        }

    # ------------------------------------------------------------------ grid build
    def _make_grid(
        self,
        tech: _Tech,
        pins_by_net: Dict[str, List[Pin]],
        obstacles: Sequence[Obstacle],
        *,
        routing_envelope_nm: Optional[object] = None,
    ) -> _Grid:
        pitch = tech.pitch_nm
        if routing_envelope_nm is not None:
            from clay.layout.route.floorplan_containment import RoutingEnvelope

            ex0, ey0, ex1, ey1 = RoutingEnvelope.coerce(routing_envelope_nm).bbox_nm
            edge_margin = max(
                1, min(tech.gi_min_width(gi) // 2 for gi in range(tech.n_layers))
            )

            def _axis_bounds(lo: int, hi: int) -> Tuple[int, int]:
                first = int(math.ceil((lo + edge_margin) / pitch) * pitch)
                last = int(math.floor((hi - edge_margin) / pitch) * pitch)
                if first <= last:
                    return first, (last - first) // pitch + 1
                return (lo + hi) // 2, 1

            x0, nx = _axis_bounds(ex0, ex1)
            y0, ny = _axis_bounds(ey0, ey1)
            return _Grid(tech, x0, y0, nx, ny)

        xs: List[float] = []
        ys: List[float] = []
        for pins in pins_by_net.values():
            for p in pins:
                xs.append(float(p.xy_um[0]))
                ys.append(float(p.xy_um[1]))
        for ob in obstacles:
            xs.extend([float(ob.x0), float(ob.x1)])
            ys.extend([float(ob.y0), float(ob.y1)])
        if not xs:
            xs = [0.0]
            ys = [0.0]
        m = self.margin_tracks * pitch
        # xs/ys are pin+obstacle coords in MICRONS; the grid is nm.
        x_min = math.floor(min(xs) * _NM_PER_UM) - m
        y_min = math.floor(min(ys) * _NM_PER_UM) - m
        x_max = math.ceil(max(xs) * _NM_PER_UM) + m
        y_max = math.ceil(max(ys) * _NM_PER_UM) + m
        # snap origin DOWN to a pitch multiple so indices stay non-negative ints.
        x0 = int(math.floor(x_min / pitch) * pitch)
        y0 = int(math.floor(y_min / pitch) * pitch)
        nx = int(math.ceil((x_max - x0) / pitch)) + 1
        ny = int(math.ceil((y_max - y0) / pitch)) + 1
        nx = max(nx, 1)
        ny = max(ny, 1)
        return _Grid(tech, x0, y0, nx, ny)

    # ------------------------------------------------------------------ preroute
    @staticmethod
    def _preroute_plan(
        tech: _Tech, preroutes: Optional[Dict[str, list]]
    ) -> Dict[str, List[Tuple[int, int, int, int, int]]]:
        """Validate + normalise caller preroute geometry (contract point 6).

        ``preroutes`` maps net -> list[LayerPolygon] (nm ints, the emit
        currency). Each polygon's ``(layer, datatype)`` MUST resolve to a layer
        INSIDE the fine routing graph: an out-of-graph preroute (unknown layer,
        or a thick/top metal the grid does not route) can neither seed the grid
        search nor block foreign tracks, so accepting it would silently break
        the same-net-seed semantics -> ValueError (rejected, never dropped).

        Returns ``{net: [(gi, x0, y0, x1, y1)]}`` nm-int bboxes (the preroute
        currency is axis-aligned rects; a non-rect polygon contributes its
        bbox). Zero-area rects are inert for seeding/blocking but still pass
        through the verbatim emit. The engine never INVENTS preroute geometry;
        content is the caller's responsibility."""
        plan: Dict[str, List[Tuple[int, int, int, int, int]]] = {}
        if not preroutes:
            return plan
        for net in sorted(preroutes):
            rects: List[Tuple[int, int, int, int, int]] = []
            for lp in preroutes[net] or ():
                key = (int(lp.layer), int(lp.datatype))
                gi = tech.resolve_layer_to_gi(key)
                if gi is None:
                    usable = [tuple(tech.gi_to_metal_key(g))
                              for g in range(tech.n_layers)]
                    raise ValueError(
                        f"preroute for net {net!r} is on layer {key}, which is "
                        f"outside the routing graph (routable metal keys: "
                        f"{usable}); an out-of-graph preroute cannot seed the "
                        "search or block tracks -- rejected, not silently dropped"
                    )
                xs = [int(v.x) for v in lp.polygon.vertices]
                ys = [int(v.y) for v in lp.polygon.vertices]
                if not xs or min(xs) >= max(xs) or min(ys) >= max(ys):
                    continue  # zero-area: nothing to seed or block
                rects.append((gi, min(xs), min(ys), max(xs), max(ys)))
            if rects:
                plan[net] = rects
        return plan

    @staticmethod
    def _preroute_obstacles(
        tech: _Tech, plan: Dict[str, List[Tuple[int, int, int, int, int]]]
    ) -> List[Obstacle]:
        """The preroute rects as OWNED obstacles (um floats, the Obstacle
        currency) -- contract 6b: every foreign net holds its width-aware
        min-space halo from the preroute (and the emit clip applies), while the
        owning net may land on it. The exact machinery device terminal pads use;
        nothing preroute-specific downstream."""
        out: List[Obstacle] = []
        for net in sorted(plan):
            for (gi, x0, y0, x1, y1) in plan[net]:
                out.append(Obstacle(
                    x0 / _NM_PER_UM, y0 / _NM_PER_UM,
                    x1 / _NM_PER_UM, y1 / _NM_PER_UM,
                    tuple(tech.gi_to_metal_key(gi)), owner_net=net,
                ))
        return out

    @staticmethod
    def _preroute_seed_cells(
        grid: _Grid, rects: List[Tuple[int, int, int, int, int]]
    ) -> Set[Cell]:
        """Rasterise one net's preroute rects into grid cells (contract 6a):
        every cell whose CENTRE lies inside a rect. A pad emitted at such a
        cell overlaps the rect by construction (its centre is interior), so a
        branch ending on a seed cell physically touches the trunk. A sub-pitch
        sliver that captures NO cell centre seeds the nearest on-grid cell to
        its centre instead (the branch pad then lands within half a pitch of
        the rect; the verbatim rect still carries the metal)."""
        cells: Set[Cell] = set()
        for (gi, x0, y0, x1, y1) in rects:
            found = False
            ix_lo = max(0, int(math.floor((x0 - grid.x0) / grid.pitch)))
            ix_hi = min(grid.nx - 1, int(math.ceil((x1 - grid.x0) / grid.pitch)))
            iy_lo = max(0, int(math.floor((y0 - grid.y0) / grid.pitch)))
            iy_hi = min(grid.ny - 1, int(math.ceil((y1 - grid.y0) / grid.pitch)))
            for ix in range(ix_lo, ix_hi + 1):
                cx = grid.x0 + ix * grid.pitch
                if cx < x0 or cx > x1:
                    continue
                for iy in range(iy_lo, iy_hi + 1):
                    cy = grid.y0 + iy * grid.pitch
                    if cy < y0 or cy > y1:
                        continue
                    cells.add((ix, iy, gi))
                    found = True
            if not found:
                ix = max(0, min(grid.nx - 1,
                                int(round(((x0 + x1) / 2 - grid.x0) / grid.pitch))))
                iy = max(0, min(grid.ny - 1,
                                int(round(((y0 + y1) / 2 - grid.y0) / grid.pitch))))
                cells.add((ix, iy, gi))
        return cells

    # ------------------------------------------------------------------ blockage
    @staticmethod
    def _obstacle_halo_nm(tech: _Tech, gi: int, wire_width_nm: int) -> int:
        """Halo (nm) to inflate an obstacle on graph-layer ``gi`` by, for a wire of
        width ``wire_width_nm``: the layer min-spacing plus HALF THE WIRE's width,
        so a wire CENTRE kept out by this halo leaves the full same-layer clearance
        from the wire EDGE to the obstacle edge. WIDTH-AWARE: a wide rail / MIM
        strap reserves its real half-width, so it does not overrun device artwork
        the way a fixed min-width halo let it (the 4919 um^2 anchor-width overlap)."""
        pdk_idx = tech._gi_to_pdk[gi]
        return tech.min_space_nm[pdk_idx] + wire_width_nm // 2

    @staticmethod
    def _raster_rect(
        grid: _Grid,
        sink: Set[Tuple[int, int]],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        halo_nm: int,
    ) -> None:
        """Mark every grid cell whose CENTRE lies inside ``[x0,x1]x[y0,y1]``
        inflated by ``halo_nm`` as blocked. A cell centre kept out of the inflated
        rect guarantees the on-pitch wire stub around that centre clears the
        original rect by the halo."""
        # x0..y1 are obstacle coords in MICRONS; the grid + halo are nm.
        lo_x = min(x0, x1) * _NM_PER_UM - halo_nm
        hi_x = max(x0, x1) * _NM_PER_UM + halo_nm
        lo_y = min(y0, y1) * _NM_PER_UM - halo_nm
        hi_y = max(y0, y1) * _NM_PER_UM + halo_nm
        ix_lo = max(0, int(math.floor((lo_x - grid.x0) / grid.pitch)))
        ix_hi = min(grid.nx - 1, int(math.ceil((hi_x - grid.x0) / grid.pitch)))
        iy_lo = max(0, int(math.floor((lo_y - grid.y0) / grid.pitch)))
        iy_hi = min(grid.ny - 1, int(math.ceil((hi_y - grid.y0) / grid.pitch)))
        for ix in range(ix_lo, ix_hi + 1):
            cx = grid.x0 + ix * grid.pitch
            if cx < lo_x or cx > hi_x:
                continue
            for iy in range(iy_lo, iy_hi + 1):
                cy = grid.y0 + iy * grid.pitch
                if cy < lo_y or cy > hi_y:
                    continue
                sink.add((ix, iy))

    def _relocate_blocked_owned_pin_cells(
        self,
        *,
        grid: _Grid,
        tech: _Tech,
        pins_by_net: Dict[str, List[Pin]],
        net_pin_cells: Dict[str, List[Cell]],
        net_layer_gis: Dict[str, List[int]],
        net_width_nm: Dict[str, int],
        obs_by_gi: Dict[int, Sequence[Tuple[float, float, float, float, Optional[str]]]],
        net_obstacle_blocked: Optional[Dict[str, Dict[int, Set[Tuple[int, int]]]]] = None,
        net_via_infeasible: Optional[Dict[str, Dict[int, Set[Tuple[int, int]]]]] = None,
        envelope_bbox: Optional[Tuple[int, int, int, int]] = None,
        native_grid=None,
        net_to_id: Optional[Dict[str, int]] = None,
    ) -> Set[str]:
        """Disambiguate pin cells that collide with another net's pin cell or snap
        strictly outside their own device terminal pad onto a foreign obstacle keep-out.

        On multi-layer stacks whose uniform routing pitch exceeds tight M1 pad-to-pad
        spacing (e.g. GF180 5LM 900nm pitch over two foreign M1 device pads separated
        by exact ``min_space = 230nm``), naive nearest-track snapping can map both
        terminals onto the same grid cell -- forcing an unresolvable PathFinder short
        and cross-net M1 bridging at emit/finalization. Moving a colliding or
        off-pad-blocked pin onto the nearest unblocked grid cell strictly inside its
        own connected device conductor component eliminates the false collision by
        construction without mutating caller pin metadata.
        """
        cell_pin_owners: Dict[Cell, Set[str]] = {}
        for net, pcells in net_pin_cells.items():
            for c in pcells:
                cell_pin_owners.setdefault(c, set()).add(net)

        owned_rects_cache: Dict[Tuple[int, str], List[Tuple[int, int, int, int]]] = {}

        def _owned_rects_for(gi: int, net: str) -> List[Tuple[int, int, int, int]]:
            key = (gi, net)
            cached = owned_rects_cache.get(key)
            if cached is not None:
                return cached
            rects = [
                (
                    int(round(min(x0, x1) * _NM_PER_UM)),
                    int(round(min(y0, y1) * _NM_PER_UM)),
                    int(round(max(x0, x1) * _NM_PER_UM)),
                    int(round(max(y0, y1) * _NM_PER_UM)),
                )
                for (x0, y0, x1, y1, own) in obs_by_gi.get(gi, ())
                if own is not None and own == net
            ]
            owned_rects_cache[key] = rects
            return rects

        def _owned_component(gi: int, net: str, px: int, py: int) -> List[Tuple[int, int, int, int]]:
            rects = _owned_rects_for(gi, net)
            if not rects:
                return []
            picked = [False] * len(rects)
            queue: List[int] = []
            for i, (rx0, ry0, rx1, ry1) in enumerate(rects):
                if rx0 - 1 <= px <= rx1 + 1 and ry0 - 1 <= py <= ry1 + 1:
                    picked[i] = True
                    queue.append(i)
            while queue:
                idx = queue.pop()
                ax0, ay0, ax1, ay1 = rects[idx]
                for j, (bx0, by0, bx1, by1) in enumerate(rects):
                    if picked[j]:
                        continue
                    if not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0):
                        picked[j] = True
                        queue.append(j)
            return [rects[i] for i, flag in enumerate(picked) if flag]

        def _is_cell_blocked(net: str, ix: int, iy: int, gi: int) -> bool:
            if native_grid is not None and net_to_id is not None:
                nid = net_to_id.get(net, -1)
                if nid >= 0:
                    return bool(native_grid.is_blocked_cell(nid, ix, iy, gi))
            if net_obstacle_blocked is not None:
                return (ix, iy) in net_obstacle_blocked.get(net, {}).get(gi, ())
            halo = self._obstacle_halo_nm(tech, gi, int(net_width_nm.get(net, tech.gi_min_width(gi))))
            cx, cy = grid.cell_xy(ix, iy)
            for (x0, y0, x1, y1, own) in obs_by_gi.get(gi, ()):
                if own is not None and own == net:
                    continue
                lo_x = min(x0, x1) * _NM_PER_UM - halo
                hi_x = max(x0, x1) * _NM_PER_UM + halo
                lo_y = min(y0, y1) * _NM_PER_UM - halo
                hi_y = max(y0, y1) * _NM_PER_UM + halo
                if lo_x <= cx <= hi_x and lo_y <= cy <= hi_y:
                    return True
            return False

        via_inf_cache: Dict[Tuple[str, int], Set[Tuple[int, int]]] = {}

        def _is_via_up_blocked(net: str, ix: int, iy: int, gi: int, lg: Sequence[int]) -> int:
            if (gi + 1) not in lg:
                return 0
            if native_grid is not None and net_to_id is not None:
                nid = net_to_id.get(net, -1)
                if nid >= 0:
                    ckey = (net, gi)
                    vset = via_inf_cache.get(ckey)
                    if vset is None:
                        vlist = native_grid.get_via_infeasible_cells(nid, gi)
                        vset = set(vlist) if vlist else set()
                        via_inf_cache[ckey] = vset
                    return 1 if (ix, iy) in vset else 0
            if net_via_infeasible is not None:
                return 1 if (ix, iy) in net_via_infeasible.get(net, {}).get(gi, ()) else 0
            return 0

        relocated_nets: Set[str] = set()
        for net in sorted(net_pin_cells):
            pins = pins_by_net.get(net, ())
            pcells = net_pin_cells[net]
            if len(pins) != len(pcells):
                continue
            lg = net_layer_gis.get(net) or list(range(tech.n_layers))
            for idx, (p, c0) in enumerate(zip(pins, pcells)):
                ix0, iy0, gi = c0
                pgi = tech.resolve_layer_to_gi(p.layer)
                if pgi is None or pgi != gi:
                    continue
                collides = len(cell_pin_owners.get(c0, ())) >= 2
                if not collides:
                    continue
                px = int(round(float(p.xy_um[0]) * _NM_PER_UM))
                py = int(round(float(p.xy_um[1]) * _NM_PER_UM))
                comp = _owned_component(gi, net, px, py)
                if not comp:
                    continue

                edge_half = max(1, tech.gi_min_width(gi) // 2)
                max_step = max(2, 2 * grid.pitch)
                best_key = None
                best_cell: Optional[Cell] = None
                seen_xy: Set[Tuple[int, int]] = set()
                for (rx0, ry0, rx1, ry1) in comp:
                    ix_lo = max(0, int(math.ceil((rx0 - grid.x0) / grid.pitch)))
                    ix_hi = min(grid.nx - 1, int(math.floor((rx1 - grid.x0) / grid.pitch)))
                    iy_lo = max(0, int(math.ceil((ry0 - grid.y0) / grid.pitch)))
                    iy_hi = min(grid.ny - 1, int(math.floor((ry1 - grid.y0) / grid.pitch)))
                    for ix in range(ix_lo, ix_hi + 1):
                        cx = grid.x0 + ix * grid.pitch
                        if cx < rx0 or cx > rx1 or abs(cx - px) > max_step:
                            continue
                        if envelope_bbox is not None:
                            if cx - edge_half < envelope_bbox[0] or cx + edge_half > envelope_bbox[2]:
                                continue
                        for iy in range(iy_lo, iy_hi + 1):
                            if (ix, iy) in seen_xy:
                                continue
                            cy = grid.y0 + iy * grid.pitch
                            if cy < ry0 or cy > ry1 or abs(cy - py) > max_step:
                                continue
                            if envelope_bbox is not None:
                                if cy - edge_half < envelope_bbox[1] or cy + edge_half > envelope_bbox[3]:
                                    continue
                            seen_xy.add((ix, iy))
                            cand = (ix, iy, gi)
                            owners = cell_pin_owners.get(cand)
                            if owners and any(o != net for o in owners):
                                continue
                            if _is_cell_blocked(net, ix, iy, gi):
                                continue
                            dist2 = (cx - px) * (cx - px) + (cy - py) * (cy - py)
                            vblk = _is_via_up_blocked(net, ix, iy, gi, lg)
                            key = (vblk, dist2, ix, iy)
                            if best_key is None or key < best_key:
                                best_key = key
                                best_cell = cand

                if best_cell is not None and best_cell != c0:
                    pcells[idx] = best_cell
                    relocated_nets.add(net)
                    if not any(other_c == c0 for other_c in pcells):
                        cell_pin_owners.get(c0, set()).discard(net)
                    cell_pin_owners.setdefault(best_cell, set()).add(net)

        return relocated_nets

    # ------------------------------------------------------------------ per-net route
    def _route_net(
        self,
        *,
        net: str,
        pins: List[Pin],
        grid: _Grid,
        tech: _Tech,
        obstacle_rects: Dict[int, List[Tuple[float, float, float, float, Optional[str]]]],
        committed: Dict[int, Dict[Tuple[int, int], str]],
        allowed_gis: Optional[List[int]],
        width_um: Optional[float],
        spacing_against: Dict[str, float],
        max_wire_width_nm: int = 0,
        max_via_pad_half_nm: int = 0,
        dir_pref: Optional[Dict[int, Tuple[str, float]]] = None,
        seed_cells: Optional[Set[Cell]] = None,
        preroute_wires: Optional[list] = None,
        mirror_cells: Optional[Set[Cell]] = None,
        native_grid=None,
        net_id: int = -1,
        checkpoint_cb=None,
        pin_cells_override: Optional[List[Cell]] = None,
    ) -> Tuple[NetRoute, Set[Cell]]:
        nr = NetRoute(net=net, pins=list(pins))
        seeds = set(seed_cells) if seed_cells else None
        pre_wires = list(preroute_wires) if preroute_wires else []
        if not pins:
            nr.connected = True  # vacuously connected (no terminals to join)
            nr.wires = list(pre_wires)  # the caller's existing metal, verbatim
            return nr, set(seeds or ())
        if len(pins) == 1 and seeds is None:
            # a single pin is its own component; emit nothing (already "connected").
            # (degenerate-rect preroute wires still pass through verbatim.)
            nr.connected = True
            nr.wires = list(pre_wires)
            return nr, set()

        layer_gis = allowed_gis if allowed_gis else list(range(tech.n_layers))
        layer_gis = sorted(set(layer_gis))

        # This net's drawn wire width -> a WIDTH-AWARE obstacle halo. Rasterise the
        # device obstacles PER NET so a wide rail / MIM strap keeps its real
        # half-width clear of artwork (a fixed min-width halo let wide wires overrun
        # devices). Other-net clearance is handled separately by _commit_halo.
        wire_width_nm = self._net_wire_width_nm(tech, layer_gis, width_um)
        use_native = native_grid is not None and net_id >= 0

        # Resolve each pin to a real start cell. route() has already rejected a
        # pin outside this net's LayerSet; no terminal is remapped to another metal.
        def pin_cell(p: Pin) -> Cell:
            ix, iy = grid.snap_index(float(p.xy_um[0]), float(p.xy_um[1]))
            seed = tech.pin_seed(
                p.layer, layer_gis, strict_layers=allowed_gis is not None
            )
            if seed is None:  # prevalidated by route(); defensive honesty
                raise ValueError(f"pin {p.device}.{p.terminal} has no legal seed")
            gi, _ladder = seed
            return (ix, iy, gi)

        pin_cells = list(pin_cells_override) if pin_cells_override is not None else [pin_cell(p) for p in pins]

        if use_native:
            def is_blocked(ix: int, iy: int, gi: int) -> bool:
                return bool(native_grid.is_blocked_cell(net_id, ix, iy, gi, True))
        else:
            obstacle_blocked: Dict[int, Set[Tuple[int, int]]] = {}
            for gi, rects in obstacle_rects.items():
                cells: Set[Tuple[int, int]] = set()
                halo = self._obstacle_halo_nm(tech, gi, wire_width_nm)
                for (x0, y0, x1, y1, owner) in rects:
                    if owner is not None and owner == net:
                        continue  # this net's OWN terminal pad -> connect there, not a keep-out
                    self._raster_rect(grid, cells, x0, y0, x1, y1, halo)
                obstacle_blocked[gi] = cells

            def is_blocked(ix: int, iy: int, gi: int) -> bool:
                if (ix, iy) in obstacle_blocked.get(gi, ()):  # device artwork on this layer
                    return True
                owner = committed[gi].get((ix, iy))
                return owner is not None and owner != net

        # --- connect pins sink-by-sink onto the growing tree -------------------
        mirror_adopted = bool(
            mirror_cells
            and seeds is None
            and self._mirror_fully_legal(
                grid, mirror_cells, pin_cells, layer_gis, is_blocked
            )
        )
        if mirror_adopted:
            wire_cells: Set[Cell] = set(mirror_cells)
        elif use_native:
            wire_cells, ok = native_grid.route_one_net_soft(
                net_id,
                pin_cells,
                layer_gis,
                seeds,
                pres_fac=0.0,
                hist_fac=0.0,
                trunk_gi=0,
                use_via_cost=False,
                dir_pref=dir_pref,
                mirror_cells=mirror_cells,
                mirror_penalty=_MIRROR_GUIDE_PENALTY if mirror_cells else 0.0,
                max_expansions=self.max_expansions,
                window_ladder=False,
                use_pf_cost=False,
                check_committed=True,
                checkpoint_cb=checkpoint_cb,
                checkpoint_interval=_CHECKPOINT_INTERVAL,
            )
            if not ok:
                nr.connected = False
                nr.wires = list(pre_wires)
                nr.length_um = 0.0
                return nr, set(seeds or ())
        else:
            mirror_cost = None
            if seeds is not None:
                tree: Set[Cell] = set(seeds)
                wire_cells = set()
                remaining: List[Cell] = list(pin_cells)
            else:
                tree = set()
                tree.add(pin_cells[0])
                wire_cells = set(tree)  # all cells the net's metal occupies
                remaining = pin_cells[1:]
            if mirror_cells:
                _mset = mirror_cells

                def mirror_cost(ix: int, iy: int, gi: int) -> float:
                    return 0.0 if (ix, iy, gi) in _mset else _MIRROR_GUIDE_PENALTY
            dir_runs: Optional[Dict[Cell, int]] = None
            if dir_pref:
                dir_runs = {}
                self._update_dir_runs(dir_runs, tree, tree, dir_pref)
            for target in remaining:
                if target in tree:
                    if seeds is not None:
                        wire_cells.add(target)
                    continue
                path = self._astar(
                    grid=grid,
                    tech=tech,
                    sources=tree,
                    target=target,
                    layer_gis=layer_gis,
                    is_blocked=is_blocked,
                    cell_cost=mirror_cost,
                    dir_pref=dir_pref,
                    dir_runs=dir_runs,
                )
                if path is None:
                    nr.connected = False
                    nr.wires = list(pre_wires)
                    nr.length_um = 0.0
                    return nr, set(seeds or ())
                for c in path:
                    tree.add(c)
                    wire_cells.add(c)
                if dir_runs is not None:
                    self._update_dir_runs(dir_runs, tree, path, dir_pref)

        width_at = (
            self._make_width_at(grid, tech, pin_cells, wire_width_nm)
            if self.taper
            else None
        )

        extra_spacing_nm = int(round(max(spacing_against.values()) * 1000.0)) if spacing_against else 0
        via_partner_half_nm = max(max_wire_width_nm // 2, max_via_pad_half_nm)
        if use_native:
            native_grid.commit_maze_halo(
                net_id,
                wire_cells,
                wire_width_nm,
                extra_spacing_nm,
                max_wire_width_nm,
                bool(self.taper),
                via_partner_half_nm,
                True,
                True,
            )
            foreign_rects = self._foreign_emit_rects(tech, obstacle_rects, net)
            raw_fm = native_grid.foreign_committed_rects(net_id)
            foreign_metal_rects = {
                tech.gi_to_metal_key(gi): rects for gi, rects in raw_fm.items()
            }

            def _taken(jx: int, jy: int, l: int, _w_ext: int) -> bool:
                return bool(native_grid.is_committed_foreign(net_id, jx, jy, l))
        else:
            for (ix, iy, gi) in wire_cells:
                committed[gi][(ix, iy)] = net
            self._commit_halo(
                grid, tech, committed, net, wire_cells, wire_width_nm, extra_spacing_nm,
                max_wire_width_nm, width_at=width_at,
            )
            self._commit_via_pad_halo(
                grid, tech, committed, net, wire_cells, extra_spacing_nm,
                via_partner_half_nm,
            )
            foreign_rects = self._foreign_emit_rects(tech, obstacle_rects, net)
            foreign_metal_rects = self._foreign_metal_rects(grid, tech, committed, net)

            def _taken(jx: int, jy: int, l: int, _w_ext: int) -> bool:
                own = committed[l].get((jx, jy))
                return own is not None and own != net

        reloc: List[Tuple[int, int, int, int]] = []
        nr.wires, nr.length_um = self._emit_wires(
            grid, tech, wire_cells, pin_cells, wire_width_nm, width_at=width_at,
            foreign_rects=foreign_rects, cell_taken=_taken, reloc_cells=reloc,
            foreign_metal_rects=foreign_metal_rects,
        )
        if reloc:
            rc_cells: Set[Cell] = {(jx, jy, l) for (jx, jy, l, _w) in reloc}
            for (jx, jy, l) in rc_cells:
                wire_cells.add((jx, jy, l))
            if use_native:
                native_grid.commit_maze_halo(
                    net_id,
                    rc_cells,
                    wire_width_nm,
                    extra_spacing_nm,
                    max_wire_width_nm,
                    False,
                    0,
                    False,
                    False,
                )
            else:
                for (jx, jy, l) in rc_cells:
                    committed[l].setdefault((jx, jy), net)
                self._commit_halo(
                    grid, tech, committed, net, rc_cells, wire_width_nm,
                    extra_spacing_nm, max_wire_width_nm,
                )
        nr.wires.extend(self._emit_pin_risers(
            grid, tech, pins, layer_gis, foreign_rects,
            strict_layers=allowed_gis is not None,
        ))
        # contract 6c: the preroute rects join the net's wires VERBATIM (no clip
        # -- they are pre-existing metal, not drawn by this engine) and flow
        # through the same per-net GDS merge as every routed fragment.
        nr.wires.extend(pre_wires)
        self._repair_clipped_terminal_gaps(
            tech, net, nr.wires, pins, obstacle_rects, foreign_rects, foreign_metal_rects
        )
        nr.connected = True
        # contract 6d: the oracles count the preroute as this net's metal.
        return nr, set(wire_cells) | set(seeds or ())

    def _emit_pin_risers(
        self,
        grid: _Grid,
        tech: _Tech,
        pins: List[Pin],
        layer_gis: List[int],
        foreign_rects,
        *,
        strict_layers: bool = False,
    ) -> List[LayerPolygon]:
        """Deterministic VIA LADDERS for pins on out-of-graph (thick/top) metals.

        For each pin whose ``pin_seed`` returned ladder levels, draw -- at the
        pin's exact xy -- one via cut per level plus a landing pad on EVERY metal
        from the seed (top graph) metal up to the pin's metal. Pad side =
        max(layer min-width, cut + 2x60nm enclosure default -- the synthesized
        deck parses no VIA4/VIA5 enclosure rule; the KLayout DRC oracle
        adjudicates). Every rect passes the same foreign-net min-space clip as
        ordinary emitted metal; a fully clipped-away ladder level is honest
        breakage the geometric-open oracle reports (never silently skipped).
        No track routing happens on these layers -- the ladder is the promised
        thick-metals-stay-via-target-layers contract made real."""
        out: List[LayerPolygon] = []
        enc = 60  # nm, conservative via enclosure default (no parsed VIA4/5 rule)

        def _clip_add(key, x0: int, y0: int, x1: int, y1: int) -> None:
            holes = None if foreign_rects is None else foreign_rects.get(key)
            rects = ([(x0, y0, x1, y1)] if not holes
                     else _subtract_rects((x0, y0, x1, y1), holes))
            floor = min(x1 - x0, y1 - y0)
            for (sx0, sy0, sx1, sy1) in rects:
                if min(sx1 - sx0, sy1 - sy0) < min(floor, 1):
                    continue
                out.append(LayerPolygon(layer=key[0], datatype=key[1],
                                        polygon=Polygon.rectangle(sx0, sy0, sx1, sy1)))

        for p in pins:
            seed = tech.pin_seed(
                p.layer, layer_gis, strict_layers=strict_layers
            )
            if seed is None:
                continue
            _gi, levels = seed
            if not levels:
                continue
            # ladder at the SNAPPED grid cell centre -- the SAME coordinate the
            # seed cell's pad is emitted at, so ladder and tree metal touch BY
            # CONSTRUCTION (drawn at the exact pin xy, a ladder up to half a
            # pitch away from the seed pad became an orphan island -- the
            # layout=$N extra net the trunk-mode LVS caught). The plate region
            # is orders larger than the snap offset, so the snapped landing
            # stays on the plate.
            ix, iy = grid.snap_index(float(p.xy_um[0]), float(p.xy_um[1]))
            cx, cy = grid.cell_xy(ix, iy)
            cut_of = {lv: self._riser_cut_nm(tech, lv) for lv in levels}
            for lv in levels:
                try:
                    vkey = tuple(tech.pdk.via_key(lv))
                except Exception:
                    continue
                vh = max(1, cut_of[lv] // 2)
                _clip_add(vkey, cx - vh, cy - vh, cx + vh, cy + vh)
            for m in range(levels[0], levels[-1] + 2):
                key = tech.metal_keys.get(m)
                if key is None:
                    continue
                cuts = [cut_of.get(m - 1, 0), cut_of.get(m, 0)]
                side = max(tech.min_width_nm.get(m, 1),
                           (max(cuts) + 2 * enc) if max(cuts) else 1)
                h = max(1, int(side) // 2)
                _clip_add(tuple(key), cx - h, cy - h, cx + h, cy + h)
        return out

    @staticmethod
    def _riser_cut_nm(tech: _Tech, lv: int) -> int:
        """The via CUT size (nm) the riser ladder draws at via level ``lv`` --
        the single source for both the emitted geometry (``_emit_pin_risers``)
        and the pre-route pad blockage (``_riser_pad_obstacles``); the two must
        never disagree on the pad footprint.

        Raises if the PDK has no parsed via-size rule for this level -- a
        silent ``or 260`` fallback used to reuse a gf180-shaped guess on ANY
        PDK (including open_pdk, whose real VIA4 cut is 360nm), masking the
        exact-size DRC rule (VIA4.W.1) with a wrong-by-construction cut."""
        via_names = set(tech.pdk.via_names())
        vname = None
        for cand in ("via%d" % lv, "vi%d" % lv, "v%d" % lv):
            if cand in via_names:
                vname = cand
                break
        size = tech.pdk.via_size(vname) if vname else None
        if not size:
            raise ValueError(
                f"{getattr(tech.pdk, 'pdk_id', tech.pdk)!r}: no via-size rule "
                f"for level {lv} (tried via{lv}/vi{lv}/v{lv} against "
                f"{sorted(via_names)}) -- the riser ladder cannot size its cut "
                f"without PDK rule data (no silent gf180-260nm fallback)"
            )
        return int(size)

    def _riser_pad_obstacles(
        self,
        grid: _Grid,
        tech: _Tech,
        pins_by_net: Dict[str, List[Pin]],
        layer_set_by_net: Dict[str, List[int]],
    ) -> List[Tuple[int, float, float, float, float, str]]:
        """RISER-PAD CLEARANCE CONTRACT: every out-of-graph pin's ladder lands a
        SEED-LAYER pad of side ``max(layer min-width, cut + 2x60nm)`` -- WIDER
        than the wire width the engines' cross-net clearance accounts for.
        Adjacent tracks are exactly min-space-tight for min-width wires
        (pitch == width + space), so the pad overhang makes any foreign wire on
        the neighbouring track a spacing violation BY CONSTRUCTION (the
        net25/VOUT 180nm case, test4 open_pdk 2026-06-11). The pad's footprint
        is a pure function of the INPUT pins, so it enters the obstacle model
        up front as an OWNED rect: foreign nets hold their width-aware halo
        from it, the owning net may still land on it -- the exact owner
        semantics device terminal pads already use. Returns
        ``(seed_gi, x0, y0, x1, y1, owner_net)`` rects in MICRONS."""
        enc = 60  # nm -- the same conservative enclosure _emit_pin_risers uses
        usable = list(range(tech.n_layers))
        out: List[Tuple[int, float, float, float, float, str]] = []
        for net in sorted(pins_by_net):
            lg = sorted(set(layer_set_by_net.get(net) or usable))
            for p in pins_by_net[net]:
                seed = tech.pin_seed(
                    p.layer, lg, strict_layers=net in layer_set_by_net
                )
                if seed is None:
                    continue
                seed_gi, levels = seed
                if not levels:
                    continue
                m_seed = levels[0]  # the seed METAL's pdk level (= lowest via level)
                cut = self._riser_cut_nm(tech, m_seed)
                side = max(tech.min_width_nm.get(m_seed, 1), cut + 2 * enc)
                ix, iy = grid.snap_index(float(p.xy_um[0]), float(p.xy_um[1]))
                cx, cy = grid.cell_xy(ix, iy)
                h = max(1, int(side) // 2)
                out.append((seed_gi, (cx - h) / 1000.0, (cy - h) / 1000.0,
                            (cx + h) / 1000.0, (cy + h) / 1000.0, net))
        return out

    @staticmethod
    def _foreign_emit_rects(
        tech: _Tech,
        obstacle_rects: Dict[int, List[Tuple[float, float, float, float, Optional[str]]]],
        net: str,
    ) -> Dict[LayerKey, List[Tuple[int, int, int, int]]]:
        """The net-aware keep-out rectangles (nm, per GDS ``(layer, datatype)``)
        that ``net`` does NOT own -- foreign cap plates + other devices' pads. The
        emit clips its rectangles against these so no foreign keep-out is grazed.

        EMIT SPACING CONTRACT: every rect is inflated by its layer's MIN-SPACE, so
        emitted metal keeps the PDK same-layer spacing to every foreign conductor
        -- not merely non-overlap. The maze search already keeps the wire BODY's
        centre min-space + half-width clear of device artwork; what it cannot
        constrain is the OFF-GRID terminal-landing flare (the pad/bridge emitted
        around a pin cell), which under the raw-rect clip could stop a sub-min-space
        sliver away from a foreign device finger (the real KLayout M1.S.1 the external_pdk
        oracle caught at the design-sourced placement). A wire legitimately running
        AT min-space alongside a plate is untouched: with integer nm coords the
        inflated hole only TOUCHES its edge (zero-area intersection -> no clip).
        ``owner is None`` (unowned device/well artwork) is foreign to every signal
        net. ``owner == net`` (this net's own terminal pad / plate) is NOT clipped
        -- the net connects there."""
        out: Dict[LayerKey, List[Tuple[int, int, int, int]]] = {}
        for gi, rects in obstacle_rects.items():
            key = tech.gi_to_metal_key(gi)
            space = int(tech.min_space_nm[tech._gi_to_pdk[gi]])
            for (x0, y0, x1, y1, owner) in rects:
                if owner is not None and owner == net:
                    continue  # this net owns it -> may land on it
                rx0 = int(round(min(x0, x1) * _NM_PER_UM)) - space
                ry0 = int(round(min(y0, y1) * _NM_PER_UM)) - space
                rx1 = int(round(max(x0, x1) * _NM_PER_UM)) + space
                ry1 = int(round(max(y0, y1) * _NM_PER_UM)) + space
                if rx0 < rx1 and ry0 < ry1:
                    out.setdefault(key, []).append((rx0, ry0, rx1, ry1))
        return out

    @staticmethod
    def _net_wire_width_nm(
        tech: _Tech, layer_gis: List[int], width_um: Optional[float]
    ) -> int:
        """The net's drawn wire width (nm): the requested width clamped UP to the
        PDK metal-min (never down), defaulting to the metal-min when unspecified.
        Uses the max metal-min over the net's allowed layers so the wire is legal
        on any layer it may use."""
        min_w = max(tech.gi_min_width(gi) for gi in layer_gis)
        if width_um is None:
            return min_w
        req = int(round(float(width_um) * 1000.0))
        return max(req, min_w)

    def _make_width_at(self, grid: _Grid, tech: _Tech, pin_cells: List[Cell], trunk_nm: int):
        """A per-net TAPERED width function ``fn(ix, iy, gi) -> width_nm``: the PDK
        min-width within a neck radius of any pin (so the wire necks to fit tight
        terminal spacing -- e.g. a drain + gate of the same device feeding two
        different nets, 1.74 um apart, which a full-trunk wire cannot clear), and
        the full ``trunk_nm`` in the open channel (so the trunk matches the
        reference per-net width). Segment-emit keeps the necked run continuous even
        below the grid pitch. The neck radius is ~one trunk-width of tracks
        (Chebyshev), so the wire only widens once it has cleared the congested
        terminal region.

        This lives on the BASE engine so both it (under ``taper=True``) and the
        :class:`~clay.layout.route.engine_pathfinder.PathFinderEngine` (which always
        tapers) share ONE definition -- PathFinder calls ``self._make_width_at`` and
        inherits this. No duplication."""
        pitch = grid.pitch
        neck_r = max(1, int(math.ceil(trunk_nm / pitch)))
        neck: Set[Tuple[int, int]] = set()
        for (px, py, _gi) in pin_cells:
            for dx in range(-neck_r, neck_r + 1):
                for dy in range(-neck_r, neck_r + 1):
                    neck.add((px + dx, py + dy))

        def width_at(ix: int, iy: int, gi: int) -> int:
            if (ix, iy) in neck:
                return tech.gi_min_width(gi)
            return trunk_nm

        return width_at

    def _widen_to_clearance(
        self,
        grid,
        tech,
        metal_by_net,
        obs_by_gi,
        trunk_by_net,
        *,
        routing_envelope_nm: Optional[object] = None,
        extra_obs_clearance_by_net_gi: Optional[
            Dict[Tuple[int, str], Sequence[Tuple[int, int, int, int, int]]]
        ] = None,
    ):
        """Per net, a width function ``fn(ix,iy,gi)->nm`` that WIDENS each routed
        cell to the most it can be without violating min-space: the open trunk
        reaches the reference width, necking only where a FOREIGN neighbour (another
        net's metal, or device artwork the net does not own) is close.

        For a cell whose nearest foreign cell is ``d`` tracks away the width is
        clamped to ``[min_width, d*pitch - min_space, trunk_target]``. When
        ``routing_envelope_nm`` is supplied, each cell's symmetric half-width is
        also capped at its distance to all four floorplan edges so widening can
        never exceed the envelope. With both
        nets of a pair using the same rule on their mutual distance ``d`` the
        edge-to-edge clearance is exactly ``min_space`` (each takes ``d*pitch -
        min_space`` and ``d*pitch - (d*pitch - min_space) == min_space``); against a
        fixed device it OVER-clears (>= min_space), so the result is DRC-clean BY
        CONSTRUCTION -- widen only into free space. This is how the reference
        achieves wide trunks with min-space channels: route thin (clean, all nets
        connected), then fatten where there is room."""
        env_bbox: Optional[Tuple[int, int, int, int]] = None
        if routing_envelope_nm is not None:
            from clay.layout.route.floorplan_containment import RoutingEnvelope

            env_bbox = RoutingEnvelope.coerce(routing_envelope_nm).bbox_nm

        def _env_width_cap(ix: int, iy: int) -> Optional[int]:
            if env_bbox is None:
                return None
            ex0, ey0, ex1, ey1 = env_bbox
            cx, cy = grid.cell_xy(ix, iy)
            edge_clear = min(cx - ex0, ex1 - cx, cy - ey0, ey1 - cy)
            return 2 * max(0, int(edge_clear))

        pitch = grid.pitch
        # owner grid per layer: cell -> owning net (metal) or '__dev__'/owner for
        # device artwork. A net's own metal + its own owned terminal pad are NOT
        # foreign to it; everything else is.
        owner: Dict[int, Dict[Tuple[int, int], str]] = {
            gi: {} for gi in range(tech.n_layers)
        }
        via_pad_by_gi: Dict[int, Dict[Tuple[int, int], Tuple[str, int]]] = {
            gi: {} for gi in range(tech.n_layers)
        }
        for net, cells in metal_by_net.items():
            by_xy_v: Dict[Tuple[int, int], Set[int]] = {}
            for (ix, iy, gi) in cells:
                owner[gi][(ix, iy)] = net
                by_xy_v.setdefault((ix, iy), set()).add(gi)
            for (ix, iy), gis in by_xy_v.items():
                for gl in gis:
                    if (gl + 1) not in gis:
                        continue
                    rb, ra = self._via_env_rule_nm(tech, gl)
                    vsz = _via_cut_size_nm(tech, gl)
                    for g, enc in ((gl, rb), (gl + 1, ra)):
                        if enc <= 0:
                            continue
                        ph = vsz // 2 + int(enc)
                        cur = via_pad_by_gi[g].get((ix, iy))
                        if cur is None or ph > cur[1]:
                            via_pad_by_gi[g][(ix, iy)] = (net, ph)
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

        mfg_grid = max(1, int(pdk_manufacturing_grid_nm(tech.pdk)))

        def _snap_half_down(half_nm: int, floor_half: int, cx_nm: int, cy_nm: int) -> int:
            if mfg_grid <= 1 or cx_nm % mfg_grid != 0 or cy_nm % mfg_grid != 0:
                return max(floor_half, half_nm)
            snapped = (half_nm // mfg_grid) * mfg_grid
            return snapped if snapped >= floor_half else floor_half

        same_net_obs_nm: Dict[Tuple[int, str], List[Tuple[int, int, int, int]]] = {}
        for gi, rects in obs_by_gi.items():
            # inflate device cells by min_space + half a cell so the foreign-device
            # footprint covers the actual poly EDGE (a poly extends ~half a pitch
            # beyond its last inside-centre cell); the widen then necks min-space
            # clear of the real device edge, not just the cell centre.
            ms_gi = tech.min_space_nm[tech._gi_to_pdk[gi]]
            dev_halo = ms_gi + grid.pitch // 2
            for (x0, y0, x1, y1, own) in rects:
                if own is not None:
                    same_net_obs_nm.setdefault((gi, own), []).append((
                        int(round(min(x0, x1) * _NM_PER_UM)),
                        int(round(min(y0, y1) * _NM_PER_UM)),
                        int(round(max(x0, x1) * _NM_PER_UM)),
                        int(round(max(y0, y1) * _NM_PER_UM)),
                    ))
                dc: Set[Tuple[int, int]] = set()
                self._raster_rect(grid, dc, x0, y0, x1, y1, dev_halo)
                for c in dc:
                    if c not in owner[gi]:  # never overwrite a net's drawn metal
                        owner[gi][c] = own if own is not None else "__dev__"

        out: Dict[str, object] = {}
        for net, cells in metal_by_net.items():
            trunk = int(trunk_by_net.get(net, 0))
            wmap: Dict[Cell, int] = {}
            for (ix, iy, gi) in cells:
                ms = tech.min_space_nm[tech._gi_to_pdk[gi]]
                mw = tech.gi_min_width(gi)
                tgt = max(mw, trunk)
                max_d = max(1, int(math.ceil((tgt + ms) / pitch)))
                og = owner[gi]
                d = max_d + 1
                for r in range(1, max_d + 1):
                    hit = False
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            if max(abs(dx), abs(dy)) != r:
                                continue
                            o = og.get((ix + dx, iy + dy))
                            if o is not None and o != net:
                                hit = True
                                break
                        if hit:
                            break
                    if hit:
                        d = r
                        break
                w = max(mw, min(tgt, int(d * pitch - ms)))
                if w > mw:
                    cx, cy = grid.cell_xy(ix, iy)
                    lo_base = mw // 2
                    vp_gi = via_pad_by_gi[gi]
                    for dx in range(-max_d, max_d + 1):
                        for dy in range(-max_d, max_d + 1):
                            if dx == 0 and dy == 0:
                                continue
                            vp = vp_gi.get((ix + dx, iy + dy))
                            if vp is None or vp[0] == net:
                                continue
                            pad_h = vp[1]
                            ax_dx = abs(dx) * pitch
                            ax_dy = abs(dy) * pitch
                            if dx == 0 or dy == 0:
                                dist = max(ax_dx, ax_dy)
                                w = min(w, max(mw, 2 * (dist - ms - pad_h)))
                            else:
                                own_h = w // 2
                                gx = max(0.0, ax_dx - own_h - pad_h)
                                gy = max(0.0, ax_dy - own_h - pad_h)
                                if math.hypot(gx, gy) < ms:
                                    lo_h = lo_base
                                    hi_h = own_h
                                    if math.hypot(
                                        max(0.0, ax_dx - lo_h - pad_h),
                                        max(0.0, ax_dy - lo_h - pad_h),
                                    ) < ms:
                                        w = mw
                                    else:
                                        while lo_h < hi_h:
                                            mid_h = (lo_h + hi_h + 1) // 2
                                            if math.hypot(
                                                max(0.0, ax_dx - mid_h - pad_h),
                                                max(0.0, ax_dy - mid_h - pad_h),
                                            ) >= ms:
                                                lo_h = mid_h
                                            else:
                                                hi_h = mid_h - 1
                                        snapped_h = _snap_half_down(lo_h, lo_base, cx, cy)
                                        w = min(w, max(mw, 2 * snapped_h))
                same_rects = same_net_obs_nm.get((gi, net))
                if same_rects and w > mw:
                    cx, cy = grid.cell_xy(ix, iy)
                    lo_base = mw // 2
                    for (bx0, by0, bx1, by1) in same_rects:
                        own_h = w // 2
                        if own_h <= lo_base:
                            break
                        dx_h = max(0, bx0 - (cx + own_h), (cx - own_h) - bx1)
                        dy_h = max(0, by0 - (cy + own_h), (cy - own_h) - by1)
                        if math.hypot(dx_h, dy_h) >= ms:
                            continue
                        ov_x = min(cx + own_h, bx1) - max(cx - own_h, bx0)
                        ov_y = min(cy + own_h, by1) - max(cy - own_h, by0)
                        if ov_x > 0 and ov_y > 0:
                            continue
                        dx_lo = max(0, bx0 - (cx + lo_base), (cx - lo_base) - bx1)
                        dy_lo = max(0, by0 - (cy + lo_base), (cy - lo_base) - by1)
                        if math.hypot(dx_lo, dy_lo) < ms:
                            continue
                        lo_h = lo_base
                        hi_h = own_h
                        while lo_h < hi_h:
                            mid_h = (lo_h + hi_h + 1) // 2
                            dx_m = max(0, bx0 - (cx + mid_h), (cx - mid_h) - bx1)
                            dy_m = max(0, by0 - (cy + mid_h), (cy - mid_h) - by1)
                            if math.hypot(dx_m, dy_m) >= ms:
                                lo_h = mid_h
                            else:
                                hi_h = mid_h - 1
                        snapped_h = _snap_half_down(lo_h, lo_base, cx, cy)
                        w = min(w, max(mw, 2 * snapped_h))
                extra_rects = (
                    extra_obs_clearance_by_net_gi.get((gi, net))
                    if extra_obs_clearance_by_net_gi
                    else None
                )
                if extra_rects and w > mw:
                    cx, cy = grid.cell_xy(ix, iy)
                    lo_base = mw // 2
                    for (bx0, by0, bx1, by1, clr) in extra_rects:
                        own_h = w // 2
                        if own_h <= lo_base:
                            break
                        dx_h = max(0, bx0 - (cx + own_h), (cx - own_h) - bx1)
                        dy_h = max(0, by0 - (cy + own_h), (cy - own_h) - by1)
                        if dx_h >= clr or dy_h >= clr:
                            continue
                        dx_lo = max(0, bx0 - (cx + lo_base), (cx - lo_base) - bx1)
                        dy_lo = max(0, by0 - (cy + lo_base), (cy - lo_base) - by1)
                        if dx_lo < clr and dy_lo < clr:
                            w = mw
                            break
                        lo_h = lo_base
                        hi_h = own_h
                        while lo_h < hi_h:
                            mid_h = (lo_h + hi_h + 1) // 2
                            dx_m = max(0, bx0 - (cx + mid_h), (cx - mid_h) - bx1)
                            dy_m = max(0, by0 - (cy + mid_h), (cy - mid_h) - by1)
                            if dx_m >= clr or dy_m >= clr:
                                lo_h = mid_h
                            else:
                                hi_h = mid_h - 1
                        snapped_h = _snap_half_down(lo_h, lo_base, cx, cy)
                        w = min(w, max(mw, 2 * snapped_h))
                env_cap = _env_width_cap(ix, iy)
                if env_cap is not None:
                    w = min(w, env_cap)
                wmap[(ix, iy, gi)] = w

            def make_fn(wmap_local, trunk_local):
                def fn(ix: int, iy: int, gi: int) -> int:
                    w = wmap_local.get((ix, iy, gi))
                    if w is not None:
                        return w
                    w = max(tech.gi_min_width(gi), trunk_local)
                    env_cap = _env_width_cap(ix, iy)
                    return min(w, env_cap) if env_cap is not None else w
                return fn

            out[net] = make_fn(wmap, trunk)
        return out

    def _commit_halo(
        self,
        grid: _Grid,
        tech: _Tech,
        committed: Dict[int, Dict[Tuple[int, int], str]],
        net: str,
        wire_cells: Set[Cell],
        wire_width_nm: int,
        extra_spacing_nm: int,
        max_other_width_nm: int = 0,
        width_at=None,
    ) -> None:
        """Reserve a halo of cells around the net's metal as blockage for OTHER
        nets, so a wider wire (or a requested inter-net spacing) keeps later nets
        clear BY CONSTRUCTION. The halo radius in TRACKS = ceil((half-width +
        min-space + extra-spacing + PARTNER half-width) / pitch): reserving the
        widest partner's half-width makes the clearance SYMMETRIC, so a later wider
        net's edge cannot land within min-space of this net's edge. Cells already
        owned by this net stay this net's; cells owned by another net are left as-is
        (a real short there is what the short oracle will surface).

        ``width_at`` (a ``fn(ix, iy, gi) -> width_nm``) makes the halo PER CELL:
        a NECKED cell (near a pin) reserves a small halo so a DIFFERENT net's close
        terminal is not over-blocked; a TRUNK cell reserves the full clearance.
        When None (the default) the constant ``wire_width_nm`` is used for every
        cell -- byte-identical to the uniform-halo behaviour. (The partner-width
        term ``max_other_width_nm`` stays the same regardless, so the clearance
        between two TRUNK runs is unchanged.)"""
        pitch = grid.pitch

        def _w(ix: int, iy: int, gi: int) -> int:
            return int(width_at(ix, iy, gi)) if width_at is not None else wire_width_nm

        # group cells by layer for the per-layer min-space
        by_layer: Dict[int, List[Tuple[int, int]]] = {}
        for (ix, iy, gi) in wire_cells:
            by_layer.setdefault(gi, []).append((ix, iy))
        for gi, cells in by_layer.items():
            min_space = tech.min_space_nm[tech._gi_to_pdk[gi]]
            for (ix, iy) in cells:
                radius_nm = (
                    _w(ix, iy, gi) // 2 + min_space + extra_spacing_nm + max_other_width_nm // 2
                )
                r_tracks = max(0, int(math.ceil(radius_nm / pitch)))
                if r_tracks == 0:
                    continue
                for dx in range(-r_tracks, r_tracks + 1):
                    for dy in range(-r_tracks, r_tracks + 1):
                        jx, jy = ix + dx, iy + dy
                        if not grid.in_bounds(jx, jy):
                            continue
                        committed[gi].setdefault((jx, jy), net)

    def _via_cells(self, wire_cells: Set[Cell]) -> List[Tuple[int, int, int]]:
        """The ``(ix, iy, gi_low)`` of every via the net's routed cells imply: a
        cell occupied on two graph-adjacent layers ``gi``/``gi+1`` carries a cut
        between them (the same condition ``_emit_wires`` uses to drop the cut)."""
        by_xy: Dict[Tuple[int, int], Set[int]] = {}
        for (ix, iy, gi) in wire_cells:
            by_xy.setdefault((ix, iy), set()).add(gi)
        out: List[Tuple[int, int, int]] = []
        for (ix, iy), gis in by_xy.items():
            for gi in gis:
                if (gi + 1) in gis:
                    out.append((ix, iy, gi))
        return out

    def _commit_via_pad_halo(
        self,
        grid: _Grid,
        tech: _Tech,
        committed: Dict[int, Dict[Tuple[int, int], str]],
        net: str,
        wire_cells: Set[Cell],
        extra_spacing_nm: int,
        partner_half_nm: int = 0,
    ) -> None:
        """Reserve the via LANDING-PAD spacing halo around every via cell, on BOTH
        metals the cut bridges, as blockage for later nets.

        The pad on each metal is the PDK rule landing (cut size + 2*per-side
        enclosure -- ``via_model`` via ``_via_env_rule_nm``), so its half-width is
        ``cut/2 + enclosure`` -- wider than the wire's. The reserved radius is
        ``pad_half + min_space + partner_half`` (the symmetric clearance the wire
        ``_commit_halo`` uses, with the pad half-width substituted for the wire's),
        so a later net's metal edge cannot land within the PDK min_space of the pad
        edge. ``partner_half_nm`` is the half-width of the WIDEST object a foreign
        net may face this pad with -- another via PAD, not merely the widest wire --
        so two foreign nets' via pads keep min_space between their pad EDGES, not
        pad-vs-wire (the gf180 M2.2a cross-net via-pad gap=230 family). Every term is
        PDK-sourced (cut size + enclosure + min_space); no per-PDK constant. A side
        the PDK declares no enclosure on draws no pad wider than the cut, so its
        radius collapses to the wire case (no change)."""
        pitch = grid.pitch
        for (ix, iy, gi_low) in self._via_cells(wire_cells):
            vsz = _via_cut_size_nm(tech, gi_low)
            rb, ra = self._via_env_rule_nm(tech, gi_low)
            for (gi, enc) in ((gi_low, rb), (gi_low + 1, ra)):
                if enc <= 0:
                    continue  # no pad wider than the cut here -> wire halo suffices
                min_space = tech.min_space_nm[tech._gi_to_pdk[gi]]
                pad_half = vsz // 2 + int(enc)
                radius_nm = pad_half + min_space + extra_spacing_nm + int(partner_half_nm)
                r_tracks = max(0, int(math.ceil(radius_nm / pitch)))
                for dx in range(-r_tracks, r_tracks + 1):
                    for dy in range(-r_tracks, r_tracks + 1):
                        jx, jy = ix + dx, iy + dy
                        if not grid.in_bounds(jx, jy):
                            continue
                        committed[gi].setdefault((jx, jy), net)

    def _foreign_metal_rects(
        self,
        grid: _Grid,
        tech: _Tech,
        committed: Dict[int, Dict[Tuple[int, int], str]],
        net: str,
    ) -> Dict[LayerKey, List[Tuple[int, int, int, int]]]:
        """The min-space-inflated keep-out rectangles (nm, per GDS layer) of the
        metal OTHER nets have already committed -- the relational foreign-metal set
        the via landing must hold ``min_space`` to.

        Each foreign-owned cell ``(jx, jy)`` on layer ``gi`` reserves a square of
        ``+/- (pitch/2 + min_space)`` about its centre: the pitch/2 covers the
        foreign cell's own half-pad and the min_space is the PDK clearance the via
        pad owes it. A via landing pad that would intersect any of these is clipped
        (and so relocated/suppressed by ``_emit_via_obligated``). Conservative by
        construction (it never under-reserves), and PDK-sourced (pitch + min_space),
        no per-PDK constant. Cells this ``net`` owns are excluded -- the net may
        land on its own metal."""
        out: Dict[LayerKey, List[Tuple[int, int, int, int]]] = {}
        half_pitch = grid.pitch // 2
        for gi, cellmap in committed.items():
            if not cellmap:
                continue
            key = tech.gi_to_metal_key(gi)
            min_space = int(tech.min_space_nm[tech._gi_to_pdk[gi]])
            reach = half_pitch + min_space
            rects: List[Tuple[int, int, int, int]] = []
            for (jx, jy), owner in cellmap.items():
                if owner == net:
                    continue  # this net's own metal -> not foreign
                cx, cy = grid.cell_xy(jx, jy)
                rects.append((cx - reach, cy - reach, cx + reach, cy + reach))
            if rects:
                out[key] = rects
        return out

    # ------------------------------------------------------------------ A*
    def _astar(
        self,
        *,
        grid: _Grid,
        tech: _Tech,
        sources: Set[Cell],
        target: Cell,
        layer_gis: List[int],
        is_blocked,
        cell_cost=None,
        dir_pref: Optional[Dict[int, Tuple[str, float]]] = None,
        dir_runs: Optional[Dict[Cell, int]] = None,
        via_blocked=None,
        via_cost=None,
        window_slack: Optional[int] = None,
    ) -> Optional[List[Cell]]:
        """Multi-source A* from any cell in ``sources`` to ``target`` over the
        multi-layer grid. Returns the path (list of cells incl. both ends) or
        ``None`` if unreachable within the expansion budget.

        ``window_slack`` (optional, STEP 2 per-net bounded window) confines the
        search to the (ix,iy) bounding box of ``sources`` + ``target`` inflated by
        ``window_slack`` tracks on every side; a neighbour outside the box is
        early-dropped so an easy net does not flood the whole pin/obstacle hull.
        ``None`` (the default for every non-PathFinder caller) -> UNBOUNDED, exactly
        the pre-STEP-2 search, so those routes stay byte-identical. The PathFinder
        soft route (``_route_one_net_soft``) passes a slack and WIDENS it on failure
        (finally ``None``) so ``connected=False`` still means truly unroutable.

        ``via_blocked`` (optional, Task D2) hard-forbids pad-infeasible layer
        transitions -- see :meth:`_neighbours`. Default None -> byte-identical.

        Cost: a step is 1 grid pitch (+ a wrong-direction soft penalty), a via is
        ``_VIA_PENALTY``. Heuristic: Manhattan distance in tracks to the target
        (admissible -- it never over-estimates the >=1-per-track real cost), so A*
        is optimal on the don't-care freedom.

        ``cell_cost`` (optional) ``fn(ix, iy, gi) -> float`` adds a per-cell ENTRY
        penalty to the step cost -- the negotiated-congestion hook (PathFinder): a
        cell shared with other nets costs more, so rip-up/reroute steers off it.
        Default None leaves the greedy cost unchanged. The heuristic stays the
        Manhattan lower bound; with non-negative cell_cost A* remains correct (it
        may expand more, never miss a cheaper path).

        ``dir_pref`` (optional) ``{gi: (axis, weight)}`` is the Direction soft
        cost (see :meth:`_neighbours`): an in-plane step against the layer's
        preferred axis pays an extra ``weight x base step cost``. Default None ->
        byte-identical to the pre-Direction cost. Non-negative extra cost keeps
        the Manhattan heuristic admissible.

        ``dir_runs`` (optional, only meaningful with ``dir_pref``) seeds each
        source cell's anti-command run length from the net's EXISTING metal
        (tree memory; see the module-level comb-contract comment). With
        ``dir_pref`` present the search runs run-extent aware
        (:meth:`_astar_directed`); without it this method is byte-identical to
        the pre-Direction engine."""
        if dir_pref:
            # run-extent-aware variant (the comb contract). Kept as a SEPARATE
            # loop so the no-Direction path below stays byte-untouched (the
            # frozen-SHA back-compat bar).
            return self._astar_directed(
                grid=grid,
                tech=tech,
                sources=sources,
                target=target,
                layer_gis=layer_gis,
                is_blocked=is_blocked,
                cell_cost=cell_cost,
                dir_pref=dir_pref,
                dir_runs=dir_runs,
                via_blocked=via_blocked,
                via_cost=via_cost,
                window_slack=window_slack,
            )
        allowed = set(layer_gis)
        tgt_ix, tgt_iy, tgt_gi = target

        def h(ix: int, iy: int, gi: int) -> float:
            # STEP 3, cost-aware admissible heuristic: the in-plane Manhattan term
            # (each step >= 1) PLUS the via term priced at the REAL per-via cost
            # (_VIA_PENALTY, not 1.0). Each layer change costs exactly _VIA_PENALTY
            # in _neighbours and disjoint moves change either (x,y) or gi (never
            # both), so the sum is a true lower bound -> admissible AND consistent
            # (h drops by <= step cost across every edge), so the closed set stays
            # optimal. Congestion is deliberately NOT added (it is not a lower bound).
            return (float(abs(ix - tgt_ix) + abs(iy - tgt_iy))
                    + abs(gi - tgt_gi) * _VIA_PENALTY)

        # priority queue of (f, -g, g, cell): ``-g`` is the goal-directed tie-break
        # (STEP 1) and ``g`` is the node's real cost, still carried through the heap
        # for the relaxation (``tentative = g + step_cost``). came_from for path
        # reconstruction.
        open_heap: List[Tuple[float, float, float, Cell]] = []
        g_score: Dict[Cell, float] = {}
        came_from: Dict[Cell, Optional[Cell]] = {}
        for s in sources:
            # a source cell must itself be usable (on an allowed layer, not an
            # obstacle). If a seed sits on a blocked cell we still allow it as a
            # START (the net already owns it); blockage is checked on NEIGHBOURS.
            if s[2] not in allowed:
                continue
            g_score[s] = 0.0
            came_from[s] = None
            # tie-break (STEP 1, goal-directed): among equal-f nodes the heap pops
            # the SMALLEST second key, so we store ``-g`` -> equal-f ties expand the
            # LARGEST g (== smallest h == closest to the goal) first, collapsing the
            # equal-cost Manhattan plateau toward the target instead of flooding it.
            # Optimality is untouched (consistent h + closed set); only order changes.
            # The final tuple element (the cell) stays the deterministic last tiebreak.
            heapq.heappush(open_heap, (h(*s), -0.0, 0.0, s))
        if not open_heap:
            return None

        # STEP 2 per-net window: confine the search to the (ix,iy) bounding box of
        # the seeds + target, inflated by ``window_slack`` tracks on every side.
        # ``None`` -> no box (unbounded, byte-identical). The target sits INSIDE the
        # box by construction, so it is never dropped -> the search can always reach
        # the goal; only far-field detour cells are pruned.
        win = None
        if window_slack is not None:
            bxs = [s[0] for s in sources]
            bys = [s[1] for s in sources]
            bxs.append(tgt_ix)
            bys.append(tgt_iy)
            win = (min(bxs) - window_slack, min(bys) - window_slack,
                   max(bxs) + window_slack, max(bys) + window_slack)

        expansions = 0
        chk = _active_checkpoint_cb()
        if chk is not None:
            chk()
        closed: Set[Cell] = set()
        while open_heap:
            f, _neg_g, g, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == target:
                return self._reconstruct(came_from, cur)
            expansions += 1
            if expansions > self.max_expansions:
                return None  # bounded effort exhausted -> honest no-path
            if chk is not None and (expansions & (_CHECKPOINT_INTERVAL - 1)) == 0:
                chk()
            cix, ciy, cgi = cur
            for (nix, niy, ngi, step_cost) in self._neighbours(
                grid, tech, cur, allowed, dir_pref=dir_pref,
                via_blocked=via_blocked, via_cost=via_cost,
            ):
                nxt = (nix, niy, ngi)
                if nxt in closed:
                    continue
                if win is not None and (
                        nix < win[0] or nix > win[2]
                        or niy < win[1] or niy > win[3]):
                    continue  # STEP 2: outside the per-net window -> pruned
                # the target is always enterable (it is the net's own pin cell);
                # every other cell must be free of obstacles + other nets.
                if nxt != target and is_blocked(nix, niy, ngi):
                    continue
                tentative = g + step_cost
                if cell_cost is not None and nxt != target:
                    tentative += cell_cost(nix, niy, ngi)
                if tentative < g_score.get(nxt, math.inf):
                    g_score[nxt] = tentative
                    came_from[nxt] = cur
                    # (f, -g, g, cell): -tentative = goal-directed tie-break (STEP 1),
                    # tentative = the real cost carried through for the next relax.
                    heapq.heappush(open_heap, (tentative + h(nix, niy, ngi),
                                               -tentative, tentative, nxt))
        return None

    def _neighbours(
        self,
        grid: _Grid,
        tech: _Tech,
        cur: Cell,
        allowed: Set[int],
        dir_pref: Optional[Dict[int, Tuple[str, float]]] = None,
        via_blocked=None,
        via_cost=None,
    ):
        """Yield ``(ix, iy, gi, step_cost)`` neighbours: the 4 in-plane moves
        (cost 1 + wrong-direction penalty) and the up/down via moves between
        graph-adjacent allowed layers (cost ``_VIA_PENALTY``).

        ``via_blocked`` (optional ``fn(ix, iy, gi_low) -> bool``, Task D2) HARD
        FORBIDS a layer-transition whose via LANDING is pad-infeasible -- the
        SAME :func:`clay.layout.route.via_model.via_landing_feasible` verdict the
        emit obligation enforces, so the search stops committing a via cell the
        emit path would then suppress (F1). Applied to the via move between
        ``gi`` and ``ngi`` at ``(ix, iy)`` keyed on ``gi_low = min(gi, ngi)``: a
        forbidden move is simply NOT yielded, so PathFinder rip-up reroutes
        around it. It seals ONLY the pad-illegal transition; in-plane wire
        legality is unchanged. Default None -> byte-identical to the pre-D2 cost.

        ``dir_pref`` (``{gi: (axis, weight)}``, from Direction constraints)
        REPLACES the built-in parity prior on that layer: the commanded axis
        steps at the BASE cost (no parity tax) and only the anti-command axis
        pays ``weight x base step cost``. The parity prior is itself only a
        DEFAULT direction (the same override principle as lower.py's
        conventional table), so an explicit command supersedes it -- STACKING
        them double-charges the commanded axis wherever the command disagrees
        with parity (the measured test4 collapse: commands V,H,V vs parity
        H,V,H made every commanded axis pay +0.5 while the parity-aligned
        spare layer was the only tax-free channel). None / absent layer (and
        weight=0 Direction, which never enters ``dir_pref``) -> the parity
        prior applies unchanged, byte-identical cost."""
        ix, iy, gi = cur
        # in-plane moves
        horiz_pref = tech.preferred_horizontal(gi)
        pref = dir_pref.get(gi) if dir_pref else None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nix, niy = ix + dx, iy + dy
            if not grid.in_bounds(nix, niy):
                continue
            is_horiz = dx != 0
            if pref is not None:
                # explicit Direction: the layer's ONLY direction cost.
                penalty = pref[1] * 1.0 if is_horiz != (pref[0] == "H") else 0.0
            else:
                penalty = 0.0 if (is_horiz == horiz_pref) else _WRONG_DIR_PENALTY
            yield (nix, niy, gi, 1.0 + penalty)
        # via moves (only between graph-adjacent layers that are both allowed)
        for ngi in (gi - 1, gi + 1):
            if 0 <= ngi < tech.n_layers and ngi in allowed:
                gl = min(gi, ngi)
                if via_blocked is not None and via_blocked(ix, iy, gl):
                    continue  # D2: pad-infeasible transition -> emit would suppress
                v_cost = _VIA_PENALTY
                if via_cost is not None:
                    v_cost += max(0.0, float(via_cost(ix, iy, gl)))
                yield (ix, iy, ngi, v_cost)

    @staticmethod
    def _reconstruct(came_from: Dict[Cell, Optional[Cell]], cur: Cell) -> List[Cell]:
        path = [cur]
        while True:
            prev = came_from.get(cur)
            if prev is None:
                break
            path.append(prev)
            cur = prev
        path.reverse()
        return path

    # ------------------------------------------------------------ directed A*
    def _astar_directed(
        self,
        *,
        grid: _Grid,
        tech: _Tech,
        sources: Set[Cell],
        target: Cell,
        layer_gis: List[int],
        is_blocked,
        cell_cost=None,
        dir_pref: Dict[int, Tuple[str, float]],
        dir_runs: Optional[Dict[Cell, int]] = None,
        via_blocked=None,
        via_cost=None,
        window_slack: Optional[int] = None,
    ) -> Optional[List[Cell]]:
        """The run-extent-aware A* used whenever an explicit Direction is in
        force (the comb contract; see the module-level comment at
        ``_DIR_RUN_FREE``). Identical to :meth:`_astar` except the search state
        carries the CUMULATIVE anti-command run length ``r`` on the current
        layer visit (capped at ``_DIR_RUN_FREE + 1`` -- beyond the allowance
        every further anti step costs the same escalation, so larger counts
        are equivalent states):

          * an anti-command in-plane step on a commanded layer increments
            ``r`` and, beyond the jog allowance, pays
            ``weight x _DIR_RUN_PENALTY`` ON TOP of the per-step
            ``weight x 1`` -- runs get dearer than a via pair, jogs stay free;
          * a commanded-axis step keeps ``r`` (cumulative since layer entry:
            zigzagging cannot launder a long anti run, and straight runs stay
            optimal because wiggles add plain steps without resetting);
          * a via resets ``r`` to 0 (climbing to the commanded layer IS the
            contract's preferred escape);
          * source cells seed ``r`` from ``dir_runs`` (the net's existing
            same-layer connected extent -- tree memory), so the Nth short hop
            of a daisy chain is priced like the long comb it extends;
          * layers WITHOUT an explicit command keep the parity prior at
            ``r = 0`` (the escalation is part of the Direction contract only).

        All extra costs are non-negative -> the Manhattan heuristic stays
        admissible; the search is deterministic (integer-tuple states)."""
        allowed = set(layer_gis)
        tgt_ix, tgt_iy, tgt_gi = target
        cap = _DIR_RUN_FREE + 1

        def h(ix: int, iy: int, gi: int) -> float:
            # STEP 3 cost-aware admissible heuristic (see _astar): via term priced
            # at _VIA_PENALTY. Still a lower bound on the directed cost (the extra
            # anti-command run escalation is non-negative), so A* stays optimal.
            return (float(abs(ix - tgt_ix) + abs(iy - tgt_iy))
                    + abs(gi - tgt_gi) * _VIA_PENALTY)

        State = Tuple[int, int, int, int]  # (ix, iy, gi, run)
        # (f, -g, g, state): -g goal-directed tie-break (STEP 1), g carried for relax.
        open_heap: List[Tuple[float, float, float, State]] = []
        g_score: Dict[State, float] = {}
        came_from: Dict[State, Optional[State]] = {}
        for s in sources:
            if s[2] not in allowed:
                continue
            r0 = 0
            if dir_runs and s[2] in dir_pref:
                r0 = min(int(dir_runs.get(s, 0)), cap)
            st = (s[0], s[1], s[2], r0)
            g_score[st] = 0.0
            came_from[st] = None
            # goal-directed equal-f tie-break (STEP 1); see _astar. -g stored so the
            # heap pops largest-g (closest to goal) first; state stays final tiebreak.
            heapq.heappush(open_heap, (h(s[0], s[1], s[2]), -0.0, 0.0, st))
        if not open_heap:
            return None

        # STEP 2 per-net window (see _astar): confine to the seeds+target (ix,iy)
        # bbox inflated by ``window_slack`` tracks. None -> unbounded, byte-identical.
        win = None
        if window_slack is not None:
            bxs = [s[0] for s in sources]
            bys = [s[1] for s in sources]
            bxs.append(tgt_ix)
            bys.append(tgt_iy)
            win = (min(bxs) - window_slack, min(bys) - window_slack,
                   max(bxs) + window_slack, max(bys) + window_slack)

        expansions = 0
        chk = _active_checkpoint_cb()
        if chk is not None:
            chk()
        closed: Set[State] = set()
        while open_heap:
            f, _neg_g, g, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            cix, ciy, cgi, crun = cur
            if (cix, ciy, cgi) == target:
                return self._reconstruct_directed(came_from, cur)
            expansions += 1
            if expansions > self.max_expansions:
                return None  # bounded effort exhausted -> honest no-path
            if chk is not None and (expansions & (_CHECKPOINT_INTERVAL - 1)) == 0:
                chk()
            pref = dir_pref.get(cgi)
            horiz_pref = tech.preferred_horizontal(cgi)
            # in-plane moves
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nix, niy = cix + dx, ciy + dy
                if not grid.in_bounds(nix, niy):
                    continue
                if win is not None and (
                        nix < win[0] or nix > win[2]
                        or niy < win[1] or niy > win[3]):
                    continue  # STEP 2: outside the per-net window (via moves keep
                    #            x/y, so bounding only the planar moves suffices)
                is_horiz = dx != 0
                if pref is not None:
                    ax, w = pref
                    if is_horiz != (ax == "H"):
                        nrun = min(crun + 1, cap)
                        penalty = w * 1.0
                        if nrun > _DIR_RUN_FREE:
                            penalty += w * _DIR_RUN_PENALTY
                    else:
                        nrun = crun
                        penalty = 0.0
                else:
                    nrun = 0
                    penalty = 0.0 if (is_horiz == horiz_pref) else _WRONG_DIR_PENALTY
                self._relax_directed(
                    grid, target, is_blocked, cell_cost, open_heap, g_score,
                    came_from, closed, h, cur, g,
                    (nix, niy, cgi, nrun), 1.0 + penalty,
                )
            # via moves (run resets: a layer change starts a fresh visit)
            for ngi in (cgi - 1, cgi + 1):
                if 0 <= ngi < tech.n_layers and ngi in allowed:
                    gl = min(cgi, ngi)
                    if via_blocked is not None and via_blocked(cix, ciy, gl):
                        continue  # D2: pad-infeasible transition (see _neighbours)
                    v_cost = _VIA_PENALTY
                    if via_cost is not None:
                        v_cost += max(0.0, float(via_cost(cix, ciy, gl)))
                    self._relax_directed(
                        grid, target, is_blocked, cell_cost, open_heap, g_score,
                        came_from, closed, h, cur, g,
                        (cix, ciy, ngi, 0), v_cost,
                    )
        return None

    @staticmethod
    def _relax_directed(
        grid, target, is_blocked, cell_cost, open_heap, g_score, came_from,
        closed, h, cur, g, nxt, step_cost,
    ) -> None:
        """One edge relaxation of the directed search (same blockage / target /
        cell_cost semantics as the base loop, on run-augmented states)."""
        if nxt in closed:
            return
        nix, niy, ngi, _nr = nxt
        cell = (nix, niy, ngi)
        if cell != target and is_blocked(nix, niy, ngi):
            return
        tentative = g + step_cost
        if cell_cost is not None and cell != target:
            tentative += cell_cost(nix, niy, ngi)
        if tentative < g_score.get(nxt, math.inf):
            g_score[nxt] = tentative
            came_from[nxt] = cur
            # (f, -g, g, state): -tentative goal-directed tie-break (STEP 1),
            # tentative = real cost carried through for the next relaxation.
            heapq.heappush(open_heap, (tentative + h(nix, niy, ngi),
                                       -tentative, tentative, nxt))

    @staticmethod
    def _reconstruct_directed(
        came_from: Dict[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]],
        cur: Tuple[int, int, int, int],
    ) -> List[Cell]:
        """Path reconstruction over run-augmented states, projected to cells."""
        path = [(cur[0], cur[1], cur[2])]
        while True:
            prev = came_from.get(cur)
            if prev is None:
                break
            path.append((prev[0], prev[1], prev[2]))
            cur = prev
        path.reverse()
        return path

    @staticmethod
    def _update_dir_runs(
        runs: Dict[Cell, int],
        tree: Set[Cell],
        new_cells,
        dir_pref: Optional[Dict[int, Tuple[str, float]]],
    ) -> None:
        """Maintain the tree-memory side of the comb contract: for every tree
        cell on a COMMANDED layer, ``runs[cell]`` is the anti-command-axis
        EXTENT (in grid cells) of the same-layer connected component of the
        net's metal containing it. A new branch growing off that component
        seeds its run length from this extent, so a daisy chain's accumulated
        anti travel is visible across separate A* calls. Incremental: only the
        components touched by ``new_cells`` are re-walked (BFS through
        ``tree``)."""
        if not dir_pref:
            return
        seen: Set[Cell] = set()
        for c in new_cells:
            if c[2] not in dir_pref or c in seen:
                continue
            comp: List[Cell] = [c]
            seen.add(c)
            stack: List[Cell] = [c]
            while stack:
                x, y, g = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (x + dx, y + dy, g)
                    if n in tree and n not in seen:
                        seen.add(n)
                        comp.append(n)
                        stack.append(n)
            axis = dir_pref[c[2]][0]
            if axis == "H":  # commanded H -> anti axis is V -> y extent
                ext = max(n[1] for n in comp) - min(n[1] for n in comp)
            else:            # commanded V -> anti axis is H -> x extent
                ext = max(n[0] for n in comp) - min(n[0] for n in comp)
            for n in comp:
                runs[n] = ext

    # ------------------------------------------------------------------ emit
    def _emit_wires(
        self,
        grid: _Grid,
        tech: _Tech,
        wire_cells: Set[Cell],
        pin_cells: List[Cell],
        wire_width_nm: int,
        width_at=None,
        foreign_rects=None,
        cell_taken=None,
        reloc_cells=None,
        foreign_metal_rects=None,
        suppressed_out=None,
        routing_envelope_nm: Optional[object] = None,
    ) -> Tuple[List[LayerPolygon], float]:
        """Turn the routed cell-set into CONTINUOUS LayerPolygon geometry (the
        toolchain's wire currency): a width-sized PAD at each occupied cell PLUS a
        BRIDGE rectangle spanning every pair of in-plane-adjacent cells, on the
        cell's metal key, plus a small via-cut square wherever the net changes layer
        at an (x,y). Returns ``(polys, length_um)``.

        The bridges are essential: a pad alone is ``width`` across, but the grid
        pitch is ``min_width + min_space`` > min_width, so min-width PADS DO NOT
        ABUT (they leave a min-space gap) -- pads-only emit is geometrically OPEN
        for any width below the pitch (i.e. the common min-width wire). The bridge
        between two adjacent cells fills that gap, so the wire is continuous at ANY
        width.

        If ``width_at`` (a ``fn(ix, iy, gi) -> width_nm``) is given, each cell/bridge
        uses its LOCAL width -- wire TAPERING (necked near pins where terminals are
        tight, full trunk in the open channel). Else the constant ``wire_width_nm``
        is used everywhere (pads emitted first, so ``wires[0]`` is a full-width pad).

        ``foreign_rects`` (optional) ``Dict[(layer, datatype) -> List[(x0,y0,x1,y1)]]``
        (nm) holds the FOREIGN net-aware keep-out rectangles -- cap plates, other
        devices' pads -- that this net does NOT own. Every emitted rectangle (pad,
        bridge, via cut) is CLIPPED against the foreign rects on its own layer before
        it is kept, so no foreign net's metal/via can graze a keep-out (a cap plate)
        even when the grid search seeded a pin cell that sat on one. The A* already
        kept the routed CHANNEL off the keep-out (it is hard-blocked on neighbours);
        the clip only removes the seed-cell pad / widened pad / via overrun that the
        per-cell emit would otherwise paint over the plate. Without it the keep-out
        could not suppress the net's OWN seed pad. When None (the default) emit is
        byte-identical to the un-clipped behaviour.

        VIA ENVELOPE OBLIGATION (the clip/via coupling): a via cut is only emitted
        where the POST-CLIP metal on BOTH layers it bridges still covers the cut
        plus its PDK-rule enclosure (clamped to the enclosure the UNCLIPPED pad of
        the local width provides -- the clip must never make the envelope WORSE
        than designed; an under-designed pad is the deck's account). A via whose
        envelope the clip broke (shaved pad -> the composite x trunk VIA1.EN.1) or
        swallowed whole (the legacy silent drop -> the net07 geometric open) is
        RELOCATED to a nearby cell on the net's own route -- extension metal that
        itself passes the clip WHOLE -- or, when no legal relocation exists,
        suppressed WHOLE (honest breakage the geometric-open oracle reports).
        Partial via cuts are never emitted.

        ``cell_taken`` (optional ``fn(ix, iy, gi, width_nm) -> bool``) answers
        whether a relocation extension of ``width_nm`` at a cell would conflict
        with ANOTHER net's metal/halo (the engines pass their committed/occupancy
        view); None = treat every cell as free (hermetic unit-test worlds).
        ``reloc_cells`` (optional list) collects ``(ix, iy, gi, width_nm)`` for
        every extension cell drawn, so the caller can commit them as the net's
        metal (blockage for later nets / the short oracle).

        ``suppressed_out`` (optional list, Task D3) collects ``(ix, iy, gi_low)``
        for every via the envelope obligation SUPPRESSED WHOLE (no cut emitted,
        no legal relocation). The caller uses it as the TRIGGER for a via-aware
        emitted-geometry openness check and downgrades the net's ``connected``
        flag ONLY when the suppression actually severed the net (native<->oracle
        fidelity: the flag must equal the emitted geometry -- a via the net routes
        around must not falsely downgrade). None -> not recorded (byte-identical)."""
        def _w(ix: int, iy: int, gi: int) -> int:
            return int(width_at(ix, iy, gi)) if width_at is not None else wire_width_nm

        # per-layer metal min-width (nm) for the residual-sliver filter in _emit.
        _min_w_of_key: Dict[LayerKey, int] = {}
        for _gi in range(tech.n_layers):
            try:
                _min_w_of_key[tech.gi_to_metal_key(_gi)] = int(tech.gi_min_width(_gi))
            except Exception:
                pass

        # post-clip KEPT metal rects per layer key -- the via envelope obligation's
        # coverage basis (exactly what this call emits, nothing else).
        kept_rects: Dict[LayerKey, List[_Rect]] = {}
        native_clipper = None
        if foreign_rects:
            _pf_mod = load_native("_pathfinder_cpp")
            if _pf_mod is not None and hasattr(_pf_mod, "NativeRectClipper"):
                native_clipper = _pf_mod.NativeRectClipper(foreign_rects)

        def _emit(polys: List[LayerPolygon], key: LayerKey, x0: int, y0: int,
                  x1: int, y1: int) -> None:
            """Append the rectangle (key, x0..y1) MINUS any foreign keep-out on the
            same layer (split into residual rectangles). A rectangle entirely inside
            a foreign keep-out vanishes (its metal would short the plate).

            RESIDUAL HYGIENE: a clip fragment whose SHORT side is below the layer's
            metal min-width is DROPPED -- the min-space-inflated clip can chop a
            wire into sub-min slivers (a real KLayout M1.S.1/M1.W case: two 75 nm
            scraps facing each other 166 nm apart), which are DRC-illegal by
            construction and electrically useless. Dropping can only OPEN a net,
            never short it, and the connectivity / geometric-open oracles report
            any such open honestly."""
            holes = None if foreign_rects is None else foreign_rects.get(key)
            if not holes:
                polys.append(LayerPolygon(layer=key[0], datatype=key[1],
                                          polygon=Polygon.rectangle(x0, y0, x1, y1)))
                kept_rects.setdefault(key, []).append((x0, y0, x1, y1))
                return
            min_w = _min_w_of_key.get(key, 1)
            if native_clipper is not None:
                keep = native_clipper.clip_emit(key, x0, y0, x1, y1, min_w)
                if keep:
                    kb = kept_rects.setdefault(key, [])
                    kl, kd = key[0], key[1]
                    for (sx0, sy0, sx1, sy1) in keep:
                        polys.append(LayerPolygon(layer=kl, datatype=kd,
                                                  polygon=Polygon.rectangle(sx0, sy0, sx1, sy1)))
                        kb.append((sx0, sy0, sx1, sy1))
                return
            full_w = min(x1 - x0, y1 - y0)
            # a rect already narrower than min-width (a sub-pitch bridge sliver the
            # grid legitimately emits) keeps its own width as the floor instead.
            floor = min(min_w, max(1, full_w))

            def _fragments(rx0, ry0, rx1, ry1):
                keep, dropped = [], False
                for (sx0, sy0, sx1, sy1) in _subtract_rects((rx0, ry0, rx1, ry1),
                                                            holes):
                    if min(sx1 - sx0, sy1 - sy0) < floor:
                        dropped = True
                        continue
                    keep.append((sx0, sy0, sx1, sy1))
                return keep, dropped

            keep, dropped = _fragments(x0, y0, x1, y1)
            if dropped and full_w > min_w:
                # WIDEN-vs-CLIP severing fix: the search reserved space for the
                # MIN width; widen/taper fattened this rect beyond it and the
                # min-space-inflated foreign clip then nibbled it into sub-min
                # scraps -- dropping those scraps SEVERED the net (a real LVS
                # orphan drain island, and the rails-path net4 open). Retry at
                # the layer min width about the same centreline, and if a hole
                # from one side still nibbles the centred spine, also check
                # edge-aligned min-width spines inside the original rect.
                candidate_spines = []
                if (x1 - x0) >= (y1 - y0):
                    mid = (y0 + y1) // 2
                    cy0 = mid - min_w // 2
                    candidate_spines.append((x0, cy0, x1, cy0 + min_w))
                    candidate_spines.append((x0, y0, x1, y0 + min_w))
                    candidate_spines.append((x0, y1 - min_w, x1, y1))
                else:
                    mid = (x0 + x1) // 2
                    cx0 = mid - min_w // 2
                    candidate_spines.append((cx0, y0, cx0 + min_w, y1))
                    candidate_spines.append((x0, y0, x0 + min_w, y1))
                    candidate_spines.append((x1 - min_w, y0, x1, y1))
                keep2, d2 = _fragments(*candidate_spines[0])
                if d2:
                    for alt_spine in candidate_spines[1:]:
                        alt_keep, alt_drop = _fragments(*alt_spine)
                        if alt_keep and not alt_drop:
                            keep2 = keep2 + alt_keep
                            break
                # keep the union: surviving wide fragments + the min-width spine
                # (overlap is harmless -- same net, merged downstream).
                keep = keep + keep2
            for (sx0, sy0, sx1, sy1) in keep:
                polys.append(LayerPolygon(layer=key[0], datatype=key[1],
                                          polygon=Polygon.rectangle(sx0, sy0, sx1, sy1)))
                kept_rects.setdefault(key, []).append((sx0, sy0, sx1, sy1))

        # via map: (ix,iy) -> the set of graph layers this net occupies there,
        # so a cell carrying a via (two graph-adjacent layers) can be sized at its
        # LANDING-PAD width for bridges (the same-net via-pad-shoulder contract).
        by_xy: Dict[Tuple[int, int], Set[int]] = {}
        for (ix, iy, gi) in wire_cells:
            by_xy.setdefault((ix, iy), set()).add(gi)

        def _via_pad_half_on(ix: int, iy: int, gi: int) -> int:
            """The via LANDING-PAD half (nm) emitted on layer ``gi`` at cell
            ``(ix,iy)`` -- ``(cut+1)//2 + enclosure`` for a via that lands on ``gi``
            (below side of via ``gi<->gi+1`` and/or above side of ``gi-1<->gi``),
            or 0 when no via touches ``gi`` here. The same dimension
            ``_emit_landing`` draws, so a bridge sized to it joins two pads
            flush. PDK-sourced (cut + enclosure); on a PDK whose wire already
            covers the cut the pad is no wider than the wire (max() below ignores
            it) -> byte-identical."""
            gis = by_xy.get((ix, iy))
            if not gis:
                return 0
            best = 0
            for (gl, is_below) in ((gi, True), (gi - 1, False)):
                if gl < 0 or (gl + 1) >= tech.n_layers:
                    continue
                if gl not in gis or (gl + 1) not in gis:
                    continue  # no via on this side here
                try:
                    rb, ra = self._via_env_rule_nm(tech, gl)
                    vsz = _via_cut_size_nm(tech, gl)
                except Exception:
                    continue
                enc = rb if is_below else ra
                if enc <= 0 or not vsz:
                    continue
                best = max(best, (int(vsz) + 1) // 2 + int(enc))
            return best

        def _eff_half(ix: int, iy: int, gi: int) -> int:
            """Effective same-net half-width at this cell on ``gi`` for BRIDGE
            sizing: the wire half raised to the via landing-pad half when a via
            here drops an EXTRA pad on ``gi`` (pad wider than the wire). A bridge
            between two via cells is then drawn at pad width, so their pad
            shoulders MERGE -- no same-net re-entrant notch < min_space between two
            adjacent landing pads (the gf180 M{k}.2a via-pad-shoulder family).
            The widened bridge is collinear with the two pads it joins (the pads
            already reach that far), so it adds NO new reach toward any foreign
            net -- same-net, additive, short-safe."""
            return max(_w(ix, iy, gi) // 2, _via_pad_half_on(ix, iy, gi))

        polys: List[LayerPolygon] = []
        # per-cell pads (emitted FIRST so wires[0] is a full-width pad)
        for (ix, iy, gi) in sorted(wire_cells):
            half = max(1, _w(ix, iy, gi) // 2)
            cx, cy = grid.cell_xy(ix, iy)
            key = tech.gi_to_metal_key(gi)
            _emit(polys, key, cx - half, cy - half, cx + half, cy + half)
        # bridges between in-plane-adjacent cells -> continuity below the pitch.
        # Sized to the NARROWER endpoint's via-pad-aware effective half (not the
        # bare wire half), so two adjacent via landing pads are joined by pad-width
        # metal and leave no same-net inter-pad notch.
        for (ix, iy, gi) in sorted(wire_cells):
            cx, cy = grid.cell_xy(ix, iy)
            key = tech.gi_to_metal_key(gi)
            for (jx, jy) in ((ix + 1, iy), (ix, iy + 1)):
                if (jx, jy, gi) not in wire_cells:
                    continue
                half = max(1, min(_eff_half(ix, iy, gi), _eff_half(jx, jy, gi)))
                nx, ny = grid.cell_xy(jx, jy)
                if cy == ny:  # horizontal bridge
                    _emit(polys, key, min(cx, nx), cy - half, max(cx, nx), cy + half)
                else:  # vertical bridge
                    _emit(polys, key, cx - half, min(cy, ny), cx + half, max(cy, ny))
        # via cuts: any (x,y) occupied on two graph-adjacent layers, each under
        # the VIA ENVELOPE OBLIGATION (see the docstring): emit at the cell only
        # if the post-clip metal still holds the cut's clamped enclosure on BOTH
        # layers; else relocate along the net's own cells; else suppress WHOLE.
        # (``by_xy`` was built above for the via-pad-aware bridge sizing.)
        for (ix, iy) in sorted(by_xy):
            gis = by_xy[(ix, iy)]
            for gi in sorted(gis):
                if (gi + 1) in gis:
                    if tech.via_key_for(gi) is None:
                        continue
                    self._emit_via_obligated(
                        grid, tech, polys, kept_rects, wire_cells, ix, iy, gi,
                        _w, foreign_rects, cell_taken, reloc_cells,
                        foreign_metal_rects, suppressed_out,
                        routing_envelope_nm=routing_envelope_nm,
                    )
        # length = number of in-plane track-steps * pitch (a proxy that does not
        # double count vias). Count distinct adjacency along the tree.
        length_nm = self._wire_length_nm(grid, wire_cells)
        return polys, length_nm / 1000.0

    def _repair_clipped_terminal_gaps(
        self,
        tech: _Tech,
        net: str,
        wires: List[LayerPolygon],
        pins: Sequence[Pin],
        obs_by_gi: Dict[int, Sequence[Tuple[float, float, float, float, Optional[str]]]],
        foreign_rects: Optional[Dict[LayerKey, Sequence[_Rect]]],
        foreign_metal_rects: Optional[Dict[LayerKey, Sequence[_Rect]]],
        *,
        routing_envelope_nm: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[LayerKey, List[_Rect]]:
        """Bridge native terminal pads left disconnected by min-space clipping.

        When a widened or off-centre wire near a tight foreign pad is clipped
        exact ``min_space`` short of an owned terminal pad that itself sits
        legally clear of the foreign pad, emit a minimal DRC-clean Manhattan
        connector (validated against foreign keep-outs and KLayout Euclidean
        ``width_check`` / ``space_check``) so the terminal does not fail open.
        """
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

        if not wires or not pins:
            return {}

        def _wire_bboxes_by_key() -> Dict[LayerKey, List[_Rect]]:
            out_b: Dict[LayerKey, List[_Rect]] = {}
            for poly in wires:
                k = (int(poly.layer), int(poly.datatype))
                vs = poly.polygon.vertices
                if not vs:
                    continue
                xs = [int(v.x) for v in vs]
                ys = [int(v.y) for v in vs]
                out_b.setdefault(k, []).append((min(xs), min(ys), max(xs), max(ys)))
            return out_b

        def _fast_connected(w_boxes: Sequence[_Rect], t_boxes: Sequence[_Rect], min_w: int) -> bool:
            for wx0, wy0, wx1, wy1 in w_boxes:
                for tx0, ty0, tx1, ty1 in t_boxes:
                    ox = min(wx1, tx1) - max(wx0, tx0)
                    oy = min(wy1, ty1) - max(wy0, ty0)
                    if ox > 0 and oy > 0:
                        return True
                    if (ox == 0 and oy >= min_w) or (oy == 0 and ox >= min_w):
                        return True
            return False

        wire_bboxes = _wire_bboxes_by_key()
        missing_by_key: Dict[LayerKey, List[Tuple[int, int, int, List[_Rect]]]] = {}
        for p in pins:
            key = tuple(p.layer)
            gi = tech.resolve_layer_to_gi(key)
            if gi is None:
                continue
            px = int(round(float(p.xy_um[0]) * _NM_PER_UM))
            py = int(round(float(p.xy_um[1]) * _NM_PER_UM))
            owned = [
                (
                    int(round(min(x0, x1) * _NM_PER_UM)),
                    int(round(min(y0, y1) * _NM_PER_UM)),
                    int(round(max(x0, x1) * _NM_PER_UM)),
                    int(round(max(y0, y1) * _NM_PER_UM)),
                )
                for (x0, y0, x1, y1, own) in obs_by_gi.get(gi, ())
                if own is not None and own == net
                and int(round(min(x0, x1) * _NM_PER_UM)) - 1 <= px <= int(round(max(x0, x1) * _NM_PER_UM)) + 1
                and int(round(min(y0, y1) * _NM_PER_UM)) - 1 <= py <= int(round(max(y0, y1) * _NM_PER_UM)) + 1
            ]
            if not owned:
                continue
            min_w = max(1, int(tech.gi_min_width(gi)))
            if _fast_connected(wire_bboxes.get(key, ()), owned, min_w):
                continue
            missing_by_key.setdefault(key, []).append((gi, px, py, owned))

        if not missing_by_key:
            return {}

        import klayout.db as kdb

        mfg_grid = max(1, int(pdk_manufacturing_grid_nm(tech.pdk)))
        added_by_key: Dict[LayerKey, List[_Rect]] = {}

        def _snap_up(val: int) -> int:
            return ((val + mfg_grid - 1) // mfg_grid) * mfg_grid

        def _box_ok_env(bx0: int, by0: int, bx1: int, by1: int) -> bool:
            if bx1 <= bx0 or by1 <= by0:
                return False
            if routing_envelope_nm is None:
                return True
            ex0, ey0, ex1, ey1 = routing_envelope_nm
            return bx0 >= ex0 and by0 >= ey0 and bx1 <= ex1 and by1 <= ey1

        for key, items in missing_by_key.items():
            own_reg = kdb.Region()
            for poly in wires:
                if (int(poly.layer), int(poly.datatype)) == key:
                    pts = [kdb.Point(int(v.x), int(v.y)) for v in poly.polygon.vertices]
                    if len(pts) >= 3:
                        own_reg.insert(kdb.Polygon(pts))
            own_reg.merge()
            if own_reg.is_empty():
                continue

            blocked_boxes: List[_Rect] = []
            blocked_reg = kdb.Region()
            for bucket in (foreign_rects, foreign_metal_rects):
                if bucket and key in bucket:
                    for bx0, by0, bx1, by1 in bucket[key]:
                        if bx1 > bx0 and by1 > by0:
                            blocked_boxes.append((int(bx0), int(by0), int(bx1), int(by1)))
                            blocked_reg.insert(kdb.Box(int(bx0), int(by0), int(bx1), int(by1)))
            blocked_reg.merge()

            for gi, px, py, owned in items:
                target_reg = kdb.Region()
                for ox0, oy0, ox1, oy1 in owned:
                    target_reg.insert(kdb.Box(ox0, oy0, ox1, oy1))
                target_reg.merge()
                min_w = max(1, int(tech.gi_min_width(gi)))
                min_s = max(0, int(tech.min_space_nm[tech._gi_to_pdk[gi]]))
                if ((own_reg & target_reg).area() > 0
                        or (own_reg.sized(1) & target_reg.sized(1)).area() >= 2 * min_w):
                    continue

                bw = max(min_w, _snap_up(min_w))
                tbox = target_reg.bbox()
                max_gap = 4 * max(bw + min_s, 600)
                poly_entries: List[Tuple[int, kdb.Region, kdb.Box]] = []
                for poly in own_reg.each():
                    pbox = poly.bbox()
                    dx = max(0, pbox.left - tbox.right, tbox.left - pbox.right)
                    dy = max(0, pbox.bottom - tbox.top, tbox.bottom - pbox.top)
                    dist = dx + dy
                    if dist <= max_gap:
                        poly_entries.append((dist, kdb.Region(poly), pbox))
                if not poly_entries:
                    continue
                poly_entries.sort(key=lambda item: item[0])
                poly_entries = poly_entries[:8]

                base_combined = (own_reg + target_reg).merged()
                base_wc = base_combined.width_check(min_w).polygons(0)
                base_sc = (
                    base_combined.space_check(min_s, False, kdb.Region.Euclidian, None, None, None).polygons(0)
                    if min_s > 0 else kdb.Region()
                )

                # Local holes intersecting neighborhood of target+closest poly
                search_x0 = min([e[2].left for e in poly_entries] + [tbox.left]) - bw
                search_y0 = min([e[2].bottom for e in poly_entries] + [tbox.bottom]) - bw
                search_x1 = max([e[2].right for e in poly_entries] + [tbox.right]) + bw
                search_y1 = max([e[2].top for e in poly_entries] + [tbox.top]) + bw
                local_holes = [
                    b for b in blocked_boxes
                    if b[2] > search_x0 and b[0] < search_x1 and b[3] > search_y0 and b[1] < search_y1
                ]

                best_choice: Optional[Tuple[int, List[_Rect], kdb.Region]] = None

                def _try_candidate(cand_boxes: Sequence[_Rect], src_reg: kdb.Region) -> None:
                    nonlocal best_choice
                    if not all(_box_ok_env(*b) for b in cand_boxes):
                        return
                    cand_reg = kdb.Region()
                    for b in cand_boxes:
                        cand_reg.insert(kdb.Box(*b))
                    cand_reg.merge()
                    if not (cand_reg & blocked_reg).is_empty():
                        return
                    if (cand_reg & src_reg).area() <= 0 or (cand_reg & target_reg).area() <= 0:
                        return
                    added_area = int((cand_reg - own_reg).area())
                    if best_choice is not None and added_area >= best_choice[0]:
                        return
                    combined = (own_reg + cand_reg + target_reg).merged()
                    joined = False
                    for mpoly in combined.each():
                        mr = kdb.Region(mpoly)
                        if not (mr & target_reg).is_empty() and not (mr & src_reg).is_empty():
                            joined = True
                            break
                    if not joined:
                        return
                    if not (combined.width_check(min_w).polygons(0) - base_wc).is_empty():
                        return
                    if min_s > 0:
                        nsc = combined.space_check(min_s, False, kdb.Region.Euclidian, None, None, None).polygons(0)
                        if not (nsc - base_sc).is_empty():
                            return
                    best_choice = (added_area, list(cand_boxes), cand_reg)

                y_bases = {tbox.bottom, tbox.top - bw, py - bw // 2}
                x_bases = {tbox.left, tbox.right - bw, px - bw // 2}
                for hx0, hy0, hx1, hy1 in local_holes:
                    y_bases.update((hy1, hy0 - bw))
                    x_bases.update((hx1, hx0 - bw))

                for _dist, src_reg, pbox in poly_entries:
                    yb_all = sorted(y_bases | {pbox.bottom, pbox.top - bw})
                    xb_all = sorted(x_bases | {pbox.left, pbox.right - bw})
                    y_spans: List[Tuple[int, int]] = [(yb, yb + bw) for yb in yb_all]
                    for y_a, y_b in (
                        (tbox.bottom, pbox.top), (pbox.bottom, tbox.top), (tbox.bottom, tbox.top)
                    ):
                        if bw <= (y_b - y_a) <= 3 * bw:
                            y_spans.append((y_a, y_b))
                    x_spans: List[Tuple[int, int]] = [(xb, xb + bw) for xb in xb_all]
                    for x_a, x_b in (
                        (tbox.left, pbox.right), (pbox.left, tbox.right), (tbox.left, tbox.right)
                    ):
                        if bw <= (x_b - x_a) <= 3 * bw:
                            x_spans.append((x_a, x_b))

                    # 1. Single horizontal bridge (pbox left or right of tbox, or close)
                    hx_pairs: List[Tuple[int, int]] = []
                    if pbox.right <= tbox.right:
                        xl_cands = [max(pbox.left, min(pbox.right, tbox.left) - bw)]
                        if pbox.right - 2 * bw >= pbox.left:
                            xl_cands.append(pbox.right - 2 * bw)
                        xr_cands = [
                            min(tbox.right, max(tbox.left, pbox.right) + bw),
                            min(tbox.right, max(tbox.left + bw, px + bw // 2)),
                            tbox.right,
                        ]
                        for xl in xl_cands:
                            for xr in xr_cands:
                                hx_pairs.append((xl, xr))
                    if pbox.left >= tbox.left:
                        xl_cands = [tbox.left, max(tbox.left, px - bw // 2), max(tbox.left, min(pbox.left, tbox.right) - bw)]
                        xr_cands = [min(pbox.right, max(tbox.right, pbox.left) + bw)]
                        for xl in xl_cands:
                            for xr in xr_cands:
                                hx_pairs.append((xl, xr))
                    for xl, xr in hx_pairs:
                        if xr <= xl:
                            continue
                        for ya, yb in y_spans:
                            _try_candidate([(xl, ya, xr, yb)], src_reg)

                    # 2. Single vertical bridge
                    vy_pairs: List[Tuple[int, int]] = []
                    if pbox.top <= tbox.top:
                        yb_cands = [max(pbox.bottom, min(pbox.top, tbox.bottom) - bw)]
                        yt_cands = [
                            min(tbox.top, max(tbox.bottom, pbox.top) + bw),
                            min(tbox.top, max(tbox.bottom + bw, py + bw // 2)),
                            tbox.top,
                        ]
                        for yb in yb_cands:
                            for yt in yt_cands:
                                vy_pairs.append((yb, yt))
                    if pbox.bottom >= tbox.bottom:
                        yb_cands = [tbox.bottom, max(tbox.bottom, py - bw // 2), max(tbox.bottom, min(pbox.bottom, tbox.top) - bw)]
                        yt_cands = [min(pbox.top, max(tbox.top, pbox.bottom) + bw)]
                        for yb in yb_cands:
                            for yt in yt_cands:
                                vy_pairs.append((yb, yt))
                    for yb, yt in vy_pairs:
                        if yt <= yb:
                            continue
                        for xa, xb in x_spans:
                            _try_candidate([(xa, yb, xb, yt)], src_reg)

                    if best_choice is not None:
                        continue

                    # 3. Clean 2-leg Manhattan L-bend fallback
                    p_cx = (pbox.left + pbox.right) // 2
                    p_cy = (pbox.bottom + pbox.top) // 2
                    t_cx = min(tbox.right - bw, max(tbox.left, px - bw // 2))
                    t_cy = min(tbox.top - bw, max(tbox.bottom, py - bw // 2))
                    x_legs = sorted({max(pbox.left, min(pbox.right - bw, p_cx - bw // 2)), *xb_all})
                    y_legs = sorted({max(pbox.bottom, min(pbox.top - bw, p_cy - bw // 2)), *yb_all})
                    for xl in x_legs:
                        xr = xl + bw
                        for yt_b in (t_cy, tbox.bottom, tbox.top - bw):
                            yt_t = yt_b + bw
                            vy0 = min(pbox.bottom, yt_b)
                            vy1 = max(pbox.top, yt_t)
                            # Keep vertical leg overlapping pbox cleanly without stretching beyond pbox+tbox
                            vy0 = max(min(pbox.bottom, yt_b), min(pbox.top - bw, yt_b))
                            vy1 = min(max(pbox.top, yt_t), max(pbox.bottom + bw, yt_t))
                            hx0 = min(xl, tbox.left)
                            hx1 = max(xr, min(tbox.right, tbox.left + bw))
                            _try_candidate([(xl, vy0, xr, vy1), (hx0, yt_b, hx1, yt_t)], src_reg)
                    for yl in y_legs:
                        yr = yl + bw
                        for xt_l in (t_cx, tbox.left, tbox.right - bw):
                            xt_r = xt_l + bw
                            hx0 = max(min(pbox.left, xt_l), min(pbox.right - bw, xt_l))
                            hx1 = min(max(pbox.right, xt_r), max(pbox.left + bw, xt_r))
                            vy0 = min(yl, tbox.bottom)
                            vy1 = max(yr, min(tbox.top, tbox.bottom + bw))
                            _try_candidate([(hx0, yl, hx1, yr), (xt_l, vy0, xt_r, vy1)], src_reg)

                if best_choice is not None:
                    _, cand_boxes, cand_reg = best_choice
                    for bx0, by0, bx1, by1 in cand_boxes:
                        wires.append(
                            LayerPolygon(
                                layer=key[0],
                                datatype=key[1],
                                polygon=Polygon.rectangle(bx0, by0, bx1, by1),
                            )
                        )
                    added_by_key.setdefault(key, []).extend(cand_boxes)
                    own_reg = (own_reg + cand_reg).merged()

        return added_by_key

    # ---------------------------------------------------------- via obligation
    @staticmethod
    def _via_env_rule_nm(tech: _Tech, gi_low: int) -> Tuple[int, int]:
        """The via enclosure (nm) ``(below, above)`` the router must draw for the
        cut between ``gi_low`` and ``gi_low+1`` -- sourced from the SAME rule the
        signoff DRC deck consumes, via :func:`via_model.via_enclosure_nm`.

        The enclosure the router draws MUST equal the enclosure the oracle deck
        ENFORCES, otherwise the two diverge:
          * drawing MORE than the deck enforces (the old behaviour: it read the
            raw consolidated ``via_rule`` whose external_pdk via1.above/via2 carry a
            node-scaled 54nm overlap the SYNTHESIZED deck does not check) fattens
            wires the deck never asked to fatten -> same-layer M2.S/M3.S spacing
            violations and a moved envelope;
          * drawing LESS leaves real foundry enclosure violations (the gf180
            V1/V2 case this round fixes).
        ``via_model.via_enclosure_nm`` returns exactly the deck-enforced value on
        BOTH paths (external_pdk parsed-deck via{k}_enclosure_metal{j}; gf180 real foundry
        via_rule) with NO per-PDK branch, so the router's physical worldview is the
        PDK contract the oracle reads -- not a private over-broad number."""
        name = _via_name_for(tech, gi_low)
        if name is None:
            return 0, 0
        # ``via_model.via_enclosure_nm`` reads the PDK contract surface
        # (``drc_rules`` / ``via_rule`` / ``via_size``); the maze's ``_Tech`` is a
        # routing-only view that delegates that surface to the underlying
        # ``LayoutTech`` it holds as ``.pdk``.
        contract = getattr(tech, "pdk", tech)
        try:
            rb, ra = via_enclosure_nm(contract, name)
        except Exception:
            return 0, 0
        return int(rb or 0), int(ra or 0)

    @staticmethod
    def _clip_pad_to_min_space(
        pad: _Rect,
        holes: List[_Rect],
        *,
        cut_size: int,
        enclosure_nm: int,
        min_width_nm: int,
    ) -> Optional[_Rect]:
        """Clip ``pad`` on its FOREIGN-FACING side(s) so it no longer intersects any
        ``hole`` (a min-space-inflated foreign keep-out), WHILE preserving the via
        enclosure landing (the result must still cover the cut grown by
        ``enclosure_nm`` per side) AND the layer ``min_width`` (the result's short
        side must stay >= ``min_width_nm``). Returns the clipped rect, or ``None``
        when no clip can hold all three (the honest 3-way coupling -- the caller
        relocates/suppresses rather than emit an enclosure/width violation).

        The enclosure-landing FLOOR is the cut grown by ``enclosure_nm`` per side
        (``cut_size + 2*enclosure_nm`` square, centred on the cut). The pad centre
        is the cut centre, so the floor is symmetric about it; any clip that crosses
        the floor on a side removes that side's enclosure -> reject. This makes the
        coupling explicit: there is clip room ONLY when the pad is wider than the
        enclosure landing (it never is for a freshly-drawn rule pad), so on gf180
        via1/via2 a foreign conductor inside the pad's min-space reach forces the
        reject path -- reported honestly, never a silent enclosure break."""
        px0, py0, px1, py1 = pad
        cx = (px0 + px1) // 2
        cy = (py0 + py1) // 2
        vh = (cut_size + 1) // 2
        floor_half = vh + max(0, int(enclosure_nm))  # enclosure-landing half-extent
        fx0, fy0, fx1, fy1 = cx - floor_half, cy - floor_half, cx + floor_half, cy + floor_half
        for (hx0, hy0, hx1, hy1) in holes:
            # current pad/hole overlap (>0 on a side means the pad reaches into it)
            if not (hx0 < px1 and px0 < hx1 and hy0 < py1 and py0 < hy1):
                continue  # this hole no longer touches the (possibly clipped) pad
            # choose the single side with the SMALLEST retreat that clears the hole.
            cand = []
            # clip right edge to hx0 ; clip left edge to hx1 ; etc.
            cand.append(("R", px1 - hx0, (px0, py0, hx0, py1)))
            cand.append(("L", hx1 - px0, (hx1, py0, px1, py1)))
            cand.append(("T", py1 - hy0, (px0, py0, px1, hy0)))
            cand.append(("B", hy1 - py0, (px0, hy1, px1, py1)))
            cand.sort(key=lambda c: c[1])
            chosen = None
            for _side, _retreat, newpad in cand:
                nx0, ny0, nx1, ny1 = newpad
                if nx1 <= nx0 or ny1 <= ny0:
                    continue
                # enclosure floor must survive on every side
                if nx0 > fx0 or ny0 > fy0 or nx1 < fx1 or ny1 < fy1:
                    continue
                # min-width: both dimensions of the clipped pad
                if (nx1 - nx0) < min_width_nm or (ny1 - ny0) < min_width_nm:
                    continue
                chosen = newpad
                break
            if chosen is None:
                return None  # no clip on any side holds enclosure + min_width
            px0, py0, px1, py1 = chosen
        return (px0, py0, px1, py1)

    def _emit_via_obligated(
        self,
        grid: _Grid,
        tech: _Tech,
        polys: List[LayerPolygon],
        kept_rects: Dict[LayerKey, List[_Rect]],
        wire_cells: Set[Cell],
        ix: int,
        iy: int,
        gi: int,
        width_fn,
        foreign_rects,
        cell_taken,
        reloc_cells,
        foreign_metal_rects=None,
        suppressed_out=None,
        routing_envelope_nm: Optional[object] = None,
    ) -> None:
        """Emit ONE via (between ``gi`` and ``gi+1`` at cell ``(ix, iy)``) under
        the POST-CLIP ENVELOPE OBLIGATION (see ``_emit_wires``):

          * healthy at the cell (post-clip metal covers cut + clamped enclosure
            on both layers, and the cut is not buried in a foreign keep-out)
            -> emit the cut WHOLE;
          * envelope broken / cut buried -> RELOCATE to the nearest own-route
            cell (deterministic ring scan) whose envelope holds, drawing
            clip-clean extension metal where the second layer is missing;
          * no legal relocation -> suppress WHOLE (honest breakage for the
            geometric-open oracle). A partial via cut is NEVER emitted.

        Wire widths below the cut size carry NO envelope obligation (the
        unclipped pad never held the cut; that is the deck's account, not the
        clip's) -- such vias keep the LEGACY behaviour byte-identical:
        wholly-buried cuts drop, everything else emits."""
        vkey = tech.via_key_for(gi)
        vsz = _via_cut_size_nm(tech, gi)
        vh = max(1, vsz // 2)
        klo, khi = tech.gi_to_metal_key(gi), tech.gi_to_metal_key(gi + 1)
        w_lo, w_hi = int(width_fn(ix, iy, gi)), int(width_fn(ix, iy, gi + 1))
        rb, ra = self._via_env_rule_nm(tech, gi)
        cx, cy = grid.cell_xy(ix, iy)
        cut = (cx - vh, cy - vh, cx + vh, cy + vh)

        via_holes: List[_Rect] = []
        if foreign_rects is not None:
            via_holes.extend(foreign_rects.get(klo, ()))
            via_holes.extend(foreign_rects.get(khi, ()))
        buried = bool(via_holes) and not _subtract_rects(cut, via_holes)

        # per-layer FOREIGN keep-out the landing pad must hold min_space to: the
        # device/cap-plate ``foreign_rects`` PLUS other nets' committed routed metal
        # (``foreign_metal_rects``, already inflated by min_space). The pad's
        # min-spacing obligation is RELATIONAL -- it is measured against these,
        # split per-source by the shared PREDICATE ``via_landing_feasible``
        # consumed in ``_emit_landing`` (device overlap = short -> reject;
        # routed-metal min-space breach = clip-or-tolerated residual).
        def _emit_landing(px: int, py: int) -> bool:
            """Draw the RULE-DRIVEN metal landing for a cut centred (px, py).

            JUDGMENT (Task D1): delegated to the SHARED predicate
            :func:`clay.layout.route.via_model.via_landing_feasible` -- the ONE
            via-landing worldview the search phase prices with (Task D2, closing
            F1: search must stop committing via cells this obligation then
            suppresses). The predicate decides, per side: no enclosure owed /
            already covered by kept metal (no pad -> landing None); device-class
            reject (pad intrudes a device/cap-plate keep-out -> ok=False, the
            caller relocates/suppresses -- a short, never tolerable); or
            spacing-residual TOLERATE (pad reaches foreign ROUTED metal ->
            ok=True with the residual named). The emit call INJECTS the
            engine-resolved cut/enclosure numbers and its PRE-min-space-inflated
            foreign sets, so predicate and executor share one set of numbers
            (byte-stable vs the pre-D1 inline judgment).

            EXECUTION stays here: for a landing the predicate owes, try the
            3-way CLIP against routed-metal keep-outs (hold cut + per-side
            enclosure AND the layer min_width -- ``_clip_pad_to_min_space``).
            If a clip holds, take it (spacing cleared, wins preserved). If NO
            clip holds (the enclosure landing has no slack -- the genuine 3-way
            coupling), DRAW THE FULL RULE PAD anyway: preserving the via
            enclosure + the net's connectivity outranks the spacing rule here,
            and the residual is reported HONESTLY by the min-spacing checker /
            foundry deck rather than hidden behind a suppressed via (an
            unreported open -- the LDO MM10-class anti-recurrence line).

            MINIMALITY (external_pdk default-safe) is the predicate's covered test: a
            side whose ALREADY-KEPT metal covers the rule landing needs NO extra
            pad -- adding one over-fattens a wire that already enclosed the cut
            and creates same-layer SPACING violations (M2.S/M3.S). external_pdk (wide
            wires) draws no pads; gf180 (min-width wires) draws the pads it
            needs.

            Generic + leakage-free: every dimension is the PDK cut size + the PDK
            enclosure + the PDK min_space + the PDK min_width, no per-PDK constant
            and no reference coordinate."""
            feas = via_landing_feasible(
                getattr(tech, "pdk", tech), gi, px, py,
                own_cover_below=kept_rects.get(klo, ()),
                own_cover_above=kept_rects.get(khi, ()),
                foreign_device_rects_by_gi={
                    gi: foreign_rects.get(klo, ()) if foreign_rects else (),
                    gi + 1: foreign_rects.get(khi, ()) if foreign_rects else (),
                },
                foreign_routed_rects_by_gi={
                    gi: foreign_metal_rects.get(klo, ())
                    if foreign_metal_rects else (),
                    gi + 1: foreign_metal_rects.get(khi, ())
                    if foreign_metal_rects else (),
                },
                min_space_nm_by_gi=None,   # emit's foreign sets are PRE-inflated
                min_width_nm_by_gi=None,   # min_width is the clip's account below
                cut_size_nm=vsz,
                enclosure_below_nm=rb,
                enclosure_above_nm=ra,
            )
            if not feas.ok:
                # device-class reject -> relocate/suppress (never a silent short)
                return False
            for (e, key, gi_side, pad) in (
                (rb, klo, gi, feas.landing_below),
                (ra, khi, gi + 1, feas.landing_above),
            ):
                if pad is None:
                    continue  # no enclosure owed / covered by kept metal
                metal_holes = list(foreign_metal_rects.get(key, ())) \
                    if foreign_metal_rects else []
                if metal_holes and _subtract_rects(pad, metal_holes) != [pad]:
                    try:
                        _mw = int(tech.gi_min_width(gi_side))
                    except Exception:
                        _mw = vsz
                    clipped = self._clip_pad_to_min_space(
                        pad, metal_holes, cut_size=vsz, enclosure_nm=e,
                        min_width_nm=_mw,
                    )
                    if clipped is not None:
                        pad = clipped  # clip cleared the spacing while holding wins
                    # else: keep the full rule pad (enclosure + connectivity win),
                    # residual spacing reported honestly.
                polys.append(LayerPolygon(layer=key[0], datatype=key[1],
                                          polygon=Polygon.rectangle(*pad)))
                kept_rects.setdefault(tuple(key), []).append(pad)
            return True

        if rb <= 0 and ra <= 0:
            # PDK declares no via enclosure -> legacy verbatim (cut only).
            if buried:
                return
            polys.append(LayerPolygon(layer=vkey[0], datatype=vkey[1],
                                      polygon=Polygon.rectangle(*cut)))
            kept_rects.setdefault(tuple(vkey), []).append(cut)
            return

        if not buried and _emit_landing(cx, cy):
            # the original cell holds full enclosure + connectivity (a spacing
            # residual to foreign metal, if any, is tolerated by _emit_landing and
            # reported honestly by the min-spacing checker -- never suppressed into
            # an open). This is byte-identical to the pre-min-spacing behaviour on
            # any via that has no foreign-metal residual (external_pdk; gf180 device-clean).
            polys.append(LayerPolygon(layer=vkey[0], datatype=vkey[1],
                                      polygon=Polygon.rectangle(*cut)))
            kept_rects.setdefault(tuple(vkey), []).append(cut)
            return

        # --- relocation: nearest own-route cell whose envelope holds ----------
        # extension metal carries the DESIGNED envelope (rule clamped to the
        # original pad width) -- demanding the full raw rule there (e.g. the
        # external_pdk via1 54nm M2 enc -> a 368nm pad) over-fattens the extension past
        # the design's own envelope and gets it rejected for clearances the
        # design never needed.
        e_lo0 = min(rb, max(0, (w_lo - vsz) // 2))
        e_hi0 = min(ra, max(0, (w_hi - vsz) // 2))
        for d in (1, 2):
            for (dx, dy) in ((d, 0), (-d, 0), (0, d), (0, -d)):
                jx, jy = ix + dx, iy + dy
                if not grid.in_bounds(jx, jy):
                    continue
                plan = self._plan_via_relocation(
                    grid, tech, kept_rects, wire_cells, ix, iy, jx, jy, gi,
                    width_fn, foreign_rects, cell_taken, vsz, vh, rb, ra,
                    e_lo0, e_hi0, foreign_metal_rects,
                    routing_envelope_nm=routing_envelope_nm,
                )
                if plan is None:
                    continue
                extras, ncx, ncy = plan
                if not _emit_landing(ncx, ncy):
                    continue  # cannot hold the full enclosure at the relocated cell
                for (key_l, l, rect, w_used, cells) in extras:
                    polys.append(LayerPolygon(
                        layer=key_l[0], datatype=key_l[1],
                        polygon=Polygon.rectangle(*rect)))
                    kept_rects.setdefault(key_l, []).append(rect)
                    if reloc_cells is not None:
                        for (ux, uy) in cells:
                            reloc_cells.append((ux, uy, l, w_used))
                ncut = (ncx - vh, ncy - vh, ncx + vh, ncy + vh)
                polys.append(LayerPolygon(layer=vkey[0], datatype=vkey[1],
                                          polygon=Polygon.rectangle(*ncut)))
                kept_rects.setdefault(tuple(vkey), []).append(ncut)
                return
        # no legal relocation -> suppress WHOLE; the geometric-open oracle
        # reports the breakage honestly (never a broken-envelope via). (This path
        # is only reached for a DEVICE-foreign / buried via -- a spacing-only via
        # never reaches here: its original-cell _emit_landing returns True with the
        # residual tolerated + reported, so it stays connected.)
        # D3: record the suppression so the caller downgrades the net's connected
        # flag (a load-bearing tree via with no cut is a real open).
        if suppressed_out is not None:
            suppressed_out.append((ix, iy, gi))
        return

    def _plan_via_relocation(
        self,
        grid: _Grid,
        tech: _Tech,
        kept_rects: Dict[LayerKey, List[_Rect]],
        wire_cells: Set[Cell],
        ix: int,
        iy: int,
        jx: int,
        jy: int,
        gi: int,
        width_fn,
        foreign_rects,
        cell_taken,
        vsz: int,
        vh: int,
        rb: int,
        ra: int,
        e_lo0: int,
        e_hi0: int,
        foreign_metal_rects=None,
        routing_envelope_nm: Optional[object] = None,
    ):
        """Plan a relocated via at candidate cell ``(jx, jy)`` (same row/col as
        the broken cell ``(ix, iy)``). Returns ``(extras, ncx, ncy)`` or None.

        Candidate legality (every condition a clip post-obligation):
          * the candidate carries the net's OWN route on at least one of the
            two layers;
          * a layer present in ``wire_cells``: its post-clip metal must cover
            the cut + clamped enclosure there (width >= cut required);
          * a layer absent: extension metal (a landing pad of
            ``max(min_width, cut + 2*designed_enclosure)`` -- ``e_lo0/e_hi0``,
            the rule clamped to the ORIGINAL pad width -- plus a straight
            bridge back to the original cell) must pass the foreign clip
            WHOLE, overlap the net's kept metal at the original end, and every
            spanned cell must be free of other nets' metal/halo
            (``cell_taken``).
        ``extras`` is ``[(layer_key, gi, rect, width_nm, cells)]``."""
        on_lo = (jx, jy, gi) in wire_cells
        on_hi = (jx, jy, gi + 1) in wire_cells
        if not (on_lo or on_hi):
            return None
        ncx, ncy = grid.cell_xy(jx, jy)
        ocx, ocy = grid.cell_xy(ix, iy)
        env_bbox: Optional[Tuple[int, int, int, int]] = None
        if routing_envelope_nm is not None:
            from clay.layout.route.floorplan_containment import RoutingEnvelope

            env_bbox = RoutingEnvelope.coerce(routing_envelope_nm).bbox_nm
            ex0, ey0, ex1, ey1 = env_bbox
            req_half = max(
                vh,
                ((vsz + 1) // 2 + rb) if rb > 0 else vh,
                ((vsz + 1) // 2 + ra) if ra > 0 else vh,
            )
            if min(ncx - ex0, ex1 - ncx, ncy - ey0, ey1 - ncy) < req_half:
                return None
        extras: List[Tuple[LayerKey, int, _Rect, int, List[Tuple[int, int]]]] = []

        def _covered(box: _Rect, key: LayerKey) -> bool:
            return not _subtract_rects(box, kept_rects.get(key, ()))

        def _overlaps_kept(rect: _Rect, key: LayerKey) -> bool:
            for (a0, b0, a1, b1) in kept_rects.get(key, ()):
                if a0 < rect[2] and rect[0] < a1 and b0 < rect[3] and rect[1] < b1:
                    return True
            return False

        for (l, on_l, e_rule, e_designed) in ((gi, on_lo, rb, e_lo0),
                                              (gi + 1, on_hi, ra, e_hi0)):
            key_l = tech.gi_to_metal_key(l)
            if on_l:
                w_l = int(width_fn(jx, jy, l))
                if w_l < vsz:
                    return None  # cannot hold the cut here either
                e = min(e_rule, (w_l - vsz) // 2)
                if not _covered((ncx - vh - e, ncy - vh - e,
                                 ncx + vh + e, ncy + vh + e), key_l):
                    return None
                continue
            # extension on the missing layer: a landing pad at the DESIGNED
            # envelope + a straight bridge home.
            mw = tech.gi_min_width(l)
            side = max(mw, vsz + 2 * e_designed)
            hs = max(1, side // 2)
            pad: _Rect = (ncx - hs, ncy - hs, ncx + hs, ncy + hs)
            bw = max(1, mw // 2)
            if jy == iy:
                bridge: _Rect = (min(ncx, ocx), ncy - bw, max(ncx, ocx), ncy + bw)
            else:
                bridge = (ncx - bw, min(ncy, ocy), ncx + bw, max(ncy, ocy))
            if env_bbox is not None:
                ex0, ey0, ex1, ey1 = env_bbox
                for rx0, ry0, rx1, ry1 in (pad, bridge):
                    if rx0 < ex0 or ry0 < ey0 or rx1 > ex1 or ry1 > ey1:
                        return None
            # relocation extension clips against DEVICE foreign keep-outs only (a
            # plate overlap is a short); the routed-metal spacing residual is
            # tolerated at the landing (full rule pad), so it is NOT a reason to
            # reject a relocation here.
            holes = () if foreign_rects is None else foreign_rects.get(key_l, ())
            for r in (pad, bridge):
                if holes and _subtract_rects(r, list(holes)) != [r]:
                    return None  # extension would be clipped -> illegal spot
            # the bridge must land ON this net's kept metal at the original end
            # (the transition cell owns layer l there by construction; its pad
            # may be clipped -- require a real overlap, the open oracle backstops).
            if not _overlaps_kept(bridge, key_l):
                return None
            # spanned cells (candidate + intermediates toward the original cell,
            # exclusive) must be free of other nets' metal/halo.
            cells: List[Tuple[int, int]] = []
            sx = 0 if jx == ix else (1 if jx > ix else -1)
            sy = 0 if jy == iy else (1 if jy > iy else -1)
            ux, uy = jx, jy
            while (ux, uy) != (ix, iy):
                cells.append((ux, uy))
                ux, uy = ux - sx, uy - sy
            if cell_taken is not None:
                for (ux, uy) in cells:
                    if cell_taken(ux, uy, l, side):
                        return None
            extras.append((key_l, l, pad, side, cells))
            extras.append((key_l, l, bridge, side, []))
        return extras, ncx, ncy

    @staticmethod
    def _wire_length_nm(grid: _Grid, wire_cells: Set[Cell]) -> int:
        """Total in-plane wire length (nm): pitch times the number of
        same-layer 4-adjacent cell pairs in the routed set (each adjacency is one
        pitch of metal). Via hops add no in-plane length."""
        cells = set(wire_cells)
        pitch = grid.pitch
        count = 0
        for (ix, iy, gi) in cells:
            # count only +x and +y neighbours to avoid double counting
            if (ix + 1, iy, gi) in cells:
                count += 1
            if (ix, iy + 1, gi) in cells:
                count += 1
        return count * pitch

    # ------------------------------------------------------------------ shorts
    @staticmethod
    def _detect_shorts(metal_cells: Dict[str, Set[Cell]]) -> List[dict]:
        """The same-layer short oracle: report any grid cell that two DIFFERENT
        nets both occupy as DRAWN METAL (same ``(ix,iy,gi)``).

        This is a real geometric check on the routed metal, not a structural
        assumption -- ``metal_cells`` holds each net's actual wire cells (no halo).
        The grid+halo construction (later nets route around earlier nets' committed
        cells, halo included) should keep this EMPTY, but the oracle exists so a
        genuine overlap is surfaced rather than hidden. Each short is a dict
        ``{'cell': (ix,iy,gi), 'nets': (a, b)}``."""
        shorts: List[dict] = []
        # cell -> first net seen there
        seen: Dict[Cell, str] = {}
        for net in sorted(metal_cells):
            for cell in metal_cells[net]:
                other = seen.get(cell)
                if other is not None and other != net:
                    shorts.append(
                        {"cell": cell, "nets": tuple(sorted((other, net)))}
                    )
                else:
                    seen[cell] = net
        return shorts


# ---- module helpers (pure, no per-circuit logic) ----------------------------


# A nm-int axis-aligned rectangle (x0, y0, x1, y1) with x0<=x1, y0<=y1.
_Rect = Tuple[int, int, int, int]


# ``_subtract_rects`` (axis-aligned rect subtraction) moved to
# :mod:`clay.layout.route.via_model` as ``subtract_rects`` (imported above
# under the old private name): the shared via-landing feasibility predicate
# judges coverage with the SAME geometry the emit path clips with -- one
# implementation, one worldview (Task D1; differential-fuzzed byte-identical).


def _via_name_for(tech: _Tech, gi_low: int) -> Optional[str]:
    """The PDK via NAME between graph-layer ``gi_low`` and ``gi_low+1`` (for a
    cut-size lookup). Resolved from the lower metal's PDK index."""
    lower_pdk = tech._gi_to_pdk[gi_low]
    for cand in (f"via{lower_pdk}", f"vi{lower_pdk}", f"v{lower_pdk}", "via"):
        if cand in set(tech.pdk.via_names()):
            return cand
    return None


def _via_cut_size_nm(tech: _Tech, gi_low: int) -> int:
    """The via cut size (nm) for the cut between ``gi_low`` and ``gi_low+1``,
    read from the PDK; falls back to the smaller usable metal-min when the PDK is
    silent (so the via square is always legal-ish, never zero)."""
    name = _via_name_for(tech, gi_low)
    if name is not None:
        try:
            v = tech.pdk.via_size(name)
            if v:
                return int(v)
        except Exception:
            pass
    return min(tech.gi_min_width(gi_low), tech.gi_min_width(gi_low + 1))


__all__ = ["MazeRouteEngine"]
