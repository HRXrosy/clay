"""CONSISTENCY VERIFIER -- judge a RoutedCell against the external_pdk reference layout.

This is the JUDGE in the intent (+) faithful-router (+) verifier architecture (see
``clay/layout/route/__init__.py``). It takes the router's OUTPUT (:class:`RoutedCell`)
and asks, across several INDEPENDENT categories, whether that output is consistent with
the original reference -- the user's requirement "compare the routed result against the
original reference for consistency". Each category returns its own PASS/FAIL plus the
measured numbers; the overall verdict is the AND of all categories. A failing category is
reported as a FAIL with its numbers, NEVER spun as a pass.

This is GEOMETRIC verification, not topological-only: connectivity is checked, but so is
wire-over-device overlap area, same-layer cross-net spacing, per-net WIRE WIDTH (the
reference routes different nets at different widths and the router must honour that), and
route-invariance facts (each net's metal-layer set, each net's width class).

LEAKAGE -- THE EXPLICIT DISTINCTION
===================================
This verifier READS the reference routing facts (per-net width / layers, via
:func:`clay.layout.extract.route_extract.extract_reference_routing`). That is LEGITIMATE:
verification ALWAYS compares an output against ground truth. It is NOT data leakage.

Data leakage -- the thing the project forbids -- is when a reference MEASUREMENT feeds the
router's INPUT/intent (e.g. memorising the reference 8.0 um rail width into the intent so
the router "reproduces" it by construction). The router's intent must be DESIGN-DERIVED
(net role / current / matching), never measured-and-replayed. Reading the reference HERE,
on the OUTPUT side, to score how close the router got, is the opposite of leakage: it is
the held-out check that catches a router that did NOT reproduce the reference. Keep this
boundary crisp: ``verify.py`` (output vs reference) = allowed; feeding reference widths
into ``RoutingIntent`` (reference into input) = forbidden leakage.

UNITS. Reference facts are in MICRONS. RoutedCell wires are :class:`LayerPolygon`
rectangles in GDS database units; for the external_pdk reference dbu = 0.001 um (1 unit = 1 nm),
so wire coordinates convert to microns by ``* DBU_UM`` (default 0.001). Pass a different
``dbu_um`` if a RoutedCell ever uses another database unit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# default database unit of a RoutedCell wire coordinate, in microns (external_pdk: 1 nm).
DBU_UM = 0.001

# relative tolerance for per-net width comparison (project rule: RELATIVE, never absolute).
DEFAULT_WIDTH_REL_TOL = 0.15

# a routed wire figure must carry at least this much length (um) to count toward the
# net's TRUNK width -- mirrors route_extract._trunk_width_um so the two trunk notions
# are comparable (a stray min-size via pad is not a trunk).
_TRUNK_MIN_LEN_UM = 1.0

# default supply-net set (matches route_extract.DEFAULT_POWER_NETS).
_DEFAULT_POWER = frozenset({"vdd!", "vss!", "gnd!", "vdd", "vss", "gnd"})


# --------------------------------------------------------------------------- #
# report data structures
# --------------------------------------------------------------------------- #
@dataclass
class CategoryResult:
    """One independent verification category's verdict + the numbers behind it.

    ``name`` is the category id. ``ok`` is its PASS/FAIL. ``detail`` is a dict of the
    measured numbers (always populated, pass OR fail). ``summary`` is a one-line human
    string. ``deferred`` marks a category that was NOT computed (with the reason in
    ``detail['reason']``); a deferred category does NOT count against ``ok`` -- it is
    neutral (counted separately) so a deferred invariant never silently fails the run."""

    name: str
    ok: bool
    detail: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    deferred: bool = False


@dataclass
class ConsistencyReport:
    """The full consistency verdict of a RoutedCell against the reference.

    ``ok`` is the AND of every NON-deferred category's ``ok`` (a deferred category is
    neutral). ``categories`` is the ordered list of :class:`CategoryResult`. Printable via
    :meth:`summary` / ``str()``."""

    categories: List[CategoryResult] = field(default_factory=list)
    ok: bool = False

    def by_name(self, name: str) -> Optional[CategoryResult]:
        for c in self.categories:
            if c.name == name:
                return c
        return None

    @property
    def failed(self) -> List[CategoryResult]:
        return [c for c in self.categories if not c.ok and not c.deferred]

    @property
    def deferred(self) -> List[CategoryResult]:
        return [c for c in self.categories if c.deferred]

    def _recompute_ok(self) -> None:
        active = [c for c in self.categories if not c.deferred]
        self.ok = all(c.ok for c in active) if active else False

    def summary(self) -> str:
        lines: List[str] = []
        verdict = "OK" if self.ok else "FAIL"
        n_fail = len(self.failed)
        n_def = len(self.deferred)
        lines.append(
            f"consistency vs reference: {verdict} "
            f"({len(self.categories) - n_def} active categor"
            f"{'y' if (len(self.categories) - n_def) == 1 else 'ies'}, "
            f"{n_fail} failing, {n_def} deferred)"
        )
        for c in self.categories:
            if c.deferred:
                tag = "DEFER"
            else:
                tag = "PASS " if c.ok else "FAIL "
            lines.append(f"  [{tag}] {c.name}: {c.summary}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - thin delegate
        return self.summary()


# --------------------------------------------------------------------------- #
# geometry helpers (RoutedCell wires are LayerPolygon rectangles in dbu)
# --------------------------------------------------------------------------- #
def _wire_bbox_um(lp: Any, dbu_um: float) -> Optional[Tuple[float, float, float, float, int, int]]:
    """Bounding box of one wire ``LayerPolygon`` in microns + its GDS (layer, datatype).

    Returns ``(x0, y0, x1, y1, layer, datatype)`` or ``None`` for a degenerate shape.
    A LayerPolygon stores ``.layer`` / ``.datatype`` and ``.polygon.vertices`` (a list of
    points with ``.x`` / ``.y`` in dbu); the wire is rectangular so its bbox IS its
    extent."""
    poly = getattr(lp, "polygon", None)
    verts = getattr(poly, "vertices", None) if poly is not None else None
    if not verts:
        return None
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    x0 = min(xs) * dbu_um
    y0 = min(ys) * dbu_um
    x1 = max(xs) * dbu_um
    y1 = max(ys) * dbu_um
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1, int(getattr(lp, "layer", -1)), int(getattr(lp, "datatype", 0)))


def _rect_overlap_area(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    """Overlap area (um^2) of two axis-aligned rectangles ``(x0,y0,x1,y1)`` (0 if
    disjoint)."""
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox <= 0 or oy <= 0:
        return 0.0
    return ox * oy


def _rect_gap(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    """Edge-to-edge clearance (um) between two rectangles. 0 if they touch or overlap;
    otherwise the Euclidean-free L-inf style gap a router lane must hold (the larger of the
    x-gap and y-gap when they are separated, the smaller axis when one axis overlaps)."""
    dx = max(a[0] - b[2], b[0] - a[2])  # >0 => separated in x by dx
    dy = max(a[1] - b[3], b[1] - a[3])  # >0 => separated in y by dy
    if dx > 0 and dy > 0:
        # diagonally separated: clearance is the min separating distance (a wire on a
        # track only needs ONE axis of clearance to not be a same-track conflict).
        return min(dx, dy)
    if dx > 0:
        return dx
    if dy > 0:
        return dy
    return 0.0  # overlapping / touching


def _net_metal_rects(
    nr: Any, dbu_um: float, via_layers: set
) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """Per-metal-layer list of routed-wire rectangles (um) for one net, EXCLUDING via cut
    layers (a via cut is not a routing wire; counting it would pollute the width/overlap
    measurement). Keyed by GDS layer number."""
    by_layer: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for lp in getattr(nr, "wires", []) or []:
        bb = _wire_bbox_um(lp, dbu_um)
        if bb is None:
            continue
        x0, y0, x1, y1, layer, dt = bb
        if (layer, dt) in via_layers:
            continue
        by_layer[layer].append((x0, y0, x1, y1))
    return by_layer


def _merge_runs_1d(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping/abutting 1-D intervals (used to coalesce abutting wire pads into
    a continuous run before measuring its length)."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for lo, hi in intervals[1:]:
        plo, phi = out[-1]
        if lo <= phi + 1e-9:
            out[-1] = (plo, max(phi, hi))
        else:
            out.append((lo, hi))
    return out


def _measure_net_widths(
    rects_by_layer: Dict[int, List[Tuple[float, float, float, float]]]
) -> Tuple[Dict[float, float], float, float]:
    """Measure a net's routed wire-WIDTH facts from its rectangles.

    A faithful router emits a wire as a run of width-sized pads (so abutting pads merge
    into a continuous wire); the wire's WIDTH is the SHORT edge of each pad. We bucket the
    short edge (rounded to nm) -> total routed LENGTH at that width (sum of the long edge,
    deduplicated so a run of N abutting equal pads counts its true coalesced length, not
    N pad-lengths). Returns ``(width_hist, dominant_width, trunk_width)`` where:
      * ``width_hist``     maps short-edge width (um) -> routed length (um);
      * ``dominant_width`` is the width carrying the most length;
      * ``trunk_width``    is the WIDEST width carrying >= _TRUNK_MIN_LEN_UM of length
        (the router-relevant per-net spine; matches route_extract's trunk notion). 0 for
        an empty net.
    """
    # group rectangles by their short-edge width, collapsing colinear abutting pads.
    runs_by_width: Dict[float, List[Tuple[float, float, float]]] = defaultdict(list)
    # value entries: (orientation_key, lo, hi) on the LONG axis, plus a perpendicular
    # bucket so we only merge pads that are actually colinear.
    width_hist: Counter = Counter()
    for layer, rects in rects_by_layer.items():
        # bucket pads by (width, orientation, perpendicular-centre) to merge colinear runs.
        buckets: Dict[Tuple[float, str, float], List[Tuple[float, float]]] = defaultdict(list)
        for (x0, y0, x1, y1) in rects:
            w = x1 - x0
            h = y1 - y0
            short = round(min(w, h), 3)
            if short <= 0:
                continue
            if w >= h:
                # long axis is x; perpendicular centre is y
                key = (short, "x", round((y0 + y1) / 2.0, 3))
                buckets[key].append((x0, x1))
            else:
                key = (short, "y", round((x0 + x1) / 2.0, 3))
                buckets[key].append((y0, y1))
        for (short, _orient, _perp), spans in buckets.items():
            for lo, hi in _merge_runs_1d(spans):
                width_hist[short] += (hi - lo)
    hist = {round(w, 3): round(l, 3) for w, l in width_hist.items()}
    if not hist:
        return {}, 0.0, 0.0
    dominant = max(hist, key=lambda w: hist[w])
    substantial = [w for w, l in hist.items() if l >= _TRUNK_MIN_LEN_UM]
    trunk = max(substantial) if substantial else max(hist)
    return hist, dominant, trunk


# --------------------------------------------------------------------------- #
# width-class bucketing (route_invariance: width_class)
# --------------------------------------------------------------------------- #
def _width_class(trunk_um: float, *, is_power: bool, cap_threshold_um: float = 10.0,
                 trunk_threshold_um: float = 4.0) -> str:
    """Bucket a per-net trunk width into a coarse CLASS the router must reproduce even when
    the exact width differs slightly. Classes:
      * ``"cap"``    -- a very wide plate strap (>= cap_threshold_um, e.g. the 28.8 um MIM
                        cap bottom-plate);
      * ``"power"``  -- a supply rail (a named power net, OR a wide trunk >= trunk_threshold
                        that is not a cap) -- the wide 8 um rails;
      * ``"signal"`` -- everything else (the ~1-2 um signal nets).
    Bucketing (not exact width) is what ``width_class`` invariance checks; the exact-width
    check is the separate per-net width category."""
    if trunk_um >= cap_threshold_um:
        return "cap"
    if is_power or trunk_um >= trunk_threshold_um:
        return "power"
    return "signal"


# --------------------------------------------------------------------------- #
# category 1: connectivity == netlist (the LVS precondition)
# --------------------------------------------------------------------------- #
def _check_connectivity(
    routed: Any, required_nets: Sequence[str]
) -> CategoryResult:
    """Every required net is present + ``connected=True`` in the RoutedCell; no shorts; no
    unsatisfied hard constraints. This is the LVS precondition -- topological, but the
    mandatory base every other category sits on."""
    nets = getattr(routed, "nets", {}) or {}
    missing: List[str] = []
    disconnected: List[str] = []
    for net in required_nets:
        nr = nets.get(net)
        if nr is None:
            missing.append(net)
        elif not getattr(nr, "connected", False):
            disconnected.append(net)
    shorts = list(getattr(routed, "shorts", []) or [])
    unsat = list(getattr(routed, "unsatisfied", []) or [])
    n_required = len(required_nets)
    n_connected = n_required - len(missing) - len(disconnected)
    ok = (not missing) and (not disconnected) and (not shorts) and (not unsat)
    detail = {
        "n_required_nets": n_required,
        "n_connected": n_connected,
        "missing_nets": missing,
        "disconnected_nets": disconnected,
        "n_shorts": len(shorts),
        "shorts": [str(s) for s in shorts][:20],
        "n_unsatisfied": len(unsat),
        "unsatisfied": [str(u) for u in unsat][:20],
    }
    summary = (
        f"{n_connected}/{n_required} nets connected, "
        f"{len(shorts)} short(s), {len(unsat)} unsatisfied constraint(s)"
    )
    if missing:
        summary += f"; MISSING {missing}"
    if disconnected:
        summary += f"; DISCONNECTED {disconnected}"
    return CategoryResult("connectivity", ok, detail, summary)


# --------------------------------------------------------------------------- #
# category 2: no wire-over-device / same-layer cross-net spacing (DRC)
# --------------------------------------------------------------------------- #
def _check_drc(
    routed: Any,
    obstacles: Sequence[Any],
    tech: Any,
    via_layers: set,
    dbu_um: float,
    *,
    spacing_um: Optional[float] = None,
) -> CategoryResult:
    """Routed wires do NOT overlap device artwork on the same layer (wire-over-device),
    and same-layer CROSS-NET wires hold a min spacing.

    ``obstacles`` is a sequence of keep-out rectangles, each exposing ``x0/y0/x1/y1`` and
    a ``layer`` (a GDS ``(layer, datatype)`` tuple or a metal-role name resolved through
    ``tech``). Reports total wire-over-device overlap area (um^2; target ~0) and the worst
    same-layer cross-net spacing violation (um; target: none below the rule)."""
    nets = getattr(routed, "nets", {}) or {}

    # per-layer routed metal rectangles tagged with their net (exclude via cuts).
    metal_by_layer: Dict[int, List[Tuple[Tuple[float, float, float, float], str]]] = defaultdict(list)
    for net, nr in nets.items():
        for layer, rects in _net_metal_rects(nr, dbu_um, via_layers).items():
            for r in rects:
                metal_by_layer[layer].append((r, net))

    # --- wire-over-device overlap ---
    obs_by_layer: Dict[int, List[Tuple[Tuple[float, float, float, float], Optional[str]]]] = defaultdict(list)
    n_obstacles = 0
    for ob in obstacles or []:
        lk = _resolve_layer(getattr(ob, "layer", None), tech)
        if lk is None:
            continue
        n_obstacles += 1
        obs_by_layer[lk[0]].append((
            (float(ob.x0), float(ob.y0), float(ob.x1), float(ob.y1)),
            getattr(ob, "owner_net", None),
        ))
    overlap_area = 0.0
    overlap_pairs = 0
    for layer, rects in metal_by_layer.items():
        obs = obs_by_layer.get(layer, [])
        if not obs:
            continue
        for (r, net) in rects:
            for (o, own) in obs:
                if own is not None and own == net:
                    continue  # a net over its OWN terminal pad is a connection
                a = _rect_overlap_area(r, o)
                if a > 1e-9:
                    overlap_area += a
                    overlap_pairs += 1

    # --- same-layer cross-net spacing ---
    worst_gap = None
    worst_pair = None
    n_space_viol = 0
    # R4 net-agnostic pre-emit diagnostic (computed in the merged path below; -1 =
    # not measured, e.g. klayout unavailable -> never a silent 0).
    n_space_viol_netagnostic = -1
    for layer, tagged in metal_by_layer.items():
        if spacing_um is not None:
            req = spacing_um
        else:
            req = _layer_min_space_um(tech, layer)
        if req is None or req <= 0:
            continue
        m = len(tagged)
        for i in range(m):
            ri, ni = tagged[i]
            for j in range(i + 1, m):
                rj, nj = tagged[j]
                if ni == nj:
                    continue  # same net may abut itself
                if _rect_overlap_area(ri, rj) > 1e-9:
                    gap = 0.0
                else:
                    gap = _rect_gap(ri, rj)
                if gap + 1e-9 < req:
                    n_space_viol += 1
                    if worst_gap is None or gap < worst_gap:
                        worst_gap = gap
                        worst_pair = (ni, nj, layer)

    # AUTHORITATIVE refine with MERGED + net-aware klayout geometry when available:
    # the segment-emit bridges fragment a wire into many pads, so the pad-rect
    # estimate above OVER-counts (a continuous wire reads as many touching pads);
    # merged per-net regions give the TRUE wire-over-FOREIGN-device area and the true
    # cross-net spacing. This OVERRIDES the pad-rect estimate.
    drc_method = "pad-rect"
    try:
        import klayout.db as kdb
        _SC = 1000.0

        def _mreg(rs):
            g = kdb.Region()
            for (x0, y0, x1, y1) in rs:
                g.insert(kdb.Box(round(x0 * _SC), round(y0 * _SC), round(x1 * _SC), round(y1 * _SC)))
            return g.merged()

        wreg: Dict[Tuple[str, int], Any] = {}
        for layer, tagged in metal_by_layer.items():
            for net in {n for (_r, n) in tagged}:
                wreg[(net, layer)] = _mreg([r for (r, n) in tagged if n == net])
        dall: Dict[int, Any] = {}
        downer: Dict[Tuple[int, str], Any] = {}
        for layer, tagged in obs_by_layer.items():
            g = kdb.Region()
            for (r, own) in tagged:
                b = kdb.Box(round(r[0] * _SC), round(r[1] * _SC), round(r[2] * _SC), round(r[3] * _SC))
                g.insert(b)
                if own is not None:
                    downer.setdefault((layer, own), kdb.Region()).insert(b)
            dall[layer] = g.merged()
        m_overlap = 0.0
        for (net, layer), wr in wreg.items():
            dev = dall.get(layer)
            if dev is None or dev.is_empty():
                continue
            foreign = dev - downer[(layer, net)].merged() if (layer, net) in downer else dev
            m_overlap += (wr & foreign).area() / (_SC * _SC)
        m_viol = 0
        m_worst = None
        m_pair = None
        bln: Dict[int, list] = defaultdict(list)
        for (net, layer), wr in wreg.items():
            bln[layer].append((net, wr))
        for layer, lst in bln.items():
            req = spacing_um if spacing_um is not None else _layer_min_space_um(tech, layer)
            if not req or req <= 0:
                continue
            rq = max(1, int(round(req * _SC)))
            for i in range(len(lst)):
                ni, ri = lst[i]
                grown = ri.sized(rq - 1)
                for j in range(i + 1, len(lst)):
                    nj, rj = lst[j]
                    if not (grown & rj).is_empty():
                        m_viol += 1
                        if m_worst is None:
                            m_worst = 0.0
                            m_pair = (ni, nj, layer)
        # NET-AGNOSTIC merged-layer spacing (R4): the foundry deck merges ALL of
        # layer k (every net together) and runs ``metal.space(minw)`` on the merged
        # result, so two shapes that PHYSICALLY touch (same net, abutting) coalesce
        # and never fire, while two shapes that do NOT touch fire if closer than the
        # rule -- WHETHER different nets OR a broken SAME-net run that neither abuts
        # nor spaces (a routed pad 350nm from its own cap plate). The per-net-pair
        # count above compares only DISTINCT nets, so it is BLIND to a same-net
        # non-abutting sub-min gap. This merges the whole layer and runs
        # ``Region.space_check`` -- byte-identical geometry + idiom to the standalone
        # foundry-faithful ``metal_min_spacing_check`` and the deck ``M{k}.2a`` rule.
        # SCOPE: this measures the IN-MEMORY routed wires (pre-emit). The canonical
        # LVS/DRC artifact additionally applies the same-net M1 heal/collar at emit,
        # which closes some same-net M1 gaps -- so on M1-healed cells this net-agnostic
        # count can EXCEED the foundry count on the emitted GDS. The authoritative
        # foundry-faithful spacing oracle is ``metal_min_spacing_check`` run on the
        # EMITTED GDS; this field is the route's own honest pre-emit net-agnostic
        # diagnostic (reported alongside, never silently masking the gap).
        m_viol_na = 0
        all_by_layer: Dict[int, Any] = {}
        for (net, layer), wr in wreg.items():
            all_by_layer.setdefault(layer, kdb.Region()).insert(wr)
        for layer, reg in all_by_layer.items():
            req = spacing_um if spacing_um is not None else _layer_min_space_um(tech, layer)
            if not req or req <= 0:
                continue
            reg.merge()
            rq = max(1, int(round(req * _SC)))
            sc = reg.space_check(rq)
            seen: Dict[Tuple[int, int], None] = {}
            for ep in sc.each():
                e = ep.first
                mx = int(round((e.x1 + e.x2) / 2.0))
                my = int(round((e.y1 + e.y2) / 2.0))
                seen[(mx, my)] = None
            m_viol_na += len(seen)
        n_space_viol_netagnostic = m_viol_na
        overlap_area, overlap_pairs = m_overlap, -1
        n_space_viol, worst_gap, worst_pair = m_viol, m_worst, m_pair
        drc_method = "merged-netaware"
    except ImportError:
        pass

    ok = (overlap_area <= 1e-6) and (n_space_viol == 0)
    detail = {
        "n_obstacles": n_obstacles,
        "wire_over_device_overlap_um2": round(overlap_area, 6),
        "n_overlap_pairs": overlap_pairs,
        "n_spacing_violations": n_space_viol,
        # R4: the net-AGNOSTIC merged-layer spacing count (same-net + cross-net),
        # measured on the in-memory routed wires (pre-emit). -1 = not measured. This
        # catches same-net non-abutting sub-min gaps the per-net-pair
        # ``n_spacing_violations`` is blind to; the authoritative post-emit foundry
        # oracle is ``metal_min_spacing_check`` on the emitted GDS.
        "n_spacing_violations_netagnostic": n_space_viol_netagnostic,
        "worst_spacing_gap_um": (round(worst_gap, 4) if worst_gap is not None else None),
        "worst_spacing_pair": worst_pair,
        "method": drc_method,
        "spacing_checked": spacing_um is not None or any(
            _layer_min_space_um(tech, l) for l in metal_by_layer
        ),
    }
    summary = (
        f"wire-over-foreign-device {round(overlap_area, 4)} um^2; "
        f"{n_space_viol} cross-net spacing violation(s) [{drc_method}]"
    )
    if worst_pair is not None:
        summary += f" (worst {round(worst_gap, 3)} um between {worst_pair[0]}/{worst_pair[1]})"
    return CategoryResult("drc_no_wire_over_device", ok, detail, summary)


# --------------------------------------------------------------------------- #
# category 3: per-net width vs reference (RELATIVE tolerance, never absolute)
# --------------------------------------------------------------------------- #
def _check_widths(
    routed: Any,
    facts: Any,
    via_layers: set,
    dbu_um: float,
    rel_tol: float,
    power_nets: set,
) -> CategoryResult:
    """Compare each routed net's TRUNK width to the reference net's trunk width using a
    RELATIVE tolerance (project rule: never an absolute-difference threshold).

    For each net present in BOTH the routed cell and the reference facts we report
    ``(ref_width, routed_width, rel_error)`` and PASS that net iff
    ``rel_error <= rel_tol``. Nets only in the reference (not routed) are reported as
    UNROUTED (they fail -- the router did not reproduce a reference wire). A net only in
    the routed cell (no reference fact, e.g. an anon cap strap that was not name-matched)
    is reported as EXTRA and is NEUTRAL (we cannot score a width we have no reference
    for), not a fail."""
    nets = getattr(routed, "nets", {}) or {}
    ref_by_name = _ref_widths_by_name(facts)

    rows: List[Dict[str, Any]] = []
    per_net_ok: List[bool] = []
    unrouted: List[str] = []
    extra: List[str] = []

    # routed trunk width per net
    routed_trunk: Dict[str, float] = {}
    for net, nr in nets.items():
        rects = _net_metal_rects(nr, dbu_um, via_layers)
        _hist, _dom, trunk = _measure_net_widths(rects)
        routed_trunk[net] = trunk

    for net, ref_w in ref_by_name.items():
        if net not in routed_trunk:
            unrouted.append(net)
            continue
        rw = routed_trunk[net]
        if ref_w <= 0:
            continue
        rel = abs(rw - ref_w) / ref_w
        net_ok = rel <= rel_tol
        per_net_ok.append(net_ok)
        rows.append({
            "net": net,
            "ref_width_um": round(ref_w, 3),
            "routed_width_um": round(rw, 3),
            "rel_error": round(rel, 4),
            "ok": net_ok,
        })

    for net in routed_trunk:
        if net not in ref_by_name and routed_trunk[net] > 0:
            extra.append(net)

    rows.sort(key=lambda r: (-r["rel_error"], r["net"]))
    ok = bool(per_net_ok) and all(per_net_ok) and not unrouted
    n_pass = sum(1 for r in rows if r["ok"])
    detail = {
        "rel_tol": rel_tol,
        "n_compared": len(rows),
        "n_within_tol": n_pass,
        "per_net": rows,
        "unrouted_reference_nets": unrouted,
        "extra_routed_nets_no_ref": extra,
        "worst_rel_error": (rows[0]["rel_error"] if rows else None),
    }
    summary = (
        f"{n_pass}/{len(rows)} net widths within {int(rel_tol * 100)}% of reference"
        + (f"; worst rel-err {rows[0]['rel_error']:.2%} on {rows[0]['net']}" if rows else "")
        + (f"; {len(unrouted)} reference net(s) UNROUTED {unrouted}" if unrouted else "")
    )
    return CategoryResult("per_net_width", ok, detail, summary)


def _ref_widths_by_name(facts: Any) -> Dict[str, float]:
    """``{net_name -> reference trunk_width_um}`` from a RoutingFacts, keyed by the net's
    attributed name (skipping the ``anon#`` placeholders, which carry no router-known net
    name to match against)."""
    out: Dict[str, float] = {}
    for n in getattr(facts, "nets", {}).values():
        name = getattr(n, "name", "")
        if not name or name.startswith("anon#"):
            continue
        tw = float(getattr(n, "trunk_width_um", 0.0) or 0.0)
        out[name] = tw
    return out


# --------------------------------------------------------------------------- #
# category 4: route_invariance -- net_layer + width_class (others deferred)
# --------------------------------------------------------------------------- #
def _check_net_layer(
    routed: Any,
    facts: Any,
    tech: Any,
    via_layers: set,
    dbu_um: float,
) -> CategoryResult:
    """route_invariance/NET_LAYER: each net's set of routing METAL layers matches the
    reference's. We compare the set of GDS metal layer numbers the routed net uses to the
    reference net's metal layer set; PASS a net iff the sets are equal. Reported per net as
    ``(ref_layers, routed_layers)`` with the metal role names."""
    nets = getattr(routed, "nets", {}) or {}
    ref_layers = _ref_layers_by_name(facts)
    name_of = _metal_namer(facts, tech)

    rows: List[Dict[str, Any]] = []
    per_ok: List[bool] = []
    for net, nr in nets.items():
        if net not in ref_layers:
            continue
        routed_set = set(_net_metal_rects(nr, dbu_um, via_layers).keys())
        ref_set = ref_layers[net]
        net_ok = routed_set == ref_set
        per_ok.append(net_ok)
        rows.append({
            "net": net,
            "ref_layers": sorted(name_of(l) for l in ref_set),
            "routed_layers": sorted(name_of(l) for l in routed_set),
            "ok": net_ok,
        })
    rows.sort(key=lambda r: (r["ok"], r["net"]))
    ok = bool(per_ok) and all(per_ok)
    detail = {
        "n_compared": len(rows),
        "n_match": sum(1 for r in rows if r["ok"]),
        "per_net": rows,
    }
    summary = f"{detail['n_match']}/{len(rows)} nets on the reference metal-layer set"
    return CategoryResult("invariance.net_layer", ok, detail, summary)


def _check_width_class(
    routed: Any,
    facts: Any,
    via_layers: set,
    dbu_um: float,
    power_nets: set,
) -> CategoryResult:
    """route_invariance/WIDTH_CLASS: each net's width CLASS (cap / power / signal, bucketed
    from the trunk width) matches the reference's. This is coarser than the exact per-net
    width category -- it catches a router that routed a wide rail as a thin signal (class
    flip) even if both happened to land within the exact-width tolerance band edges. PASS a
    net iff its routed class == its reference class.

    The REFERENCE class uses the reference's ``is_power`` flag (legitimate ground truth: a
    supply net is power-class even if its trunk were narrow). The ROUTED class is measured
    PURELY from the routed geometry (``is_power=False``) -- so a router that draws a power
    rail at a thin signal width is caught as a class flip, instead of being excused because
    the net is named a supply. The two are equal only when the router gave the rail a
    physically wide trunk, which is the whole point."""
    nets = getattr(routed, "nets", {}) or {}
    ref_trunk = _ref_widths_by_name(facts)
    ref_power = _ref_power_names(facts, power_nets)

    rows: List[Dict[str, Any]] = []
    per_ok: List[bool] = []
    for net, nr in nets.items():
        if net not in ref_trunk:
            continue
        rects = _net_metal_rects(nr, dbu_um, via_layers)
        _h, _d, routed_trunk = _measure_net_widths(rects)
        is_power = net in ref_power or net in power_nets
        ref_cls = _width_class(ref_trunk[net], is_power=is_power)
        # routed class from geometry ONLY -- never excuse a thin rail by its supply name.
        routed_cls = _width_class(routed_trunk, is_power=False)
        net_ok = ref_cls == routed_cls
        per_ok.append(net_ok)
        rows.append({
            "net": net,
            "ref_class": ref_cls,
            "routed_class": routed_cls,
            "ref_trunk_um": round(ref_trunk[net], 3),
            "routed_trunk_um": round(routed_trunk, 3),
            "ok": net_ok,
        })
    rows.sort(key=lambda r: (r["ok"], r["net"]))
    ok = bool(per_ok) and all(per_ok)
    detail = {
        "n_compared": len(rows),
        "n_match": sum(1 for r in rows if r["ok"]),
        "per_net": rows,
    }
    summary = f"{detail['n_match']}/{len(rows)} nets in the reference width-class"
    return CategoryResult("invariance.width_class", ok, detail, summary)


def _ref_layers_by_name(facts: Any) -> Dict[str, set]:
    """``{net_name -> set of reference GDS metal layer numbers}`` (skip anon placeholders)."""
    out: Dict[str, set] = {}
    for n in getattr(facts, "nets", {}).values():
        name = getattr(n, "name", "")
        if not name or name.startswith("anon#"):
            continue
        out[name] = {l[0] for l in getattr(n, "layers", [])}
    return out


def _ref_power_names(facts: Any, power_nets: set) -> set:
    out = set()
    for n in getattr(facts, "nets", {}).values():
        if getattr(n, "is_power", False):
            out.add(getattr(n, "name", ""))
    return out | set(power_nets)


# --------------------------------------------------------------------------- #
# layer / spacing resolution helpers
# --------------------------------------------------------------------------- #
def _resolve_layer(layer: Any, tech: Any) -> Optional[Tuple[int, int]]:
    """Resolve an obstacle/wire layer spec to a GDS ``(layer, datatype)`` tuple. Accepts a
    2-tuple directly, or a metal role name ('M1'/'M2'/...) resolved through ``tech``."""
    if layer is None:
        return None
    if isinstance(layer, (tuple, list)) and len(layer) == 2:
        return (int(layer[0]), int(layer[1]))
    if isinstance(layer, str) and tech is not None:
        s = layer.upper().strip()
        if s.startswith("M") and s[1:].isdigit():
            try:
                k = tech.metal_key(int(s[1:]))
                return (int(k[0]), int(k[1]))
            except Exception:
                return None
    return None


def _via_layer_set(facts: Any, tech: Any) -> set:
    """The set of GDS ``(layer, datatype)`` that are VIA CUTS (not routing wires), so the
    width/overlap/layer measurements skip them. Sourced from the reference layer stack
    (``facts.stack``) plus the contact, with a tech fallback."""
    out: set = set()
    stack = getattr(facts, "stack", None)
    if stack is not None:
        for v in getattr(stack, "vias", {}).values():
            out.add((int(v[0]), int(v[1])))
        c = getattr(stack, "contact", None)
        if c is not None:
            out.add((int(c[0]), int(c[1])))
    if tech is not None:
        for n in range(0, 6):
            try:
                k = tech.via_key(n)
                out.add((int(k[0]), int(k[1])))
            except Exception:
                pass
    return out


def _metal_namer(facts: Any, tech: Any):
    """Return a function GDS layer-number -> short metal name ('M1'/...). Uses the
    reference layer stack first, then tech, then a generic ``L<n>`` label."""
    stack = getattr(facts, "stack", None)
    rev: Dict[int, str] = {}
    if stack is not None:
        for i, lk in getattr(stack, "metals", {}).items():
            rev[int(lk[0])] = f"M{i}"

    def name_of(layer: int) -> str:
        if layer in rev:
            return rev[layer]
        return f"L{layer}"

    return name_of


# per-PDK metal rule-container name (the LayoutTech min_space key) by metal index.
def _layer_min_space_um(tech: Any, gds_layer: int) -> Optional[float]:
    """Min same-layer spacing (um) for the metal carrying GDS layer ``gds_layer``,
    resolved through ``tech``. Maps the GDS layer number back to a metal index, then asks
    the LayoutTech rule container for that metal's spacing. Returns ``None`` when the PDK
    declares no rule (the DRC spacing sub-check then no-ops for that layer rather than
    inventing a number)."""
    if tech is None:
        return None
    # find the metal index whose metal_key matches this GDS layer.
    for n in range(1, 7):
        try:
            k = tech.metal_key(n)
        except Exception:
            continue
        if int(k[0]) == gds_layer:
            for nm in (f"metal{n}", f"met{n}", f"m{n}"):
                v = tech.min_space(nm)
                if v is not None:
                    return v / 1000.0
            return None
    return None


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #
def verify_consistency(
    routed: Any,
    *,
    intent: Any,
    ref_gds: str,
    tech: Any,
    pins_by_net: Optional[Dict[str, Sequence[Any]]] = None,
    obstacles: Sequence[Any] = (),
    dbu_um: float = DBU_UM,
    width_rel_tol: float = DEFAULT_WIDTH_REL_TOL,
    spacing_um: Optional[float] = None,
    power_nets: Sequence[str] = (),
    _facts: Any = None,
) -> ConsistencyReport:
    """Compare a :class:`RoutedCell` against the reference layout across independent
    categories and return a structured :class:`ConsistencyReport` (overall ``ok`` = AND of
    the active categories; a deferred category is neutral). NEVER spins a failing category
    as pass.

    Categories
    ----------
    1. ``connectivity``            -- every required net connected, no shorts, no
                                      unsatisfied hard constraints (the LVS precondition).
    2. ``drc_no_wire_over_device`` -- routed wires do not overlap ``obstacles`` (device
                                      artwork) on the same layer; same-layer cross-net min
                                      spacing held. Reports overlap area (um^2).
    3. ``per_net_width``           -- each routed net's TRUNK width vs the reference net's,
                                      RELATIVE tolerance ``width_rel_tol`` (never absolute).
    4. ``invariance.net_layer``    -- each net's metal-layer SET matches the reference.
       ``invariance.width_class``  -- each net's width class (cap/power/signal) matches.

    Parameters
    ----------
    ``routed``       the RoutedCell to judge.
    ``intent``       the netlist intent for the cell (drives required-net discovery + the
                     reference routing extraction). DESIGN-derived; reading it is not
                     leakage.
    ``ref_gds``      path to the reference layout.gds (READ-only) -- the ground truth.
    ``tech``         a :class:`clay.layout.tech.LayoutTech` (layer keys + spacing rules).
    ``pins_by_net``  optional ``{net -> pins}``; its keys define the required-net set for
                     connectivity. Falls back to the netlist nets in ``intent``.
    ``obstacles``    device-artwork keep-out rectangles for the wire-over-device check.
    ``dbu_um``       database unit of a RoutedCell wire coordinate, in microns.
    ``width_rel_tol``relative width tolerance (default 15%).
    ``spacing_um``   override the same-layer min spacing (else read per-layer from tech).
    ``power_nets``   extra supply-net names (augment the default vdd!/gnd!/... set).
    ``_facts``       (testing) inject pre-extracted RoutingFacts to skip the GDS read.
    """
    from clay.layout.extract.route_extract import extract_reference_routing

    facts = _facts if _facts is not None else extract_reference_routing(
        ref_gds, intent, power_nets=power_nets
    )
    power = set(_DEFAULT_POWER) | {str(p) for p in power_nets}
    via_layers = _via_layer_set(facts, tech)

    required = list(pins_by_net.keys()) if pins_by_net else _intent_nets(intent)

    report = ConsistencyReport()
    report.categories.append(_check_connectivity(routed, required))
    report.categories.append(
        _check_drc(routed, obstacles, tech, via_layers, dbu_um, spacing_um=spacing_um)
    )
    report.categories.append(
        _check_widths(routed, facts, via_layers, dbu_um, width_rel_tol, power)
    )
    report.categories.append(
        _check_net_layer(routed, facts, tech, via_layers, dbu_um)
    )
    report.categories.append(
        _check_width_class(routed, facts, via_layers, dbu_um, power)
    )
    # the 6 route_invariance categories we do NOT compute from (RoutedCell + reference)
    # are recorded as DEFERRED (neutral) with an honest reason -- never silently dropped.
    for name, reason in _DEFERRED_INVARIANTS:
        report.categories.append(
            CategoryResult(f"invariance.{name}", True, {"reason": reason},
                           f"DEFERRED: {reason}", deferred=True)
        )

    report._recompute_ok()
    return report


def _intent_nets(intent: Any) -> List[str]:
    """All net names referenced by the netlist intent (devices + passives) -- the
    connectivity required-net set when no explicit ``pins_by_net`` is given."""
    out: set = set()
    for d in getattr(intent, "devices", []) or []:
        out.update(str(v) for v in (getattr(d, "nets", {}) or {}).values())
    for p in getattr(intent, "passives", []) or []:
        out.update(str(v) for v in (getattr(p, "nets", {}) or {}).values())
    return sorted(out)


# The 6 of the 8 route_invariance categories NOT computed here, each with the honest
# reason it is deferred (it needs information a RoutedCell + the reference fact reader do
# not currently expose). net_layer + width_class ARE implemented above.
_DEFERRED_INVARIANTS: Tuple[Tuple[str, str], ...] = (
    ("track_topology",
     "needs the reference per-net track centrelines + ordering; route_extract reports "
     "per-net widths/layers but not per-channel track indices, so there is no reference "
     "track topology to compare a RoutedCell's tracks against yet."),
    ("pin_access",
     "needs per-terminal access side/layer from BOTH the reference (not extracted by "
     "route_extract) and the RoutedCell (NetRoute carries wires, not per-pin entry "
     "sides); the Access constraint exists but no reference access fact does."),
    ("via_structure",
     "needs reference via-array geometry (cut count/rows) per net; route_extract treats "
     "via cells as stack-definition only and does not report per-net via arrays, so the "
     "reference via structure is not available to compare."),
    ("route_symmetry",
     "needs the reference differential-pair mirror axis + matched geometry; the GDS is "
     "left/right symmetric but route_extract reports VIN/VIP as twin-AMBIGUOUS (no "
     "X-position), so there is no trustworthy reference mirror fact to verify against."),
    ("length_class",
     "needs reference per-net routed LENGTH; route_extract measures wire WIDTH (and a "
     "length histogram per width) but does not report a per-net total route length to "
     "bucket into a length class."),
    ("shield_spacing",
     "needs reference shield-net identity + victim/aggressor spacing; the reference has "
     "no shield-net annotation and route_extract extracts no shield fact, so there is "
     "nothing to compare a RoutedCell's shield spacing to."),
)


__all__ = [
    "ConsistencyReport",
    "CategoryResult",
    "verify_consistency",
    "DBU_UM",
    "DEFAULT_WIDTH_REL_TOL",
]
