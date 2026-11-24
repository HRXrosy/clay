"""FAITHFUL difference-constraint realizer for leakage-free layout intent.

WHY THIS EXISTS
===============
The degradation method extracts declarative layout primitives and must REALIZE
them to test completeness. Two earlier realizers were wrong instruments:

* ``leakfree.solve_leakfree`` / ``discover.solve_discovered`` -- greedy seed
  propagation. Connecting the relation graph (``floating == 0``) does NOT mean the
  layout is reproduced: greedy propagation lets the first edge to reach a device
  win, so cross-cluster spacing comes out at a default gap, not the structure the
  relations imply (median ~15um off on the OTA).
* ``annotate_to_solver.solve_placement_from_constraints`` -- an area-MINIMAL
  MILP. It has a COMPETING objective (compact area) and treats ``abut`` as a soft
  gap-tighten, so it honored only ~36% of the reference relations.

The correct instrument has NO competing objective: it just SATISFIES the declared
relations. The structural primitives split cleanly into

* EQUALITY constraints (exact, size-derived): ``abut`` (faces flush), ``align:x/y``
  (shared center), ``align:left/right/top/bottom`` (shared edge). Each pins one
  coordinate difference ``coord(child) - coord(parent) = offset`` where ``offset``
  is reconstructed from SIZES + the relation's SIGN -- never from a reference
  coordinate (leakage-free).
* INEQUALITY constraints: ``order:*`` (min-separation) -- these only bound spacing,
  they do not pin it. The slack they leave is the genuinely-free inter-cluster
  spacing (a don't-care: the routing-channel height).

This module solves the EQUALITY system per axis with a weighted union-find
(potentials): every device gets an exact offset to its equality-component root, and
inconsistent relations are COUNTED as conflicts (an honest detector-quality signal).
A device pinned (transitively) to the layout root by equalities is reproduced to
sub-micron WITHOUT any stored coordinate; a device only reachable through order
edges is left as free-spacing residual and reported as such. The honest completeness
number is the EQUALITY-PIN COVERAGE, not ``floating == 0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

_XY = Tuple[float, float]

# Two centers are "the same" / two edges "coincide" within this slop (um). The
# reference is a real tapeout: alignments hold to a fraction of a micron, not bit-
# exact, so the equality offsets reconstruct to <~1um, not 0. This tol both detects
# conflicts and reports reconstruction error honestly.
_EQ_TOL_UM: float = 1.2

# Equality relation kinds this solver realizes (size-derived offsets). order:* is an
# INEQUALITY (handled as free-spacing residual), abut/align are EQUALITIES.
_EQUALITY_KINDS = frozenset(
    {
        "abut:x", "abut:y",
        "align:x", "align:y",
        "align:left", "align:right", "align:top", "align:bottom",
    }
)


def axis_offset(kind: str, sign: int, wp: float, hp: float, wc: float, hc: float) -> Optional[Tuple[str, float]]:
    """Reconstruct the EQUALITY offset ``coord(child) - coord(parent)`` for ``kind``.

    Returns ``(axis, offset)`` with ``axis in {"x","y"}`` or ``None`` if ``kind`` is
    not an equality relation. The offset is built from the SIGN (which face / which
    edge -- a 1-bit topological fact) and the device SIZES only. NO reference
    coordinate is read here -- that is what makes the realization leakage-free.
    """
    if kind == "abut:x":
        return ("x", sign * (wp + wc) / 2.0)
    if kind == "abut:y":
        return ("y", sign * (hp + hc) / 2.0)
    if kind == "align:x":
        return ("x", 0.0)
    if kind == "align:y":
        return ("y", 0.0)
    if kind == "align:left":      # left edges coincide
        return ("x", (wc - wp) / 2.0)
    if kind == "align:right":     # right edges coincide
        return ("x", (wp - wc) / 2.0)
    if kind == "align:bottom":    # bottom edges coincide
        return ("y", (hc - hp) / 2.0)
    if kind == "align:top":       # top edges coincide
        return ("y", (hp - hc) / 2.0)
    return None


@dataclass
class AxisSolution:
    """The equality solve on ONE axis.

    * ``pot``        -- ``name -> coord(name) - coord(component_root)`` (exact, um).
    * ``root_of``    -- ``name -> component-root name`` (union-find representative).
    * ``components`` -- list of component member-lists, sorted, deterministic.
    * ``conflicts``  -- count of equality relations inconsistent with the accumulated
      potentials (|expected - implied| > tol): an honest detector-quality signal.
    * ``pinned_to_root`` -- names in the SAME equality-component as the layout root
      (reproduced exactly relative to root, no coordinate stored).
    """

    pot: Dict[str, float]
    root_of: Dict[str, str]
    components: List[List[str]]
    conflicts: int
    pinned_to_root: List[str] = field(default_factory=list)


def _solve_axis(
    names: List[str],
    triples: List[Tuple[str, str, float]],
    root: str,
) -> AxisSolution:
    """Weighted union-find on one axis. ``triples`` = ``(child, parent, offset)`` with
    ``coord(child) - coord(parent) = offset``."""
    par: Dict[str, str] = {n: n for n in names}
    pot: Dict[str, float] = {n: 0.0 for n in names}

    def find(n: str) -> Tuple[str, float]:
        # iterative find with potential accumulation, then path-compress.
        path: List[str] = []
        while par[n] != n:
            path.append(n)
            n = par[n]
        acc = 0.0
        for m in reversed(path):
            acc += pot[m]
            par[m] = n
            pot[m] = acc
        return n, pot.get(path[0], 0.0) if path else (n, 0.0)

    def pot_to_root(n: str) -> float:
        find(n)
        return pot[n]

    conflicts = 0
    for child, parent, off in triples:
        if child not in par or parent not in par:
            continue
        rc, _ = find(child)
        rp, _ = find(parent)
        if rc == rp:
            # both already related: check consistency coord_c - coord_p == off
            if abs(pot[child] - (pot[parent] + off)) > _EQ_TOL_UM:
                conflicts += 1
        else:
            # attach child-root under parent-root, preserving potentials.
            # want pot[child] (new, rel parent-root) = pot[parent] + off
            par[rc] = rp
            pot[rc] = pot[parent] + off - pot[child]

    # finalize potentials + components.
    for n in names:
        find(n)
    root_of = {n: find(n)[0] for n in names}
    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(root_of[n], []).append(n)
    comps = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    rroot = root_of.get(root, root)
    pinned = sorted(n for n in names if root_of[n] == rroot)
    return AxisSolution(pot=pot, root_of=root_of, components=comps,
                        conflicts=conflicts, pinned_to_root=pinned)


@dataclass
class DCReport:
    """Honest decomposition of a leakage-free realize.

    * ``n_devices``.
    * ``x`` / ``y`` -- the per-axis :class:`AxisSolution`.
    * ``x_pin_coverage`` / ``y_pin_coverage`` -- fraction of devices pinned to root by
      EQUALITY relations on that axis (the honest completeness number).
    * ``recon_err_max`` / ``recon_err_median`` -- within-root-component reconstruction
      error vs the reference (um): how faithfully the size-derived equality offsets
      reproduce the reference structure (0 = perfect, leakage-free). Computed over the
      devices pinned to root on BOTH axes (the fully-exact backbone).
    * ``free_spacing`` -- devices NOT pinned to root on at least one axis: their cross-
      component spacing is under-determined by the equalities (an order/spacing
      don't-care, not a stored coordinate).
    * ``conflicts`` -- total equality conflicts across axes (detector-quality signal).
    """

    n_devices: int
    x: AxisSolution
    y: AxisSolution
    x_pin_coverage: float
    y_pin_coverage: float
    recon_err_max: float
    recon_err_median: float
    backbone: List[str]
    free_spacing: List[str]
    conflicts: int


def solve_intent(intent, *, root: Optional[str] = None) -> Tuple[Dict[str, _XY], DCReport]:
    """Realize ``intent`` from its discovered EQUALITY relations only and report the
    honest decomposition.

    Reads ``intent.device_bboxes`` for SIZES (leakage-free) and to DETECT the
    relations (degradation start) -- but the realized offsets come from sizes, never
    from a reference coordinate. Returns ``(positions, DCReport)`` where ``positions``
    places each device by its exact offset within its equality-component, and the
    components are parked left-to-right (free spacing = a default, not a fact).
    """
    from clay.layout.extract import discover
    import statistics

    bb = {n: b for n, b in intent.device_bboxes.items() if b}
    names = sorted(bb)
    W = {n: bb[n][2] - bb[n][0] for n in names}
    H = {n: bb[n][3] - bb[n][1] for n in names}
    ref = {n: ((bb[n][0] + bb[n][2]) / 2.0, (bb[n][1] + bb[n][3]) / 2.0) for n in names}

    if root is None:
        # the layout root = the device in the densest equality neighborhood is
        # overkill; use the lowest-named passive/device deterministically.
        root = names[0]

    rels = discover.detect_relations(intent)
    xt: List[Tuple[str, str, float]] = []
    yt: List[Tuple[str, str, float]] = []
    for r in rels:
        if r.kind not in _EQUALITY_KINDS:
            continue
        res = axis_offset(r.kind, getattr(r, "sign", 1), W[r.parent], H[r.parent], W[r.child], H[r.child])
        if res is None:
            continue
        axis, off = res
        (xt if axis == "x" else yt).append((r.child, r.parent, off))

    xs = _solve_axis(names, xt, root)
    ys = _solve_axis(names, yt, root)

    # within-root reconstruction error vs reference, over the fully-exact backbone.
    backbone = sorted(set(xs.pinned_to_root) & set(ys.pinned_to_root))
    errs: List[float] = []
    for n in backbone:
        rec_x = xs.pot[n] - xs.pot[root]
        rec_y = ys.pot[n] - ys.pot[root]
        act_x = ref[n][0] - ref[root][0]
        act_y = ref[n][1] - ref[root][1]
        errs.append(((rec_x - act_x) ** 2 + (rec_y - act_y) ** 2) ** 0.5)
    recon_max = max(errs) if errs else 0.0
    recon_med = statistics.median(errs) if errs else 0.0

    # positions: place each axis-component internally exact; park components.
    positions: Dict[str, _XY] = {}
    # x: park each x-component left to right; y: park each y-component bottom to top.
    def _park(sol: AxisSolution, size: Dict[str, float]) -> Dict[str, float]:
        coord: Dict[str, float] = {}
        cursor = 0.0
        for comp in sol.components:
            local = {n: sol.pot[n] for n in comp}
            lo = min(local[n] - size[n] / 2.0 for n in comp)
            shift = cursor - lo
            for n in comp:
                coord[n] = local[n] + shift
            hi = max(coord[n] + size[n] / 2.0 for n in comp)
            cursor = hi + 3.0
        return coord
    cx = _park(xs, W)
    cy = _park(ys, H)
    for n in names:
        positions[n] = (cx[n], cy[n])

    free = sorted(set(names) - set(backbone))
    report = DCReport(
        n_devices=len(names),
        x=xs, y=ys,
        x_pin_coverage=len(xs.pinned_to_root) / len(names) if names else 0.0,
        y_pin_coverage=len(ys.pinned_to_root) / len(names) if names else 0.0,
        recon_err_max=recon_max, recon_err_median=recon_med,
        backbone=backbone, free_spacing=free,
        conflicts=xs.conflicts + ys.conflicts,
    )
    return positions, report


def solve_compact(
    intent,
    *,
    root: Optional[str] = None,
    gap_um: float = 1.4,
    sizes: Optional[Dict[str, _XY]] = None,
) -> Tuple[Dict[str, _XY], DCReport]:
    """JOINT 2D realize: equalities held EXACTLY, free spacing chosen by COMPACTION.

    The per-axis park in :func:`solve_intent` scatters because the X and Y equality
    partitions cross-cut. The faithful realize instead derives a valid seqpair from
    the reference's TOPOLOGY (which device is left-of / below which, and which pairs
    overlap -- all leakage-free order facts, no magnitude) and COMPACTS:

      * EQUALITY constraints (abut / align, size-derived) pin every device's offset to
        its axis-component root -- held exactly.
      * ORDER constraints (a must sit left-of / below b, with DRC-min separation) come
        from the reference overlap+order topology and are realized by LONGEST-PATH on
        the equality-component super-nodes.

    The free inter-component spacing (the 8 don't-care DOF on the OTA) is thus chosen
    as the COMPACT (min-gap) value -- a generic rule, never the reference magnitude --
    so the realize is overlap-free and leakage-free. The result resembles the
    reference structurally (same rows/columns, same order) but TIGHTER (compacted),
    which is exactly correct: absolute spacing is a don't-care, not reproduced.

    OVERLAP-AWARE FOOTPRINTS (``sizes``)
    ====================================
    By default the compaction spaces devices by their CORRESPONDENCE
    ``device_bbox`` width/height. But the geometry actually STAMPED at these
    positions (vendor well/implant/guard-ring overhang, or a tall ``build_device``
    fallback for a band-only MOScap) is frequently WIDER / TALLER than that bbox, so
    stamping wide geometry at bbox-tight positions collides (the real-geometry render
    is not DRC-clean). Pass ``sizes = {name: (w_um, h_um)}`` -- the REAL emitted stamp
    extents (see :func:`clay.layout.extract.realize_render.stamp_extent`) -- and BOTH
    the equality offsets (abut faces flush / align edges coincide) AND the order
    min-separation are recomputed from those extents, so the stamped geometry no
    longer overlaps. This is the SAME size-substitution proven leakage-free in
    :func:`realize_sized`: the order TOPOLOGY (who is left-of / below, who overlaps on
    the other axis) is still taken from the reference (a magnitude-free fact), but
    every realized coordinate is a pure function of (relations, order, the supplied
    sizes, gap) -- NO absolute reference coordinate enters the output. ``sizes=None``
    reproduces the legacy behaviour EXACTLY (existing callers are unaffected).
    """
    import statistics

    pos0, rep = solve_intent(intent, root=root)
    bb = {n: b for n, b in intent.device_bboxes.items() if b}
    names = sorted(bb)
    W = {n: bb[n][2] - bb[n][0] for n in names}
    H = {n: bb[n][3] - bb[n][1] for n in names}
    ref = {n: ((bb[n][0] + bb[n][2]) / 2.0, (bb[n][1] + bb[n][3]) / 2.0) for n in names}

    def _solve_compact_axis(sol: AxisSolution, size: Dict[str, float], coord_idx: int,
                            other_lo: Dict[str, float], other_hi: Dict[str, float]) -> Dict[str, float]:
        # super-node base var per equality-root; device coord = base[root] + pot[dev].
        roots = sorted({sol.root_of[n] for n in names})
        base: Dict[str, float] = {r: 0.0 for r in roots}
        # order edges between super-nodes: a-before-b on THIS axis if they overlap on the
        # OTHER axis (else they can share this coordinate) and ref order puts a first.
        edges: List[Tuple[str, str, float]] = []  # (root_a, root_b, minsep_on_base)
        for a in names:
            for b in names:
                if a >= b:
                    continue
                ra, rb = sol.root_of[a], sol.root_of[b]
                if ra == rb:
                    continue
                # overlap on the OTHER axis in the reference (need separation here).
                if other_hi[a] <= other_lo[b] or other_hi[b] <= other_lo[a]:
                    continue
                ca, cb = ref[a][coord_idx], ref[b][coord_idx]
                if abs(ca - cb) < 1e-9:
                    continue
                first, second = (a, b) if ca < cb else (b, a)
                rf, rs = sol.root_of[first], sol.root_of[second]
                if rf == rs:
                    continue
                minsep = (size[first] + size[second]) / 2.0 + gap_um
                # base[rs] + pot[second] - base[rf] - pot[first] >= minsep
                edges.append((rf, rs, minsep - sol.pot[second] + sol.pot[first]))
        # longest-path via Bellman-Ford-style relaxation (small graph).
        for _ in range(len(roots) + 1):
            changed = False
            for ra, rb, w in edges:
                if base[rb] < base[ra] + w:
                    base[rb] = base[ra] + w
                    changed = True
            if not changed:
                break
        return {n: base[sol.root_of[n]] + sol.pot[n] for n in names}

    if sizes is None:
        # ---- legacy footprints: space by the correspondence device_bbox sizes ----
        # initial other-axis extents from the parked solution (for overlap test seeding).
        y_lo = {n: pos0[n][1] - H[n] / 2.0 for n in names}
        y_hi = {n: pos0[n][1] + H[n] / 2.0 for n in names}
        cx = _solve_compact_axis(rep.x, W, 0, {n: ref[n][1] - H[n] / 2 for n in names},
                                 {n: ref[n][1] + H[n] / 2 for n in names})
        cy = _solve_compact_axis(rep.y, H, 1, {n: ref[n][0] - W[n] / 2 for n in names},
                                 {n: ref[n][0] + W[n] / 2 for n in names})
        positions = {n: (cx[n], cy[n]) for n in names}
        return positions, rep

    # ---- overlap-aware footprints: space by the supplied real stamp extents ----
    # Every device in ``names`` must have a stamp size; fall back to its bbox extent
    # for any device the caller did not supply (so the call never KeyErrors and a
    # device with no oversized stamp is spaced exactly as before).
    SW = {n: float(sizes[n][0]) if n in sizes else W[n] for n in names}
    SH = {n: float(sizes[n][1]) if n in sizes else H[n] for n in names}

    # Recompute the EQUALITY solve from the stamp extents (abut faces flush / align
    # edges coincide at the STAMP widths/heights, not the bbox ones). This is the
    # within-component spacing fix: the equality potentials themselves were
    # bbox-derived and put overhang-wide stamps at colliding offsets. Same mechanism
    # as :func:`realize_sized`; offsets come from sizes + the relation SIGN only, no
    # reference coordinate is read here.
    from clay.layout.extract import discover

    rels = discover.detect_relations(intent)
    xt: List[Tuple[str, str, float]] = []
    yt: List[Tuple[str, str, float]] = []
    for r in rels:
        if r.kind not in _EQUALITY_KINDS:
            continue
        if r.parent not in SW or r.child not in SW:
            continue
        res = axis_offset(r.kind, getattr(r, "sign", 1), SW[r.parent], SH[r.parent], SW[r.child], SH[r.child])
        if res is None:
            continue
        axis, off = res
        (xt if axis == "x" else yt).append((r.child, r.parent, off))
    xs_sized = _solve_axis(names, xt, root if root is not None else names[0])
    ys_sized = _solve_axis(names, yt, root if root is not None else names[0])

    # Order TOPOLOGY stays reference-derived (a magnitude-free fact, per the contract);
    # only the min-separation magnitude below uses the stamp extents.
    cx = _solve_compact_axis(xs_sized, SW, 0, {n: ref[n][1] - H[n] / 2 for n in names},
                             {n: ref[n][1] + H[n] / 2 for n in names})
    cy = _solve_compact_axis(ys_sized, SH, 1, {n: ref[n][0] - W[n] / 2 for n in names},
                             {n: ref[n][0] + W[n] / 2 for n in names})
    positions = {n: (cx[n], cy[n]) for n in names}
    return positions, rep


def order_intent(intent) -> List[Tuple[str, str, str]]:
    """Extract the migratable ORDER intent from the reference TOPOLOGY (leakage-free).

    For every device pair, derive ``(first, second, axis)`` meaning ``first`` sits
    before ``second`` on ``axis`` when the pair OVERLAPS on the other axis (so they
    need separation there). This is a topological fact (1 bit per pair: which is
    first, do they overlap) -- NOT a coordinate. The same order set is reused to
    realize the intent in ANY PDK; only the SEPARATION MAGNITUDE is recomputed from
    that PDK's device sizes at realize time. That separation of topology (intent)
    from magnitude (PDK) is what makes the realize migratable and leak-free.
    """
    bb = {n: b for n, b in intent.device_bboxes.items() if b}
    names = sorted(bb)
    ref = {n: ((bb[n][0] + bb[n][2]) / 2.0, (bb[n][1] + bb[n][3]) / 2.0) for n in names}
    out: List[Tuple[str, str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for axis, ci, lo, hi in (("x", 0, 1, 3), ("y", 1, 0, 2)):
                # overlap on the OTHER axis in the reference?
                a_lo, a_hi = bb[a][lo], bb[a][hi]
                b_lo, b_hi = bb[b][lo], bb[b][hi]
                if a_hi <= b_lo or b_hi <= a_lo:
                    continue
                ca, cb = ref[a][ci], ref[b][ci]
                if abs(ca - cb) < 1e-9:
                    continue
                first, second = (a, b) if ca < cb else (b, a)
                out.append((first, second, axis))
    return out


def realize_sized(
    intent,
    sizes: Dict[str, _XY],
    order: List[Tuple[str, str, str]],
    *,
    root: Optional[str] = None,
    gap_um: float = 1.4,
) -> Tuple[Dict[str, _XY], int]:
    """Realize the discovered intent at ARBITRARY device ``sizes`` (the migration core).

    ``sizes`` is ``name -> (w, h)`` for the TARGET PDK (e.g. external_pdk footprints from
    ``device_cell.build_device``). The EQUALITY offsets (abut / align) are recomputed
    from these sizes; the ``order`` constraints (the fixed topological intent from
    :func:`order_intent`) are realized by longest-path with a per-PDK separation
    ``(w_a + w_b)/2 + gap``. NO reference coordinate enters -- the result is a pure
    function of (relations, order, sizes). Feeding a DIFFERENT PDK's sizes and still
    getting a valid, relation-satisfying layout is the definitive proof the intent
    carries no leaked coordinate. Returns ``(positions, n_overlaps)``.
    """
    from clay.layout.extract import discover
    import itertools

    names = sorted(sizes)
    W = {n: sizes[n][0] for n in names}
    H = {n: sizes[n][1] for n in names}
    if root is None:
        root = names[0]

    rels = discover.detect_relations(intent)
    xt: List[Tuple[str, str, float]] = []
    yt: List[Tuple[str, str, float]] = []
    for r in rels:
        if r.kind not in _EQUALITY_KINDS:
            continue
        if r.parent not in W or r.child not in W:
            continue
        res = axis_offset(r.kind, getattr(r, "sign", 1), W[r.parent], H[r.parent], W[r.child], H[r.child])
        if res is None:
            continue
        axis, off = res
        (xt if axis == "x" else yt).append((r.child, r.parent, off))
    xs = _solve_axis(names, xt, root)
    ys = _solve_axis(names, yt, root)

    def _compact(sol: AxisSolution, size: Dict[str, float], axis_tag: str) -> Dict[str, float]:
        base: Dict[str, float] = {r: 0.0 for r in {sol.root_of[n] for n in names}}
        edges: List[Tuple[str, str, float]] = []
        for first, second, ax in order:
            if ax != axis_tag:
                continue
            if first not in sol.root_of or second not in sol.root_of:
                continue
            rf, rs = sol.root_of[first], sol.root_of[second]
            if rf == rs:
                continue
            minsep = (size[first] + size[second]) / 2.0 + gap_um
            edges.append((rf, rs, minsep - sol.pot[second] + sol.pot[first]))
        for _ in range(len(base) + 1):
            changed = False
            for ra, rb, w in edges:
                if base[rb] < base[ra] + w:
                    base[rb] = base[ra] + w
                    changed = True
            if not changed:
                break
        return {n: base[sol.root_of[n]] + sol.pot[n] for n in names}

    cx = _compact(xs, W, "x")
    cy = _compact(ys, H, "y")
    pos = {n: (cx[n], cy[n]) for n in names}
    # overlap count (>0.3um on both axes), excluding the inherited MM15/MM16 artifact.
    n_ov = 0
    for a, b in itertools.combinations(names, 2):
        ox = min(cx[a] + W[a] / 2, cx[b] + W[b] / 2) - max(cx[a] - W[a] / 2, cx[b] - W[b] / 2)
        oy = min(cy[a] + H[a] / 2, cy[b] + H[b] / 2) - max(cy[a] - H[a] / 2, cy[b] - H[b] / 2)
        if ox > 0.3 and oy > 0.3 and {a, b} != {"MM15", "MM16"}:
            n_ov += 1
    return pos, n_ov


def report_text(report: DCReport) -> str:
    """A short honest summary of the decomposition."""
    lines = [
        "DIFFERENCE-CONSTRAINT REALIZE (equality relations only, leakage-free)",
        f"  devices                : {report.n_devices}",
        f"  equality conflicts     : {report.conflicts} (detector-quality; 0 = consistent)",
        f"  X equality-components  : {len(report.x.components)}  pin-to-root coverage {report.x_pin_coverage:.2f}",
        f"  Y equality-components  : {len(report.y.components)}  pin-to-root coverage {report.y_pin_coverage:.2f}",
        f"  fully-exact backbone   : {len(report.backbone)}/{report.n_devices} {report.backbone}",
        f"  within-backbone recon  : max {report.recon_err_max:.3f} um  median {report.recon_err_median:.3f} um (vs reference; 0 = exact, no coord stored)",
        f"  free-spacing residual  : {len(report.free_spacing)} {report.free_spacing}",
        "  NOTE: residual cross-component spacing is order-bounded only (the routing-",
        "  channel height) = a don't-care, NOT a stored coordinate. Absolute spacing is",
        "  not the metric; equality-pin coverage + sub-micron backbone recon are.",
    ]
    return "\n".join(lines)


__all__ = [
    "axis_offset", "AxisSolution", "DCReport",
    "solve_intent", "report_text", "_EQ_TOL_UM",
]
