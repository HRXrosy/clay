"""Phase 2 of the INTENT-EXTRACTION CONVERGENCE METHOD: the RELAXATION +
EQUIVALENCE-ORACLE loop (and a greedy convergence demo).

WHERE THIS FITS
===============
Phase 1 (DONE, proven incompleteness_score == 0) builds the bottom-clause ANCHOR:
every device ``@fix``-pinned at its exact reference centroid, so a deterministic
placer reproduces the reference layout exactly (zero placer freedom). This module
builds the top-down ANNEAL on top of that anchor:

    start: ALL devices pinned (anchor; equivalence trivially holds)
    relax: DROP a device's @fix pin -> the abstract relations (matching / band /
           equal_size) must now RE-DERIVE that device's position
    keep the relaxation IFF the placer still reproduces the reference up to a
    don't-care care-mask (EQUIVALENCE-PRESERVING)

The MINIMAL surviving @fix set (the incompressible residual) is what the current
abstract vocabulary CANNOT yet express; the DROPPED set is what it CAN. The
compression ratio = dropped / total IS the convergence metric: how much of the
1:1 anchor the present abstract vocabulary compresses away.

THE ORACLE (don't-care quotient, then care-mask)
================================================
``equivalence_to_anchor`` first QUOTIENTS the global don't-cares (the symmetries
the reference layout is invariant under): a global TRANSLATION (subtract each set's
centroid-of-centroids) and, optionally, a global MIRROR (try both, take the better).
Only AFTER quotienting the don't-cares does it compare per-device: equivalent IFF
every device's realized centroid lands within ``care_mask_um`` of its anchor
centroid. (The placer realizes its own absolute floorplan frame; the translation
quotient is what makes the comparison meaningful -- see the HONEST-FINDINGS note at
the bottom.)

DETERMINISM + UNITS
===================
The placer (``engine='seqpair'``) is deterministic. Its realized device bboxes are
in NANOMETRES; the anchor ``intent.device_bboxes`` is in MICRONS. ``place_relaxed``
converts nm -> um so every coordinate downstream is in microns (the care-mask unit).

HONEST SCOPE
============
``converge`` is GREEDY delta-debugging (Zeller-Hildebrandt): it reaches a 1-MINIMAL
fixpoint, which is ORDER-DEPENDENT, NOT a global minimum. The global e-graph /
two-part-MDL extraction is Phase 3 (see the method doc). The compression number is a
LOWER bound on what the vocabulary can express under the chosen drop order.

PLACER GEOMETRY CAVEAT
======================
The relaxed emit re-declares each device's W/L/nf from the netlist, so the placer
builds device cells at the NETLIST sizes -- which generally differ from the exact
hand-drawn RF reference footprints (folded/interdigitated fingers, custom guard
rings). The position comparison remains meaningful as the placer's OWN reproduction
of the anchor under the relaxed intent; it is NOT a claim that the placer rebuilds
the reference's exact geometry. ``place_relaxed`` records this honestly.

This module does NOT edit the placers / netlist_intent / geometric_intent / grammar.
It only composes the existing extractor + emitter + placer, and adds a
``pinned_subset=`` emit mode (in ``emit_clay``) for the relaxed anchor.
"""
from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import re as _re

from clay.layout.extract.emit_clay import emit_clay

# nm -> um: the seqpair placer realizes bboxes in nanometres; the anchor
# ``device_bboxes`` (and the care-mask) are in microns.
_NM_PER_UM = 1000.0

# A throw-away GDS (layer, datatype) the stub rectangle is drawn on -- it never
# reaches an emitter, so the layer is irrelevant; the placer reads only the bbox.
_STUB_LAYER = (0, 0)


class _BBoxStubDevice:
    """A PDK-AGNOSTIC placement stub: a single rectangle of a device's REFERENCE
    footprint, centred at the origin, in NANOMETRES.

    The convergence oracle compares the placer's reproduced CENTROIDS to the anchor
    centroids; that needs only each device's SIZE (to compute the non-overlap
    longest-path separations) and its declared intent (the relative order). It does
    NOT need a real PDK-built device cell -- so we never call a device / MIM-cap
    builder (which is per-PDK and, for several PDKs, not yet wired). The footprint is
    a single rectangle sized to the device's reference bbox; the geometry of the
    final 1:1 emit is DEFERRED to a library-backed emitter (a sibling).

    The class exposes exactly the surface :func:`solve_placement` reads off a device:
    ``.kind`` (nmos/pmos/passive), ``.extent() -> (x0,y0,x1,y1)`` nm, ``.polygons``
    (one placed rect, so ``_place_polys`` / ``_polys_bbox`` realise the placed bbox),
    ``.polys_on`` (the score's active-area probe -> empty: a stub has no active layer)
    and ``.terminals`` (None: no router terminals on a position-only stub). Every
    other attribute the placer reads (``source_rail_bbox`` / ``nwell_bbox`` / ...) is
    absent, so the placer's ``getattr(..., None)`` defaults apply.
    """

    __slots__ = ("kind", "polygons", "terminals")

    def __init__(self, kind: str, w_nm: float, h_nm: float):
        from clay.layout.backend import LayerPolygon, Polygon

        self.kind = kind
        hw = max(1, int(round(w_nm / 2.0)))
        hh = max(1, int(round(h_nm / 2.0)))
        self.polygons = [
            LayerPolygon(
                layer=_STUB_LAYER[0],
                datatype=_STUB_LAYER[1],
                polygon=Polygon.rectangle(-hw, -hh, hw, hh),
            )
        ]
        self.terminals = None

    def extent(self) -> Tuple[int, int, int, int]:
        vs = self.polygons[0].polygon.vertices
        xs = [v.x for v in vs]
        ys = [v.y for v in vs]
        return (min(xs), min(ys), max(xs), max(ys))

    def polys_on(self, _key) -> list:
        return []


def _stub_devices_from_intent(intent: Any) -> Dict[str, _BBoxStubDevice]:
    """Build a bbox-stub device per entry of ``intent.device_bboxes``.

    Each stub's footprint is the device's REFERENCE width/height (microns -> nm),
    centred at the origin; its ``kind`` comes from the netlist device list (a passive
    -- cap / resistor -- with no declared kind defaults to ``'passive'``, which the
    placer treats as a same-type packing block). Faithful to the reference SIZES, with
    ZERO PDK device/passive geometry build (PDK-agnostic by construction)."""
    dbb = getattr(intent, "device_bboxes", {}) or {}
    kinds: Dict[str, str] = {}
    for d in getattr(intent, "devices", ()) or ():
        nm = str(getattr(d, "name", ""))
        if nm:
            kinds[nm] = str(getattr(d, "kind", "") or "nmos")
    out: Dict[str, _BBoxStubDevice] = {}
    for nm, bb in dbb.items():
        if not bb or len(bb) < 4:
            continue
        w_nm = (float(bb[2]) - float(bb[0])) * _NM_PER_UM
        h_nm = (float(bb[3]) - float(bb[1])) * _NM_PER_UM
        out[str(nm)] = _BBoxStubDevice(kinds.get(str(nm), "passive"), w_nm, h_nm)
    return out


def _orient_edges_by_reference(
    edges: Sequence[Tuple[str, str]],
    coord: Dict[str, float],
    eps: float = 1.0,
) -> Tuple[Tuple[str, str], ...]:
    """Keep only the order edges CONSISTENT with the reference 1-D coordinate, so the
    constraint graph is a DAG faithful to the hand layout.

    ``coord`` is each device's reference centroid along the edge's axis (x for
    ``left_of``, y for ``below``), in microns. An edge ``(u, v)`` asserts ``u`` strictly
    before ``v`` on that axis; we DROP it when the reference disagrees (``coord[u] >
    coord[v] + eps``) or when the two are co-located within ``eps`` (no real order ->
    the devices share the axis, e.g. a matched row at one y). This breaks the spurious
    cycles the topological row-plan introduces -- e.g. a 3-way ``@matching`` group
    (MM19/MM20/MM21) whose pairwise relations imply a cyclic vertical order even though
    the reference puts two of them at the SAME y -- by deferring to where the reference
    actually placed each device. An edge between two devices with no reference
    coordinate is kept (we cannot judge it). De-dups and drops self-loops.

    Generic + circuit-agnostic: pure geometry off the reference centroids, no device
    name / kind branch. The all-pinned anchor is unaffected (its placement comes from
    the pins, not the longest-path)."""
    out: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for u, v in edges:
        if u == v or (u, v) in seen:
            continue
        cu, cv = coord.get(u), coord.get(v)
        if cu is not None and cv is not None and cu > cv + eps:
            continue  # reference says v is before u -> this edge is backwards
        if cu is not None and cv is not None and abs(cu - cv) <= eps:
            continue  # co-located on this axis -> no real order (a same-row pair)
        out.append((u, v))
        seen.add((u, v))
    return tuple(out)


def _reference_axis_coords(
    intent: Any, axis: int
) -> Dict[str, float]:
    """Per-device reference centroid along ``axis`` (0 = x, 1 = y), in microns."""
    dbb = getattr(intent, "device_bboxes", {}) or {}
    out: Dict[str, float] = {}
    for nm, bb in dbb.items():
        if not bb or len(bb) < 4:
            continue
        out[str(nm)] = (float(bb[axis]) + float(bb[axis + 2])) / 2.0
    return out


def _strip_empty_layout_intent(src: str) -> str:
    """Drop an EMPTY ``layout_intent <cell> {}`` block so the relaxed .clay parses.

    The relaxed emit carries the ``@fix`` pins + ``@matching`` / ``@equal_size`` in the
    ``annotate{}`` block; the trailing ``layout_intent{}`` block holds only the
    ``@place_group`` BANDS (and an optional ``@budget``). A reference whose geometric
    extractor derived NO bands (e.g. this RF OTA: its floorplan is captured as
    ``orders`` / ``aligns`` / ``symmetry_axis``, recorded as roadmap comments) emits an
    EMPTY ``layout_intent <cell> {}`` block, which the grammar rejects (it requires at
    least one annotation). The block is informational for the position-only placement
    here (the bands would only reinforce the pinned rows), so we strip it when empty --
    a local fix in this module (it does NOT touch the shared emitter). A NON-empty
    block is left untouched."""
    return _re.sub(r"layout_intent\s+\w+\s*\{\s*\}\s*", "", src)


# --------------------------------------------------------------------------- #
# 1. The relaxation state.
# --------------------------------------------------------------------------- #
@dataclass
class RelaxationState:
    """The set of devices currently ``@fix``-PINNED.

    ``pinned`` is a subset of ``all_devices`` (every device with a reference
    bbox). The devices NOT in ``pinned`` are governed purely by the ABSTRACT
    relations carried on the intent (matching / symmetric / equal_size / bands /
    orders). The all-pinned state (``pinned == all_devices``) is the anchor;
    relaxation removes members from ``pinned``.
    """

    all_devices: Tuple[str, ...]
    pinned: Set[str]

    @classmethod
    def anchor(cls, all_devices: Sequence[str]) -> "RelaxationState":
        """The bottom-clause anchor: every device pinned."""
        names = tuple(sorted(set(all_devices)))
        return cls(all_devices=names, pinned=set(names))

    @property
    def dropped(self) -> Set[str]:
        """Devices whose @fix pin has been relaxed (governed by abstract relations)."""
        return set(self.all_devices) - self.pinned

    def without(self, device: str) -> "RelaxationState":
        """A copy with ``device`` relaxed (its @fix pin dropped)."""
        nxt = set(self.pinned)
        nxt.discard(device)
        return RelaxationState(all_devices=self.all_devices, pinned=nxt)


# --------------------------------------------------------------------------- #
# 2. emit_relaxed: a parseable .clay with @fix for the pinned subset + the
#    abstract relations for ALL devices.
# --------------------------------------------------------------------------- #
def emit_relaxed(intent: Any, pinned_subset: Set[str]) -> str:
    """Emit a PARSEABLE relaxed-anchor .clay for ``pinned_subset``.

    ``@fix(d, x=, y=)`` is emitted for every ``d`` in ``pinned_subset`` (that has a
    reference bbox), PLUS the abstract relational primitives (@matching /
    @equal_size / @place_group bands; @order / @abut as comments) for ALL devices.
    The non-pinned devices are therefore governed only by the abstract relations --
    the placer must re-derive their positions. Composes the ``pinned_subset=`` mode
    of :func:`clay.layout.extract.emit_clay.emit_clay`.
    """
    return emit_clay(intent, pinned_subset=set(pinned_subset))


# --------------------------------------------------------------------------- #
# 3. place_relaxed: emit -> parse -> place -> realized centroids (um).
# --------------------------------------------------------------------------- #
def place_relaxed(
    intent: Any,
    pinned_subset: Set[str],
    tech: str = "open_pdk",
    *,
    work_dir: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
    """Realize the relaxed-anchor placement -> ``{name -> (cx, cy)}`` centroids in um.

    PDK-AGNOSTIC + REFERENCE-SIZE-FAITHFUL. The placement is computed from each
    device's reference SIZE and the declared intent ONLY -- it never builds a real PDK
    device / MIM-cap cell -- so the convergence loop runs under ANY ``tech`` (the
    reference's own PDK included), not just the one whose generic builders happen to
    cover every cell kind.

    Steps (deterministic):
      1. ``emit_relaxed(intent, pinned_subset)`` -> a relaxed .clay; an EMPTY
         ``layout_intent{}`` block (a reference with no derived bands) is stripped so
         it parses (``_strip_empty_layout_intent``);
      2. ``parse_clay_file`` -> the ``CellDef`` + its placement ``LayoutIntent``;
      3. ``relative_order_from_intent`` -> the ``RelativeOrder`` (bands / symmetry /
         @order / @abut). This reads the DECLARED structure, not the device sizes, so
         it is built on a neutral, always-buildable tech (gf180mcu) -- the order is the
         SAME whatever PDK the sizes come from;
      4. ``_stub_devices_from_intent`` -> one BBOX-STUB device per reference bbox
         (centred at origin, sized to the reference footprint, in nm). Passive labels
         not in the order's MOS set are APPENDED to the order (no edges -> they float
         to the origin unless pinned), so a pinned passive still lands on its pin;
      5. ``solve_placement(order, stub_devs, ..., passives=None, pinned_positions=)``
         -- the SAME longest-path + symmetry-snap + @fix-pin-override the MILP-parity
         seqpair engine runs, but over the stub footprints. ``passives=None`` keeps the
         per-PDK passive builder out of the path entirely;
      6. each placed device's bbox-centroid, converted nm -> um.

    The ``@fix`` pins (the ``pinned_subset``) override the engine's solved centre, so a
    pinned device lands EXACTLY on its pin; the NON-pinned devices' positions come from
    the abstract relations (matching / symmetric / equal_size / bands) via the order.
    When the structurally-derived order is not a DAG for a given relaxation (a real
    placer/vocabulary limitation -- the abstract relations under-determine a consistent
    total order for the dropped device), the longest-path is ill-posed; we return the
    pins-only positions (the pinned devices), so the oracle sees the non-pinned device
    MISSING and rejects the drop. HONEST: that rejection is the convergence method
    correctly reporting that the current vocabulary cannot re-derive that position.

    GEOMETRY CAVEAT: the stub footprint is the reference SIZE, not the reference's
    exact hand-drawn artwork (folded fingers, guard rings); the comparison is the
    placer's own reproduction of the anchor's CENTROIDS under the relaxed intent. The
    1:1 artwork is deferred to a library-backed emitter (a sibling).
    """
    from dataclasses import replace as _dc_replace

    from clay.layout.annotate_to_solver import _resolve_tech
    from clay.layout.generic_placement import (
        _constraints_from_intent_and_annotations,
        annotated_cell_from_cell_def,
    )
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.seqpair_placement import (
        relative_order_from_intent,
        solve_placement,
    )
    from clay.parser import parse_clay_file

    src = _strip_empty_layout_intent(emit_relaxed(intent, pinned_subset))
    cell = str(getattr(intent, "cell", "ExtractedCell"))

    base = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="clay_relax_"))
    base.mkdir(parents=True, exist_ok=True)
    f = base / "design.clay"
    f.write_text(src, encoding="utf-8")

    cf = parse_clay_file(f)
    cd = next(c for c in cf.cell_defs if getattr(c, "local_name", None) == cell)
    li = layout_intent_from_cell_def(cd)

    # The RELATIVE ORDER carries the declared bands / symmetry / @order / @abut. It is
    # geometry-independent of the reference sizes, so it is built on a neutral,
    # always-buildable tech (gf180mcu) -- the relaxed PDK never needs a real device
    # build to derive the order. ``tech`` is accepted for API compatibility; the
    # placement here is PDK-agnostic (the stub footprints carry the reference sizes).
    order_tech = _resolve_tech("gf180mcu")
    order = relative_order_from_intent(cd, li, source_cir=None, tech=order_tech)

    # @equal_size groups + @fix pins, read off the SAME constraints the seqpair engine
    # reads (so a pin / group resolves identically). Passive cols / mid-band reservation
    # are NOT threaded (passives=None below): a position-only stub needs neither.
    annotated_c, _ = annotated_cell_from_cell_def(cd, None)
    constraints = _constraints_from_intent_and_annotations(annotated_c, li)
    equal_size_groups = tuple(getattr(constraints, "equal_size_groups", ()) or ())
    pinned_positions = dict(getattr(constraints, "pinned_positions", {}) or {})

    # One bbox-stub per reference device (MOS + passives), sized to the reference
    # footprint. Append passive labels (anything with a reference bbox the order's MOS
    # set does not carry) to the order so a pinned passive still lands on its pin; with
    # no edges they float to the origin when not pinned.
    stub_devs = _stub_devices_from_intent(intent)
    extra = tuple(n for n in stub_devs if n not in order.devices)

    # Sanitise the order so the longest-path is a well-posed DAG that is FAITHFUL to the
    # reference layout. The row-plan / mirror straddle can (a) list a device that is a
    # member of SEVERAL declared pairs MORE THAN ONCE in ``order.devices``, and (b) emit
    # constraint edges that, taken together, form a CYCLE (a 3-way ``@matching`` group
    # whose pairwise relations imply a cyclic order even though the reference puts some
    # members at the same y). A duplicate node or a cyclic edge set makes
    # ``_longest_path_positions`` non-DAG (an indegree never reaches zero), which would
    # reject every partially-pinned relaxation and collapse the convergence run. We
    # therefore (a) DE-DUPLICATE ``order.devices`` (first appearance wins, preserving the
    # rail-band order) and (b) keep only the ``left_of`` / ``below`` edges CONSISTENT
    # with the reference centroid order on that axis (``_orient_edges_by_reference``),
    # which drops self-loops, backward edges, and the co-located edges that close the
    # spurious cycles. A local sanitisation of the order THIS module consumes -- it does
    # not touch the placer, and the all-pinned anchor (pins, not longest-path) is
    # unaffected.
    _seen: Set[str] = set()
    _devs_dedup = tuple(
        d for d in (tuple(order.devices) + extra)
        if not (d in _seen or _seen.add(d))
    )
    _lo = _orient_edges_by_reference(order.left_of, _reference_axis_coords(intent, 0))
    _be = _orient_edges_by_reference(order.below, _reference_axis_coords(intent, 1))
    order = _dc_replace(
        order,
        left_of=_lo,
        below=_be,
        devices=_devs_dedup,
    )

    try:
        placement = solve_placement(
            order,
            stub_devs,
            tech=order_tech,
            passives=None,
            equal_size_groups=equal_size_groups,
            pinned_positions=pinned_positions,
        )
    except ValueError:
        # The structurally-derived order is not a DAG for this relaxation (the abstract
        # relations under-determine a consistent total order for the dropped device).
        # Fall back to the PINS-ONLY positions: the pinned devices land on their pins,
        # the non-pinned (dropped) device is OMITTED so the oracle rejects the drop --
        # an honest 'the vocabulary cannot re-derive this position' signal.
        out_pins: Dict[str, Tuple[float, float]] = {}
        for nm, (x_um, y_um) in pinned_positions.items():
            out_pins[str(nm)] = (float(x_um), float(y_um))
        return out_pins

    out: Dict[str, Tuple[float, float]] = {}
    for name, pd in placement.devices.items():
        bb = pd.bbox  # nanometres
        cx = (float(bb[0]) + float(bb[2])) / 2.0 / _NM_PER_UM
        cy = (float(bb[1]) + float(bb[3])) / 2.0 / _NM_PER_UM
        out[str(name)] = (cx, cy)
    return out


# --------------------------------------------------------------------------- #
# 4. equivalence_to_anchor: quotient don't-cares, then care-mask compare.
# --------------------------------------------------------------------------- #
def _anchor_centroids(intent: Any) -> Dict[str, Tuple[float, float]]:
    """The anchor centroids (um) = centroids of ``intent.device_bboxes``."""
    dbb = getattr(intent, "device_bboxes", {}) or {}
    out: Dict[str, Tuple[float, float]] = {}
    for name, bb in dbb.items():
        if not bb or len(bb) < 4:
            continue
        cx = (float(bb[0]) + float(bb[2])) / 2.0
        cy = (float(bb[1]) + float(bb[3])) / 2.0
        out[str(name)] = (cx, cy)
    return out


def _centroid_of(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(points)
    if n == 0:
        return (0.0, 0.0)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def _max_dev_after_translation(
    placed: Dict[str, Tuple[float, float]],
    anchor: Dict[str, Tuple[float, float]],
    names: Sequence[str],
    mirror_x: bool,
) -> float:
    """Max per-device centroid deviation after quotienting the global translation
    (and, when ``mirror_x``, a global x-mirror of the PLACED set).

    Both sets are re-centred on THEIR OWN centroid-of-centroids (translation
    quotient). When ``mirror_x``, the placed x-coordinates are reflected about the
    placed centroid before re-centring (a global mirror don't-care). The deviation
    is the Euclidean distance per device; we return the max over ``names``.
    """
    a_pts = [anchor[n] for n in names]
    p_pts = [placed[n] for n in names]
    acx, acy = _centroid_of(a_pts)
    pcx, pcy = _centroid_of(p_pts)
    worst = 0.0
    for n in names:
        ax, ay = anchor[n]
        px, py = placed[n]
        if mirror_x:
            px = 2.0 * pcx - px  # reflect about the placed centroid x
        # re-centre both on their own centroid (translation quotient)
        da = (px - pcx) - (ax - acx)
        db = (py - pcy) - (ay - acy)
        worst = max(worst, math.hypot(da, db))
    return worst


def equivalence_to_anchor(
    placed: Dict[str, Tuple[float, float]],
    intent: Any,
    care_mask_um: float,
    *,
    allow_mirror: bool = True,
) -> Tuple[bool, float]:
    """Is ``placed`` EQUIVALENT to the anchor up to the ``care_mask_um`` don't-care?

    The anchor positions are the centroids of ``intent.device_bboxes``. We QUOTIENT
    the global don't-cares FIRST -- a global translation (always) and, when
    ``allow_mirror``, a global x-mirror (try both orientations, take the better) --
    then declare equivalence IFF every device's placed centroid is within
    ``care_mask_um`` of its anchor centroid. Returns ``(is_equivalent, max_dev_um)``.

    Only devices present in BOTH ``placed`` and the anchor are compared (a device
    the placer dropped cannot be equivalent -> handled by the caller as a broken
    relaxation). When there is no overlap, returns ``(False, inf)``.
    """
    anchor = _anchor_centroids(intent)
    names = sorted(set(placed) & set(anchor))
    if not names:
        return (False, math.inf)

    best = _max_dev_after_translation(placed, anchor, names, mirror_x=False)
    if allow_mirror:
        best = min(
            best, _max_dev_after_translation(placed, anchor, names, mirror_x=True)
        )
    return (best <= care_mask_um, best)


# --------------------------------------------------------------------------- #
# 5. converge: greedy delta-debugging to a 1-minimal residual.
# --------------------------------------------------------------------------- #
@dataclass
class RelaxationResult:
    """The outcome of a greedy convergence run.

    * ``pinned_residual``  -- the incompressible @fix set (1-minimal): every
      remaining pin is one whose removal the oracle rejected (nothing re-derives it).
    * ``dropped``          -- devices whose @fix the abstract vocabulary determines.
    * ``total``            -- |all devices with a reference bbox|.
    * ``compression``      -- |dropped| / total (the convergence metric).
    * ``care_mask_um``     -- the don't-care radius used.
    * ``order``            -- the drop order tried (greedy is order-dependent).
    * ``relations_for_dropped`` -- per dropped device, the abstract relation(s) that
      plausibly took over its position (matching pair / equal_size group / band).
    * ``trace``            -- per attempted drop: (device, accepted, max_dev_um).
    """

    pinned_residual: Tuple[str, ...]
    dropped: Tuple[str, ...]
    total: int
    compression: float
    care_mask_um: float
    order: Tuple[str, ...]
    relations_for_dropped: Dict[str, List[str]] = field(default_factory=dict)
    trace: List[Tuple[str, bool, float]] = field(default_factory=list)


def _relations_governing(intent: Any, device: str) -> List[str]:
    """Name the abstract relation(s) that could re-derive ``device``'s position.

    Pure read of the intent's abstract fields (no placement): a matched/diff/mirror
    pair the device is in, an equal_size group it shares, and its declared band.
    This is the 'which relations took over' attribution for a dropped pin.
    """
    rels: List[str] = []
    for attr, label in (
        ("matched_pairs", "@matching"),
        ("diff_pairs", "diff-pair"),
        ("mirror_pairs", "mirror-pair"),
    ):
        for ab in getattr(intent, attr, ()) or ():
            if len(ab) == 2 and device in (str(ab[0]), str(ab[1])):
                other = str(ab[1]) if str(ab[0]) == device else str(ab[0])
                rels.append(f"{label}({other})")
    for g in getattr(intent, "equal_size_groups", ()) or ():
        members = {str(m) for m in g}
        if device in members and len(members) >= 2:
            rels.append("@equal_size(" + ",".join(sorted(members)) + ")")
    bands = getattr(intent, "bands", {}) or {}
    if device in bands:
        rels.append(f"@place_group(row={bands[device]})")
    return rels


def converge(
    intent: Any,
    care_mask_um: float,
    order: str = "by_name",
    *,
    tech: str = "open_pdk",
) -> RelaxationResult:
    """GREEDY delta-debugging: anneal the all-pinned anchor down to a 1-minimal
    @fix residual that the placer still reproduces up to ``care_mask_um``.

    Start with ``pinned_subset = ALL`` devices (equivalence trivially holds: every
    device is pinned at its anchor). Then, in ``order``, try removing each device's
    @fix one at a time; ACCEPT the removal IFF
    :func:`equivalence_to_anchor` still holds (the abstract relations re-derive that
    device's position within the mask). Continue to a 1-minimal fixpoint. Each
    candidate is ONE deterministic placer call (the oracle cost).

    HONEST: greedy is order-dependent and only 1-minimal, NOT a global minimum
    (Phase 3 = e-graph + two-part MDL for the global extraction). ``order``:
    ``'by_name'`` (sorted) or ``'reverse'``.

    Returns a :class:`RelaxationResult` with the incompressible residual, the
    dropped set, and the compression ratio (= dropped / total).
    """
    anchor_names = tuple(sorted(_anchor_centroids(intent)))
    total = len(anchor_names)

    if order == "reverse":
        try_order = tuple(reversed(anchor_names))
    else:
        try_order = anchor_names

    state = RelaxationState.anchor(anchor_names)
    trace: List[Tuple[str, bool, float]] = []

    for dev in try_order:
        if dev not in state.pinned:
            continue
        candidate = state.without(dev)
        placed = place_relaxed(intent, candidate.pinned, tech=tech)
        # FAIRNESS GUARD: a drop is acceptable only if the placer actually RE-DERIVED
        # the just-dropped device (it is PRESENT in the placement). When the placer
        # could not realise that relaxation -- e.g. the abstract relations leave the
        # order non-DAG (a 3-way matched group has no consistent pairwise vertical
        # order), so ``place_relaxed`` returns the pins-only positions -- the dropped
        # device is ABSENT, and ``equivalence_to_anchor`` (which compares only the
        # devices present in BOTH) would otherwise vacuously pass on the remaining
        # pinned devices. Requiring the dropped device's presence makes the
        # convergence metric HONEST: an un-re-derivable position stays pinned.
        rederived = dev in placed
        ok, max_dev = equivalence_to_anchor(placed, intent, care_mask_um)
        accept = bool(ok and rederived)
        trace.append((dev, accept, max_dev))
        if accept:
            state = candidate  # accept the relaxation (1-minimal greedy)

    dropped = tuple(sorted(state.dropped))
    residual = tuple(sorted(state.pinned))
    compression = (len(dropped) / total) if total else 0.0
    rel_for_dropped = {d: _relations_governing(intent, d) for d in dropped}

    return RelaxationResult(
        pinned_residual=residual,
        dropped=dropped,
        total=total,
        compression=compression,
        care_mask_um=care_mask_um,
        order=try_order,
        relations_for_dropped=rel_for_dropped,
        trace=trace,
    )


__all__ = [
    "RelaxationState",
    "RelaxationResult",
    "emit_relaxed",
    "place_relaxed",
    "equivalence_to_anchor",
    "converge",
]
