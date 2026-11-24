"""The INVARIANCE HARNESS: extract canonical layout invariants from a placement,
and quantify intent incompleteness by disagreement across faithful placers.

WHAT THIS IS (and is NOT)
=========================
This is the RULER, not the gap-closer. It measures how much of a placement is
pinned by declared intent. It NEVER edits a placer and NEVER adds a primitive.

It consumes a placement RESULT object. Both production engines
(:func:`clay.layout.generic_placement.place_generic` via the MILP path and
:func:`clay.layout.seqpair_placement.place_seqpair`) return the SAME drop-in
:class:`clay.layout.generic_placement.Placement`:

  * ``placement.devices`` : ``{label -> PlacedDevice}`` where each ``PlacedDevice``
    carries ``.bbox = (x0, y0, x1, y1)`` (absolute, manufacturing-grid dbu/nm),
    ``.kind``, ``.terminals``, ``.polys``.
  * ``placement.axis_x`` : the solved vertical symmetry axis (mirror line).
  * ``placement.constraints`` : a :class:`SolverConstraints` carrying the DECLARED
    ``symmetry_pairs`` (matched mirror pairs ``(left, right)``).
  * ``placement.relative_order`` (seqpair only, attached attr) : carries
    ``mirror_pairs``; we read declared pairs from ``constraints`` so the harness is
    engine-symmetric.

THE 8 CANONICAL INVARIANT CATEGORIES (shared taxonomy -- do not invent others):
  1. relative_order : sign of (x_a - x_b) and (y_a - y_b) for every device pair.
  2. adjacency      : which device pairs abut (edge-to-edge within epsilon).
  3. alignment      : equivalence classes sharing a common x-center / y-center.
  4. band           : partition of devices into horizontal / vertical bands.
  5. symmetry_axis  : for each declared matched pair, the mirror-axis coordinate
                      (+ whether it actually mirrors), quotiented to global frame.
  6. size_class     : equivalence classes of equal device footprints (w, h).
  7. spacing        : min inter-band channel spacing on each axis.
  8. orientation    : each device's mirror state + finger direction.

GLOBAL DON'T-CARES (legitimately NOT pinned -- quotiented out before comparison):
  * absolute translation : every coordinate-bearing invariant is made
    translation-invariant by referencing a canonical frame (centroid-of-centroids
    origin) or by being a pure relation (order / adjacency / class membership).
  * global mirror about the cell's vertical axis : detected and quotiented out
    when it makes two runs agree. See :func:`incompleteness_report`.

DETERMINISM: every returned structure is built from SORTED keys / SORTED tuples,
so two equal placements produce byte-equal dicts (asserted in the test).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The canonical taxonomy, in canonical order. Shared across the round.
INVARIANT_CATEGORIES: Tuple[str, ...] = (
    "relative_order",
    "adjacency",
    "alignment",
    "band",
    "symmetry_axis",
    "size_class",
    "spacing",
    "orientation",
)

# Geometric tolerance (nm / dbu). gf180mcu manufacturing grid is 5 nm; device
# extents are hundreds of nm to tens of um. EPS absorbs grid-snap residue so an
# "equal" center / footprint / abutment is recognised as equal, without merging
# genuinely distinct positions. Kept small relative to device pitch.
EPS: int = 10


# --------------------------------------------------------------------------- #
# placement-result access (engine-agnostic, off the public Placement shape)    #
# --------------------------------------------------------------------------- #


def _bbox(pd: Any) -> Tuple[int, int, int, int]:
    """Normalised ``(x0, y0, x1, y1)`` with x0<=x1, y0<=y1 from a PlacedDevice."""
    b = pd.bbox
    x0, y0, x1, y1 = b
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _wh(b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return (b[2] - b[0], b[3] - b[1])


def _devices(placement: Any) -> Dict[str, Any]:
    """``{label -> PlacedDevice}`` off the public Placement (sorted access elsewhere)."""
    return dict(placement.devices)


def _declared_pairs(placement: Any) -> Tuple[Tuple[str, str], ...]:
    """Declared matched mirror pairs ``(a, b)`` (canonical, sorted within pair).

    Read off ``placement.constraints.symmetry_pairs`` (the engine-symmetric source;
    seqpair also attaches ``relative_order.mirror_pairs`` -- the same declared set).
    Each pair is normalised to ``(min(a,b), max(a,b))`` by label so the SET of
    declared pairs is order-independent (a declared pair is unordered; which member
    ends up left is a PLACEMENT decision measured by symmetry_axis/orientation).
    """
    pairs: List[Tuple[str, str]] = []
    cons = getattr(placement, "constraints", None)
    sp = getattr(cons, "symmetry_pairs", ()) if cons is not None else ()
    devs = _devices(placement)
    for ab in sp or ():
        if len(ab) != 2:
            continue
        a, b = ab
        if a in devs and b in devs:
            pairs.append((a, b) if a <= b else (b, a))
    # de-dup + sort for canonical form
    return tuple(sorted(set(pairs)))


# --------------------------------------------------------------------------- #
# 1-D clustering (for bands + alignment classes), deterministic               #
# --------------------------------------------------------------------------- #


def _cluster_1d(items: Sequence[Tuple[str, float]], eps: int) -> List[List[str]]:
    """Single-linkage 1-D clustering by value, gap > eps splits a band.

    Sort by value; start a new cluster whenever the gap to the previous value
    exceeds ``eps``. Deterministic (sorted by value then label). Returns clusters
    as label lists, each label list SORTED, clusters ordered by ascending value.
    """
    if not items:
        return []
    ordered = sorted(items, key=lambda kv: (kv[1], kv[0]))
    clusters: List[List[str]] = [[ordered[0][0]]]
    prev = ordered[0][1]
    for lbl, val in ordered[1:]:
        if val - prev > eps:
            clusters.append([lbl])
        else:
            clusters[-1].append(lbl)
        prev = val
    return [sorted(c) for c in clusters]


def _class_of(clusters: List[List[str]]) -> Dict[str, int]:
    """Map each label to its cluster index (clusters already in canonical order)."""
    out: Dict[str, int] = {}
    for i, c in enumerate(clusters):
        for lbl in c:
            out[lbl] = i
    return out


# --------------------------------------------------------------------------- #
# the 8 extractors                                                            #
# --------------------------------------------------------------------------- #


def _inv_relative_order(devs: Dict[str, Any], eps: int) -> Dict[str, str]:
    """For every device pair (a<b by label): the sign of (x_a - x_b), (y_a - y_b).

    Returned as ``"a|b" -> "<x-rel>,<y-rel>"`` with each rel in {"-","0","+"}
    ("-" = a left/below b, "+" = a right/above b, "0" = same within eps). Keyed by
    SORTED pair label so the partial order is canonical and translation-invariant
    (a pure relation: pure difference of coordinates).
    """
    out: Dict[str, str] = {}
    centers = {lbl: _center(_bbox(pd)) for lbl, pd in devs.items()}

    def _sign(d: float) -> str:
        if d < -eps:
            return "-"
        if d > eps:
            return "+"
        return "0"

    for a, b in combinations(sorted(devs), 2):
        xa, ya = centers[a]
        xb, yb = centers[b]
        out[f"{a}|{b}"] = f"{_sign(xa - xb)},{_sign(ya - yb)}"
    return out


def _inv_adjacency(devs: Dict[str, Any], eps: int) -> Dict[str, bool]:
    """Which device pairs ABUT (edge-to-edge within eps) vs are separated.

    A pair abuts iff their bboxes touch on one axis (gap within eps, possibly
    slightly overlapping) AND overlap on the other axis (share an edge span). Keyed
    by SORTED pair label -> bool. Translation-invariant (relative gap only).
    """
    out: Dict[str, bool] = {}
    boxes = {lbl: _bbox(pd) for lbl, pd in devs.items()}

    def _abut(p: Tuple[int, int, int, int], q: Tuple[int, int, int, int]) -> bool:
        # vertical-edge abut: right of one within eps of left of other, y-spans overlap.
        x_touch = (
            abs(p[2] - q[0]) <= eps or abs(q[2] - p[0]) <= eps
        )
        y_overlap = min(p[3], q[3]) - max(p[1], q[1]) > eps
        if x_touch and y_overlap:
            return True
        # horizontal-edge abut: top of one within eps of bottom of other, x-spans overlap.
        y_touch = (
            abs(p[3] - q[1]) <= eps or abs(q[3] - p[1]) <= eps
        )
        x_overlap = min(p[2], q[2]) - max(p[0], q[0]) > eps
        return bool(y_touch and x_overlap)

    for a, b in combinations(sorted(devs), 2):
        out[f"{a}|{b}"] = _abut(boxes[a], boxes[b])
    return out


def _inv_alignment(devs: Dict[str, Any], eps: int) -> Dict[str, List[List[str]]]:
    """Equivalence classes of devices sharing a common x-center / y-center.

    Each is a clustering of the respective center coordinate; gap>eps splits a
    class. Returned as ``{"x_center": [[...], ...], "y_center": [[...], ...]}`` with
    each class SORTED and classes in ascending-coordinate order. Membership is
    translation-invariant (clustering of relative spread). Singleton classes are
    INCLUDED (a device aligned with nobody is its own class) so the partition is
    total and comparable across runs.
    """
    centers = {lbl: _center(_bbox(pd)) for lbl, pd in devs.items()}
    xs = [(lbl, c[0]) for lbl, c in centers.items()]
    ys = [(lbl, c[1]) for lbl, c in centers.items()]
    return {
        "x_center": _cluster_1d(xs, eps),
        "y_center": _cluster_1d(ys, eps),
    }


def _inv_band(devs: Dict[str, Any], eps: int) -> Dict[str, List[List[str]]]:
    """Partition into HORIZONTAL bands (cluster y-centers) and VERTICAL bands
    (cluster x-centers).

    A horizontal band is a row tier (devices at a common y); a vertical band is a
    column tier (common x). Uses a band-scale epsilon: device pitch, not grid
    residue, separates bands, so we cluster with ``max(eps, 0.4 * median height)``
    on y and ``max(eps, 0.4 * median width)`` on x -- a gap larger than ~40% of a
    typical device dimension starts a new tier. Returned as
    ``{"horizontal": [...], "vertical": [...]}`` (each band SORTED, bands ascending).
    Translation-invariant (clusters of relative spread).
    """
    centers = {lbl: _center(_bbox(pd)) for lbl, pd in devs.items()}
    whs = [_wh(_bbox(pd)) for pd in devs.values()]
    if whs:
        med_w = sorted(w for w, _h in whs)[len(whs) // 2]
        med_h = sorted(h for _w, h in whs)[len(whs) // 2]
    else:
        med_w = med_h = 0
    eps_y = max(eps, int(0.4 * med_h))
    eps_x = max(eps, int(0.4 * med_w))
    ys = [(lbl, c[1]) for lbl, c in centers.items()]
    xs = [(lbl, c[0]) for lbl, c in centers.items()]
    return {
        "horizontal": _cluster_1d(ys, eps_y),
        "vertical": _cluster_1d(xs, eps_x),
    }


def _inv_symmetry_axis(
    devs: Dict[str, Any], pairs: Tuple[Tuple[str, str], ...], eps: int,
    frame_x: float,
) -> Dict[str, Dict[str, Any]]:
    """For each declared matched pair: the mirror-axis x they reflect about, and
    whether they ACTUALLY mirror.

    For a pair (a, b): axis = midpoint of their x-centers, made translation-
    invariant by subtracting ``frame_x`` (the canonical centroid origin). ``mirrors``
    is True iff (i) their y-centers coincide within eps AND (ii) their footprints
    are equal within eps (a real mirror pair sits at one y-baseline with equal
    geometry). Keyed by the SORTED pair label ``"a|b"``. The axis value is rounded to
    EPS so grid residue does not spuriously disagree.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for a, b in pairs:
        ba, bb = _bbox(devs[a]), _bbox(devs[b])
        ca, cb = _center(ba), _center(bb)
        axis = (ca[0] + cb[0]) / 2.0 - frame_x
        same_y = abs(ca[1] - cb[1]) <= eps
        wa, ha = _wh(ba)
        wb, hb = _wh(bb)
        same_size = abs(wa - wb) <= eps and abs(ha - hb) <= eps
        out[f"{a}|{b}"] = {
            "axis_x": round(axis / EPS) * EPS,
            "mirrors": bool(same_y and same_size),
        }
    return out


def _inv_size_class(devs: Dict[str, Any], eps: int) -> List[List[str]]:
    """Equivalence classes of equal device footprints (w, h within eps).

    Two devices share a class iff |w_a-w_b|<=eps and |h_a-h_b|<=eps. Built by
    canonical-bucketing the rounded (w, h); each class SORTED, classes ordered by
    their (w, h) key. Footprint is intrinsically translation-invariant.
    """
    buckets: Dict[Tuple[int, int], List[str]] = {}
    for lbl in sorted(devs):
        w, h = _wh(_bbox(devs[lbl]))
        key = (round(w / max(1, eps)) * max(1, eps),
               round(h / max(1, eps)) * max(1, eps))
        buckets.setdefault(key, []).append(lbl)
    return [sorted(buckets[k]) for k in sorted(buckets)]


def _inv_spacing(devs: Dict[str, Any], eps: int) -> Dict[str, Optional[int]]:
    """Min inter-BAND channel spacing on each axis (band = clustered tier).

    The smallest edge-to-edge gap between ADJACENT bands (horizontal bands -> the
    vertical channel between row tiers; vertical bands -> the horizontal channel
    between column tiers). A pure difference of edges, so translation-invariant.
    Returned as ``{"horizontal_channel": int|None, "vertical_channel": int|None}``
    (None when there is fewer than 2 bands on that axis). Rounded to EPS.
    """
    bands = _inv_band(devs, eps)
    boxes = {lbl: _bbox(pd) for lbl, pd in devs.items()}

    def _channel(band_list: List[List[str]], lo_i: int, hi_i: int) -> Optional[int]:
        if len(band_list) < 2:
            return None
        gaps: List[int] = []
        for i in range(len(band_list) - 1):
            lower = band_list[i]
            upper = band_list[i + 1]
            lower_hi = max(boxes[d][hi_i] for d in lower)
            upper_lo = min(boxes[d][lo_i] for d in upper)
            gaps.append(upper_lo - lower_hi)
        g = min(gaps)
        return round(g / EPS) * EPS

    return {
        # horizontal bands are stacked vertically -> the channel is along y (idx 1/3)
        "vertical_channel": _channel(bands["horizontal"], 1, 3),
        # vertical bands are side-by-side -> the channel is along x (idx 0/2)
        "horizontal_channel": _channel(bands["vertical"], 0, 2),
    }


def _inv_orientation(
    devs: Dict[str, Any], pairs: Tuple[Tuple[str, str], ...], eps: int,
    frame_x: float,
) -> Dict[str, str]:
    """Each device's mirror state + finger direction.

    The public Placement does not expose a per-device rotation field, so we derive
    the OBSERVABLE orientation from geometry + the declared pairs:

      * ``side`` : which side of the canonical frame axis the device center sits on
        ("L" left, "R" right, "C" on-axis within eps) -- this is the placement
        decision a global mirror flips.
      * ``finger`` : finger direction inferred from the bbox aspect ("H" wide / "V"
        tall / "S" square within eps) -- a proxy for the multi-finger run direction
        the public polys imply (we do not parse poly layers here; aspect is the
        engine-symmetric observable).
      * ``mirrored_member`` : "Y" if the device is the member of a declared pair that
        is geometrically the reflected (right-of-axis) twin, "N" otherwise / unpaired.

    Keyed by device label -> ``"<side>/<finger>/<mirrored_member>"``. ``side`` and
    ``mirrored_member`` are relative to the canonical frame so translation-invariant;
    under a global mirror, ``side`` L<->R flips coherently (the quotient detects it).
    """
    # which labels are the right-of-axis (reflected) twin of a declared pair
    reflected: Dict[str, bool] = {}
    for a, b in pairs:
        xa = _center(_bbox(devs[a]))[0]
        xb = _center(_bbox(devs[b]))[0]
        if xa <= xb:
            reflected[a] = False
            reflected[b] = True
        else:
            reflected[a] = True
            reflected[b] = False

    out: Dict[str, str] = {}
    for lbl in sorted(devs):
        b = _bbox(devs[lbl])
        cx = _center(b)[0]
        dx = cx - frame_x
        if dx < -eps:
            side = "L"
        elif dx > eps:
            side = "R"
        else:
            side = "C"
        w, h = _wh(b)
        if abs(w - h) <= eps:
            finger = "S"
        elif w > h:
            finger = "H"
        else:
            finger = "V"
        mm = "Y" if reflected.get(lbl, False) else "N"
        out[lbl] = f"{side}/{finger}/{mm}"
    return out


# --------------------------------------------------------------------------- #
# canonical frame + the top-level extractor                                   #
# --------------------------------------------------------------------------- #


def _frame_x(devs: Dict[str, Any]) -> float:
    """Canonical x-origin: the mean of device-center x's (centroid-of-centroids).

    Subtracting it makes every x-referenced invariant translation-invariant on the
    x axis (the only axis where the global-mirror don't-care lives). y is handled by
    pure relations (order / band clustering) so it needs no explicit origin.
    """
    if not devs:
        return 0.0
    xs = [_center(_bbox(pd))[0] for pd in devs.values()]
    return sum(xs) / len(xs)


def extract_invariants(placement: Any, *, eps: int = EPS) -> Dict[str, Any]:
    """Extract the 8 canonical invariants from a placement result.

    ``placement`` is the drop-in :class:`clay.layout.generic_placement.Placement`
    both engines return (``.devices`` -> PlacedDevice with ``.bbox``, plus
    ``.constraints.symmetry_pairs`` for declared matched pairs).

    Returns a CANONICAL (sorted-key, deterministic) dict keyed by the 8 category
    names in :data:`INVARIANT_CATEGORIES`. Two equal placements produce byte-equal
    output (the basis for the placer-invariance test). Every coordinate-bearing
    invariant is translation-invariant; the global-mirror don't-care is quotiented
    later in :func:`incompleteness_report` (not here -- here we report the raw
    canonical frame).
    """
    devs = _devices(placement)
    pairs = _declared_pairs(placement)
    fx = _frame_x(devs)
    return {
        "relative_order": _inv_relative_order(devs, eps),
        "adjacency": _inv_adjacency(devs, eps),
        "alignment": _inv_alignment(devs, eps),
        "band": _inv_band(devs, eps),
        "symmetry_axis": _inv_symmetry_axis(devs, pairs, eps, fx),
        "size_class": _inv_size_class(devs, eps),
        "spacing": _inv_spacing(devs, eps),
        "orientation": _inv_orientation(devs, pairs, eps, fx),
    }


# --------------------------------------------------------------------------- #
# run adapter onto the EXISTING placement entry point                         #
# --------------------------------------------------------------------------- #


def run_placement(
    cell_def: Any,
    engine: str,
    *,
    intent: Any = None,
    criteria: Any = None,
    source_cir: Any = None,
    tech: Any = None,
    seed: Optional[int] = None,
) -> Any:
    """Invoke the EXISTING placement entry point and return the final Placement.

    Thin adapter onto :func:`clay.layout.place_refine.place_to_intent` -- the same
    entry point the flow uses -- forcing ``engine in {"milp", "seqpair"}`` (so the
    declared ``@place_policy(engine=...)`` can be overridden either way). Returns
    ``ConvergedPlacement.placement`` (the final :class:`Placement`).

    SEED / PERTURBATION HONESTY: ``place_to_intent`` accepts NO tie-break seed -- the
    MILP and seqpair solvers are deterministic given the inputs. So ``seed`` is NOT a
    randomized tie-break; it is accepted only for API symmetry and, when non-None,
    documented as a no-op (it does not perturb the solve). The REAL perturbations the
    harness applies are exercised by :func:`incompleteness_report`:
      (i)  the two ENGINES (milp vs seqpair) -- different faithful placers;
      (ii) deterministic INPUT PERMUTATION of the device declaration order (rebuilt
           CellDef with reordered devices), which both engines consume order-sensitively.
    See that function's docstring for exactly what was applied to a given report.
    """
    from clay.layout.place_refine import place_to_intent

    cp = place_to_intent(
        cell_def,
        intent,
        criteria,
        source_cir=source_cir,
        tech=tech,
        engine=engine,
    )
    return cp.placement


def _reorder_cell_devices(cell_def: Any, perm: Sequence[int]) -> Any:
    """Return a COPY of ``cell_def`` with its topology devices reordered by ``perm``.

    A deterministic INPUT PERMUTATION: both placers read the device declaration
    order (the row-plan / longest-path node order derive from it), so permuting it
    is a faithful perturbation that must NOT change any invariant-that-matters if
    the intent is complete. Returns ``None`` (caller skips this perturbation) if the
    cell shape does not expose a reorderable device list -- reported honestly.

    We locate the device list generically: a ``CellDef`` exposes its topology body;
    we look for a ``devices`` sequence on the cell or its ``topology``/``body``. If
    none is found we return None (do not fabricate a permutation).
    """
    import copy

    cd = copy.deepcopy(cell_def)
    # Probe a few generic shapes for the device list.
    for holder_name in ("topology", "body", "block"):
        holder = getattr(cd, holder_name, None)
        if holder is None:
            continue
        for attr in ("devices", "device_defs", "instances"):
            seq = getattr(holder, attr, None)
            if isinstance(seq, list) and len(seq) == len(perm):
                reordered = [seq[i] for i in perm]
                try:
                    setattr(holder, attr, reordered)
                    return cd
                except Exception:  # noqa: BLE001 - frozen / immutable -> not reorderable
                    return None
    # direct on the cell
    for attr in ("devices", "device_defs"):
        seq = getattr(cd, attr, None)
        if isinstance(seq, list) and len(seq) == len(perm):
            try:
                setattr(cd, attr, [seq[i] for i in perm])
                return cd
            except Exception:  # noqa: BLE001
                return None
    return None


# --------------------------------------------------------------------------- #
# disagreement + the incompleteness report                                    #
# --------------------------------------------------------------------------- #


def _global_mirror_invariants(inv: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the GLOBAL-MIRROR don't-care to an invariant dict (vertical-axis flip).

    A global mirror about the cell's vertical axis is a legitimate don't-care: it
    negates every x relation, swaps L<->R sides, and negates each symmetry axis_x
    (the frame is the centroid so the mirror fixes it). It leaves y-relations,
    adjacency, alignment/band MEMBERSHIP, size classes, and finger H/V unchanged.
    Returns the mirrored view; :func:`_disagreements` tries BOTH the raw and the
    mirrored view of run B and keeps whichever agrees more with run A (quotienting
    out the don't-care).
    """
    out: Dict[str, Any] = {}

    def _flip_sign(s: str) -> str:
        return {"-": "+", "+": "-", "0": "0"}.get(s, s)

    # relative_order: negate the x-relation, keep the y-relation.
    ro = {}
    for k, v in inv["relative_order"].items():
        xr, yr = v.split(",")
        ro[k] = f"{_flip_sign(xr)},{yr}"
    out["relative_order"] = ro

    # adjacency / size_class: mirror preserves abutment + footprint classes.
    out["adjacency"] = dict(inv["adjacency"])
    out["size_class"] = inv["size_class"]

    # alignment / band: membership is preserved, but a vertical mirror REVERSES the
    # x-ordered class sequence (ascending-x becomes descending-x). The y-ordered
    # classes are unchanged. Reverse only the x-referenced sub-axes so the canonical
    # (coordinate-ordered) partition matches A after the mirror.
    out["alignment"] = {
        "x_center": list(reversed(inv["alignment"]["x_center"])),
        "y_center": inv["alignment"]["y_center"],
    }
    out["band"] = {
        "horizontal": inv["band"]["horizontal"],
        "vertical": list(reversed(inv["band"]["vertical"])),
    }

    # symmetry_axis: negate axis_x (centroid frame is mirror-fixed), keep mirrors.
    # The two pair members swap which is the reflected twin, but the pair KEY is the
    # sorted label tuple (order-independent), so the keyed axis/mirrors map is stable.
    sa = {}
    paired_members: set = set()
    for k, v in inv["symmetry_axis"].items():
        sa[k] = {"axis_x": -v["axis_x"], "mirrors": v["mirrors"]}
        a, b = k.split("|")
        paired_members.add(a)
        paired_members.add(b)
    out["symmetry_axis"] = sa

    # spacing: channels are unsigned widths -> unchanged.
    out["spacing"] = dict(inv["spacing"])

    # orientation: flip L<->R side (keep finger). The mirrored-member flag (mm) flips
    # ONLY for devices that are members of a declared pair (the right-of-axis twin
    # becomes the left-of-axis twin under the mirror); an UNPAIRED device keeps its
    # mm=N (it has no reflected-twin role to swap).
    ori = {}
    for lbl, v in inv["orientation"].items():
        side, finger, mm = v.split("/")
        side = {"L": "R", "R": "L", "C": "C"}[side]
        if lbl in paired_members and mm in ("Y", "N"):
            mm = {"Y": "N", "N": "Y"}[mm]
        ori[lbl] = f"{side}/{finger}/{mm}"
    out["orientation"] = ori
    return out


def _disagreements_one(a: Any, b: Any) -> List[str]:
    """Flat list of disagreeing KEYS between two invariant SUB-structures.

    Handles the dict-of-scalar (relative_order / adjacency / orientation /
    symmetry_axis / spacing) and the list-of-classes (alignment-axis / band-axis /
    size_class) shapes. For a dict, a disagreement is any key whose value differs (or
    is missing on one side). For a list-of-classes, a disagreement is reported as the
    keyed string of the whole partition if the partitions differ (the partition is the
    decision, so it counts as one disagreeing decision per differing axis/class set).
    """
    out: List[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                out.append(k)
    elif isinstance(a, list) and isinstance(b, list):
        if a != b:
            out.append("partition")
    else:
        if a != b:
            out.append("value")
    return out


def _category_disagreements(inv_a: Dict[str, Any], inv_b: Dict[str, Any]) -> Dict[str, List[str]]:
    """Per-category disagreeing decisions between two FULL invariant dicts.

    For the nested categories (alignment / band have an x/y or h/v sub-axis;
    symmetry_axis values are dicts; spacing values are scalars), we descend one level
    where needed and prefix the disagreeing key with the sub-axis so the report names
    the exact decision (e.g. ``alignment.x_center.partition``).
    """
    disagree: Dict[str, List[str]] = {c: [] for c in INVARIANT_CATEGORIES}

    # flat-dict categories
    for cat in ("relative_order", "adjacency", "orientation"):
        disagree[cat] = _disagreements_one(inv_a[cat], inv_b[cat])

    # size_class: a single list-of-classes
    disagree["size_class"] = _disagreements_one(inv_a["size_class"], inv_b["size_class"])

    # alignment + band: dict of {sub_axis -> list-of-classes}
    for cat in ("alignment", "band"):
        d: List[str] = []
        a, b = inv_a[cat], inv_b[cat]
        for sub in sorted(set(a) | set(b)):
            for k in _disagreements_one(a.get(sub), b.get(sub)):
                d.append(f"{sub}.{k}")
        disagree[cat] = d

    # symmetry_axis: dict of {pair -> {axis_x, mirrors}}
    d = []
    a, b = inv_a["symmetry_axis"], inv_b["symmetry_axis"]
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            d.append(k)
    disagree["symmetry_axis"] = d

    # spacing: dict of {channel -> int|None}
    disagree["spacing"] = _disagreements_one(inv_a["spacing"], inv_b["spacing"])

    return disagree


def _total_disagreements(disagree: Dict[str, List[str]]) -> int:
    return sum(len(v) for v in disagree.values())


def _best_quotiented_disagreements(
    inv_a: Dict[str, Any], inv_b: Dict[str, Any]
) -> Tuple[Dict[str, List[str]], bool]:
    """Disagreements between two invariant dicts, quotienting out the GLOBAL MIRROR.

    Compute disagreements against run B AS-IS and against the GLOBAL-MIRRORED view of
    run B; keep whichever total is smaller (so a pure global-mirror difference -- a
    legitimate don't-care -- scores 0, not the full x-relation flip). Returns
    ``(disagreements, mirror_applied)``.
    """
    raw = _category_disagreements(inv_a, inv_b)
    mirrored = _category_disagreements(inv_a, _global_mirror_invariants(inv_b))
    if _total_disagreements(mirrored) < _total_disagreements(raw):
        return mirrored, True
    return raw, False


@dataclass
class IncompletenessReport:
    """The placer-invariance report for one design.

    ``incompleteness_score`` = total count of disagreeing invariants across the runs
    (0 = complete up to the global don't-cares). ``per_category`` maps each of the 8
    categories to the list of disagreeing decisions. ``runs`` records what actually
    ran (engine + perturbation label + whether it succeeded). ``errors`` records any
    engine/perturbation that FAILED to produce a placement (itself a realization-gap
    incompleteness). ``mirror_quotiented`` flags whether a global-mirror don't-care
    was detected + removed. ``perturbations_applied`` documents EXACTLY what was done.
    """

    incompleteness_score: int
    per_category: Dict[str, List[str]]
    runs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    pairwise: List[Dict[str, Any]] = field(default_factory=list)
    mirror_quotiented: bool = False
    perturbations_applied: List[str] = field(default_factory=list)
    device_sets: Dict[str, List[str]] = field(default_factory=dict)
    realization_gaps: List[str] = field(default_factory=list)


def incompleteness_report(
    cell_def: Any,
    *,
    engines: Sequence[str] = ("milp", "seqpair"),
    seeds: Sequence[Optional[int]] = (None,),
    intent: Any = None,
    criteria: Any = None,
    source_cir: Any = None,
    tech: Any = None,
    permute_inputs: bool = True,
    eps: int = EPS,
) -> IncompletenessReport:
    """Run the SAME declared intent through different faithful placers + input
    permutations and quantify the disagreement (the incompleteness).

    PERTURBATIONS ACTUALLY APPLIED:
      * ENGINES: each engine in ``engines`` (default milp + seqpair) -- two distinct
        faithful placers consuming the identical declared intent.
      * SEEDS: ``place_to_intent`` accepts NO tie-break seed (both solvers are
        deterministic), so ``seeds`` does not randomize the solve. We still iterate it
        for API symmetry; with the default ``(None,)`` it is a single run per engine.
        This is documented honestly in ``perturbations_applied``.
      * INPUT PERMUTATION (when ``permute_inputs`` and the cell exposes a reorderable
        device list): one deterministic reversal of the device declaration order,
        re-run through each engine. Both placers are order-sensitive, so a complete
        intent must be invariant to it. If the cell shape does not expose a
        reorderable device list, this perturbation is SKIPPED and that is recorded.

    The score is the total disagreeing-invariant count across all successful runs
    (each pair compared after quotienting absolute translation + the global mirror).
    An engine/perturbation that ERRORS is recorded in ``errors`` -- an engine that
    cannot realize the declared intent is itself a realization/incompleteness gap.
    """
    runs: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    perturbations: List[str] = []

    # Build the (cell, perturbation-label) list.
    variants: List[Tuple[Any, str]] = [(cell_def, "decl_order")]
    perturbations.append("engines=" + "+".join(engines))
    perturbations.append(
        "seeds=" + ",".join(str(s) for s in seeds)
        + " (NOTE: place_to_intent takes no tie-break seed; seeds are a no-op, "
        "documented honestly -- deterministic solvers)"
    )
    if permute_inputs:
        # one deterministic permutation: reverse the device declaration order.
        # probe the count via a deep attribute walk done inside _reorder; we pass a
        # reversed identity once we know the length, so build it lazily below.
        perm_cell = _try_reverse_devices(cell_def)
        if perm_cell is not None:
            variants.append((perm_cell, "reversed_decl_order"))
            perturbations.append("input_permutation=reversed_declaration_order")
        else:
            perturbations.append(
                "input_permutation=SKIPPED (cell exposes no reorderable device list)"
            )

    # Run every (engine, variant) combo.
    extracted: List[Dict[str, Any]] = []
    for engine in engines:
        for seed in seeds:
            for (variant_cell, plabel) in variants:
                tag = f"{engine}/{plabel}/seed={seed}"
                try:
                    pl = run_placement(
                        variant_cell, engine,
                        intent=intent, criteria=criteria,
                        source_cir=source_cir, tech=tech, seed=seed,
                    )
                    inv = extract_invariants(pl, eps=eps)
                    extracted.append({"tag": tag, "engine": engine,
                                      "perturbation": plabel, "inv": inv,
                                      "labels": sorted(pl.devices)})
                    runs.append({"tag": tag, "engine": engine,
                                 "perturbation": plabel, "ok": True,
                                 "n_devices": len(pl.devices)})
                except Exception as exc:  # noqa: BLE001 - a failed realize IS a finding
                    errors.append({"tag": tag, "error": f"{type(exc).__name__}: {exc}"})
                    runs.append({"tag": tag, "engine": engine,
                                 "perturbation": plabel, "ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"})

    # Aggregate disagreements over every pair of successful runs.
    agg: Dict[str, set] = {c: set() for c in INVARIANT_CATEGORIES}
    pairwise: List[Dict[str, Any]] = []
    mirror_any = False
    for i in range(len(extracted)):
        for j in range(i + 1, len(extracted)):
            da, db = extracted[i], extracted[j]
            disagree, mirrored = _best_quotiented_disagreements(da["inv"], db["inv"])
            mirror_any = mirror_any or mirrored
            n = _total_disagreements(disagree)
            pairwise.append({
                "a": da["tag"], "b": db["tag"],
                "mirror_quotiented": mirrored,
                "n_disagreements": n,
                "per_category": {c: v for c, v in disagree.items() if v},
            })
            for c, keys in disagree.items():
                for k in keys:
                    agg[c].add(f"{da['tag']} vs {db['tag']}: {k}")

    per_category = {c: sorted(agg[c]) for c in INVARIANT_CATEGORIES}
    score = sum(len(v) for v in per_category.values())

    # device-set summary + REALIZATION GAPS: when two runs realize DIFFERENT device
    # sets the same declared intent produced incomparable layouts -- a realization gap
    # (an engine that drops declared devices cannot satisfy the intent at all). This is
    # surfaced explicitly so it is not buried inside the per-pair invariant disagreement
    # count (a set difference inflates relative_order / adjacency mechanically).
    device_sets = {e["tag"]: e["labels"] for e in extracted}
    realization_gaps: List[str] = []
    for i in range(len(extracted)):
        for j in range(i + 1, len(extracted)):
            la, lb = set(extracted[i]["labels"]), set(extracted[j]["labels"])
            if la != lb:
                only_a = sorted(la - lb)
                only_b = sorted(lb - la)
                realization_gaps.append(
                    f"{extracted[i]['tag']} ({len(la)} dev) vs "
                    f"{extracted[j]['tag']} ({len(lb)} dev): "
                    f"only in A={only_a or '[]'}; only in B={only_b or '[]'}"
                )

    return IncompletenessReport(
        incompleteness_score=score,
        per_category=per_category,
        runs=runs,
        errors=errors,
        pairwise=pairwise,
        mirror_quotiented=mirror_any,
        perturbations_applied=perturbations,
        device_sets=device_sets,
        realization_gaps=realization_gaps,
    )


def _try_reverse_devices(cell_def: Any) -> Any:
    """Return a copy of ``cell_def`` with its device declaration order REVERSED, or
    None if no reorderable device list is exposed.

    Locates the device list generically (probing ``topology/body/block`` holders and
    direct cell attrs) and reverses it in place on a deep copy. Reversal is a
    deterministic, total permutation that maximally stresses any order-dependence.
    """
    import copy

    cd = copy.deepcopy(cell_def)
    for holder_name in (None, "topology", "body", "block"):
        holder = cd if holder_name is None else getattr(cd, holder_name, None)
        if holder is None:
            continue
        for attr in ("devices", "device_defs", "instances"):
            seq = getattr(holder, attr, None)
            if isinstance(seq, list) and len(seq) >= 2:
                try:
                    setattr(holder, attr, list(reversed(seq)))
                    return cd
                except Exception:  # noqa: BLE001 - immutable -> not reorderable here
                    continue
    return None
