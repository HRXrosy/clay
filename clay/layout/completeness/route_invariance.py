"""The ROUTING INVARIANCE HARNESS: extract canonical ROUTING invariants from a
routed cell, and quantify routing-intent incompleteness by disagreement across
faithful routers (the sibling of :mod:`clay.layout.completeness.invariance`,
which does the SAME for PLACEMENT).

WHAT THIS IS (and is NOT)
=========================
This is the RULER, not the gap-closer. It measures how much of a route is
pinned by declared ROUTING intent + placement + PDK rules. It NEVER edits the
router and NEVER adds a primitive.

Completeness definition (the routing analog of the placement one):

  A declared ROUTING intent (``diff_pairs`` / ``shields`` / ``wire_widths`` /
  ``match_length_pairs``) PLUS the placement PLUS the PDK rules is COMPLETE iff it
  pins the route down to DRC-LEGAL DON'T-CARES -- the exact track coordinate
  within a legal range, the exact via x/y within a legal window, the exact jog
  geometry are all don't-cares -- so that ANY faithful router produces the SAME
  route in every invariant that matters. Whatever VARIES under a faithful
  perturbation is incompleteness to close LATER with GENERAL primitives (never by
  per-circuit router tuning).

THE ROUTED-RESULT SHAPE IT CONSUMES
===================================
:func:`clay.layout.route_refine.route_cell_to_intent` returns a
:class:`clay.layout.route_refine.ConvergedRouteCell` whose ``.cell`` is the
generic-:class:`clay.layout.generic_router.RoutedCell`-shaped adapter
(``RouteCellResult.to_routed_cell()``). We extract off that PUBLIC shape:

  * ``cell.tagged`` : ``[(net, (layer, datatype), bbox), ...]`` -- every routed
    shape tagged with its net + GDS (layer, datatype) + absolute bbox. This is
    the primary surface: net -> layers, net -> wire geometry, net -> via cuts.
  * ``cell.shorts`` : a :class:`~clay.layout.generic_router.ShortsCheck`
    (``distinct`` / ``merged_pairs`` / ``net_components`` / ``shape_roots``) --
    the union-find over ``tagged`` (``generic_router.py:1242``). Drives the
    net-completeness / short report.
  * ``cell.net_complete`` : a
    :class:`~clay.layout.generic_router.NetCompletenessResult` (``complete`` /
    ``n_total`` / ``n_complete`` / ``opens`` [:class:`NetOpen`] / ``shorts``,
    ``generic_router.py:1381``) -- the LVS-faithfulness gate over the SCHEMATIC
    net set, with per-net open/short diagnostics. Drives the realization gap.
  * ``cell.labels`` : ``net -> (x, y)`` text-label anchors (used only for the
    pin-access invariant's anchor side).

The DECLARED routing intent (:class:`clay.layout.intent.LayoutIntent`,
``clay/layout/intent.py:297``) supplies ``diff_pairs`` (matched/differential
nets), ``shields`` (``Shield(aggressor, victim, shield_net)``), ``wire_widths``
(net -> nm) and ``match_length_pairs`` -- read off the intent so route_symmetry /
length_class / shield_spacing / width_class are scored against what was DECLARED.

THE 8 CANONICAL ROUTING INVARIANT CATEGORIES (shared taxonomy -- do not invent):
  1. net_layer       : per net the sorted SET of metal (layer, datatype) used.
  2. track_topology  : relative order of parallel net trunks in each channel +
                       which net pairs CROSS (inner/outer order). The exact track
                       COORDINATE is quotiented; the ORDER / crossing is kept.
  3. pin_access      : per (net, device, terminal) the access SIDE the riser
                       leaves the terminal from (L/R/U/D/on) -- a relation, not a
                       coordinate.
  4. via_structure   : per net the layer-transition MULTISET (how many cuts of
                       each cut-layer, between which metal pair). Exact via xy is
                       quotiented; the count + layer-pair is kept.
  5. route_symmetry  : per declared matched/diff net pair, whether the two routes
                       are MIRROR-symmetric about the axis (same layer set,
                       mirrored trunk order, equal via multiset).
  6. length_class    : per @match_length group, the relative spread of routed net
                       lengths (the matched-length invariant), bucketed.
  7. shield_spacing  : per declared shield, whether the shield net actually runs
                       BETWEEN aggressor and victim + the inter-net spacing class.
  8. width_class     : per net the realized wire WIDTH class (the @wire_width
                       invariant) -- the trunk width bucket.

DRC-LEGAL DON'T-CARES (quotiented out before comparison -- documented precisely):
  * absolute translation : every coordinate-bearing invariant is a pure RELATION
    (order / side / count / class membership) or is referenced to a canonical
    frame (centroid-of-trunk-x), so a global translation never disagrees.
  * exact track coordinate : track_topology keeps only the ORDER of parallel
    trunks (and who crosses whom), not the y / x value -- a faithful router may
    place a trunk anywhere in its legal track range.
  * exact via x/y within a legal window : via_structure keeps the layer-transition
    MULTISET (count + metal pair), not the cut coordinate -- a via may sit
    anywhere in its legal landing window.
  * exact jog geometry : a jog is a track-to-lane detour; only the access SIDE
    (pin_access) and the resulting layer set / via count survive, not the jog's
    exact length or bend coordinates.
  * width / length / spacing are bucketed to CLASSES (relative), never compared as
    absolute nm, so grid-snap residue does not spuriously disagree.

DETERMINISM: every returned structure is built from SORTED keys / SORTED tuples,
so two equal routes produce byte-equal dicts (asserted in the test).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The canonical taxonomy, in canonical order. Shared across the round.
ROUTE_INVARIANT_CATEGORIES: Tuple[str, ...] = (
    "net_layer",
    "track_topology",
    "pin_access",
    "via_structure",
    "route_symmetry",
    "length_class",
    "shield_spacing",
    "width_class",
)

# Geometric tolerance (nm / dbu). The gf180mcu manufacturing grid is 5 nm and the
# router snaps lanes to ``LANE_GRID_NM`` (720 nm); routed metals are 400 nm wide.
# EPS absorbs grid-snap residue so an "equal" coordinate / overlap is recognised
# as equal, kept small relative to a routing pitch.
EPS: int = 10

# Width-class bucket (nm): two trunk widths within one bucket are the same width
# class. The PDK metal minimum is 400 nm and ``@wire_width`` declarations step in
# whole microns, so a 200 nm bucket separates a declared-widened net from a
# default-width net while folding grid residue.
_WIDTH_BUCKET_NM: int = 200

# Length-class bucket (relative): two routed net lengths whose relative spread is
# within this fraction are the same length class (the @match_length invariant is a
# RELATIVE balance, never an absolute nm threshold -- project-wide rule).
_LENGTH_REL_BUCKET: float = 0.10


# --------------------------------------------------------------------------- #
# routed-result access (off the public RoutedCell shape)                      #
# --------------------------------------------------------------------------- #


def _tagged(routed: Any) -> List[Tuple[str, Tuple[int, int], Tuple[int, int, int, int]]]:
    """``[(net, (layer, datatype), bbox), ...]`` off ``RoutedCell.tagged``.

    Normalises each bbox to ``(x0, y0, x1, y1)`` with x0<=x1, y0<=y1. Empty when
    the cell carries no tagged geometry (a never-routed cell)."""
    out: List[Tuple[str, Tuple[int, int], Tuple[int, int, int, int]]] = []
    for item in getattr(routed, "tagged", ()) or ():
        try:
            net, key, box = item
        except (ValueError, TypeError):
            continue
        x0, y0, x1, y1 = box
        nb = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        out.append((str(net), (int(key[0]), int(key[1])), nb))
    return out


def _shorts(routed: Any) -> Any:
    return getattr(routed, "shorts", None)


def _net_complete(routed: Any) -> Any:
    return getattr(routed, "net_complete", None)


def _nets(tagged: Sequence) -> List[str]:
    """Sorted distinct net names present in ``tagged`` (excludes sentinels)."""
    return sorted({n for (n, _k, _b) in tagged if not n.startswith("\x00")})


def _via_layers(routed: Any) -> frozenset:
    """The SET of cut-layer (layer, datatype) keys that bridge two metals.

    Read from the routed cell's own tech when present; otherwise the gf180 Via1
    key alone (the :func:`~clay.layout.generic_router.check_shorts` default), so a
    hand-built test cell with no tech still classifies its via cuts. We derive the
    cut set from ``shorts`` indirectly is not possible (it does not store the via
    set), so we probe the well-known gf180 Via1/Via2 keys, matched by being the
    keys that are NEITHER a metal trunk NOR a label -- a robust heuristic for the
    extractor (a cut layer is a small square shared by two metal stacks)."""
    try:
        from clay.layout.m2_crossing_route import VIA1_KEY
        return frozenset({tuple(VIA1_KEY)})
    except Exception:
        return frozenset()


# --------------------------------------------------------------------------- #
# declared-intent access (engine-symmetric)                                   #
# --------------------------------------------------------------------------- #


def _declared_pairs(intent: Any) -> Tuple[Tuple[str, str], ...]:
    """Declared matched/differential net pairs ``(a, b)`` (canonical, sorted).

    Read off ``intent.diff_pairs`` (the genuinely-differential matched pairs). Each
    pair is normalised to ``(min, max)`` by net name so the SET is order-independent
    (a matched pair is unordered)."""
    pairs: List[Tuple[str, str]] = []
    for ab in getattr(intent, "diff_pairs", ()) or ():
        if len(tuple(ab)) != 2:
            continue
        a, b = tuple(ab)
        pairs.append((a, b) if a <= b else (b, a))
    return tuple(sorted(set(pairs)))


def _declared_shields(intent: Any) -> Tuple[Tuple[str, str, str], ...]:
    """Declared shields as sorted ``(aggressor, victim, shield_net)`` triples."""
    out: List[Tuple[str, str, str]] = []
    for sh in getattr(intent, "shields", ()) or ():
        a = str(getattr(sh, "aggressor", ""))
        v = str(getattr(sh, "victim", ""))
        s = str(getattr(sh, "shield_net", "gnd"))
        if a and v:
            out.append((a, v, s))
    return tuple(sorted(set(out)))


def _declared_widths(intent: Any) -> Dict[str, int]:
    """``net -> declared width (nm)`` off ``intent.wire_widths`` (may be empty)."""
    ww = getattr(intent, "wire_widths", {}) or {}
    out: Dict[str, int] = {}
    for k, v in ww.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _declared_length_pairs(intent: Any) -> Tuple[Tuple[str, str], ...]:
    """Declared @match_length pairs ``(a, b)`` (canonical, sorted)."""
    pairs: List[Tuple[str, str]] = []
    for ab in getattr(intent, "match_length_pairs", ()) or ():
        if len(tuple(ab)) != 2:
            continue
        a, b = tuple(ab)
        pairs.append((a, b) if a <= b else (b, a))
    return tuple(sorted(set(pairs)))


# --------------------------------------------------------------------------- #
# geometry helpers                                                            #
# --------------------------------------------------------------------------- #


def _wh(b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return (b[2] - b[0], b[3] - b[1])


def _center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _is_horizontal(b: Tuple[int, int, int, int]) -> bool:
    w, h = _wh(b)
    return w >= h


def _frame_x(tagged: Sequence) -> float:
    """Canonical x-origin: mean of trunk/shape center-x (centroid-of-centroids).

    Subtracting it makes x-referenced invariants translation-invariant on x (the
    axis the global-mirror don't-care lives on)."""
    xs = [_center(b)[0] for (_n, _k, b) in tagged]
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


# --------------------------------------------------------------------------- #
# the 8 extractors                                                            #
# --------------------------------------------------------------------------- #


def _inv_net_layer(tagged: Sequence) -> Dict[str, List[List[int]]]:
    """Per net the SORTED set of metal (layer, datatype) keys used.

    A pure SET of layer keys per net -- which metals the net's route lives on
    (M1 device pads, M2 trunk, M3 riser, the via cuts). Intrinsically translation-
    invariant. Returned as ``{net -> [[layer, datatype], ...]}`` sorted."""
    by_net: Dict[str, set] = {}
    for (n, k, _b) in tagged:
        if n.startswith("\x00"):
            continue
        by_net.setdefault(n, set()).add(k)
    return {n: [list(k) for k in sorted(by_net[n])] for n in sorted(by_net)}


def _trunks(tagged: Sequence) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """Per net the HORIZONTAL trunk shapes (the channel trunks whose ORDER matters).

    A trunk is the widest horizontal metal run of a net (the M2 channel trunk in
    the comparator paradigm). We keep every horizontal shape; the topology extractor
    reduces them to the net's representative track-y."""
    out: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for (n, _k, b) in tagged:
        if n.startswith("\x00"):
            continue
        if _is_horizontal(b):
            out.setdefault(n, []).append(b)
    return out


def _inv_track_topology(tagged: Sequence) -> Dict[str, Any]:
    """Relative ORDER of parallel net trunks in the channel + crossing topology.

    For each net with a horizontal trunk we take its representative trunk as the
    WIDEST horizontal shape (the channel trunk, not a short jog) and its track-y /
    x-span. The invariant is:

      * ``track_order`` : the nets ordered by ascending trunk-y (the parallel-track
        stacking order). The exact y is QUOTIENTED -- only the order survives.
      * ``crossings`` : for each unordered net pair whose x-spans OVERLAP (they
        share a column band, so their trunks are parallel in the same region), the
        sign of the y-order (which is the inner/outer / above-below relation). Two
        nets that do not share a column band are not comparable (no crossing
        relation) and are omitted.

    Both are pure RELATIONS -> translation-invariant. The order is the routing
    decision a faithful router must reproduce; the absolute track is the don't-care.
    """
    trunks = _trunks(tagged)
    rep: Dict[str, Tuple[float, int, int]] = {}  # net -> (track_y, x0, x1)
    for n, boxes in trunks.items():
        # representative = widest horizontal shape (the channel trunk).
        widest = max(boxes, key=lambda b: _wh(b)[0])
        cy = _center(widest)[1]
        rep[n] = (cy, widest[0], widest[2])
    # track_order: nets by ascending trunk-y (ties by net name -> deterministic).
    order = sorted(rep, key=lambda n: (rep[n][0], n))
    # crossings: for nets sharing a column band, the above/below sign.
    crossings: Dict[str, str] = {}
    for a, b in combinations(sorted(rep), 2):
        ya, xa0, xa1 = rep[a]
        yb, xb0, xb1 = rep[b]
        overlap = min(xa1, xb1) - max(xa0, xb0)
        if overlap <= EPS:
            continue  # no shared column band -> not parallel here, no relation
        if ya < yb - EPS:
            crossings[f"{a}|{b}"] = "-"   # a below b
        elif ya > yb + EPS:
            crossings[f"{a}|{b}"] = "+"   # a above b
        else:
            crossings[f"{a}|{b}"] = "0"   # same track within eps
    return {"track_order": order, "crossings": crossings}


def _inv_pin_access(tagged: Sequence, terminal_xy: Dict[Tuple[str, str], Tuple[int, int]],
                    term_net: Dict[Tuple[str, str], str]) -> Dict[str, str]:
    """Per (net, device, terminal): the SIDE the route leaves the terminal from.

    For each declared terminal we find the net's nearest routed shape and classify
    the direction from the terminal point to that shape's center as one of
    L / R / U / D / on (on = the shape sits over the terminal within eps). This is
    the access-side relation -- which side / access point the router used -- a pure
    relation (sign of a coordinate difference), translation-invariant. Keyed by
    ``"net@device.term"`` sorted. Terminals with no same-net routed shape are
    OMITTED (an unconnected terminal is a net-completeness gap, surfaced separately).
    """
    # bucket routed shapes by net for nearest lookup.
    by_net: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for (n, _k, b) in tagged:
        if n.startswith("\x00"):
            continue
        by_net.setdefault(n, []).append(b)

    out: Dict[str, str] = {}
    for (label, tk), (tx, ty) in terminal_xy.items():
        net = term_net.get((label, tk))
        if not net or net not in by_net:
            continue
        # nearest same-net shape (by center distance), excluding the terminal pad
        # itself (a shape exactly over the terminal counts as access "on").
        best = None
        best_d = None
        for b in by_net[net]:
            cx, cy = _center(b)
            d = (cx - tx) ** 2 + (cy - ty) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best = b
        if best is None:
            continue
        # if a shape covers the terminal, the access is "on" (a direct landing);
        # else classify by the dominant axis to the nearest shape center.
        covers = any(b[0] - EPS <= tx <= b[2] + EPS and b[1] - EPS <= ty <= b[3] + EPS
                     for b in by_net[net])
        if covers:
            side = "on"
        else:
            cx, cy = _center(best)
            dx, dy = cx - tx, cy - ty
            if abs(dx) >= abs(dy):
                side = "R" if dx > 0 else "L"
            else:
                side = "U" if dy > 0 else "D"
        out[f"{net}@{label}.{tk}"] = side
    return {k: out[k] for k in sorted(out)}


def _inv_via_structure(tagged: Sequence, via_keys: frozenset) -> Dict[str, List[List[int]]]:
    """Per net the layer-transition MULTISET: how many cuts of each cut-layer.

    A cut shape (a shape whose (layer, datatype) is in ``via_keys``) is a metal<->
    metal transition. We count, per net, how many cuts land on each cut-layer key.
    The exact via xy is QUOTIENTED -- only the COUNT per cut-layer survives (the
    layer-transition multiset). Returned as ``{net -> [[layer, datatype, count], ...]}``
    sorted. A net with no vias maps to ``[]`` (still listed if it has any route)."""
    by_net: Dict[str, Dict[Tuple[int, int], int]] = {}
    present: set = set()
    for (n, k, _b) in tagged:
        if n.startswith("\x00"):
            continue
        present.add(n)
        if k in via_keys:
            by_net.setdefault(n, {})[k] = by_net.setdefault(n, {}).get(k, 0) + 1
    out: Dict[str, List[List[int]]] = {}
    for n in sorted(present):
        counts = by_net.get(n, {})
        out[n] = [[k[0], k[1], counts[k]] for k in sorted(counts)]
    return out


def _net_layer_set(tagged: Sequence, net: str) -> frozenset:
    return frozenset(k for (n, k, _b) in tagged if n == net)


def _net_via_multiset(tagged: Sequence, net: str, via_keys: frozenset) -> Tuple[Tuple[Tuple[int, int], int], ...]:
    counts: Dict[Tuple[int, int], int] = {}
    for (n, k, _b) in tagged:
        if n == net and k in via_keys:
            counts[k] = counts.get(k, 0) + 1
    return tuple(sorted(counts.items()))


def _inv_route_symmetry(tagged: Sequence, pairs: Tuple[Tuple[str, str], ...],
                        via_keys: frozenset, frame_x: float) -> Dict[str, Dict[str, Any]]:
    """Per declared matched/diff pair: whether the two routes are MIRROR-symmetric.

    For a pair (a, b) we test the three observable symmetry signals (each a
    don't-care-quotiented relation):

      * ``same_layers``  : the two nets use the same SET of metal layers (a mirror
        keeps the layer assignment).
      * ``same_via_count``: the two nets have the same via layer-transition multiset
        (a mirror keeps the via structure; the xy is quotiented).
      * ``mirrored_x``   : the two nets' trunk center-x are reflections about the
        canonical frame (``|xa + xb - 2*frame_x| <= eps`` -- they sit symmetric
        about the axis). A faithful symmetric route mirrors the trunks.

    ``mirrors`` is True iff all three hold. Keyed by the sorted pair ``"a|b"``.
    Translation-invariant (layer set / via count are intrinsic; mirrored_x uses the
    centroid frame)."""
    # representative trunk center-x per net (widest horizontal shape).
    trunks = _trunks(tagged)
    cx_of: Dict[str, Optional[float]] = {}
    for n, boxes in trunks.items():
        widest = max(boxes, key=lambda b: _wh(b)[0])
        cx_of[n] = _center(widest)[0]
    out: Dict[str, Dict[str, Any]] = {}
    for a, b in pairs:
        la, lb = _net_layer_set(tagged, a), _net_layer_set(tagged, b)
        va = _net_via_multiset(tagged, a, via_keys)
        vb = _net_via_multiset(tagged, b, via_keys)
        same_layers = la == lb and bool(la)
        same_via = va == vb
        xa, xb = cx_of.get(a), cx_of.get(b)
        if xa is None or xb is None:
            mirrored_x = False
        else:
            mirrored_x = abs((xa - frame_x) + (xb - frame_x)) <= EPS
        out[f"{a}|{b}"] = {
            "same_layers": bool(same_layers),
            "same_via_count": bool(same_via),
            "mirrored_x": bool(mirrored_x),
            "mirrors": bool(same_layers and same_via and mirrored_x),
        }
    return out


def _net_routed_length(tagged: Sequence, net: str, via_keys: frozenset) -> int:
    """Total routed METAL run length (nm) of a net: sum of the LONG edge of each
    non-cut shape. A wire's contribution is its longer dimension (its run length);
    via cuts (square, in ``via_keys``) are excluded so the length is the metal run,
    not the contact count."""
    total = 0
    for (n, k, b) in tagged:
        if n != net or k in via_keys:
            continue
        w, h = _wh(b)
        total += max(w, h)
    return total


def _inv_length_class(tagged: Sequence, length_pairs: Tuple[Tuple[str, str], ...],
                      via_keys: frozenset) -> Dict[str, Dict[str, Any]]:
    """Per @match_length pair: the RELATIVE spread of the two routed net lengths.

    For each declared matched-length pair (a, b) we measure each net's routed metal
    run length and report the RELATIVE imbalance ``|La - Lb| / max(La, Lb)`` bucketed
    to :data:`_LENGTH_REL_BUCKET` (so grid residue does not disagree), plus a
    ``balanced`` flag (imbalance within one bucket). RELATIVE only (project rule) ->
    scale-invariant. Keyed by the sorted pair ``"a|b"``. Empty when no @match_length
    is declared (the invariant is vacuous -- not every cell matches lengths)."""
    out: Dict[str, Dict[str, Any]] = {}
    for a, b in length_pairs:
        la = _net_routed_length(tagged, a, via_keys)
        lb = _net_routed_length(tagged, b, via_keys)
        denom = max(la, lb)
        imbalance = 0.0 if denom == 0 else abs(la - lb) / denom
        bucket = int(imbalance / _LENGTH_REL_BUCKET)
        out[f"{a}|{b}"] = {
            "imbalance_bucket": bucket,
            "balanced": bool(bucket == 0),
        }
    return out


def _inv_shield_spacing(tagged: Sequence, shields: Tuple[Tuple[str, str, str], ...]) -> Dict[str, Dict[str, Any]]:
    """Per declared shield: whether the shield net runs BETWEEN aggressor & victim.

    For a declared ``Shield(aggressor, victim, shield_net)`` we test whether the
    shield net has a trunk whose track-y sits BETWEEN the aggressor's and victim's
    trunk track-y (the shield is interposed). This is a pure ORDER relation
    (the shield is between the two), translation-invariant; the exact spacing nm is
    a DRC don't-care so only the betweenness + presence survive. Keyed by
    ``"aggressor~victim~shield"`` sorted. ``present`` = the shield net was routed at
    all; ``between`` = it is interposed between the pair."""
    trunks = _trunks(tagged)

    def _track_y(net: str) -> Optional[float]:
        boxes = trunks.get(net)
        if not boxes:
            return None
        return _center(max(boxes, key=lambda b: _wh(b)[0]))[1]

    out: Dict[str, Dict[str, Any]] = {}
    for (agg, vic, sh) in shields:
        ya, yv, ys = _track_y(agg), _track_y(vic), _track_y(sh)
        present = ys is not None
        if present and ya is not None and yv is not None:
            lo, hi = (ya, yv) if ya <= yv else (yv, ya)
            between = lo - EPS <= ys <= hi + EPS
        else:
            between = False
        out[f"{agg}~{vic}~{sh}"] = {
            "present": bool(present),
            "between": bool(between),
        }
    return {k: out[k] for k in sorted(out)}


def _inv_width_class(tagged: Sequence, declared_widths: Dict[str, int]) -> Dict[str, int]:
    """Per net the realized wire WIDTH class (the @wire_width invariant).

    For each net we take the representative TRUNK (the widest horizontal shape) and
    bucket its width (the SHORT dimension of the horizontal trunk -- the wire width)
    to :data:`_WIDTH_BUCKET_NM`. The width is the realized geometry; the bucket folds
    grid residue. Keyed by net -> bucket index. Intrinsically translation-invariant.
    Nets with no horizontal trunk are omitted. ``declared_widths`` is accepted so a
    future variant can compare realized vs declared, but the invariant itself is the
    realized class (a faithful router must realize the SAME width class for a net)."""
    trunks = _trunks(tagged)
    out: Dict[str, int] = {}
    for n, boxes in trunks.items():
        widest = max(boxes, key=lambda b: _wh(b)[0])
        w, h = _wh(widest)
        width = min(w, h)  # the wire width = the short edge of a horizontal trunk
        out[n] = int(width // _WIDTH_BUCKET_NM)
    return {k: out[k] for k in sorted(out)}


# --------------------------------------------------------------------------- #
# top-level extractor                                                         #
# --------------------------------------------------------------------------- #


def extract_route_invariants(
    routed: Any,
    *,
    intent: Any = None,
    terminal_xy: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None,
    term_net: Optional[Dict[Tuple[str, str], str]] = None,
    via_keys: Optional[frozenset] = None,
    eps: int = EPS,
) -> Dict[str, Any]:
    """Extract the 8 canonical ROUTING invariants from a routed-cell result.

    ``routed`` is a :class:`clay.layout.generic_router.RoutedCell` (or any object
    exposing ``.tagged`` of ``(net, (layer, datatype), bbox)`` triples, plus
    optionally ``.shorts`` / ``.net_complete`` / ``.labels``).

    ``intent`` (a :class:`clay.layout.intent.LayoutIntent`) supplies the DECLARED
    ``diff_pairs`` / ``shields`` / ``wire_widths`` / ``match_length_pairs`` the
    route_symmetry / shield_spacing / length_class / width_class invariants are
    scored against; ``None`` yields those categories EMPTY (nothing declared to
    measure -- not a disagreement).

    ``terminal_xy`` (``(device, term) -> (x, y)``) + ``term_net`` (``(device, term)
    -> net``) drive the pin_access invariant; ``None`` yields an empty pin_access
    (no terminal map available, e.g. a hand-built test cell).

    Returns a CANONICAL (sorted-key, deterministic) dict keyed by the 8 category
    names in :data:`ROUTE_INVARIANT_CATEGORIES`. Two equal routes produce byte-equal
    output. Every coordinate-bearing invariant is translation-invariant; the
    global-mirror don't-care is quotiented later in
    :func:`route_incompleteness_report`."""
    tagged = _tagged(routed)
    vk = via_keys if via_keys is not None else _via_layers(routed)
    pairs = _declared_pairs(intent) if intent is not None else ()
    shields = _declared_shields(intent) if intent is not None else ()
    widths = _declared_widths(intent) if intent is not None else {}
    length_pairs = _declared_length_pairs(intent) if intent is not None else ()
    fx = _frame_x(tagged)
    txy = terminal_xy or {}
    tnet = term_net or {}
    return {
        "net_layer": _inv_net_layer(tagged),
        "track_topology": _inv_track_topology(tagged),
        "pin_access": _inv_pin_access(tagged, txy, tnet),
        "via_structure": _inv_via_structure(tagged, vk),
        "route_symmetry": _inv_route_symmetry(tagged, pairs, vk, fx),
        "length_class": _inv_length_class(tagged, length_pairs, vk),
        "shield_spacing": _inv_shield_spacing(tagged, shields),
        "width_class": _inv_width_class(tagged, widths),
    }


# --------------------------------------------------------------------------- #
# run adapter onto the EXISTING route entry point                             #
# --------------------------------------------------------------------------- #


def _reorder_netlist(netlist: Any, perm: Sequence[int]) -> Any:
    """Return a COPY of ``netlist`` with its ``devices`` list reordered by ``perm``.

    Net ordering is the perturbation that stresses router order-dependence: both
    :func:`~clay.layout.generic_router.net_terminals_from_netlist` and the per-net
    trunk routing iterate the netlist's device order (first-seen net order), so a
    complete intent must be INVARIANT to it. Returns ``None`` when the netlist does
    not expose a reorderable ``devices`` list (reported honestly)."""
    import copy

    nl = copy.deepcopy(netlist)
    seq = getattr(nl, "devices", None)
    if isinstance(seq, list) and len(seq) == len(perm):
        try:
            nl.devices = [seq[i] for i in perm]
            return nl
        except Exception:  # noqa: BLE001 - immutable -> not reorderable
            return None
    return None


def run_route(
    cell_def: Any,
    engine: str,
    *,
    intent: Any = None,
    criteria: Any = None,
    source_cir: Any = None,
    tech: Any = None,
    placement: Any = None,
    net_order: Optional[Sequence[int]] = None,
    netlist: Any = None,
    max_iters: int = 1,
) -> Any:
    """FIX a placement, then ROUTE the declared intent; return the RoutedCell.

    Thin adapter mirroring :func:`clay.layout.completeness.invariance.run_placement`:

      1. PLACEMENT: when ``placement`` is passed it is used directly (the harness
         FIXES one placement and routes it under perturbations -- the same
         placement, so a routing disagreement is a ROUTING decision, not a placement
         one). Otherwise it is solved once via
         :func:`clay.layout.place_refine.place_to_intent` the SAME way
         ``run_placement`` does, with ``engine in {"milp", "seqpair"}``.
      2. DEVICE RECORDS: built from the ``source_cir`` MOS + passive records (the
         SAME ``name -> {kind, W/L, pins}`` map the production flow + the placer
         consume), so a cell whose devices live only in the netlist still routes.
      3. ROUTE: through the GENERAL negotiated-congestion PathFinder engine via
         :func:`clay.layout.route_refine.route_cell_to_intent` -- the SAME single
         general router the production flow (``clay/flow/step.py::_run_generic``) and
         the comparator / OTA real-deck signoffs drive. Returns the
         generic-``RoutedCell``-shaped ``ConvergedRouteCell.cell`` (the
         ``RouteCellResult.to_routed_cell()`` adapter) the invariant extractor reads.

    ``net_order`` (a permutation of ``range(len(netlist.devices))``) reorders the
    derived device records BEFORE routing -- the harness's input-order perturbation.

    PERTURBATION HONESTY: ``route_cell`` (PathFinderEngine) takes NO tie-break seed
    and is DETERMINISTIC + net-complete-by-construction given (placement, cell_def,
    intent, tech). Its net binding comes from ``cell_def.devices`` when materialised
    (else the derived ``device_records``), so for a materialised cell the net-order
    permutation of the derived records does NOT perturb the route -- the route is
    pinned by the placement + intent + PDK, not the netlist iteration order (this is
    a STRONGER invariance than the legacy hand-rolled router, which iterated
    first-seen net order). The faithful perturbations that remain are (i) the
    net-order permutation of the device records and (ii) the upstream PLACEMENT
    ENGINE (milp vs seqpair). ``max_iters`` is accepted for signature parity;
    route_cell needs no spacing-knob convergence loop (single deterministic pass)."""
    from clay.layout.route_refine import route_cell_to_intent

    if placement is None:
        from clay.layout.place_refine import place_to_intent
        cp = place_to_intent(
            cell_def, intent, criteria,
            source_cir=source_cir, tech=tech, engine=engine,
        )
        placement = cp.placement

    device_records = _build_cir_records(source_cir)
    if net_order is not None and device_records:
        keys = list(device_records.keys())
        if len(keys) == len(net_order):
            try:
                device_records = {keys[i]: device_records[keys[i]] for i in net_order}
            except Exception:  # noqa: BLE001 - a bad perm degrades to declared order
                pass

    cr = route_cell_to_intent(
        placement, cell_def, intent, tech,
        device_records=device_records or None, max_iters=max_iters,
    )
    return cr.cell


def _build_cir_records(source_cir: Any) -> Dict[str, Dict[str, Any]]:
    """The parsed ``name -> {kind, W/L, pins}`` device records from a ``.cir`` path.

    Merges the MOS + passive (R/C) records -- the SAME ``device_records`` map the
    production flow (``clay/flow/step.py::_run_generic``) and the placer consume, so
    ``route_cell`` binds a cell whose devices live only in the netlist. Empty when no
    ``source_cir`` is given (a materialised ``cell_def`` then carries the binding)."""
    from clay.layout.generic_placement import (
        parse_cir_mos_records,
        parse_cir_passive_records,
    )

    records: Dict[str, Dict[str, Any]] = {}
    if source_cir is not None:
        records.update(parse_cir_mos_records(str(source_cir)))
        records.update(parse_cir_passive_records(str(source_cir)))
    return records


def _build_cir_netlist(source_cir: Any) -> Any:
    """Build the router's ``_CirNetlist`` from a ``.cir`` path (the flow's adapter).

    Mirrors ``clay/flow/step.py:1576`` -- merges the MOS + passive (R/C) records so
    the router routes resistor p/n nets like any MOS net. Returns a duck-typed
    netlist exposing ``.devices`` (each with ``instance_name`` + ``pins``)."""
    records = _build_cir_records(source_cir)

    class _Dev:
        __slots__ = ("instance_name", "pins")

        def __init__(self, instance_name: str, pins: Dict[str, str]) -> None:
            self.instance_name = instance_name
            self.pins = pins

    class _Netlist:
        def __init__(self, recs: Dict[str, Dict[str, Any]]) -> None:
            self.devices = [
                _Dev(lbl, dict(rec.get("pins", {}))) for lbl, rec in recs.items()
            ]

    return _Netlist(records)


def _terminal_maps(placement: Any, netlist: Any) -> Tuple[
    Dict[Tuple[str, str], Tuple[int, int]], Dict[Tuple[str, str], str]
]:
    """Build ``(terminal_xy, term_net)`` for the pin_access invariant.

    ``terminal_xy`` : ``(device, term) -> (x, y)`` off the placement's per-device
    terminals. ``term_net`` : ``(device, term) -> net`` off the netlist's net->
    terminals map (the same map the router builds). Both empty when the inputs do
    not expose the needed shape (a hand-built test cell)."""
    from clay.layout.routed_cell_types import net_terminals_from_netlist

    terminal_xy: Dict[Tuple[str, str], Tuple[int, int]] = {}
    devices = getattr(placement, "devices", {}) or {}
    for label, pd in devices.items():
        terms = getattr(pd, "terminals", {}) or {}
        for tk, xy in terms.items():
            try:
                terminal_xy[(label, tk)] = (int(xy[0]), int(xy[1]))
            except (TypeError, ValueError, IndexError):
                continue

    term_net: Dict[Tuple[str, str], str] = {}
    try:
        net_terms = net_terminals_from_netlist(netlist, placement)
        for net, terms in net_terms.items():
            for (label, tk) in terms:
                term_net[(label, tk)] = net
    except Exception:  # noqa: BLE001 - duck-typed test inputs may not support it
        pass
    return terminal_xy, term_net


# --------------------------------------------------------------------------- #
# global-mirror don't-care + disagreements                                    #
# --------------------------------------------------------------------------- #


def _global_mirror_route_invariants(inv: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the GLOBAL-MIRROR don't-care (vertical-axis flip) to a route invariant
    dict.

    A global mirror about the cell's vertical axis is a legitimate routing don't-
    care: it preserves the per-net LAYER set, the via MULTISET, the track stacking
    ORDER (mirroring is horizontal, the tracks are horizontal so their y-order is
    unchanged), the length classes, the shield betweenness and the width classes.
    It FLIPS only the x-referenced relations:

      * track_topology.crossings : a horizontal mirror does NOT change above/below
        (y-order), so crossings are UNCHANGED. (A vertical mirror flips x, not y.)
      * pin_access : L<->R swap (the access side flips horizontally); U/D/on
        unchanged.
      * route_symmetry.mirrored_x : unchanged (a symmetric pair stays symmetric
        about the mirrored axis -- the centroid frame is mirror-fixed).

    So under a global mirror only pin_access L/R flips. Returns the mirrored view;
    :func:`_route_disagreements` tries BOTH views of run B and keeps whichever
    agrees more (quotienting the don't-care)."""
    out: Dict[str, Any] = {}
    out["net_layer"] = inv["net_layer"]
    out["track_topology"] = inv["track_topology"]
    out["via_structure"] = inv["via_structure"]
    out["route_symmetry"] = inv["route_symmetry"]
    out["length_class"] = inv["length_class"]
    out["shield_spacing"] = inv["shield_spacing"]
    out["width_class"] = inv["width_class"]
    # pin_access: flip L<->R, keep U/D/on.
    pa = {}
    for k, v in inv["pin_access"].items():
        pa[k] = {"L": "R", "R": "L"}.get(v, v)
    out["pin_access"] = pa
    return out


def _disagree_dict(a: Dict, b: Dict) -> List[str]:
    """Disagreeing keys between two flat dicts (missing on one side counts)."""
    out: List[str] = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append(k)
    return out


def _route_category_disagreements(inv_a: Dict[str, Any], inv_b: Dict[str, Any]) -> Dict[str, List[str]]:
    """Per-category disagreeing decisions between two FULL route-invariant dicts.

    Each category names the EXACT decision that disagreed (a net for net_layer /
    via_structure / width_class, a pair for route_symmetry / length_class, a
    terminal for pin_access, a shield for shield_spacing, ``track_order`` /
    ``crossings.<pair>`` for track_topology)."""
    d: Dict[str, List[str]] = {c: [] for c in ROUTE_INVARIANT_CATEGORIES}

    # flat dict-of-value categories.
    d["net_layer"] = _disagree_dict(inv_a["net_layer"], inv_b["net_layer"])
    d["via_structure"] = _disagree_dict(inv_a["via_structure"], inv_b["via_structure"])
    d["width_class"] = _disagree_dict(inv_a["width_class"], inv_b["width_class"])
    d["pin_access"] = _disagree_dict(inv_a["pin_access"], inv_b["pin_access"])

    # dict-of-dict categories (each value is a small dict -> compare whole value).
    d["route_symmetry"] = _disagree_dict(inv_a["route_symmetry"], inv_b["route_symmetry"])
    d["length_class"] = _disagree_dict(inv_a["length_class"], inv_b["length_class"])
    d["shield_spacing"] = _disagree_dict(inv_a["shield_spacing"], inv_b["shield_spacing"])

    # track_topology: order list + crossings dict.
    tt: List[str] = []
    a, b = inv_a["track_topology"], inv_b["track_topology"]
    if a.get("track_order") != b.get("track_order"):
        tt.append("track_order")
    for k in _disagree_dict(a.get("crossings", {}), b.get("crossings", {})):
        tt.append(f"crossings.{k}")
    d["track_topology"] = tt
    return d


def _total(disagree: Dict[str, List[str]]) -> int:
    return sum(len(v) for v in disagree.values())


def _best_quotiented_route_disagreements(
    inv_a: Dict[str, Any], inv_b: Dict[str, Any]
) -> Tuple[Dict[str, List[str]], bool]:
    """Disagreements between two route invariants, quotienting the GLOBAL MIRROR.

    Compute disagreements against run B AS-IS and against the global-mirrored view;
    keep whichever total is smaller (a pure global-mirror difference -- a legitimate
    don't-care -- scores 0). Returns ``(disagreements, mirror_applied)``."""
    raw = _route_category_disagreements(inv_a, inv_b)
    mirrored = _route_category_disagreements(inv_a, _global_mirror_route_invariants(inv_b))
    if _total(mirrored) < _total(raw):
        return mirrored, True
    return raw, False


# --------------------------------------------------------------------------- #
# net-completeness (the routing REALIZATION gap)                              #
# --------------------------------------------------------------------------- #


@dataclass
class NetCompletenessGap:
    """The routing realization gap for one routed run (off ``RoutedCell.net_complete``).

    The routing analog of the placement MILP realization gap: a router that leaves
    nets OPEN (split / never-routed) or SHORTED cannot satisfy intent at all, no
    matter how its invariants compare. Surfaced explicitly so it is not buried.

      * ``n_total`` / ``n_complete`` : schematic nets total vs fully-connected.
      * ``open_nets`` / ``short_nets`` : named failing nets.
      * ``complete`` : every schematic net is exactly one component, no short.
      * ``summary`` : the one-line honest summary from the gate.
    """

    n_total: int
    n_complete: int
    open_nets: Tuple[str, ...]
    short_nets: Tuple[str, ...]
    complete: bool
    summary: str = ""


def _net_completeness_gap(routed: Any) -> Optional[NetCompletenessGap]:
    """Read the :class:`NetCompletenessResult` off a routed cell into a gap record.

    Returns ``None`` when the cell carries no ``net_complete`` (a hand-built cell)."""
    nc = _net_complete(routed)
    if nc is None:
        return None
    return NetCompletenessGap(
        n_total=int(getattr(nc, "n_total", 0)),
        n_complete=int(getattr(nc, "n_complete", 0)),
        open_nets=tuple(getattr(nc, "open_nets", ()) or ()),
        short_nets=tuple(getattr(nc, "short_nets", ()) or ()),
        complete=bool(getattr(nc, "complete", False)),
        summary=str(nc.summary()) if hasattr(nc, "summary") else "",
    )


# --------------------------------------------------------------------------- #
# the route incompleteness report                                            #
# --------------------------------------------------------------------------- #


@dataclass
class RouteIncompletenessReport:
    """The router-invariance report for one design.

    ``route_incompleteness_score`` = total count of disagreeing routing invariants
    across the runs (0 = complete up to the global don't-cares). ``per_category``
    maps each of the 8 categories to the list of disagreeing decisions. ``runs``
    records what actually ran (engine + perturbation label + ok). ``errors`` records
    any (engine, perturbation) that FAILED to route. ``mirror_quotiented`` flags a
    detected+removed global-mirror don't-care. ``perturbations_applied`` documents
    EXACTLY what was done. ``net_completeness`` is the per-run realization gap (the
    routing analog of the placement MILP realization gap) -- connected vs declared
    nets, named open/shorted nets. ``realization_gaps`` lists runs whose route was
    NOT net-complete (an open/shorted route cannot satisfy intent)."""

    route_incompleteness_score: int
    per_category: Dict[str, List[str]]
    runs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    pairwise: List[Dict[str, Any]] = field(default_factory=list)
    mirror_quotiented: bool = False
    perturbations_applied: List[str] = field(default_factory=list)
    net_completeness: Dict[str, NetCompletenessGap] = field(default_factory=dict)
    realization_gaps: List[str] = field(default_factory=list)


def _net_order_perturbations(netlist: Any) -> List[Tuple[Optional[Sequence[int]], str]]:
    """The net-order perturbation list: identity + one reversal of the device order.

    Reversal maximally stresses any first-seen-order dependence in
    ``net_terminals_from_netlist`` and the per-net trunk routing. Returns
    ``[(None, "decl_order"), (reversed_perm, "reversed_net_order")]``; the reversal is
    omitted when the netlist exposes fewer than 2 reorderable devices."""
    devs = getattr(netlist, "devices", None)
    variants: List[Tuple[Optional[Sequence[int]], str]] = [(None, "decl_order")]
    if isinstance(devs, list) and len(devs) >= 2:
        variants.append((list(reversed(range(len(devs)))), "reversed_net_order"))
    return variants


def route_incompleteness_report(
    cell_def: Any,
    *,
    engines: Sequence[str] = ("milp", "seqpair"),
    intent: Any = None,
    criteria: Any = None,
    source_cir: Any = None,
    tech: Any = None,
    placement: Any = None,
    netlist: Any = None,
    permute_net_order: bool = True,
    eps: int = EPS,
) -> RouteIncompletenessReport:
    """Run the SAME declared ROUTING intent through faithful perturbations and
    quantify the disagreement (the routing incompleteness).

    PERTURBATIONS ACTUALLY APPLIED (documented honestly in the report):
      * PLACEMENT ENGINE: each engine in ``engines`` (default milp + seqpair)
        produces a faithful placement of the SAME intent; the SAME routing intent is
        then routed onto each. (When ``placement`` is passed it is FIXED and reused
        for every run, so the only perturbation is net-order -- the cleanest test of
        ROUTING completeness in isolation.)
      * NET-ORDER PERMUTATION: one deterministic REVERSAL of the netlist device
        order, re-routed. Both the net->terminals build and the per-net trunk router
        consume first-seen order, so a complete intent must be invariant to it.

      The ROUTER ITSELF takes NO tie-break seed and is DETERMINISTIC given
      (placement, netlist, intent, tech); there is no alternate randomized router
      strategy to perturb. This is recorded in ``perturbations_applied`` exactly as
      the placement harness recorded its no-seed honesty.

    The score is the total disagreeing-invariant count across all successful runs,
    each pair compared after quotienting absolute translation + the global mirror,
    BUT only when the two runs realized the SAME schematic-net coverage (a run whose
    route is NOT net-complete is a REALIZATION gap surfaced separately, not folded
    into the invariant count where a missing net would mechanically inflate it)."""
    runs: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    perturbations: List[str] = []

    fixed_placement = placement is not None
    if fixed_placement:
        perturbations.append("placement=FIXED (one placement reused; routing isolated)")
        run_engines: Sequence[str] = ("fixed",)
    else:
        perturbations.append("placement_engines=" + "+".join(engines))
        run_engines = engines

    # Build the netlist once so the perturbation list (and its reversal) is stable
    # across engines.
    base_netlist = netlist if netlist is not None else _build_cir_netlist(source_cir)
    if permute_net_order:
        net_variants = _net_order_perturbations(base_netlist)
        if len(net_variants) > 1:
            perturbations.append("net_order_permutation=reversed_device_order")
        else:
            perturbations.append(
                "net_order_permutation=SKIPPED (<2 reorderable devices)"
            )
    else:
        net_variants = [(None, "decl_order")]
    perturbations.append(
        "router_seed=NONE (route_cell/PathFinder is deterministic + net-complete by "
        "construction; no alternate randomized router strategy -- net-order + "
        "placement-engine are the faithful perturbations)"
    )

    # Run every (engine, net-order) combination.
    extracted: List[Dict[str, Any]] = []
    net_gaps: Dict[str, NetCompletenessGap] = {}
    for engine in run_engines:
        # resolve the placement for this engine (fixed, or solved once).
        try:
            if fixed_placement:
                eng_placement = placement
            else:
                from clay.layout.place_refine import place_to_intent
                cp = place_to_intent(
                    cell_def, intent, criteria,
                    source_cir=source_cir, tech=tech, engine=engine,
                )
                eng_placement = cp.placement
        except Exception as exc:  # noqa: BLE001 - a failed placement IS a finding
            errors.append({"tag": f"{engine}/place", "error": f"{type(exc).__name__}: {exc}"})
            runs.append({"tag": f"{engine}/place", "engine": engine, "ok": False,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue

        for (perm, plabel) in net_variants:
            tag = f"{engine}/{plabel}"
            try:
                routed = run_route(
                    cell_def, engine,
                    intent=intent, criteria=criteria,
                    source_cir=source_cir, tech=tech,
                    placement=eng_placement, netlist=base_netlist,
                    net_order=perm,
                )
                txy, tnet = _terminal_maps(eng_placement, base_netlist)
                inv = extract_route_invariants(
                    routed, intent=intent, terminal_xy=txy, term_net=tnet, eps=eps,
                )
                gap = _net_completeness_gap(routed)
                if gap is not None:
                    net_gaps[tag] = gap
                extracted.append({
                    "tag": tag, "engine": engine, "perturbation": plabel,
                    "inv": inv,
                    "coverage": tuple(sorted(_nets(_tagged(routed)))),
                    "complete": (gap.complete if gap is not None else True),
                })
                runs.append({
                    "tag": tag, "engine": engine, "perturbation": plabel, "ok": True,
                    "net_complete": (gap.summary if gap is not None else "n/a"),
                })
            except Exception as exc:  # noqa: BLE001 - a failed route IS a finding
                errors.append({"tag": tag, "error": f"{type(exc).__name__}: {exc}"})
                runs.append({"tag": tag, "engine": engine, "perturbation": plabel,
                             "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    # Aggregate disagreements over every pair of successful runs that realized the
    # SAME net coverage (a coverage mismatch is a realization gap, not an invariant
    # disagreement -- a missing net would mechanically inflate net_layer / topology).
    agg: Dict[str, set] = {c: set() for c in ROUTE_INVARIANT_CATEGORIES}
    pairwise: List[Dict[str, Any]] = []
    mirror_any = False
    realization_gaps: List[str] = []
    for i in range(len(extracted)):
        for j in range(i + 1, len(extracted)):
            da, db = extracted[i], extracted[j]
            if da["coverage"] != db["coverage"]:
                only_a = sorted(set(da["coverage"]) - set(db["coverage"]))
                only_b = sorted(set(db["coverage"]) - set(da["coverage"]))
                realization_gaps.append(
                    f"{da['tag']} ({len(da['coverage'])} nets) vs "
                    f"{db['tag']} ({len(db['coverage'])} nets): "
                    f"only in A={only_a or '[]'}; only in B={only_b or '[]'}"
                )
                continue
            disagree, mirrored = _best_quotiented_route_disagreements(da["inv"], db["inv"])
            mirror_any = mirror_any or mirrored
            n = _total(disagree)
            pairwise.append({
                "a": da["tag"], "b": db["tag"],
                "mirror_quotiented": mirrored,
                "n_disagreements": n,
                "per_category": {c: v for c, v in disagree.items() if v},
            })
            for c, keys in disagree.items():
                for k in keys:
                    agg[c].add(f"{da['tag']} vs {db['tag']}: {k}")

    # A run that is not net-complete is itself a realization gap (an open/shorted
    # route cannot satisfy intent), surfaced explicitly.
    for e in extracted:
        if not e["complete"]:
            gap = net_gaps.get(e["tag"])
            realization_gaps.append(
                f"{e['tag']}: NOT net-complete -- "
                f"{gap.summary if gap is not None else 'open/short'}"
            )

    per_category = {c: sorted(agg[c]) for c in ROUTE_INVARIANT_CATEGORIES}
    score = sum(len(v) for v in per_category.values())

    return RouteIncompletenessReport(
        route_incompleteness_score=score,
        per_category=per_category,
        runs=runs,
        errors=errors,
        pairwise=pairwise,
        mirror_quotiented=mirror_any,
        perturbations_applied=perturbations,
        net_completeness=net_gaps,
        realization_gaps=realization_gaps,
    )


__all__ = [
    "ROUTE_INVARIANT_CATEGORIES",
    "extract_route_invariants",
    "run_route",
    "route_incompleteness_report",
    "RouteIncompletenessReport",
    "NetCompletenessGap",
]
