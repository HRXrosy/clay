"""GEOMETRIC INTENT EXTRACTOR -- the dual of the placer.

WHAT THIS IS
============
The placer takes DECLARED intent (bands / orders / abuts / aligns / equal_size /
symmetry / spacing) and produces device bboxes. This module runs that map
BACKWARDS: given each named device's REFERENCE bbox (the positions a reference
design actually realised), it MECHANICALLY derives the geometric placement intent
that those positions imply -- the same intent the placer would have consumed to
reproduce them. That closes the round-trip ``reference -> extract -> generic place
-> fidelity``.

CONSISTENCY POINT (one taxonomy, two directions)
================================================
The intent is derived through the EXACT SAME 8-category invariant taxonomy the
verification oracles use
(:func:`clay.layout.completeness.invariance.extract_invariants` /
:func:`clay.layout.completeness.gds_invariants.extract_invariants_from_gds`). We do
NOT re-derive geometry with a second, parallel rule set -- we call the canonical
extractor on the reference bboxes and TRANSLATE its canonical 8-category dict into
the :class:`ExtractedIntent` fields. So the extractor and the oracle agree by
construction: anything the oracle would later MEASURE on a placement is exactly the
thing this extractor DECLARED from the reference. There is no device-name
special-casing anywhere; every rule is a pure geometric predicate applied uniformly.

THE 8 CANONICAL INVARIANT CATEGORIES -> ExtractedIntent FIELDS (the translation)
================================================================================
The canonical dict (keys ``relative_order`` / ``adjacency`` / ``alignment`` /
``band`` / ``symmetry_axis`` / ``size_class`` / ``spacing`` / ``orientation``) maps
onto the ExtractedIntent geometric fields as follows:

  band          -> ``bands{name -> int}``
        canonical ``band["horizontal"]`` is a list of y-clustered tiers in
        ASCENDING-y order (bottom..top). We index it 0=bottom..N=top and assign each
        member its tier index. This is what @place_group ``row=`` consumes.

  relative_order-> ``orders[(a, rel, b)]`` rel in {'left_of','above'}
        WITHIN a band: ``relative_order["a|b"]`` x-sign "-" (a left of b) ->
        ``(a,'left_of',b)``; "+" -> ``(b,'left_of',a)``. This is the within-band
        column order @order consumes. ACROSS bands we additionally record the
        dominant vertical ``above`` relation (lower-band member is BELOW the
        upper-band member) so the partial order is explicit, not only implied by the
        band index.

  adjacency     -> ``abuts[(a, b)]``
        every ``adjacency["a|b"] == True`` (bboxes edge-adjacent within eps) ->
        ``(a,b)`` an @abut pair.

  alignment     -> ``aligns[(axis, [names])]`` axis in {'x','y','centroid'}
        ``alignment["x_center"]`` classes of >= 2 members sharing an x-centre ->
        ``('x', members)`` (vertical alignment, a shared column). ``y_center``
        classes -> ``('y', members)`` (horizontal alignment, a shared row). A class
        that is BOTH an x-class and a y-class (same centroid) -> ``('centroid',
        members)`` (it subsumes the separate x/y emission for those members).

  size_class    -> ``equal_size_groups[set]``
        each ``size_class`` bucket of >= 2 equal-footprint members -> a frozenset
        (the two equal caps come back as one group).

  symmetry_axis -> ``symmetry_axis (float | None)``
        if ANY declared/derived mirror pair ``mirrors == True``, the axis_x of the
        line they reflect about (re-expressed in the absolute reference frame). When
        no pair mirrors, ``None``. (Pairs are derived geometrically here -- see
        ``_derive_mirror_pairs`` -- since the reference carries no declared pair
        list; the canonical extractor's symmetry_axis is then computed on those
        derived pairs so the SAME helper produces the axis.)

  spacing       -> ``spacings[(a, b, min_um)]``
        the representative inter-band channel (``spacing["vertical_channel"]`` -> a
        min gap between the two closest stacked bands) plus notable narrow
        inter-device gaps. Each becomes a ``(a, b, min_um)`` minimum-spacing
        obligation in MICRONS.

  orientation   -> (not emitted as a geometric-placement field) -- orientation is a
        per-device mirror/finger observable the placer realises from symmetry +
        device cell, not a standalone placement constraint the round-trip declares,
        so we do not translate it into an ExtractedIntent field (honest residual).

UNITS
=====
``device_bboxes`` are in MICRONS (the shared contract: ``(x0,y0,x1,y1) um``). The
canonical extractor is unit-agnostic (it clusters / compares with an ``eps`` in the
same unit as the bboxes), so we run it in microns with a micron-scale ``eps``. The
only field that names an explicit unit is ``spacings`` (``min_um``), already in um.

DETERMINISM: every field is built from SORTED members / SORTED tuples, so the same
reference bboxes produce byte-equal ExtractedIntent fields.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from clay.layout.completeness.invariance import extract_invariants

# Micron-scale clustering tolerance. ``device_bboxes`` are in microns; two centres /
# footprints within EPS_UM are "the same" (a shared band / column / equal size).
# Chosen small relative to a device pitch (devices are units-to-tens of um apart,
# aligned features coincide to a fraction of a um) but large enough to absorb the
# sub-um residue of a reference layout's grid snap. Bands are separated at a
# coarser, pitch-derived scale inside the canonical band extractor (0.4 * median
# device dimension), so EPS_UM does NOT set the band granularity.
EPS_UM: float = 0.5

_Bbox = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# placement-view adapter: feed device_bboxes to the CANONICAL extractor       #
# --------------------------------------------------------------------------- #


class _PlacedDeviceView:
    """A device exposing ``.bbox`` -- the shape the canonical extractor reads."""

    __slots__ = ("bbox",)

    def __init__(self, bbox: _Bbox) -> None:
        self.bbox = bbox


class _ConstraintsView:
    """Carries ``symmetry_pairs`` for the canonical symmetry_axis / orientation
    extractors (the reference has no declared pairs, so we feed DERIVED ones)."""

    __slots__ = ("symmetry_pairs",)

    def __init__(self, symmetry_pairs: Sequence[Tuple[str, str]]) -> None:
        self.symmetry_pairs = tuple(symmetry_pairs)


class _PlacementView:
    """Minimal :class:`Placement`-shaped object the canonical extractor consumes.

    The canonical :func:`extract_invariants` reads ``placement.devices`` (``{label ->
    obj with .bbox}``) and ``placement.constraints.symmetry_pairs``. We synthesise
    exactly that from ``device_bboxes`` so the SAME extractor runs on the reference
    positions -- no second geometry implementation.
    """

    __slots__ = ("devices", "constraints")

    def __init__(
        self,
        device_bboxes: Dict[str, _Bbox],
        symmetry_pairs: Sequence[Tuple[str, str]] = (),
    ) -> None:
        self.devices = {
            str(lbl): _PlacedDeviceView(_norm_bbox(bb))
            for lbl, bb in device_bboxes.items()
        }
        self.constraints = _ConstraintsView(symmetry_pairs)


def _norm_bbox(bb: Any) -> _Bbox:
    x0, y0, x1, y1 = (float(v) for v in bb)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _center(bb: _Bbox) -> Tuple[float, float]:
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def _wh(bb: _Bbox) -> Tuple[float, float]:
    return (bb[2] - bb[0], bb[3] - bb[1])


# --------------------------------------------------------------------------- #
# derive mirror pairs (reference carries no declared pair list)               #
# --------------------------------------------------------------------------- #


def _declared_pairs_from_intent(
    intent: Any, device_bboxes: Dict[str, _Bbox]
) -> Tuple[Tuple[str, str], ...]:
    """The netlist half's DECLARED symmetric pairs that have a reference bbox.

    Reads ``intent.matched_pairs`` / ``diff_pairs`` / ``mirror_pairs`` (the topology
    pairs the netlist extractor filled) and keeps every 2-tuple whose BOTH members
    appear in ``device_bboxes``. Returns SORTED canonical pairs ``(min, max)``. Empty
    when the intent declares none (or is a stand-in without those attrs) -> the caller
    falls back to the geometric derivation.
    """
    present = set(device_bboxes)
    pairs: List[Tuple[str, str]] = []
    for attr in ("matched_pairs", "diff_pairs", "mirror_pairs"):
        for ab in getattr(intent, attr, ()) or ():
            if len(ab) != 2:
                continue
            a, b = str(ab[0]), str(ab[1])
            if a in present and b in present:
                pairs.append((a, b) if a <= b else (b, a))
    return tuple(sorted(set(pairs)))


def _derive_mirror_pairs(
    device_bboxes: Dict[str, _Bbox], eps: float
) -> Tuple[Tuple[str, str], ...]:
    """Geometrically derive matched mirror pairs from the reference bboxes.

    A pair ``(a, b)`` is a mirror candidate iff they sit at the SAME y-centre and
    have an EQUAL footprint (the two attributes the canonical ``symmetry_axis``
    extractor's ``mirrors`` flag itself tests). We additionally require that they
    reflect about a COMMON axis shared by the whole candidate set: we group all
    equal-footprint same-y devices, find the dominant reflection axis (the mean of
    candidate pair midpoints), and keep the pairs that mirror about it within eps.

    This is mechanical (no device names): equal size + equal y + reflect-about-a-line
    is the geometric signature of a declared symmetric pair. Returns SORTED canonical
    pairs ``(min(a,b), max(a,b))``. When fewer than one valid pair exists, returns ().
    """
    devs = {lbl: _norm_bbox(bb) for lbl, bb in device_bboxes.items()}
    # bucket by (y-centre, footprint) -> candidate twins sit in the same bucket.
    buckets: Dict[Tuple[int, int, int], List[str]] = {}
    for lbl in sorted(devs):
        bb = devs[lbl]
        cy = _center(bb)[1]
        w, h = _wh(bb)
        key = (
            round(cy / max(eps, 1e-9)),
            round(w / max(eps, 1e-9)),
            round(h / max(eps, 1e-9)),
        )
        buckets.setdefault(key, []).append(lbl)

    # collect every same-y, same-size candidate pair + its midpoint axis.
    cand: List[Tuple[str, str, float]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            mid = (_center(devs[a])[0] + _center(devs[b])[0]) / 2.0
            cand.append((a, b, mid))
    if not cand:
        return ()

    # dominant axis = median of candidate midpoints (robust to an odd one out).
    mids = sorted(m for _a, _b, m in cand)
    axis = mids[len(mids) // 2]

    pairs: List[Tuple[str, str]] = []
    used: set = set()
    # keep pairs reflecting about the dominant axis, longest-arm first so a real
    # outer twin pair is preferred over a degenerate near-axis coincidence.
    for a, b, mid in sorted(
        cand, key=lambda t: -abs(_center(devs[t[0]])[0] - _center(devs[t[1]])[0])
    ):
        if a in used or b in used:
            continue
        if abs(mid - axis) <= eps:
            xa = _center(devs[a])[0]
            xb = _center(devs[b])[0]
            # require an actual reflection (centres on opposite sides of the axis)
            if abs(abs(xa - axis) - abs(xb - axis)) <= eps and (xa - axis) * (
                xb - axis
            ) <= eps * eps:
                pairs.append((a, b) if a <= b else (b, a))
                used.add(a)
                used.add(b)
    return tuple(sorted(set(pairs)))


# --------------------------------------------------------------------------- #
# translate canonical 8-category dict -> ExtractedIntent geometric fields      #
# --------------------------------------------------------------------------- #


def _bands_from_canonical(canonical: Dict[str, Any]) -> Dict[str, int]:
    """``band["horizontal"]`` (ascending-y tiers) -> ``{name -> tier index}``.

    The canonical horizontal bands are y-clustered tiers in ascending-y order, so
    index 0 is the BOTTOM tier and the last index is the TOP tier -- exactly the
    @place_group ``row=`` convention.
    """
    tiers: List[List[str]] = canonical["band"]["horizontal"]
    out: Dict[str, int] = {}
    for idx, tier in enumerate(tiers):
        for name in tier:
            out[name] = idx
    return out


def _orders_from_canonical(
    canonical: Dict[str, Any], bands: Dict[str, int]
) -> List[Tuple[str, str, str]]:
    """``relative_order`` -> within-band ``left_of`` + cross-band ``above`` tuples.

    For a pair "a|b" with x-sign "-": a is left of b. WITHIN the same band that is
    the column order (@order): ``(a,'left_of',b)``. ACROSS bands we emit the dominant
    vertical relation from the y-sign: y-sign "+" means a is ABOVE b ->
    ``(a,'above',b)``; "-" -> ``(b,'above',a)``. Cross-band ``left_of`` is NOT
    emitted (column order is only meaningful within a tier); same-band ``above`` is
    not emitted (a tier shares a y baseline). Deterministic: sorted by tuple.
    """
    ro: Dict[str, str] = canonical["relative_order"]
    out: List[Tuple[str, str, str]] = []
    for key, rel in ro.items():
        a, b = key.split("|")
        xr, yr = rel.split(",")
        same_band = bands.get(a) is not None and bands.get(a) == bands.get(b)
        if same_band:
            # within-band column order from the x relation
            if xr == "-":
                out.append((a, "left_of", b))
            elif xr == "+":
                out.append((b, "left_of", a))
        else:
            # cross-band dominant vertical order from the y relation
            if yr == "+":
                out.append((a, "above", b))
            elif yr == "-":
                out.append((b, "above", a))
    return sorted(set(out))


def _abuts_from_canonical(canonical: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``adjacency`` True entries -> ``(a, b)`` abut pairs (sorted canonical)."""
    adj: Dict[str, bool] = canonical["adjacency"]
    out: List[Tuple[str, str]] = []
    for key, touching in adj.items():
        if touching:
            a, b = key.split("|")
            out.append((a, b) if a <= b else (b, a))
    return sorted(set(out))


def _aligns_from_canonical(
    canonical: Dict[str, Any],
) -> List[Tuple[str, List[str]]]:
    """``alignment`` x/y classes -> ``(axis, [names])`` align groups.

    An x_center class of >= 2 -> a vertical alignment ``('x', members)``; a y_center
    class of >= 2 -> a horizontal alignment ``('y', members)``. When the SAME member
    set appears in BOTH an x-class and a y-class the devices share a full centroid ->
    ``('centroid', members)`` is emitted INSTEAD of the separate x/y pair (it is the
    stronger, subsuming constraint). Deterministic: members sorted, groups sorted.
    """
    x_classes = [sorted(c) for c in canonical["alignment"]["x_center"] if len(c) >= 2]
    y_classes = [sorted(c) for c in canonical["alignment"]["y_center"] if len(c) >= 2]

    x_sets = {tuple(c) for c in x_classes}
    y_sets = {tuple(c) for c in y_classes}
    centroid_sets = x_sets & y_sets

    out: List[Tuple[str, List[str]]] = []
    for c in sorted(centroid_sets):
        out.append(("centroid", list(c)))
    for c in x_classes:
        if tuple(c) not in centroid_sets:
            out.append(("x", c))
    for c in y_classes:
        if tuple(c) not in centroid_sets:
            out.append(("y", c))
    # sort by (axis, members) for determinism
    return sorted(out, key=lambda t: (t[0], tuple(t[1])))


def _equal_size_from_canonical(canonical: Dict[str, Any]) -> List["frozenset[str]"]:
    """``size_class`` buckets of >= 2 -> equal-footprint groups (frozensets)."""
    out: List[frozenset] = []
    for cls in canonical["size_class"]:
        if len(cls) >= 2:
            out.append(frozenset(cls))
    return sorted(out, key=lambda s: tuple(sorted(s)))


def _symmetry_axis_from_canonical(
    canonical: Dict[str, Any], frame_x: float
) -> Optional[float]:
    """``symmetry_axis`` -> the absolute x-coordinate of the mirror line, or None.

    The canonical ``symmetry_axis["a|b"]["axis_x"]`` is expressed relative to the
    centroid frame (``frame_x`` subtracted) so it is translation-invariant; we ADD
    ``frame_x`` back to report the axis in the ABSOLUTE reference frame the caller's
    bboxes live in. If multiple pairs mirror, we take the mean of their (equal) axes.
    Returns None when no pair actually ``mirrors``.
    """
    axes: List[float] = []
    for _key, v in canonical["symmetry_axis"].items():
        if v.get("mirrors"):
            axes.append(float(v["axis_x"]) + frame_x)
    if not axes:
        return None
    return sum(axes) / len(axes)


def _spacings_from_canonical(
    canonical: Dict[str, Any],
    device_bboxes: Dict[str, _Bbox],
    bands: Dict[str, int],
) -> List[Tuple[str, str, float]]:
    """Representative inter-band channel + notable narrow inter-device gaps.

    (1) Inter-band channel: the canonical ``spacing["vertical_channel"]`` is the min
    edge-to-edge gap between adjacent horizontal (stacked) bands. We attribute it to
    the specific (lower-band member, upper-band member) that realises that minimum
    gap, emitting ``(lower, upper, gap_um)``. (2) Notable inter-device gaps: among
    NON-abutting device pairs, the few narrowest centre-region gaps are recorded as
    minimum-spacing obligations so the round-trip preserves a deliberately small
    channel. Deterministic: sorted tuples. Values already in microns.
    """
    out: List[Tuple[str, str, float]] = []
    devs = {lbl: _norm_bbox(bb) for lbl, bb in device_bboxes.items()}

    # (1) inter-band vertical channel attributed to the realising pair.
    vch = canonical["spacing"].get("vertical_channel")
    if vch is not None:
        # find adjacent-band member pair with the minimum vertical edge gap.
        tiers: List[List[str]] = canonical["band"]["horizontal"]
        best: Optional[Tuple[float, str, str]] = None
        for i in range(len(tiers) - 1):
            lower, upper = tiers[i], tiers[i + 1]
            for ld in lower:
                for ud in upper:
                    gap = devs[ud][1] - devs[ld][3]  # upper.y0 - lower.y1
                    if best is None or gap < best[0]:
                        best = (gap, ld, ud)
        if best is not None and best[0] > 0:
            out.append((best[1], best[2], round(best[0], 4)))
    inter_band_pairs = {(a, b) for a, b, _ in out}

    # (2) notable narrow gaps among non-abutting pairs (representative few).
    adj: Dict[str, bool] = canonical["adjacency"]
    gaps: List[Tuple[float, str, str]] = []
    for a, b in combinations(sorted(devs), 2):
        key = f"{a}|{b}"
        if adj.get(key):
            continue  # abutting -> recorded as @abut, not a spacing
        ba, bb = devs[a], devs[b]
        # edge-to-edge gap on the axis with positive separation (min over x/y).
        gx = max(ba[0] - bb[2], bb[0] - ba[2])
        gy = max(ba[1] - bb[3], bb[1] - ba[3])
        sep = max(gx, gy)  # >0 only when they do not overlap on at least one axis
        if sep > 0:
            gaps.append((round(sep, 4), a, b))
    # keep the narrowest few (those are the deliberate tight channels worth pinning).
    gaps.sort()
    seen = set(inter_band_pairs)
    for sep, a, b in gaps[:2]:
        if (a, b) not in seen:
            out.append((a, b, sep))
            seen.add((a, b))
    # dedup any (a,b) pair (the inter-band channel and the narrow-gap pass can pick
    # the same realising pair); keep the first (smallest) occurrence.
    deduped: List[Tuple[str, str, float]] = []
    emitted: set = set()
    for a, b, sep in sorted(out, key=lambda t: (t[0], t[1])):
        if (a, b) in emitted:
            continue
        emitted.add((a, b))
        deduped.append((a, b, sep))
    return deduped


# --------------------------------------------------------------------------- #
# the public entry point                                                      #
# --------------------------------------------------------------------------- #


def extract_geometric_intent(intent: Any, gds_path: Optional[str] = None) -> None:
    """Fill the GEOMETRIC fields of ``intent`` from ``intent.device_bboxes``.

    Mutates ``intent`` IN PLACE (returns None), filling the geometric placement-intent
    fields of the shared :class:`ExtractedIntent`:

      * ``bands``             ``{name -> tier index}`` (0 = bottom)
      * ``orders``            ``[(a, rel, b)]`` rel in {'left_of','above'}
      * ``abuts``             ``[(a, b)]``
      * ``aligns``            ``[(axis, [names])]`` axis in {'x','y','centroid'}
      * ``equal_size_groups`` ``[frozenset(names)]``
      * ``symmetry_axis``     float x-coord (absolute frame) or None
      * ``spacings``          ``[(a, b, min_um)]``

    Derivation is MECHANICAL and identical for any design (no device-name special
    casing): it runs the CANONICAL 8-category invariant extractor on the reference
    bboxes and translates the canonical dict into these fields (see module docstring
    for the per-category translation). When ``gds_path`` is given we use
    :func:`clay.layout.completeness.gds_invariants.extract_invariants_from_gds` so the
    invariants come from the REAL emitted geometry (cap footprints = their FuseTop
    plate); otherwise we run :func:`extract_invariants` directly on the bboxes. Both
    return the identical canonical dict shape, so the translation is the same.

    ``intent.device_bboxes`` must be ``{name -> (x0,y0,x1,y1)}`` in microns (filled by
    the sibling extractor). If it is empty/absent, every geometric field is set to its
    safe empty default and ``symmetry_axis`` to None.
    """
    device_bboxes_raw = getattr(intent, "device_bboxes", None) or {}
    device_bboxes: Dict[str, _Bbox] = {
        str(k): _norm_bbox(v) for k, v in device_bboxes_raw.items()
    }

    if not device_bboxes:
        _assign(intent, "bands", {})
        _assign(intent, "orders", [])
        _assign(intent, "abuts", [])
        _assign(intent, "aligns", [])
        _assign(intent, "equal_size_groups", [])
        _assign(intent, "symmetry_axis", None)
        _assign(intent, "spacings", [])
        return

    # Mirror pairs for the canonical symmetry_axis extractor. PREFER the netlist
    # half's DECLARED topology pairs (matched/diff/mirror) when present on the intent
    # -- they are the ground-truth symmetric roles -- restricted to pairs both of
    # whose members have a reference bbox. Fall back to a GEOMETRIC derivation (equal
    # size + equal y + reflect-about-a-line) when the netlist declared none.
    pairs = _declared_pairs_from_intent(intent, device_bboxes)
    if not pairs:
        pairs = _derive_mirror_pairs(device_bboxes, EPS_UM)

    # canonical frame origin (centroid-of-centroids x) -- the canonical extractor
    # subtracts it from every x-referenced invariant; we add it back for the
    # absolute-frame symmetry_axis.
    xs = [_center(bb)[0] for bb in device_bboxes.values()]
    frame_x = sum(xs) / len(xs)

    # run the CANONICAL extractor (GDS-truth when a path is given, else on bboxes).
    if gds_path is not None:
        from clay.layout.completeness.gds_invariants import extract_invariants_from_gds

        canonical = extract_invariants_from_gds(
            gds_path,
            device_labels=device_bboxes,
            intent=_PairsIntent(pairs),
        )
    else:
        view = _PlacementView(device_bboxes, symmetry_pairs=pairs)
        canonical = extract_invariants(view, eps=EPS_UM)

    bands = _bands_from_canonical(canonical)

    _assign(intent, "bands", bands)
    _assign(intent, "orders", _orders_from_canonical(canonical, bands))
    _assign(intent, "abuts", _abuts_from_canonical(canonical))
    _assign(intent, "aligns", _aligns_from_canonical(canonical))
    _assign(intent, "equal_size_groups", _equal_size_from_canonical(canonical))
    _assign(
        intent, "symmetry_axis", _symmetry_axis_from_canonical(canonical, frame_x)
    )
    _assign(
        intent,
        "spacings",
        _spacings_from_canonical(canonical, device_bboxes, bands),
    )


class _PairsIntent:
    """Tiny intent carrying derived ``symmetry_pairs`` for the GDS extractor path.

    :func:`extract_invariants_from_gds` reads declared pairs off
    ``intent.symmetry_pairs`` (see ``gds_invariants._declared_symmetry_pairs``); we
    thread our geometrically-derived pairs through this minimal object so the
    GDS-truth path uses the SAME pairs the bbox path does.
    """

    __slots__ = ("symmetry_pairs",)

    def __init__(self, pairs: Sequence[Tuple[str, str]]) -> None:
        self.symmetry_pairs = tuple(pairs)


def _assign(obj: Any, field: str, value: Any) -> None:
    """Set ``obj.field = value`` defensively (the shared dataclass declares the slot;
    a dict-shaped fallback intent is also supported for default-safety)."""
    try:
        setattr(obj, field, value)
    except Exception:  # noqa: BLE001 - a mapping-shaped intent
        try:
            obj[field] = value  # type: ignore[index]
        except Exception:  # noqa: BLE001
            pass


__all__ = ["extract_geometric_intent", "EPS_UM"]
