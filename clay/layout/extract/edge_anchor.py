"""The ``@edge_anchor`` layout intent: a large passive (a MIM cap) anchors
FLUSH against ONE die edge so its anchored side carries NO active circuit and
the die DOES NOT GROW (E / V; the L side lives in
:mod:`clay.layout.extract.place_relax`).

WHY THIS, NOT "@peripheral_device" (the prior, wrong, model -- retired here):
the earlier intent declared "keep a large cap out of a central rectangle" and
lowered it by pushing the cap along its NARROW axis out of that rectangle. On a
fixed-height core that shoves a wide cap UP/DOWN and GROWS the die (measured on
t4: outline 7978 -> 9157 um^2, 1.15x; utilisation fell). That is the opposite of
the goal. The goal (measured on the signed external_pdk test4 reference) is:

  die x[-38,95] y[0,80]; cap x[31,91] -- the cap's RIGHT edge sits 4 um from the
  die right wall (FLUSH), and the band RIGHT OF the cap has active = 0 um^2,
  while the left/top/bottom bands carry 698 / 1478 / 1014 um^2 of active. The
  cap turns ONE of its sides to the die boundary; the signal does not have to
  route around it on that side. The die is NOT enlarged to make room -- the cap
  occupies a corner of the EXISTING outline.

THE INTENT CHAIN (a DECLARED design prior -- no engine-baked policy, no
per-circuit branch, no reference read):

  E -- :func:`declare_edge_anchor` constructs intent entries
       ``{'device': name-or-'*'|kind 'cap', 'provenance': 'declared'}`` on
       ``ExtractedIntent.edge_anchor_devices``. A DESIGN PRIOR ("a large MIM cap
       anchors flush to a die edge so that edge is signal-free"), the general
       analog-layout practice -- NOT a fact lifted off a reference layout. The
       provenance is therefore always ``'declared'`` and NO coordinate ever
       enters an entry (the leakage boundary). The EDGE is NOT declared: the
       lowering SOLVES for the edge that keeps the die smallest.
       :func:`is_edge_anchor` is the ONE matching rule (kind ``'cap'`` matches
       any cap; an exact device name; or ``'*'``) shared by the lowering and
       this obligation.

  L -- :mod:`clay.layout.extract.place_relax` adds an EDGE-ANCHOR potential to
       its relaxation energy: every matched device is pulled so ONE of its
       sides goes FLUSH against the nearest die wall on the SOLVED anchor axis,
       WITHOUT pushing the die wall outward (the outline term already resists
       growth; the anchor target is the seed die wall, never beyond it). Default
       OFF (no matching entry) is byte-identical.

  V -- :func:`edge_anchor_obligation` measures the obligation on the PLACED
       artwork: for each matched cap, the side of the cap facing its nearest die
       edge must have ZERO active (device) area between the cap and that wall
       (the cap is not surrounded on all four sides -- one side opens to the die
       boundary) AND the whole-layout outline area must be <= the supplied
       ``canonical_area_um2`` budget (the die did not grow). 5-state verdict
       (pass / fail / unmeasurable / unsupported / advisory) -- never a silent
       pass.

THE DECLARED PHYSICAL BASIS (recorded, not extracted): on the external_pdk canonical the
signed layout anchors CC1 flush to the RIGHT die wall, leaving that side active-
free while the other three sides hold the circuit. That is GENERAL practice; the
intent declares it generically (the anchor edge and the active fill on the other
three sides are SOLVED, never read off the reference).
"""
from __future__ import annotations

from clay.layout.placed_geometry import placed_point_nm

from typing import Any, Dict, List, Mapping, Optional, Tuple

_Bbox = Tuple[float, float, float, float]
_XY = Tuple[float, float]

#: A cap side counts as FLUSH against a die wall when the gap to that wall is
#: at most this fraction of the cap's span on the perpendicular axis. Relative
#: (scale-free). The reference cap's right gap is 4 um against a 60 um-wide cap
#: (~0.07); the obligation's load-bearing fact is the ZERO active on the open
#: side, this flag corroborates "the open side faces the boundary".
_FLUSH_FRAC: float = 0.25

#: Active area below this (um^2) on the cap's open side counts as ZERO -- a few
#: stray fill/keepout slivers must not flip the verdict. Absolute, tiny.
_OPEN_SIDE_EPS_UM2: float = 1.0

#: The whole-layout outline may exceed the canonical budget by at most this
#: relative slack and still pass (numeric press jitter; NOT a licence to grow).
_AREA_SLACK_FRAC: float = 0.0


# --------------------------------------------------------------------------- #
# E -- declared construction + the shared matching rule
# --------------------------------------------------------------------------- #
def declare_edge_anchor(device: str = "cap") -> List[Dict[str, Any]]:
    """Construct a DECLARED ``@edge_anchor`` intent entry (E side).

    ``device`` is a device NAME, the kind token ``'cap'`` (any cap), or ``'*'``
    (any device). The entry is a design PRIOR: ``provenance='declared'`` and it
    carries NO coordinate and NO edge -- a designer's statement that this class
    of device anchors flush to SOME die edge, the edge SOLVED by the lowering.
    Returns a one-element list (compose by concatenation) so it slots straight
    onto ``ExtractedIntent.edge_anchor_devices``."""
    return [{"device": str(device), "provenance": "declared"}]


def is_edge_anchor(entries: Any, name: str, rec: Any = None) -> bool:
    """The ONE matching rule (shared by the lowering and the obligation).

    ``name`` is a device/passive name and ``rec`` its record (for the kind
    test). An entry matches when its ``device`` is the exact ``name``, the
    wildcard ``'*'``, or the record's KIND token ('cap' matches a cap passive,
    'res' a resistor, 'nmos'/'pmos' a MOS). No coordinate is consulted."""
    kind = str(getattr(rec, "kind", "") or "").lower()
    for e in entries or ():
        tgt = str(e.get("device", "*"))
        if tgt == "*" or tgt == str(name):
            return True
        t = tgt.lower()
        if t == "cap" and (kind.startswith("cap") or kind == "c"):
            return True
        if t == "res" and (kind.startswith("res") or kind == "r"):
            return True
        if t in ("nmos", "pmos") and kind == t:
            return True
    return False


def _network_anchor_members(intent: Any) -> List[str]:
    """Members DERIVED from the P2-EXTRACTED relation network (P4 closed loop).

    A device is an edge-anchor member when the EXTRACTED network records an
    ANCHORED device<->frame TPP relation for it (the cap hugs ONE die-boundary
    side -- ``relate_frame(cap, side, {'TPP'}, REQUIRED)`` from
    ``extract_placement_relations``). This is a STYLE-TRANSFER fact lifted off
    the reference structure, NOT a declared prior -- the loop the P4 test pins.

    LEAKAGE BOUNDARY: this reads ONLY relation symbols from the network (the
    network is zero-coordinate by construction; see placement_relations.py P1).
    It NEVER touches ``intent.device_bboxes`` or any geometry, so the derived
    member set is invariant to any translation of the reference bboxes (the
    closure test's translation-invariance assertion). No ``if name == ...``
    branch: membership is the categorical anchored-TPP-frame relation, the same
    rule for every device.

    Deterministic order (the network's sorted device order)."""
    net = getattr(intent, "placement_relations", None)
    if net is None:
        return []
    # The network must expose its symbolic-relation accessors. Read frame sides
    # and the device list from its public symbol surface only.
    try:
        sides = list(net.frame_sides())
        devices = list(net.devices())
    except Exception:
        return []
    if not sides:
        return []
    out: List[str] = []
    for node in devices:  # already deterministic (sorted in RelationNetwork)
        nm = str(getattr(node, "name", ""))
        if not nm:
            continue
        for side in sides:
            if not net.has_frame_relation(nm, side):
                continue
            edge = net.relation_frame(nm, side)
            # ANCHORED == the relation's allowed-symbol set is a single symbol
            # (a sharp boundary-hug, not a free/ordered disjunction). The
            # anchoring is computed from the symbol-set cardinality by the
            # network itself -- a pure symbol fact, no coordinate.
            if getattr(edge, "anchoring", None) == "anchored":
                out.append(nm)
                break
    return out


def edge_anchor_members(intent: Any, placement: Any = None) -> List[str]:
    """The list of device/passive NAMES that anchor flush to a die edge (the
    lowering's member set). ``placement`` (optional) restricts to placed names.
    Deterministic order. No coordinate is ever read.

    TWO SOURCES, taken as a UNION (P4 closed loop):

      * DECLARED -- ``intent.edge_anchor_devices`` entries (the design prior,
        :func:`declare_edge_anchor`). Back-compat: an explicit @edge_anchor.
      * EXTRACTED -- ``intent.placement_relations`` (the P2 RelationNetwork):
        a device with an ANCHORED device<->frame TPP relation (cap hugs a die
        side) is a member, NO declared prior required. The extract->lower loop:
        the cap's boundary-hug intent comes from STYLE-TRANSFER off the
        reference, lifted as relation symbols only (zero-coordinate, leakage-
        free; see :func:`_network_anchor_members`).

    Default OFF (neither source present) returns ``[]`` -- byte-identical to
    the prior placement (the relax energy term is gated on a non-empty set)."""
    declared: List[str] = []
    entries = getattr(intent, "edge_anchor_devices", None) or []
    if entries:
        for rec in (list(getattr(intent, "devices", []) or [])
                    + list(getattr(intent, "passives", []) or [])):
            nm = str(getattr(rec, "name", ""))
            if not nm:
                continue
            if is_edge_anchor(entries, nm, rec):
                declared.append(nm)

    extracted = _network_anchor_members(intent)

    # UNION, deterministic: declared order first (device then passive order),
    # then any extracted-only names in network order. A name in both appears
    # once, at its declared position.
    out: List[str] = []
    seen = set()
    for nm in declared + extracted:
        if nm in seen:
            continue
        seen.add(nm)
        if placement is not None and nm not in placement:
            continue
        out.append(nm)
    return out


# --------------------------------------------------------------------------- #
# geometry helpers (pure functions, no reference read)
# --------------------------------------------------------------------------- #
def _box_of(name: str, placement: Mapping[str, _XY],
            sizes: Mapping[str, _XY],
            feet: Optional[Mapping[str, _Bbox]] = None) -> Optional[_Bbox]:
    if feet is not None and name in feet:
        return tuple(float(v) for v in feet[name])  # canonical artwork foot
    pos = placement.get(name)
    s = sizes.get(name)
    if pos is None or s is None:
        return None
    w, h = float(s[0]), float(s[1])
    return (float(pos[0]) - w / 2.0, float(pos[1]) - h / 2.0,
            float(pos[0]) + w / 2.0, float(pos[1]) + h / 2.0)


def _outline(placement: Mapping[str, _XY], sizes: Mapping[str, _XY],
             feet: Optional[Mapping[str, _Bbox]] = None) -> Optional[_Bbox]:
    xs0: List[float] = []; ys0: List[float] = []
    xs1: List[float] = []; ys1: List[float] = []
    names = set(placement) | (set(feet) if feet else set())
    for nm in names:
        bb = _box_of(nm, placement, sizes, feet)
        if bb is None:
            continue
        xs0.append(bb[0]); ys0.append(bb[1]); xs1.append(bb[2]); ys1.append(bb[3])
    if not xs0:
        return None
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _nearest_edge(cap: _Bbox, die: _Bbox) -> Tuple[str, float]:
    """Return (side, gap) for the die wall the cap is closest to, where ``side``
    is one of 'L','R','B','T' and ``gap`` the distance from the cap's that-side
    edge to the die wall. The OPEN side of the cap (the side that should carry no
    active) is the one facing the nearest die wall."""
    gaps = {
        "L": cap[0] - die[0],
        "R": die[2] - cap[2],
        "B": cap[1] - die[1],
        "T": die[3] - cap[3],
    }
    side = min(gaps, key=lambda k: gaps[k])
    return side, gaps[side]


def _open_side_band(cap: _Bbox, die: _Bbox, side: str) -> _Bbox:
    """The rectangle BETWEEN the cap's ``side`` edge and the die wall on that
    side, spanning the cap's extent on the perpendicular axis -- the region that
    must be active-free for the cap to face the boundary on its open side."""
    if side == "R":
        return (cap[2], cap[1], die[2], cap[3])
    if side == "L":
        return (die[0], cap[1], cap[0], cap[3])
    if side == "T":
        return (cap[0], cap[3], cap[2], die[3])
    return (cap[0], die[1], cap[2], cap[1])  # 'B'


def _overlap_area(a: _Bbox, b: _Bbox) -> float:
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox <= 0.0 or oy <= 0.0:
        return 0.0
    return ox * oy


# --------------------------------------------------------------------------- #
# V (AS-BUILT) -- the obligation measured on the canonical AsBuiltBundle frame
# --------------------------------------------------------------------------- #
#: The active layers the open-side census counts, identical to the independent
#: oracle (tests/test_cap_not_surrounded.py): diffusion plus signal metal 1..4.
#: A cap side is "clear" only when NO device diffusion AND NO routed signal
#: metal lands between the cap and the die wall on that side -- the cap is not a
#: road-block the signal must wrap around. (The earlier obligation measured the
#: PLACEMENT-STAMP frame -- device boxes only, no routed metal -- and reported a
#: surrounded as-built cap as PASS: the native<->oracle disagreement this path
#: closes. Both now read the same as-built geometry.)
_DIFF_LAYER: Tuple[int, int] = (3, 0)
_ACTIVE_METAL_N: Tuple[int, ...] = (1, 2, 3, 4)


def _asbuilt_side_active(
    intent: Any,
    placement: Mapping[str, _XY],
    geometry: Any,
    routed: Any,
    tech: Any,
    matched_caps: List[str],
) -> Optional[Dict[str, Any]]:
    """Measure each matched cap's four-side ACTIVE census on the canonical
    as-built geometry (device diffusion + M1..M4 + routed signal metal) in the
    ABSOLUTE GDS frame -- the SAME region algebra and the SAME layer set the
    independent oracle uses, so native and oracle agree by construction.

    Returns ``{name: {footprint, open_side, side_active_um2, satisfied}}`` plus a
    ``'__die__'`` key holding the as-built die bbox (um) and its area, or ``None``
    when the as-built geometry cannot be assembled (klayout absent / no routed
    result / no cap plate) -- the caller then falls back to the placement-frame
    measure and labels the verdict honestly (never a silent pass on no evidence).

    Leakage-free: every input is design-self-produced (the placed device cells,
    the cap's canonical footprint, the router's own wires). No reference layout
    is read."""
    if routed is None or tech is None or geometry is None:
        return None
    try:
        import klayout.db as kdb
    except Exception:
        return None

    _NM = 1000.0
    metal_keys = []
    for k in _ACTIVE_METAL_N:
        try:
            metal_keys.append(tuple(tech.metal_key(k)))
        except Exception:
            break
    active_keys = set(metal_keys) | {_DIFF_LAYER}

    # cap plate layers (the plate dwarfs any wire on the same layer): the cap
    # footprint comes from the canonical extractor, NOT a layer scan, so this
    # set is only used to EXCLUDE the plate from the active census.
    cap_plate_keys = set()
    for k in (5, 4):
        try:
            cap_plate_keys.add(tuple(tech.metal_key(k)))
        except Exception:
            pass

    # ACTIVE region (diffusion + signal metal 1..4), absolute frame.
    active = kdb.Region()
    full = kdb.Region()  # the whole as-built extent (for the die bbox)
    for nm, g in dict(geometry or {}).items():
        pos = placement.get(nm)
        if pos is None:
            continue
        for lk, rings in (getattr(g, "polys_by_layer", {}) or {}).items():
            for ring in rings:
                if len(ring) < 3:
                    continue
                poly = kdb.Polygon([
                    kdb.Point(*placed_point_nm(float(x), float(y), pos))
                    for (x, y) in ring])
                full.insert(poly)
                if tuple(lk) in active_keys:
                    active.insert(poly)
    for lp in getattr(routed, "wires", []) or []:
        key = (int(lp.layer), int(lp.datatype))
        poly = kdb.Polygon([kdb.Point(int(v.x), int(v.y))
                            for v in lp.polygon.vertices])
        full.insert(poly)
        if key in active_keys:
            active.insert(poly)
    active = active.merged()
    full = full.merged()
    if full.is_empty():
        return None
    db = full.bbox()

    # cap footprints (canonical artwork, the same source the router keep-out and
    # the plate-distinct check read) -- the absolute frame.
    feet: Dict[str, _Bbox] = {}
    try:
        from clay.layout.extract.under_cap_intent import cap_footprints_um
        feet = cap_footprints_um(intent, dict(placement), geometry, tech=tech)
    except Exception:
        feet = {}

    out: Dict[str, Any] = {}
    for nm in matched_caps:
        bb = feet.get(nm)
        if bb is None:
            return None  # no canonical cap plate -> cannot measure as-built
        cap = kdb.Box(int(round(bb[0] * _NM)), int(round(bb[1] * _NM)),
                      int(round(bb[2] * _NM)), int(round(bb[3] * _NM)))
        sides = {
            "R": kdb.Box(cap.right, cap.bottom, db.right, cap.top),
            "L": kdb.Box(db.left, cap.bottom, cap.left, cap.top),
            "T": kdb.Box(cap.left, cap.top, cap.right, db.top),
            "B": kdb.Box(cap.left, db.bottom, cap.right, cap.bottom),
        }
        side_active = {k: round((active & kdb.Region(v)).area() / 1e6, 4)
                       for k, v in sides.items()}
        open_side = min(side_active, key=lambda k: side_active[k])
        out[nm] = {
            "footprint": tuple(round(v, 4) for v in bb),
            "open_side": open_side,
            "open_side_active_um2": side_active[open_side],
            "side_active_um2": side_active,
            "satisfied": bool(side_active[open_side] <= _OPEN_SIDE_EPS_UM2),
            "frame": "as-built",
        }
    out["__die__"] = {
        "bbox": (round(db.left / _NM, 4), round(db.bottom / _NM, 4),
                 round(db.right / _NM, 4), round(db.top / _NM, 4)),
        "area_um2": round(db.width() / _NM * db.height() / _NM, 4),
    }
    return out


# --------------------------------------------------------------------------- #
# V -- the obligation checker (5 states, never a silent pass)
# --------------------------------------------------------------------------- #
def edge_anchor_obligation(
    intent: Any,
    placement: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    *,
    canonical_area_um2: Optional[float] = None,
    geometry: Any = None,
    tech: Any = None,
    routed: Any = None,
) -> Dict[str, Any]:
    """Measure the ``@edge_anchor`` obligation on a placed layout (V side).

    For every CAP an edge-anchor entry matches, BOTH must hold:

      (1) OPEN SIDE FACES THE BOUNDARY -- the band between the cap's side that
          faces its nearest die wall and that wall carries NO active (device)
          area (the cap is not surrounded on all four sides; the signal does not
          have to route around it on the open side). The cap footprint is the
          canonical ``under_cap_intent.cap_footprints_um`` when ``geometry`` +
          ``tech`` are supplied (the same source the router keep-out uses --
          single canonical artwork), else the ``sizes`` rectangle.

      (2) THE DIE DID NOT GROW -- the whole-layout outline area is <=
          ``canonical_area_um2`` (when supplied). Anchoring to a die edge must
          free a signal channel, NOT claim new die area.

    5-STATE VERDICT (never a silent pass):
      * ``unsupported``  -- no edge-anchor entry declared (nothing to check);
      * ``unmeasurable`` -- entries declared but NO matched cap is placed/sized
                            (vacuous: refuse to report pass on no evidence);
      * ``fail``         -- at least one matched cap is surrounded on all four
                            sides (active on the open side), OR the outline grew
                            past the canonical budget;
      * ``pass``         -- every matched cap's open side is active-free AND the
                            outline is within budget.

    Returns ``{'verdict', 'caps': {name: {...}}, 'die_bbox', 'outline_area_um2',
    'canonical_area_um2', 'n_caps'}``.
    """
    entries = getattr(intent, "edge_anchor_devices", None) or []
    if not entries:
        return {"verdict": "unsupported", "caps": {}, "die_bbox": None,
                "outline_area_um2": None, "canonical_area_um2": canonical_area_um2,
                "n_caps": 0, "reason": "no @edge_anchor entry declared"}

    # THE SINGLE CANONICAL ARTWORK FRAME: when ``geometry`` is supplied (the
    # routed AsBuiltBundle), every box is read in the geometry's ABSOLUTE frame
    # -- device stamps from their poly extents, caps from
    # ``under_cap_intent.cap_footprints_um`` (the same source the router keep-out
    # uses). Mixing the placement-centre frame (sizes rectangle centred on the
    # placement) with the absolute geometry frame mis-measured every side when a
    # stamp's poly centroid differs from its placement key. The sizes rectangle
    # is the FALLBACK only (no geometry -- the synthetic path).
    feet: Dict[str, _Bbox] = {}
    geom_boxes: Dict[str, _Bbox] = {}
    if geometry is not None and tech is not None:
        try:
            from clay.layout.extract.under_cap_intent import cap_footprints_um

            feet = cap_footprints_um(intent, dict(placement), geometry, tech=tech)
        except Exception:
            feet = {}
    if geometry is not None:
        for nm, g in dict(geometry).items():
            xs = [pt[0] for rings in getattr(g, "polys_by_layer", {}).values()
                  for ring in rings for pt in ring]
            ys = [pt[1] for rings in getattr(g, "polys_by_layer", {}).values()
                  for ring in rings for pt in ring]
            if xs:
                geom_boxes[nm] = (min(xs), min(ys), max(xs), max(ys))

    def _abs_box(nm: str) -> Optional[_Bbox]:
        if nm in feet:
            return tuple(float(v) for v in feet[nm])   # cap canonical foot
        if nm in geom_boxes:
            return geom_boxes[nm]                       # stamp absolute extent
        return _box_of(nm, placement, sizes, None)      # synthetic fallback

    matched_caps: List[str] = []
    for p in getattr(intent, "passives", []) or []:
        nm = str(getattr(p, "name", ""))
        kind = str(getattr(p, "kind", "")).lower()
        if not (kind.startswith("cap") or kind == "c"):
            continue
        if not is_edge_anchor(entries, nm, p):
            continue
        matched_caps.append(nm)

    # AS-BUILT FRAME (the iron law: every checker eats the SAME canonical
    # AsBuiltBundle). When ``routed`` is supplied the open-side census reads the
    # device diffusion + signal metal 1..4 + the ROUTED wires -- identical layer
    # algebra to the independent oracle -- and the die is the as-built emitted
    # extent. The placement-stamp path below is the FALLBACK only (synthetic
    # tests / no routed result), and its verdict is labelled ``frame: placement``
    # so a placement-frame pass is never mistaken for the as-built truth.
    if routed is not None and matched_caps:
        ab = _asbuilt_side_active(intent, placement, geometry, routed, tech,
                                  matched_caps)
        if ab is not None:
            die_info = ab.pop("__die__")
            caps = ab
            outline_area = float(die_info["area_um2"])
            die = tuple(die_info["bbox"])
            open_ok = all(c["satisfied"] for c in caps.values())
            area_ok = True
            if canonical_area_um2 is not None:
                budget = float(canonical_area_um2) * (1.0 + _AREA_SLACK_FRAC)
                area_ok = outline_area <= budget + 1e-6
            verdict = "pass" if (open_ok and area_ok) else "fail"
            reason = None
            if not open_ok:
                reason = ("a matched cap is surrounded on all four sides in the "
                          "AS-BUILT layout (device diffusion or routed signal "
                          "metal between the cap and the die wall on every side "
                          "-- the signal must route around it)")
            elif not area_ok:
                reason = (f"the as-built outline {outline_area:.1f} um^2 exceeds "
                          f"the canonical budget {canonical_area_um2:.1f} um^2 "
                          "(anchoring grew the die instead of freeing a channel)")
            res = {
                "verdict": verdict,
                "caps": caps,
                "die_bbox": die,
                "outline_area_um2": round(outline_area, 4),
                "canonical_area_um2": canonical_area_um2,
                "n_caps": len(caps),
                "frame": "as-built",
            }
            if reason is not None:
                res["reason"] = reason
            return res

    cap_names = set(matched_caps)
    # the outline + active boxes in ONE frame (absolute when geometry present).
    member_names = set(placement) | set(geom_boxes) | set(feet)
    xs0: List[float] = []; ys0: List[float] = []
    xs1: List[float] = []; ys1: List[float] = []
    active_boxes: List[_Bbox] = []
    for nm in member_names:
        bb = _abs_box(nm)
        if bb is None:
            continue
        xs0.append(bb[0]); ys0.append(bb[1]); xs1.append(bb[2]); ys1.append(bb[3])
        if nm not in cap_names:
            active_boxes.append(bb)
    if not xs0:
        return {"verdict": "unmeasurable", "caps": {}, "die_bbox": None,
                "outline_area_um2": None, "canonical_area_um2": canonical_area_um2,
                "n_caps": 0, "reason": "no placed device with a known size"}
    die = (min(xs0), min(ys0), max(xs1), max(ys1))
    outline_area = (die[2] - die[0]) * (die[3] - die[1])

    caps: Dict[str, Any] = {}
    for nm in matched_caps:
        bb = _abs_box(nm)
        if bb is None:
            continue
        side, gap = _nearest_edge(bb, die)
        band = _open_side_band(bb, die, side)
        open_active = sum(_overlap_area(band, ab) for ab in active_boxes)
        # corroborator: is the cap flush against the wall on its open side?
        perp_span = (bb[3] - bb[1]) if side in ("L", "R") else (bb[2] - bb[0])
        flush = gap <= _FLUSH_FRAC * max(perp_span, 1e-9)
        # the four-side active census (the user-quantified comparison).
        sides = {}
        for s in ("L", "R", "B", "T"):
            b = _open_side_band(bb, die, s)
            sides[s] = round(sum(_overlap_area(b, ab) for ab in active_boxes), 4)
        caps[nm] = {
            "footprint": tuple(round(v, 4) for v in bb),
            "open_side": side,
            "open_side_gap_um": round(gap, 4),
            "open_side_active_um2": round(open_active, 4),
            "flush": bool(flush),
            "side_active_um2": sides,  # the four-side census (L/R/B/T)
            # the load-bearing fact: the open side carries no active circuit.
            "satisfied": bool(open_active <= _OPEN_SIDE_EPS_UM2),
            "frame": "placement",  # device boxes only -- no routed metal seen
        }

    if not caps:
        return {"verdict": "unmeasurable", "caps": {}, "die_bbox": tuple(
            round(v, 4) for v in die), "outline_area_um2": round(outline_area, 4),
            "canonical_area_um2": canonical_area_um2, "n_caps": 0,
            "reason": "an edge-anchor cap entry is declared but no matched cap "
                      "is placed and sized (vacuous-pass guard)"}

    open_ok = all(c["satisfied"] for c in caps.values())
    area_ok = True
    if canonical_area_um2 is not None:
        budget = float(canonical_area_um2) * (1.0 + _AREA_SLACK_FRAC)
        area_ok = outline_area <= budget + 1e-6

    if open_ok and area_ok:
        verdict = "pass"
    else:
        verdict = "fail"
    reason = None
    if not open_ok:
        reason = "a matched cap is surrounded on all four sides (active on its " \
                 "open side -- the signal must route around it)"
    elif not area_ok:
        reason = (f"the outline area {outline_area:.1f} um^2 exceeds the "
                  f"canonical budget {canonical_area_um2:.1f} um^2 (anchoring "
                  "grew the die instead of freeing a channel)")
    out = {
        "verdict": verdict,
        "caps": caps,
        "die_bbox": tuple(round(v, 4) for v in die),
        "outline_area_um2": round(outline_area, 4),
        "canonical_area_um2": canonical_area_um2,
        "n_caps": len(caps),
        "frame": "placement",
    }
    if reason is not None:
        out["reason"] = reason
    return out


__all__ = [
    "declare_edge_anchor",
    "is_edge_anchor",
    "edge_anchor_members",
    "edge_anchor_obligation",
]
