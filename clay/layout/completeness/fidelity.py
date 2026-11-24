"""The FIDELITY ORACLE (Oracle 2 -- paradigm VALIDATION, geometric half).

WHAT THIS IS (and is NOT)
=========================
This is the *geometric* half of validating that our declared primitives captured
the REFERENCE design's layout *intent*. It asks a single question:

    Does our generated GDS reproduce the REFERENCE design's layout PARADIGM?

It produces a FIDELITY SCORE in ``[0, 1]`` by extracting CLASS-LEVEL paradigm
invariants from BOTH the reference GDS and ours and diffing them.

This is ORTHOGONAL to two neighbours:

  * the existing ``incompleteness_score`` (``invariance.py``) asks "is our layout
    *uniquely determined* by declared intent" (self-consistency / placer-invariance).
    FIDELITY instead asks "is our layout *like the reference*".
  * LVS (delegated to KLayout) is the *topological* half -- exact device/net
    correspondence against the reference netlist. We do NOT reimplement LVS or
    graph isomorphism here. We do NOT attempt instance-by-instance device matching
    (the two GDSs use different cell names *and different PDKs*, so exact device
    matching is out of scope and would be meaningless).

EVERYTHING here is RELATIVE / STRUCTURAL. There is NO absolute-coordinate compare:
the reference is external_pdk (133.6 x 80.1 um) and ours is gf180mcu (~101 x 105 um).
Every sub-score is a normalized class-level relation (counts, orderings, equality
classes, mirror flags, ratios) so the two layouts can be compared despite different
PDKs and different absolute sizes.

DEVICE CLASS IS IDENTIFIED GEOMETRICALLY (no cell names, no labels):
  * ``nwell`` layer present over an active region  -> PMOS region
  * active region with NO overlying nwell          -> NMOS region
  * MIM / cap-marker layer                          -> capacitor
  * precision-resistor marker layer                 -> resistor
  * a poly stripe overlapping active                -> a transistor FINGER

THE PARADIGM INVARIANTS (each a normalized [0,1] sub-score; weighted total):
  1. TIER STRUCTURE  -- number of horizontal device-class bands + their vertical
                        ORDER by class (e.g. NMOS-bottom / cap-middle / PMOS-top).
  2. CAP PARADIGM    -- number of caps + EQUAL-footprint + side-by-side (same band).
  3. MULTI-FINGER    -- are the MOS multi-finger (poly-stripe count per band > 1)
                        vs single-stick.
  4. SYMMETRY        -- is there an approximate vertical mirror axis for the matched
                        devices (reflective symmetry of the relevant band).
  5. ASPECT/COMPACT  -- cell aspect ratio (h/w) vs reference, + area ratio
                        (lightly weighted: a coarse compactness signal, not a gate).

The WEIGHTS are documented in :data:`DEFAULT_WEIGHTS` and are TUNABLE (pass a
``weights`` dict to :func:`fidelity_score`).

This module STANDS ALONE: it does not depend on ``gds_invariants`` (a sibling owns
that) -- we have no ``device_labels`` for the reference GDS anyway, so class-level
geometric extraction is the only thing that works on both inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:  # klayout.db is the production GDS reader (env has it)
    import klayout.db as kdb
except Exception:  # pragma: no cover - import guard only
    kdb = None  # type: ignore


# --------------------------------------------------------------------------- #
# PDK layer-role tables.
#
# We need to know, per PDK, which GDS (layer, datatype) carries which ROLE so we
# can classify device shapes geometrically. These are the two PDKs this oracle
# must handle: the external_pdk reference and our gf180mcu output. Each entry maps a
# ROLE -> list of (layer, datatype) candidates (a list because a PDK may draw a
# role on more than one layer / a "cap" may be marked by several co-located
# layers; we union them).
#
# The tables are intentionally SMALL and EXPLICIT. PDK auto-detection (below)
# picks the right table by probing which signature layers are present, so the
# extractor needs no caller-supplied PDK name.
# --------------------------------------------------------------------------- #
LayerKey = Tuple[int, int]

# external_pdk (the external_drc-signed reference). Verified empirically against
# refCircuit/.../layout.gds: nwell=L3, active/OD=L13, poly=L17, MIM cap marker=
# L131 (the clean 2x 32.8um equal-cap marker), precision resistor marker=L54.

# gf180mcu (our output). Verified against the LayoutTech Gf180mcuLayerMap +
# emitted GDS: nwell=L21, active/COMP=L22, poly=L30, MIM cap marker=L117(dt5)
# / fusetop=L75. gf180 has no precision-resistor marker emitted here.
GF180_ROLES: Dict[str, List[LayerKey]] = {
    "nwell": [(21, 0)],
    "active": [(22, 0)],
    "poly": [(30, 0)],
    "cap": [(117, 5), (75, 0)],
    "resistor": [],
}

# Signature layers used to auto-detect the PDK (presence of any => that table).
_PDK_SIGNATURES: List[Tuple[str, Dict[str, List[LayerKey]], List[LayerKey]]] = [
    ("gf180mcu", GF180_ROLES, [(22, 0), (21, 0)]),
]


# --------------------------------------------------------------------------- #
# Data structures.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Band:
    """One horizontal device-class band (a 'tier')."""

    cls: str  # "nmos" | "pmos" | "cap" | "resistor"
    y0: float  # band bottom (um)
    y1: float  # band top (um)
    n_shapes: int  # number of device shapes assigned to the band
    fingers: int  # poly-over-active finger columns (MOS only; 0 otherwise)
    symmetric: bool  # reflective symmetry about cell vertical mid-axis

    @property
    def yc(self) -> float:
        return 0.5 * (self.y0 + self.y1)


@dataclass(frozen=True)
class ParadigmFeatures:
    """Class-level paradigm features extracted from ONE GDS. PDK-agnostic."""

    pdk: str
    width_um: float
    height_um: float
    bands: Tuple[Band, ...]  # ordered bottom -> top
    # cap paradigm
    n_caps: int
    caps_equal: bool  # all caps share footprint within tolerance
    caps_side_by_side: bool  # caps co-located in one y-band
    cap_footprint_um: Optional[Tuple[float, float]]  # (w, h) of a representative cap
    # mos
    mos_multifinger: bool  # every MOS band has > 1 finger
    has_diff_symmetry: bool  # at least one MOS band is reflectively symmetric

    @property
    def n_tiers(self) -> int:
        return len(self.bands)

    @property
    def tier_class_order(self) -> Tuple[str, ...]:
        """Vertical order of device classes, bottom -> top (collapsing repeats)."""
        out: List[str] = []
        for b in self.bands:
            if not out or out[-1] != b.cls:
                out.append(b.cls)
        return tuple(out)

    @property
    def aspect(self) -> float:
        """Cell aspect ratio h/w."""
        return self.height_um / self.width_um if self.width_um else 0.0

    @property
    def area_um2(self) -> float:
        return self.width_um * self.height_um


@dataclass(frozen=True)
class FidelityReport:
    """Per-feature sub-scores + weighted total + human-readable breakdown."""

    fidelity_score: float
    sub_scores: Dict[str, float]
    weights: Dict[str, float]
    reference: ParadigmFeatures
    ours: ParadigmFeatures
    notes: Dict[str, str]

    def breakdown(self) -> str:
        lines = [
            "FIDELITY REPORT (reference paradigm <-> our layout)",
            "=" * 56,
            f"  reference PDK : {self.reference.pdk}  "
            f"({self.reference.width_um:.1f} x {self.reference.height_um:.1f} um, "
            f"aspect {self.reference.aspect:.2f})",
            f"  our       PDK : {self.ours.pdk}  "
            f"({self.ours.width_um:.1f} x {self.ours.height_um:.1f} um, "
            f"aspect {self.ours.aspect:.2f})",
            "",
            f"  reference tiers : {self.reference.tier_class_order}",
            f"  our       tiers : {self.ours.tier_class_order}",
            f"  reference caps  : n={self.reference.n_caps} "
            f"equal={self.reference.caps_equal} sbs={self.reference.caps_side_by_side}",
            f"  our       caps  : n={self.ours.n_caps} "
            f"equal={self.ours.caps_equal} sbs={self.ours.caps_side_by_side}",
            f"  reference mos   : multifinger={self.reference.mos_multifinger} "
            f"diff_sym={self.reference.has_diff_symmetry}",
            f"  our       mos   : multifinger={self.ours.mos_multifinger} "
            f"diff_sym={self.ours.has_diff_symmetry}",
            "",
            "  sub-scores (weight x score):",
        ]
        for k in self.weights:
            w = self.weights[k]
            s = self.sub_scores.get(k, 0.0)
            note = self.notes.get(k, "")
            lines.append(f"    {k:14s} w={w:.2f}  score={s:.3f}  {note}")
        lines.append("")
        lines.append(f"  FIDELITY SCORE = {self.fidelity_score:.3f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Default weighting. TUNABLE -- pass `weights=` to fidelity_score().
# Rationale:
#   tier_structure dominates (the defining macro-paradigm: 3 class bands in order).
#   cap_paradigm + multifinger are the next strongest design-intent fingerprints.
#   symmetry confirms the diff-pair intent. aspect is LIGHT (coarse compactness,
#   explicitly NOT a hard gate -- the two PDKs differ in absolute size).
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS: Dict[str, float] = {
    "tier_structure": 0.35,
    "cap_paradigm": 0.25,
    "multifinger": 0.15,
    "symmetry": 0.15,
    "aspect": 0.10,
}

# Geometric tolerances (um).
_BAND_GAP_UM = 6.0  # y-gap that separates two device bands
_EQUAL_TOL = 0.10  # 10% relative tolerance for "equal footprint"
_SYM_TOL_UM = 2.0  # mirror-residual tolerance for reflective symmetry
_SYM_FRACTION = 0.80  # fraction of band x-centers that must have a mirror partner
_FINGER_X_TOL = 0.25  # x-merge tolerance for distinct finger columns
_MIN_DEV_AREA = 1.0  # um^2 -- active fragments below this are taps/guard-ring noise
_CAP_DEDUPE_TOL = 1.0  # um -- one physical cap = one CENTER cluster (a MIM plate is
#                         often marked by >1 concentric layer: fusetop + cap_mk,
#                         differing only by a small enclosure on each edge)


def _dedupe_cap_boxes(
    boxes: List[Tuple[float, float, float, float]]
) -> List[Tuple[float, float, float, float]]:
    """Collapse cap boxes that mark the SAME physical plate. A MIM cap is often
    drawn as concentric markers on co-located layers (e.g. gf180 fusetop + cap_mk,
    differing by a per-edge enclosure). We cluster by CENTER (not corners): two
    boxes whose centers agree within :data:`_CAP_DEDUPE_TOL` are ONE cap. We keep
    the LARGEST-footprint box of each cluster as the representative plate."""
    reps: List[Tuple[float, float, float, float]] = []
    for b in sorted(boxes, key=lambda x: -((x[2] - x[0]) * (x[3] - x[1]))):
        bcx, bcy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        dup = False
        for k in reps:
            kcx, kcy = 0.5 * (k[0] + k[2]), 0.5 * (k[1] + k[3])
            if abs(bcx - kcx) <= _CAP_DEDUPE_TOL and abs(bcy - kcy) <= _CAP_DEDUPE_TOL:
                dup = True
                break
        if not dup:
            reps.append(b)
    return reps


# --------------------------------------------------------------------------- #
# GDS access helpers.
# --------------------------------------------------------------------------- #
def _require_kdb():
    if kdb is None:  # pragma: no cover
        raise RuntimeError(
            "klayout.db is required for fidelity extraction but is not importable"
        )


def _detect_pdk(layout) -> Tuple[str, Dict[str, List[LayerKey]]]:
    """Pick the PDK role-table by probing which signature layers are present."""
    present = set()
    for li in layout.layer_indexes():
        info = layout.get_info(li)
        present.add((info.layer, info.datatype))
    for name, roles, sig in _PDK_SIGNATURES:
        if all(k in present for k in sig):
            return name, roles
    # Unknown PDK: fall back to gf180 role table (closest to "ours"), tagged.
    return "unknown", GF180_ROLES


def _region(layout, top, keys: Sequence[LayerKey], *, merge: bool = True):
    """Union region of all shapes on the given (layer, datatype) keys.

    ``merge=True`` (default) coalesces touching shapes -- right for class masks.
    ``merge=False`` preserves per-shape identity -- right for caps, where two
    equal MIM plates ABUT and must NOT be fused into one footprint.
    """
    reg = kdb.Region()
    for (lay, dt) in keys:
        li = layout.find_layer(lay, dt)
        if li is None:
            continue
        reg += kdb.Region(top.begin_shapes_rec(li))
    if merge:
        reg.merge()
    return reg


def _raw_boxes_um(layout, top, keys: Sequence[LayerKey], dbu: float):
    """Per-SHAPE bbox in um (no cross-shape merge); dedupe exact duplicates."""
    seen = set()
    out = []
    for (lay, dt) in keys:
        li = layout.find_layer(lay, dt)
        if li is None:
            continue
        for s in top.begin_shapes_rec(li):
            b = s.shape().bbox().transformed(s.itrans())
            key = (b.left, b.bottom, b.right, b.top)
            if key in seen:
                continue
            seen.add(key)
            out.append((b.left * dbu, b.bottom * dbu, b.right * dbu, b.top * dbu))
    return out


def _dominant_component(region, dbu: float):
    """Return (y0, y1, x0, x1, area_um2) of the LARGEST-area merged component, or
    None when the region is empty. Used to find the dominant nwell (PMOS) block,
    ignoring narrow full-height well-straps / guard rings on the edges."""
    best = None
    best_a = 0.0
    for p in region.merged().each():
        a = p.area() * dbu * dbu
        if a > best_a:
            best_a = a
            b = p.bbox()
            best = (
                b.bottom * dbu,
                b.top * dbu,
                b.left * dbu,
                b.right * dbu,
                a,
            )
    return best


def _polys_um(region, dbu: float) -> List[Tuple[float, float, float, float]]:
    """Per-polygon bbox in um: (x0, y0, x1, y1)."""
    out = []
    for p in region.each():
        b = p.bbox()
        out.append((b.left * dbu, b.bottom * dbu, b.right * dbu, b.top * dbu))
    return out


def _cluster_y(centers: List[float], gap: float) -> List[Tuple[float, float, int]]:
    """Cluster 1-D y-values into bands separated by >= `gap`. Returns
    (y0, y1, count) per band, bottom->top."""
    if not centers:
        return []
    sc = sorted(centers)
    bands: List[List[float]] = [[sc[0]]]
    for y in sc[1:]:
        if y - bands[-1][-1] > gap:
            bands.append([y])
        else:
            bands[-1].append(y)
    return [(b[0], b[-1], len(b)) for b in bands]


# --------------------------------------------------------------------------- #
# Finger detection: count distinct poly-over-active x-columns inside a y-band.
# This is the PDK-agnostic finger proxy: a transistor finger = a poly stripe
# crossing the active region. Works for both heavily-fractured external_pdk poly and
# clean gf180 poly because it counts the INTERSECTION geometry, not raw poly.
# --------------------------------------------------------------------------- #
def _finger_columns(
    layout, top, poly_keys, active_keys, y0: float, y1: float, dbu: float
) -> int:
    poly = _region(layout, top, poly_keys)
    act = _region(layout, top, active_keys)
    if poly.is_empty() or act.is_empty():
        return 0
    pad = 3.0  # widen band slightly so edge gates are included
    clip = kdb.Region(
        kdb.Box(
            int(-1e5),
            int((y0 - pad) / dbu),
            int(1e5),
            int((y1 + pad) / dbu),
        )
    )
    gate = (poly & act & clip).merged()
    if gate.is_empty():
        return 0
    xs = sorted(
        0.5 * (p.bbox().left + p.bbox().right) * dbu for p in gate.each()
    )
    cols = 0
    last = None
    for x in xs:
        if last is None or (x - last) > _FINGER_X_TOL:
            cols += 1
        last = x
    return cols


# --------------------------------------------------------------------------- #
# Reflective symmetry: does the set of device-shape centroids in a band mirror
# about the cell's vertical mid-axis? Approximate / structural only.
# --------------------------------------------------------------------------- #
def _band_is_symmetric(
    boxes: List[Tuple[float, float, float, float]], axis_x: Optional[float] = None
) -> bool:
    """Is the band reflectively symmetric in x?

    TRANSLATION-INVARIANT: the mirror axis is the band's OWN device-cluster
    centroid (mean x-center), NOT the cell bbox center -- the cell bbox is shifted
    by routing / labels and is not the device symmetry axis. We test whether the
    sorted x-center multiset mirrors about its own mean (outermost pairs equidistant
    from the centroid, an odd device sitting on it). ``axis_x`` is accepted but
    ignored (kept for call-site compatibility)."""
    xcs = sorted(0.5 * (b[0] + b[2]) for b in boxes)
    n = len(xcs)
    if n == 0:
        return False
    if n == 1:
        return True  # a lone device is trivially symmetric about itself
    centroid = sum(xcs) / n
    # A band is symmetric if MOST x-centers have a mirror partner about the
    # cluster centroid. We allow a small fraction of unmatched centers (dummies /
    # guard-ring fragments) so a real, slightly-fractured layout is not penalised
    # for a couple of asymmetric satellites. The mirror image of x is
    # 2*centroid - x; we look for a center within _SYM_TOL_UM of that image.
    matched = 0
    for x in xcs:
        mirror = 2.0 * centroid - x
        if any(abs(other - mirror) <= _SYM_TOL_UM for other in xcs):
            matched += 1
    return (matched / n) >= _SYM_FRACTION


# --------------------------------------------------------------------------- #
# THE EXTRACTOR.
# --------------------------------------------------------------------------- #
def extract_paradigm_features(gds_path: str) -> ParadigmFeatures:
    """Read a GDS and return CLASS-LEVEL paradigm features. Works on BOTH the
    external_pdk reference and our gf180mcu output -- device class is identified
    GEOMETRICALLY (nwell -> PMOS, bare active -> NMOS, MIM marker -> cap)."""
    _require_kdb()
    layout = kdb.Layout()
    layout.read(str(gds_path))
    dbu = layout.dbu
    top = layout.top_cell()
    bb = top.bbox()
    width_um = bb.width() * dbu
    height_um = bb.height() * dbu
    axis_x = 0.5 * (bb.left + bb.right) * dbu

    pdk, roles = _detect_pdk(layout)

    nwell = _region(layout, top, roles.get("nwell", []))
    active = _region(layout, top, roles.get("active", []))
    res_reg = _region(layout, top, roles.get("resistor", []))

    # =====================================================================
    # CLASS-LEVEL MACRO TIERS.
    #
    # We extract the *macro* device-class paradigm, not every active fragment.
    # A real signed analog amp interleaves taps / guard-rings / well-straps with
    # the device rows; per-fragment y-clustering shatters into noise bands. The
    # class-level paradigm is defined by the DOMINANT class markers:
    #
    #   * the DOMINANT nwell component  -> the PMOS tier zone (ignores narrow
    #     full-height well-straps / guard rings on the cell edges).
    #   * the cap marker layer          -> the cap tier zone.
    #   * active OUTSIDE the dominant nwell, below the cap zone -> the NMOS tier.
    #
    # Each zone becomes ONE band, ordered by y. This yields the 3-tier macro
    # paradigm (NMOS-bottom / cap-middle / PMOS-top) the reference embodies, and
    # the same extraction applied to ours produces a directly comparable result.
    # =====================================================================
    bands: List[Band] = []

    # ---- caps (read RAW: abutting equal plates must NOT be fused) -----------
    cap_boxes = _raw_boxes_um(layout, top, roles.get("cap", []), dbu)
    # Some PDKs draw a cap on >1 co-located layer (e.g. gf180 fusetop + cap_mk).
    # Collapse near-duplicate boxes (same footprint within tolerance & same y)
    # so a 2-plate cap array does not count as 4.
    cap_boxes = _dedupe_cap_boxes(cap_boxes)
    n_caps = len(cap_boxes)
    cap_y0 = min((b[1] for b in cap_boxes), default=None)
    cap_y1 = max((b[3] for b in cap_boxes), default=None)

    # ---- PMOS tier: dominant nwell block -----------------------------------
    pmos_zone = _dominant_component(nwell, dbu) if not nwell.is_empty() else None

    # ---- NMOS tier: active outside the dominant nwell, below the cap zone ---
    if pmos_zone is not None:
        pz_lo, pz_hi = pmos_zone[0], pmos_zone[1]
        pmos_region = kdb.Region(
            kdb.Box(int(-1e6), int(pz_lo / dbu), int(1e6), int(pz_hi / dbu))
        )
        nmos_active = active - pmos_region
    else:
        pz_lo = pz_hi = None
        nmos_active = active
    # restrict NMOS to area-significant active, take its y-extent as one band
    nmos_boxes = [b for b in _polys_um(nmos_active, dbu) if (b[2] - b[0]) * (b[3] - b[1]) >= _MIN_DEV_AREA]
    # drop active that sits at/above the cap zone (those belong to upper tiers)
    if cap_y0 is not None:
        nmos_boxes = [b for b in nmos_boxes if 0.5 * (b[1] + b[3]) < cap_y0]

    # Build NMOS band
    if nmos_boxes:
        n_y0 = min(b[1] for b in nmos_boxes)
        n_y1 = max(b[3] for b in nmos_boxes)
        n_fing = _finger_columns(layout, top, roles["poly"], roles["active"], n_y0, n_y1, dbu)
        bands.append(
            Band(
                cls="nmos",
                y0=n_y0,
                y1=n_y1,
                n_shapes=len(nmos_boxes),
                fingers=n_fing,
                symmetric=_band_is_symmetric(nmos_boxes, axis_x),
            )
        )

    # Build resistor band (reference middle tier, below/within cap zone)
    res_boxes = _polys_um(res_reg, dbu)
    if res_boxes:
        bands.append(
            Band(
                cls="resistor",
                y0=min(b[1] for b in res_boxes),
                y1=max(b[3] for b in res_boxes),
                n_shapes=len(res_boxes),
                fingers=0,
                symmetric=_band_is_symmetric(res_boxes, axis_x),
            )
        )

    # Build cap band
    if n_caps >= 1:
        bands.append(
            Band(
                cls="cap",
                y0=cap_y0,
                y1=cap_y1,
                n_shapes=n_caps,
                fingers=0,
                symmetric=_band_is_symmetric(cap_boxes, axis_x),
            )
        )

    # Build PMOS band
    if pmos_zone is not None:
        p_fing = _finger_columns(layout, top, roles["poly"], roles["active"], pz_lo, pz_hi, dbu)
        # PMOS device shapes = active intersect the dominant nwell zone
        pmos_active = active & kdb.Region(
            kdb.Box(int(-1e6), int(pz_lo / dbu), int(1e6), int(pz_hi / dbu))
        )
        pmos_boxes = [b for b in _polys_um(pmos_active, dbu) if (b[2] - b[0]) * (b[3] - b[1]) >= _MIN_DEV_AREA]
        bands.append(
            Band(
                cls="pmos",
                y0=pz_lo,
                y1=pz_hi,
                n_shapes=len(pmos_boxes),
                fingers=p_fing,
                symmetric=_band_is_symmetric(pmos_boxes, axis_x),
            )
        )

    # ---- cap paradigm sub-features -----------------------------------------
    cap_footprint: Optional[Tuple[float, float]] = None
    caps_equal = False
    caps_side_by_side = False
    if n_caps >= 1:
        ws = [b[2] - b[0] for b in cap_boxes]
        hs = [b[3] - b[1] for b in cap_boxes]
        cap_footprint = (ws[0], hs[0])
        if n_caps >= 2:
            wmax, wmin = max(ws), min(ws)
            hmax, hmin = max(hs), min(hs)
            caps_equal = (
                (wmax - wmin) / wmax <= _EQUAL_TOL
                and (hmax - hmin) / hmax <= _EQUAL_TOL
            )
            # side-by-side = all cap y-centers fall in one y-band
            ycs = [0.5 * (b[1] + b[3]) for b in cap_boxes]
            caps_side_by_side = len(_cluster_y(ycs, _BAND_GAP_UM)) == 1
        else:
            caps_equal = True  # a single cap is trivially "equal" to itself
            caps_side_by_side = True

    bands.sort(key=lambda b: b.yc)  # bottom -> top

    mos_bands = [b for b in bands if b.cls in ("nmos", "pmos")]
    mos_multifinger = bool(mos_bands) and all(b.fingers > 1 for b in mos_bands)
    has_diff_symmetry = any(b.symmetric for b in mos_bands)

    return ParadigmFeatures(
        pdk=pdk,
        width_um=width_um,
        height_um=height_um,
        bands=tuple(bands),
        n_caps=n_caps,
        caps_equal=caps_equal,
        caps_side_by_side=caps_side_by_side,
        cap_footprint_um=cap_footprint,
        mos_multifinger=mos_multifinger,
        has_diff_symmetry=has_diff_symmetry,
    )


# --------------------------------------------------------------------------- #
# Sub-scorers. Each returns (score in [0,1], human note). All RELATIVE.
# --------------------------------------------------------------------------- #
def _score_tier_structure(
    ref: ParadigmFeatures, ours: ParadigmFeatures
) -> Tuple[float, str]:
    ro = ref.tier_class_order
    oo = ours.tier_class_order
    # count-of-tiers component (how many distinct class bands)
    nt_ref = len(ro)
    nt_ours = len(oo)
    count_score = 1.0 - abs(nt_ref - nt_ours) / max(nt_ref, nt_ours, 1)
    # order component: longest-common-subsequence over the class sequences,
    # normalized by the reference length. This rewards correct vertical ORDER
    # of classes (NMOS-bottom / cap / PMOS-top) without absolute coordinates.
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                dp[i][j] = (
                    dp[i + 1][j + 1] + 1
                    if a[i] == b[j]
                    else max(dp[i + 1][j], dp[i][j + 1])
                )
        return dp[0][0]

    order_score = lcs(ro, oo) / max(len(ro), 1)
    score = 0.5 * count_score + 0.5 * order_score
    return score, f"ref{ro} vs ours{oo} (count {count_score:.2f}, order {order_score:.2f})"


def _score_cap_paradigm(
    ref: ParadigmFeatures, ours: ParadigmFeatures
) -> Tuple[float, str]:
    # count match (relative), equal flag, side-by-side flag -- three equal thirds
    if ref.n_caps == 0:
        # reference has no caps -> reward matching absence
        return (1.0 if ours.n_caps == 0 else 0.0), "ref has no caps"
    count_score = 1.0 - abs(ref.n_caps - ours.n_caps) / max(ref.n_caps, ours.n_caps, 1)
    equal_score = 1.0 if ref.caps_equal == ours.caps_equal else 0.0
    sbs_score = 1.0 if ref.caps_side_by_side == ours.caps_side_by_side else 0.0
    score = (count_score + equal_score + sbs_score) / 3.0
    return score, (
        f"n ref={ref.n_caps} ours={ours.n_caps}; "
        f"equal ref={ref.caps_equal} ours={ours.caps_equal}; "
        f"sbs ref={ref.caps_side_by_side} ours={ours.caps_side_by_side}"
    )


def _score_multifinger(
    ref: ParadigmFeatures, ours: ParadigmFeatures
) -> Tuple[float, str]:
    score = 1.0 if ref.mos_multifinger == ours.mos_multifinger else 0.0
    return score, f"ref={ref.mos_multifinger} ours={ours.mos_multifinger}"


def _score_symmetry(
    ref: ParadigmFeatures, ours: ParadigmFeatures
) -> Tuple[float, str]:
    score = 1.0 if ref.has_diff_symmetry == ours.has_diff_symmetry else 0.0
    return score, f"ref={ref.has_diff_symmetry} ours={ours.has_diff_symmetry}"


def _score_aspect(
    ref: ParadigmFeatures, ours: ParadigmFeatures
) -> Tuple[float, str]:
    # aspect ratio similarity (h/w), plus a light area-ratio report (folded in).
    a_ref = ref.aspect
    a_ours = ours.aspect
    if a_ref <= 0 or a_ours <= 0:
        return 0.0, "degenerate aspect"
    # symmetric relative closeness: 1 when equal, -> 0 as they diverge.
    aspect_score = min(a_ref, a_ours) / max(a_ref, a_ours)
    area_ratio = (
        ours.area_um2 / ref.area_um2 if ref.area_um2 else 0.0
    )
    return aspect_score, (
        f"aspect ref={a_ref:.2f} ours={a_ours:.2f} (sim {aspect_score:.2f}); "
        f"area ratio ours/ref={area_ratio:.2f}"
    )


def fidelity_score(
    reference_gds: str,
    our_gds: str,
    weights: Optional[Dict[str, float]] = None,
) -> FidelityReport:
    """Extract paradigm features from BOTH GDSs and return a weighted FidelityReport.

    Each sub-score is RELATIVE / structural (no absolute-coordinate compare). The
    ``weights`` default to :data:`DEFAULT_WEIGHTS` and are TUNABLE.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    ref = extract_paradigm_features(reference_gds)
    ours = extract_paradigm_features(our_gds)

    scorers = {
        "tier_structure": _score_tier_structure,
        "cap_paradigm": _score_cap_paradigm,
        "multifinger": _score_multifinger,
        "symmetry": _score_symmetry,
        "aspect": _score_aspect,
    }
    sub_scores: Dict[str, float] = {}
    notes: Dict[str, str] = {}
    for key, fn in scorers.items():
        s, note = fn(ref, ours)
        sub_scores[key] = max(0.0, min(1.0, s))
        notes[key] = note

    total_w = sum(w.get(k, 0.0) for k in scorers) or 1.0
    total = sum(w.get(k, 0.0) * sub_scores[k] for k in scorers) / total_w

    return FidelityReport(
        fidelity_score=total,
        sub_scores=sub_scores,
        weights={k: w.get(k, 0.0) for k in scorers},
        reference=ref,
        ours=ours,
        notes=notes,
    )
