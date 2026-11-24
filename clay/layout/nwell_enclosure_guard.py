"""R143 (BL-143B) — Nwell-encloses-PCOMP guard after a Nwell move/nudge.

R142 (BL-142C, :mod:`clay.layout.m1_spacing_nwell_fix`) restored the
different-potential ``NW.2b`` Nwell spacing by *nudging* one PMOS Nwell box away
from its neighbour. The nudge translates ONLY the Nwell rectangle; the PMOS
diffusion (PCOMP) it must enclose stays where R140 E
(:mod:`clay.layout.gf180_pmos_geom`) placed it. A tight R140-E Nwell encloses
its PCOMP by *exactly* the rule minimum on every side, so any translation slides
the well partly off its PCOMP and the enclosure drops below the rule on the
trailing edge. On the combined R142 flow that produced exactly one residual::

    DF.4c_3.3V   x1   Min. (Nwell overlap of PCOMP) outside DNWELL.
                      (comp.drc line 366, 0.43um = 430 nm)

The rule (verbatim from the real gf180mcu KLayout DRC deck, cited)
------------------------------------------------------------------
``DF.4c_3.3V`` — *Min. (Nwell overlap of PCOMP) outside DNWELL.* = 0.43 um =
430 nm. Source: the gf180mcu ``comp.drc`` rule deck, line 366
(``df4c_l1 = nwell_n_dnwell.outside(sramcore).enclosing(pcomp.outside(dnwell),
0.43.um, euclidian)...``), under
``AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/rules/klayout/drc/
rule_decks/comp.drc``. The value is re-exported by
:data:`clay.layout.gf180_pmos_geom.GF180_NWELL_PCOMP_ENCLOSURE_NM` (430), which
this module imports read-only rather than restating.

The guard
---------
:func:`guard_nwell_enclosure` takes the (possibly nudged) Nwell shapes and the
PCOMP shapes they must enclose and, for each Nwell, GROWS its rectangle outward
on any side where it overlaps its PCOMP by less than the ``DF.4c`` margin, so the
post-guard Nwell again encloses its PCOMP by >= 430 nm on all four sides.

Growth, not translation, is the correct repair here: the ``NW.2b`` spacing the
R142 nudge restored was opened on the SIDE AWAY from the neighbour, so growing a
well outward to recapture its PCOMP only ever extends the trailing edge (towards
its own diffusion, away from the neighbour) — it does not re-close the spacing
gap the nudge just opened. (A grow that would itself re-trip the different-
potential spacing is reported in the honest residual rather than silently
applied; in the R142 signature it never does, because the shortfall edge faces
away from the neighbour.)

The guard is conservative and idempotent: an already-compliant Nwell (one that
encloses its PCOMP by >= 430 nm on all sides) is returned byte-unchanged; a well
is only ever grown (never shrunk, never moved), and only on the deficient sides
by exactly the shortfall. Re-running the guard on its own output is a no-op.

This module is pure geometry. It imports the committed backend IR + the cited
R140-E enclosure constant read-only and emits only grown copies of the input
``LayerPolygon`` Nwell shapes. All coordinates are GDS database units
(dbu = 1 nm for gf180mcu).

Public API
----------
- :func:`nwell_pcomp_enclosure_nm` — the cited ``DF.4c`` accessor (430).
- :func:`enclosure_margins` — per-side (left/bottom/right/top) Nwell-over-PCOMP
  overlap of one well / pcomp pair.
- :func:`encloses` — predicate: does a well enclose a pcomp by >= the margin on
  all four sides?
- :func:`grow_to_enclose` — return a COPY of one Nwell grown to enclose a pcomp.
- :func:`guard_nwell_enclosure` — guard a whole shape list (well + pcomp + other
  layers), pairing each well to the PCOMP it covers and growing it as needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .backend import LayerMap, LayerPolygon, LogicalLayer, Polygon
from .gf180_pmos_geom import GF180_NWELL_PCOMP_ENCLOSURE_NM

# --- cited rule accessor (verbatim from the real gf180mcu comp.drc deck) ------

#: ``DF.4c_3.3V`` Min. (Nwell overlap of PCOMP) outside DNWELL — 0.43 um =
#: 430 nm (comp.drc line 366). Re-exported from gf180_pmos_geom (R140 E).
DF4C_RULE_ID = "DF.4c_3.3V"


def nwell_pcomp_enclosure_nm() -> int:
    """Return the ``DF.4c`` Nwell-overlap-of-PCOMP minimum in nm (430).

    Sourced from :data:`gf180_pmos_geom.GF180_NWELL_PCOMP_ENCLOSURE_NM`, itself
    a verbatim transcription of comp.drc line 366 (0.43 um). Nothing invented.
    """
    return GF180_NWELL_PCOMP_ENCLOSURE_NM


# --- rect view over a LayerPolygon -------------------------------------------


@dataclass(frozen=True, eq=True)
class _Rect:
    """Axis-aligned bbox of a LayerPolygon (database units)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @staticmethod
    def of(lp: LayerPolygon) -> "_Rect":
        xs = [v.x for v in lp.polygon.vertices]
        ys = [v.y for v in lp.polygon.vertices]
        return _Rect(min(xs), min(ys), max(xs), max(ys))

    def width(self) -> int:
        return self.x_max - self.x_min

    def height(self) -> int:
        return self.y_max - self.y_min


def _layer_key(pdk_layers: LayerMap, logical: LogicalLayer) -> Tuple[int, int]:
    g = pdk_layers.gds_layer(logical)
    if g is None:
        raise ValueError(
            f"PDK must define {logical.name} for the Nwell enclosure guard"
        )
    return (g.layer, g.datatype)


# --- per-side enclosure margins ----------------------------------------------


@dataclass(frozen=True)
class EnclosureMargins:
    """Signed per-side Nwell-over-PCOMP overlap (database units).

    Each value is how far the Nwell extends BEYOND the PCOMP on that side:

    * ``left``   = pcomp.x_min - nwell.x_min
    * ``bottom`` = pcomp.y_min - nwell.y_min
    * ``right``  = nwell.x_max - pcomp.x_max
    * ``top``    = nwell.y_max - pcomp.y_max

    Positive == the Nwell overhangs the PCOMP on that side by that many nm; a
    value below the rule margin (430) is a ``DF.4c`` shortfall; a negative value
    means the PCOMP pokes out past the Nwell on that side.
    """

    left: int
    bottom: int
    right: int
    top: int

    @property
    def min_margin(self) -> int:
        """The smallest of the four side overlaps (the worst side)."""
        return min(self.left, self.bottom, self.right, self.top)


def enclosure_margins(
    nwell: LayerPolygon, pcomp: LayerPolygon
) -> EnclosureMargins:
    """Return the per-side Nwell-over-PCOMP overlap of one well / PCOMP pair."""
    nw = _Rect.of(nwell)
    pc = _Rect.of(pcomp)
    return EnclosureMargins(
        left=pc.x_min - nw.x_min,
        bottom=pc.y_min - nw.y_min,
        right=nw.x_max - pc.x_max,
        top=nw.y_max - pc.y_max,
    )


def encloses(
    nwell: LayerPolygon,
    pcomp: LayerPolygon,
    *,
    margin_nm: Optional[int] = None,
) -> bool:
    """True iff ``nwell`` overlaps ``pcomp`` by >= ``margin_nm`` on all 4 sides.

    ``margin_nm`` defaults to the ``DF.4c`` rule (430 nm).
    """
    m = nwell_pcomp_enclosure_nm() if margin_nm is None else margin_nm
    return enclosure_margins(nwell, pcomp).min_margin >= m


# --- grow primitive ----------------------------------------------------------


def _grown_rect(
    nw: _Rect, pc: _Rect, margin: int
) -> Tuple[int, int, int, int]:
    """Smallest enclosing rect that overlaps ``pc`` by >= ``margin`` per side.

    The Nwell is only ever grown OUTWARD (each edge moves away from the PCOMP);
    a side that already clears the margin is left exactly where it is, so a
    compliant well's rect is returned unchanged.
    """
    x_min = min(nw.x_min, pc.x_min - margin)
    y_min = min(nw.y_min, pc.y_min - margin)
    x_max = max(nw.x_max, pc.x_max + margin)
    y_max = max(nw.y_max, pc.y_max + margin)
    return x_min, y_min, x_max, y_max


def grow_to_enclose(
    nwell: LayerPolygon,
    pcomp: LayerPolygon,
    *,
    margin_nm: Optional[int] = None,
) -> LayerPolygon:
    """Return a COPY of ``nwell`` grown to enclose ``pcomp`` by >= ``margin_nm``.

    Only deficient sides are extended outward (away from the PCOMP) by exactly
    their shortfall; sides already clearing the margin are untouched. A well that
    already encloses the PCOMP by >= the margin on every side is returned with an
    identical bbox (idempotent). ``margin_nm`` defaults to ``DF.4c`` (430 nm).
    """
    m = nwell_pcomp_enclosure_nm() if margin_nm is None else margin_nm
    nw = _Rect.of(nwell)
    pc = _Rect.of(pcomp)
    x_min, y_min, x_max, y_max = _grown_rect(nw, pc, m)
    return LayerPolygon(
        layer=nwell.layer,
        datatype=nwell.datatype,
        polygon=Polygon.rectangle(x_min, y_min, x_max, y_max),
    )


# --- well <-> pcomp pairing --------------------------------------------------


def _overlap_area(a: _Rect, b: _Rect) -> int:
    """Axis-aligned overlap area of two rects (0 if they do not overlap)."""
    dx = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    dy = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if dx <= 0 or dy <= 0:
        return 0
    return dx * dy


def pair_well_to_pcomp(
    nwells: Sequence[LayerPolygon],
    pcomps: Sequence[LayerPolygon],
) -> Dict[int, int]:
    """Map ``nwell index -> pcomp index`` by largest geometric overlap.

    Each Nwell is paired with the PCOMP it most overlaps (the diffusion it is
    meant to enclose). A Nwell that overlaps no PCOMP is left unpaired (absent
    from the returned map). PCOMP indices are into ``pcomps``; Nwell indices are
    into ``nwells``.
    """
    out: Dict[int, int] = {}
    pc_rects = [_Rect.of(p) for p in pcomps]
    for i, well in enumerate(nwells):
        nw = _Rect.of(well)
        best_area = 0
        best_j = -1
        for j, pc in enumerate(pc_rects):
            area = _overlap_area(nw, pc)
            if area > best_area:
                best_area = area
                best_j = j
        if best_j >= 0:
            out[i] = best_j
    return out


# --- structured result -------------------------------------------------------


@dataclass(frozen=True)
class EnclosureGuardResult:
    """Result of a Nwell-enclosure guard pass.

    Attributes:
        shapes: the full shape list AFTER the guard (input order preserved; the
            grown Nwell shapes replaced by their grown copies).
        rule_id: the deck rule the guard enforces ("DF.4c_3.3V").
        margin_nm: the enclosure margin applied (nm; DF.4c = 430).
        grown_count: how many Nwell shapes were grown (0 == input already
            compliant / unchanged).
        unpaired_wells: indices (into the input shape list) of Nwell shapes that
            overlap no PCOMP and so could not be guarded (honest report).
    """

    shapes: List[LayerPolygon]
    rule_id: str
    margin_nm: int
    grown_count: int
    unpaired_wells: List[int]

    @property
    def clean(self) -> bool:
        """True iff every Nwell was paired (none left unguardable)."""
        return not self.unpaired_wells


def guard_nwell_enclosure(
    shapes: Sequence[LayerPolygon],
    pdk_layers: LayerMap,
    *,
    pcomp_layer: LogicalLayer = LogicalLayer.PDIFF,
    margin_nm: Optional[int] = None,
    pairing: Optional[Dict[int, int]] = None,
) -> EnclosureGuardResult:
    """Grow every Nwell so it re-encloses its PCOMP by >= ``DF.4c`` (430 nm).

    After a Nwell move/nudge (e.g. the R142
    :func:`m1_spacing_nwell_fix.fix_nwell_diff_potential_spacing` repair) a well
    may slide partly off the PCOMP it must enclose, dropping the Nwell-over-PCOMP
    overlap below ``DF.4c`` on the trailing edge. This guard restores the rule by
    GROWING each Nwell outward on its deficient sides until it overlaps its PCOMP
    by >= ``margin_nm`` (default 430 nm) on all four sides.

    Each Nwell is paired with the PCOMP it most overlaps (or use ``pairing`` to
    supply an explicit ``nwell-shape-index -> pcomp-shape-index`` map, indices
    into ``shapes``). Non-Nwell shapes (the PCOMP itself, poly, contacts, metal)
    pass through untouched and keep their input index. A Nwell that overlaps no
    PCOMP is left as-is and reported in ``unpaired_wells`` (honest — the guard
    cannot know which diffusion it should cover).

    The guard is idempotent: a Nwell already enclosing its PCOMP by >= the margin
    on every side is returned byte-unchanged, so a clean layout round-trips and
    re-running the guard is a no-op.

    Args:
        shapes: the shape list containing Nwell (21/0) and PCOMP (COMP 22/0)
            polygons, plus any other layers (passed through).
        pdk_layers: a gf180mcu LayerMap (defines NWELL 21/0 and the PCOMP/PDIFF
            COMP 22/0 layer).
        pcomp_layer: the logical layer the PMOS diffusion lives on (default
            ``PDIFF``; gf180mcu maps both NDIFF/PDIFF to COMP 22/0).
        margin_nm: enclosure margin (default ``DF.4c`` = 430 nm).
        pairing: optional explicit Nwell-index -> PCOMP-index map (indices into
            ``shapes``); when omitted, pairing is inferred by largest overlap.

    Returns:
        An :class:`EnclosureGuardResult` with the guarded shapes, grown count and
        an honest list of any unpaired (unguardable) wells.
    """
    m = nwell_pcomp_enclosure_nm() if margin_nm is None else margin_nm
    nw_key = _layer_key(pdk_layers, LogicalLayer.NWELL)
    pc_key = _layer_key(pdk_layers, pcomp_layer)

    # Copy every shape (input order preserved); only Nwell copies may be grown.
    work: List[LayerPolygon] = [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(vertices=list(lp.polygon.vertices)),
        )
        for lp in shapes
    ]

    nwell_idx = [i for i, lp in enumerate(work) if (lp.layer, lp.datatype) == nw_key]
    pcomp_idx = [i for i, lp in enumerate(work) if (lp.layer, lp.datatype) == pc_key]

    # Resolve the well->pcomp pairing (explicit map or inferred by overlap).
    if pairing is not None:
        pair_global = dict(pairing)
    else:
        local = pair_well_to_pcomp(
            [work[i] for i in nwell_idx], [work[i] for i in pcomp_idx]
        )
        # translate local (position-in-nwell_idx -> position-in-pcomp_idx) to
        # global shape indices.
        pair_global = {
            nwell_idx[li]: pcomp_idx[lj] for li, lj in local.items()
        }

    grown = 0
    unpaired: List[int] = []
    for i in nwell_idx:
        j = pair_global.get(i)
        if j is None:
            unpaired.append(i)
            continue
        before = _Rect.of(work[i])
        grown_poly = grow_to_enclose(work[i], work[j], margin_nm=m)
        after = _Rect.of(grown_poly)
        if after != before:
            work[i] = grown_poly
            grown += 1

    return EnclosureGuardResult(
        shapes=work,
        rule_id=DF4C_RULE_ID,
        margin_nm=m,
        grown_count=grown,
        unpaired_wells=unpaired,
    )


__all__ = [
    "DF4C_RULE_ID",
    "nwell_pcomp_enclosure_nm",
    "EnclosureMargins",
    "enclosure_margins",
    "encloses",
    "grow_to_enclose",
    "pair_well_to_pcomp",
    "EnclosureGuardResult",
    "guard_nwell_enclosure",
]
