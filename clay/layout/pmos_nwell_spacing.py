"""R144 (BL-144B) — placement-level PMOS Nwell different-potential spacing.

Backup repair for the last residual ``NW.2b`` violation when merging the two
PMOS Nwells into one equi-potential well is NOT valid — i.e. the two PMOS sit on
DIFFERENT potential nets, so their wells must stay electrically distinct and
cannot be unioned (R142 C's ``merge_same_net`` path is off the table here).

The structural tension (characterised in the R143 isolation experiment, see the
CHANGELOG R144 path note) is::

    a single PMOS Nwell must enclose its own PCOMP by DF.4c (0.43um = 430 nm),
    but two such tight wells on different potentials must also stay >= NW.2b
    (1.4um = 1400 nm) apart edge-to-edge.

R142 C repairs this at the *shape* level (``fix_nwell_diff_potential_spacing``
nudges one already-built Nwell rectangle away from its neighbour). This module
repairs it at the *placement* level instead: it takes the PMOS DEVICE placements
(each PMOS's center position) and increases the center-to-center spacing so that
each PMOS's OWN Nwell — its COMP bbox grown by the DF.4c enclosure on every side
— keeps a >= NW.2b edge gap from every other DIFFERENT-potential PMOS well.
Moving a device shifts ALL of that device's layers together (COMP, poly,
contacts, Nwell), so intra-device geometry / connectivity is untouched; only the
device's location changes. Same-potential wells are left where they are (they may
legally abut down to the equi-potential ``NW.2a`` = 600 nm, or be merged by a
separate same-net path).

Why a placement widen instead of a per-shape nudge?
---------------------------------------------------
A per-shape nudge (R142 C) moves only the Nwell rectangle, which then no longer
encloses its own PCOMP — fine when the well is the only thing on that net, but it
desynchronises the well from the device it belongs to. Widening the device
center spacing keeps the well rigidly attached to its PCOMP (the DF.4c enclosure
is preserved by construction, because the well is recomputed from the moved COMP
bbox), so the fix never trades a ``NW.2b`` violation for a ``DF.4c`` one. This is
the correct repair when the wells genuinely cannot merge.

Rule values used (transcribed verbatim from the real gf180mcu KLayout DRC deck
via ``clay.layout.gf180_rules`` / ``clay.layout.gf180_pmos_geom``; nothing
invented):

* ``NW.2b_3.3V``  Min. Nwell Space (Outside DNWELL) [Different potential]
      = 1.4 um = 1400 nm   (gf180mcu.drc deck line 871)
* ``DF.4c_3.3V``  Min. (Nwell overlap of PCOMP) outside DNWELL
      = 0.43 um = 430 nm   (comp.drc rule deck line 366) — the enclosure that
      defines each PMOS's own Nwell extent around its PCOMP.

This module imports the committed backend IR, ``gf180_rules`` and
``gf180_pmos_geom`` read-only and emits only translated copies of input shapes /
new placement offsets. All coordinates are GDS database units (dbu = 1 nm for
gf180mcu).

Public API
----------
- :func:`nwell_diff_potential_spacing_nm` — the cited NW.2b accessor (1400).
- :func:`pmos_nwell_enclosure_nm` — the cited DF.4c accessor (430).
- :data:`PmosWellPlacement` — one PMOS's center + COMP half-extents + potential.
- :func:`well_box_for` — the DF.4c-grown Nwell bbox for one placement.
- :func:`diff_potential_violations` — diagnostic: distinct-potential well pairs
  whose edge gap is below NW.2b.
- :func:`widen_pmos_for_nwell_spacing` — THE entry point: take PMOS placements
  (records, an Nwell-shape list, or ``Gf180PmosGeometry`` objects) and return new
  placements / shifted shapes whose distinct-potential wells all clear NW.2b,
  overlap-free, with connectivity unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .backend import LayerPolygon, Point, Polygon
from .gf180_pmos_geom import (
    GF180_NWELL_PCOMP_ENCLOSURE_NM,
    Gf180PmosGeometry,
)
from .gf180_rules import NWELL_SPACE_DIFF_POT


# --- cited rule accessors (verbatim from the real deck) ----------------------

#: ``NW.2b_3.3V`` different-potential Nwell space — 1.4 um = 1400 nm (line 871).
NWELL_DIFF_POT_RULE_ID = NWELL_SPACE_DIFF_POT.rule_id  # "NW.2b_3.3V"
#: ``DF.4c_3.3V`` Nwell-overlap-of-PCOMP enclosure — 0.43 um = 430 nm (line 366).
NWELL_PCOMP_ENCLOSURE_RULE_ID = "DF.4c_3.3V"


def nwell_diff_potential_spacing_nm() -> int:
    """Return ``NW.2b`` different-potential Nwell spacing in nm (1400)."""
    return NWELL_SPACE_DIFF_POT.value_nm


def pmos_nwell_enclosure_nm() -> int:
    """Return ``DF.4c`` Nwell-overlap-of-PCOMP enclosure in nm (430)."""
    return GF180_NWELL_PCOMP_ENCLOSURE_NM


# --- placement record --------------------------------------------------------


@dataclass(frozen=True, eq=True)
class PmosWellPlacement:
    """One PMOS device's placement, sufficient to derive its own Nwell box.

    A PMOS's Nwell is its PCOMP (COMP diffusion) bounding box grown outward by
    the DF.4c enclosure on every side (see ``gf180_pmos_geom``). So the well
    extent is fully determined by the COMP bbox + enclosure; we store the COMP
    bbox (absolute, database units) plus the device name and a potential label.

    Attributes:
        device: device instance name (used to key offsets; never reordered).
        comp_bbox: (x_min, y_min, x_max, y_max) of the device's PCOMP, absolute.
        potential: net/potential label of this PMOS's well. Two placements with
            the same non-None ``potential`` are SAME-potential (their wells may
            legally abut / merge and are NOT widened against each other); a None
            potential is treated as a unique potential (conservative — always
            widened against everything else).
        enclosure_nm: the DF.4c Nwell-overlap-of-PCOMP used (default 430 nm).
    """

    device: str
    comp_bbox: Tuple[int, int, int, int]
    potential: Optional[str] = None
    enclosure_nm: int = GF180_NWELL_PCOMP_ENCLOSURE_NM

    @property
    def comp_center(self) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.comp_bbox
        return ((x0 + x1) // 2, (y0 + y1) // 2)


# --- well-box geometry -------------------------------------------------------


def well_box_for(placement: PmosWellPlacement) -> Tuple[int, int, int, int]:
    """Return the DF.4c-grown Nwell bbox for one PMOS placement.

    The well is the PCOMP bbox extended outward by ``enclosure_nm`` (DF.4c =
    430 nm) on all four sides — exactly the construction in
    ``gf180_pmos_geom.build_gf180_pmos_device``. So a placement's well moves
    rigidly with its COMP: widening the device center spacing recomputes the
    well from the moved COMP, preserving the DF.4c enclosure by construction.
    """
    x0, y0, x1, y1 = placement.comp_bbox
    e = placement.enclosure_nm
    return (x0 - e, y0 - e, x1 + e, y1 + e)


def _gap_xy(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Tuple[int, int]:
    """Signed (dx, dy) edge clearance between two well bboxes.

    Positive == separated by that many nm on that axis; <= 0 == overlapping or
    touching on that axis. Mirrors ``m1_spacing_nwell_fix._gap_xy``.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1)
    dy = max(by0 - ay1, ay0 - by1)
    return dx, dy


def well_edge_gap(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Optional[int]:
    """Edge-to-edge separation of two well bboxes, or None if they touch/overlap.

    Returns ``None`` when the two wells overlap or touch on both axes; otherwise
    the positive minimum clearance (nm): the single positive axis when the
    projections overlap, else the Euclidean corner distance — matching the deck's
    space check (same convention as ``m1_spacing_nwell_fix.edge_to_edge_gap``).
    """
    dx, dy = _gap_xy(a, b)
    if dx <= 0 and dy <= 0:
        return None
    if dx > 0 and dy > 0:
        return int(round((dx * dx + dy * dy) ** 0.5))
    return dx if dx > 0 else dy


def _same_potential(a: PmosWellPlacement, b: PmosWellPlacement) -> bool:
    """True iff a and b are on the SAME (known) potential -> not widened apart."""
    return a.potential is not None and a.potential == b.potential


# --- diagnostic --------------------------------------------------------------


def diff_potential_violations(
    placements: Sequence[PmosWellPlacement],
    rule_nm: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """List ``(i, j, gap)`` distinct-potential well pairs with 0 <= gap < rule.

    A pair is reported when the two placements are NOT same-potential and their
    DF.4c-grown wells either overlap/touch (reported with ``gap == 0``) or sit a
    positive distance below ``rule_nm`` (NW.2b, default 1400). Same-potential
    pairs are skipped — they may legally abut (equi-potential NW.2a = 600) or be
    merged by a separate same-net path.
    """
    r = nwell_diff_potential_spacing_nm() if rule_nm is None else rule_nm
    out: List[Tuple[int, int, int]] = []
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            if _same_potential(placements[i], placements[j]):
                continue
            gap = well_edge_gap(well_box_for(placements[i]), well_box_for(placements[j]))
            g = 0 if gap is None else gap
            if g < r:
                out.append((i, j, g))
    return out


# --- structured result -------------------------------------------------------


@dataclass(frozen=True)
class WellWidenResult:
    """Result of a placement-level Nwell-spacing widen pass.

    Attributes:
        placements: the PMOS placements AFTER widening (input order preserved;
            same device names; only ``comp_bbox`` translated for moved devices).
        offsets: ``{device: (dx, dy)}`` translation applied to each device (the
            shift to apply to ALL of that device's geometry). ``(0, 0)`` for a
            device that did not move.
        rule_id: the deck rule the widen enforces ("NW.2b_3.3V").
        rule_nm: the rule value applied (nm, 1400).
        enclosure_nm: the DF.4c enclosure used to size each well (nm, 430).
        moved_count: how many devices were translated (0 == input already clean).
        residual: ``(i, j, gap)`` distinct-potential pairs STILL below the rule
            after the pass (honest about anything the widen could not resolve).
    """

    placements: List[PmosWellPlacement]
    offsets: Dict[str, Tuple[int, int]]
    rule_id: str
    rule_nm: int
    enclosure_nm: int
    moved_count: int
    residual: List[Tuple[int, int, int]]

    @property
    def clean(self) -> bool:
        return not self.residual


# --- core widen over placement records ---------------------------------------


def _widen_placements(
    placements: Sequence[PmosWellPlacement],
    rule_nm: int,
    max_passes: int,
) -> WellWidenResult:
    """Increase center-to-center spacing so distinct-potential wells clear rule.

    Greedy, conservative, and overlap-free by construction:

    * Each pass finds the tightest distinct-potential violating pair and pushes
      the higher-index device away from the lower-index one along the axis with
      the smaller shortfall (the cheaper repair), by exactly enough to open the
      well edge gap to ``rule_nm`` on that axis.
    * Moving a device translates ALL of its geometry (here, its COMP bbox, from
      which the well is recomputed) — the well stays rigidly enclosing its PCOMP,
      so DF.4c is preserved. Connectivity is untouched (only the location moves).
    * Because every move only ever GROWS the offending gap (push direction is
      away from the fixed device) and opens it to the >= 1400 nm rule, the moved
      wells never end up overlapping the fixed one; the pass also never moves a
      device that already clears every neighbour, so a clean input round-trips.

    The honest ``residual`` reports any distinct-potential pair still under the
    rule after the last pass (e.g. a device pinned between two same-axis
    neighbours that the single-axis greedy widen could not fully separate).
    """
    work: List[PmosWellPlacement] = list(placements)
    accum: Dict[str, Tuple[int, int]] = {p.device: (0, 0) for p in work}

    for _ in range(max_passes):
        viols = diff_potential_violations(work, rule_nm)
        if not viols:
            break
        # Resolve the tightest (smallest current gap) pair first.
        viols.sort(key=lambda t: t[2])
        i, j, _gap = viols[0]
        wi = well_box_for(work[i])
        wj = well_box_for(work[j])
        dx, dy = _gap_xy(wi, wj)
        # Per-axis shortfall to reach the rule gap.
        short_x = rule_nm - dx
        short_y = rule_nm - dy
        # Push device j away from device i along the cheaper axis.
        if short_x <= short_y:
            ci = (wi[0] + wi[2])
            cj = (wj[0] + wj[2])
            direction = 1 if cj >= ci else -1
            mv = (direction * short_x, 0)
        else:
            ci = (wi[1] + wi[3])
            cj = (wj[1] + wj[3])
            direction = 1 if cj >= ci else -1
            mv = (0, direction * short_y)
        x0, y0, x1, y1 = work[j].comp_bbox
        work[j] = replace(
            work[j],
            comp_bbox=(x0 + mv[0], y0 + mv[1], x1 + mv[0], y1 + mv[1]),
        )
        oa, ob = accum[work[j].device]
        accum[work[j].device] = (oa + mv[0], ob + mv[1])

    residual = diff_potential_violations(work, rule_nm)
    moved = sum(1 for off in accum.values() if off != (0, 0))
    return WellWidenResult(
        placements=work,
        offsets=accum,
        rule_id=NWELL_DIFF_POT_RULE_ID,
        rule_nm=rule_nm,
        enclosure_nm=(work[0].enclosure_nm if work else GF180_NWELL_PCOMP_ENCLOSURE_NM),
        moved_count=moved,
        residual=residual,
    )


# --- adapters: shapes / geometry -> placements -------------------------------


def _rect_of(lp: LayerPolygon) -> Tuple[int, int, int, int]:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _placements_from_nwell_shapes(
    nwell_shapes: Sequence[LayerPolygon],
    potential_of: Optional[Dict[int, str]],
    enclosure_nm: int,
) -> List[PmosWellPlacement]:
    """Treat each Nwell shape as a PMOS well; back out its PCOMP bbox.

    A well shape (R140 E) is the PCOMP grown by ``enclosure_nm`` on every side,
    so the implied PCOMP is the well bbox shrunk by ``enclosure_nm``. This lets a
    caller that only has the Nwell rectangles still use the placement-level
    widen (the recomputed well from the implied COMP is byte-identical to the
    input well, so nothing is lost).
    """
    out: List[PmosWellPlacement] = []
    for idx, lp in enumerate(nwell_shapes):
        wx0, wy0, wx1, wy1 = _rect_of(lp)
        comp = (wx0 + enclosure_nm, wy0 + enclosure_nm,
                wx1 - enclosure_nm, wy1 - enclosure_nm)
        pot = None if potential_of is None else potential_of.get(idx)
        out.append(PmosWellPlacement(
            device=f"pmos_{idx}", comp_bbox=comp, potential=pot,
            enclosure_nm=enclosure_nm,
        ))
    return out


def _placements_from_geom(
    geoms: Sequence[Gf180PmosGeometry],
    potential_of: Optional[Dict[int, str]],
) -> List[PmosWellPlacement]:
    """Build placements from real ``Gf180PmosGeometry`` devices (uses comp_bbox)."""
    out: List[PmosWellPlacement] = []
    for idx, g in enumerate(geoms):
        pot = None if potential_of is None else potential_of.get(idx)
        out.append(PmosWellPlacement(
            device=f"pmos_{idx}", comp_bbox=tuple(g.comp_bbox), potential=pot,
            enclosure_nm=g.nwell_enclosure_nm,
        ))
    return out


# --- public API --------------------------------------------------------------

PlacementInput = Union[
    Sequence[PmosWellPlacement],
    Sequence[LayerPolygon],
    Sequence[Gf180PmosGeometry],
]


def widen_pmos_for_nwell_spacing(
    placement_or_geom: PlacementInput,
    *,
    potential_of: Optional[Dict[int, str]] = None,
    rule_nm: Optional[int] = None,
    enclosure_nm: int = GF180_NWELL_PCOMP_ENCLOSURE_NM,
    max_passes: int = 16,
) -> WellWidenResult:
    """Widen PMOS center spacing so distinct-potential Nwells clear ``NW.2b``.

    This is the placement-level backup repair for the last ``NW.2b`` residual on
    the path where the two PMOS wells are on DIFFERENT potentials and therefore
    CANNOT be merged into one equi-potential well. Each PMOS's own Nwell is its
    PCOMP grown by ``DF.4c`` (430 nm) on every side; this function increases the
    center-to-center spacing of the devices so that every DIFFERENT-potential
    well pair keeps a >= ``NW.2b`` (1400 nm) edge gap. Moving a device shifts ALL
    of its geometry together, so the well stays rigidly enclosing its PCOMP
    (DF.4c preserved) and intra-device connectivity is untouched.

    Accepted input forms (auto-detected):

    * a list of :class:`PmosWellPlacement` records (the explicit form: device
      name + PCOMP bbox + potential label);
    * a list of Nwell ``LayerPolygon`` boxes (each well's implied PCOMP is backed
      out by shrinking the well by ``enclosure_nm``);
    * a list of ``Gf180PmosGeometry`` devices (the real R140-E PMOS; uses each
      device's ``comp_bbox`` and ``nwell_enclosure_nm``).

    Args:
        placement_or_geom: the PMOS placements / well shapes / geometries.
        potential_of: for the shape/geometry forms, an INDEX -> potential-label
            map marking which wells share a potential (same-potential wells are
            left alone). For the record form, the records' own ``potential`` field
            is authoritative and this arg is ignored. ``None`` (or a missing
            index) means a unique potential -> conservatively widened against
            everything.
        rule_nm: override the spacing rule (default ``NW.2b`` = 1400 nm).
        enclosure_nm: DF.4c enclosure used when backing PCOMP out of raw Nwell
            shapes (default 430 nm). Ignored for the record / geometry forms,
            which carry their own enclosure.
        max_passes: iteration cap for the greedy widen loop.

    Returns:
        A :class:`WellWidenResult` with the widened placements, the per-device
        offsets to apply, and an honest residual.

    Raises:
        ValueError: if the input list mixes record / shape / geometry kinds, or
            if a ``LayerPolygon`` well is too small to back out a positive PCOMP
            at the given ``enclosure_nm``.
    """
    r = nwell_diff_potential_spacing_nm() if rule_nm is None else rule_nm

    items = list(placement_or_geom)
    if not items:
        return WellWidenResult(
            placements=[], offsets={}, rule_id=NWELL_DIFF_POT_RULE_ID,
            rule_nm=r, enclosure_nm=enclosure_nm, moved_count=0, residual=[],
        )

    kinds = {type(it) for it in items}
    if PmosWellPlacement in kinds:
        if len(kinds) != 1:
            raise ValueError("cannot mix PmosWellPlacement with other input kinds")
        placements = list(items)
    elif Gf180PmosGeometry in kinds:
        if len(kinds) != 1:
            raise ValueError("cannot mix Gf180PmosGeometry with other input kinds")
        placements = _placements_from_geom(items, potential_of)
    elif LayerPolygon in kinds:
        if len(kinds) != 1:
            raise ValueError("cannot mix LayerPolygon with other input kinds")
        for lp in items:
            wx0, wy0, wx1, wy1 = _rect_of(lp)
            if (wx1 - wx0) <= 2 * enclosure_nm or (wy1 - wy0) <= 2 * enclosure_nm:
                raise ValueError(
                    f"Nwell shape {(_rect_of(lp))} is too small to back out a "
                    f"positive PCOMP at enclosure {enclosure_nm} nm"
                )
        placements = _placements_from_nwell_shapes(items, potential_of, enclosure_nm)
    else:
        raise ValueError(
            f"unsupported input kind(s): {sorted(k.__name__ for k in kinds)}; "
            f"pass PmosWellPlacement records, Nwell LayerPolygons, or "
            f"Gf180PmosGeometry devices"
        )

    return _widen_placements(placements, r, max_passes)


# --- shape application (translate full device geometry by the widen offsets) -


def apply_offsets_to_shapes(
    shapes: Sequence[LayerPolygon],
    offset: Tuple[int, int],
) -> List[LayerPolygon]:
    """Return COPIES of ``shapes`` translated by ``offset`` (whole-device move).

    Use this to apply a ``WellWidenResult.offsets[device]`` shift to ALL of a
    device's layer shapes (COMP, poly, contacts, Nwell) at once, so the well and
    its enclosed PCOMP move together and the DF.4c enclosure is preserved.
    """
    dx, dy = offset
    if dx == 0 and dy == 0:
        return [
            LayerPolygon(lp.layer, lp.datatype,
                         Polygon(vertices=list(lp.polygon.vertices)))
            for lp in shapes
        ]
    return [
        LayerPolygon(
            lp.layer, lp.datatype,
            Polygon(vertices=[Point(v.x + dx, v.y + dy) for v in lp.polygon.vertices]),
        )
        for lp in shapes
    ]


def widened_nwell_shapes(
    result: WellWidenResult,
    nwell_layer: int,
    nwell_datatype: int,
) -> List[LayerPolygon]:
    """Materialise each widened placement's Nwell box as a ``LayerPolygon``.

    The well is recomputed from the (possibly moved) PCOMP via :func:`well_box_for`,
    so the emitted boxes are exactly the DF.4c-grown wells at their new
    positions — ready to drop into a GDS for the gated DRC signoff.
    """
    out: List[LayerPolygon] = []
    for p in result.placements:
        wx0, wy0, wx1, wy1 = well_box_for(p)
        out.append(LayerPolygon(
            layer=nwell_layer, datatype=nwell_datatype,
            polygon=Polygon.rectangle(wx0, wy0, wx1, wy1),
        ))
    return out


__all__ = [
    "NWELL_DIFF_POT_RULE_ID",
    "NWELL_PCOMP_ENCLOSURE_RULE_ID",
    "nwell_diff_potential_spacing_nm",
    "pmos_nwell_enclosure_nm",
    "PmosWellPlacement",
    "well_box_for",
    "well_edge_gap",
    "diff_potential_violations",
    "WellWidenResult",
    "widen_pmos_for_nwell_spacing",
    "apply_offsets_to_shapes",
    "widened_nwell_shapes",
]
