"""Geometric-fidelity spec pack -- the QUALITY-tier's measurable obligations.

PHASE F0 (measure-only; no gating yet -- F1 makes these gate).

ROOT (established with the user): Clay gated only LEGALITY (DRC / LVS), never
geometric QUALITY, so "green" meant LEGAL, not GOOD -- the user's eye was the
only geometric oracle. This module MECHANISES that eye: five metrics measured
from the DESIGN's own topology + the EMITTED GDS, each verdicted against a
reference-calibrated band.

Two non-negotiable design rules (the leakage line + the optimizer seam):

  * METRIC = design + emitted geometry. Every value is measured from (a) the
    netlist intent's TOPOLOGY (mirror groups / diff pairs / kind+body) and
    (b) the EMITTED GDS the oracles already ran on (the single canonical
    artifact -- no second layout path). A coordinate from the REFERENCE never
    enters a metric's input; the reference only CALIBRATES the band scalars
    (clay.config.fidelity_thresholds). By-construction fake results (the
    cardinal sin) are structurally impossible here: the metric cannot read the
    answer because the answer (the reference layout) is not an input.

  * THRESHOLD = config parameter. Every band edge is read via
    ``clay.config.fidelity_thresholds()`` (env ``CLAY_FIDELITY_*`` -> toml ->
    reference-calibrated default). A future optimizer (tSS-BO / adjoint /
    ``clay tune``) auto-calibrates a band by exporting an env var; no source
    edit. The checkers embed no numeric constant.

The five F0 metrics
-------------------
    enclosure_efficiency : (device-OD area inside rings) / (ring-enclosed area).
                           The reference rings tightly (~0.85); the WRONG state
                           wraps mostly-empty bands (~0.33).
    ring_granularity     : closed tap-ring FRAMES in the GDS vs the design's
                           COMPACT-MATCHED-GROUP count (a topology-derived count:
                           mirror groups + diff pairs + leftover singletons by
                           kind+body -- NOT the coarse (kind,body) banding class).
                           3 huge band-rings vs ~9 per-group rings is RED.
    fill_density         : OD-area / cell-area. The reference packs ~0.33.
    aspect_band          : cell W/H against the band bracketing the reference
                           ~1.67. The WRONG state is portrait (~0.60).
    dead_region          : largest CONTIGUOUS empty rectangle (no element
                           footprint) / cell area -- the GENERIC mosaic-
                           completeness metric (RETIRES the cap-center proxy). A
                           dense all-element mosaic leaves only small gaps; a cap
                           dumped as a side/top slab leaves a big contiguous dead
                           region. No cap special-case.

GDS layer numbers (the production layermap): OD=3, poly=13, CONT=15, M1=16,
VIA1=17, NWELL=2, PIMP=7, NIMP=8.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence, Tuple

from clay.config import FidelityThresholds, fidelity_thresholds


# GDS layer numbers (production layermap). A guard ring's TAP DIFFUSION sits on
# the well/implant tap layer: an n-well ring shows on NWELL (2), a p-substrate
# ring shows on PIMP (7). Both render as an ANNULAR (holed) polygon -- the
# closed tap frame this module counts.
GDS_OD = 3
GDS_NWELL = 2
GDS_PIMP = 7
GDS_NIMP = 8
GDS_CONT = 15
GDS_M1 = 16

# the tap-diffusion layers a closed guard-ring frame can live on.
_RING_TAP_LAYERS = (GDS_NWELL, GDS_PIMP)

_MOS = ("nmos", "pmos")


# --------------------------------------------------------------------------- #
# Metric verdict record (the 5-state honest shape, measure-only at F0).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricVerdict:
    """One geometric-fidelity metric measured + verdicted against its band.

    Fields
    ------
    name : str
        The metric name (one of the five F0 metrics).
    value : float or None
        The measured value (None when unobservable).
    expected : float or None
        The design-derived expectation, when the metric has one (e.g. the
        compact-matched-group count for ring_granularity). None otherwise.
    threshold : str
        A human-readable description of the band the value was compared against
        (the resolved config parameter -- provenance for the verdict).
    satisfied : bool or None
        True/False when observable + decidable; None only when unobservable.
    observable : bool
        Whether the metric could be measured from the emitted GDS + topology.
        F0 NEVER turns an unobservable metric into a silent pass.
    detail : str
        Human-readable measured account.
    """

    name: str
    value: Optional[float]
    expected: Optional[float]
    threshold: str
    satisfied: Optional[bool]
    observable: bool
    detail: str


@dataclass(frozen=True)
class FidelityReport:
    """All five metrics measured over one emitted GDS + design intent."""

    area_um2: float
    aspect: float
    fill: float
    enclosure: float
    ring_count: int
    bbox_fill: float = 0.0
    row_balance: Optional[float] = None
    dead_region: float = 0.0
    metrics: Tuple[MetricVerdict, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# GDS geometry helpers (read the EMITTED artifact only -- one canonical path).
# --------------------------------------------------------------------------- #
def _open_top(gds_path: str):
    import klayout.db as kdb

    ly = kdb.Layout()
    ly.read(str(gds_path))
    top = ly.top_cells()[0]
    li = {ly.get_info(idx).layer: idx for idx in ly.layer_indexes()}
    return kdb, ly, top, li


def _layer_region(kdb, top, li, layer_num: int):
    """Merged Region for one GDS layer, or None if the layer is absent."""
    if layer_num not in li:
        return None
    reg = kdb.Region(top.begin_shapes_rec(li[layer_num]))
    reg.merge()
    return reg


def cell_extent_um(gds_path: str) -> Tuple[float, float]:
    """The emitted cell's (W, H) in um from its top-cell bbox."""
    _kdb, _ly, top, _li = _open_top(gds_path)
    bb = top.bbox()
    return (bb.width() / 1000.0, bb.height() / 1000.0)


def principal_physical_body_extent_um(gds_path: str) -> Tuple[float, float]:
    """Return the largest connected all-layer physical component's extent.

    This is a read-only GDS measurement for human/reference comparisons.  It
    unions every recursively emitted polygon in two dimensions, independent of
    layer and coordinates, then selects the unique component with greatest
    physical union area.  A disconnected probe/via hierarchy instance cannot
    therefore widen the layout body.  Empty or area-tied bodies fail closed
    instead of choosing by position or input order.
    """

    kdb, layout, top, _li = _open_top(gds_path)
    physical = kdb.Region()
    for layer_index in layout.layer_indexes():
        physical += kdb.Region(top.begin_shapes_rec(layer_index))
    physical.merge()
    components = tuple(physical.each())
    if not components:
        raise ValueError("GDS has no measurable physical body")
    ranked = sorted(
        ((int(component.area()), component.bbox()) for component in components),
        key=lambda item: item[0],
        reverse=True,
    )
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        raise ValueError("GDS principal physical body is area-ambiguous")
    bbox = ranked[0][1]
    dbu_um = float(layout.dbu)
    return (float(bbox.width()) * dbu_um, float(bbox.height()) * dbu_um)


def measure_principal_physical_body_area_um2(gds_path: str) -> float:
    """Area of the coordinate-free principal physical-body bbox in um2."""

    width_um, height_um = principal_physical_body_extent_um(gds_path)
    return width_um * height_um


def _ring_frames(kdb, top, li):
    """The closed tap-ring FRAMES in the emitted GDS.

    A guard ring's tap diffusion renders as an ANNULAR polygon (a polygon with
    >= 1 hole) on a well/implant tap layer (NWELL n-rings, PIMP p-rings). We
    return one (outer_box, hole_box) per holed polygon whose hole actually
    encloses device OD -- a frame that wraps NOTHING is not a device ring. This
    is measured purely from the emitted artifact; no route metadata, no ref."""
    od = _layer_region(kdb, top, li, GDS_OD)
    frames: List[Tuple[Any, Any]] = []
    for layer_num in _RING_TAP_LAYERS:
        idx = li.get(layer_num)
        if idx is None:
            continue
        reg = kdb.Region(top.begin_shapes_rec(idx))
        reg.merge()
        for poly in reg.each():
            if poly.holes() <= 0:
                continue
            outer = poly.bbox()
            # the largest hole is the enclosed window.
            hole_box = None
            best = -1
            for h in range(poly.holes()):
                pts = list(poly.each_point_hole(h))
                if not pts:
                    continue
                xs = [p.x for p in pts]
                ys = [p.y for p in pts]
                hb = kdb.Box(min(xs), min(ys), max(xs), max(ys))
                if hb.area() > best:
                    best = hb.area()
                    hole_box = hb
            if hole_box is None:
                continue
            # frame must actually enclose device OD (a real device ring).
            if od is not None:
                window = kdb.Region(hole_box)
                if (od & window).area() <= 0:
                    continue
            frames.append((outer, hole_box))
    return frames


# --------------------------------------------------------------------------- #
# topology-derived compact-matched-group count (ring_granularity's expectation).
# --------------------------------------------------------------------------- #
def compact_matched_group_count(intent: Any) -> int:
    """The number of COMPACT MATCHED GROUPS the design's MOS partition into.

    A "compact matched group" is the ring-granularity unit the reference rings
    at (~9 tight per-group rings on the OTA). It is DESIGN-DERIVED, never the
    coarse (kind, body) well-island banding (which collapses to ~3 bands):

      * each MIRROR GROUP (connected component of ``mirror_pairs`` -- a
        shared-gate matched set) is ONE compact group;
      * each DIFF PAIR not already inside a mirror group is ONE compact group;
      * every leftover SINGLETON MOS is folded, by (kind, body net), into ONE
        compact group per leftover (kind, body) class -- same-bulk singletons
        ring together (they place adjacent in their band). This is the
        "physical adjacency" the plan asks for, expressed from topology
        (same well + same bulk net => adjacency-eligible) so it stays
        coordinate-free and translation-invariant.

    Pure topology: mirror_pairs / diff_pairs / device kind + body. No
    coordinate is read, so a uniform translation of the reference bboxes leaves
    the count identical (leakage guard).

    SINGLE SOURCE: the count is ``len(compact_matched_groups(intent))`` -- the
    same partition the materializer rings at (clay.layout.extract.well_islands.
    compact_matched_groups). The expectation a fidelity metric measures against
    and the partition the materializer builds rings from are one function, so
    they cannot drift (no second path).
    """
    from clay.layout.extract.well_islands import compact_matched_groups

    return len(compact_matched_groups(intent))


# --------------------------------------------------------------------------- #
# the five metric measurements (from emitted GDS + topology).
# --------------------------------------------------------------------------- #
def measure_area_um2(gds_path: str) -> float:
    w, h = cell_extent_um(gds_path)
    return w * h


def measure_aspect(gds_path: str) -> float:
    w, h = cell_extent_um(gds_path)
    return (w / h) if h else float("inf")


def measure_fill(gds_path: str) -> float:
    kdb, _ly, top, li = _open_top(gds_path)
    bb = top.bbox()
    cell_area = float(bb.width()) * float(bb.height())  # nm^2
    od = _layer_region(kdb, top, li, GDS_OD)
    od_area = float(od.area()) if od is not None else 0.0
    return (od_area / cell_area) if cell_area else 0.0


def measure_enclosure(gds_path: str) -> Tuple[float, int]:
    """(enclosure efficiency, ring_count) from the emitted GDS.

    enclosure = sum(device-OD area inside each ring window) / sum(ring window
    area). The ring WINDOW is the frame's enclosed (hole) box -- the area the
    ring reserves for devices. ring_count is the number of device-enclosing
    closed tap frames."""
    kdb, _ly, top, li = _open_top(gds_path)
    frames = _ring_frames(kdb, top, li)
    od = _layer_region(kdb, top, li, GDS_OD)
    tot_window = 0.0
    tot_od_in = 0.0
    for _outer, window in frames:
        wreg = kdb.Region(window)
        tot_window += float(window.area())
        if od is not None:
            tot_od_in += float((od & wreg).area())
    enclosure = (tot_od_in / tot_window) if tot_window else 0.0
    return enclosure, len(frames)


def total_mos_fingers(intent: Any) -> int:
    """Total MOS finger multiplicity the design declares (sum of M over MOS).

    DESIGN-DERIVED: read from the netlist intent's device parameters (kind, M),
    never from a coordinate. This is the denominator of contact_density -- the
    number of channel fingers the synthesised devices carry, so the metric
    measures contacts-PER-FINGER (a per-finger S/D landing density independent of
    how many devices the design has). Translation-invariant (M is a parameter, not
    a position)."""
    from clay.layout.extract.realize_render import (
        _intent_device_params,
        _norm_kind,
    )

    params = _intent_device_params(intent)
    tot = 0
    for _name, p in (params or {}).items():
        if not p:
            continue
        kind, _W, _L, M = p
        if _norm_kind(kind) in _MOS:
            tot += int(M)
    return tot


def measure_contact_density(gds_path: str, intent: Any) -> Tuple[float, int, int]:
    """(contacts per MOS finger, CONT shape count, total MOS fingers).

    The reference packs a FULL contact-cut column per source/drain face (~67
    CONT per finger on the OTA); a single mid-channel cut per S/D leaves ~22. The
    contact count comes from the EMITTED GDS (the single canonical artifact); the
    denominator (total MOS fingers) is DESIGN-DERIVED from topology. The
    reference's measured 5813 CONT only CALIBRATES the band scalar, never an
    input. Returns ratio 0.0 (unobservable) when the design declares no MOS
    finger -- never a silent pass."""
    kdb, _ly, top, li = _open_top(gds_path)
    cont = _layer_region(kdb, top, li, GDS_CONT)
    # CONT cuts never overlap (uniform 0.22 um cuts) so the raw shape count is the
    # contact count; count discrete shapes (merge would still keep them disjoint).
    n_cont = 0
    if GDS_CONT in li:
        n_cont = sum(1 for _ in top.begin_shapes_rec(li[GDS_CONT]))
    fingers = total_mos_fingers(intent)
    ratio = (n_cont / fingers) if fingers else 0.0
    return ratio, n_cont, fingers


def sd_column_heights(
    gds_path: str,
    cont_layer: int = GDS_CONT,
    *,
    x_tol_nm: int = 60,
    y_gap_max_nm: int = 800,
) -> List[int]:
    """Per-S/D-COLUMN contact-cut heights from a GDS (GDS<->GDS, klayout.db).

    A "column" is the longest y-CONTIGUOUS run of CONT cuts that share an
    x-bucket (rounded to ``x_tol_nm``); its height is the cut COUNT. Measured
    IDENTICALLY on the vendor reference (``ota_dev/ref/reference_layout.gds``)
    and on the emitted layout, so the contacts-per-S/D-column FIDELITY is a real
    geometry-vs-geometry comparison: the vendor stacks a full-height lattice
    (~44 cuts per 21um S/D column on the OTA), a single mid-channel groove
    leaves ~1. Returns the SORTED per-column heights (ascending); the caller
    reads e.g. the tall-column median for the dense-lattice signal. Empty when
    the layer is absent (never a silent pass -- an empty list is observably
    non-dense). Pure relative geometry (x-bucket + y-run); a uniform translation
    leaves every column height identical (the leakage line)."""
    from collections import defaultdict

    kdb, _ly, top, li = _open_top(gds_path)
    if cont_layer not in li:
        return []
    cols: dict = defaultdict(list)
    for s in top.begin_shapes_rec(li[cont_layer]):
        b = s.shape().bbox().transformed(s.itrans())
        cx = (b.left + b.right) // 2
        cy = (b.top + b.bottom) // 2
        cols[round(cx / x_tol_nm)].append(cy)
    heights: List[int] = []
    for ys in cols.values():
        ys.sort()
        run = longest = 1
        for i in range(1, len(ys)):
            run = run + 1 if (ys[i] - ys[i - 1]) <= y_gap_max_nm else 1
            if run > longest:
                longest = run
        heights.append(longest)
    heights.sort()
    return heights


def tall_column_median(heights: List[int], tall_min: int = 20) -> int:
    """Median height of the TALL S/D columns (>= ``tall_min`` cuts) in a
    :func:`sd_column_heights` distribution -- the dense-lattice column signal,
    robust to the many 1-cut incidental columns (gate/tap/rail landings). 0 when
    no column is tall (the sparse groove state)."""
    tall = sorted(h for h in heights if h >= tall_min)
    return tall[len(tall) // 2] if tall else 0


def _od_row_cluster(kdb, top, li):
    """Cluster the emitted OD diffusion into device ROWS by y-gap.

    Reads ONLY the emitted GDS (the single canonical artifact). Each merged OD
    polygon is a diffusion cluster (one device's S/D + channel fingers touch);
    we sort the clusters by centroid-y and split into rows wherever the vertical
    gap between adjacent clusters exceeds max(3 * median-cluster-height, 8 um) --
    the SAME row-banding gap rule the band-stack uses (relgraph.cluster_bands),
    so the metric measures exactly the row paradigm it is meant to expose. Pure
    relative geometry (gaps + spans), so a uniform translation leaves every row
    span identical (leakage line). Returns a list of rows; each row is a list of
    OD-cluster bboxes (kdb.Box, nm units)."""
    import statistics

    od = _layer_region(kdb, top, li, GDS_OD)
    if od is None or od.is_empty():
        return []
    boxes = [poly.bbox() for poly in od.each()]
    boxes.sort(key=lambda b: (b.bottom + b.top) / 2.0)
    if not boxes:
        return []
    heights = [(b.top - b.bottom) / 1000.0 for b in boxes]
    med_h = statistics.median(heights) if heights else 0.0
    split_gap = max(3.0 * med_h, 8.0)  # um -- same rule as cluster_bands
    rows: List[List[Any]] = []
    cur = [boxes[0]]
    for prev, b in zip(boxes, boxes[1:]):
        gap_um = (b.bottom - prev.top) / 1000.0
        if gap_um > split_gap:
            rows.append(cur)
            cur = [b]
        else:
            cur.append(b)
    rows.append(cur)
    return rows


def measure_bbox_fill(gds_path: str) -> float:
    """(sum of per-device-footprint bbox area) / (cell-bbox area).

    The PLACEMENT-tier occupancy: how densely the placer packs DEVICE FOOTPRINTS
    into the cell bbox, distinct from fill_density (device-layer OD / cell). The
    occupied numerator is the sum of each merged-OD diffusion cluster's bounding
    box (a device's footprint, including its own S/D extent), measured purely
    from the emitted GDS; the denominator is the emitted cell bbox. A 2D pack
    fills the bbox densely; the band-stack ROW paradigm leaves the narrow row's
    tail empty, so the cell bbox (= widest row x narrowest row stack) carries
    dead area the device footprints do not occupy -> a low bbox_fill. Relative
    (areas / areas), so translation-invariant (leakage line)."""
    kdb, _ly, top, li = _open_top(gds_path)
    bb = top.bbox()
    cell_area = float(bb.width()) * float(bb.height())  # nm^2
    if cell_area <= 0:
        return 0.0
    od = _layer_region(kdb, top, li, GDS_OD)
    if od is None or od.is_empty():
        return 0.0
    occ = 0.0
    for poly in od.each():
        b = poly.bbox()
        occ += float(b.width()) * float(b.height())
    return occ / cell_area


def measure_row_balance(gds_path: str) -> Tuple[Optional[float], int]:
    """(max-row-span / min-row-span, n_rows) over the emitted device rows.

    Clusters the emitted OD into device ROWS (``_od_row_cluster``) and measures
    each row's x-span (max-right minus min-left of its OD clusters). The ratio
    of the widest row span to the narrowest is the ROW-BALANCE: the band-stack's
    pmos-row (~159 um) vs nmos-row (~71 um) gives ~2.2 (the cell width = widest
    row by construction wastes the narrow row's tail); a balanced 2D pack keeps
    rows near-equal width (~1.0-1.3). Pure relative spans (translation-
    invariant, leakage line). Returns (None, n) when fewer than two rows exist
    (a single-row layout has no imbalance to measure -- never a silent pass; the
    verdict marks it unobservable)."""
    kdb, _ly, top, li = _open_top(gds_path)
    rows = _od_row_cluster(kdb, top, li)
    if len(rows) < 2:
        return None, len(rows)
    spans: List[float] = []
    for row in rows:
        xl = min(b.left for b in row)
        xr = max(b.right for b in row)
        spans.append((xr - xl) / 1000.0)
    spans = [s for s in spans if s > 0]
    if len(spans) < 2:
        return None, len(rows)
    return (max(spans) / min(spans)), len(rows)


# GENERIC OCCUPIED LAYERS for the dead-region metric: every ELEMENT footprint
# (device OD/poly/local-metal AND the MIM cap plate stack). A passive cap carries
# no OD but IS a real occupied tile, so the dead-region search must count its plate
# metals as occupied -- otherwise an internally co-placed cap reads as "empty" (a
# metric artifact). These are PRODUCTION-layermap layer numbers, the same emitted
# artifact the LVS/DRC oracles ran on; no reference, no per-circuit constant.
GDS_POLY = 13
GDS_M2 = 18
# MIM cap plate / cover / via stack layers (external_pdk layermap). Generic: any cap on
# this PDK lands here regardless of circuit.
GDS_MIM_LAYERS = (28, 33, 38, 67)
_OCCUPIED_LAYERS = (GDS_OD, GDS_POLY, GDS_M1, GDS_M2) + GDS_MIM_LAYERS


def _occupied_region(kdb, top, li):
    """The union of every ELEMENT footprint on the emitted GDS (device layers +
    MIM cap plates). GENERIC -- the cap is just another occupied tile, no cap
    branch. Reads only the emitted artifact (no reference, no route metadata)."""
    occ = kdb.Region()
    for lnum in _OCCUPIED_LAYERS:
        r = _layer_region(kdb, top, li, lnum)
        if r is not None:
            occ += r
    occ.merge()
    return occ


def measure_dead_region(gds_path: str, *, grid_nm: int = 1000
                        ) -> Tuple[float, Tuple[float, float, float, float]]:
    """(largest-contiguous-dead-rectangle fraction, rect_um) on the emitted GDS.

    The LARGEST axis-aligned EMPTY rectangle inside the cell bbox that touches NO
    element footprint, as a fraction of the cell bbox area. This is the GENERIC
    mosaic-completeness metric the unified all-element placement is judged by: a
    dense mosaic (devices wrapping the cap, no parked side/top slab) has a SMALL
    largest dead rect; a cap dumped as a side block leaves a big contiguous dead
    region (the retired cap-mode symptom -- measured ~0.42 on the planar bolt-on).

    Discretised to ``grid_nm`` and solved by the classic largest-rectangle-in-
    histogram over the empty-cell raster (deterministic, O(cells)). Pure relative
    geometry (areas / areas), so a uniform translation leaves it identical (the
    leakage line). NO cap special-case -- the cap is one occupied tile like any
    other (``_occupied_region``)."""
    kdb, _ly, top, li = _open_top(gds_path)
    bb = top.bbox()
    cell_area_nm = float(bb.width()) * float(bb.height())
    if cell_area_nm <= 0:
        return 0.0, (0.0, 0.0, 0.0, 0.0)
    occ = _occupied_region(kdb, top, li)
    W, H = bb.width(), bb.height()
    nx = max(1, W // grid_nm)
    ny = max(1, H // grid_nm)
    empty = [[False] * nx for _ in range(ny)]
    for j in range(ny):
        y0 = bb.bottom + j * grid_nm
        y1 = y0 + grid_nm
        if (kdb.Region(kdb.Box(bb.left, y0, bb.right, y1)) & occ).is_empty():
            for i in range(nx):
                empty[j][i] = True
            continue
        for i in range(nx):
            x0 = bb.left + i * grid_nm
            if (kdb.Region(kdb.Box(x0, y0, x0 + grid_nm, y1)) & occ).is_empty():
                empty[j][i] = True
    heights = [0] * nx
    best_cells = 0
    best_rect = (0, 0, 0, 0)
    for j in range(ny):
        for i in range(nx):
            heights[i] = heights[i] + 1 if empty[j][i] else 0
        stack: List[Tuple[int, int]] = []
        for i in range(nx + 1):
            cur = heights[i] if i < nx else 0
            start = i
            while stack and stack[-1][1] > cur:
                s_i, s_h = stack.pop()
                area = s_h * (i - s_i)
                if area > best_cells:
                    best_cells = area
                    best_rect = (
                        bb.left + s_i * grid_nm,
                        bb.bottom + (j - s_h + 1) * grid_nm,
                        bb.left + i * grid_nm,
                        bb.bottom + (j + 1) * grid_nm,
                    )
                start = s_i
            stack.append((start, cur))
    frac = best_cells / (nx * ny) if nx * ny else 0.0
    rect_um = tuple(round(v / 1000.0, 2) for v in best_rect)
    return frac, rect_um  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# over-cell routing fraction (the ROUTER-CONTRACT QUALITY obligation).
#
# ROOT (the OTA density hard-core experiment, ota_dev/analysis/
# ota_density_hardcore_r1_2026-07-18.html): the OTA density gap on the ROUTED
# artifact is a placement-INDEPENDENT ROUTER capability gap -- OVER-CELL routing.
# The reference runs 57.7% of its routing-TRUNK metal (M2/M3) directly OVER the
# device-active footprint, so clusters abut with no dedicated routing channel and
# the core stays tight; our router keeps the trunk in the apron beside the cells
# (~24%, INVARIANT to placement density: FINE-dense 25.9%, production-loose
# 24.3%), so metal sprawls (dense-in != dense-out). None of the 8 F0/S1 metrics
# names this -- they are all full-bbox / pure-GDS. This metric NAMES the gap.
#
# ROUTING layers (open_pdk VERIFIED layermap -- from pdks/open_pdk/.../
# external_pdk.layermap, confirmed by loading reference_layout.gds; see
# ota_dev/analysis/decompose.py): the routing TRUNK metals are M2=18 and M3=28
# (NOT device-local M1=16 pads). The device-active (device-cell) footprint is
# OD=3 + poly=13 (the gate stack). NOTE: the dead_region metric above lists 28 in
# GDS_MIM_LAYERS -- a pre-existing labeling that treats 28 as a MIM plate; the
# verified open_pdk layermap assigns 28 to M3 (67 is the MIM plate). That
# discrepancy is left untouched here (dead_region only needs "any occupied tile",
# so counting 28 as occupied is harmless for it); this metric uses the verified
# assignment for the routing-trunk numerator.
# --------------------------------------------------------------------------- #
GDS_M3 = 28
_ROUTE_METAL_LAYERS = (GDS_M2, GDS_M3)          # 18, 28 -- routing-trunk metals
_DEVICE_ACTIVE_LAYERS = (GDS_OD, GDS_POLY)      # 3, 13  -- device-cell footprint


def _merged_layers(kdb, ly, top, layer_nums):
    """Merged Region over ALL datatypes of the given GDS layer NUMBERS.

    Collects every layer index whose layer number is in ``layer_nums`` (so a
    layer split across datatypes is fully captured -- the SAME all-datatype
    semantics the decompose measurement instrument uses), unions + merges them.
    Reads only the emitted artifact (no reference, no route metadata)."""
    want = set(layer_nums)
    reg = kdb.Region()
    for idx in ly.layer_indexes():
        if ly.get_info(idx).layer in want:
            reg += kdb.Region(top.begin_shapes_rec(idx))
    reg.merge()
    return reg


def measure_over_cell_fraction(
    gds_path: str,
) -> Tuple[Optional[float], float, float]:
    """(over-cell fraction, route-metal-over-device um^2, total route-metal um^2).

    The ROUTER-CONTRACT metric the density hard-core experiment named: the
    fraction of ROUTING-TRUNK metal (M2/M3 -- NOT device-local M1 pads) whose
    area sits DIRECTLY OVER the device-active footprint (OD + poly gates):

        over_cell = area(route-metal INTERSECT device-active) / area(route-metal)

    Measured purely from the EMITTED GDS (the single canonical artifact the
    LVS/DRC oracles ran on); no reference coordinate enters -- it is a RELATIVE
    area/area, so a uniform translation leaves it identical (the leakage line).
    The reference's 0.577 only CALIBRATES the band scalar (config), never an
    input.

    Returns ``(None, 0.0, 0.0)`` when the layout carries NO route-metal (a
    place-only / unrouted GDS): the fraction is UNMEASURABLE, never a silent 0.0
    pass (the verdict marks it unobservable, which still blocks a routed-artifact
    signoff as an un-run required class)."""
    kdb, ly, top, _li = _open_top(gds_path)
    route = _merged_layers(kdb, ly, top, _ROUTE_METAL_LAYERS)
    dbu = ly.dbu
    route_area = float(route.area()) * dbu * dbu
    if route_area <= 0.0:
        return None, 0.0, 0.0
    active = _merged_layers(kdb, ly, top, _DEVICE_ACTIVE_LAYERS)
    over_area = float((route & active).area()) * dbu * dbu
    return (over_area / route_area), over_area, route_area


# --------------------------------------------------------------------------- #
# F2 / RG6 route-owned per-layer profile.
#
# Unlike ``measure_over_cell_fraction`` above, this instrument does not treat
# every polygon on a routing layer as router artwork.  It intersects the exact
# emitted GDS with the canonical route ownership manifest.  Device-local metal
# and MIM artwork therefore cannot be credited to the router merely because it
# shares a GDS layer.  PDK preferred axes and the selected resource plan are
# inputs; neither a circuit identity nor a reference-layout coordinate is.
# --------------------------------------------------------------------------- #
_MAX_PROFILE_TAGS = 100_000
_MAX_PROFILE_GROUPS = 4_096
_MAX_PROFILE_GROUP_BOXES = 16_384
_MAX_PROFILE_PROGRAMS = 1_024
_MAX_PROFILE_TRACKS = 100_000
_MAX_PROFILE_GROUP_BOX_COUNT = 100_000
_MAX_PROFILE_JOIN_EVALUATIONS = 2_000_000
_LONG_RUN_ASPECT_MIN = 4.0

LayerKey = Tuple[int, int]
BBoxNm = Tuple[int, int, int, int]


def _bounded_sequence(values: Sequence[Any], limit: int, what: str) -> Tuple[Any, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{what} must be a bounded tuple/list")
    if len(values) > limit:
        raise ValueError(f"{what} exceeds bounded limit {limit}")
    return tuple(values)


def _profile_layer_key(value: Any) -> LayerKey:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError("route profile layer key must contain two integers")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise TypeError("route profile layer key must contain two integers")
    return int(value[0]), int(value[1])


def _profile_bbox(value: Any, what: str = "bbox") -> BBoxNm:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise TypeError(f"{what} must contain four integer-nm coordinates")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise TypeError(f"{what} must contain four integer-nm coordinates")
    box = tuple(int(item) for item in value)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{what} must be a non-empty half-open bbox")
    return box  # type: ignore[return-value]


@dataclass(frozen=True)
class DeviceGroupFootprint:
    """One independent placed group, represented by final integer-nm boxes."""

    group_id: str
    bboxes_nm: Tuple[BBoxNm, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise TypeError("device group id must be a non-empty string")
        boxes = _bounded_sequence(
            self.bboxes_nm, _MAX_PROFILE_GROUP_BOXES, "device group boxes"
        )
        if not boxes:
            raise ValueError("device group must contain at least one bbox")
        object.__setattr__(
            self,
            "bboxes_nm",
            tuple(_profile_bbox(box, "device group bbox") for box in boxes),
        )


@dataclass(frozen=True)
class PlannerLayerAllocation:
    """Selected corridor/track capacity assigned to one physical metal layer."""

    layer_n: int
    layer_key: LayerKey
    use_count: int
    track_count: int
    area_um2: float
    nets: Tuple[str, ...]
    candidate_pool_hash: str
    materialized_shape_count: int = 0
    track_uses: Tuple["PlannerTrackUseEvidence", ...] = ()


@dataclass(frozen=True)
class PlannerTrackUseEvidence:
    """Exact selected use bound to its owner, interval, lane and emitted bbox."""

    net: str
    candidate_id: str
    use_id: str
    layer_key: LayerKey
    interval_nm: Tuple[int, int]
    selected_track_centers_nm: Tuple[int, ...]
    width_nm: int
    emitted_bbox_nm: BBoxNm


class RouteProfileLineage(str, Enum):
    """The construction lineage measured by the exact-GDS route profile."""

    EXACT_RESOURCE_PLAN = "EXACT_RESOURCE_PLAN"
    NEGOTIATED_GEOMETRY = "NEGOTIATED_GEOMETRY"


@dataclass(frozen=True)
class NegotiatedLayerGeometry:
    """Exact manifest/GDS evidence for one negotiated routing metal layer."""

    layer_n: int
    layer_key: LayerKey
    shape_count: int
    covered_shape_count: int
    area_um2: float
    nets: Tuple[str, ...]
    route_manifest_sha256: str

    def __post_init__(self) -> None:
        if type(self.layer_n) is not int or self.layer_n <= 0:
            raise ValueError("negotiated layer number must be a positive exact int")
        if (
            type(self.layer_key) is not tuple
            or len(self.layer_key) != 2
            or any(type(value) is not int for value in self.layer_key)
        ):
            raise TypeError("negotiated layer key must be two exact ints")
        for name in ("shape_count", "covered_shape_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"negotiated {name} must be a positive exact int")
        if self.covered_shape_count != self.shape_count:
            raise ValueError("negotiated manifest coverage is incomplete")
        if (
            type(self.area_um2) is not float
            or not math.isfinite(self.area_um2)
            or self.area_um2 <= 0.0
        ):
            raise ValueError("negotiated route-owned area must be finite and positive")
        if (
            type(self.nets) is not tuple
            or not self.nets
            or any(type(net) is not str or not net for net in self.nets)
            or self.nets != tuple(sorted(set(self.nets)))
        ):
            raise ValueError("negotiated nets must be a non-empty canonical tuple")
        digest = self.route_manifest_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("negotiated route manifest digest is malformed")


@dataclass(frozen=True)
class LayerRouteProfile:
    """Exact route-owned geometry and allocation evidence for one metal layer."""

    layer_n: int
    layer_key: LayerKey
    preferred_axis: str
    route_owned_area_um2: float
    over_active_area_um2: float
    over_active_fraction: Optional[float]
    preferred_axis_purity: Optional[float]
    long_run_count: int
    non_mim_area_um2: float
    planner_allocation: Optional[PlannerLayerAllocation] = None
    negotiated_geometry: Optional[NegotiatedLayerGeometry] = None

    def __post_init__(self) -> None:
        if type(self.layer_n) is not int or self.layer_n <= 0:
            raise ValueError("route profile layer number must be a positive exact int")
        if (
            type(self.layer_key) is not tuple
            or len(self.layer_key) != 2
            or any(type(value) is not int for value in self.layer_key)
        ):
            raise TypeError("route profile layer key must be two exact ints")
        if self.preferred_axis not in ("H", "V", "horizontal", "vertical"):
            raise ValueError("route profile preferred axis is invalid")
        for name in (
            "route_owned_area_um2",
            "over_active_area_um2",
            "non_mim_area_um2",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"route profile {name} must be finite and nonnegative")
        if self.route_owned_area_um2 <= 0.0:
            raise ValueError("route profile route-owned area must be positive")
        for name in ("over_active_fraction", "preferred_axis_purity"):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
                or value > 1.0
            ):
                raise ValueError(f"route profile {name} must lie in [0, 1]")
        if type(self.long_run_count) is not int or self.long_run_count < 0:
            raise ValueError("route profile long-run count must be a nonnegative int")
        planned = type(self.planner_allocation) is PlannerLayerAllocation
        negotiated = type(self.negotiated_geometry) is NegotiatedLayerGeometry
        if planned == negotiated:
            raise ValueError("route profile requires exactly one allocation lineage")
        if negotiated:
            evidence = self.negotiated_geometry
            if (
                self.preferred_axis not in ("H", "V")
                or
                evidence.layer_n != self.layer_n
                or evidence.layer_key != self.layer_key
                or evidence.area_um2 != self.route_owned_area_um2
            ):
                raise ValueError("negotiated layer evidence does not replay profile")


@dataclass(frozen=True)
class CrossDeviceRouteComponent:
    """A physically connected route-owned component touching independent groups."""

    component_id: str
    nets: Tuple[str, ...]
    device_group_ids: Tuple[str, ...]
    layer_keys: Tuple[LayerKey, ...]
    route_owned_area_um2: float


@dataclass(frozen=True)
class LayerRouteProfileReport:
    """Five-state F2 evidence; missing ownership/allocation is never a pass."""

    status: str
    layers: Tuple[LayerRouteProfile, ...] = ()
    cross_device_components: Tuple[CrossDeviceRouteComponent, ...] = ()
    unplanned_layer_numbers: Tuple[int, ...] = ()
    manifest_available: bool = False
    manifest_entry_count: int = 0
    expected_materialized_shape_count: int = 0
    candidate_pool_hash: str = ""
    resource_plan_hash: str = ""
    artifact_sha256: str = ""
    route_manifest_sha256: str = ""
    plan_verified: bool = False
    materialization_verified: bool = False
    device_group_evidence_available: bool = False
    mim_evidence_available: bool = False
    reason: str = ""
    independent_device_group_ids: Tuple[str, ...] = ()
    evidence_lineage: str = RouteProfileLineage.EXACT_RESOURCE_PLAN.value
    planner_evidence_applicable: bool = True
    ownership_manifest_verified: bool = False
    manifest_covered_entry_count: int = 0
    exact_geometry_verified: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "pass"


@dataclass
class _ProfileJoinBudget:
    limit: int
    used: int = 0

    def consume(self, what: str) -> None:
        self.used += 1
        if self.used > self.limit:
            raise ValueError(f"{what} exceeds bounded join limit {self.limit}")


def _profile_box_region(kdb, layout, bbox_nm: BBoxNm):
    scale = float(layout.dbu) * 1000.0
    coords = tuple(round(value / scale) for value in bbox_nm)
    return kdb.Region(kdb.Box(*coords))


def _profile_region_touches(left: Any, right: Any) -> bool:
    if not (left & right).is_empty():
        return True
    # KLayout's area intersection is empty for two conductor rectangles sharing
    # an edge.  A one-dbu sizing on only one side recognizes that exact electrical
    # contact without accepting a positive gap.
    return not (left.sized(1) & right).is_empty()


def _planner_layer_allocations(
    candidate_pool: Any,
    resource_plan: Any,
    materialization_certificate: Any,
    tech: Any,
    metal_by_key: Dict[LayerKey, int],
    axis_by_key: Dict[LayerKey, str],
) -> Tuple[
    Dict[LayerKey, PlannerLayerAllocation],
    Tuple[Tuple[str, LayerKey, BBoxNm], ...],
    str,
    str,
]:
    """Deep-replay the selected plan and bind its exact materialized geometry."""
    from clay.layout.route.engine_gridless import (
        GridlessAllocationReplayCertificate,
        _verify_gridless_allocation_replay_certificate,
    )
    from clay.layout.route.resource_plan import (
        CandidatePool,
        RouteResourcePlan,
        verify_route_resource_plan,
    )
    from clay.layout.route.resource_world import pdk_provenance_digest

    if type(candidate_pool) is not CandidatePool:
        raise TypeError("exact CandidatePool is unavailable")
    if type(resource_plan) is not RouteResourcePlan:
        raise TypeError("exact RouteResourcePlan is unavailable")
    if type(materialization_certificate) is not GridlessAllocationReplayCertificate:
        raise TypeError("final gridless allocation replay certificate is unavailable")
    _verify_gridless_allocation_replay_certificate(materialization_certificate)
    if not verify_route_resource_plan(candidate_pool, resource_plan):
        raise ValueError("selected route resource plan did not replay")
    pool_hash = candidate_pool.sha256()
    plan_hash = resource_plan.sha256()
    expected_selected = tuple(
        (selected.net, selected.candidate_id)
        for selected in resource_plan.selected_nets
    )
    certificate = materialization_certificate
    if (
        resource_plan.candidate_pool_hash != pool_hash
        or certificate.candidate_pool_hash != pool_hash
        or certificate.resource_plan_hash != plan_hash
        or certificate.selected_candidates != expected_selected
        or certificate.materialization_complete is not True
        or certificate.current_pdk_provenance_digest != pdk_provenance_digest(tech)
    ):
        raise ValueError("final materialization certificate does not bind the selected plan")
    world_digests = tuple(sorted({
        program.routing_world_provenance_digest
        for program in resource_plan.allocated_programs
    }))
    if certificate.routing_world_provenance_digests != world_digests:
        raise ValueError("final materialization certificate has stale routing-world provenance")
    if resource_plan.materialization_complete is not True:
        raise ValueError("selected route resource plan has incomplete allocation IR")

    programs = _bounded_sequence(
        resource_plan.allocated_programs,
        _MAX_PROFILE_PROGRAMS,
        "allocated route programs",
    )
    if not programs:
        raise ValueError("selected route resource plan has no allocated programs")
    accum: Dict[LayerKey, Dict[str, Any]] = {}
    geometry_records: list[Tuple[str, LayerKey, BBoxNm]] = []
    certificate_geometry_records: list[Tuple[str, int, int, BBoxNm]] = []
    track_total = 0
    for program in programs:
        net = str(getattr(program, "net", "")).strip()
        candidate_id = str(getattr(program, "candidate_id", "")).strip()
        if not net:
            raise ValueError("allocated route program has no net owner")
        tracks = _bounded_sequence(
            getattr(program, "tracks", None),
            _MAX_PROFILE_TRACKS,
            "allocated tracks",
        )
        track_total += len(tracks)
        if track_total > _MAX_PROFILE_TRACKS:
            raise ValueError(
                f"allocated tracks exceeds bounded limit {_MAX_PROFILE_TRACKS}"
            )
        for track in tracks:
            resource = getattr(track, "resource", None)
            key = _profile_layer_key(getattr(resource, "gds_layer_key", None))
            if key not in metal_by_key:
                raise ValueError(f"allocated track names unknown metal layer {key}")
            interval = getattr(track, "interval_nm", None)
            if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                raise TypeError("allocated track interval must contain two integers")
            if any(isinstance(v, bool) or not isinstance(v, Integral) for v in interval):
                raise TypeError("allocated track interval must contain two integers")
            run = int(interval[1]) - int(interval[0])
            width = getattr(track, "width_nm", None)
            centers = _bounded_sequence(
                getattr(track, "selected_track_centers_nm", None),
                _MAX_PROFILE_TRACKS,
                "selected track centers",
            )
            if (
                run <= 0
                or isinstance(width, bool)
                or not isinstance(width, Integral)
                or int(width) <= 0
                or not centers
                or any(isinstance(v, bool) or not isinstance(v, Integral) for v in centers)
            ):
                raise ValueError("allocated track geometry is incomplete")
            rec = accum.setdefault(
                key, {
                    "uses": 0,
                    "tracks": 0,
                    "area_nm2": 0,
                    "nets": set(),
                    "shape_count": 0,
                    "track_uses": [],
                }
            )
            rec["uses"] += 1
            rec["tracks"] += len(centers)
            rec["nets"].add(net)
            resource_axis = str(getattr(resource, "preferred_axis", "")).upper()
            if resource_axis not in ("H", "V"):
                raise ValueError(f"{key}: planner axis is undeclared")
            if key not in axis_by_key:
                raise ValueError(
                    f"selected metal{metal_by_key[key]} has no PDK "
                    "preferred-axis declaration"
                )
            if resource_axis != axis_by_key[key]:
                raise ValueError(
                    f"{key}: planner axis {resource_axis} contradicts "
                    f"PDK axis {axis_by_key[key]}"
                )
            low_half = int(width) // 2
            high_half = int(width) - low_half
            if resource_axis == "H":
                emitted_bbox = (
                    int(interval[0]),
                    int(centers[0]) - low_half,
                    int(interval[1]),
                    int(centers[-1]) + high_half,
                )
            else:
                emitted_bbox = (
                    int(centers[0]) - low_half,
                    int(interval[0]),
                    int(centers[-1]) + high_half,
                    int(interval[1]),
                )
            emitted_bbox = _profile_bbox(emitted_bbox, "allocated track bbox")
            area_nm2 = (
                emitted_bbox[2] - emitted_bbox[0]
            ) * (emitted_bbox[3] - emitted_bbox[1])
            rec["area_nm2"] += area_nm2
            rec["shape_count"] += 1
            rec["track_uses"].append(
                PlannerTrackUseEvidence(
                    net=net,
                    candidate_id=candidate_id,
                    use_id=str(getattr(track, "use_id", "")),
                    layer_key=key,
                    interval_nm=(int(interval[0]), int(interval[1])),
                    selected_track_centers_nm=tuple(int(value) for value in centers),
                    width_nm=int(width),
                    emitted_bbox_nm=emitted_bbox,
                )
            )
            geometry_records.append((net, key, emitted_bbox))
            certificate_geometry_records.append((net, key[0], key[1], emitted_bbox))

        landing_options = _bounded_sequence(
            getattr(program, "landing_options", None),
            _MAX_PROFILE_TRACKS,
            "allocated landing options",
        )
        for allocation in landing_options:
            sites = _bounded_sequence(
                getattr(allocation, "landing_sites", None),
                _MAX_PROFILE_TRACKS,
                "allocated landing sites",
            )
            for site in sites:
                lower_n = int(getattr(site, "lower_layer_n", 0))
                upper_n = int(getattr(site, "upper_layer_n", 0))
                try:
                    lower_key = _profile_layer_key(tech.metal_key(lower_n))
                    upper_key = _profile_layer_key(tech.metal_key(upper_n))
                    via_key = _profile_layer_key(tech.via_key(lower_n))
                except Exception as exc:
                    raise ValueError("allocated landing names an incomplete PDK stack") from exc
                emitted = (
                    (lower_key, _profile_bbox(site.lower_pad_bbox_nm, "lower landing pad")),
                    (upper_key, _profile_bbox(site.upper_pad_bbox_nm, "upper landing pad")),
                    (via_key, _profile_bbox(site.cut_bbox_nm, "landing via cut")),
                )
                for key, bbox in emitted:
                    geometry_records.append((net, key, bbox))
                    certificate_geometry_records.append((net, key[0], key[1], bbox))
                    if key not in metal_by_key:
                        continue
                    rec = accum.setdefault(
                        key,
                        {
                            "uses": 0,
                            "tracks": 0,
                            "area_nm2": 0,
                            "nets": set(),
                            "shape_count": 0,
                            "track_uses": [],
                        },
                    )
                    rec["uses"] += 1
                    rec["area_nm2"] += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    rec["nets"].add(net)
                    rec["shape_count"] += 1

    geometry_payload = [
        [net, layer, datatype, list(bbox)]
        for net, layer, datatype, bbox in certificate_geometry_records
    ]
    expected_geometry_digest = hashlib.sha256(json.dumps(
        geometry_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if certificate.geometry_digest != expected_geometry_digest:
        raise ValueError("final materialization geometry digest does not replay from selected allocation")
    out = {
        key: PlannerLayerAllocation(
            layer_n=metal_by_key[key],
            layer_key=key,
            use_count=int(rec["uses"]),
            track_count=int(rec["tracks"]),
            area_um2=float(rec["area_nm2"]) / 1_000_000.0,
            nets=tuple(sorted(rec["nets"])),
            candidate_pool_hash=pool_hash,
            materialized_shape_count=int(rec["shape_count"]),
            track_uses=tuple(sorted(
                rec["track_uses"],
                key=lambda item: (item.net, item.candidate_id, item.use_id),
            )),
        )
        for key, rec in accum.items()
    }
    return out, tuple(geometry_records), pool_hash, plan_hash


def _profile_pdk_stack(
    tech: Any,
    max_metal: int,
) -> Tuple[Dict[LayerKey, int], Dict[LayerKey, str], Dict[LayerKey, Tuple[LayerKey, LayerKey]]]:
    if type(max_metal) is not int or max_metal < 2:
        raise ValueError("max_metal must be an integer >= 2")
    probed: list[Optional[LayerKey]] = []
    for layer_n in range(1, max_metal + 1):
        try:
            probed.append(_profile_layer_key(tech.metal_key(layer_n)))
        except Exception:
            probed.append(None)
    present = [index for index, key in enumerate(probed) if key is not None]
    if not present:
        raise ValueError("PDK declares no routing metal stack")
    last = present[-1]
    holes = tuple(index + 1 for index, key in enumerate(probed[: last + 1]) if key is None)
    if holes:
        raise ValueError(
            "incomplete PDK metal stack; missing "
            + ", ".join(f"M{layer_n}" for layer_n in holes)
        )
    metals = tuple(key for key in probed[: last + 1] if key is not None)
    if len(metals) < 2 or len(metals) != len(set(metals)):
        raise ValueError("PDK routing metal stack is incomplete or reuses a layer key")
    metal_by_key = {key: layer_n for layer_n, key in enumerate(metals, start=1)}
    axis_by_key = {}
    for key, layer_n in metal_by_key.items():
        declared = getattr(tech, "metal_preferred_axis_declared", lambda _n: None)(
            layer_n
        )
        axis = str(declared or "").strip().upper()
        # A complete foundry layermap can legitimately include conductor layers
        # which this design uses only inside fixed device artwork (for example a
        # MIM top plate).  Preferred-axis authority is required only once a layer
        # participates in the selected/owned routing evidence below; do not make
        # an unused, merely mappable layer poison the whole profile.
        if axis in ("H", "V"):
            axis_by_key[key] = axis
    via_between = {}
    for lower_n in range(1, len(metals)):
        try:
            via_key = _profile_layer_key(tech.via_key(lower_n))
        except Exception as exc:
            raise ValueError(
                f"incomplete PDK routing stack; VIA{lower_n} is missing"
            ) from exc
        if via_key in via_between:
            raise ValueError("PDK routing stack reuses a via layer key")
        via_between[via_key] = (metals[lower_n - 1], metals[lower_n])
    return metal_by_key, axis_by_key, via_between


def measure_layer_route_profile(
    gds_path: str,
    tech: Any,
    *,
    tagged: Optional[Sequence[Any]],
    candidate_pool: Any = None,
    resource_plan: Any = None,
    materialization_certificate: Any = None,
    device_groups: Optional[Sequence[DeviceGroupFootprint]] = None,
    mim_footprints_nm: Optional[Sequence[BBoxNm]] = None,
    max_metal: int = 8,
) -> LayerRouteProfileReport:
    """Measure F2 layer use from exact GDS plus exhaustive route ownership.

    ``tagged`` is the canonical router-owned ``(net, layer_key, bbox_nm)``
    manifest.  Its boxes identify ownership only; area is always clipped to the
    physical conductor in ``gds_path``.  Preferred axes come solely from
    ``tech``.  A complete planner triple deep-replays the legacy selected-plan
    lineage.  An all-``None`` triple measures negotiated geometry directly from
    the same exact GDS and manifest; a partial triple is rejected rather than
    inventing planner provenance.
    """
    artifact_digest = ""
    manifest_digest = ""
    route_tags: Tuple[Any, ...] = ()
    try:
        from clay.layout.route.via_provenance_check import (
            emitted_artifact_sha256,
            route_manifest_sha256,
        )

        if tagged is None:
            raise ValueError("route ownership manifest is unavailable")
        route_tags = _bounded_sequence(tagged, _MAX_PROFILE_TAGS, "route ownership tags")
        manifest_digest = route_manifest_sha256(route_tags)

        metal_by_key, axis_by_key, via_between = _profile_pdk_stack(tech, max_metal)

        # Once the PDK declares a coherent conductor ladder, bind the requested
        # immutable GDS before planner replay or geometry measurement.  Every
        # later unmeasurable observation therefore still identifies the exact
        # bytes and ownership manifest it attempted to measure.
        artifact_digest = emitted_artifact_sha256(gds_path)

        if device_groups is None:
            raise ValueError("device-group footprint evidence is unavailable")
        if mim_footprints_nm is None:
            raise ValueError("MIM footprint evidence is unavailable")
        groups = _bounded_sequence(device_groups, _MAX_PROFILE_GROUPS, "device groups")
        if any(not isinstance(group, DeviceGroupFootprint) for group in groups):
            raise TypeError("device_groups must contain DeviceGroupFootprint values")
        group_ids = tuple(sorted({group.group_id for group in groups}))
        group_box_count = sum(len(group.bboxes_nm) for group in groups)
        if group_box_count > _MAX_PROFILE_GROUP_BOX_COUNT:
            raise ValueError(
                "device group boxes exceed cumulative bounded limit "
                f"{_MAX_PROFILE_GROUP_BOX_COUNT}"
            )
        mim_boxes = tuple(
            _profile_bbox(box, "MIM footprint")
            for box in _bounded_sequence(
                mim_footprints_nm, _MAX_PROFILE_GROUP_BOXES, "MIM footprints"
            )
        )

        planner_inputs = (
            candidate_pool,
            resource_plan,
            materialization_certificate,
        )
        if all(value is None for value in planner_inputs):
            lineage = RouteProfileLineage.NEGOTIATED_GEOMETRY
            allocation_by_key = {}
            expected_geometry = ()
            pool_hash = ""
            plan_hash = ""
        elif all(value is not None for value in planner_inputs):
            lineage = RouteProfileLineage.EXACT_RESOURCE_PLAN
            (
                allocation_by_key,
                expected_geometry,
                pool_hash,
                plan_hash,
            ) = _planner_layer_allocations(
                candidate_pool,
                resource_plan,
                materialization_certificate,
                tech,
                metal_by_key,
                axis_by_key,
            )
        else:
            raise ValueError(
                "candidate pool, resource plan, and materialization certificate "
                "must be all present or all absent"
            )

        kdb, layout, top, _li = _open_top(str(gds_path))
        # F2 binds exact (layer, datatype) keys; unlike the aggregate legacy
        # metric, it must not merge all datatypes sharing a layer number.
        physical_by_key = {}
        for key in tuple(metal_by_key) + tuple(via_between):
            li = layout.find_layer(*key)
            region = kdb.Region() if li is None else kdb.Region(top.begin_shapes_rec(li))
            region.merge()
            physical_by_key[key] = region

        nodes = []
        manifest_geometry: list[Tuple[str, LayerKey, BBoxNm]] = []
        unresolved = []
        for index, entry in enumerate(route_tags):
            try:
                if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                    raise TypeError("tag must be (net, layer_key, bbox_nm)")
                net = str(entry[0]).strip()
                if not net:
                    raise ValueError("tag has unknown owner")
                key = _profile_layer_key(entry[1])
                box = _profile_bbox(entry[2], "route ownership bbox")
                if key not in metal_by_key and key not in via_between:
                    raise ValueError(
                        f"router-owned layer {key} is absent from the complete PDK routing stack"
                    )
                canonical = (net, key, box)
                ownership_region = _profile_box_region(kdb, layout, box)
                region = physical_by_key[key] & ownership_region
                region.merge()
                if region.is_empty():
                    raise ValueError("ownership bbox matches no emitted conductor")
                if not (ownership_region - physical_by_key[key]).is_empty():
                    raise ValueError(
                        "ownership bbox is not fully covered by emitted conductor"
                    )
                nodes.append((net, key, box, region, key in metal_by_key))
                manifest_geometry.append(canonical)
            except Exception as exc:
                unresolved.append(f"tag[{index}]: {exc}")
        if unresolved:
            raise ValueError("; ".join(unresolved))
        if lineage is RouteProfileLineage.EXACT_RESOURCE_PLAN:
            expected_counter = Counter(expected_geometry)
            manifest_counter = Counter(manifest_geometry)
            if manifest_counter != expected_counter:
                missing = tuple(sorted((expected_counter - manifest_counter).elements()))
                extra = tuple(sorted((manifest_counter - expected_counter).elements()))
                raise ValueError(
                    "route ownership manifest does not exactly cover selected "
                    f"materialization; missing={missing[:8]} extra={extra[:8]}"
                )
        metal_nodes = [node for node in nodes if node[4]]
        if not metal_nodes:
            raise ValueError("route manifest contains no emitted router-owned metal")

        active = kdb.Region()
        active_keys = []
        for kind in ("diff", "ndiff", "pdiff", "poly"):
            try:
                key = tech.build_layer(kind)
            except Exception:
                key = None
            if key is not None:
                normalised = _profile_layer_key(key)
                if normalised not in active_keys:
                    active_keys.append(normalised)
        for key in active_keys:
            li = layout.find_layer(*key)
            if li is not None:
                active += kdb.Region(top.begin_shapes_rec(li))
        active.merge()
        mim = kdb.Region()
        for box in mim_boxes:
            mim += _profile_box_region(kdb, layout, box)
        mim.merge()

        profiles = []
        actual_keys = {node[1] for node in metal_nodes}
        for key in sorted(actual_keys | set(allocation_by_key), key=lambda k: metal_by_key[k]):
            layer_n = metal_by_key[key]
            if key not in axis_by_key:
                authority_owner = (
                    "router-owned and selected"
                    if key in actual_keys and key in allocation_by_key
                    else "router-owned"
                    if key in actual_keys
                    else "selected"
                )
                raise ValueError(
                    f"{authority_owner} metal{layer_n} has no PDK "
                    "preferred-axis declaration"
                )
            layer_nodes = [node for node in metal_nodes if node[1] == key]
            route = kdb.Region()
            directional = kdb.Region()
            preferred = kdb.Region()
            long_runs = set()
            axis = axis_by_key[key]
            for net, _key, box, region, _is_metal in layer_nodes:
                route += region
                width = box[2] - box[0]
                height = box[3] - box[1]
                if width != height:
                    directional += region
                    shape_axis = "H" if width > height else "V"
                    if shape_axis == axis:
                        preferred += region
                    if max(width, height) / min(width, height) >= _LONG_RUN_ASPECT_MIN:
                        long_runs.add((net, box))
            route.merge()
            directional.merge()
            preferred.merge()
            route_nm2 = float(route.area())
            directional_nm2 = float(directional.area())
            over_nm2 = float((route & active).area())
            non_mim_nm2 = float((route - mim).area())
            allocation = None
            negotiated_geometry = None
            if lineage is RouteProfileLineage.EXACT_RESOURCE_PLAN:
                allocation = allocation_by_key.get(
                    key,
                    PlannerLayerAllocation(
                        layer_n=layer_n,
                        layer_key=key,
                        use_count=0,
                        track_count=0,
                        area_um2=0.0,
                        nets=(),
                        candidate_pool_hash=pool_hash,
                    ),
                )
            else:
                negotiated_geometry = NegotiatedLayerGeometry(
                    layer_n=layer_n,
                    layer_key=key,
                    shape_count=len(layer_nodes),
                    covered_shape_count=len(layer_nodes),
                    area_um2=route_nm2 * layout.dbu * layout.dbu,
                    nets=tuple(sorted({node[0] for node in layer_nodes})),
                    route_manifest_sha256=manifest_digest,
                )
            profiles.append(
                LayerRouteProfile(
                    layer_n=layer_n,
                    layer_key=key,
                    preferred_axis=axis,
                    route_owned_area_um2=route_nm2 * layout.dbu * layout.dbu,
                    over_active_area_um2=over_nm2 * layout.dbu * layout.dbu,
                    over_active_fraction=(over_nm2 / route_nm2 if route_nm2 else None),
                    preferred_axis_purity=(
                        float(preferred.area()) / directional_nm2
                        if directional_nm2
                        else None
                    ),
                    long_run_count=len(long_runs),
                    non_mim_area_um2=non_mim_nm2 * layout.dbu * layout.dbu,
                    planner_allocation=allocation,
                    negotiated_geometry=negotiated_geometry,
                )
            )

        # Physical route connectivity graph: same-layer conductor contact plus
        # declared via cuts joining only their PDK lower/upper layers.
        parent = list(range(len(nodes)))
        join_budget = _ProfileJoinBudget(_MAX_PROFILE_JOIN_EVALUATIONS)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[a] = b

        def connect_touching(left_indexes: Sequence[int], right_indexes: Sequence[int]) -> None:
            """Sweep broad phase before exact Region contact (never all-pairs)."""
            symmetric = left_indexes is right_indexes
            ordered_left = sorted(left_indexes, key=lambda i: (nodes[i][2][0], i))
            ordered_right = sorted(right_indexes, key=lambda i: (nodes[i][2][0], i))
            active_right: List[int] = []
            cursor = 0
            for left in ordered_left:
                left_box = nodes[left][2]
                while (
                    cursor < len(ordered_right)
                    and nodes[ordered_right[cursor]][2][0] <= left_box[2] + 1
                ):
                    active_right.append(ordered_right[cursor])
                    cursor += 1
                retained_right = []
                for right in active_right:
                    join_budget.consume("route-component sweep maintenance")
                    if nodes[right][2][2] + 1 >= left_box[0]:
                        retained_right.append(right)
                active_right = retained_right
                for right in active_right:
                    if right == left:
                        continue
                    if symmetric and right < left:
                        continue
                    join_budget.consume("route-component geometry join")
                    right_box = nodes[right][2]
                    if (
                        right_box[3] + 1 < left_box[1]
                        or left_box[3] + 1 < right_box[1]
                    ):
                        continue
                    if _profile_region_touches(nodes[left][3], nodes[right][3]):
                        union(left, right)

        indexes_by_key: Dict[LayerKey, List[int]] = {}
        for index, node in enumerate(nodes):
            indexes_by_key.setdefault(node[1], []).append(index)
        for key in metal_by_key:
            same_layer = indexes_by_key.get(key, ())
            connect_touching(same_layer, same_layer)
        for via_key, (lower_key, upper_key) in via_between.items():
            via_indexes = indexes_by_key.get(via_key, ())
            connect_touching(via_indexes, indexes_by_key.get(lower_key, ()))
            connect_touching(via_indexes, indexes_by_key.get(upper_key, ()))

        components: Dict[int, List[int]] = {}
        for index in range(len(nodes)):
            components.setdefault(find(index), []).append(index)
        group_regions = []
        for group in groups:
            group_region = kdb.Region()
            for box in group.bboxes_nm:
                group_region += _profile_box_region(kdb, layout, box)
            group_region &= active
            group_regions.append((group.group_id, group_region))

        cross = []
        for ordinal, indexes in enumerate(
            sorted(components.values(), key=lambda xs: tuple(xs))
        ):
            metal_indexes = [index for index in indexes if nodes[index][4]]
            if not metal_indexes:
                continue
            touched = set()
            for group_id, group_region in group_regions:
                for index in metal_indexes:
                    join_budget.consume("route-component-to-device-group join")
                    if _profile_region_touches(nodes[index][3], group_region):
                        touched.add(group_id)
                        break
            if len(touched) < 2:
                continue
            area_nm2 = 0.0
            for key in sorted({nodes[index][1] for index in metal_indexes}):
                component_region = kdb.Region()
                for index in metal_indexes:
                    if nodes[index][1] == key:
                        component_region += nodes[index][3]
                component_region.merge()
                area_nm2 += float(component_region.area())
            cross.append(
                CrossDeviceRouteComponent(
                    component_id=f"route-component-{ordinal:04d}",
                    nets=tuple(sorted({nodes[index][0] for index in indexes})),
                    device_group_ids=tuple(sorted(touched)),
                    layer_keys=tuple(sorted({nodes[index][1] for index in metal_indexes})),
                    route_owned_area_um2=area_nm2 * layout.dbu * layout.dbu,
                )
            )

        unplanned = ()
        if lineage is RouteProfileLineage.EXACT_RESOURCE_PLAN:
            unplanned = tuple(
                sorted(
                    metal_by_key[key]
                    for key in actual_keys
                    if allocation_by_key.get(key) is None
                    or allocation_by_key[key].materialized_shape_count <= 0
                )
            )
        planned_lineage = lineage is RouteProfileLineage.EXACT_RESOURCE_PLAN
        lineage_report = {
            "evidence_lineage": lineage.value,
            "planner_evidence_applicable": planned_lineage,
            "ownership_manifest_verified": True,
            "manifest_covered_entry_count": len(manifest_geometry),
            "exact_geometry_verified": True,
        }
        if unplanned:
            return LayerRouteProfileReport(
                status="fail",
                layers=tuple(profiles),
                independent_device_group_ids=group_ids,
                cross_device_components=tuple(cross),
                unplanned_layer_numbers=unplanned,
                manifest_available=True,
                manifest_entry_count=len(manifest_geometry),
                expected_materialized_shape_count=(
                    len(expected_geometry) if planned_lineage else 0
                ),
                candidate_pool_hash=pool_hash,
                resource_plan_hash=plan_hash,
                artifact_sha256=artifact_digest,
                route_manifest_sha256=manifest_digest,
                plan_verified=True,
                materialization_verified=True,
                device_group_evidence_available=True,
                mim_evidence_available=True,
                **lineage_report,
                reason=(
                    "router-owned metal lacks planner capacity allocation on metal "
                    + ", ".join(str(layer_n) for layer_n in unplanned)
                ),
            )
        if len(group_ids) >= 2 and not cross:
            return LayerRouteProfileReport(
                status="fail",
                layers=tuple(profiles),
                independent_device_group_ids=group_ids,
                cross_device_components=(),
                manifest_available=True,
                manifest_entry_count=len(manifest_geometry),
                expected_materialized_shape_count=len(expected_geometry),
                candidate_pool_hash=pool_hash,
                resource_plan_hash=plan_hash,
                artifact_sha256=artifact_digest,
                route_manifest_sha256=manifest_digest,
                device_group_evidence_available=True,
                mim_evidence_available=True,
                plan_verified=planned_lineage,
                materialization_verified=planned_lineage,
                **lineage_report,
                reason=(
                    "cross-device routing is applicable to two or more independent "
                    "device groups, but no exact route-owned component touches at "
                    "least two groups"
                ),
            )
        return LayerRouteProfileReport(
            status="pass",
            layers=tuple(profiles),
            independent_device_group_ids=group_ids,
            cross_device_components=tuple(cross),
            manifest_available=True,
            manifest_entry_count=len(manifest_geometry),
            expected_materialized_shape_count=(
                len(expected_geometry) if planned_lineage else 0
            ),
            candidate_pool_hash=pool_hash,
            resource_plan_hash=plan_hash,
            artifact_sha256=artifact_digest,
            route_manifest_sha256=manifest_digest,
            plan_verified=planned_lineage,
            materialization_verified=planned_lineage,
            device_group_evidence_available=True,
            mim_evidence_available=True,
            **lineage_report,
            reason=(
                "all route-owned metal is measured against PDK axes and exact "
                "resource-plan capacity"
                if planned_lineage
                else "all negotiated route-owned metal is covered by the exact GDS "
                "ownership manifest and measured against PDK axes"
            ),
        )
    except Exception as exc:
        return LayerRouteProfileReport(
            status="unmeasurable",
            manifest_available=tagged is not None,
            manifest_entry_count=len(route_tags),
            artifact_sha256=artifact_digest,
            route_manifest_sha256=manifest_digest,
            reason=str(exc),
        )


# --------------------------------------------------------------------------- #
# per-metric verdicts (each reads its band from config -- no embedded constant).
# --------------------------------------------------------------------------- #
def verdict_enclosure(value: float,
                      th: Optional[FidelityThresholds] = None) -> MetricVerdict:
    th = th or fidelity_thresholds()
    sat = value >= th.enclosure_min
    return MetricVerdict(
        name="enclosure_efficiency", value=value, expected=None,
        threshold=f">= {th.enclosure_min} (CLAY_FIDELITY_ENCLOSURE_MIN)",
        satisfied=sat, observable=True,
        detail=f"device-OD inside rings / ring window = {value:.3f}",
    )


def verdict_aspect(value: float,
                   th: Optional[FidelityThresholds] = None) -> MetricVerdict:
    th = th or fidelity_thresholds()
    sat = th.aspect_lo <= value <= th.aspect_hi
    return MetricVerdict(
        name="aspect_band", value=value, expected=None,
        threshold=f"[{th.aspect_lo}, {th.aspect_hi}] "
                  "(CLAY_FIDELITY_ASPECT_LO/HI)",
        satisfied=sat, observable=True,
        detail=f"cell W/H = {value:.3f}",
    )


def verdict_fill(value: float,
                 th: Optional[FidelityThresholds] = None) -> MetricVerdict:
    th = th or fidelity_thresholds()
    sat = value >= th.fill_min
    return MetricVerdict(
        name="fill_density", value=value, expected=None,
        threshold=f">= {th.fill_min} (CLAY_FIDELITY_FILL_MIN)",
        satisfied=sat, observable=True,
        detail=f"OD area / cell area = {value:.3f}",
    )


def verdict_ring_granularity(
    ring_count: int, expected_groups: int,
    th: Optional[FidelityThresholds] = None,
) -> MetricVerdict:
    th = th or fidelity_thresholds()
    # RED when the ring scheme is too COARSE: fewer rings than the design's
    # compact-matched-group count (minus a tolerance). 3 huge band-rings vs ~9
    # per-group rings is RED.
    sat = ring_count >= (expected_groups - th.ring_tol)
    return MetricVerdict(
        name="ring_granularity", value=float(ring_count),
        expected=float(expected_groups),
        threshold=f">= compact_groups - {th.ring_tol} "
                  "(CLAY_FIDELITY_RING_TOL)",
        satisfied=sat, observable=True,
        detail=(f"{ring_count} ring frames vs {expected_groups} compact "
                f"matched groups"),
    )


def verdict_contact_density(
    ratio: float, n_cont: int, fingers: int,
    th: Optional[FidelityThresholds] = None,
) -> MetricVerdict:
    th = th or fidelity_thresholds()
    observable = fingers > 0
    if not observable:
        return MetricVerdict(
            name="contact_density", value=None, expected=None,
            threshold=f">= {th.cont_per_finger_min} "
                      "(CLAY_FIDELITY_CONT_PER_FINGER_MIN)",
            satisfied=None, observable=False,
            detail="design declares no MOS finger to measure contacts per",
        )
    sat = ratio >= th.cont_per_finger_min
    return MetricVerdict(
        name="contact_density", value=ratio, expected=None,
        threshold=f">= {th.cont_per_finger_min} "
                  "(CLAY_FIDELITY_CONT_PER_FINGER_MIN)",
        satisfied=sat, observable=True,
        detail=(f"{n_cont} CONT / {fingers} MOS fingers = "
                f"{ratio:.1f} per finger"),
    )


def verdict_bbox_fill(value: float,
                      th: Optional[FidelityThresholds] = None) -> MetricVerdict:
    th = th or fidelity_thresholds()
    sat = value >= th.bbox_fill_min
    return MetricVerdict(
        name="bbox_fill", value=value, expected=None,
        threshold=f">= {th.bbox_fill_min} (CLAY_FIDELITY_BBOX_FILL_MIN)",
        satisfied=sat, observable=True,
        detail=f"sum device-footprint bbox / cell bbox = {value:.3f}",
    )


def verdict_row_balance(
    ratio: Optional[float], n_rows: int,
    th: Optional[FidelityThresholds] = None,
) -> MetricVerdict:
    th = th or fidelity_thresholds()
    if ratio is None:
        return MetricVerdict(
            name="row_balance", value=None, expected=None,
            threshold=f"<= {th.row_balance_max} (CLAY_FIDELITY_ROW_BALANCE_MAX)",
            satisfied=None, observable=False,
            detail=(f"{n_rows} device row(s) -- need >= 2 rows with positive "
                    "span to measure a row-balance ratio"),
        )
    sat = ratio <= th.row_balance_max
    return MetricVerdict(
        name="row_balance", value=ratio, expected=None,
        threshold=f"<= {th.row_balance_max} (CLAY_FIDELITY_ROW_BALANCE_MAX)",
        satisfied=sat, observable=True,
        detail=(f"widest/narrowest device-row x-span = {ratio:.3f} "
                f"over {n_rows} rows"),
    )


def verdict_dead_region(
    frac: float, rect_um: Tuple[float, float, float, float],
    th: Optional[FidelityThresholds] = None,
) -> MetricVerdict:
    """GENERIC mosaic-completeness verdict (RETIRES the cap-center proxy): the
    largest contiguous DEAD rectangle (empty of any element footprint) must be a
    SMALL fraction of the cell. A dense mosaic packs every element with no parked
    side/top slab; a cap dumped as a side block leaves a big contiguous dead
    region. No cap special-case -- the cap is just an occupied tile."""
    th = th or fidelity_thresholds()
    sat = frac <= th.dead_region_max
    return MetricVerdict(
        name="dead_region", value=frac, expected=None,
        threshold=f"<= {th.dead_region_max} (CLAY_FIDELITY_DEAD_REGION_MAX)",
        satisfied=sat, observable=True,
        detail=(f"largest contiguous dead rect = {frac:.3f} of cell "
                f"(rect {rect_um} um)"),
    )


def verdict_over_cell(
    frac: Optional[float], over_um2: float, route_um2: float,
    th: Optional[FidelityThresholds] = None,
) -> MetricVerdict:
    """ROUTER-CONTRACT verdict: the routing trunk must run OVER the device cells.

    UNOBSERVABLE (satisfied None, observable False) when the layout carries no
    route-metal -- never a silent pass. FAIL (blocking) when the over-cell
    fraction is BELOW the reference-calibrated floor: the reference routes 0.577
    of its trunk over the cells; the apron-sprawl default measures ~0.24. This
    NAMES the over-cell router-capability gap; it does NOT close it (closing it
    is the router-capability wave that runs trunk metal over the device cells)."""
    th = th or fidelity_thresholds()
    if frac is None:
        return MetricVerdict(
            name="over_cell_fraction", value=None, expected=None,
            threshold=f">= {th.over_cell_min} (CLAY_FIDELITY_OVER_CELL_MIN)",
            satisfied=None, observable=False,
            detail=("no routing-trunk metal (M2/M3) in the layout -- over-cell "
                    "fraction unmeasurable (place-only / unrouted GDS)"),
        )
    sat = frac >= th.over_cell_min
    return MetricVerdict(
        name="over_cell_fraction", value=frac, expected=None,
        threshold=f">= {th.over_cell_min} (CLAY_FIDELITY_OVER_CELL_MIN)",
        satisfied=sat, observable=True,
        detail=(f"routing-trunk (M2/M3) over device-active / route-metal = "
                f"{frac:.3f} ({over_um2:.1f} / {route_um2:.1f} um^2)"),
    )


# --------------------------------------------------------------------------- #
# wirelength locality (the routing/floorplan QUALITY obligation).
#
# Source = the canonical AsBuiltBundle's ``pins_by_net`` (the SAME routing
# terminals lvs_gate emits + LVS reads -- no second path). HPWL per net; a
# LOW-FANOUT signal net (touching <= signal_fanout_max distinct devices) must
# stay LOCAL. Power/ground nets (high fanout) intentionally distribute and are
# classified out of the gate (reported separately, never silently passed).
# --------------------------------------------------------------------------- #
def _pin_xy(p) -> Optional[Tuple[float, float]]:
    xy = getattr(p, "xy_um", None)
    if xy is None and isinstance(p, (tuple, list)) and len(p) >= 2:
        # bare (x, y) or (..., (x,y)); accept a trailing 2-tuple too.
        if isinstance(p[-1], (tuple, list)) and len(p[-1]) == 2:
            xy = p[-1]
        else:
            xy = (p[0], p[1])
    if xy is None:
        return None
    return (float(xy[0]), float(xy[1]))


def _pin_device(p) -> str:
    d = getattr(p, "device", None)
    if d is None and isinstance(p, (tuple, list)) and len(p) >= 2:
        d = p[1]
    return str(d)


def measure_wirelength(
    pins_by_net: Dict[str, Any],
    extent_wh: Tuple[float, float],
    th: Optional[FidelityThresholds] = None,
) -> Dict[str, Any]:
    """Measure per-net HPWL locality from the canonical ``pins_by_net``.

    A net's fanout is the number of DISTINCT devices it touches (not raw pins --
    a multi-finger device contributes many same-net pins). Nets touching
    ``<= signal_fanout_max`` devices are SIGNAL nets (locality-gated); the rest
    are power/distribution nets (exempt from the max-span gate, reported only).

    ``span_frac`` = HPWL / (W + H): a scale-free fraction of the cell
    half-perimeter, so the band is a RELATIVE threshold (no coordinate).

    Returns a dict (never raises on an odd pin shape -- a net with < 2 placed
    pins is skipped, honest, not a silent fake).
    """
    th = th or fidelity_thresholds()
    W, H = float(extent_wh[0]), float(extent_wh[1])
    perim = W + H
    fmax = int(th.signal_fanout_max)

    total_hpwl = 0.0
    signal: list = []   # (span_frac, hpwl, net, n_dev)
    power: list = []
    for net, pins in (pins_by_net or {}).items():
        xys = [_pin_xy(p) for p in pins]
        xys = [c for c in xys if c is not None]
        if len(xys) < 2:
            continue
        devs = {_pin_device(p) for p in pins}
        xs = [c[0] for c in xys]
        ys = [c[1] for c in xys]
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        total_hpwl += hpwl
        frac = hpwl / perim if perim > 0 else 0.0
        rec = (frac, hpwl, str(net), len(devs))
        (signal if len(devs) <= fmax else power).append(rec)

    signal.sort(reverse=True)
    worst = signal[0] if signal else None
    return {
        "total_hpwl": total_hpwl,
        "max_signal_span_frac": (worst[0] if worst else None),
        "worst_signal_net": (worst[2] if worst else None),
        "worst_signal_hpwl": (worst[1] if worst else None),
        "n_signal_nets": len(signal),
        "n_power_nets": len(power),
        "signal_spans": signal,
    }


def verdict_wirelength(
    measured: Dict[str, Any],
    th: Optional[FidelityThresholds] = None,
) -> MetricVerdict:
    """QUALITY verdict for wirelength locality. None/unobservable when no signal
    net is measurable (never a silent pass); FAIL when the worst low-fanout
    signal net spans more than ``wirelength_span_max`` of the cell perimeter."""
    th = th or fidelity_thresholds()
    frac = measured.get("max_signal_span_frac")
    if frac is None:
        return MetricVerdict(
            name="wirelength_locality", value=None, expected=None,
            threshold=(f"<= {th.wirelength_span_max} "
                       "(CLAY_FIDELITY_WIRELENGTH_SPAN_MAX)"),
            satisfied=None, observable=False,
            detail=(f"{measured.get('n_power_nets', 0)} power/distribution net(s), "
                    "0 low-fanout signal net(s) -- nothing to gate locality on"),
        )
    sat = frac <= th.wirelength_span_max
    return MetricVerdict(
        name="wirelength_locality", value=frac, expected=None,
        threshold=(f"<= {th.wirelength_span_max} "
                   "(CLAY_FIDELITY_WIRELENGTH_SPAN_MAX)"),
        satisfied=sat, observable=True,
        detail=(f"worst signal net {measured.get('worst_signal_net')!r} "
                f"HPWL={measured.get('worst_signal_hpwl'):.1f}um = {frac:.3f} of "
                f"cell perimeter ({measured.get('n_signal_nets')} signal / "
                f"{measured.get('n_power_nets')} power nets)"),
    )


# --------------------------------------------------------------------------- #
# the aggregate measurement (one canonical pass over the emitted GDS + intent).
# --------------------------------------------------------------------------- #
def measure_fidelity(gds_path: str, intent: Any,
                     th: Optional[FidelityThresholds] = None,
                     pins_by_net: Optional[Dict[str, Any]] = None,
                     extent_wh: Optional[Tuple[float, float]] = None,
                     ) -> FidelityReport:
    """Measure all geometric-fidelity metrics over ONE emitted GDS.

    The emitted GDS is the single canonical artifact (the one the LVS/DRC
    oracles ran on). ``intent`` supplies topology only (the compact-matched-
    group count + which devices are caps). The reference is NOT an input; the
    bands come from config (``th`` or the resolved default).

    ``pins_by_net`` (+ ``extent_wh``, else read from the GDS) is the canonical
    AsBuiltBundle routing-terminal map; when supplied the WIRELENGTH-LOCALITY
    metric is appended (the routing/floorplan obligation). Absent, it is simply
    not measured (the GDS alone cannot recover net membership) -- never a silent
    pass, the metric just does not appear.
    """
    th = th or fidelity_thresholds()

    area = measure_area_um2(gds_path)
    aspect = measure_aspect(gds_path)
    fill = measure_fill(gds_path)
    enclosure, ring_count = measure_enclosure(gds_path)
    expected_groups = compact_matched_group_count(intent)
    dead_frac, dead_rect = measure_dead_region(gds_path)
    cont_ratio, n_cont, n_fingers = measure_contact_density(gds_path, intent)
    bbox_fill = measure_bbox_fill(gds_path)
    row_bal, _n_rows = measure_row_balance(gds_path)
    oc_frac, oc_over, oc_route = measure_over_cell_fraction(gds_path)

    metrics = (
        verdict_enclosure(enclosure, th),
        verdict_ring_granularity(ring_count, expected_groups, th),
        verdict_fill(fill, th),
        verdict_aspect(aspect, th),
        verdict_dead_region(dead_frac, dead_rect, th),
        verdict_contact_density(cont_ratio, n_cont, n_fingers, th),
        verdict_bbox_fill(bbox_fill, th),
        verdict_row_balance(row_bal, _n_rows, th),
        verdict_over_cell(oc_frac, oc_over, oc_route, th),
    )
    if pins_by_net is not None:
        ext = extent_wh
        if ext is None:
            # cell perimeter from the routing-terminal bbox (the SAME currency
            # the wirelength band was calibrated against -- a pin-bbox W+H, not
            # the device-inclusive GDS bbox).
            axs = [_pin_xy(p)[0] for ps in pins_by_net.values() for p in ps
                   if _pin_xy(p) is not None]
            ays = [_pin_xy(p)[1] for ps in pins_by_net.values() for p in ps
                   if _pin_xy(p) is not None]
            ext = ((max(axs) - min(axs), max(ays) - min(ays))
                   if axs else (0.0, 0.0))
        wl = measure_wirelength(pins_by_net, ext, th)
        metrics = metrics + (verdict_wirelength(wl, th),)
    return FidelityReport(
        area_um2=area, aspect=aspect, fill=fill, enclosure=enclosure,
        ring_count=ring_count, bbox_fill=bbox_fill, row_balance=row_bal,
        dead_region=dead_frac, metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# the SpecPack: the five metrics as QUALITY-tier obligations (the empty shell
# the construction plan named -- filled here, gated in F1).
# --------------------------------------------------------------------------- #
def _detect_whole_design(intent):
    """The geometric-fidelity metrics are CELL-LEVEL: one obligation per metric
    over the whole placed design (not per device-pair). The detector yields one
    group naming the cell so apply_packs records exactly one obligation per
    metric. Pure topology (the MOS member names); no coordinate."""
    members = tuple(str(d.name)
                    for d in (getattr(intent, "devices", None) or []))
    return [members] if members else []


def _emit_noop(intent, group, net):
    """The fidelity metrics verdict from the EMITTED GDS (measure_fidelity), not
    from a zero-coordinate placement relation, so the pack emitter adds no QSR
    hyperedge. The obligation ledger entry (apply_packs) carries the metric
    provenance; F1 wires the GDS-measured verdict into signoff."""
    return None


def geometric_fidelity_pack():
    """The geometric-fidelity SpecPack (five QUALITY-tier obligations).

    Each rule is ``fidelity`` hardness (the F1 gating geometric-quality class)
    and derives to the QUALITY tier -- NOT advisory. A RED measurement is a
    BLOCKING FAIL (like DRC/LVS), so a legal-but-geometrically-wrong layout
    cannot be reported green. The pack DECLARES the obligations + names their
    checkers; the verdict is produced by :func:`measure_fidelity` over the
    emitted GDS and bridged to signoff CheckerVerdicts by
    :mod:`clay.layout.spec.fidelity_signoff` (F1).
    """
    from clay.layout.spec.packs import SpecPack, SpecRule

    def rule(name, metric, effect, checker):
        return SpecRule(
            name=name, metric=metric, effect=effect, object="cell",
            hardness="fidelity", detect=_detect_whole_design,
            emit=_emit_noop, checker=checker, tier="QUALITY",
        )

    return SpecPack(
        name="geometric_fidelity",
        applies_to=["*"],
        rules=[
            rule("enclosure_efficiency", "enclosure", "ring_efficiency",
                 "fidelity_enclosure"),
            rule("ring_granularity", "ring_count", "ring_granularity",
                 "fidelity_ring_granularity"),
            rule("fill_density", "fill", "area_efficiency",
                 "fidelity_fill"),
            rule("aspect_band", "aspect", "floorplan_aspect",
                 "fidelity_aspect"),
            rule("dead_region", "dead_region", "mosaic_completeness",
                 "fidelity_dead_region"),
            rule("contact_density", "cont_per_finger", "contact_density",
                 "fidelity_contact_density"),
            rule("bbox_fill", "bbox_fill", "placement_occupancy",
                 "fidelity_bbox_fill"),
            rule("row_balance", "row_balance", "placement_row_balance",
                 "fidelity_row_balance"),
            rule("over_cell_fraction", "over_cell", "resource_plan_coverage",
                 "fidelity_over_cell"),
        ],
    )


__all__ = [
    "MetricVerdict",
    "FidelityReport",
    "compact_matched_group_count",
    "measure_area_um2",
    "measure_aspect",
    "measure_fill",
    "measure_enclosure",
    "measure_dead_region",
    "measure_contact_density",
    "sd_column_heights",
    "tall_column_median",
    "measure_bbox_fill",
    "measure_row_balance",
    "measure_over_cell_fraction",
    "DeviceGroupFootprint",
    "PlannerLayerAllocation",
    "PlannerTrackUseEvidence",
    "LayerRouteProfile",
    "CrossDeviceRouteComponent",
    "LayerRouteProfileReport",
    "measure_layer_route_profile",
    "verdict_over_cell",
    "measure_wirelength",
    "verdict_wirelength",
    "total_mos_fingers",
    "measure_fidelity",
    "verdict_enclosure",
    "verdict_aspect",
    "verdict_fill",
    "verdict_ring_granularity",
    "verdict_dead_region",
    "verdict_contact_density",
    "verdict_bbox_fill",
    "verdict_row_balance",
    "geometric_fidelity_pack",
    "cell_extent_um",
]
