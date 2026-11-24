"""Floorplan geometric-fidelity metric bank -- the STRUCTURAL convergence axis the
scalar (area / LVS / contact) gate was blind to.

ROOT-CURE STEP D. The scalar signoff gate answered ONE question -- is the layout
LEGAL (DRC / LVS clean) and small (area) -- and nothing gated the FLOORPLAN
STRUCTURE. So an area-driven re-pack (``maybe_envelope_replace``) could scramble
the reference's banded analog floorplan (NMOS band / cap band / PMOS band,
diff pairs on a common centroid axis, the two Miller caps colocated) and still
report green: the layout lost its spatial intent because no metric measured it.
This module MECHANISES that lost intent into four measured, 5-state metrics so the
tier-/axis-/cap-preserving placement fix has a RED -> GREEN checker.

The four floorplan-fidelity metrics
-----------------------------------
    tier_interleave       : fraction of NMOS/PMOS devices sitting on the WRONG
                            side of the N/P band midline. 0.0 = cleanly banded
                            (the reference intent); a scrambled compact ~0.23.
                            (Same logic as ``ota_dev/analysis/tier_structure``.)
    cap_colocation        : the two MIM caps must sit in ONE band (same y) AND be
                            x-adjacent. Measures the centroid dy (band separation,
                            normalised by cap height) and the edge-to-edge x gap
                            (adjacency, normalised by cap width). The reference
                            colocates them (same y, adjacent); ours scatters them.
    symmetry_axis         : every declared @symmetric / matched (diff / mirror)
                            pair must share ONE common vertical axis. Measures the
                            per-pair midpoint-x SPREAD normalised by cell width.
                            0 = one shared axis; a scattered placement is large.
    current_path_locality : devices adjacent in the declared signal path must be
                            spatially close. Measures per-signal-net HPWL / cell
                            half-perimeter; gates the WORST low-fanout signal net.

Two non-negotiable design rules (the leakage line + the optimizer seam)
-----------------------------------------------------------------------
  * LEAKAGE LINE -- every metric is measured from OUR OWN geometry: the in-process
    ``place_generic`` / ``maybe_envelope_replace`` placement centroids, or the
    on-disk EMITTED GDS (``ota_dev/out/ota_tsmc180rf.clay_generic.gds``, MIM plates
    on layer 67). A coordinate from the REFERENCE layout is NEVER a metric input.
    The reference contributes ONLY the calibrated band SCALARS (below); it cannot
    leak a position, because the answer (the reference layout) is not an input.
    The topology inputs (matched pairs, netlist connectivity) are pure topology
    (device names / net names), so every metric is translation- AND scale-invariant.

  * THRESHOLD = CONFIG PARAMETER -- every band edge resolves through
    ``floorplan_thresholds()`` (env ``CLAY_FLOORPLAN_*`` -> reference-calibrated
    default), the SAME env -> default seam ``clay.config.fidelity_thresholds`` uses
    for the geometric-fidelity bands. A future optimizer (tSS-BO / adjoint /
    ``clay tune``) auto-calibrates a band by exporting an env var; no source edit.
    (Folding these into ``clay.config.FidelityThresholds`` is the graduation step
    when this measure-only bank starts GATING signoff -- this file is step D, the
    metric bank; it embeds no numeric constant in a checker.)

The verdict shape is the SAME 5-state ``MetricVerdict`` the geometric-fidelity
pack uses (pass / fail via ``satisfied``; unmeasurable via ``observable=False``),
so an unobservable metric is NEVER turned into a silent pass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Reuse the exact 5-state verdict record the geometric-fidelity pack uses, so the
# floorplan bank and the geometric bank share ONE honest verdict shape.
from clay.layout.spec.geometric_fidelity import MetricVerdict


# --------------------------------------------------------------------------- #
# Threshold bands -- config parameters (env seam), reference-calibrated defaults.
# --------------------------------------------------------------------------- #
ENV_FP_TIER_INTERLEAVE_MAX = "CLAY_FLOORPLAN_TIER_INTERLEAVE_MAX"
ENV_FP_CAP_DY_FRAC_MAX = "CLAY_FLOORPLAN_CAP_DY_FRAC_MAX"
ENV_FP_CAP_ADJACENCY_FRAC_MAX = "CLAY_FLOORPLAN_CAP_ADJACENCY_FRAC_MAX"
ENV_FP_SYMMETRY_AXIS_FRAC_MAX = "CLAY_FLOORPLAN_SYMMETRY_AXIS_FRAC_MAX"
ENV_FP_PATH_LOCALITY_SPAN_MAX = "CLAY_FLOORPLAN_PATH_LOCALITY_SPAN_MAX"
ENV_FP_SIGNAL_FANOUT_MAX = "CLAY_FLOORPLAN_SIGNAL_FANOUT_MAX"

# The MIM top-plate GDS layer (open_pdk layermap). The two Miller caps render as
# two polygons on this layer in the emitted GDS -- GENERIC (any cap on this PDK
# lands here regardless of circuit); config-overridable via the reader argument.
GDS_MIM_PLATE = 67


@dataclass(frozen=True)
class FloorplanThresholds:
    """Reference-calibrated band edges for the floorplan-fidelity metrics.

    Every field is a calibrated SCALAR a future optimizer may override via
    ``CLAY_FLOORPLAN_*``; nothing here is a coordinate.

    Fields
    ------
    tier_interleave_max : float
        Max acceptable fraction of MOS on the wrong side of the N/P band midline.
        The reference bands cleanly (0.0); the area-driven scramble reaches ~0.23.
    cap_dy_frac_max : float
        Max acceptable |cy_A - cy_B| / mean-cap-height for the two MIM caps -- the
        SAME-BAND (shared y) condition. The reference caps share a y (0.0).
    cap_adjacency_frac_max : float
        Max acceptable edge-to-edge x gap / mean-cap-width for the two MIM caps --
        the ADJACENCY condition. The reference caps abut (small gap).
    symmetry_axis_frac_max : float
        Max acceptable per-pair midpoint-x SPREAD / cell-width across all declared
        matched (diff / mirror) pairs -- the ONE-shared-axis (common-centroid)
        condition. Matches the placement-symmetry band in
        ``clay.layout.spec.electrical_obligations`` (SYMMETRY_AXIS_FRAC_MAX = 0.02).
    path_locality_span_max : float
        Max acceptable worst low-fanout signal-net HPWL / cell half-perimeter -- a
        signal net's devices must stay LOCAL. Calibrated to the wirelength-locality
        band (``clay.config`` wirelength_span_max = 0.55).
    signal_fanout_max : int
        A net touching <= this many distinct placed devices is a SIGNAL net
        (locality-gated); higher-fanout nets are power/distribution (exempt).
    """

    tier_interleave_max: float = 0.05
    cap_dy_frac_max: float = 0.10
    cap_adjacency_frac_max: float = 0.15
    symmetry_axis_frac_max: float = 0.02
    path_locality_span_max: float = 0.55
    signal_fanout_max: int = 5


_FP_DEFAULTS = FloorplanThresholds()
_GLOBAL_FP: Optional[FloorplanThresholds] = None


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def floorplan_thresholds(*, fresh: bool = False) -> FloorplanThresholds:
    """Resolve the floorplan-fidelity threshold bands.

    Resolution order (per field): environment variable (``CLAY_FLOORPLAN_*`` --
    the optimizer seam) -> the reference-calibrated default scalar. ``fresh=True``
    re-reads the environment (an optimizer loop / a test that flips a band).
    """
    global _GLOBAL_FP
    if not fresh and _GLOBAL_FP is not None:
        return _GLOBAL_FP

    def pf(env_name: str, default: float) -> float:
        v = _env_float(env_name)
        return default if v is None else v

    def pi(env_name: str, default: int) -> int:
        v = _env_int(env_name)
        return default if v is None else v

    th = FloorplanThresholds(
        tier_interleave_max=pf(ENV_FP_TIER_INTERLEAVE_MAX,
                               _FP_DEFAULTS.tier_interleave_max),
        cap_dy_frac_max=pf(ENV_FP_CAP_DY_FRAC_MAX,
                           _FP_DEFAULTS.cap_dy_frac_max),
        cap_adjacency_frac_max=pf(ENV_FP_CAP_ADJACENCY_FRAC_MAX,
                                  _FP_DEFAULTS.cap_adjacency_frac_max),
        symmetry_axis_frac_max=pf(ENV_FP_SYMMETRY_AXIS_FRAC_MAX,
                                  _FP_DEFAULTS.symmetry_axis_frac_max),
        path_locality_span_max=pf(ENV_FP_PATH_LOCALITY_SPAN_MAX,
                                  _FP_DEFAULTS.path_locality_span_max),
        signal_fanout_max=pi(ENV_FP_SIGNAL_FANOUT_MAX,
                             _FP_DEFAULTS.signal_fanout_max),
    )
    _GLOBAL_FP = th
    return th


# --------------------------------------------------------------------------- #
# geometry helpers (duck-typed over any placement with a ``devices`` map whose
# values carry ``.kind`` + ``.bbox`` -- the generic_placement.Placement shape and
# any lightweight stub, so the metrics never depend on the solver being present).
# --------------------------------------------------------------------------- #
Bbox = Tuple[float, float, float, float]


def _dev_kind(pd: Any) -> Optional[str]:
    return getattr(pd, "kind", None)


def _dev_bbox(pd: Any) -> Optional[Bbox]:
    return getattr(pd, "bbox", None)


def _centroid(bbox: Bbox) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _devices(placement: Any) -> Dict[str, Any]:
    return getattr(placement, "devices", None) or {}


def _centroids_by_name(placement: Any) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for name, pd in _devices(placement).items():
        b = _dev_bbox(pd)
        if b is not None:
            out[str(name)] = _centroid(b)
    return out


def _cell_extent(placement: Any) -> Tuple[float, float]:
    """(W, H) of the placement bbox over ALL placed device footprints.

    Pure relative extent of OUR placement; no reference. Returns (0, 0) when the
    placement is empty (a metric then reports unobservable, never a silent pass)."""
    xs: List[float] = []
    ys: List[float] = []
    for pd in _devices(placement).values():
        b = _dev_bbox(pd)
        if b is None:
            continue
        xs.extend((b[0], b[2]))
        ys.extend((b[1], b[3]))
    if not xs:
        return (0.0, 0.0)
    return (max(xs) - min(xs), max(ys) - min(ys))


def _median(v: Sequence[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# --------------------------------------------------------------------------- #
# METRIC 1 -- tier_interleave (NMOS / PMOS band separation).
# Reimplements the logic of ``ota_dev/analysis/tier_structure.tier_interleave``.
# --------------------------------------------------------------------------- #
def measure_tier_interleave(placement: Any) -> Tuple[Optional[float], dict]:
    """Fraction of MOS devices on the WRONG side of the N/P band midline.

    0.0 = the NMOS and PMOS y-bands are cleanly separated (the reference intent);
    higher = the bands interleave (the area-driven scramble). The midline is
    halfway between the NMOS median-y and the PMOS median-y; a device is MISPLACED
    if it sits on the far side of the midline from its own band. Measured from OUR
    placement centroids (device kind + y); no reference coordinate."""
    byk: Dict[str, List[float]] = {}
    for _name, pd in _devices(placement).items():
        k = _dev_kind(pd)
        if k not in ("nmos", "pmos"):
            continue
        b = _dev_bbox(pd)
        if b is None:
            continue
        byk.setdefault(k, []).append((b[1] + b[3]) / 2.0)
    ny, py = byk.get("nmos", []), byk.get("pmos", [])
    if not ny or not py:
        return None, {"observable": False,
                      "reason": "need both nmos and pmos devices to band"}
    nmed, pmed = _median(ny), _median(py)
    lo_kind, hi_kind = ("nmos", "pmos") if nmed < pmed else ("pmos", "nmos")
    mid = 0.5 * (nmed + pmed)
    lo_ys, hi_ys = byk[lo_kind], byk[hi_kind]
    misplaced = sum(1 for y in lo_ys if y > mid) + sum(1 for y in hi_ys if y < mid)
    total = len(lo_ys) + len(hi_ys)
    frac = misplaced / total if total else 0.0
    return frac, {
        "observable": True,
        "lower_band": lo_kind,
        "nmos_median_um": round(nmed / 1000.0, 2),
        "pmos_median_um": round(pmed / 1000.0, 2),
        "misplaced": misplaced,
        "total": total,
    }


def verdict_tier_interleave(
    frac: Optional[float], detail: dict,
    th: Optional[FloorplanThresholds] = None,
) -> MetricVerdict:
    th = th or floorplan_thresholds()
    band = f"<= {th.tier_interleave_max} ({ENV_FP_TIER_INTERLEAVE_MAX})"
    if frac is None:
        return MetricVerdict(
            name="tier_interleave", value=None, expected=0.0, threshold=band,
            satisfied=None, observable=False,
            detail=str(detail.get("reason", "unobservable")))
    return MetricVerdict(
        name="tier_interleave", value=frac, expected=0.0, threshold=band,
        satisfied=frac <= th.tier_interleave_max, observable=True,
        detail=(f"{detail['misplaced']}/{detail['total']} MOS on the wrong side of "
                f"the {detail['lower_band']}-below midline; interleave={frac:.3f} "
                f"(nmos y~{detail['nmos_median_um']}um, pmos y~"
                f"{detail['pmos_median_um']}um)"))


# --------------------------------------------------------------------------- #
# METRIC 2 -- cap_colocation (two MIM caps in ONE band, x-adjacent).
# --------------------------------------------------------------------------- #
def cap_boxes_from_placement(placement: Any) -> List[Tuple[str, Bbox]]:
    """The (name, bbox) of every cap device in the placement (kind == 'cap')."""
    return [(str(n), _dev_bbox(pd)) for n, pd in _devices(placement).items()
            if _dev_kind(pd) == "cap" and _dev_bbox(pd) is not None]


def cap_boxes_from_gds(gds_path: str,
                       mim_layer: int = GDS_MIM_PLATE) -> List[Bbox]:
    """The MIM plate bboxes (in dbu = nm) on ``mim_layer`` of the emitted GDS.

    Reads ONLY the emitted artifact the LVS/DRC oracles ran on (one canonical
    path); no reference, no route metadata. Merges the layer so a plate drawn as
    several touching shapes counts once. Returns [] when klayout / the layer is
    absent (the caller then reports the metric unobservable -- never a silent
    pass)."""
    try:
        import klayout.db as kdb
    except Exception:  # noqa: BLE001
        return []
    ly = kdb.Layout()
    ly.read(str(gds_path))
    top = ly.top_cells()[0]
    idx = None
    for i in ly.layer_indexes():
        if ly.get_info(i).layer == mim_layer:
            idx = i
            break
    if idx is None:
        return []
    reg = kdb.Region(top.begin_shapes_rec(idx))
    reg.merge()
    out: List[Bbox] = []
    for poly in reg.each():
        b = poly.bbox()
        out.append((float(b.left), float(b.bottom), float(b.right), float(b.top)))
    return out


def measure_cap_colocation(cap_boxes: Sequence[Bbox]) -> dict:
    """Same-band (dy) + adjacency (edge x-gap) measurement for the two MIM caps.

    Measured from OUR cap geometry only (placement or emitted GDS). Both outputs
    are RELATIVE fractions (dy / mean-cap-height, edge-gap / mean-cap-width) so a
    uniform translation OR scale leaves them identical (the leakage line). Reports
    unobservable (never a silent pass) unless EXACTLY two caps are present."""
    boxes = [b for b in cap_boxes if b is not None]
    if len(boxes) != 2:
        return {"observable": False, "n_caps": len(boxes),
                "reason": f"need exactly 2 MIM caps to colocate, saw {len(boxes)}"}
    b1, b2 = boxes[0], boxes[1]
    c1, c2 = _centroid(b1), _centroid(b2)
    w1, h1 = (b1[2] - b1[0]), (b1[3] - b1[1])
    w2, h2 = (b2[2] - b2[0]), (b2[3] - b2[1])
    mean_w = max((w1 + w2) / 2.0, 1e-9)
    mean_h = max((h1 + h2) / 2.0, 1e-9)
    dy = abs(c1[1] - c2[1])
    xgap = abs(c1[0] - c2[0])
    edge_gap = max(0.0, xgap - mean_w)  # centroid distance minus one mean cap width
    return {
        "observable": True,
        "n_caps": 2,
        "dy_um": round(dy / 1000.0, 3),
        "xgap_um": round(xgap / 1000.0, 3),
        "dy_frac": dy / mean_h,
        "adjacency_frac": edge_gap / mean_w,
    }


def verdict_cap_colocation(
    measured: dict, th: Optional[FloorplanThresholds] = None,
) -> MetricVerdict:
    """PASS iff the two caps share a band (dy_frac small) AND are x-adjacent
    (adjacency_frac small). The reported ``value`` is the worse of the two
    fractions (the binding sub-condition)."""
    th = th or floorplan_thresholds()
    band = (f"dy_frac <= {th.cap_dy_frac_max} ({ENV_FP_CAP_DY_FRAC_MAX}) AND "
            f"adjacency_frac <= {th.cap_adjacency_frac_max} "
            f"({ENV_FP_CAP_ADJACENCY_FRAC_MAX})")
    if not measured.get("observable"):
        return MetricVerdict(
            name="cap_colocation", value=None, expected=0.0, threshold=band,
            satisfied=None, observable=False,
            detail=str(measured.get("reason", "unobservable")))
    dyf = measured["dy_frac"]
    adjf = measured["adjacency_frac"]
    same_band = dyf <= th.cap_dy_frac_max
    adjacent = adjf <= th.cap_adjacency_frac_max
    return MetricVerdict(
        name="cap_colocation", value=max(dyf, adjf), expected=0.0, threshold=band,
        satisfied=bool(same_band and adjacent), observable=True,
        detail=(f"caps dy={measured['dy_um']}um (dy_frac={dyf:.3f}"
                f"{'' if same_band else ' [not same band]'}), centroid x-gap="
                f"{measured['xgap_um']}um (adjacency_frac={adjf:.3f}"
                f"{'' if adjacent else ' [not adjacent]'})"))


# --------------------------------------------------------------------------- #
# METRIC 3 -- symmetry_axis (matched pairs share ONE vertical axis).
# --------------------------------------------------------------------------- #
def matched_axis_pairs(intent: Any) -> List[Tuple[str, str]]:
    """The 2-member matched pairs whose reflection axes must agree, from topology.

    Union of ``mirror_pairs`` (@symmetric / current-mirror legs) and ``diff_pairs``
    (differential pairs). Pure topology (device names); no coordinate, so a
    translation of the placement leaves the pair set identical (leakage line)."""
    pairs: List[Tuple[str, str]] = []
    for p in (getattr(intent, "mirror_pairs", None) or []):
        if len(p) == 2:
            pairs.append((str(p[0]), str(p[1])))
    for p in (getattr(intent, "diff_pairs", None) or []):
        if len(p) == 2:
            pairs.append((str(p[0]), str(p[1])))
    return pairs


def measure_symmetry_axis(placement: Any,
                          intent: Any) -> Tuple[Optional[float], dict]:
    """Per-pair midpoint-x SPREAD across all matched pairs, / cell width.

    A diff pair / 2-leg mirror reflects about ONE vertical axis whose x is the
    pair midpoint ((x_A + x_B)/2). Every matched pair in the cell must agree on
    that axis (a shared common-centroid line), so the spread of the per-pair
    midpoints is the axis-disagreement. Measured from OUR placement centroids,
    normalised by OUR cell width (translation- AND scale-invariant); no reference.
    Unobservable (never a silent pass) when fewer than two pairs are placed."""
    pairs = matched_axis_pairs(intent) if intent is not None else []
    C = _centroids_by_name(placement)
    mids: List[Tuple[str, float]] = []
    for a, b in pairs:
        if a in C and b in C:
            mids.append((f"{a}/{b}", (C[a][0] + C[b][0]) / 2.0))
    if len(mids) < 2:
        return None, {"observable": False,
                      "reason": (f"need >= 2 placed matched pairs to measure an "
                                 f"axis spread, saw {len(mids)}"),
                      "n_pairs": len(mids)}
    W, _H = _cell_extent(placement)
    if W <= 0:
        return None, {"observable": False, "reason": "zero cell width",
                      "n_pairs": len(mids)}
    xs = [m for _lbl, m in mids]
    spread = (max(xs) - min(xs)) / W
    return spread, {
        "observable": True,
        "n_pairs": len(mids),
        "axis_mids_um": [(lbl, round(m / 1000.0, 2)) for lbl, m in mids],
        "cell_width_um": round(W / 1000.0, 2),
    }


def verdict_symmetry_axis(
    spread: Optional[float], detail: dict,
    th: Optional[FloorplanThresholds] = None,
) -> MetricVerdict:
    th = th or floorplan_thresholds()
    band = f"<= {th.symmetry_axis_frac_max} ({ENV_FP_SYMMETRY_AXIS_FRAC_MAX})"
    if spread is None:
        return MetricVerdict(
            name="symmetry_axis", value=None, expected=0.0, threshold=band,
            satisfied=None, observable=False,
            detail=str(detail.get("reason", "unobservable")))
    return MetricVerdict(
        name="symmetry_axis", value=spread, expected=0.0, threshold=band,
        satisfied=spread <= th.symmetry_axis_frac_max, observable=True,
        detail=(f"{detail['n_pairs']} matched pair(s); midpoint-x spread="
                f"{spread:.3f} of cell width ({detail['cell_width_um']}um); "
                f"axes(um)={detail['axis_mids_um']}"))


# --------------------------------------------------------------------------- #
# METRIC 4 -- current_path_locality (signal-path devices stay close).
# --------------------------------------------------------------------------- #
# Power / ground / global nets are HIGH-FANOUT distribution nets that intentionally
# span the cell -- they are classified out of the locality gate (never a defect).
_POWER_NETS = frozenset(
    {"vdd", "vdd!", "vcc", "vcc!", "vss", "vss!", "gnd", "gnd!", "0"}
)


def net_to_devices_from_cir(source_cir: Any) -> Dict[str, set]:
    """``net -> {device labels}`` connectivity from a SPICE ``.cir`` (topology).

    Uses the SAME generic MOS + passive card parsers the placer uses
    (``clay.layout.generic_placement``), so the connectivity matches the placed
    device labels. Pure netlist topology (net names / device names) -- no
    coordinate; a translation of the placement cannot change it (leakage line)."""
    from clay.layout.generic_placement import (
        parse_cir_mos_records,
        parse_cir_passive_records,
    )

    records: Dict[str, Any] = {}
    records.update(parse_cir_mos_records(source_cir))
    records.update(parse_cir_passive_records(source_cir))
    n2d: Dict[str, set] = {}
    for lbl, rec in records.items():
        for _role, net in (rec.get("pins") or {}).items():
            n2d.setdefault(str(net).strip().lower(), set()).add(str(lbl))
    return n2d


def measure_current_path_locality(
    placement: Any, net_to_devices: Dict[str, set],
    th: Optional[FloorplanThresholds] = None,
) -> dict:
    """Per-signal-net HPWL / cell half-perimeter over the netlist connectivity.

    For each net touching >= 2 placed devices (and <= ``signal_fanout_max`` of
    them -- a SIGNAL net, not power/distribution), the HPWL is the placed-device
    centroid bounding-box half-perimeter; normalised by the cell half-perimeter it
    is a scale-free locality fraction. Reports the MEAN (the headline locality)
    and the WORST net. Measured from OUR placement + netlist topology; no
    reference. Unobservable when no signal net is placed (never a silent pass)."""
    th = th or floorplan_thresholds()
    C = _centroids_by_name(placement)
    W, H = _cell_extent(placement)
    perim = W + H
    fmax = int(th.signal_fanout_max)
    spans: List[Tuple[float, str, List[str]]] = []
    for net, devs in (net_to_devices or {}).items():
        placed = [d for d in devs if d in C]
        if len(placed) < 2:
            continue
        if str(net).lower() in _POWER_NETS or len(placed) > fmax:
            continue
        xs = [C[d][0] for d in placed]
        ys = [C[d][1] for d in placed]
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        frac = hpwl / perim if perim > 0 else 0.0
        spans.append((frac, str(net), sorted(placed)))
    if not spans:
        return {"observable": False,
                "reason": "no low-fanout signal net with >= 2 placed devices"}
    spans.sort(reverse=True)
    mean = sum(s[0] for s in spans) / len(spans)
    return {
        "observable": True,
        "mean_span_frac": mean,
        "max_span_frac": spans[0][0],
        "worst_net": spans[0][1],
        "worst_devices": spans[0][2],
        "n_signal_nets": len(spans),
    }


def verdict_current_path_locality(
    measured: dict, th: Optional[FloorplanThresholds] = None,
) -> MetricVerdict:
    """PASS iff the WORST low-fanout signal net stays local (span <= band). The
    reported ``value`` is the worst-net span; the mean is carried in the detail."""
    th = th or floorplan_thresholds()
    band = f"worst signal span <= {th.path_locality_span_max} ({ENV_FP_PATH_LOCALITY_SPAN_MAX})"
    if not measured.get("observable"):
        return MetricVerdict(
            name="current_path_locality", value=None, expected=None, threshold=band,
            satisfied=None, observable=False,
            detail=str(measured.get("reason", "unobservable")))
    worst = measured["max_span_frac"]
    return MetricVerdict(
        name="current_path_locality", value=worst, expected=None, threshold=band,
        satisfied=worst <= th.path_locality_span_max, observable=True,
        detail=(f"worst signal net {measured['worst_net']!r} span={worst:.3f} of "
                f"cell half-perimeter (devices {measured['worst_devices']}); "
                f"mean over {measured['n_signal_nets']} signal nets="
                f"{measured['mean_span_frac']:.3f}"))


# --------------------------------------------------------------------------- #
# aggregate report (one pass over the placement + optional GDS / netlist).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FloorplanReport:
    """The four floorplan-fidelity metrics measured over one placement."""

    tier_interleave: Optional[float]
    cap_dy_um: Optional[float]
    cap_xgap_um: Optional[float]
    symmetry_axis_spread: Optional[float]
    path_locality_mean: Optional[float]
    metrics: Tuple[MetricVerdict, ...] = field(default_factory=tuple)


def measure_floorplan_fidelity(
    placement: Any, *,
    intent: Any = None,
    gds_path: Optional[str] = None,
    mim_layer: int = GDS_MIM_PLATE,
    source_cir: Any = None,
    net_to_devices: Optional[Dict[str, set]] = None,
    th: Optional[FloorplanThresholds] = None,
) -> FloorplanReport:
    """Measure all four floorplan-fidelity metrics over ONE placement.

    ``placement`` supplies OUR device centroids + kinds (the in-process
    place_generic / maybe_envelope_replace result). ``intent`` supplies the
    matched pairs (topology) for symmetry_axis. The MIM caps are read from
    ``gds_path`` (MIM layer, the emitted canonical artifact) when given, else from
    the placement's cap devices. ``source_cir`` (or a precomputed
    ``net_to_devices``) supplies the netlist connectivity for path locality. Any
    input absent -> that metric is honestly unobservable (never a silent pass).
    The reference is NOT an input; the bands come from config.
    """
    th = th or floorplan_thresholds()

    tier_frac, tier_d = measure_tier_interleave(placement)

    if gds_path is not None:
        cap_boxes: Sequence[Bbox] = cap_boxes_from_gds(gds_path, mim_layer)
    else:
        cap_boxes = [b for _n, b in cap_boxes_from_placement(placement)]
    cap_m = measure_cap_colocation(cap_boxes)

    sym_spread, sym_d = measure_symmetry_axis(placement, intent)

    if net_to_devices is None and source_cir is not None:
        net_to_devices = net_to_devices_from_cir(source_cir)
    if net_to_devices:
        path_m = measure_current_path_locality(placement, net_to_devices, th)
    else:
        path_m = {"observable": False,
                  "reason": "no netlist connectivity supplied for path locality"}

    metrics = (
        verdict_tier_interleave(tier_frac, tier_d, th),
        verdict_cap_colocation(cap_m, th),
        verdict_symmetry_axis(sym_spread, sym_d, th),
        verdict_current_path_locality(path_m, th),
    )
    return FloorplanReport(
        tier_interleave=tier_frac,
        cap_dy_um=(cap_m.get("dy_um") if cap_m.get("observable") else None),
        cap_xgap_um=(cap_m.get("xgap_um") if cap_m.get("observable") else None),
        symmetry_axis_spread=sym_spread,
        path_locality_mean=(path_m.get("mean_span_frac")
                            if path_m.get("observable") else None),
        metrics=metrics,
    )


__all__ = [
    "FloorplanThresholds",
    "floorplan_thresholds",
    "MetricVerdict",
    "FloorplanReport",
    "GDS_MIM_PLATE",
    "measure_tier_interleave",
    "verdict_tier_interleave",
    "cap_boxes_from_placement",
    "cap_boxes_from_gds",
    "measure_cap_colocation",
    "verdict_cap_colocation",
    "matched_axis_pairs",
    "measure_symmetry_axis",
    "verdict_symmetry_axis",
    "net_to_devices_from_cir",
    "measure_current_path_locality",
    "verdict_current_path_locality",
    "measure_floorplan_fidelity",
]
