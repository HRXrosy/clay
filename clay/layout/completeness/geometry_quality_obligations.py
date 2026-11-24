"""GEOMETRIC-QUALITY obligations -- two independent checkers on the emitted GDS.

WHAT THIS IS
============
Two gate-worthy GEOMETRIC-QUALITY (fidelity-class) obligations that a routed +
placed analog cell owes, each an INDEPENDENT oracle reading the emitted GDS via
``klayout.db`` -- never the router's / placer's internal state. They name two
facts the DRC/LVS gates are structurally blind to (a legal layout can be
geometrically bad), so they exist to make a legal-but-ugly layout report RED
instead of a silent green:

  1. TRUNK-CLEANLINESS -- routed metal should be clean trunks, not zigzags.
     Measured as bends-per-merged-polygon on a routing metal (default M1), the
     SAME quantity ``ota_dev/analysis/structure_metrics.direction_purity`` reports
     (``(num_points_hull - 4)//2`` per merged polygon). A zigzag route inflates
     the vertex count of its merged metal far above a clean Manhattan trunk.

  2. ARRAYING-DENSITY -- matched devices should share diffusion into dense
     multi-gate arrays. Measured GEOMETRICALLY: merge the OD layer into connected
     islands; for each island count the distinct poly-over-OD gate STRIPES (the
     ``fingers`` convention of ``structure_metrics.OdIsland`` / decompose
     ``_finger_columns`` -- gate x-columns collapsed at 0.25um); take the mean
     over gate-bearing islands (fingers > 0 and not a ring frame). A fragmented
     under-arrayed layout scatters matched fingers across many low-gate islands
     instead of a few dense shared-diffusion islands.

FIVE-STATE VERDICT (never a silent pass)
========================================
Each obligation returns one of Clay's five explicit states
(``clay.layout.spec.verdict.FIVE_STATES``):

    pass         -- oracle read the geometry; ours is within the reference band.
    fail         -- oracle read the geometry; ours is outside the band (RED). A
                    fidelity RED is a BLOCKING fail (like DRC/LVS), not advisory.
    unmeasurable -- the oracle could NOT read the relevant geometry, OR no
                    reference calibration was supplied. NEVER a silent pass.
    unsupported  -- klayout / the layer table is unavailable (capability absent).
    advisory     -- reserved for a purely-structural report (not used by these
                    two quantitative gates).

RELATIVE BAND DISCIPLINE (reference calibrates; it never feeds an engine)
========================================================================
Per the repo convention "relative consistency thresholds only" (no absolute-
difference / absolute-magnitude gate), the verdict is decided on a RELATIVE
RATIO against a REFERENCE-CALIBRATED value -- never on a hardcoded absolute count.
The reference value is read ONLY as an ORACLE (a reference GDS measured live, or
a precomputed ``ref_value`` float) to calibrate the acceptance band; it is never
returned to, or consumed by, any placer/router. If NO reference calibration is
supplied the obligation is ``unmeasurable`` -- it refuses to invent an absolute
gate.

  * trunk-cleanliness (lower is better): RED when
        measured_bends_per_poly > max_ratio * reference_bends_per_poly.
    Default ``max_ratio = 2.0`` (a clean trunk may carry up to twice the
    reference's incidental corners before it reads as a zigzag).

  * arraying-density (higher is better): RED when
        measured_gates_per_island < min_ratio * reference_gates_per_island.
    Default ``min_ratio = 0.70`` (array density within 30% of the reference-
    calibrated target is acceptable style variation; below that is
    under-arraying / diffusion fragmentation).

INDEPENDENCE / REUSE
====================
Layer numbers + the merged-region basis come from the single-source
``ota_dev.analysis.decompose`` (``LAYER`` / ``_load`` / ``_region``); this module
imports them, it does not edit them. The measurement is pure geometry on the
emitted artifact -- it does not import or consult any routing/placement engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Reuse the canonical 5-state set (single source; not edited here).
from clay.layout.spec.verdict import FIVE_STATES

# klayout + the verified open_pdk layer table / region basis come from decompose.
# decompose raises SystemExit if klayout is unavailable, so guard BOTH Exception
# and SystemExit to keep this module importable and let the obligations report an
# honest ``unsupported`` verdict instead of crashing.
try:  # pragma: no cover - klayout is always present in this repo's CI
    import klayout.db as _kdb
    from ota_dev.analysis import decompose as _D

    _IMPORT_ERR: Optional[str] = None
except (Exception, SystemExit) as _exc:  # noqa: BLE001
    _kdb = None  # type: ignore[assignment]
    _D = None  # type: ignore[assignment]
    _IMPORT_ERR = str(_exc)


# Column-merge tolerance for collapsing near-coincident gate x-stripes into one
# finger (the 0.25um convention shared with decompose._finger_columns and
# structure_metrics.LayoutProbe._build_islands). Not a gate threshold -- a
# measurement quantisation of the SAME quantity those modules already report.
_COL_MERGE_UM = 0.25


# --------------------------------------------------------------------------- #
# Pure-geometry measurements (no router/placer, no reference).
# --------------------------------------------------------------------------- #
def measure_bends_per_polygon(
    gds_path: str, role: str = "m1"
) -> Optional[Tuple[int, int, float]]:
    """(total_bends, n_polygons, bends_per_polygon) on the merged ``role`` layer.

    ``bends`` per merged polygon is ``max(0, (num_points_hull - 4) // 2)`` -- the
    SAME measure as ``structure_metrics.direction_purity``. Returns ``None`` if
    klayout/decompose is unavailable, or ``(0, 0, 0.0)`` if the layer is empty
    (the caller distinguishes empty -> unmeasurable from a real 0.0)."""
    if _D is None or _kdb is None:
        return None
    ly, top, _dbu = _D._load(gds_path)
    reg = _D._region(ly, top, role)  # merged
    bends = 0
    polys = 0
    for p in reg.each():
        polys += 1
        bends += max(0, (p.num_points_hull() - 4) // 2)
    return bends, polys, (bends / polys if polys else 0.0)


def _gate_columns_over(island_poly, poly_reg, dbu: float) -> int:
    """Distinct poly-over-OD gate STRIPES crossing one OD island.

    The gate columns are the x-centres of ``poly & island`` shapes, collapsed at
    ``_COL_MERGE_UM`` -- the ``fingers`` count of one island (decompose
    ``_finger_columns`` / structure_metrics ``OdIsland``)."""
    gate = (poly_reg & _kdb.Region(island_poly)).merged()
    xs = sorted(
        0.5 * (g.bbox().left + g.bbox().right) * dbu for g in gate.each()
    )
    cols: List[float] = []
    for x in xs:
        if not cols or (x - cols[-1]) > _COL_MERGE_UM:
            cols.append(x)
    return len(cols)


@dataclass
class ArrayingMeasurement:
    """Geometric arraying inventory measured on one GDS (no reference)."""

    total_islands: int = 0
    gate_bearing_islands: int = 0
    total_gates: int = 0
    mean_gates_per_island: float = 0.0
    multi_gate_islands: int = 0
    frac_multi_gate: float = 0.0
    ring_frames: int = 0


def measure_arraying_density(gds_path: str) -> Optional[ArrayingMeasurement]:
    """Merge OD -> connected islands; count gate stripes per island; aggregate.

    A ring frame is an OD island whose merged polygon has a HOLE (a guard-ring
    frame) -- excluded from the gate-bearing set. A gate-bearing island has >= 1
    gate stripe and is not a ring frame. Returns ``None`` when klayout/decompose
    is unavailable."""
    if _D is None or _kdb is None:
        return None
    ly, top, dbu = _D._load(gds_path)
    od = _D._region(ly, top, "od")      # merged -> connected components
    poly = _D._region(ly, top, "poly")  # L13 gates
    gate_bearing: List[int] = []
    n_rings = 0
    total = 0
    for p in od.each():
        total += 1
        is_ring = p.holes() > 0
        if is_ring:
            n_rings += 1
            continue
        g = _gate_columns_over(p, poly, dbu)
        if g > 0:
            gate_bearing.append(g)
    n_gb = len(gate_bearing)
    total_gates = sum(gate_bearing)
    multi = sum(1 for g in gate_bearing if g > 1)
    return ArrayingMeasurement(
        total_islands=total,
        gate_bearing_islands=n_gb,
        total_gates=total_gates,
        mean_gates_per_island=round(total_gates / n_gb, 4) if n_gb else 0.0,
        multi_gate_islands=multi,
        frac_multi_gate=round(multi / n_gb, 4) if n_gb else 0.0,
        ring_frames=n_rings,
    )


# --------------------------------------------------------------------------- #
# Obligation reports.
# --------------------------------------------------------------------------- #
@dataclass
class TrunkCleanlinessReport:
    """Verdict of the trunk-cleanliness obligation on one GDS.

    ``status`` is one of :data:`FIVE_STATES`. ``value`` is the measured
    bends-per-polygon, ``reference`` the reference-calibrated value, ``ratio``
    ``value / reference`` (the relative quantity the gate decides on), and
    ``max_ratio`` the relative band ceiling."""

    status: str = "unmeasurable"
    role: str = "m1"
    value: Optional[float] = None
    reference: Optional[float] = None
    ratio: Optional[float] = None
    max_ratio: float = 2.0
    n_polygons: int = 0
    total_bends: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def red(self) -> bool:
        return self.status == "fail"


@dataclass
class ArrayingDensityReport:
    """Verdict of the arraying-density obligation on one GDS.

    ``value`` is the measured mean gates per gate-bearing OD island;
    ``reference`` the reference-calibrated target; ``ratio`` ``value /
    reference``; ``min_ratio`` the relative band floor."""

    status: str = "unmeasurable"
    value: Optional[float] = None
    reference: Optional[float] = None
    ratio: Optional[float] = None
    min_ratio: float = 0.70
    measurement: Optional[ArrayingMeasurement] = None
    reference_measurement: Optional[ArrayingMeasurement] = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def red(self) -> bool:
        return self.status == "fail"


# --------------------------------------------------------------------------- #
# Obligation 1: trunk-cleanliness (bends per polygon).
# --------------------------------------------------------------------------- #
def check_trunk_cleanliness(
    gds_path: str,
    *,
    ref_gds: Optional[str] = None,
    ref_value: Optional[float] = None,
    role: str = "m1",
    max_ratio: float = 2.0,
) -> TrunkCleanlinessReport:
    """Trunk-cleanliness obligation: routed metal must be clean trunks.

    Measures bends-per-polygon on ``role`` (default M1) of ``gds_path`` and
    verdicts it RELATIVE to a reference-calibrated value (``ref_value`` directly,
    or measured live from ``ref_gds`` as an oracle). RED (``fail``) when the
    measured value exceeds ``max_ratio`` times the reference band; ``pass`` when
    within it. ``unmeasurable`` when the metal layer is empty or NO reference
    calibration is supplied (it refuses to invent an absolute gate);
    ``unsupported`` when klayout / the layer table is unavailable.
    """
    rep = TrunkCleanlinessReport(role=role, max_ratio=max_ratio)
    if _IMPORT_ERR is not None:
        rep.status = "unsupported"
        rep.detail = f"klayout/decompose unavailable: {_IMPORT_ERR}"
        return rep

    meas = measure_bends_per_polygon(gds_path, role)
    if meas is None:  # defence in depth (import guard already handled above)
        rep.status = "unsupported"
        rep.detail = "measurement backend unavailable"
        return rep
    bends, polys, bpp = meas
    rep.total_bends = bends
    rep.n_polygons = polys
    rep.value = round(bpp, 4)
    if polys == 0:
        rep.status = "unmeasurable"
        rep.detail = f"no {role} polygons in {gds_path}"
        return rep

    ref = _resolve_ref_bends(ref_gds, ref_value, role)
    if ref is None or ref <= 0.0:
        rep.status = "unmeasurable"
        rep.detail = (
            "no positive reference-calibrated bends/poly "
            "(supply ref_gds or ref_value); relative band cannot be formed"
        )
        return rep
    rep.reference = round(ref, 4)
    rep.ratio = round(bpp / ref, 4)
    if rep.ratio > max_ratio:
        rep.status = "fail"
        rep.detail = (
            f"{role} bends/poly {rep.value} = {rep.ratio}x the reference-"
            f"calibrated {rep.reference} (> {max_ratio}x band ceiling): zigzag"
        )
    else:
        rep.status = "pass"
        rep.detail = (
            f"{role} bends/poly {rep.value} = {rep.ratio}x the reference-"
            f"calibrated {rep.reference} (<= {max_ratio}x band ceiling)"
        )
    return rep


def _resolve_ref_bends(
    ref_gds: Optional[str], ref_value: Optional[float], role: str
) -> Optional[float]:
    if ref_value is not None:
        return float(ref_value)
    if ref_gds is None:
        return None
    meas = measure_bends_per_polygon(ref_gds, role)
    if meas is None or meas[1] == 0:
        return None
    return meas[2]


# --------------------------------------------------------------------------- #
# Obligation 2: arraying-density (gates per OD island).
# --------------------------------------------------------------------------- #
def check_arraying_density(
    gds_path: str,
    *,
    ref_gds: Optional[str] = None,
    ref_value: Optional[float] = None,
    min_ratio: float = 0.70,
) -> ArrayingDensityReport:
    """Arraying-density obligation: matched devices must share dense diffusion.

    Measures the mean gate stripes per gate-bearing OD island of ``gds_path`` and
    verdicts it RELATIVE to a reference-calibrated target (``ref_value`` directly,
    or measured live from ``ref_gds`` as an oracle). RED (``fail``) when the
    measured density falls below ``min_ratio`` times the reference target (an
    under-arrayed / fragmented layout); ``pass`` when within it. ``unmeasurable``
    when there are no gate-bearing OD islands or NO reference calibration is
    supplied; ``unsupported`` when klayout / the layer table is unavailable.
    """
    rep = ArrayingDensityReport(min_ratio=min_ratio)
    if _IMPORT_ERR is not None:
        rep.status = "unsupported"
        rep.detail = f"klayout/decompose unavailable: {_IMPORT_ERR}"
        return rep

    meas = measure_arraying_density(gds_path)
    if meas is None:
        rep.status = "unsupported"
        rep.detail = "measurement backend unavailable"
        return rep
    rep.measurement = meas
    rep.value = meas.mean_gates_per_island
    if meas.gate_bearing_islands == 0:
        rep.status = "unmeasurable"
        rep.detail = f"no gate-bearing OD islands in {gds_path}"
        return rep

    ref, ref_meas = _resolve_ref_arraying(ref_gds, ref_value)
    rep.reference_measurement = ref_meas
    if ref is None or ref <= 0.0:
        rep.status = "unmeasurable"
        rep.detail = (
            "no positive reference-calibrated gates/island "
            "(supply ref_gds or ref_value); relative band cannot be formed"
        )
        return rep
    rep.reference = round(ref, 4)
    rep.ratio = round(rep.value / ref, 4)
    if rep.ratio < min_ratio:
        rep.status = "fail"
        rep.detail = (
            f"mean gates/OD-island {rep.value} = {rep.ratio}x the reference-"
            f"calibrated {rep.reference} (< {min_ratio}x band floor): "
            f"under-arrayed ({meas.gate_bearing_islands} gate islands, "
            f"frac multi-gate {meas.frac_multi_gate})"
        )
    else:
        rep.status = "pass"
        rep.detail = (
            f"mean gates/OD-island {rep.value} = {rep.ratio}x the reference-"
            f"calibrated {rep.reference} (>= {min_ratio}x band floor)"
        )
    return rep


def _resolve_ref_arraying(
    ref_gds: Optional[str], ref_value: Optional[float]
) -> Tuple[Optional[float], Optional[ArrayingMeasurement]]:
    if ref_value is not None:
        return float(ref_value), None
    if ref_gds is None:
        return None, None
    meas = measure_arraying_density(ref_gds)
    if meas is None or meas.gate_bearing_islands == 0:
        return None, meas
    return meas.mean_gates_per_island, meas


__all__ = [
    "FIVE_STATES",
    "ArrayingMeasurement",
    "TrunkCleanlinessReport",
    "ArrayingDensityReport",
    "measure_bends_per_polygon",
    "measure_arraying_density",
    "check_trunk_cleanliness",
    "check_arraying_density",
]
