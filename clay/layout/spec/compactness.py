"""COMPACTNESS obligation -- the missing contract behind the OTA 1.55x area gap.

ROOT (measured, ref_compare.py): the reference (10704 um2) and the band-stack
(16555 um2) carry NEARLY IDENTICAL device / ring-implant / metal (M1-M6) areas.
The 1.55x gap is purely SPATIAL: the reference CLUSTERS devices into a compact
2D block with n- and p-devices INTERLEAVED (OD bbox 7157 um2, device-region
density ~0.50, n-well bbox ~ OD bbox), while the band-stack SPREADS them into
sparse wide segregated rows (OD bbox 13951 um2, density ~0.24, n-well bbox 4577
<< OD bbox 13951). Clay gated only LEGALITY (DRC/LVS) and the existing fidelity
metrics never NAMED this device-region packing obligation as a gate -- so a
legal-but-spread layout reported green. This module NAMES that contract.

The obligation
--------------
A layout's DEVICE-REGION must be COMPACT and N/P INTERLEAVED, not spread into
sparse segregated rows. Two measurable metrics, both from the artifact-under-
test's OWN emitted GDS (leakage-free) and both TWO-SIDED bands (over-compaction
is also a defect -- internal slack is a routing resource, so fill->1 is wrong):

  device_region_density : (merged device-diffusion area) / (device bbox area)
                          = OD(layer 3) merged area / OD bounding-box area. The
                          reference packs ~0.50; the band-stack spreads to ~0.24
                          (below the floor -> FAIL).
  np_interleave         : (n-well bbox area) / (device bbox area) = NWELL(layer 2)
                          bbox / OD bbox. When n- and p-devices interleave across
                          the region the n-well bbox spans nearly the whole device
                          bbox (reference ~0.998); a SEGREGATED layout (all n
                          above all p) gives an n-well bbox far smaller than the
                          device bbox (band-stack ~0.33 -> FAIL).

Leakage line: every value is measured from the artifact-under-test's OWN emitted
GDS (the single canonical artifact the LVS/DRC oracles ran on). The reference
NEVER enters a metric's input -- it only CALIBRATED the band scalars in
clay.config.FidelityThresholds (devregion_density_lo/hi, np_interleave_lo/hi),
which are config parameters (CLAY_FIDELITY_* env overridable). Both metrics are
RELATIVE ratios (area/bbox-area), so a uniform translation of any layout leaves
every value identical -- a coordinate cannot leak through a ratio of areas.

GDS layer numbers (production layermap): OD/diff = 3, NWELL = 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from clay.config import FidelityThresholds, fidelity_thresholds
from clay.layout.spec.geometric_fidelity import (
    MetricVerdict,
    _layer_region,
    _open_top,
)

GDS_OD = 3
GDS_NWELL = 2


@dataclass(frozen=True)
class CompactnessReport:
    """Both compactness metrics measured over one emitted GDS."""

    device_region_density: Optional[float]
    np_interleave: Optional[float]
    od_bbox_um2: Optional[float]
    nwell_bbox_um2: Optional[float]
    metrics: Tuple[MetricVerdict, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# measurements (read the artifact-under-test's OWN emitted GDS -- leakage-free).
# --------------------------------------------------------------------------- #
def measure_device_region_density(gds_path: str) -> Tuple[Optional[float],
                                                          Optional[float]]:
    """(device-region density, OD-bbox area um2) from the emitted GDS.

    density = merged device-diffusion (OD, layer 3) area / OD bounding-box area.
    A compact 2D cluster fills its own bbox densely (~0.50); sparse wide rows
    leave the bbox mostly empty (~0.24). Returns (None, None) when no OD exists
    -- never a silent pass (the verdict marks it unobservable)."""
    kdb, _ly, top, li = _open_top(gds_path)
    od = _layer_region(kdb, top, li, GDS_OD)
    if od is None or od.is_empty():
        return None, None
    od_area = float(od.area())
    bb = od.bbox()
    bb_area = float(bb.width()) * float(bb.height())
    if bb_area <= 0:
        return None, None
    return od_area / bb_area, bb_area / 1e6


def measure_np_interleave(gds_path: str) -> Tuple[Optional[float],
                                                  Optional[float],
                                                  Optional[float]]:
    """(np_interleave, nwell-bbox area um2, OD-bbox area um2) from emitted GDS.

    np_interleave = n-well (NWELL, layer 2) bbox area / device (OD, layer 3)
    bbox area. INTERLEAVED n/p devices make the n-well span nearly the whole
    device region (ratio ~1.0); a SEGREGATED stack (all n above all p) gives a
    small n-well bbox over a tall device bbox (ratio ~0.33). Returns Nones when
    OD or NWELL is absent -- never a silent pass."""
    kdb, _ly, top, li = _open_top(gds_path)
    od = _layer_region(kdb, top, li, GDS_OD)
    nw = _layer_region(kdb, top, li, GDS_NWELL)
    if od is None or od.is_empty() or nw is None or nw.is_empty():
        return None, None, None
    od_bb = od.bbox()
    od_bb_area = float(od_bb.width()) * float(od_bb.height())
    nw_bb = nw.bbox()
    nw_bb_area = float(nw_bb.width()) * float(nw_bb.height())
    if od_bb_area <= 0:
        return None, None, None
    return nw_bb_area / od_bb_area, nw_bb_area / 1e6, od_bb_area / 1e6


# --------------------------------------------------------------------------- #
# per-metric verdicts (each reads its TWO-SIDED band from config -- no constant).
# --------------------------------------------------------------------------- #
def verdict_device_region_density(
    value: Optional[float],
    th: Optional[FidelityThresholds] = None,
) -> MetricVerdict:
    th = th or fidelity_thresholds()
    band = (f"[{th.devregion_density_lo}, {th.devregion_density_hi}] "
            "(CLAY_FIDELITY_DEVREGION_DENSITY_LO/HI)")
    if value is None:
        return MetricVerdict(
            name="device_region_density", value=None, expected=None,
            threshold=band, satisfied=None, observable=False,
            detail="no device diffusion (OD layer 3) in emitted GDS to measure",
        )
    sat = th.devregion_density_lo <= value <= th.devregion_density_hi
    return MetricVerdict(
        name="device_region_density", value=value, expected=None,
        threshold=band, satisfied=sat, observable=True,
        detail=(f"merged OD area / OD bbox area = {value:.4f} "
                f"(below floor = spread; above ceiling = over-compacted)"),
    )


def verdict_np_interleave(
    value: Optional[float],
    th: Optional[FidelityThresholds] = None,
) -> MetricVerdict:
    th = th or fidelity_thresholds()
    band = (f"[{th.np_interleave_lo}, {th.np_interleave_hi}] "
            "(CLAY_FIDELITY_NP_INTERLEAVE_LO/HI)")
    if value is None:
        return MetricVerdict(
            name="np_interleave", value=None, expected=None,
            threshold=band, satisfied=None, observable=False,
            detail="no NWELL (layer 2) or OD (layer 3) in emitted GDS",
        )
    sat = th.np_interleave_lo <= value <= th.np_interleave_hi
    return MetricVerdict(
        name="np_interleave", value=value, expected=None,
        threshold=band, satisfied=sat, observable=True,
        detail=(f"NWELL bbox / OD bbox = {value:.4f} "
                f"(below floor = N/P segregated rows; "
                f"above ceiling = well sprawls past devices)"),
    )


# --------------------------------------------------------------------------- #
# aggregate + 5-state obligation verdict.
# --------------------------------------------------------------------------- #
def measure_compactness(gds_path: str,
                        th: Optional[FidelityThresholds] = None
                        ) -> CompactnessReport:
    """Measure both compactness metrics over ONE emitted GDS (leakage-free).

    The GDS is the artifact-under-test's OWN emitted layout (the single
    canonical artifact). The reference is NOT an input; the bands come from
    config (``th`` or the resolved default)."""
    th = th or fidelity_thresholds()
    density, od_bb = measure_device_region_density(gds_path)
    interleave, nw_bb, od_bb2 = measure_np_interleave(gds_path)
    metrics = (
        verdict_device_region_density(density, th),
        verdict_np_interleave(interleave, th),
    )
    return CompactnessReport(
        device_region_density=density,
        np_interleave=interleave,
        od_bbox_um2=od_bb if od_bb is not None else od_bb2,
        nwell_bbox_um2=nw_bb,
        metrics=metrics,
    )


def compactness_verdict(gds_path: str,
                        th: Optional[FidelityThresholds] = None
                        ) -> Tuple[str, CompactnessReport]:
    """The COMPACTNESS obligation's 5-state verdict over an emitted GDS.

    Returns (verdict_string, report) where verdict_string is one of:
      ``pass``        -- both metrics observable AND inside their bands;
      ``fail``        -- at least one observable metric is OUTSIDE its band;
      ``unmeasurable``-- a metric could not be measured (no OD / no NWELL) and
                         no observable metric failed (never a silent pass: an
                         unmeasured obligation is reported, not assumed green);
      ``blocked``     -- reserved for an integration gap (the emitted GDS could
                         not be produced/opened); raised by the caller, not here.

    No silent pass: an unobservable metric NEVER counts as satisfied."""
    report = measure_compactness(gds_path, th)
    any_fail = any(m.satisfied is False for m in report.metrics)
    any_unobservable = any(not m.observable for m in report.metrics)
    if any_fail:
        return "fail", report
    if any_unobservable:
        return "unmeasurable", report
    return "pass", report


def compactness_pack():
    """The COMPACTNESS SpecPack (two QUALITY-tier obligations).

    Both rules are ``fidelity`` hardness (the gating geometric-quality class)
    and derive to the QUALITY tier -- NOT advisory. A RED measurement is a
    BLOCKING FAIL: a legal-but-spread layout (DRC/LVS clean yet density 0.24 /
    n-well bbox << device bbox) cannot be reported green. The pack DECLARES the
    obligations + names their checkers; the verdict is produced by
    :func:`measure_compactness` over the emitted GDS."""
    from clay.layout.spec.geometric_fidelity import _detect_whole_design, _emit_noop
    from clay.layout.spec.packs import SpecPack, SpecRule

    def rule(name, metric, effect, checker):
        return SpecRule(
            name=name, metric=metric, effect=effect, object="cell",
            hardness="fidelity", detect=_detect_whole_design,
            emit=_emit_noop, checker=checker, tier="QUALITY",
        )

    return SpecPack(
        name="compactness",
        applies_to=["*"],
        rules=[
            rule("device_region_density", "device_region_density",
                 "placement_compactness", "compactness_device_region_density"),
            rule("np_interleave", "np_interleave",
                 "placement_np_interleave", "compactness_np_interleave"),
        ],
    )


__all__ = [
    "CompactnessReport",
    "measure_device_region_density",
    "measure_np_interleave",
    "verdict_device_region_density",
    "verdict_np_interleave",
    "measure_compactness",
    "compactness_verdict",
    "compactness_pack",
]
