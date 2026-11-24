"""Phase 14 cell template benchmark utility.

For each cell template factory in ``clay.layout.cell_templates``, build a
representative instance, count the lowered device slots, then run the
Phase 4 (W x L x finger_count) Pareto co-sizing sweep on the input pair
sizing surface. The Pareto front is extracted and the two extreme corners
(minimum sigma_mv, minimum area_um2) are reported per template so users can
compare cell topologies on the same PDK on equal terms.

This is a benchmark / reporting layer; it does not modify any of the
underlying Phase 4 cosizing or Phase 6 cell template APIs.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from .annotations import AnnotatedCell
from .backend import DeviceSizing
from .cell_templates import (
    build_5t_ota,
    build_bandgap_reference,
    build_cascoded_current_mirror,
    build_current_mirror,
    build_diff_pair,
    build_folded_cascode,
    build_telescopic_cascode,
    build_two_stage_miller,
)
from .cosizing import CosizingPoint, joint_sweep, pareto_optimal
from .pdk_dims import DeviceType, PdkDims


@dataclass(frozen=True)
class TemplateBenchmark:
    """Per-template benchmark result.

    Attributes:
        template_name: name of the build factory (e.g. ``"diff_pair"``).
        device_count: number of device slots in the built cell.
        pareto_size: number of points on the cosizing Pareto front.
        min_sigma_mv: smallest sigma_delta_vth across the Pareto front
            (best-mismatch extreme corner). ``float('inf')`` if Pareto is
            empty.
        min_area_um2: smallest area across the Pareto front (smallest-area
            extreme corner). ``float('inf')`` if Pareto is empty.
    """

    template_name: str
    device_count: int
    pareto_size: int
    min_sigma_mv: float
    min_area_um2: float


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregate benchmark over all known cell template factories on a
    given PDK / device-type / fin-pitch context.
    """

    pdk_name: str
    benchmarks: List[TemplateBenchmark] = field(default_factory=list)


def _count_devices(cell: AnnotatedCell, sizing: DeviceSizing) -> int:
    """Device count = number of named slots in the lowered cell.

    Both ``cell.layouts`` and ``sizing`` should index the same set of
    devices; we union them defensively in case a factory stores symbolic
    layout slots without explicit sizing (e.g. capacitor placeholders).
    """
    slots = set(cell.layouts.keys()) | set(sizing.keys())
    return len(slots)


def _front_extremes(front: List[CosizingPoint]) -> Tuple[float, float]:
    """Return (min_sigma_mv, min_area_um2) over a Pareto front.

    Empty fronts emit a warning and return (inf, inf). The caller (this
    module's ``benchmark_template``) decides whether to surface this
    upstream; we keep it conservative so summary rows still type-check.
    """
    if not front:
        warnings.warn(
            "empty Pareto front: cosizing grid produced no points",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("inf"), float("inf")
    min_sigma = min(p.sigma_mv for p in front)
    min_area = min(p.area_um2 for p in front)
    return min_sigma, min_area


def benchmark_template(
    name: str,
    build_fn: Callable[..., Tuple[AnnotatedCell, DeviceSizing]],
    kwargs: Dict[str, Any],
    pdk: PdkDims,
    pdk_layers: Any,
    device_type: DeviceType,
    fin_pitch_nm: float,
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> TemplateBenchmark:
    """Benchmark one cell template.

    Args:
        name: short name for the report (e.g. ``"diff_pair"``).
        build_fn: a factory from ``clay.layout.cell_templates`` returning
            ``(AnnotatedCell, DeviceSizing)``.
        kwargs: keyword arguments forwarded to ``build_fn`` so it can be
            called as ``build_fn(name=..., **kwargs)``.
        pdk: PDK dimensional surface (Sky130 / Gf180mcu).
        pdk_layers: PDK layer map (accepted for future use; not used by
            the Phase 4 cosizing sweep itself).
        device_type: device type the input pair will be sized against
            (NMOS for n-input pairs; PMOS for p-input pairs).
        fin_pitch_nm: fin pitch hint (accepted for future use; not used
            by the Phase 4 cosizing sweep itself).
        w_range / l_range / finger_range: cosizing sweep axes.

    Returns:
        ``TemplateBenchmark`` summarizing device count + Pareto extremes.

    Notes:
        ``pdk_layers`` and ``fin_pitch_nm`` are accepted so callers can
        pass the same PDK context they use elsewhere; the joint sweep
        derives Avt from ``pdk`` and the grid, which is sufficient at the
        Phase 4 level. Future Phase 14+ extensions may incorporate them.
    """
    # Build the cell using its factory.
    cell, sizing = build_fn(name=name, **kwargs)
    device_count = _count_devices(cell, sizing)

    # Run joint Pareto sweep over the input pair sizing surface.
    points = joint_sweep(w_range, l_range, finger_range, pdk, device_type)
    front = pareto_optimal(points)
    min_sigma, min_area = _front_extremes(front)

    return TemplateBenchmark(
        template_name=name,
        device_count=device_count,
        pareto_size=len(front),
        min_sigma_mv=min_sigma,
        min_area_um2=min_area,
    )


# Registry of (factory, minimal kwargs) for all known cell templates.
# kwargs are intentionally minimal: only positional-required arguments for
# the input pair / shared device sizing, since the cosizing sweep itself
# decides the actual swept range.
_TEMPLATE_REGISTRY: List[Tuple[str, Callable[..., Tuple[AnnotatedCell, DeviceSizing]], Dict[str, Any]]] = [
    (
        "diff_pair",
        build_diff_pair,
        {"w_um": 4.0, "l_um": 1.0, "finger_count": 4},
    ),
    (
        "current_mirror",
        build_current_mirror,
        {"w_um": 4.0, "l_um": 1.0, "finger_count_ref": 4, "finger_count_out": 4},
    ),
    (
        "cascoded_mirror",
        build_cascoded_current_mirror,
        {
            "w_um_low": 4.0,
            "l_um_low": 1.0,
            "w_um_cas": 4.0,
            "l_um_cas": 0.5,
            "finger_count_low": 4,
            "finger_count_cas": 4,
        },
    ),
    (
        "5t_ota",
        build_5t_ota,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    (
        "folded_cascode",
        build_folded_cascode,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    (
        "telescopic_cascode",
        build_telescopic_cascode,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    (
        "two_stage_miller",
        build_two_stage_miller,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    (
        "bandgap_reference",
        build_bandgap_reference,
        {},  # All defaults; bandgap factory does not take input pair sizing positionally
    ),
]


def benchmark_all_templates(
    pdk: PdkDims,
    pdk_layers: Any,
    device_type: DeviceType,
    fin_pitch_nm: float,
    w_range: List[float] = None,
    l_range: List[float] = None,
    finger_range: List[int] = None,
) -> BenchmarkReport:
    """Run ``benchmark_template`` for every known factory.

    Returns a ``BenchmarkReport`` ordered to match the factory registry:
    diff_pair, current_mirror, cascoded_mirror, 5t_ota, folded_cascode,
    telescopic_cascode, two_stage_miller, bandgap_reference (8 entries).
    """
    if w_range is None:
        w_range = [2.0, 4.0, 8.0]
    if l_range is None:
        l_range = [0.5, 1.0]
    if finger_range is None:
        finger_range = [1, 4, 16]

    benchmarks: List[TemplateBenchmark] = []
    for name, build_fn, kwargs in _TEMPLATE_REGISTRY:
        bench = benchmark_template(
            name=name,
            build_fn=build_fn,
            kwargs=kwargs,
            pdk=pdk,
            pdk_layers=pdk_layers,
            device_type=device_type,
            fin_pitch_nm=fin_pitch_nm,
            w_range=w_range,
            l_range=l_range,
            finger_range=finger_range,
        )
        benchmarks.append(bench)

    return BenchmarkReport(pdk_name=pdk.name(), benchmarks=benchmarks)
