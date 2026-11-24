"""Side-by-side comparison utility for two cell templates.

This module sits on top of the Phase 14 :mod:`template_benchmark` layer and
runs the same cosizing sweep (W x L x finger_count) for two named cell
templates on the same PDK context, then emits a four-row diff report
covering device count, Pareto front size, the best (lowest) sigma_mv, and
the best (lowest) area_um2.

The comparison is intentionally simple: "lower is better" for all four
metrics, so the winner of each row is the template whose value is
numerically smaller. Equal values produce a ``"tie"`` row. The renderer is
a fixed-width text table suitable for printing into logs or test output.

This is a reporting / convenience layer; it does not modify any of the
underlying cell template factories, joint sweep, or Pareto front APIs.
"""
from __future__ import annotations

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
from .cosizing import joint_sweep, pareto_optimal
from .pdk_dims import DeviceType, PdkDims


# Registry shared in spirit with template_benchmark: keep the kwargs the
# same so the two layers agree on what "minimal" build args look like.
_TEMPLATE_REGISTRY: Dict[
    str, Tuple[Callable[..., Tuple[AnnotatedCell, DeviceSizing]], Dict[str, Any]]
] = {
    "diff_pair": (
        build_diff_pair,
        {"w_um": 4.0, "l_um": 1.0, "finger_count": 4},
    ),
    "current_mirror": (
        build_current_mirror,
        {"w_um": 4.0, "l_um": 1.0, "finger_count_ref": 4, "finger_count_out": 4},
    ),
    "cascoded_mirror": (
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
    "5t_ota": (
        build_5t_ota,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    "folded_cascode": (
        build_folded_cascode,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    "telescopic_cascode": (
        build_telescopic_cascode,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    "two_stage_miller": (
        build_two_stage_miller,
        {"input_w_um": 4.0, "input_l_um": 1.0, "input_finger_count": 4},
    ),
    "bandgap_reference": (
        build_bandgap_reference,
        {},
    ),
}


# Metric labels used in both the data rows and the rendered text table.
_METRIC_LABELS: Tuple[str, str, str, str] = (
    "device_count",
    "pareto_size",
    "min_sigma_mv",
    "min_area_um2",
)


@dataclass(frozen=True)
class ComparisonRow:
    """One metric's side-by-side row.

    Attributes:
        metric: short metric label (one of ``_METRIC_LABELS``).
        value_a: value computed for template A.
        value_b: value computed for template B.
        winner: ``"a"`` if A's value is strictly smaller, ``"b"`` if B's
            value is strictly smaller, ``"tie"`` if the two values are
            equal (under direct numeric equality).
    """

    metric: str
    value_a: Any
    value_b: Any
    winner: str


@dataclass(frozen=True)
class TemplateComparison:
    """Aggregate comparison between two templates over four metrics."""

    name_a: str
    name_b: str
    rows: List[ComparisonRow] = field(default_factory=list)


def _measure_template(
    name: str,
    pdk: PdkDims,
    pdk_layers: Any,
    device_type: DeviceType,
    fin_pitch_nm: float,
    w_range: List[float],
    l_range: List[float],
    finger_range: List[int],
) -> Tuple[int, int, float, float]:
    """Build + sweep one template, return (devices, pareto, min_sigma, min_area).

    Empty Pareto fronts return ``(devices, 0, inf, inf)`` rather than
    warning; comparing two empty fronts would otherwise raise on min().
    """
    if name not in _TEMPLATE_REGISTRY:
        raise ValueError(
            f"unknown template {name!r}; "
            f"known templates: {sorted(_TEMPLATE_REGISTRY.keys())}"
        )
    build_fn, kwargs = _TEMPLATE_REGISTRY[name]
    cell, sizing = build_fn(name=name, **kwargs)
    device_count = len(set(cell.layouts.keys()) | set(sizing.keys()))

    points = joint_sweep(w_range, l_range, finger_range, pdk, device_type)
    front = pareto_optimal(points)
    if front:
        min_sigma = min(p.sigma_mv for p in front)
        min_area = min(p.area_um2 for p in front)
    else:
        min_sigma = float("inf")
        min_area = float("inf")
    return device_count, len(front), min_sigma, min_area


def _winner(value_a: Any, value_b: Any) -> str:
    """Return ``"a"`` / ``"b"`` / ``"tie"`` for the lower-is-better rule.

    NaN-safe insofar as both branches use direct ``<`` comparison: NaN
    propagates to ``"tie"`` since both ``<`` checks are False.
    """
    if value_a < value_b:
        return "a"
    if value_b < value_a:
        return "b"
    return "tie"


def compare_templates(
    name_a: str,
    name_b: str,
    pdk: PdkDims,
    pdk_layers: Any,
    device_type: DeviceType,
    fin_pitch_nm: float,
    w_range: List[float] = None,
    l_range: List[float] = None,
    finger_range: List[int] = None,
) -> TemplateComparison:
    """Run joint sweep + Pareto front for both templates and diff them.

    Args:
        name_a / name_b: template registry keys (e.g. ``"diff_pair"``).
            Unknown names raise ``ValueError`` with the list of known
            templates.
        pdk: PDK dimensional surface (Sky130 / Gf180mcu).
        pdk_layers: PDK layer map (forwarded for context parity; not used
            by the Phase 4 cosizing sweep itself).
        device_type: ``DeviceType.NMOS`` / ``DeviceType.PMOS`` for the
            input-pair sweep.
        fin_pitch_nm: fin pitch hint (forwarded for context parity).
        w_range / l_range / finger_range: cosizing sweep axes; default to
            ``[2.0, 4.0, 8.0]`` / ``[0.5, 1.0]`` / ``[1, 4, 16]`` to match
            the Phase 14 benchmark defaults.

    Returns:
        ``TemplateComparison`` with exactly four rows, one per metric in
        ``_METRIC_LABELS``. The winner is the template with the smaller
        value (lower-is-better for all four metrics).
    """
    if w_range is None:
        w_range = [2.0, 4.0, 8.0]
    if l_range is None:
        l_range = [0.5, 1.0]
    if finger_range is None:
        finger_range = [1, 4, 16]

    dev_a, par_a, sigma_a, area_a = _measure_template(
        name_a, pdk, pdk_layers, device_type, fin_pitch_nm,
        w_range, l_range, finger_range,
    )
    dev_b, par_b, sigma_b, area_b = _measure_template(
        name_b, pdk, pdk_layers, device_type, fin_pitch_nm,
        w_range, l_range, finger_range,
    )

    rows: List[ComparisonRow] = [
        ComparisonRow("device_count", dev_a, dev_b, _winner(dev_a, dev_b)),
        ComparisonRow("pareto_size", par_a, par_b, _winner(par_a, par_b)),
        ComparisonRow("min_sigma_mv", sigma_a, sigma_b, _winner(sigma_a, sigma_b)),
        ComparisonRow("min_area_um2", area_a, area_b, _winner(area_a, area_b)),
    ]
    return TemplateComparison(name_a=name_a, name_b=name_b, rows=rows)


def render_comparison_text(comp: TemplateComparison) -> str:
    """Render a fixed-width multi-line text table for a comparison.

    The table has one header line, one separator line, and one line per
    metric row. Numeric values are formatted with up to 6 significant
    digits (via ``"{:g}"``); ints are formatted as plain integers.

    Args:
        comp: ``TemplateComparison`` to render.

    Returns:
        A multi-line string ending with a trailing newline. Template
        names appear in the header so they are searchable.
    """

    def _fmt(value: Any) -> str:
        if isinstance(value, bool):
            return "T" if value else "F"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    # Build all cell strings up front, then compute column widths so the
    # output looks like a real table regardless of value magnitudes.
    metric_strs = [r.metric for r in comp.rows]
    a_strs = [_fmt(r.value_a) for r in comp.rows]
    b_strs = [_fmt(r.value_b) for r in comp.rows]
    win_strs = [r.winner for r in comp.rows]

    metric_header = "metric"
    a_header = comp.name_a
    b_header = comp.name_b
    win_header = "winner"

    w_metric = max(len(metric_header), *(len(s) for s in metric_strs))
    w_a = max(len(a_header), *(len(s) for s in a_strs))
    w_b = max(len(b_header), *(len(s) for s in b_strs))
    w_win = max(len(win_header), *(len(s) for s in win_strs))

    def _row(metric: str, va: str, vb: str, win: str) -> str:
        return (
            f"{metric:<{w_metric}}  "
            f"{va:>{w_a}}  "
            f"{vb:>{w_b}}  "
            f"{win:<{w_win}}"
        )

    lines: List[str] = []
    lines.append(_row(metric_header, a_header, b_header, win_header))
    lines.append(
        f"{'-' * w_metric}  {'-' * w_a}  {'-' * w_b}  {'-' * w_win}"
    )
    for metric, va, vb, win in zip(metric_strs, a_strs, b_strs, win_strs):
        lines.append(_row(metric, va, vb, win))

    return "\n".join(lines) + "\n"
