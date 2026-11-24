"""Layout statistics dashboard.

Combines existing per-feature analyzers (parasitic extraction, congestion
estimator, optional klayout DRC) into a single ``LayoutStats`` summary
for a ``ParameterizedLibrary``. Pure aggregation — does not modify any of
the underlying APIs (``extract_parasitic``, ``estimate_congestion``,
``run_drc``) and does not introduce new physical models. All numeric
totals come from the upstream analyzers; this module only walks the
library, groups counts, and renders text/Markdown reports.

DRC handling:
    The DRC pass is opt-in (``run_drc_check=True`` by default). It only
    fires when both ``KLAYOUT_AVAILABLE`` is True and the Rust CLI
    ``cargo_binary_available()`` returns True (DRC needs a real binary
    GDS file). When either dependency is missing, or when the caller
    passes ``run_drc_check=False``, ``drc_clean`` and
    ``drc_violation_count`` remain ``None`` — the field is tri-state so
    callers can distinguish "not run" from "run and passed".
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .backend import LayerPolygon, ParameterizedLibrary
from .binary_gds_subprocess import (
    cargo_binary_available,
    emit_binary_gds_via_subprocess,
)
from .congestion import estimate_congestion
from .drc import KLAYOUT_AVAILABLE, SKY130_MIN_DRC_RULES, run_drc
from .parasitic import extract_parasitic


__all__ = [
    "LayoutStats",
    "compute_layout_stats",
    "render_stats_text",
    "render_stats_markdown",
]


@dataclass
class LayoutStats:
    """Aggregated layout metrics for a ParameterizedLibrary.

    Attributes:
        library_name: name of the library being summarized.
        cell_count: number of cells in the library.
        total_polygons: sum of polygons over all cells.
        layer_polygon_counts: ``(layer, datatype)`` -> polygon count,
            grouped across every cell in the library.
        parasitic_total_fF: total estimated parasitic capacitance in
            femtofarads, derived from ``extract_parasitic`` (which sums
            in attofarads; this field is /1000).
        routing_congestion_crossings: number of horizontal/vertical wire
            crossings as reported by ``estimate_congestion`` on the flat
            list of all polygons in the library.
        drc_clean: tri-state DRC result. ``None`` means DRC was not run
            (either ``run_drc_check=False`` or klayout/Rust binary
            missing). ``True``/``False`` reflect a real DRC pass.
        drc_violation_count: paired with ``drc_clean``. ``None`` when
            DRC didn't run; an integer (0 means clean) otherwise.
    """

    library_name: str
    cell_count: int
    total_polygons: int
    layer_polygon_counts: Dict[Tuple[int, int], int]
    parasitic_total_fF: float
    routing_congestion_crossings: int
    drc_clean: Optional[bool] = None
    drc_violation_count: Optional[int] = None


def _flatten_polygons(lib: ParameterizedLibrary) -> List[LayerPolygon]:
    """Return every ``LayerPolygon`` in every cell as a flat list."""
    flat: List[LayerPolygon] = []
    for cell in lib.cells:
        flat.extend(cell.polygons)
    return flat


def _group_layer_counts(
    lib: ParameterizedLibrary,
) -> Dict[Tuple[int, int], int]:
    """Group polygon counts by ``(layer, datatype)`` over the whole library."""
    counts: Dict[Tuple[int, int], int] = {}
    for cell in lib.cells:
        for lp in cell.polygons:
            key = (lp.layer, lp.datatype)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _maybe_run_drc(
    lib: ParameterizedLibrary,
) -> Tuple[Optional[bool], Optional[int]]:
    """Attempt a DRC pass. Returns ``(None, None)`` if dependencies are
    missing or the pass cannot complete; otherwise the actual result.

    Uses ``SKY130_MIN_DRC_RULES`` since the dashboard exists primarily
    to summarize sky130-style libraries; rules silently skip layers
    that aren't present in the library, so this is safe to run against
    gf180mcu libraries too (it will just report zero violations because
    no rules match).
    """
    if not KLAYOUT_AVAILABLE:
        return (None, None)
    if not cargo_binary_available():
        return (None, None)
    # We need a temp .gds path; tempfile.NamedTemporaryFile on Windows
    # cannot be reopened by another process while the handle is open,
    # so use TemporaryDirectory and place a file inside it.
    try:
        with tempfile.TemporaryDirectory(prefix="clay_stats_drc_") as td:
            gds_path = Path(td) / f"{lib.name or 'lib'}.gds"
            emit_binary_gds_via_subprocess(lib, gds_path)
            report = run_drc(gds_path, SKY130_MIN_DRC_RULES)
    except Exception:
        # Emission or DRC failed for some environmental reason; surface as
        # "not run" rather than crashing the whole dashboard.
        return (None, None)
    return (report.clean, len(report.violations))


def compute_layout_stats(
    lib: ParameterizedLibrary,
    pdk_layers,
    run_drc_check: bool = True,
) -> LayoutStats:
    """Walk a ``ParameterizedLibrary`` and assemble a ``LayoutStats``.

    Args:
        lib: the library to summarize.
        pdk_layers: a PDK ``LayerMap`` (``Sky130LayerMap`` /
            ``Gf180mcuLayerMap``) passed straight through to
            ``extract_parasitic``.
        run_drc_check: when True (default), attempt to run klayout DRC
            via the Rust GDS emitter. When False, ``drc_clean`` and
            ``drc_violation_count`` are left as ``None``.

    Returns:
        A ``LayoutStats`` dataclass.
    """
    layer_counts = _group_layer_counts(lib)
    total_polygons = sum(layer_counts.values())
    cell_count = len(lib.cells)

    parasitic_report = extract_parasitic(lib, pdk_layers)
    parasitic_total_fF = parasitic_report.total_cap_aF / 1000.0

    congestion_report = estimate_congestion(_flatten_polygons(lib))
    crossings = congestion_report.crossings

    drc_clean: Optional[bool] = None
    drc_violation_count: Optional[int] = None
    if run_drc_check:
        drc_clean, drc_violation_count = _maybe_run_drc(lib)

    return LayoutStats(
        library_name=lib.name,
        cell_count=cell_count,
        total_polygons=total_polygons,
        layer_polygon_counts=layer_counts,
        parasitic_total_fF=parasitic_total_fF,
        routing_congestion_crossings=crossings,
        drc_clean=drc_clean,
        drc_violation_count=drc_violation_count,
    )


def _drc_status_text(stats: LayoutStats) -> str:
    if stats.drc_clean is None:
        return "not run"
    if stats.drc_clean:
        return f"clean (0 violations)"
    return f"violations: {stats.drc_violation_count}"


def render_stats_text(stats: LayoutStats) -> str:
    """Render a multi-line ASCII summary of a ``LayoutStats``.

    The layout is deliberately verbose so the output can be embedded in
    a release note or CI log; each metric occupies its own line.
    """
    lines: List[str] = []
    lines.append("Layout Statistics Dashboard")
    lines.append("=" * 40)
    lines.append(f"Library: {stats.library_name}")
    lines.append(f"Cells: {stats.cell_count}")
    lines.append(f"Total polygons: {stats.total_polygons}")
    lines.append(f"Parasitic total: {stats.parasitic_total_fF:.4f} fF")
    lines.append(
        f"Routing congestion crossings: {stats.routing_congestion_crossings}"
    )
    lines.append(f"DRC: {_drc_status_text(stats)}")
    lines.append("")
    lines.append("Layer polygon counts:")
    if not stats.layer_polygon_counts:
        lines.append("  (none)")
    else:
        # Sort by (layer, datatype) for stable output.
        for (layer, datatype), count in sorted(stats.layer_polygon_counts.items()):
            lines.append(f"  ({layer}, {datatype}): {count}")
    return "\n".join(lines)


def render_stats_markdown(stats: LayoutStats) -> str:
    """Render a Markdown summary of a ``LayoutStats``.

    The output uses an ATX ``##`` heading for the dashboard title and a
    Markdown table for the per-layer breakdown, so it drops cleanly into
    a README or release-note section.
    """
    lines: List[str] = []
    lines.append(f"## Layout Statistics: {stats.library_name}")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Cells | {stats.cell_count} |")
    lines.append(f"| Total polygons | {stats.total_polygons} |")
    lines.append(
        f"| Parasitic total (fF) | {stats.parasitic_total_fF:.4f} |"
    )
    lines.append(
        f"| Routing congestion crossings | {stats.routing_congestion_crossings} |"
    )
    lines.append(f"| DRC | {_drc_status_text(stats)} |")
    lines.append("")
    lines.append("### Layer polygon counts")
    lines.append("")
    lines.append("| Layer | Datatype | Polygon count |")
    lines.append("| ---: | ---: | ---: |")
    if not stats.layer_polygon_counts:
        lines.append("| (none) | (none) | 0 |")
    else:
        for (layer, datatype), count in sorted(stats.layer_polygon_counts.items()):
            lines.append(f"| {layer} | {datatype} | {count} |")
    return "\n".join(lines)
