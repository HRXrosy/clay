"""Phase 14 benchmark report renderers (text-format output).

This module provides pure functions that turn a
``clay.layout.template_benchmark.BenchmarkReport`` into human / machine
readable text so it can be embedded in docs (Markdown), exported for
spreadsheets (CSV), or pasted into release notes as a one-line summary.

The renderers are non-mutating: they read a ``BenchmarkReport`` and
return a ``str``. They do not modify the report or its benchmarks, and
they do not touch the underlying Phase 4 cosizing or Phase 6 template
APIs.

Column conventions (Markdown + CSV):
    Template | Devices | Pareto size | Min sigma (mV) | Min area (um^2)

Numeric formatting:
    ``min_sigma_mv`` and ``min_area_um2`` are formatted with two
    fractional digits (``f"{x:.2f}"``); ``float('inf')`` is rendered as
    the literal ``"inf"`` (and ``-inf`` / ``nan`` are passed through
    Python's default repr) so empty Pareto fronts do not silently lose
    information.
"""
from __future__ import annotations

import math
from typing import Iterable, List

from .template_benchmark import BenchmarkReport, TemplateBenchmark


# ---------------------------------------------------------------------------
# Column metadata (kept in one place so all three renderers stay in sync).
# ---------------------------------------------------------------------------
_MD_COLUMNS = (
    "Template",
    "Devices",
    "Pareto size",
    "Min σ (mV)",
    "Min area (µm²)",
)
_CSV_COLUMNS = (
    "template",
    "devices",
    "pareto_size",
    "min_sigma_mv",
    "min_area_um2",
)


def _fmt_float(value: float) -> str:
    """Format a float with two decimal places.

    ``inf`` / ``-inf`` / ``nan`` are rendered with Python's default
    string form (``"inf"`` / ``"-inf"`` / ``"nan"``) so an empty Pareto
    front surfaces visibly in the report rather than being printed as
    ``"inf.00"``.
    """
    if math.isinf(value) or math.isnan(value):
        return str(value)
    return f"{value:.2f}"


def _markdown_row(cells: Iterable[str]) -> str:
    """Compose one Markdown table row from a sequence of stringified cells."""
    return "| " + " | ".join(cells) + " |"


# ---------------------------------------------------------------------------
# Markdown renderer.
# ---------------------------------------------------------------------------
def render_markdown_table(report: BenchmarkReport) -> str:
    """Render ``report`` as a Markdown table.

    The table always has a header row and an alignment row (``|---|``
    per column). An empty report (no benchmarks) returns just the header
    + alignment row with no data rows.

    Example output::

        | Template | Devices | Pareto size | Min σ (mV) | Min area (µm²) |
        |----------|---------|-------------|------------|----------------|
        | diff_pair | 2 | 5 | 0.31 | 1.00 |
    """
    lines: List[str] = []
    lines.append(_markdown_row(_MD_COLUMNS))
    # Alignment row: a column of dashes per header column. Width is
    # padded to the visible header length so the raw Markdown source is
    # still readable in a plain-text editor.
    align_cells = ["-" * max(3, len(col)) for col in _MD_COLUMNS]
    lines.append(_markdown_row(align_cells))

    for bench in report.benchmarks:
        lines.append(
            _markdown_row(
                (
                    bench.template_name,
                    str(bench.device_count),
                    str(bench.pareto_size),
                    _fmt_float(bench.min_sigma_mv),
                    _fmt_float(bench.min_area_um2),
                )
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV renderer.
# ---------------------------------------------------------------------------
def render_csv(report: BenchmarkReport) -> str:
    """Render ``report`` as comma-separated values (one row per template).

    The first line is the column header; one row per ``TemplateBenchmark``
    follows. We deliberately do not depend on ``csv.writer`` because the
    fields here are simple identifiers + numerics with no embedded
    commas / quotes / newlines (template names come from the factory
    registry).
    """
    lines: List[str] = []
    lines.append(",".join(_CSV_COLUMNS))
    for bench in report.benchmarks:
        lines.append(
            ",".join(
                (
                    bench.template_name,
                    str(bench.device_count),
                    str(bench.pareto_size),
                    _fmt_float(bench.min_sigma_mv),
                    _fmt_float(bench.min_area_um2),
                )
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One-paragraph summary renderer.
# ---------------------------------------------------------------------------
def _min_across(
    benchmarks: List[TemplateBenchmark], attr: str
) -> float:
    """Smallest value of ``attr`` across all benchmarks; ``inf`` if empty."""
    if not benchmarks:
        return float("inf")
    return min(getattr(b, attr) for b in benchmarks)


def render_summary(report: BenchmarkReport) -> str:
    """Render a one-paragraph summary of the report.

    Mentions the PDK name, the number of templates benchmarked, and the
    minimum ``sigma_mv`` and ``area_um2`` corners observed across all
    templates. Useful as a release-note blurb or commit-message line.
    """
    n_templates = len(report.benchmarks)
    min_sigma = _min_across(report.benchmarks, "min_sigma_mv")
    min_area = _min_across(report.benchmarks, "min_area_um2")
    return (
        f"PDK {report.pdk_name}: benchmarked {n_templates} templates; "
        f"minimum σ = {_fmt_float(min_sigma)} mV, "
        f"minimum area = {_fmt_float(min_area)} µm² "
        f"across the Pareto fronts."
    )
