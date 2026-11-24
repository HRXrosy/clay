"""Self-contained HTML report generator for layout utility outputs.

Combines outputs from existing layout APIs into a single multi-section
HTML page intended as a release artifact / CI dashboard snapshot:

* ``LayoutStats`` (clay.layout.stats_dashboard) — per-library aggregate
  numbers (cell count, polygons, parasitic total, congestion, DRC).
* ``LibraryParasiticReport`` (clay.layout.parasitic) — per-cell, per-layer
  parasitic capacitance breakdown.
* ``TemplateDrcReport`` (clay.layout.template_drc) — per-template DRC
  outcome with skipped-reason annotation.
* ``template_list`` — flat list of registered template names from
  ``clay.layout.template_introspect.list_all_templates()`` (or any
  caller-curated subset).

The renderer is a pure aggregator: it does not call any of the upstream
analyzers, does not invent numbers, and does not modify any existing
API. Output is a single self-contained HTML string with inline CSS and
no external dependencies (no jinja2, no JS, no external fonts).

Public API:

* ``render_html_report(title, library_stats=None, parasitic_report=None,
  drc_report=None, template_list=None) -> str``
* ``write_html_report(output_path, **kwargs) -> Path``
* ``render_minimal_report(title) -> str``

Narrative text uses Chinese curly quotes per project convention; HTML
attribute values use ASCII double quotes (HTML syntax).
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, List, Optional, Union


__all__ = [
    "render_html_report",
    "write_html_report",
    "render_minimal_report",
]


# ---------------------------------------------------------------------------
# Inline CSS — Microsoft YaHei font, table borders, dark blue h2.
# ---------------------------------------------------------------------------
_INLINE_CSS = """
body {
    font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
    margin: 2em;
    color: #222;
    background: #fafafa;
}
h1 {
    color: #000;
    border-bottom: 2px solid #1a3a6c;
    padding-bottom: 0.3em;
}
h2 {
    color: #1a3a6c;
    margin-top: 1.5em;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.2em;
}
h3 {
    color: #2a4a7c;
    margin-top: 1em;
}
table {
    border-collapse: collapse;
    margin: 0.5em 0 1em 0;
}
table, th, td {
    border: 1px solid #888;
}
th, td {
    padding: 4px 10px;
    text-align: left;
}
th {
    background: #e8edf3;
    color: #1a3a6c;
}
.section {
    margin-bottom: 2em;
}
.muted {
    color: #666;
    font-style: italic;
}
code {
    background: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
    font-family: Consolas, "Courier New", monospace;
}
ul.template-list {
    margin: 0.5em 0 1em 1.5em;
}
""".strip()


def _esc(value: Any) -> str:
    """HTML-escape any value (cast through ``str`` first)."""
    return html.escape(str(value), quote=True)


def _wrap_document(title: str, body_inner: str) -> str:
    """Wrap inner body markup with the full HTML5 boilerplate."""
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"UTF-8\">\n"
        f"<title>{_esc(title)}</title>\n"
        "<style>\n"
        f"{_INLINE_CSS}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{_esc(title)}</h1>\n"
        f"{body_inner}"
        "</body>\n"
        "</html>\n"
    )


# ---------------------------------------------------------------------------
# Section renderers — each returns either a ``<section>...</section>`` block
# or an empty string when the corresponding input is ``None``.
# ---------------------------------------------------------------------------
def _render_drc_status(stats: Any) -> str:
    """Tri-state DRC status: not run / clean / violations."""
    clean = getattr(stats, "drc_clean", None)
    count = getattr(stats, "drc_violation_count", None)
    if clean is None:
        return "<span class=\"muted\">not run</span>"
    if clean:
        return "clean (0 violations)"
    return f"violations: {_esc(count)}"


def _render_library_stats_section(stats: Any) -> str:
    """Render LayoutStats as <section> with summary table + layer breakdown."""
    if stats is None:
        return ""
    rows: List[str] = []
    rows.append(f"<tr><th>Library</th><td>{_esc(stats.library_name)}</td></tr>")
    rows.append(f"<tr><th>Cells</th><td>{_esc(stats.cell_count)}</td></tr>")
    rows.append(
        f"<tr><th>Total polygons</th><td>{_esc(stats.total_polygons)}</td></tr>"
    )
    rows.append(
        "<tr><th>Parasitic total (fF)</th>"
        f"<td>{stats.parasitic_total_fF:.4f}</td></tr>"
    )
    rows.append(
        "<tr><th>Routing congestion crossings</th>"
        f"<td>{_esc(stats.routing_congestion_crossings)}</td></tr>"
    )
    rows.append(
        f"<tr><th>DRC</th><td>{_render_drc_status(stats)}</td></tr>"
    )
    summary_tbl = (
        "<table class=\"stats-summary\">\n"
        + "\n".join(rows)
        + "\n</table>"
    )

    # Layer polygon breakdown
    layer_rows: List[str] = []
    layer_counts = getattr(stats, "layer_polygon_counts", {}) or {}
    if not layer_counts:
        layer_rows.append(
            "<tr><td colspan=\"3\" class=\"muted\">(no polygons)</td></tr>"
        )
    else:
        for (layer, datatype), count in sorted(layer_counts.items()):
            layer_rows.append(
                f"<tr><td>{_esc(layer)}</td>"
                f"<td>{_esc(datatype)}</td>"
                f"<td>{_esc(count)}</td></tr>"
            )
    layer_tbl = (
        "<h3>Layer polygon counts</h3>\n"
        "<table class=\"layer-counts\">\n"
        "<tr><th>Layer</th><th>Datatype</th><th>Polygon count</th></tr>\n"
        + "\n".join(layer_rows)
        + "\n</table>"
    )

    return (
        "<section class=\"section library-stats\">\n"
        "<h2>Library Statistics</h2>\n"
        "<p>聚合自 <code>compute_layout_stats</code>，覆盖单元、多边形、寄生、"
        "拥塞与 DRC 状态。</p>\n"
        f"{summary_tbl}\n"
        f"{layer_tbl}\n"
        "</section>\n"
    )


def _render_parasitic_section(report: Any) -> str:
    """Render LibraryParasiticReport as per-cell table with totals."""
    if report is None:
        return ""

    cells = getattr(report, "cells", []) or []
    lib_name = getattr(report, "library_name", "(unnamed)")
    total_aF = getattr(report, "total_cap_aF", 0.0)

    body_parts: List[str] = []
    body_parts.append(
        "<section class=\"section parasitic\">\n"
        "<h2>Parasitic Report</h2>\n"
        "<p>来自 <code>extract_parasitic</code> 的逐单元、逐层电容估计；"
        "数字单位 aF（10⁻¹⁸ F）。</p>\n"
        f"<p><strong>Library:</strong> {_esc(lib_name)} &nbsp; "
        f"<strong>Total cap:</strong> {total_aF:.2f} aF "
        f"({total_aF / 1000.0:.4f} fF)</p>\n"
    )

    if not cells:
        body_parts.append(
            "<p class=\"muted\">(no cells in parasitic report)</p>\n"
        )
    else:
        for cell in cells:
            cell_name = getattr(cell, "cell_name", "(unnamed)")
            cell_total = getattr(cell, "total_cap_aF", 0.0)
            body_parts.append(
                f"<h3>Cell: {_esc(cell_name)} "
                f"<span class=\"muted\">(total {cell_total:.2f} aF)</span></h3>\n"
            )
            per_layer = getattr(cell, "per_layer", []) or []
            if not per_layer:
                body_parts.append(
                    "<p class=\"muted\">(no layers extracted)</p>\n"
                )
                continue
            rows: List[str] = []
            for entry in per_layer:
                rows.append(
                    f"<tr><td>{_esc(entry.layer)}</td>"
                    f"<td>{_esc(entry.datatype)}</td>"
                    f"<td>{_esc(entry.logical_layer)}</td>"
                    f"<td>{entry.total_area_um2:.3f}</td>"
                    f"<td>{entry.total_perimeter_um:.3f}</td>"
                    f"<td>{entry.area_cap_aF:.2f}</td>"
                    f"<td>{entry.fringe_cap_aF:.2f}</td>"
                    f"<td>{entry.total_cap_aF:.2f}</td></tr>"
                )
            body_parts.append(
                "<table class=\"parasitic-cell\">\n"
                "<tr>"
                "<th>Layer</th><th>Datatype</th><th>Logical</th>"
                "<th>Area (μm²)</th><th>Perimeter (μm)</th>"
                "<th>Area cap (aF)</th><th>Fringe cap (aF)</th>"
                "<th>Total cap (aF)</th>"
                "</tr>\n"
                + "\n".join(rows)
                + "\n</table>\n"
            )

    body_parts.append("</section>\n")
    return "".join(body_parts)


def _render_drc_section(report: Any) -> str:
    """Render TemplateDrcReport as per-template table."""
    if report is None:
        return ""

    pdk_name = getattr(report, "pdk_name", "unknown")
    overall_clean = getattr(report, "overall_clean", False)
    results = getattr(report, "results", []) or []

    overall_label = "clean" if overall_clean else "DIRTY"
    rows: List[str] = []
    if not results:
        rows.append(
            "<tr><td colspan=\"5\" class=\"muted\">(no template results)</td></tr>"
        )
    else:
        for r in results:
            status = "clean" if r.clean else "FAIL"
            skipped = r.skipped_reason if r.skipped_reason else ""
            by_rule = getattr(r, "violations_by_rule", {}) or {}
            by_rule_str = (
                ", ".join(f"{k}={v}" for k, v in sorted(by_rule.items()))
                if by_rule
                else ""
            )
            rows.append(
                f"<tr><td>{_esc(r.template_name)}</td>"
                f"<td>{_esc(status)}</td>"
                f"<td>{_esc(r.violation_count)}</td>"
                f"<td>{_esc(by_rule_str)}</td>"
                f"<td>{_esc(skipped)}</td></tr>"
            )

    return (
        "<section class=\"section drc\">\n"
        "<h2>Template DRC Report</h2>\n"
        "<p>来自 <code>drc_check_all_templates</code>；跳过的检查（缺 klayout / "
        "Rust 二进制）以 vacuous clean 计入总体状态。</p>\n"
        f"<p><strong>PDK:</strong> {_esc(pdk_name)} &nbsp; "
        f"<strong>Overall:</strong> {_esc(overall_label)}</p>\n"
        "<table class=\"drc-results\">\n"
        "<tr><th>Template</th><th>Status</th><th>Violations</th>"
        "<th>By rule</th><th>Skipped reason</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>\n"
        "</section>\n"
    )


def _render_template_list_section(template_list: Optional[List[str]]) -> str:
    """Render template name list as <ul>."""
    if template_list is None:
        return ""

    if not template_list:
        items_html = (
            "<p class=\"muted\">(no templates registered)</p>"
        )
    else:
        items = "\n".join(
            f"  <li><code>{_esc(name)}</code></li>" for name in template_list
        )
        items_html = (
            "<ul class=\"template-list\">\n"
            f"{items}\n"
            "</ul>"
        )

    return (
        "<section class=\"section template-list-section\">\n"
        "<h2>Registered Cell Templates</h2>\n"
        "<p>来自 <code>list_all_templates()</code> 的工厂名清单。</p>\n"
        f"{items_html}\n"
        "</section>\n"
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def render_html_report(
    title: str,
    library_stats: Any = None,
    parasitic_report: Any = None,
    drc_report: Any = None,
    template_list: Optional[List[str]] = None,
) -> str:
    """Render a multi-section HTML report combining layout utility outputs.

    Args:
        title: page title (rendered into both ``<title>`` and ``<h1>``).
        library_stats: optional ``LayoutStats`` from
            ``clay.layout.stats_dashboard.compute_layout_stats``.
        parasitic_report: optional ``LibraryParasiticReport`` from
            ``clay.layout.parasitic.extract_parasitic``.
        drc_report: optional ``TemplateDrcReport`` from
            ``clay.layout.template_drc.drc_check_all_templates``.
        template_list: optional list of registered template names from
            ``clay.layout.template_introspect.list_all_templates()`` or
            any caller-curated subset.

    Returns:
        A self-contained HTML5 string with inline CSS and no external
        dependencies. Sections corresponding to ``None`` inputs are
        omitted entirely from the output.
    """
    body_parts: List[str] = []
    body_parts.append(_render_library_stats_section(library_stats))
    body_parts.append(_render_parasitic_section(parasitic_report))
    body_parts.append(_render_drc_section(drc_report))
    body_parts.append(_render_template_list_section(template_list))

    # If all four inputs are None we still produce a valid page with just
    # the <h1> — same shape as ``render_minimal_report`` but reached via
    # the main entry point.
    body_inner = "".join(body_parts)
    return _wrap_document(title, body_inner)


def write_html_report(
    output_path: Union[str, Path], **kwargs: Any
) -> Path:
    """Render an HTML report and write it to ``output_path``.

    Args:
        output_path: destination file path; parent directory must exist
            (callers are responsible for ``mkdir`` if needed).
        **kwargs: forwarded verbatim to ``render_html_report``; at minimum
            ``title`` is required (TypeError otherwise).

    Returns:
        ``Path`` to the written file (matches ``output_path`` cast to Path).
    """
    html_text = render_html_report(**kwargs)
    path = Path(output_path)
    path.write_text(html_text, encoding="utf-8")
    return path


def render_minimal_report(title: str) -> str:
    """Render a minimal valid HTML report with only the title.

    Equivalent to calling ``render_html_report(title)`` with no section
    inputs, but kept as a separate function so callers signalling "I just
    want the shell" don't accidentally pass mismatched kwargs.

    Args:
        title: page title (rendered into both ``<title>`` and ``<h1>``).

    Returns:
        A self-contained HTML5 string with no body sections beyond the
        ``<h1>`` heading.
    """
    return _wrap_document(title, "")
