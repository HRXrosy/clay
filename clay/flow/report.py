"""Management dashboard for Clay's flow + maturity state.

Aggregates, per registered circuit (``clay.flow.spec.discover_registry``):

* the EVIDENCE-computed maturity vs the design's DECLARED maturity
  (``clay.casepkg.assess_maturity_detail``), with a drift flag;
* the KPI acceptance table — each ``KpiSpec``'s human target vs the measured
  value read from the circuit's ``post_results.json`` (the explicit five-state
  contract from ``KpiSpec.evaluate``);
* the ``@require`` hard gates and the pre/post ``@consistency`` rows;
* the last per-step verdict from a ``_flow_run/work/flow_state.json`` if one
  has been recorded.

Plus a repo-level summary (circuit count, maturity histogram, audit
overclaimed / invalid-declared counts).

This module only READS the frozen ``spec.py`` / ``casepkg.py`` surfaces and the
real on-disk results; it never fabricates a KPI value. Where a required value
is absent it shows ``unmeasurable``.

Run as ``python -m clay.flow.report [--out docs/clay_flow_status.html]``.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Optional

from clay.flow.spec import CircuitSpec, discover_registry
from clay.casepkg import (
    MATURITIES,
    MATURITY_RANK,
    assess_maturity_detail,
    audit_maturity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = "docs/clay_flow_status.html"


# --------------------------------------------------------------------------- #
# Data aggregation                                                            #
# --------------------------------------------------------------------------- #
def _load_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a broken results file = no data, not a crash
        return {}
    return doc if isinstance(doc, dict) else {}


def _display_value(raw: Any, scale: float) -> Optional[float]:
    """Apply the KPI display scale to a raw numeric value (or None)."""
    if raw is None:
        return None
    try:
        return float(raw) * float(scale)
    except (TypeError, ValueError):
        return None


def _kpi_rows(spec: CircuitSpec, post: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kpi in spec.kpis:
        raw = post.get(kpi.key)
        status = kpi.evaluate(raw)
        rows.append(
            {
                "key": kpi.key,
                "label": kpi.label or kpi.key,
                "unit": kpi.unit,
                "target": kpi.target,
                "measured": _display_value(raw, kpi.scale),
                "raw": raw,
                "status": status,
            }
        )
    return rows


def _consistency_rows(spec: CircuitSpec, post: dict[str, Any]) -> list[dict[str, Any]]:
    pre_block = post.get("pre") if isinstance(post.get("pre"), dict) else {}
    rows: list[dict[str, Any]] = []
    for cons in spec.consistency:
        pre = pre_block.get(cons.pre_key)
        post_val = post.get(cons.post_key)
        status = cons.evaluate(pre, post_val)
        rel: Optional[float] = None
        try:
            p = float(pre)
            q = float(post_val)
            if p != 0:
                rel = abs(q - p) / abs(p)
        except (TypeError, ValueError):
            rel = None
        rows.append(
            {
                "label": cons.label,
                "pre_key": cons.pre_key,
                "post_key": cons.post_key,
                "pre": pre,
                "post": post_val,
                "rel": rel,
                "rel_tol": cons.rel_tol,
                "status": status,
            }
        )
    return rows


def _flow_state(spec: CircuitSpec) -> dict[str, str]:
    """Last recorded per-step verdict, if a flow_state.json exists."""
    candidate = spec.case_dir / "_flow_run" / "work" / "flow_state.json"
    doc = _load_json(candidate)
    steps = doc.get("steps") if isinstance(doc.get("steps"), dict) else {}
    verdicts: dict[str, str] = {}
    for name, entry in steps.items():
        if isinstance(entry, dict):
            verdicts[str(name)] = str(entry.get("status") or "?")
    return verdicts


def build_status_report(repo_root: str | Path = ".") -> dict[str, Any]:
    """Aggregate the full flow + maturity dashboard payload.

    Returns a dict with ``circuits`` (keyed by ``CircuitSpec.key``) and a
    repo-level ``summary``. Each circuit entry carries its computed/declared
    maturity (+ drift), the KPI table, gate list, consistency rows, and the
    last per-step flow verdict.
    """
    root = Path(repo_root).resolve() if str(repo_root) not in (".", "") else REPO_ROOT
    registry = discover_registry(root / "cases")

    audit = audit_maturity(root)

    circuits: dict[str, Any] = {}
    maturity_hist = {level: 0 for level in MATURITIES}

    for key in sorted(registry):
        spec = registry[key]
        post = _load_json(spec.post_results)
        try:
            maturity = assess_maturity_detail(spec.case_dir, repo_root=root)
        except Exception:  # noqa: BLE001 - keep the dashboard rendering
            maturity = None

        computed = maturity.computed if maturity else None
        if computed in maturity_hist:
            maturity_hist[computed] += 1

        kpi_rows = _kpi_rows(spec, post)
        cons_rows = _consistency_rows(spec, post)
        flow_steps = _flow_state(spec)

        state_counts = {
            state: sum(1 for row in kpi_rows if row["status"] == state)
            for state in ("pass", "fail", "unmeasurable", "unsupported", "advisory")
        }

        circuits[key] = {
            "key": key,
            "display_name": spec.display_name,
            "design_id": spec.design_id,
            "top_cell": spec.top_cell,
            "pdk": spec.pdk,
            "vdd": spec.vdd,
            "cell_type": spec.cell_type,
            "case_dir": str(spec.case_dir),
            "post_results": str(spec.post_results),
            "post_results_present": Path(spec.post_results).is_file(),
            "steps": list(spec.steps),
            "maturity": {
                "computed": computed,
                "declared": maturity.declared if maturity else None,
                "drift": bool(maturity.drift) if maturity else False,
                "overclaimed": bool(maturity.overclaimed) if maturity else False,
                "declared_valid": bool(maturity.declared_valid) if maturity else False,
            },
            "kpis": kpi_rows,
            "kpi_summary": {
                **state_counts,
                "total": len(kpi_rows),
            },
            "consistency": cons_rows,
            "gates": [g.kind for g in spec.gates],
            "flow_state": flow_steps,
        }

    summary = {
        "n_circuits": len(circuits),
        "maturity_histogram": maturity_hist,
        "audit_total": audit.summary.get("total", 0),
        "audit_overclaimed": audit.summary.get("overclaimed", 0),
        "audit_invalid_declared": audit.summary.get("invalid_declared", 0),
        "audit_drift": audit.summary.get("drift", 0),
        "audit_overclaimed_ids": list(audit.summary.get("overclaimed_ids", [])),
        "audit_invalid_ids": list(audit.summary.get("invalid_ids", [])),
    }

    return {"circuits": circuits, "summary": summary}


# --------------------------------------------------------------------------- #
# HTML rendering                                                              #
# --------------------------------------------------------------------------- #
_STYLE = """
:root { --pass:#1f8b4c; --warn:#c47f00; --fail:#c0392b; --pending:#7a7a7a; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "Segoe UI", "Source Han Sans SC", system-ui, sans-serif;
  margin: 0; padding: 2rem; color: #1f2329; background: #f6f7f9; line-height: 1.5;
}
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.2rem; margin: 0; }
.sub { color: #6b7280; margin: 0 0 1.5rem; font-size: .9rem; }
section.circuit {
  background: #fff; border: 1px solid #e3e6ea; border-radius: 8px;
  padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
}
.circuit-head { display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap; }
.circuit-head .meta { color: #6b7280; font-size: .85rem; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0 0; font-size: .9rem; }
th, td { border: 1px solid #dfe3e8; padding: .4rem .6rem; text-align: left; }
th { background: #eef1f4; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.status { font-weight: 600; padding: .1rem .5rem; border-radius: 4px; color: #fff; display: inline-block; }
.status.pass { background: var(--pass); }
.status.warn { background: var(--warn); }
.status.fail { background: var(--fail); }
.status.unmeasurable { background: var(--fail); }
.status.unsupported { background: var(--fail); }
.status.advisory { background: var(--warn); }
.status.skipped { background: var(--pending); }
.status.pending { background: var(--pending); }
.badge { padding: .15rem .55rem; border-radius: 4px; font-size: .8rem; border: 1px solid #cfd4da; }
.badge.drift { background: #fff4e0; border-color: #e6c074; color: #8a5a00; }
.badge.ok { background: #e7f5ec; border-color: #9ed4b3; color: #1f8b4c; }
.summary-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.card {
  background: #fff; border: 1px solid #e3e6ea; border-radius: 8px;
  padding: 1rem 1.25rem; min-width: 160px;
}
.card .n { font-size: 1.6rem; font-weight: 700; }
.card .lbl { color: #6b7280; font-size: .85rem; }
.card.alert .n { color: var(--fail); }
.hist td.num { font-variant-numeric: tabular-nums; }
.gates span { margin-right: .5rem; }
.flow-steps span { margin-right: .4rem; font-size: .82rem; }
footer { color: #9aa1a9; font-size: .8rem; margin-top: 2rem; }
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fmt_num(value: Optional[float], unit: str = "") -> str:
    if value is None:
        return "unmeasurable"
    if abs(value) >= 1000 or (value != 0 and abs(value) < 0.01):
        text = f"{value:.4g}"
    else:
        text = f"{value:.4g}"
    return f"{text} {unit}".strip() if unit else text


def _status_cell(status: str) -> str:
    cls = status if status in (
        "pass", "fail", "unmeasurable", "unsupported", "advisory",
        "warn", "skipped", "pending",
    ) else "unsupported"
    return f'<span class="status {cls}">{_esc(status)}</span>'


def _summary_html(summary: dict[str, Any]) -> str:
    hist = summary.get("maturity_histogram", {})
    hist_rows = "".join(
        f"<tr><td>{_esc(level)}</td><td class='num'>{hist.get(level, 0)}</td></tr>"
        for level in MATURITIES
    )
    overclaimed = summary.get("audit_overclaimed", 0)
    invalid = summary.get("audit_invalid_declared", 0)
    cards = [
        ("n_circuits", "registered circuits", summary.get("n_circuits", 0), False),
        ("audit_total", "case packages audited", summary.get("audit_total", 0), False),
        ("audit_drift", "maturity drift", summary.get("audit_drift", 0), False),
        ("audit_overclaimed", "overclaimed", overclaimed, overclaimed > 0),
        ("audit_invalid", "invalid declared", invalid, invalid > 0),
    ]
    cards_html = "".join(
        f'<div class="card{" alert" if alert else ""}">'
        f'<div class="n">{_esc(n)}</div><div class="lbl">{_esc(lbl)}</div></div>'
        for _id, lbl, n, alert in cards
    )
    return (
        "<div class='summary-grid'>" + cards_html + "</div>"
        "<table class='hist'><thead><tr><th>Maturity level</th>"
        "<th>circuits</th></tr></thead><tbody>" + hist_rows + "</tbody></table>"
    )


def _circuit_html(circuit: dict[str, Any]) -> str:
    mat = circuit["maturity"]
    drift = mat.get("drift")
    drift_badge = (
        f'<span class="badge drift">drift: declared {_esc(mat.get("declared"))} '
        f'&rarr; computed {_esc(mat.get("computed"))}</span>'
        if drift
        else f'<span class="badge ok">maturity {_esc(mat.get("computed"))}</span>'
    )
    overclaim_badge = (
        '<span class="badge drift">OVERCLAIMED</span>' if mat.get("overclaimed") else ""
    )

    # KPI table
    if circuit["kpis"]:
        kpi_body = "".join(
            "<tr>"
            f"<td>{_esc(r['label'])}</td>"
            f"<td>{_esc(r['target'])} {_esc(r['unit'])}</td>"
            f"<td class='num'>{_esc(_fmt_num(r['measured'], r['unit']))}</td>"
            f"<td>{_status_cell(r['status'])}</td>"
            "</tr>"
            for r in circuit["kpis"]
        )
        kpi_table = (
            "<table><thead><tr><th>KPI</th><th>Target</th><th>Measured (post)</th>"
            "<th>Status</th></tr></thead><tbody>" + kpi_body + "</tbody></table>"
        )
    else:
        kpi_table = "<p class='sub'>no KPI rows declared</p>"

    # Consistency table
    if circuit["consistency"]:
        cons_body = "".join(
            "<tr>"
            f"<td>{_esc(r['label'])}</td>"
            f"<td class='num'>{_esc(_fmt_num(r['pre']) if r['pre'] is not None else 'pending')}</td>"
            f"<td class='num'>{_esc(_fmt_num(r['post']) if r['post'] is not None else 'pending')}</td>"
            f"<td class='num'>{_esc(f'{r['rel'] * 100:.1f}%' if r['rel'] is not None else 'pending')}"
            f" / {_esc(f'{r['rel_tol'] * 100:.0f}%')}</td>"
            f"<td>{_status_cell(r['status'])}</td>"
            "</tr>"
            for r in circuit["consistency"]
        )
        cons_table = (
            "<table><thead><tr><th>Consistency (pre/post)</th><th>Pre</th><th>Post</th>"
            "<th>Rel / tol</th><th>Status</th></tr></thead><tbody>"
            + cons_body
            + "</tbody></table>"
        )
    else:
        cons_table = ""

    # Gates
    gates = circuit.get("gates", [])
    gates_html = (
        "<p class='gates'><strong>Required gates (@require):</strong> "
        + "".join(f"<span class='badge'>{_esc(g)}</span> " for g in gates)
        + "</p>"
        if gates
        else ""
    )

    # Flow state
    flow = circuit.get("flow_state", {})
    if flow:
        ordered = [s for s in circuit.get("steps", []) if s in flow]
        ordered += [s for s in flow if s not in ordered]
        flow_html = (
            "<p class='flow-steps'><strong>Last flow verdict:</strong> "
            + "".join(
                f"{_esc(name)} {_status_cell(flow[name])}" for name in ordered
            )
            + "</p>"
        )
    else:
        flow_html = "<p class='sub'>no recorded flow run (flow_state.json absent)</p>"

    ks = circuit["kpi_summary"]
    return (
        "<section class='circuit'>"
        "<div class='circuit-head'>"
        f"<h2>{_esc(circuit['display_name'])}</h2>"
        f"<span class='meta'>key=<code>{_esc(circuit['key'])}</code> | "
        f"top_cell={_esc(circuit['top_cell'])} | pdk={_esc(circuit['pdk'])} | "
        f"Vdd={_esc(circuit['vdd'])} V</span>"
        f"{drift_badge}{overclaim_badge}"
        "</div>"
        f"<p class='sub'>KPI: {ks['pass']} pass / {ks['fail']} fail / "
        f"{ks['unmeasurable']} unmeasurable / {ks['unsupported']} unsupported / "
        f"{ks['advisory']} advisory of {ks['total']} | "
        f"post_results: {'present' if circuit['post_results_present'] else 'MISSING'}</p>"
        + kpi_table
        + cons_table
        + gates_html
        + flow_html
        + "</section>"
    )


def render_html(report: dict[str, Any]) -> str:
    """Render the aggregated report into a self-contained HTML dashboard."""
    summary = report.get("summary", {})
    circuits = report.get("circuits", {})

    circuit_sections = "".join(
        _circuit_html(circuits[key]) for key in sorted(circuits)
    )

    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        "<title>Clay Flow + Maturity Status</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n<body>\n"
        "<h1>Clay Flow + Maturity Status</h1>\n"
        "<p class='sub'>Evidence-computed maturity, KPI acceptance vs measured "
        "post-layout results, required gates, pre/post consistency, and the last "
        "recorded flow verdict per registered circuit.</p>\n"
        "<h2>Repo summary</h2>\n"
        + _summary_html(summary)
        + "\n<h2 style='margin-top:2rem'>Circuits</h2>\n"
        + circuit_sections
        + "<footer>Generated by clay.flow.report from real post_results.json, "
        "flow_state.json, and casepkg maturity evidence. Absent required values shown as "
        "<em>unmeasurable</em>.</footer>\n"
        "</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# CLI entry                                                                   #
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m clay.flow.report",
        description="Build the Clay flow + maturity management dashboard.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"output HTML path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="repo root to scan (default: current dir)",
    )
    args = parser.parse_args(argv)

    report = build_status_report(args.repo_root)
    html_text = render_html(report)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        root = Path(args.repo_root).resolve() if args.repo_root not in (".", "") else REPO_ROOT
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")

    s = report["summary"]
    print(
        f"wrote {out_path} | {s['n_circuits']} circuits | "
        f"audit drift={s['audit_drift']} overclaimed={s['audit_overclaimed']} "
        f"invalid={s['audit_invalid_declared']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
