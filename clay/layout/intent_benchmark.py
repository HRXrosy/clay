"""Cross-cell BENCHMARK MATRIX over the layout-intent signoff score.

The layout-intent contract scores ONE cell at a time: ``build_signoff_record``
(``clay.layout.intent_record``) folds the certify + score pipeline into a single
JSON record, and ``layout_intent_benchmark_row`` (``clay.layout.intent_gate``)
flattens that record into a comparable per-cell row (cell, pdk, score, passed,
token_signed, n_pass/n_fail, worst_margin). What is still missing is the step
UP from one cell to a COMPARISON: a matrix that lines several cells (or several
flows -- a Clay layout vs a TED one of the same cell) side by side on the same
relative-quality score axis, so the question "which layout is better, and which
cell is the weakest link?" has a single machine-readable answer.

This module is exactly that step. It does NOT re-derive any score: every row is
the existing ``layout_intent_benchmark_row(cell, build_signoff_record(cell,
report))``, so the matrix inherits the contract's honesty boundaries verbatim (a
withheld token stays withheld; an unmeasurable budget still blocks; the score is
the same normalized scalar in ``[0, 1]``). The matrix simply COLLECTS those rows
and adds an aggregate summary + a leaderboard ordering.

Self-contained, framework-neutral: a "row" only needs a cell name and either a
PexReport (we build the record) or a pre-built signoff record. Nothing here
requires Magic, TED, or any external tool -- a synthetic ``PexReport`` is enough
for the matrix, which keeps the benchmark unit-testable without a simulator.

All scores are the contract's RELATIVE-quality scalar (project-wide rule: never
absolute aF deltas), so the comparison is process- and scale-neutral.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union

from .intent_gate import layout_intent_benchmark_row
from .intent_record import SCHEMA as _RECORD_SCHEMA
from .intent_record import build_signoff_record
from .pex.base import PexReport

SCHEMA = "clay.layout_intent_benchmark/v1"

# An entry is one of:
#   * (cell_name, PexReport)          -> we build the signoff record;
#   * (cell_name, signoff_record_dict) -> use the pre-built record as-is.
Entry = Tuple[str, Union[PexReport, Dict[str, object]]]


def _row_for_entry(entry: Entry, **tol) -> Dict[str, object]:
    """Build one benchmark row from an ``(cell, PexReport | record)`` entry.

    A ``PexReport`` second element is run through the SAME certify+score
    pipeline (:func:`build_signoff_record`) the single-cell path uses; a dict
    second element is taken as an already-built signoff record and only flattened
    (no re-derivation). Either way the flattening goes through the shared
    :func:`layout_intent_benchmark_row`, so a matrix row is byte-identical to the
    single-cell row for the same cell + report.
    """
    cell, payload = entry
    if isinstance(payload, PexReport):
        record = build_signoff_record(cell, payload, **tol)
    elif isinstance(payload, dict):
        record = payload
    else:
        raise TypeError(
            f"benchmark entry for cell {cell!r} must pair it with a PexReport or "
            f"a pre-built signoff-record dict, got {type(payload).__name__}"
        )
    return layout_intent_benchmark_row(cell, record)


def build_layout_benchmark(
    entries: Iterable[Entry],
    **tol,
) -> Dict[str, object]:
    """Assemble a cross-cell benchmark matrix from per-cell ``entries``.

    ``entries`` is an iterable of ``(cell_name, PexReport | signoff_record)``
    pairs (a ``PexReport`` is scored here; a dict is taken as a pre-built record).
    Each becomes one row via :func:`layout_intent_benchmark_row`, in input order.
    Any ``**tol`` keyword tolerances are forwarded to :func:`build_signoff_record`
    for the PexReport entries (RELATIVE fractions only).

    Returns a fully JSON-serializable dict::

        {
          "schema": "clay.layout_intent_benchmark/v1",
          "rows": [ <benchmark_row>, ... ],   # one per entry, input order
          "summary": {
            "n_cells":     <count of rows>,
            "n_passed":    <count of rows whose layout met its declared intent>,
            "mean_score":  <arithmetic mean of the row scores, 0.0 if no rows>,
            "weakest_cell": <cell name with the lowest score, or None if no rows>,
          },
        }

    ``weakest_cell`` is the lowest-scoring cell (ties broken by input order -- the
    first row at the minimum score wins), the natural "weakest link" to repair
    next. With zero rows the summary is the empty matrix (``n_cells=0``,
    ``mean_score=0.0``, ``weakest_cell=None``) rather than an error.
    """
    rows: List[Dict[str, object]] = [_row_for_entry(e, **tol) for e in entries]

    n_cells = len(rows)
    n_passed = sum(1 for r in rows if bool(r["passed"]))
    mean_score = (
        sum(float(r["score"]) for r in rows) / n_cells if n_cells else 0.0
    )
    # Lowest-scoring cell; ties resolve to the FIRST row at the minimum so the
    # weakest link is deterministic. min() over (score, index) keeps input order.
    weakest_cell = None
    if rows:
        weakest = min(
            range(n_cells), key=lambda i: (float(rows[i]["score"]), i)
        )
        weakest_cell = str(rows[weakest]["cell"])

    return {
        "schema": SCHEMA,
        "rows": rows,
        "summary": {
            "n_cells": n_cells,
            "n_passed": n_passed,
            "mean_score": mean_score,
            "weakest_cell": weakest_cell,
        },
    }


def rank_cells(matrix: Dict[str, object]) -> List[Dict[str, object]]:
    """Leaderboard: the matrix rows sorted by ``score`` DESCENDING (best first).

    A stable sort keyed by the negated score, so cells at the same score keep
    their input (matrix) order. Returns the row dicts unchanged (the caller can
    read every benchmark field, not just the score); the input matrix is not
    mutated.
    """
    rows: List[Dict[str, object]] = list(matrix.get("rows", []))  # type: ignore[arg-type]
    return sorted(rows, key=lambda r: -float(r["score"]))


def format_benchmark(matrix: Dict[str, object]) -> str:
    """Render an aligned text table of a benchmark ``matrix`` + a summary line.

    Columns: ``cell | score | pass | token | n_pass/n_fail | worst_margin``. The
    rows are printed in the matrix's stored order (use :func:`rank_cells` first if
    a leaderboard ordering is wanted). ``worst_margin`` prints ``n/a`` when no
    obligation was measured. The trailing summary line reports the cell count,
    pass count, mean score, and the weakest cell.
    """
    rows: List[Dict[str, object]] = list(matrix.get("rows", []))  # type: ignore[arg-type]

    headers = ("cell", "score", "pass", "token", "n_pass/n_fail", "worst_margin")
    table_rows: List[Tuple[str, ...]] = []
    for r in rows:
        worst = r.get("worst_margin")
        worst_str = "n/a" if worst is None else f"{float(worst):.4f}"
        table_rows.append(
            (
                str(r["cell"]),
                f"{float(r['score']):.4f}",
                "PASS" if bool(r["passed"]) else "FAIL",
                "signed" if bool(r["token_signed"]) else "withheld",
                f"{int(r['n_pass'])}/{int(r['n_fail'])}",
                worst_str,
            )
        )

    widths = [len(h) for h in headers]
    for tr in table_rows:
        for i, cellv in enumerate(tr):
            widths[i] = max(widths[i], len(cellv))

    def _line(cols: Iterable[object]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols))

    sep = "  ".join("-" * w for w in widths)

    out: List[str] = []
    out.append("Layout-intent benchmark matrix")
    out.append("")
    out.append(_line(headers))
    out.append(sep)
    if table_rows:
        for tr in table_rows:
            out.append(_line(tr))
    else:
        out.append("(no cells in benchmark)")
    out.append("")

    summary = matrix.get("summary", {})  # type: ignore[assignment]
    n_cells = int(summary.get("n_cells", 0))  # type: ignore[union-attr]
    n_passed = int(summary.get("n_passed", 0))  # type: ignore[union-attr]
    mean_score = float(summary.get("mean_score", 0.0))  # type: ignore[union-attr]
    weakest = summary.get("weakest_cell")  # type: ignore[union-attr]
    weakest_str = "none" if weakest is None else str(weakest)
    out.append(
        f"Summary: {n_passed}/{n_cells} cells PASS  mean_score={mean_score:.4f}  "
        f"weakest_cell={weakest_str}"
    )

    return "\n".join(out)


__all__ = [
    "SCHEMA",
    "build_layout_benchmark",
    "rank_cells",
    "format_benchmark",
]
