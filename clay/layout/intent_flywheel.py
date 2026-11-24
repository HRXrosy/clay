"""MULTI-CELL FLYWHEEL RATCHET over the layout-intent quality gate.

The single-cell ratchet (:mod:`clay.layout.intent_gate`) guards ONE cell: it
turns a signoff record into a monotone bar and FAILS when that cell de-certifies
or its relative-quality score erodes. The cross-cell benchmark
(:mod:`clay.layout.intent_benchmark`) lines several cells up on one score axis
for comparison. What is still missing is the CI-facing step that fuses the two:
a SINGLE gate over the WHOLE library that fails if ANY cell regresses, seeds new
cells without tripping, and ratchets every cell's bar monotone-up so layout
quality cannot silently backslide across the project.

This module is that step. It is a thin, framework-neutral fan-out over the
single-cell gate -- every per-cell verdict is the existing
:func:`clay.layout.intent_gate.evaluate_gate` against that cell's stored bar, so
the flywheel inherits the gate's honesty boundaries verbatim (a de-certified
token still fails; a withheld budget still blocks; the score tolerance stays a
RELATIVE fraction, never an absolute aF / score delta -> scale-invariant). The
flywheel adds three things on top: a multi-cell baseline (a bar PER cell), an
overall verdict that fails when ANY cell regresses, and a per-cell ratchet that
raises bars and admits new cells.

Multi-cell baseline schema (``clay.layout_intent_flywheel/v1``)::

    {
      "schema": "clay.layout_intent_flywheel/v1",
      "cells": {
        "<cell>": {                      # one single-cell gate baseline each
          "schema": "clay.layout_intent_gate_baseline/v1",
          "cell": "<cell>", "pdk": "<pdk>",
          "score": <best score seen>,
          "signed": <was the token signed?>,
          "passing_kinds": [ <obligation kinds that passed> ],
        },
        ...
      },
    }

CI flow (a CI step wires these together):

  1. ``baseline = load_flywheel(path)`` -- the blessed multi-cell bar (or build a
     fresh one with :func:`build_flywheel_baseline` on the very first run).
  2. build the CURRENT signoff records, one per cell, into
     ``records_by_cell = {cell -> signoff_record}`` (each via
     :func:`clay.layout.intent_record.build_signoff_record`).
  3. ``result = check_flywheel(records_by_cell, baseline)`` -- run the per-cell
     gate. A cell NOT in the baseline is a NEW cell: it seeds + passes (you must
     bless a starting point before the ratchet can catch its drift).
  4. ``print(format_flywheel(result))`` for a CI-readable summary, then EXIT
     NONZERO when ``result["passed"]`` is False (some cell regressed).
  5. On PASS only, ``baseline = ratchet_flywheel(baseline, records_by_cell)``
     (monotone-up per cell + admit new cells) and ``save_flywheel(baseline,
     path)`` so the bar tightens for the next run.

Seeding semantics mirror the single-cell gate: the FIRST time a cell is seen it
PASSES and seeds, so a brand-new cell never trips the gate merely by existing.
The bar only ever goes UP, per cell, so accepting a degraded run as the new
baseline is impossible.

Self-contained JSON: the baseline is a plain dict persisted via
:func:`save_flywheel` / :func:`load_flywheel`. We deliberately do NOT bolt onto
the repo's structural fast-gate anchor (the ``10500``-style test-count ratchet),
which counts passing tests; the layout flywheel ratchets a CONTINUOUS
relative-quality score per cell, a different axis with no clean hook into the
test-count anchor. No external framework (e.g. TED) is required.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from .intent_gate import (
    BASELINE_SCHEMA as _GATE_BASELINE_SCHEMA,
    evaluate_gate,
    ratchet_baseline,
    seed_baseline,
)

FLYWHEEL_SCHEMA = "clay.layout_intent_flywheel/v1"

# Re-export the per-cell baseline schema so callers can assert on it without
# reaching back into intent_gate.
CELL_BASELINE_SCHEMA = _GATE_BASELINE_SCHEMA


def _cells_map(baseline: Optional[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """The ``cells`` sub-map of a flywheel ``baseline`` (empty when no baseline)."""
    if baseline is None:
        return {}
    return dict(baseline.get("cells", {}))  # type: ignore[arg-type]


def build_flywheel_baseline(
    records_by_cell: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """Build a fresh multi-cell baseline from per-cell signoff records.

    ``records_by_cell`` maps ``cell -> signoff_record`` (each the output of
    :func:`clay.layout.intent_record.build_signoff_record`). Every cell is
    seeded with its single-cell gate baseline (best score / signed flag /
    passing kinds) via :func:`clay.layout.intent_gate.seed_baseline`. This is the
    blessed starting bar for the whole library; persist it with
    :func:`save_flywheel`.

    Returns a JSON-serializable dict matching ``clay.layout_intent_flywheel/v1``.
    """
    cells: Dict[str, Dict[str, object]] = {
        str(cell): seed_baseline(record)
        for cell, record in records_by_cell.items()
    }
    return {"schema": FLYWHEEL_SCHEMA, "cells": cells}


def load_flywheel(path: str) -> Dict[str, object]:
    """Load a persisted multi-cell flywheel baseline from ``path`` (UTF-8 JSON)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_flywheel(baseline: Dict[str, object], path: str) -> None:
    """Persist a flywheel ``baseline`` to ``path`` as pretty-printed UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def check_flywheel(
    current_records_by_cell: Dict[str, Dict[str, object]],
    baseline: Optional[Dict[str, object]],
    *,
    score_rel_tol: Optional[float] = None,
) -> Dict[str, object]:
    """Run the per-cell ratchet gate over the whole library against ``baseline``.

    For each cell in ``current_records_by_cell`` the stored per-cell bar is
    looked up in ``baseline["cells"]`` and the single-cell
    :func:`clay.layout.intent_gate.evaluate_gate` is run against it:

      * a cell PRESENT in the baseline is gated against its stored bar -- it
        FAILS if it de-certified, dropped its score below the bar by more than
        the RELATIVE tolerance, or lost a previously-passing obligation kind;
      * a cell ABSENT from the baseline is a NEW cell: ``evaluate_gate(record,
        None)`` seeds it and PASSES (a brand-new cell never trips the gate just
        by existing -- you bless its starting bar first).

    The OVERALL verdict FAILS iff ANY cell regressed (a new cell never causes an
    overall failure). ``score_rel_tol`` (a RELATIVE fraction) is forwarded to the
    per-cell gate when given; otherwise the gate's default tolerance applies.

    Returns a JSON-serializable dict::

        {
          "passed": bool,                       # False iff any cell regressed
          "per_cell": { "<cell>": <gate result>, ... },  # one evaluate_gate dict
          "regressed_cells": [ <cells that failed the gate>, ... ],  # sorted
          "new_cells": [ <cells not in the baseline>, ... ],         # sorted
        }
    """
    stored = _cells_map(baseline)
    gate_kw = {} if score_rel_tol is None else {"score_rel_tol": score_rel_tol}

    per_cell: Dict[str, Dict[str, object]] = {}
    regressed_cells: List[str] = []
    new_cells: List[str] = []

    for cell, record in current_records_by_cell.items():
        cell = str(cell)
        cell_baseline = stored.get(cell)
        if cell_baseline is None:
            new_cells.append(cell)
        result = evaluate_gate(record, cell_baseline, **gate_kw)
        per_cell[cell] = result
        if not bool(result["passed"]):
            regressed_cells.append(cell)

    return {
        "passed": not regressed_cells,
        "per_cell": per_cell,
        "regressed_cells": sorted(regressed_cells),
        "new_cells": sorted(new_cells),
    }


def ratchet_flywheel(
    baseline: Optional[Dict[str, object]],
    current_records_by_cell: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """Raise every cell's bar monotone-up and admit new cells.

    For each cell in ``current_records_by_cell`` the stored per-cell bar (or
    ``None`` for a new cell) is passed through the single-cell
    :func:`clay.layout.intent_gate.ratchet_baseline`, which seeds a new cell and,
    for an existing cell, adopts the run only when its score is at least as good
    (NEVER lowering the stored bar).

    Cells present in ``baseline`` but ABSENT from this run are carried forward
    unchanged -- a library subset run never erases the bars of cells it did not
    touch.

    Pure: does not mutate the input ``baseline``. Returns a fresh, JSON-
    serializable flywheel baseline.
    """
    stored = _cells_map(baseline)
    # Start from the existing bars (carry forward untouched cells), then ratchet
    # / admit each cell from the current run.
    cells: Dict[str, Dict[str, object]] = dict(stored)
    for cell, record in current_records_by_cell.items():
        cell = str(cell)
        cells[cell] = ratchet_baseline(stored.get(cell), record)

    return {"schema": FLYWHEEL_SCHEMA, "cells": cells}


def format_flywheel(check_result: Dict[str, object]) -> str:
    """Render a CI-readable summary of a :func:`check_flywheel` result.

    One line per cell -- ``PASS`` / ``FAIL``, a ``(new)`` tag for a freshly-seeded
    cell, the new score (and the baseline score + delta for an existing cell),
    and the regression reasons on a FAILing cell. The trailing verdict line is
    the overall ``PASS`` / ``FAIL`` plus the regressed-cell list, so a CI log
    shows at a glance both the verdict and the weakest link.
    """
    per_cell: Dict[str, Dict[str, object]] = dict(
        check_result.get("per_cell", {})  # type: ignore[arg-type]
    )
    regressed = list(check_result.get("regressed_cells", []))  # type: ignore[arg-type]
    new_cells = set(check_result.get("new_cells", []))  # type: ignore[arg-type]
    overall_passed = bool(check_result.get("passed", False))

    out: List[str] = []
    out.append("Layout-intent FLYWHEEL ratchet (multi-cell)")
    out.append("")

    for cell in sorted(per_cell):
        result = per_cell[cell]
        passed = bool(result["passed"])
        verdict = "PASS" if passed else "FAIL"
        tags = " (new)" if cell in new_cells else ""

        score = float(result["score"])
        baseline_score = result.get("baseline_score")
        if baseline_score is None:
            score_str = f"score={score:.4f} (seeded)"
        else:
            delta = float(result.get("delta", 0.0))  # type: ignore[arg-type]
            score_str = (
                f"score={score:.4f} baseline={float(baseline_score):.4f} "
                f"delta={delta:+.4f}"
            )

        out.append(f"  [{verdict}] {cell}{tags}  {score_str}")
        if not passed:
            for reason in result.get("regressions", []):  # type: ignore[union-attr]
                out.append(f"           - {reason}")

    out.append("")
    overall = "PASS" if overall_passed else "FAIL"
    if regressed:
        out.append(
            f"Verdict: {overall}  ({len(regressed)} regressed: "
            f"{', '.join(regressed)})"
        )
    else:
        out.append(
            f"Verdict: {overall}  ({len(per_cell)} cells, "
            f"{len(new_cells)} new, 0 regressed)"
        )

    return "\n".join(out)


__all__ = [
    "FLYWHEEL_SCHEMA",
    "CELL_BASELINE_SCHEMA",
    "build_flywheel_baseline",
    "load_flywheel",
    "save_flywheel",
    "check_flywheel",
    "ratchet_flywheel",
    "format_flywheel",
]
