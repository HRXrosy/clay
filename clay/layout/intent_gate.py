"""RATCHET QUALITY GATE over the layout-intent signoff record.

The layout-intent contract already emits a machine-readable signoff record
(:func:`clay.layout.intent_record.build_signoff_record`, schema
``clay.layout_intent_signoff/v1``). A single record answers "is this layout
signed off, and how good is it?" but it cannot, on its own, catch a REGRESSION:
a routing/placement change that quietly de-certifies a layout or erodes its
relative-quality score. This module turns the record into a monotone ratchet
gate so a layout can guard the flywheel / benchmark and FAIL when it backslides.

What "regression" means here (three independent failure modes):

  * DE-CERTIFICATION -- the proof token was SIGNED in the baseline but is
    WITHHELD now. A previously-certified layout that no longer certifies is the
    most serious regression and always fails the gate, regardless of score.
  * SCORE DROP -- the relative-quality score fell below the baseline score by
    more than a small RELATIVE tolerance. The threshold is a fraction of the
    baseline score (never an absolute delta), so it is scale-invariant: scaling
    every extracted capacitance by a constant leaves the score (and therefore
    the verdict) unchanged.
  * OBLIGATION REGRESSION -- an obligation kind that PASSED in the baseline now
    fails (or goes unmeasurable). A previously-met budget that is no longer met
    is a regression even if the aggregate score happens to stay high.

Ratchet (monotone-up) semantics, mirroring the spirit of
:mod:`clay.layout.regression_baseline` (a structural-identity baseline) but for
a continuous score: a NEW run that is at least as good RAISES the stored bar to
its score; a worse run NEVER lowers it. The bar only goes up, so the gate can
never be silently relaxed by accepting a degraded run as the new baseline.

Seeding: the first run with no baseline PASSES and seeds the baseline (you must
bless a starting point before the ratchet can catch drift), exactly like
``save_baseline`` blesses a starting library before ``compare_to_baseline``.

Self-contained JSON baseline: the baseline is a plain dict persisted as JSON via
:func:`save_baseline` / :func:`load_baseline`. We do NOT reuse
``clay.layout.regression_baseline`` because that idiom keys baselines by
``(dir, template_name)`` and stores a structural library snapshot for an
exact-identity diff; a quality ratchet needs a path-addressed, single-cell,
continuous-score baseline. No external framework (e.g. TED) is required.

All thresholds are RELATIVE fractions (project-wide rule: never absolute aF /
absolute-score deltas).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

BASELINE_SCHEMA = "clay.layout_intent_gate_baseline/v1"

# Default RELATIVE score-drop tolerance: a new score is a regression only if it
# falls below the baseline score by more than this FRACTION of the baseline
# score. Small (the score band is [0, 1] and we want to catch real erosion) but
# > 0 so floating-point jitter / a hair of measurement noise does not trip the
# gate. RELATIVE, never an absolute score delta -> scale-invariant verdict.
_DEFAULT_SCORE_REL_TOL = 0.01


def _passing_kinds(record: Dict[str, object]) -> List[str]:
    """Sorted, de-duplicated obligation kinds whose status is ``pass`` in ``record``.

    A kind counts as passing iff at least one obligation of that kind passed; a
    kind appears in this list only when it has a measured pass (an unmeasurable
    or failing obligation does not bless the kind).
    """
    kinds = {
        str(ob["kind"])
        for ob in record.get("obligations", [])  # type: ignore[union-attr]
        if ob.get("status") == "pass"
    }
    return sorted(kinds)


def _failing_kinds(record: Dict[str, object]) -> List[str]:
    """Sorted obligation kinds with at least one NON-pass obligation in ``record``."""
    kinds = {
        str(ob["kind"])
        for ob in record.get("obligations", [])  # type: ignore[union-attr]
        if ob.get("status") != "pass"
    }
    return sorted(kinds)


def _record_score(record: Dict[str, object]) -> float:
    """The aggregate relative-quality scalar from a signoff record's ``score`` block."""
    return float(record["score"]["score"])  # type: ignore[index]


def _token_signed(record: Dict[str, object]) -> bool:
    """Whether the record carries a SIGNED proof token (vs withheld)."""
    return bool(record["token"]["signed"])  # type: ignore[index]


def seed_baseline(record: Dict[str, object]) -> Dict[str, object]:
    """Build a fresh baseline dict from a signoff ``record`` (the blessed start).

    Captures the three things the gate ratchets against: the best ``score`` seen,
    whether the token was ``signed``, and the obligation kinds that ``passed``.
    """
    return {
        "schema": BASELINE_SCHEMA,
        "cell": str(record.get("cell", "")),
        "pdk": str(record.get("pdk", "")),
        "score": _record_score(record),
        "signed": _token_signed(record),
        "passing_kinds": _passing_kinds(record),
    }


def load_baseline(path: str) -> Dict[str, object]:
    """Load a persisted gate baseline dict from ``path`` (UTF-8 JSON)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_baseline(baseline: Dict[str, object], path: str) -> None:
    """Persist ``baseline`` to ``path`` as pretty-printed UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def evaluate_gate(
    record: Dict[str, object],
    baseline: Optional[Dict[str, object]],
    *,
    score_rel_tol: float = _DEFAULT_SCORE_REL_TOL,
) -> Dict[str, object]:
    """Compare a fresh signoff ``record`` against the stored ``baseline``.

    First run (``baseline is None``) PASSES and seeds the baseline: there is
    nothing to regress against yet, so the run is blessed as the starting bar.

    Otherwise the gate FAILS when ANY of these regressions hold:

      * the token is WITHHELD now but was SIGNED in the baseline
        (a certified layout de-certified);
      * the score dropped below the baseline score by more than ``score_rel_tol``
        as a FRACTION of the baseline score (relative, scale-invariant);
      * an obligation kind that PASSED in the baseline now fails / is unmeasurable.

    Returns a dict::

        {
          "passed": bool,
          "regressions": [ <human-readable regression strings> ],
          "score": <new score>,
          "baseline_score": <stored score> | None,   # None on first run
          "delta": <new - baseline> | None,           # None on first run
          "newly_failing": [ <kinds that passed in baseline, fail now> ],
          "seeded": bool,                              # True iff first run
          "baseline": <baseline to persist>,           # seeded or unchanged
        }
    """
    score = _record_score(record)
    signed = _token_signed(record)

    # ── First run: bless and seed. Nothing to regress against. ──────────────
    if baseline is None:
        seeded = seed_baseline(record)
        return {
            "passed": True,
            "regressions": [],
            "score": score,
            "baseline_score": None,
            "delta": None,
            "newly_failing": [],
            "seeded": True,
            "baseline": seeded,
        }

    baseline_score = float(baseline["score"])  # type: ignore[index]
    baseline_signed = bool(baseline["signed"])  # type: ignore[index]
    baseline_kinds = set(str(k) for k in baseline.get("passing_kinds", []))  # type: ignore[union-attr]

    delta = score - baseline_score
    regressions: List[str] = []

    # 1) De-certification: signed in baseline, withheld now.
    if baseline_signed and not signed:
        regressions.append(
            "de-certified: proof token was SIGNED in the baseline but is "
            "WITHHELD now (a required parasitic budget failed)"
        )

    # 2) Score drop beyond the RELATIVE tolerance (fraction of baseline score).
    #    abs_floor is a fraction of the baseline score, never an absolute delta.
    abs_floor = baseline_score * score_rel_tol
    if delta < -abs_floor:
        regressions.append(
            f"score regressed: {score:.4f} < baseline {baseline_score:.4f} "
            f"(delta {delta:.4f}, allowed drop {abs_floor:.4f} = "
            f"{score_rel_tol:.2%} of baseline)"
        )

    # 3) Obligation regression: a kind that passed in the baseline now fails.
    now_passing = set(_passing_kinds(record))
    newly_failing = sorted(baseline_kinds - now_passing)
    for kind in newly_failing:
        regressions.append(
            f"obligation regressed: kind {kind!r} passed in the baseline "
            f"but no longer passes"
        )

    return {
        "passed": not regressions,
        "regressions": regressions,
        "score": score,
        "baseline_score": baseline_score,
        "delta": delta,
        "newly_failing": newly_failing,
        "seeded": False,
        "baseline": baseline,
    }


def ratchet_baseline(
    baseline: Optional[Dict[str, object]],
    record: Dict[str, object],
) -> Dict[str, object]:
    """Raise the stored bar to ``record`` only when the new run is at least as good.

    Monotone UP: the ratchet never lowers the bar.

      * No baseline yet -> seed from ``record`` (the blessed starting point).
      * New score >= stored score -> adopt the new run wholesale (its score,
        signed flag, and passing kinds become the new bar). A run that is at
        least as good is allowed to TIGHTEN the obligation set / re-bless the
        signed flag, because it dominates the old bar on the score axis.
      * New score < stored score -> keep the old baseline unchanged (the bar
        only goes up). The score field is never lowered.

    Returns the (possibly unchanged) baseline dict. Pure: does not mutate the
    input baseline.
    """
    if baseline is None:
        return seed_baseline(record)

    new_score = _record_score(record)
    stored_score = float(baseline["score"])  # type: ignore[index]

    if new_score >= stored_score:
        return seed_baseline(record)

    # Worse run: never lower the bar. Return an unchanged copy.
    return dict(baseline)


def layout_intent_benchmark_row(
    cell: str,
    record: Dict[str, object],
) -> Dict[str, object]:
    """A flat, JSON-serializable benchmark-matrix row for one cell's signoff.

    Self-contained: derives every field from the signoff ``record`` alone (no
    external framework such as TED is required). Suitable as one row in a
    cross-layout benchmark matrix (e.g. Clay vs another flow), comparable on the
    relative-quality ``score`` axis.

    Fields::

        {
          "cell": <cell name>,
          "pdk": <process kit>,
          "score": <relative-quality scalar in [0, 1]>,
          "passed": <did the layout meet its declared intent?>,
          "token_signed": <was a proof token signed?>,
          "n_pass": <count of passing obligations>,
          "n_fail": <count of failing obligations>,
          "worst_margin": <smallest pass headroom> | None,
        }
    """
    score_block = record["score"]  # type: ignore[index]
    worst = score_block["worst_margin"]  # type: ignore[index]
    return {
        "cell": str(cell),
        "pdk": str(record.get("pdk", "")),
        "score": float(score_block["score"]),  # type: ignore[index]
        "passed": bool(record.get("passed", False)),
        "token_signed": _token_signed(record),
        "n_pass": int(score_block["n_pass"]),  # type: ignore[index]
        "n_fail": int(score_block["n_fail"]),  # type: ignore[index]
        "worst_margin": None if worst is None else float(worst),
    }


__all__ = [
    "BASELINE_SCHEMA",
    "seed_baseline",
    "load_baseline",
    "save_baseline",
    "evaluate_gate",
    "ratchet_baseline",
    "layout_intent_benchmark_row",
]
