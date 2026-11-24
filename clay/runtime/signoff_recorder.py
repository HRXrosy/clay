"""Wire Clay's signoff stages (DRC + LVS + layout-intent consistency) into a RunRecorder.

This module is the thin bridge between three independently shipped signoff
backends and :class:`clay.runtime.recorder.RunRecorder`, so the schematic_viewer
can replay or live-monitor signoff progress on the same file-based event
contract the runtime already uses for sizing, schematic, and PEX events.

The three stages and the events they emit:

  * **DRC** -- :func:`run_drc_with_recorder` wraps
    :func:`tools.klayout_drc_signoff_v139.run.run_gf180_drc` (REAL gf180mcu
    KLayout deck, not a mock):

      - ``drc_started``    before the run; carries ``gds_path``,
                           ``deck_available``, ``klayout_available``.
      - ``drc_completed``  after a real run; carries ``is_clean``,
                           ``n_violations``, ``ran_real_deck`` and a
                           ``by_rule`` count map.
      - ``drc_skipped``    when the KLayout binary or the gf180mcu deck is
                           absent; carries a ``reason`` and returns ``None``
                           (the run never invents a verdict).

  * **LVS** -- :func:`record_lvs` takes a STRUCTURED report dict that the
    caller already produced (the LVS netlist comparator runner may live in a
    sibling tool / outside this repo). The helper deliberately accepts the
    dict directly so it can be wired in WITHOUT a hard dependency on a
    specific LVS engine. Events:

      - ``lvs_completed``  when ``clean`` is truthy.
      - ``lvs_failed``     when ``clean`` is falsy; the worst mismatch is
                           surfaced on the event for triage.

  * **Layout-intent consistency** -- :func:`run_constraints_with_recorder`
    runs the three EXISTING checkers that certify a routed cell against its
    declared intent and the global structural rules:

      - :func:`clay.layout.global_constraints.check_global_constraints` --
        the STRUCTURAL axis (containment within rails, no signal crossing a
        rail, info metrics);
      - :func:`clay.layout.intent_obligations.check_layout_intent` -- the
        PARASITIC axis (matched-cap balance / kickback / clk_regen budgets,
        measured against a real :class:`PexReport`);
      - :func:`clay.layout.placement_symmetry.placement_symmetry_obligations`
        + :func:`mirror_placement_obligations` -- the PLACEMENT axis
        (matched device centroids mirror about the cell axis).

    One ``consistency_completed`` event summarises all three axes; if any
    axis errors, an additional ``consistency_violation`` event is emitted
    per failing axis carrying the worst offender of that axis.

Every helper is a pure pass-through when ``recorder is None`` (no side
effects, no events written) so existing call sites keep working unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from clay.runtime.recorder import RunRecorder


__all__ = [
    "run_drc_with_recorder",
    "record_lvs",
    "run_constraints_with_recorder",
]


# ─────────────────────────────────────────────────────────────────────────────
# DRC
# ─────────────────────────────────────────────────────────────────────────────


def run_drc_with_recorder(
    gds_path,
    *,
    recorder: Optional[RunRecorder] = None,
    revision: str = "drc",
    summary: Optional[str] = None,
):
    """Run the REAL gf180mcu DRC deck on ``gds_path`` and record structured events.

    Wraps :func:`tools.klayout_drc_signoff_v139.run.run_gf180_drc`. Behaviour:

      * If ``recorder`` is None the function is a pure pass-through to
        ``run_gf180_drc(gds_path)`` -- no events emitted, original exception
        propagation preserved.

      * If ``recorder`` is provided:

          - Emits ``drc_started`` with the gds path, the KLayout binary
            availability, and whether the gf180mcu DRC deck is present on
            disk.

          - If EITHER the binary or the deck is missing, emits
            ``drc_skipped`` (with the explicit reason) and returns ``None``.
            The function never fabricates a verdict.

          - On a real run, emits ``drc_completed`` carrying ``is_clean``,
            ``n_violations``, ``ran_real_deck`` and the per-rule count map
            (``by_rule``). The :class:`Gf180DrcReport` is returned.

          - On any exception during the deck run, emits ``drc_skipped`` with
            the exception class / message and returns ``None`` -- the run
            never silently passes when the deck failed to execute.

    Returns the :class:`Gf180DrcReport` on a successful run, or ``None`` when
    skipped / failed.
    """
    # Import the harness lazily so importing this module does not depend on the
    # tools/ subtree (and so tests can monkeypatch the import target).
    from tools.klayout_drc_signoff_v139 import run as drc_run

    if recorder is None:
        return drc_run.run_gf180_drc(gds_path)

    klayout_ok = drc_run.klayout_binary_available()
    deck_ok = drc_run.DEFAULT_DECK.exists()

    recorder.event(
        "drc_started",
        summary=summary or f"Starting gf180mcu DRC on {gds_path}.",
        revision=revision,
        data={
            "title": "gf180mcu DRC starting",
            "gds_path": str(gds_path),
            "klayout_available": bool(klayout_ok),
            "deck_available": bool(deck_ok),
            "source": "tools.klayout_drc_signoff_v139.run.run_gf180_drc",
        },
    )

    if not (klayout_ok and deck_ok):
        reasons = []
        if not klayout_ok:
            reasons.append("KLayout binary unavailable")
        if not deck_ok:
            reasons.append("gf180mcu DRC deck not found on disk")
        reason = "; ".join(reasons)
        recorder.event(
            "drc_skipped",
            summary=f"DRC skipped: {reason}.",
            revision=revision,
            data={
                "title": "DRC skipped",
                "gds_path": str(gds_path),
                "reason": reason,
                "klayout_available": bool(klayout_ok),
                "deck_available": bool(deck_ok),
                "source": "tools.klayout_drc_signoff_v139.run",
            },
        )
        return None

    try:
        report = drc_run.run_gf180_drc(gds_path)
    except Exception as exc:  # noqa: BLE001 -- record honestly, return None
        recorder.event(
            "drc_skipped",
            summary=f"DRC skipped: deck run failed ({type(exc).__name__}).",
            revision=revision,
            data={
                "title": "DRC skipped (deck run failed)",
                "gds_path": str(gds_path),
                "reason": "gf180mcu DRC deck raised",
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "source": "tools.klayout_drc_signoff_v139.run.run_gf180_drc",
            },
        )
        return None

    _emit_drc_completed(recorder, report, revision=revision, summary=summary)
    return report


def _emit_drc_completed(
    recorder: RunRecorder,
    report,
    *,
    revision: str,
    summary: Optional[str],
) -> None:
    """Emit ``drc_completed`` with structured per-rule counts."""
    by_rule = report.counts_by_rule()
    n_violations = int(report.total_violations)
    clean = bool(report.is_clean())
    data: Dict[str, Any] = {
        "title": "gf180mcu DRC completed",
        "gds_path": str(report.gds_path) if report.gds_path is not None else None,
        "lyrdb_path": str(report.lyrdb_path) if report.lyrdb_path is not None else None,
        "is_clean": clean,
        "n_violations": n_violations,
        "rule_count": int(report.rule_count),
        "ran_real_deck": bool(report.ran_real_deck),
        "by_rule": dict(by_rule),
        "source": "tools.klayout_drc_signoff_v139.run.run_gf180_drc",
    }
    recorder.event(
        "drc_completed",
        summary=summary or report.summary(),
        revision=revision,
        data=data,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LVS
# ─────────────────────────────────────────────────────────────────────────────


def record_lvs(
    recorder: Optional[RunRecorder],
    *,
    lvs_report: Mapping[str, Any],
    revision: str = "lvs",
    summary: Optional[str] = None,
) -> Mapping[str, Any]:
    """Record an LVS verdict on the run archive from a STRUCTURED report dict.

    The LVS netlist comparator (e.g. netgen) is not wrapped end-to-end in this
    helper on purpose: not every Clay run has a single canonical LVS runner,
    and a future ``run_lvs`` entry point should be free to evolve without
    breaking this recorder. The helper therefore takes the report dict
    DIRECTLY -- the caller is responsible for producing it (from netgen, from
    a structural topology check, or from a stand-in placeholder while the LVS
    backend is still being wired).

    Expected ``lvs_report`` shape (every field optional except ``clean``)::

        {
            "clean": bool,
            "mismatches": [
                {"net": "...", "kind": "...", ...},  # whatever the LVS tool emits
                ...
            ],
            "extracted_devices": int,
            "schematic_devices": int,
            # ... any extra structured fields the caller wants to surface
        }

    Behaviour:

      * If ``recorder`` is None this is a no-op and the input dict is
        returned unchanged.

      * If ``recorder`` is provided:

          - ``lvs_completed`` is emitted when ``clean`` is truthy.
          - ``lvs_failed`` is emitted when ``clean`` is falsy, carrying the
            number of mismatches and (when present) the first mismatch as a
            ``worst`` field for fast triage.

    Returns the input ``lvs_report`` (so a caller can chain it into a final
    artifact write).
    """
    if recorder is None:
        return lvs_report

    clean = bool(lvs_report.get("clean", False))
    mismatches = list(lvs_report.get("mismatches", []) or [])
    n_mismatches = len(mismatches)
    extracted = lvs_report.get("extracted_devices")
    schematic = lvs_report.get("schematic_devices")

    data: Dict[str, Any] = {
        "title": "LVS completed" if clean else "LVS failed",
        "clean": clean,
        "n_mismatches": n_mismatches,
        "extracted_devices": extracted,
        "schematic_devices": schematic,
        "source": "clay.runtime.signoff_recorder.record_lvs",
    }
    # Surface a worst-offender slot when the caller listed mismatches.
    if mismatches:
        data["worst"] = mismatches[0]
        data["mismatches"] = mismatches
    # Forward any extra structured keys verbatim (skip the ones already lifted).
    for key, value in lvs_report.items():
        if key in {"clean", "mismatches", "extracted_devices", "schematic_devices"}:
            continue
        data.setdefault(key, value)

    if clean:
        recorder.event(
            "lvs_completed",
            summary=summary or (
                f"LVS clean ({extracted} extracted vs {schematic} schematic devices)."
                if extracted is not None and schematic is not None
                else "LVS clean."
            ),
            revision=revision,
            data=data,
        )
    else:
        recorder.event(
            "lvs_failed",
            summary=summary or f"LVS failed with {n_mismatches} mismatch(es).",
            revision=revision,
            data=data,
        )
    return lvs_report


# ─────────────────────────────────────────────────────────────────────────────
# Layout-intent consistency (structural + parasitic + placement)
# ─────────────────────────────────────────────────────────────────────────────


def run_constraints_with_recorder(
    routed_cell,
    intent,
    pex_report,
    *,
    recorder: Optional[RunRecorder] = None,
    revision: str = "consistency",
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the three Clay consistency checkers and record their joint verdict.

    The three axes the layout-intent contract certifies:

      * STRUCTURAL  -- ``check_global_constraints(routed_cell)``: every rule
        in ``global_layout_constraints.yaml`` (containment within rails, no
        signal crosses rails, area / wire info).
      * PARASITIC   -- ``check_layout_intent(intent, pex_report)``: one
        :class:`LayoutObligation` per declared :class:`ParasiticBudget`.
      * PLACEMENT   -- ``placement_symmetry_obligations(routed_cell, intent)``
        + ``mirror_placement_obligations(routed_cell, intent)``: matched
        device centroids must mirror about the cell axis.

    Behaviour:

      * Always returns a dict ``{"global": {...}, "intent": {...},
        "placement": {...}, "passed": bool}`` -- the SAME shape whether or
        not a recorder is attached. ``passed`` is True iff EVERY axis is
        clean.

      * If ``recorder`` is provided:

          - Emits exactly ONE ``consistency_completed`` event with all three
            axes' summary numbers under ``data``.

          - For each axis that ERRORS (structural errors, intent failures or
            unmeasurable obligations, placement obligations with status
            ``"fail"``), emits an additional ``consistency_violation`` event
            naming the axis and surfacing the worst offender for that axis.

      * If ``recorder`` is None this is a pure pass-through (no events).

    The function does not raise on a failing verdict; the caller decides how
    to act on the returned dict.
    """
    from clay.layout.global_constraints import (
        check_global_constraints,
        constraint_summary,
    )
    from clay.layout.intent_obligations import (
        check_layout_intent,
        layout_intent_verdict,
    )
    from clay.layout.placement_symmetry import (
        mirror_placement_obligations,
        placement_symmetry_obligations,
        placement_symmetry_verdict,
    )

    # STRUCTURAL axis ──────────────────────────────────────────────────────
    structural_viols = check_global_constraints(routed_cell)
    struct_summary = constraint_summary(structural_viols)
    structural_axis: Dict[str, Any] = {
        "passed": bool(struct_summary.get("passed", True)),
        "n_errors": int(struct_summary.get("n_error", 0)),
        "n_warnings": int(struct_summary.get("n_warning", 0)),
        "per_rule": dict(struct_summary.get("per_rule", {})),
    }
    # Worst structural offender = first ERROR severity violation, if any.
    structural_errors = [v for v in structural_viols if v.severity == "error"]
    if structural_errors:
        worst = structural_errors[0]
        structural_axis["worst"] = {
            "rule_id": worst.rule_id,
            "severity": worst.severity,
            "message": worst.message,
            "net": worst.net,
            "layer": worst.layer,
            "device": worst.device,
        }

    # PARASITIC (intent) axis ────────────────────────────────────────────
    intent_obls = check_layout_intent(intent, pex_report)
    intent_verdict = layout_intent_verdict(intent_obls)
    intent_axis: Dict[str, Any] = {
        "passed": bool(intent_verdict.get("passed", False)),
        "n_obligations": len(intent_obls),
        "n_failed": len(intent_verdict.get("failed", [])),
        "n_unmeasurable": len(intent_verdict.get("unmeasurable", [])),
        "empty": bool(intent_verdict.get("empty", False)),
    }
    worst_obl = intent_verdict.get("worst")
    if worst_obl is not None:
        intent_axis["worst"] = {
            "kind": worst_obl.kind,
            "pair": list(worst_obl.pair),
            "measured_imbalance": worst_obl.measured_imbalance,
            "rel_tol": worst_obl.rel_tol,
            "status": worst_obl.status,
            "over_budget": worst_obl.over_budget,
        }

    # PLACEMENT axis ────────────────────────────────────────────────────
    diff_obls = list(placement_symmetry_obligations(routed_cell, intent))
    mirror_obls = list(mirror_placement_obligations(routed_cell, intent))
    all_placement = diff_obls + mirror_obls
    placement_verdict = placement_symmetry_verdict(all_placement)
    placement_axis: Dict[str, Any] = {
        "passed": bool(placement_verdict.get("passed", True)),
        "n_obligations": int(placement_verdict.get("n_obligations", len(all_placement))),
        "n_failed": len(placement_verdict.get("failed", [])),
        "worst_pair_offset": float(placement_verdict.get("worst", 0.0)),
    }
    worst_pair = placement_verdict.get("worst_pair")
    if worst_pair is not None:
        placement_axis["worst_pair"] = {
            "devices": list(worst_pair.get("devices", ())),
            "pair": list(worst_pair.get("pair", ())),
        }

    overall_passed = (
        structural_axis["passed"]
        and intent_axis["passed"]
        and placement_axis["passed"]
    )
    verdict: Dict[str, Any] = {
        "global": structural_axis,
        "intent": intent_axis,
        "placement": placement_axis,
        "passed": bool(overall_passed),
    }

    if recorder is None:
        return verdict

    cell_name = getattr(intent, "cell", None) or getattr(routed_cell, "cell_name", None)
    recorder.event(
        "consistency_completed",
        summary=summary or (
            f"Consistency check {'passed' if overall_passed else 'failed'}"
            + (f" for {cell_name}" if cell_name else "")
            + "."
        ),
        revision=revision,
        data={
            "title": "Layout-intent consistency completed",
            "cell": cell_name,
            "passed": bool(overall_passed),
            "global": structural_axis,
            "intent": intent_axis,
            "placement": placement_axis,
            "source": "clay.runtime.signoff_recorder.run_constraints_with_recorder",
        },
    )

    # Per-axis violation events for fast triage in the viewer.
    if not structural_axis["passed"]:
        recorder.event(
            "consistency_violation",
            summary=(
                f"Structural axis failed ({structural_axis['n_errors']} error(s))."
            ),
            revision=revision,
            data={
                "title": "Structural-constraint violation",
                "axis": "global",
                "cell": cell_name,
                "n_errors": structural_axis["n_errors"],
                "worst": structural_axis.get("worst"),
                "per_rule": structural_axis["per_rule"],
                "source": "clay.layout.global_constraints.check_global_constraints",
            },
        )
    if not intent_axis["passed"]:
        recorder.event(
            "consistency_violation",
            summary=(
                f"Parasitic-intent axis failed "
                f"({intent_axis['n_failed']} failed, "
                f"{intent_axis['n_unmeasurable']} unmeasurable)."
            ),
            revision=revision,
            data={
                "title": "Parasitic-intent obligation violation",
                "axis": "intent",
                "cell": cell_name,
                "n_failed": intent_axis["n_failed"],
                "n_unmeasurable": intent_axis["n_unmeasurable"],
                "worst": intent_axis.get("worst"),
                "source": "clay.layout.intent_obligations.check_layout_intent",
            },
        )
    if not placement_axis["passed"]:
        recorder.event(
            "consistency_violation",
            summary=(
                f"Placement-symmetry axis failed "
                f"({placement_axis['n_failed']} mis-mirrored pair(s))."
            ),
            revision=revision,
            data={
                "title": "Placement-symmetry violation",
                "axis": "placement",
                "cell": cell_name,
                "n_failed": placement_axis["n_failed"],
                "worst_pair_offset": placement_axis["worst_pair_offset"],
                "worst_pair": placement_axis.get("worst_pair"),
                "source": "clay.layout.placement_symmetry.placement_symmetry_obligations",
            },
        )

    return verdict
