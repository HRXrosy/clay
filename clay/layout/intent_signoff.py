"""Fold declared layout-intent (parasitic) budgets into the CERTIFICATION path.

``clay.layout.intent_obligations.check_layout_intent`` turns a cell's declared
:class:`~clay.layout.intent.LayoutIntent` budgets into a list of
:class:`~clay.layout.intent.LayoutObligation` verdicts against a real
:class:`~clay.layout.pex.base.PexReport`. Those verdicts live OUTSIDE the
proof-token gate: a failing parasitic budget does not, on its own, withhold a
cell's certificate.

This module bridges that gap. It maps each ``LayoutObligation`` into a
proof-obligation row in the EXACT shape ``clay.signoff_obligation_trace`` emits
for DRC/LVS/ERC signoff (same dict fields, same ``required``/``status``
conventions), so the proof-token gate
(``clay.verifier._certify_obligation_rows``) treats a "declared parasitic intent
satisfied" obligation just like a DRC obligation:

  * a budget WITHIN tolerance  -> a ``required`` row with ``status == "pass"``;
  * a budget OVER tolerance    -> a ``required`` row with ``status == "fail"``,
    which is non-PASS and therefore blocks the proof token exactly the way a DRC
    violation row does (the gate signs a token only when every required row is
    ``pass``).

The obligation-id scheme is ``layout_intent:<kind>:<pairA>-<pairB>`` (stable and
unique per budget). The row's structural shape is produced by reusing
:func:`clay.signoff_obligation_trace._row` (the single source of truth for the
row dataclass/dict shape) and then stamping the layout-intent obligation id /
trace id over the signoff defaults, so no row shape is duplicated here.

Everything reads a ``PexReport``, so this is unit-testable WITHOUT Magic/Docker.
All thresholds are RELATIVE fractions (project-wide rule: never absolute aF
deltas).
"""
from __future__ import annotations

from typing import Dict, List

from clay.signoff_obligation_trace import _row

from typing import Optional

from .intent import LayoutIntent, LayoutObligation
from .intent_obligations import LdeGeometry, check_layout_intent
from .pex.base import PexReport

# Source kind tag distinguishing layout-intent rows from DRC/LVS layout_signoff
# rows while keeping the same row shape the proof-token gate consumes.
SOURCE_KIND = "layout_intent_signoff"


def _pair_token(pair) -> str:
    """``(a, b)`` -> ``a-b`` for a stable, unique id suffix."""
    a, b = pair
    return f"{a}-{b}"


def _obligation_row(
    cell_name: str,
    obligation: LayoutObligation,
    *,
    required: bool,
) -> dict:
    """One proof-obligation row for a single layout-intent verdict.

    Built on :func:`clay.signoff_obligation_trace._row` so the dict shape
    (obligation_id / trace_id / cell / source_kind / obligation_kind / required /
    status / source / coverage / evidence) is byte-for-byte the signoff shape.
    The signoff-style ``obl:...`` id/trace are then replaced by the requested
    ``layout_intent:<kind>:<pairA>-<pairB>`` scheme.
    """
    check = _pair_token(obligation.pair)
    status = obligation.status  # "pass" / "fail" / "unmeasurable" from LayoutObligation
    measured = obligation.measured_imbalance
    is_matching = obligation.kind == "matching"
    if is_matching:
        coverage = {
            "artifact_kind": "gds",
            # Matching is a real exact-artwork measurement only when it reached
            # a measured pass/fail verdict.  Capability absence and missing
            # geometry remain unsupported/unmeasurable, never a tool run.
            "ran_real_tool": measured is not None and status in {"pass", "fail"},
            "metric": "max_matching_geometry_imbalance",
        }
    else:
        coverage = {
            "artifact_kind": "pex_report",
            # Preserve the established parasitic-budget coverage contract.
            "ran_real_tool": measured is not None,
            "metric": "relative_imbalance",
        }
    row = _row(
        cell_name=cell_name,
        obligation_kind=obligation.kind,
        check=check,
        status=status,
        coverage=coverage,
        evidence={
            # None when the budget references a net absent from the extraction
            "measured_imbalance": None if measured is None else float(measured),
            "rel_tol": float(obligation.rel_tol),
            "over_budget": float(obligation.over_budget),
            "pair": list(obligation.pair),
            "missing_nets": list(getattr(obligation, "missing_nets", ()) or ()),
            # Closed verification state + source declaration identity survive
            # the proof-row boundary.  In particular, capability absence stays
            # ``unsupported`` and reciprocal @matching declarations remain
            # independently auditable after collapsing to one unordered row.
            "verification_status": str(obligation.status),
            "declaration_ids": list(
                getattr(obligation, "declaration_ids", ()) or ()
            ),
            "measurements": dict(
                getattr(obligation, "measurements", {}) or {}
            ),
        },
    )
    # Re-stamp the signoff defaults with the layout-intent identity. The shape
    # (the set of keys + nested source keys) is preserved; only the id/trace/
    # source_kind/required/source-detail values change.
    obligation_id = f"layout_intent:{obligation.kind}:{check}"
    trace_id = f"tr:{cell_name}:layout_intent:{obligation.kind}:{check}"
    row["obligation_id"] = obligation_id
    row["trace_id"] = trace_id
    row["source_kind"] = SOURCE_KIND
    row["required"] = bool(required)
    row["source"] = {
        "cell": cell_name,
        "name": check,
        "field": obligation.kind,
        "ir_symbol": f"{cell_name}.layout_intent.{obligation.kind}.{check}",
        "source_ref": {
            "kind": "layout_intent",
            "path": f"cell[{cell_name}].layout_intent[{obligation.kind}][{check}]",
        },
    }
    return row


def layout_intent_obligation_rows(
    intent: LayoutIntent,
    pex_report: PexReport,
    *,
    required: bool = True,
    geometry: Optional[LdeGeometry] = None,
    placement_geometry: Optional[object] = None,
    routed_geometry: Optional[object] = None,
) -> List[dict]:
    """Proof-obligation rows for a cell's declared parasitic intent.

    Runs :func:`check_layout_intent` and maps every resulting
    :class:`LayoutObligation` to a row in the same shape
    ``clay.signoff_obligation_trace`` uses for DRC/LVS signoff. Each row's
    ``status`` is the obligation's own pass/fail verdict; ``required`` controls
    whether a failing budget blocks the proof token (default ``True``, matching
    how DRC violation rows block it).

    ``geometry`` carries the device geometry the device-environment (LDE) budgets
    measure against (well-edge / OD-length / OD-spacing). It is threaded straight
    into :func:`check_layout_intent`; when omitted, any declared LDE budget is
    UNMEASURABLE and its row is non-PASS -> the proof token is withheld exactly
    like a missing-net cap budget.

    ``routed_geometry`` carries the exact-artifact-bound route ownership view for
    ``@wire_width`` / ``@match_length``. It is forwarded to the same checker used
    to build the returned obligations so proof rows and the public verdict cannot
    disagree about routed evidence.
    """
    obligations = check_layout_intent(
        intent, pex_report, geometry=geometry,
        placement_geometry=placement_geometry,
        routed_geometry=routed_geometry,
    )
    return [
        _obligation_row(intent.cell, obligation, required=required)
        for obligation in obligations
    ]


def _record_layout_episode_best_effort(
    intent: LayoutIntent,
    pex_report: PexReport,
) -> None:
    """Deposit a layout-flywheel episode for this signoff -- best-effort, never
    raises, output-neutral.

    R2 (layout half of self-bootstrapping): every real layout-intent signoff
    leaves a transferable trace of which builder/router cleanly satisfied a given
    intent, so the WarmStart policy (``clay.layout.layout_policy``) can later
    propose a choice.  This call sits at the TAIL of ``layout_intent_signoff``
    AFTER the verdict is fully computed, and its result is discarded -- it can
    NEVER change the returned signoff record.  Any failure (recording disabled,
    import issue, store error) is swallowed.

    The parasitic-intent path does not re-run DRC/LVS, so the recorded
    ``drc_clean`` / ``lvs_match`` are left at their honest default (True for "not
    a blocker on this path"); the record's strength is the obligation
    satisfaction the checker actually measured against ``pex_report``.  The
    builder/router are unknown here, so neutral markers are recorded -- a richer
    hook (e.g. ``full_layout_signoff`` with a routed cell in scope) can record
    the concrete builder/router.
    """
    try:
        from .intent_obligations import check_layout_intent
        from .layout_episode import record_layout_episode

        obligations = check_layout_intent(intent, pex_report)
        record_layout_episode(
            intent,
            obligations,
            builder="layout_intent_signoff",
            router_generation="",
            params={"pex_backend": str(getattr(pex_report, "backend", ""))},
        )
    except Exception:  # noqa: BLE001 -- recording must never affect the verdict
        return


def layout_intent_signoff(
    intent: LayoutIntent,
    pex_report: PexReport,
) -> Dict[str, object]:
    """Summary signoff record for a cell's declared parasitic intent.

    Mirrors the signoff-record style: ``rows`` is the full proof-obligation row
    list, ``passed`` is True iff every row passed, and ``blocking`` lists the
    kinds whose budget failed (the ones that withhold the proof token).
    """
    rows = layout_intent_obligation_rows(intent, pex_report)
    blocking = [row["obligation_kind"] for row in rows if row["status"] != "pass"]
    result = {
        "cell": intent.cell,
        "source_kind": SOURCE_KIND,
        "rows": rows,
        "passed": not blocking,
        "blocking": blocking,
    }
    # ── R2 layout-flywheel hook: additive, output-neutral, best-effort ───────
    # Deposit an episode AFTER the verdict is computed; the result above is the
    # return value and is NOT touched by the recording call.
    _record_layout_episode_best_effort(intent, pex_report)
    return result
