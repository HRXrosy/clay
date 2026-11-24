"""Public CERTIFY entry-point for the layout-intent contract.

The layout-intent contract DECLARES net-level parasitic budgets
(``clay.layout.intent.derive_layout_intent``), CHECKS them against a real PEX
extraction (``clay.layout.intent_obligations.check_layout_intent``), and folds
each verdict into a proof-obligation row in the exact signoff shape the
proof-token gate consumes (``clay.layout.intent_signoff``). This module is the
single clean public call that runs that whole pipeline and either signs a proof
token or withholds it.

The token-withholding policy is IDENTICAL to DRC: the gate
(``clay.verifier._certify_obligation_rows``) signs a token only when every
``required`` row is ``status == "pass"``. A parasitic budget OVER tolerance
produces a ``required`` row with ``status == "fail"``, which is non-PASS and
therefore withholds the token exactly the way a DRC violation row does. A
withheld token is ``None`` -- it is NEVER fabricated.

Everything reads a :class:`~clay.layout.pex.base.PexReport`, so the certify call
is unit-testable WITHOUT Magic/Docker by passing a synthetic report. All
thresholds are RELATIVE fractions (project-wide rule: never absolute aF deltas).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from clay.verifier import _certify_obligation_rows

from .intent import LayoutIntent, derive_layout_intent
from .intent_completeness import completeness_obligations
from .intent_obligations import LdeGeometry, check_layout_intent
from .intent_signoff import SOURCE_KIND, _obligation_row, layout_intent_obligation_rows
from .pex.base import PexReport

# The certify path is a layout (not electrical-corner) signoff; it carries no
# real PDK corner / temperature, so stamp neutral placeholders into the proof
# token gate (which is corner-aware for electrical specs but corner-agnostic for
# a geometric/parasitic layout obligation).
_LAYOUT_PDK_CORNER = "layout"
_LAYOUT_TEMP = 27.0


def certify_layout_intent(
    cell,
    pex_report: PexReport,
    *,
    required: bool = True,
    intent: Optional[LayoutIntent] = None,
    geometry: Optional[LdeGeometry] = None,
    placement_geometry: Optional[object] = None,
    routed_geometry: Optional[object] = None,
    annotated=None,
    **tol,
) -> Dict[str, object]:
    """Run the full layout-intent CERTIFY pipeline and sign-or-withhold a token.

    Pipeline (one clean call):

      1. ``derive_layout_intent(cell, **tol)`` -- the declared parasitic budgets.
         Skipped when ``intent=`` is supplied EXPLICITLY (the explicit-intent path,
         see below).
      2. ``check_layout_intent(intent, pex_report)`` -- one obligation per budget.
      3. ``layout_intent_obligation_rows(...)`` -- map each verdict into a
         proof-obligation row in the signoff shape.
      4. ``clay.verifier._certify_obligation_rows(rows=rows, ...)`` -- the SAME
         proof-token gate DRC/LVS use: it returns ``(status, token)`` and signs a
         token only when every required row is ``pass``.

    Device-environment (LDE) budgets
    --------------------------------
    A cell that declares ``@equal_well_edge`` / ``@equal_od`` /
    ``@equal_od_spacing`` budgets needs DEVICE GEOMETRY to measure them (the PEX
    report carries caps, not per-device well/OD geometry). Pass that geometry via
    ``geometry=`` (a :class:`~clay.layout.intent_obligations.LdeGeometry`, e.g.
    ``LdeGeometry.from_cell(routed_cell, pdk_layers)``). When a cell declares an
    LDE budget but no geometry is supplied, that budget is UNMEASURABLE and the
    token is WITHHELD -- never a silent pass.

    Routed-geometry budgets
    -----------------------
    ``@wire_width`` and ``@match_length`` declarations require a
    :class:`~clay.layout.geometric_intent_checks.RoutedGeometry` supplied through
    ``routed_geometry=``. The exact flow builds that view from the emitted
    artifact's exhaustive ownership manifest. Missing routed evidence is
    UNMEASURABLE and withholds the token.

    The v2 explicit-intent path
    ----------------------------
    Callers that already HAVE a :class:`LayoutIntent` (e.g. a profile-composed
    one, a designer-merged one, a parsed-from-source one) pass it through
    ``intent=`` and ``cell`` is taken from ``intent.cell``. The ``cell`` positional
    can then be either the cell name or the same intent (we honour both); the
    legacy form ``certify_layout_intent("comparator", report)`` still re-derives
    via ``derive_layout_intent(cell, **tol)`` so nothing else changes.

    Returns::

        {
          "cell": <cell name>,
          "rows": [ <proof-obligation row>, ... ],     # signoff-shaped
          "obligations": [ <LayoutObligation>, ... ],  # the raw verdicts
          "status": "pass" | "fail" | "no_obligation" | ...,
          "token": <signed proof token dict> | None,   # None when withheld
          "passed": bool,                               # status == "pass"
          "blocking": [ <failing obligation kind>, ... ],
          "intent_digest": <16-hex stable hash of the certified intent>,
          "intent_source": <"auto" | "category-profile" | "declared" | ...>,
          "intent_profile_id": <profile-id or None>,
        }

    ``token`` is ``None`` whenever any required budget fails -- withheld exactly
    the way DRC withholds it. A token is NEVER fabricated.
    """
    # Resolve the intent to certify: explicit > cell-as-intent > re-derive by name.
    if intent is None:
        if isinstance(cell, LayoutIntent):
            intent = cell
        else:
            intent = derive_layout_intent(cell, **tol)
    obligations = check_layout_intent(
        intent, pex_report, geometry=geometry,
        placement_geometry=placement_geometry,
        routed_geometry=routed_geometry,
    )
    rows = layout_intent_obligation_rows(
        intent, pex_report, required=required, geometry=geometry,
        placement_geometry=placement_geometry,
        routed_geometry=routed_geometry,
    )

    # ── NO-SILENT-IGNORE completeness gate ──────────────────────────────────
    # Every DECLARED intent must be backed by a covering obligation above. Any
    # declared intent with NO obligation is silently ignored today -- the layout
    # claims to honor it but nothing checks it. The completeness gate turns each
    # such gap into a BLOCKING ``unchecked_intent:<kind>`` obligation so the proof
    # token is WITHHELD with an explicit "declared but not verified" reason rather
    # than silently signed. Driven purely by the declared-intent set vs the
    # obligation set (no per-circuit branching); a no-op when every declared
    # intent is already covered. Pass ``annotated=`` (an AnnotatedCell) to also
    # demand checks for the device-level ``annotate`` block claims
    # (@interdigitated / @common_centroid / @symmetric / @adjacent / ...).
    gate_obligations = completeness_obligations(
        intent, obligations, annotated=annotated
    )
    if gate_obligations:
        obligations = list(obligations) + gate_obligations
        rows = list(rows) + [
            _obligation_row(intent.cell, ob, required=required)
            for ob in gate_obligations
        ]

    status, token = _certify_obligation_rows(
        cell_name=intent.cell,
        pdk_corner=_LAYOUT_PDK_CORNER,
        temp=_LAYOUT_TEMP,
        rows=rows,
    )

    blocking: List[str] = [
        row["obligation_kind"] for row in rows if row["status"] != "pass"
    ]

    return {
        "cell": intent.cell,
        "rows": rows,
        "obligations": obligations,
        "status": status,
        "token": token,
        "passed": status == "pass",
        "blocking": blocking,
        "source_kind": SOURCE_KIND,
        "intent_digest": intent.digest,
        "intent_source": intent.source,
        "intent_profile_id": intent.profile_id,
    }


def _fmt_pair(row: dict) -> str:
    pair = row.get("evidence", {}).get("pair") or []
    if len(pair) == 2:
        return f"{pair[0]}~{pair[1]}"
    # fall back to the check token already on the row's source
    return str(row.get("source", {}).get("name", ""))


def format_layout_intent_report(certify_result: Dict[str, object]) -> str:
    """Render an aligned human-readable text report for a certify result.

    Columns: ``kind | pair | measured | budget | PASS/FAIL``. Followed by the
    verdict line and the proof-token state (signed digest / WITHHELD).
    """
    cell = str(certify_result.get("cell", "?"))
    rows: List[dict] = list(certify_result.get("rows", []))  # type: ignore[arg-type]

    headers = ("kind", "pair", "measured", "budget", "verdict")
    table_rows: List[tuple] = []
    for row in rows:
        ev = row.get("evidence", {})
        raw_measured = ev.get("measured_imbalance")
        budget = float(ev.get("rel_tol", 0.0))
        status = row.get("status")
        if status == "pass":
            verdict = "PASS"
        elif raw_measured is None:  # net absent from the extraction
            verdict = "UNMEAS"
        else:
            verdict = "FAIL"
        measured_str = " n/a   " if raw_measured is None else f"{float(raw_measured):7.4f}"
        table_rows.append(
            (
                str(row.get("obligation_kind", "?")),
                _fmt_pair(row),
                measured_str,
                f"{budget:7.4f}",
                verdict,
            )
        )

    # Column widths from headers + content.
    widths = [len(h) for h in headers]
    for tr in table_rows:
        for i, cellv in enumerate(tr):
            widths[i] = max(widths[i], len(cellv))

    def _line(cols) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols))

    sep = "  ".join("-" * w for w in widths)

    out: List[str] = []
    out.append(f"Layout-intent certification: cell={cell}")
    out.append("")
    out.append(_line(headers))
    out.append(sep)
    if table_rows:
        for tr in table_rows:
            out.append(_line(tr))
    else:
        out.append("(no declared parasitic budgets)")
    out.append("")

    status = str(certify_result.get("status", "?"))
    passed = bool(certify_result.get("passed", False))
    blocking = list(certify_result.get("blocking", []))  # type: ignore[arg-type]
    verdict_word = "CERTIFIED" if passed else "NOT CERTIFIED"
    out.append(f"Verdict: {verdict_word} (status={status})")
    if blocking:
        out.append(f"Blocking budgets: {', '.join(blocking)}")

    token = certify_result.get("token")
    if token is None:
        out.append(
            "Proof token: WITHHELD (a required parasitic budget failed; "
            "withheld exactly like a DRC violation -- never fabricated)"
        )
    else:
        digest = ""
        if isinstance(token, dict):
            digest = str(token.get("trace_digest", ""))
        out.append(f"Proof token: SIGNED (trace_digest={digest})")

    return "\n".join(out)


__all__ = ["certify_layout_intent", "format_layout_intent_report"]
