"""UNIFIED machine-readable SIGNOFF RECORD: parasitics AND placement in one dict.

The layout-intent contract is certified along TWO axes -- net-level ROUTING
parasitics and DEVICE PLACEMENT -- and ``clay.layout.full_signoff.full_layout_signoff``
already fuses them into ONE verdict + combined score + proof token (the token is
signed only when BOTH axes pass; a failure on either withholds it exactly like a
DRC violation). What that call returns is a live result object: it carries frozen
dataclasses (``LayoutObligation``) and tuples (net ``pair``s, device ``pair``s),
so it cannot be diffed, archived, or fed into a cross-cell benchmark matrix
without coercion.

This module is the JSON counterpart of :mod:`clay.layout.intent_record` (which
records the PARASITIC-only signoff) but for the BOTH-AXES signoff. It runs the
SAME ``full_layout_signoff`` pipeline and folds the result into one fully
JSON-serializable record (schema ``clay.layout_full_signoff/v1``):

  * ``parasitic_obligations`` -- one JSON-clean entry per parasitic budget
    (kind / net pair / measured relative imbalance / tolerance / status), the
    frozen ``LayoutObligation`` dataclass and its tuple ``pair`` coerced to plain
    types -- the SAME entry shape ``intent_record`` uses, so a parasitic obligation
    reads identically in either record.
  * ``placement_obligations`` -- one JSON-clean entry per matched DEVICE pair
    (diff-pair nets / device labels / relative dx/dy misalignment / tolerance /
    status / match kind), the obligation dict's tuple ``pair`` / ``devices``
    coerced to lists.
  * ``score`` -- the combined scalar plus its two per-axis numbers
    (``parasitic_score``, ``placement_pass_fraction``), all plain floats.
  * ``token`` -- the signed token's digest + the obligation ids it covers, or a
    withheld record (``signed=False`` / ``digest=None``) when EITHER axis failed.
    The digest is pulled from the SIGNED token's ``trace_digest``; a withheld
    token has no digest, which is reported as ``None`` and is NEVER fabricated --
    a placement failure withholds it exactly the way a parasitic failure does.
  * ``provenance`` -- the extraction backend / calibration / element + net counts,
    plus the placement obligation count (placement is measured straight off the
    placed geometry, no PEX tool run).

Serializability guarantee: every value placed in the record is a JSON scalar,
list, or dict -- no tuple, no frozenset, no dataclass survives. Tuples (net /
device pairs) become 2-element lists; the ``frozenset`` of extraction nets is
reduced to a COUNT (``n_nets``), never embedded raw. So ``json.dumps(record)``
always succeeds and the record round-trips byte-for-byte through ``json.loads``.
All thresholds remain RELATIVE fractions (project-wide rule: never absolute aF /
nm deltas).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from .full_signoff import PLACEMENT_SOURCE_KIND, full_layout_signoff
from .pex.base import PexReport

SCHEMA = "clay.layout_full_signoff/v1"


def _parasitic_entry(obligation) -> Dict[str, object]:
    """One JSON-serializable entry from a parasitic :class:`LayoutObligation`.

    Coerces the frozen dataclass + its tuple ``pair`` into plain JSON types,
    matching :func:`clay.layout.intent_record._obligation_entry` so a parasitic
    obligation reads identically in the parasitic-only and the unified record.
    The net pair becomes a 2-element list; the measured imbalance stays a float
    (or ``None`` when the relationship was never extracted).
    """
    measured = obligation.measured_imbalance
    a, b = obligation.pair
    return {
        "axis": "parasitic",
        "kind": str(obligation.kind),
        "pair": [str(a), str(b)],
        "measured_imbalance": None if measured is None else float(measured),
        "rel_tol": float(obligation.rel_tol),
        "status": str(obligation.status),
        "over_budget": float(obligation.over_budget),
    }


def _placement_entry(obligation: Dict[str, object]) -> Dict[str, object]:
    """One JSON-serializable entry from a placement-symmetry obligation dict.

    ``placement_symmetry_obligations`` returns plain dicts but with TUPLE ``pair``
    (the diff-pair nets) and TUPLE ``devices`` (the matched device labels); both
    are coerced to 2-element lists here so nothing tuple-shaped survives into the
    record. The relative dx/dy misalignments, tolerance, status, and match kind
    are taken straight off the obligation.
    """
    a, b = obligation["pair"]
    da, db = obligation["devices"]
    return {
        "axis": "placement",
        "pair": [str(a), str(b)],
        "devices": [str(da), str(db)],
        "dx_imbalance": float(obligation["dx_imbalance"]),
        "dy_offset": float(obligation["dy_offset"]),
        "rel_tol": float(obligation["rel_tol"]),
        "status": str(obligation["status"]),
        "match_kind": str(obligation.get("match_kind", "differential")),
    }


def _score_entry(score: Dict[str, object]) -> Dict[str, object]:
    """The combined + per-axis quality numbers, JSON-clean (plain floats only).

    The full-signoff ``score`` block also nests the whole parasitic score dict
    (which carries a tuple-bearing ``weakest`` entry); that is intentionally NOT
    embedded raw. The record exposes only the three plain scalars the benchmark
    cares about so the whole block is trivially serializable.
    """
    return {
        "combined": float(score["combined"]),
        "parasitic_score": float(score["parasitic_score"]),
        "placement_pass_fraction": float(score["placement_pass_fraction"]),
    }


def _token_entry(token: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Token sub-record: signed digest pulled from the token, or withheld -> None.

    The digest is the SIGNED token's ``trace_digest``. A withheld token is
    ``None`` (a required parasitic OR placement obligation failed), in which case
    ``signed`` is False, ``digest`` is None, and ``obligation_ids`` is empty. A
    digest is NEVER fabricated -- a placement failure withholds the token exactly
    the way a parasitic failure does, and the withheld record reflects that.
    """
    if token is None:
        return {"signed": False, "digest": None, "obligation_ids": []}
    digest = token.get("trace_digest")
    return {
        "signed": True,
        "digest": None if digest is None else str(digest),
        "obligation_ids": [str(i) for i in token.get("obligation_ids", [])],
    }


def _provenance_entry(
    pex_report: PexReport,
    n_placement_obligations: int,
) -> Dict[str, object]:
    """Extraction + placement provenance for the unified record.

    The parasitic axis reads a ``PexReport`` (backend / calibration / element +
    net counts -- the ``frozenset`` of nets is reduced to a COUNT, never embedded
    raw). The placement axis reads the placed GEOMETRY (no PEX tool run), so its
    provenance is just how many matched device pairs were measured.
    """
    return {
        "pex_backend": str(pex_report.backend),
        "calibrated": bool(pex_report.calibrated),
        "n_caps": len(pex_report.caps),
        "n_nets": len(pex_report.nets()),
        # A real extraction tool ran iff the report claims calibration (a
        # geometric estimate or a synthetic report is uncalibrated).
        "ran_real_pex_tool": bool(pex_report.calibrated),
        # Placement is measured off the placed geometry -- no PEX tool, no Magic.
        "placement_measured_from_geometry": True,
        "n_placement_obligations": int(n_placement_obligations),
    }


def build_full_signoff_record(
    cell,
    routed_cell,
    pex_report: PexReport,
    *,
    intent=None,
    check_shields: bool = False,
    pdk: str = "gf180mcu",
    **tol,
) -> Dict[str, object]:
    """Assemble the UNIFIED (parasitics + placement) machine-readable signoff record.

    Runs the SAME both-axes certify pipeline as the human verdict
    (:func:`clay.layout.full_signoff.full_layout_signoff`, which derives the layout
    intent, checks every parasitic budget against ``pex_report``, measures every
    matched device pair's placement symmetry off ``routed_cell``'s geometry, feeds
    the COMBINED required rows to the single proof-token gate, and returns the
    combined score) and folds BOTH axes into one fully JSON-serializable dict.

    ``cell`` is the cell NAME (str), or a cell object exposing ``.name``;
    ``routed_cell`` is the placed/routed cell whose geometry the placement axis
    reads (e.g. ``build_routed_comparator()``); ``pex_report`` carries the
    parasitics for the budget axis. ``**tol`` forwards RELATIVE-tolerance overrides
    to the intent deriver.

    The token digest is the SIGNED token's ``trace_digest``; when the token is
    withheld (a required obligation on EITHER axis failed) the record reports
    ``signed=False`` / ``digest=None`` and the blocking obligation ids -- the
    digest is never fabricated, so a placement failure withholds the token exactly
    like a parasitic failure.

    Returns a dict matching ``clay.layout_full_signoff/v1``::

        {
          "schema", "cell", "pdk", "status", "passed", "blocking",
          "parasitic_obligations": [ ... ], "placement_obligations": [ ... ],
          "score": {"combined", "parasitic_score", "placement_pass_fraction"},
          "token": {"signed", "digest", "obligation_ids"},
          "provenance": { ... },
        }
    """
    result = full_layout_signoff(
        cell, routed_cell, pex_report,
        intent=intent, check_shields=check_shields,
        **tol,
    )

    parasitic_obligations = result["parasitic_obligations"]
    placement_obligations = result["placement_obligations"]

    return {
        "schema": SCHEMA,
        "cell": str(result["cell"]),
        "pdk": str(pdk),
        "status": str(result["status"]),
        "passed": bool(result["passed"]),
        "blocking": [str(b) for b in result["blocking"]],  # type: ignore[union-attr]
        "parasitic_obligations": [
            _parasitic_entry(o) for o in parasitic_obligations  # type: ignore[union-attr]
        ],
        "placement_obligations": [
            _placement_entry(o) for o in placement_obligations  # type: ignore[union-attr]
        ],
        "score": _score_entry(result["score"]),  # type: ignore[arg-type]
        "token": _token_entry(result["token"]),  # type: ignore[arg-type]
        "provenance": _provenance_entry(pex_report, len(placement_obligations)),  # type: ignore[arg-type]
        # v2: bind the record to the EXACT intent that was certified; see
        # ``LayoutIntent.digest`` / ``clay.layout.intent_record``.
        "intent_digest": str(result["intent_digest"]),
        "intent_source": str(result["intent_source"]),
        "intent_profile_id": (
            None if result["intent_profile_id"] is None
            else str(result["intent_profile_id"])
        ),
    }


def _count_status(entries: List[Dict[str, object]], status: str) -> int:
    """Count obligation entries whose ``status`` equals ``status``."""
    return sum(1 for e in entries if e.get("status") == status)


def full_benchmark_row(cell, record: Dict[str, object]) -> Dict[str, object]:
    """A flat, JSON-serializable benchmark-matrix row that reflects BOTH axes.

    Self-contained: derives every field from the unified signoff ``record`` alone
    (no external framework required). Suitable as one row in a cross-cell matrix
    that now scores PLACEMENT alongside parasitics, so a cell that routes its
    parasitics perfectly but places a differential pair asymmetrically no longer
    looks identical to a fully-symmetric one.

    A non-pass parasitic obligation (a budget FAIL or an UNMEASURABLE one) counts
    as a parasitic fail; only a measured ``pass`` counts as a parasitic pass.

    Fields::

        {
          "cell": <cell name>,
          "pdk": <process kit>,
          "combined_score": <combined parasitic+placement scalar in [0, 1]>,
          "passed": <did the layout meet its declared intent on BOTH axes?>,
          "token_signed": <was a unified proof token signed?>,
          "n_parasitic_pass", "n_parasitic_fail",
          "n_placement_pass", "n_placement_fail",
        }
    """
    parasitic = list(record.get("parasitic_obligations", []))  # type: ignore[arg-type]
    placement = list(record.get("placement_obligations", []))  # type: ignore[arg-type]
    n_parasitic_pass = _count_status(parasitic, "pass")
    n_placement_pass = _count_status(placement, "pass")
    score = record.get("score", {}) or {}
    token = record.get("token", {}) or {}
    return {
        "cell": str(cell),
        "pdk": str(record.get("pdk", "")),
        "combined_score": float(score.get("combined", 0.0)),  # type: ignore[union-attr]
        "passed": bool(record.get("passed", False)),
        "token_signed": bool(token.get("signed", False)),  # type: ignore[union-attr]
        "n_parasitic_pass": n_parasitic_pass,
        # everything that is not a measured pass (fail / unmeasurable) is a fail
        "n_parasitic_fail": len(parasitic) - n_parasitic_pass,
        "n_placement_pass": n_placement_pass,
        "n_placement_fail": len(placement) - n_placement_pass,
    }


def record_is_serializable(record: Dict[str, object]) -> bool:
    """True iff ``record`` round-trips through JSON unchanged.

    Stronger than a bare ``json.dumps`` success check: it confirms no tuple was
    silently coerced to a list in a way that changes equality, by re-loading the
    serialized form and comparing it to the original dict (which, if the record is
    built correctly, already contains only JSON-native types).
    """
    try:
        text = json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        return False
    return json.loads(text) == record


def write_full_record(record: Dict[str, object], path: str) -> None:
    """Write ``record`` to ``path`` as pretty-printed UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


__all__ = [
    "SCHEMA",
    "build_full_signoff_record",
    "full_benchmark_row",
    "record_is_serializable",
    "write_full_record",
]
