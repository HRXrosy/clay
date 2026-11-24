"""Machine-readable SIGNOFF RECORD for the layout-intent contract.

The layout-intent contract already produces two human-facing artifacts: a text
report (``clay.layout.intent_certify.format_layout_intent_report``) and a signed
or withheld proof token (``certify_layout_intent``). Neither is convenient to
diff, archive, or feed into a quality-gate flywheel / cross-framework benchmark
(e.g. Clay layout vs a TED one). This module assembles a single, fully
JSON-serializable record from the SAME certify + score pipeline so a signoff can
be persisted, compared, and re-loaded without re-running Magic.

The record folds together three sources, each already the authoritative one:

  * ``certify_layout_intent`` -- runs the obligation check AND the proof-token
    gate, returning the signoff-shaped rows, the raw obligations, the verdict,
    the blocking budget kinds, and the signed-or-withheld token. The record's
    obligation rows and token both come from here; nothing is re-derived.
  * ``layout_intent_score`` -- the normalized quality scalar in ``[0, 1]`` plus
    its pass fraction / weakest obligation breakdown.
  * the ``PexReport`` -- the extraction provenance (backend, calibrated flag,
    element + net counts, whether a real tool ran).

Token honesty: the digest is pulled from the SIGNED token's ``trace_digest``
field. When the token is withheld (any required parasitic budget failed), the
certify call returns ``token is None`` and this module reports ``signed=False``
with ``digest=None``. A digest is NEVER fabricated -- a withheld token has no
digest, exactly mirroring how DRC withholds a token on a violation.

Serializability: every value placed in the record is a JSON scalar, list, or
dict. The frozen dataclasses (``LayoutObligation``) and tuples (net ``pair``s,
``PexReport.nets()``) that the upstream layers traffic in are coerced to plain
floats / strings / lists here, so ``json.dumps(record)`` always succeeds and the
record round-trips byte-for-byte through ``json.loads``. All thresholds remain
RELATIVE fractions (project-wide rule: never absolute aF deltas).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from .intent import LayoutIntent
from .intent_certify import certify_layout_intent
from .intent_score import layout_intent_score
from .pex.base import PexReport

SCHEMA = "clay.layout_intent_signoff/v1"


def _obligation_entry(obligation) -> Dict[str, object]:
    """One JSON-serializable obligation entry from a :class:`LayoutObligation`.

    Coerces the frozen dataclass + its tuple ``pair`` into plain JSON types:
    the net pair becomes a 2-element list, the measured imbalance stays a float
    (or ``None`` when the relationship was never extracted), and the derived
    ``status`` / ``over_budget`` are taken straight off the obligation.
    """
    measured = obligation.measured_imbalance
    a, b = obligation.pair
    return {
        "kind": str(obligation.kind),
        "pair": [str(a), str(b)],
        "measured_imbalance": None if measured is None else float(measured),
        "rel_tol": float(obligation.rel_tol),
        "status": str(obligation.status),
        "over_budget": float(obligation.over_budget),
    }


def _score_entry(score: Dict[str, object]) -> Dict[str, object]:
    """The benchmark-relevant subset of the quality score, JSON-clean.

    ``weakest`` (which the full score carries) holds a tuple ``pair`` and so is
    intentionally NOT embedded raw; the record exposes only plain scalars from
    the score so the whole dict is trivially serializable.
    """
    return {
        "score": float(score["score"]),
        "pass_fraction": float(score["pass_fraction"]),
        "worst_margin": (
            None if score["worst_margin"] is None else float(score["worst_margin"])
        ),
        "weakest": (
            None
            if score["weakest"] is None
            else _weakest_entry(score["weakest"])  # type: ignore[arg-type]
        ),
        "n_pass": int(score["n_pass"]),
        "n_fail": int(score["n_fail"]),
        "n_unmeasurable": int(score["n_unmeasurable"]),
    }


def _weakest_entry(weakest: Dict[str, object]) -> Dict[str, object]:
    """JSON-clean view of the score's ``weakest`` obligation (its ``pair`` is a tuple)."""
    a, b = weakest["pair"]  # type: ignore[misc]
    out: Dict[str, object] = {
        "kind": str(weakest["kind"]),
        "pair": [str(a), str(b)],
        "label": str(weakest["label"]),
        "status": str(weakest["status"]),
        "quality": float(weakest["quality"]),  # type: ignore[arg-type]
    }
    # The score attaches exactly one of headroom (pass) / over_budget (fail).
    if "headroom" in weakest and weakest["headroom"] is not None:
        out["headroom"] = float(weakest["headroom"])  # type: ignore[arg-type]
    if "over_budget" in weakest and weakest["over_budget"] is not None:
        out["over_budget"] = float(weakest["over_budget"])  # type: ignore[arg-type]
    return out


def _token_entry(token: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Token sub-record: signed digest pulled from the token, or withheld -> None.

    The digest is the SIGNED token's ``trace_digest``. A withheld token is
    ``None`` (a required budget failed), in which case ``signed`` is False,
    ``digest`` is None, and ``obligation_ids`` is empty. A digest is never
    fabricated.
    """
    if token is None:
        return {"signed": False, "digest": None, "obligation_ids": []}
    digest = token.get("trace_digest")
    return {
        "signed": True,
        "digest": None if digest is None else str(digest),
        "obligation_ids": [str(i) for i in token.get("obligation_ids", [])],
    }


def _provenance_entry(pex_report: PexReport) -> Dict[str, object]:
    """Extraction provenance from the report: backend / calibration / sizes."""
    return {
        "backend": str(pex_report.backend),
        "calibrated": bool(pex_report.calibrated),
        "n_caps": len(pex_report.caps),
        "n_nets": len(pex_report.nets()),
        # A real extraction tool ran iff the report claims calibration (a
        # geometric estimate or a synthetic report is uncalibrated).
        "ran_real_tool": bool(pex_report.calibrated),
    }


def build_signoff_record(
    cell,
    pex_report: PexReport,
    *,
    pdk: str = "gf180mcu",
    intent: Optional[LayoutIntent] = None,
    **tol,
) -> Dict[str, object]:
    """Assemble the machine-readable layout-intent signoff record.

    Runs the SAME certify pipeline as the human report
    (:func:`clay.layout.intent_certify.certify_layout_intent`, which itself runs
    the obligation check and the proof-token gate) plus the normalized quality
    score (:func:`clay.layout.intent_score.layout_intent_score`), then folds the
    results into one fully JSON-serializable dict.

    The token digest is the SIGNED token's ``trace_digest``; when the token is
    withheld (a required budget failed) the record reports ``signed=False`` /
    ``digest=None`` and the blocking kind(s) -- a digest is never fabricated.

    All ``**tol`` keyword tolerances are forwarded to the deriver (RELATIVE
    fractions only). Returns a dict matching ``clay.layout_intent_signoff/v1``.
    """
    certified = certify_layout_intent(cell, pex_report, intent=intent, **tol)
    obligations = certified["obligations"]
    score = layout_intent_score(obligations)  # type: ignore[arg-type]

    return {
        "schema": SCHEMA,
        "cell": str(certified["cell"]),
        "pdk": str(pdk),
        "status": str(certified["status"]),
        "passed": bool(certified["passed"]),
        "blocking": [str(k) for k in certified["blocking"]],  # type: ignore[union-attr]
        "obligations": [_obligation_entry(o) for o in obligations],  # type: ignore[union-attr]
        "score": _score_entry(score),
        "token": _token_entry(certified["token"]),  # type: ignore[arg-type]
        "provenance": _provenance_entry(pex_report),
        # v2: bind the record to the EXACT intent that was certified, so a
        # benchmark / a diff / a flywheel comparison can tell that two records
        # for the "same cell" were actually certifying different intents
        # (different aggressors, different clock_net, different declared
        # overrides). The digest is a stable 16-hex sha256 of the intent's
        # user-visible contents (see ``LayoutIntent.digest``); ``intent_source``
        # / ``intent_profile_id`` carry the provenance label.
        "intent_digest": str(certified["intent_digest"]),
        "intent_source": str(certified["intent_source"]),
        "intent_profile_id": (
            None if certified["intent_profile_id"] is None
            else str(certified["intent_profile_id"])
        ),
    }


def record_is_serializable(record: Dict[str, object]) -> bool:
    """True iff ``record`` round-trips through JSON unchanged.

    Stronger than a bare ``json.dumps`` success check: it confirms no tuple was
    silently coerced to a list in a way that changes equality, by re-loading the
    serialized form and comparing it to the original dict (which, if the record
    is built correctly, already contains only JSON-native types).
    """
    try:
        text = json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        return False
    return json.loads(text) == record


def write_signoff_record(record: Dict[str, object], path: str) -> None:
    """Write ``record`` to ``path`` as pretty-printed UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


__all__ = [
    "SCHEMA",
    "build_signoff_record",
    "record_is_serializable",
    "write_signoff_record",
]
