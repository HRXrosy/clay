"""Fail-closed evidence validation for compact structural-intent search."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from clay.layout.compact_intent import CompactIntentMetrics


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CompactIntentGateError(ValueError):
    """A candidate lacks coherent physical or POST-PEX truth evidence."""


def _mapping(row: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise CompactIntentGateError(f"{key} gate evidence is missing")
    return value


def _require_pass(row: Mapping[str, object], gate: str) -> Mapping[str, object]:
    evidence = _mapping(row, gate)
    if evidence.get("status") != "pass":
        raise CompactIntentGateError(
            f"{gate} gate is not pass: {evidence.get('status')!r}"
        )
    return evidence


def _sha(value: object, *, label: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256.fullmatch(normalized):
        raise CompactIntentGateError(f"{label} is not a SHA-256 digest")
    return normalized


def _validate_body(body: Mapping[str, object]) -> None:
    for polarity in ("pmos", "nmos"):
        evidence = body.get(polarity)
        if not isinstance(evidence, Mapping):
            raise CompactIntentGateError(f"{polarity} body evidence is missing")
        expected = evidence.get("expected")
        matched = evidence.get("matched")
        expected_net = str(evidence.get("expected_net", ""))
        tie_net = str(evidence.get("tie_net", ""))
        if (
            type(expected) is not int
            or expected <= 0
            or type(matched) is not int
            or matched != expected
            or not expected_net
            or tie_net != expected_net
        ):
            raise CompactIntentGateError(
                f"{polarity} body obligation failed: expected={expected!r}, "
                f"matched={matched!r}, expected_net={expected_net!r}, "
                f"tie_net={tie_net!r}"
            )


def validate_signed_candidate(
    row: Mapping[str, object],
) -> CompactIntentMetrics:
    """Validate all hard gates and return the candidate's ranking metrics."""

    _sha(row.get("candidate_sha256"), label="candidate_sha256")
    exact = _require_pass(row, "exact_emit")
    drc = _require_pass(row, "drc")
    lvs = _require_pass(row, "lvs")
    body = _require_pass(row, "body")
    pex = _require_pass(row, "pex")
    post = _require_pass(row, "post_pex")

    if drc.get("violations") != 0:
        raise CompactIntentGateError(
            f"drc gate has non-zero violations: {drc.get('violations')!r}"
        )
    if lvs.get("matched") is not True:
        raise CompactIntentGateError("lvs gate did not report matched=true")
    if pex.get("extracted") is not True:
        raise CompactIntentGateError("pex gate did not report extracted=true")
    if post.get("all_constraints_passed") is not True:
        raise CompactIntentGateError(
            "post_pex gate did not report all_constraints_passed=true"
        )
    _validate_body(body)

    exact_sha = _sha(exact.get("gds_sha256"), label="exact_emit GDS SHA-256")
    bound_hashes = {
        "drc": _sha(drc.get("gds_sha256"), label="drc GDS SHA-256"),
        "lvs": _sha(lvs.get("gds_sha256"), label="lvs GDS SHA-256"),
        "lvs oracle": _sha(
            lvs.get("oracle_gds_sha256"), label="lvs oracle GDS SHA-256"
        ),
        "pex": _sha(pex.get("gds_sha256"), label="pex GDS SHA-256"),
        "post_pex": _sha(
            post.get("gds_sha256"), label="post_pex GDS SHA-256"
        ),
    }
    disagreements = {
        label: digest for label, digest in bound_hashes.items() if digest != exact_sha
    }
    if disagreements:
        raise CompactIntentGateError(
            f"GDS SHA-256 disagreement: exact={exact_sha}, others={disagreements}"
        )
    pex_netlist = _sha(
        pex.get("netlist_sha256"), label="pex netlist SHA-256"
    )
    post_netlist = _sha(
        post.get("pex_netlist_sha256"), label="post_pex netlist SHA-256"
    )
    if pex_netlist != post_netlist:
        raise CompactIntentGateError(
            "POST-PEX simulation did not consume the signed PEX netlist SHA-256"
        )

    raw_metrics = _mapping(row, "metrics")
    try:
        metrics = CompactIntentMetrics(
            bbox_area_um2=float(raw_metrics["bbox_area_um2"]),
            whitespace_um2=float(raw_metrics["whitespace_um2"]),
            wire_length_um=float(raw_metrics["wire_length_um"]),
            via_count=raw_metrics["via_count"],  # type: ignore[arg-type]
            ring_perimeter_um=float(raw_metrics["ring_perimeter_um"]),
            post_pex_fom=float(raw_metrics["post_pex_fom"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CompactIntentGateError(f"invalid ranking metrics: {exc}") from exc
    post_fom = post.get("fom")
    if (
        not isinstance(post_fom, (int, float))
        or not math.isfinite(float(post_fom))
        or not math.isclose(
            float(post_fom), metrics.post_pex_fom, rel_tol=1e-12, abs_tol=1e-18
        )
    ):
        raise CompactIntentGateError(
            "ranking FOM disagrees with signed POST-PEX FOM"
        )
    return metrics


def write_compact_intent_ledger(
    path: Path | str,
    payload: Mapping[str, Any],
) -> Path:
    """Atomically write deterministic JSON after the whole search completes."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "CompactIntentGateError",
    "validate_signed_candidate",
    "write_compact_intent_ledger",
]
