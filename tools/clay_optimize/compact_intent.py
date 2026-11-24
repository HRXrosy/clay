"""Deterministic bounded search for compact structural Clay intent."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from clay.flow.compact_intent_ledger import (
    CompactIntentGateError,
    validate_signed_candidate,
    write_compact_intent_ledger,
)
from clay.layout.compact_intent import (
    CompactIntentCandidate,
    CompactIntentMetrics,
    rank_signed_candidates,
)


CheapEvaluator = Callable[
    [CompactIntentCandidate], Mapping[str, float | int | str]
]
SignoffEvaluator = Callable[[CompactIntentCandidate], Mapping[str, object]]


class CandidateSelector(Protocol):
    def __call__(
        self,
        candidates: tuple[CompactIntentCandidate, ...],
        cheap_metrics: Mapping[str, Mapping[str, float | int | str]],
        *,
        maximum: int,
    ) -> tuple[CompactIntentCandidate, ...]: ...


@dataclass(frozen=True)
class CompactIntentSearchResult:
    winner: CompactIntentCandidate | None
    ledger_path: Path
    signed_count: int
    optimality: str


_CHEAP_KEYS = (
    "bbox_area_um2",
    "whitespace_um2",
    "wire_length_um",
    "via_count",
    "ring_perimeter_um",
)

_PHYSICAL_SIGNATURE_KEY = "physical_signature_sha256"
_OPTIONAL_PHYSICAL_KEYS = (
    "ring_max_perimeter_um",
    "ring_span_waste_um2",
)


def _validate_cheap_metrics(
    raw: Mapping[str, float | int | str],
) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    for key in _CHEAP_KEYS:
        if key not in raw:
            raise ValueError(f"cheap evaluation missing {key}")
        value = raw[key]
        if key == "via_count":
            if type(value) is not int or value < 0:
                raise ValueError("cheap via_count must be a non-negative integer")
            out[key] = value
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"cheap {key} must be finite and non-negative")
        out[key] = number
    signature = raw.get(_PHYSICAL_SIGNATURE_KEY)
    if signature is not None:
        if (
            type(signature) is not str
            or len(signature) != 64
            or any(char not in "0123456789abcdef" for char in signature)
        ):
            raise ValueError(
                "cheap physical_signature_sha256 must be one canonical SHA-256"
            )
        out[_PHYSICAL_SIGNATURE_KEY] = signature
    for key in _OPTIONAL_PHYSICAL_KEYS:
        if key not in raw:
            continue
        number = float(raw[key])
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"cheap {key} must be finite and non-negative")
        out[key] = number
    return out


def _dominates(
    left: Mapping[str, float | int | str],
    right: Mapping[str, float | int | str],
) -> bool:
    """Cheap frontier is area versus route length; signoff ranking is stricter."""

    keys = ("bbox_area_um2", "wire_length_um")
    return all(float(left[key]) <= float(right[key]) for key in keys) and any(
        float(left[key]) < float(right[key]) for key in keys
    )


def _domain_digest(domain: Iterable[CompactIntentCandidate]) -> str:
    payload = [candidate.canonical_data() for candidate in sorted(domain, key=lambda c: c.digest)]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cheap_rank_key(
    candidate: CompactIntentCandidate,
    cheap_pass: Mapping[str, Mapping[str, float | int | str]],
) -> tuple[float | str, ...]:
    metrics = cheap_pass[candidate.digest]
    return (
        *(float(metrics[key]) for key in _CHEAP_KEYS),
        candidate.digest,
    )


def _pareto_layers(
    ordered: tuple[CompactIntentCandidate, ...],
    cheap_pass: Mapping[str, Mapping[str, float | int | str]],
) -> tuple[tuple[CompactIntentCandidate, ...], ...]:
    """Peel deterministic area/wire frontiers until every cheap pass is ranked."""

    remaining = {candidate.digest for candidate in ordered if candidate.digest in cheap_pass}
    layers: list[tuple[CompactIntentCandidate, ...]] = []
    while remaining:
        layer_digests = {
            digest
            for digest in remaining
            if not any(
                other_digest != digest
                and _dominates(cheap_pass[other_digest], cheap_pass[digest])
                for other_digest in remaining
            )
        }
        if not layer_digests:
            raise RuntimeError("compact intent Pareto peeling made no progress")
        layer = tuple(
            sorted(
                (candidate for candidate in ordered if candidate.digest in layer_digests),
                key=lambda candidate: _cheap_rank_key(candidate, cheap_pass),
            )
        )
        layers.append(layer)
        remaining.difference_update(layer_digests)
    return tuple(layers)


def run_compact_intent_search(
    *,
    domain: Iterable[CompactIntentCandidate],
    cheap_evaluator: CheapEvaluator,
    signoff_evaluator: SignoffEvaluator,
    ledger_path: Path | str,
    source_sha256: str,
    commit: str,
    seed: int,
    domain_exhausted: bool = True,
    max_signoffs: int | None = None,
    candidate_selector: CandidateSelector | None = None,
) -> CompactIntentSearchResult:
    """Evaluate a bounded domain and write a complete, fail-closed ledger."""

    if max_signoffs is not None and max_signoffs <= 0:
        raise ValueError("max_signoffs must be positive when provided")
    ordered = tuple(sorted(set(domain), key=lambda candidate: candidate.digest))
    if not ordered:
        raise ValueError("compact intent search domain must be non-empty")
    rows: dict[str, dict[str, object]] = {}
    cheap_pass: dict[str, dict[str, float | int | str]] = {}
    for candidate in ordered:
        row: dict[str, object] = {
            "candidate": candidate.canonical_data(),
            "candidate_sha256": candidate.digest,
            "stage": "cheap",
        }
        try:
            metrics = _validate_cheap_metrics(cheap_evaluator(candidate))
        except Exception as exc:  # fail one candidate, not the whole bounded run
            row.update(
                {
                    "cheap": {"status": "fail"},
                    "failure": f"{type(exc).__name__}: {exc}",
                    "stage": "cheap_fail",
                }
            )
        else:
            row.update({"cheap": {"status": "pass", "metrics": metrics}})
            cheap_pass[candidate.digest] = metrics
        rows[candidate.digest] = row

    layers = _pareto_layers(ordered, cheap_pass)
    frontier_ordered = layers[0] if layers else ()
    signoff_ordered = tuple(candidate for layer in layers for candidate in layer)
    representatives: list[CompactIntentCandidate] = []
    representative_by_signature: dict[str, CompactIntentCandidate] = {}
    physical_aliases: dict[str, CompactIntentCandidate] = {}
    for candidate in signoff_ordered:
        signature = str(
            cheap_pass[candidate.digest].get(
                _PHYSICAL_SIGNATURE_KEY,
                candidate.digest,
            )
        )
        representative = representative_by_signature.get(signature)
        if representative is None:
            representative_by_signature[signature] = candidate
            representatives.append(candidate)
        else:
            physical_aliases[candidate.digest] = representative
    physical_unique_ordered = tuple(representatives)
    selection_limit = (
        len(physical_unique_ordered) if max_signoffs is None else max_signoffs
    )
    if candidate_selector is None:
        selected_for_signoff = physical_unique_ordered[:selection_limit]
    else:
        selected_for_signoff = tuple(
            candidate_selector(
                physical_unique_ordered,
                cheap_pass,
                maximum=selection_limit,
            )
        )
        selected_set = set(selected_for_signoff)
        if len(selected_set) != len(selected_for_signoff):
            raise ValueError("candidate selector returned duplicate candidates")
        if len(selected_for_signoff) > selection_limit:
            raise ValueError("candidate selector exceeded the signoff budget")
        invalid = [
            candidate.digest
            for candidate in selected_for_signoff
            if candidate not in physical_unique_ordered
        ]
        if invalid:
            raise ValueError(
                "candidate selector returned unmeasured or aliased candidates: "
                f"{invalid}"
            )
    selected_digests = {candidate.digest for candidate in selected_for_signoff}
    signoff_attempted = 0
    signed: list[tuple[CompactIntentCandidate, CompactIntentMetrics]] = []
    for candidate in ordered:
        if candidate.digest not in cheap_pass:
            continue
        row = rows[candidate.digest]
        alias_of = physical_aliases.get(candidate.digest)
        if alias_of is not None:
            row["stage"] = "physical_alias"
            row["physical_alias_of"] = alias_of.digest
            continue
        if candidate.digest not in selected_digests:
            row["stage"] = "signoff_budget"

    # Execute expensive truth evaluations in the explicit Pareto/ranking order.
    # Iterating the digest-sorted ledger rows here would make a harmless candidate
    # hash change reorder the physical experiments and defeat the selected cohort.
    for candidate in selected_for_signoff:
        row = rows[candidate.digest]
        signoff_attempted += 1
        try:
            signoff = dict(signoff_evaluator(candidate))
            metrics = validate_signed_candidate(signoff)
            if signoff.get("candidate_sha256") != candidate.digest:
                raise CompactIntentGateError(
                    "signoff row candidate SHA-256 disagrees with evaluated candidate"
                )
        except Exception as exc:
            row.update(
                {
                    "signoff": signoff if "signoff" in locals() else {},
                    "failure": f"{type(exc).__name__}: {exc}",
                    "stage": "signoff_fail",
                }
            )
        else:
            row.clear()
            row.update(signoff)
            row["stage"] = "signed"
            signed.append((candidate, metrics))
        finally:
            if "signoff" in locals():
                del signoff

    ranked = rank_signed_candidates(signed)
    winner = ranked[0][0] if ranked else None
    signoff_truncated = len(selected_for_signoff) < len(physical_unique_ordered)
    optimality = (
        "proven_within_domain"
        if domain_exhausted and not signoff_truncated
        else "best_found"
    )
    payload: dict[str, object] = {
        "schema": "clay.compact-intent-ledger/v1",
        "commit": str(commit),
        "source_sha256": str(source_sha256),
        "seed": int(seed),
        "domain_sha256": _domain_digest(ordered),
        "optimality": optimality,
        "winner_sha256": winner.digest if winner is not None else None,
        "counts": {
            "enumerated": len(ordered),
            "cheap_pass": len(cheap_pass),
            "physical_unique": len(physical_unique_ordered),
            "frontier": len(frontier_ordered),
            "signoff_attempted": signoff_attempted,
            "signed": len(signed),
        },
        "candidates": [rows[candidate.digest] for candidate in ordered],
    }
    written = write_compact_intent_ledger(ledger_path, payload)
    return CompactIntentSearchResult(
        winner=winner,
        ledger_path=written,
        signed_count=len(signed),
        optimality=optimality,
    )


__all__ = [
    "CheapEvaluator",
    "CandidateSelector",
    "CompactIntentSearchResult",
    "SignoffEvaluator",
    "run_compact_intent_search",
]
