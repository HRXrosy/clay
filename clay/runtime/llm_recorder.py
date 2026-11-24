"""LLM-driven design-revision recording helpers for Clay run archives.

This module wires the LLM-authoring loop into :class:`clay.runtime.recorder.RunRecorder`
so the schematic_viewer (and any other downstream consumer of the run archive)
can replay or live-monitor each LLM suggestion, acceptance, rejection, or skip
in a uniform schema.

Event kinds emitted (all carried by the existing ``RunRecorder.event`` API):

* ``llm_proposed``    — a *batch* of candidate revisions before one is picked.
* ``llm_accepted``    — the loop picked one of the proposed candidates.
* ``llm_rejected``    — the loop rejected a candidate with a textual reason.
* ``llm_revision``    — a single (prompt, response, suggested_action) suggestion;
                        emitted by :func:`record_llm_revision` and automatically
                        by :func:`with_llm_call`.
* ``llm_skipped``     — the LLM step was skipped (e.g. no ``DEEPSEEK_API_KEY``);
                        recorded so a downstream replay can still see the gap.

Hashing convention
------------------

Every prompt and response is fingerprinted with ``sha256`` and truncated to the
first 12 hex characters.  That keeps the event small (the full text is *not*
embedded - we only keep a 240-character ``*_excerpt`` for the viewer) while
still giving an auditor a stable handle to cross-reference against the source
prompt log that ``tools.llm_authoring.real_api_run`` writes.

All helpers tolerate ``recorder=None`` so production code can call them in
contexts where archive recording is not configured (unit tests, dry-runs).

Test contract
-------------

The test surface in ``tests/test_llm_recorder.py`` deliberately mocks
``llm_call`` so no API key is required at unit-test time, mirroring the way
``clay.llm_counterfactual_trace`` and ``clay.distance_to_sat`` guard their
real-LLM paths with environment-variable checks.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable, Mapping, Sequence

from clay.runtime.recorder import RunRecorder


__all__ = [
    "EXCERPT_LENGTH",
    "HASH_LENGTH",
    "record_llm_revision",
    "record_llm_proposed",
    "record_llm_accepted",
    "record_llm_rejected",
    "record_llm_skipped",
    "with_llm_call",
]


# How many hex characters we keep from sha256 for prompt/response fingerprints.
HASH_LENGTH = 12

# How many leading characters we keep from prompt/response strings for the
# event excerpt fields.  Long enough to be human-recognizable in a viewer,
# short enough to keep the event payload modest.
EXCERPT_LENGTH = 240


def _short_hash(text: str) -> str:
    """Return the first ``HASH_LENGTH`` hex chars of sha256(text)."""

    if text is None:
        text = ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:HASH_LENGTH]


def _excerpt(text: str) -> str:
    """Return the first ``EXCERPT_LENGTH`` characters of ``text``."""

    if text is None:
        return ""
    return text[:EXCERPT_LENGTH]


def _candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Project a single revision candidate down to the viewer-relevant fields."""

    summary: dict[str, Any] = {}
    if "kind" in candidate:
        summary["kind"] = candidate["kind"]
    # ``target`` is the device/net the revision touches; ``device`` is the
    # equivalent key used elsewhere in Clay -- accept either spelling so the
    # caller does not have to translate.
    if "target" in candidate:
        summary["target"] = candidate["target"]
    elif "device" in candidate:
        summary["target"] = candidate["device"]
    if "delta" in candidate:
        summary["delta"] = candidate["delta"]
    return summary


def record_llm_revision(
    recorder: RunRecorder | None,
    *,
    prompt: str,
    response: str,
    suggested_action: Mapping[str, Any],
    revision: str,
    summary: str | None = None,
    confidence: float | None = None,
    tokens: Mapping[str, int] | None = None,
    model: str | None = None,
    extra_data: Mapping[str, Any] | None = None,
) -> None:
    """Emit a single ``llm_revision`` event.

    Parameters
    ----------
    recorder:
        Target archive writer.  ``None`` is a graceful no-op.
    prompt, response:
        Full prompt and response strings.  Only their sha256-12 fingerprint
        and a 240-character excerpt are embedded in the event payload.
    suggested_action:
        The LLM's structured revision suggestion (for example
        ``{"kind": "resize", "device": "M1", "new_W": "12u"}``).
    revision:
        The Clay revision label this suggestion belongs to (e.g. ``"r2"``).
    summary:
        Human-readable description for the viewer event log.
    confidence:
        Optional LLM-reported confidence in ``[0, 1]``.
    tokens:
        Optional token-usage dict, typically
        ``{"prompt": int, "completion": int}``.
    model:
        Optional model identifier (e.g. ``"deepseek-v4-pro"``).
    extra_data:
        Optional caller-supplied key/value pairs that are merged into the
        event ``data`` dict after the structured fields.
    """

    if recorder is None:
        return

    data: dict[str, Any] = {
        "model": model,
        "prompt_hash": _short_hash(prompt),
        "response_hash": _short_hash(response),
        "suggested_action": dict(suggested_action) if suggested_action is not None else {},
        "confidence": confidence,
        "tokens": dict(tokens) if tokens is not None else None,
        "prompt_excerpt": _excerpt(prompt),
        "response_excerpt": _excerpt(response),
    }
    if extra_data:
        for key, value in extra_data.items():
            data[key] = value

    recorder.event(
        "llm_revision",
        summary=summary or "LLM proposed a revision",
        revision=revision,
        data=data,
    )


def record_llm_proposed(
    recorder: RunRecorder | None,
    *,
    prompt: str,
    candidates: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    revision: str,
    summary: str | None = None,
) -> None:
    """Emit one ``llm_proposed`` event describing a *batch* of candidates.

    The event records ``n_candidates``, the prompt fingerprint, and a
    compact summary list ``candidate_summaries`` so the viewer can show
    the set without re-embedding every full revision payload.
    """

    if recorder is None:
        return

    candidate_list = list(candidates)
    candidate_summaries = [_candidate_summary(c) for c in candidate_list]

    data: dict[str, Any] = {
        "n_candidates": len(candidate_list),
        "prompt_hash": _short_hash(prompt),
        "prompt_excerpt": _excerpt(prompt),
        "candidate_summaries": candidate_summaries,
    }
    recorder.event(
        "llm_proposed",
        summary=summary or f"LLM proposed {len(candidate_list)} candidate revisions",
        revision=revision,
        data=data,
    )


def record_llm_accepted(
    recorder: RunRecorder | None,
    *,
    choice_index: int,
    candidate: Mapping[str, Any],
    revision: str,
    summary: str | None = None,
) -> None:
    """Emit one ``llm_accepted`` event recording which proposal was chosen."""

    if recorder is None:
        return

    data: dict[str, Any] = {
        "choice_index": choice_index,
        "candidate": dict(candidate) if candidate is not None else {},
    }
    recorder.event(
        "llm_accepted",
        summary=summary or f"Accepted LLM candidate #{choice_index}",
        revision=revision,
        data=data,
    )


def record_llm_rejected(
    recorder: RunRecorder | None,
    *,
    choice_index: int,
    candidate: Mapping[str, Any],
    reason: str,
    revision: str,
    summary: str | None = None,
) -> None:
    """Emit one ``llm_rejected`` event recording an explicit rejection."""

    if recorder is None:
        return

    data: dict[str, Any] = {
        "choice_index": choice_index,
        "candidate": dict(candidate) if candidate is not None else {},
        "reason": reason,
    }
    recorder.event(
        "llm_rejected",
        summary=summary or f"Rejected LLM candidate #{choice_index}: {reason}",
        revision=revision,
        data=data,
    )


def record_llm_skipped(
    recorder: RunRecorder | None,
    *,
    reason: str,
    revision: str,
    summary: str | None = None,
) -> None:
    """Emit one ``llm_skipped`` event (e.g. no API key) and continue.

    The function never raises -- a missing recorder still returns cleanly,
    matching the contract used elsewhere in the runtime helpers.
    """

    if recorder is None:
        return

    recorder.event(
        "llm_skipped",
        summary=summary or f"LLM step skipped: {reason}",
        revision=revision,
        data={"reason": reason},
    )


def with_llm_call(
    recorder: RunRecorder | None,
    *,
    model: str,
    prompt: str,
    llm_call: Callable[[str], Any],
    revision: str,
    summary: str | None = None,
) -> Any:
    """Invoke ``llm_call(prompt)`` and emit a matching ``llm_revision`` event.

    The ``llm_call`` callable is expected to return either a plain string or
    a mapping with at least one of the fields ``text``/``response`` and
    optionally ``suggested_action``, ``confidence``, ``tokens``.  The wrapper
    normalises both shapes before recording, so a test can supply a simple
    string-returning mock without conforming to the full schema.

    Returns whatever ``llm_call`` returned, unchanged, so callers can keep
    threading the original response through their pipeline.
    """

    raw = llm_call(prompt)

    if isinstance(raw, Mapping):
        response_text = str(
            raw.get("text") or raw.get("response") or raw.get("content") or ""
        )
        suggested_action = raw.get("suggested_action") or {}
        confidence = raw.get("confidence")
        tokens = raw.get("tokens")
    else:
        response_text = str(raw) if raw is not None else ""
        suggested_action = {}
        confidence = None
        tokens = None

    record_llm_revision(
        recorder,
        prompt=prompt,
        response=response_text,
        suggested_action=suggested_action,
        revision=revision,
        summary=summary,
        confidence=confidence,
        tokens=tokens,
        model=model,
    )

    return raw
