"""Bridge between Clay's ngspice simulation pipeline and RunRecorder.

Wraps :func:`clay.verifier.verify` so each AC/DC/transient/op simulation
emits structured ``sim_started`` / ``sim_completed`` (or ``sim_failed``)
events into the run archive. Existing callers that do not pass a recorder
get the unmodified ``verify`` result back (pure pass-through).

Public API
----------

``verify_with_recorder(tb, *, recorder=None, work_dir=None, driver=None,
revision="sim", summary=None) -> VerifyResult``
    Thin wrapper around :func:`clay.verifier.verify`. When ``recorder`` is
    given, emits ``sim_started`` before the simulator runs and
    ``sim_completed`` after, or ``sim_failed`` if the underlying call
    raises (the exception is re-raised unchanged).

``record_simulation(recorder, *, kind, cell, measurements, summary=None,
revision="sim", extra_data=None) -> None``
    Lower-level helper for tests or non-verify paths. Emits a single
    ``sim_measure`` event carrying a measurements dict.

Event kinds emitted by this module
----------------------------------

* ``sim_started`` — before the simulator runs; ``data`` carries cell,
  pdk_corner, temp, analysis_kind, and a stable tb fingerprint hash.
* ``sim_completed`` — after a successful run; ``data`` carries the
  ``passed`` boolean, ``measurements`` dict, ``runtime_s`` wall-clock,
  ``log_path``, and ``netlist_path`` (when present).
* ``sim_failed`` — emitted in lieu of ``sim_completed`` when the wrapped
  call raises; ``data`` carries the error class name plus the first line
  of the message.
* ``sim_measure`` — emitted only by :func:`record_simulation` for ad-hoc
  measurement recording outside the verify pipeline.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import is_dataclass, fields as _fields
from pathlib import Path
from typing import Any, Mapping

from clay.runtime.recorder import RunRecorder


__all__ = ["verify_with_recorder", "record_simulation"]


def verify_with_recorder(
    tb: Any,
    *,
    recorder: RunRecorder | None = None,
    work_dir: Path | str | None = None,
    driver: Any | None = None,
    revision: str = "sim",
    summary: str | None = None,
) -> Any:
    """Run :func:`clay.verifier.verify` with optional RunRecorder events.

    When ``recorder is None`` this is a pure pass-through: the underlying
    ``verify(tb, work_dir=..., driver=...)`` is invoked and its result
    returned unchanged.

    When ``recorder`` is provided, two structured events bracket the
    simulation:

    * ``sim_started`` — emitted before the simulator runs.
    * ``sim_completed`` — emitted after a successful run.

    If the wrapped ``verify`` call raises, a ``sim_failed`` event is
    emitted and the original exception is re-raised unchanged.
    """
    # Lazy import keeps this module testable without dragging in the
    # full verifier dependency graph at import time.
    from clay.verifier import verify

    verify_kwargs: dict[str, Any] = {}
    if work_dir is not None:
        verify_kwargs["work_dir"] = work_dir
    if driver is not None:
        verify_kwargs["driver"] = driver

    if recorder is None:
        return verify(tb, **verify_kwargs)

    cell_name = _cell_name(tb)
    analysis_kind = _analysis_kind(tb)
    started_data = {
        "cell": cell_name,
        "pdk_corner": getattr(tb, "pdk_corner", None),
        "temp": getattr(tb, "temp", None),
        "analysis_kind": analysis_kind,
        "tb_fingerprint": _tb_fingerprint(tb),
    }
    started_summary = summary or (
        f"sim started: cell={cell_name} analysis={analysis_kind}"
    )
    recorder.event(
        "sim_started",
        summary=started_summary,
        revision=revision,
        data=started_data,
    )

    t0 = time.monotonic()
    try:
        result = verify(tb, **verify_kwargs)
    except Exception as exc:  # noqa: BLE001 — we re-raise after recording.
        runtime_s = time.monotonic() - t0
        recorder.event(
            "sim_failed",
            summary=(
                f"sim failed: cell={cell_name} "
                f"error={type(exc).__name__}: {_first_line(str(exc))}"
            ),
            revision=revision,
            data={
                "cell": cell_name,
                "analysis_kind": analysis_kind,
                "error_class": type(exc).__name__,
                "error_message": _first_line(str(exc)),
                "runtime_s": runtime_s,
            },
        )
        raise

    runtime_s = time.monotonic() - t0

    measurements = _measurements_dict(result)
    passed = _result_passed(result)
    log_path = getattr(result, "log_path", "") or ""
    netlist_path = getattr(result, "netlist_path", "") or ""

    artifacts: dict[str, str] | None = None
    saved_log_path = log_path
    if log_path:
        try:
            src = Path(log_path)
            if src.exists() and src.is_file():
                safe_name = f"{cell_name}_{revision}_{src.name}"
                dest = recorder.copy_final_artifact(
                    f"sim_log:{cell_name}:{revision}",
                    src,
                    filename=safe_name,
                )
                rel = dest.relative_to(recorder.run_dir).as_posix()
                artifacts = {"sim_log": rel}
                saved_log_path = rel
        except OSError:
            # Best-effort artifact copy; do not fail the simulation
            # recording because the log was unreadable.
            artifacts = None

    completed_summary = summary or (
        f"sim completed: cell={cell_name} analysis={analysis_kind} "
        f"passed={passed}"
    )
    recorder.event(
        "sim_completed",
        summary=completed_summary,
        revision=revision,
        artifacts=artifacts,
        data={
            "cell": cell_name,
            "analysis_kind": analysis_kind,
            "passed": passed,
            "measurements": measurements,
            "runtime_s": runtime_s,
            "log_path": saved_log_path,
            "netlist_path": netlist_path,
        },
    )
    return result


def record_simulation(
    recorder: RunRecorder,
    *,
    kind: str,
    cell: str,
    measurements: Mapping[str, Any],
    summary: str | None = None,
    revision: str = "sim",
    extra_data: Mapping[str, Any] | None = None,
) -> None:
    """Emit a single ``sim_measure`` event with a measurements dict.

    The ``kind`` argument identifies the analysis category (``ac``,
    ``dc``, ``tran``, ``op``, etc.) and is stored under ``data.kind``;
    the event ``kind`` itself is always ``sim_measure`` so viewers have
    a stable channel to subscribe to.
    """
    if recorder is None:
        raise ValueError("record_simulation requires a non-None recorder")
    data: dict[str, Any] = {
        "kind": kind,
        "cell": cell,
        "measurements": dict(measurements),
    }
    if extra_data:
        for key, value in extra_data.items():
            if key in data:
                # Caller-supplied extras override defaults so a test
                # can inject a deterministic cell or kind label.
                data[key] = value
            else:
                data[key] = value
    recorder.event(
        "sim_measure",
        summary=summary or f"measurements recorded: cell={cell} kind={kind}",
        revision=revision,
        data=data,
    )


# ─── helpers ────────────────────────────────────────────────────


def _cell_name(tb: Any) -> str:
    cell = getattr(tb, "cell", None)
    name = getattr(cell, "name", None)
    if isinstance(name, str) and name:
        return name
    return "<unknown>"


def _analysis_kind(tb: Any) -> str:
    """Concise label like ``ac``, ``ac+tran``, or ``op``."""
    analyses = getattr(tb, "analyses", None) or []
    kinds: list[str] = []
    for an in analyses:
        cls = type(an).__name__
        if cls == "AcAnalysis":
            kinds.append("ac")
        elif cls == "DcAnalysis":
            kinds.append("dc")
        elif cls == "TranAnalysis":
            kinds.append("tran")
        elif cls == "OpAnalysis":
            kinds.append("op")
        else:
            kinds.append(cls.lower().replace("analysis", "") or cls.lower())
    if not kinds:
        return "none"
    # Preserve order, but de-duplicate so ``ac+ac`` collapses to ``ac``.
    seen: list[str] = []
    for k in kinds:
        if k not in seen:
            seen.append(k)
    return "+".join(seen)


def _tb_fingerprint(tb: Any) -> str:
    """Stable short hash of the testbench shape.

    Captures cell name, port list, pdk corner, temp, supply, analysis
    kinds, and measure names. Not a cryptographic identity — just a
    short token so the viewer can correlate ``sim_started`` to
    ``sim_completed`` across replays.
    """
    parts: list[str] = []
    parts.append(_cell_name(tb))
    cell = getattr(tb, "cell", None)
    ports = getattr(cell, "ports", None) or []
    parts.append(",".join(str(p) for p in ports))
    parts.append(str(getattr(tb, "pdk_corner", "")))
    parts.append(str(getattr(tb, "temp", "")))
    parts.append(str(getattr(tb, "supply_volt", "")))
    parts.append(_analysis_kind(tb))
    measures = getattr(tb, "measures", None) or []
    parts.append(",".join(getattr(m, "name", "") for m in measures))
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _measurements_dict(result: Any) -> dict[str, Any]:
    measures = getattr(result, "measures", None)
    if measures is None:
        return {}
    try:
        return {str(k): _coerce_scalar(v) for k, v in measures.items()}
    except AttributeError:
        return {}


def _coerce_scalar(value: Any) -> Any:
    """JSON-friendly scalar coercion (floats stay floats, NaN -> string)."""
    if isinstance(value, float):
        # JSON cannot encode NaN/Inf in strict mode; keep the event log
        # archivable across viewers without surprises.
        if value != value:  # NaN
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return repr(value)


def _result_passed(result: Any) -> bool:
    """Best-effort pass extraction, tolerant of synthetic results."""
    all_passed = getattr(result, "all_passed", None)
    if isinstance(all_passed, bool):
        return all_passed
    if callable(all_passed):
        try:
            return bool(all_passed())
        except Exception:  # noqa: BLE001
            pass
    sim_error = getattr(result, "simulator_error", None)
    if sim_error:
        return False
    assertions = getattr(result, "assertions", None) or []
    if not assertions:
        # No assertions present (e.g. op-only sweep): treat a clean run
        # without a simulator_error as a pass so downstream views don't
        # mis-label a successful op-point sim as failed.
        return True
    try:
        return all(bool(getattr(a, "passed", False)) for a in assertions)
    except TypeError:
        return False


def _first_line(message: str) -> str:
    if not message:
        return ""
    return message.splitlines()[0].strip()
