"""Per-invocation timing, cooperative budgets, and durable phase events.

Sessions have one execution owner thread; only their cancellation token may be
used from another thread. Activation may be nested or repeated and always
continues the same invocation, whose wall time starts at construction. CPU time
is telemetry, while deterministic budget usage comes only from explicit charges.
No session data belongs in a physical certificate or tool-independent cache key.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from clay.flow.execution_budget import (
    DETERMINISTIC_BUDGET_EXCEEDED,
    EXECUTION_CANCELLED,
    WALL_BUDGET_EXCEEDED,
    CancellationToken,
    ExecutionBudget,
    ExecutionStopped,
    _validate_seconds,
)


DEFAULT_EVENT_MAX_BYTES = 64 * 1024
DEFAULT_JOURNAL_MAX_BYTES = 8 * 1024 * 1024
_DETAIL_MAX_NODES = 2048
_DETAIL_MAX_DEPTH = 16


class JournalLimitExceeded(RuntimeError):
    """A new phase cannot own durable entry and terminal observations."""

    def __init__(self, *, limit_bytes: int, written_bytes: int, phase: str):
        self.limit_bytes, self.written_bytes, self.phase = limit_bytes, written_bytes, phase
        super().__init__("DIAGNOSTIC_BUDGET_EXCEEDED: event journal admission")

    def flow_error_data(self) -> dict[str, Any]:
        return {"reason": "DIAGNOSTIC_BUDGET_EXCEEDED", "scope": "event_journal",
                "limit_bytes": self.limit_bytes, "written_bytes": self.written_bytes,
                "phase": self.phase}


class _DetailLimit(ValueError):
    pass


def _bounded_snapshot(value: Mapping[str, Any], max_bytes: int) -> dict[str, Any]:
    """Copy only JSON values, bounding traversal before allocating/encoding them."""
    if not isinstance(value, Mapping):
        raise TypeError("details must be a mapping of JSON values")
    remaining, nodes = max_bytes, _DETAIL_MAX_NODES
    int_conversion_limit = sys.get_int_max_str_digits()
    ancestors: set[int] = set()

    def consume(count: int) -> None:
        nonlocal remaining
        remaining -= count
        if remaining < 0:
            raise _DetailLimit("detail byte limit")

    def walk(item: Any, depth: int = 0) -> Any:
        nonlocal nodes
        nodes -= 1
        if nodes < 0 or depth > _DETAIL_MAX_DEPTH:
            raise _DetailLimit("detail node/depth limit")
        if item is None or type(item) in (bool, int, float, str):
            if type(item) is float and not math.isfinite(item):
                raise ValueError("details must contain finite JSON values")
            if type(item) is str and len(item) > remaining:
                raise _DetailLimit("detail string limit")
            if type(item) is int:
                digits = item.bit_length() * 30103 // 100000 + 1 + int(item < 0)
                if digits > remaining or (int_conversion_limit and digits > int_conversion_limit):
                    raise _DetailLimit("detail integer limit")
            # Scalar construction is bounded above before json can escape it.
            consume(len(json.dumps(item, allow_nan=False)))
            return item
        if not isinstance(item, (Mapping, list, tuple)):
            raise TypeError("details must contain JSON values")
        if id(item) in ancestors:
            raise ValueError("circular details")
        if len(item) > nodes:
            raise _DetailLimit("detail collection limit")
        ancestors.add(id(item))
        try:
            consume(2)
            if isinstance(item, Mapping):
                result = {}
                for key, child in item.items():
                    if type(key) not in (str, int, float, bool, type(None)):
                        raise TypeError("details keys must be JSON keys")
                    walked_key = walk(key, depth + 1)
                    key_text = walked_key if type(walked_key) is str else json.dumps(walked_key)
                    consume(3)
                    result[key_text] = walk(child, depth + 1)
                return result
            result = []
            for child in item:
                consume(1)
                result.append(walk(child, depth + 1))
            return result
        finally:
            ancestors.remove(id(item))

    return walk(value)


def _encode_event(record: Mapping[str, Any], maximum: int) -> bytes:
    """Never construct an encoded event larger than its configured allowance."""
    chunks, used = [], 1  # Include the JSONL newline.
    for chunk in json.JSONEncoder(allow_nan=False, sort_keys=True).iterencode(record):
        encoded = chunk.encode("utf-8")
        used += len(encoded)
        if used > maximum:
            raise _DetailLimit("encoded event byte limit")
        chunks.append(encoded)
    return b"".join(chunks) + b"\n"


_CURRENT_SESSION: ContextVar[ExecutionSession | None] = ContextVar(
    "clay_execution_session", default=None,
)


def current_session() -> ExecutionSession | None:
    """The active invocation in this context, or None for legacy callers."""
    return _CURRENT_SESSION.get()


@dataclass
class _PhaseHandle:
    session: ExecutionSession
    phase_id: int
    parent_phase_id: int | None
    name: str
    budget: ExecutionBudget
    wall_start: float
    cpu_start: float
    _details: dict[str, Any] = field(default_factory=dict)
    deterministic_seconds: float = 0.0
    details_omitted: int = 0
    updates_omitted: int = 0
    entered: bool = False

    def annotate(self, **details: Any) -> None:
        """Durably publish a detached snapshot before the next expensive call."""
        self.session._check_owner()
        if not any(phase is self for phase in self.session._phases):
            raise RuntimeError("phase annotations require an active phase")
        try:
            additions = _bounded_snapshot(details, self.session._detail_max_bytes)
            self._details = _bounded_snapshot(
                {**self._details, **additions}, self.session._detail_max_bytes,
            )
        except _DetailLimit:
            self.details_omitted += 1
        self.session._emit(self, "phase_update", "running")


class ExecutionSession:
    """An invocation's budget owner and exclusive JSONL event writer.

    Phases can be used without activation when the caller holds the session.
    ``activate()`` exposes it to cooperative downstream adapters and restores
    the previous context on every exit. Returning from activation does not close
    or reset the invocation. Each phase emits and fsyncs its enter event before
    checking admission; an already stopped invocation records a phase error
    without executing the phase body. Budgets are exhausted at their limit.
    """

    def __init__(
        self,
        *,
        project_id: str,
        run_id: str,
        invocation_id: str,
        event_path: Path,
        budget: ExecutionBudget = ExecutionBudget(),
        cancellation: CancellationToken | None = None,
        wall_clock: Callable[[], float] = time.monotonic,
        cpu_clock: Callable[[], float] = time.process_time,
        event_max_bytes: int = DEFAULT_EVENT_MAX_BYTES,
        journal_max_bytes: int = DEFAULT_JOURNAL_MAX_BYTES,
    ) -> None:
        if not isinstance(budget, ExecutionBudget):
            raise TypeError("budget must be an ExecutionBudget")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise TypeError("cancellation must be a CancellationToken")
        if type(event_max_bytes) is not int or event_max_bytes < 8192:
            raise ValueError("event_max_bytes must be an integer at least 8192")
        if type(journal_max_bytes) is not int or journal_max_bytes < 2 * event_max_bytes:
            raise ValueError("journal_max_bytes must hold at least two maximum events")
        for identifier in (project_id, run_id, invocation_id):
            if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
                raise ValueError("execution identifiers must contain 1..128 characters")
        self.project_id = project_id
        self.run_id = run_id
        self.invocation_id = invocation_id
        self.event_path = Path(event_path)
        self.budget = budget
        self.cancellation = cancellation if cancellation is not None else CancellationToken()
        self._wall_clock = wall_clock
        self._cpu_clock = cpu_clock
        self._wall_start = wall_clock()
        self._cpu_start = cpu_clock()
        self._owner_thread = threading.get_ident()
        self._deterministic_seconds = 0.0
        self._phases: list[_PhaseHandle] = []
        self._seq = 0
        self._phase_seq = 0
        self.event_max_bytes = event_max_bytes
        self.journal_max_bytes = journal_max_bytes
        self._detail_max_bytes = event_max_bytes // 4
        self._journal_bytes = 0
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation is the authority to write this invocation journal.
        # Each subsequent event opens and closes the file so readers see it now.
        with self.event_path.open("x", encoding="utf-8"):
            pass

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("ExecutionSession may only execute on its owner thread")

    @contextmanager
    def activate(self) -> Iterator[ExecutionSession]:
        self._check_owner()
        token = _CURRENT_SESSION.set(self)
        try:
            yield self
        finally:
            _CURRENT_SESSION.reset(token)

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        budget: ExecutionBudget | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> Iterator[_PhaseHandle]:
        self._check_owner()
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError("phase name must contain 1..128 characters")
        if budget is not None and not isinstance(budget, ExecutionBudget):
            raise TypeError("phase budget must be an ExecutionBudget")
        omitted = 0
        try:
            snapshot = _bounded_snapshot(details if details is not None else {}, self._detail_max_bytes)
        except _DetailLimit:
            snapshot, omitted = {}, 1
        self._phase_seq += 1
        phase = _PhaseHandle(
            session=self,
            phase_id=self._phase_seq,
            parent_phase_id=self._phases[-1].phase_id if self._phases else None,
            name=name,
            budget=budget if budget is not None else ExecutionBudget(),
            wall_start=self._wall_clock(),
            cpu_start=self._cpu_clock(),
            _details=snapshot,
            details_omitted=omitted,
        )
        self._phases.append(phase)
        try:
            self._emit(phase, "phase_enter", "running")
            try:
                self.checkpoint()
                yield phase
                self.checkpoint()
            except BaseException as exc:
                try:
                    # Avoid arbitrary __str__/repr traversal of a failed live world.
                    message = exc.args[0][:512] if exc.args and type(exc.args[0]) is str else "message omitted"
                    error = {"error_type": type(exc).__name__[:128], "error": message}
                    if isinstance(exc, ExecutionStopped):
                        error["reason"] = exc.reason
                        try:
                            error.update(_bounded_snapshot(exc._details, self._detail_max_bytes))
                        except _DetailLimit:
                            error["details_omitted"] = True
                    elif isinstance(exc, JournalLimitExceeded):
                        error.update(exc.flow_error_data())
                    self._emit(phase, "phase_error", "error", error=error)
                except BaseException as observation_error:
                    try:
                        exc.add_note("Execution phase error observation failed: " + type(observation_error).__name__)
                    except BaseException:
                        pass
                raise
            else:
                self._emit(phase, "phase_exit", "ok")
        finally:
            self._phases.pop()

    def _budget_states(self) -> Iterator[tuple[ExecutionBudget, float, float, _PhaseHandle | None]]:
        now = self._wall_clock()
        yield self.budget, now - self._wall_start, self._deterministic_seconds, None
        for phase in self._phases:
            yield phase.budget, now - phase.wall_start, phase.deterministic_seconds, phase

    def _stop(self, reason: str, phase: _PhaseHandle | None = None, **details: Any) -> ExecutionStopped:
        return ExecutionStopped(
            reason,
            project_id=self.project_id,
            run_id=self.run_id,
            invocation_id=self.invocation_id,
            scope="phase" if phase is not None else "invocation",
            phase=phase.name if phase is not None else None,
            phase_id=phase.phase_id if phase is not None else None,
            **details,
        )

    def checkpoint(self) -> None:
        """Cooperatively reject cancellation or any exhausted active budget."""
        self._check_owner()
        cancellation_reason = self.cancellation.reason
        if cancellation_reason is not None:
            raise self._stop(EXECUTION_CANCELLED, cancellation_reason=cancellation_reason)
        for budget, wall_elapsed, deterministic, phase in self._budget_states():
            if budget.wall_seconds is not None and wall_elapsed >= budget.wall_seconds:
                raise self._stop(
                    WALL_BUDGET_EXCEEDED, phase,
                    limit_seconds=budget.wall_seconds, consumed_seconds=wall_elapsed,
                )
            if budget.deterministic_seconds is not None and deterministic >= budget.deterministic_seconds:
                raise self._stop(
                    DETERMINISTIC_BUDGET_EXCEEDED, phase,
                    limit_seconds=budget.deterministic_seconds, consumed_seconds=deterministic,
                )

    def remaining_wall_seconds(self) -> float | None:
        """Minimum nonnegative wall allowance across all active budgets."""
        self._check_owner()
        remaining = [
            max(0.0, budget.wall_seconds - wall_elapsed)
            for budget, wall_elapsed, _, _ in self._budget_states()
            if budget.wall_seconds is not None
        ]
        return min(remaining) if remaining else None

    def remaining_deterministic_seconds(self) -> float | None:
        """Minimum nonnegative charged-work allowance across active budgets."""
        self._check_owner()
        remaining = [
            max(0.0, budget.deterministic_seconds - deterministic)
            for budget, _, deterministic, _ in self._budget_states()
            if budget.deterministic_seconds is not None
        ]
        return min(remaining) if remaining else None

    def charge_deterministic(self, seconds: float) -> None:
        """Charge work once to this invocation and each currently active phase."""
        self._check_owner()
        seconds = _validate_seconds(seconds, "deterministic charge", allow_zero=True)
        totals = [self._deterministic_seconds + seconds]
        totals.extend(phase.deterministic_seconds + seconds for phase in self._phases)
        if not all(math.isfinite(total) for total in totals):
            raise ValueError("deterministic charge totals must remain finite")
        self._deterministic_seconds = totals[0]
        for phase, total in zip(self._phases, totals[1:]):
            phase.deterministic_seconds = total
        self.checkpoint()

    def _emit(
        self, phase: _PhaseHandle, event: str, status: str,
        *, error: Mapping[str, Any] | None = None,
    ) -> None:
        wall_now, cpu_now = self._wall_clock(), self._cpu_clock()
        record = {
            "schema": "clay.execution-event/v1",
            "seq": self._seq + 1,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "invocation_id": self.invocation_id,
            "event": event,
            "phase_id": phase.phase_id,
            "parent_phase_id": phase.parent_phase_id,
            "phase": phase.name,
            "status": status,
            "wall_elapsed_seconds": wall_now - phase.wall_start,
            "cpu_elapsed_seconds": cpu_now - phase.cpu_start,
            "deterministic_seconds": phase.deterministic_seconds,
            "invocation_wall_elapsed_seconds": wall_now - self._wall_start,
            "invocation_cpu_elapsed_seconds": cpu_now - self._cpu_start,
            "invocation_deterministic_seconds": self._deterministic_seconds,
            "budget": phase.budget.to_dict(),
            "invocation_budget": self.budget.to_dict(),
            "details": phase._details,
        }
        if error is not None:
            record["error"] = error
        if phase.details_omitted or phase.updates_omitted:
            record["diagnostics"] = {"details_omitted": phase.details_omitted,
                "updates_omitted": phase.updates_omitted,
                "event_max_bytes": self.event_max_bytes,
                "journal_max_bytes": self.journal_max_bytes}
        try:
            line = _encode_event(record, self.event_max_bytes)
        except _DetailLimit:
            record["details"] = {}
            record["diagnostics"] = {"event_payload_omitted": True,
                "details_omitted": phase.details_omitted, "updates_omitted": phase.updates_omitted}
            if error is not None:
                record["error"] = {key: error[key] for key in ("error_type", "reason") if key in error}
            line = _encode_event(record, self.event_max_bytes)
        terminal = event in {"phase_exit", "phase_error"}
        active = sum(item.entered for item in self._phases)
        reserved = (active - int(terminal) + int(event == "phase_enter")) * self.event_max_bytes
        if self._journal_bytes + len(line) + reserved > self.journal_max_bytes:
            if event == "phase_update":
                phase.updates_omitted += 1
                return
            raise JournalLimitExceeded(limit_bytes=self.journal_max_bytes,
                                       written_bytes=self._journal_bytes, phase=phase.name)
        with self.event_path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self._journal_bytes += len(line)
        if event == "phase_enter":
            phase.entered = True
        self._seq += 1


_CHECKPOINT_PRODUCERS = (ExecutionSession.checkpoint, ExecutionSession._check_owner,
                         ExecutionSession._budget_states, ExecutionSession._stop)
_SESSION_CLOCK_PRODUCERS = (time.monotonic, time.process_time)
_CURRENT_SESSION_PRODUCER = current_session
