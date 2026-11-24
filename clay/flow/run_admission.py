"""Persistent failure admission shared by the CLI, API and portfolio runner.

The ledger belongs to the project/task, never the output directory or input
hash. It records execution control, not artifact ownership or physical proof.
It is not a sandbox against a caller deleting files or inventing task identities.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from clay.flow.execution_budget import ExecutionBudget
from clay.flow.run_workspace import acquire_file_lock, atomic_json, _owned_path, _identifier


_CURRENT: ContextVar[RunAdmission | None] = ContextVar("clay_run_admission", default=None)
RETRY_REASONS = frozenset({"input_changed", "producer_changed", "environment_repaired", "budget_experiment"})
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def current_admission():
    return _CURRENT.get()


class RunAdmissionError(RuntimeError):
    def __init__(self, reason: str, **details):
        self.reason, self.details = reason, details
        super().__init__(reason)

    def flow_error_data(self):
        return {"reason": self.reason, **self.details}

    def to_dict(self):
        return {"schema": "clay.run-admission/v1", "status": "blocked", **self.flow_error_data()}


@dataclass(frozen=True)
class RetryRequest:
    """An explicit retry proposal, bound to the exact previous terminal receipt."""
    prior_receipt_sha256: str
    reason: str
    evidence: tuple[Path, ...]

    def __post_init__(self):
        if (not isinstance(self.prior_receipt_sha256, str) or len(self.prior_receipt_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.prior_receipt_sha256)):
            raise ValueError("retry requires the previous receipt's SHA256")
        if self.reason not in RETRY_REASONS:
            raise ValueError("unknown retry reason")
        if not 1 <= len(self.evidence) <= 8:
            raise ValueError("retry requires one to eight evidence files")
        object.__setattr__(self, "evidence", tuple(Path(p) for p in self.evidence))


def project_root_for(case_dir: Path) -> Path:
    """Use a stable project ancestor across versioned cases; external cases stand alone."""
    case = Path(case_dir).resolve()
    for parent in (case, *case.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").is_file():
            return parent
    return case


def _identity(path: Path, *, checkpoint=None):
    # Stream even large receipts: failure handling must not read a GDS/JSON into RAM.
    check = checkpoint if checkpoint is not None else lambda: None
    check()
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise RunAdmissionError("ADMISSION_EVIDENCE_TOO_LARGE", path=str(path), max_bytes=MAX_EVIDENCE_BYTES)
    digest, size = sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check()
            size += len(block)
            if size > MAX_EVIDENCE_BYTES:
                raise RunAdmissionError("ADMISSION_EVIDENCE_TOO_LARGE", path=str(path), max_bytes=MAX_EVIDENCE_BYTES)
            digest.update(block)
    check()
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": size}


def _read_small(path: Path):
    if path.stat().st_size > 64 * 1024:
        raise RunAdmissionError("INVALID_ADMISSION_STATE")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise RunAdmissionError("INVALID_ADMISSION_STATE") from exc


class RunAdmission:
    def __init__(self, spec, *, project_root=None, task_id=None, budget=None, retry=None):
        self.root = Path(project_root).resolve() if project_root is not None else project_root_for(spec.case_dir)
        self.task_id = task_id if task_id is not None else spec.key
        if not isinstance(self.task_id, str) or not 1 <= len(self.task_id) <= 128:
            raise RunAdmissionError("TASK_ID_INVALID", detail="task_id must contain 1..128 characters")
        self.key = sha256(self.task_id.encode("utf-8")).hexdigest()[:32]
        self.directory = self.root / ".clay" / "flow-runs" / self.key
        self.budget = budget if budget is not None else ExecutionBudget()
        self.retry = retry
        self.invocation_id = uuid4().hex
        self.attempt_dir = self.directory / "attempts" / self.invocation_id
        self.receipt_path = self.attempt_dir / "receipt.json"
        self._lock = None
        self._token = None
        self._finished = False
        self._start = time.monotonic()
        self.control = {}
        self.native_row_candidates = None
        self._primitive_observation_head = None

    def record_primitive_observation_unknown(self, work_dir, provider, *, checkpoint):
        self._append_primitive_observation('unknown', {
            "work_dir": str(Path(work_dir).resolve()), "provider": str(provider),
            "status": "UNKNOWN", "scope": "no primitive stage admission",
        }, checkpoint=checkpoint)

    def primitive_observation_head(self):
        return None if self._primitive_observation_head is None else dict(self._primitive_observation_head)

    def record_primitive_observation(self, work_dir, manifest, identity, *, checkpoint):
        """Bind a completed capture to this invocation, without granting admission."""
        manifest = Path(manifest).absolute()
        if manifest.parent != self.attempt_dir or self._finished or self._token is None:
            raise ValueError("primitive observation does not belong to this attempt")
        self._append_primitive_observation('capture', {
            "path": str(manifest), **identity,
            "scope": "capture-time-only; fresh replay required; no stage or physical admission",
            "work_dir": str(Path(work_dir).resolve()),
        }, checkpoint=checkpoint)

    def _append_primitive_observation(self, kind, record, *, checkpoint):
        if self._finished or self._token is None:
            raise ValueError("primitive observation does not belong to an active attempt")
        checkpoint()
        previous = self.primitive_observation_head()
        sequence = 1 if previous is None else previous['count'] + 1
        link = {"schema": "clay.primitive-observation-link/v1", "invocation_id": self.invocation_id,
                "sequence": sequence, "previous": previous, "kind": kind, "record": record}
        encoded = (json.dumps(link, indent=2, sort_keys=True, allow_nan=False) + "\n").encode('utf-8')
        if len(encoded) > 64 * 1024:
            raise ValueError('primitive observation link exceeds the bounded record size')
        path = self.attempt_dir / ('primitive-link-' + uuid4().hex + '.json')
        atomic_json(path, link)
        identity = _identity(path)
        if identity['sha256'] != sha256(encoded).hexdigest() or identity['size_bytes'] != len(encoded):
            raise ValueError('primitive observation link changed during publication')
        checkpoint()
        self._primitive_observation_head = {**identity, 'count': sequence}

    def __enter__(self):
        _owned_path(self.root, str(self.directory.relative_to(self.root)))
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self._lock = acquire_file_lock(self.directory / "owner.lock")
        except ValueError as exc:
            raise RunAdmissionError("RUN_BUSY", task_id=self.task_id) from exc
        try:
            state_path = self.directory / "state.json"
            if state_path.exists():
                self.state = _read_small(state_path)
                self._validate_state()
                if self.state["schema"] == "clay.task-admission/v2":
                    _recover_primitive_stage(self)
            else:
                self.state = {"schema": "clay.task-admission/v1", "task_id": self.task_id,
                              "attempts": 0, "failed_attempts": 0, "total_wall_seconds": 0.0,
                              "status": "new"}
            _reconcile_production(self, state_path)
            previous = self.state.get("last_receipt")
            blocked = self.state["status"] in {"failed", "incomplete"}
            retry_data = None
            if blocked:
                if self.retry is None:
                    raise RunAdmissionError("PREVIOUS_FAILURE_REQUIRES_REVIEW", task_id=self.task_id,
                                            receipt=previous, failed_attempts=self.state["failed_attempts"],
                                            total_wall_seconds=self.state["total_wall_seconds"])
                if self.budget.wall_seconds is None:
                    raise RunAdmissionError("RETRY_REQUIRES_FINITE_WALL_BUDGET")
                if (previous is None or self.retry.prior_receipt_sha256 != previous["sha256"]
                        or _identity(Path(previous["path"])) != previous):
                    raise RunAdmissionError("RETRY_RECEIPT_MISMATCH")
                retry_data = {"prior_receipt_sha256": self.retry.prior_receipt_sha256,
                              "reason": self.retry.reason,
                              "evidence": [_identity(p) for p in self.retry.evidence],
                              "evidence_verified": False}
            elif self.retry is not None:
                raise RunAdmissionError("RETRY_WITHOUT_PREVIOUS_FAILURE")
            self.attempt_dir.mkdir(parents=True)
            self.state.update(status="running", invocation_id=self.invocation_id,
                              attempts=self.state["attempts"] + 1)
            atomic_json(self.attempt_dir / "start.json", {
                "schema": "clay.task-attempt/v1", "task_id": self.task_id,
                "invocation_id": self.invocation_id, "budget": self.budget.to_dict(),
                "retry": retry_data, "previous_receipt": previous,
                "physical_admission": False})
            atomic_json(state_path, self.state)
            self._token = _CURRENT.set(self)
            return self
        except BaseException:
            self._lock.close()
            self._lock = None
            raise

    def finish(self, *, passed: bool, diagnostic: dict):
        if self._finished:
            raise RuntimeError("attempt already finished")
        from clay.flow.execution_session import _bounded_snapshot, _DetailLimit
        try:
            diagnostic = _bounded_snapshot(diagnostic, 16 * 1024)
        except (_DetailLimit, TypeError, ValueError):
            diagnostic = {"reason": "DIAGNOSTIC_LIMIT_EXCEEDED", "details_omitted": True}
        elapsed = time.monotonic() - self._start
        status = "complete" if passed else "failed"
        receipt = {"schema": "clay.task-attempt/v1", "task_id": self.task_id,
                   "invocation_id": self.invocation_id, "status": status,
                   "wall_seconds": elapsed, "diagnostic": diagnostic,
                   "physical_admission": False}
        if self._primitive_observation_head:
            receipt["primitive_observation_index"] = self.primitive_observation_head()
        if self.native_row_candidates is not None:
            receipt['native_row_candidates'] = self.native_row_candidates
        if self.receipt_path.exists():
            raise RuntimeError("attempt terminal already published")
        atomic_json(self.receipt_path, receipt)
        # The terminal is authoritative even if updating the index fails.
        self._finished = True
        self.state.update(status=status, last_receipt=_identity(self.receipt_path),
                          total_wall_seconds=self.state["total_wall_seconds"] + elapsed,
                          failed_attempts=self.state["failed_attempts"] + (not passed))
        atomic_json(self.directory / "state.json", self.state)
        self.control = {"schema": "clay.run-admission/v1", "task_id": self.task_id,
                        "status": status, "receipt": self.state["last_receipt"],
                        "attempts": self.state["attempts"],
                        "failed_attempts": self.state["failed_attempts"],
                        "total_wall_seconds": self.state["total_wall_seconds"],
                        "unaccounted_interruption": self.state.get("unaccounted_interruption", False),
                        "budget": self.budget.to_dict(), "physical_admission": False}

    def _validate_state(self):
        state = self.state
        if (not isinstance(state, dict) or state.get("schema") not in {"clay.task-admission/v1", "clay.task-admission/v2"}
                or state.get("task_id") != self.task_id
                or state.get("status") not in ({"running", "complete", "failed", "incomplete", "new"}
                                                  if state.get("schema") == "clay.task-admission/v2"
                                                  else {"running", "complete", "failed", "incomplete"})
                or any(type(state.get(key)) is not int or state[key] < 0
                       for key in ("attempts", "failed_attempts"))
                or type(state.get("total_wall_seconds")) not in (int, float)
                or not math.isfinite(state["total_wall_seconds"]) or state["total_wall_seconds"] < 0):
            raise RunAdmissionError("INVALID_ADMISSION_STATE")
        try:
            if state["status"] == "new":
                if (state["attempts"] != 0 or state["failed_attempts"] != 0
                        or state.get("invocation_id") is not None or state.get("last_receipt") is not None):
                    raise ValueError("invalid new production state")
            else:
                _identifier(state.get("invocation_id"))
                _owned_path(self.directory, "attempts/" + state["invocation_id"])
            if state.get("last_receipt") is not None:
                recorded = state["last_receipt"]
                path = Path(recorded["path"])
                relative = path.relative_to(self.directory)
                _owned_path(self.directory, str(relative))
                if _identity(path) != recorded:
                    raise ValueError("receipt changed")
                if state["status"] != "running":
                    terminal = _read_small(path)
                    if (not isinstance(terminal, dict) or terminal.get("schema") != "clay.task-attempt/v1"
                            or terminal.get("task_id") != self.task_id
                            or terminal.get("status") != state["status"]
                            or terminal.get("invocation_id") != state["invocation_id"]):
                        raise ValueError("terminal disagrees with task index")
            elif state["status"] not in {"running", "new"}:
                raise ValueError("terminal receipt missing")
            if state["schema"] == "clay.task-admission/v2":
                _validate_primitive_stage_state(self)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise RunAdmissionError("INVALID_ADMISSION_STATE") from exc

    def __exit__(self, exc_type, exc, tb):
        try:
            if not self._finished:
                try:
                    diagnostic = {"reason": "RUN_INTERRUPTED" if isinstance(exc, BaseException)
                                  and not isinstance(exc, Exception) else "INTERNAL_ERROR",
                                  "error_type": type(exc).__name__ if exc else None}
                    notes = getattr(exc, "__notes__", None)
                    if type(notes) is list:
                        captured = [note[:512] for note in notes[:8] if type(note) is str]
                        if captured:
                            diagnostic["exception_notes"] = captured
                    self.finish(passed=False, diagnostic=diagnostic)
                except Exception as observation_error:
                    if exc is None:
                        raise
                    exc.add_note("Run terminal observation failed: " + type(observation_error).__name__)
        finally:
            if self._token is not None:
                _CURRENT.reset(self._token)
            self._lock.close()


def _reconcile_production(admission, state_path):
    """Finish indexing the existing production attempt while the task lock is held."""
    state = admission.state
    if state["status"] != "running":
        return
    prior = _owned_path(admission.directory, "attempts/" + state["invocation_id"] + "/receipt.json")
    if prior.exists():
        terminal = _read_small(prior)
        elapsed = terminal.get("wall_seconds")
        if (terminal.get("schema") != "clay.task-attempt/v1"
                or terminal.get("task_id") != admission.task_id
                or terminal.get("invocation_id") != state["invocation_id"]
                or terminal.get("status") not in {"complete", "failed", "incomplete"}
                or (elapsed is not None and (type(elapsed) not in (int, float)
                    or not math.isfinite(elapsed) or elapsed < 0))):
            raise RunAdmissionError("INVALID_ADMISSION_STATE")
        state.update(status=terminal["status"],
                     total_wall_seconds=state["total_wall_seconds"] + (elapsed or 0))
    else:
        atomic_json(prior, {"schema": "clay.task-attempt/v1", "status": "incomplete",
                           "reason": "RUN_INTERRUPTED", "task_id": admission.task_id,
                           "invocation_id": state["invocation_id"],
                           "wall_seconds": None, "physical_admission": False})
        state.update(status="incomplete", unaccounted_interruption=True)
    state.update(last_receipt=_identity(prior),
                 failed_attempts=state["failed_attempts"] + (state["status"] != "complete"))
    atomic_json(state_path, state)


def _stage_path(admission, identity, *, filename):
    """Validate the owned path of a stage document before reading it."""
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256", "size_bytes"}:
        raise ValueError("invalid stage identity")
    path = Path(identity["path"])
    relative = path.relative_to(admission.directory)
    parts = relative.parts
    if len(parts) != 3 or parts[0] != "attempts" or parts[2] != filename:
        raise ValueError("stage identity path differs")
    _identifier(parts[1])
    _owned_path(admission.directory, str(relative))
    if str(path.resolve(strict=True)) != identity["path"]:
        raise ValueError("stage identity path differs")
    return path


def _stage_document(admission, identity, *, filename, checkpoint=lambda: None):
    path = _stage_path(admission, identity, filename=filename)
    limit = MAX_EVIDENCE_BYTES if filename == "stage-sources.json" else 64 * 1024
    return path, _read_bound_stage_json(identity, checkpoint=checkpoint, limit=limit)


def _checked_role_trace_request(request, budget):
    from clay.flow.execution_budget import _validate_seconds
    if (not isinstance(request, dict) or set(request) != {'schema', 'capability', 'wall_seconds', 'scope'}
            or (request['schema'], request['capability'], request['scope']) not in {
                ('clay.primitive-role-trace-request/v1', 'pdk-role-trace', 'sky130-mos-guard1-nf1-m1'),
                ('clay.primitive-role-trace-request/v2', 'pdk-role-trace', 'sky130-mos-guard1-nf1-m1-and-mim-single')}):
        raise ValueError('invalid primitive role trace request')
    seconds = _validate_seconds(request['wall_seconds'], 'role_trace_seconds')
    if (not isinstance(budget, dict) or type(budget.get('wall_seconds')) not in (int, float)
            or not math.isfinite(budget['wall_seconds']) or seconds > budget['wall_seconds']):
        raise ValueError('role trace budget must fit inside the stage wall budget')
    return request


def _validate_primitive_stage_state(admission):
    state = admission.state
    checkpoint = getattr(admission, "checkpoint", lambda: None)
    stage = state.get("primitive_stage")
    if (not isinstance(stage, dict)
            or set(stage) != {"attempts", "failed_attempts", "total_wall_seconds",
                              "unaccounted_interruption", "last_receipt", "active"}
            or any(type(stage.get(key)) is not int or stage[key] < 0
                   for key in ("attempts", "failed_attempts"))
            or stage["failed_attempts"] > stage["attempts"]
            or type(stage.get("total_wall_seconds")) not in (int, float)
            or not math.isfinite(stage["total_wall_seconds"]) or stage["total_wall_seconds"] < 0
            or type(stage.get("unaccounted_interruption")) is not bool):
        raise ValueError("invalid primitive stage state")
    previous = stage["last_receipt"]
    if previous is not None:
        path, receipt = _stage_document(admission, previous, filename="receipt.json", checkpoint=checkpoint)
        if (receipt.get("schema") not in {"clay.primitive-stage-receipt/v1", "clay.primitive-stage-receipt/v2",
                                           "clay.primitive-stage-recovery-receipt/v1"}
                or receipt.get("task_id") != admission.task_id
                or receipt.get("invocation_id") != path.parent.name
                or receipt.get("status") not in {"complete", "failed", "unknown"}
                or receipt.get("physical_admission") is not False):
            raise ValueError("invalid primitive stage terminal")
        if receipt["schema"] == "clay.primitive-stage-recovery-receipt/v1":
            checked = _checked_stage_terminal(admission, receipt["active"], checkpoint=checkpoint,
                                              expected_identity=previous)
            if checked is None or checked[0] != previous:
                raise ValueError("primitive recovery terminal differs")
    active = stage["active"]
    if active is None:
        return
    if (not isinstance(active, dict)
            or set(active) != {"invocation_id", "start", "work_dir", "plan"}
            or state["status"] == "running"):
        raise ValueError("invalid active primitive stage")
    _identifier(active["invocation_id"])
    start, doc = _stage_document(admission, active["start"], filename="stage-start.json", checkpoint=checkpoint)
    if start.parent.name != active["invocation_id"]:
        raise ValueError("active stage invocation differs")
    if (doc.get("schema") not in {"clay.primitive-stage-start/v1", "clay.primitive-stage-start/v2"}
            or doc.get("task_id") != admission.task_id
            or doc.get("invocation_id") != active["invocation_id"]
            or doc.get("previous_stage_receipt") != previous
            or doc.get("previous_production_receipt") != state.get("last_receipt")):
        raise ValueError("active stage lineage differs")
    if doc['schema'] == 'clay.primitive-stage-start/v2':
        _checked_role_trace_request(doc.get('role_trace_request'), doc.get('budget'))
    elif 'role_trace_request' in doc:
        raise ValueError('legacy primitive stage cannot carry role trace request')
    if active["work_dir"] is not None:
        work = Path(active["work_dir"])
        if not work.is_absolute() or str(work.resolve()) != active["work_dir"]:
            raise ValueError("active stage work directory differs")
    if active["plan"] is not None:
        if active["work_dir"] is None:
            raise ValueError("active plan has no work directory")
        plan, _ = _stage_document(admission, active["plan"], filename="native-container-plan.json", checkpoint=checkpoint)
        if plan.parent.name != active["invocation_id"]:
            raise ValueError("active stage plan differs")


def _checked_stage_terminal(admission, active, *, checkpoint, expected_identity=None):
    """Reverify a previously written stage terminal before indexing it."""
    _, start = _stage_document(admission, active["start"], filename="stage-start.json", checkpoint=checkpoint)
    receipt_path = _owned_path(admission.directory, "attempts/" + active["invocation_id"] + "/receipt.json")
    if not receipt_path.is_file():
        return None
    if expected_identity is None:
        receipt, receipt_identity = _read_unbound_stage_json(receipt_path, checkpoint=checkpoint)
    else:
        if Path(expected_identity["path"]) != receipt_path:
            raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
        receipt = _read_bound_stage_json(expected_identity, checkpoint=checkpoint, limit=64 * 1024)
        receipt_identity = expected_identity
    if receipt.get("schema") == "clay.primitive-stage-recovery-receipt/v1":
        return _checked_precreate_recovery_terminal(admission, active, start, receipt,
                                                     receipt_identity, checkpoint=checkpoint)
    elapsed = receipt.get("wall_seconds")
    schema_pair = (start.get("schema"), receipt.get("schema"))
    if (schema_pair not in {("clay.primitive-stage-start/v1", "clay.primitive-stage-receipt/v1"),
                            ("clay.primitive-stage-start/v2", "clay.primitive-stage-receipt/v2")}
            or receipt.get("task_id") != admission.task_id
            or receipt.get("invocation_id") != active["invocation_id"]
            or receipt.get("status") not in {"complete", "failed", "unknown"}
            or receipt.get("physical_admission") is not False
            or receipt.get("verdict") != "UNKNOWN"
            or receipt.get("managed") is not start.get("managed")
            or receipt.get("start") != active["start"]
            or receipt.get("resource_plan") != active["plan"]
            or receipt.get("previous_production_receipt") != start["previous_production_receipt"]
            or receipt.get("previous_stage_receipt") != start["previous_stage_receipt"]
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
        raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
    if schema_pair[0] == 'clay.primitive-stage-start/v2':
        request = _checked_role_trace_request(start.get('role_trace_request'), start.get('budget'))
        if receipt.get('role_trace_request') != request:
            raise RunAdmissionError('INVALID_PRIMITIVE_STAGE_TERMINAL')
    elif 'role_trace_request' in start or 'role_trace_request' in receipt:
        raise RunAdmissionError('INVALID_PRIMITIVE_STAGE_TERMINAL')
    if receipt.get('production_quiescence') != start.get('production_quiescence'):
        raise RunAdmissionError('INVALID_PRIMITIVE_STAGE_TERMINAL')
    observation = start.get('production_quiescence')
    if observation is not None:
        if (observation.get('schema') != 'clay.current-quiescence-observation/v1'
                or any(observation.get(key) != 'UNKNOWN' for key in
                       ('historical_cleanup', 'other_endpoints', 'host_processes', 'scope_complete'))
                or observation.get('physical_admission') is not False):
            raise RunAdmissionError('INVALID_PRIMITIVE_STAGE_TERMINAL')
        for row in observation.get('commands', ()):
            for key in ('stdout', 'stderr'):
                identity = row[key]
                if (Path(identity['path']).parent != receipt_path.parent/'current-quiescence'
                        or _identity(Path(identity['path']), checkpoint=checkpoint) != identity):
                    raise RunAdmissionError('INVALID_PRIMITIVE_STAGE_TERMINAL')
    bindings = _validate_stage_source_bindings(admission, start, checkpoint=checkpoint)
    if active["plan"] is None:
        if receipt["status"] not in {"unknown", "failed"} or receipt.get("resource_evidence") is not None:
            raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
    else:
        _, plan = _stage_document(admission, active["plan"], filename="native-container-plan.json", checkpoint=checkpoint)
        runtime = _read_bound_stage_json(bindings["runtime"], checkpoint=checkpoint)
        if (plan.get("runtime_contract_sha256") != _runtime_contract_sha256(runtime)
                or plan.get("tmpfs") != {"/tmp": "rw"}):
            raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
        if ((schema_pair[0] == 'clay.primitive-stage-start/v2'
             and (plan.get('schema') != 'clay.native-container-plan/v2'
                  or plan.get('role_trace_request') != start['role_trace_request']))
                or (schema_pair[0] == 'clay.primitive-stage-start/v1'
                    and (plan.get('schema') != 'clay.native-container-plan/v1'
                         or 'role_trace_request' in plan))):
            raise RunAdmissionError('INVALID_PRIMITIVE_STAGE_TERMINAL')
        _validate_stage_plan_inputs(plan, checkpoint=checkpoint)
        evidence = receipt.get("resource_evidence")
        if (not isinstance(evidence, dict)
                or Path(evidence["path"]).parent != Path(plan["work_dir"])
                or Path(evidence["path"]).name != "native-supervision.json"
                or _identity(Path(evidence["path"])) != evidence):
            raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
        from clay.flow.native_owned_container import verify_native_container_evidence
        actual = verify_native_container_evidence(Path(evidence["path"]), plan=plan,
            plan_identity=active["plan"], checkpoint=checkpoint, expected_identity=evidence)
        actual = json.loads(json.dumps(actual, allow_nan=False, sort_keys=True))
        if actual != receipt.get("resource_verification"):
            raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
        if receipt["status"] == "complete" and actual.get("outcome") != "COMPLETE":
            raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
        if receipt["status"] == "complete":
            trace_identity = _validate_stage_capture(admission, active, receipt.get("primitive_observation_index"), plan,
                                                    evidence_identity=evidence, checkpoint=checkpoint)
            if schema_pair[0] == 'clay.primitive-stage-start/v2' and receipt.get('role_trace_manifest') != trace_identity:
                raise RunAdmissionError('INVALID_PRIMITIVE_STAGE_TERMINAL')
    _validate_stage_managed_terminal(admission, active, receipt.get("primitive_observation_index"),
                                     receipt.get("managed_terminal"), receipt["status"], checkpoint=checkpoint)
    return receipt_identity, receipt


def _checked_precreate_recovery_terminal(admission, active, start, receipt, identity, *, checkpoint):
    recovery_wall = receipt.get("recovery_wall_seconds")
    budget = receipt.get("recovery_budget")
    if (receipt.get("task_id") != admission.task_id
            or receipt.get("invocation_id") != active["invocation_id"]
            or start.get("task_id") != admission.task_id
            or start.get("invocation_id") != active["invocation_id"]
            or Path(active["start"]["path"]).parent.name != active["invocation_id"]
            or receipt.get("scope") != "primitive-precreate-recovery"
            or receipt.get("status") != "failed" or receipt.get("verdict") != "UNKNOWN"
            or receipt.get("physical_admission") is not False or receipt.get("wall_seconds") is not None
            or start.get("managed") is not False or receipt.get("managed") is not False
            or receipt.get("active") != active or receipt.get("start") != active["start"]
            or receipt.get("resource_plan") != active["plan"]
            or receipt.get("previous_production_receipt") != start["previous_production_receipt"]
            or receipt.get("previous_stage_receipt") != start["previous_stage_receipt"]
            or type(recovery_wall) not in (int, float) or not math.isfinite(recovery_wall) or recovery_wall < 0
            or not isinstance(budget, dict) or type(budget.get("wall_seconds")) not in (int, float)
            or not math.isfinite(budget["wall_seconds"]) or budget["wall_seconds"] <= 0
            or recovery_wall > budget["wall_seconds"]):
        raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
    evidence = receipt.get("recovery_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256", "size_bytes"}:
        raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
    evidence_path = Path(evidence["path"])
    attempt_dir = admission.directory / "attempts" / active["invocation_id"]
    if (evidence_path.parent.parent != attempt_dir
            or not evidence_path.parent.name.startswith("precreate-recovery-")
            or _identity(evidence_path, checkpoint=checkpoint) != evidence):
        raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
    _owned_path(attempt_dir, str(evidence_path.relative_to(attempt_dir)))
    from clay.flow.primitive_precreate_recovery import verify_precreate_recovery_evidence
    verify_precreate_recovery_evidence(admission=admission, active=active,
        evidence_identity=receipt.get("recovery_evidence"),
        preparation_identity=receipt.get("preparation_manifest"), checkpoint=checkpoint)
    return identity, receipt


def _validate_stage_plan_inputs(plan, *, checkpoint):
    """Recheck declared generated input bytes before and after execution."""
    import re
    work = Path(plan["work_dir"])
    records = plan.get("inputs_sha256")
    if not isinstance(records, dict) or not records:
        raise ValueError("primitive plan lacks generated input identities")
    folders = [path for path in work.iterdir() if path.name.startswith("pcell_")]
    if (not folders or any(not path.is_dir() or not re.fullmatch(r"pcell_[0-9a-f]{20}", path.name)
                           for path in folders)):
        raise ValueError("primitive plan PCell input inventory differs")
    expected = {work / "generate.py", work / "toolchain.py", work / "reference.spice"}
    for folder in folders:
        expected.update((folder / "request.json", folder / "generate.tcl"))
    if plan.get('schema') == 'clay.native-container-plan/v2':
        from clay.flow.primitive_role_trace import eligible_spec
        for folder in folders:
            checkpoint()
            request_bytes, _ = _read_stage_bytes(folder / 'request.json', checkpoint=checkpoint,
                                                  limit=MAX_EVIDENCE_BYTES)
            from clay.flow.primitive_mim_role_trace import eligible_for_request
            if eligible_for_request(json.loads(request_bytes), plan['role_trace_request']):
                expected.add(folder / 'trace' / 'generate.tcl')
    if set(records) != {str(path) for path in expected}:
        raise ValueError("primitive plan generated input inventory differs")
    for raw, digest in records.items():
        checkpoint()
        path = Path(raw)
        if (not path.is_absolute() or str(path.resolve(strict=True)) != raw
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("invalid primitive generated input identity")
        _owned_path(work, str(path.relative_to(work)))
        if (not path.is_file() or path.stat().st_nlink != 1
                or _identity(path, checkpoint=checkpoint)["sha256"] != digest):
            raise ValueError("primitive generated input changed")
    cli = plan.get("docker_cli")
    if not isinstance(cli, dict) or set(cli) != {"path", "sha256", "size_bytes"}:
        raise ValueError("primitive plan Docker CLI observation missing")
    path = Path(cli["path"])
    if (not path.is_absolute() or str(path.resolve(strict=True)) != str(path)
            or type(cli["size_bytes"]) is not int or cli["size_bytes"] <= 0
            or not isinstance(cli["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", cli["sha256"])):
        raise ValueError("primitive plan Docker CLI identity differs")
    digest, size = sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checkpoint()
            digest.update(block)
            size += len(block)
    if size != cli["size_bytes"] or digest.hexdigest() != cli["sha256"]:
        raise ValueError("primitive plan Docker CLI changed")
    checkpoint()


def _read_stage_bytes(path, *, checkpoint=lambda: None, limit=64 * 1024):
    checkpoint()
    chunks, size = [], 0
    with path.open("rb") as stream:
        while True:
            checkpoint()
            block = stream.read(min(1024 * 1024, limit + 1 - size))
            if not block:
                break
            size += len(block)
            if size > limit:
                raise ValueError("primitive stage document too large")
            chunks.append(block)
    checkpoint()
    data = b"".join(chunks)
    identity = {"path": str(path.resolve()), "sha256": sha256(data).hexdigest(), "size_bytes": len(data)}
    return data, identity


def _read_unbound_stage_json(path, *, checkpoint=lambda: None, limit=64 * 1024):
    data, identity = _read_stage_bytes(path, checkpoint=checkpoint, limit=limit)
    value = json.loads(data)
    checkpoint()
    return value, identity


def _read_bound_stage_json(identity, *, checkpoint=lambda: None, limit=MAX_EVIDENCE_BYTES):
    if (not isinstance(identity, dict) or set(identity) != {"path", "sha256", "size_bytes"}
            or type(identity["size_bytes"]) is not int or not 0 <= identity["size_bytes"] <= limit):
        raise ValueError("invalid primitive stage identity")
    path = Path(identity["path"])
    data, observed = _read_stage_bytes(path, checkpoint=checkpoint, limit=identity["size_bytes"])
    if observed != identity:
        raise ValueError("primitive stage observation changed")
    value = json.loads(data)
    checkpoint()
    return value


def _runtime_contract_sha256(runtime):
    encoded = json.dumps(runtime, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_stage_source_bindings(admission, start, *, checkpoint):
    _, bindings = _stage_document(admission, start["source_bindings"], filename="stage-sources.json", checkpoint=checkpoint)
    if (not isinstance(bindings, dict) or set(bindings) != {"design", "spice", "runtime", "dependencies"}
            or not isinstance(bindings["dependencies"], list) or not bindings["dependencies"]):
        raise ValueError("invalid primitive source bindings")
    for role in ("design", "spice", "runtime"):
        identity = bindings[role]
        if identity is None:
            if role != "runtime":
                raise ValueError("invalid primitive source bindings")
            continue
        if _identity(Path(identity["path"]), checkpoint=checkpoint) != identity:
            raise ValueError("primitive stage source changed")
    for identity in bindings["dependencies"]:
        if _identity(Path(identity["path"]), checkpoint=checkpoint) != identity:
            raise ValueError("primitive stage dependency changed")
    return bindings


def _validate_stage_capture(admission, active, head, plan, *, checkpoint, evidence_identity=None):
    """Tie the unique producer capture to the planned geometry and exact sources."""
    if not isinstance(head, dict) or set(head) != {"path", "sha256", "size_bytes", "count"} or head["count"] != 1:
        raise ValueError("primitive stage requires exactly one capture")
    link_identity = {key: head[key] for key in ("path", "sha256", "size_bytes")}
    link_path = Path(link_identity["path"])
    if link_path.parent != admission.directory / "attempts" / active["invocation_id"]:
        raise ValueError("primitive capture link belongs to another attempt")
    _owned_path(admission.directory, str(link_path.relative_to(admission.directory)))
    if _identity(link_path) != link_identity:
        raise ValueError("primitive capture link changed")
    link = _read_bound_stage_json(link_identity)
    if (link.get("schema") != "clay.primitive-observation-link/v1"
            or link.get("invocation_id") != active["invocation_id"]
            or link.get("sequence") != 1 or link.get("previous") is not None
            or link.get("kind") != "capture"):
        raise ValueError("primitive stage capture link differs")
    record = link.get("record")
    if not isinstance(record, dict):
        raise ValueError("primitive stage capture record missing")
    manifest = Path(record["path"])
    if manifest.parent != link_path.parent:
        raise ValueError("primitive stage manifest belongs to another attempt")
    manifest_identity = {key: record[key] for key in ("path", "sha256", "size_bytes")}
    _owned_path(admission.directory, str(manifest.relative_to(admission.directory)))
    if _identity(manifest) != manifest_identity:
        raise ValueError("primitive stage manifest changed")
    payload = _read_bound_stage_json(manifest_identity)
    if (payload.get("schema") != "clay.primitive-producer-observation/v1"
            or payload.get("invocation_id") != active["invocation_id"]
            or payload.get("output_roots") != [plan["work_dir"]]
            or payload.get("row_members") != [] or payload.get("rows") != []
            or payload.get("row_program_sha256") is not None):
        raise ValueError("primitive capture is not the planned standalone producer")
    work = Path(record["work_dir"]).resolve(strict=True)
    output = Path(plan["work_dir"]).resolve(strict=True)
    if str(work) != active["work_dir"] or output.parent != work:
        raise ValueError("primitive capture work directory differs")
    from clay.flow.primitive_observation import _files, _identity as observation_identity
    if _files(output, work, checkpoint) != payload.get("outputs"):
        raise ValueError("primitive capture outputs changed")
    for path, identity in payload.get("dependencies", {}).items():
        if observation_identity(path, checkpoint) != identity:
            raise ValueError("primitive capture dependency changed")
    _, start = _stage_document(admission, active["start"], filename="stage-start.json", checkpoint=checkpoint)
    bindings_identity = start["source_bindings"]
    _, bindings = _stage_document(admission, bindings_identity, filename="stage-sources.json", checkpoint=checkpoint)
    sources = payload.get("source_files", {})
    for role in ("design", "spice", "runtime"):
        identity = bindings[role]
        captured = sources.get(identity["path"])
        if captured != {"sha256": identity["sha256"], "size_bytes": identity["size_bytes"]}:
            raise ValueError("primitive capture source identity differs")
    bound_dependencies = {row["path"]: {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}
                          for row in bindings["dependencies"]}
    captured_dependencies = payload.get("dependencies")
    if (not isinstance(captured_dependencies, dict) or not captured_dependencies
            or any(bound_dependencies.get(path) != identity
                   for path, identity in captured_dependencies.items())):
        raise ValueError("primitive capture producer dependency closure differs")
    import inspect
    from clay.layout.native_sky130 import NativeGeometryBundle
    producer = str(Path(inspect.getsourcefile(NativeGeometryBundle)).resolve())
    if (producer not in captured_dependencies
            or payload.get("producer_sha256") != captured_dependencies[producer]["sha256"]):
        raise ValueError("primitive capture producer identity differs")
    request = start.get('role_trace_request')
    if request is None:
        if 'role_trace' in payload:
            raise ValueError('unrequested primitive role trace in capture')
        return None
    if start.get('schema') != 'clay.primitive-stage-start/v2' or plan.get('role_trace_request') != request:
        raise ValueError('primitive role trace scope differs')
    if evidence_identity is None:
        evidence_identity = _identity(output / 'native-supervision.json', checkpoint=checkpoint)
    from clay.flow.primitive_role_trace_capture import verify_live_role_trace_manifest
    trace_identity = verify_live_role_trace_manifest(output=output, request=request,
        invocation_id=active['invocation_id'], source_identity=payload['source_identity'],
        source_file_identity=bindings['design'],
        runtime_sha256=bindings['runtime']['sha256'],
        runtime_file_identity=bindings['runtime'],
        plan_identity=active['plan'], evidence_identity=evidence_identity, checkpoint=checkpoint)
    if payload.get('role_trace') != {'request': request, 'manifest': trace_identity}:
        raise ValueError('primitive capture role trace binding differs')
    return trace_identity


def _validate_stage_managed_terminal(admission, active, head, identity, status, *, checkpoint=lambda: None):
    _, start = _stage_document(admission, active["start"], filename="stage-start.json", checkpoint=checkpoint)
    managed = start.get("managed")
    if type(managed) is not bool:
        raise ValueError("primitive stage managed identity missing")
    if not managed:
        if identity is not None:
            raise ValueError("legacy primitive stage cannot have a managed terminal")
        return
    work = active["work_dir"]
    if work is None or not isinstance(identity, dict):
        raise ValueError("managed primitive stage requires its terminal")
    terminal_path = Path(identity["path"])
    expected_path = Path(work).parent / "terminal.json"
    if terminal_path != expected_path or _identity(terminal_path) != identity:
        raise ValueError("managed primitive terminal identity differs")
    terminal = _read_bound_stage_json(identity, checkpoint=checkpoint)
    if (terminal.get("schema") != "clay.run-terminal/v1"
            or terminal.get("invocation_id") != terminal_path.parent.name):
        raise ValueError("managed primitive terminal scope differs")
    if status in {"complete", "unknown"}:
        if terminal.get("status") != "complete" or terminal.get("verdict") != "unknown":
            raise ValueError("managed primitive execution did not complete")
    elif terminal.get("status") not in {"failed", "cancelled", "timeout"} or terminal.get("verdict") != "fail":
        raise ValueError("managed primitive failure terminal differs")
    index = terminal.get("primitive_observation_index")
    if head is None:
        if index is not None:
            raise ValueError("managed primitive terminal has an unexpected capture")
    elif (not isinstance(index, dict) or index.get("head") != head
          or index.get("work_dir") != work
          or index.get("scope") != "only records for this work_dir through this head"):
        raise ValueError("managed primitive terminal capture differs")


def _recover_primitive_stage(admission, *, checkpoint=None):
    from clay.flow.execution_budget import ExecutionStopped
    check = checkpoint if checkpoint is not None else getattr(admission, "checkpoint", lambda: None)
    stage = admission.state["primitive_stage"]
    active = stage["active"]
    if active is None:
        return
    try:
        terminal = _checked_stage_terminal(admission, active, checkpoint=check)
    except ExecutionStopped:
        raise
    except Exception as error:
        raise RunAdmissionError("PRIMITIVE_STAGE_RECOVERY_REQUIRED",
                                invocation_id=active["invocation_id"]) from error
    if terminal is None:
        if not stage["unaccounted_interruption"]:
            stage["unaccounted_interruption"] = True
            atomic_json(admission.directory / "state.json", admission.state)
        raise RunAdmissionError("PRIMITIVE_STAGE_RECOVERY_REQUIRED",
                                invocation_id=active["invocation_id"])
    receipt_identity, receipt = terminal
    stage.update(last_receipt=receipt_identity, active=None,
                 total_wall_seconds=stage["total_wall_seconds"] + (receipt["wall_seconds"] or 0),
                 failed_attempts=stage["failed_attempts"] + (receipt["status"] != "complete"))
    if receipt["wall_seconds"] is None:
        stage["unaccounted_interruption"] = True
    check()
    atomic_json(admission.directory / "state.json", admission.state)


def recover_precreate_primitive(spec, *, project_root, task_id, preparation_manifest,
                                budget, cancellation=None):
    """Recover one proven pre-create stage without altering its production lineage."""
    from clay.flow.execution_budget import CancellationToken, ExecutionStopped
    from clay.flow.primitive_precreate_recovery import verify_and_scan_precreate
    started = time.monotonic()
    if not isinstance(budget, ExecutionBudget) or budget.wall_seconds is None:
        raise RunAdmissionError("PRIMITIVE_RECOVERY_REQUIRES_FINITE_WALL_BUDGET")
    if task_id is None or project_root is None:
        raise RunAdmissionError("PRIMITIVE_RECOVERY_REQUIRES_TASK_IDENTITY")
    if cancellation is not None and not isinstance(cancellation, CancellationToken):
        raise TypeError("primitive recovery cancellation must be a CancellationToken")
    def checkpoint():
        if cancellation is not None and cancellation.reason is not None:
            raise ExecutionStopped("EXECUTION_CANCELLED", cancellation_reason=cancellation.reason)
        elapsed = time.monotonic() - started
        if elapsed >= budget.wall_seconds:
            raise ExecutionStopped("WALL_BUDGET_EXCEEDED", limit_seconds=budget.wall_seconds,
                                   consumed_seconds=elapsed)
    admission = RunAdmission(spec, project_root=project_root, task_id=task_id, budget=budget)
    admission.checkpoint = checkpoint
    checkpoint()
    _owned_path(admission.root, str(admission.directory.relative_to(admission.root)))
    if not (admission.directory / "state.json").is_file():
        raise RunAdmissionError("PRIMITIVE_RECOVERY_TASK_MISSING", task_id=task_id)
    try:
        lock = acquire_file_lock(admission.directory / "owner.lock")
    except ValueError as exc:
        raise RunAdmissionError("RUN_BUSY", task_id=task_id) from exc
    try:
        checkpoint()
        admission.state = _read_small(admission.directory / "state.json")
        admission._validate_state()
        if admission.state.get("schema") != "clay.task-admission/v2":
            raise RunAdmissionError("PRIMITIVE_RECOVERY_STAGE_MISSING", task_id=task_id)
        stage = admission.state["primitive_stage"]
        active = stage["active"]
        preparation_path = Path(preparation_manifest).resolve(strict=True)
        preparation = _identity(preparation_path, checkpoint=checkpoint)
        if active is None:
            previous = stage["last_receipt"]
            if previous is None:
                raise RunAdmissionError("PRIMITIVE_RECOVERY_STAGE_MISSING", task_id=task_id)
            _, receipt = _stage_document(admission, previous, filename="receipt.json", checkpoint=checkpoint)
            if (receipt.get("schema") != "clay.primitive-stage-recovery-receipt/v1"
                    or receipt.get("preparation_manifest") != preparation):
                raise RunAdmissionError("PRIMITIVE_RECOVERY_ALREADY_TERMINAL", task_id=task_id)
            checked = _checked_stage_terminal(admission, receipt["active"], checkpoint=checkpoint,
                                              expected_identity=previous)
            if checked is None or checked[0] != previous:
                raise RunAdmissionError("INVALID_PRIMITIVE_STAGE_TERMINAL")
            return {"schema": "clay.primitive-precreate-recovery-control/v1", "status": "failed",
                    "task_id": task_id, "receipt": previous, "already_recovered": True,
                    "physical_admission": False}
        attempt_dir = _owned_path(admission.directory, "attempts/" + active["invocation_id"])
        receipt_path = _owned_path(admission.directory, "attempts/" + active["invocation_id"] + "/receipt.json")
        if receipt_path.exists():
            checked = _checked_stage_terminal(admission, active, checkpoint=checkpoint)
            if (checked is None or checked[1].get("schema") != "clay.primitive-stage-recovery-receipt/v1"
                    or checked[1].get("preparation_manifest") != preparation):
                raise RunAdmissionError("PRIMITIVE_RECOVERY_ALREADY_TERMINAL", task_id=task_id)
        else:
            _, start = _stage_document(admission, active["start"], filename="stage-start.json", checkpoint=checkpoint)
            if start.get("managed") is not False:
                raise RunAdmissionError("PRIMITIVE_RECOVERY_MANAGED_UNSUPPORTED", task_id=task_id)
            recovery_dir = _owned_path(attempt_dir, "precreate-recovery-" + uuid4().hex)
            recovery_dir.mkdir()
            evidence = verify_and_scan_precreate(admission=admission, active=active,
                preparation_manifest=preparation_path, preparation_sha256=preparation["sha256"],
                recovery_dir=recovery_dir, budget=budget, checkpoint=checkpoint)
            if (not isinstance(evidence, dict) or set(evidence) != {"path", "sha256", "size_bytes"}
                    or Path(evidence["path"]).parent != recovery_dir
                    or _identity(Path(evidence["path"]), checkpoint=checkpoint) != evidence):
                raise RunAdmissionError("INVALID_PRIMITIVE_RECOVERY_EVIDENCE")
            _owned_path(recovery_dir, Path(evidence["path"]).name)
            checkpoint()
            receipt = {"schema": "clay.primitive-stage-recovery-receipt/v1", "task_id": task_id,
                "invocation_id": active["invocation_id"], "scope": "primitive-precreate-recovery",
                "status": "failed", "verdict": "UNKNOWN", "physical_admission": False,
                "wall_seconds": None, "managed": start["managed"], "active": active,
                "start": active["start"], "resource_plan": active["plan"],
                "previous_production_receipt": start["previous_production_receipt"],
                "previous_stage_receipt": start["previous_stage_receipt"],
                "preparation_manifest": preparation, "recovery_evidence": evidence,
                "recovery_budget": budget.to_dict(), "recovery_wall_seconds": time.monotonic() - started}
            from clay.flow.primitive_precreate_recovery import verify_precreate_recovery_evidence
            verify_precreate_recovery_evidence(admission=admission, active=active,
                evidence_identity=evidence, preparation_identity=preparation, checkpoint=checkpoint)
            checkpoint()
            atomic_json(receipt_path, receipt)
        _recover_primitive_stage(admission, checkpoint=checkpoint)
        return {"schema": "clay.primitive-precreate-recovery-control/v1", "status": "failed",
                "task_id": task_id, "receipt": stage["last_receipt"], "already_recovered": False,
                "physical_admission": False}
    finally:
        lock.close()


class PrimitiveStageAdmission(RunAdmission):
    """A separately accounted, source-bound primitive observation on the same task."""

    is_primitive_stage = True

    def __init__(self, spec, *, project_root=None, task_id=None, budget=None,
                 source_paths, retry=None, stage_retry=None, cancellation=None, managed=False,
                 unwind_budget_seconds=None, initial_wall_start=None, role_trace_seconds=None, role_trace_version=None,
                 production_quiescence=None):
        from clay.flow.execution_budget import CancellationToken, _validate_seconds
        super().__init__(spec, project_root=project_root, task_id=task_id, budget=budget, retry=retry)
        if initial_wall_start is not None:
            if (type(initial_wall_start) not in (int, float) or not math.isfinite(initial_wall_start)
                    or initial_wall_start > self._start):
                raise ValueError("invalid primitive stage initial wall start")
            self._start = initial_wall_start
        if not isinstance(self.budget, ExecutionBudget) or self.budget.wall_seconds is None:
            raise RunAdmissionError("PRIMITIVE_STAGE_REQUIRES_FINITE_WALL_BUDGET")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise TypeError("primitive stage cancellation must be a CancellationToken")
        if type(managed) is not bool:
            raise TypeError("primitive stage managed flag must be boolean")
        self.cancellation = cancellation if cancellation is not None else CancellationToken()
        self.managed = managed
        self.unwind_budget_seconds = (None if unwind_budget_seconds is None else
                                      _validate_seconds(unwind_budget_seconds, "unwind_budget_seconds"))
        if role_trace_version is not None and (type(role_trace_version) is not int or role_trace_version != 2 or role_trace_seconds is None):
            raise ValueError("role trace version 2 requires a role trace budget")
        self.role_trace_request = None
        if role_trace_seconds is not None:
            seconds = _validate_seconds(role_trace_seconds, "role_trace_seconds")
            if seconds > self.budget.wall_seconds:
                raise ValueError("role trace budget must fit inside the stage wall budget")
            self.role_trace_request = {
                "schema": "clay.primitive-role-trace-request/v1", "capability": "pdk-role-trace",
                "wall_seconds": seconds, "scope": "sky130-mos-guard1-nf1-m1"}
            if role_trace_version == 2:
                self.role_trace_request.update(schema="clay.primitive-role-trace-request/v2",
                    scope="sky130-mos-guard1-nf1-m1-and-mim-single")
        self._execution = None
        if not isinstance(source_paths, dict) or set(source_paths) != {"design", "spice", "runtime", "dependencies"}:
            raise ValueError("primitive stage requires explicit source and dependency paths")
        if (not isinstance(source_paths["dependencies"], (tuple, list))
                or not source_paths["dependencies"]):
            raise ValueError("primitive stage requires a nonempty implementation closure")
        self.source_paths = {
            "design": Path(source_paths["design"]),
            "spice": Path(source_paths["spice"]),
            "runtime": (None if source_paths["runtime"] is None else Path(source_paths["runtime"])),
            "dependencies": tuple(Path(p) for p in source_paths["dependencies"]),
        }
        self.stage_retry = stage_retry
        self.production_quiescence = production_quiescence
        self._stage_plan = None
        self._stage_plan_identity = None
        self._stage_start_identity = None
        self._stage_sources = None

    def __enter__(self):
        if _CURRENT.get() is not None:
            raise RunAdmissionError("NESTED_PRIMITIVE_STAGE_UNSUPPORTED")
        _owned_path(self.root, str(self.directory.relative_to(self.root)))
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self._lock = acquire_file_lock(self.directory / "owner.lock")
        except ValueError as exc:
            raise RunAdmissionError("RUN_BUSY", task_id=self.task_id) from exc
        try:
            self.checkpoint()
            state_path = self.directory / "state.json"
            if state_path.exists():
                self.state = _read_small(state_path)
                self._validate_state()
                if self.state["schema"] == "clay.task-admission/v2":
                    _recover_primitive_stage(self)
            else:
                self.state = {"schema": "clay.task-admission/v1", "task_id": self.task_id,
                              "attempts": 0, "failed_attempts": 0, "total_wall_seconds": 0.0,
                              "status": "new"}
            _reconcile_production(self, state_path)
            previous = self.state.get("last_receipt")
            blocked = self.state["status"] in {"failed", "incomplete"}
            if self.production_quiescence is not None and self.state['status'] != 'incomplete':
                raise RunAdmissionError('PRIMITIVE_CURRENT_OBSERVATION_REQUIRES_INCOMPLETE_PRODUCTION')
            retry_data = self._checked_retry(self.retry, previous, required=blocked,
                                             missing="PREVIOUS_FAILURE_REQUIRES_REVIEW",
                                             unexpected="RETRY_WITHOUT_PREVIOUS_FAILURE")
            # The ordinary retry contract records evidence identities but does
            # not verify that an interrupted external process tree is gone.
            # A stage must not use the released file lock as cleanup proof.
            production_observation = None
            if self.state["status"] == "incomplete" or self.state.get("unaccounted_interruption", False):
                if self.production_quiescence is None:
                    raise RunAdmissionError("PRIMITIVE_STAGE_PRODUCTION_CLEANUP_UNVERIFIED",
                                            task_id=self.task_id, receipt=previous)
                if not isinstance(self.retry, RetryRequest) or self.retry.reason != 'environment_repaired':
                    raise RunAdmissionError('PRIMITIVE_CURRENT_OBSERVATION_REQUIRES_ENVIRONMENT_RETRY')
                from clay.flow.current_quiescence import observe_current_quiescence
                production_observation = observe_current_quiescence(self.production_quiescence,
                    project_root=self.root, directory=self.attempt_dir/'current-quiescence',
                    checkpoint=self.checkpoint, remaining=self.remaining_wall_seconds)
            elif self.production_quiescence is not None:
                raise RunAdmissionError('PRIMITIVE_CURRENT_OBSERVATION_NOT_REQUIRED')
            if self.state["schema"] == "clay.task-admission/v1":
                self.state["schema"] = "clay.task-admission/v2"
                self.state["primitive_stage"] = {"attempts": 0, "failed_attempts": 0,
                    "total_wall_seconds": 0.0, "unaccounted_interruption": False,
                    "last_receipt": None, "active": None}
            stage = self.state["primitive_stage"]
            stage_previous = stage["last_receipt"]
            stage_blocked = False
            if stage_previous is not None:
                _, prior = _stage_document(self, stage_previous, filename="receipt.json", checkpoint=self.checkpoint)
                stage_blocked = prior["status"] != "complete"
            stage_retry_data = self._checked_retry(self.stage_retry, stage_previous,
                required=stage_blocked, missing="PREVIOUS_PRIMITIVE_STAGE_REQUIRES_REVIEW",
                unexpected="PRIMITIVE_STAGE_RETRY_WITHOUT_PREVIOUS_FAILURE")
            sources = {}
            for role in ("design", "spice", "runtime"):
                if self.source_paths[role] is None:
                    sources[role] = None
                    continue
                path = self.source_paths[role].resolve(strict=True)
                if role in {"design", "spice"} and not path.is_relative_to(self.root):
                    raise ValueError("primitive source is outside the task project")
                sources[role] = _identity(path, checkpoint=self.checkpoint)
            sources["dependencies"] = [_identity(path.resolve(strict=True), checkpoint=self.checkpoint)
                                       for path in self.source_paths["dependencies"]]
            if len({row["path"] for row in sources["dependencies"]}) != len(sources["dependencies"]):
                raise ValueError("duplicate primitive stage dependency")
            runtime = None
            if sources["runtime"] is not None:
                import re
                self.checkpoint()
                runtime_bytes = Path(sources["runtime"]["path"]).read_bytes()
                self.checkpoint()
                if sha256(runtime_bytes).hexdigest() != sources["runtime"]["sha256"]:
                    raise ValueError("primitive stage runtime changed")
                runtime = json.loads(runtime_bytes)
                if (runtime.get("schema") != "clay.native-physical-runtime/v1"
                        or not isinstance(runtime.get("image"), str)
                        or not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["image"])
                        or not isinstance(runtime.get("pdk_root"), str)
                        or not runtime["pdk_root"].startswith("/")
                        or type(runtime.get("timeout_seconds")) not in (int, float)
                        or not math.isfinite(runtime["timeout_seconds"])
                        or runtime["timeout_seconds"] <= 0):
                    raise ValueError("unsupported primitive stage runtime")
            self.checkpoint()
            self.attempt_dir.mkdir(parents=True, exist_ok=production_observation is not None)
            sources_path = self.attempt_dir / "stage-sources.json"
            atomic_json(sources_path, sources)
            sources_identity = _identity(sources_path)
            start = {"schema": "clay.primitive-stage-start/v1", "task_id": self.task_id,
                     "invocation_id": self.invocation_id, "scope": "primitive-only",
                     "physical_admission": False, "managed": self.managed, "budget": self.budget.to_dict(),
                     "unwind_budget_seconds": self.unwind_budget_seconds,
                     "previous_production_receipt": previous, "production_retry": retry_data,
                     "previous_stage_receipt": stage_previous, "stage_retry": stage_retry_data,
                     "source_bindings": sources_identity}
            if production_observation is not None:
                start['production_quiescence'] = production_observation
            if self.role_trace_request is not None:
                start["schema"] = "clay.primitive-stage-start/v2"
                start["role_trace_request"] = self.role_trace_request
            start_path = self.attempt_dir / "stage-start.json"
            self.checkpoint()
            atomic_json(start_path, start)
            self._stage_start_identity = _identity(start_path)
            self._stage_sources = sources
            self._runtime = runtime
            stage.update(attempts=stage["attempts"] + 1,
                         active={"invocation_id": self.invocation_id,
                                 "start": self._stage_start_identity, "work_dir": None, "plan": None})
            self.checkpoint()
            atomic_json(state_path, self.state)
            self._token = _CURRENT.set(self)
            return self
        except BaseException:
            self._lock.close()
            self._lock = None
            raise

    def _checked_retry(self, proposal, previous, *, required, missing, unexpected):
        if required:
            if not isinstance(proposal, RetryRequest):
                raise RunAdmissionError(missing, task_id=self.task_id, receipt=previous)
            if (previous is None or proposal.prior_receipt_sha256 != previous["sha256"]
                    or _identity(Path(previous["path"]), checkpoint=self.checkpoint) != previous):
                raise RunAdmissionError("RETRY_RECEIPT_MISMATCH")
            return {"prior_receipt_sha256": proposal.prior_receipt_sha256,
                    "reason": proposal.reason, "evidence": [_identity(p, checkpoint=self.checkpoint) for p in proposal.evidence],
                    "evidence_verified": False}
        if proposal is not None:
            raise RunAdmissionError(unexpected)
        return None

    def bind_work_dir(self, work_dir, *, execution):
        from clay.flow.execution_session import ExecutionSession
        if self._finished or self._token is None:
            raise ValueError("primitive stage is not active")
        if (not isinstance(execution, ExecutionSession) or execution.budget != self.budget
                or execution.cancellation is not self.cancellation):
            raise ValueError("primitive stage execution session differs")
        active = self.state["primitive_stage"]["active"]
        if active["work_dir"] is not None or active["plan"] is not None:
            raise ValueError("primitive stage work directory already bound")
        work = Path(work_dir)
        if not work.is_absolute() or not work.is_dir():
            raise ValueError("primitive stage work directory is not an absolute directory")
        resolved = work.resolve(strict=True)
        if str(resolved) != str(work):
            raise ValueError("primitive stage work directory is not canonical")
        if any(resolved.iterdir()):
            raise ValueError("primitive stage requires a fresh work directory")
        active["work_dir"] = str(resolved)
        atomic_json(self.directory / "state.json", self.state)
        self._execution = execution
        return str(resolved)

    def checkpoint(self):
        from clay.flow.execution_budget import ExecutionStopped
        if self._execution is not None:
            self._execution.checkpoint()
        if self.cancellation.reason is not None:
            raise ExecutionStopped("EXECUTION_CANCELLED", task_id=self.task_id,
                                   invocation_id=self.invocation_id,
                                   cancellation_reason=self.cancellation.reason)
        elapsed = time.monotonic() - self._start
        if self.role_trace_request is not None and elapsed >= self.role_trace_request['wall_seconds']:
            raise ExecutionStopped("WALL_BUDGET_EXCEEDED", task_id=self.task_id,
                                   invocation_id=self.invocation_id, scope='primitive-role-trace',
                                   limit_seconds=self.role_trace_request['wall_seconds'], consumed_seconds=elapsed)
        if elapsed >= self.budget.wall_seconds:
            raise ExecutionStopped("WALL_BUDGET_EXCEEDED", task_id=self.task_id,
                                   invocation_id=self.invocation_id,
                                   limit_seconds=self.budget.wall_seconds, consumed_seconds=elapsed)

    def remaining_wall_seconds(self):
        self.checkpoint()
        remaining = self.budget.wall_seconds - (time.monotonic() - self._start)
        if self.role_trace_request is not None:
            remaining = min(remaining, self.role_trace_request['wall_seconds'] - (time.monotonic() - self._start))
        if self._execution is not None:
            session_remaining = self._execution.remaining_wall_seconds()
            if session_remaining is not None:
                remaining = min(remaining, session_remaining)
        if remaining <= 0:
            self.checkpoint()
            from clay.flow.execution_budget import ExecutionStopped
            raise ExecutionStopped("WALL_BUDGET_EXCEEDED", task_id=self.task_id,
                                   invocation_id=self.invocation_id,
                                   limit_seconds=self.budget.wall_seconds,
                                   consumed_seconds=time.monotonic() - self._start)
        return remaining

    @property
    def has_resource_plan(self):
        return self._stage_plan_identity is not None

    def record_primitive_observation(self, work_dir, manifest, identity, *, checkpoint):
        if not self.has_resource_plan:
            raise ValueError("primitive capture requires a bound resource plan")
        return super().record_primitive_observation(work_dir, manifest, identity, checkpoint=checkpoint)

    def record_primitive_observation_unknown(self, work_dir, provider, *, checkpoint):
        if not self.has_resource_plan:
            raise ValueError("primitive observation requires a bound resource plan")
        return super().record_primitive_observation_unknown(work_dir, provider, checkpoint=checkpoint)

    def bind_resource_plan(self, plan):
        if self._finished or self._token is None or self._stage_plan is not None:
            raise ValueError("primitive resource plan is not available")
        self.checkpoint()
        if self.production_quiescence is not None:
            import os
            if (os.environ.get('DOCKER_HOST') != self.production_quiescence.endpoint
                    or os.environ.get('DOCKER_CONTEXT')):
                raise ValueError('stage Docker environment changed after current observation')
            _, stage_start = _stage_document(self, self._stage_start_identity,
                                              filename='stage-start.json', checkpoint=self.checkpoint)
            if plan.get('docker_cli') != stage_start['production_quiescence']['docker_cli']:
                raise ValueError('stage Docker executable differs from current observation')
        if self._runtime is None:
            raise ValueError("primitive resource plan requires a source-bound runtime")
        bound_work = self.state["primitive_stage"]["active"]["work_dir"]
        if bound_work is None:
            raise ValueError("primitive stage requires a bound work directory")
        required = {"schema", "invocation_id", "name", "nonce", "image", "work_dir",
                    "runtime_sha256", "pdk_root", "network", "read_only_root", "timeout_seconds", "cleanup_seconds",
                    "inputs_sha256", "docker_cli", "runtime_contract_sha256", "tmpfs"}
        if self.role_trace_request is not None:
            required.add('role_trace_request')
        if not isinstance(plan, dict) or set(plan) != required:
            raise ValueError("invalid primitive resource plan")
        timeout, cleanup = plan["timeout_seconds"], plan["cleanup_seconds"]
        if (plan["schema"] != ("clay.native-container-plan/v2" if self.role_trace_request is not None
                              else "clay.native-container-plan/v1")
                or (self.role_trace_request is not None and plan['role_trace_request'] != self.role_trace_request)
                or plan["invocation_id"] != self.invocation_id
                or plan["nonce"] != self.invocation_id
                or plan["name"] != "clay-primitive-" + self.invocation_id
                or plan["image"] != self._runtime["image"]
                or plan["runtime_sha256"] != self._stage_sources["runtime"]["sha256"]
                or plan["runtime_contract_sha256"] != _runtime_contract_sha256(self._runtime)
                or plan["pdk_root"] != self._runtime.get("pdk_root")
                or plan["network"] != "none" or plan["read_only_root"] is not True
                or plan["tmpfs"] != {"/tmp": "rw"}
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in (timeout, cleanup))
                or timeout > self.budget.wall_seconds
                or (self.role_trace_request is not None and timeout > self.role_trace_request['wall_seconds'])
                or timeout > self._runtime["timeout_seconds"]):
            raise ValueError("primitive resource plan differs from the source or budget")
        work = Path(plan["work_dir"])
        if not work.is_absolute() or not work.is_dir():
            raise ValueError("primitive resource work directory is not an existing absolute directory")
        work = work.resolve(strict=True)
        if str(work) != str(plan["work_dir"]):
            raise ValueError("primitive resource work directory is not canonical")
        if work.parent != Path(bound_work):
            raise ValueError("primitive resource work directory differs from the active invocation")
        _owned_path(Path(bound_work), work.name)
        _validate_stage_plan_inputs(plan, checkpoint=self.checkpoint)
        path = self.attempt_dir / "native-container-plan.json"
        atomic_json(path, plan)
        identity = _identity(path)
        self.state["primitive_stage"]["active"]["plan"] = identity
        atomic_json(self.directory / "state.json", self.state)
        self._stage_plan, self._stage_plan_identity = dict(plan), identity
        return dict(identity)

    def finish(self, *, passed, diagnostic):
        raise ValueError("primitive stage cannot publish a production terminal")

    def _verify_sources(self, checkpoint):
        checkpoint()
        _, start = _stage_document(self, self._stage_start_identity, filename="stage-start.json", checkpoint=checkpoint)
        bindings = _validate_stage_source_bindings(self, start, checkpoint=checkpoint)
        if bindings != self._stage_sources:
            raise ValueError("primitive stage source bindings changed")
        checkpoint()

    def read_bound_design_bytes(self):
        """Read and verify the exact design bytes consumed by this stage."""
        if (self._finished or self._token is None or self._lock is None
                or _CURRENT.get() is not self or self._stage_sources is None
                or self.state["primitive_stage"]["active"]["invocation_id"] != self.invocation_id):
            raise ValueError("primitive stage is not active")
        identity = self._stage_sources["design"]
        path = Path(identity["path"])
        digest, size, chunks = sha256(), 0, []
        self.checkpoint()
        with path.open("rb") as stream:
            while True:
                self.checkpoint()
                block = stream.read(min(1024 * 1024, MAX_EVIDENCE_BYTES + 1 - size))
                if not block:
                    break
                size += len(block)
                if size > MAX_EVIDENCE_BYTES:
                    raise RunAdmissionError("ADMISSION_EVIDENCE_TOO_LARGE", path=str(path),
                                            max_bytes=MAX_EVIDENCE_BYTES)
                digest.update(block)
                chunks.append(block)
        self.checkpoint()
        if size != identity["size_bytes"] or digest.hexdigest() != identity["sha256"]:
            raise ValueError("primitive stage design changed")
        return b"".join(chunks)

    def _finish_with_stop(self, *, failed, stop, checkpoint, operation,
                          original_error=None, prior_error_diagnostic=None):
        from clay.flow.control_stop import record_unwind_failure
        from clay.flow.execution_budget import ExecutionStopped
        if stop is None:
            def check():
                self.checkpoint()
                checkpoint()
            if not failed or original_error is None or self.role_trace_request is None:
                return operation(check)
            try:
                return operation(check)
            except ExecutionStopped as observed_after_error:
                if isinstance(prior_error_diagnostic, dict):
                    prior_error_diagnostic['post_error_unwind_trigger'] = observed_after_error.flow_error_data()
                try:
                    if self.unwind_budget_seconds is None:
                        raise TimeoutError('PRIMITIVE_STAGE_UNWIND_INCOMPLETE: no unwind budget')
                    deadline = time.monotonic() + self.unwind_budget_seconds
                    def unwind_check():
                        if time.monotonic() >= deadline:
                            raise TimeoutError('PRIMITIVE_STAGE_UNWIND_INCOMPLETE: unwind budget exhausted')
                    return operation(unwind_check)
                except BaseException as ending:
                    record_unwind_failure(original_error, ending)
                    raise original_error
            except Exception as ending:
                record_unwind_failure(original_error, ending)
                raise original_error
        if not failed or not isinstance(stop, ExecutionStopped):
            raise ValueError("primitive unwind requires a typed failure stop")
        try:
            if self.unwind_budget_seconds is None:
                raise TimeoutError("PRIMITIVE_STAGE_UNWIND_INCOMPLETE: no unwind budget")
            deadline = time.monotonic() + self.unwind_budget_seconds
            def check():
                if time.monotonic() >= deadline:
                    raise TimeoutError("PRIMITIVE_STAGE_UNWIND_INCOMPLETE: unwind budget exhausted")
            return operation(check)
        except BaseException as ending:
            record_unwind_failure(stop, ending)
            raise stop

    def finish_stage(self, *, passed, diagnostic, evidence_path, checkpoint,
                     managed_terminal_path=None, stop=None, original_error=None):
        from clay.flow.execution_budget import ExecutionStopped
        if type(passed) is not bool or self._finished or self._token is None or self._stage_plan is None:
            raise ValueError("primitive stage is not ready to finish")
        if stop is not None:
            if passed or not isinstance(stop, ExecutionStopped):
                raise ValueError("primitive unwind requires a typed failure stop")
            diagnostic = {"diagnostic": diagnostic, "execution_stop": stop.flow_error_data()}
        def operation(check):
            self._finish_stage_verified(passed=passed, diagnostic=diagnostic,
                                        evidence_path=evidence_path, check=check,
                                        managed_terminal_path=managed_terminal_path)
        return self._finish_with_stop(failed=not passed, stop=stop, checkpoint=checkpoint,
                                      operation=operation, original_error=original_error,
                                      prior_error_diagnostic=diagnostic)

    def _finish_stage_verified(self, *, passed, diagnostic, evidence_path, check,
                               managed_terminal_path):
        from clay.flow.execution_session import _bounded_snapshot
        from clay.flow.native_owned_container import verify_native_container_evidence
        check()
        self._verify_sources(check)
        _validate_stage_plan_inputs(self._stage_plan, checkpoint=check)
        evidence_path = Path(evidence_path).resolve(strict=True)
        if (evidence_path.parent != Path(self._stage_plan["work_dir"])
                or evidence_path.name != "native-supervision.json"):
            raise ValueError("primitive stage supervision is outside its work directory")
        evidence = _identity(evidence_path, checkpoint=check)
        verified = verify_native_container_evidence(evidence_path, plan=self._stage_plan,
                    plan_identity=self._stage_plan_identity, checkpoint=check,
                    expected_identity=evidence)
        verified = json.loads(json.dumps(verified, allow_nan=False, sort_keys=True))
        if _identity(evidence_path, checkpoint=check) != evidence:
            raise ValueError("primitive supervision changed during verification")
        if passed and self._primitive_observation_head is None:
            raise ValueError("primitive stage success requires a source-bound capture")
        if passed and verified.get("outcome") != "COMPLETE":
            raise ValueError("primitive stage cannot promote a failed producer")
        trace_identity = None
        if passed:
            trace_identity = _validate_stage_capture(self, self.state["primitive_stage"]["active"],
                                    self._primitive_observation_head, self._stage_plan,
                                    evidence_identity=evidence, checkpoint=check)
        managed_terminal = (None if managed_terminal_path is None
                            else _identity(Path(managed_terminal_path).resolve(strict=True), checkpoint=check))
        _validate_stage_managed_terminal(self, self.state["primitive_stage"]["active"],
            self._primitive_observation_head, managed_terminal, "complete" if passed else "failed", checkpoint=check)
        check()
        self._publish_stage("complete" if passed else "failed",
            diagnostic=_bounded_snapshot(diagnostic, 16 * 1024), evidence=evidence, verified=verified,
            managed_terminal=managed_terminal, checkpoint=check, role_trace_manifest=trace_identity)

    def finish_without_launch_unknown(self, *, diagnostic, failed=False, managed_terminal_path=None,
                                      checkpoint=lambda: None, stop=None, original_error=None):
        from clay.flow.execution_budget import ExecutionStopped
        if (type(failed) is not bool or self._finished or self._token is None
                or self._stage_plan is not None or self._primitive_observation_head is not None):
            raise ValueError("primitive stage cannot claim no launch after binding a plan")
        if stop is not None:
            if not failed or not isinstance(stop, ExecutionStopped):
                raise ValueError("primitive unwind requires a typed failure stop")
            diagnostic = {"diagnostic": diagnostic, "execution_stop": stop.flow_error_data()}
        def operation(check):
            self._finish_without_launch_verified(diagnostic=diagnostic, failed=failed,
                                                  managed_terminal_path=managed_terminal_path,
                                                  check=check)
        return self._finish_with_stop(failed=failed, stop=stop, checkpoint=checkpoint,
                                      operation=operation, original_error=original_error,
                                      prior_error_diagnostic=diagnostic)

    def _finish_without_launch_verified(self, *, diagnostic, failed, managed_terminal_path, check):
        from clay.flow.execution_session import _bounded_snapshot
        self._verify_sources(check)
        managed_terminal = (None if managed_terminal_path is None
                            else _identity(Path(managed_terminal_path).resolve(strict=True), checkpoint=check))
        status = "failed" if failed else "unknown"
        _validate_stage_managed_terminal(self, self.state["primitive_stage"]["active"],
                                         None, managed_terminal, status, checkpoint=check)
        check()
        self._publish_stage(status, diagnostic=_bounded_snapshot(diagnostic, 16 * 1024),
                            evidence=None, verified=None, managed_terminal=managed_terminal, checkpoint=check)

    def _publish_stage(self, status, *, diagnostic, evidence, verified, managed_terminal, checkpoint,
                       role_trace_manifest=None):
        _, start = _stage_document(self, self._stage_start_identity, filename="stage-start.json", checkpoint=checkpoint)
        elapsed = time.monotonic() - self._start
        receipt = {"schema": "clay.primitive-stage-receipt/v1", "task_id": self.task_id,
                   "invocation_id": self.invocation_id, "scope": "primitive-only",
                   "status": status, "verdict": "UNKNOWN", "wall_seconds": elapsed, "diagnostic": diagnostic,
                   "physical_admission": False, "managed": self.managed, "start": self._stage_start_identity,
                   "resource_plan": self._stage_plan_identity, "resource_evidence": evidence,
                   "resource_verification": verified, "managed_terminal": managed_terminal,
                   "previous_production_receipt": start["previous_production_receipt"],
                   "previous_stage_receipt": start["previous_stage_receipt"]}
        if 'production_quiescence' in start:
            receipt['production_quiescence'] = start['production_quiescence']
        if self.role_trace_request is not None:
            receipt["schema"] = "clay.primitive-stage-receipt/v2"
            receipt["role_trace_request"] = self.role_trace_request
            receipt["role_trace_manifest"] = role_trace_manifest
        if self._primitive_observation_head:
            receipt["primitive_observation_index"] = self.primitive_observation_head()
        if self.receipt_path.exists():
            raise RuntimeError("primitive stage terminal already published")
        atomic_json(self.receipt_path, receipt)
        self._finished = True
        stage = self.state["primitive_stage"]
        stage.update(last_receipt=_identity(self.receipt_path), active=None,
                     total_wall_seconds=stage["total_wall_seconds"] + elapsed,
                     failed_attempts=stage["failed_attempts"] + (status != "complete"))
        atomic_json(self.directory / "state.json", self.state)
        self.control = {"schema": "clay.primitive-stage-control/v1", "task_id": self.task_id,
                        "status": status, "receipt": stage["last_receipt"],
                        "production_receipt": self.state.get("last_receipt"),
                        "physical_admission": False, "budget": self.budget.to_dict()}
        if 'production_quiescence' in start:
            self.control['production_quiescence_scope'] = {
                'endpoint': start['production_quiescence']['endpoint'],
                'output_roots': start['production_quiescence']['output_roots'],
                'historical_cleanup': 'UNKNOWN', 'other_endpoints': 'UNKNOWN',
                'host_processes': 'UNKNOWN', 'scope_complete': 'UNKNOWN'}

    def __exit__(self, exc_type, exc, tb):
        # Without a verified terminal, leave the active record in place.  The
        # next task owner must not infer that a Docker daemon has stopped.
        try:
            if not self._finished and exc is None:
                raise RunAdmissionError("PRIMITIVE_STAGE_RECOVERY_REQUIRED",
                                        invocation_id=self.invocation_id)
        finally:
            if self._token is not None:
                _CURRENT.reset(self._token)
            self._lock.close()
