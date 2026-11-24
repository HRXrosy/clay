"""Owned run directories and byte-bound recovery, independent of flow steps.

The installed caller declares the dependency closure. This module verifies it;
it neither discovers PDK dependencies nor treats an old result as executable state.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4


def _identifier(value: str) -> None:
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                *(f"LPT{i}" for i in range(10))}
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value)
            or value.upper() in reserved):
        raise ValueError("INVALID_RUN_IDENTIFIER: use a portable alphanumeric name")


def json_snapshot(value: Any) -> Any:
    """Copy supported declaration data, rejecting arbitrary live Python objects."""
    if is_dataclass(value) and not isinstance(value, type):
        value = {f.name: json_snapshot(getattr(value, f.name)) for f in fields(value)}
    elif isinstance(value, Path):
        value = str(value.resolve())
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("run metadata keys must be strings")
        value = {key: json_snapshot(item) for key, item in value.items()}
    elif isinstance(value, (tuple, list)):
        value = [json_snapshot(item) for item in value]
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError(f"unsupported run metadata: {type(value).__name__}")
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def acquire_file_lock(path: Path):
    """Own a nonblocking process lock; closing the handle releases ownership."""
    lock = path.open("a+b")
    if lock.seek(0, os.SEEK_END) == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise ValueError("RUN_BUSY: another invocation owns this run") from exc
    return lock


def _contained(base: Path, path: Path, reason: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(reason + ": " + str(path))
    return resolved


def _owned_path(base: Path, relative: str) -> Path:
    candidate = base / relative
    _contained(base, candidate, "ARTIFACT_OUTSIDE_ATTEMPT")
    current = candidate
    while current != base:
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError("RUN_PATH_ALIAS: " + str(current))
        current = current.parent
    return candidate


def _file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()  # Missing and unreadable dependencies fail closed.
    return {"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}


def artifact_identity(work_dir: Path, raw_path: str | Path) -> dict[str, Any]:
    path = Path(raw_path)
    if not path.is_absolute():
        path = work_dir / path
    _contained(work_dir, path, "ARTIFACT_OUTSIDE_ATTEMPT")
    _owned_path(work_dir, str(path.relative_to(work_dir)))
    path = path.resolve()
    if path.stat().st_nlink != 1:
        raise ValueError("RUN_PATH_ALIAS: artifact has multiple hard links")
    return {"path": path.relative_to(work_dir).as_posix(), **_file_identity(path)}


@dataclass(frozen=True)
class InputBinding:
    role: str
    path: Path

    def __post_init__(self) -> None:
        if self.role not in {"source", "reference", "implementation", "pdk"}:
            raise ValueError("UNKNOWN_BINDING_ROLE: " + str(self.role))
        object.__setattr__(self, "path", Path(self.path).absolute())


@dataclass(frozen=True)
class RunRequest:
    project_id: str
    run_id: str
    project_root: Path
    files: tuple[InputBinding, ...]
    tool_versions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _identifier(self.project_id)
        _identifier(self.run_id)
        root = Path(self.project_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("project_root must be a directory")
        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "tool_versions", tuple(tuple(pair) for pair in self.tool_versions))
        roles = {entry.role for entry in self.files}
        if not {"source", "implementation", "pdk"}.issubset(roles):
            raise ValueError("BINDING_ROLE_MISSING: source, implementation and pdk are required")
        if len({(e.role, e.path) for e in self.files}) != len(self.files):
            raise ValueError("DUPLICATE_INPUT_BINDING")
        for entry in self.files:
            if entry.role in {"source", "reference"}:
                _contained(root, entry.path, "INPUT_OUTSIDE_PROJECT")
        if (not self.tool_versions or any(len(pair) != 2 or not all(isinstance(v, str) and v for v in pair)
                                         for pair in self.tool_versions)
                or len({pair[0] for pair in self.tool_versions}) != len(self.tool_versions)):
            raise ValueError("TOOL_IDENTITY_REQUIRED: distinct tool names and versions")

    def snapshot(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id, "run_id": self.run_id,
            "project_root": str(self.project_root), "tool_versions": sorted(self.tool_versions),
            "files": [{"role": e.role, "path": str(e.path), "resolved_path": str(e.path.resolve()),
                       **_file_identity(e.path)}
                      for e in sorted(self.files, key=lambda e: (e.role, str(e.path)))],
        }


class RunWorkspace:
    """One exclusive run, one fresh attempt, one atomic terminal publication.

    Resume verifies the declared world and every previously committed artifact.
    Partial directories remain evidence; no cached status is promoted to PASS.
    """

    def __init__(self, request: RunRequest, out_dir: str | Path, *,
                 configuration: Mapping[str, Any], resume: bool = False):
        self.request = request
        self.configuration = json_snapshot(configuration)
        self.out_dir = Path(out_dir).resolve()
        self.resume = resume
        self.invocation_id = uuid4().hex
        self.run_dir = self.out_dir / request.project_id / request.run_id
        self.attempt_dir = self.run_dir / "attempts" / self.invocation_id
        self.work_dir = self.attempt_dir / "work"
        self.event_path = self.attempt_dir / "events.jsonl"
        self._lock = None
        self._entered = False
        self._finished = False
        self._identity: dict[str, Any] = {}

    def _acquire(self) -> None:
        lock_path = _owned_path(self.run_dir, "owner.lock")
        self._lock = acquire_file_lock(lock_path)

    def _release(self) -> None:
        if self._lock is not None:
            self._lock.close()  # OS lock release also works after process death.
            self._lock = None

    def __enter__(self) -> RunWorkspace:
        if self._entered:
            raise ValueError("workspace cannot be entered twice")
        try:
            self._identity = json_snapshot({"request": self.request.snapshot(), "configuration": self.configuration})
        except OSError as exc:
            if self.resume:
                raise ValueError("RUN_IDENTITY_CHANGED: unreadable bound file") from exc
            raise
        _owned_path(self.out_dir, str(self.run_dir.relative_to(self.out_dir)))
        if self.resume:
            if not self.run_dir.is_dir():
                raise ValueError("RUN_NOT_FOUND")
        else:
            self.run_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.run_dir.mkdir()
            except FileExistsError as exc:
                raise ValueError("RUN_ALREADY_EXISTS: use explicit resume") from exc
        self._acquire()
        try:
            manifest_path = _owned_path(self.run_dir, "run.json")
            if self.resume:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema") != "clay.run/v1" or manifest.get("identity") != self._identity:
                    raise ValueError("RUN_IDENTITY_CHANGED")
                self._verify_artifacts(manifest["attempts"])
            else:
                manifest = {"schema": "clay.run/v1", "identity": self._identity, "attempts": []}
            _owned_path(self.run_dir, str(self.work_dir.relative_to(self.run_dir)))
            self.work_dir.mkdir(parents=True)
            manifest["attempts"].append(self.invocation_id)
            atomic_json(manifest_path, manifest)
            self._entered = True
            return self
        except BaseException:
            self._release()
            raise

    def _verify_artifacts(self, attempts: list[str]) -> None:
        for attempt in attempts:
            _identifier(attempt)
            directory = _owned_path(self.run_dir, "attempts/" + attempt)
            terminal_path = _owned_path(directory, "terminal.json")
            if not terminal_path.exists():
                continue  # process died before terminal publication; never cache it
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            if terminal.get("schema") != "clay.run-terminal/v1" or terminal.get("invocation_id") != attempt:
                raise ValueError("INVALID_RUN_TERMINAL")
            for artifact in [*terminal["artifacts"], *terminal.get("observations", [])]:
                path = _owned_path(directory, artifact["path"])
                try:
                    matches = _file_identity(path) == {k: artifact[k] for k in ("sha256", "size_bytes")}
                except OSError:
                    matches = False
                if not matches:
                    raise ValueError("RUN_ARTIFACT_CHANGED: " + str(path))

    def verify_identity(self) -> None:
        try:
            current = json_snapshot({"request": self.request.snapshot(), "configuration": self.configuration})
        except OSError as exc:
            raise ValueError("RUN_IDENTITY_CHANGED: unreadable bound file") from exc
        if current != self._identity:
            raise ValueError("RUN_IDENTITY_CHANGED: bound file changed during execution")

    def finish(self, *, status: str, verdict: str, artifacts: Mapping[str, str | Path],
               expected_artifacts: Mapping[str, Any] | None = None, checkpoint=None) -> None:
        if not self._entered or self._finished:
            raise ValueError("run attempt is not open")
        if status not in {"complete", "failed", "cancelled", "timeout"}:
            raise ValueError("invalid terminal status")
        self.verify_identity()
        published = []
        if status == "complete" and verdict == "pass":
            for name, raw_path in sorted(artifacts.items()):
                identity = artifact_identity(self.work_dir, raw_path)
                if expected_artifacts is not None and name in expected_artifacts and identity != expected_artifacts[name]:
                    raise ValueError("RUN_ARTIFACT_CHANGED: completed stage output was modified: " + name)
                published.append({**identity, "name": name, "path": "work/" + identity["path"]})
        self._terminal(status, verdict, published, checkpoint=checkpoint)

    def stop(self, reason: str) -> None:
        """Publish a typed unsuccessful stop without admitting success artifacts."""
        if not self._entered or self._finished:
            raise ValueError("run attempt is not open")
        status = "cancelled" if reason == "EXECUTION_CANCELLED" else "timeout"
        self._terminal(status, "fail", [], reason)

    def _terminal(self, status: str, verdict: str, artifacts: list[dict[str, Any]], error: str = "", checkpoint=None) -> None:
        observations = []
        if self.event_path.is_file():
            observations.append({"name": "events", "path": "events.jsonl", **_file_identity(self.event_path)})
        # Partial group-route files are observations, not failed-run admission artifacts.
        if (self.work_dir / "group-fallback").is_dir():
            from clay.layout.route.group_fallback_artifacts import observations as group_observations
            for name, path in group_observations(self.work_dir).items():
                identity = artifact_identity(self.work_dir, path)
                observations.append({**identity, "name": name,
                                     "path": "work/" + identity["path"]})
        if checkpoint is not None:
            checkpoint()  # Success publication admission follows all potentially large hashes.
        from clay.flow.run_admission import current_admission
        admission = current_admission()
        primitive_head = admission.primitive_observation_head() if admission else None
        terminal = {
            "schema": "clay.run-terminal/v1", "project_id": self.request.project_id,
            "run_id": self.request.run_id, "invocation_id": self.invocation_id,
            "status": status, "verdict": verdict, "artifacts": artifacts,
            "observations": observations, "error": error,
        }
        if primitive_head:
            terminal["primitive_observation_index"] = {
                "head": primitive_head, "work_dir": str(self.work_dir.resolve()),
                "scope": "only records for this work_dir through this head",
            }
        atomic_json(self.attempt_dir / "terminal.json", terminal)
        self._finished = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._entered and not self._finished:
                self._terminal("error" if exc is not None else "interrupted", "fail", [], str(exc or ""))
        finally:
            self._release()
