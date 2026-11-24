"""Portable run archive writer for Clay design demonstrations.

Live-tail convention
--------------------
``event()`` opens, writes, flushes, fsyncs, and closes ``events.jsonl`` for
every event. That makes each newly appended line visible to an external
tail-reader (``clay.runtime.live_server``) within milliseconds, with no
shared file handle to coordinate. Manifest writes are also full-file
rewrites, so a live reader always sees a complete document.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


Clock = Callable[[], str]
VIEWER_CONTRACT = {
    "schema": "clay.viewer.run_archive/v1",
    "artifacts": ["schematic", "layout", "layout_json", "kpis", "gates"],
}
_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


class RunRecorder:
    """Write a browser-readable Clay run archive.

    The archive is intentionally file-based: a stable manifest, an append-only
    JSONL event stream, and final artifacts under ``final/``. That keeps live
    monitoring and offline replay on the same contract.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        design_id: str,
        display_name: str | None = None,
        cell_type: str | None = None,
        schema_version: int = 1,
        viewer_contract: dict[str, Any] | bool | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.final_dir = self.run_dir / "final"
        self.manifest_path = self.run_dir / "run.manifest.json"
        self.events_path = self.run_dir / "events.jsonl"
        self._clock = clock or _utc_now
        self._manifest: dict[str, Any] = {
            "schema_version": schema_version,
            "design_id": design_id,
            "display_name": display_name or design_id,
            "cell_type": cell_type,
            "status": "created",
            "started_at": None,
            "finished_at": None,
            "final_revision": None,
            "events": "events.jsonl",
            "artifacts": {},
        }
        if viewer_contract:
            self._manifest["viewer_contract"] = dict(
                VIEWER_CONTRACT if viewer_contract is True else viewer_contract
            )

    @classmethod
    def attach(
        cls,
        run_dir: str | Path,
        *,
        clock: Clock | None = None,
    ) -> "RunRecorder":
        """Open an existing run archive without truncating its event stream."""
        root = Path(run_dir)
        manifest_path = root / "run.manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"run manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorder = cls(
            root,
            design_id=str(manifest.get("design_id") or root.name),
            display_name=manifest.get("display_name"),
            cell_type=manifest.get("cell_type"),
            schema_version=int(manifest.get("schema_version", 1)),
            viewer_contract=manifest.get("viewer_contract"),
            clock=clock,
        )
        recorder._manifest = manifest
        recorder.run_dir.mkdir(parents=True, exist_ok=True)
        recorder.final_dir.mkdir(parents=True, exist_ok=True)
        if not recorder.events_path.exists():
            recorder.events_path.write_text("", encoding="utf-8")
        return recorder

    def start(
        self,
        *,
        clay_source: str | None = None,
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")
        self._manifest["status"] = status
        self._manifest["started_at"] = self._clock()
        if clay_source is not None:
            self._manifest["clay_source"] = clay_source
        if metadata is not None:
            self._manifest["metadata"] = metadata
        self._write_manifest()

    def event(
        self,
        kind: str,
        *,
        summary: str = "",
        revision: str | None = None,
        targets: list[str] | None = None,
        artifacts: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        t: str | None = None,
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        item: dict[str, Any] = {
            "t": t or self._clock(),
            "kind": kind,
            "summary": summary,
        }
        if revision is not None:
            item["revision"] = revision
        if targets is not None:
            item["targets"] = targets
        if artifacts is not None:
            item["artifacts"] = _validate_artifact_map(artifacts)
        if data is not None:
            item["data"] = data
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except (OSError, ValueError):
                # fsync may be unavailable on some streams (e.g. virtual FS in
                # tests); the explicit flush above is sufficient for tailers.
                pass

    def copy_final_artifact(
        self,
        kind: str,
        src: str | Path,
        *,
        filename: str | None = None,
    ) -> Path:
        source = Path(src)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        dest = self.final_dir / _safe_final_filename(filename or source.name)
        shutil.copyfile(source, dest)
        self._record_artifact(kind, dest)
        return dest

    def write_final_json(
        self,
        kind: str,
        payload: dict[str, Any] | list[Any],
        *,
        filename: str | None = None,
    ) -> Path:
        self.final_dir.mkdir(parents=True, exist_ok=True)
        dest = self.final_dir / _safe_final_filename(filename or f"{kind}.json")
        dest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._record_artifact(kind, dest)
        return dest

    def finalize(self, *, final_revision: str = "final", status: str = "passed") -> None:
        self._manifest["status"] = status
        self._manifest["final_revision"] = final_revision
        self._manifest["finished_at"] = self._clock()
        self._write_manifest()

    def _record_artifact(self, kind: str, dest: Path) -> None:
        try:
            rel = dest.resolve().relative_to(self.run_dir.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"unsafe artifact path for {kind}: {dest}") from exc
        if not _is_safe_relative_artifact_path(rel):
            raise ValueError(f"unsafe artifact path for {kind}: {rel!r}")
        self._manifest.setdefault("artifacts", {})[kind] = rel
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self._manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_artifact_map(artifacts: dict[str, str]) -> dict[str, str]:
    checked: dict[str, str] = {}
    for kind, value in artifacts.items():
        key = str(kind)
        if not _is_safe_relative_artifact_path(value):
            raise ValueError(f"unsafe artifact path for {key}: {value!r}")
        checked[key] = value
    return checked


def _safe_final_filename(filename: str) -> str:
    if not filename or "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise ValueError(f"unsafe artifact filename: {filename!r}")
    if any(ord(ch) < 32 for ch in filename) or _URL_SCHEME_RE.match(filename):
        raise ValueError(f"unsafe artifact filename: {filename!r}")
    return filename


def _is_safe_relative_artifact_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if any(ord(ch) < 32 for ch in value):
        return False
    if value.startswith(("/", "\\", "//", "\\\\")):
        return False
    if "\\" in value:
        return False
    if _URL_SCHEME_RE.match(value):
        return False
    parts = value.replace("\\", "/").split("/")
    return all(part not in {"", ".", ".."} for part in parts)
