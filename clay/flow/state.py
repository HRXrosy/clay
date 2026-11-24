"""Persisted per-step flow state for incremental / resumable re-runs.

The state file records, for each step, the SHA-256 of the union of its declared
``inputs()`` plus the status it produced. ``should_skip`` compares the freshly
hashed inputs against the recorded hash: a step whose inputs are byte-identical
and previously passed/skipped can be skipped on the next run, exactly like a
build cache (skip PEX if the GDS is unchanged, etc.).

The state is a plain JSON map ``{step_name: {"input_sha": str, "status": str}}``
written under the run work directory; nothing here is circuit-specific.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


STATE_FILENAME = "flow_state.json"

#: Statuses that mean "this step is done and need not be re-run if its inputs
#: are unchanged". A prior ``fail`` is NOT cached: a failing step is always
#: retried so a fix to an upstream dependency can re-green it.
_SKIPPABLE_STATUSES = ("pass", "warn", "skipped")


def _hash_inputs(inputs: Iterable[Path]) -> str:
    """Content-hash the ordered set of input files.

    Missing files contribute their absence (a fixed marker) so that a step
    whose input later appears is correctly seen as dirty. Paths are folded into
    the digest so a rename also invalidates the cache.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in inputs):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except (OSError, IsADirectoryError):
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


class FlowState:
    """Load/save the per-step incremental state for one run directory."""

    def __init__(self, work_dir: str | Path) -> None:
        self.work_dir = Path(work_dir)
        self.path = self.work_dir / STATE_FILENAME
        self._steps: dict[str, dict[str, str]] = {}
        self.load()

    def load(self) -> "FlowState":
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            steps = data.get("steps", data) if isinstance(data, dict) else {}
            self._steps = {
                str(name): dict(rec)
                for name, rec in steps.items()
                if isinstance(rec, dict)
            }
        return self

    def save(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        payload = {"steps": self._steps}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record(self, step: str, inputs: Iterable[Path], status: str) -> str:
        """Persist the hashed inputs + status for ``step`` and return the hash."""
        input_sha = _hash_inputs(inputs)
        self._steps[step] = {"input_sha": input_sha, "status": str(status)}
        self.save()
        return input_sha

    def recorded(self, step: str) -> dict[str, str] | None:
        return self._steps.get(step)

    def should_skip(self, step: str, inputs: Iterable[Path], force: bool) -> bool:
        """Return True iff ``step`` can be skipped on this run.

        Skip only when: not forced, a prior record exists with a skippable
        status, AND the freshly hashed inputs match the recorded hash. Any
        change to an input file (or a prior failure) makes the step dirty.
        """
        if force:
            return False
        record = self._steps.get(step)
        if not record:
            return False
        if record.get("status") not in _SKIPPABLE_STATUSES:
            return False
        return record.get("input_sha") == _hash_inputs(inputs)
