"""Portable, content-addressed evidence primitives for signed runners."""

from __future__ import annotations

import hashlib
import errno
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

from clay.flow.portfolio_signoff import ArtifactLocator


ARTIFACT_MANIFEST_SCHEMA = "clay.signoff-artifact-manifest/v1"
ARTIFACT_BUNDLE_PREFIX = PurePosixPath("signoff/artifacts/sha256")
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
SIGNED_RUN_PARENT_MANIFEST_ENV = "CLAY_SIGNED_RUN_PARENT_MANIFEST"
SIGNED_RUN_PARENT_PARTICIPANT_ENV = "CLAY_SIGNED_RUN_PARENT_PARTICIPANT"
SIGNED_RUN_PARENT_DESCENDANT_ENV = "CLAY_SIGNED_RUN_PARENT_DESCENDANT"
SIGNED_RUN_PARENT_OWNER_PID_ENV = "CLAY_SIGNED_RUN_PARENT_OWNER_PID"
_NON_AUTHORITATIVE_VISUAL_SUFFIXES = frozenset({".png"})


class SignoffArtifactError(ValueError):
    """Portable evidence is missing, ambiguous, or outside its declared scope."""


class _SignedRunOwnerLock:
    """Process-held advisory lock proving a transaction owner is still alive."""

    def __init__(self, path: Path, handle) -> None:
        self.path = path
        self.handle = handle

    def release(self) -> None:
        if self.handle is None:
            return
        handle, self.handle = self.handle, None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __del__(self) -> None:
        try:
            self.release()
        except (OSError, ValueError):
            pass


def _signed_run_owner_lock_path(root: Path, manifest: Path) -> Path:
    identity = hashlib.sha256(
        (str(root) + "\0" + str(manifest)).encode("utf-8")
    ).hexdigest()
    directory = Path(tempfile.gettempdir()) / "clay-signed-run-owner-locks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{identity}.lock"


def _try_acquire_signed_run_owner_lock(
    root: Path,
    manifest: Path,
) -> _SignedRunOwnerLock | None:
    path = _signed_run_owner_lock_path(root, manifest)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return None
        raise SignoffArtifactError(
            f"signed transaction owner lock failed: {path}"
        ) from exc
    return _SignedRunOwnerLock(path, handle)


def _acquire_signed_run_owner_lock(
    root: Path,
    manifest: Path,
) -> _SignedRunOwnerLock:
    lock = _try_acquire_signed_run_owner_lock(root, manifest)
    if lock is None:
        raise SignoffArtifactError(
            "signed transaction manifest already has a live owner"
        )
    return lock


def _require_live_signed_run_owner(root: Path, manifest: Path) -> None:
    probe = _try_acquire_signed_run_owner_lock(root, manifest)
    if probe is None:
        return
    probe.release()
    raise SignoffArtifactError(
        "parent signed transaction has no live process owner"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root(path: str | Path, *, create: bool = False) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise SignoffArtifactError("repo root must be an explicit absolute path")
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    try:
        root = raw.resolve(strict=True)
    except OSError as exc:
        raise SignoffArtifactError(f"repo root is missing: {raw}") from exc
    if not root.is_dir():
        raise SignoffArtifactError(f"repo root is not a directory: {root}")
    return root


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SignoffArtifactError("artifact path must be a non-empty relative path")
    if "\\" in value:
        raise SignoffArtifactError("artifact path must use repository-relative POSIX form")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise SignoffArtifactError(f"absolute artifact path is forbidden: {value}")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise SignoffArtifactError(f"artifact parent traversal is forbidden: {value}")
    return posix


def _resolve_locator_path(root: Path, relative: PurePosixPath) -> Path:
    lexical = root.joinpath(*relative.parts)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise SignoffArtifactError(f"required artifact is missing: {relative}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SignoffArtifactError(f"artifact symlink escape is forbidden: {relative}") from exc
    return resolved


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(
        getattr(metadata, "st_file_attributes", 0) & reparse_flag
    )


def _prepare_confined_directory(root: Path, relative: PurePosixPath) -> Path:
    """Create a directory one component at a time without traversing links."""

    current = root
    for part in relative.parts:
        candidate = current / part
        if not candidate.exists() and not candidate.is_symlink():
            try:
                candidate.mkdir()
            except FileExistsError:
                pass
        if _is_link_or_reparse(candidate):
            raise SignoffArtifactError(
                f"artifact bundle parent symlink/reparse is forbidden: {relative}"
            )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise SignoffArtifactError(
                f"artifact bundle parent escapes repo root: {relative}"
            ) from exc
        if not resolved.is_dir():
            raise SignoffArtifactError(
                f"artifact bundle parent is not a directory: {relative}"
            )
        current = resolved
    return current


def _is_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 4096:
        return False
    with path.open("rb") as stream:
        return stream.read(len(_LFS_HEADER)) == _LFS_HEADER


def package_artifact(
    repo_root: str | Path,
    source: str | Path,
    *,
    logical_id: str,
    artifact_type: str = "file",
    required: bool = True,
) -> ArtifactLocator:
    """Copy one evidence file into the repository-owned SHA-256 bundle."""

    root = _repo_root(repo_root, create=True)
    if not logical_id or not logical_id.strip():
        raise SignoffArtifactError("artifact logical id must be non-empty")
    if artifact_type not in {"file", "manifest"}:
        raise SignoffArtifactError(f"unknown artifact type: {artifact_type!r}")
    path = Path(source)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SignoffArtifactError(f"artifact source is missing: {path}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SignoffArtifactError(f"artifact source is empty or not a file: {resolved}")
    if _is_lfs_pointer(resolved):
        raise SignoffArtifactError(f"artifact source is an unresolved Git LFS pointer: {resolved}")
    digest = _sha256(resolved)
    destination_relative = ARTIFACT_BUNDLE_PREFIX / digest / resolved.name
    destination_parent = _prepare_confined_directory(
        root, destination_relative.parent
    )
    destination = destination_parent / resolved.name
    if destination.exists():
        if _is_link_or_reparse(destination) or not destination.is_file():
            raise SignoffArtifactError(f"artifact bundle destination is not a regular file: {destination}")
        if _sha256(destination) != digest:
            raise SignoffArtifactError(f"artifact bundle collision at {destination}")
    else:
        destination_parent = _prepare_confined_directory(
            root, destination_relative.parent
        )
        destination = destination_parent / resolved.name
        handle, temporary_name = tempfile.mkstemp(
            prefix=resolved.name + ".", suffix=".tmp", dir=destination_parent
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(resolved, temporary)
            if _sha256(temporary) != digest:
                raise SignoffArtifactError("artifact changed while it was packaged")
            destination_parent = _prepare_confined_directory(
                root, destination_relative.parent
            )
            destination = destination_parent / resolved.name
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return ArtifactLocator(
        logical_id=logical_id,
        path=destination.relative_to(root).as_posix(),
        sha256=digest,
        artifact_type=artifact_type,  # type: ignore[arg-type]
        required=required,
    )


def verify_artifact_locator(
    repo_root: str | Path, locator: ArtifactLocator
) -> Path:
    """Resolve and hash-check a locator without trusting checkout location."""

    root = _repo_root(repo_root)
    relative = _safe_relative_path(locator.path)
    if locator.artifact_type not in {"file", "manifest"}:
        raise SignoffArtifactError(
            f"unknown artifact type for {locator.logical_id}: {locator.artifact_type!r}"
        )
    if not isinstance(locator.required, bool):
        raise SignoffArtifactError(f"artifact required flag is not boolean: {locator.logical_id}")
    digest = str(locator.sha256).casefold()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise SignoffArtifactError(f"artifact digest is not SHA-256: {locator.logical_id}")
    bundle_parts = ARTIFACT_BUNDLE_PREFIX.parts
    if locator.required:
        if relative.parts[: len(bundle_parts)] != bundle_parts:
            raise SignoffArtifactError(
                f"required artifact must live under signoff/artifacts: {locator.path}"
            )
        if len(relative.parts) <= len(bundle_parts) or relative.parts[len(bundle_parts)] != digest:
            raise SignoffArtifactError(
                f"content-address directory does not match digest: {locator.path}"
            )
    lexical = root.joinpath(*relative.parts)
    if not lexical.exists() and not lexical.is_symlink():
        if not locator.required:
            return lexical
        raise SignoffArtifactError(f"required artifact is missing: {locator.path}")
    resolved = _resolve_locator_path(root, relative)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SignoffArtifactError(f"artifact is empty or not a file: {locator.path}")
    if _is_lfs_pointer(resolved):
        raise SignoffArtifactError(
            f"artifact is an unresolved Git LFS pointer: {locator.path}"
        )
    if _sha256(resolved) != digest:
        raise SignoffArtifactError(f"artifact digest mismatch: {locator.path}")
    return resolved


def _locator_from_mapping(value: object) -> ArtifactLocator:
    if not isinstance(value, Mapping):
        raise SignoffArtifactError("manifest artifact locator is not an object")
    expected = {"logical_id", "path", "sha256", "artifact_type", "required"}
    if set(value) != expected:
        raise SignoffArtifactError(
            "manifest artifact locator fields differ: "
            f"missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)}"
        )
    try:
        return ArtifactLocator(**{key: value[key] for key in expected})  # type: ignore[arg-type]
    except TypeError as exc:
        raise SignoffArtifactError(f"manifest artifact locator is invalid: {exc}") from exc


def package_artifact_manifest(
    repo_root: str | Path,
    *,
    logical_id: str,
    artifacts: Iterable[ArtifactLocator],
) -> ArtifactLocator:
    """Create and package a typed manifest whose children are verified transitively."""

    root = _repo_root(repo_root, create=True)
    children = tuple(artifacts)
    payload = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "artifacts": [child.canonical_data() for child in children],
    }
    staging = root / "signoff" / ".artifact-staging"
    staging.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in logical_id)
    if not safe_name:
        raise SignoffArtifactError("manifest logical id must be non-empty")
    handle, temporary_name = tempfile.mkstemp(
        prefix=safe_name + ".", suffix=".json", dir=staging
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return package_artifact(
            root,
            temporary_name,
            logical_id=logical_id,
            artifact_type="manifest",
            required=True,
        )
    finally:
        Path(temporary_name).unlink(missing_ok=True)
        try:
            staging.rmdir()
        except OSError:
            pass


def verify_artifact_graph(
    repo_root: str | Path, roots: Sequence[ArtifactLocator]
) -> tuple[ArtifactLocator, ...]:
    """Verify roots and every typed manifest child in deterministic DFS order."""

    verified: list[ArtifactLocator] = []
    active: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    locator_semantics: dict[tuple[str, str], tuple[str, bool]] = {}

    def visit(locator: ArtifactLocator) -> None:
        identity = (locator.path, locator.sha256)
        semantics = (locator.artifact_type, locator.required)
        previous_semantics = locator_semantics.get(identity)
        if previous_semantics is not None and previous_semantics != semantics:
            raise SignoffArtifactError(
                "artifact locator type/required conflict for "
                f"{locator.path}: {previous_semantics!r} != {semantics!r}"
            )
        locator_semantics[identity] = semantics
        if identity in active:
            raise SignoffArtifactError(f"artifact manifest cycle at {locator.path}")
        if identity in seen:
            return
        path = verify_artifact_locator(repo_root, locator)
        seen.add(identity)
        verified.append(locator)
        if locator.artifact_type != "manifest":
            return
        active.add(identity)
        try:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise SignoffArtifactError(
                    f"artifact manifest is invalid JSON: {locator.path}"
                ) from exc
            if not isinstance(payload, dict) or set(payload) != {"schema", "artifacts"}:
                raise SignoffArtifactError(
                    f"artifact manifest fields differ: {locator.path}"
                )
            if payload.get("schema") != ARTIFACT_MANIFEST_SCHEMA:
                raise SignoffArtifactError(
                    f"artifact manifest schema is unsupported: {locator.path}"
                )
            raw_children = payload.get("artifacts")
            if not isinstance(raw_children, list):
                raise SignoffArtifactError(
                    f"artifact manifest has no artifact list: {locator.path}"
                )
            for child in raw_children:
                visit(_locator_from_mapping(child))
        finally:
            active.remove(identity)

    for root_locator in roots:
        visit(root_locator)
    return tuple(verified)


_AUTHORITATIVE_EXACT_PATHS = frozenset(
    {
        "signoff/topology_registry.json",
        "ldo_dev/signoff/resource_cuts.json",
        "ldo_dev/signoff/retry_exhaustion.json",
    }
)
_AUTHORITATIVE_ANALYSIS_NAMES = frozenset(
    {
        "reference_intent_hints.json",
        "reference_physical_targets.json",
        "reference_electrical_spec.json",
    }
)
_LOADABLE_AUTHORITATIVE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".pyx",
        ".pxd",
        ".pyd",
        ".so",
        ".dll",
        ".dylib",
        ".clay",
        ".cir",
        ".spice",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
    }
)
_GENERATED_CACHE_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)


def _git(root: Path, *arguments: str, allow_failure: bool = False) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode and not allow_failure:
        raise SignoffArtifactError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout if completed.returncode == 0 else ""


def _normalize_repo_relative(value: str | Path) -> str:
    raw = str(value).replace("\\", "/").rstrip("/")
    relative = _safe_relative_path(raw)
    return relative.as_posix()


def _is_non_authoritative_visual(relative: str) -> bool:
    return (
        PurePosixPath(relative).suffix.casefold()
        in _NON_AUTHORITATIVE_VISUAL_SUFFIXES
    )


def _matches_extra_path(relative: str, extras: Sequence[str]) -> bool:
    return any(relative == item or relative.startswith(item + "/") for item in extras)


def _is_authoritative_path(relative: str, extras: Sequence[str] = ()) -> bool:
    path = PurePosixPath(relative)
    name = path.name.casefold()
    parts = tuple(part.casefold() for part in path.parts)
    if relative in _AUTHORITATIVE_EXACT_PATHS:
        return True
    if "ref" in parts:
        return True
    if "analysis" in parts and name in _AUTHORITATIVE_ANALYSIS_NAMES:
        return True
    if name in {"resource_cuts.json", "retry_exhaustion.json"}:
        return True
    if "selected_candidates" in name or "selection_receipt" in name:
        return True
    return _matches_extra_path(relative, extras)


def _is_ignored_loadable_shadow(relative: str, extras: Sequence[str]) -> bool:
    path = PurePosixPath(relative)
    folded_parts = {part.casefold() for part in path.parts}
    if folded_parts.intersection(_GENERATED_CACHE_PARTS):
        return False
    if path.suffix.casefold() not in _LOADABLE_AUTHORITATIVE_SUFFIXES:
        return False
    if _matches_extra_path(relative, extras):
        return True
    return any(
        path.parent == PurePosixPath(extra).parent
        and path.stem.casefold() == PurePosixPath(extra).stem.casefold()
        for extra in extras
        if PurePosixPath(extra).suffix
    )


def _tracked_paths(root: Path) -> set[str]:
    output = _git(root, "ls-files", "-z")
    return {
        item.replace("\\", "/")
        for item in output.split("\0")
        if item
    }


def require_tracked_repo_input(
    repo_root: str | Path, value: str | Path
) -> str:
    """Return one real, tracked CLI input as a confined repo-relative path."""

    root = _repo_root(repo_root)
    raw = Path(value)
    lexical = raw if raw.is_absolute() else root / raw
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise SignoffArtifactError(f"authoritative CLI input is missing: {value}") from exc
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise SignoffArtifactError(
            f"authoritative CLI input escapes repo root: {value}"
        ) from exc
    if not resolved.is_file():
        raise SignoffArtifactError(
            f"authoritative CLI input is not a regular file: {relative}"
        )
    if relative not in _tracked_paths(root):
        raise SignoffArtifactError(
            f"authoritative CLI input is not tracked: {relative}"
        )
    return relative


def _head(root: Path) -> str:
    head = _git(root, "rev-parse", "HEAD").strip().casefold()
    if len(head) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in head):
        raise SignoffArtifactError("signed run base commit is not a full object id")
    return head


def _declared_digest_paths(payload: object) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if isinstance(child, str) and (
                    key == "path" or key == "source" or key.endswith("_path")
                ):
                    digest = value.get(f"{key}_sha256")
                    if digest is None and key == "source":
                        digest = value.get("source_sha256")
                    if digest is None and key == "path":
                        digest = value.get("sha256")
                    if isinstance(digest, str):
                        found.append((child, digest.casefold()))
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                visit(child)

    visit(payload)
    return tuple(found)


@dataclass(frozen=True)
class AuthoritativeInputSnapshot:
    """Digest-stable closure of every clean semantic signoff input."""

    base_commit: str
    paths: tuple[str, ...]
    file_sha256: tuple[tuple[str, str], ...]
    sha256: str

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.authoritative-input-closure/v1",
            "base_commit": self.base_commit,
            "paths": list(self.paths),
            "file_sha256": dict(self.file_sha256),
            "sha256": self.sha256,
        }


def authoritative_input_closure(
    repo_root: str | Path,
    *,
    extra_paths: Sequence[str | Path] = (),
    excluded_outputs: Sequence[str | Path] = (),
) -> AuthoritativeInputSnapshot:
    """Build the transitive, tracked authoritative-input closure."""

    root = _repo_root(repo_root)
    extras = tuple(_normalize_repo_relative(item) for item in extra_paths)
    if any(_is_non_authoritative_visual(item) for item in extras):
        raise SignoffArtifactError("PNG cannot be an authoritative signoff input")
    excluded = frozenset(_normalize_repo_relative(item) for item in excluded_outputs)
    tracked = _tracked_paths(root)
    selected = {
        relative
        for relative in tracked
        if _is_authoritative_path(relative, extras)
        and relative not in excluded
        and not _is_non_authoritative_visual(relative)
    }
    changed = True
    while changed:
        changed = False
        for relative in tuple(sorted(selected)):
            path = root / PurePosixPath(relative)
            if not path.is_file() or path.suffix.casefold() != ".json":
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise SignoffArtifactError(
                    f"authoritative input is invalid JSON: {relative}"
                ) from exc
            for raw_path, declared_digest in _declared_digest_paths(payload):
                try:
                    candidate = _normalize_repo_relative(raw_path)
                except SignoffArtifactError:
                    # Machine-specific external provenance is represented by the
                    # manifest bytes but is not a tracked semantic child.
                    continue
                if _is_non_authoritative_visual(candidate):
                    raise SignoffArtifactError(
                        "PNG cannot be a digest-bound authoritative signoff input"
                    )
                if candidate not in tracked or candidate in excluded:
                    continue
                child = root / PurePosixPath(candidate)
                if not child.is_file():
                    selected.add(candidate)
                    continue
                actual = _sha256(child)
                if declared_digest != actual:
                    raise SignoffArtifactError(
                        f"authoritative digest-bound input mismatch: {candidate}"
                    )
                if candidate not in selected:
                    selected.add(candidate)
                    changed = True
    records: list[tuple[str, str]] = []
    for relative in sorted(selected):
        path = root / PurePosixPath(relative)
        records.append((relative, _sha256(path) if path.is_file() else "missing"))
    base_commit = _head(root)
    canonical = {
        "base_commit": base_commit,
        "file_sha256": records,
    }
    closure_sha = hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return AuthoritativeInputSnapshot(
        base_commit=base_commit,
        paths=tuple(relative for relative, _ in records),
        file_sha256=tuple(records),
        sha256=closure_sha,
    )


def _porcelain_entries(root: Path) -> tuple[tuple[str, str], ...]:
    raw = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    chunks = raw.split("\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        index += 1
        if not chunk:
            continue
        if len(chunk) < 4:
            raise SignoffArtifactError(f"cannot parse Git status entry: {chunk!r}")
        status, relative = chunk[:2], chunk[3:].replace("\\", "/")
        entries.append((status, relative))
        if "R" in status or "C" in status:
            if index < len(chunks) and chunks[index]:
                entries.append((status, chunks[index].replace("\\", "/")))
                index += 1
    return tuple(entries)


def _ignored_untracked_paths(root: Path) -> set[str]:
    raw = _git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    return {
        item.replace("\\", "/") for item in raw.split("\0") if item
    }


def _transitive_authoritative_locator_paths(
    root: Path,
    start_paths: Sequence[str],
    *,
    excluded: Sequence[str] = (),
) -> set[str]:
    """Collect exact repo-local locator children, including ignored shadows."""

    excluded_set = set(excluded)
    queue = [
        relative
        for relative in start_paths
        if PurePosixPath(relative).suffix.casefold() == ".json"
    ]
    visited: set[str] = set()
    candidates: set[str] = set()
    while queue:
        relative = queue.pop()
        if relative in visited:
            continue
        visited.add(relative)
        path = _confined_existing_path(root, relative)
        if path is None or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SignoffArtifactError(
                f"authoritative input is invalid JSON: {relative}"
            ) from exc
        for raw_path, declared_digest in _declared_digest_paths(payload):
            try:
                candidate = _normalize_repo_relative(raw_path)
            except SignoffArtifactError:
                continue
            if candidate in excluded_set:
                continue
            candidates.add(candidate)
            child = _confined_existing_path(root, candidate)
            if child is None:
                continue
            if not child.is_file():
                raise SignoffArtifactError(
                    f"authoritative locator is not a regular file: {candidate}"
                )
            if _sha256(child) != declared_digest:
                raise SignoffArtifactError(
                    f"authoritative digest-bound input mismatch: {candidate}"
                )
            if PurePosixPath(candidate).suffix.casefold() == ".json":
                queue.append(candidate)
    return candidates


def require_clean_authoritative_inputs(
    repo_root: str | Path,
    *,
    extra_paths: Sequence[str | Path] = (),
    allowed_outputs: Sequence[str | Path] = (),
) -> AuthoritativeInputSnapshot:
    """Fail before launch on tracked, staged, or untracked authoritative dirt."""

    root = _repo_root(repo_root)
    extras = tuple(_normalize_repo_relative(item) for item in extra_paths)
    allowed = frozenset(_normalize_repo_relative(item) for item in allowed_outputs)
    snapshot = authoritative_input_closure(
        root, extra_paths=extras, excluded_outputs=tuple(allowed)
    )
    protected = set(snapshot.paths)
    exact_locator_paths = _transitive_authoritative_locator_paths(
        root,
        snapshot.paths,
        excluded=tuple(allowed),
    )
    dirty = [
        (status, relative)
        for status, relative in _porcelain_entries(root)
        if relative not in allowed
        and (
            relative in protected
            or relative in exact_locator_paths
            or _is_authoritative_path(relative, extras)
            or _is_ignored_loadable_shadow(relative, extras)
        )
    ]
    exact_extra_files = {
        relative
        for relative in extras
        if root.joinpath(*PurePosixPath(relative).parts).is_file()
    }
    ignored_dirty = sorted(
        relative
        for relative in _ignored_untracked_paths(root)
        if relative not in allowed
        and (
            relative in exact_locator_paths
            or relative in exact_extra_files
            or _is_authoritative_path(relative)
            or _is_ignored_loadable_shadow(relative, extras)
        )
    )
    if ignored_dirty:
        raise SignoffArtifactError(
            "ignored untracked authoritative shadow is dirty: "
            + ", ".join(ignored_dirty)
        )
    if dirty:
        rendered = ", ".join(f"{status} {path}" for status, path in dirty)
        kind = "untracked authoritative shadow" if any(s == "??" for s, _ in dirty) else "authoritative input"
        raise SignoffArtifactError(f"{kind} is dirty: {rendered}")
    return snapshot


def authoritative_inputs_unchanged_between(
    repo_root: str | Path,
    base_commit: str,
    current_commit: str,
    *,
    extra_paths: Sequence[str | Path] = (),
) -> bool:
    """Verify a strict-ancestor evidence commit has identical authority bytes."""

    root = _repo_root(repo_root)
    base = str(base_commit).casefold()
    current = str(current_commit).casefold()
    if (
        len(base) not in {40, 64}
        or len(current) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in base + current)
        or _head(root) != current
    ):
        return False
    if base == current:
        return True
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", base, current),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        return False
    extras = tuple(_normalize_repo_relative(path) for path in extra_paths)
    snapshot = authoritative_input_closure(root, extra_paths=extras)
    authoritative = set(snapshot.paths)
    changed = _git(root, "diff", "--name-only", "-z", f"{base}..{current}")
    changed_paths = {
        item.replace("\\", "/") for item in changed.split("\0") if item
    }
    if any(
        relative in authoritative or _is_authoritative_path(relative, extras)
        for relative in changed_paths
    ):
        return False
    for relative, current_digest in snapshot.file_sha256:
        if current_digest == "missing":
            return False
        historical = subprocess.run(
            ("git", "show", f"{base}:{relative}"),
            cwd=root,
            check=False,
            capture_output=True,
        )
        contemporary = subprocess.run(
            ("git", "show", f"{current}:{relative}"),
            cwd=root,
            check=False,
            capture_output=True,
        )
        if (
            historical.returncode != 0
            or contemporary.returncode != 0
            or hashlib.sha256(historical.stdout).digest()
            != hashlib.sha256(contemporary.stdout).digest()
        ):
            return False
    return True


def _confined_manifest_path(root: Path, value: str | Path) -> tuple[Path, str]:
    """Resolve a transaction manifest without permitting lexical or symlink escape."""

    raw = Path(value)
    if raw.is_absolute():
        candidate = raw.resolve(strict=False)
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise SignoffArtifactError("transaction manifest escapes repo root") from exc
        normalized = _normalize_repo_relative(relative)
    else:
        try:
            normalized = _normalize_repo_relative(str(value))
        except SignoffArtifactError as exc:
            raise SignoffArtifactError(
                "transaction manifest must be relative and confined to repo root"
            ) from exc
        candidate = root.joinpath(*PurePosixPath(normalized).parts).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SignoffArtifactError("transaction manifest escapes repo root") from exc
    return candidate, normalized


def _confined_existing_path(root: Path, relative: str) -> Path | None:
    normalized = _normalize_repo_relative(relative)
    lexical = root.joinpath(*PurePosixPath(normalized).parts)
    if not lexical.exists() and not lexical.is_symlink():
        return None
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise SignoffArtifactError(f"transaction path cannot be resolved: {normalized}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SignoffArtifactError(
            f"transaction path symlink escape is forbidden: {normalized}"
        ) from exc
    return resolved


def _path_digest(root: Path, relative: str) -> str | None:
    path = _confined_existing_path(root, relative)
    if path is None:
        return None
    if not path.is_file():
        raise SignoffArtifactError(f"transaction path is not a regular file: {relative}")
    return _sha256(path)


def _normalize_output_prefix(value: str | Path) -> str:
    prefix = _normalize_repo_relative(value)
    if PurePosixPath(prefix) != ARTIFACT_BUNDLE_PREFIX:
        raise SignoffArtifactError(
            "signed transaction output prefixes are restricted to "
            f"{ARTIFACT_BUNDLE_PREFIX.as_posix()}"
        )
    return prefix


def _content_addressed_tree(root: Path, prefix: str) -> dict[str, str]:
    """Return a confined, regular-file-only snapshot of one CAS output tree."""

    normalized = _normalize_output_prefix(prefix)
    lexical = root.joinpath(*PurePosixPath(normalized).parts)
    if not lexical.exists() and not lexical.is_symlink():
        return {}
    resolved = _confined_existing_path(root, normalized)
    if resolved is None or not resolved.is_dir():
        raise SignoffArtifactError(
            f"content-addressed output prefix is not a directory: {normalized}"
        )
    records: dict[str, str] = {}
    for path in sorted(lexical.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise SignoffArtifactError(
                f"content-addressed output symlink is forbidden: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise SignoffArtifactError(
                f"content-addressed output is not a regular file: {relative}"
            )
        digest = _sha256(path)
        parts = PurePosixPath(relative).parts
        prefix_parts = ARTIFACT_BUNDLE_PREFIX.parts
        if len(parts) <= len(prefix_parts) + 1:
            raise SignoffArtifactError(
                f"content-addressed output has invalid locator shape: {relative}"
            )
        directory_digest = parts[len(prefix_parts)]
        if (
            len(directory_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in directory_digest)
            or digest != directory_digest
        ):
            raise SignoffArtifactError(
                f"content-addressed output digest mismatch: {relative}"
            )
        if _is_lfs_pointer(path):
            raise SignoffArtifactError(
                f"content-addressed output is an unresolved Git LFS pointer: {relative}"
            )
        records[relative] = digest
    return records


def _git_content_addressed_paths(
    root: Path,
    commit: str,
    prefix: str,
) -> set[str]:
    """Return the CAS file paths that already belonged to the base Git tree."""

    normalized = _normalize_output_prefix(prefix)
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "-z",
        "--name-only",
        "--full-tree",
        commit,
        "--",
        normalized,
    )
    paths = {value for value in raw.split("\0") if value}
    if any(_normalize_repo_relative(path) != path for path in paths):
        raise SignoffArtifactError(
            "base content-addressed output paths are non-canonical"
        )
    return paths


def _git_blob_oid(root: Path, commit: str, relative: str) -> str | None:
    normalized = _normalize_repo_relative(relative)
    completed = subprocess.run(
        ("git", "rev-parse", "--verify", f"{commit}:{normalized}"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return None
    oid = completed.stdout.strip().casefold()
    kind = _git(root, "cat-file", "-t", oid).strip()
    if kind != "blob":
        raise SignoffArtifactError(
            f"ordinary output base path is not a blob: {normalized}"
        )
    return oid


def _git_path_digest(root: Path, commit: str, relative: str) -> str | None:
    """Return the SHA-256 of one base-tree blob, or None when it was absent."""

    oid = _git_blob_oid(root, commit, relative)
    if oid is None:
        return None
    completed = subprocess.run(
        ("git", "cat-file", "blob", oid),
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise SignoffArtifactError(
            f"ordinary output base blob cannot be read: {relative}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _ordinary_output_matches_base(
    root: Path,
    commit: str,
    relative: str,
    base_digest: str | None,
) -> bool:
    """Compare a working output with its base blob using Git path filters."""

    path = _confined_existing_path(root, relative)
    if base_digest is None:
        return path is None
    if path is None:
        return False
    if not path.is_file():
        raise SignoffArtifactError(
            f"ordinary output is not a regular file: {relative}"
        )
    base_oid = _git_blob_oid(root, commit, relative)
    if base_oid is None:
        return False
    normalized = _normalize_repo_relative(relative)
    completed = subprocess.run(
        ("git", "hash-object", f"--path={normalized}", "--", normalized),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise SignoffArtifactError(
            f"ordinary output cannot be compared with the base commit: {relative}"
        )
    return completed.stdout.strip().casefold() == base_oid


def _code_digest(root: Path, paths: Sequence[str]) -> str:
    records = []
    for relative in paths:
        digest = _path_digest(root, relative)
        if digest is None:
            raise SignoffArtifactError(f"participant code is missing: {relative}")
        records.append((relative, digest))
    return hashlib.sha256(
        json.dumps(records, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not any(ch not in "0123456789abcdef" for ch in value)
    )


def _descendant_receipt_sha256(receipt: Mapping[str, object]) -> str:
    core = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class SignedRunParticipant:
    name: str
    code_paths: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    predecessors: tuple[str, ...] = ()
    output_prefixes: tuple[str, ...] = ()
    descendants: tuple["SignedRunParticipant", ...] = ()


def _same_participant_declaration(
    left: SignedRunParticipant,
    right: SignedRunParticipant,
) -> bool:
    """Compare manifest-backed path maps as order-insensitive multisets."""

    return (
        left.name == right.name
        and left.code_paths == right.code_paths
        and tuple(sorted(left.inputs)) == tuple(sorted(right.inputs))
        and tuple(sorted(left.outputs)) == tuple(sorted(right.outputs))
        and left.predecessors == right.predecessors
        and left.output_prefixes == right.output_prefixes
        and len(left.descendants) == len(right.descendants)
        and all(
            _same_participant_declaration(left_child, right_child)
            for left_child, right_child in zip(
                left.descendants, right.descendants
            )
        )
    )


@dataclass(frozen=True)
class SignedRunParentContract:
    """Canonical parent graph accepted by one signed child entry point."""

    manifest_path: str
    ordered_participants: tuple[str, ...]
    participant_name: str
    predecessors: tuple[str, ...] = ()


class SignedRunTransaction:
    """Parent-owned ordered permission envelope for signed evidence writers."""

    SCHEMA = "clay.signed-run-transaction/v3"

    def __init__(
        self,
        root: Path,
        manifest: Path,
        participants: tuple[SignedRunParticipant, ...],
        snapshot: AuthoritativeInputSnapshot,
        input_digests: dict[str, dict[str, str]],
        dynamic_inputs: dict[str, dict[str, str]],
        output_before: dict[str, dict[str, str | None]],
        prefix_before: dict[str, dict[str, str]],
        code_digests: dict[str, str],
        owner_lock: _SignedRunOwnerLock | None = None,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.participants = participants
        self.snapshot = snapshot
        self.input_digests = input_digests
        self.dynamic_inputs = dynamic_inputs
        self.output_before = output_before
        self.output_after: dict[str, dict[str, str]] = {}
        self.prefix_before = {
            prefix: dict(records) for prefix, records in prefix_before.items()
        }
        self.prefix_state = {
            prefix: dict(records) for prefix, records in prefix_before.items()
        }
        self.content_addressed_output_after: dict[str, dict[str, str]] = {}
        self.descendant_receipts: dict[str, list[dict[str, object]]] = {
            item.name: [] for item in participants
        }
        self.code_digests = code_digests
        self.completed: list[str] = []
        self._owner_lock = owner_lock

    @classmethod
    def begin(
        cls,
        repo_root: str | Path,
        manifest_path: str | Path,
        participants: Sequence[SignedRunParticipant],
        *,
        authoritative_snapshot: AuthoritativeInputSnapshot | None = None,
    ) -> "SignedRunTransaction":
        root = _repo_root(repo_root)
        manifest, manifest_relative = _confined_manifest_path(root, manifest_path)
        declared = tuple(participants)
        if not declared or len(declared) > 4:
            raise SignoffArtifactError("signed transaction has an unknown fourth participant set")
        names = tuple(item.name for item in declared)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise SignoffArtifactError("signed transaction participant names are invalid")
        all_outputs: set[str] = set()
        all_prefixes: set[str] = set()
        extras: set[str] = set()
        normalized: list[SignedRunParticipant] = []
        for index, item in enumerate(declared):
            code = tuple(_normalize_repo_relative(path) for path in item.code_paths)
            inputs = tuple(_normalize_repo_relative(path) for path in item.inputs)
            outputs = tuple(_normalize_repo_relative(path) for path in item.outputs)
            output_prefixes = tuple(
                _normalize_output_prefix(path) for path in item.output_prefixes
            )
            descendants: list[SignedRunParticipant] = []
            descendant_names: set[str] = set()
            descendant_outputs: set[str] = set()
            for child in item.descendants:
                child_code = tuple(
                    _normalize_repo_relative(path) for path in child.code_paths
                )
                child_inputs = tuple(
                    _normalize_repo_relative(path) for path in child.inputs
                )
                child_outputs = tuple(
                    _normalize_repo_relative(path) for path in child.outputs
                )
                child_prefixes = tuple(
                    _normalize_output_prefix(path)
                    for path in child.output_prefixes
                )
                if (
                    not child.name
                    or child.name in descendant_names
                    or not child_code
                    or not child_outputs
                    or child.predecessors
                    or child.descendants
                    or not set(child_code).issubset(code)
                    or not set(child_inputs).issubset(inputs)
                    or not set(child_outputs).issubset(outputs)
                    or not set(child_prefixes).issubset(output_prefixes)
                    or descendant_outputs.intersection(child_outputs)
                ):
                    raise SignoffArtifactError(
                        f"participant {item.name} has invalid descendant declarations"
                    )
                descendant_names.add(child.name)
                descendant_outputs.update(child_outputs)
                descendants.append(
                    SignedRunParticipant(
                        child.name,
                        child_code,
                        child_inputs,
                        child_outputs,
                        (),
                        child_prefixes,
                    )
                )
            if not code or not outputs:
                raise SignoffArtifactError(f"participant {item.name} has incomplete code/output declarations")
            if any(path in all_outputs for path in outputs):
                raise SignoffArtifactError("cross-participant output declaration is forbidden")
            unknown_predecessors = set(item.predecessors) - set(names[:index])
            if unknown_predecessors:
                raise SignoffArtifactError(
                    f"participant {item.name} has undeclared/future predecessors: {sorted(unknown_predecessors)}"
                )
            all_outputs.update(outputs)
            all_prefixes.update(output_prefixes)
            extras.update(code)
            extras.update(inputs)
            normalized.append(
                SignedRunParticipant(
                    item.name,
                    code,
                    inputs,
                    outputs,
                    tuple(item.predecessors),
                    output_prefixes,
                    tuple(descendants),
                )
            )
        initial_dirty = _porcelain_entries(root)
        if initial_dirty:
            raise SignoffArtifactError(
                "signed transaction must begin from a clean worktree: "
                + ", ".join(f"{status} {path}" for status, path in initial_dirty)
            )
        if authoritative_snapshot is None:
            snapshot = require_clean_authoritative_inputs(
                root,
                extra_paths=tuple(sorted(extras)),
                allowed_outputs=tuple(sorted(all_outputs | {manifest_relative})),
            )
        else:
            snapshot = authoritative_snapshot
            if snapshot.base_commit != _head(root):
                raise SignoffArtifactError(
                    "direct signed-run preflight belongs to another base commit"
                )
            required_snapshot_paths = {
                path for item in normalized for path in item.code_paths
            }
            required_snapshot_paths.update(
                path
                for item in normalized
                for path in item.inputs
                if path not in all_outputs
            )
            missing_from_closure = required_snapshot_paths - set(snapshot.paths)
            if missing_from_closure:
                raise SignoffArtifactError(
                    "participant input is outside the authoritative closure: "
                    + ", ".join(sorted(missing_from_closure))
                )
        input_digests: dict[str, dict[str, str]] = {}
        producer_by_output = {
            output: item.name for item in normalized for output in item.outputs
        }
        dynamic_inputs: dict[str, dict[str, str]] = {}
        output_before: dict[str, dict[str, str | None]] = {}
        code_digests: dict[str, str] = {}
        prefix_before = {
            prefix: _content_addressed_tree(root, prefix)
            for prefix in sorted(all_prefixes)
        }
        for prefix, records in prefix_before.items():
            base_paths = _git_content_addressed_paths(
                root,
                snapshot.base_commit,
                prefix,
            )
            if set(records) != base_paths:
                raise SignoffArtifactError(
                    "content-addressed begin snapshot differs from the base commit: "
                    + prefix
                )
        for item in normalized:
            per_input: dict[str, str] = {}
            per_dynamic: dict[str, str] = {}
            for relative in item.inputs:
                producer = producer_by_output.get(relative)
                if producer is not None:
                    if producer not in item.predecessors:
                        raise SignoffArtifactError(
                            f"participant {item.name} consumes undeclared predecessor output: {relative}"
                        )
                    per_dynamic[relative] = producer
                digest = _path_digest(root, relative)
                if digest is None:
                    if producer is None:
                        raise SignoffArtifactError(f"participant input is missing: {relative}")
                    per_input[relative] = ""
                else:
                    per_input[relative] = digest
            input_digests[item.name] = per_input
            dynamic_inputs[item.name] = per_dynamic
            per_output_before: dict[str, str | None] = {}
            for relative in item.outputs:
                base_digest = _git_path_digest(
                    root, snapshot.base_commit, relative
                )
                if not _ordinary_output_matches_base(
                    root, snapshot.base_commit, relative, base_digest
                ):
                    raise SignoffArtifactError(
                        "ordinary output begin snapshot differs from the base commit: "
                        + relative
                    )
                per_output_before[relative] = base_digest
            output_before[item.name] = per_output_before
            code_digests[item.name] = _code_digest(root, item.code_paths)
        owner_lock = _acquire_signed_run_owner_lock(root, manifest)
        transaction = cls(
            root,
            manifest,
            tuple(normalized),
            snapshot,
            input_digests,
            dynamic_inputs,
            output_before,
            prefix_before,
            code_digests,
            owner_lock,
        )
        try:
            transaction._write()
        except Exception:
            owner_lock.release()
            raise
        return transaction

    def _payload(self, *, finalized: bool = False) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "base_commit": self.snapshot.base_commit,
            "authoritative_input_closure_sha256": self.snapshot.sha256,
            "authoritative_input_paths": list(self.snapshot.paths),
            "content_addressed_outputs_before": self.prefix_before,
            "ordered_participants": [item.name for item in self.participants],
            "participants": [
                {
                    "name": item.name,
                    "code_paths": list(item.code_paths),
                    "code_sha256": self.code_digests[item.name],
                    "inputs": self.input_digests[item.name],
                    "outputs_before": self.output_before[item.name],
                    "outputs_after": self.output_after.get(item.name),
                    "output_prefixes": list(item.output_prefixes),
                    "content_addressed_outputs_after": self.content_addressed_output_after.get(
                        item.name
                    ),
                    "authorized_descendants": [
                        {
                            "name": child.name,
                            "code_paths": list(child.code_paths),
                            "inputs": list(child.inputs),
                            "outputs": list(child.outputs),
                            "output_prefixes": list(child.output_prefixes),
                        }
                        for child in item.descendants
                    ],
                    "descendant_receipts": self.descendant_receipts[item.name],
                    "predecessors": list(item.predecessors),
                }
                for item in self.participants
            ],
            "completed_participants": list(self.completed),
            "finalized": finalized,
        }

    def _write(self, *, finalized: bool = False) -> None:
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=self.manifest.name + ".", suffix=".tmp", dir=self.manifest.parent
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self._payload(finalized=finalized), stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary_name, self.manifest)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def _verify_inputs_unchanged(self) -> None:
        for participant in self.participants:
            if _code_digest(self.root, participant.code_paths) != self.code_digests[participant.name]:
                raise SignoffArtifactError(
                    f"participant input code changed: {participant.name}"
                )
            for relative, expected in self.input_digests[participant.name].items():
                if relative in self.dynamic_inputs[participant.name]:
                    continue
                if _path_digest(self.root, relative) != expected:
                    raise SignoffArtifactError(f"participant input changed: {relative}")

    def record_participant(self, name: str) -> None:
        names = tuple(item.name for item in self.participants)
        if name not in names:
            raise SignoffArtifactError(f"unknown signed transaction participant: {name}")
        expected_name = names[len(self.completed)] if len(self.completed) < len(names) else None
        if name != expected_name:
            raise SignoffArtifactError(
                f"participant is out of order: expected {expected_name!r}, got {name!r}"
            )
        participant = self.participants[len(self.completed)]
        if not set(participant.predecessors).issubset(self.completed):
            raise SignoffArtifactError(f"participant predecessors are incomplete: {name}")
        receipts = self.descendant_receipts[participant.name]
        if len(receipts) != len(participant.descendants):
            raise SignoffArtifactError(
                f"participant descendant receipts are incomplete: {name}"
            )
        self._verify_inputs_unchanged()
        for relative, producer in self.dynamic_inputs[participant.name].items():
            expected = self.output_after.get(producer, {}).get(relative)
            actual = _path_digest(self.root, relative)
            if expected is None or actual != expected:
                raise SignoffArtifactError(
                    f"participant input is not the declared predecessor output: {relative}"
                )
            self.input_digests[participant.name][relative] = actual
        for future in self.participants[len(self.completed) + 1 :]:
            for relative, before in self.output_before[future.name].items():
                if not _ordinary_output_matches_base(
                    self.root, self.snapshot.base_commit, relative, before
                ):
                    raise SignoffArtifactError(
                        f"cross-participant output changed before {future.name}: {relative}"
                    )
        after: dict[str, str] = {}
        for relative in participant.outputs:
            digest = _path_digest(self.root, relative)
            if digest is None:
                raise SignoffArtifactError(
                    f"participant has an undeclared/missing output: {relative}"
                )
            after[relative] = digest
            if _ordinary_output_matches_base(
                self.root,
                self.snapshot.base_commit,
                relative,
                self.output_before[participant.name][relative],
            ):
                raise SignoffArtifactError(
                    f"participant output is unchanged; fresh output required: {relative}"
                )
        content_addressed_after: dict[str, str] = {}
        for prefix in participant.output_prefixes:
            previous = self.prefix_state[prefix]
            current = _content_addressed_tree(self.root, prefix)
            changed_existing = {
                relative
                for relative, digest in previous.items()
                if current.get(relative) != digest
            }
            if changed_existing:
                raise SignoffArtifactError(
                    "content-addressed outputs are append-only: "
                    + ", ".join(sorted(changed_existing))
                )
            additions = {
                relative: digest
                for relative, digest in current.items()
                if relative not in previous
            }
            content_addressed_after.update(additions)
            self.prefix_state[prefix] = current
        manifest_relative = self.manifest.relative_to(self.root).as_posix()
        allowed = {
            manifest_relative,
            *(
                output
                for completed_name in (*self.completed, name)
                for output in next(
                    item.outputs for item in self.participants if item.name == completed_name
                )
            ),
        }
        allowed.update(
            relative
            for records in self.content_addressed_output_after.values()
            for relative in records
        )
        allowed.update(
            relative
            for receipts in self.descendant_receipts.values()
            for receipt in receipts
            for relative in dict(receipt["content_addressed_outputs_after"])
        )
        allowed.update(content_addressed_after)
        unexpected = [
            (status, relative)
            for status, relative in _porcelain_entries(self.root)
            if relative not in allowed
        ]
        if unexpected:
            raise SignoffArtifactError(
                "undeclared output or changed input in signed transaction: "
                + ", ".join(f"{status} {path}" for status, path in unexpected)
            )
        self.output_after[name] = after
        self.content_addressed_output_after[name] = content_addressed_after
        self.completed.append(name)
        self._write()

    def record_descendant(self, descendant: SignedRunParticipant) -> None:
        if len(self.completed) >= len(self.participants):
            raise SignoffArtifactError(
                "signed transaction has no pending participant for descendant"
            )
        parent = self.participants[len(self.completed)]
        normalized = SignedRunParticipant(
            name=descendant.name,
            code_paths=tuple(
                _normalize_repo_relative(path) for path in descendant.code_paths
            ),
            inputs=tuple(_normalize_repo_relative(path) for path in descendant.inputs),
            outputs=tuple(
                _normalize_repo_relative(path) for path in descendant.outputs
            ),
            predecessors=(),
            output_prefixes=tuple(
                _normalize_output_prefix(path) for path in descendant.output_prefixes
            ),
        )
        if (
            not normalized.name
            or not normalized.code_paths
            or not normalized.outputs
            or not set(normalized.code_paths).issubset(parent.code_paths)
            or not set(normalized.inputs).issubset(parent.inputs)
            or not set(normalized.outputs).issubset(parent.outputs)
            or not set(normalized.output_prefixes).issubset(parent.output_prefixes)
        ):
            raise SignoffArtifactError(
                "descendant declaration is outside the parent participant bindings"
            )
        existing = self.descendant_receipts[parent.name]
        ordinal = len(existing)
        if (
            ordinal >= len(parent.descendants)
            or normalized != parent.descendants[ordinal]
        ):
            raise SignoffArtifactError(
                "descendant declaration is unknown or out of authorized order"
            )
        if any(receipt.get("name") == normalized.name for receipt in existing):
            raise SignoffArtifactError(
                f"descendant signed receipt replay is forbidden: {normalized.name}"
            )
        claimed_outputs = {
            relative
            for receipt in existing
            for relative in dict(receipt["outputs_after"])
        }
        if claimed_outputs.intersection(normalized.outputs):
            raise SignoffArtifactError("descendant output is already owned")
        code_sha = _code_digest(self.root, normalized.code_paths)
        inputs: dict[str, str] = {}
        for relative in normalized.inputs:
            digest = _path_digest(self.root, relative)
            if digest is None:
                raise SignoffArtifactError(
                    f"descendant input is missing: {relative}"
                )
            parent_digest = self.input_digests[parent.name].get(relative)
            if parent_digest != digest:
                raise SignoffArtifactError(
                    f"descendant input differs from parent binding: {relative}"
                )
            inputs[relative] = digest
        outputs: dict[str, str] = {}
        for relative in normalized.outputs:
            digest = _path_digest(self.root, relative)
            if digest is None:
                raise SignoffArtifactError(
                    f"descendant output is missing: {relative}"
                )
            if _ordinary_output_matches_base(
                self.root,
                self.snapshot.base_commit,
                relative,
                self.output_before[parent.name][relative],
            ):
                raise SignoffArtifactError(
                    f"descendant output is unchanged; fresh output required: {relative}"
                )
            outputs[relative] = digest
        cas_outputs: dict[str, str] = {}
        for prefix in normalized.output_prefixes:
            previous = self.prefix_state[prefix]
            current = _content_addressed_tree(self.root, prefix)
            changed_existing = {
                relative
                for relative, digest in previous.items()
                if current.get(relative) != digest
            }
            if changed_existing:
                raise SignoffArtifactError(
                    "content-addressed outputs are append-only: "
                    + ", ".join(sorted(changed_existing))
                )
            cas_outputs.update(
                {
                    relative: digest
                    for relative, digest in current.items()
                    if relative not in previous
                }
            )
            self.prefix_state[prefix] = current
        receipt: dict[str, object] = {
            "ordinal": ordinal,
            "name": normalized.name,
            "code_paths": list(normalized.code_paths),
            "code_sha256": code_sha,
            "inputs": inputs,
            "outputs_after": outputs,
            "output_prefixes": list(normalized.output_prefixes),
            "content_addressed_outputs_after": cas_outputs,
        }
        receipt["receipt_sha256"] = _descendant_receipt_sha256(receipt)
        manifest_relative = self.manifest.relative_to(self.root).as_posix()
        allowed = {manifest_relative, *outputs, *cas_outputs}
        allowed.update(
            output
            for receipts in self.descendant_receipts.values()
            for previous_receipt in receipts
            for output in dict(previous_receipt["outputs_after"])
        )
        allowed.update(
            output
            for receipts in self.descendant_receipts.values()
            for previous_receipt in receipts
            for output in dict(
                previous_receipt["content_addressed_outputs_after"]
            )
        )
        unexpected = [
            (status, relative)
            for status, relative in _porcelain_entries(self.root)
            if relative not in allowed
        ]
        if unexpected:
            raise SignoffArtifactError(
                "undeclared descendant output or changed input: "
                + ", ".join(f"{status} {path}" for status, path in unexpected)
            )
        existing.append(receipt)
        self._write()

    def _adopt_validated_state(self, loaded: "SignedRunTransaction") -> None:
        if len(loaded.participants) != len(self.participants) or not all(
            _same_participant_declaration(loaded_item, current_item)
            for loaded_item, current_item in zip(
                loaded.participants, self.participants
            )
        ):
            raise SignoffArtifactError(
                "parent signed transaction participant declarations changed"
            )
        self.snapshot = loaded.snapshot
        self.input_digests = loaded.input_digests
        self.dynamic_inputs = loaded.dynamic_inputs
        self.output_before = loaded.output_before
        self.output_after = loaded.output_after
        self.prefix_before = loaded.prefix_before
        self.prefix_state = loaded.prefix_state
        self.content_addressed_output_after = (
            loaded.content_addressed_output_after
        )
        self.descendant_receipts = loaded.descendant_receipts
        self.code_digests = loaded.code_digests
        self.completed = loaded.completed

    def sync_recorded_participant(self, name: str) -> None:
        names = tuple(item.name for item in self.participants)
        expected = names[len(self.completed)] if len(self.completed) < len(names) else None
        if name != expected:
            raise SignoffArtifactError(
                f"parent signed transaction sync is out of order: {name}"
            )
        participant = self.participants[len(self.completed)]
        manifest_relative = self.manifest.relative_to(self.root).as_posix()
        try:
            loaded = _load_pending_parent_transaction(
                self.root,
                self.manifest,
                manifest_relative,
                participant_name=None,
                accepted_participants=(),
                expected_participant=participant,
            )
        except SignoffArtifactError as exc:
            raise SignoffArtifactError(
                f"parent signed transaction is invalid: {exc}"
            ) from exc
        if loaded.completed != [*self.completed, name]:
            raise SignoffArtifactError(
                "parent signed transaction did not record exactly one expected participant"
            )
        self._adopt_validated_state(loaded)

    def record_or_sync_participant(self, name: str) -> None:
        expected_bytes = (
            json.dumps(self._payload(), indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        try:
            current_bytes = self.manifest.read_bytes()
        except OSError as exc:
            raise SignoffArtifactError(
                "parent signed transaction manifest is missing"
            ) from exc
        if current_bytes == expected_bytes:
            self.record_participant(name)
            return
        participant = self.participants[len(self.completed)]
        manifest_relative = self.manifest.relative_to(self.root).as_posix()
        try:
            loaded = _load_pending_parent_transaction(
                self.root,
                self.manifest,
                manifest_relative,
                participant_name=None,
                accepted_participants=(),
                expected_participant=participant,
                allow_current_participant_outputs=True,
            )
        except SignoffArtifactError as exc:
            raise SignoffArtifactError(
                f"parent signed transaction is invalid: {exc}"
            ) from exc
        if loaded.completed == self.completed:
            self._adopt_validated_state(loaded)
            self.record_participant(name)
            return
        if loaded.completed == [*self.completed, name]:
            self._adopt_validated_state(loaded)
            return
        raise SignoffArtifactError(
            "parent signed transaction completion changed unexpectedly"
        )


@dataclass
class DirectSignedRun:
    """One-participant transaction handle, or a verified parent-owned join."""

    root: Path
    manifest_relative: str
    participant_name: str
    transaction: SignedRunTransaction | None
    descendant_participant: SignedRunParticipant | None = None

    def child_environment(
        self,
        base: Mapping[str, str] | None = None,
        *,
        descendant_name: str | None = None,
    ) -> dict[str, str]:
        environment = dict(os.environ if base is None else base)
        environment[SIGNED_RUN_PARENT_MANIFEST_ENV] = self.manifest_relative
        environment[SIGNED_RUN_PARENT_PARTICIPANT_ENV] = self.participant_name
        environment[SIGNED_RUN_PARENT_OWNER_PID_ENV] = str(os.getpid())
        if descendant_name is None:
            environment.pop(SIGNED_RUN_PARENT_DESCENDANT_ENV, None)
        else:
            environment[SIGNED_RUN_PARENT_DESCENDANT_ENV] = descendant_name
        return environment

    def record_success(self) -> None:
        if self.transaction is not None:
            if self.descendant_participant is None:
                self.transaction.record_or_sync_participant(self.participant_name)
            else:
                self.transaction.record_descendant(self.descendant_participant)


def _load_pending_parent_transaction(
    root: Path,
    manifest: Path,
    manifest_relative: str,
    *,
    participant_name: str | None,
    accepted_participants: Sequence[str],
    expected_participant: SignedRunParticipant | None,
    descendant_name: str | None = None,
    allow_current_participant_outputs: bool = False,
) -> SignedRunTransaction:
    try:
        raw_bytes = manifest.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SignoffArtifactError("manifest is missing or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SignoffArtifactError("manifest payload is not an object")
    canonical = (
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    if raw_bytes != canonical:
        raise SignoffArtifactError("manifest bytes are non-canonical")
    required_top_level = {
        "schema",
        "base_commit",
        "authoritative_input_closure_sha256",
        "authoritative_input_paths",
        "content_addressed_outputs_before",
        "ordered_participants",
        "participants",
        "completed_participants",
        "finalized",
    }
    if set(payload) != required_top_level:
        raise SignoffArtifactError("manifest fields are incomplete or unknown")
    if payload.get("schema") != SignedRunTransaction.SCHEMA:
        raise SignoffArtifactError("manifest schema is unsupported")
    if payload.get("finalized") is not False:
        raise SignoffArtifactError("manifest is already finalized")
    if payload.get("base_commit") != _head(root):
        raise SignoffArtifactError("base commit is not current")

    ordered = payload.get("ordered_participants")
    completed = payload.get("completed_participants")
    raw_records = payload.get("participants")
    if (
        not isinstance(ordered, list)
        or not ordered
        or len(ordered) > 4
        or not all(isinstance(name, str) and name for name in ordered)
        or len(set(ordered)) != len(ordered)
        or not isinstance(completed, list)
        or completed != ordered[: len(completed)]
        or not isinstance(raw_records, list)
        or len(raw_records) != len(ordered)
    ):
        raise SignoffArtifactError("participant order/bindings are incomplete")
    pending_index = len(completed)
    if participant_name is not None:
        if pending_index >= len(ordered):
            raise SignoffArtifactError("no participant is pending")
        if (
            participant_name not in set(accepted_participants)
            or ordered[pending_index] != participant_name
        ):
            raise SignoffArtifactError("current participant is not owned by parent")

    required_record_fields = {
        "name",
        "code_paths",
        "code_sha256",
        "inputs",
        "outputs_before",
        "outputs_after",
        "output_prefixes",
        "content_addressed_outputs_after",
        "authorized_descendants",
        "descendant_receipts",
        "predecessors",
    }
    participants: list[SignedRunParticipant] = []
    input_digests: dict[str, dict[str, str]] = {}
    output_before: dict[str, dict[str, str | None]] = {}
    code_digests: dict[str, str] = {}
    output_after: dict[str, dict[str, str]] = {}
    cas_after: dict[str, dict[str, str]] = {}
    declared_output_owner: dict[str, str] = {}
    raw_by_name: dict[str, dict[str, object]] = {}
    all_prefixes: set[str] = set()

    for index, (name, raw_record) in enumerate(zip(ordered, raw_records)):
        if not isinstance(raw_record, dict) or set(raw_record) != required_record_fields:
            raise SignoffArtifactError("participant record fields are incomplete or unknown")
        if raw_record.get("name") != name:
            raise SignoffArtifactError("participant record is out of order")
        code_paths = raw_record.get("code_paths")
        inputs = raw_record.get("inputs")
        outputs_before_raw = raw_record.get("outputs_before")
        output_prefixes = raw_record.get("output_prefixes")
        raw_authorized_descendants = raw_record.get("authorized_descendants")
        predecessors = raw_record.get("predecessors")
        if (
            not isinstance(code_paths, list)
            or not code_paths
            or not all(isinstance(path, str) for path in code_paths)
            or not isinstance(inputs, dict)
            or not all(
                isinstance(path, str) and isinstance(digest, str)
                for path, digest in inputs.items()
            )
            or not isinstance(outputs_before_raw, dict)
            or not outputs_before_raw
            or not all(
                isinstance(path, str)
                and (digest is None or _is_sha256(digest))
                for path, digest in outputs_before_raw.items()
            )
            or not isinstance(output_prefixes, list)
            or not all(isinstance(path, str) for path in output_prefixes)
            or not isinstance(raw_authorized_descendants, list)
            or not isinstance(predecessors, list)
            or not all(isinstance(item, str) for item in predecessors)
            or any(item not in ordered[:index] for item in predecessors)
        ):
            raise SignoffArtifactError("participant bindings are invalid")
        normalized_code = tuple(_normalize_repo_relative(path) for path in code_paths)
        normalized_inputs = tuple(_normalize_repo_relative(path) for path in inputs)
        normalized_outputs = tuple(
            _normalize_repo_relative(path) for path in outputs_before_raw
        )
        normalized_prefixes = tuple(
            _normalize_output_prefix(path) for path in output_prefixes
        )
        if (
            list(normalized_code) != code_paths
            or set(normalized_inputs) != set(inputs)
            or set(normalized_outputs) != set(outputs_before_raw)
            or list(normalized_prefixes) != output_prefixes
        ):
            raise SignoffArtifactError("participant paths are non-canonical")
        if _code_digest(root, normalized_code) != raw_record.get("code_sha256"):
            raise SignoffArtifactError("participant code binding changed")
        authorized_descendants: list[SignedRunParticipant] = []
        authorized_names: set[str] = set()
        authorized_outputs: set[str] = set()
        authorized_fields = {
            "name",
            "code_paths",
            "inputs",
            "outputs",
            "output_prefixes",
        }
        for raw_child in raw_authorized_descendants:
            if not isinstance(raw_child, dict) or set(raw_child) != authorized_fields:
                raise SignoffArtifactError(
                    "authorized descendant declaration fields are invalid"
                )
            child_name = raw_child.get("name")
            child_code = raw_child.get("code_paths")
            child_inputs = raw_child.get("inputs")
            child_outputs = raw_child.get("outputs")
            child_prefixes = raw_child.get("output_prefixes")
            if (
                not isinstance(child_name, str)
                or not child_name
                or child_name in authorized_names
                or not isinstance(child_code, list)
                or not child_code
                or not all(isinstance(path, str) for path in child_code)
                or not isinstance(child_inputs, list)
                or not all(isinstance(path, str) for path in child_inputs)
                or not isinstance(child_outputs, list)
                or not child_outputs
                or not all(isinstance(path, str) for path in child_outputs)
                or not isinstance(child_prefixes, list)
                or not all(isinstance(path, str) for path in child_prefixes)
            ):
                raise SignoffArtifactError(
                    "authorized descendant declaration is invalid"
                )
            normalized_child_code = tuple(
                _normalize_repo_relative(path) for path in child_code
            )
            normalized_child_inputs = tuple(
                _normalize_repo_relative(path) for path in child_inputs
            )
            normalized_child_outputs = tuple(
                _normalize_repo_relative(path) for path in child_outputs
            )
            normalized_child_prefixes = tuple(
                _normalize_output_prefix(path) for path in child_prefixes
            )
            if (
                list(normalized_child_code) != child_code
                or list(normalized_child_inputs) != child_inputs
                or list(normalized_child_outputs) != child_outputs
                or list(normalized_child_prefixes) != child_prefixes
                or not set(normalized_child_code).issubset(normalized_code)
                or not set(normalized_child_inputs).issubset(normalized_inputs)
                or not set(normalized_child_outputs).issubset(normalized_outputs)
                or not set(normalized_child_prefixes).issubset(
                    normalized_prefixes
                )
                or authorized_outputs.intersection(normalized_child_outputs)
            ):
                raise SignoffArtifactError(
                    "authorized descendant is outside parent bindings"
                )
            authorized_names.add(child_name)
            authorized_outputs.update(normalized_child_outputs)
            authorized_descendants.append(
                SignedRunParticipant(
                    name=child_name,
                    code_paths=normalized_child_code,
                    inputs=normalized_child_inputs,
                    outputs=normalized_child_outputs,
                    output_prefixes=normalized_child_prefixes,
                )
            )
        participant = SignedRunParticipant(
            name=name,
            code_paths=normalized_code,
            inputs=normalized_inputs,
            outputs=normalized_outputs,
            predecessors=tuple(predecessors),
            output_prefixes=normalized_prefixes,
            descendants=tuple(authorized_descendants),
        )
        participants.append(participant)
        input_digests[name] = dict(inputs)
        output_before[name] = dict(outputs_before_raw)
        for relative, digest in output_before[name].items():
            if _git_path_digest(root, str(payload["base_commit"]), relative) != digest:
                raise SignoffArtifactError(
                    "ordinary output begin snapshot differs from the base commit: "
                    + relative
                )
        code_digests[name] = str(raw_record["code_sha256"])
        raw_by_name[name] = raw_record
        all_prefixes.update(normalized_prefixes)
        for output in normalized_outputs:
            if output in declared_output_owner:
                raise SignoffArtifactError("cross-participant output binding")
            declared_output_owner[output] = name

    raw_prefix_before = payload.get("content_addressed_outputs_before")
    if (
        not isinstance(raw_prefix_before, dict)
        or set(raw_prefix_before) != all_prefixes
    ):
        raise SignoffArtifactError(
            "content-addressed begin snapshot prefixes are incomplete"
        )
    prefix_before: dict[str, dict[str, str]] = {}
    baseline_cas: dict[str, str] = {}
    current_prefix_trees: dict[str, dict[str, str]] = {}
    for prefix in sorted(all_prefixes):
        raw_records = raw_prefix_before.get(prefix)
        if not isinstance(raw_records, dict):
            raise SignoffArtifactError(
                "content-addressed begin snapshot is invalid"
            )
        normalized_records: dict[str, str] = {}
        for relative, digest in raw_records.items():
            if (
                not isinstance(relative, str)
                or not _is_sha256(digest)
                or _normalize_repo_relative(relative) != relative
                or not (
                    relative == prefix or relative.startswith(prefix + "/")
                )
                or _path_digest(root, relative) != digest
            ):
                raise SignoffArtifactError(
                    f"content-addressed begin snapshot changed: {relative}"
                )
            if relative in baseline_cas and baseline_cas[relative] != digest:
                raise SignoffArtifactError(
                    "content-addressed begin snapshot has conflicting bindings"
                )
            normalized_records[relative] = digest
            baseline_cas[relative] = digest
        if set(normalized_records) != _git_content_addressed_paths(
            root,
            str(payload["base_commit"]),
            prefix,
        ):
            raise SignoffArtifactError(
                "content-addressed begin snapshot differs from the base commit"
            )
        prefix_before[prefix] = normalized_records
        current_prefix_trees[prefix] = _content_addressed_tree(root, prefix)

    expected_code = (
        ()
        if expected_participant is None
        else tuple(
            _normalize_repo_relative(path)
            for path in expected_participant.code_paths
        )
    )
    expected_inputs = (
        set()
        if expected_participant is None
        else {
            _normalize_repo_relative(path)
            for path in expected_participant.inputs
        }
    )
    expected_outputs = (
        set()
        if expected_participant is None
        else {
            _normalize_repo_relative(path)
            for path in expected_participant.outputs
        }
    )
    expected_prefixes = (
        ()
        if expected_participant is None
        else tuple(
            _normalize_output_prefix(path)
            for path in expected_participant.output_prefixes
        )
    )
    expected_records = (
        []
        if expected_participant is None
        else [
            item
            for item in participants
            if item.name == expected_participant.name
        ]
    )
    if descendant_name is None and expected_participant is not None:
        if len(expected_records) != 1:
            raise SignoffArtifactError(
                "pending participant is absent from parent declarations"
            )
        expected_record = expected_records[0]
        if (
            expected_record.code_paths != expected_code
            or set(expected_record.inputs) != expected_inputs
            or set(expected_record.outputs) != expected_outputs
            or expected_record.output_prefixes != expected_prefixes
            or expected_record.descendants != expected_participant.descendants
        ):
            raise SignoffArtifactError(
                "pending participant bindings differ from child"
            )
    if descendant_name is not None:
        if (
            expected_participant is None
            or participant_name is None
            or descendant_name != expected_participant.name
            or pending_index >= len(participants)
        ):
            raise SignoffArtifactError("descendant envelope identity is invalid")
        parent_record = participants[pending_index]
        raw_receipts = raw_by_name[parent_record.name].get("descendant_receipts")
        if not isinstance(raw_receipts, list):
            raise SignoffArtifactError("descendant receipt list is invalid")
        if (
            not set(expected_code).issubset(parent_record.code_paths)
            or not expected_inputs.issubset(parent_record.inputs)
            or not expected_outputs.issubset(parent_record.outputs)
            or not set(expected_prefixes).issubset(parent_record.output_prefixes)
        ):
            raise SignoffArtifactError(
                "descendant declaration is outside parent participant bindings"
            )
        if (
            len(raw_receipts) >= len(parent_record.descendants)
            or parent_record.descendants[len(raw_receipts)] != expected_participant
        ):
            raise SignoffArtifactError(
                "descendant declaration is unknown or out of authorized order"
            )

    dynamic_inputs: dict[str, dict[str, str]] = {}
    descendant_receipts: dict[str, list[dict[str, object]]] = {}
    concrete_cas: set[str] = set()
    claimed_cas: dict[str, str] = {}
    allowed_dirty = {manifest_relative}
    for index, participant in enumerate(participants):
        raw_record = raw_by_name[participant.name]
        raw_descendants = raw_record.get("descendant_receipts")
        if not isinstance(raw_descendants, list):
            raise SignoffArtifactError("descendant receipt list is invalid")
        normalized_descendants: list[dict[str, object]] = []
        descendant_names: set[str] = set()
        descendant_outputs: dict[str, str] = {}
        descendant_cas: dict[str, str] = {}
        required_descendant_fields = {
            "ordinal",
            "name",
            "code_paths",
            "code_sha256",
            "inputs",
            "outputs_after",
            "output_prefixes",
            "content_addressed_outputs_after",
            "receipt_sha256",
        }
        for receipt_index, raw_descendant in enumerate(raw_descendants):
            if (
                not isinstance(raw_descendant, dict)
                or set(raw_descendant) != required_descendant_fields
            ):
                raise SignoffArtifactError("descendant receipt fields are invalid")
            child_name = raw_descendant.get("name")
            child_ordinal = raw_descendant.get("ordinal")
            child_code = raw_descendant.get("code_paths")
            child_inputs = raw_descendant.get("inputs")
            child_outputs = raw_descendant.get("outputs_after")
            child_prefixes = raw_descendant.get("output_prefixes")
            child_cas = raw_descendant.get("content_addressed_outputs_after")
            receipt_sha = raw_descendant.get("receipt_sha256")
            if (
                type(child_ordinal) is not int
                or child_ordinal != receipt_index
                or not isinstance(child_name, str)
                or not child_name
                or child_name in descendant_names
                or not isinstance(child_code, list)
                or not child_code
                or not all(isinstance(path, str) for path in child_code)
                or not isinstance(child_inputs, dict)
                or not isinstance(child_outputs, dict)
                or not child_outputs
                or not isinstance(child_prefixes, list)
                or not all(isinstance(path, str) for path in child_prefixes)
                or not isinstance(child_cas, dict)
                or not _is_sha256(receipt_sha)
                or _descendant_receipt_sha256(raw_descendant) != receipt_sha
            ):
                raise SignoffArtifactError("descendant receipt bindings are invalid")
            normalized_child_code = tuple(
                _normalize_repo_relative(path) for path in child_code
            )
            normalized_child_inputs = {
                _normalize_repo_relative(path): digest
                for path, digest in child_inputs.items()
            }
            normalized_child_outputs = {
                _normalize_repo_relative(path): digest
                for path, digest in child_outputs.items()
            }
            normalized_child_prefixes = tuple(
                _normalize_output_prefix(path) for path in child_prefixes
            )
            if (
                list(normalized_child_code) != child_code
                or not set(normalized_child_code).issubset(participant.code_paths)
                or not set(normalized_child_inputs).issubset(participant.inputs)
                or not set(normalized_child_outputs).issubset(participant.outputs)
                or not set(normalized_child_prefixes).issubset(
                    participant.output_prefixes
                )
                or _code_digest(root, normalized_child_code)
                != raw_descendant.get("code_sha256")
                or receipt_index >= len(participant.descendants)
                or participant.descendants[receipt_index].name != child_name
                or participant.descendants[receipt_index].code_paths
                != normalized_child_code
                or set(participant.descendants[receipt_index].inputs)
                != set(normalized_child_inputs)
                or set(participant.descendants[receipt_index].outputs)
                != set(normalized_child_outputs)
                or participant.descendants[receipt_index].output_prefixes
                != normalized_child_prefixes
            ):
                raise SignoffArtifactError("descendant code/path binding changed")
            for relative, digest in normalized_child_inputs.items():
                if (
                    not _is_sha256(digest)
                    or input_digests[participant.name].get(relative) != digest
                    or _path_digest(root, relative) != digest
                ):
                    raise SignoffArtifactError(
                        f"descendant input binding changed: {relative}"
                    )
            for relative, digest in normalized_child_outputs.items():
                if (
                    relative in descendant_outputs
                    or not _is_sha256(digest)
                    or _ordinary_output_matches_base(
                        root,
                        str(payload["base_commit"]),
                        relative,
                        output_before[participant.name].get(relative),
                    )
                    or _path_digest(root, relative) != digest
                ):
                    raise SignoffArtifactError(
                        f"descendant output binding changed: {relative}"
                    )
                descendant_outputs[relative] = digest
                allowed_dirty.add(relative)
            normalized_child_cas: dict[str, str] = {}
            for relative, digest in child_cas.items():
                normalized_relative = _normalize_repo_relative(relative)
                if (
                    normalized_relative != relative
                    or relative in descendant_cas
                    or relative in concrete_cas
                    or not _is_sha256(digest)
                    or relative in baseline_cas
                    or not any(
                        relative.startswith(prefix + "/")
                        for prefix in normalized_child_prefixes
                    )
                    or _path_digest(root, relative) != digest
                ):
                    raise SignoffArtifactError(
                        f"descendant content-addressed binding changed: {relative}"
                    )
                parts = PurePosixPath(relative).parts
                if (
                    len(parts) <= len(ARTIFACT_BUNDLE_PREFIX.parts)
                    or parts[len(ARTIFACT_BUNDLE_PREFIX.parts)] != digest
                    or _is_lfs_pointer(root.joinpath(*parts))
                ):
                    raise SignoffArtifactError(
                        f"descendant content-addressed digest mismatch: {relative}"
                    )
                descendant_cas[relative] = digest
                concrete_cas.add(relative)
                claimed_cas[relative] = digest
                normalized_child_cas[relative] = digest
                allowed_dirty.add(relative)
            descendant_names.add(child_name)
            normalized_descendants.append(
                {
                    "ordinal": child_ordinal,
                    "name": child_name,
                    "code_paths": list(normalized_child_code),
                    "code_sha256": raw_descendant["code_sha256"],
                    "inputs": normalized_child_inputs,
                    "outputs_after": normalized_child_outputs,
                    "output_prefixes": list(normalized_child_prefixes),
                    "content_addressed_outputs_after": normalized_child_cas,
                    "receipt_sha256": receipt_sha,
                }
            )
        if descendant_name is not None and index == pending_index:
            if descendant_name in descendant_names:
                raise SignoffArtifactError("descendant receipt replay is forbidden")
        descendant_receipts[participant.name] = normalized_descendants
        per_dynamic: dict[str, str] = {}
        for relative, digest in input_digests[participant.name].items():
            producer = declared_output_owner.get(relative)
            if producer is None:
                if not _is_sha256(digest) or _path_digest(root, relative) != digest:
                    raise SignoffArtifactError(
                        f"participant input binding changed: {relative}"
                    )
                continue
            if producer not in participant.predecessors:
                raise SignoffArtifactError(
                    f"participant input has undeclared producer: {relative}"
                )
            per_dynamic[relative] = producer
            if index < pending_index:
                producer_digest = output_after.get(producer, {}).get(relative)
                if producer_digest is None or producer_digest != digest:
                    raise SignoffArtifactError(
                        f"completed predecessor input differs: {relative}"
                    )
        dynamic_inputs[participant.name] = per_dynamic

        completed_record = index < pending_index
        if completed_record and len(raw_descendants) != len(
            participant.descendants
        ):
            raise SignoffArtifactError(
                "completed participant descendant receipts are incomplete"
            )
        if index > pending_index and raw_descendants:
            raise SignoffArtifactError(
                "future participant descendant receipts must be empty"
            )
        raw_outputs_after = raw_record.get("outputs_after")
        raw_cas_after = raw_record.get("content_addressed_outputs_after")
        if completed_record:
            if not isinstance(raw_outputs_after, dict) or not isinstance(
                raw_cas_after, dict
            ):
                raise SignoffArtifactError("completed participant bindings are incomplete")
        elif raw_outputs_after is not None or raw_cas_after is not None:
            raise SignoffArtifactError("pending/future participant is already recorded")

        if completed_record:
            if set(raw_outputs_after) != set(participant.outputs):
                raise SignoffArtifactError("completed output bindings differ")
            normalized_after: dict[str, str] = {}
            for relative, digest in raw_outputs_after.items():
                if (
                    not isinstance(relative, str)
                    or not _is_sha256(digest)
                    or _ordinary_output_matches_base(
                        root,
                        str(payload["base_commit"]),
                        relative,
                        output_before[participant.name].get(relative),
                    )
                    or _path_digest(root, relative) != digest
                ):
                    raise SignoffArtifactError(
                        f"completed output binding changed: {relative}"
                    )
                normalized_after[relative] = digest
                allowed_dirty.add(relative)
            output_after[participant.name] = normalized_after
            normalized_cas: dict[str, str] = {}
            for relative, digest in raw_cas_after.items():
                if (
                    not isinstance(relative, str)
                    or not _is_sha256(digest)
                    or relative in baseline_cas
                    or relative in concrete_cas
                    or not any(
                        relative.startswith(prefix + "/")
                        for prefix in participant.output_prefixes
                    )
                    or _path_digest(root, relative) != digest
                ):
                    raise SignoffArtifactError(
                        f"content-addressed output binding changed: {relative}"
                    )
                parts = PurePosixPath(relative).parts
                if (
                    len(parts) <= len(ARTIFACT_BUNDLE_PREFIX.parts)
                    or parts[len(ARTIFACT_BUNDLE_PREFIX.parts)] != digest
                    or _is_lfs_pointer(root.joinpath(*parts))
                ):
                    raise SignoffArtifactError(
                        f"content-addressed output digest mismatch: {relative}"
                    )
                concrete_cas.add(relative)
                claimed_cas[relative] = digest
                normalized_cas[relative] = digest
                allowed_dirty.add(relative)
            cas_after[participant.name] = normalized_cas
        else:
            for relative, before in output_before[participant.name].items():
                descendant_digest = descendant_outputs.get(relative)
                if descendant_digest is not None:
                    unchanged = _path_digest(root, relative) == descendant_digest
                else:
                    unchanged = _ordinary_output_matches_base(
                        root,
                        str(payload["base_commit"]),
                        relative,
                        before,
                    )
                recording_current_parent_output = (
                    allow_current_participant_outputs
                    and index == pending_index
                    and expected_participant is not None
                    and participant.name == expected_participant.name
                    and len(raw_descendants) == len(participant.descendants)
                    and descendant_digest is None
                )
                if recording_current_parent_output:
                    allowed_dirty.add(relative)
                if not unchanged and not recording_current_parent_output:
                    raise SignoffArtifactError(
                        f"pending/future output changed before launch: {relative}"
                    )

    prefix_state: dict[str, dict[str, str]] = {}
    for prefix in sorted(all_prefixes):
        expected_tree = dict(prefix_before[prefix])
        expected_tree.update(
            {
                relative: digest
                for relative, digest in claimed_cas.items()
                if relative == prefix or relative.startswith(prefix + "/")
            }
        )
        if current_prefix_trees[prefix] != expected_tree:
            raise SignoffArtifactError(
                "content-addressed tree differs from its begin snapshot and "
                f"authorized append-only delta: {prefix}"
            )
        prefix_state[prefix] = expected_tree

    authoritative_paths = payload.get("authoritative_input_paths")
    if (
        not isinstance(authoritative_paths, list)
        or not all(isinstance(path, str) for path in authoritative_paths)
        or authoritative_paths != sorted(set(authoritative_paths))
        or any(_normalize_repo_relative(path) != path for path in authoritative_paths)
    ):
        raise SignoffArtifactError("authoritative closure paths are invalid")
    closure_extras = set(authoritative_paths)
    closure_extras.update(
        path for participant in participants for path in participant.code_paths
    )
    closure_extras.update(
        path for participant in participants for path in participant.inputs
    )
    all_outputs = {
        path for participant in participants for path in participant.outputs
    }
    snapshot = authoritative_input_closure(
        root,
        extra_paths=tuple(sorted(closure_extras)),
        excluded_outputs=tuple(sorted(all_outputs | {manifest_relative})),
    )
    if (
        snapshot.base_commit != payload.get("base_commit")
        or list(snapshot.paths) != authoritative_paths
        or snapshot.sha256 != payload.get("authoritative_input_closure_sha256")
    ):
        raise SignoffArtifactError("authoritative closure binding changed")
    unexpected = [
        (status, relative)
        for status, relative in _porcelain_entries(root)
        if relative not in allowed_dirty
    ]
    if unexpected:
        raise SignoffArtifactError(
            "undeclared worktree change before participant: "
            + ", ".join(f"{status} {path}" for status, path in unexpected)
        )

    transaction = SignedRunTransaction(
        root,
        manifest,
        tuple(participants),
        snapshot,
        input_digests,
        dynamic_inputs,
        output_before,
        prefix_before,
        code_digests,
    )
    transaction.prefix_state = prefix_state
    transaction.output_after = output_after
    transaction.content_addressed_output_after = cas_after
    transaction.descendant_receipts = descendant_receipts
    transaction.completed = list(completed)
    return transaction


def _verified_parent_signed_run(
    root: Path,
    *,
    manifest_value: str,
    participant_name: str,
    accepted_participants: Sequence[str],
    expected_participant: SignedRunParticipant,
    parent_contract: SignedRunParentContract | None = None,
    descendant_name: str | None = None,
) -> tuple[str, str, SignedRunTransaction]:
    manifest, manifest_relative = _confined_manifest_path(root, manifest_value)
    try:
        _require_live_signed_run_owner(root, manifest)
        transaction = _load_pending_parent_transaction(
            root,
            manifest,
            manifest_relative,
            participant_name=participant_name,
            accepted_participants=accepted_participants,
            expected_participant=expected_participant,
            descendant_name=descendant_name,
        )
    except SignoffArtifactError as exc:
        raise SignoffArtifactError(
            f"parent signed transaction is invalid: {exc}"
        ) from exc
    if parent_contract is not None:
        ordered = tuple(item.name for item in transaction.participants)
        pending_index = len(transaction.completed)
        if (
            manifest_relative
            != _normalize_repo_relative(parent_contract.manifest_path)
            or ordered != parent_contract.ordered_participants
            or pending_index >= len(transaction.participants)
            or transaction.participants[pending_index].name
            != parent_contract.participant_name
            or transaction.participants[pending_index].predecessors
            != parent_contract.predecessors
        ):
            raise SignoffArtifactError(
                "parent signed transaction graph differs from the canonical child contract"
            )
    return manifest_relative, participant_name, transaction


def begin_direct_signed_run(
    repo_root: str | Path,
    manifest_path: str | Path,
    participant: SignedRunParticipant,
    *,
    preflight: Callable[[], AuthoritativeInputSnapshot],
    accepted_parent_participants: Sequence[str] = (),
    accepted_parent_contracts: Sequence[SignedRunParentContract] = (),
    environment: Mapping[str, str] | None = None,
    require_parent: bool = False,
) -> DirectSignedRun:
    """Run the shared preflight once, then own or join one signed participant."""

    root = _repo_root(repo_root)
    snapshot = preflight()
    if not isinstance(snapshot, AuthoritativeInputSnapshot):
        raise SignoffArtifactError(
            "direct signed-run preflight did not return an authoritative snapshot"
        )
    active_environment = os.environ if environment is None else environment
    parent_manifest = active_environment.get(SIGNED_RUN_PARENT_MANIFEST_ENV)
    parent_participant = active_environment.get(SIGNED_RUN_PARENT_PARTICIPANT_ENV)
    parent_descendant = active_environment.get(SIGNED_RUN_PARENT_DESCENDANT_ENV)
    parent_owner_pid = active_environment.get(SIGNED_RUN_PARENT_OWNER_PID_ENV)
    if bool(parent_manifest) != bool(parent_participant):
        raise SignoffArtifactError("parent signed transaction environment is incomplete")
    if parent_descendant and not parent_manifest:
        raise SignoffArtifactError("parent descendant environment is incomplete")
    if parent_owner_pid and not parent_manifest:
        raise SignoffArtifactError("parent owner environment is incomplete")
    if parent_manifest and parent_participant:
        parent_contract: SignedRunParentContract | None = None
        if accepted_parent_contracts:
            normalized_contracts: list[SignedRunParentContract] = []
            for raw_contract in accepted_parent_contracts:
                manifest_relative = _normalize_repo_relative(
                    raw_contract.manifest_path
                )
                ordered = tuple(raw_contract.ordered_participants)
                predecessors = tuple(raw_contract.predecessors)
                if (
                    not ordered
                    or len(ordered) > 4
                    or len(set(ordered)) != len(ordered)
                    or raw_contract.participant_name not in ordered
                    or any(not name for name in ordered)
                    or any(
                        predecessor
                        not in ordered[: ordered.index(raw_contract.participant_name)]
                        for predecessor in predecessors
                    )
                ):
                    raise SignoffArtifactError(
                        "signed child parent contract is invalid"
                    )
                normalized_contracts.append(
                    SignedRunParentContract(
                        manifest_path=manifest_relative,
                        ordered_participants=ordered,
                        participant_name=raw_contract.participant_name,
                        predecessors=predecessors,
                    )
                )
            matches = [
                contract
                for contract in normalized_contracts
                if contract.manifest_path
                == _normalize_repo_relative(parent_manifest)
                and contract.participant_name == parent_participant
            ]
            if len(matches) != 1:
                raise SignoffArtifactError(
                    "parent manifest identity is not authorized for this signed child"
                )
            parent_contract = matches[0]
            try:
                owner_pid = int(parent_owner_pid or "")
            except ValueError as exc:
                raise SignoffArtifactError(
                    "parent owner process binding is invalid"
                ) from exc
            if owner_pid <= 0 or owner_pid != os.getppid():
                raise SignoffArtifactError(
                    "parent owner process does not own this signed child"
                )
            accepted = (parent_contract.participant_name,)
        else:
            accepted = tuple(accepted_parent_participants) or (participant.name,)
        manifest_relative, owned_name, transaction = _verified_parent_signed_run(
            root,
            manifest_value=parent_manifest,
            participant_name=parent_participant,
            accepted_participants=accepted,
            expected_participant=participant,
            parent_contract=parent_contract,
            descendant_name=parent_descendant,
        )
        return DirectSignedRun(
            root,
            manifest_relative,
            owned_name,
            transaction,
            participant if parent_descendant else None,
        )
    if require_parent:
        raise SignoffArtifactError(
            "resource child requires a verified parent signed transaction envelope"
        )
    transaction = SignedRunTransaction.begin(
        root,
        manifest_path,
        (participant,),
        authoritative_snapshot=snapshot,
    )
    return DirectSignedRun(
        root,
        transaction.manifest.relative_to(root).as_posix(),
        participant.name,
        transaction,
    )


def finalize_signed_run_transaction(
    repo_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_participants: Sequence[str],
) -> dict[str, object]:
    """Act only as a byte finalizer for one complete, current-base transaction."""

    root = _repo_root(repo_root)
    manifest, manifest_relative = _confined_manifest_path(root, manifest_path)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SignoffArtifactError("signed transaction manifest is missing or invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SignedRunTransaction.SCHEMA:
        raise SignoffArtifactError("signed transaction schema is unsupported")
    if payload.get("finalized") is True:
        raise SignoffArtifactError("signed transaction replay is forbidden")
    if payload.get("base_commit") != _head(root):
        raise SignoffArtifactError("signed transaction replay after base commit is forbidden")
    ordered = payload.get("ordered_participants")
    expected = list(expected_participants)
    if ordered != expected or len(expected) > 4:
        raise SignoffArtifactError("signed transaction has an unknown or out-of-order participant")
    if payload.get("completed_participants") != expected:
        raise SignoffArtifactError("signed transaction participant set is incomplete")
    raw_participants = payload.get("participants")
    if not isinstance(raw_participants, list) or len(raw_participants) != len(expected):
        raise SignoffArtifactError("signed transaction participant records are incomplete")
    validated = _load_pending_parent_transaction(
        root,
        manifest,
        manifest_relative,
        participant_name=None,
        accepted_participants=(),
        expected_participant=None,
    )
    if validated._payload() != payload:
        raise SignoffArtifactError(
            "signed transaction manifest differs from its validated bindings"
        )
    declared_output_owner: dict[str, str] = {}
    for participant_name, raw_record in zip(expected, raw_participants):
        if not isinstance(raw_record, dict) or not isinstance(
            raw_record.get("outputs_after"), dict
        ):
            raise SignoffArtifactError(
                "signed transaction participant output records are incomplete"
            )
        for raw_relative in raw_record["outputs_after"]:
            if not isinstance(raw_relative, str):
                raise SignoffArtifactError(
                    "signed transaction output path is invalid"
                )
            relative = _normalize_repo_relative(raw_relative)
            if relative in declared_output_owner:
                raise SignoffArtifactError(
                    f"cross-participant output is forbidden: {relative}"
                )
            declared_output_owner[relative] = participant_name
    allowed = {manifest_relative}
    allowed.update(
        relative
        for receipts in validated.descendant_receipts.values()
        for receipt in receipts
        for relative in dict(receipt["content_addressed_outputs_after"])
    )
    raw_authoritative_paths = payload.get("authoritative_input_paths")
    if not isinstance(raw_authoritative_paths, list) or not all(
        isinstance(path, str) for path in raw_authoritative_paths
    ):
        raise SignoffArtifactError(
            "signed transaction authoritative input paths are invalid"
        )
    extra_inputs: set[str] = {
        _normalize_repo_relative(path) for path in raw_authoritative_paths
    }
    concrete_outputs: set[str] = set()
    participant_outputs: dict[str, dict[str, str]] = {}
    for index, record in enumerate(raw_participants):
        if not isinstance(record, dict):
            raise SignoffArtifactError("signed transaction participant record is invalid")
        if record.get("name") != expected[index]:
            raise SignoffArtifactError(
                "signed transaction participant record is out of order"
            )
        code_paths = record.get("code_paths")
        inputs = record.get("inputs")
        outputs = record.get("outputs_after")
        outputs_before = record.get("outputs_before")
        output_prefixes = record.get("output_prefixes")
        cas_outputs = record.get("content_addressed_outputs_after")
        predecessors = record.get("predecessors")
        if (
            not isinstance(code_paths, list)
            or not all(isinstance(path, str) for path in code_paths)
            or not isinstance(inputs, dict)
            or not isinstance(outputs, dict)
            or not isinstance(outputs_before, dict)
            or not isinstance(output_prefixes, list)
            or not all(isinstance(path, str) for path in output_prefixes)
            or not isinstance(cas_outputs, dict)
            or not isinstance(predecessors, list)
            or not all(isinstance(name, str) for name in predecessors)
        ):
            raise SignoffArtifactError("signed transaction participant bindings are incomplete")
        normalized_code = tuple(_normalize_repo_relative(path) for path in code_paths)
        if list(normalized_code) != code_paths:
            raise SignoffArtifactError("signed transaction code paths are non-canonical")
        extra_inputs.update(normalized_code)
        if _code_digest(root, normalized_code) != record.get("code_sha256"):
            raise SignoffArtifactError("signed transaction participant code changed")
        if any(name not in expected[:index] for name in predecessors):
            raise SignoffArtifactError(
                "signed transaction has undeclared/future predecessor edges"
            )
        for relative, digest in inputs.items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise SignoffArtifactError("signed transaction input binding is invalid")
            normalized = _normalize_repo_relative(relative)
            if normalized != relative:
                raise SignoffArtifactError("signed transaction input path is non-canonical")
            producer = declared_output_owner.get(normalized)
            if producer is not None:
                if producer not in predecessors or producer not in expected[:index]:
                    raise SignoffArtifactError(
                        f"signed transaction input consumes undeclared predecessor output: {relative}"
                    )
                if participant_outputs.get(producer, {}).get(normalized) != digest:
                    raise SignoffArtifactError(
                        f"signed transaction predecessor input digest differs: {relative}"
                    )
            else:
                extra_inputs.add(normalized)
            if _path_digest(root, normalized) != digest:
                raise SignoffArtifactError(f"signed transaction input changed: {relative}")
        if set(outputs) != set(outputs_before):
            raise SignoffArtifactError(
                "signed transaction pre/post output bindings differ"
            )
        normalized_after: dict[str, str] = {}
        for relative, digest in outputs.items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise SignoffArtifactError("signed transaction output binding is invalid")
            normalized = _normalize_repo_relative(relative)
            if normalized != relative:
                raise SignoffArtifactError("signed transaction output path is non-canonical")
            if _ordinary_output_matches_base(
                root,
                str(payload["base_commit"]),
                relative,
                outputs_before.get(relative),
            ):
                raise SignoffArtifactError(
                    f"signed transaction output is unchanged; replay forbidden: {relative}"
                )
            if normalized in concrete_outputs:
                raise SignoffArtifactError(
                    f"cross-participant output is forbidden: {normalized}"
                )
            concrete_outputs.add(normalized)
            normalized_after[normalized] = digest
            allowed.add(normalized)
            if _path_digest(root, normalized) != digest:
                raise SignoffArtifactError(f"signed transaction output changed: {relative}")
        participant_outputs[expected[index]] = normalized_after
        prefixes = tuple(_normalize_output_prefix(path) for path in output_prefixes)
        if list(prefixes) != output_prefixes:
            raise SignoffArtifactError(
                "signed transaction output prefixes are non-canonical"
            )
        for relative, digest in cas_outputs.items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise SignoffArtifactError(
                    "signed transaction content-addressed output binding is invalid"
                )
            normalized = _normalize_repo_relative(relative)
            if normalized != relative or not any(
                normalized == prefix or normalized.startswith(prefix + "/")
                for prefix in prefixes
            ):
                raise SignoffArtifactError(
                    f"content-addressed output is outside its declared prefix: {relative}"
                )
            if normalized in concrete_outputs:
                raise SignoffArtifactError(
                    f"cross-participant output is forbidden: {normalized}"
                )
            concrete_outputs.add(normalized)
            allowed.add(normalized)
            if _path_digest(root, normalized) != digest:
                raise SignoffArtifactError(
                    f"signed transaction content-addressed output changed: {relative}"
                )
            parts = PurePosixPath(normalized).parts
            directory_digest = parts[len(ARTIFACT_BUNDLE_PREFIX.parts)]
            if directory_digest != digest or _is_lfs_pointer(
                root.joinpath(*PurePosixPath(normalized).parts)
            ):
                raise SignoffArtifactError(
                    f"signed transaction content-addressed output digest mismatch: {relative}"
                )
    entries = _porcelain_entries(root)
    staged = [(status, path) for status, path in entries if status[0] not in {" ", "?"}]
    if staged:
        raise SignoffArtifactError(
            "signed transaction finalizer refuses staged files: "
            + ", ".join(f"{status} {path}" for status, path in staged)
        )
    unexpected = [(status, path) for status, path in entries if path not in allowed]
    if unexpected:
        raise SignoffArtifactError(
            "signed transaction has undeclared output or untracked shadow: "
            + ", ".join(f"{status} {path}" for status, path in unexpected)
        )
    final_snapshot = require_clean_authoritative_inputs(
        root,
        extra_paths=tuple(sorted(extra_inputs)),
        allowed_outputs=tuple(sorted(allowed)),
    )
    if (
        final_snapshot.base_commit != payload.get("base_commit")
        or final_snapshot.sha256
        != payload.get("authoritative_input_closure_sha256")
    ):
        raise SignoffArtifactError(
            "signed transaction authoritative input closure changed"
        )
    payload["finalized"] = True
    handle, temporary_name = tempfile.mkstemp(
        prefix=manifest.name + ".", suffix=".tmp", dir=manifest.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, manifest)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return payload


__all__ = [
    "ARTIFACT_BUNDLE_PREFIX",
    "ARTIFACT_MANIFEST_SCHEMA",
    "SignoffArtifactError",
    "AuthoritativeInputSnapshot",
    "DirectSignedRun",
    "SIGNED_RUN_PARENT_DESCENDANT_ENV",
    "SIGNED_RUN_PARENT_MANIFEST_ENV",
    "SIGNED_RUN_PARENT_OWNER_PID_ENV",
    "SIGNED_RUN_PARENT_PARTICIPANT_ENV",
    "SignedRunParentContract",
    "SignedRunParticipant",
    "SignedRunTransaction",
    "authoritative_inputs_unchanged_between",
    "authoritative_input_closure",
    "begin_direct_signed_run",
    "finalize_signed_run_transaction",
    "package_artifact",
    "package_artifact_manifest",
    "require_clean_authoritative_inputs",
    "require_tracked_repo_input",
    "verify_artifact_graph",
    "verify_artifact_locator",
]
