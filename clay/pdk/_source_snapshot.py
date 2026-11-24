"""Immutable exact-source snapshots for commercial PDK deck generation.

Foundry source bytes are read into memory only.  The persisted runtime cache
contains generated KLayout decks, addressed by their exact content; proprietary
source payloads are never copied into the repository or the cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional

_DECK_CAS_SCHEMA = "v1"


@dataclass(frozen=True)
class PdkSourceBlob:
    role: str
    relative_path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class PdkSourceSnapshot:
    pdk_id: str
    root: Path
    sources: tuple[PdkSourceBlob, ...]
    installed_authority_sha256: Optional[str]
    _rules_json: str
    _layermap_items: tuple[tuple[str, int, int], ...]
    snapshot_sha256: str

    def rules(self) -> dict[str, dict]:
        """Return a defensive copy of parsed rule facts."""

        return json.loads(self._rules_json)

    def layermap(self) -> dict[str, tuple[int, int]]:
        """Return a defensive copy of captured layer facts."""

        return {
            token: (int(layer), int(datatype))
            for token, layer, datatype in self._layermap_items
        }


@dataclass(frozen=True)
class _CapturedBytes:
    blob: PdkSourceBlob
    payload: bytes


def _confined_regular_path(root: Path, raw: Path) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = Path(raw).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"PDK source {candidate} escapes install root {resolved_root}"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"PDK source is not a regular file: {candidate}")
    return candidate


def _stable_read(root: Path, role: str, raw: Path) -> _CapturedBytes:
    path = _confined_regular_path(root, raw)

    def signature() -> tuple[int, int, int, int]:
        stat = path.stat()
        return (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
            int(getattr(stat, "st_ino", 0)),
        )

    before = signature()
    payload = path.read_bytes()
    after = signature()
    if after != before:
        raise ValueError(f"PDK source changed while reading: {path}")
    if not payload:
        raise ValueError(f"PDK source is empty: {path}")
    relative = path.relative_to(root.resolve(strict=True)).as_posix()
    return _CapturedBytes(
        blob=PdkSourceBlob(
            role=str(role),
            relative_path=relative,
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        ),
        payload=payload,
    )


def _selected_sources(root: Path) -> tuple[Optional[Path], Optional[Path], str, Optional[Path]]:
    from clay.frontend.pdk_layermap import _detect_source
    from clay.frontend.pdk_register import find_antenna_deck, find_drc_deck

    drc = find_drc_deck(str(root))
    antenna = find_antenna_deck(str(root))
    layermap_kind, layermap = _detect_source(root)
    return drc, antenna, str(layermap_kind), layermap


def _normalized_selection(
    root: Path,
    selected: tuple[Optional[Path], Optional[Path], str, Optional[Path]],
) -> tuple[Optional[str], Optional[str], str, Optional[str]]:
    drc, antenna, kind, layermap = selected

    def relative(path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        confined = _confined_regular_path(root, path)
        return confined.relative_to(root.resolve(strict=True)).as_posix()

    return relative(drc), relative(antenna), kind, relative(layermap)


def capture_pdk_source_snapshot(adapter: Any, root: Path) -> PdkSourceSnapshot:
    """Capture one root, its selected locators, exact bytes, and parsed facts."""

    resolved_root = Path(root).resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError(f"PDK install root is not a directory: {resolved_root}")

    before_raw = _selected_sources(resolved_root)
    before = _normalized_selection(resolved_root, before_raw)
    drc_path, antenna_path, layermap_kind, layermap_path = before_raw

    captured: dict[str, _CapturedBytes] = {}
    captured_by_path: dict[Path, _CapturedBytes] = {}

    def capture(role: str, raw: Optional[Path]) -> None:
        if raw is None:
            return
        path = _confined_regular_path(resolved_root, raw)
        exact = captured_by_path.get(path)
        if exact is None:
            exact = _stable_read(resolved_root, role, path)
            captured_by_path[path] = exact
        captured[role] = _CapturedBytes(
            blob=PdkSourceBlob(
                role=role,
                relative_path=exact.blob.relative_path,
                sha256=exact.blob.sha256,
                size=exact.blob.size,
            ),
            payload=exact.payload,
        )

    # One physical file can be both the DRC-rule and layermap authority.  Read
    # each unique path once so two parsed fact classes cannot observe different
    # generations of that file.
    capture("drc", drc_path)
    capture("antenna", antenna_path)
    capture("layermap", layermap_path)

    after = _normalized_selection(resolved_root, _selected_sources(resolved_root))
    if after != before:
        raise ValueError(
            "PDK source selection changed while capturing: "
            f"before={before!r}, after={after!r}"
        )

    rules: dict[str, dict] = {}
    if "drc" in captured:
        from clay.frontend.pdk_register import parsed_rules_from_deck_bytes

        rules = dict(
            parsed_rules_from_deck_bytes(
                captured["drc"].payload,
                antenna_bytes=(
                    captured["antenna"].payload
                    if "antenna" in captured
                    else None
                ),
            )
        )

    parsed_layermap: dict[str, tuple[int, int]] = {}
    if "layermap" in captured:
        from clay.frontend.pdk_layermap import parse_layermap_bytes

        parsed_layermap = parse_layermap_bytes(
            layermap_kind, captured["layermap"].payload
        )

    extras: Mapping[str, tuple[int, int]] = {}
    installed_digest: Optional[str] = None
    installed_blobs: tuple[PdkSourceBlob, ...] = ()
    raw_extras = getattr(adapter, "layermap_extra", {})
    capture_at_root = getattr(raw_extras, "capture_at_root", None)
    snapshot_at_root = getattr(raw_extras, "snapshot_at_root", None)
    if callable(capture_at_root):
        captured_extras, authority_capture = capture_at_root(resolved_root)
        extras = dict(captured_extras)
        installed_digest = authority_capture.sha256
        installed_blobs = tuple(
            PdkSourceBlob(
                role=f"installed-authority:{ordinal}",
                relative_path=source.relative_path,
                sha256=source.sha256,
                size=source.size,
            )
            for ordinal, source in enumerate(authority_capture.sources)
        )
    elif callable(snapshot_at_root):
        captured_extras, installed_digest = snapshot_at_root(resolved_root)
        extras = dict(captured_extras)
    else:
        extras = dict(raw_extras)

    layermap = dict(parsed_layermap)
    layermap.update(extras)
    if rules and "drc" not in captured:
        raise ValueError("non-empty PDK rules lack an exact source")
    if parsed_layermap and "layermap" not in captured:
        raise ValueError("non-empty PDK layermap lacks an exact source")
    if extras and installed_digest is None and snapshot_at_root is not None:
        raise ValueError("installed layermap facts lack captured authority")

    final_selection = _normalized_selection(
        resolved_root, _selected_sources(resolved_root)
    )
    if final_selection != before:
        raise ValueError(
            "PDK source selection changed while capturing: "
            f"before={before!r}, after={final_selection!r}"
        )
    for path, exact in captured_by_path.items():
        verified = _stable_read(resolved_root, exact.blob.role, path)
        if verified.payload != exact.payload:
            raise ValueError(f"PDK source changed while capturing: {path}")

    rules_json = json.dumps(
        rules, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    layermap_items = tuple(
        (str(token), int(value[0]), int(value[1]))
        for token, value in sorted(layermap.items())
    )
    blobs = tuple(
        sorted(
            tuple(item.blob for item in captured.values()) + installed_blobs,
            key=lambda blob: (blob.role, blob.relative_path),
        )
    )
    identity_payload = {
        "schema": "clay.pdk-source-snapshot.v1",
        "pdk_id": str(adapter.pdk_id),
        "sources": [
            {
                "role": blob.role,
                "relative_path": blob.relative_path,
                "sha256": blob.sha256,
                "size": blob.size,
            }
            for blob in blobs
        ],
        "installed_authority_sha256": installed_digest,
        "rules_sha256": hashlib.sha256(rules_json.encode("utf-8")).hexdigest(),
        "layermap": layermap_items,
    }
    snapshot_sha256 = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return PdkSourceSnapshot(
        pdk_id=str(adapter.pdk_id),
        root=resolved_root,
        sources=blobs,
        installed_authority_sha256=installed_digest,
        _rules_json=rules_json,
        _layermap_items=layermap_items,
        snapshot_sha256=snapshot_sha256,
    )


def publish_content_addressed_deck(
    *,
    pdk_id: str,
    snapshot_sha256: str,
    suffix: str,
    text: str,
) -> Path:
    """Atomically publish immutable generated deck bytes into the runtime CAS."""

    if suffix not in {"drc", "lvs"}:
        raise ValueError(f"unsupported generated deck suffix: {suffix!r}")
    payload = text.encode("utf-8")
    content_sha256 = hashlib.sha256(payload).hexdigest()
    safe_pdk = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(pdk_id)
    )
    cache = Path(tempfile.gettempdir()) / "clay_pdk_decks"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (
        f"{safe_pdk}.{_DECK_CAS_SCHEMA}.{snapshot_sha256}."
        f"{content_sha256}.{suffix}"
    )

    def verify_existing() -> Path:
        try:
            existing = target.read_bytes()
        except OSError as exc:
            raise ValueError(f"generated deck CAS entry is unreadable: {target}") from exc
        if existing != payload:
            raise ValueError(
                f"generated deck CAS content mismatch for immutable path: {target}"
            )
        return target

    if target.exists():
        return verify_existing()

    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache,
        prefix=f".{safe_pdk}.{snapshot_sha256}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except OSError:
            if not target.exists():
                raise
            verify_existing()
    finally:
        temporary.unlink(missing_ok=True)
    return verify_existing()
