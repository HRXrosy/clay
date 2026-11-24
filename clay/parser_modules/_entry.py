"""
Public parse entry points (v0.1.13 parser split).

Extracted from `clay/parser.py` (4467 LOC) in v0.1.13 (R90 Agent B).

  * `parse_clay_source(text)` — parses an in-memory string.
  * `parse_clay_file(path)`   — parses a file, recursively loading
                                 any `import './lib.clay';` references
                                 and merging their cells in.

Both build a `ClayFile` IR, run the `_attach_clay_file_to_cell_defs`
back-edge stash and `_resolve_verify_refs`, and (in opt-in mode) run
the v0.0.97 R74 D Rust-Python equivalence audit.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import threading
from typing import Any

from lark.exceptions import VisitError

from ..ir import ClayFile
from ._ast_builder import _ClayTransformer
from ._grammar import _get_parser
from ._resolution import _attach_clay_file_to_cell_defs, _resolve_verify_refs
from ._rust_delegation import _audit_rust_role_digital_delegation
from .file_imports import _merge_imported_cells, _resolve_file_import

_PARSE_TREE_CACHE_MAX = 64
_PARSE_TREE_CACHE: OrderedDict[tuple[int, bytes], tuple[Any, str, Any]] = OrderedDict()
_PARSE_TREE_CACHE_LOCK = threading.Lock()


def _parse_cache_disabled() -> bool:
    val = os.environ.get("CLAY_DISABLE_PARSE_CACHE", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def clear_parse_tree_cache() -> None:
    """Clear the bounded Lark parse-tree LRU cache (for tests and A/B checks)."""
    with _PARSE_TREE_CACHE_LOCK:
        _PARSE_TREE_CACHE.clear()


def _parse_text_to_tree(text: str):
    """Parse normalized Clay source text into a Lark Tree with bounded LRU caching.

    Callers always read the real bytes from disk (or receive the live string)
    before calling this helper, so disk edits / TOCTOU / tamper checks stay 100%
    intact. Only the immutable ``lark.Tree`` is cached; ``_ClayTransformer`` runs
    fresh on every call without mutating the cached tree, producing a brand-new
    ``ClayFile`` IR with path-specific source spans.
    """
    parser = _get_parser()
    if _parse_cache_disabled():
        return parser.parse(text)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    key = (id(parser), digest)
    with _PARSE_TREE_CACHE_LOCK:
        entry = _PARSE_TREE_CACHE.get(key)
        if entry is not None and entry[0] is parser and entry[1] == text:
            _PARSE_TREE_CACHE.move_to_end(key)
            return entry[2]
    tree = parser.parse(text)
    with _PARSE_TREE_CACHE_LOCK:
        _PARSE_TREE_CACHE[key] = (parser, text, tree)
        _PARSE_TREE_CACHE.move_to_end(key)
        while len(_PARSE_TREE_CACHE) > _PARSE_TREE_CACHE_MAX:
            _PARSE_TREE_CACHE.popitem(last=False)
    return tree


def _transform(tree) -> ClayFile:
    """Run the AST transformer, unwrapping lark's ``VisitError``.

    Structural-rejection rules in the transformer (duplicate device name,
    pin-assigned-twice, device/port collision, ...) ``raise ValueError`` from
    inside a transformer callback. Lark wraps any non-``GrammarAnalyzer`` error
    raised in a callback as a ``VisitError`` whose ``.orig_exc`` is the real
    exception. Leaking ``VisitError`` to callers breaks the parser's contract of
    raising a clear, catchable domain error (``ValueError`` / typed parse error)
    and surfaces lark-transformer internals in tracebacks. Re-raise the original
    so a malformed input yields the clean typed error the rule intended.
    """
    try:
        return _ClayTransformer().transform(tree)
    except VisitError as exc:
        raise exc.orig_exc from exc


def parse_clay_source(text: str) -> ClayFile:
    """Parse Clay source text into a ClayFile IR.

    File-path imports are NOT followed in this entry point — they
    can't be, because we don't have a directory context. If the
    source contains any `import 'path.clay';` directives, this
    function raises ValueError. Callers that need file-import
    support should use `parse_clay_file` instead.

    v0.0.97 R74 D: when `CLAY_USE_RUST_PARSER=1` is set, an audit hook
    re-runs the role_decl + digital_decl expansion through the Rust
    semantic transformers and surfaces any drift in
    `cf._rust_audit_mismatches`. The Python IR is the source-of-truth
    either way — mismatches never change the returned ClayFile.
    """
    # A leading BOM glyph (U+FEFF) on a pre-decoded string would otherwise hit
    # the grammar as a stray char and raise a misleading UnexpectedCharacters at
    # col 1; strip it so an in-memory string read with a plain UTF-8 decode (no
    # -sig) parses the same as the file path entry point.
    if text and text[0] == "﻿":
        text = text[1:]
    tree = _parse_text_to_tree(text)
    cf: ClayFile = _transform(tree)
    pending_files = cf.__dict__.get("_pending_file_imports", [])
    if pending_files:
        raise ValueError(
            "file-path imports require a file context; "
            "call parse_clay_file(...) instead of "
            "parse_clay_source(...). Offending import(s): "
            f"{[p for p, _ in pending_files]}"
        )
    cf.__dict__.pop("_pending_file_imports", None)
    _attach_clay_file_to_cell_defs(cf)
    _resolve_verify_refs(cf)
    # v0.0.97 R74 D: opt-in Rust-Python equivalence audit. Gated on
    # `CLAY_USE_RUST_PARSER=1`; no-op otherwise. The audit is purely
    # observational — it never modifies the returned IR.
    _audit_rust_role_digital_delegation(cf, text)
    return cf


def parse_clay_file(
    path: str | Path,
    _import_stack: tuple[Path, ...] = (),
    *,
    _source_bytes: bytes | None = None,
) -> ClayFile:
    """Parse a .clay file from disk into a ClayFile IR.

    v0.0.28: also recursively loads any `import './path.clay';`
    directives, merging the imported file's cell_aliases /
    cell_defs into this file's IR. The `_import_stack` parameter
    is internal — it tracks files currently being parsed for cycle
    detection. External callers should NOT pass it.
    """
    abs_path = Path(path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(
            f"clay file not found: {abs_path}"
        )
    if abs_path in _import_stack:
        # Cycle: this file is already being parsed somewhere above
        # in the import chain.
        chain = " -> ".join(str(p) for p in _import_stack) + f" -> {abs_path}"
        raise ValueError(
            f"file import cycle detected: {chain}"
        )
    new_stack = _import_stack + (abs_path,)
    # Read as UTF-8-SIG so a leading byte-order mark (common on Windows-authored
    # files) is stripped rather than fed to the grammar as a stray glyph — a BOM
    # otherwise surfaces as a misleading `UnexpectedCharacters at col 1` on an
    # otherwise-valid file. A genuinely non-UTF-8 file is re-raised as a clear,
    # file-named ValueError instead of a bare UnicodeDecodeError, matching the
    # FileNotFoundError/ValueError decoration the rest of this function uses.
    try:
        if _source_bytes is not None and type(_source_bytes) is not bytes:
            raise TypeError('bound Clay source must be bytes')
        text = (abs_path.read_text(encoding="utf-8-sig") if _source_bytes is None
                else _source_bytes.decode('utf-8-sig'))
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"clay file is not valid UTF-8: {abs_path} "
            f"({exc.reason} at byte {exc.start})"
        ) from exc
    tree = _parse_text_to_tree(text)
    cf: ClayFile = _transform(tree)
    _attach_source_file_to_spans(cf, abs_path)

    # Recursively process file imports, merging their cells in.
    pending: list[tuple[str, str | None]] = cf.__dict__.pop(
        "_pending_file_imports", []
    )
    for raw_path, alias in pending:
        try:
            target = _resolve_file_import(raw_path, abs_path)
        except ValueError as exc:
            raise ValueError(
                f"in {abs_path}: {exc}"
            ) from exc
        if not target.exists():
            raise FileNotFoundError(
                f"in {abs_path}: imported file not found: {target} "
                f"(from `import {raw_path!r};`)"
            )
        try:
            imported_cf = parse_clay_file(target, _import_stack=new_stack)
        except (ValueError, FileNotFoundError) as exc:
            # Decorate downstream errors with our chain context.
            # Re-raise the same exception class so callers can still
            # catch FileNotFoundError specifically.
            if isinstance(exc, FileNotFoundError):
                raise FileNotFoundError(
                    f"while importing {target!s} from {abs_path!s}: {exc}"
                ) from exc
            raise ValueError(
                f"while importing {target!s} from {abs_path!s}: {exc}"
            ) from exc
        _merge_imported_cells(
            importing=cf,
            imported=imported_cf,
            namespace=alias,
            importing_path=abs_path,
            imported_path=target,
        )

    _attach_clay_file_to_cell_defs(cf)
    _resolve_verify_refs(cf)
    return cf


def _attach_source_file_to_spans(cf: ClayFile, abs_path: Path) -> None:
    file_name = str(abs_path)

    def attach_spec(spec) -> None:
        if spec is None:
            return
        for field in getattr(spec, "fields", []) or []:
            span = getattr(field, "source_span", None)
            if isinstance(span, dict) and "file" not in span:
                span["file"] = file_name

    for cd in cf.cell_defs:
        span = getattr(cd, "source_span", None)
        if isinstance(span, dict) and "file" not in span:
            span["file"] = file_name
        attach_spec(getattr(cd, "spec", None))
        for role in getattr(cd, "functional_roles", []) or []:
            span = getattr(role, "source_span", None)
            if isinstance(span, dict) and "file" not in span:
                span["file"] = file_name
        for decl in getattr(cd, "layout_intents", []) or []:
            span = getattr(decl, "source_span", None)
            if isinstance(span, dict) and "file" not in span:
                span["file"] = file_name
    for alias in cf.cell_aliases:
        attach_spec(getattr(alias, "spec", None))
    for job in cf.verify_jobs:
        span = getattr(job, "source_span", None)
        if isinstance(span, dict) and "file" not in span:
            span["file"] = file_name
        attach_spec(getattr(job, "spec", None))
        for setup in getattr(job, "setups", []) or []:
            attach_spec(getattr(setup, "spec", None))
    pending_refs = cf.__dict__.get("_pending_verify_refs", []) or []
    for item in pending_refs:
        if isinstance(item, dict):
            span = item.get("source_span")
            if isinstance(span, dict) and "file" not in span:
                span["file"] = file_name
