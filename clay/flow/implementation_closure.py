"""Capture static Python imports inside declared repository package namespaces.

This never executes imported code. Explicit seeds must separately name dynamic
Python dependencies and non-Python code inputs; runtime binaries and installed
PDK/data authority remain separate typed bindings. Inactive import branches are
included conservatively. The returned rows bind the exact bytes parsed.
"""
from __future__ import annotations

from collections import deque
import hashlib
import os
from pathlib import Path, PurePosixPath

from clay.flow._static_import_syntax import analyze_python


class StaticImplementationClosureError(ValueError):
    """Static resolution cannot produce one fresh confined source identity."""


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _canonical_path(value):
    if (type(value) is not str or not value or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or PurePosixPath(value).as_posix() != value):
        raise StaticImplementationClosureError("PATH: expected a canonical repository-relative locator")
    return value


class _Capture:
    def __init__(self, root, seeds, local_package_roots):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise StaticImplementationClosureError("PATH: repository root is not a directory")
        self.seeds = tuple(sorted({_canonical_path(seed) for seed in seeds}))
        if not self.seeds:
            raise StaticImplementationClosureError("SEED: at least one explicit input is required")
        self.local_roots = set(local_package_roots)
        self.local_roots.update(PurePosixPath(seed).parts[0] for seed in self.seeds
                                if seed.endswith(".py") and len(PurePosixPath(seed).parts) > 1)
        if any(type(name) is not str or not name.isidentifier() for name in self.local_roots):
            raise StaticImplementationClosureError("PATH: local package roots must be identifiers")
        self.watched = {}
        self.payloads = {}
        self.syntax = {}
        self.pending = deque()
        self.queued = set()

    def probe(self, relative):
        path = (self.root / relative).resolve()
        try:
            canonical = path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise StaticImplementationClosureError(f"PATH: input escapes repository: {relative}") from exc
        if os.path.normcase(canonical) != os.path.normcase(relative):
            raise StaticImplementationClosureError(f"PATH: input aliases a different locator: {relative}")
        if path.is_file():
            return "file"
        if path.is_dir():
            return "directory"
        if path.exists():
            raise StaticImplementationClosureError(f"PATH: unsupported filesystem input: {relative}")
        return "absent"

    def watch(self, relative):
        # One capture owns one resolution view. Every observed path (including
        # absence and directories) is independently reprobed at capture ending.
        # No filesystem result or resolved-module identity survives this call.
        if relative not in self.watched:
            self.watched[relative] = self.probe(relative)
        return self.watched[relative]

    def enqueue(self, relative, package):
        if self.watch(relative) != "file":
            raise StaticImplementationClosureError(f"IMPORT: required source is missing: {relative}")
        context = relative, package
        if context not in self.queued:
            self.queued.add(context)
            self.pending.append(context)

    def resolve_module(self, name, *, required=True):
        parts = name.split(".")
        if not parts or any(not part.isidentifier() for part in parts):
            raise StaticImplementationClosureError(f"IMPORT: invalid static module name: {name}")
        if parts[0] not in self.local_roots:
            return None
        kind = None
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            module, initializer = prefix + ".py", prefix + "/__init__.py"
            is_module = self.watch(module) == "file"
            is_package = self.watch(initializer) == "file"
            is_directory = self.watch(prefix) == "directory"
            if is_module and is_package:
                raise StaticImplementationClosureError(f"AMBIGUOUS: module and package both exist: {prefix}")
            if is_package:
                kind = "package"
                self.enqueue(initializer, ".".join(parts[:index]))
            elif is_module:
                kind = "module"
                self.enqueue(module, ".".join(parts[:index - 1]))
            elif is_directory:
                kind = "namespace"
            else:
                if not required and index == len(parts):
                    return None
                raise StaticImplementationClosureError(f"IMPORT: local module is missing: {name}")
            if index < len(parts) and kind == "module":
                raise StaticImplementationClosureError(f"IMPORT: module is not a package: {prefix}")
        return kind

    def seed(self, relative):
        if self.watch(relative) != "file":
            raise StaticImplementationClosureError(f"SEED: explicit source is missing: {relative}")
        if not relative.endswith(".py"):
            self.enqueue(relative, None)
            return
        parts = PurePosixPath(relative).parts
        package = ".".join(parts[:-1])
        if package and self.resolve_module(package) not in {"package", "namespace"}:
            raise StaticImplementationClosureError(f"IMPORT: seed parent is not a package: {relative}")
        self.enqueue(relative, package)

    def imports(self, syntax, package):
        for node in syntax.imports:
            if not node.is_from:
                for name in node.names:
                    self.resolve_module(name)
            else:
                if node.level:
                    parts = package.split(".") if package else []
                    if node.level > len(parts):
                        raise StaticImplementationClosureError("IMPORT: relative import escapes package root")
                    base = ".".join(parts[:len(parts) - node.level + 1])
                    if node.module:
                        base += "." + node.module
                else:
                    base = node.module
                kind = self.resolve_module(base)
                if kind not in {"package", "namespace"}:
                    continue
                for name in node.names:
                    if name == "*":
                        if kind == "package":
                            for exported in self.star_exports(base):
                                self.resolve_module(base + "." + exported, required=False)
                    else:
                        # A package attribute need not be a module. Bind any
                        # matching child source conservatively, and watch absence.
                        self.resolve_module(base + "." + name,
                                            required=kind == "namespace")

    def source_syntax(self, relative):
        if relative not in self.payloads:
            try:
                self.payloads[relative] = _read_bytes(self.root / relative)
            except OSError as exc:
                raise StaticImplementationClosureError(
                    f"DRIFT: required source is unavailable: {relative}") from exc
        if relative not in self.syntax:
            try:
                self.syntax[relative] = analyze_python(self.payloads[relative], relative)
            except (SyntaxError, UnicodeError, ValueError) as exc:
                raise StaticImplementationClosureError(f"SYNTAX: cannot parse Python source: {relative}") from exc
        return self.syntax[relative]

    def star_exports(self, package):
        """Bind literal, untouched package exports; unresolved exports reject.

        Reassignments are conservatively unioned, including inactive branches.
        Runtime export discovery/mutation remains outside this static service.
        """
        syntax = self.source_syntax(package.replace(".", "/") + "/__init__.py")
        if syntax.export_error is not None:
            raise StaticImplementationClosureError(f"EXPORT: {syntax.export_error}: {package}")
        return syntax.exports

    def capture(self):
        for relative in self.seeds:
            self.seed(relative)
        while self.pending:
            relative, package = self.pending.popleft()
            if package is None:
                try:
                    self.payloads[relative] = _read_bytes(self.root / relative)
                except OSError as exc:
                    raise StaticImplementationClosureError(f"DRIFT: required source is unavailable: {relative}") from exc
                continue
            self.imports(self.source_syntax(relative), package)
        for relative, payload in sorted(self.payloads.items()):
            try:
                current = _read_bytes(self.root / relative)
            except OSError as exc:
                raise StaticImplementationClosureError(f"DRIFT: captured source is unavailable: {relative}") from exc
            if current != payload:
                raise StaticImplementationClosureError(f"DRIFT: captured source bytes changed: {relative}")
        for relative, expected in sorted(self.watched.items()):
            if self.probe(relative) != expected:
                raise StaticImplementationClosureError(f"DRIFT: import resolution changed: {relative}")
        return tuple((relative, hashlib.sha256(payload).hexdigest())
                     for relative, payload in sorted(self.payloads.items()))


def capture_static_local_implementation_state(repo_root, seed_paths, *, local_package_roots=("clay",)):
    """Bind static imports within Clay and explicitly seeded producer namespaces.

    Only existing Python modules and package initializers are traversed. Explicit
    non-Python seeds are hashed leaves. Dynamic imports, runtime binaries, and
    installed PDK/data dependencies must be supplied by separate host bindings.
    """
    if isinstance(seed_paths, (str, bytes)) or isinstance(local_package_roots, (str, bytes)):
        raise StaticImplementationClosureError("SEED: expected sequences of paths and package roots")
    return _Capture(repo_root, seed_paths, local_package_roots).capture()
