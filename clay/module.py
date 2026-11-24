"""
Clay module system v0.0.8 (Python-embedded form).

Each Clay module is — for now — a Python module that declares its package
name via `package(...)` and registers exported cells via @register_cell.
A future `.clay` text parser will translate `package` / `import` syntax
into equivalent runtime calls; the underlying registry is the same.

Public API:

    from clay.module import package, register_cell, resolve, ModuleNotFound

    # Module author side
    package("my_lib.opamps", version="0.1.0")

    @register_cell("MyOpamp")
    def MyOpamp(spec, **kwargs):
        return build_cell(...)        # returns a clay.ir.Cell

    # User side
    mod = resolve("my_lib.opamps")
    cell = mod.cell("MyOpamp")(spec=my_spec)
    # or
    cell = mod.MyOpamp(spec=my_spec)
"""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .ir import Cell


# ─── Errors ───────────────────────────────────────────────────

class ModuleNotFound(LookupError):
    """Structured error envelope when a package can't be resolved."""

    def __init__(self, package_name: str, searched: list[str], suggestion: str = ""):
        self.package_name = package_name
        self.searched = list(searched)
        self.suggestion = suggestion
        msg = (
            f"clay module not found: {package_name!r}\n"
            f"  searched: {self.searched}\n"
            f"  suggestion: {suggestion or '(none)'}"
        )
        super().__init__(msg)

    def as_envelope(self) -> dict:
        return {
            "error": "ModuleNotFound",
            "package": self.package_name,
            "searched": self.searched,
            "suggestion": self.suggestion,
        }


class SymbolNotFound(LookupError):
    def __init__(self, package_name: str, symbol: str, available: list[str]):
        self.package_name = package_name
        self.symbol = symbol
        self.available = available
        msg = (
            f"clay symbol {symbol!r} not in package {package_name!r}\n"
            f"  available: {available}"
        )
        super().__init__(msg)


# ─── Module + Package IR ──────────────────────────────────────

CellBuilder = Callable[..., Cell]
TestbenchBuilder = Callable[[Cell], "object"]   # Cell -> Testbench


@dataclass
class CellEntry:
    """One registered cell: builder + default testbench builder."""
    builder: CellBuilder
    default_testbench: Optional[TestbenchBuilder] = None


@dataclass
class Module:
    """An importable Clay module with named cell builders."""
    name: str
    version: str = "0.0.0"
    cells: Dict[str, CellEntry] = field(default_factory=dict)
    source_path: Optional[Path] = None

    def cell(self, name: str) -> CellBuilder:
        """Return the cell-builder function (for direct construction)."""
        return self.cell_entry(name).builder

    def cell_entry(self, name: str) -> CellEntry:
        """Return the full CellEntry (builder + testbench)."""
        if name not in self.cells:
            raise SymbolNotFound(self.name, name, sorted(self.cells))
        return self.cells[name]

    def testbench_for(self, name: str) -> Optional[TestbenchBuilder]:
        """Return the registered default testbench builder (or None)."""
        return self.cell_entry(name).default_testbench

    def __getattr__(self, item: str) -> CellBuilder:
        # Allow attribute-style access: mod.MyOpamp(...) returns the builder
        cells = self.__dict__.get("cells", {})
        if item in cells:
            return cells[item].builder
        raise AttributeError(f"module {self.name!r} has no cell {item!r}")

    @property
    def exported(self) -> list[str]:
        return sorted(self.cells)


# ─── Registry ────────────────────────────────────────────────

class _Registry:
    """Process-wide module registry. Singleton via module-level _REG."""

    def __init__(self) -> None:
        self._modules: Dict[str, Module] = {}
        # During @register_cell decoration, we need to know which Module
        # the caller is in. The most recent package() call wins.
        self._current_package: Optional[str] = None

    def declare_package(self, name: str, *, version: str, source_path: Optional[Path] = None) -> Module:
        if name in self._modules:
            mod = self._modules[name]
            if version and mod.version != version:
                # Allow re-declaration only if version matches (idempotent imports)
                raise RuntimeError(
                    f"package {name!r} already registered at version {mod.version!r}; "
                    f"refusing to overwrite with {version!r}"
                )
        else:
            mod = Module(name=name, version=version, source_path=source_path)
            self._modules[name] = mod
        self._current_package = name
        return mod

    def register_cell(
        self,
        cell_name: str,
        builder: CellBuilder,
        default_testbench: Optional[TestbenchBuilder] = None,
    ) -> None:
        if self._current_package is None:
            raise RuntimeError(
                "register_cell called without a preceding package(...) declaration. "
                "Each Clay module must declare its package name first."
            )
        mod = self._modules[self._current_package]
        if cell_name in mod.cells:
            raise RuntimeError(
                f"duplicate cell {cell_name!r} in package {mod.name!r}"
            )
        mod.cells[cell_name] = CellEntry(
            builder=builder,
            default_testbench=default_testbench,
        )

    def resolve(self, package_name: str) -> Module:
        # 1. Already in our Clay registry?
        if package_name in self._modules:
            return self._modules[package_name]
        # 2. Auto-import via Python — ONLY for allow-listed prefixes (RCE
        #    guard). User code that needs to load a non-allow-listed
        #    package must do `import that_package` themselves first; the
        #    package's top-level `package(...)` call then registers it.
        searched = []
        candidates = [package_name]
        if _is_allowed_for_auto_import(package_name):
            for cand in candidates:
                searched.append(cand)
                try:
                    if cand in sys.modules:
                        # Python cache hit; the module may have run package()
                        # before our registry was reset. Force reload so
                        # decorators re-execute.
                        importlib.reload(sys.modules[cand])
                    else:
                        importlib.import_module(cand)
                except ImportError:
                    continue
                if package_name in self._modules:
                    return self._modules[package_name]
        else:
            searched.append(f"(blocked by allow-list: {package_name})")
        # 3. Failure.
        raise ModuleNotFound(
            package_name,
            searched=searched,
            suggestion=(
                f"Auto-import is restricted to prefixes "
                f"{list(_ALLOWED_IMPORT_PREFIXES)} for security. To use "
                f"{package_name!r}, either pre-register it in trusted code "
                f"(import the Python module yourself first, then resolve), "
                f"or widen the allow-list with "
                f"clay.module.set_allowed_import_prefixes(...)."
            ),
        )

    def list_modules(self) -> list[str]:
        return sorted(self._modules)


_REG = _Registry()


# ─── Security: import allow-list ──────────────────────────────
# `resolve()` calls importlib on a user-supplied string. If a .clay file
# is generated by an LLM (or other untrusted source), an arbitrary string
# here can execute any module on PYTHONPATH (RCE risk). We restrict
# auto-import to a strict allow-list of package prefixes. Anything else
# must be pre-registered explicitly via package(...) in trusted code
# loaded by the user themselves.
_ALLOWED_IMPORT_PREFIXES: tuple[str, ...] = (
    "clay.stdlib.",      # Built-in standard library
    "clay.pdk.",         # PDK adapter packages (future)
)


def set_allowed_import_prefixes(prefixes: tuple[str, ...]) -> None:
    """Override the package-import allow-list.

    Use with caution: widening the allow-list re-introduces RCE risk for
    untrusted .clay sources. Intended for advanced users / CI who load
    third-party Clay packages from a known trusted set.
    """
    global _ALLOWED_IMPORT_PREFIXES
    _ALLOWED_IMPORT_PREFIXES = tuple(prefixes)


def _is_allowed_for_auto_import(package_name: str) -> bool:
    return any(package_name.startswith(p) for p in _ALLOWED_IMPORT_PREFIXES)


# ─── Public functional API ────────────────────────────────────

def package(name: str, *, version: str = "0.0.0") -> Module:
    """Declare the current Clay module's package name and version.

    Must be called before any @register_cell in the same file.
    """
    # Determine source path of the caller for diagnostics.
    frame = inspect.stack()[1]
    src = Path(frame.filename).resolve() if frame.filename else None
    return _REG.declare_package(name, version=version, source_path=src)


def register_cell(
    cell_name: str,
    *,
    testbench: Optional[TestbenchBuilder] = None,
) -> Callable[[CellBuilder], CellBuilder]:
    """Decorator: register a cell-builder function in the current package.

    Args:
        cell_name: name under which the cell is accessible.
        testbench: optional default testbench builder. When the CLI or
            other consumers verify this cell, they look up the testbench
            here rather than hardcoding `build_ac_testbench`. Cells with
            no registered testbench fall back to the AC testbench for
            backward compat.
    """
    def deco(fn: CellBuilder) -> CellBuilder:
        _REG.register_cell(cell_name, fn, default_testbench=testbench)
        return fn
    return deco


def resolve(package_name: str) -> Module:
    """Resolve a package name to its Module, importing the Python module
    if necessary."""
    return _REG.resolve(package_name)


def list_modules() -> list[str]:
    """All packages currently registered in this process."""
    return _REG.list_modules()


def reset_registry() -> None:
    """Test helper: clear all registered modules. Production code should
    not call this — use it only in unit tests."""
    global _REG
    _REG = _Registry()
