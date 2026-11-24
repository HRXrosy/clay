"""Bounded reuse of immutable syntax derived from exact Python source bytes.

This component knows no repository, path existence, file digest or invocation.
Its values cannot establish current implementation identity. The live resolver
must reopen every file and check its entire resolution view on every capture.
"""
import ast
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class StaticImport:
    is_from: bool
    module: str | None
    level: int
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StaticSyntax:
    imports: tuple[StaticImport, ...]
    exports: tuple[str, ...]
    export_error: str | None


@lru_cache(maxsize=1024)
def analyze_python(payload: bytes, filename: str) -> StaticSyntax:
    """Reuse only value analysis; never retain an AST or mutable source view."""
    tree = ast.parse(payload, filename=filename)
    imports, targets, accesses, exports = [], set(), set(), set()
    error = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.append(StaticImport(False, None, 0, tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.ImportFrom):
            imports.append(StaticImport(True, node.module, node.level,
                                        tuple(alias.name for alias in node.names)))
        if isinstance(node, ast.Name) and node.id == '__all__':
            accesses.add(id(node))
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(target, ast.Name) and target.id == '__all__':
            targets.add(id(target))
            try:
                values = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                error = error or 'unresolved package exports'
            else:
                if (type(values) not in {list, tuple}
                        or any(type(value) is not str or not value.isidentifier() for value in values)):
                    error = error or 'unsupported package exports'
                else:
                    exports.update(values)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name) == '__all__' for alias in node.names):
            error = error or 'imported package exports'
    if accesses - targets:
        error = error or 'unresolved package export access or mutation'
    return StaticSyntax(tuple(imports), tuple(sorted(exports)), error)
