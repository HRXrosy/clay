"""ParameterizedLibrary diff utility.

Compares two :class:`ParameterizedLibrary` instances and reports cell-level
and polygon-level differences.

Cells are matched by name (set difference over the union of cell names).
For cells present in both libraries, polygons are compared as multisets
of :class:`LayerPolygon` values (which are dataclasses with structural
equality, so two polygons with the same layer/datatype/vertices compare
equal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .backend import LayerPolygon, NamedCell, ParameterizedLibrary


@dataclass
class CellDiff:
    """Per-cell polygon-level diff between two libraries."""

    cell_name: str
    only_in_a: List[LayerPolygon] = field(default_factory=list)
    only_in_b: List[LayerPolygon] = field(default_factory=list)


@dataclass
class LibraryDiff:
    """Library-level diff: which cells differ, which are unique to each side."""

    cells_only_in_a: List[str] = field(default_factory=list)
    cells_only_in_b: List[str] = field(default_factory=list)
    cells_differ: List[CellDiff] = field(default_factory=list)
    cells_identical: List[str] = field(default_factory=list)

    @property
    def libraries_identical(self) -> bool:
        return not (self.cells_only_in_a or self.cells_only_in_b or self.cells_differ)


def _index_cells(lib: ParameterizedLibrary) -> Dict[str, NamedCell]:
    """Index cells by name. If duplicate names exist, the last wins (matches
    Python dict-overwrite semantics; libraries normally have unique names)."""
    return {cell.name: cell for cell in lib.cells}


def _polygon_multiset_diff(
    polys_a: List[LayerPolygon], polys_b: List[LayerPolygon]
) -> tuple[List[LayerPolygon], List[LayerPolygon]]:
    """Return (only_in_a, only_in_b) multiset diff over LayerPolygon equality.

    LayerPolygon is a dataclass with eq=True, so structural equality compares
    layer/datatype/polygon (which in turn compares the vertices list). We can't
    hash these (Polygon has a mutable List field), so we use an O(n*m) list
    walk with a "consumed" mask on the b side.
    """
    consumed_b = [False] * len(polys_b)
    only_in_a: List[LayerPolygon] = []
    for pa in polys_a:
        match_idx = -1
        for j, pb in enumerate(polys_b):
            if consumed_b[j]:
                continue
            if pa == pb:
                match_idx = j
                break
        if match_idx >= 0:
            consumed_b[match_idx] = True
        else:
            only_in_a.append(pa)
    only_in_b = [polys_b[j] for j in range(len(polys_b)) if not consumed_b[j]]
    return only_in_a, only_in_b


def diff_libraries(
    a: ParameterizedLibrary, b: ParameterizedLibrary
) -> LibraryDiff:
    """Compare two libraries cell-by-cell.

    Cells are matched by name. For cells present in both, polygons are diffed
    as multisets (using LayerPolygon structural equality).
    """
    idx_a = _index_cells(a)
    idx_b = _index_cells(b)

    names_a = set(idx_a.keys())
    names_b = set(idx_b.keys())

    diff = LibraryDiff()
    diff.cells_only_in_a = sorted(names_a - names_b)
    diff.cells_only_in_b = sorted(names_b - names_a)

    for name in sorted(names_a & names_b):
        only_a, only_b = _polygon_multiset_diff(
            idx_a[name].polygons, idx_b[name].polygons
        )
        if only_a or only_b:
            diff.cells_differ.append(
                CellDiff(cell_name=name, only_in_a=only_a, only_in_b=only_b)
            )
        else:
            diff.cells_identical.append(name)

    return diff


def summarize_diff(diff: LibraryDiff) -> str:
    """One-line human-readable summary of a LibraryDiff."""
    if diff.libraries_identical:
        return "Libraries identical"
    parts: List[str] = []
    if diff.cells_differ:
        n = len(diff.cells_differ)
        parts.append(f"{n} cell{'s' if n != 1 else ''} differ")
    if diff.cells_only_in_a:
        n = len(diff.cells_only_in_a)
        parts.append(f"{n} only in A")
    if diff.cells_only_in_b:
        n = len(diff.cells_only_in_b)
        parts.append(f"{n} only in B")
    return ", ".join(parts)
