"""ParameterizedLibrary merger.

Combines two :class:`ParameterizedLibrary` instances into a single library.
Optionally applies per-side cell name prefixes to disambiguate cells, and
supports several collision policies for cells whose (prefixed) names still
clash:

- ``"raise"``    — raise :class:`ValueError` listing the colliding names.
- ``"prefer_a"`` — keep the cell from ``a``, drop the one from ``b``.
- ``"prefer_b"`` — keep the cell from ``b``, drop the one from ``a``.
- ``"rename"``   — rename the colliding cell from ``b`` by appending ``__b``.

Cells are matched purely by name; polygon content is not inspected by the
collision detector. The merger only re-points cell ``name`` fields and does
not mutate the input libraries.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List

from .backend import LayerPolygon, NamedCell, ParameterizedLibrary


_VALID_POLICIES = ("raise", "prefer_a", "prefer_b", "rename")


@dataclass
class MergeConflict:
    """Description of a single cell-name collision encountered during merge.

    ``side`` is the side whose cell ended up being kept under the chosen
    policy: ``"left"`` (i.e. ``a``) for ``prefer_a``, ``"right"`` for
    ``prefer_b`` or ``rename`` (the right cell is kept under a new name).
    """

    cell_name: str
    side: str  # "left" or "right"


def _prefixed_cells(lib: ParameterizedLibrary, prefix: str) -> List[NamedCell]:
    """Deep-copy ``lib``'s cells and prepend ``prefix`` to each cell name."""
    out: List[NamedCell] = []
    for cell in lib.cells:
        new_polys = [
            LayerPolygon(
                layer=lp.layer,
                datatype=lp.datatype,
                polygon=deepcopy(lp.polygon),
            )
            for lp in cell.polygons
        ]
        out.append(NamedCell(name=prefix + cell.name, polygons=new_polys))
    return out


def find_collisions(
    a: ParameterizedLibrary,
    b: ParameterizedLibrary,
    prefix_a: str = "",
    prefix_b: str = "",
) -> List[str]:
    """Return the list of cell names that would collide if the two libraries
    were merged with the given prefixes applied.

    Names are returned in the order they first appear in ``a``.
    """
    a_names = [prefix_a + c.name for c in a.cells]
    b_names_set = {prefix_b + c.name for c in b.cells}
    seen: set = set()
    collisions: List[str] = []
    for name in a_names:
        if name in b_names_set and name not in seen:
            collisions.append(name)
            seen.add(name)
    return collisions


def merge_libraries(
    a: ParameterizedLibrary,
    b: ParameterizedLibrary,
    *,
    merged_name: str = "merged",
    prefix_a: str = "",
    prefix_b: str = "",
    on_collision: str = "raise",
) -> ParameterizedLibrary:
    """Merge ``a`` and ``b`` into a new :class:`ParameterizedLibrary`.

    ``merged_name`` is the resulting library name. Unit fields are taken from
    ``a`` (callers that need a specific unit convention should rename or
    re-emit the merged library).

    Cell name collisions (after applying prefixes) are resolved per
    ``on_collision``: ``"raise"``, ``"prefer_a"``, ``"prefer_b"``, or
    ``"rename"`` (appends ``__b`` to the ``b`` cell's name).
    """
    if on_collision not in _VALID_POLICIES:
        raise ValueError(
            f"unknown on_collision policy: {on_collision!r}; "
            f"expected one of {_VALID_POLICIES}"
        )

    a_cells = _prefixed_cells(a, prefix_a)
    b_cells = _prefixed_cells(b, prefix_b)

    a_names = {c.name for c in a_cells}
    b_names = {c.name for c in b_cells}
    collision_names = a_names & b_names

    if collision_names and on_collision == "raise":
        ordered = [n for n in (c.name for c in a_cells) if n in collision_names]
        raise ValueError(
            "cell name collisions during merge: " + ", ".join(ordered)
        )

    out = ParameterizedLibrary(
        name=merged_name,
        unit_user=a.unit_user,
        unit_db_meters=a.unit_db_meters,
    )

    if on_collision == "prefer_a":
        # Emit every a-cell; emit b-cells whose name is not in collision set.
        for c in a_cells:
            out.add_cell(c)
        for c in b_cells:
            if c.name not in collision_names:
                out.add_cell(c)
        return out

    if on_collision == "prefer_b":
        # Emit a-cells whose name is not in collision set; then every b-cell.
        for c in a_cells:
            if c.name not in collision_names:
                out.add_cell(c)
        for c in b_cells:
            out.add_cell(c)
        return out

    if on_collision == "rename":
        # Emit every a-cell; for colliding b-cells, rename to <name>__b
        # (re-resolve if the renamed name itself collides with an existing
        # name by appending more __b suffixes).
        for c in a_cells:
            out.add_cell(c)
        existing = set(a_names)
        for c in b_cells:
            if c.name in collision_names:
                new_name = c.name + "__b"
                while new_name in existing:
                    new_name += "__b"
                existing.add(new_name)
                out.add_cell(NamedCell(name=new_name, polygons=c.polygons))
            else:
                existing.add(c.name)
                out.add_cell(c)
        return out

    # No collisions and policy is "raise": straight concat.
    for c in a_cells:
        out.add_cell(c)
    for c in b_cells:
        out.add_cell(c)
    return out
