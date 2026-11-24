"""Hierarchical cell reference support for layout libraries.

Adds a thin hierarchy layer over the flat ``ParameterizedLibrary`` produced by
``clay/layout/backend.py``. A ``HierarchicalLibrary`` wraps a base library plus
a list of ``CellInstance`` records that declare "cell X includes cell Y at
offset (dx, dy)". ``flatten_hierarchical`` resolves the instance graph by
recursively translating child polygons into the parent's coordinate frame,
returning a new flat ``ParameterizedLibrary``.

Notes
-----
* The flattener is recursive but memoizes per-cell results so a cell that is
  instantiated many times is only flattened once per call to
  ``flatten_hierarchical``.
* Cycle detection is performed up-front (``detect_cycles``) and at flatten
  time (a recursion stack guard raises ``ValueError``). Callers that want a
  full enumeration of cycles should use ``detect_cycles`` directly.
* The original ``ParameterizedLibrary`` in ``HierarchicalLibrary.base`` is
  never mutated; flatten returns a fresh library with copied polygons.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Set, Tuple

from .backend import (
    LayerPolygon,
    NamedCell,
    ParameterizedLibrary,
    Point,
    Polygon,
)


@dataclass(frozen=True)
class CellInstance:
    """A placement of one cell inside another at a fixed (dx, dy) offset.

    Attributes
    ----------
    parent_cell:
        Name of the cell (in the same library) that holds this instance.
    child_cell:
        Name of the cell being instantiated. May refer to any other cell in
        the library, including (transitively) itself if the user wants a
        cycle (which will then be reported by ``detect_cycles``).
    offset_x_dbu, offset_y_dbu:
        Translation applied to every vertex of the child's polygons when
        flattening into the parent's coordinate frame, in database units.
    instance_label:
        Optional caller-provided tag. Two instances may share a label; the
        flattener does not use it for deduplication.
    """

    parent_cell: str
    child_cell: str
    offset_x_dbu: int
    offset_y_dbu: int
    instance_label: str = ""


@dataclass
class HierarchicalLibrary:
    """A ``ParameterizedLibrary`` plus a list of cell instances.

    The base library is left untouched; ``flatten_hierarchical`` produces a
    fresh flat library by walking the instance graph.
    """

    base: ParameterizedLibrary
    instances: List[CellInstance] = field(default_factory=list)


def _cells_by_name(lib: ParameterizedLibrary) -> Dict[str, NamedCell]:
    return {c.name: c for c in lib.cells}


def _instances_by_parent(
    instances: List[CellInstance],
) -> Dict[str, List[CellInstance]]:
    out: Dict[str, List[CellInstance]] = {}
    for inst in instances:
        out.setdefault(inst.parent_cell, []).append(inst)
    return out


def _translate_polygons(
    polys: List[LayerPolygon], dx: int, dy: int
) -> List[LayerPolygon]:
    return [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(
                vertices=[Point(v.x + dx, v.y + dy) for v in lp.polygon.vertices]
            ),
        )
        for lp in polys
    ]


def _copy_polygons(polys: List[LayerPolygon]) -> List[LayerPolygon]:
    return [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(vertices=[Point(v.x, v.y) for v in lp.polygon.vertices]),
        )
        for lp in polys
    ]


def detect_cycles(hlib: HierarchicalLibrary) -> List[List[str]]:
    """Detect circular cell references in the instance graph.

    Builds a directed graph ``parent -> child`` from ``hlib.instances`` and
    runs DFS with a recursion stack to find back-edges. Returns one cycle
    per back-edge encountered, each presented as a list of cell names that
    closes onto itself (``[A, B, A]`` for ``A -> B -> A``). A self-loop
    appears as ``[A, A]``.

    Cells that are referenced as a child but never declared as a parent are
    treated as leaves (no outgoing edges). Cells that appear only in
    ``hlib.base.cells`` and never in any instance simply have no node in the
    graph and trivially contribute no cycles.
    """
    adj: Dict[str, List[str]] = {}
    for inst in hlib.instances:
        adj.setdefault(inst.parent_cell, []).append(inst.child_cell)
        adj.setdefault(inst.child_cell, adj.get(inst.child_cell, []))

    cycles: List[List[str]] = []
    visited: Set[str] = set()
    on_stack: Set[str] = set()
    stack: List[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        on_stack.add(node)
        stack.append(node)
        for nxt in adj.get(node, []):
            if nxt in on_stack:
                # Back-edge: extract cycle from stack.
                try:
                    start = stack.index(nxt)
                    cycles.append(stack[start:] + [nxt])
                except ValueError:
                    cycles.append([nxt, nxt])
            elif nxt not in visited:
                dfs(nxt)
        on_stack.discard(node)
        stack.pop()

    for node in list(adj.keys()):
        if node not in visited:
            dfs(node)

    return cycles


def add_instance(
    hlib: HierarchicalLibrary,
    parent: str,
    child: str,
    offset_x: int,
    offset_y: int,
    label: str = "",
) -> HierarchicalLibrary:
    """Return a new ``HierarchicalLibrary`` with one extra ``CellInstance``.

    The input ``hlib`` is not mutated: its ``instances`` list is copied and
    extended, and the resulting library shares the same ``base`` reference
    (the base library is read-only from this module's perspective).
    """
    new_instances = list(hlib.instances) + [
        CellInstance(
            parent_cell=parent,
            child_cell=child,
            offset_x_dbu=offset_x,
            offset_y_dbu=offset_y,
            instance_label=label,
        )
    ]
    return HierarchicalLibrary(base=hlib.base, instances=new_instances)


def flatten_hierarchical(hlib: HierarchicalLibrary) -> ParameterizedLibrary:
    """Resolve every ``CellInstance`` by recursive coordinate translation.

    Returns a new ``ParameterizedLibrary`` with the same name and units as
    ``hlib.base``. Every cell that exists in ``hlib.base.cells`` appears in
    the output exactly once, with polygons equal to its own polygons plus
    the (translated) polygons of every child it transitively includes.

    Raises ``ValueError`` if any instance refers to a parent or child cell
    that is not present in ``hlib.base.cells``, or if a cycle is reachable
    from any cell being flattened.
    """
    by_name = _cells_by_name(hlib.base)
    children_of = _instances_by_parent(hlib.instances)

    # Validate all referenced cell names up front (so callers see a clear
    # error even if the offending cell is never reached transitively).
    for inst in hlib.instances:
        if inst.parent_cell not in by_name:
            raise ValueError(
                f"CellInstance references unknown parent cell "
                f"'{inst.parent_cell}'"
            )
        if inst.child_cell not in by_name:
            raise ValueError(
                f"CellInstance references unknown child cell "
                f"'{inst.child_cell}' (in parent '{inst.parent_cell}')"
            )

    memo: Dict[str, List[LayerPolygon]] = {}
    on_stack: Set[str] = set()

    def flatten_cell(name: str) -> List[LayerPolygon]:
        if name in memo:
            return memo[name]
        if name in on_stack:
            raise ValueError(
                f"cycle detected while flattening cell '{name}'; "
                f"use detect_cycles() for the full list"
            )
        on_stack.add(name)
        cell = by_name[name]
        result: List[LayerPolygon] = _copy_polygons(cell.polygons)
        for inst in children_of.get(name, []):
            child_polys = flatten_cell(inst.child_cell)
            result.extend(
                _translate_polygons(
                    child_polys, inst.offset_x_dbu, inst.offset_y_dbu
                )
            )
        on_stack.discard(name)
        memo[name] = result
        return result

    new_cells: List[NamedCell] = []
    for cell in hlib.base.cells:
        flat = flatten_cell(cell.name)
        new_cells.append(NamedCell(name=cell.name, polygons=flat))

    return ParameterizedLibrary(
        name=hlib.base.name,
        unit_user=hlib.base.unit_user,
        unit_db_meters=hlib.base.unit_db_meters,
        cells=new_cells,
    )


__all__ = [
    "CellInstance",
    "HierarchicalLibrary",
    "add_instance",
    "detect_cycles",
    "flatten_hierarchical",
]


# ``replace`` is intentionally imported (dataclasses.replace) to make it
# trivially available for callers who want to tweak a frozen CellInstance.
_ = replace  # silence unused-import warnings; exposed for downstream use.

# Module-level sentinel for trivial Tuple typing in future extensions.
_OffsetXY = Tuple[int, int]
