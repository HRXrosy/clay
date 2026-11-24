"""Linear adapters from ASIR topology into Clay's existing placement contracts."""

from __future__ import annotations

from clay.layout.place.constraints import RelativeOrder as PlacementRelativeOrder
from clay.layout.seqpair_model import RelativeOrder as SeqpairRelativeOrder
from clay.layout.spatial.abstract_ir import (
    AbstractSpatialIR,
    SpatialIrSchemaError,
    _canonical_text,
    _exact_tuple,
)


def _validated_groups(
    value: object,
    *,
    field: str,
    known_nodes: frozenset[str],
) -> tuple[tuple[str, ...], ...]:
    rows = _exact_tuple(value, field=field)
    result: list[tuple[str, ...]] = []
    for row in rows:
        if type(row) is not tuple or not row:
            raise TypeError(f"{field} must contain non-empty exact tuples")
        if any(type(node) is not str or node not in known_nodes for node in row):
            raise SpatialIrSchemaError(f"{field} names an unknown ASIR node")
        if len(set(row)) != len(row):
            raise SpatialIrSchemaError(f"{field} rows cannot repeat a node")
        result.append(row)  # type: ignore[arg-type]
    return tuple(result)


def to_seqpair_relative_order(
    ir: AbstractSpatialIR,
    *,
    cell: str,
    rows: tuple[tuple[str, ...], ...] = (),
    mirror_pairs: tuple[tuple[str, str], ...] = (),
) -> SeqpairRelativeOrder:
    """Map only canonical H/V reduction edges to the existing sequence-pair type."""

    if type(ir) is not AbstractSpatialIR:
        raise TypeError("ir must be an exact AbstractSpatialIR")
    cell = _canonical_text(cell, field="cell")
    known = frozenset(ir.nodes)
    canonical_rows = _validated_groups(rows, field="rows", known_nodes=known)
    canonical_mirrors = _validated_groups(
        mirror_pairs,
        field="mirror_pairs",
        known_nodes=known,
    )
    if any(len(pair) != 2 for pair in canonical_mirrors):
        raise SpatialIrSchemaError("mirror_pairs must contain two-node pairs")
    return SeqpairRelativeOrder(
        cell=cell,
        devices=ir.nodes,
        left_of=ir.horizontal.edges,
        below=ir.vertical.edges,
        rows=canonical_rows,
        mirror_pairs=canonical_mirrors,  # type: ignore[arg-type]
    )


def to_placement_constraints(
    ir: AbstractSpatialIR,
) -> tuple[PlacementRelativeOrder, ...]:
    """Map reduced topology edges to the closed placement constraint vocabulary."""

    if type(ir) is not AbstractSpatialIR:
        raise TypeError("ir must be an exact AbstractSpatialIR")
    horizontal = tuple(
        PlacementRelativeOrder(before, after, "left_of")
        for before, after in ir.horizontal.edges
    )
    vertical = tuple(
        PlacementRelativeOrder(upper, lower, "above")
        for lower, upper in ir.vertical.edges
    )
    return horizontal + vertical


__all__ = ["to_placement_constraints", "to_seqpair_relative_order"]
