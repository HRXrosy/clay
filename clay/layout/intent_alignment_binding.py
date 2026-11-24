"""Reconcile parsed placement declarations with their lowered source intent.

This is a source-level coverage check.  It does not measure placement or
change the physical verification status of either kind of alignment.
"""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceAlignmentReconciliation:
    edge_count: int
    center_count: int
    # (center declaration ordinal, group name, source occurrences), for
    # repeated names. Group-center semantics are a union, so repetition does
    # not change the relation but remains visible in this source record.
    repeated_center_groups: tuple
    place_group_count: int = 0
    order_source_count: int = 0
    order_count: int = 0
    order_ignored_count: int = 0
    abut_source_count: int = 0
    abut_count: int = 0
    abut_ignored_count: int = 0


def reconcile_source_alignments(declarations, intent):
    """Check parsed-to-typed alignment and placement declarations.

    Offset tokens in parsed attributes are already expressed in nanometres;
    integer conversion matches the existing lowering without unit conversion.
    Endpoint reversal negates the equation's offset.  Each edge is part of the
    identity, and center groups use the existing union/set semantics.
    """
    source_edges = []
    source_centers = []
    source_groups = []
    source_orders = []
    source_abuts = []
    order_source_count = abut_source_count = 0
    order_ignored_count = abut_ignored_count = 0
    repeated = []
    for decl in declarations:
        attrs = decl.attributes or {}
        if decl.kind == "align":
            first = str(attrs.get("first") or decl.nets[0])
            second = str(attrs.get("second") or decl.nets[1])
            edge = str(attrs.get("edge") or "")
            offset = int(attrs.get("offset_nm") or 0)
            if second < first:
                first, second, offset = second, first, -offset
            source_edges.append((first, second, edge, offset))
        elif decl.kind == "center_group":
            groups = tuple(attrs["groups"])
            counts = Counter(groups)
            ordinal = len(source_centers)
            repeated.extend((ordinal, group, count) for group, count in sorted(counts.items()) if count > 1)
            source_centers.append((str(attrs["target"]), tuple(sorted(counts))))
        elif decl.kind == "place_group":
            col = attrs.get("col")
            if col is not None:
                try:
                    col = int(str(col))
                except (TypeError, ValueError):
                    col = None
            pack_rows = attrs.get("pack_rows")
            if pack_rows is not None:
                pack_rows = int(str(pack_rows))
            source_groups.append((str(decl.group) if decl.group is not None else "",
                                  tuple(decl.devices), decl.row_min, decl.row_max,
                                  decl.priority, col, pack_rows, attrs.get("packing")))
        elif decl.kind == "order":
            order_source_count += 1
            a = str(attrs.get("a") or (decl.nets[0] if decl.nets else ""))
            b = str(attrs.get("b") or (decl.nets[1] if len(decl.nets) > 1 else ""))
            rel = str(attrs.get("rel") or "")
            if a and b and a != b and rel in {"left_of", "above"}:
                item = (a, b, rel)
                if item not in source_orders:
                    source_orders.append(item)
            else:
                order_ignored_count += 1
        elif decl.kind == "abut":
            abut_source_count += 1
            pair = attrs.get("pair")
            if not pair and decl.nets:
                pair = (decl.nets[0], decl.nets[1]) if len(decl.nets) >= 2 else None
            if pair and len(tuple(pair)) >= 2:
                a, b = str(pair[0]), str(pair[1])
                if a and b and a != b:
                    item = (a, b)
                    if item not in source_abuts:
                        source_abuts.append(item)
                    continue
            abut_ignored_count += 1

    typed_edges = [(row.first, row.second, row.edge, row.offset_nm)
                   for row in intent.edge_alignments]
    typed_centers = [(row.target, tuple(row.groups))
                     for row in intent.group_center_alignments]
    if Counter(source_edges) != Counter(typed_edges):
        raise ValueError("parsed source and typed edge alignment relations differ")
    if Counter(source_centers) != Counter(typed_centers):
        raise ValueError("parsed source and typed group center relations differ")
    typed_groups = [(row.group, tuple(row.devices), row.row_min, row.row_max,
                     row.priority, row.col, row.pack_rows, row.packing)
                    for row in intent.place_groups]
    if source_groups != typed_groups:
        raise ValueError("parsed source and typed place group declarations differ")
    if tuple(source_orders) != intent.order_constraints:
        raise ValueError("parsed source and typed order relations differ")
    if tuple(source_abuts) != intent.abut_pairs:
        raise ValueError("parsed source and typed abut relations differ")
    return SourceAlignmentReconciliation(
        len(source_edges), len(source_centers), tuple(repeated),
        len(source_groups), order_source_count, len(source_orders), order_ignored_count,
        abut_source_count, len(source_abuts), abut_ignored_count)
