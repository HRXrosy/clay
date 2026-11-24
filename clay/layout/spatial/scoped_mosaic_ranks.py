"""Complete coordinate-free anchor ranks within explicit source MOS macros.

Raw artwork variants remain distinct. Macro ownership provides the comparison
scope; measured anchors supply exact ordinal ranks, with no pitch or physical
coordinates in the emitted facts. Aggregate envelope ordering is independent.
"""
from clay.layout.spatial.abstract_ir import HardInvariant, SpatialFactKind


def derive_mosaic_scope_ranks(occurrences, scope_by_logical):
    if type(scope_by_logical) is not tuple:
        raise TypeError('mosaic scope ownership must be an exact tuple')
    if not scope_by_logical:
        return ()
    if (any(type(row) is not tuple or len(row) != 2
            or any(type(value) is not str or not value or value != value.strip() for value in row)
            for row in scope_by_logical)
        or tuple(sorted(scope_by_logical)) != scope_by_logical
        or len({owner for owner, _ in scope_by_logical}) != len(scope_by_logical)
        or {owner for owner, _ in scope_by_logical} != {row.logical_owner for row in occurrences}):
        raise ValueError('mosaic scope ownership must be one canonical complete logical cover')
    from clay.layout.spatial.derive import _bindings, _evidence, _axis_ranks
    scope_map = dict(scope_by_logical)
    facts = []
    for scope in sorted(set(scope_map.values())):
        members = tuple(sorted((row for row in occurrences if scope_map[row.logical_owner] == scope),
                               key=lambda row: row.node_id))
        for axis, coordinate in (('horizontal', 0), ('vertical', 1)):
            ranks = _axis_ranks(tuple(row.physical.anchor_nm[coordinate] for row in members))
            groups = {}
            for row in members:
                groups.setdefault(ranks[row.physical.anchor_nm[coordinate]], []).append(row.node_id)
            sequence = ';'.join(','.join(sorted(groups[rank])) for rank in sorted(groups))
            facts.append(HardInvariant(fact_id=f'mosaic_scope_rank:{axis}:{scope}',
                kind=SpatialFactKind.SCOPE_RANK_SEQUENCE, bindings=_bindings(members),
                attributes=(('axis', axis), ('scope', scope), ('sequence', sequence)),
                evidence_ids=_evidence(members), derivation_rule='semantic-source-macro-anchor-ranks/v1'))
    return tuple(facts)
