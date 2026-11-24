"""Project an exact occupied lattice from existing source ASIR hard facts.

This explicit entry preserves complete column ranks across sparse owner rows.
It accepts one owner, or several owners covered by the same artwork-class or
explicit source-macro rank facts. Exact alignments support every shared rank; bounded class-row
stagger alone cannot authorize a constructive row. No pitch, coordinate, PDK,
source admission, or native primitive suitability is inferred here.
"""
from clay.layout.spatial.abstract_ir import AbstractSpatialIR, HardInvariant, SpatialFactKind
from clay.layout.spatial.intent_projection import (
    DiscreteMosaicIntent, SpatialIntentProjectionError, _attributes, _fact_nodes,
    _sorted_texts, _text, _unique_fact, project_discrete_mosaic,
)


def _require(condition, message):
    if not condition:
        raise SpatialIntentProjectionError(message)


def _class_axis(ir, selected, axis):
    scope_facts = tuple(fact for fact in ir.hard_facts
        if fact.kind is SpatialFactKind.SCOPE_RANK_SEQUENCE
        and _attributes(fact).get('axis') == axis and selected == set(_fact_nodes(fact)))
    fact = _unique_fact(scope_facts or tuple(fact for fact in ir.hard_facts
        if fact.kind is SpatialFactKind.CLASS_RANK_SEQUENCE
        and _attributes(fact).get('axis') == axis
        and selected <= set(_fact_nodes(fact))), label=f'lattice mosaic {axis} class-rank fact')
    groups = tuple(tuple(group.split(',')) for group in _attributes(fact).get('sequence', '').split(';'))
    nodes = tuple(node for group in groups for node in group)
    bound = _fact_nodes(fact)
    _require(all(group and all(group) for group in groups)
        and len(nodes) == len(set(nodes)) == len(bound) and set(nodes) == set(bound),
        'lattice mosaic class-rank sequence lacks its exact bound occurrence cover')
    raw = {node: rank for rank, group in enumerate(groups) for node in group if node in selected}
    normalized = {rank: index for index, rank in enumerate(sorted(set(raw.values())))}
    return fact, {node: normalized[rank] for node, rank in raw.items()}


def _local_ranks_agree(mosaic, xs, ys):
    expected_x = {f'{owner}[{ordinal:03d}]': rank for owner, ordinal, rank in mosaic.column_ranks}
    expected_y = {item.node_id: rank for rank, row in enumerate(mosaic.rows) for item in row}
    for actual, expected in ((xs, expected_x), (ys, expected_y)):
        normalized = {rank: index for index, rank in enumerate(sorted({actual[node] for node in expected}))}
        _require(all(normalized[actual[node]] == rank for node, rank in expected.items()),
            'lattice mosaic class ranks conflict with the owner exact lattice')


def _exact_rank_alignments(ir, selected, ranks, axis):
    # A horizontal coordinate rank requires vertical anchor alignment, and vice versa.
    alignment_axis = 'vertical' if axis == 'horizontal' else 'horizontal'
    used = []
    for rank in sorted(set(ranks.values())):
        group = {node for node, value in ranks.items() if value == rank}
        if len(group) == 1:
            continue
        fact = _unique_fact(tuple(fact for fact in ir.hard_facts
            if fact.kind is SpatialFactKind.ALIGNMENT
            and _attributes(fact).get('axis') == alignment_axis
            and _attributes(fact).get('reference') == 'anchor_rank'
            and group == set(_fact_nodes(fact)) & selected),
            label=f'lattice mosaic {axis} rank exact alignment')
        used.append(fact)
    return tuple(used)


def _orders_agree(ir, selected, xs, ys):
    for dag, ranks in ((ir.horizontal, xs), (ir.vertical, ys)):
        _require(all(ranks[before] < ranks[after] for before, after in dag.closure_edges()
            if before in selected and after in selected),
            'lattice mosaic class ranks conflict with the source DAG order')


def project_lattice_mosaic(
    ir: AbstractSpatialIR, *, mosaic_id: str, logical_owners: tuple[str, ...],
) -> DiscreteMosaicIntent:
    """Return the complete source occurrence lattice, without assigning a pitch.

    The existing projector remains unchanged. Its single-owner exact partition,
    lattice and transform validation is reused independently for each owner.
    The host must still replay these ASIR facts from its bound source overlay.
    """
    if type(ir) is not AbstractSpatialIR:
        raise TypeError('lattice mosaic projection requires an exact AbstractSpatialIR')
    mosaic_id = _text(mosaic_id, 'lattice mosaic id')
    owners = _sorted_texts(logical_owners, 'lattice mosaic logical owners')
    single = tuple(project_discrete_mosaic(ir, mosaic_id=mosaic_id, logical_owners=(owner,))
                   for owner in owners)
    if len(single) == 1:
        return single[0]
    occurrences = {item.node_id: item for mosaic in single for item in mosaic.occurrences}
    selected = set(occurrences)
    _require(len(selected) == sum(len(mosaic.occurrences) for mosaic in single),
             'lattice mosaic partitions overlap in their exact occurrence cover')
    horizontal, xs = _class_axis(ir, selected, 'horizontal')
    vertical, ys = _class_axis(ir, selected, 'vertical')
    key = 'scope' if horizontal.kind is SpatialFactKind.SCOPE_RANK_SEQUENCE else 'class_key'
    class_key = _attributes(horizontal).get(key)
    _require(horizontal.kind is vertical.kind and class_key and class_key == _attributes(vertical).get(key),
             'lattice mosaic horizontal and vertical rank scope identities differ')
    for mosaic in single:
        _local_ranks_agree(mosaic, xs, ys)
    _require(len({(xs[node], ys[node]) for node in selected}) == len(selected),
             'lattice mosaic repeats an occupied row/column slot')
    alignments = (*_exact_rank_alignments(ir, selected, xs, 'horizontal'),
                  *_exact_rank_alignments(ir, selected, ys, 'vertical'))
    _orders_agree(ir, selected, xs, ys)
    used_ids = {identity for mosaic in single for identity in mosaic.source_fact_ids}
    used_ids.update(fact.fact_id for fact in (horizontal, vertical, *alignments))
    used: tuple[HardInvariant, ...] = tuple(fact for fact in ir.hard_facts if fact.fact_id in used_ids)
    ordered = sorted(selected, key=lambda node: (ys[node], xs[node]))
    return DiscreteMosaicIntent(mosaic_id=mosaic_id, rows=tuple(
        tuple(occurrences[node] for node in ordered if ys[node] == rank)
        for rank in range(len(set(ys.values())))), source_asir_sha256=ir.sha256,
        source_fact_ids=tuple(sorted(used_ids)),
        evidence_ids=tuple(sorted({evidence for fact in used for evidence in fact.evidence_ids})),
        column_ranks=tuple(sorted((item.logical_owner, item.occurrence_ordinal, xs[node])
            for node, item in occurrences.items())))
