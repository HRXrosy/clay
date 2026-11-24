"""Conservative frame classification over one complete measured native graph.

This pure geometry utility grants no reference authority. The current reference
producer alone supplies the full graph, leaf multiset and source associations.
Unknown or incomplete artwork remains in the comparison frame.
"""
from collections import Counter, defaultdict
from dataclasses import asdict

import klayout.db as kdb

from clay.frontend.reference_device_shapes import _region, _require

FULL_FRAME = 'all_nontext_polygons_all_layers_complete_selected_top'
ASSOCIATED_FRAME = 'all_nontext_artwork_except_proven_complete_unmapped_interconnect_leaves'


def require_reference_frame_scope(value):
    _require(type(value) is str and value in ('full', 'circuit_associated'),
             'unsupported reference frame scope')


def _frame(parts, scope):
    points = tuple(point for _key, shapes in parts for shape in shapes for point in shape.hull_nm)
    _require(bool(points), 'complete reference artwork frame is empty')
    return dict(scope=scope, bbox_nm=(min(x for x, _ in points), min(y for _, y in points),
        max(x for x, _ in points), max(y for _, y in points)),
        layers=tuple(key for key, shapes in parts if shapes), polygons=sum(len(shapes) for _, shapes in parts))


def _node_parts(node, parts, metals, vias, checkpoint):
    """Keep original polygons; never clip a raw shape to a measured node."""
    active = {key: value for key, value in node.items() if not value.is_empty()}
    if not active or not set(active) <= metals:
        return None, 'retained_nonrouting_conductor'
    claimed, coverage = Counter(), defaultdict(kdb.Region)
    for key, shapes in parts:
        checkpoint()
        conductor = active.get(key)
        if conductor is None:
            contacts = tuple(active[lower] & active[upper] for cut, lower, upper in vias
                             if cut == key and lower in active and upper in active)
            conductor = sum(contacts, kdb.Region()) if contacts else None
        if conductor is None:
            continue
        for shape in shapes:
            checkpoint()
            region = shape.to_region()
            if (region & conductor).is_empty():
                continue
            if not (region - conductor).is_empty():
                return None, 'retained_partial_raw_polygon'
            claimed[(key, shape)] += 1
            if key in active:
                coverage[key] += region
    if any(not (value ^ coverage[key]).is_empty() for key, value in active.items()):
        return None, 'retained_incomplete_raw_node'
    return claimed, None


def _text_associated(path, parts, texts):
    envelope = _region(tuple(shape for _key, shape in parts)).bbox()
    return any(text_path == path or text_path.startswith(path + '/') or
               (envelope.left <= point[0] <= envelope.right and
                envelope.bottom <= point[1] <= envelope.top)
               for text_path, _layer, _text, point in texts)


def classify_reference_frame(*, all_parts, nodes, connect_model, routing_metal_layers,
        mapped_nodes, device_nodes, gate_nodes, protected_paths, leaves, texts,
        checkpoint, frame_scope='full'):
    """Exclude only a whole, unmapped routing-only leaf equalling one whole node.

    Both raw polygon multiplicity and the complete node universe are preserved
    in the classification. Labels, device conductors, unknown layers, cross-leaf
    nodes and any partial raw polygon prevent exclusion.
    """
    require_reference_frame_scope(frame_scope)
    checkpoint()
    _require(type(nodes) is tuple and nodes and type(all_parts) is tuple,
             'complete current graph and raw artwork require exact tuples')
    universe = tuple(f'node:{index}' for index in range(len(nodes)))
    _require(set((*mapped_nodes, *device_nodes, *gate_nodes)) <= set(universe),
             'unknown protected node outside complete current node denominator')
    _require(len({path for path, _master, _parts in leaves}) == len(leaves),
             'duplicate reference leaf occurrence path')
    metals = set(routing_metal_layers) & set(connect_model.metals)
    derived = {item.name for item in connect_model.derived}
    metals -= derived
    vias = tuple((cut, lower, upper) for cut, lower, upper in connect_model.vias
                 if cut not in derived and lower in metals and upper in metals)
    allowed = metals | {cut for cut, _lower, _upper in vias}
    raw = Counter((key, shape) for key, shapes in all_parts for shape in shapes)
    protected = set(mapped_nodes) | set(device_nodes) | set(gate_nodes)
    removed, exclusions, classifications = Counter(), [], []
    for node_id, node in zip(universe, nodes):
        checkpoint()
        if node_id in protected:
            classifications.append(dict(node_id=node_id, classification='retained_source_or_device_associated'))
            continue
        if frame_scope == 'full':
            classifications.append(dict(node_id=node_id, classification='retained_full_frame_policy'))
            continue
        claimed, reason = _node_parts(node, all_parts, metals, vias, checkpoint)
        if reason is not None:
            classifications.append(dict(node_id=node_id, classification=reason))
            continue
        matches = []
        for path, master, parts in leaves:
            checkpoint()
            if not parts or any(key not in allowed for key, _shape in parts):
                continue
            if any(path == base or path.startswith(base + '/') for base in protected_paths):
                continue
            if Counter(parts) != claimed or _text_associated(path, parts, texts):
                continue
            matches.append((path, master, parts))
        if len(matches) != 1 or not claimed or (claimed - raw) or (claimed & removed):
            classifications.append(dict(node_id=node_id, classification='retained_no_unique_complete_unassociated_leaf'))
            continue
        path, master, parts = matches[0]
        removed.update(claimed)
        exclusions.append(dict(node_id=node_id, path=path, master=master, polygons=len(parts),
            parts=tuple((key, asdict(shape)) for key, shape in parts),
            classification='complete_unmapped_interconnect_leaf_and_complete_node'))
        classifications.append(dict(node_id=node_id, classification='excluded_complete_unmapped_interconnect_leaf'))
    remaining = raw - removed
    _require(remaining + removed == raw, 'reference frame raw polygon multiset partition differs')
    by_layer = defaultdict(list)
    for (key, shape), count in sorted(remaining.items()):
        by_layer[key].extend((shape,) * count)
    selected = tuple((key, tuple(shapes)) for key, shapes in sorted(by_layer.items()))
    full = _frame(all_parts, FULL_FRAME)
    comparison = full if frame_scope == 'full' else _frame(selected, ASSOCIATED_FRAME)
    return dict(scope=frame_scope, full_frame=full, comparison_frame=comparison,
        excluded_leaves=tuple(exclusions), excluded_polygons=sum(removed.values()),
        node_count=len(universe), node_classifications=tuple(classifications),
        mapped_nodes=tuple(sorted(set(mapped_nodes))), device_nodes=tuple(sorted(set(device_nodes))),
        gate_nodes=tuple(sorted(set(gate_nodes))), protected_paths=tuple(sorted(set(protected_paths))),
        text_records=tuple(texts))
