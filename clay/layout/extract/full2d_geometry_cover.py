"""Exact rigid-owner collision covers without materializing atom-pair objects."""
from itertools import combinations


def build_geometry_cover(sizes, ownership, layers, gap, compatible=()):
    """Return the exact forbidden integer displacement union and representatives.

    Each rectangle is the same strict-inequality conversion used by the full
    atom-pair encoding. Geometry is neither enlarged nor approximated. Only
    one owner pair's raw rectangles is retained at a time.
    """
    from clay.layout.place.full2d_pair_cover import _exact_forbidden_union

    sizes = dict(sizes)
    layers = dict(layers)
    groups = {}
    for node, owner, offset in ownership:
        w, h = sizes[node]
        groups.setdefault(owner, {}).setdefault(layers[node], []).append(
            (node, (offset[0], offset[1], offset[0]+w, offset[1]+h)))
    compatible = set(compatible)
    covers, representatives, pair_count = [], [], 0
    for first_owner, second_owner in combinations(sorted(groups), 2):
        if (first_owner, second_owner) in compatible:
            continue
        first_layers, second_layers = groups[first_owner], groups[second_owner]
        forbidden, representative = set(), None
        for first_layer, first_rows in first_layers.items():
            for second_layer, second_rows in second_layers.items():
                if first_layer is not None and second_layer is not None and first_layer != second_layer:
                    continue
                pair_count += len(first_rows)*len(second_rows)
                for first_name, a in first_rows:
                    for second_name, b in second_rows:
                        pair = tuple(sorted((first_name, second_name)))
                        if representative is None or pair < representative:
                            representative = pair
                        forbidden.add((a[0]-b[2]-gap+1, a[1]-b[3]-gap+1,
                                       a[2]+gap-b[0], a[3]+gap-b[1]))
        if representative is not None:
            covers.append(((first_owner,second_owner), _exact_forbidden_union(forbidden)))
            representatives.append(representative)
    return tuple(covers), tuple(sorted(representatives)), pair_count


def cover_collisions(covers, boxes):
    """Replay every physical collision region against a master placement."""
    for (first, second), rectangles in covers:
        dx = boxes[second][0]-boxes[first][0]
        dy = boxes[second][1]-boxes[first][1]
        if any(x0 <= dx < x1 and y0 <= dy < y1 for x0,y0,x1,y1 in rectangles):
            yield first, second
