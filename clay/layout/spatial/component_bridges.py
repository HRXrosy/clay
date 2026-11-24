"""Ordered closest-gap bridges for a fixed set of axis-aligned boxes.

This preserves the legacy bbox-based bridge rule, including corner contact and
input-order tie breaking. It does not confer layer legality or fill cavities in
other collision contracts. A component forest owns all state for this call.
"""


def _edge(a, b, i, j):
    candidates = []
    for direction, axis in enumerate((0, 1)):
        lo, hi = max(a[axis], b[axis]), min(a[axis+2], b[axis+2])
        if lo >= hi:
            continue
        other = 1-axis
        if a[other+2] <= b[other]:
            start, end = a[other+2], b[other]
        elif b[other+2] <= a[other]:
            start, end = b[other+2], a[other]
        else:
            continue
        box = (lo, start, hi, end) if axis == 0 else (start, lo, end, hi)
        candidates.append((end-start, i, j, direction, box))
    return min(candidates) if candidates else None


def closest_gap_bridges(boxes):
    """Return the exact greedy bridge order using linear auxiliary storage.

    The original scan chooses edges by (gap, input pair, vertical-before-
    horizontal). This total order defines one minimum spanning forest after
    touching boxes are joined. Each component's minimum outgoing edge belongs
    to that same forest. Boruvka rounds find its edges without rescanning every
    pair after each single merge; sorting the selected edges restores the exact
    original greedy output order. No new strap becomes an input candidate box.
    """
    boxes = tuple(tuple(box) for box in boxes)
    parent = list(range(len(boxes)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(i, j):
        left, right = find(i), find(j)
        if left == right:
            return False
        parent[left] = right
        return True

    for i, a in enumerate(boxes):
        for j in range(i+1, len(boxes)):
            b = boxes[j]
            if max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3]):
                union(i, j)
    selected = []
    while len({find(i) for i in range(len(boxes))}) > 1:
        nearest = {}
        for i, a in enumerate(boxes):
            left = find(i)
            for j in range(i+1, len(boxes)):
                right = find(j)
                if left == right:
                    continue
                edge = _edge(a, boxes[j], i, j)
                if edge is None:
                    continue
                for component in (left, right):
                    if component not in nearest or edge < nearest[component]:
                        nearest[component] = edge
        if not nearest:
            break
        for edge in sorted(set(nearest.values())):
            if union(edge[1], edge[2]):
                selected.append(edge)
    return tuple(edge[-1] for edge in sorted(selected))
