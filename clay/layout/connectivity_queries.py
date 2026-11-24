"""Invocation-owned conductor candidates, retaining original fragment order.

Bounding boxes only exclude separated artwork. Connectivity must still use
the original regions, including their holes, for every remaining candidate.
"""
from clay.layout.spatial.collision_queries import RectangleIndex


def region_box(region):
    box = region.bbox()
    return box.left, box.bottom, box.right, box.top


class FragmentQueries:
    def __init__(self, fragments):
        members = {}
        for index, (layer, region) in enumerate(fragments):
            members.setdefault(layer, {}).setdefault(region_box(region), []).append(index)
        self._members = members
        self._indexes = {layer: RectangleIndex(tuple(boxes)) for layer, boxes in members.items()}

    def candidates(self, layer, box, *, include_touching=False):
        index = self._indexes.get(layer)
        if index is None:
            return ()
        members = self._members[layer]
        boxes = index.intersecting(box, include_touching=include_touching)
        # Sorting preserves the original exact-query and disjoint-set union
        # sequence. Equal bounding boxes must retain every distinct fragment.
        return tuple(sorted(member for box in boxes for member in members[box]))
