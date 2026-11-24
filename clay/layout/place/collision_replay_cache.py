"""Exact translation-invariant cache for repeated rigid geometry replays."""
from dataclasses import dataclass
from functools import lru_cache


def _checkpoint():
    from clay.flow.execution_session import current_session
    session = current_session()
    return session.checkpoint if session is not None else None


@dataclass(frozen=True, eq=False)
class OwnerGeometry:
    # Identity is the cache token. Contents are created privately and never
    # changed; different PDK geometry/layer inventories cannot share a token.
    layers: dict
    indexes: dict


def collision_groups(dims, rigid, layered):
    from clay.layout.place.constraints_spatial import _collision_atoms_by_owner
    # Preserve original input validation on every call. Tuple hashing considers
    # bool/int values equal, so caching before strict geometry validation could
    # incorrectly admit a malformed inventory after a valid equal-valued one.
    local = _collision_atoms_by_owner(dims=dict(dims),
        rigid_compound_footprints=rigid, layered_collision_footprints=layered)
    return _indexed_groups(tuple(sorted(local.items())))


@lru_cache(maxsize=8)
def _indexed_groups(local):
    from clay.layout.spatial.collision_queries import RectangleIndex
    groups = {}
    checkpoint = _checkpoint()
    for name, atoms in local:
        layers = {}
        for layer, box in atoms:
            layers.setdefault(layer, []).append(box)
        layers = {layer:tuple(boxes) for layer,boxes in layers.items()}
        groups[name] = OwnerGeometry(layers, {
            layer:RectangleIndex(boxes, checkpoint=checkpoint)
            for layer,boxes in layers.items()})
    return groups


@lru_cache(maxsize=65536)
def relative_pair_conflicts(left, right, left_layer, right_layer, gap, dx, dy):
    """Replay the original rectangles, in right-owner local coordinates.

    The effective process/pair/layer clearance is part of the key. Pair and
    layer exemptions remain caller decisions, before consulting this cache.
    Global translations and movement of unrelated owners cannot change it.
    """
    translated = tuple((x0+dx,y0+dy,x1+dx,y1+dy)
                       for x0,y0,x1,y1 in left.layers[left_layer])
    return right.indexes[right_layer].conflicts(
        translated, clearance_nm=gap, checkpoint=_checkpoint())
