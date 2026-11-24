"""Canonical ranking of group landing alternatives across source and portal classes."""
from itertools import zip_longest
import json

from clay.layout.route.canonical_input_capture import _json_value
from clay.layout.route.engine_gridless import _group_route_pin_key


def ordered_alternatives(net, candidates, selected, envelope=None):
    """Deterministically rank group landing alternatives without pruning the domain.

    When ``envelope`` is provided, groups portals by boundary side(s) and layer
    within each source terminal stream before round-robin interleaving.
    """
    bounds = None if envelope is None else tuple(envelope)

    def source(row):
        return json.dumps(
            _json_value(_group_route_pin_key(net, row[4])),
            sort_keys=True,
            separators=(',', ':'),
        )

    def portal_class(row):
        if bounds is None:
            return tuple(row[10])
        x0, y0, x1, y1 = bounds
        x, y, layer = row[10]
        sides = tuple(
            name
            for condition, name in (
                (x == x0, 'left'),
                (x == x1, 'right'),
                (y == y0, 'bottom'),
                (y == y1, 'top'),
            )
            if condition
        )
        if not sides:
            raise ValueError('group portal is not on its declared envelope')
        return sides, layer

    def rank(row):
        return (
            *row[:4],
            source(row),
            row[5],
            tuple(row[6]),
            tuple(row[7]),
            row[9],
            tuple(row[10]),
        )

    by_source = {}
    for row in candidates:
        if row != selected:
            by_source.setdefault(source(row), {}).setdefault(portal_class(row), []).append(row)
    access_streams = []
    for by_portal in by_source.values():
        for rows in by_portal.values():
            rows.sort(key=rank)
        portals = sorted(by_portal.values(), key=lambda rows: rank(rows[0]))
        stream = (row for depth in zip_longest(*portals) for row in depth if row is not None)
        access_streams.append((min(rank(rows[0]) for rows in portals), stream))
    access_streams.sort(key=lambda pair: pair[0])
    return (
        row
        for depth in zip_longest(*(stream for _, stream in access_streams))
        for row in depth
        if row is not None
    )


__all__ = ['ordered_alternatives']
