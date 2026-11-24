"""Exact half-open integer rectangle unions without saturated polygon merging.

The sweep owns its temporary coverage tree; there is no cross-candidate cache.
Its output partitions the same plane region into disjoint rectangles, including
holes, and accepts Python integer coordinates without a geometry-library cast.
"""
from collections import defaultdict
from collections.abc import Iterable

from clay._native import load_native

Box = tuple[int, int, int, int]
_ORIG_DEFAULTDICT = defaultdict


def _rectangle_union_cover_py(boxes: Iterable[Box], *, checkpoint=None) -> tuple[Box, ...]:
    if checkpoint:
        checkpoint()
    rows = tuple(boxes)
    endpoints = set()
    for index, row in enumerate(rows):
        if checkpoint and index % 256 == 0:
            checkpoint()
        if len(row) != 4 or any(type(v) is not int for v in row):
            raise TypeError('rectangle union requires four exact integers')
        if row[0] >= row[2] or row[1] >= row[3]:
            raise ValueError('rectangle union requires positive rectangles')
        endpoints.update((row[1], row[3]))
    if not rows:
        return ()
    ys = sorted(endpoints)
    rank = {}
    for i, v in enumerate(ys):
        if checkpoint and i % 256 == 0:
            checkpoint()
        rank[v] = i
    events = defaultdict(list)
    for index, (x0, y0, x1, y1) in enumerate(rows):
        if checkpoint and index % 256 == 0:
            checkpoint()
        lo, hi = rank[y0], rank[y1]
        events[x0].append((lo, hi, 1))
        events[x1].append((lo, hi, -1))
    counts = [0] * (4 * len(ys))
    occupied = [False] * len(counts)
    visited_nodes = 0

    def update(node, lo, hi, qlo, qhi, delta):
        if qlo <= lo and hi <= qhi:
            counts[node] += delta
        else:
            mid = (lo + hi) // 2
            if qlo < mid:
                update(node * 2, lo, mid, qlo, qhi, delta)
            if qhi > mid:
                update(node * 2 + 1, mid, hi, qlo, qhi, delta)
        occupied[node] = counts[node] > 0 or (
            hi - lo > 1 and (occupied[node * 2] or occupied[node * 2 + 1]))

    def intervals(node, lo, hi, out):
        nonlocal visited_nodes
        if checkpoint:
            visited_nodes += 1
            if visited_nodes % 256 == 0:
                checkpoint()
        if not occupied[node]:
            return
        if counts[node] > 0:
            if out and out[-1][1] == ys[lo]:
                out[-1] = (out[-1][0], ys[hi])
            else:
                out.append((ys[lo], ys[hi]))
        else:
            mid = (lo + hi) // 2
            intervals(node * 2, lo, mid, out)
            intervals(node * 2 + 1, mid, hi, out)

    cover = []
    open_boxes = {}
    previous = None
    updated_events = 0
    for x in sorted(events):
        if checkpoint:
            checkpoint()
        if previous is not None and x > previous:
            spans = []
            intervals(1, 0, len(ys)-1, spans)
            current = {}
            for index, (y0, y1) in enumerate(spans):
                if checkpoint and index % 256 == 0:
                    checkpoint()
                old = open_boxes.pop((y0, y1), None)
                current[(y0, y1)] = (old[0] if old else previous, y0, x, y1)
            cover.extend(open_boxes.values())
            open_boxes = current
        for lo, hi, delta in events[x]:
            updated_events += 1
            if checkpoint and updated_events % 256 == 0:
                checkpoint()
            update(1, 0, len(ys)-1, lo, hi, delta)
        previous = x
    cover.extend(open_boxes.values())
    result = tuple(sorted(cover))
    if checkpoint:
        checkpoint()
    return result


def rectangle_union_cover(boxes: Iterable[Box], *, checkpoint=None) -> tuple[Box, ...]:
    native = load_native("_place_cpp")
    if native is not None and defaultdict is _ORIG_DEFAULTDICT:
        if type(boxes) in (tuple, list):
            try:
                return native.rectangle_union_cover_cpp(boxes, checkpoint)
            except OverflowError:
                return _rectangle_union_cover_py(boxes, checkpoint=checkpoint)
        if checkpoint:
            checkpoint()
        rows = tuple(boxes)
        try:
            return native.rectangle_union_cover_cpp(rows, checkpoint)
        except OverflowError:
            return _rectangle_union_cover_py(rows, checkpoint=checkpoint)
    return _rectangle_union_cover_py(boxes, checkpoint=checkpoint)
