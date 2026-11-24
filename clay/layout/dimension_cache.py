"""Memoized dimension cache for polygon bbox / area / perimeter.

The shoelace area and edge-sum perimeter loops in clay/layout/parasitic.py
are called once per polygon per extract_parasitic pass. When the same
polygon is queried repeatedly (e.g. multi-pass parasitic re-estimation,
adjoint sensitivity sweeps, or DRC re-runs), recomputing them is wasted
work. DimensionCache memoizes by a tuple of polygon vertex (x, y) pairs.

Design notes:
- Cache key is the (op, vertex_tuple) pair. The DBU dimension (area_dbu /
  perim_dbu) is what's expensive (loop over vertices); the final scalar
  multiplication by dbu_to_um / dbu_to_um**2 is cheap and not memoized,
  which lets one cached entry serve any dbu_to_um value.
- No canonicalization of vertex order: a polygon listed (0,0)-(10,0)-
  (10,10)-(0,10) is a different cache key from (10,10)-(0,10)-(0,0)-
  (10,0). Callers wanting normalization must do it upstream.
- OrderedDict backs an LRU: hits move to most-recent; on insertion past
  max_size, the oldest entry is evicted.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Tuple


_BBOX = "bbox"
_AREA_DBU = "area_dbu"
_PERIM_DBU = "perim_dbu"


def _vertex_key(polygon) -> Tuple[Tuple[int, int], ...]:
    """Build a hashable key from polygon.vertices preserving order."""
    return tuple((v.x, v.y) for v in polygon.vertices)


class DimensionCache:
    """LRU-memoized polygon dimension cache.

    Stores bbox / area-in-DBU / perimeter-in-DBU keyed by (op, vertex
    tuple). Area and perimeter return values are scaled by dbu_to_um at
    call time so different unit conversions still share a cache entry.
    """

    def __init__(self, max_size: int = 4096):
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self._max_size = max_size
        self._store: "OrderedDict[Tuple[str, Tuple[Tuple[int, int], ...]], object]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    # --- properties ---

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        return len(self._store)

    # --- public API ---

    def bbox(self, polygon) -> Tuple[int, int, int, int]:
        """Return (x_min, y_min, x_max, y_max) in DBU. Memoized."""
        vkey = _vertex_key(polygon)
        key = (_BBOX, vkey)
        cached = self._lookup(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        if not vkey:
            raise ValueError("cannot compute bbox of empty polygon")
        x_min = vkey[0][0]
        y_min = vkey[0][1]
        x_max = x_min
        y_max = y_min
        for (vx, vy) in vkey[1:]:
            if vx < x_min:
                x_min = vx
            if vx > x_max:
                x_max = vx
            if vy < y_min:
                y_min = vy
            if vy > y_max:
                y_max = vy
        result = (x_min, y_min, x_max, y_max)
        self._insert(key, result)
        return result

    def area_um2(self, polygon, dbu_to_um: float = 0.001) -> float:
        """Polygon area in um^2 via shoelace formula. DBU result is cached."""
        vkey = _vertex_key(polygon)
        key = (_AREA_DBU, vkey)
        cached = self._lookup(key)
        if cached is None:
            n = len(vkey)
            if n < 3:
                area_dbu2 = 0.0
            else:
                s = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    s += vkey[i][0] * vkey[j][1]
                    s -= vkey[j][0] * vkey[i][1]
                area_dbu2 = abs(s) / 2.0
            self._insert(key, area_dbu2)
        else:
            area_dbu2 = cached  # type: ignore[assignment]
        return area_dbu2 * (dbu_to_um ** 2)

    def perimeter_um(self, polygon, dbu_to_um: float = 0.001) -> float:
        """Polygon perimeter in um (closed loop). DBU result is cached."""
        vkey = _vertex_key(polygon)
        key = (_PERIM_DBU, vkey)
        cached = self._lookup(key)
        if cached is None:
            n = len(vkey)
            if n < 2:
                perim_dbu = 0.0
            else:
                perim_dbu = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    dx = vkey[j][0] - vkey[i][0]
                    dy = vkey[j][1] - vkey[i][1]
                    perim_dbu += math.hypot(dx, dy)
            self._insert(key, perim_dbu)
        else:
            perim_dbu = cached  # type: ignore[assignment]
        return perim_dbu * dbu_to_um

    def clear(self) -> None:
        """Drop all cached entries and reset hit/miss counters."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    # --- internal LRU plumbing ---

    def _lookup(self, key):
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def _insert(self, key, value) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)
