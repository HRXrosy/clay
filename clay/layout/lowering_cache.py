"""LRU cache for Phase 5 DRC-complete lowering.

Wraps ``lower_annotated_cell_drc_complete`` so repeated calls with the same
(cell name, PDK type, fin_pitch_nm, frozen sizing, DRC kwargs) tuple skip
recomputation. Useful in optimization loops (cosizing, parasitic-aware
resizing) where the same cell topology is re-lowered many times with
identical inputs while only an outer loop variable changes.

The cache is keyed on coarse identity (cell_name + PDK class name) plus the
full sizing + DRC kwargs, not on full structural hashing of the
AnnotatedCell — callers should treat distinct topologies as distinct
``cell_name`` values, which matches existing convention in
``clay/layout/cell_templates``.

LRU eviction uses ``collections.OrderedDict`` (move-to-end on hit, popitem
front on overflow).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Tuple

from .backend import lower_annotated_cell_drc_complete


class LoweringCache:
    """LRU cache for Phase 5.3 DRC-complete lowering.

    Parameters
    ----------
    max_size : int
        Maximum number of cached entries. Oldest entry by last-access
        order is evicted on overflow. Must be >= 1.
    """

    def __init__(self, max_size: int = 128):
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._store: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _build_key(
        cell,
        pdk_layers,
        fin_pitch_nm: float,
        sizing,
        min_poly_space_nm: Optional[float],
        min_well_diff_enclosure_nm: Optional[float],
    ) -> Tuple[Any, ...]:
        # Freeze sizing: dict[str, tuple[float, float]] -> sorted tuple of items.
        frozen_sizing = tuple(sorted(sizing.items()))
        return (
            cell.cell_name,
            type(pdk_layers).__name__,
            fin_pitch_nm,
            frozen_sizing,
            min_poly_space_nm,
            min_well_diff_enclosure_nm,
        )

    def get_or_compute(
        self,
        cell,
        pdk_layers,
        fin_pitch_nm,
        sizing,
        *,
        min_poly_space_nm=None,
        min_well_diff_enclosure_nm=None,
    ):
        """Lookup or compute the DRC-complete ``ParameterizedLibrary``.

        On hit, the cached library is returned without recomputation and
        the entry is moved to the most-recently-used position. On miss,
        ``lower_annotated_cell_drc_complete`` is invoked and the result
        is stored (with LRU eviction when ``len(cache) > max_size``).
        """
        key = self._build_key(
            cell,
            pdk_layers,
            fin_pitch_nm,
            sizing,
            min_poly_space_nm,
            min_well_diff_enclosure_nm,
        )
        if key in self._store:
            self._hits += 1
            self._store.move_to_end(key)
            return self._store[key]

        self._misses += 1
        lib = lower_annotated_cell_drc_complete(
            cell,
            pdk_layers,
            fin_pitch_nm,
            sizing,
            min_poly_space_nm=min_poly_space_nm,
            min_well_diff_enclosure_nm=min_well_diff_enclosure_nm,
        )
        self._store[key] = lib
        if len(self._store) > self._max_size:
            # Evict least-recently-used (front of OrderedDict).
            self._store.popitem(last=False)
        return lib

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        """Drop all cached entries and reset hit/miss counters."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
