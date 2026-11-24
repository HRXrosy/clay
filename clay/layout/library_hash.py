"""Canonical hash for ``ParameterizedLibrary``.

Produces a deterministic, order-independent SHA-256 fingerprint that lets two
libraries with semantically identical content (cells / polygons / layers /
units / name) but different *insertion order* hash to the same digest.

Design
------
* The canonical form is JSON (stdlib ``json``) emitted with ``sort_keys=True``
  at every level so the same dictionary structure always serializes to the
  same byte sequence.
* Top-level fields (``name``, ``unit_user``, ``unit_db_meters``, ``cells``)
  follow the existing layout in :mod:`clay.layout.json_serde`, so the hash is
  stable against any future caller that compares ``canonical_json`` against
  a normalized JSON file on disk.
* ``cells`` are sorted by ``name`` (ASCII lexicographic order). This is the
  order-independence axis: callers that build libraries by appending cells
  in different orders still hash to the same digest.
* Inside each cell, ``layer_polygons`` are sorted by
  ``(layer, datatype, polygon-vertices)`` so two equivalent cells whose
  polygons were added in different orders also normalize to the same form.
  ``LogicalLayer`` is not stored on ``LayerPolygon`` — the Phase 3a IR carries
  ``(layer, datatype)`` GDS integer pairs, which serve as the same identity
  key as the spec's ``logical_layer.name``.
* Point lists *inside* a polygon are NOT reordered: vertex order is
  geometrically meaningful (it defines the polygon's winding / outline), so
  reordering would corrupt the polygon. The vertex tuple is, however, used
  as a sort key when ordering polygons that share the same ``(layer, datatype)``.

The hash is pure stdlib (``hashlib`` + ``json``) — no external deps, no
pickling, and no dependence on Python's dict-insertion order beyond
``sort_keys=True`` handling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .backend import LayerPolygon, NamedCell, ParameterizedLibrary, Point, Polygon


@dataclass(frozen=True)
class LibraryHashResult:
    """Result of hashing a :class:`ParameterizedLibrary`.

    Attributes
    ----------
    sha256_hex
        64-character lowercase hex digest of the canonical JSON form.
    cell_count
        Number of ``NamedCell`` entries in the library.
    polygon_count
        Total ``LayerPolygon`` count across every cell and every layer.
    """

    sha256_hex: str
    cell_count: int
    polygon_count: int


# --- Canonical form construction ---


def _point_pair(p: Point) -> List[int]:
    """Render a Point as a ``[x, y]`` JSON list (vertex-order preserving)."""
    return [int(p.x), int(p.y)]


def _polygon_dict(poly: Polygon) -> Dict[str, Any]:
    """Render a Polygon: vertex list preserved in original (geometric) order."""
    return {"vertices": [_point_pair(v) for v in poly.vertices]}


def _polygon_sort_key(lp: LayerPolygon) -> Tuple[int, int, Tuple[Tuple[int, int], ...]]:
    """Sort key for layer polygons inside a cell.

    The Phase 3a backend carries ``(layer, datatype)`` integer pairs in place
    of an explicit ``logical_layer.name`` — these GDS keys uniquely identify
    the same logical layer for a fixed PDK, and they sort deterministically
    without needing the ``LogicalLayer`` enum.
    """
    verts = tuple((int(v.x), int(v.y)) for v in lp.polygon.vertices)
    return (int(lp.layer), int(lp.datatype), verts)


def _layer_polygon_dict(lp: LayerPolygon) -> Dict[str, Any]:
    return {
        "layer": int(lp.layer),
        "datatype": int(lp.datatype),
        "polygon": _polygon_dict(lp.polygon),
    }


def _named_cell_dict(cell: NamedCell) -> Dict[str, Any]:
    """Render a NamedCell with its polygons sorted canonically."""
    sorted_polys = sorted(cell.polygons, key=_polygon_sort_key)
    return {
        "name": str(cell.name),
        "polygons": [_layer_polygon_dict(lp) for lp in sorted_polys],
    }


def _library_dict(lib: ParameterizedLibrary) -> Dict[str, Any]:
    sorted_cells = sorted(lib.cells, key=lambda c: c.name)
    return {
        "name": str(lib.name),
        "unit_user": float(lib.unit_user),
        "unit_db_meters": float(lib.unit_db_meters),
        "cells": [_named_cell_dict(c) for c in sorted_cells],
    }


# --- Public API ---


def canonical_json(lib: ParameterizedLibrary) -> str:
    """Render ``lib`` to its canonical JSON string form.

    The output is stable across:

    * cell insertion order (cells sorted by ``name``);
    * polygon insertion order within a cell (sorted by layer / datatype /
      vertex tuple);
    * Python dict ordering (``json.dumps`` is called with ``sort_keys=True``).

    Point order *inside* a polygon is preserved (vertex order is geometric).
    """
    return json.dumps(
        _library_dict(lib),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def library_hash(lib: ParameterizedLibrary) -> LibraryHashResult:
    """Compute SHA-256 over ``canonical_json(lib)``.

    Returns the digest plus accounting counts useful for cache invalidation
    or quick sanity-checks (cell count, total polygon count).
    """
    payload = canonical_json(lib).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()
    cell_count = len(lib.cells)
    polygon_count = sum(len(c.polygons) for c in lib.cells)
    return LibraryHashResult(
        sha256_hex=sha,
        cell_count=cell_count,
        polygon_count=polygon_count,
    )


def compare_hashes(lib1: ParameterizedLibrary, lib2: ParameterizedLibrary) -> bool:
    """Return True iff the two libraries hash to the same digest."""
    return library_hash(lib1).sha256_hex == library_hash(lib2).sha256_hex
