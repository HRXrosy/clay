"""Layout coverage report.

Given a ``ParameterizedLibrary`` and an ``AnnotatedCell`` (the design intent),
compute device-level and annotation-level coverage metrics:

* layout coverage  -- fraction of devices in the AnnotatedCell that have a
  corresponding NamedCell entry in the library (the device has been laid out).
* annotation coverage -- fraction of devices that carry at least one
  ``LayoutAnnotation``.
* matched-pairs count -- number of ``Matching`` annotation pairs, with
  bidirectional pairs (A->B and B->A) counted once.
* layer kinds present -- sorted list of layer identifiers actually used by any
  polygon in the library; logical names (POLY / NDIFF / METAL1 / ...) are
  preferred when the (layer, datatype) tuple is recognized by the sky130 or
  gf180mcu LayerMap, otherwise a ``L<layer>/D<datatype>`` fallback is used.

The module is pure Python with no external dependencies and does not touch
any of the upstream layout modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple

from .annotations import AnnotatedCell, Matching
from .backend import (
    Gf180mcuLayerMap,
    LogicalLayer,
    ParameterizedLibrary,
    Sky130LayerMap,
)


# Build a reverse lookup table once at import time: (layer, datatype) -> name.
# When sky130 and gf180mcu disagree on (layer, datatype) we still produce a
# stable logical name because both maps share the same LogicalLayer enum.
def _build_layer_name_table() -> dict:
    table: dict = {}
    for pdk in (Sky130LayerMap(), Gf180mcuLayerMap()):
        for logical in LogicalLayer:
            gds = pdk.gds_layer(logical)
            if gds is None:
                continue
            key = (gds.layer, gds.datatype)
            # Prefer the first registered logical name for deterministic output.
            table.setdefault(key, logical.name)
    return table


_LAYER_NAME_TABLE = _build_layer_name_table()


def _layer_label(layer: int, datatype: int) -> str:
    """Return the logical name for a (layer, datatype) tuple, or a fallback."""
    name = _LAYER_NAME_TABLE.get((layer, datatype))
    if name is not None:
        return name
    return f"L{layer}/D{datatype}"


@dataclass(frozen=True)
class CoverageReport:
    """Layout coverage summary.

    Attributes
    ----------
    devices_total:
        Number of devices defined in the AnnotatedCell (cell.layouts).
    devices_with_layout:
        Number of devices that have a matching NamedCell entry in the library
        (matched by device name == NamedCell.name).
    devices_with_annotations:
        Number of devices that carry one or more LayoutAnnotation entries.
    matched_pairs_count:
        Number of unique Matching annotation pairs across all devices; A->B
        and B->A are counted as one pair.
    layer_kinds_present:
        Sorted list of layer labels (logical name when known, else
        L<layer>/D<datatype>) appearing in any NamedCell.polygons of the
        library.
    """

    devices_total: int
    devices_with_layout: int
    devices_with_annotations: int
    matched_pairs_count: int
    layer_kinds_present: List[str] = field(default_factory=list)

    @property
    def layout_coverage_pct(self) -> float:
        if self.devices_total == 0:
            return 0.0
        return 100.0 * self.devices_with_layout / self.devices_total

    @property
    def annotation_coverage_pct(self) -> float:
        if self.devices_total == 0:
            return 0.0
        return 100.0 * self.devices_with_annotations / self.devices_total


def _collect_layer_labels(lib: ParameterizedLibrary) -> List[str]:
    seen: Set[str] = set()
    for named_cell in lib.cells:
        for lp in named_cell.polygons:
            seen.add(_layer_label(lp.layer, lp.datatype))
    return sorted(seen)


def _count_matched_pairs(cell: AnnotatedCell) -> int:
    """Count unique Matching pairs across devices.

    Each ``Matching(other=B)`` on device ``A`` represents the directed edge
    ``A -> B``. We treat the relation as symmetric for counting purposes so a
    bidirectional declaration (A->B and B->A) is counted once. Self-loops
    (A->A) are counted as well, once each.
    """
    pairs: Set[Tuple[str, str]] = set()
    for device, anns in cell.annotations.items():
        for ann in anns:
            if isinstance(ann, Matching):
                other = ann.other
                # canonical ordering deduplicates both directions
                key = (device, other) if device <= other else (other, device)
                pairs.add(key)
    return len(pairs)


def compute_coverage(
    cell: AnnotatedCell,
    lib: ParameterizedLibrary,
) -> CoverageReport:
    """Walk ``cell`` devices, cross-reference with ``lib``, return a report.

    A device is considered "laid out" iff there exists a NamedCell in ``lib``
    whose ``name`` matches the device key. A device is considered "annotated"
    iff it has at least one entry in ``cell.annotations``.
    """
    device_keys = list(cell.layouts.keys())
    cell_names: Set[str] = {nc.name for nc in lib.cells}

    devices_total = len(device_keys)
    devices_with_layout = sum(1 for d in device_keys if d in cell_names)
    devices_with_annotations = sum(
        1
        for d in device_keys
        if cell.annotations.get(d)  # non-empty list
    )

    matched_pairs = _count_matched_pairs(cell)
    layer_kinds = _collect_layer_labels(lib)

    return CoverageReport(
        devices_total=devices_total,
        devices_with_layout=devices_with_layout,
        devices_with_annotations=devices_with_annotations,
        matched_pairs_count=matched_pairs,
        layer_kinds_present=layer_kinds,
    )


def render_coverage_text(report: CoverageReport) -> str:
    """Plain-text multi-line summary suitable for terminals / logs."""
    layers = ", ".join(report.layer_kinds_present) if report.layer_kinds_present else "(none)"
    lines = [
        f"Layout coverage: {report.devices_with_layout}/{report.devices_total} devices ({report.layout_coverage_pct:.0f}%)",
        f"Annotation coverage: {report.devices_with_annotations}/{report.devices_total} devices ({report.annotation_coverage_pct:.0f}%)",
        f"Matched pairs: {report.matched_pairs_count}",
        f"Layers present: {layers}",
    ]
    return "\n".join(lines)


__all__ = [
    "CoverageReport",
    "compute_coverage",
    "render_coverage_text",
]
