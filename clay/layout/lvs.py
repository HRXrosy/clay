"""Phase 12 (bonus) — Python-side LVS topology match baseline.

Compares a :class:`ParameterizedLibrary` against an expected device manifest,
checking device-count parity and per-device per-layer polygon counts.

This is intentionally a topology / counts-only baseline (NOT a geometric LVS):
it answers the cheap question "did the lowering produce the expected set of
device cells, each with the expected GDS layer footprint?" before reaching for
a real LVS engine (netgen / magic).

Mirrors the style of Rust-side ``crates/clay_layout/src/backend/*`` Python
twins. Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .backend import ParameterizedLibrary


# (layer, datatype) GDS key.
LayerKey = Tuple[int, int]


@dataclass
class LvsDeviceExpectation:
    """One expected device cell + its per-(layer, datatype) polygon counts.

    ``name`` is the expected :class:`NamedCell` name in the library.
    ``expected_layer_counts`` maps a GDS (layer, datatype) pair to the number
    of polygons that the cell is expected to contain on that layer.
    """

    name: str
    expected_layer_counts: Dict[LayerKey, int] = field(default_factory=dict)


@dataclass
class LvsLayerCountMismatch:
    """Per-device-per-layer count divergence (helper record for LvsResult)."""

    device: str
    layer: LayerKey
    expected: int
    actual: int


@dataclass
class LvsResult:
    """Outcome of :func:`check_topology_match`.

    ``matched`` is ``True`` iff there are no missing devices, no extra devices,
    and no layer-count mismatches.
    """

    matched: bool
    missing_devices: List[str] = field(default_factory=list)
    extra_devices: List[str] = field(default_factory=list)
    layer_count_mismatches: List[LvsLayerCountMismatch] = field(default_factory=list)


def _layer_counts_for_cell(cell) -> Dict[LayerKey, int]:
    """Tally polygons in ``cell`` keyed by (layer, datatype)."""
    counts: Dict[LayerKey, int] = {}
    for lp in cell.polygons:
        key = (lp.layer, lp.datatype)
        counts[key] = counts.get(key, 0) + 1
    return counts


def check_topology_match(
    lib: ParameterizedLibrary,
    expected: List[LvsDeviceExpectation],
) -> LvsResult:
    """Return an :class:`LvsResult` for ``lib`` vs ``expected``.

    Cell-name matching is **case-sensitive** (mirrors GDS structure-name
    semantics — names are bytes/ASCII identifiers that real tools treat as
    exact strings). A device that appears in ``expected`` but not in ``lib``
    is reported as ``missing``; a device that appears in ``lib.cells`` but
    not in ``expected`` is reported as ``extra``. Per-layer polygon counts
    are only compared for devices present in both sides.

    A device whose ``expected_layer_counts`` is empty is considered to match
    on the layer axis regardless of what polygons the actual cell contains.
    """
    expected_by_name: Dict[str, LvsDeviceExpectation] = {
        exp.name: exp for exp in expected
    }
    lib_by_name = {cell.name: cell for cell in lib.cells}

    missing_devices: List[str] = [
        name for name in expected_by_name if name not in lib_by_name
    ]
    extra_devices: List[str] = [
        name for name in lib_by_name if name not in expected_by_name
    ]

    mismatches: List[LvsLayerCountMismatch] = []
    for name, exp in expected_by_name.items():
        cell = lib_by_name.get(name)
        if cell is None:
            continue
        actual_counts = _layer_counts_for_cell(cell)
        for layer_key, expected_count in exp.expected_layer_counts.items():
            actual_count = actual_counts.get(layer_key, 0)
            if actual_count != expected_count:
                mismatches.append(
                    LvsLayerCountMismatch(
                        device=name,
                        layer=layer_key,
                        expected=expected_count,
                        actual=actual_count,
                    )
                )

    matched = (
        not missing_devices
        and not extra_devices
        and not mismatches
    )
    return LvsResult(
        matched=matched,
        missing_devices=missing_devices,
        extra_devices=extra_devices,
        layer_count_mismatches=mismatches,
    )
