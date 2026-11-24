"""Snapshot diff utility for ``AnnotatedCell``.

Compare two ``AnnotatedCell`` instances (e.g. before/after a parameter
change) and report the structural delta:

* cell name changes,
* device additions / removals,
* ``SymbolicLayout`` field changes (``finger_count`` / ``pitch_multiplier``
  / ``stripe_count``) on devices present in both snapshots,
* annotation set differences per device (added / removed annotations).

The module is read-only with respect to its inputs and never mutates the
``AnnotatedCell`` instances passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .annotations import AnnotatedCell, LayoutAnnotation
from .symbolic_layout import SymbolicLayout


# ---------------------------------------------------------------------------
# Diff record dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceDiff:
    """A single ``SymbolicLayout`` field change for one device."""

    device: str
    field: str  # "finger_count" / "pitch_multiplier" / "stripe_count"
    before: Any
    after: Any


@dataclass(frozen=True)
class AnnotationDiff:
    """An annotation that was added, removed, or modified on a device."""

    device: str
    change_type: str  # "added" / "removed" / "modified"
    annotation_repr: str


@dataclass
class CellSnapshotDiff:
    """Aggregate of all differences between two ``AnnotatedCell`` snapshots."""

    cell_name_change: Optional[Tuple[str, str]] = None
    devices_added: List[str] = field(default_factory=list)
    devices_removed: List[str] = field(default_factory=list)
    device_field_diffs: List[DeviceDiff] = field(default_factory=list)
    annotation_diffs: List[AnnotationDiff] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.cell_name_change
            or self.devices_added
            or self.devices_removed
            or self.device_field_diffs
            or self.annotation_diffs
        )


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


_LAYOUT_FIELDS: Tuple[str, ...] = (
    "finger_count",
    "pitch_multiplier",
    "stripe_count",
)


def _diff_layout(
    device: str,
    before: SymbolicLayout,
    after: SymbolicLayout,
) -> List[DeviceDiff]:
    """Per-field diff between two ``SymbolicLayout`` instances."""
    out: List[DeviceDiff] = []
    for fname in _LAYOUT_FIELDS:
        b = getattr(before, fname)
        a = getattr(after, fname)
        if b != a:
            out.append(DeviceDiff(device=device, field=fname, before=b, after=a))
    return out


def _diff_annotations(
    device: str,
    before: List[LayoutAnnotation],
    after: List[LayoutAnnotation],
) -> List[AnnotationDiff]:
    """Multi-set diff of annotations.

    Annotations are compared by ``repr`` because the underlying annotation
    dataclasses are ``eq=True`` and produce deterministic reprs; this lets
    us handle duplicate annotation instances (multiset semantics) without
    requiring hashability.
    """
    before_reprs = [repr(a) for a in before]
    after_reprs = [repr(a) for a in after]

    # Build multiset counts.
    from collections import Counter

    bc = Counter(before_reprs)
    ac = Counter(after_reprs)

    diffs: List[AnnotationDiff] = []

    # Removed: in before but not (or fewer) in after.
    for r, n in bc.items():
        delta = n - ac.get(r, 0)
        for _ in range(max(delta, 0)):
            diffs.append(
                AnnotationDiff(
                    device=device, change_type="removed", annotation_repr=r
                )
            )

    # Added: in after but not (or fewer) in before.
    for r, n in ac.items():
        delta = n - bc.get(r, 0)
        for _ in range(max(delta, 0)):
            diffs.append(
                AnnotationDiff(
                    device=device, change_type="added", annotation_repr=r
                )
            )

    return diffs


def diff_cells(before: AnnotatedCell, after: AnnotatedCell) -> CellSnapshotDiff:
    """Compare two ``AnnotatedCell`` instances.

    Identifies cell-name changes, device additions/removals, per-device
    ``SymbolicLayout`` field changes, and per-device annotation set
    differences. The returned ``CellSnapshotDiff`` carries empty lists
    (and ``cell_name_change=None``) when the snapshots are equivalent.
    """
    result = CellSnapshotDiff()

    if before.cell_name != after.cell_name:
        result.cell_name_change = (before.cell_name, after.cell_name)

    before_devices = set(before.layouts.keys())
    after_devices = set(after.layouts.keys())

    result.devices_added = sorted(after_devices - before_devices)
    result.devices_removed = sorted(before_devices - after_devices)

    # Per-device layout field diffs (common devices only).
    for dev in sorted(before_devices & after_devices):
        result.device_field_diffs.extend(
            _diff_layout(dev, before.layouts[dev], after.layouts[dev])
        )

    # Per-device annotation diffs across the union of devices that carry
    # annotations on either side, intersected with devices still present
    # on at least one side. Added/removed devices contribute their
    # annotations as added/removed too.
    annotated_devices = (
        set(before.annotations.keys()) | set(after.annotations.keys())
    )
    for dev in sorted(annotated_devices):
        b_anns = before.annotations.get(dev, [])
        a_anns = after.annotations.get(dev, [])
        result.annotation_diffs.extend(_diff_annotations(dev, b_anns, a_anns))

    return result


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def summarize_diff(diff: CellSnapshotDiff) -> str:
    """One-line summary of a ``CellSnapshotDiff``; ``'No changes'`` if empty."""
    if not diff.has_changes:
        return "No changes"

    parts: List[str] = []
    if diff.cell_name_change is not None:
        b, a = diff.cell_name_change
        parts.append(f"cell renamed {b}->{a}")
    if diff.devices_added:
        parts.append(f"+{len(diff.devices_added)} device(s)")
    if diff.devices_removed:
        parts.append(f"-{len(diff.devices_removed)} device(s)")
    if diff.device_field_diffs:
        parts.append(f"{len(diff.device_field_diffs)} layout field change(s)")
    if diff.annotation_diffs:
        added = sum(
            1 for d in diff.annotation_diffs if d.change_type == "added"
        )
        removed = sum(
            1 for d in diff.annotation_diffs if d.change_type == "removed"
        )
        modified = sum(
            1 for d in diff.annotation_diffs if d.change_type == "modified"
        )
        ann_bits = []
        if added:
            ann_bits.append(f"+{added}")
        if removed:
            ann_bits.append(f"-{removed}")
        if modified:
            ann_bits.append(f"~{modified}")
        parts.append("annotations " + "/".join(ann_bits))

    return "; ".join(parts)
