"""Cell extension / inheritance helpers.

Provides ``clone_cell``, ``extend_cell``, and ``merge_devices`` for
deriving new :class:`AnnotatedCell` instances from existing bases by
deep-cloning, appending new devices, and merging annotations.

``LayoutAnnotation`` subclasses are treated as effectively immutable
value objects: they are copied by reference. ``SymbolicLayout`` is
mutable, so each layout is copied by value.
"""

from copy import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .annotations import AnnotatedCell
from .symbolic_layout import SymbolicLayout


@dataclass(frozen=True)
class DeviceAddition:
    """Specification for a new device to graft onto a base cell."""

    device_name: str
    layout: SymbolicLayout
    annotations: List[Any] = field(default_factory=list)


def _copy_layout(layout: SymbolicLayout) -> SymbolicLayout:
    """Value-copy of a SymbolicLayout (Fraction is immutable)."""
    return SymbolicLayout(
        finger_count=layout.finger_count,
        pitch_multiplier=layout.pitch_multiplier,
        stripe_count=layout.stripe_count,
    )


def clone_cell(base: AnnotatedCell, new_cell_name: str) -> AnnotatedCell:
    """Deep-clone ``base`` under ``new_cell_name``.

    Layouts are copied by value (since SymbolicLayout is mutable).
    Annotations are copied by reference (each LayoutAnnotation is
    treated as an immutable value object), but the per-device list
    container is freshly constructed so the clone can be mutated
    independently of the original.
    """
    new_layouts = {
        name: _copy_layout(layout) for name, layout in base.layouts.items()
    }
    new_annotations = {
        device: list(ann_list) for device, ann_list in base.annotations.items()
    }
    return AnnotatedCell(
        cell_name=new_cell_name,
        layouts=new_layouts,
        annotations=new_annotations,
    )


def extend_cell(
    base: AnnotatedCell,
    new_cell_name: str,
    add_devices: List[DeviceAddition],
    extra_annotations: Dict[str, List[Any]] | None = None,
) -> AnnotatedCell:
    """Derive a new cell from ``base`` by adding devices + merging annotations.

    Parameters
    ----------
    base:
        Source cell. Not mutated.
    new_cell_name:
        Name of the derived cell.
    add_devices:
        New devices to graft. Each ``DeviceAddition`` becomes a fresh
        device on the derived cell with its associated annotations.
    extra_annotations:
        Optional mapping of existing device name -> list of additional
        annotations to merge onto that device.

    Raises
    ------
    ValueError
        If a ``DeviceAddition`` references a device name already present
        in ``base``.
    """
    derived = clone_cell(base, new_cell_name)

    for addition in add_devices:
        if derived.has_device(addition.device_name):
            raise ValueError(
                f"device {addition.device_name!r} already exists in base cell "
                f"{base.cell_name!r}; cannot extend"
            )
        derived.set_layout(addition.device_name, _copy_layout(addition.layout))
        for ann in addition.annotations:
            derived.add_annotation(addition.device_name, ann)

    if extra_annotations:
        for device, ann_list in extra_annotations.items():
            for ann in ann_list:
                derived.add_annotation(device, ann)

    return derived


def merge_devices(
    a: AnnotatedCell,
    b: AnnotatedCell,
    merged_cell_name: str,
) -> AnnotatedCell:
    """Combine two cells' device sets under ``merged_cell_name``.

    Devices from ``b`` are added on top of a clone of ``a``. The two
    device sets must be disjoint.

    Raises
    ------
    ValueError
        On any device name collision between ``a`` and ``b``.
    """
    collisions = set(a.layouts) & set(b.layouts)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(
            f"cannot merge cells {a.cell_name!r} and {b.cell_name!r}: "
            f"device name collision on {{{names}}}"
        )

    merged = clone_cell(a, merged_cell_name)
    for device, layout in b.layouts.items():
        merged.set_layout(device, _copy_layout(layout))
    for device, ann_list in b.annotations.items():
        for ann in ann_list:
            merged.add_annotation(device, ann)
    return merged
