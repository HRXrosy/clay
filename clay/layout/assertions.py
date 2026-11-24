"""Declarative layout assertion language for ``AnnotatedCell``.

This module exposes a small library of pure-Python predicates that examine
an :class:`~clay.layout.annotations.AnnotatedCell` and return an
:class:`AssertionReport`. Reports compose: ``run_assertions`` merges a list
of independently constructed reports into a single one.

The assertions are deliberately structural and PDK-agnostic; they read
only :class:`~clay.layout.symbolic_layout.SymbolicLayout` integer
co-variables and the typed :class:`~clay.layout.annotations.LayoutAnnotation`
subclasses already attached to the cell. The API is read-only with respect
to ``AnnotatedCell`` -- no mutation is performed.
"""

from dataclasses import dataclass, field
from typing import Callable, List

from .annotations import AnnotatedCell, LayoutAnnotation, Matching


@dataclass(frozen=True)
class AssertionError_:
    """One failure observed while checking an assertion rule."""

    rule_id: str
    message: str


@dataclass
class AssertionReport:
    """Outcome of one or more assertion checks.

    ``passed`` is True iff ``errors`` is empty. When multiple reports are
    merged via :func:`run_assertions`, ``passed`` is recomputed from the
    accumulated error list.
    """

    passed: bool
    errors: List[AssertionError_] = field(default_factory=list)


def _ok(rule_id: str) -> AssertionReport:
    return AssertionReport(passed=True, errors=[])


def _fail(rule_id: str, message: str) -> AssertionReport:
    return AssertionReport(
        passed=False,
        errors=[AssertionError_(rule_id=rule_id, message=message)],
    )


def assert_device_count(
    cell: AnnotatedCell, expected: int, rule_id: str = "device_count"
) -> AssertionReport:
    """Assert the cell has exactly ``expected`` devices (layouts)."""
    actual = len(cell.layouts)
    if actual == expected:
        return _ok(rule_id)
    return _fail(
        rule_id,
        f"expected {expected} devices in cell '{cell.cell_name}', got {actual}",
    )


def assert_min_finger_count(
    cell: AnnotatedCell,
    device: str,
    min_n: int,
    rule_id: str = "min_finger",
) -> AssertionReport:
    """Assert ``device`` has a layout with at least ``min_n`` fingers."""
    if not cell.has_device(device):
        return _fail(
            rule_id,
            f"device '{device}' has no layout in cell '{cell.cell_name}'",
        )
    layout = cell.layouts[device]
    if layout.finger_count < min_n:
        return _fail(
            rule_id,
            f"device '{device}' finger_count={layout.finger_count} < {min_n}",
        )
    return _ok(rule_id)


def assert_annotation_present(
    cell: AnnotatedCell,
    device: str,
    annotation_type: type,
    rule_id: str = "anno_present",
) -> AssertionReport:
    """Assert ``device`` carries at least one annotation of the given type."""
    if not cell.has_device(device) and device not in cell.annotations:
        return _fail(
            rule_id,
            f"device '{device}' not present in cell '{cell.cell_name}'",
        )
    annos = cell.annotations_for(device)
    for ann in annos:
        if isinstance(ann, annotation_type):
            return _ok(rule_id)
    return _fail(
        rule_id,
        f"device '{device}' missing annotation of type "
        f"{annotation_type.__name__}",
    )


def assert_matching_pair(
    cell: AnnotatedCell,
    dev_a: str,
    dev_b: str,
    rule_id: str = "matching",
) -> AssertionReport:
    """Assert dev_a has a Matching annotation referencing dev_b (or vice versa).

    Symmetric: either direction counts.
    """
    annos_a = cell.annotations_for(dev_a)
    annos_b = cell.annotations_for(dev_b)
    for ann in annos_a:
        if isinstance(ann, Matching) and ann.other == dev_b:
            return _ok(rule_id)
    for ann in annos_b:
        if isinstance(ann, Matching) and ann.other == dev_a:
            return _ok(rule_id)
    return _fail(
        rule_id,
        f"no Matching annotation links '{dev_a}' and '{dev_b}' "
        f"in cell '{cell.cell_name}'",
    )


def run_assertions(
    cell: AnnotatedCell, rules: List[Callable[[AnnotatedCell], AssertionReport]]
) -> AssertionReport:
    """Run each callable rule against ``cell``, merging errors.

    Each callable must accept the cell and return an ``AssertionReport``.
    Typically you build these with ``functools.partial`` or a lambda that
    closes over the per-rule arguments.
    """
    accumulated: List[AssertionError_] = []
    for rule in rules:
        report = rule(cell)
        accumulated.extend(report.errors)
    return AssertionReport(passed=len(accumulated) == 0, errors=accumulated)
