"""Template equivalence checker for ``AnnotatedCell`` instances.

This module provides a lightweight *structural* equivalence check between two
:class:`~clay.layout.annotations.AnnotatedCell` instances. The intent is to
answer the question "are these two cells the same layout up to renaming of
devices?" without performing a full graph isomorphism on the underlying
schematic.

The check is deliberately bulk / multiset-based:

* Two cells must have the same number of devices.
* The multiset of :class:`~clay.layout.symbolic_layout.SymbolicLayout`
  *signatures* (a hashable tuple of layout co-variables) must match.
* The multiset of *annotation kinds* (by class name) must match.

This is enough to detect "same template, different instance names" while
remaining far cheaper than full topology matching. It is **not** sufficient
for verifying schematic correctness; for that, use the LVS / topology layers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Tuple

from .annotations import AnnotatedCell
from .symbolic_layout import SymbolicLayout


@dataclass(frozen=True)
class EquivalenceResult:
    """Outcome of a structural equivalence check.

    Attributes
    ----------
    is_equivalent:
        ``True`` if cells are structurally equivalent up to device renaming.
    reason:
        Human readable explanation. Empty when ``is_equivalent`` is ``True``;
        populated with the first failure description otherwise.
    device_count_a:
        Number of devices (layouts) in the first cell.
    device_count_b:
        Number of devices (layouts) in the second cell.
    matched_layout_signatures:
        Count of devices whose :func:`signature_of_device` signature appears
        in the *intersection* of the two cells' signature multisets. When
        ``is_equivalent`` is ``True`` this equals ``device_count_a``.
    """

    is_equivalent: bool
    reason: str
    device_count_a: int
    device_count_b: int
    matched_layout_signatures: int


def signature_of_device(layout: SymbolicLayout) -> Tuple:
    """Return a hashable signature for a ``SymbolicLayout``.

    The signature is the tuple
    ``(finger_count, pitch_multiplier_num, pitch_multiplier_den, stripe_count)``.
    Two layouts with the same signature are interchangeable for the purpose
    of bulk structural comparison.
    """
    pm = layout.pitch_multiplier
    return (
        int(layout.finger_count),
        int(pm.numerator),
        int(pm.denominator),
        int(layout.stripe_count),
    )


def signature_of_cell(cell: AnnotatedCell) -> Counter:
    """Return a multiset of device signatures for a cell.

    The resulting :class:`collections.Counter` maps each signature (as
    returned by :func:`signature_of_device`) to the number of devices that
    carry that signature. Device names are intentionally discarded so the
    multiset is rename-invariant.
    """
    counter: Counter = Counter()
    for layout in cell.layouts.values():
        counter[signature_of_device(layout)] += 1
    return counter


def annotation_signature_multiset(cell: AnnotatedCell) -> Counter:
    """Return a multiset of annotation kinds (by class name).

    Iterates every annotation attached to every device and counts the bare
    Python class name of each annotation (e.g. ``"Matching"``,
    ``"GuardRing"``). The owning device is intentionally ignored so the
    result is rename-invariant.
    """
    counter: Counter = Counter()
    for ann_list in cell.annotations.values():
        for ann in ann_list:
            counter[type(ann).__name__] += 1
    return counter


def cells_equivalent(a: AnnotatedCell, b: AnnotatedCell) -> EquivalenceResult:
    """Test two cells for structural equivalence up to device renaming.

    The cells are considered equivalent when all three hold:

    1. They have the same number of devices.
    2. Their :func:`signature_of_cell` multisets are equal.
    3. Their :func:`annotation_signature_multiset` multisets are equal.

    This is *not* a full graph isomorphism. Two cells may share device-shape
    and annotation-kind multisets while wiring devices together differently.
    The check is intended as a fast template-identity probe.
    """
    count_a = len(a.layouts)
    count_b = len(b.layouts)

    sig_a = signature_of_cell(a)
    sig_b = signature_of_cell(b)

    # matched_layout_signatures = total devices in A whose signature also
    # appears in B at least as many times as in A. Using the multiset
    # intersection gives a symmetric, useful "how much overlapped" number.
    intersection = sig_a & sig_b
    matched = sum(intersection.values())

    if count_a != count_b:
        return EquivalenceResult(
            is_equivalent=False,
            reason=(
                f"device count mismatch: A has {count_a}, B has {count_b}"
            ),
            device_count_a=count_a,
            device_count_b=count_b,
            matched_layout_signatures=matched,
        )

    if sig_a != sig_b:
        return EquivalenceResult(
            is_equivalent=False,
            reason=(
                "device layout signature multiset mismatch: "
                f"A={dict(sig_a)} vs B={dict(sig_b)}"
            ),
            device_count_a=count_a,
            device_count_b=count_b,
            matched_layout_signatures=matched,
        )

    ann_a = annotation_signature_multiset(a)
    ann_b = annotation_signature_multiset(b)
    if ann_a != ann_b:
        return EquivalenceResult(
            is_equivalent=False,
            reason=(
                "annotation kind multiset mismatch: "
                f"A={dict(ann_a)} vs B={dict(ann_b)}"
            ),
            device_count_a=count_a,
            device_count_b=count_b,
            matched_layout_signatures=matched,
        )

    return EquivalenceResult(
        is_equivalent=True,
        reason="",
        device_count_a=count_a,
        device_count_b=count_b,
        matched_layout_signatures=matched,
    )
