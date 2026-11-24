from dataclasses import dataclass
from typing import List
from fractions import Fraction

from .annotations import (
    AnnotatedCell, LayoutAnnotation,
    Matching, GuardRing, WellIsolated, Symmetric, Interdigitated, Adjacent,
)


class SpatialTypeError:
    """Base class for structured spatial type errors."""
    pass


@dataclass
class MatchingFingerCountMismatch(SpatialTypeError):
    device: str
    other: str
    device_finger_count: int
    other_finger_count: int


@dataclass
class SymmetricOddDeviceCount(SpatialTypeError):
    device: str
    count: int


@dataclass
class WellTypeConflict(SpatialTypeError):
    device: str
    conflicting_device: str


@dataclass
class UnknownDeviceReference(SpatialTypeError):
    from_device: str
    referenced: str


@dataclass
class InterdigitatedZeroFingerPairs(SpatialTypeError):
    device: str
    with_device: str


@dataclass
class AdjacentNonPositiveSeparation(SpatialTypeError):
    device: str
    to_device: str


def validate(cell: AnnotatedCell) -> List[SpatialTypeError]:
    """Return list of SpatialTypeError; empty list means valid.

    Iterates devices in cell.annotations in stable (sorted) order so error list
    ordering is deterministic across runs.
    """
    errors: List[SpatialTypeError] = []

    for device in sorted(cell.annotations.keys()):
        anns = cell.annotations[device]
        for ann in anns:
            if isinstance(ann, Matching):
                if not cell.has_device(ann.other):
                    errors.append(UnknownDeviceReference(from_device=device, referenced=ann.other))
                else:
                    self_fc = cell.layouts[device].finger_count
                    other_fc = cell.layouts[ann.other].finger_count
                    if self_fc != other_fc:
                        errors.append(MatchingFingerCountMismatch(
                            device=device, other=ann.other,
                            device_finger_count=self_fc, other_finger_count=other_fc))

            elif isinstance(ann, GuardRing):
                # GuardRing self-only; conflict logic below covers cross-device
                pass

            elif isinstance(ann, WellIsolated):
                if not cell.has_device(ann.from_device):
                    errors.append(UnknownDeviceReference(from_device=device, referenced=ann.from_device))
                else:
                    # Check well type conflict
                    self_gr = [a for a in cell.annotations.get(device, []) if isinstance(a, GuardRing)]
                    other_gr = [a for a in cell.annotations.get(ann.from_device, []) if isinstance(a, GuardRing)]
                    if self_gr and other_gr:
                        if self_gr[0].well_type == other_gr[0].well_type:
                            errors.append(WellTypeConflict(
                                device=device, conflicting_device=ann.from_device))

            elif isinstance(ann, Symmetric):
                if len(ann.devices) % 2 != 0:
                    errors.append(SymmetricOddDeviceCount(device=device, count=len(ann.devices)))
                for d in ann.devices:
                    if not cell.has_device(d):
                        errors.append(UnknownDeviceReference(from_device=device, referenced=d))

            elif isinstance(ann, Interdigitated):
                if not cell.has_device(ann.with_device):
                    errors.append(UnknownDeviceReference(from_device=device, referenced=ann.with_device))
                if ann.finger_pairs < 1:
                    errors.append(InterdigitatedZeroFingerPairs(device=device, with_device=ann.with_device))

            elif isinstance(ann, Adjacent):
                if not cell.has_device(ann.to):
                    errors.append(UnknownDeviceReference(from_device=device, referenced=ann.to))
                if ann.max_separation_multiplier <= 0:
                    errors.append(AdjacentNonPositiveSeparation(device=device, to_device=ann.to))

    return errors
