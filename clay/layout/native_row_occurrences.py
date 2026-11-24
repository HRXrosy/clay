"""Transport validated native-row fingers into the final stamp's local frame."""

from __future__ import annotations

from typing import Any


_ROTATIONS = frozenset(("R0", "R90", "R180", "R270"))
_ORIENTATIONS = _ROTATIONS | frozenset(("MR0", "MR90", "MR180", "MR270"))


def _member(value: object, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if value != value.strip() or (not value and not allow_empty):
        raise ValueError(f"{label} must be canonical member text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer in nm")
    return value


def native_row_occurrences_by_member(
    array: Any,
    *,
    consumed_devices: tuple[str, ...],
    native_origin_nm: tuple[int, int],
) -> tuple[tuple[str, tuple[tuple[float, float, str], ...]], ...]:
    """Keep every actual finger center and orientation, grouped by source owner.

    The row-program materializer already validates the full finger-owner sequence
    against its active and dummy devices. Older shared-OD geometry leaves dummy
    ``member`` fields empty; restore them only when the consumed inventory has
    exactly one unnamed owner. This is transport of that validated inventory,
    not independent proof that a missing source member is electrically a dummy.

    Finger coordinates are already oriented native nm. Subtract the actual stamp
    origin once and convert to local um; never rotate or reconstruct the geometry.
    """
    if type(consumed_devices) is not tuple or not consumed_devices:
        raise ValueError("consumed_devices must be a nonempty exact tuple")
    consumed = tuple(_member(value, "consumed member") for value in consumed_devices)
    if len(set(consumed)) != len(consumed):
        raise ValueError("consumed_devices must not contain duplicate members")
    if type(native_origin_nm) is not tuple or len(native_origin_nm) != 2:
        raise ValueError("native_origin_nm must be an exact 2-tuple")
    origin_x, origin_y = (
        _integer(value, "native origin") for value in native_origin_nm
    )

    fingers = getattr(array, "fingers", None)
    if type(fingers) not in (tuple, list) or not fingers:
        raise ValueError("native row must contain an explicit nonempty finger inventory")
    owners = tuple(
        _member(getattr(finger, "member", None), "finger member", allow_empty=True)
        for finger in fingers
    )
    named = set(owners) - {""}
    foreign = named - set(consumed)
    if foreign:
        raise ValueError(f"native row has foreign finger members: {sorted(foreign)!r}")
    missing = set(consumed) - named
    anonymous_owner = ""
    if "" in owners:
        if len(missing) != 1:
            raise ValueError(
                "anonymous fingers require exactly one consumed member absent from named fingers"
            )
        anonymous_owner = next(iter(missing))
    elif missing:
        raise ValueError(f"native row is missing consumed members: {sorted(missing)!r}")

    orientation = getattr(array, "materialized_orientation", "R0")
    if type(orientation) is not str or orientation not in _ROTATIONS:
        raise ValueError("native row materialized orientation is unavailable")
    bbox = getattr(array, "comp_bbox", None)
    if type(bbox) is not tuple or len(bbox) != 4:
        raise ValueError("native row comp_bbox must be an exact 4-tuple")
    x0, y0, x1, y1 = (_integer(value, "native row bbox") for value in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("native row comp_bbox must have positive extent")
    default_cy = (y0 + y1) // 2
    by_member: dict[str, list[tuple[float, float, str]]] = {}
    for finger, owner in zip(fingers, owners):
        finger_x0 = _integer(getattr(finger, "x0", None), "finger x0")
        finger_x1 = _integer(getattr(finger, "x1", None), "finger x1")
        # A quarter turn leaves the two x probes equal; that is a valid center.
        if finger_x1 < finger_x0:
            raise ValueError("native row finger x1 must not precede x0")
        cy = getattr(finger, "cy", None)
        cy = default_cy if cy is None else _integer(cy, "finger cy")
        finger_orientation = getattr(finger, "orientation", "")
        if type(finger_orientation) is not str:
            raise TypeError("finger orientation must be an exact string")
        finger_orientation = finger_orientation or orientation
        if finger_orientation not in _ORIENTATIONS:
            raise ValueError("native row finger orientation is unavailable")
        by_member.setdefault(owner or anonymous_owner, []).append((
            (((finger_x0 + finger_x1) // 2) - origin_x) / 1000,
            (cy - origin_y) / 1000,
            finger_orientation,
        ))
    if set(by_member) != set(consumed):
        raise ValueError("native row occurrences must cover every consumed member")
    return tuple((member, tuple(sorted(rows))) for member, rows in sorted(by_member.items()))
