"""Measure a declared centroid over a fully replayed, source-owned row.

The caller supplies its authenticated canonical declaration/compiler entry and
retained materialization. This geometric operation does not mint source or
placement authority. Occurrence order and physical channel metrics remain
separate, including when a source unit contains several channels.
"""
from __future__ import annotations

import klayout.db as kdb

from clay.layout.occurrence_channels import replay_row_channel_inventory
from clay.layout.place.row_calculus import RowAtomKind
from clay.layout.place.row_program import _atom_owner


_ORIENTATION_MATRICES = {
    'R0': (1, 0, 0, 1), 'R90': (0, -1, 1, 0),
    'R180': (-1, 0, 0, -1), 'R270': (0, 1, -1, 0),
    'MR0': (1, 0, 0, -1), 'MR90': (0, 1, 1, 0),
    'MR180': (-1, 0, 0, 1), 'MR270': (0, -1, -1, 0),
}


def _compose_channel_orientation(parent, member):
    if parent not in _ORIENTATION_MATRICES or member not in _ORIENTATION_MATRICES:
        raise ValueError('physical channel orientation lacks canonical provenance')
    a, b, c, d = _ORIENTATION_MATRICES[parent]
    e, f, g, h = _ORIENTATION_MATRICES[member]
    result = (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)
    return next(name for name, matrix in _ORIENTATION_MATRICES.items() if matrix == result)


def _continuous_array_channel_orientation(*, geometry, entry, finger):
    """Resolve the canonical continuous-array inheritance, never a fallback.

    This is called only after the independently rebuilt macro's complete
    artwork has matched its source certificate and the reread exact GDS.
    Fixed-PCell occurrences use their separate, explicit transform instead.
    """
    from clay.layout.common_centroid_array import CommonCentroidArray, _CCFinger

    if (type(geometry) is not CommonCentroidArray or type(finger) is not _CCFinger
            or finger.orientation != ''
            or getattr(geometry, 'channel_inventory', None) is not None):
        raise ValueError('channel orientation lacks canonical continuous-array inheritance')
    inherited = getattr(geometry, 'materialized_orientation', None)
    if inherited not in ('R0', 'R90', 'R180', 'R270') or inherited != entry.orientation:
        raise ValueError('continuous-array orientation differs from canonical row materialization')
    return inherited


def measure_owned_centroid_channels(*, declaration, canonical_entry, entry,
                                    geometry, dx_nm, dy_nm, emitted_od, emitted_poly,
                                    member_channel_geometry=None):
    if member_channel_geometry is not None and (
            type(member_channel_geometry) is not list or member_channel_geometry):
        raise ValueError('channel geometry output must be a fresh empty list')
    inventory = replay_row_channel_inventory(geometry=geometry, consumed_devices=entry.consumed_devices)
    if entry.dummy_devices or entry.macro.consumed_dummy_endcaps:
        raise ValueError('typed centroid channels require explicit dummy tie semantics')
    members = tuple(canonical_entry.active_devices)
    if len(members) != 2 or not set(members) <= set(entry.active_devices):
        raise ValueError('declared centroid lacks complete enclosing source ownership')

    # Use signed owned OD, never an aggregate AABB: a declared cavity can hold
    # another source stamp whose channels do not belong to this row.
    owned_od = kdb.Region()
    for occurrence in inventory.occurrences:
        for key, polygons in occurrence.owned_mask_polygons:
            if key in inventory.od_layer_keys:
                for polygon in polygons:
                    owned_od += polygon.translated(dx_nm, dy_nm).to_region()
    owned_od.merge()
    actual = tuple(kdb.Region(polygon) for polygon in (emitted_od & emitted_poly).merged().each()
                   if not (kdb.Region(polygon) & owned_od).is_empty())
    used = set()
    observed = []
    measured_geometry = []
    for occurrence, finger in zip(inventory.occurrences, geometry.fingers):
        for channel in occurrence.channels:
            placed = channel.translated(dx_nm, dy_nm).to_region()
            matches = [index for index, region in enumerate(actual) if (region ^ placed).is_empty()]
            if len(matches) != 1 or matches[0] in used:
                raise ValueError('exact GDS does not uniquely realize every owned row channel')
            index = matches[0]
            used.add(index)
            if occurrence.logical_owner in members:
                box = actual[index].bbox()
                observed.append((occurrence.logical_owner, int(box.left + box.right), int(box.bottom + box.top)))
                measured_geometry.append((occurrence.logical_owner,
                    (int(box.left), int(box.bottom), int(box.right), int(box.top)),
                    _compose_channel_orientation(entry.orientation, finger.orientation)))
    if len(used) != len(actual):
        raise ValueError('exact GDS contains unowned channels on the row-owned OD')
    if not observed or {owner for owner, _x, _y in observed} != set(members):
        raise ValueError('declared centroid channel population is incomplete')

    # Order the source occurrences in the declaration's macro frame. Internal
    # channel bands must not increase the declared row or interleave counts.
    rows = {}
    for occurrence, finger in zip(inventory.occurrences, geometry.fingers):
        if occurrence.logical_owner not in members:
            continue
        if any(type(value) is not int for value in (finger.x0, finger.x1, finger.cy)):
            raise ValueError('centroid occurrence frame requires exact integer coordinates')
        x2, y2 = finger.x0 + finger.x1 + 2 * dx_nm, 2 * (finger.cy + dy_nm)
        row_key, order_key = ((y2, x2) if declaration.orientation in ('R0', 'R180') else (x2, y2))
        rows.setdefault(row_key, []).append((order_key, occurrence.logical_owner, finger.orientation))
    ordered_rows = tuple(tuple(sorted(rows[key], reverse=declaration.orientation in ('R180', 'R270')))
                         for key in sorted(rows, reverse=declaration.orientation in ('R90', 'R180')))
    if any(len({item[0] for item in row}) != len(row) for row in ordered_rows):
        raise ValueError('centroid source occurrence positions are ambiguous')
    owner_rows = tuple(tuple(owner for _rank, owner, _orientation in row) for row in ordered_rows)
    expected_rows = []
    for row in canonical_entry.macro.rows:
        owners = []
        for atom in row.atoms:
            if atom.kind is RowAtomKind.ACTIVE:
                owner = _atom_owner(atom, canonical_entry.active_devices)
                owners.extend((owner,) * (2 if ':pair:' in f':{atom.atom_id}:' else 1))
        expected_rows.append(tuple(owners))
    if len(owner_rows) != declaration.rows or owner_rows != tuple(expected_rows):
        raise ValueError('centroid occurrence rows/order differ from canonical declaration')

    if declaration.member_transforms:
        matrices = {
            'R0': (1, 0, 0, 1), 'R90': (0, -1, 1, 0),
            'R180': (-1, 0, 0, -1), 'R270': (0, 1, -1, 0),
            'MR0': (1, 0, 0, -1), 'MR90': (0, 1, 1, 0),
            'MR180': (-1, 0, 0, 1), 'MR270': (0, -1, -1, 0),
        }
        def compose(left, right):
            a, b, c, d = left
            e, f, g, h = right
            return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)
        left_transforms = dict(zip(members, declaration.member_transforms))
        for row in ordered_rows:
            if len(row) % 2:
                raise ValueError('centroid occurrence transform halves are incomplete')
            for index, (_rank, owner, actual_transform) in enumerate(row):
                expected = matrices[left_transforms[owner]]
                if index >= len(row) // 2:
                    a, b, c, d = expected
                    expected = (-a, -b, c, d)
                if actual_transform not in matrices or compose(matrices[entry.orientation], matrices[actual_transform]) != compose(
                        matrices[declaration.orientation], expected):
                    raise ValueError('centroid occurrence transform differs from canonical declaration')
    if member_channel_geometry is not None:
        member_channel_geometry.extend(measured_geometry)
    return tuple(observed), owner_rows
