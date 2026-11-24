"""Current-frame integration of layered physical accessories and source owners.

Each operation replays its current inputs. This module retains no graph result,
placement verdict or authority across materialization boundaries.
"""
from __future__ import annotations


class InvalidCompactionGroupOwnership(ValueError):
    """Canonical source groups cannot be used as physical ownership data."""


def compaction_source_groups(cell_def):
    """Resolve and copy complete source group ownership before ring projection."""
    from clay.frontend.craft_infer import group_members
    raw = group_members(cell_def)
    if type(raw) is not dict:
        raise InvalidCompactionGroupOwnership('canonical craft group resolver returned a foreign mapping')
    result = {}
    for name, members in raw.items():
        if (type(name) is not str or not name or type(members) is not list
                or any(type(member) is not str or not member for member in members)
                or len(set(members)) != len(members)):
            raise InvalidCompactionGroupOwnership('canonical craft group resolver returned malformed ownership')
        result[name] = tuple(members)
    return result


def restore_layered_compaction_inputs(bindings, *, source_cell, tech, placement,
        geometry, placement_um, loose_size_um):
    """Restore actual tap copies before the native solver samples its envelope."""
    if not bindings and not any(getattr(stamp, 'layered_placement_tap_child', None) is not None
                                for stamp in geometry.values()):
        return geometry, placement_um, bindings, *loose_size_um
    from clay.layout.layered_placement_assembly import restore_layered_pmos_assemblies
    from clay.layout.compact_blocks import _envelope
    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm
    restored = restore_layered_pmos_assemblies(bindings, source_cell=source_cell, tech=tech,
        row_program=getattr(placement, 'row_program', None),
        row_materialization=getattr(placement, 'row_materialization', None),
        geometry=geometry, placement_um=placement_um)
    width, height, _ = _envelope(_placed_stamp_bboxes_nm(restored.geometry, restored.placement_um))
    return restored.geometry, restored.placement_um, restored.assemblies, width, height


def final_layered_compaction_view(certificate, *, bindings, source_cell, tech, placement,
        intent, geometry, placement_um, ring_names):
    """Replay the full actual scene, then project the existing ring-free view."""
    if bindings or any(getattr(stamp, 'layered_placement_tap_child', None) is not None
                       for stamp in geometry.values()):
        from clay.layout.layered_placement_assembly import require_layered_placement_certificate
        require_layered_placement_certificate(certificate, source_cell=source_cell, tech=tech,
            row_program=getattr(placement, 'row_program', None),
            row_materialization=getattr(placement, 'row_materialization', None),
            intent=intent, geometry=geometry, placement_um=placement_um)
    return ({name: value for name, value in placement_um.items() if name not in ring_names},
            {name: value for name, value in geometry.items() if name not in ring_names})


def materialized_source_snapshot(*, required, blocks, stamps, geometry, placement_um,
        canonical_device_names):
    """Capture the full logical partition and artwork in one current frame.

    Physical TAP children remain in block ownership; only canonical source names
    enter the logical denominator. A supplied source inventory always owns its
    final snapshot, independent of later routing or intent-fact consumers. The
    legacy required flag also rejects missing inventory when a consumer needs it.
    """
    if not required and not canonical_device_names:
        return (), ()
    from clay.layout.place.constraints_well_guard import _resolve_block_member_devices
    from clay.layout.place.placement_geometry import (
        _canonical_materialized_block_source_members, _source_bbox_refinement_rows,
    )
    ownership = _canonical_materialized_block_source_members(blocks,
        _resolve_block_member_devices(stamps, geometry), canonical_device_names)
    if tuple(sorted(name for _block, names in ownership for name in names)) != canonical_device_names:
        raise ValueError('materialized source ownership does not cover the complete logical inventory')
    artwork = _source_bbox_refinement_rows(geometry, placement_um, ownership) if ownership else ()
    return ownership, artwork
