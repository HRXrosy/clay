"""Exact in-process snapshots for prepared compact placement diagnostics."""
from clay.layout.place.current_structure_operands import _native_values, _placed_values, _row_values
from clay.layout.place.final_placement_state import final_placement_state_payload
from clay.layout.place.row_replay_snapshot import freeze_row_values


def _scene(value):
    from clay.layout.route.route_cell import _PlacedStamp, _PlacedBjtStamp, _PlacedPassiveStamp
    if type(value) is not tuple or len(value) != 2 or type(value[0]) is not dict or type(value[1]) is not dict:
        raise TypeError('compact scene requires actual placement and geometry dictionaries')
    positions, geometry = value
    if set(geometry) - set(positions):
        raise ValueError('compact scene has an unplaced stamp')
    stamps = []
    for name, stamp in sorted(geometry.items()):
        if type(stamp) in (_PlacedStamp, _PlacedBjtStamp, _PlacedPassiveStamp):
            raw = {slot: getattr(stamp, slot) for cls in type(stamp).__mro__
                   for slot in cls.__dict__.get('__slots__', ()) if hasattr(stamp, slot)}
            fields = _native_values(raw)
        else:
            fields = _native_values(stamp)
        stamps.append((name, type(stamp), id(stamp), fields))
    return freeze_row_values(positions), tuple(stamps)


def compact_value(compact):
    if compact is None:
        return None
    return (_scene(compact.override), _scene(compact.ring_inclusive),
        freeze_row_values((compact.member_ring_pad_top_nm, compact.member_ring_pad_bottom_nm,
        compact.n_rings, compact.n_declared_rings, compact.n_source_guard_ring_declarations,
        compact.shared_well_guard_envelope_plans, compact.shared_well_guard_ring_inventory,
        compact.shared_well_guard_child_plans, compact.shared_well_flexible_content_bboxes_nm)),
        _placed_values(compact.logical_placement_basis), _row_values(compact.logical_placement_basis),
        freeze_row_values(final_placement_state_payload(compact.final_placement_certificate)))
