"""Live, source-bound matching evidence for two distinct native rows.

This is placement evidence only.  It never substitutes for exact-GDS signoff.
"""
from dataclasses import dataclass
from weakref import WeakKeyDictionary


@dataclass(frozen=True, eq=False, init=False)
class NativeCrossRowMatchingInventory:
    pairs: tuple
    placement: object
    cell: object
    geometry: object
    placement_um: object
    tech: object
    record: object
    same_row_inventory: object

    def __init__(self, *args, **kwargs):
        raise TypeError('cross-row matching requires live source-bound capture')


_CAPTURED = WeakKeyDictionary()


def _source_pairs(placement, cell):
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place.row_program import _group_map
    from clay.layout.route.exact_artifact_evidence import _replayed_source_cell_authority
    span = getattr(cell, 'source_span', None)
    native = any(getattr(d, 'device_class', None) == 'native'
                 for d in getattr(cell, 'devices', ()))
    has_file = isinstance(span, dict) and bool(span.get('file'))
    projection = getattr(getattr(placement, 'row_materialization', None), 'source_projection', None)
    placement_native = (getattr(placement, 'native_geometry', None) is not None
        or getattr(placement, 'native_cross_row_cp_proof', None) is not None
        or getattr(placement, 'native_cross_row_final_proof', None) is not None
        or getattr(projection, 'native_witness', None) is not None)
    if not has_file and placement_native:
        raise ValueError('cross-row matching native placement lacks source authority')
    if has_file and not native and not placement_native:
        # A caller can mutate every in-memory device class. Inspect the real
        # source before deciding that the native authority is irrelevant, but
        # preserve ordinary nonnative caller semantics when it is nonnative.
        from clay.parser import parse_clay_file
        parsed = parse_clay_file(span['file'])
        matches = [item for item in parsed.cell_defs if item.local_name == cell.local_name]
        if len(matches) != 1:
            raise ValueError('cross-row matching source cell is not unique')
        native = any(getattr(d, 'device_class', None) == 'native' for d in matches[0].devices)
    if not native and not placement_native:
        return ()
    canonical = _replayed_source_cell_authority(cell)[3] if has_file else cell
    intent = layout_intent_from_cell_def(canonical)
    groups = _group_map(canonical)
    rows = {row.group: set(groups.get(row.group, ())) for row in intent.arrays
            if row.style == 'native_diffusion'}
    result = []
    for match in intent.matching_intents:
        pair = tuple(sorted(match.members))
        owners = [tuple(name for name, members in rows.items() if member in members)
                  for member in pair]
        if len(pair) != 2 or any(len(names) > 1 for names in owners):
            raise ValueError('cross-row matching source ownership is ambiguous')
        if all(len(names) == 1 for names in owners) and owners[0] != owners[1]:
            result.append(pair)
    return tuple(sorted(set(result)))


def _replay(placement, cell, geometry, placement_um, tech, pairs):
    from clay.layout.generic_placement import verify_row_source_compiler_authority, verify_stored_row_projection
    from clay.layout.place.cross_row_matching import (native_cross_row_inventory_from_projection,
        native_cross_row_stamp_centres, replay_native_cross_row_final_proof)
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.exact_artifact_evidence import _replayed_source_cell_authority
    canonical = _replayed_source_cell_authority(cell)[3]
    # The verifier returns a freshly replayed, canonically equal program, not
    # the invocation-owned object accepted by the current row replay scope.
    # Preserve its source verification and pass the original bound program on.
    _replayed_program, materialization = verify_row_source_compiler_authority(placement, canonical, tech)
    program = placement.row_program
    verify_stored_row_projection(placement)
    inventory = native_cross_row_inventory_from_projection(program, materialization, tech)
    if inventory is None or tuple(pair.members for pair in inventory.pairs) != pairs:
        raise ValueError('cross-row matching source pair inventory differs')
    proof = getattr(placement, 'native_cross_row_final_proof', None)
    cp = getattr(placement, 'native_cross_row_cp_proof', None)
    if proof is None or cp is None or proof.cp_proof != cp:
        raise ValueError('cross-row matching final or CP proof is missing')
    directions = {pair: placement.matching_pair_axes[pair][0] for pair in pairs}
    centers = native_cross_row_stamp_centres(inventory, program, materialization, geometry, placement_um)
    axes = replay_native_cross_row_final_proof(inventory, proof, directions, centers,
        pdk_manufacturing_grid_nm(tech), row_program=program,
        row_materialization=materialization, tech=tech)
    if any(placement.matching_pair_axes[pair] != (axes[pair][0], axes[pair][1] / 4.0)
           for pair in pairs):
        raise ValueError('cross-row matching actual pair axis differs from final proof')
    return inventory.sha256, proof.sha256, tuple(sorted(centers.items())), tuple(sorted(axes.items()))


def capture_native_cross_row_matching(placement, cell, *, geometry=None, placement_um=None, tech=None,
                                      same_row_inventory=None):
    pairs = _source_pairs(placement, cell)
    if not pairs:
        return None
    if (geometry is None) != (placement_um is None):
        raise ValueError('cross-row matching requires both actual stamps and positions')
    record = None
    if geometry is not None:
        if tech is None:
            raise ValueError('cross-row matching requires current PDK authority')
        if (getattr(placement, 'row_program', None) is None
                or getattr(placement, 'row_materialization', None) is None):
            raise ValueError('cross-row matching source row witness is missing')
        from clay.layout.place.row_replay_session import native_matching_row_scope
        with native_matching_row_scope(program=placement.row_program,
                materialization=placement.row_materialization, tech=tech):
            record = _replay(placement, cell, geometry, placement_um, tech, pairs)
    result = object.__new__(NativeCrossRowMatchingInventory)
    for name, value in (('pairs', pairs), ('placement', placement), ('cell', cell),
                        ('geometry', geometry), ('placement_um', placement_um),
                        ('tech', tech), ('record', record), ('same_row_inventory', same_row_inventory)):
        object.__setattr__(result, name, value)
    _CAPTURED[result] = (pairs, id(placement), id(cell), id(geometry), id(placement_um), id(tech),
                         record, id(same_row_inventory))
    return result


def require_native_cross_row_matching_inventory(inventory):
    if type(inventory) is not NativeCrossRowMatchingInventory:
        raise ValueError('cross-row matching inventory is missing or foreign')
    expected = (inventory.pairs, id(inventory.placement), id(inventory.cell), id(inventory.geometry),
                id(inventory.placement_um), id(inventory.tech), inventory.record,
                id(inventory.same_row_inventory))
    if _CAPTURED.get(inventory) != expected:
        raise ValueError('cross-row matching inventory changed')
    return inventory


def replay_native_cross_row_matching(inventory, pair):
    require_native_cross_row_matching_inventory(inventory)
    if pair not in inventory.pairs:
        raise ValueError('cross-row matching pair changed')
    if _source_pairs(inventory.placement, inventory.cell) != inventory.pairs:
        raise ValueError('cross-row matching source declarations changed')
    if inventory.record is None:
        return None
    from clay.layout.place.row_replay_session import native_matching_row_scope
    with native_matching_row_scope(program=inventory.placement.row_program,
            materialization=inventory.placement.row_materialization, tech=inventory.tech):
        actual = _replay(inventory.placement, inventory.cell, inventory.geometry,
                         inventory.placement_um, inventory.tech, inventory.pairs)
    if actual != inventory.record:
        raise ValueError('cross-row matching live authority changed')
    return (0.0, 0.0, 0.0)
