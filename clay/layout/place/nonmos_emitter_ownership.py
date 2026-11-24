"""Separate current bound native emitters from their complete source members."""
from collections.abc import Mapping


def _require(value, message):
    if not value:
        raise ValueError('non-MOS emitter ownership '+message)


def canonical_current_nonmos_block_members(plans, *, block_member_devices,
        geometry=None, placement_um=None, block_stamps=None):
    """Normalize only an actual typed emitter proved against one complete leaf.

    Public block inventories also carry physical stamp labels for placement
    relations. Those labels are not source devices. No spelling convention or
    intersection with expected source names grants permission to remove one.
    """
    if plans is None:
        return block_member_devices
    from clay.layout.nonrow_macro_plan import RigidNonmosMacroPlanSet
    from clay.layout.source_bound_nonmos_plan import SourceBoundNonmosMacroPlan
    from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
    from clay.layout.source_resistor_placement import (
        SourceResistorStamp, _stamp_binding, resistor_stamp_owner_records,
    )
    _require(type(plans) is RigidNonmosMacroPlanSet, 'requires the exact native plan set')
    plans.__post_init__()
    native = tuple(plan for plan in plans.plans if type(plan) is SourceBoundNonmosMacroPlan
                   and type(plan.assembly) is CurrentResistorPCellAssembly)
    if not native:
        return block_member_devices
    _require(all(isinstance(value, Mapping) for value in
                 (geometry, placement_um, block_stamps, block_member_devices)),
             'requires actual geometry, positions and complete source/stamp leaf mappings')
    canonical = {}
    for name, members in block_member_devices.items():
        _require(type(name) is str and bool(name)
            and type(members) in (tuple, list, set, frozenset) and bool(members)
            and all(type(member) is str and bool(member) for member in members)
            and len(members) == len(set(members)), 'source member inventory is malformed or repeated')
        canonical[name] = set(members)
    for plan in native:
        assembly, members = plan.assembly, set(plan.members)
        assembly.validate()
        _require(plan.pdk_id == plans.tech_id, 'native source PDK differs')
        claims = tuple((name, stamp) for name, stamp in geometry.items()
            if name in members or members.intersection(
                getattr(stamp, 'composite_members', ()) or ()))
        _require(len(claims) == 1, 'complete source assembly needs one unique actual emitter')
        name, stamp = claims[0]
        _require(type(stamp) is SourceResistorStamp, 'source composite is not an exact bound native stamp')
        _require(_stamp_binding(stamp).assembly is assembly,
                 'actual stamp lost the same current assembly object')
        _require(name not in members and name in placement_um,
                 'actual physical emitter label/placement is missing or shadows source')
        resistor_stamp_owner_records(stamp, placement_um[name], plan.members)
        leaves = tuple(leaf for leaf, stamps in block_stamps.items() if name in stamps)
        _require(len(leaves) == 1, 'actual emitter has missing or repeated physical leaf ownership')
        leaf, = leaves
        _require(leaf in canonical and set(block_stamps[leaf]) == {name},
                 'complete source assembly must own exactly one complete physical leaf')
        _require(canonical[leaf] == members | {name},
                 'physical leaf differs from the complete source members plus verified emitter')
        canonical[leaf] = members
    return canonical
