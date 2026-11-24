"""Resolve declared passive rings through complete current native PCell stamps.

The source supplies well/tie membership. Actual typed stamp parts and source
terminal records establish which shared emitter owns those members. A tuple of
names alone never grants passive composite ownership.
"""
from clay.layout.source_resistor_placement import (
    SourceResistorStamp, is_resistor_stamp, _stamp_binding,
    require_resistor_member_record, resistor_stamp_owner_records,
)


def _require(value, message):
    if not value:
        raise ValueError('current passive ring '+message)


def current_passive_ring_owners(geometry, placement, intent, tech_id, *,
                                island_specs, member_well_tie):
    """Return logical-to-emitter aliases after current full raw/native replay."""
    passives = tuple(getattr(intent, 'passives', ()) or ())
    by_name = {row.name: row for row in passives}
    candidates = []
    for name, stamp in geometry.items():
        members = getattr(stamp, 'composite_members', ()) or ()
        if is_resistor_stamp(stamp) or set(members).intersection(by_name):
            candidates.append((name, stamp))
    if not candidates:
        return {}
    _require(len(by_name) == len(passives), 'source passive member inventory is repeated')
    producers, mapping = {}, {}
    for name, stamp in candidates:
        _require(type(stamp) is SourceResistorStamp,
                 'passive composite requires an exact bound current resistor stamp')
        binding = _stamp_binding(stamp)
        assembly = binding.assembly
        _require(assembly.pdk_id == tech_id, 'source resistor PDK differs from ring materializer')
        if id(assembly.producer) not in producers:
            assembly.producer.require_current()
            producers[id(assembly.producer)] = assembly.producer
        members = assembly.plan.members
        _require(set(members) <= set(by_name), 'complete source resistor member cover is missing')
        _require(name in placement, 'actual resistor composite placement is missing')
        resistor_stamp_owner_records(stamp, placement[name], members)
        for owner in members:
            source = by_name[owner]
            require_resistor_member_record(dict(kind=source.kind, value=source.value, M=source.M,
                model=source.model, pins=source.nets), assembly, owner, source.orient)
            emitters = [key for key, value in geometry.items()
                if owner in (getattr(value, 'composite_members', ()) or (key,))]
            _require(emitters == [name] and owner not in mapping,
                     'source resistor has duplicated direct/composite emitter ownership')
        touched = tuple(spec for spec in (island_specs or ()) if set(spec[0]).intersection(members))
        if touched:
            _require(len(touched) == 1 and set(members) <= set(touched[0][0]),
                     'declared ring splits or repeats complete resistor PCell members')
            declared, well, tie = touched[0]
            _require(len(declared) == len(set(declared)) and set(declared) <= set(by_name),
                     'declared passive island has foreign or repeated source members')
            _require(well in ('p', 'n') and type(tie) is str and bool(tie) and tie == tie.strip(),
                     'passive composite ring needs explicit source well and tie')
            _require(all(member_well_tie.get(owner) == (well, tie) for owner in declared),
                     'declared passive composite well/tie membership differs')
        mapping.update((owner, name) for owner in members)
    for producer in producers.values():
        producer.require_current()
    return mapping
