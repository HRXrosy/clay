"""Preserve the complete authored module cover for reference quality targets.

This projection does not make groups rigid owners. In particular, parent groups
remain independent measurement obligations beside their declared children.
The calling producer authenticates the parsed source and its lifetime.
"""
from clay.layout.intent import layout_intent_from_cell_def
from clay.layout.reference_quality_contract import _module_cover, _name, _require


def source_quality_groups(cell):
    names = tuple(device.instance_name for device in cell.devices)
    _require(bool(names) and len(set(names)) == len(names),
             'quality source device inventory is empty or duplicated')
    for name in names:
        _name(name)
    groups = layout_intent_from_cell_def(cell).place_groups
    result = []
    for group in groups:
        _name(group.group)
        members = tuple(group.devices)
        _require(bool(members) and len(set(members)) == len(members),
                 'quality source group members are empty or duplicated')
        for name in members:
            _name(name)
        result.append((group.group, tuple(sorted(members))))
    _require(len({name for name, _members in result}) == len(result),
             'quality source group identities are duplicated')
    result = tuple(sorted(result))
    _module_cover(result, names, hierarchical=True)
    return result
