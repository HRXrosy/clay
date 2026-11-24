"""One structural finger-owner sequence for materialization and its projections."""
from .row_calculus import RowAtom, RowAtomKind, RowMacro, RowString, RowStructureKind
from .row_program_primitives import RowProgramError


def expected_row_finger_owners(entry) -> tuple[str, ...]:
    """Replay the typed source entry; names alone never imply owner or weight."""
    from .row_program import RowProgramEntry, _atom_owner

    if type(entry) is not RowProgramEntry or type(entry.macro) is not RowMacro:
        raise RowProgramError('finger ownership requires an exact row program entry and macro')
    if type(entry.macro.rows) is not tuple or any(type(row) is not RowString for row in entry.macro.rows):
        raise RowProgramError('finger ownership requires exact source rows')
    if any(type(row.atoms) is not tuple or any(type(atom) is not RowAtom for atom in row.atoms)
           for row in entry.macro.rows):
        raise RowProgramError('finger ownership requires exact source atoms')
    entry.__post_init__()
    result = []
    for row in entry.macro.rows:
        for atom in row.atoms:
            if atom.kind is RowAtomKind.ACTIVE:
                owner = _atom_owner(atom, entry.active_devices)
                weight = (2 if entry.structure_kind is RowStructureKind.COMMON_CENTROID
                          and atom.match_key == 'cc:matched-pair-unit' else 1)
                result.extend((owner,) * weight)
            elif atom.kind is RowAtomKind.DUMMY:
                owners = tuple(device for device in entry.dummy_devices
                               if f':{device}:' in f':{atom.atom_id}:')
                if len(owners) != 1:
                    raise RowProgramError(f'dummy atom {atom.atom_id!r} lacks one exact owner')
                result.append(owners[0])
            else:
                raise RowProgramError('finger ownership requires a typed active or dummy atom')
    if set(result) != set(entry.consumed_devices):
        raise RowProgramError('finger owner sequence differs from the complete consumed inventory')
    return tuple(result)
