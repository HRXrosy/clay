"""Explicit per-edge requests for fixed-artwork mosaic construction.

An abutment requests zero envelope gap, not the legacy positive shared-span
overlap. Its content reference must be consumed by the physical constructor;
these records alone prove neither terminal connectivity nor PDK legality.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RowBoundaryJoin:
    left_atom_id: str
    right_atom_id: str
    mode: str
    gap_nm: int
    interface_sha256: str
    declaration_ref: str

    def __post_init__(self):
        for name in ('left_atom_id', 'right_atom_id', 'declaration_ref'):
            value = getattr(self, name)
            if type(value) is not str or not 0 < len(value) <= 4096 or value != value.strip():
                raise ValueError('row boundary ' + name + ' must be a bounded canonical string')
        if self.left_atom_id == self.right_atom_id:
            raise ValueError('row boundary must join two distinct atoms')
        if type(self.gap_nm) is not int or not 0 <= self.gap_nm <= 1_000_000_000:
            raise ValueError('row boundary gap must be a bounded exact integer')
        if type(self.mode) is not str or self.mode not in {'isolated', 'fixed_artwork_abutment', 'fixed_artwork_shared_access'}:
            raise ValueError('row boundary mode is unsupported')
        if self.mode == 'isolated':
            if self.gap_nm <= 0 or type(self.interface_sha256) is not str or self.interface_sha256:
                raise ValueError('isolated boundary requires positive gap and no abutment interface')
        elif (self.gap_nm != 0 or type(self.interface_sha256) is not str
              or len(self.interface_sha256) != 64
              or any(c not in '0123456789abcdef' for c in self.interface_sha256)):
            raise ValueError('fixed artwork interface requires zero gap and an exact interface SHA256')

    def payload(self):
        self.__post_init__()
        return asdict(self)


def checked_boundary_joins(value, atom_ids, declaration_refs):
    if type(value) is not tuple or len(value) != len(atom_ids)-1:
        raise ValueError('row boundary joins must cover the exact adjacent-pair inventory')
    for item, pair in zip(value, zip(atom_ids, atom_ids[1:])):
        if type(item) is not RowBoundaryJoin:
            raise TypeError('row boundary join requires the exact immutable type')
        item.__post_init__()
        if (item.left_atom_id, item.right_atom_id) != pair:
            raise ValueError('row boundary join order or adjacency differs')
        if item.declaration_ref not in declaration_refs:
            raise ValueError('row boundary retains an undeclared join reference')
    return value
