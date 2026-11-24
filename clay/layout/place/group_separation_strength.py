"""Canonical encoding of a source group envelope separation in nanometres."""

def separation_gap_nm(strength):
    if strength == 'envelope_separation':
        return 0
    prefix = 'envelope_separation_nm:'
    if type(strength) is not str or not strength.startswith(prefix):
        return None
    raw = strength[len(prefix):]
    if not raw.isascii() or not raw.isdigit() or raw.startswith('0'):
        raise ValueError('invalid group separation distance')
    value = int(raw)
    if value > 1_000_000_000:
        raise ValueError('group separation distance exceeds limit')
    return value

def separation_strength(gap_nm):
    if type(gap_nm) is not int or not 0 <= gap_nm <= 1_000_000_000:
        raise ValueError('group separation distance must be a bounded nonnegative integer')
    return 'envelope_separation' if not gap_nm else f'envelope_separation_nm:{gap_nm}'
