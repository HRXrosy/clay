"""Resolve nested declarations to their physical owner before graph validation."""


def array_owner_parent(members, macros, functional_parent):
    """An internal array belongs to its declared physical aggregate when present.

    Functional groups describe purpose; their physical children own geometry.
    Ambiguous partial/overlapping owners remain errors in the graph validator.
    """
    selected = set(members)
    owners = [m for m in macros if m.kind in ('power_array', 'passive_bank', 'bjt_cluster')
              and selected <= set(m.members)]
    narrowest = [m for m in owners if not any(set(n.members) < set(m.members) for n in owners)]
    if len(narrowest) > 1:
        raise ValueError('internal array has ambiguous declared physical owners')
    return narrowest[0].macro_id if narrowest else functional_parent
