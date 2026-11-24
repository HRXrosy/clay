"""Resolve band authority and compile its native full-width row constraints.

Band labels also serve seeds and local keepouts; their existence alone does not
override an explicit request to leave global row order unconstrained.
"""
from clay.layout.place.placement_constants import _BAND_CENTER


def resolve_band_order(value: bool | None, *, hierarchical_macro_orders) -> bool:
    """Preserve the historical default only when no explicit policy was given."""
    if value is not None and type(value) is not bool:
        raise TypeError('band order must be an exact bool or None')
    return not bool(hierarchical_macro_orders) if value is None else value


def add_native_band_constraints(model, ys, dims, *, bands, core_ids, present,
                                flexible_envelope_constraints, flexible_vars,
                                gap, h_bound, bottom_center_keepout_nm,
                                enforce_legacy_band_order):
    """Add independent declared band constraints without replacing a hierarchy.

    Preserve legacy constraint creation order, including its centre keepout;
    the caller independently owns all physical collision and hierarchy rows.
    """
    if type(enforce_legacy_band_order) is not bool:
        raise TypeError('resolved band order must be an exact bool')
    if not enforce_legacy_band_order:
        for bid in core_ids:
            if bands[bid] == _BAND_CENTER:
                model.Add(ys[bid] >= int(bottom_center_keepout_nm))
        return
    band_floor = {b: model.NewIntVar(0, h_bound, f'bf_{b}') for b in present}
    model.Add(band_floor[present[0]] == 0)
    for bid in core_ids:
        model.Add(ys[bid] >= band_floor[bands[bid]])
        if bands[bid] == _BAND_CENTER:
            model.Add(ys[bid] >= int(bottom_center_keepout_nm))
    for row in flexible_envelope_constraints:
        _gx0, gy0, _gx1, _gy1 = flexible_vars[row.group]
        model.Add(gy0 >= band_floor[bands[row.members[0]]])
    for index in range(1, len(present)):
        current, previous = present[index], present[index - 1]
        for bid in core_ids:
            if bands[bid] == previous:
                model.Add(band_floor[current] >= ys[bid] + dims[bid][1] + gap)
        for row in flexible_envelope_constraints:
            if bands[row.members[0]] == previous:
                _gx0, _gy0, _gx1, gy1 = flexible_vars[row.group]
                model.Add(band_floor[current] >= gy1 + gap)
