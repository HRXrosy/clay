"""Select solver rail bounds from one materializer's current rail inventory.

These are derived placement arguments, not a source or geometry capability.
Only the materializer knows that its coarse keepouts and exact reservation
come from the same declarations. Public CP callers retain their explicit rules.
The original rail widths/net inventory still governs physical ring padding.
"""
from clay.layout.place.placement_constants import _BAND_CENTER


def materialized_rail_kwargs(*, edge_rail_reservation,
        bottom_center_keepout_nm, bottom_edge_keepout_nm,
        bottom_edge_keepout_exempt, bottom_edge_keepout_clear,
        top_edge_keepout_nm, top_edge_keepout_clear):
    if edge_rail_reservation is not None:
        from clay.layout.place.edge_rail_reservation import EdgeRailReservationIR
        if type(edge_rail_reservation) is not EdgeRailReservationIR:
            raise TypeError('materialized rail policy requires an exact rail reservation')
        edge_rail_reservation.payload()
        return dict(bottom_center_keepout_nm=0, bottom_edge_keepout_nm=0,
            bottom_edge_keepout_exempt=(), bottom_edge_keepout_clear=(),
            top_edge_keepout_nm=0, top_edge_keepout_clear=())
    return dict(bottom_center_keepout_nm=bottom_center_keepout_nm,
        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
        bottom_edge_keepout_exempt=bottom_edge_keepout_exempt,
        bottom_edge_keepout_clear=bottom_edge_keepout_clear,
        top_edge_keepout_nm=top_edge_keepout_nm,
        top_edge_keepout_clear=top_edge_keepout_clear)


def materialized_replay_seed_positions(blocks, bands, rail_kwargs, *, flexible_envelope_constraints=()):
    """Normalize the measured frame using the same bounds as its next solver.

    Legacy callers keep their original rail floor translation. A current exact
    reservation has already retired those coarse bounds, so this step cannot add
    a second rail-dependent translation to the measured candidate.
    """
    blocks = tuple(blocks)
    outline = [box for _master,box in blocks]
    if flexible_envelope_constraints:
        from clay.layout.place.constraints_spatial import _flexible_seed_physical_views
        physical,groups = _flexible_seed_physical_views(
            {master:box[:2] for master,box in blocks},
            dims={master:(box[2]-box[0],box[3]-box[1]) for master,box in blocks},
            constraints=flexible_envelope_constraints)
        outline.extend(physical.values())
        outline.extend(groups.values())
    xmin = min(box[0] for box in outline)
    ymin = min(box[1] for box in outline)
    exempt = set(rail_kwargs['bottom_edge_keepout_exempt'])
    clear = set(rail_kwargs['bottom_edge_keepout_clear'])
    edge = rail_kwargs['bottom_edge_keepout_nm']
    center = rail_kwargs['bottom_center_keepout_nm']
    shift = max(0, max((max(
        edge if master not in exempt and master not in clear else 0,
        center if bands[master] == _BAND_CENTER else 0,
    ) - (box[1] - ymin) for master, box in blocks), default=0))
    return {master: (box[0] - xmin, box[1] - ymin + shift)
            for master, box in blocks}
