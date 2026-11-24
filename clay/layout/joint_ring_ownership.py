"""Replay an already rigid, fully owned MOS island before free-dummy packing."""
from collections import Counter, defaultdict


def resolve_dummy_ring_owners(geometry, members):
    """Resolve either disjoint direct emitters or one complete joint emitter.

    This only resolves names. The caller must replay source and actual joint
    artwork before accepting the latter as a rigid physical island.
    """
    from clay.layout.owned_artwork import _require
    provenance = {}
    for name, stamp in geometry.items():
        composite = getattr(stamp, 'composite_members', ())
        _require(type(name) is str and bool(name) and type(composite) is tuple
                 and all(type(member) is str and bool(member) for member in composite)
                 and len(set(composite)) == len(composite),
                 'free dummy stamp provenance must be an exact tuple')
        provenance[name] = composite
    owner_by_member = {}
    for member in sorted(members):
        owners = sorted(name for name, composite in provenance.items()
                        if name == member or member in composite)
        _require(len(owners) == 1, f'free dummy ring member {member!r} lacks one direct stamp owner')
        owner_by_member[member] = owners[0]
    joint_owner = next(iter(owner_by_member.values())) if len(set(owner_by_member.values())) == 1 else None
    if joint_owner is not None:
        _require(set(provenance[joint_owner]) == set(members), 'free dummy joint ring member cover is incomplete')
    else:
        _require(all(member == owner for member, owner in owner_by_member.items()),
                 'free dummy ring members do not have disjoint stamp ownership')
    return owner_by_member, joint_owner


def require_joint_ring_artwork(stamp, *, placement_um, members, devices, tech):
    """Return complete placed owner artwork boxes; no position is generated."""
    from clay.layout.device_cell import _resolve_layer_keys
    from clay.layout.joint_occurrence_center_basis import require_joint_stamp_center_basis
    from clay.layout.joint_row_geometry import _region
    from clay.layout.joint_row_stamp import JointRowStampBinding, joint_stamp_owner_records
    from clay.layout.owned_artwork import _bbox, _require
    from clay.layout.place.row_program import _nm_from_record
    from clay.layout.terminal_body_geometry import mos_channel_dimensions_nm

    binding = getattr(stamp, 'joint_row_stamp_binding', None)
    _require(type(binding) is JointRowStampBinding, 'complete joint MOS ring binding is unavailable')
    binding.__post_init__()
    _require(type(members) is tuple and members == tuple(sorted(set(members)))
             and binding.consumed_devices == members
             and set(stamp.composite_members) == set(members), 'joint MOS ring full member cover differs')
    _require(binding.source.pdk_id == tech.pdk_id, 'joint MOS ring current PDK differs')
    family_basis = binding.source.quality_centers
    _require(family_basis is not None and all(devices[name].kind == family_basis.family for name in members),
             'joint MOS ring current source family differs from the native binding')
    # This replays the current installed PDK, complete raw master/part/site graphs,
    # and the exact emitted multiset. Composite names or a receipt cannot replace it.
    records = joint_stamp_owner_records(stamp, placement_um, members)
    require_joint_stamp_center_basis(stamp)
    counts = Counter(occ.owner for occ in binding.source.occurrences)
    _require(counts == Counter({name: devices[name].M for name in members}),
             'joint MOS ring source occurrence count differs')
    role_names = dict(g='gate', d='drain', s='source')
    sites_by_occurrence = defaultdict(list)
    for site in binding.source.expected_sites:
        sites_by_occurrence[site.occurrence_id].append(site)
    # The bridge exposes microns after an SI float conversion. Re-enter the
    # same integer-nm boundary used by the source compiler/reference producer.
    dimensions_by_member = {name: tuple(_nm_from_record(
        {'W': devices[name].W * 1e-6, 'L': devices[name].L * 1e-6}, key)
        for key in ('W', 'L')) for name in members}
    for occurrence in binding.source.occurrences:
        device = devices[occurrence.owner]
        _require(device.kind in ('nmos', 'pmos'), 'joint MOS ring source family differs')
        keys = _resolve_layer_keys(tech, device.kind)
        layers = dict(occurrence.master.geometry_by_layer_nm)
        actual_dimensions = mos_channel_dimensions_nm(_region(layers[keys.active]), _region(layers[keys.poly]))
        dimensions = dimensions_by_member[occurrence.owner]
        _require(dimensions == actual_dimensions,
                 'joint MOS ring source channel dimensions differ')
        nets = getattr(device, 'nets', None)
        _require(type(nets) is dict and all(type(nets.get(role)) is str and nets[role]
                 for role in role_names.values()), 'joint MOS ring source terminal nets are unavailable')
        sites = sites_by_occurrence[occurrence.occurrence_id]
        raw_sites = {site.site_id: site for site in occurrence.master.terminal_sites}
        _require(len(sites) == 3 and Counter(site.role_group for site in raw_sites.values()) == {'gate': 1, 'sd': 2},
                 'joint MOS ring complete physical terminal inventory differs')
        for site in sites:
            local_id = site.site_id[len(occurrence.occurrence_id)+1:]
            raw_site = raw_sites.get(local_id)
            _require(raw_site is not None, 'joint MOS ring physical terminal identity differs')
            roles = ('g',) if raw_site.role_group == 'gate' else ('d', 's')
            _require(site.owner == occurrence.owner and site.allowed_roles == tuple(
                role for role in roles if nets[role_names[role]] == site.source_net),
                'joint MOS ring current source terminal binding differs')
    return {owner: _bbox(records[owner]) for owner in members}
