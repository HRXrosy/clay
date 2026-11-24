"""Current measured reference plates to explicit source MIM construction.

The source association uses the producer's freshly checked complete mapping.
Only intrinsic master artwork and exact D4 reach construction; global placement
coordinates are removed. This module does not establish whole-cell acceptance.
"""
from collections import Counter
import hashlib
import json

from clay.frontend.reference_device_artwork import CurrentReferenceDeviceArtwork
from clay.frontend.reference_device_shapes import _map_shape
from clay.frontend.reference_occurrence_binding import LogicalOwner, PhysicalOccurrence
from clay.frontend.unit_master_geometry import (
    UnitMasterSource, UnitTerminalSite, normalize_unit_master_geometry,
)
from clay.layout.capacitor_unit_occurrences import _compose
from clay.layout.source_mim_materialization import (
    MimSourceRecipe, _member, _require, capture_mim_source_association,
    materialize_source_mim,
)


def _inverse(frame):
    a, b, c, d, dx, dy = frame
    return a, c, b, d, -a*dx-c*dy, -b*dx-d*dy


def _master(record, *, pdk_id, gds_sha256, observation_sha256):
    _require(record.family == 'capacitor' and record.primitive.center_basis == 'ctm_plate_bbox_center',
             'current reference MIM requires an actual CTM plate')
    inverse = _inverse(record.raw_to_global)
    sites = tuple(sorted((UnitTerminalSite(role, 'plate', ((layer, tuple(sorted(
        _map_shape(shape, inverse) for shape in shapes))),))
        for role, layer, shapes in record.primitive.terminals), key=lambda site: site.site_id))
    _require(tuple(site.site_id for site in sites) == ('bottom', 'top'),
             'current reference MIM requires complete original top/bottom sites')
    actual = Counter((key, shape) for key, shapes in record.raw_master_geometry for shape in shapes)
    replayed = Counter((key, _map_shape(shape, inverse)) for key, shape in record.parts)
    _require(actual == replayed, 'current raw MIM complete master multiset differs')
    _require(all(count == 1 for count in actual.values()),
             'current raw MIM duplicate records require a multiset-capable unit master')
    return normalize_unit_master_geometry(pdk_id=pdk_id, parameter_class=record.parameter_class,
        sources=(UnitMasterSource(gds_sha256, record.master_name, observation_sha256),),
        geometry_by_layer_nm=record.raw_master_geometry, terminal_sites=sites,
        unresolved_access_obligations=())


def materialize_current_reference_mims(source_cell, *, reference, tech):
    """Return every source capacitor as a replayable typed native materialization."""
    _require(type(reference) is CurrentReferenceDeviceArtwork, 'requires current exact reference producer')
    reference.require_current()
    data = reference.canonical_data()
    payload = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    observation_sha = hashlib.sha256(payload).hexdigest()
    evidence = tuple(sorted(data['input_sha256']))
    gds = tuple(path for path in evidence if path.lower().endswith('.gds'))
    _require(len(gds) == 1, 'current reference requires one explicitly measured GDS')
    expected = {device.instance_name for device in source_cell.devices if device.device_class == 'cap'}
    grouped = {name: tuple(row for row in reference.records if row.logical_id == name) for name in expected}
    _require({row.logical_id for row in reference.records if row.family == 'capacitor'} == expected,
             'current complete source capacitor population differs')
    from clay.layout.generic_placement import _resistor_orient_map
    orientations = _resistor_orient_map(source_cell)
    inverse_names = {'R0':'R0', 'R90':'R270', 'R180':'R180', 'R270':'R90',
                     'MR0':'MR0', 'MR90':'MR90', 'MR180':'MR180', 'MR270':'MR270'}
    net_nodes, result = dict(reference.net_to_node), {}
    for name, records in sorted(grouped.items()):
        reference.require_current()
        member, _value, count, pins = _member(source_cell, name)
        _require(count == len(records) == 1,
                 'multiunit reference MIM requires an explicit source bank recipe')
        record, = records
        master = _master(record, pdk_id=tech.pdk_id, gds_sha256=data['input_sha256'][gds[0]],
                         observation_sha256=observation_sha)
        physical_nodes = dict(record.physical_terminals)
        site_roles = []
        for site in master.terminal_sites:
            roles = tuple(role for role, net in sorted(pins.items()) if net_nodes[net] == physical_nodes[site.site_id])
            _require(len(roles) == 1, 'current reference plate has no unique source terminal role')
            site_roles.append((site.site_id, roles[0]))
        physical = PhysicalOccurrence(record.occurrence_id, record.parameter_class,
            tuple((role, (physical_nodes[site],)) for site, role in site_roles))
        logical = LogicalOwner(name, record.parameter_class,
            tuple((role, (net,)) for role, net in sorted(pins.items())), count)
        association = capture_mim_source_association(source_cell, logical_id=name,
            master=master, source_member=member, physical_occurrences=(physical,),
            logical_owners=(logical,), port_anchors=tuple(sorted((net, net_nodes[net]) for net in set(pins.values()))),
            observed_variants=((record.occurrence_id, master.variant_sha256),), evidence_files=evidence, tech=tech)
        outer = orientations.get(name, 'R0')
        recipe = MimSourceRecipe(association, master, tuple(site_roles),
            unit_transforms=(_compose(inverse_names[outer], record.orientation),), orient=outer,
            quality_plate_site='top')
        result[name] = materialize_source_mim(source_cell, recipe=recipe, tech=tech)
    reference.require_current()
    return result
