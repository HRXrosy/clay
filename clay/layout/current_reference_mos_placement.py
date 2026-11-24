"""Current source/reference MOS ingredients to ordinary placement operands.

Source guard-ring closures and existing row names define complete macros. The
original arrays, common centroids, dummies and floorplan declarations remain
intact. Observed coordinates enter only ASIR; native assembly accepts discrete
order, complete raw artwork and independently bound terminal interfaces.
"""
from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
from types import MappingProxyType


def _require(value, message):
    if not value:
        raise ValueError('current reference MOS placement: ' + message)


def plan_reference_mos_groups(*, source_cell, intent, source_members, row_program):
    """Partition the full MOS source without splitting a declared native row."""
    from clay.layout.place.row_program import _group_map
    owners = {name for name, member in source_members.items() if member.kind in ('nmos', 'pmos')}
    _require(owners, 'complete MOS source inventory is empty')
    entries = tuple((entry.group, frozenset(entry.consumed_devices)) for entry in row_program.entries
                    if owners.intersection(entry.consumed_devices))
    _require(all(names <= owners for _, names in entries), 'existing row mixes MOS and foreign source owners')
    groups = _group_map(source_cell)
    preferred = {item.target for item in intent.dummy_flanks}
    preferred.update(item.group for item in (*intent.arrays, *intent.common_centroids))
    partitions = []
    for ring in intent.guard_rings:
        names = frozenset(owners.intersection(ring.members))
        if not names:
            continue
        _require(not any(names.intersection(prior) for prior in partitions), 'overlapping source guard-ring partitions')
        _require(all(not names.intersection(row) or row <= names for _, row in entries),
                 'source guard-ring partition splits an existing row')
        partitions.append(names)
    covered = set().union(*partitions) if partitions else set()
    for _, names in entries:
        if names <= covered:
            continue
        _require(not names.intersection(covered), 'source partition splits an existing row')
        partitions.append(names)
        covered.update(names)
    partitions.extend(frozenset((name,)) for name in sorted(owners-covered))
    requests = []
    for names in partitions:
        _require(len({source_members[name].kind for name in names}) == 1,
                 'source guard-ring macro mixes MOS polarities')
        exact = sorted({name for name, members in entries if members == names})
        candidates = sorted((len(set(members)), name in preferred, name) for name, members in groups.items()
                            if members and set(members) <= names)
        if exact:
            _require(len(exact) == 1, 'source row has ambiguous existing names')
            name = exact[0]
        elif candidates:
            best = max((size, declared) for size, declared, _ in candidates)
            name = min(name for size, declared, name in candidates if (size, declared) == best)
        elif len(names) == 1:
            name = next(iter(names))
        else:
            name = 'mos_' + hashlib.sha256('\0'.join(sorted(names)).encode()).hexdigest()[:16]
        requests.append((name, tuple(sorted(names))))
    _require(len({name for name, _ in requests}) == len(requests), 'source macro names collide')
    _require(Counter(owner for _, names in requests for owner in names) == Counter(owners),
             'complete source owner partition differs')
    return tuple(sorted(requests))


def _boundary_intent(mosaic, ingredients, members, provenance):
    """Bind each measured facing SD site to the actual source net aliases."""
    from clay.layout.spatial.mosaic_boundary_joins import (
        MosaicInterfaceEndpoint, MosaicAbutmentInterface, MosaicSharedAccessInterface, MosaicBoundaryJoinIntent,
        replay_mosaic_boundary_join_intent,
    )
    if not ingredients['interfaces']:
        return None
    sites = {site.site_id: site for unit in ingredients['units'] for site in unit.sites}
    interfaces = {}
    for interface in ingredients['interfaces']:
        _require(interface.left_site_id in sites and interface.right_site_id in sites,
                 'shared boundary lacks a complete raw source site')
        left, right = sites[interface.left_site_id], sites[interface.right_site_id]
        _require(left.source_net == right.source_net, 'shared boundary source nets differ')
        endpoints = tuple(MosaicInterfaceEndpoint(site.occurrence_id, site.owner,
            site.site_id.rsplit('/', 1)[1], site.variant_sha256, site.allowed_roles) for site in (left, right))
        member = members[left.owner]
        from clay.layout.mos_mosaic_materializer import MosMosaicSharedAccessInterface
        constructor = MosaicSharedAccessInterface if type(interface) is MosMosaicSharedAccessInterface else MosaicAbutmentInterface
        value = constructor(*endpoints, left.source_net, member.kind, member.oxide,
            member.body_net, 'declared_not_traced', provenance)
        pair = left.occurrence_id, right.occurrence_id
        _require(pair not in interfaces, 'shared boundary adjacency is duplicated')
        interfaces[pair] = value
    ordered = tuple(interfaces[(left.node_id, right.node_id)] for row in mosaic.rows
        for left, right in zip(row, row[1:]) if (left.node_id, right.node_id) in interfaces)
    _require(len(ordered) == len(interfaces), 'shared boundary splits rows or lacks oriented adjacency')
    return replay_mosaic_boundary_join_intent(MosaicBoundaryJoinIntent(mosaic.sha256, ordered),
                                             mosaic=mosaic, members=members)


class ReferenceMosConstructionConflict(ValueError):
    """A source construction declaration contradicts measured channel geometry."""
    def __init__(self, conflicts):
        self.conflicts = tuple(conflicts)
        super().__init__('SOURCE_CONSTRUCTION_CONFLICT: ' + repr(self.conflicts))


def _require_construction_centers(source_cell, intent, centers):
    from fractions import Fraction
    from clay.layout.place.row_program import _group_map
    from clay.layout.mosaic_geometry import _D4
    groups = _group_map(source_cell)
    conflicts = []
    declarations = tuple(('common_centroid', item) for item in intent.common_centroids)
    declarations += tuple(('array', item) for item in intent.arrays)
    for kind, declaration in declarations:
        owners = groups.get(declaration.group, (declaration.group,))
        if not set(owners).intersection(centers):
            continue  # Passive arrays retain their existing independent consumer.
        _require(set(owners) <= set(centers), 'construction source owner denominator is incomplete')
        if kind == 'array' and declaration.style != 'shared_diffusion':
            # A complete reference mosaic is not a consumer of source discrete
            # transforms, pin facing, alignment, order or bank/unit patterns.
            # Keep the unsupported family explicit even when row counts agree.
            conflicts.append((kind, declaration.group, 'unsupported_mos_style',
                              declaration.style, 'requires_source_discrete_array_consumer'))
        orientation = getattr(declaration, 'orientation', 'R0')
        (a, b), (c, d) = _D4[orientation]
        local = {owner: tuple((a*x+c*y, b*x+d*y) for x, y in centers[owner]) for owner in owners}
        rows = len({y for points in local.values() for _, y in points})
        if rows != declaration.rows:
            conflicts.append((kind, declaration.group, 'rows_in_declared_frame', declaration.rows, rows))
        if kind == 'common_centroid':
            centroids = {owner: tuple(Fraction(sum(point[axis] for point in points), len(points))
                                      for axis in (0, 1)) for owner, points in local.items()}
            if len(set(centroids.values())) != 1:
                conflicts.append((kind, declaration.group, 'centroid', 'coincident',
                                  tuple((owner, tuple(str(v) for v in point)) for owner, point in sorted(centroids.items()))))
            if declaration.first_member:
                first = min((y, x, owner) for owner, points in local.items() for x, y in points)[2]
                if first != declaration.first_member:
                    conflicts.append((kind, declaration.group, 'first_member', declaration.first_member, first))
            if declaration.member_transforms:
                conflicts.append((kind, declaration.group, 'member_transforms',
                                  declaration.member_transforms, 'reference_raw_frame_relation_requires_explicit_replay'))
    if conflicts:
        raise ReferenceMosConstructionConflict(conflicts)


def require_reference_mos_construction(*, source_cell, intent, overlay):
    """Check explicit rows and centroid semantics before reference projection wins.

    These geometric conflict checks are not a replacement for native terminal,
    all-layer legality, matching or final reference-quality verification.
    """
    centers = {}
    for occurrence in overlay.occurrences:
        centers.setdefault(occurrence.logical_owner, []).append(occurrence.physical.anchor_nm)
    _require_construction_centers(source_cell, intent, centers)


@dataclass(frozen=True)
class CurrentReferenceMosPlacement:
    """Placement operands whose active reference lifetime must remain current."""
    intent: object
    row_source_projection: object
    row_geometry_overrides: object
    reference_inputs: object
    _snapshot: object = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'row_geometry_overrides', MappingProxyType(dict(self.row_geometry_overrides)))
        object.__setattr__(self, '_snapshot', self._values())

    def _values(self):
        from clay.layout.place.row_replay_snapshot import geometry_snapshot, freeze_row_values
        program = self.row_source_projection.projected_program
        members = {entry.owner: entry.consumed_devices for entry in program.entries}
        _require(set(members) == set(self.row_geometry_overrides), 'complete macro geometry overrides differ')
        return (freeze_row_values((self.intent, self.row_source_projection)),
                tuple((owner, geometry_snapshot(geometry, members[owner]))
                      for owner, geometry in sorted(self.row_geometry_overrides.items())))

    def require_current(self):
        from clay.layout.place.row_source_projection import replay_row_source_projection
        self.reference_inputs.require_current()
        replay_row_source_projection(self.row_source_projection,
            self.reference_inputs.source_cell, self.reference_inputs.tech)
        _require(self._values() == self._snapshot, 'captured source intent or actual geometry changed')


def capture_current_reference_mos_placement(source_cell, *, reference, tech, checkpoint=None):
    """Capture fresh ingredients, project complete groups and construct native rows."""
    from clay.frontend.current_reference_mos_inputs import capture_current_reference_mos_inputs
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place.row_program import compile_row_program
    from clay.layout.place.row_source_projection import _capture_row_source_projection
    from clay.layout.spatial import derive_abstract_spatial_ir, project_discrete_mosaic
    from clay.layout.mos_mosaic_materializer import assemble_mos_mosaic
    from clay.layout.joint_row_geometry import joint_checkpoint
    check = joint_checkpoint(checkpoint)
    inputs = capture_current_reference_mos_inputs(source_cell, reference=reference, tech=tech)
    intent = layout_intent_from_cell_def(source_cell)
    require_reference_mos_construction(source_cell=source_cell, intent=intent, overlay=inputs.overlay)
    program = compile_row_program(source_cell, intent, tech)
    requests = plan_reference_mos_groups(source_cell=source_cell, intent=intent,
                                        source_members=inputs.source_members, row_program=program)
    scopes = tuple(sorted((owner, name) for name, owners in requests for owner in owners))
    asir = derive_abstract_spatial_ir(inputs.overlay, mosaic_scope_by_logical=scopes)
    mosaics = tuple(project_discrete_mosaic(asir, mosaic_id=name, logical_owners=owners)
                    for name, owners in requests)
    ingredients = tuple(inputs.for_mosaic(mosaic, mosaic_scope_by_logical=scopes) for mosaic in mosaics)
    boundaries = tuple(_boundary_intent(mosaic, selected, inputs.source_members, inputs.provenance_sha256)
                       for mosaic, selected in zip(mosaics, ingredients))
    joins = tuple(boundary for boundary in boundaries if boundary is not None)
    projection = _capture_row_source_projection(canonical_source_cell=source_cell,
        authoring_source_cell=source_cell, tech=tech, reference_overlay=inputs.overlay, reference_asir=asir,
        mosaic_requests=requests, discrete_mosaics=mosaics, mosaic_boundary_joins=joins,
        mosaic_scope_by_logical=scopes)
    intent = replace(intent, discrete_mosaics=mosaics, mosaic_boundary_joins=joins)
    entries = {entry.group: entry for entry in projection.projected_program.entries}
    _require(set(entries) == {name for name, _ in requests}, 'projected macro denominator differs')
    overrides = {}
    for mosaic, selected, boundary in zip(mosaics, ingredients, boundaries):
        check()
        geometry = assemble_mos_mosaic(mosaic=mosaic, **selected, tech=tech,
            connect_model=inputs.connect_model, source_members=inputs.source_members,
            provenance_sha256=inputs.provenance_sha256, checkpoint=check)
        if boundary is not None:
            geometry = replace(geometry, mosaic_boundary_intent=boundary, tech=tech)
        entry = entries[mosaic.mosaic_id]
        from clay.layout.mosaic_boundary_materialization import replay_mosaic_boundary_entry_geometry
        replay_mosaic_boundary_entry_geometry(entry, geometry, tech=tech, members=inputs.source_members)
        overrides[entry.owner] = geometry
    from clay.layout.joint_row_geometry import joint_occurrence_centers
    current_centers = {owner: tuple((round(x*1000), round(y*1000)) for x, y, _pose in values)
        for geometry in overrides.values()
        for owner, values in joint_occurrence_centers(geometry.joint_row_binding, (0, 0))}
    _require_construction_centers(source_cell, intent, current_centers)
    inputs.require_current()
    result = CurrentReferenceMosPlacement(intent, projection, overrides, inputs)
    result.require_current()
    return result
