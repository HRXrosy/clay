"""Sealed layered physical ownership joined to a final placement certificate.

This module performs immutable data checks only. Current source, PDK and actual
geometry replay belongs to the materialization boundary, not certificate hashing.
"""
from __future__ import annotations

from collections import Counter


def with_layered_assembly_payload(certificate, payload):
    """Version the final state only after its complete layered proof joins."""
    proof = layered_assembly_certificate_payload(certificate)
    return {**payload, **proof, 'schema': 'clay.final-placement-state.v34'} if proof else payload


def require_materialized_footprint_certificate(certificate):
    """Replay the complete final footprint/cavity/rail and full-2D graph join.

    These are independent physical proof currencies. Their canonical operands
    must agree at the same final positions; an assembly proof replaces none of
    these obligations and never grants a blanket overlap exception.
    """
    from clay.layout.mosaic_guest_slot_projection import mosaic_guest_slot_satisfied
    from clay.layout.place.constraints_spatial import (
        _canonical_layered_collision_footprints, _canonical_overlap_compatible_pairs,
        _canonical_mosaic_guest_slot_projections,
        _canonical_row_bounded_cavity_projections, _row_bounded_cavity_satisfied,
    )
    from clay.layout.place.placement_replay import _require_edge_rail_reservation
    dimensions = {name: (bbox[2] - bbox[0], bbox[3] - bbox[1]) for name, bbox in certificate.blocks}
    canonical_layered_footprints = _canonical_layered_collision_footprints(
        certificate.layered_collision_footprints, dims=dimensions)
    canonical_cavity_projections = _canonical_row_bounded_cavity_projections(
        certificate.row_bounded_cavity_projections, dims=dimensions)
    if canonical_cavity_projections != certificate.row_bounded_cavity_projections:
        raise ValueError('row-bounded cavity projections are not canonical')
    canonical_slots = _canonical_mosaic_guest_slot_projections(certificate.mosaic_guest_slot_projections,
        dims=dimensions, minimum_clearance_nm=certificate.declared_gap_nm)
    if canonical_slots != certificate.mosaic_guest_slot_projections:
        raise ValueError('mosaic guest-slot projections are not canonical')
    positions = {name: (bbox[0], bbox[1]) for name, bbox in certificate.blocks}
    _require_edge_rail_reservation(certificate.edge_rail_reservation, dims=dimensions)
    if certificate.edge_rail_reservation is not None and not certificate.edge_rail_reservation.replay(
            dims=dimensions, positions=positions, physical_bbox_nm=certificate.bbox_nm):
        raise ValueError('edge rail reservation does not replay final placement')
    if any(not mosaic_guest_slot_satisfied(positions, row) for row in canonical_slots):
        raise ValueError('mosaic guest-slot projection does not replay')
    if any(not _row_bounded_cavity_satisfied(positions, dims=dimensions, projection=row,
            minimum_clearance_nm=certificate.declared_gap_nm) for row in canonical_cavity_projections):
        raise ValueError('row-bounded cavity projection does not replay')
    canonical_overlap_pairs = _canonical_overlap_compatible_pairs(certificate.overlap_compatible_pairs,
        block_ids=tuple(name for name, _bbox in certificate.blocks))
    if certificate.full2d_compaction_graph is not None:
        from clay.layout.place.full2d_geometry_boundary import require_full2d_geometry_proof
        require_full2d_geometry_proof(certificate.full2d_compaction_graph, certificate.blocks,
            overlap_compatible_pairs=canonical_overlap_pairs,
            layered_footprints_nm=canonical_layered_footprints, solver_evidence=certificate.full2d_solver_evidence)


def layered_assembly_certificate_payload(certificate):
    bindings = getattr(certificate, 'layered_placement_assemblies', ())
    replay = getattr(certificate, 'layered_placement_replay', None)
    projection = getattr(certificate, 'layered_placement_projection', None)
    if type(bindings) is not tuple or len(bindings) > 4096:
        raise ValueError('layered assembly certificate binding inventory must be an exact bounded tuple')
    if not bindings:
        if replay is not None or projection is not None:
            raise ValueError('layered assembly certificate proof requires its assembly inventory')
        return {}
    from clay.layout.layered_placement_assembly import (
        LayeredPlacementAssemblyBinding, LayeredPlacementAssemblyBlockProjection,
        LayeredPlacementAssemblyReplay, _bbox, _move, _require, _same, joint_checkpoint,
    )
    _require(all(type(binding) is LayeredPlacementAssemblyBinding for binding in bindings),
             'certificate assembly inventory is foreign')
    _require(type(replay) is LayeredPlacementAssemblyReplay
             and type(projection) is LayeredPlacementAssemblyBlockProjection,
             'certificate assembly requires paired replay and projection proof')
    for binding in bindings:
        joint_checkpoint()()
        binding.validate()
    replay.validate()
    projection.validate()
    _require(tuple(binding.assembly_id for binding in bindings)
             == tuple(sorted({binding.assembly_id for binding in bindings})),
             'certificate assembly inventory is not canonical and unique')
    _require(projection.replay_sha256 == replay.sha256,
             'certificate projection differs from its current replay')
    poses = {pose.assembly_id: pose for pose in replay.assemblies}
    _require(len(poses) == len(replay.assemblies) and set(poses) == {b.assembly_id for b in bindings},
             'certificate assembly pose denominator differs')
    well = replay.well_access
    _require(well is not None, 'certificate has no current well-access replay')
    well.validate()
    actual = dict(replay.stamp_artwork_records)
    projected = dict(projection.block_rows)
    _require(len(actual) == len(replay.stamp_artwork_records), 'certificate actual stamp inventory repeats')
    partition = Counter(stamp for _, stamps in projection.stamp_rows for stamp in stamps)
    partition.update(stamp for _, stamp in projection.ring_rows)
    _require(partition == Counter({name: 1 for name in actual}),
             'certificate projection lost complete actual stamp partition')
    stamps, rings = dict(projection.stamp_rows), dict(projection.ring_rows)
    for name, box in projection.block_rows:
        joint_checkpoint()()
        physical = (*stamps[name], *((rings[name],) if name in rings else ()))
        _require(box == _bbox(tuple(record for stamp in physical for record in actual[stamp])),
                 'certificate projection bbox differs from actual artwork')
    _require(getattr(certificate, 'materialized_block_bboxes', ()) == projection.block_rows,
             'certificate materialized bbox inventory differs from layered projection')
    master_rows = getattr(certificate, 'materialized_block_members', ())
    masters = dict(getattr(certificate, 'blocks', ()))
    _require(type(master_rows) is tuple and {row[0] for row in master_rows} == set(masters)
             and Counter(member for _, members in master_rows for member in members)
             == Counter({name: 1 for name in projected}),
             'certificate master members do not partition layered physical blocks')
    for master, members in master_rows:
        outer = masters[master]
        for member in members:
            box = projected[member]
            _require(outer[0] <= box[0] < box[2] <= outer[2]
                     and outer[1] <= box[1] < box[3] <= outer[3],
                     'certificate master bbox does not contain current layered member')
    source_rows = getattr(certificate, 'materialized_block_source_members', ())
    _require(type(source_rows) is tuple and len(source_rows) == len(projected)
             and {row[0] for row in source_rows} == set(projected),
             'certificate lacks complete materialized source member denominator')
    source_members = Counter(member for _, members in source_rows for member in members)
    _require(source_members == Counter({row[0]: 1 for row in well.source_member_inventory}),
             'certificate source members differ from full canonical source inventory')
    source_by_block = dict(source_rows)
    block_by_assembly = dict(projection.assembly_block_ids)
    _require(set(block_by_assembly) == set(poses), 'certificate assembly block denominator differs')
    seen_members = set()
    for binding in bindings:
        joint_checkpoint()()
        pose = poses[binding.assembly_id]
        _require((pose.binding_sha256, pose.row_stamp_name, pose.owned_stamp_names, pose.source_members)
                 == (binding.sha256, binding.row_stamp_name, binding.owned_stamp_names, binding.source_members),
                 'certificate assembly pose differs from captured source ownership')
        _require((binding.source_sha256, binding.source_inventory_sha256,
                  binding.source_member_inventory, binding.technology_identity)
                 == (well.source_sha256, well.source_inventory_sha256,
                     well.source_member_inventory, well.technology_identity),
                 'certificate current source/PDK differs from assembly binding')
        _require(not seen_members.intersection(binding.source_members), 'certificate repeats assembly source members')
        seen_members.update(binding.source_members)
        block = block_by_assembly[binding.assembly_id]
        _require(source_by_block[block] == binding.source_members
                 and stamps[block] == binding.owned_stamp_names and block not in rings,
                 'certificate assembly physical/source ownership differs')
        for name, records in binding.owned_artwork_records:
            _require(name in actual, 'certificate current assembly child is missing')
            _same(actual[name], _move(records, pose.translation_nm),
                  'certificate assembly child artwork differs from common pose')
    _require(tuple(sorted(seen_members)) == well.covered_members,
             'certificate well-access denominator differs from assembly sources')
    return dict(layered_placement_assemblies=tuple(binding.sha256 for binding in bindings),
                layered_placement_replay=replay.sha256, layered_placement_projection=projection.sha256)
