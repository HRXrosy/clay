"""Measure a complete final observation against declared reference targets.

The observed certificate owns rigid movement; each logical artwork bbox owns
its module measurement. Source authenticity and structural acceptance remain
independent producer checks. This arithmetic does not rerun placement or signoff.
"""
from clay.layout.place.final_quality_observation import FinalQualityGeometryObservation
from clay.layout.reference_quality_contract import (
    RigidOccurrence, RigidMaterializedBlock, RigidQualityGeometryIR,
)
from clay.layout.reference_quality_metrics import evaluate_reference_quality


def project_final_quality_geometry(observation):
    """Retain the actual final master owners, using source artwork for metrics."""
    if type(observation) is not FinalQualityGeometryObservation:
        raise TypeError('quality projection requires an exact final observation')
    observation.__post_init__()
    if not observation.source_artwork_bboxes_nm:
        raise ValueError('complete owned artwork observation is required for current quality metrics')
    artwork = dict(observation.source_artwork_bboxes_nm)
    state = observation.canonical_data()['final_placement_state']
    masters = {name: tuple(box) for name, box in state['blocks']}
    master_of = {physical: master for master, physicals in state['materialized_block_members']
                 for physical in physicals}
    logical_owner = {logical: master_of[physical]
                     for physical, logicals in state['materialized_block_source_members']
                     for logical in logicals}
    positions = {name: box[:2] for name, box in masters.items()}
    occurrences, artworks = [], []
    for row in observation.logical_geometry:
        owner = logical_owner[row.logical_id]
        dx, dy = positions[owner]
        x0, y0, x1, y1 = artwork[row.logical_id]
        artworks.append(RigidMaterializedBlock(row.logical_id, owner,
                        (x0-dx, y0-dy, x1-dx, y1-dy), (row.logical_id,)))
        occurrences.extend(RigidOccurrence(row.logical_id, owner, (x-dx, y-dy), orient)
                           for x, y, orient in row.occurrences)
    geometry = RigidQualityGeometryIR(
        tuple((name, box[2]-box[0], box[3]-box[1]) for name, box in sorted(masters.items())),
        tuple(sorted(occurrences, key=lambda r: (r.logical_id, r.center_offset_nm, r.orientation, r.owner_id))),
        tuple(sorted(artworks, key=lambda r: r.block_id)))
    return geometry, positions


def evaluate_final_quality_observation(*, observation, contract,
                                     required_structure_status='unverified',
                                     current_structure_status='unverified'):
    """Bind all metric results to both inputs; omitted structure stays a blocker."""
    geometry, positions = project_final_quality_geometry(observation)
    state = observation.canonical_data()['final_placement_state']
    evaluation = evaluate_reference_quality(positions=positions, geometry=geometry,
        contract=contract, frame_nm=tuple(state['bbox_nm']),
        required_structure_status=required_structure_status,
        current_structure_status=current_structure_status)
    return {'schema': 'clay.final-reference-quality-evaluation/v1',
            'observation_sha256': observation.sha256, 'contract_sha256': contract.sha256,
            'final_placement_state_sha256': observation.final_placement_state_sha256,
            'geometry_projection_sha256': geometry.sha256, 'evaluation': evaluation}
