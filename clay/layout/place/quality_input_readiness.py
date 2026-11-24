"""Inspect all final quality operand domains before a complete comparison.

This is a diagnostic inventory, never a reference-quality or physical verdict.
Independent stamps are inspected together so one missing provider does not hide
the next. Complete observation still uses its original strict consumer.
"""
from clay.layout.place.final_placement_state import final_placement_state_digest
from clay.layout.place.final_quality_observation import _kinds
from clay.layout.place.placement_certificates import FinalPlacementCertificate
from clay.layout.place.placement_geometry import _source_device_bboxes_nm
from clay.layout.place.reference_quality_geometry import collect_live_source_occurrences
from clay.layout.owned_artwork import quality_source_artwork_bboxes_nm


def check_quality_inputs(*, geometry, placement_um, logical_kinds, certificate=None):
    """Inspect a live materialized source, optionally bound to its final state."""
    if certificate is not None and type(certificate) is not FinalPlacementCertificate:
        raise TypeError('quality input check requires the actual final certificate')
    kinds = dict(_kinds(logical_kinds))
    names = tuple(kinds)
    digest = None if certificate is None else final_placement_state_digest(certificate)
    blockers, occurrences, artwork = [], {}, {}

    def blocked(domain, sources, error, stamp=None):
        blockers.append(dict(domain=domain, sources=list(sources), stamp=stamp,
                             error_type=type(error).__name__, error=str(error)))

    boxes = _source_device_bboxes_nm(geometry, placement_um, names)
    if certificate is not None:
        owned = tuple(sorted(n for _block, members in certificate.materialized_block_source_members
                             for n in members))
        if owned != names:
            blocked('source_ownership', names, ValueError('complete certified logical partition differs'))
        if tuple(sorted(boxes.items())) != certificate.materialized_source_member_bboxes:
            blocked('source_geometry', names, ValueError('actual source bboxes differ from certificate'))
    for stamp_name, stamp in sorted(geometry.items()):
        raw_members = getattr(stamp, 'composite_members', ())
        if type(raw_members) is not tuple or any(type(n) is not str for n in raw_members):
            blocked('stamp_ownership', names, ValueError('malformed composite members'), stamp_name)
            continue
        selected = tuple(sorted(set(raw_members or (stamp_name,)) & set(names)))
        if not selected:
            continue
        selected_geometry = {stamp_name: stamp}
        for domain in ('occurrences', 'owned_artwork'):
            try:
                if domain == 'occurrences':
                    rows = collect_live_source_occurrences(geometry=selected_geometry,
                        placement_um=placement_um, logical_kinds={n: kinds[n] for n in selected},
                        source_device_bboxes={n: boxes[n] for n in selected})
                    target = occurrences
                else:
                    rows = quality_source_artwork_bboxes_nm(selected_geometry, placement_um, selected)
                    target = artwork
                if set(rows) & set(target):
                    raise ValueError('logical sources have multiple current stamps')
                target.update(rows)
            except (ValueError, TypeError, KeyError) as error:
                blocked(domain, selected, error, stamp_name)
    missing_occurrences = sorted(set(names)-set(occurrences))
    missing_artwork = sorted(set(names)-set(artwork))
    if not blockers and (missing_occurrences or missing_artwork):
        blocked('complete_inventory', names, ValueError('source members have no observed stamp'))
    if certificate is not None and final_placement_state_digest(certificate) != digest:
        raise ValueError('final certificate changed during quality input inspection')
    return dict(schema='clay.quality-input-readiness/v1',
        status='blocked' if blockers else 'ready', physical_admission=False,
        scope='input_availability_only', final_placement_state_sha256=digest,
        logical_count=len(names), observed_occurrence_count=sum(len(v) for v in occurrences.values()),
        missing_occurrence_sources=missing_occurrences, missing_artwork_sources=missing_artwork,
        observed_occurrences={n: [list(r) for r in rows] for n, rows in sorted(occurrences.items())},
        observed_artwork_bboxes_nm={n: list(box) for n, box in sorted(artwork.items())}, blockers=blockers)
