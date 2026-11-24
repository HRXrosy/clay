"""Derive actual joint-guard ownership without replacing native device proof.

The original collision graph remains mandatory for every device mask. Actual
joint guard polygons are proved against their typed envelope and complete
foreign scene; source ownership and polygon inventories remain exact.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SharedGuardArtwork:
    group: str
    actual_id: str
    guard_stamp: str
    native_ids: tuple
    masters: tuple
    sources: tuple
    bbox_nm: tuple
    foreign_clearances_nm: tuple


def shared_guard_partition(certificate, snapshot, partition, stamp_bboxes):
    native_sources = dict(certificate.materialized_block_source_members)
    native_boxes = dict(certificate.materialized_block_bboxes)
    native_by_sources = {sources: name for name,sources in native_sources.items()}
    master_by_native = {native: master for master, children in certificate.materialized_block_members
                        for native in children}
    sources_by_actual = dict(partition.source_members)
    actual_boxes = dict(partition.blocks)
    stamps = {stamp.name: stamp for stamp in snapshot.stamps}
    groups = []
    replacements = {}
    for actual_id, sources in partition.source_members:
        if sources in native_by_sources:
            continue
        matches = []
        for projection in certificate.flexible_envelope_projections:
            native_ids = tuple(sorted(native for native, master in master_by_native.items()
                                      if master in projection.constraint.members))
            expected = tuple(sorted(source for native in native_ids for source in native_sources[native]))
            if expected == sources:
                matches.append((projection, native_ids))
        if len(matches) != 1:
            raise ValueError('actual shared guard requires one exact typed source partition')
        projection, native_ids = matches[0]
        constraint = projection.constraint
        if constraint.foreign_exclusion_mode != 'envelope':
            raise ValueError('actual joint guard requires complete envelope collision reservation')
        guard = partition.block_gring.get(actual_id)
        if guard not in stamps:
            raise ValueError('actual shared guard is missing its emitted guard stamp')
        guard_stamp = stamps[guard]
        if (guard_stamp.source != 'group_guard_ring' or guard_stamp.guard_members != sources
                or not guard_stamp.guard_sha256
                or (sources, guard_stamp.guard_sha256) not in snapshot.guards):
            raise ValueError('actual shared guard declaration ownership differs')
        if actual_boxes[actual_id] != projection.bbox_nm or stamp_bboxes[guard] != projection.bbox_nm:
            raise ValueError('actual shared guard differs from the certified physical envelope')
        box = projection.bbox_nm
        clearances = []
        for foreign, other in partition.blocks:
            if foreign == actual_id:
                continue
            clearance = max(other[0]-box[2], box[0]-other[2], other[1]-box[3], box[1]-other[3])
            if clearance < constraint.foreign_clearance_nm:
                raise ValueError('actual shared guard envelope contains foreign physical geometry')
            clearances.append((foreign, clearance))
        assigned = {native: [] for native in native_ids}
        actual_source_cover = {native: [] for native in native_ids}
        for name in partition.block_stamps[actual_id]:
            if name in partition.tap_names:
                raise ValueError('shared guard tap replacement requires explicit typed assembly support')
            own = stamps[name].members or (name,)
            owners = [native for native in native_ids if set(own) <= set(native_sources[native])]
            if len(owners) != 1:
                raise ValueError('actual shared device stamp crosses native source owners')
            owner = owners[0]
            bbox = stamp_bboxes[name]
            authority = native_boxes[owner]
            if not (authority[0] <= bbox[0] < bbox[2] <= authority[2]
                    and authority[1] <= bbox[1] < bbox[3] <= authority[3]):
                raise ValueError('actual shared device escapes its native physical frame')
            assigned[owner].append(name)
            actual_source_cover[owner].extend(own)
        for native in native_ids:
            if tuple(sorted(actual_source_cover[native])) != native_sources[native]:
                raise ValueError('actual shared device masks lack complete native source coverage')
        replacements[actual_id] = tuple((native, tuple(sorted(assigned[native]))) for native in native_ids)
        groups.append(SharedGuardArtwork(constraint.group, actual_id, guard, native_ids,
            constraint.members, sources, box, tuple(sorted(clearances))))
    return tuple(sorted(groups, key=lambda group: group.group)), replacements
