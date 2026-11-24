"""Measure complete actual stamp ownership at final geometry consumers.

This joins current polygons and typed accessory metadata to an existing native
certificate. It neither rebuilds artwork from certificate boxes nor grants new
source, body-access, execution, or signoff authority. Actual block identifiers
remain extractor identifiers; consumers may join them to native IDs by sources.
"""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ActualMaterializedPartition:
    blocks: tuple
    block_stamps: dict
    block_gring: dict
    gring_names: set
    tap_names: frozenset
    source_members: tuple


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def actual_materialized_partition(certificate, geometry, positions, intent):
    """Extract, verify and project one complete actual materialized scene."""
    from clay.layout.compact_blocks import extract_blocks
    from clay.layout.layered_placement_assembly import (
        _markers, _same, project_layered_placement_assemblies,
    )
    from clay.layout.owned_artwork import _stamp_records
    from clay.layout.place.layered_assembly_certificate import layered_assembly_certificate_payload

    layered_assembly_certificate_payload(certificate)
    markers = _markers(geometry)
    children = tuple(child for binding in certificate.layered_placement_assemblies
                     for child in binding.tap_children)
    _require(len({child.stamp_name for child in children}) == len(children),
             'final typed tap ownership repeats a child')
    _require({name: marker.payload() for name, marker in markers.items()} ==
             {child.stamp_name: child.payload() for child in children},
             'final actual tap marker ledger differs from required ownership binding')
    blocks, stamps, rings, ring_names = extract_blocks(geometry, positions, intent)
    if certificate.layered_placement_assemblies:
        replay = certificate.layered_placement_replay
        recorded = dict(replay.stamp_artwork_records)
        _require(set(recorded) == set(geometry), 'final layered actual stamp cover differs')
        for name, stamp in geometry.items():
            _same(recorded[name], _stamp_records(stamp, positions[name]),
                  'final layered stamp artwork differs from the exact certificate scene')
        projected = project_layered_placement_assemblies(replay, blocks=tuple(blocks),
            block_stamps=stamps, block_gring=rings, gring_names=ring_names)
        _require(projected.sha256 == certificate.layered_placement_projection.sha256,
                 'final layered actual block ownership differs')
        blocks, stamps, rings, ring_names = (projected.blocks, projected.block_stamps,
                                             projected.block_gring, projected.gring_names)
    partition = Counter(name for members in stamps.values() for name in members)
    partition.update(rings.values())
    _require(partition == Counter({name: 1 for name in geometry})
             and set(rings.values()) == set(ring_names),
             'final actual stamp partition is incomplete or repeated')
    ownership = []
    for block, _box in blocks:
        sources = []
        for name in sorted(stamps[block]):
            if name in markers:
                continue  # Only the complete verified typed ledger grants this role.
            members = getattr(geometry[name], 'composite_members', ()) or (name,)
            _require(type(members) is tuple and all(type(member) is str and member for member in members),
                     'final actual source member inventory is malformed')
            sources.extend(members)
        _require(sources and len(sources) == len(set(sources)),
                 'final actual physical source ownership is empty or repeated')
        ownership.append((block, tuple(sorted(sources))))
    expected = tuple(sorted(source for _block, sources in certificate.materialized_block_source_members
                            for source in sources))
    actual = tuple(sorted(source for _block, sources in ownership for source in sources))
    _require(expected and len(expected) == len(set(expected)) and actual == expected,
             'final actual source ownership has missing, repeated or foreign members')
    return ActualMaterializedPartition(tuple(blocks), stamps, rings, ring_names,
                                       frozenset(markers), tuple(sorted(ownership)))
