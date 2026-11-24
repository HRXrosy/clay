"""Complete quality targets from one live current reference observation.

The current producer owns the explicitly selected comparison frame, while its
full raw polygon frame remains evidence. This adapter grants no acceptance.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from clay.frontend.reference_device_artwork import CurrentReferenceDeviceArtwork
from clay.layout.reference_quality_contract import ReferenceSourceAuthority
from clay.layout.reference_quality_targets import ObservedReferenceOccurrence, build_reference_quality_contract
from clay.layout.reference_structure_plan import SourceUnitRequirement, compile_reference_structure_plan
from clay.layout.source_quality_groups import source_quality_groups


def build_current_reference_quality(source_cell, *, reference, repo_root):
    if type(reference) is not CurrentReferenceDeviceArtwork:
        raise TypeError('quality targets require the current reference producer')
    reference.require_current()
    root = Path(repo_root).resolve(strict=True)
    data = reference.canonical_data()
    mapping = tuple((row.occurrence_id, row.logical_id, row.parameter_class,
        row.source_terminals, row.physical_terminals) for row in reference.records)
    mapping_sha = hashlib.sha256(json.dumps((mapping, reference.net_to_node),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    artifacts = []
    for path, sha in sorted(data['input_sha256'].items()):
        resolved = Path(path).resolve(strict=True)
        if resolved.is_relative_to(root):
            relative = resolved.relative_to(root).as_posix()
            artifacts.append((relative, relative, sha))
    identity = data['pdk_identity']
    authority = ReferenceSourceAuthority(tuple(sorted(artifacts)), *identity,
        tuple(sorted({tuple(key) for _name, key in data['semantic_layers']})),
        mapping_sha, tuple(map(tuple, data['implementation'])))
    observed = tuple(ObservedReferenceOccurrence(row.occurrence_id, row.logical_id,
        row.parameter_class, row.owned_bbox_nm, row.orientation,
        tuple(value//2 for value in row.center2_nm)) for row in reference.records)
    contract = build_reference_quality_contract(source_authority=authority,
        reference_bbox_nm=tuple(data['comparison_bbox_nm']), occurrences=observed,
        group_members=source_quality_groups(source_cell))
    classes = {row.logical_id:(row.family, row.parameter_class) for row in reference.records}
    counts = Counter(row.logical_id for row in reference.records)
    if set(classes) != {device.instance_name for device in source_cell.devices}:
        raise ValueError('quality source population differs from current reference')
    requirements = tuple(SourceUnitRequirement(name, *classes[name], counts[name])
                         for name in sorted(classes))
    plan = compile_reference_structure_plan(source_sha256=data['source_sha256'],
        requirements=requirements, observed=observed, contract=contract)
    reference.require_current()
    return contract, plan
