"""Necessary frame dimensions from rigid artwork and explicit quality targets.

This pure arithmetic boundary neither places owners nor tests collision or
source authority. An empty domain disproves only this current rigid construction
within the supplied ceiling; a nonempty domain is not a feasibility result.
"""
from collections import Counter
from fractions import Fraction

from clay.layout.reference_quality_contract import ReferenceQualityContract, RigidQualityGeometryIR


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _ratio(value):
    return [value.numerator, value.denominator]


def _inputs(geometry, contract):
    if type(geometry) is not RigidQualityGeometryIR or type(contract) is not ReferenceQualityContract:
        raise TypeError('frame bounds require exact current geometry IR and reference contract')
    geometry.__post_init__()
    contract.__post_init__()
    expected = {row.logical_id:len(row.occurrence_ids) for row in contract.logical_population}
    _require(dict(Counter(row.logical_id for row in geometry.occurrences)) == expected,
        'current logical/multiplicity population differs from reference contract')
    points = {name:tuple(row for row in geometry.occurrences if row.logical_id == name) for name in expected}
    populations = {('logical', row.logical_id):(row.logical_id,) for row in contract.logical_population}
    populations.update({('class', row.class_id):row.logical_ids for row in contract.class_population})
    modules = {}
    for target in contract.modules:
        logicals = set(target.logical_ids)
        selected = tuple(row for row in geometry.materialized_blocks if logicals.intersection(row.logical_ids))
        _require(all(set(row.logical_ids) <= logicals for row in selected)
            and {name for row in selected for name in row.logical_ids} == logicals,
            'current materialized module logical partition differs from reference contract: ' + target.module_id)
        modules[target.module_id] = selected
    return points, populations, modules


def necessary_reference_frame_bounds(geometry, contract, *, grid_nm, max_span_nm=1000000):
    """Return JSON-ready closed W/H limits, exact witnesses and contradictions.

Only single-owner populations have a fixed intrinsic span or a common unknown
translation. Multi-owner class/module spans are explicitly skipped. All source
logical counts and module block partitions are still required before skipping.

For envelopes, the reference span is expanded by twice the contract tolerance
plus 1 ppm for the two rounded endpoints. For centroid coordinates, the existing
Euclidean limit is relaxed to per-axis limits plus 2 ppm for normalization and
mean rounding. Every resulting bound is necessary; no quality tolerance changes.
The normalization proof is limited to integer frame spans at most 1,000,000 nm,
matching the quality compiler. Larger caller ceilings are unsupported inputs.
"""
    for name, value in (('grid', grid_nm), ('span ceiling', max_span_nm)):
        _require(type(value) is int and value > 0, name + ' must be a positive exact integer nm')
    _require(max_span_nm <= 1000000, 'span ceiling exceeds the proved normalization domain')
    points, populations, modules = _inputs(geometry, contract)
    witnesses, contradictions, skipped = [], [], []

    def add(**row):
        row['id'] = 'w' + str(len(witnesses)).zfill(6)
        witnesses.append(row)
        return row['id']

    def skip(scope, target_id, owner_ids):
        skipped.append(dict(scope=scope, target_id=target_id, owner_ids=sorted(owner_ids),
            reason='multiple_rigid_owners'))

    def envelope(scope, target_id, owner_id, box, reference):
        margin = 2*contract.envelope_limit_ppm + 1
        for axis, label in enumerate(('x', 'y')):
            span, delta = box[axis+2]-box[axis], reference[axis+2]-reference[axis]
            lower = Fraction(span*1000000, delta+margin)
            upper = Fraction(span*1000000, delta-margin) if delta > margin else None
            add(kind='envelope_span', scope=scope, target_id=target_id, owner_id=owner_id, axis=label,
                target_names=[scope+':'+target_id], span_nm=span, reference_span_ppm=delta,
                lower_nm=_ceil(lower), upper_nm=None if upper is None else upper.numerator//upper.denominator)

    for axis, label in enumerate(('x', 'y')):
        add(kind='span_ceiling', axis=label, target_names=['max_span_nm'], lower_nm=1, upper_nm=max_span_nm)
        for owner, width, height in geometry.owners:
            add(kind='owner_extent', owner_id=owner, axis=label, target_names=['owner:'+owner],
                lower_nm=(width, height)[axis], upper_nm=None)

    centroids = {owner:[] for owner, _width, _height in geometry.owners}
    for target in contract.location_targets:
        selected = tuple(row for name in populations[target.scope, target.target_id] for row in points[name])
        owner_ids = {row.owner_id for row in selected}
        if len(owner_ids) != 1:
            skip(target.scope, target.target_id, owner_ids)
            continue
        owner, = owner_ids
        box = tuple(function(row.center_offset_nm[axis] for row in selected)
            for function in (min, max) for axis in (0, 1))
        envelope(target.scope, target.target_id, owner, box, target.envelope_ppm)
        center = tuple(Fraction(sum(row.center_offset_nm[axis] for row in selected), len(selected))
            for axis in (0, 1))
        centroids[owner].append((target.scope+':'+target.target_id, center, target.centroid_ppm))

    for target in contract.modules:
        selected = modules[target.module_id]
        owner_ids = {row.owner_id for row in selected}
        if len(owner_ids) != 1:
            skip('module', target.module_id, owner_ids)
            continue
        owner, = owner_ids
        box = tuple(function(row.bbox_offset_nm[axis] for row in selected)
            for function, axes in ((min, (0, 1)), (max, (2, 3))) for axis in axes)
        envelope('module', target.module_id, owner, box, target.envelope_ppm)

    radius = contract.centroid_limit_ppm + 2
    for owner, width, height in geometry.owners:
        for axis, label in enumerate(('x', 'y')):
            # Each pair eliminates the same owner's unknown physical origin:
            # lower_a*span + lower_b <= origin <= upper_a*span + upper_b.
            lower = [(Fraction(0), Fraction(0), 'physical_min')]
            upper = [(Fraction(1), Fraction(-(width, height)[axis]), 'physical_max')]
            for name, center, reference in centroids[owner]:
                lower.append((Fraction(reference[axis]-radius, 1000000), -center[axis], name))
                upper.append((Fraction(reference[axis]+radius, 1000000), -center[axis], name))
            for la, lb, ln in lower:
                for ua, ub, un in upper:
                    a, b = la-ua, ub-lb
                    low, high = 0, None
                    if a < 0:
                        low = max(0, _ceil(b/a))
                    elif a > 0:
                        bound = b/a
                        high = bound.numerator//bound.denominator
                    identity = add(kind='centroid_origin_pair', owner_id=owner, axis=label,
                        lower_target=ln, upper_target=un, target_names=[ln, un],
                        inequality_a=_ratio(a), inequality_b=_ratio(b), lower_nm=low, upper_nm=high)
                    if a == 0 and b < 0:
                        contradictions.append(dict(kind='constant_inequality', owner_id=owner, axis=label,
                            target_names=[ln, un], witness_ids=[identity]))

    limits = {}
    for label in ('x', 'y'):
        selected = [row for row in witnesses if row['axis'] == label]
        raw_low = max(row['lower_nm'] for row in selected)
        raw_high = min(row['upper_nm'] for row in selected if row['upper_nm'] is not None)
        lower = ((raw_low+grid_nm-1)//grid_nm)*grid_nm
        upper = (raw_high//grid_nm)*grid_nm
        lows = [row for row in selected if row['lower_nm'] == raw_low]
        highs = [row for row in selected if row['upper_nm'] == raw_high]
        limits[label] = dict(lower_nm=lower, upper_nm=upper,
            lower_witnesses=[row['id'] for row in lows], upper_witnesses=[row['id'] for row in highs])
        if lower > upper:
            edges = {row['id']:row for row in (*lows, *highs)}
            contradictions.append(dict(kind='empty_grid_domain', axis=label,
                target_names=sorted({name for row in edges.values() for name in row['target_names']}),
                witness_ids=list(edges)))

    return dict(schema='clay.necessary-reference-frame-bounds/v1',
        scope='NECESSARY_ONLY_NOT_FEASIBILITY_OR_ADMISSION',
        status='CONTRADICTION' if contradictions else 'NONEMPTY_NECESSARY_DOMAIN',
        geometry_sha256=geometry.sha256, contract_sha256=contract.sha256,
        grid_nm=grid_nm, max_span_nm=max_span_nm,
        normalization_outer_margin_ppm=dict(envelope=1, centroid=2),
        limits=limits, witnesses=witnesses, contradictions=contradictions,
        skipped_targets=skipped)
