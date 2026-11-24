"""Pure quality targets from an explicitly joined reference population.

This boundary contains no circuit names, file readers or ownership heuristics.
The producer authenticates original inputs and establishes the complete join;
these records and arithmetic alone do not grant source authority or acceptance.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from clay.layout.reference_quality_contract import (
    Box, ORIENTATIONS, ReferenceSourceAuthority, ReferenceQualityContract,
    ReferenceOccurrence, LogicalPopulation, ClassPopulation, LocationTarget,
    LogicalPatternTarget, ClassPatternTarget, ModuleTarget, ModuleRelationTarget,
    _check_record, _name, _box, _require, _module_cover,
)
from clay.layout.reference_quality_metrics import (
    normalize_point, normalized_population, physical_class_pattern, rank_pattern,
)


@dataclass(frozen=True, slots=True)
class ObservedReferenceOccurrence:
    """Owned artwork and an independently measured primitive center.

    Odd coordinate sums use the existing integer geometry metric convention;
    this is not a claim that a half-nm geometric center lies on the nm grid.
    An omitted center retains that convention. An explicit channel/unit center
    does not move or shrink its asymmetric physical artwork envelope.
    """
    occurrence_id: str
    logical_id: str
    parameter_class: str
    bbox_nm: Box
    orientation: str
    center_nm: tuple[int, ...] = ()

    def __post_init__(self):
        _check_record(self)
        for text in (self.occurrence_id, self.logical_id, self.parameter_class):
            _name(text)
        _box(self.bbox_nm, positive=True)
        _require(self.orientation in ORIENTATIONS, 'reference orientation differs')
        _require(not self.center_nm or (len(self.center_nm) == 2
            and self.bbox_nm[0] <= self.center_nm[0] <= self.bbox_nm[2]
            and self.bbox_nm[1] <= self.center_nm[1] <= self.bbox_nm[3]),
            'reference center must lie inside its owned artwork')

    @property
    def representative_center_nm(self):
        return self.center_nm or _center(self.bbox_nm)


def _center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _normalized_box(box, frame):
    return (*normalize_point(box[:2], frame), *normalize_point(box[2:], frame))


def _join(occurrences, group_members):
    _require(type(occurrences) is tuple and occurrences,
             'reference occurrences must be a nonempty exact immutable tuple')
    for row in occurrences:
        _require(type(row) is ObservedReferenceOccurrence, 'reference occurrence type differs')
        row.__post_init__()
    _require(len({r.occurrence_id for r in occurrences}) == len(occurrences),
             'duplicate observed occurrence identity')
    _require(type(group_members) is tuple and group_members,
             'source module cover must be immutable and nonempty')
    names, groups = [], []
    for entry in group_members:
        _require(type(entry) is tuple and len(entry) == 2, 'module entry must be an exact pair')
        group, members = entry
        _require(type(group) is str, 'module identity must be an exact string')
        _name(group)
        _require(type(members) is tuple and members and all(type(n) is str for n in members),
                 'module members must be an immutable nonempty string tuple')
        _require(members == tuple(sorted(set(members))), 'module members repeat or are unordered')
        names.extend(members)
        groups.append(group)
    _require(groups == sorted(set(groups)), 'module identities repeat or are unordered')
    _module_cover(group_members, {r.logical_id for r in occurrences}, hierarchical=True)
    by_logical = {name: tuple(sorted((r for r in occurrences if r.logical_id == name),
                                    key=lambda r: r.occurrence_id)) for name in sorted(set(names))}
    _require(all(len({r.parameter_class for r in rows}) == 1 for rows in by_logical.values()),
             'logical occurrence parameter class is ambiguous')
    return by_logical


def build_reference_quality_contract(*, source_authority, reference_bbox_nm,
                                     occurrences, group_members, domain_policy=()):
    """Normalize every observed occurrence; retain exact ranks and class labels.

    Groups must completely cover logical IDs, with overlaps confined to nested
    parent/child declarations. Every supplied group retains its own target and
    every pair retains both relation checks; parents add obligations. Parameter
    classes are the producer's resolved semantic classes, independent of raw
    layout master spelling. Each bbox midpoint is floored to nm before the
    existing per-occurrence ppm normalization. Class row clustering follows the metric's
    half-minimum-observed-height convention. Public thresholds remain unchanged.
    """
    _require(type(source_authority) is ReferenceSourceAuthority, 'reference source authority type differs')
    source_authority.__post_init__()
    _require(type(reference_bbox_nm) is tuple and len(reference_bbox_nm) == 4
             and all(type(v) is int for v in reference_bbox_nm), 'reference frame must be integer nm')
    _box(reference_bbox_nm, positive=True)
    joined = _join(occurrences, group_members)
    _require(all(reference_bbox_nm[0] <= r.bbox_nm[0] < r.bbox_nm[2] <= reference_bbox_nm[2]
                 and reference_bbox_nm[1] <= r.bbox_nm[1] < r.bbox_nm[3] <= reference_bbox_nm[3]
                 for r in occurrences), 'reference occurrence escapes its full frame')
    reference_rows = tuple(ReferenceOccurrence(r.occurrence_id, r.logical_id, r.parameter_class,
        normalize_point(r.representative_center_nm, reference_bbox_nm),
        _normalized_box(r.bbox_nm, reference_bbox_nm), r.orientation)
        for r in sorted(occurrences, key=lambda r: r.occurrence_id))
    logicals, patterns, locations = [], [], []
    points = {name: tuple(r.representative_center_nm for r in rows) for name, rows in joined.items()}
    for name, rows in joined.items():
        logicals.append(LogicalPopulation(name, rows[0].parameter_class, tuple(r.occurrence_id for r in rows)))
        locations.append(LocationTarget('logical', name, *normalized_population(points[name], reference_bbox_nm)))
        patterns.append(LogicalPatternTarget(name, rank_pattern(points[name]),
                        tuple(sorted(Counter(r.orientation for r in rows).items()))))
    classes, class_patterns = [], []
    for label in sorted({r.parameter_class for r in occurrences}):
        names = tuple(name for name, rows in joined.items() if rows[0].parameter_class == label)
        rows = tuple(r for r in occurrences if r.parameter_class == label)
        classes.append(ClassPopulation(label, names, tuple(sorted(r.occurrence_id for r in rows))))
        locations.append(LocationTarget('class', label,
            *normalized_population(tuple(p for name in names for p in points[name]), reference_bbox_nm)))
        tolerance = min(r.bbox_nm[3] - r.bbox_nm[1] for r in rows) // 2
        pattern, _, _ = physical_class_pattern({n: points[n] for n in names}, row_stagger_tolerance_nm=tolerance)
        class_patterns.append(ClassPatternTarget(label, pattern, tolerance,
                              tuple(sorted(Counter(r.orientation for r in rows).items()))))
    modules = []
    for name, members in group_members:
        rows = tuple(r for r in occurrences if r.logical_id in members)
        box = _union(tuple(r.bbox_nm for r in rows))
        modules.append(ModuleTarget(name, members, tuple(sorted(r.occurrence_id for r in rows)),
                       normalize_point(_center(box), reference_bbox_nm), _normalized_box(box, reference_bbox_nm)))
    relations = tuple(ModuleRelationTarget(a.module_id, b.module_id, axis,
                      'before' if a.centroid_ppm[i] < b.centroid_ppm[i] else
                      'after' if a.centroid_ppm[i] > b.centroid_ppm[i] else 'coincident')
                      for a, b in combinations(modules, 2)
                      for i, axis in enumerate(('horizontal', 'vertical')))
    memberships = tuple(name for _group, members in group_members for name in members)
    schema = ('clay.reference-quality-contract/v1' if len(memberships) == len(set(memberships))
              else 'clay.reference-quality-contract/v2')
    return ReferenceQualityContract(source_authority, reference_bbox_nm, reference_rows,
        tuple(logicals), tuple(classes), tuple(sorted(locations, key=lambda r: (r.scope, r.target_id))),
        tuple(patterns), tuple(class_patterns), tuple(modules), relations, domain_policy=domain_policy,
        schema=schema)
