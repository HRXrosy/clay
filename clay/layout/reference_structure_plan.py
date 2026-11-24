"""Retain complete reference quality observations before construction planning.

Plans retain ranks, occupied cells and each unit's orientation/identity. They
contain no placement coordinates and do not select an artwork variant or infer
terminal roles. A producer must authenticate source requirements and reference
inputs; geometry binding and native consumption remain subsequent obligations.
Exact center ranks are not uniform-pitch construction cells. In particular,
different artwork variants can shift centers without adding a physical column.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json

from clay.layout.reference_quality_contract import ReferenceQualityContract, _module_cover
from clay.layout.reference_quality_targets import build_reference_quality_contract


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    _require(type(value) is str and value and value == value.strip(), 'noncanonical structure identity')


def _sha(value):
    _require(type(value) is str and len(value) == 64
             and all(c in '0123456789abcdef' for c in value), 'noncanonical structure SHA-256')


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceUnitRequirement:
    logical_id: str
    kind: str
    parameter_class: str
    unit_count: int

    def __post_init__(self):
        _text(self.logical_id)
        _text(self.parameter_class)
        _require(type(self.kind) is str and self.kind in {
            'nmos', 'pmos', 'capacitor', 'resistor', 'pnp', 'npn'}, 'unsupported source unit kind')
        _require(type(self.unit_count) is int and self.unit_count > 0,
                 'source unit count must be an exact positive integer')


@dataclass(frozen=True, slots=True)
class ObservedUnitRanks:
    """Quality rank equivalence classes; deliberately not a UnitGridPattern.

    Coincident occurrences retain their multiplicity and each orientation.
    Equal rank pairs follow their associated occurrence identities, without
    imposing construction-grid occupancy or an orientation ordering.
    """
    placements: tuple[tuple[int, int, str], ...]

    def __post_init__(self):
        _require(type(self.placements) is tuple and self.placements,
                 'observed rank placements must be a nonempty exact tuple')
        orientations = {'R0', 'R90', 'R180', 'R270', 'MR0', 'MR90', 'MR180', 'MR270'}
        cells = []
        for placement in self.placements:
            _require(type(placement) is tuple and len(placement) == 3,
                     'observed rank placement must be an exact triple')
            row, column, orientation = placement
            _require(type(row) is int and type(column) is int and row >= 0 and column >= 0,
                     'observed ranks must be nonnegative exact integers')
            _require(type(orientation) is str and orientation in orientations,
                     'observed rank orientation is unsupported')
            cells.append((row, column))
        _require(cells == sorted(cells), 'observed ranks must have canonical row-major order')
        for axis in (0, 1):
            ranks = sorted({cell[axis] for cell in cells})
            _require(ranks == list(range(len(ranks))), 'observed rank axes must be normalized')

    @property
    def unit_count(self):
        return len(self.placements)

    @property
    def rows(self):
        return 1 + max(row for row, _column, _orientation in self.placements)

    @property
    def columns(self):
        return 1 + max(column for _row, column, _orientation in self.placements)

    def canonical_data(self):
        self.__post_init__()
        return {'schema': 'clay.observed-unit-ranks/v1',
                'placements': [list(row) for row in self.placements]}


@dataclass(frozen=True, slots=True)
class LogicalUnitPlan:
    requirement: SourceUnitRequirement
    pattern: ObservedUnitRanks
    occurrence_ids: tuple[str, ...]

    def __post_init__(self):
        _require(type(self.requirement) is SourceUnitRequirement
                 and type(self.pattern) is ObservedUnitRanks, 'unit plan requires exact immutable records')
        self.requirement.__post_init__()
        self.pattern.__post_init__()
        _require(type(self.occurrence_ids) is tuple, 'unit plan identities must be immutable')
        for name in self.occurrence_ids:
            _text(name)
        _require(len(set(self.occurrence_ids)) == len(self.occurrence_ids)
                 == self.pattern.unit_count == self.requirement.unit_count,
                 'unit plan population differs from its source')

    def canonical_data(self):
        self.__post_init__()
        return {'source_requirement': asdict(self.requirement),
                'pattern': self.pattern.canonical_data(),
                'occurrence_ids': list(self.occurrence_ids)}


def _source_groups(rows, names, label, *, hierarchical=False):
    _require(type(rows) is tuple and rows, f'{label} requires immutable complete groups')
    keys = []
    for row in rows:
        _require(type(row) is tuple and len(row) == 2, f'{label} group row differs')
        key, members = row
        _text(key)
        _require(type(members) is tuple and members, f'{label} members must be immutable')
        for name in members:
            _text(name)
        _require(members == tuple(sorted(set(members))), f'{label} members are not canonical')
        keys.append(key)
    _require(keys == sorted(set(keys)), f'{label} group identities repeat or are unordered')
    _module_cover(rows, names, hierarchical=hierarchical)


@dataclass(frozen=True, slots=True)
class ReferenceStructurePlan:
    source_sha256: str
    reference_contract_sha256: str
    logical_units: tuple[LogicalUnitPlan, ...]
    parameter_classes: tuple[tuple[str, tuple[str, ...]], ...]
    functional_groups: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self):
        _sha(self.source_sha256)
        _sha(self.reference_contract_sha256)
        _require(type(self.logical_units) is tuple and self.logical_units
                 and all(type(row) is LogicalUnitPlan for row in self.logical_units),
                 'structure plan requires an immutable complete unit inventory')
        for row in self.logical_units:
            row.__post_init__()
        names = [row.requirement.logical_id for row in self.logical_units]
        units = [name for row in self.logical_units for name in row.occurrence_ids]
        _require(names == sorted(set(names)) and len(units) == len(set(units)),
                 'structure plan logical or occurrence ownership repeats')
        _source_groups(self.parameter_classes, names, 'parameter classes')
        _source_groups(self.functional_groups, names, 'functional groups', hierarchical=True)
        declared = {name: key for key, members in self.parameter_classes for name in members}
        _require(all(declared[row.requirement.logical_id] == row.requirement.parameter_class
                     for row in self.logical_units), 'structure plan parameter ownership differs')

    def canonical_data(self):
        self.__post_init__()
        memberships = tuple(name for _group, members in self.functional_groups for name in members)
        schema = ('clay.reference-structure-plan/v2' if len(memberships) == len(set(memberships))
                  else 'clay.reference-structure-plan/v3')
        return {'schema': schema,
                'scope': 'quality_observation_only_requires_construction_and_native_consumption',
                'source_sha256': self.source_sha256,
                'reference_contract_sha256': self.reference_contract_sha256,
                'logical_units': [row.canonical_data() for row in self.logical_units],
                'parameter_classes': [[key, list(names)] for key, names in self.parameter_classes],
                'functional_groups': [[key, list(names)] for key, names in self.functional_groups]}

    @property
    def sha256(self):
        return _digest(self.canonical_data())


def compile_reference_structure_plan(*, source_sha256, requirements, observed, contract):
    """Recheck the full target population, then discard coordinates for structure.

    Observed ranks use the same independently measured representative centers
    as the quality contract, with the original floored-nm bbox-center fallback.
    They do not establish artwork anchors, pitch, physical rows or a primitive
    fit; no uniform-grid or per-logical rigid-owner decision is made here.
    Parameter classes remain a complete partition. Functional groups retain
    their complete nested cover, including every parent/child declaration;
    logical unit ownership and counts are independent of repeated memberships.
    Neither grouping is silently made a rigid physical owner. Cross-owner class
    patterns and normalized targets remain in the separately bound contract.
    """
    _sha(source_sha256)
    _require(type(requirements) is tuple and requirements
             and all(type(row) is SourceUnitRequirement for row in requirements),
             'source requirements must be an immutable complete inventory')
    for row in requirements:
        row.__post_init__()
    _require(len({row.logical_id for row in requirements}) == len(requirements),
             'source requirements repeat a logical owner')
    _require(type(contract) is ReferenceQualityContract, 'reference contract type differs')
    contract.__post_init__()
    groups = tuple((row.module_id, row.logical_ids) for row in contract.modules)
    replay = build_reference_quality_contract(source_authority=contract.source_authority,
        reference_bbox_nm=contract.reference_bbox_nm, occurrences=observed,
        group_members=groups, domain_policy=contract.domain_policy)
    _require(replay.canonical_data() == contract.canonical_data(),
             'reference contract differs from actual observed unit geometry')
    counts = Counter(row.logical_id for row in observed)
    _require(dict(counts) == {row.logical_id: row.unit_count for row in requirements},
             'reference unit population differs from complete source requirements')
    plans = []
    for requirement in sorted(requirements, key=lambda row: row.logical_id):
        rows = tuple(row for row in observed if row.logical_id == requirement.logical_id)
        _require({row.parameter_class for row in rows} == {requirement.parameter_class},
                 'reference parameter class differs from source requirement')
        centers = {row.occurrence_id: row.representative_center_nm for row in rows}
        xs = {value: rank for rank, value in enumerate(sorted({p[0] for p in centers.values()}))}
        ys = {value: rank for rank, value in enumerate(sorted({p[1] for p in centers.values()}))}
        ordered = sorted(rows, key=lambda row: (centers[row.occurrence_id][1],
                                                centers[row.occurrence_id][0], row.occurrence_id))
        pattern = ObservedUnitRanks(tuple((ys[centers[row.occurrence_id][1]],
            xs[centers[row.occurrence_id][0]], row.orientation) for row in ordered))
        plans.append(LogicalUnitPlan(requirement, pattern,
            tuple(row.occurrence_id for row in ordered)))
    classes = tuple((row.class_id, row.logical_ids) for row in contract.class_population)
    return ReferenceStructurePlan(source_sha256, contract.sha256, tuple(plans), classes, groups)
