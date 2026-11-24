"""Explicit public reference intent and freshly materialized rigid geometry.

These records validate declared data and its internal population cover. They do
not authenticate files or promote a saved observation to source authority. The
caller must independently reopen the declared original sources and compare the
fresh extraction before supplying a contract to a production execution.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import get_args, get_origin, get_type_hints


Point = tuple[int, int]
Box = tuple[int, int, int, int]
OrientationCounts = tuple[tuple[str, int], ...]
ORIENTATIONS = frozenset(("R0", "R90", "R180", "R270", "MR0", "MR90", "MR180", "MR270"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_shape(value: object, expected: object, label: str) -> None:
    if get_origin(expected) is tuple:
        if type(value) is not tuple:
            raise TypeError(f"{label} must be an immutable exact tuple")
        arguments = get_args(expected)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            for item in value:
                _check_shape(item, arguments[0], label)
        else:
            if len(value) != len(arguments):
                raise TypeError(f"{label} tuple arity differs")
            for item, item_type in zip(value, arguments):
                _check_shape(item, item_type, label)
    elif isinstance(expected, type) and is_dataclass(expected):
        if type(value) is not expected:
            raise TypeError(f"{label} must be an exact {expected.__name__}")
        value.__post_init__()
    elif type(value) is not expected:
        raise TypeError(f"{label} has a foreign primitive type")


def _check_record(value: object) -> None:
    hints = get_type_hints(type(value))
    for item in fields(value):
        _check_shape(getattr(value, item.name), hints[item.name], item.name)


def _decode(expected: object, value: object, label: str) -> object:
    if get_origin(expected) is tuple:
        if type(value) is not list:
            raise TypeError(f"{label} must be an exact JSON array")
        arguments = get_args(expected)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode(arguments[0], item, label) for item in value)
        if len(value) != len(arguments):
            raise TypeError(f"{label} JSON array arity differs")
        return tuple(_decode(item_type, item, label) for item, item_type in zip(value, arguments))
    if isinstance(expected, type) and is_dataclass(expected):
        definitions = fields(expected)
        if type(value) is not dict or set(value) != {item.name for item in definitions}:
            raise ValueError(f"{label} has missing or unknown fields")
        hints = get_type_hints(expected)
        return expected(**{item.name: _decode(hints[item.name], value[item.name], item.name)
                           for item in definitions})
    _check_shape(value, expected, label)
    return value


def _encode(value: object) -> object:
    if is_dataclass(value):
        return {item.name: _encode(getattr(value, item.name)) for item in fields(value)}
    if type(value) is tuple:
        return [_encode(item) for item in value]
    return value


class _CanonicalData:
    __slots__ = ()

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return _encode(self)

    @classmethod
    def from_data(cls, value: object):
        return _decode(cls, value, cls.__name__)

    @property
    def sha256(self) -> str:
        raw = json.dumps(self.canonical_data(), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
        return hashlib.sha256(raw).hexdigest()


def _name(value: str) -> None:
    _require(bool(value) and value == value.strip(), "identity must be a nonempty canonical string")


def _names(value: tuple[str, ...]) -> None:
    _require(bool(value) and value == tuple(sorted(set(value))), "population identities are empty, repeated, or unordered")
    for item in value:
        _name(item)


def _ordered(rows: tuple, key) -> None:
    identities = tuple(key(row) for row in rows)
    _require(bool(rows) and identities == tuple(sorted(set(identities))), "record identities are empty, repeated, or unordered")


def _box(value: Box, *, positive: bool = False) -> None:
    _require(value[0] < value[2] and value[1] < value[3] if positive else
             value[0] <= value[2] and value[1] <= value[3], "box extent is invalid")


def _ppm(values: tuple[int, ...]) -> None:
    _require(all(0 <= value <= 1_000_000 for value in values), "reference normalized coordinate is outside its full frame")


def _orientations(value: OrientationCounts) -> None:
    _ordered(value, lambda item: item[0])
    _require(all(name in ORIENTATIONS and count > 0 for name, count in value), "orientation multiset is invalid")


def _module_cover(groups, logicals, *, hierarchical=False):
    """Validate a full partition or a nested cover; never discard parent targets."""
    memberships = tuple(set(members) for _name, members in groups)
    _require(bool(memberships) and set().union(*memberships) == set(logicals),
             'module logical population must cover the complete source')
    for index, first in enumerate(memberships):
        for second in memberships[index + 1:]:
            if first & second:
                _require(hierarchical, 'module logical population must be a complete partition')
                _require(first <= second or second <= first,
                         'source module hierarchy has crossing memberships')


def _ranks(value: tuple, *, labelled: bool = False) -> None:
    _require(bool(value) and value == tuple(sorted(value)), "rank pattern must be a nonempty sorted multiset")
    for axis in (0, 1):
        ranks = {row[axis] for row in value}
        _require(tuple(sorted(ranks)) == tuple(range(len(ranks))), "rank pattern axes must be contiguous from zero")
    if labelled:
        for row in value:
            _name(row[2])


def _path(value: str) -> None:
    _name(value)
    path = PurePosixPath(value)
    _require(bool(path.parts) and not path.is_absolute() and "\\" not in value
             and ":" not in value and ".." not in path.parts and path.as_posix() == value,
             "source path must be canonical and repository relative")


def _sha(value: str) -> None:
    _require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "source identity is not a canonical SHA-256")


@dataclass(frozen=True, slots=True)
class ReferenceSourceAuthority(_CanonicalData):
    """Declared source manifest; authenticity is checked by the fresh adapter."""
    artifacts: tuple[tuple[str, str, str], ...]
    pdk_id: str
    pdk_snapshot_sha256: str
    pdk_installed_authority_sha256: str
    layermap_pairs: tuple[tuple[int, int], ...]
    mapping_sha256: str
    implementation_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _check_record(self)
        _ordered(self.artifacts, lambda row: row[0])
        for role, path, digest in self.artifacts:
            _name(role)
            _path(path)
            _sha(digest)
        _name(self.pdk_id)
        for digest in (self.pdk_snapshot_sha256, self.pdk_installed_authority_sha256, self.mapping_sha256):
            _sha(digest)
        _require(bool(self.layermap_pairs) and self.layermap_pairs == tuple(sorted(set(self.layermap_pairs)))
                 and all(min(row) >= 0 for row in self.layermap_pairs), "PDK layer surface is not canonical")
        _ordered(self.implementation_sha256, lambda row: row[0])
        for path, digest in self.implementation_sha256:
            _path(path)
            _sha(digest)


@dataclass(frozen=True, slots=True)
class ReferenceOccurrence(_CanonicalData):
    occurrence_id: str
    logical_id: str
    class_id: str
    center_normalized_ppm: Point
    bbox_normalized_ppm: Box
    orientation: str

    def __post_init__(self) -> None:
        _check_record(self)
        for name in (self.occurrence_id, self.logical_id, self.class_id):
            _name(name)
        _ppm(self.center_normalized_ppm)
        _ppm(self.bbox_normalized_ppm)
        _box(self.bbox_normalized_ppm)
        x, y = self.center_normalized_ppm
        x0, y0, x1, y1 = self.bbox_normalized_ppm
        _require(x0 <= x <= x1 and y0 <= y <= y1, "reference occurrence center escapes its selected box")
        _require(self.orientation in ORIENTATIONS, "reference orientation is invalid")


@dataclass(frozen=True, slots=True)
class LogicalPopulation(_CanonicalData):
    logical_id: str
    class_id: str
    occurrence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_record(self)
        _name(self.logical_id)
        _name(self.class_id)
        _names(self.occurrence_ids)


@dataclass(frozen=True, slots=True)
class ClassPopulation(_CanonicalData):
    class_id: str
    logical_ids: tuple[str, ...]
    occurrence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_record(self)
        _name(self.class_id)
        _names(self.logical_ids)
        _names(self.occurrence_ids)


@dataclass(frozen=True, slots=True)
class LocationTarget(_CanonicalData):
    scope: str
    target_id: str
    centroid_ppm: Point
    envelope_ppm: Box

    def __post_init__(self) -> None:
        _check_record(self)
        _require(self.scope in {"logical", "class"}, "location target scope is invalid")
        _name(self.target_id)
        _ppm(self.centroid_ppm)
        _ppm(self.envelope_ppm)
        _box(self.envelope_ppm)


@dataclass(frozen=True, slots=True)
class LogicalPatternTarget(_CanonicalData):
    logical_id: str
    rank_pattern: tuple[tuple[int, int], ...]
    orientation_counts: OrientationCounts

    def __post_init__(self) -> None:
        _check_record(self)
        _name(self.logical_id)
        _ranks(self.rank_pattern)
        _orientations(self.orientation_counts)


@dataclass(frozen=True, slots=True)
class ClassPatternTarget(_CanonicalData):
    class_id: str
    labelled_rank_pattern: tuple[tuple[int, int, str], ...]
    row_stagger_tolerance_nm: int
    orientation_counts: OrientationCounts

    def __post_init__(self) -> None:
        _check_record(self)
        _name(self.class_id)
        _ranks(self.labelled_rank_pattern, labelled=True)
        _require(self.row_stagger_tolerance_nm >= 0, "row stagger tolerance must be nonnegative")
        _orientations(self.orientation_counts)


@dataclass(frozen=True, slots=True)
class ModuleTarget(_CanonicalData):
    module_id: str
    logical_ids: tuple[str, ...]
    occurrence_ids: tuple[str, ...]
    centroid_ppm: Point
    envelope_ppm: Box

    def __post_init__(self) -> None:
        _check_record(self)
        _name(self.module_id)
        _names(self.logical_ids)
        _names(self.occurrence_ids)
        _ppm(self.centroid_ppm)
        _ppm(self.envelope_ppm)
        _box(self.envelope_ppm)
        x, y = self.centroid_ppm
        x0, y0, x1, y1 = self.envelope_ppm
        _require(x0 <= x <= x1 and y0 <= y <= y1, "module centroid escapes its declared envelope")


@dataclass(frozen=True, slots=True)
class ModuleRelationTarget(_CanonicalData):
    first_module: str
    second_module: str
    axis: str
    relation: str

    def __post_init__(self) -> None:
        _check_record(self)
        _name(self.first_module)
        _name(self.second_module)
        _require(self.first_module < self.second_module, "module relation endpoints are not canonical")
        _require(self.axis in {"horizontal", "vertical"}, "module relation axis is invalid")
        _require(self.relation in {"before", "after", "coincident"}, "module relation value is invalid")


@dataclass(frozen=True, slots=True)
class ReferenceQualityContract(_CanonicalData):
    source_authority: ReferenceSourceAuthority
    reference_bbox_nm: Box
    occurrences: tuple[ReferenceOccurrence, ...]
    logical_population: tuple[LogicalPopulation, ...]
    class_population: tuple[ClassPopulation, ...]
    location_targets: tuple[LocationTarget, ...]
    logical_patterns: tuple[LogicalPatternTarget, ...]
    class_patterns: tuple[ClassPatternTarget, ...]
    modules: tuple[ModuleTarget, ...]
    module_relations: tuple[ModuleRelationTarget, ...]
    domain_policy: tuple[tuple[str, str, str], ...] = ()
    mode: str = "reference_reconstruction"
    centroid_limit_ppm: int = 50_000
    envelope_limit_ppm: int = 100_000
    normalization: str = "per_occurrence_ties_even_then_mean_ties_even"
    priority: tuple[str, ...] = ("hard_legality", "complete_quality", "Q")
    schema: str = "clay.reference-quality-contract/v1"

    def __post_init__(self) -> None:
        _check_record(self)
        _require(self.schema in {"clay.reference-quality-contract/v1", "clay.reference-quality-contract/v2"}
                 and self.mode == "reference_reconstruction",
                 "reference reconstruction mode/schema differs")
        _require(self.centroid_limit_ppm == 50_000 and self.envelope_limit_ppm == 100_000,
                 "public location limits must remain 50000/100000 ppm")
        _require(self.normalization == "per_occurrence_ties_even_then_mean_ties_even", "normalization semantics differ")
        _require(self.priority == ("hard_legality", "complete_quality", "Q"), "quality priority differs")
        _box(self.reference_bbox_nm, positive=True)
        _require(self.domain_policy == tuple(sorted(set(self.domain_policy))), "domain policy declarations are not canonical")
        for row in self.domain_policy:
            for item in row:
                _name(item)
        for rows, key in (
            (self.occurrences, lambda row: row.occurrence_id),
            (self.logical_population, lambda row: row.logical_id),
            (self.class_population, lambda row: row.class_id),
            (self.location_targets, lambda row: (row.scope, row.target_id)),
            (self.logical_patterns, lambda row: row.logical_id),
            (self.class_patterns, lambda row: row.class_id),
            (self.modules, lambda row: row.module_id),
        ):
            _ordered(rows, key)
        occurrences = {row.occurrence_id: row for row in self.occurrences}
        logicals = {row.logical_id: row for row in self.logical_population}
        classes = {row.class_id: row for row in self.class_population}
        _require(set(logicals) == {row.logical_id for row in self.occurrences}, "logical occurrence population is incomplete")
        _require(set(classes) == {row.class_id for row in self.occurrences}, "class occurrence population is incomplete")
        populations = {}
        for logical, row in logicals.items():
            expected = tuple(item.occurrence_id for item in self.occurrences if item.logical_id == logical)
            _require(row.occurrence_ids == expected and all(occurrences[item].class_id == row.class_id for item in expected),
                     "logical occurrence/class cover differs")
            populations[("logical", logical)] = expected
        for class_id, row in classes.items():
            expected = tuple(item.occurrence_id for item in self.occurrences if item.class_id == class_id)
            expected_logicals = tuple(name for name, item in logicals.items() if item.class_id == class_id)
            _require(row.occurrence_ids == expected and row.logical_ids == expected_logicals, "class occurrence/logical cover differs")
            populations[("class", class_id)] = expected
        _require({(row.scope, row.target_id) for row in self.location_targets} == set(populations), "location target population is incomplete")
        for target in self.location_targets:
            points = tuple(occurrences[item].center_normalized_ppm for item in populations[(target.scope, target.target_id)])
            centroid = tuple(int(round(sum(point[axis] for point in points) / len(points))) for axis in (0, 1))
            envelope = (min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points))
            _require(target.centroid_ppm == centroid and target.envelope_ppm == envelope,
                     "location target differs from declared normalized occurrence multiset")
        _require(tuple(row.logical_id for row in self.logical_patterns) == tuple(logicals), "logical pattern population is incomplete")
        _require(tuple(row.class_id for row in self.class_patterns) == tuple(classes), "class pattern population is incomplete")
        for scope, pattern_rows in (("logical", self.logical_patterns), ("class", self.class_patterns)):
            for row in pattern_rows:
                name = row.logical_id if scope == "logical" else row.class_id
                ids = populations[(scope, name)]
                pattern = row.rank_pattern if scope == "logical" else row.labelled_rank_pattern
                counts = tuple(sorted(Counter(occurrences[item].orientation for item in ids).items()))
                _require(len(pattern) == len(ids) and row.orientation_counts == counts, "pattern occurrence/orientation population differs")
                if scope == "class":
                    expected_labels = Counter(occurrences[item].logical_id for item in ids)
                    _require(Counter(item[2] for item in pattern) == expected_labels, "class pattern logical labels differ")
        _module_cover(tuple((row.module_id, row.logical_ids) for row in self.modules), logicals,
                      hierarchical=self.schema == 'clay.reference-quality-contract/v2')
        for row in self.modules:
            expected = tuple(item.occurrence_id for item in self.occurrences if item.logical_id in row.logical_ids)
            _require(row.occurrence_ids == expected, "module occurrence population differs")
            boxes = tuple(occurrences[item].bbox_normalized_ppm for item in expected)
            envelope = (min(box[0] for box in boxes), min(box[1] for box in boxes),
                        max(box[2] for box in boxes), max(box[3] for box in boxes))
            _require(row.envelope_ppm == envelope, "module envelope differs from declared occurrence box union")
        module_by_id = {row.module_id: row for row in self.modules}
        required_relations = tuple((first.module_id, second.module_id, axis)
            for index, first in enumerate(self.modules) for second in self.modules[index + 1:]
            for axis in ("horizontal", "vertical"))
        _require(tuple((row.first_module, row.second_module, row.axis) for row in self.module_relations) == required_relations,
                 "module relation population is incomplete")
        for row in self.module_relations:
            axis = 0 if row.axis == "horizontal" else 1
            first = module_by_id[row.first_module].centroid_ppm[axis]
            second = module_by_id[row.second_module].centroid_ppm[axis]
            relation = "before" if first < second else "after" if first > second else "coincident"
            _require(row.relation == relation, "module relation differs from declared target centers")


@dataclass(frozen=True, slots=True)
class RigidOccurrence(_CanonicalData):
    logical_id: str
    owner_id: str
    center_offset_nm: Point
    orientation: str

    def __post_init__(self) -> None:
        _check_record(self)
        _name(self.logical_id)
        _name(self.owner_id)
        _require(self.orientation in ORIENTATIONS, "current occurrence orientation is invalid")


@dataclass(frozen=True, slots=True)
class RigidMaterializedBlock(_CanonicalData):
    block_id: str
    owner_id: str
    bbox_offset_nm: Box
    logical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_record(self)
        _name(self.block_id)
        _name(self.owner_id)
        _box(self.bbox_offset_nm, positive=True)
        _names(self.logical_ids)


@dataclass(frozen=True, slots=True)
class RigidQualityGeometryIR(_CanonicalData):
    """Current occurrence and artwork coordinates relative to live solver owners."""
    owners: tuple[tuple[str, int, int], ...]
    occurrences: tuple[RigidOccurrence, ...]
    materialized_blocks: tuple[RigidMaterializedBlock, ...]

    def __post_init__(self) -> None:
        _check_record(self)
        _ordered(self.owners, lambda row: row[0])
        dims = {}
        for owner, width, height in self.owners:
            _name(owner)
            _require(width > 0 and height > 0, "live owner dimensions must be positive")
            dims[owner] = (width, height)
        _ordered(self.materialized_blocks, lambda row: row.block_id)
        logical_owner = {}
        logical_block_boxes = {}
        for row in self.materialized_blocks:
            _require(row.owner_id in dims, "materialized block has an unknown owner")
            width, height = dims[row.owner_id]
            box = row.bbox_offset_nm
            _require(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height,
                     "materialized block escapes its live owner")
            for name in row.logical_ids:
                _require(name not in logical_owner, "logical member has repeated materialized ownership")
                logical_owner[name] = row.owner_id
                logical_block_boxes[name] = box
        _require({row.owner_id for row in self.materialized_blocks} == set(dims), "materialized owner cover is incomplete")
        _require(bool(self.occurrences) and self.occurrences == tuple(sorted(self.occurrences,
            key=lambda row: (row.logical_id, row.center_offset_nm, row.orientation, row.owner_id))),
            "current occurrence multiset is empty or unordered")
        _require({row.logical_id for row in self.occurrences} == set(logical_owner), "current occurrence logical population is incomplete")
        for row in self.occurrences:
            _require(logical_owner.get(row.logical_id) == row.owner_id, "current occurrence owner differs from materialized ownership")
            width, height = dims[row.owner_id]
            _require(0 <= row.center_offset_nm[0] <= width and 0 <= row.center_offset_nm[1] <= height,
                     "current occurrence escapes its live owner")
            x, y = row.center_offset_nm
            x0, y0, x1, y1 = logical_block_boxes[row.logical_id]
            _require(x0 <= x <= x1 and y0 <= y <= y1,
                     "current occurrence escapes its logical materialized block")
