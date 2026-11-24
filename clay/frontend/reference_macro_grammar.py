"""Coordinate-free reference macro grammar over exact placement-group owners.

The complete reference GDS exposes anonymous PCell-family regions.  A legacy
DEF can bind logical instance names to those PCell families even when it has no
component placement.  This module combines those two authorities into abstract
group ownership and a complete group permutation.  Coordinates never leave the
spatial observer and component placement in the DEF is rejected.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "ReferenceDefMasterInventory",
    "ReferenceMacroGrammar",
    "ReferenceMacroGrammarError",
    "ReferenceMacroGrammarReceipt",
    "build_reference_macro_grammar_receipt",
    "derive_reference_macro_grammar",
    "parse_reference_def_master_inventory",
    "reference_def_master_inventory_from_data",
    "reference_macro_grammar_from_data",
    "reference_macro_grammar_receipt_from_data",
    "replay_reference_macro_grammar_receipt",
]


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_COMPONENT_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")
_COMPONENTS_RE = re.compile(
    r"(?ms)^COMPONENTS[ \t]+([0-9]+)[ \t]*;[ \t]*\r?\n"
    r"(.*?)^END COMPONENTS[ \t]*;?[ \t]*$"
)
_COMPONENT_RE = re.compile(r"(?m)^-[ \t]+(\S+)[ \t]+(\S+)")
_PLACEMENT_RE = re.compile(r"(?i)\+[ \t]*(?:PLACED|FIXED|COVER)\b")
_REPLICA_RE = re.compile(r"(.+)\.([1-9][0-9]*)\Z")
_MAX_DEF_BYTES = 64 * 1024 * 1024
_MAX_COMPONENTS = 1_000_000
_SEMANTIC_VERSION = "clay.reference-obligation-semantics/v1"
_IMPLEMENTATION_PATHS = (
    "clay/flow/optimized_clay.py",
    "clay/flow/optimized_layout_style.py",
    "clay/frontend/reference_macro_grammar.py",
    "clay/layout/compact_intent.py",
    "clay/layout/compact_intent_domain.py",
    "clay/layout/physical_macro_graph.py",
    "clay/parser.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ReferenceMacroGrammarError(ValueError):
    """One typed failure at the reference macro-grammar boundary."""

    def __init__(self, code: str, detail: str) -> None:
        if (
            type(code) is not str
            or not code.startswith("REFERENCE_")
            or type(detail) is not str
            or not detail
        ):
            raise TypeError("reference macro grammar error is not typed")
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ReferenceDefMasterInventory:
    """Logical-device to PCell-family identity, with replica counts only."""

    logical_master: Mapping[str, str]
    replica_count: Mapping[str, int]

    def __post_init__(self) -> None:
        if (
            type(self.logical_master) is not MappingProxyType
            or type(self.replica_count) is not MappingProxyType
            or not self.logical_master
            or set(self.logical_master) != set(self.replica_count)
            or tuple(self.logical_master) != tuple(sorted(self.logical_master))
            or tuple(self.replica_count) != tuple(sorted(self.replica_count))
        ):
            raise ValueError("reference DEF master inventory is not canonical")
        for name, master in self.logical_master.items():
            if (
                type(name) is not str
                or _IDENTIFIER_RE.fullmatch(name) is None
                or type(master) is not str
                or _IDENTIFIER_RE.fullmatch(master) is None
            ):
                raise ValueError("reference DEF master identity is not canonical")
            count = self.replica_count[name]
            if type(count) is not int or count < 1:
                raise ValueError("reference DEF replica count is not canonical")

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": "clay.reference-def-master-inventory/v1",
            "logical_master": dict(self.logical_master),
            "replica_count": dict(self.replica_count),
        }


@dataclass(frozen=True, slots=True)
class ReferenceMacroGrammar:
    """A complete group-to-region map and abstract two-dimensional floorplan."""

    reference_spatial_sha256: str
    group_region: Mapping[str, str]
    group_order: tuple[str, ...]
    group_floorplan: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        if (
            type(self.reference_spatial_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.reference_spatial_sha256) is None
            or type(self.group_region) is not MappingProxyType
            or not self.group_region
            or tuple(self.group_region) != tuple(sorted(self.group_region))
            or type(self.group_order) is not tuple
            or set(self.group_order) != set(self.group_region)
            or len(self.group_order) != len(self.group_region)
            or type(self.group_floorplan) is not tuple
        ):
            raise ValueError("reference macro grammar is not canonical")
        if any(
            type(group) is not str
            or _IDENTIFIER_RE.fullmatch(group) is None
            or type(region) is not str
            or _IDENTIFIER_RE.fullmatch(region) is None
            for group, region in self.group_region.items()
        ) or any(type(group) is not str for group in self.group_order):
            raise ValueError("reference macro grammar identity is not canonical")
        if any(
            type(item) is not tuple
            or len(item) != 3
            or type(item[0]) is not str
            or type(item[1]) is not int
            or type(item[2]) is not int
            or item[1] < 0
            or item[2] < 0
            for item in self.group_floorplan
        ):
            raise ValueError("reference macro floorplan is not canonical")
        if tuple(item[0] for item in self.group_floorplan) != self.group_order:
            raise ValueError("reference macro floorplan ownership is not canonical")
        tiers = {tier for _group, tier, _column in self.group_floorplan}
        if tiers and tiers != set(range(max(tiers) + 1)):
            raise ValueError("reference macro tiers are not compact ordinals")
        region_cell: dict[str, tuple[int, int]] = {}
        cell_region: dict[tuple[int, int], str] = {}
        for group, tier, column in self.group_floorplan:
            region = self.group_region[group]
            cell = (tier, column)
            if region in region_cell and region_cell[region] != cell:
                raise ValueError(
                    "one anonymous reference region spans multiple macro cells"
                )
            if cell in cell_region and cell_region[cell] != region:
                raise ValueError(
                    "distinct anonymous reference regions share one macro cell"
                )
            region_cell[region] = cell
            cell_region[cell] = region
        for tier in sorted(tiers):
            columns = {
                column
                for _group, item_tier, column in self.group_floorplan
                if item_tier == tier
            }
            if sorted(columns) != list(range(len(columns))):
                raise ValueError(
                    "reference macro column classes are not compact per tier"
                )

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": "clay.reference-macro-grammar/v3",
            "reference_spatial_sha256": self.reference_spatial_sha256,
            "group_region": dict(self.group_region),
            "group_order": list(self.group_order),
            "group_floorplan": [list(item) for item in self.group_floorplan],
        }


@dataclass(frozen=True, slots=True)
class ReferenceMacroGrammarReceipt:
    """Replayable authority for one reference-derived placement permutation."""

    semantic_version: str
    circuit_id: str
    pdk_id: str
    observation_receipt_repository_path: str
    observation_receipt_file_sha256: str
    observation_receipt_sha256: str
    def_repository_path: str
    def_sha256: str
    def_inventory_sha256: str
    placement_source_repository_path: str
    placement_source_sha256: str
    top_cell: str
    base_candidate_sha256: str | None
    physical_macro_graph_sha256: str
    implementation_sha256: Mapping[str, str]
    grammar: ReferenceMacroGrammar

    def __post_init__(self) -> None:
        if self.semantic_version != _SEMANTIC_VERSION:
            raise ValueError("reference macro receipt semantic version is not canonical")
        for field_name in ("circuit_id", "pdk_id"):
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or re.fullmatch(r"[a-z][a-z0-9_.]*", value) is None
            ):
                raise ValueError(f"reference macro receipt {field_name} is not canonical")
        for value, suffix in (
            (self.observation_receipt_repository_path, ".json"),
            (self.def_repository_path, ".def"),
            (self.placement_source_repository_path, ".clay"),
        ):
            pure = PurePosixPath(value)
            if (
                type(value) is not str
                or not value
                or "\\" in value
                or pure.is_absolute()
                or ".." in pure.parts
                or pure.as_posix() != value
                or pure.suffix.casefold() != suffix
            ):
                raise ValueError("reference macro receipt path is not canonical")
        for field_name in (
            "observation_receipt_file_sha256",
            "observation_receipt_sha256",
            "def_sha256",
            "def_inventory_sha256",
            "placement_source_sha256",
            "physical_macro_graph_sha256",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"reference macro receipt {field_name} is not SHA-256")
        if self.base_candidate_sha256 is not None and (
            type(self.base_candidate_sha256) is not str
            or _SHA256_RE.fullmatch(self.base_candidate_sha256) is None
        ):
            raise ValueError("reference macro receipt candidate is not SHA-256")
        if (
            type(self.top_cell) is not str
            or _IDENTIFIER_RE.fullmatch(self.top_cell) is None
        ):
            raise ValueError("reference macro receipt top cell is not canonical")
        if (
            type(self.implementation_sha256) is not MappingProxyType
            or tuple(self.implementation_sha256) != _IMPLEMENTATION_PATHS
            or any(
                type(value) is not str or _SHA256_RE.fullmatch(value) is None
                for value in self.implementation_sha256.values()
            )
        ):
            raise ValueError("reference macro implementation closure is not canonical")
        if type(self.grammar) is not ReferenceMacroGrammar:
            raise ValueError("reference macro receipt lacks an exact grammar")
        self.grammar.__post_init__()

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": "clay.reference-macro-grammar-receipt/v2",
            "semantic_version": self.semantic_version,
            "circuit_id": self.circuit_id,
            "pdk_id": self.pdk_id,
            "observation_receipt_repository_path": (
                self.observation_receipt_repository_path
            ),
            "observation_receipt_file_sha256": (
                self.observation_receipt_file_sha256
            ),
            "observation_receipt_sha256": self.observation_receipt_sha256,
            "def_repository_path": self.def_repository_path,
            "def_sha256": self.def_sha256,
            "def_inventory_sha256": self.def_inventory_sha256,
            "placement_source_repository_path": (
                self.placement_source_repository_path
            ),
            "placement_source_sha256": self.placement_source_sha256,
            "top_cell": self.top_cell,
            "base_candidate_sha256": self.base_candidate_sha256,
            "physical_macro_graph_sha256": self.physical_macro_graph_sha256,
            "implementation_sha256": dict(self.implementation_sha256),
            "grammar": self.grammar.canonical_data(),
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self.canonical_data())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256_bytes(payload)


def _bound_repository_file(root: Path, relative: str, *, suffix: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        type(relative) is not str
        or not relative
        or "\\" in relative
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != relative
        or pure.suffix.casefold() != suffix
    ):
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_PATH_MALFORMED",
            f"repository path is not canonical: {relative!r}",
        )
    resolved_root = root.resolve()
    path = (resolved_root / Path(*pure.parts)).resolve()
    if resolved_root not in path.parents or not path.is_file():
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_INPUT_MISSING",
            f"bound repository input is unavailable: {relative}",
        )
    return path


def _implementation_state(root: Path) -> dict[str, str]:
    state = {
        relative: _sha256_file(_bound_repository_file(root, relative, suffix=".py"))
        for relative in _IMPLEMENTATION_PATHS
    }
    if tuple(state) != _IMPLEMENTATION_PATHS:
        raise AssertionError("reference macro implementation closure is unordered")
    return state


def _logical_component_name(name: str) -> str:
    match = _REPLICA_RE.fullmatch(name)
    return name if match is None else match.group(1)


def parse_reference_def_master_inventory(
    def_text: str,
) -> ReferenceDefMasterInventory:
    """Read only component/master identity from a DEF with no placements."""

    if type(def_text) is not str or not def_text or len(def_text) > _MAX_DEF_BYTES:
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_MALFORMED",
            "reference DEF must be one bounded exact string",
        )
    sections = tuple(_COMPONENTS_RE.finditer(def_text))
    if len(sections) != 1:
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_COMPONENT_SECTION_MALFORMED",
            "reference DEF must contain exactly one COMPONENTS section",
        )
    section = sections[0]
    declared = int(section.group(1))
    if declared < 1 or declared > _MAX_COMPONENTS:
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_COMPONENT_COUNT_MISMATCH",
            "reference DEF component count is outside the bounded domain",
        )
    body = section.group(2)
    if _PLACEMENT_RE.search(body) is not None:
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_COMPONENT_PLACEMENT_FORBIDDEN",
            "reference DEF component placement is not a macro-grammar authority",
        )
    components = tuple(_COMPONENT_RE.finditer(body))
    if len(components) != declared:
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_COMPONENT_COUNT_MISMATCH",
            f"declared={declared}, observed={len(components)}",
        )

    from clay.frontend.reference_spatial_intent import (
        _region_master_from_cell_name,
    )

    observed_names: set[str] = set()
    masters_by_logical: dict[str, set[str]] = {}
    replica_count: dict[str, int] = {}
    for match in components:
        component_name, master_name = match.groups()
        if (
            _COMPONENT_IDENTIFIER_RE.fullmatch(component_name) is None
            or component_name in observed_names
        ):
            raise ReferenceMacroGrammarError(
                "REFERENCE_DEF_COMPONENT_IDENTITY_MALFORMED",
                f"component identity is invalid or duplicated: {component_name!r}",
            )
        observed_names.add(component_name)
        logical_name = _logical_component_name(component_name)
        if _IDENTIFIER_RE.fullmatch(logical_name) is None:
            raise ReferenceMacroGrammarError(
                "REFERENCE_DEF_COMPONENT_IDENTITY_MALFORMED",
                f"logical component identity is invalid: {logical_name!r}",
            )
        master = _region_master_from_cell_name(master_name)
        if master is None:
            # Schematic DEFs may encode top-level I/O connectors as components.
            # They are not devices and therefore do not enter logical ownership;
            # the receipt still binds the complete DEF bytes so they are not an
            # unobserved mutation surface.
            continue
        masters_by_logical.setdefault(logical_name, set()).add(master)
        replica_count[logical_name] = replica_count.get(logical_name, 0) + 1

    mismatched = {
        logical: tuple(sorted(masters))
        for logical, masters in masters_by_logical.items()
        if len(masters) != 1
    }
    if mismatched:
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_REPLICA_MASTER_MISMATCH",
            f"logical replicas disagree on PCell family: {mismatched!r}",
        )
    logical_master = {
        logical: next(iter(masters))
        for logical, masters in masters_by_logical.items()
    }
    return ReferenceDefMasterInventory(
        logical_master=MappingProxyType(dict(sorted(logical_master.items()))),
        replica_count=MappingProxyType(dict(sorted(replica_count.items()))),
    )


def reference_def_master_inventory_from_data(
    data: object,
) -> ReferenceDefMasterInventory:
    expected = {"schema", "logical_master", "replica_count"}
    if (
        type(data) is not dict
        or set(data) != expected
        or data.get("schema") != "clay.reference-def-master-inventory/v1"
        or type(data.get("logical_master")) is not dict
        or type(data.get("replica_count")) is not dict
    ):
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_INVENTORY_MALFORMED",
            "reference DEF inventory payload is not exact",
        )
    try:
        result = ReferenceDefMasterInventory(
            logical_master=MappingProxyType(dict(sorted(data["logical_master"].items()))),
            replica_count=MappingProxyType(dict(sorted(data["replica_count"].items()))),
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_INVENTORY_MALFORMED",
            "reference DEF inventory payload is not canonical",
        ) from exc
    if result.canonical_data() != data:
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_INVENTORY_MALFORMED",
            "reference DEF inventory payload changes under replay",
        )
    return result


def derive_reference_macro_grammar(
    placement_groups: tuple[object, ...],
    *,
    spatial_intent: object,
    def_inventory: ReferenceDefMasterInventory,
) -> ReferenceMacroGrammar:
    """Map every optimized placement group to one anonymous GDS region."""

    from clay.flow.optimized_layout_style import OptimizedPlacementGroup
    from clay.frontend.reference_spatial_intent import ReferenceSpatialIntent

    if (
        type(placement_groups) is not tuple
        or not placement_groups
        or any(type(group) is not OptimizedPlacementGroup for group in placement_groups)
    ):
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_GROUP_INVENTORY_MALFORMED",
            "placement groups must be one non-empty exact tuple",
        )
    if type(spatial_intent) is not ReferenceSpatialIntent:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_SPATIAL_INTENT_MALFORMED",
            "spatial intent must be one exact coordinate-free observation",
        )
    spatial_intent.__post_init__()
    if type(def_inventory) is not ReferenceDefMasterInventory:
        raise ReferenceMacroGrammarError(
            "REFERENCE_DEF_MALFORMED",
            "DEF master inventory must be one exact typed value",
        )
    def_inventory.__post_init__()

    group_names = tuple(group.name for group in placement_groups)
    if len(set(group_names)) != len(group_names):
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_GROUP_OWNERSHIP_DUPLICATED",
            "placement group names must be unique",
        )
    member_owners: dict[str, list[str]] = {}
    for group in placement_groups:
        for member in group.members:
            member_owners.setdefault(member, []).append(group.name)
    duplicate_owners = {
        member: tuple(owners)
        for member, owners in member_owners.items()
        if len(owners) != 1
    }
    if duplicate_owners:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_GROUP_OWNERSHIP_DUPLICATED",
            f"logical devices have multiple placement owners: {duplicate_owners!r}",
        )
    group_region: dict[str, str] = {}
    source_order: dict[str, int] = {}
    known_regions = set(spatial_intent.regions)
    for index, group in enumerate(placement_groups):
        missing = tuple(
            member
            for member in group.members
            if member not in def_inventory.logical_master
        )
        if missing:
            raise ReferenceMacroGrammarError(
                "REFERENCE_MACRO_GROUP_MEMBER_MISSING",
                f"group {group.name!r} lacks DEF master identity: {missing!r}",
            )
        regions = {
            def_inventory.logical_master[member] for member in group.members
        }
        if len(regions) != 1:
            raise ReferenceMacroGrammarError(
                "REFERENCE_MACRO_GROUP_MIXED_REGION",
                f"group {group.name!r} spans anonymous regions {sorted(regions)!r}",
            )
        region = next(iter(regions))
        if region not in known_regions:
            raise ReferenceMacroGrammarError(
                "REFERENCE_MACRO_GROUP_REGION_MISSING",
                f"group {group.name!r} maps to unobserved region {region!r}",
            )
        group_region[group.name] = region
        source_order[group.name] = index

    order = tuple(sorted(
        group_region,
        key=lambda group: (
            spatial_intent.region_tier[group_region[group]],
            spatial_intent.region_column[group_region[group]],
            source_order[group],
        ),
    ))
    regions_by_tier: dict[int, list[str]] = {}
    for region in sorted(
        set(group_region.values()),
        key=lambda item: (
            spatial_intent.region_tier[item],
            spatial_intent.region_column[item],
            item,
        ),
    ):
        regions_by_tier.setdefault(spatial_intent.region_tier[region], []).append(region)
    region_column_class = {
        region: column
        for tier in sorted(regions_by_tier)
        for column, region in enumerate(regions_by_tier[tier])
    }
    floorplan: list[tuple[str, int, int]] = []
    for group in order:
        region = group_region[group]
        floorplan.append(
            (
                group,
                spatial_intent.region_tier[region],
                region_column_class[region],
            )
        )
    return ReferenceMacroGrammar(
        reference_spatial_sha256=spatial_intent.reference_scope_sha256,
        group_region=MappingProxyType(dict(sorted(group_region.items()))),
        group_order=order,
        group_floorplan=tuple(floorplan),
    )


def reference_macro_grammar_from_data(data: object) -> ReferenceMacroGrammar:
    expected = {
        "schema",
        "reference_spatial_sha256",
        "group_region",
        "group_order",
        "group_floorplan",
    }
    if (
        type(data) is not dict
        or set(data) != expected
        or data.get("schema") != "clay.reference-macro-grammar/v3"
        or type(data.get("group_region")) is not dict
        or type(data.get("group_order")) is not list
        or type(data.get("group_floorplan")) is not list
    ):
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_GRAMMAR_MALFORMED",
            "reference macro grammar payload is not exact",
        )
    try:
        result = ReferenceMacroGrammar(
            reference_spatial_sha256=data["reference_spatial_sha256"],
            group_region=MappingProxyType(dict(sorted(data["group_region"].items()))),
            group_order=tuple(data["group_order"]),
            group_floorplan=tuple(
                tuple(item) if type(item) is list else item
                for item in data["group_floorplan"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_GRAMMAR_MALFORMED",
            "reference macro grammar payload is not canonical",
        ) from exc
    if result.canonical_data() != data:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_GRAMMAR_MALFORMED",
            "reference macro grammar payload changes under replay",
        )
    return result


def reference_macro_grammar_receipt_from_data(
    data: object,
) -> ReferenceMacroGrammarReceipt:
    expected = {
        "schema",
        "semantic_version",
        "circuit_id",
        "pdk_id",
        "observation_receipt_repository_path",
        "observation_receipt_file_sha256",
        "observation_receipt_sha256",
        "def_repository_path",
        "def_sha256",
        "def_inventory_sha256",
        "placement_source_repository_path",
        "placement_source_sha256",
        "top_cell",
        "base_candidate_sha256",
        "physical_macro_graph_sha256",
        "implementation_sha256",
        "grammar",
    }
    if (
        type(data) is not dict
        or set(data) != expected
        or data.get("schema") != "clay.reference-macro-grammar-receipt/v2"
        or type(data.get("implementation_sha256")) is not dict
    ):
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_RECEIPT_MALFORMED",
            "reference macro receipt payload is not exact",
        )
    try:
        result = ReferenceMacroGrammarReceipt(
            semantic_version=data["semantic_version"],
            circuit_id=data["circuit_id"],
            pdk_id=data["pdk_id"],
            observation_receipt_repository_path=data[
                "observation_receipt_repository_path"
            ],
            observation_receipt_file_sha256=data[
                "observation_receipt_file_sha256"
            ],
            observation_receipt_sha256=data["observation_receipt_sha256"],
            def_repository_path=data["def_repository_path"],
            def_sha256=data["def_sha256"],
            def_inventory_sha256=data["def_inventory_sha256"],
            placement_source_repository_path=data[
                "placement_source_repository_path"
            ],
            placement_source_sha256=data["placement_source_sha256"],
            top_cell=data["top_cell"],
            base_candidate_sha256=data["base_candidate_sha256"],
            physical_macro_graph_sha256=data["physical_macro_graph_sha256"],
            implementation_sha256=MappingProxyType(
                dict(data["implementation_sha256"])
            ),
            grammar=reference_macro_grammar_from_data(data["grammar"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ReferenceMacroGrammarError):
            raise
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_RECEIPT_MALFORMED",
            "reference macro receipt payload is not canonical",
        ) from exc
    if result.canonical_data() != data:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_RECEIPT_MALFORMED",
            "reference macro receipt payload changes under replay",
        )
    return result


def build_reference_macro_grammar_receipt(
    repo_root: str | Path,
    *,
    observation_receipt_repository_path: str,
    def_repository_path: str,
    placement_source_repository_path: str,
    top_cell: str,
    pdk_source_snapshot_sha256: str,
    pdk_installed_authority_sha256: str,
    pdk_layermap: Mapping[str, tuple[int, int]],
    implementation_root: str | Path | None = None,
) -> ReferenceMacroGrammarReceipt:
    """Build a macro grammar and reread every bound authority before return."""

    root = Path(repo_root)
    code_root = root if implementation_root is None else Path(implementation_root)
    observation_path = _bound_repository_file(
        root,
        observation_receipt_repository_path,
        suffix=".json",
    )
    def_path = _bound_repository_file(root, def_repository_path, suffix=".def")
    placement_source_path = _bound_repository_file(
        root,
        placement_source_repository_path,
        suffix=".clay",
    )
    before = {
        "observation_receipt": _sha256_file(observation_path),
        "def": _sha256_file(def_path),
        "placement_source": _sha256_file(placement_source_path),
        "implementation": _implementation_state(code_root),
    }
    try:
        observation_data = json.loads(
            observation_path.read_text(encoding="utf-8")
        )
        def_text = def_path.read_text(encoding="utf-8")
        placement_source = placement_source_path.read_text(encoding="utf-8-sig")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_INPUT_MALFORMED",
            "a bound receipt, DEF, or Clay source is not strict text",
        ) from exc

    from clay.frontend.reference_observation import (
        ReferenceObservationError,
        reference_observation_receipt_from_data,
        replay_reference_observation_receipt,
    )

    try:
        observation = reference_observation_receipt_from_data(observation_data)
        replayed_observation = replay_reference_observation_receipt(
            root,
            observation,
            pdk_source_snapshot_sha256=pdk_source_snapshot_sha256,
            pdk_installed_authority_sha256=pdk_installed_authority_sha256,
            pdk_layermap=pdk_layermap,
        )
    except ReferenceObservationError as exc:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_OBSERVATION_REPLAY_FAILED",
            str(exc),
        ) from exc
    if top_cell != replayed_observation.expected_top_cell:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_TOP_CELL_MISMATCH",
            "placement source top cell differs from the reference observation",
        )

    inventory = parse_reference_def_master_inventory(def_text)
    from clay.flow.optimized_clay import declared_compact_candidate
    from clay.flow.optimized_layout_style import derive_optimized_placement_groups
    from clay.layout.compact_intent_domain import (
        physical_macro_graph_for_candidate,
        physical_macro_graph_for_cell,
    )
    from clay.parser import parse_clay_source

    parsed = parse_clay_source(placement_source)
    cells = tuple(
        cell for cell in parsed.cell_defs if cell.local_name == top_cell
    )
    if len(cells) != 1:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_TOP_CELL_MISMATCH",
            f"placement source contains {len(cells)} matching top cells",
        )
    cell = cells[0]
    candidate = declared_compact_candidate(placement_source)
    if candidate is not None and candidate.group_order:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_BASE_ORDER_PRESENT",
            "macro grammar must be derived from a source with no selected order",
        )
    groups = derive_optimized_placement_groups(cell, candidate=candidate)
    grouped_devices = {
        member for group in groups for member in group.members
    }
    source_devices = {
        str(device.instance_name) for device in cell.devices
    }
    if grouped_devices != source_devices:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_GROUP_COVERAGE_INCOMPLETE",
            "derived placement groups do not cover the source device inventory: "
            f"missing={sorted(source_devices - grouped_devices)!r}, "
            f"foreign={sorted(grouped_devices - source_devices)!r}",
        )
    graph = (
        physical_macro_graph_for_cell(cell)
        if candidate is None
        else physical_macro_graph_for_candidate(cell, candidate)
    )

    gds_path = _bound_repository_file(
        root,
        replayed_observation.gds_repository_path,
        suffix=".gds",
    )
    topology_path = _bound_repository_file(
        root,
        replayed_observation.topology_repository_path,
        suffix=".cir",
    )
    try:
        topology_text = topology_path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_INPUT_MALFORMED",
            "bound topology is not strict UTF-8",
        ) from exc
    from clay.frontend.reference_spatial_intent import (
        extract_reference_spatial_intent,
    )

    spatial = extract_reference_spatial_intent(
        str(gds_path),
        source_netlist=topology_text,
    )
    if spatial.reference_scope_sha256 != replayed_observation.spatial_sha256:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_SPATIAL_REPLAY_MISMATCH",
            "coordinate-free spatial observation differs from its bound receipt",
        )
    grammar = derive_reference_macro_grammar(
        groups,
        spatial_intent=spatial,
        def_inventory=inventory,
    )
    after = {
        "observation_receipt": _sha256_file(observation_path),
        "def": _sha256_file(def_path),
        "placement_source": _sha256_file(placement_source_path),
        "implementation": _implementation_state(code_root),
    }
    if after != before:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_INPUT_CHANGED_DURING_REPLAY",
            "a bound source, receipt, DEF, or implementation changed",
        )

    return ReferenceMacroGrammarReceipt(
        semantic_version=_SEMANTIC_VERSION,
        circuit_id=replayed_observation.circuit_id,
        pdk_id=replayed_observation.pdk_id,
        observation_receipt_repository_path=(
            observation_receipt_repository_path
        ),
        observation_receipt_file_sha256=before["observation_receipt"],
        observation_receipt_sha256=replayed_observation.receipt_sha256,
        def_repository_path=def_repository_path,
        def_sha256=before["def"],
        def_inventory_sha256=_canonical_sha256(inventory.canonical_data()),
        placement_source_repository_path=placement_source_repository_path,
        placement_source_sha256=before["placement_source"],
        top_cell=top_cell,
        base_candidate_sha256=(None if candidate is None else candidate.digest),
        physical_macro_graph_sha256=graph.sha256,
        implementation_sha256=MappingProxyType(
            dict(before["implementation"])
        ),
        grammar=grammar,
    )


def replay_reference_macro_grammar_receipt(
    repo_root: str | Path,
    receipt: ReferenceMacroGrammarReceipt,
    *,
    pdk_source_snapshot_sha256: str,
    pdk_installed_authority_sha256: str,
    pdk_layermap: Mapping[str, tuple[int, int]],
    implementation_root: str | Path | None = None,
) -> ReferenceMacroGrammarReceipt:
    """Reread all authorities and reproduce one exact macro-grammar receipt."""

    if type(receipt) is not ReferenceMacroGrammarReceipt:
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_RECEIPT_MALFORMED",
            "receipt must be one exact typed row",
        )
    receipt.__post_init__()
    root = Path(repo_root)
    code_root = root if implementation_root is None else Path(implementation_root)
    if _implementation_state(code_root) != dict(receipt.implementation_sha256):
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_IMPLEMENTATION_CHANGED",
            "macro-grammar implementation no longer matches the receipt",
        )
    bound = (
        (
            "REFERENCE_MACRO_OBSERVATION_RECEIPT_CHANGED",
            receipt.observation_receipt_repository_path,
            ".json",
            receipt.observation_receipt_file_sha256,
        ),
        (
            "REFERENCE_MACRO_DEF_CHANGED",
            receipt.def_repository_path,
            ".def",
            receipt.def_sha256,
        ),
        (
            "REFERENCE_MACRO_SOURCE_CHANGED",
            receipt.placement_source_repository_path,
            ".clay",
            receipt.placement_source_sha256,
        ),
    )
    for code, relative, suffix, expected_sha256 in bound:
        path = _bound_repository_file(root, relative, suffix=suffix)
        if _sha256_file(path) != expected_sha256:
            raise ReferenceMacroGrammarError(
                code,
                f"bound macro-grammar input changed: {relative}",
            )
    replayed = build_reference_macro_grammar_receipt(
        root,
        observation_receipt_repository_path=(
            receipt.observation_receipt_repository_path
        ),
        def_repository_path=receipt.def_repository_path,
        placement_source_repository_path=receipt.placement_source_repository_path,
        top_cell=receipt.top_cell,
        pdk_source_snapshot_sha256=pdk_source_snapshot_sha256,
        pdk_installed_authority_sha256=pdk_installed_authority_sha256,
        pdk_layermap=pdk_layermap,
        implementation_root=code_root,
    )
    if replayed.canonical_data() != receipt.canonical_data():
        raise ReferenceMacroGrammarError(
            "REFERENCE_MACRO_RECEIPT_REPLAY_MISMATCH",
            "replayed macro grammar differs from its bound receipt",
        )
    return replayed
