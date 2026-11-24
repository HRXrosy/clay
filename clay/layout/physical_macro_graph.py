"""Typed, canonical physical-macro hierarchy for analog layout intent.

The graph is circuit- and PDK-neutral.  It carries authored/derived structure
identity and validates ambiguity; geometry lowering belongs to later stages.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import TYPE_CHECKING, Literal, Mapping

from clay.ir import CellDef
from clay.layout.functional_roles import functional_group_records_from_cell
from clay.layout.physical_macro_ownership import array_owner_parent
from clay.layout.intent import (
    LayoutIntent,
    replay_passive_bank_partition_intent,
)
from clay.layout.spatial.intent_projection import replay_hierarchical_macro_dag_intent

if TYPE_CHECKING:
    from clay.layout.intent_closure import IntentEntailment
    from clay.layout.intent_facts import IntentFactSet, SourceInventory


MacroKind = Literal[
    "matched_array",
    "power_array",
    "passive_bank",
    "bjt_cluster",
    "well_island",
    "functional_cluster",
    "matching_relation",
]
RelationKind = Literal["contains", "adjacent", "above", "left_of", "owns_well"]

_MACRO_KINDS = frozenset(
    {
        "matched_array",
        "power_array",
        "passive_bank",
        "bjt_cluster",
        "well_island",
        "functional_cluster",
        "matching_relation",
    }
)
_RELATION_KINDS = frozenset(
    {"contains", "adjacent", "above", "left_of", "owns_well"}
)


def _unique_sorted(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicates")
    return tuple(sorted(values))


@dataclass(frozen=True)
class PhysicalMacro:
    macro_id: str
    kind: MacroKind
    members: tuple[str, ...]
    parent: str | None = None
    declaration_refs: tuple[str, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.macro_id, str) or not self.macro_id:
            raise ValueError("physical macro id must be a non-empty string")
        if self.kind not in _MACRO_KINDS:
            raise ValueError(f"unknown physical macro kind {self.kind!r}")
        if not self.members:
            raise ValueError(f"physical macro {self.macro_id!r} has no members")
        if self.parent == self.macro_id:
            raise ValueError(f"physical macro {self.macro_id!r} cannot parent itself")
        if self.parent is not None and (
            not isinstance(self.parent, str) or not self.parent
        ):
            raise ValueError("physical macro parent must be a non-empty string or None")

        members = _unique_sorted(tuple(self.members), label="physical macro members")
        refs = _unique_sorted(
            tuple(self.declaration_refs), label="physical macro declaration refs"
        )
        attributes = tuple(self.attributes)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
            for item in attributes
        ):
            raise ValueError("physical macro attributes must be string pairs")
        keys = [key for key, _value in attributes]
        if len(keys) != len(set(keys)):
            raise ValueError("physical macro attributes contain duplicate keys")

        object.__setattr__(self, "members", members)
        object.__setattr__(self, "declaration_refs", refs)
        object.__setattr__(self, "attributes", tuple(sorted(attributes)))


@dataclass(frozen=True)
class MacroRelation:
    left: str
    kind: RelationKind
    right: str
    declaration_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.left, str) or not self.left:
            raise ValueError("macro relation left endpoint must be non-empty")
        if not isinstance(self.right, str) or not self.right:
            raise ValueError("macro relation right endpoint must be non-empty")
        if self.left == self.right:
            raise ValueError("macro relation endpoints must be distinct")
        if self.kind not in _RELATION_KINDS:
            raise ValueError(f"unknown macro relation kind {self.kind!r}")
        object.__setattr__(
            self,
            "declaration_refs",
            _unique_sorted(
                tuple(self.declaration_refs), label="macro relation declaration refs"
            ),
        )


@dataclass(frozen=True)
class PhysicalMacroGraph:
    cell: str
    macros: tuple[PhysicalMacro, ...]
    relations: tuple[MacroRelation, ...]
    devices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell, str) or not self.cell:
            raise ValueError("physical macro graph cell must be non-empty")
        devices = _unique_sorted(tuple(self.devices), label="graph devices")
        macros = tuple(self.macros)
        relations = tuple(self.relations)
        if any(not isinstance(macro, PhysicalMacro) for macro in macros):
            raise TypeError("graph macros must be PhysicalMacro values")
        if any(not isinstance(relation, MacroRelation) for relation in relations):
            raise TypeError("graph relations must be MacroRelation values")

        ids = [macro.macro_id for macro in macros]
        if len(ids) != len(set(ids)):
            duplicate = next(item for item in ids if ids.count(item) > 1)
            raise ValueError(f"duplicate physical macro id {duplicate!r}")
        by_id = {macro.macro_id: macro for macro in macros}
        known_devices = set(devices)
        for macro in macros:
            unknown = sorted(set(macro.members) - known_devices)
            if unknown:
                raise ValueError(
                    f"physical macro {macro.macro_id!r} has unknown device member "
                    f"{unknown[0]!r}"
                )
            if macro.parent is not None and macro.parent not in by_id:
                raise ValueError(
                    f"physical macro {macro.macro_id!r} has unknown parent "
                    f"{macro.parent!r}"
                )
            if macro.parent is not None and not set(macro.members) <= set(
                by_id[macro.parent].members
            ):
                raise ValueError(
                    "physical macro containment member mismatch: "
                    f"{macro.parent!r} does not own every member of "
                    f"{macro.macro_id!r}"
                )

        # A cross-row matching declaration relates two existing native row
        # owners; it is not a second owner of individual channels.  Validate
        # both endpoints against the exact source row partition before allowing
        # this narrowly typed non-owning relation to overlap those owners.
        matching_relation_owners = {}
        for macro in macros:
            if macro.kind != "matching_relation":
                continue
            attrs = dict(macro.attributes)
            try:
                owners = json.loads(attrs["row_owners"])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError("matching relation lacks native row authority") from exc
            if (len(macro.members) != 2 or attrs.get("matching") != "true"
                    or attrs.get("requires_interdigitation") != "false"
                    or not macro.declaration_refs or type(owners) is not list
                    or len(owners) != 2 or any(type(owner) is not str for owner in owners)
                    or owners[0] == owners[1]):
                raise ValueError("matching relation source binding is malformed")
            for member, owner in zip(macro.members, owners):
                row = by_id.get(owner)
                if (row is None or row.kind != "matched_array"
                        or dict(row.attributes).get("style") != "native_diffusion"
                        or member not in row.members):
                    raise ValueError("matching relation native row owner differs")
                if sum(member in candidate.members for candidate in macros
                       if candidate.kind == "matched_array"
                       and dict(candidate.attributes).get("style") == "native_diffusion") != 1:
                    raise ValueError("matching relation native row owner is ambiguous")
            matching_relation_owners[macro.macro_id] = frozenset(owners)

        relation_keys = [
            (relation.left, relation.kind, relation.right) for relation in relations
        ]
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("duplicate physical macro relation")
        for relation in relations:
            for endpoint in (relation.left, relation.right):
                if endpoint not in by_id:
                    raise ValueError(f"unknown relation endpoint {endpoint!r}")
            if relation.kind == "contains" and not set(
                by_id[relation.right].members
            ) <= set(by_id[relation.left].members):
                raise ValueError(
                    "physical macro containment member mismatch: "
                    f"{relation.left!r} does not own every member of "
                    f"{relation.right!r}"
                )

        contains: dict[str, set[str]] = {macro_id: set() for macro_id in by_id}
        containment_parents: dict[str, set[str]] = {
            macro_id: set() for macro_id in by_id
        }
        for macro in macros:
            if macro.parent is not None:
                contains[macro.parent].add(macro.macro_id)
                containment_parents[macro.macro_id].add(macro.parent)
        for relation in relations:
            if relation.kind == "contains":
                contains[relation.left].add(relation.right)
                containment_parents[relation.right].add(relation.left)
        ambiguous_child = next(
            (
                macro_id
                for macro_id, parents in sorted(containment_parents.items())
                if len(parents) > 1
            ),
            None,
        )
        if ambiguous_child is not None:
            raise ValueError(
                "physical macro has multiple containment parents: "
                f"{ambiguous_child!r}"
            )
        _assert_acyclic_containment(contains)

        owns_well = {
            frozenset((relation.left, relation.right))
            for relation in relations
            if relation.kind == "owns_well"
        }
        for index, left in enumerate(macros):
            for right in macros[index + 1 :]:
                shared = set(left.members) & set(right.members)
                if not shared:
                    continue
                pair = frozenset((left.macro_id, right.macro_id))
                nonowning_match = (
                    (left.kind == "matching_relation" and right.macro_id in matching_relation_owners[left.macro_id])
                    or (right.kind == "matching_relation" and left.macro_id in matching_relation_owners[right.macro_id])
                )
                if (
                    left.kind == "functional_cluster"
                    or right.kind == "functional_cluster"
                    or nonowning_match
                    or _reachable(contains, left.macro_id, right.macro_id)
                    or _reachable(contains, right.macro_id, left.macro_id)
                    or pair in owns_well
                ):
                    continue
                raise ValueError(
                    "overlapping macro ownership without containment/owns_well: "
                    f"{left.macro_id!r}, {right.macro_id!r}, members={sorted(shared)!r}"
                )

        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "macros", tuple(sorted(macros, key=lambda m: m.macro_id)))
        object.__setattr__(
            self,
            "relations",
            tuple(sorted(relations, key=lambda r: (r.left, r.kind, r.right))),
        )

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_data(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.physical-macro-graph/v1",
            "cell": self.cell,
            "devices": list(self.devices),
            "macros": [
                {
                    "macro_id": macro.macro_id,
                    "kind": macro.kind,
                    "members": list(macro.members),
                    "parent": macro.parent,
                    "declaration_refs": list(macro.declaration_refs),
                    "attributes": [list(item) for item in macro.attributes],
                }
                for macro in self.macros
            ],
            "relations": [
                {
                    "left": relation.left,
                    "kind": relation.kind,
                    "right": relation.right,
                    "declaration_refs": list(relation.declaration_refs),
                }
                for relation in self.relations
            ],
        }


def _reachable(graph: dict[str, set[str]], start: str, target: str) -> bool:
    pending = list(graph.get(start, ()))
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(graph.get(current, ()))
    return False


def _assert_acyclic_containment(graph: dict[str, set[str]]) -> None:
    for node in graph:
        if _reachable(graph, node, node):
            raise ValueError(f"physical macro containment cycle at {node!r}")


_DECLARED_STRUCTURE_ROLES = frozenset(
    {"power_array", "passive_bank", "bjt_cluster"}
)


def _wrap_partitioned_passive_banks_in_rigid_place_groups(
    macros: list[PhysicalMacro],
    relations: list[MacroRelation],
    place_groups: Mapping[str, object],
) -> tuple[list[PhysicalMacro], list[MacroRelation]]:
    """Give an exact rigid place group one passive-bank physical owner.

    Partition declarations remain visible as functional children.  This keeps
    their semantic identities while preventing the non-MOS materializer from
    treating each partition as an independently movable/guarded macro.
    """

    rebound = list(macros)
    rewritten_relations = list(relations)
    for group_name, declaration in sorted(place_groups.items()):
        if getattr(declaration, "packing", None) != "rigid":
            continue
        group_members = frozenset(str(item) for item in declaration.devices)
        if not group_members:
            continue
        candidates = tuple(
            macro
            for macro in rebound
            if macro.kind == "passive_bank"
            and frozenset(macro.members) <= group_members
        )
        if len(candidates) < 2:
            continue
        parents = {macro.parent for macro in candidates}
        if None in parents or len(parents) != 1:
            continue
        claimed: set[str] = set()
        for candidate in candidates:
            overlap = claimed & set(candidate.members)
            if overlap:
                raise ValueError(
                    "rigid passive-bank place group has overlapping partitions: "
                    f"{group_name!r}, member={min(overlap)!r}"
                )
            claimed.update(candidate.members)
        if claimed != set(group_members):
            continue

        parent_id = next(iter(parents))
        outer_id = f"passive_bank:{group_name}"
        if any(macro.macro_id == outer_id for macro in rebound):
            raise ValueError(f"duplicate rigid passive-bank owner {outer_id!r}")
        refs = tuple(
            sorted(
                {
                    f"@place_group({group_name},packing=rigid)",
                    *(ref for macro in candidates for ref in macro.declaration_refs),
                }
            )
        )
        candidate_ids = {macro.macro_id for macro in candidates}
        rebound = [
            replace(
                macro,
                kind="functional_cluster",
                parent=outer_id,
            )
            if macro.macro_id in candidate_ids
            else macro
            for macro in rebound
        ]
        rebound.append(
            PhysicalMacro(
                macro_id=outer_id,
                kind="passive_bank",
                members=tuple(group_members),
                parent=parent_id,
                declaration_refs=refs,
                attributes=(
                    ("packing", "rigid"),
                    ("place_group", group_name),
                ),
            )
        )
        rewritten_relations = [
            relation
            for relation in rewritten_relations
            if not (
                relation.kind == "contains"
                and relation.left == parent_id
                and relation.right in candidate_ids
            )
        ]
        rewritten_relations.append(
            MacroRelation(
                left=parent_id,
                kind="contains",
                right=outer_id,
                declaration_refs=refs,
            )
        )
        for candidate in candidates:
            rewritten_relations.append(
                MacroRelation(
                    left=outer_id,
                    kind="contains",
                    right=candidate.macro_id,
                    declaration_refs=candidate.declaration_refs,
                )
            )
    return rebound, rewritten_relations


def physical_macro_graph_from_intent(
    cell: CellDef,
    intent: LayoutIntent,
    *,
    device_records: Mapping[str, Mapping[str, object]],
) -> PhysicalMacroGraph:
    """Build the canonical macro hierarchy from authored Clay intent.

    This lowering is deliberately declarative: power/passive/BJT structures
    require an explicit ``structure_role``; matched arrays and wells require
    their existing layout-intent declarations.  Device, circuit, net, and PDK
    names never select a structure.
    """

    normalized_records: dict[str, Mapping[str, object]] = {
        str(name): dict(record) for name, record in device_records.items()
    }
    for device in (getattr(cell, "devices", []) or []):
        name = str(getattr(device, "instance_name", "") or "")
        if not name:
            continue
        normalized_records.setdefault(
            name,
            {
                "kind": str(getattr(device, "device_class", "") or ""),
                "pins": dict(getattr(device, "pins", {}) or {}),
            },
        )
    device_records = normalized_records

    groups = functional_group_records_from_cell(cell)
    group_members: dict[str, tuple[str, ...]] = {
        group.name: group.members for group in groups
    }
    # Reuse the established topology-derived CRAFT labels consumed by the
    # existing array materializer (mirror_nbias1, degen_<device>, and peers).
    # These labels are still derived from circuit topology, never guessed from
    # a macro target string here.
    from clay.frontend.craft_infer import group_members as craft_group_members

    for name, raw_members in craft_group_members(cell).items():
        members = tuple(sorted(str(item) for item in raw_members))
        existing = group_members.get(str(name))
        if existing is not None and existing != members:
            raise ValueError(
                f"group {str(name)!r} has contradictory functional/CRAFT membership"
            )
        group_members[str(name)] = members
    place_groups: dict[str, object] = {}
    for declaration in intent.place_groups:
        name = str(declaration.group)
        if name in place_groups:
            raise ValueError(f"duplicate place group {name!r}")
        place_groups[name] = declaration
        members = tuple(sorted(str(item) for item in declaration.devices))
        if not members:
            continue
        existing = group_members.get(name)
        if existing is not None and existing != members:
            raise ValueError(
                f"group {name!r} has contradictory functional/place membership"
            )
        group_members[name] = members

    known_devices = set(str(name) for name in device_records)

    raw_partitions = getattr(intent, "passive_bank_partitions", ())
    if type(raw_partitions) is not tuple:
        raise TypeError("passive bank partitions must be an exact tuple")
    partitions_by_parent: dict[str, list[object]] = {}
    for raw_partition in raw_partitions:
        partition = replay_passive_bank_partition_intent(raw_partition)
        partitions_by_parent.setdefault(partition.parent, []).append(partition)
    known_groups = {group.name: group for group in groups}
    for parent, partitions in sorted(partitions_by_parent.items()):
        group = known_groups.get(parent)
        if group is None:
            raise ValueError(
                f"passive bank partition has unknown parent group {parent!r}"
            )
        if group.structure_role != "passive_bank":
            raise ValueError(
                f"passive bank partition parent {parent!r} is not a passive_bank"
            )
        if len(partitions) < 2:
            raise ValueError(
                f"passive bank partition {parent!r} requires at least two sub-banks"
            )
        bank_names = [partition.bank for partition in partitions]
        if len(bank_names) != len(set(bank_names)):
            raise ValueError(
                f"passive bank partition {parent!r} contains duplicate bank names"
            )
        claimed: set[str] = set()
        for partition in partitions:
            overlap = claimed & set(partition.members)
            if overlap:
                raise ValueError(
                    f"passive bank partition {parent!r} reuses member "
                    f"{min(overlap)!r}"
                )
            claimed.update(partition.members)
            for member in partition.members:
                record = device_records.get(member)
                kind = str((record or {}).get("kind", "")).strip().lower()
                if kind not in {"res", "resistor", "cap", "capacitor"}:
                    raise ValueError(
                        f"passive bank partition member {member!r} is not a passive"
                    )
        expected = set(group.members)
        if claimed != expected:
            raise ValueError(
                f"passive bank partition {parent!r} must exactly cover its parent; "
                f"missing={tuple(sorted(expected - claimed))!r}, "
                f"unexpected={tuple(sorted(claimed - expected))!r}"
            )

    macros: list[PhysicalMacro] = []
    relations: list[MacroRelation] = []
    functional_ids: dict[str, str] = {}
    for group in groups:
        macro_id = f"functional:{group.name}"
        functional_ids[group.name] = macro_id
        macros.append(
            PhysicalMacro(
                macro_id=macro_id,
                kind="functional_cluster",
                members=group.members,
                declaration_refs=(f"functional_group:{group.name}",),
                attributes=group.attributes,
            )
        )
        # ``structure_role`` predates this graph and intentionally accepts
        # free-form viewer/functional taxonomy (for example a cascode subtype).
        # Only the three canonical physical-owner tokens opt into a typed macro;
        # every other token remains visible on the functional cluster instead of
        # being misread as an exclusive ownership claim.
        if group.structure_role == "passive_bank" and group.name in partitions_by_parent:
            for partition in sorted(
                partitions_by_parent[group.name], key=lambda item: item.bank
            ):
                structural_id = (
                    f"passive_bank:{group.name}:{partition.bank}"
                )
                refs = (
                    f"@passive_bank_partition({group.name},{partition.bank})",
                    f"functional_group:{group.name}",
                )
                macros.append(
                    PhysicalMacro(
                        macro_id=structural_id,
                        kind="passive_bank",
                        members=partition.members,
                        parent=macro_id,
                        declaration_refs=refs,
                        attributes=(
                            ("bank", partition.bank),
                            ("role", group.role),
                        ),
                    )
                )
                relations.append(
                    MacroRelation(
                        left=macro_id,
                        kind="contains",
                        right=structural_id,
                        declaration_refs=refs,
                    )
                )
        elif group.structure_role in _DECLARED_STRUCTURE_ROLES:
            structural_id = f"{group.structure_role}:{group.name}"
            macros.append(
                PhysicalMacro(
                    macro_id=structural_id,
                    kind=group.structure_role,  # type: ignore[arg-type]
                    members=group.members,
                    parent=macro_id,
                    declaration_refs=(f"functional_group:{group.name}",),
                    attributes=(("role", group.role),),
                )
            )
            relations.append(
                MacroRelation(
                    left=macro_id,
                    kind="contains",
                    right=structural_id,
                    declaration_refs=(f"functional_group:{group.name}",),
                )
            )

    raw_macro_dag = getattr(intent, "hierarchical_macro_dag", None)
    if raw_macro_dag is not None:
        macro_dag_intent = replay_hierarchical_macro_dag_intent(raw_macro_dag)
        hierarchy = macro_dag_intent.hierarchy
        if hierarchy.root.member_nodes != tuple(sorted(known_devices)):
            raise ValueError(
                "hierarchical macro DAG root must exactly cover graph devices"
            )
        refs = tuple(
            sorted(
                (
                    f"hierarchical_macro_dag:{macro_dag_intent.sha256}",
                    f"asir:{macro_dag_intent.source_asir_sha256}",
                    f"source_hierarchy:{macro_dag_intent.source_hierarchy_sha256}",
                )
            )
        )
        root_id = f"macro_dag:scope:{hierarchy.root.scope_id}"
        macros.append(
            PhysicalMacro(
                macro_id=root_id,
                kind="functional_cluster",
                members=hierarchy.root.member_nodes,
                declaration_refs=refs,
                attributes=(("macro_dag_root", "true"),),
            )
        )
        pending = [hierarchy.root]
        visited: set[str] = set()
        while pending:
            level = pending.pop(0)
            if level.scope_id in visited:
                raise ValueError("hierarchical macro DAG replay contains a cycle")
            visited.add(level.scope_id)
            parent_id = (
                root_id
                if level.parent_node is None
                else f"macro_dag:node:{level.parent_node}"
            )
            for node in level.nodes:
                macros.append(
                    PhysicalMacro(
                        macro_id=f"macro_dag:node:{node.node_id}",
                        kind="functional_cluster",
                        members=node.member_nodes,
                        parent=parent_id,
                        declaration_refs=refs,
                        attributes=(
                            ("macro_dag_level", level.scope_id),
                            ("macro_dag_node", "true"),
                        ),
                    )
                )
            for order in level.orders:
                before_id = f"macro_dag:node:{order.before}"
                after_id = f"macro_dag:node:{order.after}"
                relations.append(
                    MacroRelation(
                        left=(after_id if order.axis == "vertical" else before_id),
                        kind=("above" if order.axis == "vertical" else "left_of"),
                        right=(before_id if order.axis == "vertical" else after_id),
                        declaration_refs=tuple(
                            sorted((*refs, f"strength:{order.strength.value}"))
                        ),
                    )
                )
            children = tuple(
                child
                for child in hierarchy.levels
                if child.parent_node
                and any(node.node_id == child.parent_node for node in level.nodes)
            )
            pending.extend(children)
        if visited != {level.scope_id for level in hierarchy.levels}:
            raise ValueError("hierarchical macro DAG replay is disconnected")

    arrays = _unique_declarations(intent.arrays, "group", "array")
    mosaics = tuple(getattr(intent, "discrete_mosaics", ()) or ())
    mosaic_members: dict[str, tuple[str, ...]] = {}
    mosaic_claimed: set[str] = set()
    for mosaic in mosaics:
        mosaic_id = str(getattr(mosaic, "mosaic_id", ""))
        members = tuple(sorted(str(item) for item in getattr(mosaic, "members", ())))
        if not mosaic_id or not members:
            raise ValueError("discrete mosaic graph declaration is malformed")
        unknown = tuple(sorted(set(members) - known_devices))
        if unknown:
            raise ValueError(
                f"discrete mosaic {mosaic_id!r} has unknown member {unknown[0]!r}"
            )
        overlap = mosaic_claimed & set(members)
        if overlap:
            raise ValueError(
                f"discrete mosaic ownership overlaps at {min(overlap)!r}"
            )
        mosaic_claimed.update(members)
        mosaic_members[mosaic_id] = members
        refs = tuple(sorted((
            f"discrete_mosaic:{mosaic_id}",
            str(getattr(mosaic, "sha256")),
            f"asir:{str(getattr(mosaic, 'source_asir_sha256'))}",
        )))
        attrs = {
            "discrete_mosaic": "true",
            "rows": str(len(getattr(mosaic, "rows", ()))),
            "source_fact_count": str(len(getattr(mosaic, "source_fact_ids", ()))),
        }
        source_array = arrays.get(mosaic_id)
        if source_array is not None and source_array.style == "native_diffusion":
            from clay.layout.spatial.intent_projection import replay_discrete_mosaic_intent
            from clay.pdk.sky130_native_geometry import native_cell_geometry_specs

            replayed = replay_discrete_mosaic_intent(mosaic)
            source_members = _resolve_group_members(
                mosaic_id, group_members=group_members, known_devices=known_devices,
                declaration="native diffusion array",
            )
            if tuple(sorted(source_members)) != members:
                raise ValueError("native mosaic source member cover differs")
            if not replayed.source_fact_ids or not replayed.evidence_ids:
                raise ValueError("native mosaic lacks source evidence")
            specs = native_cell_geometry_specs(cell)
            expected = {(owner, ordinal) for owner in members
                        for ordinal in range(specs[owner]["occurrences"])}
            actual = {(item.logical_owner, item.occurrence_ordinal)
                      for row in replayed.rows for item in row}
            if expected != actual:
                raise ValueError("native mosaic source occurrence cover differs")
            if source_array.occurrences:
                declared = tuple((item.logical_owner, item.occurrence_ordinal)
                                 for item in source_array.occurrences)
                observed = tuple((item.logical_owner, item.occurrence_ordinal)
                                 for row in replayed.rows for item in row)
                if declared != observed:
                    raise ValueError("native mosaic source occurrence order differs")
            attrs["style"] = "native_diffusion"
            refs = tuple(sorted((*refs, f"@array({mosaic_id})")))
        macros.append(PhysicalMacro(
            macro_id=f"matched:{mosaic_id}",
            kind="matched_array",
            members=members,
            declaration_refs=refs,
            attributes=tuple(attrs.items()),
        ))

    centroids = _unique_declarations(
        intent.common_centroids, "group", "common-centroid"
    )
    dummies = _unique_declarations(intent.dummy_flanks, "target", "dummy-flank")
    for target in sorted(set(arrays) | set(centroids)):
        members = _resolve_group_members(
            target,
            group_members=group_members,
            known_devices=known_devices,
            declaration="array",
        )
        overlap = set(members) & mosaic_claimed
        if overlap:
            if set(members) <= mosaic_claimed:
                # The evidence-bound exact occurrence cover is the stronger row
                # owner; the nested declaration remains present in its digest.
                continue
            raise ValueError(
                f"array {target!r} partially overlaps a discrete mosaic at "
                f"{min(overlap)!r}"
            )
        refs: list[str] = []
        attrs: dict[str, str] = {}
        array = arrays.get(target)
        if array is not None:
            refs.append(f"@array({target})")
            attrs.update(
                {
                    "style": str(array.style),
                    "rows": str(array.rows),
                    "order": str(array.order),
                }
            )
        centroid = centroids.get(target)
        if centroid is not None:
            refs.append(f"@common_centroid({target})")
            attrs["common_centroid"] = "true"
            attrs["rows"] = str(centroid.rows)
            if centroid.first_member:
                attrs["first_member"] = str(centroid.first_member)
            if centroid.member_transforms:
                attrs["member_transforms"] = ",".join(centroid.member_transforms)
        dummy = dummies.get(target)
        if dummy is not None:
            refs.append(f"@dummy_flank({target})")
            attrs.update(
                {
                    "dummy_per_end": str(dummy.per_end),
                    "dummy_tie": str(dummy.tie),
                    "dummy_share_diffusion": str(bool(dummy.share_diffusion)).lower(),
                }
            )
        if target in place_groups:
            refs.append(f"@place_group({target})")

        # A singleton/aggregate physical owner declared by ``structure_role`` is
        # already the exclusive macro for its members.  A same-group discrete
        # ``@array`` refines that owner's internal topology; compiling a second
        # matched-array owner would create contradictory overlapping ownership.
        # Common-centroid and dummy declarations remain on their established
        # structural path because they carry additional ownership semantics.
        declared_owner = next(
            (
                group
                for group in groups
                if group.name == target
                and group.structure_role in _DECLARED_STRUCTURE_ROLES
                and tuple(group.members) == tuple(members)
            ),
            None,
        )
        if (
            declared_owner is not None
            and array is not None
            and str(array.style).strip().lower() == "discrete"
            and centroid is None
            and dummy is None
        ):
            structural_id = f"{declared_owner.structure_role}:{declared_owner.name}"
            matches = [
                index
                for index, macro in enumerate(macros)
                if macro.macro_id == structural_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"declared array owner is absent or ambiguous: {structural_id}"
                )
            index = matches[0]
            owner = macros[index]
            owner_attributes = dict(owner.attributes)
            for key, value in attrs.items():
                if key in owner_attributes and owner_attributes[key] != value:
                    raise ValueError(
                        f"array refinement changes owner attribute {key!r}: "
                        f"{structural_id}"
                    )
                owner_attributes[key] = value
            macros[index] = replace(
                owner,
                declaration_refs=tuple(
                    sorted(set(owner.declaration_refs) | set(refs))
                ),
                attributes=tuple(owner_attributes.items()),
            )
            continue

        owner_groups = tuple(
            sorted(
                (
                    group
                    for group in groups
                    if set(members) <= set(group.members)
                ),
                key=lambda group: (
                    group.name != target,
                    len(group.members),
                    group.name,
                ),
            )
        )
        parent = array_owner_parent(
            members, macros,
            functional_ids[owner_groups[0].name] if owner_groups else None,
        )
        macro_id = f"matched:{target}"
        macros.append(
            PhysicalMacro(
                macro_id=macro_id,
                kind="matched_array",
                members=members,
                parent=parent,
                declaration_refs=tuple(refs),
                attributes=tuple(attrs.items()),
            )
        )
        if parent is not None:
            relations.append(
                MacroRelation(
                    left=parent,
                    kind="contains",
                    right=macro_id,
                    declaration_refs=tuple(refs),
                )
            )

    # Direct device matching is physical structure, not merely a downstream
    # scalar constraint.  Canonical reciprocal annotations have already been
    # collapsed by LayoutIntent.  Merge an exact pair into an authored row macro;
    # otherwise create one child under the narrowest declared structural owner.
    # This is entirely label/topology driven: no placement coordinate, reference
    # artifact, circuit name, or PDK branch participates.
    matching_intents = tuple(getattr(intent, "matching_intents", ()) or ())
    matching_edges: list[tuple[tuple[str, str], object]] = []
    matching_adjacency: dict[str, set[str]] = {}
    for matching in matching_intents:
        members = tuple(sorted(str(item) for item in matching.members))
        if len(members) != 2 or members[0] == members[1]:
            raise ValueError("matching intent must contain two distinct members")
        unknown = sorted(set(members) - known_devices)
        if unknown:
            raise ValueError(f"matching intent has unknown member {unknown[0]!r}")
        matching_edges.append((members, matching))
        matching_adjacency.setdefault(members[0], set()).add(members[1])
        matching_adjacency.setdefault(members[1], set()).add(members[0])

    pending_matching = set(matching_adjacency)
    matching_components: list[tuple[str, ...]] = []
    while pending_matching:
        seed = min(pending_matching)
        component: set[str] = set()
        frontier = [seed]
        while frontier:
            member = frontier.pop()
            if member in component:
                continue
            component.add(member)
            frontier.extend(sorted(matching_adjacency.get(member, ()), reverse=True))
        pending_matching -= component
        matching_components.append(tuple(sorted(component)))

    for members in sorted(matching_components):
        component_edges = [
            (edge, matching)
            for edge, matching in matching_edges
            if set(edge) <= set(members)
        ]
        refs = tuple(sorted({
            str(ref)
            for _edge, matching in component_edges
            for ref in matching.declaration_ids
        }))
        classes: dict[str, str] = {}
        for _edge, matching in component_edges:
            for member, device_class in zip(
                matching.members, matching.device_classes
            ):
                name = str(member)
                value = str(device_class)
                prior = classes.get(name)
                if prior is not None and prior != value:
                    raise ValueError(
                        f"matching member {name!r} has contradictory device classes"
                    )
                classes[name] = value
        matching_attrs = {
            "device_classes": json.dumps(
                [classes[member] for member in members],
                separators=(",", ":"),
            ),
            "matching": "true",
            "requires_interdigitation": str(
                any(
                    bool(matching.requires_interdigitation)
                    for _edge, matching in component_edges
                )
            ).lower(),
        }
        if len(component_edges) > 1:
            matching_attrs["matching_edges"] = json.dumps(
                [
                    list(edge)
                    for edge, _matching in sorted(
                        component_edges, key=lambda item: item[0]
                    )
                ],
                separators=(",", ":"),
            )
        exact = [
            (index, macro)
            for index, macro in enumerate(macros)
            if macro.kind == "matched_array" and macro.members == members
        ]
        if len(exact) > 1:
            raise ValueError(
                f"matching members {members!r} have multiple exact physical owners"
            )
        if exact:
            index, macro = exact[0]
            attrs = dict(macro.attributes)
            for key, value in matching_attrs.items():
                prior = attrs.get(key)
                if prior is not None and prior != value:
                    raise ValueError(
                        f"matching members {members!r} contradict attribute {key!r}"
                    )
                attrs[key] = value
            macros[index] = replace(
                macro,
                declaration_refs=tuple((*macro.declaration_refs, *refs)),
                attributes=tuple(attrs.items()),
            )
            continue

        # A source matching pair may cross two distinct source-owned native
        # rows.  Preserve that pair as a relation to their existing owners,
        # rather than inventing a second partially overlapping row owner.
        # Interdigitation has stronger materialization semantics and remains
        # on the existing exclusive-owner path.
        if len(members) == 2 and not any(m.requires_interdigitation for _,m in component_edges):
            bound = []
            for member in members:
                candidates = [macro for macro in macros
                              if macro.kind == "matched_array"
                              and dict(macro.attributes).get("style") == "native_diffusion"
                              and member in macro.members]
                if len(candidates) != 1:
                    break
                bound.append(candidates[0])
            if len(bound) == 2 and bound[0].macro_id != bound[1].macro_id:
                matching_attrs["row_owners"] = json.dumps([m.macro_id for m in bound], separators=(",", ":"))
                macros.append(PhysicalMacro(
                    macro_id=f"matching_relation:{'+'.join(members)}",
                    kind="matching_relation", members=members,
                    declaration_refs=refs, attributes=tuple(matching_attrs.items()),
                ))
                continue

        # Per-device discrete arrays are internal structure, while a matching
        # component owns the relation between those arrays.  When singleton
        # array declarations exactly partition the matched members, make them
        # children of the matching macro instead of leaving two overlapping
        # sibling owners.  This is declaration/membership driven and carries no
        # placement coordinate or circuit-specific rule.
        nested_arrays = [
            (index, macro)
            for index, macro in enumerate(macros)
            if macro.kind == "matched_array"
            and set(macro.members) < set(members)
            and set(macro.members) <= set(members)
        ]
        nested_members = [member for _index, macro in nested_arrays for member in macro.members]
        has_exact_nested_partition = (
            bool(nested_arrays)
            and len(nested_members) == len(set(nested_members))
            and set(nested_members) == set(members)
        )

        owner_candidates = [
            macro
            for macro in macros
            if set(members) <= set(macro.members)
        ]
        owner_candidates.sort(
            key=lambda macro: (
                macro.kind == "functional_cluster",
                len(macro.members),
                macro.macro_id,
            )
        )
        parent = owner_candidates[0].macro_id if owner_candidates else None
        macro_id = f"matched:matching:{'+'.join(members)}"
        macros.append(
            PhysicalMacro(
                macro_id=macro_id,
                kind="matched_array",
                members=members,
                parent=parent,
                declaration_refs=refs,
                attributes=tuple(matching_attrs.items()),
            )
        )
        if parent is not None:
            relations.append(
                MacroRelation(
                    left=parent,
                    kind="contains",
                    right=macro_id,
                    declaration_refs=refs,
                )
            )
        if has_exact_nested_partition:
            nested_ids = {macro.macro_id for _index, macro in nested_arrays}
            relations[:] = [
                relation
                for relation in relations
                if not (
                    relation.kind == "contains"
                    and relation.right in nested_ids
                )
            ]
            for index, macro in nested_arrays:
                macros[index] = replace(macro, parent=macro_id)
                relations.append(
                    MacroRelation(
                        left=macro_id,
                        kind="contains",
                        right=macro.macro_id,
                        declaration_refs=macro.declaration_refs,
                    )
                )

    well_specs: dict[tuple[str, ...], dict[str, object]] = {}
    for ring in intent.guard_rings:
        members = tuple(sorted(str(member) for member in ring.members))
        if not members:
            raise ValueError("guard ring must contain at least one member")
        unknown = sorted(set(members) - known_devices)
        if unknown:
            raise ValueError(f"guard ring has unknown member {unknown[0]!r}")
        bodies = _mos_body_nets(members, device_records)
        if len(bodies) > 1:
            raise ValueError(
                f"guard ring members cross incompatible body nets {sorted(bodies)!r}"
            )
        body = str(ring.tie_net or (next(iter(bodies)) if bodies else ""))
        if bodies and ring.tie_net and body not in bodies:
            raise ValueError(
                f"guard ring tie {ring.tie_net!r} disagrees with body net "
                f"{next(iter(bodies))!r}"
            )
        if not body:
            raise ValueError("guard ring cannot establish one body net")
        _validate_well_polarity(members, str(ring.well), device_records, "guard ring")
        _merge_well_spec(
            well_specs,
            members,
            body=body,
            well=str(ring.well),
            ref=f"@guard_ring({','.join(members)})",
            label="",
        )

    well_islands = _unique_declarations(
        intent.well_islands, "group", "well-island"
    )
    for target, island in well_islands.items():
        members = _resolve_group_members(
            target,
            group_members=group_members,
            known_devices=known_devices,
            declaration="well-island",
        )
        bodies = _mos_body_nets(members, device_records)
        if len(bodies) > 1:
            raise ValueError(
                f"well island target {target!r} crosses body nets {sorted(bodies)!r}"
            )
        body = str(island.body or (next(iter(bodies)) if bodies else ""))
        if bodies and island.body and body not in bodies:
            raise ValueError(
                f"well island {target!r} body {island.body!r} disagrees with "
                f"device body {next(iter(bodies))!r}"
            )
        if not body:
            raise ValueError(f"well island {target!r} cannot establish one body net")
        well = _derived_well_polarity(members, device_records)
        _merge_well_spec(
            well_specs,
            members,
            body=body,
            well=well,
            ref=f"@well_island({target})",
            label=target,
        )

    well_member_sets = tuple(sorted(well_specs))
    for index, left in enumerate(well_member_sets):
        for right in well_member_sets[index + 1 :]:
            shared = set(left) & set(right)
            if shared:
                raise ValueError(
                    "overlapping well-island ownership for members "
                    f"{sorted(shared)!r}"
                )

    for members, spec in sorted(well_specs.items()):
        label = str(spec["label"] or _short_members_digest(members))
        macro_id = f"well:{label}"
        macros.append(
            PhysicalMacro(
                macro_id=macro_id,
                kind="well_island",
                members=members,
                declaration_refs=tuple(spec["refs"]),
                attributes=(
                    ("body", str(spec["body"])),
                    ("well", str(spec["well"])),
                ),
            )
        )
        for other in tuple(macros):
            if other.macro_id == macro_id:
                continue
            if set(members) & set(other.members):
                relations.append(
                    MacroRelation(
                        left=macro_id,
                        kind="owns_well",
                        right=other.macro_id,
                        declaration_refs=tuple(spec["refs"]),
                    )
                )

    macros, relations = _wrap_partitioned_passive_banks_in_rigid_place_groups(
        macros,
        relations,
        place_groups,
    )

    # A selected internal row count is executable macro-shape intent, distinct
    # from a top-level ``row=N`` floorplan coordinate.  Bind it only to typed
    # aggregate non-MOS owners and only through a unique narrowest containing
    # place-group declaration.  With no selection, graph bytes remain stable.
    selected_groups = []
    for declaration in intent.place_groups:
        selected_rows = getattr(declaration, "pack_rows", None)
        if selected_rows is None:
            continue
        if type(selected_rows) is not int or selected_rows < 1:
            raise TypeError("place-group pack_rows authority is malformed")
        if selected_rows > len(tuple(getattr(declaration, "devices", ()) or ())):
            raise ValueError("place-group pack_rows exceeds its member capacity")
        selected_groups.append(declaration)
    rebound_macros = []
    for macro in macros:
        if macro.kind not in {"passive_bank", "bjt_cluster"}:
            rebound_macros.append(macro)
            continue
        containers = [
            group
            for group in selected_groups
            if set(macro.members) <= set(group.devices)
        ]
        if not containers:
            rebound_macros.append(macro)
            continue
        narrowest_size = min(len(group.devices) for group in containers)
        narrowest = [
            group for group in containers if len(group.devices) == narrowest_size
        ]
        if len(narrowest) != 1:
            raise ValueError(
                f"aggregate non-MOS macro has ambiguous pack_rows owners: "
                f"{macro.macro_id}"
            )
        owner = narrowest[0]
        attributes = dict(macro.attributes)
        if "pack_rows" in attributes:
            raise ValueError(
                f"aggregate non-MOS macro repeats pack_rows: {macro.macro_id}"
            )
        attributes["pack_rows"] = str(owner.pack_rows)
        rebound_macros.append(replace(
            macro,
            attributes=tuple(attributes.items()),
        ))
    macros = rebound_macros

    return PhysicalMacroGraph(
        cell=str(intent.cell or cell.local_name),
        macros=tuple(macros),
        relations=tuple(relations),
        devices=tuple(known_devices),
    )


def _unique_declarations(
    declarations: tuple[object, ...], key: str, label: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for declaration in declarations:
        target = str(getattr(declaration, key))
        if target in result:
            raise ValueError(f"duplicate {label} target {target!r}")
        result[target] = declaration
    return result


def _resolve_group_members(
    target: str,
    *,
    group_members: Mapping[str, tuple[str, ...]],
    known_devices: set[str],
    declaration: str,
) -> tuple[str, ...]:
    if target in group_members:
        members = group_members[target]
    elif target in known_devices:
        members = (target,)
    else:
        raise ValueError(f"{declaration} target {target!r} is not a declared group/device")
    unknown = sorted(set(members) - known_devices)
    if unknown:
        raise ValueError(
            f"{declaration} target {target!r} has unknown member {unknown[0]!r}"
        )
    return tuple(sorted(members))


def _record_kind(record: Mapping[str, object]) -> str:
    return str(record.get("kind") or record.get("device_class") or "").lower()


def _mos_well_polarity(kind: str) -> str:
    normalized = str(kind).lower()
    if "pmos" in normalized or normalized.startswith(("pch", "pfet")):
        return "n"
    if "nmos" in normalized or normalized.startswith(("nch", "nfet")):
        return "p"
    return ""


def _mos_body_nets(
    members: tuple[str, ...],
    device_records: Mapping[str, Mapping[str, object]],
) -> set[str]:
    bodies: set[str] = set()
    for member in members:
        record = device_records.get(member)
        if not record or not _mos_well_polarity(_record_kind(record)):
            continue
        pins = record.get("pins")
        if isinstance(pins, Mapping):
            body = str(pins.get("b") or "")
            if body:
                bodies.add(body)
    return bodies


def _derived_well_polarity(
    members: tuple[str, ...],
    device_records: Mapping[str, Mapping[str, object]],
) -> str:
    polarities: set[str] = set()
    for member in members:
        record = device_records.get(member)
        if not record:
            continue
        polarity = _mos_well_polarity(_record_kind(record))
        if polarity:
            polarities.add(polarity)
    if len(polarities) != 1:
        raise ValueError(
            f"well island members require one MOS polarity, got {sorted(polarities)!r}"
        )
    return next(iter(polarities))


def _validate_well_polarity(
    members: tuple[str, ...],
    declared: str,
    device_records: Mapping[str, Mapping[str, object]],
    label: str,
) -> None:
    mos_members = tuple(
        member
        for member in members
        if member in device_records
        and _mos_well_polarity(_record_kind(device_records[member]))
    )
    if not mos_members:
        return
    expected = _derived_well_polarity(mos_members, device_records)
    if declared != expected:
        raise ValueError(
            f"{label} well {declared!r} disagrees with member polarity {expected!r}"
        )


def _merge_well_spec(
    specs: dict[tuple[str, ...], dict[str, object]],
    members: tuple[str, ...],
    *,
    body: str,
    well: str,
    ref: str,
    label: str,
) -> None:
    existing = specs.get(members)
    if existing is None:
        specs[members] = {
            "body": body,
            "well": well,
            "refs": [ref],
            "label": label,
        }
        return
    if existing["body"] != body or existing["well"] != well:
        raise ValueError(
            "contradictory guard-ring/well-island declarations for "
            f"members {members!r}"
        )
    refs = existing["refs"]
    assert isinstance(refs, list)
    refs.append(ref)
    if label:
        prior_label = str(existing["label"] or "")
        if prior_label and prior_label != label:
            raise ValueError(
                f"duplicate well ownership labels {prior_label!r} and {label!r}"
            )
        existing["label"] = label


def _short_members_digest(members: tuple[str, ...]) -> str:
    return hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()[:12]


def physical_macro_partition_entailments(
    *,
    graph: PhysicalMacroGraph,
    fact_set: "IntentFactSet",
    source_inventory: "SourceInventory",
) -> tuple["IntentEntailment", ...]:
    """Entail exact module-partition facts from an independently built graph.

    A caller cannot cover a target merely by naming its ``fact_id``.  This
    adapter first reconstructs the coordinate-free module/member relation from
    ``PhysicalMacroGraph`` and maps the target ID only after the full relation
    body matches.  Circuit names, geometry, solver output, and reference paths
    are deliberately absent.
    """

    from clay.layout.intent_closure import IntentEntailment
    from clay.layout.intent_facts import (
        EntityKind,
        EntityRef,
        IntentFactKind,
        IntentFactSet,
        SourceInventory,
        verify_intent_fact_authority,
    )

    if type(graph) is not PhysicalMacroGraph:
        raise TypeError("graph must be an exact PhysicalMacroGraph")
    if type(fact_set) is not IntentFactSet:
        raise TypeError("fact_set must be an exact IntentFactSet")
    if type(source_inventory) is not SourceInventory:
        raise TypeError("source_inventory must be an exact SourceInventory")
    verify_intent_fact_authority(
        fact_set=fact_set,
        source_inventory=source_inventory,
    )

    actual = {
        (
            tuple(
                sorted(
                    (
                        EntityRef(EntityKind.MODULE, macro.macro_id),
                        *(
                            EntityRef(EntityKind.DEVICE, member)
                            for member in macro.members
                        ),
                    )
                )
            ),
            '{"complete":true}',
        ): macro.macro_id
        for macro in graph.macros
    }
    target_module_partitions = tuple(
        fact
        for fact in fact_set.facts
        if fact.kind is IntentFactKind.MEMBER_PARTITION
        and any(subject.kind is EntityKind.MODULE for subject in fact.subjects)
    )
    if not target_module_partitions:
        raise ValueError("fact set declares no module member-partition facts")

    by_macro: dict[str, list[str]] = {}
    missing: list[str] = []
    for fact in target_module_partitions:
        macro_id = actual.get((fact.subjects, fact.payload_json))
        if macro_id is None:
            missing.append(fact.fact_id)
            continue
        by_macro.setdefault(macro_id, []).append(fact.fact_id)
    if missing:
        raise ValueError(
            "physical macro graph differs from exact module partitions: "
            f"missing={tuple(sorted(missing))!r}"
        )

    return tuple(
        sorted(
            (
                IntentEntailment(
                    atom_id=f"physical_macro:{macro_id}",
                    fact_ids=tuple(sorted(fact_ids)),
                )
                for macro_id, fact_ids in by_macro.items()
            ),
            key=lambda row: row.atom_id,
        )
    )


__all__ = [
    "MacroKind",
    "MacroRelation",
    "PhysicalMacro",
    "PhysicalMacroGraph",
    "RelationKind",
    "physical_macro_graph_from_intent",
    "physical_macro_partition_entailments",
]
