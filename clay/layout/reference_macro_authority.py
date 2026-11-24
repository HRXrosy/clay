"""Tracked authority for coordinate-free reference macro relations.

The signed receipt names relation facts, but current ``@place_group`` state is
the evidence used to replay them.  Fact IDs are attached only after an ID-free
membership, ordinal-cell, or partial-order relation has been reconstructed and
matched exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from clay.layout.intent_basis import ClaimScope
from clay.layout.intent_closure import IntentEntailment
from clay.layout.intent_facts import (
    EntityKind,
    EntityRef,
    FactAuthority,
    IntentFact,
    IntentFactKind,
    IntentFactSet,
    SourceInventory,
    verify_intent_fact_authority,
)


_BUNDLE_SCHEMA = "clay.intent-authority-bundle/v1"
_RECEIPT_SCHEMA = "clay.reference-macro-role-authority/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_FACT_KINDS = frozenset(
    {
        IntentFactKind.MEMBER_PARTITION,
        IntentFactKind.ORDINAL_CELL,
        IntentFactKind.ORDER,
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_data(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"reference macro authority {field} must be a SHA-256")
    return value


def _bundle_member(data: object, *, member: str) -> object:
    expected = {
        "schema",
        "physical_macro_partition_authority",
        "reference_macro_role_authority",
    }
    if (
        type(data) is not dict
        or set(data) != expected
        or data.get("schema") != _BUNDLE_SCHEMA
    ):
        raise ValueError("intent authority bundle schema differs")
    return data[member]


def _relation_key(fact: IntentFact) -> tuple[IntentFactKind, tuple[EntityRef, ...], str]:
    return (fact.kind, fact.subjects, fact.payload_json)


def _relation_atom_id(fact: IntentFact) -> str:
    relation = {
        "kind": fact.kind.value,
        "subjects": [subject.canonical_data() for subject in fact.subjects],
        "payload": fact.payload,
    }
    return f"source_relation:{_sha256_data(relation)}"


def _partition_container_and_members(
    fact: IntentFact,
) -> tuple[EntityRef, tuple[EntityRef, ...]]:
    groups = tuple(
        subject for subject in fact.subjects if subject.kind is EntityKind.PLACE_GROUP
    )
    members = tuple(
        subject for subject in fact.subjects if subject.kind is EntityKind.DEVICE
    )
    if len(groups) != 1 or not members or len(fact.subjects) != len(members) + 1:
        raise ValueError("reference macro member partition has a foreign signature")
    return groups[0], members


def _order_relation(
    *,
    left: str,
    right: str,
    left_cell: tuple[int, int],
    right_cell: tuple[int, int],
    axis: str,
) -> str | None:
    left_tier, left_column = left_cell
    right_tier, right_column = right_cell
    if axis in {"column", "horizontal", "x"}:
        if left_tier != right_tier or left_column == right_column:
            return None
        if axis == "column":
            return "before" if left_column < right_column else "after"
        return "left_of" if left_column < right_column else "right_of"
    if axis in {"tier", "vertical", "y"}:
        if left_column != right_column or left_tier == right_tier:
            return None
        if axis == "tier":
            return "before" if left_tier < right_tier else "after"
        return "below" if left_tier < right_tier else "above"
    raise ValueError(f"reference macro order axis {axis!r} is not registered")


@dataclass(frozen=True)
class ReferenceMacroRoleAuthorityReceipt:
    circuit_id: str
    source_sha256: str
    topology_sha256: str
    observation_receipt_sha256: str
    reference_gds_sha256: str
    reference_spatial_sha256: str
    pdk_id: str
    pdk_source_snapshot_sha256: str
    pdk_installed_authority_sha256: str
    pdk_layermap_sha256: str
    fact_set: IntentFactSet
    source_inventory: SourceInventory

    def __post_init__(self) -> None:
        for field in ("circuit_id", "pdk_id"):
            value = getattr(self, field)
            if type(value) is not str or not value or value != value.strip():
                raise TypeError(f"reference macro authority {field} must be exact")
        for field in (
            "source_sha256",
            "topology_sha256",
            "observation_receipt_sha256",
            "reference_gds_sha256",
            "reference_spatial_sha256",
            "pdk_source_snapshot_sha256",
            "pdk_installed_authority_sha256",
            "pdk_layermap_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if type(self.fact_set) is not IntentFactSet:
            raise TypeError("reference macro fact_set has a foreign type")
        if type(self.source_inventory) is not SourceInventory:
            raise TypeError("reference macro source_inventory has a foreign type")
        if (
            self.fact_set.circuit_id != self.circuit_id
            or self.source_inventory.circuit_id != self.circuit_id
            or self.fact_set.source_sha256 != self.source_sha256
            or self.source_inventory.source_sha256 != self.source_sha256
            or self.fact_set.topology_sha256 != self.topology_sha256
            or self.source_inventory.topology_sha256 != self.topology_sha256
        ):
            raise ValueError("reference macro source lineage differs")
        if (
            self.fact_set.claim_scope
            is not ClaimScope.REFERENCE_CONDITIONED_SINGLE_CIRCUIT
            or self.fact_set.target_reference_receipt_sha256
            != self.observation_receipt_sha256
        ):
            raise ValueError("reference macro observation authority differs")
        if (
            self.fact_set.pdk_snapshot_sha256
            != self.pdk_source_snapshot_sha256
            or self.fact_set.pdk_installed_authority_sha256
            != self.pdk_installed_authority_sha256
        ):
            raise ValueError("reference macro PDK authority differs")
        verify_intent_fact_authority(
            fact_set=self.fact_set,
            source_inventory=self.source_inventory,
        )
        self._verify_relation_schema()

    def _verify_relation_schema(self) -> None:
        facts = self.fact_set.facts
        if not facts or any(fact.kind not in _ALLOWED_FACT_KINDS for fact in facts):
            raise ValueError(
                "reference macro authority permits only membership, ordinal, and order"
            )
        partitions = tuple(
            fact for fact in facts if fact.kind is IntentFactKind.MEMBER_PARTITION
        )
        ordinals = tuple(
            fact for fact in facts if fact.kind is IntentFactKind.ORDINAL_CELL
        )
        orders = tuple(fact for fact in facts if fact.kind is IntentFactKind.ORDER)
        if (
            not partitions
            or len(partitions) != len(ordinals)
            or any(fact.authority is not FactAuthority.DESIGN for fact in partitions)
            or any(
                fact.authority is not FactAuthority.TARGET_REFERENCE_RELATION
                for fact in (*ordinals, *orders)
            )
        ):
            raise ValueError("reference macro fact authorities or denominators differ")

        group_inventory = tuple(
            entity.name
            for entity in self.source_inventory.entities
            if entity.kind is EntityKind.PLACE_GROUP
        )
        device_inventory = tuple(
            entity.name
            for entity in self.source_inventory.entities
            if entity.kind is EntityKind.DEVICE
        )
        partition_groups: list[str] = []
        partition_members: list[str] = []
        for fact in partitions:
            group, members = _partition_container_and_members(fact)
            partition_groups.append(group.name)
            partition_members.extend(member.name for member in members)
        if (
            tuple(sorted(partition_groups)) != group_inventory
            or len(set(partition_groups)) != len(partition_groups)
            or tuple(sorted(partition_members)) != device_inventory
            or len(set(partition_members)) != len(partition_members)
        ):
            raise ValueError("reference macro facts are not an exact disjoint partition")

        ordinal_by_group: dict[str, tuple[int, int]] = {}
        for fact in ordinals:
            group = fact.subjects[0].name
            payload = fact.payload
            ordinal_by_group[group] = (payload["tier"], payload["column"])
        if tuple(sorted(ordinal_by_group)) != group_inventory:
            raise ValueError("reference macro ordinal denominator differs")
        # This complete authority stores ranks from
        # reference_macro_relation_projection, not reference positions.  Its
        # axes are global across the complete group inventory; individual
        # rows can therefore legitimately omit a column used in another row.
        # Do not apply this complete-denominator rule to sparse IntentBasis
        # selections, whose original rank holes are intentionally preserved.
        for axis, values in (
            ("tier", {tier for tier, _column in ordinal_by_group.values()}),
            ("column", {column for _tier, column in ordinal_by_group.values()}),
        ):
            if sorted(values) != list(range(len(values))):
                raise ValueError(
                    f"reference macro {axis} must use compact global reference ordinals"
                )
        for fact in orders:
            left, right = (subject.name for subject in fact.subjects)
            payload = fact.payload
            observed = _order_relation(
                left=left,
                right=right,
                left_cell=ordinal_by_group[left],
                right_cell=ordinal_by_group[right],
                axis=payload["axis"],
            )
            if observed != payload["relation"]:
                raise ValueError("reference macro order contradicts ordinal authority")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _RECEIPT_SCHEMA,
            "circuit_id": self.circuit_id,
            "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "observation_receipt_sha256": self.observation_receipt_sha256,
            "reference_gds_sha256": self.reference_gds_sha256,
            "reference_spatial_sha256": self.reference_spatial_sha256,
            "pdk_id": self.pdk_id,
            "pdk_source_snapshot_sha256": self.pdk_source_snapshot_sha256,
            "pdk_installed_authority_sha256": (
                self.pdk_installed_authority_sha256
            ),
            "pdk_layermap_sha256": self.pdk_layermap_sha256,
            "fact_set": self.fact_set.canonical_data(),
            "source_inventory": self.source_inventory.canonical_data(),
        }

    @property
    def sha256(self) -> str:
        return _sha256_data(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "ReferenceMacroRoleAuthorityReceipt":
        fields = {
            "schema",
            "circuit_id",
            "source_sha256",
            "topology_sha256",
            "observation_receipt_sha256",
            "reference_gds_sha256",
            "reference_spatial_sha256",
            "pdk_id",
            "pdk_source_snapshot_sha256",
            "pdk_installed_authority_sha256",
            "pdk_layermap_sha256",
            "fact_set",
            "source_inventory",
        }
        if (
            type(data) is not dict
            or set(data) != fields
            or data.get("schema") != _RECEIPT_SCHEMA
        ):
            raise ValueError("reference macro authority receipt schema differs")
        result = cls(
            circuit_id=data["circuit_id"],
            source_sha256=data["source_sha256"],
            topology_sha256=data["topology_sha256"],
            observation_receipt_sha256=data["observation_receipt_sha256"],
            reference_gds_sha256=data["reference_gds_sha256"],
            reference_spatial_sha256=data["reference_spatial_sha256"],
            pdk_id=data["pdk_id"],
            pdk_source_snapshot_sha256=data["pdk_source_snapshot_sha256"],
            pdk_installed_authority_sha256=data[
                "pdk_installed_authority_sha256"
            ],
            pdk_layermap_sha256=data["pdk_layermap_sha256"],
            fact_set=IntentFactSet.from_data(data["fact_set"]),
            source_inventory=SourceInventory.from_data(data["source_inventory"]),
        )
        if result.canonical_data() != data:
            raise ValueError("reference macro authority receipt is not canonical")
        return result


@dataclass(frozen=True)
class ReferenceMacroRoleAuthority:
    receipt: ReferenceMacroRoleAuthorityReceipt
    fact_set: IntentFactSet
    source_inventory: SourceInventory
    entailments: tuple[IntentEntailment, ...]

    def __post_init__(self) -> None:
        if self.fact_set != self.receipt.fact_set:
            raise ValueError("reference macro fact set differs from receipt")
        if self.source_inventory != self.receipt.source_inventory:
            raise ValueError("reference macro inventory differs from receipt")
        if (
            type(self.entailments) is not tuple
            or any(type(row) is not IntentEntailment for row in self.entailments)
        ):
            raise TypeError("reference macro entailments must be exact")
        covered = {fact_id for row in self.entailments for fact_id in row.fact_ids}
        if covered != {fact.fact_id for fact in self.fact_set.facts}:
            raise ValueError("reference macro entailments do not cover exact facts")


def _source_relations(
    *,
    source_device_names: tuple[str, ...],
    place_groups: tuple[object, ...],
) -> set[tuple[IntentFactKind, tuple[EntityRef, ...], str]]:
    if (
        type(source_device_names) is not tuple
        or tuple(sorted(set(source_device_names))) != source_device_names
        or any(type(name) is not str or not name for name in source_device_names)
    ):
        raise ValueError("reference macro source device inventory is not canonical")
    if type(place_groups) is not tuple:
        raise TypeError("reference macro place groups must be an exact tuple")
    names: list[str] = []
    members_seen: list[str] = []
    ordinal_by_group: dict[str, tuple[int, int]] = {}
    relations: set[tuple[IntentFactKind, tuple[EntityRef, ...], str]] = set()

    def add(
        kind: IntentFactKind,
        subjects: tuple[EntityRef, ...],
        payload: dict[str, object],
    ) -> None:
        key = (
            kind,
            tuple(sorted(subjects)),
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
        if key in relations:
            raise ValueError("reference macro source emitted a duplicate relation")
        relations.add(key)

    for group in place_groups:
        name = getattr(group, "group", None)
        members = getattr(group, "devices", None)
        row_min = getattr(group, "row_min", None)
        row_max = getattr(group, "row_max", None)
        column = getattr(group, "col", None)
        if (
            type(name) is not str
            or not name
            or type(members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
            or row_min != row_max
            or type(row_min) is not int
            or row_min < 0
            or type(column) is not int
            or column < 0
        ):
            raise ValueError("reference macro source place-group relation is malformed")
        names.append(name)
        members_seen.extend(members)
        ordinal_by_group[name] = (row_min, column)
        group_ref = EntityRef(EntityKind.PLACE_GROUP, name)
        add(
            IntentFactKind.MEMBER_PARTITION,
            (
                group_ref,
                *(EntityRef(EntityKind.DEVICE, member) for member in members),
            ),
            {"complete": True},
        )
        add(
            IntentFactKind.ORDINAL_CELL,
            (group_ref,),
            {"tier": row_min, "column": column},
        )
    if (
        tuple(names) != tuple(sorted(set(names)))
        or tuple(sorted(members_seen)) != source_device_names
        or len(set(members_seen)) != len(members_seen)
    ):
        raise ValueError("reference macro source is not an exact disjoint partition")

    axes = ("column", "horizontal", "tier", "vertical", "x", "y")
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for axis in axes:
                relation = _order_relation(
                    left=left,
                    right=right,
                    left_cell=ordinal_by_group[left],
                    right_cell=ordinal_by_group[right],
                    axis=axis,
                )
                if relation is not None:
                    add(
                        IntentFactKind.ORDER,
                        (
                            EntityRef(EntityKind.PLACE_GROUP, left),
                            EntityRef(EntityKind.PLACE_GROUP, right),
                        ),
                        {"axis": axis, "relation": relation},
                    )
    return relations


def load_reference_macro_role_authority(
    authority_path: str | Path,
    *,
    source_bytes: bytes,
    topology_bytes: bytes,
    observation_data: object,
    circuit_id: str,
    source_device_names: tuple[str, ...],
    place_groups: tuple[object, ...],
) -> ReferenceMacroRoleAuthority:
    """Replay signed macro relations against current parsed source state."""

    if type(source_bytes) is not bytes or type(topology_bytes) is not bytes:
        raise TypeError("reference macro source and topology must be exact bytes")
    data = json.loads(Path(authority_path).read_text(encoding="utf-8"))
    receipt = ReferenceMacroRoleAuthorityReceipt.from_data(
        _bundle_member(data, member="reference_macro_role_authority")
    )
    if circuit_id != receipt.circuit_id:
        raise ValueError("reference macro circuit identity differs")
    if _sha256_bytes(source_bytes) != receipt.source_sha256:
        raise ValueError("reference macro source SHA-256 differs")
    if _sha256_bytes(topology_bytes) != receipt.topology_sha256:
        raise ValueError("reference macro topology SHA-256 differs")

    from clay.frontend.reference_observation import (
        reference_observation_receipt_from_data,
    )

    observation = reference_observation_receipt_from_data(observation_data)
    if (
        observation.receipt_sha256 != receipt.observation_receipt_sha256
        or observation.gds_sha256 != receipt.reference_gds_sha256
        or observation.spatial_sha256 != receipt.reference_spatial_sha256
        or observation.expected_top_cell != receipt.circuit_id
    ):
        raise ValueError("reference macro observation lineage differs")
    if (
        observation.pdk_id != receipt.pdk_id
        or observation.pdk_source_snapshot_sha256
        != receipt.pdk_source_snapshot_sha256
        or observation.pdk_installed_authority_sha256
        != receipt.pdk_installed_authority_sha256
        or observation.pdk_layermap_sha256 != receipt.pdk_layermap_sha256
    ):
        raise ValueError("reference macro PDK lineage differs")

    actual = _source_relations(
        source_device_names=source_device_names,
        place_groups=place_groups,
    )
    expected = {_relation_key(fact): fact for fact in receipt.fact_set.facts}
    missing = tuple(sorted(fact.fact_id for key, fact in expected.items() if key not in actual))
    if missing:
        raise ValueError(
            "reference macro source relation differs from signed authority: "
            f"missing={missing}"
        )
    entailments = tuple(
        sorted(
            (
                IntentEntailment(
                    atom_id=_relation_atom_id(fact),
                    fact_ids=(fact.fact_id,),
                )
                for fact in receipt.fact_set.facts
            ),
            key=lambda row: row.atom_id,
        )
    )
    return ReferenceMacroRoleAuthority(
        receipt=receipt,
        fact_set=receipt.fact_set,
        source_inventory=receipt.source_inventory,
        entailments=entailments,
    )


def reference_macro_relation_projection(
    boxes: tuple[tuple[str, int, int, int, int], ...],
) -> tuple[tuple[object, ...], ...]:
    """Reduce an in-memory role fixture to translation-invariant relations."""

    if type(boxes) is not tuple:
        raise TypeError("reference macro boxes must be an exact tuple")
    centers: list[tuple[str, int, int]] = []
    for row in boxes:
        if type(row) is not tuple or len(row) != 5:
            raise TypeError("reference macro box rows must be exact five-tuples")
        name, x0, y0, x1, y1 = row
        if (
            type(name) is not str
            or not name
            or any(type(value) is not int for value in (x0, y0, x1, y1))
            or x1 <= x0
            or y1 <= y0
        ):
            raise ValueError("reference macro box is malformed")
        centers.append((name, x0 + x1, y0 + y1))
    if tuple(name for name, _x, _y in centers) != tuple(
        sorted({name for name, _x, _y in centers})
    ):
        raise ValueError("reference macro boxes must be sorted with unique names")
    x_rank = {value: index for index, value in enumerate(sorted({x for _, x, _ in centers}))}
    y_rank = {value: index for index, value in enumerate(sorted({y for _, _, y in centers}))}
    cells = {name: (y_rank[y], x_rank[x]) for name, x, y in centers}
    rows: list[tuple[object, ...]] = [
        ("ordinal_cell", name, tier, column)
        for name, (tier, column) in sorted(cells.items())
    ]
    for left_index, left in enumerate(sorted(cells)):
        for right in sorted(cells)[left_index + 1 :]:
            for axis in ("horizontal", "vertical"):
                relation = _order_relation(
                    left=left,
                    right=right,
                    left_cell=cells[left],
                    right_cell=cells[right],
                    axis=axis,
                )
                if relation is not None:
                    rows.append(("order", left, right, axis, relation))
    return tuple(sorted(rows))


__all__ = [
    "ReferenceMacroRoleAuthority",
    "ReferenceMacroRoleAuthorityReceipt",
    "load_reference_macro_role_authority",
    "reference_macro_relation_projection",
]
