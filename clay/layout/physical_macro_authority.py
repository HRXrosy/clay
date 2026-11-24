"""Tracked, source-bound authority for coordinate-free physical macro facts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from clay.layout.intent_basis import ClaimScope
from clay.layout.intent_closure import IntentEntailment
from clay.layout.intent_facts import (
    FactAuthority,
    IntentFactKind,
    IntentFactSet,
    SourceInventory,
    verify_intent_fact_authority,
)
from clay.layout.physical_macro_graph import (
    PhysicalMacroGraph,
    physical_macro_partition_entailments,
)


_SCHEMA = "clay.physical-macro-partition-authority/v1"
_BUNDLE_SCHEMA = "clay.intent-authority-bundle/v1"
_MIGRATION = "coordinate_free_intent_extension"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class PhysicalMacroPartitionAuthorityReceipt:
    """Canonical relation authority; deliberately contains no artifact locator."""

    circuit_id: str
    case_id: str
    source_sha256: str
    topology_sha256: str
    physical_macro_graph_sha256: str
    observation_receipt_sha256: str
    observation_source_sha256: str
    case_reference_source_sha256: str
    source_migration: str
    fact_set: IntentFactSet
    source_inventory: SourceInventory

    def __post_init__(self) -> None:
        for field in ("circuit_id", "case_id"):
            value = getattr(self, field)
            if type(value) is not str or not value:
                raise TypeError(f"physical macro authority {field} must be exact")
        for field in (
            "source_sha256",
            "topology_sha256",
            "physical_macro_graph_sha256",
            "observation_receipt_sha256",
            "observation_source_sha256",
            "case_reference_source_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.source_migration != _MIGRATION:
            raise ValueError("physical macro authority source migration is unsupported")
        if type(self.fact_set) is not IntentFactSet:
            raise TypeError("physical macro authority fact_set has a foreign type")
        if type(self.source_inventory) is not SourceInventory:
            raise TypeError(
                "physical macro authority source_inventory has a foreign type"
            )
        if (
            self.fact_set.circuit_id != self.circuit_id
            or self.source_inventory.circuit_id != self.circuit_id
            or self.fact_set.source_sha256 != self.source_sha256
            or self.source_inventory.source_sha256 != self.source_sha256
            or self.fact_set.topology_sha256 != self.topology_sha256
            or self.source_inventory.topology_sha256 != self.topology_sha256
        ):
            raise ValueError("physical macro authority source lineage differs")
        if self.fact_set.claim_scope is not ClaimScope.REFERENCE_FREE_SINGLE_CIRCUIT:
            raise ValueError("physical macro partitions require reference-free facts")
        if not self.fact_set.facts or any(
            fact.kind is not IntentFactKind.MEMBER_PARTITION
            or fact.authority is not FactAuthority.DESIGN
            for fact in self.fact_set.facts
        ):
            raise ValueError(
                "physical macro authority permits only design member partitions"
            )
        verify_intent_fact_authority(
            fact_set=self.fact_set,
            source_inventory=self.source_inventory,
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "circuit_id": self.circuit_id,
            "case_id": self.case_id,
            "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "physical_macro_graph_sha256": self.physical_macro_graph_sha256,
            "observation_receipt_sha256": self.observation_receipt_sha256,
            "observation_source_sha256": self.observation_source_sha256,
            "case_reference_source_sha256": self.case_reference_source_sha256,
            "source_migration": self.source_migration,
            "fact_set": self.fact_set.canonical_data(),
            "source_inventory": self.source_inventory.canonical_data(),
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_data(cls, data: object) -> "PhysicalMacroPartitionAuthorityReceipt":
        fields = {
            "schema",
            "circuit_id",
            "case_id",
            "source_sha256",
            "topology_sha256",
            "physical_macro_graph_sha256",
            "observation_receipt_sha256",
            "observation_source_sha256",
            "case_reference_source_sha256",
            "source_migration",
            "fact_set",
            "source_inventory",
        }
        if type(data) is not dict or set(data) != fields or data.get("schema") != _SCHEMA:
            raise ValueError("physical macro authority receipt schema differs")
        receipt = cls(
            circuit_id=data["circuit_id"],  # type: ignore[arg-type]
            case_id=data["case_id"],  # type: ignore[arg-type]
            source_sha256=data["source_sha256"],  # type: ignore[arg-type]
            topology_sha256=data["topology_sha256"],  # type: ignore[arg-type]
            physical_macro_graph_sha256=data[
                "physical_macro_graph_sha256"
            ],  # type: ignore[arg-type]
            observation_receipt_sha256=data[
                "observation_receipt_sha256"
            ],  # type: ignore[arg-type]
            observation_source_sha256=data[
                "observation_source_sha256"
            ],  # type: ignore[arg-type]
            case_reference_source_sha256=data[
                "case_reference_source_sha256"
            ],  # type: ignore[arg-type]
            source_migration=data["source_migration"],  # type: ignore[arg-type]
            fact_set=IntentFactSet.from_data(data["fact_set"]),
            source_inventory=SourceInventory.from_data(data["source_inventory"]),
        )
        if receipt.canonical_data() != data:
            raise ValueError("physical macro authority receipt is not canonical")
        return receipt


@dataclass(frozen=True)
class PhysicalMacroPartitionAuthority:
    receipt: PhysicalMacroPartitionAuthorityReceipt
    fact_set: IntentFactSet
    source_inventory: SourceInventory
    entailments: tuple[IntentEntailment, ...]

    def __post_init__(self) -> None:
        if self.fact_set != self.receipt.fact_set:
            raise ValueError("physical macro authority fact set differs from receipt")
        if self.source_inventory != self.receipt.source_inventory:
            raise ValueError("physical macro authority inventory differs from receipt")
        if type(self.entailments) is not tuple or any(
            type(row) is not IntentEntailment for row in self.entailments
        ):
            raise TypeError("physical macro authority entailments must be exact")
        covered = {
            fact_id for row in self.entailments for fact_id in row.fact_ids
        }
        if covered != {fact.fact_id for fact in self.fact_set.facts}:
            raise ValueError("physical macro authority does not cover exact facts")


def load_physical_macro_partition_authority(
    authority_path: str | Path,
    *,
    repo_root: str | Path,
    case_data: object,
    source_bytes: bytes,
    topology_bytes: bytes,
    observation_data: object,
    graph: PhysicalMacroGraph,
) -> PhysicalMacroPartitionAuthority:
    """Replay one tracked authority against current source, topology and graph."""

    if type(source_bytes) is not bytes or type(topology_bytes) is not bytes:
        raise TypeError("physical macro authority inputs must be exact bytes")
    if type(graph) is not PhysicalMacroGraph:
        raise TypeError("physical macro authority graph has a foreign type")
    if type(case_data) is not dict:
        raise TypeError("physical macro authority case must be an exact mapping")
    path = Path(authority_path)
    authority_data = json.loads(path.read_text(encoding="utf-8"))
    if type(authority_data) is dict and authority_data.get("schema") == _BUNDLE_SCHEMA:
        expected = {
            "schema",
            "physical_macro_partition_authority",
            "reference_macro_role_authority",
        }
        if set(authority_data) != expected:
            raise ValueError("intent authority bundle schema differs")
        authority_data = authority_data["physical_macro_partition_authority"]
    receipt = PhysicalMacroPartitionAuthorityReceipt.from_data(authority_data)
    if _sha256(source_bytes) != receipt.source_sha256:
        raise ValueError("physical macro authority source SHA-256 differs")
    if _sha256(topology_bytes) != receipt.topology_sha256:
        raise ValueError("physical macro authority topology SHA-256 differs")
    if graph.cell != receipt.circuit_id or graph.sha256 != receipt.physical_macro_graph_sha256:
        raise ValueError("physical macro authority macro graph differs")

    from clay.frontend.reference_observation import (
        reference_observation_receipt_from_data,
    )

    observation = reference_observation_receipt_from_data(observation_data)
    if (
        observation.receipt_sha256 != receipt.observation_receipt_sha256
        or observation.source_sha256 != receipt.observation_source_sha256
        or observation.topology_sha256 != receipt.topology_sha256
        or observation.source_sha256 == receipt.source_sha256
    ):
        raise ValueError("physical macro authority observation migration differs")

    case_source = case_data.get("source")
    case_metadata = case_data.get("metadata")
    if (
        case_data.get("schema") != "clay.case/v1"
        or case_data.get("id") != receipt.case_id
        or type(case_source) is not dict
        or type(case_metadata) is not dict
        or case_source.get("top_cell") != receipt.circuit_id
        or case_metadata.get("key") != receipt.case_id
        or case_metadata.get("pdk") != observation.pdk_id
        or case_metadata.get("reference_source_sha256")
        != receipt.case_reference_source_sha256
        or receipt.case_reference_source_sha256 == receipt.source_sha256
    ):
        raise ValueError("physical macro authority case migration differs")

    root = Path(repo_root).resolve()

    def _case_artifact_bytes(value: object, *, suffix: str) -> bytes:
        if type(value) is not str:
            raise ValueError("physical macro authority case locator is not canonical")
        locator = PurePosixPath(value)
        if (
            locator.is_absolute()
            or "\\" in value
            or ".." in locator.parts
            or locator.as_posix() != value
            or locator.suffix.casefold() != suffix
        ):
            raise ValueError("physical macro authority case locator is not canonical")
        resolved = (root / Path(*locator.parts)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "physical macro authority case locator escapes repository"
            ) from exc
        return resolved.read_bytes()

    if _case_artifact_bytes(case_source.get("clay"), suffix=".clay") != source_bytes:
        raise ValueError("physical macro authority case source lineage differs")
    if (
        _case_artifact_bytes(case_metadata.get("source_spice"), suffix=".cir")
        != topology_bytes
    ):
        raise ValueError("physical macro authority case topology lineage differs")

    entailments = physical_macro_partition_entailments(
        graph=graph,
        fact_set=receipt.fact_set,
        source_inventory=receipt.source_inventory,
    )
    return PhysicalMacroPartitionAuthority(
        receipt=receipt,
        fact_set=receipt.fact_set,
        source_inventory=receipt.source_inventory,
        entailments=entailments,
    )


__all__ = [
    "PhysicalMacroPartitionAuthority",
    "PhysicalMacroPartitionAuthorityReceipt",
    "load_physical_macro_partition_authority",
]
