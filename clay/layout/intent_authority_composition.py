"""Compose independently replayed intent axes into one fact denominator.

This module grants no fact authority of its own.  Each input component must
already bind an exact fact set to an exact source inventory.  Composition is
permitted only when every component names the same circuit, source, topology,
and installed PDK.  The result is therefore one consumer denominator rather
than a collection of independently satisfiable partial receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from clay.layout.intent_basis import ClaimScope
from clay.layout.intent_closure import IntentEntailment
from clay.layout.intent_facts import (
    FactAuthority,
    IntentFact,
    IntentFactAuthorityReceipt,
    IntentFactSet,
    SourceInventory,
    verify_intent_fact_authority,
)


_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*\Z")


def _component_name(value: object) -> str:
    if type(value) is not str or _NAME_RE.fullmatch(value) is None:
        raise ValueError("intent authority component name is not canonical")
    return value


def _canonical_entailments(
    rows: object,
    *,
    fact_ids: frozenset[str],
) -> tuple[IntentEntailment, ...]:
    if type(rows) is not tuple or any(
        type(row) is not IntentEntailment for row in rows
    ):
        raise TypeError("intent authority entailments must be one exact tuple")
    if tuple(sorted(set(rows))) != rows:
        raise ValueError("intent authority entailments must be sorted and unique")
    atom_ids = tuple(row.atom_id for row in rows)
    if len(set(atom_ids)) != len(atom_ids):
        raise ValueError("intent authority contains a duplicate atom")
    covered = frozenset(fact_id for row in rows for fact_id in row.fact_ids)
    if covered != fact_ids:
        raise ValueError("intent authority entailments do not cover its exact facts")
    return rows


@dataclass(frozen=True)
class IntentAuthorityComponent:
    """One source-replayed axis supplied to the composition boundary."""

    name: str
    fact_set: IntentFactSet
    source_inventory: SourceInventory
    entailments: tuple[IntentEntailment, ...]

    def __post_init__(self) -> None:
        _component_name(self.name)
        if type(self.fact_set) is not IntentFactSet:
            raise TypeError("intent authority component fact_set has a foreign type")
        if type(self.source_inventory) is not SourceInventory:
            raise TypeError(
                "intent authority component source_inventory has a foreign type"
            )
        verify_intent_fact_authority(
            fact_set=self.fact_set,
            source_inventory=self.source_inventory,
        )
        _canonical_entailments(
            self.entailments,
            fact_ids=frozenset(fact.fact_id for fact in self.fact_set.facts),
        )

    @property
    def authority_receipt(self) -> IntentFactAuthorityReceipt:
        return verify_intent_fact_authority(
            fact_set=self.fact_set,
            source_inventory=self.source_inventory,
        )


@dataclass(frozen=True)
class CompositeIntentFactAuthority:
    """The unique fact, inventory, and entailment denominator for a candidate."""

    fact_set: IntentFactSet
    source_inventory: SourceInventory
    entailments: tuple[IntentEntailment, ...]
    component_authority_sha256: tuple[tuple[str, str], ...]
    receipt: IntentFactAuthorityReceipt

    def __post_init__(self) -> None:
        if type(self.fact_set) is not IntentFactSet:
            raise TypeError("composite fact_set has a foreign type")
        if type(self.source_inventory) is not SourceInventory:
            raise TypeError("composite source_inventory has a foreign type")
        replayed = verify_intent_fact_authority(
            fact_set=self.fact_set,
            source_inventory=self.source_inventory,
        )
        if type(self.receipt) is not IntentFactAuthorityReceipt:
            raise TypeError("composite authority receipt has a foreign type")
        if replayed != self.receipt:
            raise ValueError("composite authority receipt differs from exact replay")
        _canonical_entailments(
            self.entailments,
            fact_ids=frozenset(fact.fact_id for fact in self.fact_set.facts),
        )
        rows = self.component_authority_sha256
        if (
            type(rows) is not tuple
            or any(
                type(row) is not tuple
                or len(row) != 2
                or _component_name(row[0]) != row[0]
                or not re.fullmatch(r"[0-9a-f]{64}", row[1])
                for row in rows
            )
            or tuple(sorted(set(rows))) != rows
            or len({name for name, _digest in rows}) != len(rows)
        ):
            raise ValueError(
                "composite component authority identities are not canonical"
            )


def compose_intent_fact_authorities(
    components: tuple[IntentAuthorityComponent, ...],
) -> CompositeIntentFactAuthority:
    """Merge exact component authorities after enforcing common lineage."""

    if type(components) is not tuple or not components:
        raise TypeError("intent authority composition requires a non-empty tuple")
    if any(type(row) is not IntentAuthorityComponent for row in components):
        raise TypeError("intent authority composition contains a foreign component")
    names = tuple(row.name for row in components)
    if names != tuple(sorted(set(names))):
        raise ValueError("intent authority components must be sorted and unique")
    for component in components:
        component.__post_init__()

    first = components[0].fact_set
    identity = (first.circuit_id, first.source_sha256, first.topology_sha256)
    pdk_identity = (
        first.pdk_snapshot_sha256,
        first.pdk_installed_authority_sha256,
    )
    for component in components[1:]:
        current = component.fact_set
        if (current.circuit_id, current.source_sha256, current.topology_sha256) != identity:
            raise ValueError(
                "intent authority component source/topology identity differs"
            )
        if (
            current.pdk_snapshot_sha256,
            current.pdk_installed_authority_sha256,
        ) != pdk_identity:
            raise ValueError("intent authority component PDK identity differs")

    reference_receipts = {
        component.fact_set.target_reference_receipt_sha256
        for component in components
        if component.fact_set.target_reference_receipt_sha256 is not None
    }
    if len(reference_receipts) > 1:
        raise ValueError("intent authority components bind a different reference receipt")
    target_reference_receipt = (
        next(iter(reference_receipts)) if reference_receipts else None
    )
    has_reference_fact = any(
        fact.authority is FactAuthority.TARGET_REFERENCE_RELATION
        for component in components
        for fact in component.fact_set.facts
    )
    if has_reference_fact and target_reference_receipt is None:
        raise ValueError("reference facts lack one common reference receipt")

    facts = tuple(
        sorted(fact for component in components for fact in component.fact_set.facts)
    )
    fact_ids = tuple(fact.fact_id for fact in facts)
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError("intent authority composition contains a duplicate fact ID")
    semantic_relations = tuple(
        (fact.kind, fact.subjects, fact.payload_json) for fact in facts
    )
    if len(set(semantic_relations)) != len(semantic_relations):
        raise ValueError("intent authority composition contains a duplicate fact relation")

    entities = tuple(
        sorted(
            {
                entity
                for component in components
                for entity in component.source_inventory.entities
            }
        )
    )
    source_inventory = SourceInventory(
        circuit_id=identity[0],
        source_sha256=identity[1],
        topology_sha256=identity[2],
        entities=entities,
    )
    fact_set = IntentFactSet(
        circuit_id=identity[0],
        claim_scope=(
            ClaimScope.REFERENCE_CONDITIONED_SINGLE_CIRCUIT
            if target_reference_receipt is not None or has_reference_fact
            else ClaimScope.REFERENCE_FREE_SINGLE_CIRCUIT
        ),
        source_sha256=identity[1],
        topology_sha256=identity[2],
        pdk_snapshot_sha256=pdk_identity[0],
        pdk_installed_authority_sha256=pdk_identity[1],
        facts=facts,
        target_reference_receipt_sha256=target_reference_receipt,
    )

    entailments = tuple(
        sorted(
            row
            for component in components
            for row in component.entailments
        )
    )
    atom_ids = tuple(row.atom_id for row in entailments)
    if len(set(atom_ids)) != len(atom_ids):
        raise ValueError("intent authority composition contains a duplicate atom ID")
    component_receipts = tuple(
        (component.name, component.authority_receipt.sha256)
        for component in components
    )
    receipt = verify_intent_fact_authority(
        fact_set=fact_set,
        source_inventory=source_inventory,
    )
    return CompositeIntentFactAuthority(
        fact_set=fact_set,
        source_inventory=source_inventory,
        entailments=entailments,
        component_authority_sha256=component_receipts,
        receipt=receipt,
    )


__all__ = [
    "CompositeIntentFactAuthority",
    "IntentAuthorityComponent",
    "compose_intent_fact_authorities",
]
