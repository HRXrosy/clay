"""Pure, coordinate-free semantic closure for a selected intent basis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from clay.layout.intent_facts import (
    IntentFact,
    IntentFactAuthorityError,
    IntentFactKind,
    IntentFactSet,
    REGISTERED_CONSUMER_ORDER,
    REGISTERED_CONSUMERS,
    SourceInventory,
    verify_intent_fact_authority,
)


_ENTAILMENT_SCHEMA = "clay.intent-entailment/v1"
_PROJECTION_SCHEMA = "clay.consumer-projection/v1"
_RECEIPT_SCHEMA = "clay.intent-closure-receipt/v2"
_STAGES = frozenset(f"G{index}" for index in range(8))
_PURE_CLOSURE_STAGES = frozenset(f"G{index}" for index in range(4))
_PURE_STAGE_BLOCKERS = {
    "G0": frozenset({"AUTHORITY_IDENTITY", "FACT_SCHEMA"}),
    "G1": frozenset({"AUTHORITY_IDENTITY", "FACT_SCHEMA"}),
    "G2": frozenset(
        {"AUTHORITY_IDENTITY", "FACT_SCHEMA", "FACT_COVERAGE"}
    ),
    "G3": frozenset(
        {
            "AUTHORITY_IDENTITY",
            "FACT_SCHEMA",
            "FACT_COVERAGE",
            "CONSUMER_HANDOFF",
        }
    ),
}
_HORIZONTAL_AXES = frozenset({"column", "horizontal", "x"})
_VERTICAL_AXES = frozenset({"tier", "vertical", "y"})
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:%+!\-\[\]]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CONSUMER_RANK = {
    consumer: index
    for index, consumer in enumerate(REGISTERED_CONSUMER_ORDER)
}


class IntentClosureSchemaError(ValueError):
    """Closure data is structurally invalid or outside the closed registry."""


class IntentClosureTypeError(TypeError):
    """A closure boundary received a mutable or inexact Python value."""


def _canonical_json(data: object) -> str:
    try:
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise IntentClosureTypeError(
            "closure data must contain exact canonical JSON values"
        ) from exc


def _sha256(data: object) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _require_identifier(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise IntentClosureTypeError(f"{field} must be an exact string")
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise IntentClosureSchemaError(f"{field} must be a canonical identifier")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise IntentClosureTypeError(f"{field} must be an exact string")
    if _SHA256_RE.fullmatch(value) is None:
        raise IntentClosureSchemaError(
            f"{field} must be a lowercase 64-hex SHA-256 digest"
        )
    return value


def _require_exact_tuple(value: object, *, field: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise IntentClosureTypeError(f"{field} must be an exact tuple")
    return value


def _require_sorted_unique_strings(
    value: object,
    *,
    field: str,
) -> tuple[str, ...]:
    rows = _require_exact_tuple(value, field=field)
    normalized = tuple(
        _require_identifier(item, field=f"{field} item") for item in rows
    )
    if tuple(sorted(set(normalized))) != normalized:
        raise IntentClosureSchemaError(
            f"{field} must contain sorted unique values"
        )
    return normalized


def _require_exact_fields(
    data: object,
    *,
    expected: frozenset[str],
    schema: str,
) -> dict[str, object]:
    if type(data) is not dict:
        raise IntentClosureTypeError("canonical replay data must be an exact dict")
    if any(type(key) is not str for key in data):
        raise IntentClosureTypeError(
            "canonical replay data keys must be exact strings"
        )
    keys = frozenset(data)
    if keys != expected:
        raise IntentClosureSchemaError(
            "canonical schema fields disagree: "
            f"missing={sorted(expected - keys)}, unknown={sorted(keys - expected)}"
        )
    raw_schema = data["schema"]
    if type(raw_schema) is not str:
        raise IntentClosureTypeError("schema discriminator must be an exact string")
    if raw_schema != schema:
        raise IntentClosureSchemaError(f"canonical schema must be {schema!r}")
    return data


def _canonical_string_list(value: object, *, field: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise IntentClosureTypeError(f"{field} canonical data must be a list")
    if any(type(item) is not str for item in value):
        raise IntentClosureTypeError(
            f"{field} canonical data must contain exact strings"
        )
    return tuple(value)


@dataclass(frozen=True, order=True)
class IntentEntailment:
    atom_id: str
    fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.atom_id, field="atom_id")
        _require_sorted_unique_strings(self.fact_ids, field="fact_ids")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _ENTAILMENT_SCHEMA,
            "atom_id": self.atom_id,
            "fact_ids": list(self.fact_ids),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "IntentEntailment":
        row = _require_exact_fields(
            data,
            expected=frozenset({"schema", "atom_id", "fact_ids"}),
            schema=_ENTAILMENT_SCHEMA,
        )
        return cls(
            atom_id=row["atom_id"],  # type: ignore[arg-type]
            fact_ids=_canonical_string_list(row["fact_ids"], field="fact_ids"),
        )


@dataclass(frozen=True, order=True)
class ConsumerProjection:
    consumer: str
    fact_ids: tuple[str, ...]
    normalized_projection_sha256: str

    def __post_init__(self) -> None:
        consumer = _require_identifier(self.consumer, field="consumer")
        if consumer not in REGISTERED_CONSUMERS:
            raise IntentClosureSchemaError(
                f"consumer {consumer!r} has no closed registration"
            )
        _require_sorted_unique_strings(self.fact_ids, field="fact_ids")
        _require_sha256(
            self.normalized_projection_sha256,
            field="normalized_projection_sha256",
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _PROJECTION_SCHEMA,
            "consumer": self.consumer,
            "fact_ids": list(self.fact_ids),
            "normalized_projection_sha256": self.normalized_projection_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "ConsumerProjection":
        row = _require_exact_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "consumer",
                    "fact_ids",
                    "normalized_projection_sha256",
                }
            ),
            schema=_PROJECTION_SCHEMA,
        )
        return cls(
            consumer=row["consumer"],  # type: ignore[arg-type]
            fact_ids=_canonical_string_list(row["fact_ids"], field="fact_ids"),
            normalized_projection_sha256=row[
                "normalized_projection_sha256"
            ],  # type: ignore[arg-type]
        )


def _projection_sort_key(
    row: ConsumerProjection,
) -> tuple[int, tuple[str, ...], str]:
    return (
        _CONSUMER_RANK[row.consumer],
        row.fact_ids,
        row.normalized_projection_sha256,
    )


def _require_canonical_projections(
    projections: tuple[ConsumerProjection, ...],
) -> None:
    if len(set(projections)) != len(projections) or tuple(
        sorted(projections, key=_projection_sort_key)
    ) != projections:
        raise IntentClosureSchemaError(
            "projections must contain unique rows in semantic consumer order"
        )


@dataclass(frozen=True)
class IntentClosureReceipt:
    fact_set_sha256: str
    fact_authority_receipt_sha256: str | None
    selected_atom_ids: tuple[str, ...]
    covered_fact_ids: tuple[str, ...]
    implied_fact_ids: tuple[str, ...]
    missing_fact_ids: tuple[str, ...]
    projections: tuple[ConsumerProjection, ...]
    maximum_stage: str
    accepted: bool
    first_blocker: str | None

    def __post_init__(self) -> None:
        _require_sha256(self.fact_set_sha256, field="fact_set_sha256")
        authority_sha = self.fact_authority_receipt_sha256
        if authority_sha is not None:
            _require_sha256(
                authority_sha,
                field="fact_authority_receipt_sha256",
            )
        _require_sorted_unique_strings(
            self.selected_atom_ids,
            field="selected_atom_ids",
        )
        fact_groups = (
            self.covered_fact_ids,
            self.implied_fact_ids,
            self.missing_fact_ids,
        )
        for field, values in zip(
            ("covered_fact_ids", "implied_fact_ids", "missing_fact_ids"),
            fact_groups,
            strict=True,
        ):
            _require_sorted_unique_strings(values, field=field)
        if any(
            set(fact_groups[index]) & set(fact_groups[other])
            for index in range(len(fact_groups))
            for other in range(index + 1, len(fact_groups))
        ):
            raise IntentClosureSchemaError(
                "covered, implied, and missing fact IDs must be disjoint"
            )
        projections = _require_exact_tuple(self.projections, field="projections")
        if any(type(row) is not ConsumerProjection for row in projections):
            raise IntentClosureTypeError(
                "projections must contain exact ConsumerProjection values"
            )
        _require_canonical_projections(projections)
        closed_fact_ids = frozenset(self.covered_fact_ids) | frozenset(
            self.implied_fact_ids
        )
        projected_fact_ids = frozenset(
            fact_id for row in projections for fact_id in row.fact_ids
        )
        if not projected_fact_ids <= closed_fact_ids:
            raise IntentClosureSchemaError(
                "projection fact IDs must stay inside covered or implied closure"
            )
        if (
            type(self.maximum_stage) is not str
            or self.maximum_stage not in _PURE_CLOSURE_STAGES
        ):
            raise IntentClosureSchemaError(
                "maximum_stage must be one of G0 through G3"
            )
        stage_index = int(self.maximum_stage[1:])
        if type(self.accepted) is not bool:
            raise IntentClosureTypeError("accepted must be an exact bool")
        if self.accepted:
            if self.first_blocker is not None:
                raise IntentClosureSchemaError(
                    "accepted closure receipt must have first_blocker=None"
                )
            if self.maximum_stage == "G3" and not projections:
                raise IntentClosureSchemaError(
                    "accepted G3 closure receipt requires projection rows"
                )
        elif type(self.first_blocker) is not str:
            raise IntentClosureSchemaError(
                "rejected closure receipt must name a closed first_blocker"
            )
        elif self.first_blocker not in _PURE_STAGE_BLOCKERS[self.maximum_stage]:
            raise IntentClosureSchemaError(
                f"first_blocker {self.first_blocker!r} is not allowed at "
                f"maximum_stage {self.maximum_stage}"
            )
        if authority_sha is None and (
            self.accepted
            or self.first_blocker not in {"AUTHORITY_IDENTITY", "FACT_SCHEMA"}
        ):
            raise IntentClosureSchemaError(
                "receipt requires a successful authority receipt SHA"
            )
        if (
            authority_sha is not None
            and self.first_blocker == "AUTHORITY_IDENTITY"
        ):
            raise IntentClosureSchemaError(
                "AUTHORITY_IDENTITY cannot claim successful authority"
            )
        if stage_index >= 2 and self.missing_fact_ids and (
            self.accepted
            or self.first_blocker
            not in {"AUTHORITY_IDENTITY", "FACT_SCHEMA", "FACT_COVERAGE"}
        ):
            raise IntentClosureSchemaError(
                "G2 or later receipt with missing facts must fail at or before "
                "FACT_COVERAGE"
            )
        if self.first_blocker == "FACT_COVERAGE" and (
            stage_index < 2 or not self.missing_fact_ids
        ):
            raise IntentClosureSchemaError(
                "FACT_COVERAGE requires G2 or later and exact missing facts"
            )
        if self.first_blocker == "CONSUMER_HANDOFF" and stage_index < 3:
            raise IntentClosureSchemaError(
                "CONSUMER_HANDOFF requires G3 or later"
            )
        consumers = tuple(row.consumer for row in projections)
        duplicate_consumer = len(set(consumers)) != len(consumers)
        if duplicate_consumer and (
            self.accepted or self.first_blocker != "FACT_COVERAGE"
        ):
            raise IntentClosureSchemaError(
                "each consumer may have at most one canonical projection"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _RECEIPT_SCHEMA,
            "fact_set_sha256": self.fact_set_sha256,
            "fact_authority_receipt_sha256": (
                self.fact_authority_receipt_sha256
            ),
            "selected_atom_ids": list(self.selected_atom_ids),
            "covered_fact_ids": list(self.covered_fact_ids),
            "implied_fact_ids": list(self.implied_fact_ids),
            "missing_fact_ids": list(self.missing_fact_ids),
            "projections": [row.canonical_data() for row in self.projections],
            "maximum_stage": self.maximum_stage,
            "accepted": self.accepted,
            "first_blocker": self.first_blocker,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())

    @classmethod
    def from_data(cls, data: object) -> "IntentClosureReceipt":
        """Replay structural canonical data without re-evaluating authority."""

        row = _require_exact_fields(
            data,
            expected=frozenset(
                {
                    "schema",
                    "fact_set_sha256",
                    "fact_authority_receipt_sha256",
                    "selected_atom_ids",
                    "covered_fact_ids",
                    "implied_fact_ids",
                    "missing_fact_ids",
                    "projections",
                    "maximum_stage",
                    "accepted",
                    "first_blocker",
                }
            ),
            schema=_RECEIPT_SCHEMA,
        )
        raw_projections = row["projections"]
        if type(raw_projections) is not list:
            raise IntentClosureTypeError(
                "projections canonical data must be a list"
            )
        blocker = row["first_blocker"]
        if blocker is not None and type(blocker) is not str:
            raise IntentClosureTypeError(
                "first_blocker canonical data must be null or a string"
            )
        authority_sha = row["fact_authority_receipt_sha256"]
        if authority_sha is not None and type(authority_sha) is not str:
            raise IntentClosureTypeError(
                "fact_authority_receipt_sha256 canonical data must be null "
                "or a string"
            )
        return cls(
            fact_set_sha256=row["fact_set_sha256"],  # type: ignore[arg-type]
            fact_authority_receipt_sha256=authority_sha,
            selected_atom_ids=_canonical_string_list(
                row["selected_atom_ids"], field="selected_atom_ids"
            ),
            covered_fact_ids=_canonical_string_list(
                row["covered_fact_ids"], field="covered_fact_ids"
            ),
            implied_fact_ids=_canonical_string_list(
                row["implied_fact_ids"], field="implied_fact_ids"
            ),
            missing_fact_ids=_canonical_string_list(
                row["missing_fact_ids"], field="missing_fact_ids"
            ),
            projections=tuple(
                ConsumerProjection.from_data(projection)
                for projection in raw_projections
            ),
            maximum_stage=row["maximum_stage"],  # type: ignore[arg-type]
            accepted=row["accepted"],  # type: ignore[arg-type]
            first_blocker=blocker,
        )


def _implied_orders(
    *,
    facts: tuple[IntentFact, ...],
    covered_fact_ids: frozenset[str],
) -> frozenset[str]:
    ordinal_by_subject = {
        fact.subjects[0]: fact
        for fact in facts
        if fact.kind is IntentFactKind.ORDINAL_CELL
        and fact.fact_id in covered_fact_ids
    }
    implied: set[str] = set()
    for fact in facts:
        if fact.kind is not IntentFactKind.ORDER:
            continue
        payload = fact.payload
        if payload["axis"] not in _HORIZONTAL_AXES:
            continue
        first = ordinal_by_subject.get(fact.subjects[0])
        second = ordinal_by_subject.get(fact.subjects[1])
        if first is None or second is None:
            continue
        first_payload = first.payload
        second_payload = second.payload
        if first_payload["tier"] != second_payload["tier"]:
            continue
        first_column = first_payload["column"]
        second_column = second_payload["column"]
        if first_column == second_column:
            continue
        relation = payload["relation"]
        if first_column < second_column:
            compatible = relation in {"before", "left_of"}
        else:
            compatible = relation in {"after", "right_of"}
        if compatible:
            if (
                first.authority is fact.authority
                and second.authority is fact.authority
            ):
                implied.add(fact.fact_id)
    return frozenset(implied)


def _implied_facts(
    *,
    facts: tuple[IntentFact, ...],
    covered_fact_ids: frozenset[str],
) -> frozenset[str]:
    implied = set(
        _implied_orders(
            facts=facts,
            covered_fact_ids=covered_fact_ids,
        )
    )
    terminal_pairs = {
        IntentFactKind.MEMBER_PARTITION: IntentFactKind.MEMBER_CONSERVATION,
        IntentFactKind.EXTERNAL_PORT_DENOMINATOR: (
            IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY
        ),
    }
    targets_by_kind: dict[IntentFactKind, tuple[IntentFact, ...]] = {
        target_kind: tuple(
            fact for fact in facts if fact.kind is target_kind
        )
        for target_kind in terminal_pairs.values()
    }
    for fact in facts:
        target_kind = terminal_pairs.get(fact.kind)
        if target_kind is None or fact.fact_id not in covered_fact_ids:
            continue
        for target in targets_by_kind[target_kind]:
            if (
                target.subjects == fact.subjects
                and target.payload_json == fact.payload_json
                and target.authority is fact.authority
            ):
                implied.add(target.fact_id)
    return frozenset(implied)


def _fact_set_schema_is_consistent(facts: tuple[IntentFact, ...]) -> bool:
    """Check semantic relations whose consistency is independent of coverage."""

    ordinal_by_subject = {
        fact.subjects[0]: fact
        for fact in facts
        if fact.kind is IntentFactKind.ORDINAL_CELL
    }
    edges_by_axis: dict[str, dict[object, set[object]]] = {
        "horizontal": {},
        "vertical": {},
    }
    for fact in facts:
        if fact.kind is not IntentFactKind.ORDER:
            continue
        payload = fact.payload
        axis = payload["axis"]
        axis_family = "horizontal" if axis in _HORIZONTAL_AXES else "vertical"
        relation = payload["relation"]
        first_subject, second_subject = fact.subjects

        if axis_family == "horizontal":
            first_cell = ordinal_by_subject.get(first_subject)
            second_cell = ordinal_by_subject.get(second_subject)
            if first_cell is not None and second_cell is not None:
                first_payload = first_cell.payload
                second_payload = second_cell.payload
                if first_payload["tier"] == second_payload["tier"]:
                    first_column = first_payload["column"]
                    second_column = second_payload["column"]
                    compatible = (
                        first_column < second_column
                        and relation in {"before", "left_of"}
                    ) or (
                        first_column > second_column
                        and relation in {"after", "right_of"}
                    )
                    if not compatible:
                        return False
        else:
            first_cell = ordinal_by_subject.get(first_subject)
            second_cell = ordinal_by_subject.get(second_subject)
            if first_cell is not None and second_cell is not None:
                first_tier = first_cell.payload["tier"]
                second_tier = second_cell.payload["tier"]
                compatible = (
                    first_tier < second_tier
                    and relation in {"before", "below"}
                ) or (
                    first_tier > second_tier
                    and relation in {"above", "after"}
                )
                if not compatible:
                    return False

        if relation in {"before", "left_of", "below"}:
            source, target = first_subject, second_subject
        else:
            source, target = second_subject, first_subject
        edges_by_axis[axis_family].setdefault(source, set()).add(target)
        edges_by_axis[axis_family].setdefault(target, set())

    for graph in edges_by_axis.values():
        visiting: set[object] = set()
        visited: set[object] = set()

        def visit(node: object) -> bool:
            if node in visiting:
                return False
            if node in visited:
                return True
            visiting.add(node)
            if any(not visit(target) for target in graph[node]):
                return False
            visiting.remove(node)
            visited.add(node)
            return True

        if any(not visit(node) for node in graph):
            return False
    return True


def _consumer_handoff_is_complete(
    *,
    facts: tuple[IntentFact, ...],
    projections: tuple[ConsumerProjection, ...],
) -> bool:
    target_by_id = {fact.fact_id: fact for fact in facts}
    if len({row.consumer for row in projections}) != len(projections):
        return False
    projection_by_consumer = {row.consumer: row for row in projections}
    for projection in projections:
        for fact_id in projection.fact_ids:
            fact = target_by_id.get(fact_id)
            if fact is None or projection.consumer not in fact.required_consumers:
                return False
    for fact in facts:
        for consumer in fact.required_consumers:
            projection = projection_by_consumer.get(consumer)
            if projection is None or fact.fact_id not in projection.fact_ids:
                return False
    placement_initial = projection_by_consumer.get("placement_initial")
    placement_replay = projection_by_consumer.get("placement_replay")
    if (
        placement_initial is not None
        and placement_replay is not None
        and placement_initial.normalized_projection_sha256
        != placement_replay.normalized_projection_sha256
    ):
        return False
    return True


def evaluate_intent_closure(
    *,
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
    entailments: tuple[IntentEntailment, ...],
    selected_atom_ids: tuple[str, ...],
    projections: tuple[ConsumerProjection, ...],
    maximum_stage: str,
) -> IntentClosureReceipt:
    """Evaluate semantic coverage and consumer handoff without performing I/O."""

    if type(fact_set) is not IntentFactSet:
        raise IntentClosureTypeError("fact_set must be an exact IntentFactSet")
    if type(source_inventory) is not SourceInventory:
        raise IntentClosureTypeError(
            "source_inventory must be an exact SourceInventory"
        )
    rows = _require_exact_tuple(entailments, field="entailments")
    if any(type(row) is not IntentEntailment for row in rows):
        raise IntentClosureTypeError(
            "entailments must contain exact IntentEntailment values"
        )
    atom_ids = tuple(row.atom_id for row in rows)
    selected = _require_sorted_unique_strings(
        selected_atom_ids,
        field="selected_atom_ids",
    )
    projection_rows = _require_exact_tuple(projections, field="projections")
    if any(type(row) is not ConsumerProjection for row in projection_rows):
        raise IntentClosureTypeError(
            "projections must contain exact ConsumerProjection values"
        )
    _require_canonical_projections(projection_rows)
    if type(maximum_stage) is not str or maximum_stage not in _STAGES:
        raise IntentClosureSchemaError("maximum_stage must be one of G0 through G7")
    if maximum_stage not in _PURE_CLOSURE_STAGES:
        raise IntentClosureSchemaError(
            "ORACLE_SCOPE: pure intent closure supports only G0 through G3"
        )

    target_fact_ids = frozenset(fact.fact_id for fact in fact_set.facts)
    try:
        authority_receipt = verify_intent_fact_authority(
            fact_set=fact_set,
            source_inventory=source_inventory,
        )
    except IntentFactAuthorityError as exc:
        selected_set = frozenset(selected)
        covered = frozenset(
            fact_id
            for row in rows
            if row.atom_id in selected_set
            for fact_id in row.fact_ids
            if fact_id in target_fact_ids
        )
        return IntentClosureReceipt(
            fact_set_sha256=fact_set.sha256,
            fact_authority_receipt_sha256=None,
            selected_atom_ids=selected,
            covered_fact_ids=tuple(sorted(covered)),
            implied_fact_ids=(),
            missing_fact_ids=tuple(sorted(target_fact_ids - covered)),
            projections=(),
            maximum_stage=maximum_stage,
            accepted=False,
            first_blocker=exc.blocker,
        )
    selected_set = frozenset(selected)
    covered_target_ids = frozenset(
        fact_id
        for row in rows
        if row.atom_id in selected_set
        for fact_id in row.fact_ids
        if fact_id in target_fact_ids
    )
    entailment_schema_invalid = (
        tuple(sorted(set(rows))) != rows
        or len(set(atom_ids)) != len(atom_ids)
        or bool(frozenset(selected) - frozenset(atom_ids))
        or any(
            not frozenset(row.fact_ids) <= target_fact_ids
            for row in rows
        )
    )
    if entailment_schema_invalid or not _fact_set_schema_is_consistent(
        fact_set.facts
    ):
        return IntentClosureReceipt(
            fact_set_sha256=fact_set.sha256,
            fact_authority_receipt_sha256=authority_receipt.sha256,
            selected_atom_ids=selected,
            covered_fact_ids=tuple(sorted(covered_target_ids)),
            implied_fact_ids=(),
            missing_fact_ids=tuple(
                sorted(target_fact_ids - covered_target_ids)
            ),
            projections=(),
            maximum_stage=maximum_stage,
            accepted=False,
            first_blocker="FACT_SCHEMA",
        )

    covered = frozenset(
        fact_id
        for row in rows
        if row.atom_id in selected_set
        for fact_id in row.fact_ids
    )
    stage_index = int(maximum_stage[1:])
    implied = frozenset()
    if stage_index >= 2:
        implied = _implied_facts(
            facts=fact_set.facts,
            covered_fact_ids=covered,
        ) - covered
    missing = target_fact_ids - covered - implied

    blocker: str | None = None
    if stage_index >= 2 and missing:
        blocker = "FACT_COVERAGE"
    elif stage_index >= 3 and not _consumer_handoff_is_complete(
        facts=fact_set.facts,
        projections=projection_rows,
    ):
        blocker = "CONSUMER_HANDOFF"

    receipt_projections = projection_rows
    if blocker == "FACT_COVERAGE":
        receipt_projections = ()
    elif blocker == "CONSUMER_HANDOFF":
        closed_fact_ids = covered | implied
        within_closure = tuple(
            row
            for row in projection_rows
            if frozenset(row.fact_ids) <= closed_fact_ids
        )
        consumer_counts = {
            row.consumer: sum(
                candidate.consumer == row.consumer
                for candidate in within_closure
            )
            for row in within_closure
        }
        receipt_projections = tuple(
            row
            for row in within_closure
            if consumer_counts[row.consumer] == 1
        )

    return IntentClosureReceipt(
        fact_set_sha256=fact_set.sha256,
        fact_authority_receipt_sha256=authority_receipt.sha256,
        selected_atom_ids=selected,
        covered_fact_ids=tuple(sorted(covered)),
        implied_fact_ids=tuple(sorted(implied)),
        missing_fact_ids=tuple(sorted(missing)),
        projections=receipt_projections,
        maximum_stage=maximum_stage,
        accepted=blocker is None,
        first_blocker=blocker,
    )


def verify_intent_closure_receipt(
    *,
    receipt: IntentClosureReceipt,
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
    entailments: tuple[IntentEntailment, ...],
    projections: tuple[ConsumerProjection, ...],
) -> IntentClosureReceipt:
    """Recompute a structural receipt and reject any semantic mismatch."""

    if type(receipt) is not IntentClosureReceipt:
        raise IntentClosureTypeError(
            "receipt must be an exact IntentClosureReceipt"
        )
    expected = evaluate_intent_closure(
        fact_set=fact_set,
        source_inventory=source_inventory,
        entailments=entailments,
        selected_atom_ids=receipt.selected_atom_ids,
        projections=projections,
        maximum_stage=receipt.maximum_stage,
    )
    if expected != receipt:
        raise IntentClosureSchemaError(
            "intent closure receipt recomputation mismatch"
        )
    return receipt


__all__ = [
    "ConsumerProjection",
    "IntentClosureReceipt",
    "IntentClosureSchemaError",
    "IntentClosureTypeError",
    "IntentEntailment",
    "evaluate_intent_closure",
    "verify_intent_closure_receipt",
]
