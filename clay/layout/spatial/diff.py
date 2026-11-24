"""Complete hierarchical differences over semantic overlays and ASIRs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json

from clay.layout.spatial.abstract_ir import (
    AbstractSpatialIR,
    HardInvariant,
    SpatialFactKind,
)
from clay.layout.spatial.refinement import (
    SpatialRequirementSet,
    evaluate_spatial_requirements,
)
from clay.layout.spatial.semantic import SemanticOccurrence, SemanticOverlay


@dataclass(frozen=True, slots=True)
class LogicalDeviceDelta:
    logical_owner: str
    parameter_master_status: str
    terminal_status: str
    body_well_status: str
    orientation_mismatch_nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.logical_owner) is not str
            or not self.logical_owner
            or any(
                value not in {"pass", "different"}
                for value in (
                    self.parameter_master_status,
                    self.terminal_status,
                    self.body_well_status,
                )
            )
            or type(self.orientation_mismatch_nodes) is not tuple
            or self.orientation_mismatch_nodes
            != tuple(sorted(set(self.orientation_mismatch_nodes)))
        ):
            raise ValueError("logical device delta is malformed")

    def canonical_data(self) -> dict[str, object]:
        return {
            "logical_owner": self.logical_owner,
            "parameter_master_status": self.parameter_master_status,
            "terminal_status": self.terminal_status,
            "body_well_status": self.body_well_status,
            "orientation_mismatch_nodes": list(self.orientation_mismatch_nodes),
        }


@dataclass(frozen=True, slots=True)
class RelationDelta:
    axis: str
    before_scope: str
    after_scope: str
    missing_edges: tuple[tuple[str, str], ...]
    extra_edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.axis not in {"horizontal", "vertical"}:
            raise ValueError("relation delta axis is unsupported")
        if any(
            type(value) is not str or not value
            for value in (self.before_scope, self.after_scope)
        ):
            raise ValueError("relation delta scope is malformed")
        for field in ("missing_edges", "extra_edges"):
            rows = getattr(self, field)
            if (
                type(rows) is not tuple
                or rows != tuple(sorted(set(rows)))
                or any(
                    type(row) is not tuple
                    or len(row) != 2
                    or any(type(value) is not str or not value for value in row)
                    for row in rows
                )
            ):
                raise ValueError(f"relation delta {field} is malformed")
        if not self.missing_edges and not self.extra_edges:
            raise ValueError("relation delta cannot be empty")

    @property
    def missing_count(self) -> int:
        return len(self.missing_edges)

    @property
    def extra_count(self) -> int:
        return len(self.extra_edges)

    def canonical_data(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "before_scope": self.before_scope,
            "after_scope": self.after_scope,
            "missing_count": self.missing_count,
            "extra_count": self.extra_count,
            "missing_edges": [list(row) for row in self.missing_edges],
            "extra_edges": [list(row) for row in self.extra_edges],
        }


@dataclass(frozen=True, slots=True)
class FactDelta:
    kind: SpatialFactKind
    scopes: tuple[str, ...]
    missing_facts: tuple[str, ...]
    extra_facts: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not SpatialFactKind:
            raise TypeError("fact delta kind must be SpatialFactKind")
        for field in ("scopes", "missing_facts", "extra_facts"):
            rows = getattr(self, field)
            if (
                type(rows) is not tuple
                or rows != tuple(sorted(set(rows)))
                or any(type(value) is not str or not value for value in rows)
            ):
                raise ValueError(f"fact delta {field} is malformed")
        if not self.scopes or (not self.missing_facts and not self.extra_facts):
            raise ValueError("fact delta cannot be empty")

    @property
    def missing_count(self) -> int:
        return len(self.missing_facts)

    @property
    def extra_count(self) -> int:
        return len(self.extra_facts)

    def canonical_data(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "scopes": list(self.scopes),
            "missing_count": self.missing_count,
            "extra_count": self.extra_count,
            "missing_facts": list(self.missing_facts),
            "extra_facts": list(self.extra_facts),
        }


@dataclass(frozen=True, slots=True)
class MetricDelta:
    metric_id: str
    unit: str
    reference_value: Decimal | None
    candidate_value: Decimal | None
    normalized_gap: Decimal | None

    def __post_init__(self) -> None:
        if (
            type(self.metric_id) is not str
            or not self.metric_id
            or type(self.unit) is not str
            or not self.unit
        ):
            raise ValueError("metric delta identity is malformed")
        for field in ("reference_value", "candidate_value", "normalized_gap"):
            value = getattr(self, field)
            if value is not None and type(value) is not Decimal:
                raise TypeError(f"metric delta {field} must be an exact Decimal")

    @staticmethod
    def _text(value: Decimal | None) -> str | None:
        if value is None:
            return None
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in {"", "-0"} else text

    def canonical_data(self) -> dict[str, str | None]:
        return {
            "metric_id": self.metric_id,
            "unit": self.unit,
            "reference_value": self._text(self.reference_value),
            "candidate_value": self._text(self.candidate_value),
            "normalized_gap": self._text(self.normalized_gap),
        }


@dataclass(frozen=True, slots=True)
class HierarchicalSpatialDiff:
    reference_asir_sha256: str
    candidate_asir_sha256: str
    strict_passed: bool
    electrical_semantic_status: str
    violation_counts: tuple[tuple[str, int], ...]
    device_deltas: tuple[LogicalDeviceDelta, ...]
    relation_deltas: tuple[RelationDelta, ...]
    fact_deltas: tuple[FactDelta, ...]
    metric_deltas: tuple[MetricDelta, ...]

    def __post_init__(self) -> None:
        if (
            type(self.strict_passed) is not bool
            or self.electrical_semantic_status not in {"pass", "different"}
            or type(self.violation_counts) is not tuple
            or self.violation_counts != tuple(sorted(self.violation_counts))
            or any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not str
                or not row[0]
                or type(row[1]) is not int
                or row[1] <= 0
                for row in self.violation_counts
            )
        ):
            raise ValueError("hierarchical spatial diff header is malformed")
        for value in (self.reference_asir_sha256, self.candidate_asir_sha256):
            if type(value) is not str or len(value) != 64:
                raise ValueError("hierarchical spatial diff ASIR identity is malformed")
        for rows, expected_type in (
            (self.device_deltas, LogicalDeviceDelta),
            (self.relation_deltas, RelationDelta),
            (self.fact_deltas, FactDelta),
            (self.metric_deltas, MetricDelta),
        ):
            if type(rows) is not tuple or any(type(row) is not expected_type for row in rows):
                raise TypeError("hierarchical spatial diff contains foreign rows")
        if self.strict_passed != (
            self.electrical_semantic_status == "pass" and not self.violation_counts
        ):
            raise ValueError("strict diff status differs from its blockers")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.hierarchical-spatial-diff/v1",
            "reference_asir_sha256": self.reference_asir_sha256,
            "candidate_asir_sha256": self.candidate_asir_sha256,
            "strict_passed": self.strict_passed,
            "electrical_semantic_status": self.electrical_semantic_status,
            "violation_counts": [list(row) for row in self.violation_counts],
            "device_deltas": [row.canonical_data() for row in self.device_deltas],
            "relation_deltas": [row.canonical_data() for row in self.relation_deltas],
            "fact_deltas": [row.canonical_data() for row in self.fact_deltas],
            "metric_deltas": [row.canonical_data() for row in self.metric_deltas],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def _base_occurrence_signature(row: SemanticOccurrence) -> tuple[object, ...]:
    return (
        row.logical_owner,
        row.device_class,
        row.parameter_key,
        row.master_content_sha256,
        row.master_artwork_sha256,
    )


def _fact_text(fact: HardInvariant) -> str:
    return json.dumps(
        {
            "kind": fact.kind.value,
            "bindings": [
                [binding.role, binding.node] for binding in fact.bindings
            ],
            "attributes": [list(row) for row in fact.attributes],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def diff_spatial_ir(
    *,
    reference_overlay: SemanticOverlay,
    candidate_overlay: SemanticOverlay,
    reference_ir: AbstractSpatialIR,
    candidate_ir: AbstractSpatialIR,
    scope_by_logical: tuple[tuple[str, str], ...] = (),
) -> HierarchicalSpatialDiff:
    """Return every semantic, order, fact, and continuous difference."""

    if any(
        type(value) is not expected
        for value, expected in (
            (reference_overlay, SemanticOverlay),
            (candidate_overlay, SemanticOverlay),
            (reference_ir, AbstractSpatialIR),
            (candidate_ir, AbstractSpatialIR),
        )
    ):
        raise TypeError("hierarchical diff inputs have foreign types")
    if (
        type(scope_by_logical) is not tuple
        or any(
            type(row) is not tuple
            or len(row) != 2
            or any(type(value) is not str or not value for value in row)
            for row in scope_by_logical
        )
        or len({row[0] for row in scope_by_logical}) != len(scope_by_logical)
    ):
        raise ValueError("logical scope map is malformed")
    scope_map = dict(scope_by_logical)
    reference_by_node = {row.node_id: row for row in reference_overlay.occurrences}
    candidate_by_node = {row.node_id: row for row in candidate_overlay.occurrences}
    shared_nodes = tuple(sorted(set(reference_by_node) & set(candidate_by_node)))

    ownership_equal = (
        reference_overlay.node_ids == candidate_overlay.node_ids
        and all(
            _base_occurrence_signature(reference_by_node[node])
            == _base_occurrence_signature(candidate_by_node[node])
            for node in shared_nodes
        )
    )
    terminals_equal = tuple(
        row.semantic_signature for row in reference_overlay.terminals
    ) == tuple(row.semantic_signature for row in candidate_overlay.terminals)
    body_equal = tuple(
        row.semantic_signature for row in reference_overlay.body_wells
    ) == tuple(row.semantic_signature for row in candidate_overlay.body_wells)
    electrical_status = (
        "pass" if ownership_equal and terminals_equal and body_equal else "different"
    )

    device_deltas = []
    owners = tuple(sorted(
        set(reference_overlay.logical_owners) | set(candidate_overlay.logical_owners)
    ))
    for owner in owners:
        reference_nodes = tuple(
            row for row in reference_overlay.occurrences if row.logical_owner == owner
        )
        candidate_nodes = tuple(
            row for row in candidate_overlay.occurrences if row.logical_owner == owner
        )
        candidate_owner_by_node = {row.node_id: row for row in candidate_nodes}
        parameter_master_status = "pass" if (
            tuple((row.node_id, _base_occurrence_signature(row)) for row in reference_nodes)
            == tuple((row.node_id, _base_occurrence_signature(row)) for row in candidate_nodes)
        ) else "different"
        reference_terminal = tuple(
            row.semantic_signature
            for row in reference_overlay.terminals
            if row.occurrence_id in {item.node_id for item in reference_nodes}
        )
        candidate_terminal = tuple(
            row.semantic_signature
            for row in candidate_overlay.terminals
            if row.occurrence_id in {item.node_id for item in candidate_nodes}
        )
        reference_body = tuple(
            row.semantic_signature
            for row in reference_overlay.body_wells
            if row.occurrence_id in {item.node_id for item in reference_nodes}
        )
        candidate_body = tuple(
            row.semantic_signature
            for row in candidate_overlay.body_wells
            if row.occurrence_id in {item.node_id for item in candidate_nodes}
        )
        orientation_mismatch = tuple(
            row.node_id
            for row in reference_nodes
            if row.node_id not in candidate_owner_by_node
            or row.physical.orientation
            != candidate_owner_by_node[row.node_id].physical.orientation
        )
        device_deltas.append(
            LogicalDeviceDelta(
                logical_owner=owner,
                parameter_master_status=parameter_master_status,
                terminal_status=(
                    "pass" if reference_terminal == candidate_terminal else "different"
                ),
                body_well_status=(
                    "pass" if reference_body == candidate_body else "different"
                ),
                orientation_mismatch_nodes=orientation_mismatch,
            )
        )

    owner_by_node = {
        **{node: row.logical_owner for node, row in candidate_by_node.items()},
        **{node: row.logical_owner for node, row in reference_by_node.items()},
    }

    def scope(node: str) -> str:
        logical = owner_by_node.get(node, node)
        return scope_map.get(logical, logical)

    relation_groups: dict[
        tuple[str, str, str], dict[str, set[tuple[str, str]]]
    ] = {}
    for axis, reference_edges, candidate_edges in (
        (
            "horizontal",
            set(reference_ir.horizontal.closure_edges()),
            set(candidate_ir.horizontal.closure_edges()),
        ),
        (
            "vertical",
            set(reference_ir.vertical.closure_edges()),
            set(candidate_ir.vertical.closure_edges()),
        ),
    ):
        for label, edges in (
            ("missing", reference_edges - candidate_edges),
            ("extra", candidate_edges - reference_edges),
        ):
            for edge in edges:
                key = (axis, scope(edge[0]), scope(edge[1]))
                relation_groups.setdefault(
                    key, {"missing": set(), "extra": set()}
                )[label].add(edge)
    relation_deltas = tuple(
        RelationDelta(
            axis=key[0],
            before_scope=key[1],
            after_scope=key[2],
            missing_edges=tuple(sorted(values["missing"])),
            extra_edges=tuple(sorted(values["extra"])),
        )
        for key, values in sorted(relation_groups.items())
    )

    reference_facts = {fact.semantic_signature: fact for fact in reference_ir.hard_facts}
    candidate_facts = {fact.semantic_signature: fact for fact in candidate_ir.hard_facts}
    fact_groups: dict[
        tuple[SpatialFactKind, tuple[str, ...]], dict[str, set[str]]
    ] = {}
    for label, facts in (
        (
            "missing",
            tuple(reference_facts[key] for key in set(reference_facts) - set(candidate_facts)),
        ),
        (
            "extra",
            tuple(candidate_facts[key] for key in set(candidate_facts) - set(reference_facts)),
        ),
    ):
        for fact in facts:
            scopes = tuple(sorted({scope(binding.node) for binding in fact.bindings}))
            key = (fact.kind, scopes)
            fact_groups.setdefault(key, {"missing": set(), "extra": set()})[
                label
            ].add(_fact_text(fact))
    fact_deltas = tuple(
        FactDelta(
            kind=key[0],
            scopes=key[1],
            missing_facts=tuple(sorted(values["missing"])),
            extra_facts=tuple(sorted(values["extra"])),
        )
        for key, values in sorted(
            fact_groups.items(), key=lambda row: (row[0][0].value, row[0][1])
        )
    )

    reference_metrics = {row.metric_id: row for row in reference_ir.observations}
    candidate_metrics = {row.metric_id: row for row in candidate_ir.observations}
    metric_deltas = []
    for metric_id in sorted(set(reference_metrics) | set(candidate_metrics)):
        reference_metric = reference_metrics.get(metric_id)
        candidate_metric = candidate_metrics.get(metric_id)
        reference_value = None if reference_metric is None else reference_metric.value
        candidate_value = None if candidate_metric is None else candidate_metric.value
        if (
            reference_value is None
            or candidate_value is None
            or reference_value == 0
        ):
            gap = None
        else:
            gap = (candidate_value - reference_value) / abs(reference_value)
        units = {
            row.unit for row in (reference_metric, candidate_metric) if row is not None
        }
        metric_deltas.append(
            MetricDelta(
                metric_id=metric_id,
                unit=next(iter(units)) if len(units) == 1 else "unit_mismatch",
                reference_value=reference_value,
                candidate_value=candidate_value,
                normalized_gap=gap,
            )
        )

    gate = evaluate_spatial_requirements(
        SpatialRequirementSet.strict_from_reference(reference_ir),
        candidate_ir,
    )
    violation_counts_by_code: dict[str, int] = {}
    for violation in gate.violations:
        key = violation.code.value
        violation_counts_by_code[key] = violation_counts_by_code.get(key, 0) + 1
    violation_counts = tuple(sorted(violation_counts_by_code.items()))
    return HierarchicalSpatialDiff(
        reference_asir_sha256=reference_ir.sha256,
        candidate_asir_sha256=candidate_ir.sha256,
        strict_passed=(gate.passed and electrical_status == "pass"),
        electrical_semantic_status=electrical_status,
        violation_counts=violation_counts,
        device_deltas=tuple(device_deltas),
        relation_deltas=relation_deltas,
        fact_deltas=fact_deltas,
        metric_deltas=tuple(metric_deltas),
    )


__all__ = [
    "FactDelta",
    "HierarchicalSpatialDiff",
    "LogicalDeviceDelta",
    "MetricDelta",
    "RelationDelta",
    "diff_spatial_ir",
]
