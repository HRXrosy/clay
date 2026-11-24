"""Coordinate-free spatial IR with canonical orthogonal topology DAGs.

The serialized graph stores only a DAG's unique transitive reduction.  A
reachability closure is derived on demand and is deliberately absent from the
canonical form, keeping the representation aligned with the existing
sequence-pair and difference-constraint placement contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import heapq
import json
import re


_DAG_SCHEMA = "clay.spatial-orthogonal-dag/v1"
_IR_SCHEMA = "clay.abstract-spatial-ir/v1"
SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX = "aggregate_alignment:"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DENIED_COORDINATE_TOKENS = frozenset(
    {"bbox", "centroid", "coordinate", "coordinates", "origin", "offset", "x", "y"}
)


class SpatialIrSchemaError(ValueError):
    """The spatial representation is cyclic, incomplete, or non-canonical."""


class SpatialFactKind(str, Enum):
    ALIGNMENT = "alignment"
    SYMMETRY = "symmetry"
    COMMON_CENTROID = "common_centroid"
    LATTICE_SEQUENCE = "lattice_sequence"
    CLASS_RANK_SEQUENCE = "class_rank_sequence"
    SCOPE_RANK_SEQUENCE = "scope_rank_sequence"
    SHARED_DIFFUSION = "shared_diffusion"
    DUMMY_FLANK = "dummy_flank"
    EXACT_PARTITION = "exact_partition"
    CONTAINMENT = "containment"
    SAME_WELL_DOMAIN = "same_well_domain"
    GUARDED_BY = "guarded_by"
    PIN_FACES = "pin_faces"
    CORRIDOR_BETWEEN = "corridor_between"
    KELVIN_INDEPENDENT = "kelvin_independent"
    AGGREGATE_ORDER = "aggregate_order"
    AGGREGATE_ALIGNMENT = "aggregate_alignment"


_FACT_ALLOWED_ATTRIBUTES: dict[SpatialFactKind, frozenset[str]] = {
    SpatialFactKind.ALIGNMENT: frozenset({"axis", "reference"}),
    SpatialFactKind.SYMMETRY: frozenset({"axis", "mode"}),
    SpatialFactKind.COMMON_CENTROID: frozenset({"axis", "pattern"}),
    SpatialFactKind.LATTICE_SEQUENCE: frozenset(
        {"axis", "sequence", "transform_class"}
    ),
    SpatialFactKind.CLASS_RANK_SEQUENCE: frozenset(
        {"axis", "class_key", "sequence"}
    ),
    SpatialFactKind.SCOPE_RANK_SEQUENCE: frozenset({"axis", "scope", "sequence"}),
    SpatialFactKind.SHARED_DIFFUSION: frozenset({"layer_class", "terminal"}),
    SpatialFactKind.DUMMY_FLANK: frozenset({"device_class", "side"}),
    SpatialFactKind.EXACT_PARTITION: frozenset({"complete", "partition_id"}),
    SpatialFactKind.CONTAINMENT: frozenset({"scope"}),
    SpatialFactKind.SAME_WELL_DOMAIN: frozenset({"body_relation", "sharing"}),
    SpatialFactKind.GUARDED_BY: frozenset({"guard_domain"}),
    SpatialFactKind.PIN_FACES: frozenset(
        {"access_class", "direction", "terminal"}
    ),
    SpatialFactKind.CORRIDOR_BETWEEN: frozenset(
        {
            "capacity_class",
            "corridor_class",
            "orientation",
            "route_class",
            "shield_reference",
        }
    ),
    SpatialFactKind.KELVIN_INDEPENDENT: frozenset(
        {"force_net", "sense_net", "target_terminal"}
    ),
    SpatialFactKind.AGGREGATE_ORDER: frozenset(
        {"after_scope", "axis", "before_scope"}
    ),
    SpatialFactKind.AGGREGATE_ALIGNMENT: frozenset(
        {"axis", "edge", "first_scope", "second_scope", "tolerance_ppm"}
    ),
}


def _exact_tuple(value: object, *, field: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be an exact tuple")
    return value


def _canonical_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SpatialIrSchemaError(f"{field} must be a non-empty canonical string")
    return value


def _reject_coordinate_key(value: str, *, field: str) -> None:
    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", value.casefold()) if token)
    if any(token in _DENIED_COORDINATE_TOKENS for token in tokens):
        raise SpatialIrSchemaError(f"{field} is a coordinate-bearing ASIR key")


def _canonical_decimal(value: object, *, field: str) -> Decimal:
    if type(value) is float:
        raise TypeError(f"{field} cannot be a binary float; use Decimal, int, or str")
    if type(value) not in {Decimal, int, str}:
        raise TypeError(f"{field} must be an exact Decimal, int, or string")
    try:
        result = value if type(value) is Decimal else Decimal(value)  # type: ignore[arg-type]
    except InvalidOperation as exc:
        raise SpatialIrSchemaError(f"{field} is not an exact decimal") from exc
    if not result.is_finite():
        raise SpatialIrSchemaError(f"{field} must be finite")
    return Decimal(0) if result == 0 else result.normalize()


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _canonical_nodes(nodes: object) -> tuple[str, ...]:
    rows = _exact_tuple(nodes, field="nodes")
    if any(type(node) is not str or not node or node != node.strip() for node in rows):
        raise SpatialIrSchemaError("nodes must contain non-empty canonical strings")
    if len(set(rows)) != len(rows):
        raise SpatialIrSchemaError("nodes must be unique")
    return tuple(sorted(rows))  # type: ignore[arg-type]


def _canonical_edges(
    edges: object,
    *,
    known_nodes: frozenset[str],
) -> tuple[tuple[str, str], ...]:
    rows = _exact_tuple(edges, field="edges")
    normalized: set[tuple[str, str]] = set()
    for edge in rows:
        if type(edge) is not tuple or len(edge) != 2:
            raise TypeError("each edge must be an exact two-item tuple")
        before, after = edge
        if type(before) is not str or type(after) is not str:
            raise TypeError("edge endpoints must be exact strings")
        if before not in known_nodes or after not in known_nodes:
            raise SpatialIrSchemaError(
                f"edge {before!r}->{after!r} names an unknown node"
            )
        if before == after:
            raise SpatialIrSchemaError("a strict order edge cannot be a self-loop")
        normalized.add((before, after))
    return tuple(sorted(normalized))


def _topological_order(
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for before, after in edges:
        outgoing[before].add(after)
        indegree[after] += 1

    ready = [node for node in nodes if indegree[node] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for after in sorted(outgoing[node]):
            indegree[after] -= 1
            if indegree[after] == 0:
                heapq.heappush(ready, after)
    if len(order) != len(nodes):
        raise SpatialIrSchemaError("orthogonal order graph contains a cycle")
    return tuple(order)


def _reachability(
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> dict[str, frozenset[str]]:
    order = _topological_order(nodes, edges)
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    for before, after in edges:
        outgoing[before].add(after)

    reach: dict[str, set[str]] = {node: set() for node in nodes}
    for node in reversed(order):
        for after in sorted(outgoing[node]):
            reach[node].add(after)
            reach[node].update(reach[after])
    return {node: frozenset(reach[node]) for node in nodes}


def _transitive_reduction(
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    reach = _reachability(nodes, edges)
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    for before, after in edges:
        outgoing[before].add(after)

    reduced: list[tuple[str, str]] = []
    for before, after in edges:
        has_alternate_path = any(
            successor != after and after in reach[successor]
            for successor in outgoing[before]
        )
        if not has_alternate_path:
            reduced.append((before, after))
    return tuple(sorted(reduced))


@dataclass(frozen=True, slots=True)
class OrthogonalDag:
    """One strict horizontal or vertical order relation over canonical nodes."""

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        nodes = _canonical_nodes(self.nodes)
        edges = _canonical_edges(self.edges, known_nodes=frozenset(nodes))
        _topological_order(nodes, edges)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", _transitive_reduction(nodes, edges))

    def reaches(self, before: str, after: str) -> bool:
        if before not in self.nodes or after not in self.nodes:
            raise SpatialIrSchemaError("reachability query names an unknown node")
        return after in _reachability(self.nodes, self.edges)[before]

    def closure_edges(self) -> tuple[tuple[str, str], ...]:
        reach = _reachability(self.nodes, self.edges)
        return tuple(
            (before, after)
            for before in self.nodes
            for after in sorted(reach[before])
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": _DAG_SCHEMA,
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
        }


class MacroOrderStrength(str, Enum):
    """Physical meaning carried by one aggregate macro-order edge."""

    CENTER = "center_order"
    CENTER_TRACK_FLOOR = "center_order_with_track_floor"
    ENVELOPE_SEPARATION = "envelope_separation"
    MEMBER_BEFORE_CENTER = "member_before_aggregate_center"
    MEMBER_BEFORE_CENTER_TRACK_FLOOR = (
        "member_before_aggregate_center_with_track_floor"
    )
    LEADING_EDGE = "leading_edge_order"
    LEADING_EDGE_EQUAL = "leading_edge_equal"


@dataclass(frozen=True, slots=True, order=True)
class MacroDagNode:
    """One sibling macro and its exact semantic-occurrence ownership."""

    node_id: str
    member_nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        _canonical_text(self.node_id, field="macro DAG node id")
        members = _exact_tuple(self.member_nodes, field="macro DAG member nodes")
        if not members:
            raise SpatialIrSchemaError("macro DAG node requires at least one member")
        canonical = tuple(
            sorted(
                _canonical_text(item, field="macro DAG member node")
                for item in members
            )
        )
        if len(set(canonical)) != len(canonical):
            raise SpatialIrSchemaError("macro DAG member nodes must be unique")
        object.__setattr__(self, "member_nodes", canonical)

    def canonical_data(self) -> dict[str, object]:
        return {"node_id": self.node_id, "member_nodes": list(self.member_nodes)}


@dataclass(frozen=True, slots=True, order=True)
class MacroDagOrder:
    """One typed strict sibling relation on a horizontal or vertical axis."""

    axis: str
    before: str
    after: str
    strength: MacroOrderStrength

    def __post_init__(self) -> None:
        if self.axis not in {"horizontal", "vertical"}:
            raise SpatialIrSchemaError("macro DAG order axis is unsupported")
        _canonical_text(self.before, field="macro DAG order before node")
        _canonical_text(self.after, field="macro DAG order after node")
        if self.before == self.after:
            raise SpatialIrSchemaError("macro DAG order cannot be a self edge")
        if type(self.strength) is not MacroOrderStrength:
            raise TypeError("macro DAG order strength must be exact")

    def canonical_data(self) -> dict[str, str]:
        return {
            "axis": self.axis,
            "before": self.before,
            "after": self.after,
            "strength": self.strength.value,
        }


@dataclass(frozen=True, slots=True, order=True)
class MacroDagAlignment:
    """One exact aggregate alignment among sibling macro nodes."""

    axis: str
    nodes: tuple[str, ...]
    edge: str = "center"
    tolerance_ppm: int = 0

    def __post_init__(self) -> None:
        if self.axis not in {"horizontal", "vertical"}:
            raise SpatialIrSchemaError("macro DAG alignment axis is unsupported")
        if self.edge not in {"center", "leading", "trailing"}:
            raise SpatialIrSchemaError("macro DAG alignment edge is unsupported")
        if (
            type(self.tolerance_ppm) is not int
            or not 0 <= self.tolerance_ppm <= 1_000_000
        ):
            raise SpatialIrSchemaError(
                "macro DAG alignment tolerance must be exact ppm in [0, 1000000]"
            )
        raw = _exact_tuple(self.nodes, field="macro DAG alignment nodes")
        nodes = tuple(
            sorted(_canonical_text(item, field="macro DAG alignment node") for item in raw)
        )
        if len(nodes) < 2 or len(set(nodes)) != len(nodes):
            raise SpatialIrSchemaError(
                "macro DAG alignment requires at least two unique nodes"
            )
        object.__setattr__(self, "nodes", nodes)

    def canonical_data(self) -> dict[str, object]:
        payload: dict[str, object] = {"axis": self.axis, "nodes": list(self.nodes)}
        if self.edge != "center":
            payload["edge"] = self.edge
        if self.tolerance_ppm:
            payload["tolerance_ppm"] = self.tolerance_ppm
        return payload


@dataclass(frozen=True, slots=True)
class MacroDagLevel:
    """One exact sibling partition with typed orders and orthogonal DAGs."""

    scope_id: str
    parent_node: str | None
    nodes: tuple[MacroDagNode, ...]
    orders: tuple[MacroDagOrder, ...] = ()
    alignments: tuple[MacroDagAlignment, ...] = ()

    def __post_init__(self) -> None:
        _canonical_text(self.scope_id, field="macro DAG scope id")
        if self.scope_id.startswith(
            SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
        ):
            raise SpatialIrSchemaError(
                "macro DAG scope uses the reserved supplemental alignment namespace"
            )
        if self.parent_node is not None:
            _canonical_text(self.parent_node, field="macro DAG parent node")
        raw_nodes = _exact_tuple(self.nodes, field="macro DAG nodes")
        if not raw_nodes or any(type(item) is not MacroDagNode for item in raw_nodes):
            raise TypeError("macro DAG nodes must contain exact MacroDagNode values")
        nodes = tuple(sorted(raw_nodes, key=lambda item: item.node_id))
        node_ids = tuple(item.node_id for item in nodes)
        if len(set(node_ids)) != len(node_ids):
            raise SpatialIrSchemaError("macro DAG sibling ids must be unique")
        owned = tuple(member for item in nodes for member in item.member_nodes)
        if len(set(owned)) != len(owned):
            raise SpatialIrSchemaError("macro DAG siblings must be an exact partition")

        raw_orders = _exact_tuple(self.orders, field="macro DAG orders")
        if any(type(item) is not MacroDagOrder for item in raw_orders):
            raise TypeError("macro DAG orders must contain exact MacroDagOrder values")
        order_keys = tuple((item.axis, item.before, item.after) for item in raw_orders)
        if len(set(order_keys)) != len(order_keys):
            raise SpatialIrSchemaError("macro DAG orders must be unique")
        if any(
            item.before not in node_ids or item.after not in node_ids
            for item in raw_orders
        ):
            raise SpatialIrSchemaError("macro DAG order names an unknown sibling")

        dags = {
            axis: OrthogonalDag(
                nodes=node_ids,
                edges=tuple(
                    (item.before, item.after)
                    for item in raw_orders
                    if item.axis == axis
                ),
            )
            for axis in ("horizontal", "vertical")
        }
        # The union DAG above still validates cycles across all strengths.
        # A center-order path does not prove envelope separation: reduce hard
        # obligations only along paths with the same axis and relation strength.
        reduced_pairs = {
            (axis, strength, before, after)
            for axis in ("horizontal", "vertical")
            for strength in MacroOrderStrength
            for before, after in _transitive_reduction(
                node_ids,
                tuple(
                    (item.before, item.after)
                    for item in raw_orders
                    if item.axis == axis and item.strength is strength
                ),
            )
        }
        orders = tuple(
            sorted(
                item
                for item in raw_orders
                if (item.axis, item.strength, item.before, item.after) in reduced_pairs
            )
        )

        raw_alignments = _exact_tuple(
            self.alignments, field="macro DAG alignments"
        )
        if any(type(item) is not MacroDagAlignment for item in raw_alignments):
            raise TypeError(
                "macro DAG alignments must contain exact MacroDagAlignment values"
            )
        alignments = tuple(sorted(raw_alignments))
        if len(set(alignments)) != len(alignments):
            raise SpatialIrSchemaError("macro DAG alignments must be unique")
        claimed_by_axis: dict[str, set[str]] = {"horizontal": set(), "vertical": set()}
        for alignment in alignments:
            if not set(alignment.nodes) <= set(node_ids):
                raise SpatialIrSchemaError("macro DAG alignment names an unknown sibling")
            if claimed_by_axis[alignment.axis] & set(alignment.nodes):
                raise SpatialIrSchemaError(
                    "macro DAG node belongs to multiple same-axis alignments"
                )
            claimed_by_axis[alignment.axis].update(alignment.nodes)
            dag = dags[alignment.axis]
            if alignment.edge == "center" and any(
                dag.reaches(first, second) or dag.reaches(second, first)
                for index, first in enumerate(alignment.nodes)
                for second in alignment.nodes[index + 1 :]
            ):
                raise SpatialIrSchemaError(
                    "macro DAG siblings cannot be both aligned and ordered"
                )

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "alignments", alignments)

    @property
    def horizontal(self) -> OrthogonalDag:
        return OrthogonalDag(
            nodes=tuple(item.node_id for item in self.nodes),
            edges=tuple(
                (item.before, item.after)
                for item in self.orders
                if item.axis == "horizontal"
            ),
        )

    @property
    def vertical(self) -> OrthogonalDag:
        return OrthogonalDag(
            nodes=tuple(item.node_id for item in self.nodes),
            edges=tuple(
                (item.before, item.after)
                for item in self.orders
                if item.axis == "vertical"
            ),
        )

    @property
    def member_nodes(self) -> tuple[str, ...]:
        return tuple(sorted(member for item in self.nodes for member in item.member_nodes))

    def canonical_data(self) -> dict[str, object]:
        return {
            "scope_id": self.scope_id,
            "parent_node": self.parent_node,
            "nodes": [item.canonical_data() for item in self.nodes],
            "horizontal": self.horizontal.canonical_data(),
            "vertical": self.vertical.canonical_data(),
            "orders": [item.canonical_data() for item in self.orders],
            "alignments": [item.canonical_data() for item in self.alignments],
        }


@dataclass(frozen=True, slots=True)
class HierarchicalMacroDag:
    """Canonical macro levels whose children exactly cover their parent node."""

    levels: tuple[MacroDagLevel, ...]

    def __post_init__(self) -> None:
        raw = _exact_tuple(self.levels, field="macro DAG levels")
        if not raw or any(type(item) is not MacroDagLevel for item in raw):
            raise TypeError("macro DAG levels must contain exact MacroDagLevel values")
        levels = tuple(sorted(raw, key=lambda item: item.scope_id))
        by_scope = {item.scope_id: item for item in levels}
        if len(by_scope) != len(levels):
            raise SpatialIrSchemaError("macro DAG scope ids must be unique")
        roots = tuple(item for item in levels if item.parent_node is None)
        if len(roots) != 1:
            raise SpatialIrSchemaError("macro DAG requires exactly one root level")

        node_owner: dict[str, MacroDagNode] = {}
        for level in levels:
            for node in level.nodes:
                if node.node_id in node_owner:
                    raise SpatialIrSchemaError("macro DAG node ids must be globally unique")
                node_owner[node.node_id] = node
        for level in levels:
            if level.parent_node is None:
                continue
            if level.scope_id != level.parent_node:
                raise SpatialIrSchemaError(
                    "child macro scope id must equal its parent node id"
                )
            parent = node_owner.get(level.parent_node)
            if parent is None:
                raise SpatialIrSchemaError("child macro level names an unknown parent node")
            if level.member_nodes != parent.member_nodes:
                raise SpatialIrSchemaError(
                    "child macro siblings must exactly cover their parent node"
                )

        for level in levels:
            seen: set[str] = set()
            current = level
            while current.parent_node is not None:
                if current.scope_id in seen:
                    raise SpatialIrSchemaError("macro DAG hierarchy contains a cycle")
                seen.add(current.scope_id)
                parent_level = next(
                    (
                        candidate
                        for candidate in levels
                        if any(
                            node.node_id == current.parent_node
                            for node in candidate.nodes
                        )
                    ),
                    None,
                )
                if parent_level is None:
                    raise SpatialIrSchemaError("macro DAG hierarchy is disconnected")
                current = parent_level
        object.__setattr__(self, "levels", levels)

    @property
    def root(self) -> MacroDagLevel:
        return next(item for item in self.levels if item.parent_node is None)

    @property
    def max_depth(self) -> int:
        depth_by_scope = {self.root.scope_id: 1}
        pending = [self.root]
        while pending:
            parent = pending.pop()
            parent_depth = depth_by_scope[parent.scope_id]
            for child in self.levels:
                if child.parent_node and any(
                    node.node_id == child.parent_node for node in parent.nodes
                ):
                    depth_by_scope[child.scope_id] = parent_depth + 1
                    pending.append(child)
        return max(depth_by_scope.values())

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.hierarchical-macro-dag/v1",
            "levels": [item.canonical_data() for item in self.levels],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class SubjectBinding:
    """One named semantic role bound to an ASIR node."""

    role: str
    node: str

    def __post_init__(self) -> None:
        _canonical_text(self.role, field="subject role")
        _canonical_text(self.node, field="subject node")

    def canonical_data(self) -> dict[str, str]:
        return {"role": self.role, "node": self.node}


@dataclass(frozen=True, slots=True)
class HardInvariant:
    """A discrete overlay fact whose semantic signature is evidence-independent."""

    fact_id: str
    kind: SpatialFactKind
    bindings: tuple[SubjectBinding, ...]
    attributes: tuple[tuple[str, str], ...] = ()
    evidence_ids: tuple[str, ...] = ()
    derivation_rule: str = "declared/v1"

    def __post_init__(self) -> None:
        _canonical_text(self.fact_id, field="fact_id")
        if type(self.kind) is not SpatialFactKind:
            raise TypeError("kind must be an exact SpatialFactKind")

        raw_bindings = _exact_tuple(self.bindings, field="bindings")
        if not raw_bindings or any(type(row) is not SubjectBinding for row in raw_bindings):
            raise TypeError("bindings must contain exact SubjectBinding values")
        bindings = tuple(sorted(raw_bindings))  # type: ignore[arg-type]
        if len(set(bindings)) != len(bindings):
            raise SpatialIrSchemaError("bindings must be unique")

        raw_attributes = _exact_tuple(self.attributes, field="attributes")
        attributes: list[tuple[str, str]] = []
        for row in raw_attributes:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("attributes must contain exact key/value tuples")
            key = _canonical_text(row[0], field="attribute key")
            value = _canonical_text(row[1], field=f"attribute {key}")
            _reject_coordinate_key(key, field=f"attribute {key}")
            attributes.append((key, value))
        if len({key for key, _ in attributes}) != len(attributes):
            raise SpatialIrSchemaError("attribute keys must be unique")
        unknown_attributes = (
            {key for key, _ in attributes} - _FACT_ALLOWED_ATTRIBUTES[self.kind]
        )
        if unknown_attributes:
            raise SpatialIrSchemaError(
                f"{self.kind.value} contains an unregistered attribute: "
                f"{sorted(unknown_attributes)!r}"
            )

        raw_evidence = _exact_tuple(self.evidence_ids, field="evidence_ids")
        evidence = tuple(
            sorted(
                _canonical_text(item, field="evidence id")
                for item in raw_evidence
            )
        )
        if len(set(evidence)) != len(evidence):
            raise SpatialIrSchemaError("evidence ids must be unique")
        _canonical_text(self.derivation_rule, field="derivation_rule")

        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "attributes", tuple(sorted(attributes)))
        object.__setattr__(self, "evidence_ids", evidence)

    @property
    def semantic_signature(
        self,
    ) -> tuple[str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
        return (
            self.kind.value,
            tuple((binding.role, binding.node) for binding in self.bindings),
            self.attributes,
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "kind": self.kind.value,
            "bindings": [binding.canonical_data() for binding in self.bindings],
            "attributes": [list(row) for row in self.attributes],
            "evidence_ids": list(self.evidence_ids),
            "derivation_rule": self.derivation_rule,
        }


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """An exact coordinate-free measurement available to admission bands."""

    metric_id: str
    value: Decimal
    unit: str

    def __post_init__(self) -> None:
        metric_id = _canonical_text(self.metric_id, field="metric_id")
        _reject_coordinate_key(metric_id, field=f"metric {metric_id}")
        unit = _canonical_text(self.unit, field="metric unit")
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "value", _canonical_decimal(self.value, field="metric value"))
        object.__setattr__(self, "unit", unit)

    def canonical_data(self) -> dict[str, str]:
        return {
            "metric_id": self.metric_id,
            "value": _decimal_text(self.value),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticMetric:
    """A non-gating measurement retained for diagnosis and candidate ranking."""

    metric_id: str
    value: Decimal
    unit: str

    def __post_init__(self) -> None:
        metric_id = _canonical_text(self.metric_id, field="metric_id")
        _reject_coordinate_key(metric_id, field=f"metric {metric_id}")
        unit = _canonical_text(self.unit, field="metric unit")
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "value", _canonical_decimal(self.value, field="metric value"))
        object.__setattr__(self, "unit", unit)

    def canonical_data(self) -> dict[str, str]:
        return {
            "metric_id": self.metric_id,
            "value": _decimal_text(self.value),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class AbstractSpatialIR:
    """One immutable ASIR receipt over a shared H/V node universe."""

    source_clm_sha256: str
    horizontal: OrthogonalDag
    vertical: OrthogonalDag
    hard_facts: tuple[HardInvariant, ...] = ()
    observations: tuple[MetricObservation, ...] = ()
    diagnostics: tuple[DiagnosticMetric, ...] = ()
    macro_hierarchy: HierarchicalMacroDag | None = None

    def __post_init__(self) -> None:
        if type(self.source_clm_sha256) is not str or _SHA256_RE.fullmatch(
            self.source_clm_sha256
        ) is None:
            raise SpatialIrSchemaError("source_clm_sha256 must be lowercase SHA-256")
        if type(self.horizontal) is not OrthogonalDag or type(self.vertical) is not OrthogonalDag:
            raise TypeError("horizontal and vertical must be exact OrthogonalDag values")
        if self.horizontal.nodes != self.vertical.nodes:
            raise SpatialIrSchemaError("horizontal and vertical DAGs must share one node set")
        if self.macro_hierarchy is not None:
            if type(self.macro_hierarchy) is not HierarchicalMacroDag:
                raise TypeError("macro_hierarchy must be an exact HierarchicalMacroDag")
            if self.macro_hierarchy.root.member_nodes != self.horizontal.nodes:
                raise SpatialIrSchemaError(
                    "root macro level must exactly cover the ASIR occurrence nodes"
                )

        hard_facts = _exact_tuple(self.hard_facts, field="hard_facts")
        if any(type(fact) is not HardInvariant for fact in hard_facts):
            raise TypeError("hard_facts must contain exact HardInvariant values")
        if any(
            binding.node not in self.horizontal.nodes
            for fact in hard_facts
            for binding in fact.bindings  # type: ignore[union-attr]
        ):
            raise SpatialIrSchemaError("hard fact binding names an unknown ASIR node")
        if len({fact.fact_id for fact in hard_facts}) != len(hard_facts):  # type: ignore[union-attr]
            raise SpatialIrSchemaError("hard fact ids must be unique")
        if len({fact.semantic_signature for fact in hard_facts}) != len(hard_facts):  # type: ignore[union-attr]
            raise SpatialIrSchemaError("hard fact semantic signatures must be unique")

        observations = _exact_tuple(self.observations, field="observations")
        if any(type(item) is not MetricObservation for item in observations):
            raise TypeError("observations must contain exact MetricObservation values")
        diagnostics = _exact_tuple(self.diagnostics, field="diagnostics")
        if any(type(item) is not DiagnosticMetric for item in diagnostics):
            raise TypeError("diagnostics must contain exact DiagnosticMetric values")
        if len({item.metric_id for item in observations}) != len(observations):  # type: ignore[union-attr]
            raise SpatialIrSchemaError("observation metric ids must be unique")
        if len({item.metric_id for item in diagnostics}) != len(diagnostics):  # type: ignore[union-attr]
            raise SpatialIrSchemaError("diagnostic metric ids must be unique")

        object.__setattr__(
            self,
            "hard_facts",
            tuple(sorted(hard_facts, key=lambda fact: fact.fact_id)),  # type: ignore[union-attr]
        )
        object.__setattr__(
            self,
            "observations",
            tuple(sorted(observations, key=lambda item: item.metric_id)),  # type: ignore[union-attr]
        )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(sorted(diagnostics, key=lambda item: item.metric_id)),  # type: ignore[union-attr]
        )

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.horizontal.nodes

    def canonical_data(self) -> dict[str, object]:
        payload = {
            "schema": _IR_SCHEMA,
            "source_clm_sha256": self.source_clm_sha256,
            "horizontal": self.horizontal.canonical_data(),
            "vertical": self.vertical.canonical_data(),
            "hard_facts": [fact.canonical_data() for fact in self.hard_facts],
            "observations": [item.canonical_data() for item in self.observations],
            "diagnostics": [item.canonical_data() for item in self.diagnostics],
        }
        if self.macro_hierarchy is not None:
            payload["macro_hierarchy"] = self.macro_hierarchy.canonical_data()
        return payload

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


__all__ = [
    "AbstractSpatialIR",
    "DiagnosticMetric",
    "HardInvariant",
    "HierarchicalMacroDag",
    "MacroDagAlignment",
    "MacroDagLevel",
    "MacroDagNode",
    "MacroDagOrder",
    "MacroOrderStrength",
    "MetricObservation",
    "OrthogonalDag",
    "SpatialFactKind",
    "SpatialIrSchemaError",
    "SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX",
    "SubjectBinding",
]
