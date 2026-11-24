"""Deterministic projection from physical semantic bindings to abstract IR."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from clay.layout.spatial.abstract_ir import (
    AbstractSpatialIR,
    DiagnosticMetric,
    HardInvariant,
    HierarchicalMacroDag,
    MacroDagAlignment,
    MacroDagLevel,
    MacroDagNode,
    MacroDagOrder,
    MacroOrderStrength,
    MetricObservation,
    OrthogonalDag,
    SpatialFactKind,
    SubjectBinding,
)
from clay.layout.spatial.semantic import SemanticOccurrence, SemanticOverlay


class MacroDagDerivationError(ValueError):
    """A requested coordinate-free macro relation is not proved by the overlay."""


@dataclass(frozen=True, slots=True, order=True)
class AggregateAlignmentSpec:
    """Requested aggregate edge-alignment proof between two semantic scopes."""

    first_scope: str
    second_scope: str
    axis: str
    edge: str
    tolerance_ppm: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value or value != value.strip()
            for value in (self.first_scope, self.second_scope)
        ):
            raise MacroDagDerivationError(
                "aggregate alignment scopes must be canonical strings"
            )
        if self.first_scope == self.second_scope:
            raise MacroDagDerivationError(
                "aggregate alignment requires two distinct scopes"
            )
        first_scope, second_scope = sorted((self.first_scope, self.second_scope))
        object.__setattr__(self, "first_scope", first_scope)
        object.__setattr__(self, "second_scope", second_scope)
        if self.axis not in {"horizontal", "vertical"}:
            raise MacroDagDerivationError("aggregate alignment axis is unsupported")
        if self.edge not in {"leading", "trailing", "center"}:
            raise MacroDagDerivationError("aggregate alignment edge is unsupported")
        if (
            type(self.tolerance_ppm) is not int
            or not 0 <= self.tolerance_ppm <= 1_000_000
        ):
            raise MacroDagDerivationError(
                "aggregate alignment tolerance must be exact ppm in [0, 1000000]"
            )


@dataclass(frozen=True, slots=True, order=True)
class MacroDagNodeSpec:
    """Diagnostic extraction request binding one macro to logical owners."""

    node_id: str
    logical_owners: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.node_id) is not str or not self.node_id:
            raise MacroDagDerivationError("macro node spec id must be non-empty")
        if type(self.logical_owners) is not tuple or not self.logical_owners:
            raise MacroDagDerivationError("macro node spec requires logical owners")
        owners = tuple(sorted(self.logical_owners))
        if (
            any(type(owner) is not str or not owner for owner in owners)
            or len(set(owners)) != len(owners)
        ):
            raise MacroDagDerivationError(
                "macro node spec logical owners must be canonical and unique"
            )
        object.__setattr__(self, "logical_owners", owners)


@dataclass(frozen=True, slots=True)
class MacroDagLevelSpec:
    """One coordinate-free sibling partition and its required topology."""

    scope_id: str
    parent_node: str | None
    nodes: tuple[MacroDagNodeSpec, ...]
    orders: tuple[MacroDagOrder, ...] = ()
    alignments: tuple[MacroDagAlignment, ...] = ()

    def __post_init__(self) -> None:
        if type(self.scope_id) is not str or not self.scope_id:
            raise MacroDagDerivationError("macro level spec id must be non-empty")
        if self.parent_node is not None and (
            type(self.parent_node) is not str or not self.parent_node
        ):
            raise MacroDagDerivationError("macro level parent spec is malformed")
        if type(self.nodes) is not tuple or any(
            type(item) is not MacroDagNodeSpec for item in self.nodes
        ):
            raise TypeError("macro level spec nodes must be exact")
        if type(self.orders) is not tuple or any(
            type(item) is not MacroDagOrder for item in self.orders
        ):
            raise TypeError("macro level spec orders must be exact")
        if type(self.alignments) is not tuple or any(
            type(item) is not MacroDagAlignment for item in self.alignments
        ):
            raise TypeError("macro level spec alignments must be exact")


def _macro_node_box(
    occurrences: tuple[SemanticOccurrence, ...],
    owners: tuple[str, ...],
) -> tuple[int, int, int, int]:
    rows = tuple(row for row in occurrences if row.logical_owner in owners)
    if not rows:
        raise MacroDagDerivationError("macro node spec resolves no occurrences")
    return (
        min(row.physical.envelope_nm[0] for row in rows),
        min(row.physical.envelope_nm[1] for row in rows),
        max(row.physical.envelope_nm[2] for row in rows),
        max(row.physical.envelope_nm[3] for row in rows),
    )


def _derive_macro_hierarchy(
    occurrences: tuple[SemanticOccurrence, ...],
    specs: tuple[MacroDagLevelSpec, ...],
    *,
    alignment_tolerance_ppm: int,
) -> HierarchicalMacroDag | None:
    if type(specs) is not tuple:
        raise TypeError("macro level specs must be one exact tuple")
    if not specs:
        return None
    if any(type(item) is not MacroDagLevelSpec for item in specs):
        raise TypeError("macro level specs must contain exact values")
    if (
        type(alignment_tolerance_ppm) is not int
        or not 0 <= alignment_tolerance_ppm <= 1_000_000
    ):
        raise MacroDagDerivationError(
            "macro alignment tolerance must be exact ppm in [0, 1000000]"
        )
    known_owners = {row.logical_owner for row in occurrences}
    levels = []
    for spec in specs:
        node_specs = {item.node_id: item for item in spec.nodes}
        if len(node_specs) != len(spec.nodes):
            raise MacroDagDerivationError("macro level spec node ids must be unique")
        requested_owners = {
            owner for item in spec.nodes for owner in item.logical_owners
        }
        if not requested_owners <= known_owners:
            missing = min(requested_owners - known_owners)
            raise MacroDagDerivationError(
                f"macro level spec names unknown logical owner {missing!r}"
            )
        boxes = {
            node_id: _macro_node_box(occurrences, node.logical_owners)
            for node_id, node in node_specs.items()
        }
        for order in spec.orders:
            if order.before not in boxes or order.after not in boxes:
                raise MacroDagDerivationError(
                    "macro order spec names an unknown sibling"
                )
            lower = 0 if order.axis == "horizontal" else 1
            upper = 2 if order.axis == "horizontal" else 3
            before_box = boxes[order.before]
            after_box = boxes[order.after]
            if order.strength is MacroOrderStrength.ENVELOPE_SEPARATION:
                if before_box[upper] > after_box[lower]:
                    raise MacroDagDerivationError(
                        "requested envelope separation is not proved by reference"
                    )
            elif order.strength is MacroOrderStrength.LEADING_EDGE:
                if before_box[lower] >= after_box[lower]:
                    raise MacroDagDerivationError(
                        "requested leading-edge order is not proved by reference"
                    )
            elif order.strength is MacroOrderStrength.LEADING_EDGE_EQUAL:
                if before_box[lower] != after_box[lower]:
                    raise MacroDagDerivationError(
                        "requested leading-edge equality is not proved by reference"
                    )
            elif order.strength in {
                MacroOrderStrength.MEMBER_BEFORE_CENTER,
                MacroOrderStrength.MEMBER_BEFORE_CENTER_TRACK_FLOOR,
            }:
                before_members = tuple(
                    row
                    for row in occurrences
                    if row.logical_owner
                    in node_specs[order.before].logical_owners
                )
                if max(
                    row.physical.envelope_nm[lower]
                    + row.physical.envelope_nm[upper]
                    for row in before_members
                ) >= after_box[lower] + after_box[upper]:
                    raise MacroDagDerivationError(
                        "requested member-to-aggregate center order is not "
                        "proved by reference"
                    )
            elif (
                before_box[lower] + before_box[upper]
                >= after_box[lower] + after_box[upper]
            ):
                raise MacroDagDerivationError(
                    "requested aggregate center order is not proved by reference"
                )
        for alignment in spec.alignments:
            lower = 0 if alignment.axis == "horizontal" else 1
            upper = 2 if alignment.axis == "horizontal" else 3
            if not set(alignment.nodes) <= set(boxes):
                raise MacroDagDerivationError(
                    "macro alignment spec names an unknown sibling"
                )
            level_span = max(box[upper] for box in boxes.values()) - min(
                box[lower] for box in boxes.values()
            )
            base = boxes[alignment.nodes[0]]
            if alignment.edge == "leading":
                base_value = base[lower]
                scale = 1
            elif alignment.edge == "trailing":
                base_value = base[upper]
                scale = 1
            else:
                base_value = base[lower] + base[upper]
                scale = 2

            def alignment_value(node: str) -> int:
                box = boxes[node]
                if alignment.edge == "leading":
                    return box[lower]
                if alignment.edge == "trailing":
                    return box[upper]
                return box[lower] + box[upper]

            proof_tolerance_ppm = (
                alignment.tolerance_ppm or alignment_tolerance_ppm
            )
            if any(
                abs(base_value - alignment_value(node))
                * 1_000_000
                > scale * level_span * proof_tolerance_ppm
                for node in alignment.nodes[1:]
            ):
                raise MacroDagDerivationError(
                    "requested aggregate alignment exceeds reference tolerance"
                )
        levels.append(
            MacroDagLevel(
                scope_id=spec.scope_id,
                parent_node=spec.parent_node,
                nodes=tuple(
                    MacroDagNode(
                        node_id=item.node_id,
                        member_nodes=tuple(
                            sorted(
                                row.node_id
                                for row in occurrences
                                if row.logical_owner in item.logical_owners
                            )
                        ),
                    )
                    for item in spec.nodes
                ),
                orders=spec.orders,
                alignments=spec.alignments,
            )
        )
    return HierarchicalMacroDag(levels=tuple(levels))


def _separation_edges(
    occurrences: tuple[SemanticOccurrence, ...],
    *,
    axis: str,
) -> tuple[tuple[str, str], ...]:
    if axis not in {"horizontal", "vertical"}:
        raise ValueError("orthogonal separation axis is unsupported")
    edges = []
    for index, first in enumerate(occurrences):
        first_box = first.physical.envelope_nm
        for second in occurrences[index + 1 :]:
            second_box = second.physical.envelope_nm
            if axis == "horizontal":
                first_end, second_start = first_box[2], second_box[0]
                second_end, first_start = second_box[2], first_box[0]
            else:
                first_end, second_start = first_box[3], second_box[1]
                second_end, first_start = second_box[3], first_box[1]
            if first_end <= second_start:
                edges.append((first.node_id, second.node_id))
            elif second_end <= first_start:
                edges.append((second.node_id, first.node_id))
    return tuple(sorted(edges))


def _bindings(occurrences: tuple[SemanticOccurrence, ...]) -> tuple[SubjectBinding, ...]:
    return tuple(
        SubjectBinding(role=f"member_{index:03d}", node=row.node_id)
        for index, row in enumerate(sorted(occurrences, key=lambda row: row.node_id))
    )


def _evidence(occurrences: tuple[SemanticOccurrence, ...]) -> tuple[str, ...]:
    return tuple(sorted({item for row in occurrences for item in row.evidence_ids}))


def _partition_facts(
    occurrences: tuple[SemanticOccurrence, ...],
) -> tuple[HardInvariant, ...]:
    owners = tuple(sorted({row.logical_owner for row in occurrences}))
    return tuple(
        HardInvariant(
            fact_id=f"partition:{owner}",
            kind=SpatialFactKind.EXACT_PARTITION,
            bindings=_bindings(tuple(
                row for row in occurrences if row.logical_owner == owner
            )),
            attributes=(("complete", "true"), ("partition_id", owner)),
            evidence_ids=_evidence(tuple(
                row for row in occurrences if row.logical_owner == owner
            )),
            derivation_rule="semantic-owner-cover/v1",
        )
        for owner in owners
    )


def _axis_ranks(values: tuple[int, ...]) -> dict[int, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}


def _bounded_axis_ranks(
    values: tuple[int, ...], *, tolerance_nm: int
) -> dict[int, int]:
    """Rank exact coordinates with a bounded, non-chaining tolerance band."""

    if type(tolerance_nm) is not int or tolerance_nm < 0:
        raise ValueError("class-rank tolerance must be a non-negative integer")
    unique = tuple(sorted(set(values)))
    if not unique:
        raise ValueError("class-rank coordinate inventory is empty")
    rank_by_value: dict[int, int] = {}
    cluster_min = unique[0]
    rank = 0
    for value in unique:
        if value - cluster_min > tolerance_nm:
            rank += 1
            cluster_min = value
        rank_by_value[value] = rank
    return rank_by_value


def _lattice_facts(
    occurrences: tuple[SemanticOccurrence, ...],
) -> tuple[HardInvariant, ...]:
    facts = []
    for owner in sorted({row.logical_owner for row in occurrences}):
        members = tuple(
            sorted(
                (row for row in occurrences if row.logical_owner == owner),
                key=lambda row: row.occurrence_ordinal,
            )
        )
        # A singleton still has one unique 1x1 orthogonal lattice.  Emitting it
        # keeps exact-cover projections total when a multi-owner mosaic contains
        # both replicated devices and single inserted columns.
        x_ranks = _axis_ranks(tuple(row.physical.anchor_nm[0] for row in members))
        y_ranks = _axis_ranks(tuple(row.physical.anchor_nm[1] for row in members))
        sequence = ";".join(
            f"{row.occurrence_ordinal}:{x_ranks[row.physical.anchor_nm[0]]},"
            f"{y_ranks[row.physical.anchor_nm[1]]}"
            for row in members
        )
        transform_class = ",".join(row.physical.orientation for row in members)
        facts.append(
            HardInvariant(
                fact_id=f"lattice:{owner}",
                kind=SpatialFactKind.LATTICE_SEQUENCE,
                bindings=_bindings(members),
                attributes=(
                    ("axis", "orthogonal_rank"),
                    ("sequence", sequence),
                    ("transform_class", transform_class),
                ),
                evidence_ids=_evidence(members),
                derivation_rule="semantic-anchor-rank/v1",
            )
        )
    return tuple(facts)


def _alignment_facts(
    occurrences: tuple[SemanticOccurrence, ...],
) -> tuple[HardInvariant, ...]:
    facts = []
    for axis, coordinate_index in (("vertical", 0), ("horizontal", 1)):
        grouped: dict[int, list[SemanticOccurrence]] = {}
        for row in occurrences:
            grouped.setdefault(row.physical.anchor_nm[coordinate_index], []).append(row)
        for rank, coordinate in enumerate(sorted(grouped)):
            members = tuple(sorted(grouped[coordinate], key=lambda row: row.node_id))
            if len(members) <= 1:
                continue
            facts.append(
                HardInvariant(
                    fact_id=f"alignment:{axis}:{rank:03d}",
                    kind=SpatialFactKind.ALIGNMENT,
                    bindings=_bindings(members),
                    attributes=(("axis", axis), ("reference", "anchor_rank")),
                    evidence_ids=_evidence(members),
                    derivation_rule="semantic-anchor-equality/v1",
                )
            )
    return tuple(facts)


def _class_rank_facts(
    occurrences: tuple[SemanticOccurrence, ...],
) -> tuple[HardInvariant, ...]:
    """Encode cross-row ranks within one exact resolved-artwork class.

    Horizontal ranks remain exact.  Vertical ranks absorb only a bounded
    sub-device origin stagger (at most half the smallest resolved-master
    height), matching physical-row semantics without storing a coordinate or
    reference pitch.  Bands are span-bounded from their first member, so a
    staircase of small offsets cannot collapse two real rows.
    """

    groups: dict[str, list[SemanticOccurrence]] = {}
    for row in occurrences:
        groups.setdefault(row.master_artwork_sha256, []).append(row)
    facts = []
    for class_key, raw_members in sorted(groups.items()):
        members = tuple(sorted(raw_members, key=lambda row: row.node_id))
        if len(members) <= 1:
            continue
        for axis, coordinate in (("horizontal", 0), ("vertical", 1)):
            tolerance_nm = 0
            if axis == "vertical":
                tolerance_nm = min(
                    row.physical.envelope_nm[3]
                    - row.physical.envelope_nm[1]
                    for row in members
                ) // 2
            rank_by_value = _bounded_axis_ranks(
                tuple(row.physical.anchor_nm[coordinate] for row in members),
                tolerance_nm=tolerance_nm,
            )
            by_rank: dict[int, list[str]] = {}
            for row in members:
                by_rank.setdefault(
                    rank_by_value[row.physical.anchor_nm[coordinate]], []
                ).append(
                    row.node_id
                )
            sequence = ";".join(
                ",".join(sorted(by_rank[rank])) for rank in sorted(by_rank)
            )
            facts.append(
                HardInvariant(
                    fact_id=f"class_rank:{axis}:{class_key}",
                    kind=SpatialFactKind.CLASS_RANK_SEQUENCE,
                    bindings=_bindings(members),
                    attributes=(
                        ("axis", axis),
                        ("class_key", class_key),
                        ("sequence", sequence),
                    ),
                    evidence_ids=_evidence(members),
                    derivation_rule="semantic-class-bounded-rank/v2",
                )
            )
    return tuple(facts)


def _aggregate_order_facts(
    occurrences: tuple[SemanticOccurrence, ...],
    scope_by_logical: tuple[tuple[str, str], ...],
) -> tuple[HardInvariant, ...]:
    """Derive strict scope-envelope midpoint order without serializing geometry."""

    if type(scope_by_logical) is not tuple:
        raise TypeError("scope ownership must be an exact tuple")
    if not scope_by_logical:
        return ()
    if (
        scope_by_logical != tuple(sorted(scope_by_logical))
        or any(
            type(row) is not tuple
            or len(row) != 2
            or any(type(value) is not str or not value for value in row)
            for row in scope_by_logical
        )
        or len({logical for logical, _scope in scope_by_logical})
        != len(scope_by_logical)
        or {logical for logical, _scope in scope_by_logical}
        != {row.logical_owner for row in occurrences}
    ):
        raise ValueError("scope ownership must be one canonical exact logical cover")
    scope_map = dict(scope_by_logical)
    by_scope = {
        scope: tuple(
            row
            for row in occurrences
            if scope_map[row.logical_owner] == scope
        )
        for scope in sorted(set(scope_map.values()))
    }
    if len(by_scope) < 2 or any(not rows for rows in by_scope.values()):
        raise ValueError("aggregate order requires at least two non-empty scopes")

    envelopes = {
        scope: (
            min(row.physical.envelope_nm[0] for row in rows),
            min(row.physical.envelope_nm[1] for row in rows),
            max(row.physical.envelope_nm[2] for row in rows),
            max(row.physical.envelope_nm[3] for row in rows),
        )
        for scope, rows in by_scope.items()
    }
    facts = []
    scopes = tuple(sorted(by_scope))
    for first_index, first_scope in enumerate(scopes):
        for second_scope in scopes[first_index + 1 :]:
            for axis, lower_index, upper_index in (
                ("horizontal", 0, 2),
                ("vertical", 1, 3),
            ):
                first_center2 = (
                    envelopes[first_scope][lower_index]
                    + envelopes[first_scope][upper_index]
                )
                second_center2 = (
                    envelopes[second_scope][lower_index]
                    + envelopes[second_scope][upper_index]
                )
                if first_center2 == second_center2:
                    continue
                before_scope, after_scope = (
                    (first_scope, second_scope)
                    if first_center2 < second_center2
                    else (second_scope, first_scope)
                )
                before_rows = by_scope[before_scope]
                after_rows = by_scope[after_scope]
                bindings = tuple(
                    SubjectBinding(role=f"before_member_{index:03d}", node=row.node_id)
                    for index, row in enumerate(
                        sorted(before_rows, key=lambda row: row.node_id)
                    )
                ) + tuple(
                    SubjectBinding(role=f"after_member_{index:03d}", node=row.node_id)
                    for index, row in enumerate(
                        sorted(after_rows, key=lambda row: row.node_id)
                    )
                )
                facts.append(
                    HardInvariant(
                        fact_id=(
                            f"aggregate_order:{axis}:{before_scope}:{after_scope}"
                        ),
                        kind=SpatialFactKind.AGGREGATE_ORDER,
                        bindings=bindings,
                        attributes=(
                            ("after_scope", after_scope),
                            ("axis", axis),
                            ("before_scope", before_scope),
                        ),
                        evidence_ids=_evidence((*before_rows, *after_rows)),
                        derivation_rule="semantic-scope-envelope-order/v1",
                    )
                )
    return tuple(facts)


def _aggregate_alignment_facts(
    occurrences: tuple[SemanticOccurrence, ...],
    scope_by_logical: tuple[tuple[str, str], ...],
    specs: tuple[AggregateAlignmentSpec, ...],
) -> tuple[HardInvariant, ...]:
    """Prove requested scope-edge proximity against the full layout span."""

    if type(specs) is not tuple:
        raise TypeError("aggregate alignment specs must be an exact tuple")
    if not specs:
        return ()
    if any(type(item) is not AggregateAlignmentSpec for item in specs):
        raise TypeError("aggregate alignment specs must contain exact values")
    identities = tuple(
        (item.axis, item.edge, item.first_scope, item.second_scope)
        for item in specs
    )
    if len(set(identities)) != len(identities):
        raise MacroDagDerivationError(
            "aggregate alignment specs must have unique unordered scope pairs"
        )
    if (
        type(scope_by_logical) is not tuple
        or not scope_by_logical
        or scope_by_logical != tuple(sorted(scope_by_logical))
        or any(
            type(row) is not tuple
            or len(row) != 2
            or any(
                type(value) is not str or not value or value != value.strip()
                for value in row
            )
            for row in scope_by_logical
        )
        or len({logical for logical, _scope in scope_by_logical})
        != len(scope_by_logical)
        or {logical for logical, _scope in scope_by_logical}
        != {row.logical_owner for row in occurrences}
    ):
        raise MacroDagDerivationError(
            "aggregate alignment requires one canonical exact logical scope cover"
        )

    scope_map = dict(scope_by_logical)
    by_scope = {
        scope: tuple(
            row
            for row in occurrences
            if scope_map[row.logical_owner] == scope
        )
        for scope in sorted(set(scope_map.values()))
    }
    envelopes = {
        scope: (
            min(row.physical.envelope_nm[0] for row in rows),
            min(row.physical.envelope_nm[1] for row in rows),
            max(row.physical.envelope_nm[2] for row in rows),
            max(row.physical.envelope_nm[3] for row in rows),
        )
        for scope, rows in by_scope.items()
    }
    facts = []
    for spec in specs:
        if spec.first_scope not in by_scope or spec.second_scope not in by_scope:
            raise MacroDagDerivationError(
                "aggregate alignment spec names an unknown scope"
            )
        lower = 0 if spec.axis == "horizontal" else 1
        upper = 2 if spec.axis == "horizontal" else 3
        full_layout_span = max(
            row.physical.envelope_nm[upper] for row in occurrences
        ) - min(row.physical.envelope_nm[lower] for row in occurrences)
        first_box = envelopes[spec.first_scope]
        second_box = envelopes[spec.second_scope]
        if spec.edge == "leading":
            first_edge = first_box[lower]
            second_edge = second_box[lower]
            scale = 1
        elif spec.edge == "trailing":
            first_edge = first_box[upper]
            second_edge = second_box[upper]
            scale = 1
        else:
            first_edge = first_box[lower] + first_box[upper]
            second_edge = second_box[lower] + second_box[upper]
            scale = 2
        if (
            abs(first_edge - second_edge) * 1_000_000
            > scale * full_layout_span * spec.tolerance_ppm
        ):
            raise MacroDagDerivationError(
                "requested aggregate alignment exceeds reference tolerance"
            )

        first_rows = by_scope[spec.first_scope]
        second_rows = by_scope[spec.second_scope]
        facts.append(
            HardInvariant(
                fact_id=(
                    f"aggregate_alignment:{spec.axis}:{spec.edge}:"
                    f"{spec.first_scope}:{spec.second_scope}"
                ),
                kind=SpatialFactKind.AGGREGATE_ALIGNMENT,
                bindings=tuple(
                    SubjectBinding(
                        role=f"first_member_{index:03d}",
                        node=row.node_id,
                    )
                    for index, row in enumerate(
                        sorted(first_rows, key=lambda row: row.node_id)
                    )
                )
                + tuple(
                    SubjectBinding(
                        role=f"second_member_{index:03d}",
                        node=row.node_id,
                    )
                    for index, row in enumerate(
                        sorted(second_rows, key=lambda row: row.node_id)
                    )
                ),
                attributes=(
                    ("axis", spec.axis),
                    ("edge", spec.edge),
                    ("first_scope", spec.first_scope),
                    ("second_scope", spec.second_scope),
                    ("tolerance_ppm", str(spec.tolerance_ppm)),
                ),
                evidence_ids=_evidence((*first_rows, *second_rows)),
                derivation_rule="semantic-scope-envelope-alignment/v1",
            )
        )
    return tuple(facts)


def _metrics(
    occurrences: tuple[SemanticOccurrence, ...],
) -> tuple[tuple[MetricObservation, ...], tuple[DiagnosticMetric, ...]]:
    x0 = min(row.physical.envelope_nm[0] for row in occurrences)
    y0 = min(row.physical.envelope_nm[1] for row in occurrences)
    x1 = max(row.physical.envelope_nm[2] for row in occurrences)
    y1 = max(row.physical.envelope_nm[3] for row in occurrences)
    width = x1 - x0
    height = y1 - y0
    occupied = sum(
        (row.physical.envelope_nm[2] - row.physical.envelope_nm[0])
        * (row.physical.envelope_nm[3] - row.physical.envelope_nm[1])
        for row in occurrences
    )
    envelope_area = width * height
    observations = (
        MetricObservation(
            metric_id="envelope.aspect_ratio",
            value=Decimal(width) / Decimal(height),
            unit="ratio",
        ),
    )
    diagnostics = (
        DiagnosticMetric(
            metric_id="occurrence.fill_ratio",
            value=Decimal(occupied) / Decimal(envelope_area),
            unit="ratio",
        ),
    )
    return observations, diagnostics


def derive_abstract_spatial_ir(
    overlay: SemanticOverlay,
    *,
    scope_by_logical: tuple[tuple[str, str], ...] = (),
    mosaic_scope_by_logical: tuple[tuple[str, str], ...] = (),
    aggregate_alignment_specs: tuple[AggregateAlignmentSpec, ...] = (),
    macro_level_specs: tuple[MacroDagLevelSpec, ...] = (),
    macro_alignment_tolerance_ppm: int = 5_000,
) -> AbstractSpatialIR:
    """Derive only translation-invariant topology and typed structural facts."""

    if type(overlay) is not SemanticOverlay:
        raise TypeError("ASIR derivation requires an exact SemanticOverlay")
    from clay.layout.spatial.scoped_mosaic_ranks import derive_mosaic_scope_ranks
    nodes = overlay.node_ids
    observations, diagnostics = _metrics(overlay.occurrences)
    return AbstractSpatialIR(
        source_clm_sha256=overlay.sha256,
        horizontal=OrthogonalDag(
            nodes=nodes,
            edges=_separation_edges(overlay.occurrences, axis="horizontal"),
        ),
        vertical=OrthogonalDag(
            nodes=nodes,
            edges=_separation_edges(overlay.occurrences, axis="vertical"),
        ),
        hard_facts=(
            *_partition_facts(overlay.occurrences),
            *_lattice_facts(overlay.occurrences),
            *_alignment_facts(overlay.occurrences),
            *_class_rank_facts(overlay.occurrences),
            *derive_mosaic_scope_ranks(overlay.occurrences, mosaic_scope_by_logical),
            *_aggregate_order_facts(
                overlay.occurrences,
                scope_by_logical,
            ),
            *_aggregate_alignment_facts(
                overlay.occurrences,
                scope_by_logical,
                aggregate_alignment_specs,
            ),
        ),
        observations=observations,
        diagnostics=diagnostics,
        macro_hierarchy=_derive_macro_hierarchy(
            overlay.occurrences,
            macro_level_specs,
            alignment_tolerance_ppm=macro_alignment_tolerance_ppm,
        ),
    )


__all__ = [
    "AggregateAlignmentSpec",
    "MacroDagDerivationError",
    "MacroDagLevelSpec",
    "MacroDagNodeSpec",
    "derive_abstract_spatial_ir",
]
