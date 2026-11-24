"""Installed-core contract for declaration-driven compact-layout candidates.

The optimizer may construct and rank candidates from ``tools/``, but admission
to the canonical physical flow must remain executable from the installed
``clay`` wheel.  This module owns the PDK-independent typed domain contract and
candidate validation used on both sides of that boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from functools import reduce
from operator import mul
from typing import Mapping, Sequence

from clay.layout.compact_intent import (
    CompactIntentCandidate,
    GuardRingAtom,
    GuardRingChoice,
    MacroPartitionChoice,
    derive_guard_ring_atoms,
    enumerate_guard_ring_partitions,
    replay_compact_intent_candidate,
    guard_atom_compatibility,
)
from clay.layout.annotations import Matching
from clay.layout.physical_macro_graph import PhysicalMacro, PhysicalMacroGraph


_ORIENTATION_ORDER = ("R0", "R90", "R180", "R270")
_SIDE_ORDER = ("center", "left", "right")
_PACK_ORDER = ("dense_2d", "rows")


@dataclass(frozen=True)
class CompactIntentDomainSpec:
    seed: int
    budget: int
    refinement_budget: int
    split_passives: tuple[str, ...]
    orientation_only_passives: tuple[str, ...]
    split_counts: tuple[int, ...]
    orientations: tuple[str, ...]
    passive_sides: tuple[str, ...]
    pack_modes: tuple[str, ...]
    left_column_groups: tuple[str, ...]
    aspect_bounds: tuple[float, float]
    aspect_nominal: float
    aspect_points: tuple[float, ...]
    fold_height_bounds_um: tuple[float, float]
    fold_height_nominal_um: float
    fold_height_points_um: tuple[float, ...]
    block_gap_bounds_um: tuple[float, float]
    block_gap_nominal_um: float
    block_gap_points_um: tuple[float, ...]
    edge_rail_width_bounds_um: tuple[float, float]
    edge_rail_width_nominal_um: float
    edge_rail_width_points_um: tuple[float, ...]


class GraphPartitionError(ValueError):
    """A named, fail-closed rejection from the typed macro partition domain."""

    def __init__(self, code: str, macro_id: str, detail: str) -> None:
        self.code = code
        self.macro_id = macro_id
        self.detail = detail
        super().__init__(f"{code}: {macro_id}: {detail}")


@dataclass(frozen=True, order=True)
class GraphPartitionRejection:
    code: str
    macro_id: str
    detail: str


@dataclass(frozen=True)
class GraphPartitionDomain:
    """Deterministic graph-derived macro partitions and truncation evidence."""

    passive_bank_axis: tuple[tuple[MacroPartitionChoice, ...], ...]
    well_island_axis: tuple[tuple[MacroPartitionChoice, ...], ...]
    rejections: tuple[GraphPartitionRejection, ...]
    exhausted: bool


def matched_passive_orientation_components(
    cell: object,
    passives: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """Group orientation axes that are joined by declared matching intent."""

    members = tuple(sorted(set(str(item) for item in passives)))
    parent = {member: member for member in members}

    def find(member: str) -> str:
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    annotations = getattr(cell, "_layout_annotations", {}) or {}
    for member in members:
        for annotation in annotations.get(member, ()):
            if isinstance(annotation, Matching):
                other = str(annotation.other)
                if other in parent:
                    union(member, other)

    groups: dict[str, list[str]] = {}
    for member in members:
        groups.setdefault(find(member), []).append(member)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def compact_structure_fingerprint(candidate: CompactIntentCandidate) -> str:
    """Hash only discrete physical structure, excluding continuous controls.

    Aspect, block gap, and rail width can tune one fixed topology; they must not
    make a search cohort look structurally diverse.  Fold height remains bound
    because it controls discrete MOS finger/row decomposition.  The retained
    fields also bind macro ownership/partitioning, passive decomposition,
    well/ring islands, ordering, orientation, side, and packing class.
    """

    if type(candidate) is not CompactIntentCandidate:
        raise TypeError(
            "compact structure fingerprint requires CompactIntentCandidate"
        )
    candidate_data = dict(candidate.canonical_data())
    for field in (
        "aspect_mode",
        "block_gap_um",
        "edge_rail_width_um",
    ):
        candidate_data.pop(field, None)
    payload = {
        "schema": "clay.compact-structure-fingerprint.v1",
        "structure": candidate_data,
    }
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _macro_by_id(graph: PhysicalMacroGraph, macro_id: str) -> PhysicalMacro:
    for macro in graph.macros:
        if macro.macro_id == macro_id:
            return macro
    raise GraphPartitionError(
        "PARTITION_MEMBER_LOSS", macro_id, "macro is absent from the bound graph"
    )


def _matching_components(
    cell: object, members: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    member_set = set(members)
    parent = {member: member for member in members}

    def find(member: str) -> str:
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    annotations = getattr(cell, "_layout_annotations", {}) or {}
    for target in members:
        for annotation in annotations.get(target, ()):
            if not isinstance(annotation, Matching):
                continue
            other = str(annotation.other)
            if other not in member_set:
                raise GraphPartitionError(
                    "MATCHING_COMPONENT_SPLIT",
                    "",
                    f"matching edge {target!r}<->{other!r} crosses macro ownership",
                )
            union(target, other)

    groups: dict[str, list[str]] = {}
    for member in members:
        groups.setdefault(find(member), []).append(member)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def _centroid_components(
    graph: PhysicalMacroGraph, macro: PhysicalMacro
) -> tuple[tuple[str, ...], ...]:
    owned = set(macro.members)
    components: list[tuple[str, ...]] = []
    for other in graph.macros:
        if other.kind != "matched_array":
            continue
        refs = set(other.declaration_refs)
        attrs = dict(other.attributes)
        if not (
            any(ref.startswith("@common_centroid(") for ref in refs)
            or attrs.get("common_centroid") == "true"
        ):
            continue
        overlap = owned & set(other.members)
        if overlap and overlap != set(other.members):
            raise GraphPartitionError(
                "CENTROID_AXIS_INFEASIBLE",
                macro.macro_id,
                f"common-centroid macro {other.macro_id!r} crosses ownership",
            )
        if overlap:
            components.append(tuple(sorted(overlap)))
    return tuple(sorted(set(components)))


def _partition_atoms(
    graph: PhysicalMacroGraph, cell: object, macro: PhysicalMacro
) -> tuple[tuple[str, ...], ...]:
    matching = _matching_components(cell, macro.members)
    centroid = _centroid_components(graph, macro)
    parent = {member: member for member in macro.members}

    def find(member: str) -> str:
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    def union_group(group: tuple[str, ...]) -> None:
        root = find(group[0])
        for member in group[1:]:
            other = find(member)
            if root != other:
                low, high = sorted((root, other))
                parent[high] = low
                root = low

    for component in (*matching, *centroid):
        union_group(component)
    groups: dict[str, list[str]] = {}
    for member in macro.members:
        groups.setdefault(find(member), []).append(member)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def validate_graph_partition_choice(
    graph: PhysicalMacroGraph,
    cell: object,
    choice: MacroPartitionChoice,
    *,
    expected_kind: str,
) -> None:
    """Validate one partition against graph ownership and analog structure."""

    macro = _macro_by_id(graph, choice.macro_id)
    if macro.kind != expected_kind:
        raise GraphPartitionError(
            "PARTITION_MEMBER_LOSS",
            choice.macro_id,
            f"expected {expected_kind!r}, found {macro.kind!r}",
        )
    flattened = tuple(member for group in choice.member_groups for member in group)
    if set(flattened) != set(macro.members) or len(flattened) != len(macro.members):
        raise GraphPartitionError(
            "PARTITION_MEMBER_LOSS",
            choice.macro_id,
            "partition must cover every owned member exactly once",
        )

    group_sets = tuple(set(group) for group in choice.member_groups)
    try:
        matching_components = _matching_components(cell, macro.members)
    except GraphPartitionError as exc:
        raise GraphPartitionError(exc.code, choice.macro_id, exc.detail) from exc
    for component in matching_components:
        if not any(set(component) <= group for group in group_sets):
            raise GraphPartitionError(
                "MATCHING_COMPONENT_SPLIT",
                choice.macro_id,
                f"matching component {component!r} is split across banks",
            )
    for component in _centroid_components(graph, macro):
        if not any(set(component) <= group for group in group_sets):
            raise GraphPartitionError(
                "CENTROID_AXIS_INFEASIBLE",
                choice.macro_id,
                f"common-centroid component {component!r} is split across banks",
            )


def _balanced_partition(
    atoms: tuple[tuple[str, ...], ...], group_count: int
) -> tuple[tuple[str, ...], ...]:
    base, remainder = divmod(len(atoms), group_count)
    groups: list[tuple[str, ...]] = []
    cursor = 0
    for index in range(group_count):
        width = base + (1 if index < remainder else 0)
        selected = atoms[cursor : cursor + width]
        groups.append(tuple(sorted(member for atom in selected for member in atom)))
        cursor += width
    return tuple(sorted(groups))


def _macro_partition_variants(
    graph: PhysicalMacroGraph, cell: object, macro: PhysicalMacro
) -> tuple[MacroPartitionChoice, ...]:
    atoms = _partition_atoms(graph, cell, macro)
    variants = {
        MacroPartitionChoice(
            macro_id=macro.macro_id,
            member_groups=_balanced_partition(atoms, group_count),
            declaration_refs=macro.declaration_refs,
        )
        for group_count in range(1, len(atoms) + 1)
    }
    return tuple(
        sorted(variants, key=lambda choice: (len(choice.member_groups), choice))
    )


def _decode_product_index(
    index: int, axes: Sequence[Sequence[object]]
) -> tuple[object, ...]:
    values: list[object] = [None] * len(axes)
    for position in range(len(axes) - 1, -1, -1):
        axis = axes[position]
        values[position] = axis[index % len(axis)]
        index //= len(axis)
    return tuple(values)


def _bounded_partition_axis(
    graph: PhysicalMacroGraph,
    cell: object,
    *,
    kind: str,
    budget: int,
    seed: int,
) -> tuple[tuple[tuple[MacroPartitionChoice, ...], ...], bool]:
    macros = tuple(macro for macro in graph.macros if macro.kind == kind)
    if not macros:
        return (((),), True)
    axes = tuple(_macro_partition_variants(graph, cell, macro) for macro in macros)
    total = reduce(mul, (len(axis) for axis in axes), 1)
    if total <= budget:
        indices = tuple(range(total))
    else:
        chosen = {0, total - 1}
        center_vector = tuple(len(axis) // 2 for axis in axes)
        center_index = 0
        for position, axis in zip(center_vector, axes):
            center_index = center_index * len(axis) + position
        chosen.add(center_index)
        nonce = 0
        while len(chosen) < budget:
            digest = hashlib.sha256(f"{seed}:{kind}:{nonce}".encode("ascii")).digest()
            chosen.add(int.from_bytes(digest[:8], "big") % total)
            nonce += 1
        anchors = tuple(sorted(chosen))
        if len(anchors) > budget:
            mandatory = {0, total - 1}
            retained = sorted(mandatory)
            retained.extend(index for index in anchors if index not in mandatory)
            indices = tuple(sorted(retained[:budget]))
        else:
            indices = anchors
    return (
        tuple(
            tuple(_decode_product_index(index, axes))  # type: ignore[arg-type]
            for index in indices
        ),
        total <= budget,
    )


def graph_partition_axes(
    graph: PhysicalMacroGraph,
    cell: object,
    *,
    budget: int,
    seed: int,
) -> GraphPartitionDomain:
    """Derive bounded passive/well partitions from the canonical macro graph."""

    if budget <= 0:
        raise ValueError("graph partition budget must be positive")
    passive_axis, passive_exhausted = _bounded_partition_axis(
        graph, cell, kind="passive_bank", budget=budget, seed=seed
    )
    well_axis, well_exhausted = _bounded_partition_axis(
        graph, cell, kind="well_island", budget=budget, seed=seed
    )
    rejections: list[GraphPartitionRejection] = []
    if not passive_exhausted:
        rejections.append(
            GraphPartitionRejection(
                code="PARTITION_BUDGET_EXHAUSTED",
                macro_id="passive_bank",
                detail=f"retained {len(passive_axis)} graph partitions within budget {budget}",
            )
        )
    if not well_exhausted:
        rejections.append(
            GraphPartitionRejection(
                code="PARTITION_BUDGET_EXHAUSTED",
                macro_id="well_island",
                detail=f"retained {len(well_axis)} graph partitions within budget {budget}",
            )
        )
    return GraphPartitionDomain(
        passive_bank_axis=passive_axis,
        well_island_axis=well_axis,
        rejections=tuple(rejections),
        exhausted=passive_exhausted and well_exhausted,
    )


def physical_macro_graph_for_cell(cell: object) -> PhysicalMacroGraph:
    """Lower one parsed cell to the canonical graph used by search/validation."""

    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.physical_macro_graph import physical_macro_graph_from_intent

    return physical_macro_graph_from_intent(  # type: ignore[arg-type]
        cell,
        layout_intent_from_cell_def(cell),  # type: ignore[arg-type]
        device_records={},
    )


def _partition_declarations_for_choices(
    choices: Sequence[MacroPartitionChoice],
) -> tuple[object, ...]:
    """Lower graph partition choices to canonical source-level bank declarations.

    A declaration-backed macro has ``passive_bank:<parent>:<bank>`` identity.
    Splitting it creates deterministic sub-bank names; keeping it whole preserves
    the authored name.  A legacy unsplit ``passive_bank:<parent>`` macro needs no
    declaration, while splitting it introduces the two-or-more declarations that
    make the partition executable by the canonical graph compiler.
    """

    from clay.layout.intent import PassiveBankPartitionIntent

    declarations: list[PassiveBankPartitionIntent] = []
    for choice in sorted(choices):
        parts = choice.macro_id.split(":")
        if len(parts) not in (2, 3) or parts[0] != "passive_bank":
            raise GraphPartitionError(
                "PARTITION_MEMBER_LOSS",
                choice.macro_id,
                "passive-bank macro identity cannot be lowered to Clay",
            )
        parent = parts[1]
        authored_bank = parts[2] if len(parts) == 3 else None
        if authored_bank is None and len(choice.member_groups) == 1:
            continue
        if len(choice.member_groups) == 1:
            bank_names = (authored_bank,)
        else:
            tag = hashlib.sha256(choice.macro_id.encode("utf-8")).hexdigest()[:12]
            bank_names = tuple(
                f"compact_{tag}_{index + 1:02d}"
                for index in range(len(choice.member_groups))
            )
        declarations.extend(
            PassiveBankPartitionIntent(
                parent=parent,
                bank=str(bank),
                members=members,
            )
            for bank, members in zip(bank_names, choice.member_groups)
        )
    return tuple(sorted(declarations, key=lambda item: (item.parent, item.bank)))


def passive_bank_partition_intents_for_candidate(
    cell: object,
    candidate: CompactIntentCandidate,
) -> tuple[object, ...]:
    """Return and validate the exact passive declarations selected by a candidate.

    The first replay sees authoring macro ids and validates choices against that
    graph.  Later replays see the already-derived graph, so they require exact
    declaration and member-group equality instead of trying to rediscover the
    removed authoring macro ids.
    """

    from clay.layout.intent import layout_intent_from_cell_def

    intent = layout_intent_from_cell_def(cell)  # type: ignore[arg-type]
    choices = tuple(candidate.passive_bank_partitions)
    if not choices:
        return tuple(intent.passive_bank_partitions)
    source_graph = physical_macro_graph_for_cell(cell)
    source_macros = {
        macro.macro_id: macro
        for macro in source_graph.macros
        if macro.kind == "passive_bank"
    }
    choice_ids = {choice.macro_id for choice in choices}
    declarations = _partition_declarations_for_choices(choices)
    if choice_ids == set(source_macros):
        for choice in choices:
            validate_graph_partition_choice(
                source_graph,
                cell,
                choice,
                expected_kind="passive_bank",
            )
        return declarations

    if tuple(intent.passive_bank_partitions) != declarations:
        raise GraphPartitionError(
            "PARTITION_MEMBER_LOSS",
            "passive_bank",
            "materialized declarations do not equal the selected partition",
        )
    observed_groups = tuple(sorted(macro.members for macro in source_macros.values()))
    selected_groups = tuple(
        sorted(group for choice in choices for group in choice.member_groups)
    )
    if observed_groups != selected_groups:
        raise GraphPartitionError(
            "PARTITION_MEMBER_LOSS",
            "passive_bank",
            "materialized macro membership does not equal the selected partition",
        )
    return declarations


def physical_macro_graph_for_candidate(
    cell: object,
    candidate: CompactIntentCandidate,
    *,
    allow_materialized_passive_arrays: bool = False,
) -> PhysicalMacroGraph:
    """Build the effective graph after selected structural intent is applied.

    ``allow_materialized_passive_arrays`` is the admission-replay mode used by
    an already materialized derivative.  In that mode every selected structural
    axis that is serialized into ordinary Clay syntax must replay exactly rather
    than be revalidated against authoring-only domains that the derivative has
    intentionally replaced (notably global floorplan rows versus internal macro
    ``pack_rows``).
    """

    if type(allow_materialized_passive_arrays) is not bool:
        raise TypeError("materialized passive-array replay flag must be an exact bool")
    candidate = replay_compact_intent_candidate(candidate)

    from clay.layout.intent import (
        ArrayIntent,
        GuardRingIntent,
        layout_intent_from_cell_def,
    )
    from clay.layout.physical_macro_graph import physical_macro_graph_from_intent

    intent = layout_intent_from_cell_def(cell)  # type: ignore[arg-type]
    if candidate.passive_bank_partitions:
        intent = replace(
            intent,
            passive_bank_partitions=passive_bank_partition_intents_for_candidate(
                cell, candidate
            ),
        )
    if candidate.guard_ring_groups:
        intent = replace(
            intent,
            guard_rings=tuple(
                GuardRingIntent(
                    members=ring.members,
                    well=ring.well,
                    tie_net=ring.tie_net,
                )
                for ring in candidate.guard_ring_groups
            ),
        )
    if candidate.passive_arrays:
        selected = {choice.passive: choice for choice in candidate.passive_arrays}
        authored: dict[str, list[ArrayIntent]] = {}
        for array in intent.arrays:
            authored.setdefault(array.group, []).append(array)
        generated: list[ArrayIntent] = []
        for target, choice in selected.items():
            expected = ArrayIntent(
                group=choice.passive,
                style="discrete",
                rows=choice.rows,
                order="ratio",
            )
            existing = authored.get(target, [])
            if not existing:
                generated.append(expected)
                continue
            if not allow_materialized_passive_arrays:
                raise ValueError(
                    "passive array candidate collides with authored array target "
                    f"{target!r}"
                )
            if len(existing) != 1:
                raise ValueError(
                    f"materialized passive array target {target!r} is not unique"
                )
            if existing[0] != expected:
                raise ValueError(
                    f"materialized passive array target {target!r} differs from candidate"
                )
        intent = replace(
            intent,
            arrays=tuple(intent.arrays) + tuple(generated),
        )
    if candidate.common_centroid_rows:
        selected_rows = dict(candidate.common_centroid_rows)
        declared_rows = {
            item.group: item.rows for item in intent.common_centroids
        }
        intent = replace(
            intent,
            common_centroids=tuple(
                replace(item, rows=selected_rows.get(item.group, item.rows))
                for item in intent.common_centroids
            ),
            dummy_flanks=tuple(
                replace(
                    item,
                    per_end=(
                        int(declared_rows[item.target]) * int(item.per_end)
                        // int(selected_rows[item.target])
                    ),
                )
                if item.target in selected_rows else item
                for item in intent.dummy_flanks
            ),
        )
    graph = physical_macro_graph_from_intent(  # type: ignore[arg-type]
        cell,
        intent,
        device_records={},
    )
    if not candidate.nonmos_macro_rows:
        return graph

    from clay.layout.physical_macro_graph import PhysicalMacroGraph

    by_id = {macro.macro_id: macro for macro in graph.macros}
    rebound = dict(by_id)
    for choice in candidate.nonmos_macro_rows:
        macro = by_id.get(choice.macro_id)
        if macro is None or macro.kind not in {"passive_bank", "bjt_cluster"}:
            raise ValueError(
                "non-MOS macro row choice must name one aggregate passive/BJT "
                f"macro: {choice.macro_id!r}"
            )
        attributes = dict(macro.attributes)
        existing = attributes.get("pack_rows")
        if allow_materialized_passive_arrays:
            if existing != str(choice.rows):
                raise ValueError(
                    "materialized non-MOS macro rows differ from candidate: "
                    f"{choice.macro_id}"
                )
            rebound[macro.macro_id] = macro
            continue
        containers = [
            group
            for group in intent.place_groups
            if set(macro.members) <= set(group.devices)
            and group.row_min is not None
            and group.row_max is not None
        ]
        if not containers:
            raise ValueError(
                f"non-MOS macro row choice lacks a declared row domain: "
                f"{choice.macro_id}"
            )
        narrowest_size = min(len(group.devices) for group in containers)
        narrowest = [
            group for group in containers if len(group.devices) == narrowest_size
        ]
        bounds = {
            (group.row_min, group.row_max) for group in narrowest
        }
        if len(bounds) != 1:
            raise ValueError(
                f"non-MOS macro row choice has ambiguous row domains: "
                f"{choice.macro_id}"
            )
        row_min, row_max = next(iter(bounds))
        if (
            type(row_min) is not int
            or type(row_max) is not int
            or not row_min <= choice.rows <= row_max
        ):
            raise ValueError(
                f"non-MOS macro row choice escapes declared row domain: "
                f"{choice.macro_id}"
            )
        if existing is not None and existing != str(choice.rows):
            raise ValueError(
                f"materialized non-MOS macro rows differ from candidate: "
                f"{choice.macro_id}"
            )
        attributes["pack_rows"] = str(choice.rows)
        rebound[macro.macro_id] = replace(
            macro, attributes=tuple(attributes.items())
        )
    return PhysicalMacroGraph(
        cell=graph.cell,
        macros=tuple(rebound.values()),
        relations=graph.relations,
        devices=graph.devices,
    )


def _number(data: Mapping[str, object], key: str) -> float:
    raw = data.get(key)
    if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError(f"compact layout intent requires finite numeric {key}")
    return float(raw)


def _positive_int(data: Mapping[str, object], key: str) -> int:
    raw = _number(data, key)
    integer = int(round(raw))
    if integer <= 0 or not math.isclose(raw, integer, abs_tol=1e-12):
        raise ValueError(f"compact layout intent {key} must be a positive integer")
    return integer


def _nonnegative_int(data: Mapping[str, object], key: str, *, default: int = 0) -> int:
    if key not in data:
        return default
    raw = _number(data, key)
    integer = int(round(raw))
    if integer < 0 or not math.isclose(raw, integer, abs_tol=1e-12):
        raise ValueError(f"compact layout intent {key} must be a non-negative integer")
    return integer


def _enabled_suffixes(
    data: Mapping[str, object],
    prefix: str,
    order: Sequence[str] | None = None,
) -> tuple[str, ...]:
    enabled = {
        str(key)[len(prefix) :]
        for key, value in data.items()
        if str(key).startswith(prefix)
        and isinstance(value, (int, float))
        and float(value) != 0.0
    }
    if order is None:
        return tuple(sorted(enabled))
    return tuple(item for item in order if item in enabled)


def _bounds(
    data: Mapping[str, object],
    low_key: str,
    high_key: str,
    *,
    scale: float = 1.0,
) -> tuple[float, float]:
    low = _number(data, low_key) * scale
    high = _number(data, high_key) * scale
    if low <= 0.0 or high < low:
        raise ValueError(
            f"invalid compact layout bounds {low_key}={low}, {high_key}={high}"
        )
    return (low, high)


def _power_counts(low: int, high: int) -> tuple[int, ...]:
    if low <= 0 or high < low:
        raise ValueError("passive split bounds must be positive and ordered")
    values = {low, high}
    point = 1
    while point < low:
        point *= 2
    while point <= high:
        values.add(point)
        point *= 2
    return tuple(sorted(values))


def _three_points(bounds: tuple[float, float]) -> tuple[float, ...]:
    low, high = bounds
    if math.isclose(low, high, rel_tol=1e-15, abs_tol=1e-15):
        return (low,)
    return tuple(sorted({low, math.sqrt(low * high), high}))


def _declared_points(
    data: Mapping[str, object],
    bounds: tuple[float, float],
    nominal_key: str,
    *,
    scale: float = 1.0,
) -> tuple[float, ...]:
    values = set(_three_points(bounds))
    if nominal_key in data:
        nominal = _number(data, nominal_key) * scale
        if not bounds[0] <= nominal <= bounds[1]:
            raise ValueError(
                f"compact layout nominal {nominal_key}={nominal} is outside {bounds}"
            )
        values.add(nominal)
    return tuple(sorted(values))


def compact_intent_domain_spec(cell: object) -> CompactIntentDomainSpec:
    """Lower the existing cell ``sizing {}`` surface into typed layout axes."""

    data = dict(getattr(cell, "sizing_kwargs", {}) or {})
    seed = _positive_int(data, "opt_layout_seed")
    budget = _positive_int(data, "opt_layout_budget")
    refinement_budget = _nonnegative_int(data, "opt_layout_refinement_budget")
    split_passives = _enabled_suffixes(data, "opt_layout_split__")
    orient_only = _enabled_suffixes(data, "opt_layout_orient__")
    if set(split_passives) & set(orient_only):
        raise ValueError("a passive cannot be both split and orientation-only")
    if not split_passives and not orient_only:
        raise ValueError("compact layout intent declares no eligible passives")
    split_low = _positive_int(data, "opt_layout_split_min")
    split_high = _positive_int(data, "opt_layout_split_max")
    orientations = tuple(
        token.upper()
        for token in _enabled_suffixes(
            data,
            "opt_layout_orientation_",
            tuple(item.casefold() for item in _ORIENTATION_ORDER),
        )
    )
    sides = _enabled_suffixes(data, "opt_layout_passive_side_", _SIDE_ORDER)
    packs = _enabled_suffixes(data, "opt_layout_pack_", _PACK_ORDER)
    if not orientations or not sides or not packs:
        raise ValueError(
            "compact layout intent requires orientation, passive-side and pack choices"
        )
    aspect_bounds = _bounds(data, "opt_layout_aspect_min", "opt_layout_aspect_max")
    fold_bounds = _bounds(
        data,
        "opt_layout_fold_height_min",
        "opt_layout_fold_height_max",
        scale=1e6,
    )
    gap_bounds = _bounds(
        data,
        "opt_layout_block_gap_min",
        "opt_layout_block_gap_max",
        scale=1e6,
    )
    rail_bounds = _bounds(
        data,
        "opt_layout_edge_rail_width_min",
        "opt_layout_edge_rail_width_max",
        scale=1e6,
    )
    from clay.layout.intent import layout_intent_from_cell_def

    left_column_groups = tuple(sorted({
        item.group
        for item in layout_intent_from_cell_def(cell).common_centroids
    }))
    return CompactIntentDomainSpec(
        seed=seed,
        budget=budget,
        refinement_budget=refinement_budget,
        split_passives=split_passives,
        orientation_only_passives=orient_only,
        split_counts=_power_counts(split_low, split_high),
        orientations=orientations,
        passive_sides=sides,
        pack_modes=packs,
        left_column_groups=left_column_groups,
        aspect_bounds=aspect_bounds,
        aspect_nominal=_number(data, "opt_layout_aspect_nominal"),
        aspect_points=_declared_points(
            data, aspect_bounds, "opt_layout_aspect_nominal"
        ),
        fold_height_bounds_um=fold_bounds,
        fold_height_nominal_um=_number(data, "opt_layout_fold_height_nominal") * 1e6,
        fold_height_points_um=_declared_points(
            data,
            fold_bounds,
            "opt_layout_fold_height_nominal",
            scale=1e6,
        ),
        block_gap_bounds_um=gap_bounds,
        block_gap_nominal_um=_number(data, "opt_layout_block_gap_nominal") * 1e6,
        block_gap_points_um=_declared_points(
            data,
            gap_bounds,
            "opt_layout_block_gap_nominal",
            scale=1e6,
        ),
        edge_rail_width_bounds_um=rail_bounds,
        edge_rail_width_nominal_um=_number(data, "opt_layout_edge_rail_width_nominal")
        * 1e6,
        edge_rail_width_points_um=_declared_points(
            data,
            rail_bounds,
            "opt_layout_edge_rail_width_nominal",
            scale=1e6,
        ),
    )


def guard_ring_choice(
    groups: Sequence[Sequence[GuardRingAtom]],
) -> tuple[GuardRingChoice, ...]:
    """Lower one complete atom partition into canonical ring DATA."""

    rings = []
    for group in groups:
        first = group[0]
        rings.append(
            GuardRingChoice(
                members=tuple(
                    sorted(member for atom in group for member in atom.members)
                ),
                well=first.well,
                tie_net=first.tie_net,
            )
        )
    return tuple(sorted(rings))


def _ring_groups_compatible(
    left: Sequence[GuardRingAtom],
    right: Sequence[GuardRingAtom],
) -> bool:
    return all(
        guard_atom_compatibility(left_atom, right_atom).compatible
        for left_atom in left
        for right_atom in right
    )


def passive_isolation_ring_anchor(
    atoms: Sequence[GuardRingAtom],
) -> tuple[GuardRingChoice, ...] | None:
    """Return the deterministic closure of compatible passive isolation banks."""

    ordered = tuple(sorted(atoms))
    clusters: dict[tuple[str, str], list[GuardRingAtom]] = {}
    for atom in ordered:
        if atom.isolated and atom.passive_only:
            clusters.setdefault((atom.well, atom.tie_net), []).append(atom)
    compatible_groups: list[tuple[GuardRingAtom, ...]] = []
    for key in sorted(clusters):
        groups: list[list[GuardRingAtom]] = []
        for atom in clusters[key]:
            destination = next(
                (
                    group
                    for group in groups
                    if all(
                        guard_atom_compatibility(atom, member).compatible
                        for member in group
                    )
                ),
                None,
            )
            if destination is None:
                groups.append([atom])
            else:
                destination.append(atom)
        compatible_groups.extend(
            tuple(group) for group in groups if len(group) > 1
        )
    merged_atoms = {atom for group in compatible_groups for atom in group}
    if not merged_atoms:
        return None
    groups = [tuple(sorted(group)) for group in compatible_groups]
    groups.extend((atom,) for atom in ordered if atom not in merged_atoms)
    return guard_ring_choice(tuple(sorted(groups)))


def bounded_guard_ring_axis(
    atoms: Sequence[GuardRingAtom],
    *,
    budget: int,
    seed: int,
) -> tuple[tuple[tuple[GuardRingChoice, ...], ...], bool]:
    """Build the one deterministic budgeted guard-ring axis used by Clay."""

    ordered = tuple(sorted(atoms))
    if not ordered:
        return (((),), True)
    if len(ordered) <= 8:
        exact = enumerate_guard_ring_partitions(ordered) or ((),)
        return (exact, True)

    choices: set[tuple[GuardRingChoice, ...]] = set()
    passive_anchor = passive_isolation_ring_anchor(ordered)
    if passive_anchor is not None:
        choices.add(passive_anchor)
    round_index = 0
    max_rounds = max(4, budget * 4)
    while len(choices) < budget and round_index < max_rounds:
        groups: list[tuple[GuardRingAtom, ...]] = [(atom,) for atom in ordered]
        choices.add(guard_ring_choice(groups))
        step = 0
        while len(choices) < budget:
            legal = [
                (left, right)
                for left in range(len(groups))
                for right in range(left + 1, len(groups))
                if _ring_groups_compatible(groups[left], groups[right])
            ]
            if not legal:
                break
            digest = hashlib.sha256(
                f"{seed}:guard-rings:{round_index}:{step}".encode("ascii")
            ).digest()
            left, right = legal[int.from_bytes(digest[:8], "big") % len(legal)]
            merged = tuple(sorted(groups[left] + groups[right]))
            groups = [
                group
                for index, group in enumerate(groups)
                if index not in {left, right}
            ]
            groups.append(merged)
            groups.sort()
            choices.add(guard_ring_choice(groups))
            step += 1
        round_index += 1
    return (tuple(sorted(choices)), False)


def _validate_guard_ring_partition(
    cell: object,
    candidate: CompactIntentCandidate,
) -> None:
    atoms = tuple(derive_guard_ring_atoms(cell))
    # The bounded axis is an evaluation sample, not the semantic intent domain.
    # Post-signoff refinement may reach a neighbouring legal partition that the
    # seeded base sample did not contain.  Legality is therefore established by
    # complete atom coverage plus the declared compatibility/body-net rules
    # below; ``opt_layout_budget`` limits work, never meaning.
    member_to_atom: dict[str, GuardRingAtom] = {}
    for atom in atoms:
        for member in atom.members:
            member_to_atom[member] = atom

    covered: set[GuardRingAtom] = set()
    for ring in candidate.guard_ring_groups:
        try:
            ring_atoms = {member_to_atom[member] for member in ring.members}
        except KeyError as exc:
            raise ValueError(
                "compact candidate guard-ring partition is outside the declared domain"
            ) from exc
        materialized_members = {
            member for atom in ring_atoms for member in atom.members
        }
        if materialized_members != set(ring.members):
            raise ValueError(
                "compact candidate guard ring splits a declared structural atom"
            )
        if any(atom in covered for atom in ring_atoms):
            raise ValueError("compact candidate guard-ring atoms overlap")
        if any(
            atom.well != ring.well or atom.tie_net != ring.tie_net
            for atom in ring_atoms
        ):
            raise ValueError(
                "compact candidate guard ring violates well/body-net intent"
            )
        ordered = tuple(sorted(ring_atoms))
        if any(
            not guard_atom_compatibility(left, right).compatible
            for left_index, left in enumerate(ordered)
            for right in ordered[left_index + 1 :]
        ):
            raise ValueError(
                "compact candidate guard-ring merge is outside the declared domain"
            )
        covered.update(ring_atoms)
    if covered != set(atoms):
        raise ValueError(
            "compact candidate guard-ring partition does not cover declared atoms"
        )


def common_centroid_row_materialization_error(
    declared_rows: object,
    selected_rows: object,
    dummy: object,
) -> str | None:
    """Return the typed blocker for one row-count/dummy materialization.

    A drain-exposed shared-diffusion dummy has a source-valued inner boundary.
    Consequently it can contribute exactly one physical finger at each row end.
    Folding two or more fingers into that continuous active row turns every inner
    dummy finger into an all-source parasitic MOS: arithmetic multiplicity remains
    conserved, but full GDS extraction gains an extra device.
    """

    if (
        type(declared_rows) is not int
        or declared_rows <= 0
        or type(selected_rows) is not int
        or selected_rows <= 0
    ):
        return "common-centroid row authority must contain exact positive integers"
    if dummy is None:
        return None
    per_end = getattr(dummy, "per_end", None)
    tie = getattr(dummy, "tie", None)
    share_diffusion = getattr(dummy, "share_diffusion", None)
    if (
        type(per_end) is not int
        or per_end <= 0
        or type(tie) is not str
        or type(share_diffusion) is not bool
    ):
        return "common-centroid dummy authority is malformed"
    budget = declared_rows * per_end
    if budget % selected_rows:
        return "common-centroid rows cannot conserve the declared dummy budget"
    selected_per_end = budget // selected_rows
    if tie == "gate_off" and share_diffusion and selected_per_end != 1:
        return (
            "shared-diffusion gate-off dummy requires exactly one finger per end "
            "per row"
        )
    return None


def validate_compact_intent_candidate(
    cell: object,
    candidate: CompactIntentCandidate,
    *,
    allow_materialized_passive_arrays: bool = False,
) -> None:
    """Fail when a candidate is not expressible by the cell's declared domain."""

    if type(allow_materialized_passive_arrays) is not bool:
        raise TypeError("materialized passive-array replay flag must be an exact bool")
    candidate = replay_compact_intent_candidate(candidate)

    # The unmodified authoring source is always an admissible search anchor.
    # Its constructor forbids attaching any derived axis values, so accepting it
    # here cannot smuggle an undeclared transformation past domain validation.
    if candidate.materialization_mode == "authoring":
        return

    from clay.layout.intent import layout_intent_from_cell_def

    spec = compact_intent_domain_spec(cell)
    if (
        candidate.left_column_group
        and candidate.left_column_group not in spec.left_column_groups
    ):
        raise ValueError(
            "compact candidate left_column_group is outside the declared "
            "common-centroid domain"
        )
    splits = {choice.passive: choice for choice in candidate.passive_splits}
    if set(splits) != set(spec.split_passives):
        raise ValueError(
            "compact candidate split passives are outside the declared domain"
        )
    for choice in splits.values():
        if (
            choice.unit_count not in spec.split_counts
            or choice.connection != "parallel"
            or choice.orientation not in spec.orientations
        ):
            raise ValueError(
                f"compact candidate split for {choice.passive!r} is outside "
                "the declared domain"
            )

    orientations = dict(candidate.passive_orientations)
    if not set(orientations).issubset(spec.orientation_only_passives):
        raise ValueError(
            "compact candidate passive orientations are outside the declared domain"
        )
    if any(value not in spec.orientations for value in orientations.values()):
        raise ValueError("compact candidate orientation is outside the declared domain")

    eligible_passive_arrays = set(spec.split_passives) | set(
        spec.orientation_only_passives
    )
    devices = {
        str(getattr(device, "instance_name", "")): device
        for device in (getattr(cell, "devices", ()) or ())
    }
    for choice in candidate.passive_arrays:
        device = devices.get(choice.passive)
        if (
            choice.passive not in eligible_passive_arrays
            or device is None
            or str(getattr(device, "device_class", "")).strip().lower()
            not in {"cap", "capacitor"}
        ):
            raise ValueError(
                f"passive array {choice.passive!r} is not an eligible parallel MIM"
            )
        sizing = getattr(device, "sizing", None)
        if type(sizing) is not dict:
            raise TypeError("passive array source sizing must be an exact dict")
        raw_m = sizing.get("M", 1)
        if (
            type(raw_m) not in {int, float}
            or isinstance(raw_m, bool)
            or not math.isfinite(float(raw_m))
            or float(raw_m) <= 1.0
            or not float(raw_m).is_integer()
        ):
            raise ValueError(
                f"passive array {choice.passive!r} requires integral source M > 1"
            )
        multiplicity = int(raw_m)
        if choice.rows > multiplicity:
            raise ValueError("passive array rows must not exceed source multiplicity")
        if multiplicity % choice.rows:
            raise ValueError(
                "passive array rows must exactly divide source multiplicity"
            )

    intent = layout_intent_from_cell_def(cell)
    centroid_rows = {item.group: item.rows for item in intent.common_centroids}
    selected_rows = dict(candidate.common_centroid_rows)
    if not set(selected_rows) <= set(centroid_rows):
        raise ValueError(
            "compact candidate common-centroid row group is outside the declared domain"
        )
    dummy_flanks = {item.target: item for item in intent.dummy_flanks}
    for group, rows in selected_rows.items():
        error = common_centroid_row_materialization_error(
            centroid_rows[group], rows, dummy_flanks.get(group)
        )
        if error is not None:
            raise ValueError(f"compact candidate {error}")

    if candidate.physical_macro_graph_sha256 is not None:
        passive_bank_partition_intents_for_candidate(cell, candidate)
        graph = physical_macro_graph_for_candidate(
            cell,
            candidate,
            allow_materialized_passive_arrays=allow_materialized_passive_arrays,
        )
        if candidate.physical_macro_graph_sha256 != graph.sha256:
            raise ValueError(
                "compact candidate physical macro graph digest does not match source"
            )
        if candidate.well_island_partitions:
            expected_well_ids = {
                macro.macro_id for macro in graph.macros if macro.kind == "well_island"
            }
            actual_well_ids = {
                choice.macro_id for choice in candidate.well_island_partitions
            }
            if actual_well_ids != expected_well_ids:
                raise ValueError(
                    "compact candidate well-island partitions do not cover source graph"
                )
            for choice in candidate.well_island_partitions:
                validate_graph_partition_choice(
                    graph, cell, choice, expected_kind="well_island"
                )

    _validate_guard_ring_partition(cell, candidate)
    if candidate.group_order or candidate.group_floorplan:
        from clay.flow.optimized_layout_style import (
            derive_optimized_placement_groups,
        )

        placement_groups = derive_optimized_placement_groups(
            cell,
            candidate=candidate,
        )
        actual_groups = {group.name for group in placement_groups}
        if candidate.group_order:
            selected_groups = set(candidate.group_order)
            if (
                len(candidate.group_order) != len(placement_groups)
                or selected_groups != actual_groups
            ):
                unknown = sorted(selected_groups - actual_groups)
                missing = sorted(actual_groups - selected_groups)
                raise ValueError(
                    "compact candidate group_order must name every derived placement "
                    f"group exactly once; unknown={unknown}, missing={missing}"
                )
        if candidate.group_floorplan:
            selected_floorplan = {
                group for group, _tier, _column in candidate.group_floorplan
            }
            if not selected_floorplan <= actual_groups:
                unknown = sorted(selected_floorplan - actual_groups)
                raise ValueError(
                    "compact candidate group_floorplan contains unknown derived "
                    f"placement groups: unknown={unknown}"
                )
    if candidate.passive_side not in spec.passive_sides:
        raise ValueError(
            "compact candidate passive_side is outside the declared domain"
        )
    if candidate.pack_mode not in spec.pack_modes:
        raise ValueError("compact candidate pack_mode is outside the declared domain")
    declared_aspects = {
        "aspect:" + format(value, ".12g") for value in spec.aspect_points
    }
    area_minimized = intent.objective == ("area", "minimize")
    if candidate.aspect_mode == "natural":
        if not area_minimized:
            raise ValueError(
                "natural compact aspect requires an area-minimize objective"
            )
    elif candidate.aspect_mode not in declared_aspects:
        raise ValueError("compact candidate aspect_mode is outside the declared domain")

    def require_bounded(
        label: str,
        value: float | None,
        bounds: tuple[float, float],
    ) -> None:
        if value is None or value < bounds[0] or value > bounds[1]:
            raise ValueError(
                f"compact candidate {label} is outside the declared domain"
            )

    require_bounded(
        "edge_rail_width_um",
        candidate.edge_rail_width_um,
        spec.edge_rail_width_bounds_um,
    )
    require_bounded(
        "block_gap_um",
        candidate.block_gap_um,
        spec.block_gap_bounds_um,
    )
    if candidate.mos_fold_height_um is None or not any(
        math.isclose(
            candidate.mos_fold_height_um,
            value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for value in spec.fold_height_points_um
    ):
        raise ValueError(
            "compact candidate mos_fold_height_um is outside the declared domain"
        )


__all__ = [
    "CompactIntentDomainSpec",
    "GraphPartitionDomain",
    "GraphPartitionError",
    "GraphPartitionRejection",
    "bounded_guard_ring_axis",
    "common_centroid_row_materialization_error",
    "compact_structure_fingerprint",
    "compact_intent_domain_spec",
    "graph_partition_axes",
    "guard_ring_choice",
    "matched_passive_orientation_components",
    "passive_bank_partition_intents_for_candidate",
    "physical_macro_graph_for_cell",
    "physical_macro_graph_for_candidate",
    "validate_compact_intent_candidate",
    "validate_graph_partition_choice",
]
