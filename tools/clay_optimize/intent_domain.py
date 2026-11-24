"""Declaration-driven compact structural-intent candidate construction.

This module is circuit- and process-name neutral.  It lowers numeric
``opt_layout_*`` DATA from a parsed Clay cell, intersects physical bounds with
the selected PDK's rules, and returns the existing ``CompactIntentCandidate``
type consumed by the existing search/ledger engine.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from functools import reduce
from operator import mul
from typing import Iterable, Sequence

from clay.layout.compact_intent import (
    CompactIntentCandidate,
    PassiveSplitChoice,
    derive_guard_ring_atoms,
)
from clay.layout.compact_intent_domain import (
    CompactIntentDomainSpec,
    bounded_guard_ring_axis as _bounded_guard_ring_axis,
    common_centroid_row_materialization_error,
    compact_structure_fingerprint,
    compact_intent_domain_spec,
    graph_partition_axes,
    guard_ring_choice as _ring_choice,
    matched_passive_orientation_components,
    passive_isolation_ring_anchor,
    physical_macro_graph_for_candidate,
    physical_macro_graph_for_cell,
    validate_compact_intent_candidate,
)


@dataclass(frozen=True)
class CompactIntentDomain:
    candidates: tuple[CompactIntentCandidate, ...]
    exhausted: bool
    total_combinations: int


def _snap_up_um(value_um: float, grid_nm: int) -> float:
    value_nm = int(math.ceil(value_um * 1000.0 - 1e-12))
    return (((value_nm + grid_nm - 1) // grid_nm) * grid_nm) / 1000.0


def _physical_rail_points(
    cell: object,
    pdk: str,
    bounds: tuple[float, float],
    declared_points: Sequence[float],
) -> tuple[float, ...]:
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.route.rail_plan import _required_metal_rule
    from clay.layout.route.width_model import WidthPolicy
    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(pdk)
    lower, upper = bounds
    lower = max(lower, float(WidthPolicy().power_min_um))
    for trunk in layout_intent_from_cell_def(cell).trunks:
        if not str(getattr(trunk, "edge", "") or "").strip():
            continue
        match = re.fullmatch(r"metal([1-9][0-9]*)", str(trunk.layer).casefold())
        if match is None:
            raise ValueError(f"cannot resolve declared trunk layer {trunk.layer!r}")
        lower = max(
            lower,
            _required_metal_rule(tech, int(match.group(1)), "min_width") / 1000.0,
        )
    if lower > upper:
        raise ValueError(
            f"declared edge-rail upper bound {upper}um is below PDK lower bound {lower}um"
        )
    grid_nm = pdk_manufacturing_grid_nm(tech)
    points = {lower, upper}
    points.update(value for value in declared_points if lower <= value <= upper)
    return tuple(sorted({_snap_up_um(value, grid_nm) for value in points}))


def _physical_gap_points(
    pdk: str, bounds: tuple[float, float], declared_points: Sequence[float]
) -> tuple[float, ...]:
    from clay.layout.extract.realize_render import _group_ring_base_clearance_nm
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(pdk)
    lower, upper = bounds
    lower = max(lower, _group_ring_base_clearance_nm(tech) / 1000.0)
    if lower > upper:
        raise ValueError(
            f"declared block-gap upper bound {upper}um is below PDK lower bound {lower}um"
        )
    grid_nm = pdk_manufacturing_grid_nm(tech)
    points = {lower, upper}
    points.update(value for value in declared_points if lower <= value <= upper)
    return tuple(sorted({_snap_up_um(value, grid_nm) for value in points}))


def _sample_indices(
    axes: Sequence[Sequence[object]],
    budget: int,
    seed: int,
    *,
    required_vectors: Sequence[Sequence[int]] = (),
) -> tuple[int, ...]:
    total = reduce(mul, (len(axis) for axis in axes), 1)
    if total <= budget:
        return tuple(range(total))

    def encode(vector: Sequence[int]) -> int:
        index = 0
        for position, axis in zip(vector, axes):
            index = index * len(axis) + position
        return index

    center = [len(axis) // 2 for axis in axes]
    required_indices = {encode(vector) for vector in required_vectors}
    chosen = {encode([0] * len(axes)), encode(center), total - 1}
    chosen.update(required_indices)

    def bounded() -> tuple[int, ...]:
        retained = set(required_indices)
        retained.update(sorted(chosen - retained)[: max(0, budget - len(retained))])
        return tuple(sorted(retained))

    # One-factor anchors expose every declared value without pinning all later
    # axes to their first point (the old linear-stride anchors heavily biased
    # large domains toward one passive side/pack/ring combination).
    for axis_index, axis in enumerate(axes):
        for position in sorted({0, len(axis) // 2, len(axis) - 1}):
            vector = [
                (axis_index + other_index) % len(other_axis)
                for other_index, other_axis in enumerate(axes)
            ]
            vector[axis_index] = position
            chosen.add(encode(vector))
            if len(chosen) >= budget:
                return bounded()
    # Seeded per-axis projections avoid correlations between equal-cardinality
    # axes (for example side and pack both having two choices).
    cycle = 0
    while len(chosen) < budget:
        vector: list[int] = []
        for axis_index, axis in enumerate(axes):
            length = len(axis)
            digest = hashlib.sha256(
                f"{seed}:{axis_index}:{cycle}".encode("ascii")
            ).digest()
            vector.append(int.from_bytes(digest[:8], "big") % length)
        chosen.add(encode(vector))
        cycle += 1
        if cycle > budget * max(len(axes), 1) * 16:
            break
    nonce = 0
    while len(chosen) < budget:
        raw = f"{seed}:{nonce}".encode("ascii")
        chosen.add(int.from_bytes(hashlib.sha256(raw).digest(), "big") % total)
        nonce += 1
    return bounded()


def _decode(index: int, axes: Sequence[Sequence[object]]) -> tuple[object, ...]:
    values: list[object] = [None] * len(axes)
    for position in range(len(axes) - 1, -1, -1):
        axis = axes[position]
        values[position] = axis[index % len(axis)]
        index //= len(axis)
    return tuple(values)


def _common_centroid_row_axis(
    cell: object,
    *,
    budget: int,
    seed: int,
) -> tuple[tuple[tuple[tuple[str, int], ...], ...], bool]:
    """Return bounded row-compaction strategies from declared centroid rows."""

    from clay.layout.intent import layout_intent_from_cell_def

    intent = layout_intent_from_cell_def(cell)
    declarations = tuple(
        sorted(
            intent.common_centroids,
            key=lambda item: item.group,
        )
    )
    if not declarations:
        return (((),), True)
    dummy_flanks = {item.target: item for item in intent.dummy_flanks}
    axes: list[tuple[int | None, ...]] = []
    for declaration in declarations:
        declared_rows = declaration.rows
        if type(declared_rows) is not int or declared_rows <= 0:
            raise ValueError("common-centroid rows must be exact positive integers")
        compact_rows = tuple(
            rows
            for rows in range(1, declared_rows)
            if declared_rows % rows == 0
            and common_centroid_row_materialization_error(
                declared_rows,
                rows,
                dummy_flanks.get(declaration.group),
            )
            is None
        )
        axes.append((None, *compact_rows))
    total = reduce(mul, (len(axis) for axis in axes), 1)
    required = [[0] * len(axes)]
    for index, axis in enumerate(axes):
        if len(axis) > 1:
            vector = [0] * len(axes)
            vector[index] = 1
            required.append(vector)
    if any(len(axis) > 1 for axis in axes):
        required.append([1 if len(axis) > 1 else 0 for axis in axes])
    indices = _sample_indices(
        axes,
        min(total, budget),
        seed,
        required_vectors=tuple(required),
    )
    strategies = tuple(
        tuple(
            (declaration.group, int(rows))
            for declaration, rows in zip(declarations, _decode(index, axes))
            if rows is not None
        )
        for index in indices
    )
    return strategies, total <= budget


def build_compact_intent_domain(
    parsed: object, cell: object, pdk: str
) -> CompactIntentDomain:
    """Build a deterministic bounded domain from parsed Clay and PDK facts."""

    spec = compact_intent_domain_spec(cell)
    physical_macro_graph = physical_macro_graph_for_cell(cell)
    graph_partitions = graph_partition_axes(
        physical_macro_graph,
        cell,
        budget=spec.budget,
        seed=spec.seed,
    )
    passive_partitions = graph_partitions.passive_bank_axis
    row_strategies, rows_exhausted = _common_centroid_row_axis(
        cell,
        budget=spec.budget,
        seed=spec.seed,
    )
    split_axes: list[tuple[tuple[int, str], ...]] = [
        tuple(
            (count, orientation)
            for count in spec.split_counts
            for orientation in spec.orientations
        )
        for _passive in spec.split_passives
    ]
    orientation_components = matched_passive_orientation_components(
        cell, spec.orientation_only_passives
    )
    orient_axes: list[tuple[str | None, ...]] = [
        (None, *spec.orientations) for _component in orientation_components
    ]
    ring_atoms = derive_guard_ring_atoms(cell)
    rings, rings_exhausted = _bounded_guard_ring_axis(
        ring_atoms, budget=spec.budget, seed=spec.seed
    )
    fixed_aspects = tuple(
        "aspect:" + format(value, ".12g") for value in spec.aspect_points
    )
    from clay.layout.intent import layout_intent_from_cell_def

    area_minimized = (
        layout_intent_from_cell_def(cell).objective == ("area", "minimize")
    )
    declared_channels = layout_intent_from_cell_def(cell).channels
    channel_strategies = (
        ("declared", "block_gap_only")
        if any(channel.orient == "vertical" for channel in declared_channels)
        else ("declared",)
    )
    aspects = (("natural",) + fixed_aspects) if area_minimized else fixed_aspects
    rails = _physical_rail_points(
        cell,
        pdk,
        spec.edge_rail_width_bounds_um,
        spec.edge_rail_width_points_um,
    )
    gaps = _physical_gap_points(pdk, spec.block_gap_bounds_um, spec.block_gap_points_um)
    folds = spec.fold_height_points_um
    left_columns = ("", *spec.left_column_groups)
    from clay.layout.policy import policy_for_cell

    declared_left_column = policy_for_cell(parsed, cell).left_column_group
    if declared_left_column not in left_columns:
        raise ValueError(
            "declared place-policy left column is outside the compact domain"
        )
    axes: list[Sequence[object]] = [
        passive_partitions,
        *split_axes,
        *orient_axes,
        rings,
    ]
    row_axis_index: int | None = None
    if len(row_strategies) > 1:
        row_axis_index = len(axes)
        axes.append(row_strategies)
    axes.append(left_columns)
    channel_axis_index: int | None = None
    if len(channel_strategies) > 1:
        channel_axis_index = len(axes)
        axes.append(channel_strategies)
    axes.append(spec.passive_sides)
    aspect_axis_index = len(axes)
    pack_axis_index = aspect_axis_index + 1
    axes.extend((aspects, spec.pack_modes, rails, folds, gaps))
    derived_total = reduce(mul, (len(axis) for axis in axes), 1)
    total = derived_total + 1
    candidates: set[CompactIntentCandidate] = {
        CompactIntentCandidate(materialization_mode="authoring")
    }
    partition_axis_index = 0
    preferred: list[int] = [
        min(
            range(len(passive_partitions)),
            key=lambda index: (
                sum(len(choice.member_groups) for choice in passive_partitions[index]),
                index,
            ),
        )
    ]
    preferred.extend(
        axis.index((1, "R0")) if (1, "R0") in axis else 0 for axis in split_axes
    )
    preferred.extend(0 for _axis in orient_axes)  # None: preserve declared orientation.
    ring_axis_index = len(preferred)
    preferred.append(
        min(range(len(rings)), key=lambda index: (len(rings[index]), index))
    )
    if row_axis_index is not None:
        preferred.append(row_strategies.index(()))
    preferred.append(left_columns.index(declared_left_column))
    if channel_axis_index is not None:
        preferred.append(channel_strategies.index("declared"))
    preferred.extend(
        (
            spec.passive_sides.index("center") if "center" in spec.passive_sides else 0,
            min(
                range(len(aspects)),
                key=lambda index: abs(
                    _aspect_number(aspects[index]) - spec.aspect_nominal
                ),
            ),
            spec.pack_modes.index("rows") if "rows" in spec.pack_modes else 0,
            min(range(len(rails)), key=lambda index: rails[index]),
            min(
                range(len(folds)),
                key=lambda index: abs(folds[index] - spec.fold_height_nominal_um),
            ),
            min(range(len(gaps)), key=lambda index: gaps[index]),
        )
    )
    # A bounded high-dimensional sample must not accidentally collapse the
    # structural ring axis to the two pathological ends: one near-global well
    # or near-singleton rings around every atom.  Retain declared-budget anchors
    # across the actual ring-count span while all other axes sit at their same
    # nominal vector.  These are ordinary candidates in the existing domain,
    # not a new syntax or a circuit-specific construction.
    ring_counts = tuple(len(choice) for choice in rings)
    low_count = min(ring_counts)
    high_count = max(ring_counts)
    required_vectors = [tuple(preferred)]
    if "natural" in aspects and len(required_vectors) < spec.budget:
        natural_vector = list(preferred)
        natural_vector[aspect_axis_index] = aspects.index("natural")
        required_vectors.append(tuple(natural_vector))
    if row_axis_index is not None:
        for row_index in range(len(row_strategies)):
            vector = list(preferred)
            vector[row_axis_index] = row_index
            candidate_vector = tuple(vector)
            if (
                candidate_vector not in required_vectors
                and len(required_vectors) < spec.budget
            ):
                required_vectors.append(candidate_vector)
    if channel_axis_index is not None:
        # A declared channel creates a real structural choice: reserve the
        # compact (natural-aspect, dense-pack) block-gap point across every
        # bounded row and passive-bank partition.  Use the central ring-count
        # anchor so this axis does not collapse to either singleton rings or
        # one near-global well.  These are coordinate-free domain vectors;
        # they retain a promoted compact structure without naming a circuit.
        central_ring_target = (low_count + high_count) / 2.0
        central_ring_index = min(
            range(len(rings)),
            key=lambda index: (
                abs(ring_counts[index] - central_ring_target),
                ring_counts[index],
                index,
            ),
        )
        block_gap_rows = (
            range(len(row_strategies)) if row_axis_index is not None else (0,)
        )
        for row_index in block_gap_rows:
            for partition_index in range(len(passive_partitions)):
                vector = list(preferred)
                vector[partition_axis_index] = partition_index
                vector[ring_axis_index] = central_ring_index
                if row_axis_index is not None:
                    vector[row_axis_index] = row_index
                vector[channel_axis_index] = channel_strategies.index(
                    "block_gap_only"
                )
                if "natural" in aspects:
                    vector[aspect_axis_index] = aspects.index("natural")
                if "dense_2d" in spec.pack_modes:
                    vector[pack_axis_index] = spec.pack_modes.index("dense_2d")
                candidate_vector = tuple(vector)
                if (
                    candidate_vector not in required_vectors
                    and len(required_vectors) < spec.budget - 1
                ):
                    required_vectors.append(candidate_vector)
    passive_ring_anchor = passive_isolation_ring_anchor(ring_atoms)
    if passive_ring_anchor is not None and passive_ring_anchor in rings:
        passive_ring_index = rings.index(passive_ring_anchor)
        for partition_index in range(len(passive_partitions)):
            vector = list(preferred)
            vector[partition_axis_index] = partition_index
            vector[ring_axis_index] = passive_ring_index
            if "natural" in aspects:
                vector[aspect_axis_index] = aspects.index("natural")
            candidate_vector = tuple(vector)
            if (
                candidate_vector not in required_vectors
                and len(required_vectors) < spec.budget
            ):
                required_vectors.append(candidate_vector)
    for numerator in (0, 1, 2, 3, 4):
        target = low_count + (high_count - low_count) * numerator / 4.0
        ring_index = min(
            range(len(rings)),
            key=lambda index: (
                abs(ring_counts[index] - target),
                ring_counts[index],
                index,
            ),
        )
        vector = list(preferred)
        vector[ring_axis_index] = ring_index
        candidate_vector = tuple(vector)
        if (
            candidate_vector not in required_vectors
            and len(required_vectors) < spec.budget
        ):
            required_vectors.append(candidate_vector)
    partition_counts = tuple(
        sum(len(choice.member_groups) for choice in partition)
        for partition in passive_partitions
    )
    low_partition_count = min(partition_counts)
    high_partition_count = max(partition_counts)
    for numerator in (0, 1, 2, 3, 4):
        target = (
            low_partition_count
            + (high_partition_count - low_partition_count) * numerator / 4.0
        )
        partition_index = min(
            range(len(passive_partitions)),
            key=lambda index: (
                abs(partition_counts[index] - target),
                partition_counts[index],
                index,
            ),
        )
        vector = list(preferred)
        vector[partition_axis_index] = partition_index
        candidate_vector = tuple(vector)
        if (
            candidate_vector not in required_vectors
            and len(required_vectors) < spec.budget
        ):
            required_vectors.append(candidate_vector)

    derived_budget = max(0, spec.budget - 1)
    if row_axis_index is not None and derived_budget:
        # Adding an orthogonal row axis must not silently erase the ring
        # structures reachable before row/channel refinement.  Replay that
        # declared-channel projection and retain one neutral-row anchor for
        # every ring choice it exposed; remaining slots still sample the full
        # cross-product below.
        projection_skip = {row_axis_index}
        if channel_axis_index is not None:
            projection_skip.add(channel_axis_index)
        projection_indices = tuple(
            index for index in range(len(axes)) if index not in projection_skip
        )
        rowless_axes = tuple(axes[index] for index in projection_indices)
        rowless_required = tuple(
            tuple(vector[index] for index in projection_indices)
            for vector in required_vectors
            if (
                channel_axis_index is None
                or vector[channel_axis_index]
                == channel_strategies.index("declared")
            )
        )
        rowless_indices = _sample_indices(
            rowless_axes,
            derived_budget,
            spec.seed,
            required_vectors=rowless_required,
        )
        retained_vectors: dict[object, tuple[int, ...]] = {}
        for index in rowless_indices:
            values = _decode(index, rowless_axes)
            ring_choice = values[ring_axis_index]
            retained_vectors.setdefault(
                ring_choice,
                tuple(
                    axis.index(value)
                    for axis, value in zip(rowless_axes, values)
                ),
            )
        for retained_ring in sorted(retained_vectors, key=rings.index):
            rowless_vector = retained_vectors[retained_ring]
            candidate = list(preferred)
            for axis_index, value in zip(projection_indices, rowless_vector):
                candidate[axis_index] = value
            candidate[row_axis_index] = row_strategies.index(())
            if channel_axis_index is not None:
                candidate[channel_axis_index] = channel_strategies.index("declared")
            candidate_vector = tuple(candidate)
            if (
                candidate_vector not in required_vectors
                and len(required_vectors) < derived_budget
            ):
                required_vectors.append(candidate_vector)
    sampled_indices = (
        ()
        if derived_budget == 0
        else _sample_indices(
            axes,
            derived_budget,
            spec.seed,
            required_vectors=tuple(required_vectors[:derived_budget]),
        )
    )
    for index in sampled_indices:
        values = _decode(index, axes)
        partition_choice = values[0]
        cursor = 1
        splits = tuple(
            PassiveSplitChoice(passive, values[cursor + offset][0], "parallel", values[cursor + offset][1])  # type: ignore[index]
            for offset, passive in enumerate(spec.split_passives)
        )
        cursor += len(spec.split_passives)
        passive_orientations = tuple(
            (passive, str(values[cursor + offset]))
            for offset, component in enumerate(orientation_components)
            for passive in component
            if values[cursor + offset] is not None
        )
        cursor += len(orientation_components)
        ring_choice = values[cursor]
        cursor += 1
        if row_axis_index is None:
            row_strategy: object = ()
        else:
            row_strategy = values[cursor]
            cursor += 1
        left_column = values[cursor]
        cursor += 1
        if channel_axis_index is None:
            channel_strategy: object = "declared"
        else:
            channel_strategy = values[cursor]
            cursor += 1
        side, aspect, pack, rail, fold, gap = values[cursor : cursor + 6]
        candidate = CompactIntentCandidate(
            passive_splits=splits,
            guard_ring_groups=tuple(ring_choice),  # type: ignore[arg-type]
            # The existing optimized-style lowerer already derives physical
            # group order from functional-role DATA.  An empty override
            # preserves that single source of truth until a group-order axis
            # is explicitly declared in Clay.
            group_order=(),
            passive_orientations=passive_orientations,
            passive_side=str(side),
            aspect_mode=str(aspect),
            pack_mode=str(pack),
            edge_rail_width_um=float(rail),
            mos_fold_height_um=float(fold),
            block_gap_um=float(gap),
            physical_macro_graph_sha256=physical_macro_graph.sha256,
            passive_bank_partitions=tuple(partition_choice),  # type: ignore[arg-type]
            common_centroid_rows=tuple(row_strategy),  # type: ignore[arg-type]
            left_column_group=str(left_column),
            channel_strategy=str(channel_strategy),
        )
        effective_graph = physical_macro_graph_for_candidate(cell, candidate)
        candidates.add(
            replace(
                candidate,
                physical_macro_graph_sha256=effective_graph.sha256,
            )
        )
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.digest))
    return CompactIntentDomain(
        candidates=ordered,
        exhausted=(
            rings_exhausted
            and rows_exhausted
            and graph_partitions.exhausted
            and derived_total <= derived_budget
        ),
        total_combinations=total,
    )


def _aspect_number(mode: str) -> float:
    if mode == "natural":
        return 1.0
    if mode.startswith("aspect:"):
        try:
            value = float(mode.split(":", 1)[1])
        except ValueError:
            return math.inf
        return value if math.isfinite(value) and value > 0.0 else math.inf
    return math.inf


def _compact_priority(candidate: CompactIntentCandidate) -> tuple[object, ...]:
    origin = 0 if candidate.materialization_mode == "authoring" else 1
    if origin == 0:
        return (origin, candidate.digest)
    numeric_axes = (
        candidate.edge_rail_width_um,
        candidate.block_gap_um,
        candidate.mos_fold_height_um,
    )
    if any(value is None for value in numeric_axes):
        raise ValueError(
            "derived compact candidate must carry finite numeric geometry axes"
        )
    return (
        origin,
        float(candidate.edge_rail_width_um),
        float(candidate.block_gap_um),
        len(candidate.guard_ring_groups),
        sum(choice.unit_count for choice in candidate.passive_splits),
        len(candidate.passive_orientations),
        abs(_aspect_number(candidate.aspect_mode) - 1.0),
        abs(float(candidate.mos_fold_height_um) - 10.0),
        candidate.digest,
    )


def rank_compact_intent_candidates(
    candidates: Iterable[CompactIntentCandidate],
) -> tuple[CompactIntentCandidate, ...]:
    """Rank compact candidates while retaining early anchor/pack diversity."""

    unique = set(candidates)
    if not unique:
        return ()
    anchors = sorted(
        (
            candidate
            for candidate in unique
            if candidate.materialization_mode == "authoring"
        ),
        key=_compact_priority,
    )
    derived = {
        candidate
        for candidate in unique
        if candidate.materialization_mode == "derived"
    }
    buckets: dict[str, list[CompactIntentCandidate]] = {}
    for candidate in derived:
        buckets.setdefault(compact_structure_fingerprint(candidate), []).append(
            candidate
        )
    for values in buckets.values():
        values.sort(key=_compact_priority)
    ordered: list[CompactIntentCandidate] = list(anchors)
    keys = sorted(buckets)
    while any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    return tuple(ordered)


def recombine_compact_intent_candidates(
    parents: Iterable[CompactIntentCandidate],
    *,
    maximum: int,
) -> tuple[CompactIntentCandidate, ...]:
    """Boundedly cross measured structure with measured geometry controls.

    The initial high-dimensional sample can discover a good guard-ring/passive
    topology only while pairing it with a poor fold/aspect/gap vector.  This
    deterministic refinement preserves topology-affecting axes from one parent
    and takes placement-control axes from another.  It is circuit/PDK neutral,
    emits only novel candidates, and never exceeds the declared budget.
    """

    if type(maximum) is not int or maximum < 0:
        raise ValueError("compact intent refinement maximum must be non-negative")
    if maximum == 0:
        return ()
    ordered = tuple(
        sorted(
            {
                candidate
                for candidate in parents
                if candidate.materialization_mode == "derived"
            },
            key=lambda candidate: candidate.digest,
        )
    )
    parent_set = set(ordered)
    geometry_axes = (
        "pack_mode",
        "passive_side",
        "aspect_mode",
        "mos_fold_height_um",
        "block_gap_um",
        "edge_rail_width_um",
        "channel_strategy",
    )
    refinements: dict[CompactIntentCandidate, tuple[object, ...]] = {}
    for structural in ordered:
        for geometry in ordered:
            if structural is geometry:
                continue
            # A physical-graph digest binds every graph-affecting structural
            # axis as one vector.  A single donated axis may inherit the
            # donor's digest only when all of the other bound axes already
            # agree; otherwise this cell-free recombiner cannot mint the
            # combined graph identity honestly.
            graph_axes = (
                "guard_ring_groups",
                "passive_bank_partitions",
                "well_island_partitions",
                "common_centroid_rows",
            )
            structural_changes = (
                (
                    "guard_ring_groups",
                    {
                        "guard_ring_groups": geometry.guard_ring_groups,
                        "physical_macro_graph_sha256": (
                            geometry.physical_macro_graph_sha256
                        ),
                    },
                ),
                (
                    "passive_bank_partitions",
                    {
                        "passive_bank_partitions": (
                            geometry.passive_bank_partitions
                        ),
                        "physical_macro_graph_sha256": (
                            geometry.physical_macro_graph_sha256
                        ),
                    },
                ),
                (
                    "left_column_group",
                    {"left_column_group": geometry.left_column_group},
                ),
                (
                    "common_centroid_rows",
                    {
                        "common_centroid_rows": geometry.common_centroid_rows,
                        "physical_macro_graph_sha256": (
                            geometry.physical_macro_graph_sha256
                        ),
                    },
                ),
            )
            for axis_index, (axis, changes) in enumerate(structural_changes):
                if getattr(structural, axis) == getattr(geometry, axis):
                    continue
                if axis in graph_axes and any(
                    getattr(structural, graph_axis) != getattr(geometry, graph_axis)
                    for graph_axis in graph_axes
                    if graph_axis != axis
                ):
                    # A donor SHA is valid only when every other graph-affecting
                    # axis is identical; otherwise a cell-bound replay must mint
                    # the combined graph rather than guessing here.
                    continue
                candidate = replace(structural, **changes)
                if candidate not in parent_set:
                    score = (
                        0,
                        axis_index,
                        structural.digest,
                        geometry.digest,
                        candidate.digest,
                    )
                    prior = refinements.get(candidate)
                    if prior is None or score < prior:
                        refinements[candidate] = score
            candidate = replace(
                structural,
                passive_side=geometry.passive_side,
                aspect_mode=geometry.aspect_mode,
                pack_mode=geometry.pack_mode,
                edge_rail_width_um=geometry.edge_rail_width_um,
                mos_fold_height_um=geometry.mos_fold_height_um,
                block_gap_um=geometry.block_gap_um,
                channel_strategy=geometry.channel_strategy,
            )
            if candidate not in parent_set:
                score = (
                    1,
                    structural.digest,
                    geometry.digest,
                    candidate.digest,
                )
                prior = refinements.get(candidate)
                if prior is None or score < prior:
                    refinements[candidate] = score
            for axis_index, axis in enumerate(geometry_axes):
                candidate = replace(
                    structural,
                    **{axis: getattr(geometry, axis)},
                )
                if candidate not in parent_set:
                    score = (
                        2,
                        axis_index,
                        structural.digest,
                        geometry.digest,
                        candidate.digest,
                    )
                    prior = refinements.get(candidate)
                    if prior is None or score < prior:
                        refinements[candidate] = score
    return tuple(
        candidate
        for candidate, _score in sorted(
            refinements.items(),
            key=lambda item: item[1],
        )[:maximum]
    )


def reconstruct_compact_intent_candidate(
    base_candidates: Iterable[CompactIntentCandidate],
    candidate_sha256: str,
) -> CompactIntentCandidate:
    """Recover one base or pairwise-refined point from its canonical digest."""

    digest = str(candidate_sha256)
    ordered = tuple(
        sorted(set(base_candidates), key=lambda candidate: candidate.digest)
    )
    direct = tuple(candidate for candidate in ordered if candidate.digest == digest)
    if len(direct) == 1:
        return direct[0]
    matches: set[CompactIntentCandidate] = set()
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            matches.update(
                candidate
                for candidate in recombine_compact_intent_candidates(
                    (left, right),
                    maximum=16,
                )
                if candidate.digest == digest
            )
    if len(matches) != 1:
        raise ValueError(
            f"compact candidate {digest!r} is not reconstructible from the "
            "declared base domain"
        )
    return next(iter(matches))


__all__ = [
    "CompactIntentDomain",
    "CompactIntentDomainSpec",
    "build_compact_intent_domain",
    "compact_intent_domain_spec",
    "rank_compact_intent_candidates",
    "recombine_compact_intent_candidates",
    "reconstruct_compact_intent_candidate",
    "validate_compact_intent_candidate",
]
