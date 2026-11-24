"""Full-2D compaction over the same declared geometry and replay obligations."""
from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from clay.layout.mosaic_guest_slot_projection import MosaicGuestSlotProjection, add_mosaic_guest_slot_constraints, mosaic_guest_slot_satisfied
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from clay.layout.place.solver_execution_policy import incumbent_replay_enabled, indexed_incumbent_geometry_enabled
from clay.flow.execution_session import current_session
from clay.flow.execution_budget import ExecutionStopped
from clay.layout.place.constraints_hierarchical import (
    _add_aggregate_envelope_order_constraints,
    _add_hierarchical_macro_constraints,
    _assert_hierarchical_macro_positions,
    _canonical_aggregate_envelope_orders,
    _canonical_hierarchical_macro_alignments,
    _canonical_hierarchical_macro_orders,
    _hierarchical_full2d_pair_axis_authority,
    _hierarchical_full2d_pair_axis_preference,
    _require_hierarchical_alignment_tolerance_scales_nm,
)
from clay.layout.place.constraints_spatial import (
    _add_adjacency_constraints,
    _add_directional_aspect_constraints,
    _add_floorplan_column_group_constraints,
    _add_manufacturing_grid_constraints,
    _add_row_bounded_cavity_constraints,
    _canonical_axis_alignment_groups,
    _canonical_center_alignment_pairs,
    _canonical_floorplan_column_groups,
    _canonical_mosaic_guest_slot_projections,
    _canonical_row_bounded_cavity_projections,
    _required_pair_gap_nm,
)
from clay.layout.place.placement_constants import (
    _AggregateEnvelopeOrder,
    _Bbox,
    _FloorplanColumnGroup,
    _HierarchicalAlignmentToleranceScalesNm,
    _HierarchicalMacroAlignment,
    _HierarchicalMacroOrder,
    _LayeredCollisionFootprint,
    _RigidCompoundFootprint,
    _RowBoundedCavityProjection,
)
from clay.layout.place.placement_diagnostics import (
    Full2DCompactionDiagnosticError,
    _canonical_full_hard_failure_rows,
    _diagnostic_safe_canonical_value,
    _full2d_compaction_failure_diagnostic,
    _full2d_compaction_position_rows,
    _full2d_input_provenance,
    _full_hard_replay_row,
)
from clay.layout.place.placement_replay import (
    _require_edge_rail_reservation,
)


def _compact_lateral_sink_positions_x(
    positions: Mapping[str, Tuple[int, int]],
    *,
    lateral_blocks: Sequence[str],
    order_pairs: Sequence[Tuple[str, str]],
    manufacturing_grid_nm: int,
    legal: Callable[[Mapping[str, Tuple[int, int]]], bool],
) -> Tuple[Dict[str, Tuple[int, int]], bool]:
    """Move ordered lateral sinks to their first replay-legal X grid point."""

    grid = int(manufacturing_grid_nm)
    if grid <= 0:
        raise ValueError("lateral sink compaction requires a positive grid")
    result = {
        str(name): (int(point[0]), int(point[1]))
        for name, point in positions.items()
    }
    if not legal(result):
        raise ValueError("lateral sink compaction requires one replay-legal seed")
    sources = {str(left) for left, _right in order_pairs}
    targets = {str(right) for _left, right in order_pairs}
    sinks = tuple(sorted(set(map(str, lateral_blocks)) & targets - sources))
    changed = False
    for sink in sinks:
        current_x, current_y = result[sink]
        if current_x % grid:
            raise ValueError("lateral sink is off the manufacturing grid")
        low = 0
        high = current_x // grid
        while low < high:
            middle = (low + high) // 2
            candidate = dict(result)
            candidate[sink] = (middle * grid, current_y)
            if legal(candidate):
                high = middle
            else:
                low = middle + 1
        compact_x = low * grid
        if compact_x < current_x:
            result[sink] = (compact_x, current_y)
            changed = True
    return result, changed

def _add_full2d_outer_hard_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    *,
    dims: Mapping[str, Tuple[int, int]],
    width_bound_nm: int,
    height_bound_nm: int,
    minimum_clearance_nm: int,
    order_pairs: Sequence[Tuple[str, str]],
    vertical_order_pairs: Sequence[Tuple[str, str]],
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    hierarchical_cohesion_span_caps_nm: Tuple[Tuple[Tuple[str, ...], int, int], ...],
    floorplan_column_groups: Tuple[_FloorplanColumnGroup, ...],
    left_edge: Sequence[str],
) -> None:
    """Append five existing outer predicates; never rebase caps or objectives.

    Other outer modes remain the responsibility of the unchanged full-hard
    caller replay. This helper is not a replacement for that acceptance gate.
    """
    if not any((order_pairs, vertical_order_pairs, pair_clearances_nm,
                hierarchical_cohesion_span_caps_nm, floorplan_column_groups, left_edge)):
        return
    known = set(dims)
    int64_max = (1 << 63) - 1
    if type(minimum_clearance_nm) is not int or not 0 <= minimum_clearance_nm <= int64_max:
        raise ValueError("full2d outer minimum clearance must be a nonnegative exact int")

    def source_pairs(values: object, label: str) -> Tuple[Tuple[str, str], ...]:
        if not isinstance(values, (tuple, list)):
            raise TypeError(f"{label} must be a sequence of exact owner pairs")
        rows = []
        for row in values:
            if (type(row) is not tuple or len(row) != 2
                    or any(type(name) is not str or name not in known for name in row)
                    or row[0] == row[1]):
                raise ValueError(f"{label} contains a malformed or unowned pair")
            rows.append(row)
        if len(rows) != len(set(rows)):
            raise ValueError(f"{label} repeats a source pair")
        return tuple(rows)

    horizontal = source_pairs(order_pairs, "order_pairs")
    vertical = source_pairs(vertical_order_pairs, "vertical_order_pairs")
    if not isinstance(pair_clearances_nm, Mapping):
        raise TypeError("pair_clearances_nm must be a mapping")
    for pair, value in pair_clearances_nm.items():
        source_pairs((pair,), "pair_clearances_nm")
        if pair != tuple(sorted(pair)) or type(value) is not int or not 0 <= value <= int64_max:
            raise ValueError("pair_clearances_nm requires canonical pairs and exact nonnegative ints")
    caps = hierarchical_cohesion_span_caps_nm
    if type(caps) is not tuple:
        raise TypeError("hierarchical_cohesion_span_caps_nm must be an exact tuple")
    cap_groups = []
    for row in caps:
        if type(row) is not tuple or len(row) != 3:
            raise TypeError("hierarchy span cap must contain group and two original limits")
        group, x_cap, y_cap = row
        if (type(group) is not tuple or len(group) < 2
                or any(type(name) is not str or name not in known for name in group)
                or group != tuple(sorted(set(group)))
                or any(type(cap) is not int or not 0 < cap <= int64_max for cap in (x_cap, y_cap))):
            raise ValueError("hierarchy span cap group/limits are malformed or unowned")
        cap_groups.append(group)
    if cap_groups != sorted(set(cap_groups)):
        raise ValueError("hierarchy span caps must have sorted unique groups")
    columns = _canonical_floorplan_column_groups(
        floorplan_column_groups,
        block_ids={name: (0, 0, width, height) for name, (width, height) in dims.items()},
    )
    if (not isinstance(left_edge, (tuple, list))
            or any(type(name) is not str or name not in known for name in left_edge)
            or len(left_edge) != len(set(left_edge))):
        raise ValueError("left_edge must name unique owned blocks")

    for left, right in horizontal:
        # Horizontal source replay intentionally uses the global gap, even
        # when a physical pair override is larger. Do not strengthen it here.
        offset = dims[left][0] + minimum_clearance_nm
        if offset > int64_max:
            raise OverflowError("full2d horizontal source offset exceeds int64")
        model.Add(xs[right] >= xs[left] + offset)
    for lower, upper in vertical:
        gap = _required_pair_gap_nm(lower, upper, minimum_clearance_nm, pair_clearances_nm)
        offset = dims[lower][1] + gap
        if offset > int64_max:
            raise OverflowError("full2d vertical source offset exceeds int64")
        model.Add(ys[upper] >= ys[lower] + offset)
    for index, (group, x_cap, y_cap) in enumerate(caps):
        for axis, variables, dimension, bound, cap in (
            ("horizontal", xs, 0, width_bound_nm, x_cap),
            ("vertical", ys, 1, height_bound_nm, y_cap),
        ):
            prefix = f"full2d_outer_cohesion_{index}_{axis}"
            low = model.NewIntVar(0, bound, prefix + "_min")
            high = model.NewIntVar(0, bound, prefix + "_max")
            model.AddMinEquality(low, [variables[name] for name in group])
            model.AddMaxEquality(high, [variables[name] + dims[name][dimension] for name in group])
            model.Add(high - low <= cap)
    if columns:
        _add_floorplan_column_group_constraints(model, xs, columns)
    for owner in left_edge:
        model.Add(xs[owner] == 0)

def _compact_full2d_aggregate_target(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    graph: Any,
    minimum_clearance_nm: int,
    manufacturing_grid_nm: int,
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]],
    center_alignment_pairs: Sequence[Tuple[str, str, str]],
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder],
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder],
    hierarchical_macro_alignments: Sequence[_HierarchicalMacroAlignment],
    row_bounded_cavity_projections: Sequence[_RowBoundedCavityProjection],
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = (),
    edge_rail_reservation: Any = None,
    order_pairs: Sequence[Tuple[str, str]] = (),
    vertical_order_pairs: Sequence[Tuple[str, str]] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
    hierarchical_cohesion_span_caps_nm: Tuple[Tuple[Tuple[str, ...], int, int], ...] = (),
    floorplan_column_groups: Tuple[_FloorplanColumnGroup, ...] = (),
    left_edge: Sequence[str] = (),
    hierarchical_alignment_tolerance_scales_nm: (
        _HierarchicalAlignmentToleranceScalesNm
    ),
    adjacency_limits_nm: Sequence[Tuple[str, str, int]],
    adjacency_scope_offsets_nm: Mapping[str, _Bbox],
    aspect_target: float,
    aspect_bounds: Optional[Tuple[float, float]],
    aspect_obligation: bool,
    solve_evidence_sink: Optional[list[Mapping[str, object]]] = None,
) -> Optional[Dict[str, Tuple[int, int]]]:
    """Minimize a frozen Full2D topology with exact aggregate constraints."""

    from ortools.sat.python import cp_model
    from clay.layout.place.full2d_pair_cover import (
        build_full2d_pair_cover,
        cover_directional_slacks,
    )

    hierarchical_alignment_tolerance_scales_nm = (
        _require_hierarchical_alignment_tolerance_scales_nm(
            hierarchical_alignment_tolerance_scales_nm,
            alignments=hierarchical_macro_alignments,
        )
    )
    collision_cover = build_full2d_pair_cover(graph)
    physical_pair_count = sum(
        row.mode == "four_way_disjunction" for row in graph.pair_constraints
    )
    use_collision_cover = graph.physical_clearance_nm is not None or sum(
        len(group["forbidden_boxes_nm"]) for group in collision_cover["groups"]
    ) < physical_pair_count
    ids = tuple(sorted(positions))
    canonical_slot_projections = _canonical_mosaic_guest_slot_projections(
        mosaic_guest_slot_projections, dims=dims, minimum_clearance_nm=minimum_clearance_nm,
    )
    bound_placements = [dict(positions)]
    if graph.pair_constraints and all(
        row.mode == "fixed_edge" for row in graph.pair_constraints
    ) and not graph.replay({
        name: (
            int(positions[name][0]),
            int(positions[name][1]),
            int(positions[name][0]) + int(dims[name][0]),
            int(positions[name][1]) + int(dims[name][1]),
        )
        for name in ids
    }):
        from clay.layout.extract.compaction import compact_full2d_master_placement

        declared_boxes = compact_full2d_master_placement(
            {
                name: (
                    int(positions[name][0]),
                    int(positions[name][1]),
                    int(positions[name][0]) + int(dims[name][0]),
                    int(positions[name][1]) + int(dims[name][1]),
                )
                for name in ids
            },
            graph,
        )
        bound_placements.append({
            name: (box[0], box[1]) for name, box in declared_boxes.items()
        })
    width_bound = max(
        int(candidate[name][0]) + int(dims[name][0])
        for candidate in bound_placements
        for name in ids
    )
    height_bound = max(
        int(candidate[name][1]) + int(dims[name][1])
        for candidate in bound_placements
        for name in ids
    )
    grid = int(manufacturing_grid_nm)
    model = cp_model.CpModel()
    xs = {
        name: model.NewIntVar(0, width_bound - int(dims[name][0]), f"full2d_x_{name}")
        for name in ids
    }
    ys = {
        name: model.NewIntVar(0, height_bound - int(dims[name][1]), f"full2d_y_{name}")
        for name in ids
    }
    _add_manufacturing_grid_constraints(model, xs, ys, grid)

    node_owner = {
        node: owner for node, owner, _offset in graph.node_owner_offsets_nm
    }
    node_offset = {
        node: offset for node, _owner, offset in graph.node_owner_offsets_nm
    }
    node_size = dict(graph.sizes_nm)

    def coordinate(node: str, axis: int) -> Any:
        owner = node_owner[node]
        return (xs if axis == 0 else ys)[owner] + int(node_offset[node][axis])

    for axis, edges in enumerate((graph.horizontal_edges, graph.vertical_edges)):
        for edge in edges:
            model.Add(
                coordinate(edge.upper, axis)
                >= coordinate(edge.lower, axis)
                + int(node_size[edge.lower][axis])
                + int(edge.clearance_nm)
            )
    direction_literals = []
    cover_direction_literals = []
    direction_names = (
        "first_before_second_horizontal",
        "first_before_second_vertical",
        "second_before_first_horizontal",
        "second_before_first_vertical",
    )
    for pair_index, row in enumerate(graph.pair_constraints):
        if row.mode != "four_way_disjunction" or use_collision_cover:
            continue
        first = row.first
        second = row.second
        clearance = int(row.clearance_nm)
        directions = {
            "first_before_second_horizontal": model.NewBoolVar(
                f"full2d_pair_{pair_index}_first_before_second_horizontal"
            ),
            "second_before_first_horizontal": model.NewBoolVar(
                f"full2d_pair_{pair_index}_second_before_first_horizontal"
            ),
            "first_before_second_vertical": model.NewBoolVar(
                f"full2d_pair_{pair_index}_first_before_second_vertical"
            ),
            "second_before_first_vertical": model.NewBoolVar(
                f"full2d_pair_{pair_index}_second_before_first_vertical"
            ),
        }
        direction_literals.append((pair_index, tuple(
            (name, directions[name]) for name in direction_names
        )))
        model.Add(
            coordinate(second, 0)
            >= coordinate(first, 0) + int(node_size[first][0]) + clearance
        ).OnlyEnforceIf(directions["first_before_second_horizontal"])
        model.Add(
            coordinate(first, 0)
            >= coordinate(second, 0) + int(node_size[second][0]) + clearance
        ).OnlyEnforceIf(directions["second_before_first_horizontal"])
        model.Add(
            coordinate(second, 1)
            >= coordinate(first, 1) + int(node_size[first][1]) + clearance
        ).OnlyEnforceIf(directions["first_before_second_vertical"])
        model.Add(
            coordinate(first, 1)
            >= coordinate(second, 1) + int(node_size[second][1]) + clearance
        ).OnlyEnforceIf(directions["second_before_first_vertical"])
        model.AddBoolOr(tuple(directions.values()))
        if row.seed_hint:
            lower, upper, axis, _hint_id = row.seed_hint
            hint_key = (
                "first_before_second" if lower == first else "second_before_first"
            ) + f"_{axis}"
            for key, literal in directions.items():
                model.AddHint(literal, int(key == hint_key))
    if use_collision_cover:
        for group_index, group in enumerate(collision_cover["groups"]):
            first_owner, second_owner = group["owners"]
            delta_x = xs[second_owner] - xs[first_owner]
            delta_y = ys[second_owner] - ys[first_owner]
            seed_delta = (
                int(positions[second_owner][0]) - int(positions[first_owner][0]),
                int(positions[second_owner][1]) - int(positions[first_owner][1]),
            )
            for box_index, box in enumerate(group["forbidden_boxes_nm"]):
                literals = tuple(
                    model.NewBoolVar(
                        f"full2d_cover_{group_index}_{box_index}_{name}"
                    )
                    for name in direction_names
                )
                cover_direction_literals.append((group_index, box_index, literals))
                x0, y0, x1, y1 = box
                model.Add(delta_x >= x1).OnlyEnforceIf(literals[0])
                model.Add(delta_y >= y1).OnlyEnforceIf(literals[1])
                model.Add(delta_x <= x0 - 1).OnlyEnforceIf(literals[2])
                model.Add(delta_y <= y0 - 1).OnlyEnforceIf(literals[3])
                model.AddBoolOr(literals)
                seed_slacks = cover_directional_slacks(box, seed_delta)
                hint_index = next(
                    (index for index, slack in enumerate(seed_slacks) if slack >= 0),
                    None,
                )
                if hint_index is not None:
                    for index, literal in enumerate(literals):
                        model.AddHint(literal, int(index == hint_index))
    for axis, equalities in enumerate((graph.x_equalities, graph.y_equalities)):
        for first, second, delta_nm in equalities:
            model.Add(
                coordinate(second, axis)
                == coordinate(first, axis) + int(delta_nm)
            )

    _add_aggregate_envelope_order_constraints(
        model,
        xs,
        ys,
        dims,
        aggregate_envelope_orders,
        width_bound_nm=width_bound,
        height_bound_nm=height_bound,
        center_step_nm=grid,
    )
    _add_hierarchical_macro_constraints(
        model,
        xs,
        ys,
        dims,
        hierarchical_macro_orders,
        hierarchical_macro_alignments,
        width_bound_nm=width_bound,
        height_bound_nm=height_bound,
        center_step_nm=grid,
        gap_nm=int(minimum_clearance_nm),
        tolerance_scales_nm=hierarchical_alignment_tolerance_scales_nm,
    )
    _add_row_bounded_cavity_constraints(
        model,
        xs,
        ys,
        dims,
        row_bounded_cavity_projections,
        minimum_clearance_nm=minimum_clearance_nm,
    )
    add_mosaic_guest_slot_constraints(model, xs, ys, canonical_slot_projections)
    adjacency_direction_literals: list[
        Tuple[int, Tuple[Tuple[str, Any], ...]]
    ] = []
    _add_adjacency_constraints(
        model,
        xs,
        ys,
        dims,
        adjacency_limits_nm,
        scope_offsets_nm=adjacency_scope_offsets_nm,
        minimum_gap_nm=int(minimum_clearance_nm),
        projection_nm=grid,
        diagnostic_literal_sink=adjacency_direction_literals,
    )
    if aspect_obligation:
        _add_directional_aspect_constraints(
            model,
            xs=xs,
            ys=ys,
            dims=dims,
            width_bound=width_bound,
            height_bound=height_bound,
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
        )

    x_min = model.NewIntVar(0, width_bound, "full2d_x_min")
    x_max = model.NewIntVar(0, width_bound, "full2d_x_max")
    y_min = model.NewIntVar(0, height_bound, "full2d_y_min")
    y_max = model.NewIntVar(0, height_bound, "full2d_y_max")
    model.AddMinEquality(x_min, [xs[name] for name in ids])
    model.AddMaxEquality(
        x_max, [xs[name] + int(dims[name][0]) for name in ids]
    )
    model.AddMinEquality(y_min, [ys[name] for name in ids])
    model.AddMaxEquality(
        y_max, [ys[name] + int(dims[name][1]) for name in ids]
    )
    model.Add(x_min == 0)
    model.Add(y_min == 0)

    movement = []
    movement_bound = max(width_bound, height_bound)
    for name in ids:
        dx = model.NewIntVar(0, movement_bound, f"full2d_dx_{name}")
        dy = model.NewIntVar(0, movement_bound, f"full2d_dy_{name}")
        model.AddAbsEquality(dx, xs[name] - int(positions[name][0]))
        model.AddAbsEquality(dy, ys[name] - int(positions[name][1]))
        movement.extend((dx, dy))
        model.AddHint(xs[name], int(positions[name][0]))
        model.AddHint(ys[name], int(positions[name][1]))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_deterministic_time = 1.0
    movement_sum = model.NewIntVar(
        0,
        movement_bound * 2 * len(ids),
        "full2d_movement_sum",
    )
    model.Add(movement_sum == sum(movement))
    _require_edge_rail_reservation(edge_rail_reservation, dims=dims)
    if edge_rail_reservation is not None:
        edge_rail_reservation.add_constraints(model, xs=xs, ys=ys, dims=dims)
    _add_full2d_outer_hard_constraints(
        model, xs, ys, dims=dims, width_bound_nm=width_bound,
        height_bound_nm=height_bound, minimum_clearance_nm=minimum_clearance_nm,
        order_pairs=order_pairs, vertical_order_pairs=vertical_order_pairs,
        pair_clearances_nm=pair_clearances_nm,
        hierarchical_cohesion_span_caps_nm=hierarchical_cohesion_span_caps_nm,
        floorplan_column_groups=floorplan_column_groups, left_edge=left_edge,
    )
    objectives = (x_max, y_max, movement_sum)
    objective_names = ("x_max", "y_max", "movement_sum")
    literal_by_pair = dict(direction_literals)
    evidence_graph_sha256 = (
        collision_cover["graph_sha256"] if solve_evidence_sink is not None else None
    )

    def capture_solve_evidence(
        *,
        solve_index: int,
        status: int,
        solution: Optional[Mapping[str, Tuple[int, int]]],
    ) -> None:
        if solve_evidence_sink is None:
            return
        node_points = None
        if solution is not None:
            node_points = {
                node: (
                    int(solution[node_owner[node]][0]) + int(node_offset[node][0]),
                    int(solution[node_owner[node]][1]) + int(node_offset[node][1]),
                )
                for node in graph.nodes
            }
        rows = []
        for pair_index, row in enumerate(graph.pair_constraints):
            if row.mode == "fixed_edge":
                selected_literal_mask = -1
                if node_points is None:
                    active_status = "UNAVAILABLE_NO_INCUMBENT"
                    slacks_nm: Tuple[int, ...] = ()
                else:
                    axis_index = 0 if row.axis == "horizontal" else 1
                    slack = (
                        node_points[str(row.upper)][axis_index]
                        - node_points[str(row.lower)][axis_index]
                        - int(node_size[str(row.lower)][axis_index])
                        - int(row.clearance_nm)
                    )
                    slacks_nm = (int(slack),)
                    active_status = (
                        "ACTIVE_ZERO_SLACK" if slack == 0
                        else "SATISFIED_POSITIVE_SLACK"
                    )
            else:
                handles = () if use_collision_cover else literal_by_pair[pair_index]
                if solution is None:
                    selected_literal_mask = -3 if use_collision_cover else -2
                    active_status = "UNAVAILABLE_NO_INCUMBENT"
                    slacks_nm = ()
                else:
                    literal_values = tuple(
                        (name, bool(solver.Value(literal)))
                        for name, literal in handles
                    )
                    selected_literal_mask = -3 if use_collision_cover else sum(
                        1 << index
                        for index, (_name, value) in enumerate(literal_values)
                        if value
                    )
                    first_x, first_y = node_points[row.first]
                    second_x, second_y = node_points[row.second]
                    first_w, first_h = node_size[row.first]
                    second_w, second_h = node_size[row.second]
                    directional_slacks = {
                        "first_before_second_horizontal": (
                            second_x - first_x - int(first_w) - int(row.clearance_nm)
                        ),
                        "first_before_second_vertical": (
                            second_y - first_y - int(first_h) - int(row.clearance_nm)
                        ),
                        "second_before_first_horizontal": (
                            first_x - second_x - int(second_w) - int(row.clearance_nm)
                        ),
                        "second_before_first_vertical": (
                            first_y - second_y - int(second_h) - int(row.clearance_nm)
                        ),
                    }
                    slacks_nm = tuple(int(directional_slacks[name]) for name in direction_names)
                    if use_collision_cover:
                        if not any(slack >= 0 for slack in slacks_nm):
                            raise ValueError(
                                "full-2D cover incumbent violates an original physical row"
                            )
                        active_status = (
                            "ACTIVE_ZERO_SLACK"
                            if max(slacks_nm) == 0
                            else "SATISFIED_POSITIVE_SLACK"
                        )
                    else:
                        active_status = (
                            "ACTIVE_ZERO_SLACK"
                            if any(
                                directional_slacks[name] == 0 and value
                                for name, value in literal_values
                            )
                            else "ENFORCED_POSITIVE_SLACK"
                        )
            active_status_code = {
                "UNAVAILABLE_NO_INCUMBENT": 0,
                "ACTIVE_ZERO_SLACK": 1,
                "SATISFIED_POSITIVE_SLACK": 2,
                "ENFORCED_POSITIVE_SLACK": 3,
            }[active_status]
            rows.append((
                pair_index,
                active_status_code,
                selected_literal_mask,
                slacks_nm,
                0,
            ))
        positions_sha256 = None
        if solution is not None:
            positions_sha256 = hashlib.sha256(json.dumps(
                tuple(sorted(solution.items())),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")).hexdigest()
        from clay.layout.place.full2d_nonpair_registry import (
            build_full2d_nonpair_constraint_registry,
        )

        selected_adjacency_directions = None
        if solution is not None:
            selected_adjacency_directions = {}
            for index, handles in adjacency_direction_literals:
                selected_names = tuple(
                    name for name, literal in handles
                    if bool(solver.Value(literal))
                )
                if len(selected_names) != 1:
                    raise ValueError(
                        "full-2D adjacency evidence requires one selected literal: "
                        f"adjacency:{index}"
                    )
                selected_adjacency_directions[index] = selected_names[0]
        nonpair_registry = build_full2d_nonpair_constraint_registry(
            positions=solution,
            dims=dims,
            axis_alignment_groups=axis_alignment_groups,
            center_alignment_pairs=center_alignment_pairs,
            aggregate_envelope_orders=aggregate_envelope_orders,
            hierarchical_macro_orders=hierarchical_macro_orders,
            hierarchical_macro_alignments=hierarchical_macro_alignments,
            row_bounded_cavity_projections=row_bounded_cavity_projections,
            **({"mosaic_guest_slot_projections": canonical_slot_projections}
               if canonical_slot_projections else {}),
            hierarchical_alignment_tolerance_scales_nm=(
                hierarchical_alignment_tolerance_scales_nm
            ),
            adjacency_limits_nm=adjacency_limits_nm,
            adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
            selected_adjacency_directions=selected_adjacency_directions,
            minimum_gap_nm=int(minimum_clearance_nm),
            projection_nm=grid,
            center_step_nm=grid,
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
        )
        evidence = {
            "active_status_codes": (
                (0, "UNAVAILABLE_NO_INCUMBENT"),
                (1, "ACTIVE_ZERO_SLACK"),
                (2, "SATISFIED_POSITIVE_SLACK"),
                (3, "ENFORCED_POSITIVE_SLACK"),
            ),
            "basis_summary": {
                "reason": "CP-SAT exposes no LP basis or constraint duals",
                "status": "UNAVAILABLE_CP_SAT_NO_BASIS",
            },
            "candidate_positions_sha256": positions_sha256,
            "constraint_row_schema": (
                "pair_constraint_index",
                "active_status_code",
                (
                    "solver_literal_mask_or_replay_sentinel"
                    if use_collision_cover else "selected_literal_mask"
                ),
                "directional_slacks_nm",
                "necessity_status_code",
            ),
            "constraint_scope": "full2d_graph_pair_constraints",
            "constraints": tuple(rows),
            "direction_order": direction_names,
            "graph_sha256": evidence_graph_sha256,
            "necessity_status_codes": (
                (0, "UNMEASURABLE_SINGLE_INCUMBENT"),
            ),
            "nonpair_constraint_registry": nonpair_registry,
            "objective_name": objective_names[solve_index],
            "schema": (
                "clay.full2d-solve-evidence/v3"
                if use_collision_cover else "clay.full2d-solve-evidence/v2"
            ),
            "solve_index": solve_index,
            "solver_status": solver.StatusName(status),
        }
        if use_collision_cover:
            cover_rows = []
            for group_index, box_index, literals in cover_direction_literals:
                if solution is None:
                    mask = -2
                    cover_slacks = ()
                else:
                    group = collision_cover["groups"][group_index]
                    first_owner, second_owner = group["owners"]
                    delta = (
                        int(solution[second_owner][0]) - int(solution[first_owner][0]),
                        int(solution[second_owner][1]) - int(solution[first_owner][1]),
                    )
                    cover_slacks = cover_directional_slacks(
                        group["forbidden_boxes_nm"][box_index], delta
                    )
                    mask = sum(
                        1 << index for index, literal in enumerate(literals)
                        if bool(solver.Value(literal))
                    )
                    if not 1 <= mask <= 15 or any(
                        slack < 0 and mask & (1 << index)
                        for index, slack in enumerate(cover_slacks)
                    ):
                        raise ValueError(
                            "full-2D cover evidence disagrees with selected solver literals"
                        )
                cover_rows.append((group_index, box_index, mask, cover_slacks))
            evidence.update({
                "collision_encoding": collision_cover,
                "cover_constraints": tuple(cover_rows),
                "candidate_positions_nm": (
                    tuple(sorted(solution.items())) if solution is not None else None
                ),
            })
        solve_evidence_sink.append(evidence)

    last_solution = None
    for index, objective in enumerate(objectives):
        model.Minimize(objective)
        status = solver.Solve(model)
        if status == cp_model.INFEASIBLE:
            capture_solve_evidence(
                solve_index=index,
                status=status,
                solution=None,
            )
            raise ValueError(
                "full-2D aggregate target contradicts its replay-legal seed"
            )
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            capture_solve_evidence(
                solve_index=index,
                status=status,
                solution=None,
            )
            return last_solution
        last_solution = {
            name: (int(solver.Value(xs[name])), int(solver.Value(ys[name])))
            for name in ids
        }
        capture_solve_evidence(
            solve_index=index,
            status=status,
            solution=last_solution,
        )
        if index + 1 < len(objectives):
            model.Add(objective == int(solver.Value(objective)))
    return last_solution

def _compact_full2d_master_positions_impl(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    minimum_clearance_nm: int,
    manufacturing_grid_nm: int,
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]] = (),
    center_alignment_pairs: Sequence[Tuple[str, str, str]] = (),
    pair_axis_authority: Tuple[
        Tuple[str, str, str, str, str, Tuple[str, ...]], ...
    ] = (),
    pair_axis_preference: Tuple[Tuple[str, str, str], ...] = (),
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder] = (),
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder] = (),
    hierarchical_macro_alignments: Sequence[_HierarchicalMacroAlignment] = (),
    hierarchical_alignment_tolerance_scales_nm: (
        _HierarchicalAlignmentToleranceScalesNm
    ) = (),
    adjacency_limits_nm: Sequence[Tuple[str, str, int]] = (),
    adjacency_scope_offsets_nm: Mapping[str, _Bbox] = MappingProxyType({}),
    aspect_target: float = 1.0,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = False,
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    row_bounded_cavity_projections: Tuple[_RowBoundedCavityProjection, ...] = (),
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = (),
    edge_rail_reservation: Any = None,
    order_pairs: Sequence[Tuple[str, str]] = (),
    vertical_order_pairs: Sequence[Tuple[str, str]] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
    hierarchical_cohesion_span_caps_nm: Tuple[Tuple[Tuple[str, ...], int, int], ...] = (),
    floorplan_column_groups: Tuple[_FloorplanColumnGroup, ...] = (),
    left_edge: Sequence[str] = (),
    legal: Callable[[Mapping[str, Tuple[int, int]]], bool],
    legal_blockers: Optional[
        Callable[[Mapping[str, Tuple[int, int]]], Sequence[str]]
    ] = None,
    legal_replay_audit: Optional[Callable[[], Mapping[str, object]]] = None,
    solve_evidence_sink: Optional[list[Mapping[str, object]]] = None,
    _diagnostic_state: Dict[str, object],
):
    """Press exact masters through a complete H/V graph, never a row surrogate."""

    from clay.layout.extract.compaction import (
        compact_full2d_master_placement,
        full2d_graph_from_seed,
    )

    _diagnostic_state["failure_stage"] = "input_validation"
    grid = int(manufacturing_grid_nm)
    if grid <= 0:
        raise ValueError("full-2D master compaction requires a positive grid")
    result = {
        str(name): (int(point[0]), int(point[1]))
        for name, point in positions.items()
    }
    _diagnostic_state["seed_positions_nm"] = tuple(sorted(result.items()))
    _diagnostic_state["candidate_positions_nm"] = tuple(sorted(result.items()))
    _diagnostic_state["dimensions_nm"] = tuple(sorted(
        (
            str(name),
            (int(size[0]), int(size[1])),
        )
        for name, size in dims.items()
    ))
    if set(result) != set(dims):
        raise ValueError(
            "full-2D master compaction requires one replay-legal placement"
        )
    canonical_slot_projections = _canonical_mosaic_guest_slot_projections(
        mosaic_guest_slot_projections, dims=dims, minimum_clearance_nm=minimum_clearance_nm,
    )

    def slot_failure_rows(candidate):
        return tuple(
            _full_hard_replay_row(
                f"mosaic_guest_slot:{row.obligation_id}", "mosaic_guest_slot",
                source_constraint_ids=(row.obligation_id,),
            )
            for row in canonical_slot_projections
            if not mosaic_guest_slot_satisfied(candidate, row)
        )

    def capture_full_hard_audit(
        candidate: Mapping[str, Tuple[int, int]],
        *,
        failed: bool,
    ) -> None:
        try:
            if legal_replay_audit is not None:
                audit = legal_replay_audit()
                if not isinstance(audit, Mapping) or set(audit) != {
                    "failed_rows",
                    "inventory",
                    "spec",
                }:
                    raise ValueError("full-hard replay audit schema differs")
                raw_rows = audit["failed_rows"]
                _diagnostic_state["full_hard_inventory"] = (
                    _canonical_full_hard_failure_rows(audit["inventory"])
                )
                _diagnostic_state["full_hard_spec"] = audit["spec"]
            elif failed and legal_blockers is not None:
                raw_rows = legal_blockers(candidate)
            else:
                raw_rows = ()
            if canonical_slot_projections:
                raw_rows = _canonical_full_hard_failure_rows(raw_rows)
                existing_ids = {str(row["row_id"]) for row in raw_rows}
                raw_rows = tuple(raw_rows) + tuple(
                    row for row in slot_failure_rows(candidate) if row["row_id"] not in existing_ids
                )
            _diagnostic_state["full_hard_failed_rows"] = (
                _canonical_full_hard_failure_rows(raw_rows)
            )
        except ExecutionStopped:
            raise
        except Exception:  # noqa: BLE001 - audit boundary
            _diagnostic_state["full_hard_failed_rows"] = ({
                "constraint_names": ("diagnostic_contract_gap",),
                "row_id": "full_hard_diagnostic:audit_collection_error",
                "source_constraint_ids": (),
            },)

    _diagnostic_state["failure_stage"] = "full_hard_seed_replay"
    try:
        _require_edge_rail_reservation(edge_rail_reservation, dims=dims)
        seed_is_legal = (
            legal(result) and not slot_failure_rows(result)
            and (edge_rail_reservation is None
                 or edge_rail_reservation.replay(dims=dims, positions=result))
        )
    except ExecutionStopped:
        raise
    except Exception:  # noqa: BLE001 - preserved by the typed outer boundary
        _diagnostic_state["full_hard_status"] = "ERROR"
        capture_full_hard_audit(result, failed=True)
        raise
    _diagnostic_state["full_hard_status"] = (
        "PASS" if seed_is_legal else "FAIL"
    )
    capture_full_hard_audit(result, failed=not seed_is_legal)
    if not seed_is_legal:
        raise ValueError(
            "full-2D master compaction requires one replay-legal placement"
        )
    _diagnostic_state["failure_stage"] = "canonicalization"
    canonical_axis_groups = _canonical_axis_alignment_groups(
        axis_alignment_groups,
        block_ids=tuple(sorted(result)),
        dims=dims,
    )
    canonical_center_pairs = _canonical_center_alignment_pairs(
        center_alignment_pairs,
        block_ids=tuple(sorted(result)),
        dims=dims,
    )
    canonical_aggregate_orders = _canonical_aggregate_envelope_orders(
        tuple(aggregate_envelope_orders),
        block_ids=tuple(sorted(result)),
    )
    canonical_hierarchical_orders = _canonical_hierarchical_macro_orders(
        tuple(hierarchical_macro_orders),
        block_ids=tuple(sorted(result)),
    )
    canonical_cavity_projections = _canonical_row_bounded_cavity_projections(
        row_bounded_cavity_projections,
        dims=dims,
    )
    canonical_pair_axis_authority = _hierarchical_full2d_pair_axis_authority(
        canonical_hierarchical_orders
    )
    canonical_pair_axis_preference = _hierarchical_full2d_pair_axis_preference(
        canonical_hierarchical_orders
    )
    if tuple(pair_axis_authority) != canonical_pair_axis_authority:
        raise ValueError(
            "full-2D pair axis authority diverges from canonical declaration inventory"
        )
    if tuple(pair_axis_preference) != canonical_pair_axis_preference:
        raise ValueError(
            "full-2D pair axis preference diverges from canonical declaration inventory"
        )
    pair_axis_authority = canonical_pair_axis_authority
    pair_axis_preference = canonical_pair_axis_preference
    canonical_hierarchical_alignments = _canonical_hierarchical_macro_alignments(
        tuple(hierarchical_macro_alignments),
        block_ids=tuple(sorted(result)),
    )
    # The full source-group FEOL proof supplies the physical constraints for
    # these pairs. Their source order remains a separate hard hierarchical
    # constraint and must not be lowered as a second atom-pair collision edge.
    from clay.layout.place.native_feol_reservation import sky130_group_spacing_master_pairs
    proven_group_pairs = set(sky130_group_spacing_master_pairs(edge_rail_reservation))
    source_ordered_group_pairs = set(overlap_compatible_pairs) & proven_group_pairs
    if source_ordered_group_pairs:
        _assert_hierarchical_macro_positions(result, dims, canonical_hierarchical_orders,
            canonical_hierarchical_alignments, gap_nm=int(minimum_clearance_nm),
            tolerance_scales_nm=hierarchical_alignment_tolerance_scales_nm)
    graph_pair_axis_authority = tuple(row for row in pair_axis_authority
                                      if row[:2] not in source_ordered_group_pairs)
    hierarchical_alignment_tolerance_scales_nm = (
        _require_hierarchical_alignment_tolerance_scales_nm(
            hierarchical_alignment_tolerance_scales_nm,
            alignments=canonical_hierarchical_alignments,
        )
    )

    def canonical_equality(
        first: str,
        second: str,
        delta_nm: int,
    ) -> Tuple[str, str, int]:
        if first > second:
            first, second, delta_nm = second, first, -delta_nm
        if int(delta_nm) % grid:
            raise ValueError("full-2D alignment equality escapes the manufacturing grid")
        return first, second, int(delta_nm)

    x_equalities = []
    y_equalities = []
    for group in canonical_axis_groups:
        first, first_offset2 = group[0]
        for second, second_offset2 in group[1:]:
            delta2 = int(first_offset2) - int(second_offset2)
            if delta2 & 1:
                raise ValueError("full-2D axis equality requires one integer-nm offset")
            x_equalities.append(
                canonical_equality(first, second, delta2 // 2)
            )
    for first, second, axis in canonical_center_pairs:
        coordinate = 0 if axis == "horizontal" else 1
        delta2 = int(dims[first][coordinate]) - int(dims[second][coordinate])
        if delta2 & 1:
            raise ValueError("full-2D center equality requires one integer-nm offset")
        equality = canonical_equality(first, second, delta2 // 2)
        (x_equalities if coordinate == 0 else y_equalities).append(equality)
    x_equalities = tuple(sorted(set(x_equalities)))
    y_equalities = tuple(sorted(set(y_equalities)))
    if len(x_equalities) != len(set(x_equalities)) or len(y_equalities) != len(
        set(y_equalities)
    ):
        raise ValueError("full-2D alignment equalities must be unique")

    def boxes(candidate: Mapping[str, Tuple[int, int]]) -> Dict[str, _Bbox]:
        return {
            name: (
                point[0],
                point[1],
                point[0] + int(dims[name][0]),
                point[1] + int(dims[name][1]),
            )
            for name, point in sorted(candidate.items())
        }

    _diagnostic_state["failure_stage"] = "graph_build"
    indexed_geometry = indexed_incumbent_geometry_enabled()
    graph_factory = full2d_graph_from_seed
    if indexed_geometry:
        from clay.layout.extract.fixed_incumbent_geometry import fixed_incumbent_geometry_from_seed
        graph_factory = fixed_incumbent_geometry_from_seed
    def geometry_phase(name):
        session = current_session() if indexed_geometry else None
        return session.phase('placement.fixed_incumbent_' + name) if session else nullcontext()
    with geometry_phase('geometry_build'):
        graph = graph_factory(
            boxes(result), minimum_clearance_nm=int(minimum_clearance_nm), grid_nm=grid,
            x_equalities=x_equalities, y_equalities=y_equalities,
            pair_axis_authority=graph_pair_axis_authority, pair_axis_preference=pair_axis_preference,
            overlap_compatible_pairs=overlap_compatible_pairs,
            footprints_nm=rigid_compound_footprints,
            layered_footprints_nm=layered_collision_footprints,
        )
    _diagnostic_state["graph"] = graph
    _diagnostic_state["failure_stage"] = "replay"

    def replay_legal(candidate: Mapping[str, Tuple[int, int]], *, collect_audit: bool = True) -> bool:
        if source_ordered_group_pairs:
            try:
                _assert_hierarchical_macro_positions(candidate, dims, canonical_hierarchical_orders,
                    canonical_hierarchical_alignments, gap_nm=int(minimum_clearance_nm),
                    tolerance_scales_nm=hierarchical_alignment_tolerance_scales_nm)
            except (ValueError, RuntimeError):
                return False
        if not collect_audit:
            try:
                if not graph.replay(boxes(candidate)):
                    return False
                boolean_legal = getattr(legal, 'boolean_replay', legal)
                return (boolean_legal(candidate) and not slot_failure_rows(candidate)
                        and (edge_rail_reservation is None
                             or edge_rail_reservation.replay(dims=dims, positions=candidate)))
            except ExecutionStopped:
                raise
            except Exception:
                # Replay the exceptional operand through the original complete
                # diagnostic boundary, rather than retaining an unrelated probe.
                return replay_legal(candidate, collect_audit=True)
        candidate_rows = tuple(sorted(
            (str(name), (int(point[0]), int(point[1])))
            for name, point in candidate.items()
        ))
        seed_names = {
            name for name, _point in _diagnostic_state["seed_positions_nm"]
        }
        if {name for name, _point in candidate_rows} == seed_names:
            _diagnostic_state["candidate_positions_nm"] = candidate_rows
        else:
            _diagnostic_state["rejected_candidate_positions_nm"] = candidate_rows
        candidate_boxes = boxes(candidate)
        with geometry_phase('geometry_replay'):
            geometry_is_legal = graph.replay(candidate_boxes)
        if not geometry_is_legal:
            _diagnostic_state["graph_failed_rows"] = graph.replay_diagnostic(
                candidate_boxes
            )
            _diagnostic_state["graph_status"] = "FAIL"
            _diagnostic_state["full_hard_status"] = "NOT_RUN"
            _diagnostic_state["full_hard_failed_rows"] = ()
            return False
        _diagnostic_state["graph_failed_rows"] = ()
        _diagnostic_state["graph_status"] = "PASS"
        try:
            with geometry_phase('full_hard_replay'):
                hard_is_legal = (
                    legal(candidate) and not slot_failure_rows(candidate)
                    and (edge_rail_reservation is None
                         or edge_rail_reservation.replay(dims=dims, positions=candidate))
                )
        except ExecutionStopped:
            raise
        except Exception:  # noqa: BLE001 - preserved by the typed outer boundary
            _diagnostic_state["full_hard_status"] = "ERROR"
            capture_full_hard_audit(candidate, failed=True)
            raise
        _diagnostic_state["full_hard_status"] = (
            "PASS" if hard_is_legal else "FAIL"
        )
        capture_full_hard_audit(candidate, failed=not hard_is_legal)
        return hard_is_legal

    if incumbent_replay_enabled():
        if not replay_legal(result):
            raise ValueError("incumbent replay failed complete graph or full hard replay")
        return result, graph, False

    has_live_pairs = any(
        row.mode == "four_way_disjunction" for row in graph.pair_constraints
    )
    target = None
    if not has_live_pairs:
        target_boxes = compact_full2d_master_placement(boxes(result), graph)
        active_x = bool(graph.horizontal_edges or graph.x_equalities)
        active_y = bool(graph.vertical_edges or graph.y_equalities)
        target = {
            name: (
                bbox[0] if active_x else result[name][0],
                bbox[1] if active_y else result[name][1],
            )
            for name, bbox in target_boxes.items()
        }
        if target != result and replay_legal(target):
            return target, graph, True

    aggregate_target = _compact_full2d_aggregate_target(
        result,
        dims=dims,
        graph=graph,
        order_pairs=order_pairs,
        vertical_order_pairs=vertical_order_pairs,
        pair_clearances_nm=pair_clearances_nm,
        hierarchical_cohesion_span_caps_nm=hierarchical_cohesion_span_caps_nm,
        floorplan_column_groups=floorplan_column_groups,
        left_edge=left_edge,
        minimum_clearance_nm=int(minimum_clearance_nm),
        manufacturing_grid_nm=grid,
        axis_alignment_groups=canonical_axis_groups,
        center_alignment_pairs=canonical_center_pairs,
        aggregate_envelope_orders=canonical_aggregate_orders,
        hierarchical_macro_orders=canonical_hierarchical_orders,
        hierarchical_macro_alignments=canonical_hierarchical_alignments,
        row_bounded_cavity_projections=canonical_cavity_projections,
        mosaic_guest_slot_projections=canonical_slot_projections,
        edge_rail_reservation=edge_rail_reservation,
        hierarchical_alignment_tolerance_scales_nm=(
            hierarchical_alignment_tolerance_scales_nm
        ),
        adjacency_limits_nm=adjacency_limits_nm,
        adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
        aspect_target=aspect_target,
        aspect_bounds=aspect_bounds,
        aspect_obligation=aspect_obligation,
        solve_evidence_sink=solve_evidence_sink,
    )
    if aggregate_target is not None and replay_legal(aggregate_target):
        return aggregate_target, graph, aggregate_target != result
    if aggregate_target is not None and legal_blockers is not None:
        print(
            "FULL2D_AGGREGATE_TARGET_REJECTED "
            f"blockers={tuple(legal_blockers(aggregate_target))!r}"
        )
    if target is None:
        target = aggregate_target if aggregate_target is not None else dict(result)

    changed = False
    iteration_limit = max(4, 4 * len(result))

    def equality_components(axis: int) -> Tuple[Tuple[str, ...], ...]:
        parent = {name: name for name in result}
        owner_by_node = {
            node: owner
            for node, owner, _offset in graph.node_owner_offsets_nm
        }

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        equalities = graph.x_equalities if axis == 0 else graph.y_equalities
        for first, second, _delta in equalities:
            first_owner = owner_by_node[first]
            second_owner = owner_by_node[second]
            if first_owner != second_owner:
                union(first_owner, second_owner)
        groups: Dict[str, List[str]] = {}
        for name in sorted(result):
            groups.setdefault(find(name), []).append(name)
        return tuple(tuple(members) for _root, members in sorted(groups.items()))

    for _iteration in range(iteration_limit):
        iteration_changed = False
        # Y then X is the established alternating compaction order. Each pass
        # moves the low-side predecessor first, so downstream DAG nodes can
        # follow without crossing or changing their assigned separation axis.
        for axis in (1, 0):
            components = sorted(
                equality_components(axis),
                key=lambda members: (
                    min(result[name][axis] for name in members),
                    members,
                ),
            )
            for members in components:
                deltas = {
                    result[name][axis] - target[name][axis] for name in members
                }
                if len(deltas) != 1:
                    raise ValueError(
                        "full-2D equality component has inconsistent target motion"
                    )
                delta = next(iter(deltas))
                if delta <= 0:
                    continue
                available_steps = delta // grid
                low = 0
                high = available_steps
                while low < high:
                    middle = (low + high + 1) // 2
                    candidate = dict(result)
                    for name in members:
                        point = list(result[name])
                        point[axis] -= middle * grid
                        candidate[name] = (point[0], point[1])
                    if replay_legal(candidate, collect_audit=False):
                        low = middle
                    else:
                        high = middle - 1
                if low:
                    for name in members:
                        point = list(result[name])
                        point[axis] -= low * grid
                        result[name] = (point[0], point[1])
                    iteration_changed = True
                    changed = True
        if not iteration_changed:
            break
    if not replay_legal(result):
        raise ValueError("full-2D master compaction lost graph or hard replay")
    return result, graph, changed

def _compact_full2d_master_positions(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    minimum_clearance_nm: int,
    manufacturing_grid_nm: int,
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]] = (),
    center_alignment_pairs: Sequence[Tuple[str, str, str]] = (),
    pair_axis_authority: Tuple[
        Tuple[str, str, str, str, str, Tuple[str, ...]], ...
    ] = (),
    pair_axis_preference: Tuple[Tuple[str, str, str], ...] = (),
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder] = (),
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder] = (),
    hierarchical_macro_alignments: Sequence[_HierarchicalMacroAlignment] = (),
    hierarchical_alignment_tolerance_scales_nm: (
        _HierarchicalAlignmentToleranceScalesNm
    ) = (),
    adjacency_limits_nm: Sequence[Tuple[str, str, int]] = (),
    adjacency_scope_offsets_nm: Mapping[str, _Bbox] = MappingProxyType({}),
    aspect_target: float = 1.0,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = False,
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    row_bounded_cavity_projections: Tuple[_RowBoundedCavityProjection, ...] = (),
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = (),
    edge_rail_reservation: Any = None,
    order_pairs: Sequence[Tuple[str, str]] = (),
    vertical_order_pairs: Sequence[Tuple[str, str]] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
    hierarchical_cohesion_span_caps_nm: Tuple[Tuple[Tuple[str, ...], int, int], ...] = (),
    floorplan_column_groups: Tuple[_FloorplanColumnGroup, ...] = (),
    left_edge: Sequence[str] = (),
    legal: Callable[[Mapping[str, Tuple[int, int]]], bool],
    legal_blockers: Optional[
        Callable[[Mapping[str, Tuple[int, int]]], Sequence[str]]
    ] = None,
    legal_replay_audit: Optional[Callable[[], Mapping[str, object]]] = None,
    solve_evidence_sink: Optional[list[Mapping[str, object]]] = None,
):
    def captured_int(value: object) -> object:
        try:
            return int(value)
        except Exception:  # noqa: BLE001 - diagnostic input boundary
            return _diagnostic_safe_canonical_value(value)

    def captured_float(value: object) -> object:
        try:
            return float(value)
        except Exception:  # noqa: BLE001 - diagnostic input boundary
            return _diagnostic_safe_canonical_value(value)

    def stage_input_payload() -> Dict[str, object]:
        return {
            "algorithm_contract": {
                "schema": "clay.full2d-compaction-algorithm/v1",
                "axis_pass_order": () if incumbent_replay_enabled() else ("vertical", "horizontal"),
                "component_search": "NOT_RUN" if incumbent_replay_enabled() else "deterministic_binary_search",
                "graph_replay": "fail_closed",
                "aggregate_solver": {"status": "NOT_RUN"} if incumbent_replay_enabled() else {
                    "num_search_workers": 1,
                    "random_seed": 0,
                    "max_deterministic_time": 1.0,
                    "objective_order": ("x_max", "y_max", "movement_sum"),
                },
                **({"execution_mode": "supplied_incumbent_feasibility"}
                   if incumbent_replay_enabled() else {}),
            },
            "minimum_clearance_nm": captured_int(minimum_clearance_nm),
            "manufacturing_grid_nm": captured_int(manufacturing_grid_nm),
            "axis_alignment_groups": _diagnostic_safe_canonical_value(
                axis_alignment_groups
            ),
            "center_alignment_pairs": _diagnostic_safe_canonical_value(
                center_alignment_pairs
            ),
            "pair_axis_authority": _diagnostic_safe_canonical_value(
                pair_axis_authority
            ),
            "pair_axis_preference": _diagnostic_safe_canonical_value(
                pair_axis_preference
            ),
            "aggregate_envelope_orders": _diagnostic_safe_canonical_value(
                aggregate_envelope_orders
            ),
            "hierarchical_macro_orders": _diagnostic_safe_canonical_value(
                hierarchical_macro_orders
            ),
            "hierarchical_macro_alignments": _diagnostic_safe_canonical_value(
                hierarchical_macro_alignments
            ),
            "hierarchical_alignment_tolerance_scales_nm": (
                _diagnostic_safe_canonical_value(
                    hierarchical_alignment_tolerance_scales_nm
                )
            ),
            "adjacency_limits_nm": _diagnostic_safe_canonical_value(
                adjacency_limits_nm
            ),
            "adjacency_scope_offsets_nm": _diagnostic_safe_canonical_value(
                adjacency_scope_offsets_nm
            ),
            "aspect_target": captured_float(aspect_target),
            "aspect_bounds": _diagnostic_safe_canonical_value(aspect_bounds),
            "aspect_obligation": bool(aspect_obligation),
            "overlap_compatible_pairs": _diagnostic_safe_canonical_value(
                overlap_compatible_pairs
            ),
            "rigid_compound_footprints": _diagnostic_safe_canonical_value(
                rigid_compound_footprints
            ),
            "layered_collision_footprints": _diagnostic_safe_canonical_value(
                layered_collision_footprints
            ),
            "row_bounded_cavity_projections": _diagnostic_safe_canonical_value(
                tuple(row.canonical_data() for row in row_bounded_cavity_projections)
            ),
            **({"mosaic_guest_slot_projections": _diagnostic_safe_canonical_value(
                tuple(row.canonical_data() for row in mosaic_guest_slot_projections))}
               if mosaic_guest_slot_projections else {}),
            **({"edge_rail_reservation": edge_rail_reservation.canonical_data()}
               if edge_rail_reservation is not None else {}),
            **{name: _diagnostic_safe_canonical_value(value) for name, value in (
                ("order_pairs", order_pairs),
                ("vertical_order_pairs", vertical_order_pairs),
                ("pair_clearances_nm", pair_clearances_nm),
                ("hierarchical_cohesion_span_caps_nm", hierarchical_cohesion_span_caps_nm),
                ("floorplan_column_groups", floorplan_column_groups),
                ("left_edge", left_edge),
            ) if value},
        }
    diagnostic_state: Dict[str, object] = {
        "seed_positions_nm": (),
        "candidate_positions_nm": (),
        "dimensions_nm": (),
        "failure_stage": "input_capture",
        "full_hard_status": "NOT_RUN",
        "solver_evidence": (),
    }
    captured_solver_evidence: list[Mapping[str, object]] = []
    if solve_evidence_sink is not None:
        if type(solve_evidence_sink) is not list:
            raise TypeError("full-2D solve evidence sink must be one exact list")
        captured_solver_evidence = solve_evidence_sink
    diagnostic_state["solver_evidence"] = captured_solver_evidence
    try:
        seed_rows = _full2d_compaction_position_rows(positions)
        diagnostic_state.update({
            "seed_positions_nm": seed_rows,
            "candidate_positions_nm": seed_rows,
            "dimensions_nm": _full2d_compaction_position_rows(dims),
            "input_provenance": _full2d_input_provenance(
                positions,
                dims=dims,
                minimum_clearance_nm=minimum_clearance_nm,
                pair_axis_authority=pair_axis_authority,
                pair_axis_preference=pair_axis_preference,
                overlap_compatible_pairs=overlap_compatible_pairs,
            ),
            "stage_input": stage_input_payload(),
        })
        return _compact_full2d_master_positions_impl(
            positions,
            dims=dims,
            minimum_clearance_nm=minimum_clearance_nm,
            manufacturing_grid_nm=manufacturing_grid_nm,
            axis_alignment_groups=axis_alignment_groups,
            center_alignment_pairs=center_alignment_pairs,
            pair_axis_authority=pair_axis_authority,
            pair_axis_preference=pair_axis_preference,
            aggregate_envelope_orders=aggregate_envelope_orders,
            hierarchical_macro_orders=hierarchical_macro_orders,
            hierarchical_macro_alignments=hierarchical_macro_alignments,
            hierarchical_alignment_tolerance_scales_nm=(
                hierarchical_alignment_tolerance_scales_nm
            ),
            adjacency_limits_nm=adjacency_limits_nm,
            adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
            overlap_compatible_pairs=overlap_compatible_pairs,
            rigid_compound_footprints=rigid_compound_footprints,
            layered_collision_footprints=layered_collision_footprints,
            row_bounded_cavity_projections=row_bounded_cavity_projections,
            mosaic_guest_slot_projections=mosaic_guest_slot_projections,
            edge_rail_reservation=edge_rail_reservation,
            order_pairs=order_pairs,
            vertical_order_pairs=vertical_order_pairs,
            pair_clearances_nm=pair_clearances_nm,
            hierarchical_cohesion_span_caps_nm=hierarchical_cohesion_span_caps_nm,
            floorplan_column_groups=floorplan_column_groups,
            left_edge=left_edge,
            legal=legal,
            legal_blockers=legal_blockers,
            legal_replay_audit=legal_replay_audit,
            solve_evidence_sink=captured_solver_evidence,
            _diagnostic_state=diagnostic_state,
        )
    except (ExecutionStopped, Full2DCompactionDiagnosticError):
        raise
    except Exception as error:  # noqa: BLE001 - named fail-closed boundary
        diagnostic = _full2d_compaction_failure_diagnostic(
            error=error,
            state=diagnostic_state,
        )
        raise Full2DCompactionDiagnosticError(
            str(error) or type(error).__name__,
            diagnostic=diagnostic,
        ) from error

def _compaction_replay_width_bound(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    configured_width_bound_nm: int,
    strict_width_cap: bool,
    exact_outline_width: bool,
) -> int:
    """Bind edge anchors to the moving outline unless width is explicitly fixed."""

    configured = int(configured_width_bound_nm)
    if configured <= 0 or set(positions) != set(dims):
        raise ValueError("compaction replay width authority is malformed")
    if type(strict_width_cap) is not bool or type(exact_outline_width) is not bool:
        raise TypeError("compaction replay width modes must be exact booleans")
    if exact_outline_width and not strict_width_cap:
        raise ValueError("exact compaction outline requires one strict width cap")
    if strict_width_cap:
        return configured
    return max(
        int(positions[name][0]) + int(dims[name][0]) for name in positions
    )

def _compact_area_frontier_positions(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    cohesion_groups: Sequence[Sequence[str]],
    lateral_blocks: Sequence[str],
    order_pairs: Sequence[Tuple[str, str]],
    manufacturing_grid_nm: int,
    legal: Callable[[Mapping[str, Tuple[int, int]]], bool],
) -> Tuple[Dict[str, Tuple[int, int]], bool]:
    """Alternately lower the top cohesion frontier and compact lateral sinks."""

    grid = int(manufacturing_grid_nm)
    if grid <= 0:
        raise ValueError("area frontier compaction requires a positive grid")
    result = {
        str(name): (int(point[0]), int(point[1]))
        for name, point in positions.items()
    }
    if set(result) != set(dims) or not legal(result):
        raise ValueError("area frontier compaction requires one replay-legal placement")
    groups = tuple(
        tuple(sorted(set(map(str, group))))
        for group in cohesion_groups
        if group
    )
    if any(not set(group) <= set(result) for group in groups):
        raise ValueError("area frontier cohesion group names an unknown block")
    changed = False
    iteration_limit = max(4, 4 * (len(groups) + len(lateral_blocks) + 1))
    for _iteration in range(iteration_limit):
        iteration_changed = False
        top = max(result[name][1] + dims[name][1] for name in result)
        for group in groups:
            if max(result[name][1] + dims[name][1] for name in group) != top:
                continue
            if any(result[name][1] % grid for name in group):
                raise ValueError("area frontier cohesion group is off grid")
            maximum_steps = min(result[name][1] for name in group) // grid
            low = 0
            high = maximum_steps
            while low < high:
                middle = (low + high + 1) // 2
                candidate = dict(result)
                for name in group:
                    x, y = result[name]
                    candidate[name] = (x, y - middle * grid)
                if legal(candidate):
                    low = middle
                else:
                    high = middle - 1
            if low:
                for name in group:
                    x, y = result[name]
                    result[name] = (x, y - low * grid)
                iteration_changed = True
                changed = True
                top = max(result[name][1] + dims[name][1] for name in result)
        result, sink_changed = _compact_lateral_sink_positions_x(
            result,
            lateral_blocks=lateral_blocks,
            order_pairs=order_pairs,
            manufacturing_grid_nm=grid,
            legal=legal,
        )
        iteration_changed = iteration_changed or sink_changed
        changed = changed or sink_changed
        if not iteration_changed:
            break
    return result, changed
