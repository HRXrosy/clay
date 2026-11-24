"""Independently replay the complete native placement hard-constraint domain."""
from __future__ import annotations

from clay.layout.mosaic_guest_slot_projection import MosaicGuestSlotProjection, mosaic_guest_slot_satisfied
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from clay.layout.spatial.abstract_ir import SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
from clay.layout.place.constraints_hierarchical import (
    _aggregate_envelope_order_satisfied,
    _assert_aggregate_envelope_order_positions,
    _hierarchical_macro_alignment_parts,
    _hierarchical_macro_alignment_satisfied,
    _hierarchical_macro_order_obligation_id,
    _hierarchical_macro_order_satisfied,
    _require_hierarchical_alignment_tolerance_scales_nm,
)
from clay.layout.place.constraints_spatial import (
    FlexibleEnvelopeConstraint,
    _assert_axis_alignment_positions,
    _assert_center_alignment_positions,
    _bbox_clearance_nm,
    _canonical_flexible_envelope_constraints,
    _canonical_mosaic_guest_slot_projections,
    _canonical_row_bounded_cavity_projections,
    _flexible_seed_physical_views,
    _required_pair_gap_nm,
    _row_bounded_cavity_satisfied,
    _scaled_aspect_bounds,
    _seed_geometry_failure_rows,
    _seed_geometry_legal,
)
from clay.layout.place.placement_constants import (
    _AggregateEnvelopeOrder,
    _BAND_CENTER,
    _Bbox,
    _FloorplanColumnGroup,
    _HierarchicalAlignmentToleranceScalesNm,
    _HierarchicalMacroAlignment,
    _HierarchicalMacroOrder,
    _LayeredCollisionFootprint,
    _RigidCompoundFootprint,
    _RowBoundedCavityProjection,
)
from clay.layout.place.placement_cuts import (
    _PreparedSolveCut,
    _cut_satisfied_by_bboxes,
)
from clay.layout.place.placement_diagnostics import (
    _canonical_full_hard_failure_rows,
    _diagnostic_canonical_value,
    _full_hard_replay_row,
)


def _adjacency_seed_satisfied(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    limits: Sequence[Tuple[str, str, int]],
    scope_offsets_nm: Mapping[str, _Bbox],
    minimum_gap_nm: int,
    projection_nm: int,
) -> bool:
    """Replay the exact four directional adjacency disjunctions on integers."""

    for left, right, maximum in limits:
        lx, ly = seed[left]
        rx, ry = seed[right]
        lw, lh = dims[left]
        rw, rh = dims[right]
        lx0, ly0, lx1, ly1 = scope_offsets_nm[left]
        rx0, ry0, rx1, ry1 = scope_offsets_nm[right]
        directions = (
            (
                lx + lw + minimum_gap_nm <= rx
                and rx + rx0 <= lx + lx1 + maximum
                and ly + ly0 + projection_nm <= ry + ry1
                and ry + ry0 + projection_nm <= ly + ly1
            ),
            (
                rx + rw + minimum_gap_nm <= lx
                and lx + lx0 <= rx + rx1 + maximum
                and ly + ly0 + projection_nm <= ry + ry1
                and ry + ry0 + projection_nm <= ly + ly1
            ),
            (
                ly + lh + minimum_gap_nm <= ry
                and ry + ry0 <= ly + ly1 + maximum
                and lx + lx0 + projection_nm <= rx + rx1
                and rx + rx0 + projection_nm <= lx + lx1
            ),
            (
                ry + rh + minimum_gap_nm <= ly
                and ly + ly0 <= ry + ry1 + maximum
                and lx + lx0 + projection_nm <= rx + rx1
                and rx + rx0 + projection_nm <= lx + lx1
            ),
        )
        if sum(bool(value) for value in directions) != 1:
            return False
    return True

def _require_edge_rail_reservation(reservation: Any, *, dims: Mapping[str, Tuple[int, int]]) -> None:
    if reservation is None:
        return
    from clay.layout.place.edge_rail_reservation import EdgeRailReservationIR

    if type(reservation) is not EdgeRailReservationIR:
        raise TypeError("edge rail reservation must be an exact EdgeRailReservationIR")
    reservation._dimensions(dims)

def _banded_seed_satisfies_hard_constraints(
    seed: Mapping[str, Tuple[int, int]],
    *,
    ids: Sequence[str],
    dims: Mapping[str, Tuple[int, int]],
    gap_nm: int,
    width_bound_nm: int,
    height_bound_nm: int,
    bands: Mapping[str, int],
    adjacency_limits_nm: Sequence[Tuple[str, str, int]],
    adjacency_scope_offsets_nm: Mapping[str, _Bbox],
    order_pairs: Sequence[Tuple[str, str]],
    vertical_order_pairs: Sequence[Tuple[str, str]],
    floorplan_column_classes: Sequence[Tuple[str, int, int, int]],
    floorplan_column_groups: Sequence[_FloorplanColumnGroup],
    bottom_center_keepout_nm: int,
    bottom_edge_keepout_nm: int,
    bottom_edge_keepout_exempt: Sequence[str],
    solve_cuts: Sequence[_PreparedSolveCut],
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]],
    center_alignment_pairs: Sequence[Tuple[str, str, str]],
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder],
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder],
    hierarchical_macro_alignments: Sequence[_HierarchicalMacroAlignment],
    hierarchical_alignment_tolerance_scales_nm: (
        _HierarchicalAlignmentToleranceScalesNm
    ) = (),
    aspect_target: float,
    aspect_bounds: Optional[Tuple[float, float]],
    aspect_obligation: bool,
    manufacturing_grid_nm: int,
    right_edge: Sequence[str],
    left_edge: Sequence[str],
    placement_objective: str,
    strict_width_cap: bool,
    exact_outline_width: bool,
    lateral_blocks: Sequence[str] = (),
    flexible_envelope_constraints: Tuple[FlexibleEnvelopeConstraint, ...] = (),
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    row_bounded_cavity_projections: Tuple[_RowBoundedCavityProjection, ...] = (),
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = (),
    edge_rail_reservation: Any = None,
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
    top_edge_keepout_nm: int = 0,
    top_edge_keepout_clear: Sequence[str] = (),
    enforce_legacy_band_order: bool = True,
    blocker_out: Optional[List[str]] = None,
    diagnostic_out: Optional[List[Dict[str, object]]] = None,
    hard_inventory_out: Optional[List[Dict[str, object]]] = None,
    hard_spec_out: Optional[Dict[str, object]] = None,
) -> bool:
    """Independently replay every hard banded-master constraint on one seed."""

    _require_edge_rail_reservation(edge_rail_reservation, dims=dims)
    canonical_slot_projections = _canonical_mosaic_guest_slot_projections(
        mosaic_guest_slot_projections, dims=dims, minimum_clearance_nm=gap_nm,
    )

    if hard_spec_out is not None:
        hard_spec_out.clear()
        hard_spec_out.update({
            "schema": "clay.banded-hard-replay-input/v1",
            "ids": _diagnostic_canonical_value(ids),
            "dimensions_nm": _diagnostic_canonical_value(dims),
            "gap_nm": int(gap_nm),
            "width_bound_nm": int(width_bound_nm),
            "height_bound_nm": int(height_bound_nm),
            "bands": _diagnostic_canonical_value(bands),
            "adjacency_limits_nm": _diagnostic_canonical_value(
                adjacency_limits_nm
            ),
            "adjacency_scope_offsets_nm": _diagnostic_canonical_value(
                adjacency_scope_offsets_nm
            ),
            "order_pairs": _diagnostic_canonical_value(order_pairs),
            "vertical_order_pairs": _diagnostic_canonical_value(
                vertical_order_pairs
            ),
            "floorplan_column_classes": _diagnostic_canonical_value(
                floorplan_column_classes
            ),
            "floorplan_column_groups": _diagnostic_canonical_value(
                floorplan_column_groups
            ),
            "bottom_center_keepout_nm": int(bottom_center_keepout_nm),
            "bottom_edge_keepout_nm": int(bottom_edge_keepout_nm),
            "bottom_edge_keepout_exempt": _diagnostic_canonical_value(
                bottom_edge_keepout_exempt
            ),
            "solve_cuts": _diagnostic_canonical_value(solve_cuts),
            "pair_clearances_nm": _diagnostic_canonical_value(
                pair_clearances_nm
            ),
            "axis_alignment_groups": _diagnostic_canonical_value(
                axis_alignment_groups
            ),
            "center_alignment_pairs": _diagnostic_canonical_value(
                center_alignment_pairs
            ),
            "aggregate_envelope_orders": _diagnostic_canonical_value(
                aggregate_envelope_orders
            ),
            "hierarchical_macro_orders": _diagnostic_canonical_value(
                hierarchical_macro_orders
            ),
            "hierarchical_macro_alignments": _diagnostic_canonical_value(
                hierarchical_macro_alignments
            ),
            "hierarchical_alignment_tolerance_scales_nm": (
                _diagnostic_canonical_value(
                    hierarchical_alignment_tolerance_scales_nm
                )
            ),
            "aspect_target": float(aspect_target),
            "aspect_bounds": _diagnostic_canonical_value(aspect_bounds),
            "aspect_obligation": bool(aspect_obligation),
            "manufacturing_grid_nm": int(manufacturing_grid_nm),
            "right_edge": _diagnostic_canonical_value(right_edge),
            "left_edge": _diagnostic_canonical_value(left_edge),
            "placement_objective": str(placement_objective),
            "strict_width_cap": bool(strict_width_cap),
            "exact_outline_width": bool(exact_outline_width),
            "lateral_blocks": _diagnostic_canonical_value(lateral_blocks),
            "flexible_envelope_constraints": _diagnostic_canonical_value(
                flexible_envelope_constraints
            ),
            "rigid_compound_footprints": _diagnostic_canonical_value(
                rigid_compound_footprints
            ),
            "layered_collision_footprints": _diagnostic_canonical_value(
                layered_collision_footprints
            ),
            "row_bounded_cavity_projections": _diagnostic_canonical_value(
                tuple(row.canonical_data() for row in row_bounded_cavity_projections)
            ),
            "overlap_compatible_pairs": _diagnostic_canonical_value(
                overlap_compatible_pairs
            ),
            "top_edge_keepout_nm": int(top_edge_keepout_nm),
            "top_edge_keepout_clear": _diagnostic_canonical_value(
                top_edge_keepout_clear
            ),
            "enforce_legacy_band_order": enforce_legacy_band_order,
        })
        if canonical_slot_projections:
            hard_spec_out["mosaic_guest_slot_projections"] = _diagnostic_canonical_value(
                tuple(row.canonical_data() for row in canonical_slot_projections)
            )
        if edge_rail_reservation is not None:
            hard_spec_out["edge_rail_reservation"] = edge_rail_reservation.canonical_data()

    hierarchical_alignment_tolerance_scales_nm = (
        _require_hierarchical_alignment_tolerance_scales_nm(
            hierarchical_alignment_tolerance_scales_nm,
            alignments=hierarchical_macro_alignments,
        )
    )

    declared_rows = []
    for left, right in order_pairs:
        source_id = f"horizontal_order:{left}:{right}"
        declared_rows.append(_full_hard_replay_row(
            source_id,
            "horizontal_order",
            source_constraint_ids=(source_id,),
        ))
    declared_rows.extend(
        _full_hard_replay_row(
            f"vertical_order:{lower}:{upper}",
            "vertical_order",
            source_constraint_ids=(f"vertical_order:{lower}:{upper}",),
        )
        for lower, upper in vertical_order_pairs
    )
    for group in axis_alignment_groups:
        source_id = "axis_alignment:" + ":".join(
            name for name, _offset2 in group
        )
        declared_rows.append(_full_hard_replay_row(
            "axis_alignment:"
            + "=".join(f"{name}@{offset2}" for name, offset2 in group),
            "axis_alignment",
            source_constraint_ids=(source_id,),
        ))
    for first, second, axis in center_alignment_pairs:
        source_id = f"center_alignment:{first}:{second}:{axis}"
        declared_rows.append(_full_hard_replay_row(
            source_id,
            "center_alignment",
            source_constraint_ids=(source_id,),
        ))
    for row in aggregate_envelope_orders:
        before, after, axis, _before_members, _after_members = row
        source_id = f"aggregate_envelope_order:{before}:{after}:{axis}"
        declared_rows.append(_full_hard_replay_row(
            source_id,
            "aggregate_envelope_order",
            source_constraint_ids=(source_id,),
        ))
    for row in hierarchical_macro_orders:
        source_id = _hierarchical_macro_order_obligation_id(row)
        declared_rows.append(_full_hard_replay_row(
            source_id,
            "hierarchical_macro_order",
            source_constraint_ids=(source_id,),
        ))
    for row in hierarchical_macro_alignments:
        scope, axis, edge, _tolerance_ppm, _groups = (
            _hierarchical_macro_alignment_parts(row)
        )
        source_id = (
            scope.removeprefix(SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX)
            if scope.startswith(SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX)
            else f"hierarchical_alignment:{scope}:{axis}:{edge}"
        )
        declared_rows.append(_full_hard_replay_row(
            source_id,
            "hierarchical_macro_alignment",
            source_constraint_ids=(source_id,),
        ))
    for row in row_bounded_cavity_projections:
        declared_rows.append(_full_hard_replay_row(
            f"row_bounded_cavity:{row.obligation_id}",
            "row_bounded_cavity",
            source_constraint_ids=(row.obligation_id,),
        ))
    for row in canonical_slot_projections:
        declared_rows.append(_full_hard_replay_row(
            f"mosaic_guest_slot:{row.obligation_id}",
            "mosaic_guest_slot",
            source_constraint_ids=(row.obligation_id,),
        ))
    for first, second, limit_nm in adjacency_limits_nm:
        source_id = f"adjacent:{first}:{second}:maximum_gap_nm={limit_nm}"
        declared_rows.append(_full_hard_replay_row(
            source_id,
            "adjacency",
            source_constraint_ids=(source_id,),
        ))
    declared_rows.extend(
        _full_hard_replay_row(
            f"solve_cut:{cut.cut_hash}",
            "solve_cut",
        )
        for cut in solve_cuts
    )
    if hard_inventory_out is not None:
        declared_inventory = _canonical_full_hard_failure_rows(declared_rows)
        hard_inventory_out.extend(declared_inventory)

    def reject(
        code: str,
        row: Dict[str, object] | Sequence[Dict[str, object]],
    ) -> bool:
        if blocker_out is not None:
            blocker_out.append(code)
        if diagnostic_out is None and hard_inventory_out is None:
            return False
        raw_rows = (row,) if isinstance(row, Mapping) else row
        canonical_rows = _canonical_full_hard_failure_rows(raw_rows)
        if diagnostic_out is not None:
            diagnostic_out.extend(canonical_rows)
        if hard_inventory_out is not None:
            known_row_ids = {
                str(existing["row_id"]) for existing in hard_inventory_out
            }
            hard_inventory_out.extend(
                item
                for item in canonical_rows
                if str(item["row_id"]) not in known_row_ids
            )
        return False

    if set(seed) != set(ids):
        return reject(
            "inventory",
            _full_hard_replay_row(
                "global:master_inventory",
                "master_inventory",
            ),
        )
    boxes = {
        name: (
            seed[name][0],
            seed[name][1],
            seed[name][0] + dims[name][0],
            seed[name][1] + dims[name][1],
        )
        for name in ids
    }
    canonical_flexible = _canonical_flexible_envelope_constraints(
        flexible_envelope_constraints,
        block_ids=tuple(ids),
        bands=bands,
    )
    flexible_member_boxes, flexible_group_boxes = _flexible_seed_physical_views(
        seed,
        dims=dims,
        constraints=canonical_flexible,
    )
    named_outline_boxes = {
        **boxes,
        **{
            f"flexible_group:{name}": box
            for name, box in flexible_group_boxes.items()
        },
    }
    outline_boxes = tuple(named_outline_boxes.values())
    outline_failures = tuple(
        _full_hard_replay_row(
            f"outline_bounds:{name}",
            "outline_bounds",
        )
        for name, (x0, y0, x1, y1) in sorted(named_outline_boxes.items())
        if x0 < 0 or y0 < 0 or x1 > width_bound_nm or y1 > height_bound_nm
    )
    if outline_failures:
        return reject("outline_bounds", outline_failures)
    x_min = min(box[0] for box in outline_boxes)
    x_max = max(box[2] for box in outline_boxes)
    y_min = min(box[1] for box in outline_boxes)
    y_max = max(box[3] for box in outline_boxes)
    physical_boxes = dict(boxes)
    physical_boxes.update(flexible_member_boxes)
    if edge_rail_reservation is not None and not edge_rail_reservation.replay(
        dims=dims, positions=seed,
    ):
        return reject("edge_rail", _full_hard_replay_row("global:edge_rail", "edge_rail"))
    top_clear = set(top_edge_keepout_clear)
    top_keepout_failures = tuple(
        _full_hard_replay_row(
            f"top_edge_keepout:{name}",
            "top_edge_keepout",
        )
        for name in ids
        if name not in top_clear
        and physical_boxes[name][3] + int(top_edge_keepout_nm) > y_max
    )
    if top_keepout_failures:
        return reject("top_edge_keepout", top_keepout_failures)
    if strict_width_cap and x_max > width_bound_nm:
        return reject(
            "strict_width_cap",
            _full_hard_replay_row(
                "global:strict_width_cap",
                "strict_width_cap",
            ),
        )
    right_rail_margin_nm = int(
        getattr(edge_rail_reservation, "right_margin_nm", 0) or 0
    )
    if exact_outline_width and (
        x_min != 0 or x_max + right_rail_margin_nm != width_bound_nm
    ):
        return reject(
            "exact_outline_width",
            _full_hard_replay_row(
                "global:exact_outline_width",
                "exact_outline_width",
            ),
        )
    if placement_objective == "area" and x_min != 0:
        return reject(
            "area_origin",
            _full_hard_replay_row("global:area_origin", "area_origin"),
        )
    if not _seed_geometry_legal(
        seed,
        dims=dims,
        gap_nm=gap_nm,
        pair_clearances_nm=pair_clearances_nm,
        rigid_compound_footprints=rigid_compound_footprints,
        layered_collision_footprints=layered_collision_footprints,
        overlap_compatible_pairs=overlap_compatible_pairs,
    ):
        if diagnostic_out is None and hard_inventory_out is None:
            return reject("geometry_clearance", ())
        return reject(
            "geometry_clearance",
            _seed_geometry_failure_rows(
                seed,
                dims=dims,
                gap_nm=gap_nm,
                pair_clearances_nm=pair_clearances_nm,
                rigid_compound_footprints=rigid_compound_footprints,
                layered_collision_footprints=layered_collision_footprints,
                overlap_compatible_pairs=overlap_compatible_pairs,
            ),
        )
    canonical_cavities = _canonical_row_bounded_cavity_projections(
        row_bounded_cavity_projections,
        dims=dims,
    )
    failed_cavities = tuple(
        _full_hard_replay_row(
            f"row_bounded_cavity:{row.obligation_id}",
            "row_bounded_cavity",
            source_constraint_ids=(row.obligation_id,),
        )
        for row in canonical_cavities
        if not _row_bounded_cavity_satisfied(
            seed,
            dims=dims,
            projection=row,
            minimum_clearance_nm=gap_nm,
        )
    )
    if failed_cavities:
        return reject("row_bounded_cavity", failed_cavities)
    failed_slots = tuple(
        _full_hard_replay_row(
            f"mosaic_guest_slot:{row.obligation_id}", "mosaic_guest_slot",
            source_constraint_ids=(row.obligation_id,),
        )
        for row in canonical_slot_projections
        if not mosaic_guest_slot_satisfied(seed, row)
    )
    if failed_slots:
        return reject("mosaic_guest_slot", failed_slots)
    for row in canonical_flexible:
        if row.foreign_exclusion_mode == "envelope":
            for foreign in ids:
                if foreign in row.members:
                    continue
                required = max(
                    max(
                        row.foreign_clearance_nm,
                        pair_clearances_nm.get(
                            tuple(sorted((member, foreign))), gap_nm
                        ),
                    )
                    for member in row.members
                )
                if (
                    _bbox_clearance_nm(
                        flexible_group_boxes[row.group], boxes[foreign]
                    )
                    < required
                ):
                    return reject(
                        "flexible_envelope_clearance",
                        _full_hard_replay_row(
                            (
                                "flexible_envelope_clearance:"
                                f"{row.group}<->{foreign}"
                            ),
                            "flexible_envelope_clearance",
                        ),
                    )
        else:
            for member in row.members:
                for foreign in ids:
                    if foreign in row.members:
                        continue
                    required = max(
                        row.foreign_clearance_nm,
                        pair_clearances_nm.get(
                            tuple(sorted((member, foreign))), gap_nm
                        ),
                    )
                    if _bbox_clearance_nm(
                        flexible_member_boxes[member], boxes[foreign]
                    ) < required:
                        return reject(
                            "flexible_member_clearance",
                            _full_hard_replay_row(
                                (
                                    "flexible_member_clearance:"
                                    f"{member}<->{foreign}"
                                ),
                                "flexible_member_clearance",
                            ),
                        )
    exempt = set(bottom_edge_keepout_exempt)
    lateral = set(str(name) for name in lateral_blocks)
    lower_by_name = {}
    for name in ids:
        lower = 0 if name in exempt else int(bottom_edge_keepout_nm)
        if name not in lateral and bands[name] == _BAND_CENTER:
            lower = max(lower, int(bottom_center_keepout_nm))
        lower_by_name[name] = lower
        if (name in exempt and seed[name][1] != 0) or seed[name][1] < lower:
            return reject(
                "bottom_keepout",
                _full_hard_replay_row(
                    f"bottom_keepout:{name}",
                    "bottom_keepout",
                ),
            )
    flexible_lower_slacks = []
    for row in canonical_flexible:
        lower = max(
            0 if all(member in exempt for member in row.members) else int(bottom_edge_keepout_nm),
            int(bottom_center_keepout_nm)
            if bands[row.members[0]] == _BAND_CENTER
            else 0,
        )
        group_y_min = flexible_group_boxes[row.group][1]
        if group_y_min < lower:
            return reject(
                "flexible_bottom_keepout",
                _full_hard_replay_row(
                    f"flexible_bottom_keepout:{row.group}",
                    "flexible_bottom_keepout",
                ),
            )
        flexible_lower_slacks.append(group_y_min - lower)
    if min(
        (*(
            seed[name][1] - lower_by_name[name]
            for name in ids
        ), *flexible_lower_slacks)
    ) != 0:
        return reject(
            "bottom_origin",
            _full_hard_replay_row("global:bottom_origin", "bottom_origin"),
        )
    grid = int(manufacturing_grid_nm)
    if grid <= 0 or any(
        seed[name][0] % grid or seed[name][1] % grid for name in ids
    ):
        return reject(
            "manufacturing_grid",
            tuple(
                _full_hard_replay_row(
                    f"manufacturing_grid:{name}",
                    "manufacturing_grid",
                )
                for name in ids
                if grid <= 0
                or seed[name][0] % grid
                or seed[name][1] % grid
            ) or (_full_hard_replay_row(
                "global:manufacturing_grid",
                "manufacturing_grid",
            ),),
        )
    left_edge_failures = tuple(
        _full_hard_replay_row(f"left_edge:{name}", "left_edge")
        for name in left_edge
        if seed[name][0] != 0
    )
    if left_edge_failures:
        return reject("left_edge", left_edge_failures)
    right_edge_failures = tuple(
        _full_hard_replay_row(f"right_edge:{name}", "right_edge")
        for name in right_edge
        if seed[name][0] + dims[name][0] != width_bound_nm
    )
    if right_edge_failures:
        return reject("right_edge", right_edge_failures)
    if type(enforce_legacy_band_order) is not bool:
        raise TypeError("enforce_legacy_band_order must be an exact bool")
    if enforce_legacy_band_order:
        core_ids = tuple(name for name in ids if name not in lateral)
        present = tuple(sorted({bands[name] for name in core_ids}))
        for index in range(1, len(present)):
            previous = present[index - 1]
            current = present[index]
            previous_tops = [
                boxes[name][3] for name in core_ids if bands[name] == previous
            ] + [
                flexible_group_boxes[row.group][3]
                for row in canonical_flexible
                if bands[row.members[0]] == previous
            ]
            previous_top = max(previous_tops)
            if any(
                seed[name][1] < previous_top + gap_nm
                for name in core_ids if bands[name] == current
            ):
                return reject(
                    "band_order",
                    _full_hard_replay_row(
                        f"band_order:{previous}->{current}",
                        "band_order",
                    ),
                )
            if any(
                flexible_group_boxes[row.group][1] < previous_top + gap_nm
                for row in canonical_flexible
                if bands[row.members[0]] == current
            ):
                affected_groups = tuple(
                    _full_hard_replay_row(
                        f"flexible_band_order:{row.group}:{previous}->{current}",
                        "flexible_band_order",
                    )
                    for row in canonical_flexible
                    if bands[row.members[0]] == current
                    and flexible_group_boxes[row.group][1]
                    < previous_top + gap_nm
                )
                return reject("flexible_band_order", affected_groups)
    for left, right in order_pairs:
        if seed[right][0] < seed[left][0] + dims[left][0] + gap_nm:
            source_id = f"horizontal_order:{left}:{right}"
            return reject(
                "horizontal_order",
                _full_hard_replay_row(
                    source_id,
                    "horizontal_order",
                    source_constraint_ids=(source_id,),
                ),
            )
    for lower, upper in vertical_order_pairs:
        if seed[upper][1] < seed[lower][1] + dims[lower][1] + _required_pair_gap_nm(
            lower, upper, gap_nm, pair_clearances_nm,
        ):
            return reject(
                "vertical_order",
                _full_hard_replay_row(
                    f"vertical_order:{lower}:{upper}",
                    "vertical_order",
                    source_constraint_ids=(f"vertical_order:{lower}:{upper}",),
                ),
            )
    try:
        _assert_axis_alignment_positions(seed, axis_alignment_groups)
    except RuntimeError:
        failed_groups = []
        for group in axis_alignment_groups:
            try:
                _assert_axis_alignment_positions(seed, (group,))
            except RuntimeError:
                source_id = "axis_alignment:" + ":".join(
                    name for name, _offset2 in group
                )
                failed_groups.append(_full_hard_replay_row(
                    "axis_alignment:"
                    + "=".join(
                        f"{name}@{offset2}" for name, offset2 in group
                    ),
                    "axis_alignment",
                    source_constraint_ids=(source_id,),
                ))
        return reject("axis_alignment", failed_groups)
    try:
        _assert_center_alignment_positions(seed, dims, center_alignment_pairs)
    except RuntimeError:
        failed_pairs = []
        for first, second, axis in center_alignment_pairs:
            try:
                _assert_center_alignment_positions(
                    seed,
                    dims,
                    ((first, second, axis),),
                )
            except RuntimeError:
                source_id = f"center_alignment:{first}:{second}:{axis}"
                failed_pairs.append(_full_hard_replay_row(
                    source_id,
                    "center_alignment",
                    source_constraint_ids=(source_id,),
                ))
        return reject("center_alignment", failed_pairs)
    try:
        _assert_aggregate_envelope_order_positions(
            seed, dims, aggregate_envelope_orders
        )
    except RuntimeError:
        failed_orders = []
        for row in aggregate_envelope_orders:
            if _aggregate_envelope_order_satisfied(boxes, row):
                continue
            before, after, axis, _before_members, _after_members = row
            source_id = f"aggregate_envelope_order:{before}:{after}:{axis}"
            failed_orders.append(_full_hard_replay_row(
                source_id,
                "aggregate_envelope_order",
                source_constraint_ids=(source_id,),
            ))
        return reject("aggregate_envelope_order", failed_orders)
    for row in hierarchical_macro_orders:
        if not _hierarchical_macro_order_satisfied(boxes, row, gap_nm=gap_nm):
            source_id = _hierarchical_macro_order_obligation_id(row)
            return reject(
                "hierarchical_order",
                _full_hard_replay_row(
                    source_id,
                    "hierarchical_macro_order",
                    source_constraint_ids=(source_id,),
                ),
            )
    for row in hierarchical_macro_alignments:
        if not _hierarchical_macro_alignment_satisfied(
            boxes,
            row,
            tolerance_scales_nm=hierarchical_alignment_tolerance_scales_nm,
        ):
            scope, axis, edge, _tolerance_ppm, _groups = (
                _hierarchical_macro_alignment_parts(row)
            )
            source_id = (
                scope.removeprefix(
                    SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
                )
                if scope.startswith(
                    SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
                )
                else f"hierarchical_alignment:{scope}:{axis}:{edge}"
            )
            return reject(
                "hierarchical_alignment",
                _full_hard_replay_row(
                    source_id,
                    "hierarchical_macro_alignment",
                    source_constraint_ids=(source_id,),
                ),
            )
    failed_floorplan_pairs = tuple(
        _full_hard_replay_row(
            (
                f"floorplan_column_order:{left}<->{right}:"
                f"{left_column}<{right_column}"
            ),
            "floorplan_column_order",
        )
        for left, left_column, _left_min2, left_max2 in floorplan_column_classes
        for right, right_column, right_min2, _right_max2 in floorplan_column_classes
        if left_column < right_column
        and 2 * boxes[left][0] + left_max2
        >= 2 * boxes[right][0] + right_min2
    )
    if failed_floorplan_pairs:
        return reject(
            "floorplan_column",
            failed_floorplan_pairs,
        )
    group_cell_counts: Dict[Tuple[int, int], int] = {}
    group_cell_sums: Dict[Tuple[int, int], int] = {}
    for _group, row_index, column, member_count, terms in floorplan_column_groups:
        cell = (row_index, column)
        group_cell_counts[cell] = group_cell_counts.get(cell, 0) + member_count
        group_cell_sums[cell] = group_cell_sums.get(cell, 0) + sum(
            2 * count * boxes[owner][0] + center_offset2_sum
            for owner, count, center_offset2_sum in terms
        )
    failed_group_cells = tuple(
        _full_hard_replay_row(
            (
                f"floorplan_group_column_order:row={left_cell[0]}:"
                f"column={left_cell[1]}->{right_cell[1]}"
            ),
            "floorplan_group_column_order",
        )
        for left_cell, left_count in sorted(group_cell_counts.items())
        for right_cell, right_count in sorted(group_cell_counts.items())
        if left_cell[0] == right_cell[0]
        and left_cell[1] < right_cell[1]
        and right_count * group_cell_sums[left_cell]
        >= left_count * group_cell_sums[right_cell]
    )
    if failed_group_cells:
        return reject(
            "floorplan_group_column",
            failed_group_cells,
        )
    failed_cuts = tuple(
        _full_hard_replay_row(
            f"solve_cut:{cut.cut_hash}",
            "solve_cut",
        )
        for cut in solve_cuts
        if not _cut_satisfied_by_bboxes(cut, boxes)
    )
    if failed_cuts:
        return reject("solve_cut", failed_cuts)
    failed_adjacencies = tuple(
        _full_hard_replay_row(
            (source_id := f"adjacent:{first}:{second}:maximum_gap_nm={maximum}"),
            "adjacency",
            source_constraint_ids=(source_id,),
        )
        for first, second, maximum in adjacency_limits_nm
        if not _adjacency_seed_satisfied(
            seed,
            dims=dims,
            limits=((first, second, maximum),),
            scope_offsets_nm=adjacency_scope_offsets_nm,
            minimum_gap_nm=gap_nm,
            projection_nm=grid,
        )
    )
    if failed_adjacencies:
        return reject("adjacency", failed_adjacencies)
    if aspect_obligation:
        lower, upper, scale = _scaled_aspect_bounds(aspect_target, aspect_bounds)
        # The CP aspect declaration remains bound to rigid content rectangles;
        # the physical flexible views independently define outline/objective.
        content_x_min = min(box[0] for box in boxes.values())
        content_x_max = max(box[2] for box in boxes.values())
        content_y_min = min(box[1] for box in boxes.values())
        content_y_max = max(box[3] for box in boxes.values())
        width = content_x_max - content_x_min
        height = content_y_max - content_y_min
        if width * scale < height * lower or width * scale > height * upper:
            return reject(
                "aspect_band",
                _full_hard_replay_row(
                    "global:aspect_band",
                    "aspect_band",
                ),
            )
    return True
