"""Canonical incumbent observations for non-pair Full2D constraints.

This module evaluates the exact integer expressions materialized by Full2D.  It
does not add constraints or participate in solving; callers may omit it without
changing the model.  The registry deliberately calls tightness ``active`` and
keeps necessity typed unmeasurable because one CP-SAT incumbent has no basis or
dual certificate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
from typing import Any
from clay.layout.place.group_separation_strength import separation_gap_nm

from clay.layout.spatial.abstract_ir import (
    SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX,
)


_FAMILIES = ("adjacency", "aggregate", "alignment", "aspect", "hierarchy")
_FAMILY_STATUSES = frozenset((
    "COMPLETE",
    "NOT_DECLARED",
    "UNAVAILABLE_HISTORICAL_FIELD_ABSENT",
))
_ACTIVE_STATUSES = frozenset((
    "ACTIVE_EQUALITY",
    "ACTIVE_ZERO_SLACK",
    "ENFORCED_POSITIVE_SLACK",
    "SATISFIED_POSITIVE_SLACK",
    "UNAVAILABLE_NO_INCUMBENT",
))
_NECESSITY_STATUS = "UNMEASURABLE_SINGLE_INCUMBENT"
_PROVENANCE_STATUSES = frozenset((
    "COMPLETE",
    "UNMEASURABLE_DECLARATION_BINDING_COLLAPSED",
))
_FAMILY_PROVENANCE_STATUSES = frozenset((
    *_PROVENANCE_STATUSES,
    "NOT_DECLARED",
    "UNAVAILABLE_HISTORICAL_FIELD_ABSENT",
))
_DIRECTIONS = ("left", "right", "below", "above")
_ROW_FIELDS = frozenset({
    "active_status",
    "axis",
    "declaration_source",
    "family",
    "model_row_id",
    "necessity_status",
    "provenance_status",
    "projection_source",
    "row_id",
    "selected_literal",
    "slack_unit",
    "slacks",
    "subjects",
    "source_constraint_ids",
})


def _digest_payload(
    *,
    family_provenance_status: object,
    family_status: object,
    incumbent_status: object,
    materialized_model_row_ids: object,
    rows: object,
) -> str:
    return hashlib.sha256(json.dumps(
        {
            "family_provenance_status": family_provenance_status,
            "family_status": family_status,
            "incumbent_status": incumbent_status,
            "materialized_model_row_ids": materialized_model_row_ids,
            "rows": rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")).hexdigest()


def _scaled_aspect_bounds(
    aspect_target: float,
    aspect_bounds: tuple[float, float] | None,
) -> tuple[int, int, int]:
    target = float(aspect_target)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("aspect target must be finite and positive")
    if aspect_bounds is None:
        declared_lower = target * 0.8
        declared_upper = target * 1.2
    else:
        declared_lower, declared_upper = map(float, aspect_bounds)
    if (
        not math.isfinite(declared_lower)
        or not math.isfinite(declared_upper)
        or declared_lower <= 0
        or declared_upper < declared_lower
    ):
        raise ValueError("aspect bounds are malformed")
    scale = 1_000_000
    lower = int(
        (Decimal(str(declared_lower)) * scale).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    upper = int(
        (Decimal(str(declared_upper)) * scale).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )
    if lower <= 0 or upper < lower:
        raise ValueError("aspect interval is not representable")
    return lower, upper, scale


def _envelope(
    positions: Mapping[str, tuple[int, int]],
    dims: Mapping[str, tuple[int, int]],
    members: Sequence[str],
    axis: str,
) -> tuple[int, int]:
    coordinate = 0 if axis == "horizontal" else 1
    return (
        min(int(positions[name][coordinate]) for name in members),
        max(
            int(positions[name][coordinate]) + int(dims[name][coordinate])
            for name in members
        ),
    )


def _status(slacks: Sequence[int], *, equality: bool = False, selected: bool = False) -> str:
    if equality:
        return "ACTIVE_EQUALITY" if all(value == 0 for value in slacks) else "ACTIVE_EQUALITY"
    if any(value < 0 for value in slacks):
        # A solver-produced incumbent can never reach this branch.  Preserving
        # the signed residual lets the shared validator fail closed.
        return "ACTIVE_ZERO_SLACK"
    if any(value == 0 for value in slacks):
        return "ACTIVE_ZERO_SLACK"
    return "ENFORCED_POSITIVE_SLACK" if selected else "SATISFIED_POSITIVE_SLACK"


def _hierarchy_obligation_id(row: Sequence[object]) -> str:
    scope, before_node, after_node, axis, strength = map(str, row[:5])
    return (
        scope.removeprefix("reference_conditioned:")
        if scope.startswith("reference_conditioned:")
        else f"hierarchical_order:{scope}:{before_node}:{after_node}:{axis}:{strength}"
    )


def _alignment_parts(row: Sequence[object]) -> tuple[str, str, str, int, object]:
    if len(row) == 3:
        scope, axis, groups = row
        return str(scope), str(axis), "center", 0, groups
    if len(row) == 4:
        scope, axis, edge, groups = row
        return str(scope), str(axis), str(edge), 0, groups
    scope, axis, edge, tolerance_ppm, groups = row
    return str(scope), str(axis), str(edge), int(tolerance_ppm), groups


def build_full2d_nonpair_constraint_registry(
    *,
    positions: Mapping[str, tuple[int, int]] | None,
    dims: Mapping[str, tuple[int, int]],
    axis_alignment_groups: Sequence[Sequence[tuple[str, int]]] = (),
    center_alignment_pairs: Sequence[tuple[str, str, str]] = (),
    aggregate_envelope_orders: Sequence[Sequence[object]] = (),
    hierarchical_macro_orders: Sequence[Sequence[object]] = (),
    hierarchical_macro_alignments: Sequence[Sequence[object]] = (),
    row_bounded_cavity_projections: Sequence[object] = (),
    mosaic_guest_slot_projections: tuple[object, ...] = (),
    hierarchical_alignment_tolerance_scales_nm: Sequence[tuple[str, int]] = (),
    adjacency_limits_nm: Sequence[tuple[str, str, int]] = (),
    adjacency_scope_offsets_nm: Mapping[str, tuple[int, int, int, int]] | None = None,
    selected_adjacency_directions: Mapping[int, str] | None = None,
    minimum_gap_nm: int,
    projection_nm: int,
    center_step_nm: int,
    aspect_target: float,
    aspect_bounds: tuple[float, float] | None,
    aspect_obligation: bool,
    unavailable_families: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the named five-family registry for one solve phase."""

    has_incumbent = positions is not None
    adjacency_scope_offsets_nm = adjacency_scope_offsets_nm or {}
    selected = dict(selected_adjacency_directions or {})
    unavailable_families = dict(unavailable_families or {})
    if set(unavailable_families) - set(_FAMILIES) or any(
        reason != "UNAVAILABLE_HISTORICAL_FIELD_ABSENT"
        for reason in unavailable_families.values()
    ):
        raise ValueError("non-pair unavailable family inventory is malformed")
    if has_incumbent and len(selected) != len(adjacency_limits_nm):
        raise ValueError("solver evidence lacks a selected adjacency literal")
    rows: list[dict[str, object]] = []

    def add(
        *,
        row_id: str,
        model_row_id: str,
        family: str,
        axis: str,
        source_constraint_ids: Sequence[str],
        declaration_source: str,
        projection_source: str,
        subjects: Sequence[str],
        provenance_status: str = "COMPLETE",
        slacks: Sequence[tuple[str, int]] = (),
        slack_unit: str = "nm",
        equality: bool = False,
        selected_literal: str | None = None,
    ) -> None:
        values = tuple(int(value) for _name, value in slacks)
        rows.append({
            "active_status": (
                "UNAVAILABLE_NO_INCUMBENT"
                if not has_incumbent
                else _status(values, equality=equality, selected=selected_literal is not None)
            ),
            "axis": axis,
            "declaration_source": declaration_source,
            "family": family,
            "model_row_id": model_row_id,
            "necessity_status": _NECESSITY_STATUS,
            "projection_source": projection_source,
            "provenance_status": provenance_status,
            "row_id": row_id,
            "selected_literal": selected_literal if has_incumbent else None,
            "slack_unit": slack_unit,
            "slacks": tuple(slacks) if has_incumbent else (),
            "subjects": tuple(sorted(set(subjects))),
            "source_constraint_ids": tuple(sorted(set(source_constraint_ids))),
        })

    for group_index, group in enumerate(axis_alignment_groups):
        source_id = "axis_alignment:" + ":".join(name for name, _offset in group)
        first, first_offset2 = group[0]
        for second, second_offset2 in group[1:]:
            residual = 0
            if has_incumbent:
                residual = (
                    2 * int(positions[second][0]) + int(second_offset2)
                    - 2 * int(positions[first][0]) - int(first_offset2)
                )
            add(
                row_id=(
                    f"{source_id}:materialized={group_index}:member={second}"
                ),
                model_row_id=f"axis_alignment:{group_index}:member={second}",
                family="alignment",
                axis="horizontal",
                source_constraint_ids=(source_id,),
                declaration_source="clay/layout/intent.py",
                provenance_status=(
                    "UNMEASURABLE_DECLARATION_BINDING_COLLAPSED"
                ),
                projection_source="clay/layout/place_blocks_2d.py:_canonical_axis_alignment_groups",
                subjects=tuple(name for name, _offset in group),
                slacks=(("signed_equality_residual_two_nm", residual),),
                slack_unit="two_nm",
                equality=True,
            )
    for pair_index, (first, second, axis) in enumerate(center_alignment_pairs):
        coordinate = 0 if axis == "horizontal" else 1
        source_id = f"center_alignment:{first}:{second}:{axis}"
        residual = 0
        if has_incumbent:
            residual = (
                2 * int(positions[second][coordinate]) + int(dims[second][coordinate])
                - 2 * int(positions[first][coordinate]) - int(dims[first][coordinate])
            )
        add(
            row_id=f"{source_id}:materialized={pair_index}",
            model_row_id=f"center_alignment:{pair_index}",
            family="alignment",
            axis=axis,
            source_constraint_ids=(source_id,),
            declaration_source=(
                "clay/layout/spatial/intent_projection.py:"
                "MosaicCenterAlignmentIntent"
            ),
            provenance_status="UNMEASURABLE_DECLARATION_BINDING_COLLAPSED",
            projection_source="clay/layout/place_blocks_2d.py:_canonical_center_alignment_pairs",
            subjects=(first, second),
            slacks=(("signed_equality_residual_two_nm", residual),),
            slack_unit="two_nm",
            equality=True,
        )

    for order_index, (
        before_scope,
        after_scope,
        axis,
        before,
        after,
    ) in enumerate(aggregate_envelope_orders):
        source_id = f"aggregate_envelope_order:{before_scope}:{after_scope}:{axis}"
        slack = 0
        if has_incumbent:
            before_low, before_high = _envelope(positions, dims, before, axis)
            after_low, after_high = _envelope(positions, dims, after, axis)
            from .constraints_hierarchical import _is_rail_halfplane_order
            step = 0 if _is_rail_halfplane_order(
                (before_scope, after_scope, axis, before, after)
            ) else 1
            slack = after_low + after_high - before_low - before_high - step
        add(
            row_id=f"{source_id}:materialized={order_index}",
            model_row_id=f"aggregate_order:{order_index}",
            family="aggregate",
            axis=str(axis),
            source_constraint_ids=(source_id,),
            declaration_source="clay/layout/spatial/intent_projection.py:AggregateOrderIntent",
            projection_source=(
                "clay/layout/place_blocks_2d.py:"
                "_aggregate_envelope_orders_from_intent"
            ),
            subjects=(*before, *after),
            slacks=(("center_order_slack_two_nm", slack),),
            slack_unit="two_nm",
        )

    for order_index, row in enumerate(hierarchical_macro_orders):
        scope, before_node, after_node, axis, strength, before, after = row
        source_id = _hierarchy_obligation_id(row)
        if has_incumbent:
            before_low, before_high = _envelope(positions, dims, before, str(axis))
            after_low, after_high = _envelope(positions, dims, after, str(axis))
        else:
            before_low = before_high = after_low = after_high = 0
        if separation_gap_nm(strength) is not None:
            components = ((None, after_low - before_high - separation_gap_nm(strength), "nm"),)
        elif strength == "leading_edge_order":
            components = ((None, after_low - before_low - int(center_step_nm), "nm"),)
        elif strength == "leading_edge_equal":
            components = ((None, after_low - before_low, "nm"),)
        elif strength in {
            "member_before_aggregate_center",
            "member_before_aggregate_center_with_track_floor",
        }:
            floor = int(
                minimum_gap_nm
                if strength.endswith("track_floor")
                else center_step_nm
            )
            coordinate = 0 if axis == "horizontal" else 1
            components = tuple(
                (
                    str(name),
                    after_low + after_high
                    - 2 * int(positions[name][coordinate])
                    - int(dims[name][coordinate])
                    - 2 * floor if has_incumbent else 0,
                    "two_nm",
                )
                for name in before
            )
        else:
            floor = (
                minimum_gap_nm
                if strength == "center_order_with_track_floor"
                else center_step_nm
            )
            components = ((
                None,
                after_low + after_high - before_low - before_high - 2 * int(floor),
                "two_nm",
            ),)
        for member, slack, unit in components:
            add(
                row_id=(
                    source_id
                    + f":materialized={order_index}"
                    + (f":member={member}" if member is not None else "")
                ),
                model_row_id=(
                    f"hierarchy_order:{order_index}"
                    + (f":member={member}" if member is not None else "")
                ),
                family="hierarchy",
                axis=str(axis),
                source_constraint_ids=(source_id,),
                declaration_source=(
                    "clay/layout/spatial/intent_projection.py:"
                    "ReferenceConditionedPlacementObligation"
                    if str(scope).startswith("reference_conditioned:")
                    else (
                        "clay/layout/spatial/intent_projection.py:"
                        "HierarchicalMacroDagIntent"
                    )
                ),
                projection_source=(
                    "clay/layout/place_blocks_2d.py:"
                    "_hierarchical_macro_constraints_from_intent"
                ),
                subjects=(*before, *after),
                slacks=(("materialized_slack", int(slack)),),
                slack_unit=unit,
                equality=strength == "leading_edge_equal",
            )

    if type(mosaic_guest_slot_projections) is not tuple:
        raise TypeError("guest-slot registry projections must be an exact tuple")
    if mosaic_guest_slot_projections:
        from clay.layout.mosaic_guest_slot_projection import (
            canonical_mosaic_guest_slot_projections, mosaic_guest_slot_bounds_nm,
        )
        canonical_mosaic_guest_slot_projections(mosaic_guest_slot_projections, dims=dims)
        for slot_index, slot in enumerate(mosaic_guest_slot_projections):
            if slot.minimum_clearance_nm != minimum_gap_nm:
                raise ValueError("guest-slot registry clearance differs from source gap")
            xmin, xmax, ymin, ymax = mosaic_guest_slot_bounds_nm(slot)
            slacks = ()
            if has_incumbent:
                hx, hy = positions[slot.host_master]
                gx, gy = positions[slot.guest_master]
                slacks = (("x_lower_slack_nm", gx-hx-xmin), ("x_upper_slack_nm", xmax-gx+hx),
                          ("y_lower_slack_nm", gy-hy-ymin), ("y_upper_slack_nm", ymax-gy+hy))
            add(row_id=f"mosaic_guest_slot:{slot.obligation_id}",
                model_row_id=f"mosaic_guest_slot:{slot_index}", family="hierarchy", axis="joint",
                source_constraint_ids=(slot.obligation_id,),
                declaration_source="clay/layout/spatial/mosaic_guest_slot.py:MosaicGuestSlot",
                projection_source="clay/layout/mosaic_guest_slot_projection.py:mosaic_guest_slot_bounds_nm",
                subjects=(slot.host_master, slot.guest_master), slacks=slacks)

    for cavity_index, cavity in enumerate(row_bounded_cavity_projections):
        obligation_id = str(getattr(cavity, "obligation_id", ""))
        host = str(getattr(cavity, "host_master", ""))
        guests = tuple(getattr(cavity, "guest_masters", ()))
        lower = int(getattr(cavity, "lower_boundary_offset_nm", -1))
        upper = int(getattr(cavity, "upper_boundary_offset_nm", -1))
        if (
            not obligation_id
            or host not in dims
            or not guests
            or any(guest not in dims for guest in guests)
            or not 0 < lower < upper
        ):
            raise ValueError("row-bounded cavity registry projection is malformed")
        for guest in guests:
            slacks = ()
            if has_incumbent:
                host_x, host_y = positions[host]
                guest_x, guest_y = positions[guest]
                slacks = (
                    ("host_left_slack_nm", int(guest_x) - int(host_x)),
                    (
                        "host_right_slack_nm",
                        int(host_x) + int(dims[host][0])
                        - int(guest_x) - int(dims[guest][0]),
                    ),
                    (
                        "lower_row_clearance_slack_nm",
                        int(guest_y) - int(host_y) - lower - int(minimum_gap_nm),
                    ),
                    (
                        "host_top_clearance_slack_nm",
                        int(host_y) + upper - int(guest_y)
                        - int(dims[guest][1]) - int(minimum_gap_nm),
                    ),
                )
            add(
                row_id=(
                    f"row_bounded_cavity:{obligation_id}:guest={guest}"
                ),
                model_row_id=(
                    f"row_bounded_cavity:{cavity_index}:guest={guest}"
                ),
                family="hierarchy",
                axis="joint",
                source_constraint_ids=(obligation_id,),
                declaration_source=(
                    "clay/layout/spatial/intent_projection.py:"
                    "ReferenceConditionedPlacementObligation"
                ),
                projection_source=(
                    "clay/layout/canonical_collision_projection.py:"
                    "CanonicalRowBoundedCavityProjection"
                ),
                subjects=(host, guest),
                slacks=slacks,
            )

    scales = dict(hierarchical_alignment_tolerance_scales_nm)
    for alignment_index, raw in enumerate(hierarchical_macro_alignments):
        scope, axis, edge, tolerance_ppm, groups = _alignment_parts(raw)
        source_id = (
            scope.removeprefix(SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX)
            if scope.startswith(SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX)
            else f"hierarchical_alignment:{scope}:{axis}:{edge}"
        )
        base_node, base_members = groups[0]
        if has_incumbent:
            base_low, base_high = _envelope(positions, dims, base_members, axis)
        else:
            base_low = base_high = 0
        base_value = (
            base_low
            if edge == "leading"
            else base_high
            if edge == "trailing"
            else base_low + base_high
        )
        scale = 2 if edge == "center" else 1
        for node, members in groups[1:]:
            if has_incumbent:
                low, high = _envelope(positions, dims, members, axis)
            else:
                low = high = 0
            value = low if edge == "leading" else high if edge == "trailing" else low + high
            if tolerance_ppm == 0:
                slack = value - base_value
                unit = "two_nm" if edge == "center" else "nm"
                equality = True
            else:
                slack = (
                    scale * int(tolerance_ppm) * int(scales[axis])
                    - abs(base_value - value) * 1_000_000
                )
                unit = "nm_ppm_product"
                equality = False
            add(
                row_id=(
                    f"{source_id}:materialized={alignment_index}:node={node}"
                ),
                model_row_id=f"hierarchy_alignment:{alignment_index}:node={node}",
                family="alignment",
                axis=axis,
                source_constraint_ids=(source_id,),
                declaration_source=(
                    "clay/layout/spatial/intent_projection.py:"
                    + (
                        "AggregateAlignmentIntent"
                        if scope.startswith(
                            SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
                        )
                        else "HierarchicalMacroDagIntent"
                    )
                ),
                projection_source=(
                    "clay/layout/place_blocks_2d.py:"
                    "_hierarchical_macro_constraints_from_intent"
                ),
                subjects=(*base_members, *members),
                slacks=(("alignment_residual", int(slack)),),
                slack_unit=unit,
                equality=equality,
            )

    for index, (left, right, maximum) in enumerate(adjacency_limits_nm):
        source_id = f"adjacent:{left}:{right}:maximum_gap_nm={maximum}"
        direction = selected.get(index)
        if has_incumbent and direction not in _DIRECTIONS:
            raise ValueError("solver evidence has an invalid selected adjacency literal")
        slacks: tuple[tuple[str, int], ...] = ()
        axis = "joint"
        if has_incumbent:
            lx, ly = positions[left]
            rx, ry = positions[right]
            lw, lh = dims[left]
            rw, rh = dims[right]
            lx0, ly0, lx1, ly1 = adjacency_scope_offsets_nm[left]
            rx0, ry0, rx1, ry1 = adjacency_scope_offsets_nm[right]
            if direction == "left":
                axis = "horizontal"
                values = (
                    rx - lx - lw - minimum_gap_nm,
                    lx + lx1 + maximum - rx - rx0,
                    ry + ry1 - ly - ly0 - projection_nm,
                    ly + ly1 - ry - ry0 - projection_nm,
                )
            elif direction == "right":
                axis = "horizontal"
                values = (
                    lx - rx - rw - minimum_gap_nm,
                    rx + rx1 + maximum - lx - lx0,
                    ry + ry1 - ly - ly0 - projection_nm,
                    ly + ly1 - ry - ry0 - projection_nm,
                )
            elif direction == "below":
                axis = "vertical"
                values = (
                    ry - ly - lh - minimum_gap_nm,
                    ly + ly1 + maximum - ry - ry0,
                    rx + rx1 - lx - lx0 - projection_nm,
                    lx + lx1 - rx - rx0 - projection_nm,
                )
            else:
                axis = "vertical"
                values = (
                    ly - ry - rh - minimum_gap_nm,
                    ry + ry1 + maximum - ly - ly0,
                    rx + rx1 - lx - lx0 - projection_nm,
                    lx + lx1 - rx - rx0 - projection_nm,
                )
            slacks = tuple(zip((
                "minimum_separation_nm",
                "scoped_maximum_gap_nm",
                "projection_overlap_first_nm",
                "projection_overlap_second_nm",
            ), map(int, values)))
        add(
            row_id=f"{source_id}:materialized={index}",
            model_row_id=f"adjacency:{index}",
            family="adjacency",
            axis=axis,
            source_constraint_ids=(source_id,),
            declaration_source="clay/layout/intent.py:AdjacentIntent",
            provenance_status="UNMEASURABLE_DECLARATION_BINDING_COLLAPSED",
            projection_source=(
                "clay/layout/place_blocks_2d.py:"
                "_canonical_adjacency_scope_offsets_nm"
            ),
            subjects=(left, right),
            slacks=slacks,
            selected_literal=direction,
        )

    if aspect_obligation:
        from clay.layout.policy import aspect_policy_binding

        _declared, aspect_policy_sha256 = aspect_policy_binding(aspect_target)
        lower, upper, scale = _scaled_aspect_bounds(aspect_target, aspect_bounds)
        if has_incumbent:
            width = max(
                positions[name][0] + dims[name][0] for name in positions
            ) - min(positions[name][0] for name in positions)
            height = max(
                positions[name][1] + dims[name][1] for name in positions
            ) - min(positions[name][1] for name in positions)
        else:
            width = height = 0
        for bound, slack in (
            ("lower", width * scale - height * lower),
            ("upper", height * upper - width * scale),
        ):
            add(
                row_id=f"global:aspect_band:{bound}",
                model_row_id=f"aspect_obligation:{bound}",
                family="aspect",
                axis="joint",
                source_constraint_ids=(
                    f"aspect_policy:{aspect_policy_sha256}",
                ),
                declaration_source="clay/layout/policy.py:aspect_policy_binding",
                projection_source=(
                    "clay/layout/place_blocks_2d.py:"
                    "_add_directional_aspect_constraints"
                ),
                subjects=tuple(sorted(dims)),
                slacks=((f"{bound}_bound_slack", int(slack)),),
                slack_unit="nm_ratio_scale",
            )

    rows.sort(key=lambda row: (str(row["row_id"]), str(row["model_row_id"])))
    declared = {
        "adjacency": bool(adjacency_limits_nm),
        "aggregate": bool(aggregate_envelope_orders),
        "alignment": bool(
            axis_alignment_groups
            or center_alignment_pairs
            or hierarchical_macro_alignments
        ),
        "aspect": bool(aspect_obligation),
        "hierarchy": bool(
            hierarchical_macro_orders or row_bounded_cavity_projections or mosaic_guest_slot_projections
        ),
    }
    family_status = tuple(
        (
            family,
            unavailable_families.get(
                family,
                "COMPLETE" if declared[family] else "NOT_DECLARED",
            ),
        )
        for family in _FAMILIES
    )
    if any(
        family in unavailable_families and declared[family]
        for family in _FAMILIES
    ):
        raise ValueError("unavailable non-pair family contains reconstructed rows")
    incumbent_status = (
        "AVAILABLE" if has_incumbent else "UNAVAILABLE_NO_INCUMBENT"
    )
    provenance_by_family = {
        family: {
            str(row["provenance_status"])
            for row in rows
            if row["family"] == family
        }
        for family in _FAMILIES
    }
    family_provenance_status = tuple(
        (
            family,
            unavailable_families.get(
                family,
                (
                    "NOT_DECLARED"
                    if not declared[family]
                    else (
                        "COMPLETE"
                        if provenance_by_family[family] == {"COMPLETE"}
                        else "UNMEASURABLE_DECLARATION_BINDING_COLLAPSED"
                    )
                ),
            ),
        )
        for family in _FAMILIES
    )
    materialized_model_row_ids = tuple(sorted(
        str(row["model_row_id"]) for row in rows
    ))
    result = {
        "family_provenance_status": family_provenance_status,
        "family_status": family_status,
        "incumbent_status": incumbent_status,
        "materialized_model_row_ids": materialized_model_row_ids,
        "registry_sha256": _digest_payload(
            family_provenance_status=family_provenance_status,
            family_status=family_status,
            incumbent_status=incumbent_status,
            materialized_model_row_ids=materialized_model_row_ids,
            rows=rows,
        ),
        "rows": tuple(rows),
        "schema": "clay.full2d-nonpair-constraint-registry/v1",
    }
    validate_full2d_nonpair_constraint_registry(result)
    return result


def validate_full2d_nonpair_constraint_registry(value: object) -> dict[str, Any]:
    """Return canonical JSON after fail-closed registry validation."""

    try:
        canonical = json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("full-2D non-pair registry is not canonical JSON") from exc
    if type(canonical) is not dict or set(canonical) != {
        "family_provenance_status", "family_status", "incumbent_status",
        "materialized_model_row_ids", "registry_sha256", "rows", "schema"
    }:
        raise ValueError("full-2D non-pair registry schema differs")
    if canonical["schema"] != "clay.full2d-nonpair-constraint-registry/v1":
        raise ValueError("full-2D non-pair registry version differs")
    family_status = canonical["family_status"]
    if (
        type(family_status) is not list
        or any(type(row) is not list or len(row) != 2 for row in family_status)
        or [row[0] for row in family_status] != list(_FAMILIES)
        or any(row[1] not in _FAMILY_STATUSES for row in family_status)
    ):
        raise ValueError("full-2D non-pair family inventory differs")
    incumbent_status = canonical["incumbent_status"]
    if incumbent_status not in {"AVAILABLE", "UNAVAILABLE_NO_INCUMBENT"}:
        raise ValueError("full-2D non-pair incumbent status differs")
    rows = canonical["rows"]
    if type(rows) is not list:
        raise ValueError("full-2D non-pair rows are malformed")
    row_ids = []
    model_row_ids = []
    families_with_rows = set()
    for row in rows:
        if type(row) is not dict or set(row) != _ROW_FIELDS:
            raise ValueError("full-2D non-pair row schema differs")
        row_id = row["row_id"]
        model_row_id = row["model_row_id"]
        family = row["family"]
        status = row["active_status"]
        slacks = row["slacks"]
        selected = row["selected_literal"]
        sources = row["source_constraint_ids"]
        subjects = row["subjects"]
        if (
            type(row_id) is not str or not row_id
            or type(model_row_id) is not str or not model_row_id
            or family not in _FAMILIES
            or status not in _ACTIVE_STATUSES
            or row["axis"] not in {"horizontal", "vertical", "joint"}
            or row["necessity_status"] != _NECESSITY_STATUS
            or row["provenance_status"] not in _PROVENANCE_STATUSES
            or type(row["declaration_source"]) is not str or not row["declaration_source"]
            or type(row["projection_source"]) is not str or not row["projection_source"]
            or type(row["slack_unit"]) is not str or not row["slack_unit"]
            or type(sources) is not list or not sources or sources != sorted(set(sources))
            or any(type(source) is not str or not source for source in sources)
            or type(subjects) is not list or not subjects or subjects != sorted(set(subjects))
            or any(type(subject) is not str or not subject for subject in subjects)
            or type(slacks) is not list
            or any(
                type(item) is not list
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0]
                or type(item[1]) is not int
                for item in slacks
            )
        ):
            raise ValueError("full-2D non-pair row authority differs")
        values = [item[1] for item in slacks]
        if status == "UNAVAILABLE_NO_INCUMBENT":
            if slacks or selected is not None:
                raise ValueError("full-2D non-pair unavailable row has observations")
            if family == "adjacency" and row["axis"] != "joint":
                raise ValueError("full-2D non-pair unavailable adjacency axis differs")
        else:
            if not slacks:
                raise ValueError("full-2D non-pair incumbent row lacks residuals")
            if status == "ACTIVE_EQUALITY" and any(value != 0 for value in values):
                raise ValueError("full-2D non-pair equality residual differs")
            if status != "ACTIVE_EQUALITY" and any(value < 0 for value in values):
                raise ValueError("full-2D non-pair row violates its incumbent")
            if status == "ACTIVE_ZERO_SLACK" and 0 not in values:
                raise ValueError("full-2D non-pair active row lacks zero slack")
            if status in {
                "SATISFIED_POSITIVE_SLACK",
                "ENFORCED_POSITIVE_SLACK",
            } and any(value <= 0 for value in values):
                raise ValueError("full-2D non-pair positive row is not positive")
            if family == "adjacency":
                if selected not in _DIRECTIONS or status not in {
                    "ACTIVE_ZERO_SLACK",
                    "ENFORCED_POSITIVE_SLACK",
                }:
                    raise ValueError("full-2D non-pair adjacency selection differs")
                selected_axis = (
                    "horizontal" if selected in {"left", "right"} else "vertical"
                )
                if row["axis"] != selected_axis:
                    raise ValueError("full-2D non-pair adjacency observation axis differs")
            elif selected is not None:
                raise ValueError("full-2D non-pair non-adjacency row selects a literal")
        row_ids.append(row_id)
        model_row_ids.append(model_row_id)
        families_with_rows.add(family)
    if row_ids != sorted(set(row_ids)):
        raise ValueError("full-2D non-pair row IDs are not canonical and unique")
    if len(model_row_ids) != len(set(model_row_ids)):
        raise ValueError(
            "full-2D non-pair model row IDs are not canonical and unique"
        )
    materialized_model_row_ids = canonical["materialized_model_row_ids"]
    if (
        type(materialized_model_row_ids) is not list
        or materialized_model_row_ids != sorted(model_row_ids)
    ):
        raise ValueError("full-2D non-pair materialized row inventory differs")
    status_by_family = dict(family_status)
    if any(
        (status_by_family[family] == "COMPLETE")
        != (family in families_with_rows)
        for family in _FAMILIES
    ):
        raise ValueError("full-2D non-pair family inventory contradicts rows")
    family_provenance_status = canonical["family_provenance_status"]
    if (
        type(family_provenance_status) is not list
        or any(
            type(row) is not list or len(row) != 2
            for row in family_provenance_status
        )
        or [row[0] for row in family_provenance_status] != list(_FAMILIES)
        or any(
            row[1] not in _FAMILY_PROVENANCE_STATUSES
            for row in family_provenance_status
        )
    ):
        raise ValueError("full-2D non-pair family provenance inventory differs")
    expected_family_provenance = []
    for family in _FAMILIES:
        status = status_by_family[family]
        if status == "NOT_DECLARED":
            expected = "NOT_DECLARED"
        elif status == "UNAVAILABLE_HISTORICAL_FIELD_ABSENT":
            expected = "UNAVAILABLE_HISTORICAL_FIELD_ABSENT"
        else:
            row_statuses = {
                row["provenance_status"]
                for row in rows
                if row["family"] == family
            }
            expected = (
                "COMPLETE"
                if row_statuses == {"COMPLETE"}
                else "UNMEASURABLE_DECLARATION_BINDING_COLLAPSED"
            )
        expected_family_provenance.append([family, expected])
    if family_provenance_status != expected_family_provenance:
        raise ValueError("full-2D non-pair family provenance contradicts rows")
    if any(
        (row["active_status"] == "UNAVAILABLE_NO_INCUMBENT")
        != (incumbent_status == "UNAVAILABLE_NO_INCUMBENT")
        for row in rows
    ):
        raise ValueError("full-2D non-pair incumbent status contradicts rows")
    actual_sha = _digest_payload(
        family_provenance_status=family_provenance_status,
        family_status=family_status,
        incumbent_status=incumbent_status,
        materialized_model_row_ids=materialized_model_row_ids,
        rows=rows,
    )
    if canonical["registry_sha256"] != actual_sha:
        raise ValueError("full-2D non-pair registry digest differs")
    return canonical


__all__ = [
    "build_full2d_nonpair_constraint_registry",
    "validate_full2d_nonpair_constraint_registry",
]
