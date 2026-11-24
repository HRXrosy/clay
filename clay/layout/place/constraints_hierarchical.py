"""Declared aggregate and hierarchical relations, lowering, and numerical replay."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from clay.layout.place.group_separation_strength import separation_gap_nm
from clay.layout.spatial.abstract_ir import SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
from clay.layout.place.constraints_spatial import (
    FlexibleEnvelopeConstraint,
    _ceil_ratio,
    _placement_primary_extent_nm,
)
from clay.layout.place.placement_constants import (
    _AggregateEnvelopeOrder,
    _BAND_CENTER,
    _Bbox,
    _HierarchicalAlignmentToleranceScalesNm,
    _HierarchicalMacroAlignment,
    _HierarchicalMacroAlignmentGroups,
    _HierarchicalMacroOrder,
    _RigidCompoundFootprint,
    _SECONDARY_AXIS_REPAIRABLE_BLOCKERS,
)
from clay.layout.place.placement_diagnostics import (
    _full_hard_replay_row,
)


def _hierarchical_macro_alignment_parts(
    row: _HierarchicalMacroAlignment,
) -> Tuple[str, str, str, int, _HierarchicalMacroAlignmentGroups]:
    if len(row) == 3:
        scope, axis, groups = row
        return scope, axis, "center", 0, groups
    if len(row) == 4:
        scope, axis, edge, groups = row
        return scope, axis, edge, 0, groups
    scope, axis, edge, tolerance_ppm, groups = row
    return scope, axis, edge, tolerance_ppm, groups

def _canonical_hierarchical_alignment_tolerance_scales_nm(
    values: object,
    *,
    alignments: Sequence[_HierarchicalMacroAlignment],
) -> _HierarchicalAlignmentToleranceScalesNm:
    """Validate the derived, coordinate-free scale used by tolerance bands."""

    if type(values) is not tuple:
        raise TypeError(
            "hierarchical alignment tolerance scales must be one exact tuple"
        )
    required_axes = tuple(
        sorted(
            {
                axis
                for row in alignments
                for _scope, axis, _edge, tolerance_ppm, _groups in (
                    _hierarchical_macro_alignment_parts(row),
                )
                if tolerance_ppm > 0
            }
        )
    )
    rows: list[Tuple[str, int]] = []
    for raw in values:
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError("hierarchical alignment tolerance scale is malformed")
        axis, scale_nm = raw
        if axis not in {"horizontal", "vertical"}:
            raise ValueError("hierarchical alignment tolerance scale has unknown axis")
        if type(scale_nm) is not int or scale_nm <= 0:
            raise ValueError(
                "hierarchical alignment tolerance scale must be positive nm"
            )
        rows.append((axis, scale_nm))
    canonical = tuple(sorted(rows))
    if canonical != values or len(canonical) != len(set(canonical)):
        raise ValueError(
            "hierarchical alignment tolerance scales must be canonical and unique"
        )
    if canonical and tuple(axis for axis, _scale_nm in canonical) != required_axes:
        raise ValueError(
            "hierarchical alignment tolerance scales differ from required axes"
        )
    return canonical

def _require_hierarchical_alignment_tolerance_scales_nm(
    values: object,
    *,
    alignments: Sequence[_HierarchicalMacroAlignment],
) -> _HierarchicalAlignmentToleranceScalesNm:
    canonical = _canonical_hierarchical_alignment_tolerance_scales_nm(
        values,
        alignments=alignments,
    )
    if any(
        _hierarchical_macro_alignment_parts(row)[3] > 0 for row in alignments
    ) and not canonical:
        raise ValueError(
            "production tolerance alignment requires its derived frozen scale"
        )
    return canonical

def _derive_hierarchical_alignment_tolerance_scales_nm(
    alignments: Sequence[_HierarchicalMacroAlignment],
    *,
    frozen_width_nm: int,
    declared_aspect: float,
    maximum_master_height_nm: int,
) -> _HierarchicalAlignmentToleranceScalesNm:
    """Freeze tolerance normalization before any seed/domain expansion."""

    axes = tuple(
        sorted(
            {
                axis
                for row in alignments
                for _scope, axis, _edge, tolerance_ppm, _groups in (
                    _hierarchical_macro_alignment_parts(row),
                )
                if tolerance_ppm > 0
            }
        )
    )
    if not axes:
        return ()
    if type(frozen_width_nm) is not int or frozen_width_nm <= 0:
        raise ValueError("alignment tolerance width scale must be positive nm")
    if not math.isfinite(declared_aspect) or declared_aspect <= 0.0:
        raise ValueError("alignment tolerance scale requires a positive aspect")
    vertical_ratio = frozen_width_nm / declared_aspect
    if not math.isfinite(vertical_ratio):
        raise ValueError("alignment tolerance scale requires a finite vertical scale")
    vertical_scale_nm = max(
        int(maximum_master_height_nm),
        int(math.ceil(vertical_ratio)),
    )
    available = {
        "horizontal": frozen_width_nm,
        "vertical": vertical_scale_nm,
    }
    return tuple((axis, available[axis]) for axis in axes)

def _is_rail_halfplane_order(row: _AggregateEnvelopeOrder) -> bool:
    before, after, axis, before_members, after_members = row
    if axis != "vertical":
        return False
    prefix = "rail_affinity:"
    if not before.startswith(prefix) or not after.startswith(prefix):
        return False
    before_group, before_role = before.rsplit(":", 1)
    after_group, after_role = after.rsplit(":", 1)
    if before_group != after_group or {before_role, after_role} != {"universe", "target"}:
        return False
    universe = before_members if before_role == "universe" else after_members
    target = after_members if before_role == "universe" else before_members
    return bool(target) and set(target) <= set(universe)


def _canonical_aggregate_envelope_orders(
    values: object,
    *,
    block_ids: Sequence[str],
) -> Tuple[_AggregateEnvelopeOrder, ...]:
    if type(values) is not tuple:
        raise TypeError("aggregate envelope orders must be one exact tuple")
    known = {str(name) for name in block_ids}
    rows: list[_AggregateEnvelopeOrder] = []
    for raw in values:
        if type(raw) is not tuple or len(raw) != 5:
            raise TypeError("aggregate envelope order rows must be exact five-item tuples")
        before_scope, after_scope, axis, raw_before, raw_after = raw
        if (
            type(before_scope) is not str
            or not before_scope
            or type(after_scope) is not str
            or not after_scope
            or before_scope == after_scope
            or axis not in {"horizontal", "vertical"}
            or type(raw_before) is not tuple
            or type(raw_after) is not tuple
        ):
            raise ValueError("aggregate envelope order semantic signature is malformed")
        before = tuple(sorted(str(name) for name in raw_before))
        after = tuple(sorted(str(name) for name in raw_after))
        if (
            not before
            or not after
            or before != raw_before
            or after != raw_after
            or len(set(before)) != len(before)
            or len(set(after)) != len(after)
            or (set(before) & set(after) and not _is_rail_halfplane_order(raw))
            or not set((*before, *after)) <= known
        ):
            raise ValueError("aggregate envelope order block ownership is malformed")
        if _is_rail_halfplane_order(raw) and set((*before, *after)) != known:
            raise ValueError("rail affinity universe differs from live master inventory")
        rows.append((before_scope, after_scope, axis, before, after))
    canonical = tuple(sorted(rows))
    if canonical != values or len(canonical) != len(set(canonical)):
        raise ValueError("aggregate envelope orders must be canonical and unique")
    return canonical

def _aggregate_envelope_order_satisfied(
    blocks: Mapping[str, _Bbox],
    row: _AggregateEnvelopeOrder,
) -> bool:
    _before_scope, _after_scope, axis, before, after = row
    lower = 0 if axis == "horizontal" else 1
    upper = 2 if axis == "horizontal" else 3
    before_center2 = min(blocks[name][lower] for name in before) + max(
        blocks[name][upper] for name in before
    )
    after_center2 = min(blocks[name][lower] for name in after) + max(
        blocks[name][upper] for name in after
    )
    return before_center2 <= after_center2 if _is_rail_halfplane_order(row) else before_center2 < after_center2

def _orthogonal_single_master_cross_tier_aggregate_keys(
    orders: Sequence[_AggregateEnvelopeOrder],
    *,
    bands: Mapping[str, int],
) -> Tuple[Tuple[Tuple[str, str], Tuple[str, str]], ...]:
    """Select exact scope/owner keys needing diagonal footprint authority."""

    if not isinstance(bands, Mapping):
        raise TypeError("aggregate refinement bands must be a mapping")

    axes_by_scope_pair: Dict[
        Tuple[Tuple[str, str], Tuple[str, str]], set[str]
    ] = {}
    for before_scope, after_scope, axis, before, after in orders:
        if len(before) != 1 or len(after) != 1:
            continue
        scope_pair = tuple(sorted((before_scope, after_scope)))
        owner_pair = tuple(sorted((before[0], after[0])))
        if (
            owner_pair[0] not in bands
            or owner_pair[1] not in bands
            or int(bands[owner_pair[0]]) == int(bands[owner_pair[1]])
        ):
            continue
        axes_by_scope_pair.setdefault((scope_pair, owner_pair), set()).add(axis)
    return tuple(sorted({
        key
        for key, axes in axes_by_scope_pair.items()
        if axes == {"horizontal", "vertical"}
    }))

def _orthogonal_single_master_aggregate_refinement_owners(
    orders: Sequence[_AggregateEnvelopeOrder],
    *,
    bands: Mapping[str, int],
) -> Tuple[str, ...]:
    """Select bounded owners for direct cross-tier diagonal relations."""

    return tuple(sorted({
        owner
        for _scopes, owners in _orthogonal_single_master_cross_tier_aggregate_keys(
            orders,
            bands=bands,
        )
        for owner in owners
    }))

def _layered_component_refinement_owners(
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder],
    *,
    bands: Mapping[str, int],
    fixed_pcell_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_placement_projection: object | None = None,
    master_members: Mapping[str, Sequence[str]] | None = None,
    rigid_nonmos_macro_plans: object | None = None,
    rigid_nonmos_placed_devices: Mapping[str, object] | None = None,
    block_member_devices: Mapping[str, object] | None = None,
    geometry: Mapping[str, object] | None = None,
    placement_um: Mapping[str, object] | None = None,
    block_stamps: Mapping[str, object] | None = None,
) -> Tuple[str, ...]:
    """Keep exact layer masks for relations and captured physical ownership."""

    if type(fixed_pcell_compound_footprints) is not tuple or any(
        type(row) is not tuple
        or len(row) != 2
        or type(row[0]) is not str
        or not row[0]
        for row in fixed_pcell_compound_footprints
    ):
        raise TypeError("fixed-PCell component refinement inventory is malformed")
    assembly_owners: set[str] = set()
    if layered_placement_projection is not None:
        from clay.layout.layered_placement_assembly import LayeredPlacementAssemblyBlockProjection

        if type(layered_placement_projection) is not LayeredPlacementAssemblyBlockProjection:
            raise TypeError("layered component refinement requires a captured block projection")
        layered_placement_projection.validate()
        if not isinstance(master_members, Mapping):
            raise TypeError("layered component refinement requires current master members")
        block_ids = {name for name, _box in layered_placement_projection.blocks}
        owners_by_block: dict[str, str] = {}
        for owner, members in master_members.items():
            if (type(owner) is not str or not owner
                or type(members) not in (tuple, list) or not members
                or any(type(member) is not str or member not in block_ids for member in members)):
                raise ValueError("layered refinement master member inventory is malformed")
            for member in members:
                if member in owners_by_block:
                    raise ValueError("layered refinement block has repeated master ownership")
                owners_by_block[member] = owner
        if set(owners_by_block) != block_ids:
            raise ValueError("layered refinement master member denominator differs")
        assembly_owners.update(owners_by_block[member]
            for member in layered_placement_projection.refinement_block_ids)
    return tuple(sorted({
        *_orthogonal_single_master_aggregate_refinement_owners(
            aggregate_envelope_orders,
            bands=bands,
        ),
        *(owner for owner, _rectangles in fixed_pcell_compound_footprints),
        *assembly_owners,
        *_native_nonmos_component_refinement_owners(rigid_nonmos_macro_plans,
            devices=rigid_nonmos_placed_devices, master_members=master_members,
            block_member_devices=block_member_devices, geometry=geometry,
            placement_um=placement_um, block_stamps=block_stamps),
    }))


def _native_nonmos_component_refinement_owners(plans, *, devices, master_members,
        block_member_devices, geometry=None, placement_um=None, block_stamps=None):
    """Join current source-bound artwork to its complete master/leaf partition.

    This selects exact occupied masks, not an overlap exemption. Ordinary rigid
    rectangle plans retain their existing guard. The shared collision projector
    carries the selected master's complete leaf set to every native consumer.
    """
    if plans is None:
        return ()
    from clay.layout.nonrow_macro_plan import RigidNonmosMacroPlanSet
    from clay.layout.source_bound_nonmos_plan import SourceBoundNonmosMacroPlan

    if type(plans) is not RigidNonmosMacroPlanSet:
        raise TypeError('non-MOS refinement requires an exact typed plan set')
    plans.__post_init__()
    native = tuple(plan for plan in plans.plans if type(plan) is SourceBoundNonmosMacroPlan)
    if not native:
        return ()
    if any(not isinstance(value, Mapping) for value in (devices, master_members, block_member_devices)):
        raise TypeError('non-MOS refinement requires current artwork and complete member mappings')
    from clay.layout.place.nonmos_emitter_ownership import canonical_current_nonmos_block_members
    block_member_devices = canonical_current_nonmos_block_members(plans,
        block_member_devices=block_member_devices, geometry=geometry,
        placement_um=placement_um, block_stamps=block_stamps)
    by_block, logical_by_block = {}, {}
    for master, blocks in master_members.items():
        if (type(master) is not str or not master or type(blocks) not in (tuple, list) or not blocks
                or any(type(block) is not str or not block for block in blocks)):
            raise ValueError('non-MOS refinement master member inventory is malformed')
        for block in blocks:
            if block in by_block:
                raise ValueError('non-MOS refinement leaf has repeated master ownership')
            logical = block_member_devices.get(block)
            if (type(logical) not in (tuple, list, set, frozenset) or not logical
                    or any(type(owner) is not str or not owner for owner in logical)
                    or len(set(logical)) != len(logical)):
                raise ValueError('non-MOS refinement leaf source member inventory is malformed')
            by_block[block], logical_by_block[block] = master, set(logical)
    refined = set()
    for plan in native:
        if plan.pdk_id != plans.tech_id:
            raise ValueError('non-MOS refinement assembly PDK differs from current plan set')
        plan.translation_nm(devices)  # Replays every actual member polygon and common integer frame.
        owned_blocks = set()
        for member in plan.members:
            hits = tuple(block for block, members in logical_by_block.items() if member in members)
            if len(hits) != 1:
                raise ValueError('non-MOS refinement source member denominator differs: '+member)
            owned_blocks.add(hits[0])
        owners = {by_block[block] for block in owned_blocks}
        if len(owners) != 1:
            raise ValueError('non-MOS refinement assembly is split across current masters')
        owner, = owners
        complete = [member for block in master_members[owner] for member in logical_by_block[block]]
        if len(complete) != len(set(complete)) or set(complete) != set(plan.members):
            raise ValueError('non-MOS refinement master differs from the complete assembly source members')
        refined.add(owner)
    return tuple(sorted(refined))


def _secondary_axis_aggregate_orders(
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder],
    *,
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder],
    bands: Mapping[str, int],
) -> Tuple[_AggregateEnvelopeOrder, ...]:
    """Return center orders deferred until after the primary DAG solve.

    The hierarchy order owns the primary non-overlap axis.  An aggregate center
    order on the same unordered node pair and the other axis is supplemental:
    it must be replayed as hard intent, but it need not participate in the
    constructive primary-DAG seed solve.  Likewise, a horizontal center order
    between two single-band aggregates is secondary when their distinct bands
    already establish the constructive vertical tier.  The one exception is a
    direct cross-tier singleton pair with both H and V reference relations: its
    component footprint and two linear center orders stay together in primary.
    """

    primary_axes: Dict[frozenset[str], set[str]] = {}
    for _scope, before_node, after_node, axis, _strength, _before, _after in (
        hierarchical_macro_orders
    ):
        primary_axes.setdefault(frozenset((before_node, after_node)), set()).add(axis)

    def uniform_band(members: Sequence[str]) -> int | None:
        values = {bands[member] for member in members if member in bands}
        return next(iter(values)) if len(values) == 1 else None

    primary_cross_tier_keys = set(
        _orthogonal_single_master_cross_tier_aggregate_keys(
            aggregate_envelope_orders,
            bands=bands,
        )
    )
    rows = []
    for row in aggregate_envelope_orders:
        pair = frozenset((row[0], row[1]))
        orthogonal_to_hierarchy = (
            pair in primary_axes and row[2] not in primary_axes[pair]
        )
        before_band = uniform_band(row[3])
        after_band = uniform_band(row[4])
        horizontal_across_tiers = (
            row[2] == "horizontal"
            and before_band is not None
            and after_band is not None
            and before_band != after_band
        )
        relation_key = (
            (
                tuple(sorted((row[0], row[1]))),
                tuple(sorted((row[3][0], row[4][0]))),
            )
            if len(row[3]) == 1 and len(row[4]) == 1
            else None
        )
        direct_cross_tier_pair = relation_key in primary_cross_tier_keys
        if (
            not direct_cross_tier_pair
            and (orthogonal_to_hierarchy or horizontal_across_tiers)
        ):
            rows.append(row)
    return tuple(rows)

def _secondary_axis_collision_y_scope(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    relation_members: Sequence[str],
    gap_nm: int,
) -> Tuple[str, ...]:
    """Return the finite seed corridor that may move during axis repair.

    X remains legalizer-owned for every block.  Y movement expands only through
    transitive overlap of the current vertical projections, which is the local
    corridor that can physically block an X-order swap.
    """

    known = set(seed)
    if set(dims) != known or not set(relation_members) <= known:
        raise ValueError("secondary-axis corridor scope has incomplete ownership")
    gap = int(gap_nm)
    if gap < 0:
        raise ValueError("secondary-axis corridor gap must be non-negative")
    scope = set(relation_members)
    changed = True
    while changed:
        changed = False
        for candidate in sorted(known - scope):
            candidate_y = seed[candidate][1]
            candidate_top = candidate_y + dims[candidate][1]
            if any(
                seed[member][1] + dims[member][1] + gap > candidate_y
                and candidate_top + gap > seed[member][1]
                for member in scope
            ):
                scope.add(candidate)
                changed = True
    return tuple(sorted(scope))

def _primary_hierarchical_macro_alignments(
    alignments: Sequence[_HierarchicalMacroAlignment],
) -> Tuple[_HierarchicalMacroAlignment, ...]:
    """Keep hierarchy-native bands and stage supplemental aggregate bands."""

    return tuple(
        row
        for row in alignments
        if (
            _hierarchical_macro_alignment_parts(row)[3] == 0
            or not _hierarchical_macro_alignment_parts(row)[0].startswith(
                SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
            )
        )
    )

def _canonical_hierarchical_macro_orders(
    values: object,
    *,
    block_ids: Sequence[str],
) -> Tuple[_HierarchicalMacroOrder, ...]:
    if type(values) is not tuple:
        raise TypeError("hierarchical macro orders must be one exact tuple")
    known = {str(name) for name in block_ids}
    rows: list[_HierarchicalMacroOrder] = []
    for raw in values:
        if type(raw) is not tuple or len(raw) != 7:
            raise TypeError(
                "hierarchical macro orders must be exact seven-item tuples"
            )
        scope, before_node, after_node, axis, strength, raw_before, raw_after = raw
        if (
            any(
                type(value) is not str or not value
                for value in (scope, before_node, after_node)
            )
            or before_node == after_node
            or axis not in {"horizontal", "vertical"}
            or (strength not in {
                "center_order",
                "center_order_with_track_floor",
                "envelope_separation",
                "member_before_aggregate_center",
                "member_before_aggregate_center_with_track_floor",
                "leading_edge_order",
                "leading_edge_equal",
            } and separation_gap_nm(strength) is None)
            or type(raw_before) is not tuple
            or type(raw_after) is not tuple
        ):
            raise ValueError("hierarchical macro order signature is malformed")
        before = tuple(sorted(str(name) for name in raw_before))
        after = tuple(sorted(str(name) for name in raw_after))
        if (
            not before
            or not after
            or before != raw_before
            or after != raw_after
            or len(set(before)) != len(before)
            or len(set(after)) != len(after)
            or set(before) & set(after)
            or not set((*before, *after)) <= known
        ):
            raise ValueError("hierarchical macro order ownership is malformed")
        rows.append(
            (scope, before_node, after_node, axis, strength, before, after)
        )
    canonical = tuple(sorted(rows))
    if canonical != values or len(canonical) != len(set(canonical)):
        raise ValueError("hierarchical macro orders must be canonical and unique")
    return canonical

def _hierarchical_macro_order_obligation_id(
    row: _HierarchicalMacroOrder,
) -> str:
    scope, before_node, after_node, axis, strength, _before, _after = row
    return (
        scope.removeprefix("reference_conditioned:")
        if scope.startswith("reference_conditioned:")
        else f"hierarchical_order:{scope}:{before_node}:{after_node}:{axis}:{strength}"
    )

def _hierarchical_full2d_pair_axis_authority(
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder],
) -> Tuple[Tuple[str, str, str, str, str, Tuple[str, ...]], ...]:
    """Lower unique hard macro separation directions with declaration IDs.

    Aggregate center/edge ordering is deliberately excluded: it does not prove
    that every cross-product pair is separated on that axis.  When independent
    H and V envelope relations govern the same pair, neither axis is privileged;
    the seed's deterministic assignment remains the embedding choice.
    """

    directions_by_pair: Dict[
        Tuple[str, str], set[Tuple[str, str, str]]
    ] = {}
    sources_by_direction: Dict[
        Tuple[Tuple[str, str], str, str, str], set[str]
    ] = {}
    for row in hierarchical_macro_orders:
        scope, before_node, after_node, axis, strength, before, after = row
        if separation_gap_nm(strength) is None:
            continue
        source_id = _hierarchical_macro_order_obligation_id(row)
        for first in before:
            for second in after:
                pair = tuple(sorted((str(first), str(second))))
                direction = (str(axis), str(first), str(second))
                directions_by_pair.setdefault(pair, set()).add(direction)
                sources_by_direction.setdefault((pair, *direction), set()).add(
                    source_id
                )
    rows = []
    for pair, directions in sorted(directions_by_pair.items()):
        if len(directions) != 1:
            continue
        axis, lower, upper = next(iter(directions))
        rows.append((
            pair[0],
            pair[1],
            axis,
            lower,
            upper,
            tuple(sorted(sources_by_direction[(pair, axis, lower, upper)])),
        ))
    return tuple(rows)

def _hierarchical_full2d_pair_axis_preference(
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder],
) -> Tuple[Tuple[str, str, str], ...]:
    """Lower unique non-separation topology to a soft graph-axis preference."""

    hard_pairs: set[Tuple[str, str]] = set()
    soft_axes_by_pair: Dict[Tuple[str, str], set[str]] = {}
    for row in hierarchical_macro_orders:
        _scope, _before_node, _after_node, axis, strength, before, after = row
        if separation_gap_nm(strength) is not None:
            for first in before:
                for second in after:
                    pair = tuple(sorted((str(first), str(second))))
                    hard_pairs.add(pair)
        elif len(before) == 1 and len(after) == 1:
            pair = tuple(sorted((str(before[0]), str(after[0]))))
            soft_axes_by_pair.setdefault(pair, set()).add(str(axis))
    return tuple(
        (pair[0], pair[1], next(iter(axes)))
        for pair, axes in sorted(soft_axes_by_pair.items())
        if pair not in hard_pairs and len(axes) == 1
    )

def _canonical_hierarchical_macro_alignments(
    values: object,
    *,
    block_ids: Sequence[str],
) -> Tuple[_HierarchicalMacroAlignment, ...]:
    if type(values) is not tuple:
        raise TypeError("hierarchical macro alignments must be one exact tuple")
    known = {str(name) for name in block_ids}
    rows: list[_HierarchicalMacroAlignment] = []
    for raw in values:
        if type(raw) is not tuple or len(raw) not in {3, 4, 5}:
            raise TypeError(
                "hierarchical macro alignments must be exact three-, four-, or five-item tuples"
            )
        if len(raw) == 3:
            scope, axis, raw_groups = raw
            edge = "center"
            tolerance_ppm = 0
        elif len(raw) == 4:
            scope, axis, edge, raw_groups = raw
            tolerance_ppm = 0
        else:
            scope, axis, edge, tolerance_ppm, raw_groups = raw
        if (
            type(scope) is not str
            or not scope
            or axis not in {"horizontal", "vertical"}
            or edge not in {"center", "leading", "trailing"}
            or type(tolerance_ppm) is not int
            or not 0 <= tolerance_ppm <= 1_000_000
            or type(raw_groups) is not tuple
            or len(raw_groups) < 2
        ):
            raise ValueError("hierarchical macro alignment signature is malformed")
        groups: list[Tuple[str, Tuple[str, ...]]] = []
        claimed: set[str] = set()
        for raw_group in raw_groups:
            if type(raw_group) is not tuple or len(raw_group) != 2:
                raise TypeError("hierarchical macro alignment group is malformed")
            node, raw_members = raw_group
            if type(node) is not str or not node or type(raw_members) is not tuple:
                raise ValueError("hierarchical macro alignment node is malformed")
            members = tuple(sorted(str(name) for name in raw_members))
            if (
                not members
                or members != raw_members
                or len(set(members)) != len(members)
                or set(members) & claimed
                or not set(members) <= known
            ):
                raise ValueError(
                    "hierarchical macro alignment ownership is malformed"
                )
            claimed.update(members)
            groups.append((node, members))
        canonical_groups = tuple(sorted(groups))
        if canonical_groups != raw_groups or len(
            {node for node, _members in groups}
        ) != len(groups):
            raise ValueError(
                "hierarchical macro alignment groups must be canonical and unique"
            )
        if tolerance_ppm:
            rows.append((scope, axis, edge, tolerance_ppm, canonical_groups))
        elif edge == "center":
            rows.append((scope, axis, canonical_groups))
        else:
            rows.append((scope, axis, edge, canonical_groups))
    canonical = tuple(
        sorted(rows, key=lambda row: _hierarchical_macro_alignment_parts(row))
    )
    if canonical != values or len(canonical) != len(set(canonical)):
        raise ValueError(
            "hierarchical macro alignments must be canonical and unique"
        )
    return canonical

def _hierarchical_macro_order_satisfied(
    blocks: Mapping[str, _Bbox],
    row: _HierarchicalMacroOrder,
    *,
    gap_nm: int,
) -> bool:
    _scope, _before_node, _after_node, axis, strength, before, after = row
    lower = 0 if axis == "horizontal" else 1
    upper = 2 if axis == "horizontal" else 3
    before_low = min(blocks[name][lower] for name in before)
    before_high = max(blocks[name][upper] for name in before)
    after_low = min(blocks[name][lower] for name in after)
    after_high = max(blocks[name][upper] for name in after)
    if separation_gap_nm(strength) is not None:
        return before_high + separation_gap_nm(strength) <= after_low
    if strength == "leading_edge_order":
        return before_low < after_low
    if strength == "leading_edge_equal":
        return before_low == after_low
    if strength == "member_before_aggregate_center":
        return max(
            blocks[name][lower] + blocks[name][upper] for name in before
        ) < after_low + after_high
    if strength == "member_before_aggregate_center_with_track_floor":
        return max(
            blocks[name][lower] + blocks[name][upper] for name in before
        ) + 2 * int(gap_nm) <= after_low + after_high
    if strength == "center_order_with_track_floor":
        return before_low + before_high + 2 * int(gap_nm) <= after_low + after_high
    return (before_low + before_high) // 2 < (after_low + after_high) // 2

def _hierarchical_macro_alignment_satisfied(
    blocks: Mapping[str, _Bbox],
    row: _HierarchicalMacroAlignment,
    *,
    tolerance_scales_nm: _HierarchicalAlignmentToleranceScalesNm = (),
) -> bool:
    _scope, axis, edge, tolerance_ppm, groups = (
        _hierarchical_macro_alignment_parts(row)
    )
    lower = 0 if axis == "horizontal" else 1
    upper = 2 if axis == "horizontal" else 3
    if edge == "leading":
        values = tuple(
            min(blocks[name][lower] for name in members)
            for _node, members in groups
        )
        scale = 1
    elif edge == "trailing":
        values = tuple(
            max(blocks[name][upper] for name in members)
            for _node, members in groups
        )
        scale = 1
    else:
        values = tuple(
            min(blocks[name][lower] for name in members)
            + max(blocks[name][upper] for name in members)
            for _node, members in groups
        )
        scale = 2
    if tolerance_ppm == 0:
        return len(set(values)) == 1
    scales_by_axis = dict(tolerance_scales_nm)
    if scales_by_axis and axis not in scales_by_axis:
        raise ValueError("hierarchical alignment tolerance scale is missing its axis")
    normalization_nm = scales_by_axis.get(axis)
    if normalization_nm is None:
        # Backward compatibility for signed evidence predating the v28 scale
        # inventory. New production certificates always supply the frozen scale.
        normalization_nm = max(box[upper] for box in blocks.values()) - min(
            box[lower] for box in blocks.values()
        )
    base = values[0]
    return all(
        abs(base - value) * 1_000_000
        <= scale * normalization_nm * tolerance_ppm
        for value in values[1:]
    )

def _add_aggregate_envelope_order_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    orders: Sequence[_AggregateEnvelopeOrder],
    *,
    width_bound_nm: int,
    height_bound_nm: int,
    center_step_nm: int,
    assumption_for: Optional[Callable[[str], Any]] = None,
) -> None:
    for index, (_before_scope, _after_scope, axis, before, after) in enumerate(orders):
        coordinates = xs if axis == "horizontal" else ys
        dimension = 0 if axis == "horizontal" else 1
        bound = int(width_bound_nm if axis == "horizontal" else height_bound_nm)
        before_low = model.NewIntVar(0, bound, f"aggregate_before_low_{index}")
        before_high = model.NewIntVar(0, bound, f"aggregate_before_high_{index}")
        after_low = model.NewIntVar(0, bound, f"aggregate_after_low_{index}")
        after_high = model.NewIntVar(0, bound, f"aggregate_after_high_{index}")
        model.AddMinEquality(before_low, [coordinates[name] for name in before])
        model.AddMaxEquality(
            before_high,
            [coordinates[name] + int(dims[name][dimension]) for name in before],
        )
        model.AddMinEquality(after_low, [coordinates[name] for name in after])
        model.AddMaxEquality(
            after_high,
            [coordinates[name] + int(dims[name][dimension]) for name in after],
        )
        step = 0 if _is_rail_halfplane_order(orders[index]) else 1
        constraint = model.Add(before_low + before_high + step <= after_low + after_high)
        if assumption_for is not None:
            constraint.OnlyEnforceIf(
                assumption_for(f"aggregate_order:{index}")
            )

def _add_hierarchical_macro_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    orders: Sequence[_HierarchicalMacroOrder],
    alignments: Sequence[_HierarchicalMacroAlignment],
    *,
    width_bound_nm: int,
    height_bound_nm: int,
    center_step_nm: int,
    gap_nm: int,
    tolerance_scales_nm: _HierarchicalAlignmentToleranceScalesNm = (),
    assumption_for: Optional[Callable[[str], Any]] = None,
) -> None:
    tolerance_scales_nm = _require_hierarchical_alignment_tolerance_scales_nm(
        tolerance_scales_nm,
        alignments=alignments,
    )
    envelopes: dict[Tuple[str, str, Tuple[str, ...]], Tuple[Any, Any]] = {}

    def envelope(axis: str, node: str, members: Tuple[str, ...]) -> Tuple[Any, Any]:
        key = (axis, node, members)
        existing = envelopes.get(key)
        if existing is not None:
            return existing
        coordinates = xs if axis == "horizontal" else ys
        dimension = 0 if axis == "horizontal" else 1
        bound = int(width_bound_nm if axis == "horizontal" else height_bound_nm)
        index = len(envelopes)
        low = model.NewIntVar(0, bound, f"macro_envelope_low_{index}")
        high = model.NewIntVar(0, bound, f"macro_envelope_high_{index}")
        model.AddMinEquality(low, [coordinates[name] for name in members])
        model.AddMaxEquality(
            high,
            [coordinates[name] + int(dims[name][dimension]) for name in members],
        )
        envelopes[key] = (low, high)
        return low, high

    for order_index, (
        _scope,
        before_node,
        after_node,
        axis,
        strength,
        before,
        after,
    ) in enumerate(orders):
        assumption = (
            assumption_for(f"hierarchy_order:{order_index}")
            if assumption_for is not None
            else None
        )

        def add_order_constraint(expression: Any) -> None:
            constraint = model.Add(expression)
            if assumption is not None:
                constraint.OnlyEnforceIf(assumption)

        before_low, before_high = envelope(axis, before_node, before)
        after_low, after_high = envelope(axis, after_node, after)
        if separation_gap_nm(strength) is not None:
            add_order_constraint(before_high + separation_gap_nm(strength) <= after_low)
        elif strength == "leading_edge_order":
            add_order_constraint(before_low + int(center_step_nm) <= after_low)
        elif strength == "leading_edge_equal":
            add_order_constraint(before_low == after_low)
        elif strength == "member_before_aggregate_center":
            for name in before:
                coordinates = xs if axis == "horizontal" else ys
                dimension = 0 if axis == "horizontal" else 1
                add_order_constraint(
                    2 * coordinates[name]
                    + int(dims[name][dimension])
                    + 2 * int(center_step_nm)
                    <= after_low + after_high
                )
        elif strength == "member_before_aggregate_center_with_track_floor":
            for name in before:
                coordinates = xs if axis == "horizontal" else ys
                dimension = 0 if axis == "horizontal" else 1
                add_order_constraint(
                    2 * coordinates[name]
                    + int(dims[name][dimension])
                    + 2 * int(gap_nm)
                    <= after_low + after_high
                )
        elif strength == "center_order_with_track_floor":
            add_order_constraint(
                before_low + before_high + 2 * int(gap_nm)
                <= after_low + after_high
            )
        else:
            add_order_constraint(
                before_low + before_high + 2 * int(center_step_nm)
                <= after_low + after_high
            )
    scales_by_axis = dict(tolerance_scales_nm)

    for alignment_index, row in enumerate(alignments):
        _scope, axis, edge, tolerance_ppm, groups = (
            _hierarchical_macro_alignment_parts(row)
        )
        assumption = (
            assumption_for(f"hierarchy_alignment:{alignment_index}")
            if assumption_for is not None
            else None
        )
        base_node, base_members = groups[0]
        base_low, base_high = envelope(axis, base_node, base_members)
        for node, members in groups[1:]:
            low, high = envelope(axis, node, members)
            if edge == "leading":
                base_value, value, scale = base_low, low, 1
            elif edge == "trailing":
                base_value, value, scale = base_high, high, 1
            else:
                base_value, value, scale = base_low + base_high, low + high, 2
            if tolerance_ppm == 0:
                constraint = model.Add(base_value == value)
                if assumption is not None:
                    constraint.OnlyEnforceIf(assumption)
            else:
                normalization_nm = scales_by_axis[axis]
                bound = int(
                    (width_bound_nm if axis == "horizontal" else height_bound_nm)
                    * scale
                )
                delta = model.NewIntVar(
                    -bound,
                    bound,
                    f"macro_alignment_delta_{alignment_index}_{node}",
                )
                magnitude = model.NewIntVar(
                    0,
                    bound,
                    f"macro_alignment_abs_{alignment_index}_{node}",
                )
                model.Add(delta == base_value - value)
                model.AddAbsEquality(magnitude, delta)
                constraint = model.Add(
                    magnitude * 1_000_000
                    <= int(scale * tolerance_ppm) * normalization_nm
                )
                if assumption is not None:
                    constraint.OnlyEnforceIf(assumption)

def _assert_aggregate_envelope_order_positions(
    positions: Mapping[str, Tuple[int, int]],
    dims: Mapping[str, Tuple[int, int]],
    orders: Sequence[_AggregateEnvelopeOrder],
) -> None:
    blocks = {
        name: (x, y, x + dims[name][0], y + dims[name][1])
        for name, (x, y) in positions.items()
    }
    if any(not _aggregate_envelope_order_satisfied(blocks, row) for row in orders):
        raise RuntimeError("hard aggregate envelope order has no certified incumbent")

def _assert_hierarchical_macro_positions(
    positions: Mapping[str, Tuple[int, int]],
    dims: Mapping[str, Tuple[int, int]],
    orders: Sequence[_HierarchicalMacroOrder],
    alignments: Sequence[_HierarchicalMacroAlignment],
    *,
    gap_nm: int,
    tolerance_scales_nm: _HierarchicalAlignmentToleranceScalesNm = (),
) -> None:
    tolerance_scales_nm = _require_hierarchical_alignment_tolerance_scales_nm(
        tolerance_scales_nm,
        alignments=alignments,
    )
    blocks = {
        name: (x, y, x + dims[name][0], y + dims[name][1])
        for name, (x, y) in positions.items()
    }
    if any(
        not _hierarchical_macro_order_satisfied(blocks, row, gap_nm=gap_nm)
        for row in orders
    ) or any(
        not _hierarchical_macro_alignment_satisfied(
            blocks,
            row,
            tolerance_scales_nm=tolerance_scales_nm,
        )
        for row in alignments
    ):
        raise RuntimeError("hard hierarchical macro relation has no certified incumbent")

def _secondary_axis_repair_admits_blockers(blockers: Sequence[str]) -> bool:
    """Admit repair only for one precisely classified structural failure."""

    return (
        len(blockers) == 1
        and blockers[0] in _SECONDARY_AXIS_REPAIRABLE_BLOCKERS
    )

def _hierarchical_cohesion_span_caps_nm(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    groups: Sequence[Sequence[str]],
) -> Tuple[Tuple[Tuple[str, ...], int, int], ...]:
    """Freeze monotone hierarchy-span upper bounds from one legal seed."""

    rows = []
    known = set(positions)
    if set(dims) != known:
        raise ValueError("hierarchy span caps require one exact placement partition")
    for raw_group in groups:
        group = tuple(raw_group)
        if (
            len(group) < 2
            or group != tuple(sorted(group))
            or len(set(group)) != len(group)
            or not set(group) <= known
        ):
            raise ValueError("hierarchy span cap group is not canonical and owned")
        x_span = (
            max(positions[name][0] + dims[name][0] for name in group)
            - min(positions[name][0] for name in group)
        )
        y_span = (
            max(positions[name][1] + dims[name][1] for name in group)
            - min(positions[name][1] for name in group)
        )
        rows.append((group, int(x_span), int(y_span)))
    canonical = tuple(sorted(rows))
    if len(canonical) != len(set(canonical)):
        raise ValueError("hierarchy span caps repeat one group")
    return canonical

def _hierarchical_cohesion_span_inventory(
    caps: Tuple[Tuple[Tuple[str, ...], int, int], ...],
) -> Tuple[Dict[str, object], ...]:
    return tuple(
        _full_hard_replay_row(
            "hierarchical_cohesion_span:"
            + ",".join(group)
            + f":{axis}",
            "hierarchical_cohesion_span_cap",
        )
        for group, _x_cap, _y_cap in caps
        for axis in ("horizontal", "vertical")
    )

def _hierarchical_cohesion_spans_within_caps(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    caps: Tuple[Tuple[Tuple[str, ...], int, int], ...],
    diagnostic_out: Optional[List[Dict[str, object]]] = None,
    hard_inventory_out: Optional[List[Dict[str, object]]] = None,
) -> bool:
    """Replay seed-monotone hierarchy spans for solver and compactor candidates."""

    inventory = _hierarchical_cohesion_span_inventory(caps)
    if hard_inventory_out is not None:
        hard_inventory_out.extend(inventory)
    try:
        observed = _hierarchical_cohesion_span_caps_nm(
            positions,
            dims=dims,
            groups=tuple(group for group, _x_cap, _y_cap in caps),
        )
    except (TypeError, ValueError, KeyError):
        if diagnostic_out is not None:
            diagnostic_out.extend(inventory)
        return False
    failed_rows = []
    for (observed_group, observed_x, observed_y), (
        cap_group,
        cap_x,
        cap_y,
    ) in zip(observed, caps):
        if observed_group != cap_group or observed_x > cap_x:
            failed_rows.append(_full_hard_replay_row(
                "hierarchical_cohesion_span:"
                + ",".join(cap_group)
                + ":horizontal",
                "hierarchical_cohesion_span_cap",
            ))
        if observed_group != cap_group or observed_y > cap_y:
            failed_rows.append(_full_hard_replay_row(
                "hierarchical_cohesion_span:"
                + ",".join(cap_group)
                + ":vertical",
                "hierarchical_cohesion_span_cap",
            ))
    if len(observed) != len(caps):
        failed_rows.extend(inventory[len(observed) * 2:])
    if diagnostic_out is not None:
        diagnostic_out.extend(failed_rows)
    return not failed_rows and len(observed) == len(caps)

def _hierarchical_vertical_leading_edge_seed(
    seed: Mapping[str, Tuple[int, int]],
    *,
    bands: Mapping[str, int],
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder],
    lateral_blocks: Sequence[str],
    manufacturing_grid_nm: int,
    bottom_center_keepout_nm: int,
    bottom_edge_keepout_nm: int,
    bottom_edge_keepout_exempt: Sequence[str],
) -> Dict[str, Tuple[int, int]]:
    """Make a vertical leading-edge relation constructive by rigid translation.

    Only a complete lateral subtree may move.  Prefer lowering the before
    subtree; when it is already pinned to the origin floor, raise the complete
    after subtree instead.  The result remains merely a CP hint until the
    independent hard-constraint replay proves every relation, clearance and
    outline obligation.
    """

    grid = int(manufacturing_grid_nm)
    if grid <= 0:
        raise ValueError("hierarchical leading-edge seed requires a positive grid")
    result = {
        str(name): (int(point[0]), int(point[1]))
        for name, point in seed.items()
    }
    lateral = {str(name) for name in lateral_blocks}
    exempt = {str(name) for name in bottom_edge_keepout_exempt}

    def lower_bound(name: str) -> int:
        lower = 0 if name in exempt else int(bottom_edge_keepout_nm)
        if name not in lateral and bands[name] == _BAND_CENTER:
            lower = max(lower, int(bottom_center_keepout_nm))
        return lower

    for row in hierarchical_macro_orders:
        _scope, _before_node, _after_node, axis, strength, before, after = row
        before_is_lateral = set(before) <= lateral
        after_is_lateral = set(after) <= lateral
        if (
            axis != "vertical"
            or strength != "leading_edge_order"
            or not (before_is_lateral or after_is_lateral)
        ):
            continue
        before_low = min(result[name][1] for name in before)
        after_low = min(result[name][1] for name in after)
        if before_low + grid <= after_low:
            continue
        if before_is_lateral:
            downward_delta = max(
                lower_bound(name) - result[name][1] for name in before
            )
            if (
                downward_delta % grid == 0
                and before_low + downward_delta + grid <= after_low
            ):
                for name in before:
                    x, y = result[name]
                    result[name] = (x, y + downward_delta)
                continue
        if after_is_lateral:
            upward_delta = before_low + grid - after_low
            upward_delta = _ceil_ratio(upward_delta, grid) * grid
            for name in after:
                x, y = result[name]
                result[name] = (x, y + upward_delta)
    return result

def _hierarchical_vertical_dag_seed(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    bands: Mapping[str, int],
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder],
    lateral_blocks: Sequence[str],
    manufacturing_grid_nm: int,
    gap_nm: int,
    bottom_center_keepout_nm: int,
    bottom_edge_keepout_nm: int,
    bottom_edge_keepout_exempt: Sequence[str],
) -> Dict[str, Tuple[int, int]]:
    """Construct one seed that consumes the complete vertical macro DAG.

    Each violated difference relation moves one complete successor macro upward;
    nested successor scopes therefore retain their internal geometry.  Repeated
    relaxation propagates a predecessor shift through parent and child levels.
    The result is still only a hint: the independent CP model and final replay
    remain the authority for every hierarchy relation and physical clearance.
    """

    result = {
        str(name): (int(point[0]), int(point[1]))
        for name, point in seed.items()
    }
    if set(result) != set(dims) or set(result) != set(bands):
        raise ValueError("hierarchical vertical DAG seed requires exact block cover")
    grid = int(manufacturing_grid_nm)
    gap = int(gap_nm)
    if grid <= 0 or gap < 0:
        raise ValueError("hierarchical vertical DAG seed has invalid physical units")
    vertical_orders = tuple(
        row for row in hierarchical_macro_orders if row[3] == "vertical"
    )
    if not vertical_orders:
        return result

    result = _hierarchical_vertical_leading_edge_seed(
        result,
        bands=bands,
        hierarchical_macro_orders=vertical_orders,
        lateral_blocks=lateral_blocks,
        manufacturing_grid_nm=grid,
        bottom_center_keepout_nm=bottom_center_keepout_nm,
        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
        bottom_edge_keepout_exempt=bottom_edge_keepout_exempt,
    )

    def boxes() -> Dict[str, _Bbox]:
        return {
            name: (
                point[0],
                point[1],
                point[0] + int(dims[name][0]),
                point[1] + int(dims[name][1]),
            )
            for name, point in result.items()
        }

    def translate_up(members: Sequence[str], delta_nm: int) -> None:
        delta = _ceil_ratio(int(delta_nm), grid) * grid
        if delta <= 0:
            return
        for name in members:
            x, y = result[name]
            result[name] = (x, y + delta)

    maximum_passes = len(vertical_orders) + len(result) + 1
    for _pass_index in range(maximum_passes):
        changed = False
        for row in vertical_orders:
            if _hierarchical_macro_order_satisfied(boxes(), row, gap_nm=gap):
                continue
            _scope, _before_node, _after_node, _axis, strength, before, after = row
            if strength == "leading_edge_order":
                adjusted = _hierarchical_vertical_leading_edge_seed(
                    result,
                    bands=bands,
                    hierarchical_macro_orders=(row,),
                    lateral_blocks=lateral_blocks,
                    manufacturing_grid_nm=grid,
                    bottom_center_keepout_nm=bottom_center_keepout_nm,
                    bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                    bottom_edge_keepout_exempt=bottom_edge_keepout_exempt,
                )
                if adjusted != result:
                    result = adjusted
                    changed = True
                if _hierarchical_macro_order_satisfied(
                    boxes(), row, gap_nm=gap
                ):
                    continue

            current = boxes()
            before_low = min(current[name][1] for name in before)
            before_high = max(current[name][3] for name in before)
            after_low = min(current[name][1] for name in after)
            after_high = max(current[name][3] for name in after)
            if strength == "leading_edge_equal":
                if before_low < after_low:
                    translate_up(before, after_low - before_low)
                else:
                    translate_up(after, before_low - after_low)
                changed = True
                continue
            if separation_gap_nm(strength) is not None:
                required_delta = before_high + separation_gap_nm(strength) - after_low
            elif strength in {
                "member_before_aggregate_center",
                "member_before_aggregate_center_with_track_floor",
            }:
                before_center2 = max(
                    2 * current[name][1] + int(dims[name][1]) for name in before
                )
                floor2 = (
                    2 * gap
                    if strength.endswith("with_track_floor")
                    else 2 * grid
                )
                required_delta = _ceil_ratio(
                    before_center2 + floor2 - (after_low + after_high),
                    2,
                )
            elif strength in {
                "center_order",
                "center_order_with_track_floor",
            }:
                floor2 = (
                    2 * gap
                    if strength.endswith("with_track_floor")
                    else 2 * grid
                )
                required_delta = _ceil_ratio(
                    before_low
                    + before_high
                    + floor2
                    - (after_low + after_high),
                    2,
                )
            else:
                required_delta = before_low + grid - after_low
            translate_up(after, required_delta)
            changed = True
        if not changed:
            final_boxes = boxes()
            if all(
                _hierarchical_macro_order_satisfied(
                    final_boxes, row, gap_nm=gap
                )
                for row in vertical_orders
            ):
                return result
            break
    raise RuntimeError("hierarchical vertical DAG seed did not converge")

def _hierarchical_composite_objective_value(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    placement_objective: str,
    hierarchical_cohesion_groups: Sequence[Sequence[str]],
    width_bound_nm: int,
    height_bound_nm: int,
    flexible_envelope_constraints: Tuple[FlexibleEnvelopeConstraint, ...] = (),
    bottom_edge_keepout_nm: int = 0,
) -> int:
    """Replay the exact lexicographic objective used by the banded CP model.

    A certified incumbent selected after solving must be reported in the same
    units as the solver objective and best bound.  Returning only its area or
    height would compare a primary extent against ``primary * M +
    hierarchy_span`` and make an otherwise legal fallback certificate
    non-authoritative.  The complete hierarchy is already hard; its span is
    only the bounded secondary tie-breaker and cannot trade a larger outline
    for tighter internal cohesion.
    """

    if placement_objective not in {"area", "height"}:
        raise ValueError("hierarchical composite objective is area or height only")
    if (
        type(width_bound_nm) is not int
        or type(height_bound_nm) is not int
        or width_bound_nm <= 0
        or height_bound_nm <= 0
    ):
        raise ValueError("hierarchical composite bounds must be positive exact ints")
    hierarchy_span = sum(
        (
            max(positions[name][0] + dims[name][0] for name in group)
            - min(positions[name][0] for name in group)
            + max(positions[name][1] + dims[name][1] for name in group)
            - min(positions[name][1] for name in group)
        )
        for group in hierarchical_cohesion_groups
    )
    primary = _placement_primary_extent_nm(
        positions,
        dims=dims,
        placement_objective=placement_objective,
        flexible_envelope_constraints=flexible_envelope_constraints,
        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
    )
    hierarchy_span_upper_bound = len(hierarchical_cohesion_groups) * (
        width_bound_nm + height_bound_nm
    )
    return int(primary) * (int(hierarchy_span_upper_bound) + 1) + int(
        hierarchy_span
    )
