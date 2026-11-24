"""Deterministic CP-SAT placement models, native acquisition, and solve orchestration."""
from __future__ import annotations
from clay.layout.place.placement_execution import cp_solve_phase
from clay.layout.place.seed_attempt import attempt_banded_seed, BandedSeedNotApplicable
from clay.layout.place.band_order_policy import resolve_band_order, add_native_band_constraints
from clay.layout.place.native_stage_observation import resolve_native_stage_observer, requires_complete_native_master
from clay.layout.place.optimization_target import AreaTarget, area_target_reached, refine_below_area_target, validate_area_target
from clay.flow.execution_budget import ExecutionStopped
from clay.flow.control_stop import CheckedControlStop
import hashlib
import math
import os
from itertools import combinations
from dataclasses import dataclass, replace
from clay.layout.mosaic_guest_slot_projection import MosaicGuestSlotProjection, add_mosaic_guest_slot_constraints, mosaic_guest_slot_satisfied
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from clay.layout.policy import aspect_policy_binding
from clay.layout.place.final_placement_state import canonical_minimum_structural_occupancy


def _contiguous_envelope_seed(seed, *, dims, bands, envelopes, columns, orders, alignments, gap_nm, grid_nm):
    """Pack each same-band functional envelope contiguously as a CP hint."""
    owner_columns = {}
    for _group, _row, column, _count, terms in columns:
        for owner, _count, _offset in terms:
            owner_columns.setdefault(owner, set()).add(column)
    by_band = {}
    grouped = set()
    for envelope in envelopes:
        members = tuple(envelope.members)
        band_set = {bands[name] for name in members}
        col_set = set().union(*(owner_columns.get(name, set()) for name in members))
        if len(band_set) != 1 or len(col_set) != 1 or grouped.intersection(members):
            return dict(seed)
        grouped.update(members)
        by_band.setdefault(next(iter(band_set)), []).append(
            (next(iter(col_set)), envelope.group, members))
    result = dict(seed)
    gap = ((gap_nm + grid_nm - 1) // grid_nm) * grid_nm
    floor = 0
    for band in sorted(set(bands.values())):
        groups = by_band.get(band, [])
        members_in_band = [name for name in seed if bands[name] == band]
        floor = max(floor, min(seed[name][1] for name in members_in_band))
        if not groups:
            shift = floor - min(seed[name][1] for name in members_in_band)
            for member in members_in_band:
                result[member] = (seed[member][0], seed[member][1] + shift)
            floor = max(result[name][1] + dims[name][1] for name in members_in_band) + gap
            continue
        cursor = 0
        for _column, _name, members in sorted(groups):
            ordered = tuple(sorted(members, key=lambda name: (seed[name][0], name)))
            area = sum((dims[name][0] + gap) * (dims[name][1] + gap) for name in ordered)
            width = max(max(dims[name][0] for name in ordered), math.isqrt(area))
            width = ((width + grid_nm - 1) // grid_nm) * grid_nm
            local, used_width, _height = _shelf_pack(ordered, dims, gap, width)
            for member in ordered:
                result[member] = (cursor + local[member][0], floor + local[member][1])
            cursor += used_width + gap
        for member in sorted((name for name in seed if bands[name] == band and name not in grouped),
                             key=lambda name: (seed[name][0], name)):
            result[member] = (cursor, floor)
            cursor += dims[member][0] + gap
        # A vertical group order is seeded with fully separated subgroups;
        # the original solver still certifies the authored center relation.
        vertical = [(before, after) for _a, _b, axis, before, after in orders
                    if axis == 'vertical' and set(before + after) <= set(members_in_band)]
        for _ in range(len(vertical)):
            for before, after in vertical:
                upper = max(result[name][1] + dims[name][1] for name in before)
                lower = min(result[name][1] for name in after)
                shift = max(0, upper + gap - lower)
                if shift:
                    for name in after:
                        result[name] = (result[name][0], result[name][1] + shift)
        floor = max(result[name][1] + dims[name][1] for name in members_in_band) + gap
    for raw in alignments:
        scope, axis, edge, tolerance, groups = _hierarchical_macro_alignment_parts(raw)
        if not scope.startswith(SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX + 'source_center_'):
            continue
        if axis != 'horizontal' or edge != 'center' or tolerance or len(groups) != 2:
            continue
        singleton = [members for _node, members in groups if len(members) == 1]
        aggregate = [members for _node, members in groups if len(members) > 1]
        if len(singleton) != 1 or len(aggregate) != 1:
            continue
        target = singleton[0][0]
        others = aggregate[0]
        left = min(result[name][0] for name in others)
        right_owner = max(others, key=lambda name: result[name][0] + dims[name][0])
        right = result[right_owner][0] + dims[right_owner][0]
        difference = left + right - dims[target][0]
        if difference % (2 * grid_nm):
            result[right_owner] = (result[right_owner][0] + grid_nm, result[right_owner][1])
            difference += grid_nm
        if difference >= 0 and difference % (2 * grid_nm) == 0:
            result[target] = (difference // 2, result[target][1])
    return result
from clay.layout.place.solver_execution_policy import apply_area_search_execution_policy, canonical_tie_break_enabled, incumbent_replay_enabled, prepare_incumbent_replay_model, require_incumbent_replay_response, _complete_incumbent_replay_stage
from clay.layout.placement_infeasibility import ConstraintObligationTrace, PlacementInfeasibilityError, SufficientCoreDiagnostic
from clay.layout.spatial.abstract_ir import SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
from clay.layout.place.constraints_hierarchical import (
    _add_aggregate_envelope_order_constraints,
    _add_hierarchical_macro_constraints,
    _aggregate_envelope_order_satisfied,
    _assert_aggregate_envelope_order_positions,
    _assert_hierarchical_macro_positions,
    _canonical_aggregate_envelope_orders,
    _canonical_hierarchical_macro_alignments,
    _canonical_hierarchical_macro_orders,
    _derive_hierarchical_alignment_tolerance_scales_nm,
    _hierarchical_cohesion_span_caps_nm,
    _hierarchical_cohesion_span_inventory,
    _hierarchical_cohesion_spans_within_caps,
    _hierarchical_composite_objective_value,
    _hierarchical_full2d_pair_axis_authority,
    _hierarchical_full2d_pair_axis_preference,
    _hierarchical_macro_alignment_parts,
    _hierarchical_macro_alignment_satisfied,
    _hierarchical_macro_order_obligation_id,
    _hierarchical_macro_order_satisfied,
    _hierarchical_vertical_dag_seed,
    _primary_hierarchical_macro_alignments,
    _require_hierarchical_alignment_tolerance_scales_nm,
    _secondary_axis_aggregate_orders,
    _secondary_axis_collision_y_scope,
    _secondary_axis_repair_admits_blockers,
)
from clay.layout.place.constraints_spatial import (
    FlexibleEnvelopeConstraint,
    FlexibleEnvelopeProjection,
    _add_adjacency_constraints,
    _add_axis_alignment_constraints,
    _add_bottom_edge_keepout_constraints,
    _add_center_alignment_constraints,
    _add_directional_aspect_constraints,
    _add_floorplan_column_constraints,
    _add_floorplan_column_group_constraints,
    _add_hard_horizontal_edge_constraints,
    _add_manufacturing_grid_constraints,
    _add_pair_clearance_constraints,
    _add_row_bounded_cavity_constraints,
    _add_top_edge_keepout_constraints,
    _assert_axis_alignment_positions,
    _assert_center_alignment_positions,
    _axis_alignment_required_width_nm,
    _banded_axis_aligned_seed,
    _banded_dense_seed,
    _banded_shelf_pack,
    _canonical_adjacency_limits_nm,
    _canonical_adjacency_scope_offsets_nm,
    _canonical_axis_alignment_groups,
    _canonical_center_alignment_pairs,
    _canonical_flexible_envelope_constraints,
    _canonical_floorplan_column_classes,
    _canonical_floorplan_column_groups,
    _canonical_layered_collision_footprints,
    _canonical_mosaic_guest_slot_projections,
    _canonical_overlap_compatible_pairs,
    _canonical_rigid_compound_footprints,
    _canonical_row_bounded_cavity_projections,
    _ceil_ratio,
    _center_aligned_lateral_seed,
    _collision_atoms_by_owner,
    _flexible_seed_physical_views,
    _floorplan_cell_cohesion_groups,
    _floorplan_column_group_order_satisfied,
    _floorplan_column_order_satisfied,
    _floorplan_seed,
    _hard_edge_compatible_banded_seed,
    _hard_order_width_nm,
    _pair_collision_forbidden_box_cover,
    _placement_primary_extent_nm,
    _prefer_certified_seed_positions,
    _project_flexible_adjacency_scope_offsets_nm,
    _required_pair_gap_nm,
    _right_edge_compatible_width_bound,
    _row_bounded_cavity_satisfied,
    _scaled_aspect_bounds,
    _seed_geometry_legal,
    _shelf_pack,
    _stable_hard_order,
    _strict_outline_aspect_height_floor_nm,
)
from clay.layout.place.constraints_well_guard import (
    _canonical_pair_clearances_nm,
)
from clay.layout.place.full2d_compactor import (
    _compact_area_frontier_positions,
    _compact_full2d_master_positions,
    _compact_lateral_sink_positions_x,
    _compaction_replay_width_bound,
)
from clay.layout.place.placement_certificates import (
    FinalPlacementCertificate,
    PlaceBlocks2DResult,
    PlacementSolveCutResidualError,
    _SolverEvidence,
    _placement_result,
    final_placement_state_digest,
)
from clay.layout.place.placement_constants import (
    _ALIGNMENT_SCALE_SOURCE_CERTIFICATE,
    _ASPECT_TOL,
    _AggregateEnvelopeOrder,
    _BAND_CENTER,
    _BAND_TOP,
    _Bbox,
    _FILL_SEED,
    _FloorplanColumnGroup,
    _GLOBAL_FLOORPLAN_COLUMN_SCOPE,
    _HierarchicalAlignmentToleranceScalesNm,
    _HierarchicalMacroAlignment,
    _HierarchicalMacroOrder,
    _LayeredCollisionFootprint,
    _MAX_ITERS,
    _RigidCompoundFootprint,
    _RowBoundedCavityProjection,
)
from clay.layout.place.placement_cuts import (
    _PreparedSolveCut,
    _add_solve_cut_constraints,
    _cut_extent_padding,
    _exact_cut_input_blocks,
    _prepare_solve_cuts,
    _required_cut_axis_extent,
)
from clay.layout.place.placement_diagnostics import (
    _canonical_full_hard_failure_rows,
    _diagnostic_canonical_value,
)
from clay.layout.place.placement_replay import (
    _banded_seed_satisfies_hard_constraints,
    _require_edge_rail_reservation,
)


@dataclass(frozen=True)
class _DeterministicBudgetLedger:
    """Named upper bounds whose exact sum never exceeds one requested budget."""

    requested: float
    allocations: Tuple[Tuple[str, float], ...]

    @classmethod
    def split(
        cls, requested: float, labels: Sequence[str]
    ) -> "_DeterministicBudgetLedger":
        total = float(requested)
        names = tuple(labels)
        if not math.isfinite(total) or total < 0.0:
            raise ValueError("max_time_s must be a finite non-negative budget")
        if (
            not names
            or any(not isinstance(label, str) or not label for label in names)
            or len(names) != len(set(names))
        ):
            raise ValueError("deterministic budget labels must be unique strings")
        share = total / len(names)
        amounts = [share for _label in names[:-1]]
        final = max(0.0, total - math.fsum(amounts))
        if math.fsum((*amounts, final)) > total:
            final = math.nextafter(final, 0.0)
        amounts.append(final)
        return cls(total, tuple(zip(names, amounts)))

    def for_phase(self, label: str) -> float:
        for phase, amount in self.allocations:
            if phase == label:
                return amount
        raise KeyError(label)

    @property
    def allocated(self) -> float:
        return math.fsum(amount for _label, amount in self.allocations)

def _replace_flexible_envelope_member_bboxes(
    blocks: Any,
    constraints: Any,
    content_bboxes_nm: Any,
) -> Tuple[Tuple[str, _Bbox], ...]:
    """Replace only proven shared-ring owners by their exact content bboxes."""

    if type(blocks) is not tuple:
        raise TypeError("flexible envelope source blocks must be one exact tuple")
    source_rows = tuple(_exact_cut_input_blocks(blocks))
    source = dict(source_rows)
    canonical = _canonical_flexible_envelope_constraints(
        constraints,
        block_ids=tuple(source),
    )
    expected = {
        member for constraint in canonical for member in constraint.members
    }
    if type(content_bboxes_nm) is not tuple:
        raise TypeError("flexible envelope content bboxes must be one exact tuple")
    rows: list[Tuple[str, _Bbox]] = []
    for row in content_bboxes_nm:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("flexible envelope content rows must be exact tuples")
        name, bbox = row
        if type(name) is not str or not name:
            raise TypeError("flexible envelope content owner must be one string")
        if (
            type(bbox) is not tuple
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise TypeError("flexible envelope content bbox must be one exact bbox")
        rows.append((name, bbox))
    canonical_rows = tuple(sorted(rows))
    if tuple(rows) != canonical_rows or len({name for name, _bbox in rows}) != len(rows):
        raise ValueError("flexible envelope content bboxes must be canonical and unique")
    content = dict(canonical_rows)
    if set(content) != expected:
        raise ValueError("flexible envelope content bboxes must cover exact members")
    for name, bbox in canonical_rows:
        outer = source[name]
        if not (
            outer[0] <= bbox[0] < bbox[2] <= outer[2]
            and outer[1] <= bbox[1] < bbox[3] <= outer[3]
        ):
            raise ValueError("flexible envelope content bbox leaves its source ring")
        constraint = next(row for row in canonical if name in row.members)
        expected_physical_offset = (
            outer[0] - bbox[0],
            outer[1] - bbox[1],
            outer[2] - bbox[0],
            outer[3] - bbox[1],
        )
        if dict(constraint.member_physical_offsets_nm)[name] != expected_physical_offset:
            raise ValueError(
                "flexible envelope physical offset differs from its source ring"
            )
    return tuple(
        (name, content.get(name, bbox))
        for name, bbox in source_rows
    )

def _solver_evidence(
    solver: Any, status: Any, *, objective_name: str = "height_nm"
) -> _SolverEvidence:
    from ortools.sat.python import cp_model

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _SolverEvidence("FALLBACK", objective_name, None, None, True, "seed")
    return _SolverEvidence(
        "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        objective_name,
        float(solver.ObjectiveValue()),
        float(solver.BestObjectiveBound()),
        False,
    )

def _configure_2d_packing_proof(solver: Any) -> None:
    """Enable exact CP-SAT propagation specialized for rectangle packing."""

    parameters = getattr(solver, "parameters", None)
    if parameters is None:
        raise TypeError("2-D packing solver must expose CP-SAT parameters")
    parameters.use_energetic_reasoning_in_no_overlap_2d = True
    parameters.use_timetabling_in_no_overlap_2d = True
    parameters.use_pairwise_reasoning_in_no_overlap_2d = True

def _canonical_source_seed_positions(
    value: Optional[Mapping[str, Tuple[int, int]]],
    *,
    ids: Sequence[str],
) -> Optional[Dict[str, Tuple[int, int]]]:
    """Validate the exact prior-master origins used as feedback CP hints."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(ids):
        raise ValueError("source seed positions must cover the exact master inventory")
    result: Dict[str, Tuple[int, int]] = {}
    for block_id in ids:
        point = value[block_id]
        if (
            not isinstance(point, (tuple, list))
            or len(point) != 2
            or any(
                type(coordinate) is not int or coordinate < 0 for coordinate in point
            )
        ):
            raise ValueError(
                "source seed positions must contain exact non-negative integer origins"
            )
        result[block_id] = (point[0], point[1])
    return result

def _repair_source_seed_for_cuts(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    cuts: Sequence[_PreparedSolveCut],
) -> Dict[str, Tuple[int, int]]:
    """Make the prior-master hint satisfy each new cut by a minimal translation.

    The result is only a CP hint, never accepted as proof.  The full master still
    rechecks no-overlap, structural constraints, aspect, and every signed cut.
    Starting with a cut-satisfying hint avoids asking CP-SAT to repair a known
    conflict before it can discover its first incumbent.
    """

    result = dict(seed)
    for cut in cuts:
        if cut.incumbent_origins:
            block_id = cut.incumbent_origins[0][0]
            x, y = result[block_id]
            result[block_id] = (x + 1, y)
            continue
        a, b = cut.owner_a, cut.owner_b
        ax, ay = result[a]
        bx, by = result[b]
        aw, ah = dims[a]
        bw, bh = dims[b]
        clearance = int(cut.required_clearance_nm)
        if cut.corridor_relation == "horizontal_separation":
            if ax <= bx:
                result[b] = (max(bx, ax + aw + clearance), by)
            else:
                result[a] = (max(ax, bx + bw + clearance), ay)
        elif cut.corridor_relation == "vertical_separation":
            if ay <= by:
                result[b] = (bx, max(by, ay + ah + clearance))
            else:
                result[a] = (ax, max(ay, by + bh + clearance))
    return result

def _select_feedback_seed(
    natural_seed: Mapping[str, Tuple[int, int]],
    source_seed: Optional[Mapping[str, Tuple[int, int]]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    cuts: Sequence[_PreparedSolveCut],
    gap_nm: int,
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
) -> Dict[str, Tuple[int, int]]:
    """Prefer a legal cut-repaired source hint, then its legal incumbent."""

    if source_seed is None:
        return dict(natural_seed)
    repaired = _repair_source_seed_for_cuts(source_seed, dims=dims, cuts=cuts)
    if _seed_geometry_legal(
        repaired,
        dims=dims,
        gap_nm=gap_nm,
        pair_clearances_nm=pair_clearances_nm,
        rigid_compound_footprints=rigid_compound_footprints,
        layered_collision_footprints=layered_collision_footprints,
        overlap_compatible_pairs=overlap_compatible_pairs,
    ):
        return repaired
    if _seed_geometry_legal(
        source_seed,
        dims=dims,
        gap_nm=gap_nm,
        pair_clearances_nm=pair_clearances_nm,
        rigid_compound_footprints=rigid_compound_footprints,
        layered_collision_footprints=layered_collision_footprints,
        overlap_compatible_pairs=overlap_compatible_pairs,
    ):
        return dict(source_seed)
    return dict(natural_seed)

def _strip_solve(
    ids: Sequence[str],
    dims: Dict[str, Tuple[int, int]],
    gap: int,
    w_cap: int,
    max_time_s: float,
    bottom_edge_keepout_nm: int = 0,
    bottom_edge_keepout_exempt: Sequence[str] = (),
    solve_cuts: Sequence[_PreparedSolveCut] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
    source_seed_positions: Optional[Mapping[str, Tuple[int, int]]] = None,
    aspect_target: float = 1.67,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    manufacturing_grid_nm: int = 1,
    right_edge: Sequence[str] = (),
    left_edge: Sequence[str] = (),
) -> Tuple[Dict[str, Tuple[int, int]], int, int, _SolverEvidence]:
    """One deterministic CP-SAT STRIP-PACK: fix the outline WIDTH to ``w_cap`` and
    MINIMIZE the height. Fixing the width (rather than searching an area outline for
    bare feasibility) turns the packing into an easy optimization CP-SAT drives to a
    tight incumbent fast -- the shelf pack (width ``w_cap``) is an immediately-valid
    hint, so an incumbent always exists. Returns ``(pos, W, H)`` in nm (``W`` = the
    achieved max right edge, ``<= w_cap``; ``H`` = the minimized height)."""
    from ortools.sat.python import cp_model

    w_cap = max(w_cap, max(dims[b][0] for b in ids))
    w_cap = max(
        w_cap + _cut_extent_padding(solve_cuts, "horizontal_separation", gap),
        _required_cut_axis_extent(solve_cuts, dims, "horizontal_separation"),
    )
    seed_gap = max((gap, *pair_clearances_nm.values()))
    exempt = set(bottom_edge_keepout_exempt or ())
    seed, _sw, seed_h = _shelf_pack(ids, dims, seed_gap, w_cap)
    seed = {
        bid: (
            xy[0],
            xy[1] if bid in exempt else max(xy[1], int(bottom_edge_keepout_nm)),
        )
        for bid, xy in seed.items()
    }
    source_seed = _canonical_source_seed_positions(
        source_seed_positions,
        ids=ids,
    )
    if source_seed is not None:
        seed = _select_feedback_seed(
            seed,
            source_seed,
            dims=dims,
            cuts=solve_cuts,
            gap_nm=gap,
            pair_clearances_nm=pair_clearances_nm,
        )
        w_cap = max(w_cap, *(seed[bid][0] + dims[bid][0] for bid in ids))
    seed_h = max(seed[bid][1] + dims[bid][1] for bid in ids)
    total = sum(dims[b][0] * dims[b][1] for b in ids)
    max_h = max(dims[b][1] for b in ids)
    h_bound = max(
        max(total * 3 // max(w_cap, 1), seed_h)
        + int(bottom_edge_keepout_nm)
        + max_h
        + 10 * seed_gap
        + _cut_extent_padding(solve_cuts, "vertical_separation", gap),
        _required_cut_axis_extent(solve_cuts, dims, "vertical_separation"),
    )
    w_cap = _right_edge_compatible_width_bound(
        w_cap, dims, right_edge, manufacturing_grid_nm
    )

    model = cp_model.CpModel()
    xs: Dict[str, Any] = {}
    ys: Dict[str, Any] = {}
    x_iv: List[Any] = []
    y_iv: List[Any] = []
    H = model.NewIntVar(0, h_bound, "H")
    for bid in ids:  # ids sorted by caller -> input-order-invariant model
        w, h = dims[bid]
        x0 = model.NewIntVar(0, w_cap - w, f"x_{bid}")
        y0 = model.NewIntVar(0, h_bound - h, f"y_{bid}")
        xs[bid], ys[bid] = x0, y0
        x_iv.append(model.NewFixedSizeIntervalVar(x0, w + gap, f"xi_{bid}"))
        y_iv.append(model.NewFixedSizeIntervalVar(y0, h + gap, f"yi_{bid}"))
        model.Add(H >= y0 + h)
        model.AddHint(x0, seed[bid][0])
        model.AddHint(y0, seed[bid][1])
    model.AddNoOverlap2D(x_iv, y_iv)
    _add_bottom_edge_keepout_constraints(
        model, ys, bottom_edge_keepout_nm, exempt
    )
    _add_manufacturing_grid_constraints(
        model, xs, ys, manufacturing_grid_nm
    )
    _add_hard_horizontal_edge_constraints(
        model,
        xs,
        dims,
        width_bound=w_cap,
        right_edge=right_edge,
        left_edge=left_edge,
    )
    _add_directional_aspect_constraints(
        model,
        xs=xs,
        ys=ys,
        dims=dims,
        width_bound=w_cap,
        height_bound=h_bound,
        aspect_target=aspect_target,
        aspect_bounds=aspect_bounds,
        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
    )
    _add_pair_clearance_constraints(model, xs, ys, dims, pair_clearances_nm)
    _add_solve_cut_constraints(model, xs, ys, dims, solve_cuts)
    model.Minimize(H)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    _configure_2d_packing_proof(solver)
    solver.parameters.max_deterministic_time = float(max_time_s)
    with cp_solve_phase(solver, model, phase="placement.strip.solve.unbanded"):
        st = solver.Solve(model)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        pos = {
            bid: (int(solver.Value(xs[bid])), int(solver.Value(ys[bid]))) for bid in ids
        }
    elif right_edge or left_edge:
        raise RuntimeError(
            "hard horizontal edge anchor has no certified incumbent "
            f"(solver_status={solver.StatusName(st)})"
        )
    else:  # nothing better than the seed within budget -> the honest fallback
        pos = dict(seed)
    w_used = max(pos[b][0] + dims[b][0] for b in ids)
    h_used = max(pos[b][1] + dims[b][1] for b in ids)
    return pos, w_used, h_used, _solver_evidence(solver, st)

class _NativeStageAcquisitionStop(CheckedControlStop):
    """Capture-only stop; no placement or solver certificate is implied."""

    def __init__(self, *, stage_inputs: Mapping[str, object], upstream_trace: Mapping[str, object]):
        super().__init__("current native stage acquired before local primary solve")
        self.stage_inputs = stage_inputs
        self.upstream_trace = upstream_trace
        self.local_execution: Mapping[str, object] = {}

def _native_stage_frame_fields() -> frozenset[str]:
    import inspect

    return frozenset(inspect.signature(_NATIVE_STAGE_DECLARATIONS[0]).parameters) - {
        "native_stage_observer",
    }

def _native_full_stage_replay_kwargs(
    frame: Mapping[str, object], *, width_bound_nm: int, height_bound_nm: int,
) -> Dict[str, object]:
    """One complete outer replay argument projection from the current frame."""
    import inspect

    if set(frame) != _native_stage_frame_fields():
        raise ValueError("complete current native frame fields differ")
    result = {
        "ids": frame["ids"], "dims": frame["dims"], "gap_nm": frame["gap"],
        "width_bound_nm": int(width_bound_nm), "height_bound_nm": int(height_bound_nm),
        "bands": frame["bands"], "adjacency_limits_nm": frame["adjacency_limits_nm"],
        "adjacency_scope_offsets_nm": frame["adjacency_scope_offsets_nm"],
        "order_pairs": frame["order_pairs"], "vertical_order_pairs": frame["vertical_order_pairs"],
        "floorplan_column_classes": frame["floorplan_column_classes"],
        "floorplan_column_groups": frame["floorplan_column_groups"],
        "bottom_center_keepout_nm": frame["bottom_center_keepout_nm"],
        "bottom_edge_keepout_nm": frame["bottom_edge_keepout_nm"],
        "bottom_edge_keepout_exempt": frame["bottom_edge_keepout_exempt"],
        "top_edge_keepout_nm": frame["top_edge_keepout_nm"],
        "top_edge_keepout_clear": frame["top_edge_keepout_clear"],
        "solve_cuts": frame["solve_cuts"], "pair_clearances_nm": frame["pair_clearances_nm"],
        "axis_alignment_groups": frame["axis_alignment_groups"],
        "center_alignment_pairs": frame["center_alignment_pairs"],
        "aggregate_envelope_orders": frame["aggregate_envelope_orders"],
        "hierarchical_macro_orders": frame["hierarchical_macro_orders"],
        "hierarchical_macro_alignments": frame["hierarchical_macro_alignments"],
        "hierarchical_alignment_tolerance_scales_nm": frame["hierarchical_alignment_tolerance_scales_nm"],
        "aspect_target": frame["aspect_target"], "aspect_bounds": frame["aspect_bounds"],
        "aspect_obligation": frame["aspect_obligation"],
        "manufacturing_grid_nm": frame["manufacturing_grid_nm"],
        "right_edge": frame["right_edge"], "left_edge": frame["left_edge"],
        "placement_objective": frame["placement_objective"],
        "strict_width_cap": frame["strict_width_cap"],
        "exact_outline_width": frame["exact_outline_width"],
        "lateral_blocks": frame["lateral_blocks"],
        "flexible_envelope_constraints": frame["flexible_envelope_constraints"],
        "rigid_compound_footprints": frame["rigid_compound_footprints"],
        "layered_collision_footprints": frame["layered_collision_footprints"],
        "row_bounded_cavity_projections": frame["row_bounded_cavity_projections"],
        "mosaic_guest_slot_projections": frame["mosaic_guest_slot_projections"],
        "edge_rail_reservation": frame["edge_rail_reservation"],
        "overlap_compatible_pairs": frame["overlap_compatible_pairs"],
        "enforce_legacy_band_order": frame["enforce_legacy_band_order"],
    }
    expected = {name for name in inspect.signature(_NATIVE_STAGE_DECLARATIONS[2]).parameters
                if name != "seed" and not name.endswith("_out")}
    if set(result) != expected:
        raise ValueError("complete native outer replay fields differ")
    return result

def _native_fixed_outline_stage_inputs(
    *, frame: Mapping[str, object], build_policy: Mapping[str, object],
) -> Dict[str, object]:
    """Project current physical operands plus an explicitly identified seed policy.

    This does not claim the seed came from a primary Solve in this invocation.
    No old geometry, reference coordinates, or missing native defaults are read.
    """
    import inspect

    policy_fields = {"seed", "width_nm", "height_nm", "movable_y",
                     "movable_y_lower_bounds_nm", "bottom_origin_lower_bounds_nm",
                     "max_time_s", "warm_start_origin"}
    if not isinstance(build_policy, Mapping) or set(build_policy) != policy_fields:
        raise ValueError("complete explicit native build policy fields differ")
    seed = build_policy["seed"]
    if (not isinstance(seed, Mapping) or not seed or set(seed) != set(frame["dims"])
            or any(not isinstance(point, (tuple, list)) or len(point) != 2
                   or any(type(value) is not int for value in point) for point in seed.values())):
        raise ValueError("explicit native warm-start owner/coordinate inventory differs")
    if not isinstance(build_policy["warm_start_origin"], Mapping) or not build_policy["warm_start_origin"]:
        raise ValueError("explicit native warm-start provenance is required")
    outer = _native_full_stage_replay_kwargs(
        frame, width_bound_nm=build_policy["width_nm"], height_bound_nm=build_policy["height_nm"],
    )
    shared = (
        "dims", "bands", "gap_nm", "manufacturing_grid_nm", "floorplan_column_classes",
        "floorplan_column_groups", "aggregate_envelope_orders", "hierarchical_macro_orders",
        "hierarchical_macro_alignments", "hierarchical_alignment_tolerance_scales_nm",
        "rigid_compound_footprints", "layered_collision_footprints", "row_bounded_cavity_projections",
        "mosaic_guest_slot_projections", "edge_rail_reservation",
        "overlap_compatible_pairs", "pair_clearances_nm", "axis_alignment_groups", "center_alignment_pairs",
        "order_pairs", "vertical_order_pairs", "adjacency_limits_nm", "adjacency_scope_offsets_nm",
        "right_edge", "left_edge", "aspect_target", "aspect_bounds", "aspect_obligation",
    )
    kwargs = {name: outer[name] for name in shared}
    kwargs.update({name: build_policy[name] for name in (
        "width_nm", "height_nm", "movable_y", "movable_y_lower_bounds_nm",
        "bottom_origin_lower_bounds_nm", "max_time_s",
    )})
    kwargs.update({"hierarchical_cohesion_groups": frame["hierarchical_cohesion_groups"],
                   "exact_outline": frame["exact_outline_width"]})
    expected = {name for name in inspect.signature(_NATIVE_STAGE_DECLARATIONS[1]).parameters
                if name != "seed" and not name.endswith("_sink")}
    if set(kwargs) != expected:
        raise ValueError("complete native legalizer input fields differ")
    return {
        "schema": "clay.current-native-fixed-outline-stage/v1",
        "input_origin": "CURRENT_NATIVE_FRAME_PLUS_EXPLICIT_BUILD_POLICY",
        "warm_start_observation": "NOT_ASSERTED_BY_INPUT_FACTORY",
        "promotion": "REJECT", "g4_status": "NOT_RUN",
        "frame": frame, "build_policy": build_policy,
        "seed": seed, "kwargs": kwargs, "outer_kwargs": outer,
        "outer_width_policy": "NATIVE_COMPACTION_REPLAY_WIDTH_BOUND",
    }

def _strip_solve_banded(
    ids: Sequence[str],
    dims: Dict[str, Tuple[int, int]],
    gap: int,
    w_cap: int,
    bands: Dict[str, int],
    right_pull: Sequence[str],
    left_pull: Sequence[str],
    pull_pairs: Sequence[Tuple[str, str]],
    max_time_s: float,
    cohesion_groups: Tuple[Tuple[str, ...], ...] = (),
    well_cohesion_groups: Tuple[Tuple[str, ...], ...] = (),
    hierarchical_cohesion_groups: Tuple[Tuple[str, ...], ...] = (),
    flexible_envelope_constraints: Tuple[FlexibleEnvelopeConstraint, ...] = (),
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    row_bounded_cavity_projections: Tuple[_RowBoundedCavityProjection, ...] = (),
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = (),
    edge_rail_reservation: Any = None,
    adjacency_limits_nm: Sequence[Tuple[str, str, int]] = (),
    adjacency_scope_offsets_nm: Mapping[str, _Bbox] = MappingProxyType({}),
    order_pairs: Sequence[Tuple[str, str]] = (),
    vertical_order_pairs: Sequence[Tuple[str, str]] = (),
    lateral_blocks: Sequence[str] = (),
    floorplan_column_classes: Sequence[Tuple[str, int, int, int]] = (),
    floorplan_column_groups: Sequence[_FloorplanColumnGroup] = (),
    bottom_center_keepout_nm: int = 0,
    bottom_edge_keepout_nm: int = 0,
    bottom_edge_keepout_exempt: Sequence[str] = (),
    top_edge_keepout_nm: int = 0,
    top_edge_keepout_clear: Sequence[str] = (),
    solve_cuts: Sequence[_PreparedSolveCut] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]] = (),
    center_alignment_pairs: Sequence[Tuple[str, str, str]] = (),
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder] = (),
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder] = (),
    hierarchical_macro_alignments: Sequence[_HierarchicalMacroAlignment] = (),
    hierarchical_alignment_tolerance_scales_nm: (
        _HierarchicalAlignmentToleranceScalesNm
    ) = (),
    pack_mode: str = "rows",
    aspect_target: float = 1.67,
    source_seed_positions: Optional[Mapping[str, Tuple[int, int]]] = None,
    certified_source_seed: bool = False,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = True,
    manufacturing_grid_nm: int = 1,
    right_edge: Sequence[str] = (),
    left_edge: Sequence[str] = (),
    placement_objective: str = "height",
    strict_width_cap: bool = False,
    exact_outline_width: bool = False,
    native_stage_observer: Optional[Callable[[Mapping[str, object]], None]] = None,
    area_target: Optional[AreaTarget] = None,
    enforce_legacy_band_order: Optional[bool] = None,
) -> Tuple[Dict[str, Tuple[int, int]], int, int, _SolverEvidence]:
    """One deterministic CP-SAT strip pack with legacy or hierarchical topology.
    An unspecified band policy retains the historical default: full-width band
    seams unless a hierarchical DAG is present. Explicit True adds band seams
    alongside the DAG; False keeps full-2D collisions without band seams.
    A SOFT objective term (strictly below the
    height minimum, so
    it never trades envelope for position) pulls the ``right_pull``/``left_pull``
    blocks toward their declared horizontal edge and each ``pull_pairs`` block
    toward its partner's x. Returns
    ``(pos, W, H)`` in nm. ``strict_width_cap=True`` prohibits every heuristic
    expansion of the requested domain. ``exact_outline_width=True`` additionally
    fixes the normalized physical bbox (minimum x zero, maximum right edge
    ``w_cap``); exact-outline enumeration requires both contracts."""
    from ortools.sat.python import cp_model
    validate_area_target(area_target)
    enforce_legacy_band_order = resolve_band_order(
        enforce_legacy_band_order, hierarchical_macro_orders=hierarchical_macro_orders)
    hierarchical_alignment_tolerance_scales_nm = (
        _require_hierarchical_alignment_tolerance_scales_nm(
            hierarchical_alignment_tolerance_scales_nm,
            alignments=hierarchical_macro_alignments,
        )
    )
    if type(placement_objective) is not str:
        raise TypeError("placement_objective must be an exact string")
    if placement_objective not in {"area", "height"}:
        raise ValueError(
            "placement_objective must be one of ['area', 'height'], "
            f"got {placement_objective!r}"
        )
    if type(strict_width_cap) is not bool:
        raise TypeError("strict_width_cap must be an exact boolean")
    if type(exact_outline_width) is not bool:
        raise TypeError("exact_outline_width must be an exact boolean")
    if type(certified_source_seed) is not bool:
        raise TypeError("certified_source_seed must be an exact boolean")
    if certified_source_seed and source_seed_positions is None:
        raise ValueError("certified_source_seed requires exact source positions")
    replay_incumbent = incumbent_replay_enabled()
    replay_seed = (
        _canonical_source_seed_positions(source_seed_positions, ids=ids)
        if replay_incumbent else None
    )
    if replay_incumbent and replay_seed is None:
        raise ValueError("incumbent replay requires complete supplied source positions")
    if exact_outline_width and not strict_width_cap:
        raise ValueError("exact_outline_width requires strict_width_cap")
    if strict_width_cap and type(w_cap) is not int:
        raise TypeError("strict width cap must be an exact integer nm value")
    if strict_width_cap and w_cap <= 0:
        raise ValueError("strict width cap must be positive")
    remaining_time_s = float(max_time_s)
    if not math.isfinite(remaining_time_s) or remaining_time_s < 0.0:
        raise ValueError("max_time_s must be a finite non-negative budget")
    canonical_overlap_pairs = _canonical_overlap_compatible_pairs(
        overlap_compatible_pairs,
        block_ids=tuple(ids),
        pair_clearances_nm=pair_clearances_nm,
    )
    overlap_compatible_set = set(canonical_overlap_pairs)
    right_pull = sorted(set(right_pull or ()))
    left_pull = sorted(set(left_pull or ()))
    pull_pairs = sorted({(str(a), str(b)) for a, b in (pull_pairs or ())})
    vertical_order_pairs = tuple(
        sorted({(str(lower), str(upper)) for lower, upper in vertical_order_pairs or ()})
    )
    if any(
        lower == upper or lower not in set(ids) or upper not in set(ids)
        for lower, upper in vertical_order_pairs
    ):
        raise ValueError("vertical order pairs must name two distinct live blocks")
    lateral = set(str(name) for name in lateral_blocks)
    if not lateral <= set(ids):
        raise ValueError("banded lateral blocks contain an unknown master")
    core_ids = tuple(name for name in ids if name not in lateral)
    if not core_ids:
        raise ValueError("banded lateral ownership consumed every master")
    core_id_set = set(core_ids)
    core_order_pairs = tuple(
        (left, right)
        for left, right in order_pairs
        if left in core_id_set and right in core_id_set
    )
    core_vertical_order_pairs = tuple(
        (lower, upper)
        for lower, upper in vertical_order_pairs
        if lower in core_id_set and upper in core_id_set
    )
    if type(cohesion_groups) is not tuple:
        raise TypeError("cohesion groups must be one exact tuple")
    canonical_cohesion: list[Tuple[str, ...]] = []
    cohesion_owners: set[str] = set()
    id_set = set(ids)
    canonical_compound_footprints = _canonical_rigid_compound_footprints(
        rigid_compound_footprints,
        dims=dims,
    )
    compound_owners = {owner for owner, _rectangles in canonical_compound_footprints}
    canonical_layered_footprints = _canonical_layered_collision_footprints(
        layered_collision_footprints,
        dims=dims,
    )
    canonical_cavity_projections = _canonical_row_bounded_cavity_projections(
        row_bounded_cavity_projections,
        dims=dims,
    )
    canonical_slot_projections = _canonical_mosaic_guest_slot_projections(
        mosaic_guest_slot_projections, dims=dims, minimum_clearance_nm=gap,
    )
    layered_owners = {owner for owner, _atoms in canonical_layered_footprints}
    collision_atoms = _collision_atoms_by_owner(
        dims=dims,
        rigid_compound_footprints=canonical_compound_footprints,
        layered_collision_footprints=canonical_layered_footprints,
    )
    for raw_group in cohesion_groups:
        if (
            type(raw_group) is not tuple
            or len(raw_group) < 2
            or any(type(owner) is not str or not owner for owner in raw_group)
        ):
            raise TypeError(
                "cohesion groups must contain exact non-empty string tuples"
            )
        group = tuple(sorted(raw_group))
        if raw_group != group:
            raise ValueError("cohesion groups must use canonical sorted order")
        if len(group) != len(set(group)) or any(owner not in id_set for owner in group):
            raise ValueError("cohesion groups repeat or name a foreign master")
        if cohesion_owners.intersection(group):
            raise ValueError("one master cannot belong to multiple cohesion groups")
        cohesion_owners.update(group)
        canonical_cohesion.append(group)
    # The outer order is the canonical floorplan-cell order already established
    # by ``_floorplan_cell_cohesion_groups``.  Preserve it verbatim: sorting by
    # owner names would replace that authority with an unrelated lexical order.
    cohesion_groups = tuple(canonical_cohesion)
    if type(well_cohesion_groups) is not tuple:
        raise TypeError("well cohesion groups must be one exact tuple")
    canonical_well_cohesion: list[Tuple[str, ...]] = []
    well_cohesion_owners: set[str] = set()
    for raw_group in well_cohesion_groups:
        if (
            type(raw_group) is not tuple
            or len(raw_group) < 2
            or any(type(owner) is not str or not owner for owner in raw_group)
        ):
            raise TypeError(
                "well cohesion groups must contain exact non-empty string tuples"
            )
        group = tuple(sorted(raw_group))
        if raw_group != group:
            raise ValueError("well cohesion groups must use canonical sorted order")
        if len(group) != len(set(group)) or any(owner not in id_set for owner in group):
            raise ValueError("well cohesion groups repeat or name a foreign master")
        if well_cohesion_owners.intersection(group):
            raise ValueError("one master cannot belong to multiple well cohesion groups")
        well_cohesion_owners.update(group)
        canonical_well_cohesion.append(group)
    well_cohesion_groups = tuple(canonical_well_cohesion)
    if type(hierarchical_cohesion_groups) is not tuple:
        raise TypeError("hierarchical cohesion groups must be one exact tuple")
    canonical_hierarchical_cohesion: list[Tuple[str, ...]] = []
    for raw_group in hierarchical_cohesion_groups:
        if (
            type(raw_group) is not tuple
            or len(raw_group) < 2
            or any(type(owner) is not str or not owner for owner in raw_group)
        ):
            raise TypeError(
                "hierarchical cohesion groups must contain exact string tuples"
            )
        group = tuple(sorted(raw_group))
        if raw_group != group:
            raise ValueError(
                "hierarchical cohesion groups must use canonical sorted order"
            )
        if len(group) != len(set(group)) or any(owner not in id_set for owner in group):
            raise ValueError(
                "hierarchical cohesion groups repeat or name a foreign master"
            )
        canonical_hierarchical_cohesion.append(group)
    if len(set(canonical_hierarchical_cohesion)) != len(
        canonical_hierarchical_cohesion
    ):
        raise ValueError("hierarchical cohesion groups must be unique")
    hierarchical_cohesion_groups = tuple(canonical_hierarchical_cohesion)
    nonhierarchical_structured_groups = (
        *cohesion_groups,
        *well_cohesion_groups,
        *(tuple(name for name, _offset in group) for group in axis_alignment_groups),
    )
    # A non-hierarchical rigid/axis structure cannot be split between the band
    # master and its lateral extension.  An explicit hierarchy node is different:
    # it is the declared cross-boundary ownership contract.  The complete CP
    # master contains both sets of coordinates, applies every hierarchy order and
    # alignment below, and independently replays them before certification, so a
    # hierarchy node may intentionally contain both banded and lateral children.
    if any(
        lateral.intersection(group)
        for group in nonhierarchical_structured_groups
    ):
        raise ValueError(
            "banded lateral blocks cannot cross a cohesion or axis structure"
        )
    flexible_envelope_constraints = _canonical_flexible_envelope_constraints(
        flexible_envelope_constraints,
        block_ids=tuple(ids),
        bands=bands,
    )
    if any(lateral.intersection(row.members) for row in flexible_envelope_constraints):
        raise ValueError("banded lateral blocks cannot cross a flexible envelope")
    flexible_physical_offsets: Dict[str, Dict[str, _Bbox]] = {}
    for row in flexible_envelope_constraints:
        offsets = dict(row.member_physical_offsets_nm)
        for member in row.members:
            x0, y0, x1, y1 = offsets[member]
            width, height = dims[member]
            if not (x0 <= 0 < width <= x1 and y0 <= 0 < height <= y1):
                raise ValueError(
                    "flexible envelope physical footprint must contain its content"
                )
        flexible_physical_offsets[row.group] = offsets
    flexible_foreign_clearances: Dict[str, Dict[Tuple[str, str], int]] = {}
    for row in flexible_envelope_constraints:
        required_by_foreign: Dict[Tuple[str, str], int] = {}
        for member in row.members:
            for foreign in ids:
                if foreign in row.members:
                    continue
                required_by_foreign[(member, foreign)] = max(
                    row.foreign_clearance_nm,
                    pair_clearances_nm.get(tuple(sorted((member, foreign))), gap),
                )
        flexible_foreign_clearances[row.group] = required_by_foreign
    present = sorted(set(bands[bid] for bid in core_ids))
    hierarchical_topology_active = bool(hierarchical_macro_orders)
    exempt = set(bottom_edge_keepout_exempt or ())
    top_clear = set(top_edge_keepout_clear or ())
    def current_stage_frame() -> Dict[str, object]:
        # Preserve every current caller input, including fields not lowered by
        # the fixed-outline model. The full outer checker remains mandatory.
        return {
            "ids": ids, "dims": dims, "gap": gap, "w_cap": w_cap, "bands": bands,
            "right_pull": right_pull, "left_pull": left_pull, "pull_pairs": pull_pairs,
            "max_time_s": max_time_s, "cohesion_groups": cohesion_groups,
            "well_cohesion_groups": well_cohesion_groups,
            "hierarchical_cohesion_groups": hierarchical_cohesion_groups,
            "flexible_envelope_constraints": flexible_envelope_constraints,
            "rigid_compound_footprints": canonical_compound_footprints,
            "layered_collision_footprints": canonical_layered_footprints,
            "row_bounded_cavity_projections": canonical_cavity_projections,
            "mosaic_guest_slot_projections": canonical_slot_projections,
            "edge_rail_reservation": edge_rail_reservation,
            "adjacency_limits_nm": adjacency_limits_nm,
            "adjacency_scope_offsets_nm": adjacency_scope_offsets_nm,
            "order_pairs": order_pairs, "vertical_order_pairs": vertical_order_pairs,
            "lateral_blocks": tuple(sorted(lateral)),
            "floorplan_column_classes": floorplan_column_classes,
            "floorplan_column_groups": floorplan_column_groups,
            "bottom_center_keepout_nm": bottom_center_keepout_nm,
            "bottom_edge_keepout_nm": bottom_edge_keepout_nm,
            "bottom_edge_keepout_exempt": tuple(sorted(exempt)),
            "top_edge_keepout_nm": top_edge_keepout_nm,
            "top_edge_keepout_clear": tuple(sorted(top_clear)), "solve_cuts": solve_cuts,
            "pair_clearances_nm": pair_clearances_nm,
            "overlap_compatible_pairs": canonical_overlap_pairs,
            "axis_alignment_groups": axis_alignment_groups,
            "center_alignment_pairs": center_alignment_pairs,
            "aggregate_envelope_orders": aggregate_envelope_orders,
            "hierarchical_macro_orders": hierarchical_macro_orders,
            "hierarchical_macro_alignments": hierarchical_macro_alignments,
            "hierarchical_alignment_tolerance_scales_nm": hierarchical_alignment_tolerance_scales_nm,
            "pack_mode": pack_mode, "aspect_target": aspect_target,
            "source_seed_positions": source_seed_positions,
            "certified_source_seed": certified_source_seed,
            "aspect_bounds": aspect_bounds, "aspect_obligation": aspect_obligation,
            "manufacturing_grid_nm": manufacturing_grid_nm,
            "right_edge": right_edge, "left_edge": left_edge,
            "placement_objective": placement_objective, "area_target": area_target,
            "strict_width_cap": strict_width_cap, "exact_outline_width": exact_outline_width,
            "enforce_legacy_band_order": enforce_legacy_band_order,
        }
    _require_edge_rail_reservation(edge_rail_reservation, dims=dims)

    if replay_incumbent:
        from clay.layout.place.generated_incumbent import require_current_generated_incumbent_frame
        require_current_generated_incumbent_frame(current_stage_frame())
    if (native_stage_observer := resolve_native_stage_observer(native_stage_observer)) is not None:
        if not callable(native_stage_observer):
            raise TypeError("native stage observer must be callable")
        frame = current_stage_frame()
        before = _diagnostic_canonical_value(frame)
        try:
            native_stage_observer(frame)
        except _NativeStageAcquisitionStop as stopped:
            if (_diagnostic_canonical_value(current_stage_frame()) != before
                    or _diagnostic_canonical_value(stopped.stage_inputs["frame"]) != before):
                raise ValueError("native stage observer changed current physical frame")
            expected_stage = _native_fixed_outline_stage_inputs(
                frame=frame, build_policy=stopped.stage_inputs["build_policy"],
            )
            if _diagnostic_canonical_value(stopped.stage_inputs) != _diagnostic_canonical_value(expected_stage):
                raise ValueError("native stage observer returned a different input projection")
            stopped.local_execution = {
                "phase": "CANONICAL_FRAME_BEFORE_LOCAL_PRIMARY_OR_MAIN_CP",
                "local_solve_count_before_observer": 0,
                "observer_execution": "NOT_INSTRUMENTED_BY_NATIVE_SEAM",
                "upstream_execution": "CALLER_TRACE_ONLY_NOT_COUNTED_HERE",
            }
            raise
        raise RuntimeError("native stage observer must stop via _NativeStageAcquisitionStop")

    secondary_axis_orders = _secondary_axis_aggregate_orders(
        aggregate_envelope_orders,
        hierarchical_macro_orders=hierarchical_macro_orders,
        bands=bands,
    )
    secondary_axis_members = tuple(
        sorted(
            {
                member
                for row in secondary_axis_orders
                for members in (row[3], row[4])
                for member in members
            }
            | {name for row in canonical_slot_projections for name in (row.host_master, row.guest_master)}
        )
    )
    if secondary_axis_orders and remaining_time_s > 0.0 and not replay_incumbent:
        secondary_set = set(secondary_axis_orders)
        primary_aggregate_orders = tuple(
            row for row in aggregate_envelope_orders if row not in secondary_set
        )
        primary_hierarchical_alignments = _primary_hierarchical_macro_alignments(
            hierarchical_macro_alignments
        )
        primary_budget = remaining_time_s * 0.85
        try:
            primary_seed, primary_w, primary_h, primary_evidence = (
                _strip_solve_banded(
                    ids,
                    dims,
                    gap,
                    w_cap,
                    bands,
                    right_pull=right_pull,
                    left_pull=left_pull,
                    pull_pairs=pull_pairs,
                    max_time_s=primary_budget,
                    cohesion_groups=cohesion_groups,
                    well_cohesion_groups=well_cohesion_groups,
                    hierarchical_cohesion_groups=hierarchical_cohesion_groups,
                    flexible_envelope_constraints=flexible_envelope_constraints,
                    rigid_compound_footprints=canonical_compound_footprints,
                    layered_collision_footprints=canonical_layered_footprints,
                    row_bounded_cavity_projections=canonical_cavity_projections,
                    mosaic_guest_slot_projections=canonical_slot_projections,
                    edge_rail_reservation=edge_rail_reservation,
                    adjacency_limits_nm=adjacency_limits_nm,
                    adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
                    order_pairs=order_pairs,
                    vertical_order_pairs=vertical_order_pairs,
                    lateral_blocks=tuple(sorted(lateral)),
                    floorplan_column_classes=floorplan_column_classes,
                    floorplan_column_groups=floorplan_column_groups,
                    bottom_center_keepout_nm=bottom_center_keepout_nm,
                    bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                    bottom_edge_keepout_exempt=bottom_edge_keepout_exempt,
                    top_edge_keepout_nm=top_edge_keepout_nm,
                    top_edge_keepout_clear=top_edge_keepout_clear,
                    solve_cuts=solve_cuts,
                    pair_clearances_nm=pair_clearances_nm,
                    overlap_compatible_pairs=canonical_overlap_pairs,
                    axis_alignment_groups=axis_alignment_groups,
                    center_alignment_pairs=center_alignment_pairs,
                    aggregate_envelope_orders=primary_aggregate_orders,
                    hierarchical_macro_orders=hierarchical_macro_orders,
                    hierarchical_macro_alignments=primary_hierarchical_alignments,
                    hierarchical_alignment_tolerance_scales_nm=tuple(
                        (axis, scale_nm)
                        for axis, scale_nm in (
                            hierarchical_alignment_tolerance_scales_nm
                        )
                        if axis
                        in {
                            _hierarchical_macro_alignment_parts(row)[1]
                            for row in primary_hierarchical_alignments
                            if _hierarchical_macro_alignment_parts(row)[3] > 0
                        }
                    ),
                    pack_mode=pack_mode,
                    aspect_target=aspect_target,
                    source_seed_positions=source_seed_positions,
                    certified_source_seed=certified_source_seed,
                    aspect_bounds=aspect_bounds,
                    aspect_obligation=aspect_obligation,
                    manufacturing_grid_nm=manufacturing_grid_nm,
                    right_edge=right_edge,
                    left_edge=left_edge,
                    placement_objective=placement_objective,
                    strict_width_cap=strict_width_cap,
                    exact_outline_width=exact_outline_width,
                    area_target=area_target,
                    enforce_legacy_band_order=enforce_legacy_band_order,
                )
            )
        except ExecutionStopped:
            raise
        except RuntimeError:
            pass
        else:
            stage_width_bound = max(int(w_cap), int(primary_w))
            stage_height_bound = max(
                int(primary_h),
                max(primary_seed[name][1] + dims[name][1] for name in ids),
            )
            latest_full_replay_audit: Dict[str, object] = {
                "failed_rows": (),
                "inventory": (),
                "spec": {},
            }

            def replay_full_stage(
                candidate: Mapping[str, Tuple[int, int]],
                *,
                width_bound_nm: int = stage_width_bound,
                height_bound_nm: int = stage_height_bound,
            ) -> Tuple[bool, Tuple[str, ...]]:
                blockers: List[str] = []
                failed_rows: List[Dict[str, object]] = []
                inventory: List[Dict[str, object]] = []
                hard_spec: Dict[str, object] = {}
                try:
                    valid = _banded_seed_satisfies_hard_constraints(
                        candidate,
                        **_native_full_stage_replay_kwargs(
                            current_stage_frame(), width_bound_nm=width_bound_nm,
                            height_bound_nm=height_bound_nm,
                        ),
                        blocker_out=blockers,
                        diagnostic_out=failed_rows,
                        hard_inventory_out=inventory,
                        hard_spec_out=hard_spec,
                    )
                finally:
                    latest_full_replay_audit.update({
                        "failed_rows": _canonical_full_hard_failure_rows(
                            failed_rows
                        ),
                        "inventory": _canonical_full_hard_failure_rows(
                            inventory
                        ),
                        "spec": hard_spec,
                    })
                return valid, tuple(blockers)

            full_valid, full_replay_blockers = replay_full_stage(primary_seed)
            if full_valid:
                return (
                    primary_seed,
                    int(primary_w),
                    int(primary_h),
                    replace(
                        primary_evidence,
                        status="FEASIBLE",
                        best_bound=None,
                        used_fallback=False,
                        fallback_kind=None,
                    ),
                )
            repaired_seed = None
            repair_error = None
            repair_infeasibility = None
            repair_budget = max(0.0, remaining_time_s - primary_budget)
            if (
                _secondary_axis_repair_admits_blockers(full_replay_blockers)
                and repair_budget > 0.0
            ):
                secondary_movable_y = _secondary_axis_collision_y_scope(
                    primary_seed,
                    dims=dims,
                    relation_members=secondary_axis_members,
                    gap_nm=gap,
                )
                secondary_y_lower_bounds = {
                    name: max(
                        0 if name in exempt else int(bottom_edge_keepout_nm),
                        (
                            int(bottom_center_keepout_nm)
                            if name not in lateral and bands[name] == _BAND_CENTER
                            else 0
                        ),
                    )
                    for name in ids
                }
                repair_scope_gap = max((gap, *pair_clearances_nm.values()))
                repair_width_bound = int(stage_width_bound)
                if not strict_width_cap:
                    repair_width_bound += sum(
                        int(dims[name][0]) for name in secondary_movable_y
                    ) + repair_scope_gap * max(0, len(secondary_movable_y) - 1)
                    repair_width_bound = _right_edge_compatible_width_bound(
                        repair_width_bound,
                        dims,
                        right_edge,
                        manufacturing_grid_nm,
                    )
                repair_height_bound = int(stage_height_bound) + sum(
                    int(dims[name][1]) for name in secondary_movable_y
                ) + repair_scope_gap * max(0, len(secondary_movable_y) - 1)

                def replay_repaired_stage(
                    candidate: Mapping[str, Tuple[int, int]],
                ) -> Tuple[bool, Tuple[str, ...]]:
                    realized_width = _compaction_replay_width_bound(
                        candidate,
                        dims=dims,
                        configured_width_bound_nm=repair_width_bound,
                        strict_width_cap=strict_width_cap,
                        exact_outline_width=exact_outline_width,
                    )
                    return replay_full_stage(
                        candidate,
                        width_bound_nm=realized_width,
                        height_bound_nm=repair_height_bound,
                    )

                try:
                    repair_inputs = _native_fixed_outline_stage_inputs(
                        frame=current_stage_frame(),
                        build_policy={
                            "seed": primary_seed,
                            "width_nm": repair_width_bound,
                            "height_nm": repair_height_bound,
                            "movable_y": secondary_movable_y,
                            "movable_y_lower_bounds_nm": {
                                name: secondary_y_lower_bounds[name]
                                for name in secondary_movable_y
                            },
                            "bottom_origin_lower_bounds_nm": (
                                secondary_y_lower_bounds
                                if not flexible_envelope_constraints
                                else MappingProxyType({})
                            ),
                            "max_time_s": repair_budget,
                            "warm_start_origin": {"kind": "CURRENT_NATIVE_PRIMARY_RESULT"},
                        },
                    )
                    repaired_seed = _legalize_fixed_outline_seed_x(
                        repair_inputs["seed"], **repair_inputs["kwargs"],
                    )
                except PlacementInfeasibilityError as exc:
                    repair_error = str(exc) or type(exc).__name__
                    repair_infeasibility = exc.diagnostic
                    repaired_seed = None
                except ExecutionStopped:
                    raise
                except RuntimeError as exc:
                    repair_error = str(exc) or type(exc).__name__
                    repaired_seed = None
            if repaired_seed is not None:
                repaired_valid, repaired_blockers = replay_repaired_stage(repaired_seed)
                if repaired_valid:
                    repaired_solver_evidence: list[Mapping[str, object]] = []
                    repaired_seed, repaired_graph, _repaired_compacted = (
                        refine_below_area_target(_compact_full2d_master_positions,
                            repaired_seed,
                            target=area_target, hard_replayed=repaired_valid,
                            area_nm2=_placement_primary_extent_nm(repaired_seed, dims=dims, placement_objective="area",
                                flexible_envelope_constraints=flexible_envelope_constraints, bottom_edge_keepout_nm=bottom_edge_keepout_nm),
                            dims=dims,
                            minimum_clearance_nm=gap,
                            manufacturing_grid_nm=manufacturing_grid_nm,
                            axis_alignment_groups=axis_alignment_groups,
                            center_alignment_pairs=center_alignment_pairs,
                            pair_axis_authority=(
                                _hierarchical_full2d_pair_axis_authority(
                                    hierarchical_macro_orders
                                )
                            ),
                            pair_axis_preference=(
                                _hierarchical_full2d_pair_axis_preference(
                                    hierarchical_macro_orders
                                )
                            ),
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
                            overlap_compatible_pairs=canonical_overlap_pairs,
                            rigid_compound_footprints=canonical_compound_footprints,
                            layered_collision_footprints=canonical_layered_footprints,
                            row_bounded_cavity_projections=canonical_cavity_projections,
                            mosaic_guest_slot_projections=canonical_slot_projections,
                            edge_rail_reservation=edge_rail_reservation,
                            order_pairs=order_pairs,
                            vertical_order_pairs=vertical_order_pairs,
                            pair_clearances_nm=pair_clearances_nm,
                            hierarchical_cohesion_span_caps_nm=(),
                            floorplan_column_groups=floorplan_column_groups,
                            left_edge=left_edge,
                            legal=lambda candidate: replay_repaired_stage(candidate)[0],
                            legal_replay_audit=lambda: dict(
                                latest_full_replay_audit
                            ),
                            solve_evidence_sink=repaired_solver_evidence,
                        )
                    )
                    repaired_valid, repaired_blockers = replay_repaired_stage(
                        repaired_seed
                    )
                if repaired_valid:
                    repaired_width = _compaction_replay_width_bound(
                        repaired_seed,
                        dims=dims,
                        configured_width_bound_nm=repair_width_bound,
                        strict_width_cap=strict_width_cap,
                        exact_outline_width=exact_outline_width,
                    )
                    repaired_height = max(
                        int(repaired_seed[name][1]) + int(dims[name][1])
                        for name in ids
                    )
                    return (
                        repaired_seed,
                        int(repaired_width),
                        int(repaired_height),
                        replace(
                            primary_evidence,
                            status="FEASIBLE",
                            best_bound=None,
                            used_fallback=False,
                            fallback_kind=None,
                            full2d_compaction_graph=repaired_graph,
                            full2d_solver_evidence=tuple(repaired_solver_evidence),
                        ),
                    )
                full_replay_blockers = repaired_blockers
            reported_seed = repaired_seed if repaired_seed is not None else primary_seed
            primary_boxes = {
                name: (
                    reported_seed[name][0],
                    reported_seed[name][1],
                    reported_seed[name][0] + dims[name][0],
                    reported_seed[name][1] + dims[name][1],
                )
                for name in ids
            }
            violated_secondary = tuple(
                (row[0], row[1], row[2])
                for row in secondary_axis_orders
                if not _aggregate_envelope_order_satisfied(primary_boxes, row)
            )
            replay_message = (
                "secondary-axis primary seed failed full hard replay: "
                f"blockers={tuple(full_replay_blockers)!r}, "
                f"violated_secondary={violated_secondary!r}, "
                f"x_legalizer_error={repair_error!r}, "
                f"primary_outline_nm={(stage_width_bound, stage_height_bound)!r}, "
                "layered_footprints="
                f"{hashlib.sha256(repr(canonical_layered_footprints).encode()).hexdigest()!r}, "
                "layered_by_owner="
                f"{dict((owner, hashlib.sha256(repr(atoms).encode()).hexdigest()[:12]) for owner, atoms in canonical_layered_footprints)!r}"
            )
            if repair_infeasibility is not None:
                raise PlacementInfeasibilityError(
                    replay_message,
                    repair_infeasibility,
                )
            raise RuntimeError(replay_message)
        remaining_time_s = max(0.0, remaining_time_s - primary_budget)

    seed_gap = max((gap, *pair_clearances_nm.values()))
    band_h = {
        b: max((dims[bid][1] for bid in core_ids if bands[bid] == b), default=0)
        for b in present
    }
    max_h = max(dims[bid][1] for bid in ids)
    h_bound = max(
        sum(band_h.values())
        + int(bottom_center_keepout_nm)
        + int(bottom_edge_keepout_nm)
        + int(top_edge_keepout_nm)
        + seed_gap * (len(present) + 10)
        + max_h
        + _cut_extent_padding(solve_cuts, "vertical_separation", gap),
        _required_cut_axis_extent(solve_cuts, dims, "vertical_separation"),
    )
    if axis_alignment_groups:
        # Several rigid pair islands in one band may share x exactly and must
        # therefore stack in y.  The historical one-max-height-per-band bound
        # is too small for that legal arrangement.
        h_bound = max(
            h_bound,
            sum(dims[bid][1] for bid in ids)
            + seed_gap * (len(ids) + len(present) + 2)
            + int(bottom_center_keepout_nm)
            + int(bottom_edge_keepout_nm)
            + int(top_edge_keepout_nm),
        )
    if pack_mode == "dense_2d":
        h_bound = max(
            h_bound,
            sum(dims[bid][1] for bid in ids)
            + seed_gap * (len(ids) + len(present) + 2)
            + int(bottom_center_keepout_nm)
            + int(bottom_edge_keepout_nm)
            + int(top_edge_keepout_nm),
        )
    if vertical_order_pairs:
        h_bound = max(
            h_bound,
            sum(dims[bid][1] for bid in ids) + seed_gap * (len(ids) + 1),
        )
    mandatory_width_nm = max(
        max(dims[bid][0] for bid in ids),
        _required_cut_axis_extent(solve_cuts, dims, "horizontal_separation"),
        _axis_alignment_required_width_nm(axis_alignment_groups, dims),
    )
    if strict_width_cap:
        if w_cap < mandatory_width_nm:
            raise ValueError(
                f"strict width cap {w_cap}nm is below mandatory horizontal "
                f"extent {mandatory_width_nm}nm"
            )
    else:
        w_cap = max(
            w_cap + _cut_extent_padding(solve_cuts, "horizontal_separation", gap),
            mandatory_width_nm,
        )
    if axis_alignment_groups:
        seed = _banded_axis_aligned_seed(
            core_ids,
            dims,
            gap,
            bands,
            present,
            axis_alignment_groups,
            bottom_center_keepout_nm=bottom_center_keepout_nm,
            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
            bottom_edge_keepout_exempt=tuple(sorted(exempt)),
            pair_clearances_nm=pair_clearances_nm,
            w_cap=(w_cap if pack_mode == "dense_2d" else None),
        )
        if not seed:
            raise RuntimeError(
                "hard differential-axis alignment has no constructive seed"
            )
    elif (
        pack_mode == "dense_2d"
        and not core_order_pairs
        and not core_vertical_order_pairs
    ):
        seed = _banded_dense_seed(
            core_ids,
            dims,
            gap,
            bands,
            present,
            w_cap,
            bottom_center_keepout_nm=bottom_center_keepout_nm,
            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
            pair_clearances_nm=pair_clearances_nm,
        )
    else:
        seed = _banded_shelf_pack(
            core_ids,
            dims,
            gap,
            bands,
            present,
            order_pairs=core_order_pairs,
            bottom_center_keepout_nm=bottom_center_keepout_nm,
            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
            bottom_edge_keepout_exempt=tuple(sorted(exempt)),
            pair_clearances_nm=pair_clearances_nm,
        )
    lateral_cursor = max(seed[name][0] + dims[name][0] for name in seed) + gap
    for name in _stable_hard_order(tuple(sorted(lateral)), order_pairs):
        seed[name] = (
            lateral_cursor,
            0 if name in exempt else int(bottom_edge_keepout_nm),
        )
        lateral_cursor += dims[name][0] + gap
    if aspect_obligation and lateral:
        aspect_lower, aspect_upper, aspect_scale = _scaled_aspect_bounds(
            aspect_target,
            aspect_bounds,
        )
        seed_width = max(seed[name][0] + dims[name][0] for name in ids)
        seed_height = max(seed[name][1] + dims[name][1] for name in ids)
        required_height = _ceil_ratio(seed_width * aspect_scale, aspect_upper)
        maximum_height = (seed_width * aspect_scale) // aspect_lower
        if seed_height < required_height:
            grid = int(manufacturing_grid_nm)
            if grid <= 0:
                raise ValueError("manufacturing grid must be positive")
            padding = _ceil_ratio(required_height - seed_height, grid) * grid
            if seed_height + padding <= maximum_height:
                highest_band = present[-1]
                seed = {
                    name: (
                        point[0],
                        point[1]
                        + (
                            padding
                            if name not in lateral and bands[name] == highest_band
                            else 0
                        ),
                    )
                    for name, point in seed.items()
                }
    seed = _center_aligned_lateral_seed(
        seed,
        dims=dims,
        center_alignment_pairs=center_alignment_pairs,
        lateral_blocks=tuple(sorted(lateral)),
        order_pairs=order_pairs,
        gap_nm=gap,
        pair_clearances_nm=pair_clearances_nm,
    )
    seed = _hierarchical_vertical_dag_seed(
        seed,
        dims=dims,
        bands=bands,
        hierarchical_macro_orders=hierarchical_macro_orders,
        lateral_blocks=tuple(sorted(lateral)),
        manufacturing_grid_nm=manufacturing_grid_nm,
        gap_nm=gap,
        bottom_center_keepout_nm=bottom_center_keepout_nm,
        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
        bottom_edge_keepout_exempt=tuple(sorted(exempt)),
    )
    if (flexible_envelope_constraints and floorplan_column_groups
            and not (axis_alignment_groups or center_alignment_pairs or order_pairs
                     or vertical_order_pairs or lateral or left_edge or right_edge
                     or replay_incumbent)):
        grouped_seed = _contiguous_envelope_seed(
            seed, dims=dims, bands=bands, envelopes=flexible_envelope_constraints,
            columns=floorplan_column_groups, orders=aggregate_envelope_orders,
            alignments=hierarchical_macro_alignments, gap_nm=seed_gap,
            grid_nm=manufacturing_grid_nm)
        if not strict_width_cap or max(grouped_seed[name][0] + dims[name][0]
                                       for name in ids) <= w_cap:
            seed = grouped_seed
    source_seed = _canonical_source_seed_positions(
        source_seed_positions,
        ids=ids,
    )
    if replay_incumbent:
        seed = dict(replay_seed)
    elif source_seed is not None:
        seed = _select_feedback_seed(
            seed,
            source_seed,
            dims=dims,
            cuts=solve_cuts,
            gap_nm=gap,
            pair_clearances_nm=pair_clearances_nm,
            rigid_compound_footprints=canonical_compound_footprints,
            layered_collision_footprints=canonical_layered_footprints,
            overlap_compatible_pairs=canonical_overlap_pairs,
        )
    seed_width = max(seed[bid][0] + dims[bid][0] for bid in ids)
    seed_height = max(seed[bid][1] + dims[bid][1] for bid in ids)
    aspect_width_basis = max(
        seed_width,
        _required_cut_axis_extent(solve_cuts, dims, "horizontal_separation"),
    )
    aspect_height_basis = max(
        seed_height,
        _required_cut_axis_extent(solve_cuts, dims, "vertical_separation"),
    )
    if not strict_width_cap:
        w_cap = max(w_cap, seed_width)
    h_bound = max(h_bound, seed_height + seed_gap)
    if aspect_obligation:
        aspect_lower, aspect_upper, aspect_scale = _scaled_aspect_bounds(
            aspect_target, aspect_bounds
        )
        if not strict_width_cap:
            w_cap = max(
                w_cap,
                _ceil_ratio(aspect_lower * aspect_height_basis, aspect_scale),
            )
        h_bound = max(
            h_bound,
            _ceil_ratio(aspect_scale * aspect_width_basis, aspect_upper) + seed_gap,
        )
    if flexible_envelope_constraints:
        flexible_minimum_width = max(
            max(
                flexible_physical_offsets[row.group][name][2]
                - flexible_physical_offsets[row.group][name][0]
                for name in row.members
            )
            for row in flexible_envelope_constraints
        )
        if strict_width_cap and w_cap < flexible_minimum_width:
            raise ValueError(
                "strict width cap is below a mandatory flexible envelope extent"
            )
        if not strict_width_cap:
            w_cap = max(w_cap, flexible_minimum_width)
        h_bound = max(
            h_bound,
            sum(dims[name][1] for name in ids)
            + seed_gap * (len(ids) + 1)
            + sum(
                max(
                    flexible_physical_offsets[row.group][name][3]
                    - flexible_physical_offsets[row.group][name][1]
                    - dims[name][1],
                    0,
                )
                for row in flexible_envelope_constraints
                for name in row.members
            ),
        )
    compatible_width_nm = _right_edge_compatible_width_bound(
        w_cap, dims, right_edge, manufacturing_grid_nm
    )
    if strict_width_cap and compatible_width_nm != w_cap:
        raise ValueError(
            f"strict width cap {w_cap}nm is incompatible with the hard right-edge "
            f"manufacturing-grid residue (next compatible width is "
            f"{compatible_width_nm}nm)"
        )
    w_cap = compatible_width_nm
    if (right_edge or left_edge) and not replay_incumbent:
        seed = _hard_edge_compatible_banded_seed(
            seed,
            dims=dims,
            gap_nm=gap,
            bands=bands,
            present=present,
            width_bound_nm=w_cap,
            right_edge=right_edge,
            left_edge=left_edge,
            bottom_center_keepout_nm=bottom_center_keepout_nm,
            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
            bottom_edge_keepout_exempt=tuple(sorted(exempt)),
            pair_clearances_nm=pair_clearances_nm,
            axis_alignment_groups=axis_alignment_groups,
            center_alignment_pairs=center_alignment_pairs,
            order_pairs=order_pairs,
            rigid_compound_footprints=canonical_compound_footprints,
            layered_collision_footprints=canonical_layered_footprints,
            overlap_compatible_pairs=canonical_overlap_pairs,
        )
        seed_height = max(seed[bid][1] + dims[bid][1] for bid in ids)
        h_bound = max(h_bound, seed_height + seed_gap)
        if aspect_obligation and not strict_width_cap:
            w_cap = _right_edge_compatible_width_bound(
                max(
                    w_cap,
                    _ceil_ratio(aspect_lower * seed_height, aspect_scale),
                ),
                dims,
                right_edge,
                manufacturing_grid_nm,
            )
            seed = _hard_edge_compatible_banded_seed(
                seed,
                dims=dims,
                gap_nm=gap,
                bands=bands,
                present=present,
                width_bound_nm=w_cap,
                right_edge=right_edge,
                left_edge=left_edge,
                bottom_center_keepout_nm=bottom_center_keepout_nm,
                bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                bottom_edge_keepout_exempt=tuple(sorted(exempt)),
                pair_clearances_nm=pair_clearances_nm,
                axis_alignment_groups=axis_alignment_groups,
                center_alignment_pairs=center_alignment_pairs,
                order_pairs=order_pairs,
                rigid_compound_footprints=canonical_compound_footprints,
                layered_collision_footprints=canonical_layered_footprints,
                overlap_compatible_pairs=canonical_overlap_pairs,
            )

    seed_blocks = tuple(
        (
            name,
            (
                seed[name][0],
                seed[name][1],
                seed[name][0] + dims[name][0],
                seed[name][1] + dims[name][1],
            ),
        )
        for name in ids
    )
    seed_boxes = dict(seed_blocks)
    adjacency_movable_y = {
        name
        for left, right, _maximum in adjacency_limits_nm
        for name in (left, right)
        if name in lateral and name not in exempt
    }
    hierarchy_movable_y: set[str] = set()
    for row in hierarchical_macro_orders:
        if (
            row[3] == "horizontal"
            and not _hierarchical_macro_order_satisfied(
                seed_boxes,
                row,
                gap_nm=gap,
            )
        ):
            hierarchy_movable_y.update(row[5])
            hierarchy_movable_y.update(row[6])
    for row in hierarchical_macro_alignments:
        _scope, axis, _edge, _tolerance_ppm, groups = (
            _hierarchical_macro_alignment_parts(row)
        )
        if axis == "horizontal" and not _hierarchical_macro_alignment_satisfied(
            seed_boxes,
            row,
            tolerance_scales_nm=hierarchical_alignment_tolerance_scales_nm,
        ):
            hierarchy_movable_y.update(
                name for _node, members in groups for name in members
            )
    legalizer_movable_y = tuple(
        sorted(adjacency_movable_y | hierarchy_movable_y)
    )
    if canonical_slot_projections:
        legalizer_movable_y = tuple(sorted(
            set(legalizer_movable_y)
            | {name for row in canonical_slot_projections
               if not mosaic_guest_slot_satisfied(seed, row)
               for name in (row.host_master, row.guest_master)}
        ))
    legalizer_y_lower_bounds = {
        name: max(
            (
                0
                if name in exempt
                else int(bottom_edge_keepout_nm)
            ),
            (
                int(bottom_center_keepout_nm)
                if name not in lateral and bands[name] == _BAND_CENTER
                else 0
            ),
        )
        for name in legalizer_movable_y
    }
    seed_x_constraints_legal = (
        all(mosaic_guest_slot_satisfied(seed, row) for row in canonical_slot_projections)
        and
        _seed_geometry_legal(
            seed,
            dims=dims,
            gap_nm=gap,
            pair_clearances_nm=pair_clearances_nm,
            rigid_compound_footprints=canonical_compound_footprints,
            layered_collision_footprints=canonical_layered_footprints,
            overlap_compatible_pairs=canonical_overlap_pairs,
        )
        and not any(
            seed[right][0]
            < seed[left][0]
            + dims[left][0]
            + _required_pair_gap_nm(left, right, gap, pair_clearances_nm)
            for left, right in order_pairs
        )
        and _floorplan_column_order_satisfied(
            seed_blocks,
            floorplan_column_classes,
        )
        and _floorplan_column_group_order_satisfied(
            seed_blocks,
            floorplan_column_groups,
        )
        and all(
            _hierarchical_macro_order_satisfied(
                seed_boxes,
                row,
                gap_nm=gap,
            )
            for row in hierarchical_macro_orders
        )
        and all(
            _hierarchical_macro_alignment_satisfied(
                seed_boxes,
                row,
                tolerance_scales_nm=(
                    hierarchical_alignment_tolerance_scales_nm
                ),
            )
            for row in hierarchical_macro_alignments
        )
    )
    if (
        not seed_x_constraints_legal
        and not replay_incumbent
        and remaining_time_s > 0.0
        and not flexible_envelope_constraints
    ):
        x_legalization_budget = remaining_time_s / 10.0
        try:
            x_legalization_width = (
                w_cap
                if strict_width_cap
                else max(
                    w_cap,
                    sum(dims[name][0] for name in ids)
                    + gap * max(0, len(ids) - 1),
                )
            )
            seed = _legalize_fixed_outline_seed_x(
                seed,
                dims=dims,
                bands=bands,
                width_nm=x_legalization_width,
                height_nm=h_bound,
                movable_y=legalizer_movable_y,
                movable_y_lower_bounds_nm=legalizer_y_lower_bounds,
                gap_nm=gap,
                max_time_s=x_legalization_budget,
                manufacturing_grid_nm=manufacturing_grid_nm,
                floorplan_column_classes=floorplan_column_classes,
                floorplan_column_groups=floorplan_column_groups,
                hierarchical_cohesion_groups=hierarchical_cohesion_groups,
                hierarchical_macro_orders=hierarchical_macro_orders,
                hierarchical_macro_alignments=hierarchical_macro_alignments,
                hierarchical_alignment_tolerance_scales_nm=(
                    hierarchical_alignment_tolerance_scales_nm
                ),
                rigid_compound_footprints=canonical_compound_footprints,
                layered_collision_footprints=canonical_layered_footprints,
                row_bounded_cavity_projections=canonical_cavity_projections,
                mosaic_guest_slot_projections=canonical_slot_projections,
                edge_rail_reservation=edge_rail_reservation,
                overlap_compatible_pairs=canonical_overlap_pairs,
                pair_clearances_nm=pair_clearances_nm,
                axis_alignment_groups=axis_alignment_groups,
                order_pairs=order_pairs,
                vertical_order_pairs=vertical_order_pairs,
                adjacency_limits_nm=adjacency_limits_nm,
                adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
                right_edge=right_edge,
                left_edge=left_edge,
                exact_outline=exact_outline_width,
            )
        except ExecutionStopped:
            raise
        except RuntimeError:
            pass
        remaining_time_s = max(0.0, remaining_time_s - x_legalization_budget)
        if not strict_width_cap:
            w_cap = max(
                w_cap,
                max(seed[name][0] + dims[name][0] for name in ids),
            )
        if aspect_obligation:
            lower, upper, scale = _scaled_aspect_bounds(
                aspect_target,
                aspect_bounds,
            )
            seed_width = max(seed[name][0] + dims[name][0] for name in ids) - min(
                seed[name][0] for name in ids
            )
            seed_height = max(
                seed[name][1] + dims[name][1] for name in ids
            ) - min(seed[name][1] for name in ids)
            required_height = _ceil_ratio(seed_width * scale, upper)
            maximum_height = (seed_width * scale) // lower
            if seed_height < required_height:
                grid = int(manufacturing_grid_nm)
                padding = _ceil_ratio(required_height - seed_height, grid) * grid
                if seed_height + padding <= maximum_height:
                    highest_band = present[-1]
                    seed = {
                        name: (
                            point[0],
                            point[1]
                            + (
                                padding
                                if name not in lateral
                                and bands[name] == highest_band
                                else 0
                            ),
                        )
                        for name, point in seed.items()
                    }
        seed_height = max(seed[name][1] + dims[name][1] for name in ids)
        h_bound = max(h_bound, seed_height + seed_gap)

    seed_fits_width_cap = all(
        0 <= seed[bid][0] <= w_cap - dims[bid][0] for bid in ids
    )
    certified_source_seed_legal = bool(
        certified_source_seed
        and source_seed is not None
        and seed == source_seed
        and _banded_seed_satisfies_hard_constraints(
            seed,
            ids=ids,
            dims=dims,
            gap_nm=gap,
            width_bound_nm=w_cap,
            height_bound_nm=h_bound,
            bands=bands,
            adjacency_limits_nm=adjacency_limits_nm,
            adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
            order_pairs=order_pairs,
            vertical_order_pairs=vertical_order_pairs,
            floorplan_column_classes=floorplan_column_classes,
            floorplan_column_groups=floorplan_column_groups,
            bottom_center_keepout_nm=bottom_center_keepout_nm,
            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
            bottom_edge_keepout_exempt=tuple(sorted(exempt)),
            top_edge_keepout_nm=top_edge_keepout_nm,
            top_edge_keepout_clear=tuple(sorted(top_clear)),
            solve_cuts=solve_cuts,
            pair_clearances_nm=pair_clearances_nm,
            axis_alignment_groups=axis_alignment_groups,
            center_alignment_pairs=center_alignment_pairs,
            aggregate_envelope_orders=aggregate_envelope_orders,
            hierarchical_macro_orders=hierarchical_macro_orders,
            hierarchical_macro_alignments=hierarchical_macro_alignments,
            hierarchical_alignment_tolerance_scales_nm=(
                hierarchical_alignment_tolerance_scales_nm
            ),
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
            manufacturing_grid_nm=manufacturing_grid_nm,
            right_edge=right_edge,
            left_edge=left_edge,
            placement_objective=placement_objective,
            strict_width_cap=strict_width_cap,
            exact_outline_width=exact_outline_width,
            lateral_blocks=tuple(sorted(lateral)),
            flexible_envelope_constraints=flexible_envelope_constraints,
            rigid_compound_footprints=canonical_compound_footprints,
            layered_collision_footprints=canonical_layered_footprints,
            row_bounded_cavity_projections=canonical_cavity_projections,
            mosaic_guest_slot_projections=canonical_slot_projections,
            edge_rail_reservation=edge_rail_reservation,
            overlap_compatible_pairs=canonical_overlap_pairs,
            enforce_legacy_band_order=enforce_legacy_band_order,
        )
    )
    seed_replay_width_bound_nm = (
        _compaction_replay_width_bound(
            seed, dims=dims, configured_width_bound_nm=w_cap,
            strict_width_cap=strict_width_cap, exact_outline_width=exact_outline_width,
        ) if replay_incumbent else w_cap
    )
    internal_seed_legal = bool(
        seed_fits_width_cap
        and _banded_seed_satisfies_hard_constraints(
            seed,
            ids=ids,
            dims=dims,
            gap_nm=gap,
            width_bound_nm=seed_replay_width_bound_nm,
            height_bound_nm=h_bound,
            bands=bands,
            adjacency_limits_nm=adjacency_limits_nm,
            adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
            order_pairs=order_pairs,
            vertical_order_pairs=vertical_order_pairs,
            floorplan_column_classes=floorplan_column_classes,
            floorplan_column_groups=floorplan_column_groups,
            bottom_center_keepout_nm=bottom_center_keepout_nm,
            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
            bottom_edge_keepout_exempt=tuple(sorted(exempt)),
            top_edge_keepout_nm=top_edge_keepout_nm,
            top_edge_keepout_clear=tuple(sorted(top_clear)),
            solve_cuts=solve_cuts,
            pair_clearances_nm=pair_clearances_nm,
            axis_alignment_groups=axis_alignment_groups,
            center_alignment_pairs=center_alignment_pairs,
            aggregate_envelope_orders=aggregate_envelope_orders,
            hierarchical_macro_orders=hierarchical_macro_orders,
            hierarchical_macro_alignments=hierarchical_macro_alignments,
            hierarchical_alignment_tolerance_scales_nm=(
                hierarchical_alignment_tolerance_scales_nm
            ),
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
            manufacturing_grid_nm=manufacturing_grid_nm,
            right_edge=right_edge,
            left_edge=left_edge,
            placement_objective=placement_objective,
            strict_width_cap=strict_width_cap,
            exact_outline_width=exact_outline_width,
            lateral_blocks=tuple(sorted(lateral)),
            flexible_envelope_constraints=flexible_envelope_constraints,
            rigid_compound_footprints=canonical_compound_footprints,
            layered_collision_footprints=canonical_layered_footprints,
            row_bounded_cavity_projections=canonical_cavity_projections,
            mosaic_guest_slot_projections=canonical_slot_projections,
            edge_rail_reservation=edge_rail_reservation,
            overlap_compatible_pairs=canonical_overlap_pairs,
            enforce_legacy_band_order=enforce_legacy_band_order,
        )
    )
    if replay_incumbent and (not internal_seed_legal or seed != replay_seed):
        raise ValueError("incumbent replay source fails complete native hard replay")
    _banded_seed_mode = os.environ.get("CLAY_BANDED_SEED_CPP", "rescue").strip().lower()
    if (
        source_seed_positions is None
        and not replay_incumbent
        and not axis_alignment_groups
        and not center_alignment_pairs
        and not flexible_envelope_constraints
        and not canonical_slot_projections
        and not canonical_cavity_projections
        and not solve_cuts
        and len(ids) >= 3
        and _banded_seed_mode != "0"
        and ((not internal_seed_legal) or _banded_seed_mode == "force")
    ):
        try:
            from clay._native import _engine_cpp as _banded_cpp

            _solve_banded_cpp = getattr(_banded_cpp, "solve_banded_area_cpp", None)
        except Exception:
            _solve_banded_cpp = None
        if _solve_banded_cpp is not None:
            _id_to_idx = {bid: i for i, bid in enumerate(ids)}
            _re_set = set(right_edge)
            _le_set = set(left_edge)
            _rp_set = set(right_pull)
            _lp_set = set(left_pull)
            _cpp_blocks = [
                (
                    bid,
                    int(dims[bid][0]),
                    int(dims[bid][1]),
                    int(bands[bid]),
                    bid in lateral,
                    bid in _le_set,
                    bid in _re_set,
                    bid in _lp_set,
                    bid in _rp_set,
                    max(
                        0 if bid in exempt else int(bottom_edge_keepout_nm),
                        int(bottom_center_keepout_nm)
                        if bid not in lateral and bands[bid] == _BAND_CENTER
                        else 0,
                    ),
                    bid in top_clear,
                )
                for bid in ids
            ]
            _cpp_ord = [
                (_id_to_idx[a], _id_to_idx[b])
                for a, b in order_pairs
                if a in _id_to_idx and b in _id_to_idx
            ]
            _cpp_vord = [
                (_id_to_idx[a], _id_to_idx[b])
                for a, b in vertical_order_pairs
                if a in _id_to_idx and b in _id_to_idx
            ]
            _cpp_pclr = [
                (_id_to_idx[a], _id_to_idx[b], int(gval))
                for (a, b), gval in pair_clearances_nm.items()
                if a in _id_to_idx and b in _id_to_idx
            ]
            _cpp_pull = [
                (_id_to_idx[a], _id_to_idx[b])
                for a, b in pull_pairs
                if a in _id_to_idx and b in _id_to_idx
            ]
            _asp_lo, _asp_hi, _asp_sc = (
                _scaled_aspect_bounds(aspect_target, aspect_bounds)
                if aspect_obligation
                else (0, 0, 1_000_000)
            )
            _cpp_sol = _solve_banded_cpp(
                _cpp_blocks,
                gap_nm=int(gap),
                grid_nm=int(manufacturing_grid_nm),
                w_cap_nm=int(w_cap),
                h_bound_nm=int(h_bound),
                bottom_edge_keepout_nm=int(bottom_edge_keepout_nm),
                top_edge_keepout_nm=int(top_edge_keepout_nm),
                strict_width_cap=bool(strict_width_cap),
                exact_outline_width=bool(exact_outline_width),
                enforce_band_order=bool(enforce_legacy_band_order),
                objective_area=(placement_objective == "area"),
                dense_2d=(pack_mode == "dense_2d"),
                aspect_obligation=bool(aspect_obligation),
                aspect_lower=int(_asp_lo),
                aspect_upper=int(_asp_hi),
                aspect_scale=int(_asp_sc),
                order_pairs=_cpp_ord,
                vertical_order_pairs=_cpp_vord,
                pair_clearances=_cpp_pclr,
                pull_pairs=_cpp_pull,
                max_evals=1500,
            )
            if _cpp_sol.get("status") in ("OPTIMAL", "FEASIBLE"):
                _cpp_xs = _cpp_sol["xs"]
                _cpp_ys = _cpp_sol["ys"]
                _cpp_seed = {
                    bid: (int(_cpp_xs[i]), int(_cpp_ys[i]))
                    for i, bid in enumerate(ids)
                }
                _cpp_fits = all(
                    0 <= _cpp_seed[bid][0] <= w_cap - dims[bid][0]
                    and 0 <= _cpp_seed[bid][1] <= h_bound - dims[bid][1]
                    for bid in ids
                )
                if _cpp_fits and _banded_seed_satisfies_hard_constraints(
                    _cpp_seed,
                    ids=ids,
                    dims=dims,
                    gap_nm=gap,
                    width_bound_nm=w_cap,
                    height_bound_nm=h_bound,
                    bands=bands,
                    adjacency_limits_nm=adjacency_limits_nm,
                    adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
                    order_pairs=order_pairs,
                    vertical_order_pairs=vertical_order_pairs,
                    floorplan_column_classes=floorplan_column_classes,
                    floorplan_column_groups=floorplan_column_groups,
                    bottom_center_keepout_nm=bottom_center_keepout_nm,
                    bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                    bottom_edge_keepout_exempt=tuple(sorted(exempt)),
                    top_edge_keepout_nm=top_edge_keepout_nm,
                    top_edge_keepout_clear=tuple(sorted(top_clear)),
                    solve_cuts=solve_cuts,
                    pair_clearances_nm=pair_clearances_nm,
                    axis_alignment_groups=axis_alignment_groups,
                    center_alignment_pairs=center_alignment_pairs,
                    aggregate_envelope_orders=aggregate_envelope_orders,
                    hierarchical_macro_orders=hierarchical_macro_orders,
                    hierarchical_macro_alignments=hierarchical_macro_alignments,
                    hierarchical_alignment_tolerance_scales_nm=(
                        hierarchical_alignment_tolerance_scales_nm
                    ),
                    aspect_target=aspect_target,
                    aspect_bounds=aspect_bounds,
                    aspect_obligation=aspect_obligation,
                    manufacturing_grid_nm=manufacturing_grid_nm,
                    right_edge=right_edge,
                    left_edge=left_edge,
                    placement_objective=placement_objective,
                    strict_width_cap=strict_width_cap,
                    exact_outline_width=exact_outline_width,
                    lateral_blocks=tuple(sorted(lateral)),
                    flexible_envelope_constraints=flexible_envelope_constraints,
                    rigid_compound_footprints=canonical_compound_footprints,
                    layered_collision_footprints=canonical_layered_footprints,
                    row_bounded_cavity_projections=canonical_cavity_projections,
                    mosaic_guest_slot_projections=canonical_slot_projections,
                    edge_rail_reservation=edge_rail_reservation,
                    overlap_compatible_pairs=canonical_overlap_pairs,
                    enforce_legacy_band_order=enforce_legacy_band_order,
                ):
                    _cpp_primary = _placement_primary_extent_nm(
                        _cpp_seed,
                        dims=dims,
                        placement_objective=placement_objective,
                        flexible_envelope_constraints=flexible_envelope_constraints,
                        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                    )
                    _cur_primary = (
                        _placement_primary_extent_nm(
                            seed,
                            dims=dims,
                            placement_objective=placement_objective,
                            flexible_envelope_constraints=flexible_envelope_constraints,
                            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                        )
                        if internal_seed_legal
                        else None
                    )
                    if _cur_primary is None or _cpp_primary < _cur_primary:
                        seed = _cpp_seed
                        seed_fits_width_cap = True
                        internal_seed_legal = True
    if internal_seed_legal and placement_objective == "area" and lateral and not replay_incumbent:
        def replay_legal(candidate: Mapping[str, Tuple[int, int]]) -> bool:
            return _banded_seed_satisfies_hard_constraints(
                candidate,
                ids=ids,
                dims=dims,
                gap_nm=gap,
                width_bound_nm=w_cap,
                height_bound_nm=h_bound,
                bands=bands,
                adjacency_limits_nm=adjacency_limits_nm,
                adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
                order_pairs=order_pairs,
                vertical_order_pairs=vertical_order_pairs,
                floorplan_column_classes=floorplan_column_classes,
                floorplan_column_groups=floorplan_column_groups,
                bottom_center_keepout_nm=bottom_center_keepout_nm,
                bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                bottom_edge_keepout_exempt=tuple(sorted(exempt)),
                top_edge_keepout_nm=top_edge_keepout_nm,
                top_edge_keepout_clear=tuple(sorted(top_clear)),
                solve_cuts=solve_cuts,
                pair_clearances_nm=pair_clearances_nm,
                axis_alignment_groups=axis_alignment_groups,
                center_alignment_pairs=center_alignment_pairs,
                aggregate_envelope_orders=aggregate_envelope_orders,
                hierarchical_macro_orders=hierarchical_macro_orders,
                hierarchical_macro_alignments=hierarchical_macro_alignments,
                hierarchical_alignment_tolerance_scales_nm=(
                    hierarchical_alignment_tolerance_scales_nm
                ),
                aspect_target=aspect_target,
                aspect_bounds=aspect_bounds,
                aspect_obligation=aspect_obligation,
                manufacturing_grid_nm=manufacturing_grid_nm,
                right_edge=right_edge,
                left_edge=left_edge,
                placement_objective=placement_objective,
                strict_width_cap=strict_width_cap,
                exact_outline_width=exact_outline_width,
                lateral_blocks=tuple(sorted(lateral)),
                flexible_envelope_constraints=flexible_envelope_constraints,
                rigid_compound_footprints=canonical_compound_footprints,
                layered_collision_footprints=canonical_layered_footprints,
                row_bounded_cavity_projections=canonical_cavity_projections,
                mosaic_guest_slot_projections=canonical_slot_projections,
                edge_rail_reservation=edge_rail_reservation,
                overlap_compatible_pairs=canonical_overlap_pairs,
                enforce_legacy_band_order=enforce_legacy_band_order,
            )

        seed, compacted_lateral_sink = _compact_lateral_sink_positions_x(
            seed,
            lateral_blocks=tuple(sorted(lateral)),
            order_pairs=order_pairs,
            manufacturing_grid_nm=manufacturing_grid_nm,
            legal=replay_legal,
        )
        if compacted_lateral_sink:
            certified_source_seed_legal = False

    hierarchical_cohesion_span_caps_nm = (
        _hierarchical_cohesion_span_caps_nm(
            seed,
            dims=dims,
            groups=hierarchical_cohesion_groups,
        )
        if internal_seed_legal and hierarchical_cohesion_groups
        else ()
    )
    hierarchy_span_cap_by_group = {
        group: (x_cap, y_cap)
        for group, x_cap, y_cap in hierarchical_cohesion_span_caps_nm
    }

    model = cp_model.CpModel()
    xs: Dict[str, Any] = {}
    ys: Dict[str, Any] = {}
    x_iv: Dict[str, Any] = {}
    y_iv: Dict[str, Any] = {}
    H = model.NewIntVar(0, h_bound, "H")
    W = None
    area_bbox_width = None
    bbox_height = None
    if placement_objective == "area":
        W = model.NewIntVar(max(dims[bid][0] for bid in ids), w_cap, "W")
    for bid in ids:  # ids sorted by caller -> input-order-invariant model
        w, h = dims[bid]
        x0 = model.NewIntVar(0, max(0, w_cap - w), f"x_{bid}")
        y0 = model.NewIntVar(0, h_bound - h, f"y_{bid}")
        xs[bid], ys[bid] = x0, y0
        x_iv[bid] = model.NewFixedSizeIntervalVar(x0, w + gap, f"xi_{bid}")
        y_iv[bid] = model.NewFixedSizeIntervalVar(y0, h + gap, f"yi_{bid}")
        if seed_fits_width_cap:
            model.AddHint(x0, seed[bid][0])
            model.AddHint(y0, seed[bid][1])
    flexible_vars: Dict[str, Tuple[Any, Any, Any, Any]] = {}
    for group_index, row in enumerate(flexible_envelope_constraints):
        outer_x_min = model.NewIntVar(0, w_cap, f"flex_outer_x_min_{group_index}")
        outer_y_min = model.NewIntVar(0, h_bound, f"flex_outer_y_min_{group_index}")
        outer_x_max = model.NewIntVar(0, w_cap, f"flex_outer_x_max_{group_index}")
        outer_y_max = model.NewIntVar(0, h_bound, f"flex_outer_y_max_{group_index}")
        offsets = flexible_physical_offsets[row.group]
        model.AddMinEquality(
            outer_x_min,
            [xs[name] + offsets[name][0] for name in row.members],
        )
        model.AddMinEquality(
            outer_y_min,
            [ys[name] + offsets[name][1] for name in row.members],
        )
        model.AddMaxEquality(
            outer_x_max,
            [xs[name] + offsets[name][2] for name in row.members],
        )
        model.AddMaxEquality(
            outer_y_max,
            [ys[name] + offsets[name][3] for name in row.members],
        )
        flexible_vars[row.group] = (
            outer_x_min,
            outer_y_min,
            outer_x_max,
            outer_y_max,
        )
    for group_index, row in enumerate(flexible_envelope_constraints):
        offsets = flexible_physical_offsets[row.group]
        if row.foreign_exclusion_mode == "envelope":
            outer_x_min, outer_y_min, outer_x_max, outer_y_max = (
                flexible_vars[row.group]
            )
            for foreign_index, name in enumerate(ids):
                if name in row.members:
                    continue
                clearance = max(
                    flexible_foreign_clearances[row.group][(member, name)]
                    for member in row.members
                )
                alternatives = tuple(
                    model.NewBoolVar(
                        f"flex_envelope_foreign_{group_index}_"
                        f"{foreign_index}_{axis}"
                    )
                    for axis in range(4)
                )
                model.Add(outer_x_max + clearance <= xs[name]).OnlyEnforceIf(
                    alternatives[0]
                )
                model.Add(
                    xs[name] + dims[name][0] + clearance <= outer_x_min
                ).OnlyEnforceIf(alternatives[1])
                model.Add(outer_y_max + clearance <= ys[name]).OnlyEnforceIf(
                    alternatives[2]
                )
                model.Add(
                    ys[name] + dims[name][1] + clearance <= outer_y_min
                ).OnlyEnforceIf(alternatives[3])
                model.AddBoolOr(alternatives)
        else:
            for member_index, member in enumerate(row.members):
                member_x0 = xs[member] + offsets[member][0]
                member_y0 = ys[member] + offsets[member][1]
                member_x1 = xs[member] + offsets[member][2]
                member_y1 = ys[member] + offsets[member][3]
                for foreign_index, name in enumerate(ids):
                    if name in row.members:
                        continue
                    clearance = flexible_foreign_clearances[row.group][
                        (member, name)
                    ]
                    alternatives = tuple(
                        model.NewBoolVar(
                            f"flex_foreign_{group_index}_{member_index}_"
                            f"{foreign_index}_{axis}"
                        )
                        for axis in range(4)
                    )
                    model.Add(member_x1 + clearance <= xs[name]).OnlyEnforceIf(
                        alternatives[0]
                    )
                    model.Add(
                        xs[name] + dims[name][0] + clearance <= member_x0
                    ).OnlyEnforceIf(alternatives[1])
                    model.Add(member_y1 + clearance <= ys[name]).OnlyEnforceIf(
                        alternatives[2]
                    )
                    model.Add(
                        ys[name] + dims[name][1] + clearance <= member_y0
                    ).OnlyEnforceIf(alternatives[3])
                    model.AddBoolOr(alternatives)
    outline_x_min_terms = [*xs.values(), *(row[0] for row in flexible_vars.values())]
    outline_x_max_terms = [
        *(xs[bid] + dims[bid][0] for bid in ids),
        *(row[2] for row in flexible_vars.values()),
    ]
    outline_y_min_terms = [*ys.values(), *(row[1] for row in flexible_vars.values())]
    outline_y_max_terms = [
        *(ys[bid] + dims[bid][1] for bid in ids),
        *(row[3] for row in flexible_vars.values()),
    ]
    if exact_outline_width:
        right_rail_margin_nm = int(
            getattr(edge_rail_reservation, "right_margin_nm", 0) or 0
        )
        strict_x_min = model.NewIntVar(0, w_cap, "strict_outline_x_min")
        strict_x_max = model.NewIntVar(
            0, max(0, w_cap - right_rail_margin_nm), "strict_outline_x_max"
        )
        model.AddMinEquality(strict_x_min, outline_x_min_terms)
        model.AddMaxEquality(strict_x_max, outline_x_max_terms)
        model.Add(strict_x_min == 0)
        model.Add(strict_x_max + right_rail_margin_nm == w_cap)
    model.AddMaxEquality(H, outline_y_max_terms)
    if edge_rail_reservation is not None:
        edge_rail_reservation.add_constraints(model, xs=xs, ys=ys, dims=dims)
    physical_top_by_master = {
        bid: ys[bid] + dims[bid][1]
        for bid in ids
    }
    for row in flexible_envelope_constraints:
        offsets = flexible_physical_offsets[row.group]
        for member in row.members:
            physical_top_by_master[member] = ys[member] + offsets[member][3]
    _add_top_edge_keepout_constraints(
        model,
        physical_top_by_master,
        H,
        top_edge_keepout_nm,
        tuple(sorted(top_clear)),
    )
    bbox_y_min = model.NewIntVar(0, h_bound, "objective_bbox_y_min")
    occupied_bbox_height = model.NewIntVar(
        1, h_bound, "objective_bbox_height"
    )
    model.AddMinEquality(bbox_y_min, outline_y_min_terms)
    model.Add(occupied_bbox_height == H - bbox_y_min)
    if int(bottom_edge_keepout_nm) > 0:
        bbox_height = model.NewIntVar(1, h_bound, "floorplan_bbox_height")
        model.Add(bbox_height == H)
    else:
        bbox_height = occupied_bbox_height
    if W is not None:
        model.AddMaxEquality(W, outline_x_max_terms)
        area_x_min = model.NewIntVar(0, w_cap, "area_bbox_x_min")
        area_bbox_width = model.NewIntVar(1, w_cap, "area_bbox_width")
        model.AddMinEquality(area_x_min, outline_x_min_terms)
        model.Add(area_x_min == 0)
        model.Add(area_bbox_width == W - area_x_min)
    objective_y_slacks = []
    for bid in ids:
        lower = 0
        if bid not in exempt:
            lower = max(lower, int(bottom_edge_keepout_nm))
        if bid not in lateral and bands[bid] == _BAND_CENTER:
            lower = max(lower, int(bottom_center_keepout_nm))
        slack = model.NewIntVar(
            0,
            h_bound - lower,
            f"objective_y_slack_{bid}",
        )
        model.Add(slack == ys[bid] - lower)
        objective_y_slacks.append(slack)
    for group_index, row in enumerate(flexible_envelope_constraints):
        _gx0, gy0, _gx1, _gy1 = flexible_vars[row.group]
        band = bands[row.members[0]]
        lower = max(
            (
                0
                if all(member in exempt for member in row.members)
                else int(bottom_edge_keepout_nm)
            ),
            (int(bottom_center_keepout_nm) if band == _BAND_CENTER else 0),
        )
        model.Add(gy0 >= lower)
        slack = model.NewIntVar(
            0,
            h_bound - lower,
            f"objective_flexible_y_slack_{group_index}",
        )
        model.Add(slack == gy0 - lower)
        objective_y_slacks.append(slack)
    objective_y_slack_min = model.NewIntVar(
        0,
        h_bound,
        "objective_y_slack_min",
    )
    model.AddMinEquality(objective_y_slack_min, objective_y_slacks)
    model.Add(objective_y_slack_min == 0)
    outline_width = W if W is not None else w_cap
    # Without hard band seams, all bands share a full-2D collision graph.
    # A lateral owner or hierarchy also retains its complete collision graph.
    collision_groups = (
        (tuple(ids),)
        if lateral or hierarchical_topology_active or not enforce_legacy_band_order
        else tuple(
            tuple(bid for bid in ids if bands[bid] == band)
            for band in present
        )
    )
    collision_index = 0
    collision_raw_atom_pairs = 0
    collision_unique_forbidden_boxes = 0
    for members in collision_groups:
        group = tuple(sorted(set(members)))
        if (
            len(group) >= 2
            and not (set(group) & (compound_owners | layered_owners))
            and not any(
                pair in overlap_compatible_set
                for pair in combinations(group, 2)
            )
        ):
            model.AddNoOverlap2D(
                [x_iv[owner] for owner in group],
                [y_iv[owner] for owner in group],
            )
            continue
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if (left, right) in overlap_compatible_set:
                    continue
                required = int(pair_clearances_nm.get((left, right), gap))
                forbidden_boxes, raw_count, unique_count = (
                    _pair_collision_forbidden_box_cover(
                        collision_atoms[left],
                        collision_atoms[right],
                        required_nm=required,
                    )
                )
                collision_raw_atom_pairs += raw_count
                collision_unique_forbidden_boxes += unique_count
                for fx0, fy0, fx1, fy1 in forbidden_boxes:
                    collision_index += 1
                    left_of = model.NewBoolVar(
                        f"atom_left_{collision_index}_{left}_{right}"
                    )
                    right_of = model.NewBoolVar(
                        f"atom_right_{collision_index}_{left}_{right}"
                    )
                    below = model.NewBoolVar(
                        f"atom_below_{collision_index}_{left}_{right}"
                    )
                    above = model.NewBoolVar(
                        f"atom_above_{collision_index}_{left}_{right}"
                    )
                    model.Add(xs[right] - xs[left] >= fx1).OnlyEnforceIf(left_of)
                    model.Add(xs[right] - xs[left] <= fx0 - 1).OnlyEnforceIf(
                        right_of
                    )
                    model.Add(ys[right] - ys[left] >= fy1).OnlyEnforceIf(below)
                    model.Add(ys[right] - ys[left] <= fy0 - 1).OnlyEnforceIf(above)
                    model.AddBoolOr((left_of, right_of, below, above))
    if not (
        0 <= collision_index
        <= collision_unique_forbidden_boxes
        <= collision_raw_atom_pairs
    ):
        raise AssertionError("layered collision quotient cardinality is inconsistent")
    _add_bottom_edge_keepout_constraints(
        model, ys, bottom_edge_keepout_nm, exempt
    )
    _add_manufacturing_grid_constraints(
        model, xs, ys, manufacturing_grid_nm
    )
    _add_hard_horizontal_edge_constraints(
        model,
        xs,
        dims,
        width_bound=outline_width,
        right_edge=right_edge,
        left_edge=left_edge,
    )
    if aspect_obligation:
        _add_directional_aspect_constraints(
            model,
            xs=xs,
            ys=ys,
            dims=dims,
            width_bound=w_cap,
            height_bound=h_bound,
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
        )
        if exact_outline_width:
            # The generic min/max equalities are exact but CP-SAT's relaxation
            # does not derive this simple quotient early.  Materialize the
            # mathematically implied bound so a fixed-width proof can close as
            # soon as its incumbent reaches the declared aspect frontier.
            model.Add(
                bbox_height
                >= _strict_outline_aspect_height_floor_nm(
                    w_cap,
                    aspect_target=aspect_target,
                    aspect_bounds=aspect_bounds,
                )
            )
    _add_pair_clearance_constraints(
        model,
        xs,
        ys,
        dims,
        pair_clearances_nm,
        excluded_blocks=tuple(sorted(compound_owners | layered_owners)),
    )
    _add_solve_cut_constraints(model, xs, ys, dims, solve_cuts)
    _add_axis_alignment_constraints(model, xs, axis_alignment_groups)
    _add_center_alignment_constraints(
        model, xs, ys, dims, center_alignment_pairs
    )
    _add_aggregate_envelope_order_constraints(
        model,
        xs,
        ys,
        dims,
        aggregate_envelope_orders,
        width_bound_nm=w_cap,
        height_bound_nm=h_bound,
        center_step_nm=manufacturing_grid_nm,
    )
    _add_hierarchical_macro_constraints(
        model,
        xs,
        ys,
        dims,
        hierarchical_macro_orders,
        hierarchical_macro_alignments,
        width_bound_nm=w_cap,
        height_bound_nm=h_bound,
        center_step_nm=manufacturing_grid_nm,
        gap_nm=gap,
        tolerance_scales_nm=hierarchical_alignment_tolerance_scales_nm,
    )
    _add_row_bounded_cavity_constraints(
        model,
        xs,
        ys,
        dims,
        canonical_cavity_projections,
        minimum_clearance_nm=gap,
    )
    add_mosaic_guest_slot_constraints(model, xs, ys, canonical_slot_projections)
    _add_floorplan_column_constraints(
        model, xs, dims, floorplan_column_classes
    )
    _add_floorplan_column_group_constraints(model, xs, floorplan_column_groups)
    _add_adjacency_constraints(
        model,
        xs,
        ys,
        dims,
        adjacency_limits_nm,
        scope_offsets_nm=adjacency_scope_offsets_nm,
        minimum_gap_nm=gap,
        projection_nm=manufacturing_grid_nm,
    )
    add_native_band_constraints(model, ys, dims, bands=bands, core_ids=core_ids,
        present=present, flexible_envelope_constraints=flexible_envelope_constraints,
        flexible_vars=flexible_vars, gap=gap, h_bound=h_bound,
        bottom_center_keepout_nm=bottom_center_keepout_nm,
        enforce_legacy_band_order=enforce_legacy_band_order)
    # ---- HARD same-band left/right order (from @order): a left_of b (caller
    #      passes only same-band pairs).
    for a, b in order_pairs or ():
        a, b = str(a), str(b)
        if a in xs and b in xs and a != b:
            model.Add(xs[b] >= xs[a] + dims[a][0] + gap)
    # ---- HARD vertical order (from @order): lower below upper.  Unlike the
    #      coarse polarity/passive bands, this is a local pair relation and does
    #      not force unrelated blocks into a new full-width tier.
    for lower, upper in vertical_order_pairs:
        model.Add(
            ys[upper]
            >= ys[lower]
            + dims[lower][1]
            + _required_pair_gap_nm(
                lower,
                upper,
                gap,
                pair_clearances_nm,
            )
        )

    # ---- SOFT position term (kept strictly below the height objective) --------
    n_pull = len(pull_pairs)
    w_edge = (n_pull + 1) if right_pull or left_pull else 0
    soft_terms: List[Any] = []
    hierarchical_terms: List[Any] = []
    for bid in right_pull:  # minimize (w_cap - x1) -> push right
        w = dims[bid][0]
        soft_terms.append(w_edge * (outline_width - w - xs[bid]))
    for bid in left_pull:  # minimize x0 -> push left
        soft_terms.append(w_edge * xs[bid])
    for a, b in pull_pairs:  # minimize |xc(a) - xc(b)|
        if a not in xs or b not in xs or a == b:
            continue
        wa, wb = dims[a][0], dims[b][0]
        d = model.NewIntVar(0, w_cap, f"pull_{a}_{b}")
        xca, xcb = xs[a] + wa // 2, xs[b] + wb // 2
        model.Add(d >= xca - xcb)
        model.Add(d >= xcb - xca)
        soft_terms.append(d)
    # One horizontal envelope per functional cell is linear in the grammar
    # size.  Height is already the local primary objective, so adding another
    # y-span term double-counts that axis and perturbs the seed without adding
    # independent information.  These are soft terms only: no spacing or
    # adjacency is inferred, and the complete master still proves every rule.
    for group_index, group in enumerate(cohesion_groups):
        x_min = model.NewIntVar(0, w_cap, f"cohesion_x_min_{group_index}")
        x_max = model.NewIntVar(0, w_cap, f"cohesion_x_max_{group_index}")
        x_span = model.NewIntVar(1, w_cap, f"cohesion_x_span_{group_index}")
        model.AddMinEquality(x_min, [xs[name] for name in group])
        model.AddMaxEquality(
            x_max, [xs[name] + dims[name][0] for name in group]
        )
        model.Add(x_span == x_max - x_min)
        soft_terms.append(x_span)
    for group_index, group in enumerate(well_cohesion_groups):
        x_min = model.NewIntVar(0, w_cap, f"well_x_min_{group_index}")
        x_max = model.NewIntVar(0, w_cap, f"well_x_max_{group_index}")
        x_span = model.NewIntVar(1, w_cap, f"well_x_span_{group_index}")
        model.AddMinEquality(x_min, [xs[name] for name in group])
        model.AddMaxEquality(x_max, [xs[name] + dims[name][0] for name in group])
        model.Add(x_span == x_max - x_min)
        soft_terms.append(x_span)
    for group_index, group in enumerate(hierarchical_cohesion_groups):
        x_min = model.NewIntVar(0, w_cap, f"hierarchy_x_min_{group_index}")
        x_max = model.NewIntVar(0, w_cap, f"hierarchy_x_max_{group_index}")
        x_span = model.NewIntVar(1, w_cap, f"hierarchy_x_span_{group_index}")
        y_min = model.NewIntVar(0, h_bound, f"hierarchy_y_min_{group_index}")
        y_max = model.NewIntVar(0, h_bound, f"hierarchy_y_max_{group_index}")
        y_span = model.NewIntVar(1, h_bound, f"hierarchy_y_span_{group_index}")
        model.AddMinEquality(x_min, [xs[name] for name in group])
        model.AddMaxEquality(x_max, [xs[name] + dims[name][0] for name in group])
        model.AddMinEquality(y_min, [ys[name] for name in group])
        model.AddMaxEquality(y_max, [ys[name] + dims[name][1] for name in group])
        model.Add(x_span == x_max - x_min)
        model.Add(y_span == y_max - y_min)
        caps = hierarchy_span_cap_by_group.get(tuple(group))
        if caps is not None:
            model.Add(x_span <= caps[0])
            model.Add(y_span <= caps[1])
        hierarchical_terms.extend((x_span, y_span))
    for group_index, row in enumerate(flexible_envelope_constraints):
        gx0, gy0, gx1, gy1 = flexible_vars[row.group]
        x_span = model.NewIntVar(1, w_cap, f"flex_x_span_{group_index}")
        y_span = model.NewIntVar(1, h_bound, f"flex_y_span_{group_index}")
        model.Add(x_span == gx1 - gx0)
        model.Add(y_span == gy1 - gy0)
        soft_terms.extend((x_span, y_span))
    soft_max = (
        w_edge * w_cap * (len(right_pull) + len(left_pull))
        + w_cap * n_pull
        + len(cohesion_groups) * w_cap
        + len(well_cohesion_groups) * w_cap
        + len(flexible_envelope_constraints) * (w_cap + h_bound)
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    _configure_2d_packing_proof(solver)
    primary = bbox_height
    objective_name = "height_weighted_plus_soft_cost" if soft_terms else "height_nm"
    phase_budget = None
    primary_evidence_offset = 0
    structural_seed_budget = 0.0
    if placement_objective == "area":
        primary = model.NewIntVar(0, w_cap * h_bound, "area_nm2")
        area_height = bbox_height
        if exact_outline_width and int(bottom_edge_keepout_nm) > 0:
            # With a fixed outline width and the hard minimum-Y-slack equality,
            # an empty bottom reserve is a mathematically constant area offset.
            # Keep the previously proven normalized-area master for CP search,
            # then add the exact offset to both objective and bound evidence.
            # This preserves the proof problem while certifying the full cell.
            area_height = occupied_bbox_height
            primary_evidence_offset = (
                0 if exempt else w_cap * int(bottom_edge_keepout_nm)
            )
        model.AddMultiplicationEquality(
            primary,
            (area_bbox_width, area_height),
        )
        if exact_outline_width:
            # Exact-outline normalization proves ``area_bbox_width == w_cap``.
            # State the resulting linear objective equality explicitly; the
            # generic multiplication propagator otherwise leaves a needlessly
            # weak bound even when the physical width is already fixed.
            model.Add(primary == w_cap * area_height)
        if internal_seed_legal:
            model.Add(
                primary
                <= _placement_primary_extent_nm(
                    seed,
                    dims=dims,
                    placement_objective="area",
                    flexible_envelope_constraints=flexible_envelope_constraints,
                    bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                )
                - primary_evidence_offset
            )
        if hierarchical_terms:
            hierarchy_secondary_bound = len(hierarchical_cohesion_groups) * (
                w_cap + h_bound
            )
            model.Minimize(
                primary * (hierarchy_secondary_bound + 1)
                + sum(hierarchical_terms)
            )
            objective_name = "area_then_hierarchy_node_span"
        else:
            model.Minimize(primary)
            objective_name = "area_nm2"
        phase_labels = (
            ("area", "soft_position")
            if soft_terms and not hierarchical_terms
            else ("area",)
        )
        # Fidelity admission is lexicographic: first prove the complete hard
        # placement on the deterministic constructive seed, then spend the
        # remaining fixed budget on area.  The reserve is part of, never added
        # to, the caller's deterministic budget.
        structural_seed_budget = (
            remaining_time_s / 20.0 if internal_seed_legal else 0.0
        )
        optimization_budget = max(
            0.0,
            remaining_time_s - structural_seed_budget,
        )
        phase_budget = _DeterministicBudgetLedger.split(
            optimization_budget,
            phase_labels,
        )
        solver.parameters.max_deterministic_time = phase_budget.for_phase("area")
    else:
        if internal_seed_legal:
            model.Add(
                bbox_height
                <= _placement_primary_extent_nm(
                    seed,
                    dims=dims,
                    placement_objective="height",
                    flexible_envelope_constraints=flexible_envelope_constraints,
                    bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                )
            )
        solver.parameters.max_deterministic_time = remaining_time_s
        if hierarchical_terms:
            hierarchy_secondary_bound = len(hierarchical_cohesion_groups) * (
                w_cap + h_bound
            )
            model.Minimize(
                bbox_height * (hierarchy_secondary_bound + 1)
                + sum(hierarchical_terms)
            )
            objective_name = "height_then_hierarchy_node_span"
        elif soft_terms:
            model.Minimize(bbox_height * (soft_max + 1) + sum(soft_terms))
        else:
            model.Minimize(bbox_height)
    validation_error = model.Validate()
    if validation_error:
        raise RuntimeError(f"invalid banded CP model: {validation_error}")
    structural_seed_status = None
    if structural_seed_budget > 0.0 and not replay_incumbent:
        structural_model = model.Clone()
        structural_model.Proto().ClearField("objective")
        structural_solver = cp_model.CpSolver()
        structural_solver.parameters.num_search_workers = 1
        structural_solver.parameters.random_seed = 0
        structural_solver.parameters.max_deterministic_time = structural_seed_budget
        structural_solver.parameters.fix_variables_to_their_hinted_value = True
        _configure_2d_packing_proof(structural_solver)
        with cp_solve_phase(structural_solver, structural_model, phase="placement.banded.solve.structural_seed"):
            structural_seed_status = structural_solver.Solve(structural_model)
    replay_model_before = replay_parameters_before = None
    if replay_incumbent:
        replay_original_model_bytes = model.Proto().SerializeToString(deterministic=True)
        model = prepare_incumbent_replay_model(
            model, coordinate_variables={name: (xs[name], ys[name]) for name in ids},
            positions=replay_seed,
        )
        solver.parameters.max_deterministic_time = remaining_time_s
        solver.parameters.fix_variables_to_their_hinted_value = True
        replay_model_before = model.Proto().SerializeToString(deterministic=True)
        replay_parameters_before = solver.parameters.SerializeToString(deterministic=True)
    tie_break_active = (
        canonical_tie_break_enabled()
        and placement_objective == "area"
        and not replay_incumbent
    )
    restore_area_execution = (
        apply_area_search_execution_policy(solver)
        if placement_objective == "area" and not replay_incumbent else None
    )
    try:
        with cp_solve_phase(solver, model, phase="placement.banded.solve.primary"):
            st = solver.Solve(model)
    finally:
        if restore_area_execution is not None:
            restore_area_execution()
    if replay_incumbent:
        if (model.Proto().SerializeToString(deterministic=True) != replay_model_before
                or solver.parameters.SerializeToString(deterministic=True) != replay_parameters_before):
            raise ValueError("incumbent replay model or parameters changed during Solve")
        replay_primitive_checks = require_incumbent_replay_response(
            model, solver, st, positions=replay_seed,
            coordinate_variables={name: (xs[name], ys[name]) for name in ids},
        )
    certified_seed_fallback = False
    better_certified_seed = False
    structurally_certified_seed = False
    phase1_primary_val: Optional[int] = None
    phase1_hier_val: Optional[int] = None
    phase1_soft_val: Optional[int] = None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        pos = {
            bid: (int(solver.Value(xs[bid])), int(solver.Value(ys[bid]))) for bid in ids
        }
        if tie_break_active:
            phase1_primary_val = int(solver.Value(primary))
            if hierarchical_terms:
                phase1_hier_val = int(solver.Value(sum(hierarchical_terms)))
            elif soft_terms:
                phase1_soft_val = int(solver.Value(sum(soft_terms)))
        if internal_seed_legal and not replay_incumbent:
            pos, better_certified_seed = _prefer_certified_seed_positions(
                pos,
                seed,
                dims=dims,
                placement_objective=placement_objective,
                flexible_envelope_constraints=flexible_envelope_constraints,
                bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                cohesion_groups=hierarchical_cohesion_groups,
            )
    elif structural_seed_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        pos = dict(seed)
        structurally_certified_seed = True
    elif (
        certified_source_seed
        and source_seed is not None
        and seed == source_seed
        and certified_source_seed_legal
    ):
        pos = dict(seed)
        certified_seed_fallback = True
    elif (
        strict_width_cap
        or axis_alignment_groups
        or center_alignment_pairs
        or aggregate_envelope_orders
        or adjacency_limits_nm
        or right_edge
        or left_edge
        or floorplan_column_classes
        or floorplan_column_groups
        or flexible_envelope_constraints
        or vertical_order_pairs
    ):
        requirements = tuple(
            name
            for name, present_requirement in (
                ("strict_width", strict_width_cap),
                ("differential_axis", bool(axis_alignment_groups)),
                ("center_alignment", bool(center_alignment_pairs)),
                ("aggregate_envelope_order", bool(aggregate_envelope_orders)),
                ("adjacency", bool(adjacency_limits_nm)),
                ("horizontal_edge", bool(right_edge or left_edge)),
                ("floorplan_columns", bool(floorplan_column_classes)),
                ("floorplan_group_columns", bool(floorplan_column_groups)),
                ("flexible_envelope", bool(flexible_envelope_constraints)),
                ("vertical_order", bool(vertical_order_pairs)),
            )
            if present_requirement
        )
        failure_label = {
            ("strict_width",): "strict width cap",
            ("differential_axis",): "hard differential-axis alignment",
            ("adjacency",): "hard adjacency",
            ("horizontal_edge",): "hard horizontal edge anchor",
        }.get(requirements, "hard placement requirements")
        raise RuntimeError(
            f"{failure_label} has no certified incumbent "
            f"(solver_status={solver.StatusName(st)}, requirements={requirements!r})"
        )
    else:  # nothing better than the band-ordered seed within budget -> honest fallback
        pos = dict(seed)
    phase1 = _solver_evidence(solver, st, objective_name=objective_name)
    if replay_incumbent:
        phase1 = _SolverEvidence("FEASIBLE", "supplied_incumbent_feasibility", None, None)
        primary_evidence_offset = 0
    if primary_evidence_offset:
        evidence_offset = primary_evidence_offset * (
            len(hierarchical_cohesion_groups) * (w_cap + h_bound) + 1
            if hierarchical_terms
            else 1
        )
        phase1 = replace(
            phase1,
            objective=(
                None
                if phase1.objective is None
                else phase1.objective + evidence_offset
            ),
            best_bound=(
                None
                if phase1.best_bound is None
                else phase1.best_bound + evidence_offset
            ),
        )
    if structurally_certified_seed:
        phase1 = _SolverEvidence(
            "FEASIBLE",
            objective_name,
            float(
                _hierarchical_composite_objective_value(
                    pos,
                    dims=dims,
                    placement_objective=placement_objective,
                    hierarchical_cohesion_groups=hierarchical_cohesion_groups,
                    width_bound_nm=w_cap,
                    height_bound_nm=h_bound,
                    flexible_envelope_constraints=flexible_envelope_constraints,
                    bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                )
            ),
            None,
            False,
            None,
        )
    if better_certified_seed:
        selected_seed_is_solver_certified = structural_seed_status in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        )
        phase1 = _SolverEvidence(
            "FEASIBLE",
            objective_name,
            float(
                _hierarchical_composite_objective_value(
                    pos,
                    dims=dims,
                    placement_objective=placement_objective,
                    hierarchical_cohesion_groups=hierarchical_cohesion_groups,
                    width_bound_nm=w_cap,
                    height_bound_nm=h_bound,
                    flexible_envelope_constraints=flexible_envelope_constraints,
                    bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                )
            ),
            phase1.best_bound,
            not selected_seed_is_solver_certified,
            (
                None
                if selected_seed_is_solver_certified
                else (
                    "better_certified_source_seed"
                    if certified_source_seed_legal
                    else "better_replayed_internal_seed"
                )
            ),
        )
    if certified_seed_fallback:
        phase1 = _SolverEvidence(
            "FEASIBLE",
            objective_name,
            None,
            None,
            True,
            "certified_source_seed",
        )
    if (
        placement_objective == "area"
        and not replay_incumbent
        and soft_terms
        and not hierarchical_terms
        and st == cp_model.OPTIMAL
    ):
        primary_value = int(solver.Value(primary))
        phase1_pos = dict(pos)
        model.Add(primary <= primary_value)
        model.Proto().ClearField("solution_hint")
        for bid in ids:
            model.AddHint(xs[bid], phase1_pos[bid][0])
            model.AddHint(ys[bid], phase1_pos[bid][1])
        model.Proto().ClearField("objective")
        model.Minimize(sum(soft_terms))
        solver.parameters.max_deterministic_time = phase_budget.for_phase(
            "soft_position"
        )
        with cp_solve_phase(solver, model, phase="placement.banded.solve.soft_position"):
            st2 = solver.Solve(model)
        if st2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            phase1 = _SolverEvidence(
                phase1.status,
                phase1.objective_name,
                phase1.objective,
                phase1.best_bound,
                True,
                "phase1_incumbent",
                phase1.status,
                phase1.objective_name,
                phase1.objective,
                phase1.best_bound,
            )
        else:
            pos = {
                bid: (int(solver.Value(xs[bid])), int(solver.Value(ys[bid])))
                for bid in ids
            }
            if tie_break_active:
                phase1_soft_val = int(solver.Value(sum(soft_terms)))
            phase2 = _solver_evidence(
                solver, st2, objective_name="soft_position_cost"
            )
            phase1 = _SolverEvidence(
                phase2.status,
                phase2.objective_name,
                phase2.objective,
                phase2.best_bound,
                False,
                None,
                phase1.status,
                phase1.objective_name,
                phase1.objective,
                phase1.best_bound,
            )
    if (
        canonical_tie_break_enabled()
        and placement_objective == "area"
        and not replay_incumbent
        and not better_certified_seed
        and st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        and phase1_primary_val is not None
        and ids
    ):
        model.Add(primary == phase1_primary_val)
        if hierarchical_terms and phase1_hier_val is not None:
            model.Add(sum(hierarchical_terms) == phase1_hier_val)
        elif soft_terms and phase1_soft_val is not None:
            if st != cp_model.OPTIMAL:
                model.Add(sum(soft_terms) <= phase1_soft_val)
                model.Proto().ClearField("solution_hint")
                for bid in ids:
                    model.AddHint(xs[bid], pos[bid][0])
                    model.AddHint(ys[bid], pos[bid][1])
                model.Proto().ClearField("objective")
                model.Minimize(sum(soft_terms))
                soft_solver = cp_model.CpSolver()
                soft_solver.parameters.num_search_workers = 1
                soft_solver.parameters.random_seed = 0
                soft_solver.parameters.max_deterministic_time = min(
                    1.0, max(0.25, remaining_time_s * 0.1)
                )
                _configure_2d_packing_proof(soft_solver)
                with cp_solve_phase(
                    soft_solver, model, phase="placement.banded.solve.soft_position_feasible"
                ):
                    st_soft = soft_solver.Solve(model)
                if st_soft in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    pos = {
                        bid: (
                            int(soft_solver.Value(xs[bid])),
                            int(soft_solver.Value(ys[bid])),
                        )
                        for bid in ids
                    }
                    phase1_soft_val = int(soft_solver.Value(sum(soft_terms)))
            model.Add(sum(soft_terms) == phase1_soft_val)
        canonical_ids = tuple(sorted(ids))
        model.Proto().ClearField("solution_hint")
        for bid in canonical_ids:
            model.AddHint(xs[bid], pos[bid][0])
            model.AddHint(ys[bid], pos[bid][1])
        model.Proto().ClearField("objective")
        model.Minimize(
            sum(
                (len(canonical_ids) - idx) * (2 * xs[bid] + ys[bid])
                for idx, bid in enumerate(canonical_ids)
            )
        )
        tie_solver = cp_model.CpSolver()
        tie_solver.parameters.num_search_workers = 1
        tie_solver.parameters.random_seed = 0
        tie_solver.parameters.max_deterministic_time = min(
            1.0, max(0.25, remaining_time_s * 0.1)
        )
        _configure_2d_packing_proof(tie_solver)
        with cp_solve_phase(
            tie_solver, model, phase="placement.banded.solve.canonical_tie_break"
        ):
            st_tie = tie_solver.Solve(model)
        if st_tie in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            pos = {
                bid: (int(tie_solver.Value(xs[bid])), int(tie_solver.Value(ys[bid])))
                for bid in ids
            }
    has_active_layered_overlap = bool(canonical_overlap_pairs) and any(
        pos[left][0] < pos[right][0] + dims[right][0]
        and pos[right][0] < pos[left][0] + dims[left][0]
        and pos[left][1] < pos[right][1] + dims[right][1]
        and pos[right][1] < pos[left][1] + dims[left][1]
        for left, right in canonical_overlap_pairs
    )
    if replay_incumbent or has_active_layered_overlap or (placement_objective == "area" and (
        lateral
        or cohesion_groups
        or well_cohesion_groups
        or hierarchical_cohesion_groups
    )):
        latest_solved_replay_audit: Dict[str, object] = {
            "failed_rows": (),
            "inventory": (),
            "spec": {},
        }

        def solved_positions_legal(
            candidate: Mapping[str, Tuple[int, int]],
            blocker_out: Optional[List[str]] = None,
            *,
            _collect_audit: bool = True,
        ) -> bool:
            replay_width_bound_nm = _compaction_replay_width_bound(
                candidate,
                dims=dims,
                configured_width_bound_nm=w_cap,
                strict_width_cap=strict_width_cap,
                exact_outline_width=exact_outline_width,
            )
            failed_rows: List[Dict[str, object]] = []
            hard_inventory: List[Dict[str, object]] = []
            hard_spec: Dict[str, object] = {}
            try:
                hard_legal = _banded_seed_satisfies_hard_constraints(
                    candidate,
                    ids=ids,
                    dims=dims,
                    gap_nm=gap,
                    width_bound_nm=replay_width_bound_nm,
                    height_bound_nm=h_bound,
                    bands=bands,
                    adjacency_limits_nm=adjacency_limits_nm,
                    adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
                    order_pairs=order_pairs,
                    vertical_order_pairs=vertical_order_pairs,
                    floorplan_column_classes=floorplan_column_classes,
                    floorplan_column_groups=floorplan_column_groups,
                    bottom_center_keepout_nm=bottom_center_keepout_nm,
                    bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                    bottom_edge_keepout_exempt=tuple(sorted(exempt)),
                    top_edge_keepout_nm=top_edge_keepout_nm,
                    top_edge_keepout_clear=tuple(sorted(top_clear)),
                    solve_cuts=solve_cuts,
                    pair_clearances_nm=pair_clearances_nm,
                    axis_alignment_groups=axis_alignment_groups,
                    center_alignment_pairs=center_alignment_pairs,
                    aggregate_envelope_orders=aggregate_envelope_orders,
                    hierarchical_macro_orders=hierarchical_macro_orders,
                    hierarchical_macro_alignments=hierarchical_macro_alignments,
                    hierarchical_alignment_tolerance_scales_nm=(
                        hierarchical_alignment_tolerance_scales_nm
                    ),
                    aspect_target=aspect_target,
                    aspect_bounds=aspect_bounds,
                    aspect_obligation=aspect_obligation,
                    manufacturing_grid_nm=manufacturing_grid_nm,
                    right_edge=right_edge,
                    left_edge=left_edge,
                    placement_objective=placement_objective,
                    strict_width_cap=strict_width_cap,
                    exact_outline_width=exact_outline_width,
                    lateral_blocks=tuple(sorted(lateral)),
                    flexible_envelope_constraints=flexible_envelope_constraints,
                    rigid_compound_footprints=canonical_compound_footprints,
                    layered_collision_footprints=canonical_layered_footprints,
                    row_bounded_cavity_projections=canonical_cavity_projections,
                    mosaic_guest_slot_projections=canonical_slot_projections,
                    edge_rail_reservation=edge_rail_reservation,
                    overlap_compatible_pairs=canonical_overlap_pairs,
                    enforce_legacy_band_order=enforce_legacy_band_order,
                    blocker_out=blocker_out,
                    diagnostic_out=failed_rows if _collect_audit else None,
                    hard_inventory_out=hard_inventory if _collect_audit else None,
                    hard_spec_out=hard_spec if _collect_audit else None,
                )
            except Exception:
                if not _collect_audit:
                    # Exceptional candidates still retain the complete original
                    # failure audit. Only ordinary Boolean probes omit its copy.
                    solved_positions_legal(candidate, _collect_audit=True)
                    raise
                hard_spec["hierarchical_cohesion_span_caps_nm"] = (
                    _diagnostic_canonical_value(
                        hierarchical_cohesion_span_caps_nm
                    )
                )
                latest_solved_replay_audit.update({
                    "failed_rows": _canonical_full_hard_failure_rows(
                        failed_rows
                    ),
                    "inventory": _canonical_full_hard_failure_rows(
                        hard_inventory
                    ),
                    "spec": hard_spec,
                })
                raise
            if not _collect_audit:
                return hard_legal and _hierarchical_cohesion_spans_within_caps(
                    candidate, dims=dims, caps=hierarchical_cohesion_span_caps_nm,
                )
            cohesion_legal = True
            cohesion_failed_rows: List[Dict[str, object]] = []
            cohesion_inventory: List[Dict[str, object]] = []
            if hard_legal:
                cohesion_legal = _hierarchical_cohesion_spans_within_caps(
                    candidate,
                    dims=dims,
                    caps=hierarchical_cohesion_span_caps_nm,
                    diagnostic_out=cohesion_failed_rows,
                    hard_inventory_out=cohesion_inventory,
                )
            else:
                cohesion_inventory.extend(
                    _hierarchical_cohesion_span_inventory(
                        hierarchical_cohesion_span_caps_nm
                    )
                )
            hard_spec["hierarchical_cohesion_span_caps_nm"] = (
                _diagnostic_canonical_value(
                    hierarchical_cohesion_span_caps_nm
                )
            )
            latest_solved_replay_audit.update({
                "failed_rows": _canonical_full_hard_failure_rows((
                    *failed_rows,
                    *cohesion_failed_rows,
                )),
                "inventory": _canonical_full_hard_failure_rows((
                    *hard_inventory,
                    *cohesion_inventory,
                )),
                "spec": hard_spec,
            })
            return hard_legal and cohesion_legal

        # Full-2D binary probes still replay this exact hard-constraint domain;
        # only terminal/final/error boundaries request the large audit artifact.
        solved_positions_legal.boolean_replay = lambda candidate: solved_positions_legal(
            candidate, _collect_audit=False,
        )
        solved_legal = solved_positions_legal(pos)
        if replay_incumbent and (pos != replay_seed or not solved_legal):
            raise ValueError("incumbent replay candidate fails full hard/caps replay")
        if solved_legal and not area_target_reached(area_target,
            area_nm2=_placement_primary_extent_nm(pos, dims=dims, placement_objective="area",
                flexible_envelope_constraints=flexible_envelope_constraints, bottom_edge_keepout_nm=bottom_edge_keepout_nm),
            hard_replayed=solved_legal, incumbent_replay=replay_incumbent):
            compacted_pos, compacted_area_frontier = (
                (dict(pos), False) if replay_incumbent else
                _compact_area_frontier_positions(
                    pos,
                    dims=dims,
                    cohesion_groups=tuple(
                        (
                            *cohesion_groups,
                            *well_cohesion_groups,
                            *hierarchical_cohesion_groups,
                            *((owner,) for owner in sorted(compound_owners)),
                        )
                    ),
                    lateral_blocks=tuple(sorted(lateral)),
                    order_pairs=order_pairs,
                    manufacturing_grid_nm=manufacturing_grid_nm,
                    legal=solved_positions_legal,
                )
            )
            if compacted_area_frontier:
                pos = compacted_pos
                compacted_area = float(
                    _placement_primary_extent_nm(
                        pos,
                        dims=dims,
                        placement_objective="area",
                        flexible_envelope_constraints=(
                            flexible_envelope_constraints
                        ),
                        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                    )
                )
                phase1 = phase1.after_area_frontier_compaction(compacted_area)
            try:
                full2d_solver_evidence: list[Mapping[str, object]] = []
                full2d_pos, full2d_graph, full2d_changed = (
                    refine_below_area_target(_compact_full2d_master_positions,
                        pos,
                        target=area_target, hard_replayed=solved_legal, incumbent_replay=replay_incumbent,
                        area_nm2=_placement_primary_extent_nm(pos, dims=dims, placement_objective="area",
                            flexible_envelope_constraints=flexible_envelope_constraints, bottom_edge_keepout_nm=bottom_edge_keepout_nm),
                        dims=dims,
                        minimum_clearance_nm=gap,
                        manufacturing_grid_nm=manufacturing_grid_nm,
                        axis_alignment_groups=axis_alignment_groups,
                        center_alignment_pairs=center_alignment_pairs,
                        pair_axis_authority=(
                            _hierarchical_full2d_pair_axis_authority(
                                hierarchical_macro_orders
                            )
                        ),
                        pair_axis_preference=(
                            _hierarchical_full2d_pair_axis_preference(
                                hierarchical_macro_orders
                            )
                        ),
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
                        overlap_compatible_pairs=canonical_overlap_pairs,
                        rigid_compound_footprints=canonical_compound_footprints,
                        layered_collision_footprints=canonical_layered_footprints,
                        row_bounded_cavity_projections=canonical_cavity_projections,
                        mosaic_guest_slot_projections=canonical_slot_projections,
                        edge_rail_reservation=edge_rail_reservation,
                        order_pairs=order_pairs,
                        vertical_order_pairs=vertical_order_pairs,
                        pair_clearances_nm=pair_clearances_nm,
                        hierarchical_cohesion_span_caps_nm=hierarchical_cohesion_span_caps_nm,
                        floorplan_column_groups=floorplan_column_groups,
                        left_edge=left_edge,
                        legal=solved_positions_legal,
                        legal_replay_audit=lambda: dict(
                            latest_solved_replay_audit
                        ),
                        solve_evidence_sink=full2d_solver_evidence,
                    )
                )
            except ValueError as exc:
                # A certified compound can deliberately contain a foreign
                # rectangular bbox in a proven artwork void. Such a seed has no
                # rectangle-only pair assignment; retain the already replayed
                # compound path instead of inventing an H/V edge for it.
                if replay_incumbent or "seed is not clearance-legal for pair" not in str(exc):
                    raise
                full2d_pos = dict(pos)
                full2d_graph = None
                full2d_changed = False
            if full2d_changed:
                pos = full2d_pos
                compacted_area = float(
                    _placement_primary_extent_nm(
                        pos,
                        dims=dims,
                        placement_objective="area",
                        flexible_envelope_constraints=flexible_envelope_constraints,
                        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
                    )
                )
                phase1 = replace(
                    phase1,
                    status="FEASIBLE",
                    objective_name="full2d_master_dag_compaction",
                    objective=compacted_area,
                    used_fallback=False,
                    fallback_kind=None,
                    full2d_compaction_graph=full2d_graph,
                    full2d_solver_evidence=tuple(full2d_solver_evidence),
                )
            else:
                if full2d_graph is not None:
                    phase1 = replace(
                        phase1,
                        full2d_compaction_graph=full2d_graph,
                        full2d_solver_evidence=tuple(full2d_solver_evidence),
                    )
    flexible_projections = tuple(
        FlexibleEnvelopeProjection(
            constraint=row,
            bbox_nm=(
                min(
                    pos[name][0]
                    + flexible_physical_offsets[row.group][name][0]
                    for name in row.members
                ),
                min(
                    pos[name][1]
                    + flexible_physical_offsets[row.group][name][1]
                    for name in row.members
                ),
                max(
                    pos[name][0]
                    + flexible_physical_offsets[row.group][name][2]
                    for name in row.members
                ),
                max(
                    pos[name][1]
                    + flexible_physical_offsets[row.group][name][3]
                    for name in row.members
                ),
            ),
            foreign_clearances_nm=tuple(
                (member, foreign, clearance)
                for (member, foreign), clearance in sorted(
                    flexible_foreign_clearances[row.group].items()
                )
            ),
        )
        for row in flexible_envelope_constraints
    )
    phase1 = replace(
        phase1,
        flexible_envelope_projections=flexible_projections,
    )
    used_bboxes = [
        (pos[name][0], pos[name][1], pos[name][0] + dims[name][0], pos[name][1] + dims[name][1])
        for name in ids
    ] + [projection.bbox_nm for projection in flexible_projections]
    right_rail_margin_nm = int(
        getattr(edge_rail_reservation, "right_margin_nm", 0) or 0
    )
    top_rail_margin_nm = int(
        getattr(edge_rail_reservation, "top_margin_nm", 0) or 0
    )
    w_used = (
        max(bbox[2] for bbox in used_bboxes)
        + right_rail_margin_nm
        - min(bbox[0] for bbox in used_bboxes)
    )
    h_used = (
        max(bbox[3] for bbox in used_bboxes)
        + top_rail_margin_nm
        - (
            0
            if int(bottom_edge_keepout_nm) > 0
            else min(bbox[1] for bbox in used_bboxes)
        )
    )
    if replay_incumbent:
        from clay.layout.place.generated_incumbent import current_generated_incumbent_phase
        if (pos != replay_seed or phase1.full2d_compaction_graph is None
                or phase1.full2d_solver_evidence
                or model.Proto().SerializeToString(deterministic=True) != replay_model_before
                or solver.parameters.SerializeToString(deterministic=True) != replay_parameters_before):
            raise ValueError("incumbent replay changed after its certified response")
        _complete_incumbent_replay_stage(
            phase=current_generated_incumbent_phase() or "standalone_banded",
            frame=current_stage_frame(), positions=pos, graph=phase1.full2d_compaction_graph,
            model=model, solver=solver, status=st, primitive_checks=replay_primitive_checks,
            full_hard_audit=latest_solved_replay_audit,
            cohesion_caps=hierarchical_cohesion_span_caps_nm,
            original_model_bytes=replay_original_model_bytes,
        )
    return (
        pos,
        w_used,
        h_used,
        phase1,
    )

def _legalize_fixed_outline_seed_x(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    bands: Mapping[str, int],
    width_nm: int,
    height_nm: Optional[int] = None,
    movable_y: Sequence[str] = (),
    movable_y_lower_bounds_nm: Mapping[str, int] = MappingProxyType({}),
    bottom_origin_lower_bounds_nm: Mapping[str, int] = MappingProxyType({}),
    gap_nm: int,
    max_time_s: float,
    manufacturing_grid_nm: int,
    floorplan_column_classes: Sequence[Tuple[str, int, int, int]],
    floorplan_column_groups: Sequence[_FloorplanColumnGroup],
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder] = (),
    hierarchical_cohesion_groups: Sequence[Sequence[str]] = (),
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder] = (),
    hierarchical_macro_alignments: Sequence[_HierarchicalMacroAlignment] = (),
    hierarchical_alignment_tolerance_scales_nm: (
        _HierarchicalAlignmentToleranceScalesNm
    ) = (),
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    row_bounded_cavity_projections: Tuple[_RowBoundedCavityProjection, ...] = (),
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = (),
    edge_rail_reservation: Any = None,
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]],
    center_alignment_pairs: Sequence[Tuple[str, str, str]] = (),
    order_pairs: Sequence[Tuple[str, str]],
    vertical_order_pairs: Sequence[Tuple[str, str]] = (),
    adjacency_limits_nm: Sequence[Tuple[str, str, int]],
    adjacency_scope_offsets_nm: Mapping[str, _Bbox],
    right_edge: Sequence[str],
    left_edge: Sequence[str],
    exact_outline: bool = True,
    aspect_target: float = 1.0,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = False,
    assumption_inventory_sink: Optional[
        Callable[[Mapping[str, object]], None]
    ] = None,
    production_evidence_context_sink: Optional[Callable[[Any], None]] = None,
    production_evidence_sink: Optional[
        Callable[[Mapping[str, object]], None]
    ] = None,
) -> Dict[str, Tuple[int, int]]:
    """Re-legalize X plus an explicitly scoped lateral-adjacency Y subset.

    Independent band solves are a fast way to prove each row's 2-D packing, but
    one rigid translation per band is incomplete: different rows may require
    different within-band X permutations to realize one global ordinal column
    grammar.  Most Y coordinates remain frozen.  Lateral endpoints of declared
    adjacency may move in Y because two neighbors of one tall lateral block can
    be impossible to realize on one frozen shelf.  The complete 2-D master still
    independently replays the returned seed before it can sign anything.

    Opt-in production evidence captures the complete model before objective
    prefix restrictions. Its receipt describes the first SAT incumbent, which
    can precede later lexicographic refinement of the returned positions.
    Evidence capture and replay do not run an additional solver.
    """
    from ortools.sat.python import cp_model

    hierarchical_alignment_tolerance_scales_nm = (
        _require_hierarchical_alignment_tolerance_scales_nm(
            hierarchical_alignment_tolerance_scales_nm,
            alignments=hierarchical_macro_alignments,
        )
    )
    ids = tuple(sorted(seed))
    if not ids or set(ids) != set(dims) or set(ids) != set(bands):
        raise ValueError("fixed-outline X legalizer requires one complete seed")
    width = int(width_nm)
    gap = int(gap_nm)
    grid = int(manufacturing_grid_nm)
    if width <= 0 or gap < 0 or grid <= 0:
        raise ValueError("fixed-outline X legalizer has invalid physical authority")
    if type(exact_outline) is not bool:
        raise TypeError("X legalizer exact_outline must be an exact boolean")
    if type(aspect_obligation) is not bool:
        raise TypeError("X legalizer aspect obligation must be an exact boolean")
    if not math.isfinite(float(max_time_s)) or float(max_time_s) <= 0.0:
        raise ValueError("fixed-outline X legalizer requires a positive solve budget")
    if any(
        type(dims[name][0]) is not int
        or type(dims[name][1]) is not int
        or dims[name][0] <= 0
        or dims[name][1] <= 0
        or dims[name][0] > width
        for name in ids
    ):
        raise ValueError("fixed-outline X legalizer has malformed block dimensions")
    try:
        vertical_rows = tuple(vertical_order_pairs)
    except TypeError as exc:
        raise ValueError("fixed-outline X legalizer vertical order is malformed") from exc
    if any(
        not isinstance(pair, (tuple, list)) or len(pair) != 2
        or any(type(name) is not str or not name for name in pair)
        or pair[0] == pair[1] or pair[0] not in dims or pair[1] not in dims
        for pair in vertical_rows
    ):
        raise ValueError("fixed-outline X legalizer vertical order must name two distinct live blocks")
    vertical_order_pairs = tuple(sorted({tuple(pair) for pair in vertical_rows}))
    movable_y_ids = tuple(movable_y)
    if (
        any(type(name) is not str or not name for name in movable_y_ids)
        or movable_y_ids != tuple(sorted(movable_y_ids))
        or len(movable_y_ids) != len(set(movable_y_ids))
        or not set(movable_y_ids) <= set(ids)
    ):
        raise ValueError("fixed-outline X legalizer movable Y scope is malformed")
    movable_y_set = set(movable_y_ids)
    if not isinstance(movable_y_lower_bounds_nm, Mapping) or any(
        type(name) is not str
        or name not in movable_y_set
        or type(value) is not int
        or value < 0
        for name, value in movable_y_lower_bounds_nm.items()
    ):
        raise ValueError("fixed-outline X legalizer movable Y bounds are malformed")
    if (
        not isinstance(bottom_origin_lower_bounds_nm, Mapping)
        or any(
            type(name) is not str
            or name not in ids
            or type(value) is not int
            or value < 0
            for name, value in bottom_origin_lower_bounds_nm.items()
        )
        or (
            bottom_origin_lower_bounds_nm
            and set(bottom_origin_lower_bounds_nm) != set(ids)
        )
    ):
        raise ValueError("fixed-outline X legalizer bottom-origin bounds are malformed")
    height = (
        int(height_nm)
        if height_nm is not None
        else max(int(seed[name][1]) + int(dims[name][1]) for name in ids) + gap
    )
    if height <= 0 or any(
        int(movable_y_lower_bounds_nm.get(name, 0)) + dims[name][1] > height
        for name in movable_y_ids
    ):
        raise ValueError("fixed-outline X legalizer height is below movable Y scope")
    cohesion_groups = tuple(
        tuple(group) for group in hierarchical_cohesion_groups
    )
    if any(
        type(raw) is not tuple
        or len(raw) < 2
        or raw != tuple(sorted(raw))
        or len(set(raw)) != len(raw)
        or not set(raw) <= set(ids)
        for raw in hierarchical_cohesion_groups
    ):
        raise ValueError(
            "fixed-outline X legalizer hierarchy groups are malformed"
        )
    collision_atoms = _collision_atoms_by_owner(
        dims=dims,
        rigid_compound_footprints=rigid_compound_footprints,
        layered_collision_footprints=layered_collision_footprints,
    )
    refined_collision_owners = {
        owner for owner, _rectangles in rigid_compound_footprints
    } | {
        owner for owner, _atoms in layered_collision_footprints
    }
    compatible_pairs = set(
        _canonical_overlap_compatible_pairs(
            overlap_compatible_pairs,
            block_ids=ids,
            pair_clearances_nm=pair_clearances_nm,
        )
    )
    refined_collision = bool(
        rigid_compound_footprints
        or layered_collision_footprints
        or compatible_pairs
    )
    canonical_cavity_projections = _canonical_row_bounded_cavity_projections(
        row_bounded_cavity_projections,
        dims=dims,
    )
    canonical_slot_projections = _canonical_mosaic_guest_slot_projections(
        mosaic_guest_slot_projections, dims=dims, minimum_clearance_nm=gap,
    )

    model = cp_model.CpModel()
    assumption_by_id: Dict[str, Any] = {}
    assumption_index_to_id: Dict[int, str] = {}
    assumption_trace_by_id: Dict[str, ConstraintObligationTrace] = {}

    def obligation_trace(core_id: str) -> Tuple[str, str]:
        family, _separator, raw_index = core_id.partition(":")
        if core_id == "layered_collision":
            return (
                "physical_nonoverlap:canonical_layered_atoms",
                "clay/layout/place_blocks_2d.py",
            )
        if core_id == "aspect_obligation":
            return "declared_aspect_ratio", "clay/layout/policy.py"
        if family == "horizontal_edge":
            return core_id, "clay/layout/intent.py"
        if family == "edge_rail":
            row = next(row for row in edge_rail_reservation.demands if row.demand_id == raw_index)
            return row.demand_id, row.declaration_ref
        if core_id == "pair_clearance":
            return (
                "physical_pair_clearance:canonical_master_pairs",
                "clay/layout/place_blocks_2d.py",
            )
        if family == "axis_alignment":
            index = int(raw_index)
            members = axis_alignment_groups[index]
            identity = ":".join(name for name, _offset in members)
            return f"axis_alignment:{identity}", "clay/layout/intent.py"
        if family == "center_alignment":
            index = int(raw_index)
            left, right, axis = center_alignment_pairs[index]
            return (
                f"center_alignment:{left}:{right}:{axis}",
                "clay/layout/spatial/intent_projection.py",
            )
        if family == "hierarchy_order":
            index = int(raw_index)
            scope, before, after, axis, strength, _left, _right = (
                hierarchical_macro_orders[index]
            )
            obligation_id = _hierarchical_macro_order_obligation_id(
                hierarchical_macro_orders[index]
            )
            return obligation_id, "clay/layout/spatial/intent_projection.py"
        if family == "hierarchy_alignment":
            index = int(raw_index)
            scope, axis, edge, _tolerance, _groups = (
                _hierarchical_macro_alignment_parts(
                    hierarchical_macro_alignments[index]
                )
            )
            obligation_id = (
                scope.removeprefix(SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX)
                if scope.startswith(SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX)
                else f"hierarchical_alignment:{scope}:{axis}:{edge}"
            )
            return obligation_id, "clay/layout/spatial/intent_projection.py"
        if family == "aggregate_order":
            index = int(raw_index)
            before, after, axis, _left, _right = aggregate_envelope_orders[index]
            return (
                f"aggregate_envelope_order:{before}:{after}:{axis}",
                "clay/layout/spatial/intent_projection.py",
            )
        if family == "row_bounded_cavity":
            index = int(raw_index)
            row = canonical_cavity_projections[index]
            return (
                row.obligation_id,
                "clay/layout/spatial/intent_projection.py",
            )
        if family == "mosaic_guest_slot":
            row = canonical_slot_projections[int(raw_index)]
            return row.obligation_id, "clay/layout/spatial/mosaic_guest_slot.py"
        if core_id == "floorplan_column":
            return "floorplan_column_order", "clay/layout/place_blocks_2d.py"
        if core_id == "floorplan_column_group":
            return "floorplan_column_group_order", "clay/layout/place_blocks_2d.py"
        if family == "adjacency":
            index = int(raw_index)
            left, right, maximum = adjacency_limits_nm[index]
            return (
                f"adjacent:{left}:{right}:maximum_gap_nm={maximum}",
                "clay/layout/intent.py",
            )
        if family == "order_pair":
            index = int(raw_index)
            left, right = order_pairs[index]
            return f"horizontal_order:{left}:{right}", "clay/layout/intent.py"
        if family == "vertical_order_pair":
            lower, upper = vertical_order_pairs[int(raw_index)]
            return f"vertical_order:{lower}:{upper}", "clay/layout/intent.py"
        raise RuntimeError(
            f"fixed-outline X legalizer lacks obligation trace for {core_id!r}"
        )

    def assumption_for(core_id: str) -> Any:
        existing = assumption_by_id.get(core_id)
        if existing is not None:
            return existing
        assumption = model.NewBoolVar(f"legalizer_assumption_{len(assumption_by_id)}")
        assumption_by_id[core_id] = assumption
        assumption_index_to_id[assumption.Index()] = core_id
        obligation_id, declaration_source = obligation_trace(core_id)
        assumption_trace_by_id[core_id] = ConstraintObligationTrace(
            row_id=core_id,
            assumption_literal=assumption.Name(),
            assumption_index=assumption.Index(),
            obligation_id=obligation_id,
            declaration_source=declaration_source,
            witness_status="NOT_EVALUATED",
        )
        model.AddAssumption(assumption)
        return assumption

    xs = {
        name: model.NewIntVar(0, width - dims[name][0], f"x_{name}")
        for name in ids
    }
    ys = {
        name: (
            model.NewIntVar(
                int(movable_y_lower_bounds_nm.get(name, 0)),
                height - dims[name][1],
                f"y_{name}",
            )
            if name in movable_y_set
            else model.NewConstant(int(seed[name][1]))
        )
        for name in ids
    }
    _require_edge_rail_reservation(edge_rail_reservation, dims=dims)
    if edge_rail_reservation is not None:
        edge_rail_reservation.add_constraints(
            model, xs=xs, ys=ys, dims=dims,
            assumption_for=lambda demand_id: assumption_for(f"edge_rail:{demand_id}"),
        )
    if bottom_origin_lower_bounds_nm:
        for name in ids:
            model.Add(ys[name] >= int(bottom_origin_lower_bounds_nm[name]))
        bottom_slack = model.NewIntVar(0, height, "physical_bottom_slack")
        model.AddMinEquality(
            bottom_slack,
            [
                ys[name] - int(bottom_origin_lower_bounds_nm[name])
                for name in ids
            ],
        )
        model.Add(bottom_slack == 0)
    collision_index = 0
    collision_raw_atom_pairs = 0
    collision_unique_forbidden_boxes = 0
    if not refined_collision:
        x_intervals = [
            model.NewFixedSizeIntervalVar(
                xs[name], dims[name][0] + gap, f"xi_{name}"
            )
            for name in ids
        ]
        y_intervals = [
            model.NewFixedSizeIntervalVar(
                ys[name], dims[name][1] + gap, f"yi_{name}"
            )
            for name in ids
        ]
        model.AddNoOverlap2D(x_intervals, y_intervals)
    else:
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                if (left, right) in compatible_pairs:
                    continue
                required = int(pair_clearances_nm.get((left, right), gap))
                collision_assumption = assumption_for("layered_collision")
                forbidden_boxes, raw_count, unique_count = (
                    _pair_collision_forbidden_box_cover(
                        collision_atoms[left],
                        collision_atoms[right],
                        required_nm=required,
                    )
                )
                collision_raw_atom_pairs += raw_count
                collision_unique_forbidden_boxes += unique_count
                for fx0, fy0, fx1, fy1 in forbidden_boxes:
                    collision_index += 1
                    left_of = model.NewBoolVar(
                        f"seed_atom_left_{collision_index}_{left}_{right}"
                    )
                    right_of = model.NewBoolVar(
                        f"seed_atom_right_{collision_index}_{left}_{right}"
                    )
                    below = model.NewBoolVar(
                        f"seed_atom_below_{collision_index}_{left}_{right}"
                    )
                    above = model.NewBoolVar(
                        f"seed_atom_above_{collision_index}_{left}_{right}"
                    )
                    model.Add(xs[right] - xs[left] >= fx1).OnlyEnforceIf(left_of)
                    model.Add(xs[right] - xs[left] <= fx0 - 1).OnlyEnforceIf(
                        right_of
                    )
                    model.Add(ys[right] - ys[left] >= fy1).OnlyEnforceIf(below)
                    model.Add(ys[right] - ys[left] <= fy0 - 1).OnlyEnforceIf(above)
                    model.AddBoolOr(
                        (left_of, right_of, below, above)
                    ).OnlyEnforceIf(collision_assumption)
        if not (
            0 <= collision_index
            <= collision_unique_forbidden_boxes
            <= collision_raw_atom_pairs
        ):
            raise AssertionError(
                "fixed-outline collision quotient cardinality is inconsistent"
            )
    x_min = model.NewIntVar(0, width, "physical_x_min")
    x_max = model.NewIntVar(0, width, "physical_x_max")
    model.AddMinEquality(x_min, [xs[name] for name in ids])
    model.AddMaxEquality(x_max, [xs[name] + dims[name][0] for name in ids])
    model.Add(x_min == 0)
    if exact_outline:
        model.Add(x_max == width)
    if aspect_obligation:
        _add_directional_aspect_constraints(
            model,
            xs=xs,
            ys=ys,
            dims=dims,
            width_bound=width,
            height_bound=height,
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
            enforcement_literal=assumption_for("aspect_obligation"),
        )
    _add_manufacturing_grid_constraints(model, xs, ys, grid)
    _add_hard_horizontal_edge_constraints(
        model,
        xs,
        dims,
        width_bound=x_max if not exact_outline else width,
        right_edge=right_edge,
        left_edge=left_edge,
        assumption_for=assumption_for,
    )
    _add_pair_clearance_constraints(
        model,
        xs,
        ys,
        dims,
        pair_clearances_nm,
        excluded_blocks=tuple(sorted(refined_collision_owners)),
        assumption_for=assumption_for,
    )
    _add_axis_alignment_constraints(
        model,
        xs,
        axis_alignment_groups,
        assumption_for=assumption_for,
    )
    _add_center_alignment_constraints(
        model,
        xs,
        ys,
        dims,
        center_alignment_pairs,
        assumption_for=assumption_for,
    )
    height_bound = height
    _add_hierarchical_macro_constraints(
        model,
        xs,
        ys,
        dims,
        hierarchical_macro_orders,
        hierarchical_macro_alignments,
        width_bound_nm=width,
        height_bound_nm=height_bound,
        center_step_nm=grid,
        gap_nm=gap,
        tolerance_scales_nm=hierarchical_alignment_tolerance_scales_nm,
        assumption_for=assumption_for,
    )
    _add_aggregate_envelope_order_constraints(
        model,
        xs,
        ys,
        dims,
        aggregate_envelope_orders,
        width_bound_nm=width,
        height_bound_nm=height_bound,
        center_step_nm=grid,
        assumption_for=assumption_for,
    )
    _add_row_bounded_cavity_constraints(
        model,
        xs,
        ys,
        dims,
        canonical_cavity_projections,
        minimum_clearance_nm=gap,
        assumption_for=assumption_for,
    )
    add_mosaic_guest_slot_constraints(
        model, xs, ys, canonical_slot_projections, assumption_for=assumption_for,
    )
    _add_floorplan_column_constraints(
        model,
        xs,
        dims,
        floorplan_column_classes,
        assumption_for=assumption_for,
    )
    _add_floorplan_column_group_constraints(
        model,
        xs,
        floorplan_column_groups,
        assumption_for=assumption_for,
    )
    _add_adjacency_constraints(
        model,
        xs,
        ys,
        dims,
        adjacency_limits_nm,
        scope_offsets_nm=adjacency_scope_offsets_nm,
        minimum_gap_nm=gap,
        projection_nm=grid,
        assumption_for=assumption_for,
    )
    for index, (left, right) in enumerate(order_pairs):
        constraint = model.Add(
            xs[right]
            >= xs[left]
            + dims[left][0]
            + _required_pair_gap_nm(
                left,
                right,
                gap,
                pair_clearances_nm,
            )
        )
        constraint.OnlyEnforceIf(
            assumption_for(f"order_pair:{index}")
        )
    for index, (lower, upper) in enumerate(vertical_order_pairs):
        model.Add(
            ys[upper] >= ys[lower] + dims[lower][1]
            + _required_pair_gap_nm(lower, upper, gap, pair_clearances_nm)
        ).OnlyEnforceIf(assumption_for(f"vertical_order_pair:{index}"))
    movements = []
    for name in ids:
        delta = model.NewIntVar(0, width, f"dx_{name}")
        model.AddAbsEquality(delta, xs[name] - int(seed[name][0]))
        movements.append(delta)
        model.AddHint(xs[name], int(seed[name][0]))
        if name in movable_y_set:
            y_delta = model.NewIntVar(0, height, f"dy_{name}")
            model.AddAbsEquality(y_delta, ys[name] - int(seed[name][1]))
            movements.append(y_delta)
            model.AddHint(ys[name], int(seed[name][1]))
    cohesion_spans = []
    for group_index, group in enumerate(cohesion_groups):
        low = model.NewIntVar(0, width, f"cohesion_x_low_{group_index}")
        high = model.NewIntVar(0, width, f"cohesion_x_high_{group_index}")
        span = model.NewIntVar(1, width, f"cohesion_x_span_{group_index}")
        model.AddMinEquality(low, [xs[name] for name in group])
        model.AddMaxEquality(
            high, [xs[name] + dims[name][0] for name in group]
        )
        model.Add(span == high - low)
        cohesion_spans.append(span)
        if set(group) & movable_y_set:
            y_low = model.NewIntVar(0, height, f"cohesion_y_low_{group_index}")
            y_high = model.NewIntVar(0, height, f"cohesion_y_high_{group_index}")
            y_span = model.NewIntVar(1, height, f"cohesion_y_span_{group_index}")
            model.AddMinEquality(y_low, [ys[name] for name in group])
            model.AddMaxEquality(
                y_high,
                [ys[name] + dims[name][1] for name in group],
            )
            model.Add(y_span == y_high - y_low)
            cohesion_spans.append(y_span)
    # Keep the objective dimensions explicit.  The former mixed-radix scalar
    # multiplied outline nm by aggregate-span nm and movement nm.  A realistic
    # LDO width plus nested cohesion groups was sufficient for CP-SAT to reject
    # that otherwise feasible model as a possible int64 overflow.  Sequential
    # solves express the same lexicographic policy without inventing a compound
    # physical unit: outline first, then summed cohesion span, then displacement.
    objective_stages = []
    if not exact_outline:
        objective_stages.append(("outline_nm", x_max))
    if cohesion_spans:
        objective_stages.append(("cohesion_span_nm", sum(cohesion_spans)))
    objective_stages.append(("movement_nm", sum(movements)))
    production_evidence_context = None
    if (
        assumption_inventory_sink is not None
        or production_evidence_context_sink is not None
        or production_evidence_sink is not None
    ):
        hard_background_ids = (
            "derived_envelopes",
            "fixed_domain",
            *(("fixed_y",) if set(ids) - movable_y_set else ()),
            "manufacturing_grid",
            "physical_origin",
            *(("bottom_origin",) if bottom_origin_lower_bounds_nm else ()),
            *(("exact_outline",) if exact_outline else ()),
            *(("no_overlap2d",) if not refined_collision else ()),
        )
        assumption_rows = tuple(
            {
                "row_kind": "ASSUMPTION",
                "family": row.row_id.split(":", 1)[0],
                **row.canonical_data(),
            }
            for row in sorted(
                assumption_trace_by_id.values(), key=lambda item: item.row_id
            )
        )
        active_assumption_literals = tuple(
            model.GetBoolVarFromProtoIndex(int(index)).Name()
            for index in model.Proto().assumptions
        )
        expected_literals = tuple(
            row["assumption_literal"] for row in assumption_rows
        )
        if set(active_assumption_literals) != set(expected_literals):
            raise RuntimeError(
                "fixed-outline active assumptions differ from attributed rows"
            )
        assumption_snapshot = {
            "schema": "clay.fixed-outline-row-inventory/v1",
            "assumption_rows": assumption_rows,
            "hard_background_row_ids": tuple(
                f"hard_background:{name}" for name in hard_background_ids
            ),
            "hard_background_rows": tuple(
                {
                    "row_kind": "HARD_BACKGROUND",
                    "row_id": f"hard_background:{name}",
                    "family": "hard_background",
                    "obligation_id": name,
                    "declaration_source": "clay/layout/place_blocks_2d.py",
                }
                for name in hard_background_ids
            ),
            "active_assumption_literals": active_assumption_literals,
            "model_proto_sha256": hashlib.sha256(
                model.Proto().SerializeToString(deterministic=True)
            ).hexdigest(),
        }
        if assumption_inventory_sink is not None:
            assumption_inventory_sink(assumption_snapshot)
        if (
            production_evidence_context_sink is not None
            or production_evidence_sink is not None
        ):
            from clay.layout.place.production_model_evidence import (
                FixedOutlineProductionEvidenceContext,
            )
            production_evidence_context = FixedOutlineProductionEvidenceContext(
                model=model,
                assumption_rows=assumption_rows,
                coordinate_indices={
                    name: (xs[name].Index(), ys[name].Index()) for name in ids
                },
                stage_inputs={
                    "seed": dict(seed), "dims": dict(dims), "bands": dict(bands),
                    "width_nm": width, "height_nm": height, "gap_nm": gap,
                    "manufacturing_grid_nm": grid,
                    "movable_y": movable_y_ids,
                    "movable_y_lower_bounds_nm": dict(movable_y_lower_bounds_nm),
                    "bottom_origin_lower_bounds_nm": dict(bottom_origin_lower_bounds_nm),
                    "floorplan_column_classes": tuple(floorplan_column_classes),
                    "floorplan_column_groups": tuple(floorplan_column_groups),
                    "aggregate_envelope_orders": tuple(aggregate_envelope_orders),
                    "hierarchical_cohesion_groups": cohesion_groups,
                    "hierarchical_macro_orders": tuple(hierarchical_macro_orders),
                    "hierarchical_macro_alignments": tuple(hierarchical_macro_alignments),
                    "hierarchical_alignment_tolerance_scales_nm": hierarchical_alignment_tolerance_scales_nm,
                    "rigid_compound_footprints": rigid_compound_footprints,
                    "layered_collision_footprints": layered_collision_footprints,
                    "row_bounded_cavity_projections": canonical_cavity_projections,
                    **({"mosaic_guest_slot_projections": canonical_slot_projections}
                       if canonical_slot_projections else {}),
                    **({"edge_rail_reservation": edge_rail_reservation}
                       if edge_rail_reservation is not None else {}),
                    "overlap_compatible_pairs": tuple(sorted(compatible_pairs)),
                    "pair_clearances_nm": dict(pair_clearances_nm),
                    "axis_alignment_groups": tuple(axis_alignment_groups),
                    "center_alignment_pairs": tuple(center_alignment_pairs),
                    "order_pairs": tuple(order_pairs),
                    # Keep historical empty/default stage serialization exact.
                    **({"vertical_order_pairs": vertical_order_pairs}
                       if vertical_order_pairs else {}),
                    "adjacency_limits_nm": tuple(adjacency_limits_nm),
                    "adjacency_scope_offsets_nm": dict(adjacency_scope_offsets_nm),
                    "right_edge": tuple(right_edge), "left_edge": tuple(left_edge),
                    "exact_outline": exact_outline, "aspect_target": aspect_target,
                    "aspect_bounds": aspect_bounds, "aspect_obligation": aspect_obligation,
                },
            )
            if production_evidence_context_sink is not None:
                production_evidence_context_sink(production_evidence_context)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    stage_budget = float(max_time_s) / len(objective_stages)
    solver.parameters.max_deterministic_time = stage_budget
    _configure_2d_packing_proof(solver)
    status = None
    certified_positions: Optional[Dict[str, Tuple[int, int]]] = None
    for stage_index, (stage_name, objective) in enumerate(objective_stages):
        model.Minimize(objective)
        validation_error = model.Validate()
        if validation_error:
            raise RuntimeError(
                "invalid fixed-outline X legalizer "
                f"at {stage_name}: {validation_error}"
            )
        with cp_solve_phase(solver, model, phase="placement.x_legalizer.solve.stage"):
            status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if status == cp_model.UNKNOWN and certified_positions is not None:
                return certified_positions
            remaining_core_budget = stage_budget * (
                len(objective_stages) - stage_index - 1
            )
            raw_core_indices = (
                tuple(solver.SufficientAssumptionsForInfeasibility())
                if status == cp_model.INFEASIBLE
                else ()
            )
            unknown_core_indices = tuple(sorted(
                index
                for index in raw_core_indices
                if index not in assumption_index_to_id
            ))
            if unknown_core_indices:
                raise RuntimeError(
                    "fixed-outline X legalizer returned unmapped assumption "
                    f"indices: {unknown_core_indices!r}"
                )
            core_ids = tuple(sorted({
                assumption_index_to_id[index] for index in raw_core_indices
            }))
            inconclusive_families: Tuple[str, ...] = ()
            if core_ids and remaining_core_budget > 0.0:
                # The failed stage cannot spend budgets reserved for later
                # lexicographic objectives. Reuse exactly that remainder for
                # deterministic family deletion probes on the objective-free
                # model. A family is removed only after the reduced assumption
                # set is independently proved INFEASIBLE; FEASIBLE and UNKNOWN
                # both retain it, so the resulting set remains a sufficient
                # core without increasing the caller's total solve budget.
                model.ClearObjective()
                families = tuple(sorted({
                    core_id.split(":", 1)[0] for core_id in core_ids
                }))
                probe_budget = remaining_core_budget / len(families)
                retained = set(core_ids)
                unknown = []
                for family in families:
                    family_ids = {
                        core_id
                        for core_id in retained
                        if core_id.split(":", 1)[0] == family
                    }
                    if not family_ids:
                        continue
                    trial = tuple(sorted(retained - family_ids))
                    model.ClearAssumptions()
                    model.AddAssumptions(
                        assumption_by_id[core_id] for core_id in trial
                    )
                    diagnostic_solver = cp_model.CpSolver()
                    diagnostic_solver.parameters.num_search_workers = 1
                    diagnostic_solver.parameters.random_seed = 0
                    diagnostic_solver.parameters.max_deterministic_time = (
                        probe_budget
                    )
                    _configure_2d_packing_proof(diagnostic_solver)
                    with cp_solve_phase(diagnostic_solver, model, phase="placement.x_legalizer.solve.diagnostic"):
                        diagnostic_status = diagnostic_solver.Solve(model)
                    if diagnostic_status == cp_model.INFEASIBLE:
                        retained.difference_update(family_ids)
                    elif diagnostic_status == cp_model.UNKNOWN:
                        unknown.append(family)
                    elif diagnostic_status not in (
                        cp_model.OPTIMAL,
                        cp_model.FEASIBLE,
                    ):
                        raise RuntimeError(
                            "fixed-outline X legalizer family-core probe "
                            f"returned {diagnostic_solver.StatusName(diagnostic_status)}"
                        )
                core_ids = tuple(sorted(retained))
                inconclusive_families = tuple(sorted(unknown))
            family_core = tuple(sorted({
                core_id.split(":", 1)[0] for core_id in core_ids
            }))
            core_sha256 = hashlib.sha256(repr(core_ids).encode()).hexdigest()
            core_row_evidence: object = core_ids
            if not core_ids:
                core_row_evidence = (
                    "hard_background_only"
                    if status == cp_model.INFEASIBLE
                    else "unavailable_for_solver_unknown"
                )
            hard_background = (
                "derived_envelopes",
                "fixed_domain",
                "fixed_y",
                "manufacturing_grid",
                "physical_origin",
                *(('bottom_origin',) if bottom_origin_lower_bounds_nm else ()),
                *(('exact_outline',) if exact_outline else ()),
                *(('no_overlap2d',) if not refined_collision else ()),
                *(('lexicographic_prefix',) if certified_positions is not None else ()),
            )
            failure_message = (
                "fixed-outline X legalizer has no certified incumbent "
                f"at {stage_name} (solver_status={solver.StatusName(status)}, "
                f"repair_domain_nm={(width, height)!r}, "
                f"exact_outline={exact_outline!r}, "
                f"movable_y={movable_y_ids!r}, "
                "layered_collision_model_counts="
                f"{(collision_raw_atom_pairs, collision_unique_forbidden_boxes, collision_index)!r}, "
                f"hard_background={hard_background!r}, "
                "sufficient_assumption_family_core_relative_to_background="
                f"{family_core!r}, core_row_count={len(core_ids)!r}, "
                f"core_rows={core_row_evidence!r}, "
                f"core_sha256={core_sha256!r}, "
                f"inconclusive_families={inconclusive_families!r})"
            )
            if not core_ids:
                raise RuntimeError(failure_message)
            diagnostic = SufficientCoreDiagnostic(
                solver_status=solver.StatusName(status),
                stage=stage_name,
                family_core=family_core,
                row_ids=core_ids,
                rows=tuple(assumption_trace_by_id[core_id] for core_id in core_ids),
                core_sha256=core_sha256,
            )
            raise PlacementInfeasibilityError(failure_message, diagnostic)
        certified_positions = {
            name: (int(solver.Value(xs[name])), int(solver.Value(ys[name])))
            for name in ids
        }
        if stage_index == 0 and production_evidence_sink is not None:
            production_evidence_sink(production_evidence_context.build_receipt(
                model=model, solver=solver, solver_status=status,
                candidate_positions=certified_positions,
            ))
        if stage_index + 1 < len(objective_stages):
            model.Add(objective == int(solver.Value(objective)))
    if certified_positions is None:
        raise RuntimeError("fixed-outline X legalizer produced no objective stage")
    return certified_positions

def _fixed_outline_banded_seed(
    ids: Sequence[str],
    dims: Dict[str, Tuple[int, int]],
    gap: int,
    width_nm: int,
    bands: Dict[str, int],
    *,
    max_time_s: float,
    adjacency_limits_nm: Sequence[Tuple[str, str, int]],
    adjacency_scope_offsets_nm: Mapping[str, _Bbox],
    order_pairs: Sequence[Tuple[str, str]],
    floorplan_column_classes: Sequence[Tuple[str, int, int, int]],
    floorplan_column_groups: Sequence[_FloorplanColumnGroup],
    seed_cohesion_groups: Tuple[Tuple[str, ...], ...] = (),
    hierarchical_cohesion_groups: Tuple[Tuple[str, ...], ...] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]],
    pack_mode: str,
    aspect_target: float,
    aspect_bounds: Optional[Tuple[float, float]],
    aspect_obligation: bool,
    manufacturing_grid_nm: int,
    right_edge: Sequence[str],
    left_edge: Sequence[str],
    bottom_center_keepout_nm: int,
    bottom_edge_keepout_nm: int,
    top_edge_keepout_nm: int = 0,
    top_edge_keepout_clear: Sequence[str] = (),
    flexible_envelope_constraints: Tuple[FlexibleEnvelopeConstraint, ...] = (),
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = (),
    edge_rail_reservation: Any = None,
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
) -> Tuple[Dict[str, Tuple[int, int]], float]:
    """Build a coordinate-free fixed-outline witness by independent bands.

    The decomposition is used only to seed the complete master.  It is legal
    exactly when every non-overlap refinement is intra-band; the final solve
    reconstructs and proves all original constraints before any certificate is
    returned.  Half of the one caller budget is divided across the bands and the
    other half remains for that complete proof.
    """

    present = tuple(sorted(set(bands[name] for name in ids)))
    canonical_slots = _canonical_mosaic_guest_slot_projections(
        mosaic_guest_slot_projections, dims=dims, minimum_clearance_nm=gap,
    )
    if any(bands[row.host_master] != bands[row.guest_master] for row in canonical_slots):
        raise BandedSeedNotApplicable("fixed-outline band seed has a cross-band mosaic guest-slot")
    if len(present) < 2:
        raise BandedSeedNotApplicable("fixed-outline band seed requires at least two bands")
    if any(
        bands[left] != bands[right] and int(clearance) > int(gap)
        for (left, right), clearance in pair_clearances_nm.items()
    ):
        raise BandedSeedNotApplicable("fixed-outline band seed has a cross-band pair clearance")
    if any(
        len({bands[name] for name, _offset in group}) != 1
        for group in axis_alignment_groups
    ):
        raise BandedSeedNotApplicable("fixed-outline band seed has a cross-band axis group")
    if any(bands[left] != bands[right] for left, right, _limit in adjacency_limits_nm):
        raise BandedSeedNotApplicable("fixed-outline band seed has a cross-band adjacency")
    if any(bands[left] != bands[right] for left, right in order_pairs):
        raise BandedSeedNotApplicable("fixed-outline band seed has a cross-band order")
    canonical_flexible = _canonical_flexible_envelope_constraints(
        flexible_envelope_constraints,
        block_ids=tuple(ids),
        bands=bands,
    )
    canonical_compounds = _canonical_rigid_compound_footprints(
        rigid_compound_footprints,
        dims=dims,
    )
    canonical_layered = _canonical_layered_collision_footprints(
        layered_collision_footprints,
        dims=dims,
    )
    canonical_overlap_pairs = _canonical_overlap_compatible_pairs(
        overlap_compatible_pairs,
        block_ids=tuple(ids),
        pair_clearances_nm=pair_clearances_nm,
    )

    hard_edge_members = tuple(sorted(set(right_edge) | set(left_edge)))
    edge_bands = {bands[name] for name in hard_edge_members}
    if len(edge_bands) > 1:
        raise BandedSeedNotApplicable("fixed-outline band seed has hard edges in multiple bands")
    exact_band = next(iter(edge_bands), present[0])
    seed_budget_total = float(max_time_s) / 2.0
    x_legalizer_budget = seed_budget_total / 3.0
    per_band_budget = (
        seed_budget_total - x_legalizer_budget
    ) / len(present)
    final_budget = max(0.0, float(max_time_s) - seed_budget_total)

    if type(seed_cohesion_groups) is not tuple:
        raise TypeError("seed cohesion groups must be one exact tuple")
    cohesion_groups = _floorplan_cell_cohesion_groups(
        tuple(floorplan_column_groups)
    )
    if any(
        len({bands[owner] for owner in group}) != 1
        for group in (
            cohesion_groups
            + seed_cohesion_groups
            + hierarchical_cohesion_groups
        )
    ):
        raise BandedSeedNotApplicable("seed cohesion group crosses exact bands")
    local_rows: list[Tuple[int, Dict[str, Tuple[int, int]], int]] = []
    for band in present:
        local_ids = tuple(sorted(name for name in ids if bands[name] == band))
        members = set(local_ids)
        local_flexible = tuple(
            row
            for row in canonical_flexible
            if all(member in members for member in row.members)
        )
        local_positions, _local_width, local_height, _local_evidence = (
            _strip_solve_banded(
                local_ids,
                {name: dims[name] for name in local_ids},
                gap,
                width_nm,
                {name: band for name in local_ids},
                right_pull=(),
                left_pull=(),
                pull_pairs=(),
                cohesion_groups=tuple(
                    group
                    for group in cohesion_groups
                    if all(owner in members for owner in group)
                ),
                well_cohesion_groups=tuple(
                    group
                    for group in seed_cohesion_groups
                    if all(owner in members for owner in group)
                ),
                hierarchical_cohesion_groups=tuple(
                    group
                    for group in hierarchical_cohesion_groups
                    if all(owner in members for owner in group)
                ),
                flexible_envelope_constraints=local_flexible,
                rigid_compound_footprints=tuple(
                    row for row in canonical_compounds if row[0] in members
                ),
                layered_collision_footprints=tuple(
                    row for row in canonical_layered if row[0] in members
                ),
                mosaic_guest_slot_projections=tuple(
                    row for row in canonical_slots
                    if row.host_master in members and row.guest_master in members
                ),
                max_time_s=per_band_budget,
                adjacency_limits_nm=tuple(
                    row
                    for row in adjacency_limits_nm
                    if row[0] in members and row[1] in members
                ),
                adjacency_scope_offsets_nm={
                    name: bbox
                    for name, bbox in adjacency_scope_offsets_nm.items()
                    if name in members
                },
                order_pairs=tuple(
                    pair
                    for pair in order_pairs
                    if pair[0] in members and pair[1] in members
                ),
                floorplan_column_classes=tuple(
                    row for row in floorplan_column_classes if row[0] in members
                ),
                floorplan_column_groups=tuple(
                    row
                    for row in floorplan_column_groups
                    if all(owner in members for owner, _count, _offset in row[4])
                ),
                bottom_center_keepout_nm=0,
                bottom_edge_keepout_nm=0,
                bottom_edge_keepout_exempt=(),
                top_edge_keepout_nm=(
                    int(top_edge_keepout_nm) if band == present[-1] else 0
                ),
                top_edge_keepout_clear=tuple(
                    name
                    for name in top_edge_keepout_clear
                    if name in members
                ),
                pair_clearances_nm={
                    pair: clearance
                    for pair, clearance in pair_clearances_nm.items()
                    if pair[0] in members and pair[1] in members
                },
                overlap_compatible_pairs=tuple(
                    pair
                    for pair in canonical_overlap_pairs
                    if pair[0] in members and pair[1] in members
                ),
                axis_alignment_groups=tuple(
                    group
                    for group in axis_alignment_groups
                    if all(name in members for name, _offset in group)
                ),
                pack_mode=pack_mode,
                aspect_target=aspect_target,
                aspect_obligation=False,
                manufacturing_grid_nm=manufacturing_grid_nm,
                right_edge=tuple(name for name in right_edge if name in members),
                left_edge=tuple(name for name in left_edge if name in members),
                placement_objective="height",
                strict_width_cap=True,
                exact_outline_width=band == exact_band,
            )
        )
        local_member_boxes, local_group_boxes = _flexible_seed_physical_views(
            local_positions,
            dims={name: dims[name] for name in local_ids},
            constraints=local_flexible,
        )
        del local_member_boxes
        local_outline = tuple(
            (
                local_positions[name][0],
                local_positions[name][1],
                local_positions[name][0] + dims[name][0],
                local_positions[name][1] + dims[name][1],
            )
            for name in local_ids
        ) + tuple(local_group_boxes.values())
        local_y_min = min(bbox[1] for bbox in local_outline)
        local_y_max = max(bbox[3] for bbox in local_outline)
        if local_height != local_y_max - local_y_min:
            raise RuntimeError(
                "fixed-outline local seed height differs from physical envelope"
            )
        local_rows.append(
            (
                band,
                {
                    name: (x, y - local_y_min)
                    for name, (x, y) in local_positions.items()
                },
                local_y_max - local_y_min,
            )
        )

    seed: Dict[str, Tuple[int, int]] = {}
    next_floor = max(int(bottom_edge_keepout_nm), 0)
    for band, local_positions, local_height in local_rows:
        if band == _BAND_CENTER:
            next_floor = max(next_floor, int(bottom_center_keepout_nm))
        for name, (x, y) in local_positions.items():
            seed[name] = (x, y + next_floor)
        next_floor += int(local_height) + int(gap)

    # Independent bands have no cross-band X relation unless the canonical
    # floorplan grammar declares one.  Every other X-sensitive relation was
    # required above to stay inside one band and was already solved there.  In
    # the column-free case, re-solving frozen Y adds no proof and can make a
    # valid two-dimensional well-cohesion seed spuriously infeasible.
    if floorplan_column_classes or floorplan_column_groups:
        seed = _legalize_fixed_outline_seed_x(
            seed,
            dims=dims,
            bands=bands,
            width_nm=width_nm,
            gap_nm=gap,
            max_time_s=x_legalizer_budget,
            manufacturing_grid_nm=manufacturing_grid_nm,
            floorplan_column_classes=floorplan_column_classes,
            floorplan_column_groups=floorplan_column_groups,
            hierarchical_cohesion_groups=hierarchical_cohesion_groups,
            rigid_compound_footprints=canonical_compounds,
            layered_collision_footprints=canonical_layered,
            overlap_compatible_pairs=canonical_overlap_pairs,
            mosaic_guest_slot_projections=canonical_slots,
            edge_rail_reservation=edge_rail_reservation,
            pair_clearances_nm=pair_clearances_nm,
            axis_alignment_groups=axis_alignment_groups,
            order_pairs=order_pairs,
            adjacency_limits_nm=adjacency_limits_nm,
            adjacency_scope_offsets_nm=adjacency_scope_offsets_nm,
            right_edge=right_edge,
            left_edge=left_edge,
        )

    if aspect_obligation:
        bbox_y_min = (
            0
            if int(bottom_edge_keepout_nm) > 0
            else min(y for _x, y in seed.values())
        )
        bbox_y_max = max(seed[name][1] + dims[name][1] for name in ids)
        height_nm = bbox_y_max - bbox_y_min
        aspect_floor_nm = _strict_outline_aspect_height_floor_nm(
            width_nm,
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
        )
        padding_nm = max(0, aspect_floor_nm - height_nm)
        if padding_nm:
            first_band = present[0]
            seed = {
                name: (x, y + (padding_nm if bands[name] != first_band else 0))
                for name, (x, y) in seed.items()
            }
    if any(not mosaic_guest_slot_satisfied(seed, row) for row in canonical_slots):
        raise RuntimeError("fixed-outline band seed mosaic guest-slot replay failed")
    return seed, final_budget

def _strip_solve_floorplan(
    ids: Sequence[str],
    dims: Dict[str, Tuple[int, int]],
    gap: int,
    bands: Dict[str, int],
    left_column: str,
    right_pull: Sequence[str],
    left_pull: Sequence[str],
    max_time_s: float,
    adjacency_limits_nm: Sequence[Tuple[str, str, int]] = (),
    adjacency_scope_offsets_nm: Mapping[str, _Bbox] = MappingProxyType({}),
    pull_pairs: Sequence[Tuple[str, str]] = (),
    order_pairs: Sequence[Tuple[str, str]] = (),
    lateral_blocks: Sequence[str] = (),
    floorplan_column_classes: Sequence[Tuple[str, int, int, int]] = (),
    floorplan_column_groups: Sequence[_FloorplanColumnGroup] = (),
    channel_nm: int = 0,
    bottom_center_keepout_nm: int = 0,
    bottom_edge_keepout_nm: int = 0,
    bottom_edge_keepout_exempt: Sequence[str] = (),
    bottom_edge_keepout_clear: Sequence[str] = (),
    pack_mode: str = "rows",
    strict_band_order: bool = False,
    strict_band_members: Sequence[str] = (),
    solve_cuts: Sequence[_PreparedSolveCut] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]] = (),
    center_alignment_pairs: Sequence[Tuple[str, str, str]] = (),
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder] = (),
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder] = (),
    hierarchical_macro_alignments: Sequence[_HierarchicalMacroAlignment] = (),
    hierarchical_alignment_tolerance_scales_nm: (
        _HierarchicalAlignmentToleranceScalesNm
    ) = (),
    aspect_target: float = 1.67,
    source_seed_positions: Optional[Mapping[str, Tuple[int, int]]] = None,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = True,
    manufacturing_grid_nm: int = 1,
    right_edge: Sequence[str] = (),
    left_edge: Sequence[str] = (),
    placement_objective: str = "height",
    edge_rail_reservation: Any = None,
) -> Tuple[Dict[str, Tuple[int, int]], _SolverEvidence]:
    """One deterministic CP-SAT solve of the REFERENCE-SKELETON floorplan.

    Structure (all HARD constraints on top of ``AddNoOverlap2D``):
      * the ``left_column`` block is pinned to the left edge in x. Its y remains
        subject to rail, alignment and placement constraints, not a corner pin;
      * every LOWER block (bands below TOP -- nmos bottom + passives/cap center)
        sits strictly to the RIGHT of the column (``x0 >= wD + gap``) and obeys
        the band ordering (ascending ordinal stacks upward: nmos below passives)
        via per-band floor ``IntVar``s;
      * every TOP block (the pmos mirrors, minus the column) is one horizontal
        strip on a shared floor ABOVE the whole lower region AND above the column
        -- so the top band spans the full width and the top-left is never empty.
      * ``order_pairs`` -- ``(a, b)`` HARD same-band left/right order (from
        ``@order(a, b, left_of)``): ``a``'s right edge <= ``b``'s left edge. The
        caller passes ONLY same-band pairs (a top-band pair packs the two mirrors
        left-to-right in the requested order).

    The outline width upper bound starts from the wider of the top-band row and
    the ``column + widest lower-band row`` and expands only when the declared
    directional ``W/H`` interval requires it.  CP either preserves the legacy
    fixed-width height objective or, when requested by canonical intent, minimizes
    the exact dynamic outline product ``W * H``. A
    SOFT term (strictly below the height minimum) pushes the ``right_pull`` and
    ``left_pull`` blocks to their declared horizontal edges, LEFT-PACKS the top band,
    and, for each ``pull_pairs`` ``(a, b)``, pulls ``a``'s x-centre toward ``b``'s
    (a small passive block hugging its declared ``@adjacent`` neighbour). Returns
    ``{block_id -> (x0, y0)}`` in nm."""
    from ortools.sat.python import cp_model

    hierarchical_alignment_tolerance_scales_nm = (
        _require_hierarchical_alignment_tolerance_scales_nm(
            hierarchical_alignment_tolerance_scales_nm,
            alignments=hierarchical_macro_alignments,
        )
    )
    if type(placement_objective) is not str:
        raise TypeError("placement_objective must be an exact string")
    if placement_objective not in {"area", "height"}:
        raise ValueError(
            "placement_objective must be one of ['area', 'height'], "
            f"got {placement_objective!r}"
        )
    _require_edge_rail_reservation(edge_rail_reservation, dims=dims)
    D = left_column
    exempt = set(bottom_edge_keepout_exempt or ())
    clear = set(bottom_edge_keepout_clear or ())
    lateral = set(str(name) for name in lateral_blocks)
    if D in lateral or not lateral <= set(ids):
        raise ValueError("lateral floorplan blocks are unknown or include the column")
    wD, hD = dims[D]
    # The column -> lower-region separation. A `@channel(orient=vertical, width=W)`
    # reserves a device-free vertical corridor of width `channel_nm` just RIGHT of
    # the left column (for the cross-band M3 straps); with no channel declared this
    # is the plain `gap` (byte-identical). Every lower block is pushed to
    # ``x0 >= wD + sep`` so the strip ``[wD, wD+sep] x [0, top_floor]`` stays free.
    seed_gap = max((gap, *pair_clearances_nm.values()))
    sep = max(gap, int(channel_nm))

    def band_of(b: str) -> int:
        return bands.get(b, _BAND_CENTER)

    top = sorted(
        b for b in ids if b != D and b not in lateral and band_of(b) == _BAND_TOP
    )
    lower = sorted(
        b for b in ids if b != D and b not in lateral and band_of(b) != _BAND_TOP
    )
    lower_bands = sorted({band_of(b) for b in lower})

    # ---- tight width cap: max(top-band row, column + widest lower-band row)
    lower_natural = 0
    for bd in lower_bands:
        row = [b for b in lower if band_of(b) == bd]
        wsum = sum(dims[b][0] for b in row) + gap * max(0, len(row) - 1)
        lower_natural = max(lower_natural, wsum)
    top_natural = sum(dims[b][0] for b in top) + gap * max(0, len(top) - 1)
    lower_extent = (wD + sep + lower_natural) if lower else wD
    w_cap = max(top_natural, lower_extent, max(dims[b][0] for b in ids))
    w_cap = max(
        w_cap + _cut_extent_padding(solve_cuts, "horizontal_separation", gap),
        _required_cut_axis_extent(solve_cuts, dims, "horizontal_separation"),
        _axis_alignment_required_width_nm(axis_alignment_groups, dims),
    )

    # ---- height bound (loose but finite)
    lower_band_h = {
        bd: max((dims[b][1] for b in lower if band_of(b) == bd), default=0)
        for bd in lower_bands
    }
    lower_h = sum(lower_band_h.values()) + seed_gap * max(1, len(lower_bands))
    top_h = max((dims[b][1] for b in top), default=0)
    max_h = max(dims[b][1] for b in ids)
    h_bound = max(
        max(lower_h, hD)
        + int(bottom_center_keepout_nm)
        + int(bottom_edge_keepout_nm)
        + seed_gap
        + top_h
        + seed_gap * 10
        + max_h
        + _cut_extent_padding(solve_cuts, "vertical_separation", gap),
        _required_cut_axis_extent(solve_cuts, dims, "vertical_separation"),
    )
    if axis_alignment_groups or center_alignment_pairs:
        h_bound = max(
            h_bound,
            sum(dims[bid][1] for bid in ids)
            + seed_gap * (len(ids) + len(lower_bands) + 2)
            + int(bottom_center_keepout_nm)
            + int(bottom_edge_keepout_nm),
        )
    natural_seed = _floorplan_seed(
        D,
        top,
        lower,
        bands,
        lower_bands,
        dims,
        gap,
        sep=sep,
        order_pairs=order_pairs,
        bottom_center_keepout_nm=bottom_center_keepout_nm,
        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
        bottom_edge_keepout_exempt=tuple(sorted(exempt)),
        bottom_edge_keepout_clear=tuple(sorted(clear)),
        pair_clearances_nm=pair_clearances_nm,
    )
    lateral_cursor = max(
        natural_seed[name][0] + dims[name][0] for name in natural_seed
    ) + gap
    for name in _stable_hard_order(tuple(sorted(lateral)), order_pairs):
        natural_seed[name] = (
            lateral_cursor,
            (
                0
                if name in exempt or name in clear
                else int(bottom_edge_keepout_nm)
            ),
        )
        lateral_cursor += dims[name][0] + gap
    if aspect_obligation and top:
        aspect_lower, aspect_upper, aspect_scale = _scaled_aspect_bounds(
            aspect_target,
            aspect_bounds,
        )
        seed_x_min = min(natural_seed[b][0] for b in ids)
        seed_x_max = max(natural_seed[b][0] + dims[b][0] for b in ids)
        seed_y_min = (
            0
            if int(bottom_edge_keepout_nm) > 0
            else min(natural_seed[b][1] for b in ids)
        )
        seed_y_max = max(natural_seed[b][1] + dims[b][1] for b in ids)
        seed_width = seed_x_max - seed_x_min
        seed_height = seed_y_max - seed_y_min
        required_height = _ceil_ratio(seed_width * aspect_scale, aspect_upper)
        maximum_height = (seed_width * aspect_scale) // aspect_lower
        if seed_height < required_height:
            grid = int(manufacturing_grid_nm)
            if grid <= 0:
                raise ValueError("manufacturing grid must be positive")
            padding = _ceil_ratio(required_height - seed_height, grid) * grid
            if seed_height + padding <= maximum_height:
                natural_seed = {
                    name: (
                        point[0],
                        point[1] + (padding if name in top else 0),
                    )
                    for name, point in natural_seed.items()
                }
    natural_seed = _center_aligned_lateral_seed(
        natural_seed,
        dims=dims,
        center_alignment_pairs=center_alignment_pairs,
        lateral_blocks=tuple(sorted(lateral)),
        order_pairs=order_pairs,
        gap_nm=gap,
        pair_clearances_nm=pair_clearances_nm,
    )
    seed = dict(natural_seed)
    source_seed = _canonical_source_seed_positions(
        source_seed_positions,
        ids=ids,
    )
    if source_seed is not None:
        seed = _select_feedback_seed(
            seed,
            source_seed,
            dims=dims,
            cuts=solve_cuts,
            gap_nm=gap,
            pair_clearances_nm=pair_clearances_nm,
        )
    # A HARD order starting at the fixed left column may shift a top-band suffix
    # rightward.  The CP outline must contain that same legal constructive seed.
    seed_width = max(seed[b][0] + dims[b][0] for b in ids)
    seed_height = max(seed[b][1] + dims[b][1] for b in ids)
    natural_seed_width = max(
        natural_seed[b][0] + dims[b][0] for b in ids
    )
    natural_seed_height = max(
        natural_seed[b][1] + dims[b][1] for b in ids
    )
    aspect_width_basis = max(
        seed_width,
        natural_seed_width,
        _required_cut_axis_extent(solve_cuts, dims, "horizontal_separation"),
    )
    aspect_height_basis = max(
        seed_height,
        natural_seed_height,
        _required_cut_axis_extent(solve_cuts, dims, "vertical_separation"),
    )
    w_cap = max(w_cap, seed_width, natural_seed_width)
    h_bound = max(h_bound, seed_height + seed_gap, natural_seed_height + seed_gap)
    if aspect_obligation:
        aspect_lower, aspect_upper, aspect_scale = _scaled_aspect_bounds(
            aspect_target, aspect_bounds
        )
        w_cap = max(
            w_cap,
            _ceil_ratio(aspect_lower * aspect_height_basis, aspect_scale),
        )
        h_bound = max(
            h_bound,
            _ceil_ratio(aspect_scale * aspect_width_basis, aspect_upper) + seed_gap,
        )
    w_cap = _right_edge_compatible_width_bound(
        w_cap, dims, right_edge, manufacturing_grid_nm
    )

    model = cp_model.CpModel()
    xs: Dict[str, Any] = {}
    ys: Dict[str, Any] = {}
    x_iv: List[Any] = []
    y_iv: List[Any] = []
    W = model.NewIntVar(max(dims[b][0] for b in ids), w_cap, "W")
    H = model.NewIntVar(max(dims[b][1] for b in ids), h_bound, "H")
    for b in ids:  # ids sorted by caller -> input-order-invariant model
        w, h = dims[b]
        x0 = model.NewIntVar(0, max(0, w_cap - w), f"x_{b}")
        y0 = model.NewIntVar(0, h_bound - h, f"y_{b}")
        xs[b], ys[b] = x0, y0
        x_iv.append(model.NewFixedSizeIntervalVar(x0, w + gap, f"xi_{b}"))
        y_iv.append(model.NewFixedSizeIntervalVar(y0, h + gap, f"yi_{b}"))
        model.AddHint(x0, seed[b][0])
        model.AddHint(y0, seed[b][1])
    model.AddMaxEquality(W, [xs[b] + dims[b][0] for b in ids])
    model.AddMaxEquality(H, [ys[b] + dims[b][1] for b in ids])
    if edge_rail_reservation is not None:
        edge_rail_reservation.add_constraints(model, xs=xs, ys=ys, dims=dims)
    outline_width = W if placement_objective == "area" else w_cap
    model.AddNoOverlap2D(x_iv, y_iv)
    _add_bottom_edge_keepout_constraints(
        model, ys, bottom_edge_keepout_nm, exempt, clear
    )
    _add_manufacturing_grid_constraints(
        model, xs, ys, manufacturing_grid_nm
    )
    _add_hard_horizontal_edge_constraints(
        model,
        xs,
        dims,
        width_bound=outline_width,
        right_edge=right_edge,
        left_edge=left_edge,
    )
    if aspect_obligation:
        _add_directional_aspect_constraints(
            model,
            xs=xs,
            ys=ys,
            dims=dims,
            width_bound=w_cap,
            height_bound=h_bound,
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
        )
    _add_pair_clearance_constraints(model, xs, ys, dims, pair_clearances_nm)
    _add_solve_cut_constraints(model, xs, ys, dims, solve_cuts)
    _add_axis_alignment_constraints(model, xs, axis_alignment_groups)
    _add_center_alignment_constraints(
        model, xs, ys, dims, center_alignment_pairs
    )
    _add_aggregate_envelope_order_constraints(
        model,
        xs,
        ys,
        dims,
        aggregate_envelope_orders,
        width_bound_nm=w_cap,
        height_bound_nm=h_bound,
        center_step_nm=manufacturing_grid_nm,
    )
    _add_hierarchical_macro_constraints(
        model,
        xs,
        ys,
        dims,
        hierarchical_macro_orders,
        hierarchical_macro_alignments,
        width_bound_nm=w_cap,
        height_bound_nm=h_bound,
        center_step_nm=manufacturing_grid_nm,
        gap_nm=gap,
        tolerance_scales_nm=hierarchical_alignment_tolerance_scales_nm,
    )
    _add_floorplan_column_constraints(
        model, xs, dims, floorplan_column_classes
    )
    _add_floorplan_column_group_constraints(model, xs, floorplan_column_groups)
    _add_adjacency_constraints(
        model,
        xs,
        ys,
        dims,
        adjacency_limits_nm,
        scope_offsets_nm=adjacency_scope_offsets_nm,
        minimum_gap_nm=gap,
        projection_nm=manufacturing_grid_nm,
    )

    # The coordinate-free column role owns x only. Rail carriers/keepouts above
    # still constrain y; the constructive seed's y=0 is not another declaration.
    model.Add(xs[D] == 0)

    # ---- lower region strictly to the RIGHT of the column (+ reserved channel):
    #      every lower block sits at x0 >= wD + sep, so the corridor [wD, wD+sep]
    #      x [0, top_floor] is a device-free vertical channel for the M3 straps.
    for b in lower:
        model.Add(xs[b] >= wD + sep)

    # ---- lower-region packing.  Historical ``rows`` gives every band a
    # full-width vertical tier.  ``dense_2d`` preserves the left column, top
    # strip, rail keepout, exact block gap and no-overlap master, but lets bottom
    # devices and centre passives share y extent when CP separates them in x.
    # This is the area-minimizing seqpair macro pattern used by TED; it does not
    # drop polarity structure like the legacy landscape probe.
    if lower:
        for b in lower:
            if band_of(b) == _BAND_CENTER:
                model.Add(ys[b] >= int(bottom_center_keepout_nm))
        if pack_mode == "rows":
            bf = {bd: model.NewIntVar(0, h_bound, f"lbf_{bd}") for bd in lower_bands}
            model.Add(bf[lower_bands[0]] == 0)
            for b in lower:
                model.Add(ys[b] >= bf[band_of(b)])
            for i in range(1, len(lower_bands)):
                hi, lo = lower_bands[i], lower_bands[i - 1]
                for b in lower:
                    if band_of(b) == lo:
                        model.Add(bf[hi] >= ys[b] + dims[b][1] + gap)

    if strict_band_order:
        strict_ids = sorted(set(strict_band_members or ids) - lateral)
        strict_bands = sorted({band_of(block_id) for block_id in strict_ids})
        strict_floors = {
            band: model.NewIntVar(0, h_bound, f"strict_band_floor_{band}")
            for band in strict_bands
        }
        for block_id in strict_ids:
            model.Add(ys[block_id] >= strict_floors[band_of(block_id)])
        for index in range(1, len(strict_bands)):
            high, low = strict_bands[index], strict_bands[index - 1]
            for block_id in strict_ids:
                if band_of(block_id) == low:
                    model.Add(
                        strict_floors[high]
                        >= ys[block_id] + dims[block_id][1] + gap
                    )

    # ---- top pmos band: one strip above the whole lower region + the column
    if top:
        tf = model.NewIntVar(0, h_bound, "top_floor")
        model.Add(tf >= ys[D] + hD + gap)
        for b in lower:
            model.Add(tf >= ys[b] + dims[b][1] + gap)
        for b in top:
            if pack_mode == "rows":
                model.Add(ys[b] == tf)
            else:
                # A hard differential axis can legitimately align two rigid
                # pair islands in x.  Dense packing must then stack those
                # islands in y instead of making the historical one-row top
                # strip contradictory.  Both remain above the lower region.
                model.Add(ys[b] >= tf)

    # ---- HARD same-band left/right order (from @order): a left_of b -> a's right
    #      edge <= b's left edge. The caller passes ONLY same-band pairs.
    for a, b in order_pairs:
        a, b = str(a), str(b)
        if a in xs and b in xs and a != b:
            model.Add(xs[b] >= xs[a] + dims[a][0] + gap)

    # ---- SOFT position objective, built once and optimized LEXICOGRAPHICALLY
    #      AFTER the height. The old single big-M form (H*(soft_max+1) + soft) is
    #      exactly lexicographic in the objective VALUE, but the height term is ~1e5x
    #      the soft term, so inside a finite budget the solver proves H-optimality and
    #      barely moves the low-order soft bits -- the @adjacent pulls came out no
    #      better than the seed. A TWO-PHASE solve fixes this: phase 1 minimizes the
    #      envelope height H; phase 2 pins H to that minimum and minimizes the soft
    #      position objective ON ITS OWN SCALE (right-align + top left-pack + pulls),
    #      so the envelope is never traded for position yet the positions are actually
    #      optimized. Deterministic (two seed=0 single-worker solves).
    w_right = 100
    w_left = 100
    w_pull = 100
    rp = sorted(set(str(b) for b in (right_pull or ())) & set(ids))
    lp = sorted(set(str(b) for b in (left_pull or ())) & set(ids))
    pp = sorted(
        {
            (str(a), str(b))
            for a, b in (pull_pairs or ())
            if str(a) in xs and str(b) in xs and str(a) != str(b)
        }
    )
    soft: List[Any] = []
    for b in rp:  # push the cap bank to the right edge
        soft.append(w_right * (outline_width - dims[b][0] - xs[b]))
    for b in lp:  # push to the leftmost legal region slot
        soft.append(w_left * xs[b])
    for b in top:  # left-pack the top band -> fill the top-left
        soft.append(xs[b])
    for a, b in pp:  # pull a's x-centre toward b's (soft proximity)
        d = model.NewIntVar(0, w_cap, f"pull_{a}_{b}")
        xca = xs[a] + dims[a][0] // 2
        xcb = xs[b] + dims[b][0] // 2
        model.Add(d >= xca - xcb)
        model.Add(d >= xcb - xca)
        soft.append(w_pull * d)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    primary_name = "area_nm2" if placement_objective == "area" else "height_nm"
    primary_phase = "area" if placement_objective == "area" else "height"
    if placement_objective == "area":
        primary = model.NewIntVar(0, w_cap * h_bound, "area_nm2")
        model.AddMultiplicationEquality(primary, (W, H))
    else:
        primary = H
    validation_error = model.Validate()
    if validation_error:
        raise RuntimeError(f"invalid floorplan CP model: {validation_error}")
    phase_labels = (primary_phase, "soft_position") if soft else (primary_phase,)
    phase_budget = _DeterministicBudgetLedger.split(max_time_s, phase_labels)

    def _pos(sol) -> Dict[str, Tuple[int, int]]:
        return {b: (int(sol.Value(xs[b])), int(sol.Value(ys[b]))) for b in ids}
    # ---- phase 1: minimize the declared primary envelope objective
    model.Minimize(primary)
    solver.parameters.max_deterministic_time = phase_budget.for_phase(primary_phase)
    from clay.layout.place.native_reference_quality import apply_current_floorplan_quality
    quality_stage = apply_current_floorplan_quality(model, xs, ys, dims, W, H)
    with cp_solve_phase(solver, model, phase="placement.floorplan.solve.primary"):
        if quality_stage is not None:
            quality_stage.before_solve(model, solver, phase="primary")
        st = solver.Solve(model)
        if quality_stage is not None:
            quality_stage.after_solve(model, solver, st, phase="primary")
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE) and (
        axis_alignment_groups
        or center_alignment_pairs
        or aggregate_envelope_orders
        or adjacency_limits_nm
        or right_edge
        or left_edge
        or floorplan_column_classes
        or floorplan_column_groups
        or edge_rail_reservation is not None
    ):
        requirements = tuple(
            name
            for name, present_requirement in (
                ("differential_axis", bool(axis_alignment_groups)),
                ("center_alignment", bool(center_alignment_pairs)),
                ("aggregate_envelope_order", bool(aggregate_envelope_orders)),
                ("adjacency", bool(adjacency_limits_nm)),
                ("horizontal_edge", bool(right_edge or left_edge)),
                ("floorplan_columns", bool(floorplan_column_classes)),
                ("floorplan_group_columns", bool(floorplan_column_groups)),
                ("edge_rail", edge_rail_reservation is not None),
            )
            if present_requirement
        )
        failure_label = {
            ("differential_axis",): "hard differential-axis alignment",
            ("adjacency",): "hard adjacency",
            ("horizontal_edge",): "hard horizontal edge anchor",
        }.get(requirements, "hard placement requirements")
        raise RuntimeError(
            f"{failure_label} has no certified incumbent "
            f"(requirements={requirements!r}, "
            + f" (solver_status={solver.StatusName(st)}, "
            + f"limits={tuple(adjacency_limits_nm)!r}, "
            + f"left_column={left_column!r})"
        )
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(seed), _solver_evidence(
            solver, st, objective_name=primary_name
        )
    primary_value = int(solver.Value(primary))
    best = _pos(solver)
    phase1 = _solver_evidence(solver, st, objective_name=primary_name)
    if not soft or (placement_objective == "area" and st != cp_model.OPTIMAL):
        return best, phase1
    # ---- phase 2: pin the primary optimum, minimize the soft position objective
    model.Add(primary <= primary_value)
    model.Proto().ClearField("solution_hint")
    for b in ids:  # warm-start from the tight-H incumbent
        model.AddHint(xs[b], best[b][0])
        model.AddHint(ys[b], best[b][1])
    model.Proto().ClearField("objective")
    model.Minimize(sum(soft))
    solver.parameters.max_deterministic_time = phase_budget.for_phase("soft_position")
    with cp_solve_phase(solver, model, phase="placement.floorplan.solve.soft_position"):
        if quality_stage is not None:
            quality_stage.before_solve(model, solver, phase="soft_position")
        st2 = solver.Solve(model)
        if quality_stage is not None:
            quality_stage.after_solve(model, solver, st2, phase="soft_position")
    if st2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return best, _SolverEvidence(
            phase1.status,
            phase1.objective_name,
            phase1.objective,
            phase1.best_bound,
            True,
            "phase1_incumbent",
            phase1.status,
            phase1.objective_name,
            phase1.objective,
            phase1.best_bound,
        )
    phase2 = _solver_evidence(solver, st2, objective_name="soft_position_cost")
    return _pos(solver), _SolverEvidence(
        phase2.status,
        phase2.objective_name,
        phase2.objective,
        phase2.best_bound,
        False,
        None,
        phase1.status,
        phase1.objective_name,
        phase1.objective,
        phase1.best_bound,
    )

def _cut_residual_incumbent_result(
    items: Sequence[Tuple[str, _Bbox]],
    bands: Any,
    source_certificate: object,
    aspect_target: float,
    channel_nm: int,
    gap: int,
    residual_status: str,
    *,
    allow_unmatched_source: bool = False,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = True,
    minimum_structural_occupancy: Optional[float] = None,
) -> PlaceBlocks2DResult:
    """Return source/input incumbent evidence without constructing a CP model."""

    incumbent_items = tuple(items)
    incumbent_bands: Dict[str, int] = (
        {
            name: int((bands or {}).get(name, _BAND_CENTER))
            for name, _bbox in incumbent_items
        }
        if bands
        else {}
    )
    fallback_kind = "input_incumbent"
    try:
        source_items = _exact_cut_input_blocks(tuple(source_certificate.blocks))
        if allow_unmatched_source or source_items == incumbent_items:
            incumbent_items = source_items
            incumbent_bands = dict(source_certificate.bands)
            fallback_kind = "source_incumbent"
    except (AttributeError, TypeError, ValueError):
        pass

    positions = {name: (bbox[0], bbox[1]) for name, bbox in incumbent_items}
    return _placement_result(
        positions,
        incumbent_items,
        incumbent_bands,
        aspect_target,
        channel_nm,
        _SolverEvidence(
            "FALLBACK",
            fallback_kind,
            None,
            None,
            True,
            fallback_kind,
        ),
        gap_nm=gap,
        solve_cuts=(),
        cut_residual_status=residual_status,
        aspect_bounds=aspect_bounds,
        aspect_obligation=aspect_obligation,
        minimum_structural_occupancy=minimum_structural_occupancy,
    )

def _place_blocks_2d_with_alignment_scale_source(
    source_certificate: FinalPlacementCertificate,
    /,
    *args: Any,
    **kwargs: Any,
) -> PlaceBlocks2DResult:
    """Run one exact replay with context-local derived-scale evidence."""

    token = _ALIGNMENT_SCALE_SOURCE_CERTIFICATE.set(source_certificate)
    try:
        return place_blocks_2d_with_certificate(*args, **kwargs)
    finally:
        _ALIGNMENT_SCALE_SOURCE_CERTIFICATE.reset(token)

def place_blocks_2d_with_certificate(
    blocks: Sequence[Tuple[str, _Bbox]],
    gap_nm: int,
    aspect_target: float = 1.67,
    max_time_s: float = 30.0,
    bands: Any = None,
    floorplan_column_classes: Any = (),
    floorplan_column_groups: Any = (),
    right_pull: Any = None,
    left_pull: Any = None,
    right_edge: Any = None,
    left_edge: Any = None,
    pull_pairs: Any = None,
    adjacency_limits_nm: Any = (),
    adjacency_scope_bboxes_nm: Any = (),
    left_column: Any = None,
    lateral_blocks: Any = (),
    order_pairs: Any = None,
    vertical_order_pairs: Any = None,
    channel_nm: int = 0,
    bottom_center_keepout_nm: int = 0,
    bottom_edge_keepout_nm: int = 0,
    bottom_edge_keepout_exempt: Any = (),
    bottom_edge_keepout_clear: Any = (),
    top_edge_keepout_nm: int = 0,
    top_edge_keepout_clear: Any = (),
    pack_mode: str = "rows",
    strict_band_order: Optional[bool] = None,
    strict_band_members: Any = (),
    pair_clearances_nm: Any = None,
    axis_alignment_groups: Any = (),
    center_alignment_pairs: Any = (),
    aggregate_envelope_orders: Any = (),
    hierarchical_macro_orders: Any = (),
    hierarchical_macro_alignments: Any = (),
    hierarchical_cohesion_groups: Any = (),
    solve_cuts: Any = (),
    solve_cut_proofs: Any = (),
    solve_cut_source_certificate: Any = None,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = True,
    manufacturing_grid_nm: int = 1,
    minimum_structural_occupancy: Optional[float] = None,
    placement_objective: str = "height",
    fixed_outline_width_nm: Optional[int] = None,
    seed_pair_clearances_nm: Any = None,
    seed_cohesion_groups: Any = (),
    incumbent_seed_positions: Any = None,
    flexible_envelope_constraints: Any = (),
    rigid_compound_footprints: Any = (),
    layered_collision_footprints: Any = (),
    row_bounded_cavity_projections: Any = (),
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = (),
    edge_rail_reservation: Any = None,
    overlap_compatible_pairs: Any = (),
    area_target: Optional[AreaTarget] = None,
    native_access_pair_clearances_nm: Any = None,
) -> PlaceBlocks2DResult:
    """Re-place rigid blocks in true 2-D; return absolute ``{block_id -> (x0_nm,
    y0_nm)}`` inside a tight outline near ``aspect_target``.

    ``bands`` (opt-in): a ``{block_id -> band ordinal}`` map. With unspecified band
    policy and no hierarchical DAG, blocks stack bottom -> top in ascending ordinal
    (0 nmos / 1 passive / 2 pmos), packing horizontally within each band, and the
    outline width is fixed to the WIDEST band's natural row width so ``W`` is minimal
    given the fixed band-stacked ``H`` (minimize W + H subject to the band order).
    ``right_pull``/``left_pull`` (block ids) and ``pull_pairs`` (``(a, b)`` id pairs)
    add a SOFT, below-envelope objective: push edge-pulled blocks to their declared
    horizontal edge and pull each ``pull_pairs`` block toward its partner's x. When
    ``bands`` is falsy the pack is the original fixed-width-outline + aspect-correction
    search (byte-identical to before -- ``right_pull`` / ``pull_pairs`` are ignored).
    ``right_edge``/``left_edge`` are the HARD block-edge equalities lowered only
    from exact direct-member source ``@anchor`` declarations; they never inherit
    heuristic net/group pulls.

    ``left_column`` (opt-in, requires ``bands``): a single block id (the tall
    common-centroid diff-pair block). When given, the pack realises the REFERENCE
    macro SKELETON instead of the plain band stack: the ``left_column`` block is
    pulled OUT of the top band and pinned as a full-height LEFT COLUMN at the left
    edge; the pmos (TOP-band) mirrors form a FULL-WIDTH strip above the lower
    region; the nmos/passive lower region packs to the RIGHT of the column (nmos
    bottom, cap bank center, ``right_pull`` to the right edge). See
    :func:`_strip_solve_floorplan`. ``pull_pairs`` on this path pulls each named
    small-passive block toward its declared ``@adjacent`` neighbour's x-centre
    (a SOFT term, below the envelope objective), and ``order_pairs`` adds HARD
    same-band left/right order.

    ``order_pairs`` (opt-in): ``(a, b)`` id pairs from ``@order(a, b, left_of)``.
    The caller passes only SAME-BAND pairs; each adds a HARD ``a.x1 <= b.x0``
    constraint so the two blocks pack left-to-right in the declared order.

    ``vertical_order_pairs`` (opt-in): ``(lower, upper)`` id pairs from
    ``@order(lower, upper, below)`` or the normalized inverse of ``above``. Each
    adds a local hard ``lower.y1 <= upper.y0`` relation without promoting either
    endpoint into a global full-width band.

    ``axis_alignment_groups`` is a tuple of hard differential-axis groups.  An
    entry ``(block, offset2_nm)`` declares twice the local mirror-axis offset
    from that block's left edge; every entry in one group shares one exact
    global x axis.  It is accepted only on the banded master where vertical
    separation can keep aligned rigid blocks overlap-free.

    ``center_alignment_pairs`` contains symmetric ``(left, right, axis)``
    relations between rigid blocks.  It preserves a coordinate-free central
    overlap fact as an exact doubled-centre equality on the selected axis.

    ``aggregate_envelope_orders`` preserves strict order between the union-box
    centres of two functional scopes, including scopes whose envelopes overlap.

    Initial production solves derive alignment tolerance scales from the dense
    width seed before any legalizer or solver-domain expansion. Exact
    materialization replay can inherit that evidence only through the private,
    context-local replay wrapper.

    ``channel_nm`` (opt-in, floorplan path only): a reserved VERTICAL routing
    channel width (nm) from ``@channel(orient=vertical, width=W)``. Every lower-
    region block is pushed to ``x0 >= column_width + max(gap, channel_nm)``, so the
    corridor just RIGHT of the left column stays free of device geometry for the
    cross-band M3 straps. ``0`` (absent) keeps the plain ``gap`` (byte-identical).

    ``bottom_center_keepout_nm`` reserves the declared bottom-edge rail band below
    centre-band passives whose upper metal would obstruct that rail. Device bands
    remain free to use the same x/y footprint on lower layers. ``0`` preserves the
    prior placement exactly.

    ``bottom_edge_keepout_nm`` is the stronger exact-edge-rail contract: every
    block must start above this floorplan band except the structural block ids in
    ``bottom_edge_keepout_exempt``.  Exemptions are intended only for a boundary
    guard-ring master tied to the rail owner net; its frame/tap may occupy the band
    while its enclosed foreign-net geometry is shifted above it by materialization.
    ``bottom_edge_keepout_clear`` is distinct: each named floorplan master has an
    exact pre-route proof that it contains neither the rail metal nor an incident
    via cut.  Such a master is not subject to the rail floor, but is not pinned to
    the boundary.  This proof-scoped contract is accepted only with ``left_column``.

    ``strict_band_order`` explicitly enables/disables global band seams in the
    banded master; True coexists with hierarchical constraints. None retains the
    historical default. In the left-column master None remains False, and True
    preserves its dense lower-band order; ``strict_band_members`` may narrow it.

    ``solve_cuts`` is an opt-in tuple of replay-lowered placement cuts.  Every
    non-empty request must also provide one exact ``PlacementCutReplayProof`` per
    cut through ``solve_cut_proofs`` and the incumbent certificate through
    ``solve_cut_source_certificate``.  The proof lowerer is rerun before any cut
    variable is built; stale, foreign, unsupported, or member-missing requests
    return a named certificate residual.  An empty tuple imports no feedback
    module and preserves the historical model and result fields.

    ``aspect_obligation=False`` keeps ``aspect_target`` as a deterministic solver
    heuristic (including the unbanded width iteration) without adding hard aspect
    inequalities or a certificate/signoff requirement.  This is used only when
    the canonical source policy declares no aspect target.

    ``blocks`` is ``[(block_id, (x0, y0, x1, y1))]`` in nm.  Empty-cut placement
    ignores the positions and uses only block sizes; a non-empty cut request also
    treats the absolute bboxes as the source incumbent and requires an exact match
    to its source certificate. ``gap_nm`` is the clearance kept between any two
    blocks (a seam for ring spacing / routing). ``pair_clearances_nm`` may raise
    that clearance for exact structural block pairs (for example two N-well ring
    islands) without inflating unrelated neighbours. ``aspect_target`` = W/H of
    the sought outline.

    ``rigid_compound_footprints`` preserves a rigid master's local member
    rectangles while exposing the empty space between them to other masters.
    It is accepted only by the banded master; the member offsets remain fixed,
    but collision is checked against the exact local rectangles rather than the
    compound's solid outer bbox.

    Method -- FIXED-WIDTH OUTLINE + aspect correction. Each :func:`_strip_solve` fixes
    the outline WIDTH ``w_cap`` and MINIMIZES the height with CP-SAT (a genuinely tight
    optimization, unlike a bare feasibility probe, and always seeded to an incumbent by
    the shelf hint). An outer loop of at most ``_MAX_ITERS`` adjusts the width from the
    measured aspect -- ``w_cap *= sqrt(aspect_target / aspect_measured)`` (a wider strip
    raises W and lowers H, so it raises the aspect) -- until the measured aspect is
    within ``_ASPECT_TOL`` of the target, keeping the solve whose aspect is closest.
    This forces the tightness that the plain ``min W+H`` proxy does not, without the
    many expensive infeasibility proofs a fixed-AREA binary search needs.

    Determinism: every strip solve uses ``num_search_workers=1`` and
    ``random_seed=0``. ``max_time_s`` is one deterministic-time ledger for this
    placement call: the unbanded aspect loop divides it across all possible
    iterations, while a one-solve band/floorplan consumes the sole allocation.
    The width sequence is pure arithmetic of prior results; blocks are sorted by
    ``block_id`` so the model (and returned incumbent) is input-order-invariant.
    Raises ``RuntimeError`` only if the CP model is malformed (never fabricates a
    placement -- an over-budget solve honestly returns its shelf seed).
    """
    if edge_rail_reservation is not None:
        if not bands:
            bands = {name: 0 for name, _bbox in blocks}
    if incumbent_replay_enabled() and (not bands or left_column is not None):
        raise ValueError("incumbent replay currently requires the banded non-left-column master")
    if incumbent_replay_enabled() and incumbent_seed_positions is None:
        raise ValueError("incumbent replay requires a supplied generated seed")
    if type(floorplan_column_classes) is not tuple:
        raise TypeError("floorplan_column_classes must be one exact tuple")
    if type(floorplan_column_groups) is not tuple:
        raise TypeError("floorplan_column_groups must be one exact tuple")
    if type(aspect_obligation) is not bool:
        raise TypeError("aspect_obligation must be an exact bool")
    if strict_band_order is not None and type(strict_band_order) is not bool:
        raise TypeError("strict_band_order must be an exact bool or None")
    if type(placement_objective) is not str:
        raise TypeError("placement_objective must be an exact string")
    if placement_objective not in {"area", "height"}:
        raise ValueError(
            "placement_objective must be one of ['area', 'height'], "
            f"got {placement_objective!r}"
        )
    if placement_objective == "area" and not bands:
        raise ValueError(
            "area placement objective requires the banded placement master"
        )
    validate_area_target(area_target)
    if area_target is not None and (placement_objective != "area" or not bands or left_column is not None):
        raise ValueError("area target requires the banded non-left-column area master")
    if fixed_outline_width_nm is not None and (
        type(fixed_outline_width_nm) is not int or fixed_outline_width_nm <= 0
    ):
        raise ValueError("fixed_outline_width_nm must be a positive exact integer")
    if fixed_outline_width_nm is not None and not bands:
        raise ValueError("fixed outline width requires the banded placement master")
    if fixed_outline_width_nm is not None and left_column is not None:
        raise ValueError("fixed outline width is unsupported by the left-column master")
    if type(seed_cohesion_groups) is not tuple:
        raise TypeError("seed_cohesion_groups must be one exact tuple")
    if (seed_pair_clearances_nm is not None or seed_cohesion_groups) and (
        fixed_outline_width_nm is None or not bands
    ):
        raise ValueError(
            "certified seed relaxation authority requires a fixed-outline banded master"
        )
    minimum_occupancy = canonical_minimum_structural_occupancy(
        minimum_structural_occupancy
    )
    fill_seed = max(_FILL_SEED, minimum_occupancy or _FILL_SEED)
    aspect_policy_binding(aspect_target)
    gap = int(gap_nm)
    if gap < 0:
        raise ValueError("gap_nm must be non-negative")
    grid = int(manufacturing_grid_nm)
    if grid <= 0:
        raise ValueError("manufacturing_grid_nm must be a positive integer nm")
    grid_kwargs = {} if grid == 1 else {"manufacturing_grid_nm": grid}
    bottom_center_keepout = int(bottom_center_keepout_nm)
    if bottom_center_keepout < 0:
        raise ValueError("bottom_center_keepout_nm must be non-negative")
    bottom_edge_keepout = int(bottom_edge_keepout_nm)
    if bottom_edge_keepout < 0:
        raise ValueError("bottom_edge_keepout_nm must be non-negative")
    if type(top_edge_keepout_nm) is not int or top_edge_keepout_nm < 0:
        raise ValueError("top_edge_keepout_nm must be a non-negative exact integer")
    top_edge_keepout = top_edge_keepout_nm
    if type(top_edge_keepout_clear) is not tuple:
        raise TypeError("top_edge_keepout_clear must be one exact tuple")
    if any(
        type(block_id) is not str or not block_id or block_id != block_id.strip()
        for block_id in top_edge_keepout_clear
    ):
        raise TypeError(
            "top_edge_keepout_clear must contain canonical non-empty strings"
        )
    canonical_top_edge_clear = tuple(sorted(top_edge_keepout_clear))
    if (
        canonical_top_edge_clear != top_edge_keepout_clear
        or len(set(canonical_top_edge_clear)) != len(canonical_top_edge_clear)
    ):
        raise ValueError("top_edge_keepout_clear must be canonical and unique")
    normalized_pack_mode = str(pack_mode).strip().lower()
    if normalized_pack_mode not in {"rows", "dense_2d"}:
        raise ValueError(
            "pack_mode must be one of ['dense_2d', 'rows'], " f"got {pack_mode!r}"
        )
    if minimum_occupancy is not None:
        normalized_pack_mode = "dense_2d"
    # Relations are mathematical sets.  Canonicalize them at the public solver
    # boundary so hash-order from connectivity discovery cannot change CP-SAT
    # variable/constraint creation or the bounded FEASIBLE incumbent.
    canonical_right_pull = tuple(sorted({str(b) for b in (right_pull or ())}))
    canonical_left_pull = tuple(sorted({str(b) for b in (left_pull or ())}))
    overlap = sorted(set(canonical_right_pull) & set(canonical_left_pull))
    if overlap:
        raise ValueError(f"blocks cannot be pulled to both horizontal edges: {overlap}")
    canonical_right_edge = tuple(sorted({str(b) for b in (right_edge or ())}))
    canonical_left_edge = tuple(sorted({str(b) for b in (left_edge or ())}))
    edge_overlap = sorted(set(canonical_right_edge) & set(canonical_left_edge))
    if edge_overlap:
        raise ValueError(
            f"blocks cannot be constrained to both horizontal edges: {edge_overlap}"
        )
    edge_solver_kwargs = (
        {
            "right_edge": canonical_right_edge,
            "left_edge": canonical_left_edge,
        }
        if canonical_right_edge or canonical_left_edge
        else {}
    )
    canonical_pull_pairs = tuple(
        sorted({(str(a), str(b)) for a, b in (pull_pairs or ())})
    )
    canonical_order_pairs = tuple(
        sorted({(str(a), str(b)) for a, b in (order_pairs or ())})
    )
    canonical_vertical_order_pairs = tuple(
        sorted(
            {
                (str(lower), str(upper))
                for lower, upper in (vertical_order_pairs or ())
            }
        )
    )
    canonical_bottom_edge_exempt = tuple(
        sorted({str(block_id) for block_id in (bottom_edge_keepout_exempt or ())})
    )
    canonical_bottom_edge_clear = tuple(
        sorted(
            {str(block_id) for block_id in (bottom_edge_keepout_clear or ())}
            - set(canonical_bottom_edge_exempt)
        )
    )
    bottom_edge_solver_kwargs = (
        {
            "bottom_edge_keepout_nm": bottom_edge_keepout,
            "bottom_edge_keepout_exempt": canonical_bottom_edge_exempt,
        }
        if bottom_edge_keepout or canonical_bottom_edge_exempt
        else {}
    )
    raw_blocks = tuple(blocks)
    try:
        normalized_solve_cuts = tuple(solve_cuts or ())
    except TypeError:
        normalized_solve_cuts = None
    if normalized_solve_cuts:
        try:
            items = list(_exact_cut_input_blocks(raw_blocks))
        except ValueError:
            return _cut_residual_incumbent_result(
                (),
                bands,
                solve_cut_source_certificate,
                aspect_target,
                channel_nm,
                gap,
                "RESIDUAL_STALE_SOLVE_CUT",
                allow_unmatched_source=True,
                aspect_bounds=aspect_bounds,
                minimum_structural_occupancy=minimum_occupancy,
            )
    else:
        items = sorted(
            (
                (
                    str(bid),
                    (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])),
                )
                for bid, bb in raw_blocks
            ),
            key=lambda kv: kv[0],
        )
    block_ids = {name for name, _bbox in items}
    dims: Dict[str, Tuple[int, int]] = {
        bid: (bb[2] - bb[0], bb[3] - bb[1]) for bid, bb in items
    }
    if native_access_pair_clearances_nm is not None:
        from collections.abc import Mapping as _NativeAccessMapping
        from .edge_rail_reservation import EdgeRailReservationIR

        if not isinstance(native_access_pair_clearances_nm, _NativeAccessMapping):
            raise ValueError("native access pair clearances must be a mapping")
        if native_access_pair_clearances_nm or edge_rail_reservation is not None:
            if type(edge_rail_reservation) is not EdgeRailReservationIR:
                raise ValueError("native access pair clearances require the exact edge rail reservation IR")
            edge_rail_reservation = edge_rail_reservation.with_native_access_pair_clearances(
                native_access_pair_clearances_nm
            )
    _require_edge_rail_reservation(edge_rail_reservation, dims=dims)
    canonical_rigid_compound_footprints = (
        _canonical_rigid_compound_footprints(
            rigid_compound_footprints,
            dims=dims,
        )
    )
    canonical_layered_collision_footprints = (
        _canonical_layered_collision_footprints(
            layered_collision_footprints,
            dims=dims,
        )
    )
    canonical_row_bounded_cavity_projections = (
        _canonical_row_bounded_cavity_projections(
            row_bounded_cavity_projections,
            dims=dims,
        )
    )
    canonical_slot_projections = _canonical_mosaic_guest_slot_projections(
        mosaic_guest_slot_projections, dims=dims, minimum_clearance_nm=gap,
    )
    if canonical_rigid_compound_footprints and not bands:
        raise ValueError("rigid compound footprints require the banded master")
    if canonical_rigid_compound_footprints and left_column is not None:
        raise ValueError(
            "rigid compound footprints are unsupported by the left-column master"
        )
    if canonical_layered_collision_footprints and not bands:
        raise ValueError("layered collision footprints require the banded master")
    if canonical_layered_collision_footprints and left_column is not None:
        raise ValueError(
            "layered collision footprints are unsupported by the left-column master"
        )
    if canonical_row_bounded_cavity_projections and not bands:
        raise ValueError("row-bounded cavities require the banded placement master")
    if canonical_row_bounded_cavity_projections and left_column is not None:
        raise ValueError(
            "row-bounded cavities are unsupported by the left-column master"
        )
    if canonical_slot_projections and (not bands or left_column is not None):
        raise ValueError("mosaic guest-slot projections require the banded placement master without a left column")
    invalid_vertical_order_pairs = tuple(
        pair
        for pair in canonical_vertical_order_pairs
        if pair[0] == pair[1]
        or pair[0] not in block_ids
        or pair[1] not in block_ids
    )
    if invalid_vertical_order_pairs:
        raise ValueError(
            "vertical order pairs must name two distinct live blocks: "
            f"{invalid_vertical_order_pairs!r}"
        )
    canonical_flexible_envelopes = _canonical_flexible_envelope_constraints(
        flexible_envelope_constraints,
        block_ids=tuple(sorted(block_ids)),
    )
    if canonical_flexible_envelopes and not bands:
        raise ValueError("flexible envelope constraints require the banded master")
    canonical_floorplan_columns = _canonical_floorplan_column_classes(
        floorplan_column_classes,
        block_ids=tuple(sorted(block_ids)),
    )
    canonical_floorplan_groups = _canonical_floorplan_column_groups(
        floorplan_column_groups,
        block_ids={name: bbox for name, bbox in items},
    )
    if (canonical_floorplan_columns or canonical_floorplan_groups) and not bands:
        raise ValueError("floorplan column authority requires the banded placement master")
    if (
        placement_objective == "area"
        and left_column is not None
        and str(left_column) not in block_ids
    ):
        raise ValueError(
            "area placement objective requires the reference floorplan master"
        )
    unknown_edge_blocks = sorted(
        (set(canonical_right_edge) | set(canonical_left_edge)) - block_ids
    )
    if unknown_edge_blocks:
        raise ValueError(
            f"horizontal edge constraints contain unknown block ids: {unknown_edge_blocks}"
        )
    canonical_lateral_blocks = tuple(
        sorted({str(block_id) for block_id in (lateral_blocks or ())})
    )
    unknown_lateral_blocks = sorted(set(canonical_lateral_blocks) - block_ids)
    if unknown_lateral_blocks:
        raise ValueError(
            f"lateral floorplan blocks are unknown: {unknown_lateral_blocks}"
        )
    canonical_adjacency_limits = _canonical_adjacency_limits_nm(
        adjacency_limits_nm,
        block_ids=tuple(sorted(block_ids)),
        base_gap_nm=gap,
    )
    canonical_adjacency_scope_offsets = _canonical_adjacency_scope_offsets_nm(
        adjacency_scope_bboxes_nm,
        blocks=items,
        flexible_envelope_constraints=canonical_flexible_envelopes,
    )
    canonical_adjacency_scope_offsets = (
        _project_flexible_adjacency_scope_offsets_nm(
            canonical_adjacency_scope_offsets,
            constraints=canonical_flexible_envelopes,
            adjacency_limits_nm=canonical_adjacency_limits,
            dims={
                name: (bbox[2] - bbox[0], bbox[3] - bbox[1])
                for name, bbox in items
            },
        )
    )
    if canonical_adjacency_limits and not bands:
        raise ValueError("hard adjacency limits require the banded placement master")
    unknown_bottom_edge_exempt = sorted(set(canonical_bottom_edge_exempt) - block_ids)
    if unknown_bottom_edge_exempt:
        raise ValueError(
            "bottom_edge_keepout_exempt contains unknown block ids: "
            f"{unknown_bottom_edge_exempt}"
        )
    unknown_bottom_edge_clear = sorted(set(canonical_bottom_edge_clear) - block_ids)
    if unknown_bottom_edge_clear:
        raise ValueError(
            "bottom_edge_keepout_clear contains unknown block ids: "
            f"{unknown_bottom_edge_clear}"
        )
    unknown_top_edge_clear = sorted(set(canonical_top_edge_clear) - block_ids)
    if unknown_top_edge_clear:
        raise ValueError(
            "top_edge_keepout_clear contains unknown block ids: "
            f"{unknown_top_edge_clear}"
        )
    if top_edge_keepout and not bands:
        raise ValueError("top edge keepout requires the banded placement master")
    if top_edge_keepout and left_column is not None:
        raise ValueError(
            "top edge keepout is not supported by the left-column placement master"
        )
    if canonical_bottom_edge_clear and (
        left_column is None or str(left_column) not in block_ids
    ):
        raise ValueError(
            "bottom_edge_keepout_clear requires the floorplan left_column master"
        )
    canonical_strict_band_members = tuple(
        sorted({str(block_id) for block_id in (strict_band_members or ())})
    )
    unknown_strict_band_members = sorted(
        set(canonical_strict_band_members) - block_ids
    )
    if unknown_strict_band_members:
        raise ValueError(
            "strict_band_members contains unknown block ids: "
            f"{unknown_strict_band_members}"
        )
    prepared_cuts, cut_residual_status = _prepare_solve_cuts(
        solve_cuts if normalized_solve_cuts is None else normalized_solve_cuts,
        solve_cut_proofs,
        items,
        solve_cut_source_certificate,
    )
    incumbent_source_seed = _canonical_source_seed_positions(
        incumbent_seed_positions,
        ids=tuple(sorted(block_ids)),
    )
    if prepared_cuts and incumbent_source_seed is not None:
        raise ValueError(
            "incumbent seed positions cannot replace solve-cut source authority"
        )
    source_seed_positions = incumbent_source_seed
    if prepared_cuts:
        source_seed_positions = {
            name: (bbox[0], bbox[1])
            for name, bbox in _exact_cut_input_blocks(
                tuple(solve_cut_source_certificate.blocks)
            )
        }
    source_seed_kwargs = (
        {"source_seed_positions": source_seed_positions}
        if source_seed_positions is not None
        else {}
    )
    if cut_residual_status is not None:
        return _cut_residual_incumbent_result(
            items,
            bands,
            solve_cut_source_certificate,
            aspect_target,
            channel_nm,
            gap,
            cut_residual_status,
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
            minimum_structural_occupancy=minimum_occupancy,
        )
    if not items:
        return _placement_result(
            {},
            items,
            {},
            aspect_target,
            channel_nm,
            _SolverEvidence("OPTIMAL", "degenerate", 0.0, 0.0),
            gap_nm=gap,
            solve_cuts=prepared_cuts,
            cut_residual_status=cut_residual_status,
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
            minimum_structural_occupancy=minimum_occupancy,
            floorplan_column_classes=canonical_floorplan_columns,
            floorplan_column_groups=canonical_floorplan_groups,
            declared_bottom_edge_keepout_nm=bottom_edge_keepout,
            bottom_edge_keepout_carrier_blocks=canonical_bottom_edge_exempt,
            bottom_edge_keepout_clear_blocks=canonical_bottom_edge_clear,
            rigid_compound_footprints=canonical_rigid_compound_footprints,
            layered_collision_footprints=canonical_layered_collision_footprints,
        )
    pair_clearances = _canonical_pair_clearances_nm(
        pair_clearances_nm,
        block_ids=tuple(name for name, _bbox in items),
        base_gap_nm=gap,
    )
    canonical_overlap_pairs = _canonical_overlap_compatible_pairs(
        overlap_compatible_pairs,
        block_ids=tuple(name for name, _bbox in items),
        pair_clearances_nm=pair_clearances,
    )
    if canonical_overlap_pairs and (not bands or left_column is not None):
        raise ValueError(
            "overlap-compatible pairs require the banded non-left-column master"
        )
    seed_pair_clearances = (
        _canonical_pair_clearances_nm(
            seed_pair_clearances_nm,
            block_ids=tuple(name for name, _bbox in items),
            base_gap_nm=gap,
        )
        if seed_pair_clearances_nm is not None
        else pair_clearances
    )
    for pair in set(pair_clearances) | set(seed_pair_clearances):
        final_required = pair_clearances.get(pair, gap)
        seed_required = seed_pair_clearances.get(pair, gap)
        if seed_required < final_required:
            raise ValueError(
                "certified seed pair clearances must be a constraint tightening"
            )
    if len(items) == 1 and edge_rail_reservation is None:
        only_id = items[0][0]
        return _placement_result(
            {
                only_id: (
                    0,
                    (
                        0
                        if only_id in canonical_bottom_edge_exempt
                        or only_id in canonical_bottom_edge_clear
                        else bottom_edge_keepout
                    ),
                )
            },
            items,
            (
                {items[0][0]: int((bands or {}).get(items[0][0], _BAND_CENTER))}
                if bands
                else {}
            ),
            aspect_target,
            channel_nm,
            _SolverEvidence("OPTIMAL", "degenerate", 0.0, 0.0),
            gap_nm=gap,
            solve_cuts=prepared_cuts,
            cut_residual_status=cut_residual_status,
            pair_clearances_nm=pair_clearances,
            overlap_compatible_pairs=canonical_overlap_pairs,
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
            minimum_structural_occupancy=minimum_occupancy,
            floorplan_column_classes=canonical_floorplan_columns,
            floorplan_column_groups=canonical_floorplan_groups,
            declared_bottom_edge_keepout_nm=bottom_edge_keepout,
            bottom_edge_keepout_carrier_blocks=canonical_bottom_edge_exempt,
            bottom_edge_keepout_clear_blocks=canonical_bottom_edge_clear,
            rigid_compound_footprints=canonical_rigid_compound_footprints,
            layered_collision_footprints=canonical_layered_collision_footprints,
        )
    ids = [bid for bid, _ in items]
    canonical_axis_alignment_groups = _canonical_axis_alignment_groups(
        axis_alignment_groups,
        block_ids=ids,
        dims=dims,
    )
    canonical_center_alignment_pairs = _canonical_center_alignment_pairs(
        center_alignment_pairs,
        block_ids=ids,
        dims=dims,
    )
    canonical_aggregate_envelope_orders = _canonical_aggregate_envelope_orders(
        aggregate_envelope_orders,
        block_ids=ids,
    )
    canonical_hierarchical_macro_orders = _canonical_hierarchical_macro_orders(
        hierarchical_macro_orders,
        block_ids=ids,
    )
    canonical_hierarchical_macro_alignments = (
        _canonical_hierarchical_macro_alignments(
            hierarchical_macro_alignments,
            block_ids=ids,
        )
    )
    if type(hierarchical_cohesion_groups) is not tuple:
        raise TypeError("hierarchical_cohesion_groups must be one exact tuple")
    canonical_hierarchical_cohesion_groups = hierarchical_cohesion_groups
    if any(
        type(group) is not tuple
        or len(group) < 2
        or group != tuple(sorted(group))
        or len(set(group)) != len(group)
        or any(type(owner) is not str or owner not in set(ids) for owner in group)
        for group in canonical_hierarchical_cohesion_groups
    ) or len(set(canonical_hierarchical_cohesion_groups)) != len(
        canonical_hierarchical_cohesion_groups
    ):
        raise ValueError("hierarchical cohesion groups are not a unique live inventory")
    if (
        canonical_axis_alignment_groups
        or canonical_center_alignment_pairs
        or canonical_aggregate_envelope_orders
        or canonical_hierarchical_macro_orders
        or canonical_hierarchical_macro_alignments
        or canonical_hierarchical_cohesion_groups
    ) and not bands:
        raise ValueError("hard alignment relations require the banded placement master")
    max_w = max(dims[b][0] for b in ids)
    total = sum(dims[b][0] * dims[b][1] for b in ids)

    # ---- BAND-CONSTRAINED path (opt-in via ``bands``) -----------------------
    if bands:
        solve_budget = _DeterministicBudgetLedger.split(
            max_time_s, ("band_or_floorplan",)
        ).for_phase("band_or_floorplan")
        band_map = {bid: int(bands.get(bid, _BAND_CENTER)) for bid in ids}
        canonical_flexible_envelopes = _canonical_flexible_envelope_constraints(
            canonical_flexible_envelopes,
            block_ids=tuple(ids),
            bands=band_map,
        )
        if canonical_flexible_envelopes and left_column is not None:
            raise ValueError(
                "flexible envelope constraints are unsupported by the left-column master"
            )
        for group, row, _column, _count, terms in canonical_floorplan_groups:
            if row == _GLOBAL_FLOORPLAN_COLUMN_SCOPE:
                continue
            owner_bands = {band_map[owner] for owner, _n, _offset in terms}
            if owner_bands != {row}:
                raise ValueError(
                    f"floorplan functional group {group!r} row disagrees with "
                    "the exact band authority"
                )
        core_ids = tuple(
            block_id for block_id in ids if block_id not in canonical_lateral_blocks
        )
        if not core_ids:
            raise ValueError("lateral floorplan ownership consumed every block")
        present = sorted(set(band_map[block_id] for block_id in core_ids))

        # Freeze the normalization basis before a source seed, legalizer, aspect
        # repair, or solver-domain compatibility adjustment can enlarge it.
        natural = 0
        for band in present:
            row = [block_id for block_id in core_ids if band_map[block_id] == band]
            row_width = sum(dims[block_id][0] for block_id in row) + gap * max(
                0, len(row) - 1
            )
            natural = max(natural, row_width)
        if fixed_outline_width_nm is not None:
            w_cap = fixed_outline_width_nm
        elif normalized_pack_mode == "dense_2d":
            padded_area = sum(
                (dims[block_id][0] + gap) * (dims[block_id][1] + gap)
                for block_id in ids
            )
            order_width = max(
                (
                    _hard_order_width_nm(
                        [
                            block_id
                            for block_id in core_ids
                            if band_map[block_id] == band
                        ],
                        dims,
                        canonical_order_pairs,
                        gap,
                        pair_clearances,
                    )
                    for band in present
                ),
                default=0,
            )
            if canonical_lateral_blocks:
                order_width = max(
                    order_width,
                    _hard_order_width_nm(
                        ids,
                        dims,
                        canonical_order_pairs,
                        gap,
                        pair_clearances,
                    ),
                )
            w_cap = max(
                max_w,
                order_width,
                _axis_alignment_required_width_nm(
                    canonical_axis_alignment_groups, dims
                ),
                int(
                    math.ceil(
                        math.sqrt(
                            max(1, padded_area)
                            / fill_seed
                            * max(float(aspect_target), 1e-9)
                        )
                    )
                ),
            )
        else:
            w_cap = max(
                max_w,
                natural,
                _hard_order_width_nm(
                    ids,
                    dims,
                    canonical_order_pairs,
                    gap,
                    pair_clearances,
                ),
            )
        source_certificate = _ALIGNMENT_SCALE_SOURCE_CERTIFICATE.get()
        if source_certificate is None:
            frozen_alignment_scales = (
                _derive_hierarchical_alignment_tolerance_scales_nm(
                    canonical_hierarchical_macro_alignments,
                    frozen_width_nm=int(w_cap),
                    declared_aspect=float(aspect_target),
                    maximum_master_height_nm=max(
                        int(dims[block_id][1]) for block_id in ids
                    ),
                )
            )
        else:
            if type(source_certificate) is not FinalPlacementCertificate:
                raise TypeError(
                    "alignment scale source must be an exact FinalPlacementCertificate"
                )
            final_placement_state_digest(source_certificate)
            if (
                source_certificate.hierarchical_macro_alignments
                != canonical_hierarchical_macro_alignments
            ):
                raise ValueError(
                    "alignment scale source certificate differs from replay alignment inventory"
                )
            source_alignment_scales = (
                source_certificate.hierarchical_alignment_tolerance_scales_nm
            )
            frozen_alignment_scales = (
                _require_hierarchical_alignment_tolerance_scales_nm(
                    source_alignment_scales,
                    alignments=canonical_hierarchical_macro_alignments,
                )
            )
        # ---- REFERENCE-SKELETON floorplan path (opt-in via ``left_column``) -----
        if left_column is not None and str(left_column) in ids:
            if canonical_vertical_order_pairs:
                raise ValueError(
                    "vertical order pairs are unsupported by the left-column "
                    "placement master"
                )
            pos, solve_evidence = _strip_solve_floorplan(
                ids,
                dims,
                gap,
                band_map,
                str(left_column),
                edge_rail_reservation=edge_rail_reservation,
                right_pull=canonical_right_pull,
                left_pull=canonical_left_pull,
                max_time_s=solve_budget,
                pull_pairs=canonical_pull_pairs,
                adjacency_limits_nm=canonical_adjacency_limits,
                adjacency_scope_offsets_nm=canonical_adjacency_scope_offsets,
                order_pairs=canonical_order_pairs,
                lateral_blocks=canonical_lateral_blocks,
                floorplan_column_classes=canonical_floorplan_columns,
                floorplan_column_groups=canonical_floorplan_groups,
                channel_nm=int(channel_nm or 0),
                bottom_center_keepout_nm=bottom_center_keepout,
                bottom_edge_keepout_clear=canonical_bottom_edge_clear,
                pack_mode=normalized_pack_mode,
                strict_band_order=bool(strict_band_order),
                strict_band_members=canonical_strict_band_members,
                solve_cuts=prepared_cuts,
                pair_clearances_nm=pair_clearances,
                axis_alignment_groups=canonical_axis_alignment_groups,
                center_alignment_pairs=canonical_center_alignment_pairs,
                aggregate_envelope_orders=canonical_aggregate_envelope_orders,
                hierarchical_macro_orders=canonical_hierarchical_macro_orders,
                hierarchical_macro_alignments=canonical_hierarchical_macro_alignments,
                hierarchical_alignment_tolerance_scales_nm=(
                    frozen_alignment_scales
                ),
                aspect_target=aspect_target,
                aspect_bounds=aspect_bounds,
                aspect_obligation=aspect_obligation,
                placement_objective=placement_objective,
                **grid_kwargs,
                **edge_solver_kwargs,
                **source_seed_kwargs,
                **bottom_edge_solver_kwargs,
            )
            _assert_axis_alignment_positions(pos, canonical_axis_alignment_groups)
            _assert_center_alignment_positions(
                pos, dims, canonical_center_alignment_pairs
            )
            _assert_aggregate_envelope_order_positions(
                pos, dims, canonical_aggregate_envelope_orders
            )
            _assert_hierarchical_macro_positions(
                pos,
                dims,
                canonical_hierarchical_macro_orders,
                canonical_hierarchical_macro_alignments,
                gap_nm=gap,
                tolerance_scales_nm=frozen_alignment_scales,
            )
            return _placement_result(
                pos,
                items,
                band_map,
                aspect_target,
                channel_nm,
                solve_evidence,
                gap_nm=gap,
                solve_cuts=prepared_cuts,
                cut_residual_status=cut_residual_status,
                pair_clearances_nm=pair_clearances,
                aspect_bounds=aspect_bounds,
                aspect_obligation=aspect_obligation,
                axis_alignment_groups=canonical_axis_alignment_groups,
                center_alignment_pairs=canonical_center_alignment_pairs,
                aggregate_envelope_orders=canonical_aggregate_envelope_orders,
                hierarchical_macro_orders=canonical_hierarchical_macro_orders,
                hierarchical_macro_alignments=canonical_hierarchical_macro_alignments,
                hierarchical_alignment_tolerance_scales_nm=(
                    frozen_alignment_scales
                ),
                minimum_structural_occupancy=minimum_occupancy,
                floorplan_column_classes=canonical_floorplan_columns,
                floorplan_column_groups=canonical_floorplan_groups,
                declared_bottom_edge_keepout_nm=bottom_edge_keepout,
                bottom_edge_keepout_carrier_blocks=canonical_bottom_edge_exempt,
                bottom_edge_keepout_clear_blocks=canonical_bottom_edge_clear,
                edge_rail_reservation=edge_rail_reservation,
            )
        final_source_seed_kwargs = source_seed_kwargs
        if incumbent_source_seed is not None:
            final_source_seed_kwargs = {
                "source_seed_positions": incumbent_source_seed,
                "certified_source_seed": True,
            }
        if (
            fixed_outline_width_nm is not None
            and len(present) > 1
            and not prepared_cuts
            and incumbent_source_seed is None
            and not requires_complete_native_master()
        ):
            seed_attempt = attempt_banded_seed(_fixed_outline_banded_seed,
                ids,
                dims,
                gap,
                fixed_outline_width_nm,
                band_map,
                max_time_s=solve_budget,
                adjacency_limits_nm=canonical_adjacency_limits,
                adjacency_scope_offsets_nm=canonical_adjacency_scope_offsets,
                order_pairs=canonical_order_pairs,
                floorplan_column_classes=canonical_floorplan_columns,
                floorplan_column_groups=canonical_floorplan_groups,
                seed_cohesion_groups=seed_cohesion_groups,
                hierarchical_cohesion_groups=(
                    canonical_hierarchical_cohesion_groups
                ),
                pair_clearances_nm=seed_pair_clearances,
                axis_alignment_groups=canonical_axis_alignment_groups,
                pack_mode=normalized_pack_mode,
                aspect_target=aspect_target,
                aspect_bounds=aspect_bounds,
                aspect_obligation=aspect_obligation,
                manufacturing_grid_nm=grid,
                right_edge=canonical_right_edge,
                left_edge=canonical_left_edge,
                bottom_center_keepout_nm=bottom_center_keepout,
                bottom_edge_keepout_nm=bottom_edge_keepout,
                top_edge_keepout_nm=top_edge_keepout,
                top_edge_keepout_clear=canonical_top_edge_clear,
                flexible_envelope_constraints=canonical_flexible_envelopes,
                rigid_compound_footprints=canonical_rigid_compound_footprints,
                layered_collision_footprints=canonical_layered_collision_footprints,
                mosaic_guest_slot_projections=canonical_slot_projections,
                edge_rail_reservation=edge_rail_reservation,
                overlap_compatible_pairs=canonical_overlap_pairs,
            )
            solve_budget = seed_attempt.remaining_budget
            final_source_seed_kwargs = seed_attempt.source_kwargs(final_source_seed_kwargs)
        pos, _w, _h, solve_evidence = _strip_solve_banded(
            ids,
            dims,
            gap,
            w_cap,
            band_map,
            right_pull=canonical_right_pull,
            left_pull=canonical_left_pull,
            pull_pairs=canonical_pull_pairs,
            adjacency_limits_nm=canonical_adjacency_limits,
            adjacency_scope_offsets_nm=canonical_adjacency_scope_offsets,
            max_time_s=solve_budget,
            lateral_blocks=canonical_lateral_blocks,
            area_target=area_target,
            flexible_envelope_constraints=canonical_flexible_envelopes,
            rigid_compound_footprints=canonical_rigid_compound_footprints,
            layered_collision_footprints=canonical_layered_collision_footprints,
            row_bounded_cavity_projections=(
                canonical_row_bounded_cavity_projections
            ),
            mosaic_guest_slot_projections=canonical_slot_projections,
            edge_rail_reservation=edge_rail_reservation,
            order_pairs=canonical_order_pairs,
            vertical_order_pairs=canonical_vertical_order_pairs,
            floorplan_column_classes=canonical_floorplan_columns,
            floorplan_column_groups=canonical_floorplan_groups,
            cohesion_groups=_floorplan_cell_cohesion_groups(
                canonical_floorplan_groups
            ),
            well_cohesion_groups=seed_cohesion_groups,
            hierarchical_cohesion_groups=(
                canonical_hierarchical_cohesion_groups
            ),
            bottom_center_keepout_nm=bottom_center_keepout,
            top_edge_keepout_nm=top_edge_keepout,
            top_edge_keepout_clear=canonical_top_edge_clear,
            solve_cuts=prepared_cuts,
            pair_clearances_nm=pair_clearances,
            overlap_compatible_pairs=canonical_overlap_pairs,
            axis_alignment_groups=canonical_axis_alignment_groups,
            center_alignment_pairs=canonical_center_alignment_pairs,
            aggregate_envelope_orders=canonical_aggregate_envelope_orders,
            hierarchical_macro_orders=canonical_hierarchical_macro_orders,
            hierarchical_macro_alignments=canonical_hierarchical_macro_alignments,
            hierarchical_alignment_tolerance_scales_nm=frozen_alignment_scales,
            pack_mode=normalized_pack_mode,
            enforce_legacy_band_order=resolve_band_order(
                strict_band_order, hierarchical_macro_orders=canonical_hierarchical_macro_orders),
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
            **grid_kwargs,
            **edge_solver_kwargs,
            **final_source_seed_kwargs,
            **bottom_edge_solver_kwargs,
            placement_objective=placement_objective,
            strict_width_cap=fixed_outline_width_nm is not None,
            exact_outline_width=fixed_outline_width_nm is not None,
        )
        _assert_axis_alignment_positions(pos, canonical_axis_alignment_groups)
        _assert_center_alignment_positions(
            pos, dims, canonical_center_alignment_pairs
        )
        _assert_aggregate_envelope_order_positions(
            pos, dims, canonical_aggregate_envelope_orders
        )
        _assert_hierarchical_macro_positions(
            pos,
            dims,
            canonical_hierarchical_macro_orders,
            canonical_hierarchical_macro_alignments,
            gap_nm=gap,
            tolerance_scales_nm=frozen_alignment_scales,
        )
        if any(
            not _row_bounded_cavity_satisfied(
                pos,
                dims=dims,
                projection=row,
                minimum_clearance_nm=gap,
            )
            for row in canonical_row_bounded_cavity_projections
        ):
            raise RuntimeError("row-bounded cavity placement replay failed")
        if any(not mosaic_guest_slot_satisfied(pos, row) for row in canonical_slot_projections):
            raise RuntimeError("mosaic guest-slot placement replay failed")
        return _placement_result(
            pos,
            items,
            band_map,
            aspect_target,
            channel_nm,
            solve_evidence,
            gap_nm=gap,
            solve_cuts=prepared_cuts,
            cut_residual_status=cut_residual_status,
            pair_clearances_nm=pair_clearances,
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
            axis_alignment_groups=canonical_axis_alignment_groups,
            center_alignment_pairs=canonical_center_alignment_pairs,
            aggregate_envelope_orders=canonical_aggregate_envelope_orders,
            hierarchical_macro_orders=canonical_hierarchical_macro_orders,
            hierarchical_macro_alignments=canonical_hierarchical_macro_alignments,
            hierarchical_alignment_tolerance_scales_nm=frozen_alignment_scales,
            minimum_structural_occupancy=minimum_occupancy,
            fixed_outline_width_nm=fixed_outline_width_nm,
            floorplan_column_classes=canonical_floorplan_columns,
            floorplan_column_groups=canonical_floorplan_groups,
            declared_bottom_edge_keepout_nm=bottom_edge_keepout,
            bottom_edge_keepout_carrier_blocks=canonical_bottom_edge_exempt,
            bottom_edge_keepout_clear_blocks=canonical_bottom_edge_clear,
            rigid_compound_footprints=canonical_rigid_compound_footprints,
            layered_collision_footprints=canonical_layered_collision_footprints,
            row_bounded_cavity_projections=(
                canonical_row_bounded_cavity_projections
            ),
            mosaic_guest_slot_projections=canonical_slot_projections,
            edge_rail_reservation=edge_rail_reservation,
            overlap_compatible_pairs=canonical_overlap_pairs,
        )

    # first strip width from the aspect target at a dense-pack fill estimate
    w_cap = max(max_w, int(round(math.sqrt(total / fill_seed * aspect_target))))

    best_pos: Dict[str, Tuple[int, int]] = {}
    best_evidence = _SolverEvidence("FALLBACK", "height_nm", None, None, True, "seed")
    best_err = float("inf")
    iteration_labels = tuple(f"aspect_iteration_{index}" for index in range(_MAX_ITERS))
    solve_ledger = _DeterministicBudgetLedger.split(max_time_s, iteration_labels)
    for iteration_label in iteration_labels:
        pos, w_used, h_used, solve_evidence = _strip_solve(
            ids,
            dims,
            gap,
            w_cap,
            solve_ledger.for_phase(iteration_label),
            solve_cuts=prepared_cuts,
            pair_clearances_nm=pair_clearances,
            aspect_target=aspect_target,
            aspect_bounds=aspect_bounds,
            **grid_kwargs,
            **edge_solver_kwargs,
            **source_seed_kwargs,
            **bottom_edge_solver_kwargs,
        )
        aspect = (w_used / h_used) if h_used else aspect_target
        err = abs(aspect - aspect_target) / aspect_target
        if err < best_err:
            best_err, best_pos, best_evidence = err, pos, solve_evidence
        if err <= _ASPECT_TOL:
            break
        # nudge the width toward the target aspect (guard against a degenerate 0)
        w_cap = max(
            max_w, int(round(w_cap * math.sqrt(aspect_target / max(aspect, 1e-6))))
        )

    return _placement_result(
        best_pos,
        items,
        {},
        aspect_target,
        channel_nm,
        best_evidence,
        gap_nm=gap,
        solve_cuts=prepared_cuts,
        cut_residual_status=cut_residual_status,
        pair_clearances_nm=pair_clearances,
        aspect_bounds=aspect_bounds,
        aspect_obligation=aspect_obligation,
        minimum_structural_occupancy=minimum_occupancy,
        declared_bottom_edge_keepout_nm=bottom_edge_keepout,
        bottom_edge_keepout_carrier_blocks=canonical_bottom_edge_exempt,
        bottom_edge_keepout_clear_blocks=canonical_bottom_edge_clear,
    )

def place_blocks_2d(
    blocks: Sequence[Tuple[str, _Bbox]],
    gap_nm: int,
    aspect_target: float = 1.67,
    max_time_s: float = 30.0,
    bands: Any = None,
    floorplan_column_classes: Any = (),
    floorplan_column_groups: Any = (),
    right_pull: Any = None,
    left_pull: Any = None,
    right_edge: Any = None,
    left_edge: Any = None,
    pull_pairs: Any = None,
    adjacency_limits_nm: Any = (),
    adjacency_scope_bboxes_nm: Any = (),
    left_column: Any = None,
    lateral_blocks: Any = (),
    order_pairs: Any = None,
    vertical_order_pairs: Any = None,
    channel_nm: int = 0,
    bottom_center_keepout_nm: int = 0,
    bottom_edge_keepout_nm: int = 0,
    bottom_edge_keepout_exempt: Any = (),
    bottom_edge_keepout_clear: Any = (),
    top_edge_keepout_nm: int = 0,
    top_edge_keepout_clear: Any = (),
    pack_mode: str = "rows",
    strict_band_order: Optional[bool] = None,
    strict_band_members: Any = (),
    axis_alignment_groups: Any = (),
    center_alignment_pairs: Any = (),
    aggregate_envelope_orders: Any = (),
    solve_cuts: Any = (),
    solve_cut_proofs: Any = (),
    solve_cut_source_certificate: Any = None,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = True,
    manufacturing_grid_nm: int = 1,
    minimum_structural_occupancy: Optional[float] = None,
    placement_objective: str = "height",
    fixed_outline_width_nm: Optional[int] = None,
    flexible_envelope_constraints: Any = (),
    edge_rail_reservation: Any = None,
) -> Dict[str, Tuple[int, int]]:
    """Return positions, but never erase a non-empty solve-cut residual.

    Empty-cut calls retain the historical positions-only behavior.  A caller
    opting into cuts must use the certificate API to inspect a rejected or
    geometrically illegal incumbent; this wrapper raises a typed stop instead
    of returning those positions as if they were a successful solve.
    """

    try:
        frozen_solve_cuts = tuple(solve_cuts or ())
    except TypeError:
        frozen_solve_cuts = (solve_cuts,)
    try:
        frozen_solve_cut_proofs = tuple(solve_cut_proofs or ())
    except TypeError:
        frozen_solve_cut_proofs = (solve_cut_proofs,)

    result = place_blocks_2d_with_certificate(
        blocks,
        gap_nm,
        edge_rail_reservation=edge_rail_reservation,
        aspect_target=aspect_target,
        max_time_s=max_time_s,
        bands=bands,
        floorplan_column_classes=floorplan_column_classes,
        floorplan_column_groups=floorplan_column_groups,
        right_pull=right_pull,
        left_pull=left_pull,
        right_edge=right_edge,
        left_edge=left_edge,
        pull_pairs=pull_pairs,
        adjacency_limits_nm=adjacency_limits_nm,
        adjacency_scope_bboxes_nm=adjacency_scope_bboxes_nm,
        left_column=left_column,
        lateral_blocks=lateral_blocks,
        order_pairs=order_pairs,
        vertical_order_pairs=vertical_order_pairs,
        channel_nm=channel_nm,
        bottom_center_keepout_nm=bottom_center_keepout_nm,
        bottom_edge_keepout_nm=bottom_edge_keepout_nm,
        bottom_edge_keepout_exempt=bottom_edge_keepout_exempt,
        bottom_edge_keepout_clear=bottom_edge_keepout_clear,
        top_edge_keepout_nm=top_edge_keepout_nm,
        top_edge_keepout_clear=top_edge_keepout_clear,
        pack_mode=pack_mode,
        strict_band_order=strict_band_order,
        strict_band_members=strict_band_members,
        axis_alignment_groups=axis_alignment_groups,
        center_alignment_pairs=center_alignment_pairs,
        aggregate_envelope_orders=aggregate_envelope_orders,
        solve_cuts=frozen_solve_cuts,
        solve_cut_proofs=frozen_solve_cut_proofs,
        solve_cut_source_certificate=solve_cut_source_certificate,
        aspect_bounds=aspect_bounds,
        aspect_obligation=aspect_obligation,
        manufacturing_grid_nm=manufacturing_grid_nm,
        minimum_structural_occupancy=minimum_structural_occupancy,
        placement_objective=placement_objective,
        fixed_outline_width_nm=fixed_outline_width_nm,
        flexible_envelope_constraints=flexible_envelope_constraints,
    )
    if frozen_solve_cuts and not (
        result.certificate.geometry_legal and result.certificate.cuts_satisfied
    ):
        raise PlacementSolveCutResidualError(result.certificate)
    return dict(result.positions)

# Preserve the declared input schemas while transparent diagnostic wrappers may
# observe the runtime callables. Actual native execution still uses the globals.
_NATIVE_STAGE_DECLARATIONS = (
    _strip_solve_banded,
    _legalize_fixed_outline_seed_x,
    _banded_seed_satisfies_hard_constraints,
)
