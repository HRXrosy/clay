"""Compatibility facade for the modular deterministic 2-D placer."""
from __future__ import annotations

import sys
from types import ModuleType

import hashlib
import json
import math
import re
from contextvars import ContextVar
from contextlib import nullcontext
from itertools import combinations
from dataclasses import asdict, dataclass, is_dataclass, replace
from clay.layout.mosaic_guest_slot_projection import MosaicGuestSlotProjection, canonical_mosaic_guest_slot_projections, add_mosaic_guest_slot_constraints, mosaic_guest_slot_satisfied
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
from clay.layout.compact_blocks import _NM_PER_UM, apply_block_deltas, extract_blocks
from clay.layout.policy import aspect_policy_binding, outline_width_policy_binding
from clay.layout.intent_basis import IntentBasis, IntentConsumptionCertificate
from clay.layout.intent_closure import ConsumerProjection
from clay.layout.intent_facts import EntityKind, IntentFactKind, IntentFactSet, SourceInventory
from clay.layout.place.final_placement_state import MatchingPairAxisProjection, canonical_minimum_structural_occupancy, functional_od_policy_binding, structural_occupancy_policy_binding, verify_matching_pair_projection_owners
from clay.layout.place.solver_execution_policy import apply_area_search_execution_policy, incumbent_replay_enabled, prepare_incumbent_replay_model, require_incumbent_replay_response, _complete_incumbent_replay_stage
from clay.layout.placement_infeasibility import ConstraintObligationTrace, PlacementInfeasibilityError, SufficientCoreDiagnostic
from clay.layout.spatial.abstract_ir import SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
from clay.layout.spatial.intent_projection import ReferenceConditionedPlacementObligation, replay_reference_conditioned_placement_obligation_inventory
from clay.layout.canonical_collision_projection import CanonicalRowBoundedCavityProjection, build_row_bounded_cavity_projection
from clay.layout.compact_blocks import assert_no_block_overlap

from clay.layout.place import placement_constants as _owner_placement_constants
from clay.layout.place import placement_cuts as _owner_placement_cuts
from clay.layout.place import placement_diagnostics as _owner_placement_diagnostics
from clay.layout.place import constraints_spatial as _owner_constraints_spatial
from clay.layout.place import constraints_hierarchical as _owner_constraints_hierarchical
from clay.layout.place import constraints_well_guard as _owner_constraints_well_guard
from clay.layout.place import placement_geometry as _owner_placement_geometry
from clay.layout.place import placement_relations as _owner_placement_relations
from clay.layout.place import placement_replay as _owner_placement_replay
from clay.layout.place import full2d_compactor as _owner_full2d_compactor
from clay.layout.place import placement_certificates as _owner_placement_certificates
from clay.layout.place import cp_model_builder as _owner_cp_model_builder
from clay.layout.place import materialized_placement as _owner_materialized_placement

from clay.layout.place.placement_constants import (
    ASPECT_RELATIVE_TOLERANCE,
    _ALIGNMENT_SCALE_SOURCE_CERTIFICATE,
    _ASPECT_CONSTRAINT_SCALE,
    _ASPECT_TOL,
    _AggregateEnvelopeOrder,
    _BAND_BOTTOM,
    _BAND_CENTER,
    _BAND_TOP,
    _Bbox,
    _FILL_SEED,
    _FloorplanColumnGroup,
    _FloorplanColumnTerm,
    _GLOBAL_FLOORPLAN_COLUMN_SCOPE,
    _HierarchicalAlignmentToleranceScalesNm,
    _HierarchicalMacroAlignment,
    _HierarchicalMacroAlignmentGroups,
    _HierarchicalMacroOrder,
    _LayeredCollisionAtom,
    _LayeredCollisionFootprint,
    _MAX_ITERS,
    _RigidCompoundFootprint,
    _RowBoundedCavityProjection,
    _SECONDARY_AXIS_REPAIRABLE_BLOCKERS,
)
from clay.layout.place.placement_cuts import (
    _PreparedSolveCut,
    _add_solve_cut_constraints,
    _cut_extent_padding,
    _cut_satisfied_by_bboxes,
    _exact_cut_input_blocks,
    _is_canonical_sha256,
    _prepare_solve_cuts,
    _required_cut_axis_extent,
)
from clay.layout.place.placement_diagnostics import (
    Full2DCompactionDiagnosticError,
    _canonical_full_hard_failure_rows,
    _diagnostic_canonical_value,
    _diagnostic_safe_canonical_value,
    _full2d_compaction_failure_diagnostic,
    _full2d_compaction_position_rows,
    _full2d_input_provenance,
    _full_hard_replay_row,
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
    _add_rigid_compound_collision_constraints,
    _add_row_bounded_cavity_constraints,
    _add_top_edge_keepout_constraints,
    _assert_axis_alignment_positions,
    _assert_center_alignment_positions,
    _assert_no_physical_block_overlap,
    _axis_alignment_required_width_nm,
    _banded_axis_aligned_seed,
    _banded_dense_seed,
    _banded_shelf_pack,
    _bbox_clearance_nm,
    _canonical_adjacency_limits_nm,
    _canonical_adjacency_scope_offsets_nm,
    _canonical_axis_alignment_groups,
    _canonical_center_alignment_pairs,
    _canonical_flexible_envelope_constraints,
    _canonical_flexible_offset_rows,
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
    _exact_matching_bbox,
    _flexible_seed_physical_views,
    _floorplan_cell_cohesion_groups,
    _floorplan_column_group_order_satisfied,
    _floorplan_column_order_satisfied,
    _floorplan_seed,
    _hard_edge_compatible_banded_seed,
    _hard_order_width_nm,
    _landscape_place_enabled,
    _ordered_row_positions,
    _pair_collision_forbidden_box_cover,
    _placement_declaration_refs,
    _placement_objective_from_intent,
    _placement_primary_extent_nm,
    _prefer_certified_seed_positions,
    _project_flexible_adjacency_scope_offsets_nm,
    _required_pair_gap_nm,
    _right_edge_compatible_width_bound,
    _row_bounded_cavity_satisfied,
    _scaled_aspect_bounds,
    _seed_geometry_failure_rows,
    _seed_geometry_legal,
    _shelf_pack,
    _stable_hard_order,
    _strict_band_order_required,
    _strict_outline_aspect_height_floor_nm,
    _translate_fixed_outline_seed_for_columns,
    _vertical_channel_nm,
    assign_bands,
    common_centroid_devices,
    floorplan_column_classes,
    floorplan_column_groups,
    identify_left_column,
)
from clay.layout.place.constraints_hierarchical import (
    _add_aggregate_envelope_order_constraints,
    _add_hierarchical_macro_constraints,
    _aggregate_envelope_order_satisfied,
    _assert_aggregate_envelope_order_positions,
    _assert_hierarchical_macro_positions,
    _canonical_aggregate_envelope_orders,
    _canonical_hierarchical_alignment_tolerance_scales_nm,
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
    _hierarchical_vertical_leading_edge_seed,
    _layered_component_refinement_owners,
    _orthogonal_single_master_aggregate_refinement_owners,
    _orthogonal_single_master_cross_tier_aggregate_keys,
    _primary_hierarchical_macro_alignments,
    _require_hierarchical_alignment_tolerance_scales_nm,
    _secondary_axis_aggregate_orders,
    _secondary_axis_collision_y_scope,
    _secondary_axis_repair_admits_blockers,
)
from clay.layout.place.constraints_well_guard import (
    _adjacency_compatible_bottom_exemptions,
    _bottom_edge_keepout_clear_blocks,
    _canonical_pair_clearances_nm,
    _compiled_edge_rail_keepouts_nm,
    _compiled_edge_rail_specs_for_placement,
    _edge_keepout_clear_blocks,
    _effective_block_gap_nm,
    _exact_member_ring_padding_rows,
    _hard_placement_compatible_bottom_exemptions,
    _installed_adjacency_pitch_nm,
    _layer_disjoint_overlap_compatible_pairs,
    _materialize_same_ring_passive_adjacency,
    _merge_pair_clearances_nm,
    _mim_pair_clearances_nm,
    _pdk_layer_clearances_nm,
    _project_pair_clearances_to_masters,
    _replay_ring_padding_on_positioned_master,
    _resolve_block_member_devices,
    _resolved_adjacency_limits_nm,
    _resolved_device_master_rows,
    _ring_pair_clearances_nm,
    _same_bulk_mergeable_pair_layers,
    _same_bulk_nwell_pair_clearances_nm,
    _shared_well_passive_gap_nm,
    _signed_member_ring_padding_names,
    _top_edge_keepout_clear_blocks,
    _well_share_ring_cohesion_groups,
    bundle_passive_composites,
    replay_canonical_member_ring_padding,
    replay_member_ring_padding,
)
from clay.layout.place.placement_geometry import (
    _bundle_declared_edge_alignment_composites,
    _bundle_mos_matching_composites,
    _bundle_rigid_nonmos_composites,
    _canonical_materialized_block_source_members,
    _canonical_mos_matching_axis_rows,
    _diff_pair_axis_alignment_groups,
    _discrete_mosaic_local_order_proofs,
    _edge_alignment_bbox_device_inventory,
    _feedback_final_master_inventory,
    _fixed_pcell_compound_footprints_from_geometry,
    _layered_collision_footprints_from_geometry,
    _matching_axis_cumulative_authority,
    _matching_axis_master_authority,
    _materialized_adjacency_scope_bboxes_nm,
    _materialized_source_ownership_requested,
    _project_mos_matching_axes_through_block_deltas,
    _project_source_device_bboxes_to_solver_masters,
    _resistor_adjacency_terms,
    _rigid_compound_footprints_from_members,
    _source_bbox_refinement_rows,
    _source_device_bboxes_nm,
    _source_device_occurrence_envelopes_nm,
)
from clay.layout.place.placement_relations import (
    _aggregate_envelope_orders_from_intent,
    _aggregate_hierarchical_macro_alignments_from_intent,
    _aggregate_scope_live_masters,
    _hierarchical_macro_cohesion_groups_from_intent,
    _hierarchical_macro_constraints_from_intent,
    _horizontal_order_is_materializable,
    _left_column_after_explicit_edge_anchor,
    _mosaic_center_alignment_pairs,
    _nonrigid_place_group_blocks,
    _reference_conditioned_envelope_separations_from_intent,
    _reference_conditioned_row_bounded_cavities_from_intent,
    _relational_place_group_envelope_edges,
    _relational_place_group_master_constraints,
)
from clay.layout.place.placement_replay import (
    _adjacency_seed_satisfied,
    _banded_seed_satisfies_hard_constraints,
    _require_edge_rail_reservation,
)
from clay.layout.place.full2d_compactor import (
    _add_full2d_outer_hard_constraints,
    _compact_area_frontier_positions,
    _compact_full2d_aggregate_target,
    _compact_full2d_master_positions,
    _compact_full2d_master_positions_impl,
    _compact_lateral_sink_positions_x,
    _compaction_replay_width_bound,
)
from clay.layout.place.placement_certificates import (
    PlacementSolveCutResidualError,
    _SolverEvidence,
    FinalPlacementCertificate,
    PlaceBlocks2DResult,
    RepositionPum2DResult,
    _assert_materialized_diff_axis_certificate,
    _bind_certificate_to_materialized_blocks,
    _certificate_objective_is_optimal,
    _placement_result,
    final_placement_state_digest,
    placement_certificate_is_signable,
    placement_certificate_objective_evidence,
    placement_certificate_proves_area_optimality,
)
from clay.layout.place.cp_model_builder import (
    _DeterministicBudgetLedger,
    _NATIVE_STAGE_DECLARATIONS,
    _NativeStageAcquisitionStop,
    _canonical_source_seed_positions,
    _configure_2d_packing_proof,
    _cut_residual_incumbent_result,
    _fixed_outline_banded_seed,
    _legalize_fixed_outline_seed_x,
    _native_fixed_outline_stage_inputs,
    _native_full_stage_replay_kwargs,
    _native_stage_frame_fields,
    _place_blocks_2d_with_alignment_scale_source,
    _repair_source_seed_for_cuts,
    _replace_flexible_envelope_member_bboxes,
    _select_feedback_seed,
    _solver_evidence,
    _strip_solve,
    _strip_solve_banded,
    _strip_solve_floorplan,
    place_blocks_2d,
    place_blocks_2d_with_certificate,
)
from clay.layout.place.materialized_placement import (
    reposition_pum_2d,
)
from clay.layout.place.lp_compactor import DifferenceConstraint, LpCompactionResult, compact_axis

_PLACEMENT_OWNERS = (
    _owner_placement_constants,
    _owner_placement_cuts,
    _owner_placement_diagnostics,
    _owner_constraints_spatial,
    _owner_constraints_hierarchical,
    _owner_constraints_well_guard,
    _owner_placement_geometry,
    _owner_placement_relations,
    _owner_placement_replay,
    _owner_full2d_compactor,
    _owner_placement_certificates,
    _owner_cp_model_builder,
    _owner_materialized_placement,
)


class _PlacementFacadeModule(ModuleType):
    """Keep historical instrumentation synchronized with every real global owner."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name.startswith("__"):
            return
        for owner in _PLACEMENT_OWNERS:
            if hasattr(owner, name):
                setattr(owner, name, value)


sys.modules[__name__].__class__ = _PlacementFacadeModule
