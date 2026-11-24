"""Final placement certificates and exact materialized-geometry binding."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from clay.layout.mosaic_guest_slot_projection import MosaicGuestSlotProjection, mosaic_guest_slot_satisfied
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from clay.layout.policy import aspect_policy_binding, outline_width_policy_binding
from clay.layout.intent_basis import IntentConsumptionCertificate
from clay.layout.intent_closure import ConsumerProjection
from clay.layout.place.final_placement_state import MatchingPairAxisProjection, canonical_minimum_structural_occupancy, functional_od_policy_binding, structural_occupancy_policy_binding, verify_matching_pair_projection_owners
from clay.layout.place.layered_assembly_certificate import require_materialized_footprint_certificate
from clay.layout.spatial.intent_projection import ReferenceConditionedPlacementObligation, replay_reference_conditioned_placement_obligation_inventory
from clay.layout.place.constraints_hierarchical import (
    _assert_aggregate_envelope_order_positions,
    _assert_hierarchical_macro_positions,
    _canonical_aggregate_envelope_orders,
    _canonical_hierarchical_alignment_tolerance_scales_nm,
    _canonical_hierarchical_macro_alignments,
    _canonical_hierarchical_macro_orders,
    _hierarchical_macro_alignment_parts,
    _require_hierarchical_alignment_tolerance_scales_nm,
)
from clay.layout.place.constraints_spatial import (
    FlexibleEnvelopeProjection,
    _assert_axis_alignment_positions,
    _assert_center_alignment_positions,
    _bbox_clearance_nm,
    _canonical_axis_alignment_groups,
    _canonical_center_alignment_pairs,
    _canonical_floorplan_column_classes,
    _canonical_floorplan_column_groups,
    _canonical_layered_collision_footprints,
    _canonical_mosaic_guest_slot_projections,
    _canonical_overlap_compatible_pairs,
    _canonical_rigid_compound_footprints,
    _canonical_row_bounded_cavity_projections,
    _floorplan_column_group_order_satisfied,
    _floorplan_column_order_satisfied,
    _row_bounded_cavity_satisfied,
    _seed_geometry_legal,
)
from clay.layout.place.constraints_well_guard import (
    _canonical_pair_clearances_nm,
)
from clay.layout.place.placement_constants import (
    _ASPECT_TOL,
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
from clay.layout.place.placement_cuts import (
    _PreparedSolveCut,
    _cut_satisfied_by_bboxes,
    _exact_cut_input_blocks,
)
from clay.layout.place.placement_geometry import (
    _diff_pair_axis_alignment_groups,
)
from clay.layout.place.placement_replay import (
    _require_edge_rail_reservation,
)


@dataclass(frozen=True)
class FinalPlacementCertificate:
    """Stable evidence for the exact rigid-block placement returned to callers."""

    blocks: Tuple[Tuple[str, _Bbox], ...]
    bands: Tuple[Tuple[str, int], ...]
    bbox_nm: _Bbox
    status: str
    objective_name: str
    objective: Optional[float]
    best_bound: Optional[float]
    used_fallback: bool
    fallback_kind: Optional[str]
    primary_status: Optional[str]
    primary_objective_name: Optional[str]
    primary_objective: Optional[float]
    primary_best_bound: Optional[float]
    measured_aspect: float
    declared_aspect: float
    aspect_residual: float
    declared_channel_nm: int
    declared_aspect_min: Optional[float] = None
    declared_aspect_max: Optional[float] = None
    geometry_legal: bool = True
    cuts_satisfied: bool = True
    aspect_satisfied: bool = True
    aspect_obligation: bool = True
    aspect_policy_sha256: str = ""
    resource_status: str = "UNAVAILABLE"
    resource_closure_certificate: Any = None
    applied_cut_hashes: Tuple[str, ...] = ()
    placement_causality_payloads: Tuple[Any, ...] = ()
    placement_causality_basis: Any = None
    placement_declaration_refs: Tuple[str, ...] = ()
    materialized_block_members: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    materialized_block_bboxes: Tuple[Tuple[str, _Bbox], ...] = ()
    materialized_block_source_members: Tuple[
        Tuple[str, Tuple[str, ...]], ...
    ] = ()
    materialized_source_member_bboxes: Tuple[Tuple[str, _Bbox], ...] = ()
    axis_alignment_groups: Tuple[Tuple[Tuple[str, int], ...], ...] = ()
    center_alignment_pairs: Tuple[Tuple[str, str, str], ...] = ()
    aggregate_envelope_orders: Tuple[_AggregateEnvelopeOrder, ...] = ()
    hierarchical_macro_orders: Tuple[_HierarchicalMacroOrder, ...] = ()
    hierarchical_macro_alignments: Tuple[_HierarchicalMacroAlignment, ...] = ()
    reference_conditioned_placement_obligations: Tuple[
        ReferenceConditionedPlacementObligation, ...
    ] = ()
    row_bounded_cavity_projections: Tuple[_RowBoundedCavityProjection, ...] = ()
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = ()
    edge_rail_reservation: Any = None
    hierarchical_alignment_tolerance_scales_nm: (
        _HierarchicalAlignmentToleranceScalesNm
    ) = ()
    declared_minimum_structural_occupancy: Optional[float] = None
    structural_occupancy_policy_sha256: str = ""
    declared_minimum_functional_od_occupancy: Optional[float] = None
    functional_od_area_nm2: int = 0
    functional_od_layer_keys: Tuple[Tuple[int, int], ...] = ()
    functional_od_policy_sha256: str = ""
    declared_outline_width_nm: Optional[int] = None
    outline_width_satisfied: bool = True
    outline_width_policy_sha256: str = ""
    floorplan_column_classes: Tuple[Tuple[str, int, int, int], ...] = ()
    floorplan_column_groups: Tuple[_FloorplanColumnGroup, ...] = ()
    intent_consumption_certificates: Tuple[
        IntentConsumptionCertificate, ...
    ] = ()
    fact_projections: Tuple[ConsumerProjection, ...] = ()
    matching_pair_axis_projections: Tuple[MatchingPairAxisProjection, ...] = ()
    native_cross_row_final_proof: Any = None
    native_same_row_final_proof: Any = None
    flexible_envelope_projections: Tuple[FlexibleEnvelopeProjection, ...] = ()
    group_route_placement_certificate: Any = None
    declared_bottom_edge_keepout_nm: int = 0
    bottom_edge_keepout_carrier_blocks: Tuple[str, ...] = ()
    bottom_edge_keepout_clear_blocks: Tuple[str, ...] = ()
    declared_gap_nm: int = 0
    full2d_compaction_graph: Any = None
    full2d_solver_evidence: Tuple[Mapping[str, object], ...] = ()
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = ()
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = ()
    layered_placement_assemblies: tuple = ()
    layered_placement_replay: Any = None
    layered_placement_projection: Any = None
    collision_exclusion_pdk_id: str = ""
    collision_excluded_layers: Tuple[Tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.native_same_row_final_proof is not None:
            from clay.layout.place.native_same_row_matching import NativeSameRowFinalProof
            if type(self.native_same_row_final_proof) is not NativeSameRowFinalProof:
                raise TypeError("final native same-row proof is foreign")
            self.native_same_row_final_proof.__post_init__()
        if self.native_cross_row_final_proof is not None:
            from clay.layout.place.cross_row_matching import NativeCrossRowFinalProof
            if type(self.native_cross_row_final_proof) is not NativeCrossRowFinalProof:
                raise TypeError("final native cross-row proof is foreign")
            self.native_cross_row_final_proof.__post_init__()
        if type(self.collision_exclusion_pdk_id) is not str or type(self.collision_excluded_layers) is not tuple:
            raise TypeError('collision exclusion authority must use exact PDK and layer inventory')
        if bool(self.collision_exclusion_pdk_id) != bool(self.collision_excluded_layers):
            raise ValueError('collision exclusion PDK and layer authority must be paired')
        if self.collision_excluded_layers:
            from clay.layout.tech import LayoutTech
            key = LayoutTech.resolve(self.collision_exclusion_pdk_id).assembly_boundary_key()
            if self.collision_excluded_layers != (key,):
                raise ValueError('collision exclusion differs from PDK assembly boundary authority')
            graph = self.full2d_compaction_graph
            if graph is not None and any(layer in self.collision_excluded_layers
                                         for _node, layer in graph.node_layers):
                raise ValueError('collision graph contains its excluded assembly boundary')
        from clay.layout.place.layered_assembly_certificate import layered_assembly_certificate_payload

        layered_assembly_certificate_payload(self)
        canonical_reference_obligations = (
            replay_reference_conditioned_placement_obligation_inventory(
                self.reference_conditioned_placement_obligations
            )
        )
        if (
            canonical_reference_obligations
            != self.reference_conditioned_placement_obligations
        ):
            raise ValueError(
                "reference-conditioned placement obligations are not canonical"
            )
        canonical_alignment_scales = (
            _require_hierarchical_alignment_tolerance_scales_nm(
                self.hierarchical_alignment_tolerance_scales_nm,
                alignments=self.hierarchical_macro_alignments,
            )
        )
        if (
            canonical_alignment_scales
            != self.hierarchical_alignment_tolerance_scales_nm
        ):
            raise ValueError("hierarchical alignment tolerance scales are not canonical")
        require_materialized_footprint_certificate(self)
        if type(self.fact_projections) is not tuple or any(
            type(row) is not ConsumerProjection for row in self.fact_projections
        ):
            raise TypeError("placement fact projections must be an exact typed tuple")
        if self.fact_projections:
            consumers = tuple(row.consumer for row in self.fact_projections)
            if consumers != ("placement_initial", "placement_replay"):
                raise ValueError(
                    "placement fact projections require initial then replay order"
                )
            if (
                self.fact_projections[0].fact_ids
                != self.fact_projections[1].fact_ids
                or self.fact_projections[0].normalized_projection_sha256
                != self.fact_projections[1].normalized_projection_sha256
            ):
                raise ValueError("placement fact projections violate replay parity")
        if type(self.intent_consumption_certificates) is not tuple or any(
            type(row) is not IntentConsumptionCertificate
            for row in self.intent_consumption_certificates
        ):
            raise TypeError(
                "intent consumption certificates must be an exact typed tuple"
            )
        consumption_stages = tuple(
            row.stage for row in self.intent_consumption_certificates
        )
        if (
            consumption_stages != tuple(sorted(consumption_stages))
            or len(set(consumption_stages)) != len(consumption_stages)
        ):
            raise ValueError(
                "intent consumption certificates must have sorted unique stages"
            )
        if self.intent_consumption_certificates:
            if len(
                {row.basis_sha256 for row in self.intent_consumption_certificates}
            ) != 1 or len(
                {row.consumed_atom_ids for row in self.intent_consumption_certificates}
            ) != 1:
                raise ValueError(
                    "intent consumption certificates disagree on selected atoms"
                )
            if len(
                {
                    row.normalized_projection_sha256
                    for row in self.intent_consumption_certificates
                }
            ) != 1:
                raise ValueError(
                    "intent consumption certificates disagree on projection parity"
                )
        raw_source_members = self.materialized_block_source_members
        if type(raw_source_members) is not tuple:
            raise TypeError(
                "materialized block source ownership must be an exact tuple"
            )
        if raw_source_members:
            physical_names = {
                name for name, _bbox in self.materialized_block_bboxes
            }
            source_rows = []
            all_sources = []
            for row in raw_source_members:
                if type(row) is not tuple or len(row) != 2:
                    raise TypeError(
                        "materialized block source ownership row must be an exact pair"
                    )
                owner, raw_sources = row
                if (
                    type(owner) is not str
                    or not owner
                    or type(raw_sources) is not tuple
                    or not raw_sources
                    or any(type(source) is not str or not source for source in raw_sources)
                ):
                    raise TypeError(
                        "materialized block source ownership row is malformed"
                    )
                sources = tuple(raw_sources)
                if sources != tuple(sorted(sources)) or len(set(sources)) != len(sources):
                    raise ValueError(
                        "materialized block source members must be canonical and unique"
                    )
                source_rows.append((owner, sources))
                all_sources.extend(sources)
            if (
                tuple(source_rows) != tuple(sorted(source_rows))
                or len({owner for owner, _sources in source_rows}) != len(source_rows)
                or {owner for owner, _sources in source_rows} != physical_names
            ):
                raise ValueError(
                    "materialized block source ownership must cover exact physical blocks"
                )
            if len(all_sources) != len(set(all_sources)):
                raise ValueError(
                    "materialized block source ownership repeats a source member"
                )
        raw_source_bboxes = self.materialized_source_member_bboxes
        if type(raw_source_bboxes) is not tuple:
            raise TypeError(
                "materialized source member bboxes must be an exact tuple"
            )
        if raw_source_bboxes:
            if not raw_source_members:
                raise ValueError(
                    "materialized source member bboxes require source ownership"
                )
            source_owner = {
                source: owner
                for owner, sources in raw_source_members
                for source in sources
            }
            physical_bboxes = dict(self.materialized_block_bboxes)
            source_bbox_rows = []
            for row in raw_source_bboxes:
                if type(row) is not tuple or len(row) != 2:
                    raise TypeError(
                        "materialized source member bbox row must be an exact pair"
                    )
                source, bbox = row
                if (
                    type(source) is not str
                    or not source
                    or type(bbox) is not tuple
                    or len(bbox) != 4
                    or any(type(value) is not int for value in bbox)
                    or bbox[2] <= bbox[0]
                    or bbox[3] <= bbox[1]
                ):
                    raise TypeError(
                        "materialized source member bbox row is malformed"
                    )
                source_bbox_rows.append((source, bbox))
            if (
                tuple(source_bbox_rows) != tuple(sorted(source_bbox_rows))
                or len({source for source, _bbox in source_bbox_rows})
                != len(source_bbox_rows)
            ):
                raise ValueError(
                    "materialized source member bboxes must be canonical and unique"
                )
            bbox_sources = {source for source, _bbox in source_bbox_rows}
            if not bbox_sources <= set(source_owner):
                raise ValueError(
                    "materialized source member bboxes name foreign source ownership"
                )
            for owner, sources in raw_source_members:
                covered = set(sources).intersection(bbox_sources)
                if covered and covered != set(sources):
                    raise ValueError(
                        "materialized source member bboxes must cover exact source "
                        f"ownership per physical owner {owner!r}"
                    )
            for source, bbox in source_bbox_rows:
                owner_bbox = physical_bboxes[source_owner[source]]
                if not (
                    owner_bbox[0] <= bbox[0] < bbox[2] <= owner_bbox[2]
                    and owner_bbox[1] <= bbox[1] < bbox[3] <= owner_bbox[3]
                ):
                    raise ValueError(
                        "materialized source member bbox must stay inside its physical owner"
                    )
        if self.group_route_placement_certificate is not None:
            from clay.layout.route.group_ports import (
                GroupRoutePlacementCertificate,
            )

            if (
                type(self.group_route_placement_certificate)
                is not GroupRoutePlacementCertificate
            ):
                raise TypeError(
                    "group-route placement certificate has a foreign type"
                )
            if not raw_source_members:
                raise ValueError(
                    "group-route placement lacks materializer source ownership"
                )
        if type(self.flexible_envelope_projections) is not tuple:
            raise TypeError("flexible envelope projections must be one exact tuple")
        if any(
            type(row) is not FlexibleEnvelopeProjection
            for row in self.flexible_envelope_projections
        ):
            raise TypeError("flexible envelope projections contain a foreign row")
        if (
            self.flexible_envelope_projections
            != tuple(sorted(self.flexible_envelope_projections))
            or len(
                {
                    row.constraint.group
                    for row in self.flexible_envelope_projections
                }
            )
            != len(self.flexible_envelope_projections)
        ):
            raise ValueError("flexible envelope projections must be canonical and unique")
        blocks_by_name = dict(self.blocks)
        flexible_members: set[str] = set()
        for projection in self.flexible_envelope_projections:
            constraint = projection.constraint
            if any(member not in blocks_by_name for member in constraint.members):
                raise ValueError("flexible envelope projection lacks a member block")
            if flexible_members.intersection(constraint.members):
                raise ValueError("flexible envelope projections repeat member ownership")
            flexible_members.update(constraint.members)
            physical_offsets = dict(constraint.member_physical_offsets_nm)
            physical_member_boxes = tuple(
                (
                    blocks_by_name[name][0] + physical_offsets[name][0],
                    blocks_by_name[name][1] + physical_offsets[name][1],
                    blocks_by_name[name][0] + physical_offsets[name][2],
                    blocks_by_name[name][1] + physical_offsets[name][3],
                )
                for name in constraint.members
            )
            expected_bbox = (
                min(box[0] for box in physical_member_boxes),
                min(box[1] for box in physical_member_boxes),
                max(box[2] for box in physical_member_boxes),
                max(box[3] for box in physical_member_boxes),
            )
            if projection.bbox_nm != expected_bbox:
                raise ValueError("flexible envelope projection does not replay its members")
            foreign_clearances = {
                (member, foreign): clearance
                for member, foreign, clearance in projection.foreign_clearances_nm
            }
            expected_foreign_rows = {
                (member, foreign)
                for member in constraint.members
                for foreign in set(blocks_by_name) - set(constraint.members)
            }
            if set(foreign_clearances) != expected_foreign_rows:
                raise ValueError(
                    "flexible envelope projection lacks exact foreign ownership"
                )
            if constraint.foreign_exclusion_mode == "envelope":
                for name, foreign_bbox in self.blocks:
                    if name in constraint.members:
                        continue
                    required = max(
                        foreign_clearances[(member, name)]
                        for member in constraint.members
                    )
                    if _bbox_clearance_nm(projection.bbox_nm, foreign_bbox) < required:
                        raise ValueError(
                            "flexible envelope projection contains foreign geometry"
                        )
            else:
                for member in constraint.members:
                    member_bbox = blocks_by_name[member]
                    offset = physical_offsets[member]
                    physical_member_bbox = (
                        member_bbox[0] + offset[0],
                        member_bbox[1] + offset[1],
                        member_bbox[0] + offset[2],
                        member_bbox[1] + offset[3],
                    )
                    for name, foreign_bbox in self.blocks:
                        if name in constraint.members:
                            continue
                        if (
                            _bbox_clearance_nm(physical_member_bbox, foreign_bbox)
                            < foreign_clearances[(member, name)]
                        ):
                            raise ValueError(
                                "flexible envelope projection overlaps foreign geometry"
                            )
        if type(self.declared_bottom_edge_keepout_nm) is not int:
            raise TypeError(
                "declared bottom-edge keepout must be an exact integer nm value"
            )
        if self.declared_bottom_edge_keepout_nm < 0:
            raise ValueError("declared bottom-edge keepout must be non-negative")
        for label, raw_names in (
            ("carrier", self.bottom_edge_keepout_carrier_blocks),
            ("clear", self.bottom_edge_keepout_clear_blocks),
        ):
            if type(raw_names) is not tuple:
                raise TypeError(
                    f"bottom-edge keepout {label} blocks must be one exact tuple"
                )
            if any(type(name) is not str or not name for name in raw_names):
                raise TypeError(
                    f"bottom-edge keepout {label} blocks must be exact named strings"
                )
            if raw_names != tuple(sorted(raw_names)) or len(set(raw_names)) != len(
                raw_names
            ):
                raise ValueError(
                    f"bottom-edge keepout {label} blocks must be canonical and unique"
                )
            if any(name not in blocks_by_name for name in raw_names):
                raise ValueError(
                    f"bottom-edge keepout {label} blocks name a foreign master"
                )
        carrier_names = set(self.bottom_edge_keepout_carrier_blocks)
        clear_names = set(self.bottom_edge_keepout_clear_blocks)
        if carrier_names.intersection(clear_names):
            raise ValueError(
                "bottom-edge keepout carrier and clear ownership must be disjoint"
            )
        allowed_in_keepout = carrier_names | clear_names
        if self.declared_bottom_edge_keepout_nm == 0:
            if allowed_in_keepout:
                raise ValueError(
                    "bottom-edge keepout ownership requires a non-zero declaration"
                )
        else:
            keepout_nm = self.declared_bottom_edge_keepout_nm
            if any(
                bbox[1] < keepout_nm
                for name, bbox in self.blocks
                if name not in allowed_in_keepout
            ):
                raise ValueError(
                    "bottom-edge keepout does not exclude every blocking master"
                )
            if any(
                projection.bbox_nm[1] < keepout_nm
                and not set(projection.constraint.members).issubset(
                    allowed_in_keepout
                )
                for projection in self.flexible_envelope_projections
            ):
                raise ValueError(
                    "bottom-edge keepout does not exclude a flexible blocker"
                )
            measured_bboxes = tuple(bbox for _name, bbox in self.blocks) + tuple(
                projection.bbox_nm
                for projection in self.flexible_envelope_projections
            )
            if not measured_bboxes:
                raise ValueError(
                    "bottom-edge keepout cannot bind an empty placement"
                )
            right_rail_margin_nm = int(
                getattr(self.edge_rail_reservation, "right_margin_nm", 0) or 0
            )
            top_rail_margin_nm = int(
                getattr(self.edge_rail_reservation, "top_margin_nm", 0) or 0
            )
            expected_bbox = (
                min(bbox[0] for bbox in measured_bboxes),
                0,
                max(bbox[2] for bbox in measured_bboxes) + right_rail_margin_nm,
                max(bbox[3] for bbox in measured_bboxes) + top_rail_margin_nm,
            )
            if self.bbox_nm != expected_bbox:
                raise ValueError(
                    "bottom-edge rail reserve does not replay the placement bbox"
                )
        if type(self.matching_pair_axis_projections) is not tuple:
            raise TypeError("matching-axis projection inventory must be an exact tuple")
        if any(
            type(row) is not MatchingPairAxisProjection
            for row in self.matching_pair_axis_projections
        ):
            raise TypeError("matching-axis projection inventory contains a foreign row")
        if (
            self.matching_pair_axis_projections
            != tuple(sorted(self.matching_pair_axis_projections))
            or len({row.pair for row in self.matching_pair_axis_projections})
            != len(self.matching_pair_axis_projections)
        ):
            raise ValueError("matching-axis projections must be canonical and unique")
        if self.matching_pair_axis_projections:
            raw_materialized_members = self.materialized_block_members
            if type(raw_materialized_members) is not tuple or any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not str
                or not row[0]
                or type(row[1]) is not tuple
                or not row[1]
                or any(type(member) is not str or not member for member in row[1])
                for row in raw_materialized_members
            ):
                raise TypeError("materialized master partition is malformed")
            materialized_names = {
                master for master, _members in raw_materialized_members
            }
            if materialized_names != set(blocks_by_name):
                raise ValueError(
                    "materialized master partition differs from solver blocks"
                )
            if any(
                row.source_master not in materialized_names
                for row in self.matching_pair_axis_projections
            ):
                raise ValueError(
                    "matching-axis projection source master lacks materialized ownership"
                )
        verify_matching_pair_projection_owners(self)
        canonical_columns = _canonical_floorplan_column_classes(
            self.floorplan_column_classes,
            block_ids=dict(self.blocks),
        )
        if not _floorplan_column_order_satisfied(self.blocks, canonical_columns):
            raise ValueError(
                "floorplan column classes do not match final block centroid order"
            )
        canonical_groups = _canonical_floorplan_column_groups(
            self.floorplan_column_groups,
            block_ids=dict(self.blocks),
        )
        if not _floorplan_column_group_order_satisfied(self.blocks, canonical_groups):
            raise ValueError(
                "floorplan column groups do not match functional-group centroid order"
            )
        _occupancy, expected_occupancy_sha256 = structural_occupancy_policy_binding(
            self.declared_minimum_structural_occupancy
        )
        if not self.structural_occupancy_policy_sha256:
            object.__setattr__(
                self,
                "structural_occupancy_policy_sha256",
                expected_occupancy_sha256,
            )
        elif self.structural_occupancy_policy_sha256 != expected_occupancy_sha256:
            raise ValueError(
                "structural occupancy policy digest does not match certificate fields"
            )
        _od, _area, _layers, expected_od_sha256 = functional_od_policy_binding(
            self.declared_minimum_functional_od_occupancy,
            self.functional_od_area_nm2,
            self.functional_od_layer_keys,
        )
        if not self.functional_od_policy_sha256:
            object.__setattr__(
                self, "functional_od_policy_sha256", expected_od_sha256
            )
        elif self.functional_od_policy_sha256 != expected_od_sha256:
            raise ValueError("functional OD policy digest does not match certificate fields")
        declared_width, expected_width_sha256 = outline_width_policy_binding(
            self.declared_outline_width_nm
        )
        if type(self.outline_width_satisfied) is not bool:
            raise TypeError("outline_width_satisfied must be an exact bool")
        measured_width = self.bbox_nm[2] - self.bbox_nm[0]
        expected_width_satisfied = (
            declared_width is None or measured_width == declared_width
        )
        if self.outline_width_satisfied is not expected_width_satisfied:
            raise ValueError("outline width verdict does not match certificate bbox")
        if not self.outline_width_policy_sha256:
            object.__setattr__(
                self,
                "outline_width_policy_sha256",
                expected_width_sha256,
            )
        elif self.outline_width_policy_sha256 != expected_width_sha256:
            raise ValueError("outline width policy digest does not match certificate field")
        if type(self.aspect_obligation) is not bool:
            raise TypeError("aspect_obligation must be an exact bool")
        _declared, expected_sha256 = aspect_policy_binding(
            self.declared_aspect if self.aspect_obligation else None
        )
        if not self.aspect_policy_sha256:
            object.__setattr__(self, "aspect_policy_sha256", expected_sha256)
        elif self.aspect_policy_sha256 != expected_sha256:
            raise ValueError("aspect policy digest does not match certificate fields")

    @property
    def placement_state_digest(self) -> str:
        """Stable identity of the exact master geometry, excluding audit metadata."""

        return final_placement_state_digest(self)

def _certificate_objective_is_optimal(certificate: object, prefix: str) -> bool:
    status = getattr(certificate, f"{prefix}status", None)
    value = getattr(certificate, f"{prefix}objective", None)
    bound = getattr(certificate, f"{prefix}best_bound", None)
    if type(status) is not str or status != "OPTIMAL":
        return False
    if (
        type(value) not in {int, float}
        or type(bound) not in {int, float}
        or not math.isfinite(float(value))
        or not math.isfinite(float(bound))
    ):
        return False
    return float(value) == float(bound)

def placement_certificate_proves_area_optimality(certificate: object) -> bool:
    """Return whether the certificate proves its area optimum, if declared.

    This quality claim is deliberately independent from physical authority.
    A deterministic FEASIBLE placement can be materialized and checked exactly
    without pretending that an NP-hard global area optimum was proven.
    """

    objective_name = getattr(certificate, "objective_name", None)
    primary_objective_name = getattr(certificate, "primary_objective_name", None)
    if primary_objective_name == "area_nm2":
        return _certificate_objective_is_optimal(certificate, "primary_")
    if objective_name == "area_nm2":
        return _certificate_objective_is_optimal(certificate, "")
    return False

def placement_certificate_objective_evidence(
    certificate: object,
) -> Dict[str, object]:
    """Expose coordinate-free solver quality separately from physical authority."""

    objective_name = getattr(certificate, "objective_name", None)
    primary_objective_name = getattr(certificate, "primary_objective_name", None)
    area_declared = (
        objective_name == "area_nm2" or primary_objective_name == "area_nm2"
    )
    return {
        "status": getattr(certificate, "status", None),
        "objective_name": objective_name,
        "objective": getattr(certificate, "objective", None),
        "best_bound": getattr(certificate, "best_bound", None),
        "primary_status": getattr(certificate, "primary_status", None),
        "primary_objective_name": primary_objective_name,
        "primary_objective": getattr(certificate, "primary_objective", None),
        "primary_best_bound": getattr(certificate, "primary_best_bound", None),
        "used_fallback": getattr(certificate, "used_fallback", None),
        "fallback_kind": getattr(certificate, "fallback_kind", None),
        "area_optimality_status": (
            "proven"
            if area_declared
            and placement_certificate_proves_area_optimality(certificate)
            else "unproven_feasible"
            if area_declared
            else "not_applicable"
        ),
        "physical_authority": (
            "signable" if placement_certificate_is_signable(certificate) else "invalid"
        ),
    }

def placement_certificate_is_signable(certificate: object) -> bool:
    """Return whether one placement certificate may enter canonical routing.

    This is the shared placement-to-route contract.  Search, feedback replay,
    and final flow must agree on it: a geometrically usable fallback is still
    not a solver-certified implementation of the declared aspect and cuts.
    """

    if certificate is None or type(getattr(certificate, "aspect_obligation", None)) is not bool:
        return False
    structural_policy_present = any(
        hasattr(certificate, field)
        for field in (
            "declared_minimum_structural_occupancy",
            "structural_occupancy_policy_sha256",
        )
    )
    od_policy_present = any(
        hasattr(certificate, field)
        for field in (
            "declared_minimum_functional_od_occupancy",
            "functional_od_area_nm2",
            "functional_od_layer_keys",
            "functional_od_policy_sha256",
        )
    )
    outline_policy_present = any(
        hasattr(certificate, field)
        for field in (
            "declared_outline_width_nm",
            "outline_width_satisfied",
            "outline_width_policy_sha256",
        )
    )
    floorplan_columns_satisfied = True
    floorplan_groups_satisfied = True
    try:
        _declared, expected_aspect_sha256 = aspect_policy_binding(
            getattr(certificate, "declared_aspect", None)
            if certificate.aspect_obligation
            else None
        )
        expected_occupancy_sha256 = None
        if structural_policy_present:
            _occupancy, expected_occupancy_sha256 = (
                structural_occupancy_policy_binding(
                    getattr(
                        certificate,
                        "declared_minimum_structural_occupancy",
                        None,
                    )
                )
            )
        minimum_od = None
        od_area = 0
        expected_od_sha256 = None
        if od_policy_present:
            minimum_od, od_area, _od_layers, expected_od_sha256 = (
                functional_od_policy_binding(
                    getattr(
                        certificate,
                        "declared_minimum_functional_od_occupancy",
                        None,
                    ),
                    getattr(certificate, "functional_od_area_nm2", 0),
                    getattr(certificate, "functional_od_layer_keys", ()),
                )
            )
        expected_outline_width_sha256 = None
        if outline_policy_present:
            _width, expected_outline_width_sha256 = outline_width_policy_binding(
                getattr(certificate, "declared_outline_width_nm", None)
            )
        if hasattr(certificate, "floorplan_column_classes"):
            certificate_blocks = tuple(getattr(certificate, "blocks", ()))
            canonical_columns = _canonical_floorplan_column_classes(
                getattr(certificate, "floorplan_column_classes", ()),
                block_ids=dict(certificate_blocks),
            )
            floorplan_columns_satisfied = _floorplan_column_order_satisfied(
                certificate_blocks,
                canonical_columns,
            )
        if hasattr(certificate, "floorplan_column_groups"):
            certificate_blocks = tuple(getattr(certificate, "blocks", ()))
            canonical_groups = _canonical_floorplan_column_groups(
                getattr(certificate, "floorplan_column_groups", ()),
                block_ids=dict(certificate_blocks),
            )
            floorplan_groups_satisfied = _floorplan_column_group_order_satisfied(
                certificate_blocks,
                canonical_groups,
            )
    except (KeyError, TypeError, ValueError):
        return False
    objective_name = getattr(certificate, "objective_name", None)
    primary_objective_name = getattr(certificate, "primary_objective_name", None)
    if type(objective_name) is not str or (
        primary_objective_name is not None
        and type(primary_objective_name) is not str
    ):
        return False
    if objective_name == "soft_position_cost" and primary_objective_name not in {
        "area_nm2",
        "height_nm",
    }:
        return False

    bbox = tuple(getattr(certificate, "bbox_nm", ()))
    od_satisfied = minimum_od is None
    if minimum_od is not None and len(bbox) == 4:
        envelope_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        od_satisfied = bool(
            envelope_area > 0
            and Decimal(od_area) >= Decimal(str(minimum_od)) * envelope_area
        )
    return bool(
        getattr(certificate, "aspect_policy_sha256", None)
        == expected_aspect_sha256
        and (
            not structural_policy_present
            or getattr(certificate, "structural_occupancy_policy_sha256", None)
            == expected_occupancy_sha256
        )
        and (
            not od_policy_present
            or getattr(certificate, "functional_od_policy_sha256", None)
            == expected_od_sha256
        )
        and (
            not outline_policy_present
            or (
                getattr(certificate, "outline_width_policy_sha256", None)
                == expected_outline_width_sha256
                and getattr(certificate, "outline_width_satisfied", None) is True
            )
        )
        and od_satisfied
        and floorplan_columns_satisfied
        and floorplan_groups_satisfied
        and not bool(getattr(certificate, "used_fallback", False))
        and str(getattr(certificate, "status", "") or "") in {"OPTIMAL", "FEASIBLE"}
        and getattr(certificate, "geometry_legal", None) is True
        and getattr(certificate, "cuts_satisfied", None) is True
        and getattr(certificate, "aspect_satisfied", None) is True
    )

def final_placement_state_digest(certificate: FinalPlacementCertificate) -> str:
    """Delegate to the cycle-free canonical final-placement identity."""

    if not isinstance(certificate, FinalPlacementCertificate):
        raise TypeError("certificate must be FinalPlacementCertificate")
    from clay.layout.place.final_placement_state import (
        final_placement_state_digest as canonical_digest,
    )

    return canonical_digest(certificate)

class PlacementSolveCutResidualError(RuntimeError):
    """Typed stop when the positions-only API cannot hide a cut residual."""

    def __init__(self, certificate: FinalPlacementCertificate) -> None:
        self.certificate = certificate
        self.status = certificate.status
        super().__init__(f"placement solve-cut residual: {self.status}")


@dataclass(frozen=True)
class _SolverEvidence:
    status: str
    objective_name: str
    objective: Optional[float]
    best_bound: Optional[float]
    used_fallback: bool = False
    fallback_kind: Optional[str] = None
    primary_status: Optional[str] = None
    primary_objective_name: Optional[str] = None
    primary_objective: Optional[float] = None
    primary_best_bound: Optional[float] = None
    flexible_envelope_projections: Tuple[FlexibleEnvelopeProjection, ...] = ()
    full2d_compaction_graph: Any = None
    full2d_solver_evidence: Tuple[Mapping[str, object], ...] = ()

    def after_area_frontier_compaction(self, area_nm2: float) -> _SolverEvidence:
        """Retain native primary evidence when optional geometry improves area."""
        if self.primary_status is None:
            return replace(self, status="FEASIBLE", objective_name="area_nm2",
                           objective=area_nm2, used_fallback=False, fallback_kind=None)
        return _SolverEvidence(
            status="FEASIBLE", objective_name="deterministic_area_frontier_compaction",
            objective=area_nm2, best_bound=None, primary_status="FEASIBLE",
            primary_objective_name="area_nm2", primary_objective=area_nm2,
            primary_best_bound=self.primary_best_bound,
        )


@dataclass(frozen=True)
class PlaceBlocks2DResult:
    positions: Mapping[str, Tuple[int, int]]
    certificate: FinalPlacementCertificate

@dataclass(frozen=True)
class RepositionPum2DResult:
    """Canonical compact geometry plus the final master certificate that made it."""

    placement_um: Dict[str, Tuple[float, float]]
    geometry: Dict[str, Any]
    final_placement_certificate: FinalPlacementCertificate
    member_ring_pad_top_nm: Tuple[Tuple[str, int], ...] = ()
    member_ring_pad_bottom_nm: Tuple[Tuple[str, int], ...] = ()
    intent_consumption_certificates: Tuple[
        IntentConsumptionCertificate, ...
    ] = ()
    fact_projections: Tuple[ConsumerProjection, ...] = ()

    def __post_init__(self) -> None:
        certificate_fact_projections = getattr(
            self.final_placement_certificate,
            "fact_projections",
            (),
        )
        if type(certificate_fact_projections) is not tuple or any(
            type(row) is not ConsumerProjection
            for row in certificate_fact_projections
        ):
            raise TypeError(
                "final placement fact projections must be an exact typed tuple"
            )
        if type(self.fact_projections) is not tuple or any(
            type(row) is not ConsumerProjection for row in self.fact_projections
        ):
            raise TypeError("reposition fact projections must be an exact typed tuple")
        if self.fact_projections != certificate_fact_projections:
            raise ValueError(
                "reposition result and final placement disagree on fact projections"
            )
        certificate_intent_consumption = getattr(
            self.final_placement_certificate,
            "intent_consumption_certificates",
            (),
        )
        if type(certificate_intent_consumption) is not tuple or any(
            type(row) is not IntentConsumptionCertificate
            for row in certificate_intent_consumption
        ):
            raise TypeError(
                "final placement intent consumption certificates must be an "
                "exact typed tuple"
            )
        if type(self.intent_consumption_certificates) is not tuple or any(
            type(row) is not IntentConsumptionCertificate
            for row in self.intent_consumption_certificates
        ):
            raise TypeError(
                "reposition intent consumption certificates must be an exact tuple"
            )
        if (
            self.intent_consumption_certificates
            != certificate_intent_consumption
        ):
            raise ValueError(
                "reposition result and final placement disagree on intent consumption"
            )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Preserve the historical two-value unpacking used by analysis probes."""

        yield self.placement_um
        yield self.geometry

def _bind_certificate_to_materialized_blocks(
    certificate: FinalPlacementCertificate,
    *,
    materialized_blocks: Sequence[Tuple[str, _Bbox]],
    master_members: Mapping[str, Sequence[str]],
    materialized_block_source_members: Any = (),
    materialized_source_member_bboxes: Any = (),
    materialized_collision_footprints: Any = (),
    layered_collision_footprints: Any = (),
    overlap_compatible_pairs: Any = (),
    gap_nm: int = 0,
    pair_clearances_nm: Any = None,
    pair_clearance_scope: str = "materialized",
    solve_cuts: Any = (),
    placement_declaration_refs: Any = (),
    aspect_obligation: bool = True,
    layered_placement_assemblies: tuple = (),
    layered_placement_replay: Any = None,
    layered_placement_projection: Any = None,
) -> FinalPlacementCertificate:
    """Rebind a solver master to the exact post-materialization block inventory."""

    if type(aspect_obligation) is not bool:
        raise TypeError("aspect_obligation must be an exact bool")
    if type(pair_clearance_scope) is not str or pair_clearance_scope not in ("materialized", "solver"):
        raise ValueError("pair-clearance scope must explicitly name materialized or solver owners")
    if certificate.aspect_obligation is not aspect_obligation:
        raise ValueError("materialized aspect obligation differs from solver certificate")
    if not placement_certificate_is_signable(certificate):
        evidence = ", ".join(
            f"{field}={getattr(certificate, field, None)!r}"
            for field in (
                "status",
                "objective_name",
                "objective",
                "best_bound",
                "primary_status",
                "primary_objective_name",
                "primary_objective",
                "primary_best_bound",
                "used_fallback",
                "fallback_kind",
                "geometry_legal",
                "cuts_satisfied",
                "declared_aspect",
                "measured_aspect",
                "bbox_nm",
                "aspect_satisfied",
            )
        )
        raise ValueError(
            "solver certificate is not authoritative for materialization: "
            + evidence
        )
    blocks = dict(_exact_cut_input_blocks(materialized_blocks))
    collision_footprints = dict(_canonical_rigid_compound_footprints(
        materialized_collision_footprints,
        dims={
            name: (bbox[2] - bbox[0], bbox[3] - bbox[1])
            for name, bbox in blocks.items()
        },
    ))
    physical_collision_boxes = {
        name: tuple(
            (
                bbox[0] + local[0],
                bbox[1] + local[1],
                bbox[0] + local[2],
                bbox[1] + local[3],
            )
            for local in collision_footprints.get(
                name,
                ((0, 0, bbox[2] - bbox[0], bbox[3] - bbox[1]),),
            )
        )
        for name, bbox in blocks.items()
    }
    if type(materialized_block_source_members) is not tuple:
        raise TypeError("materialized block source ownership must be an exact tuple")
    if materialized_block_source_members:
        source_owner_names = []
        all_source_members = []
        for row in materialized_block_source_members:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError(
                    "materialized block source ownership row must be an exact pair"
                )
            owner, sources = row
            if (
                type(owner) is not str
                or not owner
                or type(sources) is not tuple
                or not sources
                or any(type(source) is not str or not source for source in sources)
            ):
                raise TypeError("materialized block source ownership is malformed")
            if sources != tuple(sorted(sources)) or len(set(sources)) != len(sources):
                raise ValueError(
                    "materialized block source ownership must be canonical and unique"
                )
            source_owner_names.append(owner)
            all_source_members.extend(sources)
        if (
            materialized_block_source_members
            != tuple(sorted(materialized_block_source_members))
            or len(source_owner_names) != len(set(source_owner_names))
            or set(source_owner_names) != set(blocks)
        ):
            raise ValueError(
                "materialized block source ownership must cover exact physical blocks"
            )
        if len(all_source_members) != len(set(all_source_members)):
            raise ValueError(
                "materialized block source ownership repeats a source member"
            )
    if type(materialized_source_member_bboxes) is not tuple:
        raise TypeError("materialized source member bboxes must be an exact tuple")
    if materialized_source_member_bboxes:
        if not materialized_block_source_members:
            raise ValueError(
                "materialized source member bboxes require source ownership"
            )
        source_owner = {
            source: owner
            for owner, sources in materialized_block_source_members
            for source in sources
        }
        source_bbox_rows = []
        for row in materialized_source_member_bboxes:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError(
                    "materialized source member bbox row must be an exact pair"
                )
            source, bbox = row
            if (
                type(source) is not str
                or not source
                or type(bbox) is not tuple
                or len(bbox) != 4
                or any(type(value) is not int for value in bbox)
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise TypeError("materialized source member bbox row is malformed")
            source_bbox_rows.append((source, bbox))
        if (
            tuple(source_bbox_rows) != tuple(sorted(source_bbox_rows))
            or len({source for source, _bbox in source_bbox_rows})
            != len(source_bbox_rows)
        ):
            raise ValueError(
                "materialized source member bboxes must be canonical and unique"
            )
        bbox_sources = {source for source, _bbox in source_bbox_rows}
        if not bbox_sources <= set(source_owner):
            raise ValueError(
                "materialized source member bboxes name foreign source ownership"
            )
        for owner, sources in materialized_block_source_members:
            covered = set(sources).intersection(bbox_sources)
            if covered and covered != set(sources):
                raise ValueError(
                    "materialized source member bboxes must cover exact source "
                    f"ownership per physical owner {owner!r}"
                )
        for source, bbox in source_bbox_rows:
            owner_bbox = blocks[source_owner[source]]
            if not (
                owner_bbox[0] <= bbox[0] < bbox[2] <= owner_bbox[2]
                and owner_bbox[1] <= bbox[1] < bbox[3] <= owner_bbox[3]
            ):
                raise ValueError(
                    "materialized source member bbox must stay inside its physical owner"
                )
    if type(gap_nm) is not int or gap_nm < 0:
        raise ValueError("materialized block gap must be a non-negative integer nm")
    certificate_names = {name for name, _bbox in certificate.blocks}
    if not isinstance(master_members, Mapping):
        raise ValueError("materialized block mapping must contain exact typed rows")
    canonical_member_rows: List[Tuple[str, Tuple[str, ...]]] = []
    for master, members in master_members.items():
        if (
            not isinstance(master, str)
            or not master
            or not isinstance(members, (tuple, list))
            or not members
            or any(not isinstance(member, str) or not member for member in members)
        ):
            raise ValueError("materialized block mapping must contain exact typed rows")
        canonical_member_rows.append((master, tuple(sorted(members))))
    canonical_members = tuple(sorted(canonical_member_rows))
    if {master for master, _members in canonical_members} != certificate_names:
        raise ValueError("materialized block mapping must cover every master block")
    flattened = [member for _master, members in canonical_members for member in members]
    if (
        any(not master or not members for master, members in canonical_members)
        or any(not member for member in flattened)
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(blocks)
    ):
        raise ValueError(
            "materialized block mapping must partition the exact block inventory"
        )
    owner_by_materialized_block = {
        member: master
        for master, members in canonical_members
        for member in members
    }
    canonical_overlap_pairs = _canonical_overlap_compatible_pairs(
        overlap_compatible_pairs,
        block_ids=tuple(sorted(certificate_names)),
    )
    if canonical_overlap_pairs != certificate.overlap_compatible_pairs:
        raise ValueError(
            "post-materialization overlap compatibility differs from solver proof"
        )
    overlap_compatible_set = set(canonical_overlap_pairs)
    certificate_dimensions = {
        name: (bbox[2] - bbox[0], bbox[3] - bbox[1])
        for name, bbox in certificate.blocks
    }
    canonical_layered_footprints = _canonical_layered_collision_footprints(
        layered_collision_footprints,
        dims=certificate_dimensions,
    )
    if canonical_layered_footprints != certificate.layered_collision_footprints:
        raise ValueError(
            "post-materialization layered collision atoms differ from solver proof"
        )
    layered_owners = {
        owner for owner, _atoms in canonical_layered_footprints
    }
    flexible_group_by_master: Dict[str, str] = {}
    flexible_projection_by_master: Dict[str, FlexibleEnvelopeProjection] = {}
    for projection in certificate.flexible_envelope_projections:
        for master in projection.constraint.members:
            if master in flexible_group_by_master:
                raise ValueError("flexible materialization repeats master ownership")
            flexible_group_by_master[master] = projection.constraint.group
            flexible_projection_by_master[master] = projection
    block_rows = tuple(sorted(blocks.items()))
    source_certificate_blocks = dict(certificate.blocks)
    physical_bbox_by_master: Dict[str, _Bbox] = {}
    for master, members in canonical_members:
        member_boxes = tuple(blocks[member] for member in members)
        physical_bbox_by_master[master] = (
            min(bbox[0] for bbox in member_boxes),
            min(bbox[1] for bbox in member_boxes),
            max(bbox[2] for bbox in member_boxes),
            max(bbox[3] for bbox in member_boxes),
        )
    for projection in certificate.flexible_envelope_projections:
        physical_offsets = dict(projection.constraint.member_physical_offsets_nm)
        for master in projection.constraint.members:
            if master not in source_certificate_blocks or master not in physical_bbox_by_master:
                raise ValueError("flexible materialization lacks exact physical members")
            content_bbox = source_certificate_blocks[master]
            offset = physical_offsets[master]
            expected_physical_bbox = (
                content_bbox[0] + offset[0],
                content_bbox[1] + offset[1],
                content_bbox[0] + offset[2],
                content_bbox[1] + offset[3],
            )
            if physical_bbox_by_master[master] != expected_physical_bbox:
                raise ValueError(
                    "flexible materialization differs from a solved member envelope"
                )
        physical_boxes = tuple(
            physical_bbox_by_master[master]
            for master in projection.constraint.members
        )
        physical_envelope = (
            min(bbox[0] for bbox in physical_boxes),
            min(bbox[1] for bbox in physical_boxes),
            max(bbox[2] for bbox in physical_boxes),
            max(bbox[3] for bbox in physical_boxes),
        )
        if physical_envelope != projection.bbox_nm:
            raise ValueError(
                "flexible materialization differs from its solved physical envelope"
            )
        required_by_master = {
            (member, foreign): clearance
            for member, foreign, clearance in projection.foreign_clearances_nm
        }
        for foreign_master in physical_bbox_by_master:
            if foreign_master in projection.constraint.members:
                continue
            required_rows = tuple(
                required_by_master.get((master, foreign_master))
                for master in projection.constraint.members
            )
            if any(required is None for required in required_rows):
                raise ValueError(
                    "flexible materialization lacks foreign clearance authority"
                )
            if projection.constraint.foreign_exclusion_mode == "envelope":
                if (
                    _bbox_clearance_nm(
                        physical_envelope,
                        physical_bbox_by_master[foreign_master],
                    )
                    < max(int(required) for required in required_rows)
                ):
                    raise ValueError(
                        "flexible materialization contains foreign envelope geometry"
                    )
            else:
                for master, required in zip(
                    projection.constraint.members,
                    required_rows,
                ):
                    if (
                        _bbox_clearance_nm(
                            physical_bbox_by_master[master],
                            physical_bbox_by_master[foreign_master],
                        )
                        < int(required)
                    ):
                        raise ValueError(
                            "flexible materialization violates foreign clearance"
                        )
    for index, (left_name, left) in enumerate(block_rows):
        for right_name, right in block_rows[index + 1 :]:
            left_owner = owner_by_materialized_block[left_name]
            right_owner = owner_by_materialized_block[right_name]
            if left_owner == right_owner or (
                flexible_group_by_master.get(left_owner) is not None
                and flexible_group_by_master.get(left_owner)
                == flexible_group_by_master.get(right_owner)
            ):
                continue
            if tuple(sorted((left_owner, right_owner))) in overlap_compatible_set:
                continue
            if left_owner in layered_owners or right_owner in layered_owners:
                # The solver certificate checked these exact master-local atoms;
                # equality above binds them to the materialized artwork.
                continue
            actual_clearance_nm = min(
                _bbox_clearance_nm(left_atom, right_atom)
                for left_atom in physical_collision_boxes[left_name]
                for right_atom in physical_collision_boxes[right_name]
            )
            if actual_clearance_nm < gap_nm:
                raise ValueError(
                    f"post-materialization blocks {left_name}/{right_name} violate "
                    "final-master clearance: "
                    f"actual={actual_clearance_nm}nm required={gap_nm}nm"
                )
    pair_clearances = _canonical_pair_clearances_nm(
        pair_clearances_nm,
        block_ids=tuple(certificate_names if pair_clearance_scope == "solver" else blocks),
        base_gap_nm=gap_nm,
    )
    if pair_clearance_scope == "solver":
        # Solver constraints name rigid masters, while the final physical replay
        # names their disjoint materialized members. Expand every master pair
        # over its complete Cartesian member inventory; never drop a rule or
        # guess the namespace from a possibly overlapping owner name.
        members_by_master = dict(canonical_members)
        expanded_clearances = {}
        for (left_master, right_master), required in pair_clearances.items():
            for left_member in members_by_master[left_master]:
                for right_member in members_by_master[right_master]:
                    pair = tuple(sorted((left_member, right_member)))
                    expanded_clearances[pair] = max(expanded_clearances.get(pair, gap_nm), required)
        pair_clearances = dict(sorted(expanded_clearances.items()))
    for (left_name, right_name), required in pair_clearances.items():
        left_owner = owner_by_materialized_block[left_name]
        right_owner = owner_by_materialized_block[right_name]
        if left_owner == right_owner or (
            flexible_group_by_master.get(left_owner) is not None
            and flexible_group_by_master.get(left_owner)
            == flexible_group_by_master.get(right_owner)
        ):
            continue
        if tuple(sorted((left_owner, right_owner))) in overlap_compatible_set:
            raise ValueError(
                "overlap-compatible owner pair cannot carry a selective clearance"
            )
        if left_owner in layered_owners or right_owner in layered_owners:
            continue
        if min(
            _bbox_clearance_nm(left_atom, right_atom)
            for left_atom in physical_collision_boxes[left_name]
            for right_atom in physical_collision_boxes[right_name]
        ) < required:
            raise ValueError(
                "post-materialization blocks violate pair-selective clearance"
            )
    try:
        raw_solve_cuts = tuple(solve_cuts or ())
    except TypeError as exc:
        raise ValueError("solve_cuts must be an exact finite sequence") from exc
    prepared_cuts: Tuple[_PreparedSolveCut, ...] = ()
    if certificate.applied_cut_hashes:
        from clay.layout.place.resource_feedback import PlacementSolveCut

        if (
            not raw_solve_cuts
            or any(type(cut) is not PlacementSolveCut for cut in raw_solve_cuts)
            or tuple(sorted(cut.cut_hash for cut in raw_solve_cuts))
            != certificate.applied_cut_hashes
        ):
            raise ValueError(
                "post-materialization cut replay is required with the exact applied cuts"
            )
        prepared_cuts = tuple(
            sorted(
                (
                    _PreparedSolveCut(
                        cut_hash=cut.cut_hash,
                        placement_hash=cut.placement_hash,
                        candidate_pool_hash=cut.candidate_pool_hash,
                        owner_a=(
                            cut.movable_block_ids[0]
                            if not cut.incumbent_origins
                            else ""
                        ),
                        owner_b=(
                            cut.movable_block_ids[1]
                            if not cut.incumbent_origins
                            else ""
                        ),
                        corridor_relation=(
                            str(cut.corridor_relation)
                            if not cut.incumbent_origins
                            else ""
                        ),
                        required_clearance_nm=cut.required_clearance_nm,
                        incumbent_origins=cut.incumbent_origins,
                    )
                    for cut in raw_solve_cuts
                ),
                key=lambda cut: cut.cut_hash,
            )
        )
    elif raw_solve_cuts:
        raise ValueError("materialized cut replay has no applied master evidence")
    rebound_blocks = []
    for master, members in canonical_members:
        if master in flexible_group_by_master:
            if master not in source_certificate_blocks:
                raise ValueError("flexible materialization lost solved ownership")
            source_bbox = source_certificate_blocks[master]
            source_width = source_bbox[2] - source_bbox[0]
            source_height = source_bbox[3] - source_bbox[1]
            projection = flexible_projection_by_master[master]
            physical_offset = dict(
                projection.constraint.member_physical_offsets_nm
            )[master]
            if physical_offset == (0, 0, source_width, source_height):
                rebound_blocks.append((master, physical_bbox_by_master[master]))
            else:
                if tuple(members) != (master,):
                    raise ValueError(
                        "flexible materialization changed virtual ownership"
                    )
                rebound_blocks.append((master, source_bbox))
        else:
            member_boxes = [blocks[member] for member in members]
            rebound_blocks.append(
                (
                    master,
                    (
                        min(bbox[0] for bbox in member_boxes),
                        min(bbox[1] for bbox in member_boxes),
                        max(bbox[2] for bbox in member_boxes),
                        max(bbox[3] for bbox in member_boxes),
                    ),
                )
            )
    rebound = tuple(sorted(rebound_blocks))
    rebound_by_name = dict(rebound)
    rebound_axis_groups = _canonical_axis_alignment_groups(
        certificate.axis_alignment_groups,
        block_ids=tuple(rebound_by_name),
        dims={
            name: (bbox[2] - bbox[0], bbox[3] - bbox[1])
            for name, bbox in rebound
        },
    )
    _assert_axis_alignment_positions(
        {name: (bbox[0], bbox[1]) for name, bbox in rebound},
        rebound_axis_groups,
    )
    rebound_dims = {
        name: (bbox[2] - bbox[0], bbox[3] - bbox[1])
        for name, bbox in rebound
    }
    rebound_center_pairs = _canonical_center_alignment_pairs(
        certificate.center_alignment_pairs,
        block_ids=tuple(rebound_by_name),
        dims=rebound_dims,
    )
    _assert_center_alignment_positions(
        {name: (bbox[0], bbox[1]) for name, bbox in rebound},
        rebound_dims,
        rebound_center_pairs,
    )
    rebound_aggregate_orders = _canonical_aggregate_envelope_orders(
        certificate.aggregate_envelope_orders,
        block_ids=tuple(rebound_by_name),
    )
    _assert_aggregate_envelope_order_positions(
        {name: (bbox[0], bbox[1]) for name, bbox in rebound},
        rebound_dims,
        rebound_aggregate_orders,
    )
    rebound_hierarchical_orders = _canonical_hierarchical_macro_orders(
        certificate.hierarchical_macro_orders,
        block_ids=tuple(rebound_by_name),
    )
    rebound_hierarchical_alignments = _canonical_hierarchical_macro_alignments(
        certificate.hierarchical_macro_alignments,
        block_ids=tuple(rebound_by_name),
    )
    rebound_alignment_scales = (
        _canonical_hierarchical_alignment_tolerance_scales_nm(
            certificate.hierarchical_alignment_tolerance_scales_nm,
            alignments=rebound_hierarchical_alignments,
        )
    )
    if any(
        _hierarchical_macro_alignment_parts(row)[3] > 0
        for row in rebound_hierarchical_alignments
    ) and not rebound_alignment_scales:
        raise ValueError(
            "materialized tolerance alignment lacks its derived frozen scale"
        )
    _assert_hierarchical_macro_positions(
        {name: (bbox[0], bbox[1]) for name, bbox in rebound},
        rebound_dims,
        rebound_hierarchical_orders,
        rebound_hierarchical_alignments,
        gap_nm=gap_nm,
        tolerance_scales_nm=rebound_alignment_scales,
    )
    if prepared_cuts and not all(
        _cut_satisfied_by_bboxes(cut, rebound_by_name) for cut in prepared_cuts
    ):
        raise PlacementSolveCutResidualError(
            replace(
                certificate,
                status="RESIDUAL_SOLVE_CUT_UNSATISFIED",
                cuts_satisfied=False,
                applied_cut_hashes=(),
            )
        )
    right_rail_margin_nm = int(
        getattr(
            getattr(certificate, "edge_rail_reservation", None),
            "right_margin_nm",
            0,
        )
        or 0
    )
    top_rail_margin_nm = int(
        getattr(
            getattr(certificate, "edge_rail_reservation", None),
            "top_margin_nm",
            0,
        )
        or 0
    )
    measured_rebound = tuple(bbox for _name, bbox in block_rows) + tuple(
        projection.bbox_nm
        for projection in certificate.flexible_envelope_projections
    )
    bbox_nm = (
        min(bbox[0] for bbox in measured_rebound),
        (
            0
            if certificate.declared_bottom_edge_keepout_nm > 0
            else min(bbox[1] for bbox in measured_rebound)
        ),
        max(bbox[2] for bbox in measured_rebound) + right_rail_margin_nm,
        max(bbox[3] for bbox in measured_rebound) + top_rail_margin_nm,
    )
    height_nm = bbox_nm[3] - bbox_nm[1]
    measured_aspect = (bbox_nm[2] - bbox_nm[0]) / height_nm if height_nm else 0.0
    declared_aspect = float(certificate.declared_aspect)
    aspect_residual = (
        abs(measured_aspect - declared_aspect) / declared_aspect
        if declared_aspect > 0.0
        else float("inf")
    )
    from clay.layout.place.resource_feedback import build_placement_causality_basis

    declaration_refs = tuple(sorted(placement_declaration_refs or ()))
    rebound_certificate = replace(
        certificate,
        blocks=rebound,
        bbox_nm=bbox_nm,
        measured_aspect=measured_aspect,
        aspect_residual=aspect_residual,
        aspect_satisfied=(
            not aspect_obligation
            or (
                math.isfinite(aspect_residual)
                and aspect_residual <= _ASPECT_TOL
                and (
                    certificate.declared_aspect_min is None
                    or measured_aspect >= certificate.declared_aspect_min
                )
                and (
                    certificate.declared_aspect_max is None
                    or measured_aspect <= certificate.declared_aspect_max
                )
            )
        ),
        materialized_block_members=canonical_members,
        materialized_block_bboxes=block_rows,
        materialized_block_source_members=materialized_block_source_members,
        materialized_source_member_bboxes=materialized_source_member_bboxes,
        layered_placement_assemblies=tuple(sorted(layered_placement_assemblies, key=lambda binding: binding.assembly_id)),
        layered_placement_replay=layered_placement_replay,
        layered_placement_projection=layered_placement_projection,
        placement_declaration_refs=declaration_refs,
        hierarchical_macro_orders=rebound_hierarchical_orders,
        hierarchical_macro_alignments=rebound_hierarchical_alignments,
        hierarchical_alignment_tolerance_scales_nm=rebound_alignment_scales,
        overlap_compatible_pairs=canonical_overlap_pairs,
        outline_width_satisfied=(
            certificate.declared_outline_width_nm is None
            or bbox_nm[2] - bbox_nm[0] == certificate.declared_outline_width_nm
        ),
    )
    return replace(
        rebound_certificate,
        placement_causality_basis=build_placement_causality_basis(
            rebound_certificate.blocks,
            master_members=canonical_members,
            declaration_refs=declaration_refs,
        ),
    )

def _placement_result(
    positions: Mapping[str, Tuple[int, int]],
    items: Sequence[Tuple[str, _Bbox]],
    bands: Mapping[str, int],
    aspect_target: float,
    channel_nm: int,
    evidence: _SolverEvidence,
    *,
    gap_nm: int = 0,
    solve_cuts: Sequence[_PreparedSolveCut] = (),
    cut_residual_status: Optional[str] = None,
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = True,
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]] = (),
    center_alignment_pairs: Sequence[Tuple[str, str, str]] = (),
    aggregate_envelope_orders: Sequence[_AggregateEnvelopeOrder] = (),
    hierarchical_macro_orders: Sequence[_HierarchicalMacroOrder] = (),
    hierarchical_macro_alignments: Sequence[_HierarchicalMacroAlignment] = (),
    hierarchical_alignment_tolerance_scales_nm: (
        _HierarchicalAlignmentToleranceScalesNm
    ) = (),
    minimum_structural_occupancy: Optional[float] = None,
    fixed_outline_width_nm: Optional[int] = None,
    floorplan_column_classes: Sequence[Tuple[str, int, int, int]] = (),
    floorplan_column_groups: Sequence[_FloorplanColumnGroup] = (),
    declared_bottom_edge_keepout_nm: int = 0,
    bottom_edge_keepout_carrier_blocks: Tuple[str, ...] = (),
    bottom_edge_keepout_clear_blocks: Tuple[str, ...] = (),
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    row_bounded_cavity_projections: Tuple[_RowBoundedCavityProjection, ...] = (),
    mosaic_guest_slot_projections: Tuple[MosaicGuestSlotProjection, ...] = (),
    edge_rail_reservation: Any = None,
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
) -> PlaceBlocks2DResult:
    hierarchical_alignment_tolerance_scales_nm = (
        _require_hierarchical_alignment_tolerance_scales_nm(
            hierarchical_alignment_tolerance_scales_nm,
            alignments=hierarchical_macro_alignments,
        )
    )
    item_dims = {
        name: (bbox[2] - bbox[0], bbox[3] - bbox[1])
        for name, bbox in items
    }
    canonical_compound_footprints = _canonical_rigid_compound_footprints(
        rigid_compound_footprints,
        dims=item_dims,
    )
    canonical_layered_footprints = _canonical_layered_collision_footprints(
        layered_collision_footprints,
        dims=item_dims,
    )
    canonical_cavity_projections = _canonical_row_bounded_cavity_projections(
        row_bounded_cavity_projections,
        dims=item_dims,
    )
    canonical_slot_projections = _canonical_mosaic_guest_slot_projections(
        mosaic_guest_slot_projections, dims=item_dims, minimum_clearance_nm=gap_nm,
    )
    canonical_overlap_pairs = _canonical_overlap_compatible_pairs(
        overlap_compatible_pairs,
        block_ids=tuple(name for name, _bbox in items),
        pair_clearances_nm=pair_clearances_nm,
    )
    frozen_positions = MappingProxyType(
        {name: (int(xy[0]), int(xy[1])) for name, xy in sorted(positions.items())}
    )
    absolute_blocks = tuple(
        (
            name,
            (
                frozen_positions[name][0],
                frozen_positions[name][1],
                frozen_positions[name][0] + bbox[2] - bbox[0],
                frozen_positions[name][1] + bbox[3] - bbox[1],
            ),
        )
        for name, bbox in items
        if name in frozen_positions
    )
    flexible_envelope_projections = tuple(
        evidence.flexible_envelope_projections
    )
    measured_bboxes = tuple(bbox for _name, bbox in absolute_blocks) + tuple(
        projection.bbox_nm for projection in flexible_envelope_projections
    )
    right_rail_margin_nm = int(
        getattr(edge_rail_reservation, "right_margin_nm", 0) or 0
    )
    top_rail_margin_nm = int(
        getattr(edge_rail_reservation, "top_margin_nm", 0) or 0
    )
    if measured_bboxes:
        bbox_nm = (
            min(bbox[0] for bbox in measured_bboxes),
            (
                0
                if int(declared_bottom_edge_keepout_nm) > 0
                else min(bbox[1] for bbox in measured_bboxes)
            ),
            max(bbox[2] for bbox in measured_bboxes) + right_rail_margin_nm,
            max(bbox[3] for bbox in measured_bboxes) + top_rail_margin_nm,
        )
        height = bbox_nm[3] - bbox_nm[1]
        measured_aspect = (bbox_nm[2] - bbox_nm[0]) / height if height else 0.0
    else:
        bbox_nm = (0, 0, 0, 0)
        measured_aspect = 0.0
    declared_aspect = float(aspect_target)
    residual = (
        abs(measured_aspect - declared_aspect) / declared_aspect
        if declared_aspect > 0.0
        else float("inf")
    )
    declared_aspect_min = None if aspect_bounds is None else float(aspect_bounds[0])
    declared_aspect_max = None if aspect_bounds is None else float(aspect_bounds[1])
    inside_declared_bounds = aspect_bounds is None or (
        declared_aspect_min <= measured_aspect <= declared_aspect_max
    )
    absolute_by_name = dict(absolute_blocks)

    def flexible_physical_bbox(
        projection: FlexibleEnvelopeProjection,
        member: str,
    ) -> _Bbox:
        content = absolute_by_name[member]
        offset = dict(projection.constraint.member_physical_offsets_nm)[member]
        return (
            content[0] + offset[0],
            content[1] + offset[1],
            content[0] + offset[2],
            content[1] + offset[3],
        )

    flexible_geometry_legal = all(
        projection.bbox_nm
        == (
            min(
                flexible_physical_bbox(projection, name)[0]
                for name in projection.constraint.members
            ),
            min(
                flexible_physical_bbox(projection, name)[1]
                for name in projection.constraint.members
            ),
            max(
                flexible_physical_bbox(projection, name)[2]
                for name in projection.constraint.members
            ),
            max(
                flexible_physical_bbox(projection, name)[3]
                for name in projection.constraint.members
            ),
        )
        and all(
            _bbox_clearance_nm(
                flexible_physical_bbox(projection, member),
                bbox,
            )
            >= {
                (left, right): clearance
                for left, right, clearance in projection.foreign_clearances_nm
            }[(member, name)]
            for member in projection.constraint.members
            for name, bbox in absolute_blocks
            if name not in projection.constraint.members
        )
        for projection in flexible_envelope_projections
    )
    geometry_legal = (
        len(absolute_blocks) == len(items)
        and {name for name, _bbox in absolute_blocks} == {name for name, _bbox in items}
        and all(
            bbox[2] >= bbox[0] and bbox[3] >= bbox[1] for _name, bbox in absolute_blocks
        )
        and _seed_geometry_legal(
            frozen_positions,
            dims=item_dims,
            gap_nm=int(gap_nm),
            pair_clearances_nm=pair_clearances_nm,
            rigid_compound_footprints=canonical_compound_footprints,
            layered_collision_footprints=canonical_layered_footprints,
            overlap_compatible_pairs=canonical_overlap_pairs,
        )
        and flexible_geometry_legal
        and all(mosaic_guest_slot_satisfied(frozen_positions, row) for row in canonical_slot_projections)
        and all(
            _row_bounded_cavity_satisfied(
                frozen_positions,
                dims=item_dims,
                projection=row,
                minimum_clearance_nm=gap_nm,
            )
            for row in canonical_cavity_projections
        )
    )
    cuts_satisfied = cut_residual_status is None and all(
        _cut_satisfied_by_bboxes(cut, absolute_by_name) for cut in solve_cuts
    )
    applied_cut_hashes = (
        tuple(sorted(cut.cut_hash for cut in solve_cuts))
        if solve_cuts and cuts_satisfied
        else ()
    )
    certificate_status = evidence.status
    if cut_residual_status is not None:
        certificate_status = cut_residual_status
    elif solve_cuts and not cuts_satisfied:
        certificate_status = "RESIDUAL_SOLVE_CUT_UNSATISFIED"
    certificate = FinalPlacementCertificate(
        blocks=absolute_blocks,
        bands=tuple(sorted((str(name), int(band)) for name, band in bands.items())),
        bbox_nm=bbox_nm,
        status=certificate_status,
        objective_name=evidence.objective_name,
        objective=evidence.objective,
        best_bound=evidence.best_bound,
        used_fallback=evidence.used_fallback,
        fallback_kind=evidence.fallback_kind,
        primary_status=evidence.primary_status,
        primary_objective_name=evidence.primary_objective_name,
        primary_objective=evidence.primary_objective,
        primary_best_bound=evidence.primary_best_bound,
        measured_aspect=measured_aspect,
        declared_aspect=declared_aspect,
        aspect_residual=residual,
        declared_channel_nm=int(channel_nm or 0),
        declared_aspect_min=declared_aspect_min,
        declared_aspect_max=declared_aspect_max,
        geometry_legal=geometry_legal,
        cuts_satisfied=cuts_satisfied,
        aspect_satisfied=(
            not aspect_obligation
            or (
                math.isfinite(residual)
                and residual <= _ASPECT_TOL
                and inside_declared_bounds
            )
        ),
        aspect_obligation=aspect_obligation,
        resource_status="UNAVAILABLE",
        applied_cut_hashes=applied_cut_hashes,
        axis_alignment_groups=tuple(
            tuple((str(name), int(offset2)) for name, offset2 in group)
            for group in axis_alignment_groups
        ),
        center_alignment_pairs=tuple(
            (str(left), str(right), str(axis))
            for left, right, axis in center_alignment_pairs
        ),
        aggregate_envelope_orders=tuple(aggregate_envelope_orders),
        hierarchical_macro_orders=tuple(hierarchical_macro_orders),
        hierarchical_macro_alignments=tuple(hierarchical_macro_alignments),
        hierarchical_alignment_tolerance_scales_nm=(
            _canonical_hierarchical_alignment_tolerance_scales_nm(
                tuple(hierarchical_alignment_tolerance_scales_nm),
                alignments=tuple(hierarchical_macro_alignments),
            )
        ),
        declared_minimum_structural_occupancy=(
            canonical_minimum_structural_occupancy(
                minimum_structural_occupancy
            )
        ),
        declared_outline_width_nm=fixed_outline_width_nm,
        outline_width_satisfied=(
            fixed_outline_width_nm is None
            or bbox_nm[2] - bbox_nm[0] == fixed_outline_width_nm
        ),
        floorplan_column_classes=tuple(floorplan_column_classes),
        floorplan_column_groups=tuple(floorplan_column_groups),
        flexible_envelope_projections=flexible_envelope_projections,
        declared_bottom_edge_keepout_nm=declared_bottom_edge_keepout_nm,
        bottom_edge_keepout_carrier_blocks=bottom_edge_keepout_carrier_blocks,
        bottom_edge_keepout_clear_blocks=bottom_edge_keepout_clear_blocks,
        declared_gap_nm=int(gap_nm),
        full2d_compaction_graph=evidence.full2d_compaction_graph,
        full2d_solver_evidence=evidence.full2d_solver_evidence,
        layered_collision_footprints=canonical_layered_footprints,
        row_bounded_cavity_projections=canonical_cavity_projections,
        mosaic_guest_slot_projections=canonical_slot_projections,
        edge_rail_reservation=edge_rail_reservation,
        overlap_compatible_pairs=canonical_overlap_pairs,
    )
    from clay.layout.place.resource_feedback import build_placement_causality_basis

    certificate = replace(
        certificate,
        placement_causality_basis=build_placement_causality_basis(certificate.blocks),
    )
    return PlaceBlocks2DResult(frozen_positions, certificate)

def _assert_materialized_diff_axis_certificate(
    certificate: FinalPlacementCertificate,
    block_member_devices: Mapping[str, Sequence[str]],
    geometry: Mapping[str, Any],
    placement_um: Mapping[str, Tuple[float, float]],
    intent: Any,
    binding: Any,
    *, row_program: Any = None, row_materialization: Any = None,
    tech: Any = None, source_cell: Any = None,
    native_left_alignment_declarations: tuple = (),
) -> None:
    """Recompute differential axes from final accesses, not only master bboxes."""

    observed = _diff_pair_axis_alignment_groups(
        certificate.blocks,
        block_member_devices,
        geometry,
        placement_um,
        intent,
        binding,
    )
    if getattr(intent, 'channel_alignments', ()):
        from clay.layout.joint_placement_terminals import build_placement_terminal_face
        from clay.layout.place.placement_geometry import _source_channel_alignment_groups
        if tech is None or row_program is None or row_materialization is None:
            raise ValueError('channel alignment final replay requires complete native row authority')
        build_placement_terminal_face(tech=tech, geometry=geometry, placement_um=placement_um,
            row_program=row_program, row_materialization=row_materialization,
            intent=intent, source_cell=source_cell)
        observed = tuple(sorted(set((*observed, *_source_channel_alignment_groups(
            certificate.blocks, block_member_devices, geometry, placement_um, intent)))))
    if native_left_alignment_declarations:
        if row_program is None or row_materialization is None:
            raise ValueError('native offset final replay requires source row authority')
        from clay.layout.place.materialized_placement import (
            _require_native_left_alignment_authority, _native_row_left_offset_alignments,
            _native_left_alignment_axis_groups, _replay_native_left_alignments,
        )
        authoring = _require_native_left_alignment_authority(
            intent, row_materialization.source_projection, row_program)
        if not set(native_left_alignment_declarations) <= set(_native_row_left_offset_alignments(authoring, row_program)):
            raise ValueError('native offset final declarations differ from source')
        replayed = _native_left_alignment_axis_groups(native_left_alignment_declarations,
            row_program, certificate.blocks, block_member_devices, geometry, placement_um)
        _replay_native_left_alignments(native_left_alignment_declarations, row_program, geometry, placement_um)
        observed = tuple(sorted(set((*observed, *replayed))))
    certified = tuple(certificate.axis_alignment_groups or ())
    if observed != certified:
        raise RuntimeError(
            "exact materialized differential accesses disagree with the final "
            f"placement certificate: certified={certified!r} observed={observed!r}"
        )
