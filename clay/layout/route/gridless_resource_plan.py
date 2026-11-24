"""Resource-plan preflight, validation, and replay materialization mixin for GridlessEngine."""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import islice
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import weakref

from clay.layout.backend import LayerPolygon, Polygon
from clay.layout.minimum_area import minimum_area_square_side_nm
from clay.layout.backend import LayerPolygon, Polygon
from clay.layout.route.constraints import (
    Access,
    Direction,
    LayerSet,
    Length,
    Mirror,
    RouteConstraint,
    Shield,
    Spacing,
    TrackOrder,
    TrunkLayer,
    TrunkLayerDeclarationError,
    Via,
    Width,
)
from clay.layout.route.engine import (
    NetRoute,
    Obstacle,
    Pin,
    RoutedCell,
    RouteResourcePlanIncompleteError,
    require_resource_plan_support,
)
from clay.layout.route.engine_maze import (
    _NM_PER_UM,
    _Tech,
    _via_cut_size_nm,
    _via_name_for,
)
from clay.layout.route.floorplan_containment import (
    RoutingEnvelope,
    require_routed_containment,
)
from clay.layout.route.gridless_types import (
    GridlessAllocationReplayCertificate,
    Rect,
    TrunkLayerResolutionError,
    _Book,
    _CONSTRAINT_TYPES,
    _GRIDLESS_REPLAY_ISSUER_AUTHORITY,
    _GRIDLESS_REPLAY_RECEIPTS,
    _HANDLED_KINDS,
    _MAX_GRIDLESS_REPLAY_SELECTED_CANDIDATES,
    _OwnedTerminalAtom,
    _RoutedNet,
    _SpatialIndex,
    _ViaSite,
    _WireSeg,
    _allocation_geometry_digest,
    _closed_rects_touch,
    _eg_attr,
    _gridless_replay_receipt_payload,
    _is_lower_sha256,
    _norm,
    _obstacle_order_key,
    _owned_terminal_atom_key,
    _pin_order_key,
    _pin_xy_nm,
    _pop,
    _rect,
    _rects_clear,
    _seg_bbox,
    _sep,
    _verify_gridless_allocation_replay_certificate,
    _via_cut_space_nm,
)
from clay.layout.route.spacing_intent import (
    check_spacing_geometry,
    compile_spacing_rules,
    fixed_spacing_conflicts,
    spacing_nm,
)
from clay.layout.route.terminal_identity import (
    pin_fact as canonical_pin_fact,
    pin_provenance,
)
from clay.layout.route.via_model import landing_rect_for_cut, via_enclosure_nm


class _GridlessResourcePlanMixin:
    """Preflight and exact replay materialization for RouteResourcePlan."""

    @staticmethod
    def _bounded_plan_values(values, limit: int) -> Tuple[object, ...]:
        """Materialize at most ``limit`` values, inspecting cap+1 at most."""
        try:
            count = len(values)
            if count > limit:
                raise ValueError("resource-plan collection exceeds replay limit")
            materialized = tuple(islice(iter(values), limit + 1))
            if len(materialized) > limit:
                raise ValueError("resource-plan collection exceeds replay limit")
            return materialized
        except Exception as exc:
            if type(exc) is ValueError:
                raise
            raise ValueError("malformed resource-plan collection") from exc

    def _raise_plan_incomplete(self, resource_plan: object) -> None:
        """Raise the typed stop without invoking hostile malformed values."""
        from clay.layout.route.resource_plan import (
            RouteResourcePlan,
            SelectedNetCandidate,
        )

        safe_plan = resource_plan
        try:
            safe = (
                type(resource_plan) is RouteResourcePlan
                and type(resource_plan.candidate_pool_hash) is str
                and type(resource_plan.selected_nets) is tuple
                and all(
                    type(selected) is SelectedNetCandidate
                    and type(selected.net) is str
                    and type(selected.candidate_id) is str
                    for selected in resource_plan.selected_nets
                )
            )
        except Exception:
            safe = False
        if not safe:
            safe_plan = RouteResourcePlan(
                candidate_pool_hash="malformed-resource-plan",
                selected_nets=(),
            )
        raise RouteResourcePlanIncompleteError(self, safe_plan)

    def _preflight_resource_plan(self, resource_plan: object) -> None:
        """Validate the closed IR shape before the generic support helper loops."""
        if resource_plan is None:
            return
        from clay.layout.route.resource_plan import (
            AllocatedLandingOption,
            AllocatedNetProgram,
            AllocatedTrack,
            CandidateBranch,
            CandidateBranchStep,
            CandidateEndpointAccess,
            CorridorLayerKey,
            GoalBranchCoverage,
            LandingDemand,
            LandingOption,
            LandingSite,
            LandingSitePairConflict,
            LandingTrackContact,
            LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
            LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY,
            MAX_LANDING_OPTION_SITE_IDS,
            MAX_REQUIRED_NETS,
            MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES,
            MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS,
            MAX_TOTAL_CORRIDOR_USES,
            MAX_TOTAL_LANDING_OPTIONS,
            OwnedConductorComponentCertificate,
            OwnedConductorIndexCertificate,
            RequiredRouteGoal,
            RoutePlanObjectiveCertificate,
            RoutePlanObjectivePhase,
            ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME,
            RouteResourcePlan,
            SelectedLanding,
            SelectedLandingOption,
            SelectedNetCandidate,
            SelectedTrack,
            TerminalDemand,
            TrackChoiceRef,
            TrackResource,
            _canonical_decision_digest,
            route_plan_objective_policy_digest,
        )
        from clay.layout.route.resource_conductors import (
            OwnedConductorComponent,
            validate_owned_conductor_component_index_certificate,
        )

        if type(resource_plan) is not RouteResourcePlan:
            raise TypeError("resource_plan must be RouteResourcePlan or None")

        allowed_records = {
            RouteResourcePlan,
            SelectedNetCandidate,
            SelectedTrack,
            SelectedLanding,
            SelectedLandingOption,
            AllocatedNetProgram,
            AllocatedTrack,
            AllocatedLandingOption,
            CandidateBranch,
            CandidateBranchStep,
            CandidateEndpointAccess,
            CorridorLayerKey,
            GoalBranchCoverage,
            LandingDemand,
            LandingOption,
            LandingSite,
            LandingSitePairConflict,
            LandingTrackContact,
            OwnedConductorComponentCertificate,
            OwnedConductorIndexCertificate,
            OwnedConductorComponent,
            RequiredRouteGoal,
            RoutePlanObjectiveCertificate,
            RoutePlanObjectivePhase,
            TerminalDemand,
            TrackChoiceRef,
            TrackResource,
        }
        active: Set[int] = set()
        objective_phase_ids = (
            "hard_feasibility",
            "new_reserved_geometry_area_nm2",
            "landing_site_cut_count",
            "canonical_semantic_tie",
        )

        def objective_structure() -> Tuple[object, ...]:
            """Check the versioned signed envelope before certificate traversal."""
            if (
                type(resource_plan.candidate_pool_hash) is not str
                or not resource_plan.candidate_pool_hash
                or type(resource_plan.materialization_complete) is not bool
                or type(resource_plan.schema) is not str
                or type(resource_plan.landing_site_conflict_domain_digest) is not str
                or type(resource_plan.lazy_landing_site_conflicts) is not tuple
                or resource_plan.schema not in {
                    "clay.route-resource-plan.v9",
                    "clay.route-resource-plan.v10",
                }
            ):
                raise ValueError("resource-plan top-level fields are malformed")
            certificate = resource_plan.objective_certificate
            if resource_plan.schema == "clay.route-resource-plan.v9" and (
                resource_plan.landing_site_conflict_domain_digest
                or resource_plan.lazy_landing_site_conflicts
            ):
                raise ValueError("signed v7 plan cannot carry conflict-domain evidence")
            if (
                resource_plan.schema == "clay.route-resource-plan.v10"
                and not resource_plan.landing_site_conflict_domain_digest
            ):
                raise ValueError("signed v8 plan lacks its conflict-domain binding")
            if type(certificate) is not RoutePlanObjectiveCertificate:
                raise ValueError("signed plan lacks an exact objective certificate")
            if (
                type(certificate.candidate_pool_hash) is not str
                or not certificate.candidate_pool_hash
                or type(certificate.objective_policy_digest) is not str
                or not certificate.objective_policy_digest
                or type(certificate.canonical_decision_digest) is not str
                or not certificate.canonical_decision_digest
                or isinstance(
                    certificate.authorized_replay_deterministic_time, bool
                )
                or not isinstance(
                    certificate.authorized_replay_deterministic_time,
                    (int, float),
                )
                or certificate.authorized_replay_deterministic_time < 0
                or not math.isfinite(
                    float(certificate.authorized_replay_deterministic_time)
                )
                or certificate.authorized_replay_deterministic_time
                > ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME
                or type(certificate.schema) is not str
                or certificate.schema
                != "clay.route-plan-objective-certificate.v2"
                or type(certificate.phases) is not tuple
            ):
                raise ValueError("objective certificate fields are malformed")
            phases = self._bounded_plan_values(
                certificate.phases, len(objective_phase_ids)
            )
            if not phases:
                raise ValueError("objective certificate phase prefix is empty")
            for phase in phases:
                if type(phase) is not RoutePlanObjectivePhase:
                    raise ValueError("objective certificate phase type is foreign")
                if (
                    type(phase.phase_id) is not str
                    or type(phase.status) is not str
                    or (
                        phase.value is not None
                        and type(phase.value) is not int
                    )
                    or (
                        phase.lower_bound is not None
                        and type(phase.lower_bound) is not int
                    )
                ):
                    raise ValueError("objective certificate phase fields are malformed")
            if tuple(phase.phase_id for phase in phases) != (
                objective_phase_ids[:len(phases)]
            ):
                raise ValueError("objective certificate phase order is non-canonical")
            return phases

        def visit(value: object, *, allow_float: bool = False) -> None:
            value_type = type(value)
            if value is None or value_type in (str, int, bool):
                return
            if value_type is float:
                if allow_float:
                    return
                raise ValueError("resource plan contains a non-canonical scalar")
            if value_type is tuple:
                values = self._bounded_plan_values(
                    value, MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS
                )
                marker = id(value)
                if marker in active:
                    raise ValueError("cyclic resource-plan collection")
                active.add(marker)
                try:
                    for item in values:
                        visit(item)
                finally:
                    active.remove(marker)
                return
            if value_type not in allowed_records:
                raise ValueError("resource plan contains a non-canonical record")
            marker = id(value)
            if marker in active:
                raise ValueError("cyclic resource-plan record")
            active.add(marker)
            try:
                for field in value_type.__dataclass_fields__.values():
                    visit(
                        getattr(value, field.name),
                        allow_float=(
                            value_type is RoutePlanObjectiveCertificate
                            and field.name
                            == "authorized_replay_deterministic_time"
                        ),
                    )
            finally:
                active.remove(marker)
            if value_type is OwnedConductorComponent:
                if (
                    type(value.atom_facts) is not tuple
                    or type(value.atom_refs) is not tuple
                    or type(value.atom_terminal_conductor_refs) is not tuple
                    or len(value.atom_terminal_conductor_refs)
                    != len(value.atom_facts)
                    or any(
                        type(fact) is not tuple
                        or len(fact) != 8
                        or any(type(item) is not int for item in fact[:6])
                        or type(fact[6]) is not str
                        or type(fact[7]) is not str
                        for fact in value.atom_facts
                    )
                    or any(type(ref) is not str for ref in value.atom_refs)
                    or any(
                        ref is not None and (type(ref) is not str or not ref)
                        for ref in value.atom_terminal_conductor_refs
                    )
                ):
                    raise ValueError("owned component atoms are non-canonical")

        try:
            objective_phases = objective_structure()
            visit(resource_plan)
            if (
                type(resource_plan.selected_nets) is not tuple
                or type(resource_plan.allocated_programs) is not tuple
            ):
                raise ValueError("resource-plan top-level collections must be tuples")
            selected = self._bounded_plan_values(
                resource_plan.selected_nets, MAX_REQUIRED_NETS
            )
            programs = self._bounded_plan_values(
                resource_plan.allocated_programs, MAX_REQUIRED_NETS
            )
        except Exception as exc:
            try:
                self._raise_plan_incomplete(resource_plan)
            except RouteResourcePlanIncompleteError as error:
                raise error from exc
        if (
            type(resource_plan.candidate_pool_hash) is not str
            or any(type(item) is not SelectedNetCandidate for item in selected)
            or any(type(item) is not AllocatedNetProgram for item in programs)
            or any(
                type(item.schema) is not str
                or item.schema != "clay.allocated-net-program.v2"
                for item in programs
            )
            or any(
                type(item.net) is not str
                or type(item.candidate_id) is not str
                for item in (*selected, *programs)
            )
        ):
            self._raise_plan_incomplete(resource_plan)

        if not selected and programs:
            self._raise_plan_incomplete(resource_plan)

        if resource_plan.schema in {
            "clay.route-resource-plan.v9",
            "clay.route-resource-plan.v10",
        }:
            try:
                certificate = resource_plan.objective_certificate
                if type(certificate) is not RoutePlanObjectiveCertificate:
                    raise ValueError("signed objective certificate is missing")

                index_certificate = (
                    resource_plan.owned_conductor_index_certificate
                )
                if index_certificate is None:
                    raise ValueError(
                        "exact plan lacks a complete owned-conductor index certificate"
                    )
                else:
                    if type(index_certificate) is not OwnedConductorIndexCertificate:
                        raise ValueError(
                            "owned-conductor index certificate has foreign type"
                        )
                    if (
                        type(index_certificate.world_digest) is not str
                        or not index_certificate.world_digest
                        or type(index_certificate.index_digest) is not str
                        or not index_certificate.index_digest
                        or type(index_certificate.schema) is not str
                        or index_certificate.schema
                        != "clay.owned-conductor-index-certificate.v1"
                        or type(index_certificate.components) is not tuple
                    ):
                        raise ValueError(
                            "owned-conductor index certificate fields are malformed"
                        )
                    full_components = self._bounded_plan_values(
                        index_certificate.components,
                        MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES,
                    )
                    if any(
                        type(item) is not OwnedConductorComponentCertificate
                        for item in full_components
                    ):
                        raise ValueError(
                            "owned-conductor index table contains foreign records"
                        )
                    validate_owned_conductor_component_index_certificate(
                        world_digest=index_certificate.world_digest,
                        components=tuple(
                            item.component for item in full_components
                        ),
                        index_digest=index_certificate.index_digest,
                    )
                    rebuilt_index_certificate = OwnedConductorIndexCertificate(
                        world_digest=index_certificate.world_digest,
                        components=tuple(full_components),
                        index_digest=index_certificate.index_digest,
                        schema=index_certificate.schema,
                    )
                    if rebuilt_index_certificate != index_certificate:
                        raise ValueError(
                            "owned-conductor index certificate is not canonical"
                        )
                    full_components_by_ref = {
                        item.component.component_ref: item
                        for item in full_components
                    }
                    if len(full_components_by_ref) != len(full_components):
                        raise ValueError(
                            "owned-conductor index table has duplicate refs"
                        )
                    for program in programs:
                        program_components = self._bounded_plan_values(
                            program.owned_conductor_components,
                            MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES,
                        )
                        if (
                            program.routing_world_provenance_digest
                            != index_certificate.world_digest
                            or program.owned_conductor_index_digest
                            != index_certificate.index_digest
                            or any(
                                full_components_by_ref.get(
                                    item.component.component_ref
                                ) != item
                                for item in program_components
                            )
                        ):
                            raise ValueError(
                                "allocated program owned-conductor evidence is "
                                "not an exact subset of the complete table"
                            )

                # Reconstruct only after exact record/field/collection preflight.
                # This applies the dataclasses' scalar semantics without letting a
                # hostile object participate in constructor coercion or sorting.
                rebuilt_phases = tuple(RoutePlanObjectivePhase(
                    phase.phase_id,
                    phase.status,
                    phase.value,
                    phase.lower_bound,
                ) for phase in objective_phases)
                rebuilt_certificate = RoutePlanObjectiveCertificate(
                    candidate_pool_hash=certificate.candidate_pool_hash,
                    objective_policy_digest=certificate.objective_policy_digest,
                    phases=rebuilt_phases,
                    canonical_decision_digest=certificate.canonical_decision_digest,
                    authorized_replay_deterministic_time=(
                        certificate.authorized_replay_deterministic_time
                    ),
                    schema=certificate.schema,
                )
                if rebuilt_certificate != certificate:
                    raise ValueError("objective certificate is not canonical")
                if (
                    certificate.candidate_pool_hash
                    != resource_plan.candidate_pool_hash
                    or certificate.objective_policy_digest
                    != route_plan_objective_policy_digest()
                    or certificate.canonical_decision_digest
                    != _canonical_decision_digest(selected)
                ):
                    raise ValueError("objective certificate signature mismatch")

                track_area = 0
                sites_by_id: Dict[str, object] = {}
                for program in programs:
                    tracks = self._bounded_plan_values(
                        program.tracks, MAX_TOTAL_CORRIDOR_USES
                    )
                    allocations = self._bounded_plan_values(
                        program.landing_options, MAX_TOTAL_LANDING_OPTIONS
                    )
                    for allocation in tracks:
                        bbox = self._allocated_track_bbox(program, allocation)
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area < 0:
                            raise ValueError("objective track bbox is malformed")
                        track_area += area
                    for allocation in allocations:
                        landing_sites = self._bounded_plan_values(
                            allocation.landing_sites,
                            MAX_LANDING_OPTION_SITE_IDS,
                        )
                        for site in landing_sites:
                            prior = sites_by_id.setdefault(site.site_id, site)
                            if prior != site:
                                raise ValueError(
                                    "objective landing-site identity is ambiguous"
                                )
                landing_area = 0
                for site in sites_by_id.values():
                    for bbox in (
                        site.lower_pad_bbox_nm,
                        site.upper_pad_bbox_nm,
                        site.cut_bbox_nm,
                    ):
                        if type(bbox) is not tuple or len(bbox) != 4:
                            raise ValueError("objective landing bbox is malformed")
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        if width < 0 or height < 0:
                            raise ValueError("objective landing bbox is malformed")
                        landing_area += width * height
                expected_values = {
                    "hard_feasibility": 0,
                    "new_reserved_geometry_area_nm2": (
                        track_area + landing_area
                    ),
                    "landing_site_cut_count": len(sites_by_id),
                }
                for phase in objective_phases:
                    if phase.phase_id == "canonical_semantic_tie":
                        if phase.value is not None or phase.lower_bound is not None:
                            raise ValueError(
                                "canonical tie phase carries scalar evidence"
                            )
                        continue
                    expected = expected_values[phase.phase_id]
                    if phase.value != expected:
                        raise ValueError(
                            "objective certificate incumbent does not replay "
                            "allocated geometry"
                        )
                    if phase.phase_id == "hard_feasibility":
                        if phase.status != "OPTIMAL" or phase.lower_bound != 0:
                            raise ValueError(
                                "hard feasibility phase is not exactly proved"
                            )
                        continue
                    if (
                        phase.status not in {"OPTIMAL", "FEASIBLE"}
                        or phase.lower_bound is None
                        or phase.lower_bound > expected
                        or (
                            phase.status == "OPTIMAL"
                            and phase.lower_bound != expected
                        )
                    ):
                        raise ValueError(
                            "objective certificate bound/status does not replay "
                            "allocated geometry"
                        )
            except Exception as exc:
                try:
                    self._raise_plan_incomplete(resource_plan)
                except RouteResourcePlanIncompleteError as error:
                    raise error from exc

    def _preflight_diagnostic_resource_program(self, program: object) -> None:
        """Validate the distinct non-signoff geometry-program transport."""

        from clay.layout.route.resource_plan import (
            AllocatedNetProgram,
            DiagnosticRouteProgramBundle,
            MAX_REQUIRED_NETS,
            SelectedNetCandidate,
        )
        from clay.layout.route.resource_conductors import (
            validate_owned_conductor_component_index_certificate,
        )

        if type(program) is not DiagnosticRouteProgramBundle:
            raise TypeError(
                "diagnostic_resource_program must be an exact "
                "DiagnosticRouteProgramBundle"
            )
        try:
            selected = self._bounded_plan_values(
                program.selected_nets, MAX_REQUIRED_NETS
            )
            allocated = self._bounded_plan_values(
                program.allocated_programs, MAX_REQUIRED_NETS
            )
            selected_pairs = tuple(
                (item.net, item.candidate_id) for item in selected
            )
            allocated_pairs = tuple(
                (item.net, item.candidate_id) for item in allocated
            )
            if (
                not selected_pairs
                or any(type(item) is not SelectedNetCandidate for item in selected)
                or any(type(item) is not AllocatedNetProgram for item in allocated)
                or selected_pairs != allocated_pairs
                or len(selected_pairs) != len(set(selected_pairs))
                or any(
                    item.candidate_pool_hash != program.frontier_digest
                    for item in allocated
                )
                or program.materialization_complete is not True
                or program.sha256() != hashlib.sha256(
                    program.canonical_json().encode("utf-8")
                ).hexdigest()
            ):
                raise ValueError("diagnostic resource program is malformed")
            index = program.owned_conductor_index_certificate
            validate_owned_conductor_component_index_certificate(
                world_digest=index.world_digest,
                components=tuple(item.component for item in index.components),
                index_digest=index.index_digest,
            )
        except Exception as exc:
            raise RouteResourcePlanIncompleteError(self, program) from exc

    @staticmethod
    def _plan_layer_key(pdk, layer_n: int) -> Optional[Tuple[int, int]]:
        """Return the current PDK's exact GDS metal key, or ``None``.

        Resource-plan layer numbers are PDK metal numbers, not Gridless graph
        indices.  In particular, a valid exact plan may use a thick upper metal
        outside ``_Tech``'s fine-routing prefix.
        """
        if type(layer_n) is not int or layer_n <= 0:
            return None
        try:
            key = tuple(pdk.metal_key(layer_n))
        except Exception:
            return None
        if len(key) != 2 or not all(type(value) is int for value in key):
            return None
        return int(key[0]), int(key[1])

    @classmethod
    def _plan_layer_n(cls, tech: _Tech, layer: object) -> Optional[int]:
        """Resolve a hard LayerSet member over all PDK metals, including tops."""
        if isinstance(layer, tuple) and len(layer) == 2:
            try:
                key = tuple(int(value) for value in layer)
            except (TypeError, ValueError):
                return None
            return tech.resolve_layer_to_pdk(key)
        if isinstance(layer, str):
            token = layer.strip().lower()
            layer_n = None
            for prefix in ("metal", "met", "m"):
                if token.startswith(prefix) and token[len(prefix):].isdigit():
                    layer_n = int(token[len(prefix):])
                    break
            if layer_n is None and token.isdigit():
                layer_n = int(token)
            if layer_n is not None and cls._plan_layer_key(tech.pdk, layer_n) is not None:
                return layer_n
        return None

    @staticmethod
    def _allocated_track_bbox(program, allocation) -> Rect:
        centers = tuple(allocation.selected_track_centers_nm)
        low_half = allocation.width_nm // 2
        high_half = allocation.width_nm - low_half
        run0, run1 = allocation.interval_nm
        if allocation.resource.preferred_axis == "H":
            return (
                run0,
                centers[0] - low_half,
                run1,
                centers[-1] + high_half,
            )
        return (
            centers[0] - low_half,
            run0,
            centers[-1] + high_half,
            run1,
        )

    def _materialize_resource_plan(
        self,
        pins_by_net: Dict[str, List[Pin]],
        constraints: Sequence[RouteConstraint],
        pdk,
        tech: _Tech,
        resource_plan,
        *,
        obstacles: Sequence[Obstacle],
        rails: Optional[Dict[str, Sequence[Tuple[Any, Tuple[float, float, float, float]]]]],
        preroutes: Optional[Dict[str, list]],
        diagnostic: bool = False,
        diagnostic_full_pins_by_net: Optional[Dict[str, List[Pin]]] = None,
    ) -> RoutedCell:
        """Replay one byte-complete allocation program as exact rectangles.

        This path is intentionally geometry-only.  Candidate discovery, topology
        inference, MST decomposition, A*, negotiated congestion, via replication,
        and style preferences have already finished (or are irrelevant) once an
        allocated program reaches this seam.
        """

        def incomplete() -> None:
            self._raise_plan_incomplete(resource_plan)

        from clay.layout.route.resource_plan import (
            AllocatedLandingOption,
            AllocatedNetProgram,
            AllocatedTrack,
            CandidateBranch,
            CandidateBranchStep,
            CandidateEndpointAccess,
            GoalBranchCoverage,
            LandingDemand,
            LandingOption,
            LandingSite,
            LandingTrackContact,
            LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
            LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY,
            MAX_CANDIDATE_BRANCH_STEPS,
            MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS,
            MAX_ENDPOINT_OWNER_COMPONENT_REFS,
            MAX_LANDING_OPTION_CONTACTS_PER_OPTION,
            MAX_LANDING_OPTION_SITE_IDS,
            MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES,
            MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS,
            MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE,
            MAX_REQUIRED_NETS,
            MAX_TOTAL_BRANCH_STEPS,
            MAX_TOTAL_CANDIDATE_BRANCHES,
            MAX_TOTAL_ENDPOINT_ACCESSES,
            MAX_TOTAL_GOAL_BRANCH_COVERAGE,
            MAX_TOTAL_LANDING_DEMANDS,
            MAX_TOTAL_LANDING_OPTIONS,
            OwnedConductorComponentCertificate,
            RequiredRouteGoal,
            SelectedLanding,
            SelectedLandingOption,
            SelectedNetCandidate,
            SelectedTrack,
            TrackResource,
        )
        from clay.layout.route.resource_conductors import OwnedConductorComponent

        def plan_values(values: object, limit: int) -> Tuple[object, ...]:
            if type(values) is not tuple:
                incomplete()
            try:
                return self._bounded_plan_values(values, limit)
            except Exception:
                incomplete()

        try:
            if (
                type(pins_by_net) is not dict
                or type(obstacles) not in (list, tuple)
                or type(constraints) not in (list, tuple)
            ):
                incomplete()
            selected_values = plan_values(
                resource_plan.selected_nets, MAX_REQUIRED_NETS
            )
            programs = plan_values(
                resource_plan.allocated_programs, MAX_REQUIRED_NETS
            )
            runtime_obstacles = self._bounded_plan_values(
                obstacles, MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS
            )
            pin_entries = self._bounded_plan_values(
                pins_by_net.items(), MAX_REQUIRED_NETS
            )
            bounded_constraints = self._bounded_plan_values(
                constraints, MAX_TOTAL_BRANCH_STEPS
            )
            if preroutes is not None and type(preroutes) is not dict:
                incomplete()
            preroute_entries = self._bounded_plan_values(
                (preroutes or {}).items(), MAX_REQUIRED_NETS
            )
            if rails is not None and type(rails) is not dict:
                incomplete()
            rail_entries = self._bounded_plan_values(
                (rails or {}).items(), MAX_REQUIRED_NETS
            )
        except Exception:
            incomplete()
        if any(type(constraint) not in _CONSTRAINT_TYPES for constraint in bounded_constraints):
            incomplete()
        if any(
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or not entry[0]
            for entry in pin_entries
        ):
            incomplete()
        if any(
            type(selected) is not SelectedNetCandidate
            or type(selected.net) is not str
            or not selected.net
            or type(selected.candidate_id) is not str
            or not selected.candidate_id
            for selected in selected_values
        ):
            incomplete()
        if any(
            type(program) is not AllocatedNetProgram
            or type(program.net) is not str
            or not program.net
            or type(program.candidate_id) is not str
            or not program.candidate_id
            for program in programs
        ):
            incomplete()
        selected_pairs = tuple(
            (selected.net, selected.candidate_id) for selected in selected_values
        )
        program_pairs = tuple(
            (program.net, program.candidate_id) for program in programs
        )
        pin_nets = {net for net, _pins in pin_entries}
        if len(pin_nets) != len(pin_entries):
            incomplete()
        selected_nets = {net for net, _candidate_id in selected_pairs}
        program_nets = {net for net, _candidate_id in program_pairs}
        if (
            len(selected_pairs) != len(set(selected_pairs))
            or len(program_pairs) != len(set(program_pairs))
            or len(selected_nets) != len(selected_pairs)
            or len(program_nets) != len(program_pairs)
            or selected_nets != program_nets
            or not selected_nets.issubset(pin_nets)
        ):
            incomplete()
        # Topology compilation deliberately omits a net with one physical
        # terminal and no structural declaration: it is electrically complete
        # without emitted route geometry.  Preserve that contract at exact
        # replay while still failing closed for every omitted multi-terminal
        # net (which would indicate an incomplete resource plan).
        unplanned_pin_nets = pin_nets - selected_nets
        for net, net_pins in pin_entries:
            if net not in unplanned_pin_nets:
                continue
            if type(net_pins) not in (list, tuple):
                incomplete()
            try:
                leaf_pins = self._bounded_plan_values(
                    net_pins, MAX_TOTAL_ENDPOINT_ACCESSES
                )
            except Exception:
                incomplete()
            if len(leaf_pins) != 1:
                incomplete()
            pin = leaf_pins[0]
            if (
                type(pin) is not Pin
                or type(pin.net) is not str
                or pin.net != net
                or type(pin.device) is not str
                or type(pin.terminal) is not str
                or type(pin.xy_um) is not tuple
                or len(pin.xy_um) != 2
                or not all(type(value) in (int, float) for value in pin.xy_um)
                or not all(math.isfinite(float(value)) for value in pin.xy_um)
                or (
                    not (
                        type(pin.layer) is tuple
                        and len(pin.layer) == 2
                        and all(type(value) is int for value in pin.layer)
                    )
                    and type(pin.layer) is not str
                )
                or self._plan_layer_n(tech, pin.layer) is None
            ):
                incomplete()
        # Preroutes are emitted route geometry, hence cannot be silently mixed
        # with an exact allocation.  Existing owned artwork is represented by the
        # program's owned-component certificates and is deliberately not re-emitted.
        for entry in preroute_entries:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not str
                or not entry[0]
            ):
                incomplete()
            net, polygons = entry
            if type(polygons) not in (list, tuple):
                incomplete()
            try:
                bounded_polygons = self._bounded_plan_values(
                    polygons, MAX_TOTAL_BRANCH_STEPS
                )
            except Exception:
                incomplete()
            if bounded_polygons:
                incomplete()

        runtime_pin_fact_counts: Dict[Tuple[object, ...], int] = {}
        certificate_pins_by_net = (
            diagnostic_full_pins_by_net
            if diagnostic and diagnostic_full_pins_by_net is not None
            else pins_by_net
        )
        if (
            type(certificate_pins_by_net) is not dict
            or len(certificate_pins_by_net) > MAX_REQUIRED_NETS
        ):
            incomplete()
        for net, net_pins in certificate_pins_by_net.items():
            if (
                type(net) is not str
                or not net
                or type(net_pins) not in (list, tuple)
            ):
                incomplete()
            try:
                bounded_pins = self._bounded_plan_values(
                    net_pins, MAX_TOTAL_ENDPOINT_ACCESSES
                )
            except Exception:
                incomplete()
            for pin in bounded_pins:
                if (
                    type(pin) is not Pin
                    or pin.net != net
                    or type(pin.device) is not str
                    or not pin.device
                    or type(pin.terminal) is not str
                    or not pin.terminal
                    or type(pin.xy_um) is not tuple
                    or len(pin.xy_um) != 2
                    or not all(
                        type(value) in (int, float)
                        and math.isfinite(float(value))
                        for value in pin.xy_um
                    )
                ):
                    incomplete()
                layer_n = self._plan_layer_n(tech, pin.layer)
                layer_key = (
                    None
                    if layer_n is None
                    else self._plan_layer_key(pdk, layer_n)
                )
                if layer_key is None:
                    incomplete()
                try:
                    pin_fact = canonical_pin_fact(pin, layer_key=layer_key)
                except (TypeError, ValueError):
                    incomplete()
                runtime_pin_fact_counts[pin_fact] = (
                    runtime_pin_fact_counts.get(pin_fact, 0) + 1
                )

        runtime_owned_facts: Set[Tuple[object, ...]] = set()
        runtime_owned_fact_counts: Dict[Tuple[object, ...], int] = {}
        runtime_terminal_refs: Dict[Tuple[object, ...], Optional[str]] = {}
        missing_terminal_ref = object()
        for obstacle in runtime_obstacles:
            if type(obstacle) is not Obstacle:
                incomplete()
            if obstacle.owner_net is not None and (
                type(obstacle.owner_net) is not str or not obstacle.owner_net
            ):
                incomplete()
            if obstacle.kind is not None and type(obstacle.kind) is not str:
                incomplete()
            if obstacle.terminal_conductor_ref is not None and (
                type(obstacle.terminal_conductor_ref) is not str
                or not obstacle.terminal_conductor_ref
                or obstacle.kind != "device"
                or obstacle.owner_net is None
            ):
                incomplete()
            if not (
                type(obstacle.layer) is tuple
                and len(obstacle.layer) == 2
                and all(type(value) is int for value in obstacle.layer)
            ) and type(obstacle.layer) is not str:
                incomplete()
            layer_n = self._plan_layer_n(tech, obstacle.layer)
            layer_key = (
                None if layer_n is None else self._plan_layer_key(pdk, layer_n)
            )
            values = (obstacle.x0, obstacle.y0, obstacle.x1, obstacle.y1)
            if (
                layer_key is None
                or not all(type(v) in (int, float) for v in values)
                or not all(math.isfinite(float(v)) for v in values)
            ):
                incomplete()
            if obstacle.owner_net is None:
                continue
            bbox = _norm(tuple(round(float(v) * _NM_PER_UM) for v in values))
            fact = (
                *bbox,
                *layer_key,
                obstacle.owner_net,
                "owned" if obstacle.kind is None else obstacle.kind,
            )
            previous_terminal_ref = runtime_terminal_refs.get(
                fact, missing_terminal_ref
            )
            if (
                previous_terminal_ref is not missing_terminal_ref
                and previous_terminal_ref != obstacle.terminal_conductor_ref
            ):
                incomplete()
            runtime_terminal_refs[fact] = obstacle.terminal_conductor_ref
            runtime_owned_facts.add(fact)
            runtime_owned_fact_counts[fact] = (
                runtime_owned_fact_counts.get(fact, 0) + 1
            )
        for entry in rail_entries:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not str
                or not entry[0]
            ):
                incomplete()
            net, net_rails = entry
            if type(net_rails) not in (list, tuple):
                incomplete()
            try:
                bounded_rails = self._bounded_plan_values(
                    net_rails, MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS
                )
            except Exception:
                incomplete()
            for rail in bounded_rails:
                if (
                    type(rail) is not tuple
                    or len(rail) != 2
                    or type(rail[1]) is not tuple
                    or len(rail[1]) != 4
                    or not all(type(v) in (int, float) for v in rail[1])
                    or not all(math.isfinite(float(v)) for v in rail[1])
                ):
                    incomplete()
                if not (
                    type(rail[0]) is tuple
                    and len(rail[0]) == 2
                    and all(type(value) is int for value in rail[0])
                ) and type(rail[0]) is not str:
                    incomplete()
                layer_n = self._plan_layer_n(tech, rail[0])
                layer_key = (
                    None if layer_n is None else self._plan_layer_key(pdk, layer_n)
                )
                if layer_key is None:
                    incomplete()
                bbox = _norm(tuple(
                    round(float(value) * _NM_PER_UM) for value in rail[1]
                ))
                fact = (*bbox, *layer_key, net, "rail")
                runtime_terminal_refs[fact] = None
                runtime_owned_facts.add(fact)
                runtime_owned_fact_counts[fact] = (
                    runtime_owned_fact_counts.get(fact, 0) + 1
                )

        index_certificate = resource_plan.owned_conductor_index_certificate
        full_component_certificates = (
            () if index_certificate is None else index_certificate.components
        )
        full_component_by_ref = {
            certificate.component.component_ref: certificate
            for certificate in full_component_certificates
        }
        if len(full_component_by_ref) != len(full_component_certificates):
            incomplete()
        certified_owned_fact_counts: Dict[Tuple[object, ...], int] = {}
        certified_pin_fact_counts: Dict[Tuple[object, ...], int] = {}
        for certificate in full_component_certificates:
            component = certificate.component
            expected_key = self._plan_layer_key(pdk, certificate.layer_n)
            if (
                expected_key is None
                or component.layer_key != expected_key
                or component.world_digest != index_certificate.world_digest
            ):
                incomplete()
            for fact, terminal_ref in zip(
                component.atom_facts,
                component.atom_terminal_conductor_refs,
                strict=True,
            ):
                certified_owned_fact_counts[fact] = (
                    certified_owned_fact_counts.get(fact, 0) + 1
                )
                if (
                    fact not in runtime_owned_facts
                    or runtime_terminal_refs.get(fact, missing_terminal_ref)
                    != terminal_ref
                ):
                    incomplete()
            for pin_fact in component.pin_accesses:
                certified_pin_fact_counts[pin_fact] = (
                    certified_pin_fact_counts.get(pin_fact, 0) + 1
                )
        if certified_owned_fact_counts != runtime_owned_fact_counts:
            incomplete()
        if certified_pin_fact_counts != runtime_pin_fact_counts:
            incomplete()

        selected_by_net = {selected.net: selected for selected in selected_values}
        programs_by_net = {program.net: program for program in programs}
        cut_site_by_geometry: Dict[Tuple[Tuple[int, int], Rect], str] = {}
        for program in programs:
            try:
                branches = plan_values(
                    program.branches, MAX_TOTAL_CANDIDATE_BRANCHES
                )
                tracks = plan_values(
                    program.tracks, MAX_TOTAL_BRANCH_STEPS
                )
                landing_options = plan_values(
                    program.landing_options, MAX_TOTAL_LANDING_OPTIONS
                )
                endpoint_accesses = plan_values(
                    program.endpoint_accesses, MAX_TOTAL_ENDPOINT_ACCESSES
                )
                landing_demands = plan_values(
                    program.landing_demands, MAX_TOTAL_LANDING_DEMANDS
                )
                required_goals = plan_values(
                    program.required_route_goals, MAX_TOTAL_GOAL_BRANCH_COVERAGE
                )
                goal_coverage = plan_values(
                    program.goal_branch_coverage, MAX_TOTAL_GOAL_BRANCH_COVERAGE
                )
                component_certificates = plan_values(
                    program.owned_conductor_components,
                    MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES,
                )
                component_use_refs = plan_values(
                    program.owned_component_use_refs,
                    MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE,
                )
            except Exception:
                incomplete()
            if (
                any(type(item) is not CandidateBranch for item in branches)
                or any(type(item) is not AllocatedTrack for item in tracks)
                or any(
                    type(item) is not AllocatedLandingOption
                    for item in landing_options
                )
                or any(
                    type(item) is not CandidateEndpointAccess
                    for item in endpoint_accesses
                )
                or any(type(item) is not LandingDemand for item in landing_demands)
                or any(type(item) is not RequiredRouteGoal for item in required_goals)
                or any(
                    type(item) is not GoalBranchCoverage for item in goal_coverage
                )
                or any(
                    type(item) is not OwnedConductorComponentCertificate
                    for item in component_certificates
                )
                or any(type(ref) is not str or not ref for ref in component_use_refs)
            ):
                incomplete()
            if program.candidate_pool_hash != resource_plan.candidate_pool_hash:
                incomplete()
            if program.program_kind not in {
                "PATTERN_TREE",
                "POWER_SPINE_LOCAL_DROPS",
                "COMPONENT_ONLY",
            }:
                incomplete()
            if program.program_kind == "COMPONENT_ONLY":
                if tracks or landing_options or not component_use_refs:
                    incomplete()
            elif not (
                tracks or landing_options or component_use_refs
            ):
                incomplete()

            if any(
                type(certificate.component) is not OwnedConductorComponent
                or type(certificate.component.component_ref) is not str
                for certificate in component_certificates
            ):
                incomplete()
            certificate_by_ref = {
                certificate.component.component_ref: certificate
                for certificate in component_certificates
            }
            if len(certificate_by_ref) != len(component_certificates):
                incomplete()
            if not set(component_use_refs).issubset(certificate_by_ref):
                incomplete()
            for certificate in component_certificates:
                component = certificate.component
                try:
                    atom_facts = plan_values(
                        component.atom_facts,
                        MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS,
                    )
                    owner_refs = plan_values(
                        component.atom_refs,
                        MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS,
                    )
                    terminal_refs = plan_values(
                        component.atom_terminal_conductor_refs,
                        MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS,
                    )
                except Exception:
                    incomplete()
                expected_key = self._plan_layer_key(pdk, certificate.layer_n)
                if (
                    expected_key is None
                    or tuple(component.layer_key) != expected_key
                    or component.net != program.net
                    or component.world_digest
                    != program.routing_world_provenance_digest
                    or len(atom_facts) != len(owner_refs)
                    or len(atom_facts) != len(terminal_refs)
                    or any(
                        type(fact) is not tuple
                        or len(fact) != 8
                        or any(type(item) is not int for item in fact[:6])
                        or type(fact[6]) is not str
                        or type(fact[7]) is not str
                        or fact not in runtime_owned_facts
                        for fact in atom_facts
                    )
                    or any(type(ref) is not str or not ref for ref in owner_refs)
                    or any(
                        ref is not None and (type(ref) is not str or not ref)
                        for ref in terminal_refs
                    )
                    or any(
                        runtime_terminal_refs.get(fact, missing_terminal_ref)
                        != terminal_ref
                        for fact, terminal_ref in zip(
                            atom_facts, terminal_refs, strict=True
                        )
                    )
                ):
                    incomplete()
            access_owner_refs: Dict[str, Tuple[str, ...]] = {}
            for access in endpoint_accesses:
                if type(access.provenance) is not str or not access.provenance:
                    incomplete()
                try:
                    refs = plan_values(
                        access.owner_component_refs,
                        MAX_ENDPOINT_OWNER_COMPONENT_REFS,
                    )
                except Exception:
                    incomplete()
                if (
                    any(type(ref) is not str or not ref for ref in refs)
                    or len(refs) != len(set(refs))
                ):
                    incomplete()
                if tuple(access.layer_key) != self._plan_layer_key(pdk, access.layer_n):
                    incomplete()
                access_owner_refs[access.provenance] = tuple(refs)

            try:
                if type(pins_by_net[program.net]) not in (list, tuple):
                    incomplete()
                runtime_pins = self._bounded_plan_values(
                    pins_by_net[program.net], MAX_TOTAL_ENDPOINT_ACCESSES
                )
            except Exception:
                incomplete()
            runtime_accesses: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
            for pin in runtime_pins:
                if (
                    type(pin) is not Pin
                    or type(pin.net) is not str
                    or type(pin.device) is not str
                    or type(pin.terminal) is not str
                    or pin.net != program.net
                    or type(pin.xy_um) is not tuple
                    or len(pin.xy_um) != 2
                    or not all(type(value) in (int, float) for value in pin.xy_um)
                ):
                    incomplete()
                try:
                    provenance = pin_provenance(pin)
                except (TypeError, ValueError):
                    incomplete()
                if provenance in runtime_accesses:
                    incomplete()
                if not (
                    type(pin.layer) is tuple
                    and len(pin.layer) == 2
                    and all(type(value) is int for value in pin.layer)
                ) and type(pin.layer) is not str:
                    incomplete()
                layer_n = self._plan_layer_n(tech, pin.layer)
                layer_key = (
                    None if layer_n is None else self._plan_layer_key(pdk, layer_n)
                )
                if (
                    layer_key is None
                    or not all(math.isfinite(float(value)) for value in pin.xy_um)
                ):
                    incomplete()
                runtime_accesses[provenance] = (
                    tuple(round(float(value) * _NM_PER_UM) for value in pin.xy_um),
                    layer_key,
                )
            expected_runtime_accesses = {
                access.provenance: (tuple(access.xy_nm), tuple(access.layer_key))
                for access in endpoint_accesses
            }
            if (
                len(expected_runtime_accesses) != len(endpoint_accesses)
                or runtime_accesses != expected_runtime_accesses
            ):
                incomplete()

            seen_use_ids: Set[str] = set()
            track_by_use: Dict[str, object] = {}
            for allocation in tracks:
                if (
                    type(allocation.resource) is not TrackResource
                    or type(allocation.use_id) is not str
                    or not allocation.use_id
                ):
                    incomplete()
                resource = allocation.resource
                if allocation.use_id in seen_use_ids:
                    incomplete()
                seen_use_ids.add(allocation.use_id)
                expected_key = self._plan_layer_key(pdk, resource.key.layer_n)
                start = allocation.track_start
                stop = start + allocation.track_span
                if (
                    expected_key is None
                    or resource.gds_layer_key is None
                    or tuple(resource.gds_layer_key) != expected_key
                    or resource.routing_world_provenance_digest
                    != program.routing_world_provenance_digest
                    or tuple(resource.track_centers_nm[start:stop])
                    != tuple(allocation.selected_track_centers_nm)
                    or resource.through_interval_nm is None
                    or allocation.interval_nm[0] < resource.through_interval_nm[0]
                    or allocation.interval_nm[1] > resource.through_interval_nm[1]
                ):
                    incomplete()
                slab_bbox = self._allocated_track_bbox(program, allocation)
                layer_min_width = tech.min_width_nm.get(resource.key.layer_n)
                if (
                    resource.corridor_bbox_nm is None
                    or not (
                        resource.corridor_bbox_nm[0] <= slab_bbox[0]
                        and resource.corridor_bbox_nm[1] <= slab_bbox[1]
                        and resource.corridor_bbox_nm[2] >= slab_bbox[2]
                        and resource.corridor_bbox_nm[3] >= slab_bbox[3]
                    )
                    or layer_min_width is None
                    or allocation.width_nm < int(layer_min_width)
                ):
                    incomplete()
                track_by_use[allocation.use_id] = allocation

            neighbour_track_use_ids_by_demand: Dict[str, Set[str]] = {}
            for branch in branches:
                for step_index, step in enumerate(branch.ordered_steps):
                    if step.kind != "landing_demand":
                        continue
                    neighbours = {
                        neighbour.ref_id
                        for neighbour_index in (step_index - 1, step_index + 1)
                        if 0 <= neighbour_index < len(branch.ordered_steps)
                        for neighbour in (branch.ordered_steps[neighbour_index],)
                        if neighbour.kind == "track_use"
                    }
                    prior = neighbour_track_use_ids_by_demand.get(step.ref_id)
                    if prior is not None and prior != neighbours:
                        incomplete()
                    neighbour_track_use_ids_by_demand[step.ref_id] = neighbours

            selected = selected_by_net[program.net]
            try:
                selected_tracks = plan_values(
                    selected.selected_tracks, MAX_TOTAL_BRANCH_STEPS
                )
                selected_options = plan_values(
                    selected.selected_landing_options, MAX_TOTAL_LANDING_OPTIONS
                )
                selected_landings = plan_values(
                    selected.selected_landings, MAX_TOTAL_LANDING_OPTIONS
                )
            except Exception:
                incomplete()
            if (
                any(type(item) is not SelectedTrack for item in selected_tracks)
                or any(
                    type(item) is not SelectedLandingOption
                    for item in selected_options
                )
                or any(type(item) is not SelectedLanding for item in selected_landings)
            ):
                incomplete()
            if {
                (item.use_id, item.resource, item.track_start)
                for item in selected_tracks
            } != {
                (item.use_id, item.resource.key, item.track_start)
                for item in tracks
            }:
                incomplete()

            seen_options: Set[Tuple[str, str]] = set()
            seen_sites: Set[str] = set()
            landing_by_demand: Dict[str, List[object]] = {}
            for allocation in landing_options:
                if (
                    type(allocation.option) is not LandingOption
                    or type(allocation.demand_id) is not str
                    or not allocation.demand_id
                ):
                    incomplete()
                try:
                    allocation_sites = plan_values(
                        allocation.landing_sites, MAX_LANDING_OPTION_SITE_IDS
                    )
                    option_site_ids = plan_values(
                        allocation.option.site_ids, MAX_LANDING_OPTION_SITE_IDS
                    )
                    option_contacts = plan_values(
                        allocation.option.contacts,
                        MAX_LANDING_OPTION_CONTACTS_PER_OPTION,
                    )
                except Exception:
                    incomplete()
                if (
                    any(type(site) is not LandingSite for site in allocation_sites)
                    or any(type(site_id) is not str for site_id in option_site_ids)
                    or any(
                        type(contact) is not LandingTrackContact
                        for contact in option_contacts
                    )
                ):
                    incomplete()
                option_key = (
                    allocation.demand_id,
                    allocation.option.option_id,
                )
                if option_key in seen_options:
                    incomplete()
                seen_options.add(option_key)
                site_ids = tuple(site.site_id for site in allocation_sites)
                if (
                    allocation.option.demand_id != allocation.demand_id
                    or tuple(option_site_ids) != tuple(sorted(site_ids))
                    or allocation.actual_multiplicity != len(allocation_sites)
                ):
                    incomplete()
                sites_by_id = {site.site_id: site for site in allocation_sites}
                if len(sites_by_id) != len(allocation_sites):
                    incomplete()
                for site in allocation_sites:
                    try:
                        lower_side_refs = plan_values(
                            site.lower_owner_component_refs,
                            MAX_ENDPOINT_OWNER_COMPONENT_REFS,
                        )
                        upper_side_refs = plan_values(
                            site.upper_owner_component_refs,
                            MAX_ENDPOINT_OWNER_COMPONENT_REFS,
                        )
                    except Exception:
                        incomplete()
                    if any(
                        type(ref) is not str or not ref
                        for ref in (*lower_side_refs, *upper_side_refs)
                    ):
                        incomplete()
                    if site.site_id in seen_sites:
                        incomplete()
                    seen_sites.add(site.site_id)
                    lower_key = self._plan_layer_key(pdk, site.lower_layer_n)
                    upper_key = self._plan_layer_key(pdk, site.upper_layer_n)
                    try:
                        via_key = tuple(pdk.via_key(site.lower_layer_n))
                    except Exception:
                        via_key = None
                    if (
                        lower_key is None
                        or upper_key is None
                        or via_key is None
                        or len(via_key) != 2
                        or site.cut_bbox_nm is None
                        or site.lower_pad_bbox_nm is None
                        or site.upper_pad_bbox_nm is None
                        or site.routing_world_provenance_digest
                        != program.routing_world_provenance_digest
                        or site.owner_net not in (None, program.net)
                    ):
                        incomplete()
                    cut_geometry = (via_key, tuple(site.cut_bbox_nm))
                    prior_site = cut_site_by_geometry.get(cut_geometry)
                    if prior_site is not None and prior_site != site.site_id:
                        incomplete()
                    cut_site_by_geometry[cut_geometry] = site.site_id

                    for layer_n, refs, pad_bbox in (
                        (
                            site.lower_layer_n,
                            lower_side_refs,
                            site.lower_pad_bbox_nm,
                        ),
                        (
                            site.upper_layer_n,
                            upper_side_refs,
                            site.upper_pad_bbox_nm,
                        ),
                    ):
                        for component_ref in refs:
                            certificate = certificate_by_ref.get(component_ref)
                            if (
                                certificate is None
                                or certificate.layer_n != layer_n
                                or certificate.component.net != program.net
                                or not certificate.component.atom_facts
                                or any(
                                    fact not in runtime_owned_facts
                                    for fact in certificate.component.atom_facts
                                )
                                or len(
                                    certificate.component.atom_terminal_conductor_refs
                                ) != len(certificate.component.atom_facts)
                                or any(
                                    runtime_terminal_refs.get(
                                        fact, missing_terminal_ref
                                    ) != terminal_ref
                                    for fact, terminal_ref in zip(
                                        certificate.component.atom_facts,
                                        certificate.component.atom_terminal_conductor_refs,
                                        strict=True,
                                    )
                                )
                                or not any(
                                    _closed_rects_touch(
                                        tuple(fact[:4]), tuple(pad_bbox)
                                    )
                                    for fact in certificate.component.atom_facts
                                )
                            ):
                                incomplete()

                for contact in option_contacts:
                    site = sites_by_id.get(contact.site_id)
                    track = track_by_use.get(contact.use_id)
                    if (
                        site is None
                        or track is None
                        or track.track_start != contact.track_start
                    ):
                        incomplete()
                    if track.resource.key.layer_n == site.lower_layer_n:
                        pad_bbox = site.lower_pad_bbox_nm
                    elif track.resource.key.layer_n == site.upper_layer_n:
                        pad_bbox = site.upper_pad_bbox_nm
                    else:
                        incomplete()
                    if not _closed_rects_touch(
                        self._allocated_track_bbox(program, track),
                        tuple(pad_bbox),
                    ):
                        incomplete()
                landing_by_demand.setdefault(
                    allocation.demand_id, []
                ).append(allocation)

            derived_demand_ids = {
                demand.demand_id
                for demand in landing_demands
                if (
                    demand.selection_mode
                    == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                )
            }
            if {
                (item.demand_id, item.option_id) for item in selected_options
            } != {
                (item.demand_id, item.option.option_id)
                for item in landing_options
                if item.demand_id not in derived_demand_ids
            } or {
                (item.demand_id, item.site_id) for item in selected_landings
            } != {
                (item.demand_id, site.site_id)
                for item in landing_options
                if item.demand_id in derived_demand_ids
                for site in item.landing_sites
            }:
                incomplete()

            # Independent replay of the self-contained closed program.  This is
            # not candidate inference: it checks that the allocated branch stream
            # names every allocated carrier exactly and that consecutive carriers
            # plus branch endpoints touch in the emitted geometry.
            if not branches:
                incomplete()
            accesses = {
                access.provenance: access for access in endpoint_accesses
            }
            if len(accesses) != len(endpoint_accesses):
                incomplete()
            branch_endpoints = {
                endpoint
                for branch in branches
                for endpoint in (branch.source_endpoint, branch.target_endpoint)
            }
            if set(accesses) != branch_endpoints:
                incomplete()
            if program.program_kind == "POWER_SPINE_LOCAL_DROPS":
                root_provenances = {
                    branch.source_endpoint for branch in branches
                }
                first_steps = tuple(
                    branch.ordered_steps[0] for branch in branches
                )
                rail_component_refs = {
                    step.ref_id
                    for step in first_steps
                    if step.kind == "owned_component"
                }
                if (
                    len(root_provenances) != 1
                    or len(rail_component_refs) != 1
                    or any(
                        step.kind != "owned_component"
                        for step in first_steps
                    )
                ):
                    incomplete()
                root_access = accesses.get(next(iter(root_provenances)))
                certificate = certificate_by_ref.get(
                    next(iter(rail_component_refs))
                )
                if root_access is None or certificate is None:
                    incomplete()
                root_x, root_y = root_access.xy_nm
                component = certificate.component
                if not any(
                    type(fact) is tuple
                    and len(fact) == 8
                    and tuple(fact[4:6]) == tuple(component.layer_key)
                    and fact[6] == program.net
                    and fact[7] == "rail"
                    and fact[0] <= root_x <= fact[2]
                    and fact[1] <= root_y <= fact[3]
                    and min(fact[2] - fact[0], fact[3] - fact[1])
                    >= program.width_nm
                    for fact in component.atom_facts
                ):
                    incomplete()
            declared_demands = {
                demand.demand_id: demand for demand in landing_demands
            }
            if (
                len(declared_demands) != len(landing_demands)
                or set(declared_demands) != set(landing_by_demand)
            ):
                incomplete()
            for demand_id, demand in declared_demands.items():
                allocations = landing_by_demand[demand_id]
                access = accesses.get(demand.terminal_provenance)
                allocated_sites = tuple(
                    site
                    for allocation in allocations
                    for site in allocation.landing_sites
                )
                site_ids = {site.site_id for site in allocated_sites}
                expected_option_count = (
                    demand.required_multiplicity
                    if demand.selection_mode
                    in {
                        LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY,
                        LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
                    }
                    else 1
                )
                try:
                    allowed_site_ids = plan_values(
                        demand.allowed_site_ids, MAX_LANDING_OPTION_SITE_IDS
                    )
                except Exception:
                    incomplete()
                if (
                    any(
                        type(site_id) is not str or not site_id
                        for site_id in allowed_site_ids
                    )
                    or len(allowed_site_ids) != len(set(allowed_site_ids))
                ):
                    incomplete()
                if (
                    access is None
                    or demand.terminal_provenance not in branch_endpoints
                    or len(allocations) != expected_option_count
                    or (
                        demand.selection_mode
                        in {
                            LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY,
                            LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
                        }
                        and (
                            len(allocated_sites)
                            != demand.required_multiplicity
                            or any(
                                allocation.actual_multiplicity != 1
                                for allocation in allocations
                            )
                        )
                    )
                    or len(allocated_sites) < max(
                        demand.required_multiplicity, program.via_multiplicity
                    )
                    or not site_ids.issubset(set(allowed_site_ids))
                ):
                    incomplete()
                if (
                    demand.selection_mode
                    in {
                        LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY,
                        LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
                    }
                ):
                    expected_use_ids = (
                        neighbour_track_use_ids_by_demand.get(demand_id)
                    )
                    if expected_use_ids is None:
                        incomplete()
                    for allocation in allocations:
                        if len(allocation.landing_sites) != 1:
                            incomplete()
                        atom_site_id = allocation.landing_sites[0].site_id
                        contacts = allocation.option.contacts
                        if (
                            len(contacts) != len(expected_use_ids)
                            or {contact.use_id for contact in contacts}
                            != expected_use_ids
                            or {contact.site_id for contact in contacts}
                            != ({atom_site_id} if contacts else set())
                        ):
                            incomplete()
            referenced_tracks: Set[str] = set()
            referenced_demands: Set[str] = set()
            referenced_components: Set[str] = set()

            def carrier_rects(step) -> Tuple[Tuple[int, Rect], ...]:
                if step.kind == "track_use":
                    allocation = track_by_use.get(step.ref_id)
                    if allocation is None:
                        incomplete()
                    return ((
                        allocation.resource.key.layer_n,
                        self._allocated_track_bbox(program, allocation),
                    ),)
                if step.kind == "landing_demand":
                    allocations = landing_by_demand.get(step.ref_id)
                    if allocations is None:
                        incomplete()
                    rects: List[Tuple[int, Rect]] = []
                    for allocation in allocations:
                        for site in allocation.landing_sites:
                            rects.append((
                                site.lower_layer_n, tuple(site.lower_pad_bbox_nm)
                            ))
                            rects.append((
                                site.upper_layer_n, tuple(site.upper_pad_bbox_nm)
                            ))
                    return tuple(rects)
                if step.kind == "owned_component":
                    if step.ref_id not in program.owned_component_use_refs:
                        incomplete()
                    certificate = certificate_by_ref.get(step.ref_id)
                    if certificate is None or not certificate.component.atom_facts:
                        incomplete()
                    return tuple(
                        (certificate.layer_n, tuple(fact[:4]))
                        for fact in certificate.component.atom_facts
                    )
                incomplete()

            def carriers_touch(left, right) -> bool:
                return any(
                    left_layer == right_layer
                    and _closed_rects_touch(left_bbox, right_bbox)
                    for left_layer, left_bbox in carrier_rects(left)
                    for right_layer, right_bbox in carrier_rects(right)
                )

            def carrier_contains(step, access) -> bool:
                if (
                    step.kind == "owned_component"
                    and step.ref_id not in access.owner_component_refs
                ):
                    return False
                x, y = access.xy_nm
                return any(
                    layer_n == access.layer_n
                    and bbox[0] <= x <= bbox[2]
                    and bbox[1] <= y <= bbox[3]
                    for layer_n, bbox in carrier_rects(step)
                )

            branch_ids: Set[str] = set()
            for branch in branches:
                try:
                    branch_steps = plan_values(
                        branch.ordered_steps, MAX_CANDIDATE_BRANCH_STEPS
                    )
                except Exception:
                    incomplete()
                if any(type(step) is not CandidateBranchStep for step in branch_steps):
                    incomplete()
                if branch.branch_id in branch_ids or not branch_steps:
                    incomplete()
                branch_ids.add(branch.branch_id)
                for step in branch_steps:
                    if step.kind == "track_use":
                        referenced_tracks.add(step.ref_id)
                    elif step.kind == "landing_demand":
                        referenced_demands.add(step.ref_id)
                    elif step.kind == "owned_component":
                        referenced_components.add(step.ref_id)
                    else:
                        incomplete()
                for endpoint, boundary_step in (
                    (branch.source_endpoint, branch_steps[0]),
                    (branch.target_endpoint, branch_steps[-1]),
                ):
                    if boundary_step.kind != "landing_demand":
                        continue
                    demand = declared_demands.get(boundary_step.ref_id)
                    allocations = landing_by_demand.get(boundary_step.ref_id)
                    access = accesses[endpoint]
                    if (
                        demand is None
                        or allocations is None
                        or demand.terminal_provenance != endpoint
                    ):
                        incomplete()
                    try:
                        contacts = tuple(
                            contact
                            for allocation in allocations
                            for contact in plan_values(
                                allocation.option.contacts,
                                MAX_LANDING_OPTION_CONTACTS_PER_OPTION,
                            )
                        )
                    except Exception:
                        incomplete()
                    boundary_reached = False
                    for site in (
                        site
                        for allocation in allocations
                        for site in allocation.landing_sites
                    ):
                        if tuple(site.xy_nm) != tuple(access.xy_nm):
                            continue
                        if access.layer_n == site.lower_layer_n:
                            side_refs = tuple(site.lower_owner_component_refs)
                        elif access.layer_n == site.upper_layer_n:
                            side_refs = tuple(site.upper_owner_component_refs)
                        else:
                            continue
                        evidence_keys = {
                            tuple(certificate_by_ref[ref].component.layer_key)
                            for ref in side_refs
                            if ref in certificate_by_ref
                        }
                        evidence_keys.update(
                            tuple(track_by_use[contact.use_id].resource.gds_layer_key)
                            for contact in contacts
                            if contact.site_id == site.site_id
                            and contact.use_id in track_by_use
                            and track_by_use[contact.use_id].resource.key.layer_n
                            == access.layer_n
                        )
                        access_refs = set(access_owner_refs[access.provenance])
                        if tuple(access.layer_key) not in evidence_keys:
                            continue
                        if access_refs and not (
                            side_refs and access_refs.intersection(side_refs)
                        ):
                            continue
                        boundary_reached = True
                        break
                    if not boundary_reached:
                        incomplete()
                if not carrier_contains(
                    branch_steps[0], accesses[branch.source_endpoint]
                ) or not carrier_contains(
                    branch_steps[-1], accesses[branch.target_endpoint]
                ):
                    incomplete()
                if any(
                    not carriers_touch(left, right)
                    for left, right in zip(
                        branch_steps, branch_steps[1:]
                    )
                ):
                    incomplete()
            if (
                referenced_tracks != set(track_by_use)
                or referenced_demands != set(landing_by_demand)
                or referenced_components != set(component_use_refs)
            ):
                incomplete()

            goals_by_ref = {
                goal.declaration_ref: goal for goal in required_goals
            }
            coverage_by_ref = {
                item.declaration_ref: item for item in goal_coverage
            }
            if (
                len(goals_by_ref) != len(required_goals)
                or len(coverage_by_ref) != len(goal_coverage)
                or set(goals_by_ref) != set(coverage_by_ref)
                or len({item.branch_id for item in goal_coverage})
                != len(goal_coverage)
            ):
                incomplete()
            branches_by_id = {branch.branch_id: branch for branch in branches}
            for declaration_ref, goal in goals_by_ref.items():
                coverage = coverage_by_ref[declaration_ref]
                branch = branches_by_id.get(coverage.branch_id)
                if (
                    goal.net != program.net
                    or goal.target is None
                    or coverage.target_provenance != goal.target.provenance
                    or branch is None
                    or coverage.target_provenance != branch.target_endpoint
                    or accesses.get(goal.target.provenance) is None
                    or tuple(accesses[goal.target.provenance].xy_nm)
                    != tuple(goal.target.xy_nm)
                    or tuple(accesses[goal.target.provenance].layer_key)
                    != tuple(goal.target.layer)
                ):
                    incomplete()

        # Hard program/materialization contracts are checked before drawing a
        # single rectangle, so mismatch is fail-closed rather than partial emit.
        unsatisfied: List[RouteConstraint] = []
        mirror_constraints: List[Mirror] = []
        length_constraints: List[Length] = []
        for constraint in bounded_constraints:
            if getattr(constraint, "hardness", "hard") != "hard":
                continue
            if isinstance(constraint, Width):
                program = programs_by_net.get(constraint.net)
                expected_nm = round(float(constraint.width_um) * _NM_PER_UM)
                if program is None and constraint.net in unplanned_pin_nets:
                    continue
                if program is None or program.width_nm < expected_nm:
                    unsatisfied.append(constraint)
            elif isinstance(constraint, LayerSet):
                program = programs_by_net.get(constraint.net)
                if program is None and constraint.net in unplanned_pin_nets:
                    continue
                allowed = {
                    layer_n
                    for layer in constraint.layers
                    if (layer_n := self._plan_layer_n(tech, layer)) is not None
                }
                used: Set[int] = set()
                if program is not None:
                    used.update(track.resource.key.layer_n for track in program.tracks)
                    for allocation in program.landing_options:
                        for site in allocation.landing_sites:
                            used.add(site.lower_layer_n)
                            used.add(site.upper_layer_n)
                    certificate_by_ref = {
                        certificate.component.component_ref: certificate
                        for certificate in program.owned_conductor_components
                    }
                    used.update(
                        certificate_by_ref[component_ref].layer_n
                        for component_ref in program.owned_component_use_refs
                    )
                if program is None or not allowed or not used.issubset(allowed):
                    unsatisfied.append(constraint)
            elif isinstance(constraint, Mirror):
                mirror_constraints.append(constraint)
            elif isinstance(constraint, Length):
                length_constraints.append(constraint)
            elif isinstance(constraint, Spacing):
                # The exact replay path checks the actual allocated geometry.
                if (constraint.hardness == "hard"
                        and spacing_nm(constraint, pins_by_net) is None):
                    unsatisfied.append(constraint)
            elif getattr(constraint, "kind", None) not in _HANDLED_KINDS:
                unsatisfied.append(constraint)

        mirror_by_b = self._index_mirrors(
            mirror_constraints, pins_by_net, unsatisfied
        )

        out = RoutedCell()
        self._group_route_required_states = {}
        self._layered_portal_wire_states = set()
        self._group_route_landing_baselines = {}
        out.mirror_reports = []
        out.length_reports = []
        out.advisory_residuals = 0
        out.advisory_residual_pairs = []
        out.deck_residual_pairs = []
        for net in sorted(pins_by_net):
            out.nets[net] = NetRoute(
                net=net,
                pins=list(pins_by_net[net]),
                connected=net in unplanned_pin_nets,
            )
        out.unsatisfied = unsatisfied

        geometry_records: List[Tuple[str, int, int, Rect]] = []
        if not unsatisfied:
            for program in programs:
                route = out.nets[program.net]
                for allocation in program.tracks:
                    run0, run1 = allocation.interval_nm
                    bbox = self._allocated_track_bbox(program, allocation)
                    key = tuple(allocation.resource.gds_layer_key)
                    route.wires.append(_rect(key, *bbox))
                    geometry_records.append((program.net, key[0], key[1], bbox))
                    route.length_um += float(run1 - run0) / _NM_PER_UM

                for allocation in program.landing_options:
                    for site in allocation.landing_sites:
                        lower_key = self._plan_layer_key(pdk, site.lower_layer_n)
                        upper_key = self._plan_layer_key(pdk, site.upper_layer_n)
                        via_key = tuple(pdk.via_key(site.lower_layer_n))
                        emitted = (
                            (lower_key, tuple(site.lower_pad_bbox_nm)),
                            (upper_key, tuple(site.upper_pad_bbox_nm)),
                            (via_key, tuple(site.cut_bbox_nm)),
                        )
                        for key, bbox in emitted:
                            route.wires.append(_rect(key, *bbox))
                            geometry_records.append(
                                (program.net, int(key[0]), int(key[1]), bbox)
                            )
                route.connected = True

        # Selection is global, but a hard matched pair is accepted only when
        # the exact selected track/landing programs form one reflected geometry
        # template.  This binds turns, layer transitions, accesses, and via
        # arrays together instead of silently treating two independent winners
        # as a satisfied Mirror constraint.
        for partner_name, mirror in sorted(mirror_by_b.items()):
            report = self._materialized_mirror_report(
                mirror,
                out.nets[str(mirror.net_a)],
                out.nets[partner_name],
            )
            out.mirror_reports.append(report)
            if not report["satisfied"] and mirror not in unsatisfied:
                unsatisfied.append(mirror)

        for constraint in sorted(
            length_constraints,
            key=lambda item: (str(item.net_a), str(item.net_b)),
        ):
            report = self._materialized_length_report(constraint, out.nets)
            out.length_reports.append(report)
            if not report["satisfied"] and constraint not in unsatisfied:
                unsatisfied.append(constraint)

        out.spacing_reports = check_spacing_geometry(
            out, constraints, obstacles, tech, rails
        )
        short_records: List[Tuple[str, str, Tuple[int, int], Rect, Rect]] = []
        pair_evaluations = 0
        for index, left in enumerate(geometry_records):
            for right in geometry_records[index + 1:]:
                if left[0] == right[0] or left[1:3] != right[1:3]:
                    continue
                pair_evaluations += 1
                if pair_evaluations > MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS:
                    incomplete()
                if _closed_rects_touch(left[3], right[3]):
                    left_net, right_net = sorted((left[0], right[0]))
                    short_records.append((
                        left_net,
                        right_net,
                        (left[1], left[2]),
                        left[3],
                        right[3],
                    ))
        out.shorts = sorted(set(short_records))
        if out.shorts:
            shorted_nets = {
                net for record in out.shorts for net in record[:2]
            }
            for net in shorted_nets:
                out.nets[net].connected = False

        geometry_digest = _allocation_geometry_digest(geometry_records)
        from clay.layout.route.resource_world import pdk_provenance_digest

        materialization_complete = bool(
            not unsatisfied
            and not out.shorts
            and all(route.connected for route in out.nets.values())
        )
        if diagnostic:
            # A diagnostic program deliberately stops before the exact replay
            # issuer.  Geometry and its digest remain available to the external
            # LVS/DRC oracle, but no allocation/signoff capability is minted.
            out.geometry_digest = geometry_digest
            out.diagnostic_only = True
            out.signoff_eligible = False
            out.allocation_replay_verified = False
            out.success = False
            return out

        out.resource_plan = resource_plan
        out.candidate_pool_hash = resource_plan.candidate_pool_hash
        out.resource_plan_hash = resource_plan.sha256()
        out.geometry_digest = geometry_digest
        out.allocation_replay_verified = False
        out.success = False
        if not materialization_complete:
            # Failed exact materialization remains inspectable through its
            # plan/hash/geometry/constraint evidence, but it cannot mint or
            # register an allocation replay capability.
            return out

        # Issue the replay capability only here, after the runtime pins, owned
        # conductors, branch continuity, constraints, emitted rectangles, and
        # cross-net shorts have all been checked above.  There is deliberately
        # no module-level claim-only issuer that a caller can import.
        certificate = object.__new__(GridlessAllocationReplayCertificate)
        certificate_values = (
            ("candidate_pool_hash", resource_plan.candidate_pool_hash),
            ("resource_plan_hash", resource_plan.sha256()),
            ("selected_candidates", selected_pairs),
            ("geometry_digest", geometry_digest),
            ("materialization_complete", True),
            ("current_pdk_provenance_digest", pdk_provenance_digest(pdk)),
            ("routing_world_provenance_digests", tuple(sorted({
                program.routing_world_provenance_digest for program in programs
            }))),
            ("verification_scope", (
                "exact allocated-program replay + runtime pins + runtime owned "
                "obstacles + current PDK rules + emitted geometry; full routing-"
                "world digest is externally preverified by route_cell"
            )),
            ("_issuer_authority", _GRIDLESS_REPLAY_ISSUER_AUTHORITY),
        )
        for name, value in certificate_values:
            object.__setattr__(certificate, name, value)
        receipt_id = id(certificate)

        def discard_replay_receipt(
            receipt_ref: weakref.ReferenceType[Any],
        ) -> None:
            current = _GRIDLESS_REPLAY_RECEIPTS.get(receipt_id)
            if current is not None and current[0] is receipt_ref:
                _GRIDLESS_REPLAY_RECEIPTS.pop(receipt_id, None)

        certificate_ref = weakref.ref(certificate, discard_replay_receipt)
        _GRIDLESS_REPLAY_RECEIPTS[receipt_id] = (
            certificate_ref,
            _gridless_replay_receipt_payload(certificate),
        )
        _verify_gridless_allocation_replay_certificate(certificate)
        out.resource_plan_hash = certificate.resource_plan_hash
        out.allocation_replay_certificate = certificate
        out.materialization_certificate = certificate
        out.allocation_replay_verified = True
        out.success = True
        return out

