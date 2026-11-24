"""Deterministic, report-only routing-resource observations.

This module is the geometry half of the C4 shadow ledger.  It deliberately has
no router or placer integration: a final placement's integer-nanometre cluster
boxes are converted into exact occupied/whitespace accounting and
coordinate-compressed free-space components.  A component touching the core
boundary is a boundary corridor; every other component is an enclosed void.

The nominal track counts reported here are observations only.  Without a
proved set of net-demand intervals or landing obligations, whitespace geometry
cannot establish routing infeasibility and must not manufacture a Benders cut.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field, replace
from fractions import Fraction
import hashlib
from itertools import islice
import json
import math
from typing import Mapping, Optional, Sequence
import weakref

from clay.layout.route.resource_conductors import (
    OwnedConductorComponent,
    OwnedConductorIndexError,
    OwnedConductorIndexLimitError,
    validate_owned_conductor_component_index_certificate,
)


BBoxNm = tuple[int, int, int, int]
PointNm = tuple[int, int]
LayerKey = tuple[int, int]

MAX_TOPOLOGY_PLAN_DEMANDS = 50
MAX_TOPOLOGY_PLAN_CANDIDATES = 400
MAX_TOPOLOGY_BRANCH_TARGETS = 512
MAX_TOPOLOGY_BRANCH_TERMINALS = 512
MAX_TOPOLOGY_REQUIRED_GOAL_REFS = 512
MAX_TOPOLOGY_LAYERS = 32
MAX_TOPOLOGY_DECLARED_EDGE_TRUNKS = 64


def _bounded_materialize(
    values: object,
    *,
    limit: int,
    label: str,
    error_type: type[ValueError] = ValueError,
) -> tuple[object, ...]:
    if type(limit) is not int or limit < 0:
        raise TypeError(f"{label} limit must be an exact non-negative integer")
    try:
        count = len(values)  # type: ignore[arg-type]
    except TypeError:
        count = None
    if count is not None and count > limit:
        raise error_type(f"{label} count {count} exceeds {limit}")
    materialized = tuple(islice(iter(values), limit + 1))  # type: ignore[arg-type]
    if len(materialized) > limit:
        raise error_type(f"{label} count exceeds {limit}")
    return materialized


@dataclass(frozen=True)
class TerminalDemand:
    """One routable terminal expressed in exact integer-nanometre coordinates."""

    xy_nm: PointNm
    layer: LayerKey
    provenance: str

    def __post_init__(self) -> None:
        if (
            len(self.xy_nm) != 2
            or not all(_is_int_nm(value) for value in self.xy_nm)
        ):
            raise ValueError(f"terminal point must be integer nm, got {self.xy_nm!r}")
        if (
            len(self.layer) != 2
            or not all(_is_int_nm(value) for value in self.layer)
        ):
            raise ValueError(f"terminal layer must be an integer key, got {self.layer!r}")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("terminal provenance must be a non-empty string")


@dataclass(frozen=True)
class RequiredRouteGoal:
    """A named route branch that must survive materialization and connectivity.

    ``target=None`` is first-class: it records that a declaration exists but its
    physical access could not be materialized, rather than silently deleting the
    obligation.
    """

    kind: str
    net: str
    target: Optional[TerminalDemand]
    declaration_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("required route goal kind must be a non-empty string")
        if not isinstance(self.net, str):
            raise ValueError("required route goal net must be a string")
        if not isinstance(self.declaration_ref, str) or not self.declaration_ref:
            raise ValueError("required route goal declaration_ref must be non-empty")


@dataclass(frozen=True)
class RoutingDemand:
    """Immutable declaration-plus-terminal demand for one electrical net."""

    net: str
    functional_role: str
    terminals: tuple[TerminalDemand, ...] = ()
    required_goals: tuple[RequiredRouteGoal, ...] = ()
    declared_trunk_layers: tuple[int, ...] = ()
    declared_edge_trunks: tuple[str, ...] = ()
    declared_power_plane_layers: tuple[int, ...] = ()
    declared_topology: Optional[str] = None
    declared_width_nm: Optional[int] = None
    declared_current_ma: Optional[float] = None
    declared_via_multiplicity: Optional[int] = None
    # Appended after the original public fields to preserve positional ABI.
    declared_allowed_layers: tuple[int, ...] = ()
    declared_max_drop_v: Optional[float] = None
    declared_max_resistance_ohm: Optional[float] = None
    declared_max_capacitance_f: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.net, str):
            raise ValueError("routing demand net must be a string")
        if not isinstance(self.functional_role, str) or not self.functional_role:
            raise ValueError("routing demand functional_role must be non-empty")
        terminals = _bounded_materialize(
            self.terminals,
            limit=MAX_TOPOLOGY_BRANCH_TERMINALS,
            label="routing demand terminals",
        )
        required_goals = _bounded_materialize(
            self.required_goals,
            limit=MAX_TOPOLOGY_REQUIRED_GOAL_REFS,
            label="routing demand required goals",
        )
        if not all(isinstance(item, TerminalDemand) for item in terminals):
            raise ValueError("routing demand terminals contain invalid values")
        if not all(isinstance(item, RequiredRouteGoal) for item in required_goals):
            raise ValueError("routing demand required goals contain invalid values")
        for field_name in (
            "declared_trunk_layers",
            "declared_allowed_layers",
            "declared_power_plane_layers",
        ):
            values = _bounded_materialize(
                getattr(self, field_name),
                limit=MAX_TOPOLOGY_LAYERS,
                label=f"routing demand {field_name}",
            )
            if not all(_is_int_nm(value) and value > 0 for value in values):
                raise ValueError(f"{field_name} must contain positive layer integers")
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        declared_edge_trunks = _bounded_materialize(
            self.declared_edge_trunks,
            limit=MAX_TOPOLOGY_DECLARED_EDGE_TRUNKS,
            label="routing demand declared edge trunks",
        )
        object.__setattr__(self, "declared_edge_trunks", tuple(sorted(set(
            str(edge) for edge in declared_edge_trunks
        ))))
        object.__setattr__(self, "terminals", tuple(sorted(
            terminals,
            key=lambda terminal: (
                terminal.xy_nm,
                terminal.layer,
                terminal.provenance,
            ),
        )))
        object.__setattr__(self, "required_goals", tuple(sorted(
            required_goals,
            key=_required_goal_sort_key,
        )))
        if self.declared_topology is not None and self.declared_topology not in {
            "local_signal",
            "shared_signal_trunk",
            "power_spine_tree",
        }:
            raise ValueError(f"unknown declared topology {self.declared_topology!r}")
        if self.declared_width_nm is not None and (
            not _is_int_nm(self.declared_width_nm) or self.declared_width_nm <= 0
        ):
            raise ValueError("declared_width_nm must be a positive integer nm")
        if self.declared_via_multiplicity is not None and (
            not _is_int_nm(self.declared_via_multiplicity)
            or self.declared_via_multiplicity <= 0
        ):
            raise ValueError("declared_via_multiplicity must be a positive integer")
        for field_name in (
            "declared_current_ma",
            "declared_max_drop_v",
            "declared_max_resistance_ohm",
            "declared_max_capacitance_f",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{field_name} must be a positive finite float")


@dataclass(frozen=True)
class TopologyCandidate:
    """One deterministic topology option with its PDK/declaration demand facts."""

    net: str
    topology_class: str
    branch_targets: tuple[PointNm, ...]
    layers: tuple[int, ...]
    width_nm: int
    via_multiplicity: int
    branch_terminals: tuple[TerminalDemand, ...] = ()
    required_goal_refs: tuple[str, ...] = ()
    # PDK-derived transition/transport closure. ``layers`` remains the semantic
    # trunk demand; every physical corridor layer must be named by one of these
    # two signed fields rather than being discovered implicitly by a pattern.
    transport_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.net, str):
            raise ValueError("topology candidate net must be a string")
        if self.topology_class not in {
            "local_signal",
            "shared_signal_trunk",
            "power_spine_tree",
        }:
            raise ValueError(f"unknown topology class {self.topology_class!r}")
        if not _is_int_nm(self.width_nm) or self.width_nm <= 0:
            raise ValueError("candidate width_nm must be a positive integer nm")
        if not _is_int_nm(self.via_multiplicity) or self.via_multiplicity < 0:
            raise ValueError("candidate via_multiplicity must be a non-negative integer")
        layers = _bounded_materialize(
            self.layers,
            limit=MAX_TOPOLOGY_LAYERS,
            label="topology candidate layers",
        )
        if not layers or not all(
            _is_int_nm(layer_n) and layer_n > 0 for layer_n in layers
        ):
            raise ValueError("candidate layers must contain positive layer integers")
        points = _bounded_materialize(
            self.branch_targets,
            limit=MAX_TOPOLOGY_BRANCH_TARGETS,
            label="topology candidate branch targets",
        )
        if any(len(point) != 2 or not all(_is_int_nm(v) for v in point) for point in points):
            raise ValueError("candidate branch targets must be integer-nm points")
        branch_terminals = _bounded_materialize(
            self.branch_terminals,
            limit=MAX_TOPOLOGY_BRANCH_TERMINALS,
            label="topology candidate branch terminals",
        )
        if not all(isinstance(item, TerminalDemand) for item in branch_terminals):
            raise ValueError("candidate branch_terminals contain invalid values")
        raw_refs = _bounded_materialize(
            self.required_goal_refs,
            limit=MAX_TOPOLOGY_REQUIRED_GOAL_REFS,
            label="topology candidate required goal refs",
        )
        object.__setattr__(self, "branch_targets", tuple(sorted(points)))
        object.__setattr__(self, "layers", tuple(sorted(set(layers))))
        transport_layers = _bounded_materialize(
            self.transport_layers,
            limit=MAX_TOPOLOGY_LAYERS,
            label="topology candidate transport layers",
        )
        if not all(
            _is_int_nm(layer_n) and layer_n > 0
            for layer_n in transport_layers
        ):
            raise ValueError(
                "candidate transport_layers must contain positive layer integers"
            )
        object.__setattr__(
            self, "transport_layers", tuple(sorted(set(transport_layers)))
        )
        object.__setattr__(self, "branch_terminals", tuple(sorted(
            branch_terminals,
            key=lambda terminal: (
                terminal.xy_nm,
                terminal.layer,
                terminal.provenance,
            ),
        )))
        refs = tuple(str(ref) for ref in raw_refs)
        if any(not ref for ref in refs):
            raise ValueError("candidate required_goal_refs must be non-empty strings")
        object.__setattr__(self, "required_goal_refs", tuple(sorted(refs)))


@dataclass(frozen=True)
class TopologyPlan:
    """Byte-deterministic shadow routing-demand and topology-candidate plan."""

    demands: tuple[RoutingDemand, ...]
    candidates: tuple[TopologyCandidate, ...]

    def __post_init__(self) -> None:
        demands = _bounded_materialize(
            self.demands,
            limit=MAX_TOPOLOGY_PLAN_DEMANDS,
            label="topology plan demands",
        )
        candidates = _bounded_materialize(
            self.candidates,
            limit=MAX_TOPOLOGY_PLAN_CANDIDATES,
            label="topology plan candidates",
        )
        if not all(isinstance(item, RoutingDemand) for item in demands):
            raise ValueError("topology plan demands contain invalid values")
        if not all(isinstance(item, TopologyCandidate) for item in candidates):
            raise ValueError("topology plan candidates contain invalid values")
        ordered_demands = tuple(sorted(demands, key=lambda demand: demand.net))
        names = [demand.net for demand in ordered_demands]
        if len(names) != len(set(names)):
            raise ValueError("topology plan contains duplicate net demands")
        object.__setattr__(self, "demands", ordered_demands)
        object.__setattr__(self, "candidates", tuple(sorted(
            candidates,
            key=_topology_candidate_sort_key,
        )))

    def by_net(self, net: str) -> RoutingDemand:
        """Return the uniquely named demand using a stable sorted lookup."""

        for demand in self.demands:
            if demand.net == net:
                return demand
        raise KeyError(net)

    def candidates_for(self, net: str) -> tuple[TopologyCandidate, ...]:
        """Return candidates for ``net`` in deterministic preference order."""

        return tuple(candidate for candidate in self.candidates if candidate.net == net)

    def canonical_json(self) -> str:
        """Canonical declaration/terminal/topology bytes for provenance binding."""

        return json.dumps(
            _topology_plan_payload(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def required_goals(self) -> tuple[RequiredRouteGoal, ...]:
        """All independent required goals, ordered by net and declaration."""

        return tuple(
            goal
            for demand in self.demands
            for goal in demand.required_goals
        )


def _required_goal_sort_key(goal: RequiredRouteGoal) -> tuple[object, ...]:
    target = goal.target
    return (
        goal.kind,
        goal.net,
        goal.declaration_ref,
        target is None,
        () if target is None else target.xy_nm,
        () if target is None else target.layer,
        "" if target is None else target.provenance,
    )


def _terminal_demand_payload(terminal: TerminalDemand) -> dict[str, object]:
    return {
        "layer": list(terminal.layer),
        "provenance": terminal.provenance,
        "xy_nm": list(terminal.xy_nm),
    }


def _required_route_goal_payload(goal: RequiredRouteGoal) -> dict[str, object]:
    return {
        "declaration_ref": goal.declaration_ref,
        "kind": goal.kind,
        "net": goal.net,
        "target": (
            None if goal.target is None else _terminal_demand_payload(goal.target)
        ),
    }


def _topology_candidate_payload(
    candidate: TopologyCandidate,
) -> dict[str, object]:
    return {
        "branch_targets": [list(point) for point in candidate.branch_targets],
        "branch_terminals": [
            _terminal_demand_payload(terminal)
            for terminal in candidate.branch_terminals
        ],
        "layers": list(candidate.layers),
        "net": candidate.net,
        "required_goal_refs": list(candidate.required_goal_refs),
        "topology_class": candidate.topology_class,
        "transport_layers": list(candidate.transport_layers),
        "via_multiplicity": candidate.via_multiplicity,
        "width_nm": candidate.width_nm,
    }


def _topology_candidate_sort_key(candidate: TopologyCandidate) -> str:
    return json.dumps(
        _topology_candidate_payload(candidate),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _topology_plan_payload(plan: TopologyPlan) -> dict[str, object]:
    return {
        "candidates": [
            _topology_candidate_payload(candidate)
            for candidate in plan.candidates
        ],
        "demands": [
            {
                "declared_current_ma": demand.declared_current_ma,
                "declared_max_capacitance_f": demand.declared_max_capacitance_f,
                "declared_max_drop_v": demand.declared_max_drop_v,
                "declared_max_resistance_ohm": demand.declared_max_resistance_ohm,
                "declared_allowed_layers": list(
                    demand.declared_allowed_layers
                ),
                "declared_edge_trunks": list(demand.declared_edge_trunks),
                "declared_power_plane_layers": list(
                    demand.declared_power_plane_layers
                ),
                "declared_topology": demand.declared_topology,
                "declared_trunk_layers": list(demand.declared_trunk_layers),
                "declared_via_multiplicity": demand.declared_via_multiplicity,
                "declared_width_nm": demand.declared_width_nm,
                "functional_role": demand.functional_role,
                "net": demand.net,
                "required_goals": [
                    _required_route_goal_payload(goal)
                    for goal in demand.required_goals
                ],
                "terminals": [
                    _terminal_demand_payload(terminal)
                    for terminal in demand.terminals
                ],
            }
            for demand in plan.demands
        ],
        "schema": "clay.topology-plan.v2",
    }


# C5 complete-net candidate selection is deliberately separate from the nominal,
# demand-free C4 observations below.  TrackResource instances can only be supplied
# explicitly by a caller that has established legal through-going tracks; no code in
# this module converts CapacityObservation.nominal_track_capacity into TrackResource.

MAX_REQUIRED_NETS = 50
MAX_CANDIDATES_PER_NET = 512
MAX_TOTAL_CANDIDATES = 8_192
MAX_TOTAL_CORRIDOR_USES = 4_096
# Matching replay uses a deterministic augmenting-path search.  Keeping this
# below Python's recursion ceiling makes the declared bound an executable one,
# rather than a nominal number that could fail before returning a typed error.
MAX_TOTAL_LANDING_DEMANDS = 4_096
# The exact lazy landing domain may project one disjunctive choice per explicit
# site.  Keep aggregate solver bounds aligned with the public 32k site domain
# (including redundant two-via options and two adjacent track contacts), while
# retaining a finite hostile-input ceiling.
MAX_TOTAL_ASSIGNMENT_ARCS = 524_288
MAX_PAIR_CONFLICTS = 16_384
MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS = 200_000
MAX_DERIVED_TRACK_CONFLICT_CLAUSES = 100_000
MAX_LANDING_SITE_PAIR_CONFLICTS = 100_000
MAX_LANDING_CONFLICT_NET_CLAUSES = 200_000
MAX_LANDING_CONFLICT_DOMAIN_LAYERS = 64
MAX_LAZY_LANDING_SITE_CONFLICTS = 4_096
MAX_LAZY_LANDING_SELECTED_PAIR_EVALUATIONS = 100_000
MAX_TOTAL_CANDIDATE_BRANCHES = 4_096
MAX_TOTAL_BRANCH_STEPS = 16_384
MAX_TOTAL_LANDING_OPTION_TEMPLATES = 32_768
MAX_TOTAL_LANDING_OPTIONS = 65_536
MAX_TOTAL_LANDING_OPTION_SITE_CLAIMS = 131_072
MAX_TOTAL_LANDING_OPTION_CONTACTS = 262_144
MAX_TOTAL_GOAL_BRANCH_COVERAGE = 4_096
MAX_TOTAL_ENDPOINT_ACCESSES = 8_192
MAX_TRACK_RESOURCES = 8_192
MAX_LANDING_SITES = 32_768
MAX_REQUIRED_GOAL_REFS = 4_096
MAX_REQUIRED_ROUTE_GOALS = 4_096
MAX_TRACK_CENTERS_PER_RESOURCE = 32_768
MAX_TOTAL_BRANCH_EDGES = 16_384
MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES = 200_000
MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS = 2_000_000
MAX_OWNED_CONDUCTOR_COMPONENT_PAIR_EVALUATIONS = 2_000_000
MAX_OWNED_CONDUCTOR_COMPONENT_SPATIAL_REPLAY_EVALUATIONS = 2_000_000
MAX_TOTAL_OWNED_COMPONENT_USE_REFS = 2_000_000
# Pool-wide hostile-input ceilings for nested resource and endpoint domains.
# These do not replace the existing per-record caps; rebuilding enforces both
# and stops on the first aggregate cap+1 element.
MAX_TOTAL_TRACK_CENTERS = MAX_TOTAL_ASSIGNMENT_ARCS
MAX_TOTAL_ENDPOINT_OWNER_COMPONENT_REFS = (
    MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS
)

# Per-record bounds are intentionally distinct from the aggregate pool bounds.
# Every iterable is cut off at cap+1 before sorting, tuple conversion, or semantic
# traversal, so hostile generators cannot make an allegedly bounded IR unbounded.
MAX_CORRIDOR_USE_ALLOWED_TRACK_STARTS = 32_768
MAX_LANDING_DEMAND_ALLOWED_SITES = 32_768
MAX_CANDIDATE_BRANCH_ORDERED_USES = 4_096
MAX_CANDIDATE_BRANCH_ORDERED_LANDING_DEMANDS = 512
MAX_CANDIDATE_BRANCH_STEPS = 16_384
MAX_LANDING_OPTION_SITE_IDS = 32_768
MAX_LANDING_OPTION_CONTACTS_PER_OPTION = 32_768
MAX_LANDING_SITE_OWNER_LAYERS = 2
MAX_LANDING_SITE_OWNER_COMPONENT_REFS_PER_SIDE = 200_000
MAX_LANDING_SITE_PAIR_RULE_REFS = 4_096
MAX_ENDPOINT_OWNER_COMPONENT_REFS = 200_000
MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE = 200_000
MAX_CANDIDATE_PAIR_DECLARATION_REFS = 4_096
# Closed-allocation replay visits option/contact adjacency after the aggregate
# assignment frontier has already been bounded.  Give that exact verification
# the same finite ceiling instead of rejecting a valid 32k-site projection.
MAX_DERIVED_CONTINUITY_EVALUATIONS = 8 * MAX_TOTAL_ASSIGNMENT_ARCS
CP_SAT_INT64_MAX = (1 << 63) - 1

_ROUTE_PLAN_OBJECTIVE_PHASE_IDS = (
    "hard_feasibility",
    "new_reserved_geometry_area_nm2",
    "landing_site_cut_count",
    "canonical_semantic_tie",
)
ROUTE_PLAN_OBJECTIVE_DEFAULT_DETERMINISTIC_TIME = 1.0
ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME = 1.0
_ROUTE_PLAN_OBJECTIVE_POLICY = {
    "hard_feasibility": {
        "fallback": "full proof configuration on remaining shared account",
        "objective": "none",
        "search": (
            "single-worker fixed-search with presolve and probing disabled"
        ),
    },
    "incumbent_handoff": (
        "full model-variable assignment hint plus witnessed area upper bound"
    ),
    "canonical_tie": {
        "decision_order": "candidate-track-landing-option-site semantic ids",
        "search": "single CP-SAT FIXED_SEARCH satisfaction solve; select max bit",
    },
    "landing_cost": "each selected physical site once: lower pad + upper pad + cut bbox area",
    "phases": list(_ROUTE_PLAN_OBJECTIVE_PHASE_IDS),
    "schema": "clay.route-plan-objective-policy.v2",
    "shared_account": (
        "all feasibility and lexicographic phases debit one caller-authorized "
        "deterministic-time account"
    ),
    "track_cost": "each new selected corridor-use closed bbox area at its exact per-use width",
}


def route_plan_objective_policy_digest() -> str:
    return hashlib.sha256(json.dumps(
        _ROUTE_PLAN_OBJECTIVE_POLICY,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


class CandidatePoolError(ValueError):
    """A bounded explicit candidate pool is malformed or semantically incomplete."""


class CandidatePoolLimitError(CandidatePoolError):
    """A candidate pool exceeds a declared deterministic structure bound."""


@dataclass
class _CandidatePoolRebuildBudget:
    track_centers: int = 0
    landing_options: int = 0
    landing_option_site_claims: int = 0
    landing_option_contacts: int = 0
    endpoint_owner_component_refs: int = 0
    factorized_corridor_uses: int = 0
    factorized_landing_demands: int = 0
    factorized_branches: int = 0
    factorized_branch_steps: int = 0
    factorized_branch_ordered_uses: int = 0
    factorized_branch_ordered_demands: int = 0
    factorized_goal_branch_coverage: int = 0
    factorized_endpoint_accesses: int = 0
    factorized_assignment_arcs: int = 0
    factorized_ir_items: int = 0
    factorized_owned_component_use_refs: int = 0
    factorized_required_goal_refs: int = 0


def _bounded_materialize_with_pool_budget(
    values: object,
    *,
    limit: int,
    label: str,
    pool_budget: Optional[_CandidatePoolRebuildBudget],
    budget_field: str,
    aggregate_limit: int,
    aggregate_label: str,
) -> tuple[object, ...]:
    """Materialize once while charging a pool-scoped cap per yielded item."""

    for bound, bound_label in (
        (limit, label),
        (aggregate_limit, aggregate_label),
    ):
        if type(bound) is not int or bound < 0:
            raise TypeError(
                f"{bound_label} limit must be an exact non-negative integer"
            )
    if pool_budget is None:
        return _bounded_materialize(
            values,
            limit=limit,
            label=label,
            error_type=CandidatePoolLimitError,
        )
    materialized = []
    for value in iter(values):  # type: ignore[arg-type]
        if len(materialized) >= limit:
            raise CandidatePoolLimitError(f"{label} count exceeds {limit}")
        aggregate_count = getattr(pool_budget, budget_field) + 1
        setattr(pool_budget, budget_field, aggregate_count)
        if aggregate_count > aggregate_limit:
            raise CandidatePoolLimitError(
                f"{aggregate_label} {aggregate_count} exceeds {aggregate_limit}"
            )
        materialized.append(value)
    return tuple(materialized)


class RoutePlanObjectiveGeometryError(CandidatePoolError):
    """A physical objective references geometry that was never materialized."""


class MissingNetCandidateError(CandidatePoolError):
    """A required net has no complete-net candidate in this pool."""


class MissingRequiredGoalError(CandidatePoolError):
    """Required Task-4 semantic linkage is absent, unresolved, or uncovered."""


class RouteResourcePlanVerificationError(ValueError):
    """A selected resource plan does not replay against its exact candidate pool."""


class RouteResourcePlanVerificationBudgetExhausted(
    RouteResourcePlanVerificationError
):
    """Objective replay ended UNKNOWN; this is not evidence that a plan is invalid."""

    state = "UNKNOWN"


class InfeasibilityCertificateVerificationError(ValueError):
    """A pool-scoped infeasibility witness is stale, altered, or not replayable."""


@dataclass(frozen=True, order=True)
class CorridorLayerKey:
    corridor_id: str
    layer_n: int

    def __post_init__(self) -> None:
        if not isinstance(self.corridor_id, str) or not self.corridor_id:
            raise CandidatePoolError("corridor_id must be a non-empty string")
        if not _is_int_nm(self.layer_n) or self.layer_n <= 0:
            raise CandidatePoolError("corridor layer_n must be a positive integer")


@dataclass(frozen=True)
class TrackResource:
    """Explicit legal tracks; never synthesized from a nominal C4 observation."""

    key: CorridorLayerKey
    preferred_axis: str
    track_centers_nm: tuple[int, ...]
    provenance: str
    gds_layer_key: Optional[LayerKey] = None
    through_interval_nm: Optional[tuple[int, int]] = None
    corridor_bbox_nm: Optional[BBoxNm] = None
    routing_world_provenance_digest: str = ""
    min_space_nm: int = 0
    owner_net: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, CorridorLayerKey):
            raise CandidatePoolError("track resource key must be CorridorLayerKey")
        if self.preferred_axis not in ("H", "V"):
            raise CandidatePoolError("track resource preferred_axis must be 'H' or 'V'")
        centers = _bounded_materialize(
            self.track_centers_nm,
            limit=MAX_TRACK_CENTERS_PER_RESOURCE,
            label="track center",
            error_type=CandidatePoolLimitError,
        )
        if not centers or not all(_is_int_nm(center) for center in centers):
            raise CandidatePoolError("track centers must be explicit integer-nm values")
        if len(centers) != len(set(centers)):
            raise CandidatePoolError("track centers must be unique")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise CandidatePoolError("track resource provenance must be non-empty")
        if self.gds_layer_key is not None:
            gds_layer_key = tuple(self.gds_layer_key)
            if len(gds_layer_key) != 2 or not all(
                _is_int_nm(value) for value in gds_layer_key
            ):
                raise CandidatePoolError(
                    "track resource gds_layer_key must be an integer GDS key"
                )
            object.__setattr__(self, "gds_layer_key", gds_layer_key)
        complete = (
            self.through_interval_nm is not None,
            self.corridor_bbox_nm is not None,
            bool(self.routing_world_provenance_digest),
        )
        if any(complete) and not all(complete):
            raise CandidatePoolError(
                "geometry-complete track resource requires through interval, "
                "corridor bbox, and routing-world digest"
            )
        if not isinstance(self.routing_world_provenance_digest, str):
            raise CandidatePoolError(
                "track routing_world_provenance_digest must be a string"
            )
        if not _is_int_nm(self.min_space_nm) or self.min_space_nm < 0:
            raise CandidatePoolError(
                "track min_space_nm must be a non-negative integer nm"
            )
        if self.owner_net is not None and (
            not isinstance(self.owner_net, str) or not self.owner_net
        ):
            raise CandidatePoolError(
                "track owner_net must be None or a non-empty string"
            )
        if all(complete):
            interval = tuple(self.through_interval_nm or ())
            bbox = tuple(self.corridor_bbox_nm or ())
            if (
                len(interval) != 2
                or not all(_is_int_nm(value) for value in interval)
                or interval[1] <= interval[0]
            ):
                raise CandidatePoolError(
                    "track through_interval_nm must be a non-empty integer interval"
                )
            if (
                len(bbox) != 4
                or not all(_is_int_nm(value) for value in bbox)
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise CandidatePoolError(
                    "track corridor_bbox_nm must be a non-empty integer bbox"
                )
            run_bounds = (bbox[0], bbox[2]) if self.preferred_axis == "H" else (
                bbox[1], bbox[3]
            )
            cross_bounds = (bbox[1], bbox[3]) if self.preferred_axis == "H" else (
                bbox[0], bbox[2]
            )
            if not (
                run_bounds[0] <= interval[0] < interval[1] <= run_bounds[1]
            ):
                raise CandidatePoolError(
                    "track through interval must lie inside corridor bbox"
                )
            if any(
                center < cross_bounds[0] or center > cross_bounds[1]
                for center in centers
            ):
                raise CandidatePoolError(
                    "track centers must lie inside corridor bbox"
                )
            object.__setattr__(self, "through_interval_nm", interval)
            object.__setattr__(self, "corridor_bbox_nm", bbox)
        object.__setattr__(self, "track_centers_nm", tuple(sorted(centers)))


@dataclass(frozen=True)
class CorridorUse:
    use_id: str
    resource: CorridorLayerKey
    interval_nm: tuple[int, int]
    track_span: int
    allowed_track_starts: tuple[int, ...]
    width_nm: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.use_id, str) or not self.use_id:
            raise CandidatePoolError("corridor use_id must be a non-empty string")
        if not isinstance(self.resource, CorridorLayerKey):
            raise CandidatePoolError("corridor use resource must be CorridorLayerKey")
        interval = tuple(self.interval_nm)
        if (
            len(interval) != 2
            or not all(_is_int_nm(value) for value in interval)
            or interval[1] <= interval[0]
        ):
            raise CandidatePoolError("corridor interval_nm must be a non-empty half-open interval")
        if not _is_int_nm(self.track_span) or self.track_span <= 0:
            raise CandidatePoolError("corridor track_span must be a positive integer")
        if self.width_nm is not None and (
            not _is_int_nm(self.width_nm) or self.width_nm <= 0
        ):
            raise CandidatePoolError(
                "corridor width_nm must be a positive integer nm when present"
            )
        starts = _bounded_materialize(
            self.allowed_track_starts,
            limit=MAX_CORRIDOR_USE_ALLOWED_TRACK_STARTS,
            label="allowed track start",
            error_type=CandidatePoolLimitError,
        )
        if not starts or not all(_is_int_nm(start) and start >= 0 for start in starts):
            raise CandidatePoolError("allowed_track_starts must contain non-negative integers")
        if len(starts) != len(set(starts)):
            raise CandidatePoolError("allowed_track_starts must be unique")
        object.__setattr__(self, "interval_nm", interval)
        object.__setattr__(self, "allowed_track_starts", tuple(sorted(starts)))


@dataclass(frozen=True, order=True)
class TrackChoiceRef:
    """An exact candidate-local track choice, without geometry inference."""

    use_id: str
    track_start: int

    def __post_init__(self) -> None:
        if not isinstance(self.use_id, str) or not self.use_id:
            raise CandidatePoolError("track choice use_id must be non-empty")
        if not _is_int_nm(self.track_start) or self.track_start < 0:
            raise CandidatePoolError(
                "track choice track_start must be a non-negative integer"
            )


@dataclass(frozen=True)
class LandingSite:
    site_id: str
    xy_nm: PointNm
    lower_layer_n: int
    upper_layer_n: int
    capacity: int
    provenance: str
    cut_bbox_nm: Optional[BBoxNm] = None
    lower_pad_bbox_nm: Optional[BBoxNm] = None
    upper_pad_bbox_nm: Optional[BBoxNm] = None
    owner_net: Optional[str] = None
    owner_layer_ns: tuple[int, ...] = ()
    lower_owner_component_refs: tuple[str, ...] = ()
    upper_owner_component_refs: tuple[str, ...] = ()
    routing_world_provenance_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.site_id, str) or not self.site_id:
            raise CandidatePoolError("landing site_id must be a non-empty string")
        point = tuple(self.xy_nm)
        if len(point) != 2 or not all(_is_int_nm(value) for value in point):
            raise CandidatePoolError("landing xy_nm must be an integer-nm point")
        if (
            not _is_int_nm(self.lower_layer_n)
            or not _is_int_nm(self.upper_layer_n)
            or self.lower_layer_n <= 0
            or self.upper_layer_n != self.lower_layer_n + 1
        ):
            raise CandidatePoolError("landing layers must be positive and adjacent")
        if not _is_int_nm(self.capacity) or self.capacity <= 0:
            raise CandidatePoolError("landing capacity must be a positive integer")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise CandidatePoolError("landing provenance must be non-empty")
        if self.owner_net is not None and (
            not isinstance(self.owner_net, str) or not self.owner_net
        ):
            raise CandidatePoolError("landing owner_net must be non-empty when present")
        owner_layers = _bounded_materialize(
            self.owner_layer_ns,
            limit=MAX_LANDING_SITE_OWNER_LAYERS,
            label="landing owner layer",
            error_type=CandidatePoolLimitError,
        )
        if (
            any(
                not _is_int_nm(layer_n)
                or layer_n not in (self.lower_layer_n, self.upper_layer_n)
                for layer_n in owner_layers
            )
            or len(owner_layers) != len(set(owner_layers))
        ):
            raise CandidatePoolError(
                "landing owner_layer_ns must be unique landing-side layer integers"
            )
        if self.owner_net is None and owner_layers:
            raise CandidatePoolError(
                "landing owner_layer_ns requires a non-empty owner_net"
            )
        side_refs: dict[str, tuple[str, ...]] = {}
        for name in (
            "lower_owner_component_refs",
            "upper_owner_component_refs",
        ):
            refs = _bounded_materialize(
                getattr(self, name),
                limit=MAX_LANDING_SITE_OWNER_COMPONENT_REFS_PER_SIDE,
                label=f"landing {name}",
                error_type=CandidatePoolLimitError,
            )
            if (
                any(not isinstance(ref, str) or not ref for ref in refs)
                or len(refs) != len(set(refs))
            ):
                raise CandidatePoolError(
                    f"landing {name} must contain unique non-empty strings"
                )
            side_refs[name] = tuple(sorted(refs))
        if self.owner_net is None and any(side_refs.values()):
            raise CandidatePoolError(
                "landing owner component refs require a non-empty owner_net"
            )
        for side, layer_n in (
            ("lower", self.lower_layer_n),
            ("upper", self.upper_layer_n),
        ):
            refs = side_refs[f"{side}_owner_component_refs"]
            if bool(refs) != (layer_n in owner_layers):
                raise CandidatePoolError(
                    f"landing {side} owner component refs must exactly match "
                    "owner_layer_ns"
                )
        complete = (
            self.cut_bbox_nm is not None,
            self.lower_pad_bbox_nm is not None,
            self.upper_pad_bbox_nm is not None,
            bool(self.routing_world_provenance_digest),
        )
        if any(complete) and not all(complete):
            raise CandidatePoolError(
                "geometry-complete landing site requires cut/lower-pad/upper-pad "
                "bboxes and routing-world digest"
            )
        if not isinstance(self.routing_world_provenance_digest, str):
            raise CandidatePoolError(
                "landing routing_world_provenance_digest must be a string"
            )
        if not any(complete) and self.owner_net is not None:
            raise CandidatePoolError(
                "landing owner_net requires geometry-complete provenance"
            )
        if all(complete):
            boxes = tuple(
                tuple(box or ())
                for box in (
                    self.cut_bbox_nm,
                    self.lower_pad_bbox_nm,
                    self.upper_pad_bbox_nm,
                )
            )
            if any(
                len(box) != 4
                or not all(_is_int_nm(value) for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
                for box in boxes
            ):
                raise CandidatePoolError(
                    "landing cut and pad bboxes must be non-empty integer bboxes"
                )
            cut, lower_pad, upper_pad = boxes
            if any(
                not (
                    pad[0] <= cut[0]
                    and pad[1] <= cut[1]
                    and pad[2] >= cut[2]
                    and pad[3] >= cut[3]
                )
                for pad in (lower_pad, upper_pad)
            ):
                raise CandidatePoolError("landing pads must enclose the via cut")
            if any(
                not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])
                for box in boxes
            ):
                raise CandidatePoolError("landing bboxes must contain site xy_nm")
            object.__setattr__(self, "cut_bbox_nm", cut)
            object.__setattr__(self, "lower_pad_bbox_nm", lower_pad)
            object.__setattr__(self, "upper_pad_bbox_nm", upper_pad)
        object.__setattr__(self, "owner_layer_ns", tuple(sorted(owner_layers)))
        for name, refs in side_refs.items():
            object.__setattr__(self, name, refs)
        object.__setattr__(self, "xy_nm", point)


@dataclass(frozen=True, order=True)
class LandingExpansionResource:
    """Hash-bound proof input for a monotone Hall landing-capacity repair.

    This is not a proposed cut.  It states that the listed legal via sites form
    the complete current lattice inside one final-master corridor boundary; the
    placement layer independently replays the named boundary before deriving an
    inequality.  Pools without this evidence can only return an exact no-good.
    """

    expansion_id: str
    placement_basis_sha256: str
    owner_block_ids: tuple[str, str]
    corridor_relation: str
    run_interval_nm: tuple[int, int]
    observed_clearance_nm: int
    site_pitch_nm: int
    legal_site_ids: tuple[str, ...]
    routing_world_provenance_digest: str
    declaration_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.expansion_id, str) or not self.expansion_id:
            raise CandidatePoolError("landing expansion id must be non-empty")
        if not _is_sha256(self.placement_basis_sha256):
            raise CandidatePoolError("landing expansion basis digest must be SHA-256")
        owners = tuple(sorted(self.owner_block_ids))
        if (
            len(owners) != 2
            or owners[0] == owners[1]
            or any(not isinstance(owner, str) or not owner for owner in owners)
        ):
            raise CandidatePoolError("landing expansion requires two master owners")
        if self.corridor_relation not in (
            "horizontal_separation", "vertical_separation"
        ):
            raise CandidatePoolError("landing expansion has unknown corridor relation")
        interval = tuple(self.run_interval_nm)
        if (
            len(interval) != 2
            or any(not _is_int_nm(value) for value in interval)
            or interval[1] <= interval[0]
        ):
            raise CandidatePoolError("landing expansion run interval is malformed")
        for name in ("observed_clearance_nm", "site_pitch_nm"):
            value = getattr(self, name)
            if not _is_int_nm(value) or value <= 0:
                raise CandidatePoolError(f"landing expansion {name} must be positive")
        sites = tuple(sorted(self.legal_site_ids))
        if (
            not sites
            or any(not isinstance(site_id, str) or not site_id for site_id in sites)
            or len(sites) != len(set(sites))
        ):
            raise CandidatePoolError("landing expansion site ids are malformed")
        if (
            not isinstance(self.routing_world_provenance_digest, str)
            or not self.routing_world_provenance_digest
        ):
            raise CandidatePoolError("landing expansion lacks routing-world provenance")
        refs = tuple(sorted(self.declaration_refs))
        if any(not isinstance(ref, str) or not ref for ref in refs) or len(refs) != len(set(refs)):
            raise CandidatePoolError("landing expansion declaration refs are malformed")
        object.__setattr__(self, "owner_block_ids", owners)
        object.__setattr__(self, "run_interval_nm", interval)
        object.__setattr__(self, "legal_site_ids", sites)
        object.__setattr__(self, "declaration_refs", refs)


@dataclass(frozen=True)
class OwnedConductorComponentCertificate:
    """Layer-number binding for one exact component-index record."""

    layer_n: int
    component: OwnedConductorComponent

    def __post_init__(self) -> None:
        if not _is_int_nm(self.layer_n) or self.layer_n <= 0:
            raise CandidatePoolError(
                "owned conductor certificate layer_n must be positive"
            )
        if not isinstance(self.component, OwnedConductorComponent):
            raise CandidatePoolError(
                "owned conductor certificate component has invalid type"
            )


@dataclass(frozen=True)
class OwnedConductorIndexCertificate:
    """One world-bound, complete owned-conductor component-index table."""

    world_digest: str
    components: tuple[OwnedConductorComponentCertificate, ...]
    index_digest: str
    schema: str = "clay.owned-conductor-index-certificate.v1"

    def __post_init__(self) -> None:
        if self.schema != "clay.owned-conductor-index-certificate.v1":
            raise CandidatePoolError(
                "owned conductor index certificate schema must be canonical v1"
            )
        if not isinstance(self.world_digest, str) or not self.world_digest:
            raise CandidatePoolError(
                "owned conductor index certificate world_digest must be non-empty"
            )
        if not isinstance(self.index_digest, str) or not self.index_digest:
            raise CandidatePoolError(
                "owned conductor index certificate index_digest must be non-empty"
            )
        certificates = _bounded_materialize(
            self.components,
            limit=MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES,
            label="owned conductor index certificate component",
            error_type=CandidatePoolLimitError,
        )
        if not all(
            isinstance(certificate, OwnedConductorComponentCertificate)
            for certificate in certificates
        ):
            raise CandidatePoolError(
                "owned conductor index certificate contains malformed components"
            )
        certificates = tuple(sorted(
            certificates,
            key=lambda certificate: certificate.component.component_ref,
        ))
        try:
            validate_owned_conductor_component_index_certificate(
                world_digest=self.world_digest,
                components=tuple(
                    certificate.component for certificate in certificates
                ),
                index_digest=self.index_digest,
            )
        except OwnedConductorIndexError as exc:
            raise CandidatePoolError(
                f"owned conductor index certificate does not replay: {exc}"
            ) from exc
        object.__setattr__(self, "components", certificates)


@dataclass(frozen=True, order=True)
class LandingSitePairConflict:
    """A replay-derived incompatibility between two distinct landing sites."""

    left_site_id: str
    right_site_id: str
    conflict_scope: str
    rule_refs: tuple[str, ...]
    routing_world_provenance_digest: str

    def __post_init__(self) -> None:
        ids = (self.left_site_id, self.right_site_id)
        if any(not isinstance(site_id, str) or not site_id for site_id in ids):
            raise CandidatePoolError(
                "landing site-pair ids must be non-empty strings"
            )
        if ids[0] == ids[1]:
            raise CandidatePoolError(
                "landing site-pair conflict requires two distinct sites"
            )
        if self.conflict_scope not in (
            "all_nets",
            "foreign_nets",
            "same_net_bridge",
        ):
            raise CandidatePoolError(
                "landing site-pair conflict_scope must be all_nets, foreign_nets, "
                "or same_net_bridge"
            )
        refs = _bounded_materialize(
            self.rule_refs,
            limit=MAX_LANDING_SITE_PAIR_RULE_REFS,
            label="landing site-pair rule ref",
            error_type=CandidatePoolLimitError,
        )
        if (
            not refs
            or any(not isinstance(ref, str) or not ref for ref in refs)
            or len(refs) != len(set(refs))
        ):
            raise CandidatePoolError(
                "landing site-pair rule_refs must be unique non-empty strings"
            )
        if (
            not isinstance(self.routing_world_provenance_digest, str)
            or not self.routing_world_provenance_digest
        ):
            raise CandidatePoolError(
                "landing site-pair conflict must be routing-world-bound"
            )
        left, right = sorted(ids)
        object.__setattr__(self, "left_site_id", left)
        object.__setattr__(self, "right_site_id", right)
        object.__setattr__(self, "rule_refs", tuple(sorted(refs)))


def _landing_site_payload(site: LandingSite) -> dict[str, object]:
    return {
        "capacity": site.capacity,
        "cut_bbox_nm": list(site.cut_bbox_nm or ()),
        "lower_layer_n": site.lower_layer_n,
        "lower_owner_component_refs": list(site.lower_owner_component_refs),
        "lower_pad_bbox_nm": list(site.lower_pad_bbox_nm or ()),
        "owner_layer_ns": list(site.owner_layer_ns),
        "owner_net": site.owner_net,
        "provenance": site.provenance,
        "routing_world_provenance_digest": (
            site.routing_world_provenance_digest
        ),
        "site_id": site.site_id,
        "upper_layer_n": site.upper_layer_n,
        "upper_owner_component_refs": list(site.upper_owner_component_refs),
        "upper_pad_bbox_nm": list(site.upper_pad_bbox_nm or ()),
        "xy_nm": list(site.xy_nm),
    }


def landing_site_collection_sha256(
    landing_sites: Sequence[LandingSite] | LandingSiteCatalog,
) -> str:
    """Hash the complete finite geometry/ownership identity of landing sites."""

    if isinstance(landing_sites, LandingSiteCatalog):
        return landing_sites.collection_sha256
    raw_sites = _bounded_materialize(
        landing_sites,
        limit=MAX_TOTAL_ASSIGNMENT_ARCS,
        label="landing site collection entry",
        error_type=CandidatePoolLimitError,
    )
    sites = tuple(sorted(
        (_rebuild_landing_site(site) for site in raw_sites),
        key=lambda site: site.site_id,
    ))
    payload = [_landing_site_payload(site) for site in sites]
    return hashlib.sha256(json.dumps(
        {
            "landing_sites": payload,
            "schema": "clay.landing-site-collection.v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _sorted_landing_site_collection_sha256(
    landing_sites: object,
) -> str:
    """Stream the canonical collection hash from already sorted exact sites."""

    digest = hashlib.sha256()
    digest.update(b'{"landing_sites":[')
    previous_site_id: str | None = None
    count = 0
    for raw_site in landing_sites:  # type: ignore[union-attr]
        if not isinstance(raw_site, LandingSite):
            raise CandidatePoolError(
                "landing site collection contains an invalid value"
            )
        site = _rebuild_landing_site(raw_site)
        if previous_site_id is not None and site.site_id <= previous_site_id:
            raise CandidatePoolError(
                "streamed landing site collection ids must be unique and sorted"
            )
        if count:
            digest.update(b",")
        digest.update(json.dumps(
            _landing_site_payload(site),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        previous_site_id = site.site_id
        count += 1
    digest.update(b'],"schema":"clay.landing-site-collection.v1"}')
    return digest.hexdigest()


def _landing_site_page_sha256(
    page_index: int,
    sites: Sequence[LandingSite],
) -> str:
    return hashlib.sha256(json.dumps(
        {
            "collection_sha256": landing_site_collection_sha256(sites),
            "page_index": page_index,
            "schema": "clay.landing-site-page.v1",
            "site_count": len(sites),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LandingSitePage:
    """One bounded, content-addressed page of an exact site projection."""

    page_index: int
    sites: tuple[LandingSite, ...]
    page_sha256: str

    def __post_init__(self) -> None:
        if type(self.page_index) is not int or self.page_index < 0:
            raise CandidatePoolError(
                "landing site page index must be a non-negative exact integer"
            )
        raw_sites = _bounded_materialize(
            self.sites,
            limit=MAX_LANDING_SITES,
            label="landing site page site",
            error_type=CandidatePoolLimitError,
        )
        if not raw_sites:
            raise CandidatePoolError("landing site pages must not be empty")
        if not all(isinstance(site, LandingSite) for site in raw_sites):
            raise CandidatePoolError(
                "landing site pages must contain LandingSite values"
            )
        sites = tuple(sorted(
            (_rebuild_landing_site(site) for site in raw_sites),
            key=lambda site: site.site_id,
        ))
        site_ids = tuple(site.site_id for site in sites)
        if len(site_ids) != len(set(site_ids)):
            raise CandidatePoolError("landing site page ids must be unique")
        expected_sha256 = _landing_site_page_sha256(self.page_index, sites)
        if self.page_sha256 != expected_sha256:
            raise CandidatePoolError("landing site page hash does not match contents")
        object.__setattr__(self, "sites", sites)

    @classmethod
    def build(
        cls,
        page_index: int,
        sites: Sequence[LandingSite],
    ) -> "LandingSitePage":
        raw_sites = _bounded_materialize(
            sites,
            limit=MAX_LANDING_SITES,
            label="landing site page site",
            error_type=CandidatePoolLimitError,
        )
        ordered = tuple(sorted(
            (_rebuild_landing_site(site) for site in raw_sites),
            key=lambda site: site.site_id,
        ))
        return cls(
            page_index=page_index,
            sites=ordered,
            page_sha256=_landing_site_page_sha256(page_index, ordered),
        )


def _prevalidate_landing_site_catalog_pages(
    raw_pages: object,
    *,
    routing_world_provenance_digest: str,
) -> tuple[
    tuple[tuple[int, tuple[LandingSite, ...], str], ...],
    int,
]:
    """Bound and validate catalog structure before any page/content hashing."""

    try:
        iterator = iter(raw_pages)  # type: ignore[call-overload]
    except TypeError as exc:
        raise CandidatePoolError(
            "landing site catalog pages must be iterable"
        ) from exc
    sentinel = object()
    current = next(iterator, sentinel)
    if current is sentinel:
        raise CandidatePoolError(
            "landing site catalog pages must contain typed non-empty pages"
        )
    prepared: list[tuple[int, tuple[LandingSite, ...], str]] = []
    total_count = 0
    previous_site_id: str | None = None
    expected_page_index = 0
    while current is not sentinel:
        following = next(iterator, sentinel)
        is_final_page = following is sentinel
        if not isinstance(current, LandingSitePage):
            raise CandidatePoolError(
                "landing site catalog pages must contain typed non-empty pages"
            )
        page = current
        if (
            type(page.page_index) is not int
            or page.page_index != expected_page_index
        ):
            raise CandidatePoolError(
                "landing site catalog page indices must be contiguous and ordered"
            )
        if not isinstance(page.page_sha256, str) or not page.page_sha256:
            raise CandidatePoolError(
                "landing site catalog page hash must be non-empty"
            )
        try:
            site_iterator = iter(page.sites)
        except TypeError as exc:
            raise CandidatePoolError(
                "landing site pages must contain LandingSite values"
            ) from exc
        page_sites: list[LandingSite] = []
        for raw_site in site_iterator:
            if len(page_sites) >= MAX_LANDING_SITES:
                raise CandidatePoolLimitError(
                    "landing site page site count exceeds "
                    f"{MAX_LANDING_SITES}"
                )
            if total_count + len(page_sites) >= MAX_TOTAL_ASSIGNMENT_ARCS:
                raise CandidatePoolLimitError(
                    "landing site catalog entry count exceeds "
                    f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
                )
            if not isinstance(raw_site, LandingSite):
                raise CandidatePoolError(
                    "landing site pages must contain LandingSite values"
                )
            page_sites.append(_rebuild_landing_site(raw_site))
        if not page_sites:
            raise CandidatePoolError("landing site pages must not be empty")
        sites = tuple(sorted(page_sites, key=lambda site: site.site_id))
        site_ids = tuple(site.site_id for site in sites)
        if len(site_ids) != len(set(site_ids)):
            raise CandidatePoolError("landing site page ids must be unique")
        if not is_final_page and len(sites) != MAX_LANDING_SITES:
            raise CandidatePoolError(
                "landing site catalog non-final pages must use the canonical "
                "maximal page size"
            )
        for site in sites:
            if (
                site.routing_world_provenance_digest
                != routing_world_provenance_digest
            ):
                raise CandidatePoolError(
                    "landing site catalog page site has stale routing world "
                    "binding"
                )
            if previous_site_id is not None and site.site_id <= previous_site_id:
                raise CandidatePoolError(
                    "landing site catalog ids must be globally unique and sorted"
                )
            previous_site_id = site.site_id
        prepared.append((page.page_index, sites, page.page_sha256))
        total_count += len(sites)
        expected_page_index += 1
        current = following
    return tuple(prepared), total_count


@dataclass(frozen=True)
class LandingSiteCatalog:
    """Globally unique exact sites split across independently bounded pages."""

    routing_world_provenance_digest: str
    landing_site_domain_digest: str
    pages: tuple[LandingSitePage, ...]
    site_count: int
    collection_sha256: str
    catalog_sha256: str

    @staticmethod
    def _catalog_sha256(
        *,
        routing_world_provenance_digest: str,
        landing_site_domain_digest: str,
        pages: Sequence[LandingSitePage],
        site_count: int,
        collection_sha256: str,
    ) -> str:
        return hashlib.sha256(json.dumps(
            {
                "collection_sha256": collection_sha256,
                "landing_site_domain_digest": landing_site_domain_digest,
                "page_sha256s": [page.page_sha256 for page in pages],
                "routing_world_provenance_digest": (
                    routing_world_provenance_digest
                ),
                "schema": "clay.landing-site-catalog.v1",
                "site_count": site_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.routing_world_provenance_digest, str)
            or not self.routing_world_provenance_digest
        ):
            raise CandidatePoolError(
                "landing site catalog must be routing-world-bound"
            )
        if (
            not isinstance(self.landing_site_domain_digest, str)
            or not self.landing_site_domain_digest
        ):
            raise CandidatePoolError(
                "landing site catalog requires an exact domain digest"
            )
        prepared_pages, site_count = _prevalidate_landing_site_catalog_pages(
            self.pages,
            routing_world_provenance_digest=(
                self.routing_world_provenance_digest
            ),
        )
        if type(self.site_count) is not int or self.site_count != site_count:
            raise CandidatePoolError(
                "landing site catalog count does not match its pages"
            )
        pages = tuple(
            LandingSitePage(
                page_index=page_index,
                sites=sites,
                page_sha256=page_sha256,
            )
            for page_index, sites, page_sha256 in prepared_pages
        )
        collection_sha256 = _sorted_landing_site_collection_sha256(
            site for page in pages for site in page.sites
        )
        if self.collection_sha256 != collection_sha256:
            raise CandidatePoolError(
                "landing site catalog collection hash does not match its pages"
            )
        expected_catalog_sha256 = self._catalog_sha256(
            routing_world_provenance_digest=(
                self.routing_world_provenance_digest
            ),
            landing_site_domain_digest=self.landing_site_domain_digest,
            pages=pages,
            site_count=site_count,
            collection_sha256=collection_sha256,
        )
        if self.catalog_sha256 != expected_catalog_sha256:
            raise CandidatePoolError(
                "landing site catalog hash does not match its contents"
            )
        object.__setattr__(self, "pages", pages)

    @classmethod
    def build(
        cls,
        sites: Sequence[LandingSite],
        *,
        routing_world_provenance_digest: str,
        landing_site_domain_digest: str,
    ) -> "LandingSiteCatalog":
        raw_sites = _bounded_materialize(
            sites,
            limit=MAX_TOTAL_ASSIGNMENT_ARCS,
            label="landing site catalog entry",
            error_type=CandidatePoolLimitError,
        )
        ordered = tuple(sorted(
            (_rebuild_landing_site(site) for site in raw_sites),
            key=lambda site: site.site_id,
        ))
        site_ids = tuple(site.site_id for site in ordered)
        if not ordered:
            raise CandidatePoolError("cannot build an empty landing site catalog")
        if len(site_ids) != len(set(site_ids)):
            raise CandidatePoolError("landing site catalog ids must be unique")
        pages = tuple(
            LandingSitePage.build(
                page_index,
                ordered[start:start + MAX_LANDING_SITES],
            )
            for page_index, start in enumerate(
                range(0, len(ordered), MAX_LANDING_SITES)
            )
        )
        return cls.from_pages(
            pages,
            routing_world_provenance_digest=(
                routing_world_provenance_digest
            ),
            landing_site_domain_digest=landing_site_domain_digest,
        )

    @classmethod
    def from_pages(
        cls,
        pages: Sequence[LandingSitePage],
        *,
        routing_world_provenance_digest: str,
        landing_site_domain_digest: str,
    ) -> "LandingSiteCatalog":
        prepared_pages, site_count = _prevalidate_landing_site_catalog_pages(
            pages,
            routing_world_provenance_digest=(
                routing_world_provenance_digest
            ),
        )
        ordered_pages = tuple(
            LandingSitePage(
                page_index=page_index,
                sites=page_sites,
                page_sha256=page_sha256,
            )
            for page_index, page_sites, page_sha256 in prepared_pages
        )
        collection_sha256 = _sorted_landing_site_collection_sha256(
            site for page in ordered_pages for site in page.sites
        )
        catalog_sha256 = cls._catalog_sha256(
            routing_world_provenance_digest=routing_world_provenance_digest,
            landing_site_domain_digest=landing_site_domain_digest,
            pages=ordered_pages,
            site_count=site_count,
            collection_sha256=collection_sha256,
        )
        return cls(
            routing_world_provenance_digest=routing_world_provenance_digest,
            landing_site_domain_digest=landing_site_domain_digest,
            pages=ordered_pages,
            site_count=site_count,
            collection_sha256=collection_sha256,
            catalog_sha256=catalog_sha256,
        )

    def iter_sites(self):
        for page in self.pages:
            yield from page.sites

    def canonical_payload(self) -> dict[str, object]:
        return {
            "catalog_sha256": self.catalog_sha256,
            "collection_sha256": self.collection_sha256,
            "landing_site_domain_digest": self.landing_site_domain_digest,
            "pages": [
                {
                    "page_index": page.page_index,
                    "page_sha256": page.page_sha256,
                    "sites": [
                        _landing_site_payload(site) for site in page.sites
                    ],
                }
                for page in self.pages
            ],
            "routing_world_provenance_digest": (
                self.routing_world_provenance_digest
            ),
            "schema": "clay.landing-site-catalog.v1",
            "site_count": self.site_count,
        }


@dataclass(frozen=True, order=True)
class LandingSiteConflictMetalRule:
    """One exact PDK metal-spacing fact consumed by conflict replay."""

    layer_n: int
    min_space_nm: int

    def __post_init__(self) -> None:
        if not _is_int_nm(self.layer_n) or self.layer_n <= 0:
            raise CandidatePoolError(
                "landing conflict metal layer_n must be positive"
            )
        if not _is_int_nm(self.min_space_nm) or self.min_space_nm < 0:
            raise CandidatePoolError(
                "landing conflict metal min_space_nm must be non-negative"
            )


@dataclass(frozen=True, order=True)
class LandingSiteConflictViaRule:
    """One exact PDK via-cut spacing fact consumed by conflict replay."""

    lower_layer_n: int
    cut_spacing_nm: int

    def __post_init__(self) -> None:
        if not _is_int_nm(self.lower_layer_n) or self.lower_layer_n <= 0:
            raise CandidatePoolError(
                "landing conflict via lower_layer_n must be positive"
            )
        if not _is_int_nm(self.cut_spacing_nm) or self.cut_spacing_nm < 0:
            raise CandidatePoolError(
                "landing conflict via cut_spacing_nm must be non-negative"
            )


@dataclass(frozen=True)
class LandingSiteConflictDomain:
    """Closed finite PDK/world facts for exact lazy site-pair separation."""

    routing_world_provenance_digest: str
    pdk_provenance_digest: str
    landing_site_collection_digest: str
    metal_rules: tuple[LandingSiteConflictMetalRule, ...]
    via_rules: tuple[LandingSiteConflictViaRule, ...]
    schema: str = "clay.landing-site-conflict-domain.v1"

    def __post_init__(self) -> None:
        for name in (
            "routing_world_provenance_digest",
            "pdk_provenance_digest",
            "landing_site_collection_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CandidatePoolError(
                    f"landing conflict domain {name} must be non-empty"
                )
        if self.schema != "clay.landing-site-conflict-domain.v1":
            raise CandidatePoolError("unknown landing conflict domain schema")
        metal_rules = tuple(sorted(_bounded_materialize(
            self.metal_rules,
            limit=MAX_LANDING_CONFLICT_DOMAIN_LAYERS,
            label="landing conflict metal rule",
            error_type=CandidatePoolLimitError,
        )))
        via_rules = tuple(sorted(_bounded_materialize(
            self.via_rules,
            limit=MAX_LANDING_CONFLICT_DOMAIN_LAYERS,
            label="landing conflict via rule",
            error_type=CandidatePoolLimitError,
        )))
        if (
            not metal_rules
            or not via_rules
            or not all(
                isinstance(rule, LandingSiteConflictMetalRule)
                for rule in metal_rules
            )
            or not all(
                isinstance(rule, LandingSiteConflictViaRule)
                for rule in via_rules
            )
        ):
            raise CandidatePoolError(
                "landing conflict domain requires typed metal and via rules"
            )
        if len({rule.layer_n for rule in metal_rules}) != len(metal_rules):
            raise CandidatePoolError(
                "landing conflict metal layer identities must be unique"
            )
        if len({rule.lower_layer_n for rule in via_rules}) != len(via_rules):
            raise CandidatePoolError(
                "landing conflict via layer identities must be unique"
            )
        object.__setattr__(self, "metal_rules", metal_rules)
        object.__setattr__(self, "via_rules", via_rules)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "metal_rules": [
                [rule.layer_n, rule.min_space_nm] for rule in self.metal_rules
            ],
            "pdk_provenance_digest": self.pdk_provenance_digest,
            "landing_site_collection_digest": (
                self.landing_site_collection_digest
            ),
            "routing_world_provenance_digest": (
                self.routing_world_provenance_digest
            ),
            "schema": self.schema,
            "via_rules": [
                [rule.lower_layer_n, rule.cut_spacing_nm]
                for rule in self.via_rules
            ],
        }

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _LandingConflictMetalFacts:
    min_space_nm: int


def _derive_landing_site_pair_conflict_from_domain(
    left: LandingSite,
    right: LandingSite,
    domain: LandingSiteConflictDomain,
) -> Optional[LandingSitePairConflict]:
    """Replay one pair from closed facts, using the canonical geometry rule."""

    if not isinstance(domain, LandingSiteConflictDomain):
        raise CandidatePoolError("exact landing conflict domain is required")
    if any(
        site.routing_world_provenance_digest
        != domain.routing_world_provenance_digest
        for site in (left, right)
    ):
        raise CandidatePoolError(
            "landing conflict pair has stale routing-world binding"
        )
    layer_by_n = {
        rule.layer_n: _LandingConflictMetalFacts(rule.min_space_nm)
        for rule in domain.metal_rules
    }
    via_spacing_by_lower_layer = {
        rule.lower_layer_n: rule.cut_spacing_nm for rule in domain.via_rules
    }
    required_metal_layers = {
        left.lower_layer_n,
        left.upper_layer_n,
        right.lower_layer_n,
        right.upper_layer_n,
    }
    required_via_layers = {left.lower_layer_n, right.lower_layer_n}
    if (
        not required_metal_layers <= set(layer_by_n)
        or not required_via_layers <= set(via_spacing_by_lower_layer)
    ):
        raise CandidatePoolError(
            "landing conflict domain is incomplete for selected pair"
        )
    from clay.layout.route.resource_world import _derive_site_pair_conflict

    return _derive_site_pair_conflict(
        left,
        right,
        layer_by_n,
        via_spacing_by_lower_layer,
        domain.routing_world_provenance_digest,
    )


def canonical_explicit_routing_resource_ledger_json(
    routing_world_provenance_digest: str,
    resources: Sequence[TrackResource],
    landing_sites: Sequence[LandingSite],
    landing_site_conflicts: Sequence[LandingSitePairConflict],
    owned_conductor_index_digest: str = "",
    landing_site_domain_digest: str = "",
    landing_sites_complete: bool = True,
    landing_site_conflict_domain: Optional[
        LandingSiteConflictDomain
    ] = None,
) -> str:
    """Canonical certificate shared by the ledger and world-bound pools."""

    payload = {
        "landing_site_conflicts": [
            {
                "conflict_scope": conflict.conflict_scope,
                "left_site_id": conflict.left_site_id,
                "right_site_id": conflict.right_site_id,
                "routing_world_provenance_digest": (
                    conflict.routing_world_provenance_digest
                ),
                "rule_refs": list(conflict.rule_refs),
            }
            for conflict in sorted(
                landing_site_conflicts,
                key=lambda item: (item.left_site_id, item.right_site_id),
            )
        ],
        **(
            {}
            if landing_site_conflict_domain is None
            else {
                "landing_site_conflict_domain": (
                    landing_site_conflict_domain.canonical_payload()
                )
            }
        ),
        "owned_conductor_index_digest": owned_conductor_index_digest,
        "landing_site_domain_digest": landing_site_domain_digest,
        "landing_sites_complete": landing_sites_complete,
        "landing_sites": [
            {
                "capacity": site.capacity,
                "cut_bbox_nm": list(site.cut_bbox_nm or ()),
                "lower_layer_n": site.lower_layer_n,
                "lower_pad_bbox_nm": list(site.lower_pad_bbox_nm or ()),
                "owner_net": site.owner_net,
                "owner_layer_ns": list(site.owner_layer_ns),
                "lower_owner_component_refs": list(
                    site.lower_owner_component_refs
                ),
                "upper_owner_component_refs": list(
                    site.upper_owner_component_refs
                ),
                "provenance": site.provenance,
                "routing_world_provenance_digest": (
                    site.routing_world_provenance_digest
                ),
                "site_id": site.site_id,
                "upper_layer_n": site.upper_layer_n,
                "upper_pad_bbox_nm": list(site.upper_pad_bbox_nm or ()),
                "xy_nm": list(site.xy_nm),
            }
            for site in sorted(landing_sites, key=lambda item: item.site_id)
        ],
        "routing_world_provenance_digest": routing_world_provenance_digest,
        "schema": (
            "clay.explicit-routing-resource-ledger.v6"
            if landing_site_conflict_domain is None
            else "clay.explicit-routing-resource-ledger.v7"
        ),
        "track_resources": [
            {
                "corridor_bbox_nm": list(resource.corridor_bbox_nm or ()),
                "key": {
                    "corridor_id": resource.key.corridor_id,
                    "layer_n": resource.key.layer_n,
                },
                "gds_layer_key": (
                    None
                    if resource.gds_layer_key is None
                    else list(resource.gds_layer_key)
                ),
                "preferred_axis": resource.preferred_axis,
                "provenance": resource.provenance,
                "routing_world_provenance_digest": (
                    resource.routing_world_provenance_digest
                ),
                "min_space_nm": resource.min_space_nm,
                "through_interval_nm": list(resource.through_interval_nm or ()),
                "track_centers_nm": list(resource.track_centers_nm),
                **(
                    {"owner_net": resource.owner_net}
                    if resource.owner_net is not None
                    else {}
                ),
            }
            for resource in sorted(resources, key=lambda item: item.key)
        ],
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def explicit_routing_resource_ledger_sha256(
    routing_world_provenance_digest: str,
    resources: Sequence[TrackResource],
    landing_sites: Sequence[LandingSite],
    landing_site_conflicts: Sequence[LandingSitePairConflict],
    owned_conductor_index_digest: str = "",
    landing_site_domain_digest: str = "",
    landing_sites_complete: bool = True,
    landing_site_conflict_domain: Optional[
        LandingSiteConflictDomain
    ] = None,
) -> str:
    return hashlib.sha256(canonical_explicit_routing_resource_ledger_json(
        routing_world_provenance_digest,
        resources,
        landing_sites,
        landing_site_conflicts,
        owned_conductor_index_digest,
        landing_site_domain_digest,
        landing_sites_complete,
        landing_site_conflict_domain,
    ).encode("utf-8")).hexdigest()


def paged_routing_resource_ledger_sha256(
    routing_world_provenance_digest: str,
    resources: Sequence[TrackResource],
    landing_site_catalog: LandingSiteCatalog,
    landing_site_conflicts: Sequence[LandingSitePairConflict],
    owned_conductor_index_digest: str = "",
    landing_site_domain_digest: str = "",
    landing_sites_complete: bool = False,
    landing_site_conflict_domain: Optional[
        LandingSiteConflictDomain
    ] = None,
) -> str:
    """Hash a projection whose exact finite site union spans bounded pages."""

    payload = json.loads(canonical_explicit_routing_resource_ledger_json(
        routing_world_provenance_digest,
        resources,
        (),
        landing_site_conflicts,
        owned_conductor_index_digest,
        landing_site_domain_digest,
        landing_sites_complete,
        landing_site_conflict_domain,
    ))
    payload["landing_site_catalog"] = landing_site_catalog.canonical_payload()
    payload["schema"] = "clay.explicit-routing-resource-ledger.v8"
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _track_matches_pool_world_binding(
    resource: TrackResource,
    routing_world_provenance_digest: str,
) -> bool:
    geometry_complete = (
        resource.through_interval_nm is not None
        and resource.corridor_bbox_nm is not None
        and resource.gds_layer_key is not None
        and bool(resource.routing_world_provenance_digest)
    )
    if routing_world_provenance_digest:
        return (
            geometry_complete
            and resource.routing_world_provenance_digest
            == routing_world_provenance_digest
        )
    return (
        resource.through_interval_nm is None
        and resource.corridor_bbox_nm is None
        and resource.gds_layer_key is None
        and not resource.routing_world_provenance_digest
    )


def _landing_matches_pool_world_binding(
    site: LandingSite,
    routing_world_provenance_digest: str,
) -> bool:
    geometry_complete = (
        site.cut_bbox_nm is not None
        and site.lower_pad_bbox_nm is not None
        and site.upper_pad_bbox_nm is not None
        and bool(site.routing_world_provenance_digest)
    )
    if routing_world_provenance_digest:
        return (
            geometry_complete
            and site.routing_world_provenance_digest
            == routing_world_provenance_digest
        )
    return (
        site.cut_bbox_nm is None
        and site.lower_pad_bbox_nm is None
        and site.upper_pad_bbox_nm is None
        and site.owner_net is None
        and not site.routing_world_provenance_digest
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _closed_bbox_touch(left: Sequence[int], right: Sequence[int]) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def _rebuild_owned_conductor_component(
    component: OwnedConductorComponent,
    *,
    member_budget: list[int],
) -> OwnedConductorComponent:
    """Bound and canonicalize every iterable before certificate replay."""

    def materialize_members(values: object, *, label: str) -> tuple[object, ...]:
        remaining = MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS - member_budget[0]
        try:
            materialized = _bounded_materialize(
                values,
                limit=max(0, remaining),
                label=label,
                error_type=CandidatePoolLimitError,
            )
        except CandidatePoolLimitError as exc:
            raise CandidatePoolLimitError(
                "owned conductor component member count exceeds "
                f"{MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS}"
            ) from exc
        member_budget[0] += len(materialized)
        return materialized

    atom_facts = materialize_members(
        component.atom_facts,
        label="owned component atom fact",
    )
    pin_accesses = materialize_members(
        component.pin_accesses,
        label="owned component pin access",
    )
    return replace(
        component,
        layer_key=_bounded_materialize(
            component.layer_key,
            limit=2,
            label="owned component layer key",
            error_type=CandidatePoolLimitError,
        ),
        bbox_nm=_bounded_materialize(
            component.bbox_nm,
            limit=4,
            label="owned component bbox coordinate",
            error_type=CandidatePoolLimitError,
        ),
        atom_facts=atom_facts,
        atom_refs=_bounded_materialize(
            component.atom_refs,
            limit=len(atom_facts),
            label="owned component atom ref",
            error_type=CandidatePoolLimitError,
        ),
        atom_terminal_conductor_refs=_bounded_materialize(
            component.atom_terminal_conductor_refs,
            limit=len(atom_facts),
            label="owned component atom terminal conductor ref",
            error_type=CandidatePoolLimitError,
        ),
        kinds=_bounded_materialize(
            component.kinds,
            limit=max(1, len(atom_facts)),
            label="owned component kind",
            error_type=CandidatePoolLimitError,
        ),
        pin_accesses=pin_accesses,
    )


def _owned_conductor_index_sha256(
    certificates: tuple[OwnedConductorComponentCertificate, ...],
    *,
    world_digest: str,
) -> str:
    components = tuple(
        sorted(
            (certificate.component for certificate in certificates),
            key=lambda component: component.component_ref,
        )
    )
    pin_bindings = tuple(
        sorted(
            (
                (pin, component.component_ref)
                for component in components
                for pin in component.pin_accesses
            ),
            key=lambda item: json.dumps(
                [list(item[0]), item[1]], separators=(",", ":")
            ),
        )
    )
    return _canonical_sha256(
        {
            "components": [component.canonical_payload() for component in components],
            "pin_bindings": [
                {"component_ref": component_ref, "pin_access": list(pin)}
                for pin, component_ref in pin_bindings
            ],
            "schema": "clay.owned-conductor-component-index.v2",
            "world_digest": world_digest,
        }
    )


def _owned_component_touches_pad(
    component: OwnedConductorComponent, pad_bbox_nm: BBoxNm
) -> bool:
    return any(
        not (
            fact[2] < pad_bbox_nm[0]
            or pad_bbox_nm[2] < fact[0]
            or fact[3] < pad_bbox_nm[1]
            or pad_bbox_nm[3] < fact[1]
        )
        for fact in component.atom_facts
    ) or any(
        pad_bbox_nm[0] <= pin[3] <= pad_bbox_nm[2]
        and pad_bbox_nm[1] <= pin[4] <= pad_bbox_nm[3]
        for pin in component.pin_accesses
    )


_OWNED_COMPONENT_REPLAY_SPATIAL_NODE_CAPACITY = 4


@dataclass(frozen=True)
class _OwnedComponentReplaySpatialLeaf:
    """One component-bbox broad-phase leaf; exact atoms remain authoritative."""

    bbox_nm: BBoxNm
    certificate: OwnedConductorComponentCertificate

    @property
    def stable_ref(self) -> str:
        return self.certificate.component.component_ref


@dataclass(frozen=True)
class _OwnedComponentReplaySpatialNode:
    bbox_nm: BBoxNm
    children: tuple[object, ...]
    leaves: bool
    stable_ref: str


def _owned_component_replay_spatial_groups(
    items: Sequence[object],
) -> tuple[tuple[object, ...], ...]:
    """Deterministic Sort-Tile-Recursive groups for replay-only broad phase."""

    if not items:
        return ()
    capacity = _OWNED_COMPONENT_REPLAY_SPATIAL_NODE_CAPACITY
    group_count = (len(items) + capacity - 1) // capacity
    root = math.isqrt(group_count)
    slice_count = root if root * root == group_count else root + 1
    groups_per_slice = (group_count + slice_count - 1) // slice_count
    slice_size = groups_per_slice * capacity
    by_x = sorted(
        items,
        key=lambda item: (
            item.bbox_nm[0] + item.bbox_nm[2],  # type: ignore[attr-defined]
            item.bbox_nm[1] + item.bbox_nm[3],  # type: ignore[attr-defined]
            item.bbox_nm,  # type: ignore[attr-defined]
            item.stable_ref,  # type: ignore[attr-defined]
        ),
    )
    groups: list[tuple[object, ...]] = []
    for start in range(0, len(by_x), slice_size):
        spatial_slice = sorted(
            by_x[start : start + slice_size],
            key=lambda item: (
                item.bbox_nm[1] + item.bbox_nm[3],  # type: ignore[attr-defined]
                item.bbox_nm[0] + item.bbox_nm[2],  # type: ignore[attr-defined]
                item.bbox_nm,  # type: ignore[attr-defined]
                item.stable_ref,  # type: ignore[attr-defined]
            ),
        )
        for offset in range(0, len(spatial_slice), capacity):
            groups.append(tuple(spatial_slice[offset : offset + capacity]))
    return tuple(groups)


def _owned_component_replay_spatial_node(
    children: Sequence[object], *, leaves: bool
) -> _OwnedComponentReplaySpatialNode:
    ordered = tuple(children)
    return _OwnedComponentReplaySpatialNode(
        bbox_nm=(
            min(child.bbox_nm[0] for child in ordered),  # type: ignore[attr-defined]
            min(child.bbox_nm[1] for child in ordered),  # type: ignore[attr-defined]
            max(child.bbox_nm[2] for child in ordered),  # type: ignore[attr-defined]
            max(child.bbox_nm[3] for child in ordered),  # type: ignore[attr-defined]
        ),
        children=ordered,
        leaves=leaves,
        stable_ref=min(
            child.stable_ref for child in ordered  # type: ignore[attr-defined]
        ),
    )


def _build_owned_component_replay_spatial_roots(
    certificates: Sequence[OwnedConductorComponentCertificate],
) -> Mapping[int, _OwnedComponentReplaySpatialNode]:
    by_layer: dict[int, list[_OwnedComponentReplaySpatialLeaf]] = {}
    for certificate in certificates:
        by_layer.setdefault(certificate.layer_n, []).append(
            _OwnedComponentReplaySpatialLeaf(
                bbox_nm=certificate.component.bbox_nm,
                certificate=certificate,
            )
        )
    roots: dict[int, _OwnedComponentReplaySpatialNode] = {}
    capacity = _OWNED_COMPONENT_REPLAY_SPATIAL_NODE_CAPACITY
    for layer_n, leaves in sorted(by_layer.items()):
        level = [
            _owned_component_replay_spatial_node(group, leaves=True)
            for group in _owned_component_replay_spatial_groups(leaves)
        ]
        while len(level) > capacity:
            level = [
                _owned_component_replay_spatial_node(group, leaves=False)
                for group in _owned_component_replay_spatial_groups(level)
            ]
        roots[layer_n] = (
            level[0]
            if len(level) == 1
            else _owned_component_replay_spatial_node(level, leaves=False)
        )
    return roots


def _owned_component_replay_spatial_candidates(
    roots: Mapping[int, _OwnedComponentReplaySpatialNode],
    *,
    layer_n: int,
    pad_bbox_nm: BBoxNm,
    traversal_budget: list[int],
) -> tuple[OwnedConductorComponentCertificate, ...]:
    """Return a complete bbox superset for the existing exact pad predicate."""

    root = roots.get(layer_n)
    if root is None:
        return ()
    candidates: list[OwnedConductorComponentCertificate] = []

    def consume_traversal() -> None:
        traversal_budget[0] += 1
        if (
            traversal_budget[0]
            > MAX_OWNED_CONDUCTOR_COMPONENT_SPATIAL_REPLAY_EVALUATIONS
        ):
            raise CandidatePoolLimitError(
                "owned-component spatial replay evaluations "
                f"{traversal_budget[0]} exceed "
                f"{MAX_OWNED_CONDUCTOR_COMPONENT_SPATIAL_REPLAY_EVALUATIONS}"
            )

    def visit(node: _OwnedComponentReplaySpatialNode) -> None:
        consume_traversal()
        if not _closed_bbox_touch(node.bbox_nm, pad_bbox_nm):
            return
        if node.leaves:
            for value in node.children:
                if not isinstance(value, _OwnedComponentReplaySpatialLeaf):
                    raise CandidatePoolError(
                        "owned-component replay spatial leaf is malformed"
                    )
                consume_traversal()
                if _closed_bbox_touch(value.bbox_nm, pad_bbox_nm):
                    candidates.append(value.certificate)
            return
        for value in node.children:
            if not isinstance(value, _OwnedComponentReplaySpatialNode):
                raise CandidatePoolError(
                    "owned-component replay spatial node is malformed"
                )
            visit(value)

    visit(root)
    return tuple(sorted(
        candidates,
        key=lambda certificate: certificate.component.component_ref,
    ))


def _owned_component_contains_point(
    component: OwnedConductorComponent, point_nm: PointNm
) -> bool:
    return any(
        fact[0] <= point_nm[0] <= fact[2]
        and fact[1] <= point_nm[1] <= fact[3]
        for fact in component.atom_facts
    ) or any(
        pin[3] == point_nm[0] and pin[4] == point_nm[1]
        for pin in component.pin_accesses
    )


def _owned_component_conductor_contains_point(
    component: OwnedConductorComponent, point_nm: PointNm
) -> bool:
    """Prove traversal from material conductor atoms, never from pin metadata."""

    return any(
        fact[0] <= point_nm[0] <= fact[2]
        and fact[1] <= point_nm[1] <= fact[3]
        for fact in component.atom_facts
    )


def _owned_component_conductors_touch(
    left: OwnedConductorComponent,
    right: OwnedConductorComponent,
) -> bool:
    if left.net != right.net or left.layer_key != right.layer_key:
        return False
    return any(
        _closed_bbox_touch(left_fact, right_fact)
        for left_fact in left.atom_facts
        for right_fact in right.atom_facts
    )


LANDING_SELECTION_EXPLICIT_OPTION = "EXPLICIT_OPTION"
LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY = (
    "FACTORIZED_EXACT_CARDINALITY"
)
LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN = "DERIVED_EXACT_SITE_DOMAIN"


@dataclass(frozen=True)
class LandingDemand:
    demand_id: str
    terminal_provenance: str
    allowed_site_ids: tuple[str, ...]
    declaration_ref: Optional[str] = None
    required_multiplicity: int = 1
    selection_mode: str = LANDING_SELECTION_EXPLICIT_OPTION

    def __post_init__(self) -> None:
        if not isinstance(self.demand_id, str) or not self.demand_id:
            raise CandidatePoolError("landing demand_id must be a non-empty string")
        if not isinstance(self.terminal_provenance, str) or not self.terminal_provenance:
            raise CandidatePoolError("landing terminal_provenance must be non-empty")
        raw_site_ids = _bounded_materialize(
            self.allowed_site_ids,
            limit=MAX_LANDING_DEMAND_ALLOWED_SITES,
            label="allowed site",
            error_type=CandidatePoolLimitError,
        )
        site_ids = tuple(str(site_id) for site_id in raw_site_ids)
        if not site_ids or any(not site_id for site_id in site_ids):
            raise CandidatePoolError("landing allowed_site_ids must be non-empty strings")
        if len(site_ids) != len(set(site_ids)):
            raise CandidatePoolError("landing allowed_site_ids must be unique")
        if self.declaration_ref is not None and (
            not isinstance(self.declaration_ref, str) or not self.declaration_ref
        ):
            raise CandidatePoolError("landing declaration_ref must be non-empty when present")
        if (
            not _is_int_nm(self.required_multiplicity)
            or self.required_multiplicity <= 0
        ):
            raise CandidatePoolError(
                "landing required_multiplicity must be a positive integer"
            )
        if self.required_multiplicity > len(site_ids):
            raise CandidatePoolError(
                "landing required_multiplicity exceeds allowed site count"
            )
        if self.selection_mode not in {
            LANDING_SELECTION_EXPLICIT_OPTION,
            LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY,
            LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
        }:
            raise CandidatePoolError(
                "landing selection_mode must be EXPLICIT_OPTION, "
                "FACTORIZED_EXACT_CARDINALITY, or DERIVED_EXACT_SITE_DOMAIN"
            )
        object.__setattr__(self, "allowed_site_ids", tuple(sorted(site_ids)))


@dataclass(frozen=True, order=True)
class CandidateBranchStep:
    kind: str
    ref_id: str

    def __post_init__(self) -> None:
        if self.kind not in (
            "track_use",
            "landing_demand",
            "owned_component",
        ):
            raise CandidatePoolError(
                "branch step kind must be track_use, landing_demand, or "
                "owned_component"
            )
        if not isinstance(self.ref_id, str) or not self.ref_id:
            raise CandidatePoolError("branch step ref_id must be non-empty")


@dataclass(frozen=True)
class CandidateBranch:
    """One ordered branch of a closed routing-allocation program."""

    branch_id: str
    source_endpoint: str
    target_endpoint: str
    ordered_use_ids: tuple[str, ...]
    ordered_landing_demand_ids: tuple[str, ...]
    ordered_steps: tuple[CandidateBranchStep, ...] = ()

    def __post_init__(self) -> None:
        for name in ("branch_id", "source_endpoint", "target_endpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CandidatePoolError(f"branch {name} must be non-empty")
        if self.source_endpoint == self.target_endpoint:
            raise CandidatePoolError("branch endpoints must be distinct")
        for name, limit, label in (
            (
                "ordered_use_ids",
                MAX_CANDIDATE_BRANCH_ORDERED_USES,
                "branch ordered use id",
            ),
            (
                "ordered_landing_demand_ids",
                MAX_CANDIDATE_BRANCH_ORDERED_LANDING_DEMANDS,
                "branch ordered landing demand id",
            ),
        ):
            values = _bounded_materialize(
                getattr(self, name),
                limit=limit,
                label=label,
                error_type=CandidatePoolLimitError,
            )
            if any(not isinstance(value, str) or not value for value in values):
                raise CandidatePoolError(f"branch {name} must contain non-empty strings")
            if len(values) != len(set(values)):
                raise CandidatePoolError(f"branch {name} must not repeat ids")
            object.__setattr__(self, name, values)
        steps = _bounded_materialize(
            self.ordered_steps,
            limit=MAX_CANDIDATE_BRANCH_STEPS,
            label="branch step",
            error_type=CandidatePoolLimitError,
        )
        if not all(isinstance(step, CandidateBranchStep) for step in steps):
            raise CandidatePoolError(
                "branch ordered_steps must contain CandidateBranchStep values"
            )
        if not steps:
            raise CandidatePoolError("closed branch requires ordered_steps")
        if tuple(
            step.ref_id for step in steps if step.kind == "track_use"
        ) != self.ordered_use_ids:
            raise CandidatePoolError(
                "branch track-use steps must exactly project ordered_use_ids"
            )
        if tuple(
            step.ref_id for step in steps if step.kind == "landing_demand"
        ) != self.ordered_landing_demand_ids:
            raise CandidatePoolError(
                "branch landing steps must exactly project ordered_landing_demand_ids"
            )
        object.__setattr__(self, "ordered_steps", steps)


def _power_lifted_root_bindings(
    branches: Sequence[CandidateBranch],
) -> tuple[tuple[str, str], ...]:
    """Project (root-use, root-demand) from lifted power branches."""

    bindings = []
    for branch in branches:
        steps = branch.ordered_steps
        if len(steps) < 3 or tuple(
            step.kind for step in steps[:3]
        ) != ("owned_component", "landing_demand", "track_use"):
            continue
        bindings.append((steps[2].ref_id, steps[1].ref_id))
    return tuple(bindings)


def _require_shared_power_lifted_root(
    branches: Sequence[CandidateBranch],
) -> None:
    bindings = _power_lifted_root_bindings(branches)
    if bindings and len(set(bindings)) != 1:
        raise CandidatePoolError(
            "POWER_SPINE_LOCAL_DROPS lifted branches must share one root "
            "corridor use and root landing demand"
        )


@dataclass(frozen=True, order=True)
class LandingTrackContact:
    site_id: str
    track_choice: TrackChoiceRef

    def __post_init__(self) -> None:
        if not isinstance(self.site_id, str) or not self.site_id:
            raise CandidatePoolError("landing contact site_id must be non-empty")
        if not isinstance(self.track_choice, TrackChoiceRef):
            raise CandidatePoolError(
                "landing contact track_choice must be TrackChoiceRef"
            )

    @property
    def use_id(self) -> str:
        return self.track_choice.use_id

    @property
    def track_start(self) -> int:
        return self.track_choice.track_start


@dataclass(frozen=True)
class LandingOption:
    option_id: str
    demand_id: str
    site_ids: tuple[str, ...]
    contacts: tuple[LandingTrackContact, ...]

    def __post_init__(self) -> None:
        for name in ("option_id", "demand_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CandidatePoolError(f"landing option {name} must be non-empty")
        site_ids = _bounded_materialize(
            self.site_ids,
            limit=MAX_LANDING_OPTION_SITE_IDS,
            label="landing option site id",
            error_type=CandidatePoolLimitError,
        )
        if (
            not site_ids
            or any(not isinstance(site_id, str) or not site_id for site_id in site_ids)
            or len(site_ids) != len(set(site_ids))
        ):
            raise CandidatePoolError(
                "landing option site_ids must be unique non-empty strings"
            )
        contacts = _bounded_materialize(
            self.contacts,
            limit=MAX_LANDING_OPTION_CONTACTS_PER_OPTION,
            label="landing option contact",
            error_type=CandidatePoolLimitError,
        )
        if not all(isinstance(contact, LandingTrackContact) for contact in contacts):
            raise CandidatePoolError(
                "landing option contacts must be LandingTrackContact values"
            )
        if len(contacts) != len(set(contacts)):
            raise CandidatePoolError("landing option contacts must be unique")
        foreign_sites = tuple(sorted(
            {contact.site_id for contact in contacts} - set(site_ids)
        ))
        if foreign_sites:
            raise CandidatePoolError(
                f"landing option contacts name sites outside option {foreign_sites}"
            )
        starts_by_use: dict[str, set[int]] = {}
        for contact in contacts:
            starts_by_use.setdefault(contact.use_id, set()).add(contact.track_start)
        contradictory = tuple(sorted(
            use_id for use_id, starts in starts_by_use.items() if len(starts) > 1
        ))
        if contradictory:
            raise CandidatePoolError(
                "landing option assigns multiple track starts to uses "
                f"{contradictory}"
            )
        object.__setattr__(self, "site_ids", tuple(sorted(site_ids)))
        object.__setattr__(self, "contacts", tuple(sorted(contacts)))


@dataclass(frozen=True, order=True)
class GoalBranchCoverage:
    declaration_ref: str
    branch_id: str
    target_provenance: str

    def __post_init__(self) -> None:
        for name in ("declaration_ref", "branch_id", "target_provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CandidatePoolError(f"goal branch {name} must be non-empty")


@dataclass(frozen=True)
class CandidateEndpointAccess:
    provenance: str
    xy_nm: PointNm
    layer_key: LayerKey
    layer_n: int
    owner_component_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, str) or not self.provenance:
            raise CandidatePoolError("endpoint access provenance must be non-empty")
        point = tuple(self.xy_nm)
        if len(point) != 2 or not all(_is_int_nm(value) for value in point):
            raise CandidatePoolError("endpoint access xy_nm must be an integer point")
        layer_key = tuple(self.layer_key)
        if len(layer_key) != 2 or not all(
            _is_int_nm(value) for value in layer_key
        ):
            raise CandidatePoolError(
                "endpoint access layer_key must be an integer GDS key"
            )
        if not _is_int_nm(self.layer_n) or self.layer_n <= 0:
            raise CandidatePoolError(
                "endpoint access layer_n must be a positive integer"
            )
        refs = _bounded_materialize(
            self.owner_component_refs,
            limit=MAX_ENDPOINT_OWNER_COMPONENT_REFS,
            label="endpoint owner component ref",
            error_type=CandidatePoolLimitError,
        )
        if (
            any(not isinstance(ref, str) or not ref for ref in refs)
            or len(refs) != len(set(refs))
        ):
            raise CandidatePoolError(
                "endpoint owner_component_refs must be unique non-empty strings"
            )
        object.__setattr__(self, "xy_nm", point)
        object.__setattr__(self, "layer_key", layer_key)
        object.__setattr__(self, "owner_component_refs", tuple(sorted(refs)))


@dataclass(frozen=True)
class CompleteNetCandidate:
    candidate_id: str
    net: str
    topology_class: str
    branch_edges: tuple[tuple[str, str], ...]
    corridor_uses: tuple[CorridorUse, ...]
    landing_demands: tuple[LandingDemand, ...]
    required_goal_refs: tuple[str, ...]
    width_nm: int
    via_multiplicity: int
    source_topology_digest: str
    program_kind: str = "PATTERN_TREE"
    branches: tuple[CandidateBranch, ...] = ()
    landing_options: tuple[LandingOption, ...] = ()
    goal_branch_coverage: tuple[GoalBranchCoverage, ...] = ()
    endpoint_accesses: tuple[CandidateEndpointAccess, ...] = ()
    owned_component_use_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise CandidatePoolError("candidate_id must be a non-empty string")
        if not isinstance(self.net, str) or not self.net:
            raise CandidatePoolError("candidate net must be a non-empty string")
        if self.topology_class not in {
            "local_signal",
            "shared_signal_trunk",
            "power_spine_tree",
        }:
            raise CandidatePoolError(f"unknown topology class {self.topology_class!r}")
        if not _is_int_nm(self.width_nm) or self.width_nm <= 0:
            raise CandidatePoolError("candidate width_nm must be positive integer nm")
        raw_edges = _bounded_materialize(
            self.branch_edges,
            limit=MAX_TOTAL_BRANCH_EDGES,
            label="branch edge",
            error_type=CandidatePoolLimitError,
        )
        edges = tuple(tuple(edge) for edge in raw_edges)
        if not edges or any(
            len(edge) != 2
            or not all(isinstance(endpoint, str) and endpoint for endpoint in edge)
            for edge in edges
        ):
            raise CandidatePoolError("candidate branch_edges must be explicit endpoint pairs")
        if len(edges) != len(set(edges)):
            raise CandidatePoolError("candidate branch_edges must be unique")
        uses = _bounded_materialize(
            self.corridor_uses,
            limit=MAX_TOTAL_CORRIDOR_USES,
            label="corridor use",
            error_type=CandidatePoolLimitError,
        )
        if not all(isinstance(use, CorridorUse) for use in uses):
            raise CandidatePoolError("candidate corridor_uses must be CorridorUse values")
        uses = tuple(
            replace(use, width_nm=self.width_nm)
            if use.width_nm is None
            else use
            for use in uses
        )
        use_ids = [use.use_id for use in uses]
        if len(use_ids) != len(set(use_ids)):
            raise CandidatePoolError("candidate corridor use ids must be unique")
        demands = _bounded_materialize(
            self.landing_demands,
            limit=MAX_TOTAL_LANDING_DEMANDS,
            label="landing demand",
            error_type=CandidatePoolLimitError,
        )
        if not all(isinstance(demand, LandingDemand) for demand in demands):
            raise CandidatePoolError("candidate landing_demands must be LandingDemand values")
        demand_ids = [demand.demand_id for demand in demands]
        if len(demand_ids) != len(set(demand_ids)):
            raise CandidatePoolError("candidate landing demand ids must be unique")
        raw_refs = _bounded_materialize(
            self.required_goal_refs,
            limit=MAX_REQUIRED_GOAL_REFS,
            label="required goal ref",
            error_type=CandidatePoolLimitError,
        )
        refs = tuple(str(ref) for ref in raw_refs)
        if any(not ref for ref in refs) or len(refs) != len(set(refs)):
            raise CandidatePoolError("candidate required_goal_refs must be unique non-empty strings")
        if not _is_int_nm(self.via_multiplicity) or self.via_multiplicity < 0:
            raise CandidatePoolError("candidate via_multiplicity must be non-negative")
        if not isinstance(self.source_topology_digest, str) or not self.source_topology_digest:
            raise CandidatePoolError("source_topology_digest must be non-empty")
        if self.program_kind not in {
            "PATTERN_TREE",
            "POWER_SPINE_LOCAL_DROPS",
            "COMPONENT_ONLY",
        }:
            raise CandidatePoolError(
                f"unknown candidate program_kind {self.program_kind!r}"
            )
        branches = _bounded_materialize(
            self.branches,
            limit=MAX_TOTAL_CANDIDATE_BRANCHES,
            label="branch",
            error_type=CandidatePoolLimitError,
        )
        if not all(isinstance(branch, CandidateBranch) for branch in branches):
            raise CandidatePoolError("candidate branches must be CandidateBranch values")
        branch_ids = tuple(branch.branch_id for branch in branches)
        if len(branch_ids) != len(set(branch_ids)):
            raise CandidatePoolError("candidate branch ids must be unique")
        component_use_refs = _bounded_materialize(
            self.owned_component_use_refs,
            limit=MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE,
            label="owned component use ref",
            error_type=CandidatePoolLimitError,
        )
        if (
            any(
                not isinstance(component_ref, str) or not component_ref
                for component_ref in component_use_refs
            )
            or len(component_use_refs) != len(set(component_use_refs))
        ):
            raise CandidatePoolError(
                "owned_component_use_refs must be unique non-empty strings"
            )
        referenced_component_refs = {
            step.ref_id
            for branch in branches
            for step in branch.ordered_steps
            if step.kind == "owned_component"
        }
        if referenced_component_refs != set(component_use_refs):
            raise CandidatePoolError(
                "closed candidate owned component steps must exactly project "
                "owned_component_use_refs"
            )
        if self.program_kind == "COMPONENT_ONLY":
            if not branches or any(
                not any(
                    step.kind == "owned_component"
                    for step in branch.ordered_steps
                )
                for branch in branches
            ):
                raise CandidatePoolError(
                    "COMPONENT_ONLY requires an owned component step in every branch"
                )
            if uses or self.via_multiplicity:
                raise CandidatePoolError(
                    "COMPONENT_ONLY cannot contain track or via resources"
                )
            if demands:
                raise CandidatePoolError(
                    "COMPONENT_ONLY cannot contain landing demands or options"
                )
        elif self.program_kind == "POWER_SPINE_LOCAL_DROPS":
            _require_shared_power_lifted_root(branches)
        if branches:
            if set(edges) != {
                (branch.source_endpoint, branch.target_endpoint)
                for branch in branches
            }:
                raise CandidatePoolError(
                    "closed candidate branch_edges must exactly derive from branches"
                )
            referenced_uses = {
                use_id for branch in branches for use_id in branch.ordered_use_ids
            }
            referenced_demands = {
                demand_id
                for branch in branches
                for demand_id in branch.ordered_landing_demand_ids
            }
            if referenced_uses != set(use_ids):
                raise CandidatePoolError(
                    "closed candidate branches must reference every corridor use exactly by id"
                )
            if referenced_demands != set(demand_ids):
                raise CandidatePoolError(
                    "closed candidate branches must reference every landing demand exactly by id"
                )
        options = _bounded_materialize(
            self.landing_options,
            limit=MAX_TOTAL_LANDING_OPTIONS,
            label="landing option",
            error_type=CandidatePoolLimitError,
        )
        if not all(isinstance(option, LandingOption) for option in options):
            raise CandidatePoolError(
                "candidate landing_options must be LandingOption values"
            )
        if self.program_kind == "COMPONENT_ONLY" and options:
            raise CandidatePoolError(
                "COMPONENT_ONLY cannot contain landing demands or options"
            )
        option_ids = tuple(option.option_id for option in options)
        if len(option_ids) != len(set(option_ids)):
            raise CandidatePoolError("candidate landing option ids must be unique")
        coverage = _bounded_materialize(
            self.goal_branch_coverage,
            limit=MAX_TOTAL_GOAL_BRANCH_COVERAGE,
            label="goal branch coverage",
            error_type=CandidatePoolLimitError,
        )
        if not all(isinstance(item, GoalBranchCoverage) for item in coverage):
            raise CandidatePoolError(
                "candidate goal_branch_coverage must contain GoalBranchCoverage values"
            )
        if (options or coverage) and not branches:
            raise CandidatePoolError(
                "allocation options and goal coverage require closed branches"
            )
        accesses = _bounded_materialize(
            self.endpoint_accesses,
            limit=MAX_TOTAL_ENDPOINT_ACCESSES,
            label="endpoint access",
            error_type=CandidatePoolLimitError,
        )
        if not all(
            isinstance(access, CandidateEndpointAccess) for access in accesses
        ):
            raise CandidatePoolError(
                "candidate endpoint_accesses must contain CandidateEndpointAccess values"
            )
        access_provenance = tuple(access.provenance for access in accesses)
        if len(access_provenance) != len(set(access_provenance)):
            raise CandidatePoolError(
                "candidate endpoint access provenance must be unique"
            )
        if accesses and not branches:
            raise CandidatePoolError(
                "endpoint accesses require closed branches"
            )
        if branches:
            branch_endpoints = {
                endpoint
                for branch in branches
                for endpoint in (branch.source_endpoint, branch.target_endpoint)
            }
            if set(access_provenance) != branch_endpoints:
                raise CandidatePoolError(
                    "closed candidate endpoint accesses must exactly cover branch endpoints"
                )
            demands_by_id = {demand.demand_id: demand for demand in demands}
            options_by_demand: dict[str, list[LandingOption]] = {
                demand_id: [] for demand_id in demand_ids
            }
            for option in options:
                if option.demand_id not in options_by_demand:
                    raise CandidatePoolError(
                        f"landing option names unknown demand {option.demand_id!r}"
                    )
                options_by_demand[option.demand_id].append(option)
            branch_neighbours_by_demand: dict[str, set[str]] = {}
            for branch in branches:
                for step_index, step in enumerate(branch.ordered_steps):
                    if step.kind != "landing_demand":
                        continue
                    demand = demands_by_id[step.ref_id]
                    if demand.terminal_provenance not in (
                        branch.source_endpoint,
                        branch.target_endpoint,
                    ):
                        raise CandidatePoolError(
                            f"landing demand {demand.demand_id!r} "
                            "terminal_provenance must name its own branch endpoint"
                        )
                    last_step_index = len(branch.ordered_steps) - 1
                    if (
                        step_index == 0
                        and step_index != last_step_index
                        and demand.terminal_provenance != branch.source_endpoint
                    ) or (
                        step_index == last_step_index
                        and step_index != 0
                        and demand.terminal_provenance != branch.target_endpoint
                    ):
                        raise CandidatePoolError(
                            f"landing demand {demand.demand_id!r} terminal_provenance "
                            "does not match its branch boundary position"
                        )
                    neighbours: set[str] = set()
                    for neighbour_index in (step_index - 1, step_index + 1):
                        if 0 <= neighbour_index < len(branch.ordered_steps):
                            neighbour = branch.ordered_steps[neighbour_index]
                            if neighbour.kind == "landing_demand":
                                raise CandidatePoolError(
                                    "closed branch cannot place consecutive landing "
                                    "demands without an explicit conductor step"
                                )
                            neighbours.add(neighbour.ref_id)
                    existing_neighbours = branch_neighbours_by_demand.get(
                        demand.demand_id
                    )
                    if (
                        existing_neighbours is not None
                        and existing_neighbours != neighbours
                    ):
                        raise CandidatePoolError(
                            f"landing demand {demand.demand_id!r} reused by "
                            "branches with different adjacent conductor resources"
                        )
                    branch_neighbours_by_demand[demand.demand_id] = neighbours
            missing_options = tuple(sorted(
                demand_id
                for demand_id, demand_options in options_by_demand.items()
                if (
                    not demand_options
                    and demands_by_id[demand_id].selection_mode
                    != LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                )
            ))
            if missing_options:
                raise CandidatePoolError(
                    f"closed candidate demands lack landing options {missing_options}"
                )
            for demand_id, demand_options in options_by_demand.items():
                demand = demands_by_id[demand_id]
                branch_neighbours = branch_neighbours_by_demand[demand_id]
                if (
                    demand.selection_mode
                    == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                ):
                    if demand_options:
                        raise CandidatePoolError(
                            f"derived landing demand {demand_id!r} must not "
                            "carry explicit landing-option aliases"
                        )
                    if demand.required_multiplicity < self.via_multiplicity:
                        raise CandidatePoolError(
                            f"derived landing demand {demand_id!r} must satisfy "
                            "candidate via multiplicity"
                        )
                    continue
                projected_sites = {
                    site_id
                    for option in demand_options
                    for site_id in option.site_ids
                }
                if projected_sites != set(demand.allowed_site_ids):
                    raise CandidatePoolError(
                        f"landing options for demand {demand_id!r} must exactly "
                        "project allowed_site_ids"
                    )
                minimum = max(demand.required_multiplicity, self.via_multiplicity)
                if (
                    demand.selection_mode
                    == LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY
                ):
                    singleton_sites = tuple(
                        option.site_ids[0]
                        for option in demand_options
                        if len(option.site_ids) == 1
                    )
                    if (
                        demand.required_multiplicity < self.via_multiplicity
                        or len(singleton_sites) != len(demand_options)
                        or len(singleton_sites) != len(set(singleton_sites))
                        or set(singleton_sites) != set(demand.allowed_site_ids)
                    ):
                        raise CandidatePoolError(
                            f"factorized landing demand {demand_id!r} must have "
                            "one unique singleton option per allowed site and "
                            "satisfy candidate via multiplicity"
                        )
                    adjacent_use_ids = (
                        set(branch_neighbours) & set(use_ids)
                    )
                    for option in demand_options:
                        atom_site_id = option.site_ids[0]
                        if (
                            len(option.contacts) != len(adjacent_use_ids)
                            or {contact.use_id for contact in option.contacts}
                            != adjacent_use_ids
                            or {
                                contact.site_id for contact in option.contacts
                            } != ({atom_site_id} if option.contacts else set())
                        ):
                            raise CandidatePoolError(
                                f"factorized landing atom {option.option_id!r} "
                                "must bind its singleton site exactly once to "
                                "every adjacent track use"
                            )
                elif any(
                    len(option.site_ids) < minimum
                    for option in demand_options
                ):
                    raise CandidatePoolError(
                        f"landing option for demand {demand_id!r} has fewer than "
                        f"required multiplicity {minimum}"
                    )
                for option in demand_options:
                    contact_uses = {contact.use_id for contact in option.contacts}
                    if not contact_uses.issubset(branch_neighbours):
                        raise CandidatePoolError(
                            f"landing option {option.option_id!r} contact uses must "
                            "belong to adjacent uses in the same branch"
                        )
            coverage_refs = tuple(item.declaration_ref for item in coverage)
            coverage_branches = tuple(item.branch_id for item in coverage)
            if (
                len(coverage_refs) != len(set(coverage_refs))
                or len(coverage_branches) != len(set(coverage_branches))
            ):
                raise CandidatePoolError(
                    "goal coverage must use unique declaration refs and dedicated branches"
                )
            if set(coverage_refs) != set(refs):
                raise CandidatePoolError(
                    "closed candidate goal coverage must exactly match required_goal_refs"
                )
            branches_by_id = {branch.branch_id: branch for branch in branches}
            branch_sources = {branch.source_endpoint for branch in branches}
            for item in coverage:
                branch = branches_by_id.get(item.branch_id)
                if branch is None:
                    raise CandidatePoolError(
                        f"goal coverage names missing branch {item.branch_id!r}"
                    )
                if branch.target_endpoint != item.target_provenance:
                    raise CandidatePoolError(
                        "goal coverage target must be the covered branch leaf"
                    )
                if item.target_provenance in branch_sources:
                    raise CandidatePoolError(
                        "goal coverage target must be a branch leaf"
                    )
        object.__setattr__(self, "branch_edges", tuple(sorted(edges)))
        object.__setattr__(self, "corridor_uses", tuple(sorted(
            uses,
            key=lambda use: (
                use.resource,
                use.interval_nm,
                use.track_span,
                use.allowed_track_starts,
                use.use_id,
            ),
        )))
        object.__setattr__(self, "landing_demands", tuple(sorted(
            demands,
            key=lambda demand: (
                demand.demand_id,
                demand.terminal_provenance,
                demand.allowed_site_ids,
                "" if demand.declaration_ref is None else demand.declaration_ref,
                demand.required_multiplicity,
                demand.selection_mode,
            ),
        )))
        object.__setattr__(self, "required_goal_refs", tuple(sorted(refs)))
        object.__setattr__(self, "branches", tuple(sorted(
            branches, key=lambda branch: branch.branch_id
        )))
        object.__setattr__(self, "landing_options", tuple(sorted(
            options, key=lambda option: option.option_id
        )))
        object.__setattr__(self, "goal_branch_coverage", tuple(sorted(coverage)))
        object.__setattr__(self, "endpoint_accesses", tuple(sorted(
            accesses, key=lambda access: access.provenance
        )))
        object.__setattr__(
            self,
            "owned_component_use_refs",
            tuple(sorted(component_use_refs)),
        )


def complete_net_candidate_identity_payload(
    candidate: CompleteNetCandidate,
) -> dict[str, object]:
    """Return the canonical compiler/verifier identity of one exact program."""

    return {
        "branch_edges": [list(edge) for edge in candidate.branch_edges],
        "branches": [
            {
                "branch_id": branch.branch_id,
                "ordered_landing_demand_ids": list(
                    branch.ordered_landing_demand_ids
                ),
                "ordered_steps": [
                    {"kind": step.kind, "ref_id": step.ref_id}
                    for step in branch.ordered_steps
                ],
                "ordered_use_ids": list(branch.ordered_use_ids),
                "source_endpoint": branch.source_endpoint,
                "target_endpoint": branch.target_endpoint,
            }
            for branch in candidate.branches
        ],
        "corridor_uses": [
            {
                "allowed_track_starts": list(use.allowed_track_starts),
                "interval_nm": list(use.interval_nm),
                "resource": _key_payload(use.resource),
                "track_span": use.track_span,
                "use_id": use.use_id,
                "width_nm": use.width_nm,
            }
            for use in candidate.corridor_uses
        ],
        "endpoint_accesses": [
            {
                "layer_key": list(access.layer_key),
                "layer_n": access.layer_n,
                "owner_component_refs": list(access.owner_component_refs),
                "provenance": access.provenance,
                "xy_nm": list(access.xy_nm),
            }
            for access in candidate.endpoint_accesses
        ],
        "goal_branch_coverage": [
            {
                "branch_id": item.branch_id,
                "declaration_ref": item.declaration_ref,
                "target_provenance": item.target_provenance,
            }
            for item in candidate.goal_branch_coverage
        ],
        "landing_demands": [
            {
                "allowed_site_ids": list(demand.allowed_site_ids),
                "declaration_ref": demand.declaration_ref,
                "demand_id": demand.demand_id,
                "required_multiplicity": demand.required_multiplicity,
                "selection_mode": demand.selection_mode,
                "terminal_provenance": demand.terminal_provenance,
            }
            for demand in candidate.landing_demands
        ],
        "landing_options": [
            {
                "contacts": [
                    {
                        "site_id": contact.site_id,
                        "track_choice": {
                            "track_start": contact.track_start,
                            "use_id": contact.use_id,
                        },
                    }
                    for contact in option.contacts
                ],
                "demand_id": option.demand_id,
                "option_id": option.option_id,
                "site_ids": list(option.site_ids),
            }
            for option in candidate.landing_options
        ],
        "net": candidate.net,
        "owned_component_use_refs": list(candidate.owned_component_use_refs),
        "program_kind": candidate.program_kind,
        "required_goal_refs": list(candidate.required_goal_refs),
        "schema": "clay.fixed-pattern-complete-net-candidate.v2",
        "source_topology_digest": candidate.source_topology_digest,
        "topology_class": candidate.topology_class,
        "via_multiplicity": candidate.via_multiplicity,
        "width_nm": candidate.width_nm,
    }


def content_address_complete_net_candidate(
    candidate: CompleteNetCandidate,
) -> CompleteNetCandidate:
    digest = hashlib.sha256(json.dumps(
        complete_net_candidate_identity_payload(candidate),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return replace(candidate, candidate_id=f"fixed-pattern-candidate:{digest}")


def landing_option_template_record(
    candidate: CompleteNetCandidate,
    option: LandingOption,
) -> tuple[str, dict[str, object], dict[str, object]]:
    """Return one pool-global physical template and its local alias binding."""

    if not isinstance(candidate, CompleteNetCandidate):
        raise CandidatePoolError(
            "landing option template requires CompleteNetCandidate"
        )
    if not isinstance(option, LandingOption):
        raise CandidatePoolError(
            "landing option template requires LandingOption"
        )
    uses_by_id = {use.use_id: use for use in candidate.corridor_uses}
    demands_by_id = {
        demand.demand_id: demand for demand in candidate.landing_demands
    }
    demand = demands_by_id.get(option.demand_id)
    if demand is None:
        raise CandidatePoolError(
            "landing option template references unknown demand "
            f"{option.demand_id!r}"
        )
    contact_rows = []
    for contact in option.contacts:
        use = uses_by_id.get(contact.use_id)
        if use is None:
            raise CandidatePoolError(
                "landing option template references unknown corridor use "
                f"{contact.use_id!r}"
            )
        # The pool-global template owns only the selected landing-site/contact
        # shape.  TrackChoiceRef is candidate-local: ``use_id`` resolves the
        # exact resource and ``track_start`` selects its consecutive tracks.
        # Keeping that binding in the alias factors repeated exact site atoms
        # without erasing a single concrete contact choice.
        site_slot = {"site_id": contact.site_id}
        contact_rows.append((
            json.dumps(site_slot, sort_keys=True, separators=(",", ":")),
            contact.use_id,
            contact.track_start,
            site_slot,
        ))
    contact_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    template_contacts = [
        {"slot_id": f"contact-slot-{ordinal}", **row[3]}
        for ordinal, row in enumerate(contact_rows)
    ]
    template_payload: dict[str, object] = {
        "contacts": template_contacts,
        "schema": "clay.landing-option-template.v2",
        "site_ids": list(option.site_ids),
    }
    template_id = hashlib.sha256(json.dumps(
        template_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    alias_payload: dict[str, object] = {
        "contact_use_bindings": [
            {
                "slot_id": f"contact-slot-{ordinal}",
                "use_id": row[1],
                "track_start": row[2],
            }
            for ordinal, row in enumerate(contact_rows)
        ],
        "demand_id": option.demand_id,
        "option_id": option.option_id,
        "template_id": template_id,
    }
    return template_id, template_payload, alias_payload


def landing_option_template_index(
    candidates: Sequence[CompleteNetCandidate],
) -> tuple[tuple[tuple[str, dict[str, object]], ...], int, str]:
    """Derive the canonical v12 template table, alias count, and digest."""

    templates: dict[str, dict[str, object]] = {}
    alias_count = 0
    for candidate in candidates:
        for option in candidate.landing_options:
            template_id, payload, _alias = landing_option_template_record(
                candidate, option
            )
            prior = templates.get(template_id)
            if prior is not None and prior != payload:
                raise CandidatePoolError(
                    "landing option template hash collision"
                )
            templates[template_id] = payload
            alias_count += 1
    ordered = tuple(
        (template_id, templates[template_id])
        for template_id in sorted(templates)
    )
    digest = hashlib.sha256(json.dumps(
        {
            "schema": "clay.landing-option-template-index.v1",
            "templates": [
                {"template_id": template_id, **payload}
                for template_id, payload in ordered
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return ordered, alias_count, digest


def candidate_has_closed_allocation_ir(candidate: CompleteNetCandidate) -> bool:
    """Return true only when the candidate declares the new closed branch IR."""

    if not isinstance(candidate, CompleteNetCandidate):
        raise TypeError("candidate must be CompleteNetCandidate")
    return bool(candidate.branches)


def _rebuild_closed_allocation_candidate(
    candidate: CompleteNetCandidate,
    *,
    pool_budget: Optional[_CandidatePoolRebuildBudget] = None,
) -> CompleteNetCandidate:
    """Re-run every nested frozen-IR constructor after possible byte tamper."""

    raw_edges = _bounded_materialize(
        candidate.branch_edges,
        limit=MAX_TOTAL_BRANCH_EDGES,
        label="branch edge",
        error_type=CandidatePoolLimitError,
    )
    edges = tuple(
        tuple(_bounded_materialize(
            edge,
            limit=2,
            label="branch edge endpoint",
            error_type=CandidatePoolLimitError,
        ))
        for edge in raw_edges
    )
    raw_uses = _bounded_materialize(
        candidate.corridor_uses,
        limit=MAX_TOTAL_CORRIDOR_USES,
        label="corridor use",
        error_type=CandidatePoolLimitError,
    )
    uses = tuple(
        replace(
            use,
            resource=replace(use.resource),
            interval_nm=_bounded_materialize(
                use.interval_nm,
                limit=2,
                label="corridor interval coordinate",
                error_type=CandidatePoolLimitError,
            ),
            allowed_track_starts=_bounded_materialize(
                use.allowed_track_starts,
                limit=MAX_CORRIDOR_USE_ALLOWED_TRACK_STARTS,
                label="allowed track start",
                error_type=CandidatePoolLimitError,
            ),
        )
        for use in raw_uses
    )
    raw_demands = _bounded_materialize(
        candidate.landing_demands,
        limit=MAX_TOTAL_LANDING_DEMANDS,
        label="landing demand",
        error_type=CandidatePoolLimitError,
    )
    demands = tuple(
        replace(
            demand,
            allowed_site_ids=_bounded_materialize(
                demand.allowed_site_ids,
                limit=MAX_LANDING_DEMAND_ALLOWED_SITES,
                label="allowed site",
                error_type=CandidatePoolLimitError,
            ),
        )
        for demand in raw_demands
    )
    raw_branches = _bounded_materialize(
        candidate.branches,
        limit=MAX_TOTAL_CANDIDATE_BRANCHES,
        label="branch",
        error_type=CandidatePoolLimitError,
    )
    branches = tuple(
        replace(
            branch,
            ordered_use_ids=_bounded_materialize(
                branch.ordered_use_ids,
                limit=MAX_CANDIDATE_BRANCH_ORDERED_USES,
                label="branch ordered use id",
                error_type=CandidatePoolLimitError,
            ),
            ordered_landing_demand_ids=_bounded_materialize(
                branch.ordered_landing_demand_ids,
                limit=MAX_CANDIDATE_BRANCH_ORDERED_LANDING_DEMANDS,
                label="branch ordered landing demand id",
                error_type=CandidatePoolLimitError,
            ),
            ordered_steps=tuple(
                replace(step)
                for step in _bounded_materialize(
                    branch.ordered_steps,
                    limit=MAX_CANDIDATE_BRANCH_STEPS,
                    label="branch step",
                    error_type=CandidatePoolLimitError,
                )
            ),
        )
        for branch in raw_branches
    )
    raw_options = _bounded_materialize_with_pool_budget(
        candidate.landing_options,
        limit=MAX_TOTAL_LANDING_OPTIONS,
        label="landing option",
        pool_budget=pool_budget,
        budget_field="landing_options",
        aggregate_limit=MAX_TOTAL_LANDING_OPTIONS,
        aggregate_label="landing option count",
    )
    options = tuple(
        replace(
            option,
            site_ids=_bounded_materialize_with_pool_budget(
                option.site_ids,
                limit=MAX_LANDING_OPTION_SITE_IDS,
                label="landing option site id",
                pool_budget=pool_budget,
                budget_field="landing_option_site_claims",
                aggregate_limit=MAX_TOTAL_LANDING_OPTION_SITE_CLAIMS,
                aggregate_label="option site claim count",
            ),
            contacts=tuple(
                replace(
                    contact,
                    track_choice=replace(contact.track_choice),
                )
                for contact in _bounded_materialize_with_pool_budget(
                    option.contacts,
                    limit=MAX_LANDING_OPTION_CONTACTS_PER_OPTION,
                    label="landing option contact",
                    pool_budget=pool_budget,
                    budget_field="landing_option_contacts",
                    aggregate_limit=MAX_TOTAL_LANDING_OPTION_CONTACTS,
                    aggregate_label="option contact count",
                )
            ),
        )
        for option in raw_options
    )
    coverage = tuple(
        replace(item)
        for item in _bounded_materialize(
            candidate.goal_branch_coverage,
            limit=MAX_TOTAL_GOAL_BRANCH_COVERAGE,
            label="goal branch coverage",
            error_type=CandidatePoolLimitError,
        )
    )
    accesses = tuple(
        replace(
            access,
            xy_nm=_bounded_materialize(
                access.xy_nm,
                limit=2,
                label="endpoint access coordinate",
                error_type=CandidatePoolLimitError,
            ),
            layer_key=_bounded_materialize(
                access.layer_key,
                limit=2,
                label="endpoint access GDS layer key",
                error_type=CandidatePoolLimitError,
            ),
            owner_component_refs=_bounded_materialize_with_pool_budget(
                access.owner_component_refs,
                limit=MAX_ENDPOINT_OWNER_COMPONENT_REFS,
                label="endpoint owner component ref",
                pool_budget=pool_budget,
                budget_field="endpoint_owner_component_refs",
                aggregate_limit=MAX_TOTAL_ENDPOINT_OWNER_COMPONENT_REFS,
                aggregate_label="endpoint owner component ref count",
            ),
        )
        for access in _bounded_materialize(
            candidate.endpoint_accesses,
            limit=MAX_TOTAL_ENDPOINT_ACCESSES,
            label="endpoint access",
            error_type=CandidatePoolLimitError,
        )
    )
    return replace(
        candidate,
        branch_edges=edges,
        corridor_uses=uses,
        landing_demands=demands,
        required_goal_refs=_bounded_materialize(
            candidate.required_goal_refs,
            limit=MAX_REQUIRED_GOAL_REFS,
            label="required goal ref",
            error_type=CandidatePoolLimitError,
        ),
        branches=branches,
        landing_options=options,
        goal_branch_coverage=coverage,
        endpoint_accesses=accesses,
        owned_component_use_refs=_bounded_materialize(
            candidate.owned_component_use_refs,
            limit=MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE,
            label="owned component use ref",
            error_type=CandidatePoolLimitError,
        ),
    )


def verify_candidate_allocation_ir(
    pool: "CandidatePool", candidate: CompleteNetCandidate
) -> bool:
    """Independently replay one closed candidate in its exact pool context."""

    if not isinstance(pool, CandidatePool) or not isinstance(
        candidate, CompleteNetCandidate
    ):
        raise TypeError("pool and candidate types must match")
    if not candidate_has_closed_allocation_ir(candidate):
        return False
    pool = _replay_candidate_pool_runtime(pool)
    matching = tuple(
        item
        for item in pool.all_candidates
        if item.candidate_id == candidate.candidate_id
    )
    if matching != (candidate,):
        raise CandidatePoolError(
            "closed candidate is not the exact candidate replayed by the pool"
        )
    rebuilt_candidate = _rebuild_closed_allocation_candidate(candidate)
    if rebuilt_candidate != candidate:
        raise CandidatePoolError(
            "closed candidate nested IR is not canonical after replay"
        )
    return True


def _rebuild_track_resource(
    resource: TrackResource,
    *,
    pool_budget: Optional[_CandidatePoolRebuildBudget] = None,
) -> TrackResource:
    return replace(
        resource,
        key=replace(resource.key),
        track_centers_nm=_bounded_materialize_with_pool_budget(
            resource.track_centers_nm,
            limit=MAX_TRACK_CENTERS_PER_RESOURCE,
            label="track center",
            pool_budget=pool_budget,
            budget_field="track_centers",
            aggregate_limit=MAX_TOTAL_TRACK_CENTERS,
            aggregate_label="track center count",
        ),
        gds_layer_key=(
            None
            if resource.gds_layer_key is None
            else _bounded_materialize(
                resource.gds_layer_key,
                limit=2,
                label="track GDS layer key",
                error_type=CandidatePoolLimitError,
            )
        ),
        through_interval_nm=(
            None
            if resource.through_interval_nm is None
            else _bounded_materialize(
                resource.through_interval_nm,
                limit=2,
                label="track through interval coordinate",
                error_type=CandidatePoolLimitError,
            )
        ),
        corridor_bbox_nm=(
            None
            if resource.corridor_bbox_nm is None
            else _bounded_materialize(
                resource.corridor_bbox_nm,
                limit=4,
                label="track corridor bbox coordinate",
                error_type=CandidatePoolLimitError,
            )
        ),
    )


def _bounded_optional_bbox(values: Optional[BBoxNm], *, label: str):
    if values is None:
        return None
    return _bounded_materialize(
        values,
        limit=4,
        label=label,
        error_type=CandidatePoolLimitError,
    )


def _rebuild_landing_site(site: LandingSite) -> LandingSite:
    return replace(
        site,
        xy_nm=_bounded_materialize(
            site.xy_nm,
            limit=2,
            label="landing coordinate",
            error_type=CandidatePoolLimitError,
        ),
        owner_layer_ns=_bounded_materialize(
            site.owner_layer_ns,
            limit=MAX_LANDING_SITE_OWNER_LAYERS,
            label="landing owner layer",
            error_type=CandidatePoolLimitError,
        ),
        lower_owner_component_refs=_bounded_materialize(
            site.lower_owner_component_refs,
            limit=MAX_LANDING_SITE_OWNER_COMPONENT_REFS_PER_SIDE,
            label="landing lower_owner_component_refs",
            error_type=CandidatePoolLimitError,
        ),
        upper_owner_component_refs=_bounded_materialize(
            site.upper_owner_component_refs,
            limit=MAX_LANDING_SITE_OWNER_COMPONENT_REFS_PER_SIDE,
            label="landing upper_owner_component_refs",
            error_type=CandidatePoolLimitError,
        ),
        cut_bbox_nm=_bounded_optional_bbox(
            site.cut_bbox_nm, label="landing cut bbox coordinate"
        ),
        lower_pad_bbox_nm=_bounded_optional_bbox(
            site.lower_pad_bbox_nm, label="landing lower pad bbox coordinate"
        ),
        upper_pad_bbox_nm=_bounded_optional_bbox(
            site.upper_pad_bbox_nm, label="landing upper pad bbox coordinate"
        ),
    )


def _rebuild_required_route_goal(goal: RequiredRouteGoal) -> RequiredRouteGoal:
    target = goal.target
    if target is not None:
        target = replace(
            target,
            xy_nm=_bounded_materialize(
                target.xy_nm,
                limit=2,
                label="required route target coordinate",
                error_type=CandidatePoolLimitError,
            ),
            layer=_bounded_materialize(
                target.layer,
                limit=2,
                label="required route target layer key",
                error_type=CandidatePoolLimitError,
            ),
        )
    return replace(goal, target=target)


def _branch_graph_admissibility_error(
    candidate: CompleteNetCandidate,
) -> Optional[str]:
    adjacency: dict[str, set[str]] = {}
    for left, right in candidate.branch_edges:
        if left == right:
            return f"branch graph has self-loop at endpoint {left!r}"
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    start = min(adjacency)
    visited = {start}
    pending = [start]
    while pending:
        node = pending.pop()
        for neighbour in sorted(adjacency[node], reverse=True):
            if neighbour not in visited:
                visited.add(neighbour)
                pending.append(neighbour)
    unreachable = tuple(sorted(set(adjacency) - visited))
    if unreachable:
        return f"branch graph is disconnected; unreachable endpoints {unreachable}"
    return None


def _track_choice_contains_endpoint(
    access: CandidateEndpointAccess,
    use: CorridorUse,
    resource: TrackResource,
    track_start: int,
) -> bool:
    """Exact endpoint-on-selected-track predicate used by CP and replay.

    Reservation intervals remain half-open for capacity accounting.  Physical
    line endpoints are closed, so a terminal at either end of the declared run
    is a valid conductor contact.
    """

    if (
        access.layer_n != resource.key.layer_n
        or access.layer_key != resource.gds_layer_key
    ):
        return False
    selected_centers = resource.track_centers_nm[
        track_start:track_start + use.track_span
    ]
    run_coordinate = (
        access.xy_nm[0]
        if resource.preferred_axis == "H"
        else access.xy_nm[1]
    )
    cross_coordinate = (
        access.xy_nm[1]
        if resource.preferred_axis == "H"
        else access.xy_nm[0]
    )
    return (
        use.interval_nm[0] <= run_coordinate <= use.interval_nm[1]
        and cross_coordinate in selected_centers
    )


def _track_choice_closed_bbox(
    use: CorridorUse,
    resource: TrackResource,
    track_start: int,
    width_nm: int,
) -> Optional[BBoxNm]:
    """Return the canonical closed metal envelope for one track choice.

    Reservation intervals remain half-open for capacity.  Electrical replay uses
    a closed physical envelope: the run endpoints are conductor endpoints and a
    multi-track span materializes as the connected slab enclosing its selected
    centers.  Odd widths use floor on the low side and ceil on the high side.
    """

    if type(track_start) is not int or type(width_nm) is not int or width_nm <= 0:
        return None
    selected_centers = resource.track_centers_nm[
        track_start:track_start + use.track_span
    ]
    if len(selected_centers) != use.track_span:
        return None
    low_half = width_nm // 2
    high_half = width_nm - low_half
    cross_low = selected_centers[0] - low_half
    cross_high = selected_centers[-1] + high_half
    if resource.preferred_axis == "H":
        return (
            use.interval_nm[0],
            cross_low,
            use.interval_nm[1],
            cross_high,
        )
    return (
        cross_low,
        use.interval_nm[0],
        cross_high,
        use.interval_nm[1],
    )


def _corridor_use_width_nm(
    use: CorridorUse, candidate_width_nm: int
) -> int:
    """Return the canonical per-use width, preserving old constructor input."""

    return candidate_width_nm if use.width_nm is None else use.width_nm


def _bbox_contains(outer: BBoxNm, inner: BBoxNm) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _closed_bboxes_touch(left: BBoxNm, right: BBoxNm) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def _track_choice_bridges_landing_pair(
    candidate: CompleteNetCandidate,
    use: CorridorUse,
    resource: TrackResource,
    track_start: int,
    left: LandingSite,
    right: LandingSite,
) -> bool:
    """Prove that one selected same-net track joins both close via pads.

    A metal-spacing conflict between two via pads disappears only when emitted
    metal physically closes the gap.  Net equality alone is insufficient: two
    disconnected polygons on the same logical net still violate a geometric
    spacing deck.  This predicate is shared by CP construction and independent
    replay so the exception is conditional on an exact allocated track choice.
    """

    if resource.gds_layer_key is None:
        return False
    track_bbox = _track_choice_closed_bbox(
        use,
        resource,
        track_start,
        _corridor_use_width_nm(use, candidate.width_nm),
    )
    if track_bbox is None:
        return False
    left_pads = {
        left.lower_layer_n: left.lower_pad_bbox_nm,
        left.upper_layer_n: left.upper_pad_bbox_nm,
    }
    right_pads = {
        right.lower_layer_n: right.lower_pad_bbox_nm,
        right.upper_layer_n: right.upper_pad_bbox_nm,
    }
    layer_n = resource.key.layer_n
    left_pad = left_pads.get(layer_n)
    right_pad = right_pads.get(layer_n)
    return (
        left_pad is not None
        and right_pad is not None
        and _closed_bboxes_touch(track_bbox, left_pad)
        and _closed_bboxes_touch(track_bbox, right_pad)
    )


def _track_choices_physically_touch(
    left_use: CorridorUse,
    left_resource: TrackResource,
    left_start: int,
    right_use: CorridorUse,
    right_resource: TrackResource,
    right_start: int,
    left_width_nm: int,
    right_width_nm: int,
) -> bool:
    if (
        left_resource.key.layer_n != right_resource.key.layer_n
        or left_resource.gds_layer_key is None
        or left_resource.gds_layer_key != right_resource.gds_layer_key
    ):
        return False
    left_bbox = _track_choice_closed_bbox(
        left_use, left_resource, left_start, left_width_nm
    )
    right_bbox = _track_choice_closed_bbox(
        right_use, right_resource, right_start, right_width_nm
    )
    return (
        left_bbox is not None
        and right_bbox is not None
        and _closed_bboxes_touch(left_bbox, right_bbox)
    )


def _consecutive_track_step_pairs(
    candidate: CompleteNetCandidate,
) -> tuple[tuple[CorridorUse, CorridorUse], ...]:
    uses_by_id = {use.use_id: use for use in candidate.corridor_uses}
    return tuple(
        (uses_by_id[left.ref_id], uses_by_id[right.ref_id])
        for branch in candidate.branches
        for left, right in zip(branch.ordered_steps, branch.ordered_steps[1:])
        if left.kind == "track_use" and right.kind == "track_use"
    )


def _landing_neighbour_use_ids(
    candidate: CompleteNetCandidate,
) -> dict[str, tuple[str, ...]]:
    neighbours: dict[str, set[str]] = {
        demand.demand_id: set() for demand in candidate.landing_demands
    }
    for branch in candidate.branches:
        for index, step in enumerate(branch.ordered_steps):
            if step.kind != "landing_demand":
                continue
            for neighbour_index in (index - 1, index + 1):
                if 0 <= neighbour_index < len(branch.ordered_steps):
                    neighbour = branch.ordered_steps[neighbour_index]
                    if neighbour.kind == "track_use":
                        neighbours[step.ref_id].add(neighbour.ref_id)
    return {
        demand_id: tuple(sorted(use_ids))
        for demand_id, use_ids in neighbours.items()
    }


def _owned_component_touches_track_bbox(
    component: OwnedConductorComponent, track_bbox: BBoxNm
) -> bool:
    return any(
        _closed_bboxes_touch(tuple(fact[:4]), track_bbox)
        for fact in component.atom_facts
    ) or any(
        track_bbox[0] <= pin[3] <= track_bbox[2]
        and track_bbox[1] <= pin[4] <= track_bbox[3]
        for pin in component.pin_accesses
    )


def _owned_component_conductor_touches_track_bbox(
    component: OwnedConductorComponent, track_bbox: BBoxNm
) -> bool:
    """Exact component/track contact; pin access alone is not a conductor."""

    return any(
        _closed_bboxes_touch(tuple(fact[:4]), track_bbox)
        for fact in component.atom_facts
    )


def _component_track_step_pairs(
    candidate: CompleteNetCandidate,
) -> tuple[tuple[str, CorridorUse], ...]:
    uses_by_id = {use.use_id: use for use in candidate.corridor_uses}
    pairs = []
    for branch in candidate.branches:
        for left, right in zip(branch.ordered_steps, branch.ordered_steps[1:]):
            if left.kind == "owned_component" and right.kind == "track_use":
                pairs.append((left.ref_id, uses_by_id[right.ref_id]))
            elif left.kind == "track_use" and right.kind == "owned_component":
                pairs.append((right.ref_id, uses_by_id[left.ref_id]))
    return tuple(pairs)


def _component_component_step_pairs(
    candidate: CompleteNetCandidate,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (left.ref_id, right.ref_id)
        for branch in candidate.branches
        for left, right in zip(branch.ordered_steps, branch.ordered_steps[1:])
        if left.kind == "owned_component" and right.kind == "owned_component"
    )


def _power_spine_root_proof_error(
    candidate: CompleteNetCandidate,
    certificates_by_ref: Mapping[
        str, OwnedConductorComponentCertificate
    ],
) -> Optional[str]:
    """Independently prove one common root on one exact wide rail atom."""

    if candidate.topology_class != "power_spine_tree":
        return None
    if not candidate.branches:
        return "power candidate has no closed branches"
    root_provenances = {
        branch.source_endpoint for branch in candidate.branches
    }
    first_steps = tuple(branch.ordered_steps[0] for branch in candidate.branches)
    component_refs = {
        step.ref_id for step in first_steps if step.kind == "owned_component"
    }
    if (
        len(root_provenances) != 1
        or len(component_refs) != 1
        or any(step.kind != "owned_component" for step in first_steps)
    ):
        return "power branches must share one root and first rail component"
    root_provenance = next(iter(root_provenances))
    accesses = {
        access.provenance: access for access in candidate.endpoint_accesses
    }
    root_access = accesses.get(root_provenance)
    component_ref = next(iter(component_refs))
    certificate = certificates_by_ref.get(component_ref)
    if root_access is None or certificate is None:
        return "power root or first rail component lacks exact provenance"
    component = certificate.component
    x, y = root_access.xy_nm
    if not any(
        type(atom) is tuple
        and len(atom) == 8
        and tuple(atom[4:6]) == tuple(component.layer_key)
        and atom[6] == candidate.net
        and atom[7] == "rail"
        and atom[0] <= x <= atom[2]
        and atom[1] <= y <= atom[3]
        and min(atom[2] - atom[0], atom[3] - atom[1])
        >= candidate.width_nm
        for atom in component.atom_facts
    ):
        return "power root lacks one exact rail atom with declared width"
    return None


def _endpoint_component_requirements(
    candidate: CompleteNetCandidate,
) -> tuple[tuple[CandidateEndpointAccess, str], ...]:
    accesses = {
        access.provenance: access for access in candidate.endpoint_accesses
    }
    requirements = []
    for branch in candidate.branches:
        for endpoint, step in (
            (branch.source_endpoint, branch.ordered_steps[0]),
            (branch.target_endpoint, branch.ordered_steps[-1]),
        ):
            if step.kind == "owned_component":
                requirements.append((accesses[endpoint], step.ref_id))
    return tuple(requirements)


def _landing_neighbour_component_refs(
    candidate: CompleteNetCandidate,
) -> dict[str, tuple[str, ...]]:
    neighbours: dict[str, set[str]] = {
        demand.demand_id: set() for demand in candidate.landing_demands
    }
    for branch in candidate.branches:
        for index, step in enumerate(branch.ordered_steps):
            if step.kind != "landing_demand":
                continue
            for neighbour_index in (index - 1, index + 1):
                if 0 <= neighbour_index < len(branch.ordered_steps):
                    neighbour = branch.ordered_steps[neighbour_index]
                    if neighbour.kind == "owned_component":
                        neighbours[step.ref_id].add(neighbour.ref_id)
    return {
        demand_id: tuple(sorted(component_refs))
        for demand_id, component_refs in neighbours.items()
    }


def _owned_component_touches_landing_option(
    component_ref: str,
    option: LandingOption,
    *,
    sites_by_id: Mapping[str, LandingSite],
) -> bool:
    return any(
        component_ref in (
            *sites_by_id[site_id].lower_owner_component_refs,
            *sites_by_id[site_id].upper_owner_component_refs,
        )
        for site_id in option.site_ids
    )


def _track_choice_touches_landing_option(
    *,
    candidate: CompleteNetCandidate,
    option: LandingOption,
    use: CorridorUse,
    resource: TrackResource,
    track_start: int,
    sites_by_id: Mapping[str, LandingSite],
    certificates_by_ref: Mapping[str, OwnedConductorComponentCertificate],
) -> bool:
    if any(
        contact.use_id == use.use_id
        and contact.track_start == track_start
        for contact in option.contacts
    ):
        return True
    track_bbox = _track_choice_closed_bbox(
        use,
        resource,
        track_start,
        _corridor_use_width_nm(use, candidate.width_nm),
    )
    if track_bbox is None or resource.gds_layer_key is None:
        return False
    for site_id in option.site_ids:
        site = sites_by_id[site_id]
        if site.owner_net != candidate.net:
            continue
        if resource.key.layer_n == site.lower_layer_n:
            refs = site.lower_owner_component_refs
        elif resource.key.layer_n == site.upper_layer_n:
            refs = site.upper_owner_component_refs
        else:
            continue
        for component_ref in refs:
            certificate = certificates_by_ref.get(component_ref)
            if certificate is None:
                continue
            component = certificate.component
            if (
                certificate.layer_n == resource.key.layer_n
                and component.layer_key == resource.gds_layer_key
                and component.net == candidate.net
                and _owned_component_touches_track_bbox(component, track_bbox)
            ):
                return True
    return False


def _track_choice_crosses_landing_site(
    use: CorridorUse,
    resource: TrackResource,
    track_start: int,
    site: LandingSite,
) -> bool:
    """Replay a derived contact from exact interval/span/site geometry."""

    if (
        resource.key.layer_n
        not in (site.lower_layer_n, site.upper_layer_n)
        or track_start not in use.allowed_track_starts
        or track_start + use.track_span > len(resource.track_centers_nm)
    ):
        return False
    selected_centers = resource.track_centers_nm[
        track_start:track_start + use.track_span
    ]
    cross_coordinate = (
        site.xy_nm[1]
        if resource.preferred_axis == "H"
        else site.xy_nm[0]
    )
    run_coordinate = (
        site.xy_nm[0]
        if resource.preferred_axis == "H"
        else site.xy_nm[1]
    )
    return (
        cross_coordinate in selected_centers
        and use.interval_nm[0] <= run_coordinate < use.interval_nm[1]
    )


def _track_choices_cross_landing_site(
    use: CorridorUse,
    resource: TrackResource,
    site: LandingSite,
) -> bool:
    """Test an exact implicit track choice without scanning every start."""

    if (
        resource.key.layer_n
        not in (site.lower_layer_n, site.upper_layer_n)
        or use.track_span > len(resource.track_centers_nm)
    ):
        return False
    cross_coordinate = (
        site.xy_nm[1]
        if resource.preferred_axis == "H"
        else site.xy_nm[0]
    )
    run_coordinate = (
        site.xy_nm[0]
        if resource.preferred_axis == "H"
        else site.xy_nm[1]
    )
    if not (use.interval_nm[0] <= run_coordinate < use.interval_nm[1]):
        return False
    center_index = bisect_left(resource.track_centers_nm, cross_coordinate)
    if (
        center_index >= len(resource.track_centers_nm)
        or resource.track_centers_nm[center_index] != cross_coordinate
    ):
        return False
    minimum_start = max(0, center_index - use.track_span + 1)
    maximum_start = min(
        center_index,
        len(resource.track_centers_nm) - use.track_span,
    )
    start_index = bisect_left(use.allowed_track_starts, minimum_start)
    return (
        start_index < len(use.allowed_track_starts)
        and use.allowed_track_starts[start_index] <= maximum_start
    )


def _derived_landing_site_replay_context(
    candidate: CompleteNetCandidate,
) -> tuple[
    Mapping[str, CorridorUse],
    Mapping[str, tuple[str, ...]],
    Mapping[str, tuple[str, ...]],
]:
    return (
        {use.use_id: use for use in candidate.corridor_uses},
        _landing_neighbour_use_ids(candidate),
        _landing_neighbour_component_refs(candidate),
    )


@dataclass
class _FactorizedLandingSiteReplayIndex:
    sites_by_id: Mapping[str, LandingSite]
    site_collection_sha256: str
    resources_by_key: Mapping[CorridorLayerKey, TrackResource]
    site_rows_by_layer_axis_cross: Mapping[
        tuple[int, str, int], tuple[tuple[int, str], ...]
    ]
    reachable_site_ids_by_use: dict[
        tuple[CorridorUse, TrackResource, frozenset[str] | None], frozenset[str]
    ]
    covered_track_centers_by_use: dict[
        tuple[CorridorUse, TrackResource], tuple[int, ...]
    ]
    allowed_site_id_sets: dict[tuple[str, ...], frozenset[str]]
    static_demand_site_id_sets: dict[
        tuple[
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[int, ...],
        ],
        frozenset[str],
    ]
    intersected_site_domains: dict[
        tuple[frozenset[str], ...], tuple[str, ...]
    ]
    site_rows_by_static_domain: dict[
        frozenset[str],
        Mapping[tuple[int, str, int], tuple[tuple[int, str], ...]],
    ]
    site_ids_by_xy_by_static_domain: dict[
        frozenset[str], Mapping[PointNm, tuple[str, ...]]
    ]
    intersected_geometry_domains: dict[
        tuple[
            frozenset[str],
            tuple[tuple[CorridorUse, TrackResource], ...],
        ],
        tuple[str, ...],
    ]
    derivation_reason_counts: dict[str, int]

    @classmethod
    def build(
        cls,
        landing_sites: Sequence[LandingSite] | LandingSiteCatalog,
        *,
        resources: Sequence[TrackResource],
        site_limit: int = MAX_LANDING_SITES,
    ) -> "_FactorizedLandingSiteReplayIndex":
        if (
            type(site_limit) is not int
            or site_limit < 0
        ):
            raise CandidatePoolLimitError(
                "factorized landing-site replay index site limit must be a "
                "non-negative exact integer"
            )
        if isinstance(landing_sites, LandingSiteCatalog):
            if landing_sites.site_count > min(
                site_limit, MAX_TOTAL_ASSIGNMENT_ARCS
            ):
                raise CandidatePoolLimitError(
                    "factorized landing-site replay index site count exceeds "
                    f"{min(site_limit, MAX_TOTAL_ASSIGNMENT_ARCS)}"
                )
            sites_by_id = {
                site.site_id: site
                for page in landing_sites.pages
                for site in page.sites
            }
            site_collection_sha256 = landing_sites.collection_sha256
        else:
            raw_sites = _bounded_materialize(
                landing_sites,
                limit=min(site_limit, MAX_TOTAL_ASSIGNMENT_ARCS),
                label="factorized landing-site replay index site",
                error_type=CandidatePoolLimitError,
            )
            if not all(isinstance(site, LandingSite) for site in raw_sites):
                raise CandidatePoolError(
                    "factorized landing-site replay index contains invalid sites"
                )
            sites = tuple(sorted(
                (_rebuild_landing_site(site) for site in raw_sites),
                key=lambda site: site.site_id,
            ))
            sites_by_id = {site.site_id: site for site in sites}
            if len(sites_by_id) != len(sites):
                raise CandidatePoolError(
                    "factorized landing-site replay index site ids must be unique"
                )
            site_collection_sha256 = landing_site_collection_sha256(sites)
        raw_resources = _bounded_materialize(
            resources,
            limit=MAX_TRACK_RESOURCES,
            label="factorized landing-site replay index track resource",
            error_type=CandidatePoolLimitError,
        )
        if not all(
            isinstance(resource, TrackResource) for resource in raw_resources
        ):
            raise CandidatePoolError(
                "factorized landing-site replay index contains invalid resources"
            )
        resource_rebuild_budget = _CandidatePoolRebuildBudget()
        rebuilt_resources = tuple(
            _rebuild_track_resource(
                resource,
                pool_budget=resource_rebuild_budget,
            )
            for resource in raw_resources
        )
        resources_by_key = {
            resource.key: resource for resource in rebuilt_resources
        }
        if len(resources_by_key) != len(rebuilt_resources):
            raise CandidatePoolError(
                "factorized landing-site replay index resource keys must be unique"
            )
        rows_by_key: dict[tuple[int, str, int], list[tuple[int, str]]] = {}
        for site in sites_by_id.values():
            for layer_n in {site.lower_layer_n, site.upper_layer_n}:
                rows_by_key.setdefault(
                    (layer_n, "H", site.xy_nm[1]), []
                ).append((site.xy_nm[0], site.site_id))
                rows_by_key.setdefault(
                    (layer_n, "V", site.xy_nm[0]), []
                ).append((site.xy_nm[1], site.site_id))
        return cls(
            sites_by_id=sites_by_id,
            site_collection_sha256=site_collection_sha256,
            resources_by_key=resources_by_key,
            site_rows_by_layer_axis_cross={
                key: tuple(sorted(rows))
                for key, rows in sorted(rows_by_key.items())
            },
            reachable_site_ids_by_use={},
            covered_track_centers_by_use={},
            allowed_site_id_sets={},
            static_demand_site_id_sets={},
            intersected_site_domains={},
            site_rows_by_static_domain={},
            site_ids_by_xy_by_static_domain={},
            intersected_geometry_domains={},
            derivation_reason_counts={},
        )

    def _consume_budget(
        self,
        derivation_budget: Optional[list[int]],
        count: int = 1,
        *,
        reason: str,
    ) -> None:
        if derivation_budget is None:
            return
        self.derivation_reason_counts[reason] = (
            self.derivation_reason_counts.get(reason, 0) + count
        )
        derivation_budget[0] += count
        if derivation_budget[0] > MAX_DERIVED_CONTINUITY_EVALUATIONS:
            reasons = tuple(sorted(
                self.derivation_reason_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:4])
            raise CandidatePoolLimitError(
                "factorized exact-site derivation evaluations exceed "
                f"{MAX_DERIVED_CONTINUITY_EVALUATIONS}; reasons={reasons}; "
                "cache_sizes="
                f"(uses={len(self.reachable_site_ids_by_use)}, "
                f"allowed={len(self.allowed_site_id_sets)}, "
                f"static={len(self.static_demand_site_id_sets)}, "
                f"intersections={len(self.intersected_site_domains)})"
            )

    def allowed_site_ids(
        self,
        site_ids: tuple[str, ...],
        *,
        derivation_budget: Optional[list[int]],
    ) -> frozenset[str]:
        cached = self.allowed_site_id_sets.get(site_ids)
        if cached is not None:
            return cached
        self._consume_budget(
            derivation_budget,
            len(site_ids),
            reason="allowed_site_domain",
        )
        unknown = tuple(
            site_id for site_id in site_ids if site_id not in self.sites_by_id
        )
        if unknown:
            raise CandidatePoolError(
                f"factorized demand names missing landing sites {unknown}"
            )
        result = frozenset(site_ids)
        self.allowed_site_id_sets[site_ids] = result
        return result

    def reachable_site_ids(
        self,
        use: CorridorUse,
        *,
        derivation_budget: Optional[list[int]],
        site_domain: Optional[frozenset[str]] = None,
    ) -> frozenset[str]:
        resource = self.resources_by_key.get(use.resource)
        if resource is None:
            raise CandidatePoolError(
                "factorized profile names missing resource "
                f"{use.resource!r}"
            )
        cache_key = (use, resource, site_domain)
        cached = self.reachable_site_ids_by_use.get(cache_key)
        if cached is not None:
            return cached
        if len(self.reachable_site_ids_by_use) >= MAX_TOTAL_ASSIGNMENT_ARCS:
            raise CandidatePoolLimitError(
                "factorized reachable-site cache count exceeds "
                f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
            )
        rows_by_key = self.site_rows_by_layer_axis_cross
        if site_domain is not None:
            rows_by_key = self.site_rows_by_static_domain.get(site_domain)
            if rows_by_key is None:
                domain_rows: dict[
                    tuple[int, str, int], list[tuple[int, str]]
                ] = {}
                for site_id in sorted(site_domain):
                    self._consume_budget(
                        derivation_budget,
                        4,
                        reason="static_domain_index_row",
                    )
                    site = self.sites_by_id[site_id]
                    for layer_n in {site.lower_layer_n, site.upper_layer_n}:
                        domain_rows.setdefault(
                            (layer_n, "H", site.xy_nm[1]), []
                        ).append((site.xy_nm[0], site.site_id))
                        domain_rows.setdefault(
                            (layer_n, "V", site.xy_nm[0]), []
                        ).append((site.xy_nm[1], site.site_id))
                rows_by_key = {
                    key: tuple(sorted(rows))
                    for key, rows in sorted(domain_rows.items())
                }
                self.site_rows_by_static_domain[site_domain] = rows_by_key
        covered_track_centers = self.covered_track_centers(
            use, derivation_budget=derivation_budget
        )
        reachable = set()
        interval_lo, interval_hi = use.interval_nm
        for cross_coordinate in covered_track_centers:
            self._consume_budget(
                derivation_budget, reason="track_coordinate_query"
            )
            rows = rows_by_key.get(
                (
                    resource.key.layer_n,
                    resource.preferred_axis,
                    cross_coordinate,
                ),
                (),
            )
            lo = bisect_left(rows, (interval_lo, ""))
            hi = bisect_left(rows, (interval_hi, ""))
            self._consume_budget(
                derivation_budget,
                hi - lo,
                reason="track_coordinate_hit",
            )
            reachable.update(site_id for _run, site_id in rows[lo:hi])
        result = frozenset(reachable)
        self.reachable_site_ids_by_use[cache_key] = result
        return result

    def covered_track_centers(
        self,
        use: CorridorUse,
        *,
        derivation_budget: Optional[list[int]],
    ) -> tuple[int, ...]:
        resource = self.resources_by_key.get(use.resource)
        if resource is None:
            raise CandidatePoolError(
                "factorized profile names missing resource "
                f"{use.resource!r}"
            )
        key = (use, resource)
        cached = self.covered_track_centers_by_use.get(key)
        if cached is not None:
            return cached
        covered_track_indices = set()
        for track_start in use.allowed_track_starts:
            self._consume_budget(
                derivation_budget,
                use.track_span,
                reason="track_span_expansion",
            )
            if track_start + use.track_span > len(resource.track_centers_nm):
                continue
            covered_track_indices.update(range(
                track_start, track_start + use.track_span
            ))
        result = tuple(
            resource.track_centers_nm[index]
            for index in sorted(covered_track_indices)
        )
        self.covered_track_centers_by_use[key] = result
        return result

    def static_demand_site_ids(
        self,
        *,
        candidate_net: str,
        allowed_site_ids: tuple[str, ...],
        neighbour_component_refs: tuple[str, ...],
        neighbour_uses: tuple[CorridorUse, ...],
        derivation_budget: Optional[list[int]],
    ) -> frozenset[str]:
        resource_layer_values = []
        for use in neighbour_uses:
            resource = self.resources_by_key.get(use.resource)
            if resource is None:
                raise CandidatePoolError(
                    "factorized profile names missing resource "
                    f"{use.resource!r}"
                )
            resource_layer_values.append(resource.key.layer_n)
        resource_layer_ns = tuple(sorted(resource_layer_values))
        key = (
            candidate_net,
            allowed_site_ids,
            tuple(sorted(neighbour_component_refs)),
            resource_layer_ns,
        )
        cached = self.static_demand_site_id_sets.get(key)
        if cached is not None:
            return cached
        allowed = self.allowed_site_ids(
            allowed_site_ids,
            derivation_budget=derivation_budget,
        )
        required_components = set(neighbour_component_refs)
        accepted = []
        for site_id in sorted(allowed):
            self._consume_budget(
                derivation_budget, reason="static_site_filter"
            )
            site = self.sites_by_id[site_id]
            site_component_refs = {
                *site.lower_owner_component_refs,
                *site.upper_owner_component_refs,
            }
            if required_components - site_component_refs:
                continue
            covered_layers = set(resource_layer_ns)
            if site.owner_net == candidate_net and site.lower_owner_component_refs:
                covered_layers.add(site.lower_layer_n)
            if site.owner_net == candidate_net and site.upper_owner_component_refs:
                covered_layers.add(site.upper_layer_n)
            if {
                site.lower_layer_n,
                site.upper_layer_n,
            }.issubset(covered_layers):
                accepted.append(site_id)
        result = frozenset(accepted)
        self.static_demand_site_id_sets[key] = result
        return result

    def intersect_site_domains(
        self,
        domains: tuple[frozenset[str], ...],
        *,
        derivation_budget: Optional[list[int]],
    ) -> tuple[str, ...]:
        self._consume_budget(
            derivation_budget,
            len(domains),
            reason="intersection_cache_lookup",
        )
        cached = self.intersected_site_domains.get(domains)
        if cached is not None:
            return cached
        seed_index = min(
            range(len(domains)),
            key=lambda index: (len(domains[index]), index),
        )
        other_domains = tuple(
            domain
            for index, domain in enumerate(domains)
            if index != seed_index
        )
        accepted = []
        for site_id in sorted(domains[seed_index]):
            self._consume_budget(
                derivation_budget, reason="intersection_seed"
            )
            present_in_all = True
            for domain in other_domains:
                self._consume_budget(
                    derivation_budget, reason="intersection_membership"
                )
                if site_id not in domain:
                    present_in_all = False
                    break
            if present_in_all:
                accepted.append(site_id)
        if len(self.intersected_site_domains) >= MAX_TOTAL_ASSIGNMENT_ARCS:
            raise CandidatePoolLimitError(
                "factorized exact-domain intersection cache count exceeds "
                f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
            )
        result = tuple(accepted)
        self.intersected_site_domains[domains] = result
        return result

    def exact_geometry_site_ids(
        self,
        static_site_ids: frozenset[str],
        neighbour_uses: tuple[CorridorUse, ...],
        *,
        derivation_budget: Optional[list[int]],
    ) -> tuple[str, ...]:
        bindings = tuple(
            (use, self.resources_by_key[use.resource])
            for use in neighbour_uses
        )
        key = (static_site_ids, bindings)
        self._consume_budget(
            derivation_budget,
            len(bindings),
            reason="geometry_cache_lookup",
        )
        cached = self.intersected_geometry_domains.get(key)
        if cached is not None:
            return cached
        if len(self.intersected_geometry_domains) >= MAX_TOTAL_ASSIGNMENT_ARCS:
            raise CandidatePoolLimitError(
                "factorized exact geometry-domain cache count exceeds "
                f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
            )
        first_horizontal_index: Optional[int] = None
        first_vertical_index: Optional[int] = None
        perpendicular_pair: Optional[tuple[int, int]] = None
        for index, (_use, resource) in enumerate(bindings):
            if resource.preferred_axis == "H":
                if first_vertical_index is not None:
                    perpendicular_pair = (first_vertical_index, index)
                    break
                if first_horizontal_index is None:
                    first_horizontal_index = index
            else:
                if first_horizontal_index is not None:
                    perpendicular_pair = (first_horizontal_index, index)
                    break
                if first_vertical_index is None:
                    first_vertical_index = index
        if perpendicular_pair is None:
            domains = (static_site_ids, *(
                self.reachable_site_ids(
                    use,
                    derivation_budget=derivation_budget,
                    site_domain=static_site_ids,
                )
                for use, _resource in bindings
            ))
            result = self.intersect_site_domains(
                domains, derivation_budget=derivation_budget
            )
            self.intersected_geometry_domains[key] = result
            return result

        left_index, right_index = perpendicular_pair
        left_use, left_resource = bindings[left_index]
        right_use, right_resource = bindings[right_index]
        if left_resource.preferred_axis == "H":
            horizontal_use, horizontal_resource = left_use, left_resource
            vertical_use, vertical_resource = right_use, right_resource
        else:
            horizontal_use, horizontal_resource = right_use, right_resource
            vertical_use, vertical_resource = left_use, left_resource
        horizontal_centers = self.covered_track_centers(
            horizontal_use, derivation_budget=derivation_budget
        )
        vertical_centers = self.covered_track_centers(
            vertical_use, derivation_budget=derivation_budget
        )
        point_index = self.site_ids_by_xy_by_static_domain.get(static_site_ids)
        if point_index is None:
            points: dict[PointNm, list[str]] = {}
            for site_id in sorted(static_site_ids):
                self._consume_budget(
                    derivation_budget, reason="static_point_index_row"
                )
                site = self.sites_by_id[site_id]
                points.setdefault(site.xy_nm, []).append(site_id)
            point_index = {
                point: tuple(sorted(site_ids))
                for point, site_ids in sorted(points.items())
            }
            self.site_ids_by_xy_by_static_domain[static_site_ids] = point_index
        other_bindings = tuple(
            binding
            for index, binding in enumerate(bindings)
            if index not in perpendicular_pair
        )
        accepted = set()
        for y_nm in horizontal_centers:
            for x_nm in vertical_centers:
                self._consume_budget(
                    derivation_budget, reason="perpendicular_point_query"
                )
                if not (
                    horizontal_use.interval_nm[0]
                    <= x_nm
                    < horizontal_use.interval_nm[1]
                    and vertical_use.interval_nm[0]
                    <= y_nm
                    < vertical_use.interval_nm[1]
                ):
                    continue
                for site_id in point_index.get((x_nm, y_nm), ()):
                    self._consume_budget(
                        derivation_budget, reason="perpendicular_point_hit"
                    )
                    site = self.sites_by_id[site_id]
                    if (
                        horizontal_resource.key.layer_n
                        not in (site.lower_layer_n, site.upper_layer_n)
                        or vertical_resource.key.layer_n
                        not in (site.lower_layer_n, site.upper_layer_n)
                    ):
                        continue
                    additional_uses_cross = True
                    for use, resource in other_bindings:
                        self._consume_budget(
                            derivation_budget,
                            1,
                            reason="additional_use_probe",
                        )
                        if not _track_choices_cross_landing_site(
                            use, resource, site
                        ):
                            additional_uses_cross = False
                            break
                    if additional_uses_cross:
                        accepted.add(site_id)
        result = tuple(sorted(accepted))
        self.intersected_geometry_domains[key] = result
        return result


def _derived_landing_site_domain_error(
    candidate: CompleteNetCandidate,
    demand_id: str,
    site: LandingSite,
    *,
    resources_by_key: Mapping[CorridorLayerKey, TrackResource],
    selected_track_starts: Optional[Mapping[str, int]] = None,
    replay_context: Optional[
        tuple[
            Mapping[str, CorridorUse],
            Mapping[str, tuple[str, ...]],
            Mapping[str, tuple[str, ...]],
        ]
    ] = None,
    prevalidated_use_crossings: bool = False,
) -> Optional[str]:
    """Return why one implicit site atom is not exact, or ``None``."""

    if replay_context is None:
        replay_context = _derived_landing_site_replay_context(candidate)
    uses_by_id, neighbour_uses, neighbour_components = replay_context
    neighbour_use_ids = neighbour_uses[demand_id]
    neighbour_component_refs = neighbour_components[demand_id]
    site_component_refs = {
        *site.lower_owner_component_refs,
        *site.upper_owner_component_refs,
    }
    missing_components = tuple(sorted(
        set(neighbour_component_refs) - site_component_refs
    ))
    if missing_components:
        return f"site lacks adjacent owned components {missing_components}"

    covered_layers = set()
    if site.owner_net == candidate.net and site.lower_owner_component_refs:
        covered_layers.add(site.lower_layer_n)
    if site.owner_net == candidate.net and site.upper_owner_component_refs:
        covered_layers.add(site.upper_layer_n)
    for use_id in neighbour_use_ids:
        use = uses_by_id[use_id]
        resource = resources_by_key[use.resource]
        if not prevalidated_use_crossings:
            crosses_site = (
                _track_choices_cross_landing_site(use, resource, site)
                if selected_track_starts is None
                else _track_choice_crosses_landing_site(
                    use,
                    resource,
                    selected_track_starts[use_id],
                    site,
                )
            )
            if not crosses_site:
                return f"site has no exact contact to adjacent use {use_id!r}"
        covered_layers.add(resource.key.layer_n)
    missing_layers = tuple(
        layer_n
        for layer_n in (site.lower_layer_n, site.upper_layer_n)
        if layer_n not in covered_layers
    )
    if missing_layers:
        return f"site has uncovered pad layers {missing_layers}"
    return None


def _endpoint_track_requirements(
    candidate: CompleteNetCandidate,
) -> tuple[tuple[CandidateEndpointAccess, str], ...]:
    accesses = {
        access.provenance: access for access in candidate.endpoint_accesses
    }
    requirements: list[tuple[CandidateEndpointAccess, str]] = []
    for branch in candidate.branches:
        for endpoint, step in (
            (branch.source_endpoint, branch.ordered_steps[0]),
            (branch.target_endpoint, branch.ordered_steps[-1]),
        ):
            if step.kind == "track_use":
                requirements.append((accesses[endpoint], step.ref_id))
    return tuple(requirements)


@dataclass(frozen=True)
class CandidatePairConflict:
    left_candidate_id: str
    right_candidate_id: str
    declaration_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = (self.left_candidate_id, self.right_candidate_id)
        if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in ids):
            raise CandidatePoolError("pair conflict candidate ids must be non-empty")
        if ids[0] == ids[1]:
            raise CandidatePoolError("pair conflict must name two distinct candidates")
        raw_refs = _bounded_materialize(
            self.declaration_refs,
            limit=MAX_CANDIDATE_PAIR_DECLARATION_REFS,
            label="pair conflict declaration ref",
            error_type=CandidatePoolLimitError,
        )
        refs = tuple(str(ref) for ref in raw_refs)
        if any(not ref for ref in refs) or len(refs) != len(set(refs)):
            raise CandidatePoolError("pair conflict declaration refs must be unique and non-empty")
        left, right = sorted(ids)
        object.__setattr__(self, "left_candidate_id", left)
        object.__setattr__(self, "right_candidate_id", right)
        object.__setattr__(self, "declaration_refs", tuple(sorted(refs)))


@dataclass(frozen=True, order=True)
class FactorizedCandidateUseRole:
    use_id: str
    resource: CorridorLayerKey
    track_span: int
    width_nm: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.use_id, str) or not self.use_id:
            raise CandidatePoolError("factorized use role id must be non-empty")
        if not isinstance(self.resource, CorridorLayerKey):
            raise CandidatePoolError("factorized use role resource is invalid")
        if not _is_int_nm(self.track_span) or self.track_span <= 0:
            raise CandidatePoolError("factorized use role span must be positive")
        if self.width_nm is not None and (
            not _is_int_nm(self.width_nm) or self.width_nm <= 0
        ):
            raise CandidatePoolError("factorized use role width must be positive")


@dataclass(frozen=True, order=True)
class FactorizedCandidateDemandRole:
    demand_id: str
    terminal_provenance: str
    allowed_site_ids: tuple[str, ...]
    declaration_ref: Optional[str]
    required_multiplicity: int

    def __post_init__(self) -> None:
        demand = LandingDemand(
            demand_id=self.demand_id,
            terminal_provenance=self.terminal_provenance,
            allowed_site_ids=self.allowed_site_ids,
            declaration_ref=self.declaration_ref,
            required_multiplicity=self.required_multiplicity,
            selection_mode=LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
        )
        object.__setattr__(self, "allowed_site_ids", demand.allowed_site_ids)


@dataclass(frozen=True, order=True)
class FactorizedCandidateUseProfile:
    use_id: str
    interval_nm: tuple[int, int]
    allowed_track_starts: tuple[int, ...]

    def __post_init__(self) -> None:
        interval = _bounded_materialize(
            self.interval_nm,
            limit=2,
            label="factorized use profile interval coordinate",
            error_type=CandidatePoolLimitError,
        )
        raw_starts = _bounded_materialize(
            self.allowed_track_starts,
            limit=MAX_CORRIDOR_USE_ALLOWED_TRACK_STARTS,
            label="factorized use profile track start",
            error_type=CandidatePoolLimitError,
        )
        starts = tuple(sorted(raw_starts))
        if (
            not isinstance(self.use_id, str)
            or not self.use_id
            or len(interval) != 2
            or not all(_is_int_nm(value) for value in interval)
            or interval[1] <= interval[0]
            or not starts
            or any(not _is_int_nm(start) or start < 0 for start in starts)
            or len(starts) != len(set(starts))
        ):
            raise CandidatePoolError("factorized use profile is malformed")
        object.__setattr__(self, "interval_nm", interval)
        object.__setattr__(self, "allowed_track_starts", starts)


@dataclass(frozen=True)
class FactorizedCandidateProfile:
    profile_id: str
    candidate_id: str
    use_profiles: tuple[FactorizedCandidateUseProfile, ...]

    def __post_init__(self) -> None:
        raw_rows = _bounded_materialize(
            self.use_profiles,
            limit=MAX_TOTAL_CORRIDOR_USES,
            label="factorized profile use row",
            error_type=CandidatePoolLimitError,
        )
        if not all(
            isinstance(row, FactorizedCandidateUseProfile) for row in raw_rows
        ):
            raise CandidatePoolError(
                "factorized profile use rows contain invalid values"
            )
        rows = tuple(sorted(raw_rows, key=lambda row: row.use_id))
        if (
            not isinstance(self.profile_id, str)
            or not self.profile_id
            or not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or not rows
            or len({row.use_id for row in rows}) != len(rows)
        ):
            raise CandidatePoolError("factorized candidate profile is malformed")
        expected = hashlib.sha256(json.dumps({
            "schema": "clay.factorized-candidate-profile.v1",
            "use_profiles": [
                {
                    "allowed_track_starts": list(row.allowed_track_starts),
                    "interval_nm": list(row.interval_nm),
                    "use_id": row.use_id,
                }
                for row in rows
            ],
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if self.profile_id != expected:
            raise CandidatePoolError(
                "factorized candidate profile_id is not content-addressed"
            )
        object.__setattr__(self, "use_profiles", rows)


@dataclass(frozen=True)
class FactorizedCompleteNetCandidateFamily:
    family_id: str
    net: str
    topology_class: str
    program_kind: str
    required_goal_refs: tuple[str, ...]
    width_nm: int
    via_multiplicity: int
    source_topology_digest: str
    use_roles: tuple[FactorizedCandidateUseRole, ...]
    demand_roles: tuple[FactorizedCandidateDemandRole, ...]
    branches: tuple[CandidateBranch, ...]
    goal_branch_coverage: tuple[GoalBranchCoverage, ...]
    endpoint_accesses: tuple[CandidateEndpointAccess, ...]
    owned_component_use_refs: tuple[str, ...]
    profiles: tuple[FactorizedCandidateProfile, ...]

    def __post_init__(self) -> None:
        for name in (
            "family_id", "net", "topology_class", "program_kind",
            "source_topology_digest",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise CandidatePoolError(f"factorized family {name} must be non-empty")
        if self.topology_class not in {
            "local_signal", "shared_signal_trunk", "power_spine_tree"
        }:
            raise CandidatePoolError("factorized family topology class is invalid")
        if not _is_int_nm(self.width_nm) or self.width_nm <= 0:
            raise CandidatePoolError(
                "factorized family width_nm must be positive integer nm"
            )
        if (
            not _is_int_nm(self.via_multiplicity)
            or self.via_multiplicity < 0
        ):
            raise CandidatePoolError(
                "factorized family via_multiplicity must be non-negative"
            )
        raw_uses = _bounded_materialize(
            self.use_roles,
            limit=MAX_TOTAL_CORRIDOR_USES,
            label="factorized family use role",
            error_type=CandidatePoolLimitError,
        )
        raw_demands = _bounded_materialize(
            self.demand_roles,
            limit=MAX_TOTAL_LANDING_DEMANDS,
            label="factorized family demand role",
            error_type=CandidatePoolLimitError,
        )
        raw_branches = _bounded_materialize(
            self.branches,
            limit=MAX_TOTAL_CANDIDATE_BRANCHES,
            label="factorized family branch",
            error_type=CandidatePoolLimitError,
        )
        raw_profiles = _bounded_materialize(
            self.profiles,
            limit=MAX_TOTAL_ASSIGNMENT_ARCS,
            label="factorized family profile",
            error_type=CandidatePoolLimitError,
        )
        raw_required_goal_refs = _bounded_materialize(
            self.required_goal_refs,
            limit=MAX_REQUIRED_GOAL_REFS,
            label="factorized family required goal ref",
            error_type=CandidatePoolLimitError,
        )
        raw_goal_coverage = _bounded_materialize(
            self.goal_branch_coverage,
            limit=MAX_TOTAL_GOAL_BRANCH_COVERAGE,
            label="factorized family goal branch coverage",
            error_type=CandidatePoolLimitError,
        )
        raw_endpoint_accesses = _bounded_materialize(
            self.endpoint_accesses,
            limit=MAX_TOTAL_ENDPOINT_ACCESSES,
            label="factorized family endpoint access",
            error_type=CandidatePoolLimitError,
        )
        raw_owned_component_refs = _bounded_materialize(
            self.owned_component_use_refs,
            limit=MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE,
            label="factorized family owned component use ref",
            error_type=CandidatePoolLimitError,
        )
        if (
            not all(isinstance(role, FactorizedCandidateUseRole) for role in raw_uses)
            or not all(
                isinstance(role, FactorizedCandidateDemandRole)
                for role in raw_demands
            )
            or not all(isinstance(branch, CandidateBranch) for branch in raw_branches)
            or not all(
                isinstance(profile, FactorizedCandidateProfile)
                for profile in raw_profiles
            )
            or not all(
                isinstance(item, GoalBranchCoverage)
                for item in raw_goal_coverage
            )
            or not all(
                isinstance(access, CandidateEndpointAccess)
                for access in raw_endpoint_accesses
            )
            or any(
                not isinstance(ref, str) or not ref
                for ref in (
                    *raw_required_goal_refs,
                    *raw_owned_component_refs,
                )
            )
        ):
            raise CandidatePoolError(
                "factorized family contains invalid role, branch, or profile values"
            )
        uses = tuple(sorted(raw_uses, key=lambda role: role.use_id))
        demands = tuple(sorted(raw_demands, key=lambda role: role.demand_id))
        branches = tuple(sorted(raw_branches, key=lambda branch: branch.branch_id))
        profiles = tuple(sorted(
            raw_profiles, key=lambda profile: profile.profile_id
        ))
        required_goal_refs = tuple(sorted(raw_required_goal_refs))
        goal_coverage = tuple(sorted(raw_goal_coverage))
        endpoint_accesses = tuple(sorted(
            raw_endpoint_accesses, key=lambda access: access.provenance
        ))
        owned_component_refs = tuple(sorted(raw_owned_component_refs))
        if (
            not profiles
            or len({role.use_id for role in uses}) != len(uses)
            or len({role.demand_id for role in demands}) != len(demands)
            or len({profile.profile_id for profile in profiles}) != len(profiles)
            or len({profile.candidate_id for profile in profiles}) != len(profiles)
            or len(required_goal_refs) != len(set(required_goal_refs))
            or len(owned_component_refs) != len(set(owned_component_refs))
        ):
            raise CandidatePoolError("factorized family roles/profiles must be unique")
        use_ids = {role.use_id for role in uses}
        demand_ids = {role.demand_id for role in demands}
        for profile in profiles:
            if {row.use_id for row in profile.use_profiles} != use_ids:
                raise CandidatePoolError(
                    "factorized profile use rows do not match family roles"
                )
        for branch in branches:
            if set(branch.ordered_use_ids) - use_ids or set(
                branch.ordered_landing_demand_ids
            ) - demand_ids:
                raise CandidatePoolError(
                    "factorized family branch names an unknown role"
                )
        object.__setattr__(self, "use_roles", uses)
        object.__setattr__(self, "demand_roles", demands)
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "required_goal_refs", required_goal_refs)
        object.__setattr__(self, "goal_branch_coverage", goal_coverage)
        object.__setattr__(self, "endpoint_accesses", endpoint_accesses)
        object.__setattr__(self, "owned_component_use_refs", owned_component_refs)
        expected = hashlib.sha256(json.dumps(
            _factorized_candidate_family_payload(self, include_family_id=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if self.family_id != expected:
            raise CandidatePoolError(
                "factorized family_id is not the content-addressed family hash"
            )


def _rebuild_factorized_candidate_family(
    family: FactorizedCompleteNetCandidateFamily,
    *,
    pool_budget: Optional[_CandidatePoolRebuildBudget] = None,
) -> FactorizedCompleteNetCandidateFamily:
    """Replay every nested row before trusting a factorized family payload."""

    if not isinstance(family, FactorizedCompleteNetCandidateFamily):
        raise CandidatePoolError(
            "factorized candidate family has invalid type"
        )
    assignment_label = (
        "factorized assignment arc count for family "
        f"net={family.net!r} family_id={family.family_id}"
    )
    raw_uses = _bounded_materialize_with_pool_budget(
        family.use_roles,
        limit=MAX_TOTAL_CORRIDOR_USES,
        label="factorized family use role",
        pool_budget=pool_budget,
        budget_field="factorized_corridor_uses",
        aggregate_limit=MAX_TOTAL_CORRIDOR_USES,
        aggregate_label="factorized corridor use count",
    )
    raw_demands = _bounded_materialize_with_pool_budget(
        family.demand_roles,
        limit=MAX_TOTAL_LANDING_DEMANDS,
        label="factorized family demand role",
        pool_budget=pool_budget,
        budget_field="factorized_landing_demands",
        aggregate_limit=MAX_TOTAL_LANDING_DEMANDS,
        aggregate_label="factorized landing demand count",
    )
    raw_branches = _bounded_materialize_with_pool_budget(
        family.branches,
        limit=MAX_TOTAL_CANDIDATE_BRANCHES,
        label="factorized family branch",
        pool_budget=pool_budget,
        budget_field="factorized_branches",
        aggregate_limit=MAX_TOTAL_CANDIDATE_BRANCHES,
        aggregate_label="factorized branch count",
    )
    raw_coverage = _bounded_materialize_with_pool_budget(
        family.goal_branch_coverage,
        limit=MAX_TOTAL_GOAL_BRANCH_COVERAGE,
        label="factorized family goal branch coverage",
        pool_budget=pool_budget,
        budget_field="factorized_goal_branch_coverage",
        aggregate_limit=MAX_TOTAL_GOAL_BRANCH_COVERAGE,
        aggregate_label="factorized goal branch coverage count",
    )
    raw_accesses = _bounded_materialize_with_pool_budget(
        family.endpoint_accesses,
        limit=MAX_TOTAL_ENDPOINT_ACCESSES,
        label="factorized family endpoint access",
        pool_budget=pool_budget,
        budget_field="factorized_endpoint_accesses",
        aggregate_limit=MAX_TOTAL_ENDPOINT_ACCESSES,
        aggregate_label="factorized endpoint access count",
    )
    raw_profiles = _bounded_materialize_with_pool_budget(
        family.profiles,
        limit=MAX_TOTAL_ASSIGNMENT_ARCS,
        label="factorized family profile",
        pool_budget=pool_budget,
        budget_field="factorized_assignment_arcs",
        aggregate_limit=MAX_TOTAL_ASSIGNMENT_ARCS,
        aggregate_label=f"{assignment_label} category=profiles",
    )
    if (
        not all(isinstance(role, FactorizedCandidateUseRole) for role in raw_uses)
        or not all(
            isinstance(role, FactorizedCandidateDemandRole)
            for role in raw_demands
        )
        or not all(isinstance(branch, CandidateBranch) for branch in raw_branches)
        or not all(
            isinstance(item, GoalBranchCoverage) for item in raw_coverage
        )
        or not all(
            isinstance(access, CandidateEndpointAccess) for access in raw_accesses
        )
        or not all(
            isinstance(profile, FactorizedCandidateProfile)
            for profile in raw_profiles
        )
    ):
        raise CandidatePoolError(
            "factorized candidate family contains invalid nested values"
        )

    rebuilt_branches = []
    for branch in raw_branches:
        raw_steps = _bounded_materialize_with_pool_budget(
            branch.ordered_steps,
            limit=MAX_CANDIDATE_BRANCH_STEPS,
            label="factorized family branch step",
            pool_budget=pool_budget,
            budget_field="factorized_branch_steps",
            aggregate_limit=MAX_TOTAL_BRANCH_STEPS,
            aggregate_label="factorized branch step count",
        )
        raw_ordered_use_ids = _bounded_materialize_with_pool_budget(
            branch.ordered_use_ids,
            limit=MAX_CANDIDATE_BRANCH_ORDERED_USES,
            label="factorized family branch ordered use id",
            pool_budget=pool_budget,
            budget_field="factorized_branch_ordered_uses",
            aggregate_limit=MAX_TOTAL_BRANCH_STEPS,
            aggregate_label="factorized branch ordered use id count",
        )
        raw_ordered_demand_ids = _bounded_materialize_with_pool_budget(
            branch.ordered_landing_demand_ids,
            limit=MAX_CANDIDATE_BRANCH_ORDERED_LANDING_DEMANDS,
            label="factorized family branch ordered landing demand id",
            pool_budget=pool_budget,
            budget_field="factorized_branch_ordered_demands",
            aggregate_limit=MAX_TOTAL_BRANCH_STEPS,
            aggregate_label=(
                "factorized branch ordered landing demand id count"
            ),
        )
        if not all(isinstance(step, CandidateBranchStep) for step in raw_steps):
            raise CandidatePoolError(
                "factorized candidate family branch contains invalid steps"
            )
        rebuilt_branches.append(replace(
            branch,
            ordered_use_ids=raw_ordered_use_ids,
            ordered_landing_demand_ids=raw_ordered_demand_ids,
            ordered_steps=tuple(replace(step) for step in raw_steps),
        ))

    rebuilt_profiles = []
    for profile in raw_profiles:
        raw_rows = _bounded_materialize_with_pool_budget(
            profile.use_profiles,
            limit=MAX_TOTAL_CORRIDOR_USES,
            label="factorized profile use row",
            pool_budget=pool_budget,
            budget_field="factorized_ir_items",
            aggregate_limit=MAX_TOTAL_ASSIGNMENT_ARCS,
            aggregate_label=(
                "factorized IR item count for family "
                f"net={family.net!r} family_id={family.family_id} "
                "category=profile_rows"
            ),
        )
        if not all(
            isinstance(row, FactorizedCandidateUseProfile) for row in raw_rows
        ):
            raise CandidatePoolError(
                "factorized candidate profile contains invalid use rows"
            )
        rebuilt_rows = []
        for row in raw_rows:
            raw_starts = _bounded_materialize_with_pool_budget(
                row.allowed_track_starts,
                limit=MAX_CORRIDOR_USE_ALLOWED_TRACK_STARTS,
                label="factorized use profile track start",
                pool_budget=pool_budget,
                budget_field="factorized_assignment_arcs",
                aggregate_limit=MAX_TOTAL_ASSIGNMENT_ARCS,
                aggregate_label=f"{assignment_label} category=track_starts",
            )
            rebuilt_rows.append(replace(
                row,
                interval_nm=_bounded_materialize(
                    row.interval_nm,
                    limit=2,
                    label="factorized use profile interval coordinate",
                    error_type=CandidatePoolLimitError,
                ),
                allowed_track_starts=raw_starts,
            ))
        rebuilt_profiles.append(replace(
            profile,
            use_profiles=tuple(rebuilt_rows),
        ))

    rebuilt_demands = []
    for role in raw_demands:
        raw_allowed_site_ids = _bounded_materialize_with_pool_budget(
            role.allowed_site_ids,
            limit=MAX_LANDING_DEMAND_ALLOWED_SITES,
            label="factorized demand role allowed site id",
            pool_budget=pool_budget,
            budget_field="factorized_ir_items",
            aggregate_limit=MAX_TOTAL_ASSIGNMENT_ARCS,
            aggregate_label=(
                "factorized IR item count for family "
                f"net={family.net!r} family_id={family.family_id} "
                "category=demand_sites"
            ),
        )
        rebuilt_demands.append(replace(
            role,
            allowed_site_ids=raw_allowed_site_ids,
        ))

    return replace(
        family,
        required_goal_refs=_bounded_materialize_with_pool_budget(
            family.required_goal_refs,
            limit=MAX_REQUIRED_GOAL_REFS,
            label="factorized family required goal ref",
            pool_budget=pool_budget,
            budget_field="factorized_required_goal_refs",
            aggregate_limit=MAX_REQUIRED_GOAL_REFS,
            aggregate_label="factorized required goal ref count",
        ),
        use_roles=tuple(replace(
            role,
            resource=replace(role.resource),
        ) for role in raw_uses),
        demand_roles=tuple(rebuilt_demands),
        branches=tuple(rebuilt_branches),
        goal_branch_coverage=tuple(replace(item) for item in raw_coverage),
        endpoint_accesses=tuple(replace(
            access,
            xy_nm=_bounded_materialize(
                access.xy_nm,
                limit=2,
                label="factorized endpoint access coordinate",
                error_type=CandidatePoolLimitError,
            ),
            layer_key=_bounded_materialize(
                access.layer_key,
                limit=2,
                label="factorized endpoint access layer key coordinate",
                error_type=CandidatePoolLimitError,
            ),
            owner_component_refs=_bounded_materialize_with_pool_budget(
                access.owner_component_refs,
                limit=MAX_ENDPOINT_OWNER_COMPONENT_REFS,
                label="factorized endpoint access owner component ref",
                pool_budget=pool_budget,
                budget_field="endpoint_owner_component_refs",
                aggregate_limit=MAX_TOTAL_ENDPOINT_OWNER_COMPONENT_REFS,
                aggregate_label="endpoint owner component ref count",
            ),
        ) for access in raw_accesses),
        owned_component_use_refs=_bounded_materialize_with_pool_budget(
            family.owned_component_use_refs,
            limit=MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE,
            label="factorized family owned component use ref",
            pool_budget=pool_budget,
            budget_field="factorized_owned_component_use_refs",
            aggregate_limit=MAX_TOTAL_OWNED_COMPONENT_USE_REFS,
            aggregate_label="factorized owned component use ref count",
        ),
        profiles=tuple(rebuilt_profiles),
    )


def _factorized_candidate_family_payload(
    family: FactorizedCompleteNetCandidateFamily,
    *,
    include_family_id: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "clay.factorized-complete-net-candidate-family.v1",
        "net": family.net,
        "topology_class": family.topology_class,
        "program_kind": family.program_kind,
        "required_goal_refs": list(family.required_goal_refs),
        "width_nm": family.width_nm,
        "via_multiplicity": family.via_multiplicity,
        "source_topology_digest": family.source_topology_digest,
        "use_roles": [
            {
                "use_id": role.use_id,
                "resource": _key_payload(role.resource),
                "track_span": role.track_span,
                "width_nm": role.width_nm,
            }
            for role in family.use_roles
        ],
        "demand_roles": [
            {
                "demand_id": role.demand_id,
                "terminal_provenance": role.terminal_provenance,
                "allowed_site_ids": list(role.allowed_site_ids),
                "declaration_ref": role.declaration_ref,
                "required_multiplicity": role.required_multiplicity,
                "selection_mode": LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
            }
            for role in family.demand_roles
        ],
        "branches": [
            {
                "branch_id": branch.branch_id,
                "source_endpoint": branch.source_endpoint,
                "target_endpoint": branch.target_endpoint,
                "ordered_use_ids": list(branch.ordered_use_ids),
                "ordered_landing_demand_ids": list(
                    branch.ordered_landing_demand_ids
                ),
                "ordered_steps": [
                    {"kind": step.kind, "ref_id": step.ref_id}
                    for step in branch.ordered_steps
                ],
            }
            for branch in family.branches
        ],
        "goal_branch_coverage": [
            {
                "declaration_ref": item.declaration_ref,
                "branch_id": item.branch_id,
                "target_provenance": item.target_provenance,
            }
            for item in family.goal_branch_coverage
        ],
        "endpoint_accesses": [
            {
                "provenance": access.provenance,
                "xy_nm": list(access.xy_nm),
                "layer_key": list(access.layer_key),
                "layer_n": access.layer_n,
                "owner_component_refs": list(access.owner_component_refs),
            }
            for access in family.endpoint_accesses
        ],
        "owned_component_use_refs": list(family.owned_component_use_refs),
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "candidate_id": profile.candidate_id,
                "use_profiles": [
                    {
                        "use_id": row.use_id,
                        "interval_nm": list(row.interval_nm),
                        "allowed_track_starts": list(row.allowed_track_starts),
                    }
                    for row in profile.use_profiles
                ],
            }
            for profile in family.profiles
        ],
    }
    if include_family_id:
        payload["family_id"] = family.family_id
    return payload


def _family_profile_candidate(
    family: FactorizedCompleteNetCandidateFamily,
    profile: FactorizedCandidateProfile,
    *,
    resources_by_key: Mapping[CorridorLayerKey, TrackResource],
    sites_by_id: Mapping[str, LandingSite],
    verify_candidate_id: bool,
    derivation_budget: Optional[list[int]] = None,
    site_replay_index: Optional[_FactorizedLandingSiteReplayIndex] = None,
) -> CompleteNetCandidate:
    profile_by_use = {row.use_id: row for row in profile.use_profiles}
    uses = tuple(CorridorUse(
        use_id=role.use_id,
        resource=role.resource,
        interval_nm=profile_by_use[role.use_id].interval_nm,
        track_span=role.track_span,
        allowed_track_starts=(
            profile_by_use[role.use_id].allowed_track_starts
        ),
        width_nm=role.width_nm,
    ) for role in family.use_roles)
    provisional_demands = tuple(LandingDemand(
        demand_id=role.demand_id,
        terminal_provenance=role.terminal_provenance,
        allowed_site_ids=role.allowed_site_ids,
        declaration_ref=role.declaration_ref,
        required_multiplicity=role.required_multiplicity,
        selection_mode=LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
    ) for role in family.demand_roles)
    provisional = CompleteNetCandidate(
        candidate_id="pending-content-address",
        net=family.net,
        topology_class=family.topology_class,
        branch_edges=tuple(
            (branch.source_endpoint, branch.target_endpoint)
            for branch in family.branches
        ),
        corridor_uses=uses,
        landing_demands=provisional_demands,
        required_goal_refs=family.required_goal_refs,
        width_nm=family.width_nm,
        via_multiplicity=family.via_multiplicity,
        source_topology_digest=family.source_topology_digest,
        program_kind=family.program_kind,
        branches=family.branches,
        landing_options=(),
        goal_branch_coverage=family.goal_branch_coverage,
        endpoint_accesses=family.endpoint_accesses,
        owned_component_use_refs=family.owned_component_use_refs,
    )
    replay_context = _derived_landing_site_replay_context(provisional)
    neighbour_use_ids_by_demand = replay_context[1]
    if site_replay_index is None:
        site_replay_index = _FactorizedLandingSiteReplayIndex.build(
            tuple(sites_by_id.values()),
            resources=tuple(resources_by_key.values()),
        )
        sites_by_id = site_replay_index.sites_by_id
        resources_by_key = site_replay_index.resources_by_key
    elif (
        sites_by_id is not site_replay_index.sites_by_id
        or resources_by_key is not site_replay_index.resources_by_key
    ):
        raise CandidatePoolError(
            "factorized landing-site replay index world binding mismatch"
        )
    exact_demands_list = []
    for demand in provisional_demands:
        neighbour_use_ids = neighbour_use_ids_by_demand[demand.demand_id]
        neighbour_uses = tuple(
            replay_context[0][use_id] for use_id in neighbour_use_ids
        )
        static_site_ids = site_replay_index.static_demand_site_ids(
            candidate_net=provisional.net,
            allowed_site_ids=demand.allowed_site_ids,
            neighbour_component_refs=replay_context[2][demand.demand_id],
            neighbour_uses=neighbour_uses,
            derivation_budget=derivation_budget,
        )
        if neighbour_uses:
            candidate_site_ids = site_replay_index.exact_geometry_site_ids(
                static_site_ids,
                neighbour_uses,
                derivation_budget=derivation_budget,
            )
        else:
            candidate_site_ids = sorted(static_site_ids)
        exact_demands_list.append(replace(
            demand, allowed_site_ids=tuple(candidate_site_ids)
        ))
    exact_demands = tuple(exact_demands_list)
    if any(not demand.allowed_site_ids for demand in exact_demands):
        raise CandidatePoolError("factorized profile has an empty exact site domain")
    realized = content_address_complete_net_candidate(replace(
        provisional, landing_demands=exact_demands
    ))
    if verify_candidate_id and realized.candidate_id != profile.candidate_id:
        raise CandidatePoolError(
            "factorized profile candidate_id does not replay from exact geometry"
        )
    return realized


def factorize_complete_net_candidate_family(
    candidates: Sequence[CompleteNetCandidate],
    *,
    resources: Sequence[TrackResource],
    landing_sites: Sequence[LandingSite],
) -> FactorizedCompleteNetCandidateFamily:
    raw_candidates = _bounded_materialize(
        candidates,
        limit=MAX_TOTAL_ASSIGNMENT_ARCS,
        label="factorized candidate",
        error_type=CandidatePoolLimitError,
    )
    resources = _bounded_materialize(
        resources,
        limit=MAX_TRACK_RESOURCES,
        label="factorization track resource",
        error_type=CandidatePoolLimitError,
    )
    landing_site_catalog = (
        landing_sites if isinstance(landing_sites, LandingSiteCatalog) else None
    )
    if landing_site_catalog is None:
        landing_sites = _bounded_materialize(
            landing_sites,
            limit=MAX_LANDING_SITES,
            label="factorization landing site",
            error_type=CandidatePoolLimitError,
        )
    ordered = tuple(sorted(
        raw_candidates, key=lambda candidate: candidate.candidate_id
    ))
    if not ordered:
        raise CandidatePoolError("cannot factor an empty candidate family")
    if any(
        candidate.landing_options
        or any(
            demand.selection_mode
            != LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
            for demand in candidate.landing_demands
        )
        for candidate in ordered
    ):
        raise CandidatePoolError(
            "v14 factorization requires alias-free derived landing domains"
        )
    for candidate in ordered:
        if content_address_complete_net_candidate(candidate) != candidate:
            raise CandidatePoolError(
                "v14 factorization requires content-addressed candidate variants"
            )

    def role_order(candidate: CompleteNetCandidate, kind: str) -> tuple[str, ...]:
        ids: list[str] = []
        branches = sorted(
            candidate.branches,
            key=lambda branch: (
                branch.source_endpoint,
                branch.target_endpoint,
                tuple(
                    (step.kind, step.ref_id if step.kind == "owned_component" else "")
                    for step in branch.ordered_steps
                ),
            ),
        )
        for branch in branches:
            for step in branch.ordered_steps:
                if step.kind == kind and step.ref_id not in ids:
                    ids.append(step.ref_id)
        return tuple(ids)

    exemplar = ordered[0]
    exemplar_use_ids = role_order(exemplar, "track_use")
    exemplar_demand_ids = role_order(exemplar, "landing_demand")

    def normalized(candidate: CompleteNetCandidate) -> dict[str, object]:
        use_ids = role_order(candidate, "track_use")
        demand_ids = role_order(candidate, "landing_demand")
        uses = {use.use_id: use for use in candidate.corridor_uses}
        demands = {
            demand.demand_id: demand for demand in candidate.landing_demands
        }
        use_pos = {use_id: ordinal for ordinal, use_id in enumerate(use_ids)}
        demand_pos = {
            demand_id: ordinal for ordinal, demand_id in enumerate(demand_ids)
        }
        branches = tuple(sorted(
            candidate.branches,
            key=lambda branch: (
                branch.source_endpoint,
                branch.target_endpoint,
                tuple(
                    (step.kind, step.ref_id if step.kind == "owned_component" else "")
                    for step in branch.ordered_steps
                ),
            ),
        ))
        branch_pos = {branch.branch_id: index for index, branch in enumerate(branches)}
        return {
            "schema": "clay.factorized-candidate-skeleton.v1",
            "net": candidate.net,
            "topology_class": candidate.topology_class,
            "program_kind": candidate.program_kind,
            "required_goal_refs": list(candidate.required_goal_refs),
            "width_nm": candidate.width_nm,
            "via_multiplicity": candidate.via_multiplicity,
            "source_topology_digest": candidate.source_topology_digest,
            "endpoint_accesses": [
                {
                    "provenance": access.provenance,
                    "xy_nm": list(access.xy_nm),
                    "layer_key": list(access.layer_key),
                    "layer_n": access.layer_n,
                    "owner_component_refs": list(access.owner_component_refs),
                }
                for access in candidate.endpoint_accesses
            ],
            "owned_component_use_refs": list(candidate.owned_component_use_refs),
            "use_roles": [
                {
                    "resource": _key_payload(uses[use_id].resource),
                    "track_span": uses[use_id].track_span,
                    "width_nm": uses[use_id].width_nm,
                }
                for use_id in use_ids
            ],
            "demand_roles": [
                {
                    "terminal_provenance": demands[
                        demand_id
                    ].terminal_provenance,
                    "declaration_ref": demands[demand_id].declaration_ref,
                    "required_multiplicity": demands[
                        demand_id
                    ].required_multiplicity,
                }
                for demand_id in demand_ids
            ],
            "branches": [
                {
                    "source_endpoint": branch.source_endpoint,
                    "target_endpoint": branch.target_endpoint,
                    "ordered_steps": [
                        {
                            "kind": step.kind,
                            "ref": (
                                use_pos[step.ref_id]
                                if step.kind == "track_use"
                                else demand_pos[step.ref_id]
                                if step.kind == "landing_demand"
                                else step.ref_id
                            ),
                        }
                        for step in branch.ordered_steps
                    ],
                }
                for branch in branches
            ],
            "goal_branch_coverage": [
                {
                    "declaration_ref": declaration_ref,
                    "branch_ordinal": branch_ordinal,
                    "target_provenance": target_provenance,
                }
                for declaration_ref, branch_ordinal, target_provenance in sorted(
                    (
                        item.declaration_ref,
                        branch_pos[item.branch_id],
                        item.target_provenance,
                    )
                    for item in candidate.goal_branch_coverage
                )
            ],
        }

    signature = normalized(exemplar)
    if any(normalized(candidate) != signature for candidate in ordered[1:]):
        raise CandidatePoolError("candidate variants do not share one v14 skeleton")
    skeleton_digest = hashlib.sha256(json.dumps(
        signature, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    stable_use_ids = tuple(
        f"factor-use:{hashlib.sha256(f'{skeleton_digest}:{ordinal}'.encode()).hexdigest()}"
        for ordinal in range(len(exemplar_use_ids))
    )
    stable_demand_ids = tuple(
        f"factor-demand:{hashlib.sha256(f'{skeleton_digest}:{ordinal}'.encode()).hexdigest()}"
        for ordinal in range(len(exemplar_demand_ids))
    )
    exemplar_uses = {use.use_id: use for use in exemplar.corridor_uses}
    exemplar_demands = {
        demand.demand_id: demand for demand in exemplar.landing_demands
    }
    use_roles = tuple(FactorizedCandidateUseRole(
        use_id=stable_use_ids[ordinal],
        resource=exemplar_uses[use_id].resource,
        track_span=exemplar_uses[use_id].track_span,
        width_nm=exemplar_uses[use_id].width_nm,
    ) for ordinal, use_id in enumerate(exemplar_use_ids))
    member_demand_rows = []
    for candidate in ordered:
        demand_ids = role_order(candidate, "landing_demand")
        demands = {demand.demand_id: demand for demand in candidate.landing_demands}
        member_demand_rows.append(tuple(demands[demand_id] for demand_id in demand_ids))
    demand_roles = tuple(FactorizedCandidateDemandRole(
        demand_id=stable_demand_ids[ordinal],
        terminal_provenance=exemplar_demands[demand_id].terminal_provenance,
        allowed_site_ids=tuple(sorted({
            site_id
            for row in member_demand_rows
            for site_id in row[ordinal].allowed_site_ids
        })),
        declaration_ref=exemplar_demands[demand_id].declaration_ref,
        required_multiplicity=exemplar_demands[demand_id].required_multiplicity,
    ) for ordinal, demand_id in enumerate(exemplar_demand_ids))

    exemplar_branches = tuple(sorted(
        exemplar.branches,
        key=lambda branch: (
            branch.source_endpoint,
            branch.target_endpoint,
            tuple(
                (step.kind, step.ref_id if step.kind == "owned_component" else "")
                for step in branch.ordered_steps
            ),
        ),
    ))
    use_pos = {use_id: ordinal for ordinal, use_id in enumerate(exemplar_use_ids)}
    demand_pos = {
        demand_id: ordinal for ordinal, demand_id in enumerate(exemplar_demand_ids)
    }
    branch_ids = tuple(
        f"factor-branch:{hashlib.sha256(f'{skeleton_digest}:{ordinal}'.encode()).hexdigest()}"
        for ordinal in range(len(exemplar_branches))
    )
    branches = tuple(CandidateBranch(
        branch_id=branch_ids[ordinal],
        source_endpoint=branch.source_endpoint,
        target_endpoint=branch.target_endpoint,
        ordered_use_ids=tuple(
            stable_use_ids[use_pos[use_id]] for use_id in branch.ordered_use_ids
        ),
        ordered_landing_demand_ids=tuple(
            stable_demand_ids[demand_pos[demand_id]]
            for demand_id in branch.ordered_landing_demand_ids
        ),
        ordered_steps=tuple(CandidateBranchStep(
            step.kind,
            stable_use_ids[use_pos[step.ref_id]]
            if step.kind == "track_use"
            else stable_demand_ids[demand_pos[step.ref_id]]
            if step.kind == "landing_demand"
            else step.ref_id,
        ) for step in branch.ordered_steps),
    ) for ordinal, branch in enumerate(exemplar_branches))
    branch_id_map = {
        branch.branch_id: branch_ids[index]
        for index, branch in enumerate(exemplar_branches)
    }
    coverage = tuple(GoalBranchCoverage(
        declaration_ref=item.declaration_ref,
        branch_id=branch_id_map[item.branch_id],
        target_provenance=item.target_provenance,
    ) for item in exemplar.goal_branch_coverage)

    resources_by_key = {resource.key: resource for resource in resources}
    sites_by_id = (
        {
            site.site_id: site
            for page in landing_site_catalog.pages
            for site in page.sites
        }
        if landing_site_catalog is not None
        else {site.site_id: site for site in landing_sites}
    )
    if len(resources_by_key) != len(resources):
        raise CandidatePoolError(
            "v14 factorization resources must have unique keys"
        )
    if len(sites_by_id) != (
        landing_site_catalog.site_count
        if landing_site_catalog is not None
        else len(landing_sites)
    ):
        raise CandidatePoolError(
            "v14 factorization landing sites must have unique ids"
        )
    missing_resources = tuple(sorted(
        {
            role.resource
            for role in use_roles
            if role.resource not in resources_by_key
        }
    ))
    missing_sites = tuple(sorted(
        {
            site_id
            for role in demand_roles
            for site_id in role.allowed_site_ids
            if site_id not in sites_by_id
        }
    ))
    if missing_resources:
        raise CandidatePoolError(
            f"v14 factorized roles name missing resources {missing_resources}"
        )
    if missing_sites:
        raise CandidatePoolError(
            f"v14 factorized roles name missing landing sites {missing_sites}"
        )
    profile_shells = []
    for candidate in ordered:
        use_ids = role_order(candidate, "track_use")
        uses = {use.use_id: use for use in candidate.corridor_uses}
        rows = tuple(FactorizedCandidateUseProfile(
            use_id=stable_use_ids[ordinal],
            interval_nm=uses[use_id].interval_nm,
            allowed_track_starts=uses[use_id].allowed_track_starts,
        ) for ordinal, use_id in enumerate(use_ids))
        profile_payload = {
            "schema": "clay.factorized-candidate-profile.v1",
            "use_profiles": [
                {
                    "use_id": row.use_id,
                    "interval_nm": list(row.interval_nm),
                    "allowed_track_starts": list(row.allowed_track_starts),
                }
                for row in sorted(rows, key=lambda row: row.use_id)
            ],
        }
        profile_id = hashlib.sha256(json.dumps(
            profile_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        profile_shells.append(FactorizedCandidateProfile(
            profile_id=profile_id,
            candidate_id="pending-profile-candidate",
            use_profiles=rows,
        ))
    placeholder_family = FactorizedCompleteNetCandidateFamily.__new__(
        FactorizedCompleteNetCandidateFamily
    )
    for name, value in {
        "family_id": "pending-family",
        "net": exemplar.net,
        "topology_class": exemplar.topology_class,
        "program_kind": exemplar.program_kind,
        "required_goal_refs": exemplar.required_goal_refs,
        "width_nm": exemplar.width_nm,
        "via_multiplicity": exemplar.via_multiplicity,
        "source_topology_digest": exemplar.source_topology_digest,
        "use_roles": use_roles,
        "demand_roles": demand_roles,
        "branches": branches,
        "goal_branch_coverage": coverage,
        "endpoint_accesses": exemplar.endpoint_accesses,
        "owned_component_use_refs": exemplar.owned_component_use_refs,
        "profiles": tuple(profile_shells),
    }.items():
        object.__setattr__(placeholder_family, name, value)
    profiles_list: list[FactorizedCandidateProfile] = []
    derivation_budget = [0]
    site_replay_index = _FactorizedLandingSiteReplayIndex.build(
        landing_site_catalog
        if landing_site_catalog is not None
        else landing_sites,
        resources=resources,
        site_limit=(
            MAX_TOTAL_ASSIGNMENT_ARCS
            if landing_site_catalog is not None
            else MAX_LANDING_SITES
        ),
    )
    replay_sites_by_id = site_replay_index.sites_by_id
    replay_resources_by_key = site_replay_index.resources_by_key
    for source_candidate, profile in zip(ordered, profile_shells):
        realized = _family_profile_candidate(
            placeholder_family,
            profile,
            resources_by_key=replay_resources_by_key,
            sites_by_id=replay_sites_by_id,
            verify_candidate_id=False,
            derivation_budget=derivation_budget,
            site_replay_index=site_replay_index,
        )
        source_demand_ids = role_order(source_candidate, "landing_demand")
        source_demands = {
            demand.demand_id: demand
            for demand in source_candidate.landing_demands
        }
        realized_demands = {
            demand.demand_id: demand for demand in realized.landing_demands
        }
        for ordinal, source_demand_id in enumerate(source_demand_ids):
            stable_demand_id = stable_demand_ids[ordinal]
            if (
                source_demands[source_demand_id].allowed_site_ids
                != realized_demands[stable_demand_id].allowed_site_ids
            ):
                raise CandidatePoolError(
                    "factorized profile exact site domain does not equal its "
                    "source candidate domain"
                )
        profiles_list.append(replace(
            profile, candidate_id=realized.candidate_id
        ))
    profiles = tuple(profiles_list)
    family_values = {
        "family_id": "pending-family",
        "net": exemplar.net,
        "topology_class": exemplar.topology_class,
        "program_kind": exemplar.program_kind,
        "required_goal_refs": tuple(sorted(exemplar.required_goal_refs)),
        "width_nm": exemplar.width_nm,
        "via_multiplicity": exemplar.via_multiplicity,
        "source_topology_digest": exemplar.source_topology_digest,
        "use_roles": tuple(sorted(use_roles, key=lambda role: role.use_id)),
        "demand_roles": tuple(sorted(
            demand_roles, key=lambda role: role.demand_id
        )),
        "branches": tuple(sorted(branches, key=lambda branch: branch.branch_id)),
        "goal_branch_coverage": tuple(sorted(coverage)),
        "endpoint_accesses": tuple(sorted(
            exemplar.endpoint_accesses, key=lambda access: access.provenance
        )),
        "owned_component_use_refs": tuple(sorted(
            exemplar.owned_component_use_refs
        )),
        "profiles": tuple(sorted(profiles, key=lambda profile: profile.profile_id)),
    }
    shell = FactorizedCompleteNetCandidateFamily.__new__(
        FactorizedCompleteNetCandidateFamily
    )
    for name, value in family_values.items():
        object.__setattr__(shell, name, value)
    family_id = hashlib.sha256(json.dumps(
        _factorized_candidate_family_payload(shell, include_family_id=False),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return FactorizedCompleteNetCandidateFamily(
        **{**family_values, "family_id": family_id}
    )


@dataclass(frozen=True)
class CompleteNetCandidateFamily:
    """An exact OR-family of fully materialized complete-net variants.

    Variants remain ordinary :class:`CompleteNetCandidate` records, so the
    allocator and every physical conflict checker still operate on their exact
    intervals.  The family is only the certified factorization boundary used
    for the per-net bounded-candidate contract; selecting any one member
    selects the family.
    """

    family_id: str
    net: str
    topology_class: str
    variant_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.net, str) or not self.net:
            raise CandidatePoolError("candidate family net must be non-empty")
        if not isinstance(self.topology_class, str) or not self.topology_class:
            raise CandidatePoolError(
                "candidate family topology_class must be non-empty"
            )
        variant_ids = _bounded_materialize(
            self.variant_candidate_ids,
            limit=MAX_TOTAL_CANDIDATES,
            label="candidate family variant",
            error_type=CandidatePoolLimitError,
        )
        if (
            not variant_ids
            or any(
                not isinstance(candidate_id, str) or not candidate_id
                for candidate_id in variant_ids
            )
            or len(variant_ids) != len(set(variant_ids))
        ):
            raise CandidatePoolError(
                "candidate family variants must be unique non-empty ids"
            )
        variant_ids = tuple(sorted(variant_ids))
        payload = {
            "net": self.net,
            "schema": "clay.complete-net-candidate-family.v1",
            "topology_class": self.topology_class,
            "variant_candidate_ids": list(variant_ids),
        }
        expected_id = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if self.family_id != expected_id:
            raise CandidatePoolError(
                "candidate family_id is not the content-addressed family hash"
            )
        object.__setattr__(self, "variant_candidate_ids", variant_ids)

    @classmethod
    def build(
        cls,
        *,
        net: str,
        topology_class: str,
        variant_candidate_ids: Sequence[str],
    ) -> "CompleteNetCandidateFamily":
        ordered = tuple(sorted(variant_candidate_ids))
        payload = {
            "net": net,
            "schema": "clay.complete-net-candidate-family.v1",
            "topology_class": topology_class,
            "variant_candidate_ids": list(ordered),
        }
        return cls(
            family_id=hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            net=net,
            topology_class=topology_class,
            variant_candidate_ids=ordered,
        )


@dataclass(frozen=True)
class CandidatePool:
    placement_provenance_digest: str
    topology_provenance_digest: str
    required_nets: tuple[str, ...]
    required_goal_refs: tuple[str, ...]
    resources: tuple[TrackResource, ...]
    landing_sites: tuple[LandingSite, ...]
    candidates: tuple[CompleteNetCandidate, ...]
    conflicts: tuple[CandidatePairConflict, ...]
    candidate_families: tuple[CompleteNetCandidateFamily, ...] = ()
    factorized_candidate_families: tuple[
        FactorizedCompleteNetCandidateFamily, ...
    ] = ()
    landing_site_catalog: Optional[LandingSiteCatalog] = None
    landing_expansion_resources: tuple[LandingExpansionResource, ...] = ()
    landing_site_conflicts: tuple[LandingSitePairConflict, ...] = ()
    required_route_goals: tuple[RequiredRouteGoal, ...] = ()
    pdk_provenance_digest: str = ""
    routing_world_provenance_digest: str = ""
    routing_resource_ledger_hash: str = ""
    owned_conductor_index_digest: str = ""
    lane_incidence_replay_digest: str = ""
    landing_site_domain_digest: str = ""
    landing_sites_complete: bool = True
    landing_site_conflict_domain: Optional[
        LandingSiteConflictDomain
    ] = None
    owned_conductor_components: tuple[
        OwnedConductorComponentCertificate, ...
    ] = ()
    _realized_candidates: tuple[CompleteNetCandidate, ...] = field(
        default=(), init=False, repr=False, compare=True, hash=False
    )

    def __post_init__(self) -> None:
        for field_name, limit, label in (
            ("required_nets", MAX_REQUIRED_NETS, "required net"),
            ("resources", MAX_TRACK_RESOURCES, "track resource"),
            ("landing_sites", MAX_LANDING_SITES, "landing site"),
            ("required_goal_refs", MAX_REQUIRED_GOAL_REFS, "required goal ref"),
            (
                "required_route_goals",
                MAX_REQUIRED_ROUTE_GOALS,
                "required route goal",
            ),
            ("candidates", MAX_TOTAL_CANDIDATES, "candidate"),
            (
                "candidate_families",
                MAX_TOTAL_CANDIDATES,
                "candidate family",
            ),
            (
                "factorized_candidate_families",
                MAX_TOTAL_CANDIDATES,
                "factorized candidate family",
            ),
            ("conflicts", MAX_PAIR_CONFLICTS, "pair conflict"),
            (
                "landing_expansion_resources",
                MAX_TRACK_RESOURCES,
                "landing expansion resource",
            ),
            (
                "landing_site_conflicts",
                MAX_LANDING_SITE_PAIR_CONFLICTS,
                "landing site-pair conflict",
            ),
            (
                "owned_conductor_components",
                MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES,
                "owned conductor component certificate",
            ),
        ):
            materialized = _bounded_materialize(
                getattr(self, field_name),
                limit=limit,
                label=label,
                error_type=CandidatePoolLimitError,
            )
            object.__setattr__(self, field_name, materialized)
        for name in ("placement_provenance_digest", "topology_provenance_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CandidatePoolError(f"{name} must be a non-empty string")
        for name in (
            "pdk_provenance_digest",
            "routing_world_provenance_digest",
            "routing_resource_ledger_hash",
            "owned_conductor_index_digest",
            "lane_incidence_replay_digest",
            "landing_site_domain_digest",
        ):
            if not isinstance(getattr(self, name), str):
                raise CandidatePoolError(f"{name} must be a string")
        if type(self.landing_sites_complete) is not bool:
            raise CandidatePoolError("landing_sites_complete must be an exact boolean")
        if not self.landing_sites_complete and not self.landing_site_domain_digest:
            raise CandidatePoolError(
                "an incomplete landing-site listing requires an exact domain digest"
            )

        if not all(
            isinstance(certificate, OwnedConductorComponentCertificate)
            for certificate in self.owned_conductor_components
        ):
            raise CandidatePoolError(
                "owned_conductor_components must contain component certificates"
            )
        component_member_budget = [0]
        rebuilt_component_certificates = []
        for certificate in self.owned_conductor_components:
            rebuilt_component_certificates.append(replace(
                certificate,
                component=_rebuild_owned_conductor_component(
                    certificate.component,
                    member_budget=component_member_budget,
                ),
            ))
        component_certificates = tuple(sorted(
            rebuilt_component_certificates,
            key=lambda certificate: certificate.component.component_ref,
        ))
        component_member_count = component_member_budget[0]
        if component_member_count > MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS:
            raise CandidatePoolLimitError(
                "owned conductor component member count "
                f"{component_member_count} exceeds "
                f"{MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS}"
            )
        component_refs = tuple(
            certificate.component.component_ref
            for certificate in component_certificates
        )
        if len(component_refs) != len(set(component_refs)):
            raise CandidatePoolError(
                "owned conductor component certificates must have unique refs"
            )
        has_world_binding = bool(self.routing_world_provenance_digest)
        has_index_binding = bool(self.owned_conductor_index_digest)
        if has_world_binding != has_index_binding:
            raise CandidatePoolError(
                "owned conductor world and component-index digests must be "
                "present together"
            )
        if component_certificates and not has_world_binding:
            raise CandidatePoolError(
                "owned conductor component certificates require world and "
                "component-index digests"
            )
        if has_world_binding:
            try:
                validate_owned_conductor_component_index_certificate(
                    world_digest=self.routing_world_provenance_digest,
                    components=tuple(
                        certificate.component
                        for certificate in component_certificates
                    ),
                    index_digest=self.owned_conductor_index_digest,
                    max_pair_evaluations=(
                        MAX_OWNED_CONDUCTOR_COMPONENT_PAIR_EVALUATIONS
                    ),
                )
            except OwnedConductorIndexLimitError as exc:
                raise CandidatePoolLimitError(
                    f"owned conductor component certificate table exceeds "
                    f"replay budget: {exc}"
                ) from exc
            except OwnedConductorIndexError as exc:
                raise CandidatePoolError(
                    "owned conductor component certificate table does not "
                    f"replay: {exc}"
                ) from exc

        required_nets = tuple(str(net) for net in self.required_nets)
        if any(not net for net in required_nets) or len(required_nets) != len(set(required_nets)):
            raise CandidatePoolError("required_nets must be unique non-empty strings")
        required_nets = tuple(sorted(required_nets))
        if len(required_nets) > MAX_REQUIRED_NETS:
            raise CandidatePoolLimitError(
                f"required net count {len(required_nets)} exceeds {MAX_REQUIRED_NETS}"
            )

        required_refs = tuple(str(ref) for ref in self.required_goal_refs)
        if any(not ref for ref in required_refs) or len(required_refs) != len(set(required_refs)):
            raise CandidatePoolError("required_goal_refs must be unique non-empty strings")
        required_refs = tuple(sorted(required_refs))

        rebuild_budget = _CandidatePoolRebuildBudget()
        resources = tuple(
            _rebuild_track_resource(resource, pool_budget=rebuild_budget)
            for resource in self.resources
        )
        if not all(isinstance(resource, TrackResource) for resource in resources):
            raise CandidatePoolError("resources must contain TrackResource values")
        resources = tuple(sorted(resources, key=lambda resource: resource.key))
        resource_keys = [resource.key for resource in resources]
        if len(resource_keys) != len(set(resource_keys)):
            raise CandidatePoolError("track resource keys must be unique")
        resources_by_key = {resource.key: resource for resource in resources}
        for resource in resources:
            if not _track_matches_pool_world_binding(
                resource, self.routing_world_provenance_digest
            ):
                binding = (
                    "world-bound" if self.routing_world_provenance_digest else "unbound"
                )
                raise CandidatePoolError(
                    f"{binding} track resource {resource.key!r} has invalid "
                    "geometry/provenance binding"
                )

        flat_sites = tuple(
            _rebuild_landing_site(site) for site in self.landing_sites
        )
        if not all(isinstance(site, LandingSite) for site in flat_sites):
            raise CandidatePoolError("landing_sites must contain LandingSite values")
        flat_sites = tuple(sorted(flat_sites, key=lambda site: site.site_id))
        catalog = self.landing_site_catalog
        if catalog is not None:
            if flat_sites:
                raise CandidatePoolError(
                    "a paged landing site catalog cannot alias flat landing_sites"
                )
            if not isinstance(catalog, LandingSiteCatalog):
                raise CandidatePoolError(
                    "landing_site_catalog must be an exact typed catalog"
                )
            catalog = LandingSiteCatalog(
                routing_world_provenance_digest=(
                    catalog.routing_world_provenance_digest
                ),
                landing_site_domain_digest=catalog.landing_site_domain_digest,
                pages=catalog.pages,
                site_count=catalog.site_count,
                collection_sha256=catalog.collection_sha256,
                catalog_sha256=catalog.catalog_sha256,
            )
            if (
                self.landing_sites_complete
                or catalog.routing_world_provenance_digest
                != self.routing_world_provenance_digest
                or catalog.landing_site_domain_digest
                != self.landing_site_domain_digest
            ):
                raise CandidatePoolError(
                    "paged landing site catalog has stale world/domain binding"
                )
            sites_by_id = {
                site.site_id: site
                for page in catalog.pages
                for site in page.sites
            }
        else:
            flat_site_ids = tuple(site.site_id for site in flat_sites)
            if len(flat_site_ids) != len(set(flat_site_ids)):
                raise CandidatePoolError("landing site ids must be unique")
            sites_by_id = {site.site_id: site for site in flat_sites}
        sites = sites_by_id.values()
        site_ids = list(sites_by_id)
        site_id_set = set(site_ids)
        expansion_resources = tuple(sorted(self.landing_expansion_resources))
        if any(
            not isinstance(item, LandingExpansionResource)
            for item in expansion_resources
        ):
            raise CandidatePoolError(
                "landing_expansion_resources must contain typed resources"
            )
        expansion_ids = tuple(item.expansion_id for item in expansion_resources)
        if len(expansion_ids) != len(set(expansion_ids)):
            raise CandidatePoolError("landing expansion ids must be unique")
        for expansion in expansion_resources:
            if (
                expansion.routing_world_provenance_digest
                != self.routing_world_provenance_digest
            ):
                raise CandidatePoolError(
                    "landing expansion belongs to a foreign routing world"
                )
            if set(expansion.legal_site_ids) - set(site_ids):
                raise CandidatePoolError(
                    "landing expansion names a missing legal site"
                )
            selected_sites = tuple(
                sites_by_id[site_id] for site_id in expansion.legal_site_ids
            )
            run_coordinates = tuple(
                site.xy_nm[0]
                if expansion.corridor_relation == "vertical_separation"
                else site.xy_nm[1]
                for site in selected_sites
            )
            cross_coordinates = tuple(sorted(
                site.xy_nm[1]
                if expansion.corridor_relation == "vertical_separation"
                else site.xy_nm[0]
                for site in selected_sites
            ))
            if any(
                coordinate < expansion.run_interval_nm[0]
                or coordinate > expansion.run_interval_nm[1]
                for coordinate in run_coordinates
            ):
                raise CandidatePoolError(
                    "landing expansion site lies outside its run interval"
                )
            if len(cross_coordinates) >= 2 and any(
                right - left != expansion.site_pitch_nm
                for left, right in zip(cross_coordinates, cross_coordinates[1:])
            ):
                raise CandidatePoolError(
                    "landing expansion sites do not replay one explicit pitch"
                )
            if any(site.capacity != 1 for site in selected_sites):
                raise CandidatePoolError(
                    "landing expansion currently proves only unit-capacity sites"
                )
        object.__setattr__(
            self, "landing_expansion_resources", expansion_resources
        )
        for site in sites:
            if not _landing_matches_pool_world_binding(
                site, self.routing_world_provenance_digest
            ):
                binding = (
                    "world-bound" if self.routing_world_provenance_digest else "unbound"
                )
                raise CandidatePoolError(
                    f"{binding} landing site {site.site_id!r} has invalid "
                    "geometry/provenance binding"
                )
        conflict_domain = self.landing_site_conflict_domain
        if catalog is not None and conflict_domain is None:
            raise CandidatePoolError(
                "paged landing site catalog requires a global exact conflict domain"
            )
        if conflict_domain is not None:
            if not isinstance(conflict_domain, LandingSiteConflictDomain):
                raise CandidatePoolError(
                    "landing_site_conflict_domain has invalid type"
                )
            conflict_domain = replace(conflict_domain)
            if (
                conflict_domain.routing_world_provenance_digest
                != self.routing_world_provenance_digest
                or conflict_domain.pdk_provenance_digest
                != self.pdk_provenance_digest
                or conflict_domain.landing_site_collection_digest
                != (
                    catalog.collection_sha256
                    if catalog is not None
                    else landing_site_collection_sha256(flat_sites)
                )
            ):
                raise CandidatePoolError(
                    "landing conflict domain has stale world/PDK/site binding"
                )
            metal_layer_ns = {rule.layer_n for rule in conflict_domain.metal_rules}
            via_layer_ns = {
                rule.lower_layer_n for rule in conflict_domain.via_rules
            }
            missing_metal_layers = tuple(sorted({
                layer_n
                for site in sites
                for layer_n in (site.lower_layer_n, site.upper_layer_n)
                if layer_n not in metal_layer_ns
            }))
            missing_via_layers = tuple(sorted({
                site.lower_layer_n
                for site in sites
                if site.lower_layer_n not in via_layer_ns
            }))
            if missing_metal_layers or missing_via_layers:
                raise CandidatePoolError(
                    "landing conflict domain does not cover all finite sites"
                )
        site_conflicts = tuple(
            replace(
                conflict,
                rule_refs=_bounded_materialize(
                    conflict.rule_refs,
                    limit=MAX_LANDING_SITE_PAIR_RULE_REFS,
                    label="landing site-pair rule ref",
                    error_type=CandidatePoolLimitError,
                ),
            )
            for conflict in self.landing_site_conflicts
        )
        if not all(
            isinstance(conflict, LandingSitePairConflict)
            for conflict in site_conflicts
        ):
            raise CandidatePoolError(
                "landing_site_conflicts must contain LandingSitePairConflict values"
            )
        site_conflicts = tuple(sorted(
            site_conflicts,
            key=lambda conflict: (
                conflict.left_site_id,
                conflict.right_site_id,
            ),
        ))
        if len(site_conflicts) > MAX_LANDING_SITE_PAIR_CONFLICTS:
            raise CandidatePoolLimitError(
                "landing site-pair conflict count exceeds "
                f"{MAX_LANDING_SITE_PAIR_CONFLICTS}"
            )
        conflict_pairs = tuple(
            (conflict.left_site_id, conflict.right_site_id)
            for conflict in site_conflicts
        )
        if len(conflict_pairs) != len(set(conflict_pairs)):
            raise CandidatePoolError(
                "landing site-pair conflict identities must be unique"
            )
        for conflict in site_conflicts:
            missing = tuple(
                site_id
                for site_id in (
                    conflict.left_site_id,
                    conflict.right_site_id,
                )
                if site_id not in site_id_set
            )
            if missing:
                raise CandidatePoolError(
                    f"landing site-pair conflict names missing sites {missing}"
                )
            if (
                not self.routing_world_provenance_digest
                or conflict.routing_world_provenance_digest
                != self.routing_world_provenance_digest
            ):
                raise CandidatePoolError(
                    "landing site-pair conflict has invalid routing-world binding"
                )

        candidates = tuple(
            _rebuild_closed_allocation_candidate(
                candidate, pool_budget=rebuild_budget
            )
            for candidate in self.candidates
        )
        if not all(isinstance(candidate, CompleteNetCandidate) for candidate in candidates):
            raise CandidatePoolError("candidates must contain CompleteNetCandidate values")
        candidates = tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise CandidatePoolError("candidate ids must be globally unique")
        for candidate in candidates:
            if candidate.source_topology_digest != self.topology_provenance_digest:
                raise CandidatePoolError(
                    f"candidate {candidate.candidate_id!r} source_topology_digest "
                    "does not match pool topology_provenance_digest"
                )
            graph_error = _branch_graph_admissibility_error(candidate)
            if graph_error is not None:
                raise CandidatePoolError(
                    f"candidate {candidate.candidate_id!r} {graph_error}"
                )
        if len(candidates) > MAX_TOTAL_CANDIDATES:
            raise CandidatePoolLimitError(
                f"candidate count {len(candidates)} exceeds {MAX_TOTAL_CANDIDATES}"
            )
        unexpected_nets = sorted({candidate.net for candidate in candidates} - set(required_nets))
        if unexpected_nets:
            raise CandidatePoolError(
                "candidate nets are not required: " + ", ".join(unexpected_nets)
            )
        candidates_by_net = {
            net: tuple(candidate for candidate in candidates if candidate.net == net)
            for net in required_nets
        }
        candidate_families = tuple(self.candidate_families)
        if not all(
            isinstance(family, CompleteNetCandidateFamily)
            for family in candidate_families
        ):
            raise CandidatePoolError(
                "candidate_families must contain CompleteNetCandidateFamily values"
            )
        candidate_families = tuple(sorted(
            candidate_families, key=lambda family: family.family_id
        ))
        family_ids = tuple(family.family_id for family in candidate_families)
        if len(family_ids) != len(set(family_ids)):
            raise CandidatePoolError("candidate family ids must be globally unique")
        families_by_net = {
            net: tuple(
                family for family in candidate_families if family.net == net
            )
            for net in required_nets
        }
        if candidate_families:
            unexpected_family_nets = sorted(
                {family.net for family in candidate_families} - set(required_nets)
            )
            if unexpected_family_nets:
                raise CandidatePoolError(
                    "candidate family nets are not required: "
                    + ", ".join(unexpected_family_nets)
                )
            candidates_by_id = {
                candidate.candidate_id: candidate for candidate in candidates
            }
            family_by_variant: dict[str, str] = {}
            for family in candidate_families:
                for candidate_id in family.variant_candidate_ids:
                    candidate = candidates_by_id.get(candidate_id)
                    if candidate is None:
                        raise CandidatePoolError(
                            f"candidate family {family.family_id!r} names missing "
                            f"variant {candidate_id!r}"
                        )
                    if (
                        candidate.net != family.net
                        or candidate.topology_class != family.topology_class
                    ):
                        raise CandidatePoolError(
                            f"candidate family {family.family_id!r} variant "
                            "does not match family net/topology"
                        )
                    previous = family_by_variant.setdefault(
                        candidate_id, family.family_id
                    )
                    if previous != family.family_id:
                        raise CandidatePoolError(
                            f"candidate variant {candidate_id!r} belongs to "
                            "multiple families"
                        )
            missing_family_variants = tuple(sorted(
                set(candidate_ids) - set(family_by_variant)
            ))
            if missing_family_variants:
                raise CandidatePoolError(
                    "candidate family projection omits variants "
                    f"{missing_family_variants}"
                )
        raw_factorized_families = tuple(self.factorized_candidate_families)
        if not all(
            isinstance(family, FactorizedCompleteNetCandidateFamily)
            for family in raw_factorized_families
        ):
            raise CandidatePoolError(
                "factorized_candidate_families contain invalid values"
            )
        factorized_families = tuple(sorted(
            (
                _rebuild_factorized_candidate_family(
                    family, pool_budget=rebuild_budget
                )
                for family in raw_factorized_families
            ),
            key=lambda family: family.family_id,
        ))
        factorized_family_ids = tuple(
            family.family_id for family in factorized_families
        )
        if len(factorized_family_ids) != len(set(factorized_family_ids)):
            raise CandidatePoolError(
                "factorized candidate family ids must be globally unique"
            )
        unexpected_factorized_nets = tuple(sorted(
            {family.net for family in factorized_families} - set(required_nets)
        ))
        if unexpected_factorized_nets:
            raise CandidatePoolError(
                "factorized candidate family nets are not required: "
                + ", ".join(unexpected_factorized_nets)
            )
        factorized_by_net = {
            net: tuple(
                family for family in factorized_families if family.net == net
            )
            for net in required_nets
        }
        factorized_support_arc_count = sum(
            1 + sum(len(row.allowed_track_starts) for row in profile.use_profiles)
            for family in factorized_families
            for profile in family.profiles
        )
        if factorized_support_arc_count > MAX_TOTAL_ASSIGNMENT_ARCS:
            raise CandidatePoolLimitError(
                "factorized profile/support assignment arc count "
                f"{factorized_support_arc_count} exceeds "
                f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
            )
        factorized_derivation_budget = [0]
        factorized_site_replay_index = (
            _FactorizedLandingSiteReplayIndex.build(
                catalog if catalog is not None else flat_sites,
                resources=resources,
                site_limit=(
                    MAX_TOTAL_ASSIGNMENT_ARCS
                    if catalog is not None
                    else MAX_LANDING_SITES
                ),
            )
        )
        replay_sites_by_id_for_profiles = (
            factorized_site_replay_index.sites_by_id
        )
        replay_resources_by_key_for_profiles = (
            factorized_site_replay_index.resources_by_key
        )
        realized_factorized_candidates = tuple(
            _family_profile_candidate(
                family,
                profile,
                resources_by_key=replay_resources_by_key_for_profiles,
                sites_by_id=replay_sites_by_id_for_profiles,
                verify_candidate_id=True,
                derivation_budget=factorized_derivation_budget,
                site_replay_index=factorized_site_replay_index,
            )
            for family in factorized_families
            for profile in family.profiles
        )
        factorized_support_arc_count += sum(
            len(demand.allowed_site_ids)
            for candidate in realized_factorized_candidates
            for demand in candidate.landing_demands
        )
        if factorized_support_arc_count > MAX_TOTAL_ASSIGNMENT_ARCS:
            raise CandidatePoolLimitError(
                "factorized profile/support assignment arc count "
                f"{factorized_support_arc_count} exceeds "
                f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
            )
        replay_candidates = tuple(sorted(
            (*candidates, *realized_factorized_candidates),
            key=lambda candidate: candidate.candidate_id,
        ))
        replay_candidate_ids = tuple(
            candidate.candidate_id for candidate in replay_candidates
        )
        if len(replay_candidate_ids) != len(set(replay_candidate_ids)):
            raise CandidatePoolError(
                "legacy and factorized profile candidate ids overlap"
            )
        if catalog is not None:
            referenced_site_ids = {
                site_id
                for candidate in replay_candidates
                for demand in candidate.landing_demands
                for site_id in demand.allowed_site_ids
            }
            referenced_site_ids.update(
                site_id
                for candidate in replay_candidates
                for option in candidate.landing_options
                for site_id in option.site_ids
            )
            if referenced_site_ids != site_id_set:
                raise CandidatePoolError(
                    "paged landing site catalog must equal the exact retained-"
                    "candidate site reference union"
                )
        for candidate in realized_factorized_candidates:
            if candidate.source_topology_digest != self.topology_provenance_digest:
                raise CandidatePoolError(
                    "factorized profile topology provenance does not match pool"
                )
            graph_error = _branch_graph_admissibility_error(candidate)
            if graph_error is not None:
                raise CandidatePoolError(
                    f"factorized profile {candidate.candidate_id!r} {graph_error}"
                )
        for net, net_candidates in candidates_by_net.items():
            factorized_net_families = factorized_by_net[net]
            if not net_candidates and not factorized_net_families:
                raise MissingNetCandidateError(f"required net {net!r} has no candidate")
            bounded_count = (
                len(families_by_net[net])
                if candidate_families
                else len(net_candidates)
            ) + len(factorized_net_families)
            if bounded_count > MAX_CANDIDATES_PER_NET:
                count_label = (
                    "candidate families"
                    if candidate_families or factorized_net_families
                    else "candidates"
                )
                raise CandidatePoolLimitError(
                    f"required net {net!r} has {bounded_count} {count_label}; "
                    f"maximum is {MAX_CANDIDATES_PER_NET}"
                )

        has_closed_allocation = any(
            candidate_has_closed_allocation_ir(candidate)
            for candidate in replay_candidates
        )
        if has_closed_allocation and not self.lane_incidence_replay_digest:
            raise CandidatePoolError(
                "closed candidate pool requires lane incidence replay digest"
            )
        if (
            self.lane_incidence_replay_digest
            and not _is_sha256(self.lane_incidence_replay_digest)
        ):
            raise CandidatePoolError(
                "lane incidence replay digest must be canonical SHA-256"
            )

        use_count = sum(
            len(candidate.corridor_uses) for candidate in candidates
        ) + sum(len(family.use_roles) for family in factorized_families)
        branch_edge_count = sum(
            len(candidate.branch_edges) for candidate in candidates
        ) + sum(len(family.branches) for family in factorized_families)
        demand_count = sum(
            len(candidate.landing_demands) for candidate in candidates
        ) + sum(len(family.demand_roles) for family in factorized_families)
        branch_count = sum(
            len(candidate.branches) for candidate in candidates
        ) + sum(len(family.branches) for family in factorized_families)
        branch_step_count = sum(
            len(branch.ordered_steps)
            for candidate in candidates
            for branch in candidate.branches
        ) + sum(
            len(branch.ordered_steps)
            for family in factorized_families
            for branch in family.branches
        )
        option_count = sum(
            len(candidate.landing_options) for candidate in candidates
        )
        option_templates, option_alias_count, _template_digest = (
            landing_option_template_index(candidates)
        )
        if option_alias_count != option_count:
            raise CandidatePoolError(
                "landing option template alias projection is incomplete"
            )
        if len(option_templates) > MAX_TOTAL_LANDING_OPTION_TEMPLATES:
            raise CandidatePoolLimitError(
                "landing option template count "
                f"{len(option_templates)} exceeds "
                f"{MAX_TOTAL_LANDING_OPTION_TEMPLATES}"
            )
        option_site_claim_count = sum(
            len(option.site_ids)
            for candidate in candidates
            for option in candidate.landing_options
        )
        option_contact_count = sum(
            len(option.contacts)
            for candidate in candidates
            for option in candidate.landing_options
        )
        goal_coverage_count = sum(
            len(candidate.goal_branch_coverage) for candidate in candidates
        ) + sum(
            len(family.goal_branch_coverage)
            for family in factorized_families
        )
        endpoint_access_count = sum(
            len(candidate.endpoint_accesses) for candidate in candidates
        ) + sum(
            len(family.endpoint_accesses) for family in factorized_families
        )
        owned_component_use_ref_count = sum(
            len(candidate.owned_component_use_refs) for candidate in candidates
        ) + sum(
            len(family.owned_component_use_refs)
            for family in factorized_families
        )
        arc_count = sum(
            len(use.allowed_track_starts)
            for candidate in candidates
            for use in candidate.corridor_uses
        ) + sum(
            len(demand.allowed_site_ids)
            for candidate in candidates
            for demand in candidate.landing_demands
        ) + option_site_claim_count + option_contact_count
        arc_count += factorized_support_arc_count
        if use_count > MAX_TOTAL_CORRIDOR_USES:
            raise CandidatePoolLimitError(
                f"corridor use count {use_count} exceeds {MAX_TOTAL_CORRIDOR_USES}"
            )
        if branch_edge_count > MAX_TOTAL_BRANCH_EDGES:
            raise CandidatePoolLimitError(
                f"branch edge count {branch_edge_count} exceeds "
                f"{MAX_TOTAL_BRANCH_EDGES}"
            )
        if demand_count > MAX_TOTAL_LANDING_DEMANDS:
            raise CandidatePoolLimitError(
                f"landing demand count {demand_count} exceeds {MAX_TOTAL_LANDING_DEMANDS}"
            )
        for count, limit, label in (
            (
                branch_count,
                MAX_TOTAL_CANDIDATE_BRANCHES,
                "branch count",
            ),
            (
                branch_step_count,
                MAX_TOTAL_BRANCH_STEPS,
                "branch step count",
            ),
            (
                option_count,
                MAX_TOTAL_LANDING_OPTIONS,
                "landing option count",
            ),
            (
                option_site_claim_count,
                MAX_TOTAL_LANDING_OPTION_SITE_CLAIMS,
                "option site claim count",
            ),
            (
                option_contact_count,
                MAX_TOTAL_LANDING_OPTION_CONTACTS,
                "option contact count",
            ),
            (
                goal_coverage_count,
                MAX_TOTAL_GOAL_BRANCH_COVERAGE,
                "goal branch coverage count",
            ),
            (
                endpoint_access_count,
                MAX_TOTAL_ENDPOINT_ACCESSES,
                "endpoint access count",
            ),
            (
                owned_component_use_ref_count,
                MAX_TOTAL_OWNED_COMPONENT_USE_REFS,
                "owned component use ref count",
            ),
        ):
            if count > limit:
                raise CandidatePoolLimitError(
                    f"{label} {count} exceeds {limit}"
                )
        if arc_count > MAX_TOTAL_ASSIGNMENT_ARCS:
            raise CandidatePoolLimitError(
                f"assignment arc count {arc_count} exceeds {MAX_TOTAL_ASSIGNMENT_ARCS}"
            )
        certificates_by_ref = {
            certificate.component.component_ref: certificate
            for certificate in component_certificates
        }
        for candidate in replay_candidates:
            power_root_error = _power_spine_root_proof_error(
                candidate, certificates_by_ref
            )
            if power_root_error is not None:
                raise CandidatePoolError(
                    f"candidate {candidate.candidate_id!r} {power_root_error}"
                )
            for component_ref in candidate.owned_component_use_refs:
                certificate = certificates_by_ref.get(component_ref)
                if certificate is None:
                    raise CandidatePoolError(
                        f"candidate {candidate.candidate_id!r} owned component "
                        f"step names unknown component {component_ref!r}"
                    )
                component = certificate.component
                if component.net != candidate.net:
                    raise CandidatePoolError(
                        f"candidate {candidate.candidate_id!r} owned component "
                        "step names a component on the wrong net"
                    )
                if not component.atom_facts:
                    raise CandidatePoolError(
                        f"candidate {candidate.candidate_id!r} pin-only component "
                        "cannot prove conductor geometry traversal"
                    )
        continuity_evaluations = 0
        for candidate in replay_candidates:
            for left_use, right_use in _consecutive_track_step_pairs(candidate):
                continuity_evaluations += len(left_use.allowed_track_starts) * len(
                    right_use.allowed_track_starts
                )
            uses_by_id = {
                use.use_id: use for use in candidate.corridor_uses
            }
            neighbour_use_ids = _landing_neighbour_use_ids(candidate)
            continuity_evaluations += sum(
                len(uses_by_id[use_id].allowed_track_starts)
                for option in candidate.landing_options
                for use_id in neighbour_use_ids[option.demand_id]
            )
            continuity_evaluations += sum(
                len(demand.allowed_site_ids)
                * len(uses_by_id[use_id].allowed_track_starts)
                for demand in candidate.landing_demands
                if (
                    demand.selection_mode
                    == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                )
                for use_id in neighbour_use_ids[demand.demand_id]
            )
            for component_ref, use in _component_track_step_pairs(candidate):
                component = certificates_by_ref[component_ref].component
                continuity_evaluations += (
                    len(use.allowed_track_starts) * len(component.atom_facts)
                )
            for left_ref, right_ref in _component_component_step_pairs(candidate):
                continuity_evaluations += (
                    len(certificates_by_ref[left_ref].component.atom_facts)
                    * len(certificates_by_ref[right_ref].component.atom_facts)
                )
            continuity_evaluations += sum(
                len(certificates_by_ref[component_ref].component.atom_facts)
                for _access, component_ref in _endpoint_component_requirements(
                    candidate
                )
            )
            if continuity_evaluations > MAX_DERIVED_CONTINUITY_EVALUATIONS:
                raise CandidatePoolLimitError(
                    "derived electrical continuity evaluations "
                    f"{continuity_evaluations} exceed "
                    f"{MAX_DERIVED_CONTINUITY_EVALUATIONS}"
                )
        site_component_evaluations = 0
        site_component_spatial_replay_budget = [0]
        component_replay_spatial_roots = (
            _build_owned_component_replay_spatial_roots(component_certificates)
        )
        for site in sites:
            for side, layer_n, pad_bbox, refs in (
                (
                    "lower",
                    site.lower_layer_n,
                    site.lower_pad_bbox_nm,
                    site.lower_owner_component_refs,
                ),
                (
                    "upper",
                    site.upper_layer_n,
                    site.upper_pad_bbox_nm,
                    site.upper_owner_component_refs,
                ),
            ):
                spatial_candidates = (
                    ()
                    if pad_bbox is None
                    else _owned_component_replay_spatial_candidates(
                        component_replay_spatial_roots,
                        layer_n=layer_n,
                        pad_bbox_nm=pad_bbox,
                        traversal_budget=site_component_spatial_replay_budget,
                    )
                )
                site_component_evaluations += len(spatial_candidates)
                if (
                    site_component_evaluations
                    > MAX_OWNED_CONDUCTOR_COMPONENT_PAIR_EVALUATIONS
                ):
                    raise CandidatePoolLimitError(
                        "landing site/component replay evaluations exceed "
                        f"{MAX_OWNED_CONDUCTOR_COMPONENT_PAIR_EVALUATIONS}"
                    )
                touching_certificates = tuple(
                    certificate
                    for certificate in spatial_candidates
                    if _owned_component_touches_pad(
                        certificate.component, pad_bbox
                    )
                )
                foreign_touching_refs = tuple(sorted(
                    certificate.component.component_ref
                    for certificate in touching_certificates
                    if certificate.component.net != site.owner_net
                ))
                if foreign_touching_refs:
                    raise CandidatePoolError(
                        f"landing site {site.site_id!r} {side} pad touches "
                        f"foreign component nets via {foreign_touching_refs}"
                    )
                expected_refs = tuple(sorted(
                    certificate.component.component_ref
                    for certificate in touching_certificates
                ))
                if tuple(refs) != expected_refs:
                    raise CandidatePoolError(
                        f"landing site {site.site_id!r} {side} owner refs must "
                        "equal the exact touching component set"
                    )
                touching_by_ref = {
                    certificate.component.component_ref: certificate
                    for certificate in touching_certificates
                }
                for component_ref in refs:
                    certificate = touching_by_ref.get(component_ref)
                    if certificate is None:
                        raise CandidatePoolError(
                            f"landing site {site.site_id!r} {side} owner ref "
                            "has no exact component certificate"
                        )
                    component = certificate.component
                    if (
                        certificate.layer_n != layer_n
                        or component.net != site.owner_net
                        or pad_bbox is None
                    ):
                        raise CandidatePoolError(
                            f"landing site {site.site_id!r} {side} component "
                            "certificate does not replay owner/layer/pad contact"
                        )
        for candidate in replay_candidates:
            uses_by_id = {
                use.use_id: use for use in candidate.corridor_uses
            }
            for access in candidate.endpoint_accesses:
                for component_ref in access.owner_component_refs:
                    certificate = certificates_by_ref.get(component_ref)
                    if certificate is None:
                        raise CandidatePoolError(
                            f"endpoint {access.provenance!r} owner ref has no "
                            "exact component certificate"
                        )
                    component = certificate.component
                    if (
                        certificate.layer_n != access.layer_n
                        or component.layer_key != access.layer_key
                        or component.net != candidate.net
                        or not _owned_component_contains_point(
                            component, access.xy_nm
                        )
                    ):
                        raise CandidatePoolError(
                            f"endpoint {access.provenance!r} component certificate "
                            "does not replay exact owner/layer/point access"
                        )
            for use in candidate.corridor_uses:
                resource = resources_by_key.get(use.resource)
                if resource is None:
                    raise CandidatePoolError(
                        f"candidate {candidate.candidate_id!r} use {use.use_id!r} "
                        f"names missing resource {use.resource}"
                    )
                if resource.owner_net not in (None, candidate.net):
                    raise CandidatePoolError(
                        f"candidate {candidate.candidate_id!r} use {use.use_id!r} "
                        f"uses track owned by foreign net {resource.owner_net!r}"
                    )
                through_interval = resource.through_interval_nm
                if through_interval is not None and not (
                    through_interval[0] <= use.interval_nm[0]
                    and use.interval_nm[1] <= through_interval[1]
                ):
                    raise CandidatePoolError(
                        f"candidate {candidate.candidate_id!r} use {use.use_id!r} "
                        "exceeds geometry-proved through interval"
                    )
                invalid = tuple(
                    start
                    for start in use.allowed_track_starts
                    if start + use.track_span > len(resource.track_centers_nm)
                )
                if invalid:
                    raise CandidatePoolError(
                        f"candidate {candidate.candidate_id!r} use {use.use_id!r} "
                        f"has out-of-range track starts {invalid}"
                    )
                if resource.corridor_bbox_nm is not None:
                    width_invalid = tuple(
                        start
                        for start in use.allowed_track_starts
                        if (
                            (bbox := _track_choice_closed_bbox(
                                use,
                                resource,
                                start,
                                _corridor_use_width_nm(use, candidate.width_nm),
                            ))
                            is None
                            or not _bbox_contains(resource.corridor_bbox_nm, bbox)
                        )
                    )
                    if width_invalid:
                        raise CandidatePoolError(
                            f"candidate {candidate.candidate_id!r} width does not "
                            f"fit corridor for use {use.use_id!r} track starts "
                            f"{width_invalid}"
                        )
            if candidate_has_closed_allocation_ir(candidate):
                for left_use, right_use in _consecutive_track_step_pairs(candidate):
                    left_resource = resources_by_key[left_use.resource]
                    right_resource = resources_by_key[right_use.resource]
                    if not any(
                        _track_choices_physically_touch(
                            left_use,
                            left_resource,
                            left_start,
                            right_use,
                            right_resource,
                            right_start,
                            _corridor_use_width_nm(
                                left_use, candidate.width_nm
                            ),
                            _corridor_use_width_nm(
                                right_use, candidate.width_nm
                            ),
                        )
                        for left_start in left_use.allowed_track_starts
                        for right_start in right_use.allowed_track_starts
                    ):
                        raise CandidatePoolError(
                            f"candidate {candidate.candidate_id!r} consecutive "
                            f"track steps {left_use.use_id!r}, "
                            f"{right_use.use_id!r} have no electrically "
                            "continuous track choices"
                        )
                for left_ref, right_ref in _component_component_step_pairs(
                    candidate
                ):
                    left_certificate = certificates_by_ref[left_ref]
                    right_certificate = certificates_by_ref[right_ref]
                    left_component = left_certificate.component
                    right_component = right_certificate.component
                    if (
                        left_ref != right_ref
                        and (
                            left_certificate.layer_n
                            != right_certificate.layer_n
                            or not _owned_component_conductors_touch(
                                left_component, right_component
                            )
                        )
                    ):
                        raise CandidatePoolError(
                            f"candidate {candidate.candidate_id!r} consecutive "
                            "owned components are not electrically connected by "
                            "exact conductor geometry"
                        )
                for component_ref, use in _component_track_step_pairs(candidate):
                    certificate = certificates_by_ref[component_ref]
                    component = certificate.component
                    resource = resources_by_key[use.resource]
                    if (
                        certificate.layer_n != resource.key.layer_n
                        or component.layer_key != resource.gds_layer_key
                        or not any(
                            (track_bbox := _track_choice_closed_bbox(
                                use,
                                resource,
                                track_start,
                                _corridor_use_width_nm(
                                    use, candidate.width_nm
                                ),
                            ))
                            is not None
                            and _owned_component_conductor_touches_track_bbox(
                                component, track_bbox
                            )
                            for track_start in use.allowed_track_starts
                        )
                    ):
                        raise CandidatePoolError(
                            f"candidate {candidate.candidate_id!r} owned component "
                            f"{component_ref!r} has no exact conductor geometry "
                            f"contact with adjacent track use {use.use_id!r}"
                        )
                for access, use_id in _endpoint_track_requirements(candidate):
                    use = uses_by_id[use_id]
                    resource = resources_by_key[use.resource]
                    if access.layer_key != resource.gds_layer_key:
                        raise CandidatePoolError(
                            f"candidate {candidate.candidate_id!r} endpoint "
                            f"{access.provenance!r} GDS layer does not match "
                            "its boundary track resource"
                        )
                    if not any(
                        _track_choice_contains_endpoint(
                            access, use, resource, track_start
                        )
                        for track_start in use.allowed_track_starts
                    ):
                        raise CandidatePoolError(
                            f"candidate {candidate.candidate_id!r} endpoint "
                            f"{access.provenance!r} is not on its boundary "
                            "track geometry"
                        )
                for access, component_ref in _endpoint_component_requirements(
                    candidate
                ):
                    certificate = certificates_by_ref[component_ref]
                    component = certificate.component
                    if component_ref not in access.owner_component_refs:
                        raise CandidatePoolError(
                            f"candidate {candidate.candidate_id!r} "
                            f"owned-component endpoint {access.provenance!r} "
                            "must name the traversed component in owner refs"
                        )
                    if (
                        certificate.layer_n != access.layer_n
                        or component.layer_key != access.layer_key
                        or component.net != candidate.net
                        or not _owned_component_conductor_contains_point(
                            component, access.xy_nm
                        )
                    ):
                        raise CandidatePoolError(
                            f"candidate {candidate.candidate_id!r} endpoint "
                            f"{access.provenance!r} does not touch exact owned "
                            "component conductor geometry"
                        )
                accesses_by_provenance = {
                    access.provenance: access
                    for access in candidate.endpoint_accesses
                }
                options_by_demand: dict[str, list[LandingOption]] = {
                    demand.demand_id: []
                    for demand in candidate.landing_demands
                }
                for option in candidate.landing_options:
                    options_by_demand.setdefault(option.demand_id, []).append(option)
                for branch in candidate.branches:
                    for endpoint, step in (
                        (branch.source_endpoint, branch.ordered_steps[0]),
                        (branch.target_endpoint, branch.ordered_steps[-1]),
                    ):
                        if step.kind != "landing_demand":
                            continue
                        access = accesses_by_provenance[endpoint]
                        for option in options_by_demand[step.ref_id]:
                            option_contacts_by_site = {
                                site_id: tuple(
                                    contact
                                    for contact in option.contacts
                                    if contact.site_id == site_id
                                )
                                for site_id in option.site_ids
                            }
                            boundary_reached = False
                            for site_id in option.site_ids:
                                site = sites_by_id[site_id]
                                if site.xy_nm != access.xy_nm:
                                    continue
                                if access.layer_n == site.lower_layer_n:
                                    side_refs = site.lower_owner_component_refs
                                elif access.layer_n == site.upper_layer_n:
                                    side_refs = site.upper_owner_component_refs
                                else:
                                    continue
                                evidence_layer_keys = {
                                    certificates_by_ref[component_ref]
                                    .component.layer_key
                                    for component_ref in side_refs
                                    if component_ref in certificates_by_ref
                                }
                                evidence_layer_keys.update(
                                    resources_by_key[uses_by_id[contact.use_id].resource]
                                    .gds_layer_key
                                    for contact in option_contacts_by_site[site_id]
                                    if resources_by_key[
                                        uses_by_id[contact.use_id].resource
                                    ].key.layer_n
                                    == access.layer_n
                                )
                                evidence_layer_keys.discard(None)
                                if access.layer_key not in evidence_layer_keys:
                                    continue
                                if access.owner_component_refs and not (
                                    set(access.owner_component_refs) & set(side_refs)
                                ):
                                    continue
                                boundary_reached = True
                                break
                            if not boundary_reached:
                                raise CandidatePoolError(
                                    f"candidate {candidate.candidate_id!r} boundary "
                                    f"landing endpoint {access.provenance!r} lacks "
                                    f"exact GDS-side evidence in option "
                                    f"{option.option_id!r}"
                                )
            for demand in candidate.landing_demands:
                missing_sites = sorted(set(demand.allowed_site_ids) - site_id_set)
                if missing_sites:
                    raise CandidatePoolError(
                        f"candidate {candidate.candidate_id!r} demand {demand.demand_id!r} "
                        f"names missing landing sites {missing_sites}"
                    )
                foreign_owned_sites = tuple(sorted(
                    site_id
                    for site_id in demand.allowed_site_ids
                    if sites_by_id[site_id].owner_net not in (None, candidate.net)
                ))
                if foreign_owned_sites:
                    raise CandidatePoolError(
                        f"candidate {candidate.candidate_id!r} demand "
                        f"{demand.demand_id!r} allows landing sites with foreign "
                        f"owner {foreign_owned_sites}"
                    )
                if (
                    demand.selection_mode
                    == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                ):
                    for site_id in demand.allowed_site_ids:
                        error = _derived_landing_site_domain_error(
                            candidate,
                            demand.demand_id,
                            sites_by_id[site_id],
                            resources_by_key=resources_by_key,
                        )
                        if error is not None:
                            raise CandidatePoolError(
                                f"candidate {candidate.candidate_id!r} derived "
                                f"demand {demand.demand_id!r} site "
                                f"{site_id!r} is not exact: {error}"
                            )
            if candidate_has_closed_allocation_ir(candidate):
                neighbour_use_ids_by_demand = _landing_neighbour_use_ids(candidate)
                neighbour_component_refs_by_demand = (
                    _landing_neighbour_component_refs(candidate)
                )
                for option in candidate.landing_options:
                    option_sites = tuple(
                        sites_by_id.get(site_id) for site_id in option.site_ids
                    )
                    if any(site is None for site in option_sites):
                        missing = tuple(sorted(
                            site_id
                            for site_id in option.site_ids
                            if site_id not in sites_by_id
                        ))
                        raise CandidatePoolError(
                            f"landing option {option.option_id!r} names missing "
                            f"sites {missing}"
                        )
                    contact_layers_by_site: dict[str, set[int]] = {
                        site_id: set() for site_id in option.site_ids
                    }
                    for contact in option.contacts:
                        use = uses_by_id.get(contact.use_id)
                        if use is None:
                            raise CandidatePoolError(
                                f"landing option {option.option_id!r} contact names "
                                f"missing use {contact.use_id!r}"
                            )
                        if contact.track_start not in use.allowed_track_starts:
                            raise CandidatePoolError(
                                f"landing option {option.option_id!r} contact names "
                                "a disallowed track start"
                            )
                        resource = resources_by_key[use.resource]
                        site = sites_by_id[contact.site_id]
                        layer_n = resource.key.layer_n
                        if layer_n not in (
                            site.lower_layer_n,
                            site.upper_layer_n,
                        ):
                            raise CandidatePoolError(
                                f"landing option {option.option_id!r} contact "
                                "resource is not on an adjacent landing layer"
                            )
                        selected_centers = resource.track_centers_nm[
                            contact.track_start:
                            contact.track_start + use.track_span
                        ]
                        cross_coordinate = (
                            site.xy_nm[1]
                            if resource.preferred_axis == "H"
                            else site.xy_nm[0]
                        )
                        run_coordinate = (
                            site.xy_nm[0]
                            if resource.preferred_axis == "H"
                            else site.xy_nm[1]
                        )
                        if (
                            cross_coordinate not in selected_centers
                            or not (
                                use.interval_nm[0]
                                <= run_coordinate
                                < use.interval_nm[1]
                            )
                        ):
                            raise CandidatePoolError(
                                f"landing option {option.option_id!r} contact "
                                "does not cross landing site on selected span: "
                                f"site={site.xy_nm}, centers={tuple(selected_centers)}, "
                                f"interval={use.interval_nm}, "
                                f"resource={resource.key!r}"
                            )
                        contact_layers_by_site[contact.site_id].add(layer_n)
                    for site in option_sites:
                        assert site is not None
                        covered_layers = contact_layers_by_site[site.site_id]
                        if (
                            site.owner_net == candidate.net
                            and site.lower_owner_component_refs
                        ):
                            covered_layers.add(site.lower_layer_n)
                        if (
                            site.owner_net == candidate.net
                            and site.upper_owner_component_refs
                        ):
                            covered_layers.add(site.upper_layer_n)
                        for side, layer_n in (
                            ("lower", site.lower_layer_n),
                            ("upper", site.upper_layer_n),
                        ):
                            if layer_n not in covered_layers:
                                raise CandidatePoolError(
                                    f"landing option {option.option_id!r} site "
                                    f"{site.site_id!r} {side} pad side is uncovered"
                                )
                    for use_id in neighbour_use_ids_by_demand[option.demand_id]:
                        use = uses_by_id[use_id]
                        resource = resources_by_key[use.resource]
                        if not any(
                            _track_choice_touches_landing_option(
                                candidate=candidate,
                                option=option,
                                use=use,
                                resource=resource,
                                track_start=track_start,
                                sites_by_id=sites_by_id,
                                certificates_by_ref=certificates_by_ref,
                            )
                            for track_start in use.allowed_track_starts
                        ):
                            raise CandidatePoolError(
                                f"landing option {option.option_id!r} has no "
                                f"electrically continuous choice for adjacent "
                                f"use {use_id!r}"
                            )
                    for component_ref in neighbour_component_refs_by_demand[
                        option.demand_id
                    ]:
                        if not _owned_component_touches_landing_option(
                            component_ref,
                            option,
                            sites_by_id=sites_by_id,
                        ):
                            raise CandidatePoolError(
                                f"landing option {option.option_id!r} has no "
                                "exact owned-component pad contact for adjacent "
                                f"component {component_ref!r}"
                            )

        reachable_nets_by_site: dict[str, set[str]] = {
            site.site_id: set() for site in sites
        }
        for candidate in replay_candidates:
            for demand in candidate.landing_demands:
                for site_id in demand.allowed_site_ids:
                    reachable_nets_by_site[site_id].add(candidate.net)
        site_conflict_clause_count = 0
        for conflict in site_conflicts:
            if conflict.conflict_scope == "all_nets":
                site_conflict_clause_count += 1
                continue
            site_conflict_clause_count += sum(
                left_net != right_net
                for left_net in reachable_nets_by_site[conflict.left_site_id]
                for right_net in reachable_nets_by_site[conflict.right_site_id]
            )
            if conflict.conflict_scope == "same_net_bridge":
                # Each same-candidate assignment pair receives one conditional
                # bridge clause.  Count the exact finite option/legacy domains
                # before model construction so hostile inputs cannot multiply
                # clauses beyond the public pool bound.
                for candidate in replay_candidates:
                    if candidate_has_closed_allocation_ir(candidate):
                        derived_demands = tuple(
                            demand
                            for demand in candidate.landing_demands
                            if demand.selection_mode
                            == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                        )
                        left_count = (
                            sum(
                                conflict.left_site_id in option.site_ids
                                for option in candidate.landing_options
                            )
                            + sum(
                                conflict.left_site_id
                                in demand.allowed_site_ids
                                for demand in derived_demands
                            )
                        )
                        right_count = (
                            sum(
                                conflict.right_site_id in option.site_ids
                                for option in candidate.landing_options
                            )
                            + sum(
                                conflict.right_site_id
                                in demand.allowed_site_ids
                                for demand in derived_demands
                            )
                        )
                    else:
                        left_count = sum(
                            conflict.left_site_id in demand.allowed_site_ids
                            for demand in candidate.landing_demands
                        )
                        right_count = sum(
                            conflict.right_site_id in demand.allowed_site_ids
                            for demand in candidate.landing_demands
                        )
                    site_conflict_clause_count += left_count * right_count
        if site_conflict_clause_count > MAX_LANDING_CONFLICT_NET_CLAUSES:
            raise CandidatePoolLimitError(
                "derived landing site-pair net clauses exceed "
                f"{MAX_LANDING_CONFLICT_NET_CLAUSES}"
            )

        _validate_derived_track_conflict_bounds(
            replay_candidates, resources_by_key
        )

        conflicts = tuple(
            replace(
                conflict,
                declaration_refs=_bounded_materialize(
                    conflict.declaration_refs,
                    limit=MAX_CANDIDATE_PAIR_DECLARATION_REFS,
                    label="pair conflict declaration ref",
                    error_type=CandidatePoolLimitError,
                ),
            )
            for conflict in self.conflicts
        )
        if not all(isinstance(conflict, CandidatePairConflict) for conflict in conflicts):
            raise CandidatePoolError("conflicts must contain CandidatePairConflict values")
        conflicts = tuple(sorted(
            conflicts,
            key=lambda conflict: (
                conflict.left_candidate_id,
                conflict.right_candidate_id,
                conflict.declaration_refs,
            ),
        ))
        if len(conflicts) > MAX_PAIR_CONFLICTS:
            raise CandidatePoolLimitError(
                f"pair conflict count {len(conflicts)} exceeds {MAX_PAIR_CONFLICTS}"
            )
        conflict_keys = [
            (conflict.left_candidate_id, conflict.right_candidate_id)
            for conflict in conflicts
        ]
        if len(conflict_keys) != len(set(conflict_keys)):
            raise CandidatePoolError("candidate pair conflicts must be unique")
        candidate_id_set = set(replay_candidate_ids)
        for conflict in conflicts:
            missing = sorted(
                {conflict.left_candidate_id, conflict.right_candidate_id} - candidate_id_set
            )
            if missing:
                raise CandidatePoolError(
                    f"candidate pair conflict names missing candidates {missing}"
                )

        goals = tuple(
            _rebuild_required_route_goal(goal)
            for goal in self.required_route_goals
        )
        if not all(isinstance(goal, RequiredRouteGoal) for goal in goals):
            raise CandidatePoolError("required_route_goals must contain RequiredRouteGoal values")
        goals = tuple(sorted(goals, key=_required_goal_sort_key))
        goal_refs = [goal.declaration_ref for goal in goals]
        if len(goal_refs) != len(set(goal_refs)):
            raise MissingRequiredGoalError("required route goal declaration refs must be unique")
        for goal in goals:
            if goal.net not in set(required_nets):
                raise MissingRequiredGoalError(
                    f"required goal {goal.declaration_ref!r} names non-required net {goal.net!r}"
                )
            if goal.target is None:
                raise MissingRequiredGoalError(
                    f"required goal {goal.declaration_ref!r} has no physical target"
                )
        unlinked_required_refs = sorted(set(required_refs) - set(goal_refs))
        if unlinked_required_refs:
            raise MissingRequiredGoalError(
                f"required goal refs {unlinked_required_refs} have no "
                "RequiredRouteGoal linkage"
            )
        linked_refs = tuple(sorted(set(required_refs) | set(goal_refs)))
        for candidate in replay_candidates:
            endpoints = {
                endpoint for edge in candidate.branch_edges for endpoint in edge
            }
            accesses_by_provenance = {
                access.provenance: access
                for access in candidate.endpoint_accesses
            }
            coverage_by_ref = {
                item.declaration_ref: item
                for item in candidate.goal_branch_coverage
            }
            for goal in (goal for goal in goals if goal.net == candidate.net):
                if goal.declaration_ref not in set(candidate.required_goal_refs):
                    raise MissingRequiredGoalError(
                        f"candidate {candidate.candidate_id!r} omits required goal "
                        f"ref {goal.declaration_ref!r}"
                    )
                if goal.target.provenance not in endpoints:
                    raise MissingRequiredGoalError(
                        f"candidate {candidate.candidate_id!r} required target "
                        f"{goal.target.provenance!r} is absent from every branch endpoint"
                    )
                if candidate_has_closed_allocation_ir(candidate):
                    access = accesses_by_provenance.get(goal.target.provenance)
                    coverage_item = coverage_by_ref.get(goal.declaration_ref)
                    if (
                        access is None
                        or access.xy_nm != goal.target.xy_nm
                        or access.layer_key != goal.target.layer
                        or coverage_item is None
                        or coverage_item.target_provenance != access.provenance
                    ):
                        raise MissingRequiredGoalError(
                            f"candidate {candidate.candidate_id!r} goal "
                            f"{goal.declaration_ref!r} does not exactly match "
                            "endpoint access and dedicated branch coverage"
                        )
        covered_refs = {
            ref
            for candidate in replay_candidates
            for ref in candidate.required_goal_refs
        }
        missing_global_refs = sorted(set(required_refs) - covered_refs)
        if missing_global_refs:
            raise MissingRequiredGoalError(
                f"required goal refs have no candidate linkage: {missing_global_refs}"
            )

        if self.routing_world_provenance_digest:
            ledger_hash_args = (
                self.routing_world_provenance_digest,
                resources,
            )
            expected_ledger_hash = (
                explicit_routing_resource_ledger_sha256(
                    *ledger_hash_args,
                    sites,
                    site_conflicts,
                    self.owned_conductor_index_digest,
                    self.landing_site_domain_digest,
                    self.landing_sites_complete,
                    conflict_domain,
                )
                if catalog is None
                else paged_routing_resource_ledger_sha256(
                    *ledger_hash_args,
                    catalog,
                    site_conflicts,
                    self.owned_conductor_index_digest,
                    self.landing_site_domain_digest,
                    self.landing_sites_complete,
                    conflict_domain,
                )
            )
            if self.routing_resource_ledger_hash != expected_ledger_hash:
                raise CandidatePoolError(
                    "routing resource ledger hash does not match pool contents"
                )
        elif self.routing_resource_ledger_hash:
            raise CandidatePoolError(
                "unbound synthetic pool cannot claim a resource ledger hash"
            )

        object.__setattr__(self, "required_nets", required_nets)
        object.__setattr__(self, "required_goal_refs", linked_refs)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "landing_sites", flat_sites)
        object.__setattr__(self, "landing_site_catalog", catalog)
        object.__setattr__(self, "landing_site_conflicts", site_conflicts)
        object.__setattr__(
            self, "landing_site_conflict_domain", conflict_domain
        )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "candidate_families", candidate_families)
        object.__setattr__(
            self, "factorized_candidate_families", factorized_families
        )
        object.__setattr__(self, "_realized_candidates", replay_candidates)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "required_route_goals", goals)
        object.__setattr__(
            self, "owned_conductor_components", component_certificates
        )

    def iter_landing_sites(self):
        if self.landing_site_catalog is None:
            yield from self.landing_sites
            return
        yield from self.landing_site_catalog.iter_sites()

    def landing_sites_by_id(self) -> dict[str, LandingSite]:
        return {site.site_id: site for site in self.iter_landing_sites()}

    def candidates_for(self, net: str) -> tuple[CompleteNetCandidate, ...]:
        return tuple(
            candidate for candidate in self.all_candidates if candidate.net == net
        )

    @property
    def all_candidates(self) -> tuple[CompleteNetCandidate, ...]:
        if "_realized_candidates" not in vars(self):
            raise CandidatePoolError(
                "candidate pool runtime replay cache is missing"
            )
        return self._realized_candidates

    def canonical_json(self) -> str:
        return json.dumps(
            _candidate_pool_payload(self), sort_keys=True, separators=(",", ":")
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _replay_candidate_pool_runtime(pool: CandidatePool) -> CandidatePool:
    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be CandidatePool")
    runtime_candidates = pool.all_candidates
    rebuilt = replace(pool)
    if (
        runtime_candidates != rebuilt.all_candidates
        or pool != rebuilt
    ):
        raise CandidatePoolError(
            "candidate pool runtime replay cache diverges from canonical IR"
        )
    return rebuilt


@dataclass(frozen=True)
class SelectedTrack:
    use_id: str
    resource: CorridorLayerKey
    track_start: int


@dataclass(frozen=True)
class SelectedLanding:
    demand_id: str
    site_id: str


@dataclass(frozen=True, order=True)
class SelectedLandingOption:
    demand_id: str
    option_id: str

    def __post_init__(self) -> None:
        for name in ("demand_id", "option_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CandidatePoolError(
                    f"selected landing option {name} must be non-empty"
                )


@dataclass(frozen=True)
class SelectedNetCandidate:
    net: str
    candidate_id: str
    selected_tracks: tuple[SelectedTrack, ...]
    selected_landings: tuple[SelectedLanding, ...]
    selected_landing_options: tuple[SelectedLandingOption, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_tracks", tuple(sorted(
            self.selected_tracks,
            key=lambda track: (track.resource, track.use_id, track.track_start),
        )))
        object.__setattr__(self, "selected_landings", tuple(sorted(
            self.selected_landings,
            key=lambda landing: (landing.demand_id, landing.site_id),
        )))
        object.__setattr__(self, "selected_landing_options", tuple(sorted(
            self.selected_landing_options,
            key=lambda option: (option.demand_id, option.option_id),
        )))


@dataclass(frozen=True)
class AllocatedTrack:
    use_id: str
    resource: TrackResource
    interval_nm: tuple[int, int]
    track_start: int
    track_span: int
    selected_track_centers_nm: tuple[int, ...]
    width_nm: int

    def __post_init__(self) -> None:
        if not isinstance(self.use_id, str) or not self.use_id:
            raise CandidatePoolError("allocated track use_id must be non-empty")
        if not isinstance(self.resource, TrackResource):
            raise CandidatePoolError(
                "allocated track resource must be a full TrackResource"
            )
        interval = tuple(self.interval_nm)
        if (
            len(interval) != 2
            or not all(_is_int_nm(value) for value in interval)
            or interval[1] <= interval[0]
        ):
            raise CandidatePoolError(
                "allocated track interval must be a non-empty integer interval"
            )
        if not _is_int_nm(self.track_start) or self.track_start < 0:
            raise CandidatePoolError(
                "allocated track start must be a non-negative integer"
            )
        if not _is_int_nm(self.track_span) or self.track_span <= 0:
            raise CandidatePoolError(
                "allocated track span must be a positive integer"
            )
        if not _is_int_nm(self.width_nm) or self.width_nm <= 0:
            raise CandidatePoolError(
                "allocated track width_nm must be a positive integer nm"
            )
        centers = tuple(self.selected_track_centers_nm)
        if len(centers) != self.track_span or not all(
            _is_int_nm(center) for center in centers
        ):
            raise CandidatePoolError(
                "allocated selected track centers must exactly match track_span"
            )
        object.__setattr__(self, "interval_nm", interval)
        object.__setattr__(self, "selected_track_centers_nm", centers)


@dataclass(frozen=True)
class AllocatedLandingOption:
    demand_id: str
    option: LandingOption
    landing_sites: tuple[LandingSite, ...]
    actual_multiplicity: int

    def __post_init__(self) -> None:
        if not isinstance(self.demand_id, str) or not self.demand_id:
            raise CandidatePoolError(
                "allocated landing demand_id must be non-empty"
            )
        if not isinstance(self.option, LandingOption):
            raise CandidatePoolError(
                "allocated landing option must be LandingOption"
            )
        sites = tuple(self.landing_sites)
        if not all(isinstance(site, LandingSite) for site in sites):
            raise CandidatePoolError(
                "allocated landing sites must be full LandingSite values"
            )
        if self.actual_multiplicity != len(sites):
            raise CandidatePoolError(
                "allocated landing actual_multiplicity must equal site count"
            )
        object.__setattr__(self, "landing_sites", tuple(sorted(
            sites, key=lambda site: site.site_id
        )))


@dataclass(frozen=True)
class AllocatedNetProgram:
    net: str
    candidate_id: str
    program_kind: str
    source_topology_digest: str
    width_nm: int
    via_multiplicity: int
    branches: tuple[CandidateBranch, ...]
    goal_branch_coverage: tuple[GoalBranchCoverage, ...]
    endpoint_accesses: tuple[CandidateEndpointAccess, ...]
    landing_demands: tuple[LandingDemand, ...]
    required_route_goals: tuple[RequiredRouteGoal, ...]
    tracks: tuple[AllocatedTrack, ...]
    landing_options: tuple[AllocatedLandingOption, ...]
    candidate_pool_hash: str
    routing_resource_ledger_hash: str
    routing_world_provenance_digest: str
    owned_conductor_index_digest: str = ""
    owned_conductor_components: tuple[
        OwnedConductorComponentCertificate, ...
    ] = ()
    owned_component_use_refs: tuple[str, ...] = ()
    schema: str = "clay.allocated-net-program.v2"

    def __post_init__(self) -> None:
        if self.schema != "clay.allocated-net-program.v2":
            raise CandidatePoolError(
                "allocated net program schema must be canonical v2"
            )
        for name in (
            "net",
            "candidate_id",
            "program_kind",
            "source_topology_digest",
            "candidate_pool_hash",
            "routing_resource_ledger_hash",
            "routing_world_provenance_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CandidatePoolError(
                    f"allocated net program {name} must be non-empty"
                )
        if not isinstance(self.owned_conductor_index_digest, str):
            raise CandidatePoolError(
                "allocated net program owned_conductor_index_digest must be a string"
            )
        object.__setattr__(self, "branches", tuple(self.branches))
        if self.program_kind == "POWER_SPINE_LOCAL_DROPS":
            if not all(
                isinstance(branch, CandidateBranch)
                for branch in self.branches
            ):
                raise CandidatePoolError(
                    "allocated power program branches must be CandidateBranch values"
                )
            _require_shared_power_lifted_root(self.branches)
        object.__setattr__(
            self, "goal_branch_coverage", tuple(self.goal_branch_coverage)
        )
        object.__setattr__(self, "endpoint_accesses", tuple(self.endpoint_accesses))
        if not all(
            isinstance(certificate, OwnedConductorComponentCertificate)
            for certificate in self.owned_conductor_components
        ):
            raise CandidatePoolError(
                "allocated program owned conductor certificates are malformed"
            )
        object.__setattr__(
            self,
            "owned_conductor_components",
            tuple(sorted(
                self.owned_conductor_components,
                key=lambda certificate: certificate.component.component_ref,
            )),
        )
        component_use_refs = _bounded_materialize(
            self.owned_component_use_refs,
            limit=MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE,
            label="allocated owned component use ref",
            error_type=CandidatePoolLimitError,
        )
        if (
            any(
                not isinstance(component_ref, str) or not component_ref
                for component_ref in component_use_refs
            )
            or len(component_use_refs) != len(set(component_use_refs))
        ):
            raise CandidatePoolError(
                "allocated owned_component_use_refs must be unique non-empty strings"
            )
        certificate_refs = {
            certificate.component.component_ref
            for certificate in self.owned_conductor_components
        }
        if not set(component_use_refs).issubset(certificate_refs):
            raise CandidatePoolError(
                "allocated owned component use refs lack exact certificates"
            )
        object.__setattr__(
            self,
            "owned_component_use_refs",
            tuple(sorted(component_use_refs)),
        )
        if not all(
            isinstance(demand, LandingDemand) for demand in self.landing_demands
        ):
            raise CandidatePoolError(
                "allocated program landing_demands must be LandingDemand values"
            )
        if not all(
            isinstance(goal, RequiredRouteGoal)
            for goal in self.required_route_goals
        ):
            raise CandidatePoolError(
                "allocated program required_route_goals must be RequiredRouteGoal values"
            )
        object.__setattr__(
            self, "landing_demands", tuple(self.landing_demands)
        )
        object.__setattr__(
            self, "required_route_goals", tuple(self.required_route_goals)
        )
        object.__setattr__(self, "tracks", tuple(sorted(
            self.tracks, key=lambda track: track.use_id
        )))
        object.__setattr__(self, "landing_options", tuple(sorted(
            self.landing_options,
            key=lambda allocation: (
                allocation.demand_id,
                allocation.option.option_id,
            ),
        )))

    def canonical_json(self) -> str:
        return json.dumps(
            _allocated_program_payload(self),
            sort_keys=True,
            separators=(",", ":"),
        )

    def sha256(self) -> str:
        return hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class RoutePlanObjectivePhase:
    """One replayable phase of the canonical global route-plan objective."""

    phase_id: str
    status: str
    value: Optional[int]
    lower_bound: Optional[int]

    def __post_init__(self) -> None:
        if self.phase_id not in _ROUTE_PLAN_OBJECTIVE_PHASE_IDS:
            raise CandidatePoolError(
                f"unknown route-plan objective phase {self.phase_id!r}"
            )
        if self.status not in {"OPTIMAL", "FEASIBLE", "UNKNOWN"}:
            raise CandidatePoolError(
                "route-plan objective phase status must be OPTIMAL, FEASIBLE, or UNKNOWN"
            )
        for name in ("value", "lower_bound"):
            item = getattr(self, name)
            if item is not None and (
                type(item) is not int or item < 0 or item > CP_SAT_INT64_MAX
            ):
                raise CandidatePoolError(
                    f"route-plan objective phase {name} must be an int64 non-negative integer"
                )
        scalar_phase = self.phase_id in {
            "new_reserved_geometry_area_nm2",
            "landing_site_cut_count",
        }
        if scalar_phase:
            if self.status in {"OPTIMAL", "FEASIBLE"} and self.value is None:
                raise CandidatePoolError(
                    "scalar objective phase requires an incumbent value"
                )
            if self.status in {"OPTIMAL", "FEASIBLE"} and self.lower_bound is None:
                raise CandidatePoolError(
                    "scalar objective phase requires a lower bound"
                )
            if self.status == "UNKNOWN" and (
                self.value is not None or self.lower_bound is not None
            ):
                raise CandidatePoolError(
                    "UNKNOWN scalar objective phase cannot claim value or lower bound"
                )
        elif self.phase_id == "hard_feasibility":
            if self.status == "OPTIMAL" and (
                self.value, self.lower_bound
            ) != (0, 0):
                raise CandidatePoolError(
                    "OPTIMAL hard feasibility phase must carry the exact zero witness"
                )
            if self.status == "UNKNOWN" and (
                self.value is not None or self.lower_bound is not None
            ):
                raise CandidatePoolError(
                    "UNKNOWN hard feasibility phase cannot claim scalar evidence"
                )
            if self.status == "FEASIBLE":
                raise CandidatePoolError(
                    "hard feasibility phase is either proved OPTIMAL or UNKNOWN"
                )
        if self.value is not None and self.lower_bound is not None:
            if self.lower_bound > self.value:
                raise CandidatePoolError(
                    "objective lower bound cannot exceed incumbent value"
                )
            if self.status == "OPTIMAL" and self.lower_bound != self.value:
                raise CandidatePoolError(
                    "OPTIMAL objective phase must close its lower bound"
                )
        if scalar_phase and self.status == "FEASIBLE" and self.lower_bound != 0:
            raise CandidatePoolError(
                "FEASIBLE scalar objective phase carries only the conservative "
                "zero lower bound; it is not an optimality certificate"
            )
        if self.phase_id == "canonical_semantic_tie" and (
            self.value is not None or self.lower_bound is not None
        ):
            raise CandidatePoolError(
                "canonical semantic tie phase has no scalar objective value"
            )


@dataclass(frozen=True)
class RoutePlanObjectiveCertificate:
    """Pool- and policy-bound evidence for the lexicographic plan solve."""

    candidate_pool_hash: str
    objective_policy_digest: str
    phases: tuple[RoutePlanObjectivePhase, ...]
    canonical_decision_digest: str
    authorized_replay_deterministic_time: float
    schema: str = "clay.route-plan-objective-certificate.v2"

    def __post_init__(self) -> None:
        if self.schema != "clay.route-plan-objective-certificate.v2":
            raise CandidatePoolError(
                "route-plan objective certificate schema must be canonical v2"
            )
        for name in (
            "candidate_pool_hash",
            "objective_policy_digest",
            "canonical_decision_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CandidatePoolError(
                    f"route-plan objective certificate {name} must be non-empty"
                )
        replay_budget = self.authorized_replay_deterministic_time
        if (
            isinstance(replay_budget, bool)
            or not isinstance(replay_budget, (int, float))
            or float(replay_budget) < 0
            or not math.isfinite(float(replay_budget))
            or float(replay_budget)
            > ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME
        ):
            raise CandidatePoolError(
                "route-plan objective authorized replay budget must be finite "
                "and within the trusted hard limit"
            )
        object.__setattr__(
            self,
            "authorized_replay_deterministic_time",
            float(replay_budget),
        )
        phases = _bounded_materialize(
            self.phases,
            limit=len(_ROUTE_PLAN_OBJECTIVE_PHASE_IDS),
            label="route-plan objective phase",
            error_type=CandidatePoolLimitError,
        )
        if not phases or not all(
            isinstance(phase, RoutePlanObjectivePhase) for phase in phases
        ):
            raise CandidatePoolError(
                "route-plan objective certificate phases are malformed"
            )
        phase_ids = tuple(phase.phase_id for phase in phases)
        if phase_ids != _ROUTE_PLAN_OBJECTIVE_PHASE_IDS[:len(phase_ids)]:
            raise CandidatePoolError(
                "route-plan objective phases must be a canonical ordered prefix"
            )
        if any(phase.status != "OPTIMAL" for phase in phases[:-1]):
            raise CandidatePoolError(
                "all preceding phases must be OPTIMAL before a later objective phase"
            )
        object.__setattr__(self, "phases", phases)

    def canonical_json(self) -> str:
        return json.dumps(
            _route_plan_objective_certificate_payload(self),
            sort_keys=True,
            separators=(",", ":"),
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RouteResourcePlan:
    candidate_pool_hash: str
    selected_nets: tuple[SelectedNetCandidate, ...]
    allocated_programs: tuple[AllocatedNetProgram, ...] = ()
    materialization_complete: bool = False
    objective_certificate: Optional[RoutePlanObjectiveCertificate] = None
    owned_conductor_index_certificate: Optional[
        OwnedConductorIndexCertificate
    ] = None
    landing_site_conflict_domain_digest: str = ""
    lazy_landing_site_conflicts: tuple[LandingSitePairConflict, ...] = ()
    schema: str = "clay.route-resource-plan.v4"

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_nets", tuple(sorted(
            self.selected_nets,
            key=lambda selected: (selected.net, selected.candidate_id),
        )))
        object.__setattr__(self, "allocated_programs", tuple(sorted(
            self.allocated_programs,
            key=lambda program: (program.net, program.candidate_id),
        )))
        if not isinstance(self.materialization_complete, bool):
            raise CandidatePoolError("materialization_complete must be boolean")
        if self.objective_certificate is not None and not isinstance(
            self.objective_certificate, RoutePlanObjectiveCertificate
        ):
            raise CandidatePoolError(
                "objective_certificate must be RoutePlanObjectiveCertificate when present"
            )
        if self.owned_conductor_index_certificate is not None and not isinstance(
            self.owned_conductor_index_certificate,
            OwnedConductorIndexCertificate,
        ):
            raise CandidatePoolError(
                "owned_conductor_index_certificate must be "
                "OwnedConductorIndexCertificate when present"
            )
        if not isinstance(self.landing_site_conflict_domain_digest, str):
            raise CandidatePoolError(
                "landing_site_conflict_domain_digest must be a string"
            )
        lazy_conflicts = tuple(sorted(_bounded_materialize(
            self.lazy_landing_site_conflicts,
            limit=MAX_LAZY_LANDING_SITE_CONFLICTS,
            label="lazy landing site conflict",
            error_type=CandidatePoolLimitError,
        )))
        if not all(
            isinstance(conflict, LandingSitePairConflict)
            for conflict in lazy_conflicts
        ):
            raise CandidatePoolError(
                "lazy_landing_site_conflicts has invalid values"
            )
        if len(lazy_conflicts) != len(set(lazy_conflicts)):
            raise CandidatePoolError(
                "lazy landing site conflicts must be unique"
            )
        if lazy_conflicts and not self.landing_site_conflict_domain_digest:
            raise CandidatePoolError(
                "lazy landing conflicts require a bound conflict domain"
            )
        object.__setattr__(self, "lazy_landing_site_conflicts", lazy_conflicts)
        if self.schema not in {
            "clay.route-resource-plan.v4",
            "clay.route-resource-plan.v5",
            "clay.route-resource-plan.v6",
            "clay.route-resource-plan.v7",
            "clay.route-resource-plan.v8",
            "clay.route-resource-plan.v9",
            "clay.route-resource-plan.v10",
        }:
            raise CandidatePoolError("unknown route resource plan schema")
        if (
            self.schema == "clay.route-resource-plan.v4"
            and self.objective_certificate is not None
        ):
            raise CandidatePoolError(
                "legacy v4 route resource plan cannot carry a v5 objective certificate"
            )
        if self.schema not in {
            "clay.route-resource-plan.v6",
            "clay.route-resource-plan.v8",
            "clay.route-resource-plan.v10",
        } and (
            self.landing_site_conflict_domain_digest or lazy_conflicts
        ):
            raise CandidatePoolError(
                "landing conflict domain evidence requires route plan v6"
            )
        if self.schema in {
            "clay.route-resource-plan.v9",
            "clay.route-resource-plan.v10",
        } and self.owned_conductor_index_certificate is None:
            raise CandidatePoolError(
                "exact route plan requires a complete owned "
                "conductor index certificate"
            )

    def canonical_json(self) -> str:
        return json.dumps(_route_resource_plan_payload(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiagnosticRouteProgramBundle:
    """Materializable bring-up programs that can never serve as an exact plan.

    A diagnostic decomposition selects and verifies bounded local/merge pools,
    but those pools are not the complete global candidate universe.  This
    distinct record transports only the resulting geometry programs to the
    diagnostic engine seam.  Exact plan replay rejects it by type, and the
    gridless engine accepts it only through its dedicated diagnostic keyword.
    """

    frontier_digest: str
    source_stop_digest: str
    family_projection_digest: str
    selected_nets: tuple[SelectedNetCandidate, ...]
    allocated_programs: tuple[AllocatedNetProgram, ...]
    owned_conductor_index_certificate: OwnedConductorIndexCertificate
    source_witness_hashes: tuple[tuple[str, str, str], ...] = ()
    failed_partition_refs: tuple[str, ...] = ()
    landing_site_conflict_domain_digest: str = ""
    lazy_landing_site_conflicts: tuple[LandingSitePairConflict, ...] = ()
    materialization_complete: bool = True
    schema: str = "clay.bringup-diagnostic-route-program.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("frontier_digest", self.frontier_digest),
            ("source_stop_digest", self.source_stop_digest),
            ("family_projection_digest", self.family_projection_digest),
        ):
            if type(value) is not str or len(value) != 64:
                raise CandidatePoolError(f"diagnostic {label} must be SHA-256")
        if self.schema != "clay.bringup-diagnostic-route-program.v1":
            raise CandidatePoolError("unknown diagnostic route-program schema")
        if self.materialization_complete is not True:
            raise CandidatePoolError(
                "diagnostic route programs must be byte-complete for their subset"
            )
        if type(self.owned_conductor_index_certificate) is not (
            OwnedConductorIndexCertificate
        ):
            raise CandidatePoolError(
                "diagnostic route programs require the exact owned-conductor index"
            )
        selected = tuple(sorted(_bounded_materialize(
            self.selected_nets,
            limit=MAX_REQUIRED_NETS,
            label="diagnostic selected net",
            error_type=CandidatePoolLimitError,
        ), key=lambda item: (item.net, item.candidate_id)))
        programs = tuple(sorted(_bounded_materialize(
            self.allocated_programs,
            limit=MAX_REQUIRED_NETS,
            label="diagnostic allocated program",
            error_type=CandidatePoolLimitError,
        ), key=lambda item: (item.net, item.candidate_id)))
        if (
            any(type(item) is not SelectedNetCandidate for item in selected)
            or any(type(item) is not AllocatedNetProgram for item in programs)
        ):
            raise CandidatePoolError(
                "diagnostic route program contains foreign selected/program values"
            )
        selected_pairs = tuple((item.net, item.candidate_id) for item in selected)
        program_pairs = tuple((item.net, item.candidate_id) for item in programs)
        if (
            not selected_pairs
            or len(selected_pairs) != len(set(selected_pairs))
            or selected_pairs != program_pairs
            or any(
                program.candidate_pool_hash != self.frontier_digest
                for program in programs
            )
        ):
            raise CandidatePoolError(
                "diagnostic selected programs do not exactly bind the frontier"
            )
        witnesses = tuple(sorted(_bounded_materialize(
            self.source_witness_hashes,
            limit=MAX_TOTAL_CANDIDATES,
            label="diagnostic source witness",
            error_type=CandidatePoolLimitError,
        )))
        if any(
            type(item) is not tuple
            or len(item) != 3
            or any(type(value) is not str or not value for value in item)
            for item in witnesses
        ):
            raise CandidatePoolError("diagnostic source witnesses are malformed")
        failures = tuple(sorted(_bounded_materialize(
            self.failed_partition_refs,
            limit=MAX_TOTAL_CANDIDATES,
            label="diagnostic failed partition",
            error_type=CandidatePoolLimitError,
        )))
        if any(type(item) is not str or not item for item in failures):
            raise CandidatePoolError("diagnostic failed partition refs are malformed")
        lazy = tuple(sorted(_bounded_materialize(
            self.lazy_landing_site_conflicts,
            limit=MAX_LAZY_LANDING_SITE_CONFLICTS,
            label="diagnostic lazy landing conflict",
            error_type=CandidatePoolLimitError,
        )))
        if any(type(item) is not LandingSitePairConflict for item in lazy):
            raise CandidatePoolError(
                "diagnostic lazy landing conflicts contain foreign values"
            )
        object.__setattr__(self, "selected_nets", selected)
        object.__setattr__(self, "allocated_programs", programs)
        object.__setattr__(self, "source_witness_hashes", witnesses)
        object.__setattr__(self, "failed_partition_refs", failures)
        object.__setattr__(self, "lazy_landing_site_conflicts", lazy)

    @property
    def candidate_pool_hash(self) -> str:
        """Compatibility name used only by the geometry materializer."""

        return self.frontier_digest

    def canonical_json(self) -> str:
        return json.dumps({
            "allocated_programs": [
                _allocated_program_payload(program)
                for program in self.allocated_programs
            ],
            "failed_partition_refs": list(self.failed_partition_refs),
            "family_projection_digest": self.family_projection_digest,
            "frontier_digest": self.frontier_digest,
            "landing_site_conflict_domain_digest": (
                self.landing_site_conflict_domain_digest
            ),
            "lazy_landing_site_conflicts": [
                {
                    "conflict_scope": item.conflict_scope,
                    "left_site_id": item.left_site_id,
                    "right_site_id": item.right_site_id,
                    "routing_world_provenance_digest": (
                        item.routing_world_provenance_digest
                    ),
                    "rule_refs": list(item.rule_refs),
                }
                for item in self.lazy_landing_site_conflicts
            ],
            "owned_conductor_index_certificate": (
                _owned_conductor_index_certificate_payload(
                    self.owned_conductor_index_certificate
                )
            ),
            "schema": self.schema,
            "selected_nets": [
                {
                    "candidate_id": item.candidate_id,
                    "net": item.net,
                    "selected_landings": [
                        {
                            "demand_id": landing.demand_id,
                            "site_id": landing.site_id,
                        }
                        for landing in item.selected_landings
                    ],
                    "selected_landing_options": [
                        {
                            "demand_id": option.demand_id,
                            "option_id": option.option_id,
                        }
                        for option in item.selected_landing_options
                    ],
                    "selected_tracks": [
                        {
                            "resource": _key_payload(track.resource),
                            "track_start": track.track_start,
                            "use_id": track.use_id,
                        }
                        for track in item.selected_tracks
                    ],
                }
                for item in self.selected_nets
            ],
            "source_stop_digest": self.source_stop_digest,
            "source_witness_hashes": [list(item) for item in self.source_witness_hashes],
        }, sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


_ROUTE_PLAN_VERIFICATION_RECEIPT_SCHEMA = (
    "clay.route-plan-verification-receipt.v1"
)
_ROUTE_PLAN_VERIFIER_VERSION = "clay.route-resource-plan-verifier.v1"
_ROUTE_PLAN_VERIFICATION_ISSUER_AUTHORITY = object()
_ROUTE_PLAN_VERIFICATION_RECEIPTS: dict[
    int, tuple[weakref.ReferenceType[object], str]
] = {}


@dataclass(frozen=True, init=False)
class RoutePlanVerificationReceipt:
    """Process-local proof that this exact pool/plan completed deep replay."""

    candidate_pool_hash: str
    route_resource_plan_hash: str
    objective_certificate_hash: str
    route_resource_plan_schema: str
    verifier_version: str
    schema: str
    _issuer_authority: object = field(repr=False, compare=False, init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "RoutePlanVerificationReceipt is a module-issued capability"
        )


def _route_plan_verification_receipt_payload(
    receipt: RoutePlanVerificationReceipt,
) -> str:
    return _canonical_sha256({
        "candidate_pool_hash": receipt.candidate_pool_hash,
        "objective_certificate_hash": receipt.objective_certificate_hash,
        "route_resource_plan_hash": receipt.route_resource_plan_hash,
        "route_resource_plan_schema": receipt.route_resource_plan_schema,
        "schema": receipt.schema,
        "verifier_version": receipt.verifier_version,
    })


def _issue_route_plan_verification_receipt(
    pool: CandidatePool,
    plan: RouteResourcePlan,
) -> RoutePlanVerificationReceipt:
    certificate = plan.objective_certificate
    if certificate is None:
        raise RouteResourcePlanVerificationError(
            "verified route plan is missing its objective certificate"
        )
    receipt = object.__new__(RoutePlanVerificationReceipt)
    object.__setattr__(receipt, "candidate_pool_hash", pool.sha256())
    object.__setattr__(receipt, "route_resource_plan_hash", plan.sha256())
    object.__setattr__(
        receipt, "objective_certificate_hash", certificate.sha256()
    )
    object.__setattr__(
        receipt, "route_resource_plan_schema", plan.schema
    )
    object.__setattr__(
        receipt, "verifier_version", _ROUTE_PLAN_VERIFIER_VERSION
    )
    object.__setattr__(
        receipt, "schema", _ROUTE_PLAN_VERIFICATION_RECEIPT_SCHEMA
    )
    object.__setattr__(
        receipt,
        "_issuer_authority",
        _ROUTE_PLAN_VERIFICATION_ISSUER_AUTHORITY,
    )
    receipt_id = id(receipt)

    def discard_receipt(receipt_ref: weakref.ReferenceType[object]) -> None:
        current = _ROUTE_PLAN_VERIFICATION_RECEIPTS.get(receipt_id)
        if current is not None and current[0] is receipt_ref:
            _ROUTE_PLAN_VERIFICATION_RECEIPTS.pop(receipt_id, None)

    receipt_ref = weakref.ref(receipt, discard_receipt)
    _ROUTE_PLAN_VERIFICATION_RECEIPTS[receipt_id] = (
        receipt_ref,
        _route_plan_verification_receipt_payload(receipt),
    )
    return receipt


def verify_route_plan_verification_receipt(
    pool: CandidatePool,
    plan: RouteResourcePlan,
    receipt: RoutePlanVerificationReceipt,
) -> bool:
    """Validate an intact selector-issued receipt without replaying CP-SAT."""

    if type(receipt) is not RoutePlanVerificationReceipt:
        raise TypeError("receipt must be RoutePlanVerificationReceipt")
    pool = _replay_candidate_pool_runtime(pool)
    if (
        getattr(receipt, "_issuer_authority", None)
        is not _ROUTE_PLAN_VERIFICATION_ISSUER_AUTHORITY
    ):
        raise TypeError("receipt must be a module-issued capability")
    issued = _ROUTE_PLAN_VERIFICATION_RECEIPTS.get(id(receipt))
    if (
        issued is None
        or issued[0]() is not receipt
        or issued[1] != _route_plan_verification_receipt_payload(receipt)
    ):
        raise TypeError("receipt must be an intact module-issued capability")
    if receipt.schema != _ROUTE_PLAN_VERIFICATION_RECEIPT_SCHEMA:
        raise RouteResourcePlanVerificationError(
            "route-plan verification receipt schema mismatch"
        )
    if receipt.verifier_version != _ROUTE_PLAN_VERIFIER_VERSION:
        raise RouteResourcePlanVerificationError(
            "route-plan verification receipt verifier version mismatch"
        )
    if receipt.candidate_pool_hash != pool.sha256():
        raise RouteResourcePlanVerificationError(
            "route-plan verification receipt pool hash mismatch"
        )
    if receipt.route_resource_plan_hash != plan.sha256():
        raise RouteResourcePlanVerificationError(
            "route-plan verification receipt plan hash mismatch"
        )
    if receipt.route_resource_plan_schema != plan.schema:
        raise RouteResourcePlanVerificationError(
            "route-plan verification receipt plan schema mismatch"
        )
    certificate = plan.objective_certificate
    if (
        certificate is None
        or receipt.objective_certificate_hash != certificate.sha256()
    ):
        raise RouteResourcePlanVerificationError(
            "route-plan verification receipt objective certificate hash mismatch"
        )
    return True


@dataclass(frozen=True)
class CapacityConflict:
    candidate_pool_hash: str
    resource: CorridorLayerKey
    overlap_interval_nm: tuple[int, int]
    track_capacity: int
    nets: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    declaration_refs: tuple[str, ...]


@dataclass(frozen=True)
class HallDeficit:
    candidate_pool_hash: str
    demand_ids: tuple[str, ...]
    reachable_site_ids: tuple[str, ...]
    deficit: int
    candidate_ids: tuple[str, ...]
    declaration_refs: tuple[str, ...]
    expansion_resource_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PoolUnsatCertificate:
    candidate_pool_hash: str
    assumption_core_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CompleteNetSelectionResult:
    status: str
    solver_status: str
    candidate_pool_hash: str
    plan: Optional[RouteResourcePlan] = None
    partial_plan: Optional[RouteResourcePlan] = None
    certificate: Optional[object] = None
    verification_receipt: Optional[RoutePlanVerificationReceipt] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def canonical_json(self) -> str:
        payload = {
            "candidate_pool_hash": self.candidate_pool_hash,
            "certificate": _certificate_payload(self.certificate),
            "partial_plan": (
                None if self.partial_plan is None else _route_resource_plan_payload(self.partial_plan)
            ),
            "plan": None if self.plan is None else _route_resource_plan_payload(self.plan),
            "solver_status": self.solver_status,
            "status": self.status,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _key_payload(key: CorridorLayerKey) -> dict[str, object]:
    return {"corridor_id": key.corridor_id, "layer_n": key.layer_n}


def _allocated_program_payload(program: AllocatedNetProgram) -> dict[str, object]:
    return {
        "branches": [
            {
                "branch_id": branch.branch_id,
                "ordered_landing_demand_ids": list(
                    branch.ordered_landing_demand_ids
                ),
                "ordered_steps": [
                    {"kind": step.kind, "ref_id": step.ref_id}
                    for step in branch.ordered_steps
                ],
                "ordered_use_ids": list(branch.ordered_use_ids),
                "source_endpoint": branch.source_endpoint,
                "target_endpoint": branch.target_endpoint,
            }
            for branch in program.branches
        ],
        "candidate_id": program.candidate_id,
        "candidate_pool_hash": program.candidate_pool_hash,
        "endpoint_accesses": [
            {
                "layer_key": list(access.layer_key),
                "layer_n": access.layer_n,
                "owner_component_refs": list(access.owner_component_refs),
                "provenance": access.provenance,
                "xy_nm": list(access.xy_nm),
            }
            for access in program.endpoint_accesses
        ],
        "goal_branch_coverage": [
            {
                "branch_id": item.branch_id,
                "declaration_ref": item.declaration_ref,
                "target_provenance": item.target_provenance,
            }
            for item in program.goal_branch_coverage
        ],
        "landing_options": [
            {
                "actual_multiplicity": allocation.actual_multiplicity,
                "demand_id": allocation.demand_id,
                "landing_sites": [
                    {
                        "capacity": site.capacity,
                        "cut_bbox_nm": list(site.cut_bbox_nm or ()),
                        "lower_layer_n": site.lower_layer_n,
                        "lower_owner_component_refs": list(
                            site.lower_owner_component_refs
                        ),
                        "lower_pad_bbox_nm": list(site.lower_pad_bbox_nm or ()),
                        "owner_layer_ns": list(site.owner_layer_ns),
                        "owner_net": site.owner_net,
                        "provenance": site.provenance,
                        "routing_world_provenance_digest": (
                            site.routing_world_provenance_digest
                        ),
                        "site_id": site.site_id,
                        "upper_layer_n": site.upper_layer_n,
                        "upper_owner_component_refs": list(
                            site.upper_owner_component_refs
                        ),
                        "upper_pad_bbox_nm": list(site.upper_pad_bbox_nm or ()),
                        "xy_nm": list(site.xy_nm),
                    }
                    for site in allocation.landing_sites
                ],
                "option": {
                    "contacts": [
                        {
                            "site_id": contact.site_id,
                            "track_choice": {
                                "track_start": contact.track_start,
                                "use_id": contact.use_id,
                            },
                        }
                        for contact in allocation.option.contacts
                    ],
                    "demand_id": allocation.option.demand_id,
                    "option_id": allocation.option.option_id,
                    "site_ids": list(allocation.option.site_ids),
                },
            }
            for allocation in program.landing_options
        ],
        "landing_demands": [
            {
                "allowed_site_ids": list(demand.allowed_site_ids),
                "declaration_ref": demand.declaration_ref,
                "demand_id": demand.demand_id,
                "required_multiplicity": demand.required_multiplicity,
                "selection_mode": demand.selection_mode,
                "terminal_provenance": demand.terminal_provenance,
            }
            for demand in program.landing_demands
        ],
        "net": program.net,
        "program_kind": program.program_kind,
        "routing_resource_ledger_hash": program.routing_resource_ledger_hash,
        "owned_conductor_index_digest": program.owned_conductor_index_digest,
        "owned_conductor_components": [
            {
                "component": certificate.component.canonical_payload(),
                "layer_n": certificate.layer_n,
            }
            for certificate in program.owned_conductor_components
        ],
        "owned_component_use_refs": list(program.owned_component_use_refs),
        "routing_world_provenance_digest": (
            program.routing_world_provenance_digest
        ),
        "schema": program.schema,
        "source_topology_digest": program.source_topology_digest,
        "required_route_goals": [
            {
                "declaration_ref": goal.declaration_ref,
                "kind": goal.kind,
                "net": goal.net,
                "target": None if goal.target is None else {
                    "layer": list(goal.target.layer),
                    "provenance": goal.target.provenance,
                    "xy_nm": list(goal.target.xy_nm),
                },
            }
            for goal in program.required_route_goals
        ],
        "tracks": [
            {
                "interval_nm": list(track.interval_nm),
                "resource": {
                    "corridor_bbox_nm": list(
                        track.resource.corridor_bbox_nm or ()
                    ),
                    "key": _key_payload(track.resource.key),
                    "gds_layer_key": (
                        None
                        if track.resource.gds_layer_key is None
                        else list(track.resource.gds_layer_key)
                    ),
                    "preferred_axis": track.resource.preferred_axis,
                    "provenance": track.resource.provenance,
                    "routing_world_provenance_digest": (
                        track.resource.routing_world_provenance_digest
                    ),
                    "through_interval_nm": list(
                        track.resource.through_interval_nm or ()
                    ),
                    "track_centers_nm": list(track.resource.track_centers_nm),
                    **(
                        {"owner_net": track.resource.owner_net}
                        if track.resource.owner_net is not None
                        else {}
                    ),
                },
                "selected_track_centers_nm": list(
                    track.selected_track_centers_nm
                ),
                "track_span": track.track_span,
                "track_start": track.track_start,
                "use_id": track.use_id,
                "width_nm": track.width_nm,
            }
            for track in program.tracks
        ],
        "via_multiplicity": program.via_multiplicity,
        "width_nm": program.width_nm,
    }


def _resolve_allocated_net_program(
    pool: CandidatePool,
    candidate: CompleteNetCandidate,
    selected: SelectedNetCandidate,
    candidate_pool_hash: str,
) -> AllocatedNetProgram:
    resources_by_key = {resource.key: resource for resource in pool.resources}
    sites_by_id = pool.landing_sites_by_id()
    tracks_by_use = {track.use_id: track for track in selected.selected_tracks}
    options_by_id = {
        option.option_id: option for option in candidate.landing_options
    }
    allocations = []
    for selected_option in selected.selected_landing_options:
        option = options_by_id[selected_option.option_id]
        sites = tuple(sites_by_id[site_id] for site_id in option.site_ids)
        allocations.append(AllocatedLandingOption(
            demand_id=selected_option.demand_id,
            option=option,
            landing_sites=sites,
            actual_multiplicity=len(sites),
        ))
    demands_by_id = {
        demand.demand_id: demand for demand in candidate.landing_demands
    }
    neighbour_use_ids_by_demand = _landing_neighbour_use_ids(candidate)
    for selected_landing in selected.selected_landings:
        demand = demands_by_id[selected_landing.demand_id]
        if (
            demand.selection_mode
            != LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
        ):
            continue
        site = sites_by_id[selected_landing.site_id]
        contacts = tuple(
            LandingTrackContact(
                site_id=site.site_id,
                track_choice=TrackChoiceRef(
                    use_id=use_id,
                    track_start=tracks_by_use[use_id].track_start,
                ),
            )
            for use_id in neighbour_use_ids_by_demand[demand.demand_id]
        )
        option_payload = {
            "contacts": [
                {
                    "site_id": contact.site_id,
                    "track_start": contact.track_start,
                    "use_id": contact.use_id,
                }
                for contact in contacts
            ],
            "demand_id": demand.demand_id,
            "schema": "clay.derived-landing-option.v1",
            "site_id": site.site_id,
        }
        option = LandingOption(
            option_id=hashlib.sha256(json.dumps(
                option_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            demand_id=demand.demand_id,
            site_ids=(site.site_id,),
            contacts=contacts,
        )
        allocations.append(AllocatedLandingOption(
            demand_id=demand.demand_id,
            option=option,
            landing_sites=(site,),
            actual_multiplicity=1,
        ))
    tracks = []
    for use in candidate.corridor_uses:
        selected_track = tracks_by_use[use.use_id]
        resource = resources_by_key[use.resource]
        centers = resource.track_centers_nm[
            selected_track.track_start:
            selected_track.track_start + use.track_span
        ]
        tracks.append(AllocatedTrack(
            use_id=use.use_id,
            resource=resource,
            interval_nm=use.interval_nm,
            track_start=selected_track.track_start,
            track_span=use.track_span,
            selected_track_centers_nm=centers,
            width_nm=_corridor_use_width_nm(use, candidate.width_nm),
        ))
    used_component_refs = {
        component_ref
        for allocation in allocations
        for site in allocation.landing_sites
        for component_ref in (
            *site.lower_owner_component_refs,
            *site.upper_owner_component_refs,
        )
    } | {
        component_ref
        for access in candidate.endpoint_accesses
        for component_ref in access.owner_component_refs
    } | set(candidate.owned_component_use_refs)
    component_certificates = tuple(
        certificate
        for certificate in pool.owned_conductor_components
        if certificate.component.component_ref in used_component_refs
    )
    return AllocatedNetProgram(
        net=candidate.net,
        candidate_id=candidate.candidate_id,
        program_kind=candidate.program_kind,
        source_topology_digest=candidate.source_topology_digest,
        width_nm=candidate.width_nm,
        via_multiplicity=candidate.via_multiplicity,
        branches=candidate.branches,
        goal_branch_coverage=candidate.goal_branch_coverage,
        endpoint_accesses=candidate.endpoint_accesses,
        landing_demands=candidate.landing_demands,
        required_route_goals=tuple(
            goal for goal in pool.required_route_goals if goal.net == candidate.net
        ),
        tracks=tuple(tracks),
        landing_options=tuple(allocations),
        candidate_pool_hash=candidate_pool_hash,
        routing_resource_ledger_hash=pool.routing_resource_ledger_hash,
        routing_world_provenance_digest=pool.routing_world_provenance_digest,
        owned_conductor_index_digest=pool.owned_conductor_index_digest,
        owned_conductor_components=component_certificates,
        owned_component_use_refs=candidate.owned_component_use_refs,
    )


def _candidate_pool_payload(pool: CandidatePool) -> dict[str, object]:
    is_v12 = bool(pool.lane_incidence_replay_digest)
    validated_candidate_families = tuple(
        replace(family) for family in pool.candidate_families
    )
    raw_factorized_families = _bounded_materialize(
        pool.factorized_candidate_families,
        limit=MAX_TOTAL_CANDIDATES,
        label="factorized candidate family",
        error_type=CandidatePoolLimitError,
    )
    factorized_rebuild_budget = _CandidatePoolRebuildBudget()
    validated_factorized_families = tuple(
        _rebuild_factorized_candidate_family(
            family, pool_budget=factorized_rebuild_budget
        )
        for family in raw_factorized_families
    )
    if validated_candidate_families:
        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in pool.candidates
        }
        projected_ids = []
        for family in validated_candidate_families:
            for candidate_id in family.variant_candidate_ids:
                candidate = candidates_by_id.get(candidate_id)
                if candidate is None:
                    raise CandidatePoolError(
                        f"candidate family {family.family_id!r} names missing "
                        f"variant {candidate_id!r}"
                    )
                if (
                    candidate.net != family.net
                    or candidate.topology_class != family.topology_class
                ):
                    raise CandidatePoolError(
                        f"candidate family {family.family_id!r} variant does "
                        "not match family net/topology"
                    )
                projected_ids.append(candidate_id)
        if (
            len(projected_ids) != len(set(projected_ids))
            or set(projected_ids) != set(candidates_by_id)
        ):
            raise CandidatePoolError(
                "candidate family projection must cover every variant exactly once"
            )
    option_templates, option_alias_count, option_template_digest = (
        landing_option_template_index(pool.candidates)
    )
    aliases_by_candidate: dict[str, list[dict[str, object]]] = {}
    if is_v12:
        for candidate in pool.candidates:
            aliases_by_candidate[candidate.candidate_id] = [
                landing_option_template_record(candidate, option)[2]
                for option in candidate.landing_options
            ]
    return {
        **(
            {
                "landing_site_catalog": (
                    pool.landing_site_catalog.canonical_payload()
                )
            }
            if pool.landing_site_catalog is not None
            else {}
        ),
        **(
            {
                "factorized_candidate_families": [
                    _factorized_candidate_family_payload(family)
                    for family in validated_factorized_families
                ]
            }
            if validated_factorized_families
            else {}
        ),
        **(
            {
                "candidate_families": [
                    {
                        "family_id": family.family_id,
                        "net": family.net,
                        "topology_class": family.topology_class,
                        "variant_candidate_ids": list(
                            family.variant_candidate_ids
                        ),
                    }
                    for family in validated_candidate_families
                ]
            }
            if validated_candidate_families
            else {}
        ),
        "candidates": [
            {
                "branches": [
                    {
                        "branch_id": branch.branch_id,
                        "ordered_landing_demand_ids": list(
                            branch.ordered_landing_demand_ids
                        ),
                        "ordered_steps": [
                            {"kind": step.kind, "ref_id": step.ref_id}
                            for step in branch.ordered_steps
                        ],
                        "ordered_use_ids": list(branch.ordered_use_ids),
                        "source_endpoint": branch.source_endpoint,
                        "target_endpoint": branch.target_endpoint,
                    }
                    for branch in candidate.branches
                ],
                "branch_edges": [list(edge) for edge in candidate.branch_edges],
                "candidate_id": candidate.candidate_id,
                "corridor_uses": [
                    {
                        "allowed_track_starts": list(use.allowed_track_starts),
                        "interval_nm": list(use.interval_nm),
                        "resource": _key_payload(use.resource),
                        "track_span": use.track_span,
                        "use_id": use.use_id,
                        "width_nm": _corridor_use_width_nm(
                            use, candidate.width_nm
                        ),
                    }
                    for use in candidate.corridor_uses
                ],
                "landing_demands": [
                    {
                        "allowed_site_ids": list(demand.allowed_site_ids),
                        "declaration_ref": demand.declaration_ref,
                        "demand_id": demand.demand_id,
                        "terminal_provenance": demand.terminal_provenance,
                        "required_multiplicity": demand.required_multiplicity,
                        "selection_mode": demand.selection_mode,
                    }
                    for demand in candidate.landing_demands
                ],
                **(
                    {
                        "landing_option_aliases": aliases_by_candidate[
                            candidate.candidate_id
                        ]
                    }
                    if is_v12
                    else {
                        "landing_options": [
                            {
                                "contacts": [
                                    {
                                        "site_id": contact.site_id,
                                        "track_choice": {
                                            "track_start": contact.track_start,
                                            "use_id": contact.use_id,
                                        },
                                    }
                                    for contact in option.contacts
                                ],
                                "demand_id": option.demand_id,
                                "option_id": option.option_id,
                                "site_ids": list(option.site_ids),
                            }
                            for option in candidate.landing_options
                        ]
                    }
                ),
                "endpoint_accesses": [
                    {
                        "layer_key": list(access.layer_key),
                        "layer_n": access.layer_n,
                        "owner_component_refs": list(
                            access.owner_component_refs
                        ),
                        "provenance": access.provenance,
                        "xy_nm": list(access.xy_nm),
                    }
                    for access in candidate.endpoint_accesses
                ],
                "goal_branch_coverage": [
                    {
                        "branch_id": item.branch_id,
                        "declaration_ref": item.declaration_ref,
                        "target_provenance": item.target_provenance,
                    }
                    for item in candidate.goal_branch_coverage
                ],
                "net": candidate.net,
                "program_kind": candidate.program_kind,
                "owned_component_use_refs": list(
                    candidate.owned_component_use_refs
                ),
                "required_goal_refs": list(candidate.required_goal_refs),
                "source_topology_digest": candidate.source_topology_digest,
                "topology_class": candidate.topology_class,
                "via_multiplicity": candidate.via_multiplicity,
                "width_nm": candidate.width_nm,
            }
            for candidate in pool.candidates
        ],
        "conflicts": [
            {
                "declaration_refs": list(conflict.declaration_refs),
                "left_candidate_id": conflict.left_candidate_id,
                "right_candidate_id": conflict.right_candidate_id,
            }
            for conflict in pool.conflicts
        ],
        "landing_sites": [
            {
                "capacity": site.capacity,
                "cut_bbox_nm": (
                    None if site.cut_bbox_nm is None else list(site.cut_bbox_nm)
                ),
                "lower_layer_n": site.lower_layer_n,
                "lower_pad_bbox_nm": (
                    None
                    if site.lower_pad_bbox_nm is None
                    else list(site.lower_pad_bbox_nm)
                ),
                "owner_net": site.owner_net,
                "owner_layer_ns": list(site.owner_layer_ns),
                "lower_owner_component_refs": list(
                    site.lower_owner_component_refs
                ),
                "upper_owner_component_refs": list(
                    site.upper_owner_component_refs
                ),
                "provenance": site.provenance,
                "routing_world_provenance_digest": (
                    site.routing_world_provenance_digest
                ),
                "site_id": site.site_id,
                "upper_layer_n": site.upper_layer_n,
                "upper_pad_bbox_nm": (
                    None
                    if site.upper_pad_bbox_nm is None
                    else list(site.upper_pad_bbox_nm)
                ),
                "xy_nm": list(site.xy_nm),
            }
            for site in pool.landing_sites
        ],
        "landing_site_conflicts": [
            {
                "conflict_scope": conflict.conflict_scope,
                "left_site_id": conflict.left_site_id,
                "right_site_id": conflict.right_site_id,
                "routing_world_provenance_digest": (
                    conflict.routing_world_provenance_digest
                ),
                "rule_refs": list(conflict.rule_refs),
            }
            for conflict in pool.landing_site_conflicts
        ],
        "landing_expansion_resources": [
            {
                "corridor_relation": item.corridor_relation,
                "declaration_refs": list(item.declaration_refs),
                "expansion_id": item.expansion_id,
                "legal_site_ids": list(item.legal_site_ids),
                "observed_clearance_nm": item.observed_clearance_nm,
                "owner_block_ids": list(item.owner_block_ids),
                "placement_basis_sha256": item.placement_basis_sha256,
                "routing_world_provenance_digest": (
                    item.routing_world_provenance_digest
                ),
                "run_interval_nm": list(item.run_interval_nm),
                "site_pitch_nm": item.site_pitch_nm,
            }
            for item in pool.landing_expansion_resources
        ],
        **(
            {}
            if pool.landing_site_conflict_domain is None
            else {
                "landing_site_conflict_domain": (
                    pool.landing_site_conflict_domain.canonical_payload()
                )
            }
        ),
        "landing_site_domain_digest": pool.landing_site_domain_digest,
        "landing_sites_complete": pool.landing_sites_complete,
        **(
            {
                "landing_option_alias_count": option_alias_count,
                "landing_option_template_count": len(option_templates),
                "landing_option_template_digest": option_template_digest,
                "landing_option_templates": [
                    {"template_id": template_id, **payload}
                    for template_id, payload in option_templates
                ],
            }
            if is_v12
            else {}
        ),
        "placement_provenance_digest": pool.placement_provenance_digest,
        "pdk_provenance_digest": pool.pdk_provenance_digest,
        "required_goal_refs": list(pool.required_goal_refs),
        "required_nets": list(pool.required_nets),
        "required_route_goals": [
            {
                "declaration_ref": goal.declaration_ref,
                "kind": goal.kind,
                "net": goal.net,
                "target": None if goal.target is None else {
                    "layer": list(goal.target.layer),
                    "provenance": goal.target.provenance,
                    "xy_nm": list(goal.target.xy_nm),
                },
            }
            for goal in pool.required_route_goals
        ],
        "resources": [
            {
                "corridor_bbox_nm": (
                    None
                    if resource.corridor_bbox_nm is None
                    else list(resource.corridor_bbox_nm)
                ),
                "key": _key_payload(resource.key),
                "gds_layer_key": (
                    None
                    if resource.gds_layer_key is None
                    else list(resource.gds_layer_key)
                ),
                "preferred_axis": resource.preferred_axis,
                "provenance": resource.provenance,
                "routing_world_provenance_digest": (
                    resource.routing_world_provenance_digest
                ),
                "min_space_nm": resource.min_space_nm,
                "through_interval_nm": (
                    None
                    if resource.through_interval_nm is None
                    else list(resource.through_interval_nm)
                ),
                "track_centers_nm": list(resource.track_centers_nm),
                **(
                    {"owner_net": resource.owner_net}
                    if resource.owner_net is not None
                    else {}
                ),
            }
            for resource in pool.resources
        ],
        "routing_world_provenance_digest": pool.routing_world_provenance_digest,
        "routing_resource_ledger_hash": pool.routing_resource_ledger_hash,
        "owned_conductor_index_digest": pool.owned_conductor_index_digest,
        "lane_incidence_replay_digest": pool.lane_incidence_replay_digest,
        "owned_conductor_components": [
            {
                "component": certificate.component.canonical_payload(),
                "layer_n": certificate.layer_n,
            }
            for certificate in pool.owned_conductor_components
        ],
        "schema": (
            "clay.complete-net-candidate-pool.v15"
            if pool.landing_site_catalog is not None
            else (
                "clay.complete-net-candidate-pool.v14"
                if pool.factorized_candidate_families
                else (
                    "clay.complete-net-candidate-pool.v13"
                    if pool.candidate_families
                    else "clay.complete-net-candidate-pool.v12"
                )
            )
            if pool.lane_incidence_replay_digest
            else (
                "clay.complete-net-candidate-pool.v9"
                if pool.landing_site_conflict_domain is None
                else "clay.complete-net-candidate-pool.v10"
            )
        ),
        "topology_provenance_digest": pool.topology_provenance_digest,
    }


def _route_plan_objective_certificate_payload(
    certificate: RoutePlanObjectiveCertificate,
) -> dict[str, object]:
    return {
        "authorized_replay_deterministic_time": (
            certificate.authorized_replay_deterministic_time
        ),
        "candidate_pool_hash": certificate.candidate_pool_hash,
        "canonical_decision_digest": certificate.canonical_decision_digest,
        "objective_policy_digest": certificate.objective_policy_digest,
        "phases": [
            {
                "lower_bound": phase.lower_bound,
                "phase_id": phase.phase_id,
                "status": phase.status,
                "value": phase.value,
            }
            for phase in certificate.phases
        ],
        "schema": certificate.schema,
    }


def _owned_conductor_index_certificate_payload(
    certificate: OwnedConductorIndexCertificate,
) -> dict[str, object]:
    return {
        "components": [
            {
                "component": item.component.canonical_payload(),
                "layer_n": item.layer_n,
            }
            for item in certificate.components
        ],
        "index_digest": certificate.index_digest,
        "schema": certificate.schema,
        "world_digest": certificate.world_digest,
    }


def _route_resource_plan_payload(plan: RouteResourcePlan) -> dict[str, object]:
    payload = {
        "allocated_programs": [
            _allocated_program_payload(program)
            for program in plan.allocated_programs
        ],
        "candidate_pool_hash": plan.candidate_pool_hash,
        "materialization_complete": plan.materialization_complete,
        "schema": plan.schema,
        "selected_nets": [
            {
                "candidate_id": selected.candidate_id,
                "net": selected.net,
                "selected_landings": [
                    {"demand_id": landing.demand_id, "site_id": landing.site_id}
                    for landing in selected.selected_landings
                ],
                "selected_landing_options": [
                    {
                        "demand_id": option.demand_id,
                        "option_id": option.option_id,
                    }
                    for option in selected.selected_landing_options
                ],
                "selected_tracks": [
                    {
                        "resource": _key_payload(track.resource),
                        "track_start": track.track_start,
                        "use_id": track.use_id,
                    }
                    for track in selected.selected_tracks
                ],
            }
            for selected in plan.selected_nets
        ],
    }
    if plan.schema in {
        "clay.route-resource-plan.v5",
        "clay.route-resource-plan.v7",
    }:
        payload["objective_certificate"] = (
            None
            if plan.objective_certificate is None
            else _route_plan_objective_certificate_payload(
                plan.objective_certificate
            )
        )
    elif plan.schema in {
        "clay.route-resource-plan.v6",
        "clay.route-resource-plan.v8",
    }:
        payload["objective_certificate"] = (
            None
            if plan.objective_certificate is None
            else _route_plan_objective_certificate_payload(
                plan.objective_certificate
            )
        )
        payload["landing_site_conflict_domain_digest"] = (
            plan.landing_site_conflict_domain_digest
        )
        payload["lazy_landing_site_conflicts"] = [
            {
                "conflict_scope": conflict.conflict_scope,
                "left_site_id": conflict.left_site_id,
                "right_site_id": conflict.right_site_id,
                "routing_world_provenance_digest": (
                    conflict.routing_world_provenance_digest
                ),
                "rule_refs": list(conflict.rule_refs),
            }
            for conflict in plan.lazy_landing_site_conflicts
        ]
    elif plan.schema in {
        "clay.route-resource-plan.v9",
        "clay.route-resource-plan.v10",
    }:
        payload["objective_certificate"] = (
            None
            if plan.objective_certificate is None
            else _route_plan_objective_certificate_payload(
                plan.objective_certificate
            )
        )
        payload["owned_conductor_index_certificate"] = (
            None
            if plan.owned_conductor_index_certificate is None
            else _owned_conductor_index_certificate_payload(
                plan.owned_conductor_index_certificate
            )
        )
        if plan.schema == "clay.route-resource-plan.v10":
            payload["landing_site_conflict_domain_digest"] = (
                plan.landing_site_conflict_domain_digest
            )
            payload["lazy_landing_site_conflicts"] = [
                {
                    "conflict_scope": conflict.conflict_scope,
                    "left_site_id": conflict.left_site_id,
                    "right_site_id": conflict.right_site_id,
                    "routing_world_provenance_digest": (
                        conflict.routing_world_provenance_digest
                    ),
                    "rule_refs": list(conflict.rule_refs),
                }
                for conflict in plan.lazy_landing_site_conflicts
            ]
    return payload


def _certificate_payload(certificate: Optional[object]) -> Optional[dict[str, object]]:
    if certificate is None:
        return None
    if isinstance(certificate, CapacityConflict):
        return {
            "candidate_ids": list(certificate.candidate_ids),
            "candidate_pool_hash": certificate.candidate_pool_hash,
            "declaration_refs": list(certificate.declaration_refs),
            "kind": "CAPACITY_CONFLICT",
            "nets": list(certificate.nets),
            "overlap_interval_nm": list(certificate.overlap_interval_nm),
            "resource": _key_payload(certificate.resource),
            "track_capacity": certificate.track_capacity,
        }
    if isinstance(certificate, HallDeficit):
        return {
            "candidate_ids": list(certificate.candidate_ids),
            "candidate_pool_hash": certificate.candidate_pool_hash,
            "declaration_refs": list(certificate.declaration_refs),
            "deficit": certificate.deficit,
            "demand_ids": list(certificate.demand_ids),
            "kind": "HALL_DEFICIT",
            "reachable_site_ids": list(certificate.reachable_site_ids),
            "expansion_resource_ids": list(certificate.expansion_resource_ids),
        }
    if isinstance(certificate, PoolUnsatCertificate):
        return {
            "assumption_core_ids": list(certificate.assumption_core_ids),
            "candidate_pool_hash": certificate.candidate_pool_hash,
            "kind": "POOL_UNSAT",
            "reason": certificate.reason,
        }
    if isinstance(certificate, RoutePlanObjectiveCertificate):
        return {
            "kind": "ROUTE_PLAN_OBJECTIVE",
            **_route_plan_objective_certificate_payload(certificate),
        }
    raise TypeError(f"unsupported certificate type {type(certificate).__name__}")


def _intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _iter_overlapping_track_use_pairs(
    candidates: tuple[CompleteNetCandidate, ...],
):
    """Yield each same-resource, half-open-overlapping use pair once.

    The sweep removes intervals ending at the next start before pairing, so
    ``[a, b)`` and ``[b, c)`` correctly share no derived conflict work.
    """

    by_resource: dict[
        CorridorLayerKey, list[tuple[CompleteNetCandidate, CorridorUse]]
    ] = {}
    for candidate in candidates:
        for use in candidate.corridor_uses:
            by_resource.setdefault(use.resource, []).append((candidate, use))
    def ordered_entries(entries):
        return tuple(sorted(
            entries,
            key=lambda entry: (
                entry[1].interval_nm[0],
                entry[1].interval_nm[1],
                entry[0].candidate_id,
                entry[1].use_id,
            ),
        ))

    for resource in sorted(by_resource):
        entries = by_resource[resource]
        by_candidate: dict[
            str, list[tuple[CompleteNetCandidate, CorridorUse]]
        ] = {}
        by_net: dict[
            str, list[tuple[CompleteNetCandidate, CorridorUse]]
        ] = {}
        for entry in entries:
            by_candidate.setdefault(entry[0].candidate_id, []).append(entry)
            by_net.setdefault(entry[0].net, []).append(entry)

        # Same-net alternatives are mutually exclusive.  Only uses inside the
        # same exact candidate can conflict on that net.
        for candidate_id in sorted(by_candidate):
            rows = ordered_entries(by_candidate[candidate_id])
            active_rows: list[
                tuple[CompleteNetCandidate, CorridorUse]
            ] = []
            for current in rows:
                start = current[1].interval_nm[0]
                active_rows = [
                    entry
                    for entry in active_rows
                    if entry[1].interval_nm[1] > start
                ]
                for previous in active_rows:
                    yield previous, current
                active_rows.append(current)

        # Cross-net pairs are the only alternative-candidate combinations that
        # can coexist.  A two-colour sweep never walks the large same-net active
        # set, which keeps a large profile family linear when it is the sole net.
        nets = tuple(sorted(by_net))
        for left_net_index, left_net in enumerate(nets):
            for right_net in nets[left_net_index + 1:]:
                events = tuple(sorted(
                    (
                        *((entry, 0) for entry in by_net[left_net]),
                        *((entry, 1) for entry in by_net[right_net]),
                    ),
                    key=lambda item: (
                        item[0][1].interval_nm[0],
                        item[0][1].interval_nm[1],
                        item[1],
                        item[0][0].candidate_id,
                        item[0][1].use_id,
                    ),
                ))
                active: list[list[tuple[CompleteNetCandidate, CorridorUse]]] = [
                    [], []
                ]
                for current, side in events:
                    opposite = 1 - side
                    start = current[1].interval_nm[0]
                    active[opposite] = [
                        entry
                        for entry in active[opposite]
                        if entry[1].interval_nm[1] > start
                    ]
                    for previous in active[opposite]:
                        yield previous, current
                    active[side].append(current)


def _closed_bbox_axis_gap_nm(
    left: BBoxNm, right: BBoxNm, *, axis: int
) -> int:
    low_index, high_index = (0, 2) if axis == 0 else (1, 3)
    return max(
        0,
        max(left[low_index], right[low_index])
        - min(left[high_index], right[high_index]),
    )


def _track_use_choice_union_bbox(
    use: CorridorUse, resource: TrackResource
) -> BBoxNm:
    if use.width_nm is None:
        raise CandidatePoolError(
            "derived track conflicts require canonical per-use widths"
        )
    choice_bboxes = tuple(
        bbox
        for track_start in use.allowed_track_starts
        for bbox in (
            _track_choice_closed_bbox(
                use, resource, track_start, use.width_nm
            ),
        )
        if bbox is not None
    )
    if len(choice_bboxes) != len(use.allowed_track_starts):
        raise CandidatePoolError(
            f"corridor use {use.use_id!r} has an invalid track choice"
        )
    return (
        min(bbox[0] for bbox in choice_bboxes),
        min(bbox[1] for bbox in choice_bboxes),
        max(bbox[2] for bbox in choice_bboxes),
        max(bbox[3] for bbox in choice_bboxes),
    )


def _iter_physical_track_use_pairs(
    candidates: tuple[CompleteNetCandidate, ...],
    resources_by_key: Mapping[CorridorLayerKey, TrackResource],
):
    """Yield every possibly conflicting use pair in one physical metal domain.

    Resource identities partition allocation capacity, not manufactured metal.
    A sweep over the union of each use's exact choice bboxes, expanded by the
    layer's largest spacing halo, covers same- and cross-resource side spacing,
    line-end spacing, and mixed-axis crossings without an all-pairs scan.
    """

    by_layer: dict[
        tuple[int, LayerKey],
        list[tuple[CompleteNetCandidate, CorridorUse, TrackResource, BBoxNm]],
    ] = {}
    for candidate in candidates:
        for use in candidate.corridor_uses:
            resource = resources_by_key[use.resource]
            if resource.gds_layer_key is None:
                continue
            by_layer.setdefault(
                (
                    resource.key.layer_n,
                    resource.gds_layer_key,
                ),
                [],
            ).append((
                candidate,
                use,
                resource,
                _track_use_choice_union_bbox(use, resource),
            ))
    comparison_count = 0

    def possibly_conflicts(previous, current) -> bool:
        nonlocal comparison_count
        comparison_count += 1
        if comparison_count > MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS:
            raise CandidatePoolLimitError(
                "physical track-use sweep comparison count "
                f"{comparison_count} exceeds "
                f"{MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS}"
            )
        required_space_nm = max(
            previous[2].min_space_nm, current[2].min_space_nm
        )
        x_gap_nm = _closed_bbox_axis_gap_nm(previous[3], current[3], axis=0)
        y_gap_nm = _closed_bbox_axis_gap_nm(previous[3], current[3], axis=1)
        return (
            (x_gap_nm == 0 and y_gap_nm == 0)
            or (
                required_space_nm > 0
                and x_gap_nm < required_space_nm
                and y_gap_nm < required_space_nm
            )
        )

    def ordered_entries(entries):
        return tuple(sorted(
            entries,
            key=lambda entry: (
                entry[3][0], entry[3][2], entry[0].candidate_id,
                entry[1].use_id,
            ),
        ))

    for layer_key in sorted(by_layer):
        entries = by_layer[layer_key]
        max_space_nm = max(entry[2].min_space_nm for entry in entries)
        by_candidate: dict[str, list[tuple]] = {}
        by_net: dict[str, list[tuple]] = {}
        for entry in entries:
            by_candidate.setdefault(entry[0].candidate_id, []).append(entry)
            by_net.setdefault(entry[0].net, []).append(entry)

        for candidate_id in sorted(by_candidate):
            rows = ordered_entries(by_candidate[candidate_id])
            active_rows: list[tuple] = []
            for current in rows:
                start = current[3][0]
                active_rows = [
                    entry
                    for entry in active_rows
                    if (
                        entry[3][2] + max_space_nm > start
                        or (
                            max_space_nm == 0
                            and entry[3][2] == start
                        )
                    )
                ]
                for previous in active_rows:
                    if possibly_conflicts(previous, current):
                        yield (previous[0], previous[1]), (
                            current[0], current[1]
                        )
                active_rows.append(current)

        nets = tuple(sorted(by_net))
        for left_net_index, left_net in enumerate(nets):
            for right_net in nets[left_net_index + 1:]:
                events = tuple(sorted(
                    (
                        *((entry, 0) for entry in by_net[left_net]),
                        *((entry, 1) for entry in by_net[right_net]),
                    ),
                    key=lambda item: (
                        item[0][3][0], item[0][3][2], item[1],
                        item[0][0].candidate_id, item[0][1].use_id,
                    ),
                ))
                active: list[list[tuple]] = [[], []]
                for current, side in events:
                    opposite = 1 - side
                    start = current[3][0]
                    active[opposite] = [
                        entry
                        for entry in active[opposite]
                        if (
                            entry[3][2] + max_space_nm > start
                            or (
                                max_space_nm == 0
                                and entry[3][2] == start
                            )
                        )
                    ]
                    for previous in active[opposite]:
                        if possibly_conflicts(previous, current):
                            yield (previous[0], previous[1]), (
                                current[0], current[1]
                            )
                    active[side].append(current)


def _same_resource_track_choices_overlap(
    left: CorridorUse,
    left_start: int,
    right: CorridorUse,
    right_start: int,
    resource: TrackResource,
) -> bool:
    if left.width_nm is None or right.width_nm is None:
        raise CandidatePoolError(
            "derived track conflicts require canonical per-use widths"
        )
    left_bbox = _track_choice_closed_bbox(
        left, resource, left_start, left.width_nm
    )
    right_bbox = _track_choice_closed_bbox(
        right, resource, right_start, right.width_nm
    )
    return (
        left_bbox is not None
        and right_bbox is not None
        and _closed_bboxes_touch(left_bbox, right_bbox)
    )


def _physical_track_choices_conflict(
    left: CorridorUse,
    left_start: int,
    left_resource: TrackResource,
    right: CorridorUse,
    right_start: int,
    right_resource: TrackResource,
    *,
    same_net: bool,
) -> bool:
    """Adjudicate exact bbox overlap and spacing on manufactured metal."""

    if left.width_nm is None or right.width_nm is None:
        raise CandidatePoolError(
            "derived track conflicts require canonical per-use widths"
        )
    if (
        left_resource.gds_layer_key is None
        or right_resource.gds_layer_key is None
        or left_resource.key.layer_n != right_resource.key.layer_n
        or left_resource.gds_layer_key != right_resource.gds_layer_key
    ):
        return False
    left_bbox = _track_choice_closed_bbox(
        left, left_resource, left_start, left.width_nm
    )
    right_bbox = _track_choice_closed_bbox(
        right, right_resource, right_start, right.width_nm
    )
    if left_bbox is None or right_bbox is None:
        return False
    x_gap_nm = _closed_bbox_axis_gap_nm(left_bbox, right_bbox, axis=0)
    y_gap_nm = _closed_bbox_axis_gap_nm(left_bbox, right_bbox, axis=1)
    touches = x_gap_nm == 0 and y_gap_nm == 0
    if touches:
        return not same_net
    required_space_nm = max(
        left_resource.min_space_nm, right_resource.min_space_nm
    )
    return (
        required_space_nm > 0
        and x_gap_nm < required_space_nm
        and y_gap_nm < required_space_nm
    )


def _incompatible_track_start_count(
    left: CorridorUse,
    right: CorridorUse,
    resource: TrackResource,
) -> int:
    """Count physical per-use-width overlap pairs on one resource."""

    return sum(
        _same_resource_track_choices_overlap(
            left, left_start, right, right_start, resource
        )
        for left_start in left.allowed_track_starts
        for right_start in right.allowed_track_starts
    )


def _validate_derived_track_conflict_bounds(
    candidates: tuple[CompleteNetCandidate, ...],
    resources_by_key: Mapping[CorridorLayerKey, TrackResource],
) -> None:
    evaluations = 0
    clauses = 0
    for (left_candidate, left), (
        right_candidate,
        right,
    ) in _iter_overlapping_track_use_pairs(candidates):
        if (
            left_candidate.net == right_candidate.net
            and left_candidate.candidate_id != right_candidate.candidate_id
        ):
            continue
        evaluations += len(left.allowed_track_starts) * len(
            right.allowed_track_starts
        )
        if evaluations > MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS:
            raise CandidatePoolLimitError(
                "derived track conflict evaluations "
                f"{evaluations} exceed {MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS}"
            )
        clauses += _incompatible_track_start_count(
            left, right, resources_by_key[left.resource]
        )
        if clauses > MAX_DERIVED_TRACK_CONFLICT_CLAUSES:
            raise CandidatePoolLimitError(
                f"derived track conflict clauses {clauses} exceed "
                f"{MAX_DERIVED_TRACK_CONFLICT_CLAUSES}"
            )
    for (left_candidate, left), (
        right_candidate,
        right,
    ) in _iter_physical_track_use_pairs(candidates, resources_by_key):
        if (
            left_candidate.net == right_candidate.net
            and left_candidate.candidate_id != right_candidate.candidate_id
        ):
            continue
        evaluations += len(left.allowed_track_starts) * len(
            right.allowed_track_starts
        )
        if evaluations > MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS:
            raise CandidatePoolLimitError(
                "derived track conflict evaluations "
                f"{evaluations} exceed {MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS}"
            )
        for left_start in left.allowed_track_starts:
            for right_start in right.allowed_track_starts:
                clauses += _physical_track_choices_conflict(
                    left,
                    left_start,
                    resources_by_key[left.resource],
                    right,
                    right_start,
                    resources_by_key[right.resource],
                    same_net=left_candidate.net == right_candidate.net,
                )
        if clauses > MAX_DERIVED_TRACK_CONFLICT_CLAUSES:
            raise CandidatePoolLimitError(
                f"derived track conflict clauses {clauses} exceed "
                f"{MAX_DERIVED_TRACK_CONFLICT_CLAUSES}"
            )


def _unavoidable_capacity_conflict(pool: CandidatePool) -> Optional[CapacityConflict]:
    """Prove only the simple exact case where every involved net has one candidate."""

    fixed_candidates = tuple(
        net_candidates[0]
        for net in pool.required_nets
        for net_candidates in (pool.candidates_for(net),)
        if len(net_candidates) == 1
    )
    uses_by_resource: dict[CorridorLayerKey, list[tuple[CompleteNetCandidate, CorridorUse]]] = {}
    for candidate in fixed_candidates:
        for use in candidate.corridor_uses:
            uses_by_resource.setdefault(use.resource, []).append((candidate, use))
    resources = {resource.key: resource for resource in pool.resources}
    pool_hash = pool.sha256()
    for key in sorted(uses_by_resource):
        entries = uses_by_resource[key]
        boundaries = sorted({point for _, use in entries for point in use.interval_nm})
        for start, end in zip(boundaries, boundaries[1:]):
            if start == end:
                continue
            active = tuple(sorted(
                (
                    (candidate, use)
                    for candidate, use in entries
                    if use.interval_nm[0] <= start and end <= use.interval_nm[1]
                ),
                key=lambda item: (item[0].net, item[0].candidate_id, item[1].use_id),
            ))
            if sum(use.track_span for _, use in active) > len(resources[key].track_centers_nm):
                return CapacityConflict(
                    candidate_pool_hash=pool_hash,
                    resource=key,
                    overlap_interval_nm=(start, end),
                    track_capacity=len(resources[key].track_centers_nm),
                    nets=tuple(candidate.net for candidate, _ in active),
                    candidate_ids=tuple(candidate.candidate_id for candidate, _ in active),
                    declaration_refs=tuple(sorted({
                        ref for candidate, _ in active for ref in candidate.required_goal_refs
                    })),
                )
    return None


def _landing_entries_matchable(
    entries: tuple[tuple[CompleteNetCandidate, LandingDemand], ...],
    capacities: Mapping[str, int],
) -> bool:
    """Return whether every demand has a site slot, using deterministic matching."""

    if not entries:
        return True
    slots_by_site = {
        site_id: tuple((site_id, slot) for slot in range(min(capacity, len(entries))))
        for site_id, capacity in sorted(capacities.items())
    }
    matched: dict[tuple[str, int], int] = {}

    def augment(entry_index: int, seen: set[tuple[str, int]]) -> bool:
        demand = entries[entry_index][1]
        for site_id in demand.allowed_site_ids:
            for slot in slots_by_site[site_id]:
                if slot in seen:
                    continue
                seen.add(slot)
                previous = matched.get(slot)
                if previous is None or augment(previous, seen):
                    matched[slot] = entry_index
                    return True
        return False

    return all(augment(index, set()) for index in range(len(entries)))


def _minimal_hall_entries(
    entries: tuple[tuple[CompleteNetCandidate, LandingDemand], ...],
    capacities: Mapping[str, int],
) -> tuple[tuple[CompleteNetCandidate, LandingDemand], ...]:
    """Find a deterministic inclusion-minimal non-matchable demand subset.

    If a proper deficient subset remains after removing a demand, the remaining
    graph is still non-matchable.  Repeated deterministic deletion therefore ends
    with a set for which every proper subset is matchable; by Hall's theorem that
    final set itself has a positive reachable-capacity deficit.
    """

    current = entries
    if _landing_entries_matchable(current, capacities):
        return ()
    changed = True
    while changed:
        changed = False
        for index in range(len(current)):
            smaller = current[:index] + current[index + 1 :]
            if smaller and not _landing_entries_matchable(smaller, capacities):
                current = smaller
                changed = True
                break
    return current


def _hall_entries_for_candidates(
    candidates: tuple[CompleteNetCandidate, ...],
    capacities: Mapping[str, int],
) -> tuple[tuple[CompleteNetCandidate, LandingDemand], ...]:
    entries = tuple(sorted(
        (
            (candidate, demand)
            for candidate in candidates
            for demand in candidate.landing_demands
        ),
        key=lambda entry: (
            entry[0].candidate_id,
            entry[1].demand_id,
            entry[1].allowed_site_ids,
        ),
    ))
    return _minimal_hall_entries(entries, capacities)


def _unavoidable_hall_deficit(pool: CandidatePool) -> Optional[HallDeficit]:
    fixed_candidates = tuple(
        net_candidates[0]
        for net in pool.required_nets
        for net_candidates in (pool.candidates_for(net),)
        if len(net_candidates) == 1
    )
    capacities = {
        site.site_id: site.capacity for site in pool.iter_landing_sites()
    }
    entries = _hall_entries_for_candidates(fixed_candidates, capacities)
    if not entries:
        return None
    reachable_sites = tuple(sorted({
        site_id for _, demand in entries for site_id in demand.allowed_site_ids
    }))
    deficit = len(entries) - sum(capacities[site_id] for site_id in reachable_sites)
    if deficit <= 0:  # defensive: a minimal non-matchable graph must violate Hall
        raise AssertionError("minimal non-matchable landing graph has no Hall deficit")
    return HallDeficit(
        candidate_pool_hash=pool.sha256(),
        demand_ids=tuple(demand.demand_id for _, demand in entries),
        reachable_site_ids=reachable_sites,
        deficit=deficit,
        candidate_ids=tuple(sorted({
            candidate.candidate_id for candidate, _ in entries
        })),
        declaration_refs=tuple(sorted({
            demand.declaration_ref
            for _, demand in entries
            if demand.declaration_ref is not None
        })),
        expansion_resource_ids=tuple(sorted(
            item.expansion_id
            for item in pool.landing_expansion_resources
            if tuple(item.legal_site_ids) == reachable_sites
        )),
    )


def _load_cp_model():
    """Narrow import seam used by tests; no fallback solver is permitted."""

    from ortools.sat.python import cp_model

    return cp_model


def _build_candidate_cp_model(
    pool: CandidatePool,
    cp_model: object,
    *,
    assumed_core_ids: Optional[tuple[str, ...]] = None,
    additional_landing_site_conflicts: Sequence[
        LandingSitePairConflict
    ] = (),
):
    """Build the bounded model shared by selection and fresh core replay.

    Required-net lower bounds are the only assumptions.  At-most-one remains a
    permanent constraint, so a replay of a sufficient subset can disable every
    non-core net cleanly while retaining all resource and pair constraints.
    """

    first_closed_candidate = next(
        (
            candidate for candidate in pool.all_candidates
            if candidate_has_closed_allocation_ir(candidate)
        ),
        None,
    )
    if first_closed_candidate is not None:
        verify_candidate_allocation_ir(pool, first_closed_candidate)

    model = cp_model.CpModel()
    candidate_vars = {
        candidate.candidate_id: model.NewBoolVar(f"candidate::{candidate.candidate_id}")
        for candidate in pool.all_candidates
    }
    enabled_assumptions = (
        {f"required_net:{net}" for net in pool.required_nets}
        if assumed_core_ids is None
        else set(assumed_core_ids)
    )
    assumption_index_to_id: dict[int, str] = {}
    for net in pool.required_nets:
        net_vars = [
            candidate_vars[candidate.candidate_id]
            for candidate in pool.candidates_for(net)
        ]
        model.AddAtMostOne(net_vars)
        core_id = f"required_net:{net}"
        required = model.NewBoolVar(f"assumption::{core_id}")
        model.Add(sum(net_vars) >= 1).OnlyEnforceIf(required)
        assumption_index_to_id[required.Index()] = core_id
        if core_id in enabled_assumptions:
            model.AddAssumption(required)
        else:
            model.Add(required == 0)

    site_capacities = {
        site.site_id: site.capacity for site in pool.iter_landing_sites()
    }
    for candidate in pool.all_candidates:
        if _hall_entries_for_candidates((candidate,), site_capacities):
            model.Add(candidate_vars[candidate.candidate_id] == 0)

    track_vars: dict[tuple[str, str, int], object] = {}
    resources_by_key = {resource.key: resource for resource in pool.resources}
    landing_sites_by_id = pool.landing_sites_by_id()
    component_certificates_by_ref = {
        certificate.component.component_ref: certificate
        for certificate in pool.owned_conductor_components
    }
    for candidate in pool.all_candidates:
        selected = candidate_vars[candidate.candidate_id]
        for use in candidate.corridor_uses:
            choices = []
            for start in use.allowed_track_starts:
                var = model.NewBoolVar(
                    f"track::{candidate.candidate_id}::{use.use_id}::{start}"
                )
                track_vars[(candidate.candidate_id, use.use_id, start)] = var
                choices.append(var)
            model.Add(sum(choices) == selected)
        if candidate_has_closed_allocation_ir(candidate):
            uses_by_id = {
                use.use_id: use for use in candidate.corridor_uses
            }
            for access, use_id in _endpoint_track_requirements(candidate):
                use = uses_by_id[use_id]
                resource = resources_by_key[use.resource]
                for start in use.allowed_track_starts:
                    if not _track_choice_contains_endpoint(
                        access, use, resource, start
                    ):
                        model.Add(
                            track_vars[(candidate.candidate_id, use_id, start)] == 0
                        )
            for left_use, right_use in _consecutive_track_step_pairs(candidate):
                left_resource = resources_by_key[left_use.resource]
                right_resource = resources_by_key[right_use.resource]
                for left_start in left_use.allowed_track_starts:
                    for right_start in right_use.allowed_track_starts:
                        if _track_choices_physically_touch(
                            left_use,
                            left_resource,
                            left_start,
                            right_use,
                            right_resource,
                            right_start,
                            _corridor_use_width_nm(
                                left_use, candidate.width_nm
                            ),
                            _corridor_use_width_nm(
                                right_use, candidate.width_nm
                            ),
                        ):
                            continue
                        model.Add(
                            track_vars[(
                                candidate.candidate_id,
                                left_use.use_id,
                                left_start,
                            )]
                            + track_vars[(
                                candidate.candidate_id,
                                right_use.use_id,
                                right_start,
                            )]
                            <= 1
                        )
            for component_ref, use in _component_track_step_pairs(candidate):
                certificate = component_certificates_by_ref[component_ref]
                component = certificate.component
                resource = resources_by_key[use.resource]
                for track_start in use.allowed_track_starts:
                    track_bbox = _track_choice_closed_bbox(
                        use,
                        resource,
                        track_start,
                        _corridor_use_width_nm(use, candidate.width_nm),
                    )
                    if (
                        certificate.layer_n == resource.key.layer_n
                        and component.layer_key == resource.gds_layer_key
                        and track_bbox is not None
                        and _owned_component_conductor_touches_track_bbox(
                            component, track_bbox
                        )
                    ):
                        continue
                    model.Add(
                        track_vars[(
                            candidate.candidate_id,
                            use.use_id,
                            track_start,
                        )]
                        == 0
                    )

    for (left_candidate, left_use), (
        right_candidate,
        right_use,
    ) in _iter_overlapping_track_use_pairs(pool.all_candidates):
        if (
            left_candidate.net == right_candidate.net
            and left_candidate.candidate_id != right_candidate.candidate_id
        ):
            continue
        for left_start in left_use.allowed_track_starts:
            for right_start in right_use.allowed_track_starts:
                if _same_resource_track_choices_overlap(
                    left_use,
                    left_start,
                    right_use,
                    right_start,
                    resources_by_key[left_use.resource],
                ):
                    model.Add(
                        track_vars[(
                            left_candidate.candidate_id,
                            left_use.use_id,
                            left_start,
                        )]
                        + track_vars[(
                            right_candidate.candidate_id,
                            right_use.use_id,
                            right_start,
                        )]
                        <= 1
                    )

    for (left_candidate, left_use), (
        right_candidate,
        right_use,
    ) in _iter_physical_track_use_pairs(
        pool.all_candidates, resources_by_key
    ):
        if (
            left_candidate.net == right_candidate.net
            and left_candidate.candidate_id != right_candidate.candidate_id
        ):
            continue
        for left_start in left_use.allowed_track_starts:
            for right_start in right_use.allowed_track_starts:
                if _physical_track_choices_conflict(
                    left_use,
                    left_start,
                    resources_by_key[left_use.resource],
                    right_use,
                    right_start,
                    resources_by_key[right_use.resource],
                    same_net=left_candidate.net == right_candidate.net,
                ):
                    model.Add(
                        track_vars[(
                            left_candidate.candidate_id,
                            left_use.use_id,
                            left_start,
                        )]
                        + track_vars[(
                            right_candidate.candidate_id,
                            right_use.use_id,
                            right_start,
                        )]
                        <= 1
                    )

    landing_vars: dict[tuple[str, str, str], object] = {}
    landing_option_vars: dict[tuple[str, str, str], object] = {}
    site_assignments: dict[str, list[object]] = {
        site.site_id: [] for site in pool.iter_landing_sites()
    }
    site_net_assignments: dict[tuple[str, str], list[object]] = {}
    site_candidate_assignments: dict[
        str, list[tuple[CompleteNetCandidate, object]]
    ] = {site.site_id: [] for site in pool.iter_landing_sites()}
    for candidate in pool.all_candidates:
        selected = candidate_vars[candidate.candidate_id]
        if candidate_has_closed_allocation_ir(candidate):
            uses_by_id = {
                use.use_id: use for use in candidate.corridor_uses
            }
            neighbour_use_ids_by_demand = _landing_neighbour_use_ids(candidate)
            options_by_demand: dict[str, list[LandingOption]] = {
                demand.demand_id: [] for demand in candidate.landing_demands
            }
            for option in candidate.landing_options:
                options_by_demand[option.demand_id].append(option)
            for demand in candidate.landing_demands:
                if (
                    demand.selection_mode
                    == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                ):
                    choices = []
                    for site_id in demand.allowed_site_ids:
                        site = landing_sites_by_id[site_id]
                        var = model.NewBoolVar(
                            f"derived-landing::{candidate.candidate_id}::"
                            f"{demand.demand_id}::{site_id}"
                        )
                        landing_vars[(
                            candidate.candidate_id,
                            demand.demand_id,
                            site_id,
                        )] = var
                        site_assignments[site_id].append(var)
                        site_candidate_assignments[site_id].append(
                            (candidate, var)
                        )
                        site_net_assignments.setdefault(
                            (site_id, candidate.net), []
                        ).append(var)
                        for use_id in neighbour_use_ids_by_demand[
                            demand.demand_id
                        ]:
                            use = uses_by_id[use_id]
                            resource = resources_by_key[use.resource]
                            compatible_track_vars = [
                                track_vars[(
                                    candidate.candidate_id,
                                    use_id,
                                    track_start,
                                )]
                                for track_start in use.allowed_track_starts
                                if _track_choice_crosses_landing_site(
                                    use, resource, track_start, site
                                )
                            ]
                            model.Add(var <= sum(compatible_track_vars))
                        choices.append(var)
                    model.Add(
                        sum(choices)
                        == demand.required_multiplicity * selected
                    )
                    continue
                choices = []
                for option in options_by_demand[demand.demand_id]:
                    var = model.NewBoolVar(
                        f"landing-option::{candidate.candidate_id}::"
                        f"{demand.demand_id}::{option.option_id}"
                    )
                    landing_option_vars[(
                        candidate.candidate_id,
                        demand.demand_id,
                        option.option_id,
                    )] = var
                    for site_id in option.site_ids:
                        site_assignments[site_id].append(var)
                        site_candidate_assignments[site_id].append(
                            (candidate, var)
                        )
                        site_net_assignments.setdefault(
                            (site_id, candidate.net), []
                        ).append(var)
                    for contact in option.contacts:
                        model.Add(
                            var
                            <= track_vars[(
                                candidate.candidate_id,
                                contact.use_id,
                                contact.track_start,
                            )]
                        )
                    for use_id in neighbour_use_ids_by_demand[demand.demand_id]:
                        use = uses_by_id[use_id]
                        resource = resources_by_key[use.resource]
                        compatible_track_vars = [
                            track_vars[(
                                candidate.candidate_id,
                                use_id,
                                track_start,
                            )]
                            for track_start in use.allowed_track_starts
                            if _track_choice_touches_landing_option(
                                candidate=candidate,
                                option=option,
                                use=use,
                                resource=resource,
                                track_start=track_start,
                                sites_by_id=landing_sites_by_id,
                                certificates_by_ref=component_certificates_by_ref,
                            )
                        ]
                        model.Add(var <= sum(compatible_track_vars))
                    choices.append(var)
                required_option_count = (
                    demand.required_multiplicity
                    if demand.selection_mode
                    == LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY
                    else 1
                )
                model.Add(
                    sum(choices) == required_option_count * selected
                )
            continue
        for demand in candidate.landing_demands:
            choices = []
            for site_id in demand.allowed_site_ids:
                var = model.NewBoolVar(
                    f"landing::{candidate.candidate_id}::{demand.demand_id}::{site_id}"
                )
                landing_vars[(candidate.candidate_id, demand.demand_id, site_id)] = var
                site_assignments[site_id].append(var)
                site_candidate_assignments[site_id].append((candidate, var))
                site_net_assignments.setdefault((site_id, candidate.net), []).append(
                    var
                )
                choices.append(var)
            model.Add(sum(choices) == selected)
    site_used: dict[str, object] = {}
    for site in pool.iter_landing_sites():
        assignments = site_assignments[site.site_id]
        model.Add(sum(assignments) <= site.capacity)
        used = model.NewBoolVar(f"landing-site-used::{site.site_id}")
        site_used[site.site_id] = used
        model.Add(sum(assignments) >= used)
        model.Add(sum(assignments) <= site.capacity * used)

    site_net_used: dict[tuple[str, str], object] = {}
    site_net_keys_by_site: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(site_net_assignments):
        assignments = site_net_assignments[key]
        used = model.NewBoolVar(f"landing-site-net-used::{key[0]}::{key[1]}")
        site_net_used[key] = used
        site_net_keys_by_site.setdefault(key[0], []).append(key)
        model.Add(sum(assignments) >= used)
        model.Add(sum(assignments) <= len(assignments) * used)

    additional_conflicts = tuple(sorted(_bounded_materialize(
        additional_landing_site_conflicts,
        limit=MAX_LAZY_LANDING_SITE_CONFLICTS,
        label="additional lazy landing site conflict",
        error_type=CandidatePoolLimitError,
    )))
    if not all(
        isinstance(conflict, LandingSitePairConflict)
        for conflict in additional_conflicts
    ):
        raise CandidatePoolError(
            "additional lazy landing conflicts have invalid values"
        )
    pool_conflict_set = set(pool.landing_site_conflicts)
    lazy_conflict_set = set(additional_conflicts) - pool_conflict_set
    all_landing_conflicts = tuple(sorted(
        pool_conflict_set | set(additional_conflicts)
    ))
    lazy_clause_count = 0

    def charge_lazy_clause(conflict: LandingSitePairConflict) -> None:
        nonlocal lazy_clause_count
        if conflict not in lazy_conflict_set:
            return
        lazy_clause_count += 1
        if lazy_clause_count > MAX_LANDING_CONFLICT_NET_CLAUSES:
            raise CandidatePoolLimitError(
                "derived lazy landing conflict clauses exceed "
                f"{MAX_LANDING_CONFLICT_NET_CLAUSES}"
            )

    for conflict in all_landing_conflicts:
        if conflict.conflict_scope == "all_nets":
            charge_lazy_clause(conflict)
            model.Add(
                site_used[conflict.left_site_id]
                + site_used[conflict.right_site_id]
                <= 1
            )
            continue
        left_keys = site_net_keys_by_site.get(conflict.left_site_id, ())
        right_keys = site_net_keys_by_site.get(conflict.right_site_id, ())
        for left_key in left_keys:
            for right_key in right_keys:
                if left_key[1] != right_key[1]:
                    charge_lazy_clause(conflict)
                    model.Add(
                        site_net_used[left_key] + site_net_used[right_key] <= 1
                    )
        if conflict.conflict_scope != "same_net_bridge":
            continue
        left_site = landing_sites_by_id[conflict.left_site_id]
        right_site = landing_sites_by_id[conflict.right_site_id]
        right_assignments_by_candidate: dict[str, list[object]] = {}
        for candidate, var in site_candidate_assignments[
            conflict.right_site_id
        ]:
            right_assignments_by_candidate.setdefault(
                candidate.candidate_id, []
            ).append(var)
        for candidate, left_var in site_candidate_assignments[
            conflict.left_site_id
        ]:
            right_vars = right_assignments_by_candidate.get(
                candidate.candidate_id, ()
            )
            if not right_vars:
                continue
            bridge_vars = []
            for use in candidate.corridor_uses:
                resource = resources_by_key[use.resource]
                for track_start in use.allowed_track_starts:
                    if _track_choice_bridges_landing_pair(
                        candidate,
                        use,
                        resource,
                        track_start,
                        left_site,
                        right_site,
                    ):
                        bridge_vars.append(track_vars[(
                            candidate.candidate_id,
                            use.use_id,
                            track_start,
                        )])
            for right_var in right_vars:
                charge_lazy_clause(conflict)
                model.Add(
                    left_var + right_var <= 1 + sum(bridge_vars)
                )

    for pair in pool.conflicts:
        model.Add(
            candidate_vars[pair.left_candidate_id]
            + candidate_vars[pair.right_candidate_id]
            <= 1
        )

    return (
        model,
        candidate_vars,
        track_vars,
        landing_vars,
        landing_option_vars,
        assumption_index_to_id,
        site_used,
    )


def _bbox_area_nm2(bbox: Optional[BBoxNm]) -> int:
    if bbox is None:
        return 0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width < 0 or height < 0:
        raise CandidatePoolError("objective bbox has negative extent")
    area = width * height
    if area > CP_SAT_INT64_MAX:
        raise CandidatePoolLimitError(
            "route-plan objective bbox area exceeds CP-SAT int64"
        )
    return area


def _route_plan_objective_coefficients(
    pool: CandidatePool,
) -> tuple[
    dict[tuple[str, str, int], int],
    dict[str, int],
]:
    """Return exact positive integer coefficients and preflight their sum."""

    resources_by_key = {resource.key: resource for resource in pool.resources}
    track_area: dict[tuple[str, str, int], int] = {}
    upper_bound = 0
    for candidate in pool.all_candidates:
        for use in candidate.corridor_uses:
            resource = resources_by_key[use.resource]
            width_nm = _corridor_use_width_nm(use, candidate.width_nm)
            for track_start in use.allowed_track_starts:
                bbox = _track_choice_closed_bbox(
                    use, resource, track_start, width_nm
                )
                if bbox is None:
                    raise CandidatePoolError(
                        f"cannot price exact track geometry for {candidate.candidate_id!r} "
                        f"use {use.use_id!r}"
                    )
                area = _bbox_area_nm2(bbox)
                key = (candidate.candidate_id, use.use_id, track_start)
                track_area[key] = area
                upper_bound += area
                if upper_bound > CP_SAT_INT64_MAX:
                    raise CandidatePoolLimitError(
                        "route-plan track-area objective upper bound exceeds CP-SAT int64"
                    )
    site_area: dict[str, int] = {}
    referenced_site_ids = {
        site_id
        for candidate in pool.all_candidates
        for demand in candidate.landing_demands
        for site_id in demand.allowed_site_ids
    }
    for site in pool.iter_landing_sites():
        if site.site_id in referenced_site_ids and any(
            bbox is None
            for bbox in (
                site.lower_pad_bbox_nm,
                site.upper_pad_bbox_nm,
                site.cut_bbox_nm,
            )
        ):
            raise RoutePlanObjectiveGeometryError(
                f"referenced landing site {site.site_id!r} lacks exact pad/cut geometry"
            )
        area = sum(_bbox_area_nm2(bbox) for bbox in (
            site.lower_pad_bbox_nm,
            site.upper_pad_bbox_nm,
            site.cut_bbox_nm,
        ))
        site_area[site.site_id] = area
        upper_bound += area
        if upper_bound > CP_SAT_INT64_MAX:
            raise CandidatePoolLimitError(
                "route-plan geometry-area objective upper bound exceeds CP-SAT int64"
            )
    if len(site_area) > CP_SAT_INT64_MAX:
        raise CandidatePoolLimitError(
            "route-plan landing count objective exceeds CP-SAT int64"
        )
    return track_area, site_area


def _selected_decision_payload(
    selected_nets: tuple[SelectedNetCandidate, ...],
) -> dict[str, object]:
    # The solver reports selected candidates in CandidatePool traversal order
    # (candidate id), while RouteResourcePlan stores them in its canonical
    # value order (net, candidate id).  Sign the latter so construction and
    # independent replay hash the same semantic decision for multi-net plans.
    canonical_selected_nets = tuple(sorted(
        selected_nets,
        key=lambda selected: (selected.net, selected.candidate_id),
    ))
    return {
        "schema": "clay.route-plan-canonical-decision.v1",
        "selected_nets": [
            {
                "candidate_id": selected.candidate_id,
                "net": selected.net,
                "landing_options": [
                    [option.demand_id, option.option_id]
                    for option in selected.selected_landing_options
                ],
                "landings": [
                    [landing.demand_id, landing.site_id]
                    for landing in selected.selected_landings
                ],
                "tracks": [
                    [
                        track.use_id,
                        track.resource.corridor_id,
                        track.resource.layer_n,
                        track.track_start,
                    ]
                    for track in selected.selected_tracks
                ],
            }
            for selected in canonical_selected_nets
        ],
    }


def _canonical_decision_digest(
    selected_nets: tuple[SelectedNetCandidate, ...],
) -> str:
    return hashlib.sha256(json.dumps(
        _selected_decision_payload(selected_nets),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _selected_objective_values(
    pool: CandidatePool,
    selected_nets: tuple[SelectedNetCandidate, ...],
) -> tuple[int, int]:
    track_area, site_area = _route_plan_objective_coefficients(pool)
    area = sum(
        track_area[(selected.candidate_id, track.use_id, track.track_start)]
        for selected in selected_nets
        for track in selected.selected_tracks
    )
    used_sites = {
        landing.site_id
        for selected in selected_nets
        for landing in selected.selected_landings
    }
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in pool.all_candidates
    }
    options_by_candidate = {
        candidate.candidate_id: {
            option.option_id: option for option in candidate.landing_options
        }
        for candidate in pool.all_candidates
    }
    for selected in selected_nets:
        if not candidate_has_closed_allocation_ir(
            candidates_by_id[selected.candidate_id]
        ):
            continue
        used_sites.update(
            site_id
            for chosen in selected.selected_landing_options
            for site_id in options_by_candidate[selected.candidate_id][
                chosen.option_id
            ].site_ids
        )
    area += sum(site_area[site_id] for site_id in used_sites)
    if area > CP_SAT_INT64_MAX:
        raise CandidatePoolLimitError(
            "selected route-plan objective exceeds CP-SAT int64"
        )
    return area, len(used_sites)


def _objective_certificate(
    pool_hash: str,
    selected_nets: tuple[SelectedNetCandidate, ...],
    phases: tuple[RoutePlanObjectivePhase, ...],
    *,
    authorized_replay_deterministic_time: float,
) -> RoutePlanObjectiveCertificate:
    return RoutePlanObjectiveCertificate(
        candidate_pool_hash=pool_hash,
        objective_policy_digest=route_plan_objective_policy_digest(),
        phases=phases,
        canonical_decision_digest=_canonical_decision_digest(selected_nets),
        authorized_replay_deterministic_time=(
            authorized_replay_deterministic_time
        ),
    )


def _selected_nets_from_solver(
    pool: CandidatePool,
    solver: object,
    candidate_vars: Mapping[str, object],
    track_vars: Mapping[tuple[str, str, int], object],
    landing_vars: Mapping[tuple[str, str, str], object],
    landing_option_vars: Mapping[tuple[str, str, str], object],
) -> tuple[SelectedNetCandidate, ...]:
    selected_nets = []
    for candidate in pool.all_candidates:
        if not solver.BooleanValue(candidate_vars[candidate.candidate_id]):
            continue
        selected_tracks = tuple(
            SelectedTrack(
                use_id=use.use_id,
                resource=use.resource,
                track_start=start,
            )
            for use in candidate.corridor_uses
            for start in use.allowed_track_starts
            if solver.BooleanValue(
                track_vars[(candidate.candidate_id, use.use_id, start)]
            )
        )
        selected_landings = tuple(
                SelectedLanding(
                    demand_id=demand.demand_id, site_id=site_id
                )
                for demand in candidate.landing_demands
                if (
                    not candidate_has_closed_allocation_ir(candidate)
                    or demand.selection_mode
                    == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                )
                for site_id in demand.allowed_site_ids
                if solver.BooleanValue(
                    landing_vars[(
                        candidate.candidate_id,
                        demand.demand_id,
                        site_id,
                    )]
                )
        )
        selected_landing_options = tuple(
            SelectedLandingOption(
                demand_id=option.demand_id,
                option_id=option.option_id,
            )
            for option in candidate.landing_options
            if solver.BooleanValue(
                landing_option_vars[(
                    candidate.candidate_id,
                    option.demand_id,
                    option.option_id,
                )]
            )
        )
        selected_nets.append(SelectedNetCandidate(
            net=candidate.net,
            candidate_id=candidate.candidate_id,
            selected_tracks=selected_tracks,
            selected_landings=selected_landings,
            selected_landing_options=selected_landing_options,
        ))
    return tuple(sorted(
        selected_nets,
        key=lambda selected: (selected.net, selected.candidate_id),
    ))


def _owned_conductor_index_certificate_from_pool(
    pool: CandidatePool,
) -> Optional[OwnedConductorIndexCertificate]:
    has_world = bool(pool.routing_world_provenance_digest)
    has_index = bool(pool.owned_conductor_index_digest)
    if not (has_world and has_index):
        return None
    return OwnedConductorIndexCertificate(
        world_digest=pool.routing_world_provenance_digest,
        components=pool.owned_conductor_components,
        index_digest=pool.owned_conductor_index_digest,
    )


def _plan_from_selected_nets(
    pool: CandidatePool,
    selected_nets: tuple[SelectedNetCandidate, ...],
    certificate: RoutePlanObjectiveCertificate,
    lazy_landing_site_conflicts: tuple[
        LandingSitePairConflict, ...
    ] = (),
) -> RouteResourcePlan:
    pool_hash = pool.sha256()
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in pool.all_candidates
    }
    allocated_programs = tuple(
        _resolve_allocated_net_program(
            pool,
            candidates_by_id[selected.candidate_id],
            selected,
            pool_hash,
        )
        for selected in selected_nets
        if candidate_has_closed_allocation_ir(
            candidates_by_id[selected.candidate_id]
        )
    )
    materialization_complete = all(
        candidate_has_closed_allocation_ir(
            candidates_by_id[selected.candidate_id]
        )
        for selected in selected_nets
    )
    return RouteResourcePlan(
        candidate_pool_hash=pool_hash,
        selected_nets=selected_nets,
        allocated_programs=allocated_programs,
        materialization_complete=materialization_complete,
        objective_certificate=certificate,
        owned_conductor_index_certificate=(
            _owned_conductor_index_certificate_from_pool(pool)
        ),
        landing_site_conflict_domain_digest=(
            ""
            if pool.landing_site_conflict_domain is None
            else pool.landing_site_conflict_domain.sha256()
        ),
        lazy_landing_site_conflicts=lazy_landing_site_conflicts,
        schema=(
            "clay.route-resource-plan.v9"
            if pool.landing_site_conflict_domain is None
            else "clay.route-resource-plan.v10"
        ),
    )


def _selected_domain_landing_conflicts(
    pool: CandidatePool,
    selected_nets: Sequence[SelectedNetCandidate],
) -> tuple[LandingSitePairConflict, ...]:
    """Return exact domain conflicts violated by one finite selected assignment."""

    domain = pool.landing_site_conflict_domain
    if domain is None:
        return ()
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in pool.all_candidates
    }
    sites_by_id = pool.landing_sites_by_id()
    resources_by_key = {resource.key: resource for resource in pool.resources}
    selected_nets_by_site: dict[str, set[str]] = {}
    selected_candidates_by_site: dict[str, set[str]] = {}
    selected_track_by_use: dict[tuple[str, str], SelectedTrack] = {}

    for selected in selected_nets:
        candidate = candidates_by_id[selected.candidate_id]
        for track in selected.selected_tracks:
            selected_track_by_use[(candidate.candidate_id, track.use_id)] = track
        if candidate_has_closed_allocation_ir(candidate):
            options_by_id = {
                option.option_id: option for option in candidate.landing_options
            }
            selected_site_ids = {
                site_id
                for selected_option in selected.selected_landing_options
                for site_id in options_by_id[selected_option.option_id].site_ids
            } | {landing.site_id for landing in selected.selected_landings}
        else:
            selected_site_ids = {
                landing.site_id for landing in selected.selected_landings
            }
        for site_id in selected_site_ids:
            selected_nets_by_site.setdefault(site_id, set()).add(candidate.net)
            selected_candidates_by_site.setdefault(site_id, set()).add(
                candidate.candidate_id
            )

    selected_site_ids = tuple(sorted(selected_nets_by_site))
    conflicts: list[LandingSitePairConflict] = []
    pair_evaluations = 0
    for right_index, right_site_id in enumerate(selected_site_ids):
        for left_site_id in selected_site_ids[:right_index]:
            pair_evaluations += 1
            if (
                pair_evaluations
                > MAX_LAZY_LANDING_SELECTED_PAIR_EVALUATIONS
            ):
                raise CandidatePoolLimitError(
                    "selected landing conflict pair evaluations exceed "
                    f"{MAX_LAZY_LANDING_SELECTED_PAIR_EVALUATIONS}"
                )
            conflict = _derive_landing_site_pair_conflict_from_domain(
                sites_by_id[left_site_id],
                sites_by_id[right_site_id],
                domain,
            )
            if conflict is None:
                continue
            left_nets = selected_nets_by_site[left_site_id]
            right_nets = selected_nets_by_site[right_site_id]
            if conflict.conflict_scope == "all_nets" or any(
                left_net != right_net
                for left_net in left_nets
                for right_net in right_nets
            ):
                conflicts.append(conflict)
                continue
            if conflict.conflict_scope != "same_net_bridge":
                continue
            common_candidate_ids = (
                selected_candidates_by_site[left_site_id]
                & selected_candidates_by_site[right_site_id]
            )
            left_site = sites_by_id[left_site_id]
            right_site = sites_by_id[right_site_id]
            bridged = any(
                _track_choice_bridges_landing_pair(
                    candidates_by_id[candidate_id],
                    use,
                    resources_by_key[use.resource],
                    selected_track_by_use[(candidate_id, use.use_id)].track_start,
                    left_site,
                    right_site,
                )
                for candidate_id in sorted(common_candidate_ids)
                for use in candidates_by_id[candidate_id].corridor_uses
            )
            if not bridged:
                conflicts.append(conflict)
    return tuple(sorted(conflicts))


def _select_complete_net_candidates_once(
    pool: CandidatePool,
    *,
    max_deterministic_time: float = (
        ROUTE_PLAN_OBJECTIVE_DEFAULT_DETERMINISTIC_TIME
    ),
    additional_landing_site_conflicts: tuple[
        LandingSitePairConflict, ...
    ] = (),
    budget_account: Optional[list[float]] = None,
) -> CompleteNetSelectionResult:
    if not isinstance(pool, CandidatePool):
        raise TypeError("pool must be CandidatePool")
    if isinstance(max_deterministic_time, bool) or not isinstance(
        max_deterministic_time, (int, float)
    ):
        raise ValueError("max_deterministic_time must be a non-negative number")
    budget = float(max_deterministic_time)
    if (
        budget < 0
        or budget != budget
        or budget == float("inf")
        or budget > ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME
    ):
        raise ValueError(
            "max_deterministic_time must be finite, non-negative, and within "
            "the trusted replay hard limit"
        )
    pool_hash = pool.sha256()
    if (
        pool.routing_world_provenance_digest
        and not pool.resources
        and not pool.landing_sites
        and pool.landing_site_catalog is None
        and not pool.owned_conductor_components
    ):
        return CompleteNetSelectionResult(
            status="UNKNOWN",
            solver_status="EMPTY_EXPLICIT_RESOURCE_LEDGER",
            candidate_pool_hash=pool_hash,
        )
    if budget == 0:
        return CompleteNetSelectionResult(
            status="UNKNOWN",
            solver_status="BUDGET_EXHAUSTED",
            candidate_pool_hash=pool_hash,
        )

    # Replay the two independently recomputable infeasibility proofs before
    # objective preflight or CP-SAT import.  Neither proof depends on physical
    # objective geometry or solver configuration; requiring either would turn a
    # valid capacity/Hall certificate into an unrelated UNKNOWN.
    conflict = _unavoidable_capacity_conflict(pool)
    if conflict is not None:
        return CompleteNetSelectionResult(
            status="INFEASIBLE",
            solver_status="PRECOMPUTED_CAPACITY_CONFLICT",
            candidate_pool_hash=pool_hash,
            certificate=conflict,
        )

    hall_deficit = _unavoidable_hall_deficit(pool)
    if hall_deficit is not None:
        return CompleteNetSelectionResult(
            status="INFEASIBLE",
            solver_status="PRECOMPUTED_HALL_DEFICIT",
            candidate_pool_hash=pool_hash,
            certificate=hall_deficit,
        )

    try:
        track_area, site_area = _route_plan_objective_coefficients(pool)
    except RoutePlanObjectiveGeometryError:
        return CompleteNetSelectionResult(
            status="UNKNOWN",
            solver_status="OBJECTIVE_GEOMETRY_INCOMPLETE",
            candidate_pool_hash=pool_hash,
        )

    try:
        cp_model = _load_cp_model()
    except (ImportError, OSError):
        return CompleteNetSelectionResult(
            status="UNKNOWN",
            solver_status="SOLVER_UNAVAILABLE",
            candidate_pool_hash=pool_hash,
        )

    (
        model,
        candidate_vars,
        track_vars,
        landing_vars,
        landing_option_vars,
        assumption_index_to_id,
        site_used,
    ) = _build_candidate_cp_model(
        pool,
        cp_model,
        additional_landing_site_conflicts=(
            additional_landing_site_conflicts
        ),
    )

    area_expression = sum(
        coefficient * track_vars[key]
        for key, coefficient in sorted(track_area.items())
    ) + sum(
        site_area[site_id] * site_used[site_id]
        for site_id in sorted(site_used)
    )
    count_expression = sum(site_used.values())
    semantic_vars = (
        [candidate_vars[key] for key in sorted(candidate_vars)]
        + [track_vars[key] for key in sorted(track_vars)]
        + [landing_option_vars[key] for key in sorted(landing_option_vars)]
        + [landing_vars[key] for key in sorted(landing_vars)]
    )
    if semantic_vars:
        model.AddDecisionStrategy(
            semantic_vars,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE,
        )

    remaining_budget = (
        budget if budget_account is None else min(budget, budget_account[0])
    )

    def solve_phase(
        *,
        fixed_search: bool = False,
        presolve: Optional[bool] = None,
        probing_level: Optional[int] = None,
    ):
        nonlocal remaining_budget
        if remaining_budget <= 0:
            return None, None, "BUDGET_EXHAUSTED"
        phase_solver = cp_model.CpSolver()
        phase_solver.parameters.num_search_workers = 1
        phase_solver.parameters.random_seed = 0
        phase_solver.parameters.max_deterministic_time = remaining_budget
        if fixed_search:
            phase_solver.parameters.search_branching = cp_model.FIXED_SEARCH
        if presolve is not None:
            phase_solver.parameters.cp_model_presolve = presolve
        if probing_level is not None:
            phase_solver.parameters.cp_model_probing_level = probing_level
        phase_status = phase_solver.Solve(model)
        response = phase_solver.ResponseProto()
        consumed = float(getattr(response, "deterministic_time", 0.0))
        if math.isfinite(consumed) and consumed > 0:
            remaining_budget = max(0.0, remaining_budget - consumed)
            if budget_account is not None:
                budget_account[0] = remaining_budget
        return phase_solver, phase_status, phase_solver.StatusName(phase_status)

    def install_assignment_hints(solver: object) -> None:
        """Replace hints atomically with one complete witnessed assignment."""

        model.ClearHints()
        response = solver.ResponseProto()
        solution = tuple(response.solution)
        variable_count = len(model.Proto().variables)
        if len(solution) != variable_count:
            raise CandidatePoolError(
                "feasibility incumbent is not a full model assignment: "
                f"solution={len(solution)} model={variable_count}"
            )
        for index, value in enumerate(solution):
            model.AddHint(
                model.GetBoolVarFromProtoIndex(index),
                int(value),
            )

    def partial_result(
        *,
        status_name: str,
        phases: tuple[RoutePlanObjectivePhase, ...],
        incumbent_solver: Optional[object],
    ) -> CompleteNetSelectionResult:
        selected = (
            ()
            if incumbent_solver is None
            else _selected_nets_from_solver(
                pool,
                incumbent_solver,
                candidate_vars,
                track_vars,
                landing_vars,
                landing_option_vars,
            )
        )
        materializable = (
            incumbent_solver is not None
            and len(selected) == len(pool.required_nets)
            and _owned_conductor_index_certificate_from_pool(pool) is not None
        )
        if materializable:
            certificate = _objective_certificate(
                pool_hash,
                selected,
                phases,
                authorized_replay_deterministic_time=budget,
            )
            plan = _plan_from_selected_nets(pool, selected, certificate)
            return CompleteNetSelectionResult(
                status="FEASIBLE",
                solver_status=f"OPTIMIZATION_INCOMPLETE_{status_name}",
                candidate_pool_hash=pool_hash,
                plan=plan,
                certificate=certificate,
            )
        certificate = _objective_certificate(
            pool_hash,
            selected,
            phases,
            authorized_replay_deterministic_time=budget,
        )
        partial_plan = (
            None
            if (
                len(selected) != len(pool.required_nets)
                or _owned_conductor_index_certificate_from_pool(pool) is None
            )
            else _plan_from_selected_nets(pool, selected, certificate)
        )
        return CompleteNetSelectionResult(
            status="UNKNOWN",
            solver_status=status_name,
            candidate_pool_hash=pool_hash,
            partial_plan=partial_plan,
            certificate=certificate,
        )

    # The certificate has always named hard feasibility as phase zero, but the
    # old selector never solved that phase independently: its first solve also
    # carried the area objective.  Dense closed landing-option families could
    # therefore consume the complete proof account without finding any
    # incumbent.  Phase zero therefore uses a lean satisfaction configuration:
    # no objective, no presolve, and no probing.  If it remains UNKNOWN, the
    # proof-oriented default configuration may consume only the account that
    # remains.  Neither path raises or renews the caller's signed budget.
    model.ClearObjective()
    feasibility_solver, solver_status, status_name = solve_phase(
        fixed_search=True,
        presolve=False,
        probing_level=0,
    )
    if solver_status in (None, cp_model.UNKNOWN):
        feasibility_solver, solver_status, status_name = solve_phase(
            fixed_search=True,
        )
    if solver_status == cp_model.INFEASIBLE:
        assert feasibility_solver is not None
        core_ids = tuple(sorted({
            assumption_index_to_id[index]
            for index in feasibility_solver.SufficientAssumptionsForInfeasibility()
            if index in assumption_index_to_id
        }))
        if not core_ids:
            return CompleteNetSelectionResult(
                status="UNKNOWN",
                solver_status="INFEASIBLE_WITHOUT_ASSUMPTION_CORE",
                candidate_pool_hash=pool_hash,
            )
        return CompleteNetSelectionResult(
            status="INFEASIBLE",
            solver_status=status_name,
            candidate_pool_hash=pool_hash,
            certificate=PoolUnsatCertificate(
                candidate_pool_hash=pool_hash,
                assumption_core_ids=core_ids,
                reason=(
                    "candidate-pool CP-SAT model is infeasible under this "
                    "sufficient assumption core; no minimal IIS claim"
                ),
            ),
        )
    if solver_status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return partial_result(
            status_name=f"HARD_FEASIBILITY_{status_name}",
            phases=(RoutePlanObjectivePhase(
                "hard_feasibility", "UNKNOWN", None, None
            ),),
            incumbent_solver=None,
        )
    assert feasibility_solver is not None
    phases = [RoutePlanObjectivePhase(
        "hard_feasibility", "OPTIMAL", 0, 0
    )]
    feasibility_area = int(feasibility_solver.Value(area_expression))
    install_assignment_hints(feasibility_solver)
    model.Add(area_expression <= feasibility_area)

    model.Minimize(area_expression)
    area_solver, solver_status, status_name = solve_phase(
        presolve=False,
        probing_level=0,
    )
    if solver_status != cp_model.OPTIMAL:
        if solver_status == cp_model.FEASIBLE:
            assert area_solver is not None
            incumbent = area_solver
            area_incumbent = int(area_solver.Value(area_expression))
            # A FEASIBLE solve witnesses only the incumbent.  Its transient
            # BestObjectiveBound is not independently stable evidence, so the
            # signed certificate deliberately carries the conservative zero.
            area_lower_bound = 0
        else:
            incumbent = feasibility_solver
            area_incumbent = feasibility_area
            area_lower_bound = 0
        phases.append(RoutePlanObjectivePhase(
            "new_reserved_geometry_area_nm2",
            "FEASIBLE",
            area_incumbent,
            area_lower_bound,
        ))
        return partial_result(
            status_name=f"AREA_{status_name}",
            phases=tuple(phases),
            incumbent_solver=incumbent,
        )
    assert area_solver is not None
    area_value = int(area_solver.Value(area_expression))
    phases.append(RoutePlanObjectivePhase(
        "new_reserved_geometry_area_nm2",
        "OPTIMAL",
        area_value,
        area_value,
    ))
    model.Add(area_expression == area_value)
    install_assignment_hints(area_solver)

    model.Minimize(count_expression)
    count_solver, solver_status, status_name = solve_phase(
        presolve=False,
        probing_level=0,
    )
    if solver_status != cp_model.OPTIMAL:
        if solver_status == cp_model.FEASIBLE:
            assert count_solver is not None
            count_incumbent = int(count_solver.Value(count_expression))
            phases.append(RoutePlanObjectivePhase(
                "landing_site_cut_count",
                "FEASIBLE",
                count_incumbent,
                0,
            ))
            incumbent = count_solver
        else:
            incumbent = area_solver
            count_incumbent = int(area_solver.Value(count_expression))
            phases.append(RoutePlanObjectivePhase(
                "landing_site_cut_count", "FEASIBLE", count_incumbent, 0
            ))
        return partial_result(
            status_name=f"LANDING_COUNT_{status_name}",
            phases=tuple(phases),
            incumbent_solver=incumbent,
        )
    assert count_solver is not None
    count_value = int(count_solver.Value(count_expression))
    phases.append(RoutePlanObjectivePhase(
        "landing_site_cut_count", "OPTIMAL", count_value, count_value
    ))
    install_assignment_hints(count_solver)
    model.Add(count_expression == count_value)

    model.ClearObjective()
    # Hints are intentionally absent from the canonical fixed-search phase;
    # its ordered semantic decision strategy, not a prior optimum's accidental
    # assignment, owns the byte-stable tie break.
    model.ClearHints()
    tie_solver, solver_status, status_name = solve_phase(fixed_search=True)
    if solver_status != cp_model.OPTIMAL:
        phases.append(RoutePlanObjectivePhase(
            "canonical_semantic_tie",
            "FEASIBLE" if solver_status == cp_model.FEASIBLE else "UNKNOWN",
            None,
            None,
        ))
        return partial_result(
            status_name=f"CANONICAL_TIE_{status_name}",
            phases=tuple(phases),
            incumbent_solver=(
                tie_solver if solver_status == cp_model.FEASIBLE else count_solver
            ),
        )
    assert tie_solver is not None
    phases.append(RoutePlanObjectivePhase(
        "canonical_semantic_tie", "OPTIMAL", None, None
    ))
    selected = _selected_nets_from_solver(
        pool,
        tie_solver,
        candidate_vars,
        track_vars,
        landing_vars,
        landing_option_vars,
    )
    certificate = _objective_certificate(
        pool_hash,
        selected,
        tuple(phases),
        authorized_replay_deterministic_time=budget,
    )
    if _owned_conductor_index_certificate_from_pool(pool) is None:
        return CompleteNetSelectionResult(
            status="UNKNOWN",
            solver_status="NON_MATERIALIZABLE_SELECTION",
            candidate_pool_hash=pool_hash,
            certificate=certificate,
        )
    plan = _plan_from_selected_nets(
        pool,
        selected,
        certificate,
        additional_landing_site_conflicts,
    )
    return CompleteNetSelectionResult(
        status="SAT",
        solver_status="OPTIMAL",
        candidate_pool_hash=pool_hash,
        plan=plan,
    )


def select_complete_net_candidates(
    pool: CandidatePool,
    *,
    max_deterministic_time: float = (
        ROUTE_PLAN_OBJECTIVE_DEFAULT_DETERMINISTIC_TIME
    ),
) -> CompleteNetSelectionResult:
    """Select all nets, lazily separating a compressed exact conflict domain."""

    pool = _replay_candidate_pool_runtime(pool)
    if pool.landing_site_conflict_domain is None:
        result = _select_complete_net_candidates_once(
            pool, max_deterministic_time=max_deterministic_time
        )
        if result.status in {"SAT", "FEASIBLE"} and result.plan is not None:
            receipt = certify_route_resource_plan_verification(
                pool, result.plan
            )
            return replace(result, verification_receipt=receipt)
        return result
    if isinstance(max_deterministic_time, bool) or not isinstance(
        max_deterministic_time, (int, float)
    ):
        raise ValueError("max_deterministic_time must be a non-negative number")
    budget = float(max_deterministic_time)
    if (
        budget < 0
        or not math.isfinite(budget)
        or budget > ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME
    ):
        raise ValueError(
            "max_deterministic_time must be finite, non-negative, and within "
            "the trusted replay hard limit"
        )
    budget_account = [budget]
    lazy_conflicts: tuple[LandingSitePairConflict, ...] = ()
    for _round in range(MAX_LAZY_LANDING_SITE_CONFLICTS + 1):
        try:
            result = _select_complete_net_candidates_once(
                pool,
                max_deterministic_time=budget_account[0],
                additional_landing_site_conflicts=lazy_conflicts,
                budget_account=budget_account,
            )
        except CandidatePoolLimitError:
            return CompleteNetSelectionResult(
                status="UNKNOWN",
                solver_status="LANDING_CONFLICT_CUT_LIMIT_EXCEEDED",
                candidate_pool_hash=pool.sha256(),
            )
        if result.status not in {"SAT", "FEASIBLE"}:
            if result.status == "INFEASIBLE" and not lazy_conflicts:
                return result
            return CompleteNetSelectionResult(
                status="UNKNOWN",
                solver_status=(
                    "LANDING_CONFLICT_SEPARATION_"
                    f"{result.solver_status}"
                ),
                candidate_pool_hash=pool.sha256(),
            )
        assert result.plan is not None
        try:
            violations = _selected_domain_landing_conflicts(
                pool, result.plan.selected_nets
            )
        except (CandidatePoolError, CandidatePoolLimitError):
            return CompleteNetSelectionResult(
                status="UNKNOWN",
                solver_status="LANDING_CONFLICT_QUERY_BUDGET_EXHAUSTED",
                candidate_pool_hash=pool.sha256(),
            )
        if not violations:
            receipt = certify_route_resource_plan_verification(
                pool, result.plan
            )
            return replace(result, verification_receipt=receipt)
        merged = tuple(sorted(set(lazy_conflicts) | set(violations)))
        if len(merged) > MAX_LAZY_LANDING_SITE_CONFLICTS:
            return CompleteNetSelectionResult(
                status="UNKNOWN",
                solver_status="LANDING_CONFLICT_CUT_LIMIT_EXCEEDED",
                candidate_pool_hash=pool.sha256(),
            )
        if merged == lazy_conflicts:
            return CompleteNetSelectionResult(
                status="UNKNOWN",
                solver_status="LANDING_CONFLICT_SEPARATION_STALLED",
                candidate_pool_hash=pool.sha256(),
            )
        lazy_conflicts = merged
    return CompleteNetSelectionResult(
        status="UNKNOWN",
        solver_status="LANDING_CONFLICT_CUT_LIMIT_EXCEEDED",
        candidate_pool_hash=pool.sha256(),
    )


def _bringup_partition_plan_rank(
    result: CompleteNetSelectionResult,
) -> tuple[object, ...]:
    """Order locally verified diagnostic winners without making a global claim."""

    if result.plan is None:
        raise CandidatePoolError("bring-up partition result has no materializable plan")
    certificate = result.plan.objective_certificate
    phases = () if certificate is None else certificate.phases
    return (
        0 if result.status == "SAT" else 1,
        tuple(
            (
                phase.phase_id,
                0 if phase.status == "OPTIMAL" else 1,
                0 if phase.value is None else phase.value,
            )
            for phase in phases
        ),
        result.plan.sha256(),
    )


def _select_bringup_partition_frontier(
    partitions: Sequence[CandidatePool],
) -> tuple[
    tuple[
        tuple[object, ...],
        CandidatePool,
        RouteResourcePlan,
        CompleteNetCandidate,
    ], ...
]:
    """Solve every bounded one-net partition into a concrete local frontier.

    This helper is intentionally private and diagnostic-only.  Every input pool
    remains subject to the ordinary CandidatePool bounds and the ordinary
    selector/verifier.  Each successful input contributes exactly one winner;
    no successful family is silently discarded.  The returned plans are *local*
    witnesses: they do not certify that winners from different pools can coexist,
    and callers must run a separate exact conflict-aware merge before geometry.
    """

    raw = _bounded_materialize(
        partitions,
        limit=MAX_TOTAL_CANDIDATES,
        label="bring-up candidate partition",
        error_type=CandidatePoolLimitError,
    )
    if not all(type(pool) is CandidatePool for pool in raw):
        raise TypeError("bring-up partitions must contain exact CandidatePool values")
    ordered = tuple(sorted(raw, key=lambda pool: (
        pool.required_nets,
        pool.sha256(),
    )))
    frontier = []
    for pool in ordered:
        if len(pool.required_nets) != 1:
            raise CandidatePoolError(
                "bring-up candidate partition must contain exactly one required net"
            )
        net = pool.required_nets[0]
        result = select_complete_net_candidates(pool)
        if result.status not in {"SAT", "FEASIBLE"} or result.plan is None:
            continue
        verify_route_resource_plan(pool, result.plan)
        rank = _bringup_partition_plan_rank(result)
        if len(result.plan.selected_nets) != 1:
            raise CandidatePoolError(
                "bring-up partition plan must select exactly one net"
            )
        selected = result.plan.selected_nets[0]
        candidate = next(
            (
                item
                for item in pool.all_candidates
                if item.candidate_id == selected.candidate_id
            ),
            None,
        )
        if candidate is None or candidate.net != net:
            raise CandidatePoolError(
                "bring-up partition winner does not replay to one exact candidate"
            )
        frontier.append((rank, pool, result.plan, candidate))
    return tuple(sorted(frontier, key=lambda item: (
        item[3].net,
        item[1].sha256(),
        item[3].candidate_id,
    )))


def _select_bringup_partition_winners(
    partitions: Sequence[CandidatePool],
) -> tuple[tuple[CandidatePool, RouteResourcePlan], ...]:
    """Retain the best locally verified partition witness for each net."""

    winners: dict[
        str,
        tuple[tuple[object, ...], CandidatePool, RouteResourcePlan],
    ] = {}
    for rank, pool, plan, candidate in _select_bringup_partition_frontier(
        partitions
    ):
        previous = winners.get(candidate.net)
        if previous is None or rank < previous[0]:
            winners[candidate.net] = (rank, pool, plan)
    return tuple((winners[net][1], winners[net][2]) for net in sorted(winners))


def _select_bringup_partition_candidates(
    partitions: Sequence[CandidatePool],
) -> tuple[CompleteNetCandidate, ...]:
    """Return the concrete local winner from every diagnostic net partition."""

    candidates = []
    for pool, plan in _select_bringup_partition_winners(partitions):
        if len(plan.selected_nets) != 1:
            raise CandidatePoolError(
                "bring-up partition plan must select exactly one net"
            )
        selected = plan.selected_nets[0]
        candidate = next(
            (
                item
                for item in pool.all_candidates
                if item.candidate_id == selected.candidate_id
            ),
            None,
        )
        if candidate is None or candidate.net != selected.net:
            raise CandidatePoolError(
                "bring-up partition winner does not replay to one exact candidate"
            )
        candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda candidate: (
        candidate.net,
        candidate.candidate_id,
    )))


def _rebuild_selected_net_candidate(
    selected: SelectedNetCandidate,
) -> SelectedNetCandidate:
    tracks = tuple(
        replace(track, resource=replace(track.resource))
        for track in _bounded_materialize(
            selected.selected_tracks,
            limit=MAX_TOTAL_CORRIDOR_USES,
            label="selected track",
            error_type=CandidatePoolLimitError,
        )
    )
    landings = tuple(
        replace(landing)
        for landing in _bounded_materialize(
            selected.selected_landings,
            limit=MAX_TOTAL_LANDING_OPTIONS,
            label="selected landing",
            error_type=CandidatePoolLimitError,
        )
    )
    options = tuple(
        replace(option)
        for option in _bounded_materialize(
            selected.selected_landing_options,
            limit=MAX_TOTAL_LANDING_OPTIONS,
            label="selected landing option",
            error_type=CandidatePoolLimitError,
        )
    )
    return replace(
        selected,
        selected_tracks=tracks,
        selected_landings=landings,
        selected_landing_options=options,
    )


def _rebuild_allocated_track(track: AllocatedTrack) -> AllocatedTrack:
    return replace(
        track,
        resource=_rebuild_track_resource(track.resource),
        interval_nm=_bounded_materialize(
            track.interval_nm,
            limit=2,
            label="allocated track interval coordinate",
            error_type=CandidatePoolLimitError,
        ),
        selected_track_centers_nm=_bounded_materialize(
            track.selected_track_centers_nm,
            limit=MAX_TRACK_CENTERS_PER_RESOURCE,
            label="allocated selected track center",
            error_type=CandidatePoolLimitError,
        ),
    )


def _rebuild_allocated_net_program(
    program: AllocatedNetProgram,
) -> AllocatedNetProgram:
    branches = tuple(
        replace(
            branch,
            ordered_use_ids=_bounded_materialize(
                branch.ordered_use_ids,
                limit=MAX_CANDIDATE_BRANCH_ORDERED_USES,
                label="branch ordered use id",
                error_type=CandidatePoolLimitError,
            ),
            ordered_landing_demand_ids=_bounded_materialize(
                branch.ordered_landing_demand_ids,
                limit=MAX_CANDIDATE_BRANCH_ORDERED_LANDING_DEMANDS,
                label="branch ordered landing demand id",
                error_type=CandidatePoolLimitError,
            ),
            ordered_steps=tuple(
                replace(step)
                for step in _bounded_materialize(
                    branch.ordered_steps,
                    limit=MAX_CANDIDATE_BRANCH_STEPS,
                    label="branch step",
                    error_type=CandidatePoolLimitError,
                )
            ),
        )
        for branch in _bounded_materialize(
            program.branches,
            limit=MAX_TOTAL_CANDIDATE_BRANCHES,
            label="allocated branch",
            error_type=CandidatePoolLimitError,
        )
    )
    coverage = tuple(
        replace(item)
        for item in _bounded_materialize(
            program.goal_branch_coverage,
            limit=MAX_TOTAL_GOAL_BRANCH_COVERAGE,
            label="allocated goal branch coverage",
            error_type=CandidatePoolLimitError,
        )
    )
    accesses = tuple(
        replace(
            access,
            xy_nm=_bounded_materialize(
                access.xy_nm,
                limit=2,
                label="endpoint access coordinate",
                error_type=CandidatePoolLimitError,
            ),
            layer_key=_bounded_materialize(
                access.layer_key,
                limit=2,
                label="endpoint access GDS layer key",
                error_type=CandidatePoolLimitError,
            ),
            owner_component_refs=_bounded_materialize(
                access.owner_component_refs,
                limit=MAX_ENDPOINT_OWNER_COMPONENT_REFS,
                label="endpoint owner component ref",
                error_type=CandidatePoolLimitError,
            ),
        )
        for access in _bounded_materialize(
            program.endpoint_accesses,
            limit=MAX_TOTAL_ENDPOINT_ACCESSES,
            label="allocated endpoint access",
            error_type=CandidatePoolLimitError,
        )
    )
    demands = tuple(
        replace(
            demand,
            allowed_site_ids=_bounded_materialize(
                demand.allowed_site_ids,
                limit=MAX_LANDING_DEMAND_ALLOWED_SITES,
                label="allowed site",
                error_type=CandidatePoolLimitError,
            ),
        )
        for demand in _bounded_materialize(
            program.landing_demands,
            limit=MAX_TOTAL_LANDING_DEMANDS,
            label="allocated landing demand",
            error_type=CandidatePoolLimitError,
        )
    )
    goals = tuple(
        _rebuild_required_route_goal(goal)
        for goal in _bounded_materialize(
            program.required_route_goals,
            limit=MAX_REQUIRED_ROUTE_GOALS,
            label="allocated required route goal",
            error_type=CandidatePoolLimitError,
        )
    )
    tracks = tuple(
        _rebuild_allocated_track(track)
        for track in _bounded_materialize(
            program.tracks,
            limit=MAX_TOTAL_CORRIDOR_USES,
            label="allocated track",
            error_type=CandidatePoolLimitError,
        )
    )
    allocations = []
    for allocation in _bounded_materialize(
        program.landing_options,
        limit=MAX_TOTAL_LANDING_OPTIONS,
        label="allocated landing option",
        error_type=CandidatePoolLimitError,
    ):
        option = allocation.option
        rebuilt_option = replace(
            option,
            site_ids=_bounded_materialize(
                option.site_ids,
                limit=MAX_LANDING_OPTION_SITE_IDS,
                label="landing option site id",
                error_type=CandidatePoolLimitError,
            ),
            contacts=tuple(
                replace(contact, track_choice=replace(contact.track_choice))
                for contact in _bounded_materialize(
                    option.contacts,
                    limit=MAX_LANDING_OPTION_CONTACTS_PER_OPTION,
                    label="landing option contact",
                    error_type=CandidatePoolLimitError,
                )
            ),
        )
        landing_sites = tuple(
            _rebuild_landing_site(site)
            for site in _bounded_materialize(
                allocation.landing_sites,
                limit=MAX_LANDING_OPTION_SITE_IDS,
                label="allocated landing site",
                error_type=CandidatePoolLimitError,
            )
        )
        allocations.append(
            replace(
                allocation,
                option=rebuilt_option,
                landing_sites=landing_sites,
            )
        )
    certificate_member_budget = [0]
    rebuilt_certificates = []
    for certificate in _bounded_materialize(
        program.owned_conductor_components,
        limit=MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES,
        label="allocated owned conductor component certificate",
        error_type=CandidatePoolLimitError,
    ):
        rebuilt_certificates.append(replace(
            certificate,
            component=_rebuild_owned_conductor_component(
                certificate.component,
                member_budget=certificate_member_budget,
            ),
        ))
    certificates = tuple(rebuilt_certificates)
    return replace(
        program,
        branches=branches,
        goal_branch_coverage=coverage,
        endpoint_accesses=accesses,
        landing_demands=demands,
        required_route_goals=goals,
        tracks=tracks,
        landing_options=tuple(allocations),
        owned_conductor_components=certificates,
        owned_component_use_refs=_bounded_materialize(
            program.owned_component_use_refs,
            limit=MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE,
            label="allocated owned component use ref",
            error_type=CandidatePoolLimitError,
        ),
    )


def _independently_replay_route_plan_objective(
    pool: CandidatePool,
    *,
    additional_landing_site_conflicts: tuple[
        LandingSitePairConflict, ...
    ],
    max_deterministic_time: float,
) -> tuple[tuple[SelectedNetCandidate, ...], int, int]:
    """Rebuild and prove the canonical lexicographic optimum from trusted inputs.

    This verifier deliberately does not call either selection entry point and
    never reads a plan objective certificate.  It reconstructs the finite CP
    model from the pool plus replayed lazy-domain clauses, closes both scalar
    objective lower bounds, then repeats the fixed-search semantic tie-break.
    Any unproved phase is a typed verification failure, never acceptance of an
    incumbent.
    """

    if isinstance(max_deterministic_time, bool) or not isinstance(
        max_deterministic_time, (int, float)
    ):
        raise RouteResourcePlanVerificationError(
            "independent objective replay budget must be a non-negative number"
        )
    remaining_budget = float(max_deterministic_time)
    if (
        remaining_budget < 0
        or not math.isfinite(remaining_budget)
    ):
        raise RouteResourcePlanVerificationError(
            "independent objective replay budget must be finite and non-negative"
        )
    if remaining_budget == 0:
        raise RouteResourcePlanVerificationBudgetExhausted(
            "independent objective replay BUDGET_EXHAUSTED"
        )

    try:
        track_area, site_area = _route_plan_objective_coefficients(pool)
        cp_model = _load_cp_model()
        (
            model,
            candidate_vars,
            track_vars,
            landing_vars,
            landing_option_vars,
            _assumption_index_to_id,
            site_used,
        ) = _build_candidate_cp_model(
            pool,
            cp_model,
            additional_landing_site_conflicts=(
                additional_landing_site_conflicts
            ),
        )
    except Exception as exc:
        raise RouteResourcePlanVerificationError(
            "independent objective replay could not rebuild the closed model: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    area_expression = sum(
        coefficient * track_vars[key]
        for key, coefficient in sorted(track_area.items())
    ) + sum(
        site_area[site_id] * site_used[site_id]
        for site_id in sorted(site_used)
    )
    count_expression = sum(site_used.values())
    semantic_vars = (
        [candidate_vars[key] for key in sorted(candidate_vars)]
        + [track_vars[key] for key in sorted(track_vars)]
        + [landing_option_vars[key] for key in sorted(landing_option_vars)]
        + [landing_vars[key] for key in sorted(landing_vars)]
    )
    if semantic_vars:
        model.AddDecisionStrategy(
            semantic_vars,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE,
        )

    def solve_phase(
        phase_id: str,
        *,
        fixed_search: bool = False,
        presolve: Optional[bool] = None,
        probing_level: Optional[int] = None,
        accept_feasible: bool = False,
    ):
        nonlocal remaining_budget
        if remaining_budget <= 0:
            raise RouteResourcePlanVerificationBudgetExhausted(
                "independent objective replay BUDGET_EXHAUSTED before "
                f"{phase_id}"
            )
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_deterministic_time = remaining_budget
        if fixed_search:
            solver.parameters.search_branching = cp_model.FIXED_SEARCH
        if presolve is not None:
            solver.parameters.cp_model_presolve = presolve
        if probing_level is not None:
            solver.parameters.cp_model_probing_level = probing_level
        try:
            status = solver.Solve(model)
            status_name = solver.StatusName(status)
            response = solver.ResponseProto()
        except Exception as exc:
            raise RouteResourcePlanVerificationError(
                "independent objective replay solver failed in "
                f"{phase_id}: {type(exc).__name__}: {exc}"
            ) from exc
        consumed = float(getattr(response, "deterministic_time", 0.0))
        if math.isfinite(consumed) and consumed > 0:
            remaining_budget = max(0.0, remaining_budget - consumed)
        if status == cp_model.UNKNOWN or (
            status == cp_model.FEASIBLE and not accept_feasible
        ):
            raise RouteResourcePlanVerificationBudgetExhausted(
                "independent objective replay remained UNKNOWN in "
                f"{phase_id}: {status_name}"
            )
        accepted_statuses = (
            (cp_model.OPTIMAL, cp_model.FEASIBLE)
            if accept_feasible
            else (cp_model.OPTIMAL,)
        )
        if status not in accepted_statuses:
            raise RouteResourcePlanVerificationError(
                "independent objective replay contradicted the verified plan in "
                f"{phase_id}: {status_name}"
            )
        return solver

    def install_assignment_hints(solver: object) -> None:
        model.ClearHints()
        solution = tuple(solver.ResponseProto().solution)
        variable_count = len(model.Proto().variables)
        if len(solution) != variable_count:
            raise RouteResourcePlanVerificationError(
                "independent objective replay incumbent is not a full model "
                "assignment"
            )
        for index, value in enumerate(solution):
            model.AddHint(
                model.GetBoolVarFromProtoIndex(index),
                int(value),
            )

    model.ClearObjective()
    try:
        feasibility_solver = solve_phase(
            "hard_feasibility",
            fixed_search=True,
            presolve=False,
            probing_level=0,
            accept_feasible=True,
        )
    except (
        RouteResourcePlanVerificationBudgetExhausted,
        RouteResourcePlanVerificationError,
    ):
        feasibility_solver = solve_phase(
            "hard_feasibility_proof",
            fixed_search=True,
            accept_feasible=True,
        )
    feasibility_area = int(feasibility_solver.Value(area_expression))
    install_assignment_hints(feasibility_solver)
    model.Add(area_expression <= feasibility_area)

    model.Minimize(area_expression)
    area_solver = solve_phase(
        "new_reserved_geometry_area_nm2",
        presolve=False,
        probing_level=0,
    )
    area_value = int(area_solver.Value(area_expression))
    model.Add(area_expression == area_value)
    install_assignment_hints(area_solver)

    model.Minimize(count_expression)
    count_solver = solve_phase(
        "landing_site_cut_count",
        presolve=False,
        probing_level=0,
    )
    landing_count = int(count_solver.Value(count_expression))
    install_assignment_hints(count_solver)
    model.Add(count_expression == landing_count)

    model.ClearObjective()
    model.ClearHints()
    tie_solver = solve_phase("canonical_semantic_tie", fixed_search=True)
    selected = _selected_nets_from_solver(
        pool,
        tie_solver,
        candidate_vars,
        track_vars,
        landing_vars,
        landing_option_vars,
    )
    return selected, area_value, landing_count


def _independently_replay_fixed_route_plan_feasibility(
    pool: CandidatePool,
    selected_nets: tuple[SelectedNetCandidate, ...],
    *,
    additional_landing_site_conflicts: tuple[
        LandingSitePairConflict, ...
    ],
    max_deterministic_time: float,
) -> None:
    """Prove a fixed optimization-incomplete plan in a fresh closed CP model.

    Every semantic decision variable is pinned below before solving.  Presolve
    therefore performs exact equality propagation and contradiction detection;
    it does not permit an alternative candidate/track/landing assignment,
    although it still scans and simplifies the complete model.  The public
    verifier has already replayed the selected geometry and capacities before
    reaching this independent closed-model consistency proof.
    """

    cp_model = _load_cp_model()
    (
        model,
        candidate_vars,
        track_vars,
        landing_vars,
        landing_option_vars,
        _assumption_index_to_id,
        _site_used,
    ) = _build_candidate_cp_model(
        pool,
        cp_model,
        additional_landing_site_conflicts=(
            additional_landing_site_conflicts
        ),
    )
    selected_candidate_ids = {
        selected.candidate_id for selected in selected_nets
    }
    selected_track_keys = {
        (selected.candidate_id, track.use_id, track.track_start)
        for selected in selected_nets
        for track in selected.selected_tracks
    }
    selected_landing_keys = {
        (selected.candidate_id, landing.demand_id, landing.site_id)
        for selected in selected_nets
        for landing in selected.selected_landings
    }
    selected_option_keys = {
        (selected.candidate_id, option.demand_id, option.option_id)
        for selected in selected_nets
        for option in selected.selected_landing_options
    }
    for candidate_id, variable in sorted(candidate_vars.items()):
        model.Add(variable == int(candidate_id in selected_candidate_ids))
    for key, variable in sorted(track_vars.items()):
        model.Add(variable == int(key in selected_track_keys))
    for key, variable in sorted(landing_vars.items()):
        model.Add(variable == int(key in selected_landing_keys))
    for key, variable in sorted(landing_option_vars.items()):
        model.Add(variable == int(key in selected_option_keys))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_deterministic_time = max_deterministic_time
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    solver.parameters.cp_model_presolve = True
    solver.parameters.cp_model_probing_level = 0
    status = solver.Solve(model)
    if status == cp_model.UNKNOWN:
        raise RouteResourcePlanVerificationBudgetExhausted(
            "independent fixed-plan replay remained UNKNOWN"
        )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        raise RouteResourcePlanVerificationError(
            "optimization-incomplete plan contradicts the complete CP model: "
            f"{solver.StatusName(status)}"
        )
    replayed = _selected_nets_from_solver(
        pool,
        solver,
        candidate_vars,
        track_vars,
        landing_vars,
        landing_option_vars,
    )
    if replayed != selected_nets:
        raise RouteResourcePlanVerificationError(
            "independent fixed-plan replay selected different route resources"
        )


def _independently_replay_claimed_optimal_objective_prefix(
    pool: CandidatePool,
    selected_nets: tuple[SelectedNetCandidate, ...],
    phases: tuple[RoutePlanObjectivePhase, ...],
    *,
    additional_landing_site_conflicts: tuple[
        LandingSitePairConflict, ...
    ],
    max_deterministic_time: float,
) -> float:
    """Reprove every OPTIMAL claim in an incomplete objective certificate.

    The replay mirrors the selector's phase order, solver parameters, full
    incumbent hints, witnessed area upper bound, and single deterministic-time
    account.  It deliberately stops before the terminal FEASIBLE/UNKNOWN phase:
    those states carry an incumbent, not an optimality claim.  The returned
    account remainder is reserved for the final independently fixed-assignment
    feasibility replay.
    """

    if isinstance(max_deterministic_time, bool) or not isinstance(
        max_deterministic_time, (int, float)
    ):
        raise RouteResourcePlanVerificationError(
            "independent objective-prefix replay budget must be numeric"
        )
    remaining_budget = float(max_deterministic_time)
    if remaining_budget < 0 or not math.isfinite(remaining_budget):
        raise RouteResourcePlanVerificationError(
            "independent objective-prefix replay budget must be finite and "
            "non-negative"
        )
    if remaining_budget == 0:
        raise RouteResourcePlanVerificationBudgetExhausted(
            "independent objective-prefix replay BUDGET_EXHAUSTED"
        )

    try:
        track_area, site_area = _route_plan_objective_coefficients(pool)
        cp_model = _load_cp_model()
        (
            model,
            candidate_vars,
            track_vars,
            landing_vars,
            landing_option_vars,
            _assumption_index_to_id,
            site_used,
        ) = _build_candidate_cp_model(
            pool,
            cp_model,
            additional_landing_site_conflicts=(
                additional_landing_site_conflicts
            ),
        )
    except Exception as exc:
        raise RouteResourcePlanVerificationError(
            "independent objective-prefix replay could not rebuild the closed "
            f"model: {type(exc).__name__}: {exc}"
        ) from exc

    area_expression = sum(
        coefficient * track_vars[key]
        for key, coefficient in sorted(track_area.items())
    ) + sum(
        site_area[site_id] * site_used[site_id]
        for site_id in sorted(site_used)
    )
    count_expression = sum(site_used.values())
    semantic_vars = (
        [candidate_vars[key] for key in sorted(candidate_vars)]
        + [track_vars[key] for key in sorted(track_vars)]
        + [landing_option_vars[key] for key in sorted(landing_option_vars)]
        + [landing_vars[key] for key in sorted(landing_vars)]
    )
    if semantic_vars:
        model.AddDecisionStrategy(
            semantic_vars,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE,
        )

    def solve_phase(
        phase_id: str,
        *,
        fixed_search: bool = False,
        presolve: Optional[bool] = None,
        probing_level: Optional[int] = None,
        feasibility_proof: bool = False,
    ) -> tuple[object, int]:
        nonlocal remaining_budget
        if remaining_budget <= 0:
            raise RouteResourcePlanVerificationBudgetExhausted(
                "independent objective-prefix replay BUDGET_EXHAUSTED before "
                f"{phase_id}"
            )
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_deterministic_time = remaining_budget
        if fixed_search:
            solver.parameters.search_branching = cp_model.FIXED_SEARCH
        if presolve is not None:
            solver.parameters.cp_model_presolve = presolve
        if probing_level is not None:
            solver.parameters.cp_model_probing_level = probing_level
        try:
            status = solver.Solve(model)
            response = solver.ResponseProto()
        except Exception as exc:
            raise RouteResourcePlanVerificationError(
                "independent objective-prefix replay solver failed in "
                f"{phase_id}: {type(exc).__name__}: {exc}"
            ) from exc
        consumed = float(getattr(response, "deterministic_time", 0.0))
        if math.isfinite(consumed) and consumed > 0:
            remaining_budget = max(0.0, remaining_budget - consumed)
        if status == cp_model.UNKNOWN or (
            status == cp_model.FEASIBLE and not feasibility_proof
        ):
            raise RouteResourcePlanVerificationBudgetExhausted(
                "independent objective-prefix replay remained UNKNOWN in "
                f"{phase_id}: {solver.StatusName(status)}"
            )
        accepted = (
            (cp_model.OPTIMAL, cp_model.FEASIBLE)
            if feasibility_proof
            else (cp_model.OPTIMAL,)
        )
        if status not in accepted:
            raise RouteResourcePlanVerificationError(
                "independent objective-prefix replay contradicted the "
                f"certificate in {phase_id}: {solver.StatusName(status)}"
            )
        return solver, status

    def install_assignment_hints(solver: object) -> None:
        model.ClearHints()
        solution = tuple(solver.ResponseProto().solution)
        variable_count = len(model.Proto().variables)
        if len(solution) != variable_count:
            raise RouteResourcePlanVerificationError(
                "independent objective-prefix replay incumbent is not a full "
                "model assignment"
            )
        for index, value in enumerate(solution):
            model.AddHint(
                model.GetBoolVarFromProtoIndex(index), int(value)
            )

    hard_phase = phases[0]
    if hard_phase != RoutePlanObjectivePhase(
        "hard_feasibility", "OPTIMAL", 0, 0
    ):
        raise RouteResourcePlanVerificationError(
            "independent objective-prefix replay requires the exact OPTIMAL "
            "hard-feasibility phase"
        )
    model.ClearObjective()
    try:
        feasibility_solver, _ = solve_phase(
            "hard_feasibility",
            fixed_search=True,
            presolve=False,
            probing_level=0,
            feasibility_proof=True,
        )
    except RouteResourcePlanVerificationBudgetExhausted:
        # Mirror the selector: only an UNKNOWN lean solve reaches the default
        # proof configuration, and it may consume only the shared remainder.
        feasibility_solver, _ = solve_phase(
            "hard_feasibility_proof",
            fixed_search=True,
            feasibility_proof=True,
        )
    feasibility_area = int(feasibility_solver.Value(area_expression))
    install_assignment_hints(feasibility_solver)
    model.Add(area_expression <= feasibility_area)

    if len(phases) >= 2 and phases[1].status == "FEASIBLE":
        selected_area, _selected_count = _selected_objective_values(
            pool, selected_nets
        )
        if selected_area > feasibility_area:
            raise RouteResourcePlanVerificationError(
                "optimization-incomplete area incumbent exceeds the "
                "independently replayed hard-witness upper bound"
            )

    if len(phases) >= 2 and phases[1].status == "OPTIMAL":
        area_phase = phases[1]
        model.Minimize(area_expression)
        area_solver, _ = solve_phase(
            "new_reserved_geometry_area_nm2",
            presolve=False,
            probing_level=0,
        )
        area_value = int(area_solver.Value(area_expression))
        if (
            area_phase.value != area_value
            or area_phase.lower_bound != area_value
        ):
            raise RouteResourcePlanVerificationError(
                "independent objective-prefix replay area value/lower bound "
                "does not match the proved optimum"
            )
        model.Add(area_expression == area_value)
        install_assignment_hints(area_solver)

    if len(phases) >= 3 and phases[2].status == "OPTIMAL":
        count_phase = phases[2]
        model.Minimize(count_expression)
        count_solver, _ = solve_phase(
            "landing_site_cut_count",
            presolve=False,
            probing_level=0,
        )
        count_value = int(count_solver.Value(count_expression))
        if (
            count_phase.value != count_value
            or count_phase.lower_bound != count_value
        ):
            raise RouteResourcePlanVerificationError(
                "independent objective-prefix replay landing-site count "
                "value/lower bound does not match the proved optimum"
            )
        model.Add(count_expression == count_value)
        install_assignment_hints(count_solver)

    # An incomplete certificate cannot claim an OPTIMAL tie phase.  If the
    # terminal phase is FEASIBLE/UNKNOWN, its status is intentionally not
    # upgraded by replay; exact materializability is proved below by fixing the
    # entire signed assignment in a fresh complete model.
    if len(phases) >= 4 and phases[3].status == "OPTIMAL":
        raise RouteResourcePlanVerificationError(
            "incomplete objective certificate cannot claim an OPTIMAL "
            "canonical semantic tie"
        )
    return remaining_budget


def _verify_route_plan_objective_certificate(
    pool: CandidatePool,
    plan: RouteResourcePlan,
    *,
    max_deterministic_time: Optional[float],
) -> None:
    certificate = plan.objective_certificate
    if certificate is None:
        if plan.schema == "clay.route-resource-plan.v4":
            raise RouteResourcePlanVerificationError(
                "legacy v4 route plan is retired from canonical verification; "
                "an independent objective certificate is required"
            )
        raise RouteResourcePlanVerificationError(
            "signed route plan is missing its objective certificate"
        )
    if plan.schema not in {
        "clay.route-resource-plan.v9",
        "clay.route-resource-plan.v10",
    }:
        raise RouteResourcePlanVerificationError(
            "route resource plan schema is not canonical v9/v10"
        )
    if (
        getattr(certificate, "schema", None)
        != "clay.route-plan-objective-certificate.v2"
    ):
        raise RouteResourcePlanVerificationError(
            "route-plan objective certificate schema is not canonical v2"
        )
    pool_hash = pool.sha256()
    if certificate.candidate_pool_hash != pool_hash:
        raise RouteResourcePlanVerificationError(
            "route-plan objective certificate candidate pool hash mismatch"
        )
    if certificate.objective_policy_digest != route_plan_objective_policy_digest():
        raise RouteResourcePlanVerificationError(
            "route-plan objective policy digest mismatch"
        )
    expected_decision_digest = _canonical_decision_digest(plan.selected_nets)
    if certificate.canonical_decision_digest != expected_decision_digest:
        raise RouteResourcePlanVerificationError(
            "route-plan objective canonical decision digest mismatch"
        )
    signed_budget = certificate.authorized_replay_deterministic_time
    if (
        isinstance(signed_budget, bool)
        or not isinstance(signed_budget, (int, float))
        or not math.isfinite(float(signed_budget))
        or float(signed_budget) < 0
        or float(signed_budget)
        > ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME
    ):
        raise RouteResourcePlanVerificationError(
            "route-plan objective replay budget is outside the trusted hard limit"
        )
    signed_budget = float(signed_budget)
    if max_deterministic_time is None:
        replay_budget = signed_budget
    else:
        if isinstance(max_deterministic_time, bool) or not isinstance(
            max_deterministic_time, (int, float)
        ):
            raise RouteResourcePlanVerificationError(
                "objective replay caller budget cap must be numeric"
            )
        caller_cap = float(max_deterministic_time)
        if caller_cap < 0 or not math.isfinite(caller_cap):
            raise RouteResourcePlanVerificationError(
                "objective replay caller budget cap must be finite and non-negative"
            )
        if caller_cap < signed_budget:
            raise RouteResourcePlanVerificationBudgetExhausted(
                "independent objective replay BUDGET_EXHAUSTED: caller cap is "
                "below plan-signed selection proof budget"
            )
        # Reuse exactly the selector's signed deterministic budget. A larger
        # caller cap authorizes the replay but does not perturb its proof search.
        replay_budget = signed_budget
    phases = certificate.phases
    try:
        if type(phases) is not tuple:
            raise CandidatePoolError(
                "objective phases must be stored as a canonical tuple"
            )
        if not phases or len(phases) > len(_ROUTE_PLAN_OBJECTIVE_PHASE_IDS):
            raise CandidatePoolError(
                "objective phase count is outside the canonical prefix bound"
            )
        # Frozen dataclasses can still be forged through object.__setattr__.
        # Reconstruct every phase so verification does not trust construction-
        # time validation that may have been bypassed after signing.
        replay_checked_phases = tuple(
            RoutePlanObjectivePhase(
                phase.phase_id,
                phase.status,
                phase.value,
                phase.lower_bound,
            )
            for phase in phases
        )
        phase_ids = tuple(phase.phase_id for phase in replay_checked_phases)
        if phase_ids != _ROUTE_PLAN_OBJECTIVE_PHASE_IDS[:len(phase_ids)]:
            raise CandidatePoolError(
                "objective phases are not a canonical ordered prefix"
            )
        if any(
            phase.status != "OPTIMAL"
            for phase in replay_checked_phases[:-1]
        ):
            raise CandidatePoolError(
                "all preceding objective phases must be OPTIMAL"
            )
    except (AttributeError, CandidatePoolError, TypeError, ValueError) as exc:
        raise RouteResourcePlanVerificationError(
            f"route-plan objective phase structure is invalid: {exc}"
        ) from exc
    phases = replay_checked_phases
    optimization_incomplete = (
        len(phases) < len(_ROUTE_PLAN_OBJECTIVE_PHASE_IDS)
        or phases[-1].status != "OPTIMAL"
    )
    if optimization_incomplete:
        if (
            len(phases) < 2
            or phases[0]
            != RoutePlanObjectivePhase(
                "hard_feasibility", "OPTIMAL", 0, 0
            )
            or phases[-1].status not in {"FEASIBLE", "UNKNOWN"}
        ):
            raise RouteResourcePlanVerificationError(
                "optimization-incomplete certificate lacks the exact hard "
                "feasibility boundary"
            )
        selected_area, selected_landing_count = (
            _selected_objective_values(pool, plan.selected_nets)
        )
        area_phase = phases[1]
        if (
            area_phase.phase_id != "new_reserved_geometry_area_nm2"
            or area_phase.status == "UNKNOWN"
            or area_phase.value != selected_area
        ):
            raise RouteResourcePlanVerificationError(
                "optimization-incomplete area incumbent differs from selected "
                "geometry"
            )
        if len(phases) >= 3:
            count_phase = phases[2]
            if (
                count_phase.phase_id != "landing_site_cut_count"
                or count_phase.status == "UNKNOWN"
                or count_phase.value != selected_landing_count
            ):
                raise RouteResourcePlanVerificationError(
                    "optimization-incomplete cut-count incumbent differs from "
                    "selected geometry"
                )
        remaining_replay_budget = (
            _independently_replay_claimed_optimal_objective_prefix(
                pool,
                plan.selected_nets,
                phases,
                additional_landing_site_conflicts=(
                    plan.lazy_landing_site_conflicts
                ),
                max_deterministic_time=replay_budget,
            )
        )
        _independently_replay_fixed_route_plan_feasibility(
            pool,
            plan.selected_nets,
            additional_landing_site_conflicts=(
                plan.lazy_landing_site_conflicts
            ),
            max_deterministic_time=remaining_replay_budget,
        )
        return
    canonical_selected, area_value, landing_count = (
        _independently_replay_route_plan_objective(
            pool,
            additional_landing_site_conflicts=(
                plan.lazy_landing_site_conflicts
            ),
            max_deterministic_time=replay_budget,
        )
    )
    if canonical_selected != plan.selected_nets:
        raise RouteResourcePlanVerificationError(
            "independent objective replay selected a different optimal/canonical plan"
        )
    expected_phases = (
        RoutePlanObjectivePhase("hard_feasibility", "OPTIMAL", 0, 0),
        RoutePlanObjectivePhase(
            "new_reserved_geometry_area_nm2",
            "OPTIMAL",
            area_value,
            area_value,
        ),
        RoutePlanObjectivePhase(
            "landing_site_cut_count",
            "OPTIMAL",
            landing_count,
            landing_count,
        ),
        RoutePlanObjectivePhase(
            "canonical_semantic_tie", "OPTIMAL", None, None
        ),
    )
    if certificate.phases != expected_phases:
        raise RouteResourcePlanVerificationError(
            "route-plan objective phase values do not replay from selected geometry"
        )


def verify_route_resource_plan(
    pool: CandidatePool,
    plan: RouteResourcePlan,
    *,
    max_deterministic_time: Optional[float] = None,
) -> bool:
    if not isinstance(pool, CandidatePool) or not isinstance(plan, RouteResourcePlan):
        raise RouteResourcePlanVerificationError("pool and plan types must match")
    try:
        rebuilt_pool = _replay_candidate_pool_runtime(pool)
    except CandidatePoolError as exc:
        raise RouteResourcePlanVerificationError(
            f"closed allocation IR failed verifier-entry replay: {exc}"
        ) from exc
    pool = rebuilt_pool
    bounded_selected_nets = _bounded_materialize(
        plan.selected_nets,
        limit=MAX_REQUIRED_NETS,
        label="selected net",
        error_type=CandidatePoolLimitError,
    )
    bounded_allocated_programs = _bounded_materialize(
        plan.allocated_programs,
        limit=MAX_REQUIRED_NETS,
        label="allocated net program",
        error_type=CandidatePoolLimitError,
    )
    rebuilt_plan = replace(
        plan,
        selected_nets=tuple(
            _rebuild_selected_net_candidate(selected)
            for selected in bounded_selected_nets
        ),
        allocated_programs=tuple(
            _rebuild_allocated_net_program(program)
            for program in bounded_allocated_programs
        ),
    )
    if rebuilt_plan != plan:
        raise RouteResourcePlanVerificationError(
            "route resource plan is not canonical after replay"
        )
    plan = rebuilt_plan
    pool_hash = pool.sha256()
    if plan.candidate_pool_hash != pool_hash:
        raise RouteResourcePlanVerificationError("candidate pool hash mismatch")
    expected_index_certificate = _owned_conductor_index_certificate_from_pool(
        pool
    )
    if plan.owned_conductor_index_certificate != expected_index_certificate:
        raise RouteResourcePlanVerificationError(
            "route plan owned-conductor index certificate does not byte-exact "
            "match the candidate pool"
        )
    if expected_index_certificate is not None:
        try:
            validate_owned_conductor_component_index_certificate(
                world_digest=expected_index_certificate.world_digest,
                components=tuple(
                    certificate.component
                    for certificate in expected_index_certificate.components
                ),
                index_digest=expected_index_certificate.index_digest,
            )
        except OwnedConductorIndexError as exc:
            raise RouteResourcePlanVerificationError(
                f"route plan owned-conductor index replay failed: {exc}"
            ) from exc
    conflict_domain = pool.landing_site_conflict_domain
    if conflict_domain is None:
        if (
            plan.landing_site_conflict_domain_digest
            or plan.lazy_landing_site_conflicts
        ):
            raise RouteResourcePlanVerificationError(
                "route plan claims an absent landing conflict domain"
            )
    else:
        if (
            plan.landing_site_conflict_domain_digest
            != conflict_domain.sha256()
        ):
            raise RouteResourcePlanVerificationError(
                "landing conflict domain digest mismatch"
            )
        sites_by_id_for_domain = {
            site.site_id: site for site in pool.iter_landing_sites()
        }
        for conflict in plan.lazy_landing_site_conflicts:
            left = sites_by_id_for_domain.get(conflict.left_site_id)
            right = sites_by_id_for_domain.get(conflict.right_site_id)
            if (
                left is None
                or right is None
                or _derive_landing_site_pair_conflict_from_domain(
                    left, right, conflict_domain
                )
                != conflict
            ):
                raise RouteResourcePlanVerificationError(
                    "lazy landing conflict does not replay from bound domain"
                )
    closed_candidate = next(
        (
            candidate
            for candidate in pool.all_candidates
            if candidate_has_closed_allocation_ir(candidate)
        ),
        None,
    )
    if closed_candidate is not None:
        try:
            verify_candidate_allocation_ir(pool, closed_candidate)
        except CandidatePoolError as exc:
            raise RouteResourcePlanVerificationError(
                f"closed allocation IR failed independent replay: {exc}"
            ) from exc
    selected_by_net = {selected.net: selected for selected in plan.selected_nets}
    if len(selected_by_net) != len(plan.selected_nets) or set(selected_by_net) != set(pool.required_nets):
        raise RouteResourcePlanVerificationError("plan must select exactly one candidate per required net")
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in pool.all_candidates
    }
    resources_by_key = {resource.key: resource for resource in pool.resources}
    landing_sites_by_id = pool.landing_sites_by_id()
    component_certificates_by_ref = {
        certificate.component.component_ref: certificate
        for certificate in pool.owned_conductor_components
    }
    for program in plan.allocated_programs:
        for access, component_ref in _endpoint_component_requirements(program):
            certificate = component_certificates_by_ref.get(component_ref)
            component = None if certificate is None else certificate.component
            if component_ref not in access.owner_component_refs:
                raise RouteResourcePlanVerificationError(
                    f"selected owned-component endpoint "
                    f"{access.provenance!r} must name the traversed component "
                    "in owner refs"
                )
            if (
                certificate is None
                or component is None
                or certificate.layer_n != access.layer_n
                or component.layer_key != access.layer_key
                or component.net != program.net
                or not _owned_component_conductor_contains_point(
                    component, access.xy_nm
                )
            ):
                raise RouteResourcePlanVerificationError(
                    f"selected owned-component endpoint "
                    f"{access.provenance!r} lacks exact atom geometry"
                )
    selected_candidates: list[CompleteNetCandidate] = []
    selected_track_by_use: dict[tuple[str, str], SelectedTrack] = {}
    selected_landings: list[SelectedLanding] = []
    selected_landing_nets: dict[str, set[str]] = {}
    selected_landing_candidate_ids: dict[str, set[str]] = {}
    for net in pool.required_nets:
        selected = selected_by_net[net]
        candidate = candidates_by_id.get(selected.candidate_id)
        if candidate is None or candidate.net != net:
            raise RouteResourcePlanVerificationError(
                f"selected candidate {selected.candidate_id!r} does not belong to net {net!r}"
            )
        graph_error = _branch_graph_admissibility_error(candidate)
        if graph_error is not None:
            raise RouteResourcePlanVerificationError(
                f"selected candidate {candidate.candidate_id!r} {graph_error}"
            )
        endpoints = {
            endpoint for edge in candidate.branch_edges for endpoint in edge
        }
        for goal in (
            goal for goal in pool.required_route_goals if goal.net == candidate.net
        ):
            if (
                goal.declaration_ref not in candidate.required_goal_refs
                or goal.target is None
                or goal.target.provenance not in endpoints
            ):
                raise RouteResourcePlanVerificationError(
                    f"selected candidate {candidate.candidate_id!r} does not "
                    f"physically cover required goal {goal.declaration_ref!r}"
                )
        selected_candidates.append(candidate)
        tracks_by_use = {track.use_id: track for track in selected.selected_tracks}
        if len(tracks_by_use) != len(selected.selected_tracks) or set(tracks_by_use) != {
            use.use_id for use in candidate.corridor_uses
        }:
            raise RouteResourcePlanVerificationError(
                f"selected tracks do not match candidate {candidate.candidate_id!r} uses"
            )
        for use in candidate.corridor_uses:
            track = tracks_by_use[use.use_id]
            if track.resource != use.resource or track.track_start not in use.allowed_track_starts:
                raise RouteResourcePlanVerificationError(
                    f"invalid selected track for use {use.use_id!r}"
                )
            resource = resources_by_key.get(track.resource)
            if resource is None:
                raise RouteResourcePlanVerificationError(
                    f"selected track resource {track.resource!r} is absent from candidate pool"
                )
            if not _track_matches_pool_world_binding(
                resource, pool.routing_world_provenance_digest
            ):
                binding = (
                    "world-bound"
                    if pool.routing_world_provenance_digest
                    else "unbound"
                )
                raise RouteResourcePlanVerificationError(
                    f"selected {binding} track resource has invalid "
                    "geometry/provenance binding"
                )
            if (
                resource.through_interval_nm is not None
                and not (
                    resource.through_interval_nm[0] <= use.interval_nm[0]
                    and use.interval_nm[1] <= resource.through_interval_nm[1]
                )
            ):
                raise RouteResourcePlanVerificationError(
                    f"selected track resource does not cover use {use.use_id!r}"
                )
            if track.track_start + use.track_span > len(
                resource.track_centers_nm
            ):
                raise RouteResourcePlanVerificationError(
                    f"selected track span exceeds resource for use {use.use_id!r}"
                )
            if resource.corridor_bbox_nm is not None:
                selected_bbox = _track_choice_closed_bbox(
                    use,
                    resource,
                    track.track_start,
                    _corridor_use_width_nm(use, candidate.width_nm),
                )
                if selected_bbox is None or not _bbox_contains(
                    resource.corridor_bbox_nm, selected_bbox
                ):
                    raise RouteResourcePlanVerificationError(
                        f"selected candidate width does not fit corridor for "
                        f"use {use.use_id!r}"
                    )
            selected_track_by_use[(candidate.candidate_id, use.use_id)] = track
        if candidate_has_closed_allocation_ir(candidate):
            uses_by_id = {
                use.use_id: use for use in candidate.corridor_uses
            }
            for access, use_id in _endpoint_track_requirements(candidate):
                use = uses_by_id[use_id]
                track = tracks_by_use[use_id]
                resource = resources_by_key[use.resource]
                if not _track_choice_contains_endpoint(
                    access, use, resource, track.track_start
                ):
                    raise RouteResourcePlanVerificationError(
                        f"selected endpoint {access.provenance!r} does not "
                        "touch its boundary track geometry"
                    )
            for left_use, right_use in _consecutive_track_step_pairs(candidate):
                if not _track_choices_physically_touch(
                    left_use,
                    resources_by_key[left_use.resource],
                    tracks_by_use[left_use.use_id].track_start,
                    right_use,
                    resources_by_key[right_use.resource],
                    tracks_by_use[right_use.use_id].track_start,
                    _corridor_use_width_nm(left_use, candidate.width_nm),
                    _corridor_use_width_nm(right_use, candidate.width_nm),
                ):
                    raise RouteResourcePlanVerificationError(
                        "selected consecutive track steps are not electrically "
                        "continuous"
                    )
            for component_ref, use in _component_track_step_pairs(candidate):
                certificate = component_certificates_by_ref[component_ref]
                component = certificate.component
                resource = resources_by_key[use.resource]
                track = tracks_by_use[use.use_id]
                track_bbox = _track_choice_closed_bbox(
                    use,
                    resource,
                    track.track_start,
                    _corridor_use_width_nm(use, candidate.width_nm),
                )
                if (
                    certificate.layer_n != resource.key.layer_n
                    or component.layer_key != resource.gds_layer_key
                    or track_bbox is None
                    or not _owned_component_conductor_touches_track_bbox(
                        component, track_bbox
                    )
                ):
                    raise RouteResourcePlanVerificationError(
                        "selected track does not electrically touch adjacent "
                        f"owned component {component_ref!r}"
                    )
            derived_demands_by_id = {
                demand.demand_id: demand
                for demand in candidate.landing_demands
                if (
                    demand.selection_mode
                    == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                )
            }
            derived_selected_by_demand: dict[str, list[SelectedLanding]] = {}
            derived_pairs = tuple(
                (landing.demand_id, landing.site_id)
                for landing in selected.selected_landings
            )
            for landing in selected.selected_landings:
                if landing.demand_id not in derived_demands_by_id:
                    raise RouteResourcePlanVerificationError(
                        "closed candidate cannot mix explicit options with a "
                        "legacy/non-derived selected landing"
                    )
                derived_selected_by_demand.setdefault(
                    landing.demand_id, []
                ).append(landing)
            if len(derived_pairs) != len(set(derived_pairs)):
                raise RouteResourcePlanVerificationError(
                    "derived selected landing sites must be unique per demand"
                )
            selected_track_starts = {
                use_id: track.track_start
                for use_id, track in tracks_by_use.items()
            }
            for demand_id, demand in derived_demands_by_id.items():
                demand_landings = derived_selected_by_demand.get(demand_id, [])
                if len(demand_landings) != demand.required_multiplicity:
                    raise RouteResourcePlanVerificationError(
                        f"derived landing cardinality does not match demand "
                        f"{demand_id!r}"
                    )
                for landing in demand_landings:
                    if landing.site_id not in demand.allowed_site_ids:
                        raise RouteResourcePlanVerificationError(
                            f"derived landing site {landing.site_id!r} is outside "
                            f"demand {demand_id!r}"
                        )
                    site = landing_sites_by_id.get(landing.site_id)
                    if site is None:
                        raise RouteResourcePlanVerificationError(
                            f"derived landing site {landing.site_id!r} is absent"
                        )
                    if not _landing_matches_pool_world_binding(
                        site, pool.routing_world_provenance_digest
                    ) or site.owner_net not in (None, candidate.net):
                        raise RouteResourcePlanVerificationError(
                            f"derived landing site {landing.site_id!r} has invalid "
                            "world/owner binding"
                        )
                    error = _derived_landing_site_domain_error(
                        candidate,
                        demand_id,
                        site,
                        resources_by_key=resources_by_key,
                        selected_track_starts=selected_track_starts,
                    )
                    if error is not None:
                        raise RouteResourcePlanVerificationError(
                            f"derived landing site {landing.site_id!r} does not "
                            f"replay: {error}"
                        )
                    selected_landings.append(landing)
                    selected_landing_nets.setdefault(
                        landing.site_id, set()
                    ).add(candidate.net)
                    selected_landing_candidate_ids.setdefault(
                        landing.site_id, set()
                    ).add(candidate.candidate_id)
            selected_option_pairs = tuple(
                (option.demand_id, option.option_id)
                for option in selected.selected_landing_options
            )
            selected_options_by_demand: dict[
                str, list[SelectedLandingOption]
            ] = {}
            for selected_option in selected.selected_landing_options:
                selected_options_by_demand.setdefault(
                    selected_option.demand_id, []
                ).append(selected_option)
            if (
                len(selected_option_pairs) != len(set(selected_option_pairs))
                or set(selected_options_by_demand)
                != {
                    demand.demand_id
                    for demand in candidate.landing_demands
                    if (
                        demand.selection_mode
                        != LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                    )
                }
            ):
                raise RouteResourcePlanVerificationError(
                    "selected landing options do not match closed candidate demands"
                )
            options_by_id = {
                option.option_id: option for option in candidate.landing_options
            }
            neighbour_use_ids_by_demand = _landing_neighbour_use_ids(candidate)
            for demand in candidate.landing_demands:
                if (
                    demand.selection_mode
                    == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                ):
                    continue
                selected_options = selected_options_by_demand[demand.demand_id]
                expected_count = (
                    demand.required_multiplicity
                    if demand.selection_mode
                    == LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY
                    else 1
                )
                if len(selected_options) != expected_count:
                    raise RouteResourcePlanVerificationError(
                        f"selected landing option cardinality does not match "
                        f"demand {demand.demand_id!r}"
                    )
                selected_site_ids_for_demand: set[str] = set()
                for selected_option in selected_options:
                    option = options_by_id.get(selected_option.option_id)
                    if option is None or option.demand_id != demand.demand_id:
                        raise RouteResourcePlanVerificationError(
                            f"invalid landing option for demand {demand.demand_id!r}"
                        )
                    for contact in option.contacts:
                        track = tracks_by_use.get(contact.use_id)
                        if track is None or track.track_start != contact.track_start:
                            raise RouteResourcePlanVerificationError(
                                "selected landing option contact does not match "
                                "exact track"
                            )
                    for use_id in neighbour_use_ids_by_demand[demand.demand_id]:
                        use = uses_by_id[use_id]
                        track = tracks_by_use[use_id]
                        if not _track_choice_touches_landing_option(
                            candidate=candidate,
                            option=option,
                            use=use,
                            resource=resources_by_key[use.resource],
                            track_start=track.track_start,
                            sites_by_id=landing_sites_by_id,
                            certificates_by_ref=component_certificates_by_ref,
                        ):
                            raise RouteResourcePlanVerificationError(
                                "selected landing option does not electrically touch "
                                f"adjacent use {use_id!r}"
                            )
                    for site_id in option.site_ids:
                        if site_id in selected_site_ids_for_demand:
                            raise RouteResourcePlanVerificationError(
                                "selected factorized landing options overlap one site"
                            )
                        selected_site_ids_for_demand.add(site_id)
                        site = landing_sites_by_id.get(site_id)
                        if site is None:
                            raise RouteResourcePlanVerificationError(
                                f"landing site {site_id!r} is absent from candidate pool"
                            )
                        if not _landing_matches_pool_world_binding(
                            site, pool.routing_world_provenance_digest
                        ):
                            raise RouteResourcePlanVerificationError(
                                "selected world-bound landing site has invalid "
                                "geometry/provenance binding"
                            )
                        if site.owner_net not in (None, candidate.net):
                            raise RouteResourcePlanVerificationError(
                                f"landing site {site_id!r} has foreign owner "
                                f"{site.owner_net!r} for net {candidate.net!r}"
                            )
                        landing = SelectedLanding(
                            demand_id=demand.demand_id, site_id=site_id
                        )
                        selected_landings.append(landing)
                        selected_landing_nets.setdefault(site_id, set()).add(
                            candidate.net
                        )
                        selected_landing_candidate_ids.setdefault(
                            site_id, set()
                        ).add(candidate.candidate_id)
        else:
            if selected.selected_landing_options:
                raise RouteResourcePlanVerificationError(
                    "legacy candidate cannot select landing options"
                )
            landings_by_demand = {
                landing.demand_id: landing for landing in selected.selected_landings
            }
            if len(landings_by_demand) != len(selected.selected_landings) or set(landings_by_demand) != {
                demand.demand_id for demand in candidate.landing_demands
            }:
                raise RouteResourcePlanVerificationError(
                    f"selected landings do not match candidate {candidate.candidate_id!r} demands"
                )
            for demand in candidate.landing_demands:
                landing = landings_by_demand[demand.demand_id]
                if landing.site_id not in demand.allowed_site_ids:
                    raise RouteResourcePlanVerificationError(
                        f"invalid landing site for demand {demand.demand_id!r}"
                    )
                site = landing_sites_by_id.get(landing.site_id)
                if site is None:
                    raise RouteResourcePlanVerificationError(
                        f"landing site {landing.site_id!r} is absent from candidate pool"
                    )
                if not _landing_matches_pool_world_binding(
                    site, pool.routing_world_provenance_digest
                ):
                    binding = (
                        "world-bound"
                        if pool.routing_world_provenance_digest
                        else "unbound"
                    )
                    raise RouteResourcePlanVerificationError(
                        f"selected {binding} landing site has invalid "
                        "geometry/provenance binding"
                    )
                if site.owner_net not in (None, candidate.net):
                    raise RouteResourcePlanVerificationError(
                        f"landing site {landing.site_id!r} has foreign owner "
                        f"{site.owner_net!r} for net {candidate.net!r}"
                    )
                selected_landings.append(landing)
                selected_landing_nets.setdefault(landing.site_id, set()).add(
                    candidate.net
                )
                selected_landing_candidate_ids.setdefault(
                    landing.site_id, set()
                ).add(candidate.candidate_id)

    for index, left_candidate in enumerate(selected_candidates):
        for left_use in left_candidate.corridor_uses:
            left_track = selected_track_by_use[(left_candidate.candidate_id, left_use.use_id)]
            for right_candidate in selected_candidates[index:]:
                for right_use in right_candidate.corridor_uses:
                    if (
                        left_candidate is right_candidate
                        and left_use.use_id >= right_use.use_id
                    ):
                        continue
                    if left_use.resource != right_use.resource or not _intervals_overlap(
                        left_use.interval_nm, right_use.interval_nm
                    ):
                        continue
                    right_track = selected_track_by_use[(right_candidate.candidate_id, right_use.use_id)]
                    if _same_resource_track_choices_overlap(
                        left_use,
                        left_track.track_start,
                        right_use,
                        right_track.track_start,
                        resources_by_key[left_use.resource],
                    ):
                        raise RouteResourcePlanVerificationError(
                            f"overlapping track spans for uses {left_use.use_id!r} and {right_use.use_id!r}"
                        )

    for (left_candidate, left_use), (
        right_candidate,
        right_use,
    ) in _iter_physical_track_use_pairs(
        tuple(selected_candidates), resources_by_key
    ):
        left_track = selected_track_by_use[
            (left_candidate.candidate_id, left_use.use_id)
        ]
        right_track = selected_track_by_use[
            (right_candidate.candidate_id, right_use.use_id)
        ]
        if _physical_track_choices_conflict(
            left_use,
            left_track.track_start,
            resources_by_key[left_use.resource],
            right_use,
            right_track.track_start,
            resources_by_key[right_use.resource],
            same_net=left_candidate.net == right_candidate.net,
        ):
            raise RouteResourcePlanVerificationError(
                "physical metal overlap/spacing conflict for uses "
                f"{left_use.use_id!r} and {right_use.use_id!r}"
            )

    chosen_ids = {candidate.candidate_id for candidate in selected_candidates}
    for conflict in pool.conflicts:
        if {conflict.left_candidate_id, conflict.right_candidate_id} <= chosen_ids:
            raise RouteResourcePlanVerificationError("selected candidates violate pair conflict")
    site_capacity = {
        site.site_id: site.capacity for site in pool.iter_landing_sites()
    }
    counts: dict[str, int] = {}
    for landing in selected_landings:
        counts[landing.site_id] = counts.get(landing.site_id, 0) + 1
    for site_id, count in counts.items():
        if count > site_capacity[site_id]:
            raise RouteResourcePlanVerificationError(
                f"landing site {site_id!r} capacity exceeded"
            )
    for conflict in pool.landing_site_conflicts:
        left_nets = selected_landing_nets.get(conflict.left_site_id, set())
        right_nets = selected_landing_nets.get(conflict.right_site_id, set())
        if not left_nets or not right_nets:
            continue
        if conflict.conflict_scope == "all_nets" or any(
            left_net != right_net
            for left_net in left_nets
            for right_net in right_nets
        ):
            raise RouteResourcePlanVerificationError(
                "selected landing sites violate a site-pair conflict"
            )
        if conflict.conflict_scope != "same_net_bridge":
            continue
        common_candidate_ids = (
            selected_landing_candidate_ids.get(conflict.left_site_id, set())
            & selected_landing_candidate_ids.get(conflict.right_site_id, set())
        )
        if not common_candidate_ids:
            raise RouteResourcePlanVerificationError(
                "same-net close landing pads lack one selected candidate bridge"
            )
        left_site = landing_sites_by_id[conflict.left_site_id]
        right_site = landing_sites_by_id[conflict.right_site_id]
        for candidate_id in common_candidate_ids:
            candidate = candidates_by_id[candidate_id]
            if any(
                _track_choice_bridges_landing_pair(
                    candidate,
                    use,
                    resources_by_key[use.resource],
                    selected_track_by_use[(candidate_id, use.use_id)].track_start,
                    left_site,
                    right_site,
                )
                for use in candidate.corridor_uses
            ):
                break
        else:
            raise RouteResourcePlanVerificationError(
                "same-net close landing pads lack an exact selected metal bridge"
            )
    if _selected_domain_landing_conflicts(pool, plan.selected_nets):
        raise RouteResourcePlanVerificationError(
            "selected landing assignments violate compressed conflict domain"
        )
    if pool.routing_world_provenance_digest:
        expected_ledger_hash = (
            explicit_routing_resource_ledger_sha256(
                pool.routing_world_provenance_digest,
                pool.resources,
                pool.landing_sites,
                pool.landing_site_conflicts,
                pool.owned_conductor_index_digest,
                pool.landing_site_domain_digest,
                pool.landing_sites_complete,
                pool.landing_site_conflict_domain,
            )
            if pool.landing_site_catalog is None
            else paged_routing_resource_ledger_sha256(
                pool.routing_world_provenance_digest,
                pool.resources,
                pool.landing_site_catalog,
                pool.landing_site_conflicts,
                pool.owned_conductor_index_digest,
                pool.landing_site_domain_digest,
                pool.landing_sites_complete,
                pool.landing_site_conflict_domain,
            )
        )
        if pool.routing_resource_ledger_hash != expected_ledger_hash:
            raise RouteResourcePlanVerificationError(
                "routing resource ledger hash mismatch during plan replay"
            )
    expected_programs = tuple(
        _resolve_allocated_net_program(
            pool,
            candidates_by_id[selected.candidate_id],
            selected,
            pool_hash,
        )
        for selected in plan.selected_nets
        if candidate_has_closed_allocation_ir(
            candidates_by_id[selected.candidate_id]
        )
    )
    expected_complete = all(
        candidate_has_closed_allocation_ir(
            candidates_by_id[selected.candidate_id]
        )
        for selected in plan.selected_nets
    )
    if (
        plan.allocated_programs != expected_programs
        or plan.materialization_complete != expected_complete
    ):
        raise RouteResourcePlanVerificationError(
            "resolved allocated program does not byte-exact replay from pool and selection"
        )
    _verify_route_plan_objective_certificate(
        pool,
        plan,
        max_deterministic_time=max_deterministic_time,
    )
    return True


def certify_route_resource_plan_verification(
    pool: CandidatePool,
    plan: RouteResourcePlan,
    *,
    max_deterministic_time: Optional[float] = None,
) -> RoutePlanVerificationReceipt:
    """Run the public deep verifier, then issue an exact runtime receipt."""

    verify_route_resource_plan(
        pool,
        plan,
        max_deterministic_time=max_deterministic_time,
    )
    return _issue_route_plan_verification_receipt(pool, plan)


def verify_infeasibility_certificate(pool: CandidatePool, certificate: object) -> bool:
    """Replay an exact simple pool witness without inferring placement causality.

    Only the two independently recomputable proof forms are accepted.  A generic
    coupled CP-SAT UNSAT result remains an honest pool-scoped diagnostic, but it is
    not promoted to a replay proof and is never described as a minimal IIS.
    """

    if not isinstance(pool, CandidatePool):
        raise InfeasibilityCertificateVerificationError("pool must be CandidatePool")
    try:
        pool = _replay_candidate_pool_runtime(pool)
    except CandidatePoolError as exc:
        raise InfeasibilityCertificateVerificationError(
            f"candidate pool runtime replay failed: {exc}"
        ) from exc
    if not isinstance(certificate, (CapacityConflict, HallDeficit, PoolUnsatCertificate)):
        raise InfeasibilityCertificateVerificationError(
            f"foreign certificate type {type(certificate).__name__}"
        )
    if certificate.candidate_pool_hash != pool.sha256():
        raise InfeasibilityCertificateVerificationError(
            "candidate pool hash mismatch"
        )
    if isinstance(certificate, CapacityConflict):
        expected = _unavoidable_capacity_conflict(pool)
        if expected != certificate:
            raise InfeasibilityCertificateVerificationError(
                "capacity conflict does not replay from explicit pool tracks and uses"
            )
        return True
    if isinstance(certificate, HallDeficit):
        expected = _unavoidable_hall_deficit(pool)
        if expected != certificate:
            raise InfeasibilityCertificateVerificationError(
                "Hall deficit does not replay from the fixed landing graph"
            )
        return True
    raise InfeasibilityCertificateVerificationError(
        "generic CP UNSAT certificate is not replayable as an independent proof"
    )


def verify_selection_certificate(pool: CandidatePool, certificate: object) -> bool:
    """Verify the proof strength actually carried by a selection certificate.

    Capacity and Hall certificates replay as independent exact witnesses.  A
    PoolUnsatCertificate carries less information: this function checks its pool
    scope and reason, then rebuilds the model and solves only the named sufficient
    required-net assumptions.  Success for that form must not be interpreted as a
    local conflict proof or a minimal IIS.
    """

    if not isinstance(pool, CandidatePool):
        raise InfeasibilityCertificateVerificationError("pool must be CandidatePool")
    try:
        pool = _replay_candidate_pool_runtime(pool)
    except CandidatePoolError as exc:
        raise InfeasibilityCertificateVerificationError(
            f"candidate pool runtime replay failed: {exc}"
        ) from exc
    if isinstance(certificate, (CapacityConflict, HallDeficit)):
        return verify_infeasibility_certificate(pool, certificate)
    if not isinstance(certificate, PoolUnsatCertificate):
        raise InfeasibilityCertificateVerificationError(
            f"foreign certificate type {type(certificate).__name__}"
        )
    if certificate.candidate_pool_hash != pool.sha256():
        raise InfeasibilityCertificateVerificationError(
            "candidate pool hash mismatch"
        )
    if not isinstance(certificate.reason, str) or not certificate.reason:
        raise InfeasibilityCertificateVerificationError(
            "generic CP UNSAT certificate requires a non-empty reason"
        )
    core_ids = tuple(certificate.assumption_core_ids)
    if not core_ids:
        raise InfeasibilityCertificateVerificationError(
            "generic CP UNSAT certificate requires a non-empty assumption core"
        )
    if (
        any(not isinstance(core_id, str) or not core_id for core_id in core_ids)
        or core_ids != tuple(sorted(set(core_ids)))
    ):
        raise InfeasibilityCertificateVerificationError(
            "generic CP UNSAT assumption core ids must be sorted unique strings"
        )
    valid_ids = {f"required_net:{net}" for net in pool.required_nets}
    foreign_ids = sorted(set(core_ids) - valid_ids)
    if foreign_ids:
        raise InfeasibilityCertificateVerificationError(
            f"generic CP UNSAT core names foreign candidates {foreign_ids}"
        )
    try:
        cp_model = _load_cp_model()
    except (ImportError, OSError) as exc:
        raise InfeasibilityCertificateVerificationError(
            f"generic CP UNSAT core replay solver unavailable: {exc}"
        ) from exc
    model, _, _, _, _, _, _ = _build_candidate_cp_model(
        pool, cp_model, assumed_core_ids=core_ids
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_deterministic_time = 10.0
    replay_status = solver.Solve(model)
    if replay_status == cp_model.INFEASIBLE:
        return True
    status_name = (
        "SAT"
        if replay_status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        else solver.StatusName(replay_status)
    )
    raise InfeasibilityCertificateVerificationError(
        f"generic CP UNSAT core replay status {status_name}, expected INFEASIBLE"
    )


class PlacementGeometryConflict(ValueError):
    """The named placement boxes cannot form an unambiguous resource ledger."""


class LayerResourceDeclarationError(ValueError):
    """A required PDK routing-layer declaration is absent or invalid."""


@dataclass(frozen=True)
class ClusterBox:
    """One named, positive-area placement stamp in integer nanometres."""

    name: str
    bbox_nm: BBoxNm


@dataclass(frozen=True)
class FreeSpaceRegion:
    """One four-connected component of coordinate-compressed free cells."""

    region_id: str
    bbox_nm: BBoxNm
    area_nm2: int
    width_nm: int
    cells_nm: tuple[BBoxNm, ...]

    def clear_span_nm(self, preferred_axis: str) -> int:
        """Return a conservative track cross-section for ``preferred_axis``.

        Horizontal tracks consume vertical span and vertical tracks consume
        horizontal span.  Taking the minimum elementary-cell cross-section is
        intentionally conservative for non-rectangular components; it never
        turns a bounding-box overestimate into an apparent capacity guarantee.
        """

        if preferred_axis in ("H", "V"):
            return _merged_cross_section_bottleneck(self.cells_nm, preferred_axis)
        raise ValueError(f"preferred_axis must be 'H' or 'V', got {preferred_axis!r}")


@dataclass(frozen=True)
class LayerResource:
    """PDK-derived nominal routing facts for one metal layer."""

    layer_n: int
    preferred_axis: str
    pitch_nm: int
    min_width_nm: int
    min_space_nm: int

    def __post_init__(self) -> None:
        if not _is_int_nm(self.layer_n) or self.layer_n <= 0:
            raise LayerResourceDeclarationError(
                f"layer_n must be a positive integer, got {self.layer_n!r}"
            )
        if self.preferred_axis not in ("H", "V"):
            raise LayerResourceDeclarationError(
                f"metal{self.layer_n} preferred axis must be 'H' or 'V', "
                f"got {self.preferred_axis!r}"
            )
        for field_name in ("pitch_nm", "min_width_nm", "min_space_nm"):
            value = getattr(self, field_name)
            if not _is_int_nm(value) or value <= 0:
                raise LayerResourceDeclarationError(
                    f"metal{self.layer_n} {field_name} must be a positive "
                    f"integer-nm declaration, got {value!r}"
                )

    @classmethod
    def from_layout_tech(cls, tech: object, *, layer_n: int) -> "LayerResource":
        """Read axis/width/space from ``LayoutTech`` without guessing.

        ``LayoutTech.metal_preferred_axis_declared`` is the sole axis source;
        its odd/even fallback accessor is intentionally not consulted.  The
        track pitch is the minimum legal repeated-wire pitch, ``min_width +
        min_space``.  A missing accessor, exception, ``None``, non-integer rule,
        or invalid axis is reported rather than replaced by a reference-derived
        or hard-coded default.
        """

        if not _is_int_nm(layer_n) or layer_n <= 0:
            raise LayerResourceDeclarationError(
                f"layer_n must be a positive integer, got {layer_n!r}"
            )
        layer_name = f"metal{layer_n}"

        axis_fn = getattr(tech, "metal_preferred_axis_declared", None)
        if not callable(axis_fn):
            raise LayerResourceDeclarationError(
                f"{layer_name} requires LayoutTech.metal_preferred_axis_declared"
            )
        try:
            axis = axis_fn(layer_n)
        except Exception as exc:
            raise LayerResourceDeclarationError(
                f"{layer_name} preferred axis declaration is unavailable: {exc}"
            ) from exc
        if axis is None:
            raise LayerResourceDeclarationError(
                f"{layer_name} has no PDK-declared preferred axis"
            )
        if axis not in ("H", "V"):
            raise LayerResourceDeclarationError(
                f"{layer_name} preferred axis must be 'H' or 'V', got {axis!r}"
            )

        width = _read_positive_metal_rule(tech, "min_width", layer_n)
        space = _read_positive_metal_rule(tech, "min_space", layer_n)
        return cls(
            layer_n=layer_n,
            preferred_axis=axis,
            pitch_nm=width + space,
            min_width_nm=width,
            min_space_nm=space,
        )


@dataclass(frozen=True)
class CapacityObservation:
    """Nominal, demand-free track-count observation for a corridor/layer."""

    corridor_id: str
    layer_n: int
    preferred_axis: str
    clear_span_nm: int
    pitch_nm: int
    nominal_track_capacity: int


@dataclass(frozen=True)
class RoutingResourceModel:
    """Exact whitespace/corridor accounting for named placement clusters."""

    clusters: tuple[ClusterBox, ...]
    core_bbox_nm: BBoxNm
    core_area_nm2: int
    occupied_area_nm2: int
    whitespace_area_nm2: int
    utilization: Fraction
    boundary_corridors: tuple[FreeSpaceRegion, ...]
    enclosed_voids: tuple[FreeSpaceRegion, ...]

    @classmethod
    def from_clusters(
        cls, cluster_boxes: Mapping[str, Sequence[int]]
    ) -> "RoutingResourceModel":
        """Build a stable integer-nm model from non-overlapping cluster boxes."""
        return cls._from_cluster_boxes(cluster_boxes, require_disjoint=True)

    @classmethod
    def from_cluster_envelope_union(
        cls, cluster_boxes: Mapping[str, Sequence[int]]
    ) -> "RoutingResourceModel":
        """Observe the union of named envelopes, retaining their full identity.

        Envelopes may overlap or contain one another. Occupancy counts their
        union once; corridors describe its complement. This observation grants
        no physical collision or placement legality, which requires independent
        verification of the actual geometry and its native placement binding.
        """
        return cls._from_cluster_boxes(cluster_boxes, require_disjoint=False)

    @classmethod
    def _from_cluster_boxes(
        cls,
        cluster_boxes: Mapping[str, Sequence[int]],
        *,
        require_disjoint: bool,
    ) -> "RoutingResourceModel":

        try:
            raw_items = tuple(cluster_boxes.items())
        except (AttributeError, TypeError) as exc:
            raise PlacementGeometryConflict(
                "cluster boxes must be a name-to-bbox mapping"
            ) from exc
        if not raw_items:
            raise PlacementGeometryConflict(
                "resource ledger requires at least one named cluster"
            )

        clusters = tuple(
            sorted(
                (_validated_cluster(name, bbox) for name, bbox in raw_items),
                key=lambda cluster: (cluster.name, cluster.bbox_nm),
            )
        )
        if require_disjoint:
            _reject_positive_area_overlaps(clusters)

        x0 = min(cluster.bbox_nm[0] for cluster in clusters)
        y0 = min(cluster.bbox_nm[1] for cluster in clusters)
        x1 = max(cluster.bbox_nm[2] for cluster in clusters)
        y1 = max(cluster.bbox_nm[3] for cluster in clusters)
        core_bbox = (x0, y0, x1, y1)
        core_area = (x1 - x0) * (y1 - y0)
        boundary, enclosed = _free_space_components(clusters, core_bbox)
        component_area = sum(r.area_nm2 for r in boundary + enclosed)
        # Coordinate compression already classifies a cell as occupied when
        # any envelope covers it. Its exact free complement therefore counts
        # overlapping envelopes once while retaining every original cluster.
        occupied_area = (
            sum(_bbox_area(cluster.bbox_nm) for cluster in clusters)
            if require_disjoint
            else core_area - component_area
        )
        whitespace_area = core_area - occupied_area
        if component_area != whitespace_area:  # defensive internal invariant
            raise AssertionError(
                "coordinate-compressed whitespace accounting is inconsistent: "
                f"components={component_area}, exact={whitespace_area}"
            )

        return cls(
            clusters=clusters,
            core_bbox_nm=core_bbox,
            core_area_nm2=core_area,
            occupied_area_nm2=occupied_area,
            whitespace_area_nm2=whitespace_area,
            utilization=Fraction(occupied_area, core_area),
            boundary_corridors=boundary,
            enclosed_voids=enclosed,
        )

    def capacity_observations(
        self, layers: Sequence[LayerResource]
    ) -> tuple[CapacityObservation, ...]:
        """Return stable conservative nominal capacities for boundary corridors."""

        supplied_layers = tuple(layers)
        for layer in supplied_layers:
            if not isinstance(layer, LayerResource):
                raise LayerResourceDeclarationError(
                    f"expected LayerResource, got {type(layer).__name__}"
                )
        ordered_layers = tuple(
            sorted(
                supplied_layers,
                key=lambda layer: (
                    layer.layer_n,
                    layer.preferred_axis,
                    layer.pitch_nm,
                    layer.min_width_nm,
                    layer.min_space_nm,
                ),
            )
        )
        seen_layers: set[int] = set()
        for layer in ordered_layers:
            if layer.layer_n in seen_layers:
                raise LayerResourceDeclarationError(
                    f"duplicate resource declaration for metal{layer.layer_n}"
                )
            seen_layers.add(layer.layer_n)

        return tuple(
            CapacityObservation(
                corridor_id=corridor.region_id,
                layer_n=layer.layer_n,
                preferred_axis=layer.preferred_axis,
                clear_span_nm=(span := corridor.clear_span_nm(layer.preferred_axis)),
                pitch_nm=layer.pitch_nm,
                nominal_track_capacity=span // layer.pitch_nm,
            )
            for layer in ordered_layers
            for corridor in self.boundary_corridors
        )

    def canonical_json(self) -> str:
        """Return a byte-stable JSON representation (including exact fractions)."""

        def region_payload(region: FreeSpaceRegion) -> dict[str, object]:
            return {
                "area_nm2": region.area_nm2,
                "bbox_nm": list(region.bbox_nm),
                "cells_nm": [list(cell) for cell in region.cells_nm],
                "region_id": region.region_id,
                "width_nm": region.width_nm,
            }

        payload = {
            "boundary_corridors": [
                region_payload(region) for region in self.boundary_corridors
            ],
            "clusters": [
                {"bbox_nm": list(cluster.bbox_nm), "name": cluster.name}
                for cluster in self.clusters
            ],
            "core_area_nm2": self.core_area_nm2,
            "core_bbox_nm": list(self.core_bbox_nm),
            "enclosed_voids": [
                region_payload(region) for region in self.enclosed_voids
            ],
            "occupied_area_nm2": self.occupied_area_nm2,
            "schema": "clay.routing-resource-observation.v1",
            "utilization": {
                "denominator": self.utilization.denominator,
                "numerator": self.utilization.numerator,
            },
            "whitespace_area_nm2": self.whitespace_area_nm2,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _is_int_nm(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def _validated_cluster(name: object, bbox: Sequence[int]) -> ClusterBox:
    if not isinstance(name, str) or not name:
        raise PlacementGeometryConflict(
            f"cluster name must be a non-empty string, got {name!r}"
        )
    try:
        values = tuple(bbox)
    except TypeError as exc:
        raise PlacementGeometryConflict(
            f"cluster {name!r} bbox must contain four integer-nm coordinates"
        ) from exc
    if len(values) != 4:
        raise PlacementGeometryConflict(
            f"cluster {name!r} bbox must contain four integer-nm coordinates, "
            f"got {values!r}"
        )
    if not all(_is_int_nm(value) for value in values):
        raise PlacementGeometryConflict(
            f"cluster {name!r} bbox requires integer-nm coordinates, got {values!r}"
        )
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise PlacementGeometryConflict(
            f"cluster {name!r} bbox must have positive area, got {values!r}"
        )
    return ClusterBox(name=name, bbox_nm=(x0, y0, x1, y1))


def _reject_positive_area_overlaps(clusters: tuple[ClusterBox, ...]) -> None:
    for index, left in enumerate(clusters):
        lx0, ly0, lx1, ly1 = left.bbox_nm
        for right in clusters[index + 1 :]:
            rx0, ry0, rx1, ry1 = right.bbox_nm
            if max(lx0, rx0) < min(lx1, rx1) and max(ly0, ry0) < min(ly1, ry1):
                raise PlacementGeometryConflict(
                    f"clusters {left.name!r} and {right.name!r} overlap with "
                    "positive area"
                )


def _free_space_components(
    clusters: tuple[ClusterBox, ...], core_bbox: BBoxNm
) -> tuple[tuple[FreeSpaceRegion, ...], tuple[FreeSpaceRegion, ...]]:
    core_x0, core_y0, core_x1, core_y1 = core_bbox
    xs = sorted(
        {coordinate for cluster in clusters for coordinate in cluster.bbox_nm[::2]}
    )
    ys = sorted(
        {
            coordinate
            for cluster in clusters
            for coordinate in (cluster.bbox_nm[1], cluster.bbox_nm[3])
        }
    )

    free: set[tuple[int, int]] = set()
    for xi in range(len(xs) - 1):
        cell_x0, cell_x1 = xs[xi], xs[xi + 1]
        for yi in range(len(ys) - 1):
            cell_y0, cell_y1 = ys[yi], ys[yi + 1]
            occupied = any(
                bx0 <= cell_x0
                and cell_x1 <= bx1
                and by0 <= cell_y0
                and cell_y1 <= by1
                for cluster in clusters
                for bx0, by0, bx1, by1 in (cluster.bbox_nm,)
            )
            if not occupied:
                free.add((xi, yi))

    components: list[tuple[tuple[BBoxNm, ...], bool]] = []
    remaining = set(free)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component_indices: list[tuple[int, int]] = []
        while stack:
            xi, yi = stack.pop()
            component_indices.append((xi, yi))
            for neighbour in (
                (xi - 1, yi),
                (xi, yi - 1),
                (xi, yi + 1),
                (xi + 1, yi),
            ):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)

        cells = tuple(
            sorted((xs[xi], ys[yi], xs[xi + 1], ys[yi + 1]) for xi, yi in component_indices)
        )
        touches_boundary = any(
            x0 == core_x0 or y0 == core_y0 or x1 == core_x1 or y1 == core_y1
            for x0, y0, x1, y1 in cells
        )
        components.append((cells, touches_boundary))

    components.sort(key=lambda item: (_cells_bbox(item[0]), item[0]))
    boundary_cells = [cells for cells, touches in components if touches]
    enclosed_cells = [cells for cells, touches in components if not touches]

    boundary = tuple(
        _region_from_cells(f"boundary-{index:03d}", cells)
        for index, cells in enumerate(boundary_cells)
    )
    enclosed = tuple(
        _region_from_cells(f"void-{index:03d}", cells)
        for index, cells in enumerate(enclosed_cells)
    )
    return boundary, enclosed


def _region_from_cells(region_id: str, cells: tuple[BBoxNm, ...]) -> FreeSpaceRegion:
    bbox = _cells_bbox(cells)
    horizontal_span = _merged_cross_section_bottleneck(cells, "H")
    vertical_span = _merged_cross_section_bottleneck(cells, "V")
    return FreeSpaceRegion(
        region_id=region_id,
        bbox_nm=bbox,
        area_nm2=sum(_bbox_area(cell) for cell in cells),
        width_nm=min(horizontal_span, vertical_span),
        cells_nm=cells,
    )


def _cells_bbox(cells: tuple[BBoxNm, ...]) -> BBoxNm:
    return (
        min(cell[0] for cell in cells),
        min(cell[1] for cell in cells),
        max(cell[2] for cell in cells),
        max(cell[3] for cell in cells),
    )


def _bbox_area(bbox: BBoxNm) -> int:
    x0, y0, x1, y1 = bbox
    return (x1 - x0) * (y1 - y0)


def _merged_cross_section_bottleneck(
    cells: tuple[BBoxNm, ...], preferred_axis: str
) -> int:
    """Measure a component without exposing coordinate-compression cuts.

    For a horizontal route, group cells by exact x slab, merge all touching y
    intervals in that slab, retain its widest contiguous opening, then take the
    minimum opening across slabs.  Vertical routing is the transposed operation.
    Thus an unrelated cluster boundary may subdivide an open corridor into more
    elementary cells, but it cannot change the measured bottleneck.
    """

    sections: dict[tuple[int, int], list[tuple[int, int]]] = {}
    if preferred_axis == "H":
        for x0, y0, x1, y1 in cells:
            sections.setdefault((x0, x1), []).append((y0, y1))
    elif preferred_axis == "V":
        for x0, y0, x1, y1 in cells:
            sections.setdefault((y0, y1), []).append((x0, x1))
    else:  # callers validate; keep this helper total for direct tests/debugging
        raise ValueError(f"preferred_axis must be 'H' or 'V', got {preferred_axis!r}")

    slab_openings: list[int] = []
    for slab in sorted(sections):
        intervals = sorted(sections[slab])
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                old_start, old_end = merged[-1]
                merged[-1] = (old_start, max(old_end, end))
            else:
                merged.append((start, end))
        slab_openings.append(max(end - start for start, end in merged))
    return min(slab_openings)


def _read_positive_metal_rule(tech: object, accessor: str, layer_n: int) -> int:
    layer_name = f"metal{layer_n}"
    fn = getattr(tech, accessor, None)
    if not callable(fn):
        raise LayerResourceDeclarationError(
            f"{layer_name} requires LayoutTech.{accessor}"
        )

    aliases = (layer_name, f"met{layer_n}", f"m{layer_n}")
    failures: list[str] = []
    for alias in aliases:
        try:
            value = fn(alias)
        except Exception as exc:
            failures.append(f"{alias}: {exc}")
            continue
        if value is None:
            failures.append(f"{alias}: absent")
            continue
        if not _is_int_nm(value) or value <= 0:
            raise LayerResourceDeclarationError(
                f"{layer_name} {accessor} alias {alias!r} must be a positive "
                f"integer-nm declaration, got {value!r}"
            )
        return value

    detail = "; ".join(failures)
    if detail:
        raise LayerResourceDeclarationError(
            f"{layer_name} {accessor} declaration is unavailable across PDK "
            f"aliases ({detail})"
        )
    raise LayerResourceDeclarationError(
        f"{layer_name} {accessor} declaration is unavailable"
    )


__all__ = [
    "LANDING_SELECTION_EXPLICIT_OPTION",
    "LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY",
    "LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN",
    "MAX_TOPOLOGY_BRANCH_TARGETS",
    "MAX_TOPOLOGY_BRANCH_TERMINALS",
    "MAX_TOPOLOGY_DECLARED_EDGE_TRUNKS",
    "MAX_TOPOLOGY_LAYERS",
    "MAX_TOPOLOGY_PLAN_CANDIDATES",
    "MAX_TOPOLOGY_PLAN_DEMANDS",
    "MAX_TOPOLOGY_REQUIRED_GOAL_REFS",
    "MAX_LANDING_SITES",
    "MAX_CORRIDOR_USE_ALLOWED_TRACK_STARTS",
    "MAX_LANDING_DEMAND_ALLOWED_SITES",
    "MAX_CANDIDATE_BRANCH_ORDERED_USES",
    "MAX_CANDIDATE_BRANCH_ORDERED_LANDING_DEMANDS",
    "MAX_CANDIDATE_BRANCH_STEPS",
    "MAX_LANDING_OPTION_SITE_IDS",
    "MAX_LANDING_OPTION_CONTACTS_PER_OPTION",
    "MAX_LANDING_SITE_OWNER_LAYERS",
    "MAX_LANDING_SITE_OWNER_COMPONENT_REFS_PER_SIDE",
    "MAX_LANDING_SITE_PAIR_RULE_REFS",
    "MAX_ENDPOINT_OWNER_COMPONENT_REFS",
    "MAX_OWNED_COMPONENT_USE_REFS_PER_CANDIDATE",
    "MAX_CANDIDATE_PAIR_DECLARATION_REFS",
    "MAX_DERIVED_CONTINUITY_EVALUATIONS",
    "MAX_OWNED_CONDUCTOR_COMPONENT_CERTIFICATES",
    "MAX_OWNED_CONDUCTOR_COMPONENT_MEMBERS",
    "MAX_OWNED_CONDUCTOR_COMPONENT_PAIR_EVALUATIONS",
    "MAX_TOTAL_OWNED_COMPONENT_USE_REFS",
    "MAX_REQUIRED_GOAL_REFS",
    "MAX_REQUIRED_ROUTE_GOALS",
    "MAX_TOTAL_BRANCH_EDGES",
    "MAX_TRACK_CENTERS_PER_RESOURCE",
    "MAX_TOTAL_TRACK_CENTERS",
    "MAX_TRACK_RESOURCES",
    "MAX_TOTAL_BRANCH_STEPS",
    "MAX_TOTAL_CANDIDATE_BRANCHES",
    "MAX_TOTAL_ENDPOINT_ACCESSES",
    "MAX_TOTAL_ENDPOINT_OWNER_COMPONENT_REFS",
    "MAX_TOTAL_GOAL_BRANCH_COVERAGE",
    "MAX_TOTAL_LANDING_OPTION_TEMPLATES",
    "MAX_TOTAL_LANDING_OPTIONS",
    "MAX_TOTAL_LANDING_OPTION_CONTACTS",
    "MAX_TOTAL_LANDING_OPTION_SITE_CLAIMS",
    "MAX_DERIVED_TRACK_CONFLICT_CLAUSES",
    "MAX_DERIVED_TRACK_CONFLICT_EVALUATIONS",
    "MAX_LANDING_CONFLICT_NET_CLAUSES",
    "MAX_LANDING_SITE_PAIR_CONFLICTS",
    "AllocatedLandingOption",
    "AllocatedNetProgram",
    "AllocatedTrack",
    "CandidateBranch",
    "CandidateBranchStep",
    "CandidateEndpointAccess",
    "CandidatePairConflict",
    "CandidatePool",
    "CandidatePoolError",
    "CandidatePoolLimitError",
    "CapacityObservation",
    "CapacityConflict",
    "ClusterBox",
    "CompleteNetCandidate",
    "CompleteNetCandidateFamily",
    "DiagnosticRouteProgramBundle",
    "FactorizedCandidateDemandRole",
    "FactorizedCandidateProfile",
    "FactorizedCandidateUseProfile",
    "FactorizedCandidateUseRole",
    "FactorizedCompleteNetCandidateFamily",
    "CompleteNetSelectionResult",
    "CorridorLayerKey",
    "CorridorUse",
    "FreeSpaceRegion",
    "GoalBranchCoverage",
    "HallDeficit",
    "InfeasibilityCertificateVerificationError",
    "LayerKey",
    "LayerResource",
    "LayerResourceDeclarationError",
    "LandingDemand",
    "LandingExpansionResource",
    "LandingOption",
    "landing_option_template_index",
    "landing_option_template_record",
    "LandingSite",
    "LandingSiteCatalog",
    "LandingSitePage",
    "LandingSiteConflictDomain",
    "LandingSiteConflictMetalRule",
    "LandingSitePairConflict",
    "LandingSiteConflictViaRule",
    "LandingTrackContact",
    "MissingNetCandidateError",
    "MissingRequiredGoalError",
    "OwnedConductorComponentCertificate",
    "OwnedConductorIndexCertificate",
    "PlacementGeometryConflict",
    "PointNm",
    "PoolUnsatCertificate",
    "RequiredRouteGoal",
    "RouteResourcePlan",
    "RoutePlanObjectiveCertificate",
    "RoutePlanObjectiveGeometryError",
    "RoutePlanObjectivePhase",
    "RoutePlanVerificationReceipt",
    "ROUTE_PLAN_OBJECTIVE_DEFAULT_DETERMINISTIC_TIME",
    "ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME",
    "RouteResourcePlanVerificationBudgetExhausted",
    "RouteResourcePlanVerificationError",
    "RoutingDemand",
    "RoutingResourceModel",
    "SelectedLanding",
    "SelectedLandingOption",
    "SelectedNetCandidate",
    "SelectedTrack",
    "TerminalDemand",
    "TrackChoiceRef",
    "TrackResource",
    "TopologyCandidate",
    "TopologyPlan",
    "canonical_explicit_routing_resource_ledger_json",
    "candidate_has_closed_allocation_ir",
    "complete_net_candidate_identity_payload",
    "content_address_complete_net_candidate",
    "explicit_routing_resource_ledger_sha256",
    "paged_routing_resource_ledger_sha256",
    "factorize_complete_net_candidate_family",
    "landing_site_collection_sha256",
    "route_plan_objective_policy_digest",
    "certify_route_resource_plan_verification",
    "select_complete_net_candidates",
    "verify_infeasibility_certificate",
    "verify_route_resource_plan",
    "verify_route_plan_verification_receipt",
    "verify_candidate_allocation_ir",
    "verify_selection_certificate",
]
