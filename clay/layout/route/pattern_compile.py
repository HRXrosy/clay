"""Deterministic fixed-pattern routing compiler primitives.

This module is intentionally downstream of the final routing-world resource
ledger.  It records exact lane/site incidence; it does not infer whitespace
capacity and it does not invoke a router.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
import hashlib
from itertools import combinations, islice
import json
from math import isqrt
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from clay.layout.route.resource_conductors import OwnedConductorComponentIndex
from clay.layout.route.resource_meter import (
    ResourceMeterScopeError, bind_budget, budget_meter, lane_index_boundary,
    lazy_access_boundary, resource_boundary,
)
from clay.layout.route.resource_plan import (
    CandidateBranch,
    CandidateBranchStep,
    CandidateEndpointAccess,
    CandidatePool,
    CandidatePoolError,
    CandidatePoolLimitError,
    CompleteNetCandidate,
    CompleteNetCandidateFamily,
    DiagnosticRouteProgramBundle,
    FactorizedCompleteNetCandidateFamily,
    CorridorLayerKey,
    CorridorUse,
    GoalBranchCoverage,
    LandingDemand,
    LandingOption,
    LandingSiteCatalog,
    LandingSitePage,
    LandingTrackContact,
    complete_net_candidate_identity_payload,
    content_address_complete_net_candidate,
    factorize_complete_net_candidate_family,
    landing_option_template_record,
    LANDING_SELECTION_EXPLICIT_OPTION,
    LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY,
    LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN,
    MAX_TOTAL_LANDING_OPTION_TEMPLATES,
    MAX_TOTAL_LANDING_OPTIONS,
    MAX_LANDING_SITES,
    MAX_TOTAL_CANDIDATES,
    MAX_TOTAL_ASSIGNMENT_ARCS,
    MAX_TOPOLOGY_BRANCH_TARGETS,
    MAX_TOPOLOGY_BRANCH_TERMINALS,
    MAX_TOPOLOGY_DECLARED_EDGE_TRUNKS,
    MAX_TOPOLOGY_LAYERS,
    MAX_TOPOLOGY_PLAN_CANDIDATES,
    MAX_TOPOLOGY_PLAN_DEMANDS,
    MAX_TOPOLOGY_REQUIRED_GOAL_REFS,
    OwnedConductorComponentCertificate,
    RequiredRouteGoal,
    InfeasibilityCertificateVerificationError,
    RouteResourcePlanVerificationError,
    RouteResourcePlanVerificationBudgetExhausted,
    RoutingDemand,
    TerminalDemand,
    TopologyCandidate,
    TopologyPlan,
    TrackChoiceRef,
    TrackResource,
    paged_routing_resource_ledger_sha256,
)
from clay.layout.route import resource_world as resource_world_module
from clay.layout.route.resource_world import (
    ExplicitRoutingResourceLedger,
    RoutingResourceEnumerationError,
    RoutingWorldProvenance,
    pdk_provenance_digest,
    _prepare_exact_landing_site_query_context,
    _owned_access_feasible_regions,
    _query_exact_landing_sites_at_points,
    _query_exact_landing_sites_for_component_lane,
    _query_exact_landing_sites_for_lane_pair,
    replay_exact_landing_site_catalog_compressed,
    replay_exact_landing_site_subset_compressed,
)
from clay.layout.route.via_model import via_landing


LayerKey = tuple[int, int]
PointNm = tuple[int, int]
MAX_INCIDENCE_EVALUATIONS = 2_000_000
MAX_PATTERN_TRACK_RESOURCES = 8_192
MAX_PATTERN_LANDING_SITES = 32_768
MAX_PATTERN_LANDING_SITE_PAGE_SIZE = MAX_PATTERN_LANDING_SITES
MAX_PATTERN_LANES = 2_000_000
MAX_PATTERN_SITE_SIDES = 65_536
MAX_PATTERN_ENDPOINTS = 1_000_000
MAX_PATTERN_UNKNOWN_INPUT_DIGESTS = 64
MAX_PATTERN_UNKNOWN_REFS = 4_096
MAX_PATTERN_SITE_TRACK_CHOICES = 2_000_000
MAX_PATTERN_SITE_OWNER_COMPONENT_REFS = 200_000
MAX_PATTERN_SITE_TRAVERSABLE_REFS = 200_000
MAX_PATTERN_RAW_SIDE_COMPONENT_REFS = 200_000
MAX_PATTERN_ENDPOINT_ACCESS_REFS = 200_000
MAX_PATTERN_ENDPOINT_CONDUCTOR_REFS = 200_000
# The default is an operational proof budget, while the hard ceiling bounds
# hostile inputs.  Keeping them distinct lets a caller replay a legitimate
# dense analog resource graph with more headroom without weakening the global
# finite-domain contract.  The external_pdk OTA exact graph was measured after spatial
# indexing at more than 450k charged probes before its independent geometry
# budget stopped the same compile, and a one-million replay then reached the
# later staircase frontier.  Two million is the operational bound for this
# exact graph; four million remains a distinct hostile-input ceiling.
DEFAULT_PATTERN_EXPANSIONS = 2_000_000
MAX_PATTERN_EXPANSIONS = 4_000_000
# Pool-global exact landing-option templates are a separate operational proof
# domain.  Candidate-local aliases retain their independent immutable 65,536
# aggregate ceiling in CandidatePool v12; this compiler limit never discounts
# aliases unless the canonical template table explicitly interns their physical
# atom and binding.
DEFAULT_PATTERN_LANDING_OPTION_TEMPLATES = 32_768
MAX_PATTERN_LANDING_OPTION_TEMPLATES = MAX_TOTAL_LANDING_OPTION_TEMPLATES
MAX_FIXED_PATTERN_CANDIDATES_PER_NET = 512
MAX_PATTERN_WORLD_PINS = 100_000
MAX_PATTERN_WORLD_OBSTACLES = 200_000
MAX_PATTERN_WORLD_RAILS = 100_000
MAX_PATTERN_WORLD_REQUIRED_NETS = 50
MAX_PATTERN_LEDGER_SITE_CONFLICTS = 100_000
MAX_PATTERN_DIAGNOSTIC_PROVENANCE_CHARS = 96
MAX_DIAGNOSTIC_PARTITION_SOLVES = 1_024
MAX_DIAGNOSTIC_MERGE_SOLVES = 1_024
MAX_DIAGNOSTIC_ELAPSED_SECONDS = 3_000.0


class PatternCompileScope(str, Enum):
    SIGNOFF = "signoff"
    BRINGUP_DIAGNOSTIC = "bringup_diagnostic"


class PatternCompileInputError(ValueError):
    """The exact-world inputs cannot form a trustworthy pattern artifact."""


class PatternCompileLimitError(PatternCompileInputError):
    """A requested compiler limit is invalid or exceeds the hard ceiling."""


def _bounded_diagnostic_provenance(provenance: str) -> str:
    """Keep failure evidence readable while imposing a strict byte ceiling."""

    encoded = provenance.encode("utf-8")
    if len(encoded) <= MAX_PATTERN_DIAGNOSTIC_PROVENANCE_CHARS:
        return provenance
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    prefix = encoded[: MAX_PATTERN_DIAGNOSTIC_PROVENANCE_CHARS - 22]
    while True:
        try:
            readable = prefix.decode("utf-8")
            break
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return f"{readable}...#{digest}"


@dataclass(frozen=True)
class PatternCompileUnknown:
    """Typed, hashable UNKNOWN result that carries no partial routing artifact."""

    code: str
    phase: str
    input_digests: tuple[tuple[str, str], ...]
    refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise PatternCompileInputError("UNKNOWN code must be non-empty")
        if not isinstance(self.phase, str) or not self.phase:
            raise PatternCompileInputError("UNKNOWN phase must be non-empty")
        digests = tuple(
            sorted(
                _bounded_materialize(
                    self.input_digests,
                    limit=MAX_PATTERN_UNKNOWN_INPUT_DIGESTS,
                    label="UNKNOWN input digest",
                )
            )
        )
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or not digest
            for name, digest in digests
        ) or len({name for name, _digest in digests}) != len(digests):
            raise PatternCompileInputError(
                "UNKNOWN input digests must have unique non-empty names and values"
            )
        refs = tuple(
            sorted(
                _bounded_materialize(
                    self.refs,
                    limit=MAX_PATTERN_UNKNOWN_REFS,
                    label="UNKNOWN ref",
                )
            )
        )
        if any(not isinstance(ref, str) or not ref for ref in refs):
            raise PatternCompileInputError("UNKNOWN refs must be non-empty strings")
        object.__setattr__(self, "input_digests", digests)
        object.__setattr__(self, "refs", refs)

    @property
    def status(self) -> str:
        return "UNKNOWN"

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "input_digests": [list(item) for item in self.input_digests],
            "phase": self.phase,
            "refs": list(self.refs),
            "schema": "clay.fixed-pattern-unknown.v1",
            "status": self.status,
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.canonical_payload())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FixedPatternCompileResult:
    """Final fixed-pattern compiler contract; UNKNOWN never carries a pool."""

    status: str
    candidate_pool: Any = None
    unknown: PatternCompileUnknown | None = None

    def __post_init__(self) -> None:
        if self.status not in ("COMPLETE", "UNKNOWN"):
            raise PatternCompileInputError("compile result status is invalid")
        if self.status == "UNKNOWN":
            if self.unknown is None or self.candidate_pool is not None:
                raise PatternCompileInputError(
                    "UNKNOWN compile result requires one reason and no pool"
                )
        else:
            if self.unknown is not None or self.candidate_pool is None:
                raise PatternCompileInputError(
                    "COMPLETE compile result requires a pool and no UNKNOWN reason"
                )
            if type(self.candidate_pool) is not CandidatePool:
                raise PatternCompileInputError(
                    "COMPLETE compile result requires an exact CandidatePool"
                )

    @classmethod
    def from_unknown(
        cls, unknown: PatternCompileUnknown
    ) -> "FixedPatternCompileResult":
        if not isinstance(unknown, PatternCompileUnknown):
            raise PatternCompileInputError(
                "from_unknown requires PatternCompileUnknown"
            )
        return cls(status="UNKNOWN", unknown=unknown)

    def canonical_json(self) -> str:
        pool_hash = ""
        if self.candidate_pool is not None:
            digest = getattr(self.candidate_pool, "sha256", None)
            if not callable(digest):
                raise PatternCompileInputError(
                    "complete candidate pool must provide sha256()"
                )
            pool_hash = str(digest())
        return _canonical_json(
            {
                "candidate_pool_hash": pool_hash,
                "schema": "clay.fixed-pattern-compile-result.v1",
                "status": self.status,
                "unknown": (
                    None if self.unknown is None else self.unknown.canonical_payload()
                ),
            }
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiagnosticDecomposedCompileResult:
    """A non-signoff exact frontier derived from bounded family-local solves.

    The carried program is the best conflict-free local-winner frontier proved
    by the bounded deterministic backtracking search.  It is never a proof about
    the original global candidate universe and remains local to bring-up.
    """

    route_program: DiagnosticRouteProgramBundle
    source_unknown: PatternCompileUnknown
    family_count: int
    family_winner_count: int
    failed_family_refs: tuple[str, ...]
    unrouted_nets: tuple[str, ...] = ()
    schema: str = "clay.diagnostic-decomposed-compile.v1"

    def __post_init__(self) -> None:
        if type(self.route_program) is not DiagnosticRouteProgramBundle:
            raise PatternCompileInputError(
                "diagnostic decomposed compile requires a diagnostic route program"
            )
        if type(self.source_unknown) is not PatternCompileUnknown:
            raise PatternCompileInputError(
                "diagnostic decomposed compile requires its source typed stop"
            )
        if (
            type(self.family_count) is not int
            or type(self.family_winner_count) is not int
            or self.family_count <= 0
            or self.family_winner_count <= 0
            or self.family_winner_count > self.family_count
        ):
            raise PatternCompileInputError(
                "diagnostic family counts are malformed"
            )
        refs = tuple(sorted(_bounded_materialize(
            self.failed_family_refs,
            limit=MAX_PATTERN_UNKNOWN_REFS,
            label="diagnostic failed family ref",
        )))
        if any(not isinstance(ref, str) or not ref for ref in refs):
            raise PatternCompileInputError(
                "diagnostic failed family refs must be non-empty strings"
            )
        object.__setattr__(self, "failed_family_refs", refs)
        nets = tuple(sorted(_bounded_materialize(
            self.unrouted_nets,
            limit=MAX_PATTERN_WORLD_REQUIRED_NETS,
            label="diagnostic unrouted net",
        )))
        if any(not isinstance(net, str) or not net for net in nets):
            raise PatternCompileInputError(
                "diagnostic unrouted nets must be non-empty strings"
            )
        object.__setattr__(self, "unrouted_nets", nets)

    @property
    def status(self) -> str:
        return "DIAGNOSTIC_DECOMPOSED"

    def canonical_json(self) -> str:
        return _canonical_json({
            "route_program_hash": self.route_program.sha256(),
            "failed_family_refs": list(self.failed_family_refs),
            "family_count": self.family_count,
            "family_winner_count": self.family_winner_count,
            "merge_policy": "DETERMINISTIC_BOUNDED_EXACT_FRONTIER_SEARCH",
            "schema": self.schema,
            "source_unknown": self.source_unknown.canonical_payload(),
            "status": self.status,
            "unrouted_nets": list(self.unrouted_nets),
        })

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class _IncidenceBudgetExceeded(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _IncidenceBudget:
    limit: int
    used: int = 0
    option_bundle_calls: int = 0
    option_bundle_sites: int = 0
    option_bundle_k: int = 0
    option_bundle_options: int = 0
    option_bundle_digest: str = hashlib.sha256(
        b"clay.landing-option-budget-diagnostics.v1"
    ).hexdigest()
    last_bundle_kind: str = ""
    last_bundle_context: str = "{}"
    last_bundle_site_count: int = 0
    last_bundle_site_digest: str = ""
    last_bundle_k: int = 0
    last_bundle_option_count: int = 0
    shared_raw_expansions: int = 0
    shared_final_unique_templates: int = 0
    shared_family_count: int = 0
    shared_final_candidate_ids: set[str] = field(default_factory=set)
    shared_final_option_template_keys: set[str] = field(
        default_factory=set
    )
    shared_raw_landing_option_cache: dict[
        tuple[Any, ...], LandingOption
    ] = field(default_factory=dict)
    shared_raw_landing_option_template_keys: set[
        tuple[Any, ...]
    ] = field(default_factory=set)
    shared_lane_component_contact_intervals: dict[
        tuple[str, str], tuple[tuple[int, int], ...]
    ] = field(default_factory=dict)
    reason_counts: dict[str, int] = field(default_factory=dict)

    def consume(self, reason: str) -> None:
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        self.used += 1
        if self.used > self.limit:
            raise _IncidenceBudgetExceeded(reason)

    def record_option_bundle(
        self,
        *,
        kind: str,
        site_ids: Sequence[str],
        required_multiplicity: int,
        option_count: int,
        context: dict[str, Any],
    ) -> None:
        ordered_site_ids = tuple(sorted(site_ids))
        self.option_bundle_calls += 1
        self.option_bundle_sites += len(ordered_site_ids)
        self.option_bundle_k += required_multiplicity
        self.option_bundle_options += option_count
        self.last_bundle_kind = kind
        self.last_bundle_context = _canonical_json(context)
        self.last_bundle_site_count = len(ordered_site_ids)
        self.last_bundle_site_digest = canonical_pattern_hash({
            "schema": "clay.landing-option-site-set.v1",
            "site_ids": list(ordered_site_ids),
        })
        self.last_bundle_k = required_multiplicity
        self.last_bundle_option_count = option_count
        self.option_bundle_digest = canonical_pattern_hash({
            "call": self.option_bundle_calls,
            "context": context,
            "kind": kind,
            "option_count": option_count,
            "prior_digest": self.option_bundle_digest,
            "required_multiplicity": required_multiplicity,
            "schema": "clay.landing-option-budget-diagnostics.v1",
            "site_ids": list(ordered_site_ids),
        })


class _SharedSignalRawOptionBudget(_IncidenceBudget):
    """Charge transient exact-family options to the search budget only."""

    def __init__(
        self,
        expansion_budget: _IncidenceBudget,
        final_budget: _IncidenceBudget,
    ) -> None:
        super().__init__(MAX_PATTERN_EXPANSIONS)
        self._expansion_budget = expansion_budget
        self._final_budget = final_budget
        self._landing_option_cache = (
            final_budget.shared_raw_landing_option_cache
        )

    def consume(self, _reason: str) -> None:
        self.used += 1
        self._final_budget.shared_raw_expansions += 1
        if self.used > self.limit:
            raise _IncidenceBudgetExceeded(
                "exact-family raw landing-option transient bound"
            )
        self._expansion_budget.consume(
            "exact-family raw landing-option expansion"
        )

    def intern_landing_option(
        self,
        signature: tuple[Any, ...],
        factory: Any,
    ) -> LandingOption:
        cached = self._landing_option_cache.get(signature)
        if cached is not None:
            return cached
        option = factory()
        self.used += 1
        self._final_budget.shared_raw_expansions += 1
        if self.used > self.limit:
            raise _IncidenceBudgetExceeded(
                "exact-family raw landing-option transient bound"
            )
        template_key = (
            option.site_ids,
            tuple(sorted(contact.site_id for contact in option.contacts)),
        )
        template_keys = (
            self._final_budget.shared_raw_landing_option_template_keys
        )
        if template_key not in template_keys:
            self._expansion_budget.consume(
                "exact-family raw landing-option template expansion"
            )
            template_keys.add(template_key)
        # A typed budget stop above publishes neither an uncharged exact alias
        # nor a partial physical-template interning table.  Candidate-local
        # TrackChoice bindings remain in ``option`` and retain their separate
        # hostile-input bound through ``self.used``.
        self._landing_option_cache[signature] = option
        return option


def _intern_landing_option(
    option_budget: _IncidenceBudget,
    *,
    reason: str,
    option_id: str,
    demand_id: str,
    site_ids: tuple[str, ...],
    contacts: tuple[LandingTrackContact, ...],
) -> LandingOption:
    signature = option_id, demand_id, site_ids, contacts

    def factory() -> LandingOption:
        return LandingOption(
            option_id=option_id,
            demand_id=demand_id,
            site_ids=site_ids,
            contacts=contacts,
        )

    if isinstance(option_budget, _SharedSignalRawOptionBudget):
        return option_budget.intern_landing_option(signature, factory)
    option_budget.consume(reason)
    return factory()


def _account_elevated_dogleg_witness(
    expansion_budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    *,
    raw_before: int,
    expansion_before: int | None = None,
) -> None:
    """Debit one witness once when its mandatory exact options already paid."""

    if isinstance(option_budget, _SharedSignalRawOptionBudget):
        # Raw aliases and physical expansion debits are deliberately separate:
        # rebinding one already-known site template creates a new exact alias
        # without paying the operational incidence account.  Only an actual
        # expansion debit may stand in for this witness.
        if (
            expansion_before is not None
            and expansion_budget.used > expansion_before
        ):
            return
        if expansion_before is None and option_budget.used <= raw_before:
            expansion_budget.consume(
                "two-pin owned-component elevated dogleg expansion"
            )
            return
        if expansion_before is None:
            # Backward-compatible direct probes cannot distinguish alias-only
            # growth.  Conservatively charge the witness rather than publish a
            # zero-cost expansion.
            expansion_budget.consume(
                "two-pin owned-component elevated dogleg expansion"
            )
            return
    expansion_budget.consume(
        "two-pin owned-component elevated dogleg expansion"
    )


@dataclass
class _SharedSignalFamilyBudgetState:
    raw_option_budget: _SharedSignalRawOptionBudget
    family_count: int = 0
    staircase_edge_cache: dict[
        tuple[str, str, str, int], tuple[Any, ...]
    ] = field(default_factory=dict)
    staircase_neighbor_cache: dict[
        tuple[str, str, int, int], dict[str, tuple[Any, tuple[Any, ...]]]
    ] = field(default_factory=dict)
    staircase_boundary_order_cache: dict[
        tuple[tuple[Any, ...], int], tuple[Any, ...]
    ] = field(default_factory=dict)


class _SharedSignalCandidateFamilyAccumulator:
    """Retain every bounded motif tier, grouped by exact family schema."""

    def __init__(self) -> None:
        self._families: dict[
            tuple[Any, ...], dict[str, CompleteNetCandidate]
        ] = {}

    def add(self, candidate: CompleteNetCandidate) -> None:
        family_key = _shared_signal_family_view(candidate)[0]
        self._families.setdefault(family_key, {})[
            candidate.candidate_id
        ] = candidate

    @property
    def family_count(self) -> int:
        return len(self._families)

    def candidates(self) -> tuple[CompleteNetCandidate, ...]:
        ordered = []
        for family_key in sorted(self._families, key=repr):
            family = self._families[family_key]
            ordered.extend(
                family[candidate_id]
                for candidate_id in sorted(family)
            )
        return tuple(ordered)


# Exact staircase frontiers are sparse relative to the geometric lane domain.
# Small leaves keep a touching node from billing a whole coarse bucket of false
# lanes before the half-open segment predicate can reject them.  Four is the
# measured crossover where leaf work falls faster than deterministic STR node
# traversal grows for the exact dense/sparse frontier regression below.
_PACKED_LANE_ADJACENCY_NODE_CAPACITY = 4
_PACKED_LANE_ADJACENCY_QUERY_THRESHOLD = 64


@dataclass(frozen=True)
class _PackedLaneAdjacencyNode:
    """One deterministic segment-bbox node for exact lane intersections."""

    bbox_nm: tuple[int, int, int, int]
    minimum_lane_id: str
    lanes: tuple[LaneRef, ...] = ()
    children: tuple["_PackedLaneAdjacencyNode", ...] = ()


def _lane_segment_bbox_nm(lane: LaneRef) -> tuple[int, int, int, int]:
    run_lo, run_hi = lane.through_interval_nm
    if lane.preferred_axis == "H":
        return run_lo, lane.center_nm, run_hi, lane.center_nm
    return lane.center_nm, run_lo, lane.center_nm, run_hi


def _staircase_neighbor_order_key(
    lane: LaneRef,
    *,
    trunk_lane: LaneRef,
    terminal: TerminalDemand,
) -> tuple[int, int, str]:
    """Prefer exact staircase prefixes geometrically closest to the trunk."""

    lane_bbox = _lane_segment_bbox_nm(lane)
    trunk_bbox = _lane_segment_bbox_nm(trunk_lane)
    axis_gaps = tuple(
        max(0, max(lane_bbox[index], trunk_bbox[index]) - min(
            lane_bbox[index + 2], trunk_bbox[index + 2]
        ))
        for index in (0, 1)
    )
    terminal_cross = (
        terminal.xy_nm[1]
        if lane.preferred_axis == "H"
        else terminal.xy_nm[0]
    )
    return (
        sum(axis_gaps),
        abs(lane.center_nm - terminal_cross),
        lane.lane_id,
    )


def _lane_bbox_union(
    bboxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    if not bboxes:
        raise PatternCompileInputError("packed lane node requires a non-empty domain")
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _lane_bboxes_touch(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    # Tree bounds are conservative/inclusive.  The final lane predicate below
    # retains CorridorUse's exact half-open run semantics.
    return (
        max(left[0], right[0]) <= min(left[2], right[2])
        and max(left[1], right[1]) <= min(left[3], right[3])
    )


def _packed_lane_groups(
    entries: Sequence[Any],
    *,
    bbox_of: Any,
    stable_id_of: Any,
) -> tuple[tuple[Any, ...], ...]:
    """Deterministically STR-pack one level of lane segments."""

    if not entries:
        return ()
    capacity = _PACKED_LANE_ADJACENCY_NODE_CAPACITY
    group_count = (len(entries) + capacity - 1) // capacity
    slice_count = isqrt(group_count)
    if slice_count * slice_count < group_count:
        slice_count += 1
    groups_per_slice = (group_count + slice_count - 1) // slice_count
    slice_size = capacity * groups_per_slice

    def x_key(entry: Any) -> tuple[Any, ...]:
        bbox = bbox_of(entry)
        return bbox[0] + bbox[2], bbox[1] + bbox[3], bbox, stable_id_of(entry)

    def y_key(entry: Any) -> tuple[Any, ...]:
        bbox = bbox_of(entry)
        return bbox[1] + bbox[3], bbox[0] + bbox[2], bbox, stable_id_of(entry)

    by_x = tuple(sorted(entries, key=x_key))
    groups = []
    for slice_start in range(0, len(by_x), slice_size):
        spatial_slice = tuple(sorted(
            by_x[slice_start:slice_start + slice_size], key=y_key
        ))
        for group_start in range(0, len(spatial_slice), capacity):
            groups.append(tuple(
                spatial_slice[group_start:group_start + capacity]
            ))
    return tuple(groups)


def _build_packed_lane_adjacency_root(
    lanes: Sequence[LaneRef],
) -> _PackedLaneAdjacencyNode | None:
    if not lanes:
        return None
    leaf_groups = _packed_lane_groups(
        lanes,
        bbox_of=_lane_segment_bbox_nm,
        stable_id_of=lambda lane: lane.lane_id,
    )
    level = tuple(
        _PackedLaneAdjacencyNode(
            bbox_nm=_lane_bbox_union(tuple(
                _lane_segment_bbox_nm(lane) for lane in group
            )),
            minimum_lane_id=min(lane.lane_id for lane in group),
            lanes=tuple(sorted(group, key=lambda lane: lane.lane_id)),
        )
        for group in leaf_groups
    )
    while len(level) > 1:
        parent_groups = _packed_lane_groups(
            level,
            bbox_of=lambda node: node.bbox_nm,
            stable_id_of=lambda node: node.minimum_lane_id,
        )
        level = tuple(
            _PackedLaneAdjacencyNode(
                bbox_nm=_lane_bbox_union(tuple(
                    child.bbox_nm for child in group
                )),
                minimum_lane_id=min(child.minimum_lane_id for child in group),
                children=tuple(group),
            )
            for group in parent_groups
        )
    return level[0]


def _query_packed_lane_adjacency(
    root: _PackedLaneAdjacencyNode | None,
    lane: LaneRef,
    *,
    budget: _IncidenceBudget,
) -> tuple[LaneRef, ...]:
    if root is None:
        return ()
    query_bbox = _lane_segment_bbox_nm(lane)
    matched: dict[str, LaneRef] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        budget.consume("exact lane adjacency spatial-index node")
        if not _lane_bboxes_touch(node.bbox_nm, query_bbox):
            continue
        if node.lanes:
            for candidate in node.lanes:
                budget.consume("exact lane adjacency query")
                if _lanes_geometrically_intersect(lane, candidate):
                    matched[candidate.lane_id] = candidate
            continue
        pending.extend(reversed(node.children))
    return tuple(matched[lane_id] for lane_id in sorted(matched))


class _BudgetedLaneLayerIndex:
    """One exact, budgeted lane partition with lazy adjacency memoization."""

    def __init__(
        self,
        incidence_index: LaneIncidenceIndex,
        *,
        shared_adjacency_index: _BudgetedLaneLayerIndex | None = None,
    ) -> None:
        self._incidence_index = incidence_index
        self._lane_ids = frozenset(lane.lane_id for lane in incidence_index.lanes)
        self._shared_adjacency_index = shared_adjacency_index
        if (
            shared_adjacency_index is not None
            and not self._lane_ids.issubset(shared_adjacency_index._lane_ids)
        ):
            raise PatternCompileInputError(
                "shared lane adjacency index is not a superset of the projection"
            )
        self._lanes_by_layer: dict[int, tuple[LaneRef, ...]] | None = None
        self._lanes_by_layer_axis: dict[
            tuple[int, str], tuple[tuple[int, ...], tuple[LaneRef, ...]]
        ] | None = None
        self._lane_spatial_roots: dict[
            tuple[int, str], _PackedLaneAdjacencyNode | None
        ] | None = None
        self._intersection_cache: dict[
            tuple[str, int], tuple[LaneRef, ...]
        ] = {}

    @lane_index_boundary
    def _build(self, *, budget: _IncidenceBudget) -> None:
        if self._lanes_by_layer is not None:
            return
        pending: dict[int, list[LaneRef]] = {}
        pending_by_axis: dict[tuple[int, str], list[LaneRef]] = {}
        for lane in self._incidence_index.lanes:
            budget.consume("exact lane layer-index build")
            pending.setdefault(lane.layer_n, []).append(lane)
            pending_by_axis.setdefault(
                (lane.layer_n, lane.preferred_axis), []
            ).append(lane)
        complete = {
            layer_n: tuple(sorted(lanes, key=lambda lane: lane.lane_id))
            for layer_n, lanes in pending.items()
        }
        complete_by_axis = {}
        for key, lanes in pending_by_axis.items():
            ordered = tuple(sorted(
                lanes, key=lambda lane: (lane.center_nm, lane.lane_id)
            ))
            complete_by_axis[key] = (
                tuple(lane.center_nm for lane in ordered),
                ordered,
            )
        # Publish only after the complete finite scan succeeds.  A typed budget
        # stop must never leave a partial index that a retry could mistake for
        # an exact domain.
        self._lanes_by_layer = complete
        self._lanes_by_layer_axis = complete_by_axis
        self._lane_spatial_roots = {
            key: _build_packed_lane_adjacency_root(rows[1])
            for key, rows in complete_by_axis.items()
        }

    @lane_index_boundary
    def lanes(
        self,
        layer_n: int,
        *,
        budget: _IncidenceBudget,
    ) -> tuple[LaneRef, ...]:
        self._build(budget=budget)
        assert self._lanes_by_layer is not None
        return self._lanes_by_layer.get(layer_n, ())

    @lane_index_boundary
    def intersecting(
        self,
        lane: LaneRef,
        layer_n: int,
        *,
        budget: _IncidenceBudget,
    ) -> tuple[LaneRef, ...]:
        cache_key = (lane.lane_id, layer_n)
        cached = self._intersection_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._shared_adjacency_index is not None:
            # A per-net incidence index is an exact lane-id subset of the base
            # index.  Reuse the completed physical adjacency proof and remove
            # only foreign owned lanes; no geometric possibility is discarded.
            complete = tuple(
                candidate
                for candidate in self._shared_adjacency_index.intersecting(
                    lane, layer_n, budget=budget
                )
                if candidate.lane_id in self._lane_ids
            )
            self._intersection_cache[cache_key] = complete
            return complete
        self._build(budget=budget)
        assert self._lanes_by_layer_axis is not None
        assert self._lane_spatial_roots is not None
        result: dict[str, LaneRef] = {}
        run_lo, run_hi = lane.through_interval_nm
        for candidate_axis in ("H", "V"):
            centers, candidates = self._lanes_by_layer_axis.get(
                (layer_n, candidate_axis), ((), ())
            )
            if candidate_axis == lane.preferred_axis:
                # Parallel lanes meet only on the same fixed coordinate.
                first = bisect_left(centers, lane.center_nm)
                after = bisect_right(centers, lane.center_nm)
            else:
                # A perpendicular candidate can meet ``lane`` only when its
                # fixed coordinate lies inside lane's finite run interval.
                first = bisect_left(centers, run_lo)
                after = bisect_right(centers, run_hi)
            frontier = candidates[first:after]
            if (
                candidate_axis != lane.preferred_axis
                and len(frontier) > _PACKED_LANE_ADJACENCY_QUERY_THRESHOLD
            ):
                for candidate in _query_packed_lane_adjacency(
                    self._lane_spatial_roots.get((layer_n, candidate_axis)),
                    lane,
                    budget=budget,
                ):
                    result[candidate.lane_id] = candidate
            else:
                for candidate in frontier:
                    budget.consume("exact lane adjacency query")
                    if _lanes_geometrically_intersect(lane, candidate):
                        result[candidate.lane_id] = candidate
        complete = tuple(result[key] for key in sorted(result))
        self._intersection_cache[cache_key] = complete
        return complete

    @lane_index_boundary
    def touching_regions(
        self,
        layer_n: int,
        *,
        regions: Sequence[tuple[int, int, int, int]],
        points: Sequence[tuple[int, int]],
        budget: _IncidenceBudget,
    ) -> tuple[LaneRef, ...]:
        """Exact fixed-coordinate range query for component-access lanes."""

        self._build(budget=budget)
        assert self._lanes_by_layer_axis is not None
        result: dict[str, LaneRef] = {}
        for axis in ("H", "V"):
            centers, lanes = self._lanes_by_layer_axis.get(
                (layer_n, axis), ((), ())
            )
            possible: dict[str, LaneRef] = {}
            for x0, y0, x1, y1 in regions:
                cross_lo, cross_hi = (y0, y1) if axis == "H" else (x0, x1)
                first = bisect_left(centers, cross_lo)
                after = bisect_right(centers, cross_hi)
                for lane in lanes[first:after]:
                    possible[lane.lane_id] = lane
            for point in points:
                cross = point[1] if axis == "H" else point[0]
                first = bisect_left(centers, cross)
                after = bisect_right(centers, cross)
                for lane in lanes[first:after]:
                    possible[lane.lane_id] = lane
            # One lane can enter many overlapping atom access windows.  The
            # spatial index forms their finite union first, then bills one exact
            # membership proof per distinct lane instead of re-probing the same
            # lane once per atom rectangle.
            for lane_id in sorted(possible):
                lane = possible[lane_id]
                budget.consume("owned-component lane spatial query")
                lane_lo, lane_hi = lane.through_interval_nm
                touches_region = any(
                    (
                        y0 <= lane.center_nm <= y1
                        and lane_lo <= x1
                        and x0 < lane_hi
                    )
                    if axis == "H"
                    else (
                        x0 <= lane.center_nm <= x1
                        and lane_lo <= y1
                        and y0 < lane_hi
                    )
                    for x0, y0, x1, y1 in regions
                )
                if touches_region or any(
                    lane.contains_point(point) for point in points
                ):
                    result[lane_id] = lane
        return tuple(result[key] for key in sorted(result))


def _bounded_materialize(
    values: Any,
    *,
    limit: int,
    label: str,
    budget: _IncidenceBudget | None = None,
    budget_reason: str = "",
) -> tuple[Any, ...]:
    """Read no more than cap+1 items, optionally charging each generator pull."""

    if type(limit) is not int or limit < 0:
        raise TypeError(f"{label} limit must be an exact non-negative integer")
    try:
        count = len(values)
    except TypeError:
        count = None
    if count is not None and count > limit:
        raise PatternCompileLimitError(f"{label} count {count} exceeds {limit}")
    materialized: list[Any] = []
    for item in islice(iter(values), limit + 1):
        if budget is not None:
            budget.consume(budget_reason)
        materialized.append(item)
    if len(materialized) > limit:
        raise PatternCompileLimitError(f"{label} count exceeds {limit}")
    return tuple(materialized)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_pattern_hash(payload: Any) -> str:
    """Return the SHA-256 of canonical compiler-side JSON."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_pattern_id(prefix: str, payload: Any) -> str:
    """Return a readable content-addressed compiler-side identifier."""

    if not isinstance(prefix, str) or not prefix:
        raise ValueError("pattern id prefix must be a non-empty string")
    return f"{prefix}:{canonical_pattern_hash(payload)}"


@dataclass(frozen=True, order=True)
class LaneRef:
    """One exact legal lane from one geometry-complete TrackResource."""

    resource_corridor_id: str
    layer_n: int
    gds_layer_key: LayerKey
    preferred_axis: str
    track_start: int
    center_nm: int
    through_interval_nm: tuple[int, int]
    corridor_cross_bounds_nm: tuple[int, int]
    routing_world_provenance_digest: str
    owner_net: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.resource_corridor_id, str)
            or not self.resource_corridor_id
        ):
            raise PatternCompileInputError("lane corridor id must be non-empty")
        if type(self.layer_n) is not int or self.layer_n <= 0:
            raise PatternCompileInputError("lane layer_n must be a positive integer")
        layer_key = tuple(self.gds_layer_key)
        if len(layer_key) != 2 or any(type(value) is not int for value in layer_key):
            raise PatternCompileInputError(
                "lane GDS layer key must contain two integers"
            )
        if self.preferred_axis not in ("H", "V"):
            raise PatternCompileInputError("lane preferred axis must be H or V")
        if type(self.track_start) is not int or self.track_start < 0:
            raise PatternCompileInputError("lane track_start must be non-negative")
        if type(self.center_nm) is not int:
            raise PatternCompileInputError("lane center must be integer nm")
        interval = tuple(self.through_interval_nm)
        if (
            len(interval) != 2
            or any(type(value) is not int for value in interval)
            or interval[1] <= interval[0]
        ):
            raise PatternCompileInputError(
                "lane interval must be non-empty and half-open"
            )
        cross_bounds = tuple(self.corridor_cross_bounds_nm)
        if (
            len(cross_bounds) != 2
            or any(type(value) is not int for value in cross_bounds)
            or cross_bounds[1] <= cross_bounds[0]
            or not cross_bounds[0] <= self.center_nm <= cross_bounds[1]
        ):
            raise PatternCompileInputError(
                "lane corridor cross bounds must contain its integer-nm center"
            )
        if (
            not isinstance(self.routing_world_provenance_digest, str)
            or not self.routing_world_provenance_digest
        ):
            raise PatternCompileInputError(
                "lane routing-world digest must be non-empty"
            )
        if self.owner_net is not None and (
            not isinstance(self.owner_net, str) or not self.owner_net
        ):
            raise PatternCompileInputError(
                "lane owner_net must be None or a non-empty string"
            )
        object.__setattr__(self, "gds_layer_key", layer_key)
        object.__setattr__(self, "through_interval_nm", interval)
        object.__setattr__(self, "corridor_cross_bounds_nm", cross_bounds)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "center_nm": self.center_nm,
            "corridor_cross_bounds_nm": list(self.corridor_cross_bounds_nm),
            "gds_layer_key": list(self.gds_layer_key),
            "layer_n": self.layer_n,
            "preferred_axis": self.preferred_axis,
            "resource_corridor_id": self.resource_corridor_id,
            "routing_world_provenance_digest": (self.routing_world_provenance_digest),
            "schema": "clay.fixed-pattern-lane-ref.v2",
            "through_interval_nm": list(self.through_interval_nm),
            "track_start": self.track_start,
            **(
                {"owner_net": self.owner_net}
                if self.owner_net is not None
                else {}
            ),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.canonical_payload())

    @cached_property
    def lane_id(self) -> str:
        return canonical_pattern_id("lane", self.canonical_payload())

    @cached_property
    def incidence_use_id(self) -> str:
        return canonical_pattern_id(
            "lane-use",
            {
                "gds_layer_key": list(self.gds_layer_key),
                "layer_n": self.layer_n,
                "preferred_axis": self.preferred_axis,
                "resource_corridor_id": self.resource_corridor_id,
                "corridor_cross_bounds_nm": list(
                    self.corridor_cross_bounds_nm
                ),
                "routing_world_provenance_digest": (
                    self.routing_world_provenance_digest
                ),
                "schema": "clay.fixed-pattern-incidence-use.v2",
                "through_interval_nm": list(self.through_interval_nm),
                **(
                    {"owner_net": self.owner_net}
                    if self.owner_net is not None
                    else {}
                ),
            },
        )

    def contains_point(self, point_nm: PointNm) -> bool:
        x_nm, y_nm = point_nm
        run_nm, cross_nm = (x_nm, y_nm) if self.preferred_axis == "H" else (y_nm, x_nm)
        return (
            cross_nm == self.center_nm
            and self.through_interval_nm[0] <= run_nm < self.through_interval_nm[1]
        )

    @cached_property
    def track_choice_ref(self) -> TrackChoiceRef:
        return TrackChoiceRef(
            use_id=self.incidence_use_id,
            track_start=self.track_start,
        )


@dataclass(frozen=True, order=True)
class SiteSideIncidence:
    """Exact lane/component contacts on one side of one via landing site."""

    site_id: str
    side: str
    layer_n: int
    gds_layer_key: LayerKey
    xy_nm: PointNm
    crossing_track_choices: tuple[TrackChoiceRef, ...] = ()
    owner_component_refs: tuple[str, ...] = ()
    traversable_conductor_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.site_id, str) or not self.site_id:
            raise PatternCompileInputError("site-side site_id must be non-empty")
        if self.side not in ("lower", "upper"):
            raise PatternCompileInputError("site side must be lower or upper")
        if type(self.layer_n) is not int or self.layer_n <= 0:
            raise PatternCompileInputError("site-side layer_n must be positive")
        layer_key = tuple(self.gds_layer_key)
        point = tuple(self.xy_nm)
        if len(layer_key) != 2 or any(type(value) is not int for value in layer_key):
            raise PatternCompileInputError("site-side GDS layer key is malformed")
        if len(point) != 2 or any(type(value) is not int for value in point):
            raise PatternCompileInputError("site-side point must be integer nm")
        choices = tuple(
            sorted(
                _bounded_materialize(
                    self.crossing_track_choices,
                    limit=MAX_PATTERN_SITE_TRACK_CHOICES,
                    label="site-side track choice",
                )
            )
        )
        if len(choices) != len(set(choices)):
            raise PatternCompileInputError("site-side track choices must be unique")
        refs = tuple(
            sorted(
                _bounded_materialize(
                    self.owner_component_refs,
                    limit=MAX_PATTERN_SITE_OWNER_COMPONENT_REFS,
                    label="site-side owner component ref",
                )
            )
        )
        if any(not isinstance(ref, str) or not ref for ref in refs) or len(refs) != len(
            set(refs)
        ):
            raise PatternCompileInputError("site-side component refs must be unique")
        traversable = tuple(
            sorted(
                _bounded_materialize(
                    self.traversable_conductor_refs,
                    limit=MAX_PATTERN_SITE_TRAVERSABLE_REFS,
                    label="site-side traversable conductor ref",
                )
            )
        )
        if (
            any(not isinstance(ref, str) or not ref for ref in traversable)
            or len(traversable) != len(set(traversable))
            or not set(traversable).issubset(refs)
        ):
            raise PatternCompileInputError(
                "traversable refs must be a unique subset of owner refs"
            )
        object.__setattr__(self, "gds_layer_key", layer_key)
        object.__setattr__(self, "xy_nm", point)
        object.__setattr__(self, "crossing_track_choices", choices)
        object.__setattr__(self, "owner_component_refs", refs)
        object.__setattr__(self, "traversable_conductor_refs", traversable)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "crossing_track_choices": [
                {"track_start": choice.track_start, "use_id": choice.use_id}
                for choice in self.crossing_track_choices
            ],
            "gds_layer_key": list(self.gds_layer_key),
            "layer_n": self.layer_n,
            "owner_component_refs": list(self.owner_component_refs),
            "schema": "clay.fixed-pattern-site-side-incidence.v1",
            "side": self.side,
            "site_id": self.site_id,
            "traversable_conductor_refs": list(self.traversable_conductor_refs),
            "xy_nm": list(self.xy_nm),
        }

    def certifies_boundary_owner(self, component_ref: str) -> bool:
        """Whether the exact landing pad contacts this certified component."""

        return component_ref in self.owner_component_refs

    def certifies_existing_conductor_traversal(self, component_ref: str) -> bool:
        """Whether the component contains conductor geometry beyond a pin point."""

        return component_ref in self.traversable_conductor_refs


@dataclass(frozen=True, order=True)
class LandingPairIncidence:
    """Exact complete-ledger sites indexed by one lower/upper lane pair."""

    lower_track_choice: TrackChoiceRef
    upper_track_choice: TrackChoiceRef
    site_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.lower_track_choice, TrackChoiceRef) or not isinstance(
            self.upper_track_choice, TrackChoiceRef
        ):
            raise PatternCompileInputError(
                "landing-pair incidence choices must be TrackChoiceRef values"
            )
        site_ids = tuple(self.site_ids)
        if (
            not site_ids
            or any(not isinstance(site_id, str) or not site_id for site_id in site_ids)
            or len(site_ids) != len(set(site_ids))
        ):
            raise PatternCompileInputError(
                "landing-pair incidence site ids must be unique non-empty strings"
            )
        object.__setattr__(self, "site_ids", tuple(sorted(site_ids)))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "lower_track_choice": {
                "track_start": self.lower_track_choice.track_start,
                "use_id": self.lower_track_choice.use_id,
            },
            "schema": "clay.fixed-pattern-landing-pair-incidence.v1",
            "site_ids": list(self.site_ids),
            "upper_track_choice": {
                "track_start": self.upper_track_choice.track_start,
                "use_id": self.upper_track_choice.use_id,
            },
        }


@dataclass(frozen=True, order=True)
class BoundaryLaneIncidence:
    """Exact complete-ledger sites joining one component to one lane."""

    component_ref: str
    lane_track_choice: TrackChoiceRef
    site_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.component_ref, str) or not self.component_ref:
            raise PatternCompileInputError(
                "boundary-lane incidence component ref must be non-empty"
            )
        if not isinstance(self.lane_track_choice, TrackChoiceRef):
            raise PatternCompileInputError(
                "boundary-lane incidence choice must be a TrackChoiceRef"
            )
        site_ids = tuple(self.site_ids)
        if (
            not site_ids
            or any(not isinstance(site_id, str) or not site_id for site_id in site_ids)
            or len(site_ids) != len(set(site_ids))
        ):
            raise PatternCompileInputError(
                "boundary-lane incidence site ids must be unique non-empty strings"
            )
        object.__setattr__(self, "site_ids", tuple(sorted(site_ids)))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "component_ref": self.component_ref,
            "lane_track_choice": {
                "track_start": self.lane_track_choice.track_start,
                "use_id": self.lane_track_choice.use_id,
            },
            "schema": "clay.fixed-pattern-boundary-lane-incidence.v1",
            "site_ids": list(self.site_ids),
        }


def _derive_landing_pair_incidence(
    site_sides: Sequence[SiteSideIncidence],
    *,
    budget: _IncidenceBudget | None = None,
) -> tuple[LandingPairIncidence, ...]:
    """Invert exact side choices into a canonical complete lane-pair table."""

    by_site: dict[str, dict[str, SiteSideIncidence]] = {}
    for side in site_sides:
        by_site.setdefault(side.site_id, {})[side.side] = side
    site_ids_by_pair: dict[
        tuple[TrackChoiceRef, TrackChoiceRef], set[str]
    ] = {}
    for site_id in sorted(by_site):
        sides = by_site[site_id]
        lower = sides.get("lower")
        upper = sides.get("upper")
        if lower is None or upper is None:
            continue
        for lower_choice in lower.crossing_track_choices:
            for upper_choice in upper.crossing_track_choices:
                if budget is not None:
                    budget.consume("landing-site lane-pair incidence")
                site_ids_by_pair.setdefault(
                    (lower_choice, upper_choice), set()
                ).add(site_id)
    return tuple(
        LandingPairIncidence(
            lower_track_choice=pair[0],
            upper_track_choice=pair[1],
            site_ids=tuple(sorted(site_ids)),
        )
        for pair, site_ids in sorted(site_ids_by_pair.items())
    )


def _derive_boundary_lane_incidence(
    site_sides: Sequence[SiteSideIncidence],
    *,
    budget: _IncidenceBudget | None = None,
) -> tuple[BoundaryLaneIncidence, ...]:
    """Invert exact component/side facts into component-to-lane site refs."""

    by_site: dict[str, dict[str, SiteSideIncidence]] = {}
    for side in site_sides:
        by_site.setdefault(side.site_id, {})[side.side] = side
    site_ids_by_join: dict[tuple[str, TrackChoiceRef], set[str]] = {}
    for site_id in sorted(by_site):
        sides = by_site[site_id]
        lower = sides.get("lower")
        upper = sides.get("upper")
        if lower is None or upper is None:
            continue
        for component_ref in lower.owner_component_refs:
            for lane_choice in upper.crossing_track_choices:
                if budget is not None:
                    budget.consume("landing-site boundary-lane incidence")
                site_ids_by_join.setdefault(
                    (component_ref, lane_choice), set()
                ).add(site_id)
        for component_ref in upper.owner_component_refs:
            for lane_choice in lower.crossing_track_choices:
                if budget is not None:
                    budget.consume("landing-site boundary-lane incidence")
                site_ids_by_join.setdefault(
                    (component_ref, lane_choice), set()
                ).add(site_id)
    return tuple(
        BoundaryLaneIncidence(
            component_ref=key[0],
            lane_track_choice=key[1],
            site_ids=tuple(sorted(site_ids)),
        )
        for key, site_ids in sorted(site_ids_by_join.items())
    )


@dataclass(frozen=True)
class EndpointIncidence:
    """One full routing-world pin fact and its exact component provenance."""

    pin_access_fact: tuple[Any, ...]
    access_component_refs: tuple[str, ...]
    conductor_component_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        pin = tuple(self.pin_access_fact)
        from clay.layout.route.terminal_identity import unpack_pin_fact

        try:
            unpack_pin_fact(pin)
        except (TypeError, ValueError) as exc:
            raise PatternCompileInputError(
                "endpoint pin access fact must contain a complete canonical terminal identity"
            ) from exc
        refs = tuple(
            sorted(
                _bounded_materialize(
                    self.access_component_refs,
                    limit=MAX_PATTERN_ENDPOINT_ACCESS_REFS,
                    label="endpoint access component ref",
                )
            )
        )
        conductor_refs = tuple(
            sorted(
                _bounded_materialize(
                    self.conductor_component_refs,
                    limit=MAX_PATTERN_ENDPOINT_CONDUCTOR_REFS,
                    label="endpoint conductor component ref",
                )
            )
        )
        if (
            any(not isinstance(ref, str) or not ref for ref in refs)
            or len(refs) != len(set(refs))
            or any(not isinstance(ref, str) or not ref for ref in conductor_refs)
            or len(conductor_refs) != len(set(conductor_refs))
            or not set(conductor_refs).issubset(refs)
        ):
            raise PatternCompileInputError(
                "endpoint component refs must be unique and conductor refs a subset"
            )
        object.__setattr__(self, "pin_access_fact", pin)
        object.__setattr__(self, "access_component_refs", refs)
        object.__setattr__(self, "conductor_component_refs", conductor_refs)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "access_component_refs": list(self.access_component_refs),
            "conductor_component_refs": list(self.conductor_component_refs),
            "pin_access_fact": list(self.pin_access_fact),
            "schema": "clay.fixed-pattern-endpoint-incidence.v1",
        }


def _component_boundary_cache_key(
    terminal: TerminalDemand,
    endpoint: EndpointIncidence,
    component_ref: str,
) -> tuple[Any, ...]:
    """Return the complete immutable identity of one boundary-lane query."""

    return (
        terminal.provenance,
        terminal.xy_nm,
        terminal.layer,
        endpoint.pin_access_fact,
        endpoint.access_component_refs,
        endpoint.conductor_component_refs,
        component_ref,
    )


@dataclass(frozen=True)
class LaneIncidenceIndex:
    """Immutable exact-lane incidence artifact (populated by its builder)."""

    routing_world_provenance_digest: str
    routing_resource_ledger_digest: str
    owned_conductor_index_digest: str
    landing_pair_incidence: tuple[LandingPairIncidence, ...]
    boundary_lane_incidence: tuple[BoundaryLaneIncidence, ...]
    lanes: tuple[LaneRef, ...] = ()
    site_sides: tuple[SiteSideIncidence, ...] = ()
    endpoints: tuple[EndpointIncidence, ...] = ()
    _landing_pair_site_ids: dict[
        tuple[TrackChoiceRef, TrackChoiceRef], tuple[str, ...]
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _boundary_lane_site_ids: dict[
        tuple[str, TrackChoiceRef], tuple[str, ...]
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _landing_sites_by_id: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _landing_sites_cache_binding: tuple[Any, str, str, str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "routing_world_provenance_digest",
            "routing_resource_ledger_digest",
            "owned_conductor_index_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PatternCompileInputError(f"{name} must be non-empty")
        lanes = tuple(
            sorted(
                _bounded_materialize(
                    self.lanes,
                    limit=MAX_PATTERN_LANES,
                    label="lane incidence index lane",
                )
            )
        )
        if len({lane.lane_id for lane in lanes}) != len(lanes):
            raise PatternCompileInputError("lane ids must be unique")
        site_sides = tuple(
            sorted(
                _bounded_materialize(
                    self.site_sides,
                    limit=MAX_PATTERN_SITE_SIDES,
                    label="lane incidence index site side",
                )
            )
        )
        keys = tuple((item.site_id, item.side) for item in site_sides)
        if len(keys) != len(set(keys)):
            raise PatternCompileInputError("site-side identities must be unique")
        sides_by_site: dict[str, set[str]] = {}
        for site_id, side in keys:
            sides_by_site.setdefault(site_id, set()).add(side)
        if any(
            sides != {"lower", "upper"}
            for sides in sides_by_site.values()
        ):
            raise PatternCompileInputError(
                "each indexed landing site must have exactly one lower and upper side"
            )
        landing_pairs = tuple(sorted(_bounded_materialize(
            self.landing_pair_incidence,
            limit=MAX_PATTERN_EXPANSIONS,
            label="lane incidence landing pair",
        )))
        pair_keys = tuple(
            (item.lower_track_choice, item.upper_track_choice)
            for item in landing_pairs
        )
        if len(pair_keys) != len(set(pair_keys)):
            raise PatternCompileInputError(
                "landing-pair incidence identities must be unique"
            )
        expected_landing_pairs = _derive_landing_pair_incidence(site_sides)
        if landing_pairs != expected_landing_pairs:
            raise PatternCompileInputError(
                "landing-pair incidence must exactly replay site-side choices"
            )
        boundary_lanes = tuple(sorted(_bounded_materialize(
            self.boundary_lane_incidence,
            limit=MAX_PATTERN_EXPANSIONS,
            label="lane incidence boundary lane",
        )))
        boundary_keys = tuple(
            (item.component_ref, item.lane_track_choice)
            for item in boundary_lanes
        )
        if len(boundary_keys) != len(set(boundary_keys)):
            raise PatternCompileInputError(
                "boundary-lane incidence identities must be unique"
            )
        expected_boundary_lanes = _derive_boundary_lane_incidence(site_sides)
        if boundary_lanes != expected_boundary_lanes:
            raise PatternCompileInputError(
                "boundary-lane incidence must exactly replay site-side choices"
            )
        endpoints = tuple(
            sorted(
                _bounded_materialize(
                    self.endpoints,
                    limit=MAX_PATTERN_ENDPOINTS,
                    label="lane incidence index endpoint",
                ),
                key=lambda item: _canonical_json(item.canonical_payload()),
            )
        )
        endpoint_keys = tuple(item.pin_access_fact for item in endpoints)
        if len(endpoint_keys) != len(set(endpoint_keys)):
            raise PatternCompileInputError("endpoint pin facts must be unique")
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "site_sides", site_sides)
        object.__setattr__(self, "landing_pair_incidence", landing_pairs)
        object.__setattr__(self, "_landing_pair_site_ids", {
            (item.lower_track_choice, item.upper_track_choice): item.site_ids
            for item in landing_pairs
        })
        object.__setattr__(self, "boundary_lane_incidence", boundary_lanes)
        object.__setattr__(self, "_boundary_lane_site_ids", {
            (item.component_ref, item.lane_track_choice): item.site_ids
            for item in boundary_lanes
        })
        object.__setattr__(self, "endpoints", endpoints)

    def site_side(self, site_id: str, side: str) -> SiteSideIncidence:
        for item in self.site_sides:
            if item.site_id == site_id and item.side == side:
                return item
        raise KeyError((site_id, side))

    def lanes_at_point(
        self, gds_layer_key: Sequence[int], point_nm: Sequence[int]
    ) -> tuple[LaneRef, ...]:
        layer_key = tuple(gds_layer_key)
        point = tuple(point_nm)
        if len(layer_key) != 2 or any(type(value) is not int for value in layer_key):
            raise PatternCompileInputError("point lookup GDS layer key is malformed")
        if len(point) != 2 or any(type(value) is not int for value in point):
            raise PatternCompileInputError("point lookup must use integer nm")
        return tuple(
            lane
            for lane in self.lanes
            if lane.gds_layer_key == layer_key and lane.contains_point(point)
        )

    def resolve_endpoint(
        self, pin_access_fact: Sequence[Any]
    ) -> EndpointIncidence | None:
        pin = tuple(pin_access_fact)
        for endpoint in self.endpoints:
            if endpoint.pin_access_fact == pin:
                return endpoint
        return None

    def _complete_sites_by_id(
        self,
        ledger: Any,
        *,
        budget: _IncidenceBudget,
    ) -> dict[str, Any]:
        world_digest = getattr(ledger, "routing_world_provenance_digest", None)
        component_digest = getattr(ledger, "owned_conductor_index_digest", None)
        if (
            world_digest != self.routing_world_provenance_digest
            or component_digest != self.owned_conductor_index_digest
        ):
            raise PatternCompileInputError(
                "complete landing lookup ledger does not match incidence certificate"
            )
        sites_by_id = self._landing_sites_by_id
        cache_binding = self._landing_sites_cache_binding
        if (
            sites_by_id is not None
            and cache_binding is not None
            and cache_binding[0] is ledger
            and cache_binding[1:] == (
                self.routing_resource_ledger_digest,
                self.routing_world_provenance_digest,
                self.owned_conductor_index_digest,
            )
        ):
            # The exact ledger is frozen.  Reusing the same previously hashed
            # object preserves the one-registration performance contract while
            # keeping all three provenance digests in the cache key.
            return sites_by_id
        ledger_sha = getattr(ledger, "sha256", None)
        if (
            not callable(ledger_sha)
            or ledger_sha() != self.routing_resource_ledger_digest
        ):
            raise PatternCompileInputError(
                "complete landing lookup ledger does not match incidence certificate"
            )
        if sites_by_id is not None:
            object.__setattr__(self, "_landing_sites_cache_binding", (
                ledger,
                self.routing_resource_ledger_digest,
                self.routing_world_provenance_digest,
                self.owned_conductor_index_digest,
            ))
            return sites_by_id
        landing_sites = _bounded_materialize(
            ledger.landing_sites,
            limit=MAX_PATTERN_LANDING_SITES,
            label="complete incidence lookup site",
            budget=budget,
            budget_reason="complete landing-site incidence lookup registration",
        )
        pending = {site.site_id: site for site in landing_sites}
        if len(pending) != len(landing_sites):
            raise PatternCompileInputError(
                "complete incidence lookup site ids must be unique"
            )
        certified_site_ids = {side.site_id for side in self.site_sides}
        if set(pending) != certified_site_ids:
            raise PatternCompileInputError(
                "complete incidence lookup site domain is incomplete"
            )
        object.__setattr__(self, "_landing_sites_by_id", pending)
        object.__setattr__(self, "_landing_sites_cache_binding", (
            ledger,
            self.routing_resource_ledger_digest,
            self.routing_world_provenance_digest,
            self.owned_conductor_index_digest,
        ))
        return pending

    def landing_sites_for_lane_pair(
        self,
        left_lane: LaneRef,
        right_lane: LaneRef,
        *,
        net: str,
        ledger: Any,
        budget: _IncidenceBudget,
        selection_limit: int,
    ) -> tuple[Any, ...]:
        """Resolve one complete-ledger pair without repeating an S-site scan."""

        if type(selection_limit) is not int or selection_limit < 0:
            raise PatternCompileInputError(
                "landing-pair selection_limit must be a non-negative exact integer"
            )
        if selection_limit == 0 or left_lane.layer_n == right_lane.layer_n:
            return ()
        lower_lane, upper_lane = (
            (left_lane, right_lane)
            if left_lane.layer_n < right_lane.layer_n
            else (right_lane, left_lane)
        )
        site_ids = self._landing_pair_site_ids.get(
            (lower_lane.track_choice_ref, upper_lane.track_choice_ref), ()
        )
        sites_by_id = self._complete_sites_by_id(ledger, budget=budget)
        selected = []
        for site_id in site_ids:
            budget.consume("signal staircase indexed landing-site query")
            site = sites_by_id.get(site_id)
            if site is None:
                raise PatternCompileInputError(
                    "landing-pair incidence names a missing landing site"
                )
            if _site_joins_lanes(
                site=site,
                left_lane=left_lane,
                right_lane=right_lane,
                incidence_index=self,
                net=net,
            ):
                selected.append(site)
                if len(selected) == selection_limit:
                    break
        return tuple(selected)

    def has_landing_pair_incidence(
        self, left_lane: LaneRef, right_lane: LaneRef
    ) -> bool:
        """Whether replayed complete incidence contains this exact lane pair."""

        lower_lane, upper_lane = (
            (left_lane, right_lane)
            if left_lane.layer_n < right_lane.layer_n
            else (right_lane, left_lane)
        )
        return bool(self._landing_pair_site_ids.get(
            (lower_lane.track_choice_ref, upper_lane.track_choice_ref), ()
        ))

    def landing_sites_for_component_lane(
        self,
        terminal: TerminalDemand,
        endpoint: EndpointIncidence,
        component_ref: str | None,
        lane: LaneRef,
        *,
        net: str,
        ledger: Any,
        budget: _IncidenceBudget,
        selection_limit: int,
    ) -> tuple[Any, ...]:
        """Resolve one exact component/lane boundary from its inverted index."""

        if type(selection_limit) is not int or selection_limit < 0:
            raise PatternCompileInputError(
                "boundary-lane selection_limit must be a non-negative exact integer"
            )
        if (
            selection_limit == 0
            or (
                component_ref is not None
                and component_ref not in endpoint.conductor_component_refs
            )
        ):
            return ()
        component_refs = (
            (component_ref,)
            if component_ref is not None
            else endpoint.access_component_refs
        )
        site_ids = tuple(sorted({
            site_id
            for ref in component_refs
            for site_id in self._boundary_lane_site_ids.get(
                (ref, lane.track_choice_ref), ()
            )
        }))
        sites_by_id = self._complete_sites_by_id(ledger, budget=budget)
        selected = []
        for site_id in site_ids:
            budget.consume("signal staircase indexed boundary-site query")
            site = sites_by_id.get(site_id)
            if site is None:
                raise PatternCompileInputError(
                    "boundary-lane incidence names a missing landing site"
                )
            if _site_connects_endpoint_to_lane(
                site=site,
                terminal=terminal,
                endpoint=endpoint,
                lane=lane,
                incidence_index=self,
                net=net,
                required_component_ref=component_ref,
            ):
                selected.append(site)
                if len(selected) == selection_limit:
                    break
        return tuple(selected)

    def canonical_json(self) -> str:
        return _canonical_json(
            {
                "boundary_lane_incidence": [
                    item.canonical_payload()
                    for item in self.boundary_lane_incidence
                ],
                "lanes": [lane.canonical_payload() for lane in self.lanes],
                "landing_pair_incidence": [
                    item.canonical_payload()
                    for item in self.landing_pair_incidence
                ],
                "endpoints": [
                    endpoint.canonical_payload() for endpoint in self.endpoints
                ],
                "owned_conductor_index_digest": self.owned_conductor_index_digest,
                "routing_resource_ledger_digest": self.routing_resource_ledger_digest,
                "routing_world_provenance_digest": (
                    self.routing_world_provenance_digest
                ),
                "schema": "clay.fixed-pattern-lane-incidence-index.v3",
                "site_sides": [item.canonical_payload() for item in self.site_sides],
            }
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _incidence_index_for_net(
    incidence_index: LaneIncidenceIndex, net: str
) -> LaneIncidenceIndex:
    """Project net-owned terminal tracks out of every foreign net's domain."""

    lanes = tuple(
        lane
        for lane in incidence_index.lanes
        if lane.owner_net in (None, net)
    )
    if len(lanes) == len(incidence_index.lanes):
        return incidence_index
    retained_choices = {lane.track_choice_ref for lane in lanes}
    landing_pair_incidence = tuple(
        item
        for item in incidence_index.landing_pair_incidence
        if item.lower_track_choice in retained_choices
        and item.upper_track_choice in retained_choices
    )
    boundary_lane_incidence = tuple(
        item
        for item in incidence_index.boundary_lane_incidence
        if item.lane_track_choice in retained_choices
    )
    return LaneIncidenceIndex(
        routing_world_provenance_digest=(
            incidence_index.routing_world_provenance_digest
        ),
        routing_resource_ledger_digest=(
            incidence_index.routing_resource_ledger_digest
        ),
        owned_conductor_index_digest=incidence_index.owned_conductor_index_digest,
        landing_pair_incidence=landing_pair_incidence,
        boundary_lane_incidence=boundary_lane_incidence,
        lanes=lanes,
        site_sides=tuple(
            replace(
                side,
                crossing_track_choices=tuple(
                    choice
                    for choice in side.crossing_track_choices
                    if choice in retained_choices
                ),
            )
            for side in incidence_index.site_sides
        ),
        endpoints=incidence_index.endpoints,
    )


def _tech_layer_key(tech: Any, layer_n: int) -> LayerKey:
    try:
        values = tuple(tech.metal_key(layer_n))
    except Exception as exc:
        raise PatternCompileInputError(
            f"PDK has no metal{layer_n} GDS layer key"
        ) from exc
    if len(values) != 2 or any(type(value) is not int for value in values):
        raise PatternCompileInputError(f"PDK metal{layer_n} GDS layer key is malformed")
    return values[0], values[1]


def _primitive_contacts_pad(
    component: Any,
    pad_bbox_nm: tuple[int, int, int, int],
    budget: _IncidenceBudget,
) -> bool:
    px0, py0, px1, py1 = pad_bbox_nm
    for fact in component.atom_facts:
        budget.consume("landing-side atom pad contact")
        x0, y0, x1, y1 = fact[:4]
        if not (x1 < px0 or px1 < x0 or y1 < py0 or py1 < y0):
            return True
    for pin in component.pin_accesses:
        budget.consume("landing-side pin pad contact")
        x_nm, y_nm = int(pin[3]), int(pin[4])
        if px0 <= x_nm <= px1 and py0 <= y_nm <= py1:
            return True
    return False


def _component_refs_for_side(
    refs: Sequence[str],
    *,
    conductor_index: Any,
    layer_key: LayerKey,
    owner_net: Any,
    side: str,
    pad_bbox_nm: Sequence[int],
    budget: _IncidenceBudget,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    pad_bbox = tuple(pad_bbox_nm)
    if (
        len(pad_bbox) != 4
        or any(type(value) is not int for value in pad_bbox)
        or pad_bbox[2] <= pad_bbox[0]
        or pad_bbox[3] <= pad_bbox[1]
    ):
        raise PatternCompileInputError(f"{side} landing pad bbox is malformed")
    certified: list[str] = []
    traversable: list[str] = []
    bounded_refs = _bounded_materialize(
        refs,
        limit=MAX_PATTERN_RAW_SIDE_COMPONENT_REFS,
        label="raw landing-side component ref",
        budget=budget,
        budget_reason="landing-side component-ref input materialization",
    )
    for component_ref in sorted(bounded_refs):
        budget.consume("landing-side component membership")
        try:
            component = conductor_index.component(component_ref)
        except Exception as exc:
            raise PatternCompileInputError(
                f"landing site names missing component {component_ref!r}"
            ) from exc
        if component.layer_key != layer_key:
            raise PatternCompileInputError(
                f"component {component_ref!r} is on the wrong landing side"
            )
        if owner_net is not None and component.net != owner_net:
            raise PatternCompileInputError(
                f"component {component_ref!r} has the wrong landing owner"
            )
        if not _primitive_contacts_pad(component, pad_bbox, budget):
            raise PatternCompileInputError(
                f"component {component_ref!r} does not contact the exact "
                f"{side} landing pad"
            )
        certified.append(component_ref)
        if component.atom_facts:
            traversable.append(component_ref)
    return tuple(certified), tuple(traversable)


def _build_lane_incidence_index_complete(
    ledger: Any,
    conductor_index: Any,
    tech: Any,
    *,
    budget: _IncidenceBudget,
) -> LaneIncidenceIndex:
    world_digest = getattr(ledger, "routing_world_provenance_digest", "")
    if (
        not world_digest
        or getattr(conductor_index, "world_digest", None) != world_digest
    ):
        raise PatternCompileInputError(
            "owned-conductor index belongs to a foreign routing world"
        )
    conductor_digest = conductor_index.sha256()
    if getattr(ledger, "owned_conductor_index_digest", "") != conductor_digest:
        raise PatternCompileInputError(
            "owned-conductor index digest does not match resource ledger"
        )

    lanes: list[LaneRef] = []
    lanes_by_coordinate: dict[tuple[int, str, int], list[LaneRef]] = {}
    track_resources = _bounded_materialize(
        ledger.track_resources,
        limit=MAX_PATTERN_TRACK_RESOURCES,
        label="pattern track resource",
        budget=budget,
        budget_reason="track-resource input materialization",
    )
    for resource in sorted(track_resources, key=lambda item: item.key):
        layer_key = _tech_layer_key(tech, resource.key.layer_n)
        if resource.gds_layer_key is None or tuple(resource.gds_layer_key) != layer_key:
            raise PatternCompileInputError(
                f"track resource {resource.key!r} has a stale GDS layer key"
            )
        if resource.routing_world_provenance_digest != world_digest:
            raise PatternCompileInputError("track resource belongs to a foreign world")
        interval = tuple(resource.through_interval_nm or ())
        corridor_bbox = tuple(resource.corridor_bbox_nm or ())
        if len(corridor_bbox) != 4:
            raise PatternCompileInputError(
                "geometry-complete track resource lacks corridor bbox"
            )
        cross_bounds = (
            (corridor_bbox[1], corridor_bbox[3])
            if resource.preferred_axis == "H"
            else (corridor_bbox[0], corridor_bbox[2])
        )
        for track_start, center_nm in enumerate(resource.track_centers_nm):
            budget.consume("track-resource lane materialization")
            lane = LaneRef(
                resource_corridor_id=resource.key.corridor_id,
                layer_n=resource.key.layer_n,
                gds_layer_key=layer_key,
                preferred_axis=resource.preferred_axis,
                track_start=track_start,
                center_nm=center_nm,
                through_interval_nm=interval,
                corridor_cross_bounds_nm=cross_bounds,
                routing_world_provenance_digest=world_digest,
                owner_net=resource.owner_net,
            )
            lanes.append(lane)
            lanes_by_coordinate.setdefault(
                (lane.layer_n, lane.preferred_axis, lane.center_nm), []
            ).append(lane)

    site_sides: list[SiteSideIncidence] = []
    landing_sites = _bounded_materialize(
        ledger.landing_sites,
        limit=MAX_PATTERN_LANDING_SITES,
        label="pattern landing site",
        budget=budget,
        budget_reason="landing-site input materialization",
    )
    for site in sorted(landing_sites, key=lambda item: item.site_id):
        if site.routing_world_provenance_digest != world_digest:
            raise PatternCompileInputError("landing site belongs to a foreign world")
        for side, layer_n, raw_refs, pad_bbox_nm in (
            (
                "lower",
                site.lower_layer_n,
                site.lower_owner_component_refs,
                site.lower_pad_bbox_nm,
            ),
            (
                "upper",
                site.upper_layer_n,
                site.upper_owner_component_refs,
                site.upper_pad_bbox_nm,
            ),
        ):
            budget.consume("landing-site side record")
            layer_key = _tech_layer_key(tech, layer_n)
            x_nm, y_nm = site.xy_nm
            candidates = (
                *lanes_by_coordinate.get((layer_n, "H", y_nm), ()),
                *lanes_by_coordinate.get((layer_n, "V", x_nm), ()),
            )
            crossing_choices: list[TrackChoiceRef] = []
            for lane in candidates:
                budget.consume("landing-site lane incidence")
                if lane.contains_point(site.xy_nm):
                    crossing_choices.append(lane.track_choice_ref)
            refs, traversable_refs = _component_refs_for_side(
                raw_refs,
                conductor_index=conductor_index,
                layer_key=layer_key,
                owner_net=site.owner_net,
                side=side,
                pad_bbox_nm=pad_bbox_nm or (),
                budget=budget,
            )
            site_sides.append(
                SiteSideIncidence(
                    site_id=site.site_id,
                    side=side,
                    layer_n=layer_n,
                    gds_layer_key=layer_key,
                    xy_nm=site.xy_nm,
                    crossing_track_choices=crossing_choices,
                    owner_component_refs=refs,
                    traversable_conductor_refs=traversable_refs,
                )
            )

    endpoints: list[EndpointIncidence] = []
    for component in conductor_index.components:
        for pin_access_fact in component.pin_accesses:
            budget.consume("endpoint component incidence")
            is_traversable = bool(component.atom_facts)
            endpoints.append(
                EndpointIncidence(
                    pin_access_fact=pin_access_fact,
                    access_component_refs=(component.component_ref,),
                    conductor_component_refs=(
                        (component.component_ref,) if is_traversable else ()
                    ),
                )
            )

    landing_pair_incidence = _derive_landing_pair_incidence(
        site_sides, budget=budget
    )
    boundary_lane_incidence = _derive_boundary_lane_incidence(
        site_sides, budget=budget
    )
    return LaneIncidenceIndex(
        routing_world_provenance_digest=world_digest,
        routing_resource_ledger_digest=ledger.sha256(),
        owned_conductor_index_digest=conductor_digest,
        landing_pair_incidence=landing_pair_incidence,
        boundary_lane_incidence=boundary_lane_incidence,
        lanes=tuple(lanes),
        site_sides=tuple(site_sides),
        endpoints=tuple(endpoints),
    )


@resource_boundary(world_argument="ledger", tech_argument="tech", ledger_world=True)
def build_lane_incidence_index(
    ledger: Any,
    conductor_index: Any,
    tech: Any,
    *,
    max_incidence_evaluations: int = MAX_INCIDENCE_EVALUATIONS,
    resource_meter: Any = None,
) -> LaneIncidenceIndex | PatternCompileUnknown:
    """Build a complete index or return typed UNKNOWN on bounded-work overflow."""

    if (
        type(max_incidence_evaluations) is not int
        or max_incidence_evaluations <= 0
        or max_incidence_evaluations > MAX_INCIDENCE_EVALUATIONS
    ):
        raise PatternCompileLimitError(
            "max_incidence_evaluations must be a positive exact integer no greater "
            f"than {MAX_INCIDENCE_EVALUATIONS}"
        )
    world_digest = str(getattr(ledger, "routing_world_provenance_digest", "") or "")
    ledger_sha = getattr(ledger, "sha256", None)
    conductor_sha = getattr(conductor_index, "sha256", None)
    if not callable(ledger_sha) or not callable(conductor_sha):
        raise PatternCompileInputError(
            "ledger and conductor index must provide sha256()"
        )
    ledger_digest = str(ledger_sha())
    conductor_digest = str(conductor_sha())
    incidence_budget = bind_budget(
        _IncidenceBudget(max_incidence_evaluations), resource_meter, "incidence"
    )
    try:
        index = _build_lane_incidence_index_complete(
            ledger,
            conductor_index,
            tech,
            budget=incidence_budget,
        )
        if resource_meter is not None:
            resource_meter.register_incidence(index, ledger, conductor_index)
        return index
    except _IncidenceBudgetExceeded as exc:
        return PatternCompileUnknown(
            code="INCIDENCE_BUDGET_EXHAUSTED",
            phase="LANE_INCIDENCE",
            input_digests=(
                ("owned_conductor_index", conductor_digest),
                ("routing_resource_ledger", ledger_digest),
                ("routing_world", world_digest),
            ),
            refs=(
                exc.reason,
                f"max_incidence_evaluations={max_incidence_evaluations}",
            ),
        )


def verify_lane_incidence_index_replay(
    incidence_index: LaneIncidenceIndex,
    ledger: Any,
    conductor_index: Any,
    tech: Any,
    *,
    max_incidence_evaluations: int = MAX_INCIDENCE_EVALUATIONS,
) -> str | PatternCompileUnknown:
    """Replay a supplied index from bound truth and return its stable digest.

    Digest labels carried by the supplied artifact are not evidence for its
    site-side payload.  Rebuild the complete lower/upper facts from the bound
    landing-site ledger, owned-conductor components, and PDK layer truth, then
    require byte-identical canonical content before any lookup may consume it.
    """

    if type(incidence_index) is not LaneIncidenceIndex:
        raise PatternCompileInputError(
            "incidence replay requires exact LaneIncidenceIndex"
        )
    replayed = build_lane_incidence_index(
        ledger,
        conductor_index,
        tech,
        max_incidence_evaluations=max_incidence_evaluations,
    )
    if isinstance(replayed, PatternCompileUnknown):
        return replayed
    if replayed.canonical_json() != incidence_index.canonical_json():
        raise PatternCompileInputError(
            "lane incidence index differs from independent ledger/conductor replay"
        )
    return replayed.sha256()


class _ExactLazyLandingAccess:
    """Exact finite queries over a non-materialized landing lattice."""

    def __init__(
        self,
        ledger: ExplicitRoutingResourceLedger,
        world: RoutingWorldProvenance,
        tech: Any,
        conductor_index: OwnedConductorComponentIndex,
        incidence_index: LaneIncidenceIndex,
        query_context: Any,
        lane_layer_index: _BudgetedLaneLayerIndex,
        *,
        shared_sites: dict[str, Any] | None = None,
    ) -> None:
        self.ledger = ledger
        self.world = world
        self.tech = tech
        self.conductor_index = conductor_index
        self.incidence_index = incidence_index
        self._diagnostic_meter = budget_meter(
            getattr(query_context, "geometry_budget", None)
        )
        if self._diagnostic_meter is not None and shared_sites is not None:
            raise ResourceMeterScopeError(
                "metered lazy access cannot adopt an unbound shared site cache"
            )
        self._resources = {
            resource.key: resource for resource in ledger.track_resources
        }
        if shared_sites is None:
            self._sites = {
                site.site_id: site for site in ledger.landing_sites
            }
        else:
            self._sites = shared_sites
            self._sites.update(
                {site.site_id: site for site in ledger.landing_sites}
            )
        self._layer_n_by_key = _layer_number_map(tech)
        self._query_context = query_context
        self._lane_layer_index = lane_layer_index
        self._internal_cache: dict[
            tuple[str, str, str], tuple[Any, ...]
        ] = {}
        self._internal_join_charged: set[tuple[str, str, str]] = set()
        self._boundary_cache: dict[
            tuple[Any, ...],
            tuple[tuple[Any, ...], tuple[tuple[str, int], ...]],
        ] = {}
        self._boundary_join_charged: set[tuple[Any, ...]] = set()
        self._boundary_use_cache: dict[
            tuple[Any, ...], tuple[Any, ...]
        ] = {}
        self._component_lane_cache: dict[str, tuple[LaneRef, ...]] = {}
        self._staircase_internal_cache: dict[
            tuple[str, str, str, int], tuple[Any, ...]
        ] = {}

    def _resource(self, lane: LaneRef) -> TrackResource:
        key = CorridorLayerKey(lane.resource_corridor_id, lane.layer_n)
        resource = self._resources.get(key)
        if resource is None:
            raise PatternCompileInputError(
                f"lazy landing query names missing lane resource {key!r}"
            )
        return resource

    def _remember(self, sites: Sequence[Any], *, retain: bool) -> tuple[Any, ...]:
        result = tuple(sites)
        if retain:
            for site in result:
                self._sites[site.site_id] = site
        return result

    @lazy_access_boundary
    def internal(
        self,
        left_lane: LaneRef,
        right_lane: LaneRef,
        *,
        net: str,
        retain: bool = True,
        budget: _IncidenceBudget | None = None,
    ) -> tuple[Any, ...]:
        cache_key = tuple(sorted((left_lane.lane_id, right_lane.lane_id))) + (net,)
        sites = self._internal_cache.get(cache_key)
        publish_sites = sites is None
        if sites is None:
            sites = _query_exact_landing_sites_for_lane_pair(
                self.ledger,
                self.world,
                self.tech,
                self.conductor_index,
                left_resource=self._resource(left_lane),
                left_track_start=left_lane.track_start,
                right_resource=self._resource(right_lane),
                right_track_start=right_lane.track_start,
                query_context=self._query_context,
            )
            # The resource-world query proves legal via geometry, including
            # pad overlap at a corridor end.  A closed-IR ``contact`` is the
            # stronger fact that the certified via center lies on both exact
            # half-open lane domains.  Mirror complete incidence replay here;
            # otherwise a pad-only overlap can later fabricate a contact whose
            # run coordinate is outside its CorridorUse.
            sites = tuple(
                site
                for site in sites
                if site.owner_net in (None, net)
                and left_lane.contains_point(site.xy_nm)
                and right_lane.contains_point(site.xy_nm)
                and {left_lane.layer_n, right_lane.layer_n}
                == {site.lower_layer_n, site.upper_layer_n}
            )
        charged = getattr(self, "_internal_join_charged", None)
        if charged is None:
            charged = set()
            self._internal_join_charged = charged
        if budget is not None and cache_key not in charged:
            for _site in sites:
                budget.consume("internal lazy landing-site join evaluation")
            # A typed stop above publishes neither an incomplete exact query
            # nor a partially charged proof receipt.
            charged.add(cache_key)
        if publish_sites:
            self._internal_cache[cache_key] = sites
        return self._remember(
            sites,
            retain=retain,
        )

    @lazy_access_boundary
    def staircase_internal(
        self,
        left_lane: LaneRef,
        right_lane: LaneRef,
        *,
        net: str,
        selection_limit: int,
        budget: _IncidenceBudget,
    ) -> tuple[Any, ...]:
        """Return one exact bounded transition frontier, charged only once."""

        left_id, right_id = sorted((left_lane.lane_id, right_lane.lane_id))
        cache_key = (left_id, right_id, net, selection_limit)
        cached = self._staircase_internal_cache.get(cache_key)
        if cached is not None:
            return cached
        selected = tuple(sorted(
            self.internal(left_lane, right_lane, net=net, retain=False),
            key=lambda site: site.site_id,
        ))[:selection_limit]
        for _site in selected:
            budget.consume("signal staircase lazy landing-site query")
        # A budget stop above leaves no partial proof in the cache.
        self._staircase_internal_cache[cache_key] = selected
        return selected

    @lazy_access_boundary
    def staircase_neighbors(
        self,
        current_lane: LaneRef,
        candidate_lanes: Sequence[LaneRef],
        *,
        net: str,
        selection_limit: int,
        budget: _IncidenceBudget,
    ) -> tuple[tuple[LaneRef, tuple[Any, ...]], ...]:
        """Batch one exact adjacent-layer frontier into sparse legal edges."""

        candidates = tuple(candidate_lanes)
        if not candidates:
            return ()
        if type(selection_limit) is not int or selection_limit <= 0:
            raise PatternCompileInputError(
                "staircase neighbor selection_limit must be positive"
            )
        target_layers = {lane.layer_n for lane in candidates}
        if (
            len(target_layers) != 1
            or any(
                abs(lane.layer_n - current_lane.layer_n) != 1
                for lane in candidates
            )
        ):
            raise PatternCompileInputError(
                "staircase neighbor batch must name one adjacent layer"
            )

        perpendicular_by_point: dict[tuple[int, int], list[LaneRef]] = {}
        same_axis = []
        cached_rows = []
        for candidate in candidates:
            left_id, right_id = sorted(
                (current_lane.lane_id, candidate.lane_id)
            )
            staircase_key = left_id, right_id, net, selection_limit
            cached = self._staircase_internal_cache.get(staircase_key)
            if cached is not None:
                if len(cached) >= selection_limit:
                    cached_rows.append((candidate, cached))
                continue
            if candidate.preferred_axis == current_lane.preferred_axis:
                same_axis.append(candidate)
                continue
            horizontal = (
                current_lane
                if current_lane.preferred_axis == "H"
                else candidate
            )
            vertical = (
                candidate
                if candidate.preferred_axis == "V"
                else current_lane
            )
            point = vertical.center_nm, horizontal.center_nm
            if current_lane.contains_point(point) and candidate.contains_point(point):
                perpendicular_by_point.setdefault(point, []).append(candidate)

        lower_layer_n = min(current_lane.layer_n, next(iter(target_layers)))
        materialized = (
            _query_exact_landing_sites_at_points(
                self.ledger,
                self.world,
                self.tech,
                self.conductor_index,
                lower_layer_n=lower_layer_n,
                points=tuple(sorted(perpendicular_by_point)),
                signed_points=True,
                query_context=self._query_context,
            )
            if perpendicular_by_point
            else ()
        )
        sites_by_point: dict[tuple[int, int], list[Any]] = {}
        for site in materialized:
            sites_by_point.setdefault(site.xy_nm, []).append(site)

        pending_internal: dict[tuple[str, str, str], tuple[Any, ...]] = {}
        pending_staircase: dict[
            tuple[str, str, str, int], tuple[Any, ...]
        ] = {}
        rows = list(cached_rows)
        for point, point_candidates in perpendicular_by_point.items():
            point_sites = tuple(sorted(
                (
                    site
                    for site in sites_by_point.get(point, ())
                    if site.owner_net in (None, net)
                    and current_lane.contains_point(site.xy_nm)
                    and {
                        current_lane.layer_n,
                        site.lower_layer_n,
                        site.upper_layer_n,
                    }
                    == {
                        current_lane.layer_n,
                        next(iter(target_layers)),
                    }
                ),
                key=lambda site: site.site_id,
            ))
            for candidate in point_candidates:
                pair_sites = tuple(
                    site
                    for site in point_sites
                    if candidate.contains_point(site.xy_nm)
                )
                left_id, right_id = sorted(
                    (current_lane.lane_id, candidate.lane_id)
                )
                internal_key = left_id, right_id, net
                staircase_key = left_id, right_id, net, selection_limit
                selected = pair_sites[:selection_limit]
                pending_internal[internal_key] = pair_sites
                pending_staircase[staircase_key] = selected
                if len(selected) >= selection_limit:
                    rows.append((candidate, selected))

        for candidate in same_axis:
            selected = self.staircase_internal(
                current_lane,
                candidate,
                net=net,
                selection_limit=selection_limit,
                budget=budget,
            )
            if len(selected) >= selection_limit:
                rows.append((candidate, selected))

        for selected in pending_staircase.values():
            for _site in selected:
                budget.consume("signal staircase lazy landing-site query")
        # Publish only after the full point batch and its pattern charge close.
        self._internal_cache.update(pending_internal)
        self._staircase_internal_cache.update(pending_staircase)
        return tuple(rows)

    @lazy_access_boundary
    def boundary(
        self,
        terminal: TerminalDemand,
        endpoint: EndpointIncidence,
        lane: LaneRef,
        *,
        net: str,
        required_component_ref: str | None,
        retain: bool = True,
        diagnostics: dict[str, int] | None = None,
        budget: _IncidenceBudget | None = None,
    ) -> tuple[Any, ...]:
        cache_key = (
            terminal.provenance,
            terminal.xy_nm,
            terminal.layer,
            endpoint.pin_access_fact,
            endpoint.access_component_refs,
            endpoint.conductor_component_refs,
            lane.lane_id,
            required_component_ref,
            net,
        )
        cached = self._boundary_cache.get(cache_key)
        if cached is not None:
            charged = getattr(self, "_boundary_join_charged", set())
            if budget is not None and cache_key not in charged:
                budget.consume("owned-component lazy boundary-lane query")
                charged.add(cache_key)
                self._boundary_join_charged = charged
            cached_sites, cached_diagnostics = cached
            if diagnostics is not None:
                for key, count in cached_diagnostics:
                    diagnostics[key] = diagnostics.get(key, 0) + count
            return self._remember(cached_sites, retain=retain)
        terminal_layer_n = self._layer_n_by_key.get(tuple(terminal.layer))
        if terminal_layer_n is None or abs(terminal_layer_n - lane.layer_n) != 1:
            return ()
        lower_layer_n = min(terminal_layer_n, lane.layer_n)
        query_diagnostics: dict[str, int] = {}
        if required_component_ref is None:
            sites = _query_exact_landing_sites_at_points(
                self.ledger,
                self.world,
                self.tech,
                self.conductor_index,
                lower_layer_n=lower_layer_n,
                points=(terminal.xy_nm,),
                signed_points=False,
                query_context=self._query_context,
                diagnostics=query_diagnostics,
            )
            query_diagnostics["landing_points"] = 1
            query_diagnostics["materialized_sites"] = len(sites)
        else:
            query_kwargs = dict(
                component_ref=required_component_ref,
                lane_resource=self._resource(lane),
                lane_track_start=lane.track_start,
                query_context=self._query_context,
            )
            query_kwargs["diagnostics"] = query_diagnostics
            sites = _query_exact_landing_sites_for_component_lane(
                self.ledger,
                self.world,
                self.tech,
                self.conductor_index,
                **query_kwargs,
            )
        accepted = []
        for site in sites:
            if (
                site.owner_net not in (None, net)
                or not lane.contains_point(site.xy_nm)
            ):
                continue
            terminal_refs = (
                site.lower_owner_component_refs
                if terminal_layer_n == site.lower_layer_n
                else site.upper_owner_component_refs
            )
            endpoint_refs = set(endpoint.access_component_refs)
            if required_component_ref is None:
                reached = (
                    site.xy_nm == terminal.xy_nm
                    and bool(endpoint_refs & set(terminal_refs))
                )
            else:
                reached = (
                    required_component_ref in endpoint.conductor_component_refs
                    and required_component_ref in terminal_refs
                )
            if reached:
                accepted.append(site)
        result = tuple(accepted)
        diagnostic_certificate = tuple(sorted(query_diagnostics.items()))
        charged = getattr(self, "_boundary_join_charged", set())
        if budget is not None and cache_key not in charged:
            budget.consume("owned-component lazy boundary-lane query")
            charged.add(cache_key)
            self._boundary_join_charged = charged
        self._boundary_cache[cache_key] = result, diagnostic_certificate
        if diagnostics is not None:
            for key, count in diagnostic_certificate:
                diagnostics[key] = diagnostics.get(key, 0) + count
        return self._remember(result, retain=retain)

    @lazy_access_boundary
    def boundary_for_use(
        self,
        terminal: TerminalDemand,
        endpoint: EndpointIncidence,
        lane: LaneRef,
        use: CorridorUse,
        *,
        net: str,
        required_component_ref: str | None,
        selected_site_ids: tuple[str, ...] | None,
        budget: _IncidenceBudget,
        retain: bool = True,
    ) -> tuple[Any, ...]:
        """Cache one exact boundary-site join through a concrete use domain."""

        selected_key = (
            None
            if selected_site_ids is None
            else tuple(sorted(selected_site_ids))
        )
        cache_key = (
            terminal.provenance,
            terminal.xy_nm,
            terminal.layer,
            endpoint.pin_access_fact,
            endpoint.access_component_refs,
            endpoint.conductor_component_refs,
            lane.lane_id,
            use.resource,
            use.interval_nm,
            use.allowed_track_starts,
            required_component_ref,
            net,
        )
        cache = getattr(self, "_boundary_use_cache", None)
        if cache is None:
            cache = {}
            self._boundary_use_cache = cache
        cached = cache.get(cache_key)
        if cached is not None:
            joined = cached
        else:
            sites = self.boundary(
                terminal,
                endpoint,
                lane,
                net=net,
                required_component_ref=required_component_ref,
                retain=False,
            )
            joined = tuple(
                site
                for site in sites
                if _compiled_use_contains_site_center(lane, use, site)
            )
            for _site in joined:
                budget.consume("boundary lazy landing-site join evaluation")
            # Publish only after the complete exact join and its budget receipt.
            cache[cache_key] = joined
        if selected_key is not None:
            selected_set = set(selected_key)
            joined = tuple(
                site for site in joined if site.site_id in selected_set
            )
        return self._remember(joined, retain=retain)

    @lazy_access_boundary
    def component_lanes(
        self,
        component_ref: str,
        *,
        budget: _IncidenceBudget,
    ) -> tuple[LaneRef, ...]:
        """Conservatively index exact lanes whose via pad can touch a component."""

        cached = self._component_lane_cache.get(component_ref)
        if cached is not None:
            return cached
        component = self.conductor_index.component(component_ref)
        component_layer_n = self._layer_n_by_key.get(component.layer_key)
        if component_layer_n is None:
            return ()
        result: dict[str, LaneRef] = {}
        manufacturing_grid_nm = self._query_context.manufacturing_grid_nm
        for adjacent_layer_n in (
            component_layer_n - 1,
            component_layer_n + 1,
        ):
            if adjacent_layer_n <= 0:
                continue
            lower_layer_n = min(component_layer_n, adjacent_layer_n)
            landing = via_landing(
                self.tech, f"via{lower_layer_n}", 0, 0
            )
            if landing is None:
                continue
            relative_pad_bbox = (
                landing.below_metal_rect
                if component_layer_n == lower_layer_n
                else landing.above_metal_rect
            )
            feasible_regions = tuple(
                region
                for atom in component.atom_facts
                for region in _owned_access_feasible_regions(
                    tuple(int(value) for value in atom[:4]),
                    relative_pad_bbox,
                    manufacturing_grid_nm,
                )
            )
            pin_points = tuple(
                (int(pin[3]), int(pin[4]))
                for pin in component.pin_accesses
            )
            for lane in self._lane_layer_index.touching_regions(
                adjacent_layer_n,
                regions=feasible_regions,
                points=pin_points,
                budget=budget,
            ):
                result[lane.lane_id] = lane
        lanes = tuple(result[key] for key in sorted(result))
        self._component_lane_cache[component_ref] = lanes
        return lanes

    @lazy_access_boundary
    def referenced_sites(
        self, candidates: Sequence[CompleteNetCandidate]
    ) -> tuple[Any, ...]:
        return self.referenced_site_ids({
            site_id
            for candidate in candidates
            for site_id in (
                *(
                    site_id
                    for option in candidate.landing_options
                    for site_id in option.site_ids
                ),
                *(
                    site_id
                    for demand in candidate.landing_demands
                    for site_id in demand.allowed_site_ids
                ),
            )
        })

    @lazy_access_boundary
    def referenced_site_ids(
        self, site_ids: Sequence[str] | set[str]
    ) -> tuple[Any, ...]:
        ids = set(site_ids)
        missing = tuple(sorted(ids - set(self._sites)))
        if missing:
            raise PatternCompileInputError(
                f"lazy landing option names unmaterialized sites {missing}"
            )
        return tuple(self._sites[site_id] for site_id in sorted(ids))


def _compile_unknown(
    code: str,
    *,
    topology_digest: str,
    world_digest: str,
    ledger_digest: str,
    conductor_digest: str,
    incidence_digest: str = "",
    refs: Sequence[str] = (),
    phase: str = "FIXED_PATTERN",
) -> FixedPatternCompileResult:
    digests = [
        ("owned_conductor_index", conductor_digest),
        ("routing_resource_ledger", ledger_digest),
        ("routing_world", world_digest),
        ("topology_plan", topology_digest),
    ]
    if incidence_digest:
        digests.append(("lane_incidence_index", incidence_digest))
    return FixedPatternCompileResult.from_unknown(
        PatternCompileUnknown(
            code=code,
            phase=phase,
            input_digests=tuple(digests),
            refs=tuple(refs),
        )
    )


def _validate_compile_limit(name: str, value: Any, hard_max: int) -> int:
    if type(value) is not int or value <= 0 or value > hard_max:
        raise PatternCompileLimitError(
            f"{name} must be a positive exact integer no greater than {hard_max}"
        )
    return value


def _layer_number_map(tech: Any) -> dict[LayerKey, int]:
    result: dict[LayerKey, int] = {}
    for layer_n in range(1, 33):
        try:
            layer_key = _tech_layer_key(tech, layer_n)
        except PatternCompileInputError:
            break
        if layer_key in result:
            raise PatternCompileInputError(
                f"PDK reuses GDS layer key {layer_key!r} for multiple metals"
            )
        result[layer_key] = layer_n
    if not result:
        raise PatternCompileInputError("PDK exposes no routable metal layer")
    return result


def _pin_fact_provenance(pin: Sequence[Any]) -> str:
    from clay.layout.route.terminal_identity import pin_fact_provenance

    try:
        return pin_fact_provenance(pin)
    except (TypeError, ValueError):
        return ""


def _resolve_terminal_fact(
    routing_world: RoutingWorldProvenance,
    *,
    net: str,
    terminal: TerminalDemand,
) -> tuple[Any, ...] | None:
    matches = tuple(
        pin
        for pin in routing_world.pins
        if pin[0] == net
        and tuple(pin[3:5]) == terminal.xy_nm
        and tuple(pin[5:7]) == terminal.layer
        and _pin_fact_provenance(pin) == terminal.provenance
    )
    if len(matches) != 1:
        return None
    return matches[0]


def _endpoint_access(
    terminal: TerminalDemand,
    endpoint: EndpointIncidence,
    *,
    layer_n_by_key: dict[LayerKey, int],
) -> CandidateEndpointAccess | None:
    layer_n = layer_n_by_key.get(terminal.layer)
    if layer_n is None:
        return None
    return CandidateEndpointAccess(
        provenance=terminal.provenance,
        xy_nm=terminal.xy_nm,
        layer_key=terminal.layer,
        layer_n=layer_n,
        owner_component_refs=endpoint.access_component_refs,
    )


def _lane_run_coordinate(lane: LaneRef, point_nm: PointNm) -> int:
    return point_nm[0] if lane.preferred_axis == "H" else point_nm[1]


def _compiled_use_contains_site_center(
    lane: LaneRef,
    use: CorridorUse,
    site: Any,
) -> bool:
    """Prove one explicit contact lies on its exact compiled track segment."""

    run_nm = _lane_run_coordinate(lane, site.xy_nm)
    return (
        use.resource
        == CorridorLayerKey(lane.resource_corridor_id, lane.layer_n)
        and lane.track_start in use.allowed_track_starts
        and lane.contains_point(site.xy_nm)
        and use.interval_nm[0] <= run_nm < use.interval_nm[1]
    )


def _width_fits_cross_bounds(
    *,
    selected_centers_nm: tuple[int, ...],
    cross_bounds_nm: tuple[int, int],
    width_nm: int,
) -> bool:
    """Replay the exact closed cross-axis metal envelope inside a corridor."""

    if (
        type(width_nm) is not int
        or width_nm <= 0
        or not selected_centers_nm
    ):
        return False
    low_half_nm = width_nm // 2
    high_half_nm = width_nm - low_half_nm
    return (
        cross_bounds_nm[0]
        <= selected_centers_nm[0] - low_half_nm
        and selected_centers_nm[-1] + high_half_nm
        <= cross_bounds_nm[1]
    )


def _lane_width_fits(lane: LaneRef, width_nm: int) -> bool:
    return _width_fits_cross_bounds(
        selected_centers_nm=(lane.center_nm,),
        cross_bounds_nm=lane.corridor_cross_bounds_nm,
        width_nm=width_nm,
    )


def _resource_use_start_width_fits(
    resource: TrackResource,
    use: CorridorUse,
    track_start: int,
    candidate_width_nm: int,
) -> bool:
    bbox = tuple(resource.corridor_bbox_nm or ())
    if len(bbox) != 4:
        return False
    cross_bounds = (
        (bbox[1], bbox[3])
        if resource.preferred_axis == "H"
        else (bbox[0], bbox[2])
    )
    selected_centers = resource.track_centers_nm[
        track_start:track_start + use.track_span
    ]
    if len(selected_centers) != use.track_span:
        return False
    width_nm = (
        use.width_nm if use.width_nm is not None else candidate_width_nm
    )
    return _width_fits_cross_bounds(
        selected_centers_nm=selected_centers,
        cross_bounds_nm=cross_bounds,
        width_nm=width_nm,
    )


def _assert_candidate_use_width_fits(
    candidate: CompleteNetCandidate,
    *,
    resources_by_key: dict[CorridorLayerKey, TrackResource],
) -> None:
    for use in candidate.corridor_uses:
        resource = resources_by_key.get(use.resource)
        if resource is None:
            raise CandidatePoolError(
                "fixed-pattern candidate names an absent corridor resource: "
                f"candidate={candidate.candidate_id}, use={use.use_id}"
            )
        for track_start in use.allowed_track_starts:
            if not _resource_use_start_width_fits(
                resource,
                use,
                track_start,
                candidate.width_nm,
            ):
                raise CandidatePoolError(
                    "fixed-pattern candidate width does not fit corridor for "
                    f"use {use.use_id}: candidate={candidate.candidate_id}, "
                    f"track_start={track_start}"
                )


def _compiled_corridor_use(
    *,
    lane: LaneRef,
    interval_nm: tuple[int, int],
    net: str,
    pattern_kind: str,
    ordinal: int,
    width_nm: int,
) -> CorridorUse | None:
    if not _lane_width_fits(lane, width_nm):
        return None
    payload = {
        "interval_nm": list(interval_nm),
        "lane": lane.canonical_payload(),
        "net": net,
        "ordinal": ordinal,
        "pattern_kind": pattern_kind,
        "schema": "clay.fixed-pattern-corridor-use.v2",
        "width_nm": width_nm,
    }
    return CorridorUse(
        use_id=canonical_pattern_id("pattern-use", payload),
        resource=CorridorLayerKey(
            lane.resource_corridor_id,
            lane.layer_n,
        ),
        interval_nm=interval_nm,
        track_span=1,
        allowed_track_starts=(lane.track_start,),
        width_nm=width_nm,
    )


def _candidate_identity_payload(candidate: CompleteNetCandidate) -> dict[str, Any]:
    return complete_net_candidate_identity_payload(candidate)


def _resolved_endpoint_rows(
    *,
    topology_candidate: TopologyCandidate,
    demand: RoutingDemand,
    routing_world: RoutingWorldProvenance,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
) -> tuple[
    tuple[TerminalDemand, EndpointIncidence, CandidateEndpointAccess], ...
]:
    terminals = tuple(topology_candidate.branch_terminals)
    if len(terminals) != 2 or terminals[0].provenance == terminals[1].provenance:
        return ()
    endpoint_rows: list[
        tuple[TerminalDemand, EndpointIncidence, CandidateEndpointAccess]
    ] = []
    for terminal in terminals:
        pin = _resolve_terminal_fact(
            routing_world,
            net=demand.net,
            terminal=terminal,
        )
        if pin is None:
            return ()
        endpoint = incidence_index.resolve_endpoint(pin)
        if endpoint is None:
            return ()
        access = _endpoint_access(
            terminal,
            endpoint,
            layer_n_by_key=layer_n_by_key,
        )
        if access is None:
            return ()
        endpoint_rows.append((terminal, endpoint, access))
    return tuple(endpoint_rows)


def _resolved_endpoint_rows_any(
    *,
    topology_candidate: TopologyCandidate,
    demand: RoutingDemand,
    routing_world: RoutingWorldProvenance,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
) -> tuple[
    tuple[
        TerminalDemand,
        EndpointIncidence,
        CandidateEndpointAccess,
        tuple[Any, ...],
    ],
    ...,
]:
    """Resolve every declared branch endpoint against one exact world pin."""

    terminals = tuple(topology_candidate.branch_terminals)
    if (
        len(terminals) < 2
        or len({terminal.provenance for terminal in terminals}) != len(terminals)
    ):
        return ()
    endpoint_rows = []
    for terminal in terminals:
        pin = _resolve_terminal_fact(
            routing_world,
            net=demand.net,
            terminal=terminal,
        )
        if pin is None:
            return ()
        endpoint = incidence_index.resolve_endpoint(pin)
        if endpoint is None:
            return ()
        access = _endpoint_access(
            terminal,
            endpoint,
            layer_n_by_key=layer_n_by_key,
        )
        if access is None:
            return ()
        endpoint_rows.append((terminal, endpoint, access, tuple(pin)))
    return tuple(endpoint_rows)


def _content_address_candidate(
    candidate: CompleteNetCandidate,
) -> CompleteNetCandidate:
    return content_address_complete_net_candidate(candidate)


def _compile_same_component_candidate(
    *,
    topology_candidate: TopologyCandidate,
    demand: RoutingDemand,
    routing_world: RoutingWorldProvenance,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
    topology_digest: str,
    budget: _IncidenceBudget,
) -> tuple[CompleteNetCandidate, ...]:
    if topology_candidate.topology_class != "local_signal":
        return ()
    endpoint_rows = _resolved_endpoint_rows(
        topology_candidate=topology_candidate,
        demand=demand,
        routing_world=routing_world,
        incidence_index=incidence_index,
        layer_n_by_key=layer_n_by_key,
    )
    if not endpoint_rows:
        return ()
    oriented = _orient_two_pin_branch(endpoint_rows, demand)
    if oriented is None:
        return ()
    source, target = oriented
    common_refs = tuple(sorted(
        set(source[1].conductor_component_refs)
        & set(target[1].conductor_component_refs)
    ))
    candidates = []
    for component_ref in common_refs:
        budget.consume("same-component two-pin expansion")
        component = conductor_index.component(component_ref)
        if component.net != demand.net or not component.atom_facts:
            continue
        branch_id = canonical_pattern_id(
            "pattern-branch",
            {
                "component_ref": component_ref,
                "net": demand.net,
                "pattern_kind": "COMPONENT_ONLY",
                "source": source[0].provenance,
                "target": target[0].provenance,
            },
        )
        candidate = CompleteNetCandidate(
            candidate_id="pending-content-address",
            net=demand.net,
            topology_class=topology_candidate.topology_class,
            branch_edges=((source[0].provenance, target[0].provenance),),
            corridor_uses=(),
            landing_demands=(),
            required_goal_refs=topology_candidate.required_goal_refs,
            width_nm=topology_candidate.width_nm,
            via_multiplicity=0,
            source_topology_digest=topology_digest,
            program_kind="COMPONENT_ONLY",
            branches=(
                CandidateBranch(
                    branch_id=branch_id,
                    source_endpoint=source[0].provenance,
                    target_endpoint=target[0].provenance,
                    ordered_use_ids=(),
                    ordered_landing_demand_ids=(),
                    ordered_steps=(
                        CandidateBranchStep("owned_component", component_ref),
                    ),
                ),
            ),
            goal_branch_coverage=tuple(
                GoalBranchCoverage(
                    declaration_ref=goal.declaration_ref,
                    branch_id=branch_id,
                    target_provenance=target[0].provenance,
                )
                for goal in demand.required_goals
            ),
            endpoint_accesses=(source[2], target[2]),
            owned_component_use_refs=(component_ref,),
        )
        candidates.append(_content_address_candidate(candidate))
    return tuple(candidates)


def _shared_signal_family_view(
    candidate: CompleteNetCandidate,
    *,
    include_intervals: bool = True,
) -> tuple[tuple[Any, ...], tuple[str, ...], tuple[str, ...], tuple[CandidateBranch, ...]]:
    """Return the exact structural schema used to merge track variants."""

    branches = tuple(sorted(
        candidate.branches,
        key=lambda branch: (
            branch.source_endpoint,
            branch.target_endpoint,
            branch.branch_id,
        ),
    ))
    use_ids: list[str] = []
    demand_ids: list[str] = []
    for branch in branches:
        for step in branch.ordered_steps:
            if step.kind == "track_use" and step.ref_id not in use_ids:
                use_ids.append(step.ref_id)
            elif (
                step.kind == "landing_demand"
                and step.ref_id not in demand_ids
            ):
                demand_ids.append(step.ref_id)
    for use in sorted(candidate.corridor_uses, key=lambda item: item.use_id):
        if use.use_id not in use_ids:
            use_ids.append(use.use_id)
    for demand in sorted(
        candidate.landing_demands, key=lambda item: item.demand_id
    ):
        if demand.demand_id not in demand_ids:
            demand_ids.append(demand.demand_id)
    use_position = {use_id: index for index, use_id in enumerate(use_ids)}
    demand_position = {
        demand_id: index for index, demand_id in enumerate(demand_ids)
    }
    uses_by_id = {use.use_id: use for use in candidate.corridor_uses}
    demands_by_id = {
        demand.demand_id: demand for demand in candidate.landing_demands
    }
    options_by_demand: dict[str, list[LandingOption]] = {}
    for option in candidate.landing_options:
        options_by_demand.setdefault(option.demand_id, []).append(option)

    def use_is_choice_bound(use_id: str) -> bool:
        # A merged allowed track start is sound only when at least one required
        # landing choice always names that use.  The CP then cannot select a
        # start that was never certified by a concrete family member.  Direct
        # same-layer component contacts have no landing option, so their starts
        # remain in the family key and are never widened here.
        if any(
            options
            and all(
                any(contact.use_id == use_id for contact in option.contacts)
                for option in options
            )
            for options in options_by_demand.values()
        ):
            return True
        derived_demand_ids = {
            demand.demand_id
            for demand in candidate.landing_demands
            if demand.selection_mode == "DERIVED_EXACT_SITE_DOMAIN"
        }
        return any(
            step.kind == "landing_demand"
            and step.ref_id in derived_demand_ids
            and any(
                0 <= neighbour_index < len(branch.ordered_steps)
                and branch.ordered_steps[neighbour_index].kind == "track_use"
                and branch.ordered_steps[neighbour_index].ref_id == use_id
                for neighbour_index in (step_index - 1, step_index + 1)
            )
            for branch in branches
            for step_index, step in enumerate(branch.ordered_steps)
        )

    normalized_branches = tuple(
        (
            branch.source_endpoint,
            branch.target_endpoint,
            tuple(
                (
                    step.kind,
                    use_position[step.ref_id]
                    if step.kind == "track_use"
                    else demand_position[step.ref_id]
                    if step.kind == "landing_demand"
                    else step.ref_id,
                )
                for step in branch.ordered_steps
            ),
        )
        for branch in branches
    )
    branch_position = {
        branch.branch_id: index for index, branch in enumerate(branches)
    }
    key = (
        candidate.net,
        candidate.topology_class,
        candidate.program_kind,
        candidate.required_goal_refs,
        candidate.width_nm,
        candidate.via_multiplicity,
        candidate.source_topology_digest,
        candidate.endpoint_accesses,
        candidate.owned_component_use_refs,
        normalized_branches,
        tuple(
            (
                (
                    uses_by_id[use_id].interval_nm
                    if include_intervals
                    else None
                ),
                uses_by_id[use_id].resource,
                uses_by_id[use_id].track_span,
                uses_by_id[use_id].width_nm,
                None
                if use_is_choice_bound(use_id)
                else uses_by_id[use_id].allowed_track_starts,
            )
            for use_id in use_ids
        ),
        tuple(
            (
                demands_by_id[demand_id].terminal_provenance,
                demands_by_id[demand_id].required_multiplicity,
                demands_by_id[demand_id].declaration_ref,
                demands_by_id[demand_id].selection_mode,
            )
            for demand_id in demand_ids
        ),
        tuple(sorted(
            (
                coverage.declaration_ref,
                branch_position[coverage.branch_id],
                coverage.target_provenance,
            )
            for coverage in candidate.goal_branch_coverage
        )),
    )
    return key, tuple(use_ids), tuple(demand_ids), branches


def _merge_shared_signal_candidate_families(
    candidates: tuple[CompleteNetCandidate, ...],
    *,
    ledger: ExplicitRoutingResourceLedger,
    lazy_landing_access: _ExactLazyLandingAccess | None,
) -> tuple[CompleteNetCandidate, ...]:
    """Merge proved shared-star track variants into exact corridor families.

    Every member is already an exact lane/site program.  Only members with the
    same branch-step schema, resources, owned components, and landing roles are
    combined.  Track starts are widened only when a required landing choice
    binds the use, so the allocator still selects a concrete proved path rather
    than an uncorrelated synthetic lane.
    """

    families: dict[tuple[Any, ...], list[CompleteNetCandidate]] = {}
    views: dict[str, tuple[tuple[Any, ...], tuple[str, ...], tuple[str, ...], tuple[CandidateBranch, ...]]] = {}
    for candidate in candidates:
        view = _shared_signal_family_view(candidate)
        views[candidate.candidate_id] = view
        families.setdefault(view[0], []).append(candidate)

    # The public landing-option cap applies to the content-addressed family IR,
    # never to pre-merge variants.  Raw enumeration has its own pattern-search
    # budget and hard transient bound, so splitting a family at the public cap
    # here would both change its content hash and prevent exact deduplication.
    work_items = tuple(
        (
            family_key,
            tuple(sorted(
                families[family_key], key=lambda item: item.candidate_id
            )),
        )
        for family_key in sorted(families, key=repr)
    )

    merged: list[CompleteNetCandidate] = []
    resources_by_key = {
        resource.key: resource for resource in ledger.track_resources
    }
    for candidate in candidates:
        _assert_candidate_use_width_fits(
            candidate,
            resources_by_key=resources_by_key,
        )
    sites_by_id = {site.site_id: site for site in ledger.landing_sites}
    if lazy_landing_access is not None:
        sites_by_id.update(lazy_landing_access._sites)

    def assert_member_contact_correlation(
        member: CompleteNetCandidate,
    ) -> None:
        uses_by_id = {use.use_id: use for use in member.corridor_uses}
        demands_by_id = {
            demand.demand_id: demand for demand in member.landing_demands
        }
        for option in member.landing_options:
            for contact in option.contacts:
                use = uses_by_id.get(contact.use_id)
                resource = (
                    None if use is None else resources_by_key.get(use.resource)
                )
                site = sites_by_id.get(contact.site_id)
                if use is None or resource is None or site is None:
                    raise CandidatePoolError(
                        "shared-signal family contact references unknown "
                        "use/resource/site"
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
                    contact.track_start not in use.allowed_track_starts
                    or cross_coordinate not in selected_centers
                    or not (
                        use.interval_nm[0]
                        <= run_coordinate
                        < use.interval_nm[1]
                    )
                ):
                    raise CandidatePoolError(
                        "shared-signal family contact has unproved "
                        "track-start/site correlation: "
                        f"cross={cross_coordinate}, "
                        f"selected_centers={selected_centers}, "
                        f"run={run_coordinate}, interval={use.interval_nm}, "
                        f"track_start={contact.track_start}, "
                        f"allowed={use.allowed_track_starts}, "
                        f"site={contact.site_id}, use={contact.use_id}, "
                        f"resource={resource.key!r}, "
                        f"terminal={demands_by_id[option.demand_id].terminal_provenance}, "
                        f"candidate={member.candidate_id}"
                    )

    for family_key, members in work_items:
        for member in members:
            assert_member_contact_correlation(member)
        if len(members) == 1:
            merged.append(members[0])
            continue
        exemplar = members[0]
        _, exemplar_use_ids, exemplar_demand_ids, exemplar_branches = views[
            exemplar.candidate_id
        ]
        member_use_rows = []
        member_demand_rows = []
        for member in members:
            _, use_ids, demand_ids, _branches = views[member.candidate_id]
            uses_by_id = {use.use_id: use for use in member.corridor_uses}
            demands_by_id = {
                demand.demand_id: demand
                for demand in member.landing_demands
            }
            member_use_rows.append(tuple(uses_by_id[use_id] for use_id in use_ids))
            member_demand_rows.append(tuple(
                demands_by_id[demand_id] for demand_id in demand_ids
            ))

        group_ref = canonical_pattern_hash({
            "candidate_ids": [member.candidate_id for member in members],
            "schema": "clay.shared-signal-corridor-family.v1",
        })
        new_uses = []
        for ordinal in range(len(exemplar_use_ids)):
            variants = tuple(row[ordinal] for row in member_use_rows)
            interval_nm = variants[0].interval_nm
            if any(use.interval_nm != interval_nm for use in variants):
                raise CandidatePoolError(
                    "exact family merge crossed corridor-use intervals"
                )
            starts = tuple(sorted({
                start
                for use in variants
                for start in use.allowed_track_starts
            }))
            new_uses.append(CorridorUse(
                use_id=canonical_pattern_id(
                    "pattern-use-family",
                    {
                        "family": group_ref,
                        "interval_nm": list(interval_nm),
                        "ordinal": ordinal,
                        "resource": {
                            "corridor_id": variants[0].resource.corridor_id,
                            "layer_n": variants[0].resource.layer_n,
                        },
                        "track_span": variants[0].track_span,
                        "track_starts": list(starts),
                    },
                ),
                resource=variants[0].resource,
                interval_nm=interval_nm,
                track_span=variants[0].track_span,
                allowed_track_starts=starts,
                width_nm=variants[0].width_nm,
            ))

        new_demand_ids = tuple(
            canonical_pattern_id(
                "pattern-landing-demand-family",
                {"family": group_ref, "ordinal": ordinal},
            )
            for ordinal in range(len(exemplar_demand_ids))
        )
        converted_options: dict[str, LandingOption] = {}
        new_uses_by_id = {use.use_id: use for use in new_uses}
        for member, use_row, demand_row in zip(
            members, member_use_rows, member_demand_rows
        ):
            use_position = {
                use.use_id: ordinal for ordinal, use in enumerate(use_row)
            }
            demand_position = {
                demand.demand_id: ordinal
                for ordinal, demand in enumerate(demand_row)
            }
            for option in member.landing_options:
                demand_id = new_demand_ids[demand_position[option.demand_id]]
                contacts = tuple(
                    LandingTrackContact(
                        site_id=contact.site_id,
                        track_choice=TrackChoiceRef(
                            use_id=new_uses[
                                use_position[contact.use_id]
                            ].use_id,
                            track_start=contact.track_start,
                        ),
                    )
                    for contact in option.contacts
                )
                option_id = canonical_pattern_id(
                    "pattern-landing-option-family",
                    {
                        "contacts": [
                            {
                                "site_id": contact.site_id,
                                "track_start": contact.track_start,
                                "use_id": contact.use_id,
                            }
                            for contact in contacts
                        ],
                        "demand_id": demand_id,
                        "site_ids": list(option.site_ids),
                    },
                )
                for contact in contacts:
                    use = new_uses_by_id[contact.use_id]
                    resource = resources_by_key[use.resource]
                    site = sites_by_id[contact.site_id]
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
                            "shared-signal family merge broke "
                            "track-start/site correlation"
                        )
                converted_options[option_id] = LandingOption(
                    option_id=option_id,
                    demand_id=demand_id,
                    site_ids=option.site_ids,
                    contacts=contacts,
                )

        options_by_new_demand = {
            demand_id: tuple(
                option
                for option in converted_options.values()
                if option.demand_id == demand_id
            )
            for demand_id in new_demand_ids
        }
        if any(not options for options in options_by_new_demand.values()):
            continue
        new_demands = []
        for ordinal, demand_id in enumerate(new_demand_ids):
            variants = tuple(row[ordinal] for row in member_demand_rows)
            new_demands.append(LandingDemand(
                demand_id=demand_id,
                terminal_provenance=variants[0].terminal_provenance,
                allowed_site_ids=tuple(sorted({
                    site_id
                    for option in options_by_new_demand[demand_id]
                    for site_id in option.site_ids
                })),
                required_multiplicity=variants[0].required_multiplicity,
                declaration_ref=variants[0].declaration_ref,
                selection_mode=variants[0].selection_mode,
            ))

        exemplar_use_position = {
            use_id: ordinal for ordinal, use_id in enumerate(exemplar_use_ids)
        }
        exemplar_demand_position = {
            demand_id: ordinal
            for ordinal, demand_id in enumerate(exemplar_demand_ids)
        }
        new_branches = []
        branch_id_map = {}
        for branch in exemplar_branches:
            steps = tuple(
                CandidateBranchStep(
                    step.kind,
                    new_uses[exemplar_use_position[step.ref_id]].use_id
                    if step.kind == "track_use"
                    else new_demand_ids[
                        exemplar_demand_position[step.ref_id]
                    ]
                    if step.kind == "landing_demand"
                    else step.ref_id,
                )
                for step in branch.ordered_steps
            )
            branch_id = canonical_pattern_id(
                "pattern-branch-family",
                {
                    "family": group_ref,
                    "source": branch.source_endpoint,
                    "target": branch.target_endpoint,
                    "steps": [
                        {"kind": step.kind, "ref_id": step.ref_id}
                        for step in steps
                    ],
                },
            )
            branch_id_map[branch.branch_id] = branch_id
            new_branches.append(CandidateBranch(
                branch_id=branch_id,
                source_endpoint=branch.source_endpoint,
                target_endpoint=branch.target_endpoint,
                ordered_use_ids=tuple(
                    step.ref_id for step in steps if step.kind == "track_use"
                ),
                ordered_landing_demand_ids=tuple(
                    step.ref_id
                    for step in steps
                    if step.kind == "landing_demand"
                ),
                ordered_steps=steps,
            ))

        candidate = CompleteNetCandidate(
            candidate_id="pending-content-address",
            net=exemplar.net,
            topology_class=exemplar.topology_class,
            branch_edges=tuple(
                (branch.source_endpoint, branch.target_endpoint)
                for branch in new_branches
            ),
            corridor_uses=tuple(new_uses),
            landing_demands=tuple(new_demands),
            required_goal_refs=exemplar.required_goal_refs,
            width_nm=exemplar.width_nm,
            via_multiplicity=exemplar.via_multiplicity,
            source_topology_digest=exemplar.source_topology_digest,
            program_kind=exemplar.program_kind,
            branches=tuple(new_branches),
            landing_options=tuple(
                converted_options[key] for key in sorted(converted_options)
            ),
            goal_branch_coverage=tuple(
                GoalBranchCoverage(
                    declaration_ref=coverage.declaration_ref,
                    branch_id=branch_id_map[coverage.branch_id],
                    target_provenance=coverage.target_provenance,
                )
                for coverage in exemplar.goal_branch_coverage
            ),
            endpoint_accesses=exemplar.endpoint_accesses,
            owned_component_use_refs=exemplar.owned_component_use_refs,
        )
        _assert_candidate_use_width_fits(
            candidate,
            resources_by_key=resources_by_key,
        )
        merged.append(_content_address_candidate(candidate))
    return tuple(sorted(merged, key=lambda item: item.candidate_id))


def _factor_final_candidate_landing_domains(
    candidate: CompleteNetCandidate,
    *,
    ledger: ExplicitRoutingResourceLedger,
    lazy_landing_access: _ExactLazyLandingAccess | None,
) -> CompleteNetCandidate:
    """Replace exact singleton alias tables by replayable site domains.

    This conversion is deliberately fail-closed.  A demand is factored only
    when every site has one explicit atom and exact geometry proves that the
    atom's track starts are the complete compatible domain.  Sparse or
    ambiguous contact relations remain explicit.
    """

    resources_by_key = {
        resource.key: resource for resource in ledger.track_resources
    }
    sites_by_id = {site.site_id: site for site in ledger.landing_sites}
    if lazy_landing_access is not None:
        sites_by_id.update(lazy_landing_access._sites)
    uses_by_id = {use.use_id: use for use in candidate.corridor_uses}
    neighbour_use_sets = {
        demand.demand_id: set() for demand in candidate.landing_demands
    }
    neighbour_component_sets = {
        demand.demand_id: set() for demand in candidate.landing_demands
    }
    for branch in candidate.branches:
        for step_index, step in enumerate(branch.ordered_steps):
            if step.kind != "landing_demand":
                continue
            for neighbour_index in (step_index - 1, step_index + 1):
                if not (0 <= neighbour_index < len(branch.ordered_steps)):
                    continue
                neighbour = branch.ordered_steps[neighbour_index]
                if neighbour.kind == "track_use":
                    neighbour_use_sets[step.ref_id].add(neighbour.ref_id)
                elif neighbour.kind == "owned_component":
                    neighbour_component_sets[step.ref_id].add(neighbour.ref_id)
    neighbour_uses = {
        demand_id: tuple(sorted(use_ids))
        for demand_id, use_ids in neighbour_use_sets.items()
    }
    neighbour_components = {
        demand_id: tuple(sorted(component_refs))
        for demand_id, component_refs in neighbour_component_sets.items()
    }
    options_by_demand: dict[str, list[LandingOption]] = {
        demand.demand_id: [] for demand in candidate.landing_demands
    }
    for option in candidate.landing_options:
        options_by_demand[option.demand_id].append(option)
    endpoint_boundary_demand_ids = {
        step.ref_id
        for branch in candidate.branches
        for step in (branch.ordered_steps[0], branch.ordered_steps[-1])
        if step.kind == "landing_demand"
    }

    derived_demand_ids: set[str] = set()
    for demand in candidate.landing_demands:
        options = tuple(options_by_demand[demand.demand_id])
        if (
            demand.demand_id in endpoint_boundary_demand_ids
            or
            demand.selection_mode
            not in {
                LANDING_SELECTION_EXPLICIT_OPTION,
                LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY,
            }
            or len(options) != len(demand.allowed_site_ids)
            or any(len(option.site_ids) != 1 for option in options)
        ):
            continue
        option_by_site = {option.site_ids[0]: option for option in options}
        if set(option_by_site) != set(demand.allowed_site_ids):
            continue
        exact = True
        for site_id in demand.allowed_site_ids:
            site = sites_by_id.get(site_id)
            option = option_by_site[site_id]
            if site is None:
                exact = False
                break
            if not set(neighbour_components[demand.demand_id]).issubset({
                *site.lower_owner_component_refs,
                *site.upper_owner_component_refs,
            }):
                exact = False
                break
            contacts_by_use = {
                contact.use_id: contact
                for contact in option.contacts
                if contact.site_id == site_id
            }
            if set(contacts_by_use) != set(neighbour_uses[demand.demand_id]):
                exact = False
                break
            covered_layers = set()
            if site.owner_net == candidate.net and site.lower_owner_component_refs:
                covered_layers.add(site.lower_layer_n)
            if site.owner_net == candidate.net and site.upper_owner_component_refs:
                covered_layers.add(site.upper_layer_n)
            for use_id in neighbour_uses[demand.demand_id]:
                use = uses_by_id[use_id]
                resource = resources_by_key[use.resource]
                compatible_starts = tuple(
                    start
                    for start in use.allowed_track_starts
                    if resource.key.layer_n
                    in (site.lower_layer_n, site.upper_layer_n)
                    and resource.track_centers_nm[
                        start:start + use.track_span
                    ]
                    and (
                        site.xy_nm[1]
                        if resource.preferred_axis == "H"
                        else site.xy_nm[0]
                    ) in resource.track_centers_nm[
                        start:start + use.track_span
                    ]
                    and use.interval_nm[0]
                    <= (
                        site.xy_nm[0]
                        if resource.preferred_axis == "H"
                        else site.xy_nm[1]
                    )
                    < use.interval_nm[1]
                )
                if compatible_starts != (contacts_by_use[use_id].track_start,):
                    exact = False
                    break
                covered_layers.add(resource.key.layer_n)
            if not exact or not {
                site.lower_layer_n, site.upper_layer_n
            }.issubset(covered_layers):
                exact = False
                break
        if exact:
            derived_demand_ids.add(demand.demand_id)

    if not derived_demand_ids:
        return candidate
    factored = replace(
        candidate,
        candidate_id="pending-content-address",
        landing_demands=tuple(
            replace(
                demand,
                selection_mode="DERIVED_EXACT_SITE_DOMAIN",
            )
            if demand.demand_id in derived_demand_ids
            else demand
            for demand in candidate.landing_demands
        ),
        landing_options=tuple(
            option
            for option in candidate.landing_options
            if option.demand_id not in derived_demand_ids
        ),
    )
    return _content_address_candidate(factored)


@dataclass(frozen=True)
class _SignalTrunkEndpointAccess:
    """One endpoint-to-trunk path compiled from exact lane/site resources."""

    corridor_uses: tuple[CorridorUse, ...]
    landing_demands: tuple[LandingDemand, ...]
    landing_options: tuple[LandingOption, ...]
    steps_to_trunk: tuple[CandidateBranchStep, ...]
    owned_component_refs: tuple[str, ...]


def _lanes_geometrically_intersect(left: LaneRef, right: LaneRef) -> bool:
    if left.preferred_axis != right.preferred_axis:
        horizontal = left if left.preferred_axis == "H" else right
        vertical = right if right.preferred_axis == "V" else left
        point = (vertical.center_nm, horizontal.center_nm)
        return horizontal.contains_point(point) and vertical.contains_point(point)
    if left.center_nm != right.center_nm:
        return False
    return max(left.through_interval_nm[0], right.through_interval_nm[0]) < min(
        left.through_interval_nm[1], right.through_interval_nm[1]
    )


def _exact_internal_sites(
    *,
    left_lane: LaneRef,
    right_lane: LaneRef,
    net: str,
    ledger: ExplicitRoutingResourceLedger,
    incidence_index: LaneIncidenceIndex,
    budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None,
    selection_limit: int,
) -> tuple[Any, ...]:
    if lazy_landing_access is not None:
        return lazy_landing_access.staircase_internal(
            left_lane,
            right_lane,
            net=net,
            selection_limit=selection_limit,
            budget=budget,
        )
    return incidence_index.landing_sites_for_lane_pair(
        left_lane,
        right_lane,
        net=net,
        ledger=ledger,
        budget=budget,
        selection_limit=selection_limit,
    )


def _exact_boundary_sites(
    *,
    terminal: TerminalDemand,
    endpoint: EndpointIncidence,
    component_ref: str,
    lane: LaneRef,
    net: str,
    ledger: ExplicitRoutingResourceLedger,
    incidence_index: LaneIncidenceIndex,
    budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None,
    selection_limit: int,
) -> tuple[Any, ...]:
    if lazy_landing_access is not None:
        sites = tuple(sorted(lazy_landing_access.boundary(
            terminal,
            endpoint,
            lane,
            net=net,
            required_component_ref=component_ref,
            retain=False,
        ), key=lambda site: site.site_id))[:selection_limit]
        for _site in sites:
            budget.consume("signal staircase lazy boundary-site query")
        return sites
    return incidence_index.landing_sites_for_component_lane(
        terminal,
        endpoint,
        component_ref,
        lane,
        net=net,
        ledger=ledger,
        budget=budget,
        selection_limit=selection_limit,
    )


def _ordered_staircase_boundary_lanes(
    lanes: Sequence[LaneRef],
    *,
    terminal: TerminalDemand,
    trunk_layer_n: int,
    cache_key: tuple[Any, ...],
    cache: dict[tuple[tuple[Any, ...], int], tuple[LaneRef, ...]],
) -> tuple[LaneRef, ...]:
    """Reuse one exact terminal-relative order across same-layer trunks."""

    order_key = cache_key, trunk_layer_n
    cached = cache.get(order_key)
    if cached is not None:
        return cached
    complete = tuple(sorted(
        lanes,
        key=lambda lane: (
            abs(lane.layer_n - trunk_layer_n),
            abs(
                lane.center_nm
                - (
                    terminal.xy_nm[1]
                    if lane.preferred_axis == "H"
                    else terminal.xy_nm[0]
                )
            ),
            lane.lane_id,
        ),
    ))
    cache[order_key] = complete
    return complete


def _compile_monotone_signal_staircase(
    *,
    row: tuple[
        TerminalDemand,
        EndpointIncidence,
        CandidateEndpointAccess,
        tuple[str, str, str, int, int, str, int],
    ],
    trunk_lane: LaneRef,
    trunk_use: CorridorUse,
    net: str,
    ordinal_base: int,
    allowed_layers: frozenset[int],
    required_multiplicity: int,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    tech: Any,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None,
    lane_layer_index: _BudgetedLaneLayerIndex,
    component_boundary_lane_cache: dict[
        tuple[Any, ...], tuple[LaneRef, ...]
    ],
    staircase_edge_cache: dict[
        tuple[str, str, str, int], tuple[Any, ...]
    ],
    staircase_neighbor_cache: dict[
        tuple[str, str, int, int],
        dict[str, tuple[LaneRef, tuple[Any, ...]]],
    ],
    staircase_boundary_order_cache: dict[
        tuple[tuple[Any, ...], int], tuple[LaneRef, ...]
    ],
) -> _SignalTrunkEndpointAccess | None:
    """Compile a monotone adjacent-layer staircase from artwork to a trunk.

    High-metal passive/device terminals are ordinary endpoint components.  Their
    route to a low trunk is a short alternating-axis path through exact PDK lanes
    and exact via sites; no cap/circuit identity or reference coordinate enters the
    search.  Every selected transition is retained as one landing obligation.
    """

    terminal, endpoint, _access, _pin = row
    multiplicity = max(1, required_multiplicity)
    def chosen_sites(sites: tuple[Any, ...]) -> tuple[Any, ...] | None:
        return sites[:multiplicity] if len(sites) >= multiplicity else None

    # Every call below belongs to the same endpoint, exact trunk, allowed
    # layer closure, and deterministic ordering.  A proven-unreachable suffix
    # is therefore independent of which certified component boundary lane
    # reached it and must be shared across all starts.
    failed: set[str] = set()

    def path_from(
        start_lane: LaneRef,
    ) -> tuple[tuple[LaneRef, ...], tuple[tuple[Any, ...], ...]] | None:
        if start_lane.layer_n == trunk_lane.layer_n:
            return None
        step = 1 if trunk_lane.layer_n > start_lane.layer_n else -1
        if any(
            layer_n not in allowed_layers
            for layer_n in range(
                start_lane.layer_n,
                trunk_lane.layer_n + step,
                step,
            )
        ):
            return None
        def edge_sites(
            left_lane: LaneRef, right_lane: LaneRef
        ) -> tuple[Any, ...] | None:
            left_id, right_id = sorted(
                (left_lane.lane_id, right_lane.lane_id)
            )
            cache_key = (net, left_id, right_id, multiplicity)
            if cache_key in staircase_edge_cache:
                cached = staircase_edge_cache[cache_key]
                return cached or None
            budget.consume("signal staircase lane expansion")
            sites = chosen_sites(_exact_internal_sites(
                left_lane=left_lane,
                right_lane=right_lane,
                net=net,
                ledger=ledger,
                incidence_index=incidence_index,
                budget=budget,
                lazy_landing_access=lazy_landing_access,
                selection_limit=multiplicity,
            ))
            staircase_edge_cache[cache_key] = () if sites is None else sites
            return sites

        def exact_neighbors(
            current: LaneRef, next_layer_n: int
        ) -> dict[str, tuple[LaneRef, tuple[Any, ...]]]:
            cache_key = (net, current.lane_id, next_layer_n, multiplicity)
            cached = staircase_neighbor_cache.get(cache_key)
            if cached is not None:
                return cached
            candidates = lane_layer_index.intersecting(
                current, next_layer_n, budget=budget
            )
            if lazy_landing_access is not None:
                ordered = tuple(sorted(
                    lazy_landing_access.staircase_neighbors(
                        current,
                        candidates,
                        net=net,
                        selection_limit=multiplicity,
                        budget=budget,
                    ),
                    key=lambda row: row[0].lane_id,
                ))
                complete = {row[0].lane_id: row for row in ordered}
                staircase_neighbor_cache[cache_key] = complete
                for candidate, sites in complete.values():
                    left_id, right_id = sorted(
                        (current.lane_id, candidate.lane_id)
                    )
                    staircase_edge_cache[
                        (net, left_id, right_id, multiplicity)
                    ] = sites
                return complete
            rows = []
            for candidate in candidates:
                # A complete ledger already carries a replayed sparse pair
                # index.  Do not call the exact site resolver for certified
                # empty pairs merely because their lanes cross geometrically.
                if (
                    not incidence_index.has_landing_pair_incidence(
                        current, candidate
                    )
                ):
                    continue
                sites = edge_sites(current, candidate)
                if sites is not None:
                    rows.append((candidate, sites))
            complete = {
                row[0].lane_id: row
                for row in sorted(rows, key=lambda row: row[0].lane_id)
            }
            staircase_neighbor_cache[cache_key] = complete
            return complete

        def visit(
            current: LaneRef,
        ) -> tuple[tuple[LaneRef, ...], tuple[tuple[Any, ...], ...]] | None:
            if current.lane_id in failed:
                return None
            next_layer_n = current.layer_n + step
            if next_layer_n == trunk_lane.layer_n:
                final_row = exact_neighbors(current, next_layer_n).get(
                    trunk_lane.lane_id
                )
                if final_row is None:
                    # This suffix state is conclusively unreachable for the
                    # exact trunk.  Record it just like a deeper exhausted
                    # state; otherwise every converging predecessor replays
                    # the same cached final miss and makes the layered DFS
                    # exponential despite complete pair certificates.
                    failed.add(current.lane_id)
                    return None
                _, sites = final_row
                return (current, trunk_lane), (sites,)
            ordered = tuple(sorted(
                exact_neighbors(current, next_layer_n).values(),
                key=lambda row: _staircase_neighbor_order_key(
                    row[0],
                    trunk_lane=trunk_lane,
                    terminal=terminal,
                ),
            ))
            for next_lane, sites in ordered:
                suffix = visit(next_lane)
                if suffix is not None:
                    return (
                        (current,) + suffix[0],
                        (sites,) + suffix[1],
                    )
            failed.add(current.lane_id)
            return None

        return visit(start_lane)

    components = _endpoint_conductor_components(
        endpoint, conductor_index=conductor_index, net=net
    )
    for component_ref, _component in components:
        cache_key = _component_boundary_cache_key(
            terminal, endpoint, component_ref
        )
        boundary_lanes = component_boundary_lane_cache.get(cache_key)
        if boundary_lanes is None:
            boundary_lanes = _component_boundary_lanes(
                terminal=terminal,
                endpoint=endpoint,
                component_ref=component_ref,
                net=net,
                ledger=ledger,
                incidence_index=incidence_index,
                budget=budget,
                lazy_landing_access=lazy_landing_access,
            )
            component_boundary_lane_cache[cache_key] = boundary_lanes
        for start_lane in _ordered_staircase_boundary_lanes(
            boundary_lanes,
            terminal=terminal,
            trunk_layer_n=trunk_lane.layer_n,
            cache_key=cache_key,
            cache=staircase_boundary_order_cache,
        ):
            if (
                start_lane.layer_n not in allowed_layers
                or start_lane.layer_n == trunk_lane.layer_n
            ):
                continue
            found = path_from(start_lane)
            if found is None:
                continue
            boundary_sites = chosen_sites(_exact_boundary_sites(
                terminal=terminal,
                endpoint=endpoint,
                component_ref=component_ref,
                lane=start_lane,
                net=net,
                ledger=ledger,
                incidence_index=incidence_index,
                budget=budget,
                lazy_landing_access=lazy_landing_access,
                selection_limit=multiplicity,
            ))
            if boundary_sites is None:
                continue
            path_lanes, transition_sites = found
            nontrunk_lanes = path_lanes[:-1]
            uses = []
            for index, lane in enumerate(nontrunk_lanes):
                incident_sites = (
                    (boundary_sites if index == 0 else transition_sites[index - 1])
                    + transition_sites[index]
                )
                coordinates = tuple(
                    _lane_run_coordinate(lane, site.xy_nm)
                    for site in incident_sites
                )
                interval_nm = min(coordinates), max(coordinates) + 1
                use = _compiled_corridor_use(
                    lane=lane,
                    interval_nm=interval_nm,
                    net=net,
                    pattern_kind="SHARED_SIGNAL_STAIRCASE",
                    ordinal=ordinal_base + index,
                    width_nm=_tech_min_width_nm(tech, lane.layer_n),
                )
                if use is None:
                    break
                uses.append(use)
            if len(uses) != len(nontrunk_lanes):
                continue
            boundary_bundle = _landing_bundle(
                net=net,
                terminal=terminal,
                endpoint=endpoint,
                lane=start_lane,
                use=uses[0],
                ledger=ledger,
                incidence_index=incidence_index,
                required_multiplicity=required_multiplicity,
                budget=budget,
                option_budget=option_budget,
                required_component_ref=component_ref,
                lazy_landing_access=lazy_landing_access,
                selected_site_ids=tuple(
                    site.site_id for site in boundary_sites
                ),
            )
            if boundary_bundle is None:
                continue
            demands = [boundary_bundle[0]]
            options = list(boundary_bundle[1])
            steps = [
                CandidateBranchStep("owned_component", component_ref),
                CandidateBranchStep(
                    "landing_demand", boundary_bundle[0].demand_id
                ),
                CandidateBranchStep("track_use", uses[0].use_id),
            ]
            complete = True
            for index, sites in enumerate(transition_sites):
                right_use = (
                    trunk_use if index + 1 == len(nontrunk_lanes) else uses[index + 1]
                )
                bundle = _internal_landing_bundle(
                    net=net,
                    terminal_provenance=terminal.provenance,
                    left_lane=path_lanes[index],
                    left_use=uses[index],
                    right_lane=path_lanes[index + 1],
                    right_use=right_use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=required_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    lazy_landing_access=lazy_landing_access,
                    selected_site_ids=tuple(site.site_id for site in sites),
                )
                if bundle is None:
                    complete = False
                    break
                demands.append(bundle[0])
                options.extend(bundle[1])
                steps.append(CandidateBranchStep(
                    "landing_demand", bundle[0].demand_id
                ))
                if index + 1 < len(nontrunk_lanes):
                    steps.append(CandidateBranchStep(
                        "track_use", uses[index + 1].use_id
                    ))
            if not complete:
                continue
            return _SignalTrunkEndpointAccess(
                tuple(uses),
                tuple(demands),
                tuple(options),
                tuple(steps),
                (component_ref,),
            )
    return None


def _shared_endpoint_riser_candidates(
    local_lane: LaneRef,
    trunk_lane: LaneRef,
    *,
    allowed_layers: frozenset[int],
    lane_layer_index: _BudgetedLaneLayerIndex,
    budget: _IncidenceBudget,
) -> tuple[LaneRef, ...]:
    """Return only adjacent risers intersecting both exact same-layer lanes."""

    if (
        local_lane.layer_n != trunk_lane.layer_n
        or local_lane.preferred_axis != trunk_lane.preferred_axis
    ):
        return ()
    result: dict[str, LaneRef] = {}
    for layer_n in sorted(allowed_layers):
        if abs(layer_n - trunk_lane.layer_n) != 1:
            continue
        local_by_id = {
            lane.lane_id: lane
            for lane in lane_layer_index.intersecting(
                local_lane, layer_n, budget=budget
            )
            if lane.preferred_axis != trunk_lane.preferred_axis
        }
        trunk_ids = {
            lane.lane_id
            for lane in lane_layer_index.intersecting(
                trunk_lane, layer_n, budget=budget
            )
            if lane.preferred_axis != trunk_lane.preferred_axis
        }
        for lane_id in local_by_id.keys() & trunk_ids:
            result[lane_id] = local_by_id[lane_id]
    return tuple(result[lane_id] for lane_id in sorted(result))


def _compile_signal_trunk_endpoint_access(
    *,
    row: tuple[
        TerminalDemand,
        EndpointIncidence,
        CandidateEndpointAccess,
        tuple[str, str, str, int, int, str, int],
    ],
    trunk_lane: LaneRef,
    trunk_use: CorridorUse,
    net: str,
    ordinal_base: int,
    allowed_layers: frozenset[int],
    required_multiplicity: int,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
    tech: Any,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None,
    lane_layer_index: _BudgetedLaneLayerIndex,
    component_boundary_lane_cache: dict[
        tuple[Any, ...], tuple[LaneRef, ...]
    ],
    component_contact_lane_cache: dict[
        tuple[str, str], tuple[tuple[LaneRef, int], ...]
    ],
    staircase_edge_cache: dict[
        tuple[str, str, str, int], tuple[Any, ...]
    ] | None = None,
    staircase_neighbor_cache: dict[
        tuple[str, str, int, int],
        dict[str, tuple[LaneRef, tuple[Any, ...]]],
    ] | None = None,
    staircase_boundary_order_cache: dict[
        tuple[tuple[Any, ...], int], tuple[LaneRef, ...]
    ] | None = None,
) -> _SignalTrunkEndpointAccess | None:
    """Compile direct, component-boundary, or M3-riser access to one trunk."""

    terminal, endpoint, _access, _pin = row
    if trunk_lane in incidence_index.lanes_at_point(
        terminal.layer, terminal.xy_nm
    ):
        return _SignalTrunkEndpointAccess((), (), (), (), ())

    components = _endpoint_conductor_components(
        endpoint, conductor_index=conductor_index, net=net
    )
    for component_ref, component in components:
        component_layer_n = layer_n_by_key[component.layer_key]
        if component_layer_n != trunk_lane.layer_n:
            continue
        terminal_run = _lane_run_coordinate(trunk_lane, terminal.xy_nm)
        if _lane_component_contact_run(
            trunk_lane, component, terminal_run, budget
        ) is None:
            continue
        # The selected trunk physically touches this same-net owned conductor.
        # No via is involved; the component is the exact carrier from the
        # off-track pin access to the trunk metal.
        return _SignalTrunkEndpointAccess(
            (),
            (),
            (),
            (CandidateBranchStep("owned_component", component_ref),),
            (component_ref,),
        )
    for component_ref, _component in components:
        direct = _landing_bundle(
            net=net,
            terminal=terminal,
            endpoint=endpoint,
            lane=trunk_lane,
            use=trunk_use,
            ledger=ledger,
            incidence_index=incidence_index,
            required_multiplicity=required_multiplicity,
            budget=budget,
            option_budget=option_budget,
            required_component_ref=component_ref,
            lazy_landing_access=lazy_landing_access,
        )
        if direct is None:
            continue
        demand, options = direct
        return _SignalTrunkEndpointAccess(
            (),
            (demand,),
            options,
            (
                CandidateBranchStep("owned_component", component_ref),
                CandidateBranchStep("landing_demand", demand.demand_id),
            ),
            (component_ref,),
        )

    # If a dense device-M1 island has no legal VIA1 landing, carry the net on
    # an exact same-layer escape lane first and place the cut at a free
    # intersection with the declared trunk.  The component/track contact is a
    # same-layer conductor join; only the lane-to-trunk transition consumes a
    # landing site.  This turns a placement-time 610/560 landing deadlock into
    # a certified resource choice without weakening via provenance.
    for component_ref, component in components:
        component_layer_n = layer_n_by_key[component.layer_key]
        if (
            abs(component_layer_n - trunk_lane.layer_n) != 1
            or component_layer_n not in allowed_layers
        ):
            continue
        contact_cache_key = (terminal.provenance, component_ref)
        contact_lanes = component_contact_lane_cache.get(contact_cache_key)
        if contact_lanes is None:
            discovered = []
            for local_lane in incidence_index.lanes:
                if (
                    local_lane.layer_n != component_layer_n
                    or local_lane.preferred_axis
                    == trunk_lane.preferred_axis
                    or not _lane_may_contact_component_bbox(
                        local_lane, component
                    )
                ):
                    continue
                terminal_run = _lane_run_coordinate(
                    local_lane, terminal.xy_nm
                )
                contact_run = _lane_component_contact_run(
                    local_lane, component, terminal_run, budget
                )
                if contact_run is not None:
                    discovered.append((local_lane, contact_run))
            contact_lanes = tuple(sorted(
                discovered,
                key=lambda item: (
                    abs(
                        item[0].center_nm
                        - (
                            terminal.xy_nm[0]
                            if item[0].preferred_axis == "V"
                            else terminal.xy_nm[1]
                        )
                    ),
                    item[0].lane_id,
                ),
            ))
            component_contact_lane_cache[contact_cache_key] = contact_lanes
        for local_lane, contact_run in contact_lanes:
            budget.consume("shared-signal same-layer escape lane expansion")
            turn_xy_nm = (
                (local_lane.center_nm, trunk_lane.center_nm)
                if trunk_lane.preferred_axis == "H"
                else (trunk_lane.center_nm, local_lane.center_nm)
            )
            if not (
                local_lane.contains_point(turn_xy_nm)
                and trunk_lane.contains_point(turn_xy_nm)
            ):
                continue
            turn_run = _lane_run_coordinate(local_lane, turn_xy_nm)
            local_interval = (
                min(contact_run, turn_run),
                max(contact_run, turn_run) + 1,
            )
            local_use = _compiled_corridor_use(
                lane=local_lane,
                interval_nm=local_interval,
                net=net,
                pattern_kind="SHARED_SIGNAL_SAME_LAYER_ESCAPE",
                ordinal=ordinal_base,
                width_nm=_tech_min_width_nm(tech, local_lane.layer_n),
            )
            if local_use is None:
                continue
            turn_bundle = _internal_landing_bundle(
                net=net,
                terminal_provenance=terminal.provenance,
                left_lane=local_lane,
                left_use=local_use,
                right_lane=trunk_lane,
                right_use=trunk_use,
                ledger=ledger,
                incidence_index=incidence_index,
                required_multiplicity=required_multiplicity,
                budget=budget,
                option_budget=option_budget,
                lazy_landing_access=lazy_landing_access,
            )
            if turn_bundle is None:
                continue
            turn_demand, turn_options = turn_bundle
            return _SignalTrunkEndpointAccess(
                (local_use,),
                (turn_demand,),
                turn_options,
                (
                    CandidateBranchStep("owned_component", component_ref),
                    CandidateBranchStep("track_use", local_use.use_id),
                    CandidateBranchStep(
                        "landing_demand", turn_demand.demand_id
                    ),
                ),
                (component_ref,),
            )

    for component_ref, component in components:
        cache_key = _component_boundary_cache_key(
            terminal, endpoint, component_ref
        )
        local_lanes = component_boundary_lane_cache.get(cache_key)
        if local_lanes is None:
            local_lanes = _component_boundary_lanes(
                terminal=terminal,
                endpoint=endpoint,
                component_ref=component_ref,
                net=net,
                ledger=ledger,
                incidence_index=incidence_index,
                budget=budget,
                lazy_landing_access=lazy_landing_access,
            )
            component_boundary_lane_cache[cache_key] = local_lanes
        for local_lane in local_lanes:
            if (
                local_lane.layer_n != trunk_lane.layer_n
                or local_lane.preferred_axis != trunk_lane.preferred_axis
                or local_lane.lane_id == trunk_lane.lane_id
            ):
                continue
            component_bounds = _component_run_bounds(local_lane, component)
            if component_bounds is None:
                continue
            terminal_run = _lane_run_coordinate(local_lane, terminal.xy_nm)
            for riser_lane in sorted(
                _shared_endpoint_riser_candidates(
                    local_lane,
                    trunk_lane,
                    allowed_layers=allowed_layers,
                    lane_layer_index=lane_layer_index,
                    budget=budget,
                ),
                key=lambda candidate: (
                    abs(candidate.center_nm - terminal_run),
                    candidate.lane_id,
                ),
            ):
                local_turn = (
                    (riser_lane.center_nm, local_lane.center_nm)
                    if local_lane.preferred_axis == "H"
                    else (local_lane.center_nm, riser_lane.center_nm)
                )
                trunk_turn = (
                    (riser_lane.center_nm, trunk_lane.center_nm)
                    if trunk_lane.preferred_axis == "H"
                    else (trunk_lane.center_nm, riser_lane.center_nm)
                )
                if not (
                    local_lane.contains_point(local_turn)
                    and riser_lane.contains_point(local_turn)
                    and trunk_lane.contains_point(trunk_turn)
                    and riser_lane.contains_point(trunk_turn)
                ):
                    continue
                local_turn_run = _lane_run_coordinate(local_lane, local_turn)
                local_interval = (
                    min(component_bounds[0], local_turn_run),
                    min(
                        local_lane.through_interval_nm[1],
                        max(component_bounds[1], local_turn_run) + 1,
                    ),
                )
                riser_runs = (
                    _lane_run_coordinate(riser_lane, local_turn),
                    _lane_run_coordinate(riser_lane, trunk_turn),
                )
                riser_interval = min(riser_runs), max(riser_runs) + 1
                if (
                    local_interval[1] <= local_interval[0]
                    or riser_interval[1] <= riser_interval[0]
                ):
                    continue
                local_use = _compiled_corridor_use(
                    lane=local_lane,
                    interval_nm=local_interval,
                    net=net,
                    pattern_kind="SHARED_SIGNAL_LOCAL_BRANCH",
                    ordinal=ordinal_base,
                    width_nm=_tech_min_width_nm(tech, local_lane.layer_n),
                )
                riser_use = _compiled_corridor_use(
                    lane=riser_lane,
                    interval_nm=riser_interval,
                    net=net,
                    pattern_kind="SHARED_SIGNAL_RISER",
                    ordinal=ordinal_base + 1,
                    width_nm=_tech_min_width_nm(tech, riser_lane.layer_n),
                )
                if local_use is None or riser_use is None:
                    continue
                endpoint_bundle = _landing_bundle(
                    net=net,
                    terminal=terminal,
                    endpoint=endpoint,
                    lane=local_lane,
                    use=local_use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=required_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    required_component_ref=component_ref,
                    lazy_landing_access=lazy_landing_access,
                )
                if endpoint_bundle is None:
                    continue
                local_turn_bundle = _internal_landing_bundle(
                    net=net,
                    terminal_provenance=terminal.provenance,
                    left_lane=local_lane,
                    left_use=local_use,
                    right_lane=riser_lane,
                    right_use=riser_use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=required_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    lazy_landing_access=lazy_landing_access,
                )
                if local_turn_bundle is None:
                    continue
                trunk_turn_bundle = _internal_landing_bundle(
                    net=net,
                    terminal_provenance=terminal.provenance,
                    left_lane=riser_lane,
                    left_use=riser_use,
                    right_lane=trunk_lane,
                    right_use=trunk_use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=required_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    lazy_landing_access=lazy_landing_access,
                )
                if trunk_turn_bundle is None:
                    continue
                endpoint_demand, endpoint_options = endpoint_bundle
                local_demand, local_options = local_turn_bundle
                trunk_demand, trunk_options = trunk_turn_bundle
                return _SignalTrunkEndpointAccess(
                    (local_use, riser_use),
                    (endpoint_demand, local_demand, trunk_demand),
                    endpoint_options + local_options + trunk_options,
                    (
                        CandidateBranchStep("owned_component", component_ref),
                        CandidateBranchStep(
                            "landing_demand", endpoint_demand.demand_id
                        ),
                        CandidateBranchStep("track_use", local_use.use_id),
                        CandidateBranchStep(
                            "landing_demand", local_demand.demand_id
                        ),
                        CandidateBranchStep("track_use", riser_use.use_id),
                        CandidateBranchStep(
                            "landing_demand", trunk_demand.demand_id
                        ),
                    ),
                    (component_ref,),
                )
    staircase = _compile_monotone_signal_staircase(
        row=row,
        trunk_lane=trunk_lane,
        trunk_use=trunk_use,
        net=net,
        ordinal_base=ordinal_base,
        allowed_layers=allowed_layers,
        required_multiplicity=required_multiplicity,
        ledger=ledger,
        conductor_index=conductor_index,
        incidence_index=incidence_index,
        tech=tech,
        budget=budget,
        option_budget=option_budget,
        lazy_landing_access=lazy_landing_access,
        lane_layer_index=lane_layer_index,
        component_boundary_lane_cache=component_boundary_lane_cache,
        staircase_edge_cache=(
            {}
            if staircase_edge_cache is None
            else staircase_edge_cache
        ),
        staircase_neighbor_cache=(
            {}
            if staircase_neighbor_cache is None
            else staircase_neighbor_cache
        ),
        staircase_boundary_order_cache=(
            {}
            if staircase_boundary_order_cache is None
            else staircase_boundary_order_cache
        ),
    )
    if staircase is not None:
        return staircase
    return None


def _compile_shared_signal_component_star_for_lane(
    *,
    topology_candidate: TopologyCandidate,
    demand: RoutingDemand,
    endpoint_rows: Sequence[
        tuple[
            TerminalDemand,
            EndpointIncidence,
            CandidateEndpointAccess,
            tuple[str, str, str, int, int, str, int],
        ]
    ],
    goal_targets: tuple[str, ...],
    lane: LaneRef,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
    tech: Any,
    topology_digest: str,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None,
    lane_layer_index: _BudgetedLaneLayerIndex,
    component_boundary_lane_cache: dict[
        tuple[Any, ...], tuple[LaneRef, ...]
    ],
    component_contact_lane_cache: dict[
        tuple[str, str], tuple[tuple[LaneRef, int], ...]
    ],
    staircase_edge_cache: dict[
        tuple[str, str, str, int], tuple[Any, ...]
    ],
    staircase_neighbor_cache: dict[
        tuple[str, str, int, int],
        dict[str, tuple[LaneRef, tuple[Any, ...]]],
    ],
    staircase_boundary_order_cache: dict[
        tuple[tuple[Any, ...], int], tuple[LaneRef, ...]
    ],
) -> CompleteNetCandidate | None:
    """Compile one multi-endpoint star without fabricating a direct pin hub."""

    root_rows = tuple(
        row for row in endpoint_rows if row[0].provenance not in goal_targets
    )
    if not root_rows:
        return None
    root = min(root_rows, key=lambda row: row[0].provenance)
    leaves = tuple(sorted(
        (row for row in endpoint_rows if row is not root),
        key=lambda row: row[0].provenance,
    ))
    trunk_use = _compiled_corridor_use(
        lane=lane,
        interval_nm=lane.through_interval_nm,
        net=demand.net,
        pattern_kind="SHARED_SIGNAL_TRUNK",
        ordinal=0,
        width_nm=max(
            topology_candidate.width_nm,
            _tech_min_width_nm(tech, lane.layer_n),
        ),
    )
    if trunk_use is None:
        return None
    allowed_layers = frozenset(
        topology_candidate.layers + topology_candidate.transport_layers
    )
    rows = (root, *leaves)
    fragments = []
    for ordinal, row in enumerate(rows):
        fragment = _compile_signal_trunk_endpoint_access(
            row=row,
            trunk_lane=lane,
            trunk_use=trunk_use,
            net=demand.net,
            ordinal_base=2 * ordinal + 1,
            allowed_layers=allowed_layers,
            required_multiplicity=topology_candidate.via_multiplicity,
            ledger=ledger,
            conductor_index=conductor_index,
            incidence_index=incidence_index,
            layer_n_by_key=layer_n_by_key,
            tech=tech,
            budget=budget,
            option_budget=option_budget,
            lazy_landing_access=lazy_landing_access,
            lane_layer_index=lane_layer_index,
            component_boundary_lane_cache=component_boundary_lane_cache,
            component_contact_lane_cache=component_contact_lane_cache,
            staircase_edge_cache=staircase_edge_cache,
            staircase_neighbor_cache=staircase_neighbor_cache,
            staircase_boundary_order_cache=staircase_boundary_order_cache,
        )
        if fragment is None:
            return None
        fragments.append(fragment)
    fragment_by_provenance = {
        row[0].provenance: fragment for row, fragment in zip(rows, fragments)
    }
    root_fragment = fragment_by_provenance[root[0].provenance]
    branches = []
    for leaf in leaves:
        leaf_fragment = fragment_by_provenance[leaf[0].provenance]
        steps = (
            root_fragment.steps_to_trunk
            + (CandidateBranchStep("track_use", trunk_use.use_id),)
            + tuple(reversed(leaf_fragment.steps_to_trunk))
        )
        branch_id = canonical_pattern_id(
            "pattern-branch",
            {
                "hub": root[0].provenance,
                "leaf": leaf[0].provenance,
                "net": demand.net,
                "pattern_kind": "SHARED_SIGNAL_COMPONENT_STAR",
                "steps": [
                    {"kind": step.kind, "ref_id": step.ref_id}
                    for step in steps
                ],
            },
        )
        branches.append(CandidateBranch(
            branch_id=branch_id,
            source_endpoint=root[0].provenance,
            target_endpoint=leaf[0].provenance,
            ordered_use_ids=tuple(
                step.ref_id for step in steps if step.kind == "track_use"
            ),
            ordered_landing_demand_ids=tuple(
                step.ref_id
                for step in steps
                if step.kind == "landing_demand"
            ),
            ordered_steps=steps,
        ))
    branches_by_target = {
        branch.target_endpoint: branch for branch in branches
    }
    if any(target not in branches_by_target for target in goal_targets):
        return None
    uses = {
        use.use_id: use
        for fragment in fragments
        for use in fragment.corridor_uses
    }
    uses[trunk_use.use_id] = trunk_use
    demands = {
        landing.demand_id: landing
        for fragment in fragments
        for landing in fragment.landing_demands
    }
    options = {
        option.option_id: option
        for fragment in fragments
        for option in fragment.landing_options
    }
    owned_refs = tuple(sorted({
        component_ref
        for fragment in fragments
        for component_ref in fragment.owned_component_refs
    }))
    candidate = CompleteNetCandidate(
        candidate_id="pending-content-address",
        net=demand.net,
        topology_class=topology_candidate.topology_class,
        branch_edges=tuple(
            (branch.source_endpoint, branch.target_endpoint)
            for branch in branches
        ),
        corridor_uses=tuple(uses[key] for key in sorted(uses)),
        landing_demands=tuple(demands[key] for key in sorted(demands)),
        required_goal_refs=topology_candidate.required_goal_refs,
        width_nm=topology_candidate.width_nm,
        via_multiplicity=topology_candidate.via_multiplicity,
        source_topology_digest=topology_digest,
        program_kind="PATTERN_TREE",
        branches=tuple(branches),
        landing_options=tuple(options[key] for key in sorted(options)),
        goal_branch_coverage=tuple(
            GoalBranchCoverage(
                declaration_ref=goal.declaration_ref,
                branch_id=branches_by_target[goal.target.provenance].branch_id,
                target_provenance=goal.target.provenance,
            )
            for goal in demand.required_goals
            if goal.target is not None
        ),
        endpoint_accesses=tuple(
            row[2]
            for row in sorted(endpoint_rows, key=lambda row: row[0].provenance)
        ),
        owned_component_use_refs=owned_refs,
    )
    return _content_address_candidate(candidate)


def _compile_shared_signal_star_candidate(
    *,
    topology_candidate: TopologyCandidate,
    demand: RoutingDemand,
    routing_world: RoutingWorldProvenance,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
    tech: Any,
    topology_digest: str,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    lane_layer_index: _BudgetedLaneLayerIndex,
    lazy_landing_access: _ExactLazyLandingAccess | None = None,
    _planned_trunk_layers: tuple[int, ...] | None = None,
    _family_budget_state: _SharedSignalFamilyBudgetState | None = None,
) -> tuple[CompleteNetCandidate, ...]:
    if topology_candidate.topology_class != "shared_signal_trunk":
        return ()
    endpoint_rows = _resolved_endpoint_rows_any(
        topology_candidate=topology_candidate,
        demand=demand,
        routing_world=routing_world,
        incidence_index=incidence_index,
        layer_n_by_key=layer_n_by_key,
    )
    if not endpoint_rows:
        return ()
    owns_family_budget = _family_budget_state is None
    if _family_budget_state is None:
        _family_budget_state = _SharedSignalFamilyBudgetState(
            raw_option_budget=_SharedSignalRawOptionBudget(
                budget, option_budget
            )
        )
    raw_option_budget = _family_budget_state.raw_option_budget
    goal_targets = tuple(
        goal.target.provenance
        for goal in demand.required_goals
        if goal.target is not None
    )
    if (
        len(goal_targets) != len(demand.required_goals)
        or len(set(goal_targets)) != len(goal_targets)
    ):
        return ()
    allowed_layers = frozenset(
        topology_candidate.layers + topology_candidate.transport_layers
    )
    # A declared trunk layer is the semantic layer-planning decision; transport
    # closure authorizes only the risers/staircases needed to reach it.  Treating
    # every transport layer as another trunk candidate multiplied each declared
    # M2 bias rail across the whole six-metal grid (1.2M exact expansions for one
    # net) and silently weakened the declaration.  Undeclared multi-pin nets use
    # the deterministic capacity/transition layer plan below.
    if demand.declared_trunk_layers:
        trunk_layers = frozenset(demand.declared_trunk_layers)
    elif _planned_trunk_layers is not None:
        trunk_layers = frozenset(_planned_trunk_layers)
    else:
        terminal_layer_ns = tuple(
            layer_n_by_key[row[0].layer] for row in endpoint_rows
        )
        layer_rows: dict[int, tuple[LaneRef, ...]] = {
            layer_n: tuple(
                lane
                for lane in incidence_index.lanes
                if lane.layer_n == layer_n
                and ":terminal-access:" not in lane.resource_corridor_id
            )
            for layer_n in sorted(allowed_layers)
        }
        endpoint_run_bounds = {
            "H": (
                min(row[0].xy_nm[0] for row in endpoint_rows),
                max(row[0].xy_nm[0] for row in endpoint_rows),
            ),
            "V": (
                min(row[0].xy_nm[1] for row in endpoint_rows),
                max(row[0].xy_nm[1] for row in endpoint_rows),
            ),
        }

        def layer_plan_key(layer_n: int) -> tuple[int, int, int, int, int]:
            lanes = layer_rows[layer_n]
            endpoint_count = sum(
                terminal_layer_n == layer_n
                for terminal_layer_n in terminal_layer_ns
            )
            transition_depth = sum(
                abs(terminal_layer_n - layer_n)
                for terminal_layer_n in terminal_layer_ns
            )
            span_capacity = 0
            for lane in lanes:
                run_lo, run_hi = endpoint_run_bounds[lane.preferred_axis]
                if (
                    lane.through_interval_nm[0] <= run_lo
                    and run_hi < lane.through_interval_nm[1]
                ):
                    span_capacity += 1
            # This is a deterministic search order, not a feasibility proxy or
            # style flag.  Whole-span capacity is the first useful layer-plan
            # fact, then transition depth and capacity break ties.  Exact
            # endpoint/staircase compilation below remains the certificate;
            # when one planned layer has no complete program, the next signed
            # layer is tried instead of turning a proxy miss into infeasibility.
            return (
                0 if span_capacity else 1,
                transition_depth,
                -span_capacity,
                -endpoint_count,
                -layer_n,
            )

        # The recursive call owns motif tiering and family merging for exactly
        # one layer.  Retain every exact-feasible layer for simultaneous CP;
        # ``budget``/``option_budget`` remain shared across the whole net, and
        # the caller rejects an over-limit merged family atomically.
        planned_candidates: list[CompleteNetCandidate] = []
        for planned_layer in sorted(allowed_layers, key=layer_plan_key):
            planned = _compile_shared_signal_star_candidate(
                topology_candidate=topology_candidate,
                demand=demand,
                routing_world=routing_world,
                ledger=ledger,
                conductor_index=conductor_index,
                incidence_index=incidence_index,
                layer_n_by_key=layer_n_by_key,
                tech=tech,
                topology_digest=topology_digest,
                budget=budget,
                option_budget=option_budget,
                lane_layer_index=lane_layer_index,
                lazy_landing_access=lazy_landing_access,
                _planned_trunk_layers=(planned_layer,),
                _family_budget_state=_family_budget_state,
            )
            planned_candidates.extend(planned)
        ordered_planned = tuple(
            {
                candidate.candidate_id: candidate
                for candidate in planned_candidates
            }[candidate_id]
            for candidate_id in sorted({
                candidate.candidate_id: candidate
                for candidate in planned_candidates
            })
        )
        if owns_family_budget:
            return _debit_final_shared_signal_candidates(
                ordered_planned,
                option_budget=option_budget,
                family_state=_family_budget_state,
            )
        return ordered_planned
    candidates = _SharedSignalCandidateFamilyAccumulator()
    component_boundary_lane_cache: dict[
        tuple[Any, ...], tuple[LaneRef, ...]
    ] = {}
    component_contact_lane_cache: dict[
        tuple[str, str], tuple[tuple[LaneRef, int], ...]
    ] = {}
    # The closed allocation IR requires every endpoint to be a certified world
    # access.  Enumerate only exact ledger lanes, and choose a deterministic
    # existing non-goal endpoint on each lane as its structural hub.  No
    # synthetic/fabricated world pin is introduced.
    for lane in incidence_index.lanes:
        if (
            lane.layer_n not in trunk_layers
            or ":terminal-access:" in lane.resource_corridor_id
        ):
            continue
        budget.consume("shared-signal star lane expansion")
        direct_rows = tuple(
            row
            for row in endpoint_rows
            if lane in incidence_index.lanes_at_point(
                row[0].layer, row[0].xy_nm
            )
        )
        root_rows = tuple(sorted(
            (
                row
                for row in direct_rows
                if row[0].provenance not in goal_targets
            ),
            key=lambda row: row[0].provenance,
        ))
        if not root_rows:
            component_candidate = _compile_shared_signal_component_star_for_lane(
                topology_candidate=topology_candidate,
                demand=demand,
                endpoint_rows=endpoint_rows,
                goal_targets=goal_targets,
                lane=lane,
                ledger=ledger,
                conductor_index=conductor_index,
                incidence_index=incidence_index,
                layer_n_by_key=layer_n_by_key,
                tech=tech,
                topology_digest=topology_digest,
                budget=budget,
                option_budget=raw_option_budget,
                lazy_landing_access=lazy_landing_access,
                lane_layer_index=lane_layer_index,
                component_boundary_lane_cache=component_boundary_lane_cache,
                component_contact_lane_cache=component_contact_lane_cache,
                staircase_edge_cache=(
                    _family_budget_state.staircase_edge_cache
                ),
                staircase_neighbor_cache=(
                    _family_budget_state.staircase_neighbor_cache
                ),
                staircase_boundary_order_cache=(
                    _family_budget_state.staircase_boundary_order_cache
                ),
            )
            if component_candidate is not None:
                candidates.add(component_candidate)
            continue
        root = root_rows[0]
        leaves = tuple(sorted(
            (row for row in endpoint_rows if row is not root),
            key=lambda row: row[0].provenance,
        ))
        coordinates = tuple(
            _lane_run_coordinate(lane, row[0].xy_nm) for row in endpoint_rows
        )
        interval_nm = min(coordinates), max(coordinates) + 1
        if interval_nm[1] <= interval_nm[0]:
            continue
        use = _compiled_corridor_use(
            lane=lane,
            interval_nm=interval_nm,
            net=demand.net,
            pattern_kind="SHARED_SIGNAL_STAR",
            ordinal=0,
            width_nm=max(
                topology_candidate.width_nm,
                _tech_min_width_nm(tech, lane.layer_n),
            ),
        )
        if use is None:
            continue
        branches = []
        landing_demands = []
        landing_options = []
        infeasible_leaf = False
        for leaf in leaves:
            budget.consume("shared-signal star leaf expansion")
            leaf_direct = lane in incidence_index.lanes_at_point(
                leaf[0].layer, leaf[0].xy_nm
            )
            landing_bundle = None
            if not leaf_direct:
                landing_bundle = _landing_bundle(
                    net=demand.net,
                    terminal=leaf[0],
                    endpoint=leaf[1],
                    lane=lane,
                    use=use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=topology_candidate.via_multiplicity,
                    budget=budget,
                    option_budget=raw_option_budget,
                    lazy_landing_access=lazy_landing_access,
                )
                if landing_bundle is None:
                    infeasible_leaf = True
                    break
                landing_demand, options = landing_bundle
                landing_demands.append(landing_demand)
                landing_options.extend(options)
            branch_id = canonical_pattern_id(
                "pattern-branch",
                {
                    "hub": root[0].provenance,
                    "leaf": leaf[0].provenance,
                    "net": demand.net,
                    "pattern_kind": "SHARED_SIGNAL_STAR",
                    "use_id": use.use_id,
                },
            )
            branches.append(
                CandidateBranch(
                    branch_id=branch_id,
                    source_endpoint=root[0].provenance,
                    target_endpoint=leaf[0].provenance,
                    ordered_use_ids=(use.use_id,),
                    ordered_landing_demand_ids=(
                        ()
                        if landing_bundle is None
                        else (landing_bundle[0].demand_id,)
                    ),
                    ordered_steps=(
                        (CandidateBranchStep("track_use", use.use_id),)
                        if landing_bundle is None
                        else (
                            CandidateBranchStep("track_use", use.use_id),
                            CandidateBranchStep(
                                "landing_demand", landing_bundle[0].demand_id
                            ),
                        )
                    ),
                )
            )
        if infeasible_leaf:
            component_candidate = _compile_shared_signal_component_star_for_lane(
                topology_candidate=topology_candidate,
                demand=demand,
                endpoint_rows=endpoint_rows,
                goal_targets=goal_targets,
                lane=lane,
                ledger=ledger,
                conductor_index=conductor_index,
                incidence_index=incidence_index,
                layer_n_by_key=layer_n_by_key,
                tech=tech,
                topology_digest=topology_digest,
                budget=budget,
                option_budget=raw_option_budget,
                lazy_landing_access=lazy_landing_access,
                lane_layer_index=lane_layer_index,
                component_boundary_lane_cache=component_boundary_lane_cache,
                component_contact_lane_cache=component_contact_lane_cache,
                staircase_edge_cache=(
                    _family_budget_state.staircase_edge_cache
                ),
                staircase_neighbor_cache=(
                    _family_budget_state.staircase_neighbor_cache
                ),
                staircase_boundary_order_cache=(
                    _family_budget_state.staircase_boundary_order_cache
                ),
            )
            if component_candidate is not None:
                candidates.add(component_candidate)
            continue
        branches_by_target = {
            branch.target_endpoint: branch for branch in branches
        }
        if any(target not in branches_by_target for target in goal_targets):
            continue
        candidate = CompleteNetCandidate(
            candidate_id="pending-content-address",
            net=demand.net,
            topology_class=topology_candidate.topology_class,
            branch_edges=tuple(
                (branch.source_endpoint, branch.target_endpoint)
                for branch in branches
            ),
            corridor_uses=(use,),
            landing_demands=tuple(landing_demands),
            required_goal_refs=topology_candidate.required_goal_refs,
            width_nm=topology_candidate.width_nm,
            via_multiplicity=0,
            source_topology_digest=topology_digest,
            branches=tuple(branches),
            landing_options=tuple(landing_options),
            goal_branch_coverage=tuple(
                GoalBranchCoverage(
                    declaration_ref=goal.declaration_ref,
                    branch_id=branches_by_target[goal.target.provenance].branch_id,
                    target_provenance=goal.target.provenance,
                )
                for goal in demand.required_goals
                if goal.target is not None
            ),
            endpoint_accesses=tuple(
                row[2]
                for row in sorted(endpoint_rows, key=lambda row: row[0].provenance)
            ),
        )
        candidates.add(_content_address_candidate(candidate))
    # Every bounded complete shared-star program is an allocation alternative.
    # A lower-complexity transport can conflict globally while a staircase is
    # feasible, so no motif tier may suppress another without an exact
    # dominance certificate.
    _family_budget_state.family_count += candidates.family_count
    merged = _merge_shared_signal_candidate_families(
        candidates.candidates(),
        ledger=ledger,
        lazy_landing_access=lazy_landing_access,
    )
    merged = tuple(
        _factor_final_candidate_landing_domains(
            candidate,
            ledger=ledger,
            lazy_landing_access=lazy_landing_access,
        )
        for candidate in merged
    )
    if owns_family_budget:
        return _debit_final_shared_signal_candidates(
            merged,
            option_budget=option_budget,
            family_state=_family_budget_state,
        )
    return merged


def _shared_signal_pattern_tier(
    candidate: CompleteNetCandidate,
) -> tuple[int, int, int]:
    """Closed structural complexity tier for one complete shared-star program."""

    branch_use_counts = tuple(
        len(branch.ordered_use_ids) for branch in candidate.branches
    )
    return (
        max(branch_use_counts, default=0),
        len(candidate.corridor_uses),
        len(candidate.landing_demands),
    )


def _debit_final_shared_signal_candidates(
    candidates: tuple[CompleteNetCandidate, ...],
    *,
    option_budget: _IncidenceBudget,
    family_state: _SharedSignalFamilyBudgetState,
    bundle_kind: str = "shared_signal_final_families",
) -> tuple[CompleteNetCandidate, ...]:
    """Atomically debit only content-addressed closed family options."""

    unique = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    ordered = tuple(unique[key] for key in sorted(unique))
    new_candidates = tuple(
        candidate
        for candidate in ordered
        if candidate.candidate_id
        not in option_budget.shared_final_candidate_ids
    )
    final_aliases = sum(
        len(candidate.landing_options) for candidate in new_candidates
    )
    option_counts = tuple(
        len(candidate.landing_options) for candidate in new_candidates
    )
    semantic_options: set[str] = set()
    for candidate in new_candidates:
        for option in candidate.landing_options:
            semantic_options.add(
                landing_option_template_record(candidate, option)[0]
            )
    new_template_keys = (
        semantic_options - option_budget.shared_final_option_template_keys
    )
    final_unique_templates = len(new_template_keys)
    option_budget.shared_final_unique_templates += final_unique_templates
    option_budget.shared_family_count += family_state.family_count
    option_budget.record_option_bundle(
        kind=bundle_kind,
        site_ids=tuple(
            site_id
            for candidate in new_candidates
            for option in candidate.landing_options
            for site_id in option.site_ids
        ),
        required_multiplicity=sum(
            demand.required_multiplicity
            for candidate in new_candidates
            for demand in candidate.landing_demands
        ),
        option_count=final_unique_templates,
        context={
            "candidate_count": len(new_candidates),
            "final_aliases": final_aliases,
            "family_count": family_state.family_count,
            "final_unique_templates": final_unique_templates,
            "max_options": max(option_counts, default=0),
            "min_options": min(option_counts, default=0),
            "raw_expansions": family_state.raw_option_budget.used,
            "dedup_options": len(semantic_options),
        },
    )
    option_budget.used += final_unique_templates
    if option_budget.used > option_budget.limit:
        raise _IncidenceBudgetExceeded(
            "exact-family final landing-option expansion"
        )
    option_budget.shared_final_candidate_ids.update(
        candidate.candidate_id for candidate in new_candidates
    )
    option_budget.shared_final_option_template_keys.update(new_template_keys)
    return ordered


def _is_structural_rail_access(
    row: tuple[
        TerminalDemand,
        EndpointIncidence,
        CandidateEndpointAccess,
        tuple[Any, ...],
    ],
    component: Any,
) -> bool:
    terminal, endpoint, _access, pin = row
    return bool(
        type(pin[1]) is str
        and type(pin[2]) is str
        and pin[2] == "rail"
        and terminal.provenance == _pin_fact_provenance(pin)
        and component.component_ref in endpoint.conductor_component_refs
        and pin in component.pin_accesses
        and "rail" in component.kinds
    )


def _component_has_exact_rail_width_atom(
    component: Any,
    *,
    root_xy_nm: PointNm,
    net: str,
    width_nm: int,
) -> bool:
    """Prove rail width from one exact atom containing the selected root."""

    x, y = root_xy_nm
    return any(
        type(atom) is tuple
        and len(atom) == 8
        and atom[6] == net
        and atom[7] == "rail"
        and atom[0] <= x <= atom[2]
        and atom[1] <= y <= atom[3]
        and min(atom[2] - atom[0], atom[3] - atom[1]) >= width_nm
        for atom in component.atom_facts
    )


def _tech_min_width_nm(tech: Any, layer_n: int) -> int:
    accessor = getattr(tech, "min_width", None)
    if not callable(accessor):
        raise PatternCompileInputError("PDK exposes no min_width accessor")
    for alias in (f"metal{layer_n}", f"met{layer_n}", f"m{layer_n}"):
        try:
            value = accessor(alias)
        except Exception:
            continue
        if type(value) is int and value > 0:
            return value
        if value is not None:
            raise PatternCompileInputError(
                f"PDK {alias} min_width must be a positive exact integer nm"
            )
    raise PatternCompileInputError(
        f"PDK exposes no exact metal{layer_n} minimum width"
    )


def _lane_component_contact_run(
    lane: LaneRef,
    component: Any,
    terminal_run_nm: int,
    budget: _IncidenceBudget,
) -> int | None:
    if lane.gds_layer_key != component.layer_key:
        return None
    # The expansion budget is created once for one fully rebound compiler
    # invocation.  It is therefore the narrowest shared scope that lets power,
    # signal, and two-pin topology variants reuse the same completed physical
    # lane/component incidence proof without crossing input worlds.
    contact_interval_cache = budget.shared_lane_component_contact_intervals
    cache_key = lane.lane_id, component.component_ref
    intervals = contact_interval_cache.get(cache_key)
    if intervals is None:
        through_lo, through_hi = lane.through_interval_nm
        discovered = []
        for atom in component.atom_facts:
            budget.consume("exact lane/component atom incidence")
            if lane.preferred_axis == "H":
                if not atom[1] <= lane.center_nm <= atom[3]:
                    continue
                atom_lo, atom_hi = atom[0], atom[2]
            else:
                if not atom[0] <= lane.center_nm <= atom[2]:
                    continue
                atom_lo, atom_hi = atom[1], atom[3]
            lo = max(atom_lo, through_lo)
            # Lane intervals are half-open.  Geometry bboxes use boundary
            # coordinates, so the last legal integer run coordinate is hi-1.
            hi = min(atom_hi, through_hi - 1)
            if lo <= hi:
                discovered.append((lo, hi))
        intervals = tuple(discovered)
        # Publish only after the complete atom domain has been scanned.
        contact_interval_cache[cache_key] = intervals
    if not intervals:
        return None
    contacts = tuple(
        min(max(terminal_run_nm, lo), hi) for lo, hi in intervals
    )
    return min(contacts, key=lambda value: (abs(value - terminal_run_nm), value))


def _lane_may_contact_component_bbox(lane: LaneRef, component: Any) -> bool:
    """Cheap exact-negative index before atom-level component incidence."""

    if lane.gds_layer_key != component.layer_key:
        return False
    x0, y0, x1, y1 = component.bbox_nm
    through_lo, through_hi = lane.through_interval_nm
    if lane.preferred_axis == "H":
        return (
            y0 <= lane.center_nm <= y1
            and through_lo <= x1
            and x0 < through_hi
        )
    return (
        x0 <= lane.center_nm <= x1
        and through_lo <= y1
        and y0 < through_hi
    )


def _endpoint_conductor_components(
    endpoint: EndpointIncidence,
    *,
    conductor_index: OwnedConductorComponentIndex,
    net: str,
) -> tuple[tuple[str, Any], ...]:
    """Return the exact traversable components that can carry one endpoint.

    A pin point is an access fact, not a requirement that every global track pass
    through that exact coordinate.  Device M1 (or another owned conductor) may
    carry the terminal to a legal track/via site elsewhere on the same component.
    """

    result = []
    for component_ref in endpoint.conductor_component_refs:
        component = conductor_index.component(component_ref)
        if component.net != net or not component.atom_facts:
            continue
        result.append((component_ref, component))
    return tuple(result)


def _component_run_bounds(
    lane: LaneRef,
    component: Any,
) -> tuple[int, int] | None:
    if not component.atom_facts:
        return None
    if lane.preferred_axis == "H":
        bounds = (min(atom[0] for atom in component.atom_facts),
                  max(atom[2] for atom in component.atom_facts))
    else:
        bounds = (min(atom[1] for atom in component.atom_facts),
                  max(atom[3] for atom in component.atom_facts))
    lo = max(bounds[0], lane.through_interval_nm[0])
    hi = min(bounds[1], lane.through_interval_nm[1] - 1)
    return None if hi < lo else (lo, hi)


class _CertifiedComponentPairLandingAccess:
    """Sparse exact lookup for vias joining two certified owned components.

    Component-to-component vias are mandatory even when the free landing lattice
    is lazy.  The lane-incidence builder has already certified both landing-pad
    sides against the owned-conductor index, so compile those sparse relations
    once per power net instead of rescanning every landing site for every leaf.
    """

    def __init__(
        self,
        ledger: ExplicitRoutingResourceLedger,
        incidence_index: LaneIncidenceIndex,
    ) -> None:
        self._sites_by_id = {site.site_id: site for site in ledger.landing_sites}
        self._sides = {
            (side.site_id, side.side): side for side in incidence_index.site_sides
        }
        self._by_net: dict[
            str,
            dict[tuple[int, int, str, str], tuple[Any, ...]],
        ] = {}

    def _build_net(
        self,
        net: str,
        budget: _IncidenceBudget,
    ) -> dict[tuple[int, int, str, str], tuple[Any, ...]]:
        mutable: dict[tuple[int, int, str, str], list[Any]] = {}
        for site in sorted(self._sites_by_id.values(), key=lambda item: item.site_id):
            if site.owner_net != net:
                continue
            try:
                lower = self._sides[(site.site_id, "lower")]
                upper = self._sides[(site.site_id, "upper")]
            except KeyError as exc:
                raise PatternCompileInputError(
                    f"landing site {site.site_id!r} lacks certified side incidence"
                ) from exc
            for lower_ref in lower.owner_component_refs:
                for upper_ref in upper.owner_component_refs:
                    budget.consume("component-pair landing relation expansion")
                    mutable.setdefault(
                        (
                            site.lower_layer_n,
                            site.upper_layer_n,
                            lower_ref,
                            upper_ref,
                        ),
                        [],
                    ).append(site)
        result = {
            key: tuple(sorted(values, key=lambda item: item.site_id))
            for key, values in mutable.items()
        }
        self._by_net[net] = result
        return result

    def sites(
        self,
        *,
        net: str,
        root_component_ref: str,
        root_layer_n: int,
        leaf_component_ref: str,
        leaf_layer_n: int,
        budget: _IncidenceBudget,
    ) -> tuple[Any, ...]:
        relations = self._by_net.get(net)
        if relations is None:
            relations = self._build_net(net, budget)
        if root_layer_n < leaf_layer_n:
            key = (
                root_layer_n,
                leaf_layer_n,
                root_component_ref,
                leaf_component_ref,
            )
        else:
            key = (
                leaf_layer_n,
                root_layer_n,
                leaf_component_ref,
                root_component_ref,
            )
        sites = relations.get(key, ())
        for _site in sites:
            budget.consume("component-pair landing-site join evaluation")
        return sites


def _component_pair_landing_bundle(
    *,
    net: str,
    terminal: TerminalDemand,
    endpoint: EndpointIncidence,
    root_component_ref: str,
    conductor_index: OwnedConductorComponentIndex,
    layer_n_by_key: dict[LayerKey, int],
    allowed_layers: frozenset[int],
    required_multiplicity: int,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    component_pair_landing_access: _CertifiedComponentPairLandingAccess,
) -> tuple[str, LandingDemand, tuple[LandingOption, ...]] | None:
    """Join an endpoint component directly to an adjacent same-net component."""

    root_component = conductor_index.component(root_component_ref)
    root_layer_n = layer_n_by_key[root_component.layer_key]
    multiplicity = max(1, required_multiplicity)
    sites_by_leaf_ref: dict[str, list[Any]] = {}
    for leaf_component_ref in sorted(endpoint.conductor_component_refs):
        budget.consume("component-pair landing leaf evaluation")
        if leaf_component_ref == root_component_ref:
            continue
        leaf_component = conductor_index.component(leaf_component_ref)
        leaf_layer_n = layer_n_by_key[leaf_component.layer_key]
        if (
            root_component.net != net
            or leaf_component.net != net
            or abs(root_layer_n - leaf_layer_n) != 1
            or root_layer_n not in allowed_layers
            or leaf_layer_n not in allowed_layers
        ):
            continue
        for site in component_pair_landing_access.sites(
            net=net,
            root_component_ref=root_component_ref,
            root_layer_n=root_layer_n,
            leaf_component_ref=leaf_component_ref,
            leaf_layer_n=leaf_layer_n,
            budget=budget,
        ):
            sites_by_leaf_ref.setdefault(leaf_component_ref, []).append(site)

    for leaf_component_ref in sorted(sites_by_leaf_ref):
        sites = tuple(sorted(
            {site.site_id: site for site in sites_by_leaf_ref[leaf_component_ref]}.values(),
            key=lambda site: site.site_id,
        ))
        demand_id = canonical_pattern_id(
            "pattern-component-pair-landing-demand",
            {
                "leaf_component_ref": leaf_component_ref,
                "net": net,
                "required_multiplicity": multiplicity,
                "root_component_ref": root_component_ref,
                "terminal_provenance": terminal.provenance,
            },
        )
        bundle_context = {
            "demand_id": demand_id,
            "leaf_component_ref": leaf_component_ref,
            "net": net,
            "root_component_ref": root_component_ref,
            "terminal_provenance": terminal.provenance,
        }
        if len(sites) < multiplicity:
            option_budget.record_option_bundle(
                kind="component_pair",
                site_ids=tuple(site.site_id for site in sites),
                required_multiplicity=multiplicity,
                option_count=0,
                context=bundle_context,
            )
            continue
        option_site_groups = (
            ((site,) for site in sites)
            if multiplicity > 1
            else combinations(sites, multiplicity)
        )
        option_budget.record_option_bundle(
            kind="component_pair",
            site_ids=tuple(site.site_id for site in sites),
            required_multiplicity=multiplicity,
            option_count=len(sites),
            context=bundle_context,
        )
        options = []
        for selected_sites in option_site_groups:
            site_ids = tuple(site.site_id for site in selected_sites)
            option_id = canonical_pattern_id(
                "pattern-component-pair-landing-option",
                {
                    "demand_id": demand_id,
                    "site_ids": list(site_ids),
                },
            )
            options.append(
                _intern_landing_option(
                    option_budget,
                    reason="component-pair landing-option expansion",
                    option_id=option_id,
                    demand_id=demand_id,
                    site_ids=site_ids,
                    contacts=(),
                )
            )
        return (
            leaf_component_ref,
            LandingDemand(
                demand_id=demand_id,
                terminal_provenance=terminal.provenance,
                allowed_site_ids=tuple(sorted(
                    {
                        site_id
                        for option in options
                        for site_id in option.site_ids
                    }
                )),
                required_multiplicity=multiplicity,
                selection_mode=(
                    LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY
                    if multiplicity > 1
                    else LANDING_SELECTION_EXPLICIT_OPTION
                ),
            ),
            tuple(options),
        )
    return None


def _lifted_power_drop_bundle(
    *,
    net: str,
    root: tuple[Any, ...],
    leaf: tuple[Any, ...],
    root_component: Any,
    root_lanes: Sequence[LaneRef],
    leaf_ordinal: int,
    root_width_nm: int,
    allowed_layers: frozenset[int],
    required_multiplicity: int,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
    tech: Any,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None,
    lane_layer_index: _BudgetedLaneLayerIndex,
    component_boundary_lane_cache: dict[
        tuple[Any, ...], tuple[LaneRef, ...]
    ],
    component_contact_lane_cache: dict[
        tuple[str, str], tuple[tuple[LaneRef, int], ...]
    ],
    shared_boundary_bundles: dict[
        tuple[Any, ...],
        tuple[LandingDemand, tuple[LandingOption, ...]] | None,
    ],
    failure_refs: list[str] | None = None,
) -> tuple[
    str,
    str,
    tuple[CorridorUse, ...],
    tuple[LandingDemand, ...],
    tuple[LandingOption, ...],
    tuple[CandidateBranchStep, ...],
] | None:
    """Compile one rail-via-riser-turn-branch-via power leaf."""

    root_layer_n = layer_n_by_key[root_component.layer_key]
    failure_stage = "root_boundary_lane_absent"
    failure_stage_rank = 0
    root_candidate_count = len(root_lanes)
    contacting_root_lane_count = 0

    def reached(stage: str, rank: int) -> None:
        nonlocal failure_stage, failure_stage_rank
        if rank > failure_stage_rank:
            failure_stage = stage
            failure_stage_rank = rank

    leaf_lane_cache: list[tuple[str, Any, LaneRef]] = []
    leaf_boundary_failure_refs: list[str] = []
    leaf_boundary_failure_stats: dict[str, int] = {}
    leaf_discovery_complete = False

    def leaf_lane_candidates() -> Iterator[tuple[str, Any, LaneRef]]:
        nonlocal leaf_discovery_complete
        yield from leaf_lane_cache
        if leaf_discovery_complete:
            return
        leaf_components = _endpoint_conductor_components(
            leaf[1], conductor_index=conductor_index, net=net
        )
        for leaf_component_ref, leaf_component in leaf_components:
            leaf_layer_n = layer_n_by_key[leaf_component.layer_key]
            if leaf_layer_n + 1 not in allowed_layers:
                continue
            leaf_lanes = _component_boundary_lanes(
                terminal=leaf[0],
                endpoint=leaf[1],
                component_ref=leaf_component_ref,
                net=net,
                ledger=ledger,
                incidence_index=incidence_index,
                budget=budget,
                lazy_landing_access=lazy_landing_access,
                failure_refs=leaf_boundary_failure_refs,
                failure_stats=leaf_boundary_failure_stats,
            )
            if not leaf_lanes:
                leaf_lanes = tuple(
                    lane
                    for lane in lane_layer_index.lanes(
                        leaf_layer_n + 1, budget=budget
                    )
                    if lane.layer_n in allowed_layers
                )
            for leaf_lane in leaf_lanes:
                budget.consume("power lifted-drop leaf-lane expansion")
                if leaf_lane.layer_n != leaf_layer_n + 1:
                    continue
                item = leaf_component_ref, leaf_component, leaf_lane
                leaf_lane_cache.append(item)
                yield item
        leaf_discovery_complete = True

    for root_lane in root_lanes:
        budget.consume("power lifted-drop root-lane expansion")
        if (
            root_lane.layer_n not in allowed_layers
            or root_lane.layer_n != root_layer_n + 1
        ):
            continue
        root_bounds = _component_run_bounds(root_lane, root_component)
        if root_bounds is None:
            continue
        contacting_root_lane_count += 1
        reached("leaf_boundary_lane_absent", 1)
        for (
            leaf_component_ref,
            leaf_component,
            leaf_lane,
        ) in leaf_lane_candidates():
            reached("turn_geometry_absent", 2)
            layer_gap = abs(root_lane.layer_n - leaf_lane.layer_n)
            if (
                layer_gap == 2
                and root_lane.preferred_axis == leaf_lane.preferred_axis
            ):
                bridge_layer_n = (
                    root_lane.layer_n + leaf_lane.layer_n
                ) // 2
                if bridge_layer_n not in allowed_layers:
                    continue
                leaf_bounds = _component_run_bounds(leaf_lane, leaf_component)
                if leaf_bounds is None:
                    continue
                for bridge_lane in lane_layer_index.intersecting(
                    root_lane,
                    bridge_layer_n,
                    budget=budget,
                ):
                    budget.consume("power lifted-drop bridge-lane expansion")
                    if (
                        bridge_lane.preferred_axis
                        == root_lane.preferred_axis
                    ):
                        continue
                    root_turn_xy_nm = (
                        (bridge_lane.center_nm, root_lane.center_nm)
                        if root_lane.preferred_axis == "H"
                        else (root_lane.center_nm, bridge_lane.center_nm)
                    )
                    leaf_turn_xy_nm = (
                        (bridge_lane.center_nm, leaf_lane.center_nm)
                        if leaf_lane.preferred_axis == "H"
                        else (leaf_lane.center_nm, bridge_lane.center_nm)
                    )
                    if not (
                        root_lane.contains_point(root_turn_xy_nm)
                        and bridge_lane.contains_point(root_turn_xy_nm)
                        and bridge_lane.contains_point(leaf_turn_xy_nm)
                        and leaf_lane.contains_point(leaf_turn_xy_nm)
                    ):
                        continue
                    leaf_turn_run = _lane_run_coordinate(
                        leaf_lane, leaf_turn_xy_nm
                    )
                    bridge_runs = (
                        _lane_run_coordinate(bridge_lane, root_turn_xy_nm),
                        _lane_run_coordinate(bridge_lane, leaf_turn_xy_nm),
                    )
                    root_interval = root_lane.through_interval_nm
                    bridge_interval = (
                        min(bridge_runs),
                        min(
                            bridge_lane.through_interval_nm[1],
                            max(bridge_runs) + 1,
                        ),
                    )
                    leaf_interval = (
                        min(leaf_bounds[0], leaf_turn_run),
                        min(
                            leaf_lane.through_interval_nm[1],
                            max(leaf_bounds[1], leaf_turn_run) + 1,
                        ),
                    )
                    if any(
                        interval[1] <= interval[0]
                        for interval in (
                            root_interval,
                            bridge_interval,
                            leaf_interval,
                        )
                    ):
                        continue
                    reached("root_landing_absent", 3)
                    pattern_kind = "POWER_LIFTED_DROP"
                    root_use = _compiled_corridor_use(
                        lane=root_lane,
                        interval_nm=root_interval,
                        net=net,
                        pattern_kind=pattern_kind,
                        ordinal=0,
                        width_nm=root_width_nm,
                    )
                    bridge_use = _compiled_corridor_use(
                        lane=bridge_lane,
                        interval_nm=bridge_interval,
                        net=net,
                        pattern_kind=pattern_kind,
                        ordinal=2 * leaf_ordinal + 1,
                        width_nm=_tech_min_width_nm(
                            tech, bridge_lane.layer_n
                        ),
                    )
                    leaf_use = _compiled_corridor_use(
                        lane=leaf_lane,
                        interval_nm=leaf_interval,
                        net=net,
                        pattern_kind=pattern_kind,
                        ordinal=2 * leaf_ordinal + 2,
                        width_nm=_tech_min_width_nm(tech, leaf_lane.layer_n),
                    )
                    if (
                        root_use is None
                        or bridge_use is None
                        or leaf_use is None
                    ):
                        continue
                    root_boundary_key = (
                        *_component_boundary_cache_key(
                            root[0], root[1], root_component.component_ref
                        ),
                        net,
                        root_lane.lane_id,
                        root_use.use_id,
                        required_multiplicity,
                    )
                    if root_boundary_key not in shared_boundary_bundles:
                        shared_boundary_bundles[root_boundary_key] = (
                            _landing_bundle(
                                net=net,
                                terminal=root[0],
                                endpoint=root[1],
                                lane=root_lane,
                                use=root_use,
                                ledger=ledger,
                                incidence_index=incidence_index,
                                required_multiplicity=required_multiplicity,
                                budget=budget,
                                option_budget=option_budget,
                                required_component_ref=(
                                    root_component.component_ref
                                ),
                                lazy_landing_access=lazy_landing_access,
                            )
                        )
                    root_boundary = shared_boundary_bundles[
                        root_boundary_key
                    ]
                    if root_boundary is None:
                        continue
                    reached("turn_landing_absent", 4)
                    root_turn = _internal_landing_bundle(
                        net=net,
                        terminal_provenance=leaf[0].provenance,
                        left_lane=root_lane,
                        left_use=root_use,
                        right_lane=bridge_lane,
                        right_use=bridge_use,
                        ledger=ledger,
                        incidence_index=incidence_index,
                        required_multiplicity=required_multiplicity,
                        budget=budget,
                        option_budget=option_budget,
                        lazy_landing_access=lazy_landing_access,
                    )
                    if root_turn is None:
                        continue
                    leaf_turn = _internal_landing_bundle(
                        net=net,
                        terminal_provenance=leaf[0].provenance,
                        left_lane=bridge_lane,
                        left_use=bridge_use,
                        right_lane=leaf_lane,
                        right_use=leaf_use,
                        ledger=ledger,
                        incidence_index=incidence_index,
                        required_multiplicity=required_multiplicity,
                        budget=budget,
                        option_budget=option_budget,
                        lazy_landing_access=lazy_landing_access,
                    )
                    if leaf_turn is None:
                        continue
                    reached("leaf_access_absent", 5)
                    leaf_fragment = _compile_signal_trunk_endpoint_access(
                        row=leaf,
                        trunk_lane=leaf_lane,
                        trunk_use=leaf_use,
                        net=net,
                        ordinal_base=4 * leaf_ordinal + 3,
                        allowed_layers=allowed_layers,
                        required_multiplicity=required_multiplicity,
                        ledger=ledger,
                        conductor_index=conductor_index,
                        incidence_index=incidence_index,
                        layer_n_by_key=layer_n_by_key,
                        tech=tech,
                        budget=budget,
                        option_budget=option_budget,
                        lazy_landing_access=lazy_landing_access,
                        lane_layer_index=lane_layer_index,
                        component_boundary_lane_cache=(
                            component_boundary_lane_cache
                        ),
                        component_contact_lane_cache=(
                            component_contact_lane_cache
                        ),
                    )
                    if leaf_fragment is None:
                        continue
                    if leaf_fragment.owned_component_refs != (
                        leaf_component_ref,
                    ):
                        continue
                    root_demand, root_options = root_boundary
                    root_turn_demand, root_turn_options = root_turn
                    leaf_turn_demand, leaf_turn_options = leaf_turn
                    lifted_steps = (
                        CandidateBranchStep(
                            "landing_demand", root_demand.demand_id
                        ),
                        CandidateBranchStep("track_use", root_use.use_id),
                        CandidateBranchStep(
                            "landing_demand", root_turn_demand.demand_id
                        ),
                        CandidateBranchStep("track_use", bridge_use.use_id),
                        CandidateBranchStep(
                            "landing_demand", leaf_turn_demand.demand_id
                        ),
                        CandidateBranchStep("track_use", leaf_use.use_id),
                        *tuple(reversed(leaf_fragment.steps_to_trunk)),
                    )
                    return (
                        leaf_component_ref,
                        root_lane.lane_id,
                        (
                            root_use,
                            bridge_use,
                            leaf_use,
                            *leaf_fragment.corridor_uses,
                        ),
                        (
                            root_demand,
                            root_turn_demand,
                            leaf_turn_demand,
                            *leaf_fragment.landing_demands,
                        ),
                        root_options
                        + root_turn_options
                        + leaf_turn_options
                        + leaf_fragment.landing_options,
                        lifted_steps,
                    )
                continue
            if (
                layer_gap != 1
                or root_lane.preferred_axis == leaf_lane.preferred_axis
            ):
                continue
            turn_xy_nm = (
                (root_lane.center_nm, leaf_lane.center_nm)
                if root_lane.preferred_axis == "V"
                else (leaf_lane.center_nm, root_lane.center_nm)
            )
            if not (
                root_lane.contains_point(turn_xy_nm)
                and leaf_lane.contains_point(turn_xy_nm)
            ):
                continue
            leaf_bounds = _component_run_bounds(leaf_lane, leaf_component)
            if leaf_bounds is None:
                continue
            leaf_turn_run = _lane_run_coordinate(leaf_lane, turn_xy_nm)
            # A power root is a shared corridor-use node in the closed route
            # DAG, not a leaf-private signal segment.  Reserve the complete
            # finite lane once so every branch refers to byte-identical root
            # geometry irrespective of leaf order or turn coordinate.
            root_interval = root_lane.through_interval_nm
            leaf_interval = (
                min(leaf_bounds[0], leaf_turn_run),
                min(
                    leaf_lane.through_interval_nm[1],
                    max(leaf_bounds[1], leaf_turn_run) + 1,
                ),
            )
            if (
                root_interval[1] <= root_interval[0]
                or leaf_interval[1] <= leaf_interval[0]
            ):
                continue
            reached("root_landing_absent", 3)
            pattern_kind = "POWER_LIFTED_DROP"
            root_use = _compiled_corridor_use(
                lane=root_lane,
                interval_nm=root_interval,
                net=net,
                pattern_kind=pattern_kind,
                ordinal=0,
                width_nm=root_width_nm,
            )
            leaf_use = _compiled_corridor_use(
                lane=leaf_lane,
                interval_nm=leaf_interval,
                net=net,
                pattern_kind=pattern_kind,
                ordinal=2 * leaf_ordinal + 1,
                width_nm=_tech_min_width_nm(tech, leaf_lane.layer_n),
            )
            if root_use is None or leaf_use is None:
                continue
            root_boundary_key = (
                *_component_boundary_cache_key(
                    root[0], root[1], root_component.component_ref
                ),
                net,
                root_lane.lane_id,
                root_use.use_id,
                required_multiplicity,
            )
            if root_boundary_key not in shared_boundary_bundles:
                shared_boundary_bundles[root_boundary_key] = _landing_bundle(
                    net=net,
                    terminal=root[0],
                    endpoint=root[1],
                    lane=root_lane,
                    use=root_use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=required_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    required_component_ref=root_component.component_ref,
                    lazy_landing_access=lazy_landing_access,
                )
            root_boundary = shared_boundary_bundles[root_boundary_key]
            if root_boundary is None:
                continue
            reached("turn_landing_absent", 4)
            turn = _internal_landing_bundle(
                net=net,
                terminal_provenance=leaf[0].provenance,
                left_lane=root_lane,
                left_use=root_use,
                right_lane=leaf_lane,
                right_use=leaf_use,
                ledger=ledger,
                incidence_index=incidence_index,
                required_multiplicity=required_multiplicity,
                budget=budget,
                option_budget=option_budget,
                lazy_landing_access=lazy_landing_access,
            )
            if turn is None:
                continue
            reached("leaf_landing_absent", 5)
            leaf_boundary = _landing_bundle(
                net=net,
                terminal=leaf[0],
                endpoint=leaf[1],
                lane=leaf_lane,
                use=leaf_use,
                ledger=ledger,
                incidence_index=incidence_index,
                required_multiplicity=required_multiplicity,
                budget=budget,
                option_budget=option_budget,
                required_component_ref=leaf_component_ref,
                lazy_landing_access=lazy_landing_access,
            )
            if leaf_boundary is None:
                continue
            root_demand, root_options = root_boundary
            turn_demand, turn_options = turn
            leaf_demand, leaf_options = leaf_boundary
            return (
                leaf_component_ref,
                root_lane.lane_id,
                (root_use, leaf_use),
                (root_demand, turn_demand, leaf_demand),
                root_options + turn_options + leaf_options,
                (
                    CandidateBranchStep(
                        "landing_demand", root_demand.demand_id
                    ),
                    CandidateBranchStep("track_use", root_use.use_id),
                    CandidateBranchStep(
                        "landing_demand", turn_demand.demand_id
                    ),
                    CandidateBranchStep("track_use", leaf_use.use_id),
                    CandidateBranchStep(
                        "landing_demand", leaf_demand.demand_id
                    ),
                    CandidateBranchStep(
                        "owned_component", leaf_component_ref
                    ),
                ),
            )
    if failure_refs is not None:
        if lazy_landing_access is not None:
            failure_refs.append(
                "lifted_root_join="
                f"{_bounded_diagnostic_provenance(root[0].provenance)}:"
                f"candidate_lanes={root_candidate_count}:"
                f"contact_lanes={contacting_root_lane_count}"
            )
        if (
            failure_stage == "leaf_boundary_lane_absent"
            and leaf_boundary_failure_stats.get("candidate_lanes", 0) > 0
        ):
            failure_stage = "leaf_boundary_landing_absent"
        failure_refs.extend(leaf_boundary_failure_refs)
        failure_refs.append(
            "lifted_stage="
            f"{_bounded_diagnostic_provenance(leaf[0].provenance)}:"
            f"{failure_stage}"
        )
    return None


def _compile_power_spine_candidate(
    *,
    topology_candidate: TopologyCandidate,
    demand: RoutingDemand,
    routing_world: RoutingWorldProvenance,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
    tech: Any,
    topology_digest: str,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    component_pair_landing_access: _CertifiedComponentPairLandingAccess,
    lazy_landing_access: _ExactLazyLandingAccess | None = None,
    lane_layer_index: _BudgetedLaneLayerIndex | None = None,
    _fixed_lifted_root: tuple[str, str, str] | None = None,
) -> tuple[tuple[CompleteNetCandidate, ...], tuple[str, tuple[str, ...]] | None]:
    if topology_candidate.topology_class != "power_spine_tree":
        return (), None
    endpoint_rows = _resolved_endpoint_rows_any(
        topology_candidate=topology_candidate,
        demand=demand,
        routing_world=routing_world,
        incidence_index=incidence_index,
        layer_n_by_key=layer_n_by_key,
    )
    rail_components = tuple(
        component
        for component in conductor_index.components
        if component.net == demand.net
        and "rail" in component.kinds
        and component.atom_facts
    )
    if not endpoint_rows:
        terminal_resolution_refs = []
        pin_match_counts: dict[
            tuple[str, PointNm, LayerKey, str], int
        ] = {}
        pin_by_key: dict[
            tuple[str, PointNm, LayerKey, str], tuple[Any, ...]
        ] = {}
        for pin in routing_world.pins:
            if pin[0] != demand.net:
                continue
            key = (
                pin[0],
                tuple(pin[3:5]),
                tuple(pin[5:7]),
                _pin_fact_provenance(pin),
            )
            pin_match_counts[key] = pin_match_counts.get(key, 0) + 1
            pin_by_key.setdefault(key, pin)
        for terminal in topology_candidate.branch_terminals:
            key = (
                demand.net,
                terminal.xy_nm,
                terminal.layer,
                terminal.provenance,
            )
            match_count = pin_match_counts.get(key, 0)
            bounded_provenance = _bounded_diagnostic_provenance(
                terminal.provenance
            )
            if match_count != 1:
                terminal_resolution_refs.append(
                    f"terminal_matches={bounded_provenance}:{match_count}"
                )
                continue
            endpoint = incidence_index.resolve_endpoint(pin_by_key[key])
            if endpoint is None:
                terminal_resolution_refs.append(
                    f"endpoint_incidence_missing={bounded_provenance}"
                )
            elif _endpoint_access(
                terminal,
                endpoint,
                layer_n_by_key=layer_n_by_key,
            ) is None:
                terminal_resolution_refs.append(
                    f"endpoint_layer_missing={bounded_provenance}"
                )
        return (), (
            "POWER_ROOT_UNAVAILABLE",
            (
                demand.net,
                "reason=no-exact-endpoint-rows",
                f"branch_terminals={len(topology_candidate.branch_terminals)}",
                f"world_net_pins={sum(pin[0] == demand.net for pin in routing_world.pins)}",
                *terminal_resolution_refs[:4],
            ),
        )
    if not rail_components:
        return (), (
            "POWER_ROOT_UNAVAILABLE",
            (
                demand.net,
                "reason=no-owned-rail-component",
                f"endpoint_rows={len(endpoint_rows)}",
            ),
        )
    allowed_layers = frozenset(
        topology_candidate.layers + topology_candidate.transport_layers
    )
    if lane_layer_index is None:
        lane_layer_index = _BudgetedLaneLayerIndex(incidence_index)

    goal_targets = tuple(
        goal.target.provenance
        for goal in demand.required_goals
        if goal.target is not None
    )
    if (
        len(goal_targets) != len(demand.required_goals)
        or len(set(goal_targets)) != len(goal_targets)
    ):
        return (), ("POWER_ROOT_UNAVAILABLE", (demand.net, "invalid-goal-leaf"))

    roots_by_component = []
    structural_access_count = 0
    wide_access_count = 0
    goal_excluded_count = 0
    for component in rail_components:
        structural_rows = tuple(
            row
            for row in endpoint_rows
            if _is_structural_rail_access(row, component)
        )
        structural_access_count += len(structural_rows)
        wide_rows = tuple(
            row
            for row in structural_rows
            if _component_has_exact_rail_width_atom(
                component,
                root_xy_nm=row[0].xy_nm,
                net=demand.net,
                width_nm=topology_candidate.width_nm,
            )
        )
        wide_access_count += len(wide_rows)
        goal_excluded_count += sum(
            row[0].provenance in goal_targets for row in wide_rows
        )
        roots = tuple(sorted(
            (
                row
                for row in wide_rows
                if row[0].provenance not in goal_targets
            ),
            key=lambda row: row[0].provenance,
        ))
        roots_by_component.extend((component, root) for root in roots)
    if not roots_by_component:
        return (), (
            "POWER_ROOT_UNAVAILABLE",
            (
                demand.net,
                "reason=no-qualified-rail-root",
                f"endpoint_rows={len(endpoint_rows)}",
                f"rail_components={len(rail_components)}",
                f"structural_accesses={structural_access_count}",
                f"wide_accesses={wide_access_count}",
                f"goal_excluded={goal_excluded_count}",
                f"required_width_nm={topology_candidate.width_nm}",
            ),
        )

    candidates = []
    local_drop_failure_refs: list[str] = []
    component_contact_lane_cache: dict[str, tuple[LaneRef, ...]] = {}
    component_boundary_lane_cache: dict[
        tuple[Any, ...], tuple[LaneRef, ...]
    ] = {}
    leaf_component_contact_lane_cache: dict[
        tuple[str, str], tuple[tuple[LaneRef, int], ...]
    ] = {}
    for component, root in sorted(
        roots_by_component,
        key=lambda item: (item[0].component_ref, item[1][0].provenance),
    ):
        budget.consume("power-spine root expansion")
        leaves = tuple(sorted(
            (row for row in endpoint_rows if row is not root),
            key=lambda row: row[0].provenance,
        ))
        if not leaves:
            continue
        outside_leaves = tuple(
            leaf
            for leaf in leaves
            if component.component_ref not in leaf[1].conductor_component_refs
        )
        component_layer_n = layer_n_by_key[component.layer_key]
        if outside_leaves and component_layer_n not in allowed_layers:
            # A local drop is physical route geometry, so its layer must be
            # named by the topology candidate's signed routing closure.  The
            # existing rail component itself remains usable for a genuinely
            # component-only candidate, which emits no corridor use.
            local_drop_failure_refs.append(
                f"unsigned_component_layer=M{component_layer_n}"
            )
            continue
        root_contact_lanes = component_contact_lane_cache.get(
            component.component_ref
        )
        if root_contact_lanes is None:
            root_contact_lanes = tuple(
                lane
                for lane in incidence_index.lanes
                if lane.layer_n == component_layer_n
                and _lane_may_contact_component_bbox(lane, component)
                and _lane_component_contact_run(
                    lane,
                    component,
                    0,
                    budget,
                ) is not None
            )
            component_contact_lane_cache[
                component.component_ref
            ] = root_contact_lanes
        branches = []
        corridor_uses: dict[str, CorridorUse] = {}
        landing_demands: dict[str, LandingDemand] = {}
        landing_options: dict[str, LandingOption] = {}
        shared_boundary_bundles: dict[
            tuple[Any, ...],
            tuple[LandingDemand, tuple[LandingOption, ...]] | None,
        ] = {}

        def record_unique(
            values: dict[str, Any], ref_id: str, value: Any, kind: str
        ) -> None:
            existing = values.get(ref_id)
            if existing is not None and existing != value:
                raise PatternCompileInputError(
                    f"shared power {kind} id has conflicting payload: {ref_id}"
                )
            values.setdefault(ref_id, value)

        def record_landing_bundle(
            demand_item: LandingDemand,
            options: Sequence[LandingOption],
        ) -> None:
            record_unique(
                landing_demands,
                demand_item.demand_id,
                demand_item,
                "landing-demand",
            )
            for option in options:
                record_unique(
                    landing_options,
                    option.option_id,
                    option,
                    "landing-option",
                )

        owned_component_refs = {component.component_ref}
        lifted_root_lanes = _component_boundary_lanes(
            terminal=root[0],
            endpoint=root[1],
            component_ref=component.component_ref,
            net=demand.net,
            ledger=ledger,
            incidence_index=incidence_index,
            budget=budget,
            lazy_landing_access=lazy_landing_access,
        )
        root_identity = (
            component.component_ref,
            root[0].provenance,
        )
        if _fixed_lifted_root is None and outside_leaves and lifted_root_lanes:
            # Root-lane choice is a candidate dimension, not mutable state of
            # a leaf-by-leaf compiler.  Fix one exact root first and compile
            # every leaf against it; only a whole-tree success may escape.
            for root_lane in lifted_root_lanes:
                fixed_candidates, fixed_failure = (
                    _compile_power_spine_candidate(
                        topology_candidate=topology_candidate,
                        demand=demand,
                        routing_world=routing_world,
                        ledger=ledger,
                        conductor_index=conductor_index,
                        incidence_index=incidence_index,
                        layer_n_by_key=layer_n_by_key,
                        tech=tech,
                        topology_digest=topology_digest,
                        budget=budget,
                        option_budget=option_budget,
                        component_pair_landing_access=(
                            component_pair_landing_access
                        ),
                        lazy_landing_access=lazy_landing_access,
                        lane_layer_index=lane_layer_index,
                        _fixed_lifted_root=(
                            *root_identity,
                            root_lane.lane_id,
                        ),
                    )
                )
                candidates.extend(fixed_candidates)
                if fixed_failure is not None:
                    local_drop_failure_refs.extend(fixed_failure[1])
            continue
        if _fixed_lifted_root is not None:
            if _fixed_lifted_root[:2] != root_identity:
                continue
            lifted_root_lanes = tuple(
                lane
                for lane in lifted_root_lanes
                if lane.lane_id == _fixed_lifted_root[2]
            )
            if not lifted_root_lanes:
                local_drop_failure_refs.append(
                    "fixed_lifted_root_lane_missing="
                    f"{_fixed_lifted_root[2]}"
                )
                continue
        component_infeasible = False
        for leaf_ordinal, leaf in enumerate(leaves):
            budget.consume("power-spine local-drop leaf expansion")
            ordered_use_ids = ()
            ordered_landing_ids = ()
            ordered_steps = [
                CandidateBranchStep("owned_component", component.component_ref)
            ]
            if component.component_ref not in leaf[1].conductor_component_refs:
                component_pair_bundle = _component_pair_landing_bundle(
                    net=demand.net,
                    terminal=leaf[0],
                    endpoint=leaf[1],
                    root_component_ref=component.component_ref,
                    conductor_index=conductor_index,
                    layer_n_by_key=layer_n_by_key,
                    allowed_layers=allowed_layers,
                    required_multiplicity=topology_candidate.via_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    component_pair_landing_access=(
                        component_pair_landing_access
                    ),
                )
                if component_pair_bundle is not None:
                    leaf_component_ref, landing_demand, options = (
                        component_pair_bundle
                    )
                    record_landing_bundle(landing_demand, options)
                    ordered_landing_ids = (landing_demand.demand_id,)
                    ordered_steps.append(
                        CandidateBranchStep(
                            "landing_demand", landing_demand.demand_id
                        )
                    )
                    ordered_steps.append(
                        CandidateBranchStep(
                            "owned_component", leaf_component_ref
                        )
                    )
                    owned_component_refs.add(leaf_component_ref)
                else:
                    selected = None
                    same_layer_lanes = 0
                    component_contact_lanes = 0
                    for lane in root_contact_lanes:
                        budget.consume("power-spine local-drop lane expansion")
                        same_layer_lanes += 1
                        terminal_run = _lane_run_coordinate(lane, leaf[0].xy_nm)
                        contact_run = _lane_component_contact_run(
                            lane,
                            component,
                            terminal_run,
                            budget,
                        )
                        if contact_run is None:
                            continue
                        component_contact_lanes += 1
                        leaf_direct = lane in incidence_index.lanes_at_point(
                            leaf[0].layer, leaf[0].xy_nm
                        )
                        interval_nm = (
                            min(contact_run, terminal_run),
                            max(contact_run, terminal_run) + 1,
                        )
                        use = _compiled_corridor_use(
                            lane=lane,
                            interval_nm=interval_nm,
                            net=demand.net,
                            pattern_kind="POWER_LOCAL_DROP",
                            ordinal=leaf_ordinal,
                            width_nm=_tech_min_width_nm(tech, lane.layer_n),
                        )
                        if use is None:
                            continue
                        bundle = None
                        if not leaf_direct:
                            bundle = _landing_bundle(
                                net=demand.net,
                                terminal=leaf[0],
                                endpoint=leaf[1],
                                lane=lane,
                                use=use,
                                ledger=ledger,
                                incidence_index=incidence_index,
                                required_multiplicity=(
                                    topology_candidate.via_multiplicity
                                ),
                                budget=budget,
                                option_budget=option_budget,
                                lazy_landing_access=lazy_landing_access,
                            )
                            if bundle is None:
                                continue
                        selected = (use, bundle)
                        break
                    if selected is not None:
                        use, bundle = selected
                        record_unique(
                            corridor_uses, use.use_id, use, "corridor-use"
                        )
                        ordered_use_ids = (use.use_id,)
                        ordered_steps.append(
                            CandidateBranchStep("track_use", use.use_id)
                        )
                        if bundle is not None:
                            landing_demand, options = bundle
                            record_landing_bundle(landing_demand, options)
                            ordered_landing_ids = (landing_demand.demand_id,)
                            ordered_steps.append(
                                CandidateBranchStep(
                                    "landing_demand", landing_demand.demand_id
                                )
                            )
                    else:
                        lifted_failure_refs: list[str] = []
                        lifted_bundle = _lifted_power_drop_bundle(
                            net=demand.net,
                            root=root,
                            leaf=leaf,
                            root_component=component,
                            root_lanes=lifted_root_lanes,
                            leaf_ordinal=leaf_ordinal,
                            root_width_nm=topology_candidate.width_nm,
                            allowed_layers=allowed_layers,
                            required_multiplicity=(
                                topology_candidate.via_multiplicity
                            ),
                            ledger=ledger,
                            conductor_index=conductor_index,
                            incidence_index=incidence_index,
                            layer_n_by_key=layer_n_by_key,
                            tech=tech,
                            budget=budget,
                            option_budget=option_budget,
                            lazy_landing_access=lazy_landing_access,
                            lane_layer_index=lane_layer_index,
                            component_boundary_lane_cache=(
                                component_boundary_lane_cache
                            ),
                            component_contact_lane_cache=(
                                leaf_component_contact_lane_cache
                            ),
                            shared_boundary_bundles=shared_boundary_bundles,
                            failure_refs=lifted_failure_refs,
                        )
                        if lifted_bundle is None:
                            provenance = _bounded_diagnostic_provenance(
                                leaf[0].provenance
                            )
                            if same_layer_lanes == 0:
                                reason = "leaf_no_layer_lane"
                            elif component_contact_lanes == 0:
                                reason = "leaf_no_component_contact"
                            else:
                                reason = "leaf_no_landing"
                            local_drop_failure_refs.append(
                                f"{reason}={provenance}:M{component_layer_n}"
                            )
                            local_drop_failure_refs.append(
                                "root_component_bbox="
                                f"M{component_layer_n}:"
                                + ",".join(
                                    str(value) for value in component.bbox_nm
                                )
                            )
                            local_drop_failure_refs.extend(lifted_failure_refs)
                            for leaf_component_ref in sorted(
                                leaf[1].conductor_component_refs
                            )[:1]:
                                leaf_component = conductor_index.component(
                                    leaf_component_ref
                                )
                                leaf_layer_n = layer_n_by_key[
                                    leaf_component.layer_key
                                ]
                                local_drop_failure_refs.append(
                                    "leaf_component_bbox="
                                    f"{provenance}:M{leaf_layer_n}:"
                                    + ",".join(
                                        str(value)
                                        for value in leaf_component.bbox_nm
                                    )
                                )
                            component_infeasible = True
                            break
                        (
                            leaf_component_ref,
                            _root_lane_id,
                            lifted_uses,
                            lifted_demands,
                            lifted_options,
                            lifted_steps,
                        ) = lifted_bundle
                        lifted_use_ids = tuple(
                            step.ref_id
                            for step in lifted_steps
                            if step.kind == "track_use"
                        )
                        lifted_landing_ids = tuple(
                            step.ref_id
                            for step in lifted_steps
                            if step.kind == "landing_demand"
                        )
                        repeated_step_kinds = tuple(
                            label
                            for label, refs in (
                                ("track_use", lifted_use_ids),
                                ("landing_demand", lifted_landing_ids),
                            )
                            if len(refs) != len(set(refs))
                        )
                        if repeated_step_kinds:
                            local_drop_failure_refs.append(
                                "lifted_branch_repeated_route_ref="
                                f"{_bounded_diagnostic_provenance(leaf[0].provenance)}:"
                                + ",".join(repeated_step_kinds)
                            )
                            component_infeasible = True
                            break
                        for use in lifted_uses:
                            record_unique(
                                corridor_uses,
                                use.use_id,
                                use,
                                "corridor-use",
                            )
                        for landing_demand in lifted_demands:
                            record_landing_bundle(
                                landing_demand,
                                tuple(
                                    option
                                    for option in lifted_options
                                    if option.demand_id
                                    == landing_demand.demand_id
                                ),
                            )
                        owned_component_refs.add(leaf_component_ref)
                        ordered_use_ids = lifted_use_ids
                        ordered_landing_ids = lifted_landing_ids
                        ordered_steps.extend(lifted_steps)
            branch_id = canonical_pattern_id(
                "pattern-branch",
                {
                    "component_ref": component.component_ref,
                    "leaf": leaf[0].provenance,
                    "net": demand.net,
                    "ordered_steps": [
                        {"kind": step.kind, "ref_id": step.ref_id}
                        for step in ordered_steps
                    ],
                    "pattern_kind": (
                        "POWER_ALREADY_ON_SPINE"
                        if not outside_leaves
                        else "POWER_SPINE_LOCAL_DROPS"
                    ),
                    "root": root[0].provenance,
                },
            )
            branches.append(
                CandidateBranch(
                    branch_id=branch_id,
                    source_endpoint=root[0].provenance,
                    target_endpoint=leaf[0].provenance,
                    ordered_use_ids=ordered_use_ids,
                    ordered_landing_demand_ids=ordered_landing_ids,
                    ordered_steps=tuple(ordered_steps),
                )
            )
        if component_infeasible:
            continue
        branches_by_target = {
            branch.target_endpoint: branch for branch in branches
        }
        if any(target not in branches_by_target for target in goal_targets):
            local_drop_failure_refs.append("required_goal_branch_missing")
            continue
        candidate = CompleteNetCandidate(
            candidate_id="pending-content-address",
            net=demand.net,
            topology_class=topology_candidate.topology_class,
            branch_edges=tuple(
                (branch.source_endpoint, branch.target_endpoint)
                for branch in branches
            ),
            corridor_uses=tuple(corridor_uses.values()),
            landing_demands=tuple(landing_demands.values()),
            required_goal_refs=topology_candidate.required_goal_refs,
            width_nm=topology_candidate.width_nm,
            via_multiplicity=(
                0 if not outside_leaves else topology_candidate.via_multiplicity
            ),
            source_topology_digest=topology_digest,
            program_kind=(
                "COMPONENT_ONLY"
                if not outside_leaves
                else "POWER_SPINE_LOCAL_DROPS"
            ),
            branches=tuple(branches),
            landing_options=tuple(landing_options.values()),
            goal_branch_coverage=tuple(
                GoalBranchCoverage(
                    declaration_ref=goal.declaration_ref,
                    branch_id=branches_by_target[goal.target.provenance].branch_id,
                    target_provenance=goal.target.provenance,
                )
                for goal in demand.required_goals
                if goal.target is not None
            ),
            endpoint_accesses=(
                root[2],
                *(leaf[2] for leaf in leaves),
            ),
            owned_component_use_refs=tuple(sorted(owned_component_refs)),
        )
        if {
            access.provenance for access in candidate.endpoint_accesses
        } != {row[0].provenance for row in endpoint_rows}:
            local_drop_failure_refs.append("endpoint_access_set_mismatch")
            continue
        candidates.append(_content_address_candidate(candidate))
    if candidates:
        unique_candidates = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        return tuple(
            unique_candidates[candidate_id]
            for candidate_id in sorted(unique_candidates)
        ), None
    return (), (
        "POWER_LOCAL_DROP_UNAVAILABLE",
        (
            demand.net,
            "no-fixed-lane-site-component-join",
            *tuple(sorted(set(local_drop_failure_refs)))[:8],
        ),
    )


def _orient_two_pin_branch(
    endpoint_rows: Sequence[
        tuple[TerminalDemand, EndpointIncidence, CandidateEndpointAccess]
    ],
    demand: RoutingDemand,
) -> tuple[
    tuple[TerminalDemand, EndpointIncidence, CandidateEndpointAccess],
    tuple[TerminalDemand, EndpointIncidence, CandidateEndpointAccess],
] | None:
    goals = tuple(demand.required_goals)
    if len(goals) > 1:
        return None
    if goals:
        goal = goals[0]
        target_rows = (
            tuple(
                row
                for row in endpoint_rows
                if row[0].provenance == goal.target.provenance
            )
            if goal.target is not None
            else ()
        )
        if len(target_rows) != 1:
            return None
        target = target_rows[0]
        source = next(row for row in endpoint_rows if row is not target)
        return source, target
    source, target = sorted(endpoint_rows, key=lambda row: row[0].provenance)
    return source, target


def _compile_direct_candidate(
    *,
    topology_candidate: TopologyCandidate,
    demand: RoutingDemand,
    routing_world: RoutingWorldProvenance,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
    tech: Any,
    topology_digest: str,
    budget: _IncidenceBudget,
) -> tuple[CompleteNetCandidate, ...]:
    endpoint_rows = _resolved_endpoint_rows(
        topology_candidate=topology_candidate,
        demand=demand,
        routing_world=routing_world,
        incidence_index=incidence_index,
        layer_n_by_key=layer_n_by_key,
    )
    if not endpoint_rows:
        return ()

    allowed_layers = frozenset(
        topology_candidate.layers + topology_candidate.transport_layers
    )
    candidates: list[CompleteNetCandidate] = []
    oriented = _orient_two_pin_branch(endpoint_rows, demand)
    if oriented is None:
        return ()
    source, target = oriented
    common_lanes = tuple(sorted(
        set(incidence_index.lanes_at_point(source[0].layer, source[0].xy_nm))
        & set(incidence_index.lanes_at_point(target[0].layer, target[0].xy_nm))
    ))
    for lane in common_lanes:
        budget.consume("two-pin direct point lane expansion")
        if lane.layer_n not in allowed_layers:
            continue
        run_coordinates = (
            _lane_run_coordinate(lane, source[0].xy_nm),
            _lane_run_coordinate(lane, target[0].xy_nm),
        )
        interval_nm = min(run_coordinates), max(run_coordinates) + 1
        if interval_nm[1] <= interval_nm[0]:
            continue
        use = _compiled_corridor_use(
            lane=lane,
            interval_nm=interval_nm,
            net=demand.net,
            pattern_kind="DIRECT",
            ordinal=0,
            width_nm=max(
                topology_candidate.width_nm,
                _tech_min_width_nm(tech, lane.layer_n),
            ),
        )
        if use is None:
            continue
        branch_id = canonical_pattern_id(
            "pattern-branch",
            {
                "net": demand.net,
                "pattern_kind": "DIRECT",
                "source": source[0].provenance,
                "target": target[0].provenance,
                "use_id": use.use_id,
            },
        )
        candidate = CompleteNetCandidate(
            candidate_id="pending-content-address",
            net=demand.net,
            topology_class=topology_candidate.topology_class,
            branch_edges=((source[0].provenance, target[0].provenance),),
            corridor_uses=(use,),
            landing_demands=(),
            required_goal_refs=topology_candidate.required_goal_refs,
            width_nm=topology_candidate.width_nm,
            via_multiplicity=topology_candidate.via_multiplicity,
            source_topology_digest=topology_digest,
            branches=(
                CandidateBranch(
                    branch_id=branch_id,
                    source_endpoint=source[0].provenance,
                    target_endpoint=target[0].provenance,
                    ordered_use_ids=(use.use_id,),
                    ordered_landing_demand_ids=(),
                    ordered_steps=(CandidateBranchStep("track_use", use.use_id),),
                ),
            ),
            goal_branch_coverage=tuple(
                GoalBranchCoverage(
                    declaration_ref=goal.declaration_ref,
                    branch_id=branch_id,
                    target_provenance=target[0].provenance,
                )
                for goal in demand.required_goals
            ),
            endpoint_accesses=(source[2], target[2]),
        )
        candidates.append(_content_address_candidate(candidate))
    source_components = _endpoint_conductor_components(
        source[1], conductor_index=conductor_index, net=demand.net
    )
    target_components = _endpoint_conductor_components(
        target[1], conductor_index=conductor_index, net=demand.net
    )
    for source_ref, source_component in source_components:
        for target_ref, target_component in target_components:
            if source_ref == target_ref:
                continue  # the COMPONENT_ONLY compiler owns this exact case
            for lane in incidence_index.lanes:
                budget.consume("two-pin direct owned-component lane expansion")
                if lane.layer_n not in allowed_layers:
                    continue
                source_run = _lane_component_contact_run(
                    lane,
                    source_component,
                    _lane_run_coordinate(lane, source[0].xy_nm),
                    budget,
                )
                target_run = _lane_component_contact_run(
                    lane,
                    target_component,
                    _lane_run_coordinate(lane, target[0].xy_nm),
                    budget,
                )
                if source_run is None or target_run is None:
                    continue
                interval_nm = min(source_run, target_run), max(source_run, target_run) + 1
                if interval_nm[1] <= interval_nm[0]:
                    continue
                use = _compiled_corridor_use(
                    lane=lane,
                    interval_nm=interval_nm,
                    net=demand.net,
                    pattern_kind="DIRECT_OWNED_ACCESS",
                    ordinal=0,
                    width_nm=max(
                        topology_candidate.width_nm,
                        _tech_min_width_nm(tech, lane.layer_n),
                    ),
                )
                if use is None:
                    continue
                branch_id = canonical_pattern_id(
                    "pattern-branch",
                    {
                        "net": demand.net,
                        "pattern_kind": "DIRECT_OWNED_ACCESS",
                        "source": source[0].provenance,
                        "source_component": source_ref,
                        "target": target[0].provenance,
                        "target_component": target_ref,
                        "use_id": use.use_id,
                    },
                )
                candidate = CompleteNetCandidate(
                    candidate_id="pending-content-address",
                    net=demand.net,
                    topology_class=topology_candidate.topology_class,
                    branch_edges=((source[0].provenance, target[0].provenance),),
                    corridor_uses=(use,),
                    landing_demands=(),
                    required_goal_refs=topology_candidate.required_goal_refs,
                    width_nm=topology_candidate.width_nm,
                    via_multiplicity=topology_candidate.via_multiplicity,
                    source_topology_digest=topology_digest,
                    program_kind="PATTERN_TREE",
                    branches=(
                        CandidateBranch(
                            branch_id=branch_id,
                            source_endpoint=source[0].provenance,
                            target_endpoint=target[0].provenance,
                            ordered_use_ids=(use.use_id,),
                            ordered_landing_demand_ids=(),
                            ordered_steps=(
                                CandidateBranchStep("owned_component", source_ref),
                                CandidateBranchStep("track_use", use.use_id),
                                CandidateBranchStep("owned_component", target_ref),
                            ),
                        ),
                    ),
                    goal_branch_coverage=tuple(
                        GoalBranchCoverage(
                            declaration_ref=goal.declaration_ref,
                            branch_id=branch_id,
                            target_provenance=target[0].provenance,
                        )
                        for goal in demand.required_goals
                    ),
                    endpoint_accesses=(source[2], target[2]),
                    owned_component_use_refs=(source_ref, target_ref),
                )
                candidates.append(_content_address_candidate(candidate))
    return tuple(candidates)


def _site_connects_endpoint_to_lane(
    *,
    site: Any,
    terminal: TerminalDemand,
    endpoint: EndpointIncidence,
    lane: LaneRef,
    incidence_index: LaneIncidenceIndex,
    net: str,
    required_component_ref: str | None = None,
) -> bool:
    if site.owner_net not in (None, net):
        return False
    terminal_layer_n = (
        site.lower_layer_n
        if tuple(terminal.layer)
        == incidence_index.site_side(site.site_id, "lower").gds_layer_key
        else site.upper_layer_n
        if tuple(terminal.layer)
        == incidence_index.site_side(site.site_id, "upper").gds_layer_key
        else None
    )
    if terminal_layer_n is None or lane.layer_n not in (
        site.lower_layer_n,
        site.upper_layer_n,
    ):
        return False
    terminal_side = "lower" if terminal_layer_n == site.lower_layer_n else "upper"
    lane_side = "lower" if lane.layer_n == site.lower_layer_n else "upper"
    if terminal_side == lane_side:
        return False
    endpoint_side = incidence_index.site_side(site.site_id, terminal_side)
    lane_side_incidence = incidence_index.site_side(site.site_id, lane_side)
    endpoint_refs = set(endpoint.access_component_refs)
    side_refs = set(endpoint_side.owner_component_refs)
    if required_component_ref is None:
        boundary_reached = site.xy_nm == terminal.xy_nm and bool(
            endpoint_refs & side_refs
        )
    else:
        boundary_reached = (
            required_component_ref in endpoint.conductor_component_refs
            and required_component_ref in side_refs
        )
    return boundary_reached and (
        lane.track_choice_ref in lane_side_incidence.crossing_track_choices
    )


def _landing_bundle(
    *,
    net: str,
    terminal: TerminalDemand,
    endpoint: EndpointIncidence,
    lane: LaneRef,
    use: CorridorUse,
    ledger: ExplicitRoutingResourceLedger,
    incidence_index: LaneIncidenceIndex,
    required_multiplicity: int,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    required_component_ref: str | None = None,
    lazy_landing_access: _ExactLazyLandingAccess | None = None,
    selected_site_ids: tuple[str, ...] | None = None,
) -> tuple[LandingDemand, tuple[LandingOption, ...]] | None:
    if lazy_landing_access is not None:
        sites = lazy_landing_access.boundary_for_use(
            terminal,
            endpoint,
            lane,
            use,
            net=net,
            required_component_ref=required_component_ref,
            selected_site_ids=selected_site_ids,
            budget=budget,
        )
    else:
        sites = incidence_index.landing_sites_for_component_lane(
            terminal,
            endpoint,
            required_component_ref,
            lane,
            net=net,
            ledger=ledger,
            budget=budget,
            selection_limit=MAX_PATTERN_LANDING_SITES,
        )
        if selected_site_ids is not None:
            selected_set = set(selected_site_ids)
            sites = tuple(
                site for site in sites if site.site_id in selected_set
            )
    # Boundary enumeration is lane-domain complete, while a compiled branch
    # may intentionally use only a shorter interval of that lane.  Retain only
    # sites whose certified center is on the actual emitted segment; pad-only
    # overlap remains geometry evidence but cannot become an explicit contact.
        sites = tuple(
            site
            for site in sites
            if _compiled_use_contains_site_center(lane, use, site)
        )
    multiplicity = max(1, required_multiplicity)
    demand_id = canonical_pattern_id(
        "pattern-landing-demand",
        {
            "lane_id": lane.lane_id,
            "net": net,
            "required_multiplicity": multiplicity,
            "required_component_ref": required_component_ref,
            "terminal_provenance": terminal.provenance,
        },
    )
    bundle_context = {
        "demand_id": demand_id,
        "lane_id": lane.lane_id,
        "net": net,
        "terminal_provenance": terminal.provenance,
    }
    if len(sites) < multiplicity:
        option_budget.record_option_bundle(
            kind="boundary",
            site_ids=tuple(site.site_id for site in sites),
            required_multiplicity=multiplicity,
            option_count=0,
            context=bundle_context,
        )
        return None
    if selected_site_ids is not None:
        selected_set = set(selected_site_ids)
        selected = tuple(
            site for site in sites if site.site_id in selected_set
        )
        if (
            len(selected_site_ids) != multiplicity
            or len(selected_set) != multiplicity
            or len(selected) != multiplicity
        ):
            option_budget.record_option_bundle(
                kind="boundary",
                site_ids=tuple(site.site_id for site in selected),
                required_multiplicity=multiplicity,
                option_count=0,
                context=bundle_context,
            )
            return None
        factorized_sites = tuple(sorted(selected, key=lambda site: site.site_id))
    else:
        factorized_sites = tuple(sorted(sites, key=lambda site: site.site_id))
    option_budget.record_option_bundle(
        kind="boundary",
        site_ids=tuple(site.site_id for site in factorized_sites),
        required_multiplicity=multiplicity,
        option_count=len(factorized_sites),
        context=bundle_context,
    )
    options = []
    for site in factorized_sites:
        site_ids = (site.site_id,)
        contacts = (
            LandingTrackContact(
                site_id=site.site_id,
                track_choice=TrackChoiceRef(
                    use_id=use.use_id,
                    track_start=lane.track_start,
                ),
            ),
        )
        option_id = canonical_pattern_id(
            "pattern-landing-option",
            {
                "contacts": [
                    {
                        "site_id": contact.site_id,
                        "track_start": contact.track_start,
                        "use_id": contact.use_id,
                    }
                    for contact in contacts
                ],
                "demand_id": demand_id,
                "site_ids": list(site_ids),
            },
        )
        options.append(
            _intern_landing_option(
                option_budget,
                reason="boundary landing-option expansion",
                option_id=option_id,
                demand_id=demand_id,
                site_ids=site_ids,
                contacts=contacts,
            )
        )
    return (
        LandingDemand(
            demand_id=demand_id,
            terminal_provenance=terminal.provenance,
            allowed_site_ids=tuple(
                sorted({site_id for option in options for site_id in option.site_ids})
            ),
            required_multiplicity=multiplicity,
            selection_mode=(
                LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY
            ),
        ),
        tuple(options),
    )


def _compile_lift_candidate(
    *,
    topology_candidate: TopologyCandidate,
    demand: RoutingDemand,
    routing_world: RoutingWorldProvenance,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
    tech: Any,
    topology_digest: str,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None = None,
) -> tuple[CompleteNetCandidate, ...]:
    endpoint_rows = _resolved_endpoint_rows(
        topology_candidate=topology_candidate,
        demand=demand,
        routing_world=routing_world,
        incidence_index=incidence_index,
        layer_n_by_key=layer_n_by_key,
    )
    if not endpoint_rows:
        return ()
    oriented = _orient_two_pin_branch(endpoint_rows, demand)
    if oriented is None:
        return ()
    source, target = oriented
    allowed_layers = frozenset(
        topology_candidate.layers + topology_candidate.transport_layers
    )
    candidates: list[CompleteNetCandidate] = []
    for lane in incidence_index.lanes:
        budget.consume("two-pin lift lane expansion")
        if lane.layer_n not in allowed_layers:
            continue
        if lane.gds_layer_key in {source[0].layer, target[0].layer}:
            continue
        run_coordinates = (
            _lane_run_coordinate(lane, source[0].xy_nm),
            _lane_run_coordinate(lane, target[0].xy_nm),
        )
        interval_nm = min(run_coordinates), max(run_coordinates) + 1
        if interval_nm[1] <= interval_nm[0]:
            continue
        use = _compiled_corridor_use(
            lane=lane,
            interval_nm=interval_nm,
            net=demand.net,
            pattern_kind="LIFT",
            ordinal=0,
            width_nm=max(
                topology_candidate.width_nm,
                _tech_min_width_nm(tech, lane.layer_n),
            ),
        )
        if use is None:
            continue
        source_bundle = _landing_bundle(
            net=demand.net,
            terminal=source[0],
            endpoint=source[1],
            lane=lane,
            use=use,
            ledger=ledger,
            incidence_index=incidence_index,
            required_multiplicity=topology_candidate.via_multiplicity,
            budget=budget,
            option_budget=option_budget,
            lazy_landing_access=lazy_landing_access,
        )
        target_bundle = _landing_bundle(
            net=demand.net,
            terminal=target[0],
            endpoint=target[1],
            lane=lane,
            use=use,
            ledger=ledger,
            incidence_index=incidence_index,
            required_multiplicity=topology_candidate.via_multiplicity,
            budget=budget,
            option_budget=option_budget,
            lazy_landing_access=lazy_landing_access,
        )
        if source_bundle is None or target_bundle is None:
            continue
        source_demand, source_options = source_bundle
        target_demand, target_options = target_bundle
        branch_id = canonical_pattern_id(
            "pattern-branch",
            {
                "net": demand.net,
                "pattern_kind": "LIFT",
                "source": source[0].provenance,
                "source_demand": source_demand.demand_id,
                "target": target[0].provenance,
                "target_demand": target_demand.demand_id,
                "use_id": use.use_id,
            },
        )
        candidate = CompleteNetCandidate(
            candidate_id="pending-content-address",
            net=demand.net,
            topology_class=topology_candidate.topology_class,
            branch_edges=((source[0].provenance, target[0].provenance),),
            corridor_uses=(use,),
            landing_demands=(source_demand, target_demand),
            required_goal_refs=topology_candidate.required_goal_refs,
            width_nm=topology_candidate.width_nm,
            via_multiplicity=topology_candidate.via_multiplicity,
            source_topology_digest=topology_digest,
            branches=(
                CandidateBranch(
                    branch_id=branch_id,
                    source_endpoint=source[0].provenance,
                    target_endpoint=target[0].provenance,
                    ordered_use_ids=(use.use_id,),
                    ordered_landing_demand_ids=(
                        source_demand.demand_id,
                        target_demand.demand_id,
                    ),
                    ordered_steps=(
                        CandidateBranchStep(
                            "landing_demand", source_demand.demand_id
                        ),
                        CandidateBranchStep("track_use", use.use_id),
                        CandidateBranchStep(
                            "landing_demand", target_demand.demand_id
                        ),
                    ),
                ),
            ),
            landing_options=source_options + target_options,
            goal_branch_coverage=tuple(
                GoalBranchCoverage(
                    declaration_ref=goal.declaration_ref,
                    branch_id=branch_id,
                    target_provenance=target[0].provenance,
                )
                for goal in demand.required_goals
            ),
            endpoint_accesses=(source[2], target[2]),
        )
        candidates.append(
            replace(
                candidate,
                candidate_id=canonical_pattern_id(
                    "fixed-pattern-candidate",
                    _candidate_identity_payload(candidate),
                ),
            )
        )
    source_components = _endpoint_conductor_components(
        source[1], conductor_index=conductor_index, net=demand.net
    )
    target_components = _endpoint_conductor_components(
        target[1], conductor_index=conductor_index, net=demand.net
    )
    for source_ref, source_component in source_components:
        for target_ref, target_component in target_components:
            if source_ref == target_ref:
                continue
            for lane in incidence_index.lanes:
                budget.consume("two-pin owned-component lift lane expansion")
                if lane.layer_n not in allowed_layers:
                    continue
                if lane.gds_layer_key in {
                    source_component.layer_key,
                    target_component.layer_key,
                }:
                    continue
                source_bounds = _component_run_bounds(lane, source_component)
                target_bounds = _component_run_bounds(lane, target_component)
                if source_bounds is None or target_bounds is None:
                    continue
                interval_nm = (
                    min(source_bounds[0], target_bounds[0]),
                    min(
                        lane.through_interval_nm[1],
                        max(source_bounds[1], target_bounds[1]) + 1,
                    ),
                )
                if interval_nm[1] <= interval_nm[0]:
                    continue
                use = _compiled_corridor_use(
                    lane=lane,
                    interval_nm=interval_nm,
                    net=demand.net,
                    pattern_kind="LIFT_OWNED_ACCESS",
                    ordinal=0,
                    width_nm=max(
                        topology_candidate.width_nm,
                        _tech_min_width_nm(tech, lane.layer_n),
                    ),
                )
                if use is None:
                    continue
                source_bundle = _landing_bundle(
                    net=demand.net,
                    terminal=source[0],
                    endpoint=source[1],
                    lane=lane,
                    use=use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=topology_candidate.via_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    required_component_ref=source_ref,
                    lazy_landing_access=lazy_landing_access,
                )
                target_bundle = _landing_bundle(
                    net=demand.net,
                    terminal=target[0],
                    endpoint=target[1],
                    lane=lane,
                    use=use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=topology_candidate.via_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    required_component_ref=target_ref,
                    lazy_landing_access=lazy_landing_access,
                )
                if source_bundle is None or target_bundle is None:
                    continue
                source_demand, source_options = source_bundle
                target_demand, target_options = target_bundle
                branch_id = canonical_pattern_id(
                    "pattern-branch",
                    {
                        "net": demand.net,
                        "pattern_kind": "LIFT_OWNED_ACCESS",
                        "source": source[0].provenance,
                        "source_component": source_ref,
                        "source_demand": source_demand.demand_id,
                        "target": target[0].provenance,
                        "target_component": target_ref,
                        "target_demand": target_demand.demand_id,
                        "use_id": use.use_id,
                    },
                )
                candidate = CompleteNetCandidate(
                    candidate_id="pending-content-address",
                    net=demand.net,
                    topology_class=topology_candidate.topology_class,
                    branch_edges=((source[0].provenance, target[0].provenance),),
                    corridor_uses=(use,),
                    landing_demands=(source_demand, target_demand),
                    required_goal_refs=topology_candidate.required_goal_refs,
                    width_nm=topology_candidate.width_nm,
                    via_multiplicity=topology_candidate.via_multiplicity,
                    source_topology_digest=topology_digest,
                    program_kind="PATTERN_TREE",
                    branches=(
                        CandidateBranch(
                            branch_id=branch_id,
                            source_endpoint=source[0].provenance,
                            target_endpoint=target[0].provenance,
                            ordered_use_ids=(use.use_id,),
                            ordered_landing_demand_ids=(
                                source_demand.demand_id,
                                target_demand.demand_id,
                            ),
                            ordered_steps=(
                                CandidateBranchStep("owned_component", source_ref),
                                CandidateBranchStep(
                                    "landing_demand", source_demand.demand_id
                                ),
                                CandidateBranchStep("track_use", use.use_id),
                                CandidateBranchStep(
                                    "landing_demand", target_demand.demand_id
                                ),
                                CandidateBranchStep("owned_component", target_ref),
                            ),
                        ),
                    ),
                    landing_options=source_options + target_options,
                    goal_branch_coverage=tuple(
                        GoalBranchCoverage(
                            declaration_ref=goal.declaration_ref,
                            branch_id=branch_id,
                            target_provenance=target[0].provenance,
                        )
                        for goal in demand.required_goals
                    ),
                    endpoint_accesses=(source[2], target[2]),
                    owned_component_use_refs=(source_ref, target_ref),
                )
                candidates.append(_content_address_candidate(candidate))
    return tuple(candidates)


def _site_joins_lanes(
    *,
    site: Any,
    left_lane: LaneRef,
    right_lane: LaneRef,
    incidence_index: LaneIncidenceIndex,
    net: str,
) -> bool:
    if site.owner_net not in (None, net) or not (
        left_lane.contains_point(site.xy_nm)
        and right_lane.contains_point(site.xy_nm)
    ):
        return False
    if {left_lane.layer_n, right_lane.layer_n} != {
        site.lower_layer_n,
        site.upper_layer_n,
    }:
        return False
    left_side = "lower" if left_lane.layer_n == site.lower_layer_n else "upper"
    right_side = "lower" if right_lane.layer_n == site.lower_layer_n else "upper"
    if left_side == right_side:
        return False
    return (
        left_lane.track_choice_ref
        in incidence_index.site_side(site.site_id, left_side).crossing_track_choices
        and right_lane.track_choice_ref
        in incidence_index.site_side(site.site_id, right_side).crossing_track_choices
    )


def _internal_landing_bundle(
    *,
    net: str,
    terminal_provenance: str,
    left_lane: LaneRef,
    left_use: CorridorUse,
    right_lane: LaneRef,
    right_use: CorridorUse,
    ledger: ExplicitRoutingResourceLedger,
    incidence_index: LaneIncidenceIndex,
    required_multiplicity: int,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None = None,
    selected_site_ids: tuple[str, ...] | None = None,
) -> tuple[LandingDemand, tuple[LandingOption, ...]] | None:
    if lazy_landing_access is not None:
        sites = lazy_landing_access.internal(
            left_lane, right_lane, net=net, budget=budget
        )
        if selected_site_ids is not None:
            selected_set = set(selected_site_ids)
            sites = tuple(site for site in sites if site.site_id in selected_set)
    else:
        sites = incidence_index.landing_sites_for_lane_pair(
            left_lane,
            right_lane,
            net=net,
            ledger=ledger,
            budget=budget,
            selection_limit=MAX_PATTERN_LANDING_SITES,
        )
        if selected_site_ids is not None:
            selected_set = set(selected_site_ids)
            sites = tuple(
                site for site in sites if site.site_id in selected_set
            )
    # Exact lane-pair incidence is broader than either shortened CorridorUse.
    # A closed landing option must correlate its center with both emitted
    # segments, otherwise the contact would be unselectable on replay.
    sites = tuple(
        site
        for site in sites
        if _compiled_use_contains_site_center(left_lane, left_use, site)
        and _compiled_use_contains_site_center(right_lane, right_use, site)
    )
    multiplicity = max(1, required_multiplicity)
    demand_id = canonical_pattern_id(
        "pattern-landing-demand",
        {
            "left_lane_id": left_lane.lane_id,
            "net": net,
            "required_multiplicity": multiplicity,
            "right_lane_id": right_lane.lane_id,
            "terminal_provenance": terminal_provenance,
        },
    )
    bundle_context = {
        "demand_id": demand_id,
        "left_lane_id": left_lane.lane_id,
        "net": net,
        "right_lane_id": right_lane.lane_id,
        "terminal_provenance": terminal_provenance,
    }
    if len(sites) < multiplicity:
        option_budget.record_option_bundle(
            kind="internal",
            site_ids=tuple(site.site_id for site in sites),
            required_multiplicity=multiplicity,
            option_count=0,
            context=bundle_context,
        )
        return None
    if selected_site_ids is not None:
        selected_set = set(selected_site_ids)
        selected = tuple(
            site for site in sites if site.site_id in selected_set
        )
        if (
            len(selected_site_ids) != multiplicity
            or len(selected_set) != multiplicity
            or len(selected) != multiplicity
        ):
            option_budget.record_option_bundle(
                kind="internal",
                site_ids=tuple(site.site_id for site in selected),
                required_multiplicity=multiplicity,
                option_count=0,
                context=bundle_context,
            )
            return None
        atom_sites = tuple(sorted(selected, key=lambda site: site.site_id))
    else:
        atom_sites = tuple(sorted(sites, key=lambda site: site.site_id))
    option_budget.record_option_bundle(
        kind="internal",
        site_ids=tuple(site.site_id for site in atom_sites),
        required_multiplicity=multiplicity,
        option_count=len(atom_sites),
        context=bundle_context,
    )
    options = []
    for site in atom_sites:
        contacts = (
            LandingTrackContact(
                site_id=site.site_id,
                track_choice=TrackChoiceRef(
                    use_id=left_use.use_id,
                    track_start=left_lane.track_start,
                ),
            ),
            LandingTrackContact(
                site_id=site.site_id,
                track_choice=TrackChoiceRef(
                    use_id=right_use.use_id,
                    track_start=right_lane.track_start,
                ),
            ),
        )
        site_ids = (site.site_id,)
        option_id = canonical_pattern_id(
            "pattern-landing-option",
            {
                "contacts": [
                    {
                        "site_id": contact.site_id,
                        "track_start": contact.track_start,
                        "use_id": contact.use_id,
                    }
                    for contact in contacts
                ],
                "demand_id": demand_id,
                "site_ids": list(site_ids),
            },
        )
        options.append(
            _intern_landing_option(
                option_budget,
                reason="internal landing-option expansion",
                option_id=option_id,
                demand_id=demand_id,
                site_ids=site_ids,
                contacts=contacts,
            )
        )
    return (
        LandingDemand(
            demand_id=demand_id,
            terminal_provenance=terminal_provenance,
            allowed_site_ids=tuple(
                sorted({site_id for option in options for site_id in option.site_ids})
            ),
            required_multiplicity=multiplicity,
            selection_mode=(
                LANDING_SELECTION_FACTORIZED_EXACT_CARDINALITY
                if multiplicity > 1
                else LANDING_SELECTION_EXPLICIT_OPTION
            ),
        ),
        tuple(options),
    )


def _component_boundary_lanes(
    *,
    terminal: TerminalDemand,
    endpoint: EndpointIncidence,
    component_ref: str,
    net: str,
    ledger: ExplicitRoutingResourceLedger,
    incidence_index: LaneIncidenceIndex,
    budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None = None,
    failure_refs: list[str] | None = None,
    failure_stats: dict[str, int] | None = None,
) -> tuple[LaneRef, ...]:
    """Sparse exact join from one owned component to boundary-access lanes."""

    lane_by_choice = {
        lane.track_choice_ref: lane for lane in incidence_index.lanes
    }
    result: dict[str, LaneRef] = {}
    if lazy_landing_access is not None:
        result = {}
        landing_diagnostics: dict[str, int] = {}
        candidate_lanes = lazy_landing_access.component_lanes(
            component_ref, budget=budget
        )
        for lane in candidate_lanes:
            if lazy_landing_access.boundary(
                terminal,
                endpoint,
                lane,
                net=net,
                required_component_ref=component_ref,
                retain=False,
                diagnostics=landing_diagnostics,
                budget=budget,
            ):
                result[lane.lane_id] = lane
        if failure_stats is not None:
            failure_stats["candidate_lanes"] = (
                failure_stats.get("candidate_lanes", 0)
                + len(candidate_lanes)
            )
            failure_stats["joined_lanes"] = (
                failure_stats.get("joined_lanes", 0) + len(result)
            )
            for key in ("landing_points", "materialized_sites"):
                failure_stats[key] = (
                    failure_stats.get(key, 0)
                    + landing_diagnostics.get(key, 0)
                )
        if failure_refs is not None and not result:
            reject_counts = tuple(
                (key, landing_diagnostics.get(key, 0))
                for key in (
                    "not_in_signed_domain",
                    "off_grid",
                    "outside_envelope",
                    "obstacle_blocked",
                    "owner_rejected",
                    "collar_terminal_ambiguous",
                    "collar_outside_envelope",
                    "collar_obstacle_blocked",
                    "collar_owner_changed",
                )
                if landing_diagnostics.get(key, 0)
            )
            bounded_provenance = _bounded_diagnostic_provenance(
                terminal.provenance
            )
            failure_refs.append(
                "component_boundary_join="
                f"{bounded_provenance}:"
                f"candidate_lanes={len(candidate_lanes)}:"
                "landing_points="
                f"{landing_diagnostics.get('landing_points', 0)}:"
                "materialized_sites="
                f"{landing_diagnostics.get('materialized_sites', 0)}:"
                "joined_lanes=0"
            )
            for key, count in reject_counts:
                failure_refs.append(
                    "component_boundary_reject="
                    f"{bounded_provenance}:{key}={count}"
                )
            for key in sorted(
                item
                for item in landing_diagnostics
                if item.startswith("collar_detail_")
                and landing_diagnostics.get(item, 0)
            ):
                failure_refs.append(
                    "component_boundary_detail="
                    f"{bounded_provenance}:{key}="
                    f"{landing_diagnostics[key]}"
                )
        return tuple(result[key] for key in sorted(result))
    for site in ledger.landing_sites:
        budget.consume("owned-component boundary-site index join")
        choices = (
            incidence_index.site_side(site.site_id, "lower").crossing_track_choices
            + incidence_index.site_side(site.site_id, "upper").crossing_track_choices
        )
        for choice in choices:
            budget.consume("owned-component boundary-lane index join")
            lane = lane_by_choice.get(choice)
            if lane is None:
                raise PatternCompileInputError(
                    "site-side incidence names a missing exact lane"
                )
            if _site_connects_endpoint_to_lane(
                site=site,
                terminal=terminal,
                endpoint=endpoint,
                lane=lane,
                incidence_index=incidence_index,
                net=net,
                required_component_ref=component_ref,
            ):
                result[lane.lane_id] = lane
    return tuple(result[key] for key in sorted(result))


def _lane_turn_neighbors(
    *,
    net: str,
    ledger: ExplicitRoutingResourceLedger,
    incidence_index: LaneIncidenceIndex,
    budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None = None,
) -> dict[str, tuple[LaneRef, ...]]:
    """Index only lane pairs joined by an exact adjacent-layer landing site."""

    lane_by_choice = {
        lane.track_choice_ref: lane for lane in incidence_index.lanes
    }
    neighbors: dict[str, dict[str, LaneRef]] = {}
    if lazy_landing_access is not None:
        lanes = incidence_index.lanes
        for left_index, left_lane in enumerate(lanes):
            for right_lane in lanes[left_index + 1:]:
                if abs(left_lane.layer_n - right_lane.layer_n) != 1:
                    continue
                budget.consume("lazy turn-site lane-pair query")
                if not lazy_landing_access.internal(
                    left_lane, right_lane, net=net, retain=False
                ):
                    continue
                neighbors.setdefault(left_lane.lane_id, {})[
                    right_lane.lane_id
                ] = right_lane
                neighbors.setdefault(right_lane.lane_id, {})[
                    left_lane.lane_id
                ] = left_lane
        return {
            lane_id: tuple(by_id[key] for key in sorted(by_id))
            for lane_id, by_id in neighbors.items()
        }
    for site in ledger.landing_sites:
        budget.consume("turn-site sparse index join")
        lower = incidence_index.site_side(site.site_id, "lower")
        upper = incidence_index.site_side(site.site_id, "upper")
        for lower_choice in lower.crossing_track_choices:
            for upper_choice in upper.crossing_track_choices:
                budget.consume("turn-site lane-pair sparse index join")
                lower_lane = lane_by_choice.get(lower_choice)
                upper_lane = lane_by_choice.get(upper_choice)
                if lower_lane is None or upper_lane is None:
                    raise PatternCompileInputError(
                        "site-side incidence names a missing exact lane"
                    )
                if not _site_joins_lanes(
                    site=site,
                    left_lane=lower_lane,
                    right_lane=upper_lane,
                    incidence_index=incidence_index,
                    net=net,
                ):
                    continue
                neighbors.setdefault(lower_lane.lane_id, {})[
                    upper_lane.lane_id
                ] = upper_lane
                neighbors.setdefault(upper_lane.lane_id, {})[
                    lower_lane.lane_id
                ] = lower_lane
    return {
        lane_id: tuple(by_id[key] for key in sorted(by_id))
        for lane_id, by_id in neighbors.items()
    }


def _exact_lazy_dogleg_turn_neighbors(
    *,
    current_lane: LaneRef,
    candidate_lanes: Sequence[LaneRef],
    net: str,
    required_multiplicity: int,
    budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess,
) -> tuple[LaneRef, ...]:
    """Close one dogleg turn from sparse geometry and exact landing facts."""

    by_layer: dict[int, dict[str, LaneRef]] = {}
    for candidate in candidate_lanes:
        if (
            abs(candidate.layer_n - current_lane.layer_n) != 1
            or candidate.preferred_axis == current_lane.preferred_axis
            or not _lanes_geometrically_intersect(current_lane, candidate)
        ):
            continue
        by_layer.setdefault(candidate.layer_n, {})[
            candidate.lane_id
        ] = candidate
    selection_limit = max(1, required_multiplicity)
    proved: dict[str, LaneRef] = {}
    for layer_n in sorted(by_layer):
        candidates = tuple(
            by_layer[layer_n][lane_id]
            for lane_id in sorted(by_layer[layer_n])
        )
        for candidate, _sites in lazy_landing_access.staircase_neighbors(
            current_lane,
            candidates,
            net=net,
            selection_limit=selection_limit,
            budget=budget,
        ):
            proved[candidate.lane_id] = candidate
    return tuple(proved[lane_id] for lane_id in sorted(proved))


class _ExactDoglegFamilyEdgeCover:
    """Retain an exact S+T-1 witness cover per compatible middle family.

    A merged elevated-dogleg family has one shared middle CorridorUse.  Its
    landing contacts bind both turns to the selected middle track, so every
    source edge can combine with every target edge carrying that same track and
    the same three exact run intervals.  One successful member per edge is
    therefore a complete exact witness for that factorized family;
    materializing its Cartesian product adds no IR fact.  Callers must include
    the interval tuple in ``family_key`` because the current CorridorUse schema
    does not encode per-choice intervals.
    """

    def __init__(self) -> None:
        self._covered: dict[Any, tuple[set[Any], set[Any]]] = {}

    def requires_member(
        self,
        family_key: Any,
        source_edge: Any,
        target_edge: Any,
    ) -> bool:
        source_edges, target_edges = self._covered.get(
            family_key, (set(), set())
        )
        return source_edge not in source_edges or target_edge not in target_edges

    def record_member(
        self,
        family_key: Any,
        source_edge: Any,
        target_edge: Any,
    ) -> None:
        source_edges, target_edges = self._covered.setdefault(
            family_key, (set(), set())
        )
        source_edges.add(source_edge)
        target_edges.add(target_edge)

    def covered_edges(
        self, family_key: Any
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        source_edges, target_edges = self._covered.get(
            family_key, (set(), set())
        )
        return (
            tuple(sorted(source_edges, key=repr)),
            tuple(sorted(target_edges, key=repr)),
        )


def _compile_dogleg_candidate(
    *,
    topology_candidate: TopologyCandidate,
    demand: RoutingDemand,
    routing_world: RoutingWorldProvenance,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    incidence_index: LaneIncidenceIndex,
    layer_n_by_key: dict[LayerKey, int],
    tech: Any,
    topology_digest: str,
    budget: _IncidenceBudget,
    option_budget: _IncidenceBudget,
    lazy_landing_access: _ExactLazyLandingAccess | None = None,
    lane_layer_index: _BudgetedLaneLayerIndex | None = None,
) -> tuple[CompleteNetCandidate, ...]:
    endpoint_rows = _resolved_endpoint_rows(
        topology_candidate=topology_candidate,
        demand=demand,
        routing_world=routing_world,
        incidence_index=incidence_index,
        layer_n_by_key=layer_n_by_key,
    )
    if not endpoint_rows:
        return ()
    oriented = _orient_two_pin_branch(endpoint_rows, demand)
    if oriented is None:
        return ()
    source, target = oriented
    allowed_layers = frozenset(
        topology_candidate.layers + topology_candidate.transport_layers
    )
    source_lanes = tuple(
        lane
        for lane in incidence_index.lanes_at_point(
            source[0].layer, source[0].xy_nm
        )
        if lane.preferred_axis == "H"
    )
    target_lanes = tuple(
        lane
        for lane in incidence_index.lanes_at_point(
            target[0].layer, target[0].xy_nm
        )
        if lane.preferred_axis == "H"
    )
    middle_lanes = tuple(
        lane for lane in incidence_index.lanes if lane.preferred_axis == "V"
    )
    target_lanes_by_layer: dict[int, tuple[LaneRef, ...]] = {
        layer_n: tuple(sorted(
            (
                lane
                for lane in target_lanes
                if lane.layer_n == layer_n and lane.layer_n in allowed_layers
            ),
            key=lambda lane: lane.lane_id,
        ))
        for layer_n in {lane.layer_n for lane in target_lanes}
    }
    candidates: list[CompleteNetCandidate] = []
    for source_lane in sorted(source_lanes, key=lambda lane: lane.lane_id):
        if source_lane.layer_n not in allowed_layers:
            continue
        compatible_targets = tuple(
            lane
            for lane in target_lanes_by_layer.get(source_lane.layer_n, ())
            if lane.lane_id != source_lane.lane_id
        )
        if not compatible_targets:
            continue
        if lane_layer_index is None:
            geometric_middle = tuple(
                lane
                for lane in middle_lanes
                if lane.layer_n in allowed_layers
                and lane.layer_n != source_lane.layer_n
                and _lanes_geometrically_intersect(source_lane, lane)
            )
        else:
            middle_by_id: dict[str, LaneRef] = {}
            for layer_n in sorted(allowed_layers):
                if layer_n == source_lane.layer_n:
                    continue
                for lane in lane_layer_index.intersecting(
                    source_lane, layer_n, budget=budget
                ):
                    if lane.preferred_axis == "V":
                        middle_by_id[lane.lane_id] = lane
            geometric_middle = tuple(
                middle_by_id[lane_id] for lane_id in sorted(middle_by_id)
            )
        for middle_lane in geometric_middle:
            for target_lane in compatible_targets:
                if not _lanes_geometrically_intersect(middle_lane, target_lane):
                    continue
                source_site_xy = (middle_lane.center_nm, source_lane.center_nm)
                target_site_xy = (middle_lane.center_nm, target_lane.center_nm)
                if not (
                    source_lane.contains_point(source_site_xy)
                    and middle_lane.contains_point(source_site_xy)
                    and middle_lane.contains_point(target_site_xy)
                    and target_lane.contains_point(target_site_xy)
                ):
                    continue
                source_interval = tuple(sorted((
                    _lane_run_coordinate(source_lane, source[0].xy_nm),
                    _lane_run_coordinate(source_lane, source_site_xy),
                )))
                middle_interval = tuple(sorted((
                    _lane_run_coordinate(middle_lane, source_site_xy),
                    _lane_run_coordinate(middle_lane, target_site_xy),
                )))
                target_interval = tuple(sorted((
                    _lane_run_coordinate(target_lane, target_site_xy),
                    _lane_run_coordinate(target_lane, target[0].xy_nm),
                )))
                source_interval = (source_interval[0], source_interval[1] + 1)
                middle_interval = (middle_interval[0], middle_interval[1] + 1)
                target_interval = (target_interval[0], target_interval[1] + 1)
                if any(interval[1] <= interval[0] for interval in (
                    source_interval,
                    middle_interval,
                    target_interval,
                )):
                    continue
                budget.consume("two-pin dogleg H-V-H expansion")
                source_use = _compiled_corridor_use(
                    lane=source_lane,
                    interval_nm=source_interval,
                    net=demand.net,
                    pattern_kind="DOGLEG_HVH",
                    ordinal=0,
                    width_nm=max(
                        topology_candidate.width_nm,
                        _tech_min_width_nm(tech, source_lane.layer_n),
                    ),
                )
                middle_use = _compiled_corridor_use(
                    lane=middle_lane,
                    interval_nm=middle_interval,
                    net=demand.net,
                    pattern_kind="DOGLEG_HVH",
                    ordinal=1,
                    width_nm=max(
                        topology_candidate.width_nm,
                        _tech_min_width_nm(tech, middle_lane.layer_n),
                    ),
                )
                target_use = _compiled_corridor_use(
                    lane=target_lane,
                    interval_nm=target_interval,
                    net=demand.net,
                    pattern_kind="DOGLEG_HVH",
                    ordinal=2,
                    width_nm=max(
                        topology_candidate.width_nm,
                        _tech_min_width_nm(tech, target_lane.layer_n),
                    ),
                )
                if (
                    source_use is None
                    or middle_use is None
                    or target_use is None
                ):
                    continue
                source_bundle = _internal_landing_bundle(
                    net=demand.net,
                    terminal_provenance=source[0].provenance,
                    left_lane=source_lane,
                    left_use=source_use,
                    right_lane=middle_lane,
                    right_use=middle_use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=topology_candidate.via_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    lazy_landing_access=lazy_landing_access,
                )
                target_bundle = _internal_landing_bundle(
                    net=demand.net,
                    terminal_provenance=target[0].provenance,
                    left_lane=middle_lane,
                    left_use=middle_use,
                    right_lane=target_lane,
                    right_use=target_use,
                    ledger=ledger,
                    incidence_index=incidence_index,
                    required_multiplicity=topology_candidate.via_multiplicity,
                    budget=budget,
                    option_budget=option_budget,
                    lazy_landing_access=lazy_landing_access,
                )
                if source_bundle is None or target_bundle is None:
                    continue
                source_demand, source_options = source_bundle
                target_demand, target_options = target_bundle
                branch_id = canonical_pattern_id(
                    "pattern-branch",
                    {
                        "net": demand.net,
                        "pattern_kind": "DOGLEG_HVH",
                        "source": source[0].provenance,
                        "source_demand": source_demand.demand_id,
                        "target": target[0].provenance,
                        "target_demand": target_demand.demand_id,
                        "use_ids": [
                            source_use.use_id,
                            middle_use.use_id,
                            target_use.use_id,
                        ],
                    },
                )
                candidate = CompleteNetCandidate(
                    candidate_id="pending-content-address",
                    net=demand.net,
                    topology_class=topology_candidate.topology_class,
                    branch_edges=((
                        source[0].provenance,
                        target[0].provenance,
                    ),),
                    corridor_uses=(source_use, middle_use, target_use),
                    landing_demands=(source_demand, target_demand),
                    required_goal_refs=topology_candidate.required_goal_refs,
                    width_nm=topology_candidate.width_nm,
                    via_multiplicity=topology_candidate.via_multiplicity,
                    source_topology_digest=topology_digest,
                    branches=(
                        CandidateBranch(
                            branch_id=branch_id,
                            source_endpoint=source[0].provenance,
                            target_endpoint=target[0].provenance,
                            ordered_use_ids=(
                                source_use.use_id,
                                middle_use.use_id,
                                target_use.use_id,
                            ),
                            ordered_landing_demand_ids=(
                                source_demand.demand_id,
                                target_demand.demand_id,
                            ),
                            ordered_steps=(
                                CandidateBranchStep(
                                    "track_use", source_use.use_id
                                ),
                                CandidateBranchStep(
                                    "landing_demand", source_demand.demand_id
                                ),
                                CandidateBranchStep(
                                    "track_use", middle_use.use_id
                                ),
                                CandidateBranchStep(
                                    "landing_demand", target_demand.demand_id
                                ),
                                CandidateBranchStep(
                                    "track_use", target_use.use_id
                                ),
                            ),
                        ),
                    ),
                    landing_options=source_options + target_options,
                    goal_branch_coverage=tuple(
                        GoalBranchCoverage(
                            declaration_ref=goal.declaration_ref,
                            branch_id=branch_id,
                            target_provenance=target[0].provenance,
                        )
                        for goal in demand.required_goals
                    ),
                    endpoint_accesses=(source[2], target[2]),
                )
                candidates.append(
                    replace(
                        candidate,
                        candidate_id=canonical_pattern_id(
                            "fixed-pattern-candidate",
                            _candidate_identity_payload(candidate),
                        ),
                    )
                )

    # A device terminal need not sit on a global lane.  The exact owned
    # conductor component may carry it to a boundary via, after which two
    # preferred-axis routing layers form the dogleg.  This is deliberately an
    # incidence join over certified components, lanes, and landing sites; it
    # does not synthesize a jog or assume that a geometric intersection is a
    # legal via.
    source_components = _endpoint_conductor_components(
        source[1], conductor_index=conductor_index, net=demand.net
    )
    target_components = _endpoint_conductor_components(
        target[1], conductor_index=conductor_index, net=demand.net
    )
    if not source_components or not target_components:
        return tuple(candidates)
    turn_neighbors = (
        {}
        if lazy_landing_access is not None
        else _lane_turn_neighbors(
            net=demand.net,
            ledger=ledger,
            incidence_index=incidence_index,
            budget=budget,
        )
    )
    lazy_middle_lanes_by_source: dict[str, tuple[LaneRef, ...]] = {}
    lazy_target_lanes_by_frontier: dict[
        tuple[str, tuple[str, ...]], tuple[LaneRef, ...]
    ] = {}
    elevated_edge_cover = _ExactDoglegFamilyEdgeCover()
    for source_ref, source_component in source_components:
        source_access_lanes = _component_boundary_lanes(
            terminal=source[0],
            endpoint=source[1],
            component_ref=source_ref,
            net=demand.net,
            ledger=ledger,
            incidence_index=incidence_index,
            budget=budget,
            lazy_landing_access=lazy_landing_access,
        )
        for target_ref, target_component in target_components:
            if source_ref == target_ref:
                continue
            target_access_lanes = _component_boundary_lanes(
                terminal=target[0],
                endpoint=target[1],
                component_ref=target_ref,
                net=demand.net,
                ledger=ledger,
                incidence_index=incidence_index,
                budget=budget,
                lazy_landing_access=lazy_landing_access,
            )
            target_access_by_id = {
                lane.lane_id: lane for lane in target_access_lanes
            }
            for source_lane in source_access_lanes:
                if lazy_landing_access is not None:
                    if lane_layer_index is None:
                        raise PatternCompileInputError(
                            "lazy dogleg compilation requires lane layer index"
                        )
                    middle_lanes = lazy_middle_lanes_by_source.get(
                        source_lane.lane_id
                    )
                    if middle_lanes is None:
                        geometric_middle: dict[str, LaneRef] = {}
                        for layer_n in sorted(allowed_layers):
                            if abs(layer_n - source_lane.layer_n) != 1:
                                continue
                            for lane in lane_layer_index.intersecting(
                                source_lane, layer_n, budget=budget
                            ):
                                geometric_middle[lane.lane_id] = lane
                        middle_lanes = _exact_lazy_dogleg_turn_neighbors(
                            current_lane=source_lane,
                            candidate_lanes=tuple(
                                geometric_middle[lane_id]
                                for lane_id in sorted(geometric_middle)
                            ),
                            net=demand.net,
                            required_multiplicity=(
                                topology_candidate.via_multiplicity
                            ),
                            budget=budget,
                            lazy_landing_access=lazy_landing_access,
                        )
                        lazy_middle_lanes_by_source[
                            source_lane.lane_id
                        ] = middle_lanes
                else:
                    middle_lanes = turn_neighbors.get(source_lane.lane_id, ())
                for middle_lane in middle_lanes:
                    if lazy_landing_access is not None:
                        possible_targets = tuple(
                            lane
                            for lane in target_access_lanes
                            if lane.layer_n == source_lane.layer_n
                            and lane.preferred_axis
                            == source_lane.preferred_axis
                        )
                        target_cache_key = (
                            middle_lane.lane_id,
                            tuple(
                                sorted(lane.lane_id for lane in possible_targets)
                            ),
                        )
                        target_lanes = lazy_target_lanes_by_frontier.get(
                            target_cache_key
                        )
                        if target_lanes is None:
                            target_lanes = _exact_lazy_dogleg_turn_neighbors(
                                current_lane=middle_lane,
                                candidate_lanes=possible_targets,
                                net=demand.net,
                                required_multiplicity=(
                                    topology_candidate.via_multiplicity
                                ),
                                budget=budget,
                                lazy_landing_access=lazy_landing_access,
                            )
                            lazy_target_lanes_by_frontier[
                                target_cache_key
                            ] = target_lanes
                    else:
                        target_lanes = tuple(
                            target_access_by_id[neighbor.lane_id]
                            for neighbor in turn_neighbors.get(
                                middle_lane.lane_id, ()
                            )
                            if neighbor.lane_id in target_access_by_id
                        )
                    for target_lane in target_lanes:
                        if source_lane.lane_id == target_lane.lane_id:
                            continue
                        if (
                            source_lane.layer_n != target_lane.layer_n
                            or source_lane.preferred_axis
                            != target_lane.preferred_axis
                            or middle_lane.layer_n == source_lane.layer_n
                            or middle_lane.preferred_axis
                            == source_lane.preferred_axis
                        ):
                            continue
                        if source_lane.gds_layer_key in {
                            source_component.layer_key,
                            target_component.layer_key,
                        } or target_lane.gds_layer_key in {
                            source_component.layer_key,
                            target_component.layer_key,
                        }:
                            continue
                        used_layer_ns = {
                            source_lane.layer_n,
                            middle_lane.layer_n,
                            target_lane.layer_n,
                        }
                        if not used_layer_ns.issubset(allowed_layers):
                            continue

                        family_key = (
                            demand.net,
                            topology_candidate.topology_class,
                            source[0].provenance,
                            target[0].provenance,
                            source_ref,
                            target_ref,
                            CorridorLayerKey(
                                source_lane.resource_corridor_id,
                                source_lane.layer_n,
                            ),
                            CorridorLayerKey(
                                middle_lane.resource_corridor_id,
                                middle_lane.layer_n,
                            ),
                            CorridorLayerKey(
                                target_lane.resource_corridor_id,
                                target_lane.layer_n,
                            ),
                            topology_candidate.required_goal_refs,
                            topology_candidate.width_nm,
                            topology_candidate.via_multiplicity,
                            topology_digest,
                        )
                        source_edge_key = (
                            source_lane.lane_id,
                            middle_lane.lane_id,
                        )
                        target_edge_key = (
                            middle_lane.lane_id,
                            target_lane.lane_id,
                        )
                        if source_lane.preferred_axis == "H":
                            source_turn_xy = (
                                middle_lane.center_nm,
                                source_lane.center_nm,
                            )
                            target_turn_xy = (
                                middle_lane.center_nm,
                                target_lane.center_nm,
                            )
                        else:
                            source_turn_xy = (
                                source_lane.center_nm,
                                middle_lane.center_nm,
                            )
                            target_turn_xy = (
                                target_lane.center_nm,
                                middle_lane.center_nm,
                            )
                        if not (
                            source_lane.contains_point(source_turn_xy)
                            and middle_lane.contains_point(source_turn_xy)
                            and middle_lane.contains_point(target_turn_xy)
                            and target_lane.contains_point(target_turn_xy)
                        ):
                            continue

                        source_bounds = _component_run_bounds(
                            source_lane, source_component
                        )
                        target_bounds = _component_run_bounds(
                            target_lane, target_component
                        )
                        if source_bounds is None or target_bounds is None:
                            continue
                        source_turn_run = _lane_run_coordinate(
                            source_lane, source_turn_xy
                        )
                        target_turn_run = _lane_run_coordinate(
                            target_lane, target_turn_xy
                        )
                        middle_runs = (
                            _lane_run_coordinate(middle_lane, source_turn_xy),
                            _lane_run_coordinate(middle_lane, target_turn_xy),
                        )
                        source_interval = (
                            min(source_bounds[0], source_turn_run),
                            min(
                                source_lane.through_interval_nm[1],
                                max(source_bounds[1], source_turn_run) + 1,
                            ),
                        )
                        target_interval = (
                            min(target_bounds[0], target_turn_run),
                            min(
                                target_lane.through_interval_nm[1],
                                max(target_bounds[1], target_turn_run) + 1,
                            ),
                        )
                        middle_interval = (
                            min(middle_runs),
                            min(
                                middle_lane.through_interval_nm[1],
                                max(middle_runs) + 1,
                            ),
                        )
                        if any(
                            interval[1] <= interval[0]
                            for interval in (
                                source_interval,
                                middle_interval,
                                target_interval,
                            )
                        ):
                            continue
                        exact_family_key = family_key + (
                            source_interval,
                            middle_interval,
                            target_interval,
                        )
                        if not elevated_edge_cover.requires_member(
                            exact_family_key,
                            source_edge_key,
                            target_edge_key,
                        ):
                            continue

                        pattern_kind = "DOGLEG_OWNED_ELEVATED"
                        source_use = _compiled_corridor_use(
                            lane=source_lane,
                            interval_nm=source_interval,
                            net=demand.net,
                            pattern_kind=pattern_kind,
                            ordinal=0,
                            width_nm=max(
                                topology_candidate.width_nm,
                                _tech_min_width_nm(tech, source_lane.layer_n),
                            ),
                        )
                        middle_use = _compiled_corridor_use(
                            lane=middle_lane,
                            interval_nm=middle_interval,
                            net=demand.net,
                            pattern_kind=pattern_kind,
                            ordinal=1,
                            width_nm=max(
                                topology_candidate.width_nm,
                                _tech_min_width_nm(tech, middle_lane.layer_n),
                            ),
                        )
                        target_use = _compiled_corridor_use(
                            lane=target_lane,
                            interval_nm=target_interval,
                            net=demand.net,
                            pattern_kind=pattern_kind,
                            ordinal=2,
                            width_nm=max(
                                topology_candidate.width_nm,
                                _tech_min_width_nm(tech, target_lane.layer_n),
                            ),
                        )
                        if (
                            source_use is None
                            or middle_use is None
                            or target_use is None
                        ):
                            continue
                        raw_before_witness = option_budget.used
                        expansion_before_witness = budget.used
                        source_boundary = _landing_bundle(
                            net=demand.net,
                            terminal=source[0],
                            endpoint=source[1],
                            lane=source_lane,
                            use=source_use,
                            ledger=ledger,
                            incidence_index=incidence_index,
                            required_multiplicity=(
                                topology_candidate.via_multiplicity
                            ),
                            budget=budget,
                            option_budget=option_budget,
                            lazy_landing_access=lazy_landing_access,
                            required_component_ref=source_ref,
                        )
                        source_turn = _internal_landing_bundle(
                            net=demand.net,
                            terminal_provenance=source[0].provenance,
                            left_lane=source_lane,
                            left_use=source_use,
                            right_lane=middle_lane,
                            right_use=middle_use,
                            ledger=ledger,
                            incidence_index=incidence_index,
                            required_multiplicity=(
                                topology_candidate.via_multiplicity
                            ),
                            budget=budget,
                            option_budget=option_budget,
                            lazy_landing_access=lazy_landing_access,
                        )
                        target_turn = _internal_landing_bundle(
                            net=demand.net,
                            terminal_provenance=target[0].provenance,
                            left_lane=middle_lane,
                            left_use=middle_use,
                            right_lane=target_lane,
                            right_use=target_use,
                            ledger=ledger,
                            incidence_index=incidence_index,
                            required_multiplicity=(
                                topology_candidate.via_multiplicity
                            ),
                            budget=budget,
                            option_budget=option_budget,
                            lazy_landing_access=lazy_landing_access,
                        )
                        target_boundary = _landing_bundle(
                            net=demand.net,
                            terminal=target[0],
                            endpoint=target[1],
                            lane=target_lane,
                            use=target_use,
                            ledger=ledger,
                            incidence_index=incidence_index,
                            required_multiplicity=(
                                topology_candidate.via_multiplicity
                            ),
                            budget=budget,
                            option_budget=option_budget,
                            required_component_ref=target_ref,
                            lazy_landing_access=lazy_landing_access,
                        )
                        if any(
                            bundle is None
                            for bundle in (
                                source_boundary,
                                source_turn,
                                target_turn,
                                target_boundary,
                            )
                        ):
                            continue
                        source_boundary_demand, source_boundary_options = (
                            source_boundary
                        )
                        source_turn_demand, source_turn_options = source_turn
                        target_turn_demand, target_turn_options = target_turn
                        target_boundary_demand, target_boundary_options = (
                            target_boundary
                        )
                        landing_demands = (
                            source_boundary_demand,
                            source_turn_demand,
                            target_turn_demand,
                            target_boundary_demand,
                        )
                        branch_id = canonical_pattern_id(
                            "pattern-branch",
                            {
                                "landing_demand_ids": [
                                    item.demand_id for item in landing_demands
                                ],
                                "net": demand.net,
                                "pattern_kind": pattern_kind,
                                "source": source[0].provenance,
                                "source_component": source_ref,
                                "target": target[0].provenance,
                                "target_component": target_ref,
                                "use_ids": [
                                    source_use.use_id,
                                    middle_use.use_id,
                                    target_use.use_id,
                                ],
                            },
                        )
                        candidate = CompleteNetCandidate(
                            candidate_id="pending-content-address",
                            net=demand.net,
                            topology_class=topology_candidate.topology_class,
                            branch_edges=((
                                source[0].provenance,
                                target[0].provenance,
                            ),),
                            corridor_uses=(
                                source_use,
                                middle_use,
                                target_use,
                            ),
                            landing_demands=landing_demands,
                            required_goal_refs=(
                                topology_candidate.required_goal_refs
                            ),
                            width_nm=topology_candidate.width_nm,
                            via_multiplicity=(
                                topology_candidate.via_multiplicity
                            ),
                            source_topology_digest=topology_digest,
                            program_kind="PATTERN_TREE",
                            branches=(
                                CandidateBranch(
                                    branch_id=branch_id,
                                    source_endpoint=source[0].provenance,
                                    target_endpoint=target[0].provenance,
                                    ordered_use_ids=(
                                        source_use.use_id,
                                        middle_use.use_id,
                                        target_use.use_id,
                                    ),
                                    ordered_landing_demand_ids=tuple(
                                        item.demand_id
                                        for item in landing_demands
                                    ),
                                    ordered_steps=(
                                        CandidateBranchStep(
                                            "owned_component", source_ref
                                        ),
                                        CandidateBranchStep(
                                            "landing_demand",
                                            source_boundary_demand.demand_id,
                                        ),
                                        CandidateBranchStep(
                                            "track_use", source_use.use_id
                                        ),
                                        CandidateBranchStep(
                                            "landing_demand",
                                            source_turn_demand.demand_id,
                                        ),
                                        CandidateBranchStep(
                                            "track_use", middle_use.use_id
                                        ),
                                        CandidateBranchStep(
                                            "landing_demand",
                                            target_turn_demand.demand_id,
                                        ),
                                        CandidateBranchStep(
                                            "track_use", target_use.use_id
                                        ),
                                        CandidateBranchStep(
                                            "landing_demand",
                                            target_boundary_demand.demand_id,
                                        ),
                                        CandidateBranchStep(
                                            "owned_component", target_ref
                                        ),
                                    ),
                                ),
                            ),
                            landing_options=(
                                source_boundary_options
                                + source_turn_options
                                + target_turn_options
                                + target_boundary_options
                            ),
                            goal_branch_coverage=tuple(
                                GoalBranchCoverage(
                                    declaration_ref=goal.declaration_ref,
                                    branch_id=branch_id,
                                    target_provenance=target[0].provenance,
                                )
                                for goal in demand.required_goals
                            ),
                            endpoint_accesses=(source[2], target[2]),
                            owned_component_use_refs=(source_ref, target_ref),
                        )
                        _account_elevated_dogleg_witness(
                            budget,
                            option_budget,
                            raw_before=raw_before_witness,
                            expansion_before=expansion_before_witness,
                        )
                        candidates.append(_content_address_candidate(candidate))
                        elevated_edge_cover.record_member(
                            exact_family_key,
                            source_edge_key,
                            target_edge_key,
                        )
    return _merge_elevated_dogleg_candidate_families(
        tuple(candidates),
        ledger=ledger,
    )


def _merge_elevated_dogleg_candidate_families(
    candidates: tuple[CompleteNetCandidate, ...],
    *,
    ledger: ExplicitRoutingResourceLedger,
) -> tuple[CompleteNetCandidate, ...]:
    """Lift per-track elevated doglegs into corridor-level CP candidates.

    The physical compiler first proves concrete lane/site paths.  Variants that
    differ only in track starts inside the same three TrackResources are one
    corridor topology, not thousands of complete-net topologies.  Landing
    options already bind every via to exact track choices, so merging those
    options preserves every concrete path and lets the simultaneous allocator
    choose the starts.  Only members with identical per-use run intervals may
    share a family because CorridorUse carries one exact interval rather than a
    per-choice interval map.
    """

    resources_by_key = {
        resource.key: resource for resource in ledger.track_resources
    }
    for candidate in candidates:
        _assert_candidate_use_width_fits(
            candidate,
            resources_by_key=resources_by_key,
        )

    passthrough: list[CompleteNetCandidate] = []
    families: dict[tuple[Any, ...], list[CompleteNetCandidate]] = {}
    for candidate in candidates:
        if (
            candidate.program_kind != "PATTERN_TREE"
            or len(candidate.branches) != 1
            or len(candidate.corridor_uses) != 3
            or len(candidate.landing_demands) != 4
            or len(candidate.owned_component_use_refs) != 2
        ):
            passthrough.append(candidate)
            continue
        branch = candidate.branches[0]
        uses_by_id = {use.use_id: use for use in candidate.corridor_uses}
        ordered_uses = tuple(uses_by_id[use_id] for use_id in branch.ordered_use_ids)
        if len(ordered_uses) != 3:
            passthrough.append(candidate)
            continue
        family_key = (
            candidate.net,
            candidate.topology_class,
            branch.source_endpoint,
            branch.target_endpoint,
            candidate.owned_component_use_refs,
            tuple(use.resource for use in ordered_uses),
            tuple(use.interval_nm for use in ordered_uses),
            tuple(use.track_span for use in ordered_uses),
            tuple(use.width_nm for use in ordered_uses),
            candidate.required_goal_refs,
            candidate.width_nm,
            candidate.via_multiplicity,
            candidate.source_topology_digest,
            candidate.endpoint_accesses,
        )
        families.setdefault(family_key, []).append(candidate)

    merged = list(passthrough)
    work_items = []
    for family_key in sorted(families, key=repr):
        ordered_family = tuple(sorted(
            families[family_key], key=lambda item: item.candidate_id
        ))
        chunks: list[list[CompleteNetCandidate]] = []
        chunk_option_count = 0
        for member in ordered_family:
            member_option_count = len(member.landing_options)
            if (
                chunks
                and chunks[-1]
                and chunk_option_count + member_option_count
                > MAX_TOTAL_LANDING_OPTIONS
            ):
                chunks.append([])
                chunk_option_count = 0
            if not chunks:
                chunks.append([])
            chunks[-1].append(member)
            chunk_option_count += member_option_count
        work_items.extend(
            (family_key + (chunk_index,), tuple(chunk))
            for chunk_index, chunk in enumerate(chunks)
        )
    for family_key, members in work_items:
        if len(members) == 1:
            merged.append(members[0])
            continue
        exemplar = members[0]
        exemplar_branch = exemplar.branches[0]
        ordered_member_uses = []
        ordered_member_demands = []
        for member in members:
            branch = member.branches[0]
            uses_by_id = {use.use_id: use for use in member.corridor_uses}
            demands_by_id = {
                demand.demand_id: demand for demand in member.landing_demands
            }
            ordered_member_uses.append(tuple(
                uses_by_id[use_id] for use_id in branch.ordered_use_ids
            ))
            ordered_member_demands.append(tuple(
                demands_by_id[demand_id]
                for demand_id in branch.ordered_landing_demand_ids
            ))

        group_ref = canonical_pattern_hash({
            "candidate_ids": [member.candidate_id for member in members],
            "schema": "clay.elevated-dogleg-corridor-family.v1",
        })
        new_uses = []
        for ordinal in range(3):
            variants = tuple(row[ordinal] for row in ordered_member_uses)
            interval = variants[0].interval_nm
            if any(use.interval_nm != interval for use in variants):
                raise CandidatePoolError(
                    "elevated dogleg family crossed corridor-use intervals"
                )
            starts = tuple(sorted({
                start
                for use in variants
                for start in use.allowed_track_starts
            }))
            payload = {
                "family": group_ref,
                "interval_nm": list(interval),
                "ordinal": ordinal,
                "resource": {
                    "corridor_id": variants[0].resource.corridor_id,
                    "layer_n": variants[0].resource.layer_n,
                },
                "track_span": variants[0].track_span,
                "track_starts": list(starts),
            }
            new_uses.append(CorridorUse(
                use_id=canonical_pattern_id("pattern-use-family", payload),
                resource=variants[0].resource,
                interval_nm=interval,
                track_span=variants[0].track_span,
                allowed_track_starts=starts,
                width_nm=variants[0].width_nm,
            ))

        new_demand_ids = tuple(
            canonical_pattern_id(
                "pattern-landing-demand-family",
                {"family": group_ref, "ordinal": ordinal},
            )
            for ordinal in range(4)
        )
        converted_options: dict[str, LandingOption] = {}
        options_by_demand: dict[str, list[LandingOption]] = {
            demand_id: [] for demand_id in new_demand_ids
        }
        for member_index, member in enumerate(members):
            branch = member.branches[0]
            old_use_position = {
                use_id: ordinal
                for ordinal, use_id in enumerate(branch.ordered_use_ids)
            }
            old_demand_position = {
                demand_id: ordinal
                for ordinal, demand_id in enumerate(
                    branch.ordered_landing_demand_ids
                )
            }
            for option in member.landing_options:
                demand_ordinal = old_demand_position[option.demand_id]
                demand_id = new_demand_ids[demand_ordinal]
                contacts = tuple(
                    LandingTrackContact(
                        site_id=contact.site_id,
                        track_choice=TrackChoiceRef(
                            use_id=new_uses[
                                old_use_position[contact.use_id]
                            ].use_id,
                            track_start=contact.track_start,
                        ),
                    )
                    for contact in option.contacts
                )
                option_id = canonical_pattern_id(
                    "pattern-landing-option-family",
                    {
                        "contacts": [
                            {
                                "site_id": contact.site_id,
                                "track_start": contact.track_start,
                                "use_id": contact.use_id,
                            }
                            for contact in contacts
                        ],
                        "demand_id": demand_id,
                        "site_ids": list(option.site_ids),
                    },
                )
                converted = LandingOption(
                    option_id=option_id,
                    demand_id=demand_id,
                    site_ids=option.site_ids,
                    contacts=contacts,
                )
                converted_options[option_id] = converted
                options_by_demand[demand_id].append(converted)

        new_demands = []
        for ordinal, demand_id in enumerate(new_demand_ids):
            variants = tuple(row[ordinal] for row in ordered_member_demands)
            role_options = tuple({
                option.option_id: option
                for option in options_by_demand[demand_id]
            }.values())
            new_demands.append(LandingDemand(
                demand_id=demand_id,
                terminal_provenance=variants[0].terminal_provenance,
                allowed_site_ids=tuple(sorted({
                    site_id
                    for option in role_options
                    for site_id in option.site_ids
                })),
                required_multiplicity=variants[0].required_multiplicity,
                declaration_ref=variants[0].declaration_ref,
                selection_mode=variants[0].selection_mode,
            ))

        old_use_position = {
            use_id: ordinal
            for ordinal, use_id in enumerate(exemplar_branch.ordered_use_ids)
        }
        old_demand_position = {
            demand_id: ordinal
            for ordinal, demand_id in enumerate(
                exemplar_branch.ordered_landing_demand_ids
            )
        }
        new_steps = tuple(
            CandidateBranchStep(
                step.kind,
                (
                    new_uses[old_use_position[step.ref_id]].use_id
                    if step.kind == "track_use"
                    else new_demand_ids[old_demand_position[step.ref_id]]
                    if step.kind == "landing_demand"
                    else step.ref_id
                ),
            )
            for step in exemplar_branch.ordered_steps
        )
        branch_id = canonical_pattern_id(
            "pattern-branch-family",
            {
                "family": group_ref,
                "steps": [
                    {"kind": step.kind, "ref_id": step.ref_id}
                    for step in new_steps
                ],
            },
        )
        new_branch = CandidateBranch(
            branch_id=branch_id,
            source_endpoint=exemplar_branch.source_endpoint,
            target_endpoint=exemplar_branch.target_endpoint,
            ordered_use_ids=tuple(use.use_id for use in new_uses),
            ordered_landing_demand_ids=new_demand_ids,
            ordered_steps=new_steps,
        )
        candidate = CompleteNetCandidate(
            candidate_id="pending-content-address",
            net=exemplar.net,
            topology_class=exemplar.topology_class,
            branch_edges=((
                new_branch.source_endpoint, new_branch.target_endpoint
            ),),
            corridor_uses=tuple(new_uses),
            landing_demands=tuple(new_demands),
            required_goal_refs=exemplar.required_goal_refs,
            width_nm=exemplar.width_nm,
            via_multiplicity=exemplar.via_multiplicity,
            source_topology_digest=exemplar.source_topology_digest,
            program_kind=exemplar.program_kind,
            branches=(new_branch,),
            landing_options=tuple(
                converted_options[key] for key in sorted(converted_options)
            ),
            goal_branch_coverage=tuple(
                GoalBranchCoverage(
                    declaration_ref=item.declaration_ref,
                    branch_id=branch_id,
                    target_provenance=item.target_provenance,
                )
                for item in exemplar.goal_branch_coverage
            ),
            endpoint_accesses=exemplar.endpoint_accesses,
            owned_component_use_refs=exemplar.owned_component_use_refs,
        )
        _assert_candidate_use_width_fits(
            candidate,
            resources_by_key=resources_by_key,
        )
        merged.append(_content_address_candidate(candidate))
    return tuple(sorted(merged, key=lambda item: item.candidate_id))


def _rebuild_topology_terminal(
    value: Any,
    *,
    label: str,
) -> TerminalDemand:
    if type(value) is not TerminalDemand:
        raise PatternCompileInputError(f"{label} must be exact TerminalDemand")
    try:
        return replace(
            value,
            xy_nm=_bounded_materialize(
                value.xy_nm,
                limit=2,
                label=f"{label} coordinate",
            ),
            layer=_bounded_materialize(
                value.layer,
                limit=2,
                label=f"{label} layer key",
            ),
        )
    except PatternCompileLimitError:
        raise
    except (TypeError, ValueError) as exc:
        raise PatternCompileInputError(f"{label} is malformed: {exc}") from exc


def _rebuild_topology_goal(value: Any) -> RequiredRouteGoal:
    if type(value) is not RequiredRouteGoal:
        raise PatternCompileInputError(
            "routing demand required goal must be exact RequiredRouteGoal"
        )
    target = (
        None
        if value.target is None
        else _rebuild_topology_terminal(
            value.target,
            label="required goal target",
        )
    )
    try:
        return replace(value, target=target)
    except (TypeError, ValueError) as exc:
        raise PatternCompileInputError(
            f"routing demand required goal is malformed: {exc}"
        ) from exc


def _rebuild_topology_demand(value: Any) -> RoutingDemand:
    if type(value) is not RoutingDemand:
        raise PatternCompileInputError(
            "topology plan demand must be exact RoutingDemand"
        )
    terminals = tuple(
        _rebuild_topology_terminal(
            terminal,
            label="routing demand terminal",
        )
        for terminal in _bounded_materialize(
            value.terminals,
            limit=MAX_TOPOLOGY_BRANCH_TERMINALS,
            label="routing demand terminal",
        )
    )
    goals = tuple(
        _rebuild_topology_goal(goal)
        for goal in _bounded_materialize(
            value.required_goals,
            limit=MAX_TOPOLOGY_REQUIRED_GOAL_REFS,
            label="routing demand required goal",
        )
    )
    try:
        return replace(
            value,
            terminals=terminals,
            required_goals=goals,
            declared_trunk_layers=_bounded_materialize(
                value.declared_trunk_layers,
                limit=MAX_TOPOLOGY_LAYERS,
                label="routing demand declared trunk layer",
            ),
            declared_allowed_layers=_bounded_materialize(
                value.declared_allowed_layers,
                limit=MAX_TOPOLOGY_LAYERS,
                label="routing demand declared allowed layer",
            ),
            declared_edge_trunks=_bounded_materialize(
                value.declared_edge_trunks,
                limit=MAX_TOPOLOGY_DECLARED_EDGE_TRUNKS,
                label="routing demand declared edge trunk",
            ),
            declared_power_plane_layers=_bounded_materialize(
                value.declared_power_plane_layers,
                limit=MAX_TOPOLOGY_LAYERS,
                label="routing demand declared power-plane layer",
            ),
        )
    except PatternCompileLimitError:
        raise
    except (TypeError, ValueError) as exc:
        raise PatternCompileInputError(
            f"topology plan routing demand is malformed: {exc}"
        ) from exc


def _rebuild_branch_target(value: Any) -> tuple[int, int]:
    try:
        point = _bounded_materialize(
            value,
            limit=2,
            label="topology candidate branch target coordinate",
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, PatternCompileLimitError):
            raise
        raise PatternCompileInputError(
            f"topology candidate branch target is malformed: {exc}"
        ) from exc
    return point  # type: ignore[return-value]


def _rebuild_topology_candidate(value: Any) -> TopologyCandidate:
    if type(value) is not TopologyCandidate:
        raise PatternCompileInputError(
            "topology plan candidate must be exact TopologyCandidate"
        )
    branch_targets = tuple(
        _rebuild_branch_target(point)
        for point in _bounded_materialize(
            value.branch_targets,
            limit=MAX_TOPOLOGY_BRANCH_TARGETS,
            label="topology candidate branch target",
        )
    )
    branch_terminals = tuple(
        _rebuild_topology_terminal(
            terminal,
            label="topology candidate branch terminal",
        )
        for terminal in _bounded_materialize(
            value.branch_terminals,
            limit=MAX_TOPOLOGY_BRANCH_TERMINALS,
            label="topology candidate branch terminal",
        )
    )
    try:
        return replace(
            value,
            branch_targets=branch_targets,
            layers=_bounded_materialize(
                value.layers,
                limit=MAX_TOPOLOGY_LAYERS,
                label="topology candidate layer",
            ),
            transport_layers=_bounded_materialize(
                value.transport_layers,
                limit=MAX_TOPOLOGY_LAYERS,
                label="topology candidate transport layer",
            ),
            branch_terminals=branch_terminals,
            required_goal_refs=_bounded_materialize(
                value.required_goal_refs,
                limit=MAX_TOPOLOGY_REQUIRED_GOAL_REFS,
                label="topology candidate required goal ref",
            ),
        )
    except PatternCompileLimitError:
        raise
    except (TypeError, ValueError) as exc:
        raise PatternCompileInputError(
            f"topology plan candidate is malformed: {exc}"
        ) from exc


def _factorization_landing_sites(
    members: Sequence[CompleteNetCandidate],
    available_sites: Mapping[str, Any],
    *,
    routing_world_provenance_digest: str,
    landing_site_domain_digest: str,
) -> tuple[Any, ...] | LandingSiteCatalog:
    site_id_set: set[str] = set()
    for candidate in members:
        for demand in candidate.landing_demands:
            for site_id in demand.allowed_site_ids:
                if not isinstance(site_id, str) or not site_id:
                    raise CandidatePoolError(
                        "factorization member domains require non-empty site ids"
                    )
                if site_id in site_id_set:
                    continue
                if len(site_id_set) >= MAX_TOTAL_ASSIGNMENT_ARCS:
                    raise CandidatePoolLimitError(
                        "factorization member site reference union exceeds "
                        f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
                    )
                site_id_set.add(site_id)
    site_ids = tuple(sorted(site_id_set))
    missing = tuple(
        site_id for site_id in site_ids if site_id not in available_sites
    )
    if missing:
        raise CandidatePoolError(
            f"factorization member domains name missing landing sites {missing}"
        )
    if len(site_ids) <= MAX_PATTERN_LANDING_SITE_PAGE_SIZE:
        return tuple(available_sites[site_id] for site_id in site_ids)
    pages = tuple(
        LandingSitePage.build(
            page_index,
            tuple(
                available_sites[site_id]
                for site_id in site_ids[
                    start:start + MAX_PATTERN_LANDING_SITE_PAGE_SIZE
                ]
            ),
        )
        for page_index, start in enumerate(range(
            0,
            len(site_ids),
            MAX_PATTERN_LANDING_SITE_PAGE_SIZE,
        ))
    )
    return LandingSiteCatalog.from_pages(
        pages,
        routing_world_provenance_digest=routing_world_provenance_digest,
        landing_site_domain_digest=landing_site_domain_digest,
    )


def _diagnostic_candidate_site_ids(
    candidates: Sequence[CompleteNetCandidate],
    factorized_families: Sequence[FactorizedCompleteNetCandidateFamily],
    *,
    max_site_ids: int = MAX_TOTAL_ASSIGNMENT_ARCS,
    deadline_guard: Callable[[str], None] | None = None,
) -> tuple[str, ...]:
    if type(max_site_ids) is not int or max_site_ids <= 0:
        raise CandidatePoolError(
            "diagnostic candidate site-id bound must be a positive integer"
        )
    guard = deadline_guard or (lambda _phase: None)
    site_ids: set[str] = set()

    def add(values: Any) -> None:
        for site_id in values:
            guard("diagnostic-candidate-site-ids")
            if type(site_id) is not str or not site_id:
                raise CandidatePoolError(
                    "diagnostic candidate site ids must be non-empty strings"
                )
            if site_id in site_ids:
                continue
            if len(site_ids) >= max_site_ids:
                raise CandidatePoolLimitError(
                    "diagnostic candidate site reference union exceeds "
                    f"{max_site_ids}"
                )
            site_ids.add(site_id)

    for candidate in candidates:
        for demand in candidate.landing_demands:
            add(demand.allowed_site_ids)
        for option in candidate.landing_options:
            add(option.site_ids)
    for family in factorized_families:
        for role in family.demand_roles:
            add(role.allowed_site_ids)
    guard("diagnostic-candidate-site-ids-complete")
    return tuple(sorted(site_ids))


def _diagnostic_site_projection_matches(
    expected_site_ids: tuple[str, ...],
    actual_site_ids: tuple[str, ...],
    *,
    complete_source_domain: bool,
) -> bool:
    """Check an induced subset or an explicitly declared complete-domain superset."""

    if (
        tuple(sorted(expected_site_ids)) != expected_site_ids
        or tuple(sorted(actual_site_ids)) != actual_site_ids
        or len(expected_site_ids) != len(set(expected_site_ids))
        or len(actual_site_ids) != len(set(actual_site_ids))
    ):
        return False
    if complete_source_domain:
        return set(expected_site_ids).issubset(actual_site_ids)
    return actual_site_ids == expected_site_ids


class _DiagnosticAggregateDeadlineExhausted(CandidatePoolLimitError):
    """The one diagnostic wall-clock deadline has expired."""


@dataclass(frozen=True)
class _DiagnosticAggregateDeadline:
    """One wall-clock deadline shared by every diagnostic replay/solve."""

    aggregate_seconds: float
    monotonic: Callable[[], float]
    started: float = field(init=False)
    deadline: float = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.aggregate_seconds, bool)
            or not isinstance(self.aggregate_seconds, (int, float))
            or float(self.aggregate_seconds) <= 0.0
            or not callable(self.monotonic)
        ):
            raise CandidatePoolLimitError(
                "diagnostic aggregate deadline must be a positive duration"
            )
        started = float(self.monotonic())
        object.__setattr__(self, "started", started)
        object.__setattr__(
            self, "deadline", started + float(self.aggregate_seconds)
        )

    def remaining(self) -> float:
        return self.deadline - float(self.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def guard(self, phase: str) -> None:
        if type(phase) is not str or not phase:
            raise CandidatePoolError(
                "diagnostic deadline phase must be a non-empty string"
            )
        if self.expired():
            raise _DiagnosticAggregateDeadlineExhausted(
                "DIAGNOSTIC_AGGREGATE_DEADLINE_EXHAUSTED "
                f"during {phase}"
            )

    def solver_budget(self, phase: str, *, hard_limit: float) -> float:
        if (
            isinstance(hard_limit, bool)
            or not isinstance(hard_limit, (int, float))
            or float(hard_limit) <= 0.0
        ):
            raise CandidatePoolLimitError(
                "diagnostic solver hard limit must be positive"
            )
        remaining = self.remaining()
        if remaining <= 0.0:
            raise _DiagnosticAggregateDeadlineExhausted(
                "DIAGNOSTIC_AGGREGATE_DEADLINE_EXHAUSTED "
                f"during {phase}"
            )
        return min(float(hard_limit), remaining)


@dataclass
class _DiagnosticDeadlineGeometryBudget:
    """Charge a replay's existing hard budget through the aggregate deadline."""

    delegate: Any
    deadline_guard: Callable[[str], None]
    phase: str

    def __post_init__(self) -> None:
        if not callable(getattr(self.delegate, "consume", None)):
            raise CandidatePoolError(
                "diagnostic replay budget delegate must expose consume"
            )
        if not callable(self.deadline_guard):
            raise CandidatePoolError(
                "diagnostic replay budget requires a deadline guard"
            )
        if type(self.phase) is not str or not self.phase:
            raise CandidatePoolError(
                "diagnostic replay budget phase must be a non-empty string"
            )

    def consume(self, reason: str) -> None:
        # Guard before every geometry debit.  The delegate retains the exact
        # deterministic operation ceiling; the callback adds an interruptible
        # aggregate wall-clock seam inside the replay's hot loops.
        self.deadline_guard(self.phase)
        self.delegate.consume(reason)


def _replay_exact_landing_site_membership_with_deadline(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    sites: Sequence[Any],
    *,
    deadline: _DiagnosticAggregateDeadline,
    phase: str,
    query_context: Any = None,
) -> tuple[Any, ...]:
    """Replay exact membership with a deadline check on every geometry debit."""

    requested = []
    try:
        iterator = iter(sites)
    except TypeError as exc:
        raise RoutingResourceEnumerationError(
            "landing-site domain projection must be iterable"
        ) from exc
    for site in iterator:
        deadline.guard(phase)
        requested.append(site)
        if len(requested) > resource_world_module.MAX_ENUMERATED_VIA_SITES:
            raise resource_world_module.RoutingResourceEnumerationLimitError(
                "landing-site domain projection count exceeds hard maximum "
                f"{resource_world_module.MAX_ENUMERATED_VIA_SITES}"
            )
    if not all(
        isinstance(site, resource_world_module.LandingSite)
        for site in requested
    ):
        raise RoutingResourceEnumerationError(
            "landing-site domain projection contains invalid values"
        )

    context = query_context
    if context is None:
        deadline.guard(f"{phase}-context")
        context = _prepare_exact_landing_site_query_context(
            ledger, world, tech, owned_conductor_index
        )
        deadline.guard(f"{phase}-context")
        context = replace(
            context,
            geometry_budget=_DiagnosticDeadlineGeometryBudget(
                delegate=context.geometry_budget,
                deadline_guard=deadline.guard,
                phase=phase,
            ),
        )

    replayed = []
    by_lower_layer: dict[int, list[Any]] = {}
    for site in requested:
        deadline.guard(phase)
        by_lower_layer.setdefault(site.lower_layer_n, []).append(site)
    for lower_layer_n, layer_sites in sorted(by_lower_layer.items()):
        deadline.guard(phase)
        replayed.extend(_query_exact_landing_sites_at_points(
            ledger,
            world,
            tech,
            owned_conductor_index,
            lower_layer_n=lower_layer_n,
            points=tuple(site.xy_nm for site in layer_sites),
            signed_points=False,
            query_context=context,
        ))
        deadline.guard(phase)

    expected = tuple(sorted(replayed, key=lambda site: site.site_id))
    actual = tuple(sorted(requested, key=lambda site: site.site_id))
    if actual != expected:
        raise RoutingResourceEnumerationError(
            "landing-site projection does not exactly replay from its domain"
        )
    return actual


def _replay_exact_landing_site_subset_compressed_with_deadline(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    sites: Sequence[Any],
    *,
    deadline: _DiagnosticAggregateDeadline,
) -> tuple[tuple[Any, ...], tuple[Any, ...], Any]:
    """Exact subset replay with deterministic and wall-clock internal bounds."""

    phase = "landing-site-subset-replay"
    actual = _replay_exact_landing_site_membership_with_deadline(
        ledger,
        world,
        tech,
        owned_conductor_index,
        sites,
        deadline=deadline,
        phase=phase,
    )
    layers = resource_world_module._routing_layer_facts(tech)
    geometry_budget = _DiagnosticDeadlineGeometryBudget(
        delegate=resource_world_module._GeometryEvaluationBudget(
            resource_world_module.DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS
        ),
        deadline_guard=deadline.guard,
        phase=phase,
    )
    try:
        conflicts = (
            resource_world_module._enumerate_landing_site_pair_conflicts(
                actual,
                layers,
                tech,
                world.sha256(),
                max_site_pair_conflicts=(
                    resource_world_module
                    .MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS
                ),
                geometry_budget=geometry_budget,
            )
        )
    except resource_world_module.LandingSitePairConflictLimitError:
        deadline.guard(phase)
        return (
            actual,
            (),
            resource_world_module._landing_site_conflict_domain(
                actual, layers, tech, world.sha256()
            ),
        )
    deadline.guard(phase)
    return actual, conflicts, None


def _replay_exact_landing_site_catalog_compressed_with_deadline(
    ledger: ExplicitRoutingResourceLedger,
    world: RoutingWorldProvenance,
    tech: Any,
    owned_conductor_index: Any,
    sites: Sequence[Any],
    *,
    deadline: _DiagnosticAggregateDeadline,
) -> tuple[LandingSiteCatalog, tuple[Any, ...], Any]:
    """Exact paged replay with a deadline seam in membership geometry loops."""

    phase = "landing-site-catalog-replay"
    page_size = min(
        resource_world_module.MAX_ENUMERATED_VIA_SITES,
        MAX_LANDING_SITES,
    )
    deadline.guard(f"{phase}-context")
    context = _prepare_exact_landing_site_query_context(
        ledger, world, tech, owned_conductor_index
    )
    deadline.guard(f"{phase}-context")
    context = replace(
        context,
        geometry_budget=_DiagnosticDeadlineGeometryBudget(
            delegate=context.geometry_budget,
            deadline_guard=deadline.guard,
            phase=phase,
        ),
    )
    pages = []
    pending = []
    previous_site_id: str | None = None
    total_count = 0

    def replay_page(page_sites: list[Any]) -> None:
        deadline.guard(phase)
        replayed_page = (
            _replay_exact_landing_site_membership_with_deadline(
                ledger,
                world,
                tech,
                owned_conductor_index,
                page_sites,
                deadline=deadline,
                phase=phase,
                query_context=context,
            )
        )
        if replayed_page != tuple(page_sites):
            raise RoutingResourceEnumerationError(
                "landing-site catalog projection does not exactly replay from "
                "its domain"
            )
        pages.append(LandingSitePage.build(
            len(pages),
            replayed_page,
        ))

    try:
        iterator = iter(sites)
    except TypeError as exc:
        raise RoutingResourceEnumerationError(
            "landing-site catalog projection must be iterable"
        ) from exc
    for site in iterator:
        deadline.guard(phase)
        if not isinstance(site, resource_world_module.LandingSite):
            raise RoutingResourceEnumerationError(
                "landing-site catalog projection contains invalid values"
            )
        if previous_site_id is not None and site.site_id <= previous_site_id:
            raise RoutingResourceEnumerationError(
                "landing-site catalog projection ids must be globally unique "
                "and sorted"
            )
        previous_site_id = site.site_id
        pending.append(site)
        total_count += 1
        if total_count > MAX_TOTAL_ASSIGNMENT_ARCS:
            raise resource_world_module.RoutingResourceEnumerationLimitError(
                "landing-site catalog projection count exceeds hard maximum "
                f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
            )
        if len(pending) == page_size:
            replay_page(pending)
            pending = []
    if pending:
        replay_page(pending)
    if not pages:
        raise RoutingResourceEnumerationError(
            "landing-site catalog projection must not be empty"
        )
    catalog = LandingSiteCatalog.from_pages(
        tuple(pages),
        routing_world_provenance_digest=world.sha256(),
        landing_site_domain_digest=ledger.landing_site_domain_digest,
    )
    domain = resource_world_module._landing_site_conflict_domain(
        catalog,
        resource_world_module._routing_layer_facts(tech),
        tech,
        world.sha256(),
        landing_site_collection_digest=catalog.collection_sha256,
    )
    deadline.guard(phase)
    return catalog, (), domain


def _build_bounded_diagnostic_site_index(
    source: Any,
    *,
    requested_site_ids: Sequence[str],
    max_indexed_sites: int,
    deadline_guard: Callable[[str], None],
) -> dict[str, Any]:
    """Scan one source catalog at most once and retain only requested site ids."""

    if type(max_indexed_sites) is not int or max_indexed_sites <= 0:
        raise CandidatePoolLimitError(
            "diagnostic site-index limit must be a positive integer"
        )
    requested = tuple(sorted(requested_site_ids))
    if (
        len(requested) != len(set(requested))
        or any(type(site_id) is not str or not site_id for site_id in requested)
    ):
        raise CandidatePoolError(
            "diagnostic site-index request ids must be unique non-empty strings"
        )
    if len(requested) > max_indexed_sites:
        raise CandidatePoolLimitError(
            "diagnostic site-index request count "
            f"{len(requested)} exceeds {max_indexed_sites}"
        )
    if not callable(deadline_guard):
        raise CandidatePoolError(
            "diagnostic site-index requires a deadline guard"
        )
    wanted = set(requested)
    found: dict[str, Any] = {}
    try:
        iterator = iter(source)
    except TypeError as exc:
        raise CandidatePoolError(
            "diagnostic landing-site source is not iterable"
        ) from exc
    for site in iterator:
        deadline_guard("landing-site-catalog-index")
        site_id = getattr(site, "site_id", None)
        if site_id not in wanted:
            continue
        if site_id in found:
            raise CandidatePoolError(
                "diagnostic source landing site ids are not unique"
            )
        found[site_id] = site
        if len(found) == len(wanted):
            break
    deadline_guard("landing-site-catalog-index")
    missing = tuple(sorted(wanted - set(found)))
    if missing:
        raise CandidatePoolError(
            f"diagnostic family names missing landing sites {missing}"
        )
    return {site_id: found[site_id] for site_id in requested}


class _BoundedDiagnosticFailureEvidence:
    """Keep every deterministic failure, grouping only the output record."""

    def __init__(self, *, max_refs: int) -> None:
        if type(max_refs) is not int or max_refs < 2:
            raise CandidatePoolError(
                "diagnostic failure evidence bound must be at least two"
            )
        self._max_refs = max_refs
        self._values: list[str] = []

    def record(self, value: str) -> None:
        if type(value) is not str or not value:
            raise CandidatePoolError(
                "diagnostic failure evidence must be a non-empty string"
            )
        self._values.append(value)

    def refs(self) -> tuple[str, ...]:
        refs = tuple(self._values)
        if len(refs) <= self._max_refs:
            return refs
        # This is exact structural grouping, not digest-only overflow or
        # evidence truncation: every original record remains recoverable from
        # the canonical diagnostic artifact.  Only the top-level tuple shape
        # is compressed so its long-standing schema bound remains intact.
        return (
            "diagnostic-failure-group:"
            + _canonical_json({
                "refs": list(refs),
                "schema": "clay.diagnostic-failure-group.v1",
            }),
        )


def _diagnostic_member_projection_digest(
    *,
    candidate_id: str,
    member_ledger_hash: str,
    member_pool_hash: str,
    member_site_ids: tuple[str, ...],
    net: str,
    source_family_ref: str,
    source_pool_hash: str,
) -> str:
    if (
        tuple(sorted(member_site_ids)) != member_site_ids
        or len(member_site_ids) != len(set(member_site_ids))
    ):
        raise CandidatePoolError(
            "diagnostic member site ids must be sorted and unique"
        )
    return hashlib.sha256(_canonical_json({
        "candidate_id": candidate_id,
        "member_ledger_hash": member_ledger_hash,
        "member_pool_hash": member_pool_hash,
        "member_site_ids": list(member_site_ids),
        "net": net,
        "schema": "clay.diagnostic-member-projection.v1",
        "source_family_ref": source_family_ref,
        "source_pool_hash": source_pool_hash,
    }).encode("utf-8")).hexdigest()


def _diagnostic_omitted_members_digest(
    *,
    ref: str,
    net: str,
    candidate_ids: tuple[str, ...],
) -> str:
    if (
        tuple(sorted(candidate_ids)) != candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise CandidatePoolError(
            "diagnostic omitted member ids must be sorted and unique"
        )
    return hashlib.sha256(_canonical_json({
        "candidate_ids": list(candidate_ids),
        "net": net,
        "ref": ref,
        "schema": "clay.diagnostic-omitted-members.v1",
    }).encode("utf-8")).hexdigest()


def _diagnostic_frontier_digest(
    *,
    failed_partition_refs: tuple[str, ...],
    family_projection_digest: str,
    merge_plan_hash: str,
    merge_pool_hash: str,
    merge_policy: str,
    merge_search_exhaustive: bool,
    merge_solve_count: int,
    source_projection_exhaustive: bool,
    selected_nets: tuple[str, ...],
    source_stop_digest: str,
    source_witness_hashes: tuple[tuple[str, str, str], ...],
) -> str:
    return hashlib.sha256(_canonical_json({
        "failed_partition_refs": sorted(failed_partition_refs),
        "family_projection_digest": family_projection_digest,
        "merge_plan_hash": merge_plan_hash,
        "merge_pool_hash": merge_pool_hash,
        "merge_policy": merge_policy,
        "merge_search_exhaustive": merge_search_exhaustive,
        "merge_solve_count": merge_solve_count,
        "source_projection_exhaustive": source_projection_exhaustive,
        "schema": "clay.diagnostic-frontier.v1",
        "selected_nets": list(selected_nets),
        "source_stop_digest": source_stop_digest,
        "source_witness_hashes": [
            list(item) for item in sorted(source_witness_hashes)
        ],
    }).encode("utf-8")).hexdigest()


def _diagnostic_round_robin_members(
    families: Sequence[
        tuple[str, str, Sequence[CompleteNetCandidate]]
    ],
) -> tuple[tuple[str, str, CompleteNetCandidate], ...]:
    """Visit one canonical member per family before any second sibling."""

    normalized = tuple(sorted((
        (
            ref,
            net,
            tuple(sorted(members, key=lambda item: item.candidate_id)),
        )
        for ref, net, members in families
    ), key=lambda item: (item[1], item[0])))
    refs = tuple(ref for ref, _net, _members in normalized)
    if len(refs) != len(set(refs)):
        raise CandidatePoolError(
            "diagnostic round-robin family refs must be unique"
        )
    if any(
        type(ref) is not str
        or not ref
        or type(net) is not str
        or not net
        or not members
        or any(
            type(candidate) is not CompleteNetCandidate
            or candidate.net != net
            for candidate in members
        )
        for ref, net, members in normalized
    ):
        raise CandidatePoolError(
            "diagnostic round-robin family membership is malformed"
        )
    schedule = []
    for ordinal in range(max(
        (len(members) for _ref, _net, members in normalized),
        default=0,
    )):
        schedule.extend(
            (ref, net, members[ordinal])
            for ref, net, members in normalized
            if ordinal < len(members)
        )
    return tuple(schedule)


@dataclass(frozen=True)
class _DiagnosticFrontierSearchResult:
    """Best exactly verified frontier found by the bounded deterministic search."""

    selected: tuple[Any, ...]
    witness: Any
    solve_count: int
    full_frontier: bool
    search_exhaustive: bool
    budget_exhausted: bool
    stop_code: str = ""
    stop_detail: str = ""


def _diagnostic_exact_frontier_search(
    ranked_winners_by_net: Mapping[str, Sequence[tuple[Any, ...]]],
    *,
    evaluate: Callable[[tuple[Any, ...]], Any | None],
    max_solves: int,
    deadline_reached: Callable[[], bool],
) -> _DiagnosticFrontierSearchResult:
    """Select a maximum conflict-free local-winner frontier with backtracking.

    ``evaluate`` is the exact feasibility seam: it returns a replay witness for a
    conflict-free prefix and ``None`` otherwise.  Candidate order is derived only
    from the canonical local rank/ref/id facts, so caller mapping/list order cannot
    affect the first maximum solution.  A full-net solution is already cardinality
    optimal and ends the search early; otherwise every unpruned branch is visited
    unless the aggregate solve/time budget expires.
    """

    if type(max_solves) is not int or max_solves <= 0:
        raise CandidatePoolLimitError(
            "diagnostic exact merge solve budget must be a positive integer"
        )
    if not callable(evaluate) or not callable(deadline_reached):
        raise CandidatePoolError(
            "diagnostic exact merge requires callable evaluation and deadline seams"
        )

    normalized: list[tuple[str, tuple[Any, ...]]] = []
    for net in sorted(ranked_winners_by_net):
        if type(net) is not str or not net:
            raise CandidatePoolError(
                "diagnostic exact merge frontier contains a malformed net"
            )
        entries = []
        for entry in ranked_winners_by_net[net]:
            if type(entry) is not tuple or len(entry) < 3:
                raise CandidatePoolError(
                    "diagnostic exact merge frontier entry is malformed"
                )
            rank, ref, candidate = entry[:3]
            if (
                type(ref) is not str
                or not ref
                or getattr(candidate, "net", None) != net
                or type(getattr(candidate, "candidate_id", None)) is not str
                or not candidate.candidate_id
            ):
                raise CandidatePoolError(
                    "diagnostic exact merge frontier entry is not net-bound"
                )
            entries.append((rank, ref, candidate))
        entries.sort(key=lambda item: (
            item[0], item[1], item[2].candidate_id
        ))
        unique = []
        seen_by_id: dict[str, Any] = {}
        for _rank, _ref, candidate in entries:
            previous = seen_by_id.get(candidate.candidate_id)
            if previous is not None:
                if previous != candidate:
                    raise CandidatePoolError(
                        "diagnostic exact merge candidate id aliases "
                        "different local winners"
                    )
                continue
            seen_by_id[candidate.candidate_id] = candidate
            unique.append(candidate)
        if unique:
            normalized.append((net, tuple(unique)))
    if not normalized:
        raise CandidatePoolError(
            "diagnostic exact merge frontier has no local winners"
        )

    best_selected: tuple[Any, ...] = ()
    best_witness: Any = None
    solve_count = 0
    budget_exhausted = False
    full_frontier = False
    stop_code = ""
    stop_detail = ""

    def canonical_trial(values: Sequence[Any]) -> tuple[Any, ...]:
        return tuple(sorted(
            values,
            key=lambda item: (item.net, item.candidate_id),
        ))

    def search(index: int, accepted: tuple[Any, ...]) -> None:
        nonlocal best_selected, best_witness
        nonlocal solve_count, budget_exhausted, full_frontier
        nonlocal stop_code, stop_detail
        if full_frontier or budget_exhausted:
            return
        remaining = len(normalized) - index
        if len(accepted) + remaining <= len(best_selected):
            return
        if index >= len(normalized):
            return

        _net, options = normalized[index]
        for candidate in options:
            if solve_count >= max_solves or deadline_reached():
                budget_exhausted = True
                return
            trial = canonical_trial((*accepted, candidate))
            solve_count += 1
            try:
                witness = evaluate(trial)
            except _DiagnosticAggregateDeadlineExhausted as exc:
                budget_exhausted = True
                stop_code = "DIAGNOSTIC_AGGREGATE_DEADLINE_EXHAUSTED"
                stop_detail = str(exc)
                return
            except RouteResourcePlanVerificationBudgetExhausted as exc:
                budget_exhausted = True
                stop_code = (
                    "DIAGNOSTIC_EXACT_MERGE_VERIFICATION_BUDGET_EXHAUSTED"
                )
                stop_detail = str(exc)
                return
            except CandidatePoolLimitError as exc:
                budget_exhausted = True
                stop_code = "DIAGNOSTIC_EXACT_MERGE_LIMIT_EXHAUSTED"
                stop_detail = str(exc)
                return
            if witness is None:
                continue
            if len(trial) > len(best_selected):
                best_selected = trial
                best_witness = witness
            if len(trial) == len(normalized):
                full_frontier = True
                return
            search(index + 1, trial)
            if full_frontier or budget_exhausted:
                return

        # Skipping a net is necessary to prove/find a maximum partial frontier
        # when no all-net combination exists.
        search(index + 1, accepted)

    search(0, ())
    return _DiagnosticFrontierSearchResult(
        selected=best_selected,
        witness=best_witness,
        solve_count=solve_count,
        full_frontier=full_frontier,
        search_exhaustive=full_frontier or not budget_exhausted,
        budget_exhausted=budget_exhausted,
        stop_code=(
            stop_code
            or (
                "DIAGNOSTIC_EXACT_MERGE_BUDGET_EXHAUSTED"
                if budget_exhausted
                else ""
            )
        ),
        stop_detail=stop_detail,
    )


def _build_diagnostic_decomposed_merge_pool(
    *,
    source_unknown: PatternCompileUnknown,
    compiled: Sequence[CompleteNetCandidate],
    flat_families: Sequence[CompleteNetCandidateFamily],
    factorized_families: Sequence[FactorizedCompleteNetCandidateFamily],
    routing_world: RoutingWorldProvenance,
    topology_plan: TopologyPlan,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    tech: Any,
    topology_digest: str,
    world_digest: str,
    conductor_digest: str,
    incidence_digest: str,
    resources: tuple[TrackResource, ...],
    sites: tuple[Any, ...],
    site_catalog: LandingSiteCatalog | None,
    site_conflicts: tuple[Any, ...],
    site_conflict_domain: Any,
    ledger_hash: str,
    landing_sites_complete: bool,
    certificates: tuple[OwnedConductorComponentCertificate, ...],
) -> DiagnosticDecomposedCompileResult:
    """Stream bounded family solves into a non-signoff materialization bundle.

    Ordinary ``CandidatePool`` and ``RouteResourcePlan`` objects exist only as
    short-lived local witnesses.  They are never returned.  Cross-family merge
    performs bounded deterministic exact backtracking over those local winners;
    a budget stop remains typed and any carried partial stays non-signoff.
    """

    from clay.layout.route.resource_plan import (
        _bringup_partition_plan_rank,
        ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME,
        select_complete_net_candidates,
        verify_route_plan_verification_receipt,
        verify_selection_certificate,
    )

    deadline = _DiagnosticAggregateDeadline(
        aggregate_seconds=MAX_DIAGNOSTIC_ELAPSED_SECONDS,
        monotonic=time.monotonic,
    )

    compiled_by_id = {
        candidate.candidate_id: candidate for candidate in compiled
    }
    if len(compiled_by_id) != len(tuple(compiled)):
        raise CandidatePoolError(
            "diagnostic family decomposition sees duplicate candidate ids"
        )
    covered_flat_ids: set[str] = set()
    family_specs: list[
        tuple[
            str,
            str,
            tuple[CompleteNetCandidate, ...],
            tuple[CompleteNetCandidateFamily, ...],
            tuple[FactorizedCompleteNetCandidateFamily, ...],
        ]
    ] = []
    for family in sorted(flat_families, key=lambda item: item.family_id):
        overlap = covered_flat_ids.intersection(family.variant_candidate_ids)
        if overlap:
            raise CandidatePoolError(
                "diagnostic family projection aliases flat candidate ids "
                f"{tuple(sorted(overlap))}"
            )
        missing = tuple(
            candidate_id
            for candidate_id in family.variant_candidate_ids
            if candidate_id not in compiled_by_id
        )
        if missing:
            raise CandidatePoolError(
                "diagnostic family projection misses flat source candidates "
                f"{missing}"
            )
        members = tuple(
            compiled_by_id[candidate_id]
            for candidate_id in family.variant_candidate_ids
        )
        if any(candidate.net != family.net for candidate in members):
            raise CandidatePoolError(
                "diagnostic flat family net differs from its source candidates"
            )
        covered_flat_ids.update(family.variant_candidate_ids)
        family_specs.append((
            f"flat:{family.family_id}",
            family.net,
            members,
            (family,),
            (),
        ))
    for candidate in sorted(compiled, key=lambda item: item.candidate_id):
        if candidate.candidate_id in covered_flat_ids:
            continue
        family_specs.append((
            f"singleton:{candidate.candidate_id}",
            candidate.net,
            (candidate,),
            (),
            (),
        ))
    for family in sorted(factorized_families, key=lambda item: item.family_id):
        family_specs.append((
            f"factorized:{family.family_id}",
            family.net,
            (),
            (),
            (family,),
        ))
    if not family_specs:
        raise CandidatePoolError(
            "diagnostic family decomposition has no candidate families"
        )

    refs = tuple(spec[0] for spec in family_specs)
    if len(refs) != len(set(refs)):
        raise CandidatePoolError(
            "diagnostic family projection has duplicate canonical refs"
        )
    factorized_profile_candidate_ids = tuple(
        profile.candidate_id
        for family in factorized_families
        for profile in family.profiles
    )
    if (
        len(factorized_profile_candidate_ids)
        != len(set(factorized_profile_candidate_ids))
        or set(factorized_profile_candidate_ids).intersection(compiled_by_id)
    ):
        raise CandidatePoolError(
            "diagnostic factorized profile candidate ids are not a disjoint bijection"
        )
    if any(spec[1] not in routing_world.required_nets for spec in family_specs):
        raise CandidatePoolError(
            "diagnostic family projection names a non-required net"
        )
    projection_payload = {
        "families": [
            {
                "candidate_ids": [item.candidate_id for item in spec[2]],
                "factorized_family_ids": [item.family_id for item in spec[4]],
                "flat_family_ids": [item.family_id for item in spec[3]],
                "net": spec[1],
                "ref": spec[0],
            }
            for spec in sorted(family_specs, key=lambda item: (item[1], item[0]))
        ],
        "schema": "clay.diagnostic-family-projection-runtime.v1",
    }
    projection_digest = hashlib.sha256(_canonical_json(
        projection_payload
    ).encode("utf-8")).hexdigest()

    requested_source_site_id_set: set[str] = set()
    source_site_ids_by_ref: dict[str, tuple[str, ...]] = {}
    member_site_ids_by_candidate_id: dict[str, tuple[str, ...]] = {}
    for ref, _net, candidates, _flat, factorized in family_specs:
        deadline.guard("landing-site-index-request")
        family_site_ids = _diagnostic_candidate_site_ids(
            candidates,
            factorized,
            deadline_guard=deadline.guard,
        )
        source_site_ids_by_ref[ref] = family_site_ids
        for candidate in candidates:
            if candidate.candidate_id in member_site_ids_by_candidate_id:
                continue
            member_site_ids_by_candidate_id[candidate.candidate_id] = (
                _diagnostic_candidate_site_ids(
                    (candidate,),
                    (),
                    deadline_guard=deadline.guard,
                )
            )
        for site_id in family_site_ids:
            deadline.guard("landing-site-index-request")
            if site_id in requested_source_site_id_set:
                continue
            if len(requested_source_site_id_set) >= MAX_TOTAL_ASSIGNMENT_ARCS:
                raise CandidatePoolLimitError(
                    "diagnostic site-index request count exceeds "
                    f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
                )
            requested_source_site_id_set.add(site_id)
    requested_source_site_ids = tuple(sorted(requested_source_site_id_set))
    source_site_index: dict[str, Any] | None = None
    complete_source_site_ids: tuple[str, ...] | None = None
    if landing_sites_complete:
        complete_source: dict[str, Any] = {}
        complete_source_iter = (
            site_catalog.iter_sites()
            if site_catalog is not None
            else iter(sites)
        )
        for site in complete_source_iter:
            deadline.guard("complete-landing-site-index")
            site_id = getattr(site, "site_id", None)
            if type(site_id) is not str or not site_id:
                raise CandidatePoolError(
                    "complete diagnostic landing-site source is malformed"
                )
            if site_id in complete_source:
                raise CandidatePoolError(
                    "complete diagnostic landing-site ids are not unique"
                )
            if len(complete_source) >= MAX_TOTAL_ASSIGNMENT_ARCS:
                raise CandidatePoolLimitError(
                    "complete diagnostic landing-site index exceeds "
                    f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
                )
            complete_source[site_id] = site
        deadline.guard("complete-landing-site-index")
        complete_source_site_ids = tuple(sorted(complete_source))
        source_site_index = {
            site_id: complete_source[site_id]
            for site_id in complete_source_site_ids
        }

    def source_sites(site_ids: tuple[str, ...]) -> tuple[Any, ...]:
        nonlocal source_site_index
        if not site_ids:
            return ()
        if source_site_index is None:
            source = (
                site_catalog.iter_sites()
                if site_catalog is not None
                else iter(sites)
            )
            source_site_index = _build_bounded_diagnostic_site_index(
                source,
                requested_site_ids=requested_source_site_ids,
                max_indexed_sites=MAX_TOTAL_ASSIGNMENT_ARCS,
                deadline_guard=deadline.guard,
            )
        missing = tuple(
            site_id for site_id in site_ids if site_id not in source_site_index
        )
        if missing:
            raise CandidatePoolError(
                f"diagnostic family names missing landing sites {missing}"
            )
        return tuple(source_site_index[site_id] for site_id in site_ids)

    projected_site_bundle_cache: dict[tuple[str, ...], tuple[Any, ...]] = {}

    def projected_site_bundle(site_ids: tuple[str, ...]):
        deadline.guard("landing-site-projection")
        if (
            tuple(sorted(site_ids)) != site_ids
            or len(site_ids) != len(set(site_ids))
        ):
            raise CandidatePoolError(
                "diagnostic projected site ids must be sorted and unique"
            )
        cached = projected_site_bundle_cache.get(site_ids)
        if cached is not None:
            deadline.guard("landing-site-projection-cache-hit")
            return cached

        def remember(bundle: tuple[Any, ...]) -> tuple[Any, ...]:
            # The cache has no independent semantic cap: its keys are already
            # bounded by the validated source-family count plus the existing
            # partition/merge solve limits.  A smaller secondary cap could
            # make an otherwise legal final member fail after all source
            # projections had populated the cache.
            projected_site_bundle_cache[site_ids] = bundle
            return bundle

        if (
            complete_source_site_ids is not None
            and site_ids == complete_source_site_ids
        ):
            return remember((
                sites,
                site_catalog,
                site_conflicts,
                site_conflict_domain,
                ledger_hash,
                True,
            ))
        if not site_ids:
            if landing_sites_complete and not ledger.landing_site_domain_digest:
                # Legacy complete ledgers may lawfully have no domain digest;
                # inventing one would sever world provenance.  Preserve that
                # complete source only for this otherwise-unprojectable empty
                # member.  Modern bound domains take the exact empty subset.
                return remember((
                    sites,
                    site_catalog,
                    site_conflicts,
                    site_conflict_domain,
                    ledger_hash,
                    True,
                ))
            empty_ledger = ExplicitRoutingResourceLedger(
                routing_world_provenance_digest=world_digest,
                track_resources=resources,
                landing_sites=(),
                landing_site_pair_conflicts=(),
                owned_conductor_index_digest=conductor_digest,
                landing_site_domain_digest=ledger.landing_site_domain_digest,
                landing_sites_complete=False,
            )
            return remember((
                (), None, (), None, empty_ledger.sha256(), False
            ))
        selected_sites = source_sites(site_ids)
        if len(selected_sites) <= MAX_PATTERN_LANDING_SITE_PAGE_SIZE:
            deadline.guard("landing-site-subset-replay")
            projected_sites, conflicts, domain = (
                _replay_exact_landing_site_subset_compressed_with_deadline(
                    ledger,
                    routing_world,
                    tech,
                    conductor_index,
                    selected_sites,
                    deadline=deadline,
                )
            )
            deadline.guard("landing-site-subset-replay")
            projected_ledger = ExplicitRoutingResourceLedger(
                routing_world_provenance_digest=world_digest,
                track_resources=resources,
                landing_sites=projected_sites,
                landing_site_pair_conflicts=conflicts,
                owned_conductor_index_digest=conductor_digest,
                landing_site_domain_digest=ledger.landing_site_domain_digest,
                landing_sites_complete=False,
                landing_site_conflict_domain=domain,
            )
            return remember((
                projected_sites,
                None,
                conflicts,
                domain,
                projected_ledger.sha256(),
                False,
            ))
        deadline.guard("landing-site-catalog-replay")
        catalog, conflicts, domain = (
            _replay_exact_landing_site_catalog_compressed_with_deadline(
            ledger,
            routing_world,
            tech,
            conductor_index,
            selected_sites,
            deadline=deadline,
        ))
        deadline.guard("landing-site-catalog-replay")
        return remember((
            (),
            catalog,
            conflicts,
            domain,
            paged_routing_resource_ledger_sha256(
                world_digest,
                resources,
                catalog,
                conflicts,
                conductor_digest,
                ledger.landing_site_domain_digest,
                False,
                domain,
            ),
            False,
        ))

    def verify_partition_projection(
        ref: str,
        net: str,
        candidates: tuple[CompleteNetCandidate, ...],
        candidate_families: tuple[CompleteNetCandidateFamily, ...],
        factorized: tuple[FactorizedCompleteNetCandidateFamily, ...],
        partition: CandidatePool,
        expected_site_ids: tuple[str, ...],
        expected_sites: tuple[Any, ...],
        expected_catalog: LandingSiteCatalog | None,
        expected_conflicts: tuple[Any, ...],
        expected_domain: Any,
        expected_ledger_hash: str,
        goals: tuple[RequiredRouteGoal, ...],
        expected_complete: bool,
    ) -> None:
        actual_sites = tuple(
            partition.landing_site_catalog.iter_sites()
            if partition.landing_site_catalog is not None
            else partition.landing_sites
        )
        if (
            partition.required_nets != (net,)
            or partition.candidates != candidates
            or partition.candidate_families != candidate_families
            or partition.factorized_candidate_families != factorized
            or partition.required_route_goals != goals
            or partition.required_goal_refs
            != tuple(goal.declaration_ref for goal in goals)
            or not _diagnostic_site_projection_matches(
                expected_site_ids,
                tuple(site.site_id for site in actual_sites),
                complete_source_domain=expected_complete,
            )
            or actual_sites != tuple(
                expected_catalog.iter_sites()
                if expected_catalog is not None
                else expected_sites
            )
            or partition.landing_site_conflicts != expected_conflicts
            or partition.landing_site_conflict_domain != expected_domain
            or partition.routing_resource_ledger_hash != expected_ledger_hash
            or partition.resources != resources
            or partition.placement_provenance_digest
            != routing_world.placement_provenance_digest
            or partition.topology_provenance_digest != topology_digest
            or partition.pdk_provenance_digest
            != routing_world.pdk_provenance_digest
            or partition.routing_world_provenance_digest != world_digest
            or partition.owned_conductor_index_digest != conductor_digest
            or partition.lane_incidence_replay_digest != incidence_digest
            or partition.landing_site_domain_digest
            != ledger.landing_site_domain_digest
            or partition.conflicts
        ):
            raise CandidatePoolError(
                f"diagnostic family projection mismatch for {ref}"
            )

    goals_by_net = {
        net: tuple(
            goal for goal in topology_plan.required_goals if goal.net == net
        )
        for net in routing_world.required_nets
    }
    local_winners_by_net: dict[
        str,
        list[
            tuple[
                tuple[object, ...],
                str,
                CompleteNetCandidate,
                str,
                str,
            ]
        ],
    ] = {}
    witness_hashes: list[tuple[str, str, str]] = []
    failure_evidence = _BoundedDiagnosticFailureEvidence(
        max_refs=MAX_PATTERN_UNKNOWN_REFS
    )

    def record_failure(value: str) -> None:
        failure_evidence.record(value)

    successful_family_count = 0
    partition_solves = 0
    projection_exhaustive = True
    projected_family_members: list[
        tuple[
            str,
            str,
            str,
            tuple[CompleteNetCandidate, ...],
            tuple[RequiredRouteGoal, ...],
        ]
    ] = []
    source_membership_payloads: list[dict[str, Any]] = []
    for ref, net, candidates, candidate_families, factorized in sorted(
        family_specs, key=lambda item: (item[1], item[0])
    ):
        if (
            partition_solves >= MAX_DIAGNOSTIC_PARTITION_SOLVES
            or deadline.expired()
        ):
            record_failure(
                f"{ref}:DIAGNOSTIC_AGGREGATE_BUDGET_EXHAUSTED"
            )
            projection_exhaustive = False
            break
        try:
            expected_site_ids = source_site_ids_by_ref[ref]
            (
                projected_sites,
                projected_catalog,
                projected_conflicts,
                projected_domain,
                projected_ledger_hash,
                projected_complete,
            ) = projected_site_bundle(expected_site_ids)
            goals = goals_by_net.get(net, ())
            deadline.guard("partition-candidate-pool-replay")
            partition = CandidatePool(
                placement_provenance_digest=(
                    routing_world.placement_provenance_digest
                ),
                topology_provenance_digest=topology_digest,
                required_nets=(net,),
                required_goal_refs=tuple(
                    goal.declaration_ref for goal in goals
                ),
                resources=resources,
                landing_sites=projected_sites,
                candidates=candidates,
                conflicts=(),
                candidate_families=candidate_families,
                factorized_candidate_families=factorized,
                landing_site_catalog=projected_catalog,
                landing_site_conflicts=projected_conflicts,
                required_route_goals=goals,
                pdk_provenance_digest=routing_world.pdk_provenance_digest,
                routing_world_provenance_digest=world_digest,
                routing_resource_ledger_hash=projected_ledger_hash,
                owned_conductor_index_digest=conductor_digest,
                lane_incidence_replay_digest=incidence_digest,
                landing_site_domain_digest=ledger.landing_site_domain_digest,
                landing_sites_complete=projected_complete,
                landing_site_conflict_domain=projected_domain,
                owned_conductor_components=certificates,
            )
            deadline.guard("partition-candidate-pool-replay")
        except _DiagnosticAggregateDeadlineExhausted:
            raise
        except CandidatePoolLimitError as exc:
            record_failure(
                f"{ref}:{type(exc).__name__}:{str(exc)[:160]}"
            )
            projection_exhaustive = False
            continue
        deadline.guard("partition-projection-verification")
        verify_partition_projection(
            ref,
            net,
            candidates,
            candidate_families,
            factorized,
            partition,
            expected_site_ids,
            projected_sites,
            projected_catalog,
            projected_conflicts,
            projected_domain,
            projected_ledger_hash,
            goals,
            projected_complete,
        )
        deadline.guard("partition-projection-verification")
        source_pool_hash = partition.sha256()
        source_candidates = tuple(sorted(
            partition.all_candidates,
            key=lambda item: item.candidate_id,
        ))
        if (
            not source_candidates
            or any(candidate.net != net for candidate in source_candidates)
        ):
            raise CandidatePoolError(
                "diagnostic source family did not realize exact net candidates"
            )
        projected_family_members.append((
            ref,
            net,
            source_pool_hash,
            source_candidates,
            goals,
        ))
        source_membership_payloads.append({
            "candidate_ids": [
                candidate.candidate_id for candidate in source_candidates
            ],
            "net": net,
            "ref": ref,
            "source_pool_hash": source_pool_hash,
        })

    verified_projection_digest = hashlib.sha256(_canonical_json({
        "declared_projection_digest": projection_digest,
        "schema": "clay.diagnostic-family-projection-runtime.v2",
        "source_memberships": sorted(
            source_membership_payloads,
            key=lambda item: (item["net"], item["ref"]),
        ),
    }).encode("utf-8")).hexdigest()

    # The source family remains a byte-exact membership/domain witness.  Its
    # members are visited round-robin by canonical ordinal so one dense family
    # cannot starve another net before the hard projection budget is reached.
    # Each content-addressed member gets its own closed pool and full replay;
    # every verified sibling remains available to the later multi-net merge.
    visited_by_ref = {ref: 0 for ref, *_rest in projected_family_members}
    winner_family_refs: set[str] = set()
    family_context_by_ref = {
        ref: (source_pool_hash, source_candidates, goals)
        for ref, _net, source_pool_hash, source_candidates, goals in (
            projected_family_members
        )
    }
    member_schedule = _diagnostic_round_robin_members(tuple(
        (ref, net, source_candidates)
        for ref, net, _source_pool_hash, source_candidates, _goals in (
            projected_family_members
        )
    ))
    for ref, net, source_candidate in member_schedule:
        source_pool_hash, _source_candidates, goals = (
            family_context_by_ref[ref]
        )
        candidate_ref = (
            f"{ref}:candidate:{source_candidate.candidate_id}:"
            f"source_pool={source_pool_hash}"
        )
        if (
            partition_solves >= MAX_DIAGNOSTIC_PARTITION_SOLVES
            or deadline.expired()
        ):
            record_failure(
                f"{candidate_ref}:DIAGNOSTIC_AGGREGATE_BUDGET_EXHAUSTED"
            )
            projection_exhaustive = False
            break
        partition_solves += 1
        visited_by_ref[ref] += 1
        try:
            member_site_ids = member_site_ids_by_candidate_id.get(
                source_candidate.candidate_id
            )
            if member_site_ids is None:
                member_site_ids = _diagnostic_candidate_site_ids(
                    (source_candidate,),
                    (),
                    deadline_guard=deadline.guard,
                )
                member_site_ids_by_candidate_id[
                    source_candidate.candidate_id
                ] = member_site_ids
            (
                member_sites,
                member_catalog,
                member_conflicts,
                member_domain,
                member_ledger_hash,
                member_complete,
            ) = projected_site_bundle(member_site_ids)
            deadline.guard("member-candidate-pool-replay")
            member_pool = CandidatePool(
                placement_provenance_digest=(
                    routing_world.placement_provenance_digest
                ),
                topology_provenance_digest=topology_digest,
                required_nets=(net,),
                required_goal_refs=tuple(
                    goal.declaration_ref for goal in goals
                ),
                resources=resources,
                landing_sites=member_sites,
                candidates=(source_candidate,),
                conflicts=(),
                landing_site_catalog=member_catalog,
                landing_site_conflicts=member_conflicts,
                required_route_goals=goals,
                pdk_provenance_digest=(
                    routing_world.pdk_provenance_digest
                ),
                routing_world_provenance_digest=world_digest,
                routing_resource_ledger_hash=member_ledger_hash,
                owned_conductor_index_digest=conductor_digest,
                lane_incidence_replay_digest=incidence_digest,
                landing_site_domain_digest=(
                    ledger.landing_site_domain_digest
                ),
                landing_sites_complete=member_complete,
                landing_site_conflict_domain=member_domain,
                owned_conductor_components=certificates,
            )
            deadline.guard("member-candidate-pool-replay")
        except _DiagnosticAggregateDeadlineExhausted:
            raise
        except CandidatePoolLimitError as exc:
            record_failure(
                f"{candidate_ref}:{type(exc).__name__}:"
                f"{str(exc)[:160]}"
            )
            projection_exhaustive = False
            continue
        deadline.guard("member-projection-verification")
        verify_partition_projection(
            candidate_ref,
            net,
            (source_candidate,),
            (),
            (),
            member_pool,
            member_site_ids,
            member_sites,
            member_catalog,
            member_conflicts,
            member_domain,
            member_ledger_hash,
            goals,
            member_complete,
        )
        deadline.guard("member-projection-verification")
        result = select_complete_net_candidates(
            member_pool,
            max_deterministic_time=deadline.solver_budget(
                "member-selector",
                hard_limit=(
                    ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME
                ),
            ),
        )
        deadline.guard("member-selector")
        if result.status not in {"SAT", "FEASIBLE"} or result.plan is None:
            record_failure(
                f"{candidate_ref}:{result.status}:{result.solver_status}"
            )
            projection_exhaustive = False
            continue
        verify_route_plan_verification_receipt(
            member_pool,
            result.plan,
            result.verification_receipt,
        )
        deadline.guard("member-plan-receipt")
        if (
            len(result.plan.selected_nets) != 1
            or result.plan.selected_nets[0].candidate_id
            != source_candidate.candidate_id
        ):
            raise CandidatePoolError(
                "diagnostic member plan does not replay to its source family"
            )
        winner_family_refs.add(ref)
        pool_hash = member_pool.sha256()
        plan_hash = result.plan.sha256()
        member_projection_digest = _diagnostic_member_projection_digest(
            candidate_id=source_candidate.candidate_id,
            member_ledger_hash=member_ledger_hash,
            member_pool_hash=pool_hash,
            member_site_ids=member_site_ids,
            net=net,
            source_family_ref=ref,
            source_pool_hash=source_pool_hash,
        )
        witness_hashes.append((
            member_projection_digest,
            pool_hash,
            plan_hash,
        ))
        local_winners_by_net.setdefault(net, []).append((
            _bringup_partition_plan_rank(result),
            member_projection_digest,
            source_candidate,
            pool_hash,
            plan_hash,
        ))

    successful_family_count = len(winner_family_refs)
    for ref, net, _source_pool_hash, source_candidates, _goals in (
        projected_family_members
    ):
        visited = visited_by_ref[ref]
        if visited >= len(source_candidates):
            continue
        projection_exhaustive = False
        omitted_ids = tuple(
            candidate.candidate_id for candidate in source_candidates[visited:]
        )
        omitted_digest = _diagnostic_omitted_members_digest(
            ref=ref,
            net=net,
            candidate_ids=omitted_ids,
        )
        record_failure(
            f"{ref}:DIAGNOSTIC_MEMBER_PROJECTION_INCOMPLETE:"
            f"visited={visited}:total={len(source_candidates)}:"
            f"omitted_digest={omitted_digest}"
        )

    if not local_winners_by_net:
        raise CandidatePoolError(
            "diagnostic family decomposition produced no local winner"
        )

    def evaluate_merge_trial(
        trial: tuple[CompleteNetCandidate, ...],
    ) -> tuple[CandidatePool, Any] | None:
        trial_nets = tuple(item.net for item in trial)
        trial_site_id_set: set[str] = set()
        for item in trial:
            cached_site_ids = member_site_ids_by_candidate_id.get(
                item.candidate_id
            )
            if cached_site_ids is None:
                raise CandidatePoolError(
                    "diagnostic merge candidate lacks a verified site projection"
                )
            for site_id in cached_site_ids:
                deadline.guard("merge-site-id-union")
                if site_id in trial_site_id_set:
                    continue
                if len(trial_site_id_set) >= MAX_TOTAL_ASSIGNMENT_ARCS:
                    raise CandidatePoolLimitError(
                        "diagnostic merge site reference union exceeds "
                        f"{MAX_TOTAL_ASSIGNMENT_ARCS}"
                    )
                trial_site_id_set.add(site_id)
        trial_site_ids = tuple(sorted(trial_site_id_set))
        (
            merge_sites,
            merge_catalog,
            merge_conflicts,
            merge_domain,
            merge_ledger_hash,
            merge_complete,
        ) = projected_site_bundle(trial_site_ids)
        merge_goals = tuple(
            goal
            for trial_net in trial_nets
            for goal in goals_by_net.get(trial_net, ())
        )
        deadline.guard("merge-candidate-pool-replay")
        trial_pool = CandidatePool(
            placement_provenance_digest=(
                routing_world.placement_provenance_digest
            ),
            topology_provenance_digest=topology_digest,
            required_nets=trial_nets,
            required_goal_refs=tuple(
                goal.declaration_ref for goal in merge_goals
            ),
            resources=resources,
            landing_sites=merge_sites,
            candidates=trial,
            conflicts=(),
            landing_site_catalog=merge_catalog,
            landing_site_conflicts=merge_conflicts,
            required_route_goals=merge_goals,
            pdk_provenance_digest=routing_world.pdk_provenance_digest,
            routing_world_provenance_digest=world_digest,
            routing_resource_ledger_hash=merge_ledger_hash,
            owned_conductor_index_digest=conductor_digest,
            lane_incidence_replay_digest=incidence_digest,
            landing_site_domain_digest=ledger.landing_site_domain_digest,
            landing_sites_complete=merge_complete,
            landing_site_conflict_domain=merge_domain,
            owned_conductor_components=certificates,
        )
        deadline.guard("merge-candidate-pool-replay")
        merge_result = select_complete_net_candidates(
            trial_pool,
            max_deterministic_time=deadline.solver_budget(
                "merge-selector",
                hard_limit=(
                    ROUTE_PLAN_OBJECTIVE_REPLAY_MAX_DETERMINISTIC_TIME
                ),
            ),
        )
        deadline.guard("merge-selector")
        if merge_result.status == "INFEASIBLE":
            deadline.guard("merge-infeasibility-verifier")
            verify_selection_certificate(trial_pool, merge_result.certificate)
            deadline.guard("merge-infeasibility-verifier")
            return None
        if merge_result.status == "UNKNOWN":
            raise CandidatePoolLimitError(
                "diagnostic exact merge selector exhausted its bounded solve: "
                f"{merge_result.solver_status}"
            )
        if (
            merge_result.status not in {"SAT", "FEASIBLE"}
            or merge_result.plan is None
        ):
            raise CandidatePoolError(
                "diagnostic exact merge selector returned malformed status "
                f"{merge_result.status}:{merge_result.solver_status}"
            )
        verify_route_plan_verification_receipt(
            trial_pool,
            merge_result.plan,
            merge_result.verification_receipt,
        )
        deadline.guard("merge-plan-receipt")
        return trial_pool, merge_result.plan

    merge_search = _diagnostic_exact_frontier_search(
        local_winners_by_net,
        evaluate=evaluate_merge_trial,
        max_solves=MAX_DIAGNOSTIC_MERGE_SOLVES,
        deadline_reached=deadline.expired,
    )
    if merge_search.budget_exhausted:
        record_failure(
            f"merge:{merge_search.stop_code}:"
            f"solves={merge_search.solve_count}:"
            f"{merge_search.stop_detail[:160]}"
        )
    if merge_search.witness is None:
        error_type = (
            CandidatePoolLimitError
            if merge_search.budget_exhausted
            else CandidatePoolError
        )
        raise error_type(
            "diagnostic exact merge produced no verified frontier"
        )
    final_merge_pool, final_merge_plan = merge_search.witness
    accepted = list(merge_search.selected)
    full_source_frontier = (
        merge_search.full_frontier and projection_exhaustive
    )
    if not merge_search.full_frontier:
        record_failure(
            "merge:"
            + (
                "DIAGNOSTIC_BOUNDED_EXACT_SEARCH_PARTIAL"
                if merge_search.budget_exhausted
                else "DIAGNOSTIC_EXACT_MAX_CARDINALITY_PARTIAL"
            )
            + f":covered={len(accepted)},"
            f"frontier_nets={len(local_winners_by_net)}"
        )

    merge_policy = (
        "DETERMINISTIC_EXACT_FULL_FRONTIER"
        if full_source_frontier
        else (
            "DETERMINISTIC_BOUNDED_EXACT_SEARCH_PARTIAL"
            if merge_search.budget_exhausted or not projection_exhaustive
            else "DETERMINISTIC_EXACT_MAX_CARDINALITY_PARTIAL"
        )
    )
    selected_nets = tuple(item.net for item in accepted)
    unrouted_nets = tuple(sorted(
        set(routing_world.required_nets) - set(selected_nets)
    ))
    failed_refs = failure_evidence.refs()
    frontier_digest = _diagnostic_frontier_digest(
        failed_partition_refs=failed_refs,
        family_projection_digest=verified_projection_digest,
        merge_plan_hash=final_merge_plan.sha256(),
        merge_pool_hash=final_merge_pool.sha256(),
        merge_policy=merge_policy,
        merge_search_exhaustive=(
            merge_search.search_exhaustive and projection_exhaustive
        ),
        merge_solve_count=merge_search.solve_count,
        source_projection_exhaustive=projection_exhaustive,
        selected_nets=selected_nets,
        source_stop_digest=source_unknown.sha256(),
        source_witness_hashes=tuple(witness_hashes),
    )
    diagnostic_programs = tuple(
        replace(program, candidate_pool_hash=frontier_digest)
        for program in final_merge_plan.allocated_programs
    )
    index_certificate = final_merge_plan.owned_conductor_index_certificate
    if index_certificate is None:
        raise CandidatePoolError(
            "diagnostic merge plan lacks owned-conductor index"
        )
    route_program = DiagnosticRouteProgramBundle(
        frontier_digest=frontier_digest,
        source_stop_digest=source_unknown.sha256(),
        family_projection_digest=verified_projection_digest,
        selected_nets=final_merge_plan.selected_nets,
        allocated_programs=diagnostic_programs,
        owned_conductor_index_certificate=index_certificate,
        source_witness_hashes=tuple(witness_hashes),
        failed_partition_refs=tuple(failed_refs),
        landing_site_conflict_domain_digest=(
            final_merge_plan.landing_site_conflict_domain_digest
        ),
        lazy_landing_site_conflicts=(
            final_merge_plan.lazy_landing_site_conflicts
        ),
    )
    return DiagnosticDecomposedCompileResult(
        route_program=route_program,
        source_unknown=source_unknown,
        family_count=len(family_specs),
        family_winner_count=successful_family_count,
        failed_family_refs=tuple(failed_refs),
        unrouted_nets=unrouted_nets,
    )


def compile_fixed_pattern_candidate_pool(
    topology_plan: TopologyPlan,
    routing_world: RoutingWorldProvenance,
    ledger: ExplicitRoutingResourceLedger,
    conductor_index: OwnedConductorComponentIndex,
    tech: Any,
    *,
    incidence_index: LaneIncidenceIndex | None = None,
    max_pattern_expansions: int = DEFAULT_PATTERN_EXPANSIONS,
    max_landing_option_templates: int = DEFAULT_PATTERN_LANDING_OPTION_TEMPLATES,
    max_candidates_per_net: int = MAX_FIXED_PATTERN_CANDIDATES_PER_NET,
    scope: PatternCompileScope = PatternCompileScope.SIGNOFF,
) -> FixedPatternCompileResult | DiagnosticDecomposedCompileResult:
    """Compile bounded exact fixed motifs without search or route inference."""

    if type(scope) is not PatternCompileScope:
        raise PatternCompileInputError("scope must be an exact PatternCompileScope")

    expansion_limit = _validate_compile_limit(
        "max_pattern_expansions",
        max_pattern_expansions,
        MAX_PATTERN_EXPANSIONS,
    )
    candidate_limit = _validate_compile_limit(
        "max_candidates_per_net",
        max_candidates_per_net,
        MAX_FIXED_PATTERN_CANDIDATES_PER_NET,
    )
    landing_option_limit = _validate_compile_limit(
        "max_landing_option_templates",
        max_landing_option_templates,
        MAX_PATTERN_LANDING_OPTION_TEMPLATES,
    )
    if type(topology_plan) is not TopologyPlan:
        raise PatternCompileInputError("topology_plan must be exact TopologyPlan")
    if type(routing_world) is not RoutingWorldProvenance:
        raise PatternCompileInputError(
            "routing_world must be exact RoutingWorldProvenance"
        )
    if type(ledger) is not ExplicitRoutingResourceLedger:
        raise PatternCompileInputError(
            "ledger must be exact ExplicitRoutingResourceLedger"
        )
    if type(conductor_index) is not OwnedConductorComponentIndex:
        raise PatternCompileInputError(
            "conductor_index must be exact OwnedConductorComponentIndex"
        )
    if incidence_index is not None and type(incidence_index) is not LaneIncidenceIndex:
        raise PatternCompileInputError(
            "incidence_index must be exact LaneIncidenceIndex when supplied"
        )

    topology_plan = TopologyPlan(
        demands=tuple(
            _rebuild_topology_demand(demand)
            for demand in _bounded_materialize(
                topology_plan.demands,
                limit=MAX_TOPOLOGY_PLAN_DEMANDS,
                label="compiler topology demand",
            )
        ),
        candidates=tuple(
            _rebuild_topology_candidate(candidate)
            for candidate in _bounded_materialize(
                topology_plan.candidates,
                limit=MAX_TOPOLOGY_PLAN_CANDIDATES,
                label="compiler topology candidate",
            )
        ),
    )
    routing_world = RoutingWorldProvenance(
        required_nets=_bounded_materialize(
            routing_world.required_nets,
            limit=MAX_PATTERN_WORLD_REQUIRED_NETS,
            label="compiler routing-world required net",
        ),
        placement_provenance_digest=routing_world.placement_provenance_digest,
        topology_provenance_digest=routing_world.topology_provenance_digest,
        pdk_provenance_digest=routing_world.pdk_provenance_digest,
        pins=_bounded_materialize(
            routing_world.pins,
            limit=MAX_PATTERN_WORLD_PINS,
            label="compiler routing-world pin",
        ),
        obstacles=_bounded_materialize(
            routing_world.obstacles,
            limit=MAX_PATTERN_WORLD_OBSTACLES,
            label="compiler routing-world obstacle",
        ),
        rails=_bounded_materialize(
            routing_world.rails,
            limit=MAX_PATTERN_WORLD_RAILS,
            label="compiler routing-world rail",
        ),
        terminal_conductor_refs=_bounded_materialize(
            routing_world.terminal_conductor_refs,
            limit=MAX_PATTERN_WORLD_OBSTACLES,
            label="compiler routing-world terminal conductor ref",
        ),
    )
    ledger = ExplicitRoutingResourceLedger(
        routing_world_provenance_digest=(
            ledger.routing_world_provenance_digest
        ),
        track_resources=_bounded_materialize(
            ledger.track_resources,
            limit=MAX_PATTERN_TRACK_RESOURCES,
            label="compiler ledger track resource",
        ),
        landing_sites=_bounded_materialize(
            ledger.landing_sites,
            limit=MAX_PATTERN_LANDING_SITES,
            label="compiler ledger landing site",
        ),
        landing_site_pair_conflicts=_bounded_materialize(
            ledger.landing_site_pair_conflicts,
            limit=MAX_PATTERN_LEDGER_SITE_CONFLICTS,
            label="compiler ledger site conflict",
        ),
        owned_conductor_index_digest=ledger.owned_conductor_index_digest,
        landing_site_domain_digest=ledger.landing_site_domain_digest,
        landing_sites_complete=ledger.landing_sites_complete,
        landing_site_conflict_domain=ledger.landing_site_conflict_domain,
    )
    if incidence_index is not None:
        incidence_index = LaneIncidenceIndex(
            routing_world_provenance_digest=(
                incidence_index.routing_world_provenance_digest
            ),
            routing_resource_ledger_digest=(
                incidence_index.routing_resource_ledger_digest
            ),
            owned_conductor_index_digest=(
                incidence_index.owned_conductor_index_digest
            ),
            landing_pair_incidence=_bounded_materialize(
                incidence_index.landing_pair_incidence,
                limit=MAX_PATTERN_EXPANSIONS,
                label="compiler incidence landing pair",
            ),
            boundary_lane_incidence=_bounded_materialize(
                incidence_index.boundary_lane_incidence,
                limit=MAX_PATTERN_EXPANSIONS,
                label="compiler incidence boundary lane",
            ),
            lanes=_bounded_materialize(
                incidence_index.lanes,
                limit=MAX_PATTERN_LANES,
                label="compiler incidence lane",
            ),
            site_sides=_bounded_materialize(
                incidence_index.site_sides,
                limit=MAX_PATTERN_SITE_SIDES,
                label="compiler incidence site side",
            ),
            endpoints=_bounded_materialize(
                incidence_index.endpoints,
                limit=MAX_PATTERN_ENDPOINTS,
                label="compiler incidence endpoint",
            ),
        )

    topology_digest = topology_plan.sha256()
    world_digest = routing_world.sha256()
    ledger_digest = ledger.sha256()
    conductor_digest = conductor_index.sha256()
    try:
        tech_digest = pdk_provenance_digest(tech)
    except Exception as exc:
        raise PatternCompileInputError(
            "tech cannot produce an exact PDK routing digest"
        ) from exc
    if routing_world.pdk_provenance_digest != tech_digest:
        raise PatternCompileInputError(
            "routing world PDK digest does not match compiler tech"
        )
    if routing_world.topology_provenance_digest != topology_digest:
        raise PatternCompileInputError(
            "routing world topology digest does not match topology plan"
        )
    if ledger.routing_world_provenance_digest != world_digest:
        raise PatternCompileInputError(
            "resource ledger belongs to a foreign routing world"
        )
    if conductor_index.world_digest != world_digest:
        raise PatternCompileInputError(
            "owned-conductor index belongs to a foreign routing world"
        )
    if ledger.owned_conductor_index_digest != conductor_digest:
        raise PatternCompileInputError(
            "resource ledger owned-conductor digest is stale"
        )
    if tuple(demand.net for demand in topology_plan.demands) != routing_world.required_nets:
        raise PatternCompileInputError(
            "routing world required nets do not match topology plan"
        )
    landing_query_context = None
    if not ledger.landing_sites_complete:
        try:
            landing_query_context = _prepare_exact_landing_site_query_context(
                ledger,
                routing_world,
                tech,
                conductor_index,
            )
        except RoutingResourceEnumerationError as exc:
            raise PatternCompileInputError(
                f"resource ledger failed canonical track-domain replay: {exc}"
            ) from exc

    if incidence_index is None:
        built = build_lane_incidence_index(ledger, conductor_index, tech)
        if isinstance(built, PatternCompileUnknown):
            return FixedPatternCompileResult.from_unknown(built)
        incidence_index = built
    if (
        incidence_index.routing_world_provenance_digest != world_digest
        or incidence_index.routing_resource_ledger_digest != ledger_digest
        or incidence_index.owned_conductor_index_digest != conductor_digest
    ):
        raise PatternCompileInputError(
            "lane incidence index digest binding does not match compile inputs"
        )
    replay_digest = verify_lane_incidence_index_replay(
        incidence_index,
        ledger,
        conductor_index,
        tech,
    )
    if isinstance(replay_digest, PatternCompileUnknown):
        return FixedPatternCompileResult.from_unknown(replay_digest)
    incidence_digest = replay_digest
    lane_layer_index = _BudgetedLaneLayerIndex(incidence_index)
    lazy_landing_access = (
        None
        if ledger.landing_sites_complete
        else _ExactLazyLandingAccess(
            ledger,
            routing_world,
            tech,
            conductor_index,
            incidence_index,
            landing_query_context,
            lane_layer_index,
        )
    )
    layer_n_by_key = _layer_number_map(tech)
    missing_component_layer_keys = tuple(
        sorted(
            {
                component.layer_key
                for component in conductor_index.components
                if component.layer_key not in layer_n_by_key
            }
        )
    )
    if missing_component_layer_keys:
        return _compile_unknown(
            "MISSING_COMPONENT_LAYER",
            topology_digest=topology_digest,
            world_digest=world_digest,
            ledger_digest=ledger_digest,
            conductor_digest=conductor_digest,
            incidence_digest=incidence_digest,
            refs=tuple(str(key) for key in missing_component_layer_keys),
        )

    component_pair_landing_access = _CertifiedComponentPairLandingAccess(
        ledger,
        incidence_index,
    )
    compiled: list[CompleteNetCandidate] = []
    candidate_families: list[CompleteNetCandidateFamily] = []
    flat_projection_families: list[CompleteNetCandidateFamily] = []
    deferred_factorization_groups: list[
        tuple[CompleteNetCandidate, ...]
    ] = []
    factorized_candidate_families: list[
        FactorizedCompleteNetCandidateFamily
    ] = []
    site_projection_ids: set[str] = set()
    current_net: str | None = None
    # These are whole-plan proof budgets.  The plan already has a bounded net
    # count, but resetting a large allowance for every net multiplies runtime
    # and permits an intermediate option collection larger than the closed IR.
    budget = _IncidenceBudget(expansion_limit)
    option_budget = _IncidenceBudget(landing_option_limit)
    try:
        for demand in topology_plan.demands:
            current_net = demand.net
            demand_incidence_index = _incidence_index_for_net(
                incidence_index, demand.net
            )
            demand_lane_layer_index = (
                lane_layer_index
                if demand_incidence_index is incidence_index
                else _BudgetedLaneLayerIndex(
                    demand_incidence_index,
                    shared_adjacency_index=lane_layer_index,
                )
            )
            demand_lazy_landing_access = (
                lazy_landing_access
                if demand_incidence_index is incidence_index
                else (
                    None
                    if ledger.landing_sites_complete
                    else _ExactLazyLandingAccess(
                        ledger,
                        routing_world,
                        tech,
                        conductor_index,
                        demand_incidence_index,
                        landing_query_context,
                        demand_lane_layer_index,
                        shared_sites=lazy_landing_access._sites,
                    )
                )
            )
            demand_component_pair_landing_access = (
                component_pair_landing_access
                if demand_incidence_index is incidence_index
                else _CertifiedComponentPairLandingAccess(
                    ledger, demand_incidence_index
                )
            )
            net_candidates: list[CompleteNetCandidate] = []
            local_candidates: list[CompleteNetCandidate] = []
            local_family_state = _SharedSignalFamilyBudgetState(
                _SharedSignalRawOptionBudget(budget, option_budget)
            )
            power_candidates: list[CompleteNetCandidate] = []
            power_family_state = _SharedSignalFamilyBudgetState(
                _SharedSignalRawOptionBudget(budget, option_budget)
            )
            specific_unknown: tuple[str, tuple[str, ...]] | None = None
            for topology_candidate in topology_plan.candidates_for(demand.net):
                if topology_candidate.topology_class == "power_spine_tree":
                    compiled_power_candidates, power_unknown = (
                        _compile_power_spine_candidate(
                        topology_candidate=topology_candidate,
                        demand=demand,
                        routing_world=routing_world,
                        ledger=ledger,
                        conductor_index=conductor_index,
                        incidence_index=demand_incidence_index,
                        layer_n_by_key=layer_n_by_key,
                        tech=tech,
                        topology_digest=topology_digest,
                        budget=budget,
                        option_budget=power_family_state.raw_option_budget,
                        component_pair_landing_access=(
                            demand_component_pair_landing_access
                        ),
                        lazy_landing_access=demand_lazy_landing_access,
                        lane_layer_index=demand_lane_layer_index,
                        )
                    )
                    power_candidates.extend(compiled_power_candidates)
                    if power_unknown is not None:
                        specific_unknown = power_unknown
                    continue
                if topology_candidate.topology_class == "shared_signal_trunk":
                    net_candidates.extend(
                        _compile_shared_signal_star_candidate(
                            topology_candidate=topology_candidate,
                            demand=demand,
                            routing_world=routing_world,
                            ledger=ledger,
                            conductor_index=conductor_index,
                            incidence_index=demand_incidence_index,
                            layer_n_by_key=layer_n_by_key,
                            tech=tech,
                            topology_digest=topology_digest,
                            budget=budget,
                            option_budget=option_budget,
                            lazy_landing_access=demand_lazy_landing_access,
                            lane_layer_index=demand_lane_layer_index,
                        )
                    )
                    continue
                # These are alternative complete programs for simultaneous
                # allocation, not dominance-ordered fallbacks.  A direct path
                # may conflict globally while a lift or dogleg remains legal;
                # retain the full bounded grammar and let the exact CP choose.
                motif_candidates = (
                    _compile_same_component_candidate(
                        topology_candidate=topology_candidate,
                        demand=demand,
                        routing_world=routing_world,
                        conductor_index=conductor_index,
                        incidence_index=demand_incidence_index,
                        layer_n_by_key=layer_n_by_key,
                        topology_digest=topology_digest,
                        budget=budget,
                    )
                    + _compile_direct_candidate(
                        topology_candidate=topology_candidate,
                        demand=demand,
                        routing_world=routing_world,
                        conductor_index=conductor_index,
                        incidence_index=demand_incidence_index,
                        layer_n_by_key=layer_n_by_key,
                        tech=tech,
                        topology_digest=topology_digest,
                        budget=budget,
                    )
                    + _compile_lift_candidate(
                        topology_candidate=topology_candidate,
                        demand=demand,
                        routing_world=routing_world,
                        ledger=ledger,
                        conductor_index=conductor_index,
                        incidence_index=demand_incidence_index,
                        layer_n_by_key=layer_n_by_key,
                        tech=tech,
                        topology_digest=topology_digest,
                        budget=budget,
                        option_budget=local_family_state.raw_option_budget,
                        lazy_landing_access=demand_lazy_landing_access,
                    )
                    + _compile_dogleg_candidate(
                        topology_candidate=topology_candidate,
                        demand=demand,
                        routing_world=routing_world,
                        ledger=ledger,
                        conductor_index=conductor_index,
                        incidence_index=demand_incidence_index,
                        layer_n_by_key=layer_n_by_key,
                        tech=tech,
                        topology_digest=topology_digest,
                        budget=budget,
                        option_budget=local_family_state.raw_option_budget,
                        lazy_landing_access=demand_lazy_landing_access,
                        lane_layer_index=demand_lane_layer_index,
                    )
                )
                local_candidates.extend(motif_candidates)
            if power_candidates:
                unique_power = {
                    candidate.candidate_id: candidate
                    for candidate in power_candidates
                }
                ordered_power = tuple(
                    unique_power[candidate_id]
                    for candidate_id in sorted(unique_power)
                )
                power_family_state.family_count = len({
                    _shared_signal_family_view(candidate)[0]
                    for candidate in ordered_power
                })
                merged_power = _merge_shared_signal_candidate_families(
                    ordered_power,
                    ledger=ledger,
                    lazy_landing_access=demand_lazy_landing_access,
                )
                merged_power = tuple(
                    _factor_final_candidate_landing_domains(
                        candidate,
                        ledger=ledger,
                        lazy_landing_access=demand_lazy_landing_access,
                    )
                    for candidate in merged_power
                )
                net_candidates.extend(
                    _debit_final_shared_signal_candidates(
                        merged_power,
                        option_budget=option_budget,
                        family_state=power_family_state,
                        bundle_kind="power_final_families",
                    )
                )
            if local_candidates:
                unique_local = {
                    candidate.candidate_id: candidate
                    for candidate in local_candidates
                }
                ordered_local = tuple(
                    unique_local[candidate_id]
                    for candidate_id in sorted(unique_local)
                )
                local_family_state.family_count = len({
                    _shared_signal_family_view(candidate)[0]
                    for candidate in ordered_local
                })
                merged_local = _merge_shared_signal_candidate_families(
                    ordered_local,
                    ledger=ledger,
                    lazy_landing_access=demand_lazy_landing_access,
                )
                merged_local = tuple(
                    _factor_final_candidate_landing_domains(
                        candidate,
                        ledger=ledger,
                        lazy_landing_access=demand_lazy_landing_access,
                    )
                    for candidate in merged_local
                )
                net_candidates.extend(
                    _debit_final_shared_signal_candidates(
                        merged_local,
                        option_budget=option_budget,
                        family_state=local_family_state,
                        bundle_kind="local_signal_final_families",
                    )
                )
            unique = {
                candidate.candidate_id: candidate for candidate in net_candidates
            }
            ordered = tuple(unique[candidate_id] for candidate_id in sorted(unique))
            interval_free_groups: dict[
                tuple[Any, ...], list[CompleteNetCandidate]
            ] = {}
            for candidate in ordered:
                family_key = _shared_signal_family_view(
                    candidate, include_intervals=False
                )[0]
                interval_free_groups.setdefault(family_key, []).append(candidate)
            sorted_group_items = tuple(sorted(
                interval_free_groups.items(), key=lambda item: repr(item[0])
            ))
            if len(sorted_group_items) > candidate_limit:
                candidate_shapes: dict[tuple[str, int, int], int] = {}
                for candidate in ordered:
                    shape = (
                        candidate.program_kind,
                        len(candidate.corridor_uses),
                        len(candidate.landing_demands),
                    )
                    candidate_shapes[shape] = candidate_shapes.get(shape, 0) + 1
                return _compile_unknown(
                    "CANDIDATE_LIMIT_EXCEEDED",
                    topology_digest=topology_digest,
                    world_digest=world_digest,
                    ledger_digest=ledger_digest,
                    conductor_digest=conductor_digest,
                    incidence_digest=incidence_digest,
                    refs=(
                        demand.net,
                        f"candidate_count={len(ordered)}",
                        f"max_candidate_families_per_net={candidate_limit}",
                        (
                            "interval_free_family_count="
                            f"{len(sorted_group_items)}"
                        ),
                        "candidate_shape=" + ";".join(
                            f"{program_kind}:uses={use_count}:"
                            f"landings={landing_count}:count={count}"
                            for (
                                program_kind,
                                use_count,
                                landing_count,
                            ), count in sorted(candidate_shapes.items())
                        ),
                    ),
                )
            if not ordered:
                if specific_unknown is not None:
                    code, refs = specific_unknown
                    return _compile_unknown(
                        code,
                        topology_digest=topology_digest,
                        world_digest=world_digest,
                        ledger_digest=ledger_digest,
                        conductor_digest=conductor_digest,
                        incidence_digest=incidence_digest,
                        refs=refs,
                    )
                return _compile_unknown(
                    "NO_FIXED_PATTERN",
                    topology_digest=topology_digest,
                    world_digest=world_digest,
                    ledger_digest=ledger_digest,
                    conductor_digest=conductor_digest,
                    incidence_digest=incidence_digest,
                    refs=(demand.net,),
                )
            site_projection_ids.update(
                site_id
                for candidate in ordered
                for demand in candidate.landing_demands
                for site_id in demand.allowed_site_ids
            )
            site_projection_ids.update(
                site_id
                for candidate in ordered
                for option in candidate.landing_options
                for site_id in option.site_ids
            )
            if len(ordered) <= candidate_limit:
                compiled.extend(ordered)
                for _family_key, members in sorted_group_items:
                    ordered_members = tuple(sorted(
                        members, key=lambda candidate: candidate.candidate_id
                    ))
                    flat_projection_families.append(
                        CompleteNetCandidateFamily.build(
                            net=members[0].net,
                            topology_class=members[0].topology_class,
                            variant_candidate_ids=tuple(
                                member.candidate_id for member in members
                            ),
                        )
                    )
                    if all(
                        candidate.corridor_uses
                        and candidate.landing_demands
                        and not candidate.landing_options
                        and all(
                            demand.selection_mode
                            == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                            for demand in candidate.landing_demands
                        )
                        for candidate in ordered_members
                    ):
                        deferred_factorization_groups.append(ordered_members)
                continue

            available_sites = {
                site.site_id: site for site in ledger.landing_sites
            }
            if demand_lazy_landing_access is not None:
                available_sites.update(demand_lazy_landing_access._sites)
            for _family_key, members in sorted_group_items:
                ordered_members = tuple(sorted(
                    members, key=lambda candidate: candidate.candidate_id
                ))
                can_factor_exact_profiles = all(
                    candidate.corridor_uses
                    and candidate.landing_demands
                    and not candidate.landing_options
                    and all(
                        demand.selection_mode
                        == LANDING_SELECTION_DERIVED_EXACT_SITE_DOMAIN
                        for demand in candidate.landing_demands
                    )
                    for candidate in ordered_members
                )
                if can_factor_exact_profiles:
                    factorized_candidate_families.append(
                        factorize_complete_net_candidate_family(
                            ordered_members,
                            resources=ledger.track_resources,
                            landing_sites=_factorization_landing_sites(
                                ordered_members,
                                available_sites,
                                routing_world_provenance_digest=world_digest,
                                landing_site_domain_digest=(
                                    ledger.landing_site_domain_digest
                                ),
                            ),
                        )
                    )
                else:
                    legacy_family = CompleteNetCandidateFamily.build(
                        net=ordered_members[0].net,
                        topology_class=ordered_members[0].topology_class,
                        variant_candidate_ids=tuple(
                            member.candidate_id
                            for member in ordered_members
                        ),
                    )
                    candidate_families.append(legacy_family)
                    flat_projection_families.append(legacy_family)
                    compiled.extend(ordered_members)
    except _IncidenceBudgetExceeded as exc:
        if exc.reason in {
            "boundary landing-option expansion",
            "component-pair landing-option expansion",
            "internal landing-option expansion",
            "exact-family final landing-option expansion",
        }:
            return _compile_unknown(
                "LANDING_OPTION_BUDGET_EXHAUSTED",
                topology_digest=topology_digest,
                world_digest=world_digest,
                ledger_digest=ledger_digest,
                conductor_digest=conductor_digest,
                incidence_digest=incidence_digest,
                refs=(
                    *(
                        ()
                        if current_net is None
                        else (f"current_net={current_net}",)
                    ),
                    exc.reason,
                    (
                        "max_landing_option_templates="
                        f"{landing_option_limit}"
                    ),
                    f"calls={option_budget.option_bundle_calls}",
                    f"sites={option_budget.option_bundle_sites}",
                    f"k={option_budget.option_bundle_k}",
                    f"options={option_budget.option_bundle_options}",
                    f"attempted_templates={option_budget.used}",
                    f"raw_expansions={option_budget.shared_raw_expansions}",
                    (
                        "final_unique_templates="
                        f"{option_budget.shared_final_unique_templates}"
                    ),
                    f"family_count={option_budget.shared_family_count}",
                    (
                        "option_bundle_digest="
                        f"{option_budget.option_bundle_digest}"
                    ),
                    f"last_bundle_kind={option_budget.last_bundle_kind}",
                    (
                        "last_bundle_context="
                        f"{option_budget.last_bundle_context}"
                    ),
                    (
                        "last_bundle_site_count="
                        f"{option_budget.last_bundle_site_count}"
                    ),
                    (
                        "last_bundle_site_digest="
                        f"{option_budget.last_bundle_site_digest}"
                    ),
                    f"last_bundle_k={option_budget.last_bundle_k}",
                    (
                        "last_bundle_option_count="
                        f"{option_budget.last_bundle_option_count}"
                    ),
                ),
            )
        return _compile_unknown(
            "PATTERN_BUDGET_EXHAUSTED",
            topology_digest=topology_digest,
            world_digest=world_digest,
            ledger_digest=ledger_digest,
            conductor_digest=conductor_digest,
            incidence_digest=incidence_digest,
            refs=(
                *(
                    ()
                    if current_net is None
                    else (f"current_net={current_net}",)
                ),
                exc.reason,
                f"max_pattern_expansions={expansion_limit}",
                "expansion_reason_counts=" + ";".join(
                    f"{reason}:{count}"
                    for reason, count in sorted(
                        budget.reason_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:8]
                ),
                *tuple(
                    f"expansion_reason_rank_{rank}={reason}:{count}"
                    for rank, (reason, count) in enumerate(
                        sorted(
                            budget.reason_counts.items(),
                            key=lambda item: (-item[1], item[0]),
                        )[:8],
                        start=1,
                    )
                ),
                f"raw_expansions={option_budget.shared_raw_expansions}",
                (
                    "final_unique_templates="
                    f"{option_budget.shared_final_unique_templates}"
                ),
                f"family_count={option_budget.shared_family_count}",
            ),
        )
    except CandidatePoolError as exc:
        return _compile_unknown(
            "NO_VALID_FIXED_PATTERN",
            topology_digest=topology_digest,
            world_digest=world_digest,
            ledger_digest=ledger_digest,
            conductor_digest=conductor_digest,
            incidence_digest=incidence_digest,
            refs=(type(exc).__name__, str(exc)),
        )
    except (PatternCompileLimitError, RoutingResourceEnumerationError) as exc:
        return _compile_unknown(
            "LANDING_DOMAIN_QUERY_FAILED",
            topology_digest=topology_digest,
            world_digest=world_digest,
            ledger_digest=ledger_digest,
            conductor_digest=conductor_digest,
            incidence_digest=incidence_digest,
            refs=(type(exc).__name__, str(exc)),
        )

    certificates = tuple(
        OwnedConductorComponentCertificate(
            layer_n=layer_n_by_key[component.layer_key],
            component=component,
        )
        for component in conductor_index.components
    )
    pool_resources = ledger.track_resources
    pool_sites = ledger.landing_sites
    pool_site_conflicts = ledger.landing_site_pair_conflicts
    pool_site_conflict_domain = ledger.landing_site_conflict_domain
    pool_landing_site_catalog: LandingSiteCatalog | None = None
    pool_ledger_hash = ledger.sha256()
    if lazy_landing_access is not None:
        try:
            projected_sites = lazy_landing_access.referenced_site_ids(
                site_projection_ids
            )
            if len(projected_sites) <= MAX_PATTERN_LANDING_SITE_PAGE_SIZE:
                (
                    projected_sites,
                    projected_conflicts,
                    projected_conflict_domain,
                ) = replay_exact_landing_site_subset_compressed(
                    ledger,
                    routing_world,
                    tech,
                    conductor_index,
                    projected_sites,
                )
                pool_sites = projected_sites
            else:
                (
                    pool_landing_site_catalog,
                    projected_conflicts,
                    projected_conflict_domain,
                ) = replay_exact_landing_site_catalog_compressed(
                    ledger,
                    routing_world,
                    tech,
                    conductor_index,
                    projected_sites,
                )
                pool_sites = ()
        except (PatternCompileInputError, RoutingResourceEnumerationError) as exc:
            return _compile_unknown(
                "LANDING_DOMAIN_REPLAY_FAILED",
                topology_digest=topology_digest,
                world_digest=world_digest,
                ledger_digest=ledger_digest,
                conductor_digest=conductor_digest,
                incidence_digest=incidence_digest,
                refs=(type(exc).__name__, str(exc)),
            )
        pool_site_conflicts = projected_conflicts
        pool_site_conflict_domain = projected_conflict_domain
        if pool_landing_site_catalog is None:
            pool_ledger = ExplicitRoutingResourceLedger(
                routing_world_provenance_digest=world_digest,
                track_resources=pool_resources,
                landing_sites=pool_sites,
                landing_site_pair_conflicts=pool_site_conflicts,
                owned_conductor_index_digest=conductor_digest,
                landing_site_domain_digest=ledger.landing_site_domain_digest,
                landing_sites_complete=False,
                landing_site_conflict_domain=pool_site_conflict_domain,
            )
            pool_ledger_hash = pool_ledger.sha256()
        else:
            pool_ledger_hash = paged_routing_resource_ledger_sha256(
                world_digest,
                pool_resources,
                pool_landing_site_catalog,
                pool_site_conflicts,
                conductor_digest,
                ledger.landing_site_domain_digest,
                False,
                pool_site_conflict_domain,
            )
    try:
        if len(compiled) > MAX_TOTAL_CANDIDATES:
            compiled_ids_to_remove: set[str] = set()
            factorized_group_id_sets: list[frozenset[str]] = []
            pool_sites_by_id = {
                site.site_id: site
                for site in (
                    pool_landing_site_catalog.iter_sites()
                    if pool_landing_site_catalog is not None
                    else pool_sites
                )
            }
            for members in sorted(
                deferred_factorization_groups,
                key=lambda group: (
                    -len(group), tuple(candidate.candidate_id for candidate in group)
                ),
            ):
                if len(compiled) - len(compiled_ids_to_remove) <= MAX_TOTAL_CANDIDATES:
                    break
                factorized_candidate_families.append(
                    factorize_complete_net_candidate_family(
                        members,
                        resources=pool_resources,
                        landing_sites=_factorization_landing_sites(
                            members,
                            pool_sites_by_id,
                            routing_world_provenance_digest=world_digest,
                            landing_site_domain_digest=(
                                ledger.landing_site_domain_digest
                            ),
                        ),
                    )
                )
                member_ids = frozenset(
                    candidate.candidate_id for candidate in members
                )
                compiled_ids_to_remove.update(member_ids)
                factorized_group_id_sets.append(member_ids)
            if compiled_ids_to_remove:
                compiled = [
                    candidate
                    for candidate in compiled
                    if candidate.candidate_id not in compiled_ids_to_remove
                ]
                flat_projection_families = [
                    family
                    for family in flat_projection_families
                    if frozenset(family.variant_candidate_ids)
                    not in factorized_group_id_sets
                ]
        pool = CandidatePool(
            placement_provenance_digest=(
                routing_world.placement_provenance_digest
            ),
            topology_provenance_digest=topology_digest,
            required_nets=routing_world.required_nets,
            required_goal_refs=tuple(
                goal.declaration_ref for goal in topology_plan.required_goals
            ),
            resources=pool_resources,
            landing_sites=pool_sites,
            candidates=tuple(compiled),
            conflicts=(),
            candidate_families=(
                tuple(flat_projection_families)
                if candidate_families or factorized_candidate_families
                else ()
            ),
            factorized_candidate_families=tuple(
                factorized_candidate_families
            ),
            landing_site_catalog=pool_landing_site_catalog,
            landing_site_conflicts=pool_site_conflicts,
            required_route_goals=topology_plan.required_goals,
            pdk_provenance_digest=routing_world.pdk_provenance_digest,
            routing_world_provenance_digest=world_digest,
            routing_resource_ledger_hash=pool_ledger_hash,
            owned_conductor_index_digest=conductor_digest,
            lane_incidence_replay_digest=incidence_digest,
            landing_site_domain_digest=ledger.landing_site_domain_digest,
            landing_sites_complete=(
                ledger.landing_sites_complete
                if lazy_landing_access is None
                else False
            ),
            landing_site_conflict_domain=(
                pool_site_conflict_domain
            ),
            owned_conductor_components=certificates,
        )
    except CandidatePoolError as exc:
        stopped = _compile_unknown(
            "NO_VALID_FIXED_PATTERN",
            topology_digest=topology_digest,
            world_digest=world_digest,
            ledger_digest=ledger_digest,
            conductor_digest=conductor_digest,
            incidence_digest=incidence_digest,
            refs=(type(exc).__name__, str(exc)),
        )
        if (
            scope is not PatternCompileScope.BRINGUP_DIAGNOSTIC
            or not isinstance(exc, CandidatePoolLimitError)
        ):
            return stopped
        try:
            return _build_diagnostic_decomposed_merge_pool(
                source_unknown=stopped.unknown,
                compiled=tuple(compiled),
                flat_families=tuple(flat_projection_families),
                factorized_families=tuple(factorized_candidate_families),
                routing_world=routing_world,
                topology_plan=topology_plan,
                ledger=ledger,
                conductor_index=conductor_index,
                tech=tech,
                topology_digest=topology_digest,
                world_digest=world_digest,
                conductor_digest=conductor_digest,
                incidence_digest=incidence_digest,
                resources=pool_resources,
                sites=pool_sites,
                site_catalog=pool_landing_site_catalog,
                site_conflicts=pool_site_conflicts,
                site_conflict_domain=pool_site_conflict_domain,
                ledger_hash=pool_ledger_hash,
                landing_sites_complete=(
                    ledger.landing_sites_complete
                    if lazy_landing_access is None
                    else False
                ),
                certificates=certificates,
            )
        except (
            CandidatePoolError,
            InfeasibilityCertificateVerificationError,
            RouteResourcePlanVerificationError,
            RoutingResourceEnumerationError,
        ) as diag_exc:
            return _compile_unknown(
                "DIAGNOSTIC_DECOMPOSITION_FAILED",
                topology_digest=topology_digest,
                world_digest=world_digest,
                ledger_digest=ledger_digest,
                conductor_digest=conductor_digest,
                incidence_digest=incidence_digest,
                refs=(
                    f"source_stop={stopped.unknown.sha256()}",
                    type(diag_exc).__name__,
                    str(diag_exc),
                ),
            )
    return FixedPatternCompileResult(status="COMPLETE", candidate_pool=pool)


__all__ = [
    "BoundaryLaneIncidence",
    "DiagnosticDecomposedCompileResult",
    "EndpointIncidence",
    "FixedPatternCompileResult",
    "LandingPairIncidence",
    "LaneIncidenceIndex",
    "LaneRef",
    "MAX_INCIDENCE_EVALUATIONS",
    "MAX_PATTERN_ENDPOINT_ACCESS_REFS",
    "MAX_PATTERN_ENDPOINT_CONDUCTOR_REFS",
    "MAX_PATTERN_ENDPOINTS",
    "MAX_PATTERN_LANDING_SITES",
    "MAX_PATTERN_LANES",
    "MAX_PATTERN_RAW_SIDE_COMPONENT_REFS",
    "MAX_PATTERN_SITE_OWNER_COMPONENT_REFS",
    "MAX_PATTERN_SITE_SIDES",
    "MAX_PATTERN_SITE_TRACK_CHOICES",
    "MAX_PATTERN_SITE_TRAVERSABLE_REFS",
    "MAX_PATTERN_TRACK_RESOURCES",
    "MAX_PATTERN_UNKNOWN_INPUT_DIGESTS",
    "MAX_PATTERN_UNKNOWN_REFS",
    "PatternCompileInputError",
    "PatternCompileLimitError",
    "PatternCompileScope",
    "PatternCompileUnknown",
    "SiteSideIncidence",
    "build_lane_incidence_index",
    "canonical_pattern_hash",
    "canonical_pattern_id",
    "compile_fixed_pattern_candidate_pool",
    "verify_lane_incidence_index_replay",
]
