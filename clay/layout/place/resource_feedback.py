"""Bounded, proof-gated route-resource feedback contracts for placement.

This module does not run a placement solver.  It lowers replayed Task-5
conflicts either to a pre-signed placement-causal repair or to the deliberately
weaker ``EXACT_PLACEMENT_NOGOOD``: exclude only the current final-master origin
vector bound to the exact pool/world/state.  The no-good makes no corridor-gap
or landing-causality claim.  Nominal observations, generic CP UNSAT, UNKNOWN,
and scalar padding requests remain unlowerable.

A :class:`PlacementSolveCut` is therefore a finite construction request, never a
claim that a placement is capacity-safe.  That claim still requires Task 6 to
materialize explicit tracks/sites and the exact emitted artifact to pass its
authoritative checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Optional, Protocol, TYPE_CHECKING, runtime_checkable

from clay.layout.place.final_placement_state import (
    PlacementCausalityBasis,
    PlacementCorridorBoundary,
    build_placement_causality_basis,
    verify_placement_causality_basis,
)

if TYPE_CHECKING:
    from clay.layout.place_blocks_2d import FinalPlacementCertificate

from clay.layout.route.resource_plan import (
    CandidatePool,
    CapacityConflict,
    CapacityObservation,
    CompleteNetSelectionResult,
    CorridorLayerKey,
    HallDeficit,
    LandingExpansionResource,
    MAX_TRACK_RESOURCES,
    PoolUnsatCertificate,
    verify_infeasibility_certificate,
)


_BBoxNm = tuple[int, int, int, int]
_MAX_LEDGER_CUTS = 64
_MAX_LEDGER_PROGRESS_STATES = 256


def _is_canonical_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def attach_proven_landing_expansion_resources(
    pool: object,
    final_placement_certificate: object,
) -> CandidatePool:
    """Derive conservative Hall-expansion resources at the production seam.

    Only a globally complete landing-site ledger can authorize these resources.
    Each emitted resource is one geometry-complete via-stack lane that exactly
    fills a signed master gap at half-pitch boundaries.  Incomplete or irregular
    domains simply emit no resource, preserving the exact-no-good fallback.
    """

    if not isinstance(pool, CandidatePool):
        raise TypeError("landing expansion compilation requires CandidatePool")
    if not pool.landing_sites_complete or not pool.landing_sites:
        return (
            pool
            if not pool.landing_expansion_resources
            else replace(pool, landing_expansion_resources=())
        )
    verify_placement_causality_basis(final_placement_certificate)
    placement_hash = final_placement_state_digest(final_placement_certificate)
    if pool.placement_provenance_digest != placement_hash:
        raise ValueError("landing expansion pool belongs to a foreign placement")
    basis = final_placement_certificate.placement_causality_basis
    block_boxes = {name: bbox for name, bbox, _members in basis.blocks}
    expansions: list[LandingExpansionResource] = []
    for boundary in basis.corridor_boundaries:
        first, second = (block_boxes[name] for name in boundary.owner_block_ids)
        if boundary.relation == "vertical_separation":
            gap = (
                (first[3], second[1]) if first[3] <= second[1]
                else (second[3], first[1]) if second[3] <= first[1]
                else None
            )
        else:
            gap = (
                (first[2], second[0]) if first[2] <= second[0]
                else (second[2], first[0]) if second[2] <= first[0]
                else None
            )
        if gap is None or gap[1] - gap[0] != boundary.observed_clearance_nm:
            continue
        lanes: dict[tuple[int, int, int], list[object]] = {}
        for site in pool.iter_landing_sites():
            if not (
                site.cut_bbox_nm is not None
                and site.lower_pad_bbox_nm is not None
                and site.upper_pad_bbox_nm is not None
                and site.routing_world_provenance_digest
                == pool.routing_world_provenance_digest
            ):
                continue
            run = (
                site.xy_nm[0]
                if boundary.relation == "vertical_separation"
                else site.xy_nm[1]
            )
            cross = (
                site.xy_nm[1]
                if boundary.relation == "vertical_separation"
                else site.xy_nm[0]
            )
            if not (
                boundary.run_interval_nm[0] <= run <= boundary.run_interval_nm[1]
                and gap[0] < cross < gap[1]
            ):
                continue
            boxes = (site.cut_bbox_nm, site.lower_pad_bbox_nm, site.upper_pad_bbox_nm)
            if boundary.relation == "vertical_separation":
                enclosed = all(
                    boundary.run_interval_nm[0] <= box[0]
                    and box[2] <= boundary.run_interval_nm[1]
                    and gap[0] <= box[1]
                    and box[3] <= gap[1]
                    for box in boxes
                )
            else:
                enclosed = all(
                    gap[0] <= box[0]
                    and box[2] <= gap[1]
                    and boundary.run_interval_nm[0] <= box[1]
                    and box[3] <= boundary.run_interval_nm[1]
                    for box in boxes
                )
            if enclosed:
                lanes.setdefault(
                    (site.lower_layer_n, site.upper_layer_n, run), []
                ).append(site)
        for lane_key, lane_sites in sorted(lanes.items()):
            ordered = tuple(sorted(
                lane_sites,
                key=lambda site: (
                    site.xy_nm[1]
                    if boundary.relation == "vertical_separation"
                    else site.xy_nm[0],
                    site.site_id,
                ),
            ))
            coordinates = tuple(
                site.xy_nm[1]
                if boundary.relation == "vertical_separation"
                else site.xy_nm[0]
                for site in ordered
            )
            if len(coordinates) < 2 or len(coordinates) != len(set(coordinates)):
                continue
            pitches = tuple(
                right - left for left, right in zip(coordinates, coordinates[1:])
            )
            pitch = pitches[0]
            if (
                pitch <= 0
                or any(value != pitch for value in pitches)
                or gap[1] - gap[0] != len(coordinates) * pitch
                or 2 * (coordinates[0] - gap[0]) != pitch
                or 2 * (gap[1] - coordinates[-1]) != pitch
                or any(site.capacity != 1 for site in ordered)
            ):
                continue
            payload = {
                "corridor_relation": boundary.relation,
                "declaration_refs": list(basis.declaration_refs),
                "legal_site_ids": [site.site_id for site in ordered],
                "observed_clearance_nm": boundary.observed_clearance_nm,
                "owner_block_ids": list(boundary.owner_block_ids),
                "placement_basis_sha256": basis.sha256,
                "routing_world_provenance_digest": (
                    pool.routing_world_provenance_digest
                ),
                "run_interval_nm": list(boundary.run_interval_nm),
                "schema": "clay.landing-expansion-resource.v1",
                "site_pitch_nm": pitch,
                "via_stack": list(lane_key[:2]),
                "run_coordinate_nm": lane_key[2],
            }
            expansions.append(LandingExpansionResource(
                expansion_id=f"landing-expansion:{_sha256(payload)}",
                placement_basis_sha256=basis.sha256,
                owner_block_ids=boundary.owner_block_ids,
                corridor_relation=boundary.relation,
                run_interval_nm=boundary.run_interval_nm,
                observed_clearance_nm=boundary.observed_clearance_nm,
                site_pitch_nm=pitch,
                legal_site_ids=tuple(site.site_id for site in ordered),
                routing_world_provenance_digest=pool.routing_world_provenance_digest,
                declaration_refs=basis.declaration_refs,
            ))
    return replace(
        pool,
        landing_expansion_resources=tuple(sorted(expansions))[:MAX_TRACK_RESOURCES],
    )


@runtime_checkable
class FinalPlacementCertificateView(Protocol):
    """Neutral structural view consumed without importing the placement module."""

    blocks: tuple[tuple[str, _BBoxNm], ...]
    bands: tuple[tuple[str, int], ...]
    bbox_nm: _BBoxNm
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
    applied_cut_hashes: tuple[str, ...]
    placement_causality_payloads: tuple[object, ...]
    materialized_block_members: tuple[tuple[str, tuple[str, ...]], ...]
    materialized_block_bboxes: tuple[tuple[str, _BBoxNm], ...]


class PlacementRepairKind(str, Enum):
    """Closed placement actions that a future final master may consume."""

    RELATIVE_BLOCK_ORDER = "relative_block_order"
    CERTIFIED_CHANNEL_SPAN = "certified_channel_span"
    CANDIDATE_LANDING_ADJACENCY = "candidate_landing_adjacency"
    EXACT_PLACEMENT_NOGOOD = "exact_placement_nogood"


@dataclass(frozen=True, order=True)
class PlacementFeedbackScope:
    """One exact final-placement/candidate-pool pair in the feedback loop."""

    candidate_pool_hash: str
    placement_hash: str

    def __post_init__(self) -> None:
        for field_name in ("candidate_pool_hash", "placement_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class PlacementCausalityPayload:
    """Placement/resource facts signed by inclusion in the placement digest.

    The witness carries only :attr:`payload_digest`.  It cannot add or replace
    these facts after the final placement digest has scoped the candidate pool
    and its replayed conflict.
    """

    repair_kind: PlacementRepairKind
    block_bboxes: tuple[tuple[str, _BBoxNm], ...]
    resource: Optional[CorridorLayerKey]
    resource_owner_block_ids: tuple[str, ...]
    preferred_axis: Optional[str]
    corridor_relation: Optional[str]
    observed_clearance_nm: int
    target_required_clearance_nm: int
    active_use_ids: tuple[tuple[str, str], ...]
    capacity_deficit: int
    track_pitch_nm: int
    candidate_ids: tuple[str, ...]
    qualified_demand_ids: tuple[tuple[str, str], ...]
    movable_site_ids: tuple[str, ...]
    declaration_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.repair_kind, PlacementRepairKind):
            raise ValueError("repair_kind must be PlacementRepairKind")
        blocks = _canonical_block_bboxes(self.block_bboxes)
        owners = _sorted_unique_strings(
            self.resource_owner_block_ids, "resource_owner_block_ids"
        )
        unknown_owners = sorted(set(owners) - {name for name, _bbox in blocks})
        if unknown_owners:
            raise ValueError(
                f"resource owners lack signed block bboxes: {unknown_owners}"
            )
        if not owners:
            raise ValueError("resource_owner_block_ids must not be empty")
        if self.resource is not None and not isinstance(self.resource, CorridorLayerKey):
            raise ValueError("resource must be CorridorLayerKey when present")
        if self.preferred_axis not in (None, "H", "V"):
            raise ValueError("preferred_axis must be 'H', 'V', or None")
        if (self.resource is None) != (self.preferred_axis is None):
            raise ValueError("resource and preferred_axis must be present together")
        if self.corridor_relation not in (
            None,
            "horizontal_separation",
            "vertical_separation",
            "channel_span",
        ):
            raise ValueError("corridor_relation is not a closed placement relation")
        for field_name in (
            "observed_clearance_nm",
            "target_required_clearance_nm",
            "capacity_deficit",
            "track_pitch_nm",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{field_name} must be non-negative integer nm")
        candidate_ids = _sorted_unique_strings(self.candidate_ids, "candidate_ids")
        active_use_ids = _sorted_unique_pairs(self.active_use_ids, "active_use_ids")
        qualified = _sorted_unique_pairs(
            self.qualified_demand_ids, "qualified_demand_ids"
        )
        unknown_active_candidates = sorted(
            {candidate_id for candidate_id, _use_id in active_use_ids}
            - set(candidate_ids)
        )
        if unknown_active_candidates:
            raise ValueError("active use candidate ids must be signed candidates")
        unknown_demand_candidates = sorted(
            {candidate_id for candidate_id, _demand_id in qualified}
            - set(candidate_ids)
        )
        if unknown_demand_candidates:
            raise ValueError("qualified demand candidate ids must be signed candidates")
        object.__setattr__(self, "block_bboxes", blocks)
        object.__setattr__(self, "resource_owner_block_ids", owners)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "active_use_ids", active_use_ids)
        object.__setattr__(self, "qualified_demand_ids", qualified)
        object.__setattr__(
            self,
            "movable_site_ids",
            _sorted_unique_strings(self.movable_site_ids, "movable_site_ids"),
        )
        object.__setattr__(
            self,
            "declaration_refs",
            _sorted_unique_strings(self.declaration_refs, "declaration_refs"),
        )

    @property
    def payload_digest(self) -> str:
        return _sha256(_causality_payload(self))


@dataclass(frozen=True)
class PlacementRepairWitness:
    """Typed causal bridge from one replayed pool conflict to movable objects."""

    witness_id: str
    conflict_hash: str
    candidate_pool_hash: str
    placement_hash: str
    repair_kind: PlacementRepairKind
    movable_block_ids: tuple[str, ...]
    movable_site_ids: tuple[str, ...]
    corridor_relation: Optional[str]
    required_clearance_nm: int
    declaration_refs: tuple[str, ...] = ()
    causality_payload_digest: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "witness_id",
            "conflict_hash",
            "candidate_pool_hash",
            "placement_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.repair_kind, PlacementRepairKind):
            raise ValueError("repair_kind must be a closed PlacementRepairKind")
        if self.corridor_relation not in (
            None,
            "horizontal_separation",
            "vertical_separation",
            "channel_span",
        ):
            raise ValueError("corridor_relation is not a closed placement relation")
        if (
            not isinstance(self.required_clearance_nm, int)
            or isinstance(self.required_clearance_nm, bool)
            or self.required_clearance_nm < 0
        ):
            raise ValueError("required_clearance_nm must be a non-negative integer nm")
        object.__setattr__(
            self,
            "movable_block_ids",
            _sorted_unique_strings(self.movable_block_ids, "movable_block_ids"),
        )
        object.__setattr__(
            self,
            "movable_site_ids",
            _sorted_unique_strings(self.movable_site_ids, "movable_site_ids"),
        )
        object.__setattr__(
            self,
            "declaration_refs",
            _sorted_unique_strings(self.declaration_refs, "declaration_refs"),
        )
        if self.causality_payload_digest is not None and (
            not isinstance(self.causality_payload_digest, str)
            or not self.causality_payload_digest
        ):
            raise ValueError(
                "causality_payload_digest must be a non-empty string when present"
            )

    @property
    def scope(self) -> PlacementFeedbackScope:
        return PlacementFeedbackScope(
            candidate_pool_hash=self.candidate_pool_hash,
            placement_hash=self.placement_hash,
        )


@dataclass(frozen=True)
class PlacementSolveCut:
    """One replay-bound, placement-causal cut for a future final master."""

    repair_kind: PlacementRepairKind
    conflict_kind: str
    conflict_hash: str
    candidate_pool_hash: str
    placement_hash: str
    witness_id: str
    nets: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    demand_ids: tuple[str, ...]
    qualified_demand_ids: tuple[tuple[str, str], ...]
    movable_block_ids: tuple[str, ...]
    movable_site_ids: tuple[str, ...]
    resource: Optional[CorridorLayerKey]
    corridor_relation: Optional[str]
    required_clearance_nm: int
    declaration_refs: tuple[str, ...]
    routing_world_hash: Optional[str] = None
    incumbent_origins: tuple[tuple[str, int, int], ...] = ()
    placement_basis_sha256: Optional[str] = None
    oriented_owner_block_ids: tuple[str, ...] = ()
    guarded_coordinates: tuple[tuple[str, str, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.repair_kind, PlacementRepairKind):
            raise ValueError("repair_kind must be PlacementRepairKind")
        if self.conflict_kind not in ("capacity_conflict", "hall_deficit"):
            raise ValueError("conflict_kind must name a replayable conflict")
        for field_name in (
            "conflict_hash",
            "candidate_pool_hash",
            "placement_hash",
            "witness_id",
        ):
            if not isinstance(getattr(self, field_name), str) or not getattr(
                self, field_name
            ):
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "nets",
            "candidate_ids",
            "movable_block_ids",
            "movable_site_ids",
            "declaration_refs",
        ):
            object.__setattr__(
                self,
                field_name,
                _sorted_unique_strings(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "demand_ids",
            _sorted_strings_allow_duplicates(self.demand_ids, "demand_ids"),
        )
        object.__setattr__(
            self,
            "qualified_demand_ids",
            _sorted_unique_pairs(self.qualified_demand_ids, "qualified_demand_ids"),
        )
        if self.resource is not None and not isinstance(self.resource, CorridorLayerKey):
            raise ValueError("resource must be CorridorLayerKey when present")
        if (
            not isinstance(self.required_clearance_nm, int)
            or isinstance(self.required_clearance_nm, bool)
            or self.required_clearance_nm < 0
        ):
            raise ValueError("required_clearance_nm must be non-negative integer nm")
        origins = tuple(sorted(self.incumbent_origins))
        if any(
            not isinstance(row, (tuple, list))
            or len(row) != 3
            or not isinstance(row[0], str)
            or not row[0]
            or not isinstance(row[1], int)
            or isinstance(row[1], bool)
            or not isinstance(row[2], int)
            or isinstance(row[2], bool)
            for row in origins
        ) or len({row[0] for row in origins}) != len(origins):
            raise ValueError("incumbent_origins must contain unique exact origin rows")
        object.__setattr__(self, "incumbent_origins", origins)
        oriented = tuple(self.oriented_owner_block_ids)
        guarded = tuple(sorted(self.guarded_coordinates))
        if any(
            not isinstance(row, (tuple, list))
            or len(row) != 3
            or not isinstance(row[0], str)
            or not row[0]
            or row[1] not in ("x", "y")
            or type(row[2]) is not int
            for row in guarded
        ) or len({(row[0], row[1]) for row in guarded}) != len(guarded):
            raise ValueError("guarded_coordinates must contain unique exact axis rows")
        guard_present = bool(
            self.placement_basis_sha256 is not None or oriented or guarded
        )
        if guard_present:
            if (
                not _is_canonical_sha256(self.placement_basis_sha256)
                or len(oriented) != 2
                or len(set(oriented)) != 2
                or set(oriented) != set(self.movable_block_ids)
                or not guarded
            ):
                raise ValueError("conditional named cut requires one complete basis guard")
        object.__setattr__(self, "oriented_owner_block_ids", oriented)
        object.__setattr__(self, "guarded_coordinates", guarded)
        if self.repair_kind is PlacementRepairKind.EXACT_PLACEMENT_NOGOOD:
            try:
                world_hash_is_canonical = (
                    isinstance(self.routing_world_hash, str)
                    and len(self.routing_world_hash) == 64
                    and self.routing_world_hash == self.routing_world_hash.lower()
                    and int(self.routing_world_hash, 16) >= 0
                )
            except ValueError:
                world_hash_is_canonical = False
            if (
                not origins
                or tuple(row[0] for row in origins) != self.movable_block_ids
                or self.movable_site_ids
                or self.corridor_relation is not None
                or self.required_clearance_nm != 0
                or not world_hash_is_canonical
                or guard_present
            ):
                raise ValueError("exact placement no-good must sign all exact origins and world")
        elif origins or self.routing_world_hash is not None:
            raise ValueError("origin/world evidence is reserved for exact placement no-good")

    @property
    def scope(self) -> PlacementFeedbackScope:
        return PlacementFeedbackScope(
            candidate_pool_hash=self.candidate_pool_hash,
            placement_hash=self.placement_hash,
        )

    @property
    def repair_hash(self) -> str:
        """Identity of the master constraint, independent of its proof round."""

        repair_class = (
            "capacity_separation"
            if self.repair_kind
            in (
                PlacementRepairKind.RELATIVE_BLOCK_ORDER,
                PlacementRepairKind.CERTIFIED_CHANNEL_SPAN,
            )
            else self.repair_kind.value
        )
        payload = (
            {
                "incumbent_origins": [list(row) for row in self.incumbent_origins],
                "repair_class": self.repair_kind.value,
                "schema": "clay.placement-repair.v1",
            }
            if self.repair_kind is PlacementRepairKind.EXACT_PLACEMENT_NOGOOD
            else
            {
                "guarded_coordinates": [list(row) for row in self.guarded_coordinates],
                "oriented_owner_block_ids": list(self.oriented_owner_block_ids),
                "placement_basis_sha256": self.placement_basis_sha256,
                "corridor_relation": self.corridor_relation,
                "movable_block_ids": list(self.movable_block_ids),
                "movable_site_ids": list(self.movable_site_ids),
                "qualified_demand_ids": [
                    list(key) for key in self.qualified_demand_ids
                ],
                "repair_class": repair_class,
                "required_clearance_nm": self.required_clearance_nm,
                "schema": "clay.placement-repair.v1",
            }
        )
        return _sha256(payload)

    @property
    def cut_hash(self) -> str:
        """Exact replay/proof identity for this conflict and feedback scope.

        ``witness_id`` is audit metadata, not repair semantics.  Excluding it
        makes two witnesses for the same replayed conflict byte-identical;
        :attr:`repair_hash` separately terminates the same master constraint
        when it is reproven in a later scope.
        """

        payload = {
                "candidate_ids": list(self.candidate_ids),
                "candidate_pool_hash": self.candidate_pool_hash,
                "conflict_hash": self.conflict_hash,
                "conflict_kind": self.conflict_kind,
                "corridor_relation": self.corridor_relation,
                "declaration_refs": list(self.declaration_refs),
                "demand_ids": list(self.demand_ids),
                "qualified_demand_ids": [
                    list(key) for key in self.qualified_demand_ids
                ],
                "movable_block_ids": list(self.movable_block_ids),
                "movable_site_ids": list(self.movable_site_ids),
                "nets": list(self.nets),
                "placement_hash": self.placement_hash,
                "repair_kind": self.repair_kind.value,
                "required_clearance_nm": self.required_clearance_nm,
                "resource": _resource_payload(self.resource),
                "guarded_coordinates": [list(row) for row in self.guarded_coordinates],
                "oriented_owner_block_ids": list(self.oriented_owner_block_ids),
                "placement_basis_sha256": self.placement_basis_sha256,
            }
        if self.repair_kind is PlacementRepairKind.EXACT_PLACEMENT_NOGOOD:
            payload["incumbent_origins"] = [list(row) for row in self.incumbent_origins]
            payload["routing_world_hash"] = self.routing_world_hash
        return _sha256(payload)


@dataclass(frozen=True)
class PlacementCutReplayProof:
    """Inputs required to replay exactly how a cut was authorized.

    The fields deliberately retain the typed source objects rather than string
    digests alone.  :meth:`CutLedger.record_cut` reruns the proof-gated lowerer
    and requires byte-for-byte dataclass equality with the proposed cut.
    """

    pool: object
    conflict: object
    witness: object
    final_placement_certificate: object
    conflict_replay_certificate: object = None
    feedback_context: object = None


@dataclass(frozen=True)
class ExactPlacementNoGoodProof:
    """Replay-complete authorization for excluding one exact origin vector."""

    feedback_context: RoutePlanningFeedbackContext

    def __post_init__(self) -> None:
        if not isinstance(self.feedback_context, RoutePlanningFeedbackContext):
            raise ValueError("exact no-good proof requires RoutePlanningFeedbackContext")


@dataclass(frozen=True)
class Unlowerable:
    """Named, non-throwing termination for an unsound or stale request."""

    reason_code: str
    detail: str
    conflict_hash: Optional[str] = None
    candidate_pool_hash: Optional[str] = None
    placement_hash: Optional[str] = None


class RoutePlanningFeedbackContextError(ValueError):
    """Exact route-planning evidence cannot be bound to one placement scope."""


@dataclass(frozen=True)
class RoutePlanningFeedbackContext:
    """Replay-complete INFEASIBLE context; never contains a proposed witness."""

    pool: CandidatePool
    conflict: CapacityConflict | HallDeficit
    routing_world: Any
    final_placement_certificate: FinalPlacementCertificateView
    placement_state_digest: str = field(init=False)
    candidate_pool_hash: str = field(init=False)
    routing_world_hash: str = field(init=False)
    conflict_hash: str = field(init=False)

    def __post_init__(self) -> None:
        from clay.layout.route.resource_world import RoutingWorldProvenance

        if not isinstance(self.pool, CandidatePool):
            raise RoutePlanningFeedbackContextError(
                "feedback context pool must be CandidatePool"
            )
        if not isinstance(self.conflict, (CapacityConflict, HallDeficit)):
            raise RoutePlanningFeedbackContextError(
                "feedback context requires CapacityConflict or HallDeficit"
            )
        if not isinstance(self.routing_world, RoutingWorldProvenance):
            raise RoutePlanningFeedbackContextError(
                "feedback context routing_world must be RoutingWorldProvenance"
            )
        if not isinstance(
            self.final_placement_certificate, FinalPlacementCertificateView
        ):
            raise RoutePlanningFeedbackContextError(
                "feedback context requires a typed final placement certificate"
            )
        try:
            placement_hash = final_placement_state_digest(
                self.final_placement_certificate
            )
            pool_hash = self.pool.sha256()
            world_hash = self.routing_world.sha256()
            actual_conflict_hash = conflict_digest(self.conflict)
            verify_infeasibility_certificate(self.pool, self.conflict)
        except Exception as exc:
            raise RoutePlanningFeedbackContextError(
                "feedback context infeasibility replay failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if self.conflict.candidate_pool_hash != pool_hash:
            raise RoutePlanningFeedbackContextError(
                "feedback conflict does not name the exact candidate pool"
            )
        if self.pool.routing_world_provenance_digest != world_hash:
            raise RoutePlanningFeedbackContextError(
                "feedback candidate pool does not name the exact routing world"
            )
        if self.pool.placement_provenance_digest != placement_hash:
            raise RoutePlanningFeedbackContextError(
                "feedback candidate pool placement provenance is not the exact "
                "stable placement state"
            )
        if self.routing_world.placement_provenance_digest != placement_hash:
            raise RoutePlanningFeedbackContextError(
                "feedback routing-world placement provenance is not the exact "
                "stable placement state"
            )
        for field_name in ("topology_provenance_digest", "pdk_provenance_digest"):
            if getattr(self.pool, field_name) != getattr(self.routing_world, field_name):
                raise RoutePlanningFeedbackContextError(
                    f"feedback pool/world {field_name} mismatch"
                )
        if self.pool.required_nets != self.routing_world.required_nets:
            raise RoutePlanningFeedbackContextError(
                "feedback pool/world required-net mismatch"
            )
        object.__setattr__(self, "placement_state_digest", placement_hash)
        object.__setattr__(self, "candidate_pool_hash", pool_hash)
        object.__setattr__(self, "routing_world_hash", world_hash)
        object.__setattr__(self, "conflict_hash", actual_conflict_hash)


@dataclass(frozen=True)
class PlacementConflictReplayCertificate:
    """Conflict-specific second proof compiled from a static placement basis."""

    conflict_kind: str
    placement_basis_sha256: str
    conflict_hash: str
    candidate_pool_hash: str
    placement_hash: str
    repair_kind: PlacementRepairKind
    resource: Optional[CorridorLayerKey]
    movable_block_ids: tuple[str, ...]
    movable_site_ids: tuple[str, ...]
    corridor_relation: Optional[str]
    observed_clearance_nm: int
    required_clearance_nm: int
    active_use_ids: tuple[tuple[str, str], ...]
    capacity_deficit: int
    track_pitch_nm: int
    candidate_ids: tuple[str, ...]
    qualified_demand_ids: tuple[tuple[str, str], ...]
    declaration_refs: tuple[str, ...]
    placement_basis: PlacementCausalityBasis
    oriented_owner_block_ids: tuple[str, str]
    guarded_coordinates: tuple[tuple[str, str, int], ...]
    schema: str = "clay.placement-conflict-replay.v2"

    def __post_init__(self) -> None:
        if self.schema != "clay.placement-conflict-replay.v2":
            raise ValueError("unknown placement conflict replay schema")
        if self.conflict_kind not in ("capacity_conflict", "hall_deficit"):
            raise ValueError("placement conflict replay has unknown conflict kind")
        for name in (
            "placement_basis_sha256",
            "conflict_hash",
            "candidate_pool_hash",
            "placement_hash",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value != value.lower()
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a canonical SHA-256")
        if not isinstance(self.repair_kind, PlacementRepairKind):
            raise ValueError("placement conflict replay has unknown repair kind")
        object.__setattr__(
            self, "movable_block_ids",
            _sorted_unique_strings(self.movable_block_ids, "movable_block_ids"),
        )
        object.__setattr__(
            self, "movable_site_ids",
            _sorted_unique_strings(self.movable_site_ids, "movable_site_ids"),
        )
        object.__setattr__(
            self, "candidate_ids",
            _sorted_unique_strings(self.candidate_ids, "candidate_ids"),
        )
        object.__setattr__(
            self, "active_use_ids",
            _sorted_unique_pairs(self.active_use_ids, "active_use_ids"),
        )
        object.__setattr__(
            self, "qualified_demand_ids",
            _sorted_unique_pairs(self.qualified_demand_ids, "qualified_demand_ids"),
        )
        object.__setattr__(
            self, "declaration_refs",
            _sorted_unique_strings(self.declaration_refs, "declaration_refs"),
        )
        guard_basis = self.placement_basis
        if (
            not isinstance(guard_basis, PlacementCausalityBasis)
            or not guard_basis.blocks
            or guard_basis.sha256 != self.placement_basis_sha256
        ):
            raise ValueError("placement conflict replay lacks complete basis guard")
        oriented = tuple(self.oriented_owner_block_ids)
        guarded = tuple(sorted(self.guarded_coordinates))
        block_origins = {
            name: (bbox[0], bbox[1]) for name, bbox, _members in guard_basis.blocks
        }
        repair_axis = (
            "y" if self.corridor_relation == "vertical_separation"
            else "x" if self.corridor_relation == "horizontal_separation"
            else None
        )
        expected_guard_keys = (
            {
                (name, axis)
                for name in block_origins
                for axis in ("x", "y")
                if (name, axis) != (oriented[1], repair_axis)
            }
            if len(oriented) == 2 and repair_axis is not None
            else set()
        )
        if (
            len(oriented) != 2
            or len(set(oriented)) != 2
            or set(oriented) != set(self.movable_block_ids)
            or any(
                not isinstance(row, (tuple, list))
                or len(row) != 3
                or row[0] not in guard_basis.landing_owner_block_ids
                or row[1] not in ("x", "y")
                or type(row[2]) is not int
                for row in guarded
            )
            or len({(row[0], row[1]) for row in guarded}) != len(guarded)
            or {(row[0], row[1]) for row in guarded} != expected_guard_keys
            or any(
                value != block_origins[name][0 if axis == "x" else 1]
                for name, axis, value in guarded
            )
        ):
            raise ValueError("placement conflict replay guard coordinates are incomplete")
        object.__setattr__(self, "oriented_owner_block_ids", oriented)
        object.__setattr__(self, "guarded_coordinates", guarded)
        for name in (
            "observed_clearance_nm",
            "required_clearance_nm",
            "capacity_deficit",
            "track_pitch_nm",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative integer nm")

    @property
    def sha256(self) -> str:
        return _sha256({
            "active_use_ids": [list(row) for row in self.active_use_ids],
            "candidate_ids": list(self.candidate_ids),
            "candidate_pool_hash": self.candidate_pool_hash,
            "capacity_deficit": self.capacity_deficit,
            "conflict_hash": self.conflict_hash,
            "conflict_kind": self.conflict_kind,
            "corridor_relation": self.corridor_relation,
            "declaration_refs": list(self.declaration_refs),
            "guarded_coordinates": [list(row) for row in self.guarded_coordinates],
            "movable_block_ids": list(self.movable_block_ids),
            "movable_site_ids": list(self.movable_site_ids),
            "observed_clearance_nm": self.observed_clearance_nm,
            "placement_basis_sha256": self.placement_basis_sha256,
            "placement_hash": self.placement_hash,
            "oriented_owner_block_ids": list(self.oriented_owner_block_ids),
            "qualified_demand_ids": [list(row) for row in self.qualified_demand_ids],
            "repair_kind": self.repair_kind.value,
            "required_clearance_nm": self.required_clearance_nm,
            "resource": _resource_payload(self.resource),
            "schema": self.schema,
            "track_pitch_nm": self.track_pitch_nm,
        })


def _compile_monotone_basis_guard(
    basis: PlacementCausalityBasis,
    boundary: PlacementCorridorBoundary,
) -> tuple[
    PlacementCausalityBasis,
    tuple[str, str],
    tuple[tuple[str, str, int], ...],
] | Unlowerable:
    """Sign one conditional monotone slice, or refuse the global gap claim.

    The first owner is the stationary face and the second owner is the repair
    face.  Every master coordinate except the repair coordinate is guarded.
    Leaving that exact slice therefore satisfies the cut without imposing a
    global pair separation.  A third block projected onto the corridor's run
    interval makes widening non-monotone and forces the exact no-good fallback.
    """

    by_name = {name: bbox for name, bbox, _members in basis.blocks}
    first_name, second_name = boundary.owner_block_ids
    first, second = by_name[first_name], by_name[second_name]
    if boundary.relation == "vertical_separation":
        if first[3] <= second[1]:
            stationary, movable = first_name, second_name
        elif second[3] <= first[1]:
            stationary, movable = second_name, first_name
        else:
            return _unlowerable(
                "monotone_slice_unavailable",
                "signed vertical boundary owners no longer have ordered faces",
            )
        run_axis = "x"
        repair_axis = "y"
        projected = lambda bbox: (bbox[0], bbox[2])
    elif boundary.relation == "horizontal_separation":
        if first[2] <= second[0]:
            stationary, movable = first_name, second_name
        elif second[2] <= first[0]:
            stationary, movable = second_name, first_name
        else:
            return _unlowerable(
                "monotone_slice_unavailable",
                "signed horizontal boundary owners no longer have ordered faces",
            )
        run_axis = "y"
        repair_axis = "x"
        projected = lambda bbox: (bbox[1], bbox[3])
    else:
        return _unlowerable(
            "monotone_slice_unavailable",
            "signed boundary has no supported separation relation",
        )

    run_lo, run_hi = boundary.run_interval_nm
    blockers = tuple(
        name
        for name, bbox, _members in basis.blocks
        if name not in boundary.owner_block_ids
        and max(run_lo, projected(bbox)[0]) < min(run_hi, projected(bbox)[1])
    )
    if blockers:
        return _unlowerable(
            "monotone_slice_unavailable",
            "corridor widening can encounter projected master blocks: "
            + ",".join(blockers),
        )

    oriented = (stationary, movable)
    guarded = tuple(sorted(
        (name, axis, bbox[0 if axis == "x" else 1])
        for name, bbox, _members in basis.blocks
        for axis in ("x", "y")
        if (name, axis) != (movable, repair_axis)
    ))
    assert (movable, run_axis) in {(name, axis) for name, axis, _value in guarded}
    return basis, oriented, guarded


def _compile_capacity_conflict_replay(
    context: RoutePlanningFeedbackContext,
) -> PlacementConflictReplayCertificate | Unlowerable:
    """Prove a monotone corridor-width inequality or refuse it by name."""

    if not isinstance(context.conflict, CapacityConflict):
        return _unlowerable(
            "capacity_conflict_required",
            "capacity replay requires CapacityConflict",
        )
    certificate = context.final_placement_certificate
    try:
        verify_placement_causality_basis(certificate)
        basis = certificate.placement_causality_basis
    except (AttributeError, TypeError, ValueError) as exc:
        return _unlowerable(
            "placement_causality_basis_unavailable",
            f"static final-master basis did not replay: {exc}",
            conflict_hash=context.conflict_hash,
            candidate_pool_hash=context.candidate_pool_hash,
            placement_hash=context.placement_state_digest,
        )
    resource = next(
        (item for item in context.pool.resources if item.key == context.conflict.resource),
        None,
    )
    if (
        resource is None
        or resource.corridor_bbox_nm is None
        or resource.through_interval_nm is None
    ):
        return _unlowerable(
            "corridor_boundary_proof_unavailable",
            "capacity resource lacks geometry-complete corridor evidence",
            conflict_hash=context.conflict_hash,
            candidate_pool_hash=context.candidate_pool_hash,
            placement_hash=context.placement_state_digest,
        )
    relation = (
        "vertical_separation"
        if resource.preferred_axis == "H"
        else "horizontal_separation"
        if resource.preferred_axis == "V"
        else None
    )
    if relation is None:
        return _unlowerable(
            "corridor_axis_unavailable",
            "capacity resource has no closed preferred axis",
        )
    conflict_interval = tuple(context.conflict.overlap_interval_nm)
    corridor_bbox = tuple(resource.corridor_bbox_nm)
    block_boxes = {name: bbox for name, bbox, _members in basis.blocks}

    def bounds_exact_corridor(boundary: PlacementCorridorBoundary) -> bool:
        if boundary.relation != relation:
            return False
        if not (
            boundary.run_interval_nm[0] <= conflict_interval[0]
            and conflict_interval[1] <= boundary.run_interval_nm[1]
            and resource.through_interval_nm[0] <= conflict_interval[0]
            and conflict_interval[1] <= resource.through_interval_nm[1]
        ):
            return False
        left, right = (block_boxes[name] for name in boundary.owner_block_ids)
        if relation == "vertical_separation":
            gap = (
                (left[3], right[1]) if left[3] <= right[1]
                else (right[3], left[1]) if right[3] <= left[1]
                else None
            )
            cross = (corridor_bbox[1], corridor_bbox[3])
        else:
            gap = (
                (left[2], right[0]) if left[2] <= right[0]
                else (right[2], left[0]) if right[2] <= left[0]
                else None
            )
            cross = (corridor_bbox[0], corridor_bbox[2])
        # Exact equality is intentionally conservative: if the enumerated
        # resource occupies only part of a gap, moving these faces is not yet a
        # proof that its capacity grows monotonically.
        return gap is not None and cross == gap

    boundaries = tuple(
        boundary
        for boundary in basis.corridor_boundaries
        if bounds_exact_corridor(boundary)
    )
    if len(boundaries) != 1:
        return _unlowerable(
            "corridor_boundary_proof_unavailable",
            "overloaded resource is not uniquely bounded by signed master faces",
            conflict_hash=context.conflict_hash,
            candidate_pool_hash=context.candidate_pool_hash,
            placement_hash=context.placement_state_digest,
        )
    active = _replayed_active_capacity_uses(
        context.pool,
        context.conflict,
        context={
            "conflict_hash": context.conflict_hash,
            "candidate_pool_hash": context.candidate_pool_hash,
            "placement_hash": context.placement_state_digest,
        },
    )
    if isinstance(active, Unlowerable):
        return active
    pitch = _explicit_positive_track_pitch_nm(resource.track_centers_nm)
    if pitch is None:
        return _unlowerable(
            "causality_pitch_unavailable",
            "explicit resource tracks do not define one positive pitch",
        )
    deficit = sum(use.track_span for _candidate, use in active) - context.conflict.track_capacity
    if deficit <= 0:
        return _unlowerable("capacity_deficit_missing", "replayed overload is vacuous")
    boundary = boundaries[0]
    guard = _compile_monotone_basis_guard(basis, boundary)
    if isinstance(guard, Unlowerable):
        return replace(
            guard,
            conflict_hash=context.conflict_hash,
            candidate_pool_hash=context.candidate_pool_hash,
            placement_hash=context.placement_state_digest,
        )
    placement_basis, oriented_owner_block_ids, guarded_coordinates = guard
    return PlacementConflictReplayCertificate(
        conflict_kind="capacity_conflict",
        placement_basis_sha256=basis.sha256,
        conflict_hash=context.conflict_hash,
        candidate_pool_hash=context.candidate_pool_hash,
        placement_hash=context.placement_state_digest,
        repair_kind=PlacementRepairKind.CERTIFIED_CHANNEL_SPAN,
        resource=context.conflict.resource,
        movable_block_ids=boundary.owner_block_ids,
        movable_site_ids=(),
        corridor_relation=relation,
        observed_clearance_nm=boundary.observed_clearance_nm,
        required_clearance_nm=boundary.observed_clearance_nm + deficit * pitch,
        active_use_ids=tuple(
            (candidate.candidate_id, use.use_id) for candidate, use in active
        ),
        capacity_deficit=deficit,
        track_pitch_nm=pitch,
        candidate_ids=context.conflict.candidate_ids,
        qualified_demand_ids=(),
        declaration_refs=context.conflict.declaration_refs,
        placement_basis=placement_basis,
        oriented_owner_block_ids=oriented_owner_block_ids,
        guarded_coordinates=guarded_coordinates,
    )


def _compile_hall_conflict_replay(
    context: RoutePlanningFeedbackContext,
) -> PlacementConflictReplayCertificate | Unlowerable:
    """Prove a necessary landing-lattice expansion inequality or refuse it.

    A Hall deficit by itself is only a pool-scoped infeasibility proof.  This
    lowering additionally requires one pool-hash-bound expansion resource and
    replays that resource against an exact final-master gap.  The strict
    half-pitch edge test makes the current site list a complete one-dimensional
    lattice: widening the gap by less than ``deficit * pitch`` cannot add the
    missing number of unit-capacity sites.
    """

    if not isinstance(context.conflict, HallDeficit):
        return _unlowerable(
            "hall_deficit_required",
            "Hall replay requires HallDeficit",
        )
    certificate = context.final_placement_certificate
    try:
        verify_placement_causality_basis(certificate)
        basis = certificate.placement_causality_basis
    except (AttributeError, TypeError, ValueError) as exc:
        return _unlowerable(
            "placement_causality_basis_unavailable",
            f"static final-master basis did not replay: {exc}",
            conflict_hash=context.conflict_hash,
            candidate_pool_hash=context.candidate_pool_hash,
            placement_hash=context.placement_state_digest,
        )
    expansion_ids = tuple(context.conflict.expansion_resource_ids)
    if len(expansion_ids) != 1:
        return _unlowerable(
            "hall_expansion_proof_unavailable",
            "Hall deficit does not name one unique landing expansion resource",
        )
    expansion = next(
        (
            item
            for item in context.pool.landing_expansion_resources
            if item.expansion_id == expansion_ids[0]
        ),
        None,
    )
    if expansion is None or expansion.placement_basis_sha256 != basis.sha256:
        return _unlowerable(
            "hall_expansion_proof_stale",
            "landing expansion resource does not bind the current placement basis",
        )
    boundaries = tuple(
        boundary
        for boundary in basis.corridor_boundaries
        if boundary.owner_block_ids == expansion.owner_block_ids
        and boundary.relation == expansion.corridor_relation
        and boundary.run_interval_nm == expansion.run_interval_nm
        and boundary.observed_clearance_nm == expansion.observed_clearance_nm
    )
    if len(boundaries) != 1:
        return _unlowerable(
            "hall_boundary_replay_failed",
            "landing expansion does not replay one exact signed master boundary",
        )
    if set(expansion.declaration_refs) - set(basis.declaration_refs):
        return _unlowerable(
            "hall_declaration_replay_failed",
            "landing expansion names declarations absent from the placement basis",
        )
    if tuple(expansion.legal_site_ids) != tuple(context.conflict.reachable_site_ids):
        return _unlowerable(
            "hall_site_domain_replay_failed",
            "Hall reachable sites are not the expansion resource's complete site domain",
        )
    sites_by_id = context.pool.landing_sites_by_id()
    selected_sites = tuple(sites_by_id[site_id] for site_id in expansion.legal_site_ids)
    block_boxes = {name: bbox for name, bbox, _members in basis.blocks}
    first, second = (block_boxes[name] for name in expansion.owner_block_ids)
    if expansion.corridor_relation == "vertical_separation":
        gap = (
            (first[3], second[1]) if first[3] <= second[1]
            else (second[3], first[1]) if second[3] <= first[1]
            else None
        )
        cross_coordinates = tuple(sorted(site.xy_nm[1] for site in selected_sites))
    else:
        gap = (
            (first[2], second[0]) if first[2] <= second[0]
            else (second[2], first[0]) if second[2] <= first[0]
            else None
        )
        cross_coordinates = tuple(sorted(site.xy_nm[0] for site in selected_sites))
    pitch = expansion.site_pitch_nm
    if (
        gap is None
        or gap[1] - gap[0] != expansion.observed_clearance_nm
        or len(cross_coordinates) != len(set(cross_coordinates))
        or expansion.observed_clearance_nm != len(cross_coordinates) * pitch
        or 2 * (cross_coordinates[0] - gap[0]) != pitch
        or 2 * (gap[1] - cross_coordinates[-1]) != pitch
        or any(
            right - left != pitch
            for left, right in zip(cross_coordinates, cross_coordinates[1:])
        )
    ):
        return _unlowerable(
            "hall_lattice_replay_failed",
            "landing sites do not form the complete signed half-pitch gap lattice",
        )
    replay_context = {
        "conflict_hash": context.conflict_hash,
        "candidate_pool_hash": context.candidate_pool_hash,
        "placement_hash": context.placement_state_digest,
    }
    entries = _replayed_qualified_hall_entries(
        context.pool, context.conflict, context=replay_context
    )
    if isinstance(entries, Unlowerable):
        return entries
    capacities = {
        site.site_id: site.capacity
        for site in context.pool.iter_landing_sites()
    }
    deficit = len(entries) - sum(
        capacities[site_id] for site_id in context.conflict.reachable_site_ids
    )
    if deficit <= 0 or deficit != context.conflict.deficit:
        return _unlowerable(
            "hall_deficit_replay_failed",
            "landing expansion replay did not reproduce the certified Hall deficit",
            **replay_context,
        )
    boundary = boundaries[0]
    guard = _compile_monotone_basis_guard(basis, boundary)
    if isinstance(guard, Unlowerable):
        return replace(
            guard,
            conflict_hash=context.conflict_hash,
            candidate_pool_hash=context.candidate_pool_hash,
            placement_hash=context.placement_state_digest,
        )
    placement_basis, oriented_owner_block_ids, guarded_coordinates = guard
    return PlacementConflictReplayCertificate(
        conflict_kind="hall_deficit",
        placement_basis_sha256=basis.sha256,
        conflict_hash=context.conflict_hash,
        candidate_pool_hash=context.candidate_pool_hash,
        placement_hash=context.placement_state_digest,
        repair_kind=PlacementRepairKind.CANDIDATE_LANDING_ADJACENCY,
        resource=None,
        movable_block_ids=expansion.owner_block_ids,
        movable_site_ids=context.conflict.reachable_site_ids,
        corridor_relation=expansion.corridor_relation,
        observed_clearance_nm=expansion.observed_clearance_nm,
        required_clearance_nm=expansion.observed_clearance_nm + deficit * pitch,
        active_use_ids=(),
        capacity_deficit=deficit,
        track_pitch_nm=pitch,
        candidate_ids=context.conflict.candidate_ids,
        qualified_demand_ids=tuple(
            (candidate.candidate_id, demand.demand_id)
            for candidate, demand in entries
        ),
        declaration_refs=context.conflict.declaration_refs,
        placement_basis=placement_basis,
        oriented_owner_block_ids=oriented_owner_block_ids,
        guarded_coordinates=guarded_coordinates,
    )


def _cut_from_conflict_replay(
    context: RoutePlanningFeedbackContext,
    replay: object,
) -> PlacementSolveCut | Unlowerable:
    if not isinstance(replay, PlacementConflictReplayCertificate):
        return _unlowerable("conflict_replay_missing", "typed conflict replay required")
    expected = (
        _compile_capacity_conflict_replay(context)
        if isinstance(context.conflict, CapacityConflict)
        else _compile_hall_conflict_replay(context)
    )
    if isinstance(expected, Unlowerable):
        return expected
    if expected != replay or expected.sha256 != replay.sha256:
        return _unlowerable(
            "conflict_replay_mismatch",
            "placement conflict replay certificate does not reproduce",
            conflict_hash=context.conflict_hash,
            candidate_pool_hash=context.candidate_pool_hash,
            placement_hash=context.placement_state_digest,
        )
    conflict = context.conflict
    return PlacementSolveCut(
        repair_kind=replay.repair_kind,
        conflict_kind=replay.conflict_kind,
        conflict_hash=replay.conflict_hash,
        candidate_pool_hash=replay.candidate_pool_hash,
        placement_hash=replay.placement_hash,
        witness_id=f"placement-conflict-replay:{replay.sha256}",
        nets=conflict.nets if isinstance(conflict, CapacityConflict) else (),
        candidate_ids=replay.candidate_ids,
        demand_ids=conflict.demand_ids if isinstance(conflict, HallDeficit) else (),
        qualified_demand_ids=replay.qualified_demand_ids,
        movable_block_ids=replay.movable_block_ids,
        movable_site_ids=replay.movable_site_ids,
        resource=replay.resource,
        corridor_relation=replay.corridor_relation,
        required_clearance_nm=replay.required_clearance_nm,
        declaration_refs=replay.declaration_refs,
        placement_basis_sha256=replay.placement_basis_sha256,
        oriented_owner_block_ids=replay.oriented_owner_block_ids,
        guarded_coordinates=replay.guarded_coordinates,
    )


def lower_route_planning_feedback(
    context: object,
) -> tuple[PlacementSolveCut, object] | Unlowerable:
    """Prefer a two-stage causal cut; honestly fall back to exact no-good."""

    if not isinstance(context, RoutePlanningFeedbackContext):
        return _unlowerable(
            "feedback_context_missing",
            "route-planning feedback lowering requires an exact context",
        )
    replay = (
        _compile_capacity_conflict_replay(context)
        if isinstance(context.conflict, CapacityConflict)
        else _compile_hall_conflict_replay(context)
    )
    if isinstance(replay, PlacementConflictReplayCertificate):
        cut = _cut_from_conflict_replay(context, replay)
        if isinstance(cut, PlacementSolveCut):
            return cut, PlacementCutReplayProof(
                pool=context.pool,
                conflict=context.conflict,
                witness=None,
                final_placement_certificate=context.final_placement_certificate,
                conflict_replay_certificate=replay,
                feedback_context=context,
            )
    fallback = lower_exact_placement_nogood(context)
    if isinstance(fallback, PlacementSolveCut):
        return fallback, ExactPlacementNoGoodProof(context)
    return fallback


def lower_exact_placement_nogood(
    feedback_context: object,
) -> PlacementSolveCut | Unlowerable:
    """Exclude only one replay-bound final-master origin vector.

    This deliberately makes no corridor or landing-causality claim.  V1 signs
    every final master because the resource IR has no exact resource-to-master
    ownership map; selecting a smaller guessed subset would be unsound.
    """

    if not isinstance(feedback_context, RoutePlanningFeedbackContext):
        return _unlowerable(
            "feedback_context_missing",
            "exact placement no-good requires typed route-planning feedback",
        )
    try:
        replayed = RoutePlanningFeedbackContext(
            pool=feedback_context.pool,
            conflict=feedback_context.conflict,
            routing_world=feedback_context.routing_world,
            final_placement_certificate=feedback_context.final_placement_certificate,
        )
        blocks = _canonical_block_bboxes(
            replayed.final_placement_certificate.blocks
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return _unlowerable(
            "feedback_context_replay_failed",
            f"exact no-good context did not replay: {type(exc).__name__}: {exc}",
        )
    if (
        replayed.placement_state_digest != feedback_context.placement_state_digest
        or replayed.candidate_pool_hash != feedback_context.candidate_pool_hash
        or replayed.routing_world_hash != feedback_context.routing_world_hash
        or replayed.conflict_hash != feedback_context.conflict_hash
    ):
        return _unlowerable(
            "feedback_context_digest_mismatch",
            "exact no-good context digest fields differ from independent replay",
        )
    conflict = replayed.conflict
    is_capacity = isinstance(conflict, CapacityConflict)
    return PlacementSolveCut(
        repair_kind=PlacementRepairKind.EXACT_PLACEMENT_NOGOOD,
        conflict_kind="capacity_conflict" if is_capacity else "hall_deficit",
        conflict_hash=replayed.conflict_hash,
        candidate_pool_hash=replayed.candidate_pool_hash,
        placement_hash=replayed.placement_state_digest,
        witness_id=f"exact-origin-vector:{replayed.conflict_hash}",
        nets=conflict.nets if is_capacity else (),
        candidate_ids=conflict.candidate_ids,
        demand_ids=() if is_capacity else conflict.demand_ids,
        qualified_demand_ids=(),
        movable_block_ids=tuple(name for name, _bbox in blocks),
        movable_site_ids=(),
        resource=conflict.resource if is_capacity else None,
        corridor_relation=None,
        required_clearance_nm=0,
        declaration_refs=conflict.declaration_refs,
        routing_world_hash=replayed.routing_world_hash,
        incumbent_origins=tuple(
            (name, bbox[0], bbox[1]) for name, bbox in blocks
        ),
    )


@dataclass(frozen=True)
class CutLedger:
    """Finite immutable record spanning consecutive exact feedback scopes.

    ``candidate_pool_hash`` and ``placement_hash`` name the current scope for
    backward compatibility.  Prior scopes and their proven cuts are retained;
    only a cut replayed against the current scope may enter the ledger.
    """

    candidate_pool_hash: str
    placement_hash: str
    scopes: tuple[PlacementFeedbackScope, ...] = field(default=(), init=False)
    geometry_progress_hashes: tuple[str, ...] = field(default=(), init=False)
    # Cuts may enter only through ``record_cut``, which replays their proof.
    # ``init=False`` closes the public dataclass-constructor bypass.
    cuts: tuple[PlacementSolveCut, ...] = field(default=(), init=False)
    progress_hashes: tuple[str, ...] = ()
    max_cuts: int = 16
    max_progress_states: int = 32

    def __post_init__(self) -> None:
        for field_name in ("candidate_pool_hash", "placement_hash"):
            if not isinstance(getattr(self, field_name), str) or not getattr(
                self, field_name
            ):
                raise ValueError(f"{field_name} must be non-empty")
        for field_name, hard_limit in (
            ("max_cuts", _MAX_LEDGER_CUTS),
            ("max_progress_states", _MAX_LEDGER_PROGRESS_STATES),
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value > hard_limit
            ):
                raise ValueError(
                    f"{field_name} must be in [1, {hard_limit}]"
                )
        cuts = tuple(sorted(self.cuts, key=lambda cut: cut.cut_hash))
        if not all(isinstance(cut, PlacementSolveCut) for cut in cuts):
            raise ValueError("cuts must contain PlacementSolveCut values")
        cut_hashes = [cut.cut_hash for cut in cuts]
        if len(cut_hashes) != len(set(cut_hashes)):
            raise ValueError("CutLedger cannot contain duplicate cuts")
        if len(cuts) > self.max_cuts:
            raise ValueError("CutLedger cuts exceed max_cuts")
        current_scope = PlacementFeedbackScope(
            candidate_pool_hash=self.candidate_pool_hash,
            placement_hash=self.placement_hash,
        )
        scopes = tuple(self.scopes) or (current_scope,)
        if not all(isinstance(scope, PlacementFeedbackScope) for scope in scopes):
            raise ValueError("scopes must contain PlacementFeedbackScope values")
        if scopes[-1] != current_scope:
            raise ValueError("last feedback scope must be the current scope")
        if len(scopes) != len(set(scopes)):
            raise ValueError("CutLedger cannot contain repeated feedback scopes")
        if len(scopes) > self.max_progress_states:
            raise ValueError("CutLedger scopes exceed max_progress_states")
        geometry_progress = tuple(self.geometry_progress_hashes)
        if any(
            not isinstance(value, str) or not value
            for value in geometry_progress
        ):
            raise ValueError(
                "geometry_progress_hashes must be non-empty strings"
            )
        if len(geometry_progress) != len(set(geometry_progress)):
            raise ValueError(
                "CutLedger cannot contain repeated geometry progress states"
            )
        if len(geometry_progress) > self.max_progress_states:
            raise ValueError(
                "CutLedger geometry progress exceeds max_progress_states"
            )
        progress = tuple(self.progress_hashes)
        if any(not isinstance(value, str) or not value for value in progress):
            raise ValueError("progress_hashes must be non-empty strings")
        if len(progress) != len(set(progress)):
            raise ValueError("CutLedger cannot contain repeated progress states")
        if len(progress) > self.max_progress_states:
            raise ValueError("CutLedger progress exceeds max_progress_states")
        object.__setattr__(self, "cuts", cuts)
        object.__setattr__(self, "scopes", scopes)
        object.__setattr__(self, "geometry_progress_hashes", geometry_progress)
        object.__setattr__(self, "progress_hashes", progress)

    @property
    def current_scope(self) -> PlacementFeedbackScope:
        return PlacementFeedbackScope(
            candidate_pool_hash=self.candidate_pool_hash,
            placement_hash=self.placement_hash,
        )

    @classmethod
    def start(
        cls,
        *,
        pool: object,
        final_placement_certificate: object,
        max_cuts: int = 16,
        max_progress_states: int = 32,
    ) -> "CutLedger | Unlowerable":
        scope = _typed_feedback_scope(
            pool=pool,
            final_placement_certificate=final_placement_certificate,
        )
        if isinstance(scope, Unlowerable):
            return scope
        progress_hash = placement_geometry_digest(final_placement_certificate)
        ledger = cls.empty(
            candidate_pool_hash=scope.candidate_pool_hash,
            placement_hash=scope.placement_hash,
            max_cuts=max_cuts,
            max_progress_states=max_progress_states,
        )
        object.__setattr__(ledger, "geometry_progress_hashes", (progress_hash,))
        return ledger

    @classmethod
    def empty(
        cls,
        *,
        candidate_pool_hash: str,
        placement_hash: str,
        max_cuts: int = 16,
        max_progress_states: int = 32,
    ) -> "CutLedger":
        return cls(
            candidate_pool_hash=candidate_pool_hash,
            placement_hash=placement_hash,
            progress_hashes=(placement_hash,),
            max_cuts=max_cuts,
            max_progress_states=max_progress_states,
        )

    def record_cut(
        self, cut: object, *, proof: object = None
    ) -> "CutLedger | Unlowerable":
        if not isinstance(cut, PlacementSolveCut):
            return _unlowerable("foreign_cut", "ledger accepts PlacementSolveCut only")
        if not isinstance(proof, (PlacementCutReplayProof, ExactPlacementNoGoodProof)):
            return _unlowerable(
                "cut_proof_missing",
                "ledger requires typed replay inputs for every proposed cut",
                conflict_hash=cut.conflict_hash,
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        replayed = (
            lower_exact_placement_nogood(proof.feedback_context)
            if isinstance(proof, ExactPlacementNoGoodProof)
            else _cut_from_conflict_replay(
                proof.feedback_context,
                proof.conflict_replay_certificate,
            )
            if (
                proof.conflict_replay_certificate is not None
                and isinstance(proof.feedback_context, RoutePlanningFeedbackContext)
            )
            else lower_placement_conflict(
                pool=proof.pool,
                conflict=proof.conflict,
                witness=proof.witness,
                final_placement_certificate=proof.final_placement_certificate,
            )
        )
        if isinstance(replayed, Unlowerable):
            return _unlowerable(
                "cut_proof_replay_failed",
                f"cut lowering proof did not replay: {replayed.reason_code}",
                conflict_hash=cut.conflict_hash,
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        if replayed != cut:
            return _unlowerable(
                "cut_proof_mismatch",
                "proposed cut differs from the replayed lowering result",
                conflict_hash=cut.conflict_hash,
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        if cut.scope != self.current_scope:
            return _unlowerable(
                "stale_cut",
                "cut hashes do not match the ledger scope",
                conflict_hash=cut.conflict_hash,
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        if cut.repair_hash in {existing.repair_hash for existing in self.cuts}:
            return _unlowerable(
                "duplicate_cut",
                f"repair {cut.repair_hash} is already recorded",
                conflict_hash=cut.conflict_hash,
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        if len(self.cuts) >= self.max_cuts:
            return _unlowerable(
                "cut_limit_reached",
                f"cut ledger reached its hard bound of {self.max_cuts}",
                conflict_hash=cut.conflict_hash,
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        updated = replace(self)
        object.__setattr__(
            updated,
            "cuts",
            tuple(sorted(self.cuts + (cut,), key=lambda item: item.cut_hash)),
        )
        object.__setattr__(updated, "scopes", self.scopes)
        object.__setattr__(
            updated, "geometry_progress_hashes", self.geometry_progress_hashes
        )
        return updated

    def advance_scope(
        self,
        *,
        pool: object,
        final_placement_certificate: object,
    ) -> "CutLedger | Unlowerable":
        """Advance after a current-scope cut to a newly certified placement.

        This is the finite lower-level state transition needed by a live outer
        loop.  It does not run the placement master or planner; callers must
        provide the next exact certificate and its pool.
        """

        next_scope = _typed_feedback_scope(
            pool=pool,
            final_placement_certificate=final_placement_certificate,
        )
        if isinstance(next_scope, Unlowerable):
            return next_scope
        if not self.geometry_progress_hashes:
            return _unlowerable(
                "typed_scope_required",
                "legacy CutLedger.empty scopes cannot enter the typed outer loop",
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        next_progress_hash = placement_geometry_digest(
            final_placement_certificate
        )
        if next_progress_hash in set(self.geometry_progress_hashes):
            return _unlowerable(
                "no_progress",
                f"placement geometry {next_progress_hash} was already observed",
                candidate_pool_hash=next_scope.candidate_pool_hash,
                placement_hash=next_scope.placement_hash,
            )
        if next_scope in set(self.scopes):
            return _unlowerable(
                "duplicate_scope",
                "feedback scope was already observed",
                candidate_pool_hash=next_scope.candidate_pool_hash,
                placement_hash=next_scope.placement_hash,
            )
        current_has_cut = any(cut.scope == self.current_scope for cut in self.cuts)
        if not current_has_cut:
            return _unlowerable(
                "scope_advance_without_cut",
                "feedback scope may advance only after recording a current-scope cut",
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        master_replay_failure = _verify_master_cut_replay(
            final_placement_certificate,
            self.cuts,
            candidate_pool_hash=next_scope.candidate_pool_hash,
            placement_hash=next_scope.placement_hash,
        )
        if master_replay_failure is not None:
            return master_replay_failure
        if len(self.geometry_progress_hashes) >= self.max_progress_states:
            return _unlowerable(
                "progress_limit_reached",
                "placement progress ledger reached its hard bound of "
                f"{self.max_progress_states}",
                candidate_pool_hash=next_scope.candidate_pool_hash,
                placement_hash=next_scope.placement_hash,
            )
        updated = replace(
            self,
            candidate_pool_hash=next_scope.candidate_pool_hash,
            placement_hash=next_scope.placement_hash,
        )
        object.__setattr__(updated, "cuts", self.cuts)
        object.__setattr__(updated, "scopes", self.scopes + (next_scope,))
        object.__setattr__(
            updated,
            "geometry_progress_hashes",
            self.geometry_progress_hashes + (next_progress_hash,),
        )
        return updated

    def record_progress(self, placement_progress_hash: str) -> "CutLedger | Unlowerable":
        if not isinstance(placement_progress_hash, str) or not placement_progress_hash:
            return _unlowerable(
                "invalid_progress",
                "placement progress hash must be a non-empty string",
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        if placement_progress_hash in set(self.progress_hashes):
            return _unlowerable(
                "no_progress",
                f"placement state {placement_progress_hash} was already observed",
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        if len(self.progress_hashes) >= self.max_progress_states:
            return _unlowerable(
                "progress_limit_reached",
                "placement progress ledger reached its hard bound of "
                f"{self.max_progress_states}",
                candidate_pool_hash=self.candidate_pool_hash,
                placement_hash=self.placement_hash,
            )
        updated = replace(
            self, progress_hashes=self.progress_hashes + (placement_progress_hash,)
        )
        object.__setattr__(updated, "cuts", self.cuts)
        object.__setattr__(updated, "scopes", self.scopes)
        object.__setattr__(
            updated, "geometry_progress_hashes", self.geometry_progress_hashes
        )
        return updated

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "candidate_pool_hash": self.candidate_pool_hash,
                "cut_hashes": [cut.cut_hash for cut in self.cuts],
                "geometry_progress_hashes": list(self.geometry_progress_hashes),
                "max_cuts": self.max_cuts,
                "max_progress_states": self.max_progress_states,
                "placement_hash": self.placement_hash,
                "progress_hashes": list(self.progress_hashes),
                "repair_hashes": [cut.repair_hash for cut in self.cuts],
                "schema": "clay.placement-cut-ledger.v2",
                "scopes": [
                    {
                        "candidate_pool_hash": scope.candidate_pool_hash,
                        "placement_hash": scope.placement_hash,
                    }
                    for scope in self.scopes
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, order=True)
class PlacementRepairRoundCertificate:
    """One proof-bound cut application in the finite outer loop."""

    round_index: int
    candidate_pool_hash: str
    placement_hash: str
    cut_hash: str
    repair_hash: str

    def __post_init__(self) -> None:
        if type(self.round_index) is not int or self.round_index < 0:
            raise ValueError("repair round index must be non-negative integer")
        for name in (
            "candidate_pool_hash", "placement_hash", "cut_hash", "repair_hash"
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value != value.lower()
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"repair round {name} must be a canonical SHA-256")


@dataclass(frozen=True)
class PlacementResourceClosureCertificate:
    """Immutable final capacity/Hall selection and repair-ledger closure."""

    resource_status: str
    final_candidate_pool_hash: str
    final_resource_plan_hash: str
    final_placement_hash: str
    cut_ledger: CutLedger
    cut_ledger_sha256: str
    repair_rounds: tuple[PlacementRepairRoundCertificate, ...]
    schema: str = "clay.placement-resource-closure.v1"

    def __post_init__(self) -> None:
        if self.resource_status != "PROVEN_FEASIBLE":
            raise ValueError("resource closure status must be PROVEN_FEASIBLE")
        if self.schema != "clay.placement-resource-closure.v1":
            raise ValueError("unknown placement resource closure schema")
        if not isinstance(self.cut_ledger, CutLedger):
            raise ValueError("resource closure requires a typed CutLedger")
        rounds = tuple(self.repair_rounds)
        if any(not isinstance(row, PlacementRepairRoundCertificate) for row in rounds):
            raise ValueError("resource closure has malformed repair rounds")
        if rounds != tuple(sorted(rounds)):
            raise ValueError("resource closure repair rounds are not canonical")
        for name in (
            "final_candidate_pool_hash",
            "final_resource_plan_hash",
            "final_placement_hash",
            "cut_ledger_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value != value.lower()
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"resource closure {name} must be a canonical SHA-256")
        actual_ledger_hash = hashlib.sha256(
            self.cut_ledger.canonical_json().encode("utf-8")
        ).hexdigest()
        if actual_ledger_hash != self.cut_ledger_sha256:
            raise ValueError("resource closure CutLedger digest does not replay")

    @property
    def sha256(self) -> str:
        return _sha256({
            "cut_ledger_sha256": self.cut_ledger_sha256,
            "final_candidate_pool_hash": self.final_candidate_pool_hash,
            "final_placement_hash": self.final_placement_hash,
            "final_resource_plan_hash": self.final_resource_plan_hash,
            "repair_rounds": [
                {
                    "candidate_pool_hash": row.candidate_pool_hash,
                    "cut_hash": row.cut_hash,
                    "placement_hash": row.placement_hash,
                    "repair_hash": row.repair_hash,
                    "round_index": row.round_index,
                }
                for row in self.repair_rounds
            ],
            "resource_status": self.resource_status,
            "schema": self.schema,
        })

    def replay(
        self,
        *,
        pool: object,
        plan: object,
        final_placement_certificate: object,
    ) -> bool:
        replayed = certify_proven_feasible_resources(
            pool=pool,
            plan=plan,
            final_placement_certificate=final_placement_certificate,
            cut_ledger=self.cut_ledger,
        )
        if replayed != self or replayed.sha256 != self.sha256:
            raise ValueError("resource closure certificate does not replay")
        return True


def certify_proven_feasible_resources(
    *,
    pool: object,
    plan: object,
    final_placement_certificate: object,
    cut_ledger: object,
) -> PlacementResourceClosureCertificate:
    """Bind one verified simultaneous plan and finite repair ledger to placement."""

    from clay.layout.route.resource_plan import (
        CandidatePool,
        RouteResourcePlan,
        verify_route_resource_plan,
    )

    if not isinstance(pool, CandidatePool) or not isinstance(plan, RouteResourcePlan):
        raise ValueError("resource closure requires typed pool and resource plan")
    if not isinstance(cut_ledger, CutLedger):
        raise ValueError("resource closure requires typed CutLedger")
    pool_hash = pool.sha256()
    placement_hash = final_placement_state_digest(final_placement_certificate)
    if plan.candidate_pool_hash != pool_hash:
        raise ValueError("resource plan does not bind the final candidate pool")
    if pool.placement_provenance_digest != placement_hash:
        raise ValueError("resource pool does not bind the final placement")
    if cut_ledger.current_scope != PlacementFeedbackScope(pool_hash, placement_hash):
        raise ValueError("resource closure CutLedger is stale")
    if not verify_route_resource_plan(pool, plan):
        raise ValueError("resource plan did not replay")
    scope_index = {scope: index for index, scope in enumerate(cut_ledger.scopes)}
    rounds = tuple(sorted(
        PlacementRepairRoundCertificate(
            round_index=scope_index[cut.scope],
            candidate_pool_hash=cut.candidate_pool_hash,
            placement_hash=cut.placement_hash,
            cut_hash=cut.cut_hash,
            repair_hash=cut.repair_hash,
        )
        for cut in cut_ledger.cuts
    ))
    ledger_hash = hashlib.sha256(
        cut_ledger.canonical_json().encode("utf-8")
    ).hexdigest()
    return PlacementResourceClosureCertificate(
        resource_status="PROVEN_FEASIBLE",
        final_candidate_pool_hash=pool_hash,
        final_resource_plan_hash=plan.sha256(),
        final_placement_hash=placement_hash,
        cut_ledger=cut_ledger,
        cut_ledger_sha256=ledger_hash,
        repair_rounds=rounds,
    )


def final_placement_digest(certificate: FinalPlacementCertificateView) -> str:
    """Compatibility alias for the sole production placement identity domain."""

    return final_placement_state_digest(certificate)


def final_placement_state_digest(
    certificate: FinalPlacementCertificateView,
) -> str:
    """Delegate to the cycle-free canonical final-placement identity."""

    if not isinstance(certificate, FinalPlacementCertificateView):
        raise TypeError("certificate must satisfy FinalPlacementCertificateView")
    from clay.layout.place.final_placement_state import (
        final_placement_state_digest as canonical_digest,
    )

    return canonical_digest(certificate)


def placement_geometry_digest(certificate: FinalPlacementCertificateView) -> str:
    """Hash the exact final-master geometry, excluding solver metadata churn."""

    if not isinstance(certificate, FinalPlacementCertificateView):
        raise TypeError("certificate must satisfy FinalPlacementCertificateView")
    # Reuse the strict typed replay before projecting the geometry-only identity.
    # This prevents a malformed member map from gaining a weaker CutLedger hash.
    final_placement_state_digest(certificate)
    blocks = _canonical_block_bboxes(certificate.blocks)
    payload = {
        "bbox_nm": list(certificate.bbox_nm),
        "blocks": [[name, list(bbox)] for name, bbox in blocks],
        "schema": "clay.final-placement-geometry.v2",
    }
    materialized_members = tuple(sorted(certificate.materialized_block_members))
    materialized_bboxes = tuple(sorted(certificate.materialized_block_bboxes))
    if materialized_members:
        payload["materialized_block_members"] = [
            [master, list(members)] for master, members in materialized_members
        ]
        payload["materialized_block_bboxes"] = [
            [name, list(bbox)] for name, bbox in materialized_bboxes
        ]
    return _sha256(payload)


def conflict_digest(conflict: CapacityConflict | HallDeficit) -> str:
    """Hash a replayable Task-5 conflict without accepting weaker certificates."""

    if isinstance(conflict, CapacityConflict):
        payload = {
            "candidate_ids": list(conflict.candidate_ids),
            "candidate_pool_hash": conflict.candidate_pool_hash,
            "declaration_refs": list(conflict.declaration_refs),
            "kind": "capacity_conflict",
            "nets": list(conflict.nets),
            "overlap_interval_nm": list(conflict.overlap_interval_nm),
            "resource": _resource_payload(conflict.resource),
            "track_capacity": conflict.track_capacity,
        }
    elif isinstance(conflict, HallDeficit):
        payload = {
            "candidate_ids": list(conflict.candidate_ids),
            "candidate_pool_hash": conflict.candidate_pool_hash,
            "declaration_refs": list(conflict.declaration_refs),
            "deficit": conflict.deficit,
            "demand_ids": list(conflict.demand_ids),
            "expansion_resource_ids": list(conflict.expansion_resource_ids),
            "kind": "hall_deficit",
            "reachable_site_ids": list(conflict.reachable_site_ids),
        }
    else:
        raise TypeError("only CapacityConflict and HallDeficit have replay digests")
    return _sha256(payload)


def lower_placement_conflict(
    *,
    pool: object,
    conflict: object,
    witness: object,
    final_placement_certificate: object,
) -> PlacementSolveCut | Unlowerable:
    """Lower one replayed, placement-causal conflict or return a named refusal."""

    if isinstance(conflict, str) or isinstance(witness, str):
        return _unlowerable(
            "string_provenance_prohibited",
            "string-only provenance cannot authorize a placement cut",
        )
    if _is_scalar(conflict) or _is_scalar(witness):
        return _unlowerable(
            "scalar_gap_prohibited",
            "a scalar gap/padding request is not a causal placement witness",
        )
    if isinstance(conflict, CapacityObservation):
        return _unlowerable(
            "nominal_observation_not_lowerable",
            "demand-free nominal capacity cannot become a placement cut",
        )
    if isinstance(conflict, PoolUnsatCertificate):
        return _unlowerable(
            "pool_unsat_not_lowerable",
            "generic candidate-pool UNSAT has no local placement cause",
        )
    if isinstance(conflict, CompleteNetSelectionResult):
        return _unlowerable(
            "unknown_not_lowerable"
            if conflict.status == "UNKNOWN"
            else "selection_result_not_lowerable",
            "a selection result must expose a replayed CapacityConflict/HallDeficit",
        )
    if not isinstance(conflict, (CapacityConflict, HallDeficit)):
        return _unlowerable(
            "foreign_conflict",
            f"unsupported conflict type {type(conflict).__name__}",
        )
    if not isinstance(pool, CandidatePool):
        return _unlowerable("foreign_pool", "pool must be CandidatePool")
    if not isinstance(final_placement_certificate, FinalPlacementCertificateView):
        return _unlowerable(
            "final_placement_certificate_missing",
            "a typed final placement certificate view is required",
        )
    if not isinstance(witness, PlacementRepairWitness):
        return _unlowerable(
            "typed_witness_missing",
            "a typed PlacementRepairWitness is required",
        )

    pool_hash = pool.sha256()
    placement_hash = final_placement_state_digest(final_placement_certificate)
    actual_conflict_hash = conflict_digest(conflict)
    context = {
        "conflict_hash": actual_conflict_hash,
        "candidate_pool_hash": pool_hash,
        "placement_hash": placement_hash,
    }
    if conflict.candidate_pool_hash != pool_hash:
        return _unlowerable(
            "conflict_pool_hash_mismatch",
            "conflict is not scoped to this candidate pool",
            **context,
        )
    if witness.conflict_hash != actual_conflict_hash:
        return _unlowerable(
            "conflict_hash_mismatch", "witness names a different conflict", **context
        )
    if witness.candidate_pool_hash != pool_hash:
        return _unlowerable(
            "pool_hash_mismatch", "witness names a different candidate pool", **context
        )
    if pool.placement_provenance_digest != placement_hash:
        return _unlowerable(
            "pool_placement_hash_mismatch",
            "candidate pool is not bound to the supplied final placement certificate",
            **context,
        )
    if witness.placement_hash != placement_hash:
        return _unlowerable(
            "placement_hash_mismatch", "witness names a different placement", **context
        )
    try:
        verify_infeasibility_certificate(pool, conflict)
    except Exception as exc:
        return _unlowerable(
            "conflict_replay_failed", f"conflict did not replay: {exc}", **context
        )
    if witness.declaration_refs != tuple(sorted(conflict.declaration_refs)):
        return _unlowerable(
            "declaration_refs_mismatch",
            "witness declaration refs differ from the replayed conflict",
            **context,
        )

    certificate_blocks = {name for name, _bbox in final_placement_certificate.blocks}
    if not witness.movable_block_ids:
        return _unlowerable(
            "movable_blocks_missing", "witness names no movable placement block", **context
        )
    unknown_blocks = sorted(set(witness.movable_block_ids) - certificate_blocks)
    if unknown_blocks:
        return _unlowerable(
            "movable_block_unknown",
            f"witness names blocks absent from final certificate: {unknown_blocks}",
            **context,
        )

    if isinstance(conflict, CapacityConflict):
        if witness.repair_kind not in (
            PlacementRepairKind.RELATIVE_BLOCK_ORDER,
            PlacementRepairKind.CERTIFIED_CHANNEL_SPAN,
        ):
            return _unlowerable(
                "repair_kind_mismatch",
                "capacity conflict requires block order or certified channel span",
                **context,
            )
        if len(witness.movable_block_ids) < 2:
            return _unlowerable(
                "movable_blocks_insufficient",
                "capacity repair requires at least two named movable blocks",
                **context,
            )
        if witness.corridor_relation is None:
            return _unlowerable(
                "corridor_relation_missing",
                "capacity repair requires a closed corridor relation",
                **context,
            )
        if witness.required_clearance_nm <= 0:
            return _unlowerable(
                "certified_clearance_missing",
                "capacity repair requires positive witness-derived clearance",
                **context,
            )
        payload = _verify_signed_causality_payload(
            pool=pool,
            conflict=conflict,
            witness=witness,
            certificate=final_placement_certificate,
            qualified_demand_ids=(),
            context=context,
        )
        if isinstance(payload, Unlowerable):
            return payload
        return PlacementSolveCut(
            repair_kind=witness.repair_kind,
            conflict_kind="capacity_conflict",
            conflict_hash=actual_conflict_hash,
            candidate_pool_hash=pool_hash,
            placement_hash=placement_hash,
            witness_id=witness.witness_id,
            nets=conflict.nets,
            candidate_ids=conflict.candidate_ids,
            demand_ids=(),
            qualified_demand_ids=(),
            movable_block_ids=witness.movable_block_ids,
            movable_site_ids=(),
            resource=conflict.resource,
            corridor_relation=witness.corridor_relation,
            required_clearance_nm=witness.required_clearance_nm,
            declaration_refs=conflict.declaration_refs,
        )

    if witness.repair_kind is not PlacementRepairKind.CANDIDATE_LANDING_ADJACENCY:
        return _unlowerable(
            "repair_kind_mismatch",
            "Hall deficit requires candidate landing adjacency",
            **context,
        )
    if not witness.movable_site_ids:
        return _unlowerable(
            "movable_sites_missing", "Hall witness names no movable landing site", **context
        )
    pool_sites = {site.site_id for site in pool.iter_landing_sites()}
    unknown_sites = sorted(set(witness.movable_site_ids) - pool_sites)
    if unknown_sites:
        return _unlowerable(
            "movable_site_unknown",
            f"witness names sites absent from candidate pool: {unknown_sites}",
            **context,
        )
    hall_entries = _replayed_qualified_hall_entries(
        pool, conflict, context=context
    )
    if isinstance(hall_entries, Unlowerable):
        return hall_entries
    qualified_demand_ids = tuple(
        (candidate.candidate_id, demand.demand_id)
        for candidate, demand in hall_entries
    )
    allowed_by_relevant_demands = {
        site_id
        for _candidate, demand in hall_entries
        for site_id in demand.allowed_site_ids
    }
    hall_repair_sites = (
        set(conflict.reachable_site_ids) & allowed_by_relevant_demands
    )
    noncausal_sites = sorted(set(witness.movable_site_ids) - hall_repair_sites)
    if noncausal_sites:
        return _unlowerable(
            "movable_site_not_hall_reachable",
            "witness sites are outside the replayed Hall demand/site graph: "
            f"{noncausal_sites}",
            **context,
        )
    payload = _verify_signed_causality_payload(
        pool=pool,
        conflict=conflict,
        witness=witness,
        certificate=final_placement_certificate,
        qualified_demand_ids=qualified_demand_ids,
        context=context,
    )
    if isinstance(payload, Unlowerable):
        return payload
    return PlacementSolveCut(
        repair_kind=witness.repair_kind,
        conflict_kind="hall_deficit",
        conflict_hash=actual_conflict_hash,
        candidate_pool_hash=pool_hash,
        placement_hash=placement_hash,
        witness_id=witness.witness_id,
        nets=(),
        candidate_ids=conflict.candidate_ids,
        demand_ids=conflict.demand_ids,
        qualified_demand_ids=qualified_demand_ids,
        movable_block_ids=witness.movable_block_ids,
        movable_site_ids=witness.movable_site_ids,
        resource=None,
        corridor_relation=None,
        required_clearance_nm=0,
        declaration_refs=conflict.declaration_refs,
    )


def _verify_signed_causality_payload(
    *,
    pool: CandidatePool,
    conflict: CapacityConflict | HallDeficit,
    witness: PlacementRepairWitness,
    certificate: FinalPlacementCertificateView,
    qualified_demand_ids: tuple[tuple[str, str], ...],
    context: dict[str, str],
) -> PlacementCausalityPayload | Unlowerable:
    signed_payloads = _certificate_causality_payloads(certificate)
    if not signed_payloads:
        return _unlowerable(
            "causality_signature_unavailable",
            "final placement certificate carries no signed placement/resource cause",
            **context,
        )
    if witness.causality_payload_digest is None:
        return _unlowerable(
            "causality_payload_missing",
            "witness must reference a payload already signed by final placement",
            **context,
        )
    matches = tuple(
        payload
        for payload in signed_payloads
        if payload.payload_digest == witness.causality_payload_digest
    )
    if not matches:
        return _unlowerable(
            "causality_payload_unknown",
            "witness payload was not signed by the final placement certificate",
            **context,
        )
    payload = matches[0]
    certificate_blocks = dict(certificate.blocks)
    signed_blocks = dict(payload.block_bboxes)
    expected_signed_blocks = {
        block_id: certificate_blocks[block_id]
        for block_id in payload.resource_owner_block_ids
        if block_id in certificate_blocks
    }
    general_mismatch = (
        payload.repair_kind is not witness.repair_kind
        or payload.resource_owner_block_ids != witness.movable_block_ids
        or signed_blocks != expected_signed_blocks
        or payload.candidate_ids != tuple(sorted(conflict.candidate_ids))
        or payload.corridor_relation != witness.corridor_relation
        or payload.target_required_clearance_nm != witness.required_clearance_nm
        or payload.movable_site_ids != witness.movable_site_ids
        or payload.declaration_refs != tuple(sorted(conflict.declaration_refs))
    )
    if general_mismatch:
        return _unlowerable(
            "causality_payload_mismatch",
            "signed placement/resource facts differ from the proposed witness",
            **context,
        )

    if isinstance(conflict, CapacityConflict):
        resource = next(
            (item for item in pool.resources if item.key == conflict.resource), None
        )
        expected_relation = (
            "vertical_separation"
            if resource is not None and resource.preferred_axis == "H"
            else "horizontal_separation"
            if resource is not None and resource.preferred_axis == "V"
            else None
        )
        if resource is None or payload.resource != conflict.resource:
            return _unlowerable(
                "causality_resource_mismatch",
                "signed resource does not replay the capacity conflict",
                **context,
            )
        if (
            payload.preferred_axis != resource.preferred_axis
            or payload.corridor_relation != expected_relation
        ):
            return _unlowerable(
                "causality_axis_mismatch",
                "capacity repair separation must be perpendicular to track direction",
                **context,
            )
        active_uses = _replayed_active_capacity_uses(
            pool, conflict, context=context
        )
        if isinstance(active_uses, Unlowerable):
            return active_uses
        active_use_ids = tuple(
            sorted(
                (candidate.candidate_id, use.use_id)
                for candidate, use in active_uses
            )
        )
        capacity_deficit = (
            sum(use.track_span for _candidate, use in active_uses)
            - conflict.track_capacity
        )
        track_pitch_nm = _explicit_positive_track_pitch_nm(
            resource.track_centers_nm
        )
        if track_pitch_nm is None:
            return _unlowerable(
                "causality_pitch_unavailable",
                "explicit resource tracks do not define one positive pitch",
                **context,
            )
        measured_clearance = _measured_pair_clearance_nm(
            payload.block_bboxes, payload.corridor_relation
        )
        if measured_clearance is None:
            return _unlowerable(
                "causality_owner_relation_unavailable",
                "signed resource owners do not bound the perpendicular corridor",
                **context,
            )
        if payload.active_use_ids != active_use_ids:
            return _unlowerable(
                "causality_active_uses_mismatch",
                "signed active corridor uses do not replay the capacity conflict",
                **context,
            )
        if payload.observed_clearance_nm != measured_clearance:
            return _unlowerable(
                "causality_observed_clearance_mismatch",
                "signed observed clearance differs from final block bboxes",
                **context,
            )
        target_required_clearance_nm = (
            measured_clearance + capacity_deficit * track_pitch_nm
        )
        if (
            payload.capacity_deficit != capacity_deficit
            or payload.track_pitch_nm != track_pitch_nm
            or payload.target_required_clearance_nm
            != target_required_clearance_nm
        ):
            return _unlowerable(
                "causality_target_clearance_mismatch",
                "signed target must equal observed + capacity_deficit * track_pitch",
                **context,
            )
        if measured_clearance >= target_required_clearance_nm:
            return _unlowerable(
                "vacuous_capacity_cut",
                "current placement already satisfies the proposed capacity cut",
                **context,
            )
        if payload.qualified_demand_ids:
            return _unlowerable(
                "causality_resource_mismatch",
                "capacity payload cannot contain Hall demand identities",
                **context,
            )
    else:
        if (
            payload.resource is not None
            or payload.preferred_axis is not None
            or payload.corridor_relation is not None
            or payload.observed_clearance_nm != 0
            or payload.target_required_clearance_nm != 0
            or payload.active_use_ids
            or payload.capacity_deficit != 0
            or payload.track_pitch_nm != 0
            or payload.qualified_demand_ids != qualified_demand_ids
        ):
            return _unlowerable(
                "causality_hall_graph_mismatch",
                "signed candidate-local Hall facts do not replay the deficit",
                **context,
            )
    return payload


def _replayed_active_capacity_uses(
    pool: CandidatePool,
    conflict: CapacityConflict,
    *,
    context: dict[str, str],
) -> tuple[tuple[Any, Any], ...] | Unlowerable:
    fixed_candidates = tuple(
        net_candidates[0]
        for net in pool.required_nets
        for net_candidates in (pool.candidates_for(net),)
        if len(net_candidates) == 1
    )
    start, end = conflict.overlap_interval_nm
    active = tuple(
        sorted(
            (
                (candidate, use)
                for candidate in fixed_candidates
                for use in candidate.corridor_uses
                if use.resource == conflict.resource
                and use.interval_nm[0] <= start
                and end <= use.interval_nm[1]
            ),
            key=lambda item: (
                item[0].net,
                item[0].candidate_id,
                item[1].use_id,
            ),
        )
    )
    resource = next(
        (item for item in pool.resources if item.key == conflict.resource), None
    )
    expected_refs = tuple(
        sorted(
            {
                ref
                for candidate, _use in active
                for ref in candidate.required_goal_refs
            }
        )
    )
    if (
        not active
        or resource is None
        or conflict.track_capacity != len(resource.track_centers_nm)
        or tuple(candidate.net for candidate, _use in active) != conflict.nets
        or tuple(candidate.candidate_id for candidate, _use in active)
        != conflict.candidate_ids
        or expected_refs != tuple(conflict.declaration_refs)
        or sum(use.track_span for _candidate, use in active)
        <= conflict.track_capacity
    ):
        return _unlowerable(
            "capacity_active_use_replay_failed",
            "capacity conflict drifted from its exact active corridor uses",
            **context,
        )
    return active


def _explicit_positive_track_pitch_nm(
    track_centers_nm: tuple[int, ...],
) -> Optional[int]:
    centers = tuple(sorted(track_centers_nm))
    if len(centers) < 2:
        return None
    pitches = tuple(
        right - left for left, right in zip(centers, centers[1:])
    )
    if any(pitch <= 0 for pitch in pitches) or len(set(pitches)) != 1:
        return None
    return pitches[0]


def _feedback_landing_entries_matchable(
    entries: tuple[tuple[Any, Any], ...], capacities: dict[str, int]
) -> bool:
    if not entries:
        return True
    slots_by_site = {
        site_id: tuple(
            (site_id, slot) for slot in range(min(capacity, len(entries)))
        )
        for site_id, capacity in sorted(capacities.items())
    }
    matched: dict[tuple[str, int], int] = {}

    def augment(entry_index: int, seen: set[tuple[str, int]]) -> bool:
        demand = entries[entry_index][1]
        for site_id in demand.allowed_site_ids:
            if site_id not in slots_by_site:
                return False
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


def _feedback_hall_entries_for_candidates(
    candidates: tuple[Any, ...], capacities: dict[str, int]
) -> tuple[tuple[Any, Any], ...]:
    current = tuple(
        sorted(
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
        )
    )
    if _feedback_landing_entries_matchable(current, capacities):
        return ()
    changed = True
    while changed:
        changed = False
        for index in range(len(current)):
            smaller = current[:index] + current[index + 1 :]
            if smaller and not _feedback_landing_entries_matchable(
                smaller, capacities
            ):
                current = smaller
                changed = True
                break
    return current


def _replayed_qualified_hall_entries(
    pool: CandidatePool,
    conflict: HallDeficit,
    *,
    context: dict[str, str],
) -> tuple[tuple[Any, Any], ...] | Unlowerable:
    """Rebuild Task-5's canonical Hall subset with candidate-local demand keys."""

    fixed_candidates = tuple(
        net_candidates[0]
        for net in pool.required_nets
        for net_candidates in (pool.candidates_for(net),)
        if len(net_candidates) == 1
    )
    capacities = {
        site.site_id: site.capacity for site in pool.iter_landing_sites()
    }
    entries = _feedback_hall_entries_for_candidates(fixed_candidates, capacities)
    if (
        tuple(demand.demand_id for _candidate, demand in entries)
        != conflict.demand_ids
        or tuple(sorted({candidate.candidate_id for candidate, _demand in entries}))
        != tuple(sorted(conflict.candidate_ids))
        or tuple(
            sorted(
                {
                    site_id
                    for _candidate, demand in entries
                    for site_id in demand.allowed_site_ids
                }
            )
        )
        != tuple(conflict.reachable_site_ids)
    ):
        return _unlowerable(
            "hall_identity_replay_failed",
            "verified Hall certificate drifted from candidate-qualified demands",
            **context,
        )
    return entries


def _certificate_causality_payloads(
    certificate: FinalPlacementCertificateView,
) -> tuple[PlacementCausalityPayload, ...]:
    raw_payloads = getattr(certificate, "placement_causality_payloads", ())
    if raw_payloads is None:
        return ()
    payloads = tuple(raw_payloads)
    if not all(isinstance(payload, PlacementCausalityPayload) for payload in payloads):
        raise TypeError(
            "placement_causality_payloads must contain PlacementCausalityPayload"
        )
    ordered = tuple(sorted(payloads, key=lambda payload: payload.payload_digest))
    digests = tuple(payload.payload_digest for payload in ordered)
    if len(digests) != len(set(digests)):
        raise ValueError("placement certificate contains duplicate causality payloads")
    return ordered


def _measured_pair_clearance_nm(
    block_bboxes: tuple[tuple[str, _BBoxNm], ...], relation: Optional[str]
) -> Optional[int]:
    if len(block_bboxes) != 2:
        return None
    left = block_bboxes[0][1]
    right = block_bboxes[1][1]
    if relation == "horizontal_separation":
        if max(left[1], right[1]) >= min(left[3], right[3]):
            return None
        return max(0, left[0] - right[2], right[0] - left[2])
    if relation == "vertical_separation":
        if max(left[0], right[0]) >= min(left[2], right[2]):
            return None
        return max(0, left[1] - right[3], right[1] - left[3])
    return None


def _verify_master_cut_replay(
    certificate: FinalPlacementCertificateView,
    cuts: tuple[PlacementSolveCut, ...],
    *,
    candidate_pool_hash: str,
    placement_hash: str,
) -> Optional[Unlowerable]:
    """Require exact master-produced application and geometry replay for all cuts."""

    expected_hashes = tuple(sorted(cut.cut_hash for cut in cuts))
    try:
        applied_hashes = tuple(certificate.applied_cut_hashes)
    except (AttributeError, TypeError):
        applied_hashes = ()
    if applied_hashes != expected_hashes:
        return _unlowerable(
            "master_cut_replay_missing",
            "final placement certificate does not name every cumulative cut "
            "applied by the master",
            candidate_pool_hash=candidate_pool_hash,
            placement_hash=placement_hash,
        )
    try:
        by_name = dict(_canonical_block_bboxes(certificate.blocks))
    except (TypeError, ValueError) as exc:
        return _unlowerable(
            "master_cut_replay_failed",
            f"final master geometry could not replay cumulative cuts: {exc}",
            candidate_pool_hash=candidate_pool_hash,
            placement_hash=placement_hash,
        )
    for cut in cuts:
        if any(owner not in by_name for owner in cut.movable_block_ids):
            return _unlowerable(
                "master_cut_replay_failed",
                f"final master omits a member of cut {cut.cut_hash}",
                candidate_pool_hash=candidate_pool_hash,
                placement_hash=placement_hash,
            )
        if cut.repair_kind is PlacementRepairKind.EXACT_PLACEMENT_NOGOOD:
            final_origins = tuple(
                (owner, by_name[owner][0], by_name[owner][1])
                for owner in cut.movable_block_ids
            )
            if final_origins == cut.incumbent_origins:
                return _unlowerable(
                    "master_cut_replay_failed",
                    f"final master did not exclude origin vector {cut.cut_hash}",
                    candidate_pool_hash=candidate_pool_hash,
                    placement_hash=placement_hash,
                )
            continue
        if cut.guarded_coordinates and any(
            by_name[name][0 if axis == "x" else 1] != incumbent
            for name, axis, incumbent in cut.guarded_coordinates
        ):
            continue
        owner_ids = (
            cut.oriented_owner_block_ids
            if cut.guarded_coordinates
            else cut.movable_block_ids
        )
        left = by_name[owner_ids[0]]
        right = by_name[owner_ids[1]]
        clearance = cut.required_clearance_nm
        if cut.corridor_relation == "horizontal_separation":
            satisfied = (
                left[2] + clearance <= right[0]
                or right[2]
                + (0 if cut.guarded_coordinates else clearance)
                <= left[0]
            )
        elif cut.corridor_relation == "vertical_separation":
            satisfied = (
                left[3] + clearance <= right[1]
                or right[3]
                + (0 if cut.guarded_coordinates else clearance)
                <= left[1]
            )
        else:
            satisfied = False
        if not satisfied:
            return _unlowerable(
                "master_cut_replay_failed",
                f"final master geometry does not satisfy cut {cut.cut_hash}",
                candidate_pool_hash=candidate_pool_hash,
                placement_hash=placement_hash,
            )
    return None


def _typed_feedback_scope(
    *,
    pool: object,
    final_placement_certificate: object,
) -> PlacementFeedbackScope | Unlowerable:
    """Derive one scope only from a pool bound to the exact placement digest."""

    if not isinstance(pool, CandidatePool):
        return _unlowerable("foreign_pool", "feedback scope requires CandidatePool")
    if not isinstance(final_placement_certificate, FinalPlacementCertificateView):
        return _unlowerable(
            "final_placement_certificate_missing",
            "feedback scope requires a typed final placement certificate view",
        )
    try:
        placement_hash = final_placement_state_digest(final_placement_certificate)
        candidate_pool_hash = pool.sha256()
    except (AttributeError, TypeError, ValueError) as exc:
        return _unlowerable(
            "invalid_feedback_scope",
            f"feedback scope could not be canonicalized: {type(exc).__name__}",
        )
    if pool.placement_provenance_digest != placement_hash:
        return _unlowerable(
            "scope_pool_placement_mismatch",
            "candidate pool is not bound to the exact final placement certificate",
            candidate_pool_hash=candidate_pool_hash,
            placement_hash=placement_hash,
        )
    return PlacementFeedbackScope(
        candidate_pool_hash=candidate_pool_hash,
        placement_hash=placement_hash,
    )


def _unlowerable(
    reason_code: str,
    detail: str,
    *,
    conflict_hash: Optional[str] = None,
    candidate_pool_hash: Optional[str] = None,
    placement_hash: Optional[str] = None,
) -> Unlowerable:
    return Unlowerable(
        reason_code=reason_code,
        detail=detail,
        conflict_hash=conflict_hash,
        candidate_pool_hash=candidate_pool_hash,
        placement_hash=placement_hash,
    )


def _resource_payload(resource: Optional[CorridorLayerKey]) -> Optional[dict[str, Any]]:
    if resource is None:
        return None
    return {"corridor_id": resource.corridor_id, "layer_n": resource.layer_n}


def _causality_payload(payload: PlacementCausalityPayload) -> dict[str, Any]:
    return {
        "active_use_ids": [list(key) for key in payload.active_use_ids],
        "block_bboxes": [
            [name, list(bbox)] for name, bbox in payload.block_bboxes
        ],
        "capacity_deficit": payload.capacity_deficit,
        "candidate_ids": list(payload.candidate_ids),
        "corridor_relation": payload.corridor_relation,
        "declaration_refs": list(payload.declaration_refs),
        "movable_site_ids": list(payload.movable_site_ids),
        "observed_clearance_nm": payload.observed_clearance_nm,
        "preferred_axis": payload.preferred_axis,
        "qualified_demand_ids": [
            list(key) for key in payload.qualified_demand_ids
        ],
        "repair_kind": payload.repair_kind.value,
        "target_required_clearance_nm": payload.target_required_clearance_nm,
        "track_pitch_nm": payload.track_pitch_nm,
        "resource": _resource_payload(payload.resource),
        "resource_owner_block_ids": list(payload.resource_owner_block_ids),
        "schema": "clay.placement-causality-payload.v1",
    }


def _canonical_block_bboxes(values: Any) -> tuple[tuple[str, _BBoxNm], ...]:
    result: list[tuple[str, _BBoxNm]] = []
    for entry in values:
        try:
            name, raw_bbox = entry
            bbox = tuple(raw_bbox)
        except Exception as exc:
            raise ValueError("block_bboxes must contain (name, bbox) pairs") from exc
        if not isinstance(name, str) or not name:
            raise ValueError("block bbox names must be non-empty strings")
        if (
            len(bbox) != 4
            or any(not isinstance(value, int) or isinstance(value, bool) for value in bbox)
            or bbox[2] < bbox[0]
            or bbox[3] < bbox[1]
        ):
            raise ValueError("block bboxes must be ordered integer nm quadruples")
        result.append((name, bbox))
    names = [name for name, _bbox in result]
    if not result or len(names) != len(set(names)):
        raise ValueError("block_bboxes must name unique non-empty blocks")
    return tuple(sorted(result))


def _sorted_unique_strings(values: Any, field_name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique non-empty strings")
    return tuple(sorted(result))


def _sorted_strings_allow_duplicates(values: Any, field_name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return tuple(sorted(result))


def _sorted_unique_pairs(values: Any, field_name: str) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for value in values:
        try:
            left, right = value
        except Exception as exc:
            raise ValueError(f"{field_name} must contain string pairs") from exc
        pair = (str(left), str(right))
        if not pair[0] or not pair[1]:
            raise ValueError(f"{field_name} must contain non-empty string pairs")
        result.append(pair)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique pairs")
    return tuple(sorted(result))


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("final placement certificate contains a non-finite value")
    return result


def _optional_finite_float(value: Any) -> Optional[float]:
    return None if value is None else _finite_float(value)


def _is_scalar(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CutLedger",
    "ExactPlacementNoGoodProof",
    "FinalPlacementCertificateView",
    "PlacementCausalityBasis",
    "PlacementCausalityPayload",
    "PlacementConflictReplayCertificate",
    "PlacementCorridorBoundary",
    "PlacementCutReplayProof",
    "PlacementFeedbackScope",
    "PlacementRepairKind",
    "PlacementRepairRoundCertificate",
    "PlacementRepairWitness",
    "PlacementResourceClosureCertificate",
    "PlacementSolveCut",
    "RoutePlanningFeedbackContext",
    "RoutePlanningFeedbackContextError",
    "Unlowerable",
    "attach_proven_landing_expansion_resources",
    "conflict_digest",
    "certify_proven_feasible_resources",
    "build_placement_causality_basis",
    "final_placement_digest",
    "final_placement_state_digest",
    "lower_placement_conflict",
    "lower_exact_placement_nogood",
    "lower_route_planning_feedback",
    "placement_geometry_digest",
    "verify_placement_causality_basis",
]
