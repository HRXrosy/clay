"""Exact route-ownership manifest replay for the canonical allocation path.

The canonical emitter needs an ownership manifest for the geometry produced by
the exact route-resource plan.  Raw router tags are not such evidence: this
module accepts only the closed candidate pool, its verified selected plan, the
final Gridless replay certificate, and the current PDK contract.

The returned tuple uses the same record order and rectangle construction as the
Gridless exact-plan materializer.  Existing owned conductors are deliberately
absent because Gridless traverses, but does not re-emit, that artwork.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Tuple

LayerKey = Tuple[int, int]
BBoxNm = Tuple[int, int, int, int]
RouteManifestEntry = Tuple[str, LayerKey, BBoxNm]
RouteManifest = Tuple[RouteManifestEntry, ...]


class ExactAllocatedRouteManifestReplayError(ValueError):
    """The exact allocation inputs do not replay to one bound manifest."""


def _fail(message: str) -> None:
    raise ExactAllocatedRouteManifestReplayError(message)


def _layer_key(value: Any, label: str) -> LayerKey:
    if (
        type(value) not in (tuple, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        _fail(f"{label} is not an exact integer GDS layer key")
    return int(value[0]), int(value[1])


def _bbox(value: Any, label: str) -> BBoxNm:
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        _fail(f"{label} is not an exact integer bbox")
    box = tuple(int(item) for item in value)
    if box[2] <= box[0] or box[3] <= box[1]:
        _fail(f"{label} is empty")
    return box  # type: ignore[return-value]


def _bounded_tuple(value: Any, limit: int, label: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        _fail(f"{label} is not a canonical tuple")
    if len(value) > limit:
        _fail(f"{label} exceeds bounded limit {limit}")
    return value


def _pdk_metal_key(tech: Any, layer_n: int) -> LayerKey:
    try:
        return _layer_key(tech.metal_key(layer_n), f"M{layer_n}")
    except ExactAllocatedRouteManifestReplayError:
        raise
    except Exception as exc:
        raise ExactAllocatedRouteManifestReplayError(
            f"current PDK does not declare M{layer_n}"
        ) from exc


def _pdk_via_key(tech: Any, lower_layer_n: int) -> LayerKey:
    try:
        return _layer_key(
            tech.via_key(lower_layer_n), f"VIA{lower_layer_n}"
        )
    except ExactAllocatedRouteManifestReplayError:
        raise
    except Exception as exc:
        raise ExactAllocatedRouteManifestReplayError(
            f"current PDK does not declare VIA{lower_layer_n}"
        ) from exc


def _track_bbox(allocation: Any) -> BBoxNm:
    interval = getattr(allocation, "interval_nm", None)
    centers = getattr(allocation, "selected_track_centers_nm", None)
    width = getattr(allocation, "width_nm", None)
    resource = getattr(allocation, "resource", None)
    if (
        type(interval) is not tuple
        or len(interval) != 2
        or any(type(value) is not int for value in interval)
        or interval[1] <= interval[0]
        or type(centers) is not tuple
        or not centers
        or any(type(value) is not int for value in centers)
        or type(width) is not int
        or width <= 0
    ):
        _fail("allocated track geometry is incomplete")
    axis = getattr(resource, "preferred_axis", None)
    low_half = width // 2
    high_half = width - low_half
    if axis == "H":
        value = (
            interval[0],
            centers[0] - low_half,
            interval[1],
            centers[-1] + high_half,
        )
    elif axis == "V":
        value = (
            centers[0] - low_half,
            interval[0],
            centers[-1] + high_half,
            interval[1],
        )
    else:
        _fail("allocated track has no exact preferred axis")
    return _bbox(value, "allocated track bbox")


def replay_exact_allocated_route_manifest(
    *,
    candidate_pool: Any,
    resource_plan: Any,
    materialization_certificate: Any,
    routing_world: Any,
    tech: Any,
) -> RouteManifest:
    """Deep-replay one exact allocation into its emitter ownership manifest.

    No geometry, ownership tags, circuit identity, or legacy route result may be
    supplied by the caller.  All manifest entries are reconstructed from the
    selected allocation after its pool, plan, world, PDK, and materialization
    certificate bindings have replayed.
    """
    from clay.layout.route.engine_gridless import (
        GridlessAllocationReplayCertificate,
        _verify_gridless_allocation_replay_certificate,
    )
    from clay.layout.route.resource_plan import (
        CandidatePool,
        MAX_LANDING_OPTION_SITE_IDS,
        MAX_REQUIRED_NETS,
        MAX_TOTAL_BRANCH_STEPS,
        MAX_TOTAL_LANDING_OPTION_SITE_CLAIMS,
        MAX_TOTAL_LANDING_OPTIONS,
        RouteResourcePlan,
        RouteResourcePlanVerificationBudgetExhausted,
        verify_route_resource_plan,
    )
    from clay.layout.route.resource_world import (
        RoutingWorldProvenance,
        pdk_provenance_digest,
        verify_candidate_pool_applicability,
    )

    if type(candidate_pool) is not CandidatePool:
        raise TypeError("exact CandidatePool is required")
    if type(resource_plan) is not RouteResourcePlan:
        raise TypeError("exact RouteResourcePlan is required")
    if type(materialization_certificate) is not GridlessAllocationReplayCertificate:
        raise TypeError("exact Gridless allocation replay certificate is required")
    _verify_gridless_allocation_replay_certificate(materialization_certificate)
    if type(routing_world) is not RoutingWorldProvenance:
        raise TypeError("exact RoutingWorldProvenance is required")

    try:
        plan_replays = verify_route_resource_plan(candidate_pool, resource_plan)
    except RouteResourcePlanVerificationBudgetExhausted:
        raise
    except Exception as exc:
        raise ExactAllocatedRouteManifestReplayError(
            "route resource plan does not replay against the candidate pool"
        ) from exc
    if plan_replays is not True:
        _fail("route resource plan does not replay against the candidate pool")

    pool_hash = candidate_pool.sha256()
    plan_hash = resource_plan.sha256()
    current_pdk_digest = pdk_provenance_digest(tech)
    expected_selected = tuple(
        (selected.net, selected.candidate_id)
        for selected in resource_plan.selected_nets
    )
    programs = _bounded_tuple(
        resource_plan.allocated_programs,
        MAX_REQUIRED_NETS,
        "allocated route programs",
    )
    if not programs:
        _fail("route resource plan has no exact allocated programs")
    world_digests = tuple(sorted({
        program.routing_world_provenance_digest for program in programs
    }))
    if (
        not candidate_pool.pdk_provenance_digest
        or candidate_pool.pdk_provenance_digest != current_pdk_digest
    ):
        _fail("candidate pool is stale for the current PDK")
    if (
        not candidate_pool.routing_world_provenance_digest
        or world_digests != (candidate_pool.routing_world_provenance_digest,)
    ):
        _fail("allocated programs are stale for the candidate-pool routing world")
    if resource_plan.candidate_pool_hash != pool_hash:
        _fail("route resource plan is stale for the candidate pool")
    if resource_plan.materialization_complete is not True:
        _fail("route resource plan does not contain complete allocation IR")

    current_world_digest = routing_world.sha256()
    world_bindings = (
        candidate_pool.required_nets == routing_world.required_nets,
        candidate_pool.placement_provenance_digest
        == routing_world.placement_provenance_digest,
        candidate_pool.topology_provenance_digest
        == routing_world.topology_provenance_digest,
        candidate_pool.pdk_provenance_digest
        == routing_world.pdk_provenance_digest,
        candidate_pool.routing_world_provenance_digest == current_world_digest,
    )
    if not all(world_bindings):
        _fail("candidate pool is stale for the current routing world")

    certificate = materialization_certificate
    certificate_bindings = (
        certificate.candidate_pool_hash == pool_hash,
        certificate.resource_plan_hash == plan_hash,
        certificate.selected_candidates == expected_selected,
        certificate.materialization_complete is True,
        certificate.current_pdk_provenance_digest == current_pdk_digest,
        certificate.routing_world_provenance_digests == world_digests,
    )
    if not all(certificate_bindings):
        _fail("materialization certificate has a stale exact-plan binding")

    try:
        applicable = verify_candidate_pool_applicability(
            candidate_pool, routing_world, tech=tech
        )
    except Exception as exc:
        raise ExactAllocatedRouteManifestReplayError(
            "candidate pool does not exactly replay from the current routing world"
        ) from exc
    if applicable is not True:
        _fail("candidate pool does not exactly replay from the current routing world")

    entries: list[RouteManifestEntry] = []
    digest_records: list[tuple[str, int, int, BBoxNm]] = []
    track_total = 0
    landing_option_total = 0
    landing_site_total = 0
    for program in programs:
        net = program.net
        if type(net) is not str or not net:
            _fail("allocated route program has no exact net owner")
        tracks = _bounded_tuple(
            program.tracks,
            MAX_TOTAL_BRANCH_STEPS,
            "allocated tracks",
        )
        track_total += len(tracks)
        if track_total > MAX_TOTAL_BRANCH_STEPS:
            _fail(f"allocated tracks exceeds bounded limit {MAX_TOTAL_BRANCH_STEPS}")
        for allocation in tracks:
            key = _layer_key(
                allocation.resource.gds_layer_key,
                "allocated track layer",
            )
            current_key = _pdk_metal_key(
                tech, allocation.resource.key.layer_n
            )
            if key != current_key:
                _fail("allocated track layer is stale for the current PDK")
            box = _track_bbox(allocation)
            entries.append((net, key, box))
            digest_records.append((net, key[0], key[1], box))

        landing_options = _bounded_tuple(
            program.landing_options,
            MAX_TOTAL_LANDING_OPTIONS,
            "allocated landing options",
        )
        landing_option_total += len(landing_options)
        if landing_option_total > MAX_TOTAL_LANDING_OPTIONS:
            _fail(
                "allocated landing options exceeds bounded limit "
                f"{MAX_TOTAL_LANDING_OPTIONS}"
            )
        for allocation in landing_options:
            sites = _bounded_tuple(
                allocation.landing_sites,
                MAX_LANDING_OPTION_SITE_IDS,
                "allocated landing sites",
            )
            landing_site_total += len(sites)
            if landing_site_total > MAX_TOTAL_LANDING_OPTION_SITE_CLAIMS:
                _fail(
                    "allocated landing sites exceeds bounded limit "
                    f"{MAX_TOTAL_LANDING_OPTION_SITE_CLAIMS}"
                )
            for site in sites:
                lower_key = _pdk_metal_key(tech, site.lower_layer_n)
                upper_key = _pdk_metal_key(tech, site.upper_layer_n)
                via_key = _pdk_via_key(tech, site.lower_layer_n)
                emitted = (
                    (lower_key, _bbox(site.lower_pad_bbox_nm, "lower landing pad")),
                    (upper_key, _bbox(site.upper_pad_bbox_nm, "upper landing pad")),
                    (via_key, _bbox(site.cut_bbox_nm, "landing via cut")),
                )
                for key, box in emitted:
                    entries.append((net, key, box))
                    digest_records.append((net, key[0], key[1], box))

    geometry_payload = [
        [net, layer, datatype, list(box)]
        for net, layer, datatype, box in digest_records
    ]
    geometry_digest = hashlib.sha256(json.dumps(
        geometry_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if certificate.geometry_digest != geometry_digest:
        _fail("materialization geometry digest does not replay")

    manifest = tuple(entries)
    if len(manifest) != len(set(manifest)):
        _fail("allocated route replay contains duplicate ownership entries")
    return manifest


__all__ = [
    "ExactAllocatedRouteManifestReplayError",
    "RouteManifest",
    "RouteManifestEntry",
    "replay_exact_allocated_route_manifest",
]
