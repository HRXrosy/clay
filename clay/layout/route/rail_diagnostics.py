"""Immutable observations of a stopped, bounded rail-placement invocation.

These snapshots explain the supplied scene and candidate pool. They have no
production authority and provide no loader for live axis receipts, placement
certificates, or solver execution. Geometry occurs once in the scene; blocker
rows refer to its canonical conductor-fact indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
from clay.layout.route.rail_plan import (
    MAX_COORD_ABS_NM,
    RailPlacementCandidatePool,
    RailPlacementInfeasibilityCertificate,
    RailPlacementScene,
    RailPlacementSelectionResult,
    _bbox_is_on_declared_internal_edge,
    _candidate_bbox,
    _canonical_json,
    _required_metal_rule,
    _violates_existing_conductor_spacing,
)
from clay.layout.route.resource_world import pdk_provenance_digest


_SCHEMA = "clay.rail-placement-stop-observation.v1"


@dataclass(frozen=True)
class RailPlacementStopObservation:
    """Frozen diagnostic bytes, rather than aliases to mutable live operands."""

    _snapshot_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self._snapshot_json) is not str:
            raise TypeError("rail stop observation requires canonical JSON text")
        payload = json.loads(self._snapshot_json)
        if (
            type(payload) is not dict
            or payload.get("schema") != _SCHEMA
            or payload.get("authority") != "diagnostic_only"
            or _canonical_json(payload) != self._snapshot_json
        ):
            raise ValueError("invalid canonical rail stop observation")

    def canonical_json(self) -> str:
        return self._snapshot_json

    def payload(self) -> dict[str, object]:
        """Return a fresh plain-data copy; this does not restore live authority."""
        return json.loads(self._snapshot_json)

    def sha256(self) -> str:
        return hashlib.sha256(self._snapshot_json.encode("utf-8")).hexdigest()

    def require_stop(self, *, phase: str, state: str, evidence: object) -> None:
        """Check that a stop transports the result captured in these bytes."""
        payload = self.payload()
        result = payload["result"]
        if phase != "RAIL_PLACEMENT":
            raise ValueError("rail observation requires a RAIL_PLACEMENT stop")
        if result["status"] == "INFEASIBLE":
            if type(evidence) is not RailPlacementInfeasibilityCertificate:
                raise TypeError("rail observation requires its typed infeasibility evidence")
            matches = (
                state == "INFEASIBLE"
                and json.loads(evidence.canonical_json())
                == payload["infeasibility_certificate"]
                and evidence.sha256() == result["certificate_hash"]
            )
        elif result["status"] == "UNKNOWN":
            if type(evidence) is not RailPlacementSelectionResult:
                raise TypeError("rail observation requires its typed UNKNOWN evidence")
            matches = (
                state == result["solver_status"]
                and json.loads(evidence.canonical_json()) == result
            )
        else:
            matches = False
        if not matches:
            raise ValueError("rail stop observation differs from stop evidence")


def observe_rail_placement_stop(
    *,
    scene: RailPlacementScene,
    pool: RailPlacementCandidatePool,
    result: RailPlacementSelectionResult,
    tech: object,
) -> RailPlacementStopObservation:
    """Capture a failed/unknown invocation without solving or changing its pool.

    The caller has already enumerated and verified the bounded pool. At most
    MAX_RAIL_DEMANDS * MAX_RAIL_SCENE_FACTS conductor predicates are evaluated;
    every matching index is retained, without sampling or duplicated geometry.
    Full compiled declarations (including their source observations) occur in
    the candidate pool's existing canonical payload.
    """
    if type(scene) is not RailPlacementScene:
        raise TypeError("scene must be RailPlacementScene")
    if type(pool) is not RailPlacementCandidatePool:
        raise TypeError("pool must be RailPlacementCandidatePool")
    if type(result) is not RailPlacementSelectionResult:
        raise TypeError("result must be RailPlacementSelectionResult")
    if result.status not in ("INFEASIBLE", "UNKNOWN"):
        raise ValueError("rail stop observation requires INFEASIBLE or UNKNOWN")

    scene_json = scene.canonical_json()
    pool_json = pool.canonical_json()
    scene_hash = hashlib.sha256(scene_json.encode("utf-8")).hexdigest()
    pool_hash = hashlib.sha256(pool_json.encode("utf-8")).hexdigest()
    if pool.rail_placement_scene_hash != scene_hash:
        raise ValueError("rail observation scene and pool hashes differ")
    if result.candidate_pool_hash != pool_hash:
        raise ValueError("rail observation result and pool hashes differ")
    for name in (
        "placement_provenance_digest", "pdk_provenance_digest",
        "declaration_provenance_digest",
    ):
        if getattr(scene, name) != getattr(pool, name):
            raise ValueError(f"rail observation scene/pool {name} differs")
    if pdk_provenance_digest(tech) != scene.pdk_provenance_digest:
        raise ValueError("rail observation scene differs from current PDK")

    observations = []
    for demand in pool.demands:
        cross_low, cross_high = (
            (scene.cell_bbox_nm[1], scene.cell_bbox_nm[3])
            if demand.preferred_axis == "H"
            else (scene.cell_bbox_nm[0], scene.cell_bbox_nm[2])
        )
        start = (
            cross_low + demand.offset_nm if demand.edge in {"bottom", "left"}
            else cross_high - demand.width_nm - demand.offset_nm
        )
        stripe = _candidate_bbox(scene.cell_bbox_nm, demand, start)
        terminal_count = sum(fact.net == demand.net for fact in scene.terminal_facts)
        spacing = _required_metal_rule(tech, demand.layer_n, "min_space")
        blockers = [
            index for index, fact in enumerate(scene.conductor_facts)
            if _violates_existing_conductor_spacing(stripe, demand, (fact,), spacing)
        ]
        reasons = []
        if not terminal_count:
            reasons.append("missing_terminal_facts")
        if any(abs(value) > MAX_COORD_ABS_NM for value in stripe):
            reasons.append("stripe_out_of_coordinate_bounds")
        if not _bbox_is_on_declared_internal_edge(stripe, scene.cell_bbox_nm, demand.edge, demand.offset_nm):
            reasons.append("stripe_not_internal_edge")
        if blockers:
            reasons.append("existing_conductor_spacing")
        observations.append({
            "demand_id": demand.demand_id,
            "stripe_bbox_nm": list(stripe),
            "terminal_fact_count": terminal_count,
            "minimum_spacing_nm": spacing,
            "candidate_ids": [item.candidate_id for item in pool.candidates_for(demand.demand_id)],
            "rejection_reasons": reasons,
            "blocking_conductor_fact_indices": blockers,
        })

    payload = {
        "schema": _SCHEMA,
        "authority": "diagnostic_only",
        "scene_scope": "captured_current_scene",
        # Unknown-net facts are retained verbatim. Scene bytes alone cannot
        # prove complete source coverage; the native inventory join does that.
        "unowned_conductor_inventory": "retained_if_supplied",
        "manufacturing_grid_nm": pdk_manufacturing_grid_nm(tech),
        "scene": json.loads(scene_json),
        "candidate_pool": json.loads(pool_json),
        "result": json.loads(result.canonical_json()),
        "infeasibility_certificate": (
            None if result.certificate is None
            else json.loads(result.certificate.canonical_json())
        ),
        "demand_observations": observations,
    }
    return RailPlacementStopObservation(_canonical_json(payload))
