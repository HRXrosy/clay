"""Route sign-off as OBLIGATION CHECKERS (codex landing step 4).

The route's correctness checks are expressed as obligations, each returning a
:class:`clay.signoff_verdict.CheckerVerdict` in one of the five explicit states. They
consume the canonical production result (``route_amp_leakfree`` -> metrics + the
AsBuiltBundle) so every checker reads the SAME as-built layout. Three families:

* **capability** -- for each device kind the design uses, is the layout primitive
  supported on this PDK? (the gf180 poly resistor is UNSUPPORTED -> a named, blocking
  gap, replacing the silent net-collapse.)
* **native** -- the router's fast connectivity model (opens / shorts / cap plates) read
  off the metrics. Fast, guides routing, does NOT sign off on its own.
* **oracle** -- the independent KLayout checks (LVS match, P6 reconcile). When not run,
  the obligation is PENDING (not absent, not a silent pass); when KLayout is unavailable,
  UNMEASURABLE; otherwise PASS/FAIL.

Nothing is silently skipped: an un-run or unsupported REQUIRED obligation blocks
(:func:`clay.signoff_verdict.is_blocking`).
"""
from __future__ import annotations

import math
import os
import pathlib
from typing import Any, Dict

from clay.core_contract import VerificationStatus
from clay.signoff_verdict import CheckerVerdict

__all__ = [
    "route_signoff_obligations",
    "device_capability_verdicts",
    "native_metric_verdicts",
    "via_provenance_required_verdict",
    "layer_route_profile_required_verdict",
    "exact_structure_profile_required_verdict",
    "exact_route_evidence_required_verdicts",
]


# --------------------------------------------------------------------------- #
# Wave F / G7 -- the geometric-fidelity GATE flag (env-driven, OFF by default).
# --------------------------------------------------------------------------- #
_FIDELITY_GATE_TRUE = frozenset({"1", "true", "on", "yes"})


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def via_provenance_required_verdict(report: Any) -> CheckerVerdict:
    """Fail-closed bridge for the independent exact-GDS via checker.

    This obligation is deliberately separate from ``oracle.lvs``: provenance is
    defense-in-depth and may never rewrite the real full-extraction LVS result.
    At a required gate, however, only an explicit checker ``pass`` is accepted;
    unresolved ownership, unsupported stacks, missing manifests, and checker
    failures become a named FAIL instead of a gray result being mistaken for
    electrical safety.
    """
    from clay.layout.route.via_provenance_check import (
        ViaProvenanceObservation,
        ViaProvenanceReport,
    )

    exact_type = type(report) is ViaProvenanceReport
    checker_status = str(getattr(report, "status", "unmeasurable") or "unmeasurable")
    checked = int(getattr(report, "checked", 0) or 0)
    observations = tuple(getattr(report, "observations", ()) or ())
    violations = tuple(getattr(report, "violations", ()) or ())
    unresolved = tuple(getattr(report, "unresolved", ()) or ())
    conflicts = tuple(getattr(report, "conflicted_components", ()) or ())
    reason = str(getattr(report, "reason", "") or "via provenance report unavailable")
    manifest_available = bool(getattr(report, "manifest_available", False))
    owned_vias = int(getattr(report, "router_owned_vias", 0) or 0)
    artifact_hash = str(getattr(report, "artifact_sha256", "") or "")
    manifest_hash = str(getattr(report, "route_manifest_sha256", "") or "")
    passed = bool(
        exact_type
        and checker_status == "pass"
        and manifest_available
        and checked == owned_vias == len(observations)
        and owned_vias > 0
        and all(
            type(item) is ViaProvenanceObservation and item.status == "pass"
            for item in observations
        )
        and _is_sha256(artifact_hash)
        and _is_sha256(manifest_hash)
        and not violations
        and not unresolved
        and not conflicts
    )
    return CheckerVerdict(
        checker="via_provenance",
        obligation_id="oracle.via_provenance",
        status=VerificationStatus.PASS if passed else VerificationStatus.FAIL,
        evidence={
            "checker_status": checker_status,
            "checked": checked,
            "manifest_available": manifest_available,
            "router_owned_vias": owned_vias,
            "exact_report_type": exact_type,
            "artifact_sha256": artifact_hash,
            "route_manifest_sha256": manifest_hash,
            "n_violations": len(violations),
            "n_unresolved": len(unresolved),
            "conflicted_components": conflicts,
            "checker_reason": reason,
            "electrical_oracle": "separate full-extraction KLayout LVS obligation",
        },
        blocker_reason="" if passed else (
            "required emitted-via provenance did not pass: "
            f"status={checker_status}; {reason}"
        ),
        required=True,
    )


def layer_route_profile_required_verdict(report: Any) -> CheckerVerdict:
    """Required fail-closed bridge for the exact selected-plan layer profile."""
    from clay.layout.spec.geometric_fidelity import (
        CrossDeviceRouteComponent,
        LayerRouteProfile,
        LayerRouteProfileReport,
        NegotiatedLayerGeometry,
        PlannerLayerAllocation,
        RouteProfileLineage,
    )

    exact_type = type(report) is LayerRouteProfileReport
    status = str(getattr(report, "status", "unmeasurable") or "unmeasurable")
    layers = tuple(getattr(report, "layers", ()) or ())
    raw_group_ids = getattr(report, "independent_device_group_ids", ())
    group_ids = raw_group_ids if type(raw_group_ids) is tuple else ()
    raw_cross_components = getattr(report, "cross_device_components", ())
    cross_components = (
        raw_cross_components if type(raw_cross_components) is tuple else ()
    )
    unplanned = tuple(getattr(report, "unplanned_layer_numbers", ()) or ())
    reason = str(getattr(report, "reason", "") or "layer profile unavailable")
    evidence_flags = {
        "manifest_available": bool(getattr(report, "manifest_available", False)),
        "plan_verified": bool(getattr(report, "plan_verified", False)),
        "materialization_verified": bool(
            getattr(report, "materialization_verified", False)
        ),
        "device_group_evidence_available": bool(
            getattr(report, "device_group_evidence_available", False)
        ),
        "mim_evidence_available": bool(
            getattr(report, "mim_evidence_available", False)
        ),
    }
    lineage = str(
        getattr(
            report,
            "evidence_lineage",
            RouteProfileLineage.EXACT_RESOURCE_PLAN.value,
        )
    )
    planner_applicable = bool(
        getattr(report, "planner_evidence_applicable", True)
    )
    ownership_verified = bool(
        getattr(report, "ownership_manifest_verified", False)
    )
    exact_geometry_verified = bool(
        getattr(report, "exact_geometry_verified", False)
    )
    covered_count = int(
        getattr(report, "manifest_covered_entry_count", 0) or 0
    )
    manifest_count = int(getattr(report, "manifest_entry_count", 0) or 0)
    expected_count = int(
        getattr(report, "expected_materialized_shape_count", 0) or 0
    )
    pool_hash = str(getattr(report, "candidate_pool_hash", "") or "")
    plan_hash = str(getattr(report, "resource_plan_hash", "") or "")
    artifact_hash = str(getattr(report, "artifact_sha256", "") or "")
    manifest_hash = str(getattr(report, "route_manifest_sha256", "") or "")
    exact_planned_layers = bool(layers) and all(
        type(layer) is LayerRouteProfile
        and type(layer.planner_allocation) is PlannerLayerAllocation
        and layer.negotiated_geometry is None
        and layer.planner_allocation.candidate_pool_hash == pool_hash
        and layer.planner_allocation.materialized_shape_count > 0
        for layer in layers
    )
    def exact_negotiated_layer(layer: Any) -> bool:
        if type(layer) is not LayerRouteProfile:
            return False
        try:
            geometry = layer.negotiated_geometry
            return bool(
                layer.planner_allocation is None
                and type(geometry) is NegotiatedLayerGeometry
                and type(layer.layer_n) is int
                and layer.layer_n > 0
                and type(layer.layer_key) is tuple
                and len(layer.layer_key) == 2
                and all(type(value) is int for value in layer.layer_key)
                and layer.preferred_axis in ("H", "V")
                and type(layer.route_owned_area_um2) is float
                and math.isfinite(layer.route_owned_area_um2)
                and layer.route_owned_area_um2 > 0.0
                and type(layer.over_active_area_um2) is float
                and math.isfinite(layer.over_active_area_um2)
                and layer.over_active_area_um2 >= 0.0
                and type(layer.non_mim_area_um2) is float
                and math.isfinite(layer.non_mim_area_um2)
                and layer.non_mim_area_um2 >= 0.0
                and type(layer.long_run_count) is int
                and layer.long_run_count >= 0
                and (
                    layer.over_active_fraction is None
                    or (
                        type(layer.over_active_fraction) is float
                        and math.isfinite(layer.over_active_fraction)
                        and 0.0 <= layer.over_active_fraction <= 1.0
                    )
                )
                and (
                    layer.preferred_axis_purity is None
                    or (
                        type(layer.preferred_axis_purity) is float
                        and math.isfinite(layer.preferred_axis_purity)
                        and 0.0 <= layer.preferred_axis_purity <= 1.0
                    )
                )
                and type(geometry.layer_n) is int
                and geometry.layer_n == layer.layer_n
                and geometry.layer_key == layer.layer_key
                and type(geometry.shape_count) is int
                and geometry.shape_count > 0
                and type(geometry.covered_shape_count) is int
                and geometry.covered_shape_count == geometry.shape_count
                and type(geometry.area_um2) is float
                and math.isfinite(geometry.area_um2)
                and geometry.area_um2 == layer.route_owned_area_um2
                and type(geometry.nets) is tuple
                and bool(geometry.nets)
                and all(type(net) is str and net for net in geometry.nets)
                and geometry.nets == tuple(sorted(set(geometry.nets)))
                and geometry.route_manifest_sha256 == manifest_hash
            )
        except (AttributeError, TypeError, ValueError):
            return False

    exact_negotiated_layers = bool(layers) and all(
        exact_negotiated_layer(layer) for layer in layers
    )
    negotiated_layer_order = tuple(
        (layer.layer_n, layer.layer_key)
        for layer in layers
        if type(layer) is LayerRouteProfile
    )
    negotiated_layers_canonical = bool(
        len(negotiated_layer_order) == len(layers)
        and negotiated_layer_order
        == tuple(sorted(set(negotiated_layer_order)))
    )
    planned_lineage_ok = bool(
        lineage == RouteProfileLineage.EXACT_RESOURCE_PLAN.value
        and planner_applicable
        and exact_planned_layers
        and all(evidence_flags.values())
        and manifest_count == expected_count
        and manifest_count > 0
        and _is_sha256(pool_hash)
        and _is_sha256(plan_hash)
    )
    negotiated_lineage_ok = bool(
        lineage == RouteProfileLineage.NEGOTIATED_GEOMETRY.value
        and not planner_applicable
        and exact_negotiated_layers
        and negotiated_layers_canonical
        and evidence_flags["manifest_available"]
        and evidence_flags["device_group_evidence_available"]
        and evidence_flags["mim_evidence_available"]
        and not evidence_flags["plan_verified"]
        and not evidence_flags["materialization_verified"]
        and ownership_verified
        and exact_geometry_verified
        and manifest_count == covered_count
        and manifest_count > 0
        and expected_count == 0
        and pool_hash == ""
        and plan_hash == ""
    )
    exact_layers = exact_planned_layers or exact_negotiated_layers
    exact_group_ids = bool(
        type(raw_group_ids) is tuple
        and all(type(group_id) is str and group_id for group_id in group_ids)
        and group_ids == tuple(sorted(set(group_ids)))
    )
    cross_device_applicable = exact_group_ids and len(group_ids) >= 2

    def exact_cross_component(component: Any) -> bool:
        if type(component) is not CrossDeviceRouteComponent:
            return False
        try:
            component_groups = component.device_group_ids
            nets = component.nets
            layer_keys = component.layer_keys
            area = component.route_owned_area_um2
            return bool(
                type(component.component_id) is str
                and component.component_id
                and type(component_groups) is tuple
                and len(component_groups) >= 2
                and all(type(group_id) is str and group_id for group_id in component_groups)
                and component_groups == tuple(sorted(set(component_groups)))
                and set(component_groups).issubset(group_ids)
                and type(nets) is tuple
                and bool(nets)
                and all(type(net) is str and net for net in nets)
                and nets == tuple(sorted(set(nets)))
                and type(layer_keys) is tuple
                and bool(layer_keys)
                and all(
                    type(key) is tuple
                    and len(key) == 2
                    and all(type(value) is int for value in key)
                    for key in layer_keys
                )
                and layer_keys == tuple(sorted(set(layer_keys)))
                and type(area) is float
                and math.isfinite(area)
                and area > 0.0
            )
        except (AttributeError, TypeError, ValueError):
            return False

    exact_cross_components = bool(
        type(raw_cross_components) is tuple
        and all(exact_cross_component(component) for component in cross_components)
    )
    cross_device_requirement_satisfied = bool(
        exact_group_ids
        and exact_cross_components
        and (
            bool(cross_components)
            if cross_device_applicable
            else not cross_components
        )
    )
    passed = bool(
        exact_type
        and status == "pass"
        and (planned_lineage_ok or negotiated_lineage_ok)
        and not unplanned
        and _is_sha256(artifact_hash)
        and _is_sha256(manifest_hash)
        and cross_device_requirement_satisfied
    )
    return CheckerVerdict(
        checker="layer_route_profile",
        obligation_id="fidelity.layer_route_profile",
        status=VerificationStatus.PASS if passed else VerificationStatus.FAIL,
        evidence={
            "checker_status": status,
            "n_layers": len(layers),
            "unplanned_layer_numbers": unplanned,
            "manifest_entry_count": manifest_count,
            "expected_materialized_shape_count": expected_count,
            "manifest_covered_entry_count": covered_count,
            "candidate_pool_hash": pool_hash,
            "resource_plan_hash": plan_hash,
            "artifact_sha256": artifact_hash,
            "route_manifest_sha256": manifest_hash,
            "exact_report_type": exact_type,
            "exact_layer_records": exact_layers,
            "exact_planned_layer_records": exact_planned_layers,
            "exact_negotiated_layer_records": exact_negotiated_layers,
            "negotiated_layers_canonical": negotiated_layers_canonical,
            "evidence_lineage": lineage,
            "planner_evidence_applicable": planner_applicable,
            "ownership_manifest_verified": ownership_verified,
            "exact_geometry_verified": exact_geometry_verified,
            "electrical_authority": False,
            "evidence_role": "RG6_CONSTRUCTION",
            "electrical_oracles": ("oracle.lvs", "oracle.drc"),
            "independent_device_group_ids": group_ids,
            "independent_device_group_count": len(group_ids),
            "cross_device_applicable": cross_device_applicable,
            "n_cross_device_components": len(cross_components),
            "exact_cross_device_records": exact_cross_components,
            "cross_device_requirement_satisfied": (
                cross_device_requirement_satisfied
            ),
            **evidence_flags,
            "checker_reason": reason,
        },
        blocker_reason="" if passed else (
            "required cross-device route component evidence did not pass for "
            f"{len(group_ids)} independent device groups"
            if not cross_device_requirement_satisfied
            else f"required exact selected-plan layer profile did not pass: "
            f"status={status}; {reason}"
        ),
        required=True,
    )


def exact_route_evidence_required_verdicts(
    *,
    via_report: Any,
    layer_profile_report: Any,
) -> Dict[str, CheckerVerdict]:
    """Canonical supplied-report seam; no checker absence can become a pass."""
    via = via_provenance_required_verdict(via_report)
    profile = layer_route_profile_required_verdict(layer_profile_report)
    if via.status == VerificationStatus.PASS and profile.status == VerificationStatus.PASS:
        via_artifact = str(via.evidence["artifact_sha256"])
        profile_artifact = str(profile.evidence["artifact_sha256"])
        via_manifest = str(via.evidence["route_manifest_sha256"])
        profile_manifest = str(profile.evidence["route_manifest_sha256"])
        if via_artifact != profile_artifact or via_manifest != profile_manifest:
            reason = (
                "required exact-route reports do not bind the same emitted "
                "artifact and route manifest"
            )
            via = CheckerVerdict(
                checker=via.checker,
                obligation_id=via.obligation_id,
                status=VerificationStatus.FAIL,
                evidence={**via.evidence, "peer_binding_match": False},
                blocker_reason=reason,
                required=True,
            )
            profile = CheckerVerdict(
                checker=profile.checker,
                obligation_id=profile.obligation_id,
                status=VerificationStatus.FAIL,
                evidence={**profile.evidence, "peer_binding_match": False},
                blocker_reason=reason,
                required=True,
            )
    return {via.obligation_id: via, profile.obligation_id: profile}


def exact_structure_profile_required_verdict(report: Any) -> CheckerVerdict:
    """Required artifact-first structure gate for an exact evidence bundle."""
    from clay.layout.spec.exact_structure_profile import (
        DummyEndcapObservation,
        ExactStructureProfileReport,
    )

    exact_type = type(report) is ExactStructureProfileReport
    status = str(getattr(report, "status", "unmeasurable") or "unmeasurable")
    reason = str(
        getattr(report, "reason", "")
        or ("no exact structure profile report" if report is None else "malformed report")
    )
    core_bbox = getattr(report, "core_bbox_nm", None)
    core_area = getattr(report, "core_area_um2", None)
    largest_void = getattr(report, "largest_internal_void_um2", None)
    shared_od = getattr(report, "shared_od_fraction", None)
    expected_dummy = int(getattr(report, "expected_dummy_endcaps", 0) or 0)
    realized_dummy = int(getattr(report, "realized_dummy_endcaps", 0) or 0)
    raw_observations = getattr(report, "dummy_endcap_observations", ())
    observations = raw_observations if type(raw_observations) is tuple else ()
    artifact_hash = str(getattr(report, "artifact_sha256", "") or "")
    manifest_hash = str(getattr(report, "route_manifest_sha256", "") or "")
    program_hash = str(getattr(report, "row_program_sha256", "") or "")
    final_hash = str(
        getattr(report, "row_final_certificate_sha256", "") or ""
    )
    expectation_hash = str(getattr(report, "expectation_sha256", "") or "")
    exact_metrics = bool(
        type(core_bbox) is tuple
        and len(core_bbox) == 4
        and all(type(value) is int for value in core_bbox)
        and core_bbox[2] > core_bbox[0]
        and core_bbox[3] > core_bbox[1]
        and type(core_area) is float
        and math.isfinite(core_area)
        and core_area > 0.0
        and type(largest_void) is float
        and math.isfinite(largest_void)
        and 0.0 <= largest_void <= core_area
        and type(shared_od) is float
        and math.isfinite(shared_od)
        and 0.0 <= shared_od <= 1.0
    )
    exact_dummy = bool(
        type(raw_observations) is tuple
        and len(observations) == expected_dummy
        and realized_dummy == expected_dummy
        and all(
            type(item) is DummyEndcapObservation and item.status == "pass"
            for item in observations
        )
    )
    valid_bindings = all(
        _is_sha256(value)
        for value in (
            artifact_hash, manifest_hash, program_hash, final_hash, expectation_hash
        )
    )
    passed = bool(
        exact_type
        and status == "pass"
        and exact_metrics
        and exact_dummy
        and valid_bindings
    )
    if passed:
        verdict_status = VerificationStatus.PASS
        blocker_reason = ""
    elif exact_type and status == "fail":
        verdict_status = VerificationStatus.FAIL
        blocker_reason = f"exact structure profile failed: {reason}"
    else:
        verdict_status = VerificationStatus.UNMEASURABLE
        blocker_reason = f"exact structure profile unmeasurable: {reason}"
    return CheckerVerdict(
        checker="exact_structure_profile",
        obligation_id="fidelity.exact_structure_profile",
        status=verdict_status,
        evidence={
            "checker_status": status,
            "exact_report_type": exact_type,
            "core_bbox_nm": core_bbox,
            "core_area_um2": core_area,
            "largest_internal_void_um2": largest_void,
            "shared_od_fraction": shared_od,
            "expected_dummy_endcaps": expected_dummy,
            "realized_dummy_endcaps": realized_dummy,
            "artifact_sha256": artifact_hash,
            "route_manifest_sha256": manifest_hash,
            "row_program_sha256": program_hash,
            "row_final_certificate_sha256": final_hash,
            "expectation_sha256": expectation_hash,
            "checker_reason": reason,
        },
        blocker_reason=blocker_reason,
        required=True,
    )


def _fidelity_gate_enabled() -> bool:
    """Is the Wave F / G7 geometric-fidelity gate engaged (env ``CLAY_FIDELITY_GATE``)?

    OFF by default so the green production signoff spine stays byte-identical (no
    fidelity obligation is added, no coverage class is required). ON turns
    ``include_fidelity`` + ``enforce_coverage`` on for ANY route signoff -- the
    route_amp path AND the general ``route_cell`` spine -- so a portrait/whitespace
    layout that is DRC/LVS-legal is BLOCKED, not silently green (invariant I9). Read
    FRESH each call (an optimizer / test may toggle it), never cached."""
    return str(os.environ.get("CLAY_FIDELITY_GATE", "")).strip().lower() in _FIDELITY_GATE_TRUE

# device-kind (intent passive ``kind``) -> layout capability probed by the CapabilityMatrix.
_KIND_TO_CAPABILITY = {
    "res": "poly_res",
    "cap": "mim_cap",
    "c": "mim_cap",
}


def device_capability_verdicts(intent: Any, tech_id: str) -> Dict[str, CheckerVerdict]:
    """One capability verdict per DISTINCT layout primitive the design uses. A design that
    contains an unsupported primitive (e.g. gf180 poly resistor) gets a NAMED blocking
    verdict, not a silent substitution."""
    from clay.layout.capability import capability_status

    kinds = set()
    for p in getattr(intent, "passives", []) or []:
        k = str(getattr(p, "kind", "")).lower()
        cap = _KIND_TO_CAPABILITY.get(k)
        if cap:
            kinds.add(cap)
    out: Dict[str, CheckerVerdict] = {}
    for cap in sorted(kinds):
        v = capability_status(cap, tech_id, required=True)
        out[v.obligation_id] = v
    return out


def _is_route_cell_result(obj: Any) -> bool:
    """True iff ``obj`` is a :class:`clay.layout.route.route_cell.RouteCellResult`
    (the GENERAL route's output bundle) -- the discriminator that routes the
    obligation engine to the route_cell adapters instead of the OTA-specialised
    ``route_amp_leakfree`` dict path. Lazy import (route_cell is heavy) so importing
    this module stays cheap and circular-import-free."""
    try:
        from clay.layout.route.route_cell import RouteCellResult
    except Exception:  # pragma: no cover - defensive (route_cell import failure)
        return False
    return type(obj) is RouteCellResult


def native_metric_verdicts(metrics: Any, *, intent: Any = None) -> Dict[str, CheckerVerdict]:
    """Native connectivity obligations read off the route metrics (fast model).

    Accepts EITHER the ``route_amp_leakfree`` metrics dict (the byte-stable OTA path,
    unchanged) OR a :class:`clay.layout.route.route_cell.RouteCellResult` (the general
    spine). For the route_cell shape the metrics are normalised from the result's
    properties and each obligation is dispositioned with 5-state honesty: the metrics
    route_cell HAS (opens / shorts / cap-plate distinctness from the union-find
    ``merged_pairs``) map to real PASS/FAIL verdicts; the cap-plate OVERLAP GEOMETRY
    route_cell does NOT compute is ADVISORY (QUALITY, gated by the real DRC oracle),
    never a spurious block."""
    if _is_route_cell_result(metrics):
        return _native_metric_verdicts_route_cell(metrics, intent)

    out: Dict[str, CheckerVerdict] = {}

    n_conn = int(metrics.get("n_connected", 0))
    n_req = int(metrics.get("n_required_nets", 0))
    opens = list(metrics.get("geometric_open_nets", []) or [])
    conn_ok = n_req > 0 and n_conn == n_req and not opens
    out["native.opens"] = CheckerVerdict(
        checker="native", obligation_id="native.opens",
        status=VerificationStatus.PASS if conn_ok else VerificationStatus.FAIL,
        evidence={"connected": n_conn, "required": n_req, "open_nets": opens},
        blocker_reason="" if conn_ok else (
            f"{n_conn}/{n_req} nets connected; geometric opens={opens}"
        ),
        required=True,
    )

    n_shorts = int(metrics.get("n_shorts", 0))
    out["native.shorts"] = CheckerVerdict(
        checker="native", obligation_id="native.shorts",
        status=VerificationStatus.PASS if n_shorts == 0 else VerificationStatus.FAIL,
        evidence={"n_shorts": n_shorts},
        blocker_reason="" if n_shorts == 0 else f"{n_shorts} same-layer cross-net shorts",
        required=True,
    )

    distinct = bool(metrics.get("caps_all_plates_distinct", False))
    cap_overlap = float(metrics.get("max_cap_cap_overlap_um2", 0.0))
    foreign = float(metrics.get("max_foreign_over_cap_plate_um2", 0.0))
    cap_ok = distinct and cap_overlap == 0.0 and foreign == 0.0
    out["native.cap_plates_distinct"] = CheckerVerdict(
        checker="native", obligation_id="native.cap_plates_distinct",
        status=VerificationStatus.PASS if cap_ok else VerificationStatus.FAIL,
        evidence={"plates_distinct": distinct, "cap_cap_overlap_um2": cap_overlap,
                  "foreign_over_plate_um2": foreign},
        blocker_reason="" if cap_ok else (
            f"cap plates: distinct={distinct} cap_overlap={cap_overlap} foreign={foreign}"
        ),
        required=True,
    )
    return out


def _cap_nets_from_intent(intent: Any) -> set:
    """The nets a MIM cap's two plates connect to, from the reference intent's
    ``passives`` (each carries ``kind`` + ``nets={"p":..,"n":..}``). Used to attribute
    a union-find ``merged_pair`` to a cap-plate short."""
    cap_nets: set = set()
    for p in getattr(intent, "passives", None) or []:
        if str(getattr(p, "kind", "")).lower() in ("cap", "c"):
            for v in (getattr(p, "nets", None) or {}).values():
                if v:
                    cap_nets.add(str(v))
    return cap_nets


def _native_metric_verdicts_route_cell(res: Any, intent: Any) -> Dict[str, CheckerVerdict]:
    """Native connectivity obligations for a ``RouteCellResult`` (the general spine).

    5-state honesty, per obligation:
      * ``native.opens`` (diagnostic advisory): the GEOMETRIC opens
        (``res.unconnected_nets`` -- emitted-metal union-find, the honest connectivity
        truth) -> PASS/FAIL.
      * ``native.shorts`` (diagnostic advisory): the independent union-find
        cross-net short oracle (``len(res.merged_pairs)``) -> PASS/FAIL.
      * ``native.cap_plates_distinct`` (CORRECTNESS): each MIM cap's two plates on
        DISTINCT nets, derived from ``merged_pairs`` restricted to the cap nets
        (``_cap_nets_from_intent``). If NO passive intent is available to attribute
        cap nets -> UNMEASURABLE (required=False, NON-blocking): honest gray, never a
        spurious False->block; ``native.shorts`` already covers general cross-net
        shorts and the independent KLayout LVS is the cap-plate short authority.
      * ``native.cap_plate_overlap_geometry`` (QUALITY): the cap-plate OVERLAP
        geometry (``max_cap_cap_overlap_um2`` / ``max_foreign_over_cap_plate_um2``)
        that only ``anchor_ota`` computes -> ADVISORY (never blocks): geometric
        quality, gated by the real DRC oracle, NOT legality.
    """
    out: Dict[str, CheckerVerdict] = {}

    n_req = int(getattr(res, "n_required", 0))
    opens = list(getattr(res, "unconnected_nets", []) or [])
    n_conn = int(getattr(res, "n_connected", 0))
    conn_ok = n_req > 0 and not opens
    out["native.opens"] = CheckerVerdict(
        checker="native", obligation_id="native.opens",
        status=(
            VerificationStatus.PASS if conn_ok else VerificationStatus.ADVISORY
        ),
        evidence={"connected": n_req - len(opens), "required": n_req,
                  "engine_n_connected": n_conn, "open_nets": opens},
        blocker_reason="" if conn_ok else (
            f"{n_req - len(opens)}/{n_req} nets connected; geometric opens={opens}"
        ),
        required=False,
    )

    merged = [tuple(p) for p in (getattr(res, "merged_pairs", None) or [])]
    n_shorts = len(merged)
    out["native.shorts"] = CheckerVerdict(
        checker="native", obligation_id="native.shorts",
        status=(
            VerificationStatus.PASS
            if n_shorts == 0
            else VerificationStatus.ADVISORY
        ),
        evidence={"n_shorts": n_shorts, "merged_pairs": merged},
        blocker_reason="" if n_shorts == 0 else (
            f"{n_shorts} cross-net union-find shorts: {merged}"
        ),
        required=False,
    )

    passives = getattr(intent, "passives", None)
    if passives is None:
        out["native.cap_plates_distinct"] = CheckerVerdict(
            checker="native", obligation_id="native.cap_plates_distinct",
            status=VerificationStatus.UNMEASURABLE,
            evidence={"reason": "no passive intent to attribute cap-plate nets"},
            blocker_reason=(
                "route_cell native path has no passive intent to attribute cap-plate "
                "nets; native.shorts covers general cross-net shorts and the independent "
                "KLayout LVS is the cap-plate short authority"
            ),
            required=False,  # non-blocking honest gray, NEVER a spurious False->block
        )
    else:
        cap_nets = _cap_nets_from_intent(intent)
        merged_cap = [(a, b) for (a, b) in merged
                      if str(a) in cap_nets or str(b) in cap_nets]
        distinct = not merged_cap
        n_caps = sum(1 for p in passives
                     if str(getattr(p, "kind", "")).lower() in ("cap", "c"))
        out["native.cap_plates_distinct"] = CheckerVerdict(
            checker="native", obligation_id="native.cap_plates_distinct",
            status=(
                VerificationStatus.PASS
                if distinct
                else VerificationStatus.ADVISORY
            ),
            evidence={"n_cap_passives": n_caps, "cap_nets": sorted(cap_nets),
                      "merged_cap_pairs": merged_cap},
            blocker_reason="" if distinct else (
                f"MIM cap plate nets collapsed to a union-find short: {merged_cap}"
            ),
            required=False,
        )

    out["native.cap_plate_overlap_geometry"] = CheckerVerdict(
        checker="native", obligation_id="native.cap_plate_overlap_geometry",
        status=VerificationStatus.ADVISORY,
        evidence={"max_cap_cap_overlap_um2": None,
                  "max_foreign_over_cap_plate_um2": None,
                  "computed_by": "anchor_ota only"},
        blocker_reason=(
            "route_cell path does not compute cap-plate overlap geometry "
            "(max_cap_cap_overlap_um2 / max_foreign_over_cap_plate_um2); the real DRC "
            "oracle gates cap-plate legality (QUALITY, not legality -> advisory)"
        ),
        required=False,
    )

    # Independent declared route goals (currently guard-ring frame ties).  These
    # are construction goals compiled at the canonical route_cell seam after the
    # final ring/rail/pin set exists.  A missing target, missing net, or incomplete
    # geometric net route stays named in ``required_unsatisfied``.  This verdict
    # remain diagnostic evidence only: same-GDS full-extraction LVS and deck DRC
    # are the electrical/geometric legality authorities.
    topology_plan = getattr(res, "topology_plan", None)
    required_goals = tuple(getattr(topology_plan, "required_goals", ()) or ())
    if required_goals:
        residuals = tuple(getattr(res, "required_unsatisfied", ()) or ())
        unsatisfied = tuple(goal for goal in required_goals if goal in residuals)

        def goal_evidence(goal: Any) -> Dict[str, Any]:
            target = getattr(goal, "target", None)
            return {
                "kind": str(getattr(goal, "kind", "")),
                "net": str(getattr(goal, "net", "")),
                "declaration_ref": str(getattr(goal, "declaration_ref", "")),
                "target": None if target is None else {
                    "xy_nm": list(target.xy_nm),
                    "layer": list(target.layer),
                    "provenance": target.provenance,
                },
            }

        clean = not unsatisfied
        unsatisfied_evidence = [goal_evidence(goal) for goal in unsatisfied]
        out["native.required_route_goals"] = CheckerVerdict(
            checker="native",
            obligation_id="native.required_route_goals",
            status=(
                VerificationStatus.PASS if clean else VerificationStatus.ADVISORY
            ),
            evidence={
                "n_required": len(required_goals),
                "n_unsatisfied": len(unsatisfied),
                "required": [goal_evidence(goal) for goal in required_goals],
                "unsatisfied": unsatisfied_evidence,
            },
            blocker_reason="" if clean else (
                "required route goals unsatisfied: "
                f"{unsatisfied_evidence}"
            ),
            required=False,
        )

    # native.guard_ring_materialized (native, ADVISORY): only when the cell DECLARES
    # `@guard_ring` intents (omitted otherwise so a cell that declares no ring keeps
    # a byte-stable obligation set). Is each declared cluster ring PRESENT + CLOSED +
    # TIED to the declared net + drawn on the declared polarity, in the EMITTED
    # artifact? Five-state honest (skipped/unsupported -> UNMEASURABLE, never a
    # silent pass). Advisory (required=False): the independent KLayout LVS / real
    # DRC stay the signoff authority; on a 3-terminal deck this is the ONLY
    # bulk-ring oracle.
    declared_rings = tuple(getattr(res, "guard_rings", ()) or ())
    if declared_rings:
        rep = getattr(res, "guard_ring_report", None)
        results = list(getattr(rep, "results", ()) or ()) if rep is not None else []
        ev_results = [
            {"members": list(r.members), "well_declared": r.well_declared,
             "tie_declared": r.tie_declared, "status": r.status,
             "drawn_well": r.drawn_well, "attached_nets": list(r.attached_nets),
             "reason": r.reason}
            for r in results
        ]
        n_pass = sum(1 for r in results if r.status == "pass")
        n_viol = sum(1 for r in results
                     if r.status not in ("pass", "skipped", "unsupported"))
        n_gray = sum(1 for r in results if r.status in ("skipped", "unsupported"))
        if n_viol:
            status = VerificationStatus.FAIL
            reason = (f"{n_viol}/{len(results)} declared guard rings NOT materialized "
                      f"(present+closed+tied+polarity): {ev_results}")
        elif n_pass and n_pass + n_gray == len(results):
            status = VerificationStatus.PASS
            reason = ""
        else:
            # every declared ring is skipped/unsupported (no placement / PDK draws
            # no nwell) -> honest gray, never a silent pass.
            status = VerificationStatus.UNMEASURABLE
            reason = ("no declared guard ring could be adjudicated "
                      f"(all skipped/unsupported): {ev_results}")
        out["native.guard_ring_materialized"] = CheckerVerdict(
            checker="native", obligation_id="native.guard_ring_materialized",
            status=status,
            evidence={"n_declared": len(declared_rings), "n_pass": n_pass,
                      "n_violation": n_viol, "n_gray": n_gray,
                      "results": ev_results},
            blocker_reason=reason,
            required=False,  # native advisory; KLayout LVS/DRC stay signoff authority
        )
    return out


def _route_cell_drc_verdict(res: Any, cell_def: Any, tech_id: str, *,
                            klayout_bin: Any = None, timeout_sec: float = 580.0,
                            work_dir: Any = None) -> Any:
    """Real-deck DRC on the route_cell as-built emit -- the geometric-LEGALITY half of
    sign-off for the general spine.

    Mirrors :func:`clay.layout.route.drc_gate.drc_gate` (same deck, same
    :class:`~clay.layout.route.drc_gate.DrcVerdict`, same honest-gray gaps) but emits
    the route_cell shape directly (``emit_labelled_gds(res.routed, ..., devices=
    well_merge_devices_from_cell_def(cell_def))``) rather than through the OTA
    ``_materialize`` -- which cannot consume a RouteCellResult (it would silently
    re-derive the route_amp OTA). Never raises on an infra gap: no deck / KLayout
    absent / emit failure -> ``ran=False`` with a reason (UNMEASURABLE upstream)."""
    import tempfile

    import klayout.db as kdb

    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.klayout_deck_gen import DEFAULT_KLAYOUT_BIN
    from clay.layout.route.drc_gate import DrcVerdict
    from clay.layout.route.lvs_gate import emit_labelled_gds, well_merge_devices_from_cell_def
    from clay.layout.tech import LayoutTech
    from tools.klayout_drc_signoff_v139.run import Gf180DrcError, run_gf180_drc

    try:
        deck = LayoutTech.resolve(tech_id).klayout_decks().get("drc")
    except Exception as exc:  # pragma: no cover - unknown tech
        return DrcVerdict(ran=False, clean=False, total=0,
                          reason=f"no LayoutTech for tech_id={tech_id!r}: {exc}")
    if deck is None or not pathlib.Path(deck).exists():
        return DrcVerdict(ran=False, clean=False, total=0,
                          reason=f"no DRC deck registered for tech_id={tech_id!r}")

    kbin = pathlib.Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
    if not kbin.exists():
        return DrcVerdict(ran=False, clean=False, total=0, deck_path=str(deck),
                          reason=f"KLayout binary not installed at {kbin}")

    if work_dir is None:
        work_dir = pathlib.Path(tempfile.mkdtemp(prefix="drc_gate_rc_"))
    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cell_name = str(getattr(cell_def, "local_name", None)
                    or getattr(cell_def, "name", None) or "cell")
    top = f"{cell_name}_routed"
    try:
        layout_intent = layout_intent_from_cell_def(cell_def)
        gds = work_dir / f"drc_gate_rc_{tech_id}.gds"
        emit_labelled_gds(res.routed, layout_intent, res.geometry, res.placement_um,
                          res.pins_by_net, gds, tech=res.tech, top_cell=top,
                          devices=well_merge_devices_from_cell_def(cell_def),
                          source_cell=cell_def,
                          row_program=getattr(res, "row_program", None),
                          row_materialization=getattr(res, "row_materialization", None),
                          row_final_certificate=getattr(res, "row_final_placement_certificate", None),
                          selected_route_engine=getattr(res,"selected_route_engine",None))
        ly = kdb.Layout()
        ly.read(str(gds))
        tops = ly.top_cells()
        if not tops:
            return DrcVerdict(ran=False, clean=False, total=0, deck_path=str(deck),
                              gds_path=str(gds), reason="emitted GDS has no top cell")
        topcell = tops[0].name
    except Exception as exc:
        return DrcVerdict(ran=False, clean=False, total=0, deck_path=str(deck),
                          reason=f"could not emit canonical GDS: {exc}")

    try:
        rep = run_gf180_drc(
            str(gds), topcell=topcell, deck=pathlib.Path(deck), klayout_bin=kbin,
            report_path=work_dir / f"drc_gate_rc_{tech_id}.lyrdb", timeout_sec=timeout_sec,
        )
    except Gf180DrcError as exc:
        return DrcVerdict(ran=False, clean=False, total=0, deck_path=str(deck),
                          gds_path=str(gds), topcell=topcell,
                          reason=f"DRC deck could not run: {exc}")

    counts = {r: n for r, n in rep.counts_by_rule().items() if n}
    total = int(rep.total_violations)
    return DrcVerdict(
        ran=bool(rep.ran_real_deck),
        clean=bool(rep.ran_real_deck and rep.is_clean()),
        total=total, counts_by_rule=counts, deck_path=str(deck), gds_path=str(gds),
        topcell=topcell, reason="" if rep.ran_real_deck else "deck did not execute",
    )


def _route_cell_reconcile_advisory(res: Any, *, verdict: Any = None,
                                   drc: Any = None) -> CheckerVerdict:
    """The route_cell native/oracle reconcile as an ADVISORY (never blocks).

    ``route_success`` is a route_amp concept; route_cell's ``net_complete`` +
    union-find ``merged_pairs`` are independent native oracles, and their consistency
    with the independent KLayout oracle is guarded separately (the via-emit
    dischargeable contract). Sign-off follows the independent KLayout LVS + real-deck
    DRC, not the native model -- so the reconcile here is informational, not a gate."""
    ev: Dict[str, Any] = {
        "net_complete": bool(getattr(res, "net_complete", False)),
        "n_merged_pairs": len(getattr(res, "merged_pairs", None) or []),
        "unconnected_nets": list(getattr(res, "unconnected_nets", []) or []),
    }
    if verdict is not None:
        ev["klayout_matched"] = bool(getattr(verdict, "matched", False))
    if drc is not None:
        ev["klayout_drc_clean"] = bool(getattr(drc, "clean", False))
    return CheckerVerdict(
        checker="oracle", obligation_id="oracle.reconcile",
        status=VerificationStatus.ADVISORY,
        evidence=ev,
        blocker_reason=(
            "route_cell native/oracle reconcile is ADVISORY (non-blocking): "
            "route_success is a route_amp concept; route_cell's net_complete + "
            "union-find merged_pairs are independent native oracles whose consistency "
            "with the independent KLayout oracle is guarded separately. Sign-off follows "
            "the independent KLayout LVS + real-deck DRC, not native."
        ),
        required=True,
    )


def _route_cell_oracle_verdicts(res: Any, cell_def: Any, tech_id: str, *,
                                timeout_sec: float = 580.0) -> Dict[str, CheckerVerdict]:
    """The INDEPENDENT KLayout oracle family for a route_cell result: LVS via the
    general ``lvs_gate_route_cell`` + real-deck DRC on the route_cell emit + an
    ADVISORY native/oracle reconcile. KLayout absent -> LVS/DRC UNMEASURABLE (required
    => blocks; the token is withheld), never a vacuous pass."""
    from clay.layout.klayout_deck_gen import DEFAULT_KLAYOUT_BIN

    out: Dict[str, CheckerVerdict] = {}
    if not DEFAULT_KLAYOUT_BIN.exists():
        for oid, what in (("oracle.lvs", "KLayout LVS"),
                          ("oracle.drc", "real foundry DRC")):
            out[oid] = CheckerVerdict(
                checker="oracle", obligation_id=oid,
                status=VerificationStatus.UNMEASURABLE,
                evidence={"klayout": str(DEFAULT_KLAYOUT_BIN)},
                blocker_reason=f"{what} unmeasurable: KLayout binary not installed",
                required=True,
            )
        out["oracle.reconcile"] = _route_cell_reconcile_advisory(res)
        return out

    from clay.layout.route.lvs_gate import lvs_gate_route_cell

    verdict = lvs_gate_route_cell(res, cell_def, tech_id=tech_id, timeout_sec=timeout_sec)
    out["oracle.lvs"] = CheckerVerdict(
        checker="oracle", obligation_id="oracle.lvs",
        status=VerificationStatus.PASS if verdict.matched else VerificationStatus.FAIL,
        evidence={"matched": verdict.matched,
                  "unmatched_devices": len(verdict.unmatched["devices"]),
                  "unmatched_nets": len(verdict.unmatched["nets"])},
        blocker_reason="" if verdict.matched else (
            f"KLayout LVS not matched: {len(verdict.unmatched['devices'])} devices, "
            f"{len(verdict.unmatched['nets'])} nets unmatched"
        ),
        required=True,
    )

    drc = _route_cell_drc_verdict(res, cell_def, tech_id, timeout_sec=timeout_sec)
    if not drc.ran:
        drc_status = VerificationStatus.UNMEASURABLE
        drc_reason = f"DRC unmeasurable: {drc.reason}"
    elif drc.clean:
        drc_status = VerificationStatus.PASS
        drc_reason = ""
    else:
        drc_status = VerificationStatus.FAIL
        drc_reason = (f"real foundry DRC not clean: {drc.total} violations "
                      f"{drc.counts_by_rule}")
    out["oracle.drc"] = CheckerVerdict(
        checker="oracle", obligation_id="oracle.drc", status=drc_status,
        evidence={"ran": drc.ran, "clean": drc.clean, "total": drc.total,
                  "counts_by_rule": drc.counts_by_rule, "topcell": drc.topcell},
        blocker_reason=drc_reason, required=True,
    )

    out["oracle.reconcile"] = _route_cell_reconcile_advisory(res, verdict=verdict, drc=drc)
    return out


def _precomputed_exact_oracle_verdicts(
    result: Any, bundle: Any
) -> Dict[str, CheckerVerdict]:
    """Bridge typed oracle results already bound to the bundle's exact GDS.

    No emitter, deck, or checker is invoked here.  Missing optional LVS/DRC
    results remain named UNMEASURABLE blockers; they never trigger a fallback
    oracle run against a second artifact.
    """
    from clay.layout.route.exact_artifact_evidence import ExactArtifactEvidenceBundle

    if type(bundle) is not ExactArtifactEvidenceBundle:
        raise TypeError("exact_artifact_evidence must be ExactArtifactEvidenceBundle")
    bundle.validate()
    out: Dict[str, CheckerVerdict] = {}
    lvs = bundle.lvs_verdict
    if lvs is None:
        out["oracle.lvs"] = CheckerVerdict(
            checker="oracle",
            obligation_id="oracle.lvs",
            status=VerificationStatus.UNMEASURABLE,
            evidence={"precomputed_exact_artifact": True, "verdict_available": False},
            blocker_reason="exact artifact bundle contains no precomputed LvsVerdict",
            required=True,
        )
    else:
        unmatched_nets = tuple((lvs.unmatched or {}).get("nets", ()) or ())
        unmatched_devices = tuple((lvs.unmatched or {}).get("devices", ()) or ())
        matched = bool(
            lvs.ran
            and lvs.matched
            and not unmatched_nets
            and not unmatched_devices
        )
        out["oracle.lvs"] = CheckerVerdict(
            checker="oracle",
            obligation_id="oracle.lvs",
            status=VerificationStatus.PASS if matched else VerificationStatus.FAIL,
            evidence={
                "precomputed_exact_artifact": True,
                "ran": bool(lvs.ran),
                "matched": bool(lvs.matched),
                "unmatched_devices": len(unmatched_devices),
                "unmatched_nets": len(unmatched_nets),
                "artifact_sha256": bundle.artifact.truth_manifest.exact_gds_sha256,
            },
            blocker_reason="" if matched else "precomputed exact-artifact KLayout LVS did not match",
            required=True,
        )

    drc = bundle.drc_verdict
    if drc is None:
        out["oracle.drc"] = CheckerVerdict(
            checker="oracle",
            obligation_id="oracle.drc",
            status=VerificationStatus.UNMEASURABLE,
            evidence={"precomputed_exact_artifact": True, "verdict_available": False},
            blocker_reason="exact artifact bundle contains no precomputed DrcVerdict",
            required=True,
        )
    else:
        counts_by_rule = dict(drc.counts_by_rule or {})
        total = int(drc.total)
        clean = bool(
            drc.ran
            and drc.clean
            and total == 0
            and not any(int(count) != 0 for count in counts_by_rule.values())
        )
        if not drc.ran:
            status = VerificationStatus.UNMEASURABLE
            reason = f"precomputed exact-artifact DRC unmeasurable: {drc.reason}"
        elif clean:
            status = VerificationStatus.PASS
            reason = ""
        else:
            status = VerificationStatus.FAIL
            reason = f"precomputed exact-artifact DRC has {total} violations"
        out["oracle.drc"] = CheckerVerdict(
            checker="oracle",
            obligation_id="oracle.drc",
            status=status,
            evidence={
                "precomputed_exact_artifact": True,
                "ran": bool(drc.ran),
                "clean": clean,
                "reported_clean": bool(drc.clean),
                "total": total,
                "counts_by_rule": counts_by_rule,
                "topcell": drc.topcell,
                "artifact_sha256": bundle.artifact.truth_manifest.exact_gds_sha256,
            },
            blocker_reason=reason,
            required=True,
        )

    if _is_route_cell_result(result):
        out["oracle.reconcile"] = _route_cell_reconcile_advisory(
            result, verdict=lvs, drc=drc
        )
    else:
        metrics = result if isinstance(result, dict) else {}
        n_required = int(metrics.get("n_required_nets", 0) or 0)
        native_clean = bool(
            n_required > 0
            and int(metrics.get("n_connected", 0) or 0) == n_required
            and not (metrics.get("geometric_open_nets", []) or [])
            and int(metrics.get("n_shorts", 0) or 0) == 0
            and bool(metrics.get("caps_all_plates_distinct", False))
        )
        oracle_clean = bool(
            lvs is not None
            and lvs.ran
            and lvs.matched
            and drc is not None
            and drc.ran
            and drc.clean
        )
        if native_clean == oracle_clean:
            reconcile_status = VerificationStatus.PASS
            reconcile_reason = ""
        elif native_clean:
            reconcile_status = VerificationStatus.FAIL
            reconcile_reason = "native blind spot against precomputed exact-artifact oracle"
        else:
            reconcile_status = VerificationStatus.ADVISORY
            reconcile_reason = "native model is stricter than the precomputed exact-artifact oracle"
        out["oracle.reconcile"] = CheckerVerdict(
            checker="oracle",
            obligation_id="oracle.reconcile",
            status=reconcile_status,
            evidence={
                "precomputed_exact_artifact": True,
                "native_clean": native_clean,
                "klayout_signoff_clean": oracle_clean,
                "agree": native_clean == oracle_clean,
            },
            blocker_reason=reconcile_reason,
            required=True,
        )
    return out


def _oracle_verdicts(result: Any, intent: Any, tech_id: str, *, run: bool,
                     cell_def: Any = None) -> Dict[str, CheckerVerdict]:
    if not run:
        # explicitly NOT run -> PENDING (required => blocks). Never absent, never a pass.
        pend = lambda oid, what: CheckerVerdict(  # noqa: E731
            checker="oracle", obligation_id=oid, status=VerificationStatus.PENDING,
            evidence={"run_oracle": False},
            blocker_reason=f"{what} not run (run_oracle=False)", required=True,
        )
        return {"oracle.lvs": pend("oracle.lvs", "KLayout LVS"),
                "oracle.reconcile": pend("oracle.reconcile", "P6 reconcile")}

    # ROUTE_CELL oracle path (the general spine): route the oracle through the already
    # green ``lvs_gate_route_cell`` + a real-deck DRC on the route_cell emit, NOT the
    # OTA-hardwired ``lvs_gate`` (whose ``_materialize`` cannot consume a
    # RouteCellResult). Guarded on cell_def so the route_amp dict path (cell_def=None)
    # is byte-stable -- it never enters this branch.
    if cell_def is not None and _is_route_cell_result(result):
        return _route_cell_oracle_verdicts(result, cell_def, tech_id)

    from clay.layout.klayout_deck_gen import DEFAULT_KLAYOUT_BIN

    if not DEFAULT_KLAYOUT_BIN.exists():
        un = lambda oid, what: CheckerVerdict(  # noqa: E731
            checker="oracle", obligation_id=oid, status=VerificationStatus.UNMEASURABLE,
            evidence={"klayout": str(DEFAULT_KLAYOUT_BIN)},
            blocker_reason=f"{what} unmeasurable: KLayout binary not installed", required=True,
        )
        return {"oracle.lvs": un("oracle.lvs", "KLayout LVS"),
                "oracle.reconcile": un("oracle.reconcile", "P6 reconcile")}

    from clay.layout.route.drc_gate import drc_gate
    from clay.layout.route.lvs_gate import lvs_gate
    from clay.layout.route.reconcile import reconcile

    out: Dict[str, CheckerVerdict] = {}
    verdict = lvs_gate(result, intent, tech_id=tech_id)
    out["oracle.lvs"] = CheckerVerdict(
        checker="oracle", obligation_id="oracle.lvs",
        status=VerificationStatus.PASS if verdict.matched else VerificationStatus.FAIL,
        evidence={"matched": verdict.matched,
                  "unmatched_devices": len(verdict.unmatched["devices"]),
                  "unmatched_nets": len(verdict.unmatched["nets"])},
        blocker_reason="" if verdict.matched else (
            f"KLayout LVS not matched: {len(verdict.unmatched['devices'])} devices, "
            f"{len(verdict.unmatched['nets'])} nets unmatched"
        ),
        required=True,
    )

    # DRC: the geometric-LEGALITY half of sign-off (LVS proves connectivity, not
    # legality). The SOLE M*.2a authority is the real foundry deck, run on the SAME
    # canonical as-built layout. Computed ONCE and reused by reconcile below.
    drc = drc_gate(result, intent, tech_id=tech_id)
    if not drc.ran:
        drc_status = VerificationStatus.UNMEASURABLE
        drc_reason = f"DRC unmeasurable: {drc.reason}"
    elif drc.clean:
        drc_status = VerificationStatus.PASS
        drc_reason = ""
    else:
        drc_status = VerificationStatus.FAIL
        drc_reason = (f"real foundry DRC not clean: {drc.total} violations "
                      f"{drc.counts_by_rule}")
    out["oracle.drc"] = CheckerVerdict(
        checker="oracle", obligation_id="oracle.drc",
        status=drc_status,
        evidence={"ran": drc.ran, "clean": drc.clean, "total": drc.total,
                  "counts_by_rule": drc.counts_by_rule, "topcell": drc.topcell},
        blocker_reason=drc_reason,
        required=True,
    )

    # reconcile compares native-clean against the FULL KLayout sign-off predicate
    # (LVS-match AND DRC-clean), reusing the DRC verdict above (no second deck run).
    rv = reconcile(result, intent, tech_id=tech_id, drc_verdict=drc)
    # DIRECTION-AWARE reconcile: the sign-off follows the INDEPENDENT oracle (native
    # guides, it does not sign off). Only the DANGEROUS direction blocks.
    #   * native-CLEAN but oracle-NOT-signoff-clean = native BLIND SPOT (an under-report:
    #     native hid a real LVS/DRC defect the authoritative deck caught). This is the
    #     precise failure mode this reconcile exists to catch -> FAIL (block).
    #   * native-DIRTY but oracle-SIGNOFF-CLEAN = native OVER-STRICT (over-conservative,
    #     e.g. the PathFinder route_success convergence flag false-negative). The
    #     authoritative independent oracle (real KLayout LVS match AND real-deck DRC clean)
    #     certifies the layout clean; native is a fast proxy that cannot model the deck.
    #     This is a native-model bug to TRACK, NOT a sign-off blocker -> ADVISORY.
    if rv.agree:
        recon_status, recon_reason = VerificationStatus.PASS, ""
    elif rv.native_clean:
        recon_status = VerificationStatus.FAIL
        recon_reason = (
            f"native BLIND SPOT: native_clean=True but klayout_signoff_clean=False "
            f"(matched={rv.klayout_matched} drc_clean={rv.klayout_drc_clean}) -- native "
            f"missed a real LVS/DRC defect the authoritative deck caught"
        )
    else:
        recon_status = VerificationStatus.ADVISORY
        recon_reason = (
            f"native OVER-STRICT (advisory, non-blocking): native_clean=False but the "
            f"authoritative oracle is signoff-clean (LVS matched AND real-deck DRC clean). "
            f"Native fast model over-conservative (e.g. route_success convergence flag); a "
            f"native-model bug to reconcile, but sign-off follows the independent oracle"
        )
    out["oracle.reconcile"] = CheckerVerdict(
        checker="oracle", obligation_id="oracle.reconcile",
        status=recon_status,
        evidence={"native_clean": rv.native_clean, "klayout_matched": rv.klayout_matched,
                  "klayout_drc_clean": rv.klayout_drc_clean,
                  "klayout_signoff_clean": rv.klayout_signoff_clean,
                  "agree": rv.agree,
                  "direction": ("agree" if rv.agree else
                                "native_blind_spot" if rv.native_clean else "native_over_strict")},
        blocker_reason=recon_reason,
        required=True,
    )
    return out


def fidelity_signoff_verdicts(result: Any, intent: Any) -> Dict[str, CheckerVerdict]:
    """The GEOMETRIC-FIDELITY obligation family (F1) for an as-built route.

    Emits the route's canonical artwork to a GDS and measures the five F0
    geometric-fidelity metrics on it (the SAME single canonical artifact path --
    no second layout build), then bridges each metric to a signoff CheckerVerdict
    via :mod:`clay.layout.spec.fidelity_signoff`. A RED metric is a BLOCKING
    FAIL, so a legal-but-geometrically-wrong layout cannot be signed off green.

    Honest incompleteness: if the route produced no measurable GDS, every
    fidelity obligation is UNMEASURABLE (required => blocks), never a silent
    pass."""
    import tempfile

    from clay.layout.route.emit_gds import emit_routed_gds
    from clay.layout.spec.fidelity_signoff import fidelity_verdicts
    from clay.layout.spec.geometric_fidelity import measure_fidelity

    bundle = result.get("_bundle") if isinstance(result, dict) else None
    routed = getattr(bundle, "routed", None)
    if routed is None:
        return {}
    work = pathlib.Path(tempfile.mkdtemp(prefix="fidelity_signoff_"))
    gds = work / "ota_fidelity.gds"
    try:
        emit_routed_gds(routed, str(gds), cell_name="ota_fidelity")
        # WIRELENGTH-LOCALITY: feed the canonical AsBuiltBundle routing terminals
        # (pins_by_net) so measure_fidelity appends the wirelength obligation. Read
        # from the SAME bundle (no re-route); tech_id from the route result.
        pins_by_net = None
        try:
            tech_id = result.get("tech_id") if isinstance(result, dict) else None
            if tech_id:
                from clay.layout.route.lvs_gate import _materialize
                _r, _g, _p, pins_by_net, _t = _materialize(result, intent, tech_id)
        except Exception:
            pins_by_net = None  # honest: no pins -> wirelength simply not measured
        report = measure_fidelity(str(gds), intent, pins_by_net=pins_by_net)
    except Exception as exc:  # pragma: no cover -- emit/measure failure
        # explicit UNMEASURABLE (required => blocks); never a silent pass.
        from clay.core_contract import VerificationStatus
        return {
            "fidelity.unmeasurable": CheckerVerdict(
                checker="geometric_fidelity",
                obligation_id="fidelity.unmeasurable",
                status=VerificationStatus.UNMEASURABLE,
                evidence={"error": str(exc)},
                blocker_reason=f"geometric fidelity unmeasurable: {exc}",
                required=True,
            )
        }
    return fidelity_verdicts(report)


def _route_cell_fidelity_verdicts(res: Any, cell_def: Any, intent: Any
                                  ) -> Dict[str, CheckerVerdict]:
    """The GEOMETRIC-FIDELITY family (F1 / G7) for a GENERAL ``route_cell`` (SPINE)
    result -- the route_amp :func:`fidelity_signoff_verdicts` dual for the spine.

    Emits the SAME canonical route_cell artwork the LVS/DRC oracles emit
    (``emit_labelled_gds`` off ``res.routed`` -- one canonical artifact, no second
    layout build) and measures the F0 geometric-fidelity metrics on it, then bridges
    each metric to a signoff :class:`CheckerVerdict`. A RED metric is a BLOCKING FAIL
    (invariant I9: a legal-but-geometrically-wrong layout cannot be signed off green).

    ``intent`` supplies TOPOLOGY only (the compact-matched-group count / MOS fingers);
    the emit intent is derived from ``cell_def`` (the single design source). An
    emit/measure failure -> a single UNMEASURABLE obligation (required => blocks),
    never a silent pass."""
    import tempfile

    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.route.lvs_gate import (
        emit_labelled_gds,
        well_merge_devices_from_cell_def,
    )
    from clay.layout.spec.fidelity_signoff import fidelity_verdicts
    from clay.layout.spec.geometric_fidelity import measure_fidelity

    routed = getattr(res, "routed", None)
    if routed is None:
        return {}
    try:
        cell_name = str(getattr(cell_def, "local_name", None)
                        or getattr(cell_def, "name", None) or "cell")
        layout_intent = layout_intent_from_cell_def(cell_def)
        work = pathlib.Path(tempfile.mkdtemp(prefix="fidelity_rc_"))
        gds = work / "route_cell_fidelity.gds"
        emit_labelled_gds(
            routed, layout_intent, res.geometry, res.placement_um,
            res.pins_by_net, gds, tech=res.tech, top_cell=f"{cell_name}_routed",
            devices=well_merge_devices_from_cell_def(cell_def),
            source_cell=cell_def,
            row_program=getattr(res, "row_program", None),
            row_materialization=getattr(res, "row_materialization", None),
            row_final_certificate=getattr(res, "row_final_placement_certificate", None),
            selected_route_engine=getattr(res,"selected_route_engine",None),
        )
        # geometry-first (pins_by_net=None): the aspect / fill / enclosure / dead-region
        # metrics that carry the portrait-defect RED never depend on the routing-terminal
        # map, so the gate stays robust even when the terminal shape is unusual.
        report = measure_fidelity(str(gds), intent)
    except Exception as exc:  # pragma: no cover -- emit/measure failure
        return {
            "fidelity.unmeasurable": CheckerVerdict(
                checker="geometric_fidelity",
                obligation_id="fidelity.unmeasurable",
                status=VerificationStatus.UNMEASURABLE,
                evidence={"error": str(exc)},
                blocker_reason=f"geometric fidelity unmeasurable (route_cell): {exc}",
                required=True,
            )
        }
    return fidelity_verdicts(report)


def route_signoff_obligations(
    tech_id: str,
    *,
    result: Any = None,
    intent: Any = None,
    cell_def: Any = None,
    run_oracle: bool = True,
    include_fidelity: bool = False,
    enforce_coverage: bool = False,
    via_provenance_report: Any = None,
    layer_route_profile_report: Any = None,
    require_exact_route_evidence: bool = False,
    exact_artifact_evidence: Any = None,
) -> Dict[str, CheckerVerdict]:
    """The full obligation set for a signed-off route on ``tech_id``: capability +
    native + oracle. Every obligation is present and in an explicit state -- a required
    gap (unsupported / un-run / failed) blocks; nothing silently passes.

    DESIGN-INDEPENDENT injection (spine unification, S1.1): when ``result`` and
    ``intent`` are BOTH ``None`` (the default), the canonical leakage-free OTA is built
    (``build_ota_intent`` + ``route_amp_leakfree``) exactly as before -- the route_amp
    default path is BYTE-STABLE. Otherwise the injected ``result`` (a
    ``route_amp_leakfree`` metrics dict OR a ``RouteCellResult``) + ``intent`` (the
    reference intent -- e.g. ``_cell_def_reference_intent(cell_def)``) drive the
    obligation set. For a ``RouteCellResult`` the oracle is routed through
    ``lvs_gate_route_cell`` + a real-deck DRC on the route_cell emit -- pass the
    originating ``cell_def`` so that oracle can derive its reference netlist / ports /
    emit intent from the single design source.

    ``include_fidelity`` adds the F1 GEOMETRIC-FIDELITY obligation family
    (measured on the route's canonical GDS); a RED geometric metric blocks.

    ``enforce_coverage`` adds the F1 OBLIGATION-CLASS COVERAGE gate (the
    "no silent legality-only pass" mechanism): the AsBuiltLayout's required
    classes are {DRC, LVS, geometric_fidelity}; any required class NOT measured
    in this obligation set emits a BLOCKING INCOMPLETE/UNMEASURABLE coverage
    verdict. (Both default OFF so existing signoff callers are byte-stable; the
    F1 OTA-reports-RED-on-fidelity gate engages them.)

    ``require_exact_route_evidence`` consumes already-produced checks of the
    single emitted artifact.  Supplying either report also engages the pair: a
    missing peer is a named required FAIL, never a silent omission.

    ``exact_artifact_evidence`` is the typed, preferred seam.  It carries one
    emitter-certified GDS/manifest pair plus precomputed reports.  When supplied,
    this function never emits a GDS and never reruns an LVS/DRC/checker oracle;
    missing precomputed oracle types stay explicit UNMEASURABLE obligations.
    """
    if exact_artifact_evidence is not None:
        from clay.layout.route.exact_artifact_evidence import ExactArtifactEvidenceBundle

        if type(exact_artifact_evidence) is not ExactArtifactEvidenceBundle:
            raise TypeError(
                "exact_artifact_evidence must be ExactArtifactEvidenceBundle"
            )
        if via_provenance_report is not None or layer_route_profile_report is not None:
            raise ValueError(
                "exact_artifact_evidence cannot be mixed with loose exact-route reports"
            )
        if result is None or intent is None:
            raise ValueError(
                "exact_artifact_evidence requires the already-produced result and intent"
            )
        if not _is_route_cell_result(result) or cell_def is None:
            raise TypeError(
                "exact_artifact_evidence requires the current RouteCellResult "
                "and originating cell_def"
            )
        exact_artifact_evidence.validate_for_route_cell_result(
            result, cell_def
        )
        exact_artifact_evidence.validate_reference_equivalence(cell_def)

    if result is None and intent is None:
        from clay.layout.extract.migrate import build_ota_intent
        from clay.layout.route.anchor_ota import route_amp_leakfree

        intent = build_ota_intent()
        result = route_amp_leakfree(tech_id=tech_id)

    obs: Dict[str, CheckerVerdict] = {}
    obs.update(device_capability_verdicts(intent, tech_id))
    obs.update(native_metric_verdicts(result, intent=intent))
    if exact_artifact_evidence is not None:
        obs.update(_precomputed_exact_oracle_verdicts(result, exact_artifact_evidence))
        via_provenance_report = exact_artifact_evidence.via_provenance_report
        layer_route_profile_report = exact_artifact_evidence.layer_route_profile_report
    else:
        obs.update(_oracle_verdicts(result, intent, tech_id, run=run_oracle,
                                    cell_def=cell_def))
    if exact_artifact_evidence is not None or (
        require_exact_route_evidence
        or via_provenance_report is not None
        or layer_route_profile_report is not None
    ):
        obs.update(exact_route_evidence_required_verdicts(
            via_report=via_provenance_report,
            layer_profile_report=layer_route_profile_report,
        ))
    if exact_artifact_evidence is not None:
        structure = exact_structure_profile_required_verdict(
            exact_artifact_evidence.exact_structure_profile_report
        )
        obs[structure.obligation_id] = structure

    # A class is measured only when this invocation actually emitted its verdict.
    # Requested flags and nominal checker capability are not evidence: for example,
    # ``run_oracle=False`` deliberately emits no ``oracle.drc``.  Treating DRC as
    # measured in that case hid the required coverage gap and allowed a legality
    # claim without a DRC observation.
    measured_classes: set[str] = set()
    if "oracle.lvs" in obs:
        measured_classes.add("LVS")
    if "oracle.drc" in obs:
        measured_classes.add("DRC")

    # Wave F / G7: the env gate (CLAY_FIDELITY_GATE) engages the geometric-fidelity
    # obligation family + the coverage gate on ANY route signoff without a caller
    # change. OFF by default -> both flags stay as passed -> byte-identical (no
    # fidelity obligation, no required geometric_fidelity class). An explicit
    # include_fidelity=True (the existing OTA-reports-RED test) still wins.
    if _fidelity_gate_enabled():
        include_fidelity = True
        enforce_coverage = True

    if include_fidelity:
        # SPINE-aware: a route_cell result emits its OWN canonical GDS (the route_amp
        # _bundle.routed path cannot consume it); dispatch to the route_cell fidelity
        # family so the general spine's portrait/whitespace layout is gated too.
        if exact_artifact_evidence is not None:
            fobs = {
                "fidelity.unmeasurable": CheckerVerdict(
                    checker="geometric_fidelity",
                    obligation_id="fidelity.unmeasurable",
                    status=VerificationStatus.UNMEASURABLE,
                    evidence={
                        "precomputed_exact_artifact": True,
                        "reason": "bundle has no typed full-fidelity report",
                    },
                    blocker_reason=(
                        "full geometric fidelity was not precomputed for the exact "
                        "artifact; refusing to re-emit a second GDS"
                    ),
                    required=True,
                )
            }
        elif _is_route_cell_result(result):
            fobs = _route_cell_fidelity_verdicts(result, cell_def, intent)
        else:
            fobs = fidelity_signoff_verdicts(result, intent)
        obs.update(fobs)
        if fobs:
            measured_classes.add("geometric_fidelity")

    if enforce_coverage:
        from clay.layout.spec.fidelity_signoff import coverage_verdicts
        obs.update(coverage_verdicts(measured_classes,
                                     artifact_type="AsBuiltLayout"))
    return obs
