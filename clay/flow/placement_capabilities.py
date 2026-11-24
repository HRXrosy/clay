"""Read-only placement policy and source-level combination checks.

These checks resolve the same effective policy consumed by placement. They do
not materialize geometry, select a physical master, or issue placement evidence.
"""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import math

from clay.flow.readiness_result import ReadinessCheck


def _locations(prepared, declarations):
    return tuple(
        dict(
            getattr(declaration, "source_span", None) or {},
            artifact=prepared.definition.source_artifact,
        )
        for declaration in declarations
        if declaration is not None
    )


def plan_placement_checks(prepared, parsed, cell, intent) -> tuple[ReadinessCheck, ...]:
    """Validate declared dimensions and combinations without physical imports."""
    from clay.layout.policy import (
        aspect_policy_binding,
        outline_width_policy_binding,
        resolve_policy,
    )

    flow = getattr(cell, "flow_decl", None)
    policy_ref = getattr(flow, "policy_ref", None)
    policy = (getattr(parsed, "policies", None) or {}).get(policy_ref)
    policy_locations = _locations(prepared, (policy if policy is not None else flow,))
    try:
        if policy_ref and policy is None:
            raise ValueError(f"selected placement policy {policy_ref!r} is not declared")
        effective = resolve_policy(policy, layout_intent=intent)
        aspect_policy_binding(effective.aspect_target)
        if effective.gap_um is not None and (
            not math.isfinite(effective.gap_um) or effective.gap_um < 0.0
        ):
            raise ValueError("@place_policy gap must be non-negative and finite")
        if effective.outline_width_um is not None:
            # Match materialized_placement's exact micrometre-to-nanometre
            # contract without importing its geometry/solver implementation.
            width_nm = Decimal(str(effective.outline_width_um)) * Decimal(1000)
            if not width_nm.is_finite() or width_nm != width_nm.to_integral_value():
                raise ValueError("@place_policy outline_width must be exactly representable in integer nm")
            outline_width_policy_binding(int(width_nm))
    except (TypeError, ValueError, ArithmeticError) as exc:
        return (ReadinessCheck(
            "placement.policy", "fail", "PLACEMENT_POLICY_INVALID", str(exc),
            policy_locations,
        ),)

    checks = [ReadinessCheck(
        "placement.policy", "pass", "PLACEMENT_POLICY_VERIFIED",
        "selected placement policy and declared dimensions are valid",
        policy_locations,
        detail={"scope": "policy_only", "effective_policy": asdict(effective)},
    )]
    declarations = tuple(getattr(cell, "layout_intents", ()) or ())
    edge_rails = tuple(
        declaration for declaration in declarations
        if declaration.kind == "trunk"
        and declaration.attributes.get("edge") in ("top", "bottom")
    )
    common_centroids = tuple(
        declaration for declaration in declarations
        if declaration.kind == "common_centroid"
    )
    incompatible_with_left_column = []
    if effective.outline_width_um is not None:
        incompatible_with_left_column.append("fixed outline width")
    incompatible_description = " and ".join(incompatible_with_left_column)
    if incompatible_with_left_column and effective.left_column_group:
        checks.append(ReadinessCheck(
            "placement.combinations", "unsupported", "UNSUPPORTED_COMBINATION",
            f"left-column master with {incompatible_description} is not supported",
            _locations(prepared, (policy, *edge_rails)),
        ))
    elif incompatible_with_left_column and common_centroids:
        checks.append(ReadinessCheck(
            "placement.combinations", "unsupported", "PLACEMENT_MASTER_RESOLUTION_UNAVAILABLE",
            f"common-centroid groups with {incompatible_description} require geometry to resolve the placement master; "
            "source-only preflight cannot determine whether an implicit left column is selected",
            _locations(prepared, (policy, *common_centroids, *edge_rails)),
        ))
    return tuple(checks)


__all__ = ["plan_placement_checks"]
