"""Independent exact-GDS obligation for coordinate-free power planes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Tuple

from clay.layout.completeness.gds_witness import rect_polygon_witness
from clay.layout.intent import replay_power_plane_inventory
from clay.layout.route.power_plane import (
    MaterializedPowerPlaneFact,
    compile_power_plane_facts,
)


@dataclass(frozen=True)
class GdsPowerPlaneObligation:
    net: str
    layer_key: Tuple[int, int] | None
    bbox_nm: Tuple[int, int, int, int] | None
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    reason: str = ""


def verify_power_planes_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    power_plane_intents: object,
    materialized_power_plane_facts: object,
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    final_placement_certificate: Any,
    placement_state_sha256: str | None,
    tech: Any,
) -> Tuple[GdsPowerPlaneObligation, ...]:
    """Recompile each plane and claim its full-envelope polygon from exact GDS."""

    declarations = replay_power_plane_inventory(power_plane_intents)
    denominator = declarations

    def blocked(reason: str) -> Tuple[GdsPowerPlaneObligation, ...]:
        return tuple(
            GdsPowerPlaneObligation(
                net=item.net,
                layer_key=None,
                bbox_nm=None,
                declaration_sha256=item.sha256,
                gds_sha256=(
                    getattr(getattr(exact_artifact, "scope", None), "exact_gds_sha256", None)
                ),
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for item in denominator
        )

    try:
        from clay.layout.completeness.gds_invariants import (
            _replay_exact_final_placement_artwork,
        )

        canonical_intent, _classes, _transformed, witness = (
            _replay_exact_final_placement_artwork(
                gds_path=gds_path,
                top_cell=top_cell,
                exact_artifact=exact_artifact,
                source_cell=source_cell,
                final_placement_certificate=final_placement_certificate,
                placement_state_sha256=placement_state_sha256,
                emitted_device_geometry=emitted_device_geometry,
                emitted_device_placement_um=emitted_device_placement_um,
                tech=tech,
            )
        )
        canonical_declarations = replay_power_plane_inventory(
            canonical_intent.power_planes
        )
        denominator = canonical_declarations or declarations
        if canonical_declarations != declarations:
            raise ValueError("power-plane inventory differs from canonical source")
        if not canonical_declarations:
            return ()
        expected = compile_power_plane_facts(
            canonical_declarations,
            tuple(final_placement_certificate.bbox_nm),
            tech,
        )
        if (
            type(materialized_power_plane_facts) is not tuple
            or any(
                type(item) is not MaterializedPowerPlaneFact
                for item in materialized_power_plane_facts
            )
            or materialized_power_plane_facts != expected
        ):
            raise ValueError("materialized power-plane facts do not replay")
        if not isinstance(emitted_device_geometry, Mapping) or not isinstance(
            emitted_device_placement_um, Mapping
        ):
            raise TypeError("power-plane checker requires emitter mappings")

        expected_by_owner = {}
        for fact in expected:
            stamp = emitted_device_geometry.get(fact.content_id)
            if stamp is None or fact.content_id not in emitted_device_placement_um:
                raise ValueError("materialized power-plane artwork is missing")
            if (
                str(getattr(stamp, "source", "")) != "power_plane"
                or str(getattr(stamp, "kind", "")) != "power_plane"
                or getattr(stamp, "power_plane_declaration_sha256", None)
                != fact.declaration_sha256
                or tuple(emitted_device_placement_um[fact.content_id]) != (0.0, 0.0)
            ):
                raise ValueError("materialized power-plane ownership differs")
            expected_by_owner[fact.content_id] = {
                fact.layer_key: (rect_polygon_witness(fact.bbox_nm),)
            }
        excluded_routes = {
            f"route:{ordinal}:{net}": {layer_key: (bbox_nm,)}
            for ordinal, (net, layer_key, bbox_nm) in enumerate(
                exact_artifact.route_manifest.entries
            )
        }
        witness.claim_polygon_multiset(
            expected_by_owner,
            eps=0,
            excluded_bboxes_by_owner=excluded_routes,
        )
    except Exception as exc:  # noqa: BLE001 - one typed exact-GDS obligation gate
        if not denominator:
            raise ValueError(
                "exact power-plane checker cannot prove an empty denominator"
            ) from exc
        return blocked(
            "exact power-plane artwork cannot be replayed: "
            f"{type(exc).__name__}: {exc}"
        )

    return tuple(
        GdsPowerPlaneObligation(
            net=fact.net,
            layer_key=fact.layer_key,
            bbox_nm=fact.bbox_nm,
            declaration_sha256=fact.declaration_sha256,
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
            status="pass",
            measured_on_gds=True,
            reason="exact GDS contains the declared full-envelope power plane",
        )
        for fact in expected
    )


__all__ = ["GdsPowerPlaneObligation", "verify_power_planes_on_gds"]
