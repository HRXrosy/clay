"""Independent exact-GDS closure for declared Kelvin target branches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from clay.layout.completeness.gds_witness import ExactGdsWitness


@dataclass(frozen=True)
class GdsKelvinObligation:
    """One source-declared force/sense pair measured on the exact GDS."""

    force_net: str
    sense_net: str
    target_device: str
    declaration_sha256: str
    gds_sha256: str | None
    force_target_covered: bool
    sense_target_covered: bool
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    reason: str


def _route_reaches_access(routed: object, access: object) -> bool:
    """Use the deck's same-layer conductor connectivity, including boundary touch."""

    return not routed.interacting(access).is_empty()


def verify_kelvins_on_gds(
    gds_path: str | Path,
    *,
    top_cell: str,
    kelvin_intents: object,
    materialized_topology_plan: object,
    pins_by_net: Mapping[str, object] | None,
    exact_artifact: object,
    source_cell: object,
    tech: object,
    emitted_device_geometry: Mapping[str, object] | None = None,
    emitted_device_placement_um: Mapping[str, object] | None = None,
) -> tuple[GdsKelvinObligation, ...]:
    """Prove both target-device branches from signed route rectangles in GDS."""

    from clay.layout.intent import (
        layout_intent_from_cell_def,
        replay_kelvin_inventory,
    )
    from clay.layout.route.exact_artifact_evidence import (
        ExactEmittedArtifact,
        _emitted_device_snapshot_sha256,
        _replayed_source_cell_authority,
        _terminal_pin_snapshot_sha256,
    )
    from clay.layout.route.resource_plan import TopologyPlan
    from clay.layout.route.topology_compile import compile_topology_plan

    declarations = replay_kelvin_inventory(kelvin_intents)
    if not declarations:
        raise ValueError("Kelvin exact-GDS checker requires a non-empty denominator")

    def blocked(reason: str, *, gds_sha256: str | None = None):
        return tuple(
            GdsKelvinObligation(
                force_net=item.force_net,
                sense_net=item.sense_net,
                target_device=item.target_device,
                declaration_sha256=item.sha256,
                gds_sha256=gds_sha256,
                force_target_covered=False,
                sense_target_covered=False,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for item in declarations
        )

    try:
        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("Kelvin checker requires ExactEmittedArtifact")
        if type(materialized_topology_plan) is not TopologyPlan:
            raise TypeError("Kelvin checker requires one exact TopologyPlan")
        if not isinstance(pins_by_net, Mapping):
            raise TypeError("Kelvin checker requires the routed pin mapping")
        source_sha, inventory_sha, _classes, canonical_cell = (
            _replayed_source_cell_authority(source_cell)
        )
        if (
            source_sha != exact_artifact.scope.source_cell_source_sha256
            or inventory_sha != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError("Kelvin source inventory differs from exact emitter scope")
        canonical_intent = layout_intent_from_cell_def(canonical_cell)
        canonical_declarations = replay_kelvin_inventory(canonical_intent.kelvins)
        if declarations != canonical_declarations:
            raise ValueError("Kelvin declarations differ from canonical source bytes")

        if not isinstance(emitted_device_geometry, Mapping) or not isinstance(
            emitted_device_placement_um, Mapping
        ):
            raise TypeError("Kelvin target pins lack emitter geometry authority")
        if (
            _emitted_device_snapshot_sha256(
                emitted_device_geometry, emitted_device_placement_um
            )
            != exact_artifact.scope.emitted_device_snapshot_sha256
        ):
            raise ValueError("Kelvin emitter geometry snapshot does not replay")
        if (
            _terminal_pin_snapshot_sha256(pins_by_net)
            != exact_artifact.scope.terminal_pin_snapshot_sha256
        ):
            raise ValueError("Kelvin terminal pin snapshot does not replay")

        replayed_plan = compile_topology_plan(
            pins_by_net,
            cell_def=canonical_cell,
            intent=canonical_intent,
            tech=tech,
        )
        expected_refs = {
            f"kelvin:{item.sha256}:{role}"
            for item in declarations
            for role in ("force", "sense")
        }
        expected_goals = tuple(
            goal
            for goal in replayed_plan.required_goals
            if goal.declaration_ref in expected_refs
        )
        observed_goals = tuple(
            goal
            for goal in materialized_topology_plan.required_goals
            if goal.declaration_ref in expected_refs
        )
        if observed_goals != expected_goals or len(expected_goals) != 2 * len(
            declarations
        ):
            raise ValueError("Kelvin topology goals do not replay from canonical pins")

        witness = ExactGdsWitness.open(gds_path, top_cell, exact_artifact)
        route_regions = witness.certified_route_regions(
            exact_artifact.route_manifest.entries
        )
        from types import SimpleNamespace

        from clay.layout.completeness.gds_invariants import _emitted_shape_witness
        from clay.layout.route.pins import _build_passive_cell, extract_pins
        from clay.layout.route.route_cell import bridge_devices_from_cell_def

        _devices, canonical_passives = bridge_devices_from_cell_def(canonical_cell)
        passive_by_name = {
            str(passive.name): passive for passive in canonical_passives
        }
        route_bboxes: dict[
            str, dict[tuple[int, int], list[tuple[int, int, int, int]]]
        ] = {}
        for net, layer_key, bbox in exact_artifact.route_manifest.entries:
            route_bboxes.setdefault(net, {}).setdefault(layer_key, []).append(bbox)
        route_exclusions = {
            f"route:{net}": {
                layer_key: tuple(sorted(bboxes))
                for layer_key, bboxes in layers.items()
            }
            for net, layers in route_bboxes.items()
        }
        canonical_target_claims = {}
        canonical_target_pins = {}
        for declaration in declarations:
            target = declaration.target_device
            passive = passive_by_name.get(target)
            if passive is None:
                raise ValueError(
                    f"Kelvin target {target!r} lacks canonical passive authority"
                )
            canonical_stamp = _build_passive_cell(passive, tech)
            position = emitted_device_placement_um.get(target)
            if canonical_stamp is None or position is None:
                raise ValueError(
                    f"Kelvin target {target!r} cannot be rebuilt from installed PDK"
                )
            polygons = tuple(getattr(canonical_stamp, "polygons", ()) or ())
            vertices = tuple(
                vertex
                for layer_polygon in polygons
                for vertex in tuple(layer_polygon.polygon.vertices)
            )
            if not vertices:
                raise ValueError(
                    f"Kelvin target {target!r} canonical PDK cell is empty"
                )
            center_x = (min(v.x for v in vertices) + max(v.x for v in vertices)) / 2.0
            center_y = (min(v.y for v in vertices) + max(v.y for v in vertices)) / 2.0
            rings_by_layer: dict[tuple[int, int], list[tuple[tuple[float, float], ...]]] = {}
            for layer_polygon in polygons:
                layer_key = (int(layer_polygon.layer), int(layer_polygon.datatype))
                rings_by_layer.setdefault(layer_key, []).append(tuple(
                    (
                        (float(vertex.x) - center_x) / 1000.0,
                        (float(vertex.y) - center_y) / 1000.0,
                    )
                    for vertex in layer_polygon.polygon.vertices
                ))
            canonical_emitter_stamp = SimpleNamespace(
                polys_by_layer={
                    layer_key: tuple(rings)
                    for layer_key, rings in rings_by_layer.items()
                }
            )
            canonical_pin_map = extract_pins(
                SimpleNamespace(devices=[], passives=[passive], bjts=[]),
                {target: position},
                tech=tech,
                include_power=True,
                geometry={target: canonical_emitter_stamp},
            )

            def pin_rows(pin_map: Mapping[str, object]):
                return tuple(sorted(
                    (
                        pin.net,
                        pin.device,
                        pin.terminal,
                        pin.layer,
                        pin.xy_nm,
                    )
                    for net_pins in pin_map.values()
                    for pin in tuple(net_pins)
                    if pin.device == target
                ))

            expected_pins = pin_rows(canonical_pin_map)
            observed_pins = pin_rows(pins_by_net)
            if not expected_pins or observed_pins != expected_pins:
                raise ValueError(
                    f"Kelvin target {target!r} pins differ from canonical source terminal access"
                )
            expected_artwork = _emitted_shape_witness(
                canonical_emitter_stamp, position
            )
            if not expected_artwork:
                raise ValueError(
                    f"Kelvin target {target!r} canonical artwork is empty"
                )
            claims = witness.claim_polygon_multiset(
                {target: expected_artwork},
                eps=0,
                excluded_bboxes_by_owner=route_exclusions,
            )
            canonical_target_claims[declaration.sha256] = claims[target]
            canonical_target_pins[declaration.sha256] = expected_pins
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        digest = getattr(getattr(exact_artifact, "scope", None), "exact_gds_sha256", None)
        return blocked(
            f"exact Kelvin artifact cannot be replayed: {type(exc).__name__}: {exc}",
            gds_sha256=digest,
        )

    goals_by_ref = {goal.declaration_ref: goal for goal in expected_goals}

    def target_access_region(
        target: object, target_device: str, declaration_sha256: str
    ):
        import klayout.db as kdb

        region = kdb.Region()
        point = kdb.Point(*target.xy_nm)
        for record in canonical_target_claims[declaration_sha256].get(
            target.layer, ()
        ):
            if not (
                record.bbox[0] <= point.x <= record.bbox[2]
                and record.bbox[1] <= point.y <= record.bbox[3]
            ):
                continue
            polygon = kdb.Polygon(
                [kdb.Point(x, y) for x, y in record.contours[0]]
            )
            for contour in record.contours[1:]:
                polygon.insert_hole([kdb.Point(x, y) for x, y in contour])
            region.insert(polygon)
        region.merge()
        if region.is_empty():
            raise ValueError(
                f"Kelvin target {target_device!r} has no on-device access artwork"
            )
        return region

    results = []
    for declaration in declarations:
        covered: dict[str, bool] = {}
        coverage_reason: dict[str, str] = {}
        try:
            for role, net in (
                ("force", declaration.force_net),
                ("sense", declaration.sense_net),
            ):
                goal = goals_by_ref[f"kelvin:{declaration.sha256}:{role}"]
                target = goal.target
                if target is None:
                    covered[role] = False
                    coverage_reason[role] = "target_terminal_missing"
                    continue
                expected_provenance = f"pin:{declaration.target_device}:"
                source_roles = tuple(
                    row[2]
                    for row in canonical_target_pins[declaration.sha256]
                    if row[0] == net
                )
                if (
                    len(source_roles) != 1
                    or target.provenance
                    != expected_provenance + source_roles[0]
                ):
                    covered[role] = False
                    coverage_reason[role] = "target_provenance_mismatch"
                    continue
                access = target_access_region(
                    target,
                    declaration.target_device,
                    declaration.sha256,
                )
                routed = route_regions.get(net, {}).get(target.layer)
                if routed is None:
                    covered[role] = False
                    coverage_reason[role] = "route_region_missing"
                elif not _route_reaches_access(routed, access):
                    covered[role] = False
                    coverage_reason[role] = "route_target_disjoint"
                else:
                    covered[role] = True
                    coverage_reason[role] = "pass"
        except Exception as exc:  # noqa: BLE001 - one typed Kelvin obligation
            results.append(
                GdsKelvinObligation(
                    force_net=declaration.force_net,
                    sense_net=declaration.sense_net,
                    target_device=declaration.target_device,
                    declaration_sha256=declaration.sha256,
                    gds_sha256=witness.gds_sha256,
                    force_target_covered=False,
                    sense_target_covered=False,
                    status="unmeasurable",
                    measured_on_gds=False,
                    reason=(
                        "Kelvin target artwork cannot be replayed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
            continue
        both = bool(covered.get("force") and covered.get("sense"))
        results.append(
            GdsKelvinObligation(
                force_net=declaration.force_net,
                sense_net=declaration.sense_net,
                target_device=declaration.target_device,
                declaration_sha256=declaration.sha256,
                gds_sha256=witness.gds_sha256,
                force_target_covered=bool(covered.get("force")),
                sense_target_covered=bool(covered.get("sense")),
                status="pass" if both else "unmeasurable",
                measured_on_gds=both,
                reason=(
                    "force and sense routes independently reach the declared target"
                    if both
                    else (
                        "Kelvin target coverage incomplete: "
                        f"force={coverage_reason.get('force', 'unknown')}, "
                        f"sense={coverage_reason.get('sense', 'unknown')}"
                    )
                ),
            )
        )
    return tuple(results)


__all__ = ["GdsKelvinObligation", "verify_kelvins_on_gds"]
