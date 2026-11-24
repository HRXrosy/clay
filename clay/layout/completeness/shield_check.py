"""Independent exact-GDS closure for declared route shields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from clay.layout.completeness.gds_witness import ExactGdsWitness


@dataclass(frozen=True)
class GdsShieldObligation:
    aggressor_net: str
    victim_net: str
    shield_net: str
    declaration_sha256: str
    gds_sha256: str | None
    fence_layer_key: tuple[int, int] | None
    fence_endpoints_nm: tuple[tuple[int, int], tuple[int, int]] | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    reason: str


def _pin_row(pin: object) -> tuple[object, ...]:
    return (
        getattr(pin, "net", None),
        getattr(pin, "device", None),
        getattr(pin, "terminal", None),
        getattr(pin, "layer", None),
        getattr(pin, "xy_nm", None),
    )


def _normalized_net(value: str) -> str:
    return value.rstrip("!").casefold()


def _fence_region(fact: object):
    import klayout.db as kdb

    (x0, y0), (x1, y1) = fact.endpoints_nm
    if fact.axis == "V" and x0 == x1 and y0 < y1:
        return kdb.Region(kdb.Box(x0, y0, x0 + 1, y1))
    if fact.axis == "H" and y0 == y1 and x0 < x1:
        return kdb.Region(kdb.Box(x0, y0, x1, y0 + 1))
    raise ValueError("shield fence endpoints do not form one nonvacuous axis")


def verify_shields_on_gds(
    gds_path: str | Path,
    *,
    top_cell: str,
    shield_intents: object,
    pins_by_net: Mapping[str, object] | None,
    exact_artifact: object,
    source_cell: object,
    tech: object,
    emitted_device_geometry: Mapping[str, object] | None = None,
    emitted_device_placement_um: Mapping[str, object] | None = None,
    row_final_placement: object | None = None,
) -> tuple[GdsShieldObligation, ...]:
    """Replay each shield fence and require its complete line in the exact GDS."""

    from clay.layout.intent import layout_intent_from_cell_def, replay_shield_inventory
    from clay.layout.route.exact_artifact_evidence import (
        ExactEmittedArtifact,
        _canonical_source_member_inventory,
        _emitted_device_snapshot_sha256,
        _replayed_source_cell_authority,
        _terminal_pin_snapshot_sha256,
    )
    from clay.layout.route.shield import (
        installed_shield_layer_policy,
        materialize_declared_shield_fences,
    )

    declarations = replay_shield_inventory(shield_intents)
    if not declarations:
        raise ValueError("shield exact-GDS checker requires a non-empty denominator")

    def blocked(reason: str, *, digest: str | None = None):
        return tuple(
            GdsShieldObligation(
                aggressor_net=item.aggressor,
                victim_net=item.victim,
                shield_net=item.shield_net,
                declaration_sha256=item.sha256,
                gds_sha256=digest,
                fence_layer_key=None,
                fence_endpoints_nm=None,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for item in declarations
        )

    try:
        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("shield checker requires ExactEmittedArtifact")
        if not isinstance(pins_by_net, Mapping):
            raise TypeError("shield checker requires the routed pin mapping")
        source_sha, inventory_sha, _classes, canonical_cell = (
            _replayed_source_cell_authority(source_cell)
        )
        if (
            source_sha != exact_artifact.scope.source_cell_source_sha256
            or inventory_sha != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError("shield source inventory differs from emitter scope")
        canonical = replay_shield_inventory(
            layout_intent_from_cell_def(canonical_cell).shields
        )
        if declarations != canonical:
            raise ValueError("shield declarations differ from canonical source bytes")
        if (
            _terminal_pin_snapshot_sha256(pins_by_net)
            != exact_artifact.scope.terminal_pin_snapshot_sha256
        ):
            raise ValueError("shield terminal pin snapshot does not replay")
        if not isinstance(emitted_device_geometry, Mapping) or not isinstance(
            emitted_device_placement_um, Mapping
        ):
            raise TypeError("shield source pins lack emitter geometry authority")
        if (
            _emitted_device_snapshot_sha256(
                emitted_device_geometry, emitted_device_placement_um
            )
            != exact_artifact.scope.emitted_device_snapshot_sha256
        ):
            raise ValueError("shield emitter geometry snapshot does not replay")

        source_inventory = _canonical_source_member_inventory(canonical_cell)
        source_names = frozenset(row[0] for row in source_inventory)
        source_bindings = {
            (row[0], role, _normalized_net(net))
            for row in source_inventory
            for role, net in row[4]
        }
        for raw_net, pins in pins_by_net.items():
            if type(raw_net) is not str or type(pins) not in (tuple, list):
                raise TypeError("shield routed pins are not canonical")
            for pin in pins:
                owner = getattr(pin, "device", None)
                if owner not in source_names:
                    continue
                if (
                    getattr(pin, "net", None) != raw_net
                    or (
                        owner,
                        getattr(pin, "terminal", None),
                        _normalized_net(raw_net),
                    )
                    not in source_bindings
                ):
                    raise ValueError(
                        "shield terminal domains differ from canonical source pins"
                    )
        from types import SimpleNamespace

        from clay.layout.route.pins import extract_pins
        from clay.layout.route.route_cell import (
            _declared_ground_ports,
            _relocate_array_member_pins,
            bridge_bjts_from_cell_def,
            bridge_devices_from_cell_def,
        )

        devices, passives = bridge_devices_from_cell_def(canonical_cell)
        bridged = SimpleNamespace(
            devices=devices,
            passives=passives,
            bjts=bridge_bjts_from_cell_def(canonical_cell),
        )
        canonical_pins = extract_pins(
            bridged,
            dict(emitted_device_placement_um),
            tech=tech,
            extra_power_nets=_declared_ground_ports(canonical_cell),
            geometry=dict(emitted_device_geometry),
            include_power=True,
            per_device_fold=True,
        )
        # Source members in a retained array have no standalone MOS stamp.
        # Replay the same authenticated native straps used before routing,
        # rather than comparing the routed world with obsolete scalar probes.
        if row_final_placement is not None:
            _relocate_array_member_pins(
                canonical_pins, bridged, canonical_cell,
                layout_intent_from_cell_def(canonical_cell), tech,
                dict(emitted_device_placement_um),
                placement_obj=row_final_placement,
                geometry=dict(emitted_device_geometry),
            )
        domain_nets = {
            _normalized_net(net)
            for declaration in declarations
            for net in (declaration.aggressor, declaration.victim)
        }

        def domain_pin_rows(pin_map: Mapping[str, object]):
            return tuple(sorted(
                (
                    _normalized_net(str(getattr(pin, "net", ""))),
                    getattr(pin, "device", None),
                    getattr(pin, "terminal", None),
                    getattr(pin, "layer", None),
                    getattr(pin, "xy_nm", None),
                )
                for raw_net, pins in pin_map.items()
                if _normalized_net(str(raw_net)) in domain_nets
                for pin in tuple(pins)
                if getattr(pin, "device", None) in source_names
            ))

        expected_domain_pins = domain_pin_rows(canonical_pins)
        observed_domain_pins = domain_pin_rows(pins_by_net)
        if not expected_domain_pins or observed_domain_pins != expected_domain_pins:
            raise ValueError(
                "shield source terminal denominator differs from canonical "
                "PDK access replay"
            )

        base_pins = {
            net: [
                pin
                for pin in pins
                if not str(getattr(pin, "device", "")).startswith(
                    "__declared_shield__:"
                )
            ]
            for net, pins in pins_by_net.items()
        }
        layer_by_axis, grid_nm, minimum_width_nm = (
            installed_shield_layer_policy(tech)
        )
        facts = materialize_declared_shield_fences(
            canonical,
            base_pins,
            layer_by_axis=layer_by_axis,
            manufacturing_grid_nm=grid_nm,
            minimum_width_nm=minimum_width_nm,
            domain_pin_owners=source_names,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            tech=tech,
            wire_widths_nm=layout_intent_from_cell_def(canonical_cell).wire_widths,
        )
        expected_pins = tuple(sorted(
            _pin_row(pin)
            for pins in base_pins.values()
            for pin in pins
            if str(getattr(pin, "device", "")).startswith(
                "__declared_shield__:"
            )
        ))
        observed_pins = tuple(sorted(
            _pin_row(pin)
            for pins in pins_by_net.values()
            for pin in pins
            if str(getattr(pin, "device", "")).startswith(
                "__declared_shield__:"
            )
        ))
        if expected_pins != observed_pins:
            raise ValueError("shield fence pins do not replay from source terminals")

        witness = ExactGdsWitness.open(gds_path, top_cell, exact_artifact)
        route_regions = witness.certified_route_regions(
            exact_artifact.route_manifest.entries
        )
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        digest = getattr(getattr(exact_artifact, "scope", None), "exact_gds_sha256", None)
        return blocked(
            f"exact shield artifact cannot be replayed: {type(exc).__name__}: {exc}",
            digest=digest,
        )

    results = []
    for declaration, fact in zip(declarations, facts):
        try:
            # The aggressor/victim terminal domains were already replayed from
            # canonical source pins and bound to the exact emitted-device
            # snapshot above.  A legal one-terminal signal intentionally has no
            # invented route rectangle, so requiring route-manifest entries for
            # those two domains makes a real fence unmeasurable.  Only the
            # shield net owns the routed fence that this obligation proves.
            matches = tuple(
                net for net in route_regions
                if _normalized_net(net) == _normalized_net(declaration.shield_net)
            )
            if len(matches) != 1:
                raise ValueError("shield net has no exact routed region")
            routed = route_regions[matches[0]].get(fact.layer_key)
            if routed is None or not (_fence_region(fact) - routed).is_empty():
                raise ValueError("shield route does not cover the complete declared fence")
            results.append(GdsShieldObligation(
                aggressor_net=declaration.aggressor,
                victim_net=declaration.victim,
                shield_net=declaration.shield_net,
                declaration_sha256=declaration.sha256,
                gds_sha256=witness.gds_sha256,
                fence_layer_key=fact.layer_key,
                fence_endpoints_nm=fact.endpoints_nm,
                status="pass",
                measured_on_gds=True,
                reason="exact GDS contains the complete source-replayed shield fence",
            ))
        except Exception as exc:  # noqa: BLE001 - one obligation is fail-closed
            results.append(GdsShieldObligation(
                aggressor_net=declaration.aggressor,
                victim_net=declaration.victim,
                shield_net=declaration.shield_net,
                declaration_sha256=declaration.sha256,
                gds_sha256=witness.gds_sha256,
                fence_layer_key=fact.layer_key,
                fence_endpoints_nm=fact.endpoints_nm,
                status="unmeasurable",
                measured_on_gds=False,
                reason=f"shield fence cannot be proven: {type(exc).__name__}: {exc}",
            ))
    return tuple(results)


__all__ = ["GdsShieldObligation", "verify_shields_on_gds"]
