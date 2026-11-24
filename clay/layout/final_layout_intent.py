"""Fail-closed intent checks over the exact emitted layout artifact.

Placement and routing certificates explain how Clay reached a layout.  This
module checks what was actually emitted.  It deliberately lives in the
installed ``clay.layout`` package so canonical flow admission does not depend
on checkout-only optimization tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

from clay.layout.compact_intent_domain import compact_intent_domain_spec
from clay.layout.completeness.gds_invariants import (
    GdsAnchorObligation,
    GdsBandObligation,
    GdsChannelObligation,
    GdsCommonCentroidObligation,
    GdsDensityWhitespaceObligation,
    GdsFingerArrayOrderObligation,
    GdsFunctionalOdFillObligation,
    GdsGuardRingObligation,
    GdsInterdigitationObligation,
    GdsPassiveBankPartitionObligation,
    GdsRailAffinityObligation,
    GdsWellIslandShareObligation,
)
from clay.layout.completeness.power_plane_check import GdsPowerPlaneObligation
from clay.layout.completeness.kelvin_check import GdsKelvinObligation
from clay.layout.completeness.shield_check import GdsShieldObligation
from clay.layout.completeness.adjacent_check import GdsAdjacentObligation
from clay.layout.place_blocks_2d import ASPECT_RELATIVE_TOLERANCE
from clay.layout.policy import aspect_policy_binding
from clay.layout.route.group_ports import GdsGroupRoutePortObligation
from clay.layout.intent_closure import ConsumerProjection
from clay.layout.intent_facts import (
    EntityKind,
    EntityRef,
    IntentFactKind,
    IntentFactSet,
    SourceInventory,
    REGISTERED_CONSUMER_ORDER,
)
from clay.layout.place.intent_consumption import (
    _ObservedRelation,
    _bind_observed_relations,
    _canonical_projection,
    _ordinal_order_relations,
    _relation_from_payload,
)


_SCHEMA = "clay.final-layout-intent-certificate/v18"


@dataclass(frozen=True)
class FinalAspectObligation:
    """One declared range the emitted top-cell aspect must satisfy."""

    source: str
    minimum: float
    maximum: float
    target: float | None = None
    relative_tolerance: float | None = None


@dataclass(frozen=True)
class RouteContainmentObligation:
    """Exact emitted-shape containment inside the pre-route floorplan."""

    status: Literal["pass", "fail", "unmeasurable"]
    floorplan_bbox_nm: tuple[int, int, int, int] | None
    max_overflow_nm: float | None
    offending_layers: tuple[str, ...]
    offending_polygon_count: int
    reason: str


@dataclass(frozen=True)
class _ArtifactMemberReadback:
    """Per-source-member artwork identities claimed from one exact GDS."""

    gds_sha256: str
    member_owner_shape_digests: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        if (
            type(self.gds_sha256) is not str
            or len(self.gds_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.gds_sha256)
        ):
            raise ValueError("artifact member readback requires an exact GDS SHA-256")
        if type(self.member_owner_shape_digests) is not tuple:
            raise TypeError("artifact member readback rows must be an exact tuple")
        rows = []
        for row in self.member_owner_shape_digests:
            if type(row) is not tuple or len(row) != 3:
                raise TypeError("artifact member readback row must be an exact triple")
            member, owner, digest = row
            if (
                type(member) is not str
                or not member
                or type(owner) is not str
                or not owner
                or type(digest) is not str
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise TypeError("artifact member readback row is malformed")
            rows.append(row)
        if tuple(rows) != tuple(sorted(rows)) or len(
            {member for member, _owner, _digest in rows}
        ) != len(rows):
            raise ValueError("artifact member readback rows must be sorted and unique")


def _well_domain_relations_from_gds_obligations(
    *,
    well_island_shares: tuple[GdsWellIslandShareObligation, ...],
    exact_gds_sha256: str,
    read_members: tuple[str, ...],
) -> tuple[_ObservedRelation, ...]:
    """Normalize only exhaustive, artifact-bound shared-well measurements."""

    if (
        type(exact_gds_sha256) is not str
        or len(exact_gds_sha256) != 64
        or any(char not in "0123456789abcdef" for char in exact_gds_sha256)
    ):
        raise ValueError("shared-well relations require one exact GDS SHA-256")
    if (
        type(read_members) is not tuple
        or read_members != tuple(sorted(set(read_members)))
        or any(type(member) is not str or not member for member in read_members)
    ):
        raise ValueError("shared-well read members must be canonical and unique")
    if type(well_island_shares) is not tuple or any(
        type(row) is not GdsWellIslandShareObligation
        for row in well_island_shares
    ):
        raise TypeError("shared-well GDS obligations must be one exact typed tuple")
    groups = tuple(row.group for row in well_island_shares)
    if groups != tuple(sorted(set(groups))):
        raise ValueError("shared-well GDS obligations must be canonical and unique")
    readable = set(read_members)
    relations: list[_ObservedRelation] = []
    for row in well_island_shares:
        if row.gds_sha256 != exact_gds_sha256:
            raise ValueError("shared-well obligation binds different GDS bytes")
        if (
            type(row.group) is not str
            or not row.group
            or type(row.body) is not str
            or not row.body
            or row.well not in {"nwell", "pwell"}
            or type(row.members) is not tuple
            or len(row.members) < 2
            or row.members != tuple(sorted(set(row.members)))
            or not set(row.members) <= readable
            or type(row.layer_key) is not tuple
            or len(row.layer_key) != 2
            or any(type(value) is not int or value < 0 for value in row.layer_key)
            or type(row.declaration_sha256) is not str
            or len(row.declaration_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in row.declaration_sha256
            )
            or type(row.island_bbox_nm) is not tuple
            or len(row.island_bbox_nm) != 4
            or any(type(value) is not int for value in row.island_bbox_nm)
            or row.island_bbox_nm[2] <= row.island_bbox_nm[0]
            or row.island_bbox_nm[3] <= row.island_bbox_nm[1]
        ):
            raise ValueError(
                "shared-well obligation lacks exhaustive measured membership"
            )
        if (
            row.status != "pass"
            or row.measured_on_gds is not True
            or type(row.foreign_members) is not tuple
            or row.foreign_members
        ):
            raise ValueError(
                "shared-well relation requires a measured exclusive component "
                "without foreign members"
            )
        domain = EntityRef(EntityKind.WELL_DOMAIN, row.group)
        relations.extend(
            (
                _relation_from_payload(
                    IntentFactKind.WELL_DOMAIN,
                    tuple(
                        sorted(
                            (
                                domain,
                                *(
                                    EntityRef(EntityKind.DEVICE, member)
                                    for member in row.members
                                ),
                            )
                        )
                    ),
                    {"body_relation": "same_body", "sharing": "shared"},
                ),
                _relation_from_payload(
                    IntentFactKind.NO_FOREIGN_ENVELOPE,
                    (domain,),
                    {"scope": "members"},
                ),
            )
        )
    canonical = tuple(sorted(relations))
    if len(set(canonical)) != len(canonical):
        raise ValueError("shared-well GDS producer emitted duplicate relations")
    return canonical


def _build_final_gds_fact_projection(
    *,
    fact_set: IntentFactSet,
    source_inventory: SourceInventory,
    place_groups: tuple[object, ...],
    route_groups: tuple[object, ...],
    floorplan_column_groups: tuple[tuple[object, ...], ...],
    member_readback: _ArtifactMemberReadback,
    group_route_ports: tuple[GdsGroupRoutePortObligation, ...],
    route_containment: RouteContainmentObligation,
    well_island_shares: tuple[GdsWellIslandShareObligation, ...] = (),
    passive_bank_partitions: tuple[
        GdsPassiveBankPartitionObligation, ...
    ] = (),
) -> ConsumerProjection:
    """Project final facts only from reread artwork and measured route evidence."""

    if type(member_readback) is not _ArtifactMemberReadback:
        raise TypeError("final GDS projection requires exact member readback")
    if type(group_route_ports) is not tuple or any(
        type(row) is not GdsGroupRoutePortObligation for row in group_route_ports
    ):
        raise TypeError("final GDS group-route evidence must be an exact tuple")
    if type(route_containment) is not RouteContainmentObligation:
        raise TypeError("final GDS projection requires exact containment evidence")
    if route_containment.status != "pass":
        raise ValueError("final GDS containment must pass before fact projection")
    port_keys = tuple((row.group, row.net) for row in group_route_ports)
    if port_keys != tuple(sorted(port_keys)) or len(set(port_keys)) != len(port_keys):
        raise ValueError("final GDS group-route evidence must be sorted and unique")
    for row in group_route_ports:
        if row.gds_sha256 != member_readback.gds_sha256:
            raise ValueError("final member and route evidence bind different GDS bytes")
        if (
            row.status != "pass"
            or not row.measured_on_gds
            or row.terminal_count <= 0
            or row.connected_terminal_count != row.terminal_count
            or row.boundary_crossing_count != 1
        ):
            raise ValueError(
                f"final GDS route port {row.group}:{row.net} lacks measured exact "
                f"terminals and one portal boundary crossing: status={row.status}, "
                f"measured_on_gds={row.measured_on_gds}, "
                f"connected_terminals={row.connected_terminal_count}/"
                f"{row.terminal_count}, boundary_crossings="
                f"{row.boundary_crossing_count}, detail={row.reason}"
            )
    if type(place_groups) is not tuple or type(route_groups) is not tuple:
        raise TypeError("final GDS declarations must be exact tuples")
    floorplan_rows = _canonical_projection(floorplan_column_groups)
    read_members = {
        member for member, _owner, _digest in member_readback.member_owner_shape_digests
    }
    actual: set[_ObservedRelation] = set()

    def add(
        kind: IntentFactKind,
        subjects: tuple[EntityRef, ...],
        payload: dict[str, object],
    ) -> None:
        relation = _relation_from_payload(kind, subjects, payload)
        if relation in actual:
            raise ValueError("final GDS producer emitted a duplicate relation")
        actual.add(relation)

    actual.update(
        _well_domain_relations_from_gds_obligations(
            well_island_shares=well_island_shares,
            exact_gds_sha256=member_readback.gds_sha256,
            read_members=tuple(sorted(read_members)),
        )
    )

    if type(passive_bank_partitions) is not tuple or any(
        type(row) is not GdsPassiveBankPartitionObligation
        for row in passive_bank_partitions
    ):
        raise TypeError("final GDS passive-bank evidence must be an exact tuple")
    partition_keys = tuple(
        (row.parent, row.bank) for row in passive_bank_partitions
    )
    if (
        partition_keys != tuple(sorted(partition_keys))
        or len(set(partition_keys)) != len(partition_keys)
    ):
        raise ValueError(
            "final GDS passive-bank evidence must be sorted and unique"
        )
    for row in passive_bank_partitions:
        if (
            row.status != "pass"
            or row.measured_on_gds is not True
            or row.gds_sha256 != member_readback.gds_sha256
            or row.claimed_bbox_nm is None
            or row.graph_sha256 is None
            or row.emission_certificate_sha256 is None
            or row.members != tuple(sorted(set(row.members)))
            or not row.members
            or not set(row.members) <= read_members
            or row.macro_id != f"passive_bank:{row.parent}:{row.bank}"
        ):
            raise ValueError(
                "final GDS passive-bank relation lacks exact measured authority"
            )
        add(
            IntentFactKind.MEMBER_PARTITION,
            (
                EntityRef(EntityKind.MODULE, row.macro_id),
                *(
                    EntityRef(EntityKind.DEVICE, member)
                    for member in row.members
                ),
            ),
            {"complete": True},
        )

    members_by_place_group: dict[str, tuple[str, ...]] = {}
    for group in place_groups:
        name = getattr(group, "group", None)
        members = getattr(group, "devices", None)
        if type(name) is not str or not name or type(members) is not tuple:
            raise TypeError("final GDS place-group declaration is malformed")
        if not members:
            raise ValueError("final GDS place-group members are empty")
        members = tuple(sorted(members))
        members_by_place_group[name] = members
        if set(members) <= read_members:
            place_ref = EntityRef(EntityKind.PLACE_GROUP, name)
            member_refs = tuple(
                EntityRef(EntityKind.DEVICE, member) for member in members
            )
            for kind in (
                IntentFactKind.MEMBER_PARTITION,
                IntentFactKind.MEMBER_CONSERVATION,
            ):
                add(kind, (place_ref, *member_refs), {"complete": True})
            add(
                IntentFactKind.FINAL_CONTAINMENT,
                (place_ref,),
                {"scope": "final_layout"},
            )

    declared_ordinal_by_group = {
        name: (tier, column)
        for name, tier, column, _count, _terms in floorplan_rows
    }
    readback_ordinal_by_group: dict[str, tuple[int, int]] = {}
    for name, (tier, column) in sorted(declared_ordinal_by_group.items()):
        members = members_by_place_group.get(name)
        if members is not None and set(members) <= read_members:
            readback_ordinal_by_group[name] = (tier, column)
            add(
                IntentFactKind.ORDINAL_CELL,
                (EntityRef(EntityKind.PLACE_GROUP, name),),
                {"tier": tier, "column": column},
            )

    actual.update(_ordinal_order_relations(readback_ordinal_by_group))

    ports_by_group: dict[str, tuple[str, ...]] = {}
    for group_name in sorted({row.group for row in group_route_ports}):
        ports_by_group[group_name] = tuple(
            sorted(row.net for row in group_route_ports if row.group == group_name)
        )
    for declaration in route_groups:
        name = getattr(declaration, "group", None)
        source_groups = getattr(declaration, "place_groups", None)
        if type(name) is not str or not name or type(source_groups) is not tuple:
            raise TypeError("final GDS route-group declaration is malformed")
        if not source_groups:
            source_groups = (name,)
        if source_groups != tuple(sorted(set(source_groups))) or any(
            group not in members_by_place_group for group in source_groups
        ):
            raise ValueError("final GDS route-group owners are not canonical")
        all_members = {
            member
            for group in source_groups
            for member in members_by_place_group[group]
        }
        if not all_members <= read_members:
            continue
        route_ref = EntityRef(EntityKind.ROUTE_GROUP, name)
        place_refs = tuple(
            EntityRef(EntityKind.PLACE_GROUP, group) for group in source_groups
        )
        for kind in (
            IntentFactKind.MEMBER_PARTITION,
            IntentFactKind.MEMBER_CONSERVATION,
        ):
            add(kind, (route_ref, *place_refs), {"complete": True})
        nets = ports_by_group.get(name, ())
        if nets:
            subjects = (
                route_ref,
                *place_refs,
                *(EntityRef(EntityKind.NET, net) for net in nets),
            )
            for kind in (
                IntentFactKind.EXTERNAL_PORT_DENOMINATOR,
                IntentFactKind.TOPOLOGY_EXTERNAL_NET_INVENTORY,
            ):
                add(kind, subjects, {"complete": True})
            add(
                IntentFactKind.FINAL_CONTAINMENT,
                (route_ref,),
                {"scope": "final_layout"},
            )

    return _bind_observed_relations(
        consumer="final_gds",
        fact_set=fact_set,
        source_inventory=source_inventory,
        observed_relations=tuple(sorted(actual)),
    )


@dataclass(frozen=True)
class FinalLayoutIntentCertificate:
    """Artifact-bound verdict for declared final-layout intent."""

    schema: str
    top_cell: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable"]
    measured_bbox_um: tuple[float, float] | None
    measured_aspect: float | None
    aspect: FinalAspectObligation | None
    reason: str
    route_containment: RouteContainmentObligation | None = None
    physical_macro_graph_sha256: str | None = None
    candidate_sha256: str | None = None
    placement_state_sha256: str | None = None
    guard_rings: tuple[GdsGuardRingObligation, ...] = ()
    bands: tuple[GdsBandObligation, ...] = ()
    power_planes: tuple[GdsPowerPlaneObligation, ...] = ()
    kelvins: tuple[GdsKelvinObligation, ...] = ()
    shields: tuple[GdsShieldObligation, ...] = ()
    adjacencies: tuple[GdsAdjacentObligation, ...] = ()
    anchors: tuple[GdsAnchorObligation, ...] = ()
    rail_affinities: tuple[GdsRailAffinityObligation, ...] = ()
    channels: tuple[GdsChannelObligation, ...] = ()
    densities: tuple[GdsDensityWhitespaceObligation, ...] = ()
    functional_od_fills: tuple[GdsFunctionalOdFillObligation, ...] = ()
    finger_array_orders: tuple[GdsFingerArrayOrderObligation, ...] = ()
    passive_bank_partitions: tuple[GdsPassiveBankPartitionObligation, ...] = ()
    well_island_shares: tuple[GdsWellIslandShareObligation, ...] = ()
    common_centroids: tuple[GdsCommonCentroidObligation, ...] = ()
    interdigitations: tuple[GdsInterdigitationObligation, ...] = ()
    group_route_ports: tuple[GdsGroupRoutePortObligation, ...] = ()
    fact_projections: tuple[ConsumerProjection, ...] = ()

    def __post_init__(self) -> None:
        if type(self.fact_projections) is not tuple or any(
            type(row) is not ConsumerProjection for row in self.fact_projections
        ):
            raise TypeError("final layout fact projections must be an exact tuple")
        if self.fact_projections:
            consumers = tuple(row.consumer for row in self.fact_projections)
            expected_order = tuple(
                consumer
                for consumer in REGISTERED_CONSUMER_ORDER
                if consumer in set(consumers)
            )
            if consumers != expected_order or len(set(consumers)) != len(consumers):
                raise ValueError(
                    "final layout fact projections violate registered consumer order"
                )

    @property
    def blocking(self) -> bool:
        return self.status != "pass"

    def canonical_data(self) -> dict[str, object]:
        data = {
            "schema": self.schema,
            "top_cell": self.top_cell,
            "gds_sha256": self.gds_sha256,
            "status": self.status,
            "measured_bbox_um": (
                list(self.measured_bbox_um)
                if self.measured_bbox_um is not None
                else None
            ),
            "measured_aspect": self.measured_aspect,
            "aspect": asdict(self.aspect) if self.aspect is not None else None,
            "route_containment": (
                asdict(self.route_containment)
                if self.route_containment is not None
                else None
            ),
            "physical_macro_graph_sha256": self.physical_macro_graph_sha256,
            "candidate_sha256": self.candidate_sha256,
            "placement_state_sha256": self.placement_state_sha256,
            "guard_rings": [asdict(obligation) for obligation in self.guard_rings],
            "bands": [asdict(obligation) for obligation in self.bands],
            "power_planes": [
                asdict(obligation) for obligation in self.power_planes
            ],
            "kelvins": [asdict(obligation) for obligation in self.kelvins],
            "shields": [asdict(obligation) for obligation in self.shields],
            "adjacencies": [
                asdict(obligation) for obligation in self.adjacencies
            ],
            "anchors": [asdict(obligation) for obligation in self.anchors],
            "rail_affinities": [
                asdict(obligation) for obligation in self.rail_affinities
            ],
            "channels": [asdict(obligation) for obligation in self.channels],
            "densities": [asdict(obligation) for obligation in self.densities],
            "functional_od_fills": [
                asdict(obligation) for obligation in self.functional_od_fills
            ],
            "finger_array_orders": [
                asdict(obligation) for obligation in self.finger_array_orders
            ],
            "passive_bank_partitions": [
                asdict(obligation) for obligation in self.passive_bank_partitions
            ],
            "well_island_shares": [
                asdict(obligation) for obligation in self.well_island_shares
            ],
            "common_centroids": [
                asdict(obligation) for obligation in self.common_centroids
            ],
            "interdigitations": [
                asdict(obligation) for obligation in self.interdigitations
            ],
            "group_route_ports": [
                asdict(obligation) for obligation in self.group_route_ports
            ],
            "reason": self.reason,
        }
        if self.fact_projections:
            data["fact_projections"] = [
                row.canonical_data() for row in self.fact_projections
            ]
        return data


def _positive_compact_budget(cell: object | None) -> bool:
    data = dict(getattr(cell, "sizing_kwargs", {}) or {})
    value = data.get("opt_layout_budget")
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _canonical_final_policy(
    authoring_cell: object | None, supplied_policy: object
) -> object:
    """Replay source bytes when available; constructed test IR keeps its input."""

    if authoring_cell is None or getattr(authoring_cell, "_clay_file", None) is None:
        return supplied_policy
    from clay.layout.policy import policy_for_cell
    from clay.layout.route.exact_artifact_evidence import (
        _replayed_source_cell_authority,
    )

    _source_sha, _inventory_sha, _classes, canonical_cell = (
        _replayed_source_cell_authority(authoring_cell)
    )
    return policy_for_cell(canonical_cell._clay_file, canonical_cell)


def declared_final_aspect_obligation(
    *,
    authoring_cell: object | None,
    effective_policy: object,
    compact_candidate: object | None = None,
) -> FinalAspectObligation | None:
    """Derive the strongest declared aspect obligation without circuit policy.

    A compact-search domain is the authoring contract and therefore takes
    precedence over one selected candidate's effective target.  Designs with
    no compact search fall back to their effective ``@place_policy(aspect=)``.
    """

    derived_fixed_aspect = (
        compact_candidate is not None
        and str(getattr(compact_candidate, "materialization_mode", "derived"))
        != "authoring"
        and str(getattr(compact_candidate, "aspect_mode", "natural")) != "natural"
    )
    if _positive_compact_budget(authoring_cell) and derived_fixed_aspect:
        domain = compact_intent_domain_spec(authoring_cell)
        return FinalAspectObligation(
            source="compact_search_domain",
            minimum=float(domain.aspect_bounds[0]),
            maximum=float(domain.aspect_bounds[1]),
            target=float(domain.aspect_nominal),
        )

    effective_policy = _canonical_final_policy(authoring_cell, effective_policy)
    raw_target = getattr(effective_policy, "aspect_target", None)
    declared, _policy_sha256 = aspect_policy_binding(raw_target)
    if not declared:
        return None
    target = float(raw_target)
    # A declared search domain is the admissible aspect interval even when the
    # canonical authoring candidate (rather than one derived fixed-aspect
    # candidate) is being signed.  The policy target remains the optimization
    # centre; it must not erase the source's explicit min/max contract and turn
    # the objective into an accidental +/-5% hard gate.
    if _positive_compact_budget(authoring_cell):
        domain = compact_intent_domain_spec(authoring_cell)
        return FinalAspectObligation(
            source="effective_place_policy",
            minimum=float(domain.aspect_bounds[0]),
            maximum=float(domain.aspect_bounds[1]),
            target=target,
        )
    tolerance = ASPECT_RELATIVE_TOLERANCE
    return FinalAspectObligation(
        source="effective_place_policy",
        minimum=target * (1.0 - tolerance),
        maximum=target * (1.0 + tolerance),
        target=target,
        relative_tolerance=tolerance,
    )


def _certificate(
    *,
    top_cell: str,
    gds_sha256: str | None,
    status: Literal["pass", "fail", "unmeasurable"],
    measured_bbox_um: tuple[float, float] | None,
    measured_aspect: float | None,
    aspect: FinalAspectObligation | None,
    reason: str,
    route_containment: RouteContainmentObligation | None = None,
    physical_macro_graph_sha256: str | None = None,
    candidate_sha256: str | None = None,
    placement_state_sha256: str | None = None,
    guard_rings: tuple[GdsGuardRingObligation, ...] = (),
    bands: tuple[GdsBandObligation, ...] = (),
    power_planes: tuple[GdsPowerPlaneObligation, ...] = (),
    kelvins: tuple[GdsKelvinObligation, ...] = (),
    shields: tuple[GdsShieldObligation, ...] = (),
    adjacencies: tuple[GdsAdjacentObligation, ...] = (),
    anchors: tuple[GdsAnchorObligation, ...] = (),
    rail_affinities: tuple[GdsRailAffinityObligation, ...] = (),
    channels: tuple[GdsChannelObligation, ...] = (),
    densities: tuple[GdsDensityWhitespaceObligation, ...] = (),
    functional_od_fills: tuple[GdsFunctionalOdFillObligation, ...] = (),
    finger_array_orders: tuple[GdsFingerArrayOrderObligation, ...] = (),
    passive_bank_partitions: tuple[GdsPassiveBankPartitionObligation, ...] = (),
    well_island_shares: tuple[GdsWellIslandShareObligation, ...] = (),
    common_centroids: tuple[GdsCommonCentroidObligation, ...] = (),
    interdigitations: tuple[GdsInterdigitationObligation, ...] = (),
    group_route_ports: tuple[GdsGroupRoutePortObligation, ...] = (),
) -> FinalLayoutIntentCertificate:
    return FinalLayoutIntentCertificate(
        schema=_SCHEMA,
        top_cell=top_cell,
        gds_sha256=gds_sha256,
        status=status,
        measured_bbox_um=measured_bbox_um,
        measured_aspect=measured_aspect,
        aspect=aspect,
        reason=reason,
        route_containment=route_containment,
        physical_macro_graph_sha256=physical_macro_graph_sha256,
        candidate_sha256=candidate_sha256,
        placement_state_sha256=placement_state_sha256,
        guard_rings=guard_rings,
        bands=bands,
        power_planes=power_planes,
        kelvins=kelvins,
        shields=shields,
        adjacencies=adjacencies,
        anchors=anchors,
        rail_affinities=rail_affinities,
        channels=channels,
        densities=densities,
        functional_od_fills=functional_od_fills,
        finger_array_orders=finger_array_orders,
        passive_bank_partitions=passive_bank_partitions,
        well_island_shares=well_island_shares,
        common_centroids=common_centroids,
        interdigitations=interdigitations,
        group_route_ports=group_route_ports,
    )


def _unmeasurable_containment(
    floorplan_bbox_nm: tuple[int, int, int, int] | None,
    reason: str,
) -> RouteContainmentObligation:
    return RouteContainmentObligation(
        status="unmeasurable",
        floorplan_bbox_nm=floorplan_bbox_nm,
        max_overflow_nm=None,
        offending_layers=(),
        offending_polygon_count=0,
        reason=reason,
    )


def _measure_route_containment(
    layout: object,
    cell: object,
    floorplan_bbox_nm: tuple[int, int, int, int],
) -> RouteContainmentObligation:
    """Measure every emitted polygon against one hard pre-route boundary.

    This follows the useful anaroute pattern without copying it: grid neighbors
    never cross the circuit boundary, and via landings are checked as their exact
    multi-layer shapes.  Here the independent artifact gate applies the same hard
    boundary to exact emitted polygons after all route/via materialization.
    """

    import klayout.db as kdb

    if (
        len(floorplan_bbox_nm) != 4
        or floorplan_bbox_nm[2] <= floorplan_bbox_nm[0]
        or floorplan_bbox_nm[3] <= floorplan_bbox_nm[1]
    ):
        return _unmeasurable_containment(
            floorplan_bbox_nm, "floorplan bbox is empty or unordered"
        )
    nm_per_dbu = float(getattr(layout, "dbu")) * 1000.0
    if not math.isfinite(nm_per_dbu) or nm_per_dbu <= 0.0:
        return _unmeasurable_containment(
            floorplan_bbox_nm, "layout DBU is non-positive or non-finite"
        )
    allowed = kdb.Box(*(int(round(value / nm_per_dbu)) for value in floorplan_bbox_nm))
    maximum_dbu = 0
    offending_count = 0
    offending_layers: set[str] = set()
    for layer_index in getattr(layout, "layer_indexes")():
        info = getattr(layout, "get_info")(layer_index)
        region = kdb.Region(getattr(cell, "begin_shapes_rec")(layer_index))
        for polygon in region.each():
            bbox = polygon.bbox()
            overflow = max(
                allowed.left - bbox.left,
                bbox.right - allowed.right,
                allowed.bottom - bbox.bottom,
                bbox.top - allowed.top,
                0,
            )
            if overflow <= 0:
                continue
            maximum_dbu = max(maximum_dbu, int(overflow))
            offending_count += 1
            offending_layers.add(f"{int(info.layer)}/{int(info.datatype)}")
    maximum_nm = float(maximum_dbu) * nm_per_dbu
    status: Literal["pass", "fail", "unmeasurable"] = (
        "fail" if offending_count else "pass"
    )
    return RouteContainmentObligation(
        status=status,
        floorplan_bbox_nm=tuple(int(value) for value in floorplan_bbox_nm),
        max_overflow_nm=maximum_nm,
        offending_layers=tuple(sorted(offending_layers)),
        offending_polygon_count=offending_count,
        reason=(
            f"{offending_count} emitted polygons exceed the floorplan by up to "
            f"{maximum_nm:.12g}nm"
            if offending_count
            else "all emitted polygons are inside the declared floorplan"
        ),
    )


def check_final_layout_intent(
    gds_path: str | Path,
    *,
    top_cell: str,
    authoring_cell: object | None,
    effective_policy: object,
    floorplan_bbox_nm: tuple[int, int, int, int] | None = None,
    require_route_containment: bool = False,
    physical_macro_graph_sha256: str | None = None,
    candidate_sha256: str | None = None,
    placement_state_sha256: str | None = None,
    compact_candidate: object | None = None,
    layout_intent: object | None = None,
    tech: object | None = None,
    emitted_device_geometry: Mapping[str, Any] | None = None,
    emitted_device_placement_um: Mapping[str, Any] | None = None,
    exact_artifact: object | None = None,
    emission_source_cell: object | None = None,
    guard_ring_member_classes: tuple[tuple[str, str], ...] = (),
    member_ring_pad_top_nm: tuple[tuple[str, int], ...] = (),
    member_ring_pad_bottom_nm: tuple[tuple[str, int], ...] = (),
    final_placement_certificate: object | None = None,
    row_program: object | None = None,
    row_materialization: object | None = None,
    row_final_placement_certificate: object | None = None,
    row_final_placement: object | None = None,
    rigid_nonmos_macro_plans: object | None = None,
    rigid_nonmos_macro_inputs: object | None = None,
    rigid_nonmos_macro_interfaces: object | None = None,
    rigid_nonmos_placement_certificate: object | None = None,
    rigid_nonmos_scope_token: object | None = None,
    rigid_nonmos_emission_certificate: object | None = None,
    pins_by_net: Mapping[str, object] | None = None,
    materialized_power_plane_facts: object = (),
    materialized_topology_plan: object | None = None,
) -> FinalLayoutIntentCertificate | None:
    """Measure declared final analog intent on the exact emitted GDS cell."""

    canonical_effective_policy = _canonical_final_policy(
        emission_source_cell if emission_source_cell is not None else authoring_cell,
        effective_policy,
    )
    obligation = declared_final_aspect_obligation(
        authoring_cell=authoring_cell,
        effective_policy=canonical_effective_policy,
        compact_candidate=compact_candidate,
    )
    if final_placement_certificate is not None:
        expected_aspect_obligation, expected_aspect_policy_sha256 = (
            aspect_policy_binding(
                getattr(canonical_effective_policy, "aspect_target", None)
            )
        )
        if (
            type(getattr(final_placement_certificate, "aspect_obligation", None))
            is not bool
            or final_placement_certificate.aspect_obligation
            is not expected_aspect_obligation
            or getattr(final_placement_certificate, "aspect_policy_sha256", None)
            != expected_aspect_policy_sha256
        ):
            raise ValueError(
                "final placement aspect policy differs from canonical source"
            )
    containment_declared = require_route_containment or floorplan_bbox_nm is not None
    declared_rings = tuple(getattr(layout_intent, "guard_rings", ()) or ())
    raw_bands = getattr(layout_intent, "bands", ())
    raw_power_planes = getattr(layout_intent, "power_planes", ())
    raw_kelvins = getattr(layout_intent, "kelvins", ())
    raw_shields = getattr(layout_intent, "shields", ())
    raw_adjacencies = getattr(layout_intent, "adjacencies", ())
    raw_anchors = getattr(layout_intent, "anchors", ())
    raw_rail_affinities = getattr(layout_intent, "rail_affinities", ())
    raw_channels = getattr(layout_intent, "channels", ())
    raw_densities = getattr(layout_intent, "densities", ())
    raw_od_fills = getattr(layout_intent, "od_fills", ())
    raw_arrays = getattr(layout_intent, "arrays", ())
    raw_common_centroids = getattr(layout_intent, "common_centroids", ())
    raw_matching_intents = getattr(layout_intent, "matching_intents", ())
    raw_passive_bank_partitions = getattr(
        layout_intent, "passive_bank_partitions", ()
    )
    raw_well_island_shares = getattr(layout_intent, "well_island_shares", ())
    raw_place_groups = getattr(layout_intent, "place_groups", ())
    if type(raw_bands) is not tuple:
        raise TypeError("layout intent bands must be an exact tuple")
    from clay.layout.intent import (
        replay_kelvin_inventory,
        replay_adjacent_inventory,
        replay_power_plane_inventory,
        replay_shield_inventory,
    )

    raw_power_planes = replay_power_plane_inventory(raw_power_planes)
    raw_kelvins = replay_kelvin_inventory(raw_kelvins)
    raw_shields = replay_shield_inventory(raw_shields)
    raw_adjacencies = replay_adjacent_inventory(raw_adjacencies)
    if type(raw_anchors) is not tuple:
        raise TypeError("layout intent anchors must be an exact tuple")
    if type(raw_channels) is not tuple:
        raise TypeError("layout intent channels must be an exact tuple")
    if type(raw_densities) is not tuple:
        raise TypeError("layout intent densities must be an exact tuple")
    if type(raw_od_fills) is not tuple:
        raise TypeError("layout intent od_fills must be an exact tuple")
    if type(raw_arrays) is not tuple:
        raise TypeError("layout intent arrays must be an exact tuple")
    if type(raw_common_centroids) is not tuple:
        raise TypeError("layout intent common_centroids must be an exact tuple")
    if type(raw_matching_intents) is not tuple:
        raise TypeError("layout intent matching_intents must be an exact tuple")
    if type(raw_passive_bank_partitions) is not tuple:
        raise TypeError(
            "layout intent passive_bank_partitions must be an exact tuple"
        )
    if type(raw_well_island_shares) is not tuple:
        raise TypeError("layout intent well_island_shares must be an exact tuple")
    if raw_arrays:
        from clay.layout.intent import ArrayIntent, replay_array_intent

        if any(type(declaration) is not ArrayIntent for declaration in raw_arrays):
            raise TypeError(
                "layout intent arrays must contain exact ArrayIntent values"
            )
        raw_arrays = tuple(
            replay_array_intent(declaration) for declaration in raw_arrays
        )
        array_sha256s = tuple(declaration.sha256 for declaration in raw_arrays)
        if len(set(array_sha256s)) != len(array_sha256s):
            raise ValueError("duplicate array declarations are forbidden")
    if raw_passive_bank_partitions:
        from clay.layout.intent import (
            PassiveBankPartitionIntent,
            replay_passive_bank_partition_intent,
        )

        if any(
            type(declaration) is not PassiveBankPartitionIntent
            for declaration in raw_passive_bank_partitions
        ):
            raise TypeError(
                "layout intent passive_bank_partitions must contain exact "
                "PassiveBankPartitionIntent values"
            )
        raw_passive_bank_partitions = tuple(
            replay_passive_bank_partition_intent(declaration)
            for declaration in raw_passive_bank_partitions
        )
        if raw_passive_bank_partitions != tuple(
            sorted(
                raw_passive_bank_partitions,
                key=lambda item: (item.parent, item.bank),
            )
        ):
            raise ValueError(
                "layout intent passive_bank_partitions must be in canonical order"
            )
        partition_keys = tuple(
            (declaration.parent, declaration.bank)
            for declaration in raw_passive_bank_partitions
        )
        if len(set(partition_keys)) != len(partition_keys):
            raise ValueError(
                "duplicate passive bank partition declarations are forbidden"
            )
    if raw_well_island_shares:
        from clay.layout.intent import replay_well_island_share_intent

        raw_well_island_shares = tuple(
            replay_well_island_share_intent(declaration)
            for declaration in raw_well_island_shares
        )
        if raw_well_island_shares != tuple(
            sorted(raw_well_island_shares, key=lambda item: item.group)
        ):
            raise ValueError(
                "layout intent well_island_shares must be in canonical order"
            )
        if len({item.group for item in raw_well_island_shares}) != len(
            raw_well_island_shares
        ):
            raise ValueError("duplicate well island share declarations are forbidden")
    if raw_common_centroids:
        from clay.layout.intent import replay_common_centroid_intent

        raw_common_centroids = tuple(
            replay_common_centroid_intent(declaration)
            for declaration in raw_common_centroids
        )
        if len({item.sha256 for item in raw_common_centroids}) != len(
            raw_common_centroids
        ):
            raise ValueError("duplicate common-centroid declarations are forbidden")
    if raw_matching_intents:
        from clay.layout.intent import replay_matching_intent

        raw_matching_intents = tuple(
            replay_matching_intent(declaration)
            for declaration in raw_matching_intents
        )
        if len({item.sha256 for item in raw_matching_intents}) != len(
            raw_matching_intents
        ):
            raise ValueError("duplicate matching declarations are forbidden")
    if raw_channels:
        from clay.layout.intent import ChannelIntent, replay_channel_intent

        if any(type(declaration) is not ChannelIntent for declaration in raw_channels):
            raise TypeError(
                "layout intent channels must contain exact ChannelIntent values"
            )
        raw_channels = tuple(
            replay_channel_intent(declaration) for declaration in raw_channels
        )
        channel_sha256s = tuple(declaration.sha256 for declaration in raw_channels)
        if len(set(channel_sha256s)) != len(channel_sha256s):
            raise ValueError("duplicate channel declarations are forbidden")
    if raw_densities:
        from clay.layout.intent import DensityIntent, replay_density_intent

        if len(raw_densities) != 1 or any(
            type(declaration) is not DensityIntent
            for declaration in raw_densities
        ):
            raise TypeError(
                "layout intent densities must contain one exact DensityIntent"
            )
        raw_densities = tuple(
            replay_density_intent(declaration) for declaration in raw_densities
        )
    if raw_od_fills:
        from clay.layout.intent import OdFillIntent, replay_od_fill_intent

        if len(raw_od_fills) != 1 or any(
            type(declaration) is not OdFillIntent for declaration in raw_od_fills
        ):
            raise TypeError(
                "layout intent od_fills must contain one exact OdFillIntent"
            )
        raw_od_fills = tuple(
            replay_od_fill_intent(declaration) for declaration in raw_od_fills
        )
    if raw_anchors:
        from clay.layout.intent import AnchorIntent

        if any(type(declaration) is not AnchorIntent for declaration in raw_anchors):
            raise TypeError("layout intent anchors must contain exact AnchorIntent values")
        anchor_sha256s = tuple(declaration.sha256 for declaration in raw_anchors)
        if len(set(anchor_sha256s)) != len(anchor_sha256s):
            raise ValueError("duplicate anchor declarations are forbidden")
    from clay.layout.intent import replay_rail_affinity_inventory

    raw_rail_affinities = replay_rail_affinity_inventory(raw_rail_affinities)
    if (
        raw_bands or raw_anchors or raw_rail_affinities
    ) and type(raw_place_groups) is not tuple:
        raise TypeError("layout intent place_groups must be an exact tuple")
    declared_bands = raw_bands
    declared_power_planes = raw_power_planes
    declared_kelvins = raw_kelvins
    declared_shields = raw_shields
    declared_adjacencies = raw_adjacencies
    declared_anchors = raw_anchors
    declared_rail_affinities = raw_rail_affinities
    declared_channels = raw_channels
    declared_densities = raw_densities
    declared_od_fills = raw_od_fills
    canonical_channel_strategy = str(
        getattr(canonical_effective_policy, "channel_strategy", "declared")
    )
    if canonical_channel_strategy not in {"declared", "block_gap_only"}:
        raise ValueError("final layout has an unsupported channel strategy")
    if canonical_channel_strategy == "block_gap_only" and compact_candidate is None:
        raise ValueError("block_gap_only requires compact candidate authority")
    if compact_candidate is not None and (
        compact_candidate.channel_strategy != canonical_channel_strategy
    ):
        raise ValueError(
            "final layout channel strategy differs from compact candidate"
        )
    if canonical_channel_strategy == "block_gap_only":
        if (
            final_placement_certificate is None
            or type(
                getattr(final_placement_certificate, "declared_channel_nm", None)
            )
            is not int
            or final_placement_certificate.declared_channel_nm != 0
        ):
            raise ValueError(
                "block_gap_only requires a zero-corridor final placement certificate"
            )
        declared_channels = ()
    declared_arrays = raw_arrays
    source_native_arrays = tuple(item for item in raw_arrays if item.style == 'native_diffusion')
    native_array_inventory_error: str | None = None
    declared_passive_bank_partitions = raw_passive_bank_partitions
    declared_well_island_shares = raw_well_island_shares
    declared_common_centroids = raw_common_centroids
    declared_matching_intents = raw_matching_intents
    passive_bank_inventory_error: str | None = None
    well_island_inventory_error: str | None = None
    centroid_inventory_error: str | None = None
    density_inventory_error: str | None = None
    od_fill_inventory_error: str | None = None
    rail_affinity_inventory_error: str | None = None
    power_plane_inventory_error: str | None = None
    kelvin_inventory_error: str | None = None
    shield_inventory_error: str | None = None
    adjacency_inventory_error: str | None = None
    source_for_replay = (
        emission_source_cell
        if emission_source_cell is not None
        else authoring_cell
    )
    if source_for_replay is not None:
        try:
            from clay.layout.intent import (
                layout_intent_from_cell_def,
                replay_array_intent,
                replay_common_centroid_intent,
                replay_density_intent,
                replay_matching_intent,
                replay_adjacent_inventory,
                replay_kelvin_inventory,
                replay_od_fill_intent,
                replay_passive_bank_partition_intent,
                replay_power_plane_inventory,
                replay_rail_affinity_inventory,
                replay_shield_inventory,
                replay_well_island_share_intent,
            )
            from clay.layout.route.exact_artifact_evidence import (
                _replayed_source_cell_authority,
            )

            (
                replayed_source_sha256,
                replayed_inventory_sha256,
                _member_classes,
                canonical_source_cell,
            ) = _replayed_source_cell_authority(source_for_replay)
            if exact_artifact is not None:
                scope = getattr(exact_artifact, "scope", None)
                if (
                    replayed_source_sha256
                    != getattr(scope, "source_cell_source_sha256", None)
                    or replayed_inventory_sha256
                    != getattr(scope, "source_member_inventory_sha256", None)
                ):
                    raise ValueError(
                        "exact artifact source scope differs from canonical source"
                    )
            canonical_layout_intent = layout_intent_from_cell_def(
                canonical_source_cell
            )
            canonical_native_arrays = tuple(
                replay_array_intent(item) for item in canonical_layout_intent.arrays
                if item.style == 'native_diffusion'
            )
            if canonical_native_arrays != source_native_arrays:
                native_array_inventory_error = 'native diffusion array inventory differs from canonical source bytes'
                source_native_arrays = canonical_native_arrays or source_native_arrays
            canonical_power_planes = replay_power_plane_inventory(
                canonical_layout_intent.power_planes
            )
            declared_power_planes = canonical_power_planes or raw_power_planes
            if canonical_power_planes != raw_power_planes:
                power_plane_inventory_error = (
                    "power-plane inventory differs from canonical source bytes"
                )
            canonical_kelvins = replay_kelvin_inventory(
                canonical_layout_intent.kelvins
            )
            declared_kelvins = canonical_kelvins or raw_kelvins
            if canonical_kelvins != raw_kelvins:
                kelvin_inventory_error = (
                    "Kelvin inventory differs from canonical source bytes"
                )
            canonical_shields = replay_shield_inventory(
                canonical_layout_intent.shields
            )
            declared_shields = canonical_shields or raw_shields
            if canonical_shields != raw_shields:
                shield_inventory_error = (
                    "shield inventory differs from canonical source bytes"
                )
            canonical_adjacencies = replay_adjacent_inventory(
                canonical_layout_intent.adjacencies
            )
            declared_adjacencies = canonical_adjacencies or raw_adjacencies
            if canonical_adjacencies != raw_adjacencies:
                adjacency_inventory_error = (
                    "adjacent inventory differs from canonical source bytes"
                )
            canonical_densities = tuple(
                replay_density_intent(item)
                for item in canonical_layout_intent.densities
            )
            declared_densities = canonical_densities
            if canonical_densities != raw_densities:
                density_inventory_error = (
                    "density inventory differs from canonical source bytes"
                )
            canonical_od_fills = tuple(
                replay_od_fill_intent(item)
                for item in canonical_layout_intent.od_fills
            )
            declared_od_fills = canonical_od_fills
            if canonical_od_fills != raw_od_fills:
                od_fill_inventory_error = (
                    "functional OD inventory differs from canonical source bytes"
                )
            canonical_rail_affinities = replay_rail_affinity_inventory(
                canonical_layout_intent.rail_affinities
            )
            declared_rail_affinities = canonical_rail_affinities
            if canonical_rail_affinities != raw_rail_affinities:
                rail_affinity_inventory_error = (
                    "rail affinity inventory differs from canonical source bytes"
                )
            canonical_partition_intents = tuple(
                replay_passive_bank_partition_intent(item)
                for item in canonical_layout_intent.passive_bank_partitions
            )
            declared_passive_bank_partitions = canonical_partition_intents
            if canonical_partition_intents != raw_passive_bank_partitions:
                passive_bank_inventory_error = (
                    "passive bank partition inventory differs from canonical "
                    "source bytes"
                )
            canonical_well_island_shares = tuple(
                replay_well_island_share_intent(item)
                for item in canonical_layout_intent.well_island_shares
            )
            declared_well_island_shares = canonical_well_island_shares
            if canonical_well_island_shares != raw_well_island_shares:
                well_island_inventory_error = (
                    "well island share inventory differs from canonical source bytes"
                )
            canonical_common_centroids = tuple(
                replay_common_centroid_intent(item)
                for item in canonical_layout_intent.common_centroids
            )
            canonical_matching_intents = tuple(
                replay_matching_intent(item)
                for item in canonical_layout_intent.matching_intents
            )
            declared_common_centroids = canonical_common_centroids
            declared_matching_intents = canonical_matching_intents
            if (
                canonical_common_centroids != raw_common_centroids
                or canonical_matching_intents != raw_matching_intents
            ):
                centroid_inventory_error = (
                    "common-centroid/interdigitation inventory differs from "
                    "canonical source bytes"
                )
        except Exception as exc:  # noqa: BLE001 - typed source authority boundary
            fallback_partition_intents = ()
            fallback_densities = ()
            fallback_od_fills = ()
            fallback_rail_affinities = ()
            fallback_power_planes = ()
            fallback_kelvins = ()
            fallback_shields = ()
            fallback_adjacencies = ()
            fallback_common_centroids = ()
            fallback_matching_intents = ()
            fallback_well_island_shares = ()
            fallback_native_arrays = ()
            try:
                from clay.layout.intent import (
                    layout_intent_from_cell_def,
                    replay_array_intent,
                    replay_common_centroid_intent,
                    replay_density_intent,
                    replay_matching_intent,
                    replay_adjacent_inventory,
                    replay_kelvin_inventory,
                    replay_od_fill_intent,
                    replay_passive_bank_partition_intent,
                    replay_power_plane_inventory,
                    replay_rail_affinity_inventory,
                    replay_shield_inventory,
                    replay_well_island_share_intent,
                )

                fallback_layout_intent = layout_intent_from_cell_def(
                    source_for_replay
                )
                fallback_native_arrays = tuple(
                    replay_array_intent(item) for item in fallback_layout_intent.arrays
                    if item.style == 'native_diffusion'
                )
                fallback_partition_intents = tuple(
                    replay_passive_bank_partition_intent(item)
                    for item in fallback_layout_intent.passive_bank_partitions
                )
                fallback_densities = tuple(
                    replay_density_intent(item)
                    for item in fallback_layout_intent.densities
                )
                fallback_od_fills = tuple(
                    replay_od_fill_intent(item)
                    for item in fallback_layout_intent.od_fills
                )
                fallback_rail_affinities = replay_rail_affinity_inventory(
                    fallback_layout_intent.rail_affinities
                )
                fallback_power_planes = replay_power_plane_inventory(
                    fallback_layout_intent.power_planes
                )
                fallback_kelvins = replay_kelvin_inventory(
                    fallback_layout_intent.kelvins
                )
                fallback_shields = replay_shield_inventory(
                    fallback_layout_intent.shields
                )
                fallback_adjacencies = replay_adjacent_inventory(
                    fallback_layout_intent.adjacencies
                )
                fallback_common_centroids = tuple(
                    replay_common_centroid_intent(item)
                    for item in fallback_layout_intent.common_centroids
                )
                fallback_matching_intents = tuple(
                    replay_matching_intent(item)
                    for item in fallback_layout_intent.matching_intents
                )
                fallback_well_island_shares = tuple(
                    replay_well_island_share_intent(item)
                    for item in fallback_layout_intent.well_island_shares
                )
            except Exception:
                fallback_partition_intents = ()
                fallback_densities = ()
                fallback_od_fills = ()
                fallback_rail_affinities = ()
                fallback_power_planes = ()
                fallback_kelvins = ()
                fallback_shields = ()
                fallback_adjacencies = ()
                fallback_common_centroids = ()
                fallback_matching_intents = ()
                fallback_well_island_shares = ()
                fallback_native_arrays = ()
            if fallback_native_arrays or source_native_arrays:
                source_native_arrays = fallback_native_arrays or source_native_arrays
                native_array_inventory_error = (
                    'native diffusion canonical source inventory cannot be replayed: '
                    f'{type(exc).__name__}: {exc}'
                )
            if fallback_densities:
                declared_densities = fallback_densities
                density_inventory_error = (
                    "density canonical source bytes cannot be replayed; "
                    "the parsed source declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_densities:
                density_inventory_error = (
                    "density canonical source inventory cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if fallback_od_fills:
                declared_od_fills = fallback_od_fills
                od_fill_inventory_error = (
                    "functional OD canonical source bytes cannot be replayed; "
                    "the parsed source declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_od_fills:
                od_fill_inventory_error = (
                    "functional OD canonical source inventory cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if fallback_rail_affinities:
                declared_rail_affinities = fallback_rail_affinities
                rail_affinity_inventory_error = (
                    "rail affinity canonical source bytes cannot be replayed; "
                    "the parsed declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_rail_affinities:
                rail_affinity_inventory_error = (
                    "rail affinity canonical source inventory cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if fallback_power_planes:
                declared_power_planes = fallback_power_planes
                power_plane_inventory_error = (
                    "power-plane canonical source bytes cannot be replayed; "
                    "the parsed declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_power_planes:
                power_plane_inventory_error = (
                    "power-plane canonical source inventory cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if fallback_kelvins:
                declared_kelvins = fallback_kelvins
                kelvin_inventory_error = (
                    "Kelvin canonical source bytes cannot be replayed; the parsed "
                    "declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_kelvins:
                kelvin_inventory_error = (
                    "Kelvin canonical source inventory cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if fallback_shields:
                declared_shields = fallback_shields
                shield_inventory_error = (
                    "shield canonical source bytes cannot be replayed; the parsed "
                    "declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_shields:
                shield_inventory_error = (
                    "shield canonical source inventory cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if fallback_adjacencies:
                declared_adjacencies = fallback_adjacencies
                adjacency_inventory_error = (
                    "adjacent canonical source bytes cannot be replayed; the parsed "
                    "declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_adjacencies:
                adjacency_inventory_error = (
                    "adjacent canonical source inventory cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if fallback_partition_intents:
                declared_passive_bank_partitions = fallback_partition_intents
                passive_bank_inventory_error = (
                    "passive bank canonical source bytes cannot be replayed; "
                    "the parsed source declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_passive_bank_partitions:
                passive_bank_inventory_error = (
                    "passive bank canonical source inventory cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if fallback_well_island_shares:
                declared_well_island_shares = fallback_well_island_shares
                well_island_inventory_error = (
                    "well island canonical source bytes cannot be replayed; "
                    "the parsed source declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_well_island_shares:
                well_island_inventory_error = (
                    "well island canonical source inventory cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
            if fallback_common_centroids or any(
                item.requires_interdigitation
                for item in fallback_matching_intents
            ):
                declared_common_centroids = fallback_common_centroids
                declared_matching_intents = fallback_matching_intents
                centroid_inventory_error = (
                    "centroid/interdigitation canonical source bytes cannot be "
                    "replayed; the parsed declaration denominator remains blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
            elif raw_common_centroids or any(
                item.requires_interdigitation for item in raw_matching_intents
            ):
                centroid_inventory_error = (
                    "centroid/interdigitation canonical source inventory cannot be "
                    f"replayed: {type(exc).__name__}: {exc}"
                )
    declared_ratio_arrays = tuple(
        declaration
        for declaration in declared_arrays
        if declaration.style == "shared_diffusion"
        and declaration.order == "ratio"
    )
    # Native diffusion uses a source-owned mosaic, but no final-GDS checker
    # currently proves the emitted channel order against that source. Keep a
    # distinct obligation rather than treating the placement inventory as GDS
    # measurement or dropping the declaration from the final denominator.
    declared_native_arrays = source_native_arrays
    declared_finger_arrays = declared_ratio_arrays + declared_native_arrays
    declared_interdigitations = tuple(
        declaration
        for declaration in declared_matching_intents
        if declaration.requires_interdigitation
    )

    def unmeasurable_bands(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsBandObligation, ...]:
        return tuple(
            GdsBandObligation(
                top=str(declaration.top),
                bottom=str(declaration.bottom),
                center=str(declaration.center),
                declaration_sha256=str(declaration.sha256),
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_bands
        )

    def unmeasurable_power_planes(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsPowerPlaneObligation, ...]:
        return tuple(
            GdsPowerPlaneObligation(
                net=declaration.net,
                layer_key=None,
                bbox_nm=None,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_power_planes
        )

    def unmeasurable_kelvins(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsKelvinObligation, ...]:
        return tuple(
            GdsKelvinObligation(
                force_net=declaration.force_net,
                sense_net=declaration.sense_net,
                target_device=declaration.target_device,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                force_target_covered=False,
                sense_target_covered=False,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_kelvins
        )

    def unmeasurable_shields(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsShieldObligation, ...]:
        return tuple(
            GdsShieldObligation(
                aggressor_net=declaration.aggressor,
                victim_net=declaration.victim,
                shield_net=declaration.shield_net,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                fence_layer_key=None,
                fence_endpoints_nm=None,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_shields
        )

    def unmeasurable_adjacencies(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsAdjacentObligation, ...]:
        return tuple(
            GdsAdjacentObligation(
                source=declaration.source,
                target=declaration.target,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                maximum_separation_nm=None,
                measured_separation_nm=None,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_adjacencies
        )

    def unmeasurable_anchors(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsAnchorObligation, ...]:
        return tuple(
            GdsAnchorObligation(
                target=declaration.target,
                edge=declaration.edge,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_anchors
        )

    def unmeasurable_rail_affinities(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsRailAffinityObligation, ...]:
        return tuple(
            GdsRailAffinityObligation(
                group=declaration.group,
                side=declaration.side,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_rail_affinities
        )

    def unmeasurable_channels(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsChannelObligation, ...]:
        return tuple(
            GdsChannelObligation(
                orient=declaration.orient,
                declared_width_nm=int(round(declaration.width_um * 1000.0)),
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_channels
        )

    def unmeasurable_densities(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsDensityWhitespaceObligation, ...]:
        return tuple(
            GdsDensityWhitespaceObligation(
                max_whitespace_fraction=declaration.max_whitespace_fraction,
                scope=declaration.scope,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_densities
        )

    def unmeasurable_functional_od_fills(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsFunctionalOdFillObligation, ...]:
        return tuple(
            GdsFunctionalOdFillObligation(
                min_fraction=declaration.min_fraction,
                scope=declaration.scope,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_od_fills
        )

    def unmeasurable_finger_arrays(
        reason: str, *, gds_sha256: str | None,
        declarations: tuple = declared_finger_arrays,
    ) -> tuple[GdsFingerArrayOrderObligation, ...]:
        return tuple(
            GdsFingerArrayOrderObligation(
                group=declaration.group,
                style=declaration.style,
                rows=declaration.rows,
                order=declaration.order,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declarations
        )

    def unmeasurable_passive_bank_partitions(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsPassiveBankPartitionObligation, ...]:
        return tuple(
            GdsPassiveBankPartitionObligation(
                parent=declaration.parent,
                bank=declaration.bank,
                members=declaration.members,
                macro_id=(
                    f"passive_bank:{declaration.parent}:{declaration.bank}"
                ),
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_passive_bank_partitions
        )

    def unmeasurable_well_island_shares(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsWellIslandShareObligation, ...]:
        return tuple(
            GdsWellIslandShareObligation(
                group=declaration.group,
                body=declaration.body,
                well="",
                members=(),
                layer_key=None,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_well_island_shares
        )

    def unmeasurable_common_centroids(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsCommonCentroidObligation, ...]:
        return tuple(
            GdsCommonCentroidObligation(
                group=declaration.group,
                members=("", ""),
                rows=declaration.rows,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_common_centroids
        )

    def unmeasurable_interdigitations(
        reason: str, *, gds_sha256: str | None
    ) -> tuple[GdsInterdigitationObligation, ...]:
        return tuple(
            GdsInterdigitationObligation(
                members=declaration.members,
                declaration_ids=declaration.interdigitation_declaration_ids,
                pattern=declaration.interdigitation_pattern,
                finger_pairs=declaration.interdigitation_finger_pairs,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declared_interdigitations
        )

    if (
        obligation is None
        and not containment_declared
        and not declared_rings
        and not declared_bands
        and not declared_power_planes
        and not declared_kelvins
        and not declared_shields
        and not declared_adjacencies
        and not declared_anchors
        and not declared_rail_affinities
        and not declared_channels
        and not declared_densities
        and not declared_od_fills
        and not declared_finger_arrays
        and not declared_passive_bank_partitions
        and not declared_well_island_shares
        and not declared_common_centroids
        and not declared_interdigitations
    ):
        return None

    path = Path(gds_path)
    if not path.is_file():
        containment = (
            _unmeasurable_containment(
                floorplan_bbox_nm,
                f"emitted GDS does not exist: {path}",
            )
            if containment_declared
            else None
        )
        return _certificate(
            top_cell=top_cell,
            gds_sha256=None,
            status="unmeasurable",
            measured_bbox_um=None,
            measured_aspect=None,
            aspect=obligation,
            reason=f"emitted GDS does not exist: {path}",
            route_containment=containment,
            physical_macro_graph_sha256=physical_macro_graph_sha256,
            candidate_sha256=candidate_sha256,
            placement_state_sha256=placement_state_sha256,
            guard_rings=tuple(
                GdsGuardRingObligation(
                    members=tuple(str(member) for member in declaration.members),
                    well=str(declaration.well),
                    tie_net=str(declaration.tie_net),
                    hug=str(getattr(declaration, "hug", "") or ""),
                    role=str(getattr(declaration, "role", "") or ""),
                    declaration_sha256=str(declaration.sha256),
                    gds_sha256=None,
                    status="unmeasurable",
                    measured_on_gds=False,
                    reason=f"emitted GDS does not exist: {path}",
                )
                for declaration in declared_rings
            ),
            bands=unmeasurable_bands(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            power_planes=unmeasurable_power_planes(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            kelvins=unmeasurable_kelvins(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            shields=unmeasurable_shields(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            adjacencies=unmeasurable_adjacencies(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            anchors=unmeasurable_anchors(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            rail_affinities=unmeasurable_rail_affinities(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            channels=unmeasurable_channels(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            densities=unmeasurable_densities(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            functional_od_fills=unmeasurable_functional_od_fills(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            finger_array_orders=unmeasurable_finger_arrays(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            passive_bank_partitions=unmeasurable_passive_bank_partitions(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            well_island_shares=unmeasurable_well_island_shares(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            common_centroids=unmeasurable_common_centroids(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
            interdigitations=unmeasurable_interdigitations(
                f"emitted GDS does not exist: {path}", gds_sha256=None
            ),
        )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ring_obligations: tuple[GdsGuardRingObligation, ...] = ()
    band_obligations: tuple[GdsBandObligation, ...] = ()
    power_plane_obligations: tuple[GdsPowerPlaneObligation, ...] = ()
    kelvin_obligations: tuple[GdsKelvinObligation, ...] = ()
    shield_obligations: tuple[GdsShieldObligation, ...] = ()
    adjacency_obligations: tuple[GdsAdjacentObligation, ...] = ()
    anchor_obligations: tuple[GdsAnchorObligation, ...] = ()
    rail_affinity_obligations: tuple[GdsRailAffinityObligation, ...] = ()
    channel_obligations: tuple[GdsChannelObligation, ...] = ()
    density_obligations: tuple[GdsDensityWhitespaceObligation, ...] = ()
    functional_od_obligations: tuple[GdsFunctionalOdFillObligation, ...] = ()
    finger_array_obligations: tuple[GdsFingerArrayOrderObligation, ...] = ()
    passive_bank_partition_obligations: tuple[
        GdsPassiveBankPartitionObligation, ...
    ] = ()
    well_island_share_obligations: tuple[GdsWellIslandShareObligation, ...] = ()
    common_centroid_obligations: tuple[GdsCommonCentroidObligation, ...] = ()
    interdigitation_obligations: tuple[GdsInterdigitationObligation, ...] = ()
    if declared_rings:
        from clay.layout.completeness.gds_invariants import verify_guard_rings_on_gds

        ring_obligations = verify_guard_rings_on_gds(
            str(path),
            top_cell=top_cell,
            guard_ring_intents=declared_rings,
            tech=tech,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            member_classes=guard_ring_member_classes,
            member_ring_pad_top_nm=member_ring_pad_top_nm,
            member_ring_pad_bottom_nm=member_ring_pad_bottom_nm,
            final_placement_certificate=final_placement_certificate,
            placement_state_sha256=placement_state_sha256,
        )
        expected_inventory = tuple(
            str(declaration.sha256) for declaration in declared_rings
        )
        observed_inventory = tuple(
            str(item.declaration_sha256) for item in ring_obligations
        )
        if observed_inventory != expected_inventory:
            ring_obligations = tuple(
                GdsGuardRingObligation(
                    members=tuple(str(member) for member in declaration.members),
                    well=str(declaration.well),
                    tie_net=str(declaration.tie_net),
                    hug=str(getattr(declaration, "hug", "") or ""),
                    role=str(getattr(declaration, "role", "") or ""),
                    declaration_sha256=str(declaration.sha256),
                    gds_sha256=digest,
                    status="unmeasurable",
                    measured_on_gds=False,
                    reason=(
                        "exact guard-ring obligation inventory mismatch: "
                        f"expected={expected_inventory}, observed={observed_inventory}"
                    ),
                )
                for declaration in declared_rings
            )
    if declared_power_planes and power_plane_inventory_error is not None:
        power_plane_obligations = unmeasurable_power_planes(
            power_plane_inventory_error,
            gds_sha256=digest,
        )
    elif declared_power_planes:
        from clay.layout.completeness.power_plane_check import (
            verify_power_planes_on_gds,
        )

        power_plane_obligations = verify_power_planes_on_gds(
            str(path),
            top_cell=top_cell,
            power_plane_intents=declared_power_planes,
            materialized_power_plane_facts=materialized_power_plane_facts,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            final_placement_certificate=final_placement_certificate,
            placement_state_sha256=placement_state_sha256,
            tech=tech,
        )
        expected_inventory = tuple(
            item.sha256 for item in declared_power_planes
        )
        observed_inventory = tuple(
            item.declaration_sha256 for item in power_plane_obligations
        )
        if observed_inventory != expected_inventory:
            power_plane_obligations = unmeasurable_power_planes(
                "exact power-plane obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_kelvins and kelvin_inventory_error is not None:
        kelvin_obligations = unmeasurable_kelvins(
            kelvin_inventory_error, gds_sha256=digest
        )
    elif declared_kelvins:
        from clay.layout.completeness.kelvin_check import verify_kelvins_on_gds

        kelvin_obligations = verify_kelvins_on_gds(
            str(path),
            top_cell=top_cell,
            kelvin_intents=declared_kelvins,
            materialized_topology_plan=materialized_topology_plan,
            pins_by_net=pins_by_net,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            tech=tech,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
        )
        expected_inventory = tuple(item.sha256 for item in declared_kelvins)
        observed_inventory = tuple(
            item.declaration_sha256 for item in kelvin_obligations
        )
        if observed_inventory != expected_inventory:
            kelvin_obligations = unmeasurable_kelvins(
                "exact Kelvin obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_shields and shield_inventory_error is not None:
        shield_obligations = unmeasurable_shields(
            shield_inventory_error, gds_sha256=digest
        )
    elif declared_shields:
        from clay.layout.completeness.shield_check import verify_shields_on_gds

        shield_obligations = verify_shields_on_gds(
            str(path),
            top_cell=top_cell,
            shield_intents=declared_shields,
            pins_by_net=pins_by_net,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            tech=tech,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            row_final_placement=row_final_placement,
        )
        expected_inventory = tuple(item.sha256 for item in declared_shields)
        observed_inventory = tuple(
            item.declaration_sha256 for item in shield_obligations
        )
        if observed_inventory != expected_inventory:
            shield_obligations = unmeasurable_shields(
                "exact shield obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_adjacencies and adjacency_inventory_error is not None:
        adjacency_obligations = unmeasurable_adjacencies(
            adjacency_inventory_error, gds_sha256=digest
        )
    elif declared_adjacencies:
        from clay.layout.completeness.adjacent_check import (
            verify_adjacencies_on_gds,
        )

        adjacency_obligations = verify_adjacencies_on_gds(
            str(path),
            top_cell=top_cell,
            adjacent_intents=declared_adjacencies,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            tech=tech,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            final_placement_certificate=final_placement_certificate,
            placement_state_sha256=placement_state_sha256,
        )
        expected_inventory = tuple(
            item.sha256 for item in declared_adjacencies
        )
        observed_inventory = tuple(
            item.declaration_sha256 for item in adjacency_obligations
        )
        if observed_inventory != expected_inventory:
            adjacency_obligations = unmeasurable_adjacencies(
                "exact adjacent obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_bands:
        from clay.layout.completeness.gds_invariants import verify_bands_on_gds

        band_obligations = verify_bands_on_gds(
            str(path),
            top_cell=top_cell,
            band_intents=declared_bands,
            place_groups=tuple(getattr(layout_intent, "place_groups", ()) or ()),
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
        )
        expected_inventory = tuple(
            str(declaration.sha256) for declaration in declared_bands
        )
        observed_inventory = tuple(
            str(item.declaration_sha256) for item in band_obligations
        )
        if observed_inventory != expected_inventory:
            band_obligations = unmeasurable_bands(
                "exact band obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_anchors:
        from clay.layout.completeness.gds_invariants import verify_anchors_on_gds

        anchor_obligations = verify_anchors_on_gds(
            str(path),
            top_cell=top_cell,
            anchor_intents=declared_anchors,
            place_groups=raw_place_groups,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            row_final_placement=row_final_placement,
            tech=tech,
        )
        expected_inventory = tuple(
            declaration.sha256 for declaration in declared_anchors
        )
        observed_inventory = tuple(
            item.declaration_sha256 for item in anchor_obligations
        )
        if observed_inventory != expected_inventory:
            anchor_obligations = unmeasurable_anchors(
                "exact anchor obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_rail_affinities and rail_affinity_inventory_error is not None:
        rail_affinity_obligations = unmeasurable_rail_affinities(
            rail_affinity_inventory_error,
            gds_sha256=digest,
        )
    elif declared_rail_affinities:
        from clay.layout.completeness.gds_invariants import (
            verify_rail_affinities_on_gds,
        )

        rail_affinity_obligations = verify_rail_affinities_on_gds(
            str(path),
            top_cell=top_cell,
            rail_affinity_intents=declared_rail_affinities,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            final_placement_certificate=final_placement_certificate,
            row_final_placement=row_final_placement,
            placement_state_sha256=placement_state_sha256,
            tech=tech,
        )
        expected_inventory = tuple(
            declaration.sha256 for declaration in declared_rail_affinities
        )
        observed_inventory = tuple(
            item.declaration_sha256 for item in rail_affinity_obligations
        )
        if observed_inventory != expected_inventory:
            rail_affinity_obligations = unmeasurable_rail_affinities(
                "exact rail affinity obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_channels:
        from clay.layout.completeness.gds_invariants import verify_channels_on_gds

        channel_obligations = verify_channels_on_gds(
            str(path),
            top_cell=top_cell,
            channel_intents=declared_channels,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            final_placement_certificate=final_placement_certificate,
            placement_state_sha256=placement_state_sha256,
            tech=tech,
        )
        expected_inventory = tuple(
            declaration.sha256 for declaration in declared_channels
        )
        observed_inventory = tuple(
            item.declaration_sha256 for item in channel_obligations
        )
        if observed_inventory != expected_inventory:
            channel_obligations = unmeasurable_channels(
                "exact channel obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_densities and density_inventory_error is not None:
        density_obligations = unmeasurable_densities(
            density_inventory_error,
            gds_sha256=digest,
        )
    elif declared_densities:
        from clay.layout.completeness.gds_invariants import (
            verify_density_whitespace_on_gds,
        )

        density_obligations = verify_density_whitespace_on_gds(
            str(path),
            top_cell=top_cell,
            density_intents=declared_densities,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            final_placement_certificate=final_placement_certificate,
            placement_state_sha256=placement_state_sha256,
            tech=tech,
        )
        expected_inventory = tuple(
            declaration.sha256 for declaration in declared_densities
        )
        observed_inventory = tuple(
            item.declaration_sha256 for item in density_obligations
        )
        if observed_inventory != expected_inventory:
            density_obligations = unmeasurable_densities(
                "exact density obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_od_fills and od_fill_inventory_error is not None:
        functional_od_obligations = unmeasurable_functional_od_fills(
            od_fill_inventory_error,
            gds_sha256=digest,
        )
    elif declared_od_fills:
        from clay.layout.completeness.gds_invariants import (
            verify_functional_od_fill_on_gds,
        )

        functional_od_obligations = verify_functional_od_fill_on_gds(
            str(path),
            top_cell=top_cell,
            od_fill_intents=declared_od_fills,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            final_placement_certificate=final_placement_certificate,
            placement_state_sha256=placement_state_sha256,
            row_final_placement=row_final_placement,
            tech=tech,
        )
        expected_inventory = tuple(
            declaration.sha256 for declaration in declared_od_fills
        )
        observed_inventory = tuple(
            item.declaration_sha256 for item in functional_od_obligations
        )
        if observed_inventory != expected_inventory:
            functional_od_obligations = unmeasurable_functional_od_fills(
                "exact functional OD obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if declared_ratio_arrays:
        from clay.layout.completeness.gds_invariants import (
            verify_finger_array_orders_on_gds,
        )

        finger_array_obligations = verify_finger_array_orders_on_gds(
            str(path),
            top_cell=top_cell,
            array_intents=declared_arrays,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            row_program=row_program,
            row_materialization=row_materialization,
            row_final_placement_certificate=row_final_placement_certificate,
            row_final_placement=row_final_placement,
            tech=tech,
        )
        expected_inventory = tuple(
            declaration.sha256 for declaration in declared_ratio_arrays
        )
        observed_inventory = tuple(
            item.declaration_sha256 for item in finger_array_obligations
        )
        if observed_inventory != expected_inventory:
            finger_array_obligations = unmeasurable_finger_arrays(
                "exact finger-array obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
                declarations=declared_ratio_arrays,
            )
    if declared_native_arrays:
        if native_array_inventory_error:
            finger_array_obligations += unmeasurable_finger_arrays(
                native_array_inventory_error,
                gds_sha256=digest,
                declarations=declared_native_arrays,
            )
        else:
            finger_array_obligations += _verify_native_diffusion_finger_orders_on_gds(
                str(path),
                top_cell=top_cell,
                array_intents=declared_arrays,
                declared_native_arrays=declared_native_arrays,
                emitted_device_geometry=emitted_device_geometry,
                emitted_device_placement_um=emitted_device_placement_um,
                exact_artifact=exact_artifact,
                source_cell=(
                    emission_source_cell
                    if emission_source_cell is not None
                    else authoring_cell
                ),
                row_program=row_program,
                row_materialization=row_materialization,
                row_final_placement_certificate=row_final_placement_certificate,
                row_final_placement=row_final_placement,
                tech=tech,
            )
    if (
        declared_common_centroids or declared_interdigitations
    ) and centroid_inventory_error is not None:
        common_centroid_obligations = unmeasurable_common_centroids(
            centroid_inventory_error,
            gds_sha256=digest,
        )
        interdigitation_obligations = unmeasurable_interdigitations(
            centroid_inventory_error,
            gds_sha256=digest,
        )
    elif declared_common_centroids or declared_interdigitations:
        from clay.layout.completeness.gds_invariants import (
            verify_common_centroid_interdigitation_on_gds,
        )

        (
            common_centroid_obligations,
            interdigitation_obligations,
        ) = verify_common_centroid_interdigitation_on_gds(
            str(path),
            top_cell=top_cell,
            common_centroid_intents=declared_common_centroids,
            matching_intents=declared_matching_intents,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            row_program=row_program,
            row_materialization=row_materialization,
            row_final_placement_certificate=row_final_placement_certificate,
            row_final_placement=row_final_placement,
            tech=tech,
        )
        expected_centroids = tuple(
            declaration.sha256 for declaration in declared_common_centroids
        )
        observed_centroids = tuple(
            item.declaration_sha256 for item in common_centroid_obligations
        )
        if observed_centroids != expected_centroids:
            common_centroid_obligations = unmeasurable_common_centroids(
                "exact common-centroid obligation inventory mismatch: "
                f"expected={expected_centroids}, observed={observed_centroids}",
                gds_sha256=digest,
            )
        expected_interdigitations = tuple(
            declaration.sha256 for declaration in declared_interdigitations
        )
        observed_interdigitations = tuple(
            item.declaration_sha256 for item in interdigitation_obligations
        )
        if observed_interdigitations != expected_interdigitations:
            interdigitation_obligations = unmeasurable_interdigitations(
                "exact interdigitation obligation inventory mismatch: "
                f"expected={expected_interdigitations}, "
                f"observed={observed_interdigitations}",
                gds_sha256=digest,
            )
    if declared_passive_bank_partitions and passive_bank_inventory_error is not None:
        passive_bank_partition_obligations = (
            unmeasurable_passive_bank_partitions(
                passive_bank_inventory_error,
                gds_sha256=digest,
            )
        )
    elif declared_passive_bank_partitions:
        from clay.layout.completeness.gds_invariants import (
            verify_passive_bank_partitions_on_gds,
        )

        passive_bank_partition_obligations = (
            verify_passive_bank_partitions_on_gds(
                str(path),
                top_cell=top_cell,
                partition_intents=declared_passive_bank_partitions,
                emitted_device_geometry=emitted_device_geometry,
                emitted_device_placement_um=emitted_device_placement_um,
                exact_artifact=exact_artifact,
                source_cell=(
                    emission_source_cell
                    if emission_source_cell is not None
                    else authoring_cell
                ),
                tech=tech,
                rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
                rigid_nonmos_macro_inputs=rigid_nonmos_macro_inputs,
                rigid_nonmos_macro_interfaces=rigid_nonmos_macro_interfaces,
                rigid_nonmos_placement_certificate=(
                    rigid_nonmos_placement_certificate
                ),
                rigid_nonmos_scope_token=rigid_nonmos_scope_token,
                rigid_nonmos_emission_certificate=(
                    rigid_nonmos_emission_certificate
                ),
                pins_by_net=pins_by_net,
            )
        )
        expected_inventory = tuple(
            declaration.sha256
            for declaration in declared_passive_bank_partitions
        )
        observed_inventory = tuple(
            item.declaration_sha256
            for item in passive_bank_partition_obligations
        )
        if observed_inventory != expected_inventory:
            passive_bank_partition_obligations = (
                unmeasurable_passive_bank_partitions(
                    "exact passive-bank partition obligation inventory mismatch: "
                    f"expected={expected_inventory}, observed={observed_inventory}",
                    gds_sha256=digest,
                )
            )
    if declared_well_island_shares and well_island_inventory_error is not None:
        well_island_share_obligations = unmeasurable_well_island_shares(
            well_island_inventory_error,
            gds_sha256=digest,
        )
    elif declared_well_island_shares:
        from clay.layout.completeness.gds_invariants import (
            verify_well_island_sharing_on_gds,
        )

        well_island_share_obligations = verify_well_island_sharing_on_gds(
            str(path),
            top_cell=top_cell,
            share_intents=declared_well_island_shares,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=(
                emission_source_cell
                if emission_source_cell is not None
                else authoring_cell
            ),
            tech=tech,
        )
        expected_inventory = tuple(
            declaration.sha256 for declaration in declared_well_island_shares
        )
        observed_inventory = tuple(
            item.declaration_sha256 for item in well_island_share_obligations
        )
        if observed_inventory != expected_inventory:
            well_island_share_obligations = unmeasurable_well_island_shares(
                "exact well island share obligation inventory mismatch: "
                f"expected={expected_inventory}, observed={observed_inventory}",
                gds_sha256=digest,
            )
    if containment_declared and floorplan_bbox_nm is None:
        containment = _unmeasurable_containment(
            None, "route containment requires a measurable pre-route floorplan"
        )
    else:
        containment = None
    try:
        import klayout.db as kdb

        layout = kdb.Layout()
        layout.read(str(path))
        cell = layout.cell(top_cell)
        top_indexes = {item.cell_index() for item in layout.top_cells()}
        if cell is None or cell.cell_index() not in top_indexes:
            raise ValueError(f"named top cell {top_cell!r} is absent")
        bbox = cell.bbox()
        width_um = float(bbox.width()) * float(layout.dbu)
        height_um = float(bbox.height()) * float(layout.dbu)
        if not all(
            math.isfinite(value) and value > 0.0 for value in (width_um, height_um)
        ):
            raise ValueError("named top cell has empty or non-finite bbox")
        measured = width_um / height_um
        if containment_declared and floorplan_bbox_nm is not None:
            containment = _measure_route_containment(layout, cell, floorplan_bbox_nm)
    except Exception as exc:  # noqa: BLE001 - typed artifact boundary
        if containment_declared:
            containment = _unmeasurable_containment(
                floorplan_bbox_nm,
                f"cannot measure emitted GDS containment: {type(exc).__name__}: {exc}",
            )
        return _certificate(
            top_cell=top_cell,
            gds_sha256=digest,
            status="unmeasurable",
            measured_bbox_um=None,
            measured_aspect=None,
            aspect=obligation,
            reason=f"cannot measure emitted GDS aspect: {type(exc).__name__}: {exc}",
            route_containment=containment,
            physical_macro_graph_sha256=physical_macro_graph_sha256,
            candidate_sha256=candidate_sha256,
            placement_state_sha256=placement_state_sha256,
            guard_rings=ring_obligations,
            bands=band_obligations,
            power_planes=power_plane_obligations,
            kelvins=kelvin_obligations,
            shields=shield_obligations,
            adjacencies=adjacency_obligations,
            anchors=anchor_obligations,
            rail_affinities=rail_affinity_obligations,
            channels=channel_obligations,
            densities=density_obligations,
            functional_od_fills=functional_od_obligations,
            finger_array_orders=finger_array_obligations,
            passive_bank_partitions=passive_bank_partition_obligations,
            well_island_shares=well_island_share_obligations,
            common_centroids=common_centroid_obligations,
            interdigitations=interdigitation_obligations,
        )

    aspect_status: Literal["pass", "fail", "unmeasurable"] = "pass"
    aspect_reason = ""
    if obligation is not None:
        inside = obligation.minimum <= measured <= obligation.maximum
        aspect_status = "pass" if inside else "fail"
        aspect_reason = (
            f"emitted aspect {measured:.12g} is inside declared bounds "
            f"[{obligation.minimum:.12g}, {obligation.maximum:.12g}]"
            if inside
            else f"emitted aspect {measured:.12g} is outside declared bounds "
            f"[{obligation.minimum:.12g}, {obligation.maximum:.12g}]"
        )
    statuses = [aspect_status]
    reasons = [aspect_reason] if aspect_reason else []
    if containment is not None:
        statuses.append(containment.status)
        reasons.append(containment.reason)
    for ring in ring_obligations:
        statuses.append(
            "pass"
            if ring.status == "pass"
            else "fail"
            if ring.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            "guard_ring["
            + ",".join(ring.members)
            + f"] {ring.status}: {ring.reason}"
        )
    for band in band_obligations:
        statuses.append(
            "pass"
            if band.status == "pass"
            else "fail"
            if band.status == "fail"
            else "unmeasurable"
        )
        reasons.append(f"band {band.status}: {band.reason}")
    for plane in power_plane_obligations:
        statuses.append(
            "pass"
            if plane.status == "pass"
            else "fail"
            if plane.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"power_plane[{plane.net}:{plane.layer_key}] "
            f"{plane.status}: {plane.reason}"
        )
    for kelvin in kelvin_obligations:
        statuses.append(
            "pass"
            if kelvin.status == "pass"
            else "fail"
            if kelvin.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"kelvin[{kelvin.force_net},{kelvin.sense_net}->{kelvin.target_device}] "
            f"{kelvin.status}: {kelvin.reason}"
        )
    for shield in shield_obligations:
        statuses.append(
            "pass"
            if shield.status == "pass"
            else "fail"
            if shield.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"shield[{shield.aggressor_net}->{shield.victim_net}:"
            f"{shield.shield_net}] {shield.status}: {shield.reason}"
        )
    for adjacency in adjacency_obligations:
        statuses.append(
            "pass"
            if adjacency.status == "pass"
            else "fail"
            if adjacency.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"adjacent[{adjacency.source}->{adjacency.target}] "
            f"{adjacency.status}: {adjacency.reason}"
        )
    for anchor in anchor_obligations:
        statuses.append(
            "pass"
            if anchor.status == "pass"
            else "fail"
            if anchor.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"anchor[{anchor.target}->{anchor.edge}] "
            f"{anchor.status}: {anchor.reason}"
        )
    for rail_affinity in rail_affinity_obligations:
        statuses.append(
            "pass"
            if rail_affinity.status == "pass"
            else "fail"
            if rail_affinity.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"rail_affinity[{rail_affinity.group}->{rail_affinity.side}] "
            f"{rail_affinity.status}: {rail_affinity.reason}"
        )
    for channel in channel_obligations:
        statuses.append(
            "pass"
            if channel.status == "pass"
            else "fail"
            if channel.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"channel[{channel.orient}:{channel.declared_width_nm}nm] "
            f"{channel.status}: {channel.reason}"
        )
    for density in density_obligations:
        statuses.append(
            "pass"
            if density.status == "pass"
            else "fail"
            if density.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            "density["
            f"max_whitespace={density.max_whitespace_fraction:.12g}] "
            f"{density.status}: {density.reason}"
        )
    for functional_od in functional_od_obligations:
        statuses.append(
            "pass"
            if functional_od.status == "pass"
            else "fail"
            if functional_od.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            "functional_od_fill["
            f"min_fraction={functional_od.min_fraction:.12g}] "
            f"{functional_od.status}: {functional_od.reason}"
        )
    for array in finger_array_obligations:
        statuses.append(
            "pass"
            if array.status == "pass"
            else "fail"
            if array.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"finger_array[{array.group}:{array.order}] "
            f"{array.status}: {array.reason}"
        )
    for partition in passive_bank_partition_obligations:
        statuses.append(
            "pass"
            if partition.status == "pass"
            else "fail"
            if partition.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"passive_bank_partition[{partition.parent}:{partition.bank}] "
            f"{partition.status}: {partition.reason}"
        )
    for share in well_island_share_obligations:
        statuses.append(
            "pass"
            if share.status == "pass"
            else "fail"
            if share.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"well_island_share[{share.group}:{share.body}] "
            f"{share.status}: {share.reason}"
        )
    for centroid in common_centroid_obligations:
        statuses.append(
            "pass"
            if centroid.status == "pass"
            else "fail"
            if centroid.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            f"common_centroid[{centroid.group}] "
            f"{centroid.status}: {centroid.reason}"
        )
    for interdigitation in interdigitation_obligations:
        statuses.append(
            "pass"
            if interdigitation.status == "pass"
            else "fail"
            if interdigitation.status == "fail"
            else "unmeasurable"
        )
        reasons.append(
            "interdigitation["
            + ",".join(interdigitation.members)
            + f"] {interdigitation.status}: {interdigitation.reason}"
        )
    status: Literal["pass", "fail", "unmeasurable"]
    if "fail" in statuses:
        status = "fail"
    elif "unmeasurable" in statuses:
        status = "unmeasurable"
    else:
        status = "pass"
    return _certificate(
        top_cell=top_cell,
        gds_sha256=digest,
        status=status,
        measured_bbox_um=(width_um, height_um),
        measured_aspect=measured,
        aspect=obligation,
        reason="; ".join(reasons),
        route_containment=containment,
        physical_macro_graph_sha256=physical_macro_graph_sha256,
        candidate_sha256=candidate_sha256,
        placement_state_sha256=placement_state_sha256,
        guard_rings=ring_obligations,
        bands=band_obligations,
        power_planes=power_plane_obligations,
        kelvins=kelvin_obligations,
        shields=shield_obligations,
        adjacencies=adjacency_obligations,
        anchors=anchor_obligations,
        rail_affinities=rail_affinity_obligations,
        channels=channel_obligations,
        densities=density_obligations,
        functional_od_fills=functional_od_obligations,
        finger_array_orders=finger_array_obligations,
        passive_bank_partitions=passive_bank_partition_obligations,
        well_island_shares=well_island_share_obligations,
        common_centroids=common_centroid_obligations,
        interdigitations=interdigitation_obligations,
    )


def _physical_layout_intent_from_route_result(
    layout_intent: Any,
    route_result: Any,
) -> Any:
    """Independently replay the physical ring view carried by a routed result."""

    source_rings = getattr(layout_intent, "guard_rings", ())
    routed_rings = getattr(route_result, "guard_rings", ())
    plans = getattr(route_result, "shared_well_guard_envelope_plans", ())
    inventory = getattr(route_result, "shared_well_guard_ring_inventory", None)
    if type(source_rings) is not tuple or type(routed_rings) is not tuple:
        raise TypeError("final intent guard-ring inventories must be exact tuples")
    if inventory is None:
        if plans != () or routed_rings != source_rings:
            raise ValueError("routed guard rings differ without shared-well authority")
        return layout_intent
    from clay.layout.well_island_sharing import (
        SharedWellGuardRingInventory,
        compile_shared_well_guard_ring_inventory,
    )

    if type(plans) is not tuple:
        raise TypeError("routed shared guard envelope plans must be one exact tuple")
    if type(inventory) is not SharedWellGuardRingInventory:
        raise TypeError("routed shared guard inventory has a foreign type")
    replayed = compile_shared_well_guard_ring_inventory(plans, source_rings)
    if replayed != inventory:
        raise ValueError("routed shared guard inventory does not replay source intent")
    if routed_rings != replayed.effective_guard_rings:
        raise ValueError("routed effective guard rings do not replay shared-well authority")
    return replace(layout_intent, guard_rings=replayed.effective_guard_rings)


def _validated_exact_gds_sha256(
    *,
    gds_path: str | Path,
    exact_artifact: object,
) -> str:
    """Validate the emitter-issued artifact and the exact current path bytes."""

    from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

    if type(exact_artifact) is not ExactEmittedArtifact:
        raise TypeError("final GDS projection requires ExactEmittedArtifact")
    requested_path = Path(gds_path).resolve()
    artifact_path = Path(exact_artifact.exact_gds_path).resolve()
    if requested_path != artifact_path:
        raise ValueError(
            "final GDS path differs from the exact emitted artifact path"
        )
    exact_artifact.validate()
    return exact_artifact.scope.exact_gds_sha256


def _final_member_expected_artwork(
    *,
    stamp: object,
    position: object,
    transformed_layers: tuple,
) -> dict:
    """Use the emitter's raw polygon currency for exact multiset ownership."""

    from clay.layout.completeness.gds_invariants import _emitted_shape_witness

    return _emitted_shape_witness(
        stamp,
        position,
        excluded_layers=transformed_layers,
    )


def _readback_final_member_artwork(
    *,
    gds_path: str | Path,
    top_cell: str,
    exact_artifact: object,
    source_cell: object,
    final_placement_certificate: object,
    placement_state_sha256: str | None,
    geometry: Mapping[str, object],
    placement_um: Mapping[str, object],
    tech: object,
) -> _ArtifactMemberReadback:
    """Claim every source member's emitter owner from the reread exact GDS."""

    exact_gds_sha256 = _validated_exact_gds_sha256(
        gds_path=gds_path,
        exact_artifact=exact_artifact,
    )
    from clay.layout.completeness.gds_invariants import (
        _replay_exact_final_placement_artwork,
        _unique_emitter_owner,
    )
    from clay.layout.route.exact_artifact_evidence import (
        _canonical_source_member_inventory,
    )

    (
        _canonical_intent,
        _member_classes,
        transformed_layers,
        witness,
    ) = _replay_exact_final_placement_artwork(
        gds_path=str(gds_path),
        top_cell=top_cell,
        exact_artifact=exact_artifact,
        source_cell=source_cell,
        final_placement_certificate=final_placement_certificate,
        placement_state_sha256=placement_state_sha256,
        emitted_device_geometry=geometry,
        emitted_device_placement_um=placement_um,
        tech=tech,
    )
    source_members = tuple(
        row[0] for row in _canonical_source_member_inventory(source_cell)
    )
    owner_for_member = {
        member: _unique_emitter_owner(member, geometry)
        for member in source_members
    }
    if not owner_for_member:
        raise ValueError("final GDS member readback has no canonical source members")
    expected_by_owner: dict[str, dict[tuple[int, int], list]] = {}
    for owner in sorted(set(owner_for_member.values())):
        stamp = geometry.get(owner)
        position = placement_um.get(owner)
        if stamp is None or position is None:
            raise ValueError(
                f"final GDS member owner {owner!r} lacks emitter artwork"
            )
        expected = _final_member_expected_artwork(
            stamp=stamp,
            position=position,
            transformed_layers=transformed_layers,
        )
        if not expected:
            raise ValueError(
                f"final GDS member owner {owner!r} has no identity-preserving artwork"
            )
        expected_by_owner[owner] = expected
    route_owned_bboxes: dict[
        str, dict[tuple[int, int], list[tuple[int, int, int, int]]]
    ] = {}
    for net, key, bbox in exact_artifact.route_manifest.entries:
        route_owned_bboxes.setdefault(net, {}).setdefault(key, []).append(bbox)
    claims = witness.claim_polygon_multiset(
        expected_by_owner,
        eps=1,
        excluded_bboxes_by_owner={
            f"route:{net}": {
                key: tuple(sorted(bboxes))
                for key, bboxes in layers.items()
            }
            for net, layers in route_owned_bboxes.items()
        },
    )
    owner_digests: dict[str, str] = {}
    for owner, layers in sorted(claims.items()):
        payload = [
            [
                list(key),
                [
                    {
                        "bbox": list(record.bbox),
                        "contours": [
                            [list(point) for point in contour]
                            for contour in record.contours
                        ],
                    }
                    for record in records
                ],
            ]
            for key, records in sorted(layers.items())
        ]
        owner_digests[owner] = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
    return _ArtifactMemberReadback(
        gds_sha256=exact_gds_sha256,
        member_owner_shape_digests=tuple(
            sorted(
                (
                    member,
                    owner,
                    owner_digests[owner],
                )
                for member, owner in owner_for_member.items()
            )
        ),
    )


def check_route_result_final_layout_intent(
    gds_path: str | Path,
    *,
    top_cell: str,
    authoring_cell: object | None,
    emission_source_cell: object,
    effective_policy: object,
    layout_intent: object,
    route_result: object,
    exact_artifact: object,
    physical_macro_graph_sha256: str,
    placement_state_sha256: str | None,
    compact_candidate: object | None = None,
    intent_fact_set: IntentFactSet | None = None,
    source_inventory: SourceInventory | None = None,
) -> FinalLayoutIntentCertificate | None:
    """Replay the one canonical RouteCellResult-to-final-intent evidence seam."""

    if (intent_fact_set is None) != (source_inventory is None):
        raise TypeError("intent_fact_set and source_inventory must be provided together")
    row_source_projection = getattr(
        getattr(route_result, "row_materialization", None), "source_projection", None
    )
    if row_source_projection is not None:
        from clay.layout.place.row_source_projection import verify_row_source_fact_authority

        verify_row_source_fact_authority(
            row_source_projection, fact_set=intent_fact_set, source_inventory=source_inventory
        )
    certificate = check_final_layout_intent(
        gds_path,
        top_cell=top_cell,
        authoring_cell=authoring_cell,
        effective_policy=effective_policy,
        floorplan_bbox_nm=tuple(route_result.routing_envelope_nm) or None,
        require_route_containment=True,
        physical_macro_graph_sha256=physical_macro_graph_sha256,
        candidate_sha256=(
            compact_candidate.digest if compact_candidate is not None else None
        ),
        placement_state_sha256=placement_state_sha256,
        compact_candidate=compact_candidate,
        layout_intent=_physical_layout_intent_from_route_result(
            layout_intent, route_result
        ),
        tech=route_result.tech,
        emitted_device_geometry=route_result.geometry,
        emitted_device_placement_um=route_result.placement_um,
        exact_artifact=exact_artifact,
        emission_source_cell=emission_source_cell,
        guard_ring_member_classes=route_result.guard_ring_member_classes,
        member_ring_pad_top_nm=route_result.member_ring_pad_top_nm,
        member_ring_pad_bottom_nm=route_result.member_ring_pad_bottom_nm,
        final_placement_certificate=route_result.final_placement_certificate,
        row_program=route_result.row_program,
        row_materialization=route_result.row_materialization,
        row_final_placement_certificate=(
            route_result.row_final_placement_certificate
        ),
        row_final_placement=route_result.row_final_placement,
        rigid_nonmos_macro_plans=route_result.rigid_nonmos_macro_plans,
        rigid_nonmos_macro_inputs=route_result.rigid_nonmos_macro_inputs,
        rigid_nonmos_macro_interfaces=route_result.rigid_nonmos_macro_interfaces,
        rigid_nonmos_placement_certificate=(
            route_result.rigid_nonmos_placement_certificate
        ),
        rigid_nonmos_scope_token=route_result.physical_macro_scope_token,
        rigid_nonmos_emission_certificate=(
            route_result.rigid_nonmos_emission_certificate
        ),
        pins_by_net=route_result.pins_by_net,
        materialized_power_plane_facts=route_result.power_plane_facts,
        materialized_topology_plan=route_result.topology_plan,
    )
    if certificate is None:
        return None
    raw_group_routes = getattr(layout_intent, "route_groups", None)
    if type(raw_group_routes) is not tuple:
        raise TypeError("final group-route declaration inventory must be exact")
    group_route_ports: tuple[GdsGroupRoutePortObligation, ...] = ()
    completed = certificate
    if not raw_group_routes:
        if getattr(
            getattr(route_result, "routed", None),
            "group_route_landing_certificate",
            None,
        ) is not None:
            raise ValueError(
                "undeclared group-route evidence cannot enter final intent"
            )
    else:
        from clay.layout.route.group_ports import verify_group_route_ports_on_gds

        group_route_ports = verify_group_route_ports_on_gds(
            source_cell=emission_source_cell,
            intent=layout_intent,
            final_placement_certificate=route_result.final_placement_certificate,
            pins_by_net=route_result.pins_by_net,
            obstacles=route_result.obstacles,
            routed=route_result.routed,
            exact_artifact=exact_artifact,
            tech=route_result.tech,
            fact_set=intent_fact_set,
            source_inventory=source_inventory,
        )
        if not group_route_ports:
            if intent_fact_set is not None:
                raise ValueError(
                    "fact-governed group routes lack measured final ports"
                )
            return certificate
        statuses = (certificate.status,) + tuple(
            obligation.status for obligation in group_route_ports
        )
        status: Literal["pass", "fail", "unmeasurable"]
        if "fail" in statuses:
            status = "fail"
        elif "unmeasurable" in statuses:
            status = "unmeasurable"
        else:
            status = "pass"
        reasons = [certificate.reason] if certificate.reason else []
        reasons.extend(
            f"group_route_port[{row.group}:{row.net}] "
            f"{row.status}: {row.reason}"
            for row in group_route_ports
        )
        completed = replace(
            certificate,
            status=status,
            reason="; ".join(reasons),
            group_route_ports=group_route_ports,
        )
    if intent_fact_set is None or source_inventory is None:
        return completed
    required_consumers = {
        consumer
        for fact in intent_fact_set.facts
        for consumer in fact.required_consumers
    }
    projections: list[ConsumerProjection] = []
    if "topology" in required_consumers:
        from clay.layout.route.group_ports import build_topology_fact_projection

        topology_projection = build_topology_fact_projection(
            emission_source_cell,
            layout_intent,
            fact_set=intent_fact_set,
            source_inventory=source_inventory,
        )
        projections.append(topology_projection)
    if required_consumers.intersection({"placement_initial", "placement_replay"}):
        placement_projections = getattr(
            route_result.final_placement_certificate,
            "fact_projections",
            (),
        )
        if (
            type(placement_projections) is not tuple
            or len(placement_projections) != 2
            or any(
                type(row) is not ConsumerProjection
                for row in placement_projections
            )
            or tuple(row.consumer for row in placement_projections)
            != ("placement_initial", "placement_replay")
        ):
            raise ValueError("final fact closure lacks exact placement projections")
        projections.extend(
            row
            for row in placement_projections
            if row.consumer in required_consumers
        )
    if "routing" in required_consumers:
        landing_certificate = getattr(
            getattr(route_result, "routed", None),
            "group_route_landing_certificate",
            None,
        )
        routing_projection = getattr(landing_certificate, "fact_projection", None)
        if type(routing_projection) is not ConsumerProjection:
            raise ValueError("final fact closure lacks routing landing projection")
        projections.append(routing_projection)
    if "final_gds" in required_consumers:
        if (
            type(route_result.geometry) is not dict
            or type(route_result.placement_um) is not dict
        ):
            raise TypeError("final member readback requires exact emitter dictionaries")
        member_readback = _readback_final_member_artwork(
            gds_path=gds_path,
            top_cell=top_cell,
            exact_artifact=exact_artifact,
            source_cell=emission_source_cell,
            final_placement_certificate=route_result.final_placement_certificate,
            placement_state_sha256=placement_state_sha256,
            geometry=route_result.geometry,
            placement_um=route_result.placement_um,
            tech=route_result.tech,
        )
        if completed.route_containment is None:
            raise ValueError("final fact closure lacks exact route containment")
        projections.append(
            _build_final_gds_fact_projection(
                fact_set=intent_fact_set,
                source_inventory=source_inventory,
                place_groups=tuple(getattr(layout_intent, "place_groups", ())),
                route_groups=raw_group_routes,
                floorplan_column_groups=tuple(
                    getattr(
                        route_result.final_placement_certificate,
                        "floorplan_column_groups",
                        (),
                    )
                ),
                member_readback=member_readback,
                group_route_ports=group_route_ports,
                route_containment=completed.route_containment,
                well_island_shares=completed.well_island_shares,
                passive_bank_partitions=completed.passive_bank_partitions,
            )
        )
    return replace(
        completed,
        schema="clay.final-layout-intent-certificate/v19",
        fact_projections=tuple(projections),
    )


def _native_unmeasurable_obligations(
    declarations: Sequence[Any],
    *,
    gds_sha256: str,
    reason: str,
    program_sha256: str | None = None,
    materialization_sha256: str | None = None,
) -> tuple[GdsFingerArrayOrderObligation, ...]:
    """Build fail-closed unmeasurable obligations for native diffusion arrays."""
    return tuple(
        GdsFingerArrayOrderObligation(
            group=str(declaration.group),
            style=str(declaration.style),
            rows=int(declaration.rows),
            order=str(declaration.order),
            declaration_sha256=str(declaration.sha256),
            gds_sha256=gds_sha256,
            status="unmeasurable",
            measured_on_gds=False,
            owner_sequence=(),
            program_sha256=program_sha256,
            materialization_sha256=materialization_sha256,
            geometry_sha256=None,
            reason=reason,
        )
        for declaration in declarations
    )


def _replay_native_diffusion_authority(
    gds_path: str,
    *,
    top_cell: str,
    gds_sha256: str,
    array_intents: Sequence[Any],
    declared_native_arrays: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    row_program: Any,
    row_materialization: Any,
    row_final_placement_certificate: Any,
    row_final_placement: Any,
    tech: Any,
) -> dict[str, Any]:
    """Replay exact GDS, canonical source chains, and row compiler authority."""
    import klayout.db as kdb
    from clay.layout.completeness.gds_invariants import (
        _certified_assembly_transformed_layers,
    )
    from clay.layout.completeness.gds_witness import (
        ExactGdsWitness,
        functional_active_layer_keys,
    )
    from clay.layout.generic_placement import (
        Placement,
        RowFinalPlacementCertificate,
        verify_row_source_compiler_authority,
        verify_stored_row_projection,
    )
    from clay.layout.intent import layout_intent_from_cell_def, replay_array_intent
    from clay.layout.native_diffusion_source import declared_native_chains
    from clay.layout.place.rail_row_replay_scope import (
        require_rail_row_materialization,
    )
    from clay.layout.place.row_calculus import RowStructureKind
    from clay.layout.place.row_program import (
        RowProgram,
        RowProgramMaterialization,
    )
    from clay.layout.place.row_source_projection import require_row_source_projection
    from clay.layout.route.exact_artifact_evidence import (
        ExactEmittedArtifact,
        _replayed_source_cell_authority,
    )

    if type(exact_artifact) is not ExactEmittedArtifact:
        raise ValueError("native diffusion final verification requires ExactEmittedArtifact")
    exact_artifact.validate()
    scope = getattr(exact_artifact, "scope", None)
    if getattr(scope, "exact_gds_sha256", None) != gds_sha256:
        raise ValueError("exact artifact GDS SHA-256 differs from reread GDS")
    if getattr(scope, "top_cell", None) != top_cell:
        raise ValueError("exact artifact top cell differs from verified cell")
    if Path(exact_artifact.exact_gds_path).resolve() != Path(gds_path).resolve():
        raise ValueError("exact artifact GDS path differs from verified GDS path")
    if not isinstance(emitted_device_geometry, Mapping) or not isinstance(
        emitted_device_placement_um, Mapping
    ):
        raise ValueError("emitted device geometry and placement mappings are required")
    if (
        type(row_program) is not RowProgram
        or type(row_materialization) is not RowProgramMaterialization
        or type(row_final_placement_certificate) is not RowFinalPlacementCertificate
        or type(row_final_placement) is not Placement
    ):
        raise ValueError("retained row program, materialization, certificate, and placement are required")
    if tech is None:
        raise ValueError("resolved PDK layout tech is required")

    active_keys = tuple(functional_active_layer_keys(tech))
    poly_raw = tech.build_layer("poly")
    if not active_keys or poly_raw is None:
        raise ValueError("PDK tech lacks functional active or poly layer keys")
    poly_key = (int(poly_raw[0]), int(poly_raw[1]))

    source_sha256, inventory_sha256, _classes, canonical_source_cell = (
        _replayed_source_cell_authority(source_cell)
    )
    if canonical_source_cell is None:
        raise ValueError("canonical source cell authority is missing")
    scope_src = getattr(scope, "source_cell_source_sha256", None)
    if scope_src and scope_src != source_sha256:
        raise ValueError("exact artifact source SHA-256 differs from canonical source bytes")
    scope_inv = getattr(scope, "source_member_inventory_sha256", None)
    if scope_inv and scope_inv != inventory_sha256:
        raise ValueError("exact artifact member inventory differs from canonical source bytes")

    canonical_intent = layout_intent_from_cell_def(canonical_source_cell)
    canonical_native = tuple(
        replay_array_intent(item)
        for item in canonical_intent.arrays
        if item.style == "native_diffusion"
    )
    replayed_declared = tuple(replay_array_intent(item) for item in declared_native_arrays)
    if canonical_native != replayed_declared:
        raise ValueError("native diffusion declarations differ from canonical source")
    caller_native = tuple(
        replay_array_intent(item)
        for item in array_intents
        if getattr(item, "style", "") == "native_diffusion"
    )
    if caller_native != canonical_native:
        raise ValueError("caller array_intents differ from canonical native declarations")
    if len({item.group for item in canonical_native}) != len(canonical_native):
        raise ValueError("duplicate native diffusion group declaration")

    chains_by_group = dict(declared_native_chains(canonical_source_cell, canonical_intent))
    if set(chains_by_group) != {item.group for item in canonical_native}:
        raise ValueError("canonical native chains do not match declared native groups")

    if getattr(row_program, "tech_id", None) != getattr(tech, "pdk_id", None):
        raise ValueError("row program PDK differs from layout tech")
    require_rail_row_materialization(
        row_materialization, program=row_program, tech_id=row_program.tech_id
    )
    projection = getattr(row_materialization, "source_projection", None)
    if projection is None:
        raise ValueError("native diffusion row materialization lacks source projection")
    require_row_source_projection(projection, program=row_program)
    if (
        row_final_placement.row_program is None
        or row_final_placement.row_program.to_canonical_bytes()
        != row_program.to_canonical_bytes()
    ):
        raise ValueError("final placement row program differs from caller row program")
    if (
        row_final_placement.row_materialization is None
        or row_final_placement.row_materialization.certificate.to_canonical_bytes()
        != row_materialization.certificate.to_canonical_bytes()
    ):
        raise ValueError("final placement row materialization differs from caller")
    if not verify_stored_row_projection(row_final_placement):
        raise ValueError("stored row final placement projection verification failed")

    replayed_program, replayed_materialization = verify_row_source_compiler_authority(
        row_final_placement, canonical_source_cell, tech
    )
    if (
        replayed_program.to_canonical_bytes() != row_program.to_canonical_bytes()
        or replayed_materialization.certificate.to_canonical_bytes()
        != row_materialization.certificate.to_canonical_bytes()
    ):
        raise ValueError("replayed row compiler authority differs from retained row artifacts")
    row_final_placement_certificate.__post_init__()
    if row_final_placement_certificate.program_sha256 != row_program.sha256:
        raise ValueError("row final placement certificate binds a different row program")
    if (
        row_final_placement_certificate.materialization_sha256
        != row_materialization.certificate.sha256
    ):
        raise ValueError("row final placement certificate binds a different materialization")
    stored_row_cert = getattr(row_final_placement, "row_final_certificate", None)
    if (
        stored_row_cert is not None
        and stored_row_cert.to_canonical_bytes()
        != row_final_placement_certificate.to_canonical_bytes()
    ):
        raise ValueError("stored row final certificate differs from caller certificate")

    entries_by_group = {}
    for entry in row_program.entries:
        if entry.structure_kind is RowStructureKind.DISCRETE_MOSAIC:
            if entry.group in entries_by_group:
                raise ValueError(f"duplicate discrete mosaic entry for {entry.group!r}")
            entries_by_group[entry.group] = entry
    realizations_by_id = {
        row.entry_id: row for row in row_materialization.realizations
    }
    bindings_by_id = {
        row.entry_id: row for row in row_final_placement_certificate.macro_bindings
    }
    transformed_layers, witness = _certified_assembly_transformed_layers(
        exact_artifact,
        gds_path,
        emitted_device_geometry,
        emitted_device_placement_um,
        return_witness=True,
        top_cell=top_cell,
    )
    if type(witness) is not ExactGdsWitness:
        raise ValueError("exact GDS witness is unavailable")

    gds_od = kdb.Region()
    for key in active_keys:
        gds_od += witness.merged_region(key)
    gds_od.merge()
    gds_poly = witness.merged_region(poly_key)
    gds_channels_all = (gds_od & gds_poly)
    gds_channels_all.merge()

    return {
        "active_keys": active_keys,
        "poly_key": poly_key,
        "chains_by_group": chains_by_group,
        "entries_by_group": entries_by_group,
        "realizations_by_id": realizations_by_id,
        "bindings_by_id": bindings_by_id,
        "mosaics_by_id": {m.mosaic_id: m for m in projection.discrete_mosaics},
        "transformed_layers": set(transformed_layers),
        "gds_od": gds_od,
        "gds_poly": gds_poly,
        "gds_channels_all": gds_channels_all,
        "program_sha256": row_program.sha256,
        "materialization_sha256": row_materialization.certificate.sha256,
    }


def _translated_polys_from_mosaic(mosaic: Any, dx_nm: int, dy_nm: int) -> list[Any]:
    from clay.layout.backend import LayerPolygon, Point, Polygon

    return [
        LayerPolygon(
            int(poly.layer),
            int(poly.datatype),
            Polygon([Point(int(v.x) + dx_nm, int(v.y) + dy_nm) for v in poly.polygon.vertices]),
        )
        for poly in mosaic.polygons
    ]


def _translated_polys_from_stamp(stamp: Any, position_um: Any) -> list[Any]:
    from clay.layout.backend import LayerPolygon, Point, Polygon
    from clay.layout.placed_geometry import placed_point_nm

    if type(position_um) not in (tuple, list) or len(position_um) != 2:
        raise ValueError("emitted stamp placement must be an exact coordinate pair")
    polys: list[Any] = []
    layers = getattr(stamp, "polys_by_layer", None)
    if not isinstance(layers, Mapping):
        raise ValueError("emitted stamp lacks polys_by_layer")
    for raw_key, rings in sorted(layers.items()):
        layer, dtype = int(raw_key[0]), int(raw_key[1])
        for ring in rings:
            vertices = [
                Point(*placed_point_nm(float(x), float(y), tuple(position_um)))
                for x, y in ring
            ]
            polys.append(LayerPolygon(layer, dtype, Polygon(vertices)))
    return polys


def _layer_regions_from_polys(
    polys: Sequence[Any],
    keys: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], Any], Any]:
    import klayout.db as kdb

    by_layer = {key: kdb.Region() for key in keys}
    for poly in polys:
        key = (int(poly.layer), int(poly.datatype))
        if key not in by_layer:
            continue
        pts = [kdb.Point(int(v.x), int(v.y)) for v in poly.polygon.vertices]
        by_layer[key].insert(kdb.Polygon(pts))
    for region in by_layer.values():
        region.merge()
    combined = kdb.Region()
    for region in by_layer.values():
        combined += region
    combined.merge()
    return by_layer, combined


def _measure_native_diffusion_row_on_gds(
    declaration: Any,
    *,
    gds_sha256: str,
    authority: Mapping[str, Any],
    emitted_device_geometry: Mapping[str, Any],
    emitted_device_placement_um: Mapping[str, Any],
    row_materialization_sha256: str,
    claimed_od: Any,
    claimed_channels: Any,
) -> tuple[GdsFingerArrayOrderObligation, Any, Any]:
    """Measure one native diffusion row's real GDS channels against canonical source."""
    import klayout.db as kdb
    from clay.layout.completeness.gds_invariants import (
        _emitted_shape_witness,
        _merged_shape_witness,
        _shape_geometry,
    )
    from clay.layout.generic_placement import _placed_artwork_sha256
    from clay.layout.mosaic_geometry import MosaicGeometry
    from clay.layout.occurrence_channels import (
        RowChannelInventory,
        replay_row_channel_inventory,
    )

    group = str(declaration.group)
    prog_sha = authority["program_sha256"]
    mat_sha = authority["materialization_sha256"]

    def make_result(
        status: str,
        reason: str,
        *,
        measured_on_gds: bool,
        owner_sequence: tuple[str, ...] = (),
        geometry_sha256: str | None = None,
    ) -> GdsFingerArrayOrderObligation:
        return GdsFingerArrayOrderObligation(
            group=group,
            style=str(declaration.style),
            rows=int(declaration.rows),
            order=str(declaration.order),
            declaration_sha256=str(declaration.sha256),
            gds_sha256=gds_sha256,
            status=status,
            measured_on_gds=measured_on_gds,
            owner_sequence=owner_sequence,
            program_sha256=prog_sha,
            materialization_sha256=mat_sha,
            geometry_sha256=geometry_sha256,
            reason=reason,
        )

    chain = authority["chains_by_group"].get(group)
    entry = authority["entries_by_group"].get(group)
    if chain is None or entry is None:
        return make_result("unmeasurable", f"missing native diffusion row entry for {group!r}", measured_on_gds=False), kdb.Region(), kdb.Region()
    realization = authority["realizations_by_id"].get(entry.entry_id)
    binding = authority["bindings_by_id"].get(entry.entry_id)
    if realization is None or binding is None:
        return make_result("unmeasurable", f"missing materialization or placement binding for {group!r}", measured_on_gds=False), kdb.Region(), kdb.Region()

    geom_sha = getattr(realization, "geometry_sha256", None)
    mosaic = realization.geometry
    expected_pose = declaration.transforms[0] if declaration.transforms else "R0"
    if (
        entry.rows != 1
        or entry.order != "row_major"
        or entry.orientation != "R0"
        or expected_pose not in ("R0", "MR0")
    ):
        return make_result("unmeasurable", f"native diffusion entry declaration mismatch for {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
    if (
        realization.owner != entry.owner
        or binding.owner != entry.owner
        or binding.entry_sha256 != entry.sha256
        or binding.materialization_sha256 != row_materialization_sha256
        or tuple(binding.consumed_devices) != tuple(entry.consumed_devices)
    ):
        return make_result("unmeasurable", f"native diffusion realization/binding authority mismatch for {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
    if type(mosaic) is not MosaicGeometry or type(mosaic.channel_inventory) is not RowChannelInventory:
        return make_result("unmeasurable", f"native diffusion realization lacks MosaicGeometry channel inventory for {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    try:
        inventory = replay_row_channel_inventory(
            geometry=mosaic, consumed_devices=entry.consumed_devices
        )
    except Exception as exc:
        return make_result("unmeasurable", f"native diffusion channel inventory replay failed for {group!r}: {exc}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
    if inventory != mosaic.channel_inventory or inventory.sha256 != mosaic.channel_inventory.sha256:
        return make_result("unmeasurable", f"native diffusion channel inventory differs from geometry replay for {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    active_keys = set(authority["active_keys"])
    poly_key = authority["poly_key"]
    if inventory.poly_layer_key != poly_key or not inventory.od_layer_keys or not set(inventory.od_layer_keys) <= active_keys:
        return make_result("unmeasurable", f"native diffusion channel layer inventory disagrees with PDK for {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    lx0, ly0, lx1, ly1 = realization.local_bbox_nm
    bx0, by0, bx1, by1 = binding.bbox_nm
    dx_nm, dy_nm = bx0 - lx0, by0 - ly0
    if (bx1 - lx1, by1 - ly1) != (dx_nm, dy_nm):
        return make_result("unmeasurable", f"native diffusion binding is not a rigid translation for {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    chain_owners = tuple(f.owner for f in chain.fingers)
    if set(entry.consumed_devices) != set(chain_owners) or set(realization.consumed_devices) != set(chain_owners):
        return make_result("unmeasurable", f"native diffusion consumed source members differ from canonical chain for {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    if group not in emitted_device_geometry or group not in emitted_device_placement_um:
        return make_result("unmeasurable", f"emitted stamp missing for native diffusion group {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
    stamp = emitted_device_geometry[group]
    pos = emitted_device_placement_um[group]
    stamp_members = getattr(stamp, "composite_members", None)
    if stamp_members is not None and tuple(sorted(stamp_members)) != tuple(entry.consumed_devices):
        return make_result("unmeasurable", f"emitted stamp members differ from row program for {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    # Independently build the row's owned active region from replayed channels & mosaic polygons.
    owned_od = kdb.Region()
    for occ in inventory.occurrences:
        for key, polygons in occ.owned_mask_polygons:
            if key in inventory.od_layer_keys:
                for poly in polygons:
                    owned_od += poly.translated(dx_nm, dy_nm).to_region()
    owned_od.merge()
    mosaic_placed_polys = _translated_polys_from_mosaic(mosaic, dx_nm, dy_nm)
    mosaic_mask_regions, mosaic_od = _layer_regions_from_polys(
        mosaic_placed_polys, set(inventory.od_layer_keys)
    )
    _mosaic_poly_map, mosaic_poly_region = _layer_regions_from_polys(
        mosaic_placed_polys, {poly_key}
    )
    if owned_od.is_empty() or not (owned_od ^ mosaic_od).is_empty():
        return make_result("unmeasurable", f"owned active diffusion region disagrees with row geometry for {group!r}", measured_on_gds=False, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    gds_od = authority["gds_od"]
    gds_poly = authority["gds_poly"]
    gds_channels_all = authority["gds_channels_all"]

    if not (claimed_od & owned_od).is_empty():
        return make_result("fail", f"native diffusion row {group!r} overlaps claimed active diffusion of another row", measured_on_gds=True, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
    if not (owned_od - gds_od).is_empty() or not (mosaic_poly_region - gds_poly).is_empty():
        return make_result("fail", f"native diffusion group {group!r} mask shapes are missing from reread GDS", measured_on_gds=True, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    actual_channels = tuple(
        kdb.Region(poly)
        for poly in gds_channels_all.each()
        if not (kdb.Region(poly) & owned_od).is_empty()
    )
    od_bbox_region = kdb.Region(owned_od.bbox())
    actual_in_bbox = tuple(
        kdb.Region(poly)
        for poly in gds_channels_all.each()
        if not (kdb.Region(poly) & od_bbox_region).is_empty()
    )

    if len(inventory.occurrences) != len(mosaic.fingers) or len(inventory.occurrences) != len(chain.fingers):
        return make_result("fail", f"native diffusion finger cardinality disagrees for {group!r}", measured_on_gds=True, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    used: set[int] = set()
    matched: list[tuple[int, int, str, int, str, Any]] = []
    matched_channel_union = kdb.Region()
    for occ, finger in zip(inventory.occurrences, mosaic.fingers):
        if occ.logical_owner != finger.member or not occ.channels:
            return make_result("fail", f"native diffusion occurrence/finger mismatch for {group!r}", measured_on_gds=True, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
        expected_occ_region = kdb.Region()
        for ch in occ.channels:
            expected_occ_region += ch.translated(dx_nm, dy_nm).to_region()
        expected_occ_region.merge()
        hits = [idx for idx, reg in enumerate(actual_channels) if (reg ^ expected_occ_region).is_empty()]
        if len(hits) != 1 or hits[0] in used:
            return make_result("fail", f"exact GDS does not uniquely realize finger {occ.logical_owner}[{occ.occurrence_index}] in {group!r}", measured_on_gds=True, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
        idx = hits[0]
        used.add(idx)
        reg = actual_channels[idx]
        if not (claimed_channels & reg).is_empty():
            return make_result("fail", f"native diffusion channel overlaps already claimed row channel in {group!r}", measured_on_gds=True, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
        matched_channel_union += reg
        box = reg.bbox()
        if int(box.width()) != int(chain.length_nm) or int(box.height()) != int(chain.width_nm):
            return make_result(
                "fail",
                f"GDS channel dimensions {(box.width(), box.height())} differ from source {(chain.length_nm, chain.width_nm)} in {group!r}",
                measured_on_gds=True,
                geometry_sha256=geom_sha,
            ), kdb.Region(), kdb.Region()
        if int(finger.x0) + dx_nm != int(box.left) or int(finger.x1) + dx_nm != int(box.right):
            return make_result("fail", f"GDS channel X bounds differ from finger geometry in {group!r}", measured_on_gds=True, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
        matched.append((int(box.left + box.right), int(box.bottom + box.top), occ.logical_owner, occ.occurrence_index, finger.gate_net, box))

    if len(used) != len(actual_channels) or len(actual_in_bbox) != len(actual_channels):
        return make_result("fail", f"reread GDS contains extra or missing channels in native diffusion row {group!r}", measured_on_gds=True, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    matched_sorted = sorted(matched, key=lambda item: (item[0], item[1]))
    for prev, cur in zip(matched_sorted, matched_sorted[1:]):
        if int(prev[5].right) >= int(cur[5].left) or prev[0] >= cur[0]:
            return make_result("fail", f"GDS channels do not form a strictly ordered horizontal chain in {group!r}", measured_on_gds=True, owner_sequence=tuple(item[2] for item in matched_sorted), geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    observed_owners = tuple(item[2] for item in matched_sorted)
    observed_gates = tuple(item[4] for item in matched_sorted)
    observed_row_indices = tuple(item[3] for item in matched_sorted)
    owner_counts: dict[str, int] = {}
    observed_occurrences = []
    for owner in observed_owners:
        ordinal = owner_counts.get(owner, 0)
        owner_counts[owner] = ordinal + 1
        observed_occurrences.append((owner, ordinal))
    observed_occurrences_tuple = tuple(observed_occurrences)
    expected_occurrences = tuple((f.owner, f.occurrence) for f in chain.fingers)
    expected_owners = tuple(f.owner for f in chain.fingers)
    expected_gates = tuple(f.gate for f in chain.fingers)
    if (
        observed_row_indices != tuple(range(len(chain.fingers)))
        or observed_occurrences_tuple != expected_occurrences
        or observed_owners != expected_owners
        or observed_gates != expected_gates
    ):
        return make_result(
            "fail",
            f"measured GDS channel order {observed_occurrences_tuple} differs from canonical source {expected_occurrences} for {group!r}",
            measured_on_gds=True,
            owner_sequence=observed_owners,
            geometry_sha256=geom_sha,
        ), kdb.Region(), kdb.Region()

    # Also verify emitter artwork & rigid binding consistency without relaxing GDS proof.
    try:
        stamp_polys = _translated_polys_from_stamp(stamp, pos)
        if _placed_artwork_sha256(mosaic_placed_polys) != binding.artwork_sha256:
            return make_result("fail", f"mosaic artwork SHA-256 differs from final binding for {group!r}", measured_on_gds=True, owner_sequence=observed_owners, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
        if _placed_artwork_sha256(stamp_polys) != binding.artwork_sha256:
            return make_result("fail", f"emitted stamp artwork SHA-256 differs from final binding for {group!r}", measured_on_gds=True, owner_sequence=observed_owners, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
        excluded = authority["transformed_layers"]
        emitted_shapes = _emitted_shape_witness(stamp, pos, excluded_layers=tuple(excluded))
        mosaic_shapes: dict[tuple[int, int], list[Any]] = {}
        for lp in mosaic_placed_polys:
            key = (int(lp.layer), int(lp.datatype))
            if key in excluded:
                continue
            geom = _shape_geometry(tuple((int(v.x), int(v.y)) for v in lp.polygon.vertices))
            if geom is None:
                raise ValueError("degenerate mosaic polygon")
            mosaic_shapes.setdefault(key, []).append(geom)
        for shapes in mosaic_shapes.values():
            shapes.sort()
        if _merged_shape_witness(emitted_shapes) != _merged_shape_witness(mosaic_shapes):
            return make_result("fail", f"emitted stamp polygons differ from canonical mosaic for {group!r}", measured_on_gds=True, owner_sequence=observed_owners, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()
    except Exception as exc:
        return make_result("fail", f"emitted stamp artwork replay failed for {group!r}: {exc}", measured_on_gds=True, owner_sequence=observed_owners, geometry_sha256=geom_sha), kdb.Region(), kdb.Region()

    return (
        make_result(
            "pass",
            "measured native diffusion GDS channel order matches canonical source",
            measured_on_gds=True,
            owner_sequence=observed_owners,
            geometry_sha256=geom_sha,
        ),
        owned_od,
        matched_channel_union,
    )


def _verify_native_diffusion_finger_orders_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    array_intents: Sequence[Any],
    declared_native_arrays: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    row_program: Any,
    row_materialization: Any,
    row_final_placement_certificate: Any,
    row_final_placement: Any,
    tech: Any,
) -> tuple[GdsFingerArrayOrderObligation, ...]:
    """Independently verify physical poly/diffusion GDS channel order for native_diffusion."""
    import klayout.db as kdb

    declarations = tuple(declared_native_arrays)
    if not declarations:
        return ()
    path = Path(gds_path)
    if not path.is_file():
        return ()
    gds_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    prog_sha = getattr(row_program, "sha256", None) if row_program is not None else None
    cert = getattr(row_materialization, "certificate", None)
    mat_sha = getattr(cert, "sha256", None) if cert is not None else None
    try:
        authority = _replay_native_diffusion_authority(
            str(path),
            top_cell=top_cell,
            gds_sha256=gds_sha256,
            array_intents=array_intents,
            declared_native_arrays=declarations,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            exact_artifact=exact_artifact,
            source_cell=source_cell,
            row_program=row_program,
            row_materialization=row_materialization,
            row_final_placement_certificate=row_final_placement_certificate,
            row_final_placement=row_final_placement,
            tech=tech,
        )
    except Exception as exc:
        return _native_unmeasurable_obligations(
            declarations,
            gds_sha256=gds_sha256,
            reason=f"native diffusion final GDS authority replay failed: {exc}",
            program_sha256=prog_sha if isinstance(prog_sha, str) else None,
            materialization_sha256=mat_sha if isinstance(mat_sha, str) else None,
        )

    claimed_od = kdb.Region()
    claimed_channels = kdb.Region()
    obligations: list[GdsFingerArrayOrderObligation] = []
    for declaration in declarations:
        obligation, row_od, row_channels = _measure_native_diffusion_row_on_gds(
            declaration,
            gds_sha256=gds_sha256,
            authority=authority,
            emitted_device_geometry=emitted_device_geometry,
            emitted_device_placement_um=emitted_device_placement_um,
            row_materialization_sha256=authority["materialization_sha256"],
            claimed_od=claimed_od,
            claimed_channels=claimed_channels,
        )
        if not row_od.is_empty():
            claimed_od += row_od
            claimed_od.merge()
        if not row_channels.is_empty():
            claimed_channels += row_channels
            claimed_channels.merge()
        obligations.append(obligation)
    return tuple(obligations)


__all__ = [
    "FinalAspectObligation",
    "FinalLayoutIntentCertificate",
    "GdsAnchorObligation",
    "GdsBandObligation",
    "GdsChannelObligation",
    "GdsCommonCentroidObligation",
    "GdsDensityWhitespaceObligation",
    "GdsFingerArrayOrderObligation",
    "GdsFunctionalOdFillObligation",
    "GdsGuardRingObligation",
    "GdsInterdigitationObligation",
    "GdsKelvinObligation",
    "GdsPassiveBankPartitionObligation",
    "GdsRailAffinityObligation",
    "GdsWellIslandShareObligation",
    "RouteContainmentObligation",
    "check_final_layout_intent",
    "check_route_result_final_layout_intent",
    "declared_final_aspect_obligation",
]
